# Grid nesting: N abutting patches of 2:1-refined resolution with per-patch
# timesteps, coupled per DeMaria et al. (1992, MWR) / Ooyama (2001, JAS):
#
# - Coarse→fine: the fine (child) patch carries a rank-3 inhomogeneous (R3X)
#   BC at each interface; its border trio comes from the parent's amplitudes
#   at the interface node, linearly interpolated in time across the parent's
#   step while the child subcycles.
# - Fine→coarse: NOT a BC. Each parent patch extends exactly one parent cell
#   past the nominal interface into the child's domain (the "collar"); the
#   parent's physical values and derivative slices at the collar mish points
#   are overwritten from the child's representation before every parent
#   tendency step, so the parent's Galerkin loads there integrate child data.
#
# Terminology: Scythe "patch" = one full grid, "tile" = a worker slice of a
# patch. Nest members are called "nest patches" and are indexed 1..N from the
# left; parent = the coarser side of a junction, child = the finer side.

using SparseArrays

export NestedModelParameters, build_nest, NestTopology, NestInterface
export integrate_nested_model

"""
    NestedModelParameters

Configuration for an N-patch nested run. The nest is a 1-D horizontal chain
of abutting patches sharing the vertical grid, variables, physics, and outer
BCs of `base`; each junction must have an exact 2:1 cell-width ratio (parent
= coarser side).

# Fields
- `boundaries::Vector{Float64}` — N+1 nominal patch edges, strictly increasing.
  Patch `i` nominally spans `[boundaries[i], boundaries[i+1]]`; parents are
  extended one parent cell past each junction internally (the collar).
- `num_cells::Vector{Int}` — nominal cell count per patch (collar cells are
  added internally).
- `ts::Vector{Float64}` — requested timestep per patch. Parentless (coarsest)
  patches keep their requested ts (which must agree); every child runs
  `n_sub = ceil(ts_parent/ts_requested)` substeps of exactly `ts_parent/n_sub`
  per parent step, so a float ratio like 1.8 is honored by rounding the
  child's ts *down* to the nearest integer division.
- `workers_per_patch::Vector{Int}` — distributed worker group sizes
  (used by `integrate_nested_model`).
- `base::ModelParameters` — template: equation set, physics, options,
  integration/output times, vertical axis, vars, outer BCs, output dir.
  `base.grid_params.iMin/iMax/num_cells` are ignored.
"""
Base.@kwdef struct NestedModelParameters
    boundaries::Vector{Float64}
    num_cells::Vector{Int}
    ts::Vector{Float64}
    workers_per_patch::Vector{Int}
    base::ModelParameters
end

"""
    NestInterface

One directed parent→child junction of a nest, carrying everything both ends
need: the Springsteel interface metadata (for the R3X trio payload) and the
parent-side collar description (for the fine→coarse injection).
"""
struct NestInterface
    parent::Int
    child::Int
    parent_side::Symbol          # side of the PARENT facing the child
    child_side::Symbol
    interface_x::Float64         # nominal junction position
    meta::PatchInterfaceMetadata
    collar_rows::Vector{Int}     # parent `physical` rows inside the collar
    collar_x::Vector{Float64}    # unique i-coordinates of the collar columns
    nslices::Int                 # derivative slices carried by `physical`
    # RL (ragged-ring) collars: explicit (r, λ) evaluation points, one per
    # collar physical row, plus the target ring's supported max wavenumber
    # (empty for tensor-product geometries, which use collar_x).
    collar_pts::Matrix{Float64}
    collar_kmax::Vector{Int}
end

"""
    NestTopology

Derived structure of a nest: directed interfaces plus per-patch scheduling.
`n_sub[i]` is the number of steps patch `i` takes per step of its parent(s)
(1 for parentless patches); `ts_actual[i]` is the constant per-patch
timestep after integer-subdivision rounding.
"""
struct NestTopology
    interfaces::Vector{NestInterface}
    parent_ifaces::Vector{Vector{Int}}   # per patch: interface idxs where it is the child
    child_ifaces::Vector{Vector{Int}}    # per patch: interface idxs where it is the parent
    n_sub::Vector{Int}
    ts_actual::Vector{Float64}
end

_nest_dx(nest::NestedModelParameters, i::Int) =
    (nest.boundaries[i + 1] - nest.boundaries[i]) / nest.num_cells[i]

function _physical_nslices(geometry::String)
    if geometry == "R"
        return 3                       # value, ∂i, ∂²i
    elseif geometry == "RiRk"
        return 5                       # value, ∂i, ∂²i, ∂k, ∂²k
    elseif geometry == "RL"
        return 5                       # value, ∂r, ∂²r, ∂λ, ∂²λ
    else
        throw(ArgumentError(
            "Nesting currently supports geometries \"R\", \"RiRk\", and \"RL\", got \"$geometry\""))
    end
end

"""
    build_nest(nest::NestedModelParameters) -> (models, topo)

Validate the nest configuration and construct the per-patch
`ModelParameters` (with collar-extended grids, interface BCs, derived
timesteps, and per-patch output directories `nest1/ … nestN/`) plus the
`NestTopology` used by the integration drivers.
"""
function build_nest(nest::NestedModelParameters)
    n = length(nest.num_cells)
    length(nest.boundaries) == n + 1 || throw(ArgumentError(
        "boundaries must have length $(n + 1) (one more than num_cells), got $(length(nest.boundaries))"))
    length(nest.ts) == n || throw(ArgumentError("ts must have length $n"))
    length(nest.workers_per_patch) == n || throw(ArgumentError(
        "workers_per_patch must have length $n"))
    all(diff(nest.boundaries) .> 0) || throw(ArgumentError(
        "boundaries must be strictly increasing"))
    all(nest.num_cells .> 0) || throw(ArgumentError("num_cells must be positive"))
    all(nest.ts .> 0) || throw(ArgumentError("ts must be positive"))

    base = nest.base
    geometry = base.grid_params.geometry
    nslices = _physical_nslices(geometry)

    dx = [_nest_dx(nest, i) for i in 1:n]

    # ── Junction parent/child determination (exact 2:1 or 1:1) ──────────────
    # 1:1 junctions (same-resolution domain decomposition through the nesting
    # machinery, identity coupling matrix) take the LEFT patch as parent by
    # convention — either choice is valid.
    junction_parent = zeros(Int, n - 1)   # patch index of the parent at junction j
    for j in 1:(n - 1)
        ratio = dx[j] / dx[j + 1]
        if isapprox(ratio, 2.0; rtol=1e-10) || isapprox(ratio, 1.0; rtol=1e-10)
            junction_parent[j] = j            # left patch coarser (or equal)
        elseif isapprox(ratio, 0.5; rtol=1e-10)
            junction_parent[j] = j + 1        # right patch is coarser
        else
            throw(ArgumentError(
                "Junction $j at x=$(nest.boundaries[j + 1]): cell-width ratio " *
                "$(ratio):1 — nesting requires exactly 2:1 or 1:1 (parent the coarser side)"))
        end
    end

    # Collar feasibility: every collar (one parent cell = two child cells)
    # must fit inside the child, and a child's two collars must not overlap.
    for i in 1:n
        collar_in = 0.0
        j_left = i - 1                       # junction on the patch's left
        j_right = i                          # junction on the patch's right
        if j_left >= 1 && junction_parent[j_left] != i
            collar_in += dx[junction_parent[j_left]]
        end
        if j_right <= n - 1 && junction_parent[j_right] != i
            collar_in += dx[junction_parent[j_right]]
        end
        width = nest.boundaries[i + 1] - nest.boundaries[i]
        collar_in < width - 1e-12 || throw(ArgumentError(
            "Patch $i (width $width) is too narrow for its parents' collars " *
            "($collar_in) — widen the patch or reduce refinement depth"))
    end

    # ── Per-patch timestep derivation ────────────────────────────────────────
    parent_ids = [Int[] for _ in 1:n]
    for j in 1:(n - 1)
        p = junction_parent[j]
        c = (p == j) ? j + 1 : j
        push!(parent_ids[c], p)
    end
    ts_actual = fill(NaN, n)
    n_sub = ones(Int, n)
    # Parentless (coarsest) patches: requested ts, must agree
    roots = [i for i in 1:n if isempty(parent_ids[i])]
    isempty(roots) && throw(ArgumentError("Nest has no coarsest (parentless) patch"))
    ts_root = nest.ts[roots[1]]
    for i in roots
        isapprox(nest.ts[i], ts_root; rtol=1e-12) || throw(ArgumentError(
            "All parentless (coarsest) patches must share one ts: patch $(roots[1]) " *
            "has $(ts_root), patch $i has $(nest.ts[i])"))
        ts_actual[i] = nest.ts[i]
    end
    # Children in refinement order until resolved
    for _ in 1:n
        for i in 1:n
            isnan(ts_actual[i]) || continue
            all(p -> !isnan(ts_actual[p]), parent_ids[i]) || continue
            tps = unique(round.(ts_actual[parent_ids[i]], sigdigits=14))
            length(tps) == 1 || throw(ArgumentError(
                "Patch $i has parents with different timesteps $(ts_actual[parent_ids[i]]) — " *
                "both parents of a child must run in lockstep"))
            tp = ts_actual[parent_ids[i][1]]
            n_sub[i] = ceil(Int, tp / nest.ts[i] - 1e-12)
            ts_actual[i] = tp / n_sub[i]
        end
    end
    any(isnan, ts_actual) && throw(ArgumentError(
        "Could not resolve per-patch timesteps — is the refinement graph a valid chain?"))

    for i in 1:n
        ratio = base.output_interval / ts_actual[i]
        if !isapprox(ratio, round(ratio); atol=1e-8)
            @warn "output_interval $(base.output_interval) is not an integer multiple of " *
                  "patch $i's actual ts $(ts_actual[i]) — snapshot times will not align"
        end
    end

    # ── Per-patch grid parameters and ModelParameters ────────────────────────
    all_vars = collect(keys(base.grid_params.vars))
    fixed_bc() = Dict{String,Any}(v => FixedBC() for v in all_vars)
    natural_bc() = Dict{String,Any}(v => NaturalBC() for v in all_vars)

    models = Vector{ModelParameters}(undef, n)
    gps = Vector{SpringsteelGridParameters}(undef, n)
    for i in 1:n
        iMin = nest.boundaries[i]
        iMax = nest.boundaries[i + 1]
        cells = nest.num_cells[i]
        # Collar extensions where this patch is the parent
        bcl = (i == 1) ? base.grid_params.BCL : nothing
        bcr = (i == n) ? base.grid_params.BCR : nothing
        if i >= 2   # left junction j = i-1
            if junction_parent[i - 1] == i
                iMin -= dx[i]; cells += 1      # parent: collar into the left child
                bcl = natural_bc()             # free collar termination
            else
                bcl = fixed_bc()               # child: R3X receives parent trio
            end
        end
        if i <= n - 1   # right junction j = i
            if junction_parent[i] == i
                iMax += dx[i]; cells += 1
                bcr = natural_bc()
            else
                bcr = fixed_bc()
            end
        end

        bgp = base.grid_params
        # RL annulus patches carry the GLOBAL ring numbering through an
        # explicit patchOffsetL (ring point counts and wavenumber support are
        # tied to the global ring index), while spectralIndexL stays 1 — the
        # patch is its own spectral frame. The (collar-extended) inner edge
        # must sit on a whole number of this patch's own cells from the global
        # origin for the offset to be well-defined.
        ring_offset = 0
        if geometry == "RL"
            dxi = (iMax - iMin) / cells
            cells_inside = (iMin - nest.boundaries[1]) / dxi
            isapprox(cells_inside, round(cells_inside); atol=1e-8) || throw(ArgumentError(
                "Patch $i: inner edge $(iMin) is not a whole number of its own " *
                "cells ($(dxi)) from the origin $(nest.boundaries[1]) — required " *
                "for the RL global ring numbering"))
            ring_offset = round(Int, cells_inside) * bgp.mubar
        end
        gps[i] = SpringsteelGridParameters(
            geometry = geometry,
            iMin = iMin, iMax = iMax,
            num_cells = cells, num_cells_i = cells,
            mubar = bgp.mubar, quadrature = bgp.quadrature,
            l_q = bgp.l_q,
            BCL = bcl, BCR = bcr,
            jMin = bgp.jMin, jMax = bgp.jMax,
            max_wavenumber = bgp.max_wavenumber,
            kMin = bgp.kMin, kMax = bgp.kMax,
            num_cells_k = bgp.num_cells_k, kDim = bgp.kDim, b_kDim = bgp.b_kDim,
            BCB = bgp.BCB, BCT = bgp.BCT,
            vars = bgp.vars,
            fourier_filter = bgp.fourier_filter,
            chebyshev_filter = bgp.chebyshev_filter,
            spline_filter = bgp.spline_filter,
            patchOffsetL = ring_offset,
        )

        ics = isempty(base.initial_conditions) ? "" :
              joinpath(dirname(base.initial_conditions),
                       "nest$(i)_" * basename(base.initial_conditions))
        models[i] = ModelParameters(
            ts = ts_actual[i],
            integration_time = base.integration_time,
            output_interval = base.output_interval,
            equation_set = base.equation_set,
            initial_conditions = ics,
            output_dir = joinpath(base.output_dir, "nest$i"),
            ref_state_file = base.ref_state_file,
            grid_params = gps[i],
            physical_params = base.physical_params,
            options = base.options,
        )
    end

    # ── Interface metadata + collar geometry (throwaway driver-side grids) ──
    grids = [createGrid(models[i].grid_params) for i in 1:n]
    interfaces = NestInterface[]
    for j in 1:(n - 1)
        p = junction_parent[j]
        c = (p == j) ? j + 1 : j
        parent_side = (p == j) ? :right : :left
        child_side = (p == j) ? :left : :right
        x_int = nest.boundaries[j + 1]

        iface = PatchInterface(grids[p], grids[c], parent_side, child_side, :i;
                               is_stacked=true)

        # Parent collar description (mish points past the nominal junction,
        # toward the child)
        ppts = getGridpoints(grids[p])
        collar_x = Float64[]
        collar_rows = Int[]
        collar_pts = zeros(Float64, 0, 2)
        collar_kmax = Int[]
        if geometry == "RL"
            # Ragged rings: the collar is one parent cell = mubar rings. Rows
            # follow the cumulative ring point counts; each point carries its
            # target ring's supported max wavenumber for the evaluation.
            pgp = models[p].grid_params
            row = 0
            pt_rows = Int[]
            for r in 1:pgp.iDim
                ri = r + pgp.patchOffsetL
                lpoints = 4 + 4 * ri
                rrad = ppts[row + 1, 1]
                in_collar = parent_side == :right ? (rrad > x_int + 1e-9) :
                                                    (rrad < x_int - 1e-9)
                if in_collar
                    append!(collar_rows, (row + 1):(row + lpoints))
                    append!(pt_rows, (row + 1):(row + lpoints))
                    append!(collar_kmax, fill(ri, lpoints))
                end
                row += lpoints
            end
            collar_pts = ppts[pt_rows, :]
        else
            xcol = geometry == "R" ? vec(ppts) : ppts[1:models[p].grid_params.kDim:end, 1]
            if parent_side == :right
                cols = findall(x -> x > x_int + 1e-9, xcol)
            else
                cols = findall(x -> x < x_int - 1e-9, xcol)
            end
            collar_x = xcol[cols]
            if geometry == "R"
                collar_rows = cols
            else
                kDim = models[p].grid_params.kDim
                for q in cols
                    append!(collar_rows, ((q - 1) * kDim + 1):(q * kDim))
                end
            end
        end

        push!(interfaces, NestInterface(p, c, parent_side, child_side, x_int,
                                        iface.metadata, collar_rows, collar_x,
                                        nslices, collar_pts, collar_kmax))
    end

    parent_ifaces = [Int[] for _ in 1:n]
    child_ifaces = [Int[] for _ in 1:n]
    for (k, ni) in enumerate(interfaces)
        push!(child_ifaces[ni.parent], k)
        push!(parent_ifaces[ni.child], k)
    end

    topo = NestTopology(interfaces, parent_ifaces, child_ifaces, n_sub, ts_actual)
    return models, topo
end

"""
    inject_collar!(parent_physical, child_grid, ni::NestInterface)

Fine→coarse feedback: evaluate the child's representation at the parent's
collar mish points and overwrite the parent's `physical` rows there (all
variables, all derivative slices). Must run after the parent's physical
state is current for the time level and before its tendency computation.
"""
function inject_collar!(parent_physical::AbstractArray{Float64,3},
                        child_grid, ni::NestInterface)
    vals = isempty(ni.collar_pts) ?
           evaluate_grid_ipoints(child_grid, ni.collar_x) :
           evaluate_grid_points(child_grid, ni.collar_pts; kmax = ni.collar_kmax)
    nvars = size(parent_physical, 2)
    @inbounds for s in 1:ni.nslices, v in 1:nvars
        for (r, row) in enumerate(ni.collar_rows)
            parent_physical[row, v, s] = vals[r, v, s]
        end
    end
    return parent_physical
end

# ────────────────────────────────────────────────────────────────────────────
# Distributed nested integration
# ────────────────────────────────────────────────────────────────────────────
#
# Each nest patch owns a disjoint worker group running the existing tile
# machinery (initialize_model / advanceTimestep / splineTransform!), driven by
# run_nested_patch on the group's master worker. Groups exchange over
# RemoteChannels:
#   down (parent master → child master): the parent's R3X trio payload, one
#       per parent step; the child brackets consecutive payloads and applies
#       lerp_payload!-interpolated trios at each of its substeps.
#   up (child master → parent master): the child's collar evaluation
#       (evaluate_grid_ipoints at the parent's collar mish points), one per
#       parent step; the parent injects it into its border tiles' physical
#       arrays before each tendency step.
#
# Schedule per parent step t (sequential-consistent, DeMaria et al. 1992):
#   parent: take collar(t-1) → advance → send payload(t) →
#   child:  subcycles n_sub steps with lerp(payload(t-1), payload(t)) →
#           sends collar(t) → parent step t+1 takes it.
# The initial exchange (state at t=0) primes both channel directions so the
# blocking takes are uniform; the chain order (roots send first) makes it
# deadlock-free.

"""Per-interface link data for a patch acting as the CHILD."""
struct NestParentLink
    meta::PatchInterfaceMetadata
    down::RemoteChannel                  # take parent trio payloads
    up::RemoteChannel                    # send my collar evaluation
    parent_collar_x::Vector{Float64}     # where the parent needs my values
    # RL: explicit (r, λ) points + per-point wavenumber truncation (empty for
    # tensor-product geometries, which use parent_collar_x)
    parent_collar_pts::Matrix{Float64}
    parent_collar_kmax::Vector{Int}
end

"""Evaluate this patch's representation at the parent's collar points."""
function _collar_evaluation(patch, pl::NestParentLink)
    if isempty(pl.parent_collar_pts)
        return evaluate_grid_ipoints(patch, pl.parent_collar_x)
    end
    return evaluate_grid_points(patch, pl.parent_collar_pts;
                                kmax = pl.parent_collar_kmax)
end

"""Per-interface link data for a patch acting as the PARENT."""
struct NestChildLink
    meta::PatchInterfaceMetadata
    down::RemoteChannel                  # send my trio payload
    up::RemoteChannel                    # take child collar values
    collar_rows::Vector{Int}             # my patch-global physical rows
    nslices::Int
end

"""
    advance_nested_timestep(mtile, sharedSpectral, haloSend, haloReceive, t, injections)

`advanceTimestep` with fine→coarse collar injection: after the tile transform
and before the tendency computation, overwrite the tile's physical rows given
by each `(tile_local_rows, values)` pair with the child-grid evaluation, so
the Galerkin loads near the interface integrate child data.
"""
function advance_nested_timestep(mtile::ModelTile, sharedSpectral::SharedArray{Float64},
        haloSend::RemoteChannel, haloReceive::RemoteChannel, t::Int64,
        injections::Vector{Tuple{Vector{Int}, Array{Float64,3}}})

    # Transform to local physical tile
    tileTransform!(sharedSpectral, mtile.tile, mtile.tile.physical, mtile.tile.spectral)

    # Fine→coarse feedback: inject child values (all vars, all derivative
    # slices) at the collar mish points owned by this tile
    for (rows, vals) in injections
        nsl = size(vals, 3)
        nv = size(vals, 2)
        @inbounds for s in 1:nsl, v in 1:nv
            for (r, row) in enumerate(rows)
                mtile.tile.physical[row, v, s] = vals[r, v, s]
            end
        end
    end

    checkCFL(mtile.tile; t=t, ts=mtile.model.ts, where="worker tile")

    if num_columns(mtile.tile) > 0
        Threads.@threads :static for c in 1:num_columns(mtile.tile)
            advance_column(mtile, c, t)
        end
    else
        advance_column(mtile, -1, t)
    end

    calcTendency(mtile)
    put!(haloSend, extract_halo_values(mtile.tile))
    write_tile_to_shared!(sharedSpectral, mtile.tile, mtile.patch_b_iDim)
    mtile.haloReceiveBuffer .= take!(haloReceive)
    accumulate_at_map!(sharedSpectral, mtile.haloReceiveMap, mtile.haloReceiveBuffer)

    return nothing
end

"""
    run_nested_patch(patch, model, workerids, n_sub, parent_links, child_links)

Time-integrate one nest patch on its worker group (runs on the group master).
Mirrors `run_model`/`model_loop` — group halo ring, shared spectral array,
per-step tile advance — plus the nest coupling: parent-payload application at
every substep (time-interpolated), child-payload emission and child-collar
injection at every own step, and collar evaluation upstream at every cycle
end.
"""
function run_nested_patch(patch::AbstractGrid, model::ModelParameters,
                          workerids::Vector{Int64}, n_sub::Int,
                          parent_links::Vector{NestParentLink},
                          child_links::Vector{NestChildLink})

    num_workers = length(workerids)
    println("Nest patch starting up with $(num_workers) workers...")

    # ── Group halo ring (as in run_model, but robust to non-contiguous ids) ─
    haloInit = RemoteChannel(()->Channel{Array{Float64}}(1), workerids[1])
    wait(save_at(workerids[1], :haloReceive, :($(haloInit))))
    for (a, b) in zip(workerids[1:end-1], workerids[2:end])
        wait(save_at(a, :haloSend, :(RemoteChannel(()->Channel{Array{Float64}}(1), $(b)))))
        receiver = get_val_from(a, :haloSend)
        wait(save_at(b, :haloReceive, :($(receiver))))
    end
    wait(save_at(last(workerids), :haloSend,
            :(RemoteChannel(()->Channel{Array{Float64}}(1), $(workerids[1])))))
    haloReceive = get_val_from(last(workerids), :haloSend)
    haloInitBuffer = zeros(Float64, 1)
    haloReceiveMap = get_val_from(last(workerids), :(mtile.haloSendMap))
    haloReceiveBuffer = zeros(Float64, nnz(haloReceiveMap))

    # ── Per-worker collar injection maps ─────────────────────────────────────
    # A collar row (patch-global) belongs to the worker whose tile covers its
    # mish column; precompute (worker → selection into collar values, tile-local rows).
    inj_maps = Vector{Vector{Tuple{Int, Vector{Int}, Vector{Int}}}}(undef, length(child_links))
    if model.grid_params.geometry == "RL"
        # Ragged rings: the tensor-product row arithmetic below does not apply.
        # v1 restricts RL nest patches to a single worker whose tile spans the
        # whole patch, so the tile-local rows ARE the patch rows.
        num_workers == 1 || throw(ErrorException(
            "RL nest patches currently support exactly 1 worker per patch " *
            "(ragged ring layout), got $(num_workers)"))
        for (li, cl) in enumerate(child_links)
            inj_maps[li] = [(workerids[1], collect(1:length(cl.collar_rows)),
                             copy(cl.collar_rows))]
        end
    else
        kDim_eff = model.grid_params.geometry == "R" ? 1 : model.grid_params.kDim
        for (li, cl) in enumerate(child_links)
            inj_maps[li] = Tuple{Int, Vector{Int}, Vector{Int}}[]
            for w in workerids
                offL, iDim = get_val_from(w, :((mtile.tile.params.patchOffsetL, mtile.tile.params.iDim)))
                lo = offL * kDim_eff
                hi = (offL + iDim) * kDim_eff
                sel = findall(r -> lo < r <= hi, cl.collar_rows)
                isempty(sel) && continue
                local_rows = [cl.collar_rows[s] - lo for s in sel]
                push!(inj_maps[li], (w, sel, local_rows))
            end
        end
    end

    # ── Initial exchange (state at t = 0) ────────────────────────────────────
    # Take parent payloads first (roots have none, so the chain resolves
    # outermost-in); refresh the local fit with the correct ahat before
    # donating to children — the ICs' spectral b is unchanged by ahat.
    p0 = Vector{InterfacePayload}(undef, length(parent_links))
    p1 = Vector{InterfacePayload}(undef, length(parent_links))
    pj = Vector{InterfacePayload}(undef, length(parent_links))
    for (li, pl) in enumerate(parent_links)
        p0[li] = take!(pl.down)
        map(wait, [get_from(w, :(Springsteel.apply_interface_payload!($(pl.meta), patch, $(p0[li])))) for w in workerids])
    end
    if !isempty(parent_links)
        gridTransform!(patch)
    end
    for (li, pl) in enumerate(parent_links)
        # Interpolation buffer sized from the metadata — NOT computed from this
        # (child) grid: the parent-side metadata's spectral block sizes belong
        # to the parent's grid and generally differ from the child's.
        pj[li] = Springsteel._allocate_payload(pl.meta)
    end
    for cl in child_links
        put!(cl.down, compute_interface_payload(cl.meta, patch))
    end
    for pl in parent_links
        put!(pl.up, _collar_evaluation(patch, pl))
    end

    # ── Shared spectral seed + t = 0 output (as in run_model) ───────────────
    sharedSpectral = SharedArray{Float64,2}((size(patch.spectral, 1), size(patch.spectral, 2)))
    sharedSpectral[:] .= patch.spectral[:]
    for w in workerids
        save_at(w, :sharedSpectral, sharedSpectral)
    end
    map(wait, [get_from(w, :(splineTransform!(sharedSpectral, patch, mtile.tile))) for w in workerids])
    patch.spectral .= sharedSpectral
    gridTransform!(patch)
    write_output(patch, model, 0.0)
    flush(stdout)
    checkCFL(patch)

    # ── Main loop ────────────────────────────────────────────────────────────
    num_ts = round(Int, model.integration_time / model.ts)
    output_int = round(Int, model.output_interval / model.ts)
    cfl_int = max(1, round(Int, get(model.options, :cfl_interval, model.output_interval) / model.ts))
    println("Integrating $(model.ts) sec increments for $(num_ts) timesteps ($(n_sub) per parent step)")
    cfl_diag_on = haskey(model.grid_params.vars, "w")
    dz_min = dx_min = c_bar = 0.0
    if cfl_diag_on
        dz_min, dx_min = grid_spacing_minima(patch, model)
        c_bar = sqrt(max(0.0, get_val_from(workerids[1], :(Scythe.sound_speed_sq(mtile.ref_state)))))
    end

    for t = 1:num_ts
        j = isempty(parent_links) ? 1 : mod1(t, n_sub)

        # New bracketing payload from each parent at cycle start
        if !isempty(parent_links) && j == 1
            for (li, pl) in enumerate(parent_links)
                p1[li] = take!(pl.down)
            end
        end

        # Child collar values for this step (child state at my time t-1)
        inj_vals = [take!(cl.up) for cl in child_links]

        # Advance all tiles (with collar injection on the owning workers)
        @turbo sharedSpectral .= 0.0
        put!(haloInit, haloInitBuffer)
        futures = Future[]
        for w in workerids
            w_inj = Tuple{Vector{Int}, Array{Float64,3}}[]
            for (li, _) in enumerate(child_links)
                for (mw, sel, local_rows) in inj_maps[li]
                    mw == w || continue
                    push!(w_inj, (local_rows, inj_vals[li][sel, :, :]))
                end
            end
            push!(futures, get_from(w, :(Scythe.advance_nested_timestep(mtile, sharedSpectral, haloSend, haloReceive, $(t), $(w_inj)))))
        end
        map(wait, futures)
        haloReceiveBuffer .= take!(haloReceive)
        accumulate_at_map!(sharedSpectral, haloReceiveMap, haloReceiveBuffer)

        # Apply the time-interpolated parent trio for this substep on every
        # group worker (the b→a solve honoring ahat runs per worker in the
        # 3-arg splineTransform!)
        if !isempty(parent_links)
            θ = j / n_sub
            for (li, pl) in enumerate(parent_links)
                lerp_payload!(pj[li], p0[li], p1[li], θ)
                map(wait, [get_from(w, :(Springsteel.apply_interface_payload!($(pl.meta), patch, $(pj[li])))) for w in workerids])
            end
        end

        # Reset tiles from the merged spectral state
        map(wait, [get_from(w, :(splineTransform!(sharedSpectral, patch, mtile.tile))) for w in workerids])

        if any(!isfinite, sharedSpectral)
            error("Non-finite spectral coefficient at t=$(round(t*model.ts; digits=3)) s ! CFL condition likely violated")
        end

        # Materialize the master patch when anything downstream needs it
        is_cfl_step = cfl_diag_on && mod(t, cfl_int) == 0
        is_output_step = mod(t, output_int) == 0
        cycle_end = !isempty(parent_links) && j == n_sub
        if !isempty(child_links) || cycle_end || is_cfl_step || is_output_step
            patch.spectral .= sharedSpectral
        end
        if !isempty(child_links) || is_cfl_step || is_output_step
            gridTransform!(patch)
        end

        # Donate my trio payload to each child (they subcycle against it)
        for cl in child_links
            put!(cl.down, compute_interface_payload(cl.meta, patch))
        end

        # Cycle end: send my collar evaluation upstream, roll the brackets
        if cycle_end
            for pl in parent_links
                put!(pl.up, _collar_evaluation(patch, pl))
            end
            for li in eachindex(parent_links)
                p0[li] = p1[li]
            end
        end

        if is_cfl_step
            cfl_diagnostics(patch, model, t, dz_min, dx_min, c_bar)
        end
        if is_output_step
            write_output(patch, model, t * model.ts)
            checkCFL(patch; t=t, ts=model.ts, where="output")
        end
        flush(stdout)
    end

    patch.spectral .= sharedSpectral
    gridTransform!(patch)
    println("Nest patch done with time integration")
    return true
end

"""
    integrate_nested_model(nest::NestedModelParameters)

Main entry point for a nested run. Builds the per-patch models and topology,
partitions the available workers into per-patch groups, initializes each
group with the existing `initialize_model`, wires the inter-group payload
channels, and runs all patches concurrently via `run_nested_patch`.

Per-patch output (including logs) goes to `base.output_dir/nest\$i/`.
Requires `sum(workers_per_patch)` worker processes.
"""
function integrate_nested_model(nest::NestedModelParameters)

    models, topo = build_nest(nest)
    n = length(models)

    ws = workers()
    needed = sum(nest.workers_per_patch)
    (ws[1] != 1 && length(ws) >= needed) || throw(ErrorException(
        "Need at least $needed worker processes for this nest, have $(ws[1] == 1 ? 0 : length(ws))"))

    groups = Vector{Vector{Int64}}(undef, n)
    off = 0
    for i in 1:n
        groups[i] = ws[(off + 1):(off + nest.workers_per_patch[i])]
        off += nest.workers_per_patch[i]
    end

    println("Starting nested model: $n patches on worker groups $(groups)...")
    for i in 1:n
        warn_timestep_stability(models[i].grid_params, models[i].ts)
        mkpath(models[i].output_dir)
    end

    # Redirect each group master's output to its nest's log files
    for i in 1:n
        gm = groups[i][1]
        outfile = joinpath(models[i].output_dir, "scythe_out.log")
        errfile = joinpath(models[i].output_dir, "scythe_err.log")
        wait(save_at(gm, :out, :(open($(outfile), "w"))))
        wait(save_at(gm, :err, :(open($(errfile), "w"))))
        wait(get_from(gm, :(redirect_stdout(out))))
        wait(get_from(gm, :(redirect_stderr(err))))
    end

    # Initialize every group's patch (concurrently; each group master builds
    # its grid, reads its ICs, and constructs the group's ModelTiles)
    map(wait, [save_at(groups[i][1], :patch,
                       :(initialize_model($(models[i]), $(groups[i])))) for i in 1:n])

    # Inter-group channels: down hosted on the child's master, up on the parent's
    down = Vector{RemoteChannel}(undef, length(topo.interfaces))
    up = Vector{RemoteChannel}(undef, length(topo.interfaces))
    for (k, ni) in enumerate(topo.interfaces)
        down[k] = RemoteChannel(() -> Channel{InterfacePayload}(2), groups[ni.child][1])
        up[k] = RemoteChannel(() -> Channel{Array{Float64,3}}(2), groups[ni.parent][1])
    end

    # Per-patch link bundles
    futures = Vector{Future}(undef, n)
    for i in 1:n
        plinks = NestParentLink[]
        for k in topo.parent_ifaces[i]
            ni = topo.interfaces[k]
            # The parent's collar coordinates are carried on the interface
            # (collar_x for tensor-product geometries, (r, λ) points + per-ring
            # wavenumber truncation for RL).
            push!(plinks, NestParentLink(ni.meta, down[k], up[k], ni.collar_x,
                                         ni.collar_pts, ni.collar_kmax))
        end
        clinks = NestChildLink[]
        for k in topo.child_ifaces[i]
            ni = topo.interfaces[k]
            push!(clinks, NestChildLink(ni.meta, down[k], up[k], ni.collar_rows, ni.nslices))
        end
        futures[i] = get_from(groups[i][1],
            :(Scythe.run_nested_patch(patch, model, $(groups[i]), $(topo.n_sub[i]),
                                      $(plinks), $(clinks))))
    end
    # Poll every patch future so an exception on ANY group master surfaces
    # promptly. A plain in-order wait blocks on patch 1 while a failed
    # neighbor's exception sits unobserved — the neighbors then deadlock on
    # the dead patch's channels and the failure looks like a silent hang.
    done = falses(n)
    while !all(done)
        for i in 1:n
            if !done[i] && isready(futures[i])
                fetch(futures[i])       # rethrows the group's exception
                done[i] = true
            end
        end
        all(done) || sleep(0.25)
    end

    # Finalize each patch and close the log files
    for i in 1:n
        gm = groups[i][1]
        wait(get_from(gm, :(finalize_model(patch, model))))
        wait(get_from(gm, :(close(out))))
        wait(get_from(gm, :(close(err))))
    end
    println("Nested integration complete!")
    return models, topo, groups
end
