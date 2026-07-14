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
    else
        throw(ArgumentError(
            "Nesting currently supports geometries \"R\" and \"RiRk\", got \"$geometry\""))
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

    # ── Junction parent/child determination (exact 2:1) ─────────────────────
    junction_parent = zeros(Int, n - 1)   # patch index of the parent at junction j
    for j in 1:(n - 1)
        ratio = dx[j] / dx[j + 1]
        if isapprox(ratio, 2.0; rtol=1e-10)
            junction_parent[j] = j            # left patch is coarser
        elseif isapprox(ratio, 0.5; rtol=1e-10)
            junction_parent[j] = j + 1        # right patch is coarser
        else
            throw(ArgumentError(
                "Junction $j at x=$(nest.boundaries[j + 1]): cell-width ratio " *
                "$(ratio):1 — nesting requires exactly 2:1 (parent the coarser side)"))
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
        gps[i] = SpringsteelGridParameters(
            geometry = geometry,
            iMin = iMin, iMax = iMax,
            num_cells = cells, num_cells_i = cells,
            mubar = bgp.mubar, quadrature = bgp.quadrature,
            l_q = bgp.l_q,
            BCL = bcl, BCR = bcr,
            kMin = bgp.kMin, kMax = bgp.kMax,
            num_cells_k = bgp.num_cells_k, kDim = bgp.kDim, b_kDim = bgp.b_kDim,
            BCB = bgp.BCB, BCT = bgp.BCT,
            vars = bgp.vars,
            fourier_filter = bgp.fourier_filter,
            chebyshev_filter = bgp.chebyshev_filter,
            spline_filter = bgp.spline_filter,
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

        # Parent collar mish columns (past the nominal junction, toward child)
        ppts = getGridpoints(grids[p])
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
            collar_rows = Int[]
            for q in cols
                append!(collar_rows, ((q - 1) * kDim + 1):(q * kDim))
            end
        end

        push!(interfaces, NestInterface(p, c, parent_side, child_side, x_int,
                                        iface.metadata, collar_rows, collar_x,
                                        nslices))
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
    vals = evaluate_grid_ipoints(child_grid, ni.collar_x)
    nvars = size(parent_physical, 2)
    @inbounds for s in 1:ni.nslices, v in 1:nvars
        for (r, row) in enumerate(ni.collar_rows)
            parent_physical[row, v, s] = vals[r, v, s]
        end
    end
    return parent_physical
end
