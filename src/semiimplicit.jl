# Functions for model integration

using Distributed
using DistributedData
using SharedArrays
using CSV
using DataFrames
using LoopVectorization
using LinearAlgebra
import Base.Threads.@spawn
using SparseArrays
using SuiteSparse

# Need to export these for distributed operations in Main namespace
export createModelTile, advanceTimestep, advanceTimestepA, advanceTimestepB
export initialize_model, run_model, finalize_model

"""
    ModelTile

Fundamental computational unit holding model state, tendencies, reference state,
and spectral transform infrastructure for a single tile in the domain decomposition.
"""
struct ModelTile{G<:AbstractGrid, R<:AbstractReferenceState,
                 H<:Factorization, D<:Factorization, MC<:NamedTuple, N, KC, SD, MSC,
                 SWS<:NamedTuple}
    model::ModelParameters
    # Concretely typed, so `mtile.tile.physical` and `mtile.tile.kbasis` infer. Declaring
    # this `AbstractGrid` made every view and broadcast in the per-column equation-set
    # bodies fall back to `Any` and box, which is what drove the GC over (12.6e9 pool
    # allocations in a 900 s straka93 run). Springsteel's grids are already concrete.
    tile::G
    var_np1::Matrix{Float64}
    expdot_incr::Matrix{Float64}
    expdot_n::Matrix{Float64}
    expdot_nm1::Matrix{Float64}
    expdot_nm2::Matrix{Float64}
    impdot_n::Matrix{Float64}
    impdot_nm1::Matrix{Float64}
    impdot_nm2::Matrix{Float64}
    # Rank-parameterized, NOT `Matrix`: `getGridpoints` returns a bare `Vector` for the 1-D
    # geometries (it hands back `ibasis.data[1,1].mishPoints` directly) and a matrix for
    # everything else — which is why the reference-state setup below reads
    # `tilepoints[:, ndims(tilepoints)]`.
    tilepoints::Array{Float64, N}
    ref_state::R
    patchMap::SparseMatrixCSC{Float64, Int64}
    haloSendMap::SparseMatrixCSC{Float64, Int64}
    haloReceiveMap::SparseMatrixCSC{Float64, Int64}
    haloReceiveBuffer::Vector{Float64}
    patch_b_iDim::Int64
    # Parameterized rather than pinned to a single factorization type: the concrete type
    # varies with both the grid and the run configuration. `h_matrix` is an `LU` over a
    # `Matrix` when semi-implicit is on but an `LU` over a `Tridiagonal` (the 2x2 dummy)
    # when it is off; `diffusion_matrix` is a `BunchKaufman` on RiRk and an `LU` on RZ.
    h_matrix::H
    diffusion_matrix::D
    # Implicit vertical-diffusion tendency history, separate from `impdot_*`. The
    # acoustic solvers own the `impdot` slots of every variable they touch (p, rho_d,
    # rho_t, w, E_t), so a set that wants implicit vertical diffusion on w — or on a
    # DIAGNOSED scalar such as the moist_compressible potential temperature — has no
    # free `impdot` column to store its AI2* history in. `diffdot_*` is that channel.
    # Only `moist_compressible_XZ` uses it; the older sets still stage their (currently
    # unconsumed) vertical diffusion in `impdot`.
    diffdot_n::Matrix{Float64}
    diffdot_nm1::Matrix{Float64}
    # Horizontal-acoustic AI2* history channel for options[:horizontal_semiimplicit]
    # (mc sets only; zero-size otherwise). Separate from `impdot_*` for the same
    # reason `diffdot_*` is: the vertical acoustic solve owns the impdot slots of
    # p/rho_d/rho_t/E_t, and the horizontal legs need their own history levels.
    # The u column is the stored applied increment of the patch-level sweep
    # (`horizontal_si_load_increment!`); the rest are staged fresh in `mc_driver!`.
    hacdot_n::Matrix{Float64}
    hacdot_nm1::Matrix{Float64}
    # Per-variable vertical-diffusion factorizations for the total-energy set, keyed
    # :u/:w/:heat and :u_first/:w_first/:heat_first. Empty NamedTuple for other sets.
    #
    # A NamedTuple rather than a Dict because these six are NOT all the same concrete type:
    # the boundary conditions differ per variable, which flips `factorize`'s symmetry
    # detection, so on RiRk `u` comes back an `LU` while `w` comes back a `BunchKaufman`.
    # A Dict would have to widen its value type to the abstract `Factorization` join and
    # box on every lookup, in the hot path. A NamedTuple is concrete AND heterogeneous.
    mc_diffusion_matrices::MC
    # Reusable vertical work columns, indexed [thread, variable]. The equation sets used to
    # `deepcopy(tile.kbasis.data[v])` a fresh column per variable, PER COLUMN, PER TIMESTEP —
    # 93 allocations a pop, and 60% of all remaining per-column allocations. The copy only
    # exists to get private `uMish`/`b`/`a` work vectors; it also clones the (read-only) basis
    # matrices and factorizations, which is the expensive part.
    #
    # Indexed by thread, not by column: per-column scratch would need 2.9 GB on a full-mode RZ
    # run (1536 columns x 1.9 MiB), whereas per-thread needs ~90 MiB. Safe because the column
    # loop in `advanceTimestep` is `@threads :static`, which pins each iteration to a fixed
    # thread, so `threadid()` is a stable owner tag. Per-VARIABLE as well as per-thread because
    # callers hold two columns live at once (`semiimplicit_adjustment_p` reads `p_nstar`, which
    # aliases the p-column's `uMish`, after transforming the w-column) and because
    # `Btransform!`/`Atransform!` consult the column's own boundary conditions.
    #
    # Empty (`Matrix{Nothing}`) for grids with no vertical basis — `NoBasisArray` has no `data`.
    scratch_columns::Matrix{KC}
    # Galerkin solve data (M0, M1, W, Nb) for a cubic B-spline k-basis; `nothing` otherwise.
    # `_vertical_solve!` used to fetch this from a global Dict behind a global lock on EVERY
    # call — 4 solves x every column x every timestep, from every thread. It is per-grid and
    # invariant, so it is hoisted here.
    solve_data::SD
    # Per-thread work vectors for `_vertical_solve!`, columns indexed by threadid() (same
    # `@threads :static` ownership rule as `scratch_columns`).
    solve_rhs::Matrix{Float64}    # (nmish, nthreads) — W .* rhs_mish, B-spline path only
    solve_load::Matrix{Float64}   # (nload, nthreads) — the Helmholtz load vector / RHS
    # One NamedTuple of kDim work vectors per thread, for `moist_compressible_XZ`'s 42 live
    # broadcast temporaries (`p = pp .+ pbar` and friends). Empty for every other set.
    # See `_allocate_mc_scratch` for why these are keyed by NAME rather than by index.
    mc_scratch::MSC
    # Consistently-retrieved diagnostics of the RESTING reference for the total-energy
    # set's vertical moist diffusion: s_tbar and rho_vbar computed through the SAME
    # retrieval pipeline the equation set runs each step, so at rest the diffused
    # perturbations s_t' and rho_v' are zero BIT-FOR-BIT and diffusion cannot cook the
    # base (the reference's own Tbar is not bit-identical to the retrieved T). Empty
    # vectors for every other equation set.
    mc_ref_diag::NamedTuple{(:s_tbar, :rho_vbar, :Pxi_prof),
                            Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}
    # One NamedTuple of full-tile-length work vectors for `Twoway_PV_mixing`'s 13 broadcast
    # temporaries. Empty for every other set. See `_allocate_sw_scratch` for why there is one
    # workspace per TILE here rather than one per thread as `mc_scratch` has.
    sw_scratch::SWS
end

"""
    scratch_column(mtile, var_index)

The calling thread's reusable vertical work column for variable slot `var_index`.
Replaces `deepcopy(mtile.tile.kbasis.data[var_index])` in the per-column hot path.

The contents carry no meaning between uses: every caller either overwrites `uMish` before
transforming, or has `_vertical_solve!` set `a` outright.
"""
@inline function scratch_column(mtile::ModelTile, var_index::Int64)
    return @inbounds mtile.scratch_columns[Threads.threadid(), var_index]
end

"""Reusable per-thread vertical work columns, or an empty matrix if the grid has no k-basis."""
_allocate_scratch_columns(::NoBasisArray, tile::AbstractGrid) = Matrix{Nothing}(undef, 0, 0)

function _allocate_scratch_columns(kbasis, tile::AbstractGrid)
    nthreads = Threads.maxthreadid()
    nvars = size(tile.physical, 2)
    return [deepcopy(kbasis.data[v]) for _ in 1:nthreads, v in 1:nvars]
end

"""
    _allocate_solve_workspace(kbasis, tile) -> (solve_data, solve_rhs, solve_load)

Hoist `_vertical_solve!`'s per-call work out of the hot path: the B-spline Galerkin solve data
(fetched behind a global lock on every call before this) and the per-thread work vectors.
"""
_allocate_solve_workspace(::NoBasisArray, tile::AbstractGrid) =
    (nothing, zeros(Float64, 0, 0), zeros(Float64, 0, 0))

function _allocate_solve_workspace(kbasis, tile::AbstractGrid)
    nthreads = Threads.maxthreadid()
    kcol = kbasis.data[1]
    if kcol isa CubicBSpline.Spline1D
        d = _rirk_solve_data(kcol)
        # W is sampled at the kDim mish points; the Galerkin load vector lives in spline space.
        return (d, zeros(Float64, length(d.W), nthreads),
                   zeros(Float64, size(d.M0, 2), nthreads))
    else
        # Chebyshev: the load vector IS the mish-point RHS with homogeneous boundary rows.
        nz = length(kcol.mishPoints)
        return (nothing, zeros(Float64, 0, nthreads), zeros(Float64, nz, nthreads))
    end
end

"""
    createModelTile(patch, tile, model, haloReceiveMap)

Create and initialize a [`ModelTile`](@ref) with allocated state arrays, reference state,
patch-to-tile mappings, halo exchange buffers, and pre-computed Helmholtz matrices.

# Arguments
- `patch::AbstractGrid`: the full domain grid (SpringsteelGrid).
- `tile::AbstractGrid`: the tile grid for this worker (SpringsteelGrid).
- `model::ModelParameters`: model configuration.
- `haloReceiveMap::SparseMatrixCSC{Float64, Int64}`: sparse map for halo receive locations.
"""
function createModelTile(patch::AbstractGrid, tile::AbstractGrid, model::ModelParameters,
        haloReceiveMap::SparseMatrixCSC{Float64, Int64})

    # Allocate some needed arrays
    var_np1 = zeros(Float64,size(tile.physical,1),size(tile.physical,2))
    expdot_incr = zeros(Float64,size(tile.physical,1),size(tile.physical,2))
    expdot_n = zeros(Float64,size(tile.physical,1),size(tile.physical,2))
    expdot_nm1 = zeros(Float64,size(tile.physical,1),size(tile.physical,2))
    expdot_nm2 = zeros(Float64,size(tile.physical,1),size(tile.physical,2))
    impdot_n = zeros(Float64,size(tile.physical,1),size(tile.physical,2))
    impdot_nm1 = zeros(Float64,size(tile.physical,1),size(tile.physical,2))
    impdot_nm2 = zeros(Float64,size(tile.physical,1),size(tile.physical,2))
    diffdot_n = zeros(Float64,size(tile.physical,1),size(tile.physical,2))
    diffdot_nm1 = zeros(Float64,size(tile.physical,1),size(tile.physical,2))
    # Setup-time validation of the exact (unsplit) 2-D semi-implicit option:
    # incompatible flags and unsupported geometries error loudly here, before
    # any allocation (defined in exact_si.jl; a no-op when the option is off).
    validate_exact_si_options(model)

    # Horizontal-acoustic history channel, allocated when either the ADI sweep
    # or the exact 2-D solve is on (exact_si shares the hacdot staging and the
    # horizontal_si_history! application; zero-size otherwise so the memory
    # cost is nil for every other run).
    hacdot_dim = (get(model.options, :horizontal_semiimplicit, false) === true ||
                  get(model.options, :exact_si, false) === true) ?
                 (size(tile.physical,1), size(tile.physical,2)) : (0, 0)
    hacdot_n = zeros(Float64, hacdot_dim...)
    hacdot_nm1 = zeros(Float64, hacdot_dim...)

    # Get the local gridpoints
    tilepoints = getGridpoints(tile)

    # Set up the reference file
    ref_state = empty_reference_state()
    if !isempty(model.ref_state_file)
        # z is the LAST gridpoint column (x/r on 2D grids have 2 columns, r/λ/z 3D
        # grids have 3 — `ndims` was wrong there, it is always 2 for a Matrix)
        z_values = tilepoints[1:model.grid_params.kDim,end]
        ref_column = reference_column(tile, model.grid_params)

        # The partial-density equation sets (primitive_equation_*_pd) read the physical
        # vapor/condensate partial-density profiles (ref_rho_v / ref_rho_c) directly, so
        # they keep the Springsteel physical reference state rather than the legacy
        # xi/mu derived view used by the older equation sets.
        physical_ref = uses_physical_reference(model.equation_set)

        if uses_pressure_reference(model.equation_set)
            # The total-energy set (moist_compressible) consumes the pressure-based
            # reference (p, partial densities, E_t, Q_ss) directly.
            # :hydrostatic_reference — build dp̄/dz to satisfy discrete hydrostatic
            # balance EXACTLY rather than to spline-fit accuracy (17% off at the TC
            # lid). Opt-in, off by default: it moves every pressure-reference baseline.
            hydro = get(model.options, :hydrostatic_reference, false)::Bool
            ref_state = model.options[:exact_reference_state] ?
                Springsteel.exact_pressure_reference_state(model.ref_state_file, z_values,
                                                           ref_column; hydrostatic=hydro) :
                Springsteel.calculate_pressure_reference_state(model.ref_state_file, z_values,
                                                               ref_column; hydrostatic=hydro)
            # Rebuild Q_ssbar through the model's own retrieval so the resting column is
            # a discrete fixed point (see consistent_qss_reference). Opt-in, off by
            # default: it moves every pressure-reference baseline (BF02, O01, Straka).
            if get(model.options, :consistent_qss_reference, false)::Bool
                ref_state = consistent_qss_reference(ref_state, z_values, ref_column)
            end
        elseif (model.options[:exact_reference_state])
            ref_state = physical_ref ?
                Springsteel.exact_reference_state(model.ref_state_file, z_values, ref_column) :
                exact_reference_state(model, z_values, ref_column)
        else
            # Build the shared physical-density reference state (Springsteel). The
            # partial-density sets consume it directly; the legacy xi/mu sets view it
            # back as a legacy ReferenceState so their bodies are unchanged.
            phys = Springsteel.calculate_reference_state(model.ref_state_file, z_values,
                ref_column; moisture=true)
            ref_state = physical_ref ? phys : legacy_reference_view(phys, ref_column)
        end
    end

    # Set up the map between the tile and the patch (returns SparseMatrixCSC directly)
    patchMap = calcPatchMap(patch, tile)

    # Set up the map between the tile and its neighbor (returns SparseMatrixCSC directly)
    haloSendMap = calcHaloMap(patch, tile)

    # Set up some buffers to avoid excessive allocations
    haloReceiveBuffer = zeros(Float64, nnz(haloReceiveMap))
    scratch_columns = _allocate_scratch_columns(tile.kbasis, tile)
    solve_data, solve_rhs, solve_load = _allocate_solve_workspace(tile.kbasis, tile)
    # Defined in moist_compressible.jl, which is included after this file — resolved at call
    # time, so the forward reference is fine.
    mc_scratch = _allocate_mc_scratch(tile, model)
    mc_ref_diag = if uses_pressure_reference(model.equation_set) &&
                     !isempty(model.ref_state_file)
        mc_reference_diagnostics(ref_state,
                                 tilepoints[1:model.grid_params.kDim, end])
    else
        (s_tbar = Float64[], rho_vbar = Float64[], Pxi_prof = Float64[])
    end

    # Pre-calculate the Helmholtz matrices. Use a dummy factorization as a
    # structural placeholder where a real one is not needed.
    h_matrix = factorize([1 2; 2 1])
    diffusion_matrix = factorize([1 2; 2 1])
    # The implicit vertical diffusion solve (diffusion_timestep) runs on every
    # step independently of the acoustic solver, so build its matrix whenever the
    # model carries a vertical diffusivity — NOT only when semi-implicit is on.
    # (Gating it on :semiimplicit left a 2×2 dummy in fully-explicit runs, which
    # diffusion_timestep then tried to apply to a kDim-length column.)
    if haskey(model.physical_params, :Kvdiff)
        diffusion_matrix = calc_Helmholtz_diffusion_matrix(tile, model, 1.25 * model.ts * model.physical_params[:Kvdiff])
    end
    # The semi-implicit acoustic solve is UNCONDITIONAL for the pressure-reference
    # (moist_compressible) sets: the explicit acoustic mode was removed when the AI2*
    # staging was made operator-consistent (see semiimplicit_adjustment_p). A config
    # that explicitly asks for the removed mode fails loudly rather than silently
    # producing different numbers; an absent :semiimplicit key simply runs SI.
    # The legacy sets keep the opt-in flag.
    if uses_pressure_reference(model.equation_set)
        if get(model.options, :semiimplicit, true) != true
            error("options[:semiimplicit] => false is not supported for the " *
                  "moist_compressible equation sets: the explicit acoustic mode was " *
                  "removed (the vertical acoustics are always integrated semi-implicitly). " *
                  "Remove the option or set it to true.")
        end
        # The LOCAL reference sound-speed profile, not the domain mean: a mean-c̄²
        # linearization is unstable above Co_z ≈ 4.5 on a stratified sounding
        # (see calc_Helmholtz_semiimplicit_matrix's profile method).
        h_matrix = calc_Helmholtz_semiimplicit_matrix(tile, model, mc_ref_diag.Pxi_prof, 1.25 * model.ts)
    elseif get(model.options, :semiimplicit, false)
        h_matrix = calc_Helmholtz_semiimplicit_matrix(tile, model, sound_speed_sq(ref_state), 1.25 * model.ts)
    end

    # The total-energy set diffuses momentum (u, w; Kvdiff), the diagnosed moist entropy
    # s_t (heat; Kvdiff_heat, defaulting to Kvdiff), and the water species (Kvdiff_water:
    # total water and vapor on the rho_t operator, rain on its own). Each needs its own
    # factorization: the boundary conditions differ (straka93 free-slip u, bf02 no-slip;
    # w a rigid lid; scalars Neumann, rho_r possibly Natural at the surface for rain
    # outflow), and the coefficients are independent. s_t rides on the Neumann E_t
    # column; rho_w' and rho_v' SHARE the rho_t operator so the implied cloud increment
    # delta_rho_c = delta_rho_w - delta_rho_v - delta_rho_r is exactly the diffusion of
    # rho_c. Both the AI2* coefficient (t >= 2) and the first-step AM2 coefficient are
    # cached, so `diffusion_timestep_mc` never factorizes per column.
    # Built only when a coefficient is positive: at K = 0 a vertical solve is not the
    # identity (it refits the column and reapplies the spectral filter), so the routine
    # skips that solve; its matrices are built anyway (cheap) to keep the NamedTuple shape.
    mc_diffusion_matrices = NamedTuple()
    Kv_mc = get(model.physical_params, :Kvdiff, 0.0)
    Kv_heat_mc = get(model.physical_params, :Kvdiff_heat, Kv_mc)
    Kv_water_mc = get(model.physical_params, :Kvdiff_water, 0.0)
    if uses_pressure_reference(model.equation_set) &&
            (Kv_mc > 0.0 || Kv_heat_mc > 0.0 || Kv_water_mc > 0.0)
        bcb = model.grid_params.BCB
        bct = model.grid_params.BCT
        mc_matrix(var, K, coeff) = calc_Helmholtz_diffusion_matrix(tile, model,
            coeff * model.ts * K; bc_bottom = bcb[var], bc_top = bct[var])
        mc_diffusion_matrices = (
            u             = mc_matrix("u",     Kv_mc,       1.25),
            u_first       = mc_matrix("u",     Kv_mc,       0.5),
            w             = mc_matrix("w",     Kv_mc,       1.25),
            w_first       = mc_matrix("w",     Kv_mc,       0.5),
            heat          = mc_matrix("E_t",   Kv_heat_mc,  1.25),
            heat_first    = mc_matrix("E_t",   Kv_heat_mc,  0.5),
            water         = mc_matrix("rho_t", Kv_water_mc, 1.25),
            water_first   = mc_matrix("rho_t", Kv_water_mc, 0.5),
            water_r       = mc_matrix("rho_r", Kv_water_mc, 1.25),
            water_r_first = mc_matrix("rho_r", Kv_water_mc, 0.5))
        # The cylindrical variants carry the tangential wind v (its own BCs) through
        # the same momentum solve as u/w.
        if haskey(model.grid_params.vars, "v")
            mc_diffusion_matrices = merge(mc_diffusion_matrices, (
                v       = mc_matrix("v", Kv_mc, 1.25),
                v_first = mc_matrix("v", Kv_mc, 0.5)))
        end
    end

    mtile = ModelTile(
        model,
        tile,
        var_np1,
        expdot_incr,
        expdot_n,
        expdot_nm1,
        expdot_nm2,
        impdot_n,
        impdot_nm1,
        impdot_nm2,
        tilepoints,
        ref_state,
        patchMap,
        haloSendMap,
        haloReceiveMap,
        haloReceiveBuffer,
        patch.params.b_iDim,
        h_matrix,
        diffusion_matrix,
        diffdot_n,
        diffdot_nm1,
        hacdot_n,
        hacdot_nm1,
        mc_diffusion_matrices,
        scratch_columns,
        solve_data,
        solve_rhs,
        solve_load,
        mc_scratch,
        mc_ref_diag,
        _allocate_sw_scratch(tile, model))
    return mtile
end

"""Extract nonzero row/col indices from a sparse map for SharedArray indexing."""
function sparse_indices(map::SparseMatrixCSC)
    rows, cols, _ = findnz(map)
    return CartesianIndex.(rows, cols)
end

"""Extract nonzero values from a sparse matrix as a dense vector."""
function sparse_values(map::SparseMatrixCSC)
    _, _, vals = findnz(map)
    return vals
end

"""
    extract_halo_values(tile)

Extract the 3-row halo (right boundary) from each wavenumber block of the tile's spectral
array as a dense vector. The ordering matches the structure of `calcHaloMap` so the result
can be directly accumulated into the shared spectral via `accumulate_at_map!`.
"""
function extract_halo_values(tile::AbstractGrid)
    b_iDim = tile.params.b_iDim
    nvars = size(tile.spectral, 2)

    if size(tile.spectral, 1) > b_iDim
        # RL/SL grid: extract from each wavenumber block
        kDim = tile.params.iDim + tile.params.patchOffsetL
        nblocks = 1 + 2 * kDim
        result = zeros(Float64, 3 * nvars * nblocks)
        pos = 1
        for v in 1:nvars
            # k=0 block: last 3 rows
            result[pos:pos+2] .= tile.spectral[b_iDim-2:b_iDim, v]
            pos += 3
            for k in 1:kDim
                p = k * 2
                # Real part: last 3 rows of block
                te = (p - 1) * b_iDim + b_iDim
                result[pos:pos+2] .= tile.spectral[te-2:te, v]
                pos += 3
                # Imaginary part: last 3 rows of block
                te = p * b_iDim + b_iDim
                result[pos:pos+2] .= tile.spectral[te-2:te, v]
                pos += 3
            end
        end
        return result
    else
        # Cartesian 1D: just the last 3 rows
        result = zeros(Float64, 3 * nvars)
        for v in 1:nvars
            result[(v-1)*3+1:v*3] .= tile.spectral[b_iDim-2:b_iDim, v]
        end
        return result
    end
end

"""
    extract_halo_values(tile::Union{RZ_Grid, RiRk_Grid})

RZ and RiRk grids store `b_kDim` consecutive spline blocks (one per vertical
mode — Chebyshev for RZ, B-spline for RiRk), so the 3-row right halo is
extracted from the end of each block. Ordering matches the column-major sparse
indices of the `calcHaloMap`. The spectral layout is identical for both, so the
same extraction applies.
"""
function extract_halo_values(tile::Union{RZ_Grid, RiRk_Grid})
    b_iDim = tile.params.b_iDim
    b_kDim = tile.params.b_kDim
    nvars = size(tile.spectral, 2)
    result = zeros(Float64, 3 * nvars * b_kDim)
    pos = 1
    for v in 1:nvars
        for z in 1:b_kDim
            te = z * b_iDim   # last row of block z
            result[pos:pos+2] .= tile.spectral[te-2:te, v]
            pos += 3
        end
    end
    return result
end

"""
    extract_halo_values(tile::Union{RLZ_Grid, RLR_Grid})

3D cylindrical grids store `b_kDim` vertical-mode blocks, each holding a k = 0
radial-spline sub-block plus real/imag sub-blocks per azimuthal wavenumber
(`1 + 2*kDim` sub-blocks of `b_iDim` rows). The 3-row right halo comes from the
end of every sub-block, in the same ascending row order as the cylindrical
`calcHaloMap`, so the result maps one-to-one onto its sparse indices.
"""
function extract_halo_values(tile::Union{RLZ_Grid, RLR_Grid})
    b_iDim = tile.params.b_iDim
    b_kDim = tile.params.b_kDim
    kDim = tile.params.iDim + tile.params.patchOffsetL
    nblocks = 1 + 2 * kDim
    nvars = size(tile.spectral, 2)
    result = zeros(Float64, 3 * nvars * nblocks * b_kDim)
    pos = 1
    for v in 1:nvars
        for z_b in 1:b_kDim
            base = (z_b - 1) * b_iDim * nblocks
            for j in 1:nblocks
                te = base + j * b_iDim   # last row of sub-block j
                result[pos:pos+2] .= tile.spectral[te-2:te, v]
                pos += 3
            end
        end
    end
    return result
end

"""Accumulate (add) spectral values at sparse map locations in a patch-sized array."""
function accumulate_at_map!(spectral::AbstractArray, map::SparseMatrixCSC, values)
    idx = sparse_indices(map)
    spectral[idx] .+= values
end

"""
    write_tile_to_shared!(sharedSpectral, tile, b_iDim_patch)

Copy the tile's inner (non-halo) spectral coefficients to the correct positions in the
patch-level shared spectral array. The tile→patch index mapping accounts for different
spectral strides per wavenumber block (b_iDim_tile vs b_iDim_patch).
"""
function write_tile_to_shared!(sharedSpectral::SharedArray{Float64}, tile::AbstractGrid,
                                b_iDim_patch::Int64)
    siL = tile.params.spectralIndexL
    b_iDim_tile = tile.params.b_iDim
    inner_rows = b_iDim_tile - 4  # inner region excludes 3-row halo

    # k=0 block (spline coefficients): patch rows siL:(siL+inner_rows) ← tile rows 1:(1+inner_rows)
    sharedSpectral[siL:siL+inner_rows, :] .= tile.spectral[1:1+inner_rows, :]

    # k>=1 Fourier wavenumber blocks only exist for RL/SL grids
    # Detect by checking if the tile spectral has more rows than a single spline block
    if size(tile.spectral, 1) > b_iDim_tile
        kDim = tile.params.iDim + tile.params.patchOffsetL
        for k in 1:kDim
            p = k * 2
            # Real part
            pp1 = (p - 1) * b_iDim_patch + siL
            tp1 = (p - 1) * b_iDim_tile + 1
            sharedSpectral[pp1:pp1+inner_rows, :] .= tile.spectral[tp1:tp1+inner_rows, :]
            # Imaginary part
            pp1 = p * b_iDim_patch + siL
            tp1 = p * b_iDim_tile + 1
            sharedSpectral[pp1:pp1+inner_rows, :] .= tile.spectral[tp1:tp1+inner_rows, :]
        end
    end
end

"""
    write_tile_to_shared!(sharedSpectral, tile::Union{RZ_Grid, RiRk_Grid}, b_iDim_patch)

RZ and RiRk grids store `b_kDim` consecutive spline blocks (one per vertical
mode); copy the inner rows of every block to its patch position. The spectral
layout is identical for both.
"""
function write_tile_to_shared!(sharedSpectral::SharedArray{Float64}, tile::Union{RZ_Grid, RiRk_Grid},
                                b_iDim_patch::Int64)
    siL = tile.params.spectralIndexL
    b_iDim_tile = tile.params.b_iDim
    b_kDim = tile.params.b_kDim
    inner_rows = b_iDim_tile - 4  # inner region excludes 3-row halo

    for z in 1:b_kDim
        pp1 = (z - 1) * b_iDim_patch + siL
        tp1 = (z - 1) * b_iDim_tile + 1
        sharedSpectral[pp1:pp1+inner_rows, :] .= tile.spectral[tp1:tp1+inner_rows, :]
    end
end

"""
    write_tile_to_shared!(sharedSpectral, tile::Union{RLZ_Grid, RLR_Grid}, b_iDim_patch)

3D cylindrical layout: `b_kDim` vertical-mode blocks of `1 + 2*kDim` radial
sub-blocks (k = 0 spline, then real/imag per azimuthal wavenumber); copy the
inner rows of every sub-block to its patch position. Sub-blocks are aligned by
wavenumber, but an inner tile carries FEWER wavenumbers than the patch (its
`kDim` follows its outermost ring), so the patch's z-level stride is taken from
the shared array itself (`rows = wn_stride_patch * b_kDim`; the vertical is
never decomposed).
"""
function write_tile_to_shared!(sharedSpectral::SharedArray{Float64}, tile::Union{RLZ_Grid, RLR_Grid},
                                b_iDim_patch::Int64)
    siL = tile.params.spectralIndexL
    b_iDim_tile = tile.params.b_iDim
    b_kDim = tile.params.b_kDim
    kDim = tile.params.iDim + tile.params.patchOffsetL
    nblocks = 1 + 2 * kDim
    wn_stride_p = size(sharedSpectral, 1) ÷ b_kDim   # patch z-level stride
    inner_rows = b_iDim_tile - 4  # inner region excludes 3-row halo

    for z_b in 1:b_kDim
        for j in 0:nblocks-1
            pp1 = (z_b - 1) * wn_stride_p + j * b_iDim_patch + siL
            tp1 = ((z_b - 1) * nblocks + j) * b_iDim_tile + 1
            sharedSpectral[pp1:pp1+inner_rows, :] .= tile.spectral[tp1:tp1+inner_rows, :]
        end
    end
end

"""
    load_initial_conditions!(patch, model)

Populate `patch` from `model.initial_conditions`, dispatching on the file
extension:

- `*.jld2` — a restart from a checkpoint. `load_grid` reconstructs the archived
  grid; its spectral coefficients (the prognostic state) are copied verbatim and
  physical space is regenerated with `gridTransform!`, so the loaded grid state
  matches the checkpoint exactly. The archive must be grid-compatible with
  `model.grid_params` (same geometry, spectral/physical sizes, and variable map)
  or an error is raised. Note this is a WARM restart, not a bit-identical
  continuation: the AB3 integrator's tendency history is not part of the grid and
  is not restored (see [`write_restart`](@ref)).
- anything else (CSV) — the physical field is read with `read_physical_grid` and
  `spectralTransform!` fits the spectral coefficients from it.

Runs on the master process before tiling; mutates and returns `patch`.
"""
function load_initial_conditions!(patch::AbstractGrid, model::ModelParameters)
    if endswith(model.initial_conditions, ".jld2")
        loaded = load_grid(model.initial_conditions)
        (typeof(loaded) == typeof(patch) &&
         size(loaded.spectral) == size(patch.spectral) &&
         size(loaded.physical) == size(patch.physical) &&
         loaded.params.vars == patch.params.vars) || error(
            "restart archive $(model.initial_conditions) is not compatible with the " *
            "configured grid_params (geometry/size/vars mismatch): archive is " *
            "$(typeof(loaded)) spectral $(size(loaded.spectral)), grid is " *
            "$(typeof(patch)) spectral $(size(patch.spectral))")
        patch.spectral .= loaded.spectral
        # The wall condition needs rho_t', which only exists after a first pass;
        # take it from the unconditioned transform, then redo the fit with the
        # walls set. Converged: the wall value of rho_t' is insensitive to p's
        # own wall derivative.
        gridTransform!(patch)
        if mc_wall_bc_active(patch)
            update_mc_wall_bc!(patch, patch.physical)
            gridTransform!(patch)
        end
    else
        read_physical_grid(model.initial_conditions, patch)
        spectralTransform!(patch)
        gridTransform!(patch)
        # Re-fit with the wall condition installed. This is the load that used to
        # destroy the balanced vortex (reference/HANDOFF_2026-07-21.md): the damage lands
        # here, on the first projection, before a single timestep is taken. The
        # first pass exists only to produce the fitted z-derivatives the wall value
        # is read from; rho_t' at the wall is insensitive to p's own condition, so
        # one iteration suffices.
        if mc_wall_bc_active(patch)
            update_mc_wall_bc!(patch, patch.physical)
            spectralTransform!(patch)
            gridTransform!(patch)
        end
    end
    return patch
end

"""
    initialize_model(model, workerids)

Set up the distributed model infrastructure by creating the grid patch, distributing
tiles across workers, initializing halo exchange maps, and preparing for time integration.
Returns the initialized patch grid.
"""
function initialize_model(model::ModelParameters, workerids::Vector{Int64})

    num_workers = length(workerids)
    println("Initializing with $(num_workers) workers and tiles")
    patch = createGrid(model.grid_params)
    println("$model")

    # Initialize the patch locally on master process (CSV read or JLD2 restart)
    load_initial_conditions!(patch, model)

    # Transfer the model and patch/tile info to each worker
    println("Initializing workers")
    # Print the tile information — calcTileSizes now returns Vector{SpringsteelGrid}
    # Tiles are assigned positionally along workerids (NOT by worker pid: nested
    # runs pass worker groups whose pids don't start at 2).
    tiles = calcTileSizes(patch, num_workers)
    for (n, w) in enumerate(workerids)
        t = tiles[n]
        println("Worker $w: $(t.params.iDim) gridpoints in $(t.params.num_cells) cells from $(t.params.iMin) to $(t.params.iMax) starting at index $(t.params.spectralIndexL)")
    end

    map(wait, [save_at(w, :model, model) for w in workerids])
    map(wait, [save_at(w, :workerids, workerids) for w in workerids])
    map(wait, [save_at(w, :num_workers, num_workers) for w in workerids])
    # Create patch on each worker for calcPatchMap/calcHaloMap
    map(wait, [save_at(w, :patch, :(createGrid(model.grid_params))) for w in workerids])

    # Send tile parameters and create grids on workers to avoid serializing CHOLMOD factors
    println("Initializing tiles on workers")
    map(wait, [save_at(w, :tile_params, tiles[n].params) for (n, w) in enumerate(workerids)])
    map(wait, [save_at(w, :tile, :(createGrid(tile_params))) for w in workerids])

    # Create the model tiles
    println("Initializing modelTiles on workers")

    # First tile receives a trivial sparse halo map from master to simplify later loops
    firstMap = sparse([1], [1], [1.0], size(patch.spectral, 1), size(patch.spectral, 2))

    # Precalculate indices and allocate buffers for shared and border transfers
    wait(save_at(workerids[1], :mtile, :(createModelTile(patch,tile,model,$(firstMap)))))
    for (w, send_index) in zip(workerids[1:end-1], workerids[2:end])
        sendMap = get_val_from(w, :(mtile.haloSendMap))
        wait(save_at(send_index, :mtile, :(createModelTile(patch,tile,model,$(sendMap)))))
    end

    # Delete tile_params from workers since the relevant info is already in the modelTile
    # Keep patch on all workers — it is needed by the 3-arg splineTransform!
    # Don't delete from the first worker in case they are also the master
    map(wait, [remove_from(w, :tile_params) for w in workerids[2:length(workerids)]])

    println("Ready for time integration!")
    flush(stdout)
    return patch
end

"""
    run_model(patch, model, workerids)

Main time integration loop. Establishes `RemoteChannel` connections between workers,
creates the shared spectral array, and drives the model forward through all timesteps.
"""
function run_model(patch::AbstractGrid, model::ModelParameters, workerids::Vector{Int64})

    num_workers = length(workerids)
    println("Model starting up with $(num_workers) workers and tiles...")

    # Establish RemoteChannel connections between workers
    println("Connecting workers")

    # Master sends to first worker (itself)
    haloInit = RemoteChannel(()->Channel{Array{Float64}}(1),workerids[1])
    wait(save_at(workerids[1], :haloReceive, :($(haloInit))))

    # Each worker passes information up the chain
    for w in workerids[1:length(workerids)-1]
        send_index = w + 1
        wait(save_at(w, :haloSend,
                :(RemoteChannel(()->Channel{Array{Float64}}(1),$(send_index)))))
        receiver = get_val_from(w, :haloSend)
        wait(save_at(send_index, :haloReceive, :($(receiver))))
    end

    # Master receives from the last worker
    wait(save_at(last(workerids), :haloSend,
            :(RemoteChannel(()->Channel{Array{Float64}}(1),workerids[1]))))
    haloReceive = get_val_from(last(workerids), :haloSend)

    # First tile receives an empty halo from master to simplify later loops
    haloInitBuffer = zeros(Float64,1)

    # Last tile is received by master process
    haloReceiveMap = get_val_from(last(workerids), :(mtile.haloSendMap))
    haloReceiveBuffer = zeros(Float64, nnz(haloReceiveMap))

    # Create a shared array for the spectral sum
    sharedSpectral = SharedArray{Float64,2}((size(patch.spectral,1),size(patch.spectral,2)))
    results = Array{Future}(undef,num_workers+1)

    # Initialize at time zero
    sharedSpectral[:] .= patch.spectral[:]
    for w in workerids
        save_at(w, :sharedSpectral, sharedSpectral)
    end
    map(wait, [get_from(w, :(splineTransform!(sharedSpectral, patch, mtile.tile))) for w in workerids])

    # Horizontal semi-implicit sweep setup (mc sets, XZ Phase 1): build the
    # patch-level solve data from a worker's reference profiles, the shared
    # applied-u-increment array, and each tile's patch row window for the
    # increment load (tile mish points are a contiguous subsequence of the
    # patch mish points, so the window is (first_i − 1)·kDim + 1 onward).
    hsi = get(model.options, :horizontal_semiimplicit, false) === true
    hsd = nothing
    hsi_u_incr = nothing
    if hsi
        uses_pressure_reference(model.equation_set) || error(
            "options[:horizontal_semiimplicit] requires a moist_compressible " *
            "(pressure-reference) equation set")
        w1 = workerids[1]
        Pxi_prof = get_val_from(w1, :(mtile.mc_ref_diag.Pxi_prof))
        rho_tprof = get_val_from(w1, :(collect(view(Scythe.ref_rho_t(mtile.ref_state), :, 1))))
        rho_dprof = get_val_from(w1, :(collect(view(Scythe.ref_rho_d(mtile.ref_state), :, 1))))
        etp_prof = get_val_from(w1, :(collect(
            view(Scythe.ref_total_energy(mtile.ref_state), :, 1) .+
            view(Scythe.ref_pressure(mtile.ref_state), :, 1))))
        hsd = create_horizontal_solve_data(patch, model, Pxi_prof,
                                           rho_tprof, rho_dprof, etp_prof)
        hsi_u_incr = SharedArray{Float64,2}((patch.params.iDim * patch.params.kDim, 6))
        xpatch = patch.ibasis.data[1, 1].mishPoints
        kDim = model.grid_params.kDim
        for w in workerids
            x1 = get_val_from(w, :(mtile.tilepoints[1, 1]))
            i1 = argmin(abs.(xpatch .- x1))
            save_at(w, :hsi_rowstart, (i1 - 1) * kDim + 1)
            save_at(w, :hsi_u_incr, hsi_u_incr)
        end
    end

    # Exact (unsplit) 2-D semi-implicit setup (options[:exact_si], exact_si.jl):
    # the patch-level solve data, the shared predictor-publication array
    # (u*, w*, p′* at the patch physical points) and the shared solve feed
    # (δu, ∂x u^{n+1}, δu/Δτ), plus each tile's patch row window.
    xsi = get(model.options, :exact_si, false) === true
    esd = nothing
    xsi_xstar = nothing
    xsi_h = nothing
    if xsi
        w1 = workerids[1]
        Pxi_prof = get_val_from(w1, :(mtile.mc_ref_diag.Pxi_prof))
        rho_t2 = get_val_from(w1, :(collect(view(Scythe.ref_rho_t(mtile.ref_state), :, 1:2))))
        rho_d2 = get_val_from(w1, :(collect(view(Scythe.ref_rho_d(mtile.ref_state), :, 1:2))))
        etp2 = get_val_from(w1, :(collect(
            view(Scythe.ref_total_energy(mtile.ref_state), :, 1:2) .+
            view(Scythe.ref_pressure(mtile.ref_state), :, 1:2))))
        if exact_si_is_rlr(model)
            esd = create_exact_si_data_rlr(patch, model, Pxi_prof, rho_t2, rho_d2, etp2)
            npts = size(patch.physical, 1)
            xsi_xstar = SharedArray{Float64,2}((npts, 4))
            xsi_h = SharedArray{Float64,2}((npts, XSI_RLR_NPLANES))
            for w in workerids
                save_at(w, :xsi_rowstart, 1)
                save_at(w, :xsi_xstar, xsi_xstar)
                save_at(w, :xsi_h, xsi_h)
            end
        else
            esd = create_exact_si_data(patch, model, Pxi_prof, rho_t2, rho_d2, etp2)
            npts = patch.params.iDim * patch.params.kDim
            xsi_xstar = SharedArray{Float64,2}((npts, 3))
            xsi_h = SharedArray{Float64,2}((npts, XSI_NPLANES))
            xpatch = patch.ibasis.data[1, 1].mishPoints
            kDim = model.grid_params.kDim
            for w in workerids
                x1 = get_val_from(w, :(mtile.tilepoints[1, 1]))
                i1 = argmin(abs.(xpatch .- x1))
                save_at(w, :xsi_rowstart, (i1 - 1) * kDim + 1)
                save_at(w, :xsi_xstar, xsi_xstar)
                save_at(w, :xsi_h, xsi_h)
            end
        end
    end

    # Output initial time
    patch.spectral .= sharedSpectral
    gridTransform!(patch)
    @async write_output(patch, model, 0.0)
    flush(stdout)
    # Check for NaNs and quit if found
    checkCFL(patch)

    # Loop through the model timesteps
    @time model_loop(patch, model, workerids, sharedSpectral, haloInit, haloReceive,
        haloInitBuffer, haloReceiveBuffer, haloReceiveMap; hsd, hsi_u_incr,
        esd, xsi_xstar, xsi_h)

    # Integration complete! Finalize the patch
    patch.spectral .= sharedSpectral
    gridTransform!(patch)
    println("Done with time integration")
    return true

end

"""
    model_loop(patch, model, workerids, sharedSpectral, haloInit, haloReceive, haloInitBuffer, haloReceiveBuffer, haloReceiveMap)

Inner time stepping loop that advances all tiles each timestep, performs halo exchanges
via `RemoteChannel`s, accumulates spectral contributions, and writes periodic output.
"""
function model_loop(patch::AbstractGrid, model::ModelParameters, workerids::Vector{Int64},
        sharedSpectral::SharedArray{Float64}, haloInit::RemoteChannel, haloReceive::RemoteChannel,
        haloInitBuffer::Array{Float64}, haloReceiveBuffer::Array{Float64}, haloReceiveMap::SparseMatrixCSC{Float64, Int64};
        hsd=nothing, hsi_u_incr=nothing, esd=nothing, xsi_xstar=nothing, xsi_h=nothing)

    # Set up the timesteps
    num_ts = round(Int,model.integration_time / model.ts)
    output_int = round(Int,model.output_interval / model.ts)
    # JLD2 restart-checkpoint cadence (0 = disabled). Independent of output_int.
    restart_int = model.restart_interval > 0 ? round(Int, model.restart_interval / model.ts) : 0
    # CFL/Courant diagnostic cadence — independent of output; defaults to the
    # output cadence so it is free (reuses the output-step gridTransform!).
    cfl_int = max(1, round(Int, get(model.options, :cfl_interval, model.output_interval) / model.ts))
    println("Integrating $(model.ts) sec increments for $(num_ts) timesteps")

    # Precompute grid spacing and mean sound speed for the Courant diagnostic.
    # Only meaningful for the convective (vertical-velocity) models; gracefully
    # disabled for e.g. 1-D advection, which has no "w". Pxi_bar (domain-mean
    # speed of sound squared) is constant, so fetch it once from a worker.
    cfl_diag_on = haskey(model.grid_params.vars, "w")
    dz_min = dx_min = c_bar = 0.0
    if cfl_diag_on
        dz_min, dx_min = grid_spacing_minima(patch, model)
        c_bar = sqrt(max(0.0, get_val_from(workerids[1], :(Scythe.sound_speed_sq(mtile.ref_state)))))
    end

    # Loop through the timesteps
    for t = 1:num_ts
        println("ts: $(t*model.ts)")

        # Master process clears the shared array and sends an empty halo to the first worker
        @turbo sharedSpectral .= 0.0
        put!(haloInit, haloInitBuffer)

        # Advance each tile (the horizontal-SI variant also hands each worker the
        # shared applied-u-increment array and its tile's patch row window, loaded
        # into hacdot_n at the top of the step as the u-leg AI2* history).
        # The exact-SI variant is two-phase: phase A publishes the (u, w, p′)
        # predictors, the master runs the unsplit patch solve, phase B completes
        # the columns and the end-of-step communication.
        if esd !== nothing
            map(wait, [get_from(w, :(advanceTimestepA(mtile, sharedSpectral, $(t),
                xsi_xstar, xsi_h, xsi_rowstart))) for w in workerids])
            if esd isa ExactSIDataRLR
                exact_si_solve_rlr!(xsi_h, esd, patch, model, t, xsi_xstar)
            else
                exact_si_solve!(xsi_h, esd, patch, model, t, xsi_xstar)
            end
            adv = [get_from(w, :(advanceTimestepB(mtile, sharedSpectral, haloSend,
                haloReceive, $(t), xsi_h, xsi_rowstart))) for w in workerids]
        else
            adv = hsd === nothing ?
                [get_from(w, :(advanceTimestep(mtile, sharedSpectral, haloSend, haloReceive, $(t)))) for w in workerids] :
                [get_from(w, :(advanceTimestep(mtile, sharedSpectral, haloSend, haloReceive, $(t), hsi_u_incr, hsi_rowstart))) for w in workerids]
        end
        map(wait, adv)

        # Get halo from previous tile
        haloReceiveBuffer .= take!(haloReceive)

        # Add it to the sharedArray
        accumulate_at_map!(sharedSpectral, haloReceiveMap, haloReceiveBuffer)

        # Horizontal semi-implicit sweep on the merged patch B coefficients —
        # the communication is already paid; the sweep's applied u increment is
        # published for the workers' next-step history load.
        if hsd !== nothing
            hsi_u_incr .= horizontal_si_correct!(sharedSpectral, patch, model, hsd, t)
        end

        # Reset the shared spectral patch to the tiles
        map(wait, [get_from(w, :(splineTransform!(sharedSpectral, patch, mtile.tile))) for w in workerids])

        # Cheap master-side blow-up trap, every step and independent of the output
        # cadence: a non-finite physical field implies non-finite spectral
        # coefficients here, so this catches an instability without any extra
        # gridTransform! (defense in depth behind the per-worker checkCFL).
        if any(!isfinite, sharedSpectral)
            error("Non-finite spectral coefficient at t=$(round(t*model.ts; digits=3)) s ! CFL condition likely violated")
        end

        # Materialize the physical field once if either the diagnostic or the
        # output cadence fires this step (they coincide by default, costing a
        # single gridTransform!).
        is_cfl_step = cfl_diag_on && mod(t, cfl_int) == 0
        is_output_step = mod(t, output_int) == 0
        is_restart_step = restart_int > 0 && mod(t, restart_int) == 0
        if is_cfl_step || is_output_step || is_restart_step
            patch.spectral .= sharedSpectral
            gridTransform!(patch)
        end

        # CFL/Courant diagnostic on its own cadence (defaults to the output cadence)
        if is_cfl_step
            cfl_diagnostics(patch, model, t, dz_min, dx_min, c_bar)
        end

        # Output if on specified time interval
        if is_output_step
            @async write_output(patch, model, (t*model.ts))
            checkCFL(patch; t=t, ts=model.ts, where="output")
        end

        # Restart checkpoint on its own (typically coarser) cadence. Synchronous,
        # unlike the analysis output: a checkpoint half-written by an @async task
        # racing process exit is useless for restart.
        if is_restart_step
            write_restart(patch, model, (t*model.ts))
        end

        # Done with this timestep
        flush(stdout)
    end
    return nothing
end

"""
    advanceTimestep(mtile, sharedSpectral, haloSend, haloReceive, t)

Advance one tile by one timestep: transform to physical space, advance all columns,
compute spectral tendencies, and exchange halo data with neighboring tiles.
"""
function advanceTimestep(mtile::ModelTile, sharedSpectral::SharedArray{Float64},
        haloSend::RemoteChannel, haloReceive::RemoteChannel, t::Int64)

    # Rigid-wall pressure compatibility condition (R1T1X). The k-direction SA
    # solve happens INSIDE tileTransform!, so the per-column wall derivative has
    # to be installed first. It is read from `tile.physical`, i.e. the PREVIOUS
    # step's fitted state — one step stale by construction, because the fitted
    # z-derivatives it needs do not exist for var_np1 yet. That is deliberate as
    # well as necessary: it keeps the wall condition off the acoustic timescale.
    # On t == 1 load_initial_conditions! has already set the walls.
    #
    # `:wall_bc_tau` [s] is the relaxation timescale. It must stay well above the
    # acoustic step (relax = ts/tau << 1) or the wall condition feeds back on
    # itself and goes unstable — see update_mc_wall_bc!.
    wall_bc = mc_wall_bc_active(mtile.tile)
    if t > 1 && wall_bc
        tau = get(mtile.model.options, :wall_bc_tau, 300.0)::Float64
        relax = min(1.0, mtile.model.ts / tau)
        # tau = Inf (the shipped TC setting) freezes the wall data, so the whole
        # update is a no-op scaled by zero — skip it rather than redo the i-basis
        # fit, which allocates, on every step of every tile.
        relax > 0.0 && update_mc_wall_bc!(mtile.tile, mtile.tile.physical; relax)
    end

    # Transform to local physical tile
    tileTransform!(sharedSpectral, mtile.tile, mtile.tile.physical, mtile.tile.spectral)

    # First step: the TILE is a fresh grid, so its wall_du is still zero even
    # though load_initial_conditions! set the PATCH's — and with :wall_bc_tau =
    # Inf it would stay zero forever, silently degrading R1T1X to plain Neumann
    # and losing the entire balance benefit. Seed it from the just-materialised
    # state (relax = 1: this is an initialization, not a tracking update) and
    # redo the transform. Costs one extra tileTransform! on step 1 only.
    if t == 1 && wall_bc
        update_mc_wall_bc!(mtile.tile, mtile.tile.physical; relax = 1.0)
        tileTransform!(sharedSpectral, mtile.tile, mtile.tile.physical, mtile.tile.spectral)
    end

    # Trap a numerical blow-up early: scan the freshly transformed physical state
    # for non-finite values *before* feeding it into the NaN-blind spline solve in
    # advance_column. This runs single-threaded (before the @threads loop) so it is
    # thread-safe, and halts the run cleanly instead of segfaulting in the solver.
    checkCFL(mtile.tile; t=t, ts=mtile.model.ts, where="worker tile")
    state_minima_trace(mtile, t)

    # Advance each column.
    #
    # `:static` (not the default `:dynamic`) because the equation sets take their vertical work
    # column from `mtile.scratch_columns[threadid(), var]`. Under dynamic scheduling a task may
    # resume on a different thread, so `threadid()` is not a stable owner tag and two columns
    # could end up sharing one work column. `:static` pins each iteration to a fixed thread.
    # The columns are equal-cost, so there is nothing for dynamic scheduling to balance anyway.
    if num_columns(mtile.tile) > 0
        Threads.@threads :static for c in 1:num_columns(mtile.tile)
            advance_column(mtile, c, t)
        end
    else
        advance_column(mtile, -1, t)
    end

    # Convert current timestep to spectral tendencies
    calcTendency(mtile)

    # Send halo to next tile (extract border spectral values in tile-local coordinates)
    put!(haloSend, extract_halo_values(mtile.tile))

    # Set the sharedArray this tile is responsible for (tile→patch index mapping)
    write_tile_to_shared!(sharedSpectral, mtile.tile, mtile.patch_b_iDim)

    # Get halo from previous tile
    mtile.haloReceiveBuffer .= take!(haloReceive)

    # Add it to the sharedArray
    accumulate_at_map!(sharedSpectral, mtile.haloReceiveMap, mtile.haloReceiveBuffer)

    return nothing
end

# Horizontal-SI variant: load this tile's window of the previous step's applied
# u increment (published by the patch-level sweep in `model_loop`) into the
# hacdot_n u column before the columns advance — the u-leg AI2* history.
function advanceTimestep(mtile::ModelTile, sharedSpectral::SharedArray{Float64},
        haloSend::RemoteChannel, haloReceive::RemoteChannel, t::Int64,
        hsi_u_incr::AbstractMatrix{Float64}, hsi_rowstart::Int64)
    if t > 1
        horizontal_si_load_increment!(mtile, hsi_u_incr, t, hsi_rowstart)
    end
    return advanceTimestep(mtile, sharedSpectral, haloSend, haloReceive, t)
end

"""
    advance_column(mtile, c, t)

Advance a single column `c` by dispatching to the configured physical model equation set.
A column index of -1 indicates an R or RL grid where all points are treated as one column.
"""
function advance_column(mtile::ModelTile, c::Int64, t::Int64)

    # Set column index range
    if c == -1
        # R or RL grid: all points are treated as one column
        colstart = 1
        colend = size(mtile.tile.physical,1)
    else
        # RZ or RLZ grid: use the vertical dimension to stride columns
        gp = mtile.model.grid_params
        vdim = gp.kDim
        colstart = (c-1) * vdim + 1
        colend = colstart + vdim - 1
    end

    # Feed physical matrices to physical equations
    physical_model(mtile, colstart, colend, t)

end

"""
    finalize_model(grid, model)

Write final model output at the end of the integration period.
"""
function finalize_model(grid::AbstractGrid, model::ModelParameters)

    write_output(grid, model, model.integration_time)
    # Guaranteed final restart checkpoint (covers both single-grid and nested,
    # which route their finalize through here). Skipped when checkpoints are off.
    if model.restart_interval > 0
        write_restart(grid, model, model.integration_time)
    end
    println("Model complete!")
end

"""
    physical_model(mtile, colstart, colend, t)

Dispatch to the appropriate equation set by looking up the function named
by `mtile.model.equation_set` in the `Scythe` module and calling it on the column range.
"""
function physical_model(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    equation_set = Symbol(mtile.model.equation_set)
    equation_call = getfield(Scythe, equation_set)
    equation_call(mtile, colstart, colend, t)
    return
end

"""
    semiimplicit_timestep_old(mtile, colstart, colend, t)

Deprecated semi-implicit timestep variant that solves a Helmholtz equation for xi
using a direct matrix solve each timestep. Superseded by [`semiimplicit_timestep`](@ref).
"""
function semiimplicit_timestep_old(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    w_index = mtile.model.grid_params.vars["w"]
    xi_index = mtile.model.grid_params.vars["xi"]
    ts = mtile.model.ts

    # Calculate xi_nstar
    xi_nstar = mtile.var_np1[colstart:colend,xi_index]
    wdot_n = view(mtile.impdot_n,colstart:colend,xi_index)
    wdot_nm1 = view(mtile.impdot_nm1,colstart:colend,xi_index)
    wdot_nm2 = view(mtile.impdot_nm2,colstart:colend,xi_index)

    # Calculate w_nstar
    w_nstar = mtile.var_np1[colstart:colend,w_index]
    xidot_n = view(mtile.impdot_n,colstart:colend,w_index)
    xidot_nm1 = view(mtile.impdot_nm1,colstart:colend,w_index)
    xidot_nm2 = view(mtile.impdot_nm2,colstart:colend,w_index)

    # Get the mean speed of sound squared from the reference state
    Pxi_bar = sound_speed_sq(mtile.ref_state)

    # Add the implicit terms
    ts_term = 0.0
    if (t == 1)
        # Use trapezoidal method (AM2) for first step
        ts_term = 0.5 * ts
        w_nstar .= w_nstar .+ (ts .* 0.5 .* xidot_n)
        xi_nstar .= xi_nstar .+ (ts .* 0.5 .* wdot_n)
    else
        # Use AI2* for second step and beyond
        ts_term = 1.25 * ts
        w_nstar .= w_nstar .- (ts .* xidot_n) .+ (ts .* 0.75 .* xidot_nm1)
        xi_nstar .= xi_nstar .- (ts .* wdot_n) .+ (ts .* 0.75 .* wdot_nm1)
    end

    # Set the n-1 and n-2 terms
    xidot_nm2 .= xidot_nm1
    xidot_nm1 .= xidot_n

    wdot_nm2 .= wdot_nm1
    wdot_nm1 .= wdot_n

    # Take the vertical derivative of w_nstar and multiply by ts term
    w_col = mtile.tile.kbasis.data[mtile.model.grid_params.vars["w"]]
    w_col.uMish .= w_nstar
    Btransform!(w_col)
    Atransform!(w_col)
    w_nstar = Itransform!(w_col)
    w_nstar_z = ts_term .* Ixtransform(w_col)

    # Calculate the Helmholtz matrix
    h_a = calc_Helmholtz_semiimplicit_matrix_xi(mtile.tile, mtile.model, Pxi_bar, ts_term)

    # Solve for the xi coefficients (RHS = xi_nstar - w_nstar_z; homogeneous BCs)
    xi_col = mtile.tile.kbasis.data[mtile.model.grid_params.vars["xi"]]
    _vertical_solve!(xi_col, h_a, xi_nstar .- w_nstar_z, mtile)
    view(mtile.var_np1,colstart:colend,xi_index) .= Itransform!(xi_col)

    # Set w_n+1
    view(mtile.var_np1,colstart:colend,w_index) .= w_nstar .- (ts_term .* Pxi_bar .* Ixtransform(xi_col))
end

"""
    semiimplicit_adjustment_xi(mtile, colstart, colend, t)

Semi-implicit adjustment that solves a spectral-collocation Helmholtz problem for xi,
using AB3 explicit extrapolation and AI2* implicit treatment of acoustic modes.
"""
function semiimplicit_adjustment_xi(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    w_index = mtile.model.grid_params.vars["w"]
    xi_index = mtile.model.grid_params.vars["xi"]
    ts = mtile.model.ts

    # Calculate xi_nstar
    xi_nstar = mtile.var_np1[colstart:colend,xi_index]
    wdot_n = view(mtile.impdot_n,colstart:colend,xi_index)
    wdot_nm1 = view(mtile.impdot_nm1,colstart:colend,xi_index)
    wdot_nm2 = view(mtile.impdot_nm2,colstart:colend,xi_index)

    # Calculate w_nstar
    w_nstar = mtile.var_np1[colstart:colend,w_index]
    xidot_n = view(mtile.impdot_n,colstart:colend,w_index)
    xidot_nm1 = view(mtile.impdot_nm1,colstart:colend,w_index)
    xidot_nm2 = view(mtile.impdot_nm2,colstart:colend,w_index)

    # Get the mean speed of sound squared
    Pxi_bar = sound_speed_sq(mtile.ref_state)

    # Subtract the explicit terms and add the implicit terms
    ts_term = 0.0
    if (t == 1)
        # Use trapezoidal method (AM2) for first step
        ts_term = 0.5 * ts
        w_nstar .= @. w_nstar - (ts * xidot_n) + (ts * 0.5 * xidot_n)
        xi_nstar .= @. xi_nstar - (ts * wdot_n) + (ts * 0.5 * wdot_n)
    elseif (t == 2)
        # Use AI2* for second step and beyond
        ts_term = 1.25 * ts
        w_nstar .= @. w_nstar - (0.5 * ts) * ((3.0 * xidot_n) - xidot_nm1) - (ts * xidot_n) + (ts * 0.75 * xidot_nm1)
        xi_nstar .= @. xi_nstar - (0.5 * ts) * ((3.0 * wdot_n) - wdot_nm1) - (ts * wdot_n) + (ts * 0.75 * wdot_nm1)
    else
        # Use AI2* for second step and beyond
        ts_term = 1.25 * ts
        w_nstar .= @. w_nstar - ((ts / 12.0) * ((23.0 * xidot_n) - (16.0 * xidot_nm1) + (5.0 * xidot_nm2))) - (ts * xidot_n) + (ts * 0.75 * xidot_nm1)
        xi_nstar .= @. xi_nstar - ((ts / 12.0) * ((23.0 * wdot_n) - (16.0 * wdot_nm1) + (5.0 * wdot_nm2))) - (ts * wdot_n) + (ts * 0.75 * wdot_nm1)
    end

    # Set the n-1 and n-2 terms
    xidot_nm2 .= xidot_nm1
    xidot_nm1 .= xidot_n

    wdot_nm2 .= wdot_nm1
    wdot_nm1 .= wdot_n

    # Take the vertical derivative of w_nstar and multiply by ts term
    w_col = deepcopy(mtile.tile.kbasis.data[mtile.model.grid_params.vars["w"]])
    w_col.uMish .= w_nstar
    Btransform!(w_col)
    Atransform!(w_col)
    w_nstar = Itransform!(w_col)
    w_nstar_z = ts_term .* Ixtransform(w_col)

    # Set up the matrix problem (RHS = xi_nstar - w_nstar_z; homogeneous BCs)
    rhs = xi_nstar .- w_nstar_z
    xi_col = deepcopy(mtile.tile.kbasis.data[mtile.model.grid_params.vars["xi"]])
    # Solve for the coefficients
    if t == 1
        # Calculate the Helmholtz matrix for the first time step
        h_a = calc_Helmholtz_semiimplicit_matrix(mtile.tile, mtile.model, Pxi_bar, ts_term)
        _vertical_solve!(xi_col, h_a, rhs, mtile)
    else
        # Use the pre-calculated one
        _vertical_solve!(xi_col, mtile.h_matrix, rhs, mtile)
    end

    # Set xi_n+1
    view(mtile.var_np1,colstart:colend,xi_index) .= Itransform!(xi_col)

    # Set w_n+1
    view(mtile.var_np1,colstart:colend,w_index) .= w_nstar .- (ts_term .* Pxi_bar .* Ixtransform(xi_col))
end

"""
    semiimplicit_adjustment(mtile, colstart, colend, t)

Semi-implicit adjustment that solves a spectral-collocation Helmholtz problem for w,
using AB3 explicit extrapolation and AI2* implicit treatment of acoustic modes.
"""
function semiimplicit_adjustment(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    w_index = mtile.model.grid_params.vars["w"]
    xi_index = mtile.model.grid_params.vars["xi"]
    ts = mtile.model.ts

    # Calculate xi_nstar
    xi_nstar = mtile.var_np1[colstart:colend,xi_index]
    wdot_n = view(mtile.impdot_n,colstart:colend,xi_index)
    wdot_nm1 = view(mtile.impdot_nm1,colstart:colend,xi_index)
    wdot_nm2 = view(mtile.impdot_nm2,colstart:colend,xi_index)

    # Calculate w_nstar
    w_nstar = mtile.var_np1[colstart:colend,w_index]
    xidot_n = view(mtile.impdot_n,colstart:colend,w_index)
    xidot_nm1 = view(mtile.impdot_nm1,colstart:colend,w_index)
    xidot_nm2 = view(mtile.impdot_nm2,colstart:colend,w_index)

    # Get the mean speed of sound squared
    Pxi_bar = sound_speed_sq(mtile.ref_state)

    # Subtract the explicit terms and add the implicit terms
    ts_term = 0.0
    if (t == 1)
        # Use trapezoidal method (AM2) for first step
        ts_term = 0.5 * ts
        w_nstar .= @. w_nstar - (ts * xidot_n) + (ts * 0.5 * xidot_n)
        xi_nstar .= @. xi_nstar - (ts * wdot_n) + (ts * 0.5 * wdot_n)
    elseif (t == 2)
        # Use AI2* for second step and beyond
        ts_term = 1.25 * ts
        w_nstar .= @. w_nstar - (0.5 * ts) * ((3.0 * xidot_n) - xidot_nm1) - (ts * xidot_n) + (ts * 0.75 * xidot_nm1)
        xi_nstar .= @. xi_nstar - (0.5 * ts) * ((3.0 * wdot_n) - wdot_nm1) - (ts * wdot_n) + (ts * 0.75 * wdot_nm1)
    else
        # Use AI2* for second step and beyond
        ts_term = 1.25 * ts
        w_nstar .= @. w_nstar - ((ts / 12.0) * ((23.0 * xidot_n) - (16.0 * xidot_nm1) + (5.0 * xidot_nm2))) - (ts * xidot_n) + (ts * 0.75 * xidot_nm1)
        xi_nstar .= @. xi_nstar - ((ts / 12.0) * ((23.0 * wdot_n) - (16.0 * wdot_nm1) + (5.0 * wdot_nm2))) - (ts * wdot_n) + (ts * 0.75 * wdot_nm1)
    end

    # Set the n-1 and n-2 terms
    xidot_nm2 .= xidot_nm1
    xidot_nm1 .= xidot_n

    wdot_nm2 .= wdot_nm1
    wdot_nm1 .= wdot_n

    # Take the vertical derivative of xi_nstar and multiply by ts term
    xi_col = deepcopy(mtile.tile.kbasis.data[mtile.model.grid_params.vars["xi"]])
    xi_col.uMish .= xi_nstar
    Btransform!(xi_col)
    Atransform!(xi_col)
    xi_nstar = Itransform!(xi_col)
    xi_nstar_z = ts_term .* Pxi_bar .* Ixtransform(xi_col)

    # Set up the matrix problem (RHS = xi_nstar_z - w_nstar; homogeneous BCs)
    rhs = xi_nstar_z .- w_nstar
    w_col = deepcopy(mtile.tile.kbasis.data[mtile.model.grid_params.vars["w"]])
    # Solve for the coefficients
    if t == 1
        # Calculate the Helmholtz matrix for the first time step
        h_a = calc_Helmholtz_semiimplicit_matrix(mtile.tile, mtile.model, Pxi_bar, ts_term)
        _vertical_solve!(w_col, h_a, rhs, mtile)
    else
        # Use the pre-calculated one
        _vertical_solve!(w_col, mtile.h_matrix, rhs, mtile)
    end

    # Set w_n+1
    view(mtile.var_np1,colstart:colend,w_index) .= Itransform!(w_col)

    # Set xi_n+1
    view(mtile.var_np1,colstart:colend,xi_index) .= xi_nstar .- (ts_term .* Ixtransform(w_col))
end

"""
    semiimplicit_adjustment_rhod(mtile, colstart, colend, t)

Semi-implicit acoustic adjustment for the linear dry-air density set. Solves the
constant-coefficient Helmholtz problem for the vertical mass flux `φ = ρ̂_d w`, which
reuses the xi-form w-solve matrix because `ρ̂_d c̄_ρ = Pxi_bar` is constant, then
recovers `w = φ/ρ̂_d` and `ρ_d' = ρ_d'* - Δτ ∂_z φ` (flux form, conserves `∫ρ_d'`).
See `reference/Semiimplicit_linear_rhod.tex`.
"""
function semiimplicit_adjustment_rhod(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    w_index = mtile.model.grid_params.vars["w"]
    rhod_index = mtile.model.grid_params.vars["rho_d"]
    ts = mtile.model.ts

    # rho_d' predictor and its implicit continuity tendency -∂_z(ρ̂_d w)
    rho_dp_nstar = mtile.var_np1[colstart:colend,rhod_index]
    cdot_n = view(mtile.impdot_n,colstart:colend,rhod_index)
    cdot_nm1 = view(mtile.impdot_nm1,colstart:colend,rhod_index)
    cdot_nm2 = view(mtile.impdot_nm2,colstart:colend,rhod_index)

    # w predictor and its implicit momentum tendency -c̄_ρ ∂_z ρ_d'
    w_nstar = mtile.var_np1[colstart:colend,w_index]
    wdot_n = view(mtile.impdot_n,colstart:colend,w_index)
    wdot_nm1 = view(mtile.impdot_nm1,colstart:colend,w_index)
    wdot_nm2 = view(mtile.impdot_nm2,colstart:colend,w_index)

    # Mean speed of sound squared and the dry-air reference density ρ̂_d
    Pxi_bar = sound_speed_sq(mtile.ref_state)
    rho_dbar = ref_rho_d(mtile.ref_state)[:,1]

    # Subtract the explicit terms and add the implicit terms (AI2*)
    ts_term = 0.0
    if (t == 1)
        # Use trapezoidal method (AM2) for first step
        ts_term = 0.5 * ts
        w_nstar .= @. w_nstar - (ts * wdot_n) + (ts * 0.5 * wdot_n)
        rho_dp_nstar .= @. rho_dp_nstar - (ts * cdot_n) + (ts * 0.5 * cdot_n)
    elseif (t == 2)
        # Use AI2* for second step and beyond
        ts_term = 1.25 * ts
        w_nstar .= @. w_nstar - (0.5 * ts) * ((3.0 * wdot_n) - wdot_nm1) - (ts * wdot_n) + (ts * 0.75 * wdot_nm1)
        rho_dp_nstar .= @. rho_dp_nstar - (0.5 * ts) * ((3.0 * cdot_n) - cdot_nm1) - (ts * cdot_n) + (ts * 0.75 * cdot_nm1)
    else
        # Use AI2* for second step and beyond
        ts_term = 1.25 * ts
        w_nstar .= @. w_nstar - ((ts / 12.0) * ((23.0 * wdot_n) - (16.0 * wdot_nm1) + (5.0 * wdot_nm2))) - (ts * wdot_n) + (ts * 0.75 * wdot_nm1)
        rho_dp_nstar .= @. rho_dp_nstar - ((ts / 12.0) * ((23.0 * cdot_n) - (16.0 * cdot_nm1) + (5.0 * cdot_nm2))) - (ts * cdot_n) + (ts * 0.75 * cdot_nm1)
    end

    # Set the n-1 and n-2 terms
    wdot_nm2 .= wdot_nm1
    wdot_nm1 .= wdot_n

    cdot_nm2 .= cdot_nm1
    cdot_nm1 .= cdot_n

    # Take the vertical derivative of the rho_d' predictor and scale by ts_term * Pxi_bar
    rho_dp_col = deepcopy(mtile.tile.kbasis.data[rhod_index])
    rho_dp_col.uMish .= rho_dp_nstar
    Btransform!(rho_dp_col)
    Atransform!(rho_dp_col)
    rho_dp_nstar = Itransform!(rho_dp_col)
    rho_dp_nstar_z = ts_term .* Pxi_bar .* Ixtransform(rho_dp_col)

    # Mass-flux Helmholtz RHS (φ = ρ̂_d w): rhs = Δτ Pxi_bar ∂_z ρ_d'* - ρ̂_d w*.
    # The operator (I - Δτ² Pxi_bar ∂_zz) is the same w-form matrix, so reuse h_matrix.
    rhs = rho_dp_nstar_z .- (rho_dbar .* w_nstar)
    phi_col = deepcopy(mtile.tile.kbasis.data[w_index])
    # Solve for the coefficients
    if t == 1
        # Calculate the Helmholtz matrix for the first time step
        h_a = calc_Helmholtz_semiimplicit_matrix(mtile.tile, mtile.model, Pxi_bar, ts_term)
        _vertical_solve!(phi_col, h_a, rhs, mtile)
    else
        # Use the pre-calculated one
        _vertical_solve!(phi_col, mtile.h_matrix, rhs, mtile)
    end

    # Recover w_n+1 = φ_n+1 / ρ̂_d
    view(mtile.var_np1,colstart:colend,w_index) .= Itransform!(phi_col) ./ rho_dbar

    # Recover rho_d'_n+1 = rho_d'* - Δτ ∂_z φ_n+1 (flux form ⇒ conserves ∫ρ_d')
    view(mtile.var_np1,colstart:colend,rhod_index) .= rho_dp_nstar .- (ts_term .* Ixtransform(phi_col))
end

"""
    semiimplicit_timestep(mtile, colstart, colend, t)

Combined explicit-implicit split timestep for acoustic modes. Applies AI2*-AB3
time integration with a Helmholtz solve for w to handle the implicit part.
"""
function semiimplicit_timestep(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    w_index = mtile.model.grid_params.vars["w"]
    xi_index = mtile.model.grid_params.vars["xi"]
    ts = mtile.model.ts

    # Calculate xi_nstar
    xi_nstar = mtile.var_np1[colstart:colend,xi_index]
    wdot_n = view(mtile.impdot_n,colstart:colend,xi_index)
    wdot_nm1 = view(mtile.impdot_nm1,colstart:colend,xi_index)
    wdot_nm2 = view(mtile.impdot_nm2,colstart:colend,xi_index)

    # Calculate w_nstar
    w_nstar = mtile.var_np1[colstart:colend,w_index]
    xidot_n = view(mtile.impdot_n,colstart:colend,w_index)
    xidot_nm1 = view(mtile.impdot_nm1,colstart:colend,w_index)
    xidot_nm2 = view(mtile.impdot_nm2,colstart:colend,w_index)

    # Get the mean speed of sound squared
    Pxi_bar = sound_speed_sq(mtile.ref_state)

    # Subtract the explicit terms and add the implicit terms
    ts_term = 0.0
    if (t == 1)
        # Use trapezoidal method (AM2) for first step
        ts_term = 0.5 * ts
        w_nstar .= @. w_nstar + (ts * 0.5 * xidot_n)
        xi_nstar .= @. xi_nstar + (ts * 0.5 * wdot_n)
    else
        # Use AI2* for second step and beyond
        ts_term = 1.25 * ts
        w_nstar .= @. w_nstar - (ts * xidot_n) + (ts * 0.75 * xidot_nm1)
        xi_nstar .= @. xi_nstar - (ts * wdot_n) + (ts * 0.75 * wdot_nm1)
    end

    # Set the n-1 and n-2 terms
    xidot_nm2 .= xidot_nm1
    xidot_nm1 .= xidot_n

    wdot_nm2 .= wdot_nm1
    wdot_nm1 .= wdot_n

    # Take the vertical derivative of xi_nstar and multiply by ts term
    xi_col = deepcopy(mtile.tile.kbasis.data[mtile.model.grid_params.vars["xi"]])
    xi_col.uMish .= xi_nstar
    Btransform!(xi_col)
    Atransform!(xi_col)
    xi_nstar = Itransform!(xi_col)
    xi_nstar_z = ts_term .* Pxi_bar .* Ixtransform(xi_col)

    # Set up the matrix problem (RHS = xi_nstar_z - w_nstar; homogeneous BCs)
    rhs = xi_nstar_z .- w_nstar
    w_col = deepcopy(mtile.tile.kbasis.data[mtile.model.grid_params.vars["w"]])
    # Solve for the coefficients
    if t == 1
        # Calculate the Helmholtz matrix for the first time step
        h_a = calc_Helmholtz_semiimplicit_matrix(mtile.tile, mtile.model, Pxi_bar, ts_term)
        _vertical_solve!(w_col, h_a, rhs, mtile)
    else
        # Use the pre-calculated one
        _vertical_solve!(w_col, mtile.h_matrix, rhs, mtile)
    end

    # Set w_n+1
    view(mtile.var_np1,colstart:colend,w_index) .= Itransform!(w_col)

    # Set xi_n+1
    view(mtile.var_np1,colstart:colend,xi_index) .= xi_nstar .- (ts_term .* Ixtransform(w_col))
end

"""
    diffusion_timestep(mtile, colstart, colend, t)

Implicit vertical diffusion timestep for thermodynamic and moisture variables (s, mu,
mu_c, mu_r, mu_sat) using a pre-factored Helmholtz matrix and AI2* time integration.
"""
function diffusion_timestep(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    s_index = mtile.model.grid_params.vars["s"]
    s = view(mtile.var_np1,colstart:colend,s_index)

    mu_index = mtile.model.grid_params.vars["mu"]
    mu = view(mtile.var_np1,colstart:colend,mu_index)

    mu_c_index = mtile.model.grid_params.vars["mu_c"]
    mu_c = view(mtile.var_np1,colstart:colend,mu_c_index)

    mu_r_index = mtile.model.grid_params.vars["mu_r"]
    mu_r = view(mtile.var_np1,colstart:colend,mu_r_index)

    mu_sat_index = mtile.model.grid_params.vars["mu_sat"]
    mu_sat = view(mtile.var_np1,colstart:colend,mu_sat_index)

    ts = mtile.model.ts

    # Calculate s_nstar
    s_nstar = mtile.var_np1[colstart:colend,s_index]
    sdot_n = view(mtile.impdot_n,colstart:colend,s_index)
    sdot_nm1 = view(mtile.impdot_nm1,colstart:colend,s_index)
    sdot_nm2 = view(mtile.impdot_nm2,colstart:colend,s_index)

    # Calculate mu_nstar
    mu_nstar = mtile.var_np1[colstart:colend,mu_index]
    mudot_n = view(mtile.impdot_n,colstart:colend,mu_index)
    mudot_nm1 = view(mtile.impdot_nm1,colstart:colend,mu_index)
    mudot_nm2 = view(mtile.impdot_nm2,colstart:colend,mu_index)

    # Calculate mu_c_nstar
    mu_c_nstar = mtile.var_np1[colstart:colend,mu_c_index]
    mu_cdot_n = view(mtile.impdot_n,colstart:colend,mu_c_index)
    mu_cdot_nm1 = view(mtile.impdot_nm1,colstart:colend,mu_c_index)
    mu_cdot_nm2 = view(mtile.impdot_nm2,colstart:colend,mu_c_index)

    # Calculate mu_r_nstar
    mu_r_nstar = mtile.var_np1[colstart:colend,mu_r_index]
    mu_rdot_n = view(mtile.impdot_n,colstart:colend,mu_r_index)
    mu_rdot_nm1 = view(mtile.impdot_nm1,colstart:colend,mu_r_index)
    mu_rdot_nm2 = view(mtile.impdot_nm2,colstart:colend,mu_r_index)

    # Calculate mu_sat_nstar
    mu_sat_nstar = mtile.var_np1[colstart:colend,mu_sat_index]
    mu_sat_dot_n = view(mtile.impdot_n,colstart:colend,mu_sat_index)
    mu_sat_dot_nm1 = view(mtile.impdot_nm1,colstart:colend,mu_sat_index)
    mu_sat_dot_nm2 = view(mtile.impdot_nm2,colstart:colend,mu_sat_index)

    # Add the implicit terms
    ts_term = 0.0
    if (t == 1)
        # Use trapezoidal method (AM2) for first step
        ts_term = 0.5 * ts * mtile.model.physical_params[:Kvdiff]
        s_nstar .= @. s_nstar + (ts * 0.5 * sdot_n)
        mu_nstar .= @. mu_nstar + (ts * 0.5 * mudot_n)
        mu_c_nstar .= @. mu_c_nstar + (ts * 0.5 * mu_cdot_n)
        mu_r_nstar .= @. mu_r_nstar + (ts * 0.5 * mu_rdot_n)
        mu_sat_nstar .= @. mu_sat_nstar + (ts * 0.5 * mu_sat_dot_n)
    else
        # Use AI2* for second step and beyond
        ts_term = 1.25 * ts * mtile.model.physical_params[:Kvdiff]
        s_nstar .= @. s_nstar - (ts * sdot_n) + (ts * 0.75 * sdot_nm1)
        mu_nstar .= @. mu_nstar - (ts * mudot_n) + (ts * 0.75 * mudot_nm1)
        mu_c_nstar .= @. mu_c_nstar - (ts * mu_cdot_n) + (ts * 0.75 * mu_cdot_nm1)
        mu_r_nstar .= @. mu_r_nstar - (ts * mu_rdot_n) + (ts * 0.75 * mu_rdot_nm1)
        mu_sat_nstar .= @. mu_sat_nstar - (ts * mu_sat_dot_n) + (ts * 0.75 * mu_sat_dot_nm1)
    end

    # Set the n-1 and n-2 terms
    sdot_nm2 .= sdot_nm1
    sdot_nm1 .= sdot_n
    mudot_nm2 .= mudot_nm1
    mudot_nm1 .= mudot_n
    mu_cdot_nm2 .= mu_cdot_nm1
    mu_cdot_nm1 .= mu_cdot_n
    mu_rdot_nm2 .= mu_rdot_nm1
    mu_rdot_nm1 .= mu_rdot_n
    mu_sat_dot_nm2 .= mu_sat_dot_nm1
    mu_sat_dot_nm1 .= mu_sat_dot_n

    # Set up the matrix problem
    nz = mtile.model.grid_params.kDim
    col = deepcopy(mtile.tile.kbasis.data[mtile.model.grid_params.vars["s"]])

    # Solve for the coefficients
    h_a = mtile.diffusion_matrix
    if t == 1
        # Calculate the Helmholtz matrix for the first time step
        h_a = calc_Helmholtz_diffusion_matrix(mtile.tile, mtile.model, ts_term)
    end

    # Set s_n+1 (homogeneous BCs)
    _vertical_solve!(col, h_a, s_nstar, mtile)
    view(mtile.var_np1,colstart:colend,s_index) .= Itransform!(col)

    # Set mu_n+1
    _vertical_solve!(col, h_a, mu_nstar, mtile)
    view(mtile.var_np1,colstart:colend,mu_index) .= Itransform!(col)

    # Set mu_c_n+1
    _vertical_solve!(col, h_a, mu_c_nstar, mtile)
    view(mtile.var_np1,colstart:colend,mu_c_index) .= Itransform!(col)

    # Set mu_r_n+1
    _vertical_solve!(col, h_a, mu_r_nstar, mtile)
    view(mtile.var_np1,colstart:colend,mu_r_index) .= Itransform!(col)

    # Set mu_sat_n+1
    _vertical_solve!(col, h_a, mu_sat_nstar, mtile)
    view(mtile.var_np1,colstart:colend,mu_sat_index) .= Itransform!(col)

end

"""
    diffusion_timestep_pd(mtile, colstart, colend, t)

Partial-density variant of [`diffusion_timestep`](@ref) for the
`primitive_equation_XZ_rhod_pd` set: the diffused thermodynamic/moisture variables are
`s`, the moisture partial densities `rho_v`, `rho_c`, `rho_r`, and the transformed
saturation ratio `mu_sat` (replacing `mu`, `mu_c`, `mu_r`). Identical AI2* implicit
vertical-diffusion solve; only the prognostic slot names differ.
"""
function diffusion_timestep_pd(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    # Slot 1 is the entropy variable: intensive "s" (rhod_pd set) or the entropy density
    # "sigma" (primitive_equation_XZ_sigma set). The vertical-diffusion solve is identical for
    # either, so resolve whichever name this model carries.
    vars = mtile.model.grid_params.vars
    s_index = haskey(vars, "sigma") ? vars["sigma"] : vars["s"]
    rho_v_index = mtile.model.grid_params.vars["rho_v"]
    rho_c_index = mtile.model.grid_params.vars["rho_c"]
    rho_r_index = mtile.model.grid_params.vars["rho_r"]
    mu_sat_index = mtile.model.grid_params.vars["mu_sat"]

    ts = mtile.model.ts

    # Predictor states and their stored implicit (vertical-diffusion) tendencies
    s_nstar = mtile.var_np1[colstart:colend,s_index]
    sdot_n = view(mtile.impdot_n,colstart:colend,s_index)
    sdot_nm1 = view(mtile.impdot_nm1,colstart:colend,s_index)
    sdot_nm2 = view(mtile.impdot_nm2,colstart:colend,s_index)

    rho_v_nstar = mtile.var_np1[colstart:colend,rho_v_index]
    rho_vdot_n = view(mtile.impdot_n,colstart:colend,rho_v_index)
    rho_vdot_nm1 = view(mtile.impdot_nm1,colstart:colend,rho_v_index)
    rho_vdot_nm2 = view(mtile.impdot_nm2,colstart:colend,rho_v_index)

    rho_c_nstar = mtile.var_np1[colstart:colend,rho_c_index]
    rho_cdot_n = view(mtile.impdot_n,colstart:colend,rho_c_index)
    rho_cdot_nm1 = view(mtile.impdot_nm1,colstart:colend,rho_c_index)
    rho_cdot_nm2 = view(mtile.impdot_nm2,colstart:colend,rho_c_index)

    rho_r_nstar = mtile.var_np1[colstart:colend,rho_r_index]
    rho_rdot_n = view(mtile.impdot_n,colstart:colend,rho_r_index)
    rho_rdot_nm1 = view(mtile.impdot_nm1,colstart:colend,rho_r_index)
    rho_rdot_nm2 = view(mtile.impdot_nm2,colstart:colend,rho_r_index)

    mu_sat_nstar = mtile.var_np1[colstart:colend,mu_sat_index]
    mu_sat_dot_n = view(mtile.impdot_n,colstart:colend,mu_sat_index)
    mu_sat_dot_nm1 = view(mtile.impdot_nm1,colstart:colend,mu_sat_index)
    mu_sat_dot_nm2 = view(mtile.impdot_nm2,colstart:colend,mu_sat_index)

    # Add the implicit terms
    ts_term = 0.0
    if (t == 1)
        # Use trapezoidal method (AM2) for first step
        ts_term = 0.5 * ts * mtile.model.physical_params[:Kvdiff]
        s_nstar .= @. s_nstar + (ts * 0.5 * sdot_n)
        rho_v_nstar .= @. rho_v_nstar + (ts * 0.5 * rho_vdot_n)
        rho_c_nstar .= @. rho_c_nstar + (ts * 0.5 * rho_cdot_n)
        rho_r_nstar .= @. rho_r_nstar + (ts * 0.5 * rho_rdot_n)
        mu_sat_nstar .= @. mu_sat_nstar + (ts * 0.5 * mu_sat_dot_n)
    else
        # Use AI2* for second step and beyond
        ts_term = 1.25 * ts * mtile.model.physical_params[:Kvdiff]
        s_nstar .= @. s_nstar - (ts * sdot_n) + (ts * 0.75 * sdot_nm1)
        rho_v_nstar .= @. rho_v_nstar - (ts * rho_vdot_n) + (ts * 0.75 * rho_vdot_nm1)
        rho_c_nstar .= @. rho_c_nstar - (ts * rho_cdot_n) + (ts * 0.75 * rho_cdot_nm1)
        rho_r_nstar .= @. rho_r_nstar - (ts * rho_rdot_n) + (ts * 0.75 * rho_rdot_nm1)
        mu_sat_nstar .= @. mu_sat_nstar - (ts * mu_sat_dot_n) + (ts * 0.75 * mu_sat_dot_nm1)
    end

    # Set the n-1 and n-2 terms
    sdot_nm2 .= sdot_nm1
    sdot_nm1 .= sdot_n
    rho_vdot_nm2 .= rho_vdot_nm1
    rho_vdot_nm1 .= rho_vdot_n
    rho_cdot_nm2 .= rho_cdot_nm1
    rho_cdot_nm1 .= rho_cdot_n
    rho_rdot_nm2 .= rho_rdot_nm1
    rho_rdot_nm1 .= rho_rdot_n
    mu_sat_dot_nm2 .= mu_sat_dot_nm1
    mu_sat_dot_nm1 .= mu_sat_dot_n

    # Set up the matrix problem
    col = deepcopy(mtile.tile.kbasis.data[s_index])

    # Solve for the coefficients
    h_a = mtile.diffusion_matrix
    if t == 1
        # Calculate the Helmholtz matrix for the first time step
        h_a = calc_Helmholtz_diffusion_matrix(mtile.tile, mtile.model, ts_term)
    end

    # Set s_n+1 (homogeneous BCs)
    _vertical_solve!(col, h_a, s_nstar, mtile)
    view(mtile.var_np1,colstart:colend,s_index) .= Itransform!(col)

    # Set rho_v_n+1
    _vertical_solve!(col, h_a, rho_v_nstar, mtile)
    view(mtile.var_np1,colstart:colend,rho_v_index) .= Itransform!(col)

    # Set rho_c_n+1
    _vertical_solve!(col, h_a, rho_c_nstar, mtile)
    view(mtile.var_np1,colstart:colend,rho_c_index) .= Itransform!(col)

    # Set rho_r_n+1
    _vertical_solve!(col, h_a, rho_r_nstar, mtile)
    view(mtile.var_np1,colstart:colend,rho_r_index) .= Itransform!(col)

    # Set mu_sat_n+1
    _vertical_solve!(col, h_a, mu_sat_nstar, mtile)
    view(mtile.var_np1,colstart:colend,mu_sat_index) .= Itransform!(col)

end

"""
    explicit_timestep(mtile, colstart, colend, t)

Advance all variables one timestep using AB3 explicit time stepping (Euler for t=1,
second-order AB for t=2, third-order AB3 thereafter per Durran and Blossey 2012).
"""
function explicit_timestep(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    for v in 1:length(mtile.model.grid_params.vars)
        physical = view(mtile.tile.physical,colstart:colend,v,1)
        var_np1 = view(mtile.var_np1,colstart:colend,v)
        expdot_n = view(mtile.expdot_n,colstart:colend,v)
        expdot_nm1 = view(mtile.expdot_nm1,colstart:colend,v)
        expdot_nm2 = view(mtile.expdot_nm2,colstart:colend,v)
        ts = mtile.model.ts

        if (t == 1)
            # Use Euler method and trapezoidal method (AM2) for first step
            var_np1 .= @. physical + (ts * expdot_n)
            expdot_nm1 .= expdot_n
        elseif (t == 2)
            # Use 2nd order A-B method and AI2* for second step
            var_np1 .= @. physical + (0.5 * ts) * ((3.0 * expdot_n) - expdot_nm1)
            expdot_nm2 .= expdot_nm1
            expdot_nm1 .= expdot_n
        else
            # Use AI2*–AB3 implicit-explicit scheme (Durran and Blossey 2012)
            var_np1 .= @. physical + ((ts / 12.0) * ((23.0 * expdot_n) - (16.0 * expdot_nm1) + (5.0 * expdot_nm2)))
            expdot_nm2 .= expdot_nm1
            expdot_nm1 .= expdot_n
        end
    end
end

"""
    explicit_increment(mtile, colstart, colend, t)

Apply an incremental explicit forcing to the current solution using AB3-consistent
weighting, and accumulate it into the stored explicit tendencies.
"""
function explicit_increment(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    for v in 1:length(mtile.model.grid_params.vars)
        var_np1 = view(mtile.var_np1,colstart:colend,v)
        expdot_incr = view(mtile.expdot_incr,colstart:colend,v)
        expdot_n = view(mtile.expdot_n,colstart:colend,v)
        expdot_nm1 = view(mtile.expdot_nm1,colstart:colend,v)
        ts = mtile.model.ts

        if (t == 1)
            # Use Euler method and trapezoidal method (AM2) for first step
            var_np1 .= @. var_np1 + (ts * expdot_incr)
            expdot_n .= expdot_n .+ expdot_incr
            expdot_nm1 .= expdot_n
        elseif (t == 2)
            # Use 2nd order A-B method and AI2* for second step
            var_np1 .= @. var_np1 + ((0.5 * ts) * (3.0 * expdot_incr))
            expdot_n .= expdot_n .+ expdot_incr
            expdot_nm1 .= expdot_n
        else
            # Use AI2*–AB3 implicit-explicit scheme (Durran and Blossey 2012)
            var_np1 .= @. var_np1 + ((ts / 12.0) * (23.0 * expdot_incr))
            expdot_n .= expdot_n .+ expdot_incr
            expdot_nm1 .= expdot_n
        end
    end
end

"""
    calcTendency(mtile)

Transform the updated physical-space state in `var_np1` to spectral space,
storing the result in the tile's spectral array for inter-tile communication.
"""
function calcTendency(mtile::ModelTile)

    # Set the current time
    mtile.tile.physical .= mtile.var_np1

    # Transform to spectral space
    spectralTransform!(mtile.tile)
end

"""
    grid_spacing_minima(patch, model) -> (dz_min, dx_min)

Smallest grid spacing along the vertical (`dz_min`) and horizontal (`dx_min`)
directions, from the patch gridpoints. Used to form Courant numbers. `dx_min`
is `Inf` for a single-column grid. Works for both RZ (Chebyshev, boundary-
clustered) and RiRk (B-spline, near-uniform) vertical bases since it measures
the actual point spacing.
"""
function grid_spacing_minima(patch, model)
    gp = getGridpoints(patch)
    kDim = model.grid_params.kDim
    npts = size(gp, 1)
    z = gp[1:kDim, end]                     # one column's vertical levels (z is the
                                            # LAST gridpoint column on 2D and 3D grids)
    dz_min = minimum(diff(sort(z)))
    # First horizontal coordinate (x, or r on the cylinders). On 3D ring grids many
    # columns share one radius, so measure the unique values; note the tightest
    # horizontal spacing there is really the outer-ring arc length, not dr — this
    # is a run-once advisory diagnostic, not a stability guard.
    x = unique(gp[1:kDim:npts, 1])
    dx_min = length(x) > 1 ? minimum(diff(sort(x))) : Inf
    return dz_min, dx_min
end

"""
    cfl_diagnostics(patch, model, t, dz_min, dx_min, c_bar)

Print CFL/Courant diagnostics for the current physical state: peak vertical and
horizontal velocity, the acoustic Courant number (`c_bar * dt / dz_min`, with
`c_bar = sqrt(Pxi_bar)` the mean sound speed) and the advective Courant numbers
(`max|w|*dt/dz_min`, `max|u|*dt/dx_min`). Cheap reductions over the already-
materialized physical field; no transform of its own.
"""
function cfl_diagnostics(patch, model, t::Int64, dz_min::Float64, dx_min::Float64, c_bar::Float64)
    vars = model.grid_params.vars
    ts = model.ts
    w = view(patch.physical, :, vars["w"], 1)
    maxw = maximum(abs, w)
    line = "  CFL diag t=$(round(t*ts; digits=2)) s: max|w|=$(round(maxw; digits=3)) m/s" *
           ", acoustic Co=$(round(c_bar * ts / dz_min; digits=3))" *
           ", w Co=$(round(maxw * ts / dz_min; digits=3))"
    if haskey(vars, "u") && isfinite(dx_min)
        u = view(patch.physical, :, vars["u"], 1)
        maxu = maximum(abs, u)
        line *= ", max|u|=$(round(maxu; digits=3)) m/s, u Co=$(round(maxu * ts / dx_min; digits=3))"
    end
    println(line)
    return nothing
end

"""
    state_minima_trace(mtile, t)

Optional per-step blow-up-precursor diagnostic for the pressure-reference (mc)
sets, gated on `options[:state_minima_trace] = interval::Int` (0/absent = off).
Every `interval` steps — and on ANY step where the tile minimum of the full dry
density `ρ_d = ρ_d' + ρ̄_d(z)` drops below half its reference — print that
minimum with its location. One cheap pass over two variables; runs
single-threaded before the column loop, so printing is race-free. Intended for
short diagnostic reruns chasing positive-definiteness undershoots (the
`log`-DomainError blow-up class); leave off in production.
"""
function state_minima_trace(mtile::ModelTile, t::Int64)
    interval = Int(get(mtile.model.options, :state_minima_trace, 0))
    interval > 0 || return nothing
    uses_pressure_reference(mtile.model.equation_set) || return nothing
    vars = mtile.model.grid_params.vars
    kDim = mtile.model.grid_params.kDim
    rho_dbar = view(ref_rho_d(mtile.ref_state), :, 1)
    rd = view(mtile.tile.physical, :, vars["rho_d"], 1)
    min_frac = Inf
    min_i = 1
    @inbounds for i in eachindex(rd)
        frac = rd[i] / rho_dbar[mod1(i, kDim)] + 1.0
        if frac < min_frac
            min_frac = frac
            min_i = i
        end
    end
    low = min_frac < 0.5
    if low || t % interval == 0
        k = mod1(min_i, kDim)
        r = mtile.tilepoints[min_i, 1]
        z = mtile.tilepoints[min_i, end]
        println("  rho_d trace t=$(round(t * mtile.model.ts; digits=2)) s: " *
                "min rho_d/ref=$(round(min_frac; digits=4)) " *
                "(rho_d=$(round(rd[min_i] + rho_dbar[k]; sigdigits=4)), " *
                "ref=$(round(rho_dbar[k]; sigdigits=4))) " *
                "at r=$(round(r; digits=1)) z=$(round(z; digits=1))" *
                (low ? "  << LOW" : ""))
    end
    return nothing
end

"""
    checkCFL(grid; t=0, ts=0.0, where="")

Scan every physical-space variable for non-finite values (`NaN` **or** `Inf`),
which indicate a likely CFL violation / numerical blow-up, and `error` on the
first one found. `Inf` typically appears one step before `NaN` in a blow-up, so
`!isfinite` traps the instability earlier than an `isnan`-only check.

Optional `t`/`ts` add the model time `t*ts` to the message, and `where` labels
the call site (e.g. `"worker tile"`). This is called as a cheap per-step trap on
each worker tile (see [`advanceTimestep`](@ref)) so a blow-up halts cleanly
*before* it reaches the NaN-blind spline solve, rather than segfaulting.
"""
function checkCFL(grid; t::Int64=0, ts::Float64=0.0, where::String="")

    # Check to see if CFL condition may have been violated
    for var in keys(grid.params.vars)
        v = grid.params.vars[var]
        testvar = view(grid.physical, :, v, 1)
        for i in eachindex(testvar)
            if !isfinite(testvar[i])
                loc = isempty(where) ? "" : " [$where]"
                tstr = ts > 0.0 ? " at t=$(round(t*ts; digits=3)) s" : ""
                error("Non-finite value ($(testvar[i])) in variable $var at index $i$loc$tstr ! CFL condition likely violated")
            end
            # Can do more extensive checks here to see if collapse is impending
            #TBD
        end
    end
    return nothing
end

"""
    _helmholtz_bc_row(bc, M0, M1, M2, row_idx)

Select the appropriate operator matrix row for a boundary condition in the Helmholtz solver.
Returns the raw row vector from the operator matrix corresponding to the BC type;
the caller is responsible for applying any physics-specific coefficients.

Dispatches on `BoundaryConditions` fields: Dirichlet → M0, Neumann → M1, SecondDeriv → M2.
"""
function _helmholtz_bc_row(bc::BoundaryConditions, M0, M1, M2, row_idx)
    if is_periodic(bc)
        error("PeriodicBC not supported in Helmholtz solver")
    elseif bc.robin !== nothing
        error("RobinBC not yet supported in Helmholtz solver")
    elseif is_inhomogeneous(bc)
        error("Inhomogeneous BCs not yet supported in Helmholtz solver")
    elseif bc.u !== nothing       # Dirichlet
        return M0[row_idx, :]
    elseif bc.du !== nothing      # Neumann
        return M1[row_idx, :]
    elseif bc.d2u !== nothing     # Second derivative
        return M2[row_idx, :]
    else                          # Natural (R0)
        return M0[row_idx, :]
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Cubic B-spline vertical (RiRk): Galerkin Helmholtz support
#
# The Chebyshev acoustic solver is a square pseudospectral collocation: the DCT
# has #points == #coefficients == kDim, so operator_matrix(:k,·) is kDim×kDim and
# `h \ g` is a square solve. A cubic B-spline instead has b_kDim = num_cells + 3
# coefficients but kDim = num_cells·mubar mish points, so operator_matrix(:k,·) is
# the rectangular (kDim × b_kDim) evaluation matrix and `h \ g` degenerates to an
# ill-posed least-squares solve.
#
# We restore a square system with a Galerkin (finite-element) discretisation on
# the spline basis. The weak form of (α ∂_zz + β) is  -α ∫ψ'φ' + β ∫ψφ, integrated
# with the spline's own Gauss quadrature (weights W at the mish points). Writing
# M0, M1 for the basis and first-derivative matrices at the mish points, the
# operator is  A = -α M1ᵀW M1 + β M0ᵀW M0  (symmetric, b_kDim×b_kDim) and the load
# vector is  b = M0ᵀW g_mish. Crucially A is built from the *same* mish operators
# the explicit tendencies use, so the implicit and explicit acoustic operators are
# consistent — a requirement of the AI2* split that a node-collocation operator
# (built at different points, and non-symmetric) violates, producing a slow
# acoustic instability. Neumann conditions are natural; Dirichlet conditions
# replace the first/last rows with the boundary-value constraint. Full derivation:
# reference/rirk_vertical_solver.tex.
# ─────────────────────────────────────────────────────────────────────────────

const _RIRK_SOLVE_CACHE = Dict{UInt, NamedTuple}()
const _RIRK_SOLVE_LOCK = ReentrantLock()
# Per-factorization Dirichlet flags (bottom, top), so the load vector can zero the
# matching boundary rows. Keyed by objectid of the factorization object.
const _RIRK_DIRICHLET = Dict{UInt, Tuple{Bool, Bool}}()

_is_dirichlet(bc::BoundaryConditions) = bc.u !== nothing

"""
Per-column cached Galerkin data: the mish basis matrix `M0` and first-derivative
matrix `M1` (both `kDim × b_kDim`), the diagonal physical Gauss-quadrature weights
`W` (length `kDim`), and the boundary-value rows `Nb` (`2 × b_kDim`, evaluated at
`z_b` and `z_t`) used to impose Dirichlet conditions. `M0`/`M1` are exactly the
operators the explicit gridTransform tendencies use, ensuring consistency.
"""
function _rirk_solve_data(kcol::CubicBSpline.Spline1D)
    lock(_RIRK_SOLVE_LOCK) do
        get!(_RIRK_SOLVE_CACHE, objectid(kcol)) do
            sp = kcol.params
            M0 = CubicBSpline.SItransform_matrix(kcol, kcol.mishPoints, 0)
            M1 = CubicBSpline.SItransform_matrix(kcol, kcol.mishPoints, 1)
            _, qw = CubicBSpline._quadrature_rule(sp.mubar, sp.quadrature)
            W  = repeat(qw .* sp.DX, outer = sp.num_cells)   # physical weights, length kDim
            Nb = CubicBSpline.SItransform_matrix(kcol, [sp.xmin, sp.xmax], 0)
            # α-independent Galerkin mass matrix, precomputed so per-step
            # profile-coefficient assemblies (the state-dependent semi-implicit)
            # only rebuild the weighted stiffness.
            Mass = M0' * (W .* M0)
            (M0 = M0, M1 = M1, W = W, Nb = Nb, Mass = Mass)
        end
    end
end

# Galerkin assembly of (α ∂_zz + β) on the spline basis; see the block comment.
function _assemble_spline_matrix(d, α::Float64, β::Float64,
        bc_bottom::BoundaryConditions, bc_top::BoundaryConditions)
    Mass  = d.M0' * (d.W .* d.M0)
    Stiff = d.M1' * (d.W .* d.M1)
    A = (-α) .* Stiff .+ β .* Mass
    db = _is_dirichlet(bc_bottom); dt = _is_dirichlet(bc_top)
    if db; A[1,   :] .= d.Nb[1, :]; end
    if dt; A[end, :] .= d.Nb[2, :]; end
    F = factorize(A)
    lock(_RIRK_SOLVE_LOCK) do
        _RIRK_DIRICHLET[objectid(F)] = (db, dt)
    end
    return F
end

"""
    _assemble_sd_helmholtz(d, α, db, dt)

Per-column, per-step Helmholtz assembly for the STATE-DEPENDENT semi-implicit:
the operator `∂z(α(z)∂z·) − 1` in the weak Galerkin form `−M1ᵀ(W·α)M1 − Mass`
with `α = Δτ²·Pξⁿ(z)` evaluated from the CURRENT column state, Dirichlet rows
per `db`/`dt`. Unlike `_assemble_spline_matrix` this reuses the precomputed
`d.Mass` and does NOT register in the `_RIRK_DIRICHLET` dict (which would take
a global lock and grow by one entry per column per step) — callers pass the
Dirichlet flags to `_vertical_solve!` explicitly.
"""
function _assemble_sd_helmholtz(d, α::AbstractVector{Float64}, db::Bool, dt::Bool)
    A = -(d.M1' * ((d.W .* α) .* d.M1)) .- d.Mass
    if db; A[1,   :] .= d.Nb[1, :]; end
    if dt; A[end, :] .= d.Nb[2, :]; end
    return factorize(A)
end

# Profile-coefficient variant, ∂_z(α(z) ∂_z ·) + β with α given at the mish points
# (the mc pressure-reference local sound speed Δτ²·Pξ̄(z)): the weak form weights
# the stiffness quadrature, ∫ψ'αφ' = M1ᵀ(W·α)M1, keeping A symmetric. A separate
# method (not a Union) so the scalar path stays bitwise-identical for the legacy
# sets and the diffusion matrices.
function _assemble_spline_matrix(d, α::AbstractVector{Float64}, β::Float64,
        bc_bottom::BoundaryConditions, bc_top::BoundaryConditions)
    Mass  = d.M0' * (d.W .* d.M0)
    Stiff = d.M1' * ((d.W .* α) .* d.M1)
    A = (-1.0) .* Stiff .+ β .* Mass
    db = _is_dirichlet(bc_bottom); dt = _is_dirichlet(bc_top)
    if db; A[1,   :] .= d.Nb[1, :]; end
    if dt; A[end, :] .= d.Nb[2, :]; end
    F = factorize(A)
    lock(_RIRK_SOLVE_LOCK) do
        _RIRK_DIRICHLET[objectid(F)] = (db, dt)
    end
    return F
end

"""
    _assemble_vertical_matrix(grid, model, α, β, bc_scale, bc_bottom, bc_top)

Assemble and factorize the vertical operator `α ∂_zz + β` for the semi-implicit
solves. For a Chebyshev k-basis this is the square `kDim × kDim` pseudospectral
collocation (`α M2 + β M0`, interior rows `2:nz-1`, BC rows scaled by `bc_scale`);
for a cubic B-spline k-basis it is the symmetric `b_kDim × b_kDim` Galerkin form.
"""
function _assemble_vertical_matrix(grid::AbstractGrid, model::ModelParameters,
        α::Float64, β::Float64, bc_scale::Float64,
        bc_bottom::BoundaryConditions, bc_top::BoundaryConditions)
    kcol = grid.kbasis.data[1]
    if kcol isa CubicBSpline.Spline1D
        return _assemble_spline_matrix(_rirk_solve_data(kcol), α, β, bc_bottom, bc_top)
    else
        nz = model.grid_params.kDim
        M0 = operator_matrix(grid, :k, 0)
        M1 = operator_matrix(grid, :k, 1)
        M2 = operator_matrix(grid, :k, 2)
        h = α .* M2 .+ β .* M0
        bc1 = bc_scale .* _helmholtz_bc_row(bc_bottom, M0, M1, M2, 1)
        bc2 = bc_scale .* _helmholtz_bc_row(bc_top, M0, M1, M2, nz)
        return factorize([bc1[:]'; bc2[:]'; h[2:nz-1, :]])
    end
end

# Profile-coefficient variant, ∂_z(α(z) ∂_z ·) + β with α at the mish points. The
# spline path is the symmetric weighted-stiffness Galerkin form; the Chebyshev
# path is the exact collocation of the product, M1·(M0 \ (α .* M1)) — the nodal
# derivative of the fitted pointwise product α·∂_z(·), mirroring the pointwise
# recovery/staging chain, with no need for ∂_z α.
function _assemble_vertical_matrix(grid::AbstractGrid, model::ModelParameters,
        α::AbstractVector{Float64}, β::Float64, bc_scale::Float64,
        bc_bottom::BoundaryConditions, bc_top::BoundaryConditions)
    kcol = grid.kbasis.data[1]
    if kcol isa CubicBSpline.Spline1D
        return _assemble_spline_matrix(_rirk_solve_data(kcol), α, β, bc_bottom, bc_top)
    else
        nz = model.grid_params.kDim
        M0 = operator_matrix(grid, :k, 0)
        M1 = operator_matrix(grid, :k, 1)
        M2 = operator_matrix(grid, :k, 2)
        h = (M1 * (M0 \ (α .* M1))) .+ β .* M0
        bc1 = bc_scale .* _helmholtz_bc_row(bc_bottom, M0, M1, M2, 1)
        bc2 = bc_scale .* _helmholtz_bc_row(bc_top, M0, M1, M2, nz)
        return factorize([bc1[:]'; bc2[:]'; h[2:nz-1, :]])
    end
end

"""
    _vertical_solve!(col, h_a, rhs_mish, mtile)

Solve the factorized vertical system `h_a` for the spectral coefficients `col.a`,
given the right-hand side `rhs_mish` sampled at the `kDim` physical mish points
(boundary rows are homogeneous). For a Chebyshev k-basis the interior mish values
are used directly; for a cubic B-spline k-basis the Galerkin load vector
`M0ᵀW rhs_mish` is formed and the Dirichlet boundary rows (if any) are zeroed.
"""
function _vertical_solve!(col, h_a, rhs_mish::AbstractVector, mtile::ModelTile;
        dirichlet::Union{Nothing, NTuple{2, Bool}}=nothing)
    # Everything here is preallocated per-thread. This runs 4x per column per timestep, so the
    # old version's cost was structural, not incidental: it allocated three temporaries
    # (`d.W .* rhs_mish`, the `M0'*` product, and `h_a \ b`) and — worse — reached
    # `_rirk_solve_data`, which took a GLOBAL LOCK and hit a Dict on every call. That is ~22 M
    # lock acquisitions across a straka93 run, from every thread at once. The solve data is
    # per-grid and invariant, so it now lives on the tile.
    tid = Threads.threadid()
    d = mtile.solve_data
    b = @inbounds view(mtile.solve_load, :, tid)

    if d !== nothing
        # Cubic B-spline (RiRk) k-basis: Galerkin load vector M0ᵀ W rhs_mish.
        w = @inbounds view(mtile.solve_rhs, :, tid)
        w .= d.W .* rhs_mish
        mul!(b, d.M0', w)
        # Dirichlet flags: explicit from the caller (per-step factorizations,
        # `_assemble_sd_helmholtz`) or from the registry keyed by the
        # precomputed factorization object.
        db, dt = dirichlet === nothing ?
            get(_RIRK_DIRICHLET, objectid(h_a), (false, false)) : dirichlet
        if db; b[1]   = 0.0; end
        if dt; b[end] = 0.0; end
    else
        # Chebyshev k-basis: interior mish values directly, homogeneous boundary rows.
        nz = length(rhs_mish)
        b[1] = 0.0
        b[2] = 0.0
        @inbounds @views b[3:nz] .= rhs_mish[2:nz-1]
    end

    ldiv!(col.a, h_a, b)
    return col
end

"""
    calc_Helmholtz_semiimplicit_matrix_xi(grid, model, Pxi_bar, ts_term; bc_bottom, bc_top)

Build and factorize the spectral-collocation Helmholtz matrix for the xi-form
semi-implicit solve. Boundary condition type is configurable via keyword arguments.

# Arguments
- `grid::AbstractGrid`: the tile grid providing basis objects for `operator_matrix`.
- `model::ModelParameters`: model configuration providing grid parameters.
- `Pxi_bar::Float64`: domain-mean speed of sound squared.
- `ts_term::Float64`: time-stepping coefficient.
- `bc_bottom::BoundaryConditions`: bottom boundary condition (default: `NeumannBC()`).
- `bc_top::BoundaryConditions`: top boundary condition (default: `NeumannBC()`).
"""
function calc_Helmholtz_semiimplicit_matrix_xi(grid::AbstractGrid, model::ModelParameters, Pxi_bar::Float64, ts_term::Float64;
        bc_bottom::BoundaryConditions=NeumannBC(), bc_top::BoundaryConditions=NeumannBC())

    c = -ts_term * ts_term * Pxi_bar
    return _assemble_vertical_matrix(grid, model, c, 1.0, c, bc_bottom, bc_top)
end

"""
    calc_Helmholtz_semiimplicit_matrix(grid, model, Pxi_bar, ts_term; bc_bottom, bc_top)

Build and factorize the spectral-collocation Helmholtz matrix for the w-form
semi-implicit solve. Boundary condition type is configurable via keyword arguments.

# Arguments
- `grid::AbstractGrid`: the tile grid providing basis objects for `operator_matrix`.
- `model::ModelParameters`: model configuration providing grid parameters.
- `Pxi_bar::Float64`: domain-mean speed of sound squared.
- `ts_term::Float64`: time-stepping coefficient.
- `bc_bottom::BoundaryConditions`: bottom boundary condition (default: `DirichletBC()`).
- `bc_top::BoundaryConditions`: top boundary condition (default: `DirichletBC()`).
"""
function calc_Helmholtz_semiimplicit_matrix(grid::AbstractGrid, model::ModelParameters, Pxi_bar::Float64, ts_term::Float64;
        bc_bottom::BoundaryConditions=DirichletBC(), bc_top::BoundaryConditions=DirichletBC())

    c = ts_term * ts_term * Pxi_bar
    return _assemble_vertical_matrix(grid, model, c, -1.0, 1.0, bc_bottom, bc_top)
end

"""
Profile variant of [`calc_Helmholtz_semiimplicit_matrix`](@ref): `Pxi_prof` is the
LOCAL reference sound speed squared γ̄_m(z)·p̄(z)/ρ̄_t(z) at the mish points (the mc
pressure-reference sets; see `mc_reference_diagnostics`). The operator becomes
`∂_z(Δτ²·Pξ̄(z)·∂_z) - 1`, so the acoustic remainder left to AB3 is O(perturbation)
at every level — a domain-mean c̄² leaves the local deviation explicit, which is
the classic reference-state SI instability (Simmons, Hoskins & Burridge 1978) and
blows up above Co_z ≈ 4.5 on a realistically stratified sounding.
"""
function calc_Helmholtz_semiimplicit_matrix(grid::AbstractGrid, model::ModelParameters, Pxi_prof::AbstractVector{Float64}, ts_term::Float64;
        bc_bottom::BoundaryConditions=DirichletBC(), bc_top::BoundaryConditions=DirichletBC())

    c = (ts_term * ts_term) .* Pxi_prof
    return _assemble_vertical_matrix(grid, model, c, -1.0, 1.0, bc_bottom, bc_top)
end

"""
    calc_Helmholtz_diffusion_matrix(grid, model, ts_term; bc_bottom, bc_top)

Build and factorize the spectral-collocation Helmholtz matrix for implicit vertical
diffusion. Boundary condition type is configurable via keyword arguments.

# Arguments
- `grid::AbstractGrid`: the tile grid providing basis objects for `operator_matrix`.
- `model::ModelParameters`: model configuration providing grid parameters.
- `ts_term::Float64`: time-stepping coefficient.
- `bc_bottom::BoundaryConditions`: bottom boundary condition (default: `NeumannBC()`).
- `bc_top::BoundaryConditions`: top boundary condition (default: `NeumannBC()`).
"""
function calc_Helmholtz_diffusion_matrix(grid::AbstractGrid, model::ModelParameters, ts_term::Float64;
        bc_bottom::BoundaryConditions=NeumannBC(), bc_top::BoundaryConditions=NeumannBC())

    return _assemble_vertical_matrix(grid, model, -ts_term, 1.0, 1.0, bc_bottom, bc_top)
end
