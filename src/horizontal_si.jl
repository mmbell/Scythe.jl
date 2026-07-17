# ─────────────────────────────────────────────────────────────────────────────
# Horizontal acoustic semi-implicit sweep for the mc (pressure-reference) sets.
#
# ADI companion to the per-column vertical solve in `semiimplicit_adjustment_p`:
# the reference-linear HORIZONTAL acoustic pair
#
#     ∂u/∂t  = −(1/ρ̄_t) ∂x p′,      ∂p′/∂t = −Pξ̄(z) ρ̄_t ∂x u
#     slaved: ∂ρ_t′/∂t = −ρ̄_t ∂x u,  ∂ρ_d′/∂t = −ρ̄_d ∂x u,
#             ∂E_t′/∂t = −(Ē_t+p̄) ∂x u
#
# is integrated with the same AI2* off-centering as the vertical legs. The
# explicit history levels are applied per column in `mc_driver!`
# (`horizontal_si_history!`, histories in the tile's `hacdot_*` channel — the
# `impdot` slots are owned by the vertical solve); the implicit level is this
# patch-level sweep, which runs on the merged patch B coefficients between
# `accumulate_at_map!` and `splineTransform!` in `model_loop`. Eliminating
# p′^{n+1} gives one scalar Helmholtz **in u** per physical vertical level z_k
# (the reference coefficients are z-only, so they are constants along the solve
# direction — the u-form mirrors the vertical φ = ρ̄_t w solve: a weak Galerkin
# solve in u's Dirichlet side-wall basis, everything else recovered by strong
# derivatives of the solved coefficients):
#
#     (I − Δτ² Pξ̄(z_k) ∂xx) u^{n+1} = u* − (Δτ/ρ̄_t(z_k)) ∂x p′*
#
#     p′^{n+1}  = p′*  − Δτ Pξ̄ ρ̄_t ∂x u^{n+1}
#     ρ_t′^{n+1} = ρ_t′* − Δτ ρ̄_t ∂x u^{n+1}      (+ the ρ_d′/E_t′ analogues)
#
# Operator consistency across the AI2* time levels (THE stability requirement —
# see tc/SI_VERTICAL_CEILING.md) is by construction:
#   • the solve reads u*, ∂x p′* through EXACTLY the model's read chain
#     (per-z_b-block i-direction SA fit → evaluate, then per-i-point vertical
#     SA fit → evaluate — the `gridTransform` chain), the same chain that
#     produces the `u_x`/`pp_x` grid slots the fresh `hacdot` staging uses;
#   • the corrections are written back as B-coefficient INCREMENTS through the
#     model's forward chain (per-i-point vertical SB, then per-z_b-block
#     i-direction SB — the `spectralTransform` chain), so the correction gets
#     the same fit treatment `calcTendency` gives the state;
#   • the u leg is recovered through the weak Galerkin solve, so its history is
#     the STORED APPLIED INCREMENT ((u^{n+1} − u*)/Δτ, the 2026-07-16 w-leg
#     lesson), carried back to the tiles through the `hsi_u_incr` shared array.
#
# Phase 1 scope: Cartesian XZ (`RiRk`) single patch, Dirichlet or Neumann u
# side walls. The cylindrical geometries take the p′-form per-(level,
# wavenumber) solves in later phases.
# ─────────────────────────────────────────────────────────────────────────────

"""
Precomputed patch-level data for the horizontal semi-implicit sweep: i-direction
Galerkin solve data for u's and p's side-wall bases, the per-physical-level
Helmholtz factorizations (production `Δτ = 1.25·ts` set and the first-step
`0.5·ts` AM2 set), the z-only reference coefficient profiles at the kDim mish
levels, and preallocated level/transform work arrays. Built once by
[`create_horizontal_solve_data`](@ref).
"""
# Parametric so the per-level solve loop infers: `du`/`dp` are the Galerkin
# NamedTuples, `fact*` the per-level factorization vectors — abstractly typed
# fields here made the recovery loop box every M0/M1 access (measured 380 ms
# per sweep vs 3 ms typed).
struct HorizontalSolveData{DU, DP, FA, FB}
    u_index::Int
    p_index::Int
    rhod_index::Int
    rhot_index::Int
    et_index::Int
    du::DU                  # (M0, M1, W, Nb) on u's i-basis
    dp::DP                  # (M0, M1, W, Nb) on p's i-basis
    dirichlet::Tuple{Bool, Bool}
    fact::FA                # per-level factorizations, Δτ = 1.25·ts
    fact_first::FB          # per-level factorizations, Δτ = 0.5·ts (t == 1)
    Pxi::Vector{Float64}    # Pξ̄(z_k), local reference sound speed squared
    rho_tbar::Vector{Float64}
    rho_dbar::Vector{Float64}
    etp_bar::Vector{Float64}    # Ē_t + p̄
    # Work arrays (single-threaded master-side sweep)
    ulev::Matrix{Float64}   # (iDim, kDim) u* at physical points
    pxlev::Matrix{Float64}  # (iDim, kDim) ∂x p′* at physical points
    dlev::Array{Float64,3}  # (iDim, kDim, 5) corrections δu, δp, δρ_d, δρ_t, δE_t
    zbuf::Matrix{Float64}   # (iDim, b_kDim) i-evaluated vertical B coefficients
    rhs::Vector{Float64}    # (iDim)
    load::Vector{Float64}   # (b_iDim)
    acoef::Vector{Float64}  # (b_iDim)
    tempcb::Matrix{Float64} # (b_kDim, iDim) forward-transform staging
    zwork::Vector{Float64}  # (kDim) ∂z work column for the w-history correction
end

# Correction plane order in `dlev` (and the write-back loop): the var each
# plane belongs to is looked up from these fields at use sites.
const _HSI_PLANES = (:u_index, :p_index, :rhod_index, :rhot_index, :et_index)

"""
    create_horizontal_solve_data(patch, model, Pxi_prof, rho_tbar, rho_dbar, etp_bar)

Build the [`HorizontalSolveData`](@ref) for `patch`. The reference profiles are
passed in (kDim vectors at the physical mish levels) so the caller can source
them from a worker's `ModelTile` (`model_loop`) or a local one (tests). XZ
(`RiRk`) only in Phase 1; the option is rejected for other geometries.
"""
function create_horizontal_solve_data(patch::AbstractGrid, model::ModelParameters,
        Pxi_prof::AbstractVector{Float64}, rho_tbar::AbstractVector{Float64},
        rho_dbar::AbstractVector{Float64}, etp_bar::AbstractVector{Float64})

    model.grid_params.geometry == "RiRk" || error(
        "options[:horizontal_semiimplicit] is only implemented for the XZ RiRk " *
        "geometry (got $(model.grid_params.geometry)); the cylindrical solves " *
        "are later phases")
    vars = model.grid_params.vars
    u_index = vars["u"]; p_index = vars["p"]
    rhod_index = vars["rho_d"]; rhot_index = vars["rho_t"]; et_index = vars["E_t"]

    iDim = patch.params.iDim; kDim = patch.params.kDim
    b_iDim = patch.params.b_iDim; b_kDim = patch.params.b_kDim
    length(Pxi_prof) == kDim || error("Pxi_prof must be a kDim-level profile")

    # i-direction Galerkin data on u's and p's side-wall bases (all z_b share one
    # SplineParameters per var, so the first block's spline serves the whole patch).
    du = _rirk_solve_data(patch.ibasis.data[1, u_index])
    dp = _rirk_solve_data(patch.ibasis.data[1, p_index])

    bcl = model.grid_params.BCL["u"]; bcr = model.grid_params.BCR["u"]
    for bc in (bcl, bcr)
        (_is_dirichlet(bc) || bc.du !== nothing) || error(
            "horizontal SI supports Dirichlet or Neumann u side walls only")
    end

    # Per-level factorizations of (I − Δτ² Pξ̄(z_k) ∂xx) in the weak form
    # Mass + Δτ²Pξ̄·Stiff (SPD before the Dirichlet rows): `_assemble_spline_matrix`
    # builds −α·Stiff + β·Mass, so α = −Δτ²Pξ̄, β = 1.
    build_set(ts_term) = [
        _assemble_spline_matrix(du, -(ts_term^2) * Pxi_prof[k], 1.0, bcl, bcr)
        for k in 1:kDim]
    fact = build_set(1.25 * model.ts)
    fact_first = build_set(0.5 * model.ts)

    return HorizontalSolveData(u_index, p_index, rhod_index, rhot_index, et_index,
        du, dp, (_is_dirichlet(bcl), _is_dirichlet(bcr)), fact, fact_first,
        collect(Pxi_prof), collect(rho_tbar), collect(rho_dbar), collect(etp_bar),
        zeros(iDim, kDim), zeros(iDim, kDim), zeros(iDim, kDim, 5),
        zeros(iDim, b_kDim), zeros(iDim), zeros(b_iDim), zeros(b_iDim),
        zeros(b_kDim, iDim), zeros(kDim))
end

# Read chain: patch B coefficients → values (or ∂x, via `deriv`) at the physical
# mish points, exactly mirroring `gridTransform`'s per-variable path so the sweep
# sees the same discrete state the grid slots see.
function _hsi_to_levels!(out::Matrix{Float64}, hsd::HorizontalSolveData,
        spectral::AbstractMatrix{Float64}, patch::AbstractGrid, v::Int, deriv::Int)
    iDim = patch.params.iDim; kDim = patch.params.kDim
    b_iDim = patch.params.b_iDim; b_kDim = patch.params.b_kDim
    zbuf = hsd.zbuf
    for z in 1:b_kDim
        isp = patch.ibasis.data[z, v]
        r1 = (z - 1) * b_iDim
        @inbounds for i in 1:b_iDim
            isp.b[i] = spectral[r1 + i, v]
        end
        SAtransform!(isp)
        if deriv == 0
            SItransform!(isp)
            @inbounds for i in 1:iDim
                zbuf[i, z] = isp.uMish[i]
            end
        else
            SIxtransform(isp, view(zbuf, :, z))
        end
    end
    kcol = patch.kbasis.data[v]
    for i in 1:iDim
        @inbounds for z in 1:b_kDim
            kcol.b[z] = zbuf[i, z]
        end
        SAtransform!(kcol)
        SItransform!(kcol)
        @inbounds for z in 1:kDim
            out[i, z] = kcol.uMish[z]
        end
    end
    return out
end

# Write-back chain: a correction field at the physical mish points → B-coefficient
# increments ADDED to the patch spectral state, mirroring `spectralTransform`'s
# per-variable path (per-i vertical SB, then per-z_b i-direction SB). Incremental,
# so untouched variables and the carried state itself are never re-fit here.
function _hsi_add_increment!(spectral::AbstractMatrix{Float64}, hsd::HorizontalSolveData,
        patch::AbstractGrid, v::Int, dfield::AbstractMatrix{Float64})
    iDim = patch.params.iDim; kDim = patch.params.kDim
    b_iDim = patch.params.b_iDim; b_kDim = patch.params.b_kDim
    tempcb = hsd.tempcb
    kcol = patch.kbasis.data[v]
    for i in 1:iDim
        @inbounds for z in 1:kDim
            kcol.uMish[z] = dfield[i, z]
        end
        SBtransform!(kcol)
        @inbounds for z in 1:b_kDim
            tempcb[z, i] = kcol.b[z]
        end
    end
    for z in 1:b_kDim
        isp = patch.ibasis.data[z, v]
        @inbounds for i in 1:iDim
            isp.uMish[i] = tempcb[z, i]
        end
        SBtransform!(isp)
        r1 = (z - 1) * b_iDim
        @inbounds for i in 1:b_iDim
            spectral[r1 + i, v] += isp.b[i]
        end
    end
    return spectral
end

"""
    horizontal_si_correct!(spectral, patch, model, hsd, t) -> incr

Apply the implicit horizontal acoustic correction to the merged patch
B coefficients `spectral` (the `sharedSpectral` array in `model_loop`, or
`patch.spectral` in single-process tests). Solves the per-level u-form
Helmholtz and updates u/p′/ρ_d′/ρ_t′/E_t′ as B-coefficient increments. Returns
an `(npts × 2)` matrix at the patch physical points (row order
`(i−1)·kDim + k`): column 1 the applied u increment `(u^{n+1} − u*)/Δτ` (the
u-leg AI2* history) and column 2 the staleness correction `−∂z δp/ρ̄_t` to the
vertical solve's stored w-leg history — both distributed to the tiles by
[`horizontal_si_load_increment!`](@ref) at the top of the next step.
"""
function horizontal_si_correct!(spectral::AbstractMatrix{Float64}, patch::AbstractGrid,
        model::ModelParameters, hsd::HorizontalSolveData, t::Int64)

    iDim = patch.params.iDim; kDim = patch.params.kDim
    # options[:hsi_scheme]: "ai2s" (default) = the off-centered AI2* weights of
    # the vertical solve; "am2" = trapezoidal (the t == 1 branch every step) on
    # the horizontal channel only — 2nd-order, single history level, neutrally
    # stable for pure-horizontal modes (an A/B lever for the factorization-
    # interaction experiments; the vertical channel keeps AI2* regardless).
    am2 = get(model.options, :hsi_scheme, "ai2s") == "am2"
    ts_term = (t == 1 || am2) ? 0.5 * model.ts : 1.25 * model.ts
    facts = (t == 1 || am2) ? hsd.fact_first : hsd.fact

    _hsi_to_levels!(hsd.ulev, hsd, spectral, patch, hsd.u_index, 0)
    _hsi_to_levels!(hsd.pxlev, hsd, spectral, patch, hsd.p_index, 1)

    du = hsd.du
    dlev = hsd.dlev
    for k in 1:kDim
        # rhs = u* − (Δτ/ρ̄_t) ∂x p′*, Galerkin load M0ᵀW·rhs, Dirichlet rows zeroed
        @inbounds for i in 1:iDim
            hsd.rhs[i] = du.W[i] * (hsd.ulev[i, k] - (ts_term / hsd.rho_tbar[k]) * hsd.pxlev[i, k])
        end
        mul!(hsd.load, du.M0', hsd.rhs)
        if hsd.dirichlet[1]; hsd.load[1] = 0.0; end
        if hsd.dirichlet[2]; hsd.load[end] = 0.0; end
        ldiv!(hsd.acoef, facts[k], hsd.load)

        # Recoveries: δu from the solved coefficients; the p/ρ_d/ρ_t/E_t legs from
        # the strong derivative ∂x u^{n+1} with the z-only reference coefficients.
        cp = ts_term * hsd.Pxi[k] * hsd.rho_tbar[k]
        ct = ts_term * hsd.rho_tbar[k]
        cd = ts_term * hsd.rho_dbar[k]
        ce = ts_term * hsd.etp_bar[k]
        @inbounds for i in 1:iDim
            unew = 0.0
            dxu = 0.0
            for j in 1:size(du.M0, 2)
                unew += du.M0[i, j] * hsd.acoef[j]
                dxu += du.M1[i, j] * hsd.acoef[j]
            end
            dlev[i, k, 1] = unew - hsd.ulev[i, k]
            dlev[i, k, 2] = -cp * dxu
            dlev[i, k, 3] = -cd * dxu
            dlev[i, k, 4] = -ct * dxu
            dlev[i, k, 5] = -ce * dxu
        end
    end

    _hsi_add_increment!(spectral, hsd, patch, hsd.u_index,    view(dlev, :, :, 1))
    _hsi_add_increment!(spectral, hsd, patch, hsd.p_index,    view(dlev, :, :, 2))
    _hsi_add_increment!(spectral, hsd, patch, hsd.rhod_index, view(dlev, :, :, 3))
    _hsi_add_increment!(spectral, hsd, patch, hsd.rhot_index, view(dlev, :, :, 4))
    _hsi_add_increment!(spectral, hsd, patch, hsd.et_index,   view(dlev, :, :, 5))

    # Per-point feeds for the workers, (i−1)·kDim + k row order. Planes 1–5:
    # the applied increments (X^{n+1} − X*)/Δτ of the (u, p, ρ_d, ρ_t, E_t)
    # legs — the exact operator the sweep applied, available as the next
    # step's AI2* histories (see horizontal_si_load_increment! for which
    # channels consume them under which options). Plane 6: the factorization
    # cross-term field G = (1/ρ̄_t)∂z(Pξ̄ρ̄_t ∂x u^{n+1}) = −∂z(δp)/(ρ̄_t Δτ),
    # the ONLY nonzero row of BA·X for this acoustic pair; the next step adds
    # +ν²·G to the w predictor before the vertical solve, converting the
    # factored composition's defect from O(ν²·X) per step to O(ν³) — the
    # Douglas–Gunn-style consistency repair (derivation in
    # reference/horizontal_si_phase2_findings.md, addendum). ∂z is taken
    # through p's own column basis, the chain that produces the p_z slot.
    incr = zeros(iDim * kDim, 6)
    @inbounds for n in 1:5, i in 1:iDim, k in 1:kDim
        incr[(i - 1) * kDim + k, n] = dlev[i, k, n] / ts_term
    end
    kcol = patch.kbasis.data[hsd.p_index]
    for i in 1:iDim
        @inbounds for k in 1:kDim
            kcol.uMish[k] = dlev[i, k, 2]
        end
        SBtransform!(kcol)
        SAtransform!(kcol)
        SIxtransform(kcol, hsd.zwork)
        @inbounds for k in 1:kDim
            incr[(i - 1) * kDim + k, 6] = -hsd.zwork[k] / (hsd.rho_tbar[k] * ts_term)
        end
    end
    return incr
end

"""
    horizontal_si_history!(mtile, colstart, colend, t)

Apply the explicit AI2* history levels of the horizontal acoustic legs to the
predictor state (`ts·(0.75·hacdot_{n−1} − hacdot_n)`; `+0.5·ts·hacdot_n` on the
AM2 first step) and roll the history. Called per column in `mc_driver!` after
`explicit_timestep` and before `semiimplicit_adjustment_p`: both dimensions'
explicit history levels belong to the star state before either implicit solve
(the ADI factorization applies the z solve first, then the x sweep at the patch
layer). p/ρ_d/ρ_t/E_t histories are staged fresh in `mc_driver!`; the u history
is the stored applied increment loaded by [`horizontal_si_load_increment!`](@ref).
"""
function horizontal_si_history!(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)
    vars = mtile.model.grid_params.vars
    ts = mtile.model.ts
    am2 = get(mtile.model.options, :hsi_scheme, "ai2s") == "am2"
    for var in ("p", "rho_d", "rho_t", "u", "E_t")
        index = vars[var]
        nstar = view(mtile.var_np1, colstart:colend, index)
        dot_n = view(mtile.hacdot_n, colstart:colend, index)
        dot_nm1 = view(mtile.hacdot_nm1, colstart:colend, index)
        if t == 1 || am2
            nstar .+= (0.5 * ts) .* dot_n
        else
            nstar .+= ts .* ((0.75 .* dot_nm1) .- dot_n)
        end
        dot_nm1 .= dot_n
    end
    # Factorization cross-term compensation: +ν²·G on the w predictor before
    # the vertical solve (G lagged one step through the increment feed; zero
    # at t == 1). See horizontal_si_correct!'s plane-6 comment and the
    # findings-note addendum for the derivation and the sign cross-check.
    if get(mtile.model.options, :hsi_cross_comp, false) === true
        ts_term = (t == 1) ? 0.5 * ts : 1.25 * ts
        w_index = vars["w"]
        view(mtile.var_np1, colstart:colend, w_index) .+=
            (ts_term^2) .* view(mtile.hacdot_n, colstart:colend, w_index)
    end
    return nothing
end

"""
    horizontal_si_load_increment!(mtile, u_incr, t)

Distribute this tile's window of the sweep's applied u increment (from
[`horizontal_si_correct!`](@ref), row order `(i−1)·kDim + k`, column 1) into
the tile's `hacdot_n` u column as the u-leg AI2* history — ONLY under
`options[:hsi_u_history] = "stored"`. The default (`"none"`) leaves the u
history zero, integrating the u acoustic leg with its implicit level alone
(θ = 1): parameter-free, and it damps only the horizontal acoustic modes the
SI exists to suppress. Measured on the resting-base sweep (2026-07-17):
"none" is neutrally stable through Co_h 9; "stored" — the exact mirror of the
vertical w-leg discipline — carries a slow Courant-independent leak (e-fold
≈ 90–110 s, a smooth oblique mode) from the AI2* explicit levels interacting
with the non-commuting z/x factorization; pointwise-fresh staging blows up in
minutes (the w-leg lesson, reconfirmed). Revisit in the Phase-2 A/B round
(all-stored-x-channel variant) before the default is finalized.

Called at the top of step `t` with the feed from step `t−1`; at `t == 2` (the
first load after the AM2 start) the n−1 level is seeded too, mirroring the
vertical w-leg's first-step seeding. `rowstart` is the tile's first physical
row in patch numbering.
"""
function horizontal_si_load_increment!(mtile::ModelTile, incr::AbstractMatrix{Float64},
        t::Int64, rowstart::Int64)
    opts = mtile.model.options
    u_hist = get(opts, :hsi_u_history, "none")
    x_hist = get(opts, :hsi_x_history, "fresh")
    am2 = get(opts, :hsi_scheme, "ai2s") == "am2"
    vars = mtile.model.grid_params.vars
    n = size(mtile.hacdot_n, 1)
    rows = rowstart:rowstart + n - 1
    # u channel: stored applied increment under "stored" (default)/AM2; "none"
    # leaves the u leg implicit-only (θ = 1 — measured to over-damp resolved
    # circulations by tens of percent at Co_h 0.6 on bf02, so pair it only
    # with configurations that tolerate the damping).
    if u_hist == "stored" || am2
        dst = view(mtile.hacdot_n, :, vars["u"])
        copyto!(dst, view(incr, rows, 1))
        t == 2 && copyto!(view(mtile.hacdot_nm1, :, vars["u"]), dst)
    end
    # p/ρ_d/ρ_t/E_t channels: under :hsi_x_history = "stored" (default) the
    # histories are the sweep's APPLIED increments — the same self-consistency
    # the vertical w leg relies on. The "fresh" alternative (staged in
    # mc_driver! from the u_x grid slot) re-evaluates the leg through the
    # refit chain, whose l_q-filtered difference from the applied operator is
    # a grid-scale residual under the explicit AI2* weights (the slow
    # oblique-mode leak measured in model_tests/hsi_growth_probe.jl).
    if x_hist == "stored"
        for (plane, var) in ((2, "p"), (3, "rho_d"), (4, "rho_t"), (5, "E_t"))
            dst = view(mtile.hacdot_n, :, vars[var])
            copyto!(dst, view(incr, rows, plane))
            t == 2 && copyto!(view(mtile.hacdot_nm1, :, vars[var]), dst)
        end
    end
    # Cross-term compensation field G (plane 6), parked in hacdot_n's otherwise
    # unused w column (the horizontal legs never touch w); consumed by
    # horizontal_si_history! as +ν²·G on the w predictor. DEFAULT OFF: as a
    # LAGGED EXPLICIT source the compensation is itself unstable wherever
    # ν²|BA| ≳ 1 (grid-scale oblique modes at Co_x·Co_z ≳ 1 — NaN within
    # 200 s on the Co_h 3 probe). The consistent repair must be the true
    # delta-form Douglas–Gunn split, with the correction inside both implicit
    # factors — which restructures the vertical solve (see the findings-note
    # addendum). Lever kept for low-Courant experiments only.
    if get(opts, :hsi_cross_comp, false) === true
        copyto!(view(mtile.hacdot_n, :, vars["w"]), view(incr, rows, 6))
    end
    return nothing
end
