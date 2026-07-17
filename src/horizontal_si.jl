# ─────────────────────────────────────────────────────────────────────────────
# Horizontal acoustic semi-implicit sweep for the mc (pressure-reference) sets:
# the delta-form Douglas–Gunn (approximate-factorization) companion to the
# per-column vertical solve in `semiimplicit_adjustment_p`.
#
# The reference-linear HORIZONTAL acoustic pair
#
#     ∂u/∂t  = −(1/ρ̄_t) ∂x p′,      ∂p′/∂t = −Pξ̄(z) ρ̄_t ∂x u
#     slaved: ∂ρ_t′/∂t = −ρ̄_t ∂x u,  ∂ρ_d′/∂t = −ρ̄_d ∂x u,
#             ∂E_t′/∂t = −(Ē_t+p̄) ∂x u
#
# is integrated with the same AI2* off-centering as the vertical legs
# (ν = 1.25·ts; 0.5·ts on the AM2 first step). Writing the combined implicit
# system for the STEP INCREMENT δ = X^{n+1} − Xⁿ,
#
#     (I − νL) δ = R,    R = [AB3 remainder] + [AI2* explicit histories] + ν·L(Xⁿ),
#
# the two-factor delta-form split (Douglas & Gunn 1964; Beam & Warming 1978)
#
#     (I − νB) δ¹ = R          (z factor)
#     (I − νA) δ  = δ¹         (x factor)
#
# has splitting defect ν²BA·δ = O(ts³) per step — 2nd-order cumulative, with the
# correction regularized by both implicit factors. The Phase-1 SEQUENTIAL
# composition (each factor solving for the full state) instead left the defect
# ν²BA·X = O(ts²)/step ⇒ O(ts) cumulative on all pressure-coupled modes:
# measured as 8–20% resolved-flow damping on the BF02 bubble plus a slow
# non-normal leak, and confirmed by von Neumann analysis (max|G| = 1.019 at
# oblique Co ≈ 0.5, over-damping 0.85 at Co 2). The delta form is neutrally
# stable (max|G| = 1.0 to 1e-8) over Co_x × Co_z ∈ [0, 30]² for both u-history
# variants — indistinguishable from the unsplit AI2* solve
# (reference/horizontal_si_phase2_findings.md; the von Neumann script is
# reproduced in the phase-2 commit message). NOTE: Ikawa (1988, JMSJ) was
# checked per the house rule and contains NO ADI-split SI (its E-HI-VI is a
# full 2-D implicit elliptic solve — the spirit of the variant-1 fallback);
# the two-factor delta form stands on Douglas–Gunn/Beam–Warming plus the
# direct analysis above. (Caution recorded: the THREE-factor delta form is
# known unstable for pure wave systems — keep future 3-D compositions at two
# factors, e.g. the per-wavenumber (r, λ) solve of the RLR phase.)
#
# Implementation shape (the algebra that keeps the vertical solve untouched):
# with Y ≡ Xⁿ + δ¹, the z factor is identical to
#
#     (I − νB) Y = X* + ν·A(Xⁿ),
#
# i.e. the EXISTING vertical state solve applied to a predictor augmented by
# the pointwise horizontal linear term — `semiimplicit_adjustment_p` is not
# restructured at all (ν·B(Xⁿ) cancels analytically), so the B-only special
# case recovers the vertical-only scheme exactly and BOTH vertical coefficient
# paths (reference-profile and `:state_dependent_si`) are handled by
# construction. The predictor addition is applied per column in
# `horizontal_si_history!` (from the `u_x`/`pp_x`-slot staging in `mc_driver!`);
# the x factor is this patch-level sweep, which solves for the increment: the
# per-physical-level u-form Helmholtz
#
#     (I − Δτ² Pξ̄(z_k) ∂xx) δ_u = δ¹_u − (Δτ/ρ̄_t(z_k)) ∂x δ¹_p
#
# with δ¹ read as the COEFFICIENT DIFFERENCE between the merged post-column
# patch state Y and the previous step's end-of-step coefficients Xⁿ (snapshot
# kept in this struct), and the recoveries
#
#     δ_p  = δ¹_p  − Δτ Pξ̄ ρ̄_t ∂x δ_u      (+ the ρ_d′/ρ_t′/E_t′ analogues),
#
# written back as B-coefficient increments δ − δ¹ through the model's forward
# chain. Operator consistency across the AI2* time levels (THE stability
# requirement — see tc/SI_VERTICAL_CEILING.md) is by construction:
#   • the sweep reads δ¹ through EXACTLY the model's read chain (per-z_b-block
#     i-direction SA fit → evaluate, then per-i-point vertical SA fit →
#     evaluate — the `gridTransform` chain), the same chain that produces the
#     `u_x`/`pp_x` grid slots the pointwise staging uses;
#   • the boundary rows are homogeneous on the increment (uⁿ and u^{n+1} both
#     satisfy the u side-wall BCs), so the Dirichlet row treatment of the
#     Phase-1 solve carries over unchanged;
#   • the predictor addition AND the stored histories use the operator feed —
#     A(X^{n+1}) evaluated by the sweep from the final coefficients through
#     its own chain, which by linearity equals the operator the step applied
#     (`horizontal_si_load_increment!`). One discrete chain for every
#     appearance of A: staging the predictor addition pointwise instead
#     leaves (A_grid − A_sweep)(Xⁿ) as a once-per-step O(ν·X) grid-scale
#     forcing — measured unstable, e-fold ≈ 20 s at Co_h 3.
#
# Phase 2 scope: Cartesian XZ (`RiRk`) single patch, Dirichlet or Neumann u
# side walls. The cylindrical geometries take the p′-form per-(level,
# wavenumber) solves in later phases.
# ─────────────────────────────────────────────────────────────────────────────

"""
Precomputed patch-level data for the horizontal semi-implicit sweep: i-direction
Galerkin solve data for u's and p's side-wall bases, the per-physical-level
Helmholtz factorizations (production `Δτ = 1.25·ts` set and the first-step
`0.5·ts` AM2 set), the z-only reference coefficient profiles at the kDim mish
levels, the previous step's end-of-step u/p patch B coefficients (the delta-form
baseline Xⁿ), and preallocated level/transform work arrays. Built once by
[`create_horizontal_solve_data`](@ref), which also takes the initial snapshot.
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
    fact_first::FB          # per-level factorizations, Δτ = 0.5·ts (t == 1 / AM2)
    Pxi::Vector{Float64}    # Pξ̄(z_k), local reference sound speed squared
    rho_tbar::Vector{Float64}
    rho_dbar::Vector{Float64}
    etp_bar::Vector{Float64}    # Ē_t + p̄
    # Delta-form baseline: previous end-of-step patch B coefficients of u and p
    prev_u::Vector{Float64}     # (b_iDim·b_kDim)
    prev_p::Vector{Float64}     # (b_iDim·b_kDim)
    # u-row predictor field A_u(Xⁿ) = −(1/ρ̄_t)∂x p′ⁿ at the physical points,
    # re-evaluated at the END of each sweep from the final p coefficients
    # (state-anchored; an accumulated stored-increment recursion here is a free
    # integrator coupled to the state — measured secular growth at every Co_h).
    # Applied master-side as +ν·au in the sweep rhs (B has no u row, so the
    # u-leg's Douglas–Gunn predictor term need not ride the tile predictors).
    au::Matrix{Float64}         # (iDim, kDim)
    initialized::Base.RefValue{Bool}
    # Work arrays (single-threaded master-side sweep)
    dcoef::Vector{Float64}  # (b_iDim·b_kDim) coefficient-difference staging
    ulev::Matrix{Float64}   # (iDim, kDim) δ¹_u at physical points
    pxlev::Matrix{Float64}  # (iDim, kDim) ∂x δ¹_p at physical points
    dlev::Array{Float64,3}  # (iDim, kDim, 5) write-back δ−δ¹: u, p, ρ_d, ρ_t, E_t
    zbuf::Matrix{Float64}   # (iDim, b_kDim) i-evaluated vertical B coefficients
    rhs::Vector{Float64}    # (iDim)
    load::Vector{Float64}   # (b_iDim)
    acoef::Vector{Float64}  # (b_iDim)
    tempcb::Matrix{Float64} # (b_kDim, iDim) forward-transform staging
end

# Correction plane order in `dlev` (and the write-back loop): the var each
# plane belongs to is looked up from these fields at use sites.
const _HSI_PLANES = (:u_index, :p_index, :rhod_index, :rhot_index, :et_index)

"""
    create_horizontal_solve_data(patch, model, Pxi_prof, rho_tbar, rho_dbar,
                                 etp_bar, init_spectral)

Build the [`HorizontalSolveData`](@ref) for `patch`. The reference profiles are
passed in (kDim vectors at the physical mish levels) so the caller can source
them from a worker's `ModelTile` (`model_loop`) or a local one (tests);
`init_spectral` is the patch B-coefficient state at t = 0 (`sharedSpectral` in
`integrate_model`, `patch.spectral` in tests), snapshotted as the delta-form
baseline for the first sweep. XZ (`RiRk`) only in Phase 2; the option is
rejected for other geometries.
"""
function create_horizontal_solve_data(patch::AbstractGrid, model::ModelParameters,
        Pxi_prof::AbstractVector{Float64}, rho_tbar::AbstractVector{Float64},
        rho_dbar::AbstractVector{Float64}, etp_bar::AbstractVector{Float64},
        init_spectral::AbstractMatrix{Float64})

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

    hsd = HorizontalSolveData(u_index, p_index, rhod_index, rhot_index, et_index,
        du, dp, (_is_dirichlet(bcl), _is_dirichlet(bcr)), fact, fact_first,
        collect(Pxi_prof), collect(rho_tbar), collect(rho_dbar), collect(etp_bar),
        zeros(b_iDim * b_kDim), zeros(b_iDim * b_kDim), zeros(iDim, kDim),
        Ref(false), zeros(b_iDim * b_kDim),
        zeros(iDim, kDim), zeros(iDim, kDim), zeros(iDim, kDim, 5),
        zeros(iDim, b_kDim), zeros(iDim), zeros(b_iDim), zeros(b_iDim),
        zeros(b_kDim, iDim))
    horizontal_si_snapshot!(hsd, init_spectral, patch)
    # Seed the u-row predictor field: A_u(X⁰) = −(1/ρ̄_t)∂x p′⁰ (identically
    # zero at a resting start).
    _hsi_to_levels!(hsd.pxlev, hsd, view(init_spectral, :, p_index), patch, p_index, 1)
    for k in 1:kDim
        @inbounds for i in 1:iDim
            hsd.au[i, k] = -hsd.pxlev[i, k] / hsd.rho_tbar[k]
        end
    end
    return hsd
end

"""
    horizontal_si_snapshot!(hsd, spectral, patch)

Record the delta-form baseline for the next sweep: `spectral`'s u and p patch
B-coefficient columns ROUND-TRIPPED through the model's eval + fit chain
(`P = fit ∘ eval`, the read chain of `_hsi_to_levels!` followed by the forward
chain of `_hsi_fit!`). The baseline must be the state AS THE NEXT STEP CARRIES
IT — the next merged state is `Y = P(Xⁿ) + fit(increments)`, so differencing
against the raw coefficients would leak `(P − I)Xⁿ`, the l_q refit residual of
the carried state (grid-scale, STATE-amplitude), into the sweep, where the
acoustic coupling re-injects it across variables every step (measured: growth
at Co_h 1.5–3 in every history/predictor variant, 2026-07-17). With the
round-tripped baseline, `δ¹ = Y − P(Xⁿ) = fit(increments)` exactly, and the
refit damping of the carried state stays out of the solve — identical to its
flag-off role. Called by [`create_horizontal_solve_data`](@ref) at t = 0 and by
[`horizontal_si_correct!`](@ref) after each write-back.
"""
function horizontal_si_snapshot!(hsd::HorizontalSolveData, spectral::AbstractMatrix{Float64},
        patch::AbstractGrid)
    _hsi_to_levels!(hsd.ulev, hsd, view(spectral, :, hsd.u_index), patch, hsd.u_index, 0)
    _hsi_fit!(hsd.prev_u, hsd, patch, hsd.u_index, hsd.ulev)
    _hsi_to_levels!(hsd.ulev, hsd, view(spectral, :, hsd.p_index), patch, hsd.p_index, 0)
    _hsi_fit!(hsd.prev_p, hsd, patch, hsd.p_index, hsd.ulev)
    hsd.initialized[] = true
    return nothing
end

# Read chain: a patch B-coefficient column → values (or ∂x, via `deriv`) at the
# physical mish points, exactly mirroring `gridTransform`'s per-variable path so
# the sweep sees the same discrete state the grid slots see. `v` selects the
# variable's bases; `coeffs` is the (b_iDim·b_kDim) coefficient vector — the
# merged state or the delta-form coefficient difference.
function _hsi_to_levels!(out::Matrix{Float64}, hsd::HorizontalSolveData,
        coeffs::AbstractVector{Float64}, patch::AbstractGrid, v::Int, deriv::Int)
    iDim = patch.params.iDim; kDim = patch.params.kDim
    b_iDim = patch.params.b_iDim; b_kDim = patch.params.b_kDim
    zbuf = hsd.zbuf
    for z in 1:b_kDim
        isp = patch.ibasis.data[z, v]
        r1 = (z - 1) * b_iDim
        @inbounds for i in 1:b_iDim
            isp.b[i] = coeffs[r1 + i]
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

# Forward chain: a field at the physical mish points → its B-coefficient fit,
# mirroring `spectralTransform`'s per-variable path (per-i vertical SB, then
# per-z_b i-direction SB) — the same discrete fit `calcTendency` applies to the
# advanced state. Writes the coefficient vector; used for the round-tripped
# delta-form baseline (see horizontal_si_snapshot!).
function _hsi_fit!(coeffs::AbstractVector{Float64}, hsd::HorizontalSolveData,
        patch::AbstractGrid, v::Int, field::AbstractMatrix{Float64})
    iDim = patch.params.iDim; kDim = patch.params.kDim
    b_iDim = patch.params.b_iDim; b_kDim = patch.params.b_kDim
    tempcb = hsd.tempcb
    kcol = patch.kbasis.data[v]
    for i in 1:iDim
        @inbounds for z in 1:kDim
            kcol.uMish[z] = field[i, z]
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
            coeffs[r1 + i] = isp.b[i]
        end
    end
    return coeffs
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

Apply the delta-form implicit horizontal acoustic correction to the merged patch
B coefficients `spectral` (the `sharedSpectral` array in `model_loop`, or
`patch.spectral` in single-process tests). Reads the z-implicit step increment
δ¹ as the coefficient difference between `spectral` and the previous end-of-step
snapshot, solves the per-level u-form Helmholtz for the final increment δ_u,
updates u/p′/ρ_d′/ρ_t′/E_t′ with the B-coefficient increments δ − δ¹, and
re-snapshots. Returns an `(npts × 5)` matrix at the patch physical points (row
order `(i−1)·kDim + k`): the operator feed `A(X^{n+1})` per leg
(u, p, ρ_d, ρ_t, E_t), evaluated through the sweep's own chain — distributed
to the tiles by [`horizontal_si_load_increment!`](@ref) at the top of the next
step as the delta-form predictor-addition field and the stored-history values.
"""
function horizontal_si_correct!(spectral::AbstractMatrix{Float64}, patch::AbstractGrid,
        model::ModelParameters, hsd::HorizontalSolveData, t::Int64)

    hsd.initialized[] || error(
        "horizontal_si_correct! called before the delta-form baseline snapshot; " *
        "create_horizontal_solve_data takes the initial spectral state")
    iDim = patch.params.iDim; kDim = patch.params.kDim
    # options[:hsi_scheme]: "ai2s" (default) = the off-centered AI2* weights of
    # the vertical solve; "am2" = trapezoidal (the t == 1 branch every step) on
    # the horizontal channel only — 2nd-order, single history level, neutrally
    # stable for pure-horizontal modes (an A/B lever; the vertical channel
    # keeps AI2* regardless).
    am2 = get(model.options, :hsi_scheme, "ai2s") == "am2"
    ts_term = (t == 1 || am2) ? 0.5 * model.ts : 1.25 * model.ts
    facts = (t == 1 || am2) ? hsd.fact_first : hsd.fact
    # Debug lever (see horizontal_si_history!): drop the u-row predictor term.
    au_w = get(model.options, :hsi_dg_predictor, true) === false ? 0.0 : ts_term

    # δ¹ from the coefficient differences against the previous end-of-step
    # snapshot: the full pre-sweep step increment (explicit remainder, both
    # dimensions' histories, the ν·A(Xⁿ) predictor addition, the vertical
    # solve, and the refit of the carried state — everything the step did).
    hsd.dcoef .= view(spectral, :, hsd.u_index) .- hsd.prev_u
    _hsi_to_levels!(hsd.ulev, hsd, hsd.dcoef, patch, hsd.u_index, 0)
    hsd.dcoef .= view(spectral, :, hsd.p_index) .- hsd.prev_p
    _hsi_to_levels!(hsd.pxlev, hsd, hsd.dcoef, patch, hsd.p_index, 1)

    du = hsd.du
    dlev = hsd.dlev
    for k in 1:kDim
        # rhs = δ¹_u + Δτ·A_u(Xⁿ) − (Δτ/ρ̄_t) ∂x δ¹_p, Galerkin load M0ᵀW·rhs;
        # Dirichlet rows zeroed — homogeneous on the increment (both time levels
        # satisfy the BC). The ν·A_u(Xⁿ) predictor addition of the u row lives
        # HERE (master-side, the state-anchored `au` field re-evaluated each
        # sweep) — B has no u row, so it need not ride the tile predictors.
        @inbounds for i in 1:iDim
            hsd.rhs[i] = du.W[i] * (hsd.ulev[i, k] + au_w * hsd.au[i, k] -
                                    (ts_term / hsd.rho_tbar[k]) * hsd.pxlev[i, k])
        end
        mul!(hsd.load, du.M0', hsd.rhs)
        if hsd.dirichlet[1]; hsd.load[1] = 0.0; end
        if hsd.dirichlet[2]; hsd.load[end] = 0.0; end
        ldiv!(hsd.acoef, facts[k], hsd.load)

        # Recoveries: δ_u from the solved coefficients; the p/ρ_d/ρ_t/E_t legs
        # from the strong derivative ∂x δ_u with the z-only reference
        # coefficients. The write-back fields are δ − δ¹ per leg.
        cp = ts_term * hsd.Pxi[k] * hsd.rho_tbar[k]
        ct = ts_term * hsd.rho_tbar[k]
        cd = ts_term * hsd.rho_dbar[k]
        ce = ts_term * hsd.etp_bar[k]
        @inbounds for i in 1:iDim
            dunew = 0.0
            dxu = 0.0
            for j in 1:size(du.M0, 2)
                dunew += du.M0[i, j] * hsd.acoef[j]
                dxu += du.M1[i, j] * hsd.acoef[j]
            end
            dlev[i, k, 1] = dunew - hsd.ulev[i, k]
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

    # New delta-form baseline: the ROUND-TRIPPED post-correction coefficients
    # (see horizontal_si_snapshot! — the state as the next step carries it).
    horizontal_si_snapshot!(hsd, spectral, patch)

    # Operator feed for the workers, (i−1)·kDim + k row order: A(X^{n+1})
    # evaluated through the sweep's own chain from the final coefficients —
    # next step's predictor-addition field ν·A(Xⁿ) (planes 2–5 tile-side;
    # plane 1's role is played by `au` master-side) and the stored-history
    # values (all planes). The u-row field `au` is re-evaluated here from the
    # final p coefficients (state-anchored).
    _hsi_to_levels!(hsd.ulev, hsd, view(spectral, :, hsd.u_index), patch, hsd.u_index, 1)
    _hsi_to_levels!(hsd.pxlev, hsd, view(spectral, :, hsd.p_index), patch, hsd.p_index, 1)
    incr = zeros(iDim * kDim, 5)
    @inbounds for i in 1:iDim, k in 1:kDim
        row = (i - 1) * kDim + k
        dxu = hsd.ulev[i, k]
        hsd.au[i, k] = -hsd.pxlev[i, k] / hsd.rho_tbar[k]
        incr[row, 1] = hsd.au[i, k]
        incr[row, 2] = -hsd.Pxi[k] * hsd.rho_tbar[k] * dxu
        incr[row, 3] = -hsd.rho_dbar[k] * dxu
        incr[row, 4] = -hsd.rho_tbar[k] * dxu
        incr[row, 5] = -hsd.etp_bar[k] * dxu
    end
    return incr
end

"""
    horizontal_si_history!(mtile, colstart, colend, t)

Per-column horizontal-SI star-state work between `explicit_timestep` and the
vertical solve: apply the explicit AI2* history levels of the horizontal
acoustic legs (`ts·(0.75·hacdot_{n−1} − hacdot_n)`; `+0.5·ts·hacdot_n` on the
AM2 first step) and roll the history, then apply the delta-form Douglas–Gunn
predictor addition `+ν·A(Xⁿ)` to all five legs — the term that puts the
cross-dimension correction INSIDE the z-implicit factor, converting the
factorization defect from O(ts²) to O(ts³) per step. The A(Xⁿ) field is the
sweep-chain operator feed loaded into `hsi_a_n` at the top of the step; the
pointwise grid-slot staging from `mc_driver!` is only the t == 1 fallback.
"""
function horizontal_si_history!(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)
    vars = mtile.model.grid_params.vars
    ts = mtile.model.ts
    opts = mtile.model.options
    am2 = get(opts, :hsi_scheme, "ai2s") == "am2"
    u_hist = get(opts, :hsi_u_history, "none")
    x_hist = get(opts, :hsi_x_history, "fresh")
    S = @inbounds mtile.mc_scratch[Threads.threadid()]
    ts_term = (t == 1 || am2) ? 0.5 * ts : 1.25 * ts

    # First-step stored-mode seeding: no sweep has run yet, so the applied
    # operator IS the pointwise staging — write it before the application so
    # the AM2 first step carries its explicit Lⁿ half (fresh mode staged
    # hacdot_n in mc_driver! already).
    if t == 1
        if u_hist == "stored" || am2
            view(mtile.hacdot_n, colstart:colend, vars["u"]) .= S.hsi_au
        end
        if x_hist == "stored"
            view(mtile.hacdot_n, colstart:colend, vars["p"]) .= S.hsi_ap
            view(mtile.hacdot_n, colstart:colend, vars["rho_d"]) .= S.hsi_ad
            view(mtile.hacdot_n, colstart:colend, vars["rho_t"]) .= S.hsi_at
            view(mtile.hacdot_n, colstart:colend, vars["E_t"]) .= S.hsi_ae
        end
    end

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

    # Delta-form predictor addition ν·A(Xⁿ): the star state handed to the
    # vertical solve becomes X* + ν·A(Xⁿ), making that solve the exact z factor
    # (I − νB)(Xⁿ + δ¹) = X* + ν·A(Xⁿ) of the Douglas–Gunn split — no change
    # inside semiimplicit_adjustment_p itself (ν·B(Xⁿ) cancels analytically).
    # The A(Xⁿ) field is the SWEEP-CHAIN evaluation from the previous step's
    # feed (hsi_a_n, loaded by horizontal_si_load_increment!) — the pointwise
    # grid-slot staging is only the t == 1 fallback (no sweep has run yet;
    # one AM2 half-step of chain inconsistency, transient by construction).
    # The u row is NOT touched here: its addition is applied master-side in
    # the sweep rhs through the stored-applied-increment field `au` (the u leg
    # is the weak-solve leg — see the hsd.au field note).
    # Debug lever :hsi_dg_predictor => false disables the additions to isolate
    # the delta-sweep machinery from the predictor-addition chain.
    get(opts, :hsi_dg_predictor, true) === false && return nothing
    if t == 1
        view(mtile.var_np1, colstart:colend, vars["p"]) .+= ts_term .* S.hsi_ap
        view(mtile.var_np1, colstart:colend, vars["rho_d"]) .+= ts_term .* S.hsi_ad
        view(mtile.var_np1, colstart:colend, vars["rho_t"]) .+= ts_term .* S.hsi_at
        view(mtile.var_np1, colstart:colend, vars["E_t"]) .+= ts_term .* S.hsi_ae
    else
        view(mtile.var_np1, colstart:colend, vars["p"]) .+=
            ts_term .* view(mtile.hsi_a_n, colstart:colend, 2)
        view(mtile.var_np1, colstart:colend, vars["rho_d"]) .+=
            ts_term .* view(mtile.hsi_a_n, colstart:colend, 3)
        view(mtile.var_np1, colstart:colend, vars["rho_t"]) .+=
            ts_term .* view(mtile.hsi_a_n, colstart:colend, 4)
        view(mtile.var_np1, colstart:colend, vars["E_t"]) .+=
            ts_term .* view(mtile.hsi_a_n, colstart:colend, 5)
    end
    return nothing
end

"""
    horizontal_si_load_increment!(mtile, incr, t, rowstart)

Load this tile's window of the sweep's operator feed `A(Xⁿ)` (from
[`horizontal_si_correct!`](@ref) at the end of step `t−1`, row order
`(i−1)·kDim + k`, plane order u/p/ρ_d/ρ_t/E_t):

  - into `hsi_a_n` — the delta-form predictor-addition field ν·A(Xⁿ) that
    [`horizontal_si_history!`](@ref) applies before the vertical solve, in
    EVERY history mode;
  - into the `hacdot_n` AI2* history channels under the stored options, where
    it is the horizontal implicit operator the previous step actually applied
    (the sweep-chain evaluation at the carried state — exact by linearity):
      · `options[:hsi_u_history] = "stored"` (or the AM2 scheme): the u leg.
        The default `"none"` leaves the u history zero — the u leg is then
        integrated with its implicit levels alone (θ = 1 at ν):
        parameter-free, damping only the horizontal acoustic modes the SI
        exists to suppress.
      · `options[:hsi_x_history] = "stored"`: the p/ρ_d/ρ_t/E_t legs. The
        `"fresh"` default re-evaluates them each step from the `u_x` grid
        slot in `mc_driver!`, whose chain difference from the applied
        operator sits only under the net explicit AI2* weights (the A/B
        lever of the Phase-2 measurement round).

`rowstart` is the tile's first physical row in patch numbering.
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
    mtile.hsi_a_n .= view(incr, rows, :)
    if u_hist == "stored" || am2
        copyto!(view(mtile.hacdot_n, :, vars["u"]), view(incr, rows, 1))
    end
    if x_hist == "stored"
        copyto!(view(mtile.hacdot_n, :, vars["p"]), view(incr, rows, 2))
        copyto!(view(mtile.hacdot_n, :, vars["rho_d"]), view(incr, rows, 3))
        copyto!(view(mtile.hacdot_n, :, vars["rho_t"]), view(incr, rows, 4))
        copyto!(view(mtile.hacdot_n, :, vars["E_t"]), view(incr, rows, 5))
    end
    return nothing
end
