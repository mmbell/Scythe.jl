# ─────────────────────────────────────────────────────────────────────────────
# Exact (unsplit) 2-D acoustic semi-implicit for the mc (pressure-reference)
# sets — variant 1 of reference/exact_si_plan.md, implemented per the amended
# reference/exact_si_derivation.md.
#
# The reference-linearized XZ acoustic pair
#
#     ∂u/∂t = −(1/ρ̄_t) ∂x p′,   ∂φ/∂t = −∂z p′,   φ = ρ̄_t w
#     ∂p′/∂t = −Pξ̄(z) ( ρ̄_t ∂x u + ∂z φ ),   slaved ρ_d′/ρ_t′/E_t′ flux forms
#
# is integrated with the AI2* weights by ONE unsplit patch-level solve per
# step: the P⁻¹-scaled weighted-mass p′-form Helmholtz (derivation §1–2)
#
#     ( 1/Pξ̄ − Δτ²(∂xx + ∂zz) ) p′^{n+1} = p′*/Pξ̄ − Δτ( ρ̄_t ∂x u* + ∂z φ* )
#
# assembled on the patch B-coefficient tensor basis as
#
#     A = (Mzw ⊗ Mx) + Δτ²( Mz ⊗ Sx + Sz ⊗ Mx ),   Mzw = M0ᵀ(W/Pξ̄)M0,
#
# factorized once at setup (dense in Stage 1; banded is Stage 3), and EVERY
# fast leg is recovered from the one solved coefficient set through the
# assembly's own operators (u, w by M1x/M1z of p̂; p by M0; the slaved legs by
# the pointwise divergence identity div^{n+1} = −δp/(Δτ Pξ̄) plus the fitted
# φ^{n+1}). When `options[:exact_si]` is on, the per-column vertical solve
# (`semiimplicit_adjustment_p`) is BYPASSED — this solve replaces it.
#
# Why all legs must come from the one solve (Stage-1 measured lesson): every
# composite that recovered some legs through the per-column φ-solve while u
# followed the 2-D solve (or the final p directly, or its own per-level
# Helmholtz fed through the z-solve) left an O(state) fraction of the fast
# operator staged outside a solve — the ε-chain mechanism of the DG round in a
# new form — and grew at e-fold 12–40 s for Co_h ≥ 3 × Co_z ≈ 2 despite
# symbol-level exactness. The all-from-one-solve form is the consistent-weak
# solve+recovery composition that von Neumann part 3(a) proved ε-INSENSITIVE
# (flat max|G| ≤ 1.003 at every chain imperfection ε ∈ [0, 0.2], Co ≤ 30²).
#
# Step structure (model_loop / the single-tile harnesses):
#   phase A (per column, `mc_driver!` early exit): expdot remainder (both
#     directions), fresh AI2* histories (vertical impdot + horizontal hacdot),
#     AB3 predictor, ALL explicit history levels applied → X*; publish
#     (u, w, p′)* to the patch-level solve.
#   patch solve (master): `exact_si_solve!` → increment planes for all legs.
#   phase B (per column, `exact_si_apply_column!`): apply the increments,
#     store the w-leg applied-increment history, implicit vertical diffusion.
#
# Options:
#   :exact_si          — master switch (opt-in; RiRk XZ only in Stage 1)
#   :xsi_u_history     — "stored" (default; the u-leg AI2* history is the
#                        applied increment δu/Δτ — full 2nd-order, the
#                        ε-insensitive fresh-history composition of VN 3(a)) or
#                        "none" (implicit-only, θ = 1: stable but leaves an ~11%
#                        max|u| bias on the BF02 bubble — the rejected A/B arm)
#   :xsi_lid_rows      — "strong" (default): the first/last vertical
#                        coefficient planes carry the ∂z-value rows
#                        ∂z p̂ = φ*(bnd)/Δτ; "natural": weak by-parts flux loads
#   :exact_si_ax0      — TEST: assemble/solve with the ∂xx block and every
#                        x-term removed (δu ≡ 0) — the vertical-equivalence
#                        comparison against the per-column φ-solve
#   :exact_si_zero_x   — TEST: bypass the exact-SI machinery entirely; phase B
#                        falls back to the per-column vertical solve, so the
#                        path must be bitwise the production vertical-only path
# Errors at setup: :exact_si + :state_dependent_si, :exact_si +
# :horizontal_semiimplicit, non-RiRk geometry, nested runs.
# ─────────────────────────────────────────────────────────────────────────────

# Increment-plane layout of the solve feed (npts × XSI_NPLANES, row order
# (i−1)·kDim + k): applied increments of (u, w, p′, ρ_d′, ρ_t′, E_t′), plane 7
# the u-leg history δu/Δτ (consumed under :xsi_u_history = "stored").
const XSI_NPLANES = 7

"""
Precomputed patch-level data for the exact (unsplit) 2-D acoustic solve: the
factorizations of the weighted-mass Helmholtz tensor operator (production
`Δτ = 1.25·ts` and first-step `0.5·ts`), the i/k Galerkin blocks, reference
profiles, spline scratch objects for the read chains, and work arrays. Built
once by [`create_exact_si_data`](@ref).
"""
struct ExactSIData{FA, FB, SU, SW, SP, KP, KU}
    u_index::Int
    w_index::Int
    p_index::Int
    rhod_index::Int
    rhot_index::Int
    et_index::Int
    fact::FA               # 2-D factorization, Δτ = 1.25·ts
    fact_first::FB         # 2-D factorization, Δτ = 0.5·ts (t == 1)
    M0x::Matrix{Float64}   # (iDim, b_iDim) i-basis at mish (p's basis)
    M1x::Matrix{Float64}
    Wx::Vector{Float64}
    Nbx::Matrix{Float64}   # (2, b_iDim) boundary-value rows
    M0z::Matrix{Float64}   # (kDim, b_kDim)
    M1z::Matrix{Float64}
    Wz::Vector{Float64}
    Nbz::Matrix{Float64}
    dirichlet_u::Tuple{Bool, Bool}
    strong_lid::Bool
    ax0::Bool              # ∂xx block and all x-terms disabled (test lever)
    axisym::Bool           # axisymmetric radial metric (r-weighted blocks/load)
    rmet::Vector{Float64}  # radial coordinate at the iDim mish points (axisym)
    r_wall_l::Float64      # radial weight of the left (r=iMin/axis) wall load
    r_wall_r::Float64      # radial weight of the right (r=iMax) wall load
    # Reference profiles at the kDim mish levels
    Pxi::Vector{Float64}
    rho_tbar::Vector{Float64}
    c_d::Vector{Float64}   # ρ̄_d/ρ̄_t
    c_d_z::Vector{Float64}
    c_e::Vector{Float64}   # (Ē_t+p̄)/ρ̄_t
    c_e_z::Vector{Float64}
    # Spline scratch objects (master-side read chains)
    isp_u::SU              # u's i-basis clone (per-level fits of u*)
    isp_p::SP              # p's i-basis clone (x-fits of the lid data)
    kcol_w::SW             # w's column-basis clone (per-column fits of φ*)
    kcol_p::KP             # p's column basis (p′* round trip + φ* lid values)
    kcol_u::KU             # u's column basis (u* round trip)
    # Work arrays (master-side, single-threaded)
    U::Matrix{Float64}     # (iDim, kDim) u* (round-tripped)
    PH::Matrix{Float64}    # φ* = ρ̄_t w* (raw)
    PHR::Matrix{Float64}   # φ* through the w-basis fit (round-tripped)
    PR::Matrix{Float64}    # p′* (round-tripped)
    W0::Matrix{Float64}    # w* (raw, the δw baseline)
    ux::Matrix{Float64}    # ∂x u*
    phiz::Matrix{Float64}  # ∂z φ*
    G::Matrix{Float64}     # quadrature-weighted load field
    L::Matrix{Float64}     # (b_iDim, b_kDim) Galerkin load
    Q::Matrix{Float64}     # solved coefficients
    P0::Matrix{Float64}    # p̂ at mish
    px::Matrix{Float64}    # ∂x p̂ at mish
    pz::Matrix{Float64}    # ∂z p̂ at mish
    ubl::Vector{Float64}   # u* at the left wall (kDim)
    ubr::Vector{Float64}
    phib::Vector{Float64}  # φ* at the surface (iDim)
    phit::Vector{Float64}
    zcol::Vector{Float64}  # (kDim) column scratch
end

"""
    exact_si_is_axisym(model) -> Bool

Whether `options[:exact_si]` runs the axisymmetric radial-metric path (the
`moist_compressible_axisym` set on the RiRk grid, Stage 3) rather than the XZ
Cartesian path. The two share the RiRk grid and differ only by the radial
`r`-weight on the Galerkin blocks and load — the sole geometry switch the
solve needs. RLR (`moist_compressible_RLR`, `"RLR"` geometry) is Stage 4 and is
rejected upstream by [`validate_exact_si_options`](@ref).
"""
@inline exact_si_is_axisym(model::ModelParameters) =
    model.equation_set == "moist_compressible_axisym"

"Whether `options[:exact_si]` runs the RLR per-wavenumber Fourier path (exact_si_rlr.jl)."
@inline exact_si_is_rlr(model::ModelParameters) =
    model.equation_set == "moist_compressible_RLR" ||
    model.grid_params.geometry == "RLR"

# A RIGID Dirichlet u wall (real prescribed value: u = 0 on the axis and the
# outer domain wall) — as opposed to a nested-interface FixedBC, whose value
# slots are NaN (R3X, pinned by the parent). Only the rigid wall contributes the
# exact_si boundary load; the R3X interface and the parent-collar Natural edge
# are homogeneous-natural (∂r p′ = 0 ⇒ δu = 0, the freeze-parent condition).
@inline _is_rigid_dirichlet(bc) = bc.u !== nothing && !isnan(bc.u)

"""
    validate_exact_si_options(model)

Setup-time validation of `options[:exact_si]` (called from `createModelTile`):
pressure-reference set on the RiRk grid — XZ Cartesian or axisymmetric r–z
(Stage 3) — and the structurally incompatible flags error loudly. The RLR
cylindrical set is Stage 4 and still errors here.
"""
function validate_exact_si_options(model::ModelParameters)
    get(model.options, :exact_si, false) === true || return nothing
    uses_pressure_reference(model.equation_set) || error(
        "options[:exact_si] requires a moist_compressible (pressure-reference) " *
        "equation set")
    if exact_si_is_rlr(model)
        # RLR is WORK IN PROGRESS: the n=0 path is verified but the coupled n≥1
        # azimuthal solve is not yet stable (exact_si_rlr.jl STATUS note). Gate it
        # behind an explicit experimental opt-in so it cannot be used in production.
        get(model.options, :xsi_rlr_experimental, false) === true || error(
            "options[:exact_si] on the RLR cylinder is experimental and not yet " *
            "stable at azimuthal wavenumber ≥ 1; set options[:xsi_rlr_experimental] " *
            "= true to run the WIP solver (see exact_si_rlr.jl)")
    else
        model.grid_params.geometry == "RiRk" || error(
            "options[:exact_si] is only implemented for the RiRk grid (XZ or " *
            "axisymmetric r–z) and the RLR cylinder; got geometry " *
            "$(model.grid_params.geometry)")
    end
    get(model.options, :state_dependent_si, false) === true && error(
        "options[:exact_si] and options[:state_dependent_si] are structurally " *
        "incompatible: the 2-D solve is reference-linearized with a precomputed " *
        "factorization (state-dependent coefficients would require per-step " *
        "refactorization of the patch operator). Use one or the other.")
    get(model.options, :horizontal_semiimplicit, false) === true && error(
        "options[:exact_si] replaces options[:horizontal_semiimplicit] (the " *
        "unsplit solve supersedes the rejected ADI sweep); enable only one.")
    return nothing
end

"""
    create_exact_si_data(patch, model, Pxi_prof, rho_t2, rho_d2, etp2) -> ExactSIData

Build the patch-level solve data: assemble and factorize the weighted-mass 2-D
Helmholtz operator `A = Mzw⊗Mx + Δτ²(Mz⊗Sx + Sz⊗Mx)` (i-fast coefficient
layout, matching the patch spectral layout) for both timestep coefficients.
`Pxi_prof` is the kDim-level local Pξ̄(z); `rho_t2`, `rho_d2`, `etp2` are
`(kDim × 2)` matrices of the (value, ∂z) reference profiles ρ̄_t, ρ̄_d and
Ē_t + p̄ (the slaved-leg coefficient chains of the vertical solve).
"""
function create_exact_si_data(patch::AbstractGrid, model::ModelParameters,
        Pxi_prof::AbstractVector{Float64}, rho_t2::AbstractMatrix{Float64},
        rho_d2::AbstractMatrix{Float64}, etp2::AbstractMatrix{Float64})

    model.grid_params.geometry == "RiRk" || error(
        "exact_si is only implemented for the XZ RiRk geometry")
    vars = model.grid_params.vars
    u_index = vars["u"]; w_index = vars["w"]; p_index = vars["p"]
    rhod_index = vars["rho_d"]; rhot_index = vars["rho_t"]; et_index = vars["E_t"]

    iDim = patch.params.iDim; kDim = patch.params.kDim
    b_iDim = patch.params.b_iDim; b_kDim = patch.params.b_kDim
    length(Pxi_prof) == kDim || error("Pxi_prof must be a kDim-level profile")

    # Galerkin blocks. i-direction on p's side-wall basis (all z_b blocks share
    # one SplineParameters per var); k-direction on p's column basis. The mish
    # operators are the model's own (operator consistency, reference/SI_VERTICAL_CEILING.md).
    isp_p0 = patch.ibasis.data[1, p_index]
    dx = _rirk_solve_data(isp_p0)
    kcol_p0 = patch.kbasis.data[p_index]
    dz = _rirk_solve_data(kcol_p0)

    # Axisymmetric radial metric (Stage 3): the horizontal Laplacian becomes the
    # cylindrical (1/r)∂r(r∂r·) and the weak Galerkin integral carries the
    # cylindrical volume element r dr dz. Net effect: re-weight ONLY the radial
    # (i) Galerkin blocks by r (the radial coordinate at the i-mish points);
    # every vertical block is unchanged. `rmet` is the i-mish radius vector, or
    # `ones` on the XZ Cartesian path (never used there).
    axisym = exact_si_is_axisym(model)
    rmet = axisym ? collect(isp_p0.mishPoints) : ones(iDim)

    # Sparse (banded) 1-D blocks: the cubic-spline Gram/stiffness matrices have
    # coefficient half-bandwidth 3, so the tensor operator is banded (half-band
    # 3·b_iDim in the i-fast layout, derivation §6). A sparse LU makes the
    # per-step back-substitution O(band·N) ≈ 1e6 flop instead of the dense
    # O(N²) — the cost gate. (droptol clears the ~1e-16 fill from the outer
    # products so the sparsity is the true band.)
    if axisym
        Sx = sparse(dx.M1' * ((dx.W .* rmet) .* dx.M1))  # Sᵣ = M1ᵣᵀ(W·r)M1ᵣ
        Mx = sparse(dx.M0' * ((dx.W .* rmet) .* dx.M0))  # Mᵣ = M0ᵣᵀ(W·r)M0ᵣ
    else
        Sx = sparse(dx.M1' * (dx.W .* dx.M1))
        Mx = sparse(dx.Mass)
    end
    droptol!(Sx, 1e-13 * maximum(abs, Sx))
    droptol!(Mx, 1e-13 * maximum(abs, Mx))
    Sz = sparse(dz.M1' * (dz.W .* dz.M1));  droptol!(Sz, 1e-13 * maximum(abs, Sz))
    Mzw = sparse(dz.M0' * ((dz.W ./ Pxi_prof) .* dz.M0))
    droptol!(Mzw, 1e-13 * maximum(abs, Mzw))
    Mz = sparse(dz.Mass);  droptol!(Mz, 1e-13 * maximum(abs, Mz))

    ax0 = get(model.options, :exact_si_ax0, false) === true
    lid_rows = get(model.options, :xsi_lid_rows, "strong")
    Nb1z = CubicBSpline.SItransform_matrix(kcol_p0, [model.grid_params.kMin,
                                                     model.grid_params.kMax], 1)

    # i-fast layout: coefficient (k_b−1)·b_iDim + i_b ⇒ A = kron(Z-block, X-block)
    function build(ts_term)
        A = kron(Mzw, Mx) .+ (ts_term^2) .* kron(Sz, Mx)
        if !ax0
            A = A .+ (ts_term^2) .* kron(Mz, Sx)
        end
        if lid_rows == "strong"
            # Replace the first/last vertical coefficient planes with the
            # ∂z-value rows Σ_kb ∂zψ_kb(z_bnd)·p̂[ib,kb] = φ*(bnd)/Δτ — pinning
            # the lid Neumann data the recovery enforces (w^{n+1} = 0).
            A = Matrix{Float64}(A)   # temporary dense for the row surgery
            for ib in 1:b_iDim
                r_bot = ib
                r_top = (b_kDim - 1) * b_iDim + ib
                A[r_bot, :] .= 0.0
                A[r_top, :] .= 0.0
                for kb in 1:b_kDim
                    A[r_bot, (kb - 1) * b_iDim + ib] = Nb1z[1, kb]
                    A[r_top, (kb - 1) * b_iDim + ib] = Nb1z[2, kb]
                end
            end
            return lu(sparse(A))
        end
        return lu(sparse(A))
    end
    fact = build(1.25 * model.ts)
    fact_first = build(0.5 * model.ts)

    # u side-wall / interface classification. Only a RIGID Dirichlet wall
    # (a real u value, u = 0 on the axis and the outer domain wall) contributes
    # the ∂r p′ = ρ̄_t u*/Δτ boundary load. Every other radial condition maps to
    # a homogeneous-natural row (∂r p′ = 0 ⇒ δu = 0): a NESTED interface FixedBC
    # (R3X, u = NaN — the parent pins the value, so freeze δu = 0 during the
    # acoustic sub-step, exact_si_stage3_plan §Stage-5) or a parent-collar
    # NaturalBC (free edge) or a Neumann u wall. Robin/periodic are unsupported.
    bcl = model.grid_params.BCL["u"]; bcr = model.grid_params.BCR["u"]
    for bc in (bcl, bcr)
        (bc.robin === nothing && !bc.periodic) || error(
            "exact_si does not support Robin or periodic u side walls")
    end

    # Slaved-leg coefficient chains (the vertical solve's non-sd formulas)
    rho_tb = rho_t2[:, 1]; rho_tb_z = rho_t2[:, 2]
    rho_db = rho_d2[:, 1]; rho_db_z = rho_d2[:, 2]
    etp = etp2[:, 1]; etp_z = etp2[:, 2]
    c_d = rho_db ./ rho_tb
    c_d_z = ((rho_db_z .* rho_tb) .- (rho_db .* rho_tb_z)) ./ (rho_tb .^ 2)
    c_e = etp ./ rho_tb
    c_e_z = ((etp_z .* rho_tb) .- (etp .* rho_tb_z)) ./ (rho_tb .^ 2)

    # Wall radii for the boundary loads: the radial Laplacian by-parts flux
    # [ψ·Δτ²·r·∂r p′] is r-weighted (derivation §3-BC/§9). The outer wall carries
    # r = iMax; the inner (r=iMin) carries r = iMin, which is 0 on the axis and
    # kills the axis load automatically (cylindrical regularity — no explicit
    # axis row). On the Cartesian path the flux has no r-factor, so both are 1.0.
    r_wall_l = axisym ? Float64(model.grid_params.iMin) : 1.0
    r_wall_r = axisym ? Float64(model.grid_params.iMax) : 1.0

    return ExactSIData(u_index, w_index, p_index, rhod_index, rhot_index, et_index,
        fact, fact_first,
        Matrix(dx.M0), Matrix(dx.M1), collect(dx.W), Matrix(dx.Nb),
        Matrix(dz.M0), Matrix(dz.M1), collect(dz.W), Matrix(dz.Nb),
        (_is_rigid_dirichlet(bcl), _is_rigid_dirichlet(bcr)), lid_rows == "strong", ax0,
        axisym, rmet, r_wall_l, r_wall_r,
        collect(Pxi_prof), collect(rho_tb), c_d, c_d_z, c_e, c_e_z,
        deepcopy(patch.ibasis.data[1, u_index]), deepcopy(patch.ibasis.data[1, p_index]),
        deepcopy(patch.kbasis.data[w_index]), deepcopy(patch.kbasis.data[p_index]),
        deepcopy(patch.kbasis.data[u_index]),
        zeros(iDim, kDim), zeros(iDim, kDim), zeros(iDim, kDim), zeros(iDim, kDim),
        zeros(iDim, kDim), zeros(iDim, kDim), zeros(iDim, kDim), zeros(iDim, kDim),
        zeros(b_iDim, b_kDim), zeros(b_iDim, b_kDim),
        zeros(iDim, kDim), zeros(iDim, kDim), zeros(iDim, kDim),
        zeros(kDim), zeros(kDim), zeros(iDim), zeros(iDim), zeros(kDim))
end

"""
    exact_si_solve!(hfields, esd, patch, model, t, xstar) -> hfields

Master-side unsplit solve. `xstar` holds the raw predictor fields at the patch
physical points (row order `(i−1)·kDim + k`): columns (u*, w*, p′*). Solves the
weighted-mass 2-D Helmholtz for p′^{n+1} and recovers EVERY fast leg from the
solved coefficients (see the module header), filling the `XSI_NPLANES`
increment planes of `hfields`.
"""
function exact_si_solve!(hfields::AbstractMatrix{Float64}, esd::ExactSIData,
        patch::AbstractGrid, model::ModelParameters, t::Int64,
        xstar::AbstractMatrix{Float64})

    if get(model.options, :exact_si_zero_x, false) === true
        fill!(hfields, 0.0)
        return hfields
    end
    iDim = patch.params.iDim; kDim = patch.params.kDim
    ts_term = (t == 1) ? 0.5 * model.ts : 1.25 * model.ts
    f = (t == 1) ? esd.fact_first : esd.fact

    U = esd.U; PH = esd.PH; PR = esd.PR; W0 = esd.W0
    @inbounds for i in 1:iDim, k in 1:kDim
        r = (i - 1) * kDim + k
        U[i, k] = xstar[r, 1]
        W0[i, k] = xstar[r, 2]
        PH[i, k] = esd.rho_tbar[k] * xstar[r, 2]   # φ* = ρ̄_t w*
        PR[i, k] = xstar[r, 3]
    end

    # Round-trip the u* and p′* inputs through their own vertical fit chains
    # (P = eval∘fit, the l_q filter the carried state undergoes — mirroring the
    # vertical solve, which replaces its p-predictor by the refit evaluation):
    # the raw AB3 predictors carry z-grid-scale content the model's own read
    # chains never see, and feeding it to the solve leaks the (I−P) refit
    # residual into the implicit operator at state amplitude (the DG round's
    # "round-tripped baseline" lesson).
    kcu = esd.kcol_u
    kcp = esd.kcol_p
    for i in 1:iDim
        @inbounds for k in 1:kDim
            kcu.uMish[k] = U[i, k]
            kcp.uMish[k] = PR[i, k]
        end
        SBtransform!(kcu); SAtransform!(kcu); SItransform!(kcu)
        SBtransform!(kcp); SAtransform!(kcp); SItransform!(kcp)
        @inbounds for k in 1:kDim
            U[i, k] = kcu.uMish[k]
            PR[i, k] = kcp.uMish[k]
        end
    end

    # ∂x u* per level (u's i-basis fit — the model's own x read chain), plus
    # the wall values of u* for the u-Dirichlet boundary loads.
    if !esd.ax0
        isp = esd.isp_u
        for k in 1:kDim
            @inbounds for i in 1:iDim
                isp.uMish[i] = U[i, k]
            end
            SBtransform!(isp)
            SAtransform!(isp)
            SIxtransform(isp, view(esd.ux, :, k))
            acc_l = 0.0; acc_r = 0.0
            @inbounds for j in 1:size(esd.Nbx, 2)
                acc_l += esd.Nbx[1, j] * isp.a[j]
                acc_r += esd.Nbx[2, j] * isp.a[j]
            end
            esd.ubl[k] = acc_l; esd.ubr[k] = acc_r
        end
    else
        fill!(esd.ux, 0.0)
        fill!(esd.ubl, 0.0); fill!(esd.ubr, 0.0)
    end

    # ∂z φ* per column and the round-tripped φ* values (w's column-basis fit —
    # the impdot staging chain). The lid/surface VALUES of φ* (the inhomogeneous
    # Neumann data ∂z p′ = φ*/Δτ) are evaluated through p's Neumann column basis
    # instead: w's Dirichlet fit forces them to zero and loses the very data the
    # boundary condition carries.
    kcw = esd.kcol_w
    for i in 1:iDim
        @inbounds for k in 1:kDim
            kcw.uMish[k] = PH[i, k]
            kcp.uMish[k] = PH[i, k]
        end
        SBtransform!(kcw)
        SAtransform!(kcw)
        SIxtransform(kcw, esd.zcol)
        @inbounds for k in 1:kDim
            esd.phiz[i, k] = esd.zcol[k]
        end
        SItransform!(kcw)
        @inbounds for k in 1:kDim
            esd.PHR[i, k] = kcw.uMish[k]
        end
        SBtransform!(kcp)
        SAtransform!(kcp)
        acc_b = 0.0; acc_t = 0.0
        @inbounds for j in 1:size(esd.Nbz, 2)
            acc_b += esd.Nbz[1, j] * kcp.a[j]
            acc_t += esd.Nbz[2, j] * kcp.a[j]
        end
        esd.phib[i] = acc_b; esd.phit[i] = acc_t
    end

    # Galerkin load: the p′* term through the 1/Pξ̄-weighted quadrature, the
    # divergence term unweighted (derivation §3), both with the 2-D Gauss
    # weights Wx·Wz; then the boundary terms (§3-BC). On the axisym path the
    # radial quadrature carries the cylindrical volume weight r (Wx → Wx·r) and
    # the divergence gains the radial metric term u*/r (the strong linear radial
    # divergence ∂r u* + u*/r; §9), both disabled under ax0.
    G = esd.G
    if esd.axisym
        @inbounds for i in 1:iDim, k in 1:kDim
            ldiv = esd.ux[i, k] + (esd.ax0 ? 0.0 : U[i, k] / esd.rmet[i])
            G[i, k] = esd.Wx[i] * esd.rmet[i] * esd.Wz[k] *
                      ((PR[i, k] / esd.Pxi[k]) -
                       ts_term * ((esd.rho_tbar[k] * ldiv) + esd.phiz[i, k]))
        end
    else
        @inbounds for i in 1:iDim, k in 1:kDim
            G[i, k] = esd.Wx[i] * esd.Wz[k] *
                      ((PR[i, k] / esd.Pxi[k]) -
                       ts_term * ((esd.rho_tbar[k] * esd.ux[i, k]) + esd.phiz[i, k]))
        end
    end
    L = esd.L
    mul!(L, esd.M0x', G * esd.M0z)
    # u-Dirichlet side walls (⇒ ∂x p′^{n+1} = ρ̄_t u*/Δτ); Neumann walls are
    # homogeneous natural (no term). Applied BEFORE the lid rows so a strong
    # lid replacement overwrites the wall flux on the corner-plane rows.
    if !esd.ax0
        if esd.dirichlet_u[2]
            L .+= (ts_term * esd.r_wall_r) .* esd.Nbx[2, :] *
                  transpose(esd.M0z' * (esd.Wz .* esd.rho_tbar .* esd.ubr))
        end
        if esd.dirichlet_u[1]
            L .-= (ts_term * esd.r_wall_l) .* esd.Nbx[1, :] *
                  transpose(esd.M0z' * (esd.Wz .* esd.rho_tbar .* esd.ubl))
        end
    end
    # Lid/surface (w-Dirichlet ⇒ ∂z p′^{n+1} = φ*/Δτ). Strong rows (default):
    # the replaced coefficient planes carry the x-fit coefficients of
    # φ*(·,bnd)/Δτ. Natural: the by-parts flux loads ±Δτ·φ*.
    if esd.strong_lid
        isp = esd.isp_p
        for (bnd, kb) in ((esd.phib, 1), (esd.phit, size(L, 2)))
            @inbounds for i in 1:iDim
                isp.uMish[i] = bnd[i] / ts_term
            end
            SBtransform!(isp)
            SAtransform!(isp)
            @inbounds for ib in 1:size(L, 1)
                L[ib, kb] = isp.a[ib]
            end
        end
    else
        L .+= ts_term .* (esd.M0x' * (esd.Wx .* esd.phit)) * transpose(esd.Nbz[2, :])
        L .-= ts_term .* (esd.M0x' * (esd.Wx .* esd.phib)) * transpose(esd.Nbz[1, :])
    end

    q = f \ vec(L)
    Q = esd.Q
    copyto!(Q, reshape(q, size(Q)))

    # Recoveries — every leg from the SOLVED coefficients through the
    # assembly's own operators (weak-consistency; derivation §3, VN 3(a)):
    #   p^{n+1} = M0 p̂;  δp = p^{n+1} − p′* (round-tripped baseline)
    #   u^{n+1} = u* − (Δτ/ρ̄_t) M1x p̂
    #   φ^{n+1} = φ*_fit − Δτ M1z p̂;  w^{n+1} = φ^{n+1}/ρ̄_t;  δw vs raw w*
    #   div^{n+1} = −δp/(Δτ Pξ̄)  (the p-recovery identity — pointwise)
    #   δρ_t = −Δτ div;  δρ_d = c_d δρ_t − Δτ c_d,z φ^{n+1};  δE_t analogous
    mul!(esd.P0, esd.M0x, Q * esd.M0z')
    mul!(esd.px, esd.M1x, Q * esd.M0z')
    mul!(esd.pz, esd.M0x, Q * esd.M1z')
    @inbounds for i in 1:iDim, k in 1:kDim
        r = (i - 1) * kDim + k
        rho = esd.rho_tbar[k]
        du = esd.ax0 ? 0.0 : -(ts_term / rho) * esd.px[i, k]
        phin1 = esd.PHR[i, k] - ts_term * esd.pz[i, k]
        dp = esd.P0[i, k] - PR[i, k]
        drt = dp / esd.Pxi[k]                       # −Δτ·div^{n+1}
        hfields[r, 1] = du
        hfields[r, 2] = phin1 / rho - W0[i, k]
        hfields[r, 3] = dp
        hfields[r, 4] = esd.c_d[k] * drt - ts_term * esd.c_d_z[k] * phin1
        hfields[r, 5] = drt
        hfields[r, 6] = esd.c_e[k] * drt - ts_term * esd.c_e_z[k] * phin1
        hfields[r, 7] = du / ts_term
    end
    return hfields
end

"""
    apply_acoustic_histories!(mtile, colstart, colend, t)

Apply the AI2* explicit history levels of the VERTICAL acoustic legs to the
predictor state and roll the history — the loop `semiimplicit_adjustment_p`
runs when it owns the histories, hoisted to phase A under `options[:exact_si]`
(the patch-level solve must see X* with every explicit level applied).
"""
function apply_acoustic_histories!(mtile::ModelTile, colstart::Int64, colend::Int64,
        t::Int64)
    vars = mtile.model.grid_params.vars
    ts = mtile.model.ts
    for var in ("p", "w", "rho_d", "rho_t", "E_t")
        index = vars[var]
        nstar = view(mtile.var_np1, colstart:colend, index)
        dot_n = view(mtile.impdot_n, colstart:colend, index)
        dot_nm1 = view(mtile.impdot_nm1, colstart:colend, index)
        if (t == 1)
            nstar .= @. nstar + (ts * 0.5 * dot_n)
        else
            nstar .= @. nstar + (ts * ((0.75 * dot_nm1) - dot_n))
        end
        dot_nm1 .= dot_n
    end
    return nothing
end

"""
    exact_si_apply_column!(mtile, colstart, colend, t, hfields, rowstart)

Phase B for one column: apply the solve's increments to every fast leg, store
the w-leg applied increment as the next step's AI2* history (the vertical
solve's discipline — `(w^{n+1} − w*)/Δτ`, seeded at t == 1), and run the
implicit vertical diffusion. Under `options[:exact_si_zero_x]` the per-column
vertical solve runs instead (histories already applied in phase A), making the
path bitwise the production vertical-only path — the A≡0 plumbing test.
`rowstart` is the tile's first physical row in patch numbering.
"""
function exact_si_apply_column!(mtile::ModelTile, colstart::Int64, colend::Int64,
        t::Int64, hfields::AbstractMatrix{Float64}, rowstart::Int64)

    model = mtile.model
    if get(model.options, :exact_si_zero_x, false) === true
        semiimplicit_adjustment_p(mtile, colstart, colend, t; apply_histories = false)
    else
        vars = model.grid_params.vars
        ts_term = (t == 1) ? 0.5 * model.ts : 1.25 * model.ts
        rows = (rowstart + colstart - 1):(rowstart + colend - 1)
        w_index = vars["w"]
        view(mtile.var_np1, colstart:colend, vars["u"]) .+= view(hfields, rows, 1)
        view(mtile.var_np1, colstart:colend, w_index) .+= view(hfields, rows, 2)
        view(mtile.var_np1, colstart:colend, vars["p"]) .+= view(hfields, rows, 3)
        view(mtile.var_np1, colstart:colend, vars["rho_d"]) .+= view(hfields, rows, 4)
        view(mtile.var_np1, colstart:colend, vars["rho_t"]) .+= view(hfields, rows, 5)
        view(mtile.var_np1, colstart:colend, vars["E_t"]) .+= view(hfields, rows, 6)
        # RLR: apply the tangential-wind acoustic increment (plane 7). The v leg
        # is integrated θ = 1 (implicit-only, no stored explicit history) — the
        # self-consistent partner of the +n²/r² operator term (exact_si_rlr.jl);
        # the stored-increment v history (plane 9) is future 2nd-order work.
        if exact_si_is_rlr(model)
            view(mtile.var_np1, colstart:colend, vars["v"]) .+= view(hfields, rows, 7)
        end
        # w-leg stored-applied-increment history (the 2026-07-16 discipline)
        view(mtile.impdot_n, colstart:colend, w_index) .=
            view(hfields, rows, 2) ./ ts_term
        if t == 1
            view(mtile.impdot_nm1, colstart:colend, w_index) .=
                view(mtile.impdot_n, colstart:colend, w_index)
        end
    end

    Kvdiff = get(model.physical_params, :Kvdiff, 0.0)
    Kvdiff_heat = get(model.physical_params, :Kvdiff_heat, Kvdiff)
    Kvdiff_water = get(model.physical_params, :Kvdiff_water, 0.0)
    if Kvdiff > 0.0 || Kvdiff_heat > 0.0 || Kvdiff_water > 0.0
        diffusion_timestep_mc(mtile, colstart, colend, t)
    end
    # Water positivity, LAST — the phase-B mirror of the call at the end of
    # mc_driver! (which this path returns from early, at the end of phase A).
    clamp_water!(mtile, colstart, colend)
    return nothing
end

"""
    exact_si_load_history!(mtile, hfields, t, rowstart)

Load the previous step's applied u increment (plane 7 of the solve feed) into
the tile's `hacdot_n` u column as the u-leg AI2* history — under the default
`options[:xsi_u_history] = "stored"`. `"none"` integrates the u leg with its
implicit level alone (θ = 1): stable, but the θ = 1 first-order component
leaves an ~11% max|u| bias on the BF02 dry bubble (G2 A/B, 2026-07-17), so it
is not the default. Mirrors [`horizontal_si_load_increment!`](@ref)'s t == 2
seeding. Unlike the rejected ADI sweep, the stored u history in the UNSPLIT
solve carries no leak (VN 3(a): the fresh strong u-history is ε-insensitive).
"""
function exact_si_load_history!(mtile::ModelTile, hfields::AbstractMatrix{Float64},
        t::Int64, rowstart::Int64)
    get(mtile.model.options, :xsi_u_history, "stored") == "stored" || return nothing
    u_index = mtile.model.grid_params.vars["u"]
    n = size(mtile.hacdot_n, 1)
    rows = rowstart:(rowstart + n - 1)
    # The u-leg history plane is 7 on the XZ/axisym feed, 8 on the RLR feed
    # (which appends the v leg + v history).
    rlr = exact_si_is_rlr(mtile.model)
    uplane = rlr ? 8 : 7
    dst = view(mtile.hacdot_n, :, u_index)
    copyto!(dst, view(hfields, rows, uplane))
    t == 2 && copyto!(view(mtile.hacdot_nm1, :, u_index), dst)
    # RLR: the tangential v leg carries the same stored-applied-increment history
    # (plane 9 = δv/Δτ), loaded into hacdot's v column.
    if rlr
        v_index = mtile.model.grid_params.vars["v"]
        dstv = view(mtile.hacdot_n, :, v_index)
        copyto!(dstv, view(hfields, rows, 9))
        t == 2 && copyto!(view(mtile.hacdot_nm1, :, v_index), dstv)
    end
    return nothing
end

"""
    advanceTimestepA(mtile, sharedSpectral, t, xstar, hfields, rowstart)

Exact-SI phase A for one tile: transform to physical space, load the stored
u-leg history (if enabled) from the previous step's feed, advance every column
through the predictor stage (`mc_driver!` exits before the implicit solve when
`options[:exact_si]` is on), and publish the (u, w, p′) predictors to the
shared patch-level array for the master's unsplit solve.
"""
function advanceTimestepA(mtile::ModelTile, sharedSpectral::SharedArray{Float64},
        t::Int64, xstar::AbstractMatrix{Float64}, hfields::AbstractMatrix{Float64},
        rowstart::Int64)

    tileTransform!(sharedSpectral, mtile.tile, mtile.tile.physical, mtile.tile.spectral)
    checkCFL(mtile.tile; t=t, ts=mtile.model.ts, where="worker tile")
    state_minima_trace(mtile, t)
    uses_pressure_reference(mtile.model.equation_set) && water_negativity_trace(mtile, t)
    uses_pressure_reference(mtile.model.equation_set) && water_budget_trace(mtile, t)
    if t > 1
        exact_si_load_history!(mtile, hfields, t, rowstart)
    end

    if num_columns(mtile.tile) > 0
        Threads.@threads :static for c in 1:num_columns(mtile.tile)
            advance_column(mtile, c, t)
        end
    else
        advance_column(mtile, -1, t)
    end

    vars = mtile.model.grid_params.vars
    n = size(mtile.var_np1, 1)
    rows = rowstart:(rowstart + n - 1)
    xstar[rows, 1] .= view(mtile.var_np1, :, vars["u"])
    xstar[rows, 2] .= view(mtile.var_np1, :, vars["w"])
    if exact_si_is_rlr(mtile.model)
        # RLR publishes the tangential wind too (column 3 = v*, 4 = p′*).
        xstar[rows, 3] .= view(mtile.var_np1, :, vars["v"])
        xstar[rows, 4] .= view(mtile.var_np1, :, vars["p"])
    else
        xstar[rows, 3] .= view(mtile.var_np1, :, vars["p"])
    end
    return nothing
end

"""
    advanceTimestepB(mtile, sharedSpectral, haloSend, haloReceive, t, hfields, rowstart)

Exact-SI phase B for one tile: complete every column from the solve's
increment planes, then the standard end-of-step sequence (spectral fit, halo
exchange, shared-array accumulation) of `advanceTimestep`.
"""
function advanceTimestepB(mtile::ModelTile, sharedSpectral::SharedArray{Float64},
        haloSend::RemoteChannel, haloReceive::RemoteChannel, t::Int64,
        hfields::AbstractMatrix{Float64}, rowstart::Int64)

    vdim = mtile.model.grid_params.kDim
    if num_columns(mtile.tile) > 0
        Threads.@threads :static for c in 1:num_columns(mtile.tile)
            colstart = (c - 1) * vdim + 1
            exact_si_apply_column!(mtile, colstart, colstart + vdim - 1, t,
                                   hfields, rowstart)
        end
    else
        exact_si_apply_column!(mtile, 1, size(mtile.tile.physical, 1), t,
                               hfields, rowstart)
    end

    calcTendency(mtile)
    put!(haloSend, extract_halo_values(mtile.tile))
    write_tile_to_shared!(sharedSpectral, mtile.tile, mtile.patch_b_iDim)
    mtile.haloReceiveBuffer .= take!(haloReceive)
    accumulate_at_map!(sharedSpectral, mtile.haloReceiveMap, mtile.haloReceiveBuffer)
    return nothing
end
