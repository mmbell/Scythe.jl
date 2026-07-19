# ─────────────────────────────────────────────────────────────────────────────
# Exact (unsplit) acoustic semi-implicit for the RLR 3-D cylindrical set
# (moist_compressible_RLR): spline-radius × Fourier-azimuth × spline-z.
#
# The azimuthal Fourier modes of the reference-linearized acoustic operator
# DECOUPLE per wavenumber n, so the 3-D solve is a FAMILY of independent 2-D
# (r,z) solves — one per n — each the axisymmetric weighted-mass p′-form
# Helmholtz (exact_si.jl) plus the azimuthal term −n²/r²:
#
#     A_n = kron(Mzw, Mᵣ) + Δτ²( kron(Sz, Mᵣ) + kron(Mz, Sᵣ) )   [ = A_axisym ]
#         + Δτ² n² kron(Mz, Mᵣᵢₙᵥ),      Mᵣᵢₙᵥ = M0ᵣᵀ(W/r)M0ᵣ
#
# (SPD for every n; n=0 ⇒ A_axisym exactly; higher n better conditioned — verified
# on production dims. reference/exact_si_stage4_orchestration.md.) The tangential
# wind v couples to p through the azimuthal PGF: per-n the divergence gains
# ρ̄_t(1/r)∂λ v and the v-leg becomes implicit, v_n^{n+1} = v_n* − Δτ(in/(rρ̄_t))p′_n
# — the coupling that produces the +n²/r² operator term on elimination.
#
# Orchestration (master-side, physical-in/physical-out like the axisym solve):
#   1. read physical predictors (u,w,v,p)* at the jDim·kDim ring points;
#   2. azimuthal-only Fourier per (radius, z) → per-n cos/sin (r,z) fields
#      (ragged: ring at radius r supports n ≤ r+patchOffsetL, higher n are 0);
#   3. per n, per {cos,sin}: the axisym (r,z) core with A_n and the azimuthal
#      divergence folded into the radial linear divergence, then recover u,w,p and
#      the slaved ρ_d/ρ_t/E_t; recover δv cross-wise from the OTHER part's p′_n;
#   4. inverse azimuthal Fourier of the per-n increment fields → physical
#      increments for every leg.
#
# n=0 is the axisym problem verbatim (no v coupling), so a WN0 field reproduces the
# axisymmetric solve (the G4-equiv gate).
#
# STATUS (2026-07-18): WORK IN PROGRESS, gated behind options[:xsi_rlr_experimental].
#   VERIFIED: the n=0 path reproduces the axisym exact-SI solve to machine
#     precision (rel ~1e-13, every leg; standalone G4-equiv), the azimuthal Fourier
#     fwd/inv round-trips to 1e-15, and the per-n operator A_n is SPD for all n with
#     conditioning improving in n. With the p↔v coupling OFF (:xsi_rlr_no_v) the
#     per-n radial+vertical solve is stable at full wavenumber.
#   OPEN BUG: the coupled n≥1 azimuthal solve is UNSTABLE (~100×/step, blow-up by
#     ~t/ts=13). The instability is sign-INDEPENDENT (all four load/recovery sign
#     combinations blow up at the same step) and survives every consistency fix
#     tried — the cos/sin R2HC convention, an n≥1 axis-Dirichlet regularity row,
#     and both a fresh and a stored-applied-increment v-history. The p↔v elimination
#     is algebraically exact and the signs verified by hand, so this is a discrete
#     stability property of the split azimuthal treatment that needs a von Neumann
#     analysis of the coupled (p,v) mode — the deferred next step. Do NOT enable
#     RLR exact_si in production until this is resolved.
# ─────────────────────────────────────────────────────────────────────────────

# Increment planes of the RLR solve feed (jDim·kDim × XSI_RLR_NPLANES):
# applied increments of (u, w, p′, ρ_d′, ρ_t′, E_t′, v), plane 8 the u-leg
# history δu/Δτ, plane 9 the v-leg history δv/Δτ.
const XSI_RLR_NPLANES = 9

struct ExactSIDataRLR{FT, SU, SP, SW, KP, KU}
    u_index::Int; w_index::Int; v_index::Int; p_index::Int
    rhod_index::Int; rhot_index::Int; et_index::Int
    facts::Vector{FT}          # per-n factorization (index n+1), Δτ = 1.25·ts
    facts_first::Vector{FT}    # per-n factorization, Δτ = 0.5·ts (t == 1)
    nmax::Int
    M0x::Matrix{Float64}; M1x::Matrix{Float64}; Wx::Vector{Float64}; Nbx::Matrix{Float64}
    M0z::Matrix{Float64}; M1z::Matrix{Float64}; Wz::Vector{Float64}; Nbz::Matrix{Float64}
    dirichlet_u::Tuple{Bool, Bool}; strong_lid::Bool; axis_dirichlet::Bool
    rmet::Vector{Float64}; r_wall_l::Float64; r_wall_r::Float64
    Pxi::Vector{Float64}; rho_tbar::Vector{Float64}
    c_d::Vector{Float64}; c_d_z::Vector{Float64}; c_e::Vector{Float64}; c_e_z::Vector{Float64}
    isp_u::SU; isp_p::SP; kcol_w::SW; kcol_p::KP; kcol_u::KU
    # Ragged ring layout (built from getGridpoints): physical rows per radius
    # (each length lpoints[r]·kDim, ordered λ-major/z-minor), and per-ring kmax.
    rows_by_r::Vector{Vector{Int}}
    lpoints::Vector{Int}
    ring_kmax::Vector{Int}
    # scratch (iDim × kDim)
    U::Matrix{Float64}; W0::Matrix{Float64}; PH::Matrix{Float64}; PR::Matrix{Float64}
    ux::Matrix{Float64}; phiz::Matrix{Float64}; G::Matrix{Float64}
    PHR::Matrix{Float64}; P0::Matrix{Float64}; px::Matrix{Float64}; pz::Matrix{Float64}
    L::Matrix{Float64}; Q::Matrix{Float64}   # (b_iDim × b_kDim)
    ubl::Vector{Float64}; ubr::Vector{Float64}
    phib::Vector{Float64}; phit::Vector{Float64}; zcol::Vector{Float64}
end

"""
    create_exact_si_data_rlr(patch, model, Pxi_prof, rho_t2, rho_d2, etp2) -> ExactSIDataRLR

Build the RLR patch-level solve data: the per-wavenumber factorizations
`A_n = A_axisym + Δτ²n²·kron(Mz, Mᵣᵢₙᵥ)` (n = 0..nmax, both timestep coefficients,
strong lid rows), the shared radial/vertical Galerkin blocks and reference
profiles (identical to the axisym build), and the ragged-ring physical-row map.
"""
function create_exact_si_data_rlr(patch::AbstractGrid, model::ModelParameters,
        Pxi_prof::AbstractVector{Float64}, rho_t2::AbstractMatrix{Float64},
        rho_d2::AbstractMatrix{Float64}, etp2::AbstractMatrix{Float64})

    vars = model.grid_params.vars
    u_index = vars["u"]; w_index = vars["w"]; v_index = vars["v"]; p_index = vars["p"]
    rhod_index = vars["rho_d"]; rhot_index = vars["rho_t"]; et_index = vars["E_t"]

    iDim = patch.params.iDim; kDim = patch.params.kDim
    b_iDim = patch.params.b_iDim; b_kDim = patch.params.b_kDim
    length(Pxi_prof) == kDim || error("Pxi_prof must be a kDim-level profile")

    isp_p0 = patch.ibasis.data[1, p_index]
    dx = _rirk_solve_data(isp_p0)
    kcol_p0 = patch.kbasis.data[p_index]
    dz = _rirk_solve_data(kcol_p0)

    rmet = collect(isp_p0.mishPoints)   # radial coordinate at the i-mish points

    # Radial blocks: r-weighted mass/stiffness (axisym) + the 1/r-weighted mass
    # (the azimuthal −n²/r² carrier); vertical blocks unchanged.
    Sx = sparse(dx.M1' * ((dx.W .* rmet) .* dx.M1))        # Sᵣ
    Mx = sparse(dx.M0' * ((dx.W .* rmet) .* dx.M0))        # Mᵣ
    Mxinv = sparse(dx.M0' * ((dx.W ./ rmet) .* dx.M0))     # Mᵣᵢₙᵥ
    Sz = sparse(dz.M1' * (dz.W .* dz.M1))
    Mzw = sparse(dz.M0' * ((dz.W ./ Pxi_prof) .* dz.M0))
    Mz = sparse(dz.Mass)
    for M in (Sx, Mx, Mxinv, Sz, Mzw, Mz)
        droptol!(M, 1e-13 * maximum(abs, M))
    end

    lid_rows = get(model.options, :xsi_lid_rows, "strong")
    strong_lid = lid_rows == "strong"
    Nb1z = CubicBSpline.SItransform_matrix(kcol_p0, [model.grid_params.kMin,
                                                     model.grid_params.kMax], 1)

    A_axisym(ts) = kron(Mzw, Mx) .+ (ts^2) .* (kron(Sz, Mx) .+ kron(Mz, Sx))
    Nb0r = Matrix(dx.Nb)   # radial boundary-value rows (axis at [1,:])
    axis_dirichlet = get(model.options, :xsi_rlr_axis_dirichlet, true) === true
    function build_n(ts, n)
        A = A_axisym(ts)
        n > 0 && (A = A .+ (ts^2 * n^2) .* kron(Mz, Mxinv))
        if strong_lid || (n > 0 && axis_dirichlet)
            A = Matrix{Float64}(A)
        end
        if strong_lid
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
        end
        # Axis regularity for n ≥ 1: enforce p_n(r=0) = 0 (Dirichlet). The p-basis
        # gives a Neumann axis row, correct only for n = 0 (∂r p = 0 by symmetry);
        # for n ≥ 1 the mode must VANISH at the axis (p_n ∼ rⁿ), and the near-axis
        # 1/r factors in the v-leg/operator amplify any residual otherwise — the
        # blow-up mechanism. Replace the axis coefficient row (i_b = 1) per z-block
        # with the radial boundary-value row; applied AFTER the lid so the axis
        # wins at the axis∩lid corner. (RHS zeroed in _rlr_core!.)
        if n > 0 && axis_dirichlet
            for kb in 1:b_kDim
                r_axis = (kb - 1) * b_iDim + 1
                A[r_axis, :] .= 0.0
                for ib in 1:b_iDim
                    A[r_axis, (kb - 1) * b_iDim + ib] = Nb0r[1, ib]
                end
            end
        end
        return lu(sparse(A))
    end

    # Ragged ring layout + nmax from the grid rings.
    ring_kmax = [patch.jbasis.data[r, 1].params.kmax for r in 1:iDim]
    lpoints = [patch.jbasis.data[r, 1].params.yDim for r in 1:iDim]
    nmax = maximum(ring_kmax)

    # Physical rows per radius (λ-major, z-minor), from the gridpoints.
    pts = getGridpoints(patch)
    radii = sort(unique(round.(view(pts, :, 1), digits = 6)))
    length(radii) == iDim || error("RLR gridpoints: $(length(radii)) radii != iDim $iDim")
    rows_by_r = Vector{Vector{Int}}(undef, iDim)
    for r in 1:iDim
        rows_by_r[r] = findall(x -> round(x, digits = 6) == radii[r], view(pts, :, 1))
        length(rows_by_r[r]) == lpoints[r] * kDim || error(
            "RLR ring $r: $(length(rows_by_r[r])) rows != lpoints·kDim $(lpoints[r]*kDim)")
    end

    facts = [build_n(1.25 * model.ts, n) for n in 0:nmax]
    facts_first = [build_n(0.5 * model.ts, n) for n in 0:nmax]

    bcl = model.grid_params.BCL["u"]; bcr = model.grid_params.BCR["u"]
    for bc in (bcl, bcr)
        (bc.robin === nothing && !bc.periodic) || error(
            "exact_si does not support Robin or periodic u side walls")
    end
    r_wall_l = Float64(model.grid_params.iMin)
    r_wall_r = Float64(model.grid_params.iMax)

    rho_tb = rho_t2[:, 1]; rho_tb_z = rho_t2[:, 2]
    rho_db = rho_d2[:, 1]; rho_db_z = rho_d2[:, 2]
    etp = etp2[:, 1]; etp_z = etp2[:, 2]
    c_d = rho_db ./ rho_tb
    c_d_z = ((rho_db_z .* rho_tb) .- (rho_db .* rho_tb_z)) ./ (rho_tb .^ 2)
    c_e = etp ./ rho_tb
    c_e_z = ((etp_z .* rho_tb) .- (etp .* rho_tb_z)) ./ (rho_tb .^ 2)

    return ExactSIDataRLR(u_index, w_index, v_index, p_index, rhod_index, rhot_index,
        et_index, facts, facts_first, nmax,
        Matrix(dx.M0), Matrix(dx.M1), collect(dx.W), Matrix(dx.Nb),
        Matrix(dz.M0), Matrix(dz.M1), collect(dz.W), Matrix(dz.Nb),
        (_is_rigid_dirichlet(bcl), _is_rigid_dirichlet(bcr)), strong_lid, axis_dirichlet,
        rmet, r_wall_l, r_wall_r,
        collect(Pxi_prof), collect(rho_tb), c_d, c_d_z, c_e, c_e_z,
        deepcopy(patch.ibasis.data[1, u_index]), deepcopy(patch.ibasis.data[1, p_index]),
        deepcopy(patch.kbasis.data[w_index]), deepcopy(patch.kbasis.data[p_index]),
        deepcopy(patch.kbasis.data[u_index]),
        rows_by_r, lpoints, ring_kmax,
        zeros(iDim, kDim), zeros(iDim, kDim), zeros(iDim, kDim), zeros(iDim, kDim),
        zeros(iDim, kDim), zeros(iDim, kDim), zeros(iDim, kDim),
        zeros(iDim, kDim), zeros(iDim, kDim), zeros(iDim, kDim), zeros(iDim, kDim),
        zeros(b_iDim, b_kDim), zeros(b_iDim, b_kDim),
        zeros(kDim), zeros(kDim), zeros(iDim), zeros(iDim), zeros(kDim))
end

# Per-part (r,z) core: the axisym weighted-mass solve on the filled predictor
# fields esd.U/W0/PH/PR, with the azimuthal divergence (n·vaz/r) folded into the
# radial linear divergence and the factorization `f` = A_n. Writes the leg
# increments into the provided (iDim×kDim) outputs and leaves the recovered
# pressure p′_n in esd.P0 (for the cross-part v-leg). Mirrors exact_si_solve!.
function _rlr_core!(esd::ExactSIDataRLR, n::Int, vaz::AbstractMatrix{Float64},
        ts_term::Float64, f, du::Matrix{Float64}, dw::Matrix{Float64},
        dp::Matrix{Float64}, drd::Matrix{Float64}, drt::Matrix{Float64},
        det::Matrix{Float64})

    iDim, kDim = size(esd.U)
    U = esd.U; PR = esd.PR; PH = esd.PH; W0 = esd.W0

    # Round-trip u*, p′* through the vertical fit chain (P = eval∘fit).
    kcu = esd.kcol_u; kcp = esd.kcol_p
    for i in 1:iDim
        @inbounds for k in 1:kDim
            kcu.uMish[k] = U[i, k]; kcp.uMish[k] = PR[i, k]
        end
        SBtransform!(kcu); SAtransform!(kcu); SItransform!(kcu)
        SBtransform!(kcp); SAtransform!(kcp); SItransform!(kcp)
        @inbounds for k in 1:kDim
            U[i, k] = kcu.uMish[k]; PR[i, k] = kcp.uMish[k]
        end
    end

    # ∂r u* per level + wall values of u*.
    isp = esd.isp_u
    for k in 1:kDim
        @inbounds for i in 1:iDim
            isp.uMish[i] = U[i, k]
        end
        SBtransform!(isp); SAtransform!(isp)
        SIxtransform(isp, view(esd.ux, :, k))
        acc_l = 0.0; acc_r = 0.0
        @inbounds for j in 1:size(esd.Nbx, 2)
            acc_l += esd.Nbx[1, j] * isp.a[j]; acc_r += esd.Nbx[2, j] * isp.a[j]
        end
        esd.ubl[k] = acc_l; esd.ubr[k] = acc_r
    end

    # ∂z φ* per column, round-tripped φ*, lid/surface φ* values (p-Neumann basis).
    kcw = esd.kcol_w
    for i in 1:iDim
        @inbounds for k in 1:kDim
            kcw.uMish[k] = PH[i, k]; kcp.uMish[k] = PH[i, k]
        end
        SBtransform!(kcw); SAtransform!(kcw)
        SIxtransform(kcw, esd.zcol)
        @inbounds for k in 1:kDim
            esd.phiz[i, k] = esd.zcol[k]
        end
        SItransform!(kcw)
        @inbounds for k in 1:kDim
            esd.PHR[i, k] = kcw.uMish[k]
        end
        SBtransform!(kcp); SAtransform!(kcp)
        acc_b = 0.0; acc_t = 0.0
        @inbounds for j in 1:size(esd.Nbz, 2)
            acc_b += esd.Nbz[1, j] * kcp.a[j]; acc_t += esd.Nbz[2, j] * kcp.a[j]
        end
        esd.phib[i] = acc_b; esd.phit[i] = acc_t
    end

    # Galerkin load: r-weighted quadrature, radial+azimuthal metric divergence
    # ldiv = ∂r u* + u*/r + (n/r)·vaz (vaz = v_sin for the cos solve, −v_cos for
    # the sin solve — the (1/r)∂λ v azimuthal divergence of this part).
    G = esd.G
    @inbounds for i in 1:iDim, k in 1:kDim
        ldiv = esd.ux[i, k] + (U[i, k] + n * vaz[i, k]) / esd.rmet[i]
        G[i, k] = esd.Wx[i] * esd.rmet[i] * esd.Wz[k] *
                  ((PR[i, k] / esd.Pxi[k]) -
                   ts_term * ((esd.rho_tbar[k] * ldiv) + esd.phiz[i, k]))
    end
    L = esd.L
    mul!(L, esd.M0x', G * esd.M0z)
    if esd.dirichlet_u[2]
        L .+= (ts_term * esd.r_wall_r) .* esd.Nbx[2, :] *
              transpose(esd.M0z' * (esd.Wz .* esd.rho_tbar .* esd.ubr))
    end
    if esd.dirichlet_u[1]
        L .-= (ts_term * esd.r_wall_l) .* esd.Nbx[1, :] *
              transpose(esd.M0z' * (esd.Wz .* esd.rho_tbar .* esd.ubl))
    end
    if esd.strong_lid
        ispp = esd.isp_p
        for (bnd, kb) in ((esd.phib, 1), (esd.phit, size(L, 2)))
            @inbounds for i in 1:iDim
                ispp.uMish[i] = bnd[i] / ts_term
            end
            SBtransform!(ispp); SAtransform!(ispp)
            @inbounds for ib in 1:size(L, 1)
                L[ib, kb] = ispp.a[ib]
            end
        end
    else
        L .+= ts_term .* (esd.M0x' * (esd.Wx .* esd.phit)) * transpose(esd.Nbz[2, :])
        L .-= ts_term .* (esd.M0x' * (esd.Wx .* esd.phib)) * transpose(esd.Nbz[1, :])
    end
    # Axis Dirichlet load (p_n(r=0) = 0) for n ≥ 1: the operator's axis row is the
    # boundary-value row, so its RHS is 0. Applied AFTER the lid (axis wins at the
    # axis∩lid corner).
    if n > 0 && esd.axis_dirichlet
        @inbounds for kb in 1:size(L, 2)
            L[1, kb] = 0.0
        end
    end

    q = f \ vec(L)
    copyto!(esd.Q, reshape(q, size(esd.Q)))
    mul!(esd.P0, esd.M0x, esd.Q * esd.M0z')
    mul!(esd.px, esd.M1x, esd.Q * esd.M0z')
    mul!(esd.pz, esd.M0x, esd.Q * esd.M1z')
    @inbounds for i in 1:iDim, k in 1:kDim
        rho = esd.rho_tbar[k]
        phin1 = esd.PHR[i, k] - ts_term * esd.pz[i, k]
        dpp = esd.P0[i, k] - PR[i, k]
        d_rt = dpp / esd.Pxi[k]
        du[i, k] = -(ts_term / rho) * esd.px[i, k]
        dw[i, k] = phin1 / rho - W0[i, k]
        dp[i, k] = dpp
        drd[i, k] = esd.c_d[k] * d_rt - ts_term * esd.c_d_z[k] * phin1
        drt[i, k] = d_rt
        det[i, k] = esd.c_e[k] * d_rt - ts_term * esd.c_e_z[k] * phin1
    end
    return nothing
end

"""
    exact_si_solve_rlr!(hfields, esd, patch, model, t, xstar) -> hfields

Master-side RLR unsplit solve. `xstar` holds the physical predictors at the
jDim·kDim ring points, columns (u*, w*, v*, p′*). Fills the `XSI_RLR_NPLANES`
increment planes of `hfields`.
"""
function exact_si_solve_rlr!(hfields::AbstractMatrix{Float64}, esd::ExactSIDataRLR,
        patch::AbstractGrid, model::ModelParameters, t::Int64,
        xstar::AbstractMatrix{Float64})

    iDim = patch.params.iDim; kDim = patch.params.kDim
    ts_term = (t == 1) ? 0.5 * model.ts : 1.25 * model.ts
    facts = (t == 1) ? esd.facts_first : esd.facts
    nmax = esd.nmax

    # ── Forward azimuthal-only Fourier of the predictors ─────────────────────
    # per-n cos/sin (iDim×kDim) fields, zero where the ring does not support n.
    mkz() = [zeros(iDim, kDim) for _ in 0:nmax]
    ucos = mkz(); usin = mkz(); wcos = mkz(); wsin = mkz()
    vcos = mkz(); vsin = mkz(); pcos = mkz(); psin = mkz()
    _rlr_fwd!(esd, patch, xstar, 1, ucos, usin)   # column 1 = u*
    _rlr_fwd!(esd, patch, xstar, 2, wcos, wsin)   # column 2 = w*
    _rlr_fwd!(esd, patch, xstar, 3, vcos, vsin)   # column 3 = v*
    _rlr_fwd!(esd, patch, xstar, 4, pcos, psin)   # column 4 = p′*

    # per-n increment fields (index n+1), cos and sin, for each leg
    zc() = [zeros(iDim, kDim) for _ in 0:nmax]
    ducos=zc(); dusin=zc(); dwcos=zc(); dwsin=zc(); dpcos=zc(); dpsin=zc()
    drdcos=zc(); drdsin=zc(); drtcos=zc(); drtsin=zc(); detcos=zc(); detsin=zc()
    dvcos=zc(); dvsin=zc()
    P0cos=zc(); P0sin=zc()
    zero_v = zeros(iDim, kDim)

    load_part! = (u, w, pr) -> begin
        @inbounds for i in 1:iDim, k in 1:kDim
            esd.U[i,k]=u[i,k]; esd.W0[i,k]=w[i,k]
            esd.PH[i,k]=esd.rho_tbar[k]*w[i,k]; esd.PR[i,k]=pr[i,k]
        end
    end

    # Isolation lever (WIP): drop the p↔v azimuthal coupling, leaving the per-n
    # radial+vertical solve (the n=0-equivalent path, which is stable). The
    # coupled n≥1 solve is not yet stable — see the module WIP note.
    no_v = get(model.options, :xsi_rlr_no_v, false) === true
    for n in 0:nmax
        f = facts[n + 1]
        # cos part: azimuthal divergence uses +v_sin
        load_part!(ucos[n+1], wcos[n+1], pcos[n+1])
        _rlr_core!(esd, n, (n == 0 || no_v ? zero_v : vsin[n+1]), ts_term, f,
                   ducos[n+1], dwcos[n+1], dpcos[n+1], drdcos[n+1], drtcos[n+1], detcos[n+1])
        copyto!(P0cos[n+1], esd.P0)
        if n > 0
            # sin part: azimuthal divergence uses −v_cos
            load_part!(usin[n+1], wsin[n+1], psin[n+1])
            _rlr_core!(esd, n, (no_v ? zero_v : _negview(vcos[n+1])), ts_term, f,
                       dusin[n+1], dwsin[n+1], dpsin[n+1], drdsin[n+1], drtsin[n+1], detsin[n+1])
            copyto!(P0sin[n+1], esd.P0)
            # v-leg recovery (cross-part): δv_cos = −Δτ n/(rρ̄_t) p′_sin ;
            #                               δv_sin = +Δτ n/(rρ̄_t) p′_cos
            if !no_v
                @inbounds for i in 1:iDim, k in 1:kDim
                    a = ts_term * n / (esd.rmet[i] * esd.rho_tbar[k])
                    dvcos[n+1][i,k] = -a * P0sin[n+1][i,k]
                    dvsin[n+1][i,k] =  a * P0cos[n+1][i,k]
                end
            end
        end
    end

    # ── Inverse azimuthal Fourier of each leg's per-n increments → physical ──
    _rlr_inv!(esd, patch, hfields, 1, ducos, dusin)
    _rlr_inv!(esd, patch, hfields, 2, dwcos, dwsin)
    _rlr_inv!(esd, patch, hfields, 3, dpcos, dpsin)
    _rlr_inv!(esd, patch, hfields, 4, drdcos, drdsin)
    _rlr_inv!(esd, patch, hfields, 5, drtcos, drtsin)
    _rlr_inv!(esd, patch, hfields, 6, detcos, detsin)
    _rlr_inv!(esd, patch, hfields, 7, dvcos, dvsin)
    # history planes: δu/Δτ (8), δv/Δτ (9)
    @inbounds for r in 1:size(hfields, 1)
        hfields[r, 8] = hfields[r, 1] / ts_term
        hfields[r, 9] = hfields[r, 7] / ts_term
    end
    return hfields
end

# a lazily-negated view helper (avoids allocating a negated copy per n)
struct _NegView{M} <: AbstractMatrix{Float64}
    m::M
end
_negview(m) = _NegView(m)
Base.size(v::_NegView) = size(v.m)
Base.@propagate_inbounds Base.getindex(v::_NegView, i::Int, j::Int) = -v.m[i, j]

# Forward azimuthal-only Fourier of predictor column `col` of xstar → per-n
# cos/sin fields on (iDim×kDim). xstar row order is the physical ring order.
function _rlr_fwd!(esd::ExactSIDataRLR, patch::AbstractGrid,
        xstar::AbstractMatrix{Float64}, col::Int,
        fcos::Vector{Matrix{Float64}}, fsin::Vector{Matrix{Float64}})
    iDim = size(esd.U, 1); kDim = size(esd.U, 2)
    for r in 1:iDim
        ring = patch.jbasis.data[r, 1]
        bD = ring.params.bDim; km = esd.ring_kmax[r]; lp = esd.lpoints[r]
        rows = esd.rows_by_r[r]
        for k in 1:kDim
            @inbounds for l in 1:lp
                ring.uMish[l] = xstar[rows[(l - 1) * kDim + k], col]
            end
            FBtransform!(ring)
            fcos[1][r, k] = ring.b[1]
            @inbounds for n in 1:km
                fcos[n + 1][r, k] = ring.b[n + 1]
                fsin[n + 1][r, k] = ring.b[bD - n + 1]
            end
        end
    end
    return nothing
end

# Inverse azimuthal Fourier of per-n increment fields → physical increments in
# hfields[:, plane].
function _rlr_inv!(esd::ExactSIDataRLR, patch::AbstractGrid,
        hfields::AbstractMatrix{Float64}, plane::Int,
        fcos::Vector{Matrix{Float64}}, fsin::Vector{Matrix{Float64}})
    iDim = size(esd.U, 1); kDim = size(esd.U, 2)
    for r in 1:iDim
        ring = patch.jbasis.data[r, 1]
        bD = ring.params.bDim; km = esd.ring_kmax[r]; lp = esd.lpoints[r]
        rows = esd.rows_by_r[r]
        for k in 1:kDim
            fill!(ring.b, 0.0)
            ring.b[1] = fcos[1][r, k]
            @inbounds for n in 1:km
                ring.b[n + 1] = fcos[n + 1][r, k]
                ring.b[bD - n + 1] = fsin[n + 1][r, k]
            end
            FAtransform!(ring); FItransform!(ring)
            @inbounds for l in 1:lp
                hfields[rows[(l - 1) * kDim + k], plane] = ring.uMish[l]
            end
        end
    end
    return nothing
end
