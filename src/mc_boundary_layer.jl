# ── Louis boundary layer + surface exchange for the total-energy set ───────────
#
# Options-gated (options[:louis_bl], default false) explicit vertical mixing of
# momentum, heat and water with a Louis (1979)-type prescribed mixing length and
# wind-speed-dependent surface drag (Komori et al. 2018, J. Phys. Oceanogr.).
# Everything lives in mc_louis_bl!, called once per column AFTER every expdot
# slot is written: all of its contributions are additive (the Q_ss saturation
# chain rule is linear in the non-condensation T/p tendencies), so the frozen
# driver lines are untouched and louis_bl = false is bit-identical.
#
# Discretization: every vertical flux divergence is fitted on the rho_r column
# basis (scratch_column(mtile, 8)) exactly like the sedimentation flux — that
# basis' bottom BC decides whether the surface node is free to carry a flux, so
# boundary-layer configurations give rho_r a NaturalBC bottom (as the O01
# rainfall benchmark already does). Sign convention: the fitted column holds
# +K ∂z(field) (minus the diffusive flux), the tendency is its ∂z, and the
# column integral of the tendency telescopes to col(top) − col(bottom). Hence
# the momentum drag enters as a POSITIVE bottom node +ρ_t Cd |U| u (a momentum
# sink) and a scalar surface GAIN enters as a negative bottom node −F_sfc.
#
# Explicit, not implicit: Kv(z,t) is state-dependent and the implicit Helmholtz
# matrices are factorized once with constant K (createModelTile); the explicit
# stability limit ts < dz²/(2 Kv) ≈ 60 s at dz = 250 m, Kv = 500 m²/s is far
# above any acoustically-limited model timestep.
#
# The mixing acts on PERTURBATIONS from the reference state (s_t − s_tbar,
# ρ_v − ρ_vbar, ρ_w' from the perturbation slots), the same convention as the
# implicit diffusion channels — a resting base state is exactly steady.

"Blackadar-blended Louis mixing length l = 1/(1/(κz) + 1/l∞), κ = 0.4; 0 at z = 0."
@inline louis_length(z, l_inf) = 1.0 / ((1.0 / (0.4 * z)) + (1.0 / l_inf))

"""
    komori_cd(U)

Wind-speed-dependent surface drag coefficient, Komori et al. (2018): 1.0e-3 below
5.2 m/s, 4.4e-4 U^0.5 to 33.6 m/s, capped at 2.55e-3 in the high-wind regime.
Selected by a NEGATIVE configured `:Cd` (the tcbl sentinel convention); a positive
`:Cd` is used as a constant, and `:Cd = 0` disables the drag.
"""
@inline function komori_cd(U)
    if U < 5.2
        return 1.0e-3
    elseif U < 33.6
        return 4.4e-4 * sqrt(U)
    end
    return 2.55e-3
end

# Vertical shear magnitude of the horizontal wind — the Louis closure is
# shear-only (prescribed length, no Richardson correction).
@inline function _louis_shear!(Ssh, ::MCCartesianXZ, u_z, vv)
    @. Ssh = abs(u_z)
    return nothing
end
@inline function _louis_shear!(Ssh, ::MCWithV, u_z, vv)
    v_z = vv.f_z
    @. Ssh = sqrt((u_z * u_z) + (v_z * v_z))
    return nothing
end

# Surface v and the v-momentum flux column (inert on the Cartesian slice)
@inline _louis_v1(::MCCartesianXZ, vv) = 0.0
@inline _louis_v1(::MCWithV, vv) = vv.f[1]

@inline function _louis_v_flux!(VD_v, col, ::MCCartesianXZ, rho_t, Kv, vv,
                                drag_coeff, v1)
    return nothing
end
@inline function _louis_v_flux!(VD_v, col, ::MCWithV, rho_t, Kv, vv,
                                drag_coeff, v1)
    v_z = vv.f_z
    col.uMish .= rho_t .* Kv .* v_z
    col.uMish[1] = rho_t[1] * drag_coeff * v1
    Btransform!(col)
    Atransform!(col)
    Ixtransform(col, VD_v)
    return nothing
end

# The apply-loop v accessors (compile-time trait dispatch keeps the XZ path
# from ever touching the `nothing` views)
@inline _louis_v(::MCCartesianXZ, vv, i) = 0.0
@inline _louis_v(::MCWithV, vv, i) = @inbounds vv.f[i]
@inline _louis_dv(::MCCartesianXZ, VD_v, rho_t, i) = 0.0
@inline _louis_dv(::MCWithV, VD_v, rho_t, i) = @inbounds VD_v[i] / rho_t[i]
@inline _louis_add_v!(::MCCartesianXZ, expdot, j, dv) = nothing
@inline function _louis_add_v!(::MCWithV, expdot, j, dv)
    @inbounds expdot[j, 9] += dv
    return nothing
end

"""
    mc_louis_bl!(mtile, S, geom, colstart, colend, z, uv, wv, vv, rtv, rdv,
                 expdot, l_inf, Cd_param, sfc_fac)

Louis boundary layer for one column of the total-energy set: eddy diffusivity
`Kv = l(z)² |∂V/∂z|` with the Blackadar-blended length `l = 1/(1/(κz) + 1/l∞)`,
applied to momentum (u, w, v), heat (the moist entropy s_t', in energy-flux form
`F_h = ρ_d T Kv ∂z s_t'` so the column energy books telescope exactly) and water
(total water ρ_w' and diagnostic vapor ρ_v'; rain is left to sedimentation).
Surface momentum drag `τ = ρ_t Cd |U₁| u₁` on the lowest mish-level wind
([`komori_cd`](@ref) when `Cd_param < 0`); the scalar surface nodes are zero here
and are filled by the surface-flux stage (options[:surface_fluxes]).

The increments are mapped onto the prognostic slots with the model's canonical
consistent mappings: momentum/E_t via the FRIC_KE invariant (E_t follows the
resolved KE down; no dissipative heating), heat via the QDOT_TH pattern (slot 1
`(R_m/C_vt)·Q̇`, slot 6 `+Q̇`, slot 7 through the saturation chain rule), and the
water sources via the fixed-T map of `_diffusion_water_step!` (slot 3 `+ρ̇_w`,
slot 1 `+R_v T ρ̇_v`, slot 6 `+(C_pv T − L_v + ke + gz) ρ̇_w + (L_v − R_v T) ρ̇_v`,
slot 7 `+ρ̇_v − ∂ρ_vs/∂p · R_v T ρ̇_v`).
"""
@noinline function mc_louis_bl!(mtile::ModelTile, S, geom::MCGeometry,
                                colstart::Int64, colend::Int64, z,
                                uv, wv, vv, rtv, rdv, expdot,
                                l_inf::Float64, Cd_param::Float64,
                                sfc_fac::Float64)
    u = uv.f; u_z = uv.f_z
    w = wv.f; w_z = wv.f_z
    rho_t = S.rho_t
    rho_d = S.rho_d
    Tk = S.Tk

    # Louis eddy diffusivity Kv = l(z)^2 |dV/dz| (Kv temporarily holds the shear)
    Kv = S.Kv
    _louis_shear!(Kv, geom, u_z, vv)
    @inbounds for i in eachindex(Kv)
        l = louis_length(z[i], l_inf)
        Kv[i] = l * l * Kv[i]
    end

    # Surface drag: bulk stress on the lowest mish-level wind
    u1 = u[1] * sfc_fac
    v1 = _louis_v1(geom, vv) * sfc_fac
    U1 = sqrt((u1 * u1) + (v1 * v1))
    Cd = Cd_param < 0.0 ? komori_cd(U1) : Cd_param
    drag_coeff = Cd * U1

    # Momentum flux divergences on the rho_r column basis: fit ρ_t Kv ∂z(u) with
    # the surface stress as the bottom node; ∂z of the fit is ρ_t du/dt.
    col = scratch_column(mtile, 8)
    VD_u = S.VD_u
    col.uMish .= rho_t .* Kv .* u_z
    col.uMish[1] = rho_t[1] * drag_coeff * u1
    Btransform!(col)
    Atransform!(col)
    Ixtransform(col, VD_u)

    VD_w = S.VD_w
    col.uMish .= rho_t .* Kv .* w_z
    col.uMish[1] = 0.0
    Btransform!(col)
    Atransform!(col)
    Ixtransform(col, VD_w)

    VD_v = S.VD_v
    _louis_v_flux!(VD_v, col, geom, rho_t, Kv, vv, drag_coeff, v1)

    # Heat: s_t' gradient from its own column basis (slot 6, the Kvdiff_heat
    # staging pattern), then the energy-flux column ρ_d T Kv ∂z(s_t') whose ∂z is
    # the volumetric heating QDOT_V [W/m³].
    s_t = S.s_t
    q_v = S.q_v; q_l = S.q_l
    @. s_t = moist_entropy_total(Tk, rho_d, q_v, q_l)
    s_col = scratch_column(mtile, 6)
    s_col.uMish .= s_t .- mtile.mc_ref_diag.s_tbar
    Btransform!(s_col)
    Atransform!(s_col)
    s_z = S.bl_s_z
    Ixtransform(s_col, s_z)
    QDOT_V = S.QDOT_V
    col.uMish .= rho_d .* Tk .* Kv .* s_z
    col.uMish[1] = 0.0                       # surface enthalpy flux stage fills this
    Btransform!(col)
    Atransform!(col)
    Ixtransform(col, QDOT_V)

    # Water: diagnostic vapor ρ_v' via its column fit (slot 3, the Kvdiff_water
    # staging pattern); total water ρ_w' straight from the perturbation slots.
    rho_v = S.rho_v
    v_col = scratch_column(mtile, 3)
    v_col.uMish .= rho_v .- mtile.mc_ref_diag.rho_vbar
    Btransform!(v_col)
    Atransform!(v_col)
    rv_z = S.bl_rv_z
    Ixtransform(v_col, rv_z)
    VDOT_v = S.VDOT_v
    col.uMish .= Kv .* rv_z
    col.uMish[1] = 0.0                       # surface moisture flux stage fills this
    Btransform!(col)
    Atransform!(col)
    Ixtransform(col, VDOT_v)

    VDOT_w = S.VDOT_w
    rho_tp_z = rtv.f_z; rho_dp_z = rdv.f_z
    col.uMish .= Kv .* (rho_tp_z .- rho_dp_z)
    col.uMish[1] = 0.0
    Btransform!(col)
    Atransform!(col)
    Ixtransform(col, VDOT_w)

    # Apply to the prognostic slots (all additive; explicit loop, not view
    # broadcasts — the SubArray elision lesson at this function size)
    R_m = S.R_m; C_vt = S.C_vt; Lv = S.Lv
    drvs_dT = S.drvs_dT; drvs_dp = S.drvs_dp
    ke = S.ke
    @inbounds for i in eachindex(Kv)
        j = colstart + i - 1
        du = VD_u[i] / rho_t[i]
        dw = VD_w[i] / rho_t[i]
        dv = _louis_dv(geom, VD_v, rho_t, i)
        expdot[j, 4] += du
        expdot[j, 5] += dw
        _louis_add_v!(geom, expdot, j, dv)

        # Heat (QDOT_TH pattern) and the fixed-T vapor pressure source
        dT_v = QDOT_V[i] / (rho_d[i] * C_vt[i])
        dp_v = (R_m[i] / C_vt[i]) * QDOT_V[i]
        dp_q = Rv * Tk[i] * VDOT_v[i]
        expdot[j, 1] += dp_v + dp_q
        expdot[j, 3] += VDOT_w[i]
        expdot[j, 6] += QDOT_V[i] +
                        (((Cpv * Tk[i]) - Lv[i] + ke[i] + (gravity * z[i])) * VDOT_w[i]) +
                        ((Lv[i] - (Rv * Tk[i])) * VDOT_v[i]) +
                        (rho_t[i] * (((u[i] * du) + (w[i] * dw)) +
                                     (_louis_v(geom, vv, i) * dv)))
        expdot[j, 7] += VDOT_v[i] - (drvs_dT[i] * dT_v) -
                        (drvs_dp[i] * (dp_v + dp_q))
    end
    return nothing
end
