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
# Discretization: the INTERIOR vertical flux divergences are fitted on the
# rho_r column basis (scratch_column(mtile, 8)) exactly like the sedimentation
# flux — the fitted column holds +K ∂z(field) (minus the diffusive flux) and
# the tendency is its ∂z. The SURFACE exchange (drag, enthalpy and moisture
# fluxes) is NOT delivered through a fitted bottom node: the spline fit of a
# single-node spike is basis-dependent (its extrapolated boundary value
# overshoots the node by ~13% on the RiRk basis, so the column would gain
# 1.13 F). Instead each surface flux F enters as the ANALYTIC divergence
# F·g(z) with g(z) = (2/δ)(1 − z/δ)₊ and ∫g dz = 1, distributed over the
# lowest grid cell (δ = twice the first-cell Gauss midpoint) — the column
# gains exactly F on any basis, and the Gauss quadrature integrates the
# linear profile exactly.
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

# `komori_cd` and the whole bulk surface exchange moved to src/mc_surface_layer.jl at
# stage S1b: MYNN-EDMF needs the same air-sea formulas, and a second copy of them is
# a second thing to keep in step. `surface_exchange` there is the one entry point.

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

@inline function _louis_v_flux!(VD_v, col, ::MCCartesianXZ, rho_t, Kv, vv)
    return nothing
end
@inline function _louis_v_flux!(VD_v, col, ::MCWithV, rho_t, Kv, vv)
    v_z = vv.f_z
    col.uMish .= rho_t .* Kv .* v_z
    Btransform!(col)
    Atransform!(col)
    Ixtransform(col, VD_v)
    return nothing
end

# The apply-loop v accessors (compile-time trait dispatch keeps the XZ path
# from ever touching the `nothing` views)
@inline _louis_v(::MCCartesianXZ, vv, i) = 0.0
@inline _louis_v(::MCWithV, vv, i) = @inbounds vv.f[i]
@inline _louis_dv(::MCCartesianXZ, VD_v, rho_t, tau_v, gz, i) = 0.0
@inline _louis_dv(::MCWithV, VD_v, rho_t, tau_v, gz, i) =
    @inbounds (VD_v[i] - (tau_v * gz)) / rho_t[i]
@inline _louis_add_v!(::MCCartesianXZ, expdot, j, dv) = nothing
@inline function _louis_add_v!(::MCWithV, expdot, j, dv)
    # Slot 10: rho_c was appended at 9, pushing the tangential wind to the end
    # (see the MC_VARS_CYL comment in mc_geometry.jl).
    @inbounds expdot[j, 10] += dv
    return nothing
end

"""
    mc_louis_bl!(mtile, S, geom, colstart, colend, z, uv, wv, vv, rcv,
                 expdot, l_inf, sfc::SurfaceLayerParams)

Louis boundary layer for one column of the total-energy set: eddy diffusivity
`Kv = l(z)² |∂V/∂z|` with the Blackadar-blended length `l = 1/(1/(κz) + 1/l∞)`,
applied to momentum (u, w, v), heat (the moist entropy s_t', in energy-flux form
`F_h = ρ_d T Kv ∂z s_t'` so the column energy books telescope exactly) and water (the vapor,
cloud and rain control variables, each a ν leg `Kv ∂z ν_k'` of its own; Louis refuses ice).
The SURFACE exchange itself is not computed here: `sfc` is the once-per-driver-call
[`SurfaceLayerParams`](@ref) and [`surface_exchange`](@ref) (src/mc_surface_layer.jl)
returns the stress `τ_u, τ_v` and the bulk fluxes `F_sh` [W/m²], `F_q` [kg/m²/s] from
the lowest mish-level state. On the DEFAULT configuration (`options[:sfc_z0] = :komori`,
`options[:sfc_stability] = false`) that returns exactly what this function used to compute
inline, bitwise: `τ = ρ_t Cd |U₁| u₁` with [`komori_cd`](@ref) when `Cd_param < 0`,
`F_sh = ρ_d1 C_pd Ck U₁ (SST − T₁)` and `F_q = Ck U₁ (ρ_vs(SST, p₁) − ρ_v1)`, the
exchange wind floored by the gustiness minimum `U_min`. `F_sh` goes into the heat column
and `F_q` into the vapor (`F_q g/α_v` on its ν). What the surface layer computes is
selectable; how it is DELIVERED (the analytic g(z) profile below) is not.

The increments are mapped onto the prognostic slots with the model's canonical consistent
mappings: momentum/E_t via the FRIC_KE invariant (E_t follows the resolved KE down; no
dissipative heating), heat via the QDOT_TH pattern (slot 1 `(R_m/C_vt)·Q̇`, slot 6 `+Q̇`, slot
7 through the saturation chain rule), and the water legs under the derived water anchor
([`_louis_apply_derived!`](@ref), plan E7): each species' ν rate lands on its own slot directly
and the energy carry is the fixed-temperature map on the REALIZED recovered rates. (The
`:slot`-anchor form, which mixed the prognostic ρ_t' − ρ_d' as total water and implied the
vapor as `ρ̇_w − ρ̇_c`, went with the ρ_t slot at Stage 4 of the ρ_t retirement.)

The rain leg is the slot's own `Kv ∂z ν_r'` (review D5), fitted on slot 8's column basis.

Where `ν < 0` under `:bhyp` the ν legs mix the lobe like any other ν (the recovered density reads
0 there), and the energy carry books what the recovered reading actually receives (see
`_mynn_nu_legs!` for the two consequences of mixing in ν space, which apply here too).
"""
@noinline function mc_louis_bl!(mtile::ModelTile, S, geom::MCGeometry,
                                colstart::Int64, colend::Int64, z,
                                uv, wv, vv, rcv, expdot,
                                l_inf::Float64, sfc::SurfaceLayerParams)
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

    # Surface exchange: ONE call into the shared bulk air-sea layer
    # (src/mc_surface_layer.jl). It applies `sfc_wind_factor`, floors the exchange
    # wind by the gustiness minimum U_min, and returns the stress and the bulk
    # enthalpy/moisture fluxes. With a calm surface wind the stress is still zero
    # since tau ∝ U1*u1. Latent heat is NOT a separate term — it enters through the
    # fixed-T energy bookkeeping of the vapor mass source below (the
    # (CpvT-Lv)+(Lv-RvT) = CvvT identity).
    #
    # `S.rho_v` is the PROGNOSTIC vapor mc_driver! staged for this column. The surface
    # moisture flux is a disequilibrium against rho_v_sat(SST), i.e. a thermodynamic
    # consumer of the partition, and it must read the same vapor the mixture
    # thermodynamics did. `S.p[1] / 100.0` inside the callee is bitwise `S.p_hPa[1]`
    # (mc_driver! stages `@. p_hPa = p / 100.0`).
    #
    # Scalars in, an isbits NamedTuple out, and `sfc` an immutable struct whose only
    # non-bits field is an interned Symbol: nothing here boxes. That is not obvious across
    # a @noinline boundary (the SubArray elision lesson — see the note at the call site in
    # moist_compressible.jl), so test/test_allocations.jl gates all six sfc_z0 x stability
    # arms at zero allocations per column.
    sx = surface_exchange(u[1], _louis_v1(geom, vv), Tk[1], rho_d[1], rho_t[1],
                          S.rho_v[1], S.p[1], z[1], sfc.SST, sfc)
    F_sh = sx.F_sh   # an exact 0.0 when options[:surface_fluxes] is off
    F_q = sx.F_q

    # Hold this column's exchange for the output writer (N2). A STORE and nothing else --
    # no arithmetic here changes, and on a run with the store off (`active == false`) this
    # is one Bool load and one branch. `colstart` indexes the flat gridpoint vector and one
    # column is `colend - colstart + 1 == kDim` points long, so the column index is that
    # division -- the same `div(colstart - 1, kDim) + 1` `mc_mynn_bl!` forms from `MY.kDim`.
    sd = mtile.surface
    sd.active && surface_record!(sd, div(colstart - 1, colend - colstart + 1) + 1, sx)

    # Surface-layer delivery profile g(z) = (2/δ)(1 − z/δ)₊, ∫g = 1, over the
    # lowest cell (the first-cell Gauss midpoint z[2] sits at δ/2 exactly for
    # odd-order Gauss quadrature). The bulk surface stresses/fluxes enter the
    # tendencies as F·g(z) — see the header comment for why not a fitted node.
    delta = 2.0 * z[2]
    inv_delta = 1.0 / delta
    tau_u = sx.tau_u
    tau_v = sx.tau_v

    # Interior momentum flux divergences on the rho_r column basis: fit
    # ρ_t Kv ∂z(u); ∂z of the fit is ρ_t du/dt.
    col = scratch_column(mtile, 8)
    VD_u = S.VD_u
    col.uMish .= rho_t .* Kv .* u_z
    Btransform!(col)
    Atransform!(col)
    Ixtransform(col, VD_u)

    VD_w = S.VD_w
    col.uMish .= rho_t .* Kv .* w_z
    Btransform!(col)
    Atransform!(col)
    Ixtransform(col, VD_w)

    VD_v = S.VD_v
    _louis_v_flux!(VD_v, col, geom, rho_t, Kv, vv)

    # Heat: s_t' gradient from its own column basis (slot 6, the Kvdiff_heat
    # staging pattern), then the energy-flux column ρ_d T Kv ∂z(s_t') whose ∂z is
    # the volumetric heating QDOT_V [W/m³].
    s_t = S.s_t
    q_v = S.q_v; q_l = S.q_l
    # `q_i` is staged by `mc_driver!` in the same scratch, and is an exact 0.0 column with ice
    # off, so this is bitwise the pre-ice entropy. It is only ever reached with ice ON if the
    # ice/louis_bl refusal in `mc_driver!` is removed — see there for what else must move
    # first (the implied vapor tendency below is the blocker, not this line).
    @. s_t = moist_entropy_total(Tk, rho_d, q_v, q_l, S.q_i)
    s_col = scratch_column(mtile, 6)
    s_col.uMish .= s_t .- mtile.mc_ref_diag.s_tbar
    Btransform!(s_col)
    Atransform!(s_col)
    s_z = S.bl_s_z
    Ixtransform(s_col, s_z)
    QDOT_V = S.QDOT_V
    col.uMish .= rho_d .* Tk .* Kv .* s_z
    Btransform!(col)
    Atransform!(col)
    Ixtransform(col, QDOT_V)

    # Water (plan E7, Louis, the derived water anchor): the VAPOR leg is fitted directly, in
    # ν space (`K ∂z ν_v'`, the slot's own gradient) — VDOT_w holds the vapor's ν rate.
    VDOT_w = S.VDOT_w
    rvv_z = view(mtile.tile.physical, colstart:colend, mtile.mc_slots.rho_v, siw_zslot(mtile.tile.physical))
    @inbounds for i in eachindex(Kv)
        col.uMish[i] = Kv[i] * rvv_z[i]
    end
    Btransform!(col)
    Atransform!(col)
    Ixtransform(col, VDOT_w)

    # The CLOUD leg, ν space too: `K ∂z ν_c'` (the slot's own gradient).
    VDOT_c = S.VDOT_v            # (the ex-vapor scratch slot, now the cloud flux)
    col.uMish .= Kv .* rcv.f_z
    Btransform!(col)
    Atransform!(col)
    Ixtransform(col, VDOT_c)
    # ...and the RAIN leg in ν space, `K ∂z ν_r'` (review D5): MYNN's ν-leg set (Louis refuses
    # ice, so vapor, cloud and rain). Fitted into the entropy-gradient scratch, free once
    # QDOT_V is formed above.
    rrv_z = view(mtile.tile.physical, colstart:colend, 8, siw_zslot(mtile.tile.physical))
    @inbounds for i in eachindex(Kv)
        col.uMish[i] = Kv[i] * rrv_z[i]
    end
    Btransform!(col)
    Atransform!(col)
    Ixtransform(col, S.bl_s_z)

    # Apply to the prognostic slots (all additive; explicit loops, not view broadcasts — the
    # SubArray elision lesson at this function size). The surface terms ride in on the
    # analytic delivery profile g(z): drag −τ·g into the momentum, F_sh·g into the heating,
    # F_q·g into the vapor.
    return _louis_apply_derived!(mtile, S, geom, colstart, z, uv, wv, vv, expdot,
                                 tau_u, tau_v, F_sh, F_q, delta, VD_u, VD_w, VD_v,
                                 QDOT_V, VDOT_w, VDOT_c)
end


"""
    _louis_apply_derived!(mtile, S, geom, colstart, z, uv, wv, vv, expdot, tau_u, tau_v,
                          F_sh, F_q, delta, VD_u, VD_w, VD_v, QDOT_V, VDOT_v, VDOT_c)

The Louis BL's slot increments under the derived water anchor (plan E7): the vapor, cloud and
rain legs are ν rates (`VDOT_v`, `VDOT_c`, and the rain's `K ∂z ν_r'` in `S.bl_s_z`, review D5 —
MYNN's ν-leg set; Louis refuses ice), fitted from the slots' own gradients, landing on their
slots directly; the liquid carry reads the cloud's and the rain's realized rates together.
Like the vapor and cloud legs, the rain leg's flux `K ∂z ν_r'` is fitted on the shared spline
column (slot 8's basis, natural at the ground in every production configuration), so the
diffusive flux at `z = 0` is FREE, not pinned to zero: whatever the fit puts there leaves or
enters the column (a design consequence). The surface moisture flux enters the vapor as
`F_q g/α_v`, the water sum moves by `Σ α_k ν̇_k + F_q g`, and the energy carry is the
fixed-temperature map on the REALIZED recovered rates, `δE_t = C_vv T ρ̇_v + (C_pv T − L_v) ρ̇_c + (ke + gz)·ρ̇_w` (so
`E_t − E_implied` is held to second order, the pool absorbing the water difference); `p` gets
`R_v T ρ̇_v` and the heating, `Q_ss` the realized vapor less the saturation chain rule.
"""
@noinline function _louis_apply_derived!(mtile::ModelTile, S, geom::MCGeometry,
                                         colstart::Int64, z, uv, wv, vv, expdot,
                                         tau_u::Float64, tau_v::Float64, F_sh::Float64,
                                         F_q::Float64, delta::Float64, VD_u, VD_w, VD_v,
                                         QDOT_V, VDOT_v, VDOT_c)
    opts = mtile.model.options; pp = mtile.model.physical_params
    ws = water_anchor_set(opts, pp, mtile.mc_slots)
    av = ws.alpha[1]; ac = ws.alpha[2]; ar = ws.alpha[3]
    ctr = condensate_transform_mode(opts); cmu = get(pp, :condensate_mu, 1.0e-7)
    rtr = rain_transform_mode(opts); rmu = get(pp, :rain_mu, 1.0e-7)
    VDOT_r = S.bl_s_z               # the rain's ν rate (`mc_louis_bl!`)
    vtr = vapor_transform_mode(opts); vmu = vapor_mu(pp)
    ts = mtile.model.ts
    phys = mtile.tile.physical
    rv_i = mtile.mc_slots.rho_v
    nvb = mc_nu_vbar(mtile)
    u = uv.f; w = wv.f
    rho_t = S.rho_t; rho_d = S.rho_d; Tk = S.Tk
    R_m = S.R_m; C_vt = S.C_vt; Lv = S.Lv
    inv_delta = 1.0 / delta
    @inbounds for i in eachindex(VD_u)
        j = colstart + i - 1
        gz = z[i] < delta ? 2.0 * inv_delta * (1.0 - (z[i] * inv_delta)) : 0.0
        du = (VD_u[i] - (tau_u * gz)) / rho_t[i]
        dw = VD_w[i] / rho_t[i]
        dv = _louis_dv(geom, VD_v, rho_t, tau_v, gz, i)
        expdot[j, 4] += du
        expdot[j, 5] += dw
        _louis_add_v!(geom, expdot, j, dv)
        qdot = QDOT_V[i] + (F_sh * gz)
        Fv = F_q * gz
        nvv = VDOT_v[i] + (Fv / av)
        ncc = VDOT_c[i]
        nuv = phys[j, rv_i, 1] + nvb[i]
        real_v = (recover_rho_v(ts * nvv, nuv, vtr, vmu) - recover_rho_v(0.0, nuv, vtr, vmu)) / ts
        real_c = _mynn_realized_rate(S.nu_c[i], ncc, ncc, ts, ctr, cmu)
        sw = ((av * VDOT_v[i]) + (ac * ncc)) + Fv
        # the rain leg (D5): its realized recovered rate joins the liquid's carry, its ν
        # rate the anchor; only where it is nonzero (a K = 0 or rainless column is bitwise)
        nrr = VDOT_r[i]
        if nrr != 0.0
            real_c += _mynn_realized_rate(S.nu_r[i], nrr, nrr, ts, rtr, rmu)
            sw += ar * nrr
            expdot[j, 8] += nrr
        end
        geo = S.ke[i] + (gravity * z[i])
        dT_v = qdot / (rho_d[i] * C_vt[i])
        dp_v = (R_m[i] / C_vt[i]) * qdot
        dp_q = Rv * Tk[i] * real_v
        expdot[j, 1] += dp_v + dp_q
        expdot[j, 9] += ncc
        expdot[j, 6] += qdot + ((((Cvv * Tk[i]) * real_v) + (((Cpv * Tk[i]) - Lv[i]) * real_c)) +
                                (geo * sw)) +
                        (rho_t[i] * (((u[i] * du) + (w[i] * dw)) + (_louis_v(geom, vv, i) * dv)))
        expdot[j, rv_i] += nvv
        expdot[j, 7] += real_v - (S.drvs_dT[i] * dT_v) - (S.drvs_dp[i] * (dp_v + dp_q))
    end
    return nothing
end
