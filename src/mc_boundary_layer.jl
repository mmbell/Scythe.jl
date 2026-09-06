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
    mc_louis_bl!(mtile, S, geom, colstart, colend, z, uv, wv, vv, rtv, rdv, rcv,
                 expdot, l_inf, sfc::SurfaceLayerParams, ctrans_on)

Louis boundary layer for one column of the total-energy set: eddy diffusivity
`Kv = l(z)² |∂V/∂z|` with the Blackadar-blended length `l = 1/(1/(κz) + 1/l∞)`,
applied to momentum (u, w, v), heat (the moist entropy s_t', in energy-flux form
`F_h = ρ_d T Kv ∂z s_t'` so the column energy books telescope exactly) and water
(total water ρ_w' and cloud ρ_c'; rain is left to sedimentation).
The SURFACE exchange itself is not computed here: `sfc` is the once-per-driver-call
[`SurfaceLayerParams`](@ref) and [`surface_exchange`](@ref) (src/mc_surface_layer.jl)
returns the stress `τ_u, τ_v` and the bulk fluxes `F_sh` [W/m²], `F_q` [kg/m²/s] from
the lowest mish-level state. On the DEFAULT configuration (`options[:sfc_z0] = :komori`,
`options[:sfc_stability] = false`) that returns exactly what this function used to compute
inline, bitwise: `τ = ρ_t Cd |U₁| u₁` with [`komori_cd`](@ref) when `Cd_param < 0`,
`F_sh = ρ_d1 C_pd Ck U₁ (SST − T₁)` and `F_q = Ck U₁ (ρ_vs(SST, p₁) − ρ_v1)`, the
exchange wind floored by the gustiness minimum `U_min`. `F_sh` goes into the heat column
and `F_q` into the TOTAL-water column (the surface source adds vapor, so it reaches both
ρ_w and the vapor slot). What the surface layer computes is now selectable; how it is
DELIVERED (the analytic g(z) profile below) is not, and did not change.

The increments are mapped onto the prognostic slots with the model's canonical
consistent mappings: momentum/E_t via the FRIC_KE invariant (E_t follows the
resolved KE down; no dissipative heating), heat via the QDOT_TH pattern (slot 1
`(R_m/C_vt)·Q̇`, slot 6 `+Q̇`, slot 7 through the saturation chain rule), and the
water sources via the fixed-T map of `_diffusion_water_step!` (slot 3 `+ρ̇_w`,
slot 9 `+ρ̇_c`, the vapor slot `+ρ̇_v`, slot 1 `+R_v T ρ̇_v`, slot 6
`+(C_pv T − L_v + ke + gz) ρ̇_w + (L_v − R_v T) ρ̇_v`, slot 7
`+ρ̇_v − ∂ρ_vs/∂p · R_v T ρ̇_v`, with `ρ̇_v = ρ̇_w − ρ̇_c`).

`ρ̇_v` is still built as `ρ̇_w − ρ̇_c` and that is deliberate: this closure mixes TOTAL water,
so the vapor's share of that flux is what the total carries minus what the cloud carries.
What has changed is where it LANDS — it now sources the prognostic vapor slot directly
instead of being inferred from the other two after the fact. The three legs stay mutually
consistent by construction, so the reconciliation gap this closure opens is exactly zero.

`ctrans_on` says whether slot 9 carries a control variable rather than the cloud density
(`Scythe.condensate_transform_mode`). It changes exactly two things, and BOTH ARE REQUIRED FOR
CORRECTNESS — before it existed this function was silently wrong under the transform:

* the cloud eddy flux is built from the perturbation DENSITY gradient `∂z ρ_c'`
  (`S.bl_rho_cp_z`, staged by `mc_driver!` from `S.rho_c_z = ∂z ν / J`) instead of from
  `rcv.f_z`, because an eddy flux of cloud water is a MASS flux. Only `f' = 1/J ∈ [1, 2]` is
  needed — never `f''`, which is what makes this exactly fixable where the horizontal
  Laplacian is not;
* the slot-9 increment is multiplied by `J`. Everything else — slots 1, 3, 6, 7 — is left
  alone, because with the flux built in density space `ρ̇_w`, `ρ̇_c` and `ρ̇_v = ρ̇_w − ρ̇_c` are
  all density rates again, which is what those slots want. (That subtraction was the real
  damage: a ν-rate minus a density rate, fed to the pressure, energy and Q_ss channels.)

With `ctrans_on = false` the flux line is the original one verbatim and `J ≡ 1.0`, so the
default path is bit-identical.

Rain is NOT affected: this function touches slot 8 only by borrowing its spline column as a
basis, never its values, so `rain_transform` needs nothing here.

Two limits worth knowing. With a CLOUDY reference (ρ̄_c ≠ 0) the resting state is quiet to
~1e-16 relative rather than exactly, because `J` is evaluated at `ahyp(bhyp(ρ̄_c))` whose round
trip is 2.2e-16; every current configuration has ρ̄_c ≡ 0 and
`Scythe.check_condensate_transform_ic` warns otherwise. And where `ν < 0` under `:bhyp`, the
reconstructed gradient is the SMOOTH branch's `ν_z/J`, not the clipped map's zero — deliberate:
the clipped form would zero the flux at the cloud edge and forbid the BL from ever mixing cloud
into cloud-free air, and the resulting increment is a bounded ν-space rate over a region
carrying no cloud.
"""
@noinline function mc_louis_bl!(mtile::ModelTile, S, geom::MCGeometry,
                                colstart::Int64, colend::Int64, z,
                                uv, wv, vv, rtv, rdv, rcv, expdot,
                                l_inf::Float64, sfc::SurfaceLayerParams,
                                ctrans_on::Bool)
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

    # Water: total water ρ_w' and CLOUD ρ_c', both straight from the perturbation
    # slots. The vapor's share of the TOTAL-water flux is ρ̇_v = ρ̇_w − ρ̇_c (rain is left
    # to sedimentation) — a decomposition of one eddy flux, not a retrieval of a field —
    # and it now SOURCES the prognostic vapor slot rather than being inferred from the
    # other two. Because all three legs come out of the same two column fits, this
    # closure leaves the partition exactly closed and opens no reconciliation gap.
    VDOT_w = S.VDOT_w
    rho_tp_z = rtv.f_z; rho_dp_z = rdv.f_z
    col.uMish .= Kv .* (rho_tp_z .- rho_dp_z)
    Btransform!(col)
    Atransform!(col)
    Ixtransform(col, VDOT_w)

    VDOT_c = S.VDOT_v            # (the ex-vapor scratch slot, now the cloud flux)
    # Under a transform slot 9 is ν' = bhyp(ρ_c) − bhyp(ρ̄_c), and `Kv .* rcv.f_z` would fit
    # `Kv ∂z ν'` — not a mass flux. `S.bl_rho_cp_z` is `∂z ρ_c'`, staged in mc_driver! from
    # `S.rho_c_z = ∂z ν / J`, so VDOT_c is a DENSITY rate on both branches and everything
    # downstream of it needs no further thought. See the docstring.
    if ctrans_on
        col.uMish .= Kv .* S.bl_rho_cp_z
    else
        col.uMish .= Kv .* rcv.f_z
    end
    Btransform!(col)
    Atransform!(col)
    Ixtransform(col, VDOT_c)

    # Apply to the prognostic slots (all additive; explicit loop, not view
    # broadcasts — the SubArray elision lesson at this function size). The
    # surface terms ride in on the analytic delivery profile g(z): drag −τ·g
    # into the momentum, F_sh·g into the heating, F_q·g into both water rates.
    R_m = S.R_m; C_vt = S.C_vt; Lv = S.Lv; Jc = S.Jc
    # Resolved once per tile in `MCSlots`, so this is a field load and not a name lookup.
    rv_i = mtile.mc_slots.rho_v
    drvs_dT = S.drvs_dT; drvs_dp = S.drvs_dp
    ke = S.ke
    @inbounds for i in eachindex(Kv)
        j = colstart + i - 1
        gz = z[i] < delta ? 2.0 * inv_delta * (1.0 - (z[i] * inv_delta)) : 0.0
        du = (VD_u[i] - (tau_u * gz)) / rho_t[i]
        dw = VD_w[i] / rho_t[i]
        dv = _louis_dv(geom, VD_v, rho_t, tau_v, gz, i)
        expdot[j, 4] += du
        expdot[j, 5] += dw
        _louis_add_v!(geom, expdot, j, dv)

        # Heat (QDOT_TH pattern) and the fixed-T vapor pressure source. The
        # surface moisture flux adds VAPOR, so it enters the TOTAL-water leg only and
        # reaches the vapor slot through the ρ̇_w − ρ̇_c decomposition below; the cloud
        # flux has no surface source (droplets do not evaporate off the sea surface).
        qdot = QDOT_V[i] + (F_sh * gz)
        vdot_w = VDOT_w[i] + (F_q * gz)
        vdot_c = VDOT_c[i]
        vdot_v = vdot_w - vdot_c
        dT_v = qdot / (rho_d[i] * C_vt[i])
        dp_v = (R_m[i] / C_vt[i]) * qdot
        dp_q = Rv * Tk[i] * vdot_v
        expdot[j, 1] += dp_v + dp_q
        expdot[j, 3] += vdot_w
        # The one Jacobian: slot 9 holds ν, so a density rate reaches it as J·ρ̇_c. `Jc` is an
        # exact 1.0 with no transform declared (mc_driver! `fill!(Jc, 1.0)`) and `1.0*x === x`
        # for every double, so this line is bit-identical on the default path — the same
        # argument the slot-9 FORCING makes in moist_compressible.jl.
        expdot[j, 9] += Jc[i] * vdot_c
        expdot[j, 6] += qdot +
                        (((Cpv * Tk[i]) - Lv[i] + ke[i] + (gravity * z[i])) * vdot_w) +
                        ((Lv[i] - (Rv * Tk[i])) * vdot_v) +
                        (rho_t[i] * (((u[i] * du) + (w[i] * dw)) +
                                     (_louis_v(geom, vv, i) * dv)))
        # The prognostic VAPOR slot takes the vapor's share of the flux directly. No
        # Jacobian: it is an untransformed perturbation (see `Scythe.vapor_slot`), so a
        # density rate reaches it unmodified — the one water slot for which that is true.
        expdot[j, rv_i] += vdot_v
        expdot[j, 7] += vdot_v - (drvs_dT[i] * dT_v) -
                        (drvs_dp[i] * (dp_v + dp_q))
    end
    return nothing
end
