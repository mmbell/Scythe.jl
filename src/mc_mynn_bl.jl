# ── MYNN-EDMF boundary layer applied in Scythe's variables ────────────────────
#
# The APPLY half of the MYNN-EDMF coupling (plan D1-D3, D7, D9, D12). Everything here
# runs inside `mc_mynn_bl!`, called once per column from the plug-in point in
# `mc_driver!` AFTER every expdot slot is written -- exactly where `mc_louis_bl!` sits,
# and for the same reason: every contribution is additive (the Q_ss saturation chain
# rule is linear in the non-condensation T/p tendencies), so the driver lines above stay
# frozen and a run without `options[:mynn]` is byte-identical.
#
# WHAT THIS FILE IS NOT. It is not a second closure. `src/mynn_closure.jl` is the
# bitwise Fortran port on plain vectors; this file is the coupling: it builds that
# closure's inputs from the prognostic state, calls it on the cadence, and turns what it
# returns -- the mixing length `el`, the stability functions `sm`/`sh` and the buoyancy
# function `gh` -- into eddy fluxes of Scythe's OWN variables, fitted and differentiated
# on the spline column exactly as the Louis BL does.
#
# ── The three things that make this different from Louis ─────────────────────
#
#  1. K IS PROGNOSTIC. `K_m = l q S_M`, `K_h = l q S_H`, `K_e = 3 K_m` with
#     `q = sqrt(2e)` read from the `rho_e` SLOT every step. Between closure updates
#     `l`, `S_M`, `S_H` are held and only `q` moves, so the diffusivity tracks the
#     turbulence the model is actually carrying rather than a prescribed length scale.
#
#  2. THE TKE IS A PROGNOSTIC ENERGY, AND THE COLUMN BOOKS CLOSE. `rho_e` gains the
#     discrete shear production, the closure's buoyancy exchange, the surface production
#     and the fitted turbulent transport, and loses the dissipation; `E_t` receives the
#     mirror image of every one of those exchanges. The identity that has to hold, and
#     that test/test_mynn_bl.jl asserts to round-off on two grids and two timesteps, is
#
#       <dE_t + drho_e> = [u S_u + w S_w + v S_v + S_h + S_Ew + S_e]_0^{ztop}
#                         + F_sh + <(c_w + c_v) F_q g>
#
#     with `<.>` the Gauss quadrature on the mish. It is exact, not approximate, and the
#     reason is a quadrature fact: three-node Gauss is exact to degree 5, every fitted
#     flux column is a cubic per cell, and both `u d/dz(S_u)` (3x2) and `u_z S_u` (2x3)
#     are degree 5. Their sum is `d/dz(u S_u)`, so the quadrature of the pair telescopes
#     to the boundary values of the PRODUCT -- and the product's boundary value is the one
#     number the model cannot hand back, because `u`'s own spline coefficients are not
#     exposed per column and the fit carries a filter, so refitting `u` is not the same
#     function. So the product `Psi = u S_u + w S_w + v S_v` is FITTED as a column of its
#     own and the shear production is defined as `rho P_s := dz(Psi) - V . dz(S_V)`, which
#     is the same quantity by the product rule and makes the identity exact by
#     construction. What it is NOT is the continuum `rho K_m |dV/dz|^2`: that form leaves
#     `<pdk - rho P_s>` unclosed, and an unclosed exchange between the resolved and the
#     subgrid energy is a leak nothing reports. `P_s,mynn = rho K_m gm` is kept as a
#     DIAGNOSTIC beside what the slot actually received, and their residual is reported.
#
#  3. THE WATER LEGS ARE SPECIES-WISE (D7). Louis mixes total water and cloud and infers
#     the vapor as the difference. Here each of rho_w', rho_v', rho_c' and rho_r gets its
#     own fitted flux, and the energy the water carries goes in FLUX form
#     (`:mynn_water_carry = :flux`): `S_Ew = Fit[c_w K_h rho_w,z + c_v K_h rho_v,z]`, so
#     E_t receives a divergence that telescopes, and the part of it that is not the
#     fixed-T enthalpy of the transported mass -- `Qdot_w = dS_Ew - c_w dS_w - c_v dS_v`,
#     which is `S_w dc_w/dz + S_v dc_v/dz` by the product rule -- is genuine thermal
#     heating and drives T, p and Q_ss. `:fixed_T` restores the local Louis map and is
#     kept as the fidelity comparison (D10).
#
# ── Explicit in time, like Louis, and why the census exists ──────────────────
# The vertical mixing is applied EXPLICITLY. The offline harness
# (model_tests/MYNN_REPLAY_README.md) measured `D_gal = K_e ts 10/dz_cell^2` on real
# model states: 0.16 on the ocean bubble at its own 0.5 s timestep, 0.0004-0.002 on the
# TC nest, but 0.52 at O01's 250 m / 0.3 s combination. So the explicit application is
# comfortable at production spacing and BINDS in the gray zone -- which is a statement
# about resolution and timestep, not about a case (feedback_resolution_general_numerics).
# Nothing here assumes either: every limit is computed per column for the actual grid and
# timestep, counted in `MYNNState`, and the same numbers are printed as a setup advisory
# beside `warn_timestep_stability` (src/reference_state.jl) before a run starts.
#
# ── Surface delivery: unchanged from Louis ───────────────────────────────────
# `surface_exchange` (src/mc_surface_layer.jl) is the ONE air-sea layer; MYNN never
# computes its own fluxes (in WRF/UFS a separate surface-layer scheme hands it
# `ust, hfx, qfx, 1/L`, and that is what happens here). The stress and the enthalpy /
# moisture fluxes are delivered as the analytic divergence `F g(z)` with
# `g(z) = (2/delta)(1 - z/delta)_+` over the lowest cell, NOT through a fitted bottom
# node -- see the header of src/mc_boundary_layer.jl for the 13 % overshoot that rule
# exists to avoid.
#
# ── Perturbation form ────────────────────────────────────────────────────────
# Every mixed quantity is a PERTURBATION from the reference state (s_t', rho_w', rho_v',
# rho_c'), so a resting reference column is exactly steady, and the fluxes are
# disequilibrium-form so a calm column over a sea at its own temperature and humidity
# produces exactly 0.0 -- not "small". test/test_mynn_bl.jl pins that as an equality.

"MYNN's land/sea flag. Every Scythe moist-compressible configuration is over water."
const MYNN_XLAND_WATER = 2.0

"Reference pressure for the Exner function [Pa]."
const MYNN_P0 = 100000.0

# The tangential-wind accessors, the same compile-time trait dispatch mc_boundary_layer.jl
# uses: no method body on the Cartesian slice ever touches the `nothing` views or the
# inert `VD_v` scratch column.
@inline _mynn_vz(::MCCartesianXZ, vv, i) = 0.0
@inline _mynn_vz(::MCWithV, vv, i) = @inbounds vv.f_z[i]
@inline _mynn_vdiv(::MCCartesianXZ, VD_v, i) = 0.0
@inline _mynn_vdiv(::MCWithV, VD_v, i) = @inbounds VD_v[i]

@inline function _mynn_v_flux!(VD_v, Sv, col, ::MCCartesianXZ, rho_t, Km, vv)
    return nothing
end
@inline function _mynn_v_flux!(VD_v, Sv, col, ::MCWithV, rho_t, Km, vv)
    v_z = vv.f_z
    col.uMish .= rho_t .* Km .* v_z
    Btransform!(col)
    Atransform!(col)
    Ixtransform(col, VD_v)
    SItransform!(col)
    copyto!(Sv, col.uMish)
    return nothing
end

"""
    _mynn_edges(col) -> (S(z_bottom), S(z_top))

The fitted column's value at the two ends of the spline domain. These are the boundary
terms of the D3 energy identity and they are REPORTED, never assumed zero: the spline's
extrapolated wall value is exactly the 13 %-overshoot quantity the surface delivery
profile exists to avoid relying on, so the budget has to carry whatever it actually is.
"""
@inline function _mynn_edges(col::Springsteel.CubicBSpline.Spline1D)
    sp = col.params
    return (Springsteel.CubicBSpline.SItransform(sp, col.a, sp.xmin, 0),
            Springsteel.CubicBSpline.SItransform(sp, col.a, sp.xmax, 0))
end

"""
    mc_bl_apply!(mtile, S, geom, colstart, colend, z, uv, wv, vv, rtv, rdv, rcv, rrv,
                 rvv, rev, expdot, l_inf, sfc, t, ctrans_on, rtrans_on, louis_bl, mynn_on)

The ONE boundary-layer application point of `mc_driver!`: the shared perturbation-density
staging, then whichever closure is configured. The two are mutually exclusive — the driver
preamble refuses `louis_bl && mynn_on` — so this is a dispatch, not a sum.

WHY THE TWO CLOSURES SHARE A CALL SITE. This is a codegen constraint, measured, not a
tidiness argument. `mc_driver!` is inlined into the equation-set body and is enormous;
its per-column temporaries are already at the edge of what the register allocator can
carry across the `@turbo` vertical-diffusion broadcasts that follow the boundary layer.
Adding the MYNN call as a SECOND `@noinline` call beside the Louis one — nine
`mc_slot_views` bundles, ~60 `SubArray`s, live across that block — pushed LLVM's
MachineScheduler over a register-pressure cliff: `sample` put 100 % of the time in
`RegPressureTracker::bumpDownwardPressure` and the compile never finished (>30 min on the
first `moist_compressible_XZ` instance of test/test_allocations.jl, against ~2 min for the
same instance with one call site). Neither shrinking the argument list to nine, nor
stubbing the callee's body, nor moving the block past the diffusion broadcasts made any
difference; collapsing the two call sites into one did, immediately.

So: exactly one call, and any future closure belongs INSIDE this function rather than
beside it. The staging is hoisted here too, which is why `rho_cbar_z` is rebuilt from
`mtile.ref_state` — the view is local to this frame and does not escape, so it costs
nothing, and `mc_driver!` no longer carries the reference-gradient view at all.

Bit-identical to the two separate call sites it replaces: same staging expression, same
arguments, same order.
"""
@noinline function mc_bl_apply!(mtile::ModelTile, S, geom::MCGeometry,
                                colstart::Int64, colend::Int64, z,
                                uv, wv, vv, rtv, rdv, rcv, rrv, rvv, rev, expdot,
                                l_inf::Float64, sfc::SurfaceLayerParams, t::Int64,
                                ctrans_on::Bool, rtrans_on::Bool,
                                louis_bl::Bool, mynn_on::Bool)
    if ctrans_on
        # The BL's cloud eddy flux is a MASS flux, so it is built from the perturbation
        # DENSITY gradient and not from the slot's: `S.rho_c_z` is the TOTAL ∂z ρ_c =
        # ∂z ν / J, and this is its first consumer. NOT used on the `:none` path -- there
        # `rho_c_z` is `rho_cp_z + rho_cbar_z` and subtracting `rho_cbar_z` back off is
        # not bitwise `rho_cp_z`, so both callees keep reading `rcv.f_z` directly when the
        # transform is off, which is.
        rho_cbar_z = view(Springsteel.ref_rho_c(mtile.ref_state), :, 2)
        @. S.bl_rho_cp_z = S.rho_c_z - rho_cbar_z
    end
    if louis_bl
        mc_louis_bl!(mtile, S, geom, colstart, colend, z, uv, wv, vv, rtv, rdv, rcv,
                     expdot, l_inf, sfc, ctrans_on)
    elseif mynn_on
        mc_mynn_bl!(mtile, S, geom, colstart, colend, z, uv, wv, vv, rtv, rdv, rcv,
                    rrv, rvv, rev, expdot, sfc, t, ctrans_on, rtrans_on)
    end
    return nothing
end

"""
    mc_mynn_bl!(mtile, S, geom, colstart, colend, z, uv, wv, vv, rtv, rdv, rcv, rrv,
                rvv, rev, expdot, sfc, t, ctrans_on, rtrans_on)

One column of the MYNN-EDMF boundary layer for the total-energy set.

# What runs when (D9)
EVERY step: the closure inputs are re-gathered from the current state, `q = sqrt(2e)` is
read from the `rho_e` slot, `K_m = l q S_M`, `K_h = l q S_H`, `K_e = 3 K_m` are formed
from the HELD `l`, `S_M`, `S_H`, the ~10 flux columns are fitted and differentiated, and
the slot increments are applied.

On an UPDATE step (`t - last_update_step >= interval_steps`, and always the first call on
a column): `GET_PBLH` -> `SCALE_AWARE` -> the surface block (`flt`, `fltv`, `rmol`, the
`zeta`-dependent `pmz`/`phh`) -> `mym_condensation!` -> `mym_turbulence!`, in exactly the
order `mynn_column_step!` (src/mynn_closure.jl) replays the Fortran driver, and the
results (`el, sm, sh, vt, vq, sgm, cldfra_bl, qc_bl, qi_bl, gh, qsq, pblh, kpbl`) are
stored on the tile. `DMP_mf` is NOT called (`:mynn_edmf = 0`; the plumes arrive at S7),
which is the column the Fortran produces for `ktop_plume = 0`.

`mym_predict!` is deliberately NOT called: the TKE equation is Scythe's own prognostic
`rho_e` slot, integrated by the model's multistep with the closure's production and
dissipation as sources. Its Level-2 `qsq` diagnostic IS reproduced here (six lines, the
`closure <= 2.5` branch of :3543-3559) because `mym_condensation!` reads `qsq` to set the
cloud-PDF width; `tsq` and `cov` are not, at CASE 2, read by anything.

# Two Fortran shapes reproduced on purpose (`:mynn_fidelity = :fortran`, D10)
  * `th_sfc = ts/exner(1)` where the driver's `ts` is ALREADY `T_sfc/exner(1)`
    (mynn_bl_driver :1063; harness README item 2). It divides by the surface Exner
    function twice, which inflates the surface potential temperature by ~3 % and so
    inflates `fltv`, `rmol` and through them the mixing length. Reproduced because the
    parity anchor is the Fortran; `:mynn_fidelity = :scythe` is where it gets fixed,
    after the deviation has been measured (D10), not before.
  * `rmol` is RECOMPUTED here from `fltv` and `ust` (:1079) rather than taken from
    `surface_exchange`'s Monin-Obukhov `inv_L`, so the closure's `zeta`, `pmz` and `phh`
    are self-consistent with the fluxes it was handed.

# The energy budget (D3), term by term
Momentum `du = (dS_u - tau_u g)/rho_t` with `dE_t^mom = rho_t(u du + w dw + v dv)` -- the
FRIC_KE invariant: E_t follows the resolved KE down, no dissipative heating. The KE that
leaves the resolved flow arrives in `rho_e` as the DISCRETE shear production
`rho P_s = dz(Psi) - (u dz(S_u) + w dz(S_w) + v dz(S_v))` with `Psi` the fitted
momentum-energy flux column (the product-rule form of `u_z S_u + ...`; see the file
header and `_mynn_flux_columns!` for why the product is fitted rather than formed
pointwise), and as `P_sfc = (tau_u u + tau_v v) g`, which cancels the drag work
POINTWISE rather than in the column integral. The closure's buoyancy
exchange `rho P_b = rho_t K_h G_H` and the dissipation `rho eps = rho_t q^3/(B_1 l)` are
equal and opposite in the two reservoirs. Heat is FLUX form,
`S_h = rho_d T K_h dz(s_t')`, so the column books telescope; `Qdot_E = Qdot_h - rho P_b +
rho eps` reaches E_t directly and (with the water's thermal part) T, p and Q_ss through
the QDOT_TH pattern.

`ctrans_on` / `rtrans_on` say whether slots 9 / 8 carry control variables. As in
`mc_louis_bl!` they change exactly two things per slot, and BOTH are required for
correctness: the eddy flux is built from the perturbation DENSITY gradient (an eddy flux
of cloud water is a MASS flux, not a flux of `nu`), and the slot increment is multiplied
by the Jacobian. `Jc`/`Jr` are an exact 1.0 with no transform declared, so the default
path is bit-identical.

Ice is REFUSED upstream (the `mc_driver!` preamble): the closure mixes one ice mixing
ratio while ISHMAEL carries three species with four moments each, so `q_i` is an exact
0.0 column here and the ice legs arrive at S8.
"""
@noinline function mc_mynn_bl!(mtile::ModelTile, S, geom::MCGeometry,
                               colstart::Int64, colend::Int64, z,
                               uv, wv, vv, rtv, rdv, rcv, rrv, rvv, rev, expdot,
                               sfc::SurfaceLayerParams, t::Int64,
                               ctrans_on::Bool, rtrans_on::Bool)
    MY = mtile.mynn
    n = colend - colstart + 1
    ci = div(colstart - 1, MY.kDim) + 1
    work = MY.work[Threads.threadid()]
    MN = MY.colscratch[Threads.threadid()]
    rho_t = S.rho_t
    rho_e = rev.f

    # ── Surface exchange: ONE call into the shared bulk air-sea layer ─────────
    # Identical to the Louis call site (src/mc_boundary_layer.jl) down to the argument
    # order, because it IS the same function: `mc_mynn_bl!` and `mc_louis_bl!` differ in
    # the closure, never in the air-sea formulas or in how their fluxes are delivered.
    sx = surface_exchange(uv.f[1], _louis_v1(geom, vv), S.Tk[1], S.rho_d[1], rho_t[1],
                          S.rho_v[1], S.p[1], z[1], sfc.SST, sfc)
    F_sh = sx.F_sh              # an exact 0.0 when options[:surface_fluxes] is off
    F_q = sx.F_q

    n_clamp = _mynn_column_inputs!(MY, S, MN, work, geom, uv, wv, vv, rho_e, n)

    # ── Cold start (`:mynn_init = :taper`) ───────────────────────────────────
    first_call = MY.last_update_step[ci] == typemin(Int)
    re_i = mtile.mc_slots.rho_e
    if first_call && MY.init_mode === :taper
        _mynn_taper_seed!(MY, MN, expdot, rho_e, rho_t, sx.ust, colstart, n, re_i)
    end

    # ── The held closure, on the cadence (D9) ────────────────────────────────
    if first_call || (t - MY.last_update_step[ci]) >= MY.interval_steps
        _mynn_closure_update!(MY, S, MN, work, colstart, n, ci, sfc.SST, sx.ust,
                              F_sh, F_q, t)
    end

    _mynn_diffusivities!(MY, MN, colstart, n, ci, n_clamp)

    ed = _mynn_flux_columns!(mtile, MY, S, MN, geom, colstart, n, z, uv, wv, vv,
                             rtv, rdv, rcv, rrv, rvv, rho_e, ctrans_on, rtrans_on)

    _mynn_apply_column!(MY, S, MN, geom, colstart, n, ci, z, uv, wv, vv, expdot,
                        mtile.mc_slots.rho_v, re_i, sx.tau_u, sx.tau_v, F_sh, F_q,
                        2.0 * z[2], ed)
    return nothing
end

"""
    _mynn_column_inputs!(MY, S, MN, work, geom, uv, wv, vv, rho_e, n) -> n_clamp

Build the closure's inputs from the retrieved state and read the TKE out of the slot.

`sqv`/`sqc`/`sqi` are SPECIFIC contents (`rho_x/rho_t`) and `rho` is the MOIST density,
the conventions of `mynnedmf_wrapper.F90` that `tools/mynn_dump_columns.jl` dumped the
Fortran reference columns in. These lines are `_mynn_gather!` (src/mynn_closure.jl) with
the frozen column replaced by the live state, and they leave the working copies in
`work.s_*` exactly where that routine leaves them, so `mym_condensation!` and
`mym_turbulence!` see the argument set the parity harness validated.

Two TKE variables come out of the one slot, and keeping them apart is what makes a
resting column produce EXACTLY zero rather than the floor's residue:

  * `MN.q = sqrt(2 max(e, 0))` -- the TKE the model is CARRYING, which sets `K = l q S`
    and the dissipation. Floored at zero only so a transported undershoot cannot make a
    NaN out of `sqrt`.
  * `MN.qke` -- the same thing with MYNN's own `[qkemin, 150]` clamp, which is what the
    length scale and the stability functions are integrated against. The clamp is
    COUNTED and never written back to the slot: the prognostic TKE is the model's, not
    the closure's.
"""
@noinline function _mynn_column_inputs!(MY::MYNNState, S, MN::MYNNColumnScratch,
                                        work::MYNNWork,
                                        geom::MCGeometry, uv, wv, vv, rho_e, n::Int64)
    c = MY.constants
    exner = MN.exner; q_col = MN.q; qke = MN.qke
    rho_t = S.rho_t; Tk = S.Tk
    n_clamp = 0
    @inbounds for i in 1:n
        ex = (S.p[i] / MYNN_P0)^c.rcp
        exner[i] = ex
        th = Tk[i] / ex
        sqv = S.rho_v[i] / rho_t[i]
        sqc = S.rho_c[i] / rho_t[i]
        sqi = 0.0                                  # ice is refused until S8
        work.s_u[i] = uv.f[i]
        work.s_v[i] = _louis_v(geom, vv, i)
        work.s_w[i] = wv.f[i]
        work.s_tk[i] = Tk[i]
        work.s_th[i] = th
        work.s_sqv[i] = sqv
        work.s_sqc[i] = sqc
        work.s_sqi[i] = sqi
        work.s_sqw[i] = sqv + sqc + sqi
        work.s_thl[i] = th - c.xlvcp/ex*sqc - c.xlscp/ex*sqi
        work.s_thetav[i] = th*(1.0 + c.p608*sqv)
        work.s_qv[i] = sqv/(1.0 - sqv)
        work.s_qc[i] = sqc/(1.0 - sqv)
        work.s_qi[i] = sqi/(1.0 - sqv)

        ee = rho_e[i] / rho_t[i]
        ee = ee > 0.0 ? ee : 0.0
        q_col[i] = sqrt(2.0 * ee)
        qk = 2.0 * ee
        if qk < MYNN_QKEMIN
            qk = MYNN_QKEMIN
            n_clamp += 1
        elseif qk > 150.0
            qk = 150.0
            n_clamp += 1
        end
        qke[i] = qk
    end
    return n_clamp
end

"""
    _mynn_taper_seed!(MY, MN, expdot, rho_e, rho_t, ust, colstart, n, re_i)

The cold start. On the FIRST call of a column whose `rho_e` is identically zero, lay down
the Fortran's Koracin-Berkowicz taper (`mynn_bl_driver` :1571-1575, the first-guess block
of `mynn_init_column!`) and deliver it as a ONE-STEP source `rho_t qke/2 / ts`.

It is a RATE because `expdot` is a rate, so what the multistep realizes on its startup
step is not exactly `rho_t qke/2`. That is acceptable on purpose: the taper only sets the
ORDER of the initial TKE, which the closure then equilibrates on the dissipation
timescale `tau_eps = B_1 l/(2q)` -- tens of seconds (the D4 estimate). What it must not do
is leave the column at exactly zero forever, because `K = l q S_M` is then identically
zero and the closure can never start. With `ust = 0` the taper is itself zero, so a
resting column stays quiet and `:mynn_init = :zero` is the same thing deliberately.
"""
@noinline function _mynn_taper_seed!(MY::MYNNState, MN::MYNNColumnScratch, expdot,
                                     rho_e, rho_t,
                                     ust::Float64, colstart::Int64, n::Int64,
                                     re_i::Int64)
    @inbounds for i in 1:n
        rho_e[i] == 0.0 || return nothing
    end
    zw = MY.zw
    qke = MN.qke; q_col = MN.q
    inv_ts = 1.0 / MY.ts
    @inbounds for i in 1:n
        qk = 5.0*ust*max((ust*700.0 - zw[i])/(max(ust, 0.01)*700.0), 0.01)
        qke[i] = max(qk, MYNN_QKEMIN)
        q_col[i] = sqrt(qk)
        expdot[colstart + i - 1, re_i] += 0.5 * rho_t[i] * qk * inv_ts
    end
    return nothing
end

"""
    _mynn_closure_update!(MY, S, MN, work, colstart, n, ci, SST, ust, F_sh, F_q, t)

One held-closure update: `GET_PBLH` -> `SCALE_AWARE` -> the surface block -> the Level-2
water variance's consumer `mym_condensation!` -> `mym_turbulence!`, in exactly the order
`mynn_column_step!` (src/mynn_closure.jl) replays the Fortran driver's per-column
sequence. `DMP_mf` is NOT called (`:mynn_edmf = 0`; the plumes arrive at S7), which is
the column the Fortran produces for `ktop_plume = 0`, and `mym_predict!` is NOT called
because the TKE equation is Scythe's own prognostic slot.

The carried state is loaded OUT of the tile first and written back after: `sm(kts)` and
`sh(kts)` are never written by `mym_turbulence!` (its interface loops start at `kts+1`),
`cldfra_bl` is read by the plume/cloud floor on the stability functions, and `qsq` sets
the cloud-PDF width -- and a per-thread scratch column is shared between columns, so
nothing in it can be assumed to still belong to this one.

Two Fortran shapes are reproduced on purpose (`:mynn_fidelity = :fortran`, D10):

  * `th_sfc = ts/exner(1)` where the driver's `ts` is ALREADY `T_sfc/exner(1)`
    (mynn_bl_driver :1063; harness README item 2). It divides by the surface Exner
    function TWICE, which inflates the surface potential temperature by ~3 % and through
    `fltv` inflates `rmol` and the mixing length. Reproduced because the parity anchor is
    the Fortran; `:mynn_fidelity = :scythe` is where it gets fixed, AFTER the deviation
    has been measured (D10), not before.
  * `rmol` is RECOMPUTED here from `fltv` and `ust` (:1079) rather than taken from
    `surface_exchange`'s Monin-Obukhov `inv_L`, so the closure's `zeta`, `pmz` and `phh`
    are self-consistent with the fluxes it was handed.

The six lines that set `qsq` are `mym_predict!`'s `closure <= 2.5` branch verbatim
(:3298-3300 for the surface copy, :3664-3673 for the diagnostic). `qsq` is the only one
of `tsq`/`qsq`/`cov` that `mym_condensation!` CASE 2 reads, and it is read on the NEXT
update, which is why it is stored rather than recomputed.
"""
@noinline function _mynn_closure_update!(MY::MYNNState, S, MN::MYNNColumnScratch,
                                         work::MYNNWork,
                                         colstart::Int64, n::Int64, ci::Int64,
                                         SST::Float64, ust::Float64,
                                         F_sh::Float64, F_q::Float64, t::Int64)
    c = MY.constants
    zn = work.z_n                      # the shared all-zero stand-in; NEVER written
    dz = MY.dz; zw = MY.zw
    exner = MN.exner; qke = MN.qke
    el = MN.el; sm = MN.sm; sh = MN.sh
    vt = MN.vt; vq = MN.vq; sgm = MN.sgm
    cfb = MN.cfb; qcb = MN.qcb; qib = MN.qib; qsq = MN.qsq

    @inbounds for i in 1:n
        j = colstart + i - 1
        el[i] = MY.el[j]; sm[i] = MY.sm[j]; sh[i] = MY.sh[j]
        vt[i] = MY.vt[j]; vq[i] = MY.vq[j]; sgm[i] = MY.sgm[j]
        cfb[i] = MY.cldfra_bl[j]; qcb[i] = MY.qc_bl[j]; qib[i] = MY.qi_bl[j]
        qsq[i] = MY.qsq[j]
    end

    zi, kzi = get_pblh!(1, n, work.s_thetav, qke, zw, dz, MYNN_XLAND_WATER)
    MY.pblh[ci] = zi
    MY.kpbl[ci] = kzi
    Psig_bl, Psig_shcu = MY.scale_aware ? scale_aware(MY.dx, zi) : (1.0, 1.0)

    # Surface fluxes, 1/L and the surface-layer stability functions
    # (mynn_bl_driver :1060-1097). `flqc = 0`: Scythe's surface layer has no liquid-water
    # flux off the sea surface (droplets do not evaporate off it).
    cpm = c.cp*(1.0 + 0.84*work.s_qv[1])
    flqv = F_q/S.rho_t[1]
    th_sfc = (SST/exner[1])/exner[1]         # the double division; see the docstring
    flq = flqv
    flt = F_sh/(S.rho_t[1]*cpm)
    fltv = flt + flqv*c.p608*th_sfc
    rmol = -c.karman*c.gtr*fltv/max(ust^3, 1.0e-6)
    zet = 0.5*dz[1]*rmol
    zet = max(zet, -20.0)
    zet = min(zet, 20.0)
    # `pmz = phim(zet) - zet` and `phh = phih(zet)` are the surface-layer stability
    # functions `mym_predict!` uses to build the log-layer TKE production `pdk1`. The
    # prognostic TKE slot takes that production as `P_sfc` (the drag sink itself,
    # delivered on the surface g(z)) instead, so they are formed here only when the
    # trace/diagnostic path asks for them -- the D10 `P_sfc from the drag sink vs pdk1`
    # fidelity switch is where they come back.
    if MY.check_values
        pmz = phim(zet) - zet
        phh = phih(zet)
        (isfinite(pmz) && isfinite(phh)) || error(
            "mc_mynn_bl!: the surface-layer stability functions are not finite " *
            "(zeta = $zet, rmol = $rmol, ust = $ust); the surface fluxes handed to the " *
            "closure are inconsistent")
    end
    MY.rmol[ci] = rmol
    MY.ust[ci] = ust

    mym_condensation!(1, n, MY.dx, dz, zw, MYNN_XLAND_WATER,
                      work.s_thl, work.s_sqw, work.s_sqv, work.s_sqc, work.s_sqi, zn,
                      S.p, exner, zn, qsq, zn, sh, el,
                      2, qcb, qib, cfb, zi, F_sh, vt, vq, work.s_th, sgm, rmol,
                      0, zn, c, work)

    # DMP_mf would run here; with :mynn_edmf = 0 every plume sum stays zero.

    mym_turbulence!(1, n, MYNN_XLAND_WATER, MY.closure, dz, MY.dx, zw,
                    work.s_u, work.s_v, work.s_thl, work.s_thetav, work.s_sqc,
                    work.s_sqw, qke, zn, zn, zn, vt, vq,
                    rmol, flt, fltv, flq, zi, work.s_th, sh, sm, el,
                    work.out_dfm, work.out_dfh, work.out_dfq,
                    work.out_tcd, work.out_qcd,
                    work.out_pdk, work.out_pdt, work.out_pdq, work.out_pdc,
                    work.out_qwt, work.out_qshear, work.out_qbuoy, work.out_qdiss,
                    0, Psig_bl, Psig_shcu, cfb, 2, zn, zn, zn, 0, zn, c, work)

    @inbounds begin
        work.out_pdq[1] = work.out_pdq[2]
        for k in 1:(n-1)
            b2l = work.qkw[k] <= 0.0 ? 0.0 :
                  MYNN_B2*0.25*(el[k+1] + el[k])/work.qkw[k]
            qsq[k] = b2l*(work.out_pdq[k+1] + work.out_pdq[k])
        end
        qsq[n] = qsq[n-1]
        # `mym_level2!`'s interface loop starts at kts+1, so `gm[kts]`/`gh[kts]` are never
        # written and still hold whatever the PREVIOUS column left in this thread's work
        # space. Zero them: the wall carries no interface. Nothing downstream can see the
        # difference today (`el[kts] = 0` makes `K[kts] = 0`, so both are multiplied by an
        # exact zero), which is precisely why a stale value here would be invisible until
        # something changed -- so it is set rather than reasoned about.
        work.gh[1] = 0.0
        work.gm[1] = 0.0
        for i in 1:n
            j = colstart + i - 1
            MY.el[j] = el[i]; MY.sm[j] = sm[i]; MY.sh[j] = sh[i]
            MY.vt[j] = vt[i]; MY.vq[j] = vq[i]; MY.sgm[j] = sgm[i]
            MY.cldfra_bl[j] = cfb[i]; MY.qc_bl[j] = qcb[i]; MY.qi_bl[j] = qib[i]
            MY.gh[j] = work.gh[i]
            MY.gm[j] = work.gm[i]
            MY.qsq[j] = qsq[i]
        end
    end
    MY.last_update_step[ci] = t
    return nothing
end

"""
    _mynn_diffusivities!(MY, MN, colstart, n, ci, n_clamp)

`K_m = l q S_M`, `K_h = l q S_H`, `K_e = Sqfac K_m` from the HELD closure and the CURRENT
TKE, plus the D12 census.

Colocated at the mish (D1): MYNN's interface averaging of `el`/`rho`/`dzk` is dropped and
`K = l q S` is formed pointwise, because every consumer downstream is a spline fit on the
mish rather than a staggered finite difference. `el[1] == 0` (the wall), so `K[1] == 0`
and no eddy flux is delivered through the bottom node -- the surface stress and fluxes
ride in on `g(z)` instead, for the reason in the header of src/mc_boundary_layer.jl.

The census reports BOTH explicit-diffusion numbers for the grid this run actually has:
`D_gal` on the vertical B-spline CELL width with the Galerkin factor 10 (the estimate the
offline harness reported and the S3 explicit-vs-implicit decision was taken on) and
`D_mish` on the smallest ACTUAL mish spacing, together with the TKE equation's own
stiffness `ts/tau_eps`. Every one is per column, written by the single thread that owns
that column, so the tile totals are exact under threading rather than racy.
`physical_params[:mynn_K_max]` is a COUNTED safety cap, never a tuning knob.
"""
@noinline function _mynn_diffusivities!(MY::MYNNState, MN::MYNNColumnScratch,
                                        colstart::Int64, n::Int64,
                                        ci::Int64, n_clamp::Int64)
    Km = MN.Km; Kh = MN.Kh; Ke = MN.Ke; q_col = MN.q
    K_max = MY.K_max
    n_cap = 0
    Kmax_m = 0.0; Kmax_h = 0.0; Kmax_e = 0.0; tstau = 0.0
    @inbounds for i in 1:n
        j = colstart + i - 1
        elj = MY.el[j]
        lq = elj * q_col[i]
        km = lq * MY.sm[j]
        kh = lq * MY.sh[j]
        km < 0.0 && (km = 0.0)
        kh < 0.0 && (kh = 0.0)
        if km > K_max; km = K_max; n_cap += 1; end
        if kh > K_max; kh = K_max; n_cap += 1; end
        ke = MYNN_SQFAC * km
        if ke > K_max; ke = K_max; n_cap += 1; end
        Km[i] = km; Kh[i] = kh; Ke[i] = ke
        MY.K_m[j] = km; MY.K_h[j] = kh
        km > Kmax_m && (Kmax_m = km)
        kh > Kmax_h && (Kmax_h = kh)
        ke > Kmax_e && (Kmax_e = ke)
        if elj > 0.0
            r = MY.ts * 2.0 * q_col[i] / (MYNN_B1 * elj)
            r > tstau && (tstau = r)
        end
    end
    D_gal = Kmax_e * MY.ts * 10.0 / (MY.dz_cell * MY.dz_cell)
    @inbounds begin
        MY.D_gal[ci] = D_gal
        MY.D_mish[ci] = Kmax_e * MY.ts / (MY.dz_min * MY.dz_min)
        MY.ts_tau[ci] = tstau
        MY.K_m_max[ci] = Kmax_m
        MY.K_h_max[ci] = Kmax_h
        MY.n_clamp_col[ci] += n_clamp
        MY.n_capK_col[ci] += n_cap
        D_gal > 0.5 && (MY.n_diffnum_col[ci] += 1)
    end
    return nothing
end

"""
    _mynn_flux_columns!(...) -> the boundary values of every fitted energy flux

The ~10 eddy-flux columns, each fitted on the slot-8 scratch basis and differentiated:
`uMish` <- the flux, `Btransform!`/`Atransform!` fit it, `Ixtransform` gives `d/dz` at the
mish (the tendency) and `SItransform!` puts the FITTED value back into `uMish` (which the
discrete shear production multiplies). Taking the value and the derivative from the SAME
spline coefficients is what makes the D3 identity exact rather than approximate.

Momentum (`rho_t K_m dz(u,w,v)`), heat in FLUX form (`rho_d T K_h dz(s_t')`, so the column
books telescope), the four SPECIES water legs (D7: total water `rho_w' = rho_t' - rho_d'`,
the prognostic vapour, cloud and rain -- Louis's total-minus-cloud inference is gone), the
water ENERGY carry `c_w K_h rho_w,z + c_v K_h rho_v,z` and the TKE transport.

Under a control-variable transform the condensate legs are built from the perturbation
DENSITY gradient, because an eddy flux of cloud or rain water is a MASS flux and not a
flux of `nu`; the slot increment then carries the Jacobian (in `_mynn_apply_column!`).
`Jc`/`Jr` are an exact 1.0 with no transform declared, so the default path is
bit-identical.

The TKE's fitted variable is the MASS-SPECIFIC `e = rho_e/rho_t` with flux
`rho_t K_e dz(e)` -- the standard eddy-diffusion form for an intensive quantity, and the
one whose boundary term is directly the turbulent energy flux through the ground and the
lid. Fitting `rho_e` itself and applying the product rule would close the budget equally
well; the choice is documented rather than silent because both were available.
"""
@noinline function _mynn_flux_columns!(mtile::ModelTile, MY::MYNNState, S,
                                       MN::MYNNColumnScratch,
                                       geom::MCGeometry, colstart::Int64, n::Int64, z,
                                       uv, wv, vv, rtv, rdv, rcv, rrv, rvv, rho_e,
                                       ctrans_on::Bool, rtrans_on::Bool)
    rho_t = S.rho_t; rho_d = S.rho_d; Tk = S.Tk
    Km = MN.Km; Kh = MN.Kh; Ke = MN.Ke
    col = scratch_column(mtile, 8)

    col.uMish .= rho_t .* Km .* uv.f_z
    Btransform!(col); Atransform!(col)
    Ixtransform(col, S.VD_u)
    SItransform!(col); copyto!(MN.Su, col.uMish)

    col.uMish .= rho_t .* Km .* wv.f_z
    Btransform!(col); Atransform!(col)
    Ixtransform(col, S.VD_w)
    SItransform!(col); copyto!(MN.Sw, col.uMish)

    _mynn_v_flux!(S.VD_v, MN.Sv, col, geom, rho_t, Km, vv)

    # The momentum ENERGY flux `Psi = u S_u + w S_w + v S_v`, fitted as ONE column.
    # This is the one place the discrete shear production departs from the plan's literal
    # `rho P_s := u_z S_u + ...`, and the reason is exactness. `u` and `S_u` are both
    # cubic per cell, so their PRODUCT is degree 6 while three-node Gauss is exact only to
    # degree 5: the sum `u dz(S_u) + u_z S_u` integrates exactly (each factor pair is
    # degree 5) but it telescopes to `u(z) S_u(z)` evaluated at the WALLS, and the model
    # does not expose `u`'s own spline coefficients per column, so that boundary value
    # cannot be formed without refitting `u` -- and the spline fit carries a filter
    # (`l_q`), so a refit is NOT the same function. Fitting the product instead and
    # defining
    #
    #     rho P_s := dz(Psi_hat) - (u dz(S_u) + w dz(S_w) + v dz(S_v))
    #
    # gives the SAME quantity to fit error (it is the product rule) and makes the column
    # identity exact by construction, because `Psi_hat` is a fitted cubic whose boundary
    # values `_mynn_edges` returns directly. The residual against the literal form is the
    # spline fit error of a degree-6 function and is reported, not assumed: see the
    # `Ps_disc` / `Ps_mynn` pair in `MYNNState`.
    @inbounds for i in 1:n
        col.uMish[i] = (uv.f[i]*MN.Su[i]) + (wv.f[i]*MN.Sw[i]) +
                       (_louis_v(geom, vv, i)*MN.Sv[i])
    end
    Btransform!(col); Atransform!(col)
    Ixtransform(col, MN.div_Pm)
    Pm0, Pm1 = _mynn_edges(col)

    s_t = S.s_t
    @. s_t = moist_entropy_total(Tk, rho_d, S.q_v, S.q_l, S.q_i)
    s_col = scratch_column(mtile, 6)
    s_col.uMish .= s_t .- mtile.mc_ref_diag.s_tbar
    Btransform!(s_col); Atransform!(s_col)
    s_z = S.bl_s_z
    Ixtransform(s_col, s_z)

    col.uMish .= rho_d .* Tk .* Kh .* s_z
    Btransform!(col); Atransform!(col)
    Ixtransform(col, S.QDOT_V)
    Sh0, Sh1 = _mynn_edges(col)

    col.uMish .= Kh .* (rtv.f_z .- rdv.f_z)
    Btransform!(col); Atransform!(col)
    Ixtransform(col, MN.div_w)

    col.uMish .= Kh .* rvv.f_z
    Btransform!(col); Atransform!(col)
    Ixtransform(col, MN.div_v)

    if ctrans_on
        col.uMish .= Kh .* S.bl_rho_cp_z
    else
        col.uMish .= Kh .* rcv.f_z
    end
    Btransform!(col); Atransform!(col)
    Ixtransform(col, MN.div_c)

    if rtrans_on
        # rho_r is a TOTAL with rho_rbar == 0, so the slot IS nu_r and the density
        # gradient is nu_r,z/J. `Jr` is strictly positive (dbhyp of a non-negative
        # density), so no guard is needed.
        wk = MN.wk
        @. wk = S.nu_r_z / S.Jr
        col.uMish .= Kh .* wk
    else
        col.uMish .= Kh .* rrv.f_z
    end
    Btransform!(col); Atransform!(col)
    Ixtransform(col, MN.div_r)

    SEw0 = 0.0; SEw1 = 0.0
    if MY.water_carry === :flux
        wk = MN.wk
        @inbounds for i in 1:n
            cw = (Cpv*Tk[i]) - S.Lv[i] + S.ke[i] + (gravity*z[i])
            cv = S.Lv[i] - (Rv*Tk[i])
            wk[i] = Kh[i]*((cw*(rtv.f_z[i] - rdv.f_z[i])) + (cv*rvv.f_z[i]))
        end
        col.uMish .= wk
        Btransform!(col); Atransform!(col)
        Ixtransform(col, MN.div_Ew)
        SEw0, SEw1 = _mynn_edges(col)
    else
        fill!(MN.div_Ew, 0.0)
    end

    e_col = MN.e
    @inbounds for i in 1:n
        e_col[i] = rho_e[i] / rho_t[i]
    end
    s_col.uMish .= e_col
    Btransform!(s_col); Atransform!(s_col)
    Ixtransform(s_col, MN.e_z)

    col.uMish .= rho_t .* Ke .* MN.e_z
    Btransform!(col); Atransform!(col)
    Ixtransform(col, MN.div_e)
    Se0, Se1 = _mynn_edges(col)

    return (; Pm0, Pm1, Sh0, Sh1, SEw0, SEw1, Se0, Se1)
end

"""
    _mynn_apply_column!(...)

Fold every leg onto the prognostic slots and accumulate the D3 boundary/surface energy
input and the two shear-production diagnostics.

Momentum `du = (dS_u - tau_u g)/rho_t` with `dE_t^mom = rho_t(u du + w dw + v dv)` -- the
FRIC_KE invariant: E_t follows the resolved KE down, and there is no dissipative heating,
because with an eddy K the resolved KE that is lost goes to unresolved scales. Here, for
the first time in this model, it goes somewhere the model CARRIES: `rho P_s` into
`rho_e`. `P_sfc` is the drag sink itself, delivered on the same `g(z)` as the stress so it
cancels the drag work POINTWISE and not merely in the column integral. `rho P_b` is the
closure's own buoyancy function (positive unstable, so a stable layer returns TKE to the
resolved flow) and `rho eps = rho_t q^3/(B_1 l)` is exactly what `bp*qke/2` removes in
`mym_predict`, with the Fortran's `diss_heat` cap and its `max(l,1)` dropped because
energy exactness needs ONE number and not a clipped one. Both appear with opposite signs
in E_t and in `rho_e`, so neither can leak.

The water energy split (D7): E_t receives the full divergence `dS_Ew` (which telescopes),
and the part of it that is NOT the fixed-T enthalpy of the transported mass --
`Qdot_w = dS_Ew - c_w dS_w - c_v dS_v`, which is `S_w dc_w/dz + S_v dc_v/dz` by the
product rule -- is genuine thermal heating and joins `Qdot_E` in driving T, and through it
p and Q_ss. With `:mynn_water_carry = :fixed_T` the local Louis map is restored and
`Qdot_w` is exactly zero. That is the D10/D7 fidelity comparison, and it is NOT
budget-neutral: `c_w rho_dot_w + c_v rho_dot_v` is not the divergence of anything, so the
column identity is open by exactly `<S_w dc_w/dz + S_v dc_v/dz>` and `bdry_E` is the exact
right-hand side only under `:flux`. test/test_mynn_bl.jl reports that residual rather than
asserting a closure the local map cannot have.

The boundary term `bdry_E` is the D3 identity's right-hand side, and every piece of it is
an EXACT spline boundary value of a fitted column -- the momentum products included, which
is the whole reason `Psi` is fitted (see `_mynn_flux_columns!`). Nothing in it is assumed
zero: the surface delivery profile `g(z)` exists precisely because the spline's
extrapolated wall value is NOT reliable as a flux, so the budget has to carry whatever
that value actually is and report it.
"""
@noinline function _mynn_apply_column!(MY::MYNNState, S, MN::MYNNColumnScratch,
                                       geom::MCGeometry,
                                       colstart::Int64, n::Int64, ci::Int64, z,
                                       uv, wv, vv, expdot, rv_i::Int64, re_i::Int64,
                                       tau_u::Float64, tau_v::Float64,
                                       F_sh::Float64, F_q::Float64, delta::Float64, ed)
    u = uv.f
    w = wv.f
    rho_t = S.rho_t; rho_d = S.rho_d; Tk = S.Tk
    R_m = S.R_m; C_vt = S.C_vt; Lv = S.Lv; Jc = S.Jc; Jr = S.Jr
    drvs_dT = S.drvs_dT; drvs_dp = S.drvs_dp; ke = S.ke
    Kh = MN.Kh; q_col = MN.q; wq = MY.w_mish
    div_Pm = MN.div_Pm
    VD_u = S.VD_u; VD_w = S.VD_w; VD_v = S.VD_v
    div_w = MN.div_w; div_v = MN.div_v; div_c = MN.div_c; div_r = MN.div_r
    div_e = MN.div_e; VD_Ew = MN.div_Ew; QDOT_V = S.QDOT_V
    flux_carry = MY.water_carry === :flux
    inv_delta = 1.0 / delta

    bdry = (ed.Pm1 - ed.Pm0) +
           (ed.Sh1 - ed.Sh0) + (ed.SEw1 - ed.SEw0) + (ed.Se1 - ed.Se0) + F_sh
    Ps_d = 0.0; Ps_m = 0.0

    @inbounds for i in 1:n
        j = colstart + i - 1
        gz = z[i] < delta ? 2.0 * inv_delta * (1.0 - (z[i] * inv_delta)) : 0.0
        du = (VD_u[i] - (tau_u * gz)) / rho_t[i]
        dw = VD_w[i] / rho_t[i]
        dv = _louis_dv(geom, VD_v, rho_t, tau_v, gz, i)
        expdot[j, 4] += du
        expdot[j, 5] += dw
        _louis_add_v!(geom, expdot, j, dv)

        vi = _louis_v(geom, vv, i)
        dE_mom = rho_t[i] * (((u[i]*du) + (w[i]*dw)) + (vi*dv))

        # The DISCRETE shear production (see `_mynn_flux_columns!` for why it is the
        # fitted momentum-energy divergence minus `V . dz(S_V)` rather than the literal
        # `u_z S_u`): the two differ by the spline fit error of a degree-6 product, and
        # only this form makes the column identity exact.
        Ps = div_Pm[i] - (((u[i]*VD_u[i]) + (w[i]*VD_w[i])) +
                          (vi*_mynn_vdiv(geom, VD_v, i)))
        Psfc = ((tau_u*u[i]) + (tau_v*vi)) * gz
        Pb = rho_t[i] * Kh[i] * MY.gh[j]
        elj = MY.el[j]
        qi = q_col[i]
        eps = elj > 0.0 ? rho_t[i]*qi*qi*qi/(MYNN_B1*elj) : 0.0
        Ps_d += wq[i] * Ps
        Ps_m += wq[i] * rho_t[i] * MN.Km[i] * MY.gm[j]

        qdot_h = QDOT_V[i] + (F_sh * gz)
        rw = div_w[i] + (F_q * gz)
        rv = div_v[i] + (F_q * gz)
        cw = (Cpv*Tk[i]) - Lv[i] + ke[i] + (gravity*z[i])
        cv = Lv[i] - (Rv*Tk[i])

        if flux_carry
            dE_w = VD_Ew[i] + ((cw + cv) * F_q * gz)
            qdot_w = VD_Ew[i] - (cw*div_w[i]) - (cv*div_v[i])
        else
            dE_w = (cw*rw) + (cv*rv)
            qdot_w = 0.0
        end

        qdot_E = (qdot_h - Pb) + eps          # into E_t as heat + the subgrid exchanges
        qdot_th = qdot_E + qdot_w             # what drives T, and through it p and Q_ss
        dT_v = qdot_th / (rho_d[i] * C_vt[i])
        dp_v = (R_m[i] / C_vt[i]) * qdot_th
        dp_q = Rv * Tk[i] * rv

        expdot[j, 1] += dp_v + dp_q
        expdot[j, 3] += rw
        expdot[j, 8] += Jr[i] * div_r[i]
        expdot[j, 9] += Jc[i] * div_c[i]
        expdot[j, rv_i] += rv
        expdot[j, 6] += (qdot_E + dE_w) + dE_mom
        expdot[j, 7] += rv - (drvs_dT[i]*dT_v) - (drvs_dp[i]*(dp_v + dp_q))
        expdot[j, re_i] += ((div_e[i] + Ps) + (Pb - eps)) + Psfc

        bdry += wq[i] * (cw + cv) * F_q * gz
    end
    @inbounds begin
        MY.bdry_E[ci] = bdry
        MY.Ps_disc[ci] = Ps_d
        MY.Ps_mynn[ci] = Ps_m
    end
    return nothing
end

"""
    mynn_census_line(MY::MYNNState) -> String

One line summarising the tile's boundary-layer census: the domain maxima of `K_m`,
`K_h`, the two explicit-diffusion numbers and `ts/tau_eps`, and the three counters --
TKE clamps and `K` caps in GRIDPOINT-steps, `n_diffnum` in COLUMN-steps (a column is
counted once per step in which its own `D_gal` exceeded 0.5). The counters are
reductions over the PER-COLUMN arrays, which is why they are exact under threading:
each column is written by the one thread that owns it, so nothing races and nothing is
lost.
"""
function mynn_census_line(MY::MYNNState)
    MY.active || return ""
    mx(v) = isempty(v) ? 0.0 : maximum(v)
    return "mynn census: max K_m=$(round(mx(MY.K_m_max); digits = 2)) " *
           "max K_h=$(round(mx(MY.K_h_max); digits = 2)) m^2/s; " *
           "max D_gal=$(round(mx(MY.D_gal); sigdigits = 3)) " *
           "max D_mish=$(round(mx(MY.D_mish); sigdigits = 3)) " *
           "max ts/tau_eps=$(round(mx(MY.ts_tau); sigdigits = 3)); " *
           "max pblh=$(round(mx(MY.pblh); digits = 1)) m; " *
           "mynn_n_clamp_e=$(sum(MY.n_clamp_col)) " *
           "mynn_n_cap_K=$(sum(MY.n_capK_col)) " *
           "mynn_n_diffnum=$(sum(MY.n_diffnum_col))"
end

"""
    mynn_write_final!(grid, model, mtile)

Print the census once at the end of a run, the `radiation_write_final!` pattern, into
the per-nest `scythe_out.log`. Also folds the per-column counters into the tile scalars
`n_clamp_e`/`n_cap_K`/`n_diffnum` so a caller that reads the state (a test, the
benchmark diagnostics) sees the same totals the line reports. The sidecar NetCDF is S9.
"""
function mynn_write_final!(mtile::ModelTile)
    MY = mtile.mynn
    MY.active || return nothing
    MY.n_clamp_e = sum(MY.n_clamp_col)
    MY.n_cap_K = sum(MY.n_capK_col)
    MY.n_diffnum = sum(MY.n_diffnum_col)
    MY.trace && println(mynn_census_line(MY))
    return nothing
end
