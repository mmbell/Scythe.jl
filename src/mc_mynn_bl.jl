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
#  4. THE MASS FLUX RIDES IN THE SAME COLUMNS (D6, S7, `:mynn_edmf = 1`). `DMP_mf` runs
#     on the closure cadence and its eight plumes are reduced to seven held sums per
#     gridpoint -- `Sigma_aw` and the plume-weighted `u, v, e, s_t, q_t, q_v`
#     (`_mynn_plume_sums!`). Every step the term `M_phi = rho_t Sigma_aw (phi_up - bar
#     phi)` is re-formed with the CURRENT environment and folded into the SAME flux column
#     as the eddy-diffusive part, before the fit. Because it is the same column, its
#     divergence, its boundary values and the momentum-energy product `Psi` all carry it,
#     so the plume's momentum work is inside `rho P_s` automatically and the D3 identity
#     above is unchanged -- nothing was added to it. The Fortran applies the environmental
#     half of the plume flux implicitly and floors the tridiagonal with `khdz` for
#     diagonal dominance; there is no counterpart to that floor in an explicit fit and it
#     is DROPPED (the plume Courant number `Sigma_aw ts/dz` is O(1e-3) -- `Sigma_aw` is
#     tenths of m/s, and it is counted in the census).
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

"""
    _mynn_add_mf!(colv, fac, s_aw, s_aw_phi, phi, colstart, n)

Fold the EDMF mass-flux transport of `phi` into a flux column that has already been filled
with its eddy-diffusive part, BEFORE the fit (D6):

    colv[i] -= fac[i] * (Sigma_aw_phi[j] - Sigma_aw[j] * phi[i])

`fac` is `rho_t` for `u`, `v`, `e` and the water legs (`rho_t * specific content` IS the
partial density, exactly as `rho_d * mixing ratio` is) and `rho_d*T` on the heat leg.

WHY IT IS A MINUS. The plan and the TeX write the plume flux as
`M_phi = rho_t Sigma_aw (phi_up - bar phi)`, a flux POSITIVE UPWARD. Every column fitted in
`_mynn_flux_columns!` is `S = +rho K dz(phi)` and the slot receives `+dz(S)`, so `S` is
MINUS the physical flux (`dphi/dt = -dz(F)`). The mass flux therefore enters the same
column with the opposite sign, and then the divergence, the boundary values `_mynn_edges`
returns and the momentum-energy product `Psi` all carry it automatically -- which is why
D3 is unchanged and the plume's momentum work needs no term of its own.

Subtraction, not addition, is also what keeps the off path bitwise. With the plumes absent
every sum is an exact `0.0`, the bracket is `0.0 - (0.0*phi) = +0.0` for any finite `phi`
(including a negative one, `0.0 - (-0.0) = +0.0`), `fac[i] > 0`, and `x - 0.0 === x` for
EVERY double -- `-0.0` included, where `x + 0.0` would flip the sign of the zero and, one
spline fit later, show up as a `-0.0`/`0.0` difference in an increment. The caller still
guards the call with `if mf`, so a run with `:mynn_edmf = 0` does not execute the loop at
all; this is the second lock, for the column that has the plumes ON and no plume firing.
"""
@inline function _mynn_add_mf!(colv, fac, s_aw::Vector{Float64},
                               s_aw_phi::Vector{Float64}, phi,
                               colstart::Int64, n::Int64)
    @inbounds for i in 1:n
        j = colstart + i - 1
        colv[i] -= fac[i] * (s_aw_phi[j] - (s_aw[j] * phi[i]))
    end
    return nothing
end

"""
    _mynn_moment_leg!(mtile, div, Kh, nu_z, J, slot, n)

One fitted eddy-diffusion leg of a TOTAL-form moment slot (S8): the twelve ISHMAEL ice
moments and the two-moment rain number.

    S = K_h ∂z(moment density) = K_h · (∂z ν / J)          fitted, then differentiated

`nu_z` is the slot's own fitted vertical gradient and `J = dν/dx` its Jacobian, both staged
by `_load_total_slot!` in the driver — so the gradient here is the gradient of the QUANTITY
and not of the control variable, exactly as the cloud and rain legs build theirs. With no
transform declared `J` is an exact `1.0` and `ν_z / 1.0 === ν_z` for every double, so the
untransformed path is bitwise the raw slot gradient. The caller multiplies the divergence by
`J` again on the way into the slot.

**Each moment is fitted on ITS OWN spline column**, `scratch_column(mtile, slot)`, not on the
shared slot-8 basis the liquid legs use. That is `_ice_flux!`'s rule and it is the same
argument: a mass slot with a Natural bottom must be free to let its crystals leave the domain
while a number or a volume moment fitted with different boundary conditions is not forced to
agree with it. It also means the twelve divergences do NOT sum to a refit of their sum; the
one place that matters is the ice energy carry, and see `_mynn_flux_columns!` for what is
done there.
"""
@noinline function _mynn_moment_leg!(mtile::ModelTile, div, Kh, nu_z, J,
                                     slot::Int64, n::Int64)
    c = scratch_column(mtile, slot)
    @inbounds for i in 1:n
        c.uMish[i] = Kh[i] * (nu_z[i] / J[i])
    end
    Btransform!(c)
    Atransform!(c)
    Ixtransform(c, div)
    return nothing
end

@inline function _mynn_v_flux!(VD_v, Sv, col, ::MCCartesianXZ, rho_t, Km, vv,
                               MY::MYNNState, colstart::Int64, n::Int64, mf::Bool)
    return nothing
end
@inline function _mynn_v_flux!(VD_v, Sv, col, ::MCWithV, rho_t, Km, vv,
                               MY::MYNNState, colstart::Int64, n::Int64, mf::Bool)
    v_z = vv.f_z
    col.uMish .= rho_t .* Km .* v_z
    if mf
        _mynn_add_mf!(col.uMish, rho_t, MY.s_aw, MY.s_aw_v, vv.f, colstart, n)
    end
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
stored on the tile. With `:mynn_edmf = 1` (S7) `DMP_mf` also runs, between the two, and
its plume ensemble is mapped onto the mish as the seven held sums `MYNNState.s_aw*`
(`_mynn_plume_sums!`); with `:mynn_edmf = 0` every one of them stays an exact zero, which
is the column the Fortran produces for `ktop_plume = 0`.

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

# Ice (S8, `options[:ice_microphysics] = :ishmael`)
The closure carries ONE ice mixing ratio and ISHMAEL carries three species with four
moments each, so the two halves are treated differently and on purpose. The CLOSURE is
handed the species sum (`q_i` into `theta_l`, `q_w` and the cloud PDF; `q_s` is an exact
zero, because the Fortran never mixes snow and there is no snow category here to mix). The
TRANSPORT is species-wise and moment-wise: each of the twelve moments gets its own fitted
`K_h dz(moment density)` leg on its own spline column, all four moments of a species
sharing one `K_h` so the mean particle does not move.

WHAT CLOSES AND WHAT DOES NOT. The total-water leg `S_w` is built from
`rho_w' = rho_t' - rho_d'`, which INCLUDES the ice, so `rho_dot_w` already carries every
species; the vapour and cloud stay their own legs and nothing is inferred as a difference
(that inference is exactly why the Louis scheme is still refused with ice). The species
sum `rho_dot_v + rho_dot_c + rho_dot_r + sum_k rho_dot_i,k` therefore equals `rho_dot_w`
only to the FIT ERROR of six independent spline columns, not identically. That residual
is the same one every other water source in this set leaves: `rho_t` stays the exactly
conserved anchor, the partition it implies is `res_rho_t = rho_t - rho_d - rho_liq -
rho_ice`, and `rho_v_reconcile` pulls the prognostic vapour onto it on `tau_rec`
(`_vapor_gap_census!` reports `max|delta|` as the drift diagnostic). Nothing here is
withheld or clipped to force a pointwise closure.

`:mynn_water_carry = :fixed_T` is REFUSED with ice (the `mc_driver!` preamble): the local
map has no ice term and `_diffusion_water_step!`, the routine it mirrors, has no ice
handling to copy.
"""
@noinline function mc_mynn_bl!(mtile::ModelTile, S, geom::MCGeometry,
                               colstart::Int64, colend::Int64, z,
                               uv, wv, vv, rtv, rdv, rcv, rrv, rvv, rev, expdot,
                               sfc::SurfaceLayerParams, t::Int64,
                               ctrans_on::Bool, rtrans_on::Bool)
    MY = mtile.mynn
    # The appended-slot gates, read from `MCSlots` exactly as `mc_driver!` reads them: the
    # indices were resolved by NAME once when the tile was built, so "the slots exist" and
    # "the legs run" cannot disagree, and this is a field load and not a Dict lookup in a
    # per-column path. No transform flag is needed anywhere below: `_load_total_slot!`
    # leaves `J` an exact `1.0` with no transform declared, and `nu_z / 1.0 === nu_z`.
    IS = mtile.mc_slots
    ice_on = ice_registered(IS)
    nr_on = MY.mix_numbers && IS.n_r > 0
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

    ed = _mynn_flux_columns!(mtile, MY, S, MN, work, geom, colstart, n, z, uv, wv, vv,
                             rtv, rdv, rcv, rrv, rvv, rho_e, ctrans_on, rtrans_on,
                             IS, ice_on, nr_on)

    _mynn_apply_column!(MY, S, MN, geom, colstart, n, ci, z, uv, wv, vv, expdot,
                        IS.rho_v, re_i, sx.tau_u, sx.tau_v, F_sh, F_q,
                        2.0 * z[2], ed, IS, ice_on, nr_on)
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
        # The closure's ONE ice mixing ratio is the SUM of the three ISHMAEL species
        # (S8). `rho_ice_t` is the column the thermodynamics reads -- floored per species
        # under `:condensate_floor` and capped at the anchor headroom -- so the closure's
        # `q_i` and the retrieval's `q_i = rho_ice_t/rho_d` are the same water, and a
        # spline undershoot in one ice mass cannot reach `theta_l` as negative latent
        # heat. With ice OFF the driver fills it with exact zeros, so `sqi` is `0.0` and
        # every line below is bitwise the pre-S8 code (`x - 0.0 === x`).
        sqi = S.rho_ice_t[i] / rho_t[i]
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
sequence -- with `dmp_mf!` between the condensation and the turbulence when
`:mynn_edmf = 1` (S7). `mym_predict!` is NOT called because the TKE equation is Scythe's
own prognostic slot.

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

    # ── DMP_mf, between the condensation and the turbulence (:1150-1180) ─────
    # With `:mynn_edmf = 0` this is not entered, the plume sums stay the exact zeros
    # `mc_mynn_state` allocated, and `mym_turbulence!` is handed the same all-zero
    # `edmf_w`/`edmf_a` stand-in it always was -- so the S5 path is bitwise.
    # With the plumes ON, `edmf_a1`/`edmf_w1` are what put the mass-flux term into the
    # mixing length (:2830-2840) and the `sm`/`sh` floors under an active plume or a
    # cloudy layer (:3050-3056, `mym_turbulence!` :1404-1407); those floors are the
    # Fortran's and they are kept.
    ed_w = zn; ed_a = zn
    if MY.edmf == 1
        ew = MY.ework[Threads.threadid()]
        _mynn_edmf_update!(MY, S, MN, work, ew, colstart, n, ci, zi, kzi, ust,
                           flt, fltv, flq, flqv, th_sfc, Psig_shcu)
        ed_w = ew.edmf_w
        ed_a = ew.edmf_a
    end

    mym_turbulence!(1, n, MYNN_XLAND_WATER, MY.closure, dz, MY.dx, zw,
                    work.s_u, work.s_v, work.s_thl, work.s_thetav, work.s_sqc,
                    work.s_sqw, qke, zn, zn, zn, vt, vq,
                    rmol, flt, fltv, flq, zi, work.s_th, sh, sm, el,
                    work.out_dfm, work.out_dfh, work.out_dfq,
                    work.out_tcd, work.out_qcd,
                    work.out_pdk, work.out_pdt, work.out_pdq, work.out_pdc,
                    work.out_qwt, work.out_qshear, work.out_qbuoy, work.out_qdiss,
                    0, Psig_bl, Psig_shcu, cfb, 2, ed_w, ed_a, zn, 0, zn, c, work)

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
    _mynn_edmf_update!(MY, S, MN, work, ew, colstart, n, ci, zi, kzi, ust, flt, fltv,
                       flq, flqv, th_sfc, Psig_shcu) -> EDMFGate

One `DMP_mf` call and the mapping of its plumes into Scythe's variables (D6, S7). Runs on
the closure cadence, between `mym_condensation!` and `mym_turbulence!` -- exactly where
`mynn_column_step_edmf!` (src/mynn_edmf.jl) and `mynn_bl_driver` (:1150-1180) call it.

The argument set is the ONE the parity harness validated (`momentum_opt = 1, tke_opt = 0,
scalar_opt = 1`, no chemistry, no aerosol, `spp_pbl = 0`), whatever `:mynn_edmf_mom` says:
that option decides whether the plume momentum reaches the FITTED COLUMNS, not what the
Fortran is asked to compute, so `dmp_mf!` is never run on a combination test/test_mynn_edmf.jl
does not cover. `dt` is dead inside `DMP_mf` (it appears only in the unreachable
`env_subs` block) and is passed as the model timestep for the record.

`vt`, `vq`, `cldfra_bl` and `qc_bl` go in as `mym_condensation!` left them and can come
back CHANGED: the Chaboureau-Bechtold shallow-cumulus block (:6630-6768) overwrites them
wherever a plume carries condensate. That is the Fortran's order and it is why the
write-back to the tile happens after `mym_turbulence!`, not before `dmp_mf!`.
"""
@noinline function _mynn_edmf_update!(MY::MYNNState, S, MN::MYNNColumnScratch,
                                      work::MYNNWork, ew::EDMFWork,
                                      colstart::Int64, n::Int64, ci::Int64,
                                      zi::Float64, kzi::Int, ust::Float64,
                                      flt::Float64, fltv::Float64, flq::Float64,
                                      flqv::Float64, th_sfc::Float64,
                                      Psig_shcu::Float64)
    zn = work.z_n
    gate = dmp_mf!(1, n, MY.ts, MY.zw, MY.dz, S.p, S.rho_t, 1, 0, 1,
                   work.s_u, work.s_v, work.s_w, work.s_th, work.s_thl, work.s_thetav,
                   work.s_tk, work.s_sqw, work.s_sqv, work.s_sqc, MN.qke,
                   MN.exner, MN.vt, MN.vq, MN.sgm,
                   ust, flt, fltv, flq, flqv, zi, kzi, MY.dx, MYNN_XLAND_WATER, th_sfc,
                   ew.edmf_a, ew.edmf_w, ew.edmf_qt, ew.edmf_thl, ew.edmf_ent, ew.edmf_qc,
                   ew.s_aw, ew.s_awthl, ew.s_awqt, ew.s_awqv, ew.s_awqc,
                   ew.s_awu, ew.s_awv, ew.s_awqke,
                   ew.sub_thl, ew.sub_sqv, ew.sub_u, ew.sub_v,
                   ew.det_thl, ew.det_sqv, ew.det_sqc, ew.det_u, ew.det_v,
                   MN.qcb, MN.cfb, zn, zn, true, true, Psig_shcu, 0, zn,
                   MY.constants, ew)
    _mynn_plume_sums!(MY, S, MN, ew, gate, colstart, n, ci)
    return gate
end

"""
    _mynn_plume_sums!(MY, S, MN, ew, gate, colstart, n, ci)

Map `DMP_mf`'s plume ensemble onto the mish and store the seven sums the every-step
tendency assembly needs (D6): `Sigma_aw` and the plume-weighted `u, v, e, s_t, q_t, q_v`.

# The interface -> mish mapping
`DMP_mf` carries the plumes on the box WALLS. Plume property index `k` lives at the wall
`zw[k+1]`, between mish points `k` and `k+1` (that is what `rhoz[k]`, `exneri[k]` and
`Pk = (p[k]dz[k+1] + p[k+1]dz[k])/(dz[k+1]+dz[k])` interpolate to, and it is why the
Fortran's own interface sums are written to `s_aw[k+1]`). Those walls are the MIDPOINTS
between mish points -- `mc_mynn_state` builds `zw` with `radiation_faces`, which is
bitwise `mynn_wall_heights(dz)` -- so the map back is the midpoint average: mish point `k`
is bounded by wall `k` (plume index `k-1`, and the GROUND at `k = 1`, where no plume
exists) and wall `k+1` (plume index `k`), and each contributes half.

# Why it is done per PLUME and not on the summed interface arrays
`Sigma_aw phi_up` is linear in the plume properties, so for `u, v, e, q_t, q_v` averaging
the eight-plume interface sums and averaging per plume are the same number. The ENTROPY is
not: `s_t` is a nonlinear function of the plume's own `T` and water, so it has to be
evaluated per plume. It is evaluated AT THE MISH, on the mass-flux-weighted mean of the
plume's `thl`, `q_t`, `q_c` over the two bounding walls, with the mish `exner` and the mish
`rho_d` -- NOT wall by wall with a wall-interpolated density. That choice is the whole
point: `s_t` carries `-R_d ln(rho_d)`, `rho_d` changes by ~1.5 % across a 150 m layer, and
`R_d * 0.015 = 4.3 J/(kg K)` would swamp the O(1) plume excess `s_t,up - bar s_t` that the
flux is made of. Evaluating the plume and its environment on the SAME `rho_d` makes the
excess a statement about the plume's temperature and composition, which is what a mass
flux transports, and nothing else.

The plume's own saturation adjustment is already in `upthl`/`upqc`: `condensation_edmf`
wrote them, so `T_up = exner*thl_up + xlvcp*qc_up` is the plume's temperature after its
condensation and `q_v,up = q_t,up - q_c,up` its vapour. `upqt`/`upqc` are SPECIFIC contents
(the `sqw` convention); `moist_entropy_total` wants MIXING RATIOS, and the conversion is
the one `_mynn_column_inputs!` applies to the environment, `q/(1 - q_v)`, so plume and
environment are converted identically.

# Nothing is recomputed
`upa`, `upw`, `upthl`, `upqt`, `upqc`, `upu`, `upv`, `upqke` are read straight out of
`EDMFWork`, where `dmp_mf!` left them, including the heat-flux limiter's column rescale
(:6483-6501 multiplies `UPA` itself, so `upa*upw` already carries `adjustment` exactly as
the Fortran's own `s_aw*` do). `Psig_w` comes back in the gate. `rhoz` is NOT used: the
Fortran's `s_aw*` are a MASS flux and the sums stored here are the kinematic
`Sigma_aw = sum a_i w_i` [m/s], so the every-step assembly can multiply by the CURRENT
`rho_t` at the mish rather than by the density the closure saw when the plumes were built.

`nup2 == 0` is the "gate passed but every plume stalled at the first wall" case
(:6288, harness README item 12) and it means NO flux, whatever `ktop` says -- so the sums
are zeroed on it exactly as `dmp_mf!` leaves its own `s_aw*` zeroed.
"""
@noinline function _mynn_plume_sums!(MY::MYNNState, S, MN::MYNNColumnScratch,
                                     ew::EDMFWork, gate::EDMFGate,
                                     colstart::Int64, n::Int64, ci::Int64)
    s_aw = MY.s_aw; s_st = MY.s_aw_st; s_qw = MY.s_aw_qw; s_qv = MY.s_aw_qv
    s_u = MY.s_aw_u; s_v = MY.s_aw_v; s_e = MY.s_aw_e
    fired = gate.nup2 > 0 && gate.ktop > 0
    if !fired
        @inbounds for i in 1:n
            j = colstart + i - 1
            s_aw[j] = 0.0; s_st[j] = 0.0; s_qw[j] = 0.0; s_qv[j] = 0.0
            s_u[j] = 0.0; s_v[j] = 0.0; s_e[j] = 0.0
        end
    end
    @inbounds begin
        gate.active && (MY.n_gate_col[ci] += 1)
        (gate.active && gate.nup2 == 0) && (MY.n_stall_col[ci] += 1)
    end
    # `plume_ktop`, `plume_ztop` and `aw_max` are RUNNING maxima over the run, and they
    # are advanced only on an update that actually produced a flux. Two reasons, both
    # learned from the first ocean-bubble arm, where 286 of 25605 gate-passing
    # column-updates made plumes and the census still printed `max Sigma_aw = 0`: a
    # last-update snapshot reports whatever the final call happened to see, which on a
    # scheme that fires intermittently is almost always nothing; and `ktop > 0` with
    # `nup2 == 0` is the STALL, not a plume, so recording it would make the active
    # fraction count columns that transported nothing.
    if !fired
        return nothing
    end
    @inbounds begin
        gate.ktop > MY.plume_ktop[ci] && (MY.plume_ktop[ci] = gate.ktop)
        gate.ztop > MY.plume_ztop[ci] && (MY.plume_ztop[ci] = gate.ztop)
    end

    c = MY.constants
    upa = ew.upa; upw = ew.upw; upthl = ew.upthl; upqt = ew.upqt; upqc = ew.upqc
    upu = ew.upu; upv = ew.upv; upqke = ew.upqke
    nup = ew.nup
    psig = gate.psig_w
    exner = MN.exner; rho_d = S.rho_d
    awmax = 0.0
    @inbounds for i in 1:n
        j = colstart + i - 1
        aw_t = 0.0; awu = 0.0; awv = 0.0; awe = 0.0
        awqw = 0.0; awqv = 0.0; awst = 0.0
        ex = exner[i]; rd = rho_d[i]
        for m in 1:nup
            # The two walls bounding mish point i: wall i (plume index i-1; the GROUND
            # when i == 1, where no plume lives) and wall i+1 (plume index i).
            a1 = upa[i,m]*upw[i,m]
            if i == 1
                den = a1
                den > 0.0 || continue
                inv = 1.0/den
                thl_up = upthl[i,m]; qt_up = upqt[i,m]; qc_up = upqc[i,m]
                u_up = upu[i,m]; v_up = upv[i,m]; qk_up = upqke[i,m]
            else
                a0 = upa[i-1,m]*upw[i-1,m]
                den = a0 + a1
                den > 0.0 || continue
                inv = 1.0/den
                thl_up = ((a0*upthl[i-1,m]) + (a1*upthl[i,m]))*inv
                qt_up  = ((a0*upqt[i-1,m])  + (a1*upqt[i,m]))*inv
                qc_up  = ((a0*upqc[i-1,m])  + (a1*upqc[i,m]))*inv
                u_up   = ((a0*upu[i-1,m])   + (a1*upu[i,m]))*inv
                v_up   = ((a0*upv[i-1,m])   + (a1*upv[i,m]))*inv
                qk_up  = ((a0*upqke[i-1,m]) + (a1*upqke[i,m]))*inv
            end
            aw = 0.5*den*psig
            # The plume's thermodynamics AT THIS MISH POINT (see the docstring): its
            # temperature after `condensation_edmf`, and its water as mixing ratios by
            # the same `q/(1 - q_v)` conversion the environment gets.
            qv_up = qt_up - qc_up
            dry = 1.0 - qv_up
            T_up = (ex*thl_up) + (c.xlvcp*qc_up)
            s_up = moist_entropy_total(T_up, rd, qv_up/dry, qc_up/dry, 0.0)
            aw_t += aw
            awu += aw*u_up
            awv += aw*v_up
            awe += aw*(0.5*qk_up)
            awqw += aw*qt_up
            awqv += aw*qv_up
            awst += aw*s_up
        end
        s_aw[j] = aw_t; s_st[j] = awst; s_qw[j] = awqw; s_qv[j] = awqv
        s_u[j] = awu; s_v[j] = awv; s_e[j] = awe
        aw_t > awmax && (awmax = aw_t)
    end
    @inbounds begin
        awmax > MY.aw_max[ci] && (MY.aw_max[ci] = awmax)
        MY.n_plume_col[ci] += 1
    end
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
two-moment rain NUMBER and the twelve ISHMAEL ICE moments (S8, each on its own spline
column through `_mynn_moment_leg!`), the water ENERGY carry
`c_w K_h rho_w,z + c_v K_h rho_v,z - L_f K_h rho_i,z` and the TKE transport.

With `:mynn_edmf = 1` the EDMF mass flux is SUBTRACTED from the same columns before they
are fitted (`_mynn_add_mf!` says why a minus): `u`, `v` (both only under
`:mynn_edmf_mom`), the heat leg with `rho_d T`, total water and vapour, the cloud leg as
`M_w - M_v`, the water energy carry with `c_w M_w + c_v M_v`, and the TKE with the plume's
`e = qke/2`. `w` and rain get none -- the Fortran carries neither a plume `w` flux nor
precipitation in the plumes. Because the term is inside the column rather than beside it,
`Ixtransform` and `_mynn_edges` deliver its divergence and its boundary value with no
further code, and `Psi = u S_u + w S_w + v S_v` picks the plume momentum work up on its
own.

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
                                       MN::MYNNColumnScratch, work::MYNNWork,
                                       geom::MCGeometry, colstart::Int64, n::Int64, z,
                                       uv, wv, vv, rtv, rdv, rcv, rrv, rvv, rho_e,
                                       ctrans_on::Bool, rtrans_on::Bool,
                                       IS::MCSlots, ice_on::Bool, nr_on::Bool)
    rho_t = S.rho_t; rho_d = S.rho_d; Tk = S.Tk
    Km = MN.Km; Kh = MN.Kh; Ke = MN.Ke
    col = scratch_column(mtile, 8)
    # The mass-flux legs (D6). `mf` is the whole scheme, `mf_mom` its momentum half
    # (`:mynn_edmf_mom = false` leaves the plumes carrying heat, water and TKE but no
    # momentum). There is NO plume flux of `w`: the Fortran carries no `s_aww` -- the
    # plume's own vertical velocity is internal to the closure, not a transported
    # property of the resolved flow.
    mf = MY.edmf == 1
    mf_mom = mf && MY.edmf_mom
    s_aw = MY.s_aw

    col.uMish .= rho_t .* Km .* uv.f_z
    if mf_mom
        _mynn_add_mf!(col.uMish, rho_t, s_aw, MY.s_aw_u, uv.f, colstart, n)
    end
    Btransform!(col); Atransform!(col)
    Ixtransform(col, S.VD_u)
    SItransform!(col); copyto!(MN.Su, col.uMish)

    col.uMish .= rho_t .* Km .* wv.f_z
    Btransform!(col); Atransform!(col)
    Ixtransform(col, S.VD_w)
    SItransform!(col); copyto!(MN.Sw, col.uMish)

    _mynn_v_flux!(S.VD_v, MN.Sv, col, geom, rho_t, Km, vv, MY, colstart, n, mf_mom)

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
    if mf
        # `M_h = rho_d T Sigma_aw (s_t,up - bar s_t)`, on the FULL entropies: the mass
        # flux transports a difference, so the reference `s_tbar` the diffusive leg is
        # built from cancels out of it exactly.
        @inbounds for i in 1:n
            j = colstart + i - 1
            col.uMish[i] -= (rho_d[i]*Tk[i]) *
                            (MY.s_aw_st[j] - (s_aw[j]*s_t[i]))
        end
    end
    Btransform!(col); Atransform!(col)
    Ixtransform(col, S.QDOT_V)
    Sh0, Sh1 = _mynn_edges(col)

    # The water legs. `work.s_sqw`/`work.s_sqv` are the SPECIFIC contents this step's
    # `_mynn_column_inputs!` built (`rho_v/rho_t`, `(rho_v+rho_c)/rho_t`) and the plume
    # sums are in the same units, so `rho_t * (q_up - bar q)` is the partial-density mass
    # flux the diffusive `K_h dz(rho')` beside it already is.
    col.uMish .= Kh .* (rtv.f_z .- rdv.f_z)
    if mf
        _mynn_add_mf!(col.uMish, rho_t, s_aw, MY.s_aw_qw, work.s_sqw, colstart, n)
    end
    Btransform!(col); Atransform!(col)
    Ixtransform(col, MN.div_w)

    col.uMish .= Kh .* rvv.f_z
    if mf
        _mynn_add_mf!(col.uMish, rho_t, s_aw, MY.s_aw_qv, work.s_sqv, colstart, n)
    end
    Btransform!(col); Atransform!(col)
    Ixtransform(col, MN.div_v)

    if ctrans_on
        col.uMish .= Kh .* S.bl_rho_cp_z
    else
        col.uMish .= Kh .* rcv.f_z
    end
    if mf
        # `M_c = M_w - M_v` exactly: with `q_i = 0` the plume's and the environment's
        # total water are both vapour + cloud, so the difference IS the cloud leg. Rain
        # gets none -- the plumes do not carry precipitation.
        @inbounds for i in 1:n
            j = colstart + i - 1
            col.uMish[i] -= rho_t[i] *
                ((MY.s_aw_qw[j] - (s_aw[j]*work.s_sqw[i])) -
                 (MY.s_aw_qv[j] - (s_aw[j]*work.s_sqv[i])))
        end
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

    # ── The two-moment rain NUMBER (S8, `:mynn_mix_numbers`) ─────────────────
    # Mixing the rain MASS and leaving `n_r` behind is a statement about the drop size:
    # the mean diameter of every column the operator touches becomes a function of the
    # diffusivity, and the fall speeds, the evaporation timescale and the self-collection
    # all read that size. So the number rides the SAME `K_h` on its own column and its own
    # boundary conditions, which is what `bl_mynn_mixscalars = 1` does for `qnc`/`qni` in
    # the Fortran. The plumes carry no precipitation, so there is no mass-flux term here.
    if nr_on
        _mynn_moment_leg!(mtile, MN.div_nr, Kh, S.nu_nr_z, S.Jnr, IS.n_r, n)
    end

    # ── The twelve ISHMAEL ICE moments (S8) ──────────────────────────────────
    # All twelve are per-volume moments advected with the product-rule continuity, so the
    # eddy flux of each is the same `K_h ∂z(moment density)` the cloud and rain masses get
    # — and all four moments of a species take the SAME `K_h`, which is what keeps the
    # crystals intact: mixing `ρ_i`, `n_i`, `a_i` and `c_i` with one diffusivity moves the
    # mean particle nowhere, while mixing the mass alone would rescale every crystal in the
    # column (the reason S4 refused this combination). There is no plume term: `DMP_mf`
    # carries no ice.
    if ice_on
        _mynn_moment_leg!(mtile, MN.div_i1q, Kh, S.nu_i1q_z, S.J_i1q, IS.i1_q, n)
        _mynn_moment_leg!(mtile, MN.div_i1n, Kh, S.nu_i1n_z, S.J_i1n, IS.i1_n, n)
        _mynn_moment_leg!(mtile, MN.div_i1a, Kh, S.nu_i1a_z, S.J_i1a, IS.i1_a, n)
        _mynn_moment_leg!(mtile, MN.div_i1c, Kh, S.nu_i1c_z, S.J_i1c, IS.i1_c, n)
        _mynn_moment_leg!(mtile, MN.div_i2q, Kh, S.nu_i2q_z, S.J_i2q, IS.i2_q, n)
        _mynn_moment_leg!(mtile, MN.div_i2n, Kh, S.nu_i2n_z, S.J_i2n, IS.i2_n, n)
        _mynn_moment_leg!(mtile, MN.div_i2a, Kh, S.nu_i2a_z, S.J_i2a, IS.i2_a, n)
        _mynn_moment_leg!(mtile, MN.div_i2c, Kh, S.nu_i2c_z, S.J_i2c, IS.i2_c, n)
        _mynn_moment_leg!(mtile, MN.div_i3q, Kh, S.nu_i3q_z, S.J_i3q, IS.i3_q, n)
        _mynn_moment_leg!(mtile, MN.div_i3n, Kh, S.nu_i3n_z, S.J_i3n, IS.i3_n, n)
        _mynn_moment_leg!(mtile, MN.div_i3a, Kh, S.nu_i3a_z, S.J_i3a, IS.i3_a, n)
        _mynn_moment_leg!(mtile, MN.div_i3c, Kh, S.nu_i3c_z, S.J_i3c, IS.i3_c, n)
    end

    SEw0 = 0.0; SEw1 = 0.0
    if MY.water_carry === :flux
        wk = MN.wk
        @inbounds for i in 1:n
            cw = (Cpv*Tk[i]) - S.Lv[i] + S.ke[i] + (gravity*z[i])
            cv = S.Lv[i] - (Rv*Tk[i])
            wk[i] = Kh[i]*((cw*(rtv.f_z[i] - rdv.f_z[i])) + (cv*rvv.f_z[i]))
        end
        if ice_on
            # THE ICE ENERGY CARRY. `rho_w' = rho_t' - rho_d'` already INCLUDES the ice, so
            # the total-water leg has given every kilogram of ice the LIQUID coefficient
            # `c_w`; what is owed is the difference between the ice and the liquid specific
            # energies. The model's own ice convention is `E_sed_i` (the sedimentation
            # energy flux, src/moist_compressible.jl):
            #
            #     c_i = C_pv T - L_s(T) + ke + g z          (liquid: c_w, with L_v)
            #
            # so `c_i - c_w = L_v - L_s = -L_f`, and the ice rides the same column as a
            # SUBTRACTED `L_f K_h dz(rho_i)`. That is the exact structural mirror of the
            # vapour's `+c_v`: one flux column, three coefficients, and `S_Ew` still
            # telescopes to a boundary value, so the D3 identity is unchanged.
            #
            # The GRADIENT is summed over the three species before the multiply (one term
            # in a column that is fitted once); the DIVERGENCE that `Qdot_w` subtracts in
            # `_mynn_apply_column!` is summed from the three separately fitted legs, the
            # same choice `Fi_z` makes for the sedimentation. They differ by the fit error
            # of a sum against the sum of fits, which is a `Qdot_w` (thermal) effect only
            # — `E_t` receives the divergence of THIS column and nothing else.
            @inbounds for i in 1:n
                cf = L_s(Tk[i]) - S.Lv[i]                             # L_f > 0
                di = ((S.nu_i1q_z[i]/S.J_i1q[i]) + (S.nu_i2q_z[i]/S.J_i2q[i])) +
                     (S.nu_i3q_z[i]/S.J_i3q[i])
                wk[i] -= cf * (Kh[i] * di)
            end
        end
        if mf
            # D7 with the plumes: `S_Ew = Fit[c_w(K_h rho_w,z + M_w) + c_v(K_h rho_v,z +
            # M_v)]`. The same two mass-flux terms the water legs took, weighted by the
            # same energy factors -- so `Qdot_w = dS_Ew - c_w dS_w - c_v dS_v` stays the
            # product-rule remainder and the D3 column identity is untouched.
            @inbounds for i in 1:n
                j = colstart + i - 1
                cw = (Cpv*Tk[i]) - S.Lv[i] + S.ke[i] + (gravity*z[i])
                cv = S.Lv[i] - (Rv*Tk[i])
                tw = rho_t[i]*(MY.s_aw_qw[j] - (s_aw[j]*work.s_sqw[i]))
                tv = rho_t[i]*(MY.s_aw_qv[j] - (s_aw[j]*work.s_sqv[i]))
                wk[i] -= (cw*tw) + (cv*tv)
            end
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
    if mf
        _mynn_add_mf!(col.uMish, rho_t, s_aw, MY.s_aw_e, e_col, colstart, n)
    end
    Btransform!(col); Atransform!(col)
    Ixtransform(col, MN.div_e)
    Se0, Se1 = _mynn_edges(col)

    return (; Pm0, Pm1, Sh0, Sh1, SEw0, SEw1, Se0, Se1)
end

"""
    _mynn_moment_apply!(expdot, div, J, slot, colstart, n)

Fold one fitted moment divergence into its prognostic slot (S8): `expdot[j, slot] += J·div`.

The Jacobian is the SLOT's, not the density's: the leg was built from `∂z ν / J` so that it
is a flux of the quantity, and the increment goes back the other way through `dν/dx`. `J` is
an exact `1.0` with no transform declared, and `1.0 * x === x`, so the untransformed slot
receives the raw divergence.
"""
@inline function _mynn_moment_apply!(expdot, div, J, slot::Int64,
                                     colstart::Int64, n::Int64)
    @inbounds for i in 1:n
        expdot[colstart + i - 1, slot] += J[i] * div[i]
    end
    return nothing
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
                                       F_sh::Float64, F_q::Float64, delta::Float64, ed,
                                       IS::MCSlots, ice_on::Bool, nr_on::Bool)
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

        # Per-gridpoint budget columns (S9), held only for the sidecar
        # (src/mynn_io.jl) -- exactly the terms above, nothing recomputed.
        MY.g_Ps[j] = Ps
        MY.g_Ps_mynn[j] = rho_t[i] * MN.Km[i] * MY.gm[j]
        MY.g_Pb[j] = Pb
        MY.g_eps[j] = eps
        MY.g_tke_transport[j] = div_e[i]

        qdot_h = QDOT_V[i] + (F_sh * gz)
        rw = div_w[i] + (F_q * gz)
        rv = div_v[i] + (F_q * gz)
        cw = (Cpv*Tk[i]) - Lv[i] + ke[i] + (gravity*z[i])
        cv = Lv[i] - (Rv*Tk[i])

        if flux_carry
            dE_w = VD_Ew[i] + ((cw + cv) * F_q * gz)
            # The ice share of the flux-form carry (S8), the mirror of the `S_Ew` column's
            # own ice term: the total-water leg gave the ice `c_w`, and `c_i - c_w = -L_f`.
            # Written INSIDE the `c_w` bracket rather than as a term of its own so that a
            # zero ice divergence is `(cw*div_w) - 0.0`, which is `=== cw*div_w` for every
            # double — `x + 0.0` would flip a `-0.0` and, one spline fit later, show up as
            # a sign-of-zero difference in a warm increment.
            qw_i = ice_on ? (L_s(Tk[i]) - Lv[i]) *
                            (((MN.div_i1q[i] + MN.div_i2q[i]) + MN.div_i3q[i])) : 0.0
            qdot_w = VD_Ew[i] - ((cw*div_w[i]) - qw_i) - (cv*div_v[i])
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
    # ── The appended moment slots (S8) ───────────────────────────────────────
    # Outside the loop above, deliberately: the warm slots' arithmetic is then untouched
    # by the ice being on, so an ice-on column whose ice is exactly zero is bitwise the
    # ice-off column in every warm increment (test_mynn_ice.jl pins that with `===`).
    # Nothing here reaches E_t: a NUMBER and a VOLUME MOMENT carry no energy, and the ice
    # MASS carried its own `-L_f K_h dz(rho_i)` inside the `S_Ew` column already.
    if nr_on
        _mynn_moment_apply!(expdot, MN.div_nr, S.Jnr, IS.n_r, colstart, n)
    end
    if ice_on
        _mynn_moment_apply!(expdot, MN.div_i1q, S.J_i1q, IS.i1_q, colstart, n)
        _mynn_moment_apply!(expdot, MN.div_i1n, S.J_i1n, IS.i1_n, colstart, n)
        _mynn_moment_apply!(expdot, MN.div_i1a, S.J_i1a, IS.i1_a, colstart, n)
        _mynn_moment_apply!(expdot, MN.div_i1c, S.J_i1c, IS.i1_c, colstart, n)
        _mynn_moment_apply!(expdot, MN.div_i2q, S.J_i2q, IS.i2_q, colstart, n)
        _mynn_moment_apply!(expdot, MN.div_i2n, S.J_i2n, IS.i2_n, colstart, n)
        _mynn_moment_apply!(expdot, MN.div_i2a, S.J_i2a, IS.i2_a, colstart, n)
        _mynn_moment_apply!(expdot, MN.div_i2c, S.J_i2c, IS.i2_c, colstart, n)
        _mynn_moment_apply!(expdot, MN.div_i3q, S.J_i3q, IS.i3_q, colstart, n)
        _mynn_moment_apply!(expdot, MN.div_i3n, S.J_i3n, IS.i3_n, colstart, n)
        _mynn_moment_apply!(expdot, MN.div_i3a, S.J_i3a, IS.i3_a, colstart, n)
        _mynn_moment_apply!(expdot, MN.div_i3c, S.J_i3c, IS.i3_c, colstart, n)
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
           "mynn_n_diffnum=$(sum(MY.n_diffnum_col))" *
           mynn_plume_census(MY)
end

"""
    mynn_plume_census(MY::MYNNState) -> String

The EDMF half of the census (S7), appended to the same line so one parse gets both. Empty
with `:mynn_edmf = 0`.

`mynn_plume_active_frac` is the fraction of COLUMNS that produced a mass flux at least
once in the run (`ktop > 0` AND `nup2 > 0` -- the stall is not a plume);
`mynn_max_mass_flux` is the largest `Sigma_aw = sum a_i w_i` [m/s] and `mynn_max_ztop_m`
the highest plume top [m] reached anywhere, at any time. All three are running maxima over
the run rather than a snapshot of its last step, because a scheme that fires
intermittently is almost never firing at the moment the run happens to end. The three counters are COLUMN-UPDATES and
they keep the gate and the stall apart, which is the distinction
tools/mynn_fortran_driver/README.md item 12 exists to make: `mynn_n_gate` passed
`fltv2 > 0.002 && maxwidth > minwidth && superadiabatic`; `mynn_n_stall` passed it and
still produced nothing because every one of the eight plumes failed to leave the first
wall (`nup2 = 0`, :6288); `mynn_n_plume` produced a flux. A run where `n_gate` is large
and `n_plume` is zero is not a broken scheme -- it is a grid whose first layer is too
thick for the plumes to accelerate out of, and it is reported as such.
"""
function mynn_plume_census(MY::MYNNState)
    MY.edmf == 1 || return ""
    mx(v) = isempty(v) ? 0.0 : maximum(v)
    nact = 0
    @inbounds for ci in eachindex(MY.plume_ktop)
        MY.plume_ktop[ci] > 0 && (nact += 1)
    end
    frac = isempty(MY.plume_ktop) ? 0.0 : nact/length(MY.plume_ktop)
    return "; mynn_plume_active_frac=$(round(frac; sigdigits = 4)) " *
           "mynn_max_mass_flux=$(round(mx(MY.aw_max); sigdigits = 4)) " *
           "mynn_max_ztop_m=$(round(mx(MY.plume_ztop); digits = 1)) " *
           "mynn_n_gate=$(sum(MY.n_gate_col)) " *
           "mynn_n_stall=$(sum(MY.n_stall_col)) " *
           "mynn_n_plume=$(sum(MY.n_plume_col))"
end

# `mynn_write_final!` (the run-end hook: folds the per-column counters, prints the
# census line, and -- S9 -- writes the final sidecar snapshot) lives in src/mynn_io.jl,
# included after this file, beside `mynn_write!` and the rest of the sidecar I/O.
