# Total-energy moist compressible equation set.
#
# Prognostic variables (XZ slice): p [Pa], rho_d, rho_t, u, w, E_t [J/m^3], Q_ss
# [kg/m^3], rho_r, rho_c. The CONDENSATE is prognostic and the VAPOR is the
# residual rho_v = rho_t - rho_d - rho_c - rho_r; temperature then follows in
# CLOSED FORM from the Bryan & Fritsch (2002) total energy (see
# retrieve_temperature), with no iteration and no dependence on the partition.
#
# It used to be the other way round — rho_c the residual of four nearly-cancelling
# fitted fields — and that was the defect diagnosed in
# reference/HANDOFF_DIAGNOSED_CLOUD.md: in cloud-free air the true Q_ss sits
# exactly ON its admissible ceiling, so fit-level error (~2e-6 kg/m^3) put rho_c
# on the wrong side of zero, the closure evaporated cloud that was not there, and
# because the residual was REGENERATED every step it was a sustained pressure sink
# (3.19 Pa/s at t = 0 on the balanced TC vortex — all of the measured tendency).
# Making the smallest field prognostic and the largest one the residual puts the
# same fit error where it is harmless: 1e-6 against a vapor density of ~1e-2 is a
# 5e-5 relative error in a field that no longer feeds back into temperature at all.
#
# Q_ss is therefore no longer thermodynamic — it is purely the microphysics'
# supersaturation driver (smooth and advected, rather than a noisy difference of
# two large numbers), reconciled toward the diagnosed rho_v - rho_vs(T,p) on
# tau_qss by `qss_relaxation`.
#
# ── The vapor has TWO discrete representations, and they fail in disjoint regimes
#
# Because Q_ss is prognostic and the temperature retrieval never reads it, the
# vapor can equally be retrieved as the SUPERSATURATION residual
#
#     res_qss   = Q_ss + rho_vs(T, p)     alongside
#     res_rho_t = rho_t - rho_d - rho_c - rho_r    (the density-budget residual above)
#
# with no new prognostic and no change to the SI slaving. Neither is uniformly
# better, and the failure is CONDITIONING, measured both ways on saved snapshots
# (reference/HANDOFF_VAPOR_RETRIEVAL.md; both sweeps recorded in the header of
# benchmarks/vapor_blend_diagnostic.jl):
#
#   * IN CLOUD res_rho_t cancels catastrophically — rho_c/rho_w reaches 1.0011 at
#     the worst point, so the vapor is the small difference of two nearly equal
#     large numbers, and the NOPRECIP storm's in-cloud negatives are 1184 of 1186;
#   * IN DRY AIR res_qss cancels catastrophically — rho_v -> 0 forces Q_ss -> -rho_vs,
#     which is the failure Scythe_moist_compressible.tex anticipates, and a straight
#     swap is a NET REGRESSION on the shipped O01 default (1.11 % -> 2.61 % negative
#     points) because the dry population dominates there.
#
# So the retrieval is a C¹ BLEND of the two, gated on `options[:vapor_retrieval]`
# (`:blend`, the shipped default since 2026-07-29, or `:residual`, the pre-blend
# density-budget route, which is retained as the bitwise A/B lever):
#
#     s = |Q_ss| / rho_vs
#     w = w_cloud(rho_liq; l0, l1) * w_trust(s; t0, t1)
#     rho_v = res_rho_t + f(w*(res_qss - res_rho_t); dcap*rho_vs)
#
# `w_cloud` smoothsteps UP in rho_liq and does the REGIME SELECTION; `w_trust`
# smoothsteps DOWN in s and does nothing but reject a Q_ss that has GROSSLY
# detached from the density budget (s ~ 1e10 and up). See `vapor_retrieval_blend`
# for why the selector is cloudiness and not s (an s-only selector was measured and
# refuted: it degenerates into a soft clamp on negative vapor), and why the trust
# thresholds sit well above s = 1.
#
# `f` is the C¹ saturator `_blend_saturate`: the identity for |c| <= dcap*rho_vs/2,
# bounded by dcap*rho_vs, odd and monotone. The cap is LOAD-BEARING, not a safety
# belt — the uncapped blend detonates the NOPRECIP storm, because the partition gap
# is unbounded RELATIVE to rho_vs (which collapses in the rising anvil) while
# w_trust, a function of the supersaturation magnitude, is blind to it.
#
# THREE INVARIANTS SURVIVE THE BLEND, and they are what make it a partition rather
# than a source:
#
#   1. rho_t is still the conserved mass and is untouched. The blended rho_v is a
#      PARTITION read by the thermodynamic consumers (q_v -> C_vt, R_m, C_pt,
#      gamma_m; Q_s_energy; the condensation closure's vapor bound; the surface
#      moisture flux). `water_mass_drift_pct` is unchanged by construction.
#   2. rho_d + rho_v + rho_c + rho_r no longer equals rho_t pointwise. That gap is
#      deliberate: it is `w*(res_qss - res_rho_t) = -w*tau_qss*QSSREL` where the
#      saturator is inert, and it is BOUNDED by `dcap*rho_vs` everywhere. (The
#      `-tau_qss*QSSREL` relation is an identity, not a bound — see
#      `qss_relaxation`; the bound is `dcap`.)
#   3. `qss_relaxation` is ALWAYS fed `res_rho_t`, NEVER the blended vapor. Fed the
#      blend it self-annihilates wherever w = 1 and the single mechanism tying Q_ss
#      to the density budget vanishes precisely where it is load-bearing.
#
# Negative water is not representable: `clamp_water!` floors rho_c and rho_r after
# every column step. Because rho_t is prognostic and rho_v is the residual, that
# floor is exactly a phase change — total water and E_t are untouched and the
# retrieval supplies the matching latent heat — so it conserves mass, water and
# energy by construction.
#
# The conserved quantities (rho_d, rho_t, E_t) are extensive flux-form prognostics,
# so the Galerkin low-pass filter preserves their integrals (no Jensen drift). The
# energy equation is the exact first law: phase change carries no energy source
# (the R_v*T*ln(H) term of the entropy identity is entropy production, which
# cancels exactly against the production hidden in T*ds_t; see
# reference/Scythe_moist_compressible.tex).
#
# All new functions for this equation set live in this file: thermodynamic
# helpers, condensation closure, the equation set RHS, its semi-implicit
# adjustment, initializers, and the reference-state writer.

using Springsteel.Thermodynamics: rho_v_sat, internal_energy_bf02

# ── Per-thread scratch for the equation-set RHS ────────────────────────────────

"""
The live broadcast temporaries of [`moist_compressible_XZ`](@ref). Each is a `kDim` column,
recomputed every column of every timestep — as fresh allocations they were ~150 of the
function's 163 per-call allocations.

Keyed by NAME, not by index. An index-numbered scratch pool (`view(pool, :, 7, tid)`) makes it
easy to hand the same buffer to two temporaries that are live at once, and the resulting
corruption is silent and hard to see. A `NamedTuple` cannot hold a duplicate field, so that
class of bug cannot be written here at all.
"""
const MC_SCRATCH_SLOTS = (
    # ── moist_compressible_XZ ──
    :p, :rho_d, :rho_t, :E_t, :Q_ss, :rho_c, :rho_r,                  # totals
    :p_z, :rho_d_z, :rho_t_z, :E_t_z, :Q_ss_z, :rho_c_z,              # total vertical gradients
    :ke, :geo, :M, :Tk, :p_hPa, :rho_vs, :rho_v, :res_rho_t, :rho_liq, :q_v, :q_l, # diagnostic state
    :rho_liq_t,   # what the THERMODYNAMICS reads; == rho_liq unless condensate_floor_mode
    :nu_c, :nu_c_z, :Jc,   # slot-9 control variable, its gradient, dnu/drho (see bhyp/ahyp)
    :nu_r, :nu_r_z, :Jr, # slot-8 control variable, its gradient, dnu/drho (rain_transform_mode)
    :C_vt, :R_m, :C_pt, :gamma_m, :Lv, :drvs_dT, :drvs_dp,            # mixture thermo
    :Q_s, :Qdot, :Qdot_r, :div,                                       # condensation, divergence
    :cap_c, :cap_r, :cap_v,                        # AB3 depletion bounds (raw, see _ab3_sink_bound)
    :AUTO_COLL, :Vt, :Fr, :Fr_z, :E_sed, :E_sed_z,                    # warm-rain microphysics
    :sd_xx, :QDOT_TH, :FRIC_KE,                                       # horizontal diffusion
    :ADV, :FORCING, :KDIFF,                                           # per-slot accumulators
    :dT_nc, :dp_nc, :SATF, :QSSREL,                                   # Q_ss chain rule
    :s_t, :stage_zz,                                                             # moist entropy (vertical heat)
    :imp_phi_z, :imp_c_d, :imp_c_d_z, :imp_c_e, :imp_c_e_z,           # acoustic AI2* history staging
    :sd_pxi, :sd_alpha,                    # state-dependent acoustic linearization
    # ── Louis boundary layer + Smagorinsky closure (mc_boundary_layer.jl) ──
    :Kv, :K_smag, :VD_u, :VD_v, :VD_w, :QDOT_V, :VDOT_w, :VDOT_v,
    # `bl_rho_cp_z` is the Louis BL's perturbation CLOUD-DENSITY gradient ∂z ρ_c', staged by
    # `mc_driver!` only under a condensate transform (it is `rcv.f_z` otherwise). It reuses the
    # column that was the retired diagnosed-ρ_v gradient.
    :bl_s_z, :bl_rho_cp_z,
    # ── semiimplicit_adjustment_p (si_ prefix) ──
    # Deliberately NOT sharing the names above. The two functions' temporaries are not live at
    # the same time today, so sharing would work — but it would be an invisible coupling, and
    # the first person to read an mc_XZ value after the adjustment call would get silent
    # corruption. Distinct names cost ~200 KB and make that unwriteable.
    :si_p_nstar, :si_w_nstar, :si_rhod_nstar, :si_rhot_nstar, :si_et_nstar,
    :si_p_nstar_z, :si_phi_z, :si_rhs, :si_c_d, :si_c_d_z, :si_c_e, :si_c_e_z,
    # ── diffusion_timestep_mc (df_ prefix) ──
    :df_u_star, :df_w_star, :df_p_star, :df_rho_d_star, :df_rho_t_star,
    :df_E_t_star, :df_ke_star, :df_M_star,
    :df_T_star, :df_p_hPa_star, :df_drvs_dT, :df_drvs_dp,
    :df_rho_vs_star, :df_rho_v_star, :df_q_v_star, :df_q_l_star,
    :df_C_vt_star, :df_R_m_star, :df_Lv_star, :df_stp_star,
    :df_u_nstar, :df_w_nstar, :df_s_nstar, :df_u_np1, :df_w_np1,
    :df_dke, :df_dE_visc, :df_ds_t, :df_dT_h, :df_dE_h, :df_dp_h, :df_dQ_h,
    :df_rw_star, :df_rv_star, :df_rw_nstar, :df_rv_nstar, :df_rr_nstar,
    :df_rw_np1, :df_rv_np1, :df_drw, :df_drv, :df_drr, :df_dE_w,
    :df_rc_star, :df_rc_nstar, :df_rc_np1, :df_drc, :df_rho_c_star, :df_rho_r_star,
    :df_rho_liq_star,
    # ── tangential wind v (cylindrical geometries; inert columns on the XZ slice) ──
    :df_v_star, :df_v_nstar, :df_v_np1)

"""
    _allocate_mc_scratch(tile, model)

One `NamedTuple` of `kDim` work vectors per thread for the total-energy set; an empty vector
for every other equation set. Indexed by `threadid()`, which is a valid owner tag because the
column loop is `Threads.@threads :static` (same rule as `scratch_columns`).
"""
function _allocate_mc_scratch(tile::AbstractGrid, model::ModelParameters)

    uses_pressure_reference(model.equation_set) || return Vector{Nothing}(undef, 0)
    kDim = model.grid_params.kDim
    return [NamedTuple{MC_SCRATCH_SLOTS}(
                ntuple(_ -> zeros(Float64, kDim), length(MC_SCRATCH_SLOTS)))
            for _ in 1:Threads.maxthreadid()]
end

# ── Thermodynamic helpers ──────────────────────────────────────────────────────

"""
    drho_vsat_dT(Tk, p_hPa)

Temperature partial of the saturation vapor density ρ_v* = e*/(R_v T) [kg/m³/K]
at constant pressure, using the Buck (1981) saturation vapor pressure.
"""
function drho_vsat_dT(Tk, p_hPa)

    return (100.0 * sat_pressure_liquid_buck_dT(Tk, p_hPa) / (Rv * Tk)) -
           (rho_v_sat(Tk, p_hPa) / Tk)
end

"""
    drho_vsat_dp(Tk, p_hPa)

Pressure partial of the saturation vapor density ρ_v* [kg/m³ per Pa] at constant
temperature. Only the Buck (1981) pressure-enhancement factor depends on total
pressure, so this is small and positive.
"""
function drho_vsat_dp(Tk, p_hPa)

    Tc = Tk - 273.15
    B = 3.20e-6
    C = 5.9e-10
    a = 6.1121
    b = 18.729
    c = 257.87
    d = 227.3
    ew4 = a * exp((b - (Tc / d)) * Tc / (Tc + c))
    # d(rho_vs)/dp_Pa = (100 * ew4 * dfw4/dp_hPa / (Rv*T)) * (dp_hPa/dp_Pa = 1/100)
    return ew4 * (B + (C * Tc^2)) / (Rv * Tk)
end

"""
    retrieve_temperature(M, rho_d, rho_t, rho_liq)

Diagnose the temperature from the prognostic variables of the total-energy set. With the
condensate prognostic the liquid density `ρ_liq = ρ_c + ρ_r` is KNOWN, so the Bryan &
Fritsch (2002) energy identity

    M = (ρ_d C_pd + ρ_w C_pv) T − ρ_liq L_v(T),    ρ_w = ρ_t − ρ_d

(where `M = p + E_t − ρ_t(v²/2 + gz)` [J/m³] is the available enthalpy density) contains no
saturation density, and `L_v(T) = L_v0 + (C_pv − C_l)(T − T_0)` is linear in T. The root is
therefore CLOSED FORM:

    T = [M + ρ_liq (L_v0 − (C_pv − C_l) T_0)] / [(ρ_d C_pd + ρ_w C_pv) − ρ_liq (C_pv − C_l)]

The denominator is `Cfactor + (C_l − C_pv) ρ_liq` with `C_l > C_pv`, hence strictly positive
for any admissible state: there is no iteration, no tolerance, no guess, and no failure mode.

(The liquid density is spelled `rho_liq`, not `rho_l`: `Springsteel.rho_l` is the bulk
density of liquid water, 1000 kg/m³, used throughout `microphysics.jl`.)

Note what is ABSENT. T depends only on `(M, ρ_d, ρ_t, ρ_liq)` — not on the vapor/cloud split,
not on `Q_ss`, and not on `ρ_vs`. That is what decouples the microphysics driver from the
thermodynamics: a `Q_ss` error is now thermodynamically inert. Latent heating still needs no
source term — condensation raises `ρ_l` at fixed `E_t` and `ρ_t`, and this expression turns
that into a warming of `δρ_l·L_v/Cfactor` automatically.

This replaces a univariate Newton iteration on the same identity carrying an extra
`ρ_vs(T,p)` term, whose clamped partition is what manufactured phantom cloud (see the file
header). The two agree to ~1e-12 K wherever the old clamp was inactive.
"""
@inline function retrieve_temperature(M, rho_d, rho_t, rho_liq)

    Cfactor = (rho_d * Cpd) + ((rho_t - rho_d) * Cpv)
    return (M + (rho_liq * (L_v0 - ((Cpv - Cl) * T_0)))) /
           (Cfactor - (rho_liq * (Cpv - Cl)))
end

"""
    moist_entropy_total(Tk, rho_d, q_v, q_l)

Specific moist entropy per unit dry-air mass [J/(kg·K)], INCLUDING the liquid contribution
that `entropy` omits:

    s_t = entropy(T, ρ_d, q_v) + q_l·Cl·log(T/T_0)

This is the heat control variable diffused by the turbulence scheme (the moist analogue of
Straka's dry-entropy diffusion), and the same integrand `conservation_drift` uses for the
total entropy. Because `∂s_t/∂T = C_vt/T` at fixed ρ_d and composition, a diffusive
increment δs_t maps to a heating ρ_d·T·δs_t (= ρ_d C_vt δT). In dry air it reduces to
`dry_entropy_pd(p, ρ_d)` up to a constant.

**`q_v` IS READ THROUGH `max(q_v, 0)`, AND IT HAS TO BE.** `entropy` evaluates
`q_v·R_v·ln(q_v ρ_d / ρ_v0)`, so a negative vapor mixing ratio is a `DomainError`, not a
number — and the vapor is a RESIDUAL (`ρ_t − ρ_d − ρ_c − ρ_r`), so it goes negative wherever
the difference of two independently fitted O(0.1 kg/m³) densities exceeds the vapor actually
present. That is the tropopause: measured on the 3-nest TC initial state, 362 of 6750 nest-1
points between 14.4 and 17.3 km, worst `ρ_v` = −4.5e-6 against a reference `ρ̄_v` of +1.5e-6
there. It killed the first Louis-BL call of the first timestep. The same clamp, for the same
stated reason, has always been on the diagnostics side of this (`benchmarks/common/diagnostics.jl`,
"in dry air the residual vapor sits at 0 ± roundoff, and entropy() takes log(q_v)").

This is the `condensate_floor = :diagnostic` pattern, not `clamp_water!`: nothing is written
back, no mass is converted, no latent heat is pumped, and `s_t` is only ever read as a
mixing control variable. It is also CONTINUOUS — `q ln q → 0` as `q → 0⁺`, so `entropy` has
no kink at zero and the clamped points join the dry limit smoothly. What it does NOT do is
make the negative vapor go away; that deficit is a property of fitting `ρ_t` and `ρ_d`
independently, is expressible as a constraint on neither (`benchmarks/FUTURE_WORK.md`), and
no condensate scheme reaches it.

`q_l` is deliberately NOT clamped: nothing takes its log, a negative condensate must stay
visible in the entropy budget, and under the water transforms it cannot be negative anyway.
"""
function moist_entropy_total(Tk, rho_d, q_v, q_l)

    return entropy(Tk, rho_d, max(q_v, 0.0)) + (q_l * Cl * log(Tk / T_0))
end

"""
    mc_reference_diagnostics(ref_state, z) -> (s_tbar, rho_vbar)

Consistently-RETRIEVED diagnostics of the resting reference for the vertical moist
diffusion: the moist entropy `s_tbar` and clamped vapor density `rho_vbar`, computed
through the exact pipeline `moist_compressible_XZ` runs each step (retrieval at rest,
clamped partition, `moist_entropy_total`). The reference's own `Tbar` is NOT
bit-identical to the retrieved temperature, so subtracting profiles built from it
would leave a spurious O(retrieval tolerance) perturbation that diffusion then acts
on; with these, a resting base has `s_t' ≡ 0` and `rho_v' ≡ 0` bit-for-bit and every
moist diffusive tendency vanishes exactly.
"""
function mc_reference_diagnostics(ref_state, z)

    pbar = ref_pressure(ref_state)
    rho_dbar = ref_rho_d(ref_state)
    rho_tbar = ref_rho_t(ref_state)
    rho_cbar = Springsteel.ref_rho_c(ref_state)
    E_tbar = ref_total_energy(ref_state)
    n = length(z)
    s_tbar = zeros(Float64, n)
    rho_vbar = zeros(Float64, n)
    Pxi_prof = zeros(Float64, n)
    for k in 1:n
        p = pbar[k, 1]
        rho_d = rho_dbar[k, 1]
        rho_t = rho_tbar[k, 1]
        # The condensate is PROGNOSTIC, so the resting cloud is the reference's own
        # fitted profile (exactly 0.0 for a condensate-free base, which fits rho_c from
        # an all-zero vector; positive on a saturated base such as BF02).
        rho_c = rho_cbar[k, 1]
        # Mirror the equation set's per-point pipeline at rest (ke = 0, rho_r = 0);
        # every expression must match mc_driver! bit-for-bit.
        geo = 0.5 * ((0.0 * 0.0) + (0.0 * 0.0)) + (gravity * z[k])
        M = p + (E_tbar[k, 1]) - (rho_t * geo)
        Tk = retrieve_temperature(M, rho_d, rho_t, rho_c)
        # The DENSITY residual under both `vapor_retrieval` modes, deliberately: at the
        # resting fixed point the reference construction makes Q_ssbar exactly the diagnosed
        # supersaturation, so res_qss ≡ res_rho_t here and the blend is the identity whatever
        # weight it would compute. Writing the residual keeps this profile bitwise.
        rho_v = rho_t - rho_d - rho_c
        q_v = rho_v / rho_d
        q_l = rho_c / rho_d
        s_tbar[k] = moist_entropy_total(Tk, rho_d, q_v, q_l)
        rho_vbar[k] = rho_v
        # Local reference sound speed squared γ_m(z)·p̄(z)/ρ̄_t(z) for the acoustic
        # linearization. A DOMAIN-MEAN c̄² (Springsteel's sound_speed_sq) leaves the
        # local deviation of the vertical acoustic operator EXPLICIT under AB3 —
        # the classic reference-state SI instability (Simmons, Hoskins & Burridge
        # 1978): on the Dunion sounding the ±20-25% c² deviations blow up above
        # Co_z ≈ 4.5 even though the operator-consistent scheme is stable to
        # Co_z ≈ 9-18 on a uniform-c base. The PROFILE makes the explicit acoustic
        # remainder O(perturbation) at every level.
        C_vt = Cvd + (q_v * Cvv) + (q_l * Cl)
        R_m = Rd + (q_v * Rv)
        Pxi_prof[k] = ((C_vt + R_m) / C_vt) * p / rho_t
    end
    return (s_tbar = s_tbar, rho_vbar = rho_vbar, Pxi_prof = Pxi_prof)
end

"""
    consistent_qss_reference(ref, z, column) -> PressureReferenceState

Rebuild the reference water partition (`Q_ssbar`, and on a cloudy base `rho_cbar`/`rho_vbar`)
so the RESTING state is a discrete fixed point of the equation set. Opt-in through
`options[:consistent_qss_reference]`; off by default and bitwise inert when off.

Springsteel builds its moisture profiles POINTWISE from the EOS temperature and then fits
them. The equation set never sees those pointwise values: at run time it retrieves T from the
FITTED `(p̄, Ē_t, ρ̄_t, ρ̄_c)` through [`retrieve_temperature`](@ref) and diagnoses
`ρ_v = ρ̄_t − ρ̄_d − ρ̄_c` from the fitted densities, so both differ from the pointwise
construction by the fit error. Two resting tendencies can survive that mismatch:

- `Q̇ = Q_ss·invtau/(1+Q_s)` — the phase change, wherever a gate in
  [`qss_condensation_rates`](@ref) is open;
- `QSSREL = −(Q_ss − (ρ_v − ρ_vs))/τ` — the [`qss_relaxation`](@ref) reconciliation, which is
  thermodynamically inert (T does not read `Q_ss`) but is still a nonzero tendency at rest.

Killing both means making `Q_ssbar` equal the model's OWN diagnosed supersaturation, which is
what this builds:

    ρ_c     = ρ̄_c (fitted)          M = p̄ + Ē_t − ρ̄_t·g·z        (rest: ke = 0, ρ_r = 0)
    T       = retrieve_temperature(M, ρ̄_d, ρ̄_t, ρ_c)
    Q_ssbar = (ρ̄_t − ρ̄_d − ρ_c) − ρ_v*(T, p̄)

The vapor here is the DENSITY residual under both `options[:vapor_retrieval]` modes, and that
is not an omission: the construction above is precisely the statement `res_qss ≡ res_rho_t` at
rest, so [`vapor_retrieval_blend`](@ref) is the IDENTITY on this state whatever weight it
computes, and building the reference from the residual leaves the fixed point intact under
`:blend` as well as `:residual`.

**Cloud-free levels** (`ρ̄_c = 0` — a subsaturated sounding: O01, the TC) are done at that
point. `q_c = 0 < 1e-8` and `S = Q_ssbar/ρ_vs < 0` shut both condensation gates, `invtau_r = 0`
for `ρ_r = 0`, the closure returns `(0.0, 0.0)` EXACTLY, and `QSSREL` vanishes by construction.
Note how much weaker the requirement is than it used to be: the condensate is prognostic, so a
subsaturated base cannot manufacture cloud however the fit falls, and this is now a refinement
rather than the load-bearing repair it was before.

**Condensate-bearing levels** (`ρ̄_c > 0` — a saturated base: Bryan & Fritsch 2002) cannot be
fixed by `Q_ssbar` alone. The cloud gate is necessarily open there, so `Q̇` vanishes only for
`Q_ssbar = 0`, while `QSSREL` vanishes only for `Q_ssbar = ρ_v − ρ_vs`. Both hold together iff
the base is EXACTLY saturated in the model's own arithmetic, so the PARTITION is what has to
move: solve

    g(ρ_c) = ρ_c − (ρ̄_t − ρ̄_d − ρ_v*(T(ρ_c), p̄)) = 0

for `ρ_c` by Newton, using `dT/dρ_c = L_v(T)/(C_factor − ρ_c(C_pv − C_l))` (differentiate the
closed-form retrieval), then set `Q_ssbar = 0` and `ρ̄_v = ρ̄_t − ρ̄_d − ρ_c`. Fixed-point
iteration is NOT usable here: `d/dρ_c` of the map is `−∂ρ_vs/∂T · L_v/C_factor ≈ −2.5`, so the
naive iteration diverges. `ρ̄_t`, `ρ̄_d`, `p̄` and `Ē_t` are untouched, so this moves mass
between the vapor and cloud slots at fixed total density — the hydrostatic balance and the
buoyancy of the base are exactly preserved, which is the whole point of a BF02 base state.

The VALUE slots are the raw targets (not the filtered fit): the fixed point depends on them
exactly, while the derivative slots — which come from a spline fit of those values — multiply
`w ≡ 0` at rest and so cannot disturb it.

Errors per level if the intended outcome is not actually reached: a cloud-free level that is
actually supersaturated (which needs a condensate-bearing reference, not a condensing one), or
a cloudy level whose Newton solve leaves no cloud (the sounding is not saturated there).
"""
function consistent_qss_reference(ref::Springsteel.PressureReferenceState,
                                  z::AbstractVector{Float64}, column)

    pbar = ref_pressure(ref)
    rho_dbar = Springsteel.ref_rho_d(ref)
    rho_tbar = ref_rho_t(ref)
    rho_cbar = Springsteel.ref_rho_c(ref)
    E_tbar = ref_total_energy(ref)
    n = length(z)
    Q_ss_new = zeros(Float64, n)
    rho_c_new = zeros(Float64, n)
    rho_v_new = zeros(Float64, n)
    cloudy_levels = 0
    for k in 1:n
        p = pbar[k, 1]
        p_hPa = p / 100.0
        rho_d = rho_dbar[k, 1]
        rho_t = rho_tbar[k, 1]
        rho_w = rho_t - rho_d
        # Mirror the driver's per-point pipeline at rest (ke = 0, rho_r = 0).
        M = p + E_tbar[k, 1] - (rho_t * (gravity * z[k]))
        # A condensate-free reference fits rho_c from an all-zero vector, so this is
        # EXACTLY 0.0 there; a saturated base (BF02) carries a positive profile.
        rho_c = rho_cbar[k, 1]
        cloudy = rho_c > 0.0

        if cloudy
            cloudy_levels += 1
            # Newton on g(rho_c) = rho_c - (rho_w - rho_vs(T(rho_c), p)), whose derivative
            # is 1 + (drho_vs/dT)(dT/drho_c) with dT/drho_c = L_v(T)/D — see the docstring.
            Cfactor = (rho_d * Cpd) + (rho_w * Cpv)
            for _ in 1:50
                Tk = retrieve_temperature(M, rho_d, rho_t, rho_c)
                g = rho_c - (rho_w - rho_v_sat(Tk, p_hPa))
                D = Cfactor - (rho_c * (Cpv - Cl))
                gp = 1.0 + (drho_vsat_dT(Tk, p_hPa) * L_v(Tk) / D)
                drc = -g / gp
                rho_c += drc
                abs(drc) < 1.0e-15 && break
            end
            rho_c_new[k] = rho_c
            # On the saturation manifold BOTH the phase change and the reconciliation
            # vanish; Q_ssbar = 0 exactly rather than the ~1e-19 the solve would leave.
            Q_ss_new[k] = 0.0
        else
            rho_c_new[k] = rho_c            # exactly 0.0
            Tk = retrieve_temperature(M, rho_d, rho_t, rho_c)
            Q_ss_new[k] = (rho_w - rho_c) - rho_v_sat(Tk, p_hPa)
        end
        rho_v_new[k] = rho_w - rho_c_new[k]

        # VERIFY the run-time outcome through the driver's own expressions, rather than
        # assuming the algebra above reached it.
        Tk_run = retrieve_temperature(M, rho_d, rho_t, rho_c_new[k])
        rho_vs = rho_v_sat(Tk_run, p_hPa)
        q_c = rho_c_new[k] / rho_d
        S = Q_ss_new[k] / rho_vs
        # QSSREL is inert but must still vanish for a true discrete fixed point; it is
        # scaled by rho_vs so the tolerance means "a relative supersaturation of 1e-10".
        rel = (Q_ss_new[k] - (rho_v_new[k] - rho_vs)) / rho_vs
        if cloudy
            # What must not happen is losing the base cloud, which is what makes a
            # saturated base neutrally buoyant.
            rho_c_new[k] > 0.0 || error("consistent_qss_reference: the saturation solve " *
                "leaves no cloud at level $k (z = $(z[k]) m): rho_cbar = " *
                "$(rho_cbar[k, 1]) kg/m^3 in the reference but the saturated partition " *
                "gives rho_c = $(rho_c_new[k]) at T = $Tk_run K, p = $p_hPa hPa. The " *
                "reference's (p, rho_d, rho_t, E_t) are not consistent with a " *
                "saturated state.")
            abs(rel) <= 1.0e-10 || error("consistent_qss_reference: the saturation " *
                "solve did not converge at level $k (z = $(z[k]) m): rho_v - rho_vs = " *
                "$(rho_v_new[k] - rho_vs) kg/m^3 (relative $rel), so the Q_ss " *
                "reconciliation would fire on the resting base.")
        else
            # Exactly the two gates in `qss_condensation_rates`; failing either leaves
            # invtau_c > 0 and the reference condenses at rest.
            (q_c <= 1.0e-8 && S <= 1.0e-4) || error("consistent_qss_reference: the " *
                "reference still condenses at level $k (z = $(z[k]) m): q_c = $q_c " *
                "(needs <= 1e-8), S = $S (needs <= 1e-4), T = $Tk_run K, " *
                "p = $p_hPa hPa. A saturated base state needs a condensate-bearing " *
                "reference (rho_c > 0), not a condensing one.")
        end
    end

    # Value slots EXACT (the fixed point depends on them bit-for-bit); derivative slots
    # from a spline fit of those values (they multiply w == 0 at rest).
    fit3 = function (vals)
        prof = zeros(Float64, n, 3)
        column.uMish[:] .= vals
        Btransform!(column)
        Atransform!(column)
        prof[:, 1] .= vals
        prof[:, 2] .= Ixtransform(column)
        prof[:, 3] .= Ixxtransform(column)
        return prof
    end

    Q_ssbar = fit3(Q_ss_new)
    # A cloud-free base leaves the partition untouched, so its profiles stay the objects
    # they already were — no refit, hence no chance of perturbing a working reference.
    rho_cbar_out = cloudy_levels == 0 ? ref.rho_cbar : fit3(rho_c_new)
    rho_vbar_out = cloudy_levels == 0 ? ref.rho_vbar : fit3(rho_v_new)

    return Springsteel.PressureReferenceState(
        ref.pbar, ref.rho_dbar, rho_vbar_out, rho_cbar_out, ref.rho_tbar,
        ref.Tbar, ref.E_tbar, Q_ssbar, ref.sound_speed_sq)
end

"""
    dry_entropy_pd(p_Pa, rho_d)

Dry-air specific entropy written as an explicit function of the prognostic pressure and
dry-air density, `s_d = C_vd·log(p) − C_pd·log(ρ_d)` (an additive constant is dropped — it
does not affect the diffusion operator). Equal to `moist_entropy_total(T, ρ_d, 0, 0)` up to
that constant via the dry EOS `T = p/(R_d ρ_d)`. Used for the dry-exact horizontal heat
diffusion, whose Laplacian is the two-term chain rule

    s_d,x  = C_vd·p_x/p − C_pd·ρ_d,x/ρ_d
    s_d,xx = C_vd·(p_xx/p − p_x²/p²) − C_pd·(ρ_d,xx/ρ_d − ρ_d,x²/ρ_d²)

on the existing `p`/`ρ_d` derivative slots (no transform of a diagnosed field — see
reference/moist_compressible_diffusion_plan.md). The moist correction terms are deferred;
see reference/moist_compressible_diffusion_handoff.md.
"""
function dry_entropy_pd(p_Pa, rho_d)

    return (Cvd * log(p_Pa)) - (Cpd * log(rho_d))
end

"""
    Q_s_energy(Tk, p_Pa, rho_d, q_v, q_l)

Energy-consistent psychrometric factor (dimensionless) for the supersaturation
relaxation, built from the condensation-induced temperature and pressure tendencies
of the total-energy equation set (density form, no ln(H) term):

    Q_s = [ ∂ρ_vs/∂T (L_v − R_v T)/ρ_d + ∂ρ_vs/∂p R_m (L_v − R_v C_pt T / R_m) ] / C_vt

The (1 + Q_s) factor cancels between the condensation rate and the saturation
chain-rule terms, leaving −Q_ss/τ as the net supersaturation forcing.
"""
function Q_s_energy(Tk, p_Pa, rho_d, q_v, q_l)

    C_vt = Cvd + (q_v * Cvv) + (q_l * Cl)
    R_m = Rd + (q_v * Rv)
    C_pt = C_vt + R_m
    p_hPa = p_Pa / 100.0
    Lv = L_v(Tk)
    Q_s = ((drho_vsat_dT(Tk, p_hPa) * (Lv - (Rv * Tk)) / rho_d) +
           (drho_vsat_dp(Tk, p_hPa) * R_m * (Lv - (Rv * C_pt * Tk / R_m)))) / C_vt
    return Q_s
end

"""
    qss_condensation_rate(Q_ss, rho_v, rho_c, rho_d, Tk, p_hPa, Q_s, ts, max_N_c=100.0)

Cloud condensation/evaporation rate [kg/m³/s] from the prognostic supersaturation
density, with the droplet-growth timescale of [`q_condensation`](@ref) (Twomey-type
nucleation, minimum droplet radius) and the energy-consistent psychrometric factor:

    Q̇_cond = Q_ss (1/τ) / (1 + Q_s)

`rho_v` is the CLAMPED diagnostic vapor density. Evaporation is limited by the
available cloud water so ρ_c cannot be driven negative, and condensation by the
available vapor, which kills phantom condensation in dry air where Q_ss tracking
drift can otherwise indicate spurious supersaturation. Subsaturated cloud-free air
returns zero.

This single-category delegate takes the DEFAULT (forward-Euler) bounds of
[`qss_condensation_rates`](@ref), which is what it has always used and what its tests
pin. Production passes the AB3-sized bounds explicitly — see [`_ab3_sink_bound`](@ref).
"""
qss_condensation_rate(Q_ss, rho_v, rho_c, rho_d, Tk, p_hPa, Q_s, ts, max_N_c=100.0) =
    qss_condensation_rates(Q_ss, rho_v, rho_c, 0.0, rho_d, Tk, p_hPa, Q_s, ts, 0.0,
                           max_N_c)[1]

"""
    qss_condensation_rates(Q_ss, rho_v, rho_c, rho_r, rho_d, Tk, p_hPa, Q_s, ts, N_r,
                           max_N_c=100.0; N_0=0.0) -> (Qdot_c, Qdot_r)

Two-category condensation/evaporation rates [kg/m³/s] from the generalized
supersaturation relaxation `1/τ = 1/τ_c + 1/τ_r` (see
reference/Scythe_moist_compressible.tex): the total rate `Q_ss (1/τ)/(1+Q_s)` splits
between the cloud and rain channels in proportion to their inverse timescales. Rain
gains a (small, `N_r`-controlled) share of supersaturated condensation ONLY where
cloud coexists (`q_c > 1e-8`, the cloud channel's own existence threshold): the
physical pathway to rain is condensation → cloud → autoconversion, and an ungated
rain channel grows rain from arbitrarily small seeds in cloud-free supersaturated
air (rate ∝ `rho_r^{1/3}` under the monodisperse fixed-`N_r` closure, non-Lipschitz
at zero). The rain-channel timescale is [`invtau_rain`](@ref) by default; the
keyword `N_0 > 0` [m⁻⁴] selects the exponential Marshall-Palmer closure
[`invtau_rain_mp`](@ref) instead (`N_r` is then unused), leaving the split
arithmetic, gate and limiters untouched. Rain evaporation in subsaturated air is
unconditional, so no separate
rain-evaporation parameterization (O01's `Q_evap`) is needed. The ventilation
enhancement lives inside [`invtau_rain`](@ref).

Limiters: cloud evaporation is bounded below by `floor_c`, rain evaporation by `floor_r`, and
if the combined condensation would exceed `ceil_v` both channels are rescaled proportionally.
With the rain channel inactive (`Qdot_r == 0`) the vapor cap reduces to the historical
`min(Qdot, ceil_v)`, keeping the single-category [`qss_condensation_rate`](@ref) delegate
bit-identical to its pre-rain behavior.

The three bounds default to the FORWARD-EULER budgets this function used to hard-code
(`−max(rho_c,0)/ts`, `−max(rho_r,0)/ts`, `+max(rho_v,0)/ts`), which is what every unit-test
caller and the single-category delegate get. The integrator is AB3, not Euler, so `mc_driver!`
passes the exact three-level bounds instead — see [`_ab3_sink_bound`](@ref) for why that
matters and by how much. Note the vapor ceiling is the SAME defect as the two condensate
floors: it is what stops condensation removing more vapor than the residual holds, and until
now it was Euler-sized and not even reachable by the `cap_factor` probe.
"""
function qss_condensation_rates(Q_ss, rho_v, rho_c, rho_r, rho_d, Tk, p_hPa, Q_s, ts,
                                N_r, max_N_c=100.0; N_0=0.0, cap_factor=1.0,
                                floor_c = -cap_factor * max(rho_c, 0.0) / ts,
                                floor_r = -cap_factor * max(rho_r, 0.0) / ts,
                                ceil_v = max(rho_v, 0.0) / ts)

    rho_vs = rho_v_sat(Tk, p_hPa)
    S = Q_ss / rho_vs                    # supersaturation (ratio - 1)
    q_c = max(rho_c, 0.0) / rho_d

    # Cloud channel: droplet number and radius logic mirrors q_condensation
    invtau_c = 0.0
    N_c = max_N_c
    r_c = cloud_droplet_radius(N_c, q_c, rho_d)
    if q_c > 1.0e-8
        if S < 0.0 && r_c < 1.0
            # Evaporating small droplets: count the 1-micron drops available
            r_c = 1.0
            N_c = cloud_droplet_number(r_c, q_c, rho_d)
            if N_c < 1.0
                N_c = 0.0
            end
        end
        if N_c > 0.0 && r_c > 0.0
            invtau_c = invtau_condensation(Tk, p_hPa, N_c, r_c)
        end
    elseif S > 1.0e-4
        # Nucleation: linear interpolation of the Twomey relationship
        if r_c < 1.0
            r_c = 1.0
            N_c = min(1.0e4 * N_c * S, max_N_c)
        end
        if N_c > 0.0 && r_c > 0.0
            invtau_c = invtau_condensation(Tk, p_hPa, N_c, r_c)
        end
    end
    # (no cloud and not supersaturated: invtau_c stays 0 — rain may still evaporate)

    # Rain CONDENSATION is gated on cloud presence (same q_c > 1e-8 existence
    # threshold as the cloud channel): the physical pathway to rain is condensation
    # -> cloud -> autoconversion, and in cloud-free supersaturated air the ungated
    # channel grows rain from arbitrarily small seeds in finite time (the rate is
    # ∝ rho_r^{1/3} under the fixed-N_r monodisperse closure, non-Lipschitz at zero
    # — the O01 spurious-blob pathway). Evaporation (Q_ss <= 0) is unconditional.
    # The channel timescale is monodisperse fixed-N_r by default; N_0 > 0 selects the
    # exponential (Marshall-Palmer) DSD closure. Both are non-Lipschitz at zero rain
    # (rho_r^{1/3} and rho_r^{1/2} respectively), so the gate applies to either.
    invtau_r = (Q_ss > 0.0 && q_c <= 1.0e-8) ? 0.0 :
               (N_0 > 0.0 ? invtau_rain_mp(Tk, p_hPa, N_0, rho_r, rho_d) :
                            invtau_rain(Tk, p_hPa, N_r, rho_r))
    invtau = invtau_c + invtau_r
    if invtau == 0.0
        return (0.0, 0.0)
    end

    Qdot = Q_ss * invtau / (1.0 + Q_s)
    # An inactive channel gets an exact 0.0 (Qdot * 0.0 would be -0.0 for evaporation);
    # a lone active channel gets Qdot exactly (invtau/invtau == 1.0), which keeps the
    # single-category delegate bit-identical.
    Qdot_c = invtau_c == 0.0 ? 0.0 : Qdot * (invtau_c / invtau)
    Qdot_r = invtau_r == 0.0 ? 0.0 : Qdot * (invtau_r / invtau)

    # No negative water: each condensate's evaporation limited by its own mass, sized for the
    # integrator that will apply it (`floor_c`/`floor_r`; see the docstring).
    Qdot_c = max(Qdot_c, floor_c)
    Qdot_r = max(Qdot_r, floor_r)

    # Condensation limited by the available vapor. With the rain channel inactive this
    # is the historical min(); with both active the channels rescale proportionally.
    if Qdot_r == 0.0
        Qdot_c = min(Qdot_c, ceil_v)
    elseif Qdot_c + Qdot_r > ceil_v
        scale = ceil_v / (Qdot_c + Qdot_r)
        Qdot_c *= scale
        Qdot_r *= scale
    end
    return (Qdot_c, Qdot_r)
end

import Springsteel: ref_pressure, ref_rho_t, ref_total_energy, ref_qss

# Canonical slot order for the total-energy set (u=4, w=5 in the shared-machinery
# positions used by the other XZ sets). rho_c is APPENDED at 9 rather than placed
# next to rho_r: slots 1-8 appear as hardcoded literals throughout the kernel, the
# acoustic solvers and mc_boundary_layer.jl, and appending keeps every one valid
# (the same rule MC_VARS_CYL follows for v — see the comment there).
const MC_VARS = ["p", "rho_d", "rho_t", "u", "w", "E_t", "Q_ss", "rho_r", "rho_c"]

# ── Rigid-wall pressure compatibility condition ───────────────────────────────

"""
    mc_wall_bc_active(grid) -> Bool

True if `p` declares the inhomogeneous Neumann condition (`CubicBSpline.R1T1X`)
at either vertical wall, i.e. this grid wants [`update_mc_wall_bc!`](@ref) called
before every fit. Cheap enough for the per-step path: two `haskey` lookups.
"""
function mc_wall_bc_active(grid)
    grid.kbasis isa Springsteel.SplineBasisArray || return false
    haskey(grid.params.vars, "p") || return false
    kcol = grid.kbasis.data[grid.params.vars["p"]]
    return haskey(kcol.params.BCL, "X1") || haskey(kcol.params.BCR, "X1")
end

"""
    update_mc_wall_bc!(grid, src)

Install the exact rigid-wall compatibility condition on `p'` before a fit.

`w` is Dirichlet at both vertical walls, so `w ≡ 0` there for ALL time. Along the
wall that also kills the advection terms (`∂w/∂r = 0` because `w` vanishes at every
radius, and `w ∂w/∂z = 0`), so the vertical momentum equation collapses to an
identity rather than an evolution equation:

    ∂p'/∂z |_wall  =  -g ρ_t' |_wall

This is NOT an assumption of hydrostatic balance in the interior — only at the two
rigid walls, where it is exact. (If vertical diffusion or a boundary-layer scheme is
ever given a nonzero `w` tendency AT the wall, its contribution `ρ_t · ∂w/∂t|_wall`
belongs on the right-hand side here; with `Kvdiff = 0` and `w` Dirichlet there is
none, and the assertion in `mc_wall_bc_requires_no_w_forcing` guards the assumption.)

Why it matters: a homogeneous `NeumannBC` is the special case `ρ_t' = 0`, which a
balanced vortex violates precisely where its surface pressure deficit lives — that
was the drain of `reference/HANDOFF_2026-07-20.md`. `SecondDerivativeBC` leaves `∂p'/∂z`
free instead, which restores the balance but costs a factor ~2.7 in stable timestep,
because the semi-implicit acoustic solve eliminates `w` (φ = ρ̄_t w is Dirichlet in
the Helmholtz solve) and therefore cannot see a wall derivative the refit injects —
see `reference/SI_WALL_BC_CEILING.md`. R1T1X resolves the conflict: the admissible subspace
stays R1T1's, so the solve stays operator-consistent and the ceiling stays high,
while the affine `ahat` offset carries the nonzero derivative the balance needs.

`src` is a `(npts, nvars, nderiv)` physical array carrying the fitted field AND its
z-derivatives (slots 1, 4, 5) — normally `grid.physical`. `ρ_t'` at the wall is a
second-order Taylor step off the nearest mish point, since the mish never lands on
the boundary itself. A 2-D `src` is accepted for tests but degrades to the value
alone.

`relax` ∈ (0, 1] is the fraction of the way the stored derivative moves toward the
target on this call, i.e. a first-order filter with timescale `Δt/relax`. **It must
be well below 1 in a run.** With `relax = 1` the wall derivative is recomputed from
scratch every acoustic step, which closes a feedback loop — `∂p'/∂z|_wall` sets the
wall pressure, the acoustic solve moves `ρ_t'` at the wall, which resets
`∂p'/∂z|_wall` — and it is unstable: the stored value alternates sign step to step
and the column goes non-finite in a few hundred steps (measured). The affine-offset
argument for R1T1X's stability holds for a FIXED `ahat`; a state-dependent one adds
an explicit path that has to be kept off the acoustic timescale. The physical
justification is the same fact: the wall gradient is a property of the BALANCED
vortex and evolves on hours, so filtering it over ~minutes loses nothing real.
`relax = 1` remains the right choice for a one-shot initialization, where there is
no loop to close.
"""
# Radial spline used only to differentiate the wall profile: same knots as the
# variable's i-basis but NATURAL boundary conditions, so neither a wall Dirichlet
# nor a nesting R3X payload can bend the wall data. Cached per (domain, cells);
# Springsteel caches the factorised template underneath, so a miss is cheap.
const _WALL_DERIV_SPLINES = Dict{NTuple{3,Any}, Springsteel.CubicBSpline.Spline1D}()
const _WALL_DERIV_LOCK = ReentrantLock()

function _wall_deriv_spline(grid, v::Int)
    isp = grid.ibasis.data[1, v]
    sp = isp.params
    key = (sp.xmin, sp.xmax, sp.num_cells)
    lock(_WALL_DERIV_LOCK) do
        get!(_WALL_DERIV_SPLINES, key) do
            Springsteel.CubicBSpline.Spline1D(
                Springsteel.CubicBSpline.SplineParameters(
                    xmin = sp.xmin, xmax = sp.xmax, num_cells = sp.num_cells,
                    BCL = Springsteel.CubicBSpline.R0,
                    BCR = Springsteel.CubicBSpline.R0))
        end
    end
end

function update_mc_wall_bc!(grid, src::AbstractArray; relax::Float64 = 1.0)
    gp = grid.params
    kDim = gp.kDim
    iDim = gp.iDim
    vars = gp.vars
    p_i = vars["p"]
    rhot_i = vars["rho_t"]
    kcol = grid.kbasis.data[p_i]
    wall_du = grid.kbasis.wall_du

    # rho_t' at the wall, estimated as the MEAN over the boundary cell (the three
    # nearest mish points).
    #
    # NOT an extrapolation to the wall itself, which is what this did first — both
    # a 3-point Lagrange fit through the mish values and a second-order Taylor step
    # off the fitted derivatives. Both blow up, and for the same reason: they weight
    # the near-wall curvature by d^2/2 ~ 1.6e3 m^2, which amplifies whatever
    # grid-scale content sits in rho_t' near the boundary. That amplified signal
    # becomes p's wall derivative, which drives the acoustic mode, which enlarges
    # the near-wall grid-scale content — a loop with gain > 1 that no amount of time
    # relaxation suppresses, because the TARGET is what is growing. Measured: the
    # resting column goes non-finite in ~30 steps with either extrapolation, and is
    # quiet indefinitely with `ahat` pinned to zero, so the feedback is the whole of
    # the instability and the R1T1X plumbing is exonerated.
    #
    # A cell mean is a smoother rather than an amplifier. It is biased by O(dz) times
    # the true gradient, which is the right trade: the wall condition needs to track
    # the BALANCED state's wall value, not resolve a grid-scale feature.
    if haskey(kcol.params.BCL, "X1")
        @inbounds for r in 1:iDim
            n = (r - 1) * kDim + 1
            rt = (src[n, rhot_i, 1] + src[n+1, rhot_i, 1] + src[n+2, rhot_i, 1]) / 3
            old = wall_du[r, p_i, 1, 1]
            wall_du[r, p_i, 1, 1] = old + relax * ((-gravity * rt) - old)
        end
    end
    if haskey(kcol.params.BCR, "X1")
        @inbounds for r in 1:iDim
            n = r * kDim
            rt = (src[n, rhot_i, 1] + src[n-1, rhot_i, 1] + src[n-2, rhot_i, 1]) / 3
            old = wall_du[r, p_i, 2, 1]
            wall_du[r, p_i, 2, 1] = old + relax * ((-gravity * rt) - old)
        end
    end

    # Radial derivatives of the wall data (levels 2 and 3). The 2-D transform
    # evaluates the i-derivative BEFORE fitting in k, so its dr = 1 / dr = 2
    # passes need d(wall)/dr and d2(wall)/dr2 as their `ahat`; feeding them
    # level 1 asserts dg/dr = g and wrecks the radial pressure gradient in the
    # boundary cell.
    #
    # Differentiated through a CLEAN natural-BC spline, NOT the variable's own
    # i-basis. A nested child patch carries an R3X junction condition whose
    # `ahat` holds the PARENT's payload; fitting the wall profile through that
    # spline would inject the parent's pressure data into the wall derivative and
    # constrain the profile to the junction trio. Both nested hold tests died on
    # nest 3, whose inner edge is exactly such a junction.
    isp = _wall_deriv_spline(grid, p_i)
    for s in 1:2
        (s == 1 ? haskey(kcol.params.BCL, "X1") : haskey(kcol.params.BCR, "X1")) || continue
        @inbounds for r in 1:iDim
            isp.uMish[r] = wall_du[r, p_i, s, 1]
        end
        Springsteel.CubicBSpline.SBtransform!(isp)
        Springsteel.CubicBSpline.SAtransform!(isp)
        d1 = Springsteel.CubicBSpline.SIxtransform(isp)
        d2 = Springsteel.CubicBSpline.SIxxtransform(isp)
        @inbounds for r in 1:iDim
            wall_du[r, p_i, s, 2] = d1[r]
            wall_du[r, p_i, s, 3] = d2[r]
        end
    end
    return grid
end

# ── Water positivity ───────────────────────────────────────────────────────────

"""
Row names of `ModelTile.mc_water_stats`, accumulated per thread by [`clamp_water!`](@ref):

| row | meaning |
|-----|---------|
| `:total`   | Σ negative water encountered [kg/m³, summed over gridpoints and steps] |
| `:min_c`   | most negative `ρ_c` seen [kg/m³] |
| `:min_r`   | most negative `ρ_r` seen [kg/m³] |
| `:worst_dT`| largest implied latent-heat kick of a single event [K] (see below) |
| `:count`   | number of gridpoint-steps carrying negative water |
| `:warned`  | internal: largest `worst_dT` already warned about (thread 1 only) |

Rows 7 onward are the per-step **production budget**, written by [`water_budget_probe!`](@ref)
and reset every step by [`water_budget_trace`](@ref); they answer which term is driving the
water negative, which rows 1-6 (cumulative extrema) structurally cannot. Each block records the
term-by-term tendency AT the gridpoint where that species is most negative this step:

| row | meaning |
|-----|---------|
| `:b_r_val`  | most negative `ρ_r` in this thread's columns this step [kg/m³] |
| `:b_r_adv`  | `-v·∇ρ_r'` there [kg/m³/s] |
| `:b_r_cdiv` | `-ρ_r ∇·v` there — the compressibility term, a sign-blind linear amplifier |
| `:b_r_src`  | `Qdot_r` (rain-channel condensation/evaporation) there |
| `:b_r_auto` | `AUTO_COLL` (autoconversion + collection from cloud) there |
| `:b_r_sed`  | `-∂F_r/∂z` (sedimentation flux divergence) there |
| `:b_r_div`  | `∇·v` there [1/s] |
| `:b_r_w`    | `w` there [m/s] |
| `:b_r_z`    | height of the point [m] |
| `:b_r_eul`  | the FORWARD-EULER projection `ρ + ts·f_n` at that same point [kg/m³] |
| `:b_r_now`  | the current value `ρ` at that same point [kg/m³] |
| `:b_c_*`    | the same for `ρ_c`, with `:b_c_src` = `Qdot` and `:b_c_auto` = `-AUTO_COLL` |

The final two blocks are the per-step **depletion census**, written by
[`water_depletion_probe!`](@ref) just before `explicit_timestep`. The budget block above says
what happens at ONE point; these say how widespread it is, and they are what separates a
limiter sized for the wrong integrator from a stiff source term:

| row | meaning |
|-----|---------|
| `:d_r_n`      | gridpoints in this thread's columns with `ρ_r > 0` |
| `:d_r_cevap`  | of those, how many have the depletion sink pinned at its bound |
| `:d_r_cauto`  | how many have `AUTO_COLL` pinned at `avail` (cloud block only; 0 for rain/vapor) |
| `:d_r_eul`    | max forward-Euler depletion fraction `-ts·f_n/ρ` |
| `:d_r_ab3`    | max ACTUAL depletion fraction `-(ρ^{n+1}-ρ)/ρ` under the run's AB3 weights |
| `:d_r_neg`    | how many points the step actually drives negative (`:d_r_ab3 > 1` pointwise) |
| `:d_r_stiff`  | how many exceed AB3's real-axis stability limit (0.545) with NO cap active |
| `:d_r_infeas` | how many have an INADMISSIBLE HISTORY — the two previous sink levels alone already carry the point negative, so no bound on the current level can fix it and [`_ab3_sink_bound`](@ref) has been clamped at 0 |
| `:d_r_mab3`   | max depletion fraction from the MICROPHYSICS ALONE, `-AB3(micro history)/ρ` |
| `:d_r_mneg`   | how many points the microphysics alone would drive negative (`:d_r_mab3` past `MICRO_DEPLETION_TOL`) |
| `:d_c_*`      | the same for `ρ_c` |
| `:d_v_*`      | the same for the residual `ρ_v`, whose slot tendency is `f_3 - f_2 - f_8 - f_9` and whose depletion sink is the condensation `-(Qdot + Qdot_r)` |
| `:v_gap`      | max over the tile this step of the VAPOR PARTITION GAP `\\|ρ_v − res_rho_t\\|` [kg/m³] — identically 0 under `options[:vapor_retrieval] = :residual`; under `:blend` it equals `w·τ_qss·\\|QSSREL\\|` where the correction is under half the cap, and is bounded by `dcap·ρ_vs` (default `0.02·ρ_vs`) wherever the saturator is active (see [`vapor_retrieval_blend`](@ref)) |

**Read `:d_*_mab3`, not `:d_*_ab3`, to judge the depletion bound.** `:d_*_ab3` is the fraction
of the FULL slot tendency, so it is dominated by advection and `-ρ∇·v` at points carrying
almost no condensate — where any fixed tendency divided by a near-zero `ρ` is large, and no
microphysical limiter has, or should have, any purchase. `:d_*_mab3` is the part the bound
governs, and under `water_cap_mode = :ab3` it is ≤ 1 by construction (`:d_*_mneg` = 0);
under `:euler` it runs to ≈ 23/12, which is the defect the mode exists to reproduce.
"""
const MC_WATER_STATS = (:total, :min_c, :min_r, :worst_dT, :count, :warned,
                        :b_r_val, :b_r_adv, :b_r_cdiv, :b_r_src, :b_r_auto, :b_r_sed,
                        :b_r_div, :b_r_w, :b_r_z, :b_r_eul, :b_r_now,
                        :b_c_val, :b_c_adv, :b_c_cdiv, :b_c_src, :b_c_auto, :b_c_sed,
                        :b_c_div, :b_c_w, :b_c_z, :b_c_eul, :b_c_now,
                        :b_pre_r, :b_pre_c,
                        :d_r_n, :d_r_cevap, :d_r_cauto, :d_r_eul, :d_r_ab3,
                        :d_r_neg, :d_r_stiff, :d_r_infeas, :d_r_mab3, :d_r_mneg,
                        :d_c_n, :d_c_cevap, :d_c_cauto, :d_c_eul, :d_c_ab3,
                        :d_c_neg, :d_c_stiff, :d_c_infeas, :d_c_mab3, :d_c_mneg,
                        :d_v_n, :d_v_cevap, :d_v_cauto, :d_v_eul, :d_v_ab3,
                        :d_v_neg, :d_v_stiff, :d_v_infeas, :d_v_mab3, :d_v_mneg,
                        :v_gap)

"""
First row of the per-step budget block for each species in `mc_water_stats`.

Both blocks are `MC_BUDGET_N` rows with the SAME layout, so one probe writes either. Cloud has
no sedimentation, so `:b_c_sed` is always zero and is not printed — giving the two blocks
different lengths silently walks off the end of the matrix into the next thread's column.
"""
const MC_BUDGET_R = 7
const MC_BUDGET_C = 18
const MC_BUDGET_N = 11
"""
First row of the per-step depletion census for each species; `MC_DEPLETION_N` rows each,
same layout, written by [`water_depletion_probe!`](@ref).
"""
const MC_DEPLETION_R = 31
const MC_DEPLETION_C = 41
const MC_DEPLETION_V = 51
const MC_DEPLETION_N = 10
"""
Row holding the per-step maximum vapor PARTITION GAP `max|ρ_v − res_rho_t|` over the tile.

Sits AFTER the three `MC_DEPLETION_N`-row blocks, deliberately outside them: it is one scalar,
not a fourth species, and giving the blocks a ragged length silently walks the census off the
end of a thread's column. It lives in the per-step region (`MC_BUDGET_FIRST:end`), so it is
reset every step and always describes the step just reported.
"""
const MC_VAPOR_GAP = 61
"""
Threshold for `:d_*_mneg`, the count of points the MICROPHYSICS alone drives negative.

A point sitting exactly on its depletion bound lands at `1 + O(eps)`, not at 1: the `t ≥ 3`
branch of [`_ab3_sink_bound`](@ref) divides by 23, which is not representable, so the bound
cannot cancel the increment exactly. Counting `> 1.0` would therefore report every capped
point as a violation. This threshold sits ~1e7 ULP above the rounding and ~1e9 below the
`23/12` a forward-Euler-sized bound produces, so `:d_*_mneg` reads 0 under
`water_cap_mode = :ab3` and the full capped population under `:euler`.
"""
const MICRO_DEPLETION_TOL = 1.0 + 1.0e-9
"""
Per-STEP pre-fit minima (rows 29/30), written by `clamp_water!` and reset every step.

Rows 2/3 (`:min_c`, `:min_r`) are cumulative, so they cannot be differenced against the
post-fit reconstruction to separate what the column step did from what the refit did. These
can: `post_t - pre_t` is the refit's contribution and `pre_t - post_{t-1}` is the column
step's, and the two sum to the step's change exactly.
"""
const MC_PRE_R = 29
const MC_PRE_C = 30
"""First budget row; rows `MC_BUDGET_FIRST:end` are reset every step."""
const MC_BUDGET_FIRST = MC_BUDGET_R

"""
    positivity_reference_profile(name, ref_state) -> AbstractVector or nothing

The reference profile a positivity-bounded prognostic is carried against, or `nothing` when
the variable is a TOTAL and its bound needs no offset.

`u, w, rho_r` are totals; `p, rho_d, rho_t, E_t, Q_ss, rho_c` are perturbations from the
pressure reference (see the prognostic-slot semantics above). A zero bound on a PERTURBATION
is not merely conservative, it is wrong: it would pin the field at or above its reference and
forbid the cloud from ever evaporating below `ρ̄_c`. Hence anything not recognised here throws
rather than silently taking the factory's constant bound.

The transformed names `nu_c`/`nu_r` deliberately fall through to the error. A coefficient bound
on a control variable is a different constraint from a bound on the density, and
[`install_positivity_bounds!`](@ref) refuses the combination before reaching here; this is the
backstop for a configuration that somehow gets past it.
"""
function positivity_reference_profile(name::AbstractString, ref_state)
    name in ("rho_r", "u", "w", "v") && return nothing         # totals: no offset
    name in ("nu_c", "nu_r") &&
        error("positivity is declared for \"$name\", which is a CONTROL VARIABLE, not a " *
              "density: a box constraint on its coefficients would bound the transform of " *
              "the field rather than the field. The transform already makes the recovered " *
              "density non-negative by construction — drop \"$name\" from positivity.")
    prof = name == "rho_c" ? Springsteel.ref_rho_c(ref_state) :
           name == "rho_d" ? ref_rho_d(ref_state) :
           name == "rho_t" ? ref_rho_t(ref_state) :
           error("positivity is declared for \"$name\", which is carried as a perturbation " *
                 "from the reference state and has no offset rule here. A constant bound on " *
                 "a perturbation pins the field at or above its reference — add the variable " *
                 "to positivity_reference_profile before enabling it.")
    # Springsteel returns the scalar 0.0 for reference states that carry no such profile.
    prof isa Number && return nothing
    return view(prof, :, 1)
end

"""
    bhyp(rho, mu) -> n
    ahyp(n, mu) -> rho
    ahyp_smooth(n, mu) -> rho
    dbhyp(rho, mu) -> dn/drho

Ooyama (2001, JAS 58, 2073–2102) **biased hyperbolic transform** and its inverses, Eqs.
4.19/4.20/4.23. `reference/ooyama_jas2001.pdf`.

    n   = bhyp(rho)  = ½[(rho + μ) − μ²/(rho + μ)]
    rho = ahyp(n)    = √(n² + μ²) + n − μ        (0 for n ≤ 0 — the published quasi-inverse)
    J   = dn/drho    = ½[1 + μ²/(rho + μ)²]

The map is linear (`n ≈ rho/2`; Ooyama's factor ½ is what removes the coefficient from the
inverse) for `rho ≫ μ`, and stretches `rho` only where it is small enough to be
meteorologically insignificant. `bhyp(0) = ahyp(0) = 0` exactly.

**Why this family and not another.** Every monotone `f: ℝ → (0, ∞)` has `f′ → 0` as `n → −∞`,
so the source quotient `S/f′` is unbounded at the cloud edge and an explicit step overshoots
there. Ooyama biases the hyperbola instead, giving range `(−μ, ∞)`; evaluating `J` at the
RECOVERED density (his Eqs. 4.21–4.22, `D_t n = (dn/drho)·[rho-space tendency]`) then keeps it
in `[0.5, 1]` for `rho ≥ 0`. Measured: the softplus and quadratic-touchdown alternatives fit
exactly as well and detonate on first nucleation with peaks of 1e7 and 1e25 against a baseline
1.2e-2. See `reference/FINDINGS_CONDENSATE_STAGE1.md` §3 and
`benchmarks/condensate_transform_probe.jl`.

`ahyp_smooth` is the strict inverse everywhere — C^∞, range `(−μ, ∞)` — where `ahyp` is
Ooyama's C⁰ quasi-inverse pinned at exactly 0 for `n ≤ 0`. Ooyama notes the strict inverse
reaches at worst `−μ`, so the clip's adjustment never exceeds `μ`. The two are measurably
equivalent (probe: transport peak 2.9376e-3 both, mass drift +1.077 both, nucleation peak
1.2167e-2 vs 1.2166e-2); `ahyp` ships because it gives `rho ≥ 0` exactly, `ahyp_smooth` is
retained for any consumer that needs `f″` at the cloud edge.

`μ` is a partial DENSITY here (Ooyama's is a mixing ratio). At `μ = 1e-7 kg/m³` the residual
negativity costs at most 7.2e-3 K, and that worst point is the model top where `rho_d ~ 0.04`
shrinks the retrieval denominator; near the ground it is ~2.5e-4 K.
"""
#
# **Both maps are written in cancellation-free form**, not as Eqs. 4.19/4.20 read. They are
# the same functions algebraically:
#
#     ½[(ρ+μ) − μ²/(ρ+μ)]  =  ρ(ρ + 2μ) / (2(ρ + μ))
#     √(n²+μ²) + n − μ      =  n + n² / (√(n²+μ²) + μ)
#
# The paper's forms subtract two nearly equal quantities when `ρ ≪ μ` and when `|n| ≪ μ`
# respectively, and that is exactly the regime the whole transform exists to represent.
# Measured over `ρ ∈ [1e-14, 1]`: the paper's forms give `bhyp(0) = 6.6e-24` instead of zero
# and a round-trip relative error reaching 1.7e-9 at `ρ = 1e-14`; these give `bhyp(0) = 0.0`
# exactly and a round trip good to 2.2e-16 everywhere. The exact zero is load-bearing --
# `condensate_slot` relies on it so a cloud-free initial condition converts to exactly 0.0.
@inline bhyp(rho, mu) = (rho * (rho + (2.0 * mu))) / (2.0 * (rho + mu))
@inline ahyp_smooth(n, mu) = n + ((n * n) / (sqrt((n * n) + (mu * mu)) + mu))
@inline ahyp(n, mu) = n <= 0.0 ? 0.0 : ahyp_smooth(n, mu)
@inline dbhyp(rho, mu) = 0.5 * (1.0 + (mu * mu) / ((rho + mu) * (rho + mu)))

"""
    condensate_slot(rho_c, rho_cbar, transform, mu) -> slot value

What an initial condition must write into slot 9 for a physical cloud density `rho_c` against
a reference `rho_cbar`. Under `:none` that is the perturbation `rho_c - rho_cbar`; under a
transform it is the CONTROL-variable deviation `bhyp(rho_c) - bhyp(rho_cbar)`, which is what
Ooyama predicts (§4d: "actual prediction of n is performed in terms of its deviation n' from
the background n̂ = bhyp(m̂)").

A cloud-free initial condition on a cloud-free reference gives exactly `0.0` under both,
because `bhyp(0) = 0` exactly — which is why every current benchmark configuration needs no
initial-condition change to run transformed.
"""
@inline function condensate_slot(rho_c, rho_cbar, transform::Symbol, mu)
    transform === :none && return rho_c - rho_cbar
    return bhyp(rho_c, mu) - bhyp(rho_cbar, mu)
end

"""
    recover_rho_c(slot, rho_cbar, transform, mu) -> rho_c

The inverse of [`condensate_slot`](@ref): the physical cloud density a slot-9 value stands
for. `:none` is the plain `slot + rho_cbar` the code always did, bit for bit.

Module-level rather than a closure at the call sites: `clamp_water!` runs this once per
gridpoint per step inside the loop the allocation tests hold at zero, and a captured closure
there is exactly the kind of thing that stops eliding.
"""
@inline function recover_rho_c(slot, rho_cbar, transform::Symbol, mu)
    transform === :none && return slot + rho_cbar
    n = slot + bhyp(rho_cbar, mu)
    return transform === :bhyp ? ahyp(n, mu) : ahyp_smooth(n, mu)
end

"""
    rain_slot(rho_r, transform, mu) -> slot value
    recover_rho_r(slot, transform, mu) -> rho_r

The rain (slot 8) pair of [`condensate_slot`](@ref) / [`recover_rho_c`](@ref), and simpler than
they are: rain is carried as a TOTAL with no reference profile (`ρ̄_r ≡ 0` in every
configuration and in the reference-state format itself), so there is no background to subtract
and the slot IS the control variable rather than its deviation.

`rain_slot(0.0, …) == 0.0` exactly, because `bhyp(0) == 0` exactly, which is why no initial
condition needs changing to run transformed — none of the `*_mc!` initializers seeds rain.
"""
@inline function rain_slot(rho_r, transform::Symbol, mu)
    transform === :none && return rho_r
    return bhyp(rho_r, mu)
end

@doc (@doc rain_slot)
@inline function recover_rho_r(slot, transform::Symbol, mu)
    transform === :none && return slot
    return transform === :bhyp ? ahyp(slot, mu) : ahyp_smooth(slot, mu)
end

"""
    condensate_transform_mode(options) -> Symbol

Which control variable slot 9 carries. `:none` (the default, and the state of every
configuration that does not set the key) means the slot IS the cloud density and the whole
transform is bitwise absent. `:bhyp` means it carries `n = bhyp(rho_c)` and the density is
recovered by [`ahyp`](@ref); `:bhyp_smooth` uses `ahyp_smooth` instead.

**What the transform buys, and what it does not.** It does not reduce the ringing: the
excursion simply happens in `n`, at the same amplitude in `rho`-equivalent terms (measured:
`min(n)` = −2.01e-4 against the untransformed `min(rho)` = −4.33e-4, exactly Ooyama's factor
½). What it buys is that the RECOVERED density is bounded below by `−μ` whatever `n` does, so
the ringing can no longer feed the retrieval. That matters because the negative region is a
**physics-free reservoir**: every rate function is `max(rho,0)`-guarded, so a negative point
has no evaporation, no autoconversion and no collection, while the positive overshoot at the
peak is consumed every step. On the shipped O01 quick run that reservoir reaches 2.75× the
real cloud mass by t = 2400 s (`reference/FINDINGS_CONDENSATE_STAGE1.md` §1).

**Why not the spline positivity limiter instead.** It removes the reservoir too, and offline
the two are indistinguishable. Run, it costs a 250–1000× rise in entropy production
(2.86e4 → 7.06e6 at `POSITIVITY=ck`, → 2.98e7 at `=1`) with the entropy drift flipping from
+0.08 % to −2.6 %, because a per-step coefficient repair is not a physical process and the
budget cannot account for it; `max_w` goes 9.87 → 17.4 → 46.5. The transform is a change of
variables: nothing is ever repaired, and `n` is never modified ("n itself is untouched, so
that the effect of m adjustments does not accumulate in the predicted n" — Ooyama §4d).

**Measured in the model (2026-07-30, quick O01, 3600 s).** `min_rho_c_gm3` is 0 exactly, with
no limiter and nothing repaired, and all five o01 target windows PASS (`max_w` 11.17,
`peak_rain_rate` 78.67, `accum_rainfall_mm` 2.167, `max_rho_r_gm3` 12.65, `rain_onset_min` 24).
Against the untransformed baseline: entropy production over the physical points 0.409 vs 0.520
(−21 %), `energy_drift_pct` 0.153 vs 0.178, `water_mass_drift_pct` −3.55 vs −4.09,
`min_rho_d_frac` 0.881 vs 0.884. For contrast, enforcing the SAME constraint with the spline
limiter gives `max_w` 46.52, `min_rho_d_frac` 0.546 and an entropy production of 203.2 —
390x the baseline. The constraint is affordable; the limiter's way of imposing it is not.

The cost is in the water partition, and it is an UNMASKING rather than a new error. `min_rho_v`
goes −0.724 → −1.199 and the negative-vapor count 750 → 4181. But under either enforcement the
vapor deficit collapses onto the TOTAL-WATER deficit (`min_rho_v` − `min_rho_w`: −0.059 here,
−0.034 under POSITIVITY=1, against **+0.353** in the baseline). The baseline's better-looking
vapor was the negative cloud reservoir cancelling part of the `rho_t` − `rho_d` deficit. That
deficit is a difference of two independently fitted fields, which `benchmarks/FUTURE_WORK.md`
already records as not expressible as a bound on either — no condensate scheme can fix it.

**Known costs, both measured.** `f` is convex near zero, so ringing rectifies into mass
(+1.077 over 3600 s on the probe's transport arm, decelerating, and 3 % of the source-driven
mass where a real source dominates); and a point whose `n` has ratcheted below zero carries a
NUCLEATION LAG of `|n|/(J·S)` before cloud reappears — ~57 s at the worst measured excursion,
against ~128 s for the untransformed field, which spends that time carrying a −54 K anomaly
instead of reading zero cloud.
"""
@inline function condensate_transform_mode(options)
    mode = get(options, :condensate_transform, :none)::Symbol
    (mode === :none || mode === :bhyp || mode === :bhyp_smooth) ||
        error("options[:condensate_transform] = :$(mode) is not recognized; use :none " *
              "(the default: slot 9 is the cloud density), :bhyp (Ooyama's biased " *
              "hyperbolic control variable with his quasi-inverse) or :bhyp_smooth " *
              "(the same forward map with the strict C^inf inverse)")
    return mode
end

"""
    rain_transform_mode(options) -> Symbol

Which control variable SLOT 8 carries, the rain analogue of
[`condensate_transform_mode`](@ref). `:none` (the default, and the state of every
configuration that does not set the key) means the slot IS the rain density.

**Why rain gets its own key.** Cloud was transformed first and alone, deliberately, so that a
rain failure could not be confused with a cloud failure. With cloud settled the same treatment
is right for rain, but the two knobs stay separate so `:bhyp`/`:none` and `:bhyp`/`:bhyp` remain
separable arms rather than an assumption.

**What rain needs that cloud did not.** Cloud has no sedimentation. Rain's dominant sink is the
flux divergence `-∂F_r/∂z`, and under the transform slot 8's tendency carries it inside the
Jacobian while `rho_t` and `E_t` continue to receive the untransformed `-∂F_r/∂z` — they are
still densities and energies. That is correct term by term, but it does mean slot 8 and slot 3
no longer receive the identical discrete number, so the exact telescoping between them is
weakened. The diagnostic for it already exists and is independent: `accum_rainfall_mm` comes
from the `rho_t - rho_d` water path while `accum_rainfall_flux_mm` comes from the `rho_r`
surface flux, and the two must agree as well under the transform as without it.

**Why the limiter is not the answer for rain either, despite working.** `GridParameters.positivity`
on `rho_r` is exact and free on a single grid — `min 0.0` with `bound_shortfall` 0 at every
output time of every run. It is not available on a NESTED run: a child patch's i-boundary is
R3X, which `set_lower_bound!` rejects, so `src/nesting.jl` gives children a k-only bound and
warns about the leak that admits. The transform has no such restriction, which is the design
reason to move rain onto it.
"""
@inline function rain_transform_mode(options)
    mode = get(options, :rain_transform, :none)::Symbol
    (mode === :none || mode === :bhyp || mode === :bhyp_smooth) ||
        error("options[:rain_transform] = :$(mode) is not recognized; use :none " *
              "(the default: slot 8 is the rain density), :bhyp (Ooyama's biased " *
              "hyperbolic control variable with his quasi-inverse) or :bhyp_smooth " *
              "(the same forward map with the strict C^inf inverse)")
    return mode
end

# ── Slot NAMES under a transform ──────────────────────────────────────────────
# A transformed slot no longer holds what its name says, and the name is what every consumer
# keys off: Springsteel builds the CSV/netCDF column headers straight from `GridParameters.vars`
# (`Springsteel/src/io.jl`), so the output file is the only thing a postprocessor sees. Keeping
# the name `rho_c` while the column held `bhyp(rho_c)` is exactly how `benchmarks/o01_movie.jl`
# came to render half-amplitude cloud with spurious negative lobes against a run whose own
# diagnostics recorded `min_rho_c = 0`. The transformed slots are therefore RENAMED, following
# Ooyama's own notation, in which `mu` is the mixing ratio and `nu` its transform. (`n` was
# considered and rejected: it reads as number concentration in a double-moment scheme.)
const MC_NU_ALIAS = Dict("rho_c" => "nu_c", "rho_r" => "nu_r")

"""
    condensate_var_name(options) -> String
    rain_var_name(options) -> String

The name slot 9 / slot 8 carries under the current options: `"rho_c"` / `"rho_r"` untransformed,
`"nu_c"` / `"nu_r"` under a transform. Use these — never a literal — when building the
`vars`, `BCL`/`BCR`/`BCB`/`BCT`, `l_q`, `positivity` or `spline_filter` dicts of a
`GridParameters`, all five of which are keyed by name.
"""
condensate_var_name(options) =
    condensate_transform_mode(options) === :none ? "rho_c" : "nu_c"
@doc (@doc condensate_var_name)
rain_var_name(options) = rain_transform_mode(options) === :none ? "rho_r" : "nu_r"

"""
    mc_var_names(options; cyl = false) -> Vector{String}

The ordered prognostic-slot names for the moist-compressible set under the current options —
[`MC_VARS`](@ref) (or `MC_VARS_CYL` for the 10-slot cylindrical/3-D variants) with slots 8 and
9 renamed by [`rain_var_name`](@ref) / [`condensate_var_name`](@ref). With no transform declared
this returns the canonical list unchanged, so every existing configuration is bit-identical.
"""
function mc_var_names(options; cyl::Bool = false)
    names = copy(cyl ? MC_VARS_CYL : MC_VARS)
    names[8] = rain_var_name(options)
    names[9] = condensate_var_name(options)
    return names
end

"""
    mc_slot(vars, role) -> Int

The slot index of a water species by ROLE rather than by name, accepting either the density
name or its transformed alias. `role` is `"rho_c"` or `"rho_r"`.

Every `vars["rho_c"]`-style lookup in the kernel goes through this, so that a transformed
configuration cannot produce a `KeyError` deep in a solver — and, more importantly, so that the
lookup can never silently *succeed* against the wrong convention.
"""
@inline function mc_slot(vars, role::AbstractString)
    haskey(vars, role) && return vars[role]
    alias = get(MC_NU_ALIAS, role, "")
    (alias != "" && haskey(vars, alias)) && return vars[alias]
    error("no slot named \"$role\"" * (alias == "" ? "" : " or \"$alias\"") *
          " in grid_params.vars (it has $(sort(collect(keys(vars))))). Build the variable " *
          "list with `Scythe.mc_var_names(options)` so the transformed slots are named " *
          "consistently.")
end

"""
    check_mc_var_names(model) -> Nothing

Refuse a configuration whose name-keyed `GridParameters` dicts disagree with the declared
transforms. A no-op when neither transform is on, so no existing configuration is affected.

This exists because the failure it catches is SILENT. `vars` is the only one of the five
name-keyed dicts whose miss is loud; `_resolve_spline_filter` returns `nothing` rather than
throwing, so a leftover `"rho_r" => NaturalBC()` under the rain transform would quietly leave
slot 8 on the default Neumann fit — which forces a zero flux derivative at the ground and traps
the falling rain at the surface instead of letting sedimentation carry it out of the domain.
Likewise a stale `l_q` key removes the water filter the O01 configuration depends on for
stability. Neither would raise anything; both would change the physics.
"""
function check_mc_var_names(model::ModelParameters)
    ctrans = condensate_transform_mode(model.options)
    rtrans = rain_transform_mode(model.options)
    (ctrans === :none && rtrans === :none) && return nothing

    gp = model.grid_params
    expected = Set(mc_var_names(model.options;
                                cyl = haskey(gp.vars, "v") && length(gp.vars) >= 10))
    stale = String[]
    ctrans === :none || push!(stale, "rho_c")
    rtrans === :none || push!(stale, "rho_r")

    for (label, d) in (("vars", gp.vars), ("BCL", gp.BCL), ("BCR", gp.BCR),
                       ("BCB", gp.BCB), ("BCT", gp.BCT), ("l_q", gp.l_q),
                       ("positivity", gp.positivity),
                       ("spline_filter", gp.spline_filter))
        for key in keys(d)
            key == "default" && continue
            key in expected && continue
            hint = key in stale ?
                " — that species is transformed, so the key must be " *
                "\"$(MC_NU_ALIAS[key])\"" : ""
            error("grid_params.$label declares \"$key\", which is not a slot name of this " *
                  "configuration$(hint). Expected names: $(sort(collect(expected))). " *
                  "Build every name-keyed dict from `Scythe.mc_var_names(options)`; a stale " *
                  "key in $label would be ignored silently rather than raising.")
        end
    end
    return nothing
end

"""
    condensate_floor_mode(options) -> Bool

Whether `rho_liq` is floored at the DIAGNOSTIC INTERFACE. `true` for
`options[:condensate_floor] = :diagnostic`; `false` for `:none`, which is the default and the
state of every configuration that does not set the key.

**What the floor is, and what it is not.** The spline undershoot puts `rho_c` a few `1e-4`
below zero on the flanks of a cloud. The raw value then enters the closed-form temperature
retrieval, where `∂T/∂ρ_liq = L_v/D` (Eq. of [`retrieve_temperature`](@ref)) turns it into a
cold anomaly reaching −54 K on O01, and through `rho_vs(T)` into a collapsed saturation
density, false supersaturation and false nucleation. `:diagnostic` replaces `rho_liq` by
`max(rho_c, 0) + max(rho_r, 0)` at the retrieval, `q_l` (hence `C_vt`/`R_m`/`gamma_m`),
`Q_s_energy`, the entropy and the sedimentation energy — and NOWHERE else.

It is **not** [`clamp_water!`](@ref). Nothing is written back to a prognostic slot, so it is
memoryless, converts no mass, and cannot pump latent heat: the continuity terms, the rate
budgets and the state itself keep reading the raw value. `clamp_water!` rectifies a two-signed
oscillation one step at a time and takes O01 non-finite in 23 min; this changes what the
thermodynamics READS on the step it reads it, and leaves no trace.

What it does NOT fix: the negative condensate is still there, still in continuity, and still
accumulating (`reference/FINDINGS_CONDENSATE_STAGE1.md` §1 — the reservoir reaches 2.75x the
real cloud mass). This addresses the consequence, not the cause. The cause needs the
control-variable transform, [`condensate_transform_mode`](@ref).

**Measured (2026-07-30, quick O01, 3600 s).** The win is where the cold anomalies live:

| diagnostic | NOPRECIP `:none` | NOPRECIP `:diagnostic` | precip `:none` | precip `:diagnostic` |
|---|---|---|---|---|
| `min_rho_v_gm3` | −0.186 | **−1.58e-4** | −0.724 | −0.759 |
| `neg_rho_v_points` | 2033 | **0** | 750 | 716 |
| `entropy_prod_rate` (over rho_v>0) | 34.81 | 44.91 | 0.520 | 0.470 |
| `max_w` | 6.600 | 6.085 | 9.869 | 11.15 |
| `peak_rain_rate_gm2s` | — | — | 59.94 | 68.64 |
| `water_mass_drift_pct` | 1.07e-6 | 1.08e-6 | −4.093 | −3.674 |

On the cloud-only storm the floor removes NEGATIVE VAPOR ENTIRELY — 2033 points to zero, the
worst value improving 1180x — which is the mechanism confirmed: a negative liquid density
drives `rho_vs` down through `T`, manufacturing supersaturation, and the vapor residual is
where that lands. With precipitation active the effect is smaller and mixed (`max_w` +13 %,
rain rate +15 %, water and energy drift both improved), because rain removes the condensate
before the lobes grow as large.

**Correction (2026-07-30).** An earlier version of this note claimed the irreversible entropy
production fell 14.5x here. It did not. `mc_entropy_production` clamped `H` at `1e-12` where
the vapor is negative, so its raw value tracked the NEGATIVE-VAPOR COUNT rather than the
irreversibility: 94.7 % of the NOPRECIP total came from clamped points. Restricted to the
points where the quantity exists, the floor RAISES production slightly (34.81 → 44.91) while
eliminating every excluded point. The diagnostic now returns that count beside the rate so the
two cannot be read apart again.

Under a working [`condensate_transform_mode`](@ref) this floor should become INACTIVE — the
recovered density is already non-negative — so a transformed run is the test of whether the
transform subsumes it.
"""
@inline function condensate_floor_mode(options)
    mode = get(options, :condensate_floor, :none)::Symbol
    (mode === :none || mode === :diagnostic) ||
        error("options[:condensate_floor] = :$(mode) is not recognized; use :none " *
              "(the default: rho_liq is read raw everywhere) or :diagnostic (floor " *
              "rho_liq at the thermodynamic interface only, leaving state and continuity raw)")
    return mode === :diagnostic
end

"""
    install_positivity_bounds!(grid, ref_state, model) -> Nothing

Convert the declared physical positivity bounds into the reference-aware coefficient bounds
each spline leg actually needs, overriding what the factory installed.

The factory can only express a CONSTANT bound, which is exactly right for a total (`rho_r`,
bound 0) and for a perturbation whose reference is identically zero (`rho_c` on a cloud-free
base, which is every current configuration). It cannot express the bound for a perturbation
against a nonzero reference, and that bound differs by leg:

- **k-leg** — fits the field itself, so the bound is `-ρ̄(z)`, applied through the
  support-minimum rule (`set_lower_bound_from_profile!`).
- **i-leg** — fits the k-direction B-coefficients `⟨φ_z, u⟩`, so the bound is the CONSTANT
  `-⟨φ_z, ρ̄⟩` for each mode `z`, i.e. the negated SB coefficients of the reference. This is
  the scaling the RiRk factory refuses to guess at.

A no-op when the reference is zero, so the common path is untouched.
"""
function install_positivity_bounds!(grid, ref_state, model::ModelParameters)

    pos = model.grid_params.positivity
    # The transform supersedes the limiter for the species it acts on: a box constraint on
    # the CONTROL variable's coefficients would bound nu, not the density, and the two are not
    # the same constraint. Declaring both is a configuration error, not something to resolve
    # silently by precedence. Checked per species, since the two transforms are independent
    # knobs — a run may transform cloud while rain still takes the coefficient bound, which is
    # the arm that showed the cloud transform did no harm.
    for (name, opt, mode) in (("rho_c", :condensate_transform,
                               condensate_transform_mode(model.options)),
                              ("rho_r", :rain_transform,
                               rain_transform_mode(model.options)))
        (mode !== :none && haskey(pos, name)) &&
            error("positivity is declared for \"$name\" while options[:$opt] is on. The " *
                  "transform makes the recovered density non-negative by construction; a " *
                  "coefficient bound would constrain the control variable instead, which is " *
                  "a different constraint. Drop \"$name\" from positivity.")
    end
    isempty(pos) && return nothing
    grid.kbasis isa Springsteel.SplineBasisArray || return nothing
    vars = model.grid_params.vars

    for (name, v) in vars
        spec = Springsteel._resolve_spline_filter(pos, name, :k)
        spec_i = Springsteel._resolve_spline_filter(pos, name, :i)
        (spec === nothing && spec_i === nothing) && continue
        prof = positivity_reference_profile(name, ref_state)
        prof === nothing && continue                 # total: the factory bound is correct
        all(iszero, prof) && continue                # zero reference: likewise

        if spec !== nothing
            spec == 0.0 || error("positivity[\"$name\"][:k] = $spec: only a bound of 0.0 is " *
                                 "supported for a variable carried against a reference.")
            set_lower_bound_from_profile!(grid.kbasis.data[v], prof)
        end
        if spec_i !== nothing
            spec_i == 0.0 || error("positivity[\"$name\"][:i] = $spec_i: only 0.0 is supported.")
            # b̄_z = ⟨φ_z, ρ̄⟩ on the k basis. Safe to borrow the k-column's buffers: this
            # runs at initialization, before any transform.
            kcol = grid.kbasis.data[v]
            kcol.uMish .= prof
            SBtransform!(kcol)
            bbar = copy(kcol.b)
            b_iDim = model.grid_params.b_iDim
            for z in 1:model.grid_params.b_kDim
                set_lower_bound!(grid.ibasis.data[z, v], fill(-bbar[z], b_iDim))
            end
        end
    end
    return nothing
end

"""
    check_condensate_transform_ic(ref_state, model) -> Nothing

Warn once when a transformed run starts on a CLOUDY reference state.

Slot 9 then carries `n' = bhyp(rho_c) - bhyp(ρ̄_c)`, and every producer of an initial
condition has to know that: [`condensate_slot`](@ref) is the conversion, and the `*_mc!`
initializers take `condensate_transform`/`condensate_mu` keywords for it. On a cloud-free
reference (`ρ̄_c ≡ 0`, which is every benchmark configuration to date) the conversion of a
cloud-free state is exactly `0.0` under both conventions, so nothing has to be threaded and
nothing can go wrong. On a cloudy reference an initial condition written in DENSITY would be
silently reinterpreted as a control variable — off by a factor of two in the linear regime,
and not detectable from the run.

It is a warning and not an error because a cloudy reference is a legitimate configuration --
`bf02_moist` is one, and it threads the conversion -- and because the condition cannot be
checked from the state: both conventions give exactly 0.0 where there is no cloud, and where
there is cloud neither is distinguishable from the other without knowing the intended density.
Refusing outright would block correct configurations to catch a mistake it cannot actually
detect; naming the requirement is the most the model can honestly do.
"""
function check_condensate_transform_ic(ref_state, model::ModelParameters)
    uses_pressure_reference(model.equation_set) || return nothing
    condensate_transform_mode(model.options) === :none && return nothing
    prof = Springsteel.ref_rho_c(ref_state)
    prof isa Number && return nothing
    all(iszero, view(prof, :, 1)) && return nothing
    @warn """options[:condensate_transform] is on and the reference state is CLOUDY
      (ρ̄_c is not identically zero). Slot 9 therefore carries bhyp(rho_c) - bhyp(ρ̄_c), NOT
      the density perturbation, and the initial condition must have been written that way:
      `condensate_slot` is the conversion and the `*_mc!` initializers take
      `condensate_transform` / `condensate_mu` keywords for it. An initial condition written
      in density is read here as a control variable and is wrong by roughly a factor of two
      in the linear regime, with nothing in the run to show it.

      This cannot be verified from the state -- both conventions give exactly 0.0 where
      there is no cloud, and neither is distinguishable from the other where there is. It is
      the configuration's responsibility. In-tree, `bf02_moist.jl` threads it; O01 and the
      TC configurations have cloud-free references and are unaffected."""
    return nothing
end

"""
    _warn_unbounded_master_output(ref_state, model)

Warn once when a reference-aware bound is in force on the workers but cannot be installed on
the master's patch.

The master builds its patch from `model.grid_params` alone (`initialize_model`), never
constructing a reference state — only `createModelTile` does, per worker. So when a bounded
perturbation has a NONZERO reference, the master's `gridTransform!` (output/CFL/restart
cadence only) reconstructs without the offset bound. The dynamics are unaffected: the state's
coefficients are bounded on the workers and it is those that are integrated. Only the written
diagnostics can show an undershoot the model itself never saw.

Silent for every current configuration, all of which have `ρ̄_c ≡ 0`.
"""
function _warn_unbounded_master_output(ref_state, model::ModelParameters)
    pos = model.grid_params.positivity
    isempty(pos) && return nothing
    for (name, _) in model.grid_params.vars
        Springsteel._resolve_spline_filter(pos, name, :k) === nothing &&
            Springsteel._resolve_spline_filter(pos, name, :i) === nothing && continue
        prof = positivity_reference_profile(name, ref_state)
        (prof === nothing || all(iszero, prof)) && continue
        @warn """Positivity bound on "$name" is reference-offset (ρ̄ is not identically zero).
          The worker tiles and patches carry the correct bound, so the INTEGRATED state is
          bounded. The master's patch is built without a reference state, so the output
          written by gridTransform! is reconstructed unbounded and may show an undershoot the
          model never integrated. See install_positivity_bounds!.""" maxlog = 1
    end
    return nothing
end

"""
    clamp_water!(mtile, colstart, colend)

MEASURE the negative water in one column, and — only under
`options[:clamp_water] = true` — floor it at zero.

# Why the measurement is the point

Negative condensate is unphysical, so the instinct to clamp it is right. But a floor
cannot manufacture resolution, and at this scheme's ringing amplitudes it does real
damage.

`ρ_c` and `ρ_r` are positive-definite fields with sharp, spiky structure — a rain shaft
or a cloud core. A cubic B-spline column with too few nodes to resolve that spike
UNDERSHOOTS on its flanks, and the undershoot scales with how badly the spike is
under-resolved, not with roundoff. Flooring it then rectifies a zero-mean oscillation:
only the negative lobes are touched, so each step converts `δ` of vapor to liquid and the
retrieval faithfully releases `L_v·δ` of latent heat, one-signed and accumulating.

The size of that kick follows from differentiating the closed-form retrieval
([`retrieve_temperature`](@ref)) — `∂T/∂ρ_liq = L_v(T)/D` with
`D = C_factor − ρ_liq(C_pv − C_l)` — so flooring `δ` implies

    ΔT = L_v·δ / D          (`:worst_dT`, estimated with L_v0 and D ≈ C_factor)

Two measured regimes, four orders of magnitude apart:

- **Fit-level noise.** The balanced TC vortex moves ~5e-7 kg/m³ per column-step:
  `ΔT ~ 1e-3 K`. A floor there is free and harmless.
- **An unresolved spike.** `o01_rainfall`'s rain shafts reach `min(ρ_r) ≈ -1.8 g/m³` at
  500 m vertical spacing — `ΔT ≈ 4.4 K` in a SINGLE step at low levels, and ~68 K in the
  thin air aloft where it actually detonated. With the floor on, that run goes non-finite
  at t ≈ 23 min as convection erupts (`T = 9 K`, `E_t < 0`, `ρ_vs = Inf`, `Qdot = NaN`);
  with it off, the same run completes.

Re-accounting the floor does not rescue the second regime, it only chooses which budget
absorbs it. Holding `p` and subtracting `L_v(T)·δ` from `E_t` alongside the partition
change leaves `T` exactly invariant — but that is a `L_v·δ ≈ 4.5 kJ/m³` energy sink per
event, ~2 % of the local `E_t` per step. A temperature bias becomes an equal-sized
conservation bias. The amplitude is the problem; the bookkeeping is not.

So the default is to MEASURE and WARN rather than to floor, and to read a large
`worst_dT` as what it is: a request for more vertical nodes.

**Not flooring is not the same as harmless.** The `max(ρ_c, 0)` guards live inside the RATE
functions only ([`qss_condensation_rates`](@ref), autoconversion/collection), so an
undershoot cannot manufacture condensation or precipitation. But the raw value feeds the
continuity/advection terms AND `ρ_liq`, hence [`retrieve_temperature`](@ref) — so it drags
the temperature with it by the same `L_v·δ/D`. Measured on `o01_rainfall` quick, the
temperature attributable to negative liquid reaches **8.1 K** on the cloud flank at
t = 50 min. (The pre-`ρ_c` formulation could not do this: its `ρ_v` was clamped to
`[0, ρ_w − ρ_r]`, so the effective liquid the retrieval saw was `ρ_w − ρ_v ≥ ρ_r ≥ 0`.)

The difference between flooring and not is therefore the SIGN STRUCTURE, not the presence
of an error: unfloored, the anomaly oscillates with the ringing (cold on the undershoot,
warm on the overshoot) and is dispersive; floored, only the negative lobes are touched and
it becomes one-signed and cumulative. That is why one detonates and the other does not, and
why the honest remedy for both is resolution.

# The floor itself, when enabled

Two rules, in order, on the TOTALS (`ρ_c = ρ_c' + ρ̄_c`, so the perturbation floor is
`−ρ̄_c`):

1. `ρ_c, ρ_r ← max(·, 0)` — the deficit is borrowed from vapor.
2. If `ρ_c + ρ_r > ρ_w` (i.e. `ρ_v < 0`), take the excess out of `ρ_c` first, then `ρ_r`.

Mass and total water are exactly conserved either way: `ρ_t` is prognostic and the vapor
is the residual, so the floor only ever moves the partition. `E_t` is untouched, which is
precisely why the retrieval turns it into latent heat.

Applied to `var_np1` at the END of the column step (after the acoustic solve and the
vertical diffusion), so the values entering the patch-level fit are admissible.
"""
function clamp_water!(mtile::ModelTile, colstart::Int64, colend::Int64)

    vars = mtile.model.grid_params.vars
    rhod_i = vars["rho_d"]
    rhot_i = vars["rho_t"]
    rhor_i = mc_slot(vars, "rho_r")
    rhoc_i = mc_slot(vars, "rho_c")
    vnp1 = mtile.var_np1
    rho_dbar = view(ref_rho_d(mtile.ref_state), :, 1)
    rho_tbar = view(ref_rho_t(mtile.ref_state), :, 1)
    rho_cbar = view(Springsteel.ref_rho_c(mtile.ref_state), :, 1)
    apply = get(mtile.model.options, :clamp_water, false)::Bool
    # Slots 8 and 9 are not necessarily densities. Under a transform the measurement must
    # report the RECOVERED species — that is the number that says whether the transform is
    # doing its job in the model (it should never fall below -mu) — and the floor cannot be
    # applied at all, because flooring the control variable is a different operation from
    # flooring the density and would be a state repair of exactly the kind the transform
    # exists to avoid.
    ctrans = condensate_transform_mode(mtile.model.options)
    cmu = get(mtile.model.physical_params, :condensate_mu, 1.0e-7)
    rtrans = rain_transform_mode(mtile.model.options)
    rmu = get(mtile.model.physical_params, :rain_mu, 1.0e-7)
    if apply && (ctrans !== :none || rtrans !== :none)
        error("options[:clamp_water] with a water transform on (condensate_transform = " *
              ":$(ctrans), rain_transform = :$(rtrans)): the transform already bounds the " *
              "recovered density below by -mu, and flooring the control variable instead " *
              "would be a state repair. Drop one.")
    end

    moved = 0.0
    min_c = 0.0
    min_r = 0.0
    worst_dT = 0.0
    nneg = 0.0
    @inbounds for (k, i) in enumerate(colstart:colend)
        rho_c = recover_rho_c(vnp1[i, rhoc_i], rho_cbar[k], ctrans, cmu)
        rho_r = recover_rho_r(vnp1[i, rhor_i], rtrans, rmu)
        negative = (rho_c < 0.0) || (rho_r < 0.0)
        # The measurement fast-path: an admissible point costs two adds and a branch.
        (negative || apply) || continue

        rho_d = vnp1[i, rhod_i] + rho_dbar[k]
        rho_w = (vnp1[i, rhot_i] + rho_tbar[k]) - rho_d

        if negative
            deficit = max(-rho_c, 0.0) + max(-rho_r, 0.0)
            # Implied single-step latent-heat kick if this were floored
            # (dT/d rho_liq = L_v/D). L_v0 and D ~ C_factor are the cheap
            # conservative stand-ins; the point is the ORDER, which separates
            # 1e-3 K noise from a 4 K detonation.
            D = (rho_d * Cpd) + (rho_w * Cpv)
            moved += deficit
            nneg += 1.0
            min_c = min(min_c, rho_c)
            min_r = min(min_r, rho_r)
            worst_dT = max(worst_dT, L_v0 * deficit / D)
        end

        if apply
            # Rule 1: no negative water. Rule 2 runs REGARDLESS of rule 1 — a
            # condensate that exceeds the water present drives the residual vapor
            # negative, which is just as inadmissible and needs no negative input
            # to happen.
            rho_c = max(rho_c, 0.0)
            rho_r = max(rho_r, 0.0)
            excess = (rho_c + rho_r) - rho_w
            if excess > 0.0
                moved += excess
                take = min(excess, rho_c)
                rho_c -= take
                rho_r = max(rho_r - (excess - take), 0.0)
            end
            vnp1[i, rhoc_i] = rho_c - rho_cbar[k]
            vnp1[i, rhor_i] = rho_r
        end
    end

    if nneg > 0.0
        st = mtile.mc_water_stats
        tid = Threads.threadid()
        @inbounds begin
            st[1, tid] += moved
            st[2, tid] = min(st[2, tid], min_c)
            st[3, tid] = min(st[3, tid], min_r)
            st[4, tid] = max(st[4, tid], worst_dT)
            st[5, tid] += nneg
            # Per-STEP pre-fit minima for the production budget (reset each step by
            # `water_budget_trace`); rows 2/3 above are cumulative and cannot be differenced
            # against the post-fit reconstruction.
            st[MC_PRE_C, tid] = min(st[MC_PRE_C, tid], min_c)
            st[MC_PRE_R, tid] = min(st[MC_PRE_R, tid], min_r)
        end
    end
    return nothing
end

"""
    water_negativity_report(mtile) -> NamedTuple

Reduce `ModelTile.mc_water_stats` across threads: `(total, min_c, min_r, worst_dT,
count)`. `total` is Σ|negative water| in kg/m³ summed over gridpoints and steps; `min_c`
and `min_r` are the most negative condensate and rain densities the tile has held; and
`worst_dT` [K] is the largest `L_v·δ/D`, which is simultaneously the temperature anomaly
the negative liquid imposes through the retrieval and the kick a floor would apply — the
number that says whether the vertical resolution can represent the condensate spike (see
[`clamp_water!`](@ref)). All zero on a run that never went negative.
"""
function water_negativity_report(mtile::ModelTile)

    st = mtile.mc_water_stats
    size(st, 2) == 0 && return (total = 0.0, min_c = 0.0, min_r = 0.0,
                                worst_dT = 0.0, count = 0.0)
    return (total = sum(view(st, 1, :)),
            min_c = minimum(view(st, 2, :)),
            min_r = minimum(view(st, 3, :)),
            worst_dT = maximum(view(st, 4, :)),
            count = sum(view(st, 5, :)))
end

"""
    water_negativity_trace(mtile, t)

Warn, once per doubling, when the negative water in the condensate fields implies a
latent-heat kick large enough to matter — i.e. when the vertical basis is failing to
represent a positive-definite spike.

Emitted from the single-threaded pre-column-loop slot next to
[`state_minima_trace`](@ref), so printing is race-free. The threshold ladder starts at
`options[:water_warn_dT]` (default 0.05 K, comfortably above the ~1e-3 K fit-level noise
of a quiet column) and each subsequent warning needs a doubling, so a run that is simply
under-resolved reports O(10) lines rather than one per step.

The message names the remedy. It is NOT a floor -- flooring converts the undershoot into
one-signed latent heating (or, re-accounted, into an equal-sized energy sink); see
[`clamp_water!`](@ref) for the measured failure. It is also no longer "more vertical
nodes", which this docstring used to say: convective width collapses with the grid, so the
spike stays at grid scale and the basis rings at a fixed RELATIVE amplitude. The remedy
differs by species, and both are now measured
(`reference/FINDINGS_CONDENSATE_STAGE1.md`):

  * `rho_r` -- the spline coefficient bound (`GridParameters.positivity`), which is exact
    and free here: rain carries no negative mass at any output time of any run.
  * `rho_c` -- the control-variable transform ([`condensate_transform_mode`](@ref)). The
    same coefficient bound applied to the cloud drives the O01 peak updraft from 9.9 to
    46.5 m/s and the entropy production to 390x baseline, because it must repair the state
    at the updraft on every step. The transform changes the variable instead and repairs
    nothing.
"""
function water_negativity_trace(mtile::ModelTile, t::Int64)

    st = mtile.mc_water_stats
    size(st, 2) == 0 && return nothing
    rep = water_negativity_report(mtile)
    rep.worst_dT > 0.0 || return nothing
    warn_dT = get(mtile.model.options, :water_warn_dT, 0.05)
    @inbounds level = st[6, 1]
    threshold = level == 0.0 ? warn_dT : 2.0 * level
    rep.worst_dT >= threshold || return nothing
    @inbounds st[6, 1] = rep.worst_dT

    applied = get(mtile.model.options, :clamp_water, false)::Bool
    @warn """Negative water: the vertical basis is undershooting a condensate spike.
      step $t: min rho_c = $(rep.min_c) kg/m^3, min rho_r = $(rep.min_r) kg/m^3
      |dT| from the negative liquid (= the kick if floored): $(rep.worst_dT) K
      $(rep.count) gridpoint-steps so far, $(rep.total) kg/m^3 total
      $(applied ? "options[:clamp_water] is ON, so that kick IS being applied, one-signed." :
                  "options[:clamp_water] is off, so this is the size of the COLD anomaly the negative liquid is currently imposing through the retrieval (oscillatory, not cumulative). Rate functions are guarded; the retrieval and advection are not.")
      GENERATOR (attributed 2026-07-26): the refit deposits a small undershoot every step
      (~0.06% of peak) and NOTHING REMOVES IT. Refined 2026-07-30: what makes it permanent
      is that the negative region has no SINK — every rate is max(rho,0)-guarded, so a
      negative point cannot evaporate, autoconvert or collect, while the positive overshoot
      IS consumed every step. On the shipped O01 run the reservoir reaches 2.75x the real
      cloud mass by t = 2400 s.
      REMEDY, by species: rho_r takes the spline coefficient bound
      (GridParameters.positivity, e.g. Dict("rho_r" => Dict(:k => 0.0))), which is exact and
      free. rho_c takes options[:condensate_transform] = :bhyp — the SAME bound applied to
      the cloud takes max_w 9.9 -> 46.5 and the entropy production to 390x, because it
      repairs the state at the updraft every step. Flooring rectifies a two-signed excursion
      into one-signed latent heating. See reference/FINDINGS_CONDENSATE_STAGE1.md."""
    return nothing
end

"""
    water_budget_probe!(mtile, offset, slot, t, colstart, val, adv, div, src, aut, autsign,
                        sed, w, z)

Record the term-by-term tendency at the gridpoint where one water species is most negative.

Called from `mc_driver!` immediately after that species' `expdot` is assembled, while its
`ADV`/source scratch is still live (the buffers are reused by the next slot). Writes into the
calling thread's column of `mc_water_stats`, keeping the worst point seen so far this step —
`water_budget_trace` resets the block after printing, so the semantics are per-step, unlike
rows 1-6.

`offset` is `MC_BUDGET_R` or `MC_BUDGET_C`. `autsign` is `+1` for rain and `-1` for cloud (the
same `AUTO_COLL` array is a source for one and a sink for the other); `sed` is `nothing` for
cloud, which has no sedimentation. All arrays are column-block-local (1-based over
`colstart:colend`).

This exists because rows 1-6 record cumulative extrema and so cannot say WHICH operation
drives the water negative — the question left open by
`reference/FINDINGS_NEGATIVE_WATER_ATTRIBUTION.md`'s CORRECTION section.

The worst point is selected on the **AB3-projected** value `ρ + Δ`, where `Δ` is the exact
combination [`explicit_timestep`](@ref) will apply at this step (Euler at `t = 1`, AB2 at
`t = 2`, AB3 thereafter) — NOT on the forward-Euler projection. That distinction is the whole
reason this probe was blind: every water depletion limiter is sized so that `ρ + ts·f_n ≥ 0`
exactly, so the Euler projection of a capped point is `0.0`, `0.0 < 0.0` is false, and the
probe skipped precisely the points that produce the negative. Both projections are recorded
(`:b_*_eul` alongside the selecting value) so their gap is readable directly.

`Δ` is read from `expdot`/`expdot_nm1`/`expdot_nm2` at `slot`, which is exact at the call site
for every configuration where nothing is added to that slot afterwards. The horizontal
water mixing (`Khdiff_water`/Smagorinsky) and the Louis boundary layer both add to slot 8/9
*after* this runs, so with either enabled the projection here understates them; the
[`water_depletion_probe!`](@ref) census, which runs immediately before `explicit_timestep`,
is exact in every configuration.
"""
@inline function water_budget_probe!(mtile::ModelTile, offset::Int64, slot::Int64, t::Int64,
                                     colstart::Int64,
                                     val, adv, div, src, aut, autsign::Float64, sed, w, z)

    st = mtile.mc_water_stats
    size(st, 2) == 0 && return nothing
    ts = mtile.model.ts
    expdot_n = mtile.expdot_n
    expdot_nm1 = mtile.expdot_nm1
    expdot_nm2 = mtile.expdot_nm2
    j = 0
    vmin = 0.0
    dmin = 0.0
    @inbounds for i in eachindex(val)
        g = colstart + i - 1
        delta = _ab3_increment(ts, t, expdot_n[g, slot], expdot_nm1[g, slot],
                               expdot_nm2[g, slot])
        projected = val[i] + delta
        if projected < vmin
            vmin = projected
            dmin = delta
            j = i
        end
    end
    j == 0 && return nothing
    tid = Threads.threadid()
    @inbounds begin
        vmin < st[offset, tid] || return nothing
        st[offset,      tid] = vmin
        st[offset +  1, tid] = adv[j]
        st[offset +  2, tid] = -val[j] * div[j]
        st[offset +  3, tid] = src[j]
        st[offset +  4, tid] = autsign * aut[j]
        st[offset +  5, tid] = sed === nothing ? 0.0 : -sed[j]
        st[offset +  6, tid] = div[j]
        st[offset +  7, tid] = w[j]
        st[offset +  8, tid] = z[j]
        st[offset +  9, tid] = val[j] + (ts * expdot_n[colstart + j - 1, slot])
        st[offset + 10, tid] = val[j]
    end
    return nothing
end

"""
    _ab3_increment(ts, t, f_n, f_nm1, f_nm2) -> Float64

The increment [`explicit_timestep`](@ref) applies to a prognostic slot at step `t`.

One definition, so neither a diagnostic nor a limiter can drift from the integrator it is sized
for: Euler at `t = 1`, second-order Adams-Bashforth at `t = 2`, AB3 (Durran & Blossey 2012)
thereafter. The leading AB3 weight is **23/12 ≈ 1.917**, which is what a sink capped at `-ρ/ts`
— a forward-Euler budget — used to get multiplied by. [`_ab3_sink_bound`](@ref) is this same
expression solved for the current level, and is how the water limiters are written now.
"""
@inline _ab3_increment(ts::Float64, t::Int64, f_n::Float64, f_nm1::Float64, f_nm2::Float64) =
    t == 1 ? ts * f_n :
    t == 2 ? (0.5 * ts) * ((3.0 * f_n) - f_nm1) :
             (ts / 12.0) * ((23.0 * f_n) - (16.0 * f_nm1) + (5.0 * f_nm2))

"""
    _ab3_sink_bound(ts, t, avail, s_nm1, s_nm2) -> Float64

The most negative CURRENT-level sink [`explicit_timestep`](@ref) can apply at step `t` without
carrying a species that has `avail ≥ 0` of itself below zero, given the two previous levels of
that species' sink. **Returned UNCLAMPED** — see the clamp note below.

This is [`_ab3_increment`](@ref) solved for `s_n`, one branch per integrator branch:

| `t` | increment | admissible `s_n` |
|---|---|---|
| 1 | `ts·s_n` | `−avail/ts` |
| 2 | `(ts/2)(3 s_n − s_nm1)` | `(−2·avail/ts + s_nm1)/3` |
| ≥3 | `(ts/12)(23 s_n − 16 s_nm1 + 5 s_nm2)` | `(−12·avail/ts + 16 s_nm1 − 5 s_nm2)/23` |

The `t = 1` branch is exactly the forward-Euler budget `−avail/ts` that every water limiter in
this file used to be written with, at every step — and that is the defect. AB3's leading weight
is 23/12, so a sink sitting on the Euler bound is applied at ≈1.92× and lands the species at
≈ `−0.92·avail`. Measured on the quick O01 at 2.34–2.82× the local cloud, regenerated at
100–190 fresh gridpoints on EVERY step from the first cloudy one onward (STAGE 3 of
reference/FINDINGS_NEGATIVE_WATER_ATTRIBUTION.md). A constant `12/23` removes most of it
(ACTUAL depletion fraction 2.345 → 1.223, STAGE 3b) but provably not all: `12/23` is only the
right factor when the sink has no history, and the residual 0.223 is the history term — which
is what the `t ≥ 3` branch above carries and no constant can.

**Scope: the MICROPHYSICS' own contribution.** By linearity of the AB3 operator, what the
phase changes contribute to `ρ^{n+1}` is `_ab3_increment` evaluated on the sink history alone,
so bounding that bounds what the physics removes — nothing more. Transport, the spectral refit
and the acoustic solve can still carry a point negative; that is the positivity limiter's job,
and deliberately not this one's. Making the microphysics surrender its sink to compensate a
transport-generated negative would be a floor in all but name, at exactly the points STAGE 3b
showed the caps have no purchase on.

**Callers must clamp.** A bound `> 0` means the two history levels alone already drive the
species negative, which limiting the current level cannot repair; forcing `s_n > 0` there would
manufacture mass, the rectification `clamp_water!` documents as the reason not to floor the
state. Every call site therefore applies `min(bound, 0.0)` (or `max` on the mirrored vapor
ceiling) and the raw value is kept so [`_depletion_census!`](@ref) can count how often the
clamp fires — a nonzero count is a real signal, not noise.
"""
@inline _ab3_sink_bound(ts::Float64, t::Int64, avail::Float64,
                        s_nm1::Float64, s_nm2::Float64) =
    t == 1 ? -avail / ts :
    t == 2 ? ((-2.0 * avail / ts) + s_nm1) / 3.0 :
             ((-12.0 * avail / ts) + (16.0 * s_nm1) - (5.0 * s_nm2)) / 23.0

"""
Columns of `ModelTile.mc_micro_n`/`mc_micro_nm1`/`mc_micro_nm2` — the per-gridpoint history of
each water channel's NET MICROPHYSICS tendency, kept so the depletion caps can be written
against the integrator's actual three-level combination ([`_ab3_sink_bound`](@ref)).

| column | channel | value |
|---|---|---|
| `MC_MICRO_C` | cloud | `Qdot − AUTO_COLL` |
| `MC_MICRO_R` | rain | `Qdot_r` (`AUTO_COLL` is a SOURCE for rain; omitting it only makes rain's budget more conservative) |
| `MC_MICRO_V` | vapor | `−(Qdot + Qdot_r)` — condensation is internal to the water, so the residual vapor pays exactly what the two condensate channels gain |

Rotated in lockstep with `expdot_n/nm1/nm2` by [`_rotate_micro_history!`](@ref), which runs
immediately after `explicit_timestep`. Zero-initialized, so step 1 sees no history and its
budget reduces to the forward-Euler one this code used to write everywhere.
"""
const MC_MICRO_C = 1
const MC_MICRO_R = 2
const MC_MICRO_V = 3
const MC_MICRO_N = 3

"""
    _rotate_micro_history!(mtile, colstart, colend)

Advance the microphysics sink history one level for this column, mirroring
[`explicit_timestep`](@ref)'s rotation of `expdot_*` so the next step's cap sees exactly the two
levels the integrator will weight.

Called from `mc_driver!` immediately after `explicit_timestep` — which is once per column per
step on BOTH paths, because the `exact_si` branch returns from `mc_driver!` further down, after
this. A second rotation would silently shift the whole history by one step and mis-size every
cap, so the placement is load-bearing.
"""
@inline function _rotate_micro_history!(mtile::ModelTile, colstart::Int64, colend::Int64)
    n = mtile.mc_micro_n
    nm1 = mtile.mc_micro_nm1
    nm2 = mtile.mc_micro_nm2
    size(n, 2) == 0 && return nothing
    # Unconditional, unlike `explicit_timestep`'s `t == 1` branch which skips `nm2 .= nm1`:
    # both levels are zero at t = 1, so the two are identical and the branch is not worth it.
    @inbounds for c in 1:MC_MICRO_N, g in colstart:colend
        nm2[g, c] = nm1[g, c]
        nm1[g, c] = n[g, c]
    end
    return nothing
end

"""
    water_depletion_probe!(mtile, colstart, colend, t, precipitation,
                           rho_c, rho_r, rho_v, res_rho_t, Qdot, Qdot_r, AUTO_COLL,
                           cap_c, cap_r, cap_v)

Census, over every gridpoint holding water, of how hard the step is depleting it.

Runs immediately before [`explicit_timestep`](@ref), so `expdot` is final for the step and the
measured increment is the one actually applied. For each channel it records how many points
exist, how many have a sink pinned at its bound, the largest forward-Euler and largest ACTUAL
depletion fractions, how many points the step drives negative outright, how many exceed AB3's
real-axis stability limit with no cap active, and how many carry an inadmissible history.

The hypotheses this is built to separate:

- **cap-vs-integrator** — under `water_cap_mode = :euler` the limiters guarantee only
  `ρ + ts·f_n ≥ 0`, so `:d_*_eul` is pinned at 1.0 at capped points while `:d_*_ab3` runs up to
  ≈ 23/12 and `:d_*_neg` tracks `:d_*_cevap` + `:d_*_cauto`. Under `:ab3` it is `:d_*_ab3` that
  the bound holds at 1, and any remaining `:d_*_neg` is transport, not microphysics;
- **stiff source** — `:d_*_stiff` is nonzero, i.e. points are being depleted faster than AB3
  can stably integrate even before any cap engages;
- **inadmissible history** — `:d_*_infeas` is nonzero, i.e. the previous two sink levels alone
  carry the point negative, which no bound on the current level can repair.

The VAPOR is censused alongside the two condensates. It is a residual, not a slot, so its
tendency is assembled from the four densities' slots and its sink is the condensation that
removes it. It has no bound of its own anywhere else in the model, which is exactly why it
needs measuring.

`rho_v` is the vapor the step actually used (the BLENDED one under
`options[:vapor_retrieval] = :blend`) and `res_rho_t` the density-budget residual; their
maximum absolute difference is recorded as [`MC_VAPOR_GAP`](@ref MC_VAPOR_GAP), the partition
gap. It is identically zero under `:residual`.

Gated on `options[:water_budget_trace]`; never called otherwise.
"""
function water_depletion_probe!(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64,
                                precipitation::Bool,
                                rho_c, rho_r, rho_v, res_rho_t, Qdot, Qdot_r, AUTO_COLL,
                                cap_c, cap_r, cap_v)

    size(mtile.mc_water_stats, 2) == 0 && return nothing
    _depletion_census!(mtile, MC_DEPLETION_C, 9, MC_MICRO_C, colstart, t, rho_c, Qdot,
                       precipitation ? AUTO_COLL : nothing, Qdot, cap_c)
    _depletion_census!(mtile, MC_DEPLETION_R, 8, MC_MICRO_R, colstart, t, rho_r, Qdot_r,
                       nothing, Qdot, cap_r)
    _vapor_census!(mtile, colstart, t, rho_v, Qdot, Qdot_r, cap_v)
    _vapor_gap_census!(mtile, rho_v, res_rho_t)
    return nothing
end

"""
    _vapor_gap_census!(mtile, rho_v, res_rho_t)

Record `max|ρ_v − res_rho_t|` over this column into [`MC_VAPOR_GAP`](@ref MC_VAPOR_GAP).

The blend is a PARTITION, so `ρ_d + ρ_v + ρ_c + ρ_r` no longer sums to `ρ_t` pointwise; this is
the size of that disagreement. Where the correction saturator is inactive it equals
`w·τ_qss·|QSSREL|` (`res_qss − res_rho_t = −τ·QSSREL` exactly — see [`qss_relaxation`](@ref)),
and everywhere it is bounded by `dcap·ρ_vs`, which is the guarantee worth checking: the
identity relates the gap to the reconciliation RATE and bounds neither, and that
misreading is what let the uncapped blend detonate the NOPRECIP storm (see
[`vapor_retrieval_blend`](@ref)). Offline on the measured snapshots the uncapped gap reaches
9.4e−4 kg/m³ in cloud, 4.3 % of the local vapor at the worst point.

Under `options[:vapor_retrieval] = :residual` the two arrays hold identical doubles and this
records exactly `0.0`, which is the cheapest possible confirmation that the option is inert.
"""
@inline function _vapor_gap_census!(mtile::ModelTile, rho_v, res_rho_t)

    st = mtile.mc_water_stats
    gap = 0.0
    @inbounds for i in eachindex(rho_v)
        d = abs(rho_v[i] - res_rho_t[i])
        d > gap && (gap = d)
    end
    tid = Threads.threadid()
    @inbounds st[MC_VAPOR_GAP, tid] = max(st[MC_VAPOR_GAP, tid], gap)
    return nothing
end

"""
    _depletion_census!(mtile, species, slot, channel, colstart, t, val, evap, aut, Qdot, cap)

One species' pass for [`water_depletion_probe!`](@ref). Split out so each call specializes on
its own argument types (`rho_c` is a scratch `Vector`, `rho_r` a grid `SubArray`), rather than
being iterated as a heterogeneous tuple. `aut === nothing` skips the `AUTO_COLL` cap test,
which is the correct behaviour for rain (where `AUTO_COLL` is a source, not a sink) and with
precipitation off.

`cap` is the RAW (unclamped) [`_ab3_sink_bound`](@ref) vector this step's rates were limited
with. Reading the same array the limiter read makes the cap test bitwise exact under either
`water_cap_mode` and makes `:d_*_infeas` (`cap > 0`, the clamp fired) directly countable.
`channel` is this species' `MC_MICRO_*` column, which gives the microphysics-only fraction the
bound actually governs.
"""
function _depletion_census!(mtile::ModelTile, species::Int64, slot::Int64, channel::Int64,
                            colstart::Int64, t::Int64, val, evap, aut, Qdot, cap)

    st = mtile.mc_water_stats
    ts = mtile.model.ts
    expdot_n = mtile.expdot_n
    expdot_nm1 = mtile.expdot_nm1
    expdot_nm2 = mtile.expdot_nm2
    micro_n = mtile.mc_micro_n
    micro_nm1 = mtile.mc_micro_nm1
    micro_nm2 = mtile.mc_micro_nm2
    # AB3's real-axis absolute-stability interval is (-0.545, 0]; a decay faster than that is
    # unstable under this integrator no matter how the sink is limited.
    AB3_REAL_LIMIT = 0.545

    n = 0.0; n_cevap = 0.0; n_cauto = 0.0; n_neg = 0.0; n_stiff = 0.0; n_infeas = 0.0
    n_mneg = 0.0
    max_eul = 0.0; max_ab3 = 0.0; max_mab3 = 0.0
    @inbounds for i in eachindex(val)
        rho = val[i]
        rho > 0.0 || continue
        n += 1.0
        g = colstart + i - 1
        f_n = expdot_n[g, slot]
        delta = _ab3_increment(ts, t, f_n, expdot_nm1[g, slot], expdot_nm2[g, slot])
        dep_eul = -(ts * f_n) / rho
        dep_ab3 = -delta / rho
        dep_eul > max_eul && (max_eul = dep_eul)
        dep_ab3 > max_ab3 && (max_ab3 = dep_ab3)
        dep_ab3 > 1.0 && (n_neg += 1.0)
        # The MICROPHYSICS' own share of the same increment — the part `_ab3_sink_bound`
        # governs, and the only fraction the depletion mode can be judged by.
        dep_mab3 = -_ab3_increment(ts, t, micro_n[g, channel], micro_nm1[g, channel],
                                   micro_nm2[g, channel]) / rho
        dep_mab3 > max_mab3 && (max_mab3 = dep_mab3)
        # Tolerance, not `> 1.0`: a point sitting exactly ON the bound lands at 1 + O(eps)
        # (the bound divides by 23, so it cannot be exact), and counting those as violations
        # would report every capped point. 1e-9 is ~1e7 ULP above that and ~1e9 below the
        # 23/12 the `:euler` mode produces, so it separates the two without ambiguity.
        dep_mab3 > MICRO_DEPLETION_TOL && (n_mneg += 1.0)
        # Cap detection is bitwise-exact against the SAME array the limiter read:
        # `max(x, floor)` returns the floor itself when it clips, and
        # `min(auto + coll, avail)` returns `avail`.
        raw = cap[i]
        raw > 0.0 && (n_infeas += 1.0)
        floor_i = min(raw, 0.0)
        capped = evap[i] == floor_i
        capped && (n_cevap += 1.0)
        if aut !== nothing
            avail = max(Qdot[i] - floor_i, 0.0)
            if avail > 0.0 && aut[i] == avail
                n_cauto += 1.0
                capped = true
            end
        end
        (!capped && dep_eul > AB3_REAL_LIMIT) && (n_stiff += 1.0)
    end

    _census_reduce!(st, species, n, n_cevap, n_cauto, max_eul, max_ab3, n_neg, n_stiff,
                    n_infeas, max_mab3, n_mneg)
    return nothing
end

"""Common per-thread accumulation for the depletion census blocks (same layout for all three)."""
@inline function _census_reduce!(st, species::Int64, n, n_cevap, n_cauto, max_eul, max_ab3,
                                 n_neg, n_stiff, n_infeas, max_mab3, n_mneg)
    tid = Threads.threadid()
    @inbounds begin
        st[species,     tid] += n
        st[species + 1, tid] += n_cevap
        st[species + 2, tid] += n_cauto
        st[species + 3, tid] = max(st[species + 3, tid], max_eul)
        st[species + 4, tid] = max(st[species + 4, tid], max_ab3)
        st[species + 5, tid] += n_neg
        st[species + 6, tid] += n_stiff
        st[species + 7, tid] += n_infeas
        st[species + 8, tid] = max(st[species + 8, tid], max_mab3)
        st[species + 9, tid] += n_mneg
    end
    return nothing
end

"""
    _vapor_census!(mtile, colstart, t, rho_v, Qdot, Qdot_r, cap)

The residual vapor's pass for [`water_depletion_probe!`](@ref), in the same layout as
[`_depletion_census!`](@ref)'s two condensate blocks.

The vapor is not a prognostic slot, so its tendency has to be assembled from the four
densities that define it, `f_v = f_3 - f_2 - f_8 - f_9`, at each of the three time levels.
That assembly is exact: all four use the same advective product-rule form with the same
pointwise divergence, and both `-v·∇ρ` and `-ρ∇·v` are linear in ρ, so the residual of the
tendencies IS the tendency of the residual. The sedimentation flux divergence cancels between
slots 3 and 8 by construction, and `AUTO_COLL` between 8 and 9, leaving condensation as the
vapor's only microphysical sink — which is what `cap` bounds.

What this census will NOT see is anything applied outside `expdot`: the acoustic solve moves
`rho_t` and `rho_d` but neither condensate, and the positivity limiter moves the condensates
but not `rho_t`. Both land wholly in the vapor and both are measured elsewhere.
"""
function _vapor_census!(mtile::ModelTile, colstart::Int64, t::Int64, rho_v, Qdot, Qdot_r, cap)

    st = mtile.mc_water_stats
    ts = mtile.model.ts
    e_n = mtile.expdot_n
    e_1 = mtile.expdot_nm1
    e_2 = mtile.expdot_nm2
    AB3_REAL_LIMIT = 0.545

    micro_n = mtile.mc_micro_n
    micro_nm1 = mtile.mc_micro_nm1
    micro_nm2 = mtile.mc_micro_nm2

    n = 0.0; n_cap = 0.0; n_neg = 0.0; n_stiff = 0.0; n_infeas = 0.0; n_mneg = 0.0
    max_eul = 0.0; max_ab3 = 0.0; max_mab3 = 0.0
    @inbounds for i in eachindex(rho_v)
        rho = rho_v[i]
        rho > 0.0 || continue
        n += 1.0
        g = colstart + i - 1
        f_n = e_n[g, 3] - e_n[g, 2] - e_n[g, 8] - e_n[g, 9]
        f_1 = e_1[g, 3] - e_1[g, 2] - e_1[g, 8] - e_1[g, 9]
        f_2 = e_2[g, 3] - e_2[g, 2] - e_2[g, 8] - e_2[g, 9]
        delta = _ab3_increment(ts, t, f_n, f_1, f_2)
        dep_eul = -(ts * f_n) / rho
        dep_ab3 = -delta / rho
        dep_eul > max_eul && (max_eul = dep_eul)
        dep_ab3 > max_ab3 && (max_ab3 = dep_ab3)
        dep_ab3 > 1.0 && (n_neg += 1.0)
        dep_mab3 = -_ab3_increment(ts, t, micro_n[g, MC_MICRO_V], micro_nm1[g, MC_MICRO_V],
                                   micro_nm2[g, MC_MICRO_V]) / rho
        dep_mab3 > max_mab3 && (max_mab3 = dep_mab3)
        # Tolerance, not `> 1.0`: a point sitting exactly ON the bound lands at 1 + O(eps)
        # (the bound divides by 23, so it cannot be exact), and counting those as violations
        # would report every capped point. 1e-9 is ~1e7 ULP above that and ~1e9 below the
        # 23/12 the `:euler` mode produces, so it separates the two without ambiguity.
        dep_mab3 > MICRO_DEPLETION_TOL && (n_mneg += 1.0)
        raw = cap[i]
        raw > 0.0 && (n_infeas += 1.0)
        # The ceiling is the negated bound. The single-channel path hits it exactly
        # (`min(Qdot_c, ceil_v)`); the two-channel rescale lands within rounding of it, so
        # this is `>=` rather than `==` and is honest about that.
        ceil_i = -min(raw, 0.0)
        capped = ceil_i > 0.0 && (Qdot[i] + Qdot_r[i]) >= ceil_i
        capped && (n_cap += 1.0)
        (!capped && dep_eul > AB3_REAL_LIMIT) && (n_stiff += 1.0)
    end

    _census_reduce!(st, MC_DEPLETION_V, n, n_cap, 0.0, max_eul, max_ab3, n_neg, n_stiff,
                    n_infeas, max_mab3, n_mneg)
    return nothing
end

"""
    _ileg_shortfall(v) -> Float64

Accumulated positivity shortfall of variable slot `v`'s i-direction splines, or `NaN` when it
cannot be read.

The i-leg bound bites on the WORKER'S PATCH, not on the tile: every call site uses the 3-arg
`splineTransform!(sharedSpectral, patch, mtile.tile)` (`semiimplicit.jl:793`, `:968`,
`nesting.jl:740`, `:879`), which runs `SAtransform_bounded` on `patch.ibasis`. The tile's own
i-splines carry the same bound but are never the ones solved, so their shortfall is
identically zero and reporting it would be a lie of omission. `ModelTile` holds no reference
to the patch — it is a worker-scope binding — hence the introspection, which is confined to
this diagnostic and returns `NaN` (never a misleading `0.0`) whenever the lookup does not
find a real spline patch.
"""
function _ileg_shortfall(v::Int64)
    isdefined(Main, :patch) || return NaN
    p = getfield(Main, :patch)
    hasproperty(p, :ibasis) || return NaN
    ib = p.ibasis
    ib isa Springsteel.NoBasisArray && return 0.0
    eltype(ib.data) <: Springsteel.CubicBSpline.Spline1D || return NaN
    v <= size(ib.data, 2) || return NaN
    return sum(Springsteel.CubicBSpline.bound_shortfall(ib.data[z, v])
               for z in axes(ib.data, 1))
end

"""
    water_budget_trace(mtile, t)

Print the per-step water production budget, then reset it for the next step.

Emitted from the same single-threaded pre-column-loop slot as
[`water_negativity_trace`](@ref), so printing is race-free and the numbers describe the step
that just finished. Gated on `options[:water_budget_trace]::Int` — the print interval in steps;
absent or `0` disables it, and `water_budget_probe!` is then never called, so the hot path is
untouched.

Also reports the POST-reconstruction minimum from `tile.physical`, which is the step-matched
partner of `clamp_water!`'s pre-fit measurement. Their difference is what the fit contributes
per step, the quantity the previous attribution had no way to see (it read only the pre-fit
number, and only from the console, where the run's `@warn` never appears — the worker's stderr
goes to `<output_dir>/scythe_err.log`).
"""
function water_budget_trace(mtile::ModelTile, t::Int64)

    st = mtile.mc_water_stats
    size(st, 2) == 0 && return nothing
    interval = get(mtile.model.options, :water_budget_trace, 0)::Int
    interval > 0 || return nothing

    if mod(t, interval) == 0
        # Reduce across threads: the thread holding the most negative value owns the budget.
        vars = mtile.model.grid_params.vars
        # Positivity limiter health: nonzero shortfall means a column was infeasible (its
        # total mass below the minimum an admissible field can carry), so the limiter
        # created mass instead of redistributing it. This must stay at zero.
        #
        # PER SPECIES and per LEG. Summing over every variable hid which species was
        # infeasible, and reading only the k-basis hid the i-leg entirely — the two together
        # are why "bound_shortfall stays exactly 0.0" was recorded for a configuration whose
        # k-leg shortfall reached 8e3 (see the STAGE 2 section of
        # reference/FINDINGS_NEGATIVE_WATER_ATTRIBUTION.md).
        #
        # The arithmetic below reads slots 8 and 9 as DENSITIES. That is unconditionally
        # correct here because `mc_driver!` refuses `water_budget_trace` under either
        # transform — the probe's ADV column would be in control-variable units while its
        # source columns are densities, so the attribution would not close.
        kb = mtile.tile.kbasis
        kshort = (name) -> kb isa Springsteel.NoBasisArray ? 0.0 :
            Springsteel.CubicBSpline.bound_shortfall(kb.data[mc_slot(vars, name)])
        shortfall = join(("$name k=$(kshort(name)) i=$(_ileg_shortfall(mc_slot(vars, name)))"
                          for name in ("rho_r", "rho_c")), ", ")
        phys = mtile.tile.physical
        kDim = mtile.model.grid_params.kDim
        rho_cbar = view(Springsteel.ref_rho_c(mtile.ref_state), :, 1)
        rho_dbar = view(ref_rho_d(mtile.ref_state), :, 1)
        rho_tbar = view(ref_rho_t(mtile.ref_state), :, 1)
        rr = mc_slot(vars, "rho_r")
        rc = mc_slot(vars, "rho_c")
        rd = vars["rho_d"]
        rt = vars["rho_t"]
        post_r = 0.0
        post_c = 0.0
        # The VAPOR is the residual rho_t - rho_d - rho_c - rho_r, so it is where any error
        # the condensate is no longer allowed to absorb has to go. Forcing rho_c >= 0
        # removes a reservoir the formulation was leaning on, and this is the number that
        # says whether the vapor can take it.
        post_v = Inf
        @inbounds for i in axes(phys, 1)
            k = mod1(i, kDim)
            post_r = min(post_r, phys[i, rr, 1])
            rho_c_tot = phys[i, rc, 1] + rho_cbar[k]
            post_c = min(post_c, rho_c_tot)
            post_v = min(post_v, (phys[i, rt, 1] + rho_tbar[k]) - (phys[i, rd, 1] + rho_dbar[k]) -
                                 rho_c_tot - phys[i, rr, 1])
        end

        # Cloud-top diagnostic: the highest gridpoint carrying condensate, and the coldest
        # temperature anywhere. A positivity limiter redistributes mass WITHIN a leg, so it
        # can only move condensate along that leg — this is what shows whether it is
        # depositing cloud at an altitude the energy budget cannot support.
        z_all = view(mtile.tilepoints, :, size(mtile.tilepoints, 2))
        z_cloud = -Inf
        @inbounds for i in axes(phys, 1)
            (phys[i, rc, 1] + rho_cbar[mod1(i, kDim)]) > 1.0e-6 && (z_cloud = max(z_cloud, z_all[i]))
        end

        # The summary always prints on the interval: with every species bounded the
        # per-species blocks below fall silent, and the limiter health has to stay visible.
        @info """water summary step $t (t = $(round(t * mtile.model.ts; digits=1)) s)
          positivity limiter shortfall (must be 0): $shortfall
          post-fit(reconstruction) minima: rho_r = $post_r, rho_c = $post_c, rho_v = $post_v
          pre-fit(var_np1) minima: rho_r = $(minimum(view(st, MC_PRE_R, :))), rho_c = $(minimum(view(st, MC_PRE_C, :)))
          highest gridpoint with rho_c > 1e-6: $(isfinite(z_cloud) ? z_cloud : NaN) m
          vapor partition gap max|rho_v - res_rho_t| (0 unless :vapor_retrieval = :blend; bounded by dcap*rho_vs where the saturator bites): $(maximum(view(st, MC_VAPOR_GAP, :))) kg/m^3"""

        # The depletion census: how widely, and how hard, the step is draining each species.
        # `euler` is bounded by 1 wherever a cap is the binding constraint; `ab3` is what the
        # integrator actually applies, and the gap between them is the leading AB3 weight.
        for (name, off, sink) in (("rho_r", MC_DEPLETION_R, "evaporation"),
                                  ("rho_c", MC_DEPLETION_C, "evaporation"),
                                  ("rho_v", MC_DEPLETION_V, "condensation"))
            npts = sum(view(st, off, :))
            npts > 0.0 || continue
            @info """water depletion [$name] step $t (t = $(round(t * mtile.model.ts; digits=1)) s)
              gridpoints with $name > 0: $(Int(npts))
              sinks pinned at their bound: $sink $(Int(sum(view(st, off + 1, :)))), auto+coll $(Int(sum(view(st, off + 2, :))))
              MICROPHYSICS depletion fraction (what the bound governs): max $(maximum(view(st, off + 8, :))), points > 1: $(Int(sum(view(st, off + 9, :))))
              FULL-tendency depletion fraction: Euler $(maximum(view(st, off + 3, :))), ACTUAL(AB3) $(maximum(view(st, off + 4, :)))
              points driven negative by the step: $(Int(sum(view(st, off + 5, :))))
              points past AB3's 0.545 stability limit with NO cap active: $(Int(sum(view(st, off + 6, :))))
              points whose sink HISTORY alone is inadmissible (bound clamped at 0): $(Int(sum(view(st, off + 7, :))))"""
        end

        for (name, offset) in (("rho_r", MC_BUDGET_R), ("rho_c", MC_BUDGET_C))
            tid = argmin(view(st, offset, :))
            val = st[offset, tid]
            val < 0.0 || continue
            trm = (name == "rho_r" ?
                   ("adv", "-rho*div", "Qdot_r", "auto+coll", "-dFr/dz") :
                   ("adv", "-rho*div", "Qdot", "-auto-coll"))
            budget = join(("$(trm[k])=$(st[offset + k, tid])" for k in 1:length(trm)), "  ")
            @info """water budget [$name] step $t (t = $(round(t * mtile.model.ts; digits=1)) s)
              worst AB3-projected point: $name $(st[offset + 10, tid]) -> $val kg/m^3 at z = $(st[offset + 8, tid]) m
              the FORWARD-EULER projection of the same point: $(st[offset + 9, tid]) kg/m^3
              tendency terms [kg/m^3/s]: $budget
              net explicit tendency: $(sum(st[offset + k, tid] for k in 1:5)) kg/m^3/s (x ts = $(mtile.model.ts * sum(st[offset + k, tid] for k in 1:5)))
              local flow: div(v) = $(st[offset + 6, tid]) 1/s, w = $(st[offset + 7, tid]) m/s"""
        end
    end

    # Reset the per-step block regardless of whether this was a print step, so the
    # recorded worst point always belongs to the step just finished.
    @inbounds fill!(view(st, MC_BUDGET_FIRST:length(MC_WATER_STATS), :), 0.0)
    return nothing
end

# ── Equation set ───────────────────────────────────────────────────────────────

"""
    mc_driver!(mtile, colstart, colend, t, geom)

Total-energy moist compressible equation set — the geometry-generic master driver.
Prognostic slots (perturbations vs the `PressureReferenceState` except u, w, rho_r,
and v on the cylindrical geometries): p' [Pa], rho_d', rho_t', u, w, E_t' [J/m^3],
Q_ss' [kg/m^3], rho_r, rho_c' (+ tangential v, slot 10, on the cylinders).

Each step the CLOSED-FORM `retrieve_temperature` gives T from the prognostic liquid
density, the vapor follows as the residual rho_v = rho_t - rho_d - rho_c - rho_r, and
the condensation rate is the limited supersaturation relaxation, which moves mass
between rho_c and the vapor at fixed rho_t and E_t. The energy equation carries no
condensation source (exact first law); the pressure equation's condensation
coefficient is (L_v - R_v*C_pt*T/R_m). See reference/Scythe_moist_compressible.tex and,
for why the condensate is prognostic rather than the residual it used to be,
reference/HANDOFF_DIAGNOSED_CLOUD.md.

Everything geometry-specific — the derivative-slot mapping, metric and curvature
terms, and the tangential-wind machinery — is dispatched on the singleton `geom`
trait (see mc_geometry.jl), so the Cartesian path compiles to exactly the
historical `moist_compressible_XZ` code. The name-dispatched equation sets are
thin wrappers below.
"""
function mc_driver!(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64,
                    geom::MCGeometry)

    grid = mtile.tile
    gridpoints = mtile.tilepoints
    expdot = mtile.expdot_n
    impdot = mtile.impdot_n
    model = mtile.model
    refstate = mtile.ref_state

    # Physical parameters. Momentum and the moist entropy s_t (heat) are the diffused
    # quantities: the masses, pressure, total energy and Q_ss carry no diffusive tendency
    # of their own (the heat/friction increments are slaved onto them). Momentum and heat
    # eddy coefficients are specified independently (eddy mixing is not a molecular-ratio
    # process); the heat coefficients default to the momentum values. Water-species mixing
    # is deferred (see the handoff doc).
    Khdiff = model.physical_params[:Khdiff]
    Kvdiff = model.physical_params[:Kvdiff]
    Khdiff_heat = get(model.physical_params, :Khdiff_heat, Khdiff)
    Kvdiff_heat = get(model.physical_params, :Kvdiff_heat, Kvdiff)
    Kvdiff_water = get(model.physical_params, :Kvdiff_water, 0.0)
    tau_qss = get(model.physical_params, :tau_qss, 10.0)
    # Turbulent Prandtl number for the Smagorinsky heat diffusion (Khdiff_heat < 0
    # sentinel, below). 1.0 = heat mixes with the same eddy diffusivity as momentum.
    Pr_t = get(model.physical_params, :Pr_t, 1.0)
    # Horizontal water-species mixing: 0.0 = off (default), < 0 = Smagorinsky
    # K_smag/Sc_t. DIAGNOSTIC and NOT energy consistent -- see the block by slot 8.
    Khdiff_water = get(model.physical_params, :Khdiff_water, 0.0)
    Sc_t = get(model.physical_params, :Sc_t, 1.0)

    # Coriolis parameter (constant f-plane) for the cylindrical geometries; the
    # Cartesian slice carries no rotation and its methods never read it.
    fcor = get(model.physical_params, :f, 0.0)

    # Rayleigh sponge (Durran-Klemp 1983 eq. 29 profile above z_damp): momentum-only,
    # damping u and w toward the resting base state. alpha = 0 (or absent keys)
    # disables it and skips the block entirely, keeping alpha = 0 configs bit-identical.
    alpha = get(model.physical_params, :alpha, 0.0)
    z_damp = get(model.physical_params, :z_damp, 0.0)

    # Warm-rain microphysics (autoconversion, collection, sedimentation, and the rain
    # channel of the supersaturation relaxation). N_r [#/cm^3] is the fixed rain-drop
    # number of the monodisperse tau_r closure; zeroing it (or the switch) makes the
    # rain channel inert and slot 8 advection-only. N_0 [m^-4] > 0 switches the
    # channel's timescale to the exponential (Marshall-Palmer) DSD closure (classic
    # value 8.0e6); absent, the monodisperse closure is bit-identical to before.
    precipitation = get(model.options, :precipitation, false)::Bool
    # Negative-water production attribution; see `water_budget_probe!`. Hoisted out of the
    # tendency block so the default path pays one Dict lookup per column, not two.
    budget_trace = get(model.options, :water_budget_trace, 0)::Int > 0
    # Stage-3 diagnostic lever on the condensate depletion caps; see `qss_condensation_rates`.
    # 1.0 (the default) is bitwise inert.
    cap_factor = get(model.physical_params, :water_cap_factor, 1.0)
    # How the water depletion budgets are sized. `:ab3` (the default) uses the integrator's
    # actual three-level combination via `_ab3_sink_bound`. `:euler` pins the budget to the
    # `t = 1` branch at every step, which reproduces the forward-Euler `rho/ts` caps this file
    # carried before -- BITWISE, so it is the A/B lever for every measurement in
    # reference/FINDINGS_NEGATIVE_WATER_ATTRIBUTION.md (runs A-H were all taken under it).
    # In `:euler` the vapor ceiling also keeps its historical form, which did NOT read
    # `cap_factor`; under `:ab3` all three channels see the probe.
    # In `options`, not `physical_params`: this is a choice of DISCRETIZATION (which
    # integrator the budget is written against), not a physical parameter, and
    # `physical_params` is a Dict{Symbol,Float64} besides.
    cap_euler = get(model.options, :water_cap_mode, :ab3)::Symbol === :euler
    cap_t = cap_euler ? 1 : t
    cap_vfac = cap_euler ? 1.0 : cap_factor
    # Which discrete representation the VAPOR is retrieved from. `:blend` (the default since
    # 2026-07-29, and the state of every configuration that does not set the key) is the regime
    # blend of `vapor_retrieval_blend`: the supersaturation residual in cloud, where the density
    # budget cancels catastrophically, and the density residual in dry air, where the
    # supersaturation residual does. `:residual` is the pre-blend density-budget route and is
    # BITWISE the code that had no option at all -- retained as the A/B lever for every
    # measurement taken under it. In `options`, not `physical_params`, for the same reason as
    # `water_cap_mode`: it is a choice of DISCRETE REPRESENTATION, not a physical parameter.
    vapor_retrieval = get(model.options, :vapor_retrieval, :blend)::Symbol
    (vapor_retrieval === :residual || vapor_retrieval === :blend) ||
        error("options[:vapor_retrieval] = :$(vapor_retrieval) is not recognized; " *
              "use :blend (the regime-blended retrieval, the default; see " *
              "vapor_retrieval_blend) or :residual (the pre-blend density-budget residual)")
    vapor_blend = vapor_retrieval === :blend
    # Blend thresholds. The rho_liq pair does the regime selection and the s pair rejects a
    # detached Q_ss; see `_vapor_blend_weight` for why these are not tuned numbers (every band
    # on the swept grid gave the same answer, the populations being ~8 decades apart in rho_liq)
    # and why the trust thresholds sit above the s = 1 ceiling. Read unconditionally -- four
    # Dict lookups per column, off the back of the one the option above already pays.
    blend_l0 = get(model.physical_params, :vapor_blend_l0, 1.0e-6)
    blend_l1 = get(model.physical_params, :vapor_blend_l1, 1.0e-4)
    blend_t0 = get(model.physical_params, :vapor_blend_t0, 2.0)
    blend_t1 = get(model.physical_params, :vapor_blend_t1, 5.0)
    # The blend's CORRECTION CAP, `|rho_v - res_rho_t| <= dcap*rho_vs`, applied through the C¹
    # saturator `_blend_saturate`. This is the load-bearing guard, not a safety belt: the
    # uncapped blend detonates the NOPRECIP storm because the partition gap is unbounded
    # RELATIVE to rho_vs (which collapses in the rising anvil) while `w_trust`, a function of
    # the supersaturation magnitude s, never sees it. See `vapor_retrieval_blend`. `Inf`
    # recovers the uncapped blend bitwise; `0.0` degenerates to `:residual`.
    blend_dcap = get(model.physical_params, :vapor_blend_dcap, 0.02)
    # Whether the thermodynamic interface reads a floored rho_liq. `:none` (the default) is
    # bitwise the code that had no option; see `condensate_floor_mode` for why this is not a
    # clamp. In `options` for the same reason as the two above: a choice of what the
    # diagnostics READ, not a physical parameter.
    cond_floor = condensate_floor_mode(model.options)
    # Which control variable slot 9 carries. `:none` (the default) is bitwise the code that
    # had no option: the branch below then multiplies by an exact 1.0 and divides by an exact
    # 1.0, both of which are identity in IEEE. See `condensate_transform_mode`.
    ctrans = condensate_transform_mode(model.options)
    ctrans_on = ctrans !== :none
    cmu = get(model.physical_params, :condensate_mu, 1.0e-7)
    # The same for slot 8 (rain). Independent knob, same machinery; see `rain_transform_mode`
    # for why the two are separate and for what rain needs that cloud did not.
    rtrans = rain_transform_mode(model.options)
    rtrans_on = rtrans !== :none
    rmu = get(model.physical_params, :rain_mu, 1.0e-7)
    if (ctrans_on || rtrans_on) && budget_trace
        # `water_budget_probe!` attributes a slot's production to named channels and reports
        # them in kg/m^3/s. Under a transform the ADV column is in CONTROL-VARIABLE units
        # while the source columns are densities, so the rows would not sum to the tendency
        # and the mismatch would be invisible. Refuse rather than print a wrong budget --
        # this file has shipped one silently before (see `water_budget_trace`).
        error("options[:water_budget_trace] with a water transform on " *
              "(condensate_transform = :$(ctrans), rain_transform = :$(rtrans)) is not " *
              "implemented: the advective term is then in control-variable units and the " *
              "source terms in density units, so the attribution does not close. Rescale " *
              "the transformed slot's columns by the Jacobian before enabling it.")
    end
    N_r = precipitation ? get(model.physical_params, :N_r, 1.0e-3) : 0.0
    N_0 = precipitation ? get(model.physical_params, :N_0, 0.0) : 0.0

    # Louis boundary layer (vertical mixing + surface drag; mc_boundary_layer.jl)
    # and Smagorinsky horizontal closure (Ls > 0 replaces the constant Khdiff in
    # the momentum diffusion). Both default OFF so existing configurations are
    # bit-identical.
    louis_bl = get(model.options, :louis_bl, false)::Bool
    # Horizontal acoustic semi-implicit (the patch-level ADI sweep in
    # horizontal_si.jl). Default OFF so existing configurations are bit-identical.
    hsi = get(model.options, :horizontal_semiimplicit, false)::Bool
    # Exact (unsplit) 2-D acoustic semi-implicit (exact_si.jl). Shares the
    # horizontal remainder/staging blocks with hsi below; the driver exits at
    # the end of phase A (before the per-column implicit solve — the solve
    # happens at the patch level, then exact_si_apply_column! completes the
    # column). :exact_si_zero_x is the A≡0 TEST lever: it zeroes the whole
    # horizontal coupling so the path must be bitwise the vertical-only path.
    xsi = get(model.options, :exact_si, false)::Bool
    xsi_zero = xsi && (get(model.options, :exact_si_zero_x, false)::Bool)
    hsi_like = hsi || (xsi && !xsi_zero)
    # State-dependent vertical acoustic linearization: the implicit pair's
    # coefficients (Pξ, ρ̂_t and the slaved-leg chains) come from the CURRENT
    # column state each step instead of the resting reference, and the
    # Helmholtz matrix is refactorized per column per step. Removes the
    # convective (finite-amplitude) SI ceiling — the resting-reference form
    # leaves δ·Co_z of the grid-scale acoustic operator explicit in a core
    # whose state deviates by δ, fatal in TC deep convection at Co_z ≳ 4
    # (reference/SI_CONVECTIVE_CEILING.md). Default OFF so existing configurations
    # are bit-identical; the TC configs enable it. Spline (RiRk) vertical only.
    sd_si = get(model.options, :state_dependent_si, false)::Bool
    l_inf = get(model.physical_params, :l_inf, 80.0)
    Cd_param = get(model.physical_params, :Cd, -1.0)
    sfc_wind_factor = get(model.physical_params, :sfc_wind_factor, 1.0)
    Ls = get(model.physical_params, :Ls, 0.0)
    K_min = get(model.physical_params, :K_min, 0.0)
    use_smag = Ls > 0.0

    # Bulk surface enthalpy/moisture fluxes over a fixed-SST ocean: the flux
    # values are the bottom nodes of the Louis heat/water flux columns, so the
    # switch requires louis_bl. SST is in KELVIN (Float64 params carry no units;
    # the guard catches the Celsius footgun).
    surface_fluxes = get(model.options, :surface_fluxes, false)::Bool
    if surface_fluxes && !louis_bl
        error("options[:surface_fluxes] requires options[:louis_bl] — the fluxes " *
              "enter as the bottom nodes of the Louis boundary-layer flux columns")
    end
    Ck = get(model.physical_params, :Ck, 1.0e-3)
    SST = get(model.physical_params, :SST, 301.15)
    if surface_fluxes && SST <= 200.0
        error("physical_params[:SST] must be in Kelvin (got $SST — 28 C is 301.15)")
    end
    U_min = get(model.physical_params, :U_min, 0.0)

    # Gridpoints: z from the geometry's vertical column; r is the geometry metric
    # handle (radius view on the cylinders, colatitude/a/Omega on the sphere,
    # `nothing` on the Cartesian geometries — see mc_metric)
    z = view(gridpoints,colstart:colend,zcoord(geom))
    r = mc_metric(geom, model, gridpoints, colstart, colend)

    # Prognostic slot views, geometry-mapped (see mc_slot_views): `_x` is ∂x on the
    # slice and ∂r on the cylinders, `_z`/`_zz` sit at zslot(geom), and the raw
    # azimuthal `f_l`/`f_ll` pair exists only on the 3D grid (`nothing` on 2D).
    pv  = mc_slot_views(grid, colstart, colend, 1, geom)   # p' [Pa]
    rdv = mc_slot_views(grid, colstart, colend, 2, geom)   # rho_d'
    rtv = mc_slot_views(grid, colstart, colend, 3, geom)   # rho_t'
    uv  = mc_slot_views(grid, colstart, colend, 4, geom)   # u
    wv  = mc_slot_views(grid, colstart, colend, 5, geom)   # w
    etv = mc_slot_views(grid, colstart, colend, 6, geom)   # E_t'
    qsv = mc_slot_views(grid, colstart, colend, 7, geom)   # Q_ss'
    rrv = mc_slot_views(grid, colstart, colend, 8, geom)   # rho_r (rho_rbar = 0)
    rcv = mc_slot_views(grid, colstart, colend, 9, geom)   # rho_c'
    vv  = mc_v_views(geom, grid, colstart, colend)         # tangential v (cylinders)

    pp = pv.f;      pp_x = pv.f_x;         pp_z = pv.f_z
    rho_dp = rdv.f; rho_dp_x = rdv.f_x;    rho_dp_z = rdv.f_z; rho_dp_zz = rdv.f_zz
    rho_tp = rtv.f; rho_tp_x = rtv.f_x;    rho_tp_z = rtv.f_z; rho_tp_zz = rtv.f_zz
    u = uv.f;       u_x = uv.f_x;          u_z = uv.f_z;       u_zz = uv.f_zz
    w = wv.f;       w_x = wv.f_x;          w_z = wv.f_z;       w_zz = wv.f_zz
    E_tp = etv.f;   E_tp_x = etv.f_x;      E_tp_z = etv.f_z
    Q_ssp = qsv.f;  Q_ssp_x = qsv.f_x;     Q_ssp_z = qsv.f_z
    rho_rp = rrv.f; rho_rp_x = rrv.f_x;    rho_rp_z = rrv.f_z; rho_rp_zz = rrv.f_zz
    rho_cp = rcv.f; rho_cp_x = rcv.f_x;    rho_cp_z = rcv.f_z; rho_cp_zz = rcv.f_zz

    # Reference state (pressure-based). Views, not `[:,1]` copies: these are read-only and
    # loop-invariant, so copying them allocated a fresh kDim vector per column per timestep.
    pbar = view(ref_pressure(refstate),:,1)
    pbar_z = view(ref_pressure(refstate),:,2)
    rho_dbar = view(ref_rho_d(refstate),:,1)
    rho_dbar_z = view(ref_rho_d(refstate),:,2)
    rho_tbar = view(ref_rho_t(refstate),:,1)
    rho_tbar_z = view(ref_rho_t(refstate),:,2)
    E_tbar = view(ref_total_energy(refstate),:,1)
    E_tbar_z = view(ref_total_energy(refstate),:,2)
    Q_ssbar = view(ref_qss(refstate),:,1)
    Q_ssbar_z = view(ref_qss(refstate),:,2)
    rho_cbar = view(Springsteel.ref_rho_c(refstate),:,1)
    rho_cbar_z = view(Springsteel.ref_rho_c(refstate),:,2)
    # LOCAL reference sound-speed-squared profile γ̄_m(z)·p̄/ρ̄_t (see
    # mc_reference_diagnostics) — the acoustic linearization must use the local
    # value so the explicit remainder is O(perturbation) at every level; the
    # domain-mean sound_speed_sq is unstable above Co_z ≈ 4.5 on a stratified base.
    Pxi_bar = mtile.mc_ref_diag.Pxi_prof

    # Per-thread work vectors for every temporary below (see `MC_SCRATCH_SLOTS`). Each `@.`
    # writes into a preallocated column instead of allocating a fresh one per column per step.
    S = @inbounds mtile.mc_scratch[Threads.threadid()]

    # Total fields (perturbation + reference)
    p = S.p;         @. p = pp + pbar
    rho_d = S.rho_d; @. rho_d = rho_dp + rho_dbar
    rho_t = S.rho_t; @. rho_t = rho_tp + rho_tbar
    E_t = S.E_t;     @. E_t = E_tp + E_tbar
    Q_ss = S.Q_ss;   @. Q_ss = Q_ssp + Q_ssbar
    # Slot 9. Under `:none` this is the cloud density and `Jc ≡ 1`; under a transform the slot
    # carries the control variable `ν' = ν − ν̄` with `ν̄ = bhyp(ρ̄_c)` (Ooyama predicts the
    # DEVIATION from the transformed background, §4d), and the density is recovered pointwise.
    # `ν̄` and `ν̄_z` are formed here rather than cached because ρ̄_c is identically zero in
    # every current configuration, making them exactly zero for two kDim-length operations.
    nu_c = S.nu_c; nu_c_z = S.nu_c_z; Jc = S.Jc
    rho_c = S.rho_c
    if ctrans_on
        @. nu_c = rho_cp + bhyp(rho_cbar, cmu)
        @. nu_c_z = rho_cp_z + (dbhyp(rho_cbar, cmu) * rho_cbar_z)
        if ctrans === :bhyp
            @. rho_c = ahyp(nu_c, cmu)
        else
            @. rho_c = ahyp_smooth(nu_c, cmu)
        end
        # Evaluated at the RECOVERED density, clamped at zero for the argument only: that is
        # what bounds J in [0.5, 1] and removes the source stiffness. Clamping the argument
        # changes no state, no mass and no energy — it perturbs the tendency by at most a
        # factor of two at points whose density is within μ of zero.
        @. Jc = dbhyp(max(rho_c, 0.0), cmu)
    else
        @. nu_c = rho_cp + rho_cbar
        @. nu_c_z = rho_cp_z + rho_cbar_z
        copyto!(rho_c, nu_c)
        fill!(Jc, 1.0)
    end
    # Slot 8, the same construction for rain — with one simplification: rain is a TOTAL with
    # `ρ̄_r ≡ 0`, so the slot IS `ν_r`, with no background to add and no `ν̄_r` to form. Under
    # `:none` the copy is of identical doubles and `Jr` is an exact 1.0, so the default path
    # stays bit-identical (the same construction slot 9 uses above).
    nu_r = S.nu_r; nu_r_z = S.nu_r_z; Jr = S.Jr
    rho_r = S.rho_r
    copyto!(nu_r, rho_rp)
    copyto!(nu_r_z, rho_rp_z)
    if rtrans_on
        if rtrans === :bhyp
            @. rho_r = ahyp(nu_r, rmu)
        else
            @. rho_r = ahyp_smooth(nu_r, rmu)
        end
        @. Jr = dbhyp(max(rho_r, 0.0), rmu)
    else
        copyto!(rho_r, nu_r)
        fill!(Jr, 1.0)
    end

    # Total vertical gradients (perturbation + reference)
    p_z = S.p_z;         @. p_z = pp_z + pbar_z
    rho_d_z = S.rho_d_z; @. rho_d_z = rho_dp_z + rho_dbar_z
    rho_t_z = S.rho_t_z; @. rho_t_z = rho_tp_z + rho_tbar_z
    E_t_z = S.E_t_z;     @. E_t_z = E_tp_z + E_tbar_z
    Q_ss_z = S.Q_ss_z;   @. Q_ss_z = Q_ssp_z + Q_ssbar_z
    # The DENSITY vertical gradient, f'(n)·∂z n = ∂z n / J — self-consistent with the recovered
    # density because both come from the same fit. Under `:none`, J is exactly 1.0 and this is
    # bitwise `rho_cp_z + rho_cbar_z`. Advection does NOT read this (it is transform-invariant
    # and reads `nu_c_z` directly); this exists for any consumer that wants dρ_c/dz.
    rho_c_z = S.rho_c_z; @. rho_c_z = nu_c_z / Jc

    # Diagnostic thermodynamic state. The condensate is PROGNOSTIC, so the liquid
    # density is known and the temperature retrieval is closed-form; the vapor is the
    # residual of the water masses. See the file header for why this direction and not
    # the other one.
    ke = S.ke;   mc_ke!(ke, geom, u, w, vv)
    geo = S.geo; @. geo = ke + (gravity * z)
    M = S.M;     @. M = p + E_t - (rho_t * geo)
    rho_liq = S.rho_liq; @. rho_liq = rho_c + rho_r
    # What the THERMODYNAMICS reads. Under the default `:none` this is a copy of identical
    # doubles, so the whole option is bitwise inert (same construction as `rho_v` under
    # `:residual`). Under `:diagnostic` it is the floored liquid, and ONLY the consumers below
    # see it: the retrieval, q_l -> C_vt/R_m/gamma_m, Q_s_energy, the entropy and E_sed.
    # `rho_liq` itself stays raw, because two of its readers must NOT be floored --
    # `res_rho_t` (flooring the water partition would manufacture vapor, which is a mass
    # source, not a reading) and `_vapor_blend_weight` (whose smoothstep already clamps a
    # small negative rho_liq to weight zero; see the TeX on Eq. blend_saturator).
    rho_liq_t = S.rho_liq_t
    if cond_floor
        @. rho_liq_t = max(rho_c, 0.0) + max(rho_r, 0.0)
    else
        copyto!(rho_liq_t, rho_liq)
    end
    Tk = S.Tk;   @. Tk = retrieve_temperature(M, rho_d, rho_t, rho_liq_t)
    p_hPa = S.p_hPa;   @. p_hPa = p / 100.0
    rho_vs = S.rho_vs; @. rho_vs = rho_v_sat(Tk, p_hPa)
    # The DENSITY-BUDGET residual, always. `S.res_rho_t` is not an alias of `S.rho_v`: the
    # reconciliation below must read this one whatever the retrieval is (see qss_relaxation),
    # and the partition-gap census differences the two.
    res_rho_t = S.res_rho_t; @. res_rho_t = rho_t - rho_d - rho_liq
    # ... and `S.rho_v` is the vapor EVERY OTHER consumer reads. Under `:residual` it is a copy
    # of the residual — a copy of identical doubles, so the whole option is bitwise inert.
    rho_v = S.rho_v
    if vapor_blend
        @. rho_v = vapor_retrieval_blend(Q_ss, rho_vs, rho_liq, res_rho_t,
                                         blend_l0, blend_l1, blend_t0, blend_t1, blend_dcap)
    else
        @. rho_v = res_rho_t
    end
    q_v = S.q_v;     @. q_v = rho_v / rho_d
    q_l = S.q_l;     @. q_l = rho_liq_t / rho_d      # thermodynamic reader: see rho_liq_t
    C_vt = S.C_vt;   @. C_vt = Cvd + (q_v * Cvv) + (q_l * Cl)
    R_m = S.R_m;     @. R_m = Rd + (q_v * Rv)
    C_pt = S.C_pt;   @. C_pt = C_vt + R_m
    gamma_m = S.gamma_m; @. gamma_m = C_pt / C_vt
    # State-dependent acoustic coefficient Pξⁿ(z) = γ_m·p/ρ_t of the CURRENT
    # column state — used by the remainder/history staging below and by
    # semiimplicit_adjustment_p (same thread, same column, so the scratch
    # column persists into the adjustment call).
    if sd_si
        @. S.sd_pxi = gamma_m * p / rho_t
    end
    Lv = S.Lv;           @. Lv = L_v(Tk)
    drvs_dT = S.drvs_dT; @. drvs_dT = drho_vsat_dT(Tk, p_hPa)
    drvs_dp = S.drvs_dp; @. drvs_dp = drho_vsat_dp(Tk, p_hPa)

    # Condensation: limited supersaturation relaxation with the energy-consistent
    # psychrometric factor, split between the cloud and rain channels in proportion to
    # their inverse timescales (1/tau = 1/tau_c + 1/tau_r). Qdot moves mass between the
    # prognostic condensate (slot 9) and the residual vapor at fixed rho_t and E_t, so
    # the latent heat comes out of the retrieval and no adjustment step is needed after
    # the timestep. With the rain channel inert (N_r = 0), Qdot is bit-identical to the
    # single-category closure and Qdot_r is exactly zero.
    Q_s = S.Q_s;   @. Q_s = Q_s_energy(Tk, p, rho_d, q_v, q_l)
    Qdot = S.Qdot          # cloud channel
    Qdot_r = S.Qdot_r      # rain channel
    # `options[:condensation] = false` switches the phase change off ENTIRELY --
    # both channels, condensation and evaporation. It is a DIAGNOSTIC control, not
    # a physics option: `:precipitation => false` only zeroes N_r/N_0, which stops
    # rain but leaves the cloud channel running, so a run advertised as "no
    # physics" still evaporates cloud every step.
    #
    # It was introduced to diagnose the phantom-cloud drain (see the file header),
    # back when rho_c was the residual and a subsaturated column could carry
    # ~2e-6 kg/m^3 of cloud that was not there, evaporate it, and have it regenerated
    # from the same fit error on the next step. With rho_c prognostic that pathway is
    # closed -- a subsaturated column has rho_c = 0 exactly and both gates shut, so
    # this switch should now make NO difference to a cloud-free initial state. That is
    # a useful regression check in its own right (model_tests/tc_discrete_balance.jl),
    # and it remains the clean separator between "the initial state is not a discrete
    # steady state" and "the moisture is doing something".

    # ── Depletion budgets ──────────────────────────────────────────────────────
    # The three channels' bounds are sized HERE, against the integrator's own three-level
    # combination (`_ab3_sink_bound`) rather than the forward-Euler `rho/ts` this code used to
    # hard-code at every step. The raw (unclamped) bounds are kept in scratch because the
    # AUTO_COLL joint cap below needs the cloud one, and the depletion census needs all three
    # to detect a binding bound bitwise and to count the points whose sink history alone is
    # already inadmissible.
    cap_c = S.cap_c
    cap_r = S.cap_r
    cap_v = S.cap_v
    micro_nm1 = mtile.mc_micro_nm1
    micro_nm2 = mtile.mc_micro_nm2
    @inbounds for i in eachindex(cap_c)
        g = colstart + i - 1
        cap_c[i] = _ab3_sink_bound(model.ts, cap_t, cap_factor * max(rho_c[i], 0.0),
                                   micro_nm1[g, MC_MICRO_C], micro_nm2[g, MC_MICRO_C])
        cap_r[i] = _ab3_sink_bound(model.ts, cap_t, cap_factor * max(rho_r[i], 0.0),
                                   micro_nm1[g, MC_MICRO_R], micro_nm2[g, MC_MICRO_R])
        # The vapor is a SINK channel too -- condensation removes it -- so its bound has the
        # same form and is negated into a ceiling on `Qdot_c + Qdot_r` at the use site.
        # DELIBERATELY the BLENDED `rho_v`, not `res_rho_t`: the bound exists to stop the
        # closure over-depleting the vapor IT READS, and the closure two blocks down reads the
        # blend. Sizing the budget off a different representation than the one being depleted
        # would be the same class of mismatch `water_cap_mode` was written to fix.
        cap_v[i] = _ab3_sink_bound(model.ts, cap_t, cap_vfac * max(rho_v[i], 0.0),
                                   micro_nm1[g, MC_MICRO_V], micro_nm2[g, MC_MICRO_V])
    end
    if get(model.options, :condensation, true)::Bool
        # The closure reads the BLENDED `rho_v` (unchanged by design): the vapor it evaporates
        # into and its evaporation bound must be the same representation, and in cloud -- where
        # the closure is actually active -- the blend is the better-conditioned one.
        for i in 1:length(Qdot)
            Qdot[i], Qdot_r[i] = qss_condensation_rates(Q_ss[i], rho_v[i], rho_c[i], rho_r[i],
                                                        rho_d[i], Tk[i], p_hPa[i], Q_s[i],
                                                        model.ts, N_r; N_0=N_0,
                                                        floor_c=min(cap_c[i], 0.0),
                                                        floor_r=min(cap_r[i], 0.0),
                                                        ceil_v=-min(cap_v[i], 0.0))
            if isnan(Qdot[i]) || isnan(Qdot_r[i])
                error("Qdot is NaN at index $i, time $(t)!\n" *
                      "  T = $(Tk[i]) K, p = $(p[i]) Pa, rho_d = $(rho_d[i]), " *
                      "rho_t = $(rho_t[i]), rho_c = $(rho_c[i]), rho_r = $(rho_r[i])\n" *
                      "  rho_liq = $(rho_liq[i]), rho_v = $(rho_v[i]), " *
                      "rho_vs = $(rho_vs[i]), Q_ss = $(Q_ss[i]), Q_s = $(Q_s[i])\n" *
                      "  M = $(M[i]), ke = $(ke[i]), E_t = $(E_t[i]), " *
                      "Qdot = $(Qdot[i]), Qdot_r = $(Qdot_r[i])")
            end
        end
    else
        fill!(Qdot, 0.0)
        fill!(Qdot_r, 0.0)
    end

    # Warm-rain conversion and sedimentation. Autoconversion + collection move cloud to
    # rain — a liquid-to-liquid exchange, thermodynamically inert (T, p, E_t, Q_ss all
    # unmoved; with the condensate prognostic it is now an equal and opposite pair of
    # sources on slots 9 and 8, and rho_liq — hence the retrieval — does not move). The
    # sedimentation flux F_r = rho_r*Vt (Vt <= 0) moves rain mass AND the energy it
    # carries: e_l(T) + ke + gz per kg of liquid, with e_l = C_pv*T - L_v(T) in the
    # BF02 internal-energy convention. Its divergence sources rho_r, rho_t and E_t with
    # the SAME fitted -dF/dz, so the two densities cannot drift apart, and the column
    # integral telescopes to the boundary fluxes — with a free (Natural) bottom BC on
    # rho_r the surface flux removes rain from the domain. T is invariant under the
    # local exchange (delta_E_t = (e_l+ke+gz)*delta_rho at delta_rho_r = delta_rho_t);
    # the residual flux terms are the physical energy transport by falling rain. No p
    # or Q_ss source: rain exerts no partial pressure, and the (small) sedimentation
    # dT/dt is omitted from the saturation chain rule below (absorbed by the
    # condensation relaxation).
    AUTO_COLL = S.AUTO_COLL
    Fr_z = S.Fr_z
    E_sed_z = S.E_sed_z
    if precipitation
        for i in 1:length(AUTO_COLL)
            auto = autoconversion_density(max(rho_c[i], 0.0), rho_d[i])
            coll = collection_density(max(rho_c[i], 0.0), rho_r[i], rho_d[i], Tk[i])
            # JOINT depletion cap: never convert more cloud than survives evaporation this
            # step. Cloud has two independent sinks — evaporation (`Qdot`, floored at
            # `min(cap_c, 0)` inside qss_condensation_rates) and conversion to rain
            # (`AUTO_COLL`) — and they share ONE budget, `Qdot - AUTO_COLL >= min(cap_c, 0)`.
            # Bounding each separately let them sum to twice the budget, which drives rho_c
            # to `-rho_c` in a single step — an O(1) negative, not a ringing lobe.
            #
            # This was invisible until the positivity bound went in: the rate functions
            # guard with `max(rho_c, 0)` so the overshoot was inert, and its mass merged
            # into the accumulated refit undershoot. With the bound active the limiter has
            # to remove that negative EVERY step, paying for it by shaving the cloud, which
            # destroys the cell.
            #
            # Capping AUTO_COLL rather than rescaling Qdot keeps the p, E_t and Q_ss
            # couplings — which have already consumed Qdot — exactly as they were.
            avail = max(Qdot[i] - min(cap_c[i], 0.0), 0.0)
            AUTO_COLL[i] = min(auto + coll, avail)
        end
        Vt = S.Vt;       @. Vt = rain_terminal_velocity(rho_r, rho_d, Tk)
        Fr = S.Fr;       @. Fr = max(rho_r, 0.0) * Vt
        E_sed = S.E_sed; @. E_sed = Fr * ((Cpv * Tk) - Lv + ke + (gravity * z))
        # Fitted flux divergences on rho_r's column basis (its BCs decide whether the
        # surface flux is free to be nonzero), same discrete d/dz as the advection.
        r_col = scratch_column(mtile, 8)
        r_col.uMish .= Fr
        Btransform!(r_col)
        Atransform!(r_col)
        Ixtransform(r_col, Fr_z)
        r_col.uMish .= E_sed
        Btransform!(r_col)
        Atransform!(r_col)
        Ixtransform(r_col, E_sed_z)
    else
        fill!(AUTO_COLL, 0.0)
        fill!(Fr_z, 0.0)
        fill!(E_sed_z, 0.0)
    end

    # Record this step's net microphysics tendency per channel, so the NEXT step's depletion
    # budget can be written against the integrator's actual three-level combination. Here,
    # after the `precipitation` branch, because the cloud channel is not final until
    # AUTO_COLL is. See `MC_MICRO_C` for the three channels and `_rotate_micro_history!` for
    # when these become the history levels.
    micro_n = mtile.mc_micro_n
    @inbounds for i in eachindex(Qdot)
        g = colstart + i - 1
        micro_n[g, MC_MICRO_C] = Qdot[i] - AUTO_COLL[i]
        micro_n[g, MC_MICRO_R] = Qdot_r[i]
        micro_n[g, MC_MICRO_V] = -(Qdot[i] + Qdot_r[i])
    end

    div = S.div; mc_divergence!(div, geom, u, u_x, w_z, vv, r)

    # ── Horizontal diffusion ───────────────────────────────────────────────────
    # Turbulence diffuses momentum (u, w) and the moist entropy s_t (heat). Horizontally,
    # s_t is diffused by the chain rule on the DRY-exact form s_d = C_vd ln p - C_pd ln rho_d
    # (an explicit function of the prognostic p, rho_d). Its Laplacian uses only the p/rho_d
    # derivative slots — no transform of a diagnosed field, which the column decomposition
    # cannot do (see reference/moist_compressible_diffusion_plan.md). The MOIST horizontal
    # correction (through the retrieval's T sensitivities) is deferred to the rainfall
    # session (reference/moist_compressible_diffusion_handoff.md). pbar/rho_dbar have no
    # x-dependence, so the total x-derivatives are the perturbation slots.
    sd_xx = S.sd_xx
    mc_sd_lap!(sd_xx, geom, p, rho_d, pv, rdv, r)

    # Smagorinsky horizontal eddy viscosity (Ls > 0): flow-dependent K(strain)
    # replaces the constant Khdiff in the momentum diffusion and its FRIC_KE
    # energy sink below. Computed BEFORE the heat diffusion because heat can be
    # tied to it (the Khdiff_heat < 0 sentinel below).
    K_smag = S.K_smag
    if use_smag
        mc_smag_k!(K_smag, geom, uv, vv, r, Ls, K_min)
    end

    # Horizontal diabatic heating [W/m^3] from the entropy diffusion: the source to internal
    # energy is rho_d*T*(ds_t/dt)_diff = rho_d*T*K_heat*d2(s_t)/dx2. This is the
    # moist analogue of Straka's rho*T*ds_d source.
    #
    # Khdiff_heat < 0 is the SENTINEL (as with Cd < 0 for the wind-dependent drag) for
    # "mix heat with the Smagorinsky eddy diffusivity K_smag/Pr_t" instead of a constant.
    # Without it, a Smagorinsky run mixes MOMENTUM only and leaves the thermodynamic
    # fields with no horizontal mixing whatsoever -- which on an axisymmetric grid is
    # especially bad, since there are no asymmetries to provide radial mixing and the
    # strong radial gradients of a TC then support undamped grid-scale buoyancy
    # structure. Tying it to K_smag rather than a constant keeps it resolution-general
    # (K scales with the resolved deformation, so it follows a nest refinement).
    # NOTE the form is K*grad^2(s), not div(K grad s): the variable-K correction
    # grad(K).grad(s) is dropped, exactly as the existing momentum diffusion does with
    # K_smag (mc_u_kdiff!) -- consistent with the surrounding scheme, not a new
    # approximation.
    QDOT_TH = S.QDOT_TH
    if Khdiff_heat < 0.0
        use_smag || error("physical_params[:Khdiff_heat] < 0 selects the Smagorinsky " *
                          "heat diffusivity, which requires Ls > 0")
        @. QDOT_TH = rho_d * Tk * (K_smag / Pr_t) * sd_xx
    else
        @. QDOT_TH = rho_d * Tk * Khdiff_heat * sd_xx
    end

    # Horizontal frictional KE change [W/m^3]. Momentum diffusion is a resolved-KE SINK to
    # the subgrid (the future TKE shear production), NOT dissipative heating: with an eddy K
    # the resolved KE lost goes to unresolved scales, and the molecular heating (proportional
    # to the far-smaller kinematic viscosity) is negligible. So E_t follows the KE down and
    # internal energy (T, p) is held. FRIC_KE is d(rho_t*ke)/dt from horizontal momentum
    # diffusion, added to E_t; p and Q_ss get NO friction term.
    FRIC_KE = S.FRIC_KE
    if use_smag
        mc_fric_ke!(FRIC_KE, geom, rho_t, K_smag, uv, wv, vv, r)
    else
        mc_fric_ke!(FRIC_KE, geom, rho_t, Khdiff, uv, wv, vv, r)
    end

    # Placeholders for intermediate calculations
    ADV = S.ADV
    FORCING = S.FORCING
    KDIFF = S.KDIFF

    # Pressure (slot 1): -v·∇p - γp∇·v + (R_m/C_vt)[(L_v - R_v C_pt T/R_m) Q̇_cond + Q̇_therm].
    # Q̇_cond is the TOTAL phase-change rate (cloud + rain channels); only the THERMAL
    # diffusion sources pressure (friction holds T, hence p; sedimentation moves no
    # partial pressure).
    #
    # The acoustic slots (1, 2, 3, 5, 6) stage the REMAINDER: the full tendency minus the
    # reference-linear vertical acoustic term in the same pointwise product-rule form, so
    # the grid-scale vertical-acoustic content cancels analytically and what AB3 advances
    # is O(perturbation). The linear part is integrated by the AI2* acoustic solve alone
    # (see semiimplicit_adjustment_p and the impdot staging block below).
    mc_advect!(ADV, geom, u, w, vv, r, pp_x, p_z, pv.f_l)
    # sd_si: the added-back linear term uses the same frozen-at-n state
    # coefficients the implicit solve applies (Pξⁿ, ρ_tⁿ and its FULL vertical
    # gradient), so the grid-scale acoustic cancellation holds at any
    # perturbation amplitude — the point of the state-dependent linearization.
    if sd_si
        FORCING .= @. (-gamma_m * p * div) + (S.sd_pxi * ((rho_t * w_z) + (rho_t_z * w))) +
                      ((R_m / C_vt) * (((Lv - (Rv * C_pt * Tk / R_m)) * (Qdot + Qdot_r)) + QDOT_TH))
    else
        FORCING .= @. (-gamma_m * p * div) + (Pxi_bar * ((rho_tbar * w_z) + (rho_tbar_z * w))) +
                      ((R_m / C_vt) * (((Lv - (Rv * C_pt * Tk / R_m)) * (Qdot + Qdot_r)) + QDOT_TH))
    end
    @turbo expdot[colstart:colend,1] .= @. ADV + FORCING

    # Dry-air mass continuity (slot 2, advective product-rule form; no mass diffusion)
    mc_advect!(ADV, geom, u, w, vv, r, rho_dp_x, rho_d_z, rdv.f_l)
    if sd_si
        @turbo FORCING .= @. (-rho_d * div) + ((rho_d * w_z) + (rho_d_z * w))
    else
        @turbo FORCING .= @. (-rho_d * div) + ((rho_dbar * w_z) + (rho_dbar_z * w))
    end
    @turbo expdot[colstart:colend,2] .= @. ADV + FORCING

    # Total mass continuity (slot 3): the only source is the sedimentation flux
    # divergence, identical to slot 8's so rain and total water cannot drift apart.
    mc_advect!(ADV, geom, u, w, vv, r, rho_tp_x, rho_t_z, rtv.f_l)
    if sd_si
        @turbo FORCING .= @. (-rho_t * div) + ((rho_t * w_z) + (rho_t_z * w)) - Fr_z
    else
        @turbo FORCING .= @. (-rho_t * div) + ((rho_tbar * w_z) + (rho_tbar_z * w)) - Fr_z
    end
    @turbo expdot[colstart:colend,3] .= @. ADV + FORCING

    # u momentum (slot 4): PGF directly from the prognostic pressure
    mc_advect!(ADV, geom, u, w, vv, r, u_x, u_z, uv.f_l)
    mc_u_forcing!(FORCING, geom, pp_x, rho_t, vv, r, fcor)
    if use_smag
        mc_u_kdiff!(KDIFF, geom, K_smag, uv, vv, r)
    else
        mc_u_kdiff!(KDIFF, geom, Khdiff, uv, vv, r)
    end
    @turbo expdot[colstart:colend,4] .= @. ADV + FORCING + KDIFF

    # w momentum (slot 5): perturbation PGF + total-density buoyancy loading, minus the
    # reference-linear PGF -pp_z/ρ̄_t (the acoustic remainder; see the slot-1 comment)
    mc_advect!(ADV, geom, u, w, vv, r, w_x, w_z, wv.f_l)
    if sd_si
        # With ρ̂_t = ρ_tⁿ the linear PGF −pp_z/ρ̂_t cancels the full
        # perturbation PGF exactly: the explicit w remainder is pure buoyancy.
        @turbo FORCING .= @. (-gravity * rho_tp) / rho_t
    else
        @turbo FORCING .= @. (((-gravity * rho_tp) - pp_z) / rho_t) + (pp_z / rho_tbar)
    end
    if use_smag
        mc_w_kdiff!(KDIFF, geom, K_smag, wv, r)
    else
        mc_w_kdiff!(KDIFF, geom, Khdiff, wv, r)
    end
    @turbo expdot[colstart:colend,5] .= @. ADV + FORCING + KDIFF

    # Total energy (slot 6): -v·∇E_t - E_t∇·v - ∇·(pv) - ∇·(E_sed) + Q̇_therm + friction
    # sink. No condensation source (exact first law: phase change is not an energy source).
    # E_sed is the energy carried by falling rain (see the microphysics block above). The
    # THERMAL diffusion heats (Q̇_therm); the momentum diffusion adds FRIC_KE = d(rho_t*ke)/dt
    # so E_t follows the resolved KE down to the subgrid (internal energy held). pbar has no
    # x-dependence so u*p_x = u*pp_x.
    mc_advect!(ADV, geom, u, w, vv, r, E_tp_x, E_t_z, etv.f_l)
    mc_et_work!(FORCING, geom, E_t, p, div, u, pp_x, w, p_z, E_sed_z, pv, vv, r)
    if sd_si
        @turbo expdot[colstart:colend,6] .= @. ADV + FORCING + QDOT_TH + FRIC_KE +
                                               (((E_t + p) * w_z) + ((E_t_z + p_z) * w))
    else
        @turbo expdot[colstart:colend,6] .= @. ADV + FORCING + QDOT_TH + FRIC_KE +
                                               (((E_tbar + pbar) * w_z) + ((E_tbar_z + pbar_z) * w))
    end

    # ── AI2* acoustic history staging ──
    # The linear vertical acoustic operator, evaluated on the CURRENT state with the same
    # discrete chain the semi-implicit solve applies: fit φⁿ = ρ̄_t wⁿ once in w's column
    # basis (the basis the Helmholtz solve works in, Dirichlet rows included), take its
    # spline derivative, and form each slot's tendency with the same coefficient profiles
    # the slaved updates use. Every AI2* time level then sees the SAME discrete operator —
    # the pointwise product-rule staging used previously left the grid-scale difference
    # between the two operators under explicit weights, which imposed a vertical-acoustic
    # Courant ceiling (see reference/SI_VERTICAL_CEILING.md). The single-φ-fit structure also
    # keeps the rho_d and rho_t histories bitwise identical under a dry reference
    # (c_d = 1, c_d_z = 0 exactly), so the two densities cannot drift apart in dry air.
    # Fresh evaluation every step (not stored solve increments) makes the history robust
    # to whatever touches the state between steps (implicit diffusion, nesting collar
    # injection, the spectral refit).
    w_col = scratch_column(mtile, 5)
    # sd_si: φⁿ = ρ_tⁿ·w and every coefficient chain from the CURRENT column
    # state (frozen at n), matching the operator the adjustment's per-column
    # state-dependent solve applies; else the resting-reference profiles.
    if sd_si
        w_col.uMish .= rho_t .* w
    else
        w_col.uMish .= rho_tbar .* w
    end
    Btransform!(w_col)
    Atransform!(w_col)
    phi_n = Itransform!(w_col)
    imp_phi_z = S.imp_phi_z
    Ixtransform(w_col, imp_phi_z)

    imp_c_d = S.imp_c_d
    imp_c_d_z = S.imp_c_d_z
    imp_c_e = S.imp_c_e
    imp_c_e_z = S.imp_c_e_z
    if sd_si
        @. imp_c_d = rho_d / rho_t
        @. imp_c_d_z = ((rho_d_z * rho_t) - (rho_d * rho_t_z)) / (rho_t^2)
        @. imp_c_e = (E_t + p) / rho_t
        @. imp_c_e_z = (((E_t_z + p_z) * rho_t) - ((E_t + p) * rho_t_z)) / (rho_t^2)
        impdot[colstart:colend,1] .= @. -S.sd_pxi * imp_phi_z
    else
        @. imp_c_d = rho_dbar / rho_tbar
        @. imp_c_d_z = ((rho_dbar_z * rho_tbar) - (rho_dbar * rho_tbar_z)) / (rho_tbar^2)
        @. imp_c_e = (E_tbar + pbar) / rho_tbar
        @. imp_c_e_z = (((E_tbar_z + pbar_z) * rho_tbar) -
                        ((E_tbar + pbar) * rho_tbar_z)) / (rho_tbar^2)
        impdot[colstart:colend,1] .= @. -Pxi_bar * imp_phi_z
    end
    impdot[colstart:colend,2] .= @. -((imp_c_d * imp_phi_z) + (imp_c_d_z * phi_n))
    impdot[colstart:colend,3] .= @. -imp_phi_z
    impdot[colstart:colend,6] .= @. -((imp_c_e * imp_phi_z) + (imp_c_e_z * phi_n))
    # w's history is NOT staged here: the Helmholtz elimination's effective operator on
    # the w leg is not expressible as a pointwise chain on the state, so the adjustment
    # stores the increment it actually applied ((w_np1 - w*)/Δτ) as the next step's
    # history — the only exact mirror. Staging -pp_z/ρ̄_t here instead leaves the
    # weak-Galerkin elimination residual under the explicit AI2* weights (vertical
    # Courant ceiling at Co_z ≈ 2-3, measured).

    # ── Horizontal acoustic remainder + AI2* history staging (horizontal SI) ──
    # The horizontal analogue of the vertical staging above (see horizontal_si.jl):
    # the remainder additions cancel the reference-linear horizontal legs out of
    # the AB3 predictor, and the fresh histories are the same legs evaluated on
    # the carried state through the grid slots (u_x IS the spline-chain derivative
    # of the post-refit state, and the reference coefficients are z-only, so no
    # fit and no product-rule chain is needed). u's history is NOT staged here —
    # it is the stored applied increment of the patch-level sweep
    # (horizontal_si_load_increment!, the w-leg discipline).
    if hsi_like
        LDIV = S.ADV                     # free between slot 6 and slot 7
        mc_linear_div!(LDIV, geom, uv, vv, r)
        @turbo expdot[colstart:colend,1] .+= @. Pxi_bar * rho_tbar * LDIV
        @turbo expdot[colstart:colend,2] .+= @. rho_dbar * LDIV
        @turbo expdot[colstart:colend,3] .+= @. rho_tbar * LDIV
        @turbo expdot[colstart:colend,4] .+= @. pp_x / rho_tbar
        @turbo expdot[colstart:colend,6] .+= @. (E_tbar + pbar) * LDIV
        # Fresh history staging only under options[:hsi_x_history] = "fresh"
        # (the A/B alternative): the default "stored" histories are the
        # sweep's applied increments, loaded into hacdot_n at the top of the
        # step (horizontal_si_load_increment!) — writing here would clobber
        # them. See the load function for the measured trade-offs.
        if get(model.options, :hsi_x_history, "fresh") == "fresh"
            hacdot = mtile.hacdot_n
            hacdot[colstart:colend,1] .= @. -Pxi_bar * rho_tbar * LDIV
            hacdot[colstart:colend,2] .= @. -rho_dbar * LDIV
            hacdot[colstart:colend,3] .= @. -rho_tbar * LDIV
            hacdot[colstart:colend,6] .= @. -(E_tbar + pbar) * LDIV
        end
    end

    # Supersaturation density (slot 7): the saturation chain-rule terms use the
    # non-condensation T and p tendencies, which carry the horizontal THERMAL diffusive
    # heating as well as the divergence work (friction holds T, so it does not enter; the
    # VERTICAL diffusive heating is applied as a slaved δQ_ss inside diffusion_timestep_mc).
    # The condensation contribution is -Q̇_cond(1+Q_s) (= -Q_ss/τ when the rate is unlimited).
    # QSSREL reconciles the prognostic Q_ss with the vapor the water masses actually imply
    # (see qss_relaxation) — Q_ss is redundant now that rho_c is prognostic, and this is
    # what keeps the two from drifting apart under splitting error.
    #
    # IT IS FED `res_rho_t`, NEVER `rho_v`. Under `:residual` the two are the same array
    # contents so this is bitwise the old line; under `:blend` it is the one place the blended
    # vapor must not be used, because `-(Q_ss - ((Q_ss + rho_vs) - rho_vs))/tau ≡ 0` — the term
    # would SELF-ANNIHILATE wherever the blend fully trusts res_qss, i.e. exactly where the
    # anchor to the water masses is load-bearing. Keeping it on the density budget is also what
    # bounds the blend's partition gap, by the identity
    #     res_qss - res_rho_t = Q_ss - (rho_v - rho_vs) = -tau_qss * QSSREL,
    # so the disagreement between the two representations is tau_qss times the rate at which
    # this term is removing it.
    dT_nc = S.dT_nc; @. dT_nc = ((-p * div) + QDOT_TH) / (rho_d * C_vt)
    dp_nc = S.dp_nc; @. dp_nc = (-gamma_m * p * div) + ((R_m / C_vt) * QDOT_TH)
    SATF = S.SATF;   @. SATF = (-rho_vs * div) - (drvs_dT * dT_nc) - (drvs_dp * dp_nc)
    QSSREL = S.QSSREL
    @. QSSREL = qss_relaxation(Q_ss, res_rho_t, rho_vs, tau_qss)
    mc_advect!(ADV, geom, u, w, vv, r, Q_ssp_x, Q_ss_z, qsv.f_l)
    FORCING .= @. (-Q_ss * div) + SATF - ((Qdot + Qdot_r) * (1.0 + Q_s)) + QSSREL
    @turbo expdot[colstart:colend,7] .= @. ADV + FORCING

    # Rain partial density (slot 8): rain-channel condensation/evaporation,
    # autoconversion + collection from cloud, and the sedimentation flux divergence
    # (no diffusion yet — water-species mixing arrives with the moist diffusion).
    #
    # Under `rain_transform_mode`, exactly as for slot 9: advection is transform-invariant and
    # reads the control variable's own fitted gradients (`nu_r_z` is bitwise `rho_rp_z` under
    # `:none`), while divergence and the sources pick up `Jr`, an exact 1.0 under `:none`.
    #
    # The sedimentation term is the one that has no cloud precedent. `-Fr_z` is a DENSITY flux
    # divergence, formed from the recovered density on slot 8's own spline column and BCs, and
    # slot 3 (rho_t) and slot 6 (E_t) below continue to receive it untransformed — they are
    # still a density and an energy. Only this slot, which is no longer a density, carries it
    # through `Jr`. That is correct term by term, but it does mean slot 8 and slot 3 stop
    # receiving the identical discrete number, so the exact telescoping between them is
    # weakened. The independent check is already in the diagnostics: `accum_rainfall_mm` comes
    # from the rho_t - rho_d water path and `accum_rainfall_flux_mm` from this surface flux.
    mc_advect!(ADV, geom, u, w, vv, r, rho_rp_x, nu_r_z, rrv.f_l)
    @turbo FORCING .= @. Jr * ((-rho_r * div) + Qdot_r + AUTO_COLL - Fr_z)
    @turbo expdot[colstart:colend,8] .= @. ADV + FORCING
    # Production attribution (off unless options[:water_budget_trace] > 0). Must run HERE:
    # ADV is reused by slot 9 two lines down.
    budget_trace && water_budget_probe!(mtile, MC_BUDGET_R, 8, t, colstart, rho_r, ADV, div,
                                        Qdot_r, AUTO_COLL, 1.0, Fr_z, w, z)

    # Cloud partial density (slot 9): the same advective product-rule continuity as every
    # other density, with the cloud-channel condensation as its source and autoconversion
    # + collection as its sink (the equal and opposite pair of slot 8's AUTO_COLL). No
    # sedimentation — cloud droplets do not fall in this scheme — and no rho_t source:
    # condensation is INTERNAL to the water, moving mass between this slot and the
    # residual vapor at fixed total density.
    #
    # This is the slot the whole formulation exists for. Because it is prognostic rather
    # than a residual of rho_t - rho_d - rho_v - rho_r, cloud in subsaturated air can only
    # arrive by advection or nucleation, and fit-level error in the density fields lands
    # in the vapor (where it is 5e-5 of the field) instead of manufacturing condensate
    # (where it was O(1) of the field). See reference/HANDOFF_DIAGNOSED_CLOUD.md.
    # ADVECTION IS TRANSFORM-INVARIANT: u·∇ρ_c = f'(n) u·∇n, and the tendency being formed is
    # dn/dt = J·(dρ_c/dt), so the advective part is just −u·∇n. It therefore reads the control
    # variable's OWN spline gradients — no chain rule, no f'', no reference-gradient round
    # trip. Under `:none`, `nu_c_z` is bitwise the old `rho_c_z` and `rho_cp_x` is unchanged.
    #
    # The rest picks up the Jacobian: dn/dt = J(ρ_c)·[−ρ_c ∇·u + Qdot − AUTO_COLL]. `Jc` is
    # exactly 1.0 under `:none`, and multiplication by 1.0 is the identity in IEEE, so the
    # default path is bit-for-bit the code that had no transform.
    mc_advect!(ADV, geom, u, w, vv, r, rho_cp_x, nu_c_z, rcv.f_l)
    @turbo FORCING .= @. Jc * ((-rho_c * div) + Qdot - AUTO_COLL)
    @turbo expdot[colstart:colend,9] .= @. ADV + FORCING
    # Cloud has no sedimentation channel, and its autoconversion sink is -AUTO_COLL.
    budget_trace && water_budget_probe!(mtile, MC_BUDGET_C, 9, t, colstart, rho_c, ADV, div,
                                        Qdot, AUTO_COLL, -1.0, nothing, w, z)

    # ── Horizontal water-species mixing (Khdiff_water; 0.0 = OFF, the default) ──
    #
    # *** DIAGNOSTIC / TEMPORARY -- NOT ENERGY CONSISTENT. READ BEFORE USING. ***
    #
    # Why it exists: an axisymmetric domain has no asymmetries to provide radial
    # mixing, so the sharp moisture gradients a TC develops are damped by nothing at
    # all -- until now the water species had NO horizontal diffusion whatsoever
    # (slot 8's "no diffusion yet" note), while momentum had Smagorinsky. That
    # leaves grid-scale structure in the water field free to grow.
    #
    # What is wrong with it: this is a bare K*grad^2 on the water species. Diffusing
    # water MASS without transporting the internal energy and latent heat that mass
    # carries is exactly the moist coupling deferred in
    # reference/moist_compressible_diffusion_handoff.md. The proper form is a product
    # rule through the retrieval's T sensitivities, sourcing E_t and p alongside the
    # mass (the `M = p + E_t - rho_t(ke+gz)` enthalpy invariant: any transport must
    # source BOTH). Because this does not, it WILL drift energy -- watch
    # conservation_drift. It is a noise control to obtain a stable run, not physics,
    # and it must be replaced by the full moist form before any production science.
    #
    # Khdiff_water < 0 selects the Smagorinsky K_smag/Sc_t (the Khdiff_heat sentinel
    # convention); > 0 is a constant diffusivity. Same K*grad^2 (not div(K grad))
    # approximation as the momentum and heat terms.
    if Khdiff_water != 0.0
        # UNDER A TRANSFORM THIS MIXES THE CONTROL VARIABLE, AND THAT IS THE INTENDED FORM.
        # The Laplacian is applied to the slot, so with `condensate_transform`/`rain_transform`
        # on it smooths ν rather than ρ. That is not an approximation to `K∇²ρ` — above the
        # knee it IS `K∇²ρ`, because `bhyp` is exactly affine there:
        #
        #     bhyp(ρ) = (ρ + μ)/2 − μ²/(2(ρ + μ))        (algebra, not an expansion)
        #
        # so `∇²ν = ∇²ρ/2 + O(μ²)` and `f'(ν) = 2 + O(μ²/ρ²)`, and the implied density rate
        # `f'(ν)·K∇²ν` agrees with `K∇²ρ` to relative O(μ²/ρ²). Measured on a Gaussian cloud at
        # μ = 1e-7: 3.7e-9 at ρ_c = 2.3e-3, 2.6e-7 at 3.2e-4, 6.2e-4 at 5.8e-6, and only
        # reaching 7.2e-2 at 3.7e-7 kg/m³ — a thousandth of the smallest meaningful cloud.
        #
        # WHY NOT THE EXACT CHAIN RULE `J·K·(f''(ν)|∇ν|² + f'(ν)∇²ν)`. Because `f''(0) = 1/μ`:
        #
        #     f'(ν)  = 2(ρ+μ)² / ((ρ+μ)² + μ²)          ∈ [1, 2]
        #     f''(ν) = 8μ²(ρ+μ)³ / ((ρ+μ)² + μ²)³       → 1/μ = 1e7 as ρ → 0
        #
        # and the region where that term matters is `μ/|∇ρ| ≈ 1.7 cm` wide against a 500 m
        # cell. Its magnitude there is `K·f''·|∇ν|²` ≈ 0.036 kg/m³/s at ρ = μ and 0.148 at
        # ρ = 1e-8 — twelve to fifty times the entire cloud amplitude, per second. Analytically
        # it is cancelled by `f'∇²ν`; discretely both come from a filtered, ringing spline fit
        # at the one place the fit is worst, and nothing makes that cancellation survive. It is
        # not a smoothing operator, it is a pointwise sample of an unresolvable knee, so it is
        # deliberately NOT offered as an option — a branch that is unrunnable at the cloud edge
        # is dead code that will be believed. (This also retires the stated reason for keeping
        # `:bhyp_smooth` in reference/FINDINGS_CONDENSATE_STAGE1.md — "for any future consumer
        # that needs f'' at the cloud edge, the water Laplacians". They do not need it.)
        #
        # Two consequences worth stating rather than leaving to be rediscovered:
        #   * slot 3 (total water) is untransformed and keeps mixing in DENSITY space while
        #     slots 8/9 mix in ν space, so the vapor residual absorbs the difference. That is
        #     O(μ²/ρ²) above the knee — not a new inconsistency class.
        #   * positivity survives whatever the mixing does, because it is a property of the
        #     recovery, not of the operator: ν monotone in ρ and `ahyp` floored at 0 (`:bhyp`)
        #     or −μ (`:bhyp_smooth`). The coefficient limiter this replaced could not say that
        #     across a nest interface at all.
        #
        # Precedent inside the model: the Galerkin low-pass `l_q` (2.0 by default) and the
        # spline filter are ALREADY applied to the slot, i.e. already smooth ν directly, and
        # the whole transform validation ladder was run with them on.
        WLAP = S.KDIFF                    # both free again after slot 9
        WLAP2 = S.FORCING
        # ACCUMULATED THROUGH AN EXPLICIT LOOP, not `expdot[colstart:colend,s] .+= …`.
        # `.+=` on an indexed expression expands to `A[r] = A[r] .+ B`, and the READ
        # materializes the slice: this block was allocating four times per column (measured;
        # it is the only mc term that did, and it was never in the allocation gate because
        # Khdiff_water is 0.0 in every shipped configuration). Same convention as the Louis
        # BL's apply loop. It is also the only additive mc block, which is why `.=`
        # everywhere else was fine.
        # The two branches stay SEPARATE and each carries its own loops: hoisting the
        # diffusivity into one variable would give it type `Union{Vector{Float64},Float64}`
        # and box every use — the same trap the state-dependent SI hit.
        if Khdiff_water < 0.0
            use_smag || error("physical_params[:Khdiff_water] < 0 selects the " *
                              "Smagorinsky water diffusivity, which requires Ls > 0")
            # Total water rho_w' = rho_t' - rho_d' (dry air is NOT mixed: only the
            # water rides on rho_t here, so rho_d's own tendency is untouched).
            mc_w_kdiff!(WLAP,  geom, K_smag, rtv, r)
            mc_w_kdiff!(WLAP2, geom, K_smag, rdv, r)
            @inbounds for i in eachindex(WLAP)
                expdot[colstart + i - 1, 3] += (WLAP[i] - WLAP2[i]) / Sc_t
            end
            # Slots 8 and 9 are the transformed ones: under a transform `rrv`/`rcv` are ν,
            # and `K∇²ν` is the intended operator (see the block comment above). No
            # Jacobian — the tendency is already in the slot's own units.
            mc_w_kdiff!(WLAP, geom, K_smag, rrv, r)
            @inbounds for i in eachindex(WLAP)
                expdot[colstart + i - 1, 8] += WLAP[i] / Sc_t
            end
            mc_w_kdiff!(WLAP, geom, K_smag, rcv, r)
            @inbounds for i in eachindex(WLAP)
                expdot[colstart + i - 1, 9] += WLAP[i] / Sc_t
            end
            mc_w_kdiff!(WLAP, geom, K_smag, qsv, r)
            @inbounds for i in eachindex(WLAP)
                expdot[colstart + i - 1, 7] += WLAP[i] / Sc_t
            end
        else
            mc_w_kdiff!(WLAP,  geom, Khdiff_water, rtv, r)
            mc_w_kdiff!(WLAP2, geom, Khdiff_water, rdv, r)
            @inbounds for i in eachindex(WLAP)
                expdot[colstart + i - 1, 3] += WLAP[i] - WLAP2[i]
            end
            mc_w_kdiff!(WLAP, geom, Khdiff_water, rrv, r)
            @inbounds for i in eachindex(WLAP)
                expdot[colstart + i - 1, 8] += WLAP[i]
            end
            mc_w_kdiff!(WLAP, geom, Khdiff_water, rcv, r)
            @inbounds for i in eachindex(WLAP)
                expdot[colstart + i - 1, 9] += WLAP[i]
            end
            mc_w_kdiff!(WLAP, geom, Khdiff_water, qsv, r)
            @inbounds for i in eachindex(WLAP)
                expdot[colstart + i - 1, 7] += WLAP[i]
            end
        end
    end

    # Tangential momentum (slot 9, cylindrical geometries only — no method body
    # executes on the Cartesian slice): advection, azimuthal PGF (3D), Coriolis +
    # curvature -u(f + v/r), and the λ-component of the cylindrical vector Laplacian.
    if use_smag
        mc_v_tendency!(expdot, geom, colstart, colend, S, u, w, uv, vv, pv, rho_t, r,
                       fcor, K_smag)
    else
        mc_v_tendency!(expdot, geom, colstart, colend, S, u, w, uv, vv, pv, rho_t, r,
                       fcor, Khdiff)
    end
    # Under the exact unsplit acoustic SI, cancel v's reference-linear azimuthal
    # PGF from the AB3 remainder (RLR only). The v-leg AI2* history is the STORED
    # applied increment (loaded in phase A, exact_si_load_history!) — the weak
    # operator the unsplit solve actually applied, self-consistent per §8; a fresh
    # pointwise −(1/ρ̄_t r)∂λp′ history would mismatch the weak solve (ε-chain).
    if hsi_like
        mc_stage_v_acoustic!(expdot, geom, colstart, colend, pv, rho_tbar, r)
    end

    # ── Rayleigh sponge (momentum-only) ──
    # Klemp-Durran absorbing layer against gravity-wave reflection off the rigid lid:
    # u and w (and v on the cylinders) relax toward the resting base state with the
    # (negative) coefficient ray(z), and E_t follows the resolved KE down exactly as
    # for friction (the FRIC_KE invariant: dE = rho_t*(u*ray*u + w*ray*w) =
    # 2*rho_t*ray*ke, with ke already carrying v on the cylinders). T, p and
    # Q_ss are held — a KE sink to the sponge, not diabatic heating. Explicit loop,
    # not `.+=` view broadcasts: the SubArray stops eliding at this function size
    # (the dE_w lesson). Behind `alpha > 0` so disabled configs are bit-identical
    # (an unconditional `+ 0.0` term could flip -0.0 tendencies).
    mc_sponge!(expdot, geom, colstart, alpha, z_damp, z, u, w, vv, rho_t, ke)

    # ── Louis boundary layer (explicit vertical mixing + surface drag) ──
    # Added AFTER every slot is written: all contributions are additive (the Q_ss
    # saturation chain rule is linear), so the lines above stay frozen and
    # louis_bl = false is bit-identical. See mc_boundary_layer.jl.
    if louis_bl
        if ctrans_on
            # The BL's cloud eddy flux is a MASS flux, so it has to be built from the
            # perturbation DENSITY gradient, not from the slot's. `rho_c_z` (line ~2703) is the
            # total ∂z ρ_c = ∂z ν / J, and this is its first consumer — it was written for one.
            #
            # Staged HERE and not inside `mc_louis_bl!` because `rho_cbar_z` is a SubArray, and
            # handing it to a @noinline callee is an escape that boxes once per column; the
            # zero-allocation gate in test/test_allocations.jl exists for exactly that.
            #
            # NOT used on the `:none` path: there `rho_c_z` is `rho_cp_z + rho_cbar_z` and
            # subtracting `rho_cbar_z` back off is not bitwise `rho_cp_z`. The callee keeps
            # reading `rcv.f_z` directly when the transform is off, which is.
            @. S.bl_rho_cp_z = rho_c_z - rho_cbar_z
        end
        mc_louis_bl!(mtile, S, geom, colstart, colend, z, uv, wv, vv, rtv, rdv, rcv,
                     expdot, l_inf, Cd_param, sfc_wind_factor,
                     surface_fluxes, Ck, SST, U_min, ctrans_on)
    end

    # ── Implicit vertical diffusion tendencies (AI2* history in the diffdot channel) ──
    # Vertical diffusion must be implicit on a Chebyshev column, where the spectral
    # second-derivative eigenvalues scale as N^4. The acoustic solver owns impdot[w],
    # impdot[p] and impdot[E_t], so these live in diffdot instead. Slot 6 (E_t) carries the
    # HEAT tendency in entropy space (not an energy tendency): diffusion_timestep_mc solves
    # for s_t' and maps the increment onto (p, E_t, Q_ss). Slots 3/8/9 carry the water
    # tendencies (total water rho_w' = rho_t' - rho_d', rain rho_r, cloud rho_c').
    #
    # The heat variable is the full moist entropy s_t (vapor + liquid contribution), a
    # retrieval-dependent diagnostic; the resting base is still bit-preserved because
    # s_tbar comes from mc_reference_diagnostics — the SAME retrieval pipeline — so at
    # rest s_t' == 0 exactly. (The horizontal heat diffusion keeps the dry-exact s_d
    # chain rule: transforming a diagnosed field horizontally needs halo machinery the
    # column decomposition doesn't have; see the handoff doc.)
    if Kvdiff > 0.0 || Kvdiff_heat > 0.0 || Kvdiff_water > 0.0
        diffdot = mtile.diffdot_n
        if Kvdiff > 0.0
            @turbo diffdot[colstart:colend,4] .= @. Kvdiff * u_zz
            @turbo diffdot[colstart:colend,5] .= @. Kvdiff * w_zz
            mc_v_diffdot!(diffdot, geom, colstart, colend, Kvdiff, vv)
        end
        if Kvdiff_heat > 0.0
            s_t = S.s_t
            @. s_t = moist_entropy_total(Tk, rho_d, q_v, q_l)
            # ∂zz(s_t') from the column basis, so the explicit AI2* tendency and the implicit
            # Helmholtz operator use the same discrete ∂zz.
            s_col = scratch_column(mtile, 6)
            s_col.uMish .= s_t .- mtile.mc_ref_diag.s_tbar
            Btransform!(s_col)
            Atransform!(s_col)
            stage_zz = S.stage_zz
            Ixxtransform(s_col, stage_zz)
            @turbo diffdot[colstart:colend,6] .= Kvdiff_heat .* stage_zz
        end
        if Kvdiff_water > 0.0
            # EVERY diffused water species is now a combination of prognostic slots, so
            # every ∂zz comes straight from the grid's derivative slots: total water
            # rho_w' = rho_t' - rho_d', cloud, rain. The vapor increment is implied
            # (delta_rho_v = delta_rho_w - delta_rho_c - delta_rho_r), which is what
            # retires the column transform of the DIAGNOSED rho_v this block used to
            # need — a fit of a diagnosed field, with the reference-profile subtraction
            # required to keep the resting base quiet.
            @turbo diffdot[colstart:colend,3] .= @. Kvdiff_water * (rho_tp_zz - rho_dp_zz)
            @turbo diffdot[colstart:colend,8] .= @. Kvdiff_water * rho_rp_zz
            @turbo diffdot[colstart:colend,9] .= @. Kvdiff_water * rho_cp_zz
        end
    end

    # Depletion census of the water species. HERE, not next to the tendency assembly: this is
    # the last point at which `expdot` is still the tendency of the step about to be taken and
    # `expdot_nm1`/`expdot_nm2` are still the previous two levels (`explicit_timestep` rotates
    # them), so the increment measured is exactly the one applied — including the horizontal
    # water mixing and the boundary-layer contributions added after slot 8/9 were assembled.
    budget_trace && water_depletion_probe!(mtile, colstart, colend, t, precipitation,
                                           rho_c, rho_r, rho_v, res_rho_t, Qdot, Qdot_r,
                                           AUTO_COLL, cap_c, cap_r, cap_v)

    # Advance the explicit terms
    explicit_timestep(mtile, colstart, colend, t)
    # ...and rotate the microphysics sink history with it, so the next step's depletion
    # budgets see the same two levels the integrator will weight. Placed HERE, not further
    # down, because the exact_si branch returns from this function below: this is the last
    # point common to both paths, and it is reached exactly once per column per step.
    _rotate_micro_history!(mtile, colstart, colend)

    # Explicit AI2* history levels of the HORIZONTAL acoustic legs: both
    # dimensions' history levels belong to the star state before either implicit
    # solve (the ADI factorization applies the vertical solve below first, then
    # the horizontal patch-level sweep after the spectral merge; the exact_si
    # unsplit solve wants ALL explicit levels inside X* before it runs).
    if hsi_like
        horizontal_si_history!(mtile, colstart, colend, t)
    end

    # Exact (unsplit) 2-D semi-implicit: phase A ends here. The vertical AI2*
    # explicit history levels are applied now (the patch-level solve must see
    # the complete X*); the implicit solve and everything after it happen in
    # exact_si_apply_column! (phase B) once the patch solve has run.
    if xsi
        apply_acoustic_histories!(mtile, colstart, colend, t)
        return
    end

    # Semi-implicit (p', ρ̄_t w) acoustic solve — unconditional: the explicit acoustic
    # mode was removed (expdot carries only the remainder; the linear vertical acoustic
    # terms are integrated here and nowhere else).
    semiimplicit_adjustment_p(mtile, colstart, colend, t)

    # Implicit vertical diffusion of u, w (friction sink), the moist entropy s_t' (heat,
    # slaved onto p, E_t, Q_ss) and the water species (rho_w', rho_c', rho_r). Skipped
    # when every coefficient is zero: a vertical solve is not the identity there — it
    # refits the column and reapplies the spectral filter.
    if Kvdiff > 0.0 || Kvdiff_heat > 0.0 || Kvdiff_water > 0.0
        diffusion_timestep_mc(mtile, colstart, colend, t, geom)
    end

    # Negative water is not a representable state (see clamp_water!). LAST, so that
    # nothing downstream of it can reintroduce one: the acoustic solve and the vertical
    # diffusion have both had their say by here.
    clamp_water!(mtile, colstart, colend)

end

# ── Name-dispatched equation-set wrappers (physical_model resolves the config's
#    equation_set string to one of these by name; all keep the moist_compressible
#    prefix so uses_pressure_reference gates their scratch/reference plumbing) ──

"Total-energy moist compressible set on a Cartesian XZ slice (RiRk/RZ grid), 9 vars."
moist_compressible_XZ(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64) =
    mc_driver!(mtile, colstart, colend, t, MCCartesianXZ())

"""
Axisymmetric r–z cylinder on the RiRk/RZ grid (gridpoint column 1 reinterpreted as
radius, so the domain must sit at r > 0), with prognostic tangential wind v
(`MC_VARS_CYL`, 10 vars) and optional f-plane rotation (`physical_params[:f]`).
"""
moist_compressible_axisym(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64) =
    mc_driver!(mtile, colstart, colend, t, MCAxisymRZ())

"3D r–λ–z cylinder on the RLR grid (`MC_VARS_CYL`, 10 vars, f-plane rotation optional)."
moist_compressible_RLR(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64) =
    mc_driver!(mtile, colstart, colend, t, MCCylindricalRLR())

"""
3D Cartesian x–y–z box on the RRR grid (`MC_VARS_CYL`, 10 vars: v is the y-wind,
f-plane rotation optional — +f v / −f u with no curvature terms).
"""
moist_compressible_RRR(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64) =
    mc_driver!(mtile, colstart, colend, t, MCCartesianRRR())

"""
3D spherical θ–λ–z shell on the SLR grid (`MC_VARS_CYL`, 10 vars: u is the
θ-ward wind, v the zonal wind). Shallow atmosphere with metric radius
`physical_params[:sphere_radius]` (default Earth) and full latitude-dependent
Coriolis f = 2Ω cosθ from `physical_params[:Omega]` (NOT the cylinders' f-plane
`:f`). See `MCSphericalSLR`.
"""
moist_compressible_SLR(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64) =
    mc_driver!(mtile, colstart, colend, t, MCSphericalSLR())

"""
    qss_relaxation(Q_ss, rho_v, rho_vs, tau)

Reconciliation of the prognostic supersaturation density toward its diagnosed value,
`-(Q_ss - (ρ_v - ρ_vs))/τ` [kg/m³/s].

With the condensate prognostic, the water masses already determine the vapor density
(`ρ_v = ρ_t - ρ_d - ρ_c - ρ_r`), so `Q_ss = ρ_v - ρ_vs(T,p)` is formally REDUNDANT. It is
carried prognostically anyway because it is the quantity the condensation closure reads, and
a prognostic, advected `Q_ss` is smooth where the diagnosed difference of two large nearly
equal numbers is not: the fit-level error that lands in ρ_v (~1e-6 kg/m³) is 5e-5 of ρ_v but
would be a comparable fraction of the supersaturation itself near saturation, right at the
1e-4 nucleation gate.

Redundancy has to be reconciled or the two drift apart under splitting error, and this is
that reconciliation — the "extra adjustment step on a slower timescale than the timestep" of
reference/HANDOFF_DIAGNOSED_CLOUD.md. `τ = physical_params[:tau_qss]` (default 10 s) is
deliberately long compared with the timestep, so the transport keeps the smooth field and
only the accumulated inconsistency is removed.

It is free of any effect on the conserved ρ_d, ρ_t and E_t, and — unlike its predecessor,
which relaxed onto water-mass bounds — it is now thermodynamically inert unconditionally:
[`retrieve_temperature`](@ref) does not read `Q_ss` at all.

**`rho_v` here is ALWAYS the DENSITY-BUDGET residual `ρ_t − ρ_d − ρ_c − ρ_r`, never the
blended vapor of [`vapor_retrieval_blend`](@ref).** Fed the blend, the term reads
`−(Q_ss − ((Q_ss + ρ_vs) − ρ_vs))/τ ≡ 0` wherever the blend fully trusts the supersaturation
residual: it SELF-ANNIHILATES, and the only mechanism tying `Q_ss` to the water masses
disappears exactly where it is load-bearing (the `POSITIVITY=1` run reaches `|Q_ss|/ρ_vs ~ 2e26`
*with* the reconciliation running). Keeping the anchor on the density budget is also what
gives the blend's partition gap its meaning, through the identity

    res_qss − res_rho_t = Q_ss − (ρ_v − ρ_vs) = −τ · QSSREL,

i.e. the disagreement between the two representations of the vapor is τ times the
reconciliation rate, and the reconciliation is what removes it.

**That relation is an IDENTITY, not a bound, and it was read as one for a while.** It says the
gap equals `τ·QSSREL`; it says nothing about the size of either, and in particular nothing
about the gap RELATIVE to `ρ_vs` — which is the ratio the retrieval's conditioning actually
depends on, and which grows without limit as `ρ_vs` collapses in a rising anvil (measured 0.46
on the NOPRECIP storm). The blend's bound on that ratio is `dcap`, in
[`vapor_retrieval_blend`](@ref); this identity is not a substitute for it.
"""
@inline function qss_relaxation(Q_ss, rho_v, rho_vs, tau)

    return -(Q_ss - (rho_v - rho_vs)) / tau
end

"""
    _blend_smoothstep(x)

The unit smoothstep `x²(3 − 2x)` on a clamped argument: `0` for `x ≤ 0`, `1` for `x ≥ 1`, C¹
across both ends because the cubic's derivative `6x(1−x)` vanishes there.

Returns EXACTLY `0.0` and EXACTLY `1.0` outside the band (the clamp makes both flat regions
bit-exact, not merely accurate), which is what lets [`vapor_retrieval_blend`](@ref) degenerate
to a pure copy of one operand in each regime limit. The clamp is also load-bearing on the low
side for a physical reason: spline ringing puts `rho_liq` a few times `1e-9` BELOW zero in
clear air routinely, and an unclamped cubic would turn that into a NEGATIVE weight — an
extrapolation past the density residual, not a blend.
"""
@inline function _blend_smoothstep(x)

    y = clamp(x, 0.0, 1.0)
    return y * y * (3.0 - 2.0 * y)
end

"""
    _vapor_blend_weight(rho_liq, s, l0, l1, t0, t1)

Weight `w ∈ [0,1]` given to the SUPERSATURATION residual `res_qss = Q_ss + ρ_vs` against the
DENSITY-BUDGET residual `res_rho_t = ρ_t − ρ_d − ρ_c − ρ_r` in [`vapor_retrieval_blend`](@ref):

    w = w_cloud(ρ_liq; l0, l1) · w_trust(s; t0, t1),        s = |Q_ss| / ρ_vs

with `w_cloud` a smoothstep UP (0 at `ρ_liq ≤ l0`, 1 at `ρ_liq ≥ l1`) and `w_trust` a
smoothstep DOWN (1 at `s ≤ t0`, 0 at `s ≥ t1`).

# Why CLOUDINESS is the selector, and `s` is not

`s` is exactly the conditioning number of the `res_qss` route — the ratio of the difference to
the terms being differenced — so it is the natural candidate, and it was swept first. **The
measurement refutes it** (benchmarks/vapor_blend_diagnostic.jl, first sweep):

- `res_qss < 0` is ALGEBRAICALLY `s > 1` (it says `Q_ss < −ρ_vs`), and the measured minimum `s`
  over the `res_qss`-negative points is `1.0000` on both runs. So no band with `s1 ≤ 1` can
  import a single `res_qss` negative, and every band that does anything is pushed up against
  `s = 1` — precisely where `res_qss` is WORST conditioned. An `s`-only selector is not
  choosing a representation, it is acting as a **soft clamp on negative vapor**, and negative
  water is a RESOLUTION DIAGNOSTIC that must not be clamped
  (reference/FINDINGS_NEGATIVE_WATER_ATTRIBUTION.md).
- The negatives are a 1–2 % tail sitting against that same ceiling in BOTH regimes (G2: every
  `res_rho_t < 0` point has `s ≥ 0.9993`; NOPRECIP: `s ∈ [0.876, 1.057]`), so the regime
  contrast the HANDOFF measured (in-cloud mean `s` 0.081 vs dry 0.73) is a statement about the
  MEANS and the tail does not follow it. On the assigned grid the `s`-only blend reproduces the
  `res_rho_t` counts and minima to the last digit: an exact no-op.

`ρ_liq` does what `s` cannot, because the two populations are separated by ~**8 decades** in it:
NOPRECIP's 719 rescued points have median `ρ_liq = 3.95e−3`, its 425 harmed points `2.3e−14`,
and G2's 1017 harmed points all sit at `|ρ_liq| < 3e−9`. Every band on the whole `(l0,l1,t0,t1)`
grid then gives G2's shipped behaviour unchanged (750 negatives, min −8.2546e−05, all dry) with
NOPRECIP's `res_qss` in-cloud count and minimum EXACTLY (474 negatives, min −8.6932e−07). That
insensitivity to every threshold is what a real regime separation looks like, and it is why the
defaults `l0 = 1e−6`, `l1 = 1e−4 kg/m³` are not tuned numbers.

# What `w_trust` is for, and why `t0 = 2` and not `t0 ≤ 1`

`w_trust` is NOT a second selector, and — as the NOPRECIP detonation established — it is not
the detachment guard either. Its ONE retained job is the GROSS-detachment exact zero: a `Q_ss`
that has left the density budget by orders of magnitude, which does happen (the `POSITIVITY=1`
run reaches `s ≈ 2e26`). Above `t1` the weight is IDENTICALLY zero (bitwise, via
[`_blend_smoothstep`](@ref)'s clamp), so such a point is handed back to the density residual
with no special-case logic and no `isfinite` test anywhere.

**It does not see RELATIVE detachment, and must not be asked to.** `s` is the supersaturation
magnitude, not `|res_qss − res_rho_t|/ρ_vs`: on the NOPRECIP storm the latter reached 0.46
while `s` stayed in 0.02–0.46, so `w_trust ≡ 1` and the guard fired on ZERO points in 3600 s —
and during the growth phase `s` is ANTI-CORRELATED with detachment. That protection is
`dcap`, the C¹ correction cap of [`vapor_retrieval_blend`](@ref).

The thresholds sit deliberately ABOVE the `s = 1` ceiling. Anything at or below 1 would make
`w_trust` a SIGN selector, and the blend would become the soft clamp above. With `t0 = 2` the
`res_qss` negatives at `s ∈ (1, t0]` are imported at FULL cloud weight, which is exactly the
property that keeps this a choice between two representations of the vapor rather than a
correction to one of them.

# C¹, not merely continuous

`ρ_v` feeds `q_v`, hence `C_vt`, `R_m`, `C_pt` and `γ_m`, hence the acoustic coefficient
`γ_m p/ρ_t` the semi-implicit solve linearizes about. A hard regime switch would put a JUMP in
the sound speed across a surface moving through the flow; a C⁰-only blend would put a kink in
it. Both smoothsteps are flat at both ends, so the composite has a continuous gradient across
all four thresholds.

Thresholds come from `physical_params[:vapor_blend_l0/_l1/_t0/_t1]`.
"""
@inline function _vapor_blend_weight(rho_liq, s, l0, l1, t0, t1)

    return _blend_smoothstep((rho_liq - l0) / (l1 - l0)) *
           _blend_smoothstep((t1 - s) / (t1 - t0))
end

"""
    _blend_saturate(c, m)

The C¹ SATURATOR applied to the blend's correction `c = w·(res_qss − res_rho_t)` in
[`vapor_retrieval_blend`](@ref), with cap `m = dcap·ρ_vs`. Odd, monotone, C¹ everywhere,
EXACTLY the identity on the inner half of the band, and bounded by `m`:

    f(c) = c                                   for |c| ≤ a
    f(c) = sign(c)·(m − a²/|c|)                for |c| > a,        a ≡ m/2

# Derivation

The requirements are: identity for small `|c|` (the corrections that are the actual fix must
pass through BIT-EXACT, not merely accurately — median 1.5e−3·ρ_vs, an order of magnitude
under the cap), a hard bound `|f| ≤ m`, monotone, odd, and a CONTINUOUS DERIVATIVE at the
junction so that `γ_m` — hence the acoustic coefficient the semi-implicit solve linearizes
about — has no kink. Take the identity out to `|c| = a` and a rational Huber-style tail `g(x) = m − k/x`
beyond it. Matching value and slope at `x = a`,

    g(a)  = m − k/a  = a      ⟹  k = a(m − a)
    g'(a) = k/a²     = 1      ⟹  k = a²

which are consistent iff `a = m/2`, and then `k = a² = m²/4`. The junction is therefore not
a free parameter: it is FORCED to half the cap by C¹ alone.

# Properties (all asserted in test/test_moist_compressible.jl)

* `f(c) = c` exactly for `|c| ≤ m/2` — no rounding, the argument is returned.
* `f'(c) = a²/c² > 0` for `|c| > a`, and `f'(a⁻) = f'(a⁺) = 1`: monotone and C¹.
* `|f| ≤ m` always, and `< m` strictly in exact arithmetic — the cap is a SUPREMUM approached
  asymptotically (`f → m − m²/(4|c|)`), so a saturated point is not pinned at a corner. In
  floating point, `|c| ≳ m/(4ε)` puts the subtrahend under one ulp of `m` and the tail rounds
  onto the cap; that is the correct rounding of the supremum, and the bound still holds.
* Odd: `f(−c) = −f(c)`, so the cap cannot bias the sign of the correction and therefore
  cannot rectify negative vapor into positive (see [`vapor_retrieval_blend`](@ref)).
* `m = Inf` (i.e. `dcap = Inf`) returns `c` for every finite `c`: the uncapped blend, bitwise.
* `m = 0` returns `±0.0`: the density residual, i.e. `dcap = 0` degenerates to `:residual`.

Contrast the C⁰ clamp this replaces (`clamp(c, −m, m)`), which was the diagnostic form: it has
the same bound but a DISCONTINUOUS derivative at `|c| = m` and a dead zone beyond it, so a
point sitting on the cap contributes nothing to `∂ρ_v/∂(state)` and the acoustic coefficient
acquires a kink on a surface moving through the flow — the same defect the smoothsteps of
[`_blend_smoothstep`](@ref) exist to avoid.
"""
@inline function _blend_saturate(c, m)

    a = 0.5 * m
    abs(c) <= a && return c
    return copysign(m - (a * a) / abs(c), c)
end

"""
    vapor_retrieval_blend(Q_ss, rho_vs, rho_liq, res_rho_t, l0, l1, t0, t1, dcap)

The regime-blended vapor density [kg/m³], under `options[:vapor_retrieval] = :blend`:

    s   = |Q_ss| / ρ_vs
    w   = _vapor_blend_weight(ρ_liq, s, l0, l1, t0, t1)
    c   = w·((Q_ss + ρ_vs) − res_rho_t)                    the CORRECTION
    ρ_v = res_rho_t + _blend_saturate(c, dcap·ρ_vs)

This is the DEFAULT retrieval (since 2026-07-29): every configuration that does not set the key
takes this path. `res_rho_t = ρ_t − ρ_d − ρ_c − ρ_r` is the density-budget residual — the
pre-blend route, and what `options[:vapor_retrieval] = :residual` returns unconditionally, kept
as the bitwise A/B lever against it. See the file header and
[`_vapor_blend_weight`](@ref) for the two-regime measurement this exists to answer, and
reference/HANDOFF_VAPOR_RETRIEVAL.md for the sweeps.

# The cap is the load-bearing guard, and `w_trust` is not

`dcap` (`physical_params[:vapor_blend_dcap]`, default **0.02**) bounds the correction RELATIVE
TO THE SATURATION DENSITY, `|ρ_v − res_rho_t| ≤ dcap·ρ_vs`. It is not a safety belt: the
uncapped blend DETONATES the `SCYTHE_O01_PRECIP=0` storm, and the cap is what the measurement
says fixes it.

The mechanism is a MISSING GUARD, not a bad weight. The partition gap `res_qss − res_rho_t` is
a property of the `Q_ss` splitting and is identical in a `:blend` run and a `:residual` run —
about `1e−4 kg/m³` in absolute terms, which is the number every sweep in the HANDOFF reports.
But it is UNBOUNDED RELATIVE TO `ρ_vs`, and `ρ_vs` is not a constant: in the rising NOPRECIP
anvil the air cools, `ρ_vs` collapses, and the measured ratio `(res_qss − res_rho_t)/ρ_vs`
climbs secularly with the storm — 0.08 at t = 1980 s, **0.46** at t = 2040 s, ~0.8 through the
mature phase. Two runs differing only in the last bits of the state detonated at the SAME
step (6828), which is what a threshold crossing looks like and not what noise looks like.

`w_trust` cannot see any of that. Its variable is `s = |Q_ss|/ρ_vs`, the SUPERSATURATION
magnitude — measured `s` at the worst-detached points is 0.02–0.46, so `w_trust ≡ 1` exactly
and the guard fired on ZERO points in 3600 s. Worse, `s` is ANTI-CORRELATED with detachment
during the growth phase. **`s` is not a detachment measure**, and the identity

    res_qss − res_rho_t = −τ_qss · QSSREL

is exactly that — an identity relating the gap to the reconciliation RATE, not a bound on
either. `w_trust` is therefore retained for ONE job only: the GROSS-detachment exact zero
(`s ≈ 1e10` and up; the `POSITIVITY=1` run reaches `s ≈ 2e26`), where it hands the point back
to the density residual bitwise with no `isfinite` test anywhere. The RELATIVE-detachment
protection — the one the anvil needed — is `dcap`.

`dcap = 0.02` was validated causally, not fitted: it touches 0 % of points before t ≈ 1680 s
and about 10 % of active cloud after, the detonation is gone, the stability ladder matches the
`:residual` control, and the MEDIAN correction is untouched (~1.5e−3·ρ_vs, an order of
magnitude below the cap and well inside the identity region `m/2 = 0.01·ρ_vs`). `dcap = Inf` recovers the uncapped blend BITWISE; `dcap = 0` degenerates to
`:residual`.

# Exactness

**The two regime limits are EXACT copies, not approximations.** `w == 0.0` returns
`res_rho_t`, and `w == 1.0` with `|c| ≤ dcap·ρ_vs/2` returns `Q_ss + ρ_vs`, both bit-for-bit,
so a run that never leaves one regime is bitwise the run that had only that representation.
That is what makes `:residual` an A/B lever against the whole attribution campaign rather than
an "agrees to 1e-14" comparison. [`_blend_saturate`](@ref) is the identity on the inner half of
the band precisely so this survives the cap: the corrections that constitute the fix are three
decades under it and pass through untouched.

# It is still not a clamp on the VAPOR

`t0 = 2` sits above the `s = 1` ceiling at which `res_qss` turns negative, so a fully trusted
point can and must import a NEGATIVE vapor; the saturator is ODD, so it cannot rectify the
sign of anything. Flooring `ρ_v` here would destroy the measurement
reference/FINDINGS_NEGATIVE_WATER_ATTRIBUTION.md rests on and manufacture water on the way.
What `dcap` bounds is the DISTANCE BETWEEN THE TWO REPRESENTATIONS, symmetrically, which is a
statement about the discretization and not about the sign of the water.

# What it costs

`ρ_d + ρ_v + ρ_c + ρ_r` no longer equals `ρ_t` pointwise. On the measured snapshots every
in-cloud point acquires a gap (G2, uncapped: 13632 points, max 9.39e−4, p99 1.95e−4, median
1.02e−7 kg/m³). Under the shipped `dcap` that gap is additionally bounded by `dcap·ρ_vs`
wherever the cap is active. [`qss_relaxation`](@ref) — which is fed `res_rho_t`, NEVER this
blend — is what absorbs it. `ρ_t` itself is untouched, so nothing about the conserved water
mass changes.
"""
@inline function vapor_retrieval_blend(Q_ss, rho_vs, rho_liq, res_rho_t, l0, l1, t0, t1,
                                       dcap = 0.02)

    s = abs(Q_ss) / rho_vs
    w = _vapor_blend_weight(rho_liq, s, l0, l1, t0, t1)
    # Branch rather than lean on `1.0*x + 0.0*y == x`: the identity holds for finite operands
    # but a branch is unconditional, says what it means, and costs nothing next to the
    # `rho_v_sat` this runs beside.
    w == 0.0 && return res_rho_t
    c = w * ((Q_ss + rho_vs) - res_rho_t)
    fc = _blend_saturate(c, dcap * rho_vs)
    if fc === c
        # THE IDENTITY REGION, `|c| ≤ dcap·ρ_vs/2` — where the fix actually lives (median
        # correction ~1.5e-3·ρ_vs) and, for `dcap = Inf`, everywhere. Reached with the
        # saturator provably inert, so the uncapped expressions below are bitwise what they
        # always were, regime limits included.
        w == 1.0 && return Q_ss + rho_vs
        return (w * (Q_ss + rho_vs)) + ((1.0 - w) * res_rho_t)
    end
    # The saturating tail. Written as an increment off `res_rho_t` because that is what the
    # cap is a statement about: the DISTANCE between the two representations.
    return res_rho_t + fc
end

# ── Semi-implicit adjustment ───────────────────────────────────────────────────

"""
    semiimplicit_adjustment_p(mtile, colstart, colend, t)

Semi-implicit acoustic solve for the total-energy set — the ONLY integrator of the
reference-linear vertical acoustic terms (the explicit acoustic mode was removed; the
AB3 predictor advances the acoustic remainder). Applies the AI2* explicit history
weights (`-1.0 Lⁿ + 0.75 Lⁿ⁻¹`, staged spline-consistently in `mc_driver!`), then
solves the Helmholtz problem `∂z(Δτ² Pξ̄(z) ∂z φ) - φ` for the vertical mass flux
`φ = ρ̄_t w` from the implicit pair `∂φ/∂t = -∂z p'`, `∂p'/∂t = -Pξ̄(z) ∂z φ`, with
`Pξ̄(z)` the LOCAL reference sound speed squared (`mc_ref_diag.Pxi_prof` — a
domain-mean c̄² is the classic SHB78 reference-state instability above Co_z ≈ 4.5
on a stratified base), and recovers `w = φ/ρ̄_t` and `p' = p'* - Δτ Pξ̄(z) ∂z φ`. The density and energy slots are
slaved to the flux in flux form: `rho_t' -= Δτ ∂z φ` (conserves `∫rho_t'`),
`rho_d' -= Δτ ∂z(ρ̄_d/ρ̄_t φ)`, `E_t' -= Δτ ∂z((Ē_t+p̄)/ρ̄_t φ)`. Because the history
staging mirrors this operator chain exactly, no part of the linear acoustic operator
is left under explicit weights — the scheme has no vertical-acoustic Courant limit.
"""
function semiimplicit_adjustment_p(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64;
        apply_histories::Bool=true)
    # apply_histories = false: the exact_si path applies the AI2* explicit
    # history levels in phase A (apply_acoustic_histories!, so the patch-level
    # solve sees the complete X*); the predictors arriving here already carry
    # them and the history roll has been done. The default (true) is the
    # production vertical-only path, bitwise unchanged.

    vars = mtile.model.grid_params.vars
    p_index = vars["p"]
    rhod_index = vars["rho_d"]
    rhot_index = vars["rho_t"]
    w_index = vars["w"]
    et_index = vars["E_t"]
    ts = mtile.model.ts

    S = @inbounds mtile.mc_scratch[Threads.threadid()]

    # Predictors (copies — they are mutated below, so these must NOT be views onto var_np1)
    # and implicit tendency histories (views).
    p_nstar = S.si_p_nstar;       copyto!(p_nstar, view(mtile.var_np1,colstart:colend,p_index))
    w_nstar = S.si_w_nstar;       copyto!(w_nstar, view(mtile.var_np1,colstart:colend,w_index))
    rhod_nstar = S.si_rhod_nstar; copyto!(rhod_nstar, view(mtile.var_np1,colstart:colend,rhod_index))
    rhot_nstar = S.si_rhot_nstar; copyto!(rhot_nstar, view(mtile.var_np1,colstart:colend,rhot_index))
    et_nstar = S.si_et_nstar;     copyto!(et_nstar, view(mtile.var_np1,colstart:colend,et_index))

    # Reference profiles and the LOCAL sound-speed-squared profile (views/vectors —
    # read-only, loop-invariant; see the staging comment in mc_driver!)
    Pxi_bar = mtile.mc_ref_diag.Pxi_prof
    rho_dbar = view(ref_rho_d(mtile.ref_state),:,1)
    rho_dbar_z = view(ref_rho_d(mtile.ref_state),:,2)
    rho_tbar = view(ref_rho_t(mtile.ref_state),:,1)
    rho_tbar_z = view(ref_rho_t(mtile.ref_state),:,2)
    E_tbar = view(ref_total_energy(mtile.ref_state),:,1)
    E_tbar_z = view(ref_total_energy(mtile.ref_state),:,2)
    pbar = view(ref_pressure(mtile.ref_state),:,1)
    pbar_z = view(ref_pressure(mtile.ref_state),:,2)

    # State-dependent linearization (options[:state_dependent_si]): every
    # coefficient of the implicit pair — Pξ, the mass-flux density ρ̂_t, and
    # the slaved-leg chains — comes from the CURRENT column state (the driver's
    # scratch, filled this column on this thread, frozen at time n), and the
    # Helmholtz matrix is refactorized per column with that Pξ profile. The
    # remainder/history staging in mc_driver! uses the SAME coefficients, so
    # the operator-consistency cancellation holds at finite amplitude — the
    # resting-reference form leaves δ·Co_z of the grid-scale operator explicit
    # in a convective core with state deviation δ (reference/SI_CONVECTIVE_CEILING.md).
    sd_si = get(mtile.model.options, :state_dependent_si, false)::Bool
    if sd_si && mtile.solve_data === nothing
        error("options[:state_dependent_si] requires the cubic B-spline (RiRk) " *
              "vertical — the per-column profile Helmholtz is not implemented " *
              "for the Chebyshev vertical")
    end
    # (Pxi_bar and S.sd_pxi are both Vector{Float64}, so this binding is
    # type-stable; ρ̂ is branched at each use site instead — a `? S.rho_t :
    # rho_tbar` union of Vector and SubArray boxes every broadcast it touches,
    # which showed up as per-column allocations in the flag-OFF path.)
    Pxi_vec = sd_si ? S.sd_pxi : Pxi_bar

    # Off-centered AI2* history terms (Durran & Blossey 2012). The AB3 predictor carries
    # only the acoustic REMAINDER, so the linear terms enter purely here: the explicit
    # history levels -1.0 Lⁿ + 0.75 Lⁿ⁻¹ now, the implicit +1.25 L̃ⁿ⁺¹ through the
    # Helmholtz solve below. impdot is staged in mc_driver! with the same discrete
    # operator chain the solve applies, so every AI2* level sees ONE operator (the old
    # subtract-AB3/add-implicit form mixed pointwise and fitted operators, leaving a
    # grid-scale residual under explicit weights — the vertical-acoustic Courant ceiling
    # of reference/SI_VERTICAL_CEILING.md). The first step is AM2 trapezoidal (+0.5 Lⁿ with the
    # ts_term = 0.5·ts solve); its history seed gives t == 2 the full AI2* weights.
    ts_term = (t == 1) ? 0.5 * ts : 1.25 * ts
    if apply_histories
        for index in (p_index, w_index, rhod_index, rhot_index, et_index)
            nstar = index == p_index ? p_nstar :
                    index == w_index ? w_nstar :
                    index == rhod_index ? rhod_nstar :
                    index == rhot_index ? rhot_nstar : et_nstar
            dot_n = view(mtile.impdot_n,colstart:colend,index)
            dot_nm1 = view(mtile.impdot_nm1,colstart:colend,index)
            if (t == 1)
                nstar .= @. nstar + (ts * 0.5 * dot_n)
            else
                nstar .= @. nstar + (ts * ((0.75 * dot_nm1) - dot_n))
            end
            dot_nm1 .= dot_n
        end
    end

    # Take the vertical derivative of the p' predictor (coefficient 1: the pair is
    # ∂φ/∂t = -∂z p'; Pxi_bar enters in the p' update instead)
    # Distinct scratch columns per variable: `p_nstar` below aliases `p_col.uMish`, and is
    # still read after `phi_col` has been transformed — so these two must not be the same
    # object. Keyed on p_index vs w_index, they are not.
    p_col = scratch_column(mtile, p_index)
    p_col.uMish .= p_nstar
    Btransform!(p_col)
    Atransform!(p_col)
    p_nstar = Itransform!(p_col)
    p_nstar_z = S.si_p_nstar_z
    Ixtransform(p_col, p_nstar_z)
    p_nstar_z .*= ts_term

    # Mass-flux Helmholtz RHS (φ = ρ̄_t w): rhs = Δτ ∂z p'* - ρ̄_t w*.
    # Elimination gives (∂z(Δτ² Pξ̄(z) ∂z ·) - I) φ (profile coefficient — the
    # weighted-stiffness Galerkin form on RiRk).
    rhs = S.si_rhs
    if sd_si
        @. rhs = p_nstar_z - (S.rho_t * w_nstar)
    else
        @. rhs = p_nstar_z - (rho_tbar * w_nstar)
    end
    phi_col = scratch_column(mtile, w_index)
    if sd_si
        # Per-column, per-step factorization with the state's Pξⁿ profile
        # (w-Dirichlet rows, flags passed explicitly — no registry traffic).
        @. S.sd_alpha = (ts_term * ts_term) * Pxi_vec
        h_sd = _assemble_sd_helmholtz(mtile.solve_data, S.sd_alpha, true, true)
        _vertical_solve!(phi_col, h_sd, rhs, mtile; dirichlet=(true, true))
    elseif t == 1
        # Calculate the Helmholtz matrix for the first time step
        h_a = calc_Helmholtz_semiimplicit_matrix(mtile.tile, mtile.model, Pxi_bar, ts_term)
        _vertical_solve!(phi_col, h_a, rhs, mtile)
    else
        # Use the pre-calculated one
        _vertical_solve!(phi_col, mtile.h_matrix, rhs, mtile)
    end

    phi = Itransform!(phi_col)
    phi_z = S.si_phi_z
    Ixtransform(phi_col, phi_z)

    # Recover w_n+1 = φ_n+1 / ρ̂_t
    if sd_si
        view(mtile.var_np1,colstart:colend,w_index) .= phi ./ S.rho_t
    else
        view(mtile.var_np1,colstart:colend,w_index) .= phi ./ rho_tbar
    end

    # Recover p'_n+1 = p'* - Δτ Pξ ∂z φ_n+1
    view(mtile.var_np1,colstart:colend,p_index) .= p_nstar .- (ts_term .* Pxi_vec .* phi_z)

    # Slaved flux-form updates ∂z(c φ) = c φ_z + c_z φ, pointwise from the solve's
    # φ and φ_z — the exact form the AI2* history staging in mc_driver! mirrors, so
    # the slaved slots see one discrete operator across all time levels. rho_t' has
    # c = 1 exactly (conserves ∫rho_t'); with a dry reference the rho_d' update is
    # then IDENTICAL to rho_t''s, so the two densities cannot drift apart.
    view(mtile.var_np1,colstart:colend,rhot_index) .= rhot_nstar .- (ts_term .* phi_z)

    c_d = S.si_c_d
    c_d_z = S.si_c_d_z
    if sd_si
        @. c_d = S.rho_d / S.rho_t
        @. c_d_z = ((S.rho_d_z * S.rho_t) - (S.rho_d * S.rho_t_z)) / (S.rho_t^2)
    else
        @. c_d = rho_dbar / rho_tbar
        @. c_d_z = ((rho_dbar_z * rho_tbar) - (rho_dbar * rho_tbar_z)) / (rho_tbar^2)
    end
    view(mtile.var_np1,colstart:colend,rhod_index) .=
        rhod_nstar .- (ts_term .* ((c_d .* phi_z) .+ (c_d_z .* phi)))

    c_e = S.si_c_e
    c_e_z = S.si_c_e_z
    if sd_si
        @. c_e = (S.E_t + S.p) / S.rho_t
        @. c_e_z = (((S.E_t_z + S.p_z) * S.rho_t) -
                    ((S.E_t + S.p) * S.rho_t_z)) / (S.rho_t^2)
    else
        @. c_e = (E_tbar + pbar) / rho_tbar
        @. c_e_z = (((E_tbar_z + pbar_z) * rho_tbar) -
                    ((E_tbar + pbar) * rho_tbar_z)) / (rho_tbar^2)
    end
    view(mtile.var_np1,colstart:colend,et_index) .=
        et_nstar .- (ts_term .* ((c_e .* phi_z) .+ (c_e_z .* phi)))

    # Store the applied implicit w increment as the next step's w history (see the
    # staging comment in mc_driver!): the exact operator the Helmholtz elimination
    # applied to the w leg, which no pointwise staging can reproduce. On the first
    # step, seed the n-1 level too so t == 2 has a full AI2* history on the w leg.
    view(mtile.impdot_n,colstart:colend,w_index) .=
        (view(mtile.var_np1,colstart:colend,w_index) .- w_nstar) ./ ts_term
    if t == 1
        view(mtile.impdot_nm1,colstart:colend,w_index) .=
            view(mtile.impdot_n,colstart:colend,w_index)
    end
end

# ── Implicit vertical diffusion ────────────────────────────────────────────────

"""
    diffusion_timestep_mc(mtile, colstart, colend, t)

Implicit vertical diffusion of `u` and `w` (momentum, `Kvdiff`), the moist entropy `s_t'`
(heat, `Kvdiff_heat`) and the water species (`Kvdiff_water`: total water `rho_w'`, cloud
`rho_c'`, rain `rho_r`) for the total-energy set, using the AI2* off-centered weights
(`+1.25 N^{n+1} - 1.0 N^n + 0.75 N^{n-1}`) with the tendency history in `mtile.diffdot_*`.
It needs its own tendency channel because the acoustic solve owns `impdot[w/p/E_t]`.
The momentum, heat and water solves are gated independently on their coefficients: a K = 0
solve is not the identity (it refits the column and reapplies the spectral filter), so a
zero coefficient must skip its solve entirely, leaving the post-acoustic state untouched.

Runs AFTER [`semiimplicit_adjustment_p`](@ref). The density fields are NOT re-slaved to the
post-diffusion `w` — momentum diffusion is a force, not a mass flux, and they were already
advanced with the acoustic flux divergence (which conserves `∫rho_t'`). The residual is the
usual O(ts·Kv) splitting error.

The heat and water paths retrieve the post-acoustic temperature (the star state) through
the same closed-form retrieval `mc_driver!` uses; the resting base stays bit-preserved
because the reference profile `s_tbar` comes from the same pipeline
(`mc_reference_diagnostics`).

Energy routing:

- **Friction is a resolved-KE sink, not heat.** With an eddy diffusivity the resolved KE the
  momentum solve removes goes to the subgrid cascade (the future TKE shear production), and
  the molecular heating is negligible. So `E_t += rho_t δke` (E_t follows the KE down),
  holding internal energy — `T` and `p` are unchanged by friction.
- **Heat.** The `s_t` increment maps at fixed `rho_d` and composition via `∂s_t/∂T = C_vt/T`:
  `δT = (T/C_vt) δs_t`, `δE_t = rho_d T δs_t`, `δp = rho_d R_m δT`, and
  `δQ_ss = -(∂ρ_vs/∂T δT + ∂ρ_vs/∂p δp)` (so `Q_ss = ρ_v - ρ_vs` tracks the diabatic heating).
- **Water.** The species increments map at FIXED temperature (verified against the
  retrieval: `δF(T) ≡ 0` under this map), moving the partial pressure and the internal +
  potential energy the water carries:
  `δp = R_v T δρ_v`, `δE_t = (C_pv T − L_v + ke + gz) δρ_w + (L_v − R_v T) δρ_v`,
  `δQ_ss = δρ_v − ∂ρ_vs/∂p · δp`, with the VAPOR increment `δρ_v = δρ_w − δρ_c − δρ_r`
  implied. Each species' Neumann solve conserves its own `∫ρ`; `∫E_t` is conserved only
  approximately (the exact multicomponent enthalpy flux needs a flux form the per-column
  architecture excludes — see the handoff doc; the residual is O(K_water × vertical
  variation of e_l + gz) and shows up in the energy-drift diagnostic).
"""
diffusion_timestep_mc(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64) =
    diffusion_timestep_mc(mtile, colstart, colend, t, MCCartesianXZ())

function diffusion_timestep_mc(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64,
                               geom::MCGeometry)

    vars = mtile.model.grid_params.vars
    p_index = vars["p"]
    rhod_index = vars["rho_d"]
    rhot_index = vars["rho_t"]
    u_index = vars["u"]
    w_index = vars["w"]
    et_index = vars["E_t"]
    qss_index = vars["Q_ss"]
    rhor_index = mc_slot(vars, "rho_r")
    rhoc_index = mc_slot(vars, "rho_c")

    ts = mtile.model.ts

    Kvdiff = mtile.model.physical_params[:Kvdiff]
    Kvdiff_heat = get(mtile.model.physical_params, :Kvdiff_heat, Kvdiff)
    Kvdiff_water = get(mtile.model.physical_params, :Kvdiff_water, 0.0)
    do_momentum = Kvdiff > 0.0
    do_heat = Kvdiff_heat > 0.0
    do_water = Kvdiff_water > 0.0

    pbar = view(ref_pressure(mtile.ref_state),:,1)
    rho_dbar = view(ref_rho_d(mtile.ref_state),:,1)
    rho_tbar = view(ref_rho_t(mtile.ref_state),:,1)
    E_tbar = view(ref_total_energy(mtile.ref_state),:,1)
    rho_cbar = view(Springsteel.ref_rho_c(mtile.ref_state),:,1)
    z = view(mtile.tilepoints,colstart:colend,zcoord(geom))

    S = @inbounds mtile.mc_scratch[Threads.threadid()]

    # Post-acoustic totals (the star state)
    vnp1 = mtile.var_np1
    # Bind the views OUTSIDE the `@.` blocks below — `@.` dots every call in the expression,
    # `view` included, which broadcasts the view itself instead of its contents.
    u_v = view(vnp1,colstart:colend,u_index)
    w_v = view(vnp1,colstart:colend,w_index)
    p_v = view(vnp1,colstart:colend,p_index)
    rhod_v = view(vnp1,colstart:colend,rhod_index)
    rhot_v = view(vnp1,colstart:colend,rhot_index)
    et_v = view(vnp1,colstart:colend,et_index)
    qss_v = view(vnp1,colstart:colend,qss_index)
    rhor_v = view(vnp1,colstart:colend,rhor_index)
    rhoc_v = view(vnp1,colstart:colend,rhoc_index)

    u_star = S.df_u_star; copyto!(u_star, u_v)
    w_star = S.df_w_star; copyto!(w_star, w_v)
    v_v = mc_v_np1_view(geom, vnp1, colstart, colend, vars)
    v_star = mc_v_star!(geom, S, v_v)
    p_star = S.df_p_star;         @. p_star = p_v + pbar
    rho_d_star = S.df_rho_d_star; @. rho_d_star = rhod_v + rho_dbar
    rho_t_star = S.df_rho_t_star; @. rho_t_star = rhot_v + rho_tbar

    # The heat and water maps need the retrieved star-state thermodynamics; the
    # momentum-only path skips the retrieval entirely.
    T_star = S.df_T_star
    p_hPa_star = S.df_p_hPa_star
    drvs_dT = S.df_drvs_dT
    drvs_dp = S.df_drvs_dp
    rho_v_star = S.df_rho_v_star
    C_vt_star = S.df_C_vt_star
    R_m_star = S.df_R_m_star
    Lv_star = S.df_Lv_star
    ke_star = S.df_ke_star
    stp_star = S.df_stp_star
    rho_c_star = S.df_rho_c_star
    rho_r_star = S.df_rho_r_star
    rho_liq_star = S.df_rho_liq_star
    if do_heat || do_water
        E_t_star = S.df_E_t_star;   @. E_t_star = et_v + E_tbar
        mc_ke_star!(ke_star, geom, u_star, w_star, v_star)
        M_star = S.df_M_star
        @. M_star = p_star + E_t_star - (rho_t_star * (ke_star + (gravity * z)))
        # Same closed-form retrieval and residual partition as mc_driver!: the
        # condensate is prognostic, so the star state needs no iteration and no clamp.
        # Slot 9 recovery, same rule as mc_driver!: the slot holds the control variable and
        # the density is recovered pointwise. `:none` is bitwise `rhoc_v + rho_cbar`.
        ct = condensate_transform_mode(mtile.model.options)
        if ct === :none
            @. rho_c_star = rhoc_v + rho_cbar
        else
            cmu_s = get(mtile.model.physical_params, :condensate_mu, 1.0e-7)
            if ct === :bhyp
                @. rho_c_star = ahyp(rhoc_v + bhyp(rho_cbar, cmu_s), cmu_s)
            else
                @. rho_c_star = ahyp_smooth(rhoc_v + bhyp(rho_cbar, cmu_s), cmu_s)
            end
        end
        # Slot 8 likewise. Rain has no reference profile, so the slot IS the control variable
        # and `:none` is a copy of identical doubles — bitwise inert.
        rt = rain_transform_mode(mtile.model.options)
        if rt === :none
            copyto!(rho_r_star, rhor_v)
        else
            rmu_s = get(mtile.model.physical_params, :rain_mu, 1.0e-7)
            if rt === :bhyp
                @. rho_r_star = ahyp(rhor_v, rmu_s)
            else
                @. rho_r_star = ahyp_smooth(rhor_v, rmu_s)
            end
        end
        # Same thermodynamic interface as mc_driver!, so the same floor applies: T_star and
        # q_l_star are read through it, and a star state that disagreed with the n state
        # about what the liquid IS would put the difference into every increment built here.
        # `:none` (the default) leaves this a copy of identical doubles -- bitwise inert.
        # NOT applied to `rho_v_star` two lines down: that is the water partition, and
        # flooring a partition manufactures vapor. See `condensate_floor_mode`.
        if condensate_floor_mode(mtile.model.options)
            @. rho_liq_star = max(rho_c_star, 0.0) + max(rho_r_star, 0.0)
        else
            @. rho_liq_star = rho_c_star + rho_r_star
        end
        @. T_star = retrieve_temperature(M_star, rho_d_star, rho_t_star, rho_liq_star)
        @. p_hPa_star = p_star / 100.0
        @. drvs_dT = drho_vsat_dT(T_star, p_hPa_star)
        @. drvs_dp = drho_vsat_dp(T_star, p_hPa_star)
        rho_vs_star = S.df_rho_vs_star
        @. rho_vs_star = rho_v_sat(T_star, p_hPa_star)
        # The DENSITY residual, under both `vapor_retrieval` modes THIS PHASE. The star state
        # is an INCREMENTS-ONLY consumer -- everything built from it here is differenced
        # against the same-form n-state quantity -- so the representation cancels to leading
        # order. That is why this stayed on the residual while the blend was measured on the
        # tendency path, and it is why it STAYS there now that `:blend` is the default: the
        # argument is about the increment, not about which retrieval ships.
        # The RAW liquid, always: this is the water partition, and flooring a partition
        # manufactures vapor rather than changing what the thermodynamics reads.
        @. rho_v_star = rho_t_star - rho_d_star - (rho_c_star + rho_r_star)
        q_v_star = S.df_q_v_star
        q_l_star = S.df_q_l_star
        @. q_v_star = rho_v_star / rho_d_star
        @. q_l_star = rho_liq_star / rho_d_star     # thermodynamic reader: floored if enabled
        @. C_vt_star = Cvd + (q_v_star * Cvv) + (q_l_star * Cl)
        @. R_m_star = Rd + (q_v_star * Rv)
        @. Lv_star = L_v(T_star)
        @. stp_star = moist_entropy_total(T_star, rho_d_star, q_v_star, q_l_star)
        stp_star .-= mtile.mc_ref_diag.s_tbar
    end

    # Implicit tendency histories (slot 6 carries the s_t ENTROPY tendency, not an E_t
    # one; slots 3/7/8 carry the water tendencies)
    udot_n = view(mtile.diffdot_n,colstart:colend,u_index)
    udot_nm1 = view(mtile.diffdot_nm1,colstart:colend,u_index)
    wdot_n = view(mtile.diffdot_n,colstart:colend,w_index)
    wdot_nm1 = view(mtile.diffdot_nm1,colstart:colend,w_index)
    sdot_n = view(mtile.diffdot_n,colstart:colend,et_index)
    sdot_nm1 = view(mtile.diffdot_nm1,colstart:colend,et_index)

    # Pre-factorized in createModelTile, per variable and per timestep coefficient (the
    # first step uses the AM2 coefficient), so nothing is factorized per column here.
    mats = mtile.mc_diffusion_matrices

    # One scratch column serves all three solves: the boundary conditions live in the
    # factorization, and neither `_vertical_solve!` nor `Itransform!` consults the
    # column's own BCs (only Btransform!/Atransform! do). Same reuse as
    # `diffusion_timestep_pd`, which shares one column across five variables.
    col = scratch_column(mtile, u_index)

    # ── Momentum (Kvdiff): friction is a resolved-KE sink, E_t follows the KE down;
    #    T, p, Q_ss held. ──
    dE_visc = S.df_dE_visc
    if do_momentum
        u_nstar = S.df_u_nstar
        w_nstar = S.df_w_nstar
        if (t == 1)
            # Use trapezoidal method (AM2) for first step
            @. u_nstar = u_star + (ts * 0.5 * udot_n)
            @. w_nstar = w_star + (ts * 0.5 * wdot_n)
        else
            # Use AI2* for second step and beyond
            @. u_nstar = u_star - (ts * udot_n) + (ts * 0.75 * udot_nm1)
            @. w_nstar = w_star - (ts * wdot_n) + (ts * 0.75 * wdot_nm1)
        end
        udot_nm1 .= udot_n
        wdot_nm1 .= wdot_n

        h_u = (t == 1) ? mats.u_first : mats.u
        h_w = (t == 1) ? mats.w_first : mats.w
        # u_np1 and w_np1 must be copied out: `col` is reused by the next solve, and
        # Itransform! returns the column's own buffer.
        u_np1 = S.df_u_np1
        w_np1 = S.df_w_np1
        _vertical_solve!(col, h_u, u_nstar, mtile)
        copyto!(u_np1, Itransform!(col))

        _vertical_solve!(col, h_w, w_nstar, mtile)
        copyto!(w_np1, Itransform!(col))

        v_np1 = mc_v_diffusion_solve!(geom, S, mtile, col, mats, ts, t,
                                      colstart, colend, v_star)

        dke = S.df_dke
        mc_dke!(dke, geom, u_np1, w_np1, u_star, w_star, v_np1, v_star)
        @. dE_visc = rho_t_star * dke

        u_v .= u_np1
        w_v .= w_np1
        mc_assign_v!(geom, v_v, v_np1)
    end

    # ── Heat (Kvdiff_heat): moist entropy increment -> (T, p, E_t, Q_ss) at fixed
    #    rho_d and composition via ∂s_t/∂T = C_vt/T (dry limit: C_vt=C_vd, R_m=R_d). ──
    dE_h = S.df_dE_h
    if do_heat
        s_nstar = S.df_s_nstar
        if (t == 1)
            @. s_nstar = stp_star + (ts * 0.5 * sdot_n)
        else
            @. s_nstar = stp_star - (ts * sdot_n) + (ts * 0.75 * sdot_nm1)
        end
        sdot_nm1 .= sdot_n

        h_h = (t == 1) ? mats.heat_first : mats.heat
        _vertical_solve!(col, h_h, s_nstar, mtile)
        stp_np1 = Itransform!(col)

        ds_t = S.df_ds_t; @. ds_t = stp_np1 - stp_star
        dT_h = S.df_dT_h; @. dT_h = (T_star / C_vt_star) * ds_t
        @. dE_h = rho_d_star * T_star * ds_t
        dp_h = S.df_dp_h; @. dp_h = rho_d_star * R_m_star * dT_h
        dQ_h = S.df_dQ_h; @. dQ_h = -((drvs_dT * dT_h) + (drvs_dp * dp_h))

        p_v .+= dp_h
        qss_v .+= dQ_h
    end

    # ── Water (Kvdiff_water): rho_w' and rho_v' on the rho_t operator, rho_r on its
    #    own; increments map at FIXED temperature (the retrieval is invariant under
    #    them), moving the vapor partial pressure and the water's internal + potential
    #    energy. delta_rho_c = delta_rho_w - delta_rho_v - delta_rho_r is implicit.
    #    Lives in its own function: inlined here it pushed diffusion_timestep_mc past
    #    the optimizer's budget and one broadcast-into-view stopped eliding its
    #    SubArray (one 64-byte allocation per column per step). ──
    dE_w = S.df_dE_w
    if do_water
        # `_diffusion_water_step!` solves the implicit vertical diffusion for slots 8 and 9 as
        # DENSITIES (`drc = rhoc_np1 - rhoc_star` feeds the mass and energy maps directly).
        # Under a transform the slot is a control variable, so the solve would have to run in
        # nu-space and the increment be `ahyp(nu_np1) - ahyp(nu_star)`. That is not written
        # yet, and `Kvdiff_water` is 0.0 in every shipped configuration (o01_rainfall.jl:89,
        # tc/tc_init.jl:205), so the combination has never run. Fail loudly rather than
        # silently solve the wrong equation for the water.
        (condensate_transform_mode(mtile.model.options) === :none &&
         rain_transform_mode(mtile.model.options) === :none) ||
            error("a water transform with physical_params[:Kvdiff_water] > 0 is not " *
                  "implemented: the implicit water diffusion solves slots 8 and 9 as " *
                  "densities, and under a transform the affected slot must be solved in the " *
                  "control variable with the increment mapped through ahyp. Set " *
                  "Kvdiff_water = 0 or implement the nu-space solve in " *
                  "_diffusion_water_step!.")
        _diffusion_water_step!(mtile, S, col, mats, ts, t, colstart, colend, z,
                               rhod_v, rhot_v, rhor_v, rhoc_v, p_v, qss_v)
    end

    # E_t update: keep the single fused add when the historical pair ran (bit-identical
    # to the pre-water combined update), and touch E_t only with computed terms.
    if do_momentum && do_heat
        et_v .+= dE_visc .+ dE_h
    elseif do_momentum
        et_v .+= dE_visc
    elseif do_heat
        et_v .+= dE_h
    end
    if do_water
        # Explicit loop, not broadcast: as the last branch of this large function
        # both the broadcast and @turbo forms stopped eliding the destination
        # SubArray wrapper (one 64-byte heap allocation per column per step)
        @inbounds for i in eachindex(dE_w)
            et_v[i] += dE_w[i]
        end
    end
end

"""
    _diffusion_water_step!(mtile, S, col, mats, ts, t, colstart, colend, z,
                           rhod_v, rhot_v, rhor_v, rhoc_v, p_v, qss_v)

The water-species half of [`diffusion_timestep_mc`](@ref): AM2/AI2* staging and
implicit solves for rho_w' (on the rho_t operator), rho_c' and rho_r (each on its
own), then the fixed-temperature increment maps onto rho_t, rho_c, rho_r, p and Q_ss,
with the vapor increment implied as drho_v = drho_w - drho_c - drho_r.
The energy increment lands in `S.df_dE_w`; the CALLER adds it to E_t so the
historical dE_visc + dE_h fusion stays bit-identical. Star-state fields
(`df_T_star` etc.) are read from the scratch where the caller computed them.

`@noinline` in its own function on purpose: inlined into diffusion_timestep_mc this
block pushed the function past the optimizer's budget and one broadcast-into-view
stopped eliding its SubArray — one 64-byte heap allocation per column per step.
"""
@noinline function _diffusion_water_step!(mtile::ModelTile, S, col, mats,
                                          ts::Float64, t::Int64,
                                          colstart::Int64, colend::Int64, z,
                                          rhod_v, rhot_v, rhor_v, rhoc_v, p_v, qss_v)
    vars = mtile.model.grid_params.vars
    rhot_index = vars["rho_t"]
    rhor_index = mc_slot(vars, "rho_r")
    rhoc_index = mc_slot(vars, "rho_c")

    T_star = S.df_T_star
    Lv_star = S.df_Lv_star
    ke_star = S.df_ke_star
    drvs_dp = S.df_drvs_dp
    dE_w = S.df_dE_w

    # Every diffused species is prognostic now: total water (on the rho_t operator),
    # cloud and rain. The VAPOR increment is the implied remainder — the mirror of the
    # old convention, which solved the diagnosed rho_v' and implied the cloud.
    rw_star = S.df_rw_star; @. rw_star = rhot_v - rhod_v
    rc_star = S.df_rc_star; copyto!(rc_star, rhoc_v)

    rwdot_n = view(mtile.diffdot_n,colstart:colend,rhot_index)
    rwdot_nm1 = view(mtile.diffdot_nm1,colstart:colend,rhot_index)
    rcdot_n = view(mtile.diffdot_n,colstart:colend,rhoc_index)
    rcdot_nm1 = view(mtile.diffdot_nm1,colstart:colend,rhoc_index)
    rrdot_n = view(mtile.diffdot_n,colstart:colend,rhor_index)
    rrdot_nm1 = view(mtile.diffdot_nm1,colstart:colend,rhor_index)

    rw_nstar = S.df_rw_nstar
    rc_nstar = S.df_rc_nstar
    rr_nstar = S.df_rr_nstar
    if (t == 1)
        @. rw_nstar = rw_star + (ts * 0.5 * rwdot_n)
        @. rc_nstar = rc_star + (ts * 0.5 * rcdot_n)
        @. rr_nstar = rhor_v + (ts * 0.5 * rrdot_n)
    else
        @. rw_nstar = rw_star - (ts * rwdot_n) + (ts * 0.75 * rwdot_nm1)
        @. rc_nstar = rc_star - (ts * rcdot_n) + (ts * 0.75 * rcdot_nm1)
        @. rr_nstar = rhor_v - (ts * rrdot_n) + (ts * 0.75 * rrdot_nm1)
    end
    rwdot_nm1 .= rwdot_n
    rcdot_nm1 .= rcdot_n
    rrdot_nm1 .= rrdot_n

    h_rw = (t == 1) ? mats.water_first : mats.water
    h_rc = (t == 1) ? mats.water_c_first : mats.water_c
    h_rr = (t == 1) ? mats.water_r_first : mats.water_r
    rw_np1 = S.df_rw_np1
    rc_np1 = S.df_rc_np1
    _vertical_solve!(col, h_rw, rw_nstar, mtile)
    copyto!(rw_np1, Itransform!(col))
    _vertical_solve!(col, h_rc, rc_nstar, mtile)
    copyto!(rc_np1, Itransform!(col))
    _vertical_solve!(col, h_rr, rr_nstar, mtile)
    rr_np1 = Itransform!(col)

    drw = S.df_drw; @. drw = rw_np1 - rw_star
    drc = S.df_drc; @. drc = rc_np1 - rc_star
    drr = S.df_drr; @. drr = rr_np1 - rhor_v
    drv = S.df_drv; @. drv = drw - drc - drr
    @. dE_w = (((Cpv * T_star) - Lv_star + ke_star + (gravity * z)) * drw) +
              ((Lv_star - (Rv * T_star)) * drv)

    rhot_v .+= drw
    rhoc_v .+= drc
    rhor_v .+= drr
    p_v .+= Rv .* T_star .* drv
    qss_v .+= drv .- (drvs_dp .* (Rv .* T_star .* drv))
    return nothing
end

# ── Initial conditions and reference writer ────────────────────────────────────

"""
    write_exact_ref_mc(path, z, p_Pa, rho_d, rho_v, rho_c)

Write a pressure-based exact reference state file (`z p rho_d rho_v rho_c` per line,
p in Pa) in the format read by `Springsteel.exact_pressure_reference_state`. `z`
values are written with `string()` so they match the model gridpoints exactly when
generated in-process.
"""
function write_exact_ref_mc(path::String, z::Vector{Float64}, p_Pa::Vector{Float64},
                            rho_d::Vector{Float64}, rho_v::Vector{Float64},
                            rho_c::Vector{Float64})
    open(path, "w") do f
        for i in 1:length(z)
            println(f, "$(z[i]) $(p_Pa[i]) $(rho_d[i]) $(rho_v[i]) $(rho_c[i])")
        end
    end
    return path
end

"""
    theta_bubble_mc!(patch, gridpoints, ref; xc, xr, zc, zr, dtheta_max)

Dry warm-bubble initial condition for the total-energy set on a
`PressureReferenceState`. The constant-pressure θ perturbation mirrors
[`theta_bubble_pd!`](@ref) but writes the moist_compressible slots: p' = 0 (constant
pressure), rho_d' = rho_t', E_t' from the BF02 internal energy at rest, and Q_ss'
tracking -ρ_v*(T, p̄) in the dry air.
"""
function theta_bubble_mc!(patch::AbstractGrid, gridpoints::Matrix{Float64},
                          ref::Springsteel.PressureReferenceState;
                          xc=10000.0, xr=2000.0, zc=2000.0, zr=2000.0, dtheta_max=2.0,
                          condensate_transform::Symbol=:none, condensate_mu=1.0e-7,
                          rain_transform::Symbol=:none, rain_mu=1.0e-7)
    vars = patch.params.vars
    p_i = vars["p"]; rho_d_i = vars["rho_d"]; rho_t_i = vars["rho_t"]
    et_i = vars["E_t"]; qss_i = vars["Q_ss"]; rho_r_i = mc_slot(vars, "rho_r")
    rho_c_i = mc_slot(vars, "rho_c")
    kDim = patch.params.kDim
    pbar = ref_pressure(ref); rho_dbar = ref_rho_d(ref); rho_tbar = ref_rho_t(ref)
    rho_cbar = Springsteel.ref_rho_c(ref)
    E_tbar = ref_total_energy(ref); Q_ssbar = ref_qss(ref)

    i = 1
    for _ in 1:num_columns(patch)
        for k in 1:kDim
            x = gridpoints[i, 1]
            z = gridpoints[i, 2]
            L = sqrt(((x - xc) / xr)^2 + ((z - zc) / zr)^2)
            dtheta = L <= 1.0 ? dtheta_max * (cos(pi * L / 2.0))^2 : 0.0
            p_ref = pbar[k, 1]                              # Pa
            rho_dref = rho_dbar[k, 1]
            T_ref = p_ref / (Rd * rho_dref)                 # dry EOS
            exner = (p_0 * 100.0 / p_ref)^(Rd / Cpd)
            theta = (T_ref * exner) + dtheta
            Tk = theta / exner
            rho_d = p_ref / (Rd * Tk)                       # constant-pressure perturbation
            E_t = (rho_d * internal_energy_bf02(Tk, 0.0, 0.0)) + (rho_d * gravity * z)
            Q_ss = -rho_v_sat(Tk, p_ref / 100.0)
            patch.physical[i, p_i, 1] = 0.0
            patch.physical[i, rho_d_i, 1] = rho_d - rho_dref
            patch.physical[i, rho_t_i, 1] = rho_d - rho_tbar[k, 1]
            patch.physical[i, et_i, 1] = E_t - E_tbar[k, 1]
            patch.physical[i, qss_i, 1] = Q_ss - Q_ssbar[k, 1]
            patch.physical[i, rho_r_i, 1] = rain_slot(0.0, rain_transform, rain_mu)
            patch.physical[i, rho_c_i, 1] =
                condensate_slot(0.0, rho_cbar[k, 1], condensate_transform, condensate_mu)
            i += 1
        end
    end
    return patch
end

"""
    temperature_bubble_mc!(patch, gridpoints, ref; xc, xr, zc, zr, dT_max)

Straka et al. (1993) cold-bubble initial condition for the total-energy set on a
`PressureReferenceState`. A constant-pressure temperature perturbation
`ΔT = dT_max (cos(πL)+1)/2` for `L ≤ 1` (identical to the `cos²(πL/2)` shape used by
[`theta_bubble_mc!`](@ref)) is applied to the dry reference profile: p' = 0,
rho_d' = rho_t' from the dry EOS at the perturbed temperature, E_t' from the BF02 internal
energy at rest, and Q_ss' tracking `-ρ_v*(T, p̄)` in the dry air.
"""
function temperature_bubble_mc!(patch::AbstractGrid, gridpoints::Matrix{Float64},
                                ref::Springsteel.PressureReferenceState;
                                xc=0.0, xr=4000.0, zc=3000.0, zr=2000.0, dT_max=-15.0,
                                condensate_transform::Symbol=:none, condensate_mu=1.0e-7,
                          rain_transform::Symbol=:none, rain_mu=1.0e-7)
    vars = patch.params.vars
    p_i = vars["p"]; rho_d_i = vars["rho_d"]; rho_t_i = vars["rho_t"]
    et_i = vars["E_t"]; qss_i = vars["Q_ss"]; rho_r_i = mc_slot(vars, "rho_r")
    rho_c_i = mc_slot(vars, "rho_c")
    kDim = patch.params.kDim
    pbar = ref_pressure(ref); rho_dbar = ref_rho_d(ref); rho_tbar = ref_rho_t(ref)
    rho_cbar = Springsteel.ref_rho_c(ref)
    E_tbar = ref_total_energy(ref); Q_ssbar = ref_qss(ref)

    i = 1
    for _ in 1:num_columns(patch)
        for k in 1:kDim
            x = gridpoints[i, 1]
            z = gridpoints[i, 2]
            L = sqrt(((x - xc) / xr)^2 + ((z - zc) / zr)^2)
            dT = L <= 1.0 ? dT_max * (cos(pi * L / 2.0))^2 : 0.0
            p_ref = pbar[k, 1]                              # Pa
            rho_dref = rho_dbar[k, 1]
            Tk = (p_ref / (Rd * rho_dref)) + dT             # dry EOS reference T, perturbed
            rho_d = p_ref / (Rd * Tk)                       # constant-pressure perturbation
            E_t = (rho_d * internal_energy_bf02(Tk, 0.0, 0.0)) + (rho_d * gravity * z)
            Q_ss = -rho_v_sat(Tk, p_ref / 100.0)
            patch.physical[i, p_i, 1] = 0.0
            patch.physical[i, rho_d_i, 1] = rho_d - rho_dref
            patch.physical[i, rho_t_i, 1] = rho_d - rho_tbar[k, 1]
            patch.physical[i, et_i, 1] = E_t - E_tbar[k, 1]
            patch.physical[i, qss_i, 1] = Q_ss - Q_ssbar[k, 1]
            patch.physical[i, rho_r_i, 1] = rain_slot(0.0, rain_transform, rain_mu)
            patch.physical[i, rho_c_i, 1] =
                condensate_slot(0.0, rho_cbar[k, 1], condensate_transform, condensate_mu)
            i += 1
        end
    end
    return patch
end

"""
    moist_temperature_bubble_mc!(patch, gridpoints, ref; xc, xr, zc, zr, dT_max)

Ooyama (2001)-style warm-rain trigger for the total-energy set on a moist (vapor-bearing)
`PressureReferenceState`: a constant-pressure temperature perturbation
`ΔT = dT_max cos²(πL/2)` for `L ≤ 1` with the vapor adjusted to PRESERVE the local
relative humidity at the perturbed temperature (`ρ_v = H·ρ_v*(T)` with
`H = ρ̄_v/ρ_v*(T̄)`), so the bubble carries extra moisture rather than drying out as it
warms. The dry density follows from the moist EOS at constant pressure; E_t and Q_ss
are computed pointwise before subtracting the reference; rho_r = 0.
"""
function moist_temperature_bubble_mc!(patch::AbstractGrid, gridpoints::Matrix{Float64},
                                      ref::Springsteel.PressureReferenceState;
                                      xc=75.0e3, xr=16.0e3, zc=500.0, zr=3000.0,
                                      dT_max=3.0, zcol=2,
                                      condensate_transform::Symbol=:none,
                                      condensate_mu=1.0e-7,
                                      rain_transform::Symbol=:none, rain_mu=1.0e-7)
    vars = patch.params.vars
    p_i = vars["p"]; rho_d_i = vars["rho_d"]; rho_t_i = vars["rho_t"]
    et_i = vars["E_t"]; qss_i = vars["Q_ss"]; rho_r_i = mc_slot(vars, "rho_r")
    rho_c_i = mc_slot(vars, "rho_c")
    kDim = patch.params.kDim
    pbar = ref_pressure(ref); rho_dbar = ref_rho_d(ref); rho_tbar = ref_rho_t(ref)
    rho_cbar = Springsteel.ref_rho_c(ref)
    rho_vbar = Springsteel.ref_rho_v(ref)
    E_tbar = ref_total_energy(ref); Q_ssbar = ref_qss(ref)

    i = 1
    for _ in 1:num_columns(patch)
        for k in 1:kDim
            # On a 3D cylindrical grid (zcol = 3) this is a WN0 torus bubble:
            # column 1 is the radius and λ (column 2) does not enter.
            x = gridpoints[i, 1]
            z = gridpoints[i, zcol]
            L = sqrt(((x - xc) / xr)^2 + ((z - zc) / zr)^2)
            dT = L <= 1.0 ? dT_max * (cos(pi * L / 2.0))^2 : 0.0
            p_ref = pbar[k, 1]                              # Pa
            rho_dref = rho_dbar[k, 1]
            rho_vref = rho_vbar[k, 1]
            # Moist EOS reference temperature (matches the reference's own Tbar)
            T_ref = p_ref / ((rho_dref * Rd) + (rho_vref * Rv))
            Tk = T_ref + dT
            rho_v = rho_vref
            if dT > 0.0
                H = rho_vref / rho_v_sat(T_ref, p_ref / 100.0)
                rho_v = H * rho_v_sat(Tk, p_ref / 100.0)
            end
            rho_d = (p_ref - (rho_v * Rv * Tk)) / (Rd * Tk) # constant-pressure moist EOS
            rho_t = rho_d + rho_v
            q_v = rho_v / rho_d
            E_t = (rho_d * internal_energy_bf02(Tk, q_v, 0.0)) + (rho_t * gravity * z)
            Q_ss = rho_v - rho_v_sat(Tk, p_ref / 100.0)
            patch.physical[i, p_i, 1] = 0.0
            patch.physical[i, rho_d_i, 1] = rho_d - rho_dref
            patch.physical[i, rho_t_i, 1] = rho_t - rho_tbar[k, 1]
            patch.physical[i, et_i, 1] = E_t - E_tbar[k, 1]
            patch.physical[i, qss_i, 1] = Q_ss - Q_ssbar[k, 1]
            patch.physical[i, rho_r_i, 1] = rain_slot(0.0, rain_transform, rain_mu)
            patch.physical[i, rho_c_i, 1] =
                condensate_slot(0.0, rho_cbar[k, 1], condensate_transform, condensate_mu)
            i += 1
        end
    end
    return patch
end

"""
    moist_buoyancy_bubble_mc!(patch, gridpoints, base, ref; q_t=0.02,
                              xc, xr, zc, zr, amp=2.0/300.0)

Total-energy variant of [`moist_buoyancy_bubble_pd!`](@ref): the identical Bryan &
Fritsch (2002) warm-bubble construction (θ_ρ inflation at constant pressure, reset to
exact saturation, re-converged with `saturation_adjustment`), written to the
moist_compressible slots. The final pressure comes from the EOS of the converged
(s, ρ_d, q_v) state so the initial temperature retrieval is exact; E_t and Q_ss are
computed pointwise from (T, p, ρ_d, ρ_v, ρ_c) before subtracting the reference.
"""
function moist_buoyancy_bubble_mc!(patch::AbstractGrid, gridpoints::Matrix{Float64},
                                   base, ref::Springsteel.PressureReferenceState; q_t=0.02,
                                   xc=10000.0, xr=2000.0, zc=2000.0, zr=2000.0,
                                   amp=2.0/300.0,
                                   condensate_transform::Symbol=:none, condensate_mu=1.0e-7,
                          rain_transform::Symbol=:none, rain_mu=1.0e-7)
    vars = patch.params.vars
    p_i = vars["p"]; rho_d_i = vars["rho_d"]; rho_t_i = vars["rho_t"]
    et_i = vars["E_t"]; qss_i = vars["Q_ss"]; rho_r_i = mc_slot(vars, "rho_r")
    rho_c_i = mc_slot(vars, "rho_c")
    kDim = patch.params.kDim
    pbar = ref_pressure(ref); rho_dbar = ref_rho_d(ref); rho_tbar = ref_rho_t(ref)
    rho_cbar = Springsteel.ref_rho_c(ref)
    E_tbar = ref_total_energy(ref); Q_ssbar = ref_qss(ref)

    i = 1
    for _ in 1:num_columns(patch)
        for k in 1:kDim
            x = gridpoints[i, 1]
            z = gridpoints[i, 2]
            L = sqrt(((x - xc) / xr)^2 + ((z - zc) / zr)^2)
            b_incr = L <= 1.0 ? amp * (cos(pi * L / 2.0))^2 : 0.0

            new_s = base.s[k]
            new_rho_d = base.rho_d[k]
            new_q_v = base.q_v[k]
            new_q_l = base.q_l[k]
            if b_incr > 0.0
                p = base.p[k]
                new_theta = base.theta_rho[k] * (1.0 + b_incr) * (1.0 + q_t) /
                            (1.0 + (base.q_v[k] / Eps))
                new_T = new_theta / (p_0 / p)^(Rd / Cpd)
                new_q_v = q_sat_liquid(new_T, p)
                new_q_l = q_t - new_q_v
                new_T_rho = new_T * (1.0 + new_q_v / Eps) / (1.0 + q_t)
                new_rho_t = (p * 100.0) / (new_T_rho * Rd)
                new_rho_d = new_rho_t / (1.0 + q_t)
                new_xi = log_dry_density(new_rho_d)
                new_mu = mu_transform(new_q_v)
                new_mu_l = mu_transform(new_q_l)
                new_s = entropy(new_T, new_rho_d, new_q_v)
                dq, _ = saturation_adjustment(new_s, new_xi, new_mu, new_mu_l, eps())
                new_s += s_condensation_relaxation(-dq, new_T, new_rho_d, new_q_v, new_q_l, p)
                new_q_v += dq
                new_q_l = q_t - new_q_v
            end

            # Final consistent state: T from the entropy, p from the EOS, so that the
            # total-energy temperature retrieval reproduces T exactly at t = 0.
            Tk = temperature(new_s, new_rho_d, new_q_v)
            p_Pa = 100.0 * pressure(new_s, new_rho_d, new_q_v)
            rho_v = new_rho_d * new_q_v
            rho_c = new_rho_d * new_q_l
            rho_t = new_rho_d + rho_v + rho_c
            E_t = (new_rho_d * internal_energy_bf02(Tk, new_q_v, new_q_l)) +
                  (rho_t * gravity * z)
            Q_ss = rho_v - rho_v_sat(Tk, p_Pa / 100.0)

            patch.physical[i, p_i, 1] = p_Pa - pbar[k, 1]
            patch.physical[i, rho_d_i, 1] = new_rho_d - rho_dbar[k, 1]
            patch.physical[i, rho_t_i, 1] = rho_t - rho_tbar[k, 1]
            patch.physical[i, et_i, 1] = E_t - E_tbar[k, 1]
            patch.physical[i, qss_i, 1] = Q_ss - Q_ssbar[k, 1]
            patch.physical[i, rho_r_i, 1] = rain_slot(0.0, rain_transform, rain_mu)
            patch.physical[i, rho_c_i, 1] =
                condensate_slot(rho_c, rho_cbar[k, 1], condensate_transform, condensate_mu)
            i += 1
        end
    end
    return patch
end
