# Total-energy moist compressible equation set.
#
# Prognostic variables (XZ slice): p [Pa], rho_d, rho_t, u, w, E_t [J/m^3], Q_ss
# [kg/m^3], rho_r. Temperature is diagnosed each step from a univariate Newton
# retrieval on the Bryan & Fritsch (2002) total energy, then the water partition
# follows diagnostically: rho_v = Q_ss + rho_vs(T, p), rho_c = rho_t - rho_d -
# rho_v - rho_r. Because rho_v and rho_c are diagnostic, no separate condensation
# adjustment step is needed - the vapor/cloud split is always consistent with the
# prognostic supersaturation.
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
    :p, :rho_d, :rho_t, :E_t, :Q_ss,                                  # totals
    :p_z, :rho_d_z, :rho_t_z, :E_t_z, :Q_ss_z,                        # total vertical gradients
    :ke, :geo, :M, :Tk, :p_hPa, :rho_vs, :rho_v, :rho_c, :q_v, :q_l,  # diagnostic state
    :C_vt, :R_m, :C_pt, :gamma_m, :Lv, :drvs_dT, :drvs_dp,            # mixture thermo
    :Q_s, :Qdot, :Qdot_r, :div,                                       # condensation, divergence
    :AUTO_COLL, :Vt, :Fr, :Fr_z, :E_sed, :E_sed_z,                    # warm-rain microphysics
    :sd_xx, :QDOT_TH, :FRIC_KE,                                       # horizontal diffusion
    :ADV, :FORCING, :KDIFF,                                           # per-slot accumulators
    :dT_nc, :dp_nc, :SATF, :QSSREL,                                   # Q_ss chain rule
    :s_t, :stage_zz,                                                             # moist entropy (vertical heat)
    :imp_phi_z, :imp_c_d, :imp_c_d_z, :imp_c_e, :imp_c_e_z,           # acoustic AI2* history staging
    :sd_pxi, :sd_alpha,                    # state-dependent acoustic linearization
    # ── Louis boundary layer + Smagorinsky closure (mc_boundary_layer.jl) ──
    :Kv, :K_smag, :VD_u, :VD_v, :VD_w, :QDOT_V, :VDOT_w, :VDOT_v,
    :bl_s_z, :bl_rv_z,
    # ── semiimplicit_adjustment_p (si_ prefix) ──
    # Deliberately NOT sharing the names above. The two functions' temporaries are not live at
    # the same time today, so sharing would work — but it would be an invisible coupling, and
    # the first person to read an mc_XZ value after the adjustment call would get silent
    # corruption. Distinct names cost ~200 KB and make that unwriteable.
    :si_p_nstar, :si_w_nstar, :si_rhod_nstar, :si_rhot_nstar, :si_et_nstar,
    :si_p_nstar_z, :si_phi_z, :si_rhs, :si_c_d, :si_c_d_z, :si_c_e, :si_c_e_z,
    # ── diffusion_timestep_mc (df_ prefix) ──
    :df_u_star, :df_w_star, :df_p_star, :df_rho_d_star, :df_rho_t_star,
    :df_E_t_star, :df_Q_ss_star, :df_ke_star, :df_M_star,
    :df_T_star, :df_p_hPa_star, :df_drvs_dT, :df_drvs_dp,
    :df_rho_vs_star, :df_rho_v_star, :df_q_v_star, :df_q_l_star,
    :df_C_vt_star, :df_R_m_star, :df_Lv_star, :df_stp_star,
    :df_u_nstar, :df_w_nstar, :df_s_nstar, :df_u_np1, :df_w_np1,
    :df_dke, :df_dE_visc, :df_ds_t, :df_dT_h, :df_dE_h, :df_dp_h, :df_dQ_h,
    :df_rw_star, :df_rv_star, :df_rw_nstar, :df_rv_nstar, :df_rr_nstar,
    :df_rw_np1, :df_rv_np1, :df_drw, :df_drv, :df_drr, :df_dE_w,
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
    retrieve_temperature(M, rho_d, rho_t, Q_ss, p_Pa, T_guess, rho_r=0.0; tol=1.0e-9, maxiter=25)

Diagnose the temperature from the prognostic variables of the total-energy set by
Newton iteration on

    F(T) = (ρ_d C_pd + (ρ_t − ρ_d) C_pv) T + (Q_ss + ρ_d + ρ_vs(T,p) − ρ_t) L_v(T) − M

where `M = p + E_t − ρ_t(v²/2 + gz)` [J/m³] is the available enthalpy density.
F is monotone increasing in T (all dominant terms positive), so the root is unique;
convergence from the reference or previous-step temperature takes 2-3 iterations.
`tol` is the temperature increment tolerance [K].
"""
function retrieve_temperature(M, rho_d, rho_t, Q_ss, p_Pa, T_guess, rho_r=0.0;
                              tol=1.0e-9, maxiter=25)

    p_hPa = p_Pa / 100.0
    Cfactor = (rho_d * Cpd) + ((rho_t - rho_d) * Cpv)
    rho_w = max(rho_t - rho_d, 0.0)                      # total water density
    rho_v_max = max(rho_w - rho_r, 0.0)                  # rain is liquid, not vapor
    Tk = T_guess
    for _ in 1:maxiter
        rho_vs = rho_v_sat(Tk, p_hPa)
        # Clamped diagnostic partition: vapor cannot be negative, nor exceed the total
        # water less the rain (otherwise the residual cloud rho_c = rho_w - rho_v - rho_r
        # goes negative). Conservation lives in (rho_d, rho_t, E_t); the partition is pure
        # bookkeeping, and the clamp makes the retrieval immune to Q_ss tracking
        # drift where rho_vs → 0 (e.g. dry air, where wfactor = 0 exactly).
        # `qss_admissible_bounds` relaxes the prognostic Q_ss onto exactly this interval.
        rho_v = clamp(Q_ss + rho_vs, 0.0, rho_v_max)
        wfactor = rho_v - rho_w                          # = -rho_l (liquid incl. rain)
        F = (Cfactor * Tk) + (wfactor * L_v(Tk)) - M
        dvdT = (0.0 < Q_ss + rho_vs) && (Q_ss + rho_vs < rho_v_max) ?
               drho_vsat_dT(Tk, p_hPa) : 0.0
        Fprime = Cfactor + (L_v(Tk) * dvdT) + (wfactor * (Cpv - Cl))
        dT = -F / Fprime
        Tk = clamp(Tk + dT, 150.0, 350.0)
        if abs(dT) < tol
            return Tk
        end
    end
    return Tk
end

"""
    qss_admissible_bounds(rho_d, rho_t, rho_r, rho_vs)

Lower and upper bounds on the prognostic supersaturation density Q_ss = ρ_v − ρ_vs(T,p),
implied by the water-mass constraint 0 ≤ ρ_v ≤ ρ_t − ρ_d − ρ_r:

    Q_lo = −ρ_vs ,   Q_hi = max(ρ_t − ρ_d − ρ_r, 0) − ρ_vs

In cloud-free air the true Q_ss sits exactly on `Q_hi`; in dry air the interval collapses
to the single point −ρ_vs (Q_ss carries no information there, since ρ_v ≡ 0 regardless).
Inside the interval — i.e. wherever cloud exists — Q_ss is load-bearing and untouched.
This is the same interval [`retrieve_temperature`](@ref) clamps the diagnostic partition to.
"""
function qss_admissible_bounds(rho_d, rho_t, rho_r, rho_vs)

    Q_lo = -rho_vs
    Q_hi = max(rho_t - rho_d - rho_r, 0.0) - rho_vs
    return Q_lo, max(Q_hi, Q_lo)
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
"""
function moist_entropy_total(Tk, rho_d, q_v, q_l)

    return entropy(Tk, rho_d, q_v) + (q_l * Cl * log(Tk / T_0))
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
    E_tbar = ref_total_energy(ref_state)
    Q_ssbar = ref_qss(ref_state)
    Tbar = ref_state.Tbar
    n = length(z)
    s_tbar = zeros(Float64, n)
    rho_vbar = zeros(Float64, n)
    Pxi_prof = zeros(Float64, n)
    for k in 1:n
        p = pbar[k, 1]
        rho_d = rho_dbar[k, 1]
        rho_t = rho_tbar[k, 1]
        Q_ss = Q_ssbar[k, 1]
        # Mirror the equation set's per-point pipeline at rest (ke = 0, rho_r = 0);
        # every expression must match moist_compressible_XZ bit-for-bit.
        geo = 0.5 * ((0.0 * 0.0) + (0.0 * 0.0)) + (gravity * z[k])
        M = p + (E_tbar[k, 1]) - (rho_t * geo)
        Tk = retrieve_temperature(M, rho_d, rho_t, Q_ss, p, Tbar[k, 1], 0.0)
        rho_vs = rho_v_sat(Tk, p / 100.0)
        rho_v = clamp(Q_ss + rho_vs, 0.0, max(rho_t - rho_d - 0.0, 0.0))
        rho_c = rho_t - rho_d - rho_v - 0.0
        q_v = rho_v / rho_d
        q_l = (max(rho_c, 0.0) + 0.0) / rho_d
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
available cloud water (`≥ −max(rho_c,0)/ts`) so the diagnostic ρ_c cannot be driven
negative; condensation is limited by the available vapor (`≤ rho_v/ts`), which kills
phantom condensation in dry air where Q_ss tracking drift can otherwise indicate
spurious supersaturation. Subsaturated cloud-free air returns zero.
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

Limiters: cloud evaporation is bounded by the available cloud (`≥ −max(rho_c,0)/ts`),
rain evaporation by the available rain (`≥ −max(rho_r,0)/ts`), and if the combined
condensation would exceed the available vapor both channels are rescaled
proportionally so `Qdot_c + Qdot_r ≤ max(rho_v,0)/ts`. With the rain channel inactive
(`Qdot_r == 0`) the vapor cap reduces to the historical `min(Qdot, rho_v/ts)`, keeping
the single-category [`qss_condensation_rate`](@ref) delegate bit-identical to its
pre-rain behavior.
"""
function qss_condensation_rates(Q_ss, rho_v, rho_c, rho_r, rho_d, Tk, p_hPa, Q_s, ts,
                                N_r, max_N_c=100.0; N_0=0.0)

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

    # No negative water: each condensate's evaporation limited by its own mass
    Qdot_c = max(Qdot_c, -max(rho_c, 0.0) / ts)
    Qdot_r = max(Qdot_r, -max(rho_r, 0.0) / ts)

    # Condensation limited by the available vapor. With the rain channel inactive this
    # is the historical min(); with both active the channels rescale proportionally.
    cap = max(rho_v, 0.0) / ts
    if Qdot_r == 0.0
        Qdot_c = min(Qdot_c, cap)
    elseif Qdot_c + Qdot_r > cap
        scale = cap / (Qdot_c + Qdot_r)
        Qdot_c *= scale
        Qdot_r *= scale
    end
    return (Qdot_c, Qdot_r)
end

import Springsteel: ref_pressure, ref_rho_t, ref_total_energy, ref_qss

# Canonical slot order for the total-energy set (u=4, w=5 in the shared-machinery
# positions used by the other XZ sets).
const MC_VARS = ["p", "rho_d", "rho_t", "u", "w", "E_t", "Q_ss", "rho_r"]

# ── Equation set ───────────────────────────────────────────────────────────────

"""
    mc_driver!(mtile, colstart, colend, t, geom)

Total-energy moist compressible equation set — the geometry-generic master driver.
Prognostic slots (perturbations vs the `PressureReferenceState` except u, w, rho_r,
and v on the cylindrical geometries): p' [Pa], rho_d', rho_t', u, w, E_t' [J/m^3],
Q_ss' [kg/m^3], rho_r (+ tangential v, slot 9, on the cylinders).

Each step the temperature is retrieved from `retrieve_temperature`, the water
partition follows diagnostically (rho_v = Q_ss + rho_vs, rho_c residual), and the
condensation rate is the limited supersaturation relaxation. The energy equation
carries no condensation source (exact first law); the pressure equation's
condensation coefficient is (L_v - R_v*C_pt*T/R_m). See
reference/Scythe_moist_compressible.tex.

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
    # State-dependent vertical acoustic linearization: the implicit pair's
    # coefficients (Pξ, ρ̂_t and the slaved-leg chains) come from the CURRENT
    # column state each step instead of the resting reference, and the
    # Helmholtz matrix is refactorized per column per step. Removes the
    # convective (finite-amplitude) SI ceiling — the resting-reference form
    # leaves δ·Co_z of the grid-scale acoustic operator explicit in a core
    # whose state deviates by δ, fatal in TC deep convection at Co_z ≳ 4
    # (tc/SI_CONVECTIVE_CEILING.md). Default OFF so existing configurations
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
    vv  = mc_v_views(geom, grid, colstart, colend)         # tangential v (cylinders)

    pp = pv.f;      pp_x = pv.f_x;         pp_z = pv.f_z
    rho_dp = rdv.f; rho_dp_x = rdv.f_x;    rho_dp_z = rdv.f_z; rho_dp_zz = rdv.f_zz
    rho_tp = rtv.f; rho_tp_x = rtv.f_x;    rho_tp_z = rtv.f_z; rho_tp_zz = rtv.f_zz
    u = uv.f;       u_x = uv.f_x;          u_z = uv.f_z;       u_zz = uv.f_zz
    w = wv.f;       w_x = wv.f_x;          w_z = wv.f_z;       w_zz = wv.f_zz
    E_tp = etv.f;   E_tp_x = etv.f_x;      E_tp_z = etv.f_z
    Q_ssp = qsv.f;  Q_ssp_x = qsv.f_x;     Q_ssp_z = qsv.f_z
    rho_rp = rrv.f; rho_rp_x = rrv.f_x;    rho_rp_z = rrv.f_z; rho_rp_zz = rrv.f_zz

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
    Tbar = view(refstate.Tbar,:,1)
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
    rho_r = rho_rp

    # Total vertical gradients (perturbation + reference)
    p_z = S.p_z;         @. p_z = pp_z + pbar_z
    rho_d_z = S.rho_d_z; @. rho_d_z = rho_dp_z + rho_dbar_z
    rho_t_z = S.rho_t_z; @. rho_t_z = rho_tp_z + rho_tbar_z
    E_t_z = S.E_t_z;     @. E_t_z = E_tp_z + E_tbar_z
    Q_ss_z = S.Q_ss_z;   @. Q_ss_z = Q_ssp_z + Q_ssbar_z

    # Diagnostic thermodynamic state: T from the total-energy retrieval, then the
    # water partition follows from the prognostic supersaturation.
    ke = S.ke;   mc_ke!(ke, geom, u, w, vv)
    geo = S.geo; @. geo = ke + (gravity * z)
    M = S.M;     @. M = p + E_t - (rho_t * geo)
    Tk = S.Tk;   @. Tk = retrieve_temperature(M, rho_d, rho_t, Q_ss, p, Tbar, rho_r)
    p_hPa = S.p_hPa;   @. p_hPa = p / 100.0
    rho_vs = S.rho_vs; @. rho_vs = rho_v_sat(Tk, p_hPa)
    # Clamped diagnostic partition (see retrieve_temperature): vapor within
    # [0, total water - rain], cloud the residual
    rho_v = S.rho_v; @. rho_v = clamp(Q_ss + rho_vs, 0.0, max(rho_t - rho_d - rho_r, 0.0))
    rho_c = S.rho_c; @. rho_c = rho_t - rho_d - rho_v - rho_r
    q_v = S.q_v;     @. q_v = rho_v / rho_d
    q_l = S.q_l;     @. q_l = (max(rho_c, 0.0) + rho_r) / rho_d
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
    # their inverse timescales (1/tau = 1/tau_c + 1/tau_r). rho_v and rho_c are
    # diagnostic, so no separate condensation-adjustment step is needed after the
    # timestep. With the rain channel inert (N_r = 0), Qdot is bit-identical to the
    # single-category closure and Qdot_r is exactly zero.
    Q_s = S.Q_s;   @. Q_s = Q_s_energy(Tk, p, rho_d, q_v, q_l)
    Qdot = S.Qdot          # cloud channel
    Qdot_r = S.Qdot_r      # rain channel
    for i in 1:length(Qdot)
        Qdot[i], Qdot_r[i] = qss_condensation_rates(Q_ss[i], rho_v[i], rho_c[i], rho_r[i],
                                                    rho_d[i], Tk[i], p_hPa[i], Q_s[i],
                                                    model.ts, N_r; N_0=N_0)
        if isnan(Qdot[i]) || isnan(Qdot_r[i])
            error("Qdot is NaN at index $i, time $(t)!")
        end
    end

    # Warm-rain conversion and sedimentation. Autoconversion + collection move cloud to
    # rain — a liquid-to-liquid exchange, thermodynamically inert (T, p, E_t, Q_ss all
    # unmoved; cloud is the diagnostic residual, so only slot 8 carries a source). The
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
            # Shared depletion cap: never convert more cloud than exists this step
            AUTO_COLL[i] = min(auto + coll, max(rho_c[i], 0.0) / model.ts)
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

    # Horizontal diabatic heating [W/m^3] from the entropy diffusion: the source to internal
    # energy is rho_d*T*(ds_t/dt)_diff = rho_d*T*Khdiff_heat*d2(s_t)/dx2. This is the
    # moist analogue of Straka's rho*T*ds_d source.
    QDOT_TH = S.QDOT_TH
    @. QDOT_TH = rho_d * Tk * Khdiff_heat * sd_xx

    # Horizontal frictional KE change [W/m^3]. Momentum diffusion is a resolved-KE SINK to
    # the subgrid (the future TKE shear production), NOT dissipative heating: with an eddy K
    # the resolved KE lost goes to unresolved scales, and the molecular heating (proportional
    # to the far-smaller kinematic viscosity) is negligible. So E_t follows the KE down and
    # internal energy (T, p) is held. FRIC_KE is d(rho_t*ke)/dt from horizontal momentum
    # diffusion, added to E_t; p and Q_ss get NO friction term.
    # Smagorinsky horizontal eddy viscosity (Ls > 0): flow-dependent K(strain)
    # replaces the constant Khdiff in the momentum diffusion and its FRIC_KE
    # energy sink below (heat keeps the constant Khdiff_heat).
    K_smag = S.K_smag
    if use_smag
        mc_smag_k!(K_smag, geom, uv, vv, r, Ls, K_min)
    end

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
    # Courant ceiling (see tc/SI_VERTICAL_CEILING.md). The single-φ-fit structure also
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
    if hsi
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
    # QSSREL relaxes Q_ss onto the water-mass-admissible interval; it is exactly zero
    # wherever cloud exists.
    dT_nc = S.dT_nc; @. dT_nc = ((-p * div) + QDOT_TH) / (rho_d * C_vt)
    dp_nc = S.dp_nc; @. dp_nc = (-gamma_m * p * div) + ((R_m / C_vt) * QDOT_TH)
    SATF = S.SATF;   @. SATF = (-rho_vs * div) - (drvs_dT * dT_nc) - (drvs_dp * dp_nc)
    QSSREL = S.QSSREL
    @. QSSREL = qss_relaxation(Q_ss, rho_d, rho_t, rho_r, rho_vs, tau_qss)
    mc_advect!(ADV, geom, u, w, vv, r, Q_ssp_x, Q_ss_z, qsv.f_l)
    FORCING .= @. (-Q_ss * div) + SATF - ((Qdot + Qdot_r) * (1.0 + Q_s)) + QSSREL
    @turbo expdot[colstart:colend,7] .= @. ADV + FORCING

    # Rain partial density (slot 8): rain-channel condensation/evaporation,
    # autoconversion + collection from cloud, and the sedimentation flux divergence
    # (no diffusion yet — water-species mixing arrives with the moist diffusion).
    mc_advect!(ADV, geom, u, w, vv, r, rho_rp_x, rho_rp_z, rrv.f_l)
    @turbo FORCING .= @. (-rho_r * div) + Qdot_r + AUTO_COLL - Fr_z
    @turbo expdot[colstart:colend,8] .= @. ADV + FORCING

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
        mc_louis_bl!(mtile, S, geom, colstart, colend, z, uv, wv, vv, rtv, rdv,
                     expdot, l_inf, Cd_param, sfc_wind_factor,
                     surface_fluxes, Ck, SST, U_min)
    end

    # ── Implicit vertical diffusion tendencies (AI2* history in the diffdot channel) ──
    # Vertical diffusion must be implicit on a Chebyshev column, where the spectral
    # second-derivative eigenvalues scale as N^4. The acoustic solver owns impdot[w],
    # impdot[p] and impdot[E_t], so these live in diffdot instead. Slot 6 (E_t) carries the
    # HEAT tendency in entropy space (not an energy tendency): diffusion_timestep_mc solves
    # for s_t' and maps the increment onto (p, E_t, Q_ss). Slots 3/7/8 carry the water
    # tendencies (total water rho_w' = rho_t' - rho_d', vapor rho_v', rain rho_r).
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
            # Total water and rain are combinations of prognostic slots, so their ∂zz
            # comes straight from the grid's derivative slots; vapor is diagnosed, so it
            # takes the column transform like s_t. rho_v' rides on rho_t's operator
            # (scratch column 3) to match the implicit `water` matrix.
            @turbo diffdot[colstart:colend,3] .= @. Kvdiff_water * (rho_tp_zz - rho_dp_zz)
            @turbo diffdot[colstart:colend,8] .= @. Kvdiff_water * rho_rp_zz
            v_col = scratch_column(mtile, 3)
            v_col.uMish .= rho_v .- mtile.mc_ref_diag.rho_vbar
            Btransform!(v_col)
            Atransform!(v_col)
            stage_zz = S.stage_zz
            Ixxtransform(v_col, stage_zz)
            @turbo diffdot[colstart:colend,7] .= Kvdiff_water .* stage_zz
        end
    end

    # Advance the explicit terms
    explicit_timestep(mtile, colstart, colend, t)

    # Explicit AI2* history levels of the HORIZONTAL acoustic legs: both
    # dimensions' history levels belong to the star state before either implicit
    # solve (the ADI factorization applies the vertical solve below first, then
    # the horizontal patch-level sweep after the spectral merge).
    if hsi
        horizontal_si_history!(mtile, colstart, colend, t)
    end

    # Semi-implicit (p', ρ̄_t w) acoustic solve — unconditional: the explicit acoustic
    # mode was removed (expdot carries only the remainder; the linear vertical acoustic
    # terms are integrated here and nowhere else).
    semiimplicit_adjustment_p(mtile, colstart, colend, t)

    # Implicit vertical diffusion of u, w (friction sink), the moist entropy s_t' (heat,
    # slaved onto p, E_t, Q_ss) and the water species (rho_w', rho_v', rho_r). Skipped
    # when every coefficient is zero: a vertical solve is not the identity there — it
    # refits the column and reapplies the spectral filter.
    if Kvdiff > 0.0 || Kvdiff_heat > 0.0 || Kvdiff_water > 0.0
        diffusion_timestep_mc(mtile, colstart, colend, t, geom)
    end

end

# ── Name-dispatched equation-set wrappers (physical_model resolves the config's
#    equation_set string to one of these by name; all keep the moist_compressible
#    prefix so uses_pressure_reference gates their scratch/reference plumbing) ──

"Total-energy moist compressible set on a Cartesian XZ slice (RiRk/RZ grid), 8 vars."
moist_compressible_XZ(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64) =
    mc_driver!(mtile, colstart, colend, t, MCCartesianXZ())

"""
Axisymmetric r–z cylinder on the RiRk/RZ grid (gridpoint column 1 reinterpreted as
radius, so the domain must sit at r > 0), with prognostic tangential wind v
(`MC_VARS_CYL`, 9 vars) and optional f-plane rotation (`physical_params[:f]`).
"""
moist_compressible_axisym(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64) =
    mc_driver!(mtile, colstart, colend, t, MCAxisymRZ())

"3D r–λ–z cylinder on the RLR grid (`MC_VARS_CYL`, 9 vars, f-plane rotation optional)."
moist_compressible_RLR(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64) =
    mc_driver!(mtile, colstart, colend, t, MCCylindricalRLR())

"""
3D Cartesian x–y–z box on the RRR grid (`MC_VARS_CYL`, 9 vars: v is the y-wind,
f-plane rotation optional — +f v / −f u with no curvature terms).
"""
moist_compressible_RRR(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64) =
    mc_driver!(mtile, colstart, colend, t, MCCartesianRRR())

"""
3D spherical θ–λ–z shell on the SLR grid (`MC_VARS_CYL`, 9 vars: u is the
θ-ward wind, v the zonal wind). Shallow atmosphere with metric radius
`physical_params[:sphere_radius]` (default Earth) and full latitude-dependent
Coriolis f = 2Ω cosθ from `physical_params[:Omega]` (NOT the cylinders' f-plane
`:f`). See `MCSphericalSLR`.
"""
moist_compressible_SLR(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64) =
    mc_driver!(mtile, colstart, colend, t, MCSphericalSLR())

"""
    qss_relaxation(Q_ss, rho_d, rho_t, rho_r, rho_vs, tau)

Relaxation of the prognostic supersaturation density onto the water-mass-admissible
interval of [`qss_admissible_bounds`](@ref), `-(Q_ss - clamp(Q_ss, Q_lo, Q_hi))/τ`
[kg/m³/s].

This is a consistency restoration, not a numerical limiter: `Q_ss = ρ_v − ρ_vs` and the
vapor density is bounded by the prognostic water masses, so an excursion outside
`[Q_lo, Q_hi]` is discretization drift and nothing else. The term is

- exactly zero in cloudy air, the strict interior of the interval, where Q_ss carries the
  vapor/cloud partition and is load-bearing;
- thermodynamically inert wherever it does fire, because `retrieve_temperature`'s clamp is
  already saturated there and T therefore does not depend on Q_ss;
- inactive on supersaturated cloud-free air, which sits exactly at the ceiling `Q_hi > 0`,
  so nucleation is not suppressed;
- free of any effect on the conserved rho_d, rho_t and E_t.

In dry air the interval is the single point `-ρ_vs`, which is where Q_ss belongs: with no
water, ρ_v ≡ 0 regardless of Q_ss, and nothing else restores it.
"""
function qss_relaxation(Q_ss, rho_d, rho_t, rho_r, rho_vs, tau)

    Q_lo, Q_hi = qss_admissible_bounds(rho_d, rho_t, rho_r, rho_vs)
    return -(Q_ss - clamp(Q_ss, Q_lo, Q_hi)) / tau
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
function semiimplicit_adjustment_p(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

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
    # in a convective core with state deviation δ (tc/SI_CONVECTIVE_CEILING.md).
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
    # of tc/SI_VERTICAL_CEILING.md). The first step is AM2 trapezoidal (+0.5 Lⁿ with the
    # ts_term = 0.5·ts solve); its history seed gives t == 2 the full AI2* weights.
    ts_term = (t == 1) ? 0.5 * ts : 1.25 * ts
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
(heat, `Kvdiff_heat`) and the water species (`Kvdiff_water`: total water `rho_w'`, vapor
`rho_v'`, rain `rho_r`) for the total-energy set, using the AI2* off-centered weights
(`+1.25 N^{n+1} - 1.0 N^n + 0.75 N^{n-1}`) with the tendency history in `mtile.diffdot_*`.
It needs its own tendency channel because the acoustic solve owns `impdot[w/p/E_t]`.
The momentum, heat and water solves are gated independently on their coefficients: a K = 0
solve is not the identity (it refits the column and reapplies the spectral filter), so a
zero coefficient must skip its solve entirely, leaving the post-acoustic state untouched.

Runs AFTER [`semiimplicit_adjustment_p`](@ref). The density fields are NOT re-slaved to the
post-diffusion `w` — momentum diffusion is a force, not a mass flux, and they were already
advanced with the acoustic flux divergence (which conserves `∫rho_t'`). The residual is the
usual O(ts·Kv) splitting error.

The heat and water paths retrieve the post-acoustic temperature (the star state) with the
full Newton retrieval; the resting base stays bit-preserved because the reference profiles
`s_tbar`/`rho_vbar` come from the same pipeline (`mc_reference_diagnostics`).

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
  `δQ_ss = δρ_v − ∂ρ_vs/∂p · δp`, with `δρ_c = δρ_w − δρ_v − δρ_r` implicit in the
  residual. Each species' Neumann solve conserves its own `∫ρ`; `∫E_t` is conserved only
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
    rhor_index = vars["rho_r"]

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
    Q_ssbar = view(ref_qss(mtile.ref_state),:,1)
    Tbar = view(mtile.ref_state.Tbar,:,1)
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
    if do_heat || do_water
        E_t_star = S.df_E_t_star;   @. E_t_star = et_v + E_tbar
        Q_ss_star = S.df_Q_ss_star; @. Q_ss_star = qss_v + Q_ssbar
        mc_ke_star!(ke_star, geom, u_star, w_star, v_star)
        M_star = S.df_M_star
        @. M_star = p_star + E_t_star - (rho_t_star * (ke_star + (gravity * z)))
        @. T_star = retrieve_temperature(M_star, rho_d_star, rho_t_star, Q_ss_star,
                                         p_star, Tbar, rhor_v)
        @. p_hPa_star = p_star / 100.0
        @. drvs_dT = drho_vsat_dT(T_star, p_hPa_star)
        @. drvs_dp = drho_vsat_dp(T_star, p_hPa_star)
        rho_vs_star = S.df_rho_vs_star
        @. rho_vs_star = rho_v_sat(T_star, p_hPa_star)
        @. rho_v_star = clamp(Q_ss_star + rho_vs_star, 0.0,
                              max(rho_t_star - rho_d_star - rhor_v, 0.0))
        q_v_star = S.df_q_v_star
        q_l_star = S.df_q_l_star
        @. q_v_star = rho_v_star / rho_d_star
        @. q_l_star = (max(rho_t_star - rho_d_star - rho_v_star - rhor_v, 0.0) + rhor_v) /
                      rho_d_star
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
        _diffusion_water_step!(mtile, S, col, mats, ts, t, colstart, colend, z,
                               rhod_v, rhot_v, rhor_v, p_v, qss_v)
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
                           rhod_v, rhot_v, rhor_v, p_v, qss_v)

The water-species half of [`diffusion_timestep_mc`](@ref): AM2/AI2* staging and
implicit solves for rho_w' and rho_v' (both on the rho_t operator) and rho_r (its
own), then the fixed-temperature increment maps onto rho_t, rho_r, p and Q_ss.
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
                                          rhod_v, rhot_v, rhor_v, p_v, qss_v)
    vars = mtile.model.grid_params.vars
    rhot_index = vars["rho_t"]
    qss_index = vars["Q_ss"]
    rhor_index = vars["rho_r"]

    T_star = S.df_T_star
    Lv_star = S.df_Lv_star
    ke_star = S.df_ke_star
    drvs_dp = S.df_drvs_dp
    rho_v_star = S.df_rho_v_star
    dE_w = S.df_dE_w

    rw_star = S.df_rw_star; @. rw_star = rhot_v - rhod_v
    rv_star = S.df_rv_star; @. rv_star = rho_v_star - mtile.mc_ref_diag.rho_vbar

    rwdot_n = view(mtile.diffdot_n,colstart:colend,rhot_index)
    rwdot_nm1 = view(mtile.diffdot_nm1,colstart:colend,rhot_index)
    rvdot_n = view(mtile.diffdot_n,colstart:colend,qss_index)
    rvdot_nm1 = view(mtile.diffdot_nm1,colstart:colend,qss_index)
    rrdot_n = view(mtile.diffdot_n,colstart:colend,rhor_index)
    rrdot_nm1 = view(mtile.diffdot_nm1,colstart:colend,rhor_index)

    rw_nstar = S.df_rw_nstar
    rv_nstar = S.df_rv_nstar
    rr_nstar = S.df_rr_nstar
    if (t == 1)
        @. rw_nstar = rw_star + (ts * 0.5 * rwdot_n)
        @. rv_nstar = rv_star + (ts * 0.5 * rvdot_n)
        @. rr_nstar = rhor_v + (ts * 0.5 * rrdot_n)
    else
        @. rw_nstar = rw_star - (ts * rwdot_n) + (ts * 0.75 * rwdot_nm1)
        @. rv_nstar = rv_star - (ts * rvdot_n) + (ts * 0.75 * rvdot_nm1)
        @. rr_nstar = rhor_v - (ts * rrdot_n) + (ts * 0.75 * rrdot_nm1)
    end
    rwdot_nm1 .= rwdot_n
    rvdot_nm1 .= rvdot_n
    rrdot_nm1 .= rrdot_n

    h_rw = (t == 1) ? mats.water_first : mats.water
    h_rr = (t == 1) ? mats.water_r_first : mats.water_r
    rw_np1 = S.df_rw_np1
    rv_np1 = S.df_rv_np1
    _vertical_solve!(col, h_rw, rw_nstar, mtile)
    copyto!(rw_np1, Itransform!(col))
    _vertical_solve!(col, h_rw, rv_nstar, mtile)
    copyto!(rv_np1, Itransform!(col))
    _vertical_solve!(col, h_rr, rr_nstar, mtile)
    rr_np1 = Itransform!(col)

    drw = S.df_drw; @. drw = rw_np1 - rw_star
    drv = S.df_drv; @. drv = rv_np1 - rv_star
    drr = S.df_drr; @. drr = rr_np1 - rhor_v
    @. dE_w = (((Cpv * T_star) - Lv_star + ke_star + (gravity * z)) * drw) +
              ((Lv_star - (Rv * T_star)) * drv)

    rhot_v .+= drw
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
                          xc=10000.0, xr=2000.0, zc=2000.0, zr=2000.0, dtheta_max=2.0)
    vars = patch.params.vars
    p_i = vars["p"]; rho_d_i = vars["rho_d"]; rho_t_i = vars["rho_t"]
    et_i = vars["E_t"]; qss_i = vars["Q_ss"]; rho_r_i = vars["rho_r"]
    kDim = patch.params.kDim
    pbar = ref_pressure(ref); rho_dbar = ref_rho_d(ref); rho_tbar = ref_rho_t(ref)
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
            patch.physical[i, rho_r_i, 1] = 0.0
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
                                xc=0.0, xr=4000.0, zc=3000.0, zr=2000.0, dT_max=-15.0)
    vars = patch.params.vars
    p_i = vars["p"]; rho_d_i = vars["rho_d"]; rho_t_i = vars["rho_t"]
    et_i = vars["E_t"]; qss_i = vars["Q_ss"]; rho_r_i = vars["rho_r"]
    kDim = patch.params.kDim
    pbar = ref_pressure(ref); rho_dbar = ref_rho_d(ref); rho_tbar = ref_rho_t(ref)
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
            patch.physical[i, rho_r_i, 1] = 0.0
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
                                      dT_max=3.0, zcol=2)
    vars = patch.params.vars
    p_i = vars["p"]; rho_d_i = vars["rho_d"]; rho_t_i = vars["rho_t"]
    et_i = vars["E_t"]; qss_i = vars["Q_ss"]; rho_r_i = vars["rho_r"]
    kDim = patch.params.kDim
    pbar = ref_pressure(ref); rho_dbar = ref_rho_d(ref); rho_tbar = ref_rho_t(ref)
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
            patch.physical[i, rho_r_i, 1] = 0.0
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
                                   amp=2.0/300.0)
    vars = patch.params.vars
    p_i = vars["p"]; rho_d_i = vars["rho_d"]; rho_t_i = vars["rho_t"]
    et_i = vars["E_t"]; qss_i = vars["Q_ss"]; rho_r_i = vars["rho_r"]
    kDim = patch.params.kDim
    pbar = ref_pressure(ref); rho_dbar = ref_rho_d(ref); rho_tbar = ref_rho_t(ref)
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
            patch.physical[i, rho_r_i, 1] = 0.0
            i += 1
        end
    end
    return patch
end
