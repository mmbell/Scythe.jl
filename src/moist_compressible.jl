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
    dry_potential_temperature(p_Pa, rho_d)

The potential temperature θ_d ≡ (p_0^κ / R_d) p^(1−κ) / ρ_d, κ = R_d/C_pd, built purely
from the prognostic pressure and dry-air density so no diagnosed field needs a spectral
transform. In dry air this is exactly Straka's θ = T (p_0/p)^κ; in moist air it evaluates
to (R_m/R_d) T (p_0/p)^κ, a dry-density-referenced virtual potential temperature. This is
the quantity the mc set diffuses; see the diffusion section of
reference/Scythe_moist_compressible.tex for the moist caveat.
"""
function dry_potential_temperature(p_Pa, rho_d)

    kappa = Rd / Cpd
    return ((100.0 * p_0)^kappa) * (p_Pa^(1.0 - kappa)) / (Rd * rho_d)
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
function qss_condensation_rate(Q_ss, rho_v, rho_c, rho_d, Tk, p_hPa, Q_s, ts, max_N_c=100.0)

    rho_vs = rho_v_sat(Tk, p_hPa)
    S = Q_ss / rho_vs                    # supersaturation (ratio - 1)
    q_c = max(rho_c, 0.0) / rho_d

    # Droplet number and radius logic mirrors q_condensation
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
    elseif S > 1.0e-4
        # Nucleation: linear interpolation of the Twomey relationship
        if r_c < 1.0
            r_c = 1.0
            N_c = min(1.0e4 * N_c * S, max_N_c)
        end
    else
        # No cloud and not supersaturated: nothing to condense or evaporate
        return 0.0
    end
    if N_c <= 0.0 || r_c <= 0.0
        return 0.0
    end

    invtau = invtau_condensation(Tk, p_hPa, N_c, r_c)
    Qdot = Q_ss * invtau / (1.0 + Q_s)

    # No negative water: evaporation limited by cloud, condensation by vapor
    Qdot = max(Qdot, -max(rho_c, 0.0) / ts)
    Qdot = min(Qdot, max(rho_v, 0.0) / ts)
    return Qdot
end

import Springsteel: ref_pressure, ref_rho_t, ref_total_energy, ref_qss

# Canonical slot order for the total-energy set (u=4, w=5 in the shared-machinery
# positions used by the other XZ sets).
const MC_VARS = ["p", "rho_d", "rho_t", "u", "w", "E_t", "Q_ss", "rho_r"]

# ── Equation set ───────────────────────────────────────────────────────────────

"""
    moist_compressible_XZ(mtile, colstart, colend, t)

Total-energy moist compressible equation set on an XZ slice. Prognostic slots
(perturbations vs the `PressureReferenceState` except u, w, rho_r):
p' [Pa], rho_d', rho_t', u, w, E_t' [J/m^3], Q_ss' [kg/m^3], rho_r.

Each step the temperature is retrieved from `retrieve_temperature`, the water
partition follows diagnostically (rho_v = Q_ss + rho_vs, rho_c residual), and the
condensation rate is the limited supersaturation relaxation. The energy equation
carries no condensation source (exact first law); the pressure equation's
condensation coefficient is (L_v - R_v*C_pt*T/R_m). See
reference/Scythe_moist_compressible.tex.
"""
function moist_compressible_XZ(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    grid = mtile.tile
    gridpoints = mtile.tilepoints
    expdot = mtile.expdot_n
    impdot = mtile.impdot_n
    model = mtile.model
    refstate = mtile.ref_state

    # Physical parameters. Momentum and the moist entropy s_t (heat) are the diffused
    # quantities: the masses, pressure, total energy and Q_ss carry no diffusive tendency
    # of their own (the heat/friction increments are slaved onto them). Heat diffusivity is
    # Kvdiff/Prandtl. Water-species mixing is deferred (see the handoff doc).
    Khdiff = model.physical_params[:Khdiff]
    Kvdiff = model.physical_params[:Kvdiff]
    Prandtl = get(model.physical_params, :Prandtl, 1.0)
    tau_qss = get(model.physical_params, :tau_qss, 10.0)

    # Gridpoints
    x = view(gridpoints,colstart:colend,1)
    z = view(gridpoints,colstart:colend,2)

    # Slot 1 is the pressure perturbation p' [Pa]
    pp = view(grid.physical,colstart:colend,1,1)
    pp_x = view(grid.physical,colstart:colend,1,2)
    pp_xx = view(grid.physical,colstart:colend,1,3)
    pp_z = view(grid.physical,colstart:colend,1,4)
    pp_zz = view(grid.physical,colstart:colend,1,5)

    # Slot 2 is the dry-air density perturbation rho_d'
    rho_dp = view(grid.physical,colstart:colend,2,1)
    rho_dp_x = view(grid.physical,colstart:colend,2,2)
    rho_dp_xx = view(grid.physical,colstart:colend,2,3)
    rho_dp_z = view(grid.physical,colstart:colend,2,4)
    rho_dp_zz = view(grid.physical,colstart:colend,2,5)

    # Slot 3 is the total density perturbation rho_t'
    rho_tp = view(grid.physical,colstart:colend,3,1)
    rho_tp_x = view(grid.physical,colstart:colend,3,2)
    rho_tp_xx = view(grid.physical,colstart:colend,3,3)
    rho_tp_z = view(grid.physical,colstart:colend,3,4)
    rho_tp_zz = view(grid.physical,colstart:colend,3,5)

    u = view(grid.physical,colstart:colend,4,1)
    u_x = view(grid.physical,colstart:colend,4,2)
    u_xx = view(grid.physical,colstart:colend,4,3)
    u_z = view(grid.physical,colstart:colend,4,4)
    u_zz = view(grid.physical,colstart:colend,4,5)

    w = view(grid.physical,colstart:colend,5,1)
    w_x = view(grid.physical,colstart:colend,5,2)
    w_xx = view(grid.physical,colstart:colend,5,3)
    w_z = view(grid.physical,colstart:colend,5,4)
    w_zz = view(grid.physical,colstart:colend,5,5)

    # Slot 6 is the total energy density perturbation E_t'
    E_tp = view(grid.physical,colstart:colend,6,1)
    E_tp_x = view(grid.physical,colstart:colend,6,2)
    E_tp_xx = view(grid.physical,colstart:colend,6,3)
    E_tp_z = view(grid.physical,colstart:colend,6,4)
    E_tp_zz = view(grid.physical,colstart:colend,6,5)

    # Slot 7 is the supersaturation density perturbation Q_ss'
    Q_ssp = view(grid.physical,colstart:colend,7,1)
    Q_ssp_x = view(grid.physical,colstart:colend,7,2)
    Q_ssp_xx = view(grid.physical,colstart:colend,7,3)
    Q_ssp_z = view(grid.physical,colstart:colend,7,4)
    Q_ssp_zz = view(grid.physical,colstart:colend,7,5)

    # Slot 8 is the rain partial density rho_r (reference rho_rbar = 0)
    rho_rp = view(grid.physical,colstart:colend,8,1)
    rho_rp_x = view(grid.physical,colstart:colend,8,2)
    rho_rp_xx = view(grid.physical,colstart:colend,8,3)
    rho_rp_z = view(grid.physical,colstart:colend,8,4)
    rho_rp_zz = view(grid.physical,colstart:colend,8,5)

    # Reference state (pressure-based)
    pbar = ref_pressure(refstate)[:,1]
    pbar_z = ref_pressure(refstate)[:,2]
    rho_dbar = ref_rho_d(refstate)[:,1]
    rho_dbar_z = ref_rho_d(refstate)[:,2]
    rho_tbar = ref_rho_t(refstate)[:,1]
    rho_tbar_z = ref_rho_t(refstate)[:,2]
    E_tbar = ref_total_energy(refstate)[:,1]
    E_tbar_z = ref_total_energy(refstate)[:,2]
    Q_ssbar = ref_qss(refstate)[:,1]
    Q_ssbar_z = ref_qss(refstate)[:,2]
    Tbar = refstate.Tbar[:,1]
    Pxi_bar = sound_speed_sq(refstate)

    # Total fields (perturbation + reference)
    p = pp .+ pbar
    rho_d = rho_dp .+ rho_dbar
    rho_t = rho_tp .+ rho_tbar
    E_t = E_tp .+ E_tbar
    Q_ss = Q_ssp .+ Q_ssbar
    rho_r = rho_rp

    # Total vertical gradients (perturbation + reference)
    p_z = pp_z .+ pbar_z
    rho_d_z = rho_dp_z .+ rho_dbar_z
    rho_t_z = rho_tp_z .+ rho_tbar_z
    E_t_z = E_tp_z .+ E_tbar_z
    Q_ss_z = Q_ssp_z .+ Q_ssbar_z

    # Diagnostic thermodynamic state: T from the total-energy retrieval, then the
    # water partition follows from the prognostic supersaturation.
    ke = @. 0.5 * ((u * u) + (w * w))
    geo = @. ke + (gravity * z)
    M = @. p + E_t - (rho_t * geo)
    Tk = retrieve_temperature.(M, rho_d, rho_t, Q_ss, p, Tbar, rho_r)
    p_hPa = p ./ 100.0
    rho_vs = rho_v_sat.(Tk, p_hPa)
    # Clamped diagnostic partition (see retrieve_temperature): vapor within
    # [0, total water - rain], cloud the residual
    rho_v = clamp.(Q_ss .+ rho_vs, 0.0, max.(rho_t .- rho_d .- rho_r, 0.0))
    rho_c = rho_t .- rho_d .- rho_v .- rho_r
    q_v = rho_v ./ rho_d
    q_l = (max.(rho_c, 0.0) .+ rho_r) ./ rho_d
    C_vt = @. Cvd + (q_v * Cvv) + (q_l * Cl)
    R_m = @. Rd + (q_v * Rv)
    C_pt = C_vt .+ R_m
    gamma_m = C_pt ./ C_vt
    Lv = L_v.(Tk)
    drvs_dT = drho_vsat_dT.(Tk, p_hPa)
    drvs_dp = drho_vsat_dp.(Tk, p_hPa)

    # Condensation: limited supersaturation relaxation with the energy-consistent
    # psychrometric factor. rho_v and rho_c are diagnostic, so no separate
    # condensation-adjustment step is needed after the timestep.
    Q_s = Q_s_energy.(Tk, p, rho_d, q_v, q_l)
    Qdot = qss_condensation_rate.(Q_ss, rho_v, rho_c, rho_d, Tk, p_hPa, Q_s, model.ts)
    for i in 1:length(Qdot)
        if isnan(Qdot[i])
            error("Qdot is NaN at index $i, time $(t)!")
        end
    end

    div = u_x .+ w_z

    # ── Horizontal diffusion ───────────────────────────────────────────────────
    # Turbulence diffuses momentum (u, w) and the moist entropy s_t (heat). Horizontally,
    # s_t is diffused by the chain rule on the DRY-exact form s_d = C_vd ln p - C_pd ln rho_d
    # (an explicit function of the prognostic p, rho_d). Its Laplacian uses only the p/rho_d
    # derivative slots — no transform of a diagnosed field, which the column decomposition
    # cannot do (see reference/moist_compressible_diffusion_plan.md). The MOIST horizontal
    # correction (through the retrieval's T sensitivities) is deferred to the rainfall
    # session (reference/moist_compressible_diffusion_handoff.md). pbar/rho_dbar have no
    # x-dependence, so the total x-derivatives are the perturbation slots.
    sd_xx = @. (Cvd * ((pp_xx / p) - (pp_x * pp_x / (p * p)))) -
               (Cpd * ((rho_dp_xx / rho_d) - (rho_dp_x * rho_dp_x / (rho_d * rho_d))))

    # Horizontal diabatic heating [W/m^3] from the entropy diffusion: the source to internal
    # energy is rho_d*T*(ds_t/dt)_diff = rho_d*T*(Khdiff/Prandtl)*d2(s_t)/dx2. This is the
    # moist analogue of Straka's rho*T*ds_d source.
    QDOT_TH = @. rho_d * Tk * (Khdiff / Prandtl) * sd_xx

    # Horizontal frictional KE change [W/m^3]. Momentum diffusion is a resolved-KE SINK to
    # the subgrid (the future TKE shear production), NOT dissipative heating: with an eddy K
    # the resolved KE lost goes to unresolved scales, and the molecular heating (proportional
    # to the far-smaller kinematic viscosity) is negligible. So E_t follows the KE down and
    # internal energy (T, p) is held. FRIC_KE is d(rho_t*ke)/dt from horizontal momentum
    # diffusion, added to E_t; p and Q_ss get NO friction term.
    FRIC_KE = @. rho_t * Khdiff * ((u * u_xx) + (w * w_xx))

    # Placeholders for intermediate calculations
    ADV = similar(Tk)
    FORCING = similar(Tk)
    KDIFF = similar(Tk)

    # Pressure (slot 1): -v·∇p - γp∇·v + (R_m/C_vt)[(L_v - R_v C_pt T/R_m) Q̇_cond + Q̇_therm].
    # Only the THERMAL diffusion sources pressure (friction holds T, hence p).
    @turbo ADV .= @. (-u * pp_x) + (-w * p_z)
    FORCING .= @. (-gamma_m * p * div) +
                  ((R_m / C_vt) * (((Lv - (Rv * C_pt * Tk / R_m)) * Qdot) + QDOT_TH))
    @turbo expdot[colstart:colend,1] .= @. ADV + FORCING

    # Implicit acoustic tendencies from the vertical mass flux φ = ρ̄_t w, in
    # pointwise product-rule form ∂z(c w) = c w_z + c_z w (no column refits: the
    # refit reapplies the spectral filter, which makes the rho_d and rho_t paths
    # inconsistent and lets the two densities drift apart in dry air).
    impdot[colstart:colend,1] .= @. -Pxi_bar * ((rho_tbar * w_z) + (rho_tbar_z * w))
    impdot[colstart:colend,3] .= @. -((rho_tbar * w_z) + (rho_tbar_z * w))

    # Dry-air mass continuity (slot 2, advective product-rule form; no mass diffusion)
    @turbo ADV .= @. (-u * rho_dp_x) + (-w * rho_d_z)
    @turbo FORCING .= @. -rho_d * div
    @turbo expdot[colstart:colend,2] .= @. ADV + FORCING
    # Implicit acoustic continuity: -∂z(ρ̄_d w), product-rule form
    impdot[colstart:colend,2] .= @. -((rho_dbar * w_z) + (rho_dbar_z * w))

    # Total mass continuity (slot 3; sources zero without flux/sedimentation)
    @turbo ADV .= @. (-u * rho_tp_x) + (-w * rho_t_z)
    @turbo FORCING .= @. -rho_t * div
    @turbo expdot[colstart:colend,3] .= @. ADV + FORCING

    # u momentum (slot 4): PGF directly from the prognostic pressure
    @turbo ADV .= @. (-u * u_x) + (-w * u_z)
    @turbo FORCING .= @. -pp_x / rho_t
    @turbo KDIFF .= @. Khdiff * u_xx
    @turbo expdot[colstart:colend,4] .= @. ADV + FORCING + KDIFF

    # w momentum (slot 5): perturbation PGF + total-density buoyancy loading
    @turbo ADV .= @. (-u * w_x) + (-w * w_z)
    @turbo FORCING .= @. ((-gravity * rho_tp) - pp_z) / rho_t
    @turbo KDIFF .= @. Khdiff * w_xx
    @turbo expdot[colstart:colend,5] .= @. ADV + FORCING + KDIFF
    # Implicit acoustic w-momentum: -(1/ρ̄_t) ∂z p'
    impdot[colstart:colend,5] .= @. -pp_z / rho_tbar

    # Total energy (slot 6): -v·∇E_t - E_t∇·v - ∇·(pv) + Q̇_therm + friction sink. No
    # condensation source (exact first law: phase change is not an energy source). The
    # THERMAL diffusion heats (Q̇_therm); the momentum diffusion adds FRIC_KE = d(rho_t*ke)/dt
    # so E_t follows the resolved KE down to the subgrid (internal energy held). pbar has no
    # x-dependence so u*p_x = u*pp_x.
    @turbo ADV .= @. (-u * E_tp_x) + (-w * E_t_z)
    @turbo FORCING .= @. (-(E_t + p) * div) - (u * pp_x) - (w * p_z)
    @turbo expdot[colstart:colend,6] .= @. ADV + FORCING + QDOT_TH + FRIC_KE
    # Implicit acoustic energy flux: -∂z((Ē_t + p̄) w), product-rule form
    impdot[colstart:colend,6] .= @. -(((E_tbar + pbar) * w_z) + ((E_tbar_z + pbar_z) * w))

    # Supersaturation density (slot 7): the saturation chain-rule terms use the
    # non-condensation T and p tendencies, which carry the horizontal THERMAL diffusive
    # heating as well as the divergence work (friction holds T, so it does not enter; the
    # VERTICAL diffusive heating is applied as a slaved δQ_ss inside diffusion_timestep_mc).
    # The condensation contribution is -Q̇_cond(1+Q_s) (= -Q_ss/τ when the rate is unlimited).
    # QSSREL relaxes Q_ss onto the water-mass-admissible interval; it is exactly zero
    # wherever cloud exists.
    dT_nc = @. ((-p * div) + QDOT_TH) / (rho_d * C_vt)
    dp_nc = @. (-gamma_m * p * div) + ((R_m / C_vt) * QDOT_TH)
    SATF = @. (-rho_vs * div) - (drvs_dT * dT_nc) - (drvs_dp * dp_nc)
    QSSREL = qss_relaxation.(Q_ss, rho_d, rho_t, rho_r, rho_vs, tau_qss)
    @turbo ADV .= @. (-u * Q_ssp_x) + (-w * Q_ss_z)
    FORCING .= @. (-Q_ss * div) + SATF - (Qdot * (1.0 + Q_s)) + QSSREL
    @turbo expdot[colstart:colend,7] .= @. ADV + FORCING

    # Rain partial density (slot 8; microphysics sources deferred, no diffusion)
    @turbo ADV .= @. (-u * rho_rp_x) + (-w * rho_rp_z)
    @turbo FORCING .= @. -rho_r * div
    @turbo expdot[colstart:colend,8] .= @. ADV + FORCING

    # ── Implicit vertical diffusion tendencies (AI2* history in the diffdot channel) ──
    # Vertical diffusion must be implicit on a Chebyshev column, where the spectral
    # second-derivative eigenvalues scale as N^4. The acoustic solver owns impdot[w],
    # impdot[p] and impdot[E_t], so these live in diffdot instead. Slot 6 (E_t) carries the
    # HEAT tendency in entropy space (not an energy tendency): diffusion_timestep_mc solves
    # for s_d' and maps the increment onto (p, E_t, Q_ss).
    #
    # The heat variable is the DRY-exact entropy s_d = C_vd ln p - C_pd ln rho_d, an explicit
    # function of the prognostic p, rho_d — so at rest s_d = s_dbar EXACTLY (no retrieval
    # dependence), the base is bit-preserved, and it matches Straka's dry-entropy diffusion.
    # The full moist entropy s_t (with the vapor/liquid contribution) is DEFERRED to the
    # rainfall session together with the water-species mixing and the moist horizontal terms
    # (see reference/moist_compressible_diffusion_handoff.md).
    s_dbar = dry_entropy_pd.(pbar, rho_dbar)
    if Kvdiff > 0.0
        s_d = dry_entropy_pd.(p, rho_d)
        diffdot = mtile.diffdot_n
        @turbo diffdot[colstart:colend,4] .= @. Kvdiff * u_zz
        @turbo diffdot[colstart:colend,5] .= @. Kvdiff * w_zz
        # ∂zz(s_d') from the column basis, so the explicit AI2* tendency and the implicit
        # Helmholtz operator use the same discrete ∂zz.
        s_col = deepcopy(mtile.tile.kbasis.data[6])
        s_col.uMish .= s_d .- s_dbar
        Btransform!(s_col)
        Atransform!(s_col)
        diffdot[colstart:colend,6] .= (Kvdiff / Prandtl) .* Ixxtransform(s_col)
    end

    # Advance the explicit terms
    explicit_timestep(mtile, colstart, colend, t)

    # Solve for semi-implicit n+1 terms ((p', ρ̄_t w) acoustic adjustment)
    if mtile.model.options[:semiimplicit]
        semiimplicit_adjustment_p(mtile, colstart, colend, t)
    end

    # Implicit vertical diffusion of u, w (friction sink) and the dry entropy s_d' (heat,
    # slaved onto p, E_t, Q_ss). Skipped at Kvdiff = 0: a vertical solve is not the identity
    # there — it refits the column and reapplies the spectral filter.
    if Kvdiff > 0.0
        diffusion_timestep_mc(mtile, colstart, colend, t)
    end

end

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

Semi-implicit acoustic adjustment for the total-energy set. Solves the
constant-coefficient Helmholtz problem for the vertical mass flux `φ = ρ̄_t w` from
the implicit pair `∂φ/∂t = -∂z p'`, `∂p'/∂t = -Pxi_bar ∂z φ` (the same operator as
the rho_d form, so `mtile.h_matrix` is reused), then recovers `w = φ/ρ̄_t` and
`p' = p'* - Δτ Pxi_bar ∂z φ`. The density and energy slots are slaved to the flux
in flux form: `rho_t' -= Δτ ∂z φ` (conserves `∫rho_t'`), `rho_d' -= Δτ ∂z(ρ̄_d/ρ̄_t φ)`,
`E_t' -= Δτ ∂z((Ē_t+p̄)/ρ̄_t φ)`.
"""
function semiimplicit_adjustment_p(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    vars = mtile.model.grid_params.vars
    p_index = vars["p"]
    rhod_index = vars["rho_d"]
    rhot_index = vars["rho_t"]
    w_index = vars["w"]
    et_index = vars["E_t"]
    ts = mtile.model.ts

    # Predictors (copies) and implicit tendency histories (views)
    p_nstar = mtile.var_np1[colstart:colend,p_index]
    w_nstar = mtile.var_np1[colstart:colend,w_index]
    rhod_nstar = mtile.var_np1[colstart:colend,rhod_index]
    rhot_nstar = mtile.var_np1[colstart:colend,rhot_index]
    et_nstar = mtile.var_np1[colstart:colend,et_index]

    # Reference profiles and mean sound speed squared
    Pxi_bar = sound_speed_sq(mtile.ref_state)
    rho_dbar = ref_rho_d(mtile.ref_state)[:,1]
    rho_dbar_z = ref_rho_d(mtile.ref_state)[:,2]
    rho_tbar = ref_rho_t(mtile.ref_state)[:,1]
    rho_tbar_z = ref_rho_t(mtile.ref_state)[:,2]
    E_tbar = ref_total_energy(mtile.ref_state)[:,1]
    E_tbar_z = ref_total_energy(mtile.ref_state)[:,2]
    pbar = ref_pressure(mtile.ref_state)[:,1]
    pbar_z = ref_pressure(mtile.ref_state)[:,2]

    # Subtract the AB3 explicit treatment of the implicit tendency and add the
    # off-centered AI2* terms (AM2 trapezoidal on the first step), then shift the
    # tendency history. Applied identically to each participating slot.
    ts_term = (t == 1) ? 0.5 * ts : 1.25 * ts
    for index in (p_index, w_index, rhod_index, rhot_index, et_index)
        nstar = index == p_index ? p_nstar :
                index == w_index ? w_nstar :
                index == rhod_index ? rhod_nstar :
                index == rhot_index ? rhot_nstar : et_nstar
        dot_n = view(mtile.impdot_n,colstart:colend,index)
        dot_nm1 = view(mtile.impdot_nm1,colstart:colend,index)
        dot_nm2 = view(mtile.impdot_nm2,colstart:colend,index)
        if (t == 1)
            nstar .= @. nstar - (ts * dot_n) + (ts * 0.5 * dot_n)
        elseif (t == 2)
            nstar .= @. nstar - (0.5 * ts) * ((3.0 * dot_n) - dot_nm1) - (ts * dot_n) + (ts * 0.75 * dot_nm1)
        else
            nstar .= @. nstar - ((ts / 12.0) * ((23.0 * dot_n) - (16.0 * dot_nm1) + (5.0 * dot_nm2))) - (ts * dot_n) + (ts * 0.75 * dot_nm1)
        end
        dot_nm2 .= dot_nm1
        dot_nm1 .= dot_n
    end

    # Take the vertical derivative of the p' predictor (coefficient 1: the pair is
    # ∂φ/∂t = -∂z p'; Pxi_bar enters in the p' update instead)
    p_col = deepcopy(mtile.tile.kbasis.data[p_index])
    p_col.uMish .= p_nstar
    Btransform!(p_col)
    Atransform!(p_col)
    p_nstar = Itransform!(p_col)
    p_nstar_z = ts_term .* Ixtransform(p_col)

    # Mass-flux Helmholtz RHS (φ = ρ̄_t w): rhs = Δτ ∂z p'* - ρ̄_t w*.
    # Elimination gives (I - Δτ² Pxi_bar ∂zz) φ, the same operator as the rho_d
    # form, so h_matrix is reused.
    rhs = p_nstar_z .- (rho_tbar .* w_nstar)
    phi_col = deepcopy(mtile.tile.kbasis.data[w_index])
    if t == 1
        # Calculate the Helmholtz matrix for the first time step
        h_a = calc_Helmholtz_semiimplicit_matrix(mtile.tile, mtile.model, Pxi_bar, ts_term)
        _vertical_solve!(phi_col, h_a, rhs, mtile.tile)
    else
        # Use the pre-calculated one
        _vertical_solve!(phi_col, mtile.h_matrix, rhs, mtile.tile)
    end

    phi = Itransform!(phi_col)
    phi_z = Ixtransform(phi_col)

    # Recover w_n+1 = φ_n+1 / ρ̄_t
    view(mtile.var_np1,colstart:colend,w_index) .= phi ./ rho_tbar

    # Recover p'_n+1 = p'* - Δτ Pxi_bar ∂z φ_n+1
    view(mtile.var_np1,colstart:colend,p_index) .= p_nstar .- (ts_term .* Pxi_bar .* phi_z)

    # Slaved flux-form updates ∂z(c φ) = c φ_z + c_z φ, pointwise from the solve's
    # φ and φ_z (no column refits — a refit reapplies the spectral filter and makes
    # the paths inconsistent). rho_t' has c = 1 exactly (conserves ∫rho_t'); with a
    # dry reference the rho_d' update is then IDENTICAL to rho_t''s, so the two
    # densities cannot drift apart.
    view(mtile.var_np1,colstart:colend,rhot_index) .= rhot_nstar .- (ts_term .* phi_z)

    c_d = rho_dbar ./ rho_tbar
    c_d_z = ((rho_dbar_z .* rho_tbar) .- (rho_dbar .* rho_tbar_z)) ./ (rho_tbar .^ 2)
    view(mtile.var_np1,colstart:colend,rhod_index) .=
        rhod_nstar .- (ts_term .* ((c_d .* phi_z) .+ (c_d_z .* phi)))

    c_e = (E_tbar .+ pbar) ./ rho_tbar
    c_e_z = (((E_tbar_z .+ pbar_z) .* rho_tbar) .-
             ((E_tbar .+ pbar) .* rho_tbar_z)) ./ (rho_tbar .^ 2)
    view(mtile.var_np1,colstart:colend,et_index) .=
        et_nstar .- (ts_term .* ((c_e .* phi_z) .+ (c_e_z .* phi)))
end

# ── Implicit vertical diffusion ────────────────────────────────────────────────

"""
    diffusion_timestep_mc(mtile, colstart, colend, t)

Implicit vertical diffusion of `u` and `w` (momentum) and the dry entropy `s_d'` (heat) for
the total-energy set, using the AI2* off-centered weights
(`+1.25 N^{n+1} - 1.0 N^n + 0.75 N^{n-1}`) with the tendency history in `mtile.diffdot_*`.
It needs its own tendency channel because the acoustic solve owns `impdot[w/p/E_t]`.

Runs AFTER [`semiimplicit_adjustment_p`](@ref). The density fields are NOT re-slaved to the
post-diffusion `w` — momentum diffusion is a force, not a mass flux, and they were already
advanced with the acoustic flux divergence (which conserves `∫rho_t'`). The residual is the
usual O(ts·Kv) splitting error.

The heat variable is the dry-exact entropy `s_d = C_vd ln p - C_pd ln rho_d`, so the whole
routine is retrieval-free: `T = p/(R_d rho_d)` from the dry EOS, and at rest `s_d = s_dbar`
exactly (bit-preserved base). The moist entropy `s_t` (and its retrieval) is deferred; see
reference/moist_compressible_diffusion_handoff.md.

Energy routing:

- **Friction is a resolved-KE sink, not heat.** With an eddy diffusivity the resolved KE the
  momentum solve removes goes to the subgrid cascade (the future TKE shear production), and
  the molecular heating is negligible. So `E_t += rho_t δke` (E_t follows the KE down),
  holding internal energy — `T` and `p` are unchanged by friction.
- **Heat.** The `s_d` increment maps at fixed `rho_d` via `∂s_d/∂T = C_vd/T`:
  `δT = (T/C_vd) δs_d`, `δE_t = rho_d T δs_d`, `δp = rho_d R_d δT`, and
  `δQ_ss = -(∂ρ_vs/∂T δT + ∂ρ_vs/∂p δp)` (so `Q_ss = ρ_v - ρ_vs` tracks the diabatic heating).
"""
function diffusion_timestep_mc(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    vars = mtile.model.grid_params.vars
    p_index = vars["p"]
    rhod_index = vars["rho_d"]
    rhot_index = vars["rho_t"]
    u_index = vars["u"]
    w_index = vars["w"]
    et_index = vars["E_t"]
    qss_index = vars["Q_ss"]

    ts = mtile.model.ts

    pbar = ref_pressure(mtile.ref_state)[:,1]
    rho_dbar = ref_rho_d(mtile.ref_state)[:,1]
    rho_tbar = ref_rho_t(mtile.ref_state)[:,1]

    # Post-acoustic totals; dry EOS temperature (no retrieval — s_d is dry-exact)
    u_star = mtile.var_np1[colstart:colend,u_index]
    w_star = mtile.var_np1[colstart:colend,w_index]
    p_star = mtile.var_np1[colstart:colend,p_index] .+ pbar
    rho_d_star = mtile.var_np1[colstart:colend,rhod_index] .+ rho_dbar
    rho_t_star = mtile.var_np1[colstart:colend,rhot_index] .+ rho_tbar
    T_star = @. p_star / (Rd * rho_d_star)
    p_hPa_star = p_star ./ 100.0
    drvs_dT = drho_vsat_dT.(T_star, p_hPa_star)
    drvs_dp = drho_vsat_dp.(T_star, p_hPa_star)
    s_dp_star = dry_entropy_pd.(p_star, rho_d_star) .- dry_entropy_pd.(pbar, rho_dbar)

    # Implicit tendency histories (slot 6 carries the s_d ENTROPY tendency, not an E_t one)
    udot_n = view(mtile.diffdot_n,colstart:colend,u_index)
    udot_nm1 = view(mtile.diffdot_nm1,colstart:colend,u_index)
    wdot_n = view(mtile.diffdot_n,colstart:colend,w_index)
    wdot_nm1 = view(mtile.diffdot_nm1,colstart:colend,w_index)
    sdot_n = view(mtile.diffdot_n,colstart:colend,et_index)
    sdot_nm1 = view(mtile.diffdot_nm1,colstart:colend,et_index)

    local u_nstar, w_nstar, s_nstar
    if (t == 1)
        # Use trapezoidal method (AM2) for first step
        u_nstar = @. u_star + (ts * 0.5 * udot_n)
        w_nstar = @. w_star + (ts * 0.5 * wdot_n)
        s_nstar = @. s_dp_star + (ts * 0.5 * sdot_n)
    else
        # Use AI2* for second step and beyond
        u_nstar = @. u_star - (ts * udot_n) + (ts * 0.75 * udot_nm1)
        w_nstar = @. w_star - (ts * wdot_n) + (ts * 0.75 * wdot_nm1)
        s_nstar = @. s_dp_star - (ts * sdot_n) + (ts * 0.75 * sdot_nm1)
    end

    # Set the n-1 terms
    udot_nm1 .= udot_n
    wdot_nm1 .= wdot_n
    sdot_nm1 .= sdot_n

    # Pre-factorized in createModelTile, per variable and per timestep coefficient (the
    # first step uses the AM2 coefficient), so nothing is factorized per column here.
    mats = mtile.mc_diffusion_matrices
    h_u = (t == 1) ? mats[:u_first] : mats[:u]
    h_w = (t == 1) ? mats[:w_first] : mats[:w]
    h_h = (t == 1) ? mats[:heat_first] : mats[:heat]

    # One scratch column serves all three solves: the boundary conditions live in the
    # factorization, and neither `_vertical_solve!` nor `Itransform!` consults the
    # column's own BCs (only Btransform!/Atransform! do). Same reuse as
    # `diffusion_timestep_pd`, which shares one column across five variables.
    col = deepcopy(mtile.tile.kbasis.data[u_index])
    _vertical_solve!(col, h_u, u_nstar, mtile.tile)
    u_np1 = copy(Itransform!(col))

    _vertical_solve!(col, h_w, w_nstar, mtile.tile)
    w_np1 = copy(Itransform!(col))

    _vertical_solve!(col, h_h, s_nstar, mtile.tile)
    s_dp_np1 = Itransform!(col)

    # Friction: resolved-KE sink, E_t follows the KE down; T, p, Q_ss held.
    dke = @. 0.5 * (((u_np1 * u_np1) + (w_np1 * w_np1)) -
                    ((u_star * u_star) + (w_star * w_star)))
    dE_visc = @. rho_t_star * dke

    # Heat: entropy increment -> (T, p, E_t, Q_ss) at fixed rho_d (dry: C_vt=C_vd, R_m=R_d).
    ds_d = s_dp_np1 .- s_dp_star
    dT_h = @. (T_star / Cvd) * ds_d
    dE_h = @. rho_d_star * T_star * ds_d
    dp_h = @. rho_d_star * Rd * dT_h
    dQ_h = @. -((drvs_dT * dT_h) + (drvs_dp * dp_h))

    view(mtile.var_np1,colstart:colend,u_index) .= u_np1
    view(mtile.var_np1,colstart:colend,w_index) .= w_np1
    view(mtile.var_np1,colstart:colend,p_index) .+= dp_h
    view(mtile.var_np1,colstart:colend,et_index) .+= dE_visc .+ dE_h
    view(mtile.var_np1,colstart:colend,qss_index) .+= dQ_h
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
