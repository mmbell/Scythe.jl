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
    retrieve_temperature(M, rho_d, rho_t, Q_ss, p_Pa, T_guess; tol=1.0e-9, maxiter=25)

Diagnose the temperature from the prognostic variables of the total-energy set by
Newton iteration on

    F(T) = (ρ_d C_pd + (ρ_t − ρ_d) C_pv) T + (Q_ss + ρ_d + ρ_vs(T,p) − ρ_t) L_v(T) − M

where `M = p + E_t − ρ_t(v²/2 + gz)` [J/m³] is the available enthalpy density.
F is monotone increasing in T (all dominant terms positive), so the root is unique;
convergence from the reference or previous-step temperature takes 2-3 iterations.
`tol` is the temperature increment tolerance [K].
"""
function retrieve_temperature(M, rho_d, rho_t, Q_ss, p_Pa, T_guess;
                              tol=1.0e-9, maxiter=25)

    p_hPa = p_Pa / 100.0
    Cfactor = (rho_d * Cpd) + ((rho_t - rho_d) * Cpv)
    Tk = T_guess
    for _ in 1:maxiter
        rho_vs = rho_v_sat(Tk, p_hPa)
        wfactor = Q_ss + rho_d + rho_vs - rho_t          # = rho_v - rho_l (diagnostic)
        F = (Cfactor * Tk) + (wfactor * L_v(Tk)) - M
        Fprime = Cfactor + (L_v(Tk) * drho_vsat_dT(Tk, p_hPa)) +
                 (wfactor * (Cpv - Cl))
        dT = -F / Fprime
        Tk = clamp(Tk + dT, 150.0, 350.0)
        if abs(dT) < tol
            return Tk
        end
    end
    return Tk
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
    qss_condensation_rate(Q_ss, rho_c, rho_d, Tk, p_hPa, Q_s, ts, max_N_c=100.0)

Cloud condensation/evaporation rate [kg/m³/s] from the prognostic supersaturation
density, with the droplet-growth timescale of [`q_condensation`](@ref) (Twomey-type
nucleation, minimum droplet radius) and the energy-consistent psychrometric factor:

    Q̇_cond = Q_ss (1/τ) / (1 + Q_s)

Evaporation is limited by the available cloud water (`≥ −max(rho_c,0)/ts`) so the
diagnostic ρ_c cannot be driven negative; condensation is limited by the available
vapor. Subsaturated cloud-free air returns zero (no droplets, 1/τ = 0).
"""
function qss_condensation_rate(Q_ss, rho_c, rho_d, Tk, p_hPa, Q_s, ts, max_N_c=100.0)

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
    Qdot = min(Qdot, max(Q_ss + rho_vs, 0.0) / ts)
    return Qdot
end
