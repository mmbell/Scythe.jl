# Thermodynamics for Scythe.
#
# The basis-agnostic equation of state and standard atmospheric diagnostics now live
# in the shared `Springsteel.Thermodynamics` submodule (so Daisho and future
# Springsteel-grid codes reuse them without duplication). Scythe imports those physical
# functions here and adds the *model-specific* layer on top: the prognostic-variable
# transforms (`xi` log-density, `mu` vapor) and the pressure derivatives / state
# reconstruction expressed in those control variables.

using Springsteel.Thermodynamics: Rd, Rv, Eps, Cvd, Cvv, Cpd, Cpv, Cl, Ci, gravity,
    L_v0, rho_l, rho_i, T_0, p_0, q0, rho_d0, rho_v0,
    sat_pressure_liquid, sat_pressure_ice, sat_pressure_liquid_buck,
    sat_pressure_liquid_buck_dT, sat_pressure_ice_buck, q_sat_liquid, q_sat_ice,
    L_v, dewpoint, entropy, vapor_entropy, temperature, pressure, vapor_pressure,
    mixing_ratio, dry_density, log_dry_density, P_s

# `potential_temperature`, `reversible_theta_e`, `theta_rho` are NOT imported: Scythe
# keeps transformed-variable (`s`, `xi`, `mu`) adapters of the same name below that
# delegate to the physical Springsteel implementations.

# Water-vapor (mu) prognostic-variable transform mode. `:linear` uses mu = q*1e5
# (the default, exactly matching the existing RZ/Chebyshev results); `:hyperbolic`
# uses the biased hyperbolic transform of Ooyama (2002) (mu = bhyp(q),
# q = ahyp(mu) ≥ 0), which keeps water vapor non-negative under overshoot. Set
# per run via `set_mu_transform!`; on distributed runs it must be set on every
# worker (e.g. `@everywhere Scythe.set_mu_transform!(:hyperbolic)`).
const _MU_HYPERBOLIC = Ref(false)

"""
    set_mu_transform!(mode::Symbol) -> Symbol

Select the water-vapor `mu` transform: `:linear` (mu = q*1e5, default) or
`:hyperbolic` (Ooyama 2002 biased hyperbolic, non-negative q_v). Affects
[`mu_transform`](@ref), [`inv_mu_transform`](@ref) and [`dmudq`](@ref).
"""
function set_mu_transform!(mode::Symbol)
    mode in (:linear, :hyperbolic) ||
        throw(ArgumentError("mu transform mode must be :linear or :hyperbolic, got :$mode"))
    _MU_HYPERBOLIC[] = (mode === :hyperbolic)
    return mode
end

"""
    bhyp(q_v)

Bijective hyperbolic transform of water vapor mixing ratio (Ooyama 2002), mapping
non-negative `q_v` to the prognostic variable `mu`. Inverse of [`ahyp`](@ref).
"""
function bhyp(q_v::Float64)

    mu = 0.5 * ( (q_v + q0) - (q0*q0/(q_v + q0)) )
    return mu
end

"""
    ahyp(mu)

Inverse hyperbolic transform recovering `q_v` from `mu`. Inverse of [`bhyp`](@ref).
Returns zero for negative `mu`.
"""
function ahyp(mu::Float64)

    if (mu < 0.0)
        return 0.0
    else
        q_v = sqrt(mu*mu + q0*q0) + mu - q0
        return q_v
    end
end

"""
    inv_xi_transform(xi)

Apply the inverse xi transform (natural logarithm). Inverse of [`xi_transform`](@ref).
"""
function inv_xi_transform(xi::Float64)

    return log(xi)

end

"""
    xi_transform(q)

Apply the xi transform (exponential). Inverse of [`inv_xi_transform`](@ref).
"""
function xi_transform(q::Float64)

    return exp(q)

end

"""
    dxidq(q)

Derivative of the xi transform with respect to `q`, i.e. `dξ/dq = exp(q)`.
"""
function dxidq(q::Float64)

    return exp(q)

end

"""
    inv_mu_transform(mu)

Recover the water vapor mixing ratio from the scaled prognostic variable `mu`.
Returns zero for negative `mu` (linear mode). Inverse of [`mu_transform`](@ref).

Linear mode: `q = mu * 10⁻⁵`.
"""
function inv_mu_transform(mu::Float64)

    if _MU_HYPERBOLIC[]
        return ahyp(mu)   # Ooyama (2002) biased hyperbolic; q_v ≥ 0
    end
    return mu < 0.0 ? 0.0 : mu * 1.0e-5
end

"""
    mu_transform(q)

Transform a water vapor mixing ratio into the scaled prognostic variable `mu`.
Inverse of [`inv_mu_transform`](@ref). Linear mode: `mu = q * 10⁵`.
"""
function mu_transform(q::Float64)

    return _MU_HYPERBOLIC[] ? bhyp(q) : q * 1.0e5
end

"""
    dmudq(mu, q_v)

Derivative of the `mu` transform with respect to mixing ratio `q_v`, i.e. `dmu/dq_v`.
Linear mode returns the constant `10⁵`.
"""
function dmudq(mu::Float64, q_v::Float64)

    return _MU_HYPERBOLIC[] ? ((q_v + q0) - mu) / (q_v + q0) : 1.0e5
end

"""
    P_xi(Tk, rho_d, q_v)

Partial derivative of pressure with respect to the log-density variable `xi`,
`∂p/∂ξ`, at constant entropy and mixing ratio [hPa].
"""
function P_xi(Tk::Float64, rho_d::Float64, q_v::Float64)

    return (Rd + (q_v * rho_d * Rv)) * ((rho_d * Tk) + P_s(Tk, rho_d, q_v))
end

"""
    P_xi_from_s(s, xi, mu)

Compute `∂p/∂ξ` directly from the prognostic variables (`s`, `xi`, `mu`) by first
recovering the thermodynamic state via [`thermodynamic_tuple`](@ref).
"""
function P_xi_from_s(s::Float64, xi::Float64, mu::Float64)

    q_v, rho_d, Tk, p = thermodynamic_tuple(s, xi, mu)
    return P_xi(Tk, rho_d, q_v)
end

"""
    P_qv(Tk, rho_d, q_v)

Partial derivative of pressure with respect to water vapor mixing ratio, `∂p/∂q_v`,
at constant entropy and dry-air density. Zero when `q_v` is zero.
"""
function P_qv(Tk::Float64, rho_d::Float64, q_v::Float64)

    if (q_v != 0.0)
        rho_v = q_v * rho_d
        qfactor = Rv * (1 + log(rho_v/rho_v0)) - (Cvv * log(Tk/T_0)) - L_v(T_0)/T_0
        qfactor *= P_s(Tk, rho_d, q_v)
        return (rho_d * Rv * Tk) + qfactor
    else
        return 0.0
    end
end

"""
    P_mu(Tk, rho_d, mu)

Partial derivative of pressure with respect to the scaled mixing ratio variable `mu`,
`∂p/∂mu`, via the chain rule through [`P_qv`](@ref) and [`dmudq`](@ref).
"""
function P_mu(Tk::Float64, rho_d::Float64, mu::Float64)

    q_v = inv_mu_transform(mu)
    return P_qv(Tk, rho_d, q_v) / dmudq(mu, q_v)
end

"""
    pressure_gradient(Tk, rho_d, q_v, s_x, xi_x, qv_x)

Pressure gradient in a spatial direction via the chain rule
`∂p/∂x = (∂p/∂s)(∂s/∂x) + (∂p/∂ξ)(∂ξ/∂x) + (∂p/∂q_v)(∂q_v/∂x)`.
"""
function pressure_gradient(Tk::Float64, rho_d::Float64, q_v::Float64,
        s_x::Float64, xi_x::Float64, qv_x::Float64)

    Ps = P_s(Tk, rho_d, q_v)
    Pxi = P_xi(Tk, rho_d, q_v)
    Pqv = P_qv(Tk, rho_d, q_v)

    return (Ps * s_x) + (Pxi * xi_x) + (Pqv * qv_x)
end

"""
    thermodynamic_tuple(s, xi, mu)

Recover the full thermodynamic state `(q_v, rho_d, Tk, p)` from the prognostic
variables (`s`, `xi`, `mu`).
"""
function thermodynamic_tuple(s::Float64, xi::Float64, mu::Float64)

    q_v = inv_mu_transform(mu)
    rho_d = dry_density(xi)
    Tk = temperature(s, rho_d, q_v)
    pd = 0.01 * Rd * Tk * rho_d
    e = 0.01 * Rv * Tk * rho_d * q_v
    p = pd + e
    return (q_v, rho_d, Tk, p)
end

"""
    thermodynamic_tuple_rhod(s, rho_d, mu)

Like [`thermodynamic_tuple`](@ref) but for the linear-`rho_d` equation set: takes the
dry-air density `rho_d` directly instead of the log-density `xi`, skipping the
`dry_density` conversion. Returns `(q_v, rho_d, Tk, p)`.
"""
function thermodynamic_tuple_rhod(s::Float64, rho_d::Float64, mu::Float64)

    q_v = inv_mu_transform(mu)
    Tk = temperature(s, rho_d, q_v)
    pd = 0.01 * Rd * Tk * rho_d
    e = 0.01 * Rv * Tk * rho_d * q_v
    p = pd + e
    return (q_v, rho_d, Tk, p)
end

"""
    potential_temperature(s, xi, mu)

Dry potential temperature [K] from the prognostic variables (`s`, `xi`, `mu`). Thin
adapter converting the control variables to physical state and delegating to
`Springsteel.Thermodynamics.potential_temperature`.
"""
function potential_temperature(s::Float64, xi::Float64, mu::Float64)

    q_v = inv_mu_transform(mu)
    rho_d = dry_density(xi)
    return Springsteel.Thermodynamics.potential_temperature(s, rho_d, q_v)
end

"""
    reversible_theta_e(s, xi, mu, mu_l=0.0)

Reversible equivalent potential temperature [K] from the prognostic variables. Thin
adapter delegating to `Springsteel.Thermodynamics.reversible_theta_e`.
"""
function reversible_theta_e(s::Float64, xi::Float64, mu::Float64, mu_l::Float64 = 0.0)

    q_v = inv_mu_transform(mu)
    rho_d = dry_density(xi)
    q_l = inv_mu_transform(mu_l)
    return Springsteel.Thermodynamics.reversible_theta_e(s, rho_d, q_v, q_l)
end

"""
    theta_rho(s, xi, mu, mu_l=0.0)

Density potential temperature [K] from the prognostic variables. Thin adapter
delegating to `Springsteel.Thermodynamics.theta_rho`.
"""
function theta_rho(s::Float64, xi::Float64, mu::Float64, mu_l::Float64 = 0.0)

    q_v = inv_mu_transform(mu)
    rho_d = dry_density(xi)
    q_l = inv_mu_transform(mu_l)
    return Springsteel.Thermodynamics.theta_rho(s, rho_d, q_v, q_l)
end

"""
    Rayleigh_damping(alpha, z, z_d, z_t)

Rayleigh damping coefficient for an upper-boundary sponge layer. Zero below the
damping onset height `z_d`; a half-cosine profile increasing to `alpha/2` at the
model top `z_t`.

# References
- Durran, D. R. and J. B. Klemp (1983). *Mon. Wea. Rev.*, 111, 2341–2361.
"""
function Rayleigh_damping(alpha::Float64, z::Float64, z_d::Float64, z_t::Float64)

    if (z <= z_d)
        return 0.0
    end

    norm_z = (z - z_d)/(z_t - z_d)
    tau = -0.5 * alpha * (1.0 - cos(norm_z * pi))
    return tau
end

"""
    thermal_conductivity(Tk)

Thermal conductivity of air [W/(m·K)] as a linear function of temperature.

# References
- Pruppacher, H. R. and J. D. Klett (1997). *Microphysics of Clouds and
  Precipitation*. 2nd ed., Kluwer Academic Publishers.
"""
function thermal_conductivity(Tk::Float64)

    Tc = Tk - 273.15
    k = (5.69 + 0.017 * Tc) * 4.184e-3
end
