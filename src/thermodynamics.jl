"""
Thermodynamic constants following Emanuel (1994) "Atmospheric Convection".

| Constant | Value | Units | Description |
|:---------|:------|:------|:------------|
| `Rd`     | 287.04 | J/(kg·K) | Gas constant for dry air |
| `Rv`     | 461.50 | J/(kg·K) | Gas constant for water vapor |
| `Eps`    | Rd/Rv  | dimensionless | Ratio of gas constants |
| `Cvd`    | 716.96 | J/(kg·K) | Specific heat of dry air at constant volume |
| `Cvv`    | 1410.0 | J/(kg·K) | Specific heat of water vapor at constant volume |
| `Cpd`    | Cvd+Rd | J/(kg·K) | Specific heat of dry air at constant pressure |
| `Cpv`    | Cvv+Rv | J/(kg·K) | Specific heat of water vapor at constant pressure |
| `Cl`     | 4186.0 | J/(kg·K) | Specific heat of liquid water |
| `Ci`     | 2106.0 | J/(kg·K) | Specific heat of ice |
| `gravity`| 9.81   | m/s² | Gravitational acceleration |
| `L_v0`   | 2.501e6 | J/kg | Latent heat of vaporization at T₀ |
| `rho_l`  | 1000.0 | kg/m³ | Density of liquid water |
| `rho_i`  | 917.0  | kg/m³ | Density of ice |
| `T_0`    | 273.16 | K | Reference temperature (triple point of water) |
| `p_0`    | 1000.0 | hPa | Reference pressure |
| `q0`     | 1.0e-5 | kg/kg | Small mixing ratio threshold |
| `rho_d0` | 100·p₀/(T₀·Rd) | kg/m³ | Reference dry air density |
| `rho_v0` | 100·eₛ(T₀)/(T₀·Rv) | kg/m³ | Reference vapor density at T₀ |

# References
- Emanuel, K. A. (1994). *Atmospheric Convection*. Oxford University Press.
"""
# Constants from Emanuel (1994)
const Rd = 287.04
const Rv = 461.50
const Eps = Rd / Rv
const Cvd = 716.96
const Cvv = 1410.0
const Cpd = Cvd + Rd
const Cpv = Cvv + Rv
const Cl = 4186.0
const Ci = 2106.0 # Ice heat capacity
const gravity = 9.81
const L_v0 = 2.501e6
const rho_l = 1000.0 # Density of liquid water in kg/m^3
const rho_i = 917.0 # Density of ice in kg/m^3

# Entropy function constants
const T_0 = 273.16
const p_0 = 1000.0
const q0 = 1.0e-5

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
    sat_pressure_liquid(Tk)

Compute saturation vapor pressure over liquid water using the Bolton (1980) formula.

# Arguments
- `Tk::Float64`: temperature [K]

# Returns
- Saturation vapor pressure over liquid water [hPa]

# Examples
```jldoctest
julia> Scythe.sat_pressure_liquid(273.16)
6.116436706236274

julia> Scythe.sat_pressure_liquid(300.0)
35.34519666889136
```

# References
- Bolton, D. (1980). *Mon. Wea. Rev.*, 108, 1046–1053.
"""
function sat_pressure_liquid(Tk::Float64)

    Tc = Tk - 273.15
    return 6.112 * exp(17.67 * Tc / (Tc + 243.5))
end

"""
    sat_pressure_ice(Tk)

Compute saturation vapor pressure over ice.

# Arguments
- `Tk::Float64`: temperature [K]

# Returns
- Saturation vapor pressure over ice [hPa]
"""
function sat_pressure_ice(Tk::Float64)

    Tc = Tk - 273.15
    return 6.112 * exp(21.8745584 * Tc / (Tc + 265.49))
end

const rho_d0 = 100.0 * p_0 / (T_0 * Rd)
const rho_v0 = 100.0 * sat_pressure_liquid(T_0) / (T_0 * Rv)

"""
    dewpoint(p, q_v)

Compute the dewpoint temperature from pressure and water vapor mixing ratio
by inverting the Bolton (1980) saturation vapor pressure formula.

# Arguments
- `p::Float64`: total pressure [hPa]
- `q_v::Float64`: water vapor mixing ratio [kg/kg]

# Returns
- Dewpoint temperature [K]
"""
function dewpoint(p::Float64, q_v::Float64)

    e = vapor_pressure(p, q_v)
    Tc = 243.5 * log(e/6.112) / (17.67 - log(e/6.112))
    return Tc + 273.15
end

"""
    L_v(Tk)

Compute the latent heat of vaporization as a linear function of temperature,
accounting for the difference in heat capacities between vapor and liquid.

# Arguments
- `Tk::Float64`: temperature [K]

# Returns
- Latent heat of vaporization [J/kg]

# Examples
```jldoctest
julia> Scythe.L_v(273.16)
2.501e6

julia> Scythe.L_v(300.0)
2.43887882e6
```

# References
- Emanuel, K. A. (1994). *Atmospheric Convection*. Oxford University Press.
"""
function L_v(Tk::Float64)

    return L_v0 + ((Cpv - Cl) * (Tk - T_0))
end

"""
    entropy(Tk, rho_d, q_v)

Compute the moist entropy per unit mass of dry air from temperature, dry air density,
and water vapor mixing ratio. Uses the entropy formulation of Emanuel (1994).

# Arguments
- `Tk::Float64`: temperature [K]
- `rho_d::Float64`: dry air density [kg/m³]
- `q_v::Float64`: water vapor mixing ratio [kg/kg]

# Returns
- Specific moist entropy [J/(kg·K)]

# Examples
```jldoctest
julia> Scythe.entropy(300.0, 1.0, 0.01)
226.56021716081108
```

# References
- Emanuel, K. A. (1994). *Atmospheric Convection*. Oxford University Press.
"""
function entropy(Tk::Float64, rho_d::Float64, q_v::Float64)

    qfactor = 0.0
    if (q_v != 0.0)
        qfactor = q_v * (Rv * log(q_v * rho_d / rho_v0) - (L_v(T_0)/T_0))
    end

    Cfactor = Cvd + (q_v * Cvv)
    s = (Cfactor * log(Tk/T_0)) - (Rd * log(rho_d/rho_d0)) - qfactor
    return s
end

"""
    vapor_entropy(Tk, rho_d, q_v)

Compute the water vapor contribution to the specific entropy. Returns zero
when the mixing ratio is non-positive.

# Arguments
- `Tk::Float64`: temperature [K]
- `rho_d::Float64`: dry air density [kg/m³]
- `q_v::Float64`: water vapor mixing ratio [kg/kg]

# Returns
- Vapor entropy contribution [J/(kg·K)]

# References
- Emanuel, K. A. (1994). *Atmospheric Convection*. Oxford University Press.
"""
function vapor_entropy(Tk::Float64, rho_d::Float64, q_v::Float64)

    if q_v > 0.0
        return (Cvv * log(Tk/T_0)) - (Rv * log(q_v * rho_d / rho_v0)) + (L_v(T_0)/T_0)
    else
        return 0.0
    end
end

"""
    temperature(s, rho_d, q_v)

Recover temperature from moist entropy, dry air density, and water vapor mixing ratio
by inverting the entropy relation of Emanuel (1994).

# Arguments
- `s::Float64`: specific moist entropy [J/(kg·K)]
- `rho_d::Float64`: dry air density [kg/m³]
- `q_v::Float64`: water vapor mixing ratio [kg/kg]

# Returns
- Temperature [K]

# Examples
```jldoctest
julia> s = Scythe.entropy(300.0, 1.0, 0.01);

julia> Scythe.temperature(s, 1.0, 0.01) ≈ 300.0
true
```

# References
- Emanuel, K. A. (1994). *Atmospheric Convection*. Oxford University Press.
"""
function temperature(s::Float64, rho_d::Float64, q_v::Float64)

    Cfactor = Cvd + (q_v * Cvv)
    qfactor = 1.0
    if (q_v != 0.0)
        qfactor = (rho_d * q_v / rho_v0)^((q_v * Rv) / Cfactor)
    end

    rhofactor = (rho_d / rho_d0)^(Rd / Cfactor)
    Tfactor = exp((s - (q_v * L_v(T_0)/T_0)) / Cfactor)

    T = T_0 * Tfactor * rhofactor * qfactor
    return T
end

"""
    pressure(s, rho_d, q_v)

Compute total pressure (dry air + vapor) from moist entropy, dry air density,
and water vapor mixing ratio.

# Arguments
- `s::Float64`: specific moist entropy [J/(kg·K)]
- `rho_d::Float64`: dry air density [kg/m³]
- `q_v::Float64`: water vapor mixing ratio [kg/kg]

# Returns
- Total pressure [hPa]

# Examples
```jldoctest
julia> s = Scythe.entropy(300.0, 1.0, 0.01);

julia> Scythe.pressure(s, 1.0, 0.01)
874.9650000000003
```
"""
function pressure(s::Float64, rho_d::Float64, q_v::Float64)

    Tk = temperature(s, rho_d, q_v)
    pd = 0.01 * Rd * Tk * rho_d
    e = 0.01 * Rv * Tk * rho_d * q_v
    return pd + e
end

"""
    vapor_pressure(p, q_v)

Compute the partial pressure of water vapor from total pressure and mixing ratio.

# Arguments
- `p::Float64`: total pressure [hPa]
- `q_v::Float64`: water vapor mixing ratio [kg/kg]

# Returns
- Vapor pressure [hPa]
"""
function vapor_pressure(p::Float64, q_v::Float64)

    # Input is total pressure in hPa, and mixing ratio in kg/kg
    # Output is vapor pressure in hPa
    e = (p * q_v)/(Eps + q_v)
end

"""
    mixing_ratio(p, e)

Compute the water vapor mixing ratio from total pressure and vapor pressure.

# Arguments
- `p::Float64`: total pressure [hPa]
- `e::Float64`: vapor pressure [hPa]

# Returns
- Water vapor mixing ratio [kg/kg]
"""
function mixing_ratio(p::Float64, e::Float64)

    q_v = (Eps * e)/(p-e)
end

"""
    sat_pressure_liquid_buck(Tk, phPa)

Compute saturation vapor pressure over liquid water using the Buck (1981) formula,
including the enhancement factor for the effect of dry air pressure.

# Arguments
- `Tk::Float64`: temperature [K]
- `phPa::Float64`: total pressure [hPa]

# Returns
- Saturation vapor pressure over liquid water [hPa]

# References
- Buck, A. L. (1981). New equations for computing vapor pressure and enhancement factor.
  *J. Appl. Meteor.*, 20, 1527–1532.
"""
function sat_pressure_liquid_buck(Tk::Float64, phPa::Float64)

    # Formula from Buck JAM (1981)
    # Includes dry air pressure enhancement effect
    # T in K, p in hPa
    Tc = Tk - 273.15
    A = 7.2e-4
    B = 3.20e-6
    C = 5.9e-10
    fw4 = 1.0 + A + (phPa * (B + (C * Tc^2)))
    
    a = 6.1121
    b = 18.729
    c = 257.87
    d = 227.3
    ew4 = a * exp( (b - (Tc / d)) * Tc / (Tc + c) )

    return fw4 * ew4
end

"""
    sat_pressure_liquid_buck_dT(Tk, phPa)

Compute the derivative of the Buck (1981) saturation vapor pressure over liquid
with respect to temperature at constant pressure.

# Arguments
- `Tk::Float64`: temperature [K]
- `phPa::Float64`: total pressure [hPa]

# Returns
- ∂eₛ/∂T at constant pressure [hPa/K]

# References
- Buck, A. L. (1981). New equations for computing vapor pressure and enhancement factor.
  *J. Appl. Meteor.*, 20, 1527–1532.
"""
function sat_pressure_liquid_buck_dT(Tk::Float64, phPa::Float64)

    # T in K, p in hPa
    # Formula from Buck JAM (1981) derivative with respect to T at constant p
    Tc = Tk - 273.15

    A = 7.2e-4
    B = 3.20e-6
    C = 5.9e-10
    fw4 = 1.0 + A + (phPa * (B + (C * Tc^2)))
    d_fw4 = 2.0 * phPa * C * Tc

    a = 6.1121
    b = 18.729
    c = 257.87
    d = 227.3
    ew4 = a * exp( (b - (Tc / d)) * Tc / (Tc + c) )
    T1 = (d * b - (2.0 * Tc)) * (d * (Tc + c)) - d* ((d * b * Tc) - Tc^2)
    T2 =  (d * (Tc + c))^2
    d_ew4 = ew4 * T1 / T2

    return ew4 * d_fw4 + fw4 * d_ew4
end

"""
    sat_pressure_ice_buck(Tk, phPa)

Compute saturation vapor pressure over ice using the Buck (1981) formula,
including the enhancement factor for the effect of dry air pressure.

# Arguments
- `Tk::Float64`: temperature [K]
- `phPa::Float64`: total pressure [hPa]

# Returns
- Saturation vapor pressure over ice [hPa]

# References
- Buck, A. L. (1981). New equations for computing vapor pressure and enhancement factor.
  *J. Appl. Meteor.*, 20, 1527–1532.
"""
function sat_pressure_ice_buck(Tk::Float64, phPa::Float64)

    # Formula from Buck JAM (1981)
    # Includes dry air pressure enhancement effect
    # T in K, p in hPa
    Tc = Tk - 273.15
    A = 2.2e-4
    B = 3.83e-6
    C = 6.4e-10
    fi4 = 1.0 + A + (phPa * (B + (C * Tc^2)))
    
    a = 6.1115
    b = 23.036
    c = 279.82
    d = 333.7
    ei3 = a * exp( (b - (Tc / d)) * Tc / (Tc + c) )

    return fi4 * ei3
end

"""
    q_sat_liquid(Tk, phPa)

Compute the saturation mixing ratio over liquid water using the Buck (1981)
saturation vapor pressure formula.

# Arguments
- `Tk::Float64`: temperature [K]
- `phPa::Float64`: total pressure [hPa]

# Returns
- Saturation mixing ratio over liquid [kg/kg]
"""
function q_sat_liquid(Tk::Float64, phPa::Float64)

    # Saturation mixing ratio over liquid
    # T in K, p in hPa
    ew = sat_pressure_liquid_buck(Tk,phPa)
    q_sat = Eps * ew / (phPa - ew)
    return q_sat
end

"""
    q_sat_ice(Tk, phPa)

Compute the saturation mixing ratio over ice using the Buck (1981)
saturation vapor pressure formula.

# Arguments
- `Tk::Float64`: temperature [K]
- `phPa::Float64`: total pressure [hPa]

# Returns
- Saturation mixing ratio over ice [kg/kg]
"""
function q_sat_ice(Tk::Float64, phPa::Float64)

    # Saturation mixing ratio over ice
    # T in K, p in hPa
    ei = sat_pressure_ice_buck(Tk,phPa)
    q_sat = Eps * ei / (phPa - ei)
    return q_sat
end

"""
    bhyp(q_v)

Compute a bijective hyperbolic transform of water vapor mixing ratio, mapping
non-negative `q_v` to a transformed variable `mu` suitable for use as a prognostic
variable. Inverse of [`ahyp`](@ref).

# Arguments
- `q_v::Float64`: water vapor mixing ratio [kg/kg]

# Returns
- Transformed mixing ratio `mu` [kg/kg]
"""
function bhyp(q_v::Float64)

    mu = 0.5 * ( (q_v + q0) - (q0*q0/(q_v + q0)) )
    return mu
end

"""
    ahyp(mu)

Compute the inverse hyperbolic transform to recover water vapor mixing ratio
from the transformed variable `mu`. Inverse of [`bhyp`](@ref). Returns zero
for negative `mu`.

# Arguments
- `mu::Float64`: transformed mixing ratio [kg/kg]

# Returns
- Water vapor mixing ratio [kg/kg]
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
    dry_density(xi)

Recover dry air density from the log-density prognostic variable `xi`.
Inverse of [`log_dry_density`](@ref).

# Arguments
- `xi::Float64`: log-density variable `ln(rho_d / rho_d0)` [dimensionless]

# Returns
- Dry air density [kg/m³]
"""
function dry_density(xi::Float64)

    return rho_d0 * exp(xi)
end

"""
    log_dry_density(rho_d)

Compute the log-density prognostic variable `xi = ln(rho_d / rho_d0)` from
dry air density. Inverse of [`dry_density`](@ref).

# Arguments
- `rho_d::Float64`: dry air density [kg/m³]

# Returns
- Log-density variable [dimensionless]
"""
function log_dry_density(rho_d::Float64)

    return log(rho_d/rho_d0)
end

"""
    inv_xi_transform(xi)

Apply the inverse xi transform (natural logarithm). Inverse of [`xi_transform`](@ref).

# Arguments
- `xi::Float64`: transformed variable [dimensionless]

# Returns
- Inverse-transformed value [dimensionless]
"""
function inv_xi_transform(xi::Float64)

    return log(xi)

end

"""
    xi_transform(q)

Apply the xi transform (exponential). Inverse of [`inv_xi_transform`](@ref).

# Arguments
- `q::Float64`: input variable [dimensionless]

# Returns
- Transformed value [dimensionless]
"""
function xi_transform(q::Float64)

    return exp(q)

end

"""
    dxidq(q)

Compute the derivative of the xi transform with respect to `q`, i.e. `dξ/dq = exp(q)`.

# Arguments
- `q::Float64`: input variable [dimensionless]

# Returns
- Derivative `dξ/dq` [dimensionless]
"""
function dxidq(q::Float64)

    # Derivative of xi with respect to Q
    return exp(q) #
    #return 1.0 / sqrt(q^2 + (1.0e-5)^2)

end

"""
    inv_mu_transform(mu)

Recover the water vapor mixing ratio from the scaled prognostic variable `mu`.
Returns zero for negative `mu`. Inverse of [`mu_transform`](@ref).

Currently uses a simple linear scaling: `q = mu * 10⁻⁵`.

# Arguments
- `mu::Float64`: scaled mixing ratio variable [dimensionless]

# Returns
- Water vapor mixing ratio [kg/kg]

# Examples
```jldoctest
julia> Scythe.inv_mu_transform(1000.0)
0.01
```
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
Inverse of [`inv_mu_transform`](@ref).

Currently uses a simple linear scaling: `mu = q * 10⁵`.

# Arguments
- `q::Float64`: water vapor mixing ratio [kg/kg]

# Returns
- Scaled mixing ratio variable [dimensionless]

# Examples
```jldoctest
julia> Scythe.mu_transform(0.01)
1000.0
```
"""
function mu_transform(q::Float64)

    return _MU_HYPERBOLIC[] ? bhyp(q) : q * 1.0e5
end

"""
    dmudq(mu, q_v)

Compute the derivative of the `mu` transform with respect to mixing ratio `q_v`,
i.e. `dmu/dq_v`. Currently returns the constant `10⁵` (linear scaling).

# Arguments
- `mu::Float64`: scaled mixing ratio variable [dimensionless]
- `q_v::Float64`: water vapor mixing ratio [kg/kg]

# Returns
- Derivative `dmu/dq_v` [dimensionless]
"""
function dmudq(mu::Float64, q_v::Float64)

    return _MU_HYPERBOLIC[] ? ((q_v + q0) - mu) / (q_v + q0) : 1.0e5
end

"""
    P_s(Tk, rho_d, q_v)

Compute the partial derivative of pressure with respect to entropy, `∂p/∂s`,
at constant dry air density and mixing ratio.

# Arguments
- `Tk::Float64`: temperature [K]
- `rho_d::Float64`: dry air density [kg/m³]
- `q_v::Float64`: water vapor mixing ratio [kg/kg]

# Returns
- `∂p/∂s` [hPa·K/J]
"""
function P_s(Tk::Float64, rho_d::Float64, q_v::Float64)

    Cfactor = Cvd + (q_v * Cvv)
    return Tk * ((rho_d * Rd) + (q_v * rho_d * Rv)) / Cfactor
end

"""
    P_xi(Tk, rho_d, q_v)

Compute the partial derivative of pressure with respect to the log-density
variable `xi`, `∂p/∂ξ`, at constant entropy and mixing ratio.

# Arguments
- `Tk::Float64`: temperature [K]
- `rho_d::Float64`: dry air density [kg/m³]
- `q_v::Float64`: water vapor mixing ratio [kg/kg]

# Returns
- `∂p/∂ξ` [hPa]
"""
function P_xi(Tk::Float64, rho_d::Float64, q_v::Float64)

    return (Rd + (q_v * rho_d * Rv)) * ((rho_d * Tk) + P_s(Tk, rho_d, q_v))
end

"""
    P_xi_from_s(s, xi, mu)

Compute `∂p/∂ξ` directly from the prognostic variables (`s`, `xi`, `mu`)
by first recovering the thermodynamic state via [`thermodynamic_tuple`](@ref).

# Arguments
- `s::Float64`: specific moist entropy [J/(kg·K)]
- `xi::Float64`: log-density variable [dimensionless]
- `mu::Float64`: scaled mixing ratio variable [dimensionless]

# Returns
- `∂p/∂ξ` [hPa]
"""
function P_xi_from_s(s::Float64, xi::Float64, mu::Float64)

    q_v, rho_d, Tk, p = thermodynamic_tuple(s, xi, mu)
    return P_xi(Tk, rho_d, q_v)
end

"""
    P_qv(Tk, rho_d, q_v)

Compute the partial derivative of pressure with respect to water vapor
mixing ratio, `∂p/∂q_v`, at constant entropy and dry air density.
Returns zero when `q_v` is zero.

# Arguments
- `Tk::Float64`: temperature [K]
- `rho_d::Float64`: dry air density [kg/m³]
- `q_v::Float64`: water vapor mixing ratio [kg/kg]

# Returns
- `∂p/∂q_v` [hPa/(kg/kg)]
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

Compute the partial derivative of pressure with respect to the scaled mixing
ratio variable `mu`, `∂p/∂mu`, using the chain rule through [`P_qv`](@ref)
and [`dmudq`](@ref).

# Arguments
- `Tk::Float64`: temperature [K]
- `rho_d::Float64`: dry air density [kg/m³]
- `mu::Float64`: scaled mixing ratio variable [dimensionless]

# Returns
- `∂p/∂mu` [hPa]
"""
function P_mu(Tk::Float64, rho_d::Float64, mu::Float64)

    q_v = inv_mu_transform(mu)
    return P_qv(Tk, rho_d, q_v) / dmudq(mu, q_v)
end

"""
    pressure_gradient(Tk, rho_d, q_v, s_x, xi_x, qv_x)

Compute the pressure gradient in a given spatial direction using the chain rule
expansion `∂p/∂x = (∂p/∂s)(∂s/∂x) + (∂p/∂ξ)(∂ξ/∂x) + (∂p/∂q_v)(∂q_v/∂x)`.

# Arguments
- `Tk::Float64`: temperature [K]
- `rho_d::Float64`: dry air density [kg/m³]
- `q_v::Float64`: water vapor mixing ratio [kg/kg]
- `s_x::Float64`: spatial gradient of entropy [J/(kg·K·m)]
- `xi_x::Float64`: spatial gradient of log-density [1/m]
- `qv_x::Float64`: spatial gradient of mixing ratio [kg/(kg·m)]

# Returns
- Pressure gradient in the given direction [hPa/m]
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

Recover the full thermodynamic state from the prognostic variables
(`s`, `xi`, `mu`). Returns a named-position tuple of physical quantities.

# Arguments
- `s::Float64`: specific moist entropy [J/(kg·K)]
- `xi::Float64`: log-density variable [dimensionless]
- `mu::Float64`: scaled mixing ratio variable [dimensionless]

# Returns
- `(q_v, rho_d, Tk, p)`: mixing ratio [kg/kg], dry density [kg/m³],
  temperature [K], total pressure [hPa]
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

# Arguments
- `s::Float64`: specific moist entropy [J/(kg·K)]
- `rho_d::Float64`: dry air density [kg/m³]
- `mu::Float64`: scaled mixing ratio variable [dimensionless]
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

Compute the dry potential temperature from the prognostic variables
(`s`, `xi`, `mu`).

# Arguments
- `s::Float64`: specific moist entropy [J/(kg·K)]
- `xi::Float64`: log-density variable [dimensionless]
- `mu::Float64`: scaled mixing ratio variable [dimensionless]

# Returns
- Potential temperature [K]
"""
function potential_temperature(s::Float64, xi::Float64, mu::Float64)

    q_v, rho_d, Tk, p = thermodynamic_tuple(s, xi, mu)
    theta = Tk * (p_0 / p)^(Rd/Cpd)
end

"""
    reversible_theta_e(s, xi, mu, mu_l=0.0)

Compute the reversible equivalent potential temperature, which accounts for
both vapor and liquid water content.

# Arguments
- `s::Float64`: specific moist entropy [J/(kg·K)]
- `xi::Float64`: log-density variable [dimensionless]
- `mu::Float64`: scaled vapor mixing ratio variable [dimensionless]
- `mu_l::Float64`: scaled liquid mixing ratio variable [dimensionless] (default `0.0`)

# Returns
- Reversible equivalent potential temperature [K]

# References
- Emanuel, K. A. (1994). *Atmospheric Convection*. Oxford University Press.
"""
function reversible_theta_e(s::Float64, xi::Float64, mu::Float64, mu_l::Float64 = 0.0)

    q_v, rho_d, Tk, p = thermodynamic_tuple(s, xi, mu)
    q_l = inv_mu_transform(mu_l)
    q_t = q_v + q_l
    e = vapor_pressure(p, q_v)
    es = sat_pressure_liquid_buck(Tk, p)
    theta_term = Tk * (p_0 / (p-e))^(Rd/(Cpd + (Cl * q_t)))
    H_term = (e/es)^((-Rv * q_v)/(Cpd + (Cl * q_t)))
    exp_term = exp(L_v(Tk) * q_v / ((Cpd + (Cl * q_t)) * Tk))
    return theta_term * H_term * exp_term
end

"""
    theta_rho(s, xi, mu, mu_l=0.0)

Compute the density potential temperature, which accounts for the effect of
water vapor and liquid water on air density (virtual temperature effect).

# Arguments
- `s::Float64`: specific moist entropy [J/(kg·K)]
- `xi::Float64`: log-density variable [dimensionless]
- `mu::Float64`: scaled vapor mixing ratio variable [dimensionless]
- `mu_l::Float64`: scaled liquid mixing ratio variable [dimensionless] (default `0.0`)

# Returns
- Density potential temperature [K]
"""
function theta_rho(s::Float64, xi::Float64, mu::Float64, mu_l::Float64 = 0.0)

    q_v, rho_d, Tk, p = thermodynamic_tuple(s, xi, mu)
    q_l = inv_mu_transform(mu_l)
    q_t = q_v + q_l
    theta = potential_temperature(s, xi, mu)
    return theta * (1.0 + (q_v / Eps)) / (1.0 + q_t)
end

"""
    Rayleigh_damping(alpha, z, z_d, z_t)

Compute the Rayleigh damping coefficient for a sponge layer at the upper boundary.
Returns zero below the damping onset height `z_d`, and a half-cosine profile
increasing to `alpha/2` at the model top `z_t`.

# Arguments
- `alpha::Float64`: maximum damping rate [1/s]
- `z::Float64`: height [m]
- `z_d::Float64`: height at which damping begins [m]
- `z_t::Float64`: height of the model top [m]

# Returns
- Damping coefficient (negative, suitable for tendency multiplication) [1/s]

# References
- Durran, D. R. and J. B. Klemp (1983). A compressible model for the simulation
  of moist mountain waves. *Mon. Wea. Rev.*, 111, 2341–2361.
"""
function Rayleigh_damping(alpha::Float64, z::Float64, z_d::Float64, z_t::Float64)

    # From Durran and Klemp (1983)
    if (z <= z_d)
        return 0.0
    end

    norm_z = (z - z_d)/(z_t - z_d)
    tau = -0.5 * alpha * (1.0 - cos(norm_z * pi))
    return tau
end

"""
    thermal_conductivity(Tk)

Compute the thermal conductivity of air as a linear function of temperature,
following the empirical fit from Pruppacher and Klett (1997), p. 418.

# Arguments
- `Tk::Float64`: temperature [K]

# Returns
- Thermal conductivity [W/(m·K)]

# References
- Pruppacher, H. R. and J. D. Klett (1997). *Microphysics of Clouds and
  Precipitation*. 2nd ed., Kluwer Academic Publishers.
"""
function thermal_conductivity(Tk::Float64)

    # From Pruppacher and Klett p. 418
    # T in K, k in W/(m*K)
    Tc = Tk - 273.15
    k = (5.69 + 0.017 * Tc) * 4.184e-3
end
