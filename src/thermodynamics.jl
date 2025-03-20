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

# Entropy function constants
const T_0 = 273.16
const p_0 = 1000.0
const q0 = 1.0e-7

function sat_pressure_liquid(Tk::Float64)

    Tc = Tk - 273.15
    return 6.112 * exp(17.67 * Tc / (Tc + 243.5))
end

function sat_pressure_ice(Tk::Float64)

    Tc = Tk - 273.15
    return 6.112 * exp(21.8745584 * Tc / (Tc + 265.49))
end

const rho_d0 = 100.0 * p_0 / (T_0 * Rd)
const rho_v0 = 100.0 * sat_pressure_liquid(T_0) / (T_0 * Rv)

function dewpoint(p::Float64, q_v::Float64)

    e = vapor_pressure(p, q_v)
    Tc = 243.5 * log(e/6.112) / (17.67 - log(e/6.112))
    return Tc + 273.15
end

function L_v(Tk::Float64)

    return L_v0 + ((Cpv - Cl) * (Tk - T_0))
end

function entropy(Tk::Float64, rho_d::Float64, q_v::Float64)

    qfactor = 0.0
    if (q_v != 0.0)
        qfactor = q_v * (Rv * log(q_v * rho_d / rho_v0) - (L_v(T_0)/T_0))
    end

    Cfactor = Cvd + (q_v * Cvv)
    s = (Cfactor * log(Tk/T_0)) - (Rd * log(rho_d/rho_d0)) - qfactor
    return s
end

function vapor_entropy(Tk::Float64, rho_d::Float64, q_v::Float64)

    if q_v > 0.0
        return (Cvv * log(Tk/T_0)) - (Rv * log(q_v * rho_d / rho_v0)) + (L_v(T_0)/T_0)
    else
        return 0.0
    end
end

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

function pressure(s::Float64, rho_d::Float64, q_v::Float64)

    Tk = temperature(s, rho_d, q_v)
    pd = 0.01 * Rd * Tk * rho_d
    e = 0.01 * Rv * Tk * rho_d * q_v
    return pd + e
end

function vapor_pressure(p::Float64, q_v::Float64)

    # Input is total pressure in hPa, and mixing ratio in kg/kg
    # Output is vapor pressure in hPa
    e = (p * q_v)/(Eps + q_v)
end

function mixing_ratio(p::Float64, e::Float64)

    q_v = (Eps * e)/(p-e)
end

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

function q_sat_liquid(Tk::Float64, phPa::Float64)

    # Saturation mixing ratio over liquid
    # T in K, p in hPa
    ew = sat_pressure_liquid_buck(Tk,phPa)
    q_sat = Eps * ew / (phPa - ew)
    return q_sat
end

function q_sat_ice(Tk::Float64, phPa::Float64)

    # Saturation mixing ratio over ice
    # T in K, p in hPa
    ei = sat_pressure_ice_buck(Tk,phPa)
    q_sat = Eps * ei / (phPa - ei)
    return q_sat
end

function bhyp(q_v::Float64)

    mu = 0.5 * ( (q_v + q0) - (q0*q0/(q_v + q0)) )
    return mu
end

function ahyp(mu::Float64)

    if (mu < 0.0)
        return 0.0
    else
        q_v = sqrt(mu*mu + q0*q0) + mu - q0
        return q_v
    end
end

function dmudq(mu::Float64, q_v::Float64)

    return ((q_v + q0) - mu)/(q_v + q0)
end

function dry_density(xi::Float64)
    
    return rho_d0 * exp(xi)
end

function log_dry_density(rho_d::Float64)
    
    return log(rho_d/rho_d0)
end

function P_s(Tk::Float64, rho_d::Float64, q_v::Float64)

    Cfactor = Cvd + (q_v * Cvv)
    return Tk * ((rho_d * Rd) + (q_v * rho_d * Rv)) / Cfactor
end

function P_xi(Tk::Float64, rho_d::Float64, q_v::Float64)

    return (Rd + (q_v * rho_d * Rv)) * ((rho_d * Tk) + P_s(Tk, rho_d, q_v))
end

function P_xi_from_s(s::Float64, xi::Float64, mu::Float64)

    q_v, rho_d, Tk, p = thermodynamic_tuple(s, xi, mu)
    return P_xi(Tk, rho_d, q_v)
end

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

function P_mu(Tk::Float64, rho_d::Float64, mu::Float64)

    q_v = ahyp(mu)
    return P_qv(Tk, rho_d, q_v) / dmudq(mu, q_v)
end

function pressure_gradient(Tk::Float64, rho_d::Float64, q_v::Float64, 
        s_x::Float64, xi_x::Float64, qv_x::Float64)
    
    Ps = P_s(Tk, rho_d, q_v)
    Pxi = P_xi(Tk, rho_d, q_v)
    Pqv = P_qv(Tk, rho_d, q_v)
    
    return (Ps * s_x) + (Pxi * xi_x) + (Pqv * qv_x)
end

function thermodynamic_tuple(s::Float64, xi::Float64, mu::Float64)

    q_v = ahyp(mu)
    rho_d = dry_density(xi)
    Tk = temperature(s, rho_d, q_v)
    pd = 0.01 * Rd * Tk * rho_d
    e = 0.01 * Rv * Tk * rho_d * q_v
    p = pd + e
    return (q_v, rho_d, Tk, p)
end

function potential_temperature(s::Float64, xi::Float64, mu::Float64)
    
    q_v, rho_d, Tk, p = thermodynamic_tuple(s, xi, mu)
    theta = Tk * (p_0 / p)^(Rd/Cpd)
end

function reversible_theta_e(s::Float64, xi::Float64, mu::Float64, mu_l::Float64 = 0.0)
    
    q_v, rho_d, Tk, p = thermodynamic_tuple(s, xi, mu)
    q_l = ahyp(mu_l)
    q_t = q_v + q_l
    e = vapor_pressure(p, q_v)
    es = sat_pressure_liquid_buck(Tk, p)
    theta_term = Tk * (p_0 / (p-e))^(Rd/(Cpd + (Cl * q_t)))
    H_term = (e/es)^((-Rv * q_v)/(Cpd + (Cl * q_t)))
    exp_term = exp(L_v(Tk) * q_v / ((Cpd + (Cl * q_t)) * Tk))
    return theta_term * H_term * exp_term
end

function theta_rho(s::Float64, xi::Float64, mu::Float64, mu_l::Float64 = 0.0)
    
    q_v, rho_d, Tk, p = thermodynamic_tuple(s, xi, mu)
    q_l = ahyp(mu_l)
    q_t = q_v + q_l
    theta = potential_temperature(s, xi, mu)
    return theta * (1.0 + (q_v / Eps)) / (1.0 + q_t)
end
