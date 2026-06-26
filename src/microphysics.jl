"""
    saturation_adjustment(s, xi, mu, mu_l, tol)

Iteratively adjust the vapor and liquid water mixing ratios to achieve thermodynamic
equilibrium at saturation using a Newton-Raphson method at constant pressure.

# Arguments
- `s`: Entropy variable
- `xi`: Mass variable (related to dry air density)
- `mu`: Total water variable (transformed mixing ratio)
- `mu_l`: Liquid water variable (transformed mixing ratio)
- `tol`: Convergence tolerance for the supersaturation residual [kg/kg]

# Returns
- `(dq, dT)`: Tuple of the change in vapor mixing ratio [kg/kg] and temperature adjustment [K]
"""
function saturation_adjustment(s, xi, mu, mu_l, tol)

    incr = 1.0e-6
    local q_v, rho_d, Tk, p = thermodynamic_tuple(s, xi, mu)

    # Check to see if evaporation or condensation are possible
    if q_v == 0.0
        # No water in this simulation
        return (0.0, 0.0)
    end

    local q_l = inv_mu_transform(mu_l)               # Liquid mixing ratio
    local q_sat = q_sat_liquid(Tk, p)
    iterations = 1
    e_s = sat_pressure_liquid_buck(Tk, p)
    dqsdT = sat_pressure_liquid_buck_dT(Tk,p) * Eps * p / (p - e_s)^2
    dq = (q_sat - q_v)/(1.0 + (L_v(Tk) * dqsdT /(Cpd + ((q_v) * Cpv) + ((q_l) * Cl))))
    SS = q_v - q_sat

    # If dq < tol (default eps) then it is numerically saturated
    if abs(SS) < tol
        return (0.0, 0.0)
    end

    # Initial guess for dT based on constant pressure
    dT = -dq * L_v(Tk) / (Cpd + ((q_v) * Cpv) + ((q_l) * Cl))
    while abs(SS) > tol && iterations < 10

        dq_up = dq + incr
        dT = -dq_up * L_v(Tk) / (Cpd + ((q_v) * Cpv) + ((q_l) * Cl))
        new_q_v = q_v + dq_up
        new_q_sat = q_sat_liquid(Tk + dT, p)
        SS_up = new_q_v - new_q_sat

        dT = -dq * L_v(Tk) / (Cpd + ((q_v) * Cpv) + ((q_l) * Cl))
        new_q_v = q_v + dq
        new_q_sat = q_sat_liquid(Tk + dT, p)
        SS = SS_down = new_q_v - new_q_sat
        dSSdq = (SS_up - SS_down) / incr
        if abs(dSSdq) > 0
            dq = dq - (SS/dSSdq)
        else
            break
        end
        #println("$iterations: $new_q_v, $new_q_sat, $dT, $SS = $dq")
        iterations += 1
    end

    # Adjust to ensure no negative water
    q_test = q_v + dq
    if q_test < 0.0
        dq = -q_v
        dT = -dq * L_v(Tk) / (Cpd + ((q_v) * Cpv) + ((q_l) * Cl))
        #println("No water left to condense")
    end
    q_test = q_l - dq
    if q_test < 0.0
        dq = q_l
        dT = -dq * L_v(Tk) / (Cpd + ((q_v) * Cpv) + ((q_l) * Cl))
        #println("Evaporated all the water")
    end

    return (dq, dT)
end

"""
    linear_saturation_adjustment(qss, Tk, p, q_v, q_l)

Compute a linearized saturation adjustment for the vapor mixing ratio, accounting for
the dependence of saturation on temperature via the `Q_s` factor.

# Arguments
- `qss`: Supersaturation mixing ratio (q_v - q_sat) [kg/kg]
- `Tk`: Temperature [K]
- `p`: Pressure [hPa]
- `q_v`: Water vapor mixing ratio [kg/kg]
- `q_l`: Liquid water mixing ratio [kg/kg]

# Returns
- `dq`: Change in vapor mixing ratio due to condensation or evaporation [kg/kg]
"""
function linear_saturation_adjustment(qss, Tk, p, q_v, q_l)

    # Check to see if evaporation or condensation are possible
    if q_v == 0.0
        # No water in this simulation
        return 0.0
    end

    q_sat = q_sat_liquid(Tk, p)
    Q_s = Q_s_factor(Tk, p, q_v, q_l)
    dq = (q_v - q_sat - qss)/(1.0 + Q_s)

    # Adjust to ensure no negative water
    dq = min(q_v, dq)
    dq = max(-q_l, dq)
    return dq
end

"""
    q_condensation_qss(qss, Tk, p, rho_d, q_v, q_c, q_r, N_c)

Compute the condensation rate scaled by the condensation timescale, using the
supersaturation mixing ratio and cloud droplet properties.

# Arguments
- `qss`: Supersaturation mixing ratio (q_v - q_sat) [kg/kg]
- `Tk`: Temperature [K]
- `p`: Pressure [hPa]
- `rho_d`: Dry air density [kg/m³]
- `q_v`: Water vapor mixing ratio [kg/kg]
- `q_c`: Cloud water mixing ratio [kg/kg]
- `q_r`: Rain water mixing ratio [kg/kg]
- `N_c`: Cloud droplet number concentration [#/cm³]

# Returns
- `q_cond`: Condensation rate scaled by the inverse condensation timescale [kg/kg/s]

# References
- Ooyama (2001)
"""
function q_condensation_qss(qss, Tk, p, rho_d, q_v, q_c, q_r, N_c)

    q_l = q_c + q_r
    Q_s = Q_s_factor(Tk, p, q_v, q_l)
    cloudtau = 0.0
    if qss >= 0.0
        # Saturated so can condense water
        r_c = cloud_droplet_radius.(N_c, q_c, rho_d)
        cloudtau = invtau_condensation(Tk, p, N_c, r_c)
    else
        if q_c > 1.0e-8
            # Unsaturated with liquid water to evaporate
            r_c = cloud_droplet_radius.(N_c, q_c, rho_d)
            cloudtau = invtau_condensation(Tk, p, N_c, r_c)
        end
    end
    q_cond = qss/(1.0 + Q_s)
    # Adjust to ensure no negative water
    q_cond = min(q_v, q_cond)
    q_cond = max(-q_c, q_cond)
    q_cond = q_cond * cloudtau
    return q_cond #, cloudtau
end

"""
    q_condensation_relaxation(qss, Tk, p, q_v, q_l, N_c, r_c)

Compute the condensation or evaporation rate from an advected supersaturation
mixing ratio using a fixed-droplet-property relaxation timescale. This is the
scheme used by the restored `BF02_test` equation set and matches the formulation
that passed the Bryan & Fritsch (2002) moist benchmark (commit a4bf2a0).

# Arguments
- `qss`: Supersaturation mixing ratio (q_v - q_sat) [kg/kg]
- `Tk`: Temperature [K]
- `p`: Pressure [hPa]
- `q_v`: Water vapor mixing ratio [kg/kg]
- `q_l`: Liquid water mixing ratio [kg/kg]
- `N_c`: Cloud droplet number concentration [#/cm³]
- `r_c`: Cloud droplet radius [microns]

# Returns
- `q_cond`: Condensation rate scaled by the inverse condensation timescale [kg/kg/s]

# References
- Ooyama (2001); Bryan & Fritsch (2002)
"""
function q_condensation_relaxation(qss, Tk, p, q_v, q_l, N_c, r_c)

    Q_s = Q_s_factor(Tk, p, q_v, q_l)
    q_cond = qss/(1.0 + Q_s)
    # Adjust to ensure no negative water
    q_cond = min(q_v, q_cond)
    q_cond = max(-q_l, q_cond)
    invtau = invtau_condensation(Tk, p, N_c, r_c)
    return q_cond*invtau
end

"""
    q_condensation(sat_ratio, Tk, p, rho_d, q_v, q_c, max_N_c)

Compute the condensation or evaporation rate for cloud droplets using explicit
droplet growth physics, including nucleation via a Twomey-type activation and
a minimum droplet radius threshold.

# Arguments
- `sat_ratio`: Saturation ratio (q_v / q_sat) [dimensionless]
- `Tk`: Temperature [K]
- `p`: Pressure [hPa]
- `rho_d`: Dry air density [kg/m³]
- `q_v`: Water vapor mixing ratio [kg/kg]
- `q_c`: Cloud water mixing ratio [kg/kg]
- `max_N_c`: Maximum cloud droplet number concentration [#/cm³]

# Returns
- `q_cond`: Condensation rate [kg/kg/s]

# References
- Ooyama (2001)
"""
function q_condensation(sat_ratio, Tk, p, rho_d, q_v, q_c, max_N_c)

    q_cond = 0.0
    # N_c in #/cm^3, r_c in microns, which cancel out the units in the calculation
    N_c = max_N_c
    r_c = cloud_droplet_radius.(N_c, q_c, rho_d)
    if q_c > 1.0e-8
        if sat_ratio < 1.0
            # Evaporation is possible
            if r_c < 1.0
                # Get the number of 1 micron cloud droplets
                r_c = 1.0 # Set a minimum radius
                N_c = cloud_droplet_number(r_c, q_c, rho_d)
                if N_c < 1.0
                    # Not enough cloud droplets to evaporate
                    N_c = 0.0
                end
            else
                # Fully activated
                N_c = max_N_c
            end
        end
    elseif sat_ratio > 1.0001
        # Check for nucleation
        if r_c < 1.0
            # No cloud droplets, so set to 1 micron
            r_c = 1.0
            # Linear interpolation of Twomey relationship
            N_c = min(1.0e4 * N_c * (sat_ratio - 1.0), max_N_c)
        end
    end
    if N_c > 0.0 && r_c > 0.0
        # Calculate the condensation rate
        G = droplet_growth_rate(Tk, p)
        q_cond = 4.0 * pi * rho_l * G * (sat_ratio - 1.0) * N_c * r_c
    end

    # Adjust to ensure no negative water
    q_cond = min(q_v, q_cond)
    q_cond = max(-q_c, q_cond)
    return q_cond
end

"""
    q_evaporation(sat_ratio, Tk, p, rho_d, q_v, q_r, mean_r)

Compute the evaporation rate of rain drops using explicit droplet growth physics.
Evaporation occurs only when the environment is subsaturated and rain water is present.

# Arguments
- `sat_ratio`: Saturation ratio (q_v / q_sat) [dimensionless]
- `Tk`: Temperature [K]
- `p`: Pressure [hPa]
- `rho_d`: Dry air density [kg/m³]
- `q_v`: Water vapor mixing ratio [kg/kg]
- `q_r`: Rain water mixing ratio [kg/kg]
- `mean_r`: Mean radius of rain drops [μm]

# Returns
- `q_evap`: Evaporation rate (positive definite, vapor source) [kg/kg/s]

# References
- Ooyama (2001)
"""
function q_evaporation(sat_ratio, Tk, p, rho_d, q_v, q_r, mean_r)

    q_evap = 0.0
    # N_r in #/cm^3, r_r in microns, which cancel out the units in the calculation
    if q_r > 1.0e-8 && sat_ratio < 1.0
        # Evaporation is possible
        r_r = mean_r # Mean radius of the rain drops in microns
        N_r = cloud_droplet_number(r_r, q_r, rho_d)
        if N_r > 0.0 && r_r > 0.0
            # Calculate the evaporation rate
            G = droplet_growth_rate(Tk, p)
            q_evap = -4.0 * pi * rho_l * G * (sat_ratio - 1.0) * N_r * r_r
           # Adjust to ensure no negative water
            q_evap = min(q_r, q_evap)
        end
    end

    return q_evap
end

"""
    s_condensation(q_cond, Tk, rho_d, q_v, q_l, p)

Compute the entropy source/sink due to condensation or evaporation of cloud water.
The entropy change accounts for latent heating, liquid heat capacity, and vapor
pressure contributions.

# Arguments
- `q_cond`: Condensation rate (positive for condensation, negative for evaporation) [kg/kg/s]
- `Tk`: Temperature [K]
- `rho_d`: Dry air density [kg/m³]
- `q_v`: Water vapor mixing ratio [kg/kg]
- `q_l`: Total liquid water mixing ratio [kg/kg]
- `p`: Pressure [hPa]

# Returns
- `ds`: Entropy tendency due to condensation [J/(kg·K·s)]
"""
function s_condensation(q_cond, Tk, rho_d, q_v, q_l, p)

    Cm = (q_l * Cl)/(Cvd + (q_v * Cvv) + (q_l * Cl))
    e = vapor_pressure(p, q_v)
    sat_e = sat_pressure_liquid_buck(Tk, p)
    RH = e / sat_e
    if RH <= 0.0
        RH = 1.0e-6
    end
    ds = q_cond * ( ((-L_v(Tk)* Cm)/Tk) -(Cl * log(Tk / T_0)) + (Rv*(log(RH)) ))
    return ds
end

"""
    s_condensation(q_evap, q_cond, Tk, rho_d, q_v, q_l, p)

Compute the entropy source/sink due to the net effect of condensation and evaporation.
This method accepts separate condensation and evaporation rates and computes the
entropy change from their difference.

# Arguments
- `q_evap`: Evaporation rate (positive definite) [kg/kg/s]
- `q_cond`: Condensation rate (positive for condensation) [kg/kg/s]
- `Tk`: Temperature [K]
- `rho_d`: Dry air density [kg/m³]
- `q_v`: Water vapor mixing ratio [kg/kg]
- `q_l`: Total liquid water mixing ratio [kg/kg]
- `p`: Pressure [hPa]

# Returns
- `ds`: Entropy tendency due to net phase change [J/(kg·K·s)]
"""
function s_condensation(q_evap, q_cond, Tk, rho_d, q_v, q_l, p)

    Cm = (q_l * Cl)/(Cvd + (q_v * Cvv) + (q_l * Cl))
    e = vapor_pressure(p, q_v)
    sat_e = sat_pressure_liquid_buck(Tk, p)
    RH = e / sat_e
    if RH <= 0.0
        RH = 1.0e-6
    end
    ds = (q_cond - q_evap) * ( ((-L_v(Tk)* Cm)/Tk) -(Cl * log(Tk / T_0)) + (Rv*(log(RH))) )
    return ds
end

"""
    s_condensation_relaxation(q_cond, Tk, rho_d, q_v, q_l, p)

Compute the entropy source/sink due to condensation for the restored `BF02_test`
qss-relaxation scheme. Matches the formulation that passed the Bryan & Fritsch
(2002) moist benchmark (commit a4bf2a0); differs from [`s_condensation`](@ref)
by the absence of the `+Rv` vapor entropy term added later for the primitive
equation set.

# Arguments
- `q_cond`: Condensation rate (positive for condensation) [kg/kg/s]
- `Tk`: Temperature [K]
- `rho_d`: Dry air density [kg/m³]
- `q_v`: Water vapor mixing ratio [kg/kg]
- `q_l`: Total liquid water mixing ratio [kg/kg]
- `p`: Pressure [hPa]

# Returns
- `ds`: Entropy tendency due to condensation [J/(kg·K·s)]
"""
function s_condensation_relaxation(q_cond, Tk, rho_d, q_v, q_l, p)

    if q_cond == 0.0
        # Avoid 0 * log(0) = NaN when there is no vapor and no condensation
        return 0.0
    end
    Cm = (q_l * Cl)/(Cvd + (q_v * Cvv) + (q_l * Cl))
    e = vapor_pressure(p, q_v)
    sat_e = sat_pressure_liquid_buck(Tk, p)
    ds = q_cond * ( ((-L_v(Tk)* Cm)/Tk) -(Cl * log(Tk / T_0)) + (Rv*log(e/sat_e)) )
    return ds
end

"""
    s_vapor_mixing(q_flux, Tk, rho_d, q_v)

Compute the entropy change due to external addition or removal of water vapor
without phase change (e.g., turbulent mixing or surface fluxes). The heating terms
are zero, so only the vapor entropy contribution remains.

# Arguments
- `q_flux`: Rate of vapor mixing ratio change from external sources [kg/kg/s]
- `Tk`: Temperature [K]
- `rho_d`: Dry air density [kg/m³]
- `q_v`: Water vapor mixing ratio [kg/kg]

# Returns
- `ds`: Entropy tendency due to vapor mixing [J/(kg·K·s)]
"""
function s_vapor_mixing(q_flux, Tk, rho_d, q_v)

    # If vapor is mixed or externally added or removed without condensation
    # then the heating terms are zero and this is the entropy change
    ds = q_flux * ( vapor_entropy(Tk, rho_d, q_v) - Rv )
    return ds
end

"""
    Q_s_factor(Tk, p, q_v, q_l)

Compute the thermodynamic factor Q_s that accounts for the temperature dependence
of the saturation mixing ratio in the linearized condensation equation. This factor
appears in the denominator of the saturation adjustment (1 + Q_s).

# Arguments
- `Tk`: Temperature [K]
- `p`: Pressure [hPa]
- `q_v`: Water vapor mixing ratio [kg/kg]
- `q_l`: Liquid water mixing ratio [kg/kg]

# Returns
- `Q_s`: Dimensionless thermodynamic factor [dimensionless]
"""
function Q_s_factor(Tk, p, q_v, q_l)

    q_sat = q_sat_liquid(Tk, p)
    e_s = sat_pressure_liquid_buck(Tk, p)
    dqsdT = sat_pressure_liquid_buck_dT(Tk,p) * Eps * p / (p - e_s)^2
    Q_s = L_v(Tk) * dqsdT /(Cpd + ((q_v) * Cpv) + ((q_l) * Cl))
end

"""
    dqsdp(Tk, p, rho_d, q_v, q_l)

Compute the derivative of the saturation mixing ratio with respect to pressure,
accounting for both the direct pressure dependence and the indirect effect through
temperature changes at constant entropy.

# Arguments
- `Tk`: Temperature [K]
- `p`: Pressure [hPa]
- `rho_d`: Dry air density [kg/m³]
- `q_v`: Water vapor mixing ratio [kg/kg]
- `q_l`: Liquid water mixing ratio [kg/kg]

# Returns
- `dqsdp`: Pressure derivative of saturation mixing ratio [kg/kg/hPa]
"""
function dqsdp(Tk, p, rho_d, q_v, q_l)

    q_sat = q_sat_liquid(Tk, p)
    e_s = sat_pressure_liquid_buck(Tk, p)
    dqsdT = sat_pressure_liquid_buck_dT(Tk,p) * Eps * p / (p - e_s)^2
    dqsdp = (q_sat/(100.0*(p-e_s)) - (dqsdT /(rho_d*(Cpd + ((q_v) * Cpv) + ((q_l) * Cl)))))
    return dqsdp
end

"""
    invtau_condensation(Tk, p, N_c, r_c)

Compute the inverse condensation timescale (1/τ) based on vapor diffusivity and
cloud droplet properties. A larger value indicates faster relaxation toward saturation.

# Arguments
- `Tk`: Temperature [K]
- `p`: Pressure [hPa]
- `N_c`: Cloud droplet number concentration [#/cm³]
- `r_c`: Cloud droplet radius [μm]

# Returns
- `invtau`: Inverse condensation timescale [1/s]
"""
function invtau_condensation(Tk, p, N_c, r_c)

    Dv = vapor_diffusivity(Tk, p)
    # Nc in #/cm^3, r_c in microns
    invtau = 4 * pi * Dv * N_c * (r_c*1.0e-4)
    return invtau
end

"""
    cloud_droplet_radius(N_c, q_c, rho_d)

Compute the mean cloud droplet radius assuming a monodisperse distribution of
spherical liquid water droplets.

# Arguments
- `N_c`: Cloud droplet number concentration [#/cm³]
- `q_c`: Cloud water mixing ratio [kg/kg]
- `rho_d`: Dry air density [kg/m³]

# Returns
- `r_c`: Mean cloud droplet radius [μm]
"""
function cloud_droplet_radius(N_c, q_c, rho_d)

    # Nc in #/cm^3, r_c in microns
    rho_c = q_c * rho_d # kg/m^3
    # Mass per cloud droplet
    kg_drop = rho_c / (N_c * 1.0e6)
    r_c = 1.0e6 * (kg_drop * 3.0 / (4000.0 * pi))^(1.0/3.0)
    # Set a minimum radius
    #if r_c < 10.0
    #    r_c = 10.0
    #end
    return r_c
end

"""
    cloud_droplet_number(r_c, q_c, rho_d)

Compute the cloud droplet number concentration given a known droplet radius and
cloud water content, assuming a monodisperse distribution of spherical droplets.
This is the inverse of [`cloud_droplet_radius`](@ref).

# Arguments
- `r_c`: Cloud droplet radius [μm]
- `q_c`: Cloud water mixing ratio [kg/kg]
- `rho_d`: Dry air density [kg/m³]

# Returns
- `N_c`: Cloud droplet number concentration [#/cm³]
"""
function cloud_droplet_number(r_c, q_c, rho_d)

    if r_c == 0.0
        return 0.0
    end
    # Nc in #/cm^3, r_c in microns
    rho_c = q_c * rho_d # kg/m^3
    # Mass per cloud droplet
    kg_drop = 4.0 * pi * (r_c * 1.0e-6)^3 * rho_l / 3.0 # kg
    N_c = rho_c / kg_drop # #/m^3
    return N_c * 1.0e-6 # Convert to #/cm^3
end

"""
    vapor_diffusivity(Tk, p)

Compute the diffusivity of water vapor in air as a function of temperature and pressure.

# Arguments
- `Tk`: Temperature [K]
- `p`: Pressure [hPa]

# Returns
- `Dv`: Vapor diffusivity [cm²/s]

# References
- Pruppacher & Klett (1997)
"""
function vapor_diffusivity(Tk, p)

    # From Pruppacher and Klett, 1997
    # Tk in K, p in hPa
    # Dv in cm^2/s
    return 0.211 * (Tk/273.15)^1.94 * (1013.25/p)
end

"""
    condensation_adjustment(mtile, colstart, colend, t)

Perform a saturation adjustment on the model tile columns, updating entropy (`s`),
total water (`mu`), and cloud water (`mu_c`) in place. Uses an explicit Euler method
with a relaxation timescale factor of 0.25. The supersaturation is diagnosed from
an advected saturation ratio variable.

# Arguments
- `mtile::ModelTile`: Model tile containing prognostic and reference state variables
- `colstart::Int64`: Starting index of the column range to adjust
- `colend::Int64`: Ending index of the column range to adjust
- `t::Int64`: Current time step index
"""
function condensation_adjustment(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    # Calculate the condensation rate from the advected variables
    s_index = mtile.model.grid_params.vars["s"]
    s = view(mtile.var_np1,colstart:colend,s_index)

    # Xi is not modified
    xi_index = mtile.model.grid_params.vars["xi"]
    xi = view(mtile.var_np1,colstart:colend,xi_index)

    mu_index = mtile.model.grid_params.vars["mu"]
    mu = view(mtile.var_np1,colstart:colend,mu_index)

    mu_c_index = mtile.model.grid_params.vars["mu_c"]
    mu_c = view(mtile.var_np1,colstart:colend,mu_c_index)

    mu_r_index = mtile.model.grid_params.vars["mu_r"]
    mu_r = view(mtile.var_np1,colstart:colend,mu_r_index)

    sat_ratio_index = mtile.model.grid_params.vars["sat_ratio"]
    sat_ratio = view(mtile.var_np1,colstart:colend,sat_ratio_index)
    #sat_ratio = inv_mu_transform.(mu_sat)
    #sat_ratio = max.(sat_ratio, 1.0e-6)

    # Get reference state
    s_total = s .+ ref_entropy(mtile.ref_state)[:,1]
    xi_total = xi .+ ref_xi(mtile.ref_state)[:,1]
    mubar = ref_mu(mtile.ref_state)[:,1]
    mu_total = mu .+ mubar
    satbar = ref_sat(mtile.ref_state)[:,1]

    thermo = thermodynamic_tuple.(s_total, xi_total, mu_total)
    q_v = [x[1] for x in thermo]    # Total water vapor mixing ratio
    rho_d = [x[2] for x in thermo]  # Dry air density
    Tk = [x[3] for x in thermo]     # Temperature in K
    p = [x[4] for x in thermo]      # Total air pressure
    q_c_total = inv_mu_transform.(mu_c) # Cloud mixing ratio
    q_c = q_c_total #.- q_v # Condensate mixing ratio
    mu_c_factor = dmudq.(mu_c, q_c_total)
    q_c[q_c .<= 1.0e-12] .= 0.0       # 4.1e-9 is a threshold for 1 micron drop per cm^3 at 1 kg/m^3
    q_r_total = inv_mu_transform.(mu_r) # Rain mixing ratio
    q_r = q_r_total #.- q_c_total # Precipitation mixing ratio
    q_r[q_r .<= 1.0e-12] .= 0.0
    q_l = q_c .+ q_r                # Liquid mixing ratio
    q_sat = q_sat_liquid.(Tk, p)
    Q_s = Q_s_factor.(Tk, p, q_v, q_l)

    #sat_ratio = inv_mu_transform.(sat_prime .+ satbar) # Saturation ratio
    q_sat = q_sat_liquid.(Tk, p)
    #sat_ratio = q_v ./ q_sat # Saturation ratio
    sat_ratio_adj = sat_ratio #max.(sat_ratio, 1.0e-6)
    qss = (q_sat .* sat_ratio_adj) .- q_sat # qss / q_sat = (q_v .- q_sat)/q_sat

    # Do the increment using explicit Euler integration
    tau_r = 0.25
    q_cond = (q_v .- q_sat .- qss) ./ (1.0 .+ Q_s)
    for i in 1:length(q_cond)
        if qss[i] < 0.0
            # Restrict spurious condensation if qss is negative
            if q_cond[i] > 0.0
                q_cond[i] = 0.0
            else
                # Restrict adjustment to available condensate
                q_cond[i] = max(-q_c[i], q_cond[i])
            end
        else
            # Restrict condensation to available water vapor
            q_cond[i] = min(q_v[i], q_cond[i])
        end
    end

    mu .= @. mu - tau_r * dmudq(mu_total, q_v) * q_cond
    mu_c .= @. mu_c + tau_r * dmudq(mu_c, q_c_total) * q_cond
    s .= @. s + tau_r * s_condensation(q_cond, Tk, rho_d, q_v, q_l, p)

    # Adjust the condensate and precipitation mixing ratios
    #mu_c_adjust = mu_transform.(q_c .+ q_v) .- ref_mu(mtile.ref_state)[:,1]
    #mu_c .= @. mu_c + tau_r * (mu_c_adjust - mu_c)

    #mu_r_adjust = mu_transform.(q_r .+ q_c .+ q_v) .- ref_mu(mtile.ref_state)[:,1]
    #mu_r .= @. mu_r + tau_r * (mu_r_adjust - mu_r)

    # Incorrect implicit method
    #dq = @. tau_r * ( ((2.0 * Q_s - 1.0) * q_v) - (0.75 * Q_s * qv_n) + q_sat + qss) / (1.0 + (1.25 * Q_s * tau_r))

end

"""
    condensation_adjustment_qss(mtile, colstart, colend, t)

Perform a saturation adjustment for the restored `BF02_test` equation set, which
carries liquid water in `mu_l` and an advected supersaturation mixing ratio in
`qss`. Updates entropy (`s`), water vapor (`mu`), and liquid water (`mu_l`) in
place using an explicit Euler increment with a relaxation factor of 0.25.

Restored from the formulation that passed the Bryan & Fritsch (2002) moist
benchmark (commit a4bf2a0), with two deliberate changes: the reference liquid
water profile (`mu_lbar`) is zero since `ReferenceState` no longer carries it,
and the condensation limiters are applied elementwise (the original applied
`min`/`max` to whole arrays, which compares lexicographically and was a no-op
in practice).

# Arguments
- `mtile::ModelTile`: Model tile containing prognostic and reference state variables
- `colstart::Int64`: Starting index of the column range to adjust
- `colend::Int64`: Ending index of the column range to adjust
- `t::Int64`: Current time step index
"""
function condensation_adjustment_qss(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    # Calculate the condensation rate from the advected variables
    s_index = mtile.model.grid_params.vars["s"]
    s = view(mtile.var_np1,colstart:colend,s_index)

    # Xi is not modified
    xi_index = mtile.model.grid_params.vars["xi"]
    xi = view(mtile.var_np1,colstart:colend,xi_index)

    mu_index = mtile.model.grid_params.vars["mu"]
    mu = view(mtile.var_np1,colstart:colend,mu_index)

    mu_l_index = mtile.model.grid_params.vars["mu_l"]
    mu_l = view(mtile.var_np1,colstart:colend,mu_l_index)

    qss_index = mtile.model.grid_params.vars["qss"]
    qss = view(mtile.var_np1,colstart:colend,qss_index)

    # Get total state from the reference profile. The reference liquid water
    # is zero, so mu_l is the full liquid water variable.
    s_total = s .+ ref_entropy(mtile.ref_state)[:,1]
    xi_total = xi .+ ref_xi(mtile.ref_state)[:,1]
    mu_total = mu .+ ref_mu(mtile.ref_state)[:,1]

    thermo = thermodynamic_tuple.(s_total, xi_total, mu_total)
    q_v = [x[1] for x in thermo]    # Total water vapor mixing ratio
    rho_d = [x[2] for x in thermo]  # Dry air density
    Tk = [x[3] for x in thermo]     # Temperature in K
    p = [x[4] for x in thermo]      # Total air pressure
    q_l = inv_mu_transform.(mu_l)   # Liquid mixing ratio
    q_sat = q_sat_liquid.(Tk, p)
    Q_s = Q_s_factor.(Tk, p, q_v, q_l)

    # Do the increment using explicit Euler integration
    tau_r = 0.25
    q_cond = (q_v .- q_sat .- qss) ./ (1.0 .+ Q_s)
    # Restrict condensation to available vapor and evaporation to available liquid
    q_cond = min.(q_v, q_cond)
    q_cond = max.(-q_l, q_cond)

    mu .= @. mu - tau_r * dmudq(mu_total, q_v) * q_cond
    mu_l .= @. mu_l + tau_r * dmudq(mu_l, q_l) * q_cond
    s .= @. s + tau_r * s_condensation_relaxation(q_cond, Tk, rho_d, q_v, q_l, p)

end

"""
    condensation_adjustment_BF02(mtile, colstart, colend, t)

Perform a saturation adjustment using the Bryan & Fritsch (2002) approach, updating
entropy (`s`), total water (`mu`), and cloud water (`mu_c`) in place. The supersaturation
is diagnosed directly from the thermodynamic state (q_v - q_sat) rather than an
advected saturation variable. Uses untransformed q_v stored in implicit forcing arrays.

# Arguments
- `mtile::ModelTile`: Model tile containing prognostic and reference state variables
- `colstart::Int64`: Starting index of the column range to adjust
- `colend::Int64`: Ending index of the column range to adjust
- `t::Int64`: Current time step index
"""
function condensation_adjustment_BF02(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    # Calculate the condensation rate from the advected variables
    s_index = mtile.model.grid_params.vars["s"]
    s = view(mtile.var_np1,colstart:colend,s_index)

    # Xi is not modified
    xi_index = mtile.model.grid_params.vars["xi"]
    xi = view(mtile.var_np1,colstart:colend,xi_index)

    mu_index = mtile.model.grid_params.vars["mu"]
    mu = view(mtile.var_np1,colstart:colend,mu_index)
    # Using mu implicit as placeholder for untransformed q_v
    qv_n = view(mtile.impdot_n,colstart:colend,mu_index)
    qv_nm1 = view(mtile.impdot_nm1,colstart:colend,mu_index)

    mu_c_index = mtile.model.grid_params.vars["mu_c"]
    mu_c = view(mtile.var_np1,colstart:colend,mu_c_index)

    #mu_r_index = mtile.model.grid_params.vars["mu_r"]
    #mu_r = view(mtile.var_np1,colstart:colend,mu_r_index)

    mu_sat_index = mtile.model.grid_params.vars["mu_sat"]
    mu_sat = view(mtile.var_np1,colstart:colend,mu_sat_index)
    sat_ratio = inv_mu_transform.(mu_sat)
    sat_ratio = max.(sat_ratio, 1.0e-6)

    # Store absolute qss in implicit forcing
    qss = view(mtile.impdot_n,colstart:colend,mu_sat_index)
    qss_nm1 = view(mtile.impdot_nm1,colstart:colend,mu_sat_index)

    # Get reference state
    s_total = s .+ ref_entropy(mtile.ref_state)[:,1]
    xi_total = xi .+ ref_xi(mtile.ref_state)[:,1]
    mu_total = mu .+ ref_mu(mtile.ref_state)[:,1]

    thermo = thermodynamic_tuple.(s_total, xi_total, mu_total)
    q_v = [x[1] for x in thermo]    # Total water vapor mixing ratio
    rho_d = [x[2] for x in thermo]  # Dry air density
    Tk = [x[3] for x in thermo]     # Temperature in K
    p = [x[4] for x in thermo]      # Total air pressure
    q_c = inv_mu_transform.(mu_c)                # Condensate mixing ratio
    q_r = 0.0 #inv_mu_transform.(mu_r .- mu_c)               # Precipitation mixing ratio
    q_l = q_c .+ q_r                # Liquid mixing ratio
    q_sat = q_sat_liquid.(Tk, p)
    Q_s = Q_s_factor.(Tk, p, q_v, q_l)

    q_sat = q_sat_liquid.(Tk, p)
    qss = q_v .- q_sat
    # Do the increment using explicit Euler integration
    tau_r = 0.25
    q_cond = (q_v .- q_sat .- qss) ./ (1.0 .+ Q_s)
    for i in 1:length(q_cond)
        if qss[i] < 0.0
            # Restrict spurious condensation if qss is negative
            if q_cond[i] > 0.0
                q_cond[i] = 0.0
            else
                # Restrict adjustment to available condensate
                q_cond[i] = max(-q_c[i], q_cond[i])
            end
        else
            # Restrict condensation to available water vapor
            q_cond[i] = min(q_v[i], q_cond[i])
        end
    end

    mu .= @. mu - tau_r * dmudq(mu_total, q_v) * q_cond
    mu_c .= @. mu_c + tau_r * dmudq(mu_c, q_c) * q_cond
    s .= @. s + tau_r * s_condensation(q_cond, Tk, rho_d, q_v, q_l, p)

    # Incorrect implicit method
    #dq = @. tau_r * ( ((2.0 * Q_s - 1.0) * q_v) - (0.75 * Q_s * qv_n) + q_sat + qss) / (1.0 + (1.25 * Q_s * tau_r))

end

"""
    condensation_adjustment_new(mtile, colstart, colend, t)

Perform a saturation adjustment using an advected saturation ratio variable with
reference state separation. Updates entropy (`s`), total water (`mu`), and cloud
water (`mu_c`) in place. Includes both cloud and rain water in the liquid budget.

# Arguments
- `mtile::ModelTile`: Model tile containing prognostic and reference state variables
- `colstart::Int64`: Starting index of the column range to adjust
- `colend::Int64`: Ending index of the column range to adjust
- `t::Int64`: Current time step index
"""
function condensation_adjustment_new(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    # Calculate the condensation rate from the advected variables
    s_index = mtile.model.grid_params.vars["s"]
    s = view(mtile.var_np1,colstart:colend,s_index)

    # Xi is not modified
    xi_index = mtile.model.grid_params.vars["xi"]
    xi = view(mtile.var_np1,colstart:colend,xi_index)

    mu_index = mtile.model.grid_params.vars["mu"]
    mu = view(mtile.var_np1,colstart:colend,mu_index)

    mu_c_index = mtile.model.grid_params.vars["mu_c"]
    mu_c = view(mtile.var_np1,colstart:colend,mu_c_index)

    mu_r_index = mtile.model.grid_params.vars["mu_r"]
    mu_r = view(mtile.var_np1,colstart:colend,mu_r_index)

    mu_sat_index = mtile.model.grid_params.vars["mu_sat"]
    mu_sat = view(mtile.var_np1,colstart:colend,mu_sat_index)

    # Get reference state
    s_total = s .+ ref_entropy(mtile.ref_state)[:,1]
    xi_total = xi .+ ref_xi(mtile.ref_state)[:,1]
    mu_total = mu .+ ref_mu(mtile.ref_state)[:,1]
    satbar = ref_sat(mtile.ref_state)[:,1]

    thermo = thermodynamic_tuple.(s_total, xi_total, mu_total)
    q_v = [x[1] for x in thermo]    # Total water vapor mixing ratio
    rho_d = [x[2] for x in thermo]  # Dry air density
    Tk = [x[3] for x in thermo]     # Temperature in K
    p = [x[4] for x in thermo]      # Total air pressure
    q_c = inv_mu_transform.(mu_c)                # Condensate mixing ratio
    q_r = inv_mu_transform.(mu_r)               # Precipitation mixing ratio
    q_l = q_c .+ q_r                # Liquid mixing ratio
    q_sat = q_sat_liquid.(Tk, p)
    Q_s = Q_s_factor.(Tk, p, q_v, q_l)

    sat_ratio = inv_mu_transform.(mu_sat .+ satbar)
    #sat_ratio = max.(sat_ratio, 1.0e-6)
    q_sat = q_sat_liquid.(Tk, p)
    #qss = q_v .- q_sat
    qss = (q_sat .* sat_ratio) .- q_sat

    # Do the increment using explicit Euler integration
    tau_r = 0.25
    q_cond = (q_v .- q_sat .- qss) ./ (1.0 .+ Q_s)
    for i in 1:length(q_cond)
        if qss[i] < 0.0
            # Restrict spurious condensation if qss is negative
            if q_cond[i] > 0.0
                q_cond[i] = 0.0
            else
                # Restrict adjustment to available condensate
                q_cond[i] = max(-q_c[i], q_cond[i])
            end
        else
            # Restrict condensation to available water vapor
            q_cond[i] = min(q_v[i], q_cond[i])
        end
    end

    mu .= @. mu - tau_r * dmudq(mu_total, q_v) * q_cond
    mu_c .= @. mu_c + tau_r * dmudq(mu_c, q_c) * q_cond
    s .= @. s + tau_r * s_condensation(q_cond, Tk, rho_d, q_v, q_l, p)

end

"""
    condensation_adjustment_new_rhod(mtile, colstart, colend, t)

Linear dry-air-density variant of [`condensation_adjustment_new`](@ref) for the
`primitive_equation_XZ_rhod` set: reads the density perturbation from the `"rho_d"`
slot and reconstructs the thermodynamic state via [`thermodynamic_tuple_rhod`](@ref)
(`rho_d = rho_d' + rhobar`) instead of `dry_density(xi + xibar)`. The density itself
is not modified — only `mu`, `mu_c`, `s`.
"""
function condensation_adjustment_new_rhod(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    # Calculate the condensation rate from the advected variables
    s_index = mtile.model.grid_params.vars["s"]
    s = view(mtile.var_np1,colstart:colend,s_index)

    # Density (rho_d') is not modified
    rho_d_index = mtile.model.grid_params.vars["rho_d"]
    rho_dp = view(mtile.var_np1,colstart:colend,rho_d_index)

    mu_index = mtile.model.grid_params.vars["mu"]
    mu = view(mtile.var_np1,colstart:colend,mu_index)

    mu_c_index = mtile.model.grid_params.vars["mu_c"]
    mu_c = view(mtile.var_np1,colstart:colend,mu_c_index)

    mu_r_index = mtile.model.grid_params.vars["mu_r"]
    mu_r = view(mtile.var_np1,colstart:colend,mu_r_index)

    mu_sat_index = mtile.model.grid_params.vars["mu_sat"]
    mu_sat = view(mtile.var_np1,colstart:colend,mu_sat_index)

    # Get reference state
    s_total = s .+ ref_entropy(mtile.ref_state)[:,1]
    rho_d = rho_dp .+ ref_rho_d(mtile.ref_state)[:,1]
    mu_total = mu .+ ref_mu(mtile.ref_state)[:,1]
    satbar = ref_sat(mtile.ref_state)[:,1]

    thermo = thermodynamic_tuple_rhod.(s_total, rho_d, mu_total)
    q_v = [x[1] for x in thermo]    # Total water vapor mixing ratio
    Tk = [x[3] for x in thermo]     # Temperature in K
    p = [x[4] for x in thermo]      # Total air pressure
    q_c = inv_mu_transform.(mu_c)                # Condensate mixing ratio
    q_r = inv_mu_transform.(mu_r)               # Precipitation mixing ratio
    q_l = q_c .+ q_r                # Liquid mixing ratio
    q_sat = q_sat_liquid.(Tk, p)
    Q_s = Q_s_factor.(Tk, p, q_v, q_l)

    sat_ratio = inv_mu_transform.(mu_sat .+ satbar)
    qss = (q_sat .* sat_ratio) .- q_sat

    # Do the increment using explicit Euler integration
    tau_r = 0.25
    q_cond = (q_v .- q_sat .- qss) ./ (1.0 .+ Q_s)
    for i in 1:length(q_cond)
        if qss[i] < 0.0
            if q_cond[i] > 0.0
                q_cond[i] = 0.0
            else
                q_cond[i] = max(-q_c[i], q_cond[i])
            end
        else
            q_cond[i] = min(q_v[i], q_cond[i])
        end
    end

    mu .= @. mu - tau_r * dmudq(mu_total, q_v) * q_cond
    mu_c .= @. mu_c + tau_r * dmudq(mu_c, q_c) * q_cond
    s .= @. s + tau_r * s_condensation(q_cond, Tk, rho_d, q_v, q_l, p)

end

"""
    condensation_adjustment_pd(mtile, colstart, colend, t)

Partial-density variant of [`condensation_adjustment_new_rhod`](@ref) for the
`primitive_equation_XZ_rhod_pd` set. Reads the vapor and cloud *partial densities*
(`rho_v`, `rho_c`) and the dry-air density (`rho_d`), reconstructs the thermodynamic
state via [`thermodynamic_tuple_pd`](@ref) (`q_v = rho_v/rho_d`), and moves condensed
mass between vapor and cloud as `Δrho_v = -Δrho_c = -τ_r·rho_d·q_cond`. Because the
update is in partial densities, the cloud+vapor water mass `rho_v + rho_c` is conserved
exactly. The advected saturation ratio is recovered from the physical reference's raw
`satbar` rescaled into the transformed (`mu`) convention (linear-mu assumption, matching
the rest of the partial-density set). Densities themselves are not modified — only
`rho_v`, `rho_c`, and `s`.
"""
function condensation_adjustment_pd(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    s_index = mtile.model.grid_params.vars["s"]
    s = view(mtile.var_np1,colstart:colend,s_index)

    # Density (rho_d') is not modified
    rho_d_index = mtile.model.grid_params.vars["rho_d"]
    rho_dp = view(mtile.var_np1,colstart:colend,rho_d_index)

    rho_v_index = mtile.model.grid_params.vars["rho_v"]
    rho_v = view(mtile.var_np1,colstart:colend,rho_v_index)

    rho_c_index = mtile.model.grid_params.vars["rho_c"]
    rho_c = view(mtile.var_np1,colstart:colend,rho_c_index)

    rho_r_index = mtile.model.grid_params.vars["rho_r"]
    rho_r = view(mtile.var_np1,colstart:colend,rho_r_index)

    mu_sat_index = mtile.model.grid_params.vars["mu_sat"]
    mu_sat = view(mtile.var_np1,colstart:colend,mu_sat_index)

    # Reference state (physical partial densities). Condensate may be absent
    # (MoistReferenceState ⇒ ref_rho_c is the scalar 0.0), so guard the accessor.
    refstate = mtile.ref_state
    rho_dbar = ref_rho_d(refstate)[:,1]
    rho_vbar = ref_rho_v(refstate)[:,1]
    rcbar = ref_rho_c(refstate)
    rho_cbar = rcbar === 0.0 ? zero(rho_dbar) : rcbar[:,1]
    satbar = mu_transform.(ref_sat(refstate)[:,1])   # raw → transformed (linear mu)

    s_total = s .+ ref_entropy(refstate)[:,1]
    rho_d = rho_dp .+ rho_dbar
    rho_v_total = rho_v .+ rho_vbar
    rho_c_total = rho_c .+ rho_cbar

    thermo = thermodynamic_tuple_pd.(s_total, rho_d, rho_v_total)
    q_v = [x[1] for x in thermo]    # Total water vapor mixing ratio
    Tk = [x[3] for x in thermo]     # Temperature in K
    p = [x[4] for x in thermo]      # Total air pressure
    q_c = rho_c_total ./ rho_d      # Cloud water mixing ratio
    q_r = rho_r ./ rho_d            # Rain water mixing ratio
    q_l = q_c .+ q_r                # Liquid mixing ratio
    q_sat = q_sat_liquid.(Tk, p)
    Q_s = Q_s_factor.(Tk, p, q_v, q_l)

    sat_ratio = inv_mu_transform.(mu_sat .+ satbar)
    qss = (q_sat .* sat_ratio) .- q_sat

    # Do the increment using explicit Euler integration
    tau_r = 0.25
    q_cond = (q_v .- q_sat .- qss) ./ (1.0 .+ Q_s)
    for i in 1:length(q_cond)
        if qss[i] < 0.0
            if q_cond[i] > 0.0
                q_cond[i] = 0.0
            else
                q_cond[i] = max(-q_c[i], q_cond[i])
            end
        else
            q_cond[i] = min(q_v[i], q_cond[i])
        end
    end

    # Move condensed mass between vapor and cloud partial densities (Δrho_v = -Δrho_c),
    # conserving rho_v + rho_c exactly.
    dmass = @. tau_r * rho_d * q_cond
    rho_v .= rho_v .- dmass
    rho_c .= rho_c .+ dmass
    s .= @. s + tau_r * s_condensation(q_cond, Tk, rho_d, q_v, q_l, p)

end

"""
    autoconversion(q_c, rho_d)

Compute the autoconversion rate of cloud water to rain water. Cloud water exceeding
a threshold of 1 g/kg is converted to rain at a rate of 0.001 per second.

# Arguments
- `q_c`: Cloud water mixing ratio [kg/kg]
- `rho_d`: Dry air density [kg/m³]

# Returns
- `q_auto`: Autoconversion rate [kg/kg/s]

# References
- Ooyama (2001)
"""
function autoconversion(q_c, rho_d)

    # From Ooyama (2001)
    q_auto = 0.001*(q_c - 0.001)
    if q_auto < 0.0
        q_auto = 0.0
    end
    return q_auto
end

"""
    collection(q_c, q_r, rho_d, Tk)

Compute the collection (accretion) rate at which rain drops collect cloud droplets,
modulated by the ice fraction factor.

# Arguments
- `q_c`: Cloud water mixing ratio [kg/kg]
- `q_r`: Rain water mixing ratio [kg/kg]
- `rho_d`: Dry air density [kg/m³]
- `Tk`: Temperature [K]

# Returns
- `q_coll`: Collection rate [kg/kg/s]

# References
- Ooyama (2001)
"""
function collection(q_c, q_r, rho_d, Tk)

    # From Ooyama (2001)
    q_coll = 2.20* q_c * (q_r)^0.875 * f_ice(Tk)
    if q_coll < 0.0
        q_coll = 0.0
    end
    return q_coll
end

"""
    f_ice(Tk)

Compute the ice fraction factor that modulates microphysical process rates at
subfreezing temperatures. Currently returns 1.0 (ice effects disabled).

# Arguments
- `Tk`: Temperature [K]

# Returns
- Ice fraction factor [dimensionless], currently always 1.0

# References
- Ooyama (2001)
"""
function f_ice(Tk)

    # Turn this off for now
    return 1.0

    # From Ooyama (2001)
    #if Tk < 273.15
    #    return 0.2 + 0.8 * sech((273.15 - Tk)/5.0)
    #else
    #    return 1.0
    #end
end

"""
    df_icedz(Tk)

Compute the vertical derivative of the ice fraction factor with respect to height.
Used in the sedimentation flux divergence calculation. Returns zero for temperatures
at or above freezing.

# Arguments
- `Tk`: Temperature [K]

# Returns
- Vertical derivative of the ice fraction [1/m]

# References
- Ooyama (2001)
"""
function df_icedz(Tk)

    # From Ooyama (2001)
    if Tk < 273.15
        df_ice = -0.16 * (273.15 - Tk) * sech((273.15 - Tk)/5.0) * tanh((273.15 - Tk)/5.0)
        # Should account for the temperature gradient as well but neglecting it for now
        return df_ice
    else
        return 0.0
    end
end

"""
    rain_evaporation(q_r, rho_d, Tk, p)

Compute the rain evaporation coefficient, which is multiplied by the supersaturation
(qss) to obtain the actual evaporation rate. Accounts for ventilation effects and
thermodynamic constraints from vapor diffusivity and thermal conductivity.

# Arguments
- `q_r`: Rain water mixing ratio [kg/kg]
- `rho_d`: Dry air density [kg/m³]
- `Tk`: Temperature [K]
- `p`: Pressure [hPa]

# Returns
- `q_evap`: Rain evaporation coefficient [1/s] (multiply by qss to get rate)

# References
- Ooyama (2001)
"""
function rain_evaporation(q_r, rho_d, Tk, p)

    # Set the minimum cloud liquid mixing ratio for evaporation to occur
    #if q_r < 1.0e-8
    #    return 0.0
    #end
    # From Ooyama (2001)
    e_s = sat_pressure_liquid_buck(Tk, p)
    rho_vs = e_s / (Rv * Tk)
    rho_r = q_r * rho_d
    q_evap = (f_ventilation(q_r, rho_d, Tk) * rho_r^0.525)/(1.0e4*((2.03 * rho_vs) + (3.337/Tk)))
    if q_evap < 0.0
        q_evap = 0.0
    end
    # This is multiplied by qss so the 1/rho_d factor is already included
    return q_evap
end

"""
    f_ventilation(q_r, rho_d, Tk)

Compute the ventilation factor for rain drops, which enhances the evaporation rate
due to air flow around falling drops. Depends on the rain water content and the
ice fraction factor.

# Arguments
- `q_r`: Rain water mixing ratio [kg/kg]
- `rho_d`: Dry air density [kg/m³]
- `Tk`: Temperature [K]

# Returns
- `f_vent`: Ventilation factor [dimensionless]

# References
- Ooyama (2001)
"""
function f_ventilation(q_r, rho_d, Tk)

    # From Ooyama (2001)
    rho_r = q_r * rho_d
    f_vent = 1.6 + 30.39 * rho_r^0.2046 * f_ice(Tk)^1.5
    if f_vent < 0.0
        f_vent = 0.0
    end
    return f_vent
end

"""
    sedimentation(q_r, rho_d, Tk)

Compute the mass-weighted terminal fall velocity of rain drops. The velocity is
negative (downward) and depends on rain water content, air density, and the ice
fraction factor.

# Arguments
- `q_r`: Rain water mixing ratio [kg/kg]
- `rho_d`: Dry air density [kg/m³]
- `Tk`: Temperature [K]

# Returns
- `Vt`: Mass-weighted terminal velocity [m/s] (negative downward)

# References
- Ooyama (2001)
"""
function sedimentation(q_r, rho_d, Tk)

    # From Ooyama (2001)
    rho_r = q_r * rho_d
    Vt = -14.164 * rho_r^0.1364 * (rho_d0/rho_d)^0.5 * f_ice(Tk)
    return Vt
end

"""
    precipitation_flux(q_r, rho_d, Tk, q_r_z, xi_z)

Compute the vertical flux divergence of precipitation due to sedimentation. This
includes contributions from the vertical gradients of rain mixing ratio and dry
air density (via xi).

# Arguments
- `q_r`: Rain water mixing ratio [kg/kg]
- `rho_d`: Dry air density [kg/m³]
- `Tk`: Temperature [K]
- `q_r_z`: Vertical gradient of rain mixing ratio [kg/kg/m]
- `xi_z`: Vertical gradient of the mass variable xi [1/m]

# Returns
- `Vt_flux`: Precipitation flux divergence tendency [kg/kg/s]

# References
- Ooyama (2001)
"""
function precipitation_flux(q_r, rho_d, Tk, q_r_z, xi_z)

    # From Ooyama (2001)
    f_ice_term = f_ice(Tk)
    term1 = 0.6364 * rho_d^(-0.3636) * xi_z * q_r^(1.1364) * f_ice_term
    term2 = 1.1364 * rho_d^(-0.3636) * q_r^(0.1364) * q_r_z * f_ice_term
    #term3 = rho_d^(-0.3636) * q_r^(0.1364) * df_icedz(Tk)
    Vt_flux = -14.164 * rho_d0^0.5 * (term1 + term2)
    return Vt_flux
end

"""
    droplet_growth_rate(Tk, p)

Compute the droplet growth rate factor G, which combines the effects of vapor
diffusivity and thermal conductivity on condensational growth. The growth rate
of a droplet is dr/dt = G * (S - 1) / r, where S is the saturation ratio.

# Arguments
- `Tk::Float64`: Temperature [K]
- `p::Float64`: Pressure [hPa]

# Returns
- `G`: Droplet growth rate factor [m²/s]

# References
- Pruppacher & Klett (1997) for vapor diffusivity
"""
function droplet_growth_rate(Tk::Float64, p::Float64)

    # Tk in K, p in hPa
    Dv = vapor_diffusivity(Tk, p)
    diffusivity = rho_l * Rv * Tk / (Dv * sat_pressure_liquid_buck(Tk, p) * 1.0e-2) # m^2/s
    k = thermal_conductivity(Tk)
    L_c = L_v(Tk)
    conductivity = rho_l * L_c^2 / (k * Rv * Tk^2)
    G = 1.0 / (diffusivity + conductivity)
    return G
end

"""
    rain_adjustment(mtile, colstart, colend, t)

Remove accumulated rain from the lowest model level by adjusting the rain water
variable (`mu_r`) at the surface. This represents instantaneous precipitation
removal at the end of each time step.

# Arguments
- `mtile::ModelTile`: Model tile containing prognostic and reference state variables
- `colstart::Int64`: Starting index of the column range to adjust
- `colend::Int64`: Ending index of the column range to adjust
- `t::Int64`: Current time step index
"""
function rain_adjustment(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    # Remove rain from the surface
    mu_c_index = mtile.model.grid_params.vars["mu_c"]
    mu_c = view(mtile.var_np1,colstart:colend,mu_c_index)

    mu_r_index = mtile.model.grid_params.vars["mu_r"]
    mu_r = view(mtile.var_np1,colstart:colend,mu_r_index)

    mubar = ref_mu(mtile.ref_state)[1,1]
    q_c_total = inv_mu_transform(mu_c[1] + mubar) # Cloud mixing ratio
    q_r_total = inv_mu_transform(mu_r[1] + mubar) # Rain mixing ratio
    q_r = q_r_total - q_c_total # Precipitation mixing ratio

    if q_r <= 1.0e-8
        # No rain to remove
        return
    end

    # Instantaneous rain removal at the end of each time step
    mu_r[1] = mu_r[1] - (dmudq(mu_r[1], q_r_total) * q_r)

end
