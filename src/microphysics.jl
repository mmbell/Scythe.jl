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

function s_condensation(q_cond, Tk, rho_d, q_v, q_l, p)

    Cm = (q_l * Cl)/(Cvd + (q_v * Cvv) + (q_l * Cl))
    e = vapor_pressure(p, q_v)
    sat_e = sat_pressure_liquid_buck(Tk, p)
    RH = e / sat_e
    if RH <= 0.0
        RH = 1.0e-6
    end
    ds = q_cond * ( ((-L_v(Tk)* Cm)/Tk) -(Cl * log(Tk / T_0)) + (Rv*(log(RH)+1.0)) )
    return ds
end

function s_condensation(q_evap, q_cond, Tk, rho_d, q_v, q_l, p)

    Cm = (q_l * Cl)/(Cvd + (q_v * Cvv) + (q_l * Cl))
    e = vapor_pressure(p, q_v)
    sat_e = sat_pressure_liquid_buck(Tk, p)
    RH = e / sat_e
    if RH <= 0.0
        RH = 1.0e-6
    end
    ds = (q_cond - q_evap) * ( ((-L_v(Tk)* Cm)/Tk) -(Cl * log(Tk / T_0)) + (Rv*(log(RH)+1.0)) )
    return ds
end

function s_vapor_mixing(q_flux, Tk, rho_d, q_v)

    # If vapor is mixed or externally added or removed without condensation
    # then the heating terms are zero and this is the entropy change
    ds = q_flux * ( vapor_entropy(Tk, rho_d, q_v) - Rv )
    return ds
end

function Q_s_factor(Tk, p, q_v, q_l)

    q_sat = q_sat_liquid(Tk, p)
    e_s = sat_pressure_liquid_buck(Tk, p)
    dqsdT = sat_pressure_liquid_buck_dT(Tk,p) * Eps * p / (p - e_s)^2
    Q_s = L_v(Tk) * dqsdT /(Cpd + ((q_v) * Cpv) + ((q_l) * Cl))
end

function dqsdp(Tk, p, rho_d, q_v, q_l)

    q_sat = q_sat_liquid(Tk, p)
    e_s = sat_pressure_liquid_buck(Tk, p)
    dqsdT = sat_pressure_liquid_buck_dT(Tk,p) * Eps * p / (p - e_s)^2
    dqsdp = (q_sat/(100.0*(p-e_s)) - (dqsdT /(rho_d*(Cpd + ((q_v) * Cpv) + ((q_l) * Cl)))))
    return dqsdp
end

function invtau_condensation(Tk, p, N_c, r_c)

    Dv = vapor_diffusivity(Tk, p)
    # Nc in #/cm^3, r_c in microns
    invtau = 4 * pi * Dv * N_c * (r_c*1.0e-4)
    return invtau
end

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

function vapor_diffusivity(Tk, p)

    # From Pruppacher and Klett, 1997
    # Tk in K, p in hPa
    # Dv in cm^2/s
    return 0.211 * (Tk/273.15)^1.94 * (1013.25/p)
end

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
    s_total = s .+ mtile.ref_state.sbar[:,1]
    xi_total = xi .+ mtile.ref_state.xibar[:,1]
    mubar = mtile.ref_state.mubar[:,1]
    mu_total = mu .+ mubar
    satbar = mtile.ref_state.satbar[:,1]

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
    #mu_c_adjust = mu_transform.(q_c .+ q_v) .- mtile.ref_state.mubar[:,1]
    #mu_c .= @. mu_c + tau_r * (mu_c_adjust - mu_c)

    #mu_r_adjust = mu_transform.(q_r .+ q_c .+ q_v) .- mtile.ref_state.mubar[:,1]
    #mu_r .= @. mu_r + tau_r * (mu_r_adjust - mu_r)

    # Incorrect implicit method
    #dq = @. tau_r * ( ((2.0 * Q_s - 1.0) * q_v) - (0.75 * Q_s * qv_n) + q_sat + qss) / (1.0 + (1.25 * Q_s * tau_r))

end

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
    s_total = s .+ mtile.ref_state.sbar[:,1]
    xi_total = xi .+ mtile.ref_state.xibar[:,1]
    mu_total = mu .+ mtile.ref_state.mubar[:,1]

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
    s_total = s .+ mtile.ref_state.sbar[:,1]
    xi_total = xi .+ mtile.ref_state.xibar[:,1]
    mu_total = mu .+ mtile.ref_state.mubar[:,1]
    satbar = mtile.ref_state.satbar[:,1]

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

function autoconversion(q_c, rho_d)

    # From Ooyama (2001)
    q_auto = 0.001*(q_c - 0.001)
    if q_auto < 0.0
        q_auto = 0.0
    end
    return q_auto
end

function collection(q_c, q_r, rho_d, Tk)

    # From Ooyama (2001)
    q_coll = 2.20* q_c * (q_r)^0.875 * f_ice(Tk)
    if q_coll < 0.0
        q_coll = 0.0
    end
    return q_coll
end

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

function f_ventilation(q_r, rho_d, Tk)

    # From Ooyama (2001)
    rho_r = q_r * rho_d
    f_vent = 1.6 + 30.39 * rho_r^0.2046 * f_ice(Tk)^1.5
    if f_vent < 0.0
        f_vent = 0.0
    end
    return f_vent
end

function sedimentation(q_r, rho_d, Tk)

    # From Ooyama (2001)
    rho_r = q_r * rho_d
    Vt = -14.164 * rho_r^0.1364 * (rho_d0/rho_d)^0.5 * f_ice(Tk)
    return Vt
end

function precipitation_flux(q_r, rho_d, Tk, q_r_z, xi_z)

    # From Ooyama (2001)
    f_ice_term = f_ice(Tk)
    term1 = 0.6364 * rho_d^(-0.3636) * xi_z * q_r^(1.1364) * f_ice_term
    term2 = 1.1364 * rho_d^(-0.3636) * q_r^(0.1364) * q_r_z * f_ice_term
    #term3 = rho_d^(-0.3636) * q_r^(0.1364) * df_icedz(Tk)
    Vt_flux = -14.164 * rho_d0^0.5 * (term1 + term2)
    return Vt_flux
end

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

function rain_adjustment(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    # Remove rain from the surface
    mu_c_index = mtile.model.grid_params.vars["mu_c"]
    mu_c = view(mtile.var_np1,colstart:colend,mu_c_index)

    mu_r_index = mtile.model.grid_params.vars["mu_r"]
    mu_r = view(mtile.var_np1,colstart:colend,mu_r_index)

    mubar = mtile.ref_state.mubar[1,1]
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
