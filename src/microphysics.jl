function saturation_adjustment(s, xi, mu, mu_l, tol)

    incr = 1.0e-6
    local q_v, rho_d, Tk, p = thermodynamic_tuple(s, xi, mu)

    # Check to see if evaporation or condensation are possible
    if q_v == 0.0
        # No water in this simulation
        return (0.0, 0.0)
    end
    
    local q_l = ahyp.(mu_l)               # Liquid mixing ratio
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

function q_condensation(qss, Tk, p, q_v, q_l, N_c, r_c)

    Q_s = Q_s_factor(Tk, p, q_v, q_l)
    q_cond = qss/(1.0 + Q_s)
    # Adjust to ensure no negative water
    q_cond = min(q_v, q_cond)
    q_cond = max(-q_l, q_cond)
    invtau = invtau_condensation(Tk, p, N_c, r_c)
    return q_cond*invtau
end

function s_condensation(q_cond, Tk, rho_d, q_v, q_l, p)

    Cm = (q_l * Cl)/(Cvd + (q_v * Cvv) + (q_l * Cl))
    #ds = q_cond * ((L_v(Tk)*(1.0 - Cm)/Tk) - Scythe.vapor_entropy(Tk, rho_d, q_v) + Rv)
    e = vapor_pressure(p, q_v)
    sat_e = sat_pressure_liquid_buck(Tk, p)

    ds = q_cond * ( ((-L_v(Tk)* Cm)/Tk) -(Cl * log(Tk / T_0)) + (Rv*log(e/sat_e)) )
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

    Dv = vapor_diffusity(Tk, p)
    # Nc in #/cm^3, r_c in microns
    invtau = 4 * pi * Dv * N_c * (r_c*1.0e-4)
    return invtau
end

function vapor_diffusity(Tk, p)

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
    # Using mu implicit as placeholder for untransformed q_v
    qv_n = view(mtile.impdot_n,colstart:colend,mu_index)
    qv_nm1 = view(mtile.impdot_nm1,colstart:colend,mu_index)

    mu_l_index = mtile.model.grid_params.vars["mu_l"]
    mu_l = view(mtile.var_np1,colstart:colend,mu_l_index)

    qss_index = mtile.model.grid_params.vars["qss"]
    qss = view(mtile.var_np1,colstart:colend,qss_index)
    qss_n = view(mtile.impdot_n,colstart:colend,qss_index)
    qss_nm1 = view(mtile.impdot_nm1,colstart:colend,qss_index)

    # Get reference state
    s_total = s .+ mtile.ref_state.sbar[:,1]
    xi_total = xi .+ mtile.ref_state.xibar[:,1]
    mu_total = mu .+ mtile.ref_state.mubar[:,1]
    mu_l_total = mu_l .+ mtile.ref_state.mu_lbar[:,1]

    thermo = thermodynamic_tuple.(s_total, xi_total, mu_total)
    q_v = [x[1] for x in thermo]    # Total water vapor mixing ratio
    rho_d = [x[2] for x in thermo]  # Dry air density
    Tk = [x[3] for x in thermo]     # Temperature in K
    p = [x[4] for x in thermo]      # Total air pressure
    q_l = ahyp.(mu_l_total)         # Liquid mixing ratio
    q_sat = q_sat_liquid.(Tk, p)
    Q_s = Q_s_factor.(Tk, p, q_v, q_l)

    # Do the increment using explicit Euler integration
    tau_r = 0.25
    #dq = @. tau_r * ( ((2.0 * Q_s - 1.0) * q_v) - (0.75 * Q_s * qv_n) + q_sat + qss) / (1.0 + (1.25 * Q_s * tau_r))
    q_cond = (q_v .- q_sat .- qss) ./ (1.0 .+ Q_s)
    q_cond = min(q_v, q_cond)
    q_cond = max(-q_l, q_cond)
    mu .= @. mu - tau_r * dmudq(mu_total, q_v) * q_cond
    mu_l .= @. mu_l + tau_r * dmudq(mu_l_total, q_l) * q_cond
    s .= @. s + tau_r * s_condensation(q_cond, Tk, rho_d, q_v, q_l, p)

    #mu .= @. mu + tau_r * dmudq(mu_total, q_v) * ((1.25 * qss) -(Q_s*qv_n + qss_n) + (0.75 * (Q_s*qv_nm1 + qss_nm1)) - (1.0 + Q_s)*q_v + q_sat) / (1.0 - 0.625*Q_s)
    #q_v_np1 = ahyp.(mu)
    #q_dotnp1 = -Q_s .* q_v_np1 .- qss
    #mu_l .= @. mu_l + tau_r * dmudq(mu_l_total, q_l) * ((1.25 * q_dotnp1) +(Q_s*qv_n + qss_n) - (0.75 * (Q_s*qv_nm1 + qss_nm1)) + (1.0 + Q_s)*q_v - q_sat)
    #s .= @. s + tau_r * ((L_v(Tk)/Tk) - Scythe.vapor_entropy(Tk, rho_d, q_v) + Rv)*((1.25 * q_dotnp1) +(Q_s*qv_n + qss_n) - (0.75 * (Q_s*qv_nm1 + qss_nm1)) + (1.0 + Q_s)*q_v - q_sat)

    #qv_nm1 .= qv_n
    #qss_nm1 .= qss_n

end
