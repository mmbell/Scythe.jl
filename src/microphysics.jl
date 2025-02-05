function saturation_adjustment(s, xi, mu, mu_l, tol = eps())

    incr = 1.0e-5
    local q_v, rho_d, Tk, p = thermodynamic_tuple(s, xi, mu)
    local q_l = ahyp.(mu_l)               # Liquid mixing ratio
    local q_sat = q_sat_liquid(Tk, p)
    local e = vapor_pressure(p,q_v)
    local sat_e = sat_pressure_liquid_buck(Tk, p)
    local RH = e / sat_e
    local q_t = q_v + q_l
    iterations = 1
    dq = q_sat - q_v
    SS = RH - 1.0

    # If dq < tol (default eps) then it is numerically saturated
    if abs(SS) < tol
        return 0.0
    end

    # Check to see if evaporation or condensation are possible
    q_test = q_v + dq
    if q_test < 0.0
        dq = -q_v
        #println("No water left to condense")
        return dq
    end
    q_test = q_l - dq
    if q_test < 0.0
        dq = q_l
        #println("Evaporated all the water")
        return dq
    end

    while abs(SS) > tol && iterations < 10

        dq_up = dq + incr
        s_incr = dq_up * ((Cl * log(Tk / T_0)) - (Rv * log(RH)))
        mu_incr = dmudq(mu, q_v) * dq_up
        new_q_v, new_rho_d, new_Tk, new_p = thermodynamic_tuple(s + s_incr, xi, mu + mu_incr)
        new_e = vapor_pressure(new_p, new_q_v)
        new_sat_e = sat_pressure_liquid_buck(new_Tk, new_p)
        RH_up = new_e / new_sat_e
        SS_up = RH_up - 1.0

        s_incr = dq * ((Cl * log(Tk / T_0)) - (Rv * log(RH)))
        mu_incr = dmudq(mu, q_v) * dq
        new_q_v, new_rho_d, new_Tk, new_p = thermodynamic_tuple(s + s_incr, xi, mu + mu_incr)
        new_e = vapor_pressure(new_p, new_q_v)
        new_sat_e = sat_pressure_liquid_buck(new_Tk, new_p)
        RH_down = new_e / new_sat_e
        SS = SS_down = RH_down - 1.0
        dSSdq = (SS_up - SS_down) / incr
        if abs(dSSdq) > 0
            dq = dq - (SS/dSSdq)
        else
            break
        end
        #println("$iterations: $new_q_v, $new_q_sat, $f, $new_q_t, $new_RH = $dq")
        iterations += 1
    end

    # Adjust to ensure no negative water
    q_test = q_v + dq
    if q_test < 0.0
        dq = -q_v
        #println("No water left to condense")
    end
    q_test = q_l - dq
    if q_test < 0.0
        dq = q_l
        #println("Evaporated all the water")
    end

    return dq
end

function saturation_adjustment_qdiff(s, xi, mu, mu_l, tol = eps())

    f = 1.0e10
    incr = 1.0e-5
    local q_v, rho_d, Tk, p = thermodynamic_tuple(s, xi, mu)
    local q_l = ahyp.(mu_l)               # Liquid mixing ratio
    local q_sat = q_sat_liquid(Tk, p)
    local e = vapor_pressure(p,q_v)
    local sat_e = sat_pressure_liquid_buck(Tk, p)
    local RH = e / sat_e
    local q_t = q_v + q_l
    iterations = 1
    dq = q_sat - q_v

    # If dq < tol (default q0) then it is numerically saturated
    if abs(dq) < q0
        return 0.0
    end

    # Check to see if evaporation or condensation are possible
    q_test = q_v + dq
    if q_test < 0.0
        dq = -q_v
        #println("No water left to condense")
        return dq
    end
    q_test = q_l - dq
    if q_test < 0.0
        dq = q_l
        #println("Evaporated all the water")
        return dq
    end

    while abs(f) > tol && iterations < 10

        dq_up = dq + incr
        s_incr = dq_up * ((Cl * log(Tk / T_0)) - (Rv * log(RH)))
        mu_incr = dmudq(mu, q_v) * dq_up
        #mu_l_incr = -dmudq(mu_l, q_l) * dq_up

        new_q_v, new_rho_d, new_Tk, new_p = thermodynamic_tuple(s + s_incr, xi, mu + mu_incr)
        #new_q_l = ahyp.(mu_l + mu_l_incr)
        new_q_sat = q_sat_liquid(new_Tk, new_p)
        f_up = new_q_sat - new_q_v

        #dq_down = dq - incr
        #s_incr = dq_down * ((Cl * log(Tk / T_0)) - (Rv * log(RH)))
        #mu_incr = dmudq(mu, q_v) * dq_down
        #mu_l_incr = -dmudq(mu, q_v) * dq_down

        #new_q_v, new_rho_d, new_Tk, new_p = thermodynamic_tuple(s + s_incr, xi, mu + mu_incr)
        #new_q_l = ahyp.(mu_l + mu_l_incr)
        #new_q_sat = q_sat_liquid(new_Tk, new_p)
        #f_down = new_q_sat - new_q_v

        s_incr = dq * ((Cl * log(Tk / T_0)) - (Rv * log(RH)))
        mu_incr = dmudq(mu, q_v) * dq
        #mu_l_incr = -dmudq(mu_l, q_l) * dq

        new_q_v, new_rho_d, new_Tk, new_p = thermodynamic_tuple(s + s_incr, xi, mu + mu_incr)
        #new_q_l = ahyp.(mu_l + mu_l_incr)
        new_q_sat = q_sat_liquid(new_Tk, new_p)
        #new_e = vapor_pressure(new_p, new_q_v)
        #new_sat_e = sat_pressure_liquid(new_Tk)
        #new_RH = new_e / new_sat_e
        #new_q_t = new_q_v + new_q_l
        f = new_q_sat - new_q_v
        dfdq = (f_up - f) / incr
        if abs(dfdq) > 0
            dq = dq - (f/dfdq)
        else
            break
        end
        #println("$iterations: $new_q_v, $new_q_sat, $f, $new_q_t, $new_RH = $dq")
        iterations += 1
    end

    # Adjust to ensure no negative water
    q_test = q_v + dq
    if q_test < 0.0
        dq = -q_v
        #println("No water left to condense")
    end
    q_test = q_l - dq
    if q_test < 0.0
        dq = q_l
        #println("Evaporated all the water")
    end

    return dq
end

function condensation(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    # Calculate the condensation rate from the advected variables
    s = view(mtile.var_np1,colstart:colend,1,1)
    xi = view(mtile.var_np1,colstart:colend,2,1)
    mu = view(mtile.var_np1,colstart:colend,3,1)
    mu_l = view(mtile.var_np1,colstart:colend,6,1)

    # Get reference state
    s_total = s .+ mtile.ref_state.sbar[:,1]
    xi_total = xi .+ mtile.ref_state.xibar[:,1]
    mu_total = mu .+ mtile.ref_state.mubar[:,1]
    
    thermo = thermodynamic_tuple.(s_total, xi_total, mu_total)
    q_v = [x[1] for x in thermo]    # Total water vapor mixing ratio
    rho_d = [x[2] for x in thermo]  # Dry air density
    Tk = [x[3] for x in thermo]     # Temperature in K
    p = [x[4] for x in thermo]      # Total air pressure
    q_l = ahyp.(mu_l)               # Liquid mixing ratio
    q_sat = q_sat_liquid.(Tk, p)
    e = vapor_pressure.(p,q_v)
    sat_e = sat_pressure_liquid.(Tk)
    RH = e ./ sat_e
    dq = saturation_adjustment.(s_total, xi_total, mu_total, mu_l)
    ts = mtile.model.ts

    # Do the increment directly using implicit timestep assuming saturation at the end of the timestep
    s .= @. s + (0.5 * ts * dq * ((Cl * log(Tk / T_0)) - (Rv * log(RH))))
    mu .= @. mu + (0.5 * ts * dq * dmudq(mu_total, q_v))
    mu_l .= @. mu_l - (0.5 * ts * dq * dmudq(mu_l, q_l))

end