function LinearAdvection1D(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    #1D Linear advection to test
    grid = mtile.tile
    expdot = mtile.expdot_n
    model = mtile.model

    c_0 = model.physical_params[:c_0]
    K = model.physical_params[:K]

    u = view(grid.physical,:,1,1)
    ur = view(grid.physical,:,1,2)
    urr = view(grid.physical,:,1,3)

    expdot[:,1] .= -(c_0 .* ur) .+ (K .* urr)

    # Advance the explicit terms
    explicit_timestep(mtile, colstart, colend, t)

end

function LinearAdvectionRZ(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    #2D RZ Linear advection to test
    grid = mtile.tile
    gridpoints = mtile.tilepoints
    expdot = mtile.expdot_n
    model = mtile.model

    K = model.physical_params[:K]

    r = view(gridpoints,:,1)
    hr = view(grid.physical,:,1,2)
    hrr = view(grid.physical,:,1,3)
    hz = view(grid.physical,:,1,4)
    hzz = view(grid.physical,:,1,5)
    u = view(grid.physical,:,2,1)
    w = view(grid.physical,:,4,1)

    @turbo expdot[:,1] .= @. (-u * hr) + (-w * hz) + (K * ((hr / r) + hrr + hzz))

    # Advance the explicit terms
    explicit_timestep(mtile, colstart, colend, t)

end

function LinearAdvectionRL(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    #2D Linear advection to test
    grid = mtile.tile
    gridpoints = mtile.tilepoints
    expdot = mtile.expdot_n
    model = mtile.model

    K = model.physical_params[:K]
    r = view(gridpoints,:,1)
    hr = view(grid.physical,:,1,2)
    hl = view(grid.physical,:,1,4)
    u = view(grid.physical,:,2,1)
    v = view(grid.physical,:,3,1)

    if K > 0.0
        hrr = view(grid.physical,:,1,3)
        hll = view(grid.physical,:,1,5)
        @turbo expdot[:,1] .= @. (-u * hr) - (v * (hl / r)) + (K * ((hr / r) + hrr + (hll / (r * r))))
    else
        @turbo expdot[:,1] .= @. (-u * hr) - (v * (hl / r))
    end

    # Advance the explicit terms
    explicit_timestep(mtile, colstart, colend, t)

end

function LinearAdvectionRLZ(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    #3D Linear advection to test
    grid = mtile.tile
    gridpoints = mtile.tilepoints
    expdot = mtile.expdot_n
    model = mtile.model
    K = model.physical_params[:K]

    r = gridpoints[:,1]
    h = view(grid.physical,:,1,1)
    hr = view(grid.physical,:,1,2)
    hrr = view(grid.physical,:,1,3)
    hl = view(grid.physical,:,1,4)
    hll = view(grid.physical,:,1,5)
    u = view(grid.physical,:,2,1)
    v = view(grid.physical,:,3,1)

    @turbo expdot[:,1] .= (-u .* hr) .- (v .* (hl ./ r)) .+ (K .* ((hr ./ r) .+ hrr .+ (hll ./ (r .* r))))

    # Advance the explicit terms
    explicit_timestep(mtile, colstart, colend, t)

end

function Euler_test(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)
    
    grid = mtile.tile
    gridpoints = mtile.tilepoints
    expdot = mtile.expdot_n
    impdot = mtile.impdot_n
    model = mtile.model
    refstate = mtile.ref_state

    # Physical parameters
    K = model.physical_params[:K]

    # Gridpoints
    x = view(gridpoints,colstart:colend,1)
    z = view(gridpoints,colstart:colend,2)

    # Variables
    s = view(grid.physical,colstart:colend,1,1)
    s_x = view(grid.physical,colstart:colend,1,2)
    s_xx = view(grid.physical,colstart:colend,1,3)
    s_z = view(grid.physical,colstart:colend,1,4)
    s_zz = view(grid.physical,colstart:colend,1,5)
    
    xi = view(grid.physical,colstart:colend,2,1)
    xi_x = view(grid.physical,colstart:colend,2,2)
    xi_xx = view(grid.physical,colstart:colend,2,3)
    xi_z = view(grid.physical,colstart:colend,2,4)
    xi_zz = view(grid.physical,colstart:colend,2,5)
    
    mu = view(grid.physical,colstart:colend,3,1)
    mu_x = view(grid.physical,colstart:colend,3,2)
    mu_xx = view(grid.physical,colstart:colend,3,3)
    mu_z = view(grid.physical,colstart:colend,3,4)
    mu_zz = view(grid.physical,colstart:colend,3,5)
    
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
    
    # Get reference state
    sbar = refstate.sbar[:,1]
    sbar_z = refstate.sbar[:,2]
    sbar_zz = refstate.sbar[:,3]

    xibar = refstate.xibar[:,1]
    xibar_z = refstate.xibar[:,2]
    xibar_zz = refstate.xibar[:,3]

    mubar = refstate.mubar[:,1]
    mubar_z = refstate.mubar[:,2]
    mubar_zz = refstate.mubar[:,3]
    
    # Fundamental thermodynamic quantities derived from model variables
    thermo = thermodynamic_tuple.(s .+ sbar, xi .+ xibar, mu .+ mubar)
    q_v = [x[1] for x in thermo]    # Total water vapor mixing ratio
    rho_d = [x[2] for x in thermo]  # Dry air density
    Tk = [x[3] for x in thermo]     # Temperature in K
    p = [x[4] for x in thermo]      # Total air pressure
    rho_t = rho_d .* (1.0 .+ q_v)   # Total air density
    #qvp = q_v .- ahyp.(mubar)       # Perturbation mixing ratio
    qvp_x = mu_x ./ dmudq.(mu .+ mubar, q_v) # Perturbation vapor gradient in x
    qvp_z = mu_z ./ dmudq.(mu .+ mubar, q_v) # Perturbation vapor gradient in z
    rhobar = dry_density.(xibar) .* (1.0 .+ ahyp.(mubar)) # Ref. air density
    rho_p = rho_t .- rhobar         # Perturbation air density
    
    # Get the mean speed of sound squared from the reference state
    Pxi_bar = mtile.ref_state.Pxi_bar

    # Placeholders for intermediate calculations
    ADV = similar(s)
    PGF = similar(s)
    KDIFF = similar(s)

    @turbo ADV .= @. (-u * s_x) + (-w * (s_z + sbar_z)) #SADV
    #No PGF
    @turbo KDIFF .= @. K * (s_xx + s_zz)
    @turbo expdot[colstart:colend,1] .= @. ADV + KDIFF
    
    @turbo ADV .= @. (-u * xi_x) + (-w * (xi_z + xibar_z)) #XI ADV
    # No PGF or mass diffusion
    @turbo expdot[colstart:colend,2] .= @. ADV - u_x - w_z
    impdot[colstart:colend,2] .= @. -w_z

    @turbo ADV .= @. (-u * mu_x) + (-w * (mu_z + mubar_z)) #SADV
    #No PGF
    @turbo KDIFF .= @. K * (mu_xx + mu_zz)
    @turbo expdot[colstart:colend,3] .= @. ADV + KDIFF
    
    @turbo ADV .= @. (-u * u_x) + (-w * u_z) #UADV
    PGF .= @. -(pressure_gradient(Tk, rho_d, q_v, s_x, xi_x, qvp_x) / rho_t) #UPGF
    @turbo KDIFF .= @. K * (u_xx + u_zz)
    @turbo expdot[colstart:colend,4] .= @. ADV + PGF + KDIFF

    @turbo ADV .= @. (-u * w_x) + (-w * w_z) #WADV
    PGF .= @.  -(gravity * rho_p / rho_t) - (pressure_gradient(Tk, rho_d, q_v, s_z, xi_z, qvp_z) / rho_t)
    @turbo KDIFF .= @. K * (w_xx + w_zz)
    @turbo expdot[colstart:colend,5] .= @. ADV + PGF + KDIFF
    impdot[colstart:colend,5] .= @. -(Pxi_bar * xi_z)

    # Advance the explicit terms
    explicit_timestep(mtile, colstart, colend, t)

    # Solve for semi-implicit n+1 terms
    if mtile.model.options[:semiimplicit]
        semiimplicit_adjustment(mtile, colstart, colend, t)
    end

end

function BF02_test(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    grid = mtile.tile
    gridpoints = mtile.tilepoints
    expdot = mtile.expdot_n
    impdot = mtile.impdot_n
    model = mtile.model
    refstate = mtile.ref_state

    # Physical parameters
    K = model.physical_params[:K]

    # Gridpoints
    x = view(gridpoints,colstart:colend,1)
    z = view(gridpoints,colstart:colend,2)

    # Variables
    s = view(grid.physical,colstart:colend,1,1)
    s_x = view(grid.physical,colstart:colend,1,2)
    s_xx = view(grid.physical,colstart:colend,1,3)
    s_z = view(grid.physical,colstart:colend,1,4)
    s_zz = view(grid.physical,colstart:colend,1,5)

    xi = view(grid.physical,colstart:colend,2,1)
    xi_x = view(grid.physical,colstart:colend,2,2)
    xi_xx = view(grid.physical,colstart:colend,2,3)
    xi_z = view(grid.physical,colstart:colend,2,4)
    xi_zz = view(grid.physical,colstart:colend,2,5)

    mu = view(grid.physical,colstart:colend,3,1)
    mu_x = view(grid.physical,colstart:colend,3,2)
    mu_xx = view(grid.physical,colstart:colend,3,3)
    mu_z = view(grid.physical,colstart:colend,3,4)
    mu_zz = view(grid.physical,colstart:colend,3,5)

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

    mu_l = view(grid.physical,colstart:colend,6,1)
    mu_l_x = view(grid.physical,colstart:colend,6,2)
    mu_l_xx = view(grid.physical,colstart:colend,6,3)
    mu_l_z = view(grid.physical,colstart:colend,6,4)
    mu_l_zz = view(grid.physical,colstart:colend,6,5)

    qss = view(grid.physical,colstart:colend,7,1)
    qss_x = view(grid.physical,colstart:colend,7,2)
    qss_xx = view(grid.physical,colstart:colend,7,3)
    qss_z = view(grid.physical,colstart:colend,7,4)
    qss_zz = view(grid.physical,colstart:colend,7,5)

    # Get reference state
    sbar = refstate.sbar[:,1]
    sbar_z = refstate.sbar[:,2]
    sbar_zz = refstate.sbar[:,3]

    xibar = refstate.xibar[:,1]
    xibar_z = refstate.xibar[:,2]
    xibar_zz = refstate.xibar[:,3]

    mubar = refstate.mubar[:,1]
    mubar_z = refstate.mubar[:,2]
    mubar_zz = refstate.mubar[:,3]

    mu_lbar = refstate.mu_lbar[:,1]
    mu_lbar_z = refstate.mu_lbar[:,2]
    mu_lbar_zz = refstate.mu_lbar[:,3]

    # Fundamental thermodynamic quantities derived from model variables
    mu_total = mu .+ mubar
    thermo = thermodynamic_tuple.(s .+ sbar, xi .+ xibar, mu_total)
    q_v = [x[1] for x in thermo]    # Total water vapor mixing ratio
    rho_d = [x[2] for x in thermo]  # Dry air density
    Tk = [x[3] for x in thermo]     # Temperature in K
    p = [x[4] for x in thermo]      # Total air pressure
    q_l = ahyp.(mu_l .+ mu_lbar)               # Liquid mixing ratio
    rho_t = rho_d .* (1.0 .+ q_v .+ q_l)   # Total air density
    qvp = q_v .- ahyp.(mubar)       # Perturbation mixing ratio
    mu_factor = dmudq.(mu_total, q_v)
    qvp_x = mu_x ./ mu_factor # Perturbation vapor gradient in x
    qvp_z = mu_z ./ mu_factor # Perturbation vapor gradient in z
    rhobar = dry_density.(xibar) .* (1.0 .+ ahyp.(mubar)) # Ref. air density
    rho_p = rho_t .- rhobar         # Perturbation air density

    # Get the mean speed of sound squared from the reference state
    Pxi_bar = mtile.ref_state.Pxi_bar

    # Pressure gradients
    dpdx = pressure_gradient.(Tk, rho_d, q_v, s_x, xi_x, qvp_x)
    dpdz = pressure_gradient.(Tk, rho_d, q_v, s_z, xi_z, qvp_z)

    # Placeholders for intermediate calculations
    ADV = similar(s)
    FORCING = similar(s)
    KDIFF = similar(s)

    # Entropy divergence forcing
    Cm = @. (q_l * Cl)/(Cvd + (q_v * Cvv) + (q_l * Cl))
    s_div = @. Cm * (Rd + q_v * Rv) * (u_x + w_z)

    # Condensation rate
    N_c = 500.0
    r_c = 10.0
    q_cond = q_condensation.(qss, Tk, p, q_v, q_l, N_c, r_c)
    s_cond = s_condensation.(q_cond, Tk, rho_d, q_v, q_l, p)
    Q_s = Q_s_factor.(Tk, p, q_v, q_l)
    invtau = invtau_condensation.(Tk, p, N_c, r_c)
    qss_cond = @. dqsdp(Tk, p, rho_d, q_v, q_l)*((u * dpdx) + (w * (dpdz - rhobar*gravity))) - qss * invtau

    @turbo ADV .= @. (-u * s_x) + (-w * (s_z + sbar_z)) #SADV
    FORCING .= @. s_cond + s_div
    @turbo KDIFF .= @. K * (s_xx + s_zz)
    @turbo expdot[colstart:colend,1] .= @. ADV + FORCING + KDIFF

    @turbo ADV .= @. (-u * xi_x) + (-w * (xi_z + xibar_z)) #XI ADV
    @turbo FORCING .= @. - u_x - w_z
    @turbo expdot[colstart:colend,2] .= @. ADV + FORCING
    impdot[colstart:colend,2] .= @. -w_z

    @turbo ADV .= @. (-u * mu_x) + (-w * (mu_z + mubar_z)) #MUADV
    FORCING .= @. -q_cond * mu_factor
    @turbo KDIFF .= @. K * (mu_xx + mu_zz)
    @turbo expdot[colstart:colend,3] .= @. ADV + FORCING + KDIFF
    @turbo impdot[colstart:colend,3] .= @. q_v

    @turbo ADV .= @. (-u * u_x) + (-w * u_z) #UADV
    @turbo FORCING .= @. -dpdx / rho_t #UPGF
    @turbo KDIFF .= @. K * (u_xx + u_zz)
    @turbo expdot[colstart:colend,4] .= @. ADV + FORCING + KDIFF

    @turbo ADV .= @. (-u * w_x) + (-w * w_z) #WADV
    @turbo FORCING .= @.  ((-gravity * rho_p) - dpdz) / rho_t
    @turbo KDIFF .= @. K * (w_xx + w_zz)
    @turbo expdot[colstart:colend,5] .= @. ADV + FORCING + KDIFF
    impdot[colstart:colend,5] .= @. -(Pxi_bar * xi_z)

    @turbo ADV .= @. (-u * mu_l_x) + (-w * (mu_l_z + mu_lbar_z)) #Q_L ADV
    FORCING .= @. q_cond * dmudq.(mu_l, q_l)
    @turbo KDIFF .= @. K * (mu_l_xx + mu_l_zz)
    @turbo expdot[colstart:colend,6] .= @. ADV + FORCING + KDIFF

    @turbo ADV .= @. (-u * qss_x) + (-w * qss_z) #QSS ADV
    FORCING .= @. qss_cond 
    @turbo expdot[colstart:colend,7] .= @. ADV + FORCING
    @turbo impdot[colstart:colend,7] .= @. qss

    # Advance the explicit terms
    explicit_timestep(mtile, colstart, colend, t)

    # Solve for semi-implicit n+1 terms
    if mtile.model.options[:semiimplicit]
        semiimplicit_adjustment(mtile, colstart, colend, t)
    end

    # Adjust the condensation rate from the advected supersaturation
    condensation_adjustment(mtile, colstart, colend, t)

    # Increment the explicit timestep terms with other forcings
    #explicit_increment(mtile, colstart, colend, t)

end

function BF02_test_alt(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    grid = mtile.tile
    gridpoints = mtile.tilepoints
    expdot = mtile.expdot_n
    impdot = mtile.impdot_n
    model = mtile.model
    refstate = mtile.ref_state

    # Physical parameters
    K = model.physical_params[:K]

    # Gridpoints
    x = view(gridpoints,colstart:colend,1)
    z = view(gridpoints,colstart:colend,2)

    # Variables
    s = view(grid.physical,colstart:colend,1,1)
    s_x = view(grid.physical,colstart:colend,1,2)
    s_xx = view(grid.physical,colstart:colend,1,3)
    s_z = view(grid.physical,colstart:colend,1,4)
    s_zz = view(grid.physical,colstart:colend,1,5)

    xi = view(grid.physical,colstart:colend,2,1)
    xi_x = view(grid.physical,colstart:colend,2,2)
    xi_xx = view(grid.physical,colstart:colend,2,3)
    xi_z = view(grid.physical,colstart:colend,2,4)
    xi_zz = view(grid.physical,colstart:colend,2,5)

    mu = view(grid.physical,colstart:colend,3,1)
    mu_x = view(grid.physical,colstart:colend,3,2)
    mu_xx = view(grid.physical,colstart:colend,3,3)
    mu_z = view(grid.physical,colstart:colend,3,4)
    mu_zz = view(grid.physical,colstart:colend,3,5)

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

    mu_c = view(grid.physical,colstart:colend,6,1)
    mu_c_x = view(grid.physical,colstart:colend,6,2)
    mu_c_xx = view(grid.physical,colstart:colend,6,3)
    mu_c_z = view(grid.physical,colstart:colend,6,4)
    mu_c_zz = view(grid.physical,colstart:colend,6,5)

    mu_sat = view(grid.physical,colstart:colend,7,1)
    mu_sat_x = view(grid.physical,colstart:colend,7,2)
    mu_sat_xx = view(grid.physical,colstart:colend,7,3)
    mu_sat_z = view(grid.physical,colstart:colend,7,4)
    mu_sat_zz = view(grid.physical,colstart:colend,7,5)

    # Get reference state
    sbar = refstate.sbar[:,1]
    sbar_z = refstate.sbar[:,2]
    sbar_zz = refstate.sbar[:,3]

    xibar = refstate.xibar[:,1]
    xibar_z = refstate.xibar[:,2]
    xibar_zz = refstate.xibar[:,3]

    mubar = refstate.mubar[:,1]
    mubar_z = refstate.mubar[:,2]
    mubar_zz = refstate.mubar[:,3]

    # Fundamental thermodynamic quantities derived from model variables
    mu_total = mu .+ mubar
    thermo = thermodynamic_tuple.(s .+ sbar, xi .+ xibar, mu_total)
    q_v = [x[1] for x in thermo]    # Total water vapor mixing ratio
    rho_d = [x[2] for x in thermo]  # Dry air density
    Tk = [x[3] for x in thermo]     # Temperature in K
    p = [x[4] for x in thermo]      # Total air pressure
    q_c = inv_mu_transform.(mu_c)               # Cloud water mixing ratio
    q_r = 0.0
    q_l = q_c .+ q_r                # Liquid mixing ratio
    q_t = q_v .+ q_l                # Total water mixing ratio
    rho_t = rho_d .* (1.0 .+ q_v .+ q_l)   # Total air density
    #qvp = q_v .- ahyp.(mubar)       # Perturbation mixing ratio
    mu_factor = dmudq.(mu_total, q_v)
    qvp_x = mu_x ./ mu_factor # Perturbation vapor gradient in x
    qvp_z = mu_z ./ mu_factor # Perturbation vapor gradient in z
    rhobar = dry_density.(xibar) .* (1.0 .+ inv_mu_transform.(mubar)) # Ref. air density
    rho_p = rho_t .- rhobar         # Perturbation air density

    # Get the mean speed of sound squared from the reference state
    Pxi_bar = mtile.ref_state.Pxi_bar

    # Pressure gradients
    dpdx = pressure_gradient.(Tk, rho_d, q_v, s_x, xi_x, qvp_x)
    dpdz = pressure_gradient.(Tk, rho_d, q_v, s_z, xi_z, qvp_z)

    # Placeholders for intermediate calculations
    ADV = similar(s)
    FORCING = similar(s)
    KDIFF = similar(s)

    # Entropy divergence forcing
    Cm = @. (q_l * Cl)/(Cvd + (q_v * Cvv) + (q_l * Cl))
    s_div = @. Cm * (Rd + q_v * Rv) * (u_x + w_z)

    # Condensation rate
    sat_ratio = inv_mu_transform.(mu_sat) # Saturation ratio
    sat_ratio = max.(sat_ratio, 1.0e-6)
    q_sat = q_sat_liquid.(Tk, p)
    #qss = q_v .- q_sat # Supersaturation mixing ratio
    qss = (q_sat .* sat_ratio) .- q_sat
    #qss = q_sat .* (sat_ratio .- 1.0) # qss / q_sat = (q_v .- q_sat)/q_sat
    max_N_c = 100.0

    # Condensation and nucleation
    q_cond = q_condensation.(sat_ratio, Tk, p, rho_d, q_v, q_c, max_N_c)
    #q_cond = q_condensation_qss.(qss, Tk, p, rho_d, q_v, q_c, q_r, max_N_c)
    for i in 1:length(q_cond)
        if isnan(q_cond[i])
            println("q_cond is NaN at index $i")
            println("sat_ratio: $(sat_ratio[i])")
            println("q_cond: $(q_cond[i])")
            println("qss: $(qss[i])")
            println("q_sat: $(q_sat[i])")
            println("q_v: $(q_v[i])")
            #println("cloudtau: $(cloudtau[i])")
            error("NaN found at time $(t)!")
        end
    end
    
    # Rain evaporation rate not used in BF02 test
    raintau = 0.0

    # Enforce a minimum value for q_v to avoid blowup
    invq = 1.0 ./ max.(q_v, q0)
    Q_s = Q_s_factor.(Tk, p, q_v, q_l)
    sat_forcing = @. (sat_ratio*(dqsdp(Tk, p, rho_d, q_v, q_l)*((u * dpdx) + (w * (dpdz - rhobar*gravity)))) + (-q_cond* (1.0 + Q_s)))/q_sat
    #qss_cond = @. dqsdp(Tk, p, rho_d, q_v, q_l)*((u * dpdx) + (w * (dpdz - rhobar*gravity)))/q_sat - (qss * (cloudtau + raintau) * invq)
    #sat_forcing = @. dqsdp(Tk, p, rho_d, q_v, q_l)*((u * dpdx) + (w * (dpdz - rhobar*gravity)))/q_sat + (-q_cond * (1.0 + Q_s) * invq)
    #for i in 1:length(qss_cond)
    #    if isnan(qss_cond[i]) || abs(qss_cond[i]) > 1.0 || t == 4
    #        q_sat_alt = q_sat_liquid(Tk[i], p[i])
    #        println("qss_cond is $(qss_cond[i]) at index $i")
    #        println("lqss: $(lqss[i])")
    #        println("qss: $(qss[i])")
    #        println("q_sat_alt: $(q_sat_alt)")
    #        println("q_v: $(q_v[i])")
    #        println("mu_total: $(mu_total[i])")
    #        println("cloudtau: $(cloudtau[i])")
    #        println("dqsdp: $(dqsdp(Tk[i], p[i], rho_d[i], q_v[i], q_l[i]))")
    #        #error("NaN found at time $(t)!")
    #    end
    #end

    # Entropy change due to condensation
    s_cond = s_condensation.(q_cond, Tk, rho_d, q_v, q_l, p)
    for i in 1:length(s_cond)
        if isnan(s_cond[i])
            println("s_cond is NaN at index $i")
            #println("lqss: $(lqss[i])")
            println("qcond: $(q_cond[i])")
            println("q_v: $(q_v[i])")
            println("q_l: $(q_l[i])")
            println("q_sat: $(q_sat[i])")
            println("Tk: $(Tk[i])")
            println("rho_d: $(rho_d[i])")
            println("p: $(p[i])")
            error("NaN found at time $(t)!")
        end
    end

    @turbo ADV .= @. (-u * s_x) + (-w * (s_z + sbar_z)) #SADV
    FORCING .= @. s_cond + s_div
    @turbo KDIFF .= @. K * (s_xx + s_zz)
    @turbo expdot[colstart:colend,1] .= @. ADV + FORCING + KDIFF

    @turbo ADV .= @. (-u * xi_x) + (-w * (xi_z + xibar_z)) #XI ADV
    @turbo FORCING .= @. - u_x - w_z
    @turbo expdot[colstart:colend,2] .= @. ADV + FORCING
    impdot[colstart:colend,2] .= @. -w_z

    @turbo ADV .= @. (-u * mu_x) + (-w * (mu_z + mubar_z)) #MUADV
    FORCING .= @. -q_cond * mu_factor
    @turbo KDIFF .= @. K * (mu_xx + mu_zz)
    @turbo expdot[colstart:colend,3] .= @. ADV + FORCING + KDIFF
    @turbo impdot[colstart:colend,3] .= @. q_v

    @turbo ADV .= @. (-u * u_x) + (-w * u_z) #UADV
    @turbo FORCING .= @. -dpdx / rho_t #UPGF
    @turbo KDIFF .= @. K * (u_xx + u_zz)
    @turbo expdot[colstart:colend,4] .= @. ADV + FORCING + KDIFF

    @turbo ADV .= @. (-u * w_x) + (-w * w_z) #WADV
    @turbo FORCING .= @.  ((-gravity * rho_p) - dpdz) / rho_t
    @turbo KDIFF .= @. K * (w_xx + w_zz)
    @turbo expdot[colstart:colend,5] .= @. ADV + FORCING + KDIFF
    impdot[colstart:colend,5] .= @. -(Pxi_bar * xi_z)

    @turbo ADV .= @. (-u * mu_c_x) + (-w * mu_c_z) #Q_L ADV
    FORCING .= @. q_cond * dmudq.(mu_c, q_c)
    @turbo KDIFF .= @. K * (mu_c_xx + mu_c_zz)
    @turbo expdot[colstart:colend,6] .= @. ADV + FORCING + KDIFF

    @turbo ADV .= @. (-u * mu_sat_x) + (-w * mu_sat_z) #QSS ADV
    FORCING .= @. sat_forcing * dmudq.(mu_sat, sat_ratio)
    @turbo expdot[colstart:colend,7] .= @. ADV + FORCING
    @turbo impdot[colstart:colend,7] .= @. qss

    # Advance the explicit terms
    explicit_timestep(mtile, colstart, colend, t)

    # Solve for semi-implicit n+1 terms
    if mtile.model.options[:semiimplicit]
        semiimplicit_adjustment(mtile, colstart, colend, t)
    end

    # Adjust the condensation rate from the advected supersaturation
    condensation_adjustment_BF02(mtile, colstart, colend, t)

    # Increment the explicit timestep terms with other forcings
    #explicit_increment(mtile, colstart, colend, t)

end

function rainfall_test(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    grid = mtile.tile
    gridpoints = mtile.tilepoints
    expdot = mtile.expdot_n
    impdot = mtile.impdot_n
    model = mtile.model
    refstate = mtile.ref_state

    # Physical parameters
    K = model.physical_params[:K]
    alpha = model.physical_params[:alpha]
    z_damp = model.physical_params[:z_damp]

    # Gridpoints
    x = view(gridpoints,colstart:colend,1)
    z = view(gridpoints,colstart:colend,2)

    # Variables
    s = view(grid.physical,colstart:colend,1,1)
    s_x = view(grid.physical,colstart:colend,1,2)
    s_xx = view(grid.physical,colstart:colend,1,3)
    s_z = view(grid.physical,colstart:colend,1,4)
    s_zz = view(grid.physical,colstart:colend,1,5)

    xi = view(grid.physical,colstart:colend,2,1)
    xi_x = view(grid.physical,colstart:colend,2,2)
    xi_xx = view(grid.physical,colstart:colend,2,3)
    xi_z = view(grid.physical,colstart:colend,2,4)
    xi_zz = view(grid.physical,colstart:colend,2,5)

    mu = view(grid.physical,colstart:colend,3,1)
    mu_x = view(grid.physical,colstart:colend,3,2)
    mu_xx = view(grid.physical,colstart:colend,3,3)
    mu_z = view(grid.physical,colstart:colend,3,4)
    mu_zz = view(grid.physical,colstart:colend,3,5)

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

    mu_c = view(grid.physical,colstart:colend,6,1)
    mu_c_x = view(grid.physical,colstart:colend,6,2)
    mu_c_xx = view(grid.physical,colstart:colend,6,3)
    mu_c_z = view(grid.physical,colstart:colend,6,4)
    mu_c_zz = view(grid.physical,colstart:colend,6,5)

    mu_r = view(grid.physical,colstart:colend,7,1)
    mu_r_x = view(grid.physical,colstart:colend,7,2)
    mu_r_xx = view(grid.physical,colstart:colend,7,3)
    mu_r_z = view(grid.physical,colstart:colend,7,4)
    mu_r_zz = view(grid.physical,colstart:colend,7,5)

    mu_sat = view(grid.physical,colstart:colend,8,1)
    mu_sat_x = view(grid.physical,colstart:colend,8,2)
    mu_sat_xx = view(grid.physical,colstart:colend,8,3)
    mu_sat_z = view(grid.physical,colstart:colend,8,4)
    mu_sat_zz = view(grid.physical,colstart:colend,8,5)

    # Get reference state
    sbar = refstate.sbar[:,1]
    sbar_z = refstate.sbar[:,2]
    sbar_zz = refstate.sbar[:,3]

    xibar = refstate.xibar[:,1]
    xibar_z = refstate.xibar[:,2]
    xibar_zz = refstate.xibar[:,3]

    mubar = refstate.mubar[:,1]
    mubar_z = refstate.mubar[:,2]
    mubar_zz = refstate.mubar[:,3]

    # Fundamental thermodynamic quantities derived from model variables
    mu_v_total = mu .+ mubar
    thermo = thermodynamic_tuple.(s .+ sbar, xi .+ xibar, mu_v_total)
    q_v = [x[1] for x in thermo]    # Total water vapor mixing ratio
    rho_d = [x[2] for x in thermo]  # Dry air density
    Tk = [x[3] for x in thermo]     # Temperature in K
    p = [x[4] for x in thermo]      # Total air pressure
    q_c_total = inv_mu_transform.(mu_c .+ mubar)
    q_c = q_c_total .- q_v  # Condensate mixing ratio
    q_c[q_c .<= 1.0e-8] .= 0.0       # 4.1e-9 is a threshold for 1 micron drop per cm^3 at 1 kg/m^3
    q_r_total = inv_mu_transform.(mu_r .+ mubar)
    q_r = q_r_total .- q_c_total # Precipitation mixing ratio
    q_r[q_r .<= 1.0e-8] .= 0.0
    q_l = q_c .+ q_r                # Liquid mixing ratio
    q_t = q_v .+ q_l                # Total water mixing ratio
    rho_t = rho_d .* (1.0 .+ q_t)   # Total air density
    q_bar = inv_mu_transform.(mubar) # Reference mixing ratio
    #qvp = q_v .- q_bar # Perturbation mixing ratio
    mu_factor = dmudq.(mu_v_total, q_v)
    qvp_x = mu_x ./ mu_factor       # Perturbation vapor gradient in x
    qvp_z = mu_z ./ mu_factor       # Perturbation vapor gradient in z
    rhobar = dry_density.(xibar) .* (1.0 .+ q_bar) # Ref. air density
    rho_p = rho_t .- rhobar         # Perturbation air density

    # Get the mean speed of sound squared from the reference state
    Pxi_bar = mtile.ref_state.Pxi_bar

    # Pressure gradients
    dpdx = pressure_gradient.(Tk, rho_d, q_v, s_x, xi_x, qvp_x)
    dpdz = pressure_gradient.(Tk, rho_d, q_v, s_z, xi_z, qvp_z)

    # Placeholders for intermediate calculations
    ADV = similar(sbar)
    FORCING = similar(sbar)
    KDIFF = similar(sbar)

    # Entropy divergence forcing
    Cm = @. ((q_l) * Cl)/(Cvd + (q_v * Cvv) + ((q_l) * Cl))
    s_div = @. Cm * (Rd + q_v * Rv) * (u_x + w_z)

    # Condensation rate
    sat_ratio = inv_mu_transform.(mu_sat) # Saturation ratio
    q_sat = q_sat_liquid.(Tk, p)
    #sat_ratio = q_v ./ q_sat
    #sat_ratio = max.(sat_ratio, 1.0e-6)
    qss = (q_sat .* sat_ratio) .- q_sat # qss / q_sat = (q_v .- q_sat)/q_sat
    max_N_c = 100.0

    # Condensation and nucleation
    q_cond = q_condensation.(sat_ratio, Tk, p, rho_d, q_v, q_c, max_N_c)
    #q_cond = q_condensation_qss.(qss, Tk, p, rho_d, q_v, q_c, q_r, max_N_c)
    #q_cond = [x[1] for x in condensation]    # Condensation rate amount
    #cloudtau = [x[2] for x in condensation]  # Cloud condensation time scale
    for i in 1:length(q_cond) 
        if isnan(q_cond[i]) || abs(q_cond[i]) > 1.0
            println("q_cond is NaN at index $i")
            println("Tk: $(Tk[i]), P: $(p[i]), rho_d: $(rho_d[i])")
            println("q_cond: $(q_cond[i])")
            println("qss: $(qss[i])")
            println("q_sat: $(q_sat[i])")
            println("q_v: $(q_v[i])")
            println("q_c: $(q_c[i])")
            #println("cloudtau: $(cloudtau[i])")
            error("NaN found at time $(t)!")
        end
    end
    # Rain evaporation rate
    # Fixed to be >=0 so that condensation only goes to cloud droplets
    raintau = rain_evaporation.(q_r, rho_d, Tk, p)
    q_evap = -qss .* raintau 

    # Enforce a minimum value for q_v to avoid blowup
    invq = 1.0 ./ max.(q_v, eps())
    Q_s = Q_s_factor.(Tk, p, q_v, q_l)
    sat_forcing = @. (sat_ratio*(dqsdp(Tk, p, rho_d, q_v, q_l)*((u * dpdx) + (w * (dpdz - rhobar*gravity)))) + (q_evap -q_cond)*(1.0 + Q_s))/q_sat
    for i in 1:length(sat_forcing)
        if isnan(sat_forcing[i]) #|| abs(qss_cond[i]) > 10.0 #|| t == 4
            println("sat_forcing is $(sat_forcing[i]) at index $i time $(t)")
            println("sat_ratio: $(sat_ratio[i])")
            println("Tk: $(Tk[i]), P: $(p[i]), rho_d: $(rho_d[i])")
            println("qss: $(qss[i])")
            println("q_sat: $(q_sat[i])")
            println("q_v: $(q_v[i])")
            println("mu_v_total: $(mu_v_total[i])")
            println("q_c: $(q_c[i])")
            #println("cloudtau: $(cloudtau[i])")
            println("dqsdp: $(dqsdp(Tk[i], p[i], rho_d[i], q_v[i], q_l[i]))")
            error("$(sat_forcing[i]) found at time $(t)!")
        end
    end

    # Entropy change due to condensation
    s_cond = s_condensation.(q_cond, Tk, rho_d, q_v, q_l, p)
    for i in 1:length(s_cond)
        if isnan(s_cond[i])
            println("s_cond is NaN at index $i time $(t)")
            println("sat_ratio: $(sat_ratio[i])")
            println("qcond: $(q_cond[i])")
            println("q_v: $(q_v[i])")
            error("NaN found at time $(t)!")
        end
    end

    # Autoconversion rate
    q_auto = autoconversion.(q_c, rho_d) 

    # Collection rate
    q_coll = collection.(q_c, q_r, rho_d, Tk) 

    # Sedimentation rate
    Vt = sedimentation.(q_r, rho_d, Tk)

    # Calculate the flux divergence of the falling precipitation
    col = deepcopy(mtile.tile.columns[mtile.model.grid_params.vars["mu_r"]])
    col.uMish .= q_r .* Vt
    CBtransform!(col)
    CAtransform!(col)
    Vt_flux = CIxtransform(col) ./ rho_d

    # Rayleigh damping
    rayleigh_coeff = Rayleigh_damping.(alpha, z, z_damp, z[end])
    
    @turbo ADV .= @. (-u * s_x) + (-w * (s_z + sbar_z)) #SADV
    FORCING .= @. s_cond + s_div 
    @turbo KDIFF .= @. K * (s_xx + s_zz) + rayleigh_coeff * s
    @turbo expdot[colstart:colend,1] .= @. ADV + FORCING + KDIFF

    @turbo ADV .= @. (-u * xi_x) + (-w * (xi_z + xibar_z)) #XI ADV
    @turbo FORCING .= @. - u_x - w_z
    @turbo expdot[colstart:colend,2] .= @. ADV + FORCING
    impdot[colstart:colend,2] .= @. -w_z

    @turbo ADV .= @. (-u * mu_x) + (-w * (mu_z + mubar_z)) #MUADV
    FORCING .= @. (q_evap - q_cond) * mu_factor
    @turbo KDIFF .= @. K * (mu_xx + mu_zz) + rayleigh_coeff * mu
    @turbo expdot[colstart:colend,3] .= @. ADV + FORCING + KDIFF
    @turbo impdot[colstart:colend,3] .= @. q_v

    @turbo ADV .= @. (-u * u_x) + (-w * u_z) #UADV
    @turbo FORCING .= @. -dpdx / rho_t  #UPGF
    @turbo KDIFF .= @. K * (u_xx + u_zz) + rayleigh_coeff * u
    @turbo expdot[colstart:colend,4] .= @. ADV + FORCING + KDIFF

    @turbo ADV .= @. (-u * w_x) + (-w * w_z) #WADV
    @turbo FORCING .= @.  ((-gravity * rho_p) - dpdz) / rho_t
    @turbo KDIFF .= @. K * (w_xx + w_zz) + rayleigh_coeff * w
    @turbo expdot[colstart:colend,5] .= @. ADV + FORCING + KDIFF
    impdot[colstart:colend,5] .= @. -(Pxi_bar * xi_z)

    @turbo ADV .= @. (-u * mu_c_x) + (-w * (mu_c_z + mubar_z)) #Q_C ADV
    FORCING .= @. (-q_auto -q_coll) .* dmudq(mu_c, q_c_total) # Condensation forcing
    for i in 1:length(FORCING)
        if isnan(FORCING[i]) || abs(FORCING[i]) > 100.0
            println("FORCING is large at index $i, $colstart")
            println("q_cond: $(q_cond[i])")
            println("q_auto: $(q_auto[i])")
            println("q_coll: $(q_coll[i])")
            println("mu_c: $(mu_c[i])")
            println("inv_mu: $(inv_mu_transform.(mu_c[i]))")
            println("q_c: $(q_c[i])")
            println("dmudq: $(dmudq.(mu_c[i], q_c[i]))")
            println("FORCING: $(FORCING[i])")
            error("NaN found at time $(t)!")
        end
    end
    @turbo KDIFF .= @. K * (mu_c_xx + mu_c_zz) #+ rayleigh_coeff * mu_c
    @turbo expdot[colstart:colend,6] .= @. ADV + FORCING + KDIFF

    @turbo ADV .= @. (-u * mu_r_x) + (-w * (mu_r_z + mubar_z)) #Q_R ADV
    FORCING .= @. (-Vt_flux) .* dmudq(mu_r, q_r_total) # Precipitation forcing
    @turbo KDIFF .= @. K * (mu_r_xx + mu_r_zz) #+ rayleigh_coeff * mu_r
    @turbo expdot[colstart:colend,7] .= @. ADV + FORCING + KDIFF

    @turbo ADV .= @. (-u * mu_sat_x) + (-w * mu_sat_z) #QSS ADV
    FORCING .= @. sat_forcing * dmudq.(mu_sat, sat_ratio)
    @turbo expdot[colstart:colend,8] .= @. ADV + FORCING
    @turbo impdot[colstart:colend,8] .= @. qss

    # Advance the explicit terms
    explicit_timestep(mtile, colstart, colend, t)

    # Solve for semi-implicit n+1 terms
    if mtile.model.options[:semiimplicit]
        semiimplicit_adjustment(mtile, colstart, colend, t)
    end

    # Adjust the condensation rate from the advected supersaturation
    condensation_adjustment(mtile, colstart, colend, t)

    # Remove rain from the surface
    rain_adjustment(mtile, colstart, colend, t)

    # Increment the explicit timestep terms with other forcings
    #explicit_increment(mtile, colstart, colend, t)

end