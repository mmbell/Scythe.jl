"""
    LinearAdvection1D(mtile, colstart, colend, t)

1D linear advection equation with constant advection speed and diffusion.
"""
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

"""
    LinearAdvectionRZ(mtile, colstart, colend, t)

2D linear advection test in r-z cylindrical coordinates with diffusion.
"""
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

"""
    LinearAdvectionRL(mtile, colstart, colend, t)

2D linear advection test in r-lambda polar coordinates with optional diffusion.
"""
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

"""
    LinearAdvectionRLZ(mtile, colstart, colend, t)

3D linear advection test in r-lambda-z coordinates with diffusion.
"""
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

"""
    Euler_test(mtile, colstart, colend, t)

Compressible Euler equations in XZ Cartesian coordinates using entropy, log-density, and
moisture variables. Supports semi-implicit treatment of acoustic modes.
"""
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
    rhobar = dry_density.(xibar) .* (1.0 .+ inv_mu_transform.(mubar)) # Ref. air density
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

"""
    BF02_test(mtile, colstart, colend, t)

Bryan & Fritsch (2002) moist benchmark test with condensation and cloud water.
Carries liquid water in `mu_l` and an advected supersaturation mixing ratio in
`qss` (7 variables: s, xi, mu, u, w, mu_l, qss). Uses the qss-relaxation
condensation scheme ([`q_condensation_relaxation`](@ref),
[`condensation_adjustment_qss`](@ref)) restored from the formulation that
passed the moist benchmark (commit a4bf2a0), ported to the linear `mu_transform`
moisture convention. Validated by `benchmarks/bf02_moist.jl`.
"""
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

    # The reference liquid water profile is zero, so mu_l is the full
    # liquid water variable

    # Fundamental thermodynamic quantities derived from model variables
    mu_total = mu .+ mubar
    thermo = thermodynamic_tuple.(s .+ sbar, xi .+ xibar, mu_total)
    q_v = [x[1] for x in thermo]    # Total water vapor mixing ratio
    rho_d = [x[2] for x in thermo]  # Dry air density
    Tk = [x[3] for x in thermo]     # Temperature in K
    p = [x[4] for x in thermo]      # Total air pressure
    q_l = inv_mu_transform.(mu_l)              # Liquid mixing ratio
    rho_t = rho_d .* (1.0 .+ q_v .+ q_l)   # Total air density
    qvp = q_v .- inv_mu_transform.(mubar)       # Perturbation mixing ratio
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
    N_c = 500.0
    r_c = 10.0
    q_cond = q_condensation_relaxation.(qss, Tk, p, q_v, q_l, N_c, r_c)
    s_cond = s_condensation_relaxation.(q_cond, Tk, rho_d, q_v, q_l, p)
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

    @turbo ADV .= @. (-u * mu_l_x) + (-w * mu_l_z) #Q_L ADV
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
    condensation_adjustment_qss(mtile, colstart, colend, t)

    # Increment the explicit timestep terms with other forcings
    #explicit_increment(mtile, colstart, colend, t)

end

"""
    BF02_test_alt(mtile, colstart, colend, t)

Alternative Bryan & Fritsch (2002) formulation that tracks saturation ratio instead of
supersaturation mixing ratio. Includes condensation adjustment.
"""
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

"""
    rainfall_test(mtile, colstart, colend, t)

Full microphysics test in XZ with cloud water, rain, condensation, autoconversion,
collection, sedimentation, and Rayleigh damping. Supports semi-implicit acoustic modes.
"""
function rainfall_test(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    grid = mtile.tile
    gridpoints = mtile.tilepoints
    expdot = mtile.expdot_n
    impdot = mtile.impdot_n
    model = mtile.model
    refstate = mtile.ref_state

    # Physical parameters
    Khdiff = model.physical_params[:Khdiff]
    Kvdiff = model.physical_params[:Kvdiff]
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

    sat_ratio = view(grid.physical,colstart:colend,8,1)
    sat_ratio_x = view(grid.physical,colstart:colend,8,2)
    sat_ratio_xx = view(grid.physical,colstart:colend,8,3)
    sat_ratio_z = view(grid.physical,colstart:colend,8,4)
    sat_ratio_zz = view(grid.physical,colstart:colend,8,5)

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

    satbar = refstate.satbar[:,1]
    satbar_z = refstate.satbar[:,2]
    satbar_zz = refstate.satbar[:,3]

    # Fundamental thermodynamic quantities derived from model variables
    mu_v_total = mu .+ mubar
    thermo = thermodynamic_tuple.(s .+ sbar, xi .+ xibar, mu_v_total)
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
    mu_r_factor = dmudq.(mu_r, q_r_total)
    q_r_z = (mu_r_z ./ mu_r_factor) #.- (mu_c_z ./ mu_c_factor) # Perturbation rain gradient in z
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
    VDIFF = similar(sbar)

    # Entropy divergence forcing
    Cm = @. ((q_l) * Cl)/(Cvd + (q_v * Cvv) + ((q_l) * Cl))
    s_div = @. Cm * (Rd + q_v * Rv) * (u_x + w_z)

    # Condensation rate
    #sat_ratio = inv_mu_transform.(sat_ratio .+ satbar) # Saturation ratio
    #mu_sat_factor = dmudq.(sat_ratio, sat_ratio)
    q_sat = q_sat_liquid.(Tk, p)
    #sat_ratio = q_v ./ q_sat # Saturation ratio
    sat_ratio_adj = sat_ratio #max.(sat_ratio, 1.0e-6)
    qss = (q_sat .* sat_ratio_adj) .- q_sat # qss / q_sat = (q_v .- q_sat)/q_sat
    max_N_c = 100.0

    # Condensation and nucleation
    q_cond = q_condensation.(sat_ratio_adj, Tk, p, rho_d, q_v, q_c, max_N_c)
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
    mean_r = 250.0 # Mean radius of rain drops in microns
    q_evap = 0.0 #q_evaporation.(sat_ratio_adj, Tk, p, rho_d, q_v, q_r, mean_r)
    for i in 1:length(q_evap)
        if isnan(q_evap[i]) || abs(q_evap[i]) > 1.0
            println("q_evap is NaN at index $i")
            println("Tk: $(Tk[i]), P: $(p[i]), rho_d: $(rho_d[i])")
            println("q_evap: $(q_evap[i])")
            println("q_v: $(q_v[i])")
            println("q_r: $(q_r[i])")
            error("NaN found at time $(t)!")
        end
    end
    # Enforce a minimum value for q_v to avoid blowup
    invq = 1.0 ./ max.(q_v, eps())
    Q_s = Q_s_factor.(Tk, p, q_v, q_l)
    sat_forcing = @. (sat_ratio_adj*(dqsdp(Tk, p, rho_d, q_v, q_l)*((u * dpdx) + (w * (dpdz - rhobar*gravity)))) + (q_evap -q_cond)*(1.0 + Q_s))/q_sat
    for i in 1:length(sat_forcing)
        if isnan(sat_forcing[i]) #|| abs(qss_cond[i]) > 10.0 #|| t == 4
            println("sat_forcing is $(sat_forcing[i]) at index $i time $(t)")
            println("sat_ratio_adj: $(sat_ratio_adj[i])")
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
    q_auto = 0.0 #autoconversion.(q_c, rho_d) 

    # Collection rate
    q_coll = 0.0 #collection.(q_c, q_r, rho_d, Tk) 

    # Sedimentation rate
    Vt = 0.0 #sedimentation.(q_r, rho_d, Tk)

    # Calculate the flux divergence of the falling precipitation

    col = deepcopy(mtile.tile.kbasis.data[mtile.model.grid_params.vars["mu_r"]])
    col.uMish .= Vt
    #col.uMish .= q_r .* rho_d .* Vt
    Btransform!(col)
    Atransform!(col)
    Vt .= Itransform!(col)
    dVtdz = Ixtransform(col) #./ rho_d
    #Vt_flux = Ixtransform(col) ./ rho_d
    Vt_flux = (q_r_z .* Vt) .+ (q_r .* dVtdz) .+ (q_r .* Vt .* xi_z) # Precipitation flux divergence
    #Vt_flux = precipitation_flux.(q_r, rho_d, Tk, q_r_z, xi_z)

    # Rayleigh damping
    rayleigh_coeff = Rayleigh_damping.(alpha, z, z_damp, z[end])

    # Calculate the vertical diffusivity
    # Mixing length based on Louis parameterization
    Sv = sqrt.((u_z .* u_z))
    lv = 1.0 ./ ((1.0 ./ (0.4 .* z)) .+ (1.0 ./ 80.0))
    Kv = 0.0 #(lv.^2) .* Sv

    # Smagorsinski length scale
    #lh = 100.0
    #Sh = sqrt.(2.0 .* ((u_x .* u_x) .+ (w_z .* w_z)))
    #Kh = (lh.^2) .* Sh

    #col = deepcopy(mtile.tile.columns[mtile.model.grid_params.vars["mu_r"]])
    col.uMish .= rho_d .* Kv .* s_z
    Btransform!(col)
    Atransform!(col)
    VDIFF .= (Ixtransform(col)) ./ rho_d

    @turbo ADV .= @. (-u * s_x) + (-w * (s_z + sbar_z)) #SADV
    FORCING .= @. s_cond + s_div 
    @turbo KDIFF .= @. Khdiff * s_xx + rayleigh_coeff * s
    @turbo expdot[colstart:colend,1] .= @. ADV + FORCING + KDIFF #+ VDIFF
    impdot[colstart:colend,1] .= @. Kvdiff * s_zz

    @turbo ADV .= @. (-u * xi_x) + (-w * (xi_z + xibar_z)) #XI ADV
    @turbo FORCING .= @. - u_x - w_z
    @turbo expdot[colstart:colend,2] .= @. ADV + FORCING
    impdot[colstart:colend,2] .= @. -w_z

    # Differentiate Kv * du/dz
    col.uMish .= rho_d .* Kv .* (mu_z)
    Btransform!(col)
    Atransform!(col)
    VDIFF .= (Ixtransform(col)) ./ rho_d

    @turbo ADV .= @. (-u * mu_x) + (-w * (mu_z + mubar_z)) #MUADV
    FORCING .= @. (q_evap - q_cond) * mu_factor
    @turbo KDIFF .= @. Khdiff * mu_xx + rayleigh_coeff * mu
    @turbo expdot[colstart:colend,3] .= @. ADV + FORCING + KDIFF #+ VDIFF
    impdot[colstart:colend,3] .= @. Kvdiff * mu_zz

    # Differentiate u turbulent flux
    col.uMish .= rho_d .* Kv .* u_z
    Btransform!(col)
    Atransform!(col)
    VDIFF .= (Ixtransform(col)) ./ rho_d

    @turbo ADV .= @. (-u * u_x) + (-w * u_z) #UADV
    @turbo FORCING .= @. -dpdx / rho_t  #UPGF
    @turbo KDIFF .= @. Khdiff * u_xx + rayleigh_coeff * u
    @turbo expdot[colstart:colend,4] .= @. ADV + FORCING + KDIFF #+ VDIFF
    impdot[colstart:colend,4] .= @. Kvdiff * u_zz

    # Differentiate w turbulent flux
    col.uMish .= rho_d .* Kv .* w_z
    Btransform!(col)
    Atransform!(col)
    VDIFF .= (Ixtransform(col)) ./ rho_d

    @turbo ADV .= @. (-u * w_x) + (-w * w_z) #WADV
    @turbo FORCING .= @.  ((-gravity * rho_p) - dpdz) / rho_t
    @turbo KDIFF .= @. Khdiff * w_xx + rayleigh_coeff * w
    @turbo expdot[colstart:colend,5] .= @. ADV + FORCING + KDIFF #+ VDIFF
    impdot[colstart:colend,5] .= @. -(Pxi_bar * xi_z)

    # Differentiate Kv * du/dz
    col.uMish .= rho_d .* Kv .* (mu_c_z)
    Btransform!(col)
    Atransform!(col)
    VDIFF .= (Ixtransform(col)) ./ rho_d

    @turbo ADV .= @. (-u * mu_c_x) + (-w * (mu_c_z + mubar_z)) #Q_C ADV
    FORCING .= @. (q_cond -q_auto -q_coll) * mu_c_factor # Condensation forcing
    for i in 1:length(FORCING)
        if isnan(FORCING[i]) #|| abs(FORCING[i]) > 100.0
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
    @turbo KDIFF .= @. Khdiff * mu_c_xx + rayleigh_coeff * mu_c
    @turbo expdot[colstart:colend,6] .= @. ADV + FORCING + KDIFF #+ VDIFF
    impdot[colstart:colend,6] .= @. Kvdiff * mu_c_zz

    # Differentiate Kv * du/dz
    col.uMish .= rho_d .* Kv .* (mu_r_z)
    Btransform!(col)
    Atransform!(col)
    VDIFF .= (Ixtransform(col)) ./ rho_d

    @turbo ADV .= @. (-u * mu_r_x) + (-w * (mu_r_z + mubar_z)) #Q_R ADV
    FORCING .= @. (q_auto +q_coll -q_evap -Vt_flux) * mu_r_factor # Precipitation forcing
    @turbo KDIFF .= @. Khdiff * mu_r_xx + Kvdiff * mu_r_zz + rayleigh_coeff * mu_r
    @turbo expdot[colstart:colend,7] .= @. 0.0 #ADV + FORCING + KDIFF + VDIFF
    impdot[colstart:colend,7] .= @. Kvdiff * mu_r_zz

    # Differentiate Kv * du/dz
    #col.uMish .= rho_d .* Kv .* (sat_ratio_z)
    #Btransform!(col)
    #Atransform!(col)
    #VDIFF .= (Ixtransform(col)) ./ rho_d

    @turbo ADV .= @. (-u * sat_ratio_x) + (-w * sat_ratio_z) #QSS ADV
    FORCING .= @. sat_forcing #* mu_sat_factor
    @turbo KDIFF .= @. Khdiff * sat_ratio_xx + rayleigh_coeff * sat_ratio
    @turbo expdot[colstart:colend,8] .= @. ADV + FORCING + KDIFF #+ VDIFF
    @turbo impdot[colstart:colend,8] .= @. Kvdiff * sat_ratio_zz 

    # Advance the explicit terms
    explicit_timestep(mtile, colstart, colend, t)

    # Solve for semi-implicit n+1 terms
    if mtile.model.options[:semiimplicit]
        semiimplicit_adjustment(mtile, colstart, colend, t)
    end

    # Adjust the condensation rate from the advected supersaturation
    condensation_adjustment(mtile, colstart, colend, t)

    # Use implicit timestep for diffusion
    #diffusion_timestep(mtile, colstart, colend, t)

    # Remove rain from the surface
    #rain_adjustment(mtile, colstart, colend, t)

    # Increment the explicit timestep terms with other forcings
    #explicit_increment(mtile, colstart, colend, t)

end