function Euler_test(mtile::ModelTile, colstart::Int64, colend::Int64)
    
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
    qvp = q_v .- ahyp.(mubar)       # Perturbation mixing ratio
    qvp_x = mu_x ./ dmudq.(mu, q_v) # Perturbation vapor gradient in x
    qvp_z = mu_z ./ dmudq.(mu, q_v) # Perturbation vapor gradient in z
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
    PGF .= @.  -(g * rho_p / rho_t) - (pressure_gradient(Tk, rho_d, q_v, s_z, xi_z, qvp_z) / rho_t)
    @turbo KDIFF .= @. K * (w_xx + w_zz)
    @turbo expdot[colstart:colend,5] .= @. ADV + PGF + KDIFF
    impdot[colstart:colend,5] .= @. -(Pxi_bar * xi_z)

end

