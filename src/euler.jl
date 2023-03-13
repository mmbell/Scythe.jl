function Euler_test(mtile::ModelTile)
    
    grid = mtile.tile
    gridpoints = mtile.tilepoints
    expdot = mtile.expdot_n
    impdot = mtile.impdot_n
    model = mtile.model
    refstate = mtile.ref_state

    # Physical parameters
    K = model.physical_params[:K]

    # Gridpoints
    x = view(gridpoints,:,1)
    z = view(gridpoints,:,2)    

    # Variables
    s = view(grid.physical,:,1,1)
    s_x = view(grid.physical,:,1,2)
    s_xx = view(grid.physical,:,1,3)
    s_z = view(grid.physical,:,1,4)
    s_zz = view(grid.physical,:,1,5)
    
    xi = view(grid.physical,:,2,1)
    xi_x = view(grid.physical,:,2,2)
    xi_xx = view(grid.physical,:,2,3)
    xi_z = view(grid.physical,:,2,4)
    xi_zz = view(grid.physical,:,2,5)
    
    mu = view(grid.physical,:,3,1)
    mu_x = view(grid.physical,:,3,2)
    mu_xx = view(grid.physical,:,3,3)
    mu_z = view(grid.physical,:,3,4)
    mu_zz = view(grid.physical,:,3,5)
    
    u = view(grid.physical,:,4,1)
    u_x = view(grid.physical,:,4,2)
    u_xx = view(grid.physical,:,4,3)
    u_z = view(grid.physical,:,4,4)
    u_zz = view(grid.physical,:,4,5)

    w = view(grid.physical,:,5,1)
    w_x = view(grid.physical,:,5,2)
    w_xx = view(grid.physical,:,5,3)
    w_z = view(grid.physical,:,5,4)
    w_zz = view(grid.physical,:,5,5)
    
    # Get reference state (need to better optimize this memory allocation
    num_columns = Int64(size(gridpoints,1) / model.grid_params.zDim)
    sbar = repeat(refstate.sbar[:,1],num_columns)
    sbar_z = repeat(refstate.sbar[:,2],num_columns)
    sbar_zz = repeat(refstate.sbar[:,3],num_columns)

    xibar = repeat(refstate.xibar[:,1],num_columns)
    xibar_z = repeat(refstate.xibar[:,2],num_columns)
    xibar_zz = repeat(refstate.xibar[:,3],num_columns)

    mubar = repeat(refstate.mubar[:,1],num_columns)
    mubar_z = repeat(refstate.mubar[:,2],num_columns)
    mubar_zz = repeat(refstate.mubar[:,3],num_columns)    
    
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
    
    # Placeholders for intermediate calculations
    ADV = similar(s)
    PGF = similar(s)
    KDIFF = similar(s)

    @turbo ADV .= @. (-u * s_x) + (-w * (s_z + sbar_z)) #SADV
    #No PGF
    @turbo KDIFF .= @. K * (s_xx + s_zz)
    @turbo expdot[:,1] .= @. ADV + KDIFF
    
    @turbo ADV .= @. (-u * xi_x) + (-w * (xi_z + xibar_z)) #XI ADV
    @turbo PGF .= @. -u_x - w_z
    # No mass diffusion
    @turbo expdot[:,2] .= @. ADV + PGF

    @turbo ADV .= @. (-u * mu_x) + (-w * (mu_z + mubar_z)) #SADV
    #No PGF
    @turbo KDIFF .= @. K * (mu_xx + mu_zz)
    @turbo expdot[:,3] .= @. ADV + KDIFF
    
    @turbo ADV .= @. (-u * u_x) + (-w * u_z) #UADV
    PGF .= @. -(pressure_gradient(Tk, rho_d, q_v, s_x, xi_x, qvp_x) / rho_t) #UPGF
    @turbo KDIFF .= @. K * (u_xx + u_zz)
    @turbo expdot[:,4] .= @. ADV + PGF + KDIFF

    @turbo ADV .= @. (-u * w_x) + (-w * w_z) #WADV
    PGF .= @. -(pressure_gradient(Tk, rho_d, q_v, s_z, xi_z, qvp_z) / rho_t) -
         (g * rho_p / rho_t) #WPGF
    @turbo KDIFF .= @. K * (w_xx + w_zz)
    @turbo expdot[:,5] .= @. ADV + PGF + KDIFF
    
end

