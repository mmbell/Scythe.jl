"""
    primitive_equation_XZ(mtile, colstart, colend, t)

Primitive equations in XZ Cartesian coordinates with full microphysics including
condensation, autoconversion, collection, sedimentation, and turbulent mixing.
Supports semi-implicit acoustic modes and implicit vertical diffusion.
"""
function primitive_equation_XZ(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    grid = mtile.tile
    gridpoints = mtile.tilepoints
    expdot = mtile.expdot_n
    impdot = mtile.impdot_n
    model = mtile.model
    refstate = mtile.ref_state

    # Physical parameters
    Khdiff = model.physical_params[:Khdiff]
    Kvdiff = model.physical_params[:Kvdiff]
    Kv_mudiff = model.physical_params[:Kv_mudiff]
    alpha = model.physical_params[:alpha]
    z_damp = model.physical_params[:z_damp]

    # Reduced form for reversible benchmarks (e.g. Bryan & Fritsch 2002,
    # where precipitation fallout is not allowed): disable autoconversion,
    # collection, sedimentation, and rain evaporation
    precipitation = get(model.options, :precipitation, true)

    # Louis shear-based vertical mixing. Disable for benchmark cases whose
    # specification has no turbulence; the explicit mixing is also unstable
    # where strong shear meets the fine Chebyshev spacing at the boundaries
    # (Kv ~ lv^2 |du/dz| can exceed the explicit diffusion limit there)
    vertical_mixing = get(model.options, :vertical_mixing, true)

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

    satbar = refstate.satbar[:,1]
    satbar_z = refstate.satbar[:,2]
    satbar_zz = refstate.satbar[:,3]

    # Fundamental thermodynamic quantities derived from model variables
    mu_total = mu .+ mubar
    thermo = thermodynamic_tuple.(s .+ sbar, xi .+ xibar, mu_total)
    q_v = [x[1] for x in thermo]    # Total water vapor mixing ratio
    rho_d = [x[2] for x in thermo]  # Dry air density
    Tk = [x[3] for x in thermo]     # Temperature in K
    p = [x[4] for x in thermo]      # Total air pressure
    q_c = inv_mu_transform.(mu_c)               # Cloud water mixing ratio
    q_r = inv_mu_transform.(mu_r)             # Rain water mixing ratio
    q_l = q_c .+ q_r                # Liquid mixing ratio
    q_t = q_v .+ q_l                # Total water mixing ratio
    rho_t = rho_d .* (1.0 .+ q_v .+ q_l)   # Total air density
    mu_c_factor = dmudq.(mu_c, q_c) # Factor for perturbation cloud mixing ratio
    mu_r_factor = dmudq.(mu_r, q_r) # Factor for perturbation rain mixing ratio
    q_r_z = (mu_r_z ./ mu_r_factor)
    #qvp = q_v .- ahyp.(mubar)       # Perturbation mixing ratio
    mu_v_factor = dmudq.(mu_total, q_v)
    qvp_x = mu_x ./ mu_v_factor # Perturbation vapor gradient in x
    qvp_z = mu_z ./ mu_v_factor # Perturbation vapor gradient in z
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
    VDIFF = similar(s)

    # Entropy divergence forcing
    Cm = @. (q_l * Cl)/(Cvd + (q_v * Cvv) + (q_l * Cl))
    s_div = @. Cm * (Rd + q_v * Rv) * (u_x + w_z)

    # Condensation rate
    sat_ratio = inv_mu_transform.(mu_sat .+ satbar) # Saturation ratio
    #sat_ratio = max.(sat_ratio, 1.0e-6)
    q_sat = q_sat_liquid.(Tk, p)
    qss = (q_sat .* sat_ratio) .- q_sat
    max_N_c = 100.0

    # Condensation and nucleation
    q_cond = q_condensation.(sat_ratio, Tk, p, rho_d, q_v, q_c, max_N_c)
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
    
    # Rain evaporation rate
    mean_r = 250.0 # Mean radius of rain drops in microns
    q_evap = precipitation ? q_evaporation.(sat_ratio, Tk, p, rho_d, q_v, q_r, mean_r) :
                             zero(q_v)
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

    # Entropy change due to condensation and evaporation
    s_cond = s_condensation.(q_evap, q_cond, Tk, rho_d, q_v, q_l, p)
    for i in 1:length(s_cond)
        if isnan(s_cond[i])
            println("s_cond is NaN at index $i")
            println("q_evap: $(q_evap[i])")
            println("q_cond: $(q_cond[i])")
            println("q_v: $(q_v[i])")
            println("q_l: $(q_l[i])")
            println("q_sat: $(q_sat[i])")
            println("Tk: $(Tk[i])")
            println("rho_d: $(rho_d[i])")
            println("p: $(p[i])")
            error("NaN found at time $(t)!")
        end
    end

    # Rayleigh damping
    rayleigh_coeff = Rayleigh_damping.(alpha, z, z_damp, z[end])

    if precipitation
        # Autoconversion rate
        q_auto = autoconversion.(q_c, rho_d)

        # Collection rate
        q_coll = collection.(q_c, q_r, rho_d, Tk)

        # Sedimentation rate
        Vt = sedimentation.(q_r, rho_d, Tk)

        # Calculate the flux divergence of the falling precipitation
        col = deepcopy(mtile.tile.kbasis.data[mtile.model.grid_params.vars["mu_r"]])
        col.uMish .= Vt
        Btransform!(col)
        Atransform!(col)
        Vt .= Itransform!(col)
        dVtdz = Ixtransform(col)
        Vt_flux = (q_r_z .* Vt) .+ (q_r .* dVtdz) .+ (q_r .* Vt .* xi_z) # Precipitation flux divergence
    else
        q_auto = zero(q_v)
        q_coll = zero(q_v)
        Vt_flux = zero(q_v)
    end

    # Calculate the vertical diffusivity
    col = deepcopy(mtile.tile.kbasis.data[mtile.model.grid_params.vars["mu"]])

    # Vertical mixing length based on Louis parameterization
    Sv = sqrt.(u_z.^2)
    lv = 1.0 ./ ((1.0 ./ (0.4 .* z)) .+ (1.0 ./ 80.0))
    Kv = vertical_mixing ? (lv.^2) .* Sv : zero(Sv)

    # Calculate vapor mixing first since it is needed for entropy and saturation ratio
    col.uMish .= rho_d .* Kv .* (mu_z .+ mubar_z)
    Btransform!(col)
    Atransform!(col)
    VDIFF .= Ixtransform(col) ./ rho_d
    q_flux = VDIFF ./ mu_v_factor

    @turbo ADV .= @. (-u * mu_x) + (-w * (mu_z + mubar_z)) #MUADV
    FORCING .= @. (q_evap -q_cond) * mu_v_factor
    @turbo KDIFF .= @. Khdiff * mu_xx
    @turbo expdot[colstart:colend,3] .= @. ADV + FORCING + KDIFF + VDIFF
    @turbo impdot[colstart:colend,3] .= @. Kvdiff * mu_zz

    # Apply the subgrid vapor flux to the entropy
    s_v_mix = s_vapor_mixing.(q_flux, Tk, rho_d, q_v)

    # Entropy mixing
    col.uMish .= rho_d .* Kv .* (s_z .+ sbar_z)
    Btransform!(col)
    Atransform!(col)
    VDIFF .= (Ixtransform(col)) ./ rho_d

    @turbo ADV .= @. (-u * s_x) + (-w * (s_z + sbar_z)) #SADV
    FORCING .= @. s_cond + s_div + s_v_mix
    @turbo KDIFF .= @. Khdiff * s_xx
    @turbo expdot[colstart:colend,1] .= @. ADV + FORCING + KDIFF + VDIFF
    @turbo impdot[colstart:colend,1] .= @. Kvdiff * s_zz

    @turbo ADV .= @. (-u * xi_x) + (-w * (xi_z + xibar_z)) #XI ADV
    @turbo FORCING .= @. - u_x - w_z
    @turbo expdot[colstart:colend,2] .= @. ADV + FORCING
    impdot[colstart:colend,2] .= @. -w_z

    col.uMish .= rho_d .* Kv .* u_z
    Btransform!(col)
    Atransform!(col)
    VDIFF .= (Ixtransform(col)) ./ rho_d

    @turbo ADV .= @. (-u * u_x) + (-w * u_z) #UADV
    @turbo FORCING .= @. -dpdx / rho_t #UPGF
    @turbo KDIFF .= @. Khdiff * u_xx 
    @turbo expdot[colstart:colend,4] .= @. ADV + FORCING + KDIFF + VDIFF
    @turbo impdot[colstart:colend,4] .= @. Kvdiff * u_zz

    @turbo ADV .= @. (-u * w_x) + (-w * w_z) #WADV
    @turbo FORCING .= @.  ((-gravity * rho_p) - dpdz) / rho_t
    @turbo KDIFF .= @. Khdiff * w_xx
    @turbo expdot[colstart:colend,5] .= @. ADV + FORCING + KDIFF
    impdot[colstart:colend,5] .= @. -(Pxi_bar * xi_z)

    col.uMish .= rho_d .* Kv .* mu_c_z
    Btransform!(col)
    Atransform!(col)
    VDIFF .= (Ixtransform(col)) ./ rho_d

    @turbo ADV .= @. (-u * mu_c_x) + (-w * mu_c_z) #Q_C ADV
    FORCING .= @. (q_cond -q_auto -q_coll) * mu_c_factor
    @turbo KDIFF .= @. Khdiff * mu_c_xx
    @turbo expdot[colstart:colend,6] .= @. ADV + FORCING + KDIFF + VDIFF
    @turbo impdot[colstart:colend,6] .= @. Kv_mudiff * mu_c_zz

    col.uMish .= rho_d .* Kv .* mu_r_z
    Btransform!(col)
    Atransform!(col)
    VDIFF .= (Ixtransform(col)) ./ rho_d

    @turbo ADV .= @. (-u * mu_r_x) + (-w * mu_r_z) #Q_R ADV
    FORCING .= @. (q_auto +q_coll -q_evap -Vt_flux) * mu_r_factor
    @turbo KDIFF .= @. Khdiff * mu_r_xx
    @turbo expdot[colstart:colend,7] .= @. ADV + FORCING + KDIFF + VDIFF
    @turbo impdot[colstart:colend,7] .= @. Kv_mudiff * mu_r_zz

    col.uMish .= rho_d .* Kv .* (mu_sat_z + satbar_z)
    Btransform!(col)
    Atransform!(col)
    VDIFF .= (Ixtransform(col)) ./ rho_d

    # Saturation forcing
    Q_s = Q_s_factor.(Tk, p, q_v, q_l)
    sat_forcing = @. (sat_ratio*(dqsdp(Tk, p, rho_d, q_v, q_l)*((u * dpdx) + (w * (dpdz - rhobar*gravity)))) + q_flux + ((q_evap - q_cond) * (1.0 + Q_s)))/q_sat

    @turbo ADV .= @. (-u * mu_sat_x) + (-w * (mu_sat_z + satbar_z)) #QSS ADV
    FORCING .= @. sat_forcing * dmudq.(mu_sat, sat_ratio)
    @turbo KDIFF .= @. Khdiff * mu_sat_xx
    @turbo expdot[colstart:colend,8] .= @. ADV + FORCING + KDIFF + VDIFF
    @turbo impdot[colstart:colend,8] .= @. Kv_mudiff * mu_sat_zz

    # Advance the explicit terms
    explicit_timestep(mtile, colstart, colend, t)

    # Solve for semi-implicit n+1 terms
    if mtile.model.options[:semiimplicit]
        semiimplicit_adjustment(mtile, colstart, colend, t)
    end

    # Adjust the condensation rate from the advected supersaturation
    condensation_adjustment_new(mtile, colstart, colend, t)

    # Use implicit timestep for diffusion
    diffusion_timestep(mtile, colstart, colend, t)

    # Increment the explicit timestep terms with other forcings
    #explicit_increment(mtile, colstart, colend, t)

end

"""
    primitive_equation_RZ(mtile, colstart, colend, t)

Primitive equations in axisymmetric r-z cylindrical coordinates with full microphysics,
surface fluxes (drag, enthalpy), Coriolis force, and turbulent mixing.
Supports semi-implicit acoustic modes and implicit vertical diffusion.
"""
function primitive_equation_RZ(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    grid = mtile.tile
    gridpoints = mtile.tilepoints
    expdot = mtile.expdot_n
    impdot = mtile.impdot_n
    model = mtile.model
    refstate = mtile.ref_state

    # Physical parameters
    Khdiff = model.physical_params[:Khdiff]
    Kvdiff = model.physical_params[:Kvdiff]
    Kv_mudiff = model.physical_params[:Kv_mudiff]
    alpha = model.physical_params[:alpha]
    z_damp = model.physical_params[:z_damp]
    Cd = model.physical_params[:Cd]
    Ck = model.physical_params[:Ck]
    f = model.physical_params[:f]
    z_ref_level = model.physical_params[:z_ref_level]
    sst = model.physical_params[:sst]

    # Gridpoints
    r = view(gridpoints,colstart:colend,1)
    z = view(gridpoints,colstart:colend,2)

    # Variables
    s = view(grid.physical,colstart:colend,1,1)
    s_r = view(grid.physical,colstart:colend,1,2)
    s_rr = view(grid.physical,colstart:colend,1,3)
    s_z = view(grid.physical,colstart:colend,1,4)
    s_zz = view(grid.physical,colstart:colend,1,5)

    xi = view(grid.physical,colstart:colend,2,1)
    xi_r = view(grid.physical,colstart:colend,2,2)
    xi_rr = view(grid.physical,colstart:colend,2,3)
    xi_z = view(grid.physical,colstart:colend,2,4)
    xi_zz = view(grid.physical,colstart:colend,2,5)

    mu_v = view(grid.physical,colstart:colend,3,1)
    mu_v_r = view(grid.physical,colstart:colend,3,2)
    mu_v_rr = view(grid.physical,colstart:colend,3,3)
    mu_v_z = view(grid.physical,colstart:colend,3,4)
    mu_v_zz = view(grid.physical,colstart:colend,3,5)

    u = view(grid.physical,colstart:colend,4,1)
    u_r = view(grid.physical,colstart:colend,4,2)
    u_rr = view(grid.physical,colstart:colend,4,3)
    u_z = view(grid.physical,colstart:colend,4,4)
    u_zz = view(grid.physical,colstart:colend,4,5)

    v = view(grid.physical,colstart:colend,5,1)
    v_r = view(grid.physical,colstart:colend,5,2)
    v_rr = view(grid.physical,colstart:colend,5,3)
    v_z = view(grid.physical,colstart:colend,5,4)
    v_zz = view(grid.physical,colstart:colend,5,5)

    w = view(grid.physical,colstart:colend,6,1)
    w_r = view(grid.physical,colstart:colend,6,2)
    w_rr = view(grid.physical,colstart:colend,6,3)
    w_z = view(grid.physical,colstart:colend,6,4)
    w_zz = view(grid.physical,colstart:colend,6,5)

    mu_c = view(grid.physical,colstart:colend,7,1)
    mu_c_r = view(grid.physical,colstart:colend,7,2)
    mu_c_rr = view(grid.physical,colstart:colend,7,3)
    mu_c_z = view(grid.physical,colstart:colend,7,4)
    mu_c_zz = view(grid.physical,colstart:colend,7,5)

    mu_r = view(grid.physical,colstart:colend,8,1)
    mu_r_r = view(grid.physical,colstart:colend,8,2)
    mu_r_rr = view(grid.physical,colstart:colend,8,3)
    mu_r_z = view(grid.physical,colstart:colend,8,4)
    mu_r_zz = view(grid.physical,colstart:colend,8,5)

    mu_sat = view(grid.physical,colstart:colend,9,1)
    mu_sat_r = view(grid.physical,colstart:colend,9,2)
    mu_sat_rr = view(grid.physical,colstart:colend,9,3)
    mu_sat_z = view(grid.physical,colstart:colend,9,4)
    mu_sat_zz = view(grid.physical,colstart:colend,9,5)

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
    mu_v_total = mu_v .+ mubar
    thermo = thermodynamic_tuple.(s .+ sbar, xi .+ xibar, mu_v_total)
    q_v = [x[1] for x in thermo]    # Total water vapor mixing ratio
    rho_d = [x[2] for x in thermo]  # Dry air density
    Tk = [x[3] for x in thermo]     # Temperature in K
    p = [x[4] for x in thermo]      # Total air pressure
    q_c = inv_mu_transform.(mu_c)               # Cloud water mixing ratio
    q_r = inv_mu_transform.(mu_r)             # Rain water mixing ratio
    q_l = q_c .+ q_r                # Liquid mixing ratio
    q_t = q_v .+ q_l                # Total water mixing ratio
    rho_t = rho_d .* (1.0 .+ q_v .+ q_l)   # Total air density
    mu_c_factor = dmudq.(mu_c, q_c) # Factor for perturbation cloud mixing ratio
    mu_r_factor = dmudq.(mu_r, q_r) # Factor for perturbation rain mixing ratio
    q_r_z = (mu_r_z ./ mu_r_factor)
    #qvp = q_v .- ahyp.(mubar)       # Perturbation mixing ratio
    mu_v_factor = dmudq.(mu_v_total, q_v)
    qvp_r = mu_v_r ./ mu_v_factor # Perturbation vapor gradient in r
    qvp_z = mu_v_z ./ mu_v_factor # Perturbation vapor gradient in z
    rhobar = dry_density.(xibar) .* (1.0 .+ inv_mu_transform.(mubar)) # Ref. air density
    rho_p = rho_t .- rhobar         # Perturbation air density

    # Get the mean speed of sound squared from the reference state
    Pxi_bar = mtile.ref_state.Pxi_bar

    # Pressure gradients
    dpdr = pressure_gradient.(Tk, rho_d, q_v, s_r, xi_r, qvp_r)
    dpdz = pressure_gradient.(Tk, rho_d, q_v, s_z, xi_z, qvp_z)

    # Get relevant surface variables
    q_sfc = q_sat_liquid(sst, p[1])
    # Use the near surface density and water vapor so that this is sensible heat flux
    s_sfc = entropy(sst, rho_d[1], q_v[1])

    # Placeholders for intermediate calculations
    ADV = similar(s)
    FORCING = similar(s)
    KDIFF = similar(s)
    VDIFF = similar(s)
    COR = similar(s)

    # Entropy divergence forcing
    Cm = @. (q_l * Cl)/(Cvd + (q_v * Cvv) + (q_l * Cl))
    s_div = @. Cm * (Rd + q_v * Rv) * (u_r + w_z)

    # Condensation rate
    sat_ratio = inv_mu_transform.(mu_sat .+ satbar) # Saturation ratio
    #sat_ratio = max.(sat_ratio, 1.0e-6)
    q_sat = q_sat_liquid.(Tk, p)
    qss = (q_sat .* sat_ratio) .- q_sat
    max_N_c = 100.0

    # Condensation and nucleation
    q_cond = q_condensation.(sat_ratio, Tk, p, rho_d, q_v, q_c, max_N_c)
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
    
    # Rain evaporation rate
    mean_drop_radius = 250.0 # Mean radius of rain drops in microns
    q_evap = q_evaporation.(sat_ratio, Tk, p, rho_d, q_v, q_r, mean_drop_radius)
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

    # Entropy change due to condensation and evaporation
    s_cond = s_condensation.(q_evap, q_cond, Tk, rho_d, q_v, q_l, p)
    for i in 1:length(s_cond)
        if isnan(s_cond[i])
            println("s_cond is NaN at index $i")
            println("q_evap: $(q_evap[i])")
            println("q_cond: $(q_cond[i])")
            println("q_v: $(q_v[i])")
            println("q_l: $(q_l[i])")
            println("q_sat: $(q_sat[i])")
            println("Tk: $(Tk[i])")
            println("rho_d: $(rho_d[i])")
            println("p: $(p[i])")
            error("NaN found at time $(t)!")
        end
    end

    # Rayleigh damping
    rayleigh_coeff = Rayleigh_damping.(alpha, z, z_damp, z[end])

    # Autoconversion rate
    q_auto = autoconversion.(q_c, rho_d) 

    # Collection rate
    q_coll = collection.(q_c, q_r, rho_d, Tk) 

    # Sedimentation rate
    Vt = sedimentation.(q_r, rho_d, Tk)

    # Calculate the flux divergence of the falling precipitation
    col = deepcopy(mtile.tile.kbasis.data[mtile.model.grid_params.vars["mu_r"]]) 
    col.uMish .= Vt
    Btransform!(col)
    Atransform!(col)
    Vt .= Itransform!(col)
    dVtdz = Ixtransform(col)
    Vt_flux = (q_r_z .* Vt) .+ (q_r .* dVtdz) .+ (q_r .* Vt .* xi_z) # Precipitation flux divergence

    # Calculate the vertical diffusivity
    col = deepcopy(mtile.tile.kbasis.data[mtile.model.grid_params.vars["mu"]])

    # Vertical mixing length based on Louis parameterization
    Sv = sqrt.((u_z .* u_z) .+ (v_z .* v_z))
    lv = 1.0 ./ ((1.0 ./ (0.4 .* z)) .+ (1.0 ./ 80.0))
    Kv = (lv.^2) .* Sv

    # Get the 10 meter wind (assuming 10 m @ z == z_ref_level)
    u10 = u[z_ref_level]
    v10 = v[z_ref_level]
    U10 = sqrt(u10^2 + v10^2)

    # Drag applies at z = 0
    if Cd < 0.0
        # Negative parameter so use a wind speed dependent drag
        # From Komori et al. (2018)
        if U10 < 5.2
            Cd = 1.0e-3
        elseif U10 < 33.6
            Cd = 4.4e-4 * U10^0.5
        else
            Cd = 2.55e-3
        end
    end

    # Calculate vapor mixing first since it is needed for entropy and saturation ratio
    q10 = q_v[z_ref_level]
    col.uMish .= rho_d .* Kv .* (mu_v_z .+ mubar_z)
    col.uMish[1] = rho_d[1] * Ck * U10 * (q_sfc - q10) * mu_v_factor[z_ref_level] # Q FLUX
    Btransform!(col)
    Atransform!(col)
    VDIFF .= (Ixtransform(col)) ./ rho_d
    q_flux = VDIFF ./ mu_v_factor

    @turbo ADV .= @. (-u * mu_v_r) + (-w * (mu_v_z + mubar_z)) #MUADV
    FORCING .= @. (q_evap -q_cond) * mu_v_factor
    @turbo KDIFF .= @. Khdiff * ((mu_v_r / r) + mu_v_rr)
    @turbo expdot[colstart:colend,3] .= @. ADV + FORCING + KDIFF + VDIFF
    @turbo impdot[colstart:colend,3] .= @. Kvdiff * mu_v_zz

    # Apply the subgrid vapor flux to the entropy
    s_v_mix = s_vapor_mixing.(q_flux, Tk, rho_d, q_v)

    # Entropy mixing
    s10 = s[z_ref_level]
    col.uMish .= rho_d .* Kv .* (s_z .+ sbar_z)
    col.uMish[1] = rho_d[1] * Ck * U10 * (s_sfc - s10) # S FLUX
    Btransform!(col)
    Atransform!(col)
    VDIFF .= (Ixtransform(col)) ./ rho_d

    @turbo ADV .= @. (-u * s_r) + (-w * (s_z + sbar_z)) #SADV
    FORCING .= @. s_cond + s_div + s_v_mix
    @turbo KDIFF .= @. Khdiff * ((s_r/r) + s_rr)
    @turbo expdot[colstart:colend,1] .= @. ADV + FORCING + KDIFF + VDIFF
    @turbo impdot[colstart:colend,1] .= @. Kvdiff * s_zz

    @turbo ADV .= @. (-u * xi_r) + (-w * (xi_z + xibar_z)) #XI ADV
    @turbo FORCING .= @. - u_r - w_z
    @turbo expdot[colstart:colend,2] .= @. ADV + FORCING
    impdot[colstart:colend,2] .= @. -w_z

    col.uMish .= rho_d .* Kv .* u_z
    col.uMish[1] = rho_d[1] * Cd * U10 * u10 #UDRAG
    Btransform!(col)
    Atransform!(col)
    VDIFF .= (Ixtransform(col)) ./ rho_d

    @turbo ADV .= @. (-u * u_r) + (-w * u_z) #UADV
    @turbo FORCING .= @. -dpdr / rho_t #UPGF
    @turbo KDIFF .= @. Khdiff * ((u_r / r)+ u_rr)
    COR .= @. (v * (f + (v / r)))
    @turbo expdot[colstart:colend,4] .= @. ADV + FORCING + COR + KDIFF + VDIFF
    @turbo impdot[colstart:colend,4] .= @. Kvdiff * u_zz

    col.uMish .= rho_d .* Kv .* v_z
    col.uMish[1] = rho_d[1] * Cd * U10 * v10 #VDRAG
    Btransform!(col)
    Atransform!(col)
    VDIFF .= Ixtransform(col)

    @turbo ADV .= @. (-u * v_r) + (-w * v_z) #VBADV
    @turbo FORCING .= 0.0 #VBPGF # There is no L pressure gradient in an axisymmetric storm
    COR .= @. (-u * (f + (v / r))) #VBCOR
    KDIFF .= @. Khdiff * ((v_r / r) + v_rr) #VHDIFF
    @turbo expdot[colstart:colend,5] .= @. ADV + FORCING + COR + KDIFF + VDIFF
    @turbo impdot[colstart:colend,5] .= @. Kvdiff * v_zz

    @turbo ADV .= @. (-u * w_r) + (-w * w_z) #WADV
    @turbo FORCING .= @.  ((-gravity * rho_p) - dpdz) / rho_t
    @turbo KDIFF .= @. Khdiff * ((w_r /r) + w_rr)
    @turbo expdot[colstart:colend,6] .= @. ADV + FORCING + KDIFF
    impdot[colstart:colend,6] .= @. -(Pxi_bar * xi_z)

    col.uMish .= rho_d .* Kv .* mu_c_z
    Btransform!(col)
    Atransform!(col)
    VDIFF .= (Ixtransform(col)) ./ rho_d

    @turbo ADV .= @. (-u * mu_c_r) + (-w * mu_c_z) #Q_C ADV
    FORCING .= @. (q_cond -q_auto -q_coll) * mu_c_factor
    @turbo KDIFF .= @. Khdiff * ((mu_c_r/r) + mu_c_rr)
    @turbo expdot[colstart:colend,7] .= @. ADV + FORCING + KDIFF + VDIFF
    @turbo impdot[colstart:colend,7] .= @. Kv_mudiff * mu_c_zz

    col.uMish .= rho_d .* Kv .* mu_r_z
    Btransform!(col)
    Atransform!(col)
    VDIFF .= (Ixtransform(col)) ./ rho_d

    @turbo ADV .= @. (-u * mu_r_r) + (-w * mu_r_z) #Q_R ADV
    FORCING .= @. (q_auto +q_coll -q_evap -Vt_flux) * mu_r_factor
    @turbo KDIFF .= @. Khdiff * ((mu_r_r/r) + mu_r_rr)
    @turbo expdot[colstart:colend,8] .= @. ADV + FORCING + KDIFF + VDIFF
    @turbo impdot[colstart:colend,8] .= @. Kv_mudiff * mu_r_zz

    # Saturation forcing
    Q_s = Q_s_factor.(Tk, p, q_v, q_l)
    sat_forcing = @. (sat_ratio*(dqsdp(Tk, p, rho_d, q_v, q_l)*((u * dpdx) + (w * (dpdz - rhobar*gravity)))) + q_flux + ((q_evap - q_cond) * (1.0 + Q_s)))/q_sat

    col.uMish .= rho_d .* Kv .* (mu_sat_z + satbar_z)
    Btransform!(col)
    Atransform!(col)
    VDIFF .= (Ixtransform(col)) ./ rho_d

    @turbo ADV .= @. (-u * mu_sat_r) + (-w * (mu_sat_z + satbar_z)) #QSS ADV
    FORCING .= @. sat_forcing * dmudq.(mu_sat, sat_ratio)
    @turbo KDIFF .= @. Khdiff * ((mu_sat_r/r) + mu_sat_rr)
    @turbo expdot[colstart:colend,9] .= @. ADV + FORCING + KDIFF + VDIFF
    @turbo impdot[colstart:colend,9] .= @. Kv_mudiff * mu_sat_zz

    # Advance the explicit terms
    explicit_timestep(mtile, colstart, colend, t)

    # Solve for semi-implicit n+1 terms
    if mtile.model.options[:semiimplicit]
        semiimplicit_adjustment(mtile, colstart, colend, t)
    end

    # Adjust the condensation rate from the advected supersaturation
    condensation_adjustment_new(mtile, colstart, colend, t)

    # Use implicit timestep for diffusion
    diffusion_timestep(mtile, colstart, colend, t)

    # Increment the explicit timestep terms with other forcings
    #explicit_increment(mtile, colstart, colend, t)

end

"""
    primitive_equation_cylindrical(mtile, colstart, colend, t)

Primitive equations in full r-lambda-z cylindrical coordinates with thermodynamics and
vertical diffusion. Deprecated: incomplete and should not be used.
"""
function primitive_equation_cylindrical(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    @warn "primitive_equation_cylindrical is incomplete and should not be used" maxlog=1

    grid = mtile.tile
    gridpoints = mtile.tilepoints
    expdot = mtile.expdot_n
    impdot = mtile.impdot_n
    model = mtile.model
    refstate = mtile.ref_state

    # Physical parameters
    g = model.physical_params[:g]
    Kh = model.physical_params[:Kh]
    Cd = model.physical_params[:Cd]
    f = model.physical_params[:f]
    Um = model.physical_params[:Um]
    Vm = model.physical_params[:Vm]

    # Assign local variables with views
    r = view(gridpoints,colstart:colend,1)
    lambda = view(gridpoints,colstart:colend,2)
    z = view(gridpoints,colstart:colend,3)

    # Gridpoints
    x = view(gridpoints,colstart:colend,1)
    z = view(gridpoints,colstart:colend,2)

    # Variables
    s = view(grid.physical,colstart:colend,1,1)
    s_r = view(grid.physical,colstart:colend,1,2)
    s_rr = view(grid.physical,colstart:colend,1,3)
    s_l = view(grid.physical,colstart:colend,1,4)
    s_ll = view(grid.physical,colstart:colend,1,5)
    s_z = view(grid.physical,colstart:colend,1,6)
    s_zz = view(grid.physical,colstart:colend,1,7)

    xi = view(grid.physical,colstart:colend,2,1)
    xi_r = view(grid.physical,colstart:colend,2,2)
    xi_rr = view(grid.physical,colstart:colend,2,3)
    xi_l = view(grid.physical,colstart:colend,2,4)
    xi_ll = view(grid.physical,colstart:colend,2,5)
    xi_z = view(grid.physical,colstart:colend,2,6)
    xi_zz = view(grid.physical,colstart:colend,2,7)
    
    mu = view(grid.physical,colstart:colend,3,1)
    mu_r = view(grid.physical,colstart:colend,3,2)
    mu_rr = view(grid.physical,colstart:colend,3,3)
    mu_l = view(grid.physical,colstart:colend,3,4)
    mu_ll = view(grid.physical,colstart:colend,3,5)
    mu_z = view(grid.physical,colstart:colend,3,6)
    mu_zz = view(grid.physical,colstart:colend,3,7)

    u = view(grid.physical,colstart:colend,4,1)
    u_r = view(grid.physical,colstart:colend,4,2)
    u_rr = view(grid.physical,colstart:colend,4,3)
    u_l = view(grid.physical,colstart:colend,4,4)
    u_ll = view(grid.physical,colstart:colend,4,5)
    u_z = view(grid.physical,colstart:colend,4,6)
    u_zz = view(grid.physical,colstart:colend,4,7)

    v = view(grid.physical,colstart:colend,5,1)
    v_r = view(grid.physical,colstart:colend,5,2)
    v_rr = view(grid.physical,colstart:colend,5,3)
    v_l = view(grid.physical,colstart:colend,5,4)
    v_ll = view(grid.physical,colstart:colend,5,5)
    v_z = view(grid.physical,colstart:colend,5,6)
    v_zz = view(grid.physical,colstart:colend,5,7)

    w = view(grid.physical,colstart:colend,6,1)
    w_r = view(grid.physical,colstart:colend,6,2)
    w_rr = view(grid.physical,colstart:colend,6,3)
    w_l = view(grid.physical,colstart:colend,6,4)
    w_ll = view(grid.physical,colstart:colend,6,5)
    w_z = view(grid.physical,colstart:colend,6,6)
    w_zz = view(grid.physical,colstart:colend,6,7)

    mu_c = view(grid.physical,colstart:colend,7,1)
    mu_c_r = view(grid.physical,colstart:colend,7,2)
    mu_c_rr = view(grid.physical,colstart:colend,7,3)
    mu_c_l = view(grid.physical,colstart:colend,7,4)
    mu_c_ll = view(grid.physical,colstart:colend,7,5)
    mu_c_z = view(grid.physical,colstart:colend,7,6)
    mu_c_zz = view(grid.physical,colstart:colend,7,7)

    mu_p = view(grid.physical,colstart:colend,8,1)
    mu_p_r = view(grid.physical,colstart:colend,8,2)
    mu_p_rr = view(grid.physical,colstart:colend,8,3)
    mu_p_l = view(grid.physical,colstart:colend,8,4)
    mu_p_ll = view(grid.physical,colstart:colend,8,5)
    mu_p_z = view(grid.physical,colstart:colend,8,6)
    mu_p_zz = view(grid.physical,colstart:colend,8,7)

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
    q_c = ahyp.(mu_c)               # Condensate mixing ratio
    q_p = ahyp.(mu_p)               # Precipitation mixing ratio
    rho_t = rho_d .* (1.0 .+ q_v .+ q_l)   # Total air density
    qvp = q_v .- ahyp.(mubar)       # Perturbation mixing ratio
    qvp_r = mu_r ./ dmudq.(mu, q_v) # Perturbation vapor gradient in r
    qvp_l = mu_l ./ dmudq.(mu, q_v) # Perturbation vapor gradient in l
    qvp_z = mu_z ./ dmudq.(mu, q_v) # Perturbation vapor gradient in z
    rhobar = dry_density.(xibar) .* (1.0 .+ ahyp.(mubar)) # Ref. air density
    rho_p = rho_t .- rhobar         # Perturbation air density

    # Get the mean speed of sound squared from the reference state
    Pxi_bar = mtile.ref_state.Pxi_bar

    # Placeholders for intermediate calculations
    ADV = similar(s)
    PGF = similar(s)
    KDIFF = similar(s)
    COR = similar(s)

    # Calculate the vertical diffusivity
    # Mixing length based on Louis parameterization
    S = sqrt.((u_z .* u_z) .+ (v_z .* v_z))
    l = 1.0 ./ ((1.0 ./ (0.4 .* z)) .+ (1.0 ./ 80.0))
    Kv = (l.^2) .* S

    @turbo ADV .= @. (-u * s_r) + (-v * s_l / r) + (-w * (s_z + sbar_z)) #SADV
    #No PGF
    @turbo KDIFF .= @. K * ((s_r / r) + s_rr + (s_ll / (r * r)) + s_zz)
    @turbo expdot[colstart:colend,1] .= @. ADV + KDIFF

    @turbo ADV .= @. (-u * xi_r) + (-v * xi_l / r) + (-w * (xi_z + xibar_z)) #XI ADV
    # No PGF or mass diffusion
    @turbo expdot[colstart:colend,2] .= @. ADV - u_x - w_z
    impdot[colstart:colend,2] .= @. -w_z

    @turbo ADV .= @. (-u * mu_x) + (-v * mu_l / r) + (-w * (mu_z + mubar_z)) #MU_ADV
    #No PGF
    @turbo KDIFF .= @. K * ((mu_r / r) + mu_rr + (mu_ll / (r * r)) + mu_zz)
    @turbo expdot[colstart:colend,3] .= @. ADV + KDIFF

    @turbo ADV .= @. (-u * u_r) + (-v * u_l / r) + (-w * u_z) #UADV
    PGF .= @. -(pressure_gradient(Tk, rho_d, q_v, s_r, xi_r, qvp_r) / rho_t) #UPGF
    @turbo KDIFF .= @. K * ((u_r / r) + u_rr + (u_ll / (r * r)) + u_zz)
    @turbo COR .= @. (v * (f + (v / r))) #UCOR

    # Surface wind speed based on storm motion
    sfcu = (Um * cos(lambda[1])) + (Vm * sin(lambda[1]))
    sfcv = (Vm * cos(lambda[1])) - (Um * sin(lambda[1]))

    # Get the 10 meter wind (assuming 10 m @ z == 2)
    u10 = u[2] + sfcu
    v10 = v[2] + sfcv
    U10 = sqrt(u10^2 + v10^2)

    # Differentiate Kv * du/dz
    col.uMish .= Kv .* u_z
    
    # Drag applies at z = 0
    # Use a wind speed dependent drag
    if U10 < 5.2
        Cd = 1.0e-3
    elseif U10 < 33.6
        Cd = 4.4e-4 * U10^0.5
    end
    col.uMish[1] = Cd * U10 * u10 #UDRAG

    Btransform!(col)
    Atransform!(col)
    VDIFF .= Ixtransform(col)
    
    @turbo expdot[colstart:colend,4] .= @. ADV + PGF + KDIFF + VDIFF + COR

    @turbo ADV .= @. (-v * v_r) + (-v * v_l / r) + (-w * v_z) #VADV
    PGF .= @. -(pressure_gradient(Tk, rho_d, q_v, s_l, xi_l, qvp_l) / rho_t) #VPGF
    @turbo KDIFF .= @. K * ((v_r / r) + v_rr + (v_ll / (r * r)) + v_zz)
    @turbo COR .= @. (-u * (f + (v / r))) #VCOR

    # Differentiate Kv * dv/dz
    col.uMish .= Kv .* v_z

    # Drag only applies at z = 0
    col.uMish[1] = Cd * U10 * v10 #VDRAG

    Btransform!(col)
    Atransform!(col)
    VDIFF .= Ixtransform(col)
    
    @turbo expdot[colstart:colend,5] .= @. ADV + PGF + KDIFF + COR

    @turbo ADV .= @. (-u * w_r) + (-v * w_l / r) + (-w * w_z) #WADV
    PGF .= @.  -(g * rho_p / rho_t) - (pressure_gradient(Tk, rho_d, q_v, s_z, xi_z, qvp_z) / rho_t)
    @turbo KDIFF .= @. K * ((w_r / r) + w_rr + (w_ll / (r * r)) + w_zz)
    @turbo expdot[colstart:colend,6] .= @. ADV + PGF + KDIFF
    impdot[colstart:colend,6] .= @. -(Pxi_bar * xi_z)

    @turbo ADV .= @. (-u * mu_c_r) + (-v * mu_c_l / r) + (-w * mu_c_z) #Q_C ADV
    #No PGF or diffusion
    @turbo expdot[colstart:colend,7] .= @. ADV

    @turbo ADV .= @. (-u * mu_p_r) + (-v * mu_p_l / r) + (-w * mu_p_z) #Q_P ADV
    #No PGF or diffusion
    @turbo expdot[colstart:colend,8] .= @. ADV

    # Advance the explicit terms
    explicit_timestep(mtile, colstart, colend, t)

    # Solve for semi-implicit n+1 terms
    if mtile.model.semiimplicit
        semiimplicit_adjustment(mtile, colstart, colend, t)
    end

end
