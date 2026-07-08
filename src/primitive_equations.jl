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
    sbar = ref_entropy(refstate)[:,1]
    sbar_z = ref_entropy(refstate)[:,2]
    sbar_zz = ref_entropy(refstate)[:,3]

    xibar = ref_xi(refstate)[:,1]
    xibar_z = ref_xi(refstate)[:,2]
    xibar_zz = ref_xi(refstate)[:,3]

    mubar = ref_mu(refstate)[:,1]
    mubar_z = ref_mu(refstate)[:,2]
    mubar_zz = ref_mu(refstate)[:,3]

    satbar = ref_sat(refstate)[:,1]
    satbar_z = ref_sat(refstate)[:,2]
    satbar_zz = ref_sat(refstate)[:,3]

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
    Pxi_bar = sound_speed_sq(mtile.ref_state)

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
    primitive_equation_XZ_rhod(mtile, colstart, colend, t)

Linear dry-air-density variant of [`primitive_equation_XZ`](@ref). The prognostic
control variable in slot 2 is the dry-air density perturbation `rho_d' = rho_d - rhobar`
(named `"rho_d"` in `grid_params.vars`) rather than the log-density `xi`. Carrying the
*linear* density makes the per-step cubic-spline `l_q` smoothing mass-conserving (it
preserves `∫rho_d`), removing the convexity/Jensen drift of smoothing `xi = ln(rho_d/rho_d0)`.
See `reference/mass_conservation_advective.tex`.

The continuity equation is the advective (product-rule) form
`∂rho_d/∂t = -v·∇rho_d - rho_d ∇·v`. All other equations are identical to the `xi` set;
the pressure gradient reuses the existing `pressure_gradient` by passing the perturbation
log-density gradients reconstructed from `rho_d'`.

NOTE: semi-implicit acoustics are NOT yet validated for this set (the implicit `impdot`
terms retain the `xi`-form linearization); run with `:semiimplicit => false` (Phase 2 will
re-derive the variable-coefficient Helmholtz operator).
"""
function primitive_equation_XZ_rhod(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

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

    precipitation = get(model.options, :precipitation, true)
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

    # Slot 2 is the dry-air density perturbation rho_d' (and its auto-derivatives)
    rho_dp = view(grid.physical,colstart:colend,2,1)
    rho_dp_x = view(grid.physical,colstart:colend,2,2)
    rho_dp_xx = view(grid.physical,colstart:colend,2,3)
    rho_dp_z = view(grid.physical,colstart:colend,2,4)
    rho_dp_zz = view(grid.physical,colstart:colend,2,5)

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
    sbar = ref_entropy(refstate)[:,1]
    sbar_z = ref_entropy(refstate)[:,2]
    sbar_zz = ref_entropy(refstate)[:,3]

    # Dry-air density reference (value + first vertical derivative)
    rho_dbar = ref_rho_d(refstate)[:,1]
    rho_dbar_z = ref_rho_d(refstate)[:,2]

    mubar = ref_mu(refstate)[:,1]
    mubar_z = ref_mu(refstate)[:,2]
    mubar_zz = ref_mu(refstate)[:,3]

    satbar = ref_sat(refstate)[:,1]
    satbar_z = ref_sat(refstate)[:,2]
    satbar_zz = ref_sat(refstate)[:,3]

    # Fundamental thermodynamic quantities derived from model variables
    mu_total = mu .+ mubar
    rho_d = rho_dp .+ rho_dbar       # Total dry air density (carried linearly)
    thermo = thermodynamic_tuple_rhod.(s .+ sbar, rho_d, mu_total)
    q_v = [x[1] for x in thermo]    # Total water vapor mixing ratio
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
    mu_v_factor = dmudq.(mu_total, q_v)
    qvp_x = mu_x ./ mu_v_factor # Perturbation vapor gradient in x
    qvp_z = mu_z ./ mu_v_factor # Perturbation vapor gradient in z

    # Perturbation log-density gradients reconstructed from the linear rho_d', so the
    # existing xi-based pressure_gradient is reused unchanged. xibar has no x-dependence;
    # xi' = ln(rho_d/rho_dbar) ⇒ xi'_z = (rho_d)_z/rho_d - (rho_dbar)_z/rho_dbar.
    xi_x = rho_dp_x ./ rho_d
    xi_z = ((rho_dp_z .+ rho_dbar_z) ./ rho_d) .- (rho_dbar_z ./ rho_dbar)

    rhobar = rho_dbar .* (1.0 .+ inv_mu_transform.(mubar)) # Ref. air density
    rho_p = rho_t .- rhobar         # Perturbation air density

    # Get the mean speed of sound squared from the reference state
    Pxi_bar = sound_speed_sq(mtile.ref_state)

    # Pressure gradients (perturbation form, reusing the xi-based chain rule)
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
    q_sat = q_sat_liquid.(Tk, p)
    qss = (q_sat .* sat_ratio) .- q_sat
    max_N_c = 100.0

    # Condensation and nucleation
    q_cond = q_condensation.(sat_ratio, Tk, p, rho_d, q_v, q_c, max_N_c)
    for i in 1:length(q_cond)
        if isnan(q_cond[i])
            error("q_cond is NaN at index $i, time $(t)!")
        end
    end

    # Rain evaporation rate
    mean_r = 250.0 # Mean radius of rain drops in microns
    q_evap = precipitation ? q_evaporation.(sat_ratio, Tk, p, rho_d, q_v, q_r, mean_r) :
                             zero(q_v)

    # Entropy change due to condensation and evaporation
    s_cond = s_condensation.(q_evap, q_cond, Tk, rho_d, q_v, q_l, p)

    # Rayleigh damping
    rayleigh_coeff = Rayleigh_damping.(alpha, z, z_damp, z[end])

    if precipitation
        q_auto = autoconversion.(q_c, rho_d)
        q_coll = collection.(q_c, q_r, rho_d, Tk)
        Vt = sedimentation.(q_r, rho_d, Tk)
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

    # Dry-air mass continuity in advective (product-rule) form:
    # ∂rho_d/∂t = -v·∇rho_d - rho_d ∇·v  (= -∇·(rho_d v)).
    @turbo ADV .= @. (-u * rho_dp_x) + (-w * (rho_dp_z + rho_dbar_z)) #RHO_D ADV
    @turbo FORCING .= @. -rho_d * (u_x + w_z)
    @turbo expdot[colstart:colend,2] .= @. ADV + FORCING
    # Implicit acoustic continuity: flux form -∂_z(ρ̂_d w), a single spline derivative
    # of the dry-air mass flux, consistent with the φ = ρ̂_d w semi-implicit solve.
    flux_col = deepcopy(mtile.tile.kbasis.data[mtile.model.grid_params.vars["w"]])
    flux_col.uMish .= rho_dbar .* w
    Btransform!(flux_col)
    Atransform!(flux_col)
    impdot[colstart:colend,2] .= .-Ixtransform(flux_col)

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
    # Implicit acoustic w-momentum: -c̄_ρ ∂_z ρ_d' with c̄_ρ = Pxi_bar/ρ̂_d
    impdot[colstart:colend,5] .= @. -(Pxi_bar / rho_dbar) * rho_dp_z

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

    # Solve for semi-implicit n+1 terms (linear rho_d mass-flux acoustic adjustment)
    if mtile.model.options[:semiimplicit]
        semiimplicit_adjustment_rhod(mtile, colstart, colend, t)
    end

    # Adjust the condensation rate from the advected supersaturation (rho_d form)
    condensation_adjustment_new_rhod(mtile, colstart, colend, t)

    # Use implicit timestep for diffusion
    diffusion_timestep(mtile, colstart, colend, t)

end

"""
    primitive_equation_XZ_rhod_pd(mtile, colstart, colend, t)

Partial-density moisture variant of [`primitive_equation_XZ_rhod`](@ref). The prognostic
moisture variables are the *partial densities* `rho_v = rho_d·q_v` (slot 3),
`rho_c = rho_d·q_c` (slot 6) and `rho_r = rho_d·q_r` (slot 7), carried as perturbations
from the reference partial densities. Because the partial densities are extensive (linear
in mass), the per-step cubic-spline `l_q` smoothing preserves each `∫rho_x`, so the
physical water mass `∫(rho_v+rho_c+rho_r)` is conserved (the mixing-ratio `mu` set only
conserved `∫mu`, not `∫rho_d q`). See `reference/phase3_moisture_partial_density.md`.

The moisture continuity equations are the advective (product-rule) form, mirroring the
`rho_d` continuity: `∂rho_x/∂t = -v·∇rho_x - rho_x ∇·v + rho_d·(sources)`. The
condensation source `rho_d·(q_evap - q_cond)` is exactly equal and opposite between vapor
and cloud, so it cancels in the water-mass sum.

This set consumes a *physical* reference state directly (`MoistReferenceState` or
`CondensateReferenceState`): `ref_rho_v`/`ref_rho_c` supply the vapor/condensate partial
densities, and `rhobar = rho_dbar + rho_vbar + rho_cbar` is condensate-inclusive so a
saturated cloudy base (Bryan & Fritsch 2002) is truly neutrally buoyant. The transformed
saturation ratio `mu_sat` (slot 8) is retained as in the `rho_d` set; the reference's raw
`satbar` is rescaled into the transformed convention (linear-mu assumption).

NOTE: semi-implicit acoustics reuse `semiimplicit_adjustment_rhod` (operates on `rho_d`
and `w`, unchanged by the moisture representation).
"""
function primitive_equation_XZ_rhod_pd(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

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

    precipitation = get(model.options, :precipitation, true)
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

    # Slot 2 is the dry-air density perturbation rho_d'
    rho_dp = view(grid.physical,colstart:colend,2,1)
    rho_dp_x = view(grid.physical,colstart:colend,2,2)
    rho_dp_xx = view(grid.physical,colstart:colend,2,3)
    rho_dp_z = view(grid.physical,colstart:colend,2,4)
    rho_dp_zz = view(grid.physical,colstart:colend,2,5)

    # Slot 3 is the vapor partial-density perturbation rho_v'
    rho_vp = view(grid.physical,colstart:colend,3,1)
    rho_vp_x = view(grid.physical,colstart:colend,3,2)
    rho_vp_xx = view(grid.physical,colstart:colend,3,3)
    rho_vp_z = view(grid.physical,colstart:colend,3,4)
    rho_vp_zz = view(grid.physical,colstart:colend,3,5)

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

    # Slot 6 is the cloud partial-density perturbation rho_c'
    rho_cp = view(grid.physical,colstart:colend,6,1)
    rho_cp_x = view(grid.physical,colstart:colend,6,2)
    rho_cp_xx = view(grid.physical,colstart:colend,6,3)
    rho_cp_z = view(grid.physical,colstart:colend,6,4)
    rho_cp_zz = view(grid.physical,colstart:colend,6,5)

    # Slot 7 is the rain partial density rho_r (reference rho_rbar = 0)
    rho_rp = view(grid.physical,colstart:colend,7,1)
    rho_rp_x = view(grid.physical,colstart:colend,7,2)
    rho_rp_xx = view(grid.physical,colstart:colend,7,3)
    rho_rp_z = view(grid.physical,colstart:colend,7,4)
    rho_rp_zz = view(grid.physical,colstart:colend,7,5)

    mu_sat = view(grid.physical,colstart:colend,8,1)
    mu_sat_x = view(grid.physical,colstart:colend,8,2)
    mu_sat_xx = view(grid.physical,colstart:colend,8,3)
    mu_sat_z = view(grid.physical,colstart:colend,8,4)
    mu_sat_zz = view(grid.physical,colstart:colend,8,5)

    # Get reference state (physical partial densities)
    sbar = ref_entropy(refstate)[:,1]
    sbar_z = ref_entropy(refstate)[:,2]
    sbar_zz = ref_entropy(refstate)[:,3]

    rho_dbar = ref_rho_d(refstate)[:,1]
    rho_dbar_z = ref_rho_d(refstate)[:,2]

    # Vapor partial-density reference (value + first vertical derivative). Guard the
    # dry-reference convention where ref_rho_v is the scalar 0.0.
    rvbar = ref_rho_v(refstate)
    rho_vbar = rvbar === 0.0 ? zero(sbar) : rvbar[:,1]
    rho_vbar_z = rvbar === 0.0 ? zero(sbar) : rvbar[:,2]

    # Condensate partial-density reference (scalar 0.0 for MoistReferenceState ⇒ a
    # cloudless base; a CondensateReferenceState carries a nonzero cloudy base profile).
    rcbar = ref_rho_c(refstate)
    rho_cbar = rcbar === 0.0 ? zero(sbar) : rcbar[:,1]
    rho_cbar_z = rcbar === 0.0 ? zero(sbar) : rcbar[:,2]

    # Saturation-ratio reference, rescaled from raw to the transformed (mu) convention
    # (linear-mu assumption; mu_transform is a constant scaling that commutes with the
    # spectral smoothing, so it applies to the value and derivative columns alike).
    satraw = ref_sat(refstate)
    satbar = satraw === 0.0 ? zero(sbar) : mu_transform.(satraw[:,1])
    satbar_z = satraw === 0.0 ? zero(sbar) : mu_transform.(satraw[:,2])

    # Total fields (perturbation + reference)
    rho_d = rho_dp .+ rho_dbar       # Total dry air density
    rho_v = rho_vp .+ rho_vbar       # Total vapor partial density
    rho_c = rho_cp .+ rho_cbar       # Total cloud partial density
    rho_r = rho_rp                   # Total rain partial density (rho_rbar = 0)

    # Fundamental thermodynamic quantities
    thermo = thermodynamic_tuple_pd.(s .+ sbar, rho_d, rho_v)
    q_v = [x[1] for x in thermo]    # Water vapor mixing ratio
    Tk = [x[3] for x in thermo]     # Temperature in K
    p = [x[4] for x in thermo]      # Total air pressure
    q_c = rho_c ./ rho_d            # Cloud water mixing ratio
    q_r = rho_r ./ rho_d            # Rain water mixing ratio
    q_l = q_c .+ q_r                # Liquid mixing ratio
    q_t = q_v .+ q_l                # Total water mixing ratio
    rho_t = rho_d .+ rho_v .+ rho_c .+ rho_r   # Total air density

    # Total vertical gradients of the partial densities (perturbation + reference)
    rho_d_z = rho_dp_z .+ rho_dbar_z
    rho_v_z = rho_vp_z .+ rho_vbar_z
    rho_c_z = rho_cp_z .+ rho_cbar_z

    # Perturbation log-density gradients reconstructed from the linear rho_d', so the
    # existing xi-based pressure_gradient is reused unchanged (xibar has no x-dependence;
    # xi' = ln(rho_d/rho_dbar) ⇒ xi'_z = rho_d_z/rho_d - rho_dbar_z/rho_dbar).
    xi_x = rho_dp_x ./ rho_d
    xi_z = (rho_d_z ./ rho_d) .- (rho_dbar_z ./ rho_dbar)

    # Perturbation vapor-mixing-ratio gradients from the partial densities via the
    # quotient rule q_v = rho_v/rho_d, minus the reference contribution (the base is
    # x-uniform so q_vbar_x = 0).
    q_vbar = rho_vbar ./ rho_dbar
    qvp_x = (rho_vp_x .- (q_v .* rho_dp_x)) ./ rho_d
    qvp_z = ((rho_v_z .- (q_v .* rho_d_z)) ./ rho_d) .-
            ((rho_vbar_z .- (q_vbar .* rho_dbar_z)) ./ rho_dbar)

    # Reference total (condensate-inclusive) air density and the perturbation
    rhobar = rho_dbar .+ rho_vbar .+ rho_cbar
    rho_p = rho_t .- rhobar

    # Get the mean speed of sound squared from the reference state
    Pxi_bar = sound_speed_sq(refstate)

    # Pressure gradients (perturbation form, reusing the xi-based chain rule)
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
    q_sat = q_sat_liquid.(Tk, p)
    max_N_c = 100.0

    # Condensation and nucleation
    q_cond = q_condensation.(sat_ratio, Tk, p, rho_d, q_v, q_c, max_N_c)
    for i in 1:length(q_cond)
        if isnan(q_cond[i])
            error("q_cond is NaN at index $i, time $(t)!")
        end
    end

    # Rain evaporation rate
    mean_r = 250.0 # Mean radius of rain drops in microns
    q_evap = precipitation ? q_evaporation.(sat_ratio, Tk, p, rho_d, q_v, q_r, mean_r) :
                             zero(q_v)

    # Entropy change due to condensation and evaporation
    s_cond = s_condensation.(q_evap, q_cond, Tk, rho_d, q_v, q_l, p)

    # Rayleigh damping
    rayleigh_coeff = Rayleigh_damping.(alpha, z, z_damp, z[end])

    if precipitation
        q_auto = autoconversion.(q_c, rho_d)
        q_coll = collection.(q_c, q_r, rho_d, Tk)
        Vt = sedimentation.(q_r, rho_d, Tk)
        # Rain mixing-ratio vertical gradient from the partial densities
        q_r_z = (rho_rp_z .- (q_r .* rho_d_z)) ./ rho_d
        col = deepcopy(mtile.tile.kbasis.data[mtile.model.grid_params.vars["rho_r"]])
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
    col = deepcopy(mtile.tile.kbasis.data[mtile.model.grid_params.vars["rho_v"]])

    # Vertical mixing length based on Louis parameterization
    Sv = sqrt.(u_z.^2)
    lv = 1.0 ./ ((1.0 ./ (0.4 .* z)) .+ (1.0 ./ 80.0))
    Kv = vertical_mixing ? (lv.^2) .* Sv : zero(Sv)

    # Subgrid vapor flux: tendency on the *extensive* rho_v is the flux divergence
    # ∂z(rho_d Kv ∂q_v/∂z); the equivalent intensive rate q_flux = (1/rho_d)∂z(...) is
    # reused by the entropy and saturation forcings.
    qv_z_total = (rho_v_z .- (q_v .* rho_d_z)) ./ rho_d   # ∂q_v/∂z (total)
    col.uMish .= rho_d .* Kv .* qv_z_total
    Btransform!(col)
    Atransform!(col)
    VDIFF .= Ixtransform(col)
    q_flux = VDIFF ./ rho_d

    @turbo ADV .= @. (-u * rho_vp_x) + (-w * rho_v_z) #RHO_V ADV
    @turbo FORCING .= @. (-rho_v * (u_x + w_z)) + (rho_d * (q_evap - q_cond))
    @turbo KDIFF .= @. Khdiff * rho_vp_xx
    @turbo expdot[colstart:colend,3] .= @. ADV + FORCING + KDIFF + VDIFF
    @turbo impdot[colstart:colend,3] .= @. Kvdiff * rho_vp_zz

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

    # Dry-air mass continuity in advective (product-rule) form
    @turbo ADV .= @. (-u * rho_dp_x) + (-w * rho_d_z) #RHO_D ADV
    @turbo FORCING .= @. -rho_d * (u_x + w_z)
    @turbo expdot[colstart:colend,2] .= @. ADV + FORCING
    # Implicit acoustic continuity: flux form -∂_z(ρ̂_d w)
    flux_col = deepcopy(mtile.tile.kbasis.data[mtile.model.grid_params.vars["w"]])
    flux_col.uMish .= rho_dbar .* w
    Btransform!(flux_col)
    Atransform!(flux_col)
    impdot[colstart:colend,2] .= .-Ixtransform(flux_col)

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
    # Implicit acoustic w-momentum: -c̄_ρ ∂_z ρ_d' with c̄_ρ = Pxi_bar/ρ̂_d
    impdot[colstart:colend,5] .= @. -(Pxi_bar / rho_dbar) * rho_dp_z

    # Cloud partial-density continuity (product-rule form) + microphysics source
    qc_z_total = (rho_c_z .- (q_c .* rho_d_z)) ./ rho_d
    col.uMish .= rho_d .* Kv .* qc_z_total
    Btransform!(col)
    Atransform!(col)
    VDIFF .= Ixtransform(col)

    @turbo ADV .= @. (-u * rho_cp_x) + (-w * rho_c_z) #RHO_C ADV
    @turbo FORCING .= @. (-rho_c * (u_x + w_z)) + (rho_d * (q_cond - q_auto - q_coll))
    @turbo KDIFF .= @. Khdiff * rho_cp_xx
    @turbo expdot[colstart:colend,6] .= @. ADV + FORCING + KDIFF + VDIFF
    @turbo impdot[colstart:colend,6] .= @. Kv_mudiff * rho_cp_zz

    # Rain partial-density continuity (product-rule form) + microphysics source
    qr_z_total = (rho_rp_z .- (q_r .* rho_d_z)) ./ rho_d
    col.uMish .= rho_d .* Kv .* qr_z_total
    Btransform!(col)
    Atransform!(col)
    VDIFF .= Ixtransform(col)

    @turbo ADV .= @. (-u * rho_rp_x) + (-w * rho_rp_z) #RHO_R ADV
    @turbo FORCING .= @. (-rho_r * (u_x + w_z)) + (rho_d * (q_auto + q_coll - q_evap - Vt_flux))
    @turbo KDIFF .= @. Khdiff * rho_rp_xx
    @turbo expdot[colstart:colend,7] .= @. ADV + FORCING + KDIFF + VDIFF
    @turbo impdot[colstart:colend,7] .= @. Kv_mudiff * rho_rp_zz

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

    # Solve for semi-implicit n+1 terms (linear rho_d mass-flux acoustic adjustment)
    if mtile.model.options[:semiimplicit]
        semiimplicit_adjustment_rhod(mtile, colstart, colend, t)
    end

    # Adjust the condensation rate from the advected supersaturation (partial-density form)
    condensation_adjustment_pd(mtile, colstart, colend, t)

    # Use implicit timestep for diffusion
    diffusion_timestep_pd(mtile, colstart, colend, t)

end

"""
    primitive_equation_XZ_sigma(mtile, colstart, colend, t)

Entropy-density variant of [`primitive_equation_XZ_rhod_pd`](@ref). Slot 1 carries the
**extensive entropy density** `σ = ρ_d·s` (variable name `"sigma"`) as a perturbation
`σ' = σ - σ̂` from the reference `σ̂ = ρ̂_d·ŝ`, instead of the intensive specific entropy `s`.
Because `σ` is extensive (linear in mass), the per-step cubic-spline `l_q` smoothing preserves
`∫σ` — putting entropy on par with the partial densities (`∫ρ_v`, `∫ρ_c`) and removing the
intensive-variable smoothing drift that the `s` set incurred. See
`reference/energy_entropy_conservation_handoff.md` (Stage 2).

The σ continuity is the product-rule form mirroring `ρ_v`: `∂σ/∂t = -v·∇σ - σ ∇·v + ρ_d·Ḟ_s`,
where `Ḟ_s = s_cond + s_div + s_v_mix` is the same intensive entropy source as the `s` set,
multiplied by `ρ_d` to act on the density. The specific entropy `s = σ/ρ_d` is recovered
wherever the EOS needs it, and the perturbation gradients `s'_x, s'_z` are reconstructed from
`σ` and `ρ_d` by the quotient rule (the template used for `q_v` from `ρ_v`), so the existing
`pressure_gradient` chain rule (`P_s ∂s'/∂x`) is reused unchanged. The mass (`ρ_d`), moisture
(`ρ_v, ρ_c, ρ_r`), momentum and saturation slots are identical to the `_pd` set; the
semi-implicit acoustics (`semiimplicit_adjustment_rhod`) never read entropy and are reused.
"""
function primitive_equation_XZ_sigma(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

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

    precipitation = get(model.options, :precipitation, true)
    vertical_mixing = get(model.options, :vertical_mixing, true)

    # Gridpoints
    x = view(gridpoints,colstart:colend,1)
    z = view(gridpoints,colstart:colend,2)

    # Slot 1 is the entropy-density perturbation sigma' = rho_d*s - sigmabar
    sigmap = view(grid.physical,colstart:colend,1,1)
    sigmap_x = view(grid.physical,colstart:colend,1,2)
    sigmap_xx = view(grid.physical,colstart:colend,1,3)
    sigmap_z = view(grid.physical,colstart:colend,1,4)
    sigmap_zz = view(grid.physical,colstart:colend,1,5)

    # Slot 2 is the dry-air density perturbation rho_d'
    rho_dp = view(grid.physical,colstart:colend,2,1)
    rho_dp_x = view(grid.physical,colstart:colend,2,2)
    rho_dp_xx = view(grid.physical,colstart:colend,2,3)
    rho_dp_z = view(grid.physical,colstart:colend,2,4)
    rho_dp_zz = view(grid.physical,colstart:colend,2,5)

    # Slot 3 is the vapor partial-density perturbation rho_v'
    rho_vp = view(grid.physical,colstart:colend,3,1)
    rho_vp_x = view(grid.physical,colstart:colend,3,2)
    rho_vp_xx = view(grid.physical,colstart:colend,3,3)
    rho_vp_z = view(grid.physical,colstart:colend,3,4)
    rho_vp_zz = view(grid.physical,colstart:colend,3,5)

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

    # Slot 6 is the cloud partial-density perturbation rho_c'
    rho_cp = view(grid.physical,colstart:colend,6,1)
    rho_cp_x = view(grid.physical,colstart:colend,6,2)
    rho_cp_xx = view(grid.physical,colstart:colend,6,3)
    rho_cp_z = view(grid.physical,colstart:colend,6,4)
    rho_cp_zz = view(grid.physical,colstart:colend,6,5)

    # Slot 7 is the rain partial density rho_r (reference rho_rbar = 0)
    rho_rp = view(grid.physical,colstart:colend,7,1)
    rho_rp_x = view(grid.physical,colstart:colend,7,2)
    rho_rp_xx = view(grid.physical,colstart:colend,7,3)
    rho_rp_z = view(grid.physical,colstart:colend,7,4)
    rho_rp_zz = view(grid.physical,colstart:colend,7,5)

    mu_sat = view(grid.physical,colstart:colend,8,1)
    mu_sat_x = view(grid.physical,colstart:colend,8,2)
    mu_sat_xx = view(grid.physical,colstart:colend,8,3)
    mu_sat_z = view(grid.physical,colstart:colend,8,4)
    mu_sat_zz = view(grid.physical,colstart:colend,8,5)

    # Get reference state (physical partial densities)
    sbar = ref_entropy(refstate)[:,1]
    sbar_z = ref_entropy(refstate)[:,2]

    rho_dbar = ref_rho_d(refstate)[:,1]
    rho_dbar_z = ref_rho_d(refstate)[:,2]

    # Entropy-density reference and its vertical gradient σ̂ = ρ̂_d·ŝ (extensive analogue of
    # sbar), taken directly from the reference state so σ̂_z is the spectrally-consistent
    # derivative computed on the reference basis (the product rule ρ̂_d_z·ŝ + ρ̂_d·ŝ_z is a
    # higher-degree polynomial outside the spline space and breaks ∫σ conservation in the
    # reference-advection term).
    sigmabar = ref_sigma(refstate)[:,1]
    sigmabar_z = ref_sigma(refstate)[:,2]

    # Vapor partial-density reference (value + first vertical derivative). Guard the
    # dry-reference convention where ref_rho_v is the scalar 0.0.
    rvbar = ref_rho_v(refstate)
    rho_vbar = rvbar === 0.0 ? zero(sbar) : rvbar[:,1]
    rho_vbar_z = rvbar === 0.0 ? zero(sbar) : rvbar[:,2]

    # Condensate partial-density reference (scalar 0.0 for MoistReferenceState ⇒ a
    # cloudless base; a CondensateReferenceState carries a nonzero cloudy base profile).
    rcbar = ref_rho_c(refstate)
    rho_cbar = rcbar === 0.0 ? zero(sbar) : rcbar[:,1]
    rho_cbar_z = rcbar === 0.0 ? zero(sbar) : rcbar[:,2]

    # Saturation-ratio reference, rescaled from raw to the transformed (mu) convention.
    satraw = ref_sat(refstate)
    satbar = satraw === 0.0 ? zero(sbar) : mu_transform.(satraw[:,1])
    satbar_z = satraw === 0.0 ? zero(sbar) : mu_transform.(satraw[:,2])

    # Total fields (perturbation + reference)
    rho_d = rho_dp .+ rho_dbar       # Total dry air density
    rho_v = rho_vp .+ rho_vbar       # Total vapor partial density
    rho_c = rho_cp .+ rho_cbar       # Total cloud partial density
    rho_r = rho_rp                   # Total rain partial density (rho_rbar = 0)
    sigma = sigmap .+ sigmabar       # Total entropy density σ = ρ_d·s

    # Total vertical gradients of the partial densities (perturbation + reference)
    rho_d_z = rho_dp_z .+ rho_dbar_z
    rho_v_z = rho_vp_z .+ rho_vbar_z
    rho_c_z = rho_cp_z .+ rho_cbar_z
    sigma_z = sigmap_z .+ sigmabar_z

    # Specific entropy recovered from the density, and its total vertical gradient via the
    # quotient rule s = σ/ρ_d (the same template as q_v = ρ_v/ρ_d below).
    s_total = sigma ./ rho_d
    s_z_total = (sigma_z .- (s_total .* rho_d_z)) ./ rho_d

    # Fundamental thermodynamic quantities
    thermo = thermodynamic_tuple_pd.(s_total, rho_d, rho_v)
    q_v = [x[1] for x in thermo]    # Water vapor mixing ratio
    Tk = [x[3] for x in thermo]     # Temperature in K
    p = [x[4] for x in thermo]      # Total air pressure
    q_c = rho_c ./ rho_d            # Cloud water mixing ratio
    q_r = rho_r ./ rho_d            # Rain water mixing ratio
    q_l = q_c .+ q_r                # Liquid mixing ratio
    q_t = q_v .+ q_l                # Total water mixing ratio
    rho_t = rho_d .+ rho_v .+ rho_c .+ rho_r   # Total air density

    # Perturbation log-density gradients reconstructed from the linear rho_d' (reused by the
    # xi-based pressure_gradient; xibar has no x-dependence).
    xi_x = rho_dp_x ./ rho_d
    xi_z = (rho_d_z ./ rho_d) .- (rho_dbar_z ./ rho_dbar)

    # Perturbation specific-entropy gradients reconstructed from σ via the quotient rule
    # s' = σ/ρ_d - ŝ (σ̂ and ŝ are x-uniform), so the existing P_s·∂s'/∂x chain rule is reused.
    sp_x = (sigmap_x .- (s_total .* rho_dp_x)) ./ rho_d
    sp_z = s_z_total .- sbar_z

    # Perturbation vapor-mixing-ratio gradients from the partial densities via the quotient
    # rule q_v = rho_v/rho_d, minus the reference contribution (base is x-uniform).
    q_vbar = rho_vbar ./ rho_dbar
    qvp_x = (rho_vp_x .- (q_v .* rho_dp_x)) ./ rho_d
    qvp_z = ((rho_v_z .- (q_v .* rho_d_z)) ./ rho_d) .-
            ((rho_vbar_z .- (q_vbar .* rho_dbar_z)) ./ rho_dbar)

    # Reference total (condensate-inclusive) air density and the perturbation
    rhobar = rho_dbar .+ rho_vbar .+ rho_cbar
    rho_p = rho_t .- rhobar

    # Get the mean speed of sound squared from the reference state
    Pxi_bar = sound_speed_sq(refstate)

    # Pressure gradients (perturbation form, reusing the xi-based chain rule)
    dpdx = pressure_gradient.(Tk, rho_d, q_v, sp_x, xi_x, qvp_x)
    dpdz = pressure_gradient.(Tk, rho_d, q_v, sp_z, xi_z, qvp_z)

    # Placeholders for intermediate calculations
    ADV = similar(sigmap)
    FORCING = similar(sigmap)
    KDIFF = similar(sigmap)
    VDIFF = similar(sigmap)

    # Entropy divergence forcing (intensive; multiplied by rho_d for the sigma tendency)
    Cm = @. (q_l * Cl)/(Cvd + (q_v * Cvv) + (q_l * Cl))
    s_div = @. Cm * (Rd + q_v * Rv) * (u_x + w_z)

    # Condensation rate
    sat_ratio = inv_mu_transform.(mu_sat .+ satbar) # Saturation ratio
    q_sat = q_sat_liquid.(Tk, p)
    max_N_c = 100.0

    # Condensation and nucleation
    q_cond = q_condensation.(sat_ratio, Tk, p, rho_d, q_v, q_c, max_N_c)
    for i in 1:length(q_cond)
        if isnan(q_cond[i])
            error("q_cond is NaN at index $i, time $(t)!")
        end
    end

    # Rain evaporation rate
    mean_r = 250.0 # Mean radius of rain drops in microns
    q_evap = precipitation ? q_evaporation.(sat_ratio, Tk, p, rho_d, q_v, q_r, mean_r) :
                             zero(q_v)

    # Entropy change due to condensation and evaporation (intensive)
    s_cond = s_condensation.(q_evap, q_cond, Tk, rho_d, q_v, q_l, p)

    # Rayleigh damping
    rayleigh_coeff = Rayleigh_damping.(alpha, z, z_damp, z[end])

    if precipitation
        q_auto = autoconversion.(q_c, rho_d)
        q_coll = collection.(q_c, q_r, rho_d, Tk)
        Vt = sedimentation.(q_r, rho_d, Tk)
        q_r_z = (rho_rp_z .- (q_r .* rho_d_z)) ./ rho_d
        col = deepcopy(mtile.tile.kbasis.data[mtile.model.grid_params.vars["rho_r"]])
        col.uMish .= Vt
        Btransform!(col)
        Atransform!(col)
        Vt .= Itransform!(col)
        dVtdz = Ixtransform(col)
        Vt_flux = (q_r_z .* Vt) .+ (q_r .* dVtdz) .+ (q_r .* Vt .* xi_z)
    else
        q_auto = zero(q_v)
        q_coll = zero(q_v)
        Vt_flux = zero(q_v)
    end

    # Calculate the vertical diffusivity
    col = deepcopy(mtile.tile.kbasis.data[mtile.model.grid_params.vars["rho_v"]])

    # Vertical mixing length based on Louis parameterization
    Sv = sqrt.(u_z.^2)
    lv = 1.0 ./ ((1.0 ./ (0.4 .* z)) .+ (1.0 ./ 80.0))
    Kv = vertical_mixing ? (lv.^2) .* Sv : zero(Sv)

    # Subgrid vapor flux (extensive: ∂z(rho_d Kv ∂q_v/∂z)); intensive rate q_flux reused
    # by the entropy and saturation forcings.
    qv_z_total = (rho_v_z .- (q_v .* rho_d_z)) ./ rho_d   # ∂q_v/∂z (total)
    col.uMish .= rho_d .* Kv .* qv_z_total
    Btransform!(col)
    Atransform!(col)
    VDIFF .= Ixtransform(col)
    q_flux = VDIFF ./ rho_d

    @turbo ADV .= @. (-u * rho_vp_x) + (-w * rho_v_z) #RHO_V ADV
    @turbo FORCING .= @. (-rho_v * (u_x + w_z)) + (rho_d * (q_evap - q_cond))
    @turbo KDIFF .= @. Khdiff * rho_vp_xx
    @turbo expdot[colstart:colend,3] .= @. ADV + FORCING + KDIFF + VDIFF
    @turbo impdot[colstart:colend,3] .= @. Kvdiff * rho_vp_zz

    # Apply the subgrid vapor flux to the entropy (intensive source)
    s_v_mix = s_vapor_mixing.(q_flux, Tk, rho_d, q_v)

    # Entropy-density mixing: extensive flux divergence ∂z(rho_d Kv ∂s/∂z) (no 1/rho_d,
    # mirroring the vapor flux), so the smoothing acts on the density σ.
    col.uMish .= rho_d .* Kv .* s_z_total
    Btransform!(col)
    Atransform!(col)
    VDIFF .= Ixtransform(col)

    # Entropy-density continuity (product-rule form), source = rho_d·(intensive sources)
    @turbo ADV .= @. (-u * sigmap_x) + (-w * sigma_z) #SIGMA ADV
    @turbo FORCING .= @. (-sigma * (u_x + w_z)) + (rho_d * (s_cond + s_div + s_v_mix))
    @turbo KDIFF .= @. Khdiff * sigmap_xx
    @turbo expdot[colstart:colend,1] .= @. ADV + FORCING + KDIFF + VDIFF
    @turbo impdot[colstart:colend,1] .= @. Kvdiff * sigmap_zz

    # Dry-air mass continuity in advective (product-rule) form
    @turbo ADV .= @. (-u * rho_dp_x) + (-w * rho_d_z) #RHO_D ADV
    @turbo FORCING .= @. -rho_d * (u_x + w_z)
    @turbo expdot[colstart:colend,2] .= @. ADV + FORCING
    # Implicit acoustic continuity: flux form -∂_z(ρ̂_d w)
    flux_col = deepcopy(mtile.tile.kbasis.data[mtile.model.grid_params.vars["w"]])
    flux_col.uMish .= rho_dbar .* w
    Btransform!(flux_col)
    Atransform!(flux_col)
    impdot[colstart:colend,2] .= .-Ixtransform(flux_col)

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
    # Implicit acoustic w-momentum: -c̄_ρ ∂_z ρ_d' with c̄_ρ = Pxi_bar/ρ̂_d
    impdot[colstart:colend,5] .= @. -(Pxi_bar / rho_dbar) * rho_dp_z

    # Cloud partial-density continuity (product-rule form) + microphysics source
    qc_z_total = (rho_c_z .- (q_c .* rho_d_z)) ./ rho_d
    col.uMish .= rho_d .* Kv .* qc_z_total
    Btransform!(col)
    Atransform!(col)
    VDIFF .= Ixtransform(col)

    @turbo ADV .= @. (-u * rho_cp_x) + (-w * rho_c_z) #RHO_C ADV
    @turbo FORCING .= @. (-rho_c * (u_x + w_z)) + (rho_d * (q_cond - q_auto - q_coll))
    @turbo KDIFF .= @. Khdiff * rho_cp_xx
    @turbo expdot[colstart:colend,6] .= @. ADV + FORCING + KDIFF + VDIFF
    @turbo impdot[colstart:colend,6] .= @. Kv_mudiff * rho_cp_zz

    # Rain partial-density continuity (product-rule form) + microphysics source
    qr_z_total = (rho_rp_z .- (q_r .* rho_d_z)) ./ rho_d
    col.uMish .= rho_d .* Kv .* qr_z_total
    Btransform!(col)
    Atransform!(col)
    VDIFF .= Ixtransform(col)

    @turbo ADV .= @. (-u * rho_rp_x) + (-w * rho_rp_z) #RHO_R ADV
    @turbo FORCING .= @. (-rho_r * (u_x + w_z)) + (rho_d * (q_auto + q_coll - q_evap - Vt_flux))
    @turbo KDIFF .= @. Khdiff * rho_rp_xx
    @turbo expdot[colstart:colend,7] .= @. ADV + FORCING + KDIFF + VDIFF
    @turbo impdot[colstart:colend,7] .= @. Kv_mudiff * rho_rp_zz

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

    # Solve for semi-implicit n+1 terms (linear rho_d mass-flux acoustic adjustment)
    if mtile.model.options[:semiimplicit]
        semiimplicit_adjustment_rhod(mtile, colstart, colend, t)
    end

    # Adjust the condensation rate from the advected supersaturation (entropy-density form)
    condensation_adjustment_sigma(mtile, colstart, colend, t)

    # Use implicit timestep for diffusion
    diffusion_timestep_pd(mtile, colstart, colend, t)

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
    sbar = ref_entropy(refstate)[:,1]
    sbar_z = ref_entropy(refstate)[:,2]
    sbar_zz = ref_entropy(refstate)[:,3]

    xibar = ref_xi(refstate)[:,1]
    xibar_z = ref_xi(refstate)[:,2]
    xibar_zz = ref_xi(refstate)[:,3]

    mubar = ref_mu(refstate)[:,1]
    mubar_z = ref_mu(refstate)[:,2]
    mubar_zz = ref_mu(refstate)[:,3]

    satbar = ref_sat(refstate)[:,1]
    satbar_z = ref_sat(refstate)[:,2]
    satbar_zz = ref_sat(refstate)[:,3]

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
    Pxi_bar = sound_speed_sq(mtile.ref_state)

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
    sbar = ref_entropy(refstate)[:,1]
    sbar_z = ref_entropy(refstate)[:,2]
    sbar_zz = ref_entropy(refstate)[:,3]

    xibar = ref_xi(refstate)[:,1]
    xibar_z = ref_xi(refstate)[:,2]
    xibar_zz = ref_xi(refstate)[:,3]

    mubar = ref_mu(refstate)[:,1]
    mubar_z = ref_mu(refstate)[:,2]
    mubar_zz = ref_mu(refstate)[:,3]

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
    Pxi_bar = sound_speed_sq(mtile.ref_state)

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
