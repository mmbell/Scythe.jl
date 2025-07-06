# Reference state functions
using Statistics
using LsqFit

struct ReferenceState
    sbar::Array{Float64}
    xibar::Array{Float64}
    mubar::Array{Float64}
    satbar::Array{Float64}
    Pxi_bar::Float64
end

function empty_reference_state()

    ReferenceState(Array{Float64}(undef), Array{Float64}(undef), Array{Float64}(undef), Array{Float64}(undef), 0.0)
end

function calculate_reference_state(model::ModelParameters, z::Array{Float64}, max_wavenumber::Int64 =-1)

    # Open the file with sounding information
    ref = open(model.ref_state_file,"r")
    
    # Allocate some empty arrays
    alt = Vector{Float64}(undef,0)
    theta_in = Vector{Float64}(undef,0)
    q_v_in = Vector{Float64}(undef,0)
    
    # Read the file
    surface = readline(ref)
    sfc_pressure = parse(Float64,split(surface)[1])
    pushfirst!(alt, 0.0)
    pushfirst!(theta_in, parse(Float64,split(surface)[2]))
    pushfirst!(q_v_in, parse(Float64,split(surface)[3]))
    while(true)
        level = readline(ref)
        if isempty(level)
            break
        end
        push!(alt, parse(Float64,split(level)[1]))
        push!(theta_in, parse(Float64,split(level)[2]))
        push!(q_v_in, parse(Float64,split(level)[3]))
    end

    # Calculate the vertical derivative
    qvdz = zeros(Float64,length(alt))
    thetadz = zeros(Float64,length(alt))
    qvdz[1] = (q_v_in[2] - q_v_in[1]) / alt[2]
    thetadz[1] = (theta_in[2] - theta_in[1]) / alt[2]
    for i = 2:(length(alt)-1)
        qvdz[i] = (q_v_in[i+1] - q_v_in[i-1]) / (alt[i+1] - alt[i-1])
        thetadz[i] = (theta_in[i+1] - theta_in[i-1]) / (alt[i+1] - alt[i-1])
    end
    qvdz[end] = (q_v_in[end] - q_v_in[end-1]) / (alt[end] - alt[end-1])
    thetadz[end] = (theta_in[end] - theta_in[end-1]) / (alt[end] - alt[end-1])

    # Convert q_v_in to log form
    #q_v_in = mu_transform.(q_v_in * 1.0e-3)  # Convert to kg/kg

    # Interpolate to model levels
    theta = zeros(Float64,length(z))
    q_v = zeros(Float64,length(z))

    # Assumes first level in both cases is the surface
    theta[1] = thetadz[1]
    q_v[1] = qvdz[1]

    for i = 2:length(z)
        found = false
        for j = 2:length(alt)
            if (alt[j-1] < z[i]) && (alt[j] > z[i])
                # Found the interpolating levels
                theta[i] = thetadz[j-1] + (z[i] - alt[j-1]) * (thetadz[j] - thetadz[j-1])/(alt[j] - alt[j-1])
                q_v[i] = qvdz[j-1] + (z[i] - alt[j-1]) * (qvdz[j] - qvdz[j-1])/(alt[j] - alt[j-1])
                found = true
            elseif alt[j] == z[i]
                # Model level and reference level are the same
                theta[i] = thetadz[j]
                q_v[i] = qvdz[j]
                found = true
            end
        end
        if !found
            # Can't find the level
            throw(DomainError(i, "Can't find an interpolating level for reference state"))
        end
    end

    # Re-integrate with Chebyshev column to get hydrostatic balance
    # If max_wavenumber is specified then use that, otherwise use the model configuration
    if (max_wavenumber > 0)
        b_zDim = max_wavenumber
    else
        b_zDim = model.grid_params.b_zDim
    end
    cp = ChebyshevParameters(
        zmin = model.grid_params.zmin,
        zmax = model.grid_params.zmax,
        zDim = model.grid_params.zDim,
        bDim = b_zDim,
        BCB = Chebyshev.R0,
        BCT = Chebyshev.R0)
    column = Chebyshev1D(cp)

    # Fit the interpolated dtheta/dz to a Chebyshev column and integrate it
    column.uMish[:] .= theta[:]
    CBtransform!(column)
    CAtransform!(column)
    theta_new = zeros(Float64, cp.zDim)
    #theta_new .= CItransform!(column)
    theta_new .= CIInttransform(column, theta_in[1])

    # Fit the water vapor
    q_v = q_v .* 1.0e-3
    #mu = mu_transform.(q_v)
    column.uMish[:] .= q_v[:]
    CBtransform!(column)
    CAtransform!(column)
    q_v_new = zeros(Float64, cp.zDim)
    q_v_new .= CIInttransform(column, q_v_in[1]*1.0e-3)

    mu_new = zeros(Float64, cp.zDim)
    column.uMish[:] = mu_transform.(q_v_new)
    CBtransform!(column)
    CAtransform!(column)
    mu_new .= CItransform!(column)
    mu_new_z = CIxtransform(column)
    mu_new_zz = CIxtransform(column)
    q_v_new = inv_mu_transform.(mu_new)
    q_v_new_z = mu_new_z ./ dmudq.(mu_new, q_v_new)

    # Combine theta and q_v to get hydrostatic pressure and density
    theta_rho = @. theta_new * (1.0 + (q_v_new / Eps)) / (1.0 + q_v_new)
    dexnerdz = -gravity ./ (Cpd .* theta_rho)
    column.uMish[:] .= dexnerdz
    CBtransform!(column)
    CAtransform!(column)
    sfc_exner = (sfc_pressure/1000.0)^(Rd/Cpd)
    exner = CIInttransform(column, sfc_exner)
    p_new = @. (exner^(Cpd/Rd))*1000.0
    rho_t_new = @. ((p_new * 100.0/(Rd * theta_rho))*(1000.0/p_new)^(Rd/Cpd))
    rho_d_new = rho_t_new./(1.0 .+ q_v_new)
    xi_new = log_dry_density.(rho_d_new)
    sfc_xi = xi_new[1]
    column.uMish[:] .= xi_new
    CBtransform!(column)
    CAtransform!(column)
    xi_new_z = CIxtransform(column)
    xi_new_zz = CIxxtransform(column)

    # Calculate the moist entropy
    Tk_new = @. (p_new - vapor_pressure(p_new, q_v_new))*100.0/(rho_d_new * Rd)
    s_new = entropy.(Tk_new, rho_d_new, q_v_new)
    column.uMish[:] .= s_new
    CBtransform!(column)
    CAtransform!(column)
    s_new .= CItransform!(column)
    s_new_z = CIxtransform(column)
    s_new_zz = CIxtransform(column)
    Tk_new = temperature.(s_new, rho_d_new, q_v_new)

    # Adjust density and temperature to refine hydrostatic balance
    for n in 1:10
        Ps = P_s.(Tk_new, rho_d_new, q_v_new)
        Pxi = P_xi.(Tk_new, rho_d_new, q_v_new)
        Pqv = P_qv.(Tk_new, rho_d_new, q_v_new)
    
        xi_new_z = ((-gravity .* rho_t_new) .- (Ps .* s_new_z) .- (Pqv .* q_v_new_z)) ./ Pxi
        column.uMish[:] .= xi_new_z[:]
        CBtransform!(column)
        CAtransform!(column)
        xi_new = CIInttransform(column, sfc_xi)
        xi_new_zz = CIxtransform(column)
        rho_d_new = dry_density.(xi_new)
        rho_t_new = rho_d_new .* (1.0 .+ q_v_new)
        Tk_new = temperature.(s_new, rho_d_new, q_v_new)
        #res = -Scythe.pressure_gradient.(Tk_new, rho_d_new, q_v_new, s_new_z, xi_new_z, q_v_new_z) .+ (-Scythe.gravity .* rho_t_new)
        #println("$(Tk_new[1]), $(s_new[1]), $(res[1]) : $(Tk_new[50]), $(s_new[50]), $(res[50])")
    end

    sbar = zeros(Float64,length(z),3)
    xibar = zeros(Float64,length(z),3)
    mubar = zeros(Float64,length(z),3)
    satbar = zeros(Float64,length(z),3)

    sbar[:,1] .= s_new
    sbar[:,2] .= s_new_z
    sbar[:,3] .= s_new_zz

    xibar[:,1] .= xi_new
    xibar[:,2] .= xi_new_z
    xibar[:,3] .= xi_new_zz

    mubar[:,1] .= mu_new
    mubar[:,2] .= mu_new_z
    mubar[:,3] .= mu_new_zz

    # Calculate the saturation ratio
    thermo = thermodynamic_tuple.(sbar[:,1], xibar[:,1], mubar[:,1])
    T_bar = [x[3] for x in thermo]     # Temperature in K
    p_bar = [x[4] for x in thermo]      # Total air pressure
    q_bar = [x[1] for x in thermo]
    q_sat = q_sat_liquid.(T_bar, p_bar)
    column.uMish[:] .= mu_transform.(q_bar ./ q_sat)
    CBtransform!(column)
    CAtransform!(column)
    sat_ratio = CItransform!(column)
    sat_ratio_z = CIxtransform(column)
    sat_ratio_zz = CIxxtransform(column)

    satbar[:,1] .= sat_ratio
    satbar[:,2] .= sat_ratio_z
    satbar[:,3] .= sat_ratio_zz

    # Get the mean speed of sound squared
    Pxi =  P_xi_from_s.(sbar[:,1], xibar[:,1], mubar[:,1])
    Pxi_bar = mean(Pxi ./ (rho_d_new .* (1.0 .+ q_v_new)))
    ref_state = ReferenceState(sbar, xibar, mubar, satbar, Pxi_bar)
    return ref_state
end

function interpolate_reference_file(model::ModelParameters, z::Array{Float64})

    # Open the file with sounding information
    ref = open(model.ref_state_file,"r")
    
    # Allocate some empty arrays
    alt = Vector{Float64}(undef,0)
    theta_in = Vector{Float64}(undef,0)
    q_v_in = Vector{Float64}(undef,0)
    
    # Read the file
    surface = readline(ref)
    sfc_pressure = parse(Float64,split(surface)[1])
    pushfirst!(alt, 0.0)
    pushfirst!(theta_in, parse(Float64,split(surface)[2]))
    pushfirst!(q_v_in, parse(Float64,split(surface)[3]))
    while(true)
        level = readline(ref)
        if isempty(level)
            break
        end
        push!(alt, parse(Float64,split(level)[1]))
        push!(theta_in, parse(Float64,split(level)[2]))
        push!(q_v_in, parse(Float64,split(level)[3]))
    end

    # Interpolate to model levels
    theta = zeros(Float64,length(z))
    q_v = zeros(Float64,length(z))

    # Assumes first level in both cases is the surface
    theta[1] = theta_in[1]
    q_v[1] = q_v_in[1]

    for i = 2:length(z)
        found = false
        for j = 2:length(alt)
            if (alt[j-1] < z[i]) && (alt[j] > z[i])
                # Found the interpolating levels
                theta[i] = theta_in[j-1] + (z[i] - alt[j-1]) * (theta_in[j] - theta_in[j-1])/(alt[j] - alt[j-1])
                q_v[i] = q_v_in[j-1] + (z[i] - alt[j-1]) * (q_v_in[j] - q_v_in[j-1])/(alt[j] - alt[j-1])
                found = true
            elseif alt[j] == z[i]
                # Model level and reference level are the same
                theta[i] = theta_in[j]
                q_v[i] = q_v_in[j]
                found = true
            end
        end
        if !found
            # Can't find the level
            throw(DomainError(i, "Can't find an interpolating level for reference state"))
        end
    end

    # Convert to needed variables and do a hydrostatic integration
    q_v = q_v .* 1.0e-3
    nlevels = length(z)
    Tk = zeros(Float64,nlevels)
    p = zeros(Float64,nlevels)
    rho_d = zeros(Float64,nlevels)
    rho_t = zeros(Float64,nlevels)

    p[1] = sfc_pressure
    e = vapor_pressure(p[1],q_v[1])
    Tk[1] = theta[1]/(p_0/p[1])^(Rd/Cpd)
    rho_d[1] = 100.0 * (p[1] - e) / (Tk[1] * Rd)
    rho_t[1] = rho_d[1] * (1.0 + q_v[1])
    dlnpdz = -gravity * rho_t[1] / (p[1] * 100.0)
    for i = 2:nlevels
        lnp = log(p[i-1]) + (dlnpdz * (z[i] - z[i-1]))
        p[i] = exp(lnp)
        Tk[i] = theta[i]/(p_0/p[i])^(Rd/Cpd)
        e = vapor_pressure(p[i],q_v[i])
        rho_d[i] = 100.0 * (p[i] - e)/ (Tk[i] * Rd)
        rho_t[i] = rho_d[i] * (1.0 + q_v[i])
        dlnpdz = -gravity * rho_t[i] / (p[i] * 100.0)
    end

    # Re-integrate with Chebyshev column to adjust T
    #cp = ChebyshevParameters(
    #    zmin = model.grid_params.zmin,
    #    zmax = model.grid_params.zmax,
    #    zDim = model.grid_params.zDim,
    #    bDim = model.grid_params.b_zDim,
    #    BCB = Chebyshev.R0,
    #    BCT = Chebyshev.R0)
    #column = Chebyshev1D(cp)
    #column.uMish[:] .= -gravity .* rho_t[:]
    #CBtransform!(column)
    #CAtransform!(column)
    #p_new = CIInttransform(column, sfc_pressure * 100.0) ./ 100.0
    #Tk = theta ./ (p_0 ./ p_new).^(Rd./Cpd)
    #e = vapor_pressure.(p_new,q_v)
    #rho_d = 100.0 .* (p_new .- e) ./ (Tk .* Rd)
    #rho_t = rho_d .* (1.0 .+ q_v)
    
    sbar = zeros(Float64,length(z),3)
    xibar = zeros(Float64,length(z),3)
    mubar = zeros(Float64,length(z),3)

    sbar[:,1] = entropy.(Tk, rho_d, q_v)
    xibar[:,1] = log_dry_density.(rho_d)
    mubar[:,1] = mu_transform.(q_v)

    # Calculate the derivatives
    transform_reference_state!(model, sbar)
    transform_reference_state!(model, xibar)
    transform_reference_state!(model, mubar)

    # Get the mean speed of sound squared
    Pxi =  P_xi_from_s.(sbar[:,1], xibar[:,1], mubar[:,1])
    rho_bar = dry_density.(xibar[:,1])
    q_bar = inv_mu_transform.(mubar[:,1])
    Pxi_bar = mean(Pxi ./ (rho_bar .* (1.0 .+ q_bar)))

    ref_state = ReferenceState(sbar, xibar, mubar, Pxi_bar)
    return ref_state
end

function transform_reference_state!(model::ModelParameters, ref::Array{Float64})

    # Calculate vertical derivatives without BCs
    cp = ChebyshevParameters(
        zmin = model.grid_params.zmin,
        zmax = model.grid_params.zmax,
        zDim = model.grid_params.zDim,
        bDim = model.grid_params.b_zDim,
        BCB = Chebyshev.R0,
        BCT = Chebyshev.R0)
    column = Chebyshev1D(cp)
    
    column.uMish[:] .= ref[:,1]
    CBtransform!(column)
    CAtransform!(column)
    ref[:,1] .= CItransform!(column)
    ref[:,2] .= CIxtransform(column)
    ref[:,3] .= CIxxtransform(column)
    return ref
end

function exact_reference_state(model::ModelParameters, z::Array{Float64})

    # Read a reference state file that has already been adjusted to hydrostatic balance
    # This function is useful for highly idealized simulations and benchmarking

    # Open the file with sounding information
    ref = open(model.ref_state_file,"r")

    # Allocate some empty arrays
    sbar = zeros(Float64,length(z),3)
    xibar = zeros(Float64,length(z),3)
    mubar = zeros(Float64,length(z),3)

    # Read the file
    for i = 1:length(z)
        lineparts = split(readline(ref))
        if lineparts[1] != string(z[i])
            throw(DomainError(i, "Model level does not match reference level"))
        end
        sbar[i,1] = parse(Float64,lineparts[2])
        xibar[i,1] = parse(Float64,lineparts[3])
        mubar[i,1] = parse(Float64,lineparts[4])
    end

    # Calculate the derivatives
    transform_reference_state!(model, sbar)
    transform_reference_state!(model, xibar)
    transform_reference_state!(model, mubar)

    # Get the mean speed of sound squared
    Pxi =  P_xi_from_s.(sbar[:,1], xibar[:,1], mubar[:,1])
    rho_bar = dry_density.(xibar[:,1])
    q_bar = inv_mu_transform.(mubar[:,1])
    Pxi_bar = mean(Pxi ./ (rho_bar .* (1.0 .+ q_bar)))

    ref_state = ReferenceState(sbar, xibar, mubar, Pxi_bar)
    return ref_state
end
