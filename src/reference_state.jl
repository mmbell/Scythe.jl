# Reference state functions
using Statistics

struct ReferenceState
    sbar::Array{Float64}
    xibar::Array{Float64}
    mubar::Array{Float64}
    Pxi_bar::Float64
end

function empty_reference_state()

    ReferenceState(Array{Float64}(undef), Array{Float64}(undef), Array{Float64}(undef), 0.0)
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
    dlnpdz = -g * rho_t[1] / (p[1] * 100.0)
    for i = 2:nlevels
        lnp = log(p[i-1]) + (dlnpdz * (z[i] - z[i-1]))
        p[i] = exp(lnp)
        Tk[i] = theta[i]/(p_0/p[i])^(Rd/Cpd)
        e = vapor_pressure(p[i],q_v[i])
        rho_d[i] = 100.0 * (p[i] - e)/ (Tk[i] * Rd)
        rho_t[i] = rho_d[i] * (1.0 + q_v[i])
        dlnpdz = -g * rho_t[i] / (p[i] * 100.0)
    end

    # Re-integrate with Chebyshev column to adjust T
    cp = ChebyshevParameters(
        zmin = model.grid_params.zmin,
        zmax = model.grid_params.zmax,
        zDim = model.grid_params.zDim,
        bDim = model.grid_params.b_zDim,
        BCB = Chebyshev.R0,
        BCT = Chebyshev.R0)
    column = Chebyshev1D(cp)
    for n in 1:10
        column.uMish[:] .= -Scythe.g .* rho_t[:]
        CBtransform!(column)
        CAtransform!(column)
        p_new = CIInttransform(column, sfc_pressure * 100.0) ./ 100.0
        Tk = theta ./ (p_0 ./ p_new).^(Rd./Cpd)
        e = vapor_pressure.(p_new,q_v)
        rho_d = 100.0 .* (p_new .- e) ./ (Tk .* Rd)
        rho_t = rho_d .* (1.0 .+ q_v)
    end
    
    sbar = zeros(Float64,length(z),3)
    xibar = zeros(Float64,length(z),3)
    mubar = zeros(Float64,length(z),3)

    sbar[:,1] = entropy.(Tk, rho_d, q_v)
    xibar[:,1] = log_dry_density.(rho_d)
    mubar[:,1] = bhyp.(q_v)

    # Calculate the derivatives
    transform_reference_state!(model, sbar)
    transform_reference_state!(model, xibar)
    transform_reference_state!(model, mubar)

    # Get the mean speed of sound squared
    Pxi =  P_xi_from_s.(sbar[:,1], xibar[:,1], mubar[:,1])
    rho_bar = dry_density.(xibar[:,1])
    q_bar = ahyp.(mubar[:,1])
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

function adjust_reference_state!(model::ModelParameters, ref::ReferenceState)

    # Adjust to maintain hydrostatic balance - requires more complicated approach than below
    #for z in 1:model.grid_params.zDim
    #    q_v, rho_d, Tk, p = Scythe.thermodynamic_tuple.(ref.sbar[z,1], ref.xibar[z,1], ref.mubar[z,1])
    #    pg = Scythe.pressure_gradient(Tk, rho_d, q_v, ref.sbar[z,2],ref.xibar[z,2],ref.mubar[z,2])
    #    adj = (pg + (rho_d * 9.81))/9.81
    #    ref.xibar[z,1] = Scythe.log_dry_density(rho_d + adj)
    #end
    #
    #column.uMish[:] .= ref.xibar[:,1]
    #CBtransform!(column)
    #CAtransform!(column)
    #ref.xibar[:,1] .= CItransform!(column)
    #ref.xibar[:,2] .= CIxtransform(column)
    #ref.xibar[:,3] .= CIxxtransform(column)
end
