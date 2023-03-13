# Reference state functions
    #Tbar = Ts - (g / Cpd) * z[i]
    #pbar = p_0/(Ts/Tbar)^(Cpd/Rd)
    #rhobar = 100.0 * pbar / (Tbar * Rd)

struct ReferenceState
    sbar::Array{Float64}
    xibar::Array{Float64}
    mubar::Array{Float64}
end

function interpolate_reference_file(ref_file::String, z::Array{Float64})

    # Open the file with sounding information
    ref = open(ref_file,"r")
    
    # Allocate some empty arrays
    alt = Vector{Float64}(undef,0)
    theta = Vector{Float64}(undef,0)
    q_v = Vector{Float64}(undef,0)
    
    # Read the file
    surface = readline(ref)
    sfc_pressure = parse(Float64,split(surface)[1])
    pushfirst!(alt, 0.0)
    pushfirst!(theta, parse(Float64,split(surface)[2]))
    pushfirst!(q_v, parse(Float64,split(surface)[3]))
    while(true)
        level = readline(ref)
        if isempty(level)
            break
        end
        push!(alt, parse(Float64,split(level)[1]))
        push!(theta, parse(Float64,split(level)[2]))
        push!(q_v, parse(Float64,split(level)[3]))
    end

    # Convert to needed variables
    q_v = q_v .* 1.0e-3
    nlevels = length(alt)
    Tk = zeros(Float64,nlevels)
    p = zeros(Float64,nlevels)
    rho_d = zeros(Float64,nlevels)
    p[1] = sfc_pressure
    e = vapor_pressure(p[1],q_v[1])
    Tk[1] = theta[1]/(p_0/p[1])^(Rd/Cpd)
    rho_d[1] = 100.0 * (p[1] - e) / (Tk[1] * Rd)
    dlnpdz = -g / (Rd * Tk[1] * (1.0 + 0.61 * q_v[1]))
    for i = 2:nlevels
        lnp = log(p[i-1]) + (dlnpdz * (alt[i] - alt[i-1]))
        p[i] = exp(lnp)
        Tk[i] = theta[i]/(p_0/p[i])^(Rd/Cpd)
        e = vapor_pressure(p[i],q_v[i])
        rho_d[i] = 100.0 * (p[i] - e)/ (Tk[i] * Rd)
        dlnpdz = -g / (Rd * Tk[i] * (1.0 + 0.61 * q_v[i]))
    end
    
    s = entropy.(Tk, rho_d, q_v)
    xi = log_dry_density.(rho_d)
    mu = bhyp.(q_v)
    
    # Interpolate to model levels
    sbar = zeros(Float64,length(z),3)
    xibar = zeros(Float64,length(z),3)
    mubar = zeros(Float64,length(z),3)
    sbar[1,1] = s[1]
    xibar[1,1] = xi[1]
    mubar[1,1] = mu[1]
    for i = 2:length(z)
        for j = 2:nlevels
            if (alt[j-1] < z[i]) && (alt[j] > z[i])
                # Found the interpolating levels
                sbar[i,1] = s[j-1] + (z[i] - alt[j-1]) * (s[j] - s[j-1])/(alt[j] - alt[j-1])
                xibar[i,1] = xi[j-1] + (z[i] - alt[j-1]) * (xi[j] - xi[j-1])/(alt[j] - alt[j-1])
                mubar[i,1] = mu[j-1] + (z[i] - alt[j-1]) * (mu[j] - mu[j-1])/(alt[j] - alt[j-1])
            end
        end
    end
    
    ref_state = ReferenceState(sbar, xibar, mubar)
    return ref_state
end

function transform_reference_state!(model::ModelParameters, ref::ReferenceState)

    # Calculate vertical derivatives without BCs
    cp = ChebyshevParameters(
        zmin = model.grid_params.zmin,
        zmax = model.grid_params.zmax,
        zDim = model.grid_params.zDim,
        bDim = model.grid_params.b_zDim,
        BCB = Chebyshev.R0,
        BCT = Chebyshev.R0)
    column = Chebyshev1D(cp)
    
    column.uMish[:] .= ref.sbar[:,1]
    CBtransform!(column)
    CAtransform!(column)
    ref.sbar[:,1] .= CItransform!(column)
    ref.sbar[:,2] .= CIxtransform(column)
    ref.sbar[:,3] .= CIxxtransform(column)

    column.uMish[:] .= ref.xibar[:,1]
    CBtransform!(column)
    CAtransform!(column)
    ref.xibar[:,1] .= CItransform!(column)
    ref.xibar[:,2] .= CIxtransform(column)
    ref.xibar[:,3] .= CIxxtransform(column)

    column.uMish[:] .= ref.mubar[:,1]
    CBtransform!(column)
    CAtransform!(column)
    ref.mubar[:,1] .= CItransform!(column)
    ref.mubar[:,2] .= CIxtransform(column)
    ref.mubar[:,3] .= CIxxtransform(column)
    
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

