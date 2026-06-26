# Reference state functions
using Statistics
using LsqFit

"""
    ReferenceState

Holds the thermodynamic reference (base) state for the model, including vertical profiles
and their first and second derivatives.

# Fields
- `sbar::Array{Float64}`: moist entropy profile, size `(nlevels, 3)` with columns for value, first derivative, and second derivative [J/(kg K)]
- `xibar::Array{Float64}`: log dry air density profile, size `(nlevels, 3)` with columns for value, first derivative, and second derivative [log(kg/m^3)]
- `rhobar::Array{Float64}`: dry air density profile, size `(nlevels, 3)` with columns for value, first derivative, and second derivative [kg/m^3]. Derived from `xibar` for the linear-`rho_d` equation set; the spectral derivatives are computed on the model basis (not via the chain rule) so they are consistent with the prognostic field's auto-derivatives.
- `mubar::Array{Float64}`: transformed water vapor mixing ratio profile, size `(nlevels, 3)` with columns for value, first derivative, and second derivative
- `satbar::Array{Float64}`: transformed saturation ratio profile, size `(nlevels, 3)` with columns for value, first derivative, and second derivative
- `Pxi_bar::Float64`: domain-mean speed of sound squared [m^2/s^2]
"""
struct ReferenceState
    sbar::Array{Float64}
    xibar::Array{Float64}
    rhobar::Array{Float64}
    mubar::Array{Float64}
    satbar::Array{Float64}
    Pxi_bar::Float64
end

# ── Reference-state accessor interface ─────────────────────────────────────────
# Equation sets and microphysics read reference profiles through these accessors
# rather than reaching into struct fields directly. This decouples the call sites
# from the concrete layout so the struct can later be replaced by a physical-density
# reference type (carried in Springsteel) without touching every consumer.
#
# Each profile accessor returns the `(nlevels, 3)` array (value, 1st, 2nd vertical
# derivative); `sound_speed_sq` returns the scalar domain-mean speed of sound squared.

"""Moist entropy reference profile `(nlevels, 3)` [J/(kg K)]."""
ref_entropy(rs::ReferenceState) = rs.sbar

"""Log dry-air density reference profile `(nlevels, 3)`."""
ref_xi(rs::ReferenceState) = rs.xibar

"""Dry-air density reference profile `(nlevels, 3)` [kg/m^3]."""
ref_rho_d(rs::ReferenceState) = rs.rhobar

"""Transformed water-vapor mixing ratio reference profile `(nlevels, 3)`."""
ref_mu(rs::ReferenceState) = rs.mubar

"""Transformed saturation-ratio reference profile `(nlevels, 3)`."""
ref_sat(rs::ReferenceState) = rs.satbar

"""Domain-mean speed of sound squared [m^2/s^2]."""
sound_speed_sq(rs::ReferenceState) = rs.Pxi_bar

"""
    rhobar_from_xibar(xibar, column) -> Array{Float64}

Build the dry-air density reference profile `rhobar` (value + first/second vertical
derivative, shape `(nlevels, 3)`) from the log-density reference `xibar` by applying
`dry_density` and then computing the derivatives spectrally on `column` (via
[`transform_reference_state!`](@ref)). Used by the linear-`rho_d` equation set so the
reference density and its gradients are consistent with the model basis.
"""
function rhobar_from_xibar(xibar::Array{Float64}, column)
    rhobar = zeros(Float64, size(xibar, 1), 3)
    rhobar[:, 1] .= dry_density.(xibar[:, 1])
    transform_reference_state!(column, rhobar)
    return rhobar
end

"""
    empty_reference_state()

Create an empty [`ReferenceState`](@ref) with undefined arrays and `Pxi_bar = 0.0`.

Useful as a placeholder when a reference state is not needed (e.g., for simple test models).

# Returns
- `ReferenceState`: a reference state with uninitialized array fields.
"""
function empty_reference_state()

    ReferenceState(Array{Float64}(undef), Array{Float64}(undef), Array{Float64}(undef), Array{Float64}(undef), Array{Float64}(undef), 0.0)
end

"""
    reference_column(grid, grid_params)

Build a vertical basis column with natural (R0) boundary conditions for
reference state derivative calculations. Reference profiles can have nonzero
gradients at the domain boundaries, so the model variables' boundary
conditions must not be imposed on them (this matches the pre-migration
behavior, which always differentiated reference profiles on an R0 column).
Falls back to a copy of the first variable's column for vertical bases other
than Chebyshev.
"""
function reference_column(grid::AbstractGrid, grid_params)
    return natural_column(grid.kbasis.data[1], grid_params)
end

function natural_column(column::Chebyshev1D, grid_params)
    cp = ChebyshevParameters(
        zmin = grid_params.kMin,
        zmax = grid_params.kMax,
        zDim = grid_params.kDim,
        bDim = grid_params.b_kDim,
        BCB = Chebyshev.R0,
        BCT = Chebyshev.R0)
    return Chebyshev1D(cp)
end

function natural_column(column::Spline1D, grid_params)
    # Cubic B-spline column with natural (R0) boundary conditions, matching the
    # model's vertical spline resolution and quadrature so the mish points align.
    # Reference profiles can have nonzero boundary gradients, so the model
    # variables' wall BCs must not be imposed on them.
    sp = SplineParameters(
        xmin = grid_params.kMin,
        xmax = grid_params.kMax,
        num_cells = grid_params.kDim ÷ grid_params.mubar,
        mubar = grid_params.mubar,
        quadrature = grid_params.quadrature,
        BCL = CubicBSpline.R0,
        BCR = CubicBSpline.R0)
    return Spline1D(sp)
end

function natural_column(column, grid_params)
    return deepcopy(column)
end

"""
    warn_timestep_stability(grid_params, ts; c_nominal=340.0, target_courant=0.5)

Advisory startup check that the timestep is consistent with the vertical
resolution, aimed at catching the common mistake of raising `kDim` without
lowering `ts`. Builds a lightweight 1-D B-spline column from `grid_params` to get
the true minimum mish-point spacing `dz_min`, estimates the acoustic Courant
number `c_nominal * ts / dz_min`, and emits an `@warn` (never aborts) if it
exceeds `target_courant`. Returns the estimated Courant number, or `nothing`
when skipped.

**RiRk only.** The check is restricted to the cubic B-spline (`"RiRk"`) vertical
geometry, whose mish points are near-uniform so a single `dz_min`-based Courant
number is meaningful. It is a deliberate no-op for the Chebyshev (`"RZ"`)
geometry: boundary clustering makes `dz_min` tiny, so stable RZ runs routinely
sit at an acoustic Courant of several (the semi-implicit scheme treats vertical
acoustics implicitly), and a nominal-sound-speed threshold would fire on every
run.

The default `target_courant=0.5` is a conservative *empirical* tripwire, not a
derived stability bound: the BF02 RiRk case is stable near Co≈0.25 (kDim=100) but
blew up near Co≈0.75 (kDim=300). The true RiRk semi-implicit margin is still
under investigation, so treat this as advisory.
"""
function warn_timestep_stability(grid_params, ts::Float64;
                                 c_nominal::Float64=340.0, target_courant::Float64=0.5)
    grid_params.geometry == "RiRk" || return nothing
    z = try
        sp = SplineParameters(
            xmin = grid_params.kMin, xmax = grid_params.kMax,
            num_cells = grid_params.kDim ÷ grid_params.mubar,
            mubar = grid_params.mubar, quadrature = grid_params.quadrature,
            BCL = CubicBSpline.R0, BCR = CubicBSpline.R0)
        Spline1D(sp).mishPoints
    catch
        return nothing       # best-effort: never let a startup check break a run
    end

    dz_min = minimum(diff(sort(z)))
    courant = c_nominal * ts / dz_min
    if courant > target_courant
        suggested = target_courant * dz_min / c_nominal
        @warn "Timestep may be too large for the RiRk vertical resolution " *
              "(kDim=$(grid_params.kDim)): estimated acoustic Courant " *
              "≈ $(round(courant; digits=2)) [c≈$(round(c_nominal)) m/s, " *
              "dz_min=$(round(dz_min; digits=2)) m, ts=$(ts) s] exceeds target " *
              "$(target_courant). Consider ts ≲ $(round(suggested; digits=4)) s. " *
              "(Advisory empirical heuristic — the semi-implicit solver treats vertical acoustics implicitly.)"
    end
    return courant
end

"""
    calculate_reference_state(model::ModelParameters, z::Array{Float64}, column)

Calculate a hydrostatic reference state from a sounding file specified in `model.ref_state_file`.

The sounding file is interpolated to model levels, then re-integrated using spectral
methods to obtain a hydrostatically balanced base state. Iteratively adjusts density and
temperature to refine the balance.

# Arguments
- `model::ModelParameters`: model configuration containing the reference state file path and grid parameters.
- `z::Array{Float64}`: vertical coordinate array of model levels [m].
- `column`: a 1D spectral basis object (e.g., `Chebyshev1D` or `Spline1D`) used for spectral integration and differentiation.

# Returns
- `ReferenceState`: the computed hydrostatic reference state with entropy, log density, transformed moisture, saturation ratio profiles and their vertical derivatives.
"""
function calculate_reference_state(model::ModelParameters, z::Array{Float64}, column)

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

    # Re-integrate with spectral column to get hydrostatic balance
    nz = length(z)

    # Fit the interpolated dtheta/dz to the column and integrate it
    column.uMish[:] .= theta[:]
    Btransform!(column)
    Atransform!(column)
    theta_new = zeros(Float64, nz)
    theta_new .= IInttransform(column, theta_in[1])

    # Fit the water vapor
    q_v = q_v .* 1.0e-3
    column.uMish[:] .= q_v[:]
    Btransform!(column)
    Atransform!(column)
    q_v_new = zeros(Float64, nz)
    q_v_new .= IInttransform(column, q_v_in[1]*1.0e-3)

    mu_new = zeros(Float64, nz)
    column.uMish[:] = mu_transform.(q_v_new)
    Btransform!(column)
    Atransform!(column)
    mu_new .= Itransform!(column)
    mu_new_z = Ixtransform(column)
    mu_new_zz = Ixxtransform(column)
    q_v_new = inv_mu_transform.(mu_new)
    q_v_new_z = mu_new_z ./ dmudq.(mu_new, q_v_new)

    # Combine theta and q_v to get hydrostatic pressure and density
    theta_rho = @. theta_new * (1.0 + (q_v_new / Eps)) / (1.0 + q_v_new)
    dexnerdz = -gravity ./ (Cpd .* theta_rho)
    column.uMish[:] .= dexnerdz
    Btransform!(column)
    Atransform!(column)
    sfc_exner = (sfc_pressure/1000.0)^(Rd/Cpd)
    exner = IInttransform(column, sfc_exner)
    p_new = @. (exner^(Cpd/Rd))*1000.0
    rho_t_new = @. ((p_new * 100.0/(Rd * theta_rho))*(1000.0/p_new)^(Rd/Cpd))
    rho_d_new = rho_t_new./(1.0 .+ q_v_new)
    xi_new = log_dry_density.(rho_d_new)
    sfc_xi = xi_new[1]
    column.uMish[:] .= xi_new
    Btransform!(column)
    Atransform!(column)
    xi_new_z = Ixtransform(column)
    xi_new_zz = Ixxtransform(column)

    # Calculate the moist entropy
    Tk_new = @. (p_new - vapor_pressure(p_new, q_v_new))*100.0/(rho_d_new * Rd)
    s_new = entropy.(Tk_new, rho_d_new, q_v_new)
    column.uMish[:] .= s_new
    Btransform!(column)
    Atransform!(column)
    s_new .= Itransform!(column)
    s_new_z = Ixtransform(column)
    s_new_zz = Ixxtransform(column)
    Tk_new = temperature.(s_new, rho_d_new, q_v_new)

    # Adjust density and temperature to refine hydrostatic balance
    for n in 1:10
        Ps = P_s.(Tk_new, rho_d_new, q_v_new)
        Pxi = P_xi.(Tk_new, rho_d_new, q_v_new)
        Pqv = P_qv.(Tk_new, rho_d_new, q_v_new)
    
        xi_new_z = ((-gravity .* rho_t_new) .- (Ps .* s_new_z) .- (Pqv .* q_v_new_z)) ./ Pxi
        column.uMish[:] .= xi_new_z[:]
        Btransform!(column)
        Atransform!(column)
        xi_new = IInttransform(column, sfc_xi)
        xi_new_zz = Ixtransform(column)
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
    Btransform!(column)
    Atransform!(column)
    sat_ratio = Itransform!(column)
    sat_ratio_z = Ixtransform(column)
    sat_ratio_zz = Ixxtransform(column)

    satbar[:,1] .= sat_ratio
    satbar[:,2] .= sat_ratio_z
    satbar[:,3] .= sat_ratio_zz

    # Get the mean speed of sound squared
    Pxi =  P_xi_from_s.(sbar[:,1], xibar[:,1], mubar[:,1])
    Pxi_bar = mean(Pxi ./ (rho_d_new .* (1.0 .+ q_v_new)))
    rhobar = rhobar_from_xibar(xibar, column)
    ref_state = ReferenceState(sbar, xibar, rhobar, mubar, satbar, Pxi_bar)
    return ref_state
end

"""
    interpolate_reference_file(model::ModelParameters, z::Array{Float64}, column)

Interpolate a sounding file to model levels and compute a reference state using simple
hydrostatic integration (without spectral re-integration of the raw profiles).

Vertical derivatives are computed afterwards via [`transform_reference_state!`](@ref).

# Arguments
- `model::ModelParameters`: model configuration containing the reference state file path and grid parameters.
- `z::Array{Float64}`: vertical coordinate array of model levels [m].
- `column`: a 1D spectral basis object (e.g., `Chebyshev1D` or `Spline1D`) for computing vertical derivatives.

# Returns
- `ReferenceState`: the interpolated reference state with entropy, log density, transformed moisture, and saturation ratio profiles.
"""
function interpolate_reference_file(model::ModelParameters, z::Array{Float64}, column)

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

    # Re-integrate with spectral column to adjust T (disabled)
    #column.uMish[:] .= -gravity .* rho_t[:]
    #Btransform!(column)
    #Atransform!(column)
    #p_new = IInttransform(column, sfc_pressure * 100.0) ./ 100.0
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
    transform_reference_state!(column, sbar)
    transform_reference_state!(column, xibar)
    transform_reference_state!(column, mubar)

    # Get the mean speed of sound squared
    Pxi =  P_xi_from_s.(sbar[:,1], xibar[:,1], mubar[:,1])
    rho_bar = dry_density.(xibar[:,1])
    q_bar = inv_mu_transform.(mubar[:,1])
    Pxi_bar = mean(Pxi ./ (rho_bar .* (1.0 .+ q_bar)))

    satbar = zeros(Float64,length(z),3)
    rhobar = rhobar_from_xibar(xibar, column)
    ref_state = ReferenceState(sbar, xibar, rhobar, mubar, satbar, Pxi_bar)
    return ref_state
end

"""
    transform_reference_state!(column, ref::Array{Float64})

Compute vertical derivatives of a reference state variable in-place using spectral transforms.

Fits the values in `ref[:, 1]` to the spectral basis, then overwrites
`ref[:, 1]` with the filtered values, `ref[:, 2]` with the first vertical derivative,
and `ref[:, 3]` with the second vertical derivative.

# Arguments
- `column`: a 1D spectral basis object (e.g., `Chebyshev1D` or `Spline1D`) for vertical transforms.
- `ref::Array{Float64}`: array of size `(nlevels, 3)` where column 1 holds the variable values; columns 2 and 3 are overwritten with derivatives.

# Returns
- `ref::Array{Float64}`: the modified array (also mutated in-place).
"""
function transform_reference_state!(column, ref::Array{Float64})

    column.uMish[:] .= ref[:,1]
    Btransform!(column)
    Atransform!(column)
    ref[:,1] .= Itransform!(column)
    ref[:,2] .= Ixtransform(column)
    ref[:,3] .= Ixxtransform(column)
    return ref
end

"""
    exact_reference_state(model::ModelParameters, z::Array{Float64}, column)

Read a pre-computed reference state from a file that has already been adjusted to
hydrostatic balance. Useful for highly idealized simulations and benchmarking.

The file must contain one line per model level with columns: altitude, entropy, log density,
and transformed moisture. Vertical derivatives are computed via [`transform_reference_state!`](@ref).

# Arguments
- `model::ModelParameters`: model configuration containing the reference state file path and grid parameters.
- `z::Array{Float64}`: vertical coordinate array of model levels [m].
- `column`: a 1D spectral basis object (e.g., `Chebyshev1D` or `Spline1D`) for computing vertical derivatives.

# Returns
- `ReferenceState`: the reference state read from file with computed vertical derivatives.

# Throws
- `DomainError` if a model level does not match the corresponding level in the file.
"""
function exact_reference_state(model::ModelParameters, z::Array{Float64}, column)

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
    transform_reference_state!(column, sbar)
    transform_reference_state!(column, xibar)
    transform_reference_state!(column, mubar)

    # Get the mean speed of sound squared
    Pxi =  P_xi_from_s.(sbar[:,1], xibar[:,1], mubar[:,1])
    rho_bar = dry_density.(xibar[:,1])
    q_bar = inv_mu_transform.(mubar[:,1])
    Pxi_bar = mean(Pxi ./ (rho_bar .* (1.0 .+ q_bar)))

    satbar = zeros(Float64,length(z),3)
    rhobar = rhobar_from_xibar(xibar, column)
    ref_state = ReferenceState(sbar, xibar, rhobar, mubar, satbar, Pxi_bar)
    return ref_state
end
