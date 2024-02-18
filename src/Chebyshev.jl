module Chebyshev

using LinearAlgebra
using FFTW

export ChebyshevParameters, Chebyshev1D
export CBtransform, CBtransform!, CAtransform!, CItransform!
export CBxtransform, CIxtransform, CIxxtransform, CIInttransform

#Define some convenient aliases
const real = Float64
const int = Int64
const uint = UInt64

# Define homogeneous boundary conditions
# Inhomgoneous conditions to be implemented later
const R0 = Dict("R0" => 0)
const R1T0 = Dict("α0" =>  0.0)
const R1T1 = Dict("α1" =>  0.0) 
const R1T2 = Dict("α2" =>  0.0)
const R2T10 = Dict("β1" => 0.0, "β2" => 0.0)
const R2T20 = Dict("β1" => 0.0, "β2" => 0.0)
const R3 = Dict("R3" => 0)

# Define the spline parameters
Base.@kwdef struct ChebyshevParameters
    zmin::real = 0.0   # Minimum z in meters
    zmax::real = 0.0   # Maximum z in meters
    zDim::int = 0      # Nodal dimension
    bDim::int = 0      # Spectral dimension
    BCB::Dict = R0     # Bottom boundary condition
    BCT::Dict = R0     # Top boundary condition
end

struct Chebyshev1D
    # Parameters for the column
    params::ChebyshevParameters
    
    # Pre-calculated Chebyshev–Gauss–Lobatto points (extrema of Chebyshev polynomials)
    mishPoints::Vector{real}
    
    # Scalar, vector, or matrix that enforces boundary conditions
    gammaBC::Array{real}
    
    # Measured FFTW Plan
    fftPlan::FFTW.r2rFFTWPlan
    
    # Filter matrix
    filter::Matrix{real}

    # uMish is the physical values
    # b is the filtered Chebyshev coefficients without BCs
    # a is the padded Chebyshev coefficients with BCs
    # ax is a buffer for derivative and integral coefficients
    uMish::Vector{real}
    b::Vector{real}
    a::Vector{real}
    ax::Vector{real}
end

function Chebyshev1D(cp::ChebyshevParameters)

    # Constructor for 1D Chebsyshev structure
    mishPoints = calcMishPoints(cp)
    gammaBC = calcGammaBC(cp)

    # Initialize the arrays
    uMish = zeros(real,cp.zDim)
    b = zeros(real,cp.bDim)
    a = zeros(real,cp.zDim)
    ax = zeros(real,cp.zDim)

    # Plan the FFT
    # From the FFTW documentation: FFTW_REDFT00 (DCT-I): even around j=0 and even around j=n-1.
    # If you specify a size-5 REDFT00 (DCT-I) of the data abcde, it corresponds to the DFT of the logical even array abcdedcb of size 8
    fftPlan = FFTW.plan_r2r(a, FFTW.REDFT00, flags=FFTW.PATIENT)

    # Pre-calculate the filter matrix
    filter = calcFilterMatrix(cp)

    # Construct a 1D Chebyshev column
    column = Chebyshev1D(cp,mishPoints,gammaBC,fftPlan,filter,uMish,b,a,ax)
    return column
end

function calcMishPoints(cp::ChebyshevParameters)

    # Calculate the physical Chebyshev points
    # The points are evenly spaced in the interval (0,π)
    # which are then mapped to the physical interval (-1,1) via the cosine function
    # and then scaled and offset to match the physical domain from zmin to zmax
    Nbasis = cp.zDim
    z = zeros(real,Nbasis)
    scale = -0.5 * (cp.zmax - cp.zmin)
    offset = 0.5 * (cp.zmin + cp.zmax)
    for n = 1:Nbasis
        z[n] = cos((n-1) * π / (Nbasis - 1)) * scale + offset
    end
    return z
end

function calcFilterMatrix(cp::ChebyshevParameters)

    # Create a matrix to truncate the coefficients to bDim after Fourier transformation
    filter = Matrix(1.0I, cp.bDim, cp.zDim)
    return filter
end

function CBtransform(cp::ChebyshevParameters, fftPlan, uMish::Vector{real})

    # Do the DCT transform and pre-scale the output based on the physical length
    b = (fftPlan * uMish) ./ (2 * (cp.zDim -1))
    return b[1:cp.bDim]
end

function CBtransform!(column::Chebyshev1D)

    # Do an in-place DCT transform
    b = (column.fftPlan * column.uMish) ./ (2 * (column.params.zDim -1))
    column.b .= column.filter * b
end

function CBtransform(column::Chebyshev1D, uMish::Vector{real})

    # Do the DCT transform and pre-scale the output based on the physical length
    b = (column.fftPlan * uMish) ./ (2 * (column.params.zDim -1))
    return column.filter * b
end

function CAtransform(cp::ChebyshevParameters, gammaBC, b::Vector{real})

    # Apply the boundary conditions and pad the coefficients with zeros back to zDim
    bfill = [b ; zeros(Float64, cp.zDim-cp.bDim)]
    a = bfill .+ (gammaBC' * bfill)
    return a
end

function CAtransform!(column::Chebyshev1D)

    # In place CA transform
    bfill = column.filter' * column.b
    a = bfill .+ (column.gammaBC' * bfill)
    column.a .= a
end
    
function CItransform(cp::ChebyshevParameters, fftPlan, a::Vector{real})

    # Do the inverse DCT transform to get back physical values
    uMish = fftPlan * a
    return uMish
end

function CItransform!(column::Chebyshev1D)
    
    # In-place inverse DCT transform to get back physical values
    column.uMish .= column.fftPlan * column.a
end


function CIIntcoefficients(cp::ChebyshevParameters, a::Vector{real}, C0::real = 0.0)

    # Calculate the integral coefficients using a recursive relationship
    # C0 is an optional constant of integration
    aInt = zeros(real,cp.zDim)
    sum = 0.0
    interval = -0.25 * (cp.zmax - cp.zmin)
    for k = 2:(cp.zDim-1)
        aInt[k] = interval * (a[k-1] - a[k+1]) / (k-1)
        sum += aInt[k]
    end
    aInt[cp.zDim] = a[cp.zDim-1]/(cp.zDim-1)
    sum += aInt[cp.zDim]
    aInt[1] = C0 - (2.0 * sum)
    return aInt
end

function CIInttransform(cp::ChebyshevParameters, fftPlan, a::Vector{real}, C0::real = 0.0)

    # Do a transform to get the integral of the column values
    # C0 is an optional constant of integration
    aInt = CIIntcoefficients(cp,a,C0)
    uInt = fftPlan * aInt
    return uInt
end

function CIInttransform(column::Chebyshev1D, C0::real = 0.0)

    # Do a transform to get the integral of the column values
    uInt = CIInttransform(column.params, column.fftPlan, column.a, C0)
    return uInt
end

function CIxcoefficients(cp::ChebyshevParameters, a::Vector{real}, ax::Vector{real})

    # Calculate the derivative coefficients using a recursive relationship
    k = cp.zDim
    ax[k-1] = (2.0 * (k-1) * a[k])
    for k = (cp.zDim-1):-1:2
        ax[k-1] = (2.0 * (k-1) * a[k]) + ax[k+1]
    end
    ax ./= (-0.5 * (cp.zmax - cp.zmin))
    return ax
end

function CIxtransform(cp::ChebyshevParameters, fftPlan, a::Vector{real}, ax::Vector{real})
    
    # Do the inverse transform to get back the first derivative in physical space
    ux = fftPlan * CIxcoefficients(cp,a,ax)
    return ux
end

function CIxtransform(column::Chebyshev1D)

    # Do the inverse transform to get back the first derivative in physical space
    ux = CIxtransform(column.params, column.fftPlan, column.a, column.ax)
    return ux
end

function CIxxtransform(column::Chebyshev1D)

    # Do the inverse transform to get back the second derivative in physical space
    ax = copy(CIxcoefficients(column.params, column.a, column.ax))
    uxx = CIxtransform(column.params, column.fftPlan, ax, column.ax)
    return uxx
end


function dct_matrix(Nbasis::Int64)
    
    # Create a matrix with the DCT as basis functions
    # This function is used for debugging and also for linear solvers
    dct = zeros(Float64,Nbasis,Nbasis)
    for i = 1:Nbasis
        t = (i-1) * π / (Nbasis - 1)
        for j = 1:Nbasis
            dct[i,j] = 2*cos((j-1)*t)
        end
    end
    dct[:,1] *= 0.5
    dct[:,Nbasis] *= 0.5
    return dct
end

function dct_1st_derivative(Nbasis::Int64, physical_length::Float64)

    # Create a 1st derivative matrix with the DCT as basis functions
    # This function is used for debugging and also for linear solvers
    dct = zeros(Float64,Nbasis,Nbasis)
    for i = 1:Nbasis
        t = (i-1) * π / (Nbasis - 1)
        for j = 1:Nbasis
            N = j-1
            if (i == 1)
                dct[i,j] = -N*N
            elseif (i == Nbasis)
                dct[i,j] = -N*N*(-1.0)^(N+1)
            else
                dct[i,j] = -N*sin(N*t)/sin(t)
            end
        end
    end
    return dct ./ (physical_length/4.0)
end

function dct_2nd_derivative(Nbasis::Int64, physical_length::Float64)

    # Create a 2nd derivative matrix with the DCT as basis functions
    # This function is used for debugging and also for linear solvers
    dct = zeros(Float64,Nbasis,Nbasis)
    for i = 1:Nbasis
        t = (i-1) * π / (Nbasis - 1)
        ct = cos(t)
        st = sin(t)
        for j = 1:Nbasis
            N = j-1
            if (i == 1)
                dct[i,j] = (N^4 - N^2)/3
            elseif (i == Nbasis)
                dct[i,j] = ((-1.0)^N)*(N^4 - N^2)/3
            else
                dct[i,j] = -N*N*cos(N*t)/(st*st) + N*sin(N*t)*ct/(st*st*st)
            end
        end
    end
    return dct ./ (physical_length^2/8.0)
end

function calcGammaBCalt(cp::ChebyshevParameters)
    
    # This works for Dirichelet BCs, but not for Neumann
    # It's also less efficient than the other methods, but not ready to delete this code yet
    Ndim = cp.zDim
    
    if (cp.BCB == R0) && (cp.BCT == R0)
        # Identity matrix
        gammaBC = Matrix(1.0I, Ndim, Ndim)
        return factorize(gammaBC)
    end
    
    # Create the BC matrix
    dctMatrix = calcDCTmatrix(Ndim)
    dctBC = calcDCTmatrix(Ndim)
    
    if haskey(cp.BCB,"α0")
        dctBC[:,1] .= cp.BCB["α0"]
    elseif haskey(cp.BCB,"α1")
        #Not implemented yet
    end

    if haskey(cp.BCT,"α0")
        dctBC[:,Ndim] .= cp.BCT["α0"]
    elseif haskey(cp.BCT,"α1")
        #Not implemented yet
    end
    
    gammaTranspose = dctMatrix' \ dctBC'    
    gammaBC = Matrix{Float64}(undef,25,25)
    gammaBC .= gammaTranspose'
    return gammaBC
end

function calcGammaBC(cp::ChebyshevParameters)

    # Calculate a matrix to apply the Neumann and Dirichelet BCs
    # The nomenclature follows Ooyama (2002) to match the cubic b-spline designations
    Ndim = cp.zDim
    
    if (cp.BCB == R0) && (cp.BCT == R0)
        # No BCs
        gammaBC = zeros(Float64,Ndim)
        return gammaBC
    
    elseif (cp.BCB == R1T0) && (cp.BCT == R0)
        #R1T0 bottom
        gammaBC = ones(Float64,Ndim)
        gammaBC[2:Ndim-1] *= 2.0
        gammaBC *= (-0.5 / (Ndim-1))
        return gammaBC
        
    elseif (cp.BCB == R1T1) && (cp.BCT == R0)
        #R1T1 bottom
        # Global coefficient method (Wang et al. 1993) for Neumann BCs
        # https://doi.org/10.1006/jcph.1993.1133
        scaleFactor = 0.0
        gammaBC = zeros(Float64,Ndim,Ndim)
        c = ones(Float64,Ndim)
        c[1] *= 2.0
        c[Ndim] *= 2.0
        for i = 1:Ndim
            n = i-1
            scaleFactor += -(n * n) / (c[i] * (Ndim-1))
        end
        for i = 1:Ndim
            n = i -1
            for j = 1:Ndim
                gammaBC[i,j] = n * n / (scaleFactor * c[j] * (Ndim-1))
            end
        end
        return gammaBC
    
    elseif (cp.BCB == R0) && (cp.BCT == R1T0)
        gammaBC = ones(Float64,Ndim,Ndim)
        for i = 1:Ndim
            for j = 1:Ndim
                gammaBC[i,j] *= -1.0* (-1.0)^(i-1) * (-1.0)^(j-1) / (Ndim-1)
            end
        end
        gammaBC[1,:] *= 0.5
        gammaBC[Ndim,:] *= 0.5
        return gammaBC
        
    elseif (cp.BCB == R0) && (cp.BCT == R1T1)    
        scaleFactor = 0.0
        gammaBC = zeros(Float64,Ndim,Ndim)
        c = ones(Float64,Ndim)
        c[1] *= 2.0
        c[Ndim] *= 2.0
        for i = 1:Ndim
            n = i-1
            scaleFactor += -(n * n) * (-1.0)^n * (-1.0)^(n+1) / (c[i] * (Ndim-1))
        end
        for i = 1:Ndim
            n = i -1
            for j = 1:Ndim
                gammaBC[i,j] = (-1.0)^(j+1) * (-1.0)^(n+1) * n * n / (scaleFactor * c[j] * (Ndim-1))
            end
        end
        return gammaBC
        
    elseif (cp.BCB == R1T0) && (cp.BCT == R1T0)
        gammaBC = ones(Float64,Ndim,Ndim)
        for i = 1:Ndim
            for j = 1:Ndim
                gammaBC[i,j] *= -1.0* (-1.0)^(i-1) * (-1.0)^(j-1) / (Ndim-1)
            end
        end
        gammaBC[1,:] *= 0.5
        gammaBC[Ndim,:] *= 0.5

        gammaL = ones(Float64,Ndim)
        gammaL[2:Ndim-1] *= 2.0
        gammaL *= (-0.5 / (Ndim-1))

        for j = 1:Ndim
            gammaBC[:,j] += gammaL
        end
        return gammaBC  
        
    elseif (cp.BCB == R1T1) && (cp.BCT == R1T1)
        scaleFactor = 0.0
        gammaBCB = zeros(Float64,Ndim,Ndim)
        c = ones(Float64,Ndim)
        c[1] *= 2.0
        c[Ndim] *= 2.0
        for i = 1:Ndim
            n = i-1
            scaleFactor += -(n * n) / (c[i] * (Ndim-1))
        end
        for i = 1:Ndim
            n = i -1
            for j = 1:Ndim
                gammaBCB[i,j] = n * n / (scaleFactor * c[j] * (Ndim-1))
            end
        end
        
        scaleFactor = 0.0
        gammaBCT = zeros(Float64,Ndim,Ndim)
        for i = 1:Ndim
            n = i-1
            scaleFactor += -(n * n) * (-1.0)^n * (-1.0)^(n+1) / (c[i] * (Ndim-1))
        end
        for i = 1:Ndim
            n = i -1
            for j = 1:Ndim
                gammaBCT[i,j] = (-1.0)^(j+1) * (-1.0)^(n+1) * n * n / (scaleFactor * c[j] * (Ndim-1))
            end
        end

        gammaBC = zeros(Float64,Ndim,Ndim)
        gammaBC .= gammaBCB + gammaBCT
        return gammaBC
    
    elseif (cp.BCB == R1T0) && (cp.BCT == R1T1)
        c = ones(Float64,Ndim)
        c[1] *= 2.0
        c[Ndim] *= 2.0
        scaleFactor = 0.0
        gammaBCT = zeros(Float64,Ndim,Ndim)
        for i = 1:Ndim
            n = i-1
            scaleFactor += -(n * n) * (-1.0)^n * (-1.0)^(n+1) / (c[i] * (Ndim-1))
        end
        for i = 1:Ndim
            n = i -1
            for j = 1:Ndim
                gammaBCT[i,j] = (-1.0)^(j+1) * (-1.0)^(n+1) * n * n / (scaleFactor * c[j] * (Ndim-1))
            end
        end
        
        gammaBCB = ones(Float64,Ndim)
        gammaBCB[2:Ndim-1] *= 2.0
        gammaBCB *= (-0.5 / (Ndim-1))

        gammaBC = zeros(Float64,Ndim,Ndim)
        gammaBC .= gammaBCB .+ gammaBCT
        return gammaBC

    elseif (cp.BCB == R1T1) && (cp.BCT == R1T0)
        c = ones(Float64,Ndim)
        c[1] *= 2.0
        c[Ndim] *= 2.0
        scaleFactor = 0.0
        gammaBCB = zeros(Float64,Ndim,Ndim)
        c = ones(Float64,Ndim)
        c[1] *= 2.0
        c[Ndim] *= 2.0
        for i = 1:Ndim
            n = i-1
            scaleFactor += -(n * n) / (c[i] * (Ndim-1))
        end
        for i = 1:Ndim
            n = i -1
            for j = 1:Ndim
                gammaBCB[i,j] = n * n / (scaleFactor * c[j] * (Ndim-1))
            end
        end

        gammaBCT = ones(Float64,Ndim,Ndim)
        for i = 1:Ndim
            for j = 1:Ndim
                gammaBCT[i,j] *= -1.0* (-1.0)^(i-1) * (-1.0)^(j-1) / (Ndim-1)
            end
        end
        gammaBCT[1,:] *= 0.5
        gammaBCT[Ndim,:] *= 0.5

        gammaBC = zeros(Float64,Ndim,Ndim)
        gammaBC .= gammaBCB .+ gammaBCT
        return gammaBC
    else
        bcs = "$(cp.BCB), $(cp.BCT)"
        throw(DomainError(bcs, "Chebyshev boundary condition combination not implemented"))
    end

end

function bvp(N::Int64, u::Array{Float64}, ux::Array{Float64}, uxx::Array{Float64},
        f::Array{Float64}, d0::Array{Float64}, d1::Array{Float64}, d2::Array{Float64},
        scale::Float64 = 1.0, alpha::Float64 = 0.0, beta::Float64 = 0.0)

    # Modified Chebyshev boundary value problem solver from Boyd (2000)
    # Useful for testing the DCT matrices defined above
    nbasis = N-2
    xi = zeros(Float64, nbasis)
    g = zeros(Float64, nbasis)
    phi = zeros(Float64, nbasis)
    phix = zeros(Float64, nbasis)
    phixx = zeros(Float64, nbasis)
    h = zeros(Float64, nbasis, nbasis)
    # Compute the interior collocation points and forcing vector G
    for i in 1:nbasis
        xi[i] = cos(π*i/(nbasis+1))
        x = xi[i]
        b = alpha*(1-x)/2.0 + beta*(1+x)/2.0
        bx = (-alpha + beta)/2.0
        g[i] = f[i+1] - d0[i+1]*b - d1[i+1]*bx
    end

    # Compute the LHS square matrix
    for i in 1:nbasis
        x = xi[i]
        phi, phix, phixx = bvp_modified_basis(x, nbasis, phi, phix, phixx, scale)
        for j in 1:nbasis
            h[i,j] = d2[i+1]*phixx[j] + d1[i+1]*phix[j] + d0[i+1]*phi[j]
        end
    end

    # Solve the linear equation set
    aphi = h \ g

    # Transform back to physical space
    u[1] = beta
    u[N] = alpha
    ux[1] = (-alpha + beta)/2.0
    ux[N] = (-alpha + beta)/2.0
    uxx[1] = 0.0
    uxx[N] = 0.0
    x = 1.0
    phi, phix, phixx = bvp_modified_basis(x, nbasis, phi, phix, phixx, scale)
    for j in 1:nbasis
        u[1] = u[1] + (aphi[j] * phi[j])
        ux[1] = ux[1] + (aphi[j] * phix[j])
        uxx[1] = uxx[1] + (aphi[j] * phixx[j])
    end
    x = -1.0
    phi, phix, phixx = bvp_modified_basis(x, nbasis, phi, phix, phixx, scale)
    for j in 1:nbasis
        u[N] = u[N] + (aphi[j] * phi[j])
        ux[N] = ux[N] + (aphi[j] * phix[j])
        uxx[N] = uxx[N] + (aphi[j] * phixx[j])
    end
    for i in 1:nbasis
        x = xi[i]
        phi, phix, phixx = bvp_modified_basis(x, nbasis, phi, phix, phixx, scale)
        u[i+1] = alpha*(1-x)/2.0 + beta*(1+x)/2.0
        ux[i+1] = (-alpha + beta)/2.0
        uxx[i+1] = 0.0
        for j in 1:nbasis
            u[i+1] = u[i+1] + (aphi[j] * phi[j])
            ux[i+1] = ux[i+1] + (aphi[j] * phix[j])
            uxx[i+1] = uxx[i+1] + (aphi[j] * phixx[j])
        end
    end

    return u
end

function bvp_modified_basis(x::Float64, nbasis::Int64, phi::Array{Float64}, phix::Array{Float64}, phixx::Array{Float64}, scale::Float64)

    if abs(x) < 1.0 # Avoid singularities at the endpoints
        t = acos(x)
        c = cos(t)
        s = sin(t)
        for i in 1:nbasis
            n = i+1
            tn = cos(n*t)
            tnt = -n * sin(n*t)
            tntt = -n * n * tn

            # Convert t-derivatives into x-derivatives
            tnx = -tnt/s
            tnxx = tntt/(s*s) - tnt*c/(s*s*s)

            # Convert to modified basis functions to enforce BCs
            if mod(n,2) == 0
                phi[i] = tn - 1.0
                phix[i] = tnx / scale
            else
                phi[i] = tn - x
                phix[i] = (tnx - 1.0) / scale
            end
            phixx[i] = tnxx / (scale * scale)
        end
    else
        for i in 1:nbasis
            phi[i] = 0.0
            n = i+1
            if mod(n,2) == 0
                phix[i] = sign(x)*n*n / scale
            else
                phix[i] = (n*n - 1) / scale
            end
            phixx[i] = sign(x)^n * n * n * ((n * n - 1.0)/3.0) / (scale * scale)
        end
    end

    return phi, phix, phixx
end

function bvp_basis(x::Float64, nbasis::Int64, phi::Array{Float64}, phix::Array{Float64}, phixx::Array{Float64})

    if abs(x) < 1.0 # Avoid singularities at the endpoints
        t = acos(x)
        c = cos(t)
        s = sin(t)
        for i in 1:nbasis
            n = i-1
            tn = cos(n*t)
            tnt = -n * sin(n*t)
            tntt = -n * n * tn

            # Convert t-derivatives into x-derivatives
            tnx = -tnt/s
            tnxx = tntt/(s*s) - tnt*c/(s*s*s)

            # Convert to modified basis functions to enforce BCs
            if mod(n,2) == 0
                phi[i] = tn #- 1.0
                phix[i] = tnx
            else
                phi[i] = tn #- x
                phix[i] = tnx #- 1.0
            end
            phixx[i] = tnxx
        end
    else
        t = acos(x)
        for i in 1:nbasis
            n = i-1
            phi[i] = cos(n*t)
            if x > 0.0
                phix[i] = -n*n
            else
                phix[i] = -n*n*(-1.0)^(n+1)
            end
            phixx[i] = ((-1.0)^n)*(n^4 - n^2)/3
        end
    end

    return phi, phix, phixx
end

end #module
