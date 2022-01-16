module Chebyshev

using LinearAlgebra
using Parameters
using FFTW

export ChebyshevParameters, Chebyshev1D
export R0, R1T0, R1T1, R1T2, R2T10, R2T20, R3, PERIODIC
#export SBtransform, SBtransform!, SAtransform!, SItransform!
#export SBxtransform, SIxtransform, SIxxtransform

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
@with_kw struct ChebyshevParameters
    zmin::real = 0.0
    zmax::real = 0.0
    zDim::int = 1
    BCB::Dict = R0
    BCT::Dict = R0
end

struct Chebyshev1D
    params::ChebyshevParameters
    mishPoints::Vector{real}
    gammaBC
    
    # In this context, uMish is the physical values
    # b is the Chebyshev coefficients without BCs
    # a is the Chebyshev coefficients with BCs
    uMish::Vector{real}
    b::Vector{real}
    a::Vector{real}
end

function Chebyshev1D(cp::ChebyshevParameters)

    mishPoints = calcMishPoints(cp)
    gammaBC = calcGammaBC(cp)

    uMish = zeros(real,cp.zDim)
    b = zeros(real,cp.zDim)
    a = zeros(real,cp.zDim)

    
    column = Chebyshev1D(cp,mishPoints,gammaBC,uMish,b,a)
    return column
end

function calcMishPoints(cp::ChebyshevParameters)
    
    Nbasis = cp.zDim
    z = zeros(real,Nbasis)
    scale = -0.5 * (cp.zmax - cp.zmin)
    offset = 0.5 * (cp.zmin + cp.zmax)
    z[1] = 1.0 * scale + offset
    for n = 2:Nbasis
        z[n] = cos((n-1) * π / (Nbasis - 1)) * scale + offset
    end
    z[Nbasis] = -1.0 * scale + offset
    return z
end

function CBtransform(cp::ChebyshevParameters, uMish::Vector{real})

    # Do the Fourier transform and pre-scale
    b = FFTW.r2r(uMish, FFTW.REDFT00) ./ (2 * (cp.zDim -1))
    return b
end

function CAtransform(cp::ChebyshevParameters, gammaBC, b::Vector{real})

    # Apply the BCs
    a = gammaBC * b
    return a
end

function CItransform(cp::ChebyshevParameters, a::Vector{real})

    # Do the inverse transform to get back physical values
    uMish = FFTW.r2r(a, FFTW.REDFT00)
    return uMish
end

function CIxcoefficients(cp::ChebyshevParameters, a::Vector{real})

    # Recursive relationship for derivative coefficients
    ax = zeros(real,cp.zDim)
    k = cp.zDim
    ax[k-1] = (2.0 * (k-1) * a[k])
    for k = (cp.zDim-1):-1:2
        ax[k-1] = (2.0 * (k-1) * a[k]) + ax[k+1]
    end
    return ax ./ (-0.5 * (cp.zmax - cp.zmin))
end

function CIIntcoefficients(cp::ChebyshevParameters, a::Vector{real}, C0::real = 0.0)

    # Recursive relationship for integral coefficients
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

function CIxtransform(cp::ChebyshevParameters, a::Vector{real})

    # Recursive relationship for derivative coefficients
    ax = zeros(real,cp.zDim)
    k = cp.zDim
    ax[k-1] = (2.0 * (k-1) * a[k])
    for k = (cp.zDim-1):-1:2
        ax[k-1] = (2.0 * (k-1) * a[k]) + ax[k+1]
    end
    ax = ax ./ (-0.5 * (cp.zmax - cp.zmin))
    
    # Do the inverse transform to get back physical values
    ux = FFTW.r2r(ax, FFTW.REDFT00) 
    return ux
end

function calcDCTmatrix(Nbasis::Int64)
    
    # Create a matrix with the DCT as basis functions for boundary conditions
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

function calcGammaBC(cp::ChebyshevParameters)

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
    elseif haskey(cp.BCT,"α2")
        #Not implemented yet
    end
    
    gammaTranspose = dctMatrix' \ dctBC'    
    gammaBC = Matrix{Float64}(undef,25,25)
    gammaBC .= gammaTranspose'
    return gammaBC
    
end


end