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
const R1T0 = Dict("α1" => 0.0, "β1" => 0.0)
const R1T1 = Dict("α1" =>  0.0, "β1" =>  1.0)
const R1T2 = Dict("α1" =>  2.0, "β1" => -1.0)
const R2T10 = Dict("α2" => 1.0, "β2" => -0.5)
const R2T20 = Dict("α2" => -1.0, "β2" => 0.0)
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

    # In this context, uMish is the physical values
    # b is the Chebyshev coefficients without BCs
    # a is the Chebyshev coefficients with BCs
    uMish::Vector{real}
    b::Vector{real}
    a::Vector{real}
end

function Chebyshev1D(cp::ChebyshevParameters)

    mishPoints = calcMishPoints(cp)
    uMish = zeros(real,cp.zDim)
    b = zeros(real,cp.zDim)
    a = zeros(real,cp.zDim)

    column = Chebyshev1D(cp,mishPoints,uMish,b,a)
    return column
end

function calcMishPoints(cp::ChebyshevParameters)
    
    if haskey(cp.BCB,"α1")
        rankB = 1
    elseif haskey(cp.BCB,"α2")
        rankB = 2
    elseif cp.BCB == R0
        rankB = 0
    elseif cp.BCB == R3
        rankB = 3
    end

    if haskey(cp.BCT,"α1")
        rankT = 1
    elseif haskey(cp.BCT,"α2")
        rankT = 2
    elseif cp.BCT == R0
        rankT = 0
    elseif cp.BCT == R3
        rankT = 3
    end

    Nbasis = cp.zDim - rankB - rankT
    z = zeros(real,cp.zDim)
    #scale = 0.5 * (cp.zmax - cp.zmin)
    #offset = 0.5 * (cp.zmin + cp.zmax)
    scale = 1.0
    offset = 0.0
    z[1] = 1.0 * scale + offset
    for n = 2:Nbasis
        z[n] = cos((n-1) * π / (Nbasis - 1)) * scale + offset
    end
    z[cp.zDim] = -1.0 * scale + offset
    return z
end

function CBtransform(cp::ChebyshevParameters, uMish::Vector{real})

    # Do the Fourier transform and pre-scale
    b = FFTW.r2r(uMish, FFTW.REDFT00) ./ (2 * (cp.zDim -1))
    return b
end

function CAtransform(cp::ChebyshevParameters, b::Vector{real})

    # Apply the BCs (to do)
    a = b
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
    return ax
end

function CIxtransform(cp::ChebyshevParameters, a::Vector{real})

    # Recursive relationship for derivative coefficients
    ax = zeros(real,cp.zDim)
    k = cp.zDim
    ax[k-1] = (2.0 * (k-1) * a[k])
    for k = (cp.zDim-1):-1:2
        ax[k-1] = (2.0 * (k-1) * a[k]) + ax[k+1]
    end
    
    # Do the inverse transform to get back physical values
    ux = FFTW.r2r(ax, FFTW.REDFT00) 
    return ux
end


end