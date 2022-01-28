module Fourier

using LinearAlgebra
using Parameters
using FFTW

export FourierParameters, Fourier1D
export FBtransform, FBtransform!, FAtransform!, FItransform!
export FBxtransform, FIxtransform, FIxxtransform

#Define some convenient aliases
const real = Float64
const int = Int64
const uint = UInt64

# Define the ring parameters
@with_kw struct FourierParameters
    ymin::real = 0.0
    ymax::real = 0.0
    yDim::int = 1
end

struct Fourier1D
    # Parameters for the column
    params::FourierParameters
    
    # Pre-calculated angular points
    mishPoints::Vector{real}
        
    # Measured FFTW Plan
    fftPlan
    ifftPlan
    
    # In this context, uMish is the physical values
    # b is the Fourier coefficients without filtering
    # a is the Fourier coefficients with filtering
    uMish::Vector{real}
    b::Vector{real}
    a::Vector{real}
end

function Fourier1D(fp::FourierParameters)

    mishPoints = calcMishPoints(fp)

    uMish = zeros(real,fp.yDim)
    b = zeros(real,fp.yDim)
    a = zeros(real,fp.yDim)
    
    # Plan the FFT
    fftPlan = FFTW.plan_r2r(a, FFTW.FFTW.R2HC, flags=FFTW.PATIENT)
    ifftPlan = FFTW.plan_r2r(a, FFTW.FFTW.HC2R, flags=FFTW.PATIENT)

    ring = Fourier1D(fp,mishPoints,fftPlan,ifftPlan,uMish,b,a)
    return ring
end

function calcMishPoints(fp::FourierParameters)

    Nbasis = fp.yDim
    y = zeros(real,Nbasis)
    for n = 1:Nbasis
        y[n] = fp.ymin + (2 * π * (n-1) / Nbasis)  
    end
    return y
end

function FBtransform(fp::FourierParameters, fftPlan, uMish::Vector{real})

    # Do the Fourier transform and pre-scale
    b = (fftPlan * uMish) ./ fp.yDim
    return b
end

function FAtransform(fp::FourierParameters, gammaBC, b::Vector{real})

    # Apply some filtering (TBD)
    a = b
    return a
end

function FItransform(fp::FourierParameters, ifftPlan, a::Vector{real})

    # Do the inverse transform to get back physical values
    uMish = ifftPlan * a
    return uMish
end

function FIxcoefficients(fp::FourierParameters, a::Vector{real})


end

function FIIntcoefficients(fp::FourierParameters, a::Vector{real}, C0::real = 0.0)

end

function FIxtransform(fp::FourierParameters, a::Vector{real})

    # Recursive relationship for derivative coefficients

    # Do the inverse transform to get back physical values
end

end
