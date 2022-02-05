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
    lmin::real = 0.0
    kmax::int = 0
    yDim::int = 0
    bDim::int = 0
end

struct Fourier1D
    # Parameters for the column
    params::FourierParameters
    
    # Pre-calculated angular points
    mishPoints::Vector{real}
        
    # Measured FFTW Plan
    fftPlan::FFTW.r2rFFTWPlan{Float64, (0,), false, 1, UnitRange{Int64}}
    ifftPlan::FFTW.r2rFFTWPlan{Float64, (1,), false, 1, UnitRange{Int64}}
    
    # In this context, uMish is the physical values
    # b is the filtered Fourier coefficients 
    # a is the Fourier coefficients with padding
    uMish::Vector{real}
    b::Vector{real}
    a::Vector{real}
end

function Fourier1D(fp::FourierParameters)

    mishPoints = calcMishPoints(fp)

    uMish = zeros(real,fp.yDim)
    b = zeros(real,fp.bDim)
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
        y[n] = fp.lmin + (2 * π * (n-1) / Nbasis)
    end
    return y
end

function FBtransform(fp::FourierParameters, fftPlan, uMish::Vector{real})

    # Do the Fourier transform, scale, and filter
    bfilter = zeros(Float64, fp.bDim)
    b = (fftPlan * uMish) ./ fp.yDim
    bfilter[1] = b[1]
    for k in 1:fp.kmax
        bfilter[k+1] = b[k+1]
        bfilter[fp.bDim-k+1] = b[fp.yDim-k+1]
    end
    return bfilter
end

function FBtransform!(ring::Fourier1D)

    # Do the Fourier transform, scale, and filter
    b = FBtransform(ring.params, ring.fftPlan, ring.uMish)
    ring.b .= b
end

function FAtransform(fp::FourierParameters, b::Vector{real})

    # Apply the padding
    a = [ b[1:fp.kmax+1] ; 
        zeros(Float64, fp.yDim-fp.bDim) ; 
        b[fp.kmax+2:fp.bDim] ]
    return a
end

function FAtransform!(ring::Fourier1D)

    # Apply the padding
    ring.a .= [ ring.b[1:ring.params.kmax+1] ; 
        zeros(Float64, ring.params.yDim-ring.params.bDim) ; 
        ring.b[ring.params.kmax+2:ring.params.bDim] ]
end

function FItransform(fp::FourierParameters, ifftPlan, a::Vector{real})

    # Do the inverse transform to get back physical values
    uMish = ifftPlan * a
    return uMish
end

function FItransform!(ring::Fourier1D)

    # Do the inverse transform to get back physical values
    ring.uMish .= ring.ifftPlan * ring.a
end

function FIxcoefficients(fp::FourierParameters, a::Vector{real})

    ax = zeros(Float64, fp.yDim)
    for k = 1:fp.kmax
        ax[k+1] = -k * a[fp.yDim-k+1]
        ax[fp.yDim-k+1] = k * a[k+1]
    end
    return ax
end

function FIxtransform(fp::FourierParameters, ifftPlan, a::Vector{real})

    # Do the inverse transform with derivative coefficients to get back physical values
    ax = FIxcoefficients(fp,a)
    ux = ifftPlan * ax
    return ux
end

function FIxtransform(ring::Fourier1D)

    # Do the inverse transform with derivative coefficients to get back physical values
    ax = FIxcoefficients(ring.params,ring.a)
    ux = ring.ifftPlan * ax
    return ux
end

function FIxxtransform(ring::Fourier1D)

    # Do the inverse transform with derivative coefficients to get back physical values
    ax = FIxcoefficients(ring.params,ring.a)
    uxx = FIxtransform(ring.params, ring.ifftPlan, ax)
    return uxx
end

function FIIntcoefficients(fp::FourierParameters, a::Vector{real}, C0::real = 0.0)
    
    aInt = zeros(Float64, fp.yDim)
    aInt[1] = C0
    for k = 1:fp.kmax
        aInt[k+1] = a[fp.yDim-k+1] / k
        aInt[fp.yDim-k+1] = -a[k+1] / k
    end
    return aInt
end

function FIInttransform(fp::FourierParameters, ifftPlan, a::Vector{real}, C0::real = 0.0)

    # Do the inverse transform with derivative coefficients to get back physical values
    aInt = FIIntcoefficients(fp,a,C0)
    uInt = ifftPlan * aInt
    return uInt
end

function FIInttransform(ring::Fourier1D, C0::real = 0.0)

    # Do the inverse transform with derivative coefficients to get back physical values
    uInt = FIInttransform(ring.params,ring.ifftPlan,ring.a,C0)
    return ux
end


end
