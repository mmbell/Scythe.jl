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
    
    # Phase-shift and filter matrix
    phasefilter::Matrix{real}
    invphasefilter::Matrix{real}
    
    # In this context, uMish is the physical values
    # b is the filtered Fourier coefficients 
    # a is the Fourier coefficients with padding
    # ax is a buffer for derivative and integral coefficients
    uMish::Vector{real}
    b::Vector{real}
    a::Vector{real}
    ax::Vector{real}
end

function Fourier1D(fp::FourierParameters)

    mishPoints = calcMishPoints(fp)

    uMish = zeros(real,fp.yDim)
    b = zeros(real,fp.bDim)
    a = zeros(real,fp.yDim)
    ax = zeros(real,fp.yDim)
    
    # Plan the FFT
    fftPlan = FFTW.plan_r2r(a, FFTW.FFTW.R2HC, flags=FFTW.PATIENT)
    ifftPlan = FFTW.plan_r2r(a, FFTW.FFTW.HC2R, flags=FFTW.PATIENT)
    
    # Pre-calculate the phase filter matrix
    phasefilter = calcPhaseFilter(fp)
    invphasefilter = calcInvPhaseFilter(fp)
    
    ring = Fourier1D(fp,mishPoints,fftPlan,ifftPlan,phasefilter,invphasefilter,uMish,b,a,ax)
    return ring
end

function calcPhaseFilter(fp::FourierParameters)
    
    phasefilter = zeros(real, fp.yDim, fp.bDim)
    phasefilter[1,1] = 1.0
    for k in 1:fp.kmax
        phasefilter[k+1,k+1] = cos(-k * fp.ymin)
        phasefilter[fp.yDim-k+1,k+1] = -sin(-k * fp.ymin)
        phasefilter[k+1,fp.bDim-k+1] = sin(-k * fp.ymin)
        phasefilter[fp.yDim-k+1,fp.bDim-k+1] = cos(-k * fp.ymin)
    end
    return phasefilter

end

function calcInvPhaseFilter(fp::FourierParameters)
    
    invphasefilter = zeros(real, fp.bDim, fp.yDim)
    invphasefilter[1,1] = 1.0
    for k in 1:fp.kmax
        invphasefilter[k+1,k+1] = cos(k * fp.ymin)
        invphasefilter[fp.bDim-k+1,k+1] = -sin(k * fp.ymin)
        invphasefilter[k+1,fp.yDim-k+1] = sin(k * fp.ymin)
        invphasefilter[fp.bDim-k+1,fp.yDim-k+1] = cos(k * fp.ymin)
    end
    return invphasefilter

end

function calcMishPoints(fp::FourierParameters)

    Nbasis = fp.yDim
    y = zeros(real,Nbasis)
    for n = 1:Nbasis
        y[n] = fp.ymin + (2 * π * (n-1) / Nbasis)
    end
    return y
end

function FBtransform(fp::FourierParameters, fftPlan, phasefilter::Matrix{real}, uMish::Vector{real})

    # Do the Fourier transform, scale, and filter
    b = (fftPlan * uMish) ./ fp.yDim
    bfilter = (b' * phasefilter)'
    return bfilter
end

function FBtransform!(ring::Fourier1D)

    # Do the Fourier transform, scale, and filter
    b = (ring.fftPlan * ring.uMish) ./ ring.params.yDim
    ring.b .= (b' * ring.phasefilter)'
end

function FAtransform(fp::FourierParameters, invphasefilter::Matrix{real}, b::Vector{real})

    # Apply the inverse phasefilter
    a = (b' * invphasefilter)'
    return a
end

function FAtransform!(ring::Fourier1D)

    # Apply the inverse phasefilter
    ring.a .= (ring.b' * ring.invphasefilter)'
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

function FIxcoefficients(fp::FourierParameters, a::Vector{real}, ax::Vector{real})

    for k = 1:fp.kmax
        ax[k+1] = -k * a[fp.yDim-k+1]
        ax[fp.yDim-k+1] = k * a[k+1]
    end
    return ax
end

function FIxtransform(fp::FourierParameters, ifftPlan, a::Vector{real}, ax::Vector{real})

    # Do the inverse transform with derivative coefficients to get back physical values
    ux = ifftPlan * FIxcoefficients(fp,a,ax)
    return ux
end

function FIxtransform(ring::Fourier1D)

    # Do the inverse transform with derivative coefficients to get back physical values
    ux = ring.ifftPlan * FIxcoefficients(ring.params,ring.a,ring.ax)
    return ux
end

function FIxtransform(ring::Fourier1D, ux::AbstractVector)

    # Do the inverse transform with derivative coefficients to get back physical values
    ux .= ring.ifftPlan * FIxcoefficients(ring.params,ring.a,ring.ax)
    return ux
end

function FIxxtransform(ring::Fourier1D)

    # Do the inverse transform with derivative coefficients to get back physical values
    ax = copy(FIxcoefficients(ring.params,ring.a,ring.ax))
    uxx = FIxtransform(ring.params, ring.ifftPlan, ax, ring.ax)
    return uxx
end

function FIIntcoefficients(fp::FourierParameters, a::Vector{real}, aInt::Vector{real}, C0::real = 0.0)

    aInt[1] = C0
    for k = 1:fp.kmax
        aInt[k+1] = a[fp.yDim-k+1] / k
        aInt[fp.yDim-k+1] = -a[k+1] / k
    end
    return aInt
end

function FIInttransform(fp::FourierParameters, ifftPlan, a::Vector{real}, aInt::Vector{real}, C0::real = 0.0)

    # Do the inverse transform with derivative coefficients to get back physical values
    return ifftPlan * FIIntcoefficients(fp,a,aInt,C0)
end

function FIInttransform(ring::Fourier1D, C0::real = 0.0)

    # Do the inverse transform with derivative coefficients to get back physical values
    return FIInttransform(ring.params,ring.ifftPlan,ring.a,ring.ax,C0)
end


end
