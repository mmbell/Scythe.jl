module Chebyshev

using LinearAlgebra
using Parameters
using FFTW

export ChebyshevParameters, Chebyshev1D
#export R0, R1T0, R1T1, R1T2, R2T10, R2T20, R3, PERIODIC
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
@with_kw struct ChebyshevParameters
    zmin::real = 0.0
    zmax::real = 0.0
    zDim::int = 0
    bDim::int = 0
    BCB::Dict = R0
    BCT::Dict = R0
end

struct Chebyshev1D
    # Parameters for the column
    params::ChebyshevParameters
    
    # Pre-calculated Chebyshev–Gauss–Lobatto points (extrema of Chebyshev polynomials)
    mishPoints::Vector{real}
    
    # Scalar, vector, or matrix that enforces boundary conditions
    gammaBC::Array{real}
    
    # Measured FFTW Plan
    fftPlan::FFTW.r2rFFTWPlan{Float64, (3,), false, 1, UnitRange{Int64}}
    
    # Filter matrix
    filter::Matrix{real}

    # uMish is the physical values
    # b is the filtered Chebyshev coefficients without BCs
    # a is the padded Chebyshev coefficients with BCs
    uMish::Vector{real}
    b::Vector{real}
    a::Vector{real}
end

function Chebyshev1D(cp::ChebyshevParameters)

    mishPoints = calcMishPoints(cp)
    gammaBC = calcGammaBC(cp)

    uMish = zeros(real,cp.zDim)
    b = zeros(real,cp.bDim)
    a = zeros(real,cp.zDim)

    # Plan the FFT
    fftPlan = FFTW.plan_r2r(a, FFTW.REDFT00, flags=FFTW.PATIENT)
    
    filter = calcFilterMatrix(cp)

    column = Chebyshev1D(cp,mishPoints,gammaBC,fftPlan,filter,uMish,b,a)
    return column
end

function calcMishPoints(cp::ChebyshevParameters)
    
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

    filter = Matrix(1.0I, cp.bDim, cp.zDim)
    return filter
end

function CBtransform(cp::ChebyshevParameters, fftPlan, uMish::Vector{real})

    # Do the DCT transform and pre-scale
    b = (fftPlan * uMish) ./ (2 * (cp.zDim -1))
    return b[1:cp.bDim]
end

function CBtransform!(column::Chebyshev1D)

    # Do the DCT transform and pre-scale
    b = (column.fftPlan * column.uMish) ./ (2 * (column.params.zDim -1))
    column.b .= column.filter * b
end

function CBtransform(column::Chebyshev1D, uMish::Vector{real})

    # Do the DCT transform and pre-scale
    b = (column.fftPlan * uMish) ./ (2 * (column.params.zDim -1))
    return column.filter * b
end

function CAtransform(cp::ChebyshevParameters, gammaBC, b::Vector{real})

    # Apply the BCs
    bfill = [b ; zeros(Float64, cp.zDim-cp.bDim)]
    a = bfill .+ (gammaBC' * bfill)
    return a
end

function CAtransform!(column::Chebyshev1D)

    # Apply the BCs
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
    
    # Do the inverse DCT transform to get back physical values
    column.uMish .= column.fftPlan * column.a
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

function CIInttransform(cp::ChebyshevParameters, fftPlan, a::Vector{real}, C0::real = 0.0)

    aInt = CIIntcoefficients(cp,a,C0)
    uInt = fftPlan * aInt
    return uInt
end

function CIInttransform(column::Chebyshev1D, C0::real = 0.0)
    
    uInt = CIInttransform(column.params, column.fftPlan, column.a, C0)
    return uInt
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

function CIxtransform(cp::ChebyshevParameters, fftPlan, a::Vector{real})

    # Recursive relationship for derivative coefficients
    ax = zeros(real,cp.zDim)
    k = cp.zDim
    ax[k-1] = (2.0 * (k-1) * a[k])
    for k = (cp.zDim-1):-1:2
        ax[k-1] = (2.0 * (k-1) * a[k]) + ax[k+1]
    end
    ax = ax ./ (-0.5 * (cp.zmax - cp.zmin))
    
    # Do the inverse transform to get back physical values
    ux = fftPlan * ax
    return ux
end

function CIxtransform(column::Chebyshev1D)
    
    ux = CIxtransform(column.params, column.fftPlan, column.a)
    return ux
end

function CIxxtransform(column::Chebyshev1D)
    
    ax = CIxcoefficients(column.params, column.a)
    uxx = CIxtransform(column.params, column.fftPlan, ax)
    return uxx
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
        # Requires two-step application (not yet implemented)
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
    
        return gammaBCB, gammaBCT
    
    elseif (cp.BCB == R1T0) && (cp.BCT == R1T1)
        # Requires two-step application (not yet implemented)
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

        return gammaBCB, gammaBCT
    end
    
end

end