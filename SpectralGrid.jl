module SpectralGrid

using CubicBSpline
using Chebyshev
using Fourier
using Parameters
using CSV
using DataFrames

#Define some convenient aliases
const real = Float64
const int = Int64
const uint = UInt64

export GridParameters, createGrid, spectralTransform, gridTransform

@with_kw struct GridParameters
    xmin::real = 0.0
    xmax::real = 0.0
    num_nodes::int = 0
    l_q::real = 2.0
    BCL::Dict = CubicBSpline.R0
    BCR::Dict = CubicBSpline.R0
    num_rings::int = 0
    zmin::real = 0.0
    zmax::real = 0.0
    zDim::int = 0
    bDim::int = 0
    BCB::Dict = Chebyshev.R0
    BCT::Dict = Chebyshev.R0
    vars::Dict = Dict("u" => 1)
end

struct R_Grid
    splines::Array{Spline1D}
end

struct Z_Grid
    columns::Array{Chebyshev1D}
end

struct RZ_Grid
    splines::Array{Spline1D}
    columns::Array{Chebyshev1D}
end

struct RL_Grid
    splines::Array{Spline1D}
    rings::Array{Fourier1D}
end

struct RLZ_Grid
    splines::Array{Spline1D}
    columns::Array{Chebyshev1D}
    rings::Array{Fourier1D}
end

function createGrid(gp::GridParameters)
    
    if gp.num_nodes > 0
        # R, RZ, RL, or RLZ grid
        
        if gp.num_rings == 0 && gp.zDim == 0
            # R grid
            splines = Array{Spline1D}(undef,1,length(values(gp.vars)))
            grid = R_Grid(splines)
            for key in keys(gp.vars)
                grid.splines[1,gp.vars[key]] = Spline1D(SplineParameters(
                    xmin = gp.xmin,
                    xmax = gp.xmax,
                    num_nodes = gp.num_nodes,
                    BCL = gp.BCL[key],
                    BCR = gp.BCR[key]))
            end
            return grid
            
        elseif gp.num_rings > 0 && gp.zDim == 0
            # RL grid
            grid = RL_Grid()
            for key in keys(grid.vars)
                for l = 1:gp.num_rings
                    grid.splines[l,gp.vars[key]] = Spline1D(SplineParameters(
                        xmin = gp.xmin,
                        xmax = gp.xmax,
                        num_nodes = gp.num_nodes,
                        BCL = gp.BCL[key], 
                        BCR = gp.BCR[key]))
                    lpoints = 4 + 4*l
                    dl = 2 * π / lpoints
                    offset = 0.5 * dl * (l-1)
                    grid.rings[l,gp.vars[key]] = Fourier1D(FourierParameters(
                        ymin = offset,
                        ymax = offset + (2 * π) - dl,
                        yDim = lpoints,
                        bDim = l*2 + 1,
                        kmax = l))
                end
            end
            return grid
            
        elseif gp.num_rings == 0 && gp.zDim > 0
            # RZ grid
            splines = Array{Spline1D}(undef,gp.bDim,length(values(gp.vars)))
            columns = Array{Chebyshev1D}(undef,gp.num_nodes*3,length(values(gp.vars)))
            grid = RZ_Grid(splines, columns)
            for key in keys(gp.vars)
                for z = 1:gp.bDim
                    grid.splines[z,gp.vars[key]] = Spline1D(SplineParameters(
                        xmin = gp.xmin,
                        xmax = gp.xmax,
                        num_nodes = gp.num_nodes,
                        BCL = gp.BCL[key], 
                        BCR = gp.BCR[key]))
                end
                for r = 1:(gp.num_nodes*3)
                    grid.columns[r,gp.vars[key]] = Chebyshev1D(ChebyshevParameters(
                        zmin = gp.zmin,
                        zmax = gp.zmax,
                        zDim = gp.zDim,
                        bDim = gp.bDim,
                        BCB = gp.BCB[key],
                        BCT = gp.BCT[key]))
                end
            end
            return grid
            
        elseif gp.num_rings > 0 && gp.zDim > 0
            # RLZ grid
            throw(DomainError(0, "RLZ not implemented yet"))
        end
    else
        # Z grid
        throw(DomainError(0, "Z column model not implemented yet"))
    end
    
end

function spectralTransform(grid::RZ_Grid)
    return "RZ"
end

function spectralTransform(grid::R_Grid)
    return "R"
end

function gridTransform()
    
end


#include("AdamsBashforth_1D_multivar_O3.jl")

#export initialize, run, finalize, integrate_1dLinearAdvection, integrate_WilliamsSlabTCBL, integrate_model

end
