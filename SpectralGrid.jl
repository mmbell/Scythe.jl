module SpectralGrid

using CubicBSpline
using Chebyshev
using Fourier
using CSV
using DataFrames
using SharedArrays
using SparseArrays

#Define some convenient aliases
const real = Float64
const int = Int64
const uint = UInt64

# Fix the spline mish to 3 points
const mubar = 3

export GridParameters, createGrid, getGridpoints, getCartesianGridpoints
export spectralTransform!, gridTransform!, spectralTransform 
export spectralxTransform, gridTransform_noBCs, integrateUp
export regularGridTransform, getRegularGridpoints, getRegularCartesianGridpoints
export AbstractGrid, R_Grid, RZ_Grid, RL_Grid
export calcTileSizes, setSpectralTile!, setSpectralTile
export sumSharedSpectral, getBorderSpectral
export calcPatchMap, calcHaloMap, allocateSplineBuffer

Base.@kwdef struct GridParameters
    geometry::String = "R"
    xmin::real = 0.0
    xmax::real = 0.0
    num_cells::int = 0
    rDim::int = num_cells * mubar
    b_rDim::int = num_cells + 3
    l_q::real = 2.0
    BCL::Dict = CubicBSpline.R0
    BCR::Dict = CubicBSpline.R0
    lDim::int = 0
    b_lDim::int = 0
    zmin::real = 0.0
    zmax::real = 0.0
    zDim::int = 0
    b_zDim::int = 0
    BCB::Dict = Chebyshev.R0
    BCT::Dict = Chebyshev.R0
    vars::Dict = Dict("u" => 1)
    # Patch indices
    spectralIndexL::int = 1
    spectralIndexR::int = spectralIndexL + b_rDim - 1
    patchOffsetL::int = (spectralIndexL - 1) * 3
    patchOffsetR::int = patchOffsetL + rDim
    tile_num::int = 0
end

abstract type AbstractGrid end

# Include functions for implemented grids
include("R_Grid.jl")
include("RZ_Grid.jl")
include("RL_Grid.jl")

# Not yet implemented
struct Z_Grid <: AbstractGrid
    params::GridParameters
    columns::Array{Chebyshev1D}
    spectral::Array{Float64}
    physical::Array{Float64}
end

# Not yet implemented
struct RLZ_Grid <: AbstractGrid
    params::GridParameters
    splines::Array{Spline1D}
    columns::Array{Chebyshev1D}
    rings::Array{Fourier1D}
    spectral::Array{Float64}
    physical::Array{Float64}
end

function createGrid(gp::GridParameters)
        
    if gp.geometry == "R"
        # R grid

        splines = Array{Spline1D}(undef,1,length(values(gp.vars)))
        spectral = zeros(Float64, gp.b_rDim, length(values(gp.vars)))
        physical = zeros(Float64, gp.rDim, length(values(gp.vars)), 3)
        grid = R_Grid(gp, splines, spectral, physical)
        for key in keys(gp.vars)
            grid.splines[1,gp.vars[key]] = Spline1D(SplineParameters(
                xmin = gp.xmin,
                xmax = gp.xmax,
                num_cells = gp.num_cells,
                BCL = gp.BCL[key],
                BCR = gp.BCR[key]))
        end
        return grid

    elseif gp.geometry == "RZ"
        # RZ grid

        splines = Array{Spline1D}(undef,gp.zDim,length(values(gp.vars)))
        columns = Array{Chebyshev1D}(undef,length(values(gp.vars)))
        spectral = zeros(Float64, gp.b_zDim * gp.b_rDim, length(values(gp.vars)))
        physical = zeros(Float64, gp.zDim * gp.rDim, length(values(gp.vars)), 5)
        grid = RZ_Grid(gp, splines, columns, spectral, physical)
        for key in keys(gp.vars)
            for z = 1:gp.zDim
                grid.splines[z,gp.vars[key]] = Spline1D(SplineParameters(
                    xmin = gp.xmin,
                    xmax = gp.xmax,
                    num_cells = gp.num_cells,
                    BCL = gp.BCL[key], 
                    BCR = gp.BCR[key]))
            end
            grid.columns[gp.vars[key]] = Chebyshev1D(ChebyshevParameters(
                zmin = gp.zmin,
                zmax = gp.zmax,
                zDim = gp.zDim,
                bDim = gp.b_zDim,
                BCB = gp.BCB[key],
                BCT = gp.BCT[key]))
        end
        return grid

    elseif gp.geometry == "RL"
        # RL grid

        # Calculate the number of points in the grid
        lpoints = 0
        blpoints = 0
        for r = 1:gp.rDim
            ri = r + gp.patchOffsetL
            lpoints += 4 + 4*ri
            blpoints += 1 + 2*ri
        end

        # Have to create a new immutable structure for the parameters
        gp2 = GridParameters(
            geometry = gp.geometry,
            xmin = gp.xmin,
            xmax = gp.xmax,
            num_cells = gp.num_cells,
            rDim = gp.rDim,
            b_rDim = gp.b_rDim,
            l_q = gp.l_q,
            BCL = gp.BCL,
            BCR = gp.BCR,
            lDim = lpoints,
            b_lDim = blpoints,
            zmin = gp.zmin,
            zmax = gp.zmax,
            zDim = gp.zDim,
            b_zDim = gp.b_zDim,
            BCB = gp.BCB,
            BCT = gp.BCT,
            vars = gp.vars,
            spectralIndexL = gp.spectralIndexL,
            spectralIndexR = gp.spectralIndexR,
            patchOffsetL = gp.patchOffsetL,
            tile_num = gp.tile_num)

        splines = Array{Spline1D}(undef,3,length(values(gp2.vars)))
        rings = Array{Fourier1D}(undef,gp2.rDim,length(values(gp2.vars)))
        spectral = zeros(Float64, gp2.b_lDim, length(values(gp2.vars)))
        physical = zeros(Float64, gp2.lDim, length(values(gp2.vars)), 5)
        grid = RL_Grid(gp2, splines, rings, spectral, physical)
        for key in keys(gp2.vars)

            # Need different BCs for wavenumber zero winds since they are undefined at r = 0
            for i = 1:3
                if (i == 1 && (key == "u" || key == "v" || key == "vgr"
                            || key == "ub" || key == "vb"))
                    grid.splines[1,gp2.vars[key]] = Spline1D(SplineParameters(
                        xmin = gp2.xmin,
                        xmax = gp2.xmax,
                        num_cells = gp2.num_cells,
                        BCL = CubicBSpline.R1T0, 
                        BCR = gp2.BCR[key]))
                else
                    grid.splines[i,gp.vars[key]] = Spline1D(SplineParameters(
                        xmin = gp2.xmin,
                        xmax = gp2.xmax,
                        num_cells = gp2.num_cells,
                        BCL = gp2.BCL[key], 
                        BCR = gp2.BCR[key]))
                end
            end

            for r = 1:gp2.rDim
                ri = r + gp2.patchOffsetL
                lpoints = 4 + 4*ri
                dl = 2 * π / lpoints
                offset = 0.5 * dl * (ri-1)
                grid.rings[r,gp2.vars[key]] = Fourier1D(FourierParameters(
                    ymin = offset,
                    # ymax = offset + (2 * π) - dl,
                    yDim = lpoints,
                    bDim = ri*2 + 1,
                    kmax = ri))
            end
        end
        return grid

    elseif gp.geometry == "RLZ"
        # RLZ grid
        throw(DomainError(0, "RLZ not implemented yet"))
        
    elseif gp.geometry == "Z"
        # Z grid
        throw(DomainError(0, "Z column model not implemented yet"))
    else
        # Unknown grid
        throw(DomainError(0, "Unknown geometry"))
    end
    
end


# Module end
end
