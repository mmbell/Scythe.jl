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
export AbstractGrid, R_Grid, RZ_Grid, RL_Grid, RLZ_Grid
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
    b_zDim::int = min(zDim, floor(((2 * zDim) - 1) / 3) + 1)
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
include("RLZ_Grid.jl")

# Not yet implemented
struct Z_Grid <: AbstractGrid
    params::GridParameters
    columns::Array{Chebyshev1D}
    spectral::Array{Float64}
    physical::Array{Float64}
end

function createGrid(gp::GridParameters)

    # Call the respective grid factory
    if gp.geometry == "R"
        # R grid
        grid = create_R_Grid(gp)
        return grid

    elseif gp.geometry == "RZ"
        # RZ grid
        grid = create_RZ_Grid(gp)
        return grid

    elseif gp.geometry == "RL"
        # RL grid
        grid = create_RL_Grid(gp)
        return grid

    elseif gp.geometry == "RLZ"
        # RLZ grid
        grid = create_RLZ_Grid(gp)
        return grid
        
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
