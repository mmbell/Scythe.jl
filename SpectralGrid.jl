module SpectralGrid

using CubicBSpline
using Chebyshev
using Fourier
using CSV
using DataFrames

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

struct R_Grid <: AbstractGrid
    params::GridParameters
    splines::Array{Spline1D}
    spectral::Array{Float64}
    physical::Array{Float64}
end

struct Z_Grid <: AbstractGrid
    params::GridParameters
    columns::Array{Chebyshev1D}
    spectral::Array{Float64}
    physical::Array{Float64}
end

struct RZ_Grid <: AbstractGrid
    params::GridParameters
    splines::Array{Spline1D}
    columns::Array{Chebyshev1D}
    spectral::Array{Float64}
    physical::Array{Float64}
end

struct RL_Grid <: AbstractGrid
    params::GridParameters
    splines::Array{Spline1D}
    rings::Array{Fourier1D}
    spectral::Array{Float64}
    physical::Array{Float64}
end

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

function calcTileSizes(patch::R_Grid, num_tiles::int)

    # Calculate the appropriate tile size for the given patch
    num_gridpoints = patch.params.rDim
    q,r = divrem(num_gridpoints, num_tiles)
    tile_sizes = [i <= r ? q+1 : q for i = 1:num_tiles]
    if any(x->x<15, tile_sizes)
        throw(DomainError(0, "Too many tiles for this grid (need at least 5 cells in R direction)"))
    end

    # Calculate the dimensions and set the parameters
    DX = (patch.params.xmax - patch.params.xmin) / patch.params.num_cells

    xmins = zeros(Float64,num_tiles)
    xmaxs = zeros(Float64,num_tiles)
    num_cells = zeros(Int64,num_tiles)
    spectralIndicesL = ones(Int64,num_tiles)

    # First tile starts on the patch boundary
    xmins[1] = patch.params.xmin
    num_cells[1] = Int64(ceil(tile_sizes[1] / 3))
    xmaxs[1] = (num_cells[1] * DX) + xmins[1]
    # Implicit spectralIndicesL = 1

    for i = 2:num_tiles-1
        xmins[i] = xmaxs[i-1]
        num_cells[i] = Int64(ceil(tile_sizes[i] / 3))
        xmaxs[i] = (num_cells[i] * DX) + xmins[i]
        spectralIndicesL[i] = num_cells[i-1] + spectralIndicesL[i-1]
    end

    # Last tile ends on the patch boundary
    if num_tiles > 1
        xmins[num_tiles] = xmaxs[num_tiles-1]
        xmaxs[num_tiles] = patch.params.xmax
        spectralIndicesL[num_tiles] = num_cells[num_tiles-1] + spectralIndicesL[num_tiles-1]
        num_cells[num_tiles] = patch.params.num_cells - spectralIndicesL[num_tiles] + 1
    end

    tile_params = vcat(xmins', xmaxs', num_cells', spectralIndicesL')
    return tile_params
end

function calcTileSizes(patch::RL_Grid, num_tiles::int)

    # Calculate the appropriate tile size for the given patch
    num_gridpoints = patch.params.lDim
    q,r = divrem(num_gridpoints, num_tiles)
    tile_sizes = [i <= r ? q+1 : q for i = 1:num_tiles]
    #@show tile_sizes

    # Calculate the dimensions and set the parameters
    DX = (patch.params.xmax - patch.params.xmin) / patch.params.num_cells

    xmins = zeros(Float64,num_tiles)
    xmaxs = zeros(Float64,num_tiles)
    num_cells = zeros(Int64,num_tiles)
    spectralIndicesL = ones(Int64,num_tiles)
    patchOffsetsL = zeros(Int64,num_tiles)

    # First tile starts on the patch boundary
    xmins[1] = patch.params.xmin
    # Implicit spectralIndicesL = 1
    # Implicit patchOffsetsL = 0

    # Find the first R that has enough points
    lpoints = 0
    # Set the threshold to the end just in case it is only 1 tile
    r_thresh = patch.params.rDim
    for r = 1:patch.params.rDim
        lpoints += 4 + 4*r
        if lpoints >= tile_sizes[1] && (r-1) % 3 == 0
            r_thresh = r-1
            break
        end
    end
    num_cells[1] = Int64(floor(r_thresh / 3))
    xmaxs[1] = (num_cells[1] * DX) + xmins[1]

    # Loop through other tiles
    for i = 2:num_tiles-1
        xmins[i] = xmaxs[i-1]
        lpoints = 0
        r_thresh = patch.params.rDim
        spectralIndicesL[i] = num_cells[i-1] + spectralIndicesL[i-1]
        patchOffsetsL[i] = (spectralIndicesL[i] - 1) * 3
        for r = (patchOffsetsL[i] + 1):patch.params.rDim
            lpoints += 4 + 4*r
            if lpoints >= tile_sizes[i] && (r-1) % 3 == 0
                r_thresh = r-1
                break
            end
        end
        num_cells[i] = Int64(floor(r_thresh / 3)) - spectralIndicesL[i] + 1
        xmaxs[i] = (num_cells[i] * DX) + xmins[i]
    end

    # Last tile ends on the patch boundary
    if num_tiles > 1
        xmins[num_tiles] = xmaxs[num_tiles-1]
        xmaxs[num_tiles] = patch.params.xmax
        spectralIndicesL[num_tiles] = num_cells[num_tiles-1] + spectralIndicesL[num_tiles-1]
        num_cells[num_tiles] = patch.params.num_cells - spectralIndicesL[num_tiles] + 1
    end

    if any(x->x<5, num_cells)
        throw(DomainError(0, "Too many tiles for this grid (need at least 5 cells in R direction)"))
    end

    tile_params = vcat(xmins', xmaxs', num_cells', spectralIndicesL')
    return tile_params
end

function getGridpoints(grid::R_Grid)

    # Return an array of the gridpoint locations
    return grid.splines[1].mishPoints
end

function getGridpoints(grid::RZ_Grid)

    # Return an array of the gridpoint locations
    gridpoints = zeros(Float64, grid.params.rDim * grid.params.zDim,2)
    g = 1
    for z = 1:grid.params.zDim
        for r = 1:grid.params.rDim
            r_m = grid.splines[1,1].mishPoints[r]
            z_m = grid.columns[1].mishPoints[z]
            gridpoints[g,1] = r_m
            gridpoints[g,2] = z_m
            g += 1
        end
    end
    return gridpoints
end

function getGridpoints(grid::RL_Grid)

    # Return an array of the gridpoint locations
    gridpoints = zeros(Float64, grid.params.lDim,2)
    g = 1
    for r = 1:grid.params.rDim
        r_m = grid.splines[1,1].mishPoints[r]
        ri = r + grid.params.patchOffsetL
        lpoints = 4 + 4*ri
        for l = 1:lpoints
            l_m = grid.rings[r,1].mishPoints[l]
            gridpoints[g,1] = r_m
            gridpoints[g,2] = l_m
            g += 1
        end
    end
    return gridpoints
end

function getCartesianGridpoints(grid::RL_Grid)

    gridpoints = zeros(Float64, grid.params.lDim,2)
    g = 1
    radii = grid.splines[1,1].mishPoints
    for r = 1:length(radii)
        angles = grid.rings[r,1].mishPoints
        for l = 1:length(angles)
            gridpoints[g,1] = radii[r] * cos(angles[l])
            gridpoints[g,2] = radii[r] * sin(angles[l])
            g += 1
        end
    end
    return gridpoints
end

function spectralTransform!(grid::R_Grid)
    
    # Transform from the grid to spectral space
    # For R grid, the only varying dimension is the variable name
    spectral = spectralTransform(grid, grid.physical, grid.spectral)
    return spectral
end

function spectralTransform(grid::R_Grid, physical::Array{real}, spectral::Array{real})
    
    # Transform from the grid to spectral space
    # For R grid, the only varying dimension is the variable name
    for i in eachindex(grid.splines)
        b = SBtransform(grid.splines[i], physical[:,i,1])
        
        # Assign the spectral array
        spectral[:,i] .= b
    end
end

function spectralxTransform(grid::R_Grid, physical::Array{real}, spectral::Array{real})
    
    # Transform from the grid to spectral space
    # For R grid, the only varying dimension is the variable name
    # Need to use a R0 BC for this!
    Fspline = Spline1D(SplineParameters(xmin = grid.params.xmin, 
            xmax = grid.params.xmax,
            num_cells = grid.params.num_cells, 
            BCL = CubicBSpline.R0, 
            BCR = CubicBSpline.R0))

    for i in eachindex(grid.splines)
        b = SBtransform(Fspline, physical[:,i,1])
        a = SAtransform(Fspline, b)
        Fx = SIxtransform(Fspline, a)
        bx = SBtransform(Fspline, Fx)
        
        # Assign the spectral array
        spectral[:,i] .= bx
    end
end

function gridTransform!(grid::R_Grid)
    
    # Transform from the spectral to grid space
    # For R grid, the only varying dimension is the variable name
    physical = gridTransform(grid, grid.physical, grid.spectral)
    return physical
end

function gridTransform(grid::R_Grid, physical::Array{real}, spectral::Array{real})
    
    # Transform from the spectral to grid space
    # For R grid, the only varying dimension is the variable name
    for i in eachindex(grid.splines)
        grid.splines[i].b .= spectral[:,i]
        SAtransform!(grid.splines[i])
        SItransform!(grid.splines[i])
        
        # Assign the grid array
        physical[:,i,1] .= grid.splines[i].uMish
        physical[:,i,2] .= SIxtransform(grid.splines[i])
        physical[:,i,3] .= SIxxtransform(grid.splines[i])
    end
    
    return physical 
end

function gridTransform!(patch::R_Grid, tile::R_Grid)

    # Transform from the spectral to grid space
    # For R grid, the only varying dimension is the variable name
    # Have to use the patch spline and spectral array
    for i in eachindex(patch.splines)
        patch.splines[i].b .= patch.spectral[:,i]
        SAtransform!(patch.splines[i])
        SItransform!(patch.splines[i])

        # Assign to the tile grid
        u1 = tile.params.patchOffsetL + 1
        u2 = tile.params.patchOffsetR
        tile.physical[:,i,1] .= patch.splines[i].uMish[u1:u2]
        tile.physical[:,i,2] .= SIxtransform(patch.splines[i])[u1:u2]
        tile.physical[:,i,3] .= SIxxtransform(patch.splines[i])[u1:u2]
    end

    return tile.physical
end

function gridTransform!(patchSplines::Array{Spline1D}, patchSpectral::Array{Float64}, pp::GridParameters, tile::R_Grid)

    # Transform from the spectral to grid space
    # For R grid, the only varying dimension is the variable name
    # Have to use the patch spline and spectral array

    # pp::GridParameters is patch parameters, but this is not needed for 1D case
    # It is retained for compatibility with calling function for more complex cases

    for i in eachindex(patchSplines)
        patchSplines[i].b .= patchSpectral[:,i]
        SAtransform!(patchSplines[i])
        SItransform!(patchSplines[i])

        # Assign to the tile grid
        u1 = tile.params.patchOffsetL + 1
        u2 = tile.params.patchOffsetR
        tile.physical[:,i,1] .= patchSplines[i].uMish[u1:u2]
        tile.physical[:,i,2] .= SIxtransform(patchSplines[i])[u1:u2]
        tile.physical[:,i,3] .= SIxxtransform(patchSplines[i])[u1:u2]
    end

    return tile.physical
end

function sumSpectralTile!(patch::R_Grid, tile::R_Grid)

    spectral = sumSpectralTile(patch.spectral, tile.spectral, tile.params.spectralIndexL, tile.params.spectralIndexR)
    return spectral
end

function sumSpectralTile(spectral_patch::Array{real}, spectral_tile::Array{real},
                         spectralIndexL::int, spectralIndexR::int)

    # Add the tile b's to the patch
    spectral_patch[spectralIndexL:spectralIndexR,:] =
        spectral_patch[spectralIndexL:spectralIndexR,:] .+ spectral_tile[:,:]
    return spectral_patch
end

function setSpectralTile!(patch::R_Grid, tile::R_Grid)

    spectral = setSpectralTile(patch.spectral, tile.spectral, tile.params.spectralIndexL, tile.params.spectralIndexR)
    return spectral
end

function setSpectralTile(patchSpectral::Array{real}, pp::GridParameters, tile::R_Grid)

    # pp::GridParameters is patch parameters, but this is not needed for 1D case
    # It is retained for compatibility with calling function for more complex cases

    # Clear the patch
    patchSpectral[:] .= 0.0

    spectralIndexL = tile.params.spectralIndexL
    spectralIndexR = tile.params.spectralIndexR

    # Add the tile b's to the patch
    patchSpectral[spectralIndexL:spectralIndexR,:] .= tile.spectral[:,:]
    return patchSpectral
end

function spectralTransform!(grid::RZ_Grid)
    
    # Transform from the RZ grid to spectral space
    # For RZ grid, varying dimensions are R, Z, and variable
    spectral = spectralTransform(grid, grid.physical, grid.spectral)
    return spectral
end

function spectralTransform(grid::RZ_Grid, physical::Array{real}, spectral::Array{real})
    
    # Transform from the RZ grid to spectral space
    # For RZ grid, varying dimensions are R, Z, and variable
    for v in values(grid.params.vars)
        i = 1
        for z = 1:grid.params.zDim
            for r = 1:grid.params.rDim
                grid.splines[z,v].uMish[r] = physical[i,v,1]
                i += 1
            end
            SBtransform!(grid.splines[z,v])
        end

        for r = 1:grid.params.b_rDim
            for z = 1:grid.params.zDim
                grid.columns[v].uMish[z] = grid.splines[z,v].b[r]
            end
            CBtransform!(grid.columns[v])

            # Assign the spectral array
            z1 = ((r-1)*grid.params.b_zDim)+1
            z2 = r*grid.params.b_zDim
            spectral[z1:z2,v] .= grid.columns[v].b
        end
    end

    return spectral
end

function gridTransform!(grid::RZ_Grid)
    
    # Transform from the spectral to grid space
    # For RZ grid, varying dimensions are R, Z, and variable
    physical = gridTransform(grid, grid.physical, grid.spectral)
    return physical
end

function gridTransform(grid::RZ_Grid, physical::Array{real}, spectral::Array{real})
    
    # Transform from the spectral to grid space
    # For RZ grid, varying dimensions are R, Z, and variable
    for v in values(grid.params.vars)
        for r = 1:grid.params.b_rDim
            z1 = ((r-1)*grid.params.b_zDim)+1
            z2 = r*grid.params.b_zDim
            grid.columns[v].b .= spectral[z1:z2,v]
            CAtransform!(grid.columns[v])
            CItransform!(grid.columns[v])
            
            for z = 1:grid.params.zDim
                grid.splines[z,v].b[r] = grid.columns[v].uMish[z]
            end    
        end
        
        for z = 1:grid.params.zDim
            SAtransform!(grid.splines[z,v])
            SItransform!(grid.splines[z,v])
            
            # Assign the grid array
            r1 = ((z-1)*grid.params.rDim)+1
            r2 = z*grid.params.rDim
            physical[r1:r2,v,1] .= grid.splines[z,v].uMish
            physical[r1:r2,v,2] .= SIxtransform(grid.splines[z,v])
            physical[r1:r2,v,3] .= SIxxtransform(grid.splines[z,v])
        end
        
        # Get the vertical derivatives
        var = reshape(physical[:,v,1],grid.params.rDim,grid.params.zDim)
        for r = 1:grid.params.rDim
            grid.columns[v].uMish .= var[r,:]
            CBtransform!(grid.columns[v])
            CAtransform!(grid.columns[v])
            varz = CIxtransform(grid.columns[v])
            varzz = CIxxtransform(grid.columns[v])

            # Assign the grid array
            for z = 1:grid.params.zDim
                ri = (z-1)*grid.params.rDim + r
                physical[ri,v,4] = varz[z]
                physical[ri,v,5] = varzz[z]
            end
        end
        
    end
    
    return grid.physical 
end

function spectralTransform_old(grid::RZ_Grid, physical::Array{real}, spectral::Array{real})
    
    # Transform from the RZ grid to spectral space
    # For RZ grid, varying dimensions are R, Z, and variable
    
    # Regular splines are OK here since BCs are only applied on grid transform
    
    varRtmp = zeros(Float64,grid.params.rDim)
    varZtmp = zeros(Float64,grid.params.zDim)
    spectraltmp = zeros(Float64,grid.params.zDim * grid.params.b_rDim,
        length(values(grid.params.vars)))
    for v in values(grid.params.vars)
        i = 1
        for z = 1:grid.params.zDim
            for r = 1:grid.params.rDim
                varRtmp[r] = physical[i,v,1]
                i += 1
            end
            b = SBtransform(grid.splines[z,v],varRtmp)
            
            # Assign a temporary spectral array
            r1 = ((z-1)*grid.params.b_rDim)+1
            r2 = z*grid.params.b_rDim
            spectraltmp[r1:r2,v] .= b
        end

        for r = 1:grid.params.b_rDim
            for z = 1:grid.params.zDim
                ri = ((z-1)*grid.params.b_rDim)+r
                varZtmp[z] = spectraltmp[ri,v]
            end
            b = CBtransform(grid.columns[v], varZtmp)
            
            # Assign the spectral array
            z1 = ((r-1)*grid.params.b_zDim)+1
            z2 = r*grid.params.b_zDim
            spectral[z1:z2,v] .= b
        end
    end

    return spectral
end

function gridTransform_noBCs(grid::RZ_Grid, physical::Array{real}, spectral::Array{real})
    
    # Transform from the spectral to grid space
    # For RZ grid, varying dimensions are R, Z, and variable
    # Need to use a R0 BC for this since there is no guarantee 
    # that tendencies should match the underlying variable 
    splines = Array{Spline1D}(undef,grid.params.zDim)
    for z = 1:grid.params.zDim
        splines[z] = Spline1D(SplineParameters(
            xmin = grid.params.xmin, 
            xmax = grid.params.xmax,
            num_cells = grid.params.num_cells, 
            BCL = CubicBSpline.R0, 
            BCR = CubicBSpline.R0))
    end
    column = Chebyshev1D(ChebyshevParameters(
            zmin = grid.params.zmin,
            zmax = grid.params.zmax,
            zDim = grid.params.zDim,
            bDim = grid.params.b_zDim,
            BCB = Chebyshev.R0,
            BCT = Chebyshev.R0))
    for v in values(grid.params.vars)
        for r = 1:grid.params.b_rDim
            z1 = ((r-1)*grid.params.b_zDim)+1
            z2 = r*grid.params.b_zDim
            column.b .= spectral[z1:z2,v]
            CAtransform!(column)
            CItransform!(column)
            
            for z = 1:grid.params.zDim
                splines[z].b[r] = column.uMish[z]
            end    
        end
        
        for z = 1:grid.params.zDim
            SAtransform!(splines[z])
            SItransform!(splines[z])
            
            # Assign the grid array
            r1 = ((z-1)*grid.params.rDim)+1
            r2 = z*grid.params.rDim
            physical[r1:r2,v,1] .= splines[z].uMish
            physical[r1:r2,v,2] .= SIxtransform(splines[z])
            physical[r1:r2,v,3] .= SIxxtransform(splines[z])
        end
        
        # Get the vertical derivatives
        var = reshape(physical[:,v,1],grid.params.rDim,grid.params.zDim)
        for r = 1:grid.params.rDim
            column.uMish .= var[r,:]
            CBtransform!(column)
            CAtransform!(column)
            varz = CIxtransform(column)
            varzz = CIxxtransform(column)

            # Assign the grid array
            for z = 1:grid.params.zDim
                ri = (z-1)*grid.params.rDim + r
                physical[ri,v,4] = varz[z]
                physical[ri,v,5] = varzz[z]
            end
        end
    end
    
    return physical 
end

function integrateUp(grid::RZ_Grid, physical::Array{real}, spectral::Array{real})
    
    # Transform from the spectral to grid space
    # For RZ grid, varying dimensions are R, Z, and variable
    # Need to use a R0 BC for this since there is no guarantee 
    # that tendencies should match the underlying variable 
    splines = Array{Spline1D}(undef,grid.params.zDim)
    for z = 1:grid.params.zDim
        splines[z] = Spline1D(SplineParameters(
            xmin = grid.params.xmin, 
            xmax = grid.params.xmax,
            num_cells = grid.params.num_cells, 
            BCL = CubicBSpline.R0, 
            BCR = CubicBSpline.R0))
    end
    column = Chebyshev1D(ChebyshevParameters(
            zmin = grid.params.zmin,
            zmax = grid.params.zmax,
            zDim = grid.params.zDim,
            bDim = grid.params.b_zDim,
            BCB = Chebyshev.R0,
            BCT = Chebyshev.R0))
    for r = 1:grid.params.b_rDim
        z1 = ((r-1)*grid.params.b_zDim)+1
        z2 = r*grid.params.b_zDim
        column.b .= spectral[z1:z2]
        CAtransform!(column)
        w = CIInttransform(column)

        for z = 1:grid.params.zDim
            splines[z].b[r] = w[z]
        end    
    end

    for z = 1:grid.params.zDim
        SAtransform!(splines[z])
        SItransform!(splines[z])

        # Assign the grid array
        r1 = ((z-1)*grid.params.rDim)+1
        r2 = z*grid.params.rDim
        physical[r1:r2] .= splines[z].uMish
    end
    
    return physical 
end


function spectralxTransform(grid::RZ_Grid, physical::Array{real}, spectral::Array{real})
    #To be implemented for delayed diffusion
end

function spectralTransform!(grid::RL_Grid)
    
    # Transform from the RL grid to spectral space
    # For RL grid, varying dimensions are R, L, and variable
    spectral = spectralTransform(grid, grid.physical, grid.spectral)
    return spectral
end

function spectralTransform(grid::RL_Grid, physical::Array{real}, spectral::Array{real})
    
    # Transform from the RL grid to spectral space
    # For RL grid, varying dimensions are R, L, and variable

    # Need to include patchOffset to get all available wavenumbers
    kDim = grid.params.rDim + grid.params.patchOffsetL

    for v in values(grid.params.vars)
        i = 1
        for r = 1:grid.params.rDim
            ri = r + grid.params.patchOffsetL
            lpoints = 4 + 4*ri
            for l = 1:lpoints
                grid.rings[r,v].uMish[l] = physical[i,v,1]
                i += 1
            end
            FBtransform!(grid.rings[r,v])
        end

        # Clear the wavenumber zero spline
        grid.splines[1,v].uMish .= 0.0
        for r = 1:grid.params.rDim
            # Wavenumber zero
            grid.splines[1,v].uMish[r] = grid.rings[r,v].b[1]
        end
        SBtransform!(grid.splines[1,v])
        
        # Assign the spectral array
        k1 = 1
        k2 = grid.params.b_rDim
        spectral[k1:k2,v] .= grid.splines[1,v].b

        for k = 1:kDim
            # Clear the splines
            grid.splines[2,v].uMish .= 0.0
            grid.splines[3,v].uMish .= 0.0
            for r = 1:grid.params.rDim
                if (k <= r + grid.params.patchOffsetL)
                    # Real part
                    rk = k+1
                    # Imaginary part
                    ik = grid.rings[r,v].params.bDim-k+1
                    grid.splines[2,v].uMish[r] = grid.rings[r,v].b[rk]
                    grid.splines[3,v].uMish[r] = grid.rings[r,v].b[ik]
                end
            end
            SBtransform!(grid.splines[2,v])
            SBtransform!(grid.splines[3,v])
            
            # Assign the spectral array
            # For simplicity, just stack the real and imaginary parts one after the other
            p = k*2
            p1 = ((p-1)*grid.params.b_rDim)+1
            p2 = p*grid.params.b_rDim
            spectral[p1:p2,v] .= grid.splines[2,v].b
            
            p1 = (p*grid.params.b_rDim)+1
            p2 = (p+1)*grid.params.b_rDim
            spectral[p1:p2,v] .= grid.splines[3,v].b
        end
    end

    return spectral
end

function gridTransform!(grid::RL_Grid)
    
    # Transform from the spectral to grid space
    # For RZ grid, varying dimensions are R, Z, and variable
    physical = gridTransform(grid, grid.physical, grid.spectral)
    return physical 
end

function gridTransform(grid::RL_Grid, physical::Array{real}, spectral::Array{real})
    
    # Transform from the spectral to grid space
    # For RL grid, varying dimensions are R, L, and variable

    # Need to include patchOffset to get all available wavenumbers
    kDim = grid.params.rDim + grid.params.patchOffsetL
    spline_r = zeros(Float64, grid.params.rDim, kDim*2+1)
    spline_rr = zeros(Float64, grid.params.rDim, kDim*2+1)
    
    for v in values(grid.params.vars)
        # Wavenumber zero
        k1 = 1
        k2 = grid.params.b_rDim
        grid.splines[1,v].b .= spectral[k1:k2,v]
        SAtransform!(grid.splines[1,v])
        SItransform!(grid.splines[1,v])
        spline_r[:,1] = SIxtransform(grid.splines[1,v])
        spline_rr[:,1] = SIxxtransform(grid.splines[1,v])
        
        for r = 1:grid.params.rDim
            grid.rings[r,v].b[1] = grid.splines[1,v].uMish[r]
        end
        
        # Higher wavenumbers
        for k = 1:kDim
            p = k*2
            p1 = ((p-1)*grid.params.b_rDim)+1
            p2 = p*grid.params.b_rDim
            grid.splines[2,v].b .= spectral[p1:p2,v]
            SAtransform!(grid.splines[2,v])
            SItransform!(grid.splines[2,v])
            spline_r[:,p] = SIxtransform(grid.splines[2,v])
            spline_rr[:,p] = SIxxtransform(grid.splines[2,v])
            
            p1 = (p*grid.params.b_rDim)+1
            p2 = (p+1)*grid.params.b_rDim
            grid.splines[3,v].b .= spectral[p1:p2,v]
            SAtransform!(grid.splines[3,v])
            SItransform!(grid.splines[3,v])
            spline_r[:,p+1] = SIxtransform(grid.splines[3,v])
            spline_rr[:,p+1] = SIxxtransform(grid.splines[3,v])
            
            for r = 1:grid.params.rDim
                if (k <= r + grid.params.patchOffsetL)
                    # Real part
                    rk = k+1
                    # Imaginary part
                    ik = grid.rings[r,v].params.bDim-k+1
                    grid.rings[r,v].b[rk] = grid.splines[2,v].uMish[r]
                    grid.rings[r,v].b[ik] = grid.splines[3,v].uMish[r]
                end
            end
        end
        
        l1 = 0
        l2 = 0
        for r = 1:grid.params.rDim
            FAtransform!(grid.rings[r,v])
            FItransform!(grid.rings[r,v])
            
            # Assign the grid array
            ri = r + grid.params.patchOffsetL
            l1 = l2 + 1
            l2 = l1 + 3 + (4*ri)
            physical[l1:l2,v,1] .= grid.rings[r,v].uMish
            physical[l1:l2,v,4] .= FIxtransform(grid.rings[r,v])
            physical[l1:l2,v,5] .= FIxxtransform(grid.rings[r,v])
        end

        # 1st radial derivative
        # Wavenumber zero
        for r = 1:grid.params.rDim
            grid.rings[r,v].b[1] = spline_r[r,1]
        end
        
        # Higher wavenumbers
        for k = 1:kDim
            p = k*2
            for r = 1:grid.params.rDim
                if (k <= r + grid.params.patchOffsetL)
                    # Real part
                    rk = k+1
                    # Imaginary part
                    ik = grid.rings[r,v].params.bDim-k+1
                    grid.rings[r,v].b[rk] = spline_r[r,p]
                    grid.rings[r,v].b[ik] = spline_r[r,p+1]
                end
            end
        end
        
        l1 = 0
        l2 = 0
        for r = 1:grid.params.rDim
            FAtransform!(grid.rings[r,v])
            FItransform!(grid.rings[r,v])
            
            # Assign the grid array
            ri = r + grid.params.patchOffsetL
            l1 = l2 + 1
            l2 = l1 + 3 + (4*ri)
            physical[l1:l2,v,2] .= grid.rings[r,v].uMish
        end
        
        # 2nd radial derivative
        # Wavenumber zero
        for r = 1:grid.params.rDim
            grid.rings[r,v].b[1] = spline_rr[r,1]
        end
        
        # Higher wavenumbers
        for k = 1:kDim
            p = k*2
            for r = 1:grid.params.rDim
                if (k <= r + grid.params.patchOffsetL)
                    # Real part
                    rk = k+1
                    # Imaginary part
                    ik = grid.rings[r,v].params.bDim-k+1
                    grid.rings[r,v].b[rk] = spline_rr[r,p]
                    grid.rings[r,v].b[ik] = spline_rr[r,p+1]
                end
            end
        end
        
        l1 = 0
        l2 = 0
        for r = 1:grid.params.rDim
            FAtransform!(grid.rings[r,v])
            FItransform!(grid.rings[r,v])
            
            # Assign the grid array
            ri = r + grid.params.patchOffsetL
            l1 = l2 + 1
            l2 = l1 + 3 + (4*ri)
            physical[l1:l2,v,3] .= grid.rings[r,v].uMish
        end

    end    
    return physical 
end

function gridTransform!(patch::RL_Grid, tile::RL_Grid)

    # Transform from the spectral to grid space
    # For RL grid, varying dimensions are R, L, and variable
    spline_r = zeros(Float64, patch.params.rDim, patch.params.rDim*2+1)
    spline_rr = zeros(Float64, patch.params.rDim, patch.params.rDim*2+1)

    for v in values(patch.params.vars)
        # Wavenumber zero
        k1 = 1
        k2 = patch.params.b_rDim
        patch.splines[1,v].b .= patch.spectral[k1:k2,v]
        SAtransform!(patch.splines[1,v])
        SItransform!(patch.splines[1,v])
        spline_r[:,1] = SIxtransform(patch.splines[1,v])
        spline_rr[:,1] = SIxxtransform(patch.splines[1,v])

        for r = 1:tile.params.rDim
            # Offset physical index
            r1 = r + tile.params.patchOffsetL
            tile.rings[r,v].b[1] = patch.splines[1,v].uMish[r1]
        end

        # Higher wavenumbers
        for k = 1:patch.params.rDim
            p = k*2
            p1 = ((p-1)*patch.params.b_rDim)+1
            p2 = p*patch.params.b_rDim
            patch.splines[2,v].b .= patch.spectral[p1:p2,v]
            SAtransform!(patch.splines[2,v])
            SItransform!(patch.splines[2,v])
            spline_r[:,p] = SIxtransform(patch.splines[2,v])
            spline_rr[:,p] = SIxxtransform(patch.splines[2,v])

            p1 = (p*patch.params.b_rDim)+1
            p2 = (p+1)*patch.params.b_rDim
            patch.splines[3,v].b .= patch.spectral[p1:p2,v]
            SAtransform!(patch.splines[3,v])
            SItransform!(patch.splines[3,v])
            spline_r[:,p+1] = SIxtransform(patch.splines[3,v])
            spline_rr[:,p+1] = SIxxtransform(patch.splines[3,v])

            for r = 1:tile.params.rDim
                if (k <= r + tile.params.patchOffsetL)
                    # Real part
                    rk = k+1
                    # Imaginary part
                    ik = tile.rings[r,v].params.bDim-k+1
                    # Offset physical index
                    r1 = r + tile.params.patchOffsetL
                    tile.rings[r,v].b[rk] = patch.splines[2,v].uMish[r1]
                    tile.rings[r,v].b[ik] = patch.splines[3,v].uMish[r1]
                end
            end
        end

        l1 = 0
        l2 = 0
        for r = 1:tile.params.rDim
            FAtransform!(tile.rings[r,v])
            FItransform!(tile.rings[r,v])

            # Assign the grid array
            ri = r + tile.params.patchOffsetL
            l1 = l2 + 1
            l2 = l1 + 3 + (4*ri)
            tile.physical[l1:l2,v,1] .= tile.rings[r,v].uMish
            tile.physical[l1:l2,v,4] .= FIxtransform(tile.rings[r,v])
            tile.physical[l1:l2,v,5] .= FIxxtransform(tile.rings[r,v])
        end

        # 1st radial derivative
        # Wavenumber zero
        for r = 1:tile.params.rDim
            # Offset physical index
            r1 = r + tile.params.patchOffsetL
            tile.rings[r,v].b[1] = spline_r[r1,1]
        end

        # Higher wavenumbers
        for k = 1:patch.params.rDim
            p = k*2
            for r = 1:tile.params.rDim
                if (k <= r + tile.params.patchOffsetL)
                    # Real part
                    rk = k+1
                    # Imaginary part
                    ik = tile.rings[r,v].params.bDim-k+1
                    # Offset physical index
                    r1 = r + tile.params.patchOffsetL
                    tile.rings[r,v].b[rk] = spline_r[r1,p]
                    tile.rings[r,v].b[ik] = spline_r[r1,p+1]
                end
            end
        end

        l1 = 0
        l2 = 0
        for r = 1:tile.params.rDim
            FAtransform!(tile.rings[r,v])
            FItransform!(tile.rings[r,v])

            # Assign the grid array
            ri = r + tile.params.patchOffsetL
            l1 = l2 + 1
            l2 = l1 + 3 + (4*ri)
            tile.physical[l1:l2,v,2] .= tile.rings[r,v].uMish
        end

        # 2nd radial derivative
        # Wavenumber zero
        for r = 1:tile.params.rDim
            # Offset physical index
            r1 = r + tile.params.patchOffsetL
            tile.rings[r,v].b[1] = spline_rr[r1,1]
        end

        # Higher wavenumbers
        for k = 1:patch.params.rDim
            p = k*2
            for r = 1:tile.params.rDim
                if (k <= r + tile.params.patchOffsetL)
                    # Real part
                    rk = k+1
                    # Imaginary part
                    ik = tile.rings[r,v].params.bDim-k+1
                    # Offset physical index
                    r1 = r + tile.params.patchOffsetL
                    tile.rings[r,v].b[rk] = spline_rr[r1,p]
                    tile.rings[r,v].b[ik] = spline_rr[r1,p+1]
                end
            end
        end

        l1 = 0
        l2 = 0
        for r = 1:tile.params.rDim
            FAtransform!(tile.rings[r,v])
            FItransform!(tile.rings[r,v])

            # Assign the grid array
            ri = r + tile.params.patchOffsetL
            l1 = l2 + 1
            l2 = l1 + 3 + (4*ri)
            tile.physical[l1:l2,v,3] .= tile.rings[r,v].uMish
        end

    end
    return tile.physical
end

function gridTransform!(patchSplines::Array{Spline1D}, patchSpectral::Array{Float64}, pp::GridParameters, tile::RL_Grid)

    # Transform from the spectral to grid space
    # For RL grid, varying dimensions are R, L, and variable
    spline_r = zeros(Float64, pp.rDim, pp.rDim*2+1)
    spline_rr = zeros(Float64, pp.rDim, pp.rDim*2+1)

    for v in values(pp.vars)
        # Wavenumber zero
        k1 = 1
        k2 = pp.b_rDim
        patchSplines[1,v].b .= patchSpectral[k1:k2,v]
        SAtransform!(patchSplines[1,v])
        SItransform!(patchSplines[1,v])
        spline_r[:,1] = SIxtransform(patchSplines[1,v])
        spline_rr[:,1] = SIxxtransform(patchSplines[1,v])

        for r = 1:tile.params.rDim
            # Offset physical index
            r1 = r + tile.params.patchOffsetL
            tile.rings[r,v].b[1] = patchSplines[1,v].uMish[r1]
        end

        # Higher wavenumbers
        for k = 1:pp.rDim
            p = k*2
            p1 = ((p-1)*pp.b_rDim)+1
            p2 = p*pp.b_rDim
            patchSplines[2,v].b .= patchSpectral[p1:p2,v]
            SAtransform!(patchSplines[2,v])
            SItransform!(patchSplines[2,v])
            spline_r[:,p] = SIxtransform(patchSplines[2,v])
            spline_rr[:,p] = SIxxtransform(patchSplines[2,v])

            p1 = (p*pp.b_rDim)+1
            p2 = (p+1)*pp.b_rDim
            patchSplines[3,v].b .= patchSpectral[p1:p2,v]
            SAtransform!(patchSplines[3,v])
            SItransform!(patchSplines[3,v])
            spline_r[:,p+1] = SIxtransform(patchSplines[3,v])
            spline_rr[:,p+1] = SIxxtransform(patchSplines[3,v])

            for r = 1:tile.params.rDim
                if (k <= r + tile.params.patchOffsetL)
                    # Real part
                    rk = k+1
                    # Imaginary part
                    ik = tile.rings[r,v].params.bDim-k+1
                    # Offset physical index
                    r1 = r + tile.params.patchOffsetL
                    tile.rings[r,v].b[rk] = patchSplines[2,v].uMish[r1]
                    tile.rings[r,v].b[ik] = patchSplines[3,v].uMish[r1]
                end
            end
        end

        l1 = 0
        l2 = 0
        for r = 1:tile.params.rDim
            FAtransform!(tile.rings[r,v])
            FItransform!(tile.rings[r,v])

            # Assign the grid array
            ri = r + tile.params.patchOffsetL
            l1 = l2 + 1
            l2 = l1 + 3 + (4*ri)
            tile.physical[l1:l2,v,1] .= tile.rings[r,v].uMish
            tile.physical[l1:l2,v,4] .= FIxtransform(tile.rings[r,v])
            tile.physical[l1:l2,v,5] .= FIxxtransform(tile.rings[r,v])
        end

        # 1st radial derivative
        # Wavenumber zero
        for r = 1:tile.params.rDim
            # Offset physical index
            r1 = r + tile.params.patchOffsetL
            tile.rings[r,v].b[1] = spline_r[r1,1]
        end

        # Higher wavenumbers
        for k = 1:pp.rDim
            p = k*2
            for r = 1:tile.params.rDim
                if (k <= r + tile.params.patchOffsetL)
                    # Real part
                    rk = k+1
                    # Imaginary part
                    ik = tile.rings[r,v].params.bDim-k+1
                    # Offset physical index
                    r1 = r + tile.params.patchOffsetL
                    tile.rings[r,v].b[rk] = spline_r[r1,p]
                    tile.rings[r,v].b[ik] = spline_r[r1,p+1]
                end
            end
        end

        l1 = 0
        l2 = 0
        for r = 1:tile.params.rDim
            FAtransform!(tile.rings[r,v])
            FItransform!(tile.rings[r,v])

            # Assign the grid array
            ri = r + tile.params.patchOffsetL
            l1 = l2 + 1
            l2 = l1 + 3 + (4*ri)
            tile.physical[l1:l2,v,2] .= tile.rings[r,v].uMish
        end

        # 2nd radial derivative
        # Wavenumber zero
        for r = 1:tile.params.rDim
            # Offset physical index
            r1 = r + tile.params.patchOffsetL
            tile.rings[r,v].b[1] = spline_rr[r1,1]
        end

        # Higher wavenumbers
        for k = 1:pp.rDim
            p = k*2
            for r = 1:tile.params.rDim
                if (k <= r + tile.params.patchOffsetL)
                    # Real part
                    rk = k+1
                    # Imaginary part
                    ik = tile.rings[r,v].params.bDim-k+1
                    # Offset physical index
                    r1 = r + tile.params.patchOffsetL
                    tile.rings[r,v].b[rk] = spline_rr[r1,p]
                    tile.rings[r,v].b[ik] = spline_rr[r1,p+1]
                end
            end
        end

        l1 = 0
        l2 = 0
        for r = 1:tile.params.rDim
            FAtransform!(tile.rings[r,v])
            FItransform!(tile.rings[r,v])

            # Assign the grid array
            ri = r + tile.params.patchOffsetL
            l1 = l2 + 1
            l2 = l1 + 3 + (4*ri)
            tile.physical[l1:l2,v,3] .= tile.rings[r,v].uMish
        end

    end
    return tile.physical
end

function spectralxTransform(grid::RL_Grid, physical::Array{real}, spectral::Array{real})
    
    #Currently just a clone to test out delayed diffusion
    spectralTransform(grid, physical, spectral)

end

function sumSpectralTile!(patch::RL_Grid, tile::RL_Grid)

    # Get the appropriate dimensions
    kDim = tile.params.rDim + tile.params.patchOffsetL
    spectralIndexL = tile.params.spectralIndexL
    spectralIndexR = tile.params.spectralIndexR

    # Add the tile b's to the patch

    # Wavenumber 0
    p1 = spectralIndexL
    p2 = spectralIndexR
    t1 = 1
    t2 = tile.params.b_rDim
    patch.spectral[p1:p2,:] = patch.spectral[p1:p2,:] .+ tile.spectral[t1:t2,:]

    # Higher wavenumbers
    for k in 1:kDim
        p = k*2
        t = k*2

        # Real part
        p1 = ((p-1)*patch.params.b_rDim) + spectralIndexL
        p2 = p1 + tile.params.b_rDim - 1
        t1 = ((t-1)*tile.params.b_rDim)+1
        t2 = t*tile.params.b_rDim
        #@show k p1 p2 t1 t2
        patch.spectral[p1:p2,:] = patch.spectral[p1:p2,:] .+ tile.spectral[t1:t2,:]

        # Imaginary part
        p1 = (p*patch.params.b_rDim) + spectralIndexL
        p2 = p1 + tile.params.b_rDim - 1
        t1 = (t*tile.params.b_rDim)+1
        t2 = (t+1)*tile.params.b_rDim
        patch.spectral[p1:p2,:] = patch.spectral[p1:p2,:] .+ tile.spectral[t1:t2,:]
    end

    return patch.spectral
end

function setSpectralTile!(patch::RL_Grid, tile::RL_Grid)

    # Clear the patch
    patch.spectral[:] .= 0.0

    # Get the appropriate dimensions
    kDim = tile.params.rDim + tile.params.patchOffsetL
    spectralIndexL = tile.params.spectralIndexL
    spectralIndexR = tile.params.spectralIndexR

    # Add the tile b's to the patch

    # Wavenumber 0
    p1 = spectralIndexL
    p2 = spectralIndexR
    t1 = 1
    t2 = tile.params.b_rDim
    patch.spectral[p1:p2,:] .= tile.spectral[t1:t2,:]

    # Higher wavenumbers
    for k in 1:kDim
        p = k*2
        t = k*2

        # Real part
        p1 = ((p-1)*patch.params.b_rDim) + spectralIndexL
        p2 = p1 + tile.params.b_rDim - 1
        t1 = ((t-1)*tile.params.b_rDim)+1
        t2 = t*tile.params.b_rDim
        #@show k p1 p2 t1 t2
        patch.spectral[p1:p2,:] .= tile.spectral[t1:t2,:]

        # Imaginary part
        p1 = (p*patch.params.b_rDim) + spectralIndexL
        p2 = p1 + tile.params.b_rDim - 1
        t1 = (t*tile.params.b_rDim)+1
        t2 = (t+1)*tile.params.b_rDim
        patch.spectral[p1:p2,:] .= tile.spectral[t1:t2,:]
    end

    return patch.spectral
end

function setSpectralTile(patchSpectral::Array{real}, pp::GridParameters, tile::RL_Grid)

    # Clear the patch
    patchSpectral[:] .= 0.0

    # Get the appropriate dimensions
    kDim = tile.params.rDim + tile.params.patchOffsetL
    spectralIndexL = tile.params.spectralIndexL
    spectralIndexR = tile.params.spectralIndexR

    # Add the tile b's to the patch

    # Wavenumber 0
    p1 = spectralIndexL
    p2 = spectralIndexR
    t1 = 1
    t2 = tile.params.b_rDim
    patchSpectral[p1:p2,:] .= tile.spectral[t1:t2,:]

    # Higher wavenumbers
    for k in 1:kDim
        p = k*2
        t = k*2

        # Real part
        p1 = ((p-1)*pp.b_rDim) + spectralIndexL
        p2 = p1 + tile.params.b_rDim - 1
        t1 = ((t-1)*tile.params.b_rDim)+1
        t2 = t*tile.params.b_rDim
        #@show k p1 p2 t1 t2
        patchSpectral[p1:p2,:] .= tile.spectral[t1:t2,:]

        # Imaginary part
        p1 = (p*pp.b_rDim) + spectralIndexL
        p2 = p1 + tile.params.b_rDim - 1
        t1 = (t*tile.params.b_rDim)+1
        t2 = (t+1)*tile.params.b_rDim
        patchSpectral[p1:p2,:] .= tile.spectral[t1:t2,:]
    end

    return patchSpectral
end

function regularGridTransform(grid::RL_Grid)
    
    # Output on regular grid
    # Transform from the spectral to grid space
    # For RZ grid, varying dimensions are R, Z, and variable
    spline = zeros(Float64, grid.params.num_cells, grid.params.rDim*2+1)
    spline_r = zeros(Float64, grid.params.num_cells, grid.params.rDim*2+1)
    spline_rr = zeros(Float64, grid.params.num_cells, grid.params.rDim*2+1)
    
    physical = zeros(Float64, grid.params.num_cells, 
        grid.params.rDim*2+1, 
        length(values(grid.params.vars)),5)

    # Generic ring of maximum size
    ring = Fourier1D(FourierParameters(
        ymin = 0.0,
        yDim = grid.params.rDim*2 + 1,
        bDim = grid.params.rDim*2 + 1,
        kmax = grid.params.rDim))

    # Output on the nodes
    rpoints = zeros(Float64, grid.params.num_cells)
    for r = 1:grid.params.num_cells
        rpoints[r] = grid.params.xmin + (r-1)*grid.splines[1,1].params.DX
    end
    
    for v in values(grid.params.vars)
        # Wavenumber zero
        k1 = 1
        k2 = grid.params.b_rDim
        a = SAtransform(grid.splines[1,v], grid.spectral[k1:k2,v])
        spline[:,1] = SItransform(grid.splines[1,v], a, rpoints)
        spline_r[:,1] = SIxtransform(grid.splines[1,v], a, rpoints)
        spline_rr[:,1] = SIxxtransform(grid.splines[1,v], a, rpoints)
        
        # Higher wavenumbers
        for k = 1:grid.params.rDim
            p = k*2
            p1 = ((p-1)*grid.params.b_rDim)+1
            p2 = p*grid.params.b_rDim
            a = SAtransform(grid.splines[2,v], grid.spectral[p1:p2,v])
            spline[:,p] = SItransform(grid.splines[2,v], a, rpoints)
            spline_r[:,p] = SIxtransform(grid.splines[2,v], a, rpoints)
            spline_rr[:,p] = SIxxtransform(grid.splines[2,v], a, rpoints)
            
            p1 = (p*grid.params.b_rDim)+1
            p2 = (p+1)*grid.params.b_rDim
            a = SAtransform(grid.splines[3,v], grid.spectral[p1:p2,v])
            spline[:,p+1] = SItransform(grid.splines[3,v], a, rpoints)
            spline_r[:,p+1] = SIxtransform(grid.splines[3,v], a, rpoints)
            spline_rr[:,p+1] = SIxxtransform(grid.splines[3,v], a, rpoints)
        end
        
        for r = 1:grid.params.num_cells
            # Value
            ring.b .= 0.0
            ring.b[1] = spline[r,1]
            for k = 1:grid.params.rDim
                # Real part
                rk = k+1
                # Imaginary part
                ik = ring.params.bDim-k+1
                p = k*2
                ring.b[rk] = spline[r,p]
                ring.b[ik] = spline[r,p+1]
            end
            FAtransform!(ring)
            l1 = 1
            l2 = ring.params.yDim
            physical[r,l1:l2,v,1] .= FItransform!(ring)
            physical[r,l1:l2,v,4] .= FIxtransform(ring)
            physical[r,l1:l2,v,5] .= FIxxtransform(ring)
            
            # dr
            ring.b .= 0.0
            ring.b[1] = spline_r[r,1]
            for k = 1:grid.params.rDim
                # Real part
                rk = k+1
                # Imaginary part
                ik = ring.params.bDim-k+1
                p = k*2
                ring.b[rk] = spline_r[r,p]
                ring.b[ik] = spline_r[r,p+1]
            end
            FAtransform!(ring)
            l1 = 1
            l2 = ring.params.yDim
            physical[r,l1:l2,v,2] .= FItransform!(ring)
            
            #drr
            ring.b .= 0.0
            ring.b[1] = spline_rr[r,1]
            for k = 1:grid.params.rDim
                # Real part
                rk = k+1
                # Imaginary part
                ik = ring.params.bDim-k+1
                p = k*2
                ring.b[rk] = spline_rr[r,p]
                ring.b[ik] = spline_rr[r,p+1]
            end
            FAtransform!(ring)
            l1 = 1
            l2 = ring.params.yDim
            physical[r,l1:l2,v,3] .= FItransform!(ring)

        end
    end
    
    
    return physical
end

function getRegularGridpoints(grid::RL_Grid)

    # Return an array of the gridpoint locations
    gridpoints = zeros(Float64, grid.params.num_cells, (grid.params.rDim*2+1), 4)
    for r = 1:grid.params.num_cells
        r_m = grid.params.xmin + (r-1)*grid.splines[1,1].params.DX
        for l = 1:(grid.params.rDim*2+1)
            l_m = 2 * π * (l-1) / (grid.params.rDim*2+1)
            gridpoints[r,l,1] = r_m
            gridpoints[r,l,2] = l_m
            gridpoints[r,l,3] = r_m * cos(l_m)
            gridpoints[r,l,4] = r_m * sin(l_m)
        end
    end
    return gridpoints
end

# Module end
end
