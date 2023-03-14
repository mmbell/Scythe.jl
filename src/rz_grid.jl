#Functions for RZ Grid

struct RZ_Grid <: AbstractGrid
    params::GridParameters
    splines::Array{Spline1D}
    columns::Array{Chebyshev1D}
    spectral::Array{Float64}
    physical::Array{Float64}
end

function create_RZ_Grid(gp::GridParameters)

    # RZ is 2-D grid with splines and Chebyshev basis
    splines = Array{Spline1D}(undef,gp.b_zDim,length(values(gp.vars)))
    columns = Array{Chebyshev1D}(undef,length(values(gp.vars)))

    spectralDim = gp.b_zDim * gp.b_rDim
    spectral = zeros(Float64, spectralDim, length(values(gp.vars)))

    physical = zeros(Float64, gp.zDim * gp.rDim, length(values(gp.vars)), 5)
    grid = RZ_Grid(gp, splines, columns, spectral, physical)
    for key in keys(gp.vars)

        grid.splines[gp.vars[key]] = Spline1D(SplineParameters(
            xmin = gp.xmin,
            xmax = gp.xmax,
            num_cells = gp.num_cells,
            BCL = gp.BCL[key],
            BCR = gp.BCR[key]))

        grid.columns[gp.vars[key]] = Chebyshev1D(ChebyshevParameters(
            zmin = gp.zmin,
            zmax = gp.zmax,
            zDim = gp.zDim,
            bDim = gp.b_zDim,
            BCB = gp.BCB[key],
            BCT = gp.BCT[key]))
    end

    return grid
end

function calcTileSizes(patch::RZ_Grid, num_tiles::int)

    # Calculate the appropriate tile size for the given patch
    num_gridpoints = patch.params.rDim
    if patch.params.num_cells / num_tiles < 3.0
        throw(DomainError(num_tiles, "Too many tiles for this grid (need at least 3 cells in R direction)"))
    end

    q,r = divrem(num_gridpoints, num_tiles)
    tile_sizes = [i <= r ? q+1 : q for i = 1:num_tiles]

    # Calculate the dimensions and set the parameters
    DX = (patch.params.xmax - patch.params.xmin) / patch.params.num_cells

    xmins = zeros(Float64,num_tiles)
    xmaxs = zeros(Float64,num_tiles)
    num_cells = zeros(Int64,num_tiles)
    spectralIndicesL = ones(Int64,num_tiles)

    # Check for the special case of only 1 tile
    if num_tiles == 1
        xmins[1] = patch.params.xmin
        xmaxs[1] = patch.params.xmax
        num_cells[1] = patch.params.num_cells
        spectralIndicesL[1] = 1
        tile_sizes[1] = patch.params.rDim
        tile_params = vcat(xmins', xmaxs', num_cells', spectralIndicesL', tile_sizes')
        return tile_params
    end

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

    tile_params = vcat(xmins', xmaxs', num_cells', spectralIndicesL', tile_sizes')

    if any(x->x<3, num_cells)
        for w in 1:num_tiles
            println("Tile $w: $(tile_params[5,w]) gridpoints in $(tile_params[3,w]) cells from $(tile_params[1,w]) to $(tile_params[2,w]) starting at index $(tile_params[4,w])")
        end
        throw(DomainError(num_tiles, "Too many tiles for this grid (need at least 3 cells in R direction)"))
    end

    return tile_params
end

function getGridpoints(grid::RZ_Grid)

    # Return an array of the gridpoint locations
    gridpoints = zeros(Float64, grid.params.rDim * grid.params.zDim,2)
    g = 1
    for r = 1:grid.params.rDim
        for z = 1:grid.params.zDim
            r_m = grid.splines[1,1].mishPoints[r]
            z_m = grid.columns[1].mishPoints[z]
            gridpoints[g,1] = r_m
            gridpoints[g,2] = z_m
            g += 1
        end
    end
    return gridpoints
end

function num_columns(grid::RZ_Grid)

    return grid.params.rDim
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
    tempcb = zeros(Float64, grid.params.b_zDim, grid.params.rDim)

    for v in values(grid.params.vars)
        i = 1
        for r = 1:grid.params.rDim
            for z = 1:grid.params.zDim
                grid.columns[v].uMish[z] = physical[i,v,1]
                i += 1
            end
            tempcb[:,r] .= CBtransform!(grid.columns[v])
        end

        for z = 1:grid.params.b_zDim
            # Clear the spline
            grid.splines[v].uMish .= 0.0
            for r = 1:grid.params.rDim
                grid.splines[v].uMish[r] = tempcb[z,r]
            end
            SBtransform!(grid.splines[v])

            # Assign the spectral array
            r1 = (z-1) * grid.params.b_rDim + 1
            r2 = r1 + grid.params.b_rDim - 1
            spectral[r1:r2,v] .= grid.splines[v].b
        end
    end

    return spectral
end

function gridTransform!(grid::RZ_Grid)
    
    # Transform from the spectral to grid space
    physical = gridTransform(grid, grid.physical, grid.spectral)
    return physical
end

function gridTransform(grid::RZ_Grid, physical::Array{real}, spectral::Array{real})
    
    # Transform from the spectral to grid space
    splineBuffer = zeros(Float64, grid.params.rDim, grid.params.b_zDim)

    for v in values(grid.params.vars)
        for dr in 0:2
            for z = 1:grid.params.b_zDim
                # Wavenumber zero only
                r1 = (z-1) * grid.params.b_rDim + 1
                r2 = r1 + grid.params.b_rDim - 1
                grid.splines[v].b .= spectral[r1:r2,v]
                SAtransform!(grid.splines[v])
                if (dr == 0)
                    splineBuffer[:,z] .= SItransform!(grid.splines[v])
                elseif (dr == 1)
                    splineBuffer[:,z] .= SIxtransform(grid.splines[v])
                else
                    splineBuffer[:,z] .= SIxxtransform(grid.splines[v])
                end
            end

            for r = 1:grid.params.rDim
                for z = 1:grid.params.b_zDim
                    grid.columns[v].b[z] = splineBuffer[r,z]
                end
                CAtransform!(grid.columns[v])
                CItransform!(grid.columns[v])

                # Assign the grid array
                z1 = (r-1)*grid.params.zDim + 1
                z2 = z1 + grid.params.zDim - 1
                if (dr == 0)
                    physical[z1:z2,v,1] .= grid.columns[v].uMish
                    physical[z1:z2,v,4] .= CIxtransform(grid.columns[v])
                    physical[z1:z2,v,5] .= CIxxtransform(grid.columns[v])
                elseif (dr == 1)
                    physical[z1:z2,v,2] .= grid.columns[v].uMish
                elseif (dr == 2)
                    physical[z1:z2,v,3] .= grid.columns[v].uMish
                end
            end
        end
    end

    return physical
end

function gridTransform!(patch::RZ_Grid, tile::RZ_Grid)

    splineBuffer = zeros(Float64, patch.params.rDim, grid.params.b_zDim)
    physical = gridTransform(patch.splines, patch.spectral, patch.params, tile, splineBuffer)
    return physical
end

function gridTransform!(patchSplines::Array{Spline1D}, patchSpectral::Array{Float64}, pp::GridParameters, tile::RZ_Grid, splineBuffer::Array{Float64})

    # Transform from the spectral to grid space
    #splineBuffer = zeros(Float64, grid.params.rDim, grid.params.b_zDim)
    
    for v in values(grid.params.vars)
        for dr in 0:2
            for z = 1:pp.b_zDim
                # Wavenumber zero only
                r1 = (z-1) * pp.b_rDim + 1
                r2 = r1 + pp.b_rDim - 1
                patchSplines[v].b .= patchSpectral[r1:r2,v]
                SAtransform!(patchSplines[v])
                if (dr == 0)
                    splineBuffer[:,z] .= SItransform!(patchSplines[v])
                elseif (dr == 1)
                    splineBuffer[:,z] .= SIxtransform(patchSplines[v])
                else
                    splineBuffer[:,z] .= SIxxtransform(patchSplines[v])
                end
            end

            for r = 1:tile.params.rDim
                ri = r + tile.params.patchOffsetL
                for z = 1:pp.b_zDim
                    tile.columns[v].b[z] = splineBuffer[ri,z]
                end
                CAtransform!(tile.columns[v])
                CItransform!(tile.columns[v])

                # Assign the grid array
                z1 = (r-1)*pp.zDim + 1
                z2 = z1 + pp.zDim - 1
                if (dr == 0)
                    tile.physical[z1:z2,v,1] .= tile.columns[v].uMish
                    tile.physical[z1:z2,v,4] .= CIxtransform(tile.columns[v])
                    tile.physical[z1:z2,v,5] .= CIxxtransform(tile.columns[v])
                elseif (dr == 1)
                    tile.physical[z1:z2,v,2] .= tile.columns[v].uMish
                elseif (dr == 2)
                    tile.physical[z1:z2,v,3] .= tile.columns[v].uMish
                end
            end
        end
    end

    return tile.physical
end

function splineTransform!(patchSplines::Array{Spline1D}, patchSpectral::Array{Float64}, pp::GridParameters, sharedSpectral::SharedArray{Float64},tile::RZ_Grid)

    # Do a partial transform from B to A for splines only
    for v in values(pp.vars)
        r1 = 1
        for z in 1:pp.b_zDim
            r2 = r1 + pp.b_rDim - 1
            patchSpectral[r1:r2,v] .= SAtransform(patchSplines[v], view(sharedSpectral,r1:r2,v))
            r1 = r2 + 1
        end
    end
end

function tileTransform!(patchSplines::Array{Spline1D}, patchSpectral::Array{Float64}, pp::GridParameters, tile::RZ_Grid, splineBuffer::Array{Float64})

    # Transform from the spectral to grid space
    #splineBuffer = zeros(Float64, tile.params.rDim, pp.b_zDim)

    for v in values(pp.vars)
        for dr in 0:2
            for z = 1:pp.b_zDim
                # Wavenumber zero
                r1 = (z-1) * pp.b_rDim + 1
                r2 = r1 + pp.b_rDim - 1
                patchSplines[v].a .= view(patchSpectral,r1:r2,v)
                if (dr == 0)
                    SItransform(patchSplines[v],tile.splines[1].mishPoints,view(splineBuffer,:,z))
                elseif (dr == 1)
                    SIxtransform(patchSplines[v],tile.splines[1].mishPoints,view(splineBuffer,:,z))
                else
                    SIxxtransform(patchSplines[v],tile.splines[1].mishPoints,view(splineBuffer,:,z))
                end
            end

            for r = 1:tile.params.rDim
                for z = 1:pp.b_zDim
                    tile.columns[v].b[z] = splineBuffer[r,z]
                end
                CAtransform!(tile.columns[v])
                CItransform!(tile.columns[v])

                # Assign the grid array
                z1 = (r-1)*pp.zDim + 1
                z2 = z1 + pp.zDim - 1
                if (dr == 0)
                    tile.physical[z1:z2,v,1] .= tile.columns[v].uMish
                    tile.physical[z1:z2,v,4] .= CIxtransform(tile.columns[v])
                    tile.physical[z1:z2,v,5] .= CIxxtransform(tile.columns[v])
                elseif (dr == 1)
                    tile.physical[z1:z2,v,2] .= tile.columns[v].uMish
                elseif (dr == 2)
                    tile.physical[z1:z2,v,3] .= tile.columns[v].uMish
                end
            end
        end
    end

    return tile.physical
end

function spectralxTransform(grid::RZ_Grid, physical::Array{real}, spectral::Array{real})
    
    # Not yet implemented

end

function calcPatchMap(patch::RZ_Grid, tile::RZ_Grid)

    patchMap = falses(size(patch.spectral))
    tileView = falses(size(tile.spectral))

    # Indices of sharedArray that won't be touched by other workers
    spectralIndexL = tile.params.spectralIndexL
    patchRstride = patch.params.b_rDim
    tileRstride = tile.params.b_rDim
    tileShare = tileRstride - 4

    for z = 1:tile.params.b_zDim
        p0 = spectralIndexL + (z-1)*patchRstride
        p1 = p0
        p2 = p1 + tileShare
        patchMap[p1:p2,:] .= true

        t0 = 1 + (z-1)*tileRstride
        t1 = t0
        t2 = t1 + tileShare
        tileView[t1:t2, :] .= true
    end

    return patchMap, view(tile.spectral, tileView)
end

function calcHaloMap(patch::RZ_Grid, tile::RZ_Grid)

    patchMap = falses(size(patch.spectral))
    tileView = falses(size(tile.spectral))

    # Indices of sharedArray that won't be touched by other workers
    spectralIndexL = tile.params.spectralIndexL
    patchRstride = patch.params.b_rDim
    tileRstride = tile.params.b_rDim
    # Index is 1 more than shared map
    tileShare = tileRstride - 3

    for z = 1:tile.params.b_zDim
        p0 = spectralIndexL + (z-1)*patchRstride
        p1 = p0 + tileShare
        p2 = p1 + 2
        patchMap[p1:p2,:] .= true

        t0 = 1 + (z-1)*tileRstride
        t1 = t0 + tileShare
        t2 = t1 + 2
        tileView[t1:t2, :] .= true
    end

    return patchMap, view(tile.spectral, tileView)
end


function regularGridTransform(grid::RZ_Grid)
    
    # Output on regular grid
    # Output on the nodes and on even levels
    # TBD
end

function getRegularGridpoints(grid::RZ_Grid)

    # Return an array of regular gridpoint locations
    # TBD
end

function allocateSplineBuffer(patch::RZ_Grid, tile::RZ_Grid)

    splineBuffer = zeros(Float64, tile.params.rDim, tile.params.b_zDim)
end
