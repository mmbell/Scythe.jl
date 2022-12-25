#Functions for RL Grid

struct RL_Grid <: AbstractGrid
    params::GridParameters
    splines::Array{Spline1D}
    rings::Array{Fourier1D}
    spectral::Array{Float64}
    physical::Array{Float64}
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

    if any(x->x<3, num_cells)
        throw(DomainError(0, "Too many tiles for this grid (need at least 3 cells in R direction)"))
    end

    tile_params = vcat(xmins', xmaxs', num_cells', spectralIndicesL', tile_sizes')
    return tile_params
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

function gridTransform!(patchSplines::Array{Spline1D}, patchSpectral::Array{Float64}, pp::GridParameters, tile::RL_Grid, splineBuffer::Array{Float64})

    # Transform from the spectral to grid space
    # For RL grid, varying dimensions are R, L, and variable
    #spline_r = zeros(Float64, pp.rDim, pp.rDim*2+1)
    #spline_rr = zeros(Float64, pp.rDim, pp.rDim*2+1)

    for v in values(pp.vars)
        # Wavenumber zero
        k1 = 1
        k2 = pp.b_rDim
        patchSplines[1,v].b .= patchSpectral[k1:k2,v]
        SAtransform!(patchSplines[1,v])
        SItransform!(patchSplines[1,v])
        splineBuffer[:,1,1] = SIxtransform(patchSplines[1,v])
        splineBuffer[:,1,2] = SIxxtransform(patchSplines[1,v])

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
            splineBuffer[:,p,1] = SIxtransform(patchSplines[2,v])
            splineBuffer[:,p,2] = SIxxtransform(patchSplines[2,v])

            p1 = (p*pp.b_rDim)+1
            p2 = (p+1)*pp.b_rDim
            patchSplines[3,v].b .= patchSpectral[p1:p2,v]
            SAtransform!(patchSplines[3,v])
            SItransform!(patchSplines[3,v])
            splineBuffer[:,p+1,1] = SIxtransform(patchSplines[3,v])
            splineBuffer[:,p+1,2] = SIxxtransform(patchSplines[3,v])

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
            tile.rings[r,v].b[1] = splineBuffer[r1,1,1]
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
                    tile.rings[r,v].b[rk] = splineBuffer[r1,p,1]
                    tile.rings[r,v].b[ik] = splineBuffer[r1,p+1,1]
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
            tile.rings[r,v].b[1] = splineBuffer[r1,1,2]
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
                    tile.rings[r,v].b[rk] = splineBuffer[r1,p,2]
                    tile.rings[r,v].b[ik] = splineBuffer[r1,p+1,2]
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

function calcPatchMap(patch::RL_Grid, tile::RL_Grid)

    patchMap = falses(size(patch.spectral))
    tileView = falses(size(tile.spectral))

    # Indices of sharedArray that won't be touched by other workers
    # Get the appropriate dimensions
    kDim = tile.params.rDim + tile.params.patchOffsetL
    spectralIndexL = tile.params.spectralIndexL
    patchStride = patch.params.b_rDim
    tileStride = tile.params.b_rDim
    tileShare = tileStride - 4

    # Wavenumber 0
    p1 = spectralIndexL
    p2 = p1 + tileShare
    t1 = 1
    t2 = t1 + tileShare
    patchMap[p1:p2,:] .= true
    tileView[t1:t2,:] .= true

    # Higher wavenumbers
    for k in 1:kDim
        i = k*2

        # Real part
        p1 = ((i-1)*patchStride) + spectralIndexL
        p2 = p1 + tileShare
        t1 = ((i-1)*tileStride) + 1
        t2 = t1 + tileShare
        patchMap[p1:p2,:] .= true
        tileView[t1:t2,:] .= true

        # Imaginary part
        p1 = (i*patchStride) + spectralIndexL
        p2 = p1 + tileShare
        t1 = (i*tileStride) + 1
        t2 = t1 + tileShare
        patchMap[p1:p2,:] .= true
        tileView[t1:t2,:] .= true
    end

    return patchMap, view(tile.spectral, tileView)
end

function calcHaloMap(patch::RL_Grid, tile::RL_Grid)

    patchMap = falses(size(patch.spectral))
    tileView = falses(size(tile.spectral))

    # Indices of sharedArray that won't be touched by other workers
    # Get the appropriate dimensions
    kDim = tile.params.rDim + tile.params.patchOffsetL
    spectralIndexL = tile.params.spectralIndexL
    patchStride = patch.params.b_rDim
    tileStride = tile.params.b_rDim
    # Index is 1 more than shared map
    tileShare = tileStride - 3

    # Wavenumber 0
    p1 = spectralIndexL + tileShare
    p2 = p1 + 2
    t1 = 1 + tileShare
    t2 = t1 + 2
    patchMap[p1:p2,:] .= true
    tileView[t1:t2,:] .= true

    # Higher wavenumbers
    for k in 1:kDim
        i = k*2

        # Real part
        p1 = ((i-1)*patchStride) + spectralIndexL + tileShare
        p2 = p1 + 2
        t1 = ((i-1)*tileStride) + 1 + tileShare
        t2 = t1 + 2
        patchMap[p1:p2,:] .= true
        tileView[t1:t2,:] .= true

        # Imaginary part
        p1 = (i*patchStride) + spectralIndexL + tileShare
        p2 = p1 + 2
        t1 = (i*tileStride) + 1 + tileShare
        t2 = t1 + 2
        patchMap[p1:p2,:] .= true
        tileView[t1:t2,:] .= true
    end

    return patchMap, view(tile.spectral, tileView)
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

function sumSpectralTile(patchSpectral::SharedArray{Float64}, pp::GridParameters, tile::RL_Grid)

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
    patchSpectral[p1:p2,:] .= patchSpectral[p1:p2,:] .+ tile.spectral[t1:t2,:]

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
        patchSpectral[p1:p2,:] .= patchSpectral[p1:p2,:] .+ tile.spectral[t1:t2,:]

        # Imaginary part
        p1 = (p*pp.b_rDim) + spectralIndexL
        p2 = p1 + tile.params.b_rDim - 1
        t1 = (t*tile.params.b_rDim)+1
        t2 = (t+1)*tile.params.b_rDim
        patchSpectral[p1:p2,:] .= patchSpectral[p1:p2,:] .+ tile.spectral[t1:t2,:]
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
        spline[:,1] = SItransform(grid.splines[1,v].params, a, rpoints)
        spline_r[:,1] = SIxtransform(grid.splines[1,v].params, a, rpoints)
        spline_rr[:,1] = SIxxtransform(grid.splines[1,v].params, a, rpoints)
        
        # Higher wavenumbers
        for k = 1:grid.params.rDim
            p = k*2
            p1 = ((p-1)*grid.params.b_rDim)+1
            p2 = p*grid.params.b_rDim
            a = SAtransform(grid.splines[2,v], grid.spectral[p1:p2,v])
            spline[:,p] = SItransform(grid.splines[2,v].params, a, rpoints)
            spline_r[:,p] = SIxtransform(grid.splines[2,v].params, a, rpoints)
            spline_rr[:,p] = SIxxtransform(grid.splines[2,v].params, a, rpoints)
            
            p1 = (p*grid.params.b_rDim)+1
            p2 = (p+1)*grid.params.b_rDim
            a = SAtransform(grid.splines[3,v], grid.spectral[p1:p2,v])
            spline[:,p+1] = SItransform(grid.splines[3,v].params, a, rpoints)
            spline_r[:,p+1] = SIxtransform(grid.splines[3,v].params, a, rpoints)
            spline_rr[:,p+1] = SIxxtransform(grid.splines[3,v].params, a, rpoints)
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

function allocateSplineBuffer(grid::RL_Grid)

    return zeros(Float64, grid.params.rDim, grid.params.rDim*2+1, 2)
end