#Functions for RLZ Grid

struct RLZ_Grid <: AbstractGrid
    params::GridParameters
    splines::Array{Spline1D}
    rings::Array{Fourier1D}
    columns::Array{Chebyshev1D}
    spectral::Array{Float64}
    physical::Array{Float64}
end

function create_RLZ_Grid(gp::GridParameters)

    # RL is 2-D grid with splines and Fourier basis
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

    splines = Array{Spline1D}(undef,gp2.b_zDim,length(values(gp2.vars)))
    rings = Array{Fourier1D}(undef,gp2.rDim, gp2.b_zDim)
    columns = Array{Chebyshev1D}(undef,length(values(gp2.vars)))
    
    kDim = gp2.rDim + gp2.patchOffsetL
    spectralDim = gp2.b_zDim * gp2.b_rDim * (1 + (kDim * 2))
    spectral = zeros(Float64, spectralDim, length(values(gp2.vars)))
    
    physical = zeros(Float64, gp2.zDim * gp2.lDim, length(values(gp2.vars)), 7)
    grid = RLZ_Grid(gp2, splines, rings, columns, spectral, physical)
    for key in keys(gp2.vars)

        # Need different BCs for wavenumber zero winds since they are undefined at r = 0
        for i = 1:gp2.b_zDim
            grid.splines[i,gp2.vars[key]] = Spline1D(SplineParameters(
                    xmin = gp2.xmin,
                    xmax = gp2.xmax,
                    num_cells = gp2.num_cells,
                    BCL = gp2.BCL[key], 
                    BCR = gp2.BCR[key]))
        end
        
        grid.columns[gp2.vars[key]] = Chebyshev1D(ChebyshevParameters(
            zmin = gp2.zmin,
            zmax = gp2.zmax,
            zDim = gp2.zDim,
            bDim = gp2.b_zDim,
            BCB = gp2.BCB[key],
            BCT = gp2.BCT[key]))
    end
    
    # For RLZ, the rings are r and b_zDim to transform Chebyshev coefficients to Fouriers
    for r = 1:gp2.rDim
        ri = r + gp2.patchOffsetL
        lpoints = 4 + 4*ri
        dl = 2 * π / lpoints
        offset = 0.5 * dl * (ri-1)
        for b = 1:gp2.b_zDim
            grid.rings[r,b] = Fourier1D(FourierParameters(
                ymin = offset,
                yDim = lpoints,
                bDim = ri*2 + 1,
                kmax = ri))
        end
    end

    return grid
end

function calcTileSizes(patch::RLZ_Grid, num_tiles::int)

    # Calculate the appropriate tile size for the given patch
    num_gridpoints = patch.params.lDim
    if patch.params.num_cells / num_tiles < 3.0
        throw(DomainError(num_tiles, "Too many tiles for this grid (need at least 3 cells in R direction)"))
    end

    # Target an even number of gridpoints per tile
    q,r = divrem(num_gridpoints, num_tiles)
    tile_targets = [i <= r ? q+1 : q for i = 1:num_tiles]
    tile_min = zeros(Int64,num_tiles)

    # Calculate the dimensions and set the parameters
    DX = (patch.params.xmax - patch.params.xmin) / patch.params.num_cells

    xmins = zeros(Float64,num_tiles)
    xmaxs = zeros(Float64,num_tiles)
    num_cells = zeros(Int64,num_tiles)
    spectralIndicesL = ones(Int64,num_tiles)
    patchOffsetsL = zeros(Int64,num_tiles)
    tile_sizes = zeros(Int64,num_tiles)

    # Check for the special case of only 1 tile
    if num_tiles == 1
        xmins[1] = patch.params.xmin
        xmaxs[1] = patch.params.xmax
        num_cells[1] = patch.params.num_cells
        spectralIndicesL[1] = 1
        tile_sizes[1] = patch.params.lDim
        tile_params = vcat(xmins', xmaxs', num_cells', spectralIndicesL', tile_sizes')
        return tile_params
    end

    # Get the number of points per ring
    lpoints = zeros(Int64,patch.params.rDim)
    for r = 1:patch.params.rDim
        lpoints[r] = 4 + 4*r
    end

    # Try to balance the tiles
    tile_count = zeros(Int64, num_tiles)
    target = 1.0
    while (any(num_cells .< 3)) && target > 0.1
        t = num_tiles
        cell_count = 0
        target -= 0.1
        tile_count[:] .= 0
        num_cells[:] .= 0
        tile_min = Int64(floor(target * tile_targets[1]))
        for r in patch.params.rDim:-1:1
            tile_count[t] += lpoints[r]
            if (r % 3 == 0)
                cell_count += 1
                if cell_count >= 3 && tile_count[t] >= tile_min
                    num_cells[t] = cell_count
                    cell_count = 0
                    t -= 1
                end
            end
            if t == 0
                break
            end
        end
        num_cells[1] = patch.params.num_cells - sum(num_cells[2:num_tiles])
    end

    # First tile starts on the patch boundary
    # Make sure each tile has at least 3 cells and 50% of the target gridpoints
    xmins[1] = patch.params.xmin
    xmaxs[1] = (num_cells[1] * DX) + xmins[1]
    tile_sizes[1] = sum(lpoints[1:num_cells[1]*3])
    # Implicit spectralIndicesL = 1
    # Implicit patchOffsetsL = 0

    # Loop through other tiles
    for i = 2:num_tiles-1
        xmins[i] = xmaxs[i-1]
        xmaxs[i] = (num_cells[i] * DX) + xmins[i]
        spectralIndicesL[i] = num_cells[i-1] + spectralIndicesL[i-1]
        ri = 1+(spectralIndicesL[i] - 1) * 3
        tile_sizes[i] = sum(lpoints[ri:ri+num_cells[i]*3-1])
    end

    # Last tile ends on the patch boundary
    if num_tiles > 1
        xmins[num_tiles] = xmaxs[num_tiles-1]
        xmaxs[num_tiles] = patch.params.xmax
        num_cells[num_tiles] = patch.params.num_cells - sum(num_cells[1:num_tiles-1])
        spectralIndicesL[num_tiles] = num_cells[num_tiles-1] + spectralIndicesL[num_tiles-1]
        ri = 1+(spectralIndicesL[num_tiles] - 1) * 3
        tile_sizes[num_tiles] = sum(lpoints[ri:ri+num_cells[num_tiles]*3-1])
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


function getGridpoints(grid::RLZ_Grid)

    # Return an array of the gridpoint locations
    gridpoints = zeros(Float64, grid.params.zDim * grid.params.lDim,3)
    g = 1
    for r = 1:grid.params.rDim
        r_m = grid.splines[1,1].mishPoints[r]
        ri = r + grid.params.patchOffsetL
        lpoints = 4 + 4*ri
        for l = 1:lpoints
            l_m = grid.rings[r,1].mishPoints[l]
            for z = 1:grid.params.zDim
                z_m = grid.columns[1].mishPoints[z]
                gridpoints[g,1] = r_m
                gridpoints[g,2] = l_m
                gridpoints[g,3] = z_m
                g += 1
            end
        end
    end
    return gridpoints
end

function getCartesianGridpoints(grid::RLZ_Grid)

    gridpoints = zeros(Float64, grid.params.zDim * grid.params.lDim,3)
    g = 1
    radii = grid.splines[1,1].mishPoints
    for r = 1:length(radii)
        angles = grid.rings[r,1].mishPoints
        for l = 1:length(angles)
            for z = 1:grid.params.zDim
                z_m = grid.columns[1].mishPoints[z]
                gridpoints[g,1] = radii[r] * cos(angles[l])
                gridpoints[g,2] = radii[r] * sin(angles[l])
                gridpoints[g,3] = z_m
                g += 1
            end
        end
    end
    return gridpoints
end

function spectralTransform!(grid::RLZ_Grid)
    
    # Transform from the RLZ grid to spectral space
    # For RLZ grid, varying dimensions are R, L, Z, and variable
    spectral = spectralTransform(grid, grid.physical, grid.spectral)
    return spectral
end

function spectralTransform(grid::RLZ_Grid, physical::Array{real}, spectral::Array{real})
    
    # Transform from the RLZ grid to spectral space
    # For RLZ grid, varying dimensions are R, L, Z, and variable

    # Need to include patchOffset to get all available wavenumbers
    kDim = grid.params.rDim + grid.params.patchOffsetL
    
    tempcb = zeros(Float64, grid.params.b_zDim, 4 + 4*kDim)

    for v in values(grid.params.vars)
        i = 1
        for r = 1:grid.params.rDim
            ri = r + grid.params.patchOffsetL
            lpoints = 4 + 4*ri
            for l = 1:lpoints
                for z = 1:grid.params.zDim
                    grid.columns[v].uMish[z] = physical[i,v,1]
                    i += 1
                end
                tempcb[:,l] .= CBtransform!(grid.columns[v])
            end
            for z = 1:grid.params.b_zDim
                grid.rings[r,z].uMish .= view(tempcb,z,1:lpoints)
                FBtransform!(grid.rings[r,z])
            end
        end

        for z = 1:grid.params.b_zDim
            # Clear the wavenumber zero spline
            grid.splines[1,v].uMish .= 0.0
            for r = 1:grid.params.rDim
                # Wavenumber zero
                grid.splines[1,v].uMish[r] = grid.rings[r,z].b[1]
            end
            SBtransform!(grid.splines[1,v])

            # Assign the spectral array
            r1 = ((z-1) * grid.params.b_rDim * (1 + (kDim * 2))) + 1
            r2 = r1 + grid.params.b_rDim - 1
            spectral[r1:r2,v] .= grid.splines[1,v].b

            for k = 1:kDim
                # Clear the splines
                grid.splines[2,v].uMish .= 0.0
                grid.splines[3,v].uMish .= 0.0
                for r = 1:grid.params.rDim
                    if (k <= r + grid.params.patchOffsetL)
                        # Real part
                        rk = k+1
                        # Imaginary part
                        ik = grid.rings[r,z].params.bDim-k+1
                        grid.splines[2,v].uMish[r] = grid.rings[r,z].b[rk]
                        grid.splines[3,v].uMish[r] = grid.rings[r,z].b[ik]
                    end
                end
                SBtransform!(grid.splines[2,v])
                SBtransform!(grid.splines[3,v])

                # Assign the spectral array
                # For simplicity, just stack the real and imaginary parts one after the other
                p = (k-1)*2
                p1 = r2 + 1 + (p*grid.params.b_rDim)
                p2 = p1 + grid.params.b_rDim - 1
                spectral[p1:p2,v] .= grid.splines[2,v].b

                p1 = p2 + 1
                p2 = p1 + grid.params.b_rDim - 1
                spectral[p1:p2,v] .= grid.splines[3,v].b
            end
        end
    end
    
    return spectral
end

function gridTransform!(grid::RLZ_Grid)
    
    # Transform from the spectral to grid space
    physical = gridTransform(grid, grid.physical, grid.spectral)
    return physical 
end

function gridTransform(grid::RLZ_Grid, physical::Array{real}, spectral::Array{real})
    
    # Transform from the spectral to grid space
    # Need to include patchOffset to get all available wavenumbers
    kDim = grid.params.rDim
    splineBuffer = zeros(Float64, grid.params.rDim, 3)
    
    for v in values(grid.params.vars)
        for dr in 0:2
            for z = 1:grid.params.b_zDim
                # Wavenumber zero
                r1 = ((z-1) * grid.params.b_rDim * (1 + (kDim * 2))) + 1
                r2 = r1 + grid.params.b_rDim - 1
                grid.splines[1,v].b .= spectral[r1:r2,v]
                SAtransform!(grid.splines[1,v])
                if (dr == 0)
                    splineBuffer[:,1] .= SItransform!(grid.splines[1,v])
                elseif (dr == 1)
                    splineBuffer[:,1] .= SIxtransform(grid.splines[1,v])
                else
                    splineBuffer[:,1] .= SIxxtransform(grid.splines[1,v])
                end
    
                for r = 1:grid.params.rDim
                    grid.rings[r,z].b[1] = splineBuffer[r,1]
                end

                # Higher wavenumbers
                for k = 1:kDim
                    p = (k-1)*2
                    p1 = r2 + 1 + (p*grid.params.b_rDim)
                    p2 = p1 + grid.params.b_rDim - 1
                    grid.splines[2,v].b .= spectral[p1:p2,v]
                    SAtransform!(grid.splines[2,v])
                    if (dr == 0)
                        splineBuffer[:,2] .= SItransform!(grid.splines[2,v])
                    elseif (dr == 1)
                        splineBuffer[:,2] .= SIxtransform(grid.splines[2,v])
                    else
                        splineBuffer[:,2] .= SIxxtransform(grid.splines[2,v])
                    end

                    p1 = p2 + 1
                    p2 = p1 + grid.params.b_rDim - 1
                    grid.splines[3,v].b .= spectral[p1:p2,v]
                    SAtransform!(grid.splines[3,v])
                    if (dr == 0)
                        splineBuffer[:,3] .= SItransform!(grid.splines[3,v])
                    elseif (dr == 1)
                        splineBuffer[:,3] .= SIxtransform(grid.splines[3,v])
                    else
                        splineBuffer[:,3] .= SIxxtransform(grid.splines[3,v])
                    end

                    for r = 1:grid.params.rDim
                        if (k <= r + grid.params.patchOffsetL)
                            # Real part
                            rk = k+1
                            # Imaginary part
                            ik = grid.rings[r,z].params.bDim-k+1
                            grid.rings[r,z].b[rk] = splineBuffer[r,2]
                            grid.rings[r,z].b[ik] = splineBuffer[r,3]
                        end
                    end
                end
                
                for r = 1:grid.params.rDim
                    FAtransform!(grid.rings[r,z])
                end
            end

            zi = 1
            for r = 1:grid.params.rDim
                ri = r + grid.params.patchOffsetL
                lpoints = 4 + 4*ri
                ringBuffer = zeros(Float64, lpoints, grid.params.b_zDim)
                for dl in 0:2
                    if (dr > 0) && (dl > 0) 
                        # No mixed derivatives
                        continue
                    end
                    for z = 1:grid.params.b_zDim
                        if (dr == 0)
                            if (dl == 0)
                                ringBuffer[:,z] .= FItransform!(grid.rings[r,z])
                            elseif (dl == 1)
                                ringBuffer[:,z] .= FIxtransform(grid.rings[r,z])
                            else
                                ringBuffer[:,z] .= FIxxtransform(grid.rings[r,z])
                            end
                        else
                            ringBuffer[:,z] .= FItransform!(grid.rings[r,z])
                        end
                    end
                    for l = 1:lpoints
                        for z = 1:grid.params.b_zDim
                            grid.columns[v].b[z] = ringBuffer[l,z]
                        end
                        CAtransform!(grid.columns[v])
                        CItransform!(grid.columns[v])

                        # Assign the grid array
                        z1 = zi + (l-1)*grid.params.zDim
                        z2 = z1 + grid.params.zDim - 1
                        if (dr == 0) && (dl == 0)
                            physical[z1:z2,v,1] .= grid.columns[v].uMish
                            physical[z1:z2,v,6] .= CIxtransform(grid.columns[v])
                            physical[z1:z2,v,7] .= CIxxtransform(grid.columns[v])
                        elseif (dr == 0) && (dl == 1)
                            physical[z1:z2,v,4] .= grid.columns[v].uMish
                        elseif (dr == 0) && (dl == 2)
                            physical[z1:z2,v,5] .= grid.columns[v].uMish
                        elseif (dr == 1)
                            physical[z1:z2,v,2] .= grid.columns[v].uMish
                        elseif (dr == 2)
                            physical[z1:z2,v,3] .= grid.columns[v].uMish
                        end
                    end
                end
                # Increment the outer index
                zi += lpoints * grid.params.zDim
            end
        end
    end

    return physical 
end

function gridTransform!(patch::RLZ_Grid, tile::RLZ_Grid)

    splineBuffer = zeros(Float64, patch.params.rDim, 3)
    physical = gridTransform(patch.splines, patch.spectral, patch.params, tile, splineBuffer)
    return physical
end

function gridTransform!(patchSplines::Array{Spline1D}, patchSpectral::Array{Float64}, pp::GridParameters, tile::RLZ_Grid, splineBuffer::Array{Float64})

    # Transform from the spectral to grid space
    # Need to include patchOffset to get all available wavenumbers
    kDim = pp.rDim
    
    for v in values(pp.vars)
        for dr in 0:2
            for z = 1:pp.b_zDim
                # Wavenumber zero
                r1 = ((z-1) * pp.b_rDim * (1 + (kDim * 2))) + 1
                r2 = r1 + pp.b_rDim - 1
                patchSplines[1,v].b .= patchSpectral[r1:r2,v]
                SAtransform!(patchSplines[1,v])
                if (dr == 0)
                    splineBuffer[:,1] .= SItransform!(patchSplines[1,v])
                elseif (dr == 1)
                    splineBuffer[:,1] .= SIxtransform(patchSplines[1,v])
                else
                    splineBuffer[:,1] .= SIxxtransform(patchSplines[1,v])
                end
    
                for r = 1:tile.params.rDim
                    ri = r + tile.params.patchOffsetL
                    tile.rings[r,z].b[1] = splineBuffer[ri,1]
                end

                # Higher wavenumbers
                for k = 1:kDim
                    p = (k-1)*2
                    p1 = r2 + 1 + (p*pp.b_rDim)
                    p2 = p1 + pp.b_rDim - 1
                    patchSplines[2,v].b .= patchSpectral[p1:p2,v]
                    SAtransform!(patchSplines[2,v])
                    if (dr == 0)
                        splineBuffer[:,2] .= SItransform!(patchSplines[2,v])
                    elseif (dr == 1)
                        splineBuffer[:,2] .= SIxtransform(patchSplines[2,v])
                    else
                        splineBuffer[:,2] .= SIxxtransform(patchSplines[2,v])
                    end

                    p1 = p2 + 1
                    p2 = p1 + pp.b_rDim - 1
                    patchSplines[3,v].b .= patchSpectral[p1:p2,v]
                    SAtransform!(patchSplines[3,v])
                    if (dr == 0)
                        splineBuffer[:,3] .= SItransform!(patchSplines[3,v])
                    elseif (dr == 1)
                        splineBuffer[:,3] .= SIxtransform(patchSplines[3,v])
                    else
                        splineBuffer[:,3] .= SIxxtransform(patchSplines[3,v])
                    end

                    for r = 1:tile.params.rDim
                        if (k <= r + tile.params.patchOffsetL)
                            # Real part
                            rk = k+1
                            # Imaginary part
                            ik = tile.rings[r,z].params.bDim-k+1
                            ri = r + tile.params.patchOffsetL
                            tile.rings[r,z].b[rk] = splineBuffer[ri,2]
                            tile.rings[r,z].b[ik] = splineBuffer[ri,3]
                        end
                    end
                end
                
                for r = 1:tile.params.rDim
                    FAtransform!(tile.rings[r,z])
                end
            end

            zi = 1
            for r = 1:tile.params.rDim
                ri = r + tile.params.patchOffsetL
                lpoints = 4 + 4*ri
                ringBuffer = zeros(Float64, lpoints, pp.b_zDim)
                for dl in 0:2
                    if (dr > 0) && (dl > 0) 
                        # No mixed derivatives
                        continue
                    end
                    for z = 1:pp.b_zDim
                        if (dr == 0)
                            if (dl == 0)
                                ringBuffer[:,z] .= FItransform!(tile.rings[r,z])
                            elseif (dl == 1)
                                ringBuffer[:,z] .= FIxtransform(tile.rings[r,z])
                            else
                                ringBuffer[:,z] .= FIxxtransform(tile.rings[r,z])
                            end
                        else
                            ringBuffer[:,z] .= FItransform!(tile.rings[r,z])
                        end
                    end
                    for l = 1:lpoints
                        for z = 1:pp.b_zDim
                            tile.columns[v].b[z] = ringBuffer[l,z]
                        end
                        CAtransform!(tile.columns[v])
                        CItransform!(tile.columns[v])

                        # Assign the grid array
                        z1 = zi + (l-1)*pp.zDim
                        z2 = z1 + pp.zDim - 1
                        if (dr == 0) && (dl == 0)
                            tile.physical[z1:z2,v,1] .= tile.columns[v].uMish
                            tile.physical[z1:z2,v,6] .= CIxtransform(tile.columns[v])
                            tile.physical[z1:z2,v,7] .= CIxxtransform(tile.columns[v])
                        elseif (dr == 0) && (dl == 1)
                            tile.physical[z1:z2,v,4] .= tile.columns[v].uMish
                        elseif (dr == 0) && (dl == 2)
                            tile.physical[z1:z2,v,5] .= tile.columns[v].uMish
                        elseif (dr == 1)
                            tile.physical[z1:z2,v,2] .= tile.columns[v].uMish
                        elseif (dr == 2)
                            tile.physical[z1:z2,v,3] .= tile.columns[v].uMish
                        end
                    end
                end
                # Increment the outer index
                zi += lpoints * pp.zDim
            end
        end
    end

    return tile.physical 
end

function splineTransform!(patchSplines::Array{Spline1D}, patchSpectral::Array{Float64}, pp::GridParameters, sharedSpectral::SharedArray{Float64},tile::RLZ_Grid)

    # Do a partial transform from B to A for splines only
    for v in values(pp.vars)
        k1 = 1
        for z in 1:pp.b_zDim
            for k in 1:(pp.rDim*2 + 1)
                k2 = k1 + pp.b_rDim - 1
                patchSpectral[k1:k2,v] .= SAtransform(patchSplines[z,v], view(sharedSpectral,k1:k2,v))
                k1 = k2 + 1
            end
        end
    end
end

function tileTransform!(patchSplines::Array{Spline1D}, patchSpectral::Array{Float64}, pp::GridParameters, tile::RLZ_Grid, splineBuffer::Array{Float64})

    # Transform from the spectral to grid space
    # Need to include patchOffset to get all available wavenumbers
    #splineBuffer = zeros(Float64, tile.params.rDim, pp.b_zDim)
    kDim = pp.rDim

    for v in values(pp.vars)
        for dr in 0:2
            for z = 1:pp.b_zDim
                # Wavenumber zero
                r1 = ((z-1) * pp.b_rDim * (1 + (kDim * 2))) + 1
                r2 = r1 + pp.b_rDim - 1
                patchSplines[z,v].a .= view(patchSpectral,r1:r2,v)
                if (dr == 0)
                    SItransform(patchSplines[z,v],tile.splines[1].mishPoints,view(splineBuffer,:,z))
                elseif (dr == 1)
                    SIxtransform(patchSplines[z,v],tile.splines[1].mishPoints,view(splineBuffer,:,z))
                else
                    SIxxtransform(patchSplines[z,v],tile.splines[1].mishPoints,view(splineBuffer,:,z))
                end

                for r = 1:tile.params.rDim
                    tile.rings[r,z].b[1] = splineBuffer[r,z]
                end

                # Higher wavenumbers
                for k = 1:kDim
                    p = (k-1)*2
                    p1 = r2 + 1 + (p*pp.b_rDim)
                    p2 = p1 + pp.b_rDim - 1
                    patchSplines[z,v].a .= view(patchSpectral,p1:p2,v)
                    if (dr == 0)
                        SItransform(patchSplines[z,v],tile.splines[1].mishPoints,view(splineBuffer,:,z))
                    elseif (dr == 1)
                        SIxtransform(patchSplines[z,v],tile.splines[1].mishPoints,view(splineBuffer,:,z))
                    else
                        SIxxtransform(patchSplines[z,v],tile.splines[1].mishPoints,view(splineBuffer,:,z))
                    end
                    for r = 1:tile.params.rDim
                        if (k <= r + tile.params.patchOffsetL)
                            # Real part
                            rk = k+1
                            tile.rings[r,z].b[rk] = splineBuffer[r,z]
                        end
                    end

                    p1 = p2 + 1
                    p2 = p1 + pp.b_rDim - 1
                    patchSplines[z,v].a .= view(patchSpectral,p1:p2,v)
                    if (dr == 0)
                        SItransform(patchSplines[z,v],tile.splines[1].mishPoints,view(splineBuffer,:,z))
                    elseif (dr == 1)
                        SIxtransform(patchSplines[z,v],tile.splines[1].mishPoints,view(splineBuffer,:,z))
                    else
                        SIxxtransform(patchSplines[z,v],tile.splines[1].mishPoints,view(splineBuffer,:,z))
                    end

                    for r = 1:tile.params.rDim
                        if (k <= r + tile.params.patchOffsetL)
                            # Imaginary part
                            ik = tile.rings[r,z].params.bDim-k+1
                            tile.rings[r,z].b[ik] = splineBuffer[r,z]
                        end
                    end
                end

                for r = 1:tile.params.rDim
                    FAtransform!(tile.rings[r,z])
                end
            end

            zi = 1
            for r = 1:tile.params.rDim
                ri = r + tile.params.patchOffsetL
                lpoints = 4 + 4*ri
                ringBuffer = zeros(Float64, lpoints, pp.b_zDim)
                for dl in 0:2
                    if (dr > 0) && (dl > 0)
                        # No mixed derivatives
                        continue
                    end
                    for z = 1:pp.b_zDim
                        if (dr == 0)
                            if (dl == 0)
                                ringBuffer[:,z] .= FItransform!(tile.rings[r,z])
                            elseif (dl == 1)
                                ringBuffer[:,z] .= FIxtransform(tile.rings[r,z])
                            else
                                ringBuffer[:,z] .= FIxxtransform(tile.rings[r,z])
                            end
                        else
                            ringBuffer[:,z] .= FItransform!(tile.rings[r,z])
                        end
                    end
                    for l = 1:lpoints
                        for z = 1:pp.b_zDim
                            tile.columns[v].b[z] = ringBuffer[l,z]
                        end
                        CAtransform!(tile.columns[v])
                        CItransform!(tile.columns[v])

                        # Assign the grid array
                        z1 = zi + (l-1)*pp.zDim
                        z2 = z1 + pp.zDim - 1
                        if (dr == 0) && (dl == 0)
                            tile.physical[z1:z2,v,1] .= tile.columns[v].uMish
                            tile.physical[z1:z2,v,6] .= CIxtransform(tile.columns[v])
                            tile.physical[z1:z2,v,7] .= CIxxtransform(tile.columns[v])
                        elseif (dr == 0) && (dl == 1)
                            tile.physical[z1:z2,v,4] .= tile.columns[v].uMish
                        elseif (dr == 0) && (dl == 2)
                            tile.physical[z1:z2,v,5] .= tile.columns[v].uMish
                        elseif (dr == 1)
                            tile.physical[z1:z2,v,2] .= tile.columns[v].uMish
                        elseif (dr == 2)
                            tile.physical[z1:z2,v,3] .= tile.columns[v].uMish
                        end
                    end
                end
                # Increment the outer index
                zi += lpoints * pp.zDim
            end
        end
    end

    return tile.physical
end

function spectralxTransform(grid::RLZ_Grid, physical::Array{real}, spectral::Array{real})
    
    # Not yet implemented

end

function calcPatchMap(patch::RLZ_Grid, tile::RLZ_Grid)

    patchMap = falses(size(patch.spectral))
    tileView = falses(size(tile.spectral))

    # Indices of sharedArray that won't be touched by other workers
    # Get the appropriate dimensions
    tilekDim = tile.params.rDim + tile.params.patchOffsetL
    patchkDim = patch.params.rDim
    spectralIndexL = tile.params.spectralIndexL
    patchRstride = patch.params.b_rDim
    tileRstride = tile.params.b_rDim
    patchZstride = patchRstride * (1 + (patchkDim * 2))
    tileZstride = tileRstride * (1 + (tilekDim * 2))
    tileShare = tileRstride - 4

    for z = 1:tile.params.b_zDim
        # Wavenumber 0
        p0 = spectralIndexL + ((z-1) * patchZstride)
        p1 = p0
        p2 = p1 + tileShare
        t0 = 1 + ((z-1) * tileZstride)
        t1 = t0
        t2 = t1 + tileShare
        patchMap[p1:p2,:] .= true
        tileView[t1:t2,:] .= true

        # Higher wavenumbers
        for k in 1:tilekDim
            i = k*2

            # Real part
            p1 = p0 + ((i-1) * patchRstride)
            p2 = p1 + tileShare
            
            t1 = t0 + ((i-1) * tileRstride)
            t2 = t1 + tileShare
            patchMap[p1:p2,:] .= true
            tileView[t1:t2,:] .= true

            # Imaginary part
            p1 = p0 + (i * patchRstride)
            p2 = p1 + tileShare
            
            t1 = t0 + (i * tileRstride)
            t2 = t1 + tileShare
            patchMap[p1:p2,:] .= true
            tileView[t1:t2,:] .= true
        end
    end
    
    return patchMap, view(tile.spectral, tileView)
end

function calcHaloMap(patch::RLZ_Grid, tile::RLZ_Grid)

    patchMap = falses(size(patch.spectral))
    tileView = falses(size(tile.spectral))

    # Indices of sharedArray that won't be touched by other workers
    # Get the appropriate dimensions
    tilekDim = tile.params.rDim + tile.params.patchOffsetL
    patchkDim = patch.params.rDim
    spectralIndexL = tile.params.spectralIndexL
    patchRstride = patch.params.b_rDim
    tileRstride = tile.params.b_rDim
    patchZstride = patchRstride * (1 + (patchkDim * 2))
    tileZstride = tileRstride * (1 + (tilekDim * 2))
    
    # Index is 1 more than shared map
    tileShare = tileRstride - 3

    for z = 1:tile.params.b_zDim
        # Wavenumber 0
        p0 = spectralIndexL + ((z-1) * patchZstride)
        p1 = p0 + tileShare
        p2 = p1 + 2
        t0 = 1 + ((z-1) * tileZstride)
        t1 = t0 + tileShare
        t2 = t1 + 2
        patchMap[p1:p2,:] .= true
        tileView[t1:t2,:] .= true

        # Higher wavenumbers
        for k in 1:tilekDim
            i = k*2

            # Real part
            p1 = p0 + ((i-1) * patchRstride) + tileShare
            p2 = p1 + 2
            t1 = t0 + ((i-1) * tileRstride) + tileShare
            t2 = t1 + 2
            patchMap[p1:p2,:] .= true
            tileView[t1:t2,:] .= true

            # Imaginary part
            p1 = p0 + (i * patchRstride) + tileShare
            p2 = p1 + 2
            t1 = t0 + (i * tileRstride) + tileShare
            t2 = t1 + 2
            patchMap[p1:p2,:] .= true
            tileView[t1:t2,:] .= true
        end
    end
    
    return patchMap, view(tile.spectral, tileView)
end


function regularGridTransform(grid::RLZ_Grid)
    
    # Output on regular grid
    kDim = grid.params.rDim + grid.params.patchOffsetL

    # Generic rings of maximum size
    rings = Array{Fourier1D}(undef,grid.params.num_cells,grid.params.b_zDim)
    lpoints = grid.params.rDim*2 + 1
    for i in 1:grid.params.num_cells
        for j in 1:grid.params.b_zDim
            rings[i,j] = Fourier1D(FourierParameters(
                ymin = 0.0,
                yDim = lpoints,
                bDim = lpoints,
                kmax = grid.params.rDim))
        end
    end
    
    # Output on the nodes
    rpoints = zeros(Float64, grid.params.num_cells)
    for r = 1:grid.params.num_cells
        rpoints[r] = grid.params.xmin + (r-1)*grid.splines[1,1].params.DX
    end
    
    # Z grid stays the same for now
    
    # Allocate memory for the regular grid and buffers
    physical = zeros(Float64, grid.params.zDim * grid.params.num_cells * lpoints,
        length(values(grid.params.vars)),7)
    splineBuffer = zeros(Float64, grid.params.num_cells, 3)
    ringBuffer = zeros(Float64, lpoints, grid.params.b_zDim)

    for v in values(grid.params.vars)
        for dr in 0:2
            for z = 1:grid.params.b_zDim
                # Wavenumber zero
                r1 = ((z-1) * grid.params.b_rDim * (1 + (kDim * 2))) + 1
                r2 = r1 + grid.params.b_rDim - 1
                a = SAtransform(grid.splines[1,v], grid.spectral[r1:r2,v])
                if (dr == 0)
                    splineBuffer[:,1] .= SItransform(grid.splines[1,v].params, a, rpoints)
                elseif (dr == 1)
                    splineBuffer[:,1] .= SIxtransform(grid.splines[1,v].params, a, rpoints)
                else
                    splineBuffer[:,1] .= SIxxtransform(grid.splines[1,v].params, a, rpoints)
                end

                # Reset the ring
                for r in eachindex(rpoints)
                    rings[r,z].b .= 0.0
                    rings[r,z].b[1] = splineBuffer[r,1]
                end

                # Higher wavenumbers
                for k = 1:kDim
                    p = (k-1)*2
                    p1 = r2 + 1 + (p*grid.params.b_rDim)
                    p2 = p1 + grid.params.b_rDim - 1
                    a = SAtransform(grid.splines[2,v], grid.spectral[p1:p2,v])
                    if (dr == 0)
                        splineBuffer[:,2] .= SItransform(grid.splines[2,v].params, a, rpoints)
                    elseif (dr == 1)
                        splineBuffer[:,2] .= SIxtransform(grid.splines[2,v].params, a, rpoints)
                    else
                        splineBuffer[:,2] .= SIxxtransform(grid.splines[2,v].params, a, rpoints)
                    end

                    p1 = p2 + 1
                    p2 = p1 + grid.params.b_rDim - 1
                    a = SAtransform(grid.splines[3,v], grid.spectral[p1:p2,v])
                    if (dr == 0)
                        splineBuffer[:,3] .= SItransform(grid.splines[3,v].params, a, rpoints)
                    elseif (dr == 1)
                        splineBuffer[:,3] .= SIxtransform(grid.splines[3,v].params, a, rpoints)
                    else
                        splineBuffer[:,3] .= SIxxtransform(grid.splines[3,v].params, a, rpoints)
                    end

                    for r in eachindex(rpoints)
                        # Real part
                        rk = k+1
                        # Imaginary part
                        ik = rings[r,z].params.bDim-k+1
                        rings[r,z].b[rk] = splineBuffer[r,2]
                        rings[r,z].b[ik] = splineBuffer[r,3]
                    end
                end
                
                for r in eachindex(rpoints)
                    FAtransform!(rings[r,z])
                end
            end

            zi = 1
            for r in eachindex(rpoints)
                ri = r + grid.params.patchOffsetL
                for dl in 0:2
                    if (dr > 0) && (dl > 0) 
                        # No mixed derivatives
                        continue
                    end
                    for z = 1:grid.params.b_zDim
                        if (dr == 0)
                            if (dl == 0)
                                ringBuffer[:,z] .= FItransform!(rings[r,z])
                            elseif (dl == 1)
                                ringBuffer[:,z] .= FIxtransform(rings[r,z])
                            else
                                ringBuffer[:,z] .= FIxxtransform(rings[r,z])
                            end
                        else
                            ringBuffer[:,z] .= FItransform!(rings[r,z])
                        end
                    end
                    for l = 1:lpoints
                        for z = 1:grid.params.b_zDim
                            grid.columns[v].b[z] = ringBuffer[l,z]
                        end
                        CAtransform!(grid.columns[v])
                        CItransform!(grid.columns[v])

                        # Assign the grid array
                        z1 = zi + (l-1)*grid.params.zDim
                        z2 = z1 + grid.params.zDim - 1
                        if (dr == 0) && (dl == 0)
                            physical[z1:z2,v,1] .= grid.columns[v].uMish
                            physical[z1:z2,v,6] .= CIxtransform(grid.columns[v])
                            physical[z1:z2,v,7] .= CIxxtransform(grid.columns[v])
                        elseif (dr == 0) && (dl == 1)
                            physical[z1:z2,v,4] .= grid.columns[v].uMish
                        elseif (dr == 0) && (dl == 2)
                            physical[z1:z2,v,5] .= grid.columns[v].uMish
                        elseif (dr == 1)
                            physical[z1:z2,v,2] .= grid.columns[v].uMish
                        elseif (dr == 2)
                            physical[z1:z2,v,3] .= grid.columns[v].uMish
                        end
                    end
                end
                # Increment the outer index
                zi += lpoints * grid.params.zDim
            end
        end
    end

    return physical 
end

function getRegularGridpoints(grid::RLZ_Grid)

    # Return an array of regular gridpoint locations
    i = 1
    gridpoints = zeros(Float64, grid.params.num_cells * (grid.params.rDim*2+1) * grid.params.zDim, 5)
    for r = 1:grid.params.num_cells
        r_m = grid.params.xmin + (r-1)*grid.splines[1,1].params.DX
        for l = 1:(grid.params.rDim*2+1)
            l_m = 2 * π * (l-1) / (grid.params.rDim*2+1)
            for z = 1:grid.params.zDim
                z_m = grid.columns[1].mishPoints[z]
                gridpoints[i,1] = r_m
                gridpoints[i,2] = l_m
                gridpoints[i,3] = z_m
                gridpoints[i,4] = r_m * cos(l_m)
                gridpoints[i,5] = r_m * sin(l_m)
                i += 1
            end
        end
    end
    return gridpoints
end

function allocateSplineBuffer(patch::RLZ_Grid, tile::RLZ_Grid)

    return zeros(Float64, tile.params.rDim, 3)
end
