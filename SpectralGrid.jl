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

# Fix the spline mish to 3 points
const mubar = 3

export GridParameters, createGrid, spectralTransform!, gridTransform!

@with_kw struct GridParameters
    xmin::real = 0.0
    xmax::real = 0.0
    num_nodes::int = 0
    rDim::int = 0
    b_rDim::int = 0
    l_q::real = 2.0
    BCL::Dict = CubicBSpline.R0
    BCR::Dict = CubicBSpline.R0
    lDim::int = 0
    zmin::real = 0.0
    zmax::real = 0.0
    zDim::int = 0
    b_zDim::int = 0
    BCB::Dict = Chebyshev.R0
    BCT::Dict = Chebyshev.R0
    vars::Dict = Dict("u" => 1)
end

struct R_Grid
    params::GridParameters
    splines::Array{Spline1D}
    spectral::Array{Float64}
    physical::Array{Float64}
end

struct Z_Grid
    params::GridParameters
    columns::Array{Chebyshev1D}
    spectral::Array{Float64}
    physical::Array{Float64}
end

struct RZ_Grid
    params::GridParameters
    splines::Array{Spline1D}
    columns::Array{Chebyshev1D}
    spectral::Array{Float64}
    physical::Array{Float64}
end

struct RL_Grid
    params::GridParameters
    splines::Array{Spline1D}
    rings::Array{Fourier1D}
    spectral::Array{Float64}
    physical::Array{Float64}
end

struct RLZ_Grid
    params::GridParameters
    splines::Array{Spline1D}
    columns::Array{Chebyshev1D}
    rings::Array{Fourier1D}
    spectral::Array{Float64}
    physical::Array{Float64}
end

function createGrid(gp::GridParameters)
    
    if gp.num_nodes > 0
        # R, RZ, RL, or RLZ grid
        
        if gp.lDim == 0 && gp.zDim == 0
            # R grid
            
            splines = Array{Spline1D}(undef,1,length(values(gp.vars)))
            spectral = zeros(Float64, gp.b_rDim, length(values(gp.vars)))
            physical = zeros(Float64, gp.rDim, length(values(gp.vars)), 3)
            grid = R_Grid(gp, splines, spectral, physical)
            for key in keys(gp.vars)
                grid.splines[1,gp.vars[key]] = Spline1D(SplineParameters(
                    xmin = gp.xmin,
                    xmax = gp.xmax,
                    num_nodes = gp.num_nodes,
                    BCL = gp.BCL[key],
                    BCR = gp.BCR[key]))
            end
            return grid
            
        elseif gp.lDim == 0 && gp.zDim > 0
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
                        num_nodes = gp.num_nodes,
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
            
        elseif gp.lDim > 0 && gp.zDim == 0
            # RL grid
            grid = RL_Grid()
            for key in keys(grid.vars)
                for l = 1:gp.rDim
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
            
        elseif gp.lDim > 0 && gp.zDim > 0
            # RLZ grid
            throw(DomainError(0, "RLZ not implemented yet"))
        end
    else
        # Z grid
        throw(DomainError(0, "Z column model not implemented yet"))
    end
    
end

function spectralTransform!(grid::R_Grid)
    
    # Transform from the grid to spectral space
    # For R grid, the only varying dimension is the variable name
    for i in eachindex(grid.splines)
        SBtransform!(grid.splines[i])
        
        # Assign the spectral array
        grid.spectral[:,i] .= grid.splines[i].b
    end
    
    return grid.spectral
end

function gridTransform!(grid::R_Grid)
    
    # Transform from the spectral to grid space
    # For R grid, the only varying dimension is the variable name
    for i in eachindex(grid.splines)
        SAtransform!(grid.splines[i])
        SItransform!(grid.splines[i])
        
        # Assign the grid array
        grid.physical[:,i,1] .= grid.splines[i].uMish
        grid.physical[:,i,2] .= SIxtransform(grid.splines[i])
        grid.physical[:,i,3] .= SIxxtransform(grid.splines[i])
    end
    
    return grid.physical 
end

function spectralTransform!(grid::RZ_Grid)
    
    # Transform from the RZ grid to spectral space
    # For RZ grid, varying dimensions are R, Z, and variable
    for v in values(grid.params.vars)
        for z = 1:grid.params.zDim
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
            grid.spectral[z1:z2,v] .= grid.columns[v].b
        end
    end

    return grid.spectral
end

function gridTransform!(grid::RZ_Grid)
    
    # Transform from the spectral to grid space
    # For RZ grid, varying dimensions are R, Z, and variable
    for v in values(grid.params.vars)
        for r = 1:grid.params.b_rDim
            z1 = ((r-1)*grid.params.b_zDim)+1
            z2 = r*grid.params.b_zDim
            grid.columns[v].b .= grid.spectral[z1:z2,v]
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
            grid.physical[r1:r2,v,1] .= grid.splines[z,v].uMish
            grid.physical[r1:r2,v,2] .= SIxtransform(grid.splines[z,v])
            grid.physical[r1:r2,v,3] .= SIxxtransform(grid.splines[z,v])
        end
        
        # Get the vertical derivatives
        var = reshape(grid.physical[:,v,1],grid.params.rDim,grid.params.zDim)
        for r = 1:grid.params.rDim
            grid.columns[v].uMish .= var[r,:]
            CBtransform!(grid.columns[v])
            CAtransform!(grid.columns[v])
            varz = CIxtransform(grid.columns[v])
            varzz = CIxxtransform(grid.columns[v])

            # Assign the grid array
            for z = 1:grid.params.zDim
                ri = (z-1)*grid.params.rDim + r
                grid.physical[ri,v,4] = varz[z]
                grid.physical[ri,v,5] = varzz[z]
            end
        end
        
    end
    
    return grid.physical 
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

end