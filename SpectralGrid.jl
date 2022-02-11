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

export GridParameters, createGrid, getGridpoints
export spectralTransform!, gridTransform!, spectralTransform 
export spectralxTransform, gridTransform_noBCs, integralTransform
export R_Grid, RZ_Grid, RL_Grid

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
    diagnostic_flag::Dict = Dict("none" => 0)
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

function spectralTransform!(grid::R_Grid)
    
    # Transform from the grid to spectral space
    # For R grid, the only varying dimension is the variable name
    for i in eachindex(grid.splines)
        grid.splines[i].uMish .= grid.physical[:,i,1]
        SBtransform!(grid.splines[i])
        
        # Assign the spectral array
        grid.spectral[:,i] .= grid.splines[i].b
    end
    
    return grid.spectral
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
            num_nodes = grid.params.num_nodes, 
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
    for i in eachindex(grid.splines)
        grid.splines[i].b .= grid.spectral[:,i]
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
        i = 1
        for z = 1:grid.params.zDim
            for r = 1:grid.params.rDim
                grid.splines[z,v].uMish[r] = grid.physical[i,v,1]
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

function spectralTransform(grid::RZ_Grid, physical::Array{real}, spectral::Array{real})
    
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
            num_nodes = grid.params.num_nodes, 
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

function integralTransform(grid::RZ_Grid, physical::Array{real}, spectral::Array{real})
    
    # Transform from the spectral to grid space
    # For RZ grid, varying dimensions are R, Z, and variable
    # Need to use a R0 BC for this since there is no guarantee 
    # that tendencies should match the underlying variable 
    splines = Array{Spline1D}(undef,grid.params.zDim)
    for z = 1:grid.params.zDim
        splines[z] = Spline1D(SplineParameters(
            xmin = grid.params.xmin, 
            xmax = grid.params.xmax,
            num_nodes = grid.params.num_nodes, 
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
end

# Module end
end
