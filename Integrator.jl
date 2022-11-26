# 3rd order Adams-Bashforth implementation multiple variables and dimensions method
__precompile__()
module Integrator

using SpectralGrid
using CubicBSpline
using Chebyshev
using Fourier
using NumericalModels
using Parameters
using CSV
using DataFrames

#Define some convenient aliases
const real = Float64
const int = Int64
const uint = UInt64

export initialize_model, run_model, finalize_model 
export integrate_LinearAdvection1D, integrate_WilliamsSlabTCBL
export integrate_Kepert2017_TCBL
export integrate_RL_ShallowWater, integrate_Oneway_ShallowWater_Slab
export integrate_Twoway_ShallowWater_Slab

function integrate_LinearAdvection1D()
    
    model = ModelParameters(
        ts = 0.05,
        integration_time = 100.0,
        output_interval = 100.0,
        equation_set = "LinearAdvection1D",
        initial_conditions = "1d_linear_advection_test_ics.csv",
        grid_params = GridParameters(
            geometry = "R",
            xmin = -50.0,
            xmax = 50.0,
            num_cells = 100,
            BCL = Dict(
                "u" => CubicBSpline.PERIODIC),
            BCR = Dict(
                "u" => CubicBSpline.PERIODIC),
            vars = Dict(
                "u" => 1)),
        physical_params = Dict(
            :c_0 => 1.0,
            :K => 0.0))
   
    grid = initialize_model(model)
    run_model(grid, model)
    finalize_model(grid, model)
end

function integrate_WilliamsSlabTCBL(ics_csv::String)
    
    nodes = 400
    model = ModelParameters(
        ts = 2.0,
        integration_time = 10800.0,
        output_interval = 3600.0,
        equation_set = "Williams2013_slabTCBL",
        output_dir = "./slabout/",
        initial_conditions = ics_csv,
        grid_params = GridParameters(
            xmin = 0.0,
            xmax = 4.0e5,
            num_cells = nodes,
            rDim = nodes*3,
            b_rDim = nodes+3,
            BCL = Dict(
                "vgr" => CubicBSpline.R0, 
                "u" => CubicBSpline.R1T0, 
                "v" => CubicBSpline.R1T0, 
                "w" => CubicBSpline.R1T0),
            BCR = Dict(
                "vgr" => CubicBSpline.R0, 
                "u" => CubicBSpline.R1T1, 
                "v" => CubicBSpline.R1T1, 
                "w" => CubicBSpline.R1T1),
            vars = Dict(
                "vgr" => 1, 
                "u" => 2, 
                "v" => 3, 
                "w" => 4)))

    grid = initialize_model(model)
    run_model(grid, model)
    finalize_model(grid, model)
end

function integrate_Kepert2017_TCBL(ics_csv::String)

    nodes = 400
    model = ModelParameters(
        ts = 2.0,
        integration_time = 10800.0,
        output_interval = 3600.0,
        equation_set = "Kepert2017_TCBL",
        initial_conditions = ics_csv,
        output_dir = "./tcblout/",
        grid_params = GridParameters(xmin = 0.0,
            xmax = 4.0e5,
            num_cells = nodes,
            rDim = nodes*3,
            b_rDim = nodes+3,
            BCL = Dict(
                "vgr" => CubicBSpline.R0, 
                "u" => CubicBSpline.R1T0, 
                "v" => CubicBSpline.R1T0, 
                "w" => CubicBSpline.R1T0),
            BCR = Dict(
                "vgr" => CubicBSpline.R0, 
                "u" => CubicBSpline.R1T1, 
                "v" => CubicBSpline.R1T1, 
                "w" => CubicBSpline.R1T1),
            zmin = 0.0,
            zmax = 2350.0,
            zDim = 25,
            b_zDim = 17,
            BCB = Dict(
                "vgr" => Chebyshev.R0, 
                "u" => Chebyshev.R0, 
                "v" => Chebyshev.R0, 
                "w" => Chebyshev.R0),
            BCT = Dict(
                "vgr" => Chebyshev.R0, 
                "u" => Chebyshev.R1T1, 
                "v" => Chebyshev.R1T1, 
                "w" => Chebyshev.R1T1),
            vars = Dict(
                "vgr" => 1, 
                "u" => 2, 
                "v" => 3, 
                "w" => 4)))
    
    grid = initialize_model(model)
    run_model(grid, model)
    finalize_model(grid, model)
end

function integrate_RL_ShallowWater(ics_csv::String)
    
    nodes = 100
    lpoints = 0
    blpoints = 0
    for r = 1:(nodes*3)
        lpoints += 4 + 4*r
        blpoints += 1 + 2*r
    end
    model = ModelParameters(
        ts = 3.0,
        integration_time = 900.0,
        output_interval = 60.0,
        equation_set = "ShallowWaterRL",
        initial_conditions = ics_csv::String,
        output_dir = "./SW_output/",
        grid_params = GridParameters(xmin = 0.0,
            xmax = 3.0e5,
            num_cells = nodes,
            rDim = nodes*3,
            b_rDim = nodes+3,
            BCL = Dict(
                "h" => CubicBSpline.R0, 
                "u" => CubicBSpline.R1T0, 
                "v" => CubicBSpline.R1T0),
            BCR = Dict(
                "h" => CubicBSpline.R0, 
                "u" => CubicBSpline.R1T1, 
                "v" => CubicBSpline.R0), 
            lDim = lpoints,
            b_lDim = blpoints,
            vars = Dict(
                "h" => 1, 
                "u" => 2, 
                "v" => 3)))
    grid = initialize_model(model)
    run_model(grid, model)
    finalize_model(grid, model)
end

function integrate_Oneway_ShallowWater_Slab(ics_csv::String)
    
    nodes = 100
    lpoints = 0
    blpoints = 0
    for r = 1:(nodes*3)
        lpoints += 4 + 4*r
        blpoints += 1 + 2*r
    end
    model = ModelParameters(
        ts = 3.0,
        integration_time = 10800.0,
        output_interval = 900.0,
        equation_set = "Oneway_ShallowWater_Slab",
        initial_conditions = ics_csv::String,
        output_dir = "./SWslab_output/",
        grid_params = GridParameters(xmin = 0.0,
            xmax = 3.0e5,
            num_cells = nodes,
            rDim = nodes*3,
            b_rDim = nodes+3,
            BCL = Dict(
                "h" => CubicBSpline.R1T1,
                "u" => CubicBSpline.R1T0,
                "v" => CubicBSpline.R1T0,
                "ub" => CubicBSpline.R1T0,
                "vb" => CubicBSpline.R1T0,
                "wb" => CubicBSpline.R1T1),
            BCR = Dict(
                "h" => CubicBSpline.R0,
                "u" => CubicBSpline.R1T1,
                "v" => CubicBSpline.R0,
                "ub" => CubicBSpline.R1T1,
                "vb" => CubicBSpline.R0,
                "wb" => CubicBSpline.R0),
            lDim = lpoints,
            b_lDim = blpoints,
            vars = Dict(
                "h" => 1,
                "u" => 2,
                "v" => 3,
                "ub" => 4,
                "vb" => 5,
                "wb" => 6)))
    grid = initialize_model(model);
    run_model(grid, model)
    finalize_model(grid,model)
end
    
function integrate_Twoway_ShallowWater_Slab(ics_csv::String)
    
    nodes = 100
    lpoints = 0
    blpoints = 0
    for r = 1:(nodes*3)
        lpoints += 4 + 4*r
        blpoints += 1 + 2*r
    end
    model = ModelParameters(
        ts = 3.0,
        integration_time = 10800.0,
        output_interval = 900.0,
        equation_set = "Twoway_ShallowWater_Slab",
        initial_conditions = ics_csv::String,
        output_dir = "./Twoway_SWslab_output/",
        grid_params = GridParameters(xmin = 0.0,
            xmax = 3.0e5,
            num_cells = nodes,
            rDim = nodes*3,
            b_rDim = nodes+3,
            BCL = Dict(
                "h" => CubicBSpline.R1T1,
                "u" => CubicBSpline.R1T0,
                "v" => CubicBSpline.R1T0,
                "ub" => CubicBSpline.R1T0,
                "vb" => CubicBSpline.R1T0,
                "wb" => CubicBSpline.R1T1),
            BCR = Dict(
                "h" => CubicBSpline.R0,
                "u" => CubicBSpline.R1T1,
                "v" => CubicBSpline.R0,
                "ub" => CubicBSpline.R1T1,
                "vb" => CubicBSpline.R0,
                "wb" => CubicBSpline.R0),
            lDim = lpoints,
            b_lDim = blpoints,
            vars = Dict(
                "h" => 1,
                "u" => 2,
                "v" => 3,
                "ub" => 4,
                "vb" => 5,
                "wb" => 6)))
    grid = initialize_model(model);
    run_model(grid, model)
    finalize_model(grid,model)
end  

function initialize_model(model::ModelParameters)
    
    gp = model.grid_params
    grid = createGrid(gp)
    println("$model")
    println("$(model.grid_params)")
    
    read_initialconditions(model.initial_conditions, grid)    
    spectralTransform!(grid)
    gridTransform!(grid)
    write_output(grid, model, 0.0) 
    return grid
end

function initialize_model(model::ModelParameters, num_tiles::int)

    println("Initializing with $(num_tiles) tiles")
    patch = createGrid(model.grid_params)
    println("$model")
    println("$(model.grid_params)")

    # Initialize the patch
    read_initialconditions(model.initial_conditions, patch)
    spectralTransform!(patch)
    gridTransform!(patch)
    write_output(patch, model, 0.0)

    # Distribute the tiles
    tile_params = calcTileSizes(patch, num_tiles)
    tiles = Array{typeof(patch)}(undef,num_tiles)
    for t in 1:num_tiles
        tiles[t] = createGrid(GridParameters(
        geometry = patch.params.geometry,
        xmin = tile_params[1,t],
        xmax = tile_params[2,t],
        num_cells = tile_params[3,t],
        BCL = Dict(key => CubicBSpline.R0 for key in keys(patch.params.vars)),
        BCR = Dict(key => CubicBSpline.R0 for key in keys(patch.params.vars)),
        vars = patch.params.vars,
        spectralIndexL = tile_params[4,t],
        tile_num = t))
    end

    for t in 1:num_tiles
        gridTransform!(patch,tiles[t])
        spectralTransform!(tiles[t])
    end

    return patch, tiles
end

function read_initialconditions(ic::String, grid::R_Grid)
    
    # 1D radius grid
    initialconditions = CSV.read(ic, DataFrame, header=1)
    
    # Check for match to grid
    if grid.params.rDim != length(initialconditions.r)
        throw(DomainError(length(initialconditions.r), 
                "IC radius dimension does not match model parameters"))
    end
    
    for key in keys(grid.params.vars)
        foundkey = false
        for name in names(initialconditions)
            if (name == key)
                foundkey = true
            end
        end
        if foundkey == false
            throw(DomainError(key, "IC missing data"))
        end
    end
    
    # Assign variables
    for (key, value) in pairs(grid.params.vars)
        grid.physical[:,value,1] .= select(initialconditions, key)
    end

end

function read_initialconditions(ic::String, grid::RZ_Grid)
    
    # 2D radius-height grid
    initialconditions = CSV.read(ic, DataFrame, header=1)
    
    # Check for match to grid
    # Can be more sophisticated here, just checking matching dimensions for now
    if (grid.params.rDim * grid.params.zDim) != length(initialconditions.r)
        throw(DomainError(length(initialconditions.r), 
                "IC dimensions do not match model parameters"))
    end
    
    for key in keys(grid.params.vars)
        foundkey = false
        for name in names(initialconditions)
            if (name == key)
                foundkey = true
            end
        end
        if foundkey == false
            throw(DomainError(key, "IC missing data"))
        end
    end
    
    # Assign variables
    for (key, value) in pairs(grid.params.vars)
        grid.physical[:,value,1] .= select(initialconditions, key)
    end

end

function read_initialconditions(ic::String, grid::RL_Grid)
    
    # 2D radius-lambda grid
    initialconditions = CSV.read(ic, DataFrame, header=1)
    
    # Check for match to grid
    # Can be more sophisticated here, just checking matching dimensions for now
    if (grid.params.lDim) != length(initialconditions.r)
        throw(DomainError(length(initialconditions.r), 
                "IC dimensions do not match model parameters"))
    end
    
    for key in keys(grid.params.vars)
        foundkey = false
        for name in names(initialconditions)
            if (name == key)
                foundkey = true
            end
        end
        if foundkey == false
            throw(DomainError(key, "IC missing data"))
        end
    end
    
    # Assign variables
    for (key, value) in pairs(grid.params.vars)
        grid.physical[:,value,1] .= select(initialconditions, key)
    end

end

function run_model(grid, model::ModelParameters)

    println("Model starting up with single tile...")

    num_ts = round(Int,model.integration_time / model.ts)
    output_int = round(Int,model.output_interval / model.ts)
    println("Integrating $(model.ts) sec increments for $(num_ts) timesteps")

    # Declare these here to avoid excessive allocations
    udot = zeros(Float64,size(grid.physical,1),size(grid.physical,2))
    fluxes = zeros(Float64,size(grid.physical,1),size(grid.physical,2))
    bdot = zeros(Float64,size(grid.spectral))
    bdot_delay = zeros(Float64,size(grid.spectral))
    b_nxt = zeros(Float64,size(grid.spectral))
    bdot_n1 = zeros(Float64,size(grid.spectral))
    bdot_n2 = zeros(Float64,size(grid.spectral))

    gridpoints = getGridpoints(grid)

    for t = 1:num_ts
        println("ts: $(t*model.ts)")

        # Feed physical matrices to physical equations
        physical_model(grid,gridpoints,udot,fluxes,model)

        # Convert to spectral tendencies
        calcTendency(grid,udot,fluxes,bdot,bdot_delay)

        # Advance the timestep
        if t > 2
            timestep(grid.spectral, bdot, bdot_delay, b_nxt, bdot_n1, bdot_n2, model.ts)
        elseif t == 2
            second_timestep(grid.spectral, bdot, bdot_delay, b_nxt, bdot_n1, bdot_n2, model.ts)
        else
            first_timestep(grid.spectral, bdot, bdot_delay, b_nxt, bdot_n1, model.ts)
        end

        # Assign b_nxt and b_now
        grid.spectral .= b_nxt
        gridTransform!(grid)

        #b_now is held in grid.spectral
        spectralTransform!(grid)

        if mod(t,output_int) == 0
            checkCFL(grid)
            write_output(grid, model, (t*model.ts))
        end
    end

    println("Done with time integration")
end

function run_model(patch, tiles, model::ModelParameters)
    
    num_tiles = length(tiles)
    println("Model starting up with $(num_tiles) tiles...")
    
    num_ts = round(Int,model.integration_time / model.ts)
    output_int = round(Int,model.output_interval / model.ts)
    println("Integrating $(model.ts) sec increments for $(num_ts) timesteps")

    #Initial test with just the patch
    grid = patch
    
    # Declare these here to avoid excessive allocations
    udot = zeros(Float64,size(grid.physical,1),size(grid.physical,2))
    fluxes = zeros(Float64,size(grid.physical,1),size(grid.physical,2))
    bdot = zeros(Float64,size(grid.spectral))
    bdot_delay = zeros(Float64,size(grid.spectral))
    b_nxt = zeros(Float64,size(grid.spectral))
    bdot_n1 = zeros(Float64,size(grid.spectral))
    bdot_n2 = zeros(Float64,size(grid.spectral))

    gridpoints = getGridpoints(grid)

    for t = 1:num_ts
        println("ts: $(t*model.ts)")

        # Feed physical matrices to physical equations
        physical_model(grid,gridpoints,udot,fluxes,model)

        # Convert to spectral tendencies
        calcTendency(grid,udot,fluxes,bdot,bdot_delay)

        # Advance the timestep
        if t > 2
            timestep(grid.spectral, bdot, bdot_delay, b_nxt, bdot_n1, bdot_n2, model.ts)
        elseif t == 2
            second_timestep(grid.spectral, bdot, bdot_delay, b_nxt, bdot_n1, bdot_n2, model.ts)
        else
            first_timestep(grid.spectral, bdot, bdot_delay, b_nxt, bdot_n1, model.ts)
        end

        # Assign b_nxt and b_now
        grid.spectral .= b_nxt
        gridTransform!(grid)

        #b_now is held in grid.spectral
        spectralTransform!(grid)

        if mod(t,output_int) == 0
            checkCFL(grid)
            write_output(grid, model, (t*model.ts))
        end
    end

    println("Done with time integration")
end

function finalize_model(grid, model::ModelParameters)
    
    write_output(grid, model, model.integration_time)
    println("Model complete!")
end

function physical_model(grid, 
            gridpoints::Array{real},
            vardot::Array{real},
            F::Array{real},
            model::ModelParameters)
    
    equation_set = Symbol(model.equation_set)
    equation_call = getfield(NumericalModels, equation_set)
    equation_call(grid, gridpoints, vardot, F, model)
    return
    
end

function first_timestep(spectral::Array{real}, 
        bdot::Array{real},
        bdot_delay::Array{real},
        b_nxt::Array{real},
        bdot_n1::Array{real},
        ts::real)    

    # Use Euler method for first step
    b_nxt .= spectral .+ (ts .* bdot) .+ (ts .* bdot_delay)
    bdot_n1 .= bdot
    
    # Override diagnostic variables with diagnostic_flag
    # TBD
    
end

function second_timestep(spectral::Array{real}, 
        bdot::Array{real},
        bdot_delay::Array{real},
        b_nxt::Array{real},
        bdot_n1::Array{real},
        bdot_n2::Array{real},
        ts::real) 
    
    # Use 2nd order A-B method for second step
    b_nxt .= spectral .+ (0.5 * ts) .* ((3.0 .* bdot) .- bdot_n1) .+ (ts .* bdot_delay)
    bdot_n1 .= bdot
    bdot_n2 .= bdot_n1
    
    # Override diagnostic variables with diagnostic_flag
    # TBD
end

function timestep(spectral::Array{real}, 
        bdot::Array{real},
        bdot_delay::Array{real},
        b_nxt::Array{real},
        bdot_n1::Array{real},
        bdot_n2::Array{real},
        ts::real) 

    # Use 3rd order A-B method for subsequent steps
    onetwelvets = ts/12.0
    b_nxt .= spectral .+ (onetwelvets .* ((23.0 .* bdot) - (16.0 .* bdot_n1) + (5.0 .* bdot_n2))) .+ (ts .* bdot_delay)
    bdot_n1 .= bdot
    bdot_n2 .= bdot_n1
    
    # Override diagnostic variables with diagnostic_flag
    # TBD
end

function calcTendency(grid,
        udot::Array{real},
        F::Array{real},
        bdot::Array{real},
        bdot_delay::Array{real})

    # Make sure any diagnostic variables are converted to bnow
    spectralTransform!(grid)
    
    # Transform udot to bdot
    spectralTransform(grid, udot, bdot)
    
    # Transform F to bdot_delay
    # This does nothing for RZ grid currently
    # RL is clone of spectralTransform for constant diffusion coefficient
    spectralxTransform(grid, F, bdot_delay)
    
end

function write_output_old(grid::R_Grid, model::ModelParameters, t::real)
    
    time = round(t; digits=2)
    if !isdir(model.output_dir)
        mkdir(model.output_dir)
    end
    
    println("Writing output to $(model.output_dir) at time $time")
    for var in keys(model.grid_params.vars)
        v = model.grid_params.vars[var]
        dir = model.output_dir
        afilename = string(dir, "model_a_", var , "_", time, ".csv")
        ufilename = string(dir, "model_", var , "_", time, ".csv")
        afile = open(afilename,"w")
        ufile = open(ufilename,"w")

        a = grid.splines[v].a    
        for i = 1:grid.splines[v].aDim
            a_i = a[i]
            write(afile,"$i, $a_i\n")
        end        

        mishPoints = grid.splines[v].mishPoints
        for i = 1:grid.splines[v].mishDim
            mp_i = mishPoints[i]
            u_i = grid.splines[v].uMish[i]
            write(ufile,"$i, $mp_i, $u_i\n")
        end
        close(afile)
        close(ufile)
    end
    
    # Write nodes to a single file, including vorticity
    #outfilename = string(model.equation_set , "_output_", time, ".csv")
    #outfile = open(outfilename,"w")
    #r = zeros(real,splines[1].aDim)
    #vort = zeros(real,splines[1].aDim)
    #for i = 1:splines[1].params.num_cells
    #    r[i] = splines[1].params.xmin + (splines[1].params.DX * (i-1))
    #end
    #
    #vgr = CubicBSpline.SItransform(splines[model.vars["vgr"]].params,splines[model.vars["vgr"]].a,r,0)
    #u = CubicBSpline.SItransform(splines[model.vars["u"]].params,splines[model.vars["u"]].a,r,0)
    #v = CubicBSpline.SItransform(splines[model.vars["v"]].params,splines[model.vars["v"]].a,r,0)
    #dvdr = CubicBSpline.SItransform(splines[model.vars["v"]].params,splines[model.vars["v"]].a,r,1)
    #w = CubicBSpline.SItransform(splines[model.vars["w"]].params,splines[model.vars["w"]].a,r,0)
    #vort .= dvdr .+ (v ./ r)
    #
    #if r[1] == 0.0
    #    vort[1] = 0.0
    #end
    #
    #write(outfile,"r,vgr,u,v,w,vort\n")
    #for i = 1:splines[1].params.num_cells
    #    data = string(r[i], ",", vgr[i], ",", u[i], ",", v[i], ",", w[i], ",", vort[i])
    #    write(outfile,"$data\n")
    #end        
    #close(outfile)
end

function write_output(grid::R_Grid, model::ModelParameters, t::real)
    
    time = round(t; digits=2)
    if !isdir(model.output_dir)
        mkdir(model.output_dir)
    end
    
    println("Writing output to $(model.output_dir) at time $time")
    dir = model.output_dir
    afilename = string(dir, "spectral_out_", time, ".csv")
    ufilename = string(dir, "physical_out_", time, ".csv")
    afile = open(afilename,"w")
    ufile = open(ufilename,"w")

    aheader = "r,"
    uheader = "r,"
    suffix = ["","_r","_rr"]
    for d = 1:3
        for var in keys(grid.params.vars)
            if (d == 1)
                aheader *= "$var,"
            end
            varout = var * suffix[d]
            uheader *= "$varout,"
        end
    end
    aheader = chop(aheader) * "\n"
    uheader = chop(uheader) * "\n"
    write(afile,aheader)
    write(ufile,uheader)
    
    for r = 1:grid.params.b_rDim
        astring = "$r,"
        for var in keys(grid.params.vars)
            v = grid.params.vars[var]
            a = grid.spectral[r,v]
            astring *= "$(a),"
        end
        astring = chop(astring) * "\n"
        write(afile,astring)
    end
    close(afile)
    
    for r = 1:grid.params.rDim
        radii = grid.splines[1].mishPoints
        ustring = "$(radii[r]),"
        for d = 1:3
            for var in keys(grid.params.vars)
                v = grid.params.vars[var]
                u = grid.physical[r,v,d]
                ustring *= "$u,"
            end
        end
        ustring = chop(ustring) * "\n"
        write(ufile,ustring)
    end
    close(ufile)
end

function write_output(grid::RZ_Grid, model::ModelParameters, t::real)
    
    time = round(t; digits=2)
    if !isdir(model.output_dir)
        mkdir(model.output_dir)
    end
    
    println("Writing output to $(model.output_dir) at time $time")
    dir = model.output_dir
    afilename = string(dir, "spectral_out_", time, ".csv")
    ufilename = string(dir, "physical_out_", time, ".csv")
    afile = open(afilename,"w")
    ufile = open(ufilename,"w")

    aheader = "r,z,"
    uheader = "r,z,"
    suffix = ["","_r","_rr","_z","_zz"]
    for d = 1:5
        for var in keys(grid.params.vars)
            if (d == 1)
                aheader *= "$var,"
            end
            varout = var * suffix[d]
            uheader *= "$varout,"
        end
    end
    aheader = chop(aheader) * "\n"
    uheader = chop(uheader) * "\n"
    write(afile,aheader)
    write(ufile,uheader)
    
    for r = 1:grid.params.b_rDim
        for z = 1:grid.params.b_zDim
            astring = "$r,$z,"
            for var in keys(grid.params.vars)
                v = grid.params.vars[var]
                z1 = ((r-1)*grid.params.b_zDim)
                a = grid.spectral[z1+z,v]
                astring *= "$(a),"
            end
            astring = chop(astring) * "\n"
            write(afile,astring)
        end
    end
    close(afile)
    
    for z = 1:grid.params.zDim
        radii = grid.splines[z,1].mishPoints
        levels = grid.columns[1].mishPoints
        for r = 1:grid.params.rDim
            ustring = "$(radii[r]),$(levels[z]),"
            r1 = ((z-1)*grid.params.rDim)
            for d = 1:5
                for var in keys(grid.params.vars)
                    v = grid.params.vars[var]
                    u = grid.physical[r1+r,v,d]
                    ustring *= "$u,"
                end
            end
            ustring = chop(ustring) * "\n"
            write(ufile,ustring)
        end
    end
    close(ufile)
end

function write_output(grid::RL_Grid, model::ModelParameters, t::real)
    
    time = round(t; digits=2)
    if !isdir(model.output_dir)
        mkdir(model.output_dir)
    end
    
    println("Writing output to $(model.output_dir) at time $time")
    dir = model.output_dir
    afilename = string(dir, "spectral_out_", time, ".csv")
    ufilename = string(dir, "physical_out_", time, ".csv")
    rfilename = string(dir, "gridded_out_", time, ".csv")
    afile = open(afilename,"w")
    ufile = open(ufilename,"w")
    rfile = open(rfilename,"w")

    aheader = "r,k,"
    uheader = "r,l,x,y,"
    rheader = "r,l,x,y,"
    suffix = ["","_r","_rr","_l","_ll"]
    for d = 1:5
        for var in keys(grid.params.vars)
            if (d == 1)
                aheader *= "$var,"
            end
            varout = var * suffix[d]
            uheader *= "$varout,"
            rheader *= "$varout,"
        end
    end
    aheader = chop(aheader) * "\n"
    uheader = chop(uheader) * "\n"
    rheader = chop(rheader) * "\n"
    write(afile,aheader)
    write(ufile,uheader)
    write(rfile,rheader)
        
    # Wave 0
    for r = 1:grid.params.b_rDim
        astring = "$r,0,"
        for var in keys(grid.params.vars)
            v = grid.params.vars[var]
            a = grid.spectral[r,v]
            astring *= "$(a),"
        end
        astring = chop(astring) * "\n"
        write(afile,astring)
    end
    
    # Higher wavenumbers
    for k = 1:grid.params.rDim
        for r = 1:grid.params.b_rDim
            astring = "$r,$(k)r,"
            kr = ((k*2-1)*grid.params.b_rDim)+r
            for var in keys(grid.params.vars)
                v = grid.params.vars[var]
                a = grid.spectral[kr,v]
                astring *= "$(a),"
            end
            astring = chop(astring) * "\n"
            write(afile,astring)
        end
        for r = 1:grid.params.b_rDim
            astring = "$r,$(k)i,"
            ki = ((k*2+1)*grid.params.b_rDim)+r
            for var in keys(grid.params.vars)
                v = grid.params.vars[var]
                a = grid.spectral[ki,v]
                astring *= "$(a),"
            end
            astring = chop(astring) * "\n"
            write(afile,astring)
        end
    end
    close(afile)
    
    l1 = 0
    l2 = 0
    gridpoints = getGridpoints(grid)
    cartesianpoints = getCartesianGridpoints(grid)
    for r = 1:grid.params.rDim
        l1 = l2 + 1
        l2 = l1 + 3 + (4*r)
        for l = l1:l2
            ustring = "$(gridpoints[l,1]),$(gridpoints[l,2]),$(cartesianpoints[l,1]),$(cartesianpoints[l,2]),"
            for d = 1:5
                for var in keys(grid.params.vars)
                    v = grid.params.vars[var]
                    u = grid.physical[l,v,d]
                    ustring *= "$u,"
                end
            end
            ustring = chop(ustring) * "\n"
            write(ufile,ustring)
        end
    end
    close(ufile)
    
    # Get regular grid
    regular_grid = regularGridTransform(grid)
    gridpoints = getRegularGridpoints(grid)
    for r = 1:grid.params.num_cells
        for l = 1:(grid.params.rDim*2+1)
            rstring = "$(gridpoints[r,l,1]),$(gridpoints[r,l,2]),$(gridpoints[r,l,3]),$(gridpoints[r,l,4]),"
            for d = 1:5
                for var in keys(grid.params.vars)
                    v = grid.params.vars[var]
                    u = regular_grid[r,l,v,d]
                    rstring *= "$u,"
                end
            end
            rstring = chop(rstring) * "\n"
            write(rfile,rstring)
        end
    end
    close(rfile)
    
end

function checkCFL(grid)
    
    # Check to see if CFL condition may have been violated 
    for var in keys(grid.params.vars)
        v = grid.params.vars[var]
        testvar = grid.physical[:,v,1]
        for i in eachindex(testvar)
            if (isnan(testvar[i]))
                error("NaN found in variable $var at index$(i) ! CFL condition likely violated")
            end
            # Can do more extensive checks here to see if collapse is impending
            #TBD
        end
    end
end

# Module end
end
