# Functions for model integration

using Distributed
using DistributedData
using SharedArrays
using CSV
using DataFrames
using MPI
using LoopVectorization
import Base.Threads.@spawn
using SparseArrays
using SuiteSparse

# Need to export these for distributed operations in Main namespace
export createModelTile, advanceTimestep
export initialize_model, run_model, finalize_model

struct ModelTile
    model::ModelParameters
    tile::AbstractGrid
    var_nxt::Array{Float64}
    expdot_n::Array{Float64}
    expdot_nm1::Array{Float64}
    expdot_nm2::Array{Float64}
    impdot_np1::Array{Float64}
    impdot_n::Array{Float64}
    impdot_nm1::Array{Float64}
    tilepoints::Array{Float64}
    patchSplines::Array{Spline1D}
    patchSpectral::Array{Float64}
    patchIndexMap::BitMatrix
    tileView::AbstractArray
    haloSendIndexMap::BitMatrix
    haloSendView::AbstractArray
    haloReceiveIndexMap::BitMatrix
    haloReceiveBuffer::Array{Float64}
    splineBuffer::Array{Float64}
end

function createModelTile(patch::AbstractGrid, tile::AbstractGrid, model::ModelParameters,
        haloReceiveIndexMap::BitMatrix)

    # Allocate some needed arrays
    var_nxt = zeros(Float64,size(tile.physical,1),size(tile.physical,2))
    expdot_n = zeros(Float64,size(tile.physical,1),size(tile.physical,2))
    expdot_nm1 = zeros(Float64,size(tile.physical,1),size(tile.physical,2))
    expdot_nm2 = zeros(Float64,size(tile.physical,1),size(tile.physical,2))
    impdot_np1 = zeros(Float64,size(tile.physical,1),size(tile.physical,2))
    impdot_n = zeros(Float64,size(tile.physical,1),size(tile.physical,2))
    impdot_nm1 = zeros(Float64,size(tile.physical,1),size(tile.physical,2))
    
    # Get the local gridpoints
    tilepoints = getGridpoints(tile)

    # Copy over the patch information
    patchSplines = copy(patch.splines)
    patchSpectral = copy(patch.spectral)

    # Set up the map between the tile and the patch
    patchMap = calcPatchMap(patch, tile)
    patchIndexMap = patchMap[1]
    tileView = patchMap[2]

    # Set up the map between the tile and its neighbor
    haloMap = calcHaloMap(patch, tile)
    haloSendIndexMap = haloMap[1]
    haloSendView = haloMap[2]

    # Set up some buffers to avoid excessive allocations
    haloReceiveBuffer = zeros(Float64,size(patch.spectral[haloReceiveIndexMap]))
    splineBuffer =  allocateSplineBuffer(patch,tile)

    mtile = ModelTile(
        model,
        tile,
        var_nxt,
        expdot_n,
        expdot_nm1,
        expdot_nm2,
        impdot_np1,
        impdot_n,
        impdot_nm1,
        tilepoints,
        patchSplines,
        patchSpectral,
        patchIndexMap,
        tileView,
        haloSendIndexMap,
        haloSendView,
        haloReceiveIndexMap,
        haloReceiveBuffer,
        splineBuffer)
    return mtile
end

function initialize_model(model::ModelParameters, workerids::Vector{Int64})

    num_workers = length(workerids)
    println("Initializing with $(num_workers) workers and tiles")
    patch = createGrid(model.grid_params)
    println("$model")

    # Initialize the patch locally on master process
    read_initialconditions(model.initial_conditions, patch)
    spectralTransform!(patch)
    gridTransform!(patch)
    write_output(patch, model, 0.0)

    # Transfer the model and patch/tile info to each worker
    println("Initializing workers")
    # Print the tile information
    tile_params = calcTileSizes(patch, num_workers)
    for w in workerids
        println("Worker $w: $(tile_params[5,w-1]) gridpoints in $(tile_params[3,w-1]) cells from $(tile_params[1,w-1]) to $(tile_params[2,w-1]) starting at index $(tile_params[4,w-1])")
    end

    map(wait, [save_at(w, :model, model) for w in workerids])
    map(wait, [save_at(w, :workerids, workerids) for w in workerids])
    map(wait, [save_at(w, :num_workers, num_workers) for w in workerids])
    map(wait, [save_at(w, :patch, :(createGrid(model.grid_params))) for w in workerids])
    map(wait, [save_at(w, :tile_params, :(calcTileSizes(patch, num_workers))) for w in workerids])

    # Distribute the tiles
    println("Initializing tiles on workers")
    map(wait, [save_at(w, :tile, :(createGrid(GridParameters(
            geometry = patch.params.geometry,
            xmin = tile_params[1,myid()-1],
            xmax = tile_params[2,myid()-1],
            num_cells = tile_params[3,myid()-1],
            zmin = patch.params.zmin,
            zmax = patch.params.zmax,
            zDim = patch.params.zDim,
            BCL = Dict(key => CubicBSpline.R0 for key in keys(patch.params.vars)),
            BCR = Dict(key => CubicBSpline.R0 for key in keys(patch.params.vars)),
            BCB = patch.params.BCB,
            BCT = patch.params.BCT,
            vars = patch.params.vars,
            spectralIndexL = tile_params[4,myid()-1],
            tile_num = myid())))) for w in workerids])

    # Create the model tiles
    println("Initializing modelTiles on workers")

    # First tile receives an empty halo from master to simplify later loops
    firstMap = falses(size(patch.spectral))
    firstMap[1] = true

    # Precalculate indices and allocate buffers for shared and border transfers
    wait(save_at(workerids[1], :mtile, :(createModelTile(patch,tile,model,$(firstMap)))))
    for w in workerids[1:length(workerids)-1]
        send_index = w + 1
        sendMap = get_val_from(w, :(mtile.haloSendIndexMap))
        wait(save_at(send_index, :mtile, :(createModelTile(patch,tile,model,$(sendMap)))))
    end

    # Delete the patch from the workers since the relevant info is already in the modelTile
    # Don't delete from the first worker in case they are also the master
    map(wait, [remove_from(w, :patch) for w in workerids[2:length(workerids)]])

    # Transform the patch and return to the main process
    spectralTransform!(patch)

    println("Ready for time integration!")
    return patch
end

function run_model(patch::AbstractGrid, model::ModelParameters, workerids::Vector{Int64})

    num_workers = length(workerids)
    println("Model starting up with $(num_workers) workers and tiles...")

    # Establish RemoteChannel connections between workers
    println("Connecting workers")

    # Master sends to first worker (itself)
    haloInit = RemoteChannel(()->Channel{Array{Float64}}(1),workerids[1])
    wait(save_at(workerids[1], :haloReceive, :($(haloInit))))

    # Each worker passes information up the chain
    for w in workerids[1:length(workerids)-1]
        send_index = w + 1
        wait(save_at(w, :haloSend,
                :(RemoteChannel(()->Channel{Array{Float64}}(1),$(send_index)))))
        receiver = get_val_from(w, :haloSend)
        wait(save_at(send_index, :haloReceive, :($(receiver))))
    end

    # Master receives from the last worker
    wait(save_at(last(workerids), :haloSend,
            :(RemoteChannel(()->Channel{Array{Float64}}(1),workerids[1]))))
    haloReceive = get_val_from(last(workerids), :haloSend)

    # First tile receives an empty halo from master to simplify later loops
    haloInitBuffer = zeros(Float64,1)

    # Last tile is received by master process
    haloReceiveIndexMap = get_val_from(last(workerids), :(mtile.haloSendIndexMap))
    haloReceiveBuffer = zeros(Float64,size(patch.spectral[haloReceiveIndexMap]))

    # Create a shared array for the spectral sum
    sharedSpectral = SharedArray{Float64,2}((size(patch.spectral,1),size(patch.spectral,2)))
    results = Array{Future}(undef,num_workers+1)

    # Initialize at time zero
    sharedSpectral[:] .= patch.spectral[:]
    for w in workerids
        save_at(w, :sharedSpectral, sharedSpectral)
    end
    map(wait, [get_from(w, :(splineTransform!(mtile.patchSplines, mtile.patchSpectral, mtile.model.grid_params, sharedSpectral,mtile.tile))) for w in workerids])

    # Loop through the model timesteps
    @time model_loop(patch, model, workerids, sharedSpectral, haloInit, haloReceive,
        haloInitBuffer, haloReceiveBuffer, haloReceiveIndexMap)

    # Integration complete! Finalize the patch
    patch.spectral .= sharedSpectral
    gridTransform!(patch)
    spectralTransform!(patch)
    println("Done with time integration")
    return true

end

function model_loop(patch::AbstractGrid, model::ModelParameters, workerids::Vector{Int64},
        sharedSpectral::SharedArray{Float64}, haloInit::RemoteChannel, haloReceive::RemoteChannel,
        haloInitBuffer::Array{Float64}, haloReceiveBuffer::Array{Float64}, haloReceiveIndexMap::BitMatrix)

    # Set up the timesteps
    num_ts = round(Int,model.integration_time / model.ts)
    output_int = round(Int,model.output_interval / model.ts)
    println("Integrating $(model.ts) sec increments for $(num_ts) timesteps")

    # Loop through the timesteps
    for t = 1:num_ts
        println("ts: $(t*model.ts)")

        # Master process clears the shared array and sends an empty halo to the first worker
        @turbo sharedSpectral .= 0.0
        put!(haloInit, haloInitBuffer)

        # Advance each tile
        map(wait, [get_from(w, :(advanceTimestep(mtile, sharedSpectral, haloSend, haloReceive, $(t)))) for w in workerids])

        # Get halo from previous tile
        haloReceiveBuffer .= take!(haloReceive)

        # Add it to the sharedArray
        @inbounds sharedSpectral[haloReceiveIndexMap] .+= haloReceiveBuffer

        # Output if on specified time interval
        if mod(t,output_int) == 0
            patch.spectral .= sharedSpectral
            gridTransform!(patch)
            spectralTransform!(patch)
            checkCFL(patch)
            @async write_output(patch, model, (t*model.ts))
        end

        # Reset the shared spectral patch to the tiles
        map(wait, [get_from(w, :(splineTransform!(mtile.patchSplines, mtile.patchSpectral, mtile.model.grid_params, sharedSpectral,mtile.tile))) for w in workerids])

        # Done with this timestep
    end
    return nothing
end

function advanceTimestep(mtile::ModelTile, sharedSpectral::SharedArray{Float64}, 
        haloSend::RemoteChannel, haloReceive::RemoteChannel, t::int)

    # Transform to local physical tile
    tileTransform!(mtile.patchSplines, mtile.patchSpectral, mtile.model.grid_params, mtile.tile, mtile.splineBuffer)

    # Feed physical matrices to physical equations
    physical_model(mtile)
    
    # Solve for semi-implicit n+1 terms
    semiimplicit_solver(mtile)

    # Advance the timestep
    timestep(mtile, t)

    # Convert current timestep to spectral tendencies
    calcTendency(mtile)

    # Send halo to next tile
    put!(haloSend, mtile.haloSendView)

    # Set the sharedArray this tile is responsible for
    sharedSpectral[mtile.patchIndexMap] .= mtile.tileView

    # Get halo from previous tile
    mtile.haloReceiveBuffer .= take!(haloReceive)

    # Add it to the sharedArray
    sharedSpectral[mtile.haloReceiveIndexMap] .+= mtile.haloReceiveBuffer

    return nothing
end

function finalize_model(grid::AbstractGrid, model::ModelParameters)
    
    write_output(grid, model, model.integration_time)
    println("Model complete!")
end

function physical_model(mtile::ModelTile)
        
    equation_set = Symbol(mtile.model.equation_set)
    equation_call = getfield(Scythe, equation_set)
    equation_call(mtile.tile, mtile.tilepoints, mtile.expdot_n, mtile.impdot_n, mtile.model)
    return
end

function semiimplicit_solver(mtile::ModelTile)
    
    # Do something here
    
end

function timestep(mtile::ModelTile,t::Int64)

    physical = view(mtile.tile.physical,:,:,1)
    ts = mtile.model.ts 

    if (t == 1)
        # Use Euler method and trapezoidal method (AM2) for first step
        mtile.var_nxt .= @. physical + 
            (ts * mtile.expdot_n) +
            (0.5 * ts * (mtile.impdot_np1 + mtile.impdot_n))
        mtile.expdot_nm1 .= mtile.expdot_n
        mtile.impdot_nm1 .= mtile.impdot_n
    elseif (t == 2) 
        # Use 2nd order A-B method and AI2* for second step
        mtile.var_nxt .= @. physical + (0.5 * ts) * ((3.0 * mtile.expdot_n) - mtile.expdot_n) + (0.25 * ts * ((5.0 * mtile.impdot_np1) - (4.0 * mtile.impdot_n) + (3.0 * mtile.impdot_nm1)))
        mtile.expdot_nm1 .= mtile.expdot_n
        mtile.expdot_nm2 .= mtile.expdot_nm1
        mtile.impdot_nm1 .= mtile.impdot_n
    else
        # Use AI2*–AB3 implicit-explicit scheme (Durran and Blossey 2012)
        mtile.var_nxt .= @. physical + ((ts / 12.0) * ((23.0 * mtile.expdot_n) - (16.0 * mtile.expdot_n) + (5.0 * mtile.expdot_nm2))) + (0.25 * ts * ((5.0 * mtile.impdot_np1) - (4.0 * mtile.impdot_n) + (3.0 * mtile.impdot_nm1)))
        mtile.expdot_nm1 .= mtile.expdot_n
        mtile.expdot_nm2 .= mtile.expdot_nm1
        mtile.impdot_nm1 .= mtile.impdot_n
    end
end

function calcTendency(mtile::ModelTile)

    # Set the current time
    mtile.tile.physical .= mtile.var_nxt
    
    # Transform to spectral space
    spectralTransform!(mtile.tile)    
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