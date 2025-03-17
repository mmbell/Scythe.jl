# Functions for model integration

using Distributed
using DistributedData
using SharedArrays
using CSV
using DataFrames
using LoopVectorization
using LinearAlgebra
import Base.Threads.@spawn
using SparseArrays
using SuiteSparse

# Need to export these for distributed operations in Main namespace
export createModelTile, advanceTimestep
export initialize_model, run_model, finalize_model

struct ModelTile
    model::ModelParameters
    tile::AbstractGrid
    var_np1::Array{Float64}
    expdot_incr::Array{Float64}
    expdot_n::Array{Float64}
    expdot_nm1::Array{Float64}
    expdot_nm2::Array{Float64}
    impdot_np1::Array{Float64}
    impdot_n::Array{Float64}
    impdot_nm1::Array{Float64}
    impdot_nm2::Array{Float64}
    tilepoints::Array{Float64}
    ref_state::ReferenceState
    patchSplines::Array{Spline1D}
    patchSpectral::Array{Float64}
    patchIndexMap::BitMatrix
    tileView::AbstractArray
    haloSendIndexMap::BitMatrix
    haloSendView::AbstractArray
    haloReceiveIndexMap::BitMatrix
    haloReceiveBuffer::Array{Float64}
    splineBuffer::Array{Float64}
    h_matrix::Factorization
end

function createModelTile(patch::AbstractGrid, tile::AbstractGrid, model::ModelParameters,
        haloReceiveIndexMap::BitMatrix)

    # Allocate some needed arrays
    var_np1 = zeros(Float64,size(tile.physical,1),size(tile.physical,2))
    expdot_incr = zeros(Float64,size(tile.physical,1),size(tile.physical,2))
    expdot_n = zeros(Float64,size(tile.physical,1),size(tile.physical,2))
    expdot_nm1 = zeros(Float64,size(tile.physical,1),size(tile.physical,2))
    expdot_nm2 = zeros(Float64,size(tile.physical,1),size(tile.physical,2))
    impdot_np1 = zeros(Float64,size(tile.physical,1),size(tile.physical,2))
    impdot_n = zeros(Float64,size(tile.physical,1),size(tile.physical,2))
    impdot_nm1 = zeros(Float64,size(tile.physical,1),size(tile.physical,2))
    impdot_nm2 = zeros(Float64,size(tile.physical,1),size(tile.physical,2))
    
    # Get the local gridpoints
    tilepoints = getGridpoints(tile)

    # Set up the reference file
    ref_state = empty_reference_state()
    if !isempty(model.ref_state_file)
        z_values = tilepoints[1:model.grid_params.zDim,ndims(tilepoints)]

        if (model.options[:exact_reference_state])
            # Use an pre-calculated exact state rather than interpolate
            ref_state = exact_reference_state(model, z_values)
        else
            ref_state = interpolate_reference_file(model, z_values)
        end
    end

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

    # Pre-calculate the Helmholtz matrix for semi-implicit adjustment
    # Declare a basic factorization for the structure if semiimplicit integration is not used
    h_matrix = factorize([1 2; 2 1])
    if model.options[:semiimplicit]
        h_matrix = calc_Helmholtz_semiimplicit_matrix(model, ref_state.Pxi_bar, 1.25 * model.ts)
    end

    mtile = ModelTile(
        model,
        tile,
        var_np1,
        expdot_incr,
        expdot_n,
        expdot_nm1,
        expdot_nm2,
        impdot_np1,
        impdot_n,
        impdot_nm1,
        impdot_nm2,
        tilepoints,
        ref_state,
        patchSplines,
        patchSpectral,
        patchIndexMap,
        tileView,
        haloSendIndexMap,
        haloSendView,
        haloReceiveIndexMap,
        haloReceiveBuffer,
        splineBuffer,
        h_matrix)
    return mtile
end

function initialize_model(model::ModelParameters, workerids::Vector{Int64})

    num_workers = length(workerids)
    println("Initializing with $(num_workers) workers and tiles")
    patch = createGrid(model.grid_params)
    println("$model")

    # Initialize the patch locally on master process
    read_physical_grid(model.initial_conditions, patch)
    spectralTransform!(patch)
    gridTransform!(patch)

    # Transfer the model and patch/tile info to each worker
    println("Initializing workers")
    # Print the tile information
    tile_params = calcTileSizes(patch, num_workers)
    # This throws a bug if there is only one worker, probably other bugs too related to that
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

    println("Ready for time integration!")
    flush(stdout)
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

    # Output initial time
    patch.spectral .= get_val_from(workerids[1],:(mtile.patchSpectral))
    tileTransform!(patch.splines, patch.spectral, model.grid_params, patch, allocateSplineBuffer(patch,patch))
    checkCFL(patch)
    @async write_output(patch, model, 0.0)
    flush(stdout)

    # Loop through the model timesteps
    @time model_loop(patch, model, workerids, sharedSpectral, haloInit, haloReceive,
        haloInitBuffer, haloReceiveBuffer, haloReceiveIndexMap)

    # Integration complete! Finalize the patch
    patch.spectral .= get_val_from(workerids[1],:(mtile.patchSpectral))
    tileTransform!(patch.splines, patch.spectral, model.grid_params, patch, allocateSplineBuffer(patch,patch))
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

        # Reset the shared spectral patch to the tiles
        map(wait, [get_from(w, :(splineTransform!(mtile.patchSplines, mtile.patchSpectral, mtile.model.grid_params, sharedSpectral, mtile.tile))) for w in workerids])

        # Output if on specified time interval
        if mod(t,output_int) == 0
            patch.spectral .= get_val_from(workerids[1],:(mtile.patchSpectral))
            tileTransform!(patch.splines, patch.spectral, model.grid_params, patch, allocateSplineBuffer(patch,patch))
            checkCFL(patch)
            @async write_output(patch, model, (t*model.ts))
        end

        # Done with this timestep
        flush(stdout)
    end
    return nothing
end

function advanceTimestep(mtile::ModelTile, sharedSpectral::SharedArray{Float64}, 
        haloSend::RemoteChannel, haloReceive::RemoteChannel, t::Int64)

    # Transform to local physical tile
    tileTransform!(mtile.patchSplines, mtile.patchSpectral, mtile.model.grid_params, mtile.tile, mtile.splineBuffer)

    # Advance each column
    if num_columns(mtile.tile) > 0
        Threads.@threads for c in 1:num_columns(mtile.tile)
            advance_column(mtile, c, t)
        end
    else
        advance_column(mtile, -1, t)
    end

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

function advance_column(mtile::ModelTile, c::Int64, t::Int64)

    # Grab a column of indices
    colstart = (c-1) * mtile.model.grid_params.zDim + 1
    colend = colstart + mtile.model.grid_params.zDim - 1

    # If R or RL grid then set to the maximum dimensions
    if c == -1
        colstart = 1
        colend = size(mtile.tile.physical,1)
    end

    # Feed physical matrices to physical equations
    physical_model(mtile, colstart, colend, t)

end

function finalize_model(grid::AbstractGrid, model::ModelParameters)
    
    write_output(grid, model, model.integration_time)
    println("Model complete!")
end

function physical_model(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)
        
    equation_set = Symbol(mtile.model.equation_set)
    equation_call = getfield(Scythe, equation_set)
    equation_call(mtile, colstart, colend, t)
    return
end

function semiimplicit_timestep_old(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    w_index = mtile.model.grid_params.vars["w"]
    xi_index = mtile.model.grid_params.vars["xi"]
    ts = mtile.model.ts

    # Calculate xi_nstar
    xi_nstar = mtile.var_np1[colstart:colend,xi_index]
    wdot_n = view(mtile.impdot_n,colstart:colend,xi_index)
    wdot_nm1 = view(mtile.impdot_nm1,colstart:colend,xi_index)
    wdot_nm2 = view(mtile.impdot_nm2,colstart:colend,xi_index)

    # Calculate w_nstar
    w_nstar = mtile.var_np1[colstart:colend,w_index]
    xidot_n = view(mtile.impdot_n,colstart:colend,w_index)
    xidot_nm1 = view(mtile.impdot_nm1,colstart:colend,w_index)
    xidot_nm2 = view(mtile.impdot_nm2,colstart:colend,w_index)

    # Get the mean speed of sound squared from the reference state
    Pxi_bar = mtile.ref_state.Pxi_bar

    # Add the implicit terms
    ts_term = 0.0
    if (t == 1)
        # Use trapezoidal method (AM2) for first step
        ts_term = 0.5 * ts
        w_nstar .= w_nstar .+ (ts .* 0.5 .* xidot_n)
        xi_nstar .= xi_nstar .+ (ts .* 0.5 .* wdot_n)
    else
        # Use AI2* for second step and beyond
        ts_term = 1.25 * ts
        w_nstar .= w_nstar .- (ts .* xidot_n) .+ (ts .* 0.75 .* xidot_nm1)
        xi_nstar .= xi_nstar .- (ts .* wdot_n) .+ (ts .* 0.75 .* wdot_nm1)
    end

    # Set the n-1 and n-2 terms
    xidot_nm2 .= xidot_nm1
    xidot_nm1 .= xidot_n

    wdot_nm2 .= wdot_nm1
    wdot_nm1 .= wdot_n

    # Take the vertical derivative of w_nstar and multiply by ts term
    w_col = mtile.tile.columns[mtile.model.grid_params.vars["w"]]
    w_col.uMish .= w_nstar
    CBtransform!(w_col)
    CAtransform!(w_col)
    w_nstar = CItransform!(w_col)
    w_nstar_z = ts_term .* CIxtransform(w_col)

    # Set up the matrix problem
    nz = mtile.model.grid_params.zDim
    nbasis = mtile.model.grid_params.b_zDim
    g = xi_nstar .- w_nstar_z
    g = [0.0 ; 0.0; g[2:nz-1]]

    # Calculate the Helmholtz matrix
    dct = Chebyshev.dct_matrix(nz)
    column_length = mtile.model.grid_params.zmax - mtile.model.grid_params.zmin
    dct2 = Chebyshev.dct_2nd_derivative(nz, column_length)
    dct1 = Chebyshev.dct_1st_derivative(nz, column_length)
    h = (-ts_term .* ts_term .* Pxi_bar) .* dct2 .+ dct
    bc1 = (-ts_term .* ts_term .* Pxi_bar) .* dct1[1,:]
    bc2 = (-ts_term .* ts_term .* Pxi_bar) .* dct1[nz,:]
    h_a = [bc1[:]'; bc2[:]'; h[2:nz-1,:]]
    
    # Solve for the coefficients
    xi_a = h_a \ g

    # Set xi_n+1
    xi_col = mtile.tile.columns[mtile.model.grid_params.vars["xi"]]
    xi_col.a .= xi_a
    view(mtile.var_np1,colstart:colend,xi_index) .= CItransform!(xi_col)

    # Set w_n+1
    view(mtile.var_np1,colstart:colend,w_index) .= w_nstar .- (ts_term .* Pxi_bar .* CIxtransform(xi_col))
end

function semiimplicit_adjustment_xi(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    w_index = mtile.model.grid_params.vars["w"]
    xi_index = mtile.model.grid_params.vars["xi"]
    ts = mtile.model.ts

    # Calculate xi_nstar
    xi_nstar = mtile.var_np1[colstart:colend,xi_index]
    wdot_n = view(mtile.impdot_n,colstart:colend,xi_index)
    wdot_nm1 = view(mtile.impdot_nm1,colstart:colend,xi_index)
    wdot_nm2 = view(mtile.impdot_nm2,colstart:colend,xi_index)

    # Calculate w_nstar
    w_nstar = mtile.var_np1[colstart:colend,w_index]
    xidot_n = view(mtile.impdot_n,colstart:colend,w_index)
    xidot_nm1 = view(mtile.impdot_nm1,colstart:colend,w_index)
    xidot_nm2 = view(mtile.impdot_nm2,colstart:colend,w_index)

    # Get the mean speed of sound squared
    Pxi_bar = mtile.ref_state.Pxi_bar

    # Subtract the explicit terms and add the implicit terms
    ts_term = 0.0
    if (t == 1)
        # Use trapezoidal method (AM2) for first step
        ts_term = 0.5 * ts
        w_nstar .= @. w_nstar - (ts * xidot_n) + (ts * 0.5 * xidot_n)
        xi_nstar .= @. xi_nstar - (ts * wdot_n) + (ts * 0.5 * wdot_n)
    elseif (t == 2)
        # Use AI2* for second step and beyond
        ts_term = 1.25 * ts
        w_nstar .= @. w_nstar - (0.5 * ts) * ((3.0 * xidot_n) - xidot_nm1) - (ts * xidot_n) + (ts * 0.75 * xidot_nm1)
        xi_nstar .= @. xi_nstar - (0.5 * ts) * ((3.0 * wdot_n) - wdot_nm1) - (ts * wdot_n) + (ts * 0.75 * wdot_nm1)
    else
        # Use AI2* for second step and beyond
        ts_term = 1.25 * ts
        w_nstar .= @. w_nstar - ((ts / 12.0) * ((23.0 * xidot_n) - (16.0 * xidot_nm1) + (5.0 * xidot_nm2))) - (ts * xidot_n) + (ts * 0.75 * xidot_nm1)
        xi_nstar .= @. xi_nstar - ((ts / 12.0) * ((23.0 * wdot_n) - (16.0 * wdot_nm1) + (5.0 * wdot_nm2))) - (ts * wdot_n) + (ts * 0.75 * wdot_nm1)
    end

    # Set the n-1 and n-2 terms
    xidot_nm2 .= xidot_nm1
    xidot_nm1 .= xidot_n

    wdot_nm2 .= wdot_nm1
    wdot_nm1 .= wdot_n

    # Take the vertical derivative of w_nstar and multiply by ts term
    w_col = deepcopy(mtile.tile.columns[mtile.model.grid_params.vars["w"]])
    w_col.uMish .= w_nstar
    CBtransform!(w_col)
    CAtransform!(w_col)
    w_nstar = CItransform!(w_col)
    w_nstar_z = ts_term .* CIxtransform(w_col)

    # Set up the matrix problem
    nz = mtile.model.grid_params.zDim
    g = xi_nstar .- w_nstar_z
    g = [0.0 ; 0.0; g[2:nz-1]]

    xi_col = deepcopy(mtile.tile.columns[mtile.model.grid_params.vars["xi"]])
    # Solve for the coefficients
    if t == 1
        # Calculate the Helmholtz matrix for the first time step
        h_a = calc_Helmholtz_semiimplicit_matrix(mtile.model, Pxi_bar, ts_term)
        xi_col.a .= h_a \ g
    else
        # Use the pre-calculated one
        xi_col.a .= mtile.h_matrix \ g
    end

    # Set xi_n+1
    view(mtile.var_np1,colstart:colend,xi_index) .= CItransform!(xi_col)

    # Set w_n+1
    view(mtile.var_np1,colstart:colend,w_index) .= w_nstar .- (ts_term .* Pxi_bar .* CIxtransform(xi_col))
end

function semiimplicit_adjustment(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    w_index = mtile.model.grid_params.vars["w"]
    xi_index = mtile.model.grid_params.vars["xi"]
    ts = mtile.model.ts

    # Calculate xi_nstar
    xi_nstar = mtile.var_np1[colstart:colend,xi_index]
    wdot_n = view(mtile.impdot_n,colstart:colend,xi_index)
    wdot_nm1 = view(mtile.impdot_nm1,colstart:colend,xi_index)
    wdot_nm2 = view(mtile.impdot_nm2,colstart:colend,xi_index)

    # Calculate w_nstar
    w_nstar = mtile.var_np1[colstart:colend,w_index]
    xidot_n = view(mtile.impdot_n,colstart:colend,w_index)
    xidot_nm1 = view(mtile.impdot_nm1,colstart:colend,w_index)
    xidot_nm2 = view(mtile.impdot_nm2,colstart:colend,w_index)

    # Get the mean speed of sound squared
    Pxi_bar = mtile.ref_state.Pxi_bar

    # Subtract the explicit terms and add the implicit terms
    ts_term = 0.0
    if (t == 1)
        # Use trapezoidal method (AM2) for first step
        ts_term = 0.5 * ts
        w_nstar .= @. w_nstar - (ts * xidot_n) + (ts * 0.5 * xidot_n)
        xi_nstar .= @. xi_nstar - (ts * wdot_n) + (ts * 0.5 * wdot_n)
    elseif (t == 2)
        # Use AI2* for second step and beyond
        ts_term = 1.25 * ts
        w_nstar .= @. w_nstar - (0.5 * ts) * ((3.0 * xidot_n) - xidot_nm1) - (ts * xidot_n) + (ts * 0.75 * xidot_nm1)
        xi_nstar .= @. xi_nstar - (0.5 * ts) * ((3.0 * wdot_n) - wdot_nm1) - (ts * wdot_n) + (ts * 0.75 * wdot_nm1)
    else
        # Use AI2* for second step and beyond
        ts_term = 1.25 * ts
        w_nstar .= @. w_nstar - ((ts / 12.0) * ((23.0 * xidot_n) - (16.0 * xidot_nm1) + (5.0 * xidot_nm2))) - (ts * xidot_n) + (ts * 0.75 * xidot_nm1)
        xi_nstar .= @. xi_nstar - ((ts / 12.0) * ((23.0 * wdot_n) - (16.0 * wdot_nm1) + (5.0 * wdot_nm2))) - (ts * wdot_n) + (ts * 0.75 * wdot_nm1)
    end

    # Set the n-1 and n-2 terms
    xidot_nm2 .= xidot_nm1
    xidot_nm1 .= xidot_n

    wdot_nm2 .= wdot_nm1
    wdot_nm1 .= wdot_n

    # Take the vertical derivative of xi_nstar and multiply by ts term
    xi_col = deepcopy(mtile.tile.columns[mtile.model.grid_params.vars["xi"]])
    xi_col.uMish .= xi_nstar
    CBtransform!(xi_col)
    CAtransform!(xi_col)
    xi_nstar = CItransform!(xi_col)
    xi_nstar_z = ts_term .* Pxi_bar .* CIxtransform(xi_col)

    # Set up the matrix problem
    nz = mtile.model.grid_params.zDim
    g = xi_nstar_z .- w_nstar
    g = [0.0 ; 0.0; g[2:nz-1]]

    w_col = deepcopy(mtile.tile.columns[mtile.model.grid_params.vars["w"]])
    # Solve for the coefficients
    if t == 1
        # Calculate the Helmholtz matrix for the first time step
        h_a = calc_Helmholtz_semiimplicit_matrix(mtile.model, Pxi_bar, ts_term)
        w_col.a .= h_a \ g
    else
        # Use the pre-calculated one
        w_col.a .= mtile.h_matrix \ g
    end

    # Set w_n+1
    view(mtile.var_np1,colstart:colend,w_index) .= CItransform!(w_col)

    # Set xi_n+1
    view(mtile.var_np1,colstart:colend,xi_index) .= xi_nstar .- (ts_term .* CIxtransform(w_col))
end

function semiimplicit_timestep(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    w_index = mtile.model.grid_params.vars["w"]
    xi_index = mtile.model.grid_params.vars["xi"]
    ts = mtile.model.ts

    # Calculate xi_nstar
    xi_nstar = mtile.var_np1[colstart:colend,xi_index]
    wdot_n = view(mtile.impdot_n,colstart:colend,xi_index)
    wdot_nm1 = view(mtile.impdot_nm1,colstart:colend,xi_index)
    wdot_nm2 = view(mtile.impdot_nm2,colstart:colend,xi_index)

    # Calculate w_nstar
    w_nstar = mtile.var_np1[colstart:colend,w_index]
    xidot_n = view(mtile.impdot_n,colstart:colend,w_index)
    xidot_nm1 = view(mtile.impdot_nm1,colstart:colend,w_index)
    xidot_nm2 = view(mtile.impdot_nm2,colstart:colend,w_index)

    # Get the mean speed of sound squared
    Pxi_bar = mtile.ref_state.Pxi_bar

    # Subtract the explicit terms and add the implicit terms
    ts_term = 0.0
    if (t == 1)
        # Use trapezoidal method (AM2) for first step
        ts_term = 0.5 * ts
        w_nstar .= @. w_nstar + (ts * 0.5 * xidot_n)
        xi_nstar .= @. xi_nstar + (ts * 0.5 * wdot_n)
    else
        # Use AI2* for second step and beyond
        ts_term = 1.25 * ts
        w_nstar .= @. w_nstar - (ts * xidot_n) + (ts * 0.75 * xidot_nm1)
        xi_nstar .= @. xi_nstar - (ts * wdot_n) + (ts * 0.75 * wdot_nm1)
    end

    # Set the n-1 and n-2 terms
    xidot_nm2 .= xidot_nm1
    xidot_nm1 .= xidot_n

    wdot_nm2 .= wdot_nm1
    wdot_nm1 .= wdot_n

    # Take the vertical derivative of xi_nstar and multiply by ts term
    xi_col = deepcopy(mtile.tile.columns[mtile.model.grid_params.vars["xi"]])
    xi_col.uMish .= xi_nstar
    CBtransform!(xi_col)
    CAtransform!(xi_col)
    xi_nstar = CItransform!(xi_col)
    xi_nstar_z = ts_term .* Pxi_bar .* CIxtransform(xi_col)

    # Set up the matrix problem
    nz = mtile.model.grid_params.zDim
    g = xi_nstar_z .- w_nstar
    g = [0.0 ; 0.0; g[2:nz-1]]

    w_col = deepcopy(mtile.tile.columns[mtile.model.grid_params.vars["w"]])
    # Solve for the coefficients
    if t == 1
        # Calculate the Helmholtz matrix for the first time step
        h_a = calc_Helmholtz_semiimplicit_matrix(mtile.model, Pxi_bar, ts_term)
        w_col.a .= h_a \ g
    else
        # Use the pre-calculated one
        w_col.a .= mtile.h_matrix \ g
    end

    # Set w_n+1
    view(mtile.var_np1,colstart:colend,w_index) .= CItransform!(w_col)

    # Set xi_n+1
    view(mtile.var_np1,colstart:colend,xi_index) .= xi_nstar .- (ts_term .* CIxtransform(w_col))
end

function explicit_timestep(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    for v in 1:length(mtile.model.grid_params.vars)
        physical = view(mtile.tile.physical,colstart:colend,v,1)
        var_np1 = view(mtile.var_np1,colstart:colend,v)
        expdot_n = view(mtile.expdot_n,colstart:colend,v)
        expdot_nm1 = view(mtile.expdot_nm1,colstart:colend,v)
        expdot_nm2 = view(mtile.expdot_nm2,colstart:colend,v)
        ts = mtile.model.ts

        if (t == 1)
            # Use Euler method and trapezoidal method (AM2) for first step
            var_np1 .= @. physical + (ts * expdot_n)
            expdot_nm1 .= expdot_n
        elseif (t == 2)
            # Use 2nd order A-B method and AI2* for second step
            var_np1 .= @. physical + (0.5 * ts) * ((3.0 * expdot_n) - expdot_nm1)
            expdot_nm2 .= expdot_nm1
            expdot_nm1 .= expdot_n
        else
            # Use AI2*–AB3 implicit-explicit scheme (Durran and Blossey 2012)
            var_np1 .= @. physical + ((ts / 12.0) * ((23.0 * expdot_n) - (16.0 * expdot_nm1) + (5.0 * expdot_nm2)))
            expdot_nm2 .= expdot_nm1
            expdot_nm1 .= expdot_n
        end
    end
end

function explicit_increment(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    for v in 1:length(mtile.model.grid_params.vars)
        var_np1 = view(mtile.var_np1,colstart:colend,v)
        expdot_incr = view(mtile.expdot_incr,colstart:colend,v)
        expdot_n = view(mtile.expdot_n,colstart:colend,v)
        expdot_nm1 = view(mtile.expdot_nm1,colstart:colend,v)
        ts = mtile.model.ts

        if (t == 1)
            # Use Euler method and trapezoidal method (AM2) for first step
            var_np1 .= @. var_np1 + (ts * expdot_incr)
            expdot_n .= expdot_n .+ expdot_incr
            expdot_nm1 .= expdot_n
        elseif (t == 2)
            # Use 2nd order A-B method and AI2* for second step
            var_np1 .= @. var_np1 + ((0.5 * ts) * (3.0 * expdot_incr))
            expdot_n .= expdot_n .+ expdot_incr
            expdot_nm1 .= expdot_n
        else
            # Use AI2*–AB3 implicit-explicit scheme (Durran and Blossey 2012)
            var_np1 .= @. var_np1 + ((ts / 12.0) * (23.0 * expdot_incr))
            expdot_n .= expdot_n .+ expdot_incr
            expdot_nm1 .= expdot_n
        end
    end
end

function calcTendency(mtile::ModelTile)

    # Set the current time
    mtile.tile.physical .= mtile.var_np1
    
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

function calc_Helmholtz_semiimplicit_matrix_xi(model::ModelParameters, Pxi_bar::Float64, ts_term::Float64)

    # Calculate the Helmholtz matrix
    nz = model.grid_params.zDim
    dct = Chebyshev.dct_matrix(nz)
    column_length = model.grid_params.zmax - model.grid_params.zmin
    dct2 = Chebyshev.dct_2nd_derivative(nz, column_length)
    dct1 = Chebyshev.dct_1st_derivative(nz, column_length)
    h = (-ts_term .* ts_term .* Pxi_bar) .* dct2 .+ dct
    bc1 = (-ts_term .* ts_term .* Pxi_bar) .* dct1[1,:]
    bc2 = (-ts_term .* ts_term .* Pxi_bar) .* dct1[nz,:]
    h_a = [bc1[:]'; bc2[:]'; h[2:nz-1,:]]
    return factorize(h_a)
end

function calc_Helmholtz_semiimplicit_matrix(model::ModelParameters, Pxi_bar::Float64, ts_term::Float64)

    # Calculate the Helmholtz matrix
    nz = model.grid_params.zDim
    dct = Chebyshev.dct_matrix(nz)
    column_length = model.grid_params.zmax - model.grid_params.zmin
    dct2 = Chebyshev.dct_2nd_derivative(nz, column_length)
    dct1 = Chebyshev.dct_1st_derivative(nz, column_length)
    h = (ts_term .* ts_term .* Pxi_bar) .* dct2 .- dct
    bc1 = (ts_term .* ts_term .* Pxi_bar) .* dct[1,:]
    bc2 = (ts_term .* ts_term .* Pxi_bar) .* dct[nz,:]
    h_a = [bc1[:]'; bc2[:]'; h[2:nz-1,:]]
    return factorize(h_a)
end
