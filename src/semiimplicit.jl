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

"""
    ModelTile

Fundamental computational unit holding model state, tendencies, reference state,
and spectral transform infrastructure for a single tile in the domain decomposition.
"""
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
    patchMap::SparseMatrixCSC{Float64, Int64}
    haloSendMap::SparseMatrixCSC{Float64, Int64}
    haloReceiveMap::SparseMatrixCSC{Float64, Int64}
    haloReceiveBuffer::Array{Float64}
    splineBuffer::Array{Float64}
    h_matrix::Factorization
    diffusion_matrix::Factorization
end

"""
    createModelTile(patch, tile, model, haloReceiveMap)

Create and initialize a [`ModelTile`](@ref) with allocated state arrays, reference state,
patch-to-tile mappings, halo exchange buffers, and pre-computed Helmholtz matrices.

# Arguments
- `patch::AbstractGrid`: the full domain grid (SpringsteelGrid).
- `tile::AbstractGrid`: the tile grid for this worker (SpringsteelGrid).
- `model::ModelParameters`: model configuration.
- `haloReceiveMap::SparseMatrixCSC{Float64, Int64}`: sparse map for halo receive locations.
"""
function createModelTile(patch::AbstractGrid, tile::AbstractGrid, model::ModelParameters,
        haloReceiveMap::SparseMatrixCSC{Float64, Int64})

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
            ref_state = exact_reference_state(model, z_values)
        else
            ref_state = calculate_reference_state(model, z_values, model.grid_params.zDim)
        end
    end

    # Set up the map between the tile and the patch (returns SparseMatrixCSC directly)
    patchMap = calcPatchMap(patch, tile)

    # Set up the map between the tile and its neighbor (returns SparseMatrixCSC directly)
    haloSendMap = calcHaloMap(patch, tile)

    # Set up some buffers to avoid excessive allocations
    haloReceiveBuffer = zeros(Float64, nnz(haloReceiveMap))
    splineBuffer = allocateSplineBuffer(tile)

    # Pre-calculate the Helmholtz matrices for semi-implicit adjustment
    # Declare a basic factorization for the structure if semiimplicit integration is not used
    h_matrix = factorize([1 2; 2 1])
    diffusion_matrix = factorize([1 2; 2 1])
    if model.options[:semiimplicit]
        diffusion_matrix = calc_Helmholtz_diffusion_matrix(model, 1.25 * model.ts * model.physical_params[:Kvdiff] )
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
        patchMap,
        haloSendMap,
        haloReceiveMap,
        haloReceiveBuffer,
        splineBuffer,
        h_matrix,
        diffusion_matrix)
    return mtile
end

"""Extract nonzero row/col indices from a sparse map for SharedArray indexing."""
function sparse_indices(map::SparseMatrixCSC)
    rows, cols, _ = findnz(map)
    return CartesianIndex.(rows, cols)
end

"""Extract spectral values at sparse map locations."""
function extract_at_map(spectral::AbstractArray, map::SparseMatrixCSC)
    idx = sparse_indices(map)
    return spectral[idx]
end

"""Write spectral values at sparse map locations."""
function write_at_map!(spectral::AbstractArray, map::SparseMatrixCSC, values)
    idx = sparse_indices(map)
    spectral[idx] .= values
end

"""Accumulate (add) spectral values at sparse map locations."""
function accumulate_at_map!(spectral::AbstractArray, map::SparseMatrixCSC, values)
    idx = sparse_indices(map)
    spectral[idx] .+= values
end

"""
    initialize_model(model, workerids)

Set up the distributed model infrastructure by creating the grid patch, distributing
tiles across workers, initializing halo exchange maps, and preparing for time integration.
Returns the initialized patch grid.
"""
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
    # Print the tile information — calcTileSizes now returns Vector{SpringsteelGrid}
    tiles = calcTileSizes(patch, num_workers)
    for w in workerids
        t = tiles[w-1]
        println("Worker $w: $(t.params.iDim) gridpoints in $(t.params.num_cells) cells from $(t.params.iMin) to $(t.params.iMax) starting at index $(t.params.spectralIndexL)")
    end

    map(wait, [save_at(w, :model, model) for w in workerids])
    map(wait, [save_at(w, :workerids, workerids) for w in workerids])
    map(wait, [save_at(w, :num_workers, num_workers) for w in workerids])
    # Create patch on each worker for calcPatchMap/calcHaloMap
    map(wait, [save_at(w, :patch, :(createGrid(model.grid_params))) for w in workerids])

    # Send pre-built tiles directly to workers (serialization works in Julia 1.12)
    println("Initializing tiles on workers")
    map(wait, [save_at(w, :tile, tiles[w-1]) for w in workerids])

    # Create the model tiles
    println("Initializing modelTiles on workers")

    # First tile receives a trivial sparse halo map from master to simplify later loops
    firstMap = sparse([1], [1], [1.0], size(patch.spectral, 1), size(patch.spectral, 2))

    # Precalculate indices and allocate buffers for shared and border transfers
    wait(save_at(workerids[1], :mtile, :(createModelTile(patch,tile,model,$(firstMap)))))
    for w in workerids[1:length(workerids)-1]
        send_index = w + 1
        sendMap = get_val_from(w, :(mtile.haloSendMap))
        wait(save_at(send_index, :mtile, :(createModelTile(patch,tile,model,$(sendMap)))))
    end

    # Delete the patch from the workers since the relevant info is already in the modelTile
    # Don't delete from the first worker in case they are also the master
    map(wait, [remove_from(w, :patch) for w in workerids[2:length(workerids)]])

    println("Ready for time integration!")
    flush(stdout)
    return patch
end

"""
    run_model(patch, model, workerids)

Main time integration loop. Establishes `RemoteChannel` connections between workers,
creates the shared spectral array, and drives the model forward through all timesteps.
"""
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
    haloReceiveMap = get_val_from(last(workerids), :(mtile.haloSendMap))
    haloReceiveBuffer = zeros(Float64, nnz(haloReceiveMap))

    # Create a shared array for the spectral sum
    sharedSpectral = SharedArray{Float64,2}((size(patch.spectral,1),size(patch.spectral,2)))
    results = Array{Future}(undef,num_workers+1)

    # Initialize at time zero
    sharedSpectral[:] .= patch.spectral[:]
    for w in workerids
        save_at(w, :sharedSpectral, sharedSpectral)
    end
    map(wait, [get_from(w, :(splineTransform!(sharedSpectral, mtile.tile))) for w in workerids])

    # Output initial time
    patch.spectral .= sharedSpectral
    gridTransform!(patch)
    @async write_output(patch, model, 0.0)
    flush(stdout)
    # Check for NaNs and quit if found
    checkCFL(patch)

    # Loop through the model timesteps
    @time model_loop(patch, model, workerids, sharedSpectral, haloInit, haloReceive,
        haloInitBuffer, haloReceiveBuffer, haloReceiveMap)

    # Integration complete! Finalize the patch
    patch.spectral .= sharedSpectral
    gridTransform!(patch)
    println("Done with time integration")
    return true

end

"""
    model_loop(patch, model, workerids, sharedSpectral, haloInit, haloReceive, haloInitBuffer, haloReceiveBuffer, haloReceiveMap)

Inner time stepping loop that advances all tiles each timestep, performs halo exchanges
via `RemoteChannel`s, accumulates spectral contributions, and writes periodic output.
"""
function model_loop(patch::AbstractGrid, model::ModelParameters, workerids::Vector{Int64},
        sharedSpectral::SharedArray{Float64}, haloInit::RemoteChannel, haloReceive::RemoteChannel,
        haloInitBuffer::Array{Float64}, haloReceiveBuffer::Array{Float64}, haloReceiveMap::SparseMatrixCSC{Float64, Int64})

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
        accumulate_at_map!(sharedSpectral, haloReceiveMap, haloReceiveBuffer)

        # Reset the shared spectral patch to the tiles
        map(wait, [get_from(w, :(splineTransform!(sharedSpectral, mtile.tile))) for w in workerids])

        # Output if on specified time interval
        if mod(t,output_int) == 0
            patch.spectral .= sharedSpectral
            gridTransform!(patch)
            @async write_output(patch, model, (t*model.ts))
            checkCFL(patch)
        end

        # Done with this timestep
        flush(stdout)
    end
    return nothing
end

"""
    advanceTimestep(mtile, sharedSpectral, haloSend, haloReceive, t)

Advance one tile by one timestep: transform to physical space, advance all columns,
compute spectral tendencies, and exchange halo data with neighboring tiles.
"""
function advanceTimestep(mtile::ModelTile, sharedSpectral::SharedArray{Float64},
        haloSend::RemoteChannel, haloReceive::RemoteChannel, t::Int64)

    # Transform to local physical tile
    tileTransform!(sharedSpectral, mtile.tile, mtile.tile.physical, mtile.tile.spectral)

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
    put!(haloSend, extract_at_map(mtile.tile.spectral, mtile.haloSendMap))

    # Set the sharedArray this tile is responsible for
    write_at_map!(sharedSpectral, mtile.patchMap, extract_at_map(mtile.tile.spectral, mtile.patchMap))

    # Get halo from previous tile
    mtile.haloReceiveBuffer .= take!(haloReceive)

    # Add it to the sharedArray
    accumulate_at_map!(sharedSpectral, mtile.haloReceiveMap, mtile.haloReceiveBuffer)

    return nothing
end

"""
    advance_column(mtile, c, t)

Advance a single column `c` by dispatching to the configured physical model equation set.
A column index of -1 indicates an R or RL grid where all points are treated as one column.
"""
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

"""
    finalize_model(grid, model)

Write final model output at the end of the integration period.
"""
function finalize_model(grid::AbstractGrid, model::ModelParameters)
    
    write_output(grid, model, model.integration_time)
    println("Model complete!")
end

"""
    physical_model(mtile, colstart, colend, t)

Dispatch to the appropriate equation set by looking up the function named
by `mtile.model.equation_set` in the `Scythe` module and calling it on the column range.
"""
function physical_model(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)
        
    equation_set = Symbol(mtile.model.equation_set)
    equation_call = getfield(Scythe, equation_set)
    equation_call(mtile, colstart, colend, t)
    return
end

"""
    semiimplicit_timestep_old(mtile, colstart, colend, t)

Deprecated semi-implicit timestep variant that solves a Helmholtz equation for xi
using a direct matrix solve each timestep. Superseded by [`semiimplicit_timestep`](@ref).
"""
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
    w_col = mtile.tile.kbasis.data[mtile.model.grid_params.vars["w"]]
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
    xi_col = mtile.tile.kbasis.data[mtile.model.grid_params.vars["xi"]]
    xi_col.a .= xi_a
    view(mtile.var_np1,colstart:colend,xi_index) .= CItransform!(xi_col)

    # Set w_n+1
    view(mtile.var_np1,colstart:colend,w_index) .= w_nstar .- (ts_term .* Pxi_bar .* CIxtransform(xi_col))
end

"""
    semiimplicit_adjustment_xi(mtile, colstart, colend, t)

Semi-implicit adjustment that solves a Chebyshev-collocation Helmholtz problem for xi,
using AB3 explicit extrapolation and AI2* implicit treatment of acoustic modes.
"""
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
    w_col = deepcopy(mtile.tile.kbasis.data[mtile.model.grid_params.vars["w"]])
    w_col.uMish .= w_nstar
    CBtransform!(w_col)
    CAtransform!(w_col)
    w_nstar = CItransform!(w_col)
    w_nstar_z = ts_term .* CIxtransform(w_col)

    # Set up the matrix problem
    nz = mtile.model.grid_params.zDim
    g = xi_nstar .- w_nstar_z
    g = [0.0 ; 0.0; g[2:nz-1]]

    xi_col = deepcopy(mtile.tile.kbasis.data[mtile.model.grid_params.vars["xi"]])
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

"""
    semiimplicit_adjustment(mtile, colstart, colend, t)

Semi-implicit adjustment that solves a Chebyshev-collocation Helmholtz problem for w,
using AB3 explicit extrapolation and AI2* implicit treatment of acoustic modes.
"""
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
    xi_col = deepcopy(mtile.tile.kbasis.data[mtile.model.grid_params.vars["xi"]])
    xi_col.uMish .= xi_nstar
    CBtransform!(xi_col)
    CAtransform!(xi_col)
    xi_nstar = CItransform!(xi_col)
    xi_nstar_z = ts_term .* Pxi_bar .* CIxtransform(xi_col)

    # Set up the matrix problem
    nz = mtile.model.grid_params.zDim
    g = xi_nstar_z .- w_nstar
    g = [0.0 ; 0.0; g[2:nz-1]]

    w_col = deepcopy(mtile.tile.kbasis.data[mtile.model.grid_params.vars["w"]])
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

"""
    semiimplicit_timestep(mtile, colstart, colend, t)

Combined explicit-implicit split timestep for acoustic modes. Applies AI2*-AB3
time integration with a Helmholtz solve for w to handle the implicit part.
"""
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
    xi_col = deepcopy(mtile.tile.kbasis.data[mtile.model.grid_params.vars["xi"]])
    xi_col.uMish .= xi_nstar
    CBtransform!(xi_col)
    CAtransform!(xi_col)
    xi_nstar = CItransform!(xi_col)
    xi_nstar_z = ts_term .* Pxi_bar .* CIxtransform(xi_col)

    # Set up the matrix problem
    nz = mtile.model.grid_params.zDim
    g = xi_nstar_z .- w_nstar
    g = [0.0 ; 0.0; g[2:nz-1]]

    w_col = deepcopy(mtile.tile.kbasis.data[mtile.model.grid_params.vars["w"]])
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

"""
    diffusion_timestep(mtile, colstart, colend, t)

Implicit vertical diffusion timestep for thermodynamic and moisture variables (s, mu,
mu_c, mu_r, mu_sat) using a pre-factored Helmholtz matrix and AI2* time integration.
"""
function diffusion_timestep(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    s_index = mtile.model.grid_params.vars["s"]
    s = view(mtile.var_np1,colstart:colend,s_index)

    mu_index = mtile.model.grid_params.vars["mu"]
    mu = view(mtile.var_np1,colstart:colend,mu_index)

    mu_c_index = mtile.model.grid_params.vars["mu_c"]
    mu_c = view(mtile.var_np1,colstart:colend,mu_c_index)

    mu_r_index = mtile.model.grid_params.vars["mu_r"]
    mu_r = view(mtile.var_np1,colstart:colend,mu_r_index)
    
    mu_sat_index = mtile.model.grid_params.vars["mu_sat"]
    mu_sat = view(mtile.var_np1,colstart:colend,mu_sat_index)

    ts = mtile.model.ts

    # Calculate s_nstar
    s_nstar = mtile.var_np1[colstart:colend,s_index]
    sdot_n = view(mtile.impdot_n,colstart:colend,s_index)
    sdot_nm1 = view(mtile.impdot_nm1,colstart:colend,s_index)
    sdot_nm2 = view(mtile.impdot_nm2,colstart:colend,s_index)

    # Calculate mu_nstar
    mu_nstar = mtile.var_np1[colstart:colend,mu_index]
    mudot_n = view(mtile.impdot_n,colstart:colend,mu_index)
    mudot_nm1 = view(mtile.impdot_nm1,colstart:colend,mu_index)
    mudot_nm2 = view(mtile.impdot_nm2,colstart:colend,mu_index)

    # Calculate mu_c_nstar
    mu_c_nstar = mtile.var_np1[colstart:colend,mu_c_index]
    mu_cdot_n = view(mtile.impdot_n,colstart:colend,mu_c_index)
    mu_cdot_nm1 = view(mtile.impdot_nm1,colstart:colend,mu_c_index)
    mu_cdot_nm2 = view(mtile.impdot_nm2,colstart:colend,mu_c_index)

    # Calculate mu_r_nstar
    mu_r_nstar = mtile.var_np1[colstart:colend,mu_r_index]
    mu_rdot_n = view(mtile.impdot_n,colstart:colend,mu_r_index)
    mu_rdot_nm1 = view(mtile.impdot_nm1,colstart:colend,mu_r_index)
    mu_rdot_nm2 = view(mtile.impdot_nm2,colstart:colend,mu_r_index)

    # Calculate mu_sat_nstar
    mu_sat_nstar = mtile.var_np1[colstart:colend,mu_sat_index]
    mu_sat_dot_n = view(mtile.impdot_n,colstart:colend,mu_sat_index)
    mu_sat_dot_nm1 = view(mtile.impdot_nm1,colstart:colend,mu_sat_index)
    mu_sat_dot_nm2 = view(mtile.impdot_nm2,colstart:colend,mu_sat_index)

    # Add the implicit terms
    ts_term = 0.0
    if (t == 1)
        # Use trapezoidal method (AM2) for first step
        ts_term = 0.5 * ts * mtile.model.physical_params[:Kvdiff]
        s_nstar .= @. s_nstar + (ts * 0.5 * sdot_n)
        mu_nstar .= @. mu_nstar + (ts * 0.5 * mudot_n)
        mu_c_nstar .= @. mu_c_nstar + (ts * 0.5 * mu_cdot_n)
        mu_r_nstar .= @. mu_r_nstar + (ts * 0.5 * mu_rdot_n)
        mu_sat_nstar .= @. mu_sat_nstar + (ts * 0.5 * mu_sat_dot_n)
    else
        # Use AI2* for second step and beyond
        ts_term = 1.25 * ts * mtile.model.physical_params[:Kvdiff]
        s_nstar .= @. s_nstar - (ts * sdot_n) + (ts * 0.75 * sdot_nm1)
        mu_nstar .= @. mu_nstar - (ts * mudot_n) + (ts * 0.75 * mudot_nm1)
        mu_c_nstar .= @. mu_c_nstar - (ts * mu_cdot_n) + (ts * 0.75 * mu_cdot_nm1)
        mu_r_nstar .= @. mu_r_nstar - (ts * mu_rdot_n) + (ts * 0.75 * mu_rdot_nm1)
        mu_sat_nstar .= @. mu_sat_nstar - (ts * mu_sat_dot_n) + (ts * 0.75 * mu_sat_dot_nm1)
    end

    # Set the n-1 and n-2 terms
    sdot_nm2 .= sdot_nm1
    sdot_nm1 .= sdot_n
    mudot_nm2 .= mudot_nm1
    mudot_nm1 .= mudot_n
    mu_cdot_nm2 .= mu_cdot_nm1
    mu_cdot_nm1 .= mu_cdot_n
    mu_rdot_nm2 .= mu_rdot_nm1
    mu_rdot_nm1 .= mu_rdot_n
    mu_sat_dot_nm2 .= mu_sat_dot_nm1
    mu_sat_dot_nm1 .= mu_sat_dot_n

    # Set up the matrix problem
    nz = mtile.model.grid_params.zDim
    col = deepcopy(mtile.tile.kbasis.data[mtile.model.grid_params.vars["s"]])

    # Solve for the coefficients
    h_a = mtile.diffusion_matrix
    if t == 1
        # Calculate the Helmholtz matrix for the first time step
        h_a = calc_Helmholtz_diffusion_matrix(mtile.model, ts_term)
    end

    # Set s_n+1
    g = s_nstar
    g = [0.0 ; 0.0; g[2:nz-1]]
    col.a .= h_a \ g
    view(mtile.var_np1,colstart:colend,s_index) .= CItransform!(col)

    # Set mu_n+1
    g = mu_nstar
    g = [0.0 ; 0.0; g[2:nz-1]]
    col.a .= h_a \ g
    view(mtile.var_np1,colstart:colend,mu_index) .= CItransform!(col)

    # Set mu_c_n+1
    g = mu_c_nstar
    g = [0.0 ; 0.0; g[2:nz-1]]
    col.a .= h_a \ g
    view(mtile.var_np1,colstart:colend,mu_c_index) .= CItransform!(col)

    # Set mu_r_n+1
    g = mu_r_nstar
    g = [0.0 ; 0.0; g[2:nz-1]]
    col.a .= h_a \ g
    view(mtile.var_np1,colstart:colend,mu_r_index) .= CItransform!(col)

    # Set mu_sat_n+1
    g = mu_sat_nstar
    g = [0.0 ; 0.0; g[2:nz-1]]
    col.a .= h_a \ g
    view(mtile.var_np1,colstart:colend,mu_sat_index) .= CItransform!(col)

end

"""
    explicit_timestep(mtile, colstart, colend, t)

Advance all variables one timestep using AB3 explicit time stepping (Euler for t=1,
second-order AB for t=2, third-order AB3 thereafter per Durran and Blossey 2012).
"""
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

"""
    explicit_increment(mtile, colstart, colend, t)

Apply an incremental explicit forcing to the current solution using AB3-consistent
weighting, and accumulate it into the stored explicit tendencies.
"""
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

"""
    calcTendency(mtile)

Transform the updated physical-space state in `var_np1` to spectral space,
storing the result in the tile's spectral array for inter-tile communication.
"""
function calcTendency(mtile::ModelTile)

    # Set the current time
    mtile.tile.physical .= mtile.var_np1
    
    # Transform to spectral space
    spectralTransform!(mtile.tile)    
end

"""
    checkCFL(grid)

Check all physical variables for NaN values, which indicate a likely CFL violation.
Throws an error if any NaN is found.
"""
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

"""
    calc_Helmholtz_semiimplicit_matrix_xi(model, Pxi_bar, ts_term)

Build and factorize the Chebyshev-collocation Helmholtz matrix for the xi-form
semi-implicit solve, with derivative boundary conditions at top and bottom.
"""
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

"""
    calc_Helmholtz_semiimplicit_matrix(model, Pxi_bar, ts_term)

Build and factorize the Chebyshev-collocation Helmholtz matrix for the w-form
semi-implicit solve, with Dirichlet boundary conditions at top and bottom.
"""
function calc_Helmholtz_semiimplicit_matrix(model::ModelParameters, Pxi_bar::Float64, ts_term::Float64)

    # Calculate the Helmholtz matrix
    nz = model.grid_params.zDim
    dct = Chebyshev.dct_matrix(nz)
    column_length = model.grid_params.zmax - model.grid_params.zmin
    dct2 = Chebyshev.dct_2nd_derivative(nz, column_length)
    #dct1 = Chebyshev.dct_1st_derivative(nz, column_length)
    h = (ts_term .* ts_term .* Pxi_bar) .* dct2 .- dct
    bc1 = dct[1,:]
    bc2 = dct[nz,:]
    h_a = [bc1[:]'; bc2[:]'; h[2:nz-1,:]]
    return factorize(h_a)
end

"""
    calc_Helmholtz_diffusion_matrix(model, ts_term)

Build and factorize the Chebyshev-collocation Helmholtz matrix for implicit vertical
diffusion, with Neumann (zero-flux) boundary conditions at top and bottom.
"""
function calc_Helmholtz_diffusion_matrix(model::ModelParameters, ts_term::Float64)

    # Calculate the Helmholtz matrix
    nz = model.grid_params.zDim
    dct = Chebyshev.dct_matrix(nz)
    column_length = model.grid_params.zmax - model.grid_params.zmin
    dct2 = Chebyshev.dct_2nd_derivative(nz, column_length)
    dct1 = Chebyshev.dct_1st_derivative(nz, column_length)
    h = dct .- (ts_term .* dct2)
    bc1 = dct1[1,:]
    bc2 = dct1[nz,:]
    h_a = [bc1[:]'; bc2[:]'; h[2:nz-1,:]]
    return factorize(h_a)
end
