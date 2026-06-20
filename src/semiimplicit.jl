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
    patch_b_iDim::Int64
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
        z_values = tilepoints[1:model.grid_params.kDim,ndims(tilepoints)]
        ref_column = reference_column(tile, model.grid_params)

        if (model.options[:exact_reference_state])
            ref_state = exact_reference_state(model, z_values, ref_column)
        else
            ref_state = calculate_reference_state(model, z_values, ref_column)
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
        diffusion_matrix = calc_Helmholtz_diffusion_matrix(tile, model, 1.25 * model.ts * model.physical_params[:Kvdiff] )
        h_matrix = calc_Helmholtz_semiimplicit_matrix(tile, model, ref_state.Pxi_bar, 1.25 * model.ts)
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
        patch.params.b_iDim,
        h_matrix,
        diffusion_matrix)
    return mtile
end

"""Extract nonzero row/col indices from a sparse map for SharedArray indexing."""
function sparse_indices(map::SparseMatrixCSC)
    rows, cols, _ = findnz(map)
    return CartesianIndex.(rows, cols)
end

"""Extract nonzero values from a sparse matrix as a dense vector."""
function sparse_values(map::SparseMatrixCSC)
    _, _, vals = findnz(map)
    return vals
end

"""
    extract_halo_values(tile)

Extract the 3-row halo (right boundary) from each wavenumber block of the tile's spectral
array as a dense vector. The ordering matches the structure of `calcHaloMap` so the result
can be directly accumulated into the shared spectral via `accumulate_at_map!`.
"""
function extract_halo_values(tile::AbstractGrid)
    b_iDim = tile.params.b_iDim
    nvars = size(tile.spectral, 2)

    if size(tile.spectral, 1) > b_iDim
        # RL/SL grid: extract from each wavenumber block
        kDim = tile.params.iDim + tile.params.patchOffsetL
        nblocks = 1 + 2 * kDim
        result = zeros(Float64, 3 * nvars * nblocks)
        pos = 1
        for v in 1:nvars
            # k=0 block: last 3 rows
            result[pos:pos+2] .= tile.spectral[b_iDim-2:b_iDim, v]
            pos += 3
            for k in 1:kDim
                p = k * 2
                # Real part: last 3 rows of block
                te = (p - 1) * b_iDim + b_iDim
                result[pos:pos+2] .= tile.spectral[te-2:te, v]
                pos += 3
                # Imaginary part: last 3 rows of block
                te = p * b_iDim + b_iDim
                result[pos:pos+2] .= tile.spectral[te-2:te, v]
                pos += 3
            end
        end
        return result
    else
        # Cartesian 1D: just the last 3 rows
        result = zeros(Float64, 3 * nvars)
        for v in 1:nvars
            result[(v-1)*3+1:v*3] .= tile.spectral[b_iDim-2:b_iDim, v]
        end
        return result
    end
end

"""
    extract_halo_values(tile::Union{RZ_Grid, RiRk_Grid})

RZ and RiRk grids store `b_kDim` consecutive spline blocks (one per vertical
mode — Chebyshev for RZ, B-spline for RiRk), so the 3-row right halo is
extracted from the end of each block. Ordering matches the column-major sparse
indices of the `calcHaloMap`. The spectral layout is identical for both, so the
same extraction applies.
"""
function extract_halo_values(tile::Union{RZ_Grid, RiRk_Grid})
    b_iDim = tile.params.b_iDim
    b_kDim = tile.params.b_kDim
    nvars = size(tile.spectral, 2)
    result = zeros(Float64, 3 * nvars * b_kDim)
    pos = 1
    for v in 1:nvars
        for z in 1:b_kDim
            te = z * b_iDim   # last row of block z
            result[pos:pos+2] .= tile.spectral[te-2:te, v]
            pos += 3
        end
    end
    return result
end

"""Accumulate (add) spectral values at sparse map locations in a patch-sized array."""
function accumulate_at_map!(spectral::AbstractArray, map::SparseMatrixCSC, values)
    idx = sparse_indices(map)
    spectral[idx] .+= values
end

"""
    write_tile_to_shared!(sharedSpectral, tile, b_iDim_patch)

Copy the tile's inner (non-halo) spectral coefficients to the correct positions in the
patch-level shared spectral array. The tile→patch index mapping accounts for different
spectral strides per wavenumber block (b_iDim_tile vs b_iDim_patch).
"""
function write_tile_to_shared!(sharedSpectral::SharedArray{Float64}, tile::AbstractGrid,
                                b_iDim_patch::Int64)
    siL = tile.params.spectralIndexL
    b_iDim_tile = tile.params.b_iDim
    inner_rows = b_iDim_tile - 4  # inner region excludes 3-row halo

    # k=0 block (spline coefficients): patch rows siL:(siL+inner_rows) ← tile rows 1:(1+inner_rows)
    sharedSpectral[siL:siL+inner_rows, :] .= tile.spectral[1:1+inner_rows, :]

    # k>=1 Fourier wavenumber blocks only exist for RL/SL grids
    # Detect by checking if the tile spectral has more rows than a single spline block
    if size(tile.spectral, 1) > b_iDim_tile
        kDim = tile.params.iDim + tile.params.patchOffsetL
        for k in 1:kDim
            p = k * 2
            # Real part
            pp1 = (p - 1) * b_iDim_patch + siL
            tp1 = (p - 1) * b_iDim_tile + 1
            sharedSpectral[pp1:pp1+inner_rows, :] .= tile.spectral[tp1:tp1+inner_rows, :]
            # Imaginary part
            pp1 = p * b_iDim_patch + siL
            tp1 = p * b_iDim_tile + 1
            sharedSpectral[pp1:pp1+inner_rows, :] .= tile.spectral[tp1:tp1+inner_rows, :]
        end
    end
end

"""
    write_tile_to_shared!(sharedSpectral, tile::Union{RZ_Grid, RiRk_Grid}, b_iDim_patch)

RZ and RiRk grids store `b_kDim` consecutive spline blocks (one per vertical
mode); copy the inner rows of every block to its patch position. The spectral
layout is identical for both.
"""
function write_tile_to_shared!(sharedSpectral::SharedArray{Float64}, tile::Union{RZ_Grid, RiRk_Grid},
                                b_iDim_patch::Int64)
    siL = tile.params.spectralIndexL
    b_iDim_tile = tile.params.b_iDim
    b_kDim = tile.params.b_kDim
    inner_rows = b_iDim_tile - 4  # inner region excludes 3-row halo

    for z in 1:b_kDim
        pp1 = (z - 1) * b_iDim_patch + siL
        tp1 = (z - 1) * b_iDim_tile + 1
        sharedSpectral[pp1:pp1+inner_rows, :] .= tile.spectral[tp1:tp1+inner_rows, :]
    end
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

    # Send tile parameters and create grids on workers to avoid serializing CHOLMOD factors
    println("Initializing tiles on workers")
    map(wait, [save_at(w, :tile_params, tiles[w-1].params) for w in workerids])
    map(wait, [save_at(w, :tile, :(createGrid(tile_params))) for w in workerids])

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

    # Delete tile_params from workers since the relevant info is already in the modelTile
    # Keep patch on all workers — it is needed by the 3-arg splineTransform!
    # Don't delete from the first worker in case they are also the master
    map(wait, [remove_from(w, :tile_params) for w in workerids[2:length(workerids)]])

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
    map(wait, [get_from(w, :(splineTransform!(sharedSpectral, patch, mtile.tile))) for w in workerids])

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
        map(wait, [get_from(w, :(splineTransform!(sharedSpectral, patch, mtile.tile))) for w in workerids])

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

    # Send halo to next tile (extract border spectral values in tile-local coordinates)
    put!(haloSend, extract_halo_values(mtile.tile))

    # Set the sharedArray this tile is responsible for (tile→patch index mapping)
    write_tile_to_shared!(sharedSpectral, mtile.tile, mtile.patch_b_iDim)

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

    # Set column index range
    if c == -1
        # R or RL grid: all points are treated as one column
        colstart = 1
        colend = size(mtile.tile.physical,1)
    else
        # RZ or RLZ grid: use the vertical dimension to stride columns
        gp = mtile.model.grid_params
        vdim = gp.kDim
        colstart = (c-1) * vdim + 1
        colend = colstart + vdim - 1
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
    Btransform!(w_col)
    Atransform!(w_col)
    w_nstar = Itransform!(w_col)
    w_nstar_z = ts_term .* Ixtransform(w_col)

    # Calculate the Helmholtz matrix
    h_a = calc_Helmholtz_semiimplicit_matrix_xi(mtile.tile, mtile.model, Pxi_bar, ts_term)

    # Solve for the xi coefficients (RHS = xi_nstar - w_nstar_z; homogeneous BCs)
    xi_col = mtile.tile.kbasis.data[mtile.model.grid_params.vars["xi"]]
    _vertical_solve!(xi_col, h_a, xi_nstar .- w_nstar_z, mtile.tile)
    view(mtile.var_np1,colstart:colend,xi_index) .= Itransform!(xi_col)

    # Set w_n+1
    view(mtile.var_np1,colstart:colend,w_index) .= w_nstar .- (ts_term .* Pxi_bar .* Ixtransform(xi_col))
end

"""
    semiimplicit_adjustment_xi(mtile, colstart, colend, t)

Semi-implicit adjustment that solves a spectral-collocation Helmholtz problem for xi,
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
    Btransform!(w_col)
    Atransform!(w_col)
    w_nstar = Itransform!(w_col)
    w_nstar_z = ts_term .* Ixtransform(w_col)

    # Set up the matrix problem (RHS = xi_nstar - w_nstar_z; homogeneous BCs)
    rhs = xi_nstar .- w_nstar_z
    xi_col = deepcopy(mtile.tile.kbasis.data[mtile.model.grid_params.vars["xi"]])
    # Solve for the coefficients
    if t == 1
        # Calculate the Helmholtz matrix for the first time step
        h_a = calc_Helmholtz_semiimplicit_matrix(mtile.tile, mtile.model, Pxi_bar, ts_term)
        _vertical_solve!(xi_col, h_a, rhs, mtile.tile)
    else
        # Use the pre-calculated one
        _vertical_solve!(xi_col, mtile.h_matrix, rhs, mtile.tile)
    end

    # Set xi_n+1
    view(mtile.var_np1,colstart:colend,xi_index) .= Itransform!(xi_col)

    # Set w_n+1
    view(mtile.var_np1,colstart:colend,w_index) .= w_nstar .- (ts_term .* Pxi_bar .* Ixtransform(xi_col))
end

"""
    semiimplicit_adjustment(mtile, colstart, colend, t)

Semi-implicit adjustment that solves a spectral-collocation Helmholtz problem for w,
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
    Btransform!(xi_col)
    Atransform!(xi_col)
    xi_nstar = Itransform!(xi_col)
    xi_nstar_z = ts_term .* Pxi_bar .* Ixtransform(xi_col)

    # Set up the matrix problem (RHS = xi_nstar_z - w_nstar; homogeneous BCs)
    rhs = xi_nstar_z .- w_nstar
    w_col = deepcopy(mtile.tile.kbasis.data[mtile.model.grid_params.vars["w"]])
    # Solve for the coefficients
    if t == 1
        # Calculate the Helmholtz matrix for the first time step
        h_a = calc_Helmholtz_semiimplicit_matrix(mtile.tile, mtile.model, Pxi_bar, ts_term)
        _vertical_solve!(w_col, h_a, rhs, mtile.tile)
    else
        # Use the pre-calculated one
        _vertical_solve!(w_col, mtile.h_matrix, rhs, mtile.tile)
    end

    # Set w_n+1
    view(mtile.var_np1,colstart:colend,w_index) .= Itransform!(w_col)

    # Set xi_n+1
    view(mtile.var_np1,colstart:colend,xi_index) .= xi_nstar .- (ts_term .* Ixtransform(w_col))
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
    Btransform!(xi_col)
    Atransform!(xi_col)
    xi_nstar = Itransform!(xi_col)
    xi_nstar_z = ts_term .* Pxi_bar .* Ixtransform(xi_col)

    # Set up the matrix problem (RHS = xi_nstar_z - w_nstar; homogeneous BCs)
    rhs = xi_nstar_z .- w_nstar
    w_col = deepcopy(mtile.tile.kbasis.data[mtile.model.grid_params.vars["w"]])
    # Solve for the coefficients
    if t == 1
        # Calculate the Helmholtz matrix for the first time step
        h_a = calc_Helmholtz_semiimplicit_matrix(mtile.tile, mtile.model, Pxi_bar, ts_term)
        _vertical_solve!(w_col, h_a, rhs, mtile.tile)
    else
        # Use the pre-calculated one
        _vertical_solve!(w_col, mtile.h_matrix, rhs, mtile.tile)
    end

    # Set w_n+1
    view(mtile.var_np1,colstart:colend,w_index) .= Itransform!(w_col)

    # Set xi_n+1
    view(mtile.var_np1,colstart:colend,xi_index) .= xi_nstar .- (ts_term .* Ixtransform(w_col))
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
    nz = mtile.model.grid_params.kDim
    col = deepcopy(mtile.tile.kbasis.data[mtile.model.grid_params.vars["s"]])

    # Solve for the coefficients
    h_a = mtile.diffusion_matrix
    if t == 1
        # Calculate the Helmholtz matrix for the first time step
        h_a = calc_Helmholtz_diffusion_matrix(mtile.tile, mtile.model, ts_term)
    end

    # Set s_n+1 (homogeneous BCs)
    _vertical_solve!(col, h_a, s_nstar, mtile.tile)
    view(mtile.var_np1,colstart:colend,s_index) .= Itransform!(col)

    # Set mu_n+1
    _vertical_solve!(col, h_a, mu_nstar, mtile.tile)
    view(mtile.var_np1,colstart:colend,mu_index) .= Itransform!(col)

    # Set mu_c_n+1
    _vertical_solve!(col, h_a, mu_c_nstar, mtile.tile)
    view(mtile.var_np1,colstart:colend,mu_c_index) .= Itransform!(col)

    # Set mu_r_n+1
    _vertical_solve!(col, h_a, mu_r_nstar, mtile.tile)
    view(mtile.var_np1,colstart:colend,mu_r_index) .= Itransform!(col)

    # Set mu_sat_n+1
    _vertical_solve!(col, h_a, mu_sat_nstar, mtile.tile)
    view(mtile.var_np1,colstart:colend,mu_sat_index) .= Itransform!(col)

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
    _helmholtz_bc_row(bc, M0, M1, M2, row_idx)

Select the appropriate operator matrix row for a boundary condition in the Helmholtz solver.
Returns the raw row vector from the operator matrix corresponding to the BC type;
the caller is responsible for applying any physics-specific coefficients.

Dispatches on `BoundaryConditions` fields: Dirichlet → M0, Neumann → M1, SecondDeriv → M2.
"""
function _helmholtz_bc_row(bc::BoundaryConditions, M0, M1, M2, row_idx)
    if is_periodic(bc)
        error("PeriodicBC not supported in Helmholtz solver")
    elseif bc.robin !== nothing
        error("RobinBC not yet supported in Helmholtz solver")
    elseif is_inhomogeneous(bc)
        error("Inhomogeneous BCs not yet supported in Helmholtz solver")
    elseif bc.u !== nothing       # Dirichlet
        return M0[row_idx, :]
    elseif bc.du !== nothing      # Neumann
        return M1[row_idx, :]
    elseif bc.d2u !== nothing     # Second derivative
        return M2[row_idx, :]
    else                          # Natural (R0)
        return M0[row_idx, :]
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Cubic B-spline vertical (RiRk): Galerkin Helmholtz support
#
# The Chebyshev acoustic solver is a square pseudospectral collocation: the DCT
# has #points == #coefficients == kDim, so operator_matrix(:k,·) is kDim×kDim and
# `h \ g` is a square solve. A cubic B-spline instead has b_kDim = num_cells + 3
# coefficients but kDim = num_cells·mubar mish points, so operator_matrix(:k,·) is
# the rectangular (kDim × b_kDim) evaluation matrix and `h \ g` degenerates to an
# ill-posed least-squares solve.
#
# We restore a square system with a Galerkin (finite-element) discretisation on
# the spline basis. The weak form of (α ∂_zz + β) is  -α ∫ψ'φ' + β ∫ψφ, integrated
# with the spline's own Gauss quadrature (weights W at the mish points). Writing
# M0, M1 for the basis and first-derivative matrices at the mish points, the
# operator is  A = -α M1ᵀW M1 + β M0ᵀW M0  (symmetric, b_kDim×b_kDim) and the load
# vector is  b = M0ᵀW g_mish. Crucially A is built from the *same* mish operators
# the explicit tendencies use, so the implicit and explicit acoustic operators are
# consistent — a requirement of the AI2* split that a node-collocation operator
# (built at different points, and non-symmetric) violates, producing a slow
# acoustic instability. Neumann conditions are natural; Dirichlet conditions
# replace the first/last rows with the boundary-value constraint. Full derivation:
# reference/rirk_vertical_solver.tex.
# ─────────────────────────────────────────────────────────────────────────────

const _RIRK_SOLVE_CACHE = Dict{UInt, NamedTuple}()
const _RIRK_SOLVE_LOCK = ReentrantLock()
# Per-factorization Dirichlet flags (bottom, top), so the load vector can zero the
# matching boundary rows. Keyed by objectid of the factorization object.
const _RIRK_DIRICHLET = Dict{UInt, Tuple{Bool, Bool}}()

_is_dirichlet(bc::BoundaryConditions) = bc.u !== nothing

"""
Per-column cached Galerkin data: the mish basis matrix `M0` and first-derivative
matrix `M1` (both `kDim × b_kDim`), the diagonal physical Gauss-quadrature weights
`W` (length `kDim`), and the boundary-value rows `Nb` (`2 × b_kDim`, evaluated at
`z_b` and `z_t`) used to impose Dirichlet conditions. `M0`/`M1` are exactly the
operators the explicit gridTransform tendencies use, ensuring consistency.
"""
function _rirk_solve_data(kcol::CubicBSpline.Spline1D)
    lock(_RIRK_SOLVE_LOCK) do
        get!(_RIRK_SOLVE_CACHE, objectid(kcol)) do
            sp = kcol.params
            M0 = CubicBSpline.SItransform_matrix(kcol, kcol.mishPoints, 0)
            M1 = CubicBSpline.SItransform_matrix(kcol, kcol.mishPoints, 1)
            _, qw = CubicBSpline._quadrature_rule(sp.mubar, sp.quadrature)
            W  = repeat(qw .* sp.DX, outer = sp.num_cells)   # physical weights, length kDim
            Nb = CubicBSpline.SItransform_matrix(kcol, [sp.xmin, sp.xmax], 0)
            (M0 = M0, M1 = M1, W = W, Nb = Nb)
        end
    end
end

# Galerkin assembly of (α ∂_zz + β) on the spline basis; see the block comment.
function _assemble_spline_matrix(d, α::Float64, β::Float64,
        bc_bottom::BoundaryConditions, bc_top::BoundaryConditions)
    Mass  = d.M0' * (d.W .* d.M0)
    Stiff = d.M1' * (d.W .* d.M1)
    A = (-α) .* Stiff .+ β .* Mass
    db = _is_dirichlet(bc_bottom); dt = _is_dirichlet(bc_top)
    if db; A[1,   :] .= d.Nb[1, :]; end
    if dt; A[end, :] .= d.Nb[2, :]; end
    F = factorize(A)
    lock(_RIRK_SOLVE_LOCK) do
        _RIRK_DIRICHLET[objectid(F)] = (db, dt)
    end
    return F
end

"""
    _assemble_vertical_matrix(grid, model, α, β, bc_scale, bc_bottom, bc_top)

Assemble and factorize the vertical operator `α ∂_zz + β` for the semi-implicit
solves. For a Chebyshev k-basis this is the square `kDim × kDim` pseudospectral
collocation (`α M2 + β M0`, interior rows `2:nz-1`, BC rows scaled by `bc_scale`);
for a cubic B-spline k-basis it is the symmetric `b_kDim × b_kDim` Galerkin form.
"""
function _assemble_vertical_matrix(grid::AbstractGrid, model::ModelParameters,
        α::Float64, β::Float64, bc_scale::Float64,
        bc_bottom::BoundaryConditions, bc_top::BoundaryConditions)
    kcol = grid.kbasis.data[1]
    if kcol isa CubicBSpline.Spline1D
        return _assemble_spline_matrix(_rirk_solve_data(kcol), α, β, bc_bottom, bc_top)
    else
        nz = model.grid_params.kDim
        M0 = operator_matrix(grid, :k, 0)
        M1 = operator_matrix(grid, :k, 1)
        M2 = operator_matrix(grid, :k, 2)
        h = α .* M2 .+ β .* M0
        bc1 = bc_scale .* _helmholtz_bc_row(bc_bottom, M0, M1, M2, 1)
        bc2 = bc_scale .* _helmholtz_bc_row(bc_top, M0, M1, M2, nz)
        return factorize([bc1[:]'; bc2[:]'; h[2:nz-1, :]])
    end
end

"""
    _vertical_solve!(col, h_a, rhs_mish, grid)

Solve the factorized vertical system `h_a` for the spectral coefficients `col.a`,
given the right-hand side `rhs_mish` sampled at the `kDim` physical mish points
(boundary rows are homogeneous). For a Chebyshev k-basis the interior mish values
are used directly; for a cubic B-spline k-basis the Galerkin load vector
`M0ᵀW rhs_mish` is formed and the Dirichlet boundary rows (if any) are zeroed.
"""
function _vertical_solve!(col, h_a, rhs_mish::AbstractVector, grid::AbstractGrid)
    kcol = grid.kbasis.data[1]
    if kcol isa CubicBSpline.Spline1D
        d = _rirk_solve_data(kcol)
        b = d.M0' * (d.W .* rhs_mish)
        db, dt = get(_RIRK_DIRICHLET, objectid(h_a), (false, false))
        if db; b[1]   = 0.0; end
        if dt; b[end] = 0.0; end
        col.a .= h_a \ b
    else
        nz = length(rhs_mish)
        col.a .= h_a \ vcat(0.0, 0.0, rhs_mish[2:nz-1])
    end
    return col
end

"""
    calc_Helmholtz_semiimplicit_matrix_xi(grid, model, Pxi_bar, ts_term; bc_bottom, bc_top)

Build and factorize the spectral-collocation Helmholtz matrix for the xi-form
semi-implicit solve. Boundary condition type is configurable via keyword arguments.

# Arguments
- `grid::AbstractGrid`: the tile grid providing basis objects for `operator_matrix`.
- `model::ModelParameters`: model configuration providing grid parameters.
- `Pxi_bar::Float64`: domain-mean speed of sound squared.
- `ts_term::Float64`: time-stepping coefficient.
- `bc_bottom::BoundaryConditions`: bottom boundary condition (default: `NeumannBC()`).
- `bc_top::BoundaryConditions`: top boundary condition (default: `NeumannBC()`).
"""
function calc_Helmholtz_semiimplicit_matrix_xi(grid::AbstractGrid, model::ModelParameters, Pxi_bar::Float64, ts_term::Float64;
        bc_bottom::BoundaryConditions=NeumannBC(), bc_top::BoundaryConditions=NeumannBC())

    c = -ts_term * ts_term * Pxi_bar
    return _assemble_vertical_matrix(grid, model, c, 1.0, c, bc_bottom, bc_top)
end

"""
    calc_Helmholtz_semiimplicit_matrix(grid, model, Pxi_bar, ts_term; bc_bottom, bc_top)

Build and factorize the spectral-collocation Helmholtz matrix for the w-form
semi-implicit solve. Boundary condition type is configurable via keyword arguments.

# Arguments
- `grid::AbstractGrid`: the tile grid providing basis objects for `operator_matrix`.
- `model::ModelParameters`: model configuration providing grid parameters.
- `Pxi_bar::Float64`: domain-mean speed of sound squared.
- `ts_term::Float64`: time-stepping coefficient.
- `bc_bottom::BoundaryConditions`: bottom boundary condition (default: `DirichletBC()`).
- `bc_top::BoundaryConditions`: top boundary condition (default: `DirichletBC()`).
"""
function calc_Helmholtz_semiimplicit_matrix(grid::AbstractGrid, model::ModelParameters, Pxi_bar::Float64, ts_term::Float64;
        bc_bottom::BoundaryConditions=DirichletBC(), bc_top::BoundaryConditions=DirichletBC())

    c = ts_term * ts_term * Pxi_bar
    return _assemble_vertical_matrix(grid, model, c, -1.0, 1.0, bc_bottom, bc_top)
end

"""
    calc_Helmholtz_diffusion_matrix(grid, model, ts_term; bc_bottom, bc_top)

Build and factorize the spectral-collocation Helmholtz matrix for implicit vertical
diffusion. Boundary condition type is configurable via keyword arguments.

# Arguments
- `grid::AbstractGrid`: the tile grid providing basis objects for `operator_matrix`.
- `model::ModelParameters`: model configuration providing grid parameters.
- `ts_term::Float64`: time-stepping coefficient.
- `bc_bottom::BoundaryConditions`: bottom boundary condition (default: `NeumannBC()`).
- `bc_top::BoundaryConditions`: top boundary condition (default: `NeumannBC()`).
"""
function calc_Helmholtz_diffusion_matrix(grid::AbstractGrid, model::ModelParameters, ts_term::Float64;
        bc_bottom::BoundaryConditions=NeumannBC(), bc_top::BoundaryConditions=NeumannBC())

    return _assemble_vertical_matrix(grid, model, -ts_term, 1.0, 1.0, bc_bottom, bc_top)
end
