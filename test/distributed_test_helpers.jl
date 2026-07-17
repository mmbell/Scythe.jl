# Helpers for testing the distributed (tiled) model workflow in a single process.
#
# These functions replicate the exact sequence of operations performed by
# model_loop + advanceTimestep (semiimplicit.jl) but without Distributed.jl
# workers or RemoteChannels, enabling deterministic single-process testing.

"""
    advance_tile_columns(mtile, t)

Advance every column of a tile, mirroring the dispatch in `advanceTimestep`:
RZ/RLZ grids advance per vertical column, R/RL grids advance all points as a
single column (`c = -1`).
"""
function advance_tile_columns(mtile, t)
    nc = Springsteel.num_columns(mtile.tile)
    if nc > 0
        for c in 1:nc
            Scythe.advance_column(mtile, c, t)
        end
    else
        Scythe.advance_column(mtile, -1, t)
    end
end

"""
    run_distributed_simulation(model, initial_spectral, num_workers, num_ts)

Simulate the distributed model_loop workflow in a single process using
`num_workers` tiles for `num_ts` timesteps.

Replicates the sequence:
  clear shared → tileTransform → advance_column → calcTendency →
  write_tile_to_shared + halo exchange → splineTransform

Returns `(physical, gridpoints)` where `physical` is the final physical
values array (npts × nvars).
"""
function run_distributed_simulation(model, initial_spectral::AbstractArray,
                                    num_workers::Int, num_ts::Int)
    patch = createGrid(model.grid_params)
    tiles = calcTileSizes(patch, num_workers)

    shared = SharedArray{Float64}(size(initial_spectral))
    shared .= initial_spectral

    # Build ModelTile chain with proper haloReceiveMaps
    # First tile gets trivial haloReceiveMap (same as master → worker1 in model_loop)
    firstMap = sparse([1], [1], [1.0],
                      size(initial_spectral, 1), size(initial_spectral, 2))
    mtiles = Scythe.ModelTile[]
    push!(mtiles, createModelTile(patch, tiles[1], model, firstMap))
    for i in 2:num_workers
        push!(mtiles, createModelTile(patch, tiles[i], model, mtiles[i-1].haloSendMap))
    end

    # Master receives from last tile (same as model_loop)
    masterHaloMap = mtiles[end].haloSendMap

    # Initialize tile-local spline data from shared spectral
    for mtile in mtiles
        splineTransform!(shared, patch, mtile.tile)
    end

    # ── Time integration loop (replicates model_loop) ──
    for t in 1:num_ts

        # Master clears shared for accumulation
        shared .= 0.0

        # Phase 1: tileTransform! on all tiles
        # Reads from tile-local spline data (set by previous splineTransform!),
        # NOT from the zeroed shared array.
        for mtile in mtiles
            tileTransform!(shared, mtile.tile, mtile.tile.physical, mtile.tile.spectral)
        end

        # Phase 2: Advance all tiles (compute tendencies and step forward)
        for mtile in mtiles
            advance_tile_columns(mtile, t)
            Scythe.calcTendency(mtile)
        end

        # Phase 3: Write inner regions and exchange halos
        halos = [Scythe.extract_halo_values(mtile.tile) for mtile in mtiles]
        for mtile in mtiles
            Scythe.write_tile_to_shared!(shared, mtile.tile, mtile.patch_b_iDim)
        end

        # Halo chain: tile i>1 receives tile (i-1)'s halo
        # (Tile 1 receives empty halo from master — trivial map, zeros → no-op)
        for i in 2:length(mtiles)
            Scythe.accumulate_at_map!(shared, mtiles[i].haloReceiveMap, halos[i-1])
        end
        # Master accumulates last tile's halo
        Scythe.accumulate_at_map!(shared, masterHaloMap, halos[end])

        # Phase 4: splineTransform! updates tile-local data for next timestep
        for mtile in mtiles
            splineTransform!(shared, patch, mtile.tile)
        end
    end

    # Reconstruct physical values from final shared spectral
    result_patch = createGrid(model.grid_params)
    result_patch.spectral .= shared
    gridTransform!(result_patch)
    return result_patch.physical[:, :, 1], getGridpoints(result_patch)
end

"""
    run_single_process_simulation(model, initial_spectral, num_ts)

Run model stepping in single-process mode (tile = patch, no tiling).
This matches the pattern used in test_linear_advection_integration.jl.

Returns `(physical, gridpoints)` where `physical` is the final physical
values array (npts × nvars).
"""
function run_single_process_simulation(model, initial_spectral::AbstractArray, num_ts::Int)
    patch = createGrid(model.grid_params)
    patch.spectral .= initial_spectral
    gridTransform!(patch)

    haloReceiveMap = sparse(Int64[], Int64[], Float64[],
                            size(patch.spectral, 1), size(patch.spectral, 2))
    mtile = createModelTile(patch, patch, model, haloReceiveMap)

    # Horizontal semi-implicit (mc sets): the patch-level sweep between
    # calcTendency and the inverse transform, mirroring model_loop.
    hsi = get(model.options, :horizontal_semiimplicit, false) === true
    hsd = nothing
    u_incr = zeros(size(patch.physical, 1), 2)
    if hsi
        hsd = Scythe.create_horizontal_solve_data(patch, model,
            mtile.mc_ref_diag.Pxi_prof,
            collect(view(Springsteel.ref_rho_t(mtile.ref_state), :, 1)),
            collect(view(Springsteel.ref_rho_d(mtile.ref_state), :, 1)),
            collect(view(Springsteel.ref_total_energy(mtile.ref_state), :, 1) .+
                    view(Springsteel.ref_pressure(mtile.ref_state), :, 1)))
    end

    for t in 1:num_ts
        if hsi && t > 1
            Scythe.horizontal_si_load_increment!(mtile, u_incr, t, 1)
        end
        advance_tile_columns(mtile, t)
        Scythe.calcTendency(mtile)
        if hsi
            u_incr .= Scythe.horizontal_si_correct!(patch.spectral, patch, model, hsd, t)
        end
        gridTransform!(mtile.tile)
    end

    return copy(mtile.tile.physical[:, :, 1]), getGridpoints(patch)
end
