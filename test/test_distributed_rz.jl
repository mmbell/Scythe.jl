using Test
using Scythe
using Springsteel
using SparseArrays
using SharedArrays
using LinearAlgebra

include("distributed_test_helpers.jl")

# Tests for the distributed (tiled) workflow on RZ grids. The RZ spectral
# layout is b_kDim consecutive blocks of b_iDim spline coefficients (one block
# per Chebyshev mode), so the patch/halo maps and tile transforms must operate
# per block rather than on a flat 1-D slice.

@testset "Distributed RZ tiling" begin

    function make_rz_advection_model(; num_cells=20, kDim=20)
        vars = Dict("h" => 1, "u" => 2, "v" => 3, "w" => 4)
        bc = Dict(v => NaturalBC() for v in keys(vars))
        gp = GridParameters(
            geometry = "RZ",
            num_cells = num_cells,
            iMin = 0.0,
            iMax = 100.0,
            kMin = 0.0,
            kMax = 100.0,
            kDim = kDim,
            BCL = bc, BCR = bc, BCB = bc, BCT = bc,
            vars = vars,
        )
        ts = 0.1
        num_ts = 100
        model = ModelParameters(
            ts = ts,
            integration_time = num_ts * ts,
            output_interval = num_ts * ts,
            equation_set = "LinearAdvectionRZ",
            initial_conditions = "",
            grid_params = gp,
            physical_params = Dict(:K => 0.0),
        )
        return model, num_ts
    end

    """Gaussian blob in h advected by uniform (u, w). Returns spectral ICs."""
    function blob_ic_spectral(model; x0=40.0, z0=40.0, sigma=12.0, u0=1.0, w0=0.5)
        patch = createGrid(model.grid_params)
        gridpoints = getGridpoints(patch)
        for i in 1:size(patch.physical, 1)
            x = gridpoints[i, 1]
            z = gridpoints[i, 2]
            patch.physical[i, 1, 1] = exp(-((x - x0)^2 + (z - z0)^2) / (2 * sigma^2))
            patch.physical[i, 2, 1] = u0
            patch.physical[i, 3, 1] = 0.0
            patch.physical[i, 4, 1] = w0
        end
        spectralTransform!(patch)
        return copy(patch.spectral), gridpoints
    end

    function analytical_blob(gridpoints; x0=40.0, z0=40.0, sigma=12.0, dx=0.0, dz=0.0)
        return [exp(-((gridpoints[i, 1] - x0 - dx)^2 + (gridpoints[i, 2] - z0 - dz)^2) /
                    (2 * sigma^2)) for i in 1:size(gridpoints, 1)]
    end

    # ──────────────────────────────────────────────
    # 1. splineTransform!/tileTransform! roundtrip (1 tile = patch size)
    # ──────────────────────────────────────────────
    @testset "Tile transform roundtrip vs gridTransform" begin
        model, _ = make_rz_advection_model()
        ic_spectral, _ = blob_ic_spectral(model)

        # Reference: full grid transform
        ref_patch = createGrid(model.grid_params)
        ref_patch.spectral .= ic_spectral
        gridTransform!(ref_patch)

        # Tiled path: shared spectral → splineTransform! → tileTransform!
        patch = createGrid(model.grid_params)
        tiles = calcTileSizes(patch, 1)
        tile = tiles[1]
        shared = SharedArray{Float64}(size(ic_spectral))
        shared .= ic_spectral
        splineTransform!(shared, patch, tile)
        tileTransform!(shared, tile, tile.physical, tile.spectral)

        for slot in 1:5
            diff = maximum(abs.(tile.physical[:, 1, slot] .- ref_patch.physical[:, 1, slot]))
            @test diff < 1.0e-10
        end
    end

    # ──────────────────────────────────────────────
    # 2. Single-process RZ advection vs analytical
    # ──────────────────────────────────────────────
    @testset "Single-process RZ advection" begin
        model, num_ts = make_rz_advection_model()
        ic_spectral, gridpoints = blob_ic_spectral(model)

        result, _ = run_single_process_simulation(model, ic_spectral, num_ts)
        analytical = analytical_blob(gridpoints; dx=1.0 * num_ts * model.ts,
                                     dz=0.5 * num_ts * model.ts)
        l2_error = norm(result[:, 1] .- analytical) / norm(analytical)
        @test l2_error < 0.05
        println("  Single-process RZ L2 error: $(round(l2_error, digits=6))")
    end

    # ──────────────────────────────────────────────
    # 3. 1-tile distributed matches single-process
    # ──────────────────────────────────────────────
    @testset "1-tile distributed vs single-process" begin
        model, num_ts = make_rz_advection_model()
        ic_spectral, _ = blob_ic_spectral(model)

        single, _ = run_single_process_simulation(model, ic_spectral, num_ts)
        dist1, _ = run_distributed_simulation(model, ic_spectral, 1, num_ts)

        @test !any(isnan, dist1)
        rel_diff = norm(single[:, 1] .- dist1[:, 1]) / norm(single[:, 1])
        @test rel_diff < 1e-10
        println("  1-tile RZ vs single-process diff: $(round(rel_diff, sigdigits=3))")
    end

    # ──────────────────────────────────────────────
    # 4. 2-tile distributed matches single-process + analytical
    # ──────────────────────────────────────────────
    @testset "2-tile distributed vs single-process" begin
        model, num_ts = make_rz_advection_model()
        ic_spectral, gridpoints = blob_ic_spectral(model)

        single, _ = run_single_process_simulation(model, ic_spectral, num_ts)
        dist2, _ = run_distributed_simulation(model, ic_spectral, 2, num_ts)
        analytical = analytical_blob(gridpoints; dx=1.0 * num_ts * model.ts,
                                     dz=0.5 * num_ts * model.ts)

        @test !any(isnan, dist2)
        l2_error = norm(dist2[:, 1] .- analytical) / norm(analytical)
        rel_diff = norm(single[:, 1] .- dist2[:, 1]) / norm(single[:, 1])
        @test l2_error < 0.05
        @test rel_diff < 0.05
        println("  2-tile RZ L2 error vs analytical: $(round(l2_error, digits=6))")
        println("  2-tile RZ vs single-process diff: $(round(rel_diff, sigdigits=3))")
    end

    # ──────────────────────────────────────────────
    # 5. 3-tile distributed (odd partition)
    # ──────────────────────────────────────────────
    @testset "3-tile distributed vs single-process" begin
        model, num_ts = make_rz_advection_model()
        ic_spectral, _ = blob_ic_spectral(model)

        single, _ = run_single_process_simulation(model, ic_spectral, num_ts)
        dist3, _ = run_distributed_simulation(model, ic_spectral, 3, num_ts)

        @test !any(isnan, dist3)
        rel_diff = norm(single[:, 1] .- dist3[:, 1]) / norm(single[:, 1])
        @test rel_diff < 0.05
        println("  3-tile RZ vs single-process diff: $(round(rel_diff, sigdigits=3))")
    end

end
