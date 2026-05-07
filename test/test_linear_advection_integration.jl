using Test
using Scythe
using Springsteel
using SparseArrays
using SharedArrays
using LinearAlgebra

@testset "LinearAdvection1D Integration" begin

    # ──────────────────────────────────────────────
    # Setup: R geometry, [-50, 50], 100 cells, R0 BCs, c=1 m/s, K=0
    # Gaussian IC at center, advect 20s so signal stays well within domain
    # ──────────────────────────────────────────────

    # 1. Create grid via GridParameters (backward compat)
    bc_dict = Dict("u" => NaturalBC())
    gp = GridParameters(
        geometry = "R",
        num_cells = 100,
        iMin = -50.0,
        iMax = 50.0,
        BCL = bc_dict,
        BCR = bc_dict,
        vars = Dict("u" => 1),
    )

    ts = 0.1
    c_0 = 1.0
    K = 0.0
    num_ts = 200  # 20 seconds — signal advects 20m, stays well within domain

    model = ModelParameters(
        ts = ts,
        integration_time = num_ts * ts,
        output_interval = num_ts * ts,
        equation_set = "LinearAdvection1D",
        initial_conditions = "",
        grid_params = gp,
        physical_params = Dict(:c_0 => c_0, :K => K),
    )

    patch = createGrid(model.grid_params)

    # 2. Set Gaussian IC (sigma=5, centered at x=0, essentially zero at boundaries)
    gridpoints = getGridpoints(patch)
    sigma = 5.0
    for i in 1:size(patch.physical, 1)
        patch.physical[i, 1, 1] = exp(-gridpoints[i, 1]^2 / (2 * sigma^2))
    end

    # 3. Transform to spectral and back to get spectrally-consistent IC
    spectralTransform!(patch)
    gridTransform!(patch)
    initial_physical = copy(patch.physical[:, 1, 1])

    # 4. Create ModelTile using patch as tile (single-process, no halo exchange)
    haloReceiveMap = sparse(Int64[], Int64[], Float64[], size(patch.spectral, 1), size(patch.spectral, 2))
    mtile = createModelTile(patch, patch, model, haloReceiveMap)

    # 5. Run N timesteps using direct patch transforms (bypasses tiling)
    for t in 1:num_ts
        # advance_column reads physical, computes tendencies, writes var_np1
        Scythe.advance_column(mtile, -1, t)
        # calcTendency: var_np1 → physical → spectral
        Scythe.calcTendency(mtile)
        # gridTransform: spectral → physical (with derivatives)
        gridTransform!(mtile.tile)
    end

    final_physical = mtile.tile.physical[:, 1, 1]

    # 6. Compute analytically-shifted Gaussian for comparison
    shift = c_0 * num_ts * ts  # 20m
    analytical = [exp(-(gridpoints[i, 1] - shift)^2 / (2 * sigma^2)) for i in 1:size(gridpoints, 1)]

    # 7. Verify solution matches shifted Gaussian
    # Use L2 relative error against the analytical shifted solution
    l2_error = norm(final_physical .- analytical) / norm(analytical)
    @test l2_error < 0.05  # Allow 5% error for spectral representation + time integration

    # Also verify the signal hasn't decayed or blown up
    @test maximum(final_physical) > 0.9  # Peak should still be close to 1
    @test maximum(final_physical) < 1.1  # No blow-up

    println("LinearAdvection1D integration test: L2 relative error = $(round(l2_error, digits=6))")
    println("  Peak value: $(round(maximum(final_physical), digits=6)) at x=$(round(gridpoints[argmax(final_physical)], digits=2))")
end
