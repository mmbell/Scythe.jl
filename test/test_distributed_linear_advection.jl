using Test
using Scythe
using Springsteel
using SparseArrays
using SharedArrays
using LinearAlgebra

include("distributed_test_helpers.jl")

@testset "Distributed LinearAdvection1D" begin

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────
    function make_advection_model(; use_springsteel=false, num_cells=100)
        if use_springsteel
            gp = SpringsteelGridParameters(
                geometry = "R",
                num_cells = num_cells,
                iMin = -50.0,
                iMax = 50.0,
                BCL = Dict("default" => NaturalBC()),
                BCR = Dict("default" => NaturalBC()),
                vars = Dict("u" => 1),
            )
        else
            bc_dict = Dict("u" => NaturalBC())
            gp = GridParameters(
                geometry = "R",
                num_cells = num_cells,
                xmin = -50.0,
                xmax = 50.0,
                BCL = bc_dict,
                BCR = bc_dict,
                vars = Dict("u" => 1),
            )
        end

        ts = 0.1
        num_ts = 200  # 20 seconds
        model = ModelParameters(
            ts = ts,
            integration_time = num_ts * ts,
            output_interval = num_ts * ts,
            equation_set = "LinearAdvection1D",
            initial_conditions = "",
            grid_params = gp,
            physical_params = Dict(:c_0 => 1.0, :K => 0.0),
        )
        return model, num_ts
    end

    function gaussian_ic_spectral(model; sigma=5.0)
        patch = createGrid(model.grid_params)
        gridpoints = getGridpoints(patch)
        for i in 1:size(patch.physical, 1)
            patch.physical[i, 1, 1] = exp(-gridpoints[i, 1]^2 / (2 * sigma^2))
        end
        spectralTransform!(patch)
        return copy(patch.spectral), gridpoints
    end

    function analytical_gaussian(gridpoints, sigma, shift)
        return [exp(-(gridpoints[i, 1] - shift)^2 / (2 * sigma^2))
                for i in 1:size(gridpoints, 1)]
    end

    sigma = 5.0
    shift = 1.0 * 200 * 0.1  # c_0 * num_ts * ts = 20 m

    # ──────────────────────────────────────────────
    # 1. Single-process baseline (validates helper against existing test)
    # ──────────────────────────────────────────────
    @testset "Single-process baseline" begin
        model, num_ts = make_advection_model()
        ic_spectral, gridpoints = gaussian_ic_spectral(model)

        result, _ = run_single_process_simulation(model, ic_spectral, num_ts)
        analytical = analytical_gaussian(gridpoints, sigma, shift)
        l2_error = norm(result[:, 1] .- analytical) / norm(analytical)

        @test l2_error < 0.05
        @test maximum(result[:, 1]) > 0.9
        @test maximum(result[:, 1]) < 1.1
        println("  Single-process L2 error: $(round(l2_error, digits=6))")
    end

    # ──────────────────────────────────────────────
    # 2. 1-tile distributed matches single-process
    # ──────────────────────────────────────────────
    @testset "1-tile distributed vs single-process" begin
        model, num_ts = make_advection_model()
        ic_spectral, _ = gaussian_ic_spectral(model)

        single, _ = run_single_process_simulation(model, ic_spectral, num_ts)
        dist1, _ = run_distributed_simulation(model, ic_spectral, 1, num_ts)

        @test !any(isnan, dist1)
        rel_diff = norm(single[:, 1] .- dist1[:, 1]) / norm(single[:, 1])
        @test rel_diff < 1e-10
        println("  1-tile vs single-process relative diff: $(round(rel_diff, sigdigits=3))")
    end

    # ──────────────────────────────────────────────
    # 3. 2-tile distributed matches analytical + single-process
    # ──────────────────────────────────────────────
    @testset "2-tile distributed vs analytical" begin
        model, num_ts = make_advection_model()
        ic_spectral, gridpoints = gaussian_ic_spectral(model)

        single, _ = run_single_process_simulation(model, ic_spectral, num_ts)
        dist2, _ = run_distributed_simulation(model, ic_spectral, 2, num_ts)
        analytical = analytical_gaussian(gridpoints, sigma, shift)

        @test !any(isnan, dist2)
        l2_error = norm(dist2[:, 1] .- analytical) / norm(analytical)
        rel_diff = norm(single[:, 1] .- dist2[:, 1]) / norm(single[:, 1])

        @test l2_error < 0.05
        @test rel_diff < 0.05
        println("  2-tile L2 error vs analytical: $(round(l2_error, digits=6))")
        println("  2-tile vs single-process diff: $(round(rel_diff, sigdigits=3))")
    end

    # ──────────────────────────────────────────────
    # 4. SpringsteelGridParameters 2-tile distributed
    # ──────────────────────────────────────────────
    @testset "SpringsteelGridParameters 2-tile distributed" begin
        model, num_ts = make_advection_model(use_springsteel=true)
        ic_spectral, gridpoints = gaussian_ic_spectral(model)

        dist2, _ = run_distributed_simulation(model, ic_spectral, 2, num_ts)
        analytical = analytical_gaussian(gridpoints, sigma, shift)

        @test !any(isnan, dist2)
        l2_error = norm(dist2[:, 1] .- analytical) / norm(analytical)
        @test l2_error < 0.05
        println("  SpringsteelGridParameters 2-tile L2 error: $(round(l2_error, digits=6))")
    end

    # ──────────────────────────────────────────────
    # 5. 3-tile distributed (odd partition)
    # ──────────────────────────────────────────────
    @testset "3-tile distributed vs single-process" begin
        model, num_ts = make_advection_model()
        ic_spectral, gridpoints = gaussian_ic_spectral(model)

        single, _ = run_single_process_simulation(model, ic_spectral, num_ts)
        dist3, _ = run_distributed_simulation(model, ic_spectral, 3, num_ts)

        @test !any(isnan, dist3)
        rel_diff = norm(single[:, 1] .- dist3[:, 1]) / norm(single[:, 1])
        @test rel_diff < 0.05
        println("  3-tile vs single-process diff: $(round(rel_diff, sigdigits=3))")
    end

end
