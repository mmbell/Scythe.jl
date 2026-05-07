using Test
using Scythe
using Springsteel
using SparseArrays
using SharedArrays
using LinearAlgebra

include("distributed_test_helpers.jl")

@testset "Oneway_ShallowWater_Slab" begin

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    """Create an Oneway_ShallowWater_Slab model with an RL grid."""
    function make_sw_slab_model(; num_cells=20, iMax=3.0e5, ts=3.0,
                                  integration_time=30.0, K=3000.0,
                                  max_wn=0)
        gp = SpringsteelGridParameters(
            geometry = "RL",
            iMin = 0.0,
            iMax = iMax,
            num_cells = num_cells,
            max_wavenumber = Dict(
                "h" => max_wn, "u" => max_wn, "v" => max_wn,
                "ub" => max_wn, "vb" => max_wn, "wb" => max_wn),
            BCL = Dict(
                "h"  => NeumannBC(),
                "u"  => DirichletBC(),
                "v"  => DirichletBC(),
                "ub" => DirichletBC(),
                "vb" => DirichletBC(),
                "wb" => NeumannBC()),
            BCR = Dict(
                "h"  => NaturalBC(),
                "u"  => NeumannBC(),
                "v"  => NaturalBC(),
                "ub" => NeumannBC(),
                "vb" => NaturalBC(),
                "wb" => NaturalBC()),
            vars = Dict(
                "h" => 1, "u" => 2, "v" => 3,
                "ub" => 4, "vb" => 5, "wb" => 6),
        )

        num_ts = round(Int, integration_time / ts)
        model = ModelParameters(
            ts = ts,
            integration_time = integration_time,
            output_interval = integration_time,
            equation_set = "Oneway_ShallowWater_Slab",
            initial_conditions = "",
            grid_params = gp,
            physical_params = Dict(
                :g => 9.81,
                :K => K,
                :Cd => 2.4e-3,
                :Hfree => 2000.0,
                :Hb => 1000.0,
                :f => 5.0e-5),
        )
        return model, num_ts
    end

    """Set Rankine vortex ICs on a patch. Returns (spectral_copy, gridpoints)."""
    function rankine_vortex_ic!(patch; Rmax=50000.0, Vmax=30.0)
        gridpoints = getGridpoints(patch)
        V0 = Vmax / Rmax
        f = 5.0e-5

        r_prev = 0.0
        h_accum = 0.0
        for i in 1:size(patch.physical, 1)
            r_m = gridpoints[i, 1]

            vbar = r_m < Rmax ? V0 * r_m : Rmax^2 * V0 / r_m

            if r_m > r_prev
                dhdr = (f * vbar + vbar^2 / r_m) / 9.81
                h_accum += dhdr * (r_m - r_prev)
                r_prev = r_m
            end

            patch.physical[i, 1, 1] = h_accum
            patch.physical[i, 2, 1] = 0.0
            patch.physical[i, 3, 1] = vbar
            patch.physical[i, 4, 1] = 0.0
            patch.physical[i, 5, 1] = vbar
            patch.physical[i, 6, 1] = 0.0
        end

        spectralTransform!(patch)
        return copy(patch.spectral), gridpoints
    end

    """Create a ModelTile for single-process testing (tile = patch)."""
    function make_single_process_mtile(model, patch)
        haloReceiveMap = sparse(Int64[], Int64[], Float64[],
                                size(patch.spectral, 1), size(patch.spectral, 2))
        return createModelTile(patch, patch, model, haloReceiveMap)
    end

    # ──────────────────────────────────────────────
    # 1. Resting state: all tendencies should be ≈ 0
    # ──────────────────────────────────────────────
    @testset "Resting state tendencies" begin
        model, _ = make_sw_slab_model(num_cells=20, ts=1.0, integration_time=1.0)
        patch = createGrid(model.grid_params)

        patch.physical[:, 1, 1] .= 100.0
        patch.physical[:, 2:6, 1] .= 0.0
        spectralTransform!(patch)
        gridTransform!(patch)

        mtile = make_single_process_mtile(model, patch)
        Scythe.advance_column(mtile, -1, 1)

        var_names = ["h", "ug", "vg", "ub", "vb", "wb"]
        for v in 1:6
            max_tend = maximum(abs.(mtile.expdot_n[:, v]))
            @test max_tend < 1e-8
            println("  $(var_names[v]) max tendency: $(round(max_tend, sigdigits=3))")
        end
    end

    # ──────────────────────────────────────────────
    # 2. Gradient wind balance: free-atmo tendencies ≈ 0
    # ──────────────────────────────────────────────
    @testset "Gradient wind balance" begin
        model, _ = make_sw_slab_model(num_cells=20, ts=1.0, integration_time=1.0, K=0.0)
        patch = createGrid(model.grid_params)
        rankine_vortex_ic!(patch)
        gridTransform!(patch)

        mtile = make_single_process_mtile(model, patch)
        Scythe.advance_column(mtile, -1, 1)

        h_tend_max = maximum(abs.(mtile.expdot_n[:, 1]))
        @test h_tend_max < 1.0

        ug_tend_max = maximum(abs.(mtile.expdot_n[:, 2]))
        @test ug_tend_max < 1.0

        vg_tend_max = maximum(abs.(mtile.expdot_n[:, 3]))
        @test vg_tend_max < 1e-6

        wb = patch.physical[:, 6, 1]
        @test maximum(abs.(wb)) < 1.0

        @test maximum(abs.(mtile.expdot_n[:, 6])) < 1e-10

        println("  Max |h tend| = $(round(h_tend_max, sigdigits=3))")
        println("  Max |ug tend| = $(round(ug_tend_max, sigdigits=3))")
        println("  Max |vg tend| = $(round(vg_tend_max, sigdigits=3))")
    end

    # ──────────────────────────────────────────────
    # 3. Coriolis tendency sign check
    # ──────────────────────────────────────────────
    @testset "Coriolis tendency sign" begin
        model, _ = make_sw_slab_model(num_cells=20, ts=1.0, integration_time=1.0, K=0.0)
        patch = createGrid(model.grid_params)
        gridpoints = getGridpoints(patch)

        for i in 1:size(patch.physical, 1)
            patch.physical[i, 1, 1] = 0.0
            patch.physical[i, 2, 1] = 0.0
            patch.physical[i, 3, 1] = 10.0
            patch.physical[i, 4, 1] = 0.0
            patch.physical[i, 5, 1] = 10.0
            patch.physical[i, 6, 1] = 0.0
        end
        spectralTransform!(patch)
        gridTransform!(patch)

        mtile = make_single_process_mtile(model, patch)
        Scythe.advance_column(mtile, -1, 1)

        mid = div(size(mtile.expdot_n, 1), 2)
        @test mtile.expdot_n[mid, 2] > 0.0
        @test mtile.expdot_n[mid, 5] < 0.0

        println("  ug tendency at midpoint: $(round(mtile.expdot_n[mid, 2], sigdigits=4))")
        println("  vb tendency at midpoint: $(round(mtile.expdot_n[mid, 5], sigdigits=4))")
    end

    # ──────────────────────────────────────────────
    # 4. Diffusion tendency is finite
    # ──────────────────────────────────────────────
    @testset "Diffusion tendency" begin
        model, _ = make_sw_slab_model(num_cells=20, ts=1.0, integration_time=1.0, K=5000.0)
        patch = createGrid(model.grid_params)
        gridpoints = getGridpoints(patch)

        Rmax = 50000.0
        sigma_r = 30000.0
        for i in 1:size(patch.physical, 1)
            r_m = gridpoints[i, 1]
            patch.physical[i, 1, 1] = 0.0
            patch.physical[i, 2, 1] = 0.0
            patch.physical[i, 3, 1] = 0.0
            patch.physical[i, 4, 1] = 0.0
            patch.physical[i, 5, 1] = 20.0 * exp(-(r_m - Rmax)^2 / (2 * sigma_r^2))
            patch.physical[i, 6, 1] = 0.0
        end
        spectralTransform!(patch)
        gridTransform!(patch)

        mtile = make_single_process_mtile(model, patch)
        Scythe.advance_column(mtile, -1, 1)

        for v in 1:6
            @test !any(isnan, mtile.expdot_n[:, v])
            @test !any(isinf, mtile.expdot_n[:, v])
        end

        peak_idx = argmax(patch.physical[:, 5, 1])
        println("  vb tendency at peak: $(round(mtile.expdot_n[peak_idx, 5], sigdigits=4))")
    end

    # ──────────────────────────────────────────────
    # 5. Diagnostic wb computation
    # ──────────────────────────────────────────────
    @testset "Diagnostic wb" begin
        model, _ = make_sw_slab_model(num_cells=20, ts=1.0, integration_time=1.0)
        patch = createGrid(model.grid_params)
        gridpoints = getGridpoints(patch)

        Rmax = 50000.0
        for i in 1:size(patch.physical, 1)
            r_m = gridpoints[i, 1]
            ub_val = -5.0 * r_m / Rmax * exp(-(r_m - Rmax)^2 / (2 * 30000.0^2))
            patch.physical[i, 1, 1] = 0.0
            patch.physical[i, 2, 1] = 0.0
            patch.physical[i, 3, 1] = 0.0
            patch.physical[i, 4, 1] = ub_val
            patch.physical[i, 5, 1] = 0.0
            patch.physical[i, 6, 1] = 0.0
        end
        spectralTransform!(patch)
        gridTransform!(patch)

        mtile = make_single_process_mtile(model, patch)
        Scythe.advance_column(mtile, -1, 1)

        wb = mtile.tile.physical[:, 6, 1]
        @test !any(isnan, wb)
        @test !any(isinf, wb)
        @test maximum(abs.(mtile.expdot_n[:, 6])) < 1e-10

        println("  Max |wb| = $(round(maximum(abs.(wb)), sigdigits=4))")
    end

    # ──────────────────────────────────────────────
    # 6. Single-process integration: no NaN after 10 steps
    # ──────────────────────────────────────────────
    @testset "Single-process integration — no NaN" begin
        model, _ = make_sw_slab_model(num_cells=20, ts=3.0, integration_time=30.0)
        patch = createGrid(model.grid_params)
        ic_spectral, _ = rankine_vortex_ic!(patch)

        result, _ = run_single_process_simulation(model, ic_spectral, 10)

        @test !any(isnan, result)
        @test !any(isinf, result)
        @test maximum(abs.(result[:, 1])) < 1e6

        println("  After 10 steps: max |h| = $(round(maximum(abs.(result[:, 1])), sigdigits=4))")
        println("  After 10 steps: max |vg| = $(round(maximum(abs.(result[:, 3])), sigdigits=4))")
    end

    # ──────────────────────────────────────────────
    # 7. 2-tile distributed integration: no NaN
    # ──────────────────────────────────────────────
    @testset "2-tile distributed integration — no NaN" begin
        model, _ = make_sw_slab_model(num_cells=20, ts=3.0, integration_time=30.0)
        patch = createGrid(model.grid_params)
        ic_spectral, _ = rankine_vortex_ic!(patch)

        dist_result, _ = run_distributed_simulation(model, ic_spectral, 2, 10)

        @test !any(isnan, dist_result)
        @test !any(isinf, dist_result)

        println("  After 10 steps: max |h| = $(round(maximum(abs.(dist_result[:, 1])), sigdigits=4))")
        println("  After 10 steps: max |vg| = $(round(maximum(abs.(dist_result[:, 3])), sigdigits=4))")
    end

    # ──────────────────────────────────────────────
    # 8. 2-tile distributed matches single-process
    # ──────────────────────────────────────────────
    @testset "2-tile distributed vs single-process" begin
        model, _ = make_sw_slab_model(num_cells=20, ts=3.0, integration_time=30.0)
        patch = createGrid(model.grid_params)
        ic_spectral, _ = rankine_vortex_ic!(patch)

        single, _ = run_single_process_simulation(model, ic_spectral, 10)
        dist2, _ = run_distributed_simulation(model, ic_spectral, 2, 10)

        var_names = ["h", "ug", "vg", "ub", "vb", "wb"]
        for v in 1:6
            max_val = max(maximum(abs.(single[:, v])), 1e-10)
            rel_diff = norm(single[:, v] .- dist2[:, v]) / max_val
            @test rel_diff < 0.05
            println("  $(var_names[v]) relative diff: $(round(rel_diff, sigdigits=3))")
        end
    end

    # ──────────────────────────────────────────────
    # 9. Symmetric spinup configuration (mixed max_wavenumber)
    # ──────────────────────────────────────────────
    @testset "Symmetric spinup configuration — no NaN" begin
        gp = SpringsteelGridParameters(
            geometry = "RL",
            iMin = 0.0,
            iMax = 3.0e5,
            num_cells = 20,
            max_wavenumber = Dict(
                "h"  => 0,  "u"  => 0,  "v"  => 0,
                "ub" => -1, "vb" => -1, "wb" => -1),
            BCL = Dict(
                "h"  => NeumannBC(),
                "u"  => DirichletBC(),
                "v"  => DirichletBC(),
                "ub" => DirichletBC(),
                "vb" => DirichletBC(),
                "wb" => NeumannBC()),
            BCR = Dict(
                "h"  => NaturalBC(),
                "u"  => NeumannBC(),
                "v"  => NaturalBC(),
                "ub" => NeumannBC(),
                "vb" => NaturalBC(),
                "wb" => NaturalBC()),
            vars = Dict(
                "h" => 1, "u" => 2, "v" => 3,
                "ub" => 4, "vb" => 5, "wb" => 6),
        )

        model = ModelParameters(
            ts = 3.0,
            integration_time = 30.0,
            output_interval = 30.0,
            equation_set = "Oneway_ShallowWater_Slab",
            initial_conditions = "",
            grid_params = gp,
            physical_params = Dict(
                :g => 9.81, :K => 3000.0, :Cd => 2.4e-3,
                :Hfree => 2000.0, :Hb => 1000.0, :f => 5.0e-5),
        )

        patch = createGrid(model.grid_params)
        ic_spectral, _ = rankine_vortex_ic!(patch)

        single, _ = run_single_process_simulation(model, ic_spectral, 10)
        @test !any(isnan, single)

        dist2, _ = run_distributed_simulation(model, ic_spectral, 2, 10)
        @test !any(isnan, dist2)

        println("  Single-process max |vg| = $(round(maximum(abs.(single[:, 3])), sigdigits=4))")
        println("  Distributed max |vg| = $(round(maximum(abs.(dist2[:, 3])), sigdigits=4))")
    end

end
