using Test
using Scythe
using Springsteel
using SparseArrays

include("distributed_test_helpers.jl")
include(joinpath(@__DIR__, "..", "benchmarks", "common", "harness.jl"))

# Pure path/lookup construction for arm-aware reference keying (benchmarks/common/
# harness.jl: reference_csv_path, load_targets). No model run, so this runs unconditionally
# (not gated behind SCYTHE_SLOW_TESTS). The design point under test: an ARMED benchmark
# variant (e.g. o01_rainfall's ISHMAEL ice arm, benchmarks/o01_rainfall.jl) must read/write
# its OWN reference file and target windows, while an UNARMED call (the default arm="")
# must be completely unaffected -- byte-identical to the path/lookup construction before
# the `arm` keyword existed at all.
@testset "Benchmark harness: arm-aware reference keying" begin
    quick_opts = BenchmarkOptions(:quick, STAGE_MC, :rirk, 1, false, false, 1.0, 1, false, false)
    full_opts  = BenchmarkOptions(:full,  STAGE_MC, :rirk, 1, false, false, 1.0, 1, false, false)

    @testset "reference_csv_path: unarmed is byte-identical to the pre-arm path" begin
        @test reference_csv_path("o01_rainfall", quick_opts) ==
              joinpath(REFERENCE_DATA_DIR, "o01_rainfall", "quick_mc_rirk_final.csv")
        @test reference_csv_path("o01_rainfall", full_opts) ==
              joinpath(REFERENCE_DATA_DIR, "o01_rainfall", "full_mc_rirk_diagnostics.csv")
        # Explicit arm="" must equal the no-keyword call exactly.
        @test reference_csv_path("o01_rainfall", quick_opts; arm="") ==
              reference_csv_path("o01_rainfall", quick_opts)
        @test reference_csv_path("o01_rainfall", full_opts; arm="") ==
              reference_csv_path("o01_rainfall", full_opts)
    end

    @testset "reference_csv_path: armed gets its own distinct path" begin
        @test reference_csv_path("o01_rainfall", quick_opts; arm="ice") ==
              joinpath(REFERENCE_DATA_DIR, "o01_rainfall", "quick_mc_rirk_ice_final.csv")
        @test reference_csv_path("o01_rainfall", full_opts; arm="ice") ==
              joinpath(REFERENCE_DATA_DIR, "o01_rainfall", "full_mc_rirk_ice_diagnostics.csv")
        # Armed and unarmed paths must never collide (the point of the whole change).
        @test reference_csv_path("o01_rainfall", quick_opts; arm="ice") !=
              reference_csv_path("o01_rainfall", quick_opts)
        @test reference_csv_path("o01_rainfall", full_opts; arm="ice") !=
              reference_csv_path("o01_rainfall", full_opts)
    end

    @testset "load_targets: unarmed is byte-identical to the pre-arm lookup" begin
        warm_names = [t.name for t in load_targets("o01_rainfall", quick_opts)]
        @test !isempty(warm_names)
        @test "peak_rain_rate_gm2s" in warm_names
        # Explicit arm="" must equal the no-keyword call exactly.
        @test [t.name for t in load_targets("o01_rainfall", quick_opts; arm="")] == warm_names
    end

    @testset "load_targets: armed lookup does NOT fall back to the warm windows" begin
        # "o01_rainfall_ice" is seeded in expected_values.jl as an EMPTY Dict (ice windows
        # not yet measured) -- an armed run must get NO targets, not the warm case's.
        ice_targets = load_targets("o01_rainfall", quick_opts; arm="ice")
        @test isempty(ice_targets)
        # An arm with no entry in BENCHMARK_EXPECTED at all behaves the same way: empty,
        # not an error and not a silent fallback to "o01_rainfall".
        @test isempty(load_targets("o01_rainfall", quick_opts; arm="nonexistent_arm"))
        # The unarmed lookup for a genuinely undefined case still errors, exactly as before.
        @test_throws ErrorException load_targets("no_such_benchmark_case", quick_opts)
    end
end

# Micro smoke test of the Straka density current configuration: a few seconds
# of integration on a tiny grid, single process. Catches gross breakage of the
# Euler_test + semi-implicit + reference state path without the cost of the
# real benchmarks (which live in benchmarks/ and run at paper-grade
# resolution). Gated behind SCYTHE_SLOW_TESTS because it still integrates a
# few hundred timesteps.

if get(ENV, "SCYTHE_SLOW_TESTS", "") == "1"
    @testset "Straka93 smoke test" begin
        mktempdir() do tmpdir
            vars = Dict("s" => 1, "xi" => 2, "mu" => 3, "u" => 4, "w" => 5)
            scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
            bc_side = merge(scalar_bc, Dict("u" => DirichletBC()))
            bc_topbot = merge(scalar_bc, Dict("w" => DirichletBC()))
            gp = GridParameters(
                geometry = "RZ",
                num_cells = 64,
                iMin = 0.0, iMax = 25.6e3,
                kMin = 0.0, kMax = 6.4e3,
                kDim = 32,
                BCL = bc_side, BCR = bc_side, BCB = bc_topbot, BCT = bc_topbot,
                vars = vars,
            )
            sounding = Scythe.write_dry_sounding(joinpath(tmpdir, "straka.ref");
                                                 theta = 300.0, zmax = 8000.0)
            model = ModelParameters(
                ts = 0.25,
                integration_time = 10.0,
                output_interval = 10.0,
                equation_set = "Euler_test",
                ref_state_file = sounding,
                grid_params = gp,
                physical_params = Dict(:K => 75.0, :Kvdiff => 0.0),
                options = Dict(:semiimplicit => true, :exact_reference_state => false),
            )

            patch = createGrid(gp)
            gridpoints = getGridpoints(patch)
            z = gridpoints[1:gp.kDim, 2]
            column = Scythe.reference_column(patch, gp)
            ref = Scythe.calculate_reference_state(model, z, column)
            patch.physical .= 0.0
            Scythe.temperature_bubble!(patch, gridpoints, ref)
            spectralTransform!(patch)

            num_ts = round(Int, model.integration_time / model.ts)
            result, _ = run_single_process_simulation(model, copy(patch.spectral), num_ts)

            @test !any(isnan, result)
            # The cold bubble must start sinking: negative w below the bubble
            w = result[:, 5]
            @test minimum(w) < -0.1
            @test maximum(abs.(w)) < 50.0   # and nothing explosive
            # Entropy perturbation still negative (cold pool present)
            @test minimum(result[:, 1]) < -10.0
        end
    end
else
    @info "Skipping benchmark smoke test (set SCYTHE_SLOW_TESTS=1 to enable)"
end
