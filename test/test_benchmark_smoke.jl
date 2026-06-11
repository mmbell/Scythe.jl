using Test
using Scythe
using Springsteel
using SparseArrays

include("distributed_test_helpers.jl")

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
