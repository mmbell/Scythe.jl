#!/usr/bin/env julia
# Recompute the conservation drifts on an ALREADY-COMPLETED bf02_moist run, using the
# current diagnostics.jl code, WITHOUT re-running the simulation. Reads the run's
# <output_dir>/*_physical.csv and reference, rebuilds the model config, and prints the
# mass / energy / entropy drift percentages.
#
# Usage (must match the completed run's mode/stage/grid):
#   julia --project=. benchmarks/reeval_conservation.jl --mode full --stage pe-rho_d-pd --grid rirk

using Scythe
using Springsteel

include(joinpath(@__DIR__, "common", "harness.jl"))
include(joinpath(@__DIR__, "common", "diagnostics.jl"))

opts = parse_benchmark_args(ARGS)

# Slot layout per stage (mirrors bf02_moist.jl)
function bf02_moist_vars(stage)
    stage == :legacy && return ["s", "xi", "mu", "u", "w", "mu_l", "qss"]
    stage == STAGE_PE_RHOD_PD && return ["s", "rho_d", "rho_v", "u", "w", "rho_c", "rho_r", "mu_sat"]
    stage == STAGE_PE_RHOD && return ["s", "rho_d", "mu", "u", "w", "mu_c", "mu_r", "mu_sat"]
    return ["s", "xi", "mu", "u", "w", "mu_c", "mu_r", "mu_sat"]
end

liquid_vars(stage) = stage == :legacy ? ["mu_l"] :
                     stage == STAGE_PE_RHOD_PD ? ["rho_c", "rho_r"] : ["mu_c", "mu_r"]

# Model builder mirroring bf02_moist.jl::bf02_moist_model — it must rebuild the *same* grid as
# the run whose output is being re-evaluated. The quick-mode counts here had drifted from
# bf02_moist (num_cells 100/kDim 50 vs 50/75), so the reconstructed grid did not actually match
# the run; they are realigned here alongside the move to cell counts.
function bf02_moist_model(opts::BenchmarkOptions)
    if opts.mode == :full
        num_cells_i = 200; num_cells_k = 100; kDim = 300; ts = 0.1/3.0; output_interval = 100.0
    else
        num_cells_i = 50;  num_cells_k = 25;  kDim = 75; ts = 0.1; output_interval = 250.0
    end
    vars = bf02_moist_vars(opts.stage)
    if opts.stage in (STAGE_PE_RHOD, STAGE_PE_RHOD_PD)
        equation_set = opts.stage == STAGE_PE_RHOD_PD ?
            "primitive_equation_XZ_rhod_pd" : "primitive_equation_XZ_rhod"
        physical_params = Dict(:Khdiff => 0.0, :Kvdiff => 0.0, :Kv_mudiff => 0.0,
                               :alpha => 0.0, :z_damp => 20.0e3)
        options = Dict(:semiimplicit => true, :exact_reference_state => true,
                       :precipitation => false, :vertical_mixing => false)
    else
        error("This re-eval helper is for the pe-rho_d / pe-rho_d-pd stages.")
    end
    ts = vertical_ts(ts, opts)
    output_dir = benchmark_output_dir("bf02_moist", opts)
    scalar_bc = Dict(v => NeumannBC() for v in vars)
    wall_bc = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))
    grid_params = GridParameters(;
        geometry = benchmark_geometry(opts),
        iMin = 0.0, iMax = 20.0e3, num_cells_i = num_cells_i,
        kMin = 0.0, kMax = 10.0e3, vertical_size(opts; num_cells_k = num_cells_k, kDim = kDim)...,
        BCL = wall_bc, BCR = wall_bc, BCB = wall_bc, BCT = wall_bc,
        vars = Dict(v => i for (i, v) in enumerate(vars)),
    )
    return ModelParameters(
        ts = ts, integration_time = 1000.0, output_interval = output_interval,
        equation_set = equation_set,
        initial_conditions = joinpath(output_dir, "bf02_moist_ics.csv"),
        output_dir = output_dir,
        ref_state_file = joinpath(output_dir, "bf02_moist_exact.ref"),
        grid_params = grid_params, physical_params = physical_params, options = options,
    )
end

model = bf02_moist_model(opts)
ref, _, _ = rebuild_reference(model)
drift = conservation_drift(model, ref; liquid_vars = liquid_vars(opts.stage))
println("Re-evaluated conservation drift for $(model.output_dir):")
for k in ("dry_mass_drift_pct", "mass_drift_pct", "water_mass_drift_pct",
          "energy_drift_pct", "entropy_drift_pct")
    haskey(drift, k) && println("  $(rpad(k, 22)) = $(drift[k])")
end
