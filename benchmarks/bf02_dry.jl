#!/usr/bin/env julia
# Bryan & Fritsch (2002) dry warm bubble benchmark.
#
# A +2 K potential temperature bubble rises through a dry, neutrally stable
# (theta = 300 K) hydrostatic atmosphere in a 20 x 10 km domain with rigid
# walls and no diffusion, forming two rotors and a thin arch after 1000 s.
# Verified against the extrema printed in Fig. 1 of the paper.
#
#   julia --project=. benchmarks/bf02_dry.jl --mode quick --stage legacy
#
# Modes: quick (200 m cells) | full (100 m cells, paper-grade)
# Stages: legacy (Euler_test) | pe (primitive_equation_XZ)
#
# Reference: Bryan & Fritsch (2002), Mon. Wea. Rev. 130, 2917-2928.
# reference/bryan_fritsch_mwr2002.pdf

using Distributed

include(joinpath(@__DIR__, "common", "harness.jl"))
opts = parse_benchmark_args(ARGS)

addprocs(opts.workers, exeflags="--threads=auto")
@everywhere using Springsteel
@everywhere using Scythe

include(joinpath(@__DIR__, "common", "diagnostics.jl"))

# ── Configuration ──────────────────────────────────────────────────────────

const BF02_DRY_VARS = ["s", "xi", "mu", "u", "w"]

function bf02_dry_model(opts::BenchmarkOptions)
    if opts.mode == :full
        num_cells = 200         # 100 m cells
        kDim = 128
        ts = 0.1
        output_interval = 100.0
    else
        num_cells = 100         # 200 m cells
        kDim = 64
        ts = 0.2
        output_interval = 500.0
    end

    if opts.stage == :legacy
        # The dry case carries no liquid water, so the 5-variable Euler_test
        # set is physically identical to the historical 6-variable BF02 run
        equation_set = "Euler_test"
        physical_params = Dict(:K => 0.0, :Kvdiff => 0.0)
    else
        error("--stage pe for bf02_dry is not wired up yet (Stage 2 of the benchmark plan)")
    end

    output_dir = benchmark_output_dir("bf02_dry", opts)
    scalar_bc = Dict(v => NeumannBC() for v in BF02_DRY_VARS)
    wall_bc = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))

    grid_params = GridParameters(
        geometry = "RZ",   # Cartesian x-z in Springsteel
        iMin = 0.0,
        iMax = 20.0e3,
        num_cells = num_cells,
        kMin = 0.0,
        kMax = 10.0e3,
        kDim = kDim,
        BCL = wall_bc,
        BCR = wall_bc,
        BCB = wall_bc,
        BCT = wall_bc,
        vars = Dict(v => i for (i, v) in enumerate(BF02_DRY_VARS)),
    )

    return ModelParameters(
        ts = ts,
        integration_time = 1000.0,
        output_interval = output_interval,
        equation_set = equation_set,
        initial_conditions = joinpath(output_dir, "bf02_dry_ics.csv"),
        output_dir = output_dir,
        ref_state_file = joinpath(output_dir, "bf02_dry.ref"),
        grid_params = grid_params,
        physical_params = physical_params,
        options = Dict(:semiimplicit => true, :exact_reference_state => false),
    )
end

# ── Initial conditions ─────────────────────────────────────────────────────

"""Generate the reference sounding and warm bubble initial conditions."""
function bf02_dry_init!(model)
    Scythe.write_dry_sounding(model.ref_state_file; theta=300.0, zmax=12000.0)

    patch = createGrid(model.grid_params)
    gridpoints = Scythe.getGridpoints(patch)
    kDim = model.grid_params.kDim
    z = gridpoints[1:kDim, 2]
    column = deepcopy(patch.kbasis.data[1])
    ref = Scythe.calculate_reference_state(model, z, column)

    patch.physical .= 0.0
    Scythe.theta_bubble!(patch, gridpoints, ref;
                         xc=10000.0, xr=2000.0, zc=2000.0, zr=2000.0, dtheta_max=2.0)
    Scythe.write_ics_csv(model.initial_conditions, patch, gridpoints)
end

# ── Diagnostics ────────────────────────────────────────────────────────────

function bf02_dry_diagnostics(model)
    df = read_final_output(model)
    ref, _, kDim = rebuild_reference(model)
    theta_p, _ = theta_perturbation(df, ref, kDim)
    return Dict(
        "max_theta_p" => maximum(theta_p),
        "min_theta_p" => minimum(theta_p),
        "max_w" => maximum(df.w),
        "min_w" => minimum(df.w),
        "max_u" => maximum(df.u),
        "min_u" => minimum(df.u),
    )
end

# ── Run ────────────────────────────────────────────────────────────────────

model = bf02_dry_model(opts)
passed = run_benchmark("bf02_dry", opts;
                       model = model,
                       init! = bf02_dry_init!,
                       diagnostics = bf02_dry_diagnostics,
                       varnames = BF02_DRY_VARS)
exit(passed ? 0 : 1)
