#!/usr/bin/env julia
# Straka et al. (1993) cold density current benchmark.
#
# A -15 K elliptical cold bubble is released in a dry, neutrally stable
# (theta = 300 K) hydrostatic atmosphere in a 25.6 x 6.4 km domain and collapses
# into a density current with Kelvin-Helmholtz rotors. Uniform diffusion
# K = 75 m^2/s makes the solution convergent. Verified against the REFC
# reference solution at t = 900 s (Tables IV and V of the paper).
#
#   julia --project=. benchmarks/straka93.jl --mode quick --stage legacy
#
# Modes: quick (100 m cells, regression-sized) | full (25 m cells, paper-grade)
# Stages: legacy (Euler_test) | pe (primitive_equation_XZ)
#
# Reference: Straka, Wilhelmson, Wicker, Anderson, Droegemeier (1993),
# Int. J. Numer. Methods Fluids 17, 1-22. reference/Straka.pdf

using Distributed

include(joinpath(@__DIR__, "common", "harness.jl"))
opts = parse_benchmark_args(ARGS)

addprocs(opts.workers, exeflags="--threads=auto")
@everywhere using Springsteel
@everywhere using Scythe

include(joinpath(@__DIR__, "common", "diagnostics.jl"))

# ── Configuration ──────────────────────────────────────────────────────────

const PE_VARS = ["s", "xi", "mu", "u", "w", "mu_c", "mu_r", "mu_sat"]
# Linear dry-air-density variant: slot 2 is "rho_d" (rho_d') instead of "xi"
const PE_VARS_RHOD = ["s", "rho_d", "mu", "u", "w", "mu_c", "mu_r", "mu_sat"]
function straka_vars(stage)
    stage == :legacy && return ["s", "xi", "mu", "u", "w"]
    stage == STAGE_PE_RHOD && return PE_VARS_RHOD
    return PE_VARS
end

function straka_model(opts::BenchmarkOptions)
    if opts.mode == :full
        num_cells = 1024        # 25 m cells
        kDim = 256
        ts = 0.015625
        output_interval = 100.0
    else
        num_cells = 256         # 100 m cells
        kDim = 64
        ts = 0.0625
        output_interval = 300.0
    end

    vars = straka_vars(opts.stage)
    if opts.stage == :legacy
        equation_set = "Euler_test"
        physical_params = Dict(:K => 75.0, :Kvdiff => 0.0)
    else
        # The PE set has explicit horizontal diffusion and implicit vertical
        # diffusion: Khdiff = Kvdiff = 75 approximates the legacy uniform
        # Laplacian K = 75 of the paper specification
        equation_set = opts.stage == STAGE_PE_RHOD ? "primitive_equation_XZ_rhod" :
                                                     "primitive_equation_XZ"
        physical_params = Dict(:Khdiff => 75.0, :Kvdiff => 75.0, :Kv_mudiff => 0.0,
                               :alpha => 0.0, :z_damp => 12.8e3)
    end
    # Semi-implicit acoustics are validated for the linear rho_d set (Phase 2),
    # so enable them there; the xi stages stay explicit for this dry benchmark.
    options = Dict(:semiimplicit => opts.stage == STAGE_PE_RHOD,
                   :exact_reference_state => false)
    if opts.stage in (:pe, STAGE_PE_RHOD)
        # The paper prescribes uniform K = 75 only (carried by Khdiff/Kvdiff);
        # no precipitation or shear-based turbulence
        options[:precipitation] = false
        options[:vertical_mixing] = false
    end

    kDim = vertical_kdim(kDim, opts)
    ts = vertical_ts(ts, opts)

    output_dir = benchmark_output_dir("straka93", opts)
    scalar_bc = Dict(v => NeumannBC() for v in vars)
    bc_side = merge(scalar_bc, Dict("u" => DirichletBC()))   # no-normal-flow walls
    bc_topbot = merge(scalar_bc, Dict("w" => DirichletBC()))

    grid_params = GridParameters(
        geometry = benchmark_geometry(opts),   # RZ (Chebyshev) or RiRk (B-spline) vertical
        iMin = 0.0,
        iMax = 25.6e3,
        num_cells = num_cells,
        kMin = 0.0,
        kMax = 6.4e3,
        kDim = kDim,
        BCL = bc_side,
        BCR = bc_side,
        BCB = bc_topbot,
        BCT = bc_topbot,
        vars = Dict(v => i for (i, v) in enumerate(vars)),
    )

    return ModelParameters(
        ts = ts,
        integration_time = 900.0,
        output_interval = output_interval,
        equation_set = equation_set,
        initial_conditions = joinpath(output_dir, "straka93_ics.csv"),
        output_dir = output_dir,
        ref_state_file = joinpath(output_dir, "straka93.ref"),
        grid_params = grid_params,
        physical_params = physical_params,
        options = options,
    )
end

# ── Initial conditions ─────────────────────────────────────────────────────

"""Generate the reference sounding and cold bubble initial conditions."""
function straka_init!(model)
    Scythe.write_dry_sounding(model.ref_state_file; theta=300.0, zmax=8000.0)

    patch = createGrid(model.grid_params)
    gridpoints = Scythe.getGridpoints(patch)
    kDim = model.grid_params.kDim
    z = gridpoints[1:kDim, 2]
    column = Scythe.reference_column(patch, model.grid_params)
    ref = Scythe.calculate_reference_state(model, z, column)

    patch.physical .= 0.0
    Scythe.temperature_bubble!(patch, gridpoints, ref;
                               xc=0.0, xr=4000.0, zc=3000.0, zr=2000.0, dT_max=-15.0,
                               control = (opts.stage == STAGE_PE_RHOD ? :rhod : :xi))
    Scythe.write_ics_csv(model.initial_conditions, patch, gridpoints)
end

# ── Diagnostics ────────────────────────────────────────────────────────────

function straka_diagnostics(model)
    df = read_final_output(model)
    ref, _, kDim = rebuild_reference(model)
    theta_p, ncols = theta_perturbation(df, ref, kDim)
    r = reshape(df.r, kDim, ncols)[1, :]
    front = front_location(r, theta_p[1, :]; threshold=-1.0)
    diags = Dict(
        "min_theta_p" => minimum(theta_p),
        "max_theta_p" => maximum(theta_p),
        "max_u" => maximum(df.u),
        "min_u" => minimum(df.u),
        "max_w" => maximum(df.w),
        "min_w" => minimum(df.w),
        "front_location" => front,
    )
    return merge(diags, conservation_drift(model, ref))
end

# ── Figures ────────────────────────────────────────────────────────────────

plotter = nothing
if opts.plot
    include(joinpath(@__DIR__, "common", "plots.jl"))
    plotter = function (model)
        df = read_final_output(model)
        ref, _, kDim = rebuild_reference(model)
        theta_p, ncols = theta_perturbation(df, ref, kDim)
        x = reshape(df.r, kDim, ncols)[1, :]
        z = reshape(df.z, kDim, ncols)[:, 1]
        w = reshape(df.w, kDim, ncols)
        save_benchmark_figure(
            joinpath(model.output_dir, "straka93_$(opts.mode)_$(opts.stage)_final.png"),
            x, z,
            [(theta_p, "θ′ (K)", -15.5:1.0:-0.5),
             (w, "w (m/s)", -16.0:2.0:14.0)];
            title = "Straka93 density current, t = $(model.integration_time) s")
    end
end

# ── Run ────────────────────────────────────────────────────────────────────

model = straka_model(opts)
passed = run_benchmark("straka93", opts;
                       model = model,
                       init! = straka_init!,
                       diagnostics = straka_diagnostics,
                       varnames = straka_vars(opts.stage),
                       plotter = plotter)
exit(passed ? 0 : 1)
