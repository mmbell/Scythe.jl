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
# Modes: quick (200 m cells, regression-sized) | full (50 m cells, paper-grade)
# Stages: legacy (Euler_test) | pe (primitive_equation_XZ) | pe-rho_d | mc
#         (moist_compressible_XZ, the total-energy set)
#
# Reference: Straka, Wilhelmson, Wicker, Anderson, Droegemeier (1993),
# Int. J. Numer. Methods Fluids 17, 1-22. reference/Straka.pdf

using Distributed

include(joinpath(@__DIR__, "common", "harness.jl"))
opts = parse_benchmark_args(ARGS)

add_benchmark_workers(opts)
@everywhere using Springsteel
@everywhere using Scythe

include(joinpath(@__DIR__, "common", "diagnostics.jl"))

# ── Configuration ──────────────────────────────────────────────────────────

const PE_VARS = ["s", "xi", "mu", "u", "w", "mu_c", "mu_r", "mu_sat"]
# Linear dry-air-density variant: slot 2 is "rho_d" (rho_d') instead of "xi"
const PE_VARS_RHOD = ["s", "rho_d", "mu", "u", "w", "mu_c", "mu_r", "mu_sat"]
# Total-energy (moist_compressible) set
const MC_VARS = ["p", "rho_d", "rho_t", "u", "w", "E_t", "Q_ss", "rho_r", "rho_c",
                 "rho_v"]
function straka_vars(stage)
    stage == :legacy && return ["s", "xi", "mu", "u", "w"]
    stage == STAGE_PE_RHOD && return PE_VARS_RHOD
    stage == STAGE_MC && return MC_VARS
    return PE_VARS
end

function straka_model(opts::BenchmarkOptions)
    # Horizontal and the B-spline (RiRk) vertical are sized by cell count, at a 1:1 aspect ratio
    # (DX_i == DX_k) over the 25.6 x 6.4 km domain: 512x128 -> 50 m cells, 128x32 -> 200 m cells.
    #
    # The Chebyshev (RZ) vertical is sized independently by `kDim`, and is NOT tied to
    # num_cells_k: its points cluster at the walls, so dz_min shrinks roughly as 1/kDim^2 and
    # the explicit acoustic CFL tightens quadratically. The legacy and pe stages integrate
    # explicitly on RZ, and sizing them to 3*num_cells_k = 96 rather than 64 makes them go
    # non-finite at t = 0.688 s. Raising RZ kDim means cutting ts to match. See `vertical_size`
    # in common/harness.jl.
    if opts.mode == :full
        num_cells_i = 512       # 50 m cells
        num_cells_k = 128       # RiRk: 50 m cells
        kDim = 384              # RZ: Chebyshev points (full runs the semi-implicit mc stage)
        ts = 0.015625
        output_interval = 100.0
    else
        num_cells_i = 128       # 200 m cells
        num_cells_k = 32        # RiRk: 200 m cells
        kDim = 64               # RZ: Chebyshev points; ts = 0.0625 is stable for explicit stages
        ts = 0.0625
        output_interval = 300.0
    end

    vars = straka_vars(opts.stage)
    if opts.stage == :legacy
        equation_set = "Euler_test"
        physical_params = Dict(:K => 75.0, :Kvdiff => 0.0)
    elseif opts.stage == STAGE_MC
        # The total-energy set diffuses u, w and the dry entropy s_d (no mass diffusion, as
        # in the paper). Kvdiff_heat defaults to Kvdiff, giving the paper's single K on
        # momentum and entropy.
        # Momentum diffusion is a resolved-KE sink to the subgrid (not dissipative heating,
        # matching Straka): E_t follows the KE down and T is held. The thermal (entropy)
        # diffusion is a genuine O(K) energy source, since rho*T*Lap(s) is not a flux
        # divergence. Both are expected; full conservation returns with a prognostic-TKE
        # closure. See reference/moist_compressible_diffusion_plan.md.
        equation_set = "moist_compressible_XZ"
        physical_params = Dict(:Khdiff => 75.0, :Kvdiff => 75.0, :Kv_mudiff => 0.0,
                               :tau_qss => 10.0,
                               :alpha => 0.0, :z_damp => 12.8e3)
    else
        # The PE set has explicit horizontal diffusion and implicit vertical
        # diffusion: Khdiff = Kvdiff = 75 approximates the legacy uniform
        # Laplacian K = 75 of the paper specification
        equation_set = opts.stage == STAGE_PE_RHOD ? "primitive_equation_XZ_rhod" :
                                                     "primitive_equation_XZ"
        physical_params = Dict(:Khdiff => 75.0, :Kvdiff => 75.0, :Kv_mudiff => 0.0,
                               :alpha => 0.0, :z_damp => 12.8e3)
    end
    # Semi-implicit acoustics are validated for the linear rho_d set (Phase 2) and the
    # total-energy set, so enable them there; the xi stages stay explicit for this dry
    # benchmark. The mc set consumes an exact pressure-based reference state.
    options = Dict{Symbol,Any}(:semiimplicit => opts.stage in (STAGE_PE_RHOD, STAGE_MC),
                   :exact_reference_state => opts.stage == STAGE_MC,
                   :output_formats => BENCHMARK_OUTPUT_FORMATS)
    if opts.stage in (:pe, STAGE_PE_RHOD, STAGE_MC)
        # The paper prescribes uniform K = 75 only (carried by Khdiff/Kvdiff);
        # no precipitation or shear-based turbulence
        options[:precipitation] = false
        options[:vertical_mixing] = false
    end

    ts = vertical_ts(ts, opts)

    output_dir = benchmark_output_dir("straka93", opts)
    scalar_bc = Dict(v => NeumannBC() for v in vars)
    bc_side = merge(scalar_bc, Dict("u" => DirichletBC()))   # no-normal-flow walls
    bc_topbot = merge(scalar_bc, Dict("w" => DirichletBC()))

    grid_params = GridParameters(;
        geometry = benchmark_geometry(opts),   # RZ (Chebyshev) or RiRk (B-spline) vertical
        iMin = 0.0,
        iMax = 25.6e3,
        num_cells_i = num_cells_i,
        kMin = 0.0,
        kMax = 6.4e3,
        vertical_size(opts; num_cells_k = num_cells_k, kDim = kDim)...,
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
    patch = createGrid(model.grid_params)
    gridpoints = Scythe.getGridpoints(patch)
    kDim = model.grid_params.kDim
    z = gridpoints[1:kDim, 2]
    column = Scythe.reference_column(patch, model.grid_params)
    patch.physical .= 0.0

    if Scythe.uses_pressure_reference(model.equation_set)
        # Straka's base state IS a constant-theta = 300 K dry adiabat, so it is the same
        # analytic Exner profile bf02_dry uses for the total-energy set -- written exactly
        # rather than via the legacy hydrostatic iteration, which does not converge on the
        # low-DOF B-spline (RiRk) column.
        theta0 = 300.0
        exner = @. 1.0 - (Scythe.gravity * z) / (Scythe.Cpd * theta0)
        T_prof = theta0 .* exner
        p_prof = @. 100000.0 * exner^(Scythe.Cpd / Scythe.Rd)   # 1000 hPa surface, Pa
        rho_d_prof = p_prof ./ (Scythe.Rd .* T_prof)
        zeros_prof = zeros(Float64, kDim)
        Scythe.write_exact_ref_mc(model.ref_state_file, z, p_prof, rho_d_prof,
                                  zeros_prof, zeros_prof)
        ref = Springsteel.exact_pressure_reference_state(model.ref_state_file, z, column)
        Scythe.temperature_bubble_mc!(patch, gridpoints, ref;
                                      xc=0.0, xr=4000.0, zc=3000.0, zr=2000.0, dT_max=-15.0)
    else
        Scythe.write_dry_sounding(model.ref_state_file; theta=300.0, zmax=8000.0)
        ref = Scythe.calculate_reference_state(model, z, column)
        Scythe.temperature_bubble!(patch, gridpoints, ref;
                                   xc=0.0, xr=4000.0, zc=3000.0, zr=2000.0, dT_max=-15.0,
                                   control = (opts.stage == STAGE_PE_RHOD ? :rhod : :xi))
    end
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
