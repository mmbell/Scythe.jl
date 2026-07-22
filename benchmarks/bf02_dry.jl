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
# Modes: quick (400 m cells) | full (100 m cells, paper-grade)
# Stages: legacy (Euler_test) | pe (primitive_equation_XZ) | pe-rho_d | pe-rho_d-pd |
#         pe-sigma | mc (total-energy set; the physical-density sets run dry here, q_v = 0)
#
# Reference: Bryan & Fritsch (2002), Mon. Wea. Rev. 130, 2917-2928.
# reference/bryan_fritsch_mwr2002.pdf

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
# Partial-density and entropy-density variants (run dry here: q_v = 0). Slot 1 is the
# intensive entropy "s" (pd) or the entropy density "sigma" (moist-compressible).
const PE_VARS_PD = ["s", "rho_d", "rho_v", "u", "w", "rho_c", "rho_r", "mu_sat"]
const PE_VARS_SIGMA = ["sigma", "rho_d", "rho_v", "u", "w", "rho_c", "rho_r", "mu_sat"]
# Total-energy variant (runs dry here: rho_t = rho_d, Q_ss tracks -rho_v_sat)
const MC_VARS = ["p", "rho_d", "rho_t", "u", "w", "E_t", "Q_ss", "rho_r", "rho_c"]
function bf02_dry_vars(stage)
    stage == :legacy && return ["s", "xi", "mu", "u", "w"]
    stage == STAGE_MC && return MC_VARS
    stage == STAGE_PE_SIGMA && return PE_VARS_SIGMA
    stage == STAGE_PE_RHOD_PD && return PE_VARS_PD
    stage == STAGE_PE_RHOD && return PE_VARS_RHOD
    return PE_VARS
end

function bf02_dry_model(opts::BenchmarkOptions)
    # The B-spline (RiRk) vertical is sized by cell count, the Chebyshev (RZ) vertical by kDim
    # (see `vertical_size` in common/harness.jl). Here they coincide -- 100 cells == kDim 300 and
    # 25 cells == kDim 75 -- so both geometries keep exactly the grid they had before.
    if opts.mode == :full
        num_cells_i = 200       # 100 m cells
        num_cells_k = 100       # RiRk: 100 m cells
        kDim = 300              # RZ: Chebyshev points
        ts = 0.025 #0.1/2.0
        output_interval = 100.0
    else
        num_cells_i = 50        # 400 m cells
        num_cells_k = 25        # RiRk: 400 m cells
        kDim = 75               # RZ: Chebyshev points
        ts = 0.1
        output_interval = 500.0
    end

    vars = bf02_dry_vars(opts.stage)
    physical_stage = Scythe.uses_physical_reference(
        opts.stage == STAGE_PE_SIGMA ? "primitive_equation_XZ_sigma" :
        opts.stage == STAGE_PE_RHOD_PD ? "primitive_equation_XZ_rhod_pd" : "")
    if opts.stage == :legacy
        # The dry case carries no liquid water, so the 5-variable Euler_test
        # set is physically identical to the historical 6-variable BF02 run
        equation_set = "Euler_test"
        physical_params = Dict(:K => 0.0, :Kvdiff => 0.0)
    else
        # No physical or computational diffusion, matching the paper; the
        # Rayleigh damping is disabled with alpha = 0
        equation_set = opts.stage == STAGE_MC ? "moist_compressible_XZ" :
            opts.stage == STAGE_PE_SIGMA ? "primitive_equation_XZ_sigma" :
            opts.stage == STAGE_PE_RHOD_PD ? "primitive_equation_XZ_rhod_pd" :
            opts.stage == STAGE_PE_RHOD ? "primitive_equation_XZ_rhod" :
            "primitive_equation_XZ"
        physical_params = Dict(:Khdiff => 0.0, :Kvdiff => 0.0, :Kv_mudiff => 0.0,
                               :tau_qss => 10.0,
                               :alpha => 0.0, :z_damp => 20.0e3)
    end
    # The physical-density sets read a physical (Springsteel) reference written as an exact
    # dry profile (rho_v = rho_c = 0), and the total-energy set a pressure-based one;
    # the legacy/pe/pe-rho_d sets build the xi/mu reference.
    exact_ref = physical_stage || opts.stage == STAGE_MC
    options = Dict(:semiimplicit => true, :exact_reference_state => exact_ref)
    if opts.stage in (:pe, STAGE_PE_RHOD, STAGE_PE_RHOD_PD, STAGE_PE_SIGMA, STAGE_MC)
        # Benchmark specification has no turbulence or precipitation
        options[:precipitation] = false
        options[:vertical_mixing] = false
        options[:horizontal_semiimplicit] = opts.hsi
        options[:exact_si] = opts.xsi
    end

    ts = vertical_ts(ts, opts)

    output_dir = benchmark_output_dir("bf02_dry", opts)
    scalar_bc = Dict(v => NeumannBC() for v in vars)
    wall_bc = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))

    grid_params = GridParameters(;
        geometry = benchmark_geometry(opts),   # RZ (Chebyshev) or RiRk (B-spline) vertical
        iMin = 0.0,
        iMax = 20.0e3,
        num_cells_i = num_cells_i,
        kMin = 0.0,
        kMax = 10.0e3,
        vertical_size(opts; num_cells_k = num_cells_k, kDim = kDim)...,
        BCL = wall_bc,
        BCR = wall_bc,
        BCB = wall_bc,
        BCT = wall_bc,
        vars = Dict(v => i for (i, v) in enumerate(vars)),
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
        options = options,
    )
end

# ── Initial conditions ─────────────────────────────────────────────────────

"""Generate the reference sounding and warm bubble initial conditions."""
function bf02_dry_init!(model)
    patch = createGrid(model.grid_params)
    gridpoints = Scythe.getGridpoints(patch)
    kDim = model.grid_params.kDim
    z = gridpoints[1:kDim, 2]
    column = Scythe.reference_column(patch, model.grid_params)
    patch.physical .= 0.0

    if Scythe.uses_pressure_reference(model.equation_set)
        # Total-energy set: the dry neutral base is the analytic constant-theta Exner
        # profile, so write it exactly (hydrostatic dp/dz = -rho*g to machine precision
        # analytically; rho_v = rho_c = 0) rather than via the legacy sounding builder,
        # whose hydrostatic iteration does not converge on the low-DOF B-spline (RiRk)
        # column (it lands on the correct adiabat displaced in pressure).
        theta0 = 300.0
        exner = @. 1.0 - (Scythe.gravity * z) / (Scythe.Cpd * theta0)
        T_prof = theta0 .* exner
        p_prof = @. 100000.0 * exner^(Scythe.Cpd / Scythe.Rd)   # 1000 hPa surface, Pa
        rho_d_prof = p_prof ./ (Scythe.Rd .* T_prof)
        zeros_prof = zeros(Float64, kDim)
        Scythe.write_exact_ref_mc(model.ref_state_file, z, p_prof, rho_d_prof,
                                  zeros_prof, zeros_prof)
        ref = Springsteel.exact_pressure_reference_state(model.ref_state_file, z, column)
        Scythe.theta_bubble_mc!(patch, gridpoints, ref;
                                xc=10000.0, xr=2000.0, zc=2000.0, zr=2000.0, dtheta_max=2.0)
    elseif Scythe.uses_physical_reference(model.equation_set)
        # Physical-density sets need a physical (Springsteel) reference. Build the balanced
        # dry profile with the legacy builder, then write it as an exact physical reference
        # with zero moisture (rho_v = rho_c = 0) and seed the dry physical-density bubble.
        sounding = joinpath(model.output_dir, "dry_sounding.ref")
        Scythe.write_dry_sounding(sounding; theta=300.0, zmax=12000.0)
        guess = ModelParameters(ts = model.ts, equation_set = model.equation_set,
                                ref_state_file = sounding, grid_params = model.grid_params,
                                physical_params = model.physical_params)
        legacy_ref = Scythe.calculate_reference_state(guess, z, column)
        s_prof = Scythe.ref_entropy(legacy_ref)[:, 1]
        rho_d_prof = Scythe.ref_rho_d(legacy_ref)[:, 1]
        zeros_prof = zeros(Float64, kDim)
        Scythe.write_exact_ref_pd(model.ref_state_file, z, s_prof, rho_d_prof,
                                  zeros_prof, zeros_prof)
        ref = Springsteel.exact_reference_state(model.ref_state_file, z, column)
        Scythe.theta_bubble_pd!(patch, gridpoints, ref;
                                xc=10000.0, xr=2000.0, zc=2000.0, zr=2000.0, dtheta_max=2.0)
    else
        Scythe.write_dry_sounding(model.ref_state_file; theta=300.0, zmax=12000.0)
        ref = Scythe.calculate_reference_state(model, z, column)
        Scythe.theta_bubble!(patch, gridpoints, ref;
                             xc=10000.0, xr=2000.0, zc=2000.0, zr=2000.0, dtheta_max=2.0,
                             control = (opts.stage == STAGE_PE_RHOD ? :rhod : :xi))
    end
    Scythe.write_ics_csv(model.initial_conditions, patch, gridpoints)
end

# ── Diagnostics ────────────────────────────────────────────────────────────

function bf02_dry_diagnostics(model)
    df = read_final_output(model)
    ref, _, kDim = rebuild_reference(model)
    theta_p, _ = theta_perturbation(df, ref, kDim)
    diags = Dict(
        "max_theta_p" => maximum(theta_p),
        "min_theta_p" => minimum(theta_p),
        "max_w" => maximum(df.w),
        "min_w" => minimum(df.w),
        "max_u" => maximum(df.u),
        "min_u" => minimum(df.u),
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
            joinpath(model.output_dir, "bf02_dry_$(opts.mode)_$(opts.stage)_final.png"),
            x, z,
            [(theta_p, "θ′ (K)", -0.2:0.2:2.2),
             (w, "w (m/s)", -10.0:2.0:16.0)];
            title = "BF02 dry thermal, t = $(model.integration_time) s")
    end
end

# ── Run ────────────────────────────────────────────────────────────────────

model = bf02_dry_model(opts)
passed = run_benchmark("bf02_dry", opts;
                       model = model,
                       init! = bf02_dry_init!,
                       diagnostics = bf02_dry_diagnostics,
                       varnames = bf02_dry_vars(opts.stage),
                       plotter = plotter)
exit(passed ? 0 : 1)
