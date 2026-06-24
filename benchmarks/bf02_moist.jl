#!/usr/bin/env julia
# Bryan & Fritsch (2002) moist warm bubble benchmark.
#
# A buoyancy perturbation identical to the dry case rises through a saturated,
# exactly neutrally stable atmosphere (uniform reversible theta_e = 320 K,
# constant total water r_t = 0.020) with reversible phase changes and no
# precipitation. Verified against the theta_e' and w extrema printed in
# Fig. 3 of the paper.
#
# Note on physics: this configuration uses the BF02_test equation set with a
# prognostic supersaturation variable (qss) rather than the paper's strict
# instantaneous saturation adjustment. Allowing supersaturation production/
# consumption softens the latent heat release slightly, so the w and theta_e'
# extrema are expected to sit a little below the published values. The total
# entropy (prognostic dry+vapor entropy plus condensate entropy) should still
# be conserved; see the conservation diagnostics.
#
#   julia --project=. benchmarks/bf02_moist.jl --mode quick --stage legacy
#
# Modes: quick (200 m cells) | full (100 m cells, paper-grade)
# Stages: legacy (BF02_test) | pe (primitive_equation_XZ)
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

const PE_VARS = ["s", "xi", "mu", "u", "w", "mu_c", "mu_r", "mu_sat"]
# Linear dry-air-density variant: slot 2 is "rho_d" (rho_d') instead of "xi"
const PE_VARS_RHOD = ["s", "rho_d", "mu", "u", "w", "mu_c", "mu_r", "mu_sat"]
function bf02_moist_vars(stage)
    stage == :legacy && return ["s", "xi", "mu", "u", "w", "mu_l", "qss"]
    stage == :perhod && return PE_VARS_RHOD
    return PE_VARS
end
# Liquid water variable(s) per stage (PE splits liquid into cloud and rain)
liquid_vars(stage) = stage == :legacy ? ["mu_l"] : ["mu_c", "mu_r"]
const Q_T = 0.02
const THETA_E = 320.0

function bf02_moist_model(opts::BenchmarkOptions)
    if opts.mode == :full
        num_cells = 200         # 100 m cells
        kDim = 100
        ts = 0.1
        output_interval = 100.0
    else
        num_cells = 100         # 200 m cells
        kDim = 50
        # The condensation relaxation timescale does not coarsen with the
        # grid, so quick mode keeps the full-mode timestep
        ts = 0.1
        output_interval = 250.0
    end

    vars = bf02_moist_vars(opts.stage)
    if opts.stage == :legacy
        equation_set = "BF02_test"
        physical_params = Dict(:K => 0.0, :Kvdiff => 0.0)
        options = Dict(:semiimplicit => true, :exact_reference_state => true)
    elseif opts.stage == :perhod
        # Linear dry-air-density prognostic variant (mass-conserving continuity)
        # with semi-implicit acoustics on the mass flux phi = rhobar_d * w
        # (Phase 2; see reference/Semiimplicit_linear_rhod.tex).
        equation_set = "primitive_equation_XZ_rhod"
        physical_params = Dict(:Khdiff => 0.0, :Kvdiff => 0.0, :Kv_mudiff => 0.0,
                               :alpha => 0.0, :z_damp => 20.0e3)
        options = Dict(:semiimplicit => true, :exact_reference_state => true,
                       :precipitation => false, :vertical_mixing => false)
    else
        equation_set = "primitive_equation_XZ"
        physical_params = Dict(:Khdiff => 0.0, :Kvdiff => 0.0, :Kv_mudiff => 0.0,
                               :alpha => 0.0, :z_damp => 20.0e3)
        # Reversible benchmark: no precipitation fallout (BF02 spec). The
        # saturated base state carries cloud water everywhere, so
        # autoconversion would otherwise generate rain domain-wide.
        options = Dict(:semiimplicit => true, :exact_reference_state => true,
                       :precipitation => false, :vertical_mixing => false)
    end

    kDim = vertical_kdim(kDim, opts)
    ts = vertical_ts(ts, opts)

    output_dir = benchmark_output_dir("bf02_moist", opts)
    scalar_bc = Dict(v => NeumannBC() for v in vars)
    wall_bc = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))

    grid_params = GridParameters(
        geometry = benchmark_geometry(opts),   # RZ (Chebyshev) or RiRk (B-spline) vertical
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
        vars = Dict(v => i for (i, v) in enumerate(vars)),
    )

    return ModelParameters(
        ts = ts,
        integration_time = 1000.0,
        output_interval = output_interval,
        equation_set = equation_set,
        initial_conditions = joinpath(output_dir, "bf02_moist_ics.csv"),
        output_dir = output_dir,
        ref_state_file = joinpath(output_dir, "bf02_moist_exact.ref"),
        grid_params = grid_params,
        physical_params = physical_params,
        # The converged saturated base state is written as an exact reference
        # at the model levels, so the model integrates pure perturbations
        options = options,
    )
end

# ── Initial conditions ─────────────────────────────────────────────────────

"""
Construct the saturated neutrally stable base state, write it as the model's
exact reference, and add the moist buoyancy bubble.
"""
function bf02_moist_init!(model)
    patch = createGrid(model.grid_params)
    gridpoints = Scythe.getGridpoints(patch)
    kDim = model.grid_params.kDim
    z = gridpoints[1:kDim, 2]
    column = Scythe.reference_column(patch, model.grid_params)

    # First-guess sounding -> hydrostatic reference -> spectral refinement
    sounding = joinpath(model.output_dir, "first_guess_sounding.ref")
    Scythe.write_moist_neutral_sounding(sounding; q_t=Q_T, theta_e=THETA_E,
                                        sfc_p_hPa=1000.0, zmax=12000.0)
    guess_model = ModelParameters(
        ts = model.ts, equation_set = model.equation_set,
        ref_state_file = sounding, grid_params = model.grid_params,
        physical_params = model.physical_params,
    )
    if column isa Spline1D
        # The saturated-neutral base-state iteration converges on a Chebyshev
        # column but not on the low-DOF cubic B-spline column, so build it on a
        # Chebyshev column (at the resolution the RZ moist case uses, where it is
        # known to converge) and spectrally interpolate the converged profiles
        # onto the spline model levels for a stable initial condition.
        kDim_cheb = opts.mode == :full ? 100 : 50
        b_cheb = min(kDim_cheb, Int(floor((2 * kDim_cheb - 1) / 3)) + 1)
        cheb = Chebyshev1D(ChebyshevParameters(
            zmin = model.grid_params.kMin, zmax = model.grid_params.kMax,
            zDim = kDim_cheb, bDim = b_cheb,
            BCB = Chebyshev.R0, BCT = Chebyshev.R0))
        z_cheb = cheb.mishPoints
        ref_guess = Scythe.calculate_reference_state(guess_model, z_cheb, cheb)
        base_cheb = Scythe.saturated_hydrostatic_profile(z_cheb, cheb, ref_guess;
                                                         q_t=Q_T, theta_e=THETA_E,
                                                         sfc_p_hPa=1000.0)
        base = Scythe.interpolate_base_state(base_cheb, cheb, z)
    else
        ref_guess = Scythe.calculate_reference_state(guess_model, z, column)
        base = Scythe.saturated_hydrostatic_profile(z, column, ref_guess;
                                                    q_t=Q_T, theta_e=THETA_E,
                                                    sfc_p_hPa=1000.0)
    end
    println("Base state: max|q_v - q_sat| = ",
            maximum(abs.(base.q_v .- Scythe.q_sat_liquid.(base.Tk, base.p))),
            ", max hydrostatic residual = ", maximum(abs.(base.residual)), " m/s²")

    # The model integrates perturbations from the converged base state
    Scythe.write_exact_ref(model.ref_state_file, z, base.s, base.xi, base.mu)
    ref = Scythe.exact_reference_state(model, z, column)

    patch.physical .= 0.0
    bubble_liquid = opts.stage == :legacy ? "mu_l" : "mu_c"
    Scythe.moist_buoyancy_bubble!(patch, gridpoints, base, ref;
                                  q_t=Q_T, xc=10000.0, xr=2000.0,
                                  zc=2000.0, zr=2000.0, amp=2.0/300.0,
                                  liquid_var=bubble_liquid,
                                  control = (opts.stage == :perhod ? :rhod : :xi))

    if opts.stage in (:pe, :perhod)
        # The PE set advects a transformed saturation ratio with satbar = 0
        # for exact reference states: initialize it from the actual state
        # (the base is saturated, so the ratio is 1 everywhere up to the
        # construction tolerance)
        vars = model.grid_params.vars
        sat_i = vars["mu_sat"]
        kDim_l = model.grid_params.kDim
        rhod_stage = opts.stage == :perhod
        dens_i = rhod_stage ? vars["rho_d"] : vars["xi"]
        i = 1
        for _ in 1:Scythe.num_columns(patch)
            for k in 1:kDim_l
                s_tot = patch.physical[i, vars["s"], 1] + ref.sbar[k, 1]
                mu_tot = patch.physical[i, vars["mu"], 1] + ref.mubar[k, 1]
                if rhod_stage
                    rho_d_tot = patch.physical[i, dens_i, 1] + ref.rhobar[k, 1]
                    q_v, _, Tk, p = Scythe.thermodynamic_tuple_rhod(s_tot, rho_d_tot, mu_tot)
                else
                    xi_tot = patch.physical[i, dens_i, 1] + ref.xibar[k, 1]
                    q_v, _, Tk, p = Scythe.thermodynamic_tuple(s_tot, xi_tot, mu_tot)
                end
                patch.physical[i, sat_i, 1] =
                    Scythe.mu_transform(q_v / Scythe.q_sat_liquid(Tk, p))
                i += 1
            end
        end
    end
    Scythe.write_ics_csv(model.initial_conditions, patch, gridpoints)

    # Save the base profile for diagnostics (theta_e perturbation baseline)
    CSV.write(joinpath(model.output_dir, "base_profile.csv"),
              DataFrame(z = z, s = base.s, xi = base.xi, mu = base.mu,
                        mu_l = base.mu_l, theta_e = base.theta_e))
end

# ── Diagnostics ────────────────────────────────────────────────────────────

# theta_e' contour (K) defining the rising thermal's cap, used for the bubble-top
# height diagnostic. BF02 Fig. 3a's thermal reaches ~8.2 km; adjust this contour
# to match the published definition if the reported height looks off.
const THETA_E_BUBBLE_THRESHOLD = 0.5

"""
Maximum altitude (m) at which the theta_e' perturbation exceeds
`threshold` — i.e. how high the rising thermal cap has reached.
`theta_e_p` is `(kDim, ncols)` and `z` is the per-level height vector.
"""
function theta_e_bubble_height(theta_e_p, z; threshold=THETA_E_BUBBLE_THRESHOLD)
    h = 0.0
    for k in 1:size(theta_e_p, 1)
        if any(>(threshold), view(theta_e_p, k, :))
            h = max(h, z[k])
        end
    end
    return h
end

"""Reconstruct theta_e' and supersaturation fields from an output DataFrame."""
function moist_fields(df, ref, kDim, base, stage)
    ncols = div(nrow(df), kDim)
    sbar = repeat(ref.sbar[:, 1], ncols)
    mubar = repeat(ref.mubar[:, 1], ncols)
    s = df.s .+ sbar
    mu = df.mu .+ mubar
    # Recover log-density xi from whichever control variable the run stored
    if "rho_d" in names(df)
        xi = Scythe.log_dry_density.(df.rho_d .+ repeat(ref.rhobar[:, 1], ncols))
    else
        xi = df.xi .+ repeat(ref.xibar[:, 1], ncols)
    end
    # Total transformed liquid: the linear mu transform makes the sum of
    # transformed cloud and rain equal the transform of their sum
    mu_liq = zero(mu)
    for lv in liquid_vars(stage)
        mu_liq = mu_liq .+ df[!, lv]
    end
    theta_e = Scythe.reversible_theta_e.(s, xi, mu, mu_liq)
    theta_e_p = reshape(theta_e .- repeat(base.theta_e, ncols), kDim, ncols)
    thermo = Scythe.thermodynamic_tuple.(s, xi, mu)
    q_v = [x[1] for x in thermo]
    Tk = [x[3] for x in thermo]
    p = [x[4] for x in thermo]
    supersat = reshape(q_v .- Scythe.q_sat_liquid.(Tk, p), kDim, ncols)
    return theta_e_p, supersat, ncols
end

function bf02_moist_diagnostics(model)
    df = read_final_output(model)
    ref, _, kDim = rebuild_reference(model)
    base = CSV.read(joinpath(model.output_dir, "base_profile.csv"), DataFrame)
    theta_e_p, supersat, ncols = moist_fields(df, ref, kDim, base, opts.stage)
    z = reshape(df.z, kDim, ncols)[:, 1]
    diags = Dict(
        "max_theta_e_p" => maximum(theta_e_p),
        "min_theta_e_p" => minimum(theta_e_p),
        "max_w" => maximum(df.w),
        "min_w" => minimum(df.w),
        "max_supersat" => maximum(supersat),
        "min_supersat" => minimum(supersat),
        "theta_e_bubble_top_km" => theta_e_bubble_height(theta_e_p, z) / 1000.0,
    )
    return merge(diags, conservation_drift(model, ref; liquid_vars=liquid_vars(opts.stage)))
end

# ── Figures ────────────────────────────────────────────────────────────────

plotter = nothing
if opts.plot
    include(joinpath(@__DIR__, "common", "plots.jl"))
    plotter = function (model)
        df = read_final_output(model)
        ref, _, kDim = rebuild_reference(model)
        base = CSV.read(joinpath(model.output_dir, "base_profile.csv"), DataFrame)
        theta_e_p, _, ncols = moist_fields(df, ref, kDim, base, opts.stage)
        x = reshape(df.r, kDim, ncols)[1, :]
        z = reshape(df.z, kDim, ncols)[:, 1]
        w = reshape(df.w, kDim, ncols)
        save_benchmark_figure(
            joinpath(model.output_dir, "bf02_moist_$(opts.mode)_$(opts.stage)_final.png"),
            x, z,
            [(theta_e_p, "θ_e′ (K)", -0.5:0.5:4.5),
             (w, "w (m/s)", -10.0:2.0:16.0)];
            title = "BF02 moist thermal, t = $(model.integration_time) s")
    end
end

# ── Run ────────────────────────────────────────────────────────────────────

model = bf02_moist_model(opts)
passed = run_benchmark("bf02_moist", opts;
                       model = model,
                       init! = bf02_moist_init!,
                       diagnostics = bf02_moist_diagnostics,
                       varnames = bf02_moist_vars(opts.stage),
                       plotter = plotter)
exit(passed ? 0 : 1)
