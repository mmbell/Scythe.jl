#!/usr/bin/env julia
# Ooyama (2001)-style warm-rain benchmark for the total-energy set.
#
# A 3 K RH-preserving warm bubble (cos² profile, 16 km x 3 km radii, centered
# at 500 m over the domain midpoint) rises through the Dunion moist-tropical
# sounding, condenses, converts cloud to rain (autoconversion + collection),
# and rains out through the surface: the rho_r bottom boundary is a free
# (Natural) fit so the sedimentation flux removes water and its energy through
# z = 0. Rain condensation/evaporation runs through the tau_r channel of the
# generalized supersaturation relaxation (1/tau = 1/tau_c + 1/tau_r) with the
# monodisperse N_r closure — no separate Qevap parameterization (this differs
# from Ooyama 2001, as do the equation set and the single non-nested grid with
# the bubble at the domain center, so the targets are comparable-magnitude
# sanity windows, not a reproduction of his figures).
#
#   julia --project=. benchmarks/o01_rainfall.jl --mode quick --stage mc --grid rirk
#
# Modes: quick (2 km cells) | full (500 m cells). Both use 500 m vertical
# nodal spacing to the 20 km rigid lid (no sponge; late-hour reflections are
# accepted — see the plan in the repo docs). Only --stage mc is supported.
#
# Reference: Ooyama (2001), J. Atmos. Sci. 58, 2073-2102 (Fig. 6: peak ground
# precipitation ~75-125 g m^-2 s^-1 at ~35-40 min for a similar bubble).
# reference/ooyama_jas2001.pdf

using Distributed

include(joinpath(@__DIR__, "common", "harness.jl"))
opts = parse_benchmark_args(ARGS)
opts.stage == STAGE_MC ||
    error("o01_rainfall supports only --stage mc (the total-energy set carries rho_r)")

add_benchmark_workers(opts)
@everywhere using Springsteel
@everywhere using Scythe

include(joinpath(@__DIR__, "common", "diagnostics.jl"))

# ── Configuration ──────────────────────────────────────────────────────────

const MC_VARS = ["p", "rho_d", "rho_t", "u", "w", "E_t", "Q_ss", "rho_r"]

# Dunion moist-tropical sounding (WRF input_sounding format), committed beside the
# reference data so the benchmark is self-contained.
const DUNION_SOUNDING = joinpath(REFERENCE_DATA_DIR, "o01_rainfall", "dunion_MT.ref")

# Rain-drop number concentration [#/cm^3] for the monodisperse tau_r closure
# (~1000 drops per m^3; the small number keeps the bulk of condensation on cloud).
const N_R = 1.0e-3

# Vertical eddy coefficients [m^2/s]. Kept small and equal across momentum, heat
# and water — the goal is the near-inviscid solution; these are the minimum found
# stable, revisit downward before re-seeding references.
const KV_MOM = 25.0
const KV_HEAT = 25.0
const KV_WATER = 25.0

function o01_model(opts::BenchmarkOptions)
    if opts.mode == :full
        num_cells_i = 300       # 500 m cells over 150 km
        ts = 0.15
    else
        num_cells_i = 75        # 2 km cells
        ts = 0.3
    end
    # Vertical: 500 m nodal spacing to 20 km in BOTH modes (the rain physics and
    # the sedimentation flux do not coarsen with the horizontal grid).
    num_cells_k = 40            # RiRk: 500 m cells
    kDim = 120                  # RZ: Chebyshev points (untargeted fallback)
    output_interval = 60.0

    ts = vertical_ts(ts, opts)

    vars = MC_VARS
    physical_params = Dict(:Khdiff => 0.0, :Kvdiff => KV_MOM,
                           :Khdiff_heat => 0.0, :Kvdiff_heat => KV_HEAT,
                           :Kvdiff_water => KV_WATER,
                           :tau_qss => 10.0, :N_r => N_R)
    options = Dict(:semiimplicit => true, :exact_reference_state => true,
                   :precipitation => true, :vertical_mixing => false)

    output_dir = benchmark_output_dir("o01_rainfall", opts)
    scalar_bc = Dict(v => NeumannBC() for v in vars)
    # Side walls: no normal flow (reflective); domain is wide enough that the
    # bubble's gravity waves arrive late.
    side_bc = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))
    # Top/bottom: rigid lids (w = 0), free-slip u. rho_r takes a FREE (Natural)
    # fit at both: a Neumann fit would force a zero boundary flux derivative and
    # trap the falling rain at the surface instead of letting the sedimentation
    # flux divergence remove it through z = 0.
    topbot_bc = merge(scalar_bc, Dict("w" => DirichletBC(), "rho_r" => NaturalBC()))

    grid_params = GridParameters(;
        geometry = benchmark_geometry(opts),   # RZ (Chebyshev) or RiRk (B-spline) vertical
        iMin = 0.0,
        iMax = 150.0e3,
        num_cells_i = num_cells_i,
        kMin = 0.0,
        kMax = 20.0e3,
        vertical_size(opts; num_cells_k = num_cells_k, kDim = kDim)...,
        BCL = side_bc,
        BCR = side_bc,
        BCB = topbot_bc,
        BCT = topbot_bc,
        vars = Dict(v => i for (i, v) in enumerate(vars)),
    )

    return ModelParameters(
        ts = ts,
        integration_time = 3600.0,
        output_interval = output_interval,
        equation_set = "moist_compressible_XZ",
        initial_conditions = joinpath(output_dir, "o01_ics.csv"),
        output_dir = output_dir,
        ref_state_file = joinpath(output_dir, "o01_exact.ref"),
        grid_params = grid_params,
        physical_params = physical_params,
        options = options,
    )
end

# ── Initial conditions ─────────────────────────────────────────────────────

"""
Balance the Dunion sounding hydrostatically on the model column, write it as the
run's exact pressure-based reference (one file for init, integration and
diagnostics), and add the RH-preserving 3 K warm bubble.
"""
function o01_init!(model)
    patch = createGrid(model.grid_params)
    gridpoints = Scythe.getGridpoints(patch)
    kDim = model.grid_params.kDim
    z = gridpoints[1:kDim, 2]
    column = Scythe.reference_column(patch, model.grid_params)

    ref_phys = Springsteel.calculate_pressure_reference_state(DUNION_SOUNDING, z, column)
    pbar = Springsteel.ref_pressure(ref_phys)[:, 1]
    rho_dbar = Springsteel.ref_rho_d(ref_phys)[:, 1]
    rho_vbar = Springsteel.ref_rho_v(ref_phys)[:, 1]
    Scythe.write_exact_ref_mc(model.ref_state_file, z, pbar, rho_dbar, rho_vbar,
                              zeros(kDim))
    ref = Springsteel.exact_pressure_reference_state(model.ref_state_file, z, column)

    # Hydrostatic sanity: residual of the balanced reference on its own column
    rho_tbar = Springsteel.ref_rho_t(ref)[:, 1]
    pbar_z = Springsteel.ref_pressure(ref)[:, 2]
    residual = pbar_z .+ (Scythe.gravity .* rho_tbar)
    println("Reference: sfc p = $(round(pbar[1] / 100.0, digits=2)) hPa, ",
            "max hydrostatic residual = $(maximum(abs.(residual))) Pa/m")

    patch.physical .= 0.0
    Scythe.moist_temperature_bubble_mc!(patch, gridpoints, ref;
                                        xc = 75.0e3, xr = 16.0e3,
                                        zc = 500.0, zr = 3000.0, dT_max = 3.0)
    Scythe.write_ics_csv(model.initial_conditions, patch, gridpoints)
end

# ── Diagnostics ─────────────────────────────────────────────────────────────

"""Sorted (time, path) pairs for every physical output snapshot of the run."""
function output_snapshots(model)
    files = filter(f -> endswith(f, "_physical.csv"), readdir(model.output_dir))
    pairs = [(parse(Float64, replace(f, "_physical.csv" => "")),
              joinpath(model.output_dir, f)) for f in files]
    return sort(pairs, by = first)
end

"""Domain-mean total water path [mm] of a snapshot (1 kg/m² of water ≡ 1 mm)."""
function water_path_mm(path, model, ref, kDim)
    df = CSV.read(path, DataFrame)
    ncols = div(nrow(df), kDim)
    rho_wbar = Springsteel.ref_rho_t(ref)[:, 1] .- Springsteel.ref_rho_d(ref)[:, 1]
    rho_w = (df.rho_t .- df.rho_d) .+ repeat(rho_wbar, ncols)
    W = domain_integral(reshape(rho_w, kDim, ncols), model)   # kg/m per unit y
    return W / (model.grid_params.iMax - model.grid_params.iMin)
end

"""
Rain diagnostics over every output snapshot: peak surface rain rate, onset time,
maximum rain density, and the time-integrated surface mass and energy fluxes
("surface" = the lowest mish level; Gauss nodes exclude z = 0 itself).
"""
function o01_rain_diagnostics(model, ref, kDim)
    gp = model.grid_params
    snaps = output_snapshots(model)
    peak_rate = 0.0
    onset = NaN
    max_rr = 0.0
    min_rr = 0.0
    times = Float64[]
    rate_int = Float64[]    # domain-integrated surface rain flux [kg/(m s) per unit y]
    eflux_int = Float64[]   # domain-integrated surface energy flux [J/(m s) per unit y]
    Wh = Float64[]
    for (t, path) in snaps
        df = CSV.read(path, DataFrame)
        ncols = div(nrow(df), kDim)
        surf = 1:kDim:nrow(df)
        Tk, _, rho_d, _, _, _ = mc_state(df, ref, kDim, ncols)
        max_rr = max(max_rr, maximum(df.rho_r))
        min_rr = min(min_rr, minimum(df.rho_r))
        rr_s = max.(df.rho_r[surf], 0.0)
        Vt = Scythe.rain_terminal_velocity.(rr_s, rho_d[surf], Tk[surf])
        R = -rr_s .* Vt                                       # kg/m²/s, >= 0
        pk = maximum(R)
        peak_rate = max(peak_rate, pk)
        if isnan(onset) && pk > 1.0e-3                        # 1 g m^-2 s^-1
            onset = t
        end
        if isempty(Wh)
            Wh = gauss_cell_weights(ncols, gp.num_cells, gp.iMax - gp.iMin,
                                    ncols ÷ gp.num_cells, gp.quadrature)
        end
        e_l = (Scythe.Cpv .* Tk[surf]) .- Scythe.L_v.(Tk[surf])   # + g*z ≈ 0 at the surface
        push!(times, t)
        push!(rate_int, sum(Wh .* R))
        push!(eflux_int, sum(Wh .* (rr_s .* Vt .* e_l)))      # F_E(z≈0), > 0 for e_l < 0
    end
    # Trapezoid over the 60 s output cadence (few-% cross-check, not the budget)
    accum_flux = 0.0
    accum_E = 0.0
    for i in 1:(length(times) - 1)
        dt = times[i+1] - times[i]
        accum_flux += 0.5 * (rate_int[i] + rate_int[i+1]) * dt
        accum_E += 0.5 * (eflux_int[i] + eflux_int[i+1]) * dt
    end
    width = gp.iMax - gp.iMin
    return Dict(
        "peak_rain_rate_gm2s" => 1000.0 * peak_rate,
        "rain_onset_min" => onset / 60.0,
        "max_rho_r_gm3" => 1000.0 * max_rr,
        "min_rho_r_gm3" => 1000.0 * min_rr,             # spline undershoot monitor
        "accum_rainfall_flux_mm" => accum_flux / width, # cross-check of the exact budget
        "precip_energy_gain_Jm2" => accum_E / width,    # predicted domain E_t gain
    )
end

function o01_diagnostics(model)
    df = read_final_output(model)
    ref, _, kDim = rebuild_reference(model)

    diags = o01_rain_diagnostics(model, ref, kDim)
    diags["max_w"] = maximum(df.w)
    diags["min_w"] = minimum(df.w)

    # Exact water budget: the surface outflow is the only sink of domain water, so
    # the accumulated rainfall IS the water-path difference (independent of the
    # 60 s flux sampling above).
    snaps = output_snapshots(model)
    wp0 = water_path_mm(snaps[1][2], model, ref, kDim)
    wpN = water_path_mm(snaps[end][2], model, ref, kDim)
    diags["accum_rainfall_mm"] = wp0 - wpN

    drift = conservation_drift(model, ref)
    # Energy closure: rain leaving through the surface CHANGES the domain E_t by the
    # boundary flux (positive: e_l < 0 leaves, so E_t rises). Compare the raw drift
    # against the predicted precipitation gain; the residual is the actual
    # conservation error of the run.
    E0 = begin
        df0 = CSV.read(snaps[1][2], DataFrame)
        ncols = div(nrow(df0), kDim)
        E_t = df0.E_t .+ repeat(Springsteel.ref_total_energy(ref)[:, 1], ncols)
        domain_integral(reshape(E_t, kDim, ncols), model)
    end
    width = model.grid_params.iMax - model.grid_params.iMin
    predicted_gain_pct = 100.0 * diags["precip_energy_gain_Jm2"] * width / E0
    diags["energy_residual_pct"] = drift["energy_drift_pct"] - predicted_gain_pct

    return merge(diags, drift)
end

# ── Figures ────────────────────────────────────────────────────────────────

plotter = nothing
if opts.plot
    include(joinpath(@__DIR__, "common", "plots.jl"))
    plotter = function (model)
        df = read_final_output(model)
        ref, _, kDim = rebuild_reference(model)
        ncols = div(nrow(df), kDim)
        x = reshape(df.r, kDim, ncols)[1, :]
        z = reshape(df.z, kDim, ncols)[:, 1]
        w = reshape(df.w, kDim, ncols)
        rho_r = reshape(df.rho_r .* 1000.0, kDim, ncols)
        save_benchmark_figure(
            joinpath(model.output_dir, "o01_rainfall_$(opts.mode)_$(opts.stage)_final.png"),
            x, z,
            [(rho_r, "ρ_r (g/m³)", 0.0:0.25:4.0),
             (w, "w (m/s)", -12.0:2.0:24.0)];
            title = "O01 warm rain, t = $(model.integration_time) s")
    end
end

# ── Run ────────────────────────────────────────────────────────────────────

model = o01_model(opts)
passed = run_benchmark("o01_rainfall", opts;
                       model = model,
                       init! = o01_init!,
                       diagnostics = o01_diagnostics,
                       varnames = MC_VARS,
                       plotter = plotter)
exit(passed ? 0 : 1)
