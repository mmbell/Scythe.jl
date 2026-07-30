#!/usr/bin/env julia
# Ooyama (2001)-style warm-rain benchmark on the AXISYMMETRIC CYLINDER
# (moist_compressible_axisym): the o01_rainfall configuration with the RiRk
# horizontal coordinate reinterpreted as radius, offset to r in [1000, 1150] km
# so the cylindrical metric terms (O(dx/r) ~ 0.2%) perturb but do not reshape
# the solution, plus the prognostic tangential wind v (slot 9) with f = 0.
#
# Purpose: validate the geometry-generic mc_driver! cylindrical path against
# the Cartesian XZ benchmark — the rain diagnostics must sit inside the same
# o01_rainfall windows, and v must stay identically zero (no source with f = 0
# and v0 = 0; any nonzero v is a leak in the 9-var machinery).
#
#   julia --project=. benchmarks/o01_axisym.jl --mode quick --stage mc --grid rirk
#
# Caveats vs a "true" axisymmetric vortex run: the domain integrals in the
# shared diagnostics use Cartesian (per-unit-azimuthal-length) weights, not
# r-weighted volume integrals — at r/L ~ 7 the difference is a few tenths of a
# percent and cancels between the budget's two sides. The bubble is a torus at
# r = 1075 km, not an axis-centered plume; axis-adjacent configurations arrive
# with the RLR grid (see benchmarks/o01_rlr.jl).

using Distributed

include(joinpath(@__DIR__, "common", "harness.jl"))
opts = parse_benchmark_args(ARGS)
opts.stage == STAGE_MC ||
    error("o01_axisym supports only --stage mc (the total-energy set carries rho_r)")
opts.nests == 1 || error("o01_axisym supports only --nests 1")
opts.grid == :rirk || error("o01_axisym targets the RiRk (B-spline vertical) grid")

add_benchmark_workers(opts; count = opts.workers)
@everywhere using Springsteel
@everywhere using Scythe

include(joinpath(@__DIR__, "common", "diagnostics.jl"))

# ── Configuration (o01_rainfall with the radial offset and the 9-var set) ────

const MC_VARS_CYL = Scythe.MC_VARS_CYL          # ["p", ..., "rho_r", "v"]
const R_INNER = 1000.0e3                        # inner radius of the annulus [m]
const R_WIDTH = 150.0e3                         # radial extent (o01's domain width)

const DUNION_SOUNDING = joinpath(REFERENCE_DATA_DIR, "o01_rainfall", "dunion_MT_hum90.ref")
const N_R = 1.0e-3
const KV_MOM = 0.0
const KV_HEAT = 0.0
const KV_WATER = 0.0

function o01_axisym_model(opts::BenchmarkOptions)
    if opts.mode == :full
        num_cells_i = 300       # 500 m cells over 150 km
        ts = 0.15
    else
        num_cells_i = 75        # 2 km cells
        ts = 0.3
    end
    num_cells_k = 50            # 500 m vertical cells to the 25 km lid
    kDim = 150
    output_interval = 60.0

    ts = vertical_ts(ts, opts)

    vars = MC_VARS_CYL
    # Same DK83 stratospheric sponge as o01_rainfall; v is damped alongside u/w
    # (the mc sponge is momentum-only with the KE routed to E_t).
    physical_params = Dict(:Khdiff => 0.0, :Kvdiff => KV_MOM,
                           :Khdiff_heat => 0.0, :Kvdiff_heat => KV_HEAT,
                           :Kvdiff_water => KV_WATER,
                           :tau_qss => 10.0, :N_r => N_R,
                           :alpha => 0.02, :z_damp => 17.0e3,
                           :f => 0.0)
    options = Dict(:semiimplicit => true, :exact_reference_state => true,
                   :precipitation => true, :vertical_mixing => false)

    output_dir = benchmark_output_dir("o01_axisym", opts)
    scalar_bc = Dict(v => NeumannBC() for v in vars)   # includes v: free-slip walls
    # Radial walls: no normal flow (u Dirichlet), tangential v free-slip (Neumann)
    side_bc = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))
    topbot_bc = merge(scalar_bc, Dict("w" => DirichletBC(), "rho_r" => NaturalBC()))

    grid_params = GridParameters(;
        geometry = "RiRk",
        iMin = R_INNER,
        iMax = R_INNER + R_WIDTH,
        num_cells_i = num_cells_i,
        kMin = 0.0,
        kMax = 25.0e3,
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
        equation_set = "moist_compressible_axisym",
        initial_conditions = joinpath(output_dir, "o01_ics.csv"),
        output_dir = output_dir,
        ref_state_file = joinpath(output_dir, "o01_exact.ref"),
        grid_params = grid_params,
        physical_params = physical_params,
        options = options,
    )
end

# ── Initial conditions (torus bubble at the annulus midpoint; v = 0) ─────────

function o01_axisym_init!(model)
    for f in readdir(model.output_dir)
        endswith(f, "_physical.csv") && rm(joinpath(model.output_dir, f))
    end

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

    rho_tbar = Springsteel.ref_rho_t(ref)[:, 1]
    pbar_z = Springsteel.ref_pressure(ref)[:, 2]
    residual = pbar_z .+ (Scythe.gravity .* rho_tbar)
    println("Reference: sfc p = $(round(pbar[1] / 100.0, digits=2)) hPa, ",
            "max hydrostatic residual = $(maximum(abs.(residual))) Pa/m")

    patch.physical .= 0.0                       # v (slot 9) starts identically zero
    Scythe.moist_temperature_bubble_mc!(patch, gridpoints, ref;
                                        xc = R_INNER + 75.0e3, xr = 16.0e3,
                                        zc = 500.0, zr = 3000.0, dT_max = 3.0)
    Scythe.write_ics_csv(model.initial_conditions, patch, gridpoints)
end

# ── Diagnostics (o01_rainfall's rain/budget set + the tangential-wind guard) ─

function output_snapshots(model)
    files = filter(f -> endswith(f, "_physical.csv"), readdir(model.output_dir))
    pairs = [(parse(Float64, replace(f, "_physical.csv" => "")),
              joinpath(model.output_dir, f)) for f in files]
    filter!(p -> p[1] <= model.integration_time + 1.0e-6, pairs)
    return sort(pairs, by = first)
end

function water_path_mm(path, model, ref, kDim)
    df = CSV.read(path, DataFrame)
    ncols = div(nrow(df), kDim)
    rho_wbar = Springsteel.ref_rho_t(ref)[:, 1] .- Springsteel.ref_rho_d(ref)[:, 1]
    rho_w = (df.rho_t .- df.rho_d) .+ repeat(rho_wbar, ncols)
    W = domain_integral(reshape(rho_w, kDim, ncols), model)
    return W / (model.grid_params.iMax - model.grid_params.iMin)
end

function o01_rain_diagnostics(model, ref, kDim)
    gp = model.grid_params
    snaps = output_snapshots(model)
    peak_rate = 0.0
    onset = NaN
    max_rr = 0.0
    min_rr = 0.0
    max_v = 0.0
    times = Float64[]
    rate_int = Float64[]
    eflux_int = Float64[]
    Wh = Float64[]
    for (t, path) in snaps
        df = CSV.read(path, DataFrame)
        ncols = div(nrow(df), kDim)
        surf = 1:kDim:nrow(df)
        # Assumes options[:condensate_transform] = :none, which is the only state this
        # script can produce (it exposes no transform knob). If one is added, pass
        # `transform =` here -- slot 9 would otherwise be read as a density.
        Tk, _, rho_d, _, _, _ = mc_state(df, ref, kDim, ncols)
        max_rr = max(max_rr, maximum(df.rho_r))
        min_rr = min(min_rr, minimum(df.rho_r))
        max_v = max(max_v, maximum(abs.(df.v)))
        rr_s = max.(df.rho_r[surf], 0.0)
        Vt = Scythe.rain_terminal_velocity.(rr_s, rho_d[surf], Tk[surf])
        R = -rr_s .* Vt
        pk = maximum(R)
        peak_rate = max(peak_rate, pk)
        if isnan(onset) && pk > 1.0e-3
            onset = t
        end
        if isempty(Wh)
            Wh = gauss_cell_weights(ncols, gp.num_cells, gp.iMax - gp.iMin,
                                    ncols ÷ gp.num_cells, gp.quadrature)
        end
        e_l = (Scythe.Cpv .* Tk[surf]) .- Scythe.L_v.(Tk[surf])
        push!(times, t)
        push!(rate_int, sum(Wh .* R))
        push!(eflux_int, sum(Wh .* (rr_s .* Vt .* e_l)))
    end
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
        "min_rho_r_gm3" => 1000.0 * min_rr,
        "max_abs_v" => max_v,
        "accum_rainfall_flux_mm" => accum_flux / width,
        "precip_energy_gain_Jm2" => accum_E / width,
    )
end

function o01_axisym_diagnostics(model)
    df = read_final_output(model)
    ref, _, kDim = rebuild_reference(model)

    diags = o01_rain_diagnostics(model, ref, kDim)
    diags["max_w"] = maximum(df.w)
    diags["min_w"] = minimum(df.w)

    snaps = output_snapshots(model)
    wp0 = water_path_mm(snaps[1][2], model, ref, kDim)
    wpN = water_path_mm(snaps[end][2], model, ref, kDim)
    diags["accum_rainfall_mm"] = wp0 - wpN

    drift = conservation_drift(model, ref)
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

# ── Run ────────────────────────────────────────────────────────────────────

model = o01_axisym_model(opts)
passed = run_benchmark("o01_axisym", opts;
                       model = model,
                       init! = o01_axisym_init!,
                       diagnostics = o01_axisym_diagnostics,
                       varnames = MC_VARS_CYL,
                       plotter = nothing)
exit(passed ? 0 : 1)
