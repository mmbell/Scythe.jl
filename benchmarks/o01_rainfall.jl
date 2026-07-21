#!/usr/bin/env julia
# Ooyama (2001)-style warm-rain benchmark for the total-energy set.
#
# A 3 K RH-preserving warm bubble (cos² profile, 16 km x 3 km radii, centered
# at 500 m over the domain midpoint) rises through a humidified Dunion
# moist-tropical sounding (see DUNION_SOUNDING below for why the humidification
# is required), condenses, converts cloud to rain (autoconversion + collection),
# and rains out through the surface: the rho_r bottom boundary is a free
# (Natural) fit so the sedimentation flux removes water and its energy through
# z = 0. Rain condensation/evaporation runs through the tau_r channel of the
# generalized supersaturation relaxation (1/tau = 1/tau_c + 1/tau_r) with the
# monodisperse N_r closure — no separate Qevap parameterization (this differs
# from Ooyama 2001, as do the equation set and the single non-nested grid with
# the bubble at the domain center, so the targets are comparable-magnitude
# sanity windows, not a reproduction of his figures). Rain CONDENSATION is
# gated on cloud presence (see qss_condensation_rates): without the gate,
# spectral-ringing rain seeds grow in the wave-driven supersaturation at the
# lid and sediment back down as a spurious upper-tropospheric rain blob.
#
#   julia --project=. benchmarks/o01_rainfall.jl --mode quick --stage mc --grid rirk
#
# Modes: quick (2 km cells) | full (500 m cells). Both use 500 m vertical
# nodal spacing to the 25 km rigid lid, with a Durran-Klemp Rayleigh sponge
# (momentum-only, KE routed to E_t) over the top 8 km — confined to the
# stratosphere (the sounding tropopause knot is at 16.59 km) so deep
# convective flow is never damped, only the radiated gravity waves.
# Only --stage mc is supported.
#
# Reference: Ooyama (2001), J. Atmos. Sci. 58, 2073-2102 (Fig. 6: peak ground
# precipitation ~75-125 g m^-2 s^-1 at ~35-40 min for a similar bubble).
# reference/ooyama_jas2001.pdf

using Distributed

include(joinpath(@__DIR__, "common", "harness.jl"))
opts = parse_benchmark_args(ARGS)
opts.stage == STAGE_MC ||
    error("o01_rainfall supports only --stage mc (the total-energy set carries rho_r)")
opts.nests in (1, 3) ||
    error("o01_rainfall supports --nests 1 (single grid) or 3 (5-patch two-way nest)")
opts.nests == 1 || opts.grid == :rirk ||
    error("--nests 3 requires --grid rirk (spline interface coupling)")

# 3-level nest = 5 abutting patches; the fine center patch gets the extra workers
# in full mode (it dominates the column-step cost).
nested_workers(opts) = opts.mode == :full ? [1, 1, 4, 1, 1] : [1, 1, 1, 1, 1]

add_benchmark_workers(opts;
                      count = opts.nests > 1 ? sum(nested_workers(opts)) : opts.workers)
@everywhere using Springsteel
@everywhere using Scythe

include(joinpath(@__DIR__, "common", "diagnostics.jl"))

# ── Configuration ──────────────────────────────────────────────────────────

const MC_VARS = ["p", "rho_d", "rho_t", "u", "w", "E_t", "Q_ss", "rho_r"]

# HUMIDIFIED Dunion moist-tropical sounding (WRF input_sounding format): RH floors
# of 0.90 (z <= 1.6 km) / 0.88 (<= 3.2 km) / 0.85 (<= 4.5 km) applied to the
# original dunion_MT.ref (committed alongside), mirroring Ooyama's own "slightly
# humidified" Jordan sounding. The UNmodified Dunion profile (low-level RH ~83%,
# falling to 56% at 4.4 km) does NOT convect from this trigger within the hour at
# ANY resolution: the wide slab's linear ascent (~w < 1 m/s) saturates a thin core
# at ~14 min but dry entrainment and CIN kill it — the archived RZ notebook run
# with the same bubble only erupted at t ~ 2.7 h, and only with Kv_mudiff = 100
# background moistening. The RH floors give O01-comparable onset (~24 min) and
# ground rain rates (quick ~28, full ~78 g m^-2 s^-1 vs O01's 75-125).
const DUNION_SOUNDING = joinpath(REFERENCE_DATA_DIR, "o01_rainfall", "dunion_MT_hum90.ref")

# Rain-drop number concentration [#/cm^3] for the monodisperse tau_r closure
# (~1000 drops per m^3; the small number keeps the bulk of condensation on cloud).
const N_R = 1.0e-3

# Marshall-Palmer intercept [m^-4] for the exponential-DSD tau_r closure, for
# sensitivity runs only (SCYTHE_O01_N0=8.0e6 selects MP; the default 0.0 keeps
# the monodisperse closure the references were seeded with, bit-identical).
const N_0_MP = parse(Float64, get(ENV, "SCYTHE_O01_N0", "0.0"))

# Vertical eddy coefficients [m^2/s]. ZERO: the run is stable fully inviscid on
# the RiRk grid at both resolutions — the cubic B-spline Galerkin filter is the
# only dissipation, which is the near-inviscid goal. (Kv = 5 and 25 were tested
# and change the solution by < 1%; raise these if a future configuration needs
# damping, they feed the momentum/heat/water solves independently.) The rain
# shafts do ring: min(rho_r) undershoots reach ~ -1.8 g/m^3 at 500 m resolution
# (all rate functions are negative-safe; reported as min_rho_r_gm3).
const KV_MOM = 0.0
const KV_HEAT = 0.0
const KV_WATER = 0.0

function o01_model(opts::BenchmarkOptions)
    if opts.mode == :full
        num_cells_i = 300       # 500 m cells over 150 km
        ts = 0.15
    else
        num_cells_i = 75        # 2 km cells
        ts = 0.3
    end
    # Vertical: 500 m nodal spacing to 25 km in BOTH modes (the rain physics and
    # the sedimentation flux do not coarsen with the horizontal grid). The top
    # 8 km (17-25 km) is the Rayleigh sponge; the 25 km lid (vs the historical
    # 20 km) buys a stratosphere-confined absorber above the 16.59 km tropopause.
    num_cells_k = 50            # RiRk: 500 m cells
    kDim = 150                  # RZ: Chebyshev points (untargeted fallback)
    output_interval = 60.0

    ts = vertical_ts(ts, opts)

    vars = MC_VARS
    # Rayleigh sponge (Durran-Klemp 1983 eq. 29 profile, momentum-only in mc):
    # onset at 17 km (just above the sounding's 16.59 km tropopause knot, so the
    # damping lives entirely in the high-static-stability stratosphere), 8 km =
    # 16 cells deep. alpha = 0.02 1/s puts the lid e-folding at ~39 s (the DK
    # profile peaks at 1.285*alpha) against stratospheric wave intrinsic
    # frequencies ~0.005-0.015 1/s — inside the Klemp-Lilly 2 <= alpha/omega <= 5
    # optimum for the dominant modes.
    physical_params = Dict(:Khdiff => 0.0, :Kvdiff => KV_MOM,
                           :Khdiff_heat => 0.0, :Kvdiff_heat => KV_HEAT,
                           :Kvdiff_water => KV_WATER,
                           :tau_qss => 10.0, :N_r => N_R, :N_0 => N_0_MP,
                           :alpha => 0.02, :z_damp => 17.0e3)
    options = merge(Dict{Symbol,Any}(:semiimplicit => true, :exact_reference_state => true,
                                     :precipitation => true, :vertical_mixing => false),
                    reference_state_options())

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
    # The output directory is shared across runs of this variant and the rain
    # diagnostics sweep EVERY snapshot in it — stale files from a previous (e.g.
    # longer) run would silently contaminate the onset/accumulation numbers.
    for f in readdir(model.output_dir)
        endswith(f, "_physical.csv") && rm(joinpath(model.output_dir, f))
    end

    patch = createGrid(model.grid_params)
    gridpoints = Scythe.getGridpoints(patch)
    kDim = model.grid_params.kDim
    z = gridpoints[1:kDim, 2]
    column = Scythe.reference_column(patch, model.grid_params)

    # The .ref file carries VALUES only, so under :hydrostatic_reference the CONVERGED
    # (p, rho_d, rho_v) triple has to be written for the balance to survive the round
    # trip -- exact_pressure_reference_state re-integrates dp/dz = -g*rho_t from it.
    hydro = reference_state_hydrostatic()
    ref_phys = Springsteel.calculate_pressure_reference_state(DUNION_SOUNDING, z, column;
                                                             hydrostatic = hydro)
    pbar = Springsteel.ref_pressure(ref_phys)[:, 1]
    rho_dbar = Springsteel.ref_rho_d(ref_phys)[:, 1]
    rho_vbar = Springsteel.ref_rho_v(ref_phys)[:, 1]
    Scythe.write_exact_ref_mc(model.ref_state_file, z, pbar, rho_dbar, rho_vbar,
                              zeros(kDim))
    ref = Springsteel.exact_pressure_reference_state(model.ref_state_file, z, column;
                                                     hydrostatic = hydro)
    # The ICs are stored as perturbations from Q̄_ss, so the bubble must be differenced
    # against the SAME Q̄_ss createModelTile will add back.
    if get(reference_state_options(), :consistent_qss_reference, false)
        ref = Scythe.consistent_qss_reference(ref, z, column)
    end

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

"""Sorted (time, path) pairs for every physical output snapshot of THIS run
(bounded by the integration time as a second guard against stale files)."""
function output_snapshots(model)
    files = filter(f -> endswith(f, "_physical.csv"), readdir(model.output_dir))
    pairs = [(parse(Float64, replace(f, "_physical.csv" => "")),
              joinpath(model.output_dir, f)) for f in files]
    filter!(p -> p[1] <= model.integration_time + 1.0e-6, pairs)
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

# ── Nested (3-level, 5-patch) configuration ─────────────────────────────────
#
# Grid nesting per DeMaria et al. (1992)/Ooyama (2001): fine center patch over
# the bubble, 2x-coarser abutting patches outward, two-way coupling (R3X trio
# down, collar tendency injection up), per-patch timesteps. The vertical grid
# is identical everywhere. Coarse-patch timesteps are capped at 0.6 s: the
# moist scheme's stability ceiling is set by the vertical acoustic Courant
# (c*ts/dz_min ≈ 1.8 at 0.6 s; a single-grid isolation run at ts = 1.2
# reproduced the blow-up with no nesting involved), so the outermost patches
# gain less than the full 2x-per-level — the nested speedup comes mostly from
# the column-count reduction.

"""Nest layout for the configured mode, wrapping the single-grid model as base."""
function o01_nest(opts::BenchmarkOptions)
    base = o01_model(opts)
    if opts.mode == :full
        boundaries = [0.0, 50.0e3, 63.0e3, 87.0e3, 100.0e3, 150.0e3]
        num_cells = [25, 13, 48, 13, 25]           # 2 | 1 | 0.5 | 1 | 2 km cells
        ts = [0.6, 0.3, 0.15, 0.3, 0.6]
    else
        boundaries = [0.0, 48.0e3, 64.0e3, 86.0e3, 102.0e3, 150.0e3]
        num_cells = [6, 4, 11, 4, 6]               # 8 | 4 | 2 | 4 | 8 km cells
        ts = [0.6, 0.6, 0.3, 0.6, 0.6]
    end
    ts = [vertical_ts(t, opts) for t in ts]
    return NestedModelParameters(
        boundaries = boundaries,
        num_cells = num_cells,
        ts = ts,
        workers_per_patch = nested_workers(opts),
        base = base)
end

"""Nominal (collar-excluded) x-bounds of nest patch `i`."""
function o01_nominal_bounds(models, topo, i)
    xlo = models[i].grid_params.iMin
    xhi = models[i].grid_params.iMax
    for k in topo.child_ifaces[i]
        ni = topo.interfaces[k]
        if ni.parent_side == :right
            xhi = ni.interface_x
        else
            xlo = ni.interface_x
        end
    end
    return xlo, xhi
end

"""Shared reference + per-patch RH-preserving bubble ICs (nested o01_init!)."""
function o01_init_nested!(models, topo)
    for m in models
        mkpath(m.output_dir)
        for f in readdir(m.output_dir)
            endswith(f, "_physical.csv") && rm(joinpath(m.output_dir, f))
        end
    end

    # The reference depends on z only; build it once and share the file.
    m1 = models[1]
    patch = createGrid(m1.grid_params)
    gridpoints = Scythe.getGridpoints(patch)
    kDim = m1.grid_params.kDim
    z = gridpoints[1:kDim, 2]
    column = Scythe.reference_column(patch, m1.grid_params)
    ref_phys = Springsteel.calculate_pressure_reference_state(DUNION_SOUNDING, z, column)
    Scythe.write_exact_ref_mc(m1.ref_state_file, z,
                              Springsteel.ref_pressure(ref_phys)[:, 1],
                              Springsteel.ref_rho_d(ref_phys)[:, 1],
                              Springsteel.ref_rho_v(ref_phys)[:, 1],
                              zeros(kDim))
    ref = Springsteel.exact_pressure_reference_state(m1.ref_state_file, z, column)
    rho_tbar = Springsteel.ref_rho_t(ref)[:, 1]
    pbar_z = Springsteel.ref_pressure(ref)[:, 2]
    residual = pbar_z .+ (Scythe.gravity .* rho_tbar)
    println("Reference: sfc p = $(round(Springsteel.ref_pressure(ref)[1, 1] / 100.0, digits=2)) hPa, ",
            "max hydrostatic residual = $(maximum(abs.(residual))) Pa/m")

    for m in models
        p = createGrid(m.grid_params)
        gpts = Scythe.getGridpoints(p)
        p.physical .= 0.0
        Scythe.moist_temperature_bubble_mc!(p, gpts, ref;
                                            xc = 75.0e3, xr = 16.0e3,
                                            zc = 500.0, zr = 3000.0, dT_max = 3.0)
        Scythe.write_ics_csv(m.initial_conditions, p, gpts)
    end
end

"""
Nested rain + budget diagnostics: per-patch surface-rain series restricted to
each patch's nominal region (collar cells excluded so abutting patches
partition the domain exactly), summed/maxed across patches; masked domain
integrals for the water and energy budgets.
"""
function o01_nested_diagnostics(models, topo)
    ref, _, kDim = rebuild_reference(models[1])
    n = length(models)
    masks = [nominal_col_mask(models[i], o01_nominal_bounds(models, topo, i)...)
             for i in 1:n]
    total_width = models[n].grid_params.iMax - models[1].grid_params.iMin

    snaps = [output_snapshots(m) for m in models]
    ntimes = minimum(length.(snaps))
    times = [snaps[1][j][1] for j in 1:ntimes]

    peak_rate = 0.0
    onset = NaN
    max_rr = 0.0
    min_rr = 0.0
    # w extrema over the WHOLE run (user decision 2026-07-14): the nested max_w
    # is defined as the run maximum, not the final-time value the single-grid
    # diagnostic reports. By the end of the hour the secondary cells have
    # propagated into the coarser outer nests where their intensity is
    # resolution-limited, so a final-time sample measures the outer-nest
    # resolution rather than the convection the benchmark targets; the run
    # maximum (reached in the fine nest) is the comparable quantity.
    max_w = -Inf
    min_w = Inf
    rate_int = zeros(ntimes)
    eflux_int = zeros(ntimes)
    Whs = [Float64[] for _ in 1:n]            # masked weights, filled lazily
    for j in 1:ntimes
        pk_t = 0.0
        for i in 1:n
            t, path = snaps[i][j]
            df = CSV.read(path, DataFrame)
            gp = models[i].grid_params
            ncols = div(nrow(df), kDim)
            surf = 1:kDim:nrow(df)
            Tk, _, rho_d, _, _, _ = mc_state(df, ref, kDim, ncols)
            mask = masks[i]
            colmask = repeat(mask, inner = kDim)
            max_rr = max(max_rr, maximum(df.rho_r[colmask]))
            min_rr = min(min_rr, minimum(df.rho_r[colmask]))
            max_w = max(max_w, maximum(df.w[colmask]))
            min_w = min(min_w, minimum(df.w[colmask]))
            rr_s = max.(df.rho_r[surf], 0.0) .* mask
            Vt = Scythe.rain_terminal_velocity.(rr_s, rho_d[surf], Tk[surf])
            R = -rr_s .* Vt
            pk_t = max(pk_t, maximum(R))
            if isempty(Whs[i])
                Whs[i] = gauss_cell_weights(ncols, gp.num_cells, gp.iMax - gp.iMin,
                                            ncols ÷ gp.num_cells, gp.quadrature) .* mask
            end
            e_l = (Scythe.Cpv .* Tk[surf]) .- Scythe.L_v.(Tk[surf])
            rate_int[j] += sum(Whs[i] .* R)
            eflux_int[j] += sum(Whs[i] .* (rr_s .* Vt .* e_l))    # F_E(z≈0), > 0 for e_l < 0
        end
        peak_rate = max(peak_rate, pk_t)
        if isnan(onset) && pk_t > 1.0e-3
            onset = times[j]
        end
    end
    accum_flux = 0.0
    accum_E = 0.0
    for j in 1:(ntimes - 1)
        dt = times[j+1] - times[j]
        accum_flux += 0.5 * (rate_int[j] + rate_int[j+1]) * dt
        accum_E += 0.5 * (eflux_int[j] + eflux_int[j+1]) * dt
    end

    diags = Dict(
        "peak_rain_rate_gm2s" => 1000.0 * peak_rate,
        "rain_onset_min" => onset / 60.0,
        "max_rho_r_gm3" => 1000.0 * max_rr,
        "min_rho_r_gm3" => 1000.0 * min_rr,
        "accum_rainfall_flux_mm" => accum_flux / total_width,
        "precip_energy_gain_Jm2" => accum_E / total_width,
    )

    # Masked water/energy budgets across the nest
    rho_wbar = Springsteel.ref_rho_t(ref)[:, 1] .- Springsteel.ref_rho_d(ref)[:, 1]
    E_tbar = Springsteel.ref_total_energy(ref)[:, 1]
    function nest_budget(j)
        W = 0.0
        E = 0.0
        for i in 1:n
            df = CSV.read(snaps[i][j][2], DataFrame)
            ncols = div(nrow(df), kDim)
            rho_w = (df.rho_t .- df.rho_d) .+ repeat(rho_wbar, ncols)
            E_t = df.E_t .+ repeat(E_tbar, ncols)
            W += domain_integral(reshape(rho_w, kDim, ncols), models[i], masks[i])
            E += domain_integral(reshape(E_t, kDim, ncols), models[i], masks[i])
        end
        return W, E
    end
    W0, E0 = nest_budget(1)
    WN, EN = nest_budget(ntimes)
    diags["accum_rainfall_mm"] = (W0 - WN) / total_width
    diags["energy_drift_pct"] = 100.0 * (EN - E0) / E0
    predicted_gain_pct = 100.0 * diags["precip_energy_gain_Jm2"] * total_width / E0
    diags["energy_residual_pct"] = diags["energy_drift_pct"] - predicted_gain_pct

    diags["max_w"] = max_w
    diags["min_w"] = min_w

    return diags
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

if opts.nests == 1
    model = o01_model(opts)
    passed = run_benchmark("o01_rainfall", opts;
                           model = model,
                           init! = o01_init!,
                           diagnostics = o01_diagnostics,
                           varnames = MC_VARS,
                           plotter = plotter)
else
    nest = o01_nest(opts)
    passed = run_nested_benchmark("o01_rainfall", opts;
                                  nest = nest,
                                  init! = o01_init_nested!,
                                  diagnostics = o01_nested_diagnostics)
end
exit(passed ? 0 : 1)
