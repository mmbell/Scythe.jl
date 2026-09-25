#!/usr/bin/env julia
# Mesoscale warm bubble over a fixed-SST ocean: the boundary-layer / full-physics-suite
# benchmark (MYNN-EDMF plan, stage S1; precursor of the radiative-convective-equilibrium
# case). The O01 warm bubble (o01_rainfall.jl) has no ocean and its full mode is
# gray-zone; this case runs at production-like spacing (the TC nests') with surface
# fluxes and drag ON, so a boundary-layer scheme is exercised where its assumptions hold.
#
#   julia --project=. benchmarks/ocean_warm_bubble.jl --mode quick --stage mc --grid rirk
#
# THIS IS NOW A THIN WRAPPER (config consolidation, stage 4a). The model that runs is
# EXACTLY simulations/ocean_warm_bubble/scythe_config.toml -- struct-default production physics
# (MYNN + ISHMAEL ice, no radiation, state_dependent_si) on the quick grid (300 km x 25 km,
# 6 km x 500 m cells, 2 h). `--mode full` scales the grid to the paper-grade spacing (3 km
# x 300 m, ts 0.25 s) through `--set`, the same numbers the legacy `full` arm used; every
# other physics/grid variation goes through `simulations/ocean_warm_bubble/run.jl` (edit a
# copy of scythe_config.toml), not this script. All the `SCYTHE_OWB_*` env-var
# knobs the legacy driver read are GONE -- there is no arm selection here anymore. The
# "arm" tag below (output/reference-CSV naming) is derived from the RESOLVED config (ice
# on/off, BL scheme, EDMF, fidelity, surface roughness/stability), so it always describes
# what actually ran rather than an env var that might disagree with it.
#
# This script exists only so the harness's diagnostics table, target checks and committed
# regression CSVs (benchmarks/reference_data/ocean_warm_bubble/) keep working; see
# simulations/ocean_warm_bubble/README.md for the case itself.
#
# Physics/grid knobs are no longer read here. The legacy driver had ~25 `SCYTHE_OWB_*` env
# vars (BL scheme, MYNN EDMF/fidelity/interval/K_max, surface roughness/stability, SST,
# ice, rain moments, RRTMGP arm, water transforms, wall BC, grid/run overrides) -- all
# retired. Vary the case by editing a copy of scythe_config.toml and running
# simulations/ocean_warm_bubble/run.jl instead (see that script and
# simulations/ocean_warm_bubble/README.md); this wrapper always runs the one production
# config, `--mode`/`--stage`/`--grid`/`--workers` aside.
#
# Diagnostics: O01's rain / ice / radiation sets (benchmarks/common/warm_bubble_diagnostics.jl)
# plus the surface-flux budget rows below. With fluxes on the column energy books are OPEN:
# `energy_residual_pct` subtracts BOTH the precipitation energy loss and the time-integrated
# surface enthalpy, vapour-energy and drag-work inputs, and `accum_rainfall_mm` is the water
# path change corrected for the surface vapour input. No published target exists;
# `BENCHMARK_EXPECTED["ocean_warm_bubble"]` is seeded from the accepted control run.

using Distributed

include(joinpath(@__DIR__, "common", "harness.jl"))

# ARGS parsing and worker/process setup are script-only (an `include` of this file --
# simulations/ocean_warm_bubble/run.jl's diagnostics post-step -- wants `owb_diagnostics`
# and the other diagnostic functions defined without spawning workers or reading real
# command-line ARGS; nothing downstream needs this block to have run).
if abspath(PROGRAM_FILE) == @__FILE__
    opts = parse_benchmark_args(ARGS)
    opts.stage == STAGE_MC ||
        error("ocean_warm_bubble supports only --stage mc (the total-energy set)")
    opts.nests == 1 || error("ocean_warm_bubble supports only --nests 1 (nesting arrives later)")
    opts.grid == :rirk || error("ocean_warm_bubble targets the RiRk (B-spline vertical) grid")

    add_benchmark_workers(opts; count = opts.workers)
    @everywhere using Springsteel
    @everywhere using Scythe
end

include(joinpath(@__DIR__, "common", "diagnostics.jl"))
include(joinpath(@__DIR__, "common", "warm_bubble_diagnostics.jl"))

# ── Configuration ────────────────────────────────────────────────────────────
#
# The model is built from simulations/ocean_warm_bubble/scythe_config.toml -- the production
# case config (struct defaults: MYNN + ISHMAEL ice, radiation off, state_dependent_si) --
# via `build_model(load_config(...))`, not a hand-assembled `ModelParameters` (that
# translation now lives once, in src/config_build.jl). `--mode full` overrides the grid to
# the paper-grade spacing the legacy `full` arm used (3 km x 300 m cells, ts 0.25 s); every
# other physics/grid variation belongs on simulations/ocean_warm_bubble/run.jl instead of
# here (this wrapper takes no `--set` of its own; it exists for the harness's diagnostics
# table and regression CSVs, not as a general launcher).

const OWB_CONFIG = joinpath(@__DIR__, "..", "simulations", "ocean_warm_bubble", "scythe_config.toml")

"""
    owb_config_set(opts::BenchmarkOptions) -> Vector{String}

`--set` overrides applied on top of the case config: the harness's own output
directory/format (CSV so the diagnostics can read `<t>_physical.csv`, the MYNN sidecar for
`owb_mynn_diagnostics`), `--mode full`'s grid (the legacy full-mode numbers), and
`--ts-factor` (`timestep.scale`, RiRk-only per `vertical_ts`'s historical convention but
harmless to pass unconditionally since `opts.grid == :rirk` is required above).
"""
function owb_config_set(opts::BenchmarkOptions)
    set = String["output.dir=" * benchmark_output_dir("ocean_warm_bubble", opts),
                 "output.formats=[\"csv\"]", "output.sidecars=true",
                 "timestep.scale=$(opts.ts_factor)"]
    if opts.mode == :full
        append!(set, ["grid.num_cells=[100]", "grid.num_cells_k=84", "grid.z_top=25200.0",
                      "timestep.ts=0.25", "output.interval=600.0"])
    end
    return set
end

# ── Surface-flux budget and boundary-layer rows ──────────────────────────────
#
# The model writes no surface-flux output, so `Scythe.surface_exchange` -- the SAME function
# `mc_louis_bl!` calls, not a second copy of the bulk formulas -- is re-evaluated on every
# snapshot's lowest mish level (the same inputs the scheme uses) and integrated in time with
# the trapezoid rule, exactly as the precipitation energy flux is. Going through the model's
# own function is what keeps these rows right for :gfdl_v7, :charnock and the stability arm
# instead of silently reporting Komori numbers for a run that used something else.
# Domain-mean quantities are per unit width [.../m^2] like the rain rows.

function owb_surface_diagnostics(model, ref, kDim)
    pp = model.physical_params
    louis = get(model.options, :louis_bl, false)
    mynn = get(model.options, :mynn, false)
    # Either closure applies the surface stress, on the same `g(z)` and from the same
    # `surface_exchange` call, so the drag WORK row belongs to both arms; only the Louis
    # eddy-diffusivity row below is Louis-specific.
    bl = louis === true || mynn === true
    fluxes = get(model.options, :surface_fluxes, false)
    SST = get(pp, :SST, NaN)
    sfc_fac = get(pp, :sfc_wind_factor, 1.0)
    l_inf = get(pp, :l_inf, 80.0)
    # The model's own resolved surface-layer configuration, built exactly as the driver
    # preamble builds it (src/moist_compressible.jl).
    sfc = Scythe.surface_layer_params(pp, model.options; surface_fluxes = fluxes)
    ctf = Scythe.condensate_transform_mode(model.options)
    cmu = get(pp, :condensate_mu, 1.0e-7)
    rtf = Scythe.rain_transform_mode(model.options)
    rmu = get(pp, :rain_mu, 1.0e-7)
    gp = model.grid_params
    snaps = output_snapshots(model)
    times = Float64[]; sh_int = Float64[]; lat_int = Float64[]; drag_int = Float64[]
    q_int = Float64[]
    Wh = Float64[]
    hfx_mean_final = 0.0; lhf_mean_final = 0.0; ustar_max = 0.0
    max_u_lowkm = -Inf; min_u_lowkm = Inf; max_K = 0.0
    for (t, path) in snaps
        df = CSV.read(path, DataFrame)
        ncols = div(nrow(df), kDim)
        surf = 1:kDim:nrow(df)
        Tk, p, rho_d, rho_v, _, rho_t, _ =
            mc_state(df, ref, kDim, ncols; transform = ctf, mu = cmu,
                     rain_transform = rtf, rain_mu = rmu)
        if isempty(Wh)
            Wh = gauss_cell_weights(ncols, gp.num_cells, gp.iMax - gp.iMin,
                                    ncols ÷ gp.num_cells, gp.quadrature)
        end
        z1 = df.z[surf]
        # One call per column into the model's surface layer: v = 0 on the XZ slice, and
        # `sfc_wind_factor` is applied inside (so `u1` below repeats it only to form the
        # drag WORK tau_u*u1, which is what the resolved flow actually loses).
        u_s = df.u[surf]
        sx = [Scythe.surface_exchange(u_s[j], 0.0, Tk[surf][j], rho_d[surf][j],
                                      rho_t[surf][j], rho_v[surf][j], p[surf][j], z1[j],
                                      SST, sfc) for j in 1:ncols]
        u1 = u_s .* sfc_fac
        ust = [r.ust for r in sx]
        F_sh = [r.F_sh for r in sx]
        F_q = [r.F_q for r in sx]
        ke1 = 0.5 .* (df.u[surf] .^ 2 .+ df.w[surf] .^ 2)
        e_v = ((Scythe.Cpv - Scythe.Rv) .* Tk[surf]) .+ ke1 .+ (Scythe.gravity .* z1)   # c_w + c_v
        drag = bl ? [r.tau_u for r in sx] .* u1 : zeros(ncols)                         # tau_u * u1
        push!(times, t)
        push!(sh_int, sum(Wh .* F_sh))
        push!(lat_int, sum(Wh .* (e_v .* F_q)))
        push!(drag_int, sum(Wh .* drag))
        push!(q_int, sum(Wh .* F_q))
        hfx_mean_final = sum(Wh .* F_sh) / (gp.iMax - gp.iMin)
        lhf_mean_final = sum(Wh .* (Scythe.L_v.(Tk[surf]) .* F_q)) / (gp.iMax - gp.iMin)
        ustar_max = max(ustar_max, maximum(ust))
        low = df.z .<= 1000.0
        max_u_lowkm = max(max_u_lowkm, maximum(df.u[low]))
        min_u_lowkm = min(min_u_lowkm, minimum(df.u[low]))
        if louis
            l = Scythe.louis_length.(df.z, l_inf)
            max_K = max(max_K, maximum(l .^ 2 .* abs.(df.u_z)))
        end
    end
    trap(v) = sum(0.5 * (v[i] + v[i+1]) * (times[i+1] - times[i]) for i in 1:(length(times) - 1); init = 0.0)
    width = gp.iMax - gp.iMin
    return Dict(
        "sfc_sensible_Jm2" => trap(sh_int) / width,
        "sfc_latent_energy_Jm2" => trap(lat_int) / width,   # (C_vv T + ke + gz) F_q
        "sfc_drag_work_Jm2" => trap(drag_int) / width,      # resolved KE removed by the drag
        "sfc_water_input_mm" => trap(q_int) / width,
        "sfc_hfx_mean_final_Wm2" => hfx_mean_final,
        "sfc_lhf_mean_final_Wm2" => lhf_mean_final,
        "sfc_ustar_max" => ustar_max,
        "max_u_lowkm" => max_u_lowkm,
        "min_u_lowkm" => min_u_lowkm,
        "max_K_louis_m2s" => max_K,
    )
end

# ── MYNN-EDMF boundary-layer rows (S9: read from the sidecar) ───────────────────
#
# Stage S9 gives the closure its own sidecar NetCDF (`Scythe.read_mynn`/`mynn_snapshots`,
# src/mynn_io.jl), written on the same output cadence as `<t>_physical.csv`. Everything
# below used to come from two much weaker sources -- `rho_e/rho_t` reconstructed from the
# physical CSV, and the tile's `mynn census:` stdout line regex-parsed back out of
# `scythe_out.log` (which needed the run's stdout captured at all, and only ever carried
# the counters' FINAL values, never a per-snapshot maximum) -- both retired here in favor
# of reading the closure's own held state directly. `NaN` reported for a domain mean whose
# denominator is empty (a run with nothing in the lowest km/2 km, or zero columns) rather
# than a `0/0` that would misread as "exactly zero".
#
# ALL INFORMATIONAL: nothing here is gated in `check_targets` (unlike the rain/ice/surface
# rows above it), the same status the retired census counters always had.
function owb_mynn_diagnostics(model, ref, kDim)
    get(model.options, :mynn, false) === true || return Dict{String,Float64}()
    tags = Scythe.mynn_snapshots(model.output_dir)
    # No sidecar: `:mynn_output = false`, or a run predating S9. Empty, not NaN-filled --
    # there is nothing to report, and an empty Dict merges into `owb_diagnostics`'s table
    # without adding or displacing any row.
    isempty(tags) && return Dict{String,Float64}()

    gp = model.grid_params
    max_K_m = 0.0; max_K_h = 0.0; max_e_lowkm = 0.0; max_pblh = 0.0
    max_mass_flux = 0.0; max_ztop = 0.0
    mean_e_lowkm = NaN; mean_pblh_final = NaN
    mean_P_s = NaN; mean_P_b = NaN; mean_eps = NaN
    n_clamp_e = NaN; n_cap_K = NaN; n_diffnum = NaN
    plume_active_frac = 0.0
    final_tag = tags[end]

    for tag in tags
        snap = Scythe.read_mynn(model.output_dir, tag)
        ncols = length(snap.x)
        max_K_m = max(max_K_m, maximum(snap.K_m))
        max_K_h = max(max_K_h, maximum(snap.K_h))
        max_pblh = max(max_pblh, maximum(snap.pblh))
        max_mass_flux = max(max_mass_flux, maximum(snap.aw_max))
        max_ztop = max(max_ztop, maximum(snap.plume_ztop))
        low = snap.z .<= 1000.0
        any(low) && (max_e_lowkm = max(max_e_lowkm, maximum(view(snap.e, :, low))))

        tag == final_tag || continue

        n_clamp_e = Float64(snap.n_clamp_e)
        n_cap_K = Float64(snap.n_cap_K)
        n_diffnum = Float64(snap.n_diffnum)
        nact = count(>(0.0), snap.plume_ktop)
        plume_active_frac = ncols > 0 ? nact / ncols : 0.0

        # Gauss-weighted domain means, in BOTH directions -- the same quadrature every
        # other integral in this file uses (`gauss_cell_weights`, `domain_integral`).
        Wh = gauss_cell_weights(ncols, gp.num_cells, gp.iMax - gp.iMin,
                                ncols ÷ gp.num_cells, gp.quadrature)
        Wv = gauss_cell_weights(kDim, gp.num_cells_k, gp.kMax - gp.kMin,
                                gp.mubar, gp.quadrature)
        num_e = 0.0; den_e = 0.0
        num_ps = 0.0; num_pb = 0.0; num_eps = 0.0; den_2km = 0.0
        for c in 1:ncols, k in 1:kDim
            z = snap.z[k]
            wgt = Wh[c] * Wv[k]
            if z <= 1000.0
                num_e += wgt * snap.e[c, k]
                den_e += wgt
            end
            if z <= 2000.0
                num_ps += wgt * snap.P_s[c, k]
                num_pb += wgt * snap.P_b[c, k]
                num_eps += wgt * snap.eps[c, k]
                den_2km += wgt
            end
        end
        mean_e_lowkm = den_e > 0.0 ? num_e / den_e : NaN
        mean_P_s = den_2km > 0.0 ? num_ps / den_2km : NaN
        mean_P_b = den_2km > 0.0 ? num_pb / den_2km : NaN
        mean_eps = den_2km > 0.0 ? num_eps / den_2km : NaN
        sum_wh = sum(Wh)
        mean_pblh_final = sum_wh > 0.0 ?
            sum(Wh[c] * snap.pblh[c] for c in 1:ncols) / sum_wh : NaN
    end

    # `Dict{String,Float64}`, not `Dict{String,Any}`: `owb_diagnostics` merges these rows
    # into the harness's target dictionary, and `check_targets` dispatches on the concrete
    # element type.
    return Dict{String,Float64}(
        "max_K_m_m2s" => max_K_m, "max_K_h_m2s" => max_K_h,
        "max_e_lowkm" => max_e_lowkm, "mean_e_lowkm" => mean_e_lowkm,
        "max_pblh_m" => max_pblh, "mean_pblh_final_m" => mean_pblh_final,
        "mynn_n_clamp_e" => n_clamp_e, "mynn_n_cap_K" => n_cap_K,
        "mynn_n_diffnum" => n_diffnum,
        "mynn_plume_active_frac" => plume_active_frac,
        "mynn_max_mass_flux" => max_mass_flux, "mynn_max_ztop_m" => max_ztop,
        "mean_P_s_lowkm_Wm3" => mean_P_s, "mean_P_b_lowkm_Wm3" => mean_P_b,
        "mean_eps_lowkm_Wm3" => mean_eps,
    )
end

function owb_diagnostics(model)
    df = read_final_output(model)
    ref, _, kDim = rebuild_reference(model)

    diags = merge(o01_rain_diagnostics(model, ref, kDim),
                  o01_ice_diagnostics(model, ref, kDim),
                  o01_radiation_diagnostics(model, ref, kDim),
                  owb_surface_diagnostics(model, ref, kDim),
                  owb_mynn_diagnostics(model, ref, kDim))
    diags["max_w"] = maximum(df.w)
    diags["min_w"] = minimum(df.w)

    snaps = output_snapshots(model)
    wp0 = water_path_mm(snaps[1][2], model, ref, kDim)
    wpN = water_path_mm(snaps[end][2], model, ref, kDim)
    diags["water_path_change_mm"] = wpN - wp0
    # rain-out = (surface vapour input) - (water path gain); the O01 definition with a source
    diags["accum_rainfall_mm"] = diags["sfc_water_input_mm"] - (wpN - wp0)

    drift = conservation_drift(model, ref)
    E0 = begin
        df0 = CSV.read(snaps[1][2], DataFrame)
        ncols = div(nrow(df0), kDim)
        E_t = df0.E_t .+ repeat(Springsteel.ref_total_energy(ref)[:, 1], ncols)
        domain_integral(reshape(E_t, kDim, ncols), model)
    end
    width = model.grid_params.iMax - model.grid_params.iMin
    predicted_pct = 100.0 * width / E0 *
        (diags["precip_energy_gain_Jm2"] + diags["sfc_sensible_Jm2"] +
         diags["sfc_latent_energy_Jm2"] - diags["sfc_drag_work_Jm2"])
    diags["sfc_energy_input_pct"] = 100.0 * width / E0 *
        (diags["sfc_sensible_Jm2"] + diags["sfc_latent_energy_Jm2"] - diags["sfc_drag_work_Jm2"])
    diags["energy_residual_pct"] = drift["energy_drift_pct"] - predicted_pct

    return merge(diags, drift)
end

# ── Run ────────────────────────────────────────────────────────────────────
#
# `cfg` is built once here (not inside `owb_config_set`, which only computes overrides) so
# both `build_model` and `initialize_case!`/the arm derivation below share the SAME
# resolved config -- `build_model`'s own `validate(cfg)` call is what actually front-loads
# the config errors, same as every other launcher.

if abspath(PROGRAM_FILE) == @__FILE__
    cfg = load_config(OWB_CONFIG; set = owb_config_set(opts))
    model = build_model(cfg)

    # The arm tag (output/reference-CSV naming) is read off the RESOLVED config/options,
    # not an env var -- it always describes what actually ran. `addarm` composes suffixes
    # the same way the legacy env-var arms did, so existing arm-qualified reference CSVs
    # (owb_default has none committed; this is for future --update-reference runs) keep
    # their naming convention.
    addarm(a, s) = isempty(s) ? a : (isempty(a) ? s : "$(a)_$(s)")
    arm = Scythe.ice_microphysics(model.options) === :ishmael ? "ice" : ""
    bl = cfg.physics.boundary_layer.scheme
    bl === :mynn || (arm = addarm(arm, String(bl)))
    get(model.options, :mynn_edmf, 0) == 1 && (arm = addarm(arm, "edmf"))
    # Anything that is not the MODEL DEFAULT fidelity gets its own suffix (`_fid_<names>`),
    # so it never looks up the default arm's expected values. The default is the `:scythe`
    # bundle, so an absent key -- and an explicit `:scythe`, which is the same run --
    # carries NO suffix, while the verbatim-Fortran control gets `_fid_fortran`. A
    # deviation list gets `_fid_<names joined by _>`; an EMPTY list resolves to `:fortran`.
    mynn_fid = get(model.options, :mynn_fidelity, :scythe)
    (mynn_fid isa AbstractVector && isempty(mynn_fid)) && (mynn_fid = :fortran)
    if mynn_fid isa Symbol
        mynn_fid === :scythe || (arm = addarm(arm, "fid_" * String(mynn_fid)))
    elseif mynn_fid isa AbstractVector
        arm = addarm(arm, "fid_" * join(string.(mynn_fid), "_"))
    end
    # A non-default SURFACE choice gets its own suffix, so a komori/non-stability run and a
    # gfdl_v7/charnock or stability arm never look up the same expected values.
    arm = addarm(arm, get(model.options, :sfc_z0, :komori) === :komori ? "" :
                      String(model.options[:sfc_z0]))
    get(model.options, :sfc_stability, false) && (arm = addarm(arm, "stab"))

    passed = run_benchmark("ocean_warm_bubble", opts;
                           model = model,
                           init! = function (m)
                               # The output directory is REUSED across runs (a fixed name
                               # from `benchmark_output_dir`); clear stale snapshot output
                               # before writing fresh initial conditions, as the legacy
                               # `owb_init!` did, so a re-run never mixes old and new
                               # `_physical.csv` files.
                               isdir(m.output_dir) && for f in readdir(m.output_dir)
                                   endswith(f, "_physical.csv") && rm(joinpath(m.output_dir, f))
                               end
                               initialize_case!(cfg, m)
                           end,
                           diagnostics = owb_diagnostics,
                           varnames = Scythe.mc_var_names(model.options),
                           plotter = nothing,
                           arm = arm)
    exit(passed ? 0 : 1)
end
