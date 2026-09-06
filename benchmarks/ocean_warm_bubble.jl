#!/usr/bin/env julia
# Mesoscale warm bubble over a fixed-SST ocean: the boundary-layer / full-physics-suite
# benchmark (MYNN-EDMF plan, stage S1; precursor of the radiative-convective-equilibrium
# case). The O01 warm bubble (o01_rainfall.jl) has no ocean and its full mode is
# gray-zone; this case runs at production-like spacing (the TC nests') with surface
# fluxes and drag ON, so a boundary-layer scheme is exercised where its assumptions hold.
#
#   julia --project=. benchmarks/ocean_warm_bubble.jl --mode quick --stage mc --grid rirk
#
# Modes: quick = 6 km x 500 m cells, ts 0.5 s; full = 3 km x 300 m cells, ts 0.25 s
# (the TC coarse / fine pairs), 300 km x 25 km, 2 h. Humidified Dunion sounding, RH-
# preserving 3 K bubble of 30 km half-width (wider than O01's 16 km so it is resolved
# on the mesoscale grid), DK83 stratospheric sponge, warm rain by default.
#
# Physics knobs (every one is a NO-OP when unset, so the committed control is reproducible):
#   SCYTHE_OWB_BL=louis|none|mynn boundary layer + surface fluxes (default louis; `mynn`
#                                 is the MYNN-EDMF arm — at stage S4 it registers and
#                                 TRANSPORTS the rho_e TKE slot with zero sources and
#                                 applies no BL tendency, so it is `none` + one tracer)
#   SCYTHE_OWB_SFC=komori|gfdl_v7|charnock   surface roughness closure (options[:sfc_z0],
#                                 default komori = the historical Komori Cd + constant Ck);
#                                 arm suffix = the value
#   SCYTHE_OWB_SFC_STAB=1         Monin-Obukhov stability functions + Beljaars gustiness
#                                 (options[:sfc_stability]); arm suffix `_stab`
#   SCYTHE_OWB_SST=302.65         sea surface temperature [K] (default tc_params SST_K)
#   SCYTHE_OWB_ICE=1              ISHMAEL ice (with two-moment rain and the bhyp transforms,
#                                 as o01_rainfall's ice arm); arm suffix `_ice`
#   SCYTHE_OWB_RAIN_MOMENTS=2     two-moment rain without ice
#   SCYTHE_OWB_RAD=lw|allsky|sw|allsky_sw|diurnal  RRTMGP arm (o01 semantics; the SST is
#                                 the radiative surface temperature), + _RAD_FORCING,
#                                 _RAD_INTERVAL, _RAD_ZMAX
#   SCYTHE_OWB_CTRANS/RTRANS=bhyp water control-variable transforms (+ _CMU, _RMU)
#   SCYTHE_OWB_WALL_BC=d2|neumann ground/lid scalar condition (default d2, the TC's;
#                                 O01 keeps Neumann)
#   SCYTHE_OWB_TS, _HOURS, _TSTOP, _OUTPUT, _NI, _NK, _XR, _ZR, _ZC, _DT   grid/run overrides
#
# Diagnostics: O01's rain / ice / radiation sets (benchmarks/common/warm_bubble_diagnostics.jl)
# plus the surface-flux budget rows below. With fluxes on the column energy books are OPEN:
# `energy_residual_pct` subtracts BOTH the precipitation energy loss and the time-integrated
# surface enthalpy, vapour-energy and drag-work inputs, and `accum_rainfall_mm` is the water
# path change corrected for the surface vapour input. No published target exists;
# `BENCHMARK_EXPECTED["ocean_warm_bubble"]` is seeded from the accepted control run.

using Distributed

include(joinpath(@__DIR__, "common", "harness.jl"))
opts = parse_benchmark_args(ARGS)
opts.stage == STAGE_MC ||
    error("ocean_warm_bubble supports only --stage mc (the total-energy set)")
opts.nests == 1 || error("ocean_warm_bubble supports only --nests 1 (nesting arrives later)")
opts.grid == :rirk || error("ocean_warm_bubble targets the RiRk (B-spline vertical) grid")

add_benchmark_workers(opts; count = opts.workers)
@everywhere using Springsteel
@everywhere using Scythe

include(joinpath(@__DIR__, "common", "diagnostics.jl"))
include(joinpath(@__DIR__, "common", "warm_bubble_diagnostics.jl"))

# ── Configuration ────────────────────────────────────────────────────────────

const DUNION_SOUNDING = joinpath(REFERENCE_DATA_DIR, "o01_rainfall", "dunion_MT_hum90.ref")
const OWB_WIDTH = 300.0e3           # [m] domain width
const OWB_XC = 150.0e3              # [m] bubble centre
const OWB_XR = parse(Float64, get(ENV, "SCYTHE_OWB_XR", "30.0e3"))   # [m] bubble half-width
const OWB_DT = parse(Float64, get(ENV, "SCYTHE_OWB_DT", "3.0"))      # [K] bubble amplitude
const OWB_ZC = parse(Float64, get(ENV, "SCYTHE_OWB_ZC", "500.0"))    # [m] bubble centre height
const OWB_ZR = parse(Float64, get(ENV, "SCYTHE_OWB_ZR", "3000.0"))   # [m] bubble half-depth
const OWB_SST = parse(Float64, get(ENV, "SCYTHE_OWB_SST", "302.65")) # [K] tc_params SST_K
const OWB_CD = -1.0                 # negative => Komori et al. (2018) wind-speed-dependent drag
const OWB_CK = 1.0e-3               # bulk enthalpy/moisture exchange coefficient
const OWB_U_MIN = 2.0               # [m/s] gustiness floor on the exchange wind
const OWB_L_INF = 80.0              # [m] asymptotic Louis mixing length
const N_R = 1.0e-3
const N_0_MP = 8.0e6                # [m^-4] Marshall-Palmer intercept

envflag(k) = ENV[k] in ("1", "true", "yes")

function owb_model(opts::BenchmarkOptions)
    if opts.mode == :full
        num_cells_i = 100       # 3 km cells over 300 km
        num_cells_k = 84        # 300 m cells
        kMax = 25.2e3
        ts = 0.25
        output_interval = 600.0
    else
        num_cells_i = 50        # 6 km cells
        num_cells_k = 50        # 500 m cells
        kMax = 25.0e3
        ts = 0.5
        output_interval = 300.0
    end
    haskey(ENV, "SCYTHE_OWB_NI") && (num_cells_i = parse(Int, ENV["SCYTHE_OWB_NI"]))
    haskey(ENV, "SCYTHE_OWB_NK") && (num_cells_k = parse(Int, ENV["SCYTHE_OWB_NK"]))
    haskey(ENV, "SCYTHE_OWB_TS") && (ts = parse(Float64, ENV["SCYTHE_OWB_TS"]))
    haskey(ENV, "SCYTHE_OWB_OUTPUT") && (output_interval = parse(Float64, ENV["SCYTHE_OWB_OUTPUT"]))
    kDim = 3 * num_cells_k
    ts = vertical_ts(ts, opts)
    integration_time = haskey(ENV, "SCYTHE_OWB_TSTOP") ? parse(Float64, ENV["SCYTHE_OWB_TSTOP"]) :
                       3600.0 * parse(Float64, get(ENV, "SCYTHE_OWB_HOURS", "2.0"))

    physical_params = Dict(:Khdiff => 0.0, :Kvdiff => 0.0,
                           :Khdiff_heat => 0.0, :Kvdiff_heat => 0.0,
                           :Kvdiff_water => 0.0,
                           :tau_qss => 10.0, :N_r => N_R, :N_0 => N_0_MP,
                           :alpha => 0.02, :z_damp => 17.0e3,
                           :Cd => OWB_CD, :Ck => OWB_CK, :U_min => OWB_U_MIN,
                           :l_inf => OWB_L_INF, :SST => OWB_SST)
    options = merge(Dict{Symbol,Any}(:semiimplicit => true, :exact_reference_state => true,
                                     :precipitation => true, :vertical_mixing => false),
                    reference_state_options())

    bl = get(ENV, "SCYTHE_OWB_BL", "louis")
    if bl == "louis"
        options[:louis_bl] = true
        options[:surface_fluxes] = true
    elseif bl == "none"
        # no boundary layer at all: the O01 physics on the mesoscale grid (SST unused)
    elseif bl == "mynn"
        # Plan stage S4: the PLUMBING only. This registers the prognostic TKE-density slot
        # `rho_e` and transports it with zero sources; no boundary-layer tendency and no
        # surface flux is applied, so the arm is the `none` arm plus one passive tracer
        # until `mc_mynn_bl!` lands at S5. Surface fluxes are deliberately NOT switched on
        # here: with no closure to carry them they would be computed and dropped.
        options[:mynn] = true
        println("OWB MYNN arm: stage S4 — the rho_e slot is registered and TRANSPORTED " *
                "with zero sources; no BL tendency and no surface fluxes until S5")
    else
        error("SCYTHE_OWB_BL must be louis | none | mynn, got \"$bl\"")
    end

    # Surface-exchange closure (src/mc_surface_layer.jl). Both are no-ops when unset:
    # :komori without stability is the historical bulk formula, bitwise.
    sfc_arm = get(ENV, "SCYTHE_OWB_SFC", "komori")
    sfc_arm in ("komori", "gfdl_v7", "charnock") ||
        error("SCYTHE_OWB_SFC must be komori | gfdl_v7 | charnock, got \"$sfc_arm\"")
    sfc_arm == "komori" || (options[:sfc_z0] = Symbol(sfc_arm))
    if haskey(ENV, "SCYTHE_OWB_SFC_STAB") && envflag("SCYTHE_OWB_SFC_STAB")
        options[:sfc_stability] = true
    end

    haskey(ENV, "SCYTHE_OWB_CTRANS") &&
        (options[:condensate_transform] = Symbol(ENV["SCYTHE_OWB_CTRANS"]))
    haskey(ENV, "SCYTHE_OWB_CMU") &&
        (physical_params[:condensate_mu] = parse(Float64, ENV["SCYTHE_OWB_CMU"]))
    haskey(ENV, "SCYTHE_OWB_RTRANS") &&
        (options[:rain_transform] = Symbol(ENV["SCYTHE_OWB_RTRANS"]))
    haskey(ENV, "SCYTHE_OWB_RMU") &&
        (physical_params[:rain_mu] = parse(Float64, ENV["SCYTHE_OWB_RMU"]))
    haskey(ENV, "SCYTHE_OWB_RAIN_MOMENTS") &&
        (options[:rain_moments] = parse(Int, ENV["SCYTHE_OWB_RAIN_MOMENTS"]))
    if haskey(ENV, "SCYTHE_OWB_ICE") && envflag("SCYTHE_OWB_ICE")
        options[:ice_microphysics] = :ishmael
        haskey(ENV, "SCYTHE_OWB_RAIN_MOMENTS") || (options[:rain_moments] = 2)
        haskey(ENV, "SCYTHE_OWB_CTRANS") || (options[:condensate_transform] = :bhyp)
        haskey(ENV, "SCYTHE_OWB_RTRANS") || (options[:rain_transform] = :bhyp)
        options[:rain_number_transform] = :bhyp
        options[:ice_transform] = :bhyp
    end

    rad_arm = get(ENV, "SCYTHE_OWB_RAD", "0")
    if rad_arm != "0" && rad_arm != ""
        options[:radiation] = :rrtmgp
        if rad_arm == "lw"
            options[:radiation_method] = :clearsky; options[:solar] = :none
        elseif rad_arm == "allsky"
            options[:radiation_method] = :allsky; options[:solar] = :none
        elseif rad_arm == "sw"
            options[:radiation_method] = :clearsky; options[:solar] = :fixed
        elseif rad_arm == "allsky_sw"
            options[:radiation_method] = :allsky; options[:solar] = :fixed
        elseif rad_arm == "diurnal"
            options[:radiation_method] = :allsky; options[:solar] = :diurnal
            physical_params[:latitude] = 20.0
            physical_params[:start_doy] = 240.0
            physical_params[:start_hour] = 0.0
        else
            error("SCYTHE_OWB_RAD = \"$rad_arm\" is not an arm; use lw, allsky, sw, " *
                  "allsky_sw, diurnal, or 0/unset")
        end
        options[:radiation_forcing] = Symbol(get(ENV, "SCYTHE_OWB_RAD_FORCING", "full"))
        options[:radiation_interval] = parse(Float64, get(ENV, "SCYTHE_OWB_RAD_INTERVAL", "300.0"))
        options[:radiation_z_max] = parse(Float64, get(ENV, "SCYTHE_OWB_RAD_ZMAX", "17.0e3"))
        physical_params[:sfc_albedo] = 0.06
        physical_params[:sfc_emissivity] = 0.98
        println("OWB radiation: arm=$rad_arm method=$(options[:radiation_method]) " *
                "solar=$(options[:solar]) forcing=$(options[:radiation_forcing]) " *
                "interval=$(options[:radiation_interval]) s z_max=$(options[:radiation_z_max]) m; " *
                "radiative surface temperature = physical_params[:SST] = $OWB_SST K")
    end

    vars = Scythe.mc_var_names(options)
    rain_name = Scythe.rain_var_name(options)
    rain_number_name = Scythe.rain_number_var_name(options)
    two_moment = Scythe.rain_moments(options) == 2
    cloud_name = Scythe.condensate_var_name(options)
    ice_on = Scythe.ice_microphysics(options) === :ishmael
    ice_names = Scythe.ice_var_names(options)

    output_dir = benchmark_output_dir("ocean_warm_bubble", opts)
    # Lateral walls as O01 (no normal flow, free-slip scalars). Ground and lid: the TC's
    # d2 condition by default -- a Neumann ground forces dp'/dz = 0 and drains a
    # surface-flux-driven boundary layer (the TC root cause of 2026-07-21).
    wall_mode = get(ENV, "SCYTHE_OWB_WALL_BC", "d2")
    vert_scalar = wall_mode == "d2" ? Dict{String,Any}(v => SecondDerivativeBC() for v in vars) :
                  wall_mode == "neumann" ? Dict{String,Any}(v => NeumannBC() for v in vars) :
                  error("SCYTHE_OWB_WALL_BC must be d2 | neumann, got \"$wall_mode\"")
    side_scalar = Dict{String,Any}(v => NeumannBC() for v in vars)
    side_bc = merge(side_scalar, Dict{String,Any}("u" => DirichletBC(), "w" => DirichletBC()))
    topbot_bc = merge(vert_scalar, Dict{String,Any}("w" => DirichletBC(), rain_name => NaturalBC()))
    two_moment && (topbot_bc[rain_number_name] = NaturalBC())
    if ice_on
        for nm in ice_names
            topbot_bc[nm] = NaturalBC()
        end
    end

    positivity = Dict("rho_r" => Dict(:i => 0.0, :k => 0.0))
    if rain_name != "rho_r"
        delete!(positivity, "rho_r")
        println("POSITIVITY: dropped \"rho_r\" — it is carried as a control variable")
    end

    grid_params = GridParameters(;
        geometry = benchmark_geometry(opts),
        iMin = 0.0,
        iMax = OWB_WIDTH,
        num_cells_i = num_cells_i,
        kMin = 0.0,
        kMax = kMax,
        vertical_size(opts; num_cells_k = num_cells_k, kDim = kDim)...,
        l_q = Dict("default" => 2.0),
        positivity = positivity,
        BCL = side_bc,
        BCR = side_bc,
        BCB = topbot_bc,
        BCT = topbot_bc,
        vars = Dict(v => i for (i, v) in enumerate(vars)),
    )

    println("OWB: $(num_cells_i) x $(num_cells_k) cells ($(OWB_WIDTH / num_cells_i / 1000) km x " *
            "$(kMax / num_cells_k) m), ts = $ts s, $(integration_time / 3600) h, output every " *
            "$output_interval s; BL = $bl, SST = $OWB_SST K, wall BC = $wall_mode, " *
            "sfc_z0 = $(get(options, :sfc_z0, :komori)), " *
            "sfc_stability = $(get(options, :sfc_stability, false)); " *
            "vars = $(join(vars, ' '))")

    return ModelParameters(
        ts = ts,
        integration_time = integration_time,
        output_interval = output_interval,
        equation_set = "moist_compressible_XZ",
        initial_conditions = joinpath(output_dir, "owb_ics.csv"),
        output_dir = output_dir,
        ref_state_file = joinpath(output_dir, "owb_exact.ref"),
        grid_params = grid_params,
        physical_params = physical_params,
        options = options,
    )
end

# ── Initial conditions (Dunion reference + wide RH-preserving bubble) ───────

function owb_init!(model)
    for f in readdir(model.output_dir)
        endswith(f, "_physical.csv") && rm(joinpath(model.output_dir, f))
    end

    patch = createGrid(model.grid_params)
    gridpoints = Scythe.getGridpoints(patch)
    kDim = model.grid_params.kDim
    z = gridpoints[1:kDim, 2]
    column = Scythe.reference_column(patch, model.grid_params)

    hydro = reference_state_hydrostatic()
    ref_phys = Springsteel.calculate_pressure_reference_state(DUNION_SOUNDING, z, column;
                                                             hydrostatic = hydro)
    pbar = Springsteel.ref_pressure(ref_phys)[:, 1]
    rho_dbar = Springsteel.ref_rho_d(ref_phys)[:, 1]
    rho_vbar = Springsteel.ref_rho_v(ref_phys)[:, 1]
    Scythe.write_exact_ref_mc(model.ref_state_file, z, pbar, rho_dbar, rho_vbar, zeros(kDim))
    ref = Springsteel.exact_pressure_reference_state(model.ref_state_file, z, column;
                                                     hydrostatic = hydro)
    if get(reference_state_options(), :consistent_qss_reference, false)
        ref = Scythe.consistent_qss_reference(ref, z, column)
    end

    rho_tbar = Springsteel.ref_rho_t(ref)[:, 1]
    pbar_z = Springsteel.ref_pressure(ref)[:, 2]
    residual = pbar_z .+ (Scythe.gravity .* rho_tbar)
    T1 = pbar[1] / (rho_dbar[1] * Scythe.Rd + rho_vbar[1] * Scythe.Rv)
    println("Reference: sfc p = $(round(pbar[1] / 100.0, digits=2)) hPa, ",
            "max hydrostatic residual = $(maximum(abs.(residual))) Pa/m, ",
            "T(z1) = $(round(T1, digits=2)) K vs SST $(get(model.physical_params, :SST, NaN)) K")

    patch.physical .= 0.0
    Scythe.moist_temperature_bubble_mc!(patch, gridpoints, ref;
                                        xc = OWB_XC, xr = OWB_XR,
                                        zc = OWB_ZC, zr = OWB_ZR, dT_max = OWB_DT)
    Scythe.write_ics_csv(model.initial_conditions, patch, gridpoints)
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
        drag = louis ? [r.tau_u for r in sx] .* u1 : zeros(ncols)                      # tau_u * u1
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

function owb_diagnostics(model)
    df = read_final_output(model)
    ref, _, kDim = rebuild_reference(model)

    diags = merge(o01_rain_diagnostics(model, ref, kDim),
                  o01_ice_diagnostics(model, ref, kDim),
                  o01_radiation_diagnostics(model, ref, kDim),
                  owb_surface_diagnostics(model, ref, kDim))
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

model = owb_model(opts)
arm = Scythe.ice_microphysics(model.options) === :ishmael ? "ice" : ""
addarm(a, s) = isempty(s) ? a : (isempty(a) ? s : "$(a)_$(s)")
bl_arm = get(ENV, "SCYTHE_OWB_BL", "louis")
bl_arm == "louis" || (arm = addarm(arm, bl_arm))
# A non-default SURFACE choice gets its own suffix, so the committed komori control and a
# gfdl_v7/charnock or stability arm never look up the same expected values.
arm = addarm(arm, get(model.options, :sfc_z0, :komori) === :komori ? "" :
                  String(model.options[:sfc_z0]))
get(model.options, :sfc_stability, false) && (arm = addarm(arm, "stab"))
passed = run_benchmark("ocean_warm_bubble", opts;
                       model = model,
                       init! = owb_init!,
                       diagnostics = owb_diagnostics,
                       varnames = Scythe.mc_var_names(model.options),
                       plotter = nothing,
                       arm = arm)
exit(passed ? 0 : 1)
