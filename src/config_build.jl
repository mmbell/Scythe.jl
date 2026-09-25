# ── RunConfig → ModelParameters: the single home of the physics expansion ─────────
#
# `build_model(cfg)` is the BACK HALF of the consolidated configuration (src/config.jl is
# the front half). It replaces three copies of the same expansion -- tc/tc_init.jl
# `make_base`/`tc_boundary_conditions`, benchmarks/ocean_rce.jl `rce_model`,
# benchmarks/ocean_warm_bubble.jl `owb_model` -- each of which turned its own environment
# variables into `options`/`physical_params`, the variable list, the four BC dicts and the
# positivity rule.
#
# WHAT IS EMITTED. For each physics key the rule is "exactly what the legacy drivers emitted
# for the same physics", which test/test_config_equivalence.jl checks key by key against the
# Stage-0 golden fingerprints (test/reference_data/legacy_models/). Where the legacy drivers
# disagreed about whether to write a key whose value equals the library `get` default
# (e.g. `:state_dependent_si => false`, `:Pr_t => 1.0`), the key is written UNIFORMLY here
# and the test lists it in `IMPLICIT_DEFAULTS`: writing a library default changes nothing
# the model reads. Keys whose library default differs from a sensible config value, or that
# only mean something when a scheme is on, are written only then (radiation keys only with
# radiation; `:sfc_z0` only when not komori; the timestep-advisory keys only when not the
# library default; `:f` only where the geometry rotates).
#
# ORDER. The physics is resolved FIRST because it decides the variable list
# (`mc_var_names`: transforms rename slots 8/9, two-moment rain and ice append slots, MYNN
# appends `rho_e`), and every name-keyed dict (BCs, positivity, max_wavenumber) is then
# built from that list -- never from literals, so a transform can never leave a stale
# `rho_r` key behind (see `check_mc_var_names`).

"""
    validate(cfg::RunConfig) -> nothing

Cross-section rules a well-formed `RunConfig` must still satisfy to be BUILDABLE (the
constructor has already checked enums and the equation-set/geometry pairing). Errors name
the config keys involved.
"""
function validate(cfg::RunConfig)
    es = EQUATION_SETS[cfg.run.equation_set]
    g = cfg.grid
    mp = cfg.physics.microphysics
    bl = cfg.physics.boundary_layer
    rad = cfg.physics.radiation
    hm = cfg.physics.horizontal_mixing
    num = cfg.numerics
    ice = mp.ice === :ishmael

    cfg.run.equation_set == "moist_compressible_RRR" && error(
        "config: run.equation_set = \"moist_compressible_RRR\" is reserved for the 3-D " *
        "case and is not yet supported by build_model")

    # ── grid / timestep shape
    n = length(g.num_cells)
    n >= 1 || error("config: grid.num_cells must have at least one entry")
    length(g.boundaries) == n + 1 || error(
        "config: grid.boundaries has $(length(g.boundaries)) entries but grid.num_cells has " *
        "$n; boundaries must have length(num_cells) + 1 (the patch edges)")
    all(diff(g.boundaries) .> 0.0) || error(
        "config: grid.boundaries = $(g.boundaries) must be strictly increasing")
    all(>(0), g.num_cells) || error("config: grid.num_cells must all be positive")
    g.num_cells_k > 0 || error("config: grid.num_cells_k must be positive")
    g.z_top > 0.0 || error("config: grid.z_top must be positive")
    ts = cfg.timestep
    (ts.ts > 0.0 && ts.scale > 0.0) || error(
        "config: timestep.ts and timestep.scale must be positive")
    isempty(ts.ts_per_patch) || length(ts.ts_per_patch) == n || error(
        "config: timestep.ts_per_patch has $(length(ts.ts_per_patch)) entries for $n " *
        "patch(es) (grid.num_cells); give one per patch or leave it empty")
    all(>(0.0), ts.ts_per_patch) || error("config: timestep.ts_per_patch must all be positive")
    g.lateral === :periodic && es.cyl && error(
        "config: grid.lateral = \"periodic\" is refused for the cylindrical equation set " *
        "run.equation_set = \"$(cfg.run.equation_set)\" (axis + outer wall)")
    g.lateral === :periodic && n > 1 && error(
        "config: grid.lateral = \"periodic\" with $n patches: nesting is not periodic")

    # ── run / output
    cfg.run.integration_time > 0.0 || error("config: run.integration_time must be positive")
    cfg.run.segment_length >= 0.0 || error("config: run.segment_length must be >= 0")
    (cfg.run.segment_length > 0.0 && cfg.output.restart_interval <= 0.0) && error(
        "config: run.segment_length = $(cfg.run.segment_length) needs " *
        "output.restart_interval > 0: each segment restarts from the previous one's final " *
        "checkpoint, which finalize_model writes only when restart_interval > 0")
    cfg.output.interval > 0.0 || error("config: output.interval must be positive")

    # ── physics pairings
    (ice && bl.scheme === :louis) && error(
        "config: physics.microphysics.ice = \"ishmael\" with physics.boundary_layer.scheme = " *
        "\"louis\" is refused: Louis infers the vapour tendency as rho_dot_w - rho_dot_c, " *
        "which attributes the ice flux to vapour. Use scheme = \"mynn\" (or \"none\")")
    (ice && mp.rain_moments == 1) && error(
        "config: physics.microphysics.ice = \"ishmael\" requires two-moment rain; set " *
        "physics.microphysics.rain_moments = 2 (or 0 = auto)")
    (ice && hm.water) && error(
        "config: physics.horizontal_mixing.water = true is refused with " *
        "physics.microphysics.ice = \"ishmael\" (the water mixing is not energy consistent " *
        "and is refused by the ice set)")
    (ice && cfg.diagnostics.water_budget_trace > 0) && error(
        "config: diagnostics.water_budget_trace > 0 is refused with " *
        "physics.microphysics.ice = \"ishmael\" (the budget probe is warm-rain only)")
    (num.exact_si && num.state_dependent_si) && error(
        "config: numerics.exact_si and numerics.state_dependent_si are structurally " *
        "incompatible; set one of them to false")
    (hm.scheme === :smagorinsky && !es.cyl) && error(
        "config: physics.horizontal_mixing.scheme = \"smagorinsky\" exists only for the " *
        "cylindrical equation sets (axisym, RLR); run.equation_set = " *
        "\"$(cfg.run.equation_set)\" is horizontally inviscid apart from the spline filter")
    (rad.solar === :diurnal && !(-90.0 <= rad.latitude <= 90.0)) && error(
        "config: physics.radiation.solar = \"diurnal\" needs physics.radiation.latitude in " *
        "[-90, 90], got $(rad.latitude)")
    if bl.scheme === :mynn
        tmax = maximum(_patch_ts_requested(cfg))
        bl.mynn_interval < tmax && error(
            "config: physics.boundary_layer.mynn_interval = $(bl.mynn_interval) s is shorter " *
            "than the time step ($(tmax) s after timestep.scale); it must be >= ts")
    end
    rad.scheme === :rrtmgp && require_rrtmgp_extension()
    return nothing
end

"Requested per-patch time steps after `timestep.scale` (before `build_nest`'s integer subdivision)."
function _patch_ts_requested(cfg::RunConfig)
    t = cfg.timestep
    n = length(cfg.grid.num_cells)
    return isempty(t.ts_per_patch) ? fill(t.ts * t.scale, n) : t.ts_per_patch .* t.scale
end

"Resolved `rain_moments`: 0 = auto (2 under ice, else 1)."
_resolved_rain_moments(mp::Microphysics) =
    mp.rain_moments != 0 ? mp.rain_moments : (mp.ice === :ishmael ? 2 : 1)

"Resolved radiative-heating taper top: 0 = the sponge onset."
_resolved_radiation_z_max(cfg::RunConfig) =
    cfg.physics.radiation.z_max != 0.0 ? cfg.physics.radiation.z_max : cfg.physics.sponge.z_damp

"""
    build_physics(cfg; start_hour = cfg.physics.radiation.start_hour) -> (options, physical_params)

The physics sections of `cfg` as the `options::Dict{Symbol,Any}` and
`physical_params::Dict{Symbol,Float64}` the moist-compressible model reads, with
`[extra.*]` merged last. See the file header for which keys are emitted when.
"""
function build_physics(cfg::RunConfig; start_hour::Float64 = cfg.physics.radiation.start_hour)
    o = Dict{Symbol,Any}()
    p = Dict{Symbol,Float64}()
    ph = cfg.physics
    mp, bl, sfc, rad, hm, sp = ph.microphysics, ph.boundary_layer, ph.surface,
                               ph.radiation, ph.horizontal_mixing, ph.sponge
    num, diag, out = cfg.numerics, cfg.diagnostics, cfg.output

    # ── solver and reference state
    o[:semiimplicit] = true
    o[:exact_reference_state] = cfg.reference_state.exact
    o[:consistent_qss_reference] = cfg.reference_state.consistent_qss
    o[:hydrostatic_reference] = cfg.reference_state.hydrostatic
    o[:state_dependent_si] = num.state_dependent_si
    o[:exact_si] = num.exact_si
    num.horizontal_semiimplicit && (o[:horizontal_semiimplicit] = true)
    o[:wall_bc_tau] = num.wall_bc_tau
    num.cfl_interval > 0.0 && (o[:cfl_interval] = num.cfl_interval)
    o[:vertical_mixing] = false      # legacy primitive-equation switch; the BL owns vertical mixing

    # ── microphysics
    ice = mp.ice === :ishmael
    nmom = _resolved_rain_moments(mp)
    o[:ice_microphysics] = mp.ice
    o[:rain_moments] = nmom
    o[:precipitation] = mp.precipitation
    o[:condensation] = mp.condensation
    o[:condensate_transform] = mp.condensate_transform
    o[:rain_transform] = mp.rain_transform
    nmom == 2 && (o[:rain_number_transform] = mp.rain_number_transform)
    ice && (o[:ice_transform] = mp.ice_transform)
    p[:N_0] = mp.N_0
    p[:N_r] = mp.N_r
    p[:tau_qss] = mp.tau_qss
    # The bhyp widths: the library default (1e-7, moist_compressible.jl) is what every
    # legacy driver ran with and never wrote; write only a deliberate change.
    mp.condensate_mu != 1.0e-7 && (p[:condensate_mu] = mp.condensate_mu)
    mp.rain_mu != 1.0e-7 && (p[:rain_mu] = mp.rain_mu)

    # ── boundary layer + surface layer
    if bl.scheme === :mynn
        o[:mynn] = true
        o[:mynn_interval] = bl.mynn_interval
        o[:mynn_edmf] = bl.mynn_edmf ? 1 : 0
        o[:mynn_fidelity] = parse_mynn_fidelity(bl.mynn_fidelity)
        o[:mynn_init] = bl.mynn_init
        isfinite(bl.K_max) && (p[:mynn_K_max] = bl.K_max)
    elseif bl.scheme === :louis
        o[:louis_bl] = true
    end
    p[:l_inf] = bl.l_inf
    o[:surface_fluxes] = sfc.fluxes
    sfc.roughness === :komori || (o[:sfc_z0] = sfc.roughness)
    sfc.stability && (o[:sfc_stability] = true)
    p[:Cd] = sfc.Cd
    p[:Ck] = sfc.Ck
    p[:U_min] = sfc.U_min

    # ── radiation (no keys at all when off: mc_radiation_state is then EMPTY_RADIATION)
    if rad.scheme !== :none
        o[:radiation] = rad.scheme
        o[:radiation_method] = rad.method
        o[:solar] = rad.solar
        o[:radiation_forcing] = rad.forcing
        o[:radiation_interval] = rad.interval
        o[:radiation_z_max] = _resolved_radiation_z_max(cfg)
        if rad.solar === :diurnal
            p[:latitude] = rad.latitude
            p[:start_doy] = rad.start_doy
            p[:start_hour] = start_hour
        end
        p[:sfc_albedo] = rad.albedo
        p[:sfc_emissivity] = rad.emissivity
    end

    # ── horizontal mixing (vertical mixing belongs to the BL; the Kvdiff* are required keys)
    if hm.scheme === :smagorinsky
        p[:Ls] = hm.Ls
        p[:K_min] = hm.K_min
        p[:Khdiff] = 0.0
        p[:Khdiff_heat] = hm.heat ? -1.0 : 0.0      # < 0 sentinel: Smagorinsky K/Pr_t
        p[:Khdiff_water] = hm.water ? -1.0 : 0.0    # < 0 sentinel: Smagorinsky K/Sc_t
    elseif hm.scheme === :constant
        p[:Khdiff] = hm.K
        p[:Khdiff_heat] = hm.heat ? hm.K / hm.Pr_t : 0.0
        p[:Khdiff_water] = hm.water ? hm.K / hm.Sc_t : 0.0
    else
        p[:Khdiff] = 0.0
        p[:Khdiff_heat] = 0.0
        p[:Khdiff_water] = 0.0
    end
    p[:Pr_t] = hm.Pr_t
    p[:Sc_t] = hm.Sc_t
    p[:Kvdiff] = 0.0
    p[:Kvdiff_heat] = 0.0
    p[:Kvdiff_water] = 0.0

    # ── sponge, environment
    p[:alpha] = sp.alpha
    p[:z_damp] = sp.z_damp
    # Exactly the legacy F_COR expression (tc_params.jl), so f matches bit for bit. The
    # Cartesian slice carries no rotation and never reads :f (moist_compressible.jl).
    cfg.run.equation_set == "moist_compressible_XZ" ||
        (p[:f] = 2.0 * 7.292e-5 * sind(cfg.environment.f_latitude))
    p[:SST] = cfg.environment.sst

    # ── output
    o[:output_formats] = copy(out.formats)
    out.netcdf_grid === :regular || (o[:netcdf_grid] = out.netcdf_grid)
    out.netcdf_derivatives && (o[:netcdf_derivatives] = true)
    o[:mynn_output] = out.sidecars
    o[:radiation_output] = out.sidecars

    # ── diagnostics: only what differs from the library default (warn/print-only keys)
    diag.state_minima_trace > 0 && (o[:state_minima_trace] = diag.state_minima_trace)
    diag.water_budget_trace > 0 && (o[:water_budget_trace] = diag.water_budget_trace)
    diag.stiffness_trace > 0 && (o[:stiffness_trace] = diag.stiffness_trace)
    diag.mynn_trace || (o[:mynn_trace] = false)
    diag.radiation_trace || (o[:radiation_trace] = false)
    diag.radiation_trace_sw && (o[:radiation_trace_sw] = true)
    adv = diag.timestep_advisory     # library defaults: reference_state.jl warn_timestep_stability
    adv.u_max != 90.0 && (o[:u_max] = adv.u_max)
    adv.w_max != 25.0 && (o[:w_max] = adv.w_max)
    adv.bl_K_ref != 3000.0 && (o[:bl_K_ref] = adv.bl_K_ref)
    adv.state_deviation != 0.25 && (o[:state_deviation] = adv.state_deviation)

    # ── the escape hatch, last (already validated and typed by the RunConfig constructor)
    merge!(o, cfg.extra.options)
    merge!(p, cfg.extra.physical_params)
    return o, p
end

"""
    build_boundary_conditions(cfg, options, vars) -> (BCL, BCR, BCB, BCT)

The four BC dicts for the base patch, from ONE rule set covering the three legacy drivers:

- lateral, cylindrical (axisym, RLR): axis = scalars Neumann + u, v Dirichlet; outer wall =
  scalars Neumann + u Dirichlet (nest junctions are set by `build_nest`).
- lateral, XZ walls: scalars Neumann + u, w Dirichlet on both sides.
- lateral, XZ periodic: `PeriodicBC` on every variable.
- vertical (`numerics.wall_bc`): `d2` SecondDerivativeBC on every scalar, `natural`
  NaturalBC, `neumann` NeumannBC, `r1t1x` d2 with the acoustic set (`rho_d, rho_t, E_t,
  Q_ss`) Neumann and `p` on `CubicBSpline.R1T1X`; then `w` Dirichlet at both ends.
- sedimentation: rain (+ rain number, + the twelve ice slots) NaturalBC at the ground so
  they can fall out; at the lid too when `numerics.sedimentation_top_bc = natural`, left on
  the wall condition when `wall`.
"""
function build_boundary_conditions(cfg::RunConfig, options, vars::Vector{String})
    es = EQUATION_SETS[cfg.run.equation_set]
    if es.cyl
        scalar = Dict{String,Any}(v => NeumannBC() for v in vars)
        bcl = merge(scalar, Dict{String,Any}("u" => DirichletBC(), "v" => DirichletBC()))
        bcr = merge(scalar, Dict{String,Any}("u" => DirichletBC()))
    elseif cfg.grid.lateral === :periodic
        bcl = Dict{String,Any}(v => PeriodicBC() for v in vars)
        bcr = bcl
    else
        scalar = Dict{String,Any}(v => NeumannBC() for v in vars)
        bcl = merge(scalar, Dict{String,Any}("u" => DirichletBC(), "w" => DirichletBC()))
        bcr = bcl
    end

    mode = cfg.numerics.wall_bc
    vert = if mode === :d2
        Dict{String,Any}(v => SecondDerivativeBC() for v in vars)
    elseif mode === :natural
        Dict{String,Any}(v => NaturalBC() for v in vars)
    elseif mode === :neumann
        Dict{String,Any}(v => NeumannBC() for v in vars)
    else # :r1t1x
        d = Dict{String,Any}(v => SecondDerivativeBC() for v in vars)
        for v in ("rho_d", "rho_t", "E_t", "Q_ss")
            d[v] = NeumannBC()
        end
        d["p"] = Springsteel.CubicBSpline.R1T1X
        d
    end

    sediment = [rain_var_name(options)]
    rain_moments(options) == 2 && push!(sediment, rain_number_var_name(options))
    ice_microphysics(options) === :ishmael && append!(sediment, ice_var_names(options))

    bcb = merge(vert, Dict{String,Any}("w" => DirichletBC()))
    bct = merge(vert, Dict{String,Any}("w" => DirichletBC()))
    for v in sediment
        bcb[v] = NaturalBC()
        cfg.numerics.sedimentation_top_bc === :natural && (bct[v] = NaturalBC())
    end
    return bcl, bcr, bcb, bct
end

"""
    build_positivity(cfg, options) -> Dict

`numerics.positivity = auto`: the untransformed rain slot (`rho_r`) is bounded on `:k`,
and on `:i` too when the lateral boundaries are walls (Springsteel refuses an `:i` bound
under a periodic BC; a cylindrical axis + wall counts as walls). A transformed rain slot
is non-negative by construction and gets no bound. `off`: no bounds.
"""
function build_positivity(cfg::RunConfig, options)
    pos = Dict{String,Any}()
    cfg.numerics.positivity === :off && return pos
    name = rain_var_name(options)
    name == "rho_r" || return pos
    walls = EQUATION_SETS[cfg.run.equation_set].cyl || cfg.grid.lateral === :walls
    pos[name] = walls ? Dict(:i => 0.0, :k => 0.0) : Dict(:k => 0.0)
    return pos
end

"""
    build_model(cfg::RunConfig; segment = nothing) -> ModelParameters | NestedModelParameters

Turn a validated `RunConfig` into the model description: a `ModelParameters` for one patch
(`length(grid.num_cells) == 1`), a `NestedModelParameters` otherwise (with
`workers_per_patch` all ones -- the driver overrides it from the execution environment).

Paths live under `root = cfg.output.dir`: `reference.ref`, `ics.csv` (nests: `build_nest`
derives `nest{i}_ics.csv`), and output in `seg_01/` (nests: `seg_01/nest{i}/`).
`integration_time` is `min(run.segment_length, run.integration_time)` when segmented.

`segment`, a NamedTuple `(index, t_offset, integration_time, initial_conditions::Vector{String})`,
overrides the output directory (`seg_<index, 2 digits>`), the integration time, the initial
conditions (single patch: the one path; nests: one per patch) and the solar clock
(`physical_params[:start_hour] = radiation.start_hour + t_offset/3600`).

Every patch is validated up front with its own actual time step (`validate_mynn_options`,
`validate_radiation_options`, `validate_output_options`, `validate_exact_si_options`,
`surface_layer_params`, `ice_microphysics`, `check_mc_var_names`) so a bad configuration
dies here, not on a worker at the first step.
"""
function build_model(cfg::RunConfig; segment::Union{Nothing,NamedTuple} = nothing)
    validate(cfg)
    es = EQUATION_SETS[cfg.run.equation_set]
    g = cfg.grid
    n = length(g.num_cells)

    start_hour = cfg.physics.radiation.start_hour +
                 (segment === nothing ? 0.0 : Float64(segment.t_offset) / 3600.0)
    options, physical = build_physics(cfg; start_hour = start_hour)

    vars = mc_var_names(options; cyl = es.cyl)
    bcl, bcr, bcb, bct = build_boundary_conditions(cfg, options, vars)
    positivity = build_positivity(cfg, options)

    # max_wavenumber: RLR carries the ring-native ragged truncation (-1 = per-ring support)
    # on every variable. The other geometries have no azimuthal/Fourier axis; the TC driver
    # passed an explicit empty Dict for axisym and the XZ drivers left the Springsteel
    # default, reproduced here so the grids fingerprint identically.
    mw = if g.geometry == "RLR"
        (; max_wavenumber = Dict{String,Int}(v => -1 for v in vars))
    elseif es.cyl
        (; max_wavenumber = Dict{String,Int}())
    else
        (;)
    end
    # The B-spline vertical is sized by cells; a Chebyshev (RZ) vertical by gridpoints.
    vsize = g.geometry == "RZ" ? (; kDim = 3 * g.num_cells_k) : (; num_cells_k = g.num_cells_k)

    grid_params = GridParameters(;
        geometry = g.geometry,
        iMin = g.boundaries[1], iMax = g.boundaries[end],
        num_cells_i = sum(g.num_cells),          # nests: per-patch grids come from build_nest
        kMin = 0.0, kMax = g.z_top,
        vsize...,
        l_q = Dict("default" => cfg.numerics.l_q),
        positivity = positivity,
        mw...,
        BCL = bcl, BCR = bcr, BCB = bcb, BCT = bct,
        vars = Dict(v => i for (i, v) in enumerate(vars)))

    root = cfg.output.dir
    seg_index = segment === nothing ? 1 : Int(segment.index)
    output_dir = joinpath(root, "seg_$(lpad(seg_index, 2, '0'))")
    integration_time = segment !== nothing ? Float64(segment.integration_time) :
        cfg.run.segment_length > 0.0 ? min(cfg.run.segment_length, cfg.run.integration_time) :
                                       cfg.run.integration_time
    ics = joinpath(root, "ics.csv")
    seg_ics = segment === nothing ? String[] : String.(collect(segment.initial_conditions))
    if segment !== nothing
        length(seg_ics) == n || error("build_model: segment.initial_conditions has " *
            "$(length(seg_ics)) path(s) for $n patch(es)")
        n == 1 && (ics = seg_ics[1])
    end

    ts_patch = _patch_ts_requested(cfg)
    base = ModelParameters(
        ts = n == 1 ? ts_patch[1] : maximum(ts_patch),
        integration_time = integration_time,
        output_interval = cfg.output.interval,
        restart_interval = cfg.output.restart_interval,
        equation_set = cfg.run.equation_set,
        initial_conditions = ics,
        output_dir = output_dir,
        ref_state_file = joinpath(root, "reference.ref"),
        grid_params = grid_params,
        physical_params = physical,
        options = options)

    if n == 1
        _validate_patch(base)
        return base
    end
    nest = NestedModelParameters(
        boundaries = copy(g.boundaries),
        num_cells = copy(g.num_cells),
        ts = ts_patch,
        workers_per_patch = ones(Int, n),
        base = base,
        initial_conditions = n > 1 && segment !== nothing ? seg_ics : String[])
    models, _ = build_nest(nest)
    foreach(_validate_patch, models)
    return nest
end

"Front-loaded per-patch validation, with the patch's own actual time step."
function _validate_patch(m::ModelParameters)
    o, pp = m.options, m.physical_params
    kDim = m.grid_params.kDim
    validate_mynn_options(o, pp, m.equation_set, m.ts, kDim)
    validate_radiation_options(o, pp, m.equation_set, m.ts, kDim)
    validate_output_options(m)
    validate_exact_si_options(m)
    surface_layer_params(pp, o; surface_fluxes = get(o, :surface_fluxes, false)::Bool)
    ice_microphysics(o)
    check_mc_var_names(m)
    return nothing
end
