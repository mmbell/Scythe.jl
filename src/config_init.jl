# ── Case initialization from a RunConfig: reference state + initial conditions ─────
#
# Factored from the three legacy `init!`s (tc/tc_init.jl `init_tc!`, benchmarks/ocean_rce.jl
# `rce_init!`, benchmarks/ocean_warm_bubble.jl `owb_init!`), which each carried their own
# copy of the reference-state step and differed only in the initial perturbation.
#
# The reference state is shared by every patch (all patches share the vertical mish), so it
# is built ONCE, on patch 1, and written to `<root>/reference.ref`. The initial conditions
# are PERTURBATIONS from that reference (and from the consistent Q̄_ss when
# `reference_state.consistent_qss` is on -- the same Q̄_ss `createModelTile` adds back, so
# the far field reduces to the reference exactly).
#
# Dropped from tc/tc_init.jl: the non-native balanced-vortex fallback
# (`options[:native_vortex_init] = false`: balance on a foreign 500 m work grid, transfer
# by linear interpolation), kept there only for A/B comparisons. The native init is 5x
# better balanced (reference/HANDOFF_INITIALIZATION.md) and is the only path here.

"""
    build_reference_state(cfg, patch, gp; path = joinpath(cfg.output.dir, "reference.ref"))
        -> Springsteel.PressureReferenceState

The shared reference state on `patch`'s vertical mish: `reference_state.sounding` →
`calculate_pressure_reference_state` (discretely hydrostatic when
`reference_state.hydrostatic`) → the converged `(p, ρ_d, ρ_v)` triple written to `path` →
re-read by `exact_pressure_reference_state` → `consistent_qss_reference` when
`reference_state.consistent_qss`. Writes `path`.
"""
function build_reference_state(cfg::RunConfig, patch, gp;
                               path::AbstractString = joinpath(cfg.output.dir, "reference.ref"))
    zcol = EQUATION_SETS[cfg.run.equation_set].zcol
    kDim = gp.kDim
    z = getGridpoints(patch)[1:kDim, zcol]
    column = reference_column(patch, gp)
    hydro = cfg.reference_state.hydrostatic
    sounding = cfg.reference_state.sounding
    isfile(sounding) || error("config: reference_state.sounding = $(repr(sounding)) does not exist")
    # The .ref file carries VALUES only, so the hydrostatic balance has to survive the round
    # trip through the density: write the CONVERGED triple and let
    # exact_pressure_reference_state re-integrate dp/dz = -g rho_t from it.
    ref_phys = Springsteel.calculate_pressure_reference_state(sounding, z, column;
                                                             hydrostatic = hydro)
    mkpath(dirname(abspath(path)))
    write_exact_ref_mc(String(path), z,
                       Springsteel.ref_pressure(ref_phys)[:, 1],
                       Springsteel.ref_rho_d(ref_phys)[:, 1],
                       Springsteel.ref_rho_v(ref_phys)[:, 1],
                       zeros(kDim))
    ref = Springsteel.exact_pressure_reference_state(String(path), z, column; hydrostatic = hydro)
    cfg.reference_state.consistent_qss &&
        (ref = consistent_qss_reference(ref, z, column; temperature = cfg.numerics.temperature,
                                        vapor_transform = cfg.physics.microphysics.vapor_transform,
                                        vapor_mu = cfg.physics.microphysics.vapor_mu,
                                        condensate_transform =
                                            cfg.physics.microphysics.condensate_transform,
                                        condensate_mu = cfg.physics.microphysics.condensate_mu))
    println("Reference: sounding = $(basename(sounding)), sfc p = " *
            "$(round(Springsteel.ref_pressure(ref)[1, 1] / 100.0, digits = 2)) hPa " *
            "(hydrostatic = $hydro, consistent_qss = $(cfg.reference_state.consistent_qss))")
    return ref
end

"""
    initialize_case!(cfg, model) -> (model, topo_or_nothing)

Write the reference state (`<root>/reference.ref`) and the initial conditions of every
patch (`model.initial_conditions`; nests: `nest{i}_ics.csv` beside it) for the `model`
returned by [`build_model`](@ref). Dispatches on `initial_conditions.method`:

- `balanced_vortex` — `balanced_vortex_native!` on the patches' own mish (every parameter
  from `[initial_conditions.balanced_vortex]`, `fcor = physical_params[:f]`);
- `temperature_noise` — `moist_temperature_noise_mc!` (seeded, RH-preserving);
- `temperature_bubble` — `moist_temperature_bubble_mc!`;
- `file` — read `initial_conditions.file.paths` (one per patch) in place; only the reference is written.

Returns the model and, for a nest, the `NestTopology` (`nothing` for one patch).
"""
function initialize_case!(cfg::RunConfig, model::Union{ModelParameters,NestedModelParameters})
    nested = model isa NestedModelParameters
    models, topo = nested ? build_nest(model) : ([model], nothing)
    base = nested ? model.base : model
    mkpath(cfg.output.dir)
    method = cfg.initial_conditions.method

    if method === :file
        paths = cfg.initial_conditions.file.paths
        length(paths) == length(models) || error("config: initial_conditions.file.paths has " *
            "$(length(paths)) path(s) for $(length(models)) patch(es)")
        # The reference is still written: the model reads it at tile construction.
        build_reference_state(cfg, createGrid(models[1].grid_params), models[1].grid_params;
                              path = base.ref_state_file)
        # The files are read IN PLACE (build_model / initial_ics_paths point every patch at
        # them): a .jld2 checkpoint has to keep its extension for load_initial_conditions!.
        for (src, m) in zip(paths, models)
            isfile(src) || error("config: initial_conditions.file.paths entry $(repr(src)) does not exist")
            abspath(src) == abspath(m.initial_conditions) || error("config: initial_conditions.file " *
                "path $(repr(src)) is not what the model was built to read ($(m.initial_conditions)); " *
                "build the model from the same config (build_model points a `file` run at its paths)")
        end
        return model, topo
    end

    patches = [createGrid(m.grid_params) for m in models]
    ref = build_reference_state(cfg, patches[1], models[1].grid_params; path = base.ref_state_file)
    es = EQUATION_SETS[cfg.run.equation_set]
    zcol = es.zcol
    o, pp = base.options, base.physical_params
    ctrans = condensate_transform_mode(o)
    rtrans = rain_transform_mode(o)
    cmu = get(pp, :condensate_mu, 1.0e-7)
    rmu = get(pp, :rain_mu, 1.0e-7)
    vtrans = vapor_transform_mode(o)
    vmu = vapor_mu(pp)

    if method === :balanced_vortex
        es.cyl || error("config: initial_conditions.method = \"balanced_vortex\" needs a " *
                        "cylindrical equation set, not $(repr(cfg.run.equation_set))")
        vx = cfg.initial_conditions.balanced_vortex
        # A single cylindrical patch still needs a NestTopology for the solver's patch order.
        vtopo = topo !== nothing ? topo :
            last(build_nest(NestedModelParameters(
                boundaries = [base.grid_params.iMin, base.grid_params.iMax],
                num_cells = [base.grid_params.num_cells_i], ts = [base.ts],
                workers_per_patch = [1], base = base)))
        balanced_vortex_native!(patches, vtopo, ref;
                                zcol = zcol, fcor = get(pp, :f, 0.0),
                                vortex_profile = vx.profile,
                                v_m = vx.v_m, r_m = vx.r_m, r_0 = vx.r_0,
                                # modified-Rankine knobs (profile = rankine): the legacy
                                # tc_params.jl values, not config fields
                                Vmax = 30.0, RMW = 50.0e3, alpha = 0.3,
                                v_top = vx.v_top, z_bt = vx.z_barotropic,
                                RH_core = vx.rh_core, r_moist = vx.r_moist,
                                z_moist = vx.z_moist, RH_max = vx.rh_max,
                                RH_bl = vx.rh_bl, z_bl = vx.z_bl,
                                moist_profile = vx.moist_profile,
                                condensate_transform = ctrans, condensate_mu = cmu,
                                rain_transform = rtrans, rain_mu = rmu,
                    vapor_transform = vtrans, vapor_mu = vmu)
    else
        for p in patches
            gpts = getGridpoints(p)
            p.physical .= 0.0
            if method === :temperature_noise
                nz = cfg.initial_conditions.temperature_noise
                moist_temperature_noise_mc!(p, gpts, ref;
                    z_max = nz.z_max, dT_max = nz.dT_max, seed = nz.seed, zcol = zcol,
                    condensate_transform = ctrans, condensate_mu = cmu,
                    rain_transform = rtrans, rain_mu = rmu,
                    vapor_transform = vtrans, vapor_mu = vmu)
            else # :temperature_bubble
                bb = cfg.initial_conditions.temperature_bubble
                moist_temperature_bubble_mc!(p, gpts, ref;
                    xc = bb.xc, xr = bb.xr, zc = bb.zc, zr = bb.zr, dT_max = bb.dT_max,
                    zcol = zcol,
                    condensate_transform = ctrans, condensate_mu = cmu,
                    rain_transform = rtrans, rain_mu = rmu,
                    vapor_transform = vtrans, vapor_mu = vmu)
            end
        end
    end

    for (p, m) in zip(patches, models)
        mkpath(dirname(m.initial_conditions))
        write_ics_csv(m.initial_conditions, p, getGridpoints(p))
    end
    return model, topo
end
