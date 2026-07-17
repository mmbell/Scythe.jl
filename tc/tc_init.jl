# ── TC nest construction and balanced-vortex initialization ─────────────────
# Include AFTER tc_params.jl (and after `@everywhere using Scythe, Springsteel`
# in distributed drivers). Provides:
#   make_base(integration_time; output_formats)  -> ModelParameters template
#   make_nest(base)                              -> NestedModelParameters
#   init_tc!(nest)                               -> writes shared reference +
#                                                   per-patch balanced ICs

const TC_VARS = Scythe.MC_VARS_CYL

function tc_boundary_conditions()
    scalar_bc = Dict(v => NeumannBC() for v in TC_VARS)
    axis_bc = merge(scalar_bc, Dict("u" => DirichletBC(), "v" => DirichletBC()))
    wall_bc = merge(scalar_bc, Dict("u" => DirichletBC()))
    bot_bc = merge(scalar_bc, Dict("w" => DirichletBC(), "rho_r" => NaturalBC()))
    top_bc = merge(scalar_bc, Dict("w" => DirichletBC()))
    return axis_bc, wall_bc, bot_bc, top_bc
end

function make_base(integration_time; output_formats=OUTPUT_FORMATS,
                   output_dir=OUTPUT_DIR,
                   initial_conditions=joinpath(output_dir, "tc_ics.csv"),
                   geometry="RiRk",
                   output_interval=OUTPUT_INTERVAL,
                   extra_options=Dict{Symbol,Any}())
    axis_bc, wall_bc, bot_bc, top_bc = tc_boundary_conditions()
    mkpath(output_dir)
    return ModelParameters(
        ts = NEST_TS[end],                     # root (outer patch) timestep
        integration_time = integration_time,
        output_interval = output_interval,
        restart_interval = RESTART_INTERVAL,
        equation_set = geometry == "RLR" ? "moist_compressible_RLR" :
                                           "moist_compressible_axisym",
        initial_conditions = initial_conditions,
        output_dir = output_dir,
        ref_state_file = joinpath(output_dir, "tc_exact.ref"),
        grid_params = GridParameters(;
            geometry = geometry,
            iMin = 0.0, iMax = NEST_BOUNDARIES[end],   # placeholders (per-patch
            num_cells_i = sum(NEST_CELLS),             # grids come from the nest)
            kMin = 0.0, kMax = Z_TOP, num_cells_k = NUM_CELLS_K,
            # RLR: ring-native ragged azimuthal truncation (the production
            # choice; max_wavenumber -1 = per-ring support)
            max_wavenumber = geometry == "RLR" ?
                Dict(v => -1 for v in TC_VARS) : Dict{String,Int64}(),
            BCL = axis_bc, BCR = wall_bc, BCB = bot_bc, BCT = top_bc,
            vars = Dict(v => i for (i, v) in enumerate(TC_VARS))),
        physical_params = Dict(:Khdiff => 0.0, :Kvdiff => 0.0,
                               :Khdiff_heat => 0.0, :Kvdiff_heat => 0.0,
                               :Kvdiff_water => 0.0,
                               :tau_qss => TAU_QSS, :N_r => 1.0e-3, :N_0 => N_0_MP,
                               :alpha => SPONGE_ALPHA, :z_damp => Z_DAMP,
                               :f => F_COR,
                               :Cd => CD, :l_inf => L_INF,
                               :Ls => LS_SMAG, :K_min => K_MIN,
                               :Ck => CK, :SST => SST_K, :U_min => U_MIN),
        options = merge(Dict{Symbol,Any}(:semiimplicit => true,
                                         :exact_reference_state => true,
                                         :precipitation => true,
                                         :vertical_mixing => false,
                                         :louis_bl => true,
                                         :surface_fluxes => true,
                                         :output_formats => output_formats),
                        extra_options))
end

function make_nest(base)
    rlr = base.grid_params.geometry == "RLR"
    return NestedModelParameters(
        boundaries = rlr ? NEST_BOUNDARIES_RLR : NEST_BOUNDARIES,
        num_cells = rlr ? NEST_CELLS_RLR : NEST_CELLS,
        ts = NEST_TS,
        workers_per_patch = NEST_WORKERS,
        base = base)
end

"""
    init_tc!(nest) -> (models, topo)

Build the nest, write the shared exact reference state from the humidified
Dunion MT sounding, solve the gradient-wind/hydrostatic balance for the
modified-Rankine vortex on a fine radial work grid (the vertical axis is the
model mish itself), and write per-patch initial conditions on the
collar-extended grids. Prints the balance-residual diagnostic.
"""
function init_tc!(nest)
    models, topo = build_nest(nest)
    println("TC nest: ts_actual = $(topo.ts_actual), n_sub = $(topo.n_sub)")
    for m in models
        mkpath(m.output_dir)
    end

    # Shared reference state on the (common) vertical mish
    patch1 = createGrid(models[1].grid_params)
    gp1 = models[1].grid_params
    kDim = gp1.kDim
    zcol = gp1.geometry == "RLR" ? 3 : 2       # z gridpoint column
    z = Scythe.getGridpoints(patch1)[1:kDim, zcol]
    column = Scythe.reference_column(patch1, gp1)
    ref_phys = Springsteel.calculate_pressure_reference_state(SOUNDING, z, column)
    Scythe.write_exact_ref_mc(nest.base.ref_state_file, z,
                              Springsteel.ref_pressure(ref_phys)[:, 1],
                              Springsteel.ref_rho_d(ref_phys)[:, 1],
                              Springsteel.ref_rho_v(ref_phys)[:, 1],
                              zeros(kDim))
    ref = Springsteel.exact_pressure_reference_state(nest.base.ref_state_file, z, column)
    println("Reference: sfc p = " *
            "$(round(Springsteel.ref_pressure(ref)[1, 1] / 100.0, digits=2)) hPa")

    # Balanced vortex on the radial work grid x model mish vertical axis
    r_outer = gp1.geometry == "RLR" ? NEST_BOUNDARIES_RLR[end] : NEST_BOUNDARIES[end]
    r_axis = collect(0.0:DR_WORK:r_outer)
    flds = Scythe.balanced_vortex_fields(r_axis, z,
                                         Springsteel.ref_pressure(ref)[:, 1],
                                         Springsteel.ref_rho_d(ref)[:, 1],
                                         Springsteel.ref_rho_v(ref)[:, 1];
                                         Vmax = VMAX, RMW = RMW,
                                         alpha = RANKINE_ALPHA, v_top = V_TOP,
                                         fcor = F_COR)
    println("Balanced vortex: Vmax = $(VMAX) m/s at $(RMW / 1000.0) km, " *
            "gradient-wind residual = $(round(flds.residual; sigdigits=3)) " *
            "(kink-limited at the RMW), supersaturated points = $(flds.n_supersat)")
    pmin = minimum(flds.p[1, :])
    println("Central surface pressure deficit: " *
            "$(round((pmin - flds.p[1, end]) / 100.0, digits=2)) hPa")

    for m in models
        p = createGrid(m.grid_params)
        gpts = Scythe.getGridpoints(p)
        p.physical .= 0.0
        Scythe.balanced_vortex_mc!(p, gpts, ref, flds, r_axis; zcol = zcol)
        Scythe.write_ics_csv(m.initial_conditions, p, gpts)
    end
    return models, topo
end
