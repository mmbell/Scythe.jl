# ── TC nest construction and balanced-vortex initialization ─────────────────
# Include AFTER tc_params.jl (and after `@everywhere using Scythe, Springsteel`
# in distributed drivers). Provides:
#   make_base(integration_time; output_formats)  -> ModelParameters template
#   make_nest(base)                              -> NestedModelParameters
#   init_tc!(nest)                               -> writes shared reference +
#                                                   per-patch balanced ICs

const TC_VARS = Scythe.MC_VARS_CYL

# WHY THE VERTICAL BCs ARE **SecondDerivativeBC**, NOT NEUMANN (2026-07-21).
# This is the dominant term in the drain documented in HANDOFF_2026-07-20.md.
#
# `NeumannBC()` maps to the cubic-B-spline R1T1 condition (Springsteel
# factory.jl `_bc_to_spline_dict`), a HARD CONSTRAINT ON THE BASIS: the fitted
# spline is forced to have ZERO DERIVATIVE at z = 0 and z = Z_TOP. A tropical
# cyclone violates that at the surface in the two places it matters most:
#
#   hydrostatic balance   dp'/dz = -g rho_t' /= 0   (that IS the surface deficit)
#   RE87 eq. (37)         dv/dz  = -V(r)/z_s  /= 0  (the vortex decays with height
#                                                    starting AT the ground)
#
# so p', rho_t' and v were being projected into a space that cannot hold the
# balanced state near the ground. The damage happens on the FIRST load, inside
# load_initial_conditions!'s spectralTransform!/gridTransform!, before a single
# timestep is taken -- so no initialization scheme, however accurate, can fix it.
#
# MEASURED (model_tests/tc_discrete_balance.jl section 5): max discrete
# hydrostatic residual -(dp'/dz + g rho_t')/rho_t over each patch [m/s^2],
# radial BCs held natural so only BCB/BCT varies --
#
#                            nest 1     nest 2     nest 3
#     NeumannBC              0.1035     0.04825    0.008255
#     SecondDerivativeBC     0.004559   0.002096   0.0003509     <- 23x better
#     NaturalBC              0.000684   0.000512   0.000157      <- but UNSTABLE
#
# The gradient-wind residual is untouched by any of these (~1.9e-4 in nest 1),
# as it must be: the vertical BC does not enter the radial derivative.
#
# WHY NOT NaturalBC, which is what the physics wants. It blows the run up:
# `tc_balance_holdtest.jl 0.1 nophysics` dies on a non-finite spectral
# coefficient at t = 41 s with p unconstrained at the ground, 166 s with rho_t,
# and 21 s with either unconstrained at the lid. E_t, rho_d, u and v are each
# individually fine -- it is specifically the VERTICAL ACOUSTIC PAIR (p, rho_t).
# The Galerkin acoustic solve (`_assemble_spline_matrix`, semiimplicit.jl) takes
# Neumann as its NATURAL weak-form condition and branches only on Dirichlet, so
# leaving p'/rho_t' completely free at a wall desynchronizes the implicit and
# explicit legs of the AI2* split there. SecondDerivativeBC keeps one constraint
# on the basis while leaving dp'/dz FREE -- exactly the degree of freedom
# hydrostatic balance needs -- and is stable.
#
# Nothing else consumes these: the vertical acoustic Helmholtz is assembled on
# the **w** column (Dirichlet, kept; calc_Helmholtz_semiimplicit_matrix takes
# Dirichlet by default and is called without BC arguments), and the
# vertical-diffusion factorizations do read BCB/BCT but are built only when some
# Kvdiff* > 0, which this config never sets.
#
# The RADIAL conditions are left alone and are correct as they stand: Neumann at
# the axis (C = v^2/r + f v -> 0 there, so dp'/dr -> 0) and at the outer wall
# (v == 0 beyond r_0 = 800 km), with u Dirichlet at both and v Dirichlet on the
# axis. Nest junctions never see these -- build_nest gives them FixedBC/NaturalBC.
function tc_boundary_conditions()
    scalar_bc = Dict(v => NeumannBC() for v in TC_VARS)
    axis_bc = merge(scalar_bc, Dict("u" => DirichletBC(), "v" => DirichletBC()))
    wall_bc = merge(scalar_bc, Dict("u" => DirichletBC()))
    # ── VERTICAL WALLS ── see tc/SI_WALL_BC_CEILING.md for the full measurements.
    #
    # SecondDerivativeBC on every scalar. This is the 2026-07-21 configuration and
    # it remains the best MEASURED one, but the timestep it needs is NOT 1.0 s:
    # the resting-column probe (model_tests/tc_lid_drift_probe.jl) puts the wall
    # ceiling at ts ~ 0.75 s with d2 walls versus ~2.0 s with the acoustic set on
    # R1T1, and NEST_TS = 1.0 was above it. That instability -- not physics, not
    # the initialization -- was ~77% of the spurious cooling aloft, ALL of the lid
    # mass gain, and most of the adjustment transient. Hence NEST_TS = 0.5.
    #
    # WHY NOT THE R1T1X WALL CONDITION, which is built and tested. The exact
    # compatibility condition at a rigid wall is dp'/dz = -g rho_t' (w is
    # Dirichlet, so w == 0 for all time and the w equation collapses to an
    # identity). CubicBSpline.R1T1X carries exactly that in an affine ahat while
    # keeping R1T1's subspace. It is nonetheless WORSE THAN d2 ON BOTH COUNTS.
    #
    # t = 0 residuals under the model's own operators, nest 1
    # (model_tests/tc_discrete_balance.jl section 1):
    #
    #                    gradient-wind      hydrostatic
    #     all d2          4.670e-05          5.091e-03
    #     R1T1X config    3.062e-03          9.932e-03
    #
    # 12 h nophysics hold test, nest 1:
    #
    #     d2   @ ts 0.5     15.8% of the deficit filled,  4.1% of the wind lost
    #     d2   @ ts 1.0     34%                          20%
    #     R1T1X@ ts 1.0     95%                          46%
    #
    # The leading suspect is NOT p itself but
    # the four variables this configuration moved from d2 to homogeneous Neumann
    # to keep the acoustic set consistent (rho_d, rho_t, E_t, Q_ss): Neumann
    # forces d(rho_t')/dz = 0 at the ground, which is the same projection damage
    # the 2026-07-21 handoff identified for p, relocated to the densities. Second
    # suspect is the FROZEN wall derivative (:wall_bc_tau => Inf) pinning dp'/dz
    # to its t = 0 value while the vortex adjusts. Neither has been isolated.
    #
    # The R1T1X machinery is kept and tested (Springsteel test/r1t1x.jl,
    # Scythe mc_wall_bc_active / update_mc_wall_bc!); it is simply not enabled.
    # Re-enable by setting vert["p"] = Springsteel.CubicBSpline.R1T1X and the
    # acoustic set to NeumannBC() -- but ISOLATE the two suspects above first.
    vert = Dict{String,Any}(v => SecondDerivativeBC() for v in TC_VARS)
    # w = 0 at the ground and at the rigid lid is the one genuine vertical BC.
    # rho_r stays NaturalBC at the ground so rain can fall out of the domain.
    bot_bc = merge(vert, Dict{String,Any}("w" => DirichletBC(), "rho_r" => NaturalBC()))
    top_bc = merge(vert, Dict{String,Any}("w" => DirichletBC()))
    return axis_bc, wall_bc, bot_bc, top_bc
end

function make_base(integration_time; output_formats=OUTPUT_FORMATS,
                   output_dir=OUTPUT_DIR,
                   initial_conditions=joinpath(output_dir, "tc_ics.csv"),
                   geometry="RiRk",
                   output_interval=OUTPUT_INTERVAL,
                   extra_options=Dict{Symbol,Any}(),
                   # Overrides merged over physical_params, so a diagnostic run
                   # can switch an individual closure off (e.g. :Khdiff_heat =>
                   # 0.0) without editing tc_params.jl -- which matters because
                   # a restart re-reads tc_params.jl, so editing it while a run
                   # is in flight silently changes what a later restart does.
                   extra_physical=Dict{Symbol,Any}())
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
        physical_params = merge(Dict(:Khdiff => 0.0, :Kvdiff => 0.0,
                               :Khdiff_heat => KH_HEAT, :Pr_t => PR_T,
                               :Khdiff_water => KH_WATER, :Sc_t => SC_T,
                               :Kvdiff_heat => 0.0,
                               :Kvdiff_water => 0.0,
                               :tau_qss => TAU_QSS, :N_r => 1.0e-3, :N_0 => N_0_MP,
                               :alpha => SPONGE_ALPHA, :z_damp => Z_DAMP,
                               :f => F_COR,
                               :Cd => CD, :l_inf => L_INF,
                               :Ls => LS_SMAG, :K_min => K_MIN,
                               :Ck => CK, :SST => SST_K, :U_min => U_MIN),
                                extra_physical),
        # Acoustic solver: the vertical-only SI with the state-dependent acoustic
        # linearization. options[:exact_si] is NOT used here -- see
        # tc/EXACT_SI_VORTEX_FAILURE.md: on this balanced vortex it drives rho_d
        # negative within ~4 timesteps, while the vertical-only SI runs the same
        # initial conditions cleanly. exact_si is opt-in via tc_run_axisym.jl's
        # --exact-si for further debugging. It would buy no timestep here in any
        # case: the run's own Courant ladder puts the horizontal acoustic mode at
        # Co 0.25/3.0 while the vertical convective ceiling binds at 2.51/2.88.
        # :wall_bc_tau is inert while the walls are SecondDerivativeBC (the
        # R1T1X path is off); kept so re-enabling needs one edit, not two.
        # Inf FREEZES the R1T1X wall derivative at the value
        # load_initial_conditions! computes from the balanced vortex. A frozen
        # (affine) offset is provably — and measurably — as stable as homogeneous
        # R1T1, whereas letting it track the state closes a feedback loop with the
        # acoustic mode that is unstable at ANY nonzero gain: the wall condition
        # sets the net vertical force at the wall to zero, which removes the
        # restoring force that would otherwise oppose a growing boundary mode.
        # Smoothing the source and relaxing over 300 s only slows it (measured:
        # non-finite at ~30 steps unsmoothed, ~900 steps smoothed+relaxed).
        # Physically the wall gradient belongs to the BALANCED vortex and evolves
        # on hours, so freezing it is a good approximation over a spin-up; it does
        # go stale as the storm deepens, which is the open item in
        # tc/SI_WALL_BC_CEILING.md.
        # REFERENCE STATE (2026-07-21, tc/HANDOFF_REFERENCE_STATE.md). Both of these
        # are opt-in and off by default in the library, because they move every
        # pressure-reference baseline; the TC run needs both.
        #
        # :consistent_qss_reference rebuilds Q̄_ss through the model's own retrieval so
        # the reference does not CONDENSE at rest. Without it the resting column is not
        # a discrete fixed point at all: expdot[p] = 2.3 Pa/s with zero perturbation,
        # zero physics and zero diffusion, and p'(top) reaches -4.7 Pa in 30 min. With
        # it the resting column is BIT-EXACT zero for at least 3 h
        # (model_tests/tc_lid_drift_probe.jl d2@0.5+qss). This was 100 % of the resting
        # drift -- the earlier 88/12 attribution to the hydrostatic imbalance was an
        # artifact of comparing two different reference states.
        #
        # :hydrostatic_reference makes dp̄/dz = -g·ρ̄_t hold EXACTLY. It contributes
        # nothing to the RESTING drift (measured: d2@0.5+hydro alone still drifts,
        # d2@0.5+qss alone is already bit-exact) because every reference-derivative
        # term in the tendencies multiplies w. It matters for the PERTURBATION
        # dynamics: the equation set carries dp̄/dz = -g·ρ̄_t as an unstated assumption,
        # so where the stored derivative violated it -- by up to 17 % above the
        # tropopause, from a hydrostatic sweep truncated at 5 iterations mid-oscillation
        # -- the model silently omitted a forcing of that size, and every reference
        # gradient the perturbations advect (p_z = pp_z + p̄_z) was wrong by it.
        options = merge(Dict{Symbol,Any}(:semiimplicit => true,
                                         :wall_bc_tau => Inf,
                                         :state_dependent_si => true,
                                         :exact_reference_state => true,
                                         :consistent_qss_reference => true,
                                         :hydrostatic_reference => true,
                                         :state_deviation => STATE_DEVIATION,
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
    # The .ref file carries VALUES only, so the hydrostatic balance has to survive the
    # round trip through the density: write the CONVERGED (p, rho_d, rho_v) triple and
    # let exact_pressure_reference_state re-integrate dp/dz = -g*rho_t from it. Writing
    # the 5-sweep-truncated triple instead leaves p and rho_t mutually inconsistent by
    # 27 % at the lid, which the reconstruction then rejects.
    hydro = get(nest.base.options, :hydrostatic_reference, false)::Bool
    ref_phys = Springsteel.calculate_pressure_reference_state(SOUNDING, z, column;
                                                             hydrostatic = hydro)
    Scythe.write_exact_ref_mc(nest.base.ref_state_file, z,
                              Springsteel.ref_pressure(ref_phys)[:, 1],
                              Springsteel.ref_rho_d(ref_phys)[:, 1],
                              Springsteel.ref_rho_v(ref_phys)[:, 1],
                              zeros(kDim))
    ref = Springsteel.exact_pressure_reference_state(nest.base.ref_state_file, z, column;
                                                     hydrostatic = hydro)
    if hydro
        rt = Springsteel.ref_rho_t(ref)[:, 1]
        resid = -(Springsteel.ref_pressure(ref)[:, 2] .+ (Scythe.gravity .* rt)) ./ rt
        println("Reference hydrostatic residual: max = " *
                "$(round(maximum(abs, resid); sigdigits=3)) m/s^2")
    end
    # The initial conditions are stored as PERTURBATIONS from Q̄_ss, so the vortex must
    # be differenced against the SAME Q̄_ss createModelTile will add back. Applying the
    # correction here (rather than only inside createModelTile) keeps the far field,
    # where the vortex vanishes and the state must reduce to the reference exactly,
    # at Q_ss' -> 0.
    if get(nest.base.options, :consistent_qss_reference, false)::Bool
        ref = Scythe.consistent_qss_reference(ref, z, column)
    end
    println("Reference: sfc p = " *
            "$(round(Springsteel.ref_pressure(ref)[1, 1] / 100.0, digits=2)) hPa")

    # Balanced vortex on the radial work grid x model mish vertical axis
    r_outer = gp1.geometry == "RLR" ? NEST_BOUNDARIES_RLR[end] : NEST_BOUNDARIES[end]
    r_axis = collect(0.0:DR_WORK:r_outer)
    flds = Scythe.balanced_vortex_fields(r_axis, z,
                                         Springsteel.ref_pressure(ref)[:, 1],
                                         Springsteel.ref_rho_d(ref)[:, 1],
                                         Springsteel.ref_rho_v(ref)[:, 1];
                                         vortex_profile = VORTEX_PROFILE,
                                         v_m = V_M, r_m = R_M, r_0 = R_0,
                                         Vmax = VMAX, RMW = RMW,
                                         alpha = RANKINE_ALPHA, v_top = V_TOP,
                                         z_bt = Z_BAROTROPIC, fcor = F_COR, RH_core = RH_CORE,
                                         r_moist = R_MOIST, z_moist = Z_MOIST,
                                         RH_max = RH_INIT_MAX,
                                         RH_bl = RH_BL, z_bl = Z_BL,
                                         moist_profile = MOIST_PROFILE)
    println("Balanced vortex ($(VORTEX_PROFILE)): max v = " *
            "$(round(maximum(flds.v); digits=2)) m/s, " *
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
