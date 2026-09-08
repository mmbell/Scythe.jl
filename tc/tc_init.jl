# ── TC nest construction and balanced-vortex initialization ─────────────────
# Include AFTER tc_params.jl (and after `@everywhere using Scythe, Springsteel`
# in distributed drivers). Provides:
#   make_base(integration_time; output_formats)  -> ModelParameters template
#   make_nest(base)                              -> NestedModelParameters
#   init_tc!(nest)                               -> writes shared reference +
#                                                   per-patch balanced ICs

# The water transforms decide what slots 8 and 9 are NAMED (rho_r/rho_c untransformed,
# nu_r/nu_c under a transform), and the name is the only thing a consumer sees --
# Springsteel builds the CSV/netCDF header straight from GridParameters.vars. So every
# name-keyed dict here (vars, the four BC dicts, l_q, positivity, spline_filter) has to
# be built from `mc_var_names`, never from a literal: `_resolve_spline_filter` returns
# `nothing` on a miss rather than raising, so a leftover "rho_r" key under the rain
# transform would SILENTLY put slot 8 back on a Neumann fit -- which forces dF/dz = 0 at
# the ground and traps the falling rain at the surface. `Scythe.check_mc_var_names`
# (called from createModelTile) refuses a stale key, which is the backstop, not the plan.
const TC_WATER_OPTS = Dict{Symbol,Any}(:condensate_transform => CONDENSATE_TRANSFORM,
                                       :rain_transform => RAIN_TRANSFORM)

# SCYTHE_TC_BL selects the boundary-layer scheme, "louis" (default) or "mynn"
# (S10a, src/mynn_state.jl). Read HERE, at file-include time rather than inside
# make_base, because the MYNN TKE density is an APPENDED `mc_var_names` slot
# ("rho_e", moist_compressible.jl) and TC_VARS below -- and everything keyed off
# it (grid_params.vars, all four BC dicts in tc_boundary_conditions) -- has to
# be built knowing whether that slot exists before a single ModelParameters is
# constructed. make_base reuses this SAME constant for options[:mynn] rather
# than re-reading the env var, so the two can never disagree.
const TC_BL_CHOICE = get(ENV, "SCYTHE_TC_BL", "louis")
TC_BL_CHOICE in ("louis", "mynn") ||
    error("SCYTHE_TC_BL = \"$(TC_BL_CHOICE)\" is not recognized; use louis or mynn")
const TC_NAME_OPTS = merge(TC_WATER_OPTS,
                           TC_BL_CHOICE == "mynn" ? Dict{Symbol,Any}(:mynn => true) :
                                                    Dict{Symbol,Any}())
const TC_VARS = Scythe.mc_var_names(TC_NAME_OPTS; cyl = true)
const TC_RAIN_VAR = Scythe.rain_var_name(TC_WATER_OPTS)
const TC_CLOUD_VAR = Scythe.condensate_var_name(TC_WATER_OPTS)

# WHY THE VERTICAL BCs ARE **SecondDerivativeBC**, NOT NEUMANN (2026-07-21).
# This is the dominant term in the drain documented in reference/HANDOFF_2026-07-20.md.
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
    # ── VERTICAL WALLS ── see reference/SI_WALL_BC_CEILING.md for the full measurements.
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
    #
    # SCYTHE_TC_WALL_BC selects the vertical wall condition WITHOUT editing this
    # file, the same way SCYTHE_TC_TS_SCALE selects the timestep -- because a
    # restart re-reads tc_params.jl/tc_init.jl, so an edit made while a run is in
    # flight silently changes what a later restart does.
    #
    #   d2      (default) SecondDerivativeBC on every scalar -- the shipped config
    #   natural NaturalBC on every scalar -- what the physics wants, historically
    #           UNSTABLE (non-finite at t = 41 s)
    #   r1t1x   p on CubicBSpline.R1T1X (the exact wall compatibility condition
    #           dp'/dz = -g rho_t' carried in an affine ahat) with the acoustic set
    #           on homogeneous Neumann
    #
    # BOTH non-default options were REJECTED on measurements that are now void.
    # The `natural` blow-up and the R1T1X comparison were taken on a reference
    # that condensed at rest (expdot[p] = 2.3 Pa/s with zero perturbation) and on
    # an initialization out of discrete hydrostatic balance by 5e-3 m/s^2 -- i.e.
    # a wall condition that leaves dp'/dz free was being asked to hold a state that
    # was not in balance to begin with, and was blamed for the resulting growth.
    # Both of those are fixed (reference/HANDOFF_REFERENCE_STATE.md,
    # reference/HANDOFF_INITIALIZATION.md), so the rejections need re-taking rather than
    # inheriting. What has NOT changed is the structural argument in the note above
    # -- the Galerkin acoustic solve takes Neumann as its natural weak-form
    # condition -- so `natural` may well still be unstable for a reason no
    # initialization can fix. That is the question, not the assumption.
    #
    # There is now a measured PAYOFF to weigh it against: on the native init,
    # relaxing d2 -> natural cuts the t = 0 hydrostatic residual in the wall cells
    # by 10.65x (model_tests/tc_balance_floor.jl). It buys only 1.26x on the
    # DOMAIN maximum, though, because re87_v's dv/dz jump at v_top is the next
    # constraint -- round that off too (`z_round`) and the pair gives 5.51x.
    wall_mode = get(ENV, "SCYTHE_TC_WALL_BC", "d2")
    vert = if wall_mode == "natural"
        Dict{String,Any}(v => NaturalBC() for v in TC_VARS)
    elseif wall_mode == "r1t1x"
        # The acoustic set goes to homogeneous Neumann to stay consistent with the
        # affine p condition; everything else keeps d2. NOTE this is the very
        # grouping the note above names as the leading suspect for R1T1X being
        # worse than d2 -- Neumann forces d(rho_t')/dz = 0 at the ground -- so a
        # re-test should also try leaving the densities on d2.
        d = Dict{String,Any}(v => SecondDerivativeBC() for v in TC_VARS)
        for v in ("rho_d", "rho_t", "E_t", "Q_ss")
            d[v] = NeumannBC()
        end
        d["p"] = Springsteel.CubicBSpline.R1T1X
        d
    elseif wall_mode == "d2"
        Dict{String,Any}(v => SecondDerivativeBC() for v in TC_VARS)
    else
        error("SCYTHE_TC_WALL_BC must be one of d2 | natural | r1t1x, got $(wall_mode)")
    end
    wall_mode == "d2" || @info "TC vertical wall condition: $(wall_mode) (non-default)"
    # w = 0 at the ground and at the rigid lid is the one genuine vertical BC.
    # Rain stays NaturalBC at the ground so it can fall out of the domain. Keyed by
    # TC_RAIN_VAR, not the literal "rho_r": under the rain transform the slot is named
    # nu_r, and a stale key here is IGNORED rather than raising -- putting slot 8 back on
    # a Neumann fit, which pins dF/dz = 0 at z = 0 and traps the rain at the surface.
    bot_bc = merge(vert, Dict{String,Any}("w" => DirichletBC(), TC_RAIN_VAR => NaturalBC()))
    top_bc = merge(vert, Dict{String,Any}("w" => DirichletBC()))
    return axis_bc, wall_bc, bot_bc, top_bc
end

"""
    tc_output_dir(path; allow_existing = false) -> path

Refuse to start a run that would OVERWRITE a preserved one, and say how to fix it.

Model output is experimental data. The convention has always been to give every run its own
tree and `mv` old ones aside, and every script here carries a tag knob for it -- but the
convention was unenforced, and on 2026-07-31 a plain
`julia model_tests/tc_balance_holdtest.jl 12 nophysics` silently overwrote 13 files of the
preserved 2026-07-21 `tc_holdtest_nophysics` tree (the three `0.0.nc`, the logs, the ICs and
`tc_exact.ref`) before it was noticed, and `tc_drain_onset.jl 10` replaced both
`tc_drainonset_cond{on,off}_ts0.5` trees outright. Nothing reported either one: the writer
just opens `<t>.nc` for writing.

So this is now checked rather than remembered, the same way `check_mc_var_names` checks a
convention that used to be remembered. It looks for real OUTPUT (`.nc`, `.jld2`, `*_physical.csv`,
`*_gridded.csv`) anywhere under `path` and raises naming the knob to use. It does NOT count the
initial conditions, the reference file or the logs, which every run legitimately rewrites.

`allow_existing = true` is for a restart, which writes into its own tree by design.
`SCYTHE_TC_FORCE_OUTDIR=1` overrides it for a deliberate re-run; that is a decision, so it is
made once, out loud, on the command line.
"""
function tc_output_dir(path::AbstractString; allow_existing::Bool = false)
    (allow_existing || get(ENV, "SCYTHE_TC_FORCE_OUTDIR", "0") == "1") && return path
    isdir(path) || return path
    found = String[]
    for (root, _, files) in walkdir(path), f in files
        (endswith(f, ".nc") || endswith(f, ".jld2") ||
         endswith(f, "_physical.csv") || endswith(f, "_gridded.csv")) || continue
        push!(found, joinpath(relpath(root, path), f))
        length(found) >= 4 && break
    end
    isempty(found) && return path
    error("""
        refusing to write into $(path): it already holds model output
        ($(join(first(found, 3), ", "))$(length(found) > 3 ? ", ..." : "")).

        Model output is experimental data -- a run that took hours is not this script's to
        overwrite, and the writer would clobber it file by file without a word. Either

          * give this run its own tree (SCYTHE_TC_TAG / SCYTHE_TC_OUTDIR, per the script), or
          * mv $(basename(path)) aside under a descriptive name first, or
          * set SCYTHE_TC_FORCE_OUTDIR=1 if you really do mean to replace it.

        Never rm -rf an output tree.""")
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

    # ── Radiation (S6) ────────────────────────────────────────────────────────
    # SCYTHE_TC_RAD selects the arm, in the style of the benchmark's SCYTHE_O01_RAD
    # (benchmarks/o01_rainfall.jl) and the SCYTHE_TC_CTRANS env-switch pattern above:
    #   unset | "0"   no radiation keys at all -- the run is BIT-IDENTICAL to a
    #                 pre-radiation TC run (mc_radiation_state returns EMPTY_RADIATION
    #                 whenever options[:radiation] is absent).
    #   "lw"          RRTMGP clear-sky longwave only, no sun (:clearsky, :solar = :none).
    #   "allsky"      the cloud-optics-coupled all-sky method (warm-rain liquid cloud
    #                 only here -- ice is out of scope for S6/S7), still no sun.
    #   "diurnal"     allsky + the full diurnal solar cycle at RAD_LATITUDE/
    #                 RAD_START_DOY/RAD_START_HOUR (tc_params.jl) -- the S6 production
    #                 arm. `physical_params[:SST]` (already set below, SST_K) becomes
    #                 the radiative surface temperature automatically through
    #                 `radiation_surface_temperature`'s `:T_sfc` > `:SST` > extrapolated
    #                 -air-temperature precedence (src/radiation.jl) -- no `:T_sfc` is
    #                 set here, so `:SST` wins.
    # SCYTHE_TC_RAD_FORCING ("full" default | "anomaly"), SCYTHE_TC_RAD_ZMAX (override
    # RAD_ZMAX) and SCYTHE_TC_RAD_INTERVAL (override RAD_INTERVAL, seconds) tune the arm
    # without editing tc_params.jl -- same reasoning as SCYTHE_TC_TS_SCALE: a restart
    # re-reads tc_params.jl/tc_init.jl, so an edit made mid-run silently changes what a
    # later restart does.
    #
    # Every nest patch is built from this SAME `base.options`/`base.physical_params`
    # (make_nest/build_nest only vary ts and grid_params per patch), so every patch gets
    # the identical radiation keys -- including the SECONDS-valued
    # `:radiation_interval`, which `validate_radiation_options` converts to
    # `interval_steps` from EACH patch's own `model.ts` (`mc_radiation_state` is called
    # once per tile, inside `createModelTile`, with that tile's own `model`), so patches
    # at different timesteps still share one 300 s cadence in wall-clock/model-time terms
    # even though the step count between calls would differ if their ts differed.
    rad_arm = get(ENV, "SCYTHE_TC_RAD", "0")
    rad_options = Dict{Symbol,Any}()
    rad_physical = Dict{Symbol,Any}()
    if rad_arm != "0" && rad_arm != ""
        rad_options[:radiation] = :rrtmgp
        if rad_arm == "lw"
            rad_options[:radiation_method] = :clearsky
            rad_options[:solar] = :none
        elseif rad_arm == "allsky"
            rad_options[:radiation_method] = :allsky
            rad_options[:solar] = :none
        elseif rad_arm == "diurnal"
            rad_options[:radiation_method] = :allsky
            rad_options[:solar] = :diurnal
            rad_physical[:latitude] = RAD_LATITUDE
            rad_physical[:start_doy] = RAD_START_DOY
            rad_physical[:start_hour] = RAD_START_HOUR
        else
            error("SCYTHE_TC_RAD = \"$rad_arm\" is not an arm; use lw, allsky, " *
                  "diurnal, or 0/unset for no radiation at all")
        end
        rad_forcing = Symbol(get(ENV, "SCYTHE_TC_RAD_FORCING", "full"))
        rad_interval = parse(Float64, get(ENV, "SCYTHE_TC_RAD_INTERVAL", string(RAD_INTERVAL)))
        rad_zmax = parse(Float64, get(ENV, "SCYTHE_TC_RAD_ZMAX", string(RAD_ZMAX)))
        rad_options[:radiation_forcing] = rad_forcing
        rad_options[:radiation_interval] = rad_interval
        rad_options[:radiation_z_max] = rad_zmax
        rad_physical[:sfc_albedo] = RAD_ALBEDO
        rad_physical[:sfc_emissivity] = RAD_EMISSIVITY
        println("TC radiation: arm=$rad_arm method=$(rad_options[:radiation_method]) " *
                "solar=$(rad_options[:solar]) forcing=$rad_forcing " *
                "interval=$rad_interval s z_max=$rad_zmax m; the radiative surface " *
                "temperature is physical_params[:SST] = $(SST_K) K (no :T_sfc set)")
    end

    # ── Boundary layer (S10a) ────────────────────────────────────────────────
    # SCYTHE_TC_BL selects the boundary-layer scheme, in the style of SCYTHE_TC_RAD.
    # The choice itself (TC_BL_CHOICE, already validated) is resolved above, at
    # file-include time -- see the comment there for why: the MYNN TKE-density
    # slot "rho_e" is an APPENDED mc_var_names slot, and TC_VARS (built at
    # include time from TC_NAME_OPTS) has to already know about it.
    #   "louis" (default, unset)   the shipped Louis bulk-formula scheme
    #                              (:louis_bl => true) -- BIT-IDENTICAL to a
    #                              pre-knob TC run, since this is today's
    #                              configuration.
    #   "mynn"                     the MYNN-EDMF port (src/mynn_state.jl):
    #                              :louis_bl is dropped and :mynn => true is set
    #                              instead. :surface_fluxes stays on either way --
    #                              MYNN consumes the same surface enthalpy/
    #                              moisture/momentum fluxes, it does not replace
    #                              that scheme.
    # SCYTHE_TC_MYNN_INTERVAL (s, default 20.0) -> options[:mynn_interval] and
    # SCYTHE_TC_MYNN_EDMF (0/1, default 0) -> options[:mynn_edmf] tune the mynn arm
    # without editing this file -- same reasoning as SCYTHE_TC_RAD_INTERVAL: a
    # restart re-reads tc_params.jl/tc_init.jl, so an edit made mid-run silently
    # changes what a later restart does.
    #
    # `Scythe.validate_mynn_options` (src/mynn_state.jl) is the actual gate: it
    # checks the equation set is a pressure-reference moist_compressible set (true
    # here), that the vertical basis is the cubic B-spline (true for both RiRk and
    # RLR geometries this driver builds), :mynn_interval >= this patch's own ts,
    # and :mynn_edmf in (0, 1) -- refusing 1 outright until S7 ports the mass-flux
    # plumes. It runs once per tile at construction, so a bad knob dies at setup.
    # ── Surface layer (S1b, shared by BOTH BL schemes) ───────────────────────
    # SCYTHE_TC_SFC selects options[:sfc_z0] (Scythe.SFC_Z0_MODES,
    # src/mc_surface_layer.jl): komori | gfdl_v7 | charnock. Resolved (and validated)
    # HERE, before the louis/mynn branch, so both println lines below can report it --
    # the surface layer is not part of either scheme's own options, it sits underneath
    # both (surface_layer_params is called the same way from mc_louis_bl! and the MYNN
    # closure). komori is the default and is left OUT of bl_options entirely when
    # selected, so the louis branch's options dict stays byte-identical to a pre-S1b TC
    # run; only a non-default value adds the key.
    sfc_z0_str = get(ENV, "SCYTHE_TC_SFC", "komori")
    sfc_z0_str in string.(Scythe.SFC_Z0_MODES) ||
        error("SCYTHE_TC_SFC = \"$(sfc_z0_str)\" is not recognized; use " *
              join(string.(Scythe.SFC_Z0_MODES), ", "))
    # SCYTHE_TC_SFC_STAB (0 default | 1) selects options[:sfc_stability] (Monin-Obukhov
    # stability functions over the neutral coefficients above); same reasoning -- only
    # "1" adds the key, so the default run's options dict is unchanged.
    sfc_stab_str = get(ENV, "SCYTHE_TC_SFC_STAB", "0")
    sfc_stab_str in ("0", "1") ||
        error("SCYTHE_TC_SFC_STAB = \"$(sfc_stab_str)\" is not recognized; use 0 or 1")
    sfc_suffix = "  sfc_z0=$sfc_z0_str sfc_stability=$(sfc_stab_str == "1")"

    bl_options = Dict{Symbol,Any}()
    if TC_BL_CHOICE == "louis"
        bl_options[:louis_bl] = true
        println("TC boundary layer: louis (default)" * sfc_suffix)
    else # "mynn" -- TC_BL_CHOICE is validated to be one of these two at include time
        mynn_interval = parse(Float64, get(ENV, "SCYTHE_TC_MYNN_INTERVAL", "20.0"))
        mynn_edmf = parse(Int, get(ENV, "SCYTHE_TC_MYNN_EDMF", "0"))
        bl_options[:mynn] = true
        bl_options[:mynn_interval] = mynn_interval
        bl_options[:mynn_edmf] = mynn_edmf
        println("TC boundary layer: mynn interval=$mynn_interval s edmf=$mynn_edmf" *
                sfc_suffix)
    end
    # Applied OUTSIDE the branch above: the surface layer is shared, not a per-scheme
    # option, and this way a future scheme added to the if/else inherits it for free.
    sfc_z0_str == "komori" || (bl_options[:sfc_z0] = Symbol(sfc_z0_str))
    sfc_stab_str == "1" && (bl_options[:sfc_stability] = true)

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
                                rad_physical, extra_physical),
        # Acoustic solver: the vertical-only SI with the state-dependent acoustic
        # linearization. options[:exact_si] is NOT used here -- see
        # reference/EXACT_SI_VORTEX_FAILURE.md: on this balanced vortex it drives rho_d
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
        # reference/SI_WALL_BC_CEILING.md.
        # REFERENCE STATE (2026-07-21, reference/HANDOFF_REFERENCE_STATE.md). Both of these
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
        #
        # WATER CONTROL VARIABLES: TC_WATER_OPTS carries :condensate_transform and
        # :rain_transform (see tc_params.jl). They must go in here, not just into the
        # names above, or the model would read a control variable as a density -- and
        # `check_mc_var_names` would catch it, because the names would then disagree.
        # Declared BEFORE extra_options so a diagnostic run can still override them.
        options = merge(TC_WATER_OPTS,
                        Dict{Symbol,Any}(:semiimplicit => true,
                                         :wall_bc_tau => Inf,
                                         :state_dependent_si => true,
                                         :exact_reference_state => true,
                                         :consistent_qss_reference => true,
                                         :hydrostatic_reference => true,
                                         :state_deviation => STATE_DEVIATION,
                                         :precipitation => true,
                                         :vertical_mixing => false,
                                         :surface_fluxes => true,
                                         :output_formats => output_formats),
                        bl_options, rad_options, extra_options))
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

    # ── NATIVE-GRID BALANCE (default) ──────────────────────────────────────────
    # Solve the balance on the patches' own mish under the patches' own fit, so the
    # DISCRETE residuals -- the ones the equation set actually computes -- are what
    # gets minimized. The legacy path below balances on a foreign 500 m work grid
    # with FD/Heun/trapezoid operators and transfers by linear interpolation; it
    # reports 3.6e-4 while the model sees 5.1e-3. Set options[:native_vortex_init]
    # = false to fall back to it for an A/B.
    if get(nest.base.options, :native_vortex_init, true)::Bool
        patches = [createGrid(m.grid_params) for m in models]
        Scythe.balanced_vortex_native!(patches, topo, ref;
                                       zcol = zcol, fcor = F_COR,
                                       vortex_profile = VORTEX_PROFILE,
                                       v_m = V_M, r_m = R_M, r_0 = R_0,
                                       Vmax = VMAX, RMW = RMW,
                                       alpha = RANKINE_ALPHA, v_top = V_TOP,
                                       z_bt = Z_BAROTROPIC, RH_core = RH_CORE,
                                       r_moist = R_MOIST, z_moist = Z_MOIST,
                                       RH_max = RH_INIT_MAX,
                                       RH_bl = RH_BL, z_bl = Z_BL,
                                       moist_profile = MOIST_PROFILE,
                                       # The initial state is cloud- and rain-free, and
                                       # condensate_slot(0) = rain_slot(0) = 0.0 EXACTLY
                                       # in both conventions, so these are belt and
                                       # braces today. Thread them anyway: the day a
                                       # cloudy perturbation is added, an untreaded
                                       # initializer would write a density into a
                                       # control-variable slot and nothing would say so.
                                       condensate_transform = CONDENSATE_TRANSFORM,
                                       condensate_mu = 1.0e-7,
                                       rain_transform = RAIN_TRANSFORM,
                                       rain_mu = 1.0e-7)
        for (p, m) in zip(patches, models)
            gpts = Scythe.getGridpoints(p)
            kDim1 = m.grid_params.kDim
            if m === models[1]
                pv = m.grid_params.vars["p"]
                vv = m.grid_params.vars["v"]
                println("Native vortex: max v = " *
                        "$(round(maximum(p.physical[:, vv, 1]); digits=2)) m/s, " *
                        "central surface p deficit = " *
                        "$(round(p.physical[1, pv, 1] / 100.0, digits=2)) hPa " *
                        "(axis mish point, z = $(round(gpts[1, zcol]; digits=1)) m)")
            end
            Scythe.write_ics_csv(m.initial_conditions, p, gpts)
        end
        return models, topo
    end

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
        Scythe.balanced_vortex_mc!(p, gpts, ref, flds, r_axis; zcol = zcol,
                                   condensate_transform = CONDENSATE_TRANSFORM,
                                   condensate_mu = 1.0e-7,
                                   rain_transform = RAIN_TRANSFORM, rain_mu = 1.0e-7)
        Scythe.write_ics_csv(m.initial_conditions, p, gpts)
    end
    return models, topo
end
