using Test
using Scythe
using Springsteel
using SparseArrays

# Driver-level tests for the MYNN-EDMF PLUMBING (plan stage S4): the appended prognostic
# TKE-density slot `rho_e`, the `ModelTile.mynn` field and its setup (`mc_mynn_state`), the
# option table (`validate_mynn_options`) and the driver's refusals.
#
# The closure itself is NOT exercised here -- it is not coupled yet. What is asserted is
# exactly the two properties this stage has to deliver:
#
#   1. WITH the option, `rho_e` is transported like any other total of the set and its
#      sources are zero. The gate is a comparison against a CONTROL slot rather than an
#      analytic advection: seeded with the same blob under the same boundary conditions,
#      the rain mass (with condensation and precipitation off, so its own sources vanish
#      identically) satisfies exactly the equation `rho_e` is supposed to satisfy, so the
#      two tendencies must agree to the last bit. That tests the fitted derivatives, the
#      geometry dispatch and the divergence term all at once, without this file
#      reimplementing any of them.
#
#   2. WITHOUT the option, NOTHING changes. Two tiles are built on the same state, one
#      with `:mynn` and one without, and every shared slot's tendency is compared with
#      `===` -- not `≈`. `x + 0.0` is not the identity for `x = -0.0`, which is why every
#      fold this stage adds is inside an `if mynn_on`.
#
# Follows test_radiation_driver.jl (the fixture and the two-tile difference design) and
# test_louis_bl.jl (the per-column `advance_column` loop).

@testset "MYNN-EDMF driver plumbing (S4)" begin

    import Springsteel.Thermodynamics: Rd, Rv, Cpd, gravity

    """Dry, neutrally stable (theta = 300 K) analytic adiabat -- the same base
    test_louis_bl.jl and test_radiation_driver.jl use."""
    function dry_adiabatic_column(z; theta0 = 300.0)
        n = length(z)
        exner = @. 1.0 - (gravity * z) / (Cpd * theta0)
        Tk = theta0 .* exner
        p_Pa = @. 100000.0 * exner^(Cpd / Rd)
        rho_d = p_Pa ./ (Rd .* Tk)
        return (; z, Tk, p_Pa, rho_d, rho_v = zeros(n), rho_c = zeros(n))
    end

    """A small `moist_compressible_XZ` RiRk tile.

    `mynn = true` appends the `rho_e` slot. Its boundary conditions are deliberately made
    IDENTICAL to the rain slot's (Natural at both walls) so the transport gate below can
    compare the two tendencies bit for bit -- a different vertical condition would change
    the fitted ∂z of the very blob being compared, and the test would be measuring the BCs
    rather than the transport.

    `:condensation => false` and `:precipitation => false` are what make the rain slot a
    clean control: with both off `Qdot_r`, `AUTO_COLL` and `Fr_z` are exact zero columns,
    the rain transform is absent so `Jr` is an exact 1.0, and slot 8 carries precisely
    `-u·∇ρ_r - ρ_r ∇·u`, which is the equation `rho_e` is given at this stage."""
    function make_mynn_mtile(tmpdir; mynn = false, stale_vars = false,
                             extra_opts = Dict{Symbol,Any}(),
                             extra_params = Dict{Symbol,Float64}())
        opts_names = merge(Dict{Symbol,Any}(), extra_opts)
        mynn && (opts_names[:mynn] = true)
        # `stale_vars = true` builds every name-keyed dict from the list the options WITHOUT
        # `:mynn` produce, i.e. a `vars` that has no "rho_e" while the options declare one.
        var_opts = stale_vars ?
            Dict{Symbol,Any}(k => v for (k, v) in opts_names if k !== :mynn) : opts_names
        varlist = Scythe.mc_var_names(var_opts; cyl = false)
        rain_name = Scythe.rain_var_name(opts_names)
        vars = Dict(v => i for (i, v) in enumerate(varlist))
        scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
        side_bc = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))
        topbot_bc = merge(scalar_bc,
                          Dict("w" => DirichletBC(), rain_name => NaturalBC()))
        # Same wall treatment as the rain: see the docstring.
        haskey(vars, "rho_e") && (topbot_bc["rho_e"] = NaturalBC())
        gp = GridParameters(geometry = "RiRk",
            iMin = 0.0, iMax = 12.0e3, num_cells_i = 6,
            kMin = 0.0, kMax = 2000.0, num_cells_k = 8,
            BCL = side_bc, BCR = side_bc, BCB = topbot_bc, BCT = topbot_bc, vars = vars)
        ref_file = joinpath(tmpdir, "mynn_pressure.ref")
        model = ModelParameters(
            ts = 0.1, integration_time = 1.0, output_interval = 1.0,
            equation_set = "moist_compressible_XZ",
            ref_state_file = ref_file, grid_params = gp,
            physical_params = merge(
                Dict(:Khdiff => 0.0, :Kvdiff => 0.0, :Khdiff_heat => 0.0,
                     :Kvdiff_heat => 0.0, :Kvdiff_water => 0.0,
                     :tau_qss => 10.0, :alpha => 0.0, :z_damp => 20.0e3),
                extra_params),
            options = merge(
                Dict{Symbol,Any}(:semiimplicit => true,
                                 :exact_reference_state => true,
                                 :precipitation => false,
                                 :condensation => false,
                                 :mynn_trace => false),
                opts_names))
        gp = model.grid_params
        patch = createGrid(gp)
        gridpoints = Scythe.getGridpoints(patch)
        kDim = gp.kDim
        z = gridpoints[1:kDim, end]
        col = dry_adiabatic_column(z)
        Scythe.write_exact_ref_mc(ref_file, z, col.p_Pa, col.rho_d, col.rho_v, col.rho_c)
        patch.physical .= 0.0
        spectralTransform!(patch)
        gridTransform!(patch)
        hrm = sparse(Int64[], Int64[], Float64[],
                     size(patch.spectral, 1), size(patch.spectral, 2))
        mtile = createModelTile(patch, patch, model, hrm)
        return mtile, patch, model, gp, gridpoints
    end

    """Advance every column of `mtile` once from the state already in `patch`."""
    function advance_all!(mtile, patch, gp)
        spectralTransform!(patch)
        gridTransform!(patch)
        ncols = div(size(patch.physical, 1), gp.kDim)
        for c in 1:ncols
            Scythe.advance_column(mtile, c, 1)
        end
        return mtile.expdot_n
    end

    # ──────────────────────────────────────────────
    # 1. The variable list
    # ──────────────────────────────────────────────
    @testset "mc_var_names: absent option changes nothing, present appends rho_e" begin
        # Every existing option combination, with `:mynn` absent, must return exactly what
        # it returned before this stage. The combinations below are the ones the appended
        # block can produce.
        combos = [
            Dict{Symbol,Any}(),
            Dict{Symbol,Any}(:rain_moments => 2),
            Dict{Symbol,Any}(:condensate_transform => :bhyp, :rain_transform => :bhyp),
            Dict{Symbol,Any}(:rain_moments => 2, :rain_number_transform => :bhyp),
            Dict{Symbol,Any}(:rain_moments => 2, :ice_microphysics => :ishmael),
            Dict{Symbol,Any}(:rain_moments => 2, :ice_microphysics => :ishmael,
                             :ice_transform => :bhyp, :condensate_transform => :bhyp,
                             :rain_transform => :bhyp, :rain_number_transform => :bhyp),
        ]
        for opts in combos, cyl in (false, true)
            base = Scythe.mc_var_names(opts; cyl = cyl)
            @test !("rho_e" in base)
            # `:mynn => false` is the same thing as an absent key.
            off = merge(opts, Dict{Symbol,Any}(:mynn => false))
            @test Scythe.mc_var_names(off; cyl = cyl) == base
            on = merge(opts, Dict{Symbol,Any}(:mynn => true))
            names_on = Scythe.mc_var_names(on; cyl = cyl)
            # Appended LAST -- after the ice family, so the twelve ice indices are unmoved.
            @test names_on == vcat(base, "rho_e")
            @test names_on[end] == "rho_e"
            @test names_on[1:length(base)] == base
        end

        # The name carries no transform alias: the TKE is not a control variable.
        @test !haskey(Scythe.MC_NU_ALIAS, "rho_e")
        # ...and it is a TOTAL, so the positivity bound needs no reference offset.
        @test Scythe.positivity_reference_profile("rho_e", nothing) === nothing

        # Index arithmetic, XZ and cylindrical, plain and with the full ice family.
        plain = Dict{Symbol,Any}(:mynn => true)
        @test findfirst(==("rho_e"), Scythe.mc_var_names(plain)) == 11
        @test findfirst(==("rho_e"), Scythe.mc_var_names(plain; cyl = true)) == 12
        full = Dict{Symbol,Any}(:mynn => true, :rain_moments => 2,
                                :ice_microphysics => :ishmael)
        @test findfirst(==("rho_e"), Scythe.mc_var_names(full)) == 24
        @test findfirst(==("rho_e"), Scythe.mc_var_names(full; cyl = true)) == 25
        # The ice block stays contiguous under the append.
        @test Scythe.mc_var_names(full)[12:23] == collect(Scythe.MC_ICE_VARS)
    end

    # ──────────────────────────────────────────────
    # 2. Option validation
    # ──────────────────────────────────────────────
    @testset "validate_mynn_options" begin
        V(o, p = Dict{Symbol,Float64}(); ts = 0.5, kDim = 24) =
            Scythe.validate_mynn_options(o, p, "moist_compressible_XZ", ts, kDim)

        # The OFF path returns the resolved shape and refuses nothing: a run with the
        # closure off must never be able to fail on a MYNN rule.
        for o in (Dict{Symbol,Any}(),
                  Dict{Symbol,Any}(:mynn => false),
                  Dict{Symbol,Any}(:mynn => false, :mynn_nonsense => 1,
                                   :mynn_closure => 2.6))
            cfg = V(o)
            @test cfg.active == false
            @test cfg.interval_steps == 1
        end
        # ...and on a non-mc equation set, too.
        @test Scythe.validate_mynn_options(Dict{Symbol,Any}(), Dict{Symbol,Float64}(),
                                           "LinearAdvection1D", 0.5, 24).active == false

        on = Dict{Symbol,Any}(:mynn => true)
        cfg = V(on)
        @test cfg.active
        @test cfg.closure == 2.5
        @test cfg.edmf == 0
        @test cfg.scale_aware
        @test cfg.init_mode === :taper
        @test cfg.fidelity === :fortran
        @test cfg.water_carry === :flux
        @test cfg.K_max == Inf
        # 20 s default cadence at ts = 0.5 s
        @test cfg.interval_steps == 40
        # Seconds -> steps, so the cadence is nest-invariant.
        @test V(on; ts = 0.25).interval_steps == 80

        # An unknown :mynn_* key is a typo, not a no-op.
        @test_throws ErrorException V(merge(on, Dict{Symbol,Any}(:mynn_intervals => 20.0)))
        @test_throws ErrorException V(merge(on, Dict{Symbol,Any}(:mynn_edfm => 1)))
        # ...but a non-:mynn key is none of this module's business.
        @test V(merge(on, Dict{Symbol,Any}(:louis_bl => false))).active

        @test_throws ErrorException V(Dict{Symbol,Any}(:mynn => 1))         # not a Bool
        @test_throws ErrorException V(merge(on, Dict{Symbol,Any}(:mynn_closure => 2.6)))
        @test_throws ErrorException V(merge(on, Dict{Symbol,Any}(:mynn_edmf => 1)))
        @test_throws ErrorException V(merge(on, Dict{Symbol,Any}(:mynn_edmf => 2)))
        @test_throws ErrorException V(merge(on, Dict{Symbol,Any}(:mynn_init => :spinup)))
        @test_throws ErrorException V(merge(on, Dict{Symbol,Any}(:mynn_fidelity => :fast)))
        @test_throws ErrorException V(merge(on,
            Dict{Symbol,Any}(:mynn_water_carry => :fixed_theta)))
        # A cadence shorter than the timestep is not a cadence.
        @test_throws ErrorException V(merge(on, Dict{Symbol,Any}(:mynn_interval => 0.25)))
        @test_throws ErrorException V(merge(on, Dict{Symbol,Any}(:mynn_interval => 0.0)))
        @test V(merge(on, Dict{Symbol,Any}(:mynn_interval => 0.5))).interval_steps == 1
        # ...and MYNN on a set with no rho_e slot to carry.
        @test_throws ErrorException Scythe.validate_mynn_options(
            on, Dict{Symbol,Float64}(), "LinearAdvection1D", 0.5, 24)
        # K_max must be a cap, not a floor at zero.
        @test_throws ErrorException V(on, Dict{Symbol,Float64}(:mynn_K_max => 0.0))
        @test V(on, Dict{Symbol,Float64}(:mynn_K_max => 1000.0)).K_max == 1000.0
    end

    # ──────────────────────────────────────────────
    # 3. The ModelTile field
    # ──────────────────────────────────────────────
    @testset "ModelTile.mynn: EMPTY_MYNN off, a live state on, concrete both ways" begin
        mktempdir() do tmp
            m_off, _, _, _, _ = make_mynn_mtile(tmp; mynn = false)
            # The SHARED singleton, not a fresh inactive instance: the off path must not
            # allocate a per-tile state, and nothing may ever write into it.
            @test m_off.mynn === Scythe.EMPTY_MYNN
            @test m_off.mynn.active == false
            @test isempty(m_off.mynn.K_m)
            @test m_off.mc_slots.rho_e == 0

            # An explicit `false` is the same thing as an absent key.
            m_f, _, _, _, _ = make_mynn_mtile(tmp;
                extra_opts = Dict{Symbol,Any}(:mynn => false))
            @test m_f.mynn === Scythe.EMPTY_MYNN

            m_on, patch_on, model_on, gp_on, _ = make_mynn_mtile(tmp; mynn = true)
            @test m_on.mynn.active
            @test m_on.mynn !== Scythe.EMPTY_MYNN
            @test m_on.mynn.kDim == gp_on.kDim
            @test m_on.mynn.ncol == div(size(patch_on.physical, 1), gp_on.kDim)
            @test length(m_on.mynn.K_m) == size(patch_on.physical, 1)
            @test length(m_on.mynn.el) == size(patch_on.physical, 1)
            @test length(m_on.mynn.pblh) == m_on.mynn.ncol
            @test all(==(typemin(Int)), m_on.mynn.last_update_step)
            # Nothing is written at this stage: every held field is exactly zero.
            for f in (:el, :sm, :sh, :vt, :vq, :sgm, :cldfra_bl, :qc_bl, :qi_bl,
                      :K_m, :K_h, :s_aw, :s_aw_st, :s_aw_qw, :s_aw_qv, :s_aw_u,
                      :s_aw_v, :s_aw_e)
                @test all(iszero, getfield(m_on.mynn, f))
            end
            @test m_on.mynn.n_clamp_e == 0
            # Per-thread work, sized for the column.
            @test length(m_on.mynn.work) == Threads.maxthreadid()
            @test m_on.mynn.work[1].n == gp_on.kDim
            # The slot is resolved by NAME, once, into the concrete MCSlots.
            @test m_on.mc_slots.rho_e == model_on.grid_params.vars["rho_e"]
            @test m_on.mc_slots.rho_e == 11
            # ...and appending it moved nothing.
            @test m_on.mc_slots.rho_v == m_off.mc_slots.rho_v

            # The whole point of the concrete field: `mtile.mynn.active` must be a load,
            # not a dynamic dispatch, in the driver preamble.
            @test fieldtype(typeof(m_on), :mynn) === Scythe.MYNNState
            @test isconcretetype(fieldtype(typeof(m_on), :mynn))
            for f in fieldnames(typeof(m_on))
                @test isconcretetype(fieldtype(typeof(m_on), f))
            end
        end
    end

    @testset "setup refusals: a vars list without rho_e, and a stale key" begin
        mktempdir() do tmp
            # `:mynn` declares the slot, so a `vars` dict built from a stale name list must
            # fail at tile creation (check_mc_var_names + mc_slots), not at the first
            # column.
            @test_throws ErrorException make_mynn_mtile(tmp; mynn = true,
                                                        stale_vars = true)
            # The well-formed configuration builds.
            m, _, _, _, _ = make_mynn_mtile(tmp; mynn = true)
            @test m.mynn.active
        end
    end

    # ──────────────────────────────────────────────
    # 4. Transport: rho_e is a total like any other
    # ──────────────────────────────────────────────
    """Seed a moving flow and an identical Gaussian blob in the rain and TKE slots."""
    function seed_blob!(patch, vars, kDim, gridpoints; with_rho_e = true)
        u_i = vars["u"]; w_i = vars["w"]
        rr_i = vars["rho_r"]
        re_i = get(vars, "rho_e", 0)
        npts = size(patch.physical, 1)
        for i in 1:npts
            x = gridpoints[i, 1]
            z = gridpoints[i, 2]
            # A sheared, divergent flow, so the -f*div term is exercised too.
            patch.physical[i, u_i, 1] = 4.0 + 1.5 * sin(2pi * x / 12.0e3) * (z / 2000.0)
            patch.physical[i, w_i, 1] = 0.6 * sin(pi * z / 2000.0) * cos(2pi * x / 12.0e3)
            blob = 1.0e-3 * exp(-(((x - 6.0e3) / 2.5e3)^2 + ((z - 900.0) / 400.0)^2))
            patch.physical[i, rr_i, 1] = blob
            (with_rho_e && re_i > 0) && (patch.physical[i, re_i, 1] = blob)
        end
        return nothing
    end

    @testset "rho_e advects exactly like a source-free control total" begin
        mktempdir() do tmp
            mtile, patch, model, gp, gridpoints = make_mynn_mtile(tmp; mynn = true)
            vars = model.grid_params.vars
            re_i = vars["rho_e"]
            seed_blob!(patch, vars, gp.kDim, gridpoints)
            D = advance_all!(mtile, patch, gp)

            # With `:condensation => false` and `:precipitation => false` the rain slot's
            # sources (`Qdot_r`, `AUTO_COLL`, `Fr_z`) are exact zero columns and `Jr` is an
            # exact 1.0, so slot 8 carries `-u·∇ρ_r - ρ_r∇·u`: precisely the equation
            # `rho_e` is given at this stage, on the same blob under the same BCs.
            @test D[:, re_i] == D[:, vars["rho_r"]]
            # Not a vacuous comparison: the blob really is being transported.
            @test maximum(abs, D[:, re_i]) > 1.0e-6

            # The slot really is prognostic -- it appears in the explicit channel and in
            # NEITHER implicit one (no acoustic leg, no vertical-diffusion history).
            @test size(mtile.expdot_n, 2) >= re_i
            @test all(iszero, view(mtile.impdot_n, :, re_i))
            @test all(iszero, view(mtile.diffdot_n, :, re_i))
        end
    end

    @testset "rho_e = 0 changes no other slot, BITWISE" begin
        mktempdir() do tmp
            mktempdir() do tmp2
                m_on, p_on, mod_on, gp_on, gpts = make_mynn_mtile(tmp; mynn = true)
                m_off, p_off, mod_off, gp_off, gpts2 = make_mynn_mtile(tmp2; mynn = false)
                # The SAME state on both, with the TKE slot left at zero.
                seed_blob!(p_on, mod_on.grid_params.vars, gp_on.kDim, gpts;
                           with_rho_e = false)
                seed_blob!(p_off, mod_off.grid_params.vars, gp_off.kDim, gpts2;
                           with_rho_e = false)
                D_on = advance_all!(m_on, p_on, gp_on)
                D_off = advance_all!(m_off, p_off, gp_off)

                # Every shared slot, bit for bit. `===` on the Float64 elements, not `≈`:
                # `x + 0.0` is not the identity for `x = -0.0`, and the whole point of the
                # `if mynn_on` gates is that the off path is untouched.
                names_off = Scythe.mc_var_names(mod_off.options)
                @test length(names_off) == 10
                for nm in names_off
                    i_on = mod_on.grid_params.vars[nm]
                    i_off = mod_off.grid_params.vars[nm]
                    @test i_on == i_off
                    @test all(D_on[j, i_on] === D_off[j, i_off]
                              for j in axes(D_on, 1))
                end
                # ...and the TKE slot's own tendency is identically zero at rho_e = 0
                # (advection of nothing, and `-0 * div`).
                @test all(iszero, D_on[:, mod_on.grid_params.vars["rho_e"]])
            end
        end
    end

    # ──────────────────────────────────────────────
    # 5. Driver refusals
    # ──────────────────────────────────────────────
    @testset "driver refusals: one boundary layer, and no ice legs yet" begin
        mktempdir() do tmp
            # :mynn with :louis_bl -- both are complete closures and their tendencies are
            # additive, so enabling both mixes every column twice.
            m, patch, model, gp, gpts = make_mynn_mtile(tmp; mynn = true,
                extra_opts = Dict{Symbol,Any}(:louis_bl => true),
                extra_params = Dict{Symbol,Float64}(:Cd => 1.0e-3, :l_inf => 80.0))
            spectralTransform!(patch); gridTransform!(patch)
            @test_throws ErrorException Scythe.advance_column(m, 1, 1)

            # :surface_fluxes without ANY boundary layer is still refused...
            m2, p2, _, gp2, _ = make_mynn_mtile(tmp; mynn = false,
                extra_opts = Dict{Symbol,Any}(:surface_fluxes => true),
                extra_params = Dict{Symbol,Float64}(:SST => 301.15))
            spectralTransform!(p2); gridTransform!(p2)
            @test_throws ErrorException Scythe.advance_column(m2, 1, 1)
            # ...but with :mynn it is now accepted (the check was generalized from
            # :louis_bl to louis_bl || mynn). The fluxes are consumed by the closure that
            # lands at S5; nothing here applies them, and nothing here throws.
            m3, p3, _, gp3, gpts3 = make_mynn_mtile(tmp; mynn = true,
                extra_opts = Dict{Symbol,Any}(:surface_fluxes => true),
                extra_params = Dict{Symbol,Float64}(:SST => 301.15))
            spectralTransform!(p3); gridTransform!(p3)
            @test (Scythe.advance_column(m3, 1, 1); true)
        end
    end

    @testset "driver refusal: :mynn with ISHMAEL ice" begin
        mktempdir() do tmp
            ice_opts = Dict{Symbol,Any}(:rain_moments => 2,
                                        :ice_microphysics => :ishmael)
            m, patch, model, gp, gpts = make_mynn_mtile(tmp; mynn = true,
                extra_opts = ice_opts,
                extra_params = Dict{Symbol,Float64}(:N_r => 1.0e-3))
            spectralTransform!(patch); gridTransform!(patch)
            @test_throws ErrorException Scythe.advance_column(m, 1, 1)
        end
    end
end
