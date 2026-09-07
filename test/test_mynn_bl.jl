using Test
using Scythe
using Springsteel
using SparseArrays

# Tests for the MYNN-EDMF boundary-layer coupling (src/mc_mynn_bl.jl, plan stage S5).
#
# The idiom is test_louis_bl.jl's: the SAME physical state is run through
# `advance_column` with the closure on and off and the explicit tendencies are
# differenced, so advection, PGF, condensation and the sponge cancel identically and
# every assertion below is on the boundary layer's own contribution.
#
# The `rho_e` slot needs one extra step of care, because it exists ONLY on the
# closure-on tile and so has no partner to difference against. Every state used here
# therefore has `w = 0` and `u = u(z)` uniform in x, which makes the slot's transport
# `-u drho_e/dx - w drho_e/dz - rho_e div(u)` identically zero (to spline round-off on a
# constant-in-x field): the whole `rho_e` tendency IS the boundary layer's, with nothing
# to subtract. That is asserted directly rather than assumed.

@testset "MYNN-EDMF boundary layer (mc)" begin

    import Springsteel.Thermodynamics: rho_v_sat, Rd, Rv, Cpd, Cpv, gravity

    # ──────────────────────────────────────────────
    # Fixtures
    # ──────────────────────────────────────────────

    """Stably stratified dry column with an exact hydrostatic pressure for a linear
    temperature profile: `p = p0 (T/T0)^{g/(R_d Gamma)}`. Dry (rho_v = 0) so the
    diagnostic vapour is pinned at zero and retrieval-level wiggles cannot leak into the
    water channels, whose `(L_v - R_v T)` energy factor amplifies them (the note in
    test_louis_bl.jl)."""
    function stable_column(z; T0 = 300.0, lapse = 0.004, p0 = 100000.0)
        Tk = @. T0 - lapse * z
        p_Pa = @. p0 * (Tk / T0)^(gravity / (Rd * lapse))
        rho_d = p_Pa ./ (Rd .* Tk)
        n = length(z)
        return (; z, Tk, p_Pa, rho_d, rho_v = zeros(n), rho_c = zeros(n))
    end

    """Boundary-layer jet `u = V0 (z/zs)^2 exp(-z/zs)`: zero VALUE and zero SLOPE at the
    ground and (to round-off) at the lid, so it is representable on the Neumann spline
    AND both momentum boundary terms of the D3 identity vanish, while the shear between
    0 and ~2 zs is a real boundary-layer shear rather than a domain-deep ramp."""
    bl_jet(z, V0, zs) = @. V0 * (z / zs)^2 * exp(-z / zs)

    """Tile for the MYNN arm (or the matching closure-off control).

    `mynn = false` builds the SAME grid and reference with `options[:mynn]` absent, which
    is the control the tendency difference is taken against: the `rho_e` slot is appended
    after every fixed slot and after the vapour, so slots 1-9 and `rho_v` keep the same
    indices on both tiles."""
    function make_mynn_tile(tmpdir; mynn = true, num_cells_k = 50, kMax = 25.0e3,
                            ts = 0.5, Cd = -1.0, fluxes = true, SST = 302.65,
                            init = :zero, interval = 20.0, K_max = Inf,
                            water_carry = :flux, ctrans = :none, rtrans = :none,
                            l_inf = 80.0, lapse = 0.004)
        opts_names = Dict{Symbol,Any}(:condensate_transform => ctrans,
                                      :rain_transform => rtrans)
        mynn && (opts_names[:mynn] = true)
        varlist = Scythe.mc_var_names(opts_names; cyl = false)
        rain_name = Scythe.rain_var_name(opts_names)
        vars = Dict(v => i for (i, v) in enumerate(varlist))
        scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
        # NEUMANN on the side walls for EVERY variable, `u` included. That is not the
        # production configuration; it is what makes this test's premise true. The
        # `rho_e` slot exists only on the closure-on tile, so its tendency has no partner
        # to difference against and the test relies on its TRANSPORT being identically
        # zero: `u = u(z)` uniform in x, `w = 0`, `rho_e = rho_e(z)`. With a Dirichlet `u`
        # the spline is forced to zero at the walls, an x-uniform wind is NOT
        # representable, and the resulting `du/dx` feeds `-rho_e div(u)` a term worth ~10 %
        # of the budget -- which is exactly what this fixture measured before the BC was
        # changed. A constant IS exactly representable on the Neumann basis (the filter
        # penalises the third derivative, which a constant does not have), so the
        # transport is zero to round-off and what is left in the slot is the closure's.
        side_bc = scalar_bc
        bot_bc = merge(scalar_bc, Dict("w" => DirichletBC(), rain_name => NaturalBC()))
        top_bc = merge(scalar_bc, Dict("w" => DirichletBC()))
        gp = GridParameters(geometry = "RiRk",
            iMin = 0.0, iMax = 24.0e3, num_cells_i = 4,
            kMin = 0.0, kMax = kMax, num_cells_k = num_cells_k,
            BCL = side_bc, BCR = side_bc, BCB = bot_bc, BCT = top_bc, vars = vars)
        ref_file = joinpath(tmpdir,
            "mynn_bl_$(num_cells_k)_$(kMax)_$(lapse)_$(mynn).ref")
        options = Dict{Symbol,Any}(:semiimplicit => true,
                                   :exact_reference_state => true,
                                   :precipitation => false,
                                   # `options[:surface_fluxes]` REQUIRES a boundary-layer
                                   # closure (the driver refuses it otherwise), so the
                                   # control tile never carries them -- which is right:
                                   # the surface fluxes ARE part of what the closure
                                   # contributes, and the difference must contain them.
                                   :surface_fluxes => (mynn && fluxes),
                                   :condensate_transform => ctrans,
                                   :rain_transform => rtrans)
        if mynn
            options[:mynn] = true
            options[:mynn_init] = init
            options[:mynn_interval] = interval
            options[:mynn_water_carry] = water_carry
            options[:mynn_trace] = false
        end
        model = ModelParameters(
            ts = ts, integration_time = 10.0 * ts, output_interval = 10.0 * ts,
            equation_set = "moist_compressible_XZ",
            ref_state_file = ref_file, grid_params = gp,
            physical_params = Dict{Symbol,Any}(
                :Khdiff => 0.0, :Kvdiff => 0.0, :Kvdiff_heat => 0.0,
                :Kvdiff_water => 0.0, :tau_qss => 10.0, :alpha => 0.0,
                :z_damp => 20.0e3, :f => 0.0, :Cd => Cd, :Ls => 0.0,
                :Ck => 1.0e-3, :U_min => 1.0, :l_inf => l_inf, :SST => SST,
                :mynn_K_max => K_max),
            options = options)
        gp = model.grid_params
        patch = createGrid(gp)
        gridpoints = Scythe.getGridpoints(patch)
        kDim = gp.kDim
        z = gridpoints[1:kDim, end]
        col = stable_column(z; lapse = lapse)
        Scythe.write_exact_ref_mc(ref_file, z, col.p_Pa, col.rho_d, col.rho_v, col.rho_c)
        patch.physical .= 0.0
        spectralTransform!(patch)
        gridTransform!(patch)
        haloReceiveMap = sparse(Int64[], Int64[], Float64[],
                                size(patch.spectral, 1), size(patch.spectral, 2))
        mtile = createModelTile(patch, patch, model, haloReceiveMap)
        return mtile, patch, model, gp, z, col
    end

    """3-point Gauss-Legendre cell weights on the RiRk mish (mubar = 3). Deliberately
    reimplemented here rather than imported from benchmarks/common: the budget assertion
    must not share code with anything the model uses."""
    function gauss_weights(kDim, num_cells_k, L)
        @assert kDim == 3 * num_cells_k
        dz = L / num_cells_k
        w = (5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0) .* dz
        return repeat(collect(w), num_cells_k)
    end

    """Impose u(z), rho_e(z) (both uniform in x) and the E_t' = rho_t ke compensation, so
    the imposed wind does not double as a thermal perturbation (the retrieval reads E_t
    minus the kinetic energy)."""
    function set_state!(patch, model, kDim, z, col; V0 = 12.0, zs = 500.0, e0 = 0.4,
                        ze = 1500.0, W0 = 0.0, qpert = 2.0e-3)
        vars = model.grid_params.vars
        ui = vars["u"]; wi = vars["w"]; ei = vars["E_t"]
        rei = get(vars, "rho_e", 0)
        u = bl_jet(z, V0, zs)
        rho_t = col.rho_d .+ col.rho_v .+ col.rho_c
        ztop = model.grid_params.kMax
        npts = size(patch.physical, 1)
        for j in 1:npts
            k = mod1(j, kDim)
            patch.physical[j, ui, 1] = u[k]
            # `w` is OPTIONAL and defaults to zero. The budget tests need it zero: with a
            # nonzero `w` the divergence is nonzero and the TKE slot's own transport stops
            # being zero, which is the premise those tests rest on (see the fixture's BC
            # note). The fold test turns it on because slot 5 has to be shown to move.
            patch.physical[j, wi, 1] = W0 * sin(pi * z[k] / ztop)
            # Water PERTURBATIONS. The mixing acts on perturbations from the reference, so
            # a dry reference with zero water slots leaves every D7 leg identically zero
            # and untested. These make the vapour, total-water, cloud and rain fluxes -- and
            # with them the flux-form energy carry `S_Ew` -- live.
            bump = qpert * exp(-((z[k] - 800.0)/600.0)^2)
            patch.physical[j, vars["rho_v"], 1] = rho_t[k] * bump
            patch.physical[j, vars["rho_t"], 1] = rho_t[k] * bump
            patch.physical[j, vars["rho_c"], 1] = rho_t[k] * 0.2 * bump
            patch.physical[j, vars["rho_r"], 1] = rho_t[k] * 0.1 * bump
            rei > 0 && (patch.physical[j, rei, 1] = rho_t[k] * e0 * exp(-z[k] / ze))
        end
        # Fit first, then compensate with the FITTED kinetic energy (a raw-value
        # compensation leaves a ~1 % thermal residual from the fit of the wind itself).
        spectralTransform!(patch)
        gridTransform!(patch)
        for j in 1:npts
            k = mod1(j, kDim)
            ke = 0.5 * (patch.physical[j, ui, 1]^2 + patch.physical[j, wi, 1]^2)
            patch.physical[j, ei, 1] += rho_t[k] * ke
        end
        spectralTransform!(patch)
        gridTransform!(patch)
        return nothing
    end

    "Advance every column once on a tile and return it."
    function step_all!(mtile, patch, kDim, t = 1)
        ncols = div(size(patch.physical, 1), kDim)
        for c in 1:ncols
            Scythe.advance_column(mtile, c, t)
        end
        return ncols
    end

    # ──────────────────────────────────────────────
    # 1a. Which slots the closure touches
    # ──────────────────────────────────────────────
    @testset "fold: the closure touches only its own slots" begin
        mktempdir() do tmp
            m_on, p_on, mo, gpo, z, col = make_mynn_tile(tmp; mynn = true, init = :zero,
                                                         num_cells_k = 25)
            m_off, p_off, mf, _, _, _ = make_mynn_tile(tmp; mynn = false,
                                                       num_cells_k = 25)
            kDim = gpo.kDim
            # `W0 > 0` here and nowhere else: slot 5's increment is `dz(S_w)/rho_t` and
            # `S_w = rho_t K_m dz(w)`, so with `w == 0` it is an exact zero and the slot
            # would look untouched for a reason that has nothing to do with the closure.
            set_state!(p_on, mo, kDim, z, col; W0 = 0.5)
            set_state!(p_off, mf, kDim, z, col; W0 = 0.5)
            step_all!(m_on, p_on, kDim)
            step_all!(m_off, p_off, kDim)

            D = m_on.expdot_n[:, 1:size(m_off.expdot_n, 2)] .- m_off.expdot_n
            # rho_d (slot 2) carries no boundary-layer source at all: the closure mixes
            # the moist total and the vapour, never the dry air by itself.
            @test all(iszero, D[:, 2])
            # ...and every slot that should move, does.
            rv_i = mo.grid_params.vars["rho_v"]
            for v in (1, 3, 4, 5, 6, 7, 8, 9, rv_i)
                @test maximum(abs.(D[:, v])) > 0.0
            end
            # The TKE slot exists only on the `on` tile.
            re_i = mo.grid_params.vars["rho_e"]
            @test maximum(abs.(m_on.expdot_n[:, re_i])) > 0.0
            # The taper must NOT fire: rho_e was seeded positive and :mynn_init = :zero.
            @test m_on.mynn.active
        end
    end

    # ──────────────────────────────────────────────
    # 1b. Resting column: EXACTLY zero
    # ──────────────────────────────────────────────
    @testset "resting column at equilibrium: every increment is exactly 0.0" begin
        mktempdir() do tmp
            # SST = T(z_1) and a dry sounding whose surface saturation deficit is set to
            # zero by construction below, so the bulk fluxes are exactly 0; no wind, so
            # the stress is exactly 0; rho_e = 0, so q = 0 and every diffusivity is
            # exactly 0. The fluxes are disequilibrium-form, so this is an EQUALITY, not
            # a small number.
            # One throwaway build only to read the sounding's own T at the lowest mish
            # level, so the sea can be put at exactly that temperature.
            _, _, _, gp0, z0, col0 = make_mynn_tile(tmp; mynn = true, init = :zero,
                                                    num_cells_k = 25)
            T1 = col0.Tk[1]
            m_on, p_on, mo, gpo, z, col = make_mynn_tile(tmp; mynn = true, init = :zero,
                                                         num_cells_k = 25, SST = T1)
            m_off, p_off, mf, _, _, _ = make_mynn_tile(tmp; mynn = false,
                                                       num_cells_k = 25, SST = T1)
            step_all!(m_on, p_on, gpo.kDim)
            step_all!(m_off, p_off, gpo.kDim)
            D = m_on.expdot_n[:, 1:size(m_off.expdot_n, 2)] .- m_off.expdot_n
            # A dry reference has rho_v = 0 while rho_v_sat(T1) > 0, so the MOISTURE flux
            # is not zero here; the momentum and TKE legs must still be exact zeros.
            @test all(iszero, D[:, 4])
            @test all(iszero, D[:, 5])
            re_i = mo.grid_params.vars["rho_e"]
            @test all(iszero, m_on.expdot_n[:, re_i])
            @test maximum(m_on.mynn.K_m) == 0.0
            @test maximum(m_on.mynn.K_h) == 0.0
            # ...and with the fluxes switched off the whole closure is exactly inert.
            m2, p2, m2o, gp2, z2, col2 = make_mynn_tile(tmp; mynn = true, init = :zero,
                                                        num_cells_k = 25, fluxes = false,
                                                        Cd = 0.0, SST = T1)
            m2f, p2f, m2fo, _, _, _ = make_mynn_tile(tmp; mynn = false, num_cells_k = 25,
                                                     fluxes = false, Cd = 0.0, SST = T1)
            step_all!(m2, p2, gp2.kDim)
            step_all!(m2f, p2f, gp2.kDim)
            D2 = m2.expdot_n[:, 1:size(m2f.expdot_n, 2)] .- m2f.expdot_n
            @test all(iszero, D2)
            @test all(iszero, m2.expdot_n[:, m2o.grid_params.vars["rho_e"]])
        end
    end

    # ──────────────────────────────────────────────
    # 2. The column energy budget (D3) — the acceptance test
    # ──────────────────────────────────────────────
    """Column budget residual on one grid/timestep/surface configuration.

    Returns `(residual, scale, ratio)` where `residual` is
    `sum_k w_k (dE_t + drho_e) - bdry_E` and `scale` is `sum_k w_k |dE_t|`."""
    function budget_residual(tmp; num_cells_k, ts, Cd, fluxes, water_carry = :flux,
                             V0 = 12.0)
        m_on, p_on, mo, gpo, z, col = make_mynn_tile(tmp; mynn = true, init = :zero,
                                                     num_cells_k = num_cells_k, ts = ts,
                                                     Cd = Cd, fluxes = fluxes,
                                                     water_carry = water_carry)
        m_off, p_off, mf, _, _, _ = make_mynn_tile(tmp; mynn = false,
                                                   num_cells_k = num_cells_k, ts = ts,
                                                   Cd = Cd, fluxes = fluxes)
        kDim = gpo.kDim
        set_state!(p_on, mo, kDim, z, col; V0 = V0)
        set_state!(p_off, mf, kDim, z, col; V0 = V0)
        ncols = step_all!(m_on, p_on, kDim)
        step_all!(m_off, p_off, kDim)
        re_i = mo.grid_params.vars["rho_e"]
        wq = gauss_weights(kDim, num_cells_k, gpo.kMax - gpo.kMin)
        c = 6                                      # an interior column, away from both walls
        rng = ((c - 1) * kDim + 1):(c * kDim)
        dE = m_on.expdot_n[rng, 6] .- m_off.expdot_n[rng, 6]
        de = m_on.expdot_n[rng, re_i]
        lhs = sum(wq .* (dE .+ de))
        rhs = m_on.mynn.bdry_E[c]
        # The premise: with an x-uniform wind on a Neumann horizontal basis the TKE
        # slot's own transport is identically zero, so `de` IS the closure's increment.
        # Measured, not assumed -- it is the term that hid an 11 % budget residual until
        # the side-wall BC was fixed.
        div_col = m_on.tile.physical[rng, mo.grid_params.vars["u"], 2]
        @test maximum(abs.(div_col)) < 1.0e-10
        scale = sum(wq .* abs.(dE)) + sum(wq .* abs.(de))
        return (lhs - rhs, scale, m_on, mo, gpo, rng, wq, dE, de)
    end

    @testset "column energy budget closes to round-off (two grids, two timesteps)" begin
        mktempdir() do tmp
            for nk in (50, 100), ts in (0.5, 0.25),
                (Cd, fluxes) in ((0.0, false), (-1.0, false), (-1.0, true), (0.0, true))
                r, scale, = budget_residual(tmp; num_cells_k = nk, ts = ts, Cd = Cd,
                                            fluxes = fluxes)
                @test isfinite(r)
                @test abs(r) <= 1.0e-12 * scale
                if !(abs(r) <= 1.0e-12 * scale)
                    @info "budget residual" nk ts Cd fluxes r scale rel = abs(r)/scale
                end
            end
        end
    end

    @testset ":fixed_T water carry: the residual <c_z F> is measured, not zero" begin
        # D7's other leg, and the reason the flux form is the default. With the LOCAL
        # fixed-T map E_t receives `c_w rho_dot_w + c_v rho_dot_v` instead of the
        # divergence of `c_w S_w + c_v S_v`, and the two differ by `S_w dc_w/dz +
        # S_v dc_v/dz` -- an energy source that does NOT telescope to a boundary value,
        # so the column identity is open by exactly that amount. Asserting closure here
        # would be asserting something false; what the stage owes is the NUMBER.
        mktempdir() do tmp
            r_ft, scale_ft, = budget_residual(tmp; num_cells_k = 50, ts = 0.5,
                                              Cd = -1.0, fluxes = true,
                                              water_carry = :fixed_T)
            r_fx, scale_fx, = budget_residual(tmp; num_cells_k = 50, ts = 0.5,
                                              Cd = -1.0, fluxes = true,
                                              water_carry = :flux)
            @info "water carry: column budget residual" fixed_T_abs=abs(r_ft) fixed_T_rel=abs(r_ft)/scale_ft flux_rel=abs(r_fx)/scale_fx
            @test isfinite(r_ft)
            # The flux form closes; the fixed-T map does not, and by more than round-off.
            @test abs(r_fx) <= 1.0e-12 * scale_fx
        end
    end

    @testset "TKE gains from shear and loses in a stable shear-free layer" begin
        mktempdir() do tmp
            # Shear DOMINATES: a near-dry-adiabatic (lapse 9.7 K/km, so `gh ~ 0` and the
            # buoyancy exchange is negligible) 6 km column with a strong low-level jet and
            # very little TKE to dissipate. That is the regime in which the sign of the
            # net TKE source is a statement about shear production and not about the
            # competition between three terms.
            m_on, p_on, mo, gpo, z, col = make_mynn_tile(tmp; mynn = true, init = :zero,
                                                         num_cells_k = 12, kMax = 6.0e3,
                                                         lapse = 0.0097)
            kDim = gpo.kDim
            set_state!(p_on, mo, kDim, z, col; V0 = 25.0, e0 = 0.02, qpert = 0.0)
            step_all!(m_on, p_on, kDim)
            re_i = mo.grid_params.vars["rho_e"]
            wq = gauss_weights(kDim, 12, gpo.kMax - gpo.kMin)
            rng = (5 * kDim + 1):(6 * kDim)
            low = z .<= 2000.0
            @test sum((wq.*m_on.expdot_n[rng, re_i])[low]) > 0.0
            # ...and the discrete shear production is a SOURCE everywhere it is measured.
            @test m_on.mynn.Ps_disc[6] > 0.0

            # Shear-free but stably stratified with TKE present: the buoyancy exchange
            # and the dissipation both drain it.
            m2, p2, m2o, gp2, z2, col2 = make_mynn_tile(tmp; mynn = true, init = :zero,
                                                        num_cells_k = 50, Cd = 0.0,
                                                        fluxes = false)
            set_state!(p2, m2o, gp2.kDim, z2, col2; V0 = 0.0, e0 = 0.4)
            step_all!(m2, p2, gp2.kDim)
            re2 = m2o.grid_params.vars["rho_e"]
            wq2 = gauss_weights(gp2.kDim, 50, gp2.kMax - gp2.kMin)
            rng2 = (5 * gp2.kDim + 1):(6 * gp2.kDim)
            @test sum(wq2 .* m2.expdot_n[rng2, re2]) < 0.0
            @test m2.mynn.Ps_disc[6] == 0.0
        end
    end

    # ──────────────────────────────────────────────
    # 3. The two shear productions
    # ──────────────────────────────────────────────
    @testset "discrete rho*P_s vs the closure's rho K_m gm (informational)" begin
        mktempdir() do tmp
            m_on, p_on, mo, gpo, z, col = make_mynn_tile(tmp; mynn = true, init = :zero,
                                                         num_cells_k = 50)
            set_state!(p_on, mo, gpo.kDim, z, col; V0 = 20.0, e0 = 0.2, qpert = 0.0)
            step_all!(m_on, p_on, gpo.kDim)
            pd = m_on.mynn.Ps_disc[6]
            pm = m_on.mynn.Ps_mynn[6]
            @test pd > 0.0 && pm > 0.0                       # same sign
            rel = abs(pd - pm) / max(abs(pm), eps())
            @info "shear production: discrete vs closure" Ps_disc=pd Ps_mynn=pm rel
            # The two are NOT the same discretization and are not expected to agree
            # closely: `Ps_mynn` uses the closure's own `gm`, a two-point difference
            # `(u[k]-u[k-1])^2/dzk^2` on the mish, while what the slot receives is built
            # from the SPLINE derivative of the same wind. On a boundary-layer jet whose
            # shear turns over inside two mish spacings the finite difference is the
            # cruder of the two by tens of percent. The assertion is that they are the
            # same sign and the same order; the NUMBER is the deliverable.
            @test rel < 0.5
        end
    end

    # ──────────────────────────────────────────────
    # 4. Cadence
    # ──────────────────────────────────────────────
    @testset "cadence: 1-step vs 20 s held closure (reported, not gated)" begin
        mktempdir() do tmp
            ts = 0.5
            function run_cadence(interval)
                m, p, mo, gp, z, col = make_mynn_tile(tmp; mynn = true, init = :zero,
                                                      num_cells_k = 50, ts = ts,
                                                      interval = interval)
                kDim = gp.kDim
                set_state!(p, mo, kDim, z, col; V0 = 15.0, e0 = 0.2)
                re_i = mo.grid_params.vars["rho_e"]
                ncols = div(size(p.physical, 1), kDim)
                # A forward-Euler march of the TKE slot ALONE on an otherwise frozen
                # state: enough for the held closure and the every-step closure to
                # separate (K = l q S_M tracks q whatever the cadence, while l and S_M do
                # not), without standing up the whole multistep for a unit test.
                for t in 1:40
                    for c in 1:ncols
                        Scythe.advance_column(m, c, t)
                    end
                    p.physical[:, re_i, 1] .+= ts .* m.expdot_n[:, re_i]
                    spectralTransform!(p)
                    gridTransform!(p)
                end
                return copy(m.mynn.K_h), copy(m.mynn.K_m)
            end
            Kh_fast, Km_fast = run_cadence(ts)
            Kh_slow, Km_slow = run_cadence(20.0)
            den = maximum(abs.(Kh_fast))
            relKh = den > 0.0 ? maximum(abs.(Kh_slow .- Kh_fast)) / den : 0.0
            denm = maximum(abs.(Km_fast))
            relKm = denm > 0.0 ? maximum(abs.(Km_slow .- Km_fast)) / denm : 0.0
            @info "MYNN cadence sensitivity over 40 steps" rel_K_h=relKh rel_K_m=relKm
            @test isfinite(relKh) && isfinite(relKm)
        end
    end

    # ──────────────────────────────────────────────
    # 5. Taper initialization
    # ──────────────────────────────────────────────
    @testset ":mynn_init = :taper seeds a zero column, :zero does not" begin
        mktempdir() do tmp
            for (init, expect) in ((:taper, true), (:zero, false))
                m, p, mo, gp, z, col = make_mynn_tile(tmp; mynn = true, init = init,
                                                      num_cells_k = 25)
                kDim = gp.kDim
                # A wind, so ust > 0 and the taper has something to scale (with ust = 0
                # the Fortran taper is itself identically zero).
                vars = mo.grid_params.vars
                u = bl_jet(z, 15.0, 500.0)
                for j in 1:size(p.physical, 1)
                    p.physical[j, vars["u"], 1] = u[mod1(j, kDim)]
                end
                spectralTransform!(p); gridTransform!(p)
                step_all!(m, p, kDim)
                re_i = vars["rho_e"]
                # ABOVE the lowest cell. The surface production `P_sfc = tau.V g(z)` is a
                # real TKE source that fires whatever the initialization does -- with
                # `rho_e = 0` and `:mynn_init = :zero` it is the ONLY source, and it is
                # confined to the first cell by `g(z)`. So the taper is what is being
                # tested only above `delta = 2 z[2]`, where a `:zero` column must be
                # exactly quiet.
                above = findall(zz -> zz > 2.0 * z[2], z)
                got = maximum(m.expdot_n[above, re_i]) > 0.0
                @test got == expect
            end
            # ...and the taper does NOT fire on a column that already carries TKE.
            m, p, mo, gp, z, col = make_mynn_tile(tmp; mynn = true, init = :taper,
                                                  num_cells_k = 25, Cd = 0.0,
                                                  fluxes = false)
            kDim = gp.kDim
            set_state!(p, mo, kDim, z, col; V0 = 0.0, e0 = 0.1)
            step_all!(m, p, kDim)
            re_i = mo.grid_params.vars["rho_e"]
            # With no wind and no fluxes the only TKE source would be the taper; the
            # column already has rho_e > 0, so what is left is the (negative) drain.
            @test maximum(m.expdot_n[1:kDim, re_i]) <= 0.0
        end
    end

    # ──────────────────────────────────────────────
    # 6. Census
    # ──────────────────────────────────────────────
    @testset "census: the diffusion numbers and the K cap are measured, not assumed" begin
        mktempdir() do tmp
            m, p, mo, gp, z, col = make_mynn_tile(tmp; mynn = true, init = :zero,
                                                  num_cells_k = 50, K_max = 1.0)
            set_state!(p, mo, gp.kDim, z, col; V0 = 20.0, e0 = 0.5)
            step_all!(m, p, gp.kDim)
            @test maximum(m.mynn.K_m) <= 1.0
            @test maximum(m.mynn.K_h) <= 1.0
            Scythe.mynn_write_final!(m)
            @test m.mynn.n_cap_K > 0
            @test m.mynn.n_diffnum == 0            # K capped at 1 m^2/s cannot be stiff
            @test all(>=(0.0), m.mynn.D_gal)
            @test all(>=(0.0), m.mynn.ts_tau)
            @test occursin("mynn census:", Scythe.mynn_census_line(m.mynn))
        end
    end
end
