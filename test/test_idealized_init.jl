using Test
using Scythe
using Springsteel

# Tests for the idealized initialization utilities (src/idealized.jl) used by
# the benchmark scripts.

@testset "Idealized initialization" begin

    function make_rz_patch(; num_cells=16, kDim=32, iMax=8000.0, kMax=4000.0)
        vars = Dict("s" => 1, "xi" => 2, "mu" => 3, "u" => 4, "w" => 5)
        bc = Dict(v => NeumannBC() for v in keys(vars))
        gp = GridParameters(
            geometry = "RZ",
            num_cells = num_cells,
            iMin = 0.0, iMax = iMax,
            kMin = 0.0, kMax = kMax,
            kDim = kDim,
            BCL = bc, BCR = bc, BCB = bc, BCT = bc,
            vars = vars,
        )
        patch = createGrid(gp)
        return gp, patch
    end

    function make_reference(tmpdir, gp, patch; theta=300.0)
        sounding = Scythe.write_dry_sounding(joinpath(tmpdir, "test.ref");
                                             theta=theta, zmax=6000.0)
        model = ModelParameters(
            ts = 0.1, equation_set = "Euler_test",
            ref_state_file = sounding, grid_params = gp,
            physical_params = Dict(:K => 0.0),
        )
        gridpoints = getGridpoints(patch)
        kDim = gp.kDim
        z = gridpoints[1:kDim, 2]
        column = Scythe.reference_column(patch, model.grid_params)
        ref = Scythe.calculate_reference_state(model, z, column)
        return ref, gridpoints, z
    end

    # ──────────────────────────────────────────────
    # 1. Dry sounding produces an isentropic hydrostatic reference
    # ──────────────────────────────────────────────
    @testset "Dry sounding reference state" begin
        mktempdir() do tmpdir
            gp, patch = make_rz_patch()
            ref, _, z = make_reference(tmpdir, gp, patch)
            prof = Scythe.reference_profiles(ref)

            # Constant potential temperature at the sounding value
            @test all(abs.(prof.theta .- 300.0) .< 0.1)

            # Dry adiabat: T = theta * (p/p0)^(Rd/Cpd)
            T_expected = 300.0 .* (prof.p ./ 1000.0) .^ (Scythe.Rd / Scythe.Cpd)
            @test all(abs.(prof.Tk .- T_expected) .< 0.1)

            # Hydrostatic balance: dp/dz = -rho_t * g (finite differences on
            # the Chebyshev levels, interior points only)
            rho_t = prof.rho_d .* (1.0 .+ prof.q_v)
            for k in 2:(length(z) - 1)
                dpdz = 100.0 * (prof.p[k+1] - prof.p[k-1]) / (z[k+1] - z[k-1])
                rhog = -Scythe.gravity * rho_t[k]
                @test isapprox(dpdz, rhog; rtol=2.0e-3)
            end
        end
    end

    # ──────────────────────────────────────────────
    # 2. Straka cold bubble reproduces the prescribed dT
    # ──────────────────────────────────────────────
    @testset "temperature_bubble!" begin
        mktempdir() do tmpdir
            gp, patch = make_rz_patch()
            ref, gridpoints, _ = make_reference(tmpdir, gp, patch)
            prof = Scythe.reference_profiles(ref)

            patch.physical .= 0.0
            Scythe.temperature_bubble!(patch, gridpoints, ref;
                                       xc=0.0, xr=4000.0, zc=2000.0, zr=1000.0,
                                       dT_max=-15.0)

            kDim = gp.kDim
            i = 1
            max_err = 0.0
            for _ in 1:Springsteel.num_columns(patch)
                for k in 1:kDim
                    x = gridpoints[i, 1]; z = gridpoints[i, 2]
                    L = sqrt((x / 4000.0)^2 + ((z - 2000.0) / 1000.0)^2)
                    dT = L <= 1.0 ? -15.0 * (cos(pi * L) + 1.0) / 2.0 : 0.0
                    s_tot = patch.physical[i, 1, 1] + ref.sbar[k, 1]
                    xi_tot = patch.physical[i, 2, 1] + ref.xibar[k, 1]
                    q_v, rho_d, Tk, p = Scythe.thermodynamic_tuple(s_tot, xi_tot, ref.mubar[k, 1])
                    max_err = max(max_err, abs(Tk - (prof.Tk[k] + dT)))
                    # Bubble is applied at constant pressure
                    max_err = max(max_err, abs(p - prof.p[k]) * 0.1)
                    i += 1
                end
            end
            @test max_err < 1.0e-6

            # Amplitude reached inside the bubble
            @test minimum(patch.physical[:, 1, 1]) < -10.0  # entropy deficit present
        end
    end

    # ──────────────────────────────────────────────
    # 3. BF02 warm bubble reproduces the prescribed dtheta
    # ──────────────────────────────────────────────
    @testset "theta_bubble!" begin
        mktempdir() do tmpdir
            gp, patch = make_rz_patch()
            ref, gridpoints, _ = make_reference(tmpdir, gp, patch)
            prof = Scythe.reference_profiles(ref)

            patch.physical .= 0.0
            Scythe.theta_bubble!(patch, gridpoints, ref;
                                 xc=4000.0, xr=2000.0, zc=2000.0, zr=2000.0,
                                 dtheta_max=2.0)

            kDim = gp.kDim
            i = 1
            max_err = 0.0
            for _ in 1:Springsteel.num_columns(patch)
                for k in 1:kDim
                    x = gridpoints[i, 1]; z = gridpoints[i, 2]
                    L = sqrt(((x - 4000.0) / 2000.0)^2 + ((z - 2000.0) / 2000.0)^2)
                    dtheta = L <= 1.0 ? 2.0 * (cos(pi * L / 2.0))^2 : 0.0
                    s_tot = patch.physical[i, 1, 1] + ref.sbar[k, 1]
                    xi_tot = patch.physical[i, 2, 1] + ref.xibar[k, 1]
                    theta = Scythe.potential_temperature(s_tot, xi_tot, ref.mubar[k, 1])
                    max_err = max(max_err, abs(theta - (prof.theta[k] + dtheta)))
                    i += 1
                end
            end
            @test max_err < 1.0e-6
        end
    end

    # ──────────────────────────────────────────────
    # Moist construction (Bryan & Fritsch 2002 base state)
    # ──────────────────────────────────────────────
    @testset "saturated_surface_state" begin
        sfc = Scythe.saturated_surface_state(q_t=0.02, theta_e=320.0, sfc_p_hPa=1000.0)
        te = Scythe.reversible_theta_e(sfc.s, sfc.xi,
                                       Scythe.mu_transform(sfc.q_v),
                                       Scythe.mu_transform(sfc.q_l))
        @test abs(te - 320.0) < 1.0e-8
        @test abs(sfc.q_v - Scythe.q_sat_liquid(sfc.T, sfc.p)) < 1.0e-12
        @test sfc.q_l > 0.0
        @test abs(sfc.s_rev - (sfc.s + sfc.q_l * Scythe.Cl * log(sfc.T / Scythe.T_0))) < 1.0e-12
    end

    @testset "saturated_hydrostatic_profile" begin
        mktempdir() do tmpdir
            vars = Dict("s" => 1, "xi" => 2, "mu" => 3, "u" => 4, "w" => 5,
                        "mu_l" => 6, "qss" => 7)
            bc = Dict(v => NeumannBC() for v in keys(vars))
            gp = GridParameters(
                geometry = "RZ",
                num_cells = 8,
                iMin = 0.0, iMax = 4000.0,
                kMin = 0.0, kMax = 10000.0,
                kDim = 50,
                BCL = bc, BCR = bc, BCB = bc, BCT = bc,
                vars = vars,
            )
            patch = createGrid(gp)
            sounding = Scythe.write_moist_neutral_sounding(
                joinpath(tmpdir, "moist.ref"); q_t=0.02, theta_e=320.0, zmax=12000.0)
            model = ModelParameters(
                ts = 0.1, equation_set = "BF02_test",
                ref_state_file = sounding, grid_params = gp,
                physical_params = Dict(:K => 0.0),
            )
            gridpoints = getGridpoints(patch)
            z = gridpoints[1:gp.kDim, 2]
            column = Scythe.reference_column(patch, gp)
            ref = Scythe.calculate_reference_state(model, z, column)

            base = Scythe.saturated_hydrostatic_profile(z, column, ref;
                                                        q_t=0.02, theta_e=320.0)

            # Exactly saturated everywhere
            q_sat = Scythe.q_sat_liquid.(base.Tk, base.p)
            @test maximum(abs.(base.q_v .- q_sat)) < 1.0e-8

            # Constant reversible entropy
            sfc = Scythe.saturated_surface_state(q_t=0.02, theta_e=320.0)
            s_rev = base.s .+ (base.q_l .* Scythe.Cl .* log.(base.Tk ./ Scythe.T_0))
            @test maximum(abs.(s_rev .- sfc.s_rev)) < 1.0e-4

            # Uniform theta_e at the specified value
            @test maximum(abs.(base.theta_e .- 320.0)) < 0.1

            # Hydrostatically balanced
            @test maximum(abs.(base.residual)) < 1.0e-4

            # Positive cloud water everywhere
            @test all(base.q_l .> 0.0)

            # Round-trip through the exact reference state file
            exact_path = Scythe.write_exact_ref(joinpath(tmpdir, "exact.ref"),
                                                z, base.s, base.xi, base.mu)
            model_exact = ModelParameters(
                ts = 0.1, equation_set = "BF02_test",
                ref_state_file = exact_path, grid_params = gp,
                physical_params = Dict(:K => 0.0),
                options = Dict(:semiimplicit => false, :exact_reference_state => true),
            )
            ref2 = Scythe.exact_reference_state(model_exact, z, column)
            # These bound the spectral-refit round-trip (write base -> read back -> refit), not
            # the physics; the physics assertions above (saturation, reversible entropy,
            # theta_e, hydrostatic residual) carry that. `saturated_hydrostatic_profile` is a
            # fixed-iteration solver (n_outer/n_inner, no convergence tolerance) seeded from
            # `ref`, so `base` is only converged to the level of its own hydrostatic residual
            # (~5e-8). The tolerances therefore have to sit above that residual, not below it —
            # the previous 1e-9/1e-6 values were pinned to one builder's exact arithmetic and
            # tripped as soon as `calculate_reference_state` moved onto the shared Springsteel
            # builder (a ~1e-9 change in the seed).
            @test maximum(abs.(ref2.sbar[:, 1] .- base.s)) < 1.0e-6
            @test maximum(abs.(ref2.xibar[:, 1] .- base.xi)) < 1.0e-8
            @test maximum(abs.(ref2.mubar[:, 1] .- base.mu)) < 1.0e-5

            # Moist bubble: theta_rho increased by the buoyancy factor at the
            # bubble center, saturation preserved, untouched outside
            patch.physical .= 0.0
            Scythe.moist_buoyancy_bubble!(patch, gridpoints, base, ref2;
                                          q_t=0.02, xc=2000.0, xr=2000.0,
                                          zc=2000.0, zr=2000.0, amp=2.0/300.0)
            kDim = gp.kDim
            i = 1
            max_sat_err = 0.0
            max_trho_err = 0.0
            outside_ok = true
            for _ in 1:Springsteel.num_columns(patch)
                for k in 1:kDim
                    x = gridpoints[i, 1]; zz = gridpoints[i, 2]
                    L = sqrt(((x - 2000.0) / 2000.0)^2 + ((zz - 2000.0) / 2000.0)^2)
                    s_tot = patch.physical[i, 1, 1] + ref2.sbar[k, 1]
                    xi_tot = patch.physical[i, 2, 1] + ref2.xibar[k, 1]
                    mu_tot = patch.physical[i, 3, 1] + ref2.mubar[k, 1]
                    q_v, rho_d, Tk, p = Scythe.thermodynamic_tuple(s_tot, xi_tot, mu_tot)
                    if L <= 1.0
                        # Saturation preserved inside the bubble
                        max_sat_err = max(max_sat_err,
                                          abs(q_v - Scythe.q_sat_liquid(Tk, p)))
                        # Density potential temperature increased by roughly
                        # 1 + b_incr; vapor resets to saturation at the warmer
                        # temperature, so the perturbation slightly overshoots
                        # the nominal target (BF02 eq. 36: "slightly more water
                        # vapor and slightly less cloud water")
                        q_l = Scythe.inv_mu_transform(patch.physical[i, 6, 1])
                        q_tot = q_v + q_l
                        theta = Scythe.potential_temperature(s_tot, xi_tot, mu_tot)
                        theta_rho = theta * (1.0 + q_v / Scythe.Eps) / (1.0 + q_tot)
                        b_incr = (2.0 / 300.0) * (cos(pi * L / 2.0))^2
                        trho_pert = theta_rho - base.theta_rho[k]
                        nominal = base.theta_rho[k] * b_incr
                        max_trho_err = max(max_trho_err, abs(trho_pert - nominal))
                        outside_ok &= trho_pert >= -1.0e-6   # perturbation is buoyant
                    else
                        # Untouched outside, modulo the spectral refit of the exact reference
                        # (bounded by the same round-trip error asserted above, not by physics)
                        outside_ok &= abs(patch.physical[i, 1, 1]) < 1.0e-5
                        outside_ok &= abs(patch.physical[i, 3, 1]) < 1.0e-5
                        outside_ok &= patch.physical[i, 6, 1] == base.mu_l[k]
                    end
                    i += 1
                end
            end
            @test max_sat_err < 1.0e-6
            # Within ~35% of the nominal cos^2 profile (vapor re-saturation
            # systematically deepens the perturbation)
            @test max_trho_err < 0.35 * (2.0 / 300.0) * maximum(base.theta_rho)
            @test outside_ok
        end
    end

    # ──────────────────────────────────────────────
    # 4. write_ics_csv round-trips through read_physical_grid
    # ──────────────────────────────────────────────
    @testset "write_ics_csv roundtrip" begin
        mktempdir() do tmpdir
            gp, patch = make_rz_patch()
            ref, gridpoints, _ = make_reference(tmpdir, gp, patch)
            patch.physical .= 0.0
            Scythe.temperature_bubble!(patch, gridpoints, ref)
            patch.physical[:, 4, 1] .= 1.5   # nonzero u to exercise all columns

            path = Scythe.write_ics_csv(joinpath(tmpdir, "ics.csv"), patch, gridpoints)
            patch2 = createGrid(gp)
            read_physical_grid(path, patch2)
            for v in 1:5
                @test patch2.physical[:, v, 1] == patch.physical[:, v, 1]
            end
        end
    end

    @testset "balanced modified-Rankine vortex" begin
        import Springsteel.Thermodynamics: Rd, Rv, gravity

        # ── v profile ──
        Vmax = 15.0; RMW = 50.0e3; alpha = 0.3; v_top = 15.0e3
        vfun(r, z) = Scythe.modified_rankine_v(r, z; Vmax, RMW, alpha, v_top)
        @test vfun(0.0, 0.0) == 0.0
        @test vfun(RMW, 0.0) == Vmax
        @test vfun(25.0e3, 0.0) == 0.5 * Vmax                 # linear inside
        @test vfun(2.0 * RMW, 0.0) ≈ Vmax * 0.5^alpha          # decay outside
        @test vfun(RMW, 7.5e3) ≈ 0.5 * Vmax                    # linear in height
        @test vfun(RMW, v_top) == 0.0
        @test vfun(RMW, 20.0e3) == 0.0                         # clamped above

        # ── thermal wind: no vortex leaves the background exactly ──
        z = collect(0.0:125.0:25.0e3)
        r_axis = collect(0.0:500.0:300.0e3)
        nz = length(z); nr = length(r_axis)
        # Analytic isothermal moist hydrostatic background (T0 = 300 keeps
        # q = 10 g/kg comfortably subsaturated at the surface)
        T0 = 300.0; q = 0.01; p0 = 1.0e5
        Reff = Rd + q * Rv
        pbar = @. p0 * exp(-(1.0 + q) * gravity * z / (Reff * T0))
        rho_dbar = pbar ./ (Reff * T0)
        rho_vbar = q .* rho_dbar
        rho_tbar = rho_dbar .+ rho_vbar
        lnrho = Scythe.thermal_wind_lnrho(r_axis, z, zeros(nz, nr), log.(rho_tbar))
        @test lnrho ≈ repeat(log.(rho_tbar), 1, nr) atol = 0.0  # exact

        flds0 = Scythe.balanced_vortex_fields(r_axis, z, pbar, rho_dbar, rho_vbar;
                                              Vmax = 0.0, fcor = 3.775e-5)
        @test flds0.rho_t ≈ repeat(rho_tbar, 1, nr)            # exactly background
        @test flds0.n_supersat == 0
        # With no vortex the pressure must be the REFERENCE pressure bitwise, not
        # merely close: the hydrostatic integration is in perturbation form
        # (dp'/dz = -rho_t' g), so rho_t' = 0 leaves p' = 0 identically. Rebuilding
        # the full profile by quadrature instead accumulated error downward from
        # the top -- on the TC's Dunion sounding that was 4.6 hPa and -12 K at the
        # outer edge, which drove RH past saturation near the tropopause.
        @test flds0.p == repeat(pbar, 1, nr)

        # ── balanced vortex: residual, warm core, low center ──
        flds = Scythe.balanced_vortex_fields(r_axis, z, pbar, rho_dbar, rho_vbar;
                                             Vmax, RMW, alpha, v_top, fcor = 3.775e-5)
        # The max residual lives exactly at the RMW, where the modified-Rankine
        # kink makes the centered-difference metric first-order (0.006 at
        # dr = 500 m, halving with dr); away from the kink the balance is tight.
        @test flds.residual < 1.0e-2
        offkink = 0.0
        dpdrmax = 1.0e-300
        for j in 2:nr-1, k in 1:nz
            abs(r_axis[j] - RMW) > 3.0e3 || continue
            dpdr = (flds.p[k, j+1] - flds.p[k, j-1]) / (r_axis[j+1] - r_axis[j-1])
            r = r_axis[j]
            C = r > 0.0 ? (flds.v[k, j]^2 / r) + (3.775e-5 * flds.v[k, j]) : 0.0
            offkink = max(offkink, abs(dpdr - (flds.rho_t[k, j] * C)))
            dpdrmax = max(dpdrmax, abs(dpdr))
        end
        @test offkink / dpdrmax < 1.0e-3
        @test flds.n_supersat == 0
        @test flds.p[1, 1] < flds.p[1, nr]                     # central pressure deficit
        @test flds.Tk[10, 1] > flds.Tk[10, nr]                 # warm core aloft
        @test all(isfinite, flds.rho_t) && all(flds.rho_t .> 0.0)
        # v = 0 above v_top: density relaxes to the background there
        k_top = findfirst(>=(16.0e3), z)
        @test maximum(abs.(flds.rho_t[k_top:end, :] .- rho_tbar[k_top:end])) /
              maximum(rho_tbar) < 1.0e-6
    end

    # ──────────────────────────────────────────────
    # 8. moist_compressible temperature perturbations: the shared constant-pressure,
    #    RH-preserving body behind the warm bubble and the RCE noise trigger.
    # ──────────────────────────────────────────────

    # A Spline x Chebyshev (RZ) patch carrying the full moist_compressible slot list.
    function make_mc_patch(; num_cells=8, kDim=12, iMax=4000.0, kMax=3000.0,
                           options=Dict{Symbol,Any}())
        varlist = Scythe.mc_var_names(options; cyl=false)
        vars = Dict(v => i for (i, v) in enumerate(varlist))
        bc = Dict(v => NeumannBC() for v in keys(vars))
        gp = GridParameters(
            geometry = "RZ", num_cells = num_cells,
            iMin = 0.0, iMax = iMax, kMin = 0.0, kMax = kMax, kDim = kDim,
            BCL = bc, BCR = bc, BCB = bc, BCT = bc, vars = vars,
        )
        return gp, createGrid(gp)
    end

    # A moist, SUBSATURATED (RH = 0.8), cloud-free pressure reference on the patch levels.
    # Subsaturated on purpose: the RH-preserving branch is only observable where the
    # reference humidity is strictly between 0 and saturation.
    function make_mc_ref(tmpdir, gp, patch; name="mc_pressure.ref")
        gridpoints = getGridpoints(patch)
        kDim = gp.kDim
        z = gridpoints[1:kDim, 2]
        Tk = 295.0 .- 0.0065 .* z
        p_Pa = 100000.0 .* exp.(-z ./ 8500.0)
        rho_v = 0.8 .* Springsteel.Thermodynamics.rho_v_sat.(Tk, p_Pa ./ 100.0)
        rho_d = (p_Pa .- (Scythe.Rv .* Tk .* rho_v)) ./ (Scythe.Rd .* Tk)
        f = Scythe.write_exact_ref_mc(joinpath(tmpdir, name), z, p_Pa, rho_d, rho_v,
                                      zeros(kDim))
        ref = Springsteel.exact_pressure_reference_state(f, z,
                  Springsteel.reference_column(patch, gp))
        return ref, gridpoints, z
    end

    function legacy_bubble!(patch::AbstractGrid, gridpoints::Matrix{Float64},
                                          ref::Springsteel.PressureReferenceState;
                                          xc=75.0e3, xr=16.0e3, zc=500.0, zr=3000.0,
                                          dT_max=3.0, zcol=2,
                                          condensate_transform::Symbol=:none,
                                          condensate_mu=1.0e-7,
                                          rain_transform::Symbol=:none, rain_mu=1.0e-7)
        vars = patch.params.vars
        p_i = vars["p"]; rho_d_i = vars["rho_d"]; rho_t_i = vars["rho_t"]
        et_i = vars["E_t"]; qss_i = vars["Q_ss"]; rho_r_i = Scythe.mc_slot(vars, "rho_r")
        rho_c_i = Scythe.mc_slot(vars, "rho_c")
        # The prognostic VAPOR slot, present in every configuration of this set (see `MC_VARS`).
        rho_v_i = Scythe.mc_slot(vars, "rho_v")
        # APPENDED optional slots: seeded only where they exist. `rain_number_slot(0.0, ...)` is
        # exactly 0.0 under every transform (`bhyp(0) == 0`), so no transform keyword has to be
        # threaded here — the same reason `rain_slot` needed none.
        n_r_i = Scythe.mc_optional_slot(vars, "n_r")
        ice_i = Scythe.mc_ice_slot_indices(vars)
        # The MYNN TKE density (options[:mynn]): a total with no transform, so its
        # seed is a plain 0.0 wherever the slot exists. Every idealized initial
        # condition in this file is a state at rest or a buoyant bubble, neither of
        # which carries resolved-scale turbulence; the taper spin-up the closure
        # wants (options[:mynn_init]) is applied by the scheme, not here.
        rho_e_i = Scythe.mc_optional_slot(vars, "rho_e")
        kDim = patch.params.kDim
        pbar = Springsteel.ref_pressure(ref); rho_dbar = Springsteel.ref_rho_d(ref); rho_tbar = Springsteel.ref_rho_t(ref)
        rho_cbar = Springsteel.ref_rho_c(ref)
        rho_vbar = Springsteel.ref_rho_v(ref)
        E_tbar = Springsteel.ref_total_energy(ref); Q_ssbar = Springsteel.ref_qss(ref)

        i = 1
        for _ in 1:Springsteel.num_columns(patch)
            for k in 1:kDim
                # On a 3D cylindrical grid (zcol = 3) this is a WN0 torus bubble:
                # column 1 is the radius and λ (column 2) does not enter.
                x = gridpoints[i, 1]
                z = gridpoints[i, zcol]
                L = sqrt(((x - xc) / xr)^2 + ((z - zc) / zr)^2)
                dT = L <= 1.0 ? dT_max * (cos(pi * L / 2.0))^2 : 0.0
                p_ref = pbar[k, 1]                              # Pa
                rho_dref = rho_dbar[k, 1]
                rho_vref = rho_vbar[k, 1]
                # Moist EOS reference temperature (matches the reference's own Tbar)
                T_ref = p_ref / ((rho_dref * Scythe.Rd) + (rho_vref * Scythe.Rv))
                Tk = T_ref + dT
                rho_v = rho_vref
                if dT > 0.0
                    H = rho_vref / Springsteel.Thermodynamics.rho_v_sat(T_ref, p_ref / 100.0)
                    rho_v = H * Springsteel.Thermodynamics.rho_v_sat(Tk, p_ref / 100.0)
                end
                rho_d = (p_ref - (rho_v * Scythe.Rv * Tk)) / (Scythe.Rd * Tk) # constant-pressure moist EOS
                rho_t = rho_d + rho_v
                q_v = rho_v / rho_d
                E_t = (rho_d * Springsteel.Thermodynamics.internal_energy_bf02(Tk, q_v, 0.0)) + (rho_t * Scythe.gravity * z)
                Q_ss = rho_v - Springsteel.Thermodynamics.rho_v_sat(Tk, p_ref / 100.0)
                patch.physical[i, p_i, 1] = 0.0
                patch.physical[i, rho_d_i, 1] = rho_d - rho_dref
                patch.physical[i, rho_t_i, 1] = rho_t - rho_tbar[k, 1]
                patch.physical[i, et_i, 1] = E_t - E_tbar[k, 1]
                patch.physical[i, qss_i, 1] = Q_ss - Q_ssbar[k, 1]
                patch.physical[i, rho_r_i, 1] = Scythe.rain_slot(0.0, rain_transform, rain_mu)
                patch.physical[i, rho_c_i, 1] =
                    Scythe.condensate_slot(0.0, rho_cbar[k, 1], condensate_transform, condensate_mu)
                # The moistened bubble's own vapor, against the DERIVED reference (the
                # cloud-free base makes rho_cbar exactly 0.0 here, but the term is written out
                # so this site cannot drift from the one the reconciliation differences).
                patch.physical[i, rho_v_i, 1] =
                    Scythe.vapor_slot(rho_v, rho_tbar[k, 1], rho_dbar[k, 1], rho_cbar[k, 1])
                n_r_i > 0 && (patch.physical[i, n_r_i, 1] = 0.0)
                Scythe.seed_ice_zero!(patch.physical, i, ice_i)
                rho_e_i > 0 && (patch.physical[i, rho_e_i, 1] = 0.0)
                i += 1
            end
        end
        return patch
    end

    @testset "moist_temperature_bubble_mc! is bitwise after the refactor" begin
        # The bubble's slot arithmetic now lives in the shared
        # `_moist_temperature_perturbation_mc!`; committed benchmark references (O01,
        # ocean_warm_bubble, o01_axisym) pin its output, so the extraction has to be
        # BITWISE, not merely close. `legacy_bubble!` above is the pre-refactor function
        # pasted verbatim (only the internal names qualified for the test module).
        mktempdir() do tmpdir
            opts = Dict{Symbol,Any}(:rain_moments => 2, :ice_microphysics => :ishmael,
                                    :mynn => true)
            gpA, pA = make_mc_patch(; options=opts)
            gpB, pB = make_mc_patch(; options=opts)
            refA, gptsA, _ = make_mc_ref(tmpdir, gpA, pA; name="a.ref")
            refB, gptsB, _ = make_mc_ref(tmpdir, gpB, pB; name="b.ref")
            @test gptsA == gptsB

            cases = (
                # The shipped defaults: xc = 75 km is off this small domain, so dT == 0
                # everywhere and the pin covers the untouched write of every slot.
                (;),
                # A bubble that actually lands, with the transforms off...
                (; xc=1500.0, xr=800.0, zc=1200.0, zr=700.0, dT_max=2.5),
                # ...and the same one with the bhyp control-variable transforms on.
                (; xc=1500.0, xr=800.0, zc=1200.0, zr=700.0, dT_max=2.5,
                   condensate_transform=:bhyp, rain_transform=:bhyp),
            )
            for kw in cases
                pA.physical .= 0.0
                pB.physical .= 0.0
                legacy_bubble!(pA, gptsA, refA; kw...)
                Scythe.moist_temperature_bubble_mc!(pB, gptsB, refB; kw...)
                @test pA.physical == pB.physical
            end
            # ...and the landing bubble is not a no-op, so the pin has something to pin.
            @test maximum(abs.(pB.physical[:, pB.params.vars["E_t"], 1])) > 0.0
        end
    end

    @testset "moist_temperature_noise_mc!" begin
        mktempdir() do tmpdir
            z_max = 1200.0
            dT_max = 0.2
            opts = Dict{Symbol,Any}(:rain_moments => 2, :ice_microphysics => :ishmael,
                                    :mynn => true)
            gp, patch = make_mc_patch(; num_cells=12, kDim=16, kMax=3000.0, options=opts)
            ref, gpts, _ = make_mc_ref(tmpdir, gp, patch)
            kDim = gp.kDim
            ncols = Springsteel.num_columns(patch)
            vars = patch.params.vars
            pbar = Springsteel.ref_pressure(ref)
            rho_dbar = Springsteel.ref_rho_d(ref)
            rho_tbar = Springsteel.ref_rho_t(ref)
            rho_cbar = Springsteel.ref_rho_c(ref)
            rho_vbar = Springsteel.ref_rho_v(ref)

            patch.physical .= 0.0
            Scythe.moist_temperature_noise_mc!(patch, gpts, ref; z_max, dT_max, seed=20260915)
            state = copy(patch.physical)

            # ── determinism: the draw is a function of the seed and the grid alone ──
            patch.physical .= 0.0
            Scythe.moist_temperature_noise_mc!(patch, gpts, ref; z_max, dT_max, seed=20260915)
            @test patch.physical == state
            patch.physical .= 0.0
            Scythe.moist_temperature_noise_mc!(patch, gpts, ref; z_max, dT_max, seed=20260916)
            @test patch.physical != state
            # ...and the RNG is seeded, never the global stream: drawing in between the two
            # calls must not move the initial condition.
            rand(1000)
            patch.physical .= 0.0
            Scythe.moist_temperature_noise_mc!(patch, gpts, ref; z_max, dT_max, seed=20260915)
            @test patch.physical == state

            # The dT_max = 0 write is the UNPERTURBED state (dT == 0 -> the branch is not
            # taken anywhere), which is what the taper must reproduce above z_max.
            patch.physical .= 0.0
            Scythe.moist_temperature_noise_mc!(patch, gpts, ref; z_max, dT_max=0.0,
                                               seed=20260915)
            rest = copy(patch.physical)

            # ── recover dT, RH and the untouched slots point by point ──
            dT = zeros(ncols, kDim)
            maxRH = 0.0
            aloft_bitwise = true
            i = 1
            for c in 1:ncols
                for k in 1:kDim
                    zk = gpts[i, 2]
                    p_ref = pbar[k, 1]
                    T_ref = p_ref / ((rho_dbar[k, 1] * Scythe.Rd) +
                                     (rho_vbar[k, 1] * Scythe.Rv))
                    rho_d = state[i, vars["rho_d"], 1] + rho_dbar[k, 1]
                    rho_v = state[i, vars["rho_v"], 1] +
                            (rho_tbar[k, 1] - rho_dbar[k, 1] - rho_cbar[k, 1])
                    Tk = p_ref / ((rho_d * Scythe.Rd) + (rho_v * Scythe.Rv))
                    dT[c, k] = Tk - T_ref
                    RH = rho_v / Springsteel.Thermodynamics.rho_v_sat(Tk, p_ref / 100.0)
                    RH_ref = rho_vbar[k, 1] /
                             Springsteel.Thermodynamics.rho_v_sat(T_ref, p_ref / 100.0)
                    maxRH = max(maxRH, abs(RH - RH_ref) / RH_ref)
                    if zk >= z_max
                        aloft_bitwise &= state[i, :, 1] == rest[i, :, 1]
                    end
                    i += 1
                end
            end

            # Amplitude. The DRAW is bounded by dT_max exactly; removing the per-level
            # mean can then carry a point past that bound by |m_k|, which is the mean of
            # `ncols` deviates uniform on [-dT_max, dT_max] and so has RMS
            # dT_max/sqrt(3 ncols). The hard bound is 2 dT_max and the 4-sigma bound is
            # below; on this grid the realized excess is ~8 %. Neither is a fitted number:
            # if the mean removal is ever dropped the first becomes tight at dT_max.
            @test maximum(abs.(dT)) <= 2.0 * dT_max
            @test maximum(abs.(dT)) <= dT_max * (1.0 + 4.0 / sqrt(3.0 * ncols))
            # ...and it is large enough to be a real trigger.
            @test maximum(abs.(dT)) > 0.3 * dT_max
            # Both signs are present -- the RH-preserving branch now admits dT < 0.
            @test minimum(dT) < 0.0 && maximum(dT) > 0.0

            # Taper: NOTHING is seeded at or above z_max, bitwise (not just small).
            @test aloft_bitwise
            # ...and the surface level is where the amplitude lives.
            @test maximum(abs.(dT[:, 1])) > maximum(abs.(dT[:, kDim]))

            # Mean removal: the domain-mean state is unperturbed at EVERY level, to the
            # round-off of the mean itself (the recovery is a floating-point inverse).
            @test maximum(abs(sum(view(dT, :, k)) / ncols) for k in 1:kDim) < 1.0e-11

            # Relative humidity is preserved at every point, warmed and cooled alike.
            @test maxRH < 1.0e-12

            # Constant pressure, and every non-thermodynamic slot seeded exactly as the
            # bubble seeds it: zero.
            @test all(state[:, vars["p"], 1] .== 0.0)
            @test all(state[:, vars["rho_r"], 1] .== 0.0)
            @test all(state[:, vars["rho_c"], 1] .== 0.0)   # rho_cbar == 0 on this base
            @test all(state[:, vars["n_r"], 1] .== 0.0)
            @test all(state[:, vars["rho_e"], 1] .== 0.0)
            for nm in Scythe.ice_var_names(opts)
                @test all(state[:, vars[nm], 1] .== 0.0)
            end
            # The thermodynamic slots did move.
            @test maximum(abs.(state[:, vars["E_t"], 1])) > 0.0
            @test maximum(abs.(state[:, vars["Q_ss"], 1])) > 0.0
        end
    end

end
