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
            @test maximum(abs.(ref2.sbar[:, 1] .- base.s)) < 1.0e-6
            @test maximum(abs.(ref2.xibar[:, 1] .- base.xi)) < 1.0e-9
            @test maximum(abs.(ref2.mubar[:, 1] .- base.mu)) < 1.0e-6

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
                        # Untouched outside, modulo the spectral refit of the
                        # exact reference (~1e-9)
                        outside_ok &= abs(patch.physical[i, 1, 1]) < 1.0e-6
                        outside_ok &= abs(patch.physical[i, 3, 1]) < 1.0e-6
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
end
