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
        column = deepcopy(patch.kbasis.data[1])
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
                    q_v, _, _, _ = Scythe.thermodynamic_tuple(s_tot, xi_tot, ref.mubar[k, 1])
                    theta = Scythe.potential_temperature(s_tot, xi_tot, q_v)
                    max_err = max(max_err, abs(theta - (prof.theta[k] + dtheta)))
                    i += 1
                end
            end
            @test max_err < 1.0e-6
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
