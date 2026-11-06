using Test
using Scythe
using Springsteel

# Tests for the shared bulk air-sea surface layer (src/mc_surface_layer.jl): the
# `:komori` default path (which must stay BITWISE what mc_louis_bl! computed inline
# before stage S1b), the GFDL/HWRF v7 roughness fits against their Fortran original,
# the Charnock/Zeng roughness, and the Monin-Obukhov stability layer.
#
# The physics of the DEFAULT path is tested by test_louis_bl.jl and
# test_surface_fluxes.jl, which are untouched by S1b; this file tests that the code
# move preserved it exactly, and that the new options behave.

@testset "Surface layer (bulk air-sea exchange)" begin

    import Springsteel.Thermodynamics: rho_v_sat, Rd, Rv, Cpd, gravity

    include(joinpath(@__DIR__, "reference", "gfdl_sfc_refs.jl"))

    # ──────────────────────────────────────────────────────────────
    # 1. The default path is BITWISE the old inline formulas
    # ──────────────────────────────────────────────────────────────
    #
    # The oracle below is the pre-S1b body of `mc_louis_bl!` (src/mc_boundary_layer.jl
    # lines 182-205 and 213-214 at commit 639c9fb), copied verbatim -- same expressions,
    # same association, same order. `===` and not `≈`: a code move that changes the last
    # bit changes every benchmark reference this model has.
    """The pre-S1b inline surface exchange of `mc_louis_bl!`, verbatim."""
    function louis_surface_oracle(u_raw, v_raw, Tk1, rho_d1, rho_t1, rho_v1, p1_hPa,
                                  SST, Cd_param, Ck, U_min, sfc_fac, surface_fluxes)
        u1 = u_raw * sfc_fac
        v1 = v_raw * sfc_fac
        U1 = max(sqrt((u1 * u1) + (v1 * v1)), U_min)
        Cd = Cd_param < 0.0 ? Scythe.komori_cd(U1) : Cd_param
        drag_coeff = Cd * U1
        F_sh = 0.0
        F_q = 0.0
        if surface_fluxes
            F_sh = rho_d1 * Cpd * Ck * U1 * (SST - Tk1)
            F_q = Ck * U1 * (rho_v_sat(SST, p1_hPa) - rho_v1)
        end
        tau_u = rho_t1 * drag_coeff * u1
        tau_v = rho_t1 * drag_coeff * v1
        # `ust` is the benchmark diagnostics' expression (owb_surface_diagnostics)
        return (ust = sqrt(Cd) * U1, tau_u = tau_u, tau_v = tau_v, F_sh = F_sh, F_q = F_q)
    end

    @testset "default (:komori, no stability) is bitwise the old inline code" begin
        rng_state = 20260905
        for trial in 1:50
            # A deterministic pseudo-random spread of plausible lowest-level states,
            # spanning the three Komori regimes and both flux signs.
            rng_state = (1103515245 * rng_state + 12345) % 2147483648
            r1 = rng_state / 2147483648
            rng_state = (1103515245 * rng_state + 12345) % 2147483648
            r2 = rng_state / 2147483648
            rng_state = (1103515245 * rng_state + 12345) % 2147483648
            r3 = rng_state / 2147483648
            rng_state = (1103515245 * rng_state + 12345) % 2147483648
            r4 = rng_state / 2147483648

            u_raw = -60.0 + (120.0 * r1)
            v_raw = -40.0 + (80.0 * r2)
            Tk1 = 285.0 + (20.0 * r3)
            SST = 288.0 + (18.0 * r4)
            p1_Pa = 95000.0 + (10000.0 * r1)
            rho_d1 = 1.05 + (0.2 * r2)
            rho_v1 = 0.002 + (0.022 * r3)
            rho_t1 = rho_d1 + rho_v1
            z1 = 56.350832689629151
            Cd_param = trial % 5 == 0 ? 1.5e-3 : -1.0     # exercise the constant-Cd arm too
            Ck = 1.0e-3
            U_min = trial % 3 == 0 ? 0.0 : 2.0
            sfc_fac = trial % 7 == 0 ? 0.85 : 1.0
            sflux = trial % 11 != 0                        # and the fluxes-off arm

            want = louis_surface_oracle(u_raw, v_raw, Tk1, rho_d1, rho_t1, rho_v1,
                                        p1_Pa / 100.0, SST, Cd_param, Ck, U_min,
                                        sfc_fac, sflux)
            params = Scythe.SurfaceLayerParams(Cd_param, Ck, U_min, sfc_fac, SST,
                                               :komori, false, sflux, 3)
            got = Scythe.surface_exchange(u_raw, v_raw, Tk1, rho_d1, rho_t1, rho_v1,
                                          p1_Pa, z1, SST, params)
            @test got.ust === want.ust
            @test got.tau_u === want.tau_u
            @test got.tau_v === want.tau_v
            @test got.F_sh === want.F_sh
            @test got.F_q === want.F_q
        end
    end

    @testset "surface_layer_params reads the driver's params and options" begin
        pp = Dict{Symbol,Float64}(:Cd => -1.0, :Ck => 1.2e-3, :U_min => 2.0,
                                  :sfc_wind_factor => 0.9, :SST => 302.65)
        p = Scythe.surface_layer_params(pp, Dict{Symbol,Any}(); surface_fluxes = true)
        @test p.z0_mode === :komori && p.stability == false && p.n_stab_iter == 3
        @test (p.Cd_param, p.Ck, p.U_min, p.sfc_fac, p.SST) ==
              (-1.0, 1.2e-3, 2.0, 0.9, 302.65)
        @test p.surface_fluxes
        p2 = Scythe.surface_layer_params(pp, Dict{Symbol,Any}(:sfc_z0 => :gfdl_v7,
                                                             :sfc_stability => true))
        @test p2.z0_mode === :gfdl_v7 && p2.stability && !p2.surface_fluxes
        # Immutable and concretely typed, so it crosses the @noinline boundary of
        # mc_louis_bl! without boxing (the driver gate in test_allocations.jl is what
        # actually proves that).
        @test isimmutable(p)
        @test all(isconcretetype, fieldtypes(Scythe.SurfaceLayerParams))
        # Unknown values die at SETUP, listing the valid ones
        err = try
            Scythe.surface_layer_params(pp, Dict{Symbol,Any}(:sfc_z0 => :coare))
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin(":komori", err.msg) && occursin(":gfdl_v7", err.msg) &&
              occursin(":charnock", err.msg)
    end

    # ──────────────────────────────────────────────────────────────
    # 2. The GFDL/HWRF v7 fits against the Fortran original
    # ──────────────────────────────────────────────────────────────
    @testset "znot_m_v7 / znot_t_v7 match the ccpp-physics Fortran" begin
        # tools/mynn_fortran_driver/sfc_ref_driver.f90 over the verbatim
        # module_sf_exchcoef.f90, gfortran -fdefault-real-8 -ffp-contract=off -O0.
        @test length(GFDL_SFC_Z0_REFS) == 17
        for (u, z0m, z0t) in GFDL_SFC_Z0_REFS
            @test Scythe.znot_m_v7(u) ≈ z0m rtol=1e-12
            @test Scythe.znot_t_v7(u) ≈ z0t rtol=1e-12
        end
        # The two constant branches, by value
        @test Scythe.znot_t_v7(3.0) == 1.1e-4
        @test Scythe.znot_m_v7(90.0) == 3.371427455376717e-04
        @test Scythe.znot_t_v7(90.0) == 6.840803042788488e-05
        # z0m peaks in the hurricane range and falls off again (the v7 high-wind fit)
        @test Scythe.znot_m_v7(30.0) > Scythe.znot_m_v7(15.0)
        @test Scythe.znot_m_v7(30.0) > Scythe.znot_m_v7(50.0)
    end

    @testset "Charnock + Zeng roughness" begin
        z1 = 56.350832689629151
        # z0m grows with wind and is capped at the UFS z0s_max
        z0m_l, z0t_l, _ = Scythe.sfc_roughness(:charnock, 5.0, z1, -1.0, 1.0e-3)
        z0m_h, z0t_h, _ = Scythe.sfc_roughness(:charnock, 50.0, z1, -1.0, 1.0e-3)
        @test 0.0 < z0m_l < z0m_h <= Scythe.SFC_Z0S_MAX
        @test Scythe.sfc_roughness(:charnock, 200.0, z1, -1.0, 1.0e-3)[1] ==
              Scythe.SFC_Z0S_MAX
        # Thermal roughness shrinks relative to z0m as the roughness Reynolds number grows
        @test z0t_h / z0m_h < z0t_l / z0m_l
        @test z0t_h > 0.0
    end

    @testset ":komori roughness inversion reproduces its own Cd and Ck" begin
        # With stability OFF the general path must reproduce the Komori Cd and the
        # constant Ck it was inverted from, to round-off.
        z1 = 56.350832689629151
        for U in (3.0, 12.0, 25.0, 45.0)
            z0m, z0t, _ = Scythe.sfc_roughness(:komori, U, z1, -1.0, 1.0e-3)
            k2 = Scythe.SFC_KARMAN^2
            @test k2 / log(z1 / z0m)^2 ≈ Scythe.komori_cd(U) rtol=1e-12
            @test k2 / (log(z1 / z0m) * log(z1 / z0t)) ≈ 1.0e-3 rtol=1e-12
        end
    end

    # ──────────────────────────────────────────────────────────────
    # 3. Monin-Obukhov stability layer
    # ──────────────────────────────────────────────────────────────
    @testset "psi_m / psi_h" begin
        @test Scythe.psi_m_sfc(0.0) == 0.0
        @test Scythe.psi_h_sfc(0.0) == 0.0
        # Stable: negative (adds to the log-law resistance) and monotone down
        @test Scythe.psi_m_sfc(0.5) < 0.0
        @test Scythe.psi_h_sfc(0.5) < 0.0
        @test Scythe.psi_m_sfc(2.0) < Scythe.psi_m_sfc(0.5)
        # Cheng & Brutsaert (2005) closed form, by construction
        @test Scythe.psi_m_sfc(1.0) ≈ -6.1 * log(1.0 + 2.0^(1 / 2.5))
        @test Scythe.psi_h_sfc(1.0) ≈ -5.3 * log(1.0 + 2.0^(1 / 1.1))
        # Unstable: positive (reduces the resistance), Businger-Dyer closed form
        @test Scythe.psi_m_sfc(-0.5) > 0.0
        @test Scythe.psi_h_sfc(-0.5) > 0.0
        x = (1.0 - (16.0 * -0.5))^0.25
        @test Scythe.psi_m_sfc(-0.5) ≈ 2 * log((1 + x) / 2) + log((1 + x^2) / 2) -
                                       2 * atan(x) + pi / 2
        @test Scythe.psi_h_sfc(-0.5) ≈ 2 * log((1 + x^2) / 2)
        # phi = 1 - zeta dpsi/dzeta must reproduce MYNN's phim/phih (module_bl_mynn.F90
        # :7528-7626) on the stable branch, where the two are the exact integral pair.
        for zeta in (0.1, 0.5, 2.0, 5.0)
            h = 1.0e-6
            dpsi_m = (Scythe.psi_m_sfc(zeta + h) - Scythe.psi_m_sfc(zeta - h)) / (2h)
            b = 2.5
            d0 = 1 + zeta^b
            d1 = zeta + d0^(1 / b)
            d11 = 1 + d0^((1 / b) - 1) * zeta^(b - 1)
            @test 1.0 - zeta * dpsi_m ≈ 1 - zeta * ((-6.1 / d1) * d11) rtol=1e-6
        end
    end

    """Lowest-level state from a tools/mynn_fortran_driver column (line 2 header,
    line 3 = first mish level): `(z1, u, T, p, rho_t, rho_v, rho_d, SST)`."""
    function driver_column(name)
        ls = readlines(joinpath(@__DIR__, "..", "tools", "mynn_fortran_driver",
                                "columns", name))
        hdr = parse.(Float64, split(ls[2]))
        l1 = parse.(Float64, split(ls[3]))
        z1, u, Tk, exner, p, rho, sqv = l1[1], l1[3], l1[6], l1[8], l1[9], l1[10], l1[11]
        rho_v = sqv * rho
        # `ts` is T_sfc/exner(1) (the MYNN wrapper convention, see the driver README)
        return (z1 = z1, u = u, T = Tk, p = p, rho_t = rho, rho_v = rho_v,
                rho_d = rho - rho_v, SST = hdr[2] * exner)
    end
    case2 = driver_column("case2_o01_sea.txt")       # O01/OWB sea, U1 = 3.6 m/s
    case5 = driver_column("case5_highwind.txt")      # hurricane-force, U1 = 9.3 m/s

    exchange(c, mode, stab, sflux, n; SST = c.SST) =
        Scythe.surface_exchange(c.u, 0.0, c.T, c.rho_d, c.rho_t, c.rho_v, c.p, c.z1,
                                SST,
                                Scythe.SurfaceLayerParams(-1.0, 1.0e-3, 2.0, 1.0, SST,
                                                          mode, stab, sflux, n))

    @testset "neutral limit: no buoyancy flux leaves the neutral coefficients" begin
        for c in (case2, case5), mode in (:komori, :gfdl_v7, :charnock)
            stable = exchange(c, mode, true, false, 3)      # fluxes off => zero buoyancy
            @test stable.inv_L == 0.0
            @test stable.w_star == 0.0
            # The NEUTRAL reference is the log law on the mode's own roughness lengths.
            # (:komori without stability takes the frozen legacy branch, which never forms
            # a roughness length -- it reports z0m = z0t = 0.0 rather than a fabricated
            # one -- so the reference is built here rather than read off that call.)
            U1 = max(abs(c.u), 2.0)
            z0m, z0t, _ = Scythe.sfc_roughness(mode, U1, c.z1, -1.0, 1.0e-3)
            k2 = Scythe.SFC_KARMAN^2
            @test stable.z0m == z0m
            @test stable.z0t == z0t
            @test stable.Cd == k2 / log(c.z1 / z0m)^2
            @test stable.Ch == k2 / (log(c.z1 / z0m) * log(c.z1 / z0t))
            # and with stability OFF the same neutral values come out
            neutral = exchange(c, mode, false, false, 3)
            if mode !== :komori
                @test neutral.Cd == stable.Cd
                @test neutral.Ch == stable.Ch
            else
                # the frozen path IS the Komori fit the inversion was built from
                @test neutral.Cd ≈ stable.Cd rtol=1e-12
                @test neutral.Ch ≈ stable.Ch rtol=1e-12
            end
        end
    end

    @testset "stable reduces, unstable increases the exchange coefficients" begin
        for c in (case2, case5), mode in (:komori, :gfdl_v7, :charnock)
            neutral = exchange(c, mode, false, true, 3)
            # A sea COLDER than the air: stable surface layer
            cold = exchange(c, mode, true, true, 3; SST = c.T - 4.0)
            @test cold.inv_L > 0.0
            @test cold.Cd < neutral.Cd
            @test cold.Ch < neutral.Ch
            @test cold.F_sh < 0.0
            @test cold.w_star == 0.0            # no gustiness under a downward flux
            # A sea WARMER than the air: unstable, and gusty
            warm = exchange(c, mode, true, true, 3; SST = c.T + 4.0)
            @test warm.inv_L < 0.0
            @test warm.Cd > neutral.Cd
            @test warm.Ch > neutral.Ch
            @test warm.F_sh > 0.0
            @test warm.w_star > 0.0
            @test warm.ust > sqrt(warm.Cd) * warm.U1   # gustiness raised the exchange wind
        end
    end

    @testset "the z/L fixed point converges" begin
        for c in (case2, case5), mode in (:komori, :gfdl_v7, :charnock),
            SST in (c.SST, c.T + 4.0, c.T - 2.0)

            # The fixed point is a CONTRACTION, not a truncation: successive iterates
            # shrink monotonically (rate ~0.05 unstable, ~0.23 stable), so by iterate 15
            # the step is below 1e-8 relative on every one of these columns.
            a = exchange(c, mode, true, true, 15; SST = SST)
            b = exchange(c, mode, true, true, 16; SST = SST)
            @test abs(b.inv_L - a.inv_L) <= 1e-8 * abs(b.inv_L)
            @test abs(b.Cd - a.Cd) <= 1e-8 * b.Cd
            @test abs(b.Ch - a.Ch) <= 1e-8 * b.Ch
            # And the production default of 3 iterations is already within a few percent
            # of that fixed point (worst case here 1.5 %, in the strongly stable limit
            # where Cd is an order of magnitude below neutral anyway) -- the surface layer
            # is not the term that needs another digit.
            three = exchange(c, mode, true, true, 3; SST = SST)
            @test abs(three.Cd - b.Cd) <= 5e-2 * b.Cd
            @test abs(three.Ch - b.Ch) <= 5e-2 * b.Ch
        end
    end

    @testset "gfdl_v7 U10 is the neutral log profile of the exchange wind" begin
        c = case5
        r = exchange(c, :gfdl_v7, false, true, 3)
        # The returned U10 is the one the fits were EVALUATED at, so these two are exact
        @test r.z0m == Scythe.znot_m_v7(r.U10)
        @test r.z0t == Scythe.znot_t_v7(r.U10)
        # and it agrees with the log profile of the returned z0m to the residual of the
        # three-step z0m <-> U10 iteration (U10 came from the previous z0m)
        @test r.U10 ≈ r.U1 * log(10.0 / r.z0m) / log(c.z1 / r.z0m) rtol=1e-4
        # a hurricane column clamps into the fitted range, never past it
        gale = Scythe.surface_exchange(120.0, 0.0, c.T, c.rho_d, c.rho_t, c.rho_v, c.p,
                                       c.z1, c.SST,
                                       Scythe.SurfaceLayerParams(-1.0, 1.0e-3, 2.0, 1.0,
                                                                 c.SST, :gfdl_v7, false,
                                                                 true, 3))
        @test gale.z0m == Scythe.znot_m_v7(85.0)
    end

    # ──────────────────────────────────────────────────────────────
    # 4. Allocation gate
    # ──────────────────────────────────────────────────────────────
    @testset "surface_exchange allocates nothing" begin
        c = case2
        for mode in (:komori, :gfdl_v7, :charnock), stab in (false, true)
            p = Scythe.SurfaceLayerParams(-1.0, 1.0e-3, 2.0, 1.0, c.SST, mode, stab,
                                          true, 3)
            Scythe.surface_exchange(c.u, 0.0, c.T, c.rho_d, c.rho_t, c.rho_v, c.p,
                                    c.z1, c.SST, p)      # compile
            @test (@allocations Scythe.surface_exchange(c.u, 0.0, c.T, c.rho_d, c.rho_t,
                                                        c.rho_v, c.p, c.z1, c.SST,
                                                        p)) == 0
        end
    end
end
