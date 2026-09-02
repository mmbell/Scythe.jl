using Test
using Scythe
using Springsteel
using SpecialFunctions: gamma

# Tests for the microphysics -> cloud-optics conversion (Stage S3a,
# src/radiation_cloud_optics.jl): CloudOpticsParams, cloud_optics_column!.
#
# Everything here is pure arithmetic on plain column arrays -- no RRTMGP, no
# ModelTile, no artifacts -- so these tests run in the base test suite with
# no network and no lookup tables, exactly like test_radiation.jl.

@testset "Cloud optics (microphysics -> RRTMGP inputs)" begin

    # A one-species-populated (nlay, 12) ice matrix helper: fills species `s`'s
    # four columns from densities (rho_i, n_i, a_i, c_i) [kg/m^3, #/m^3, m^3/m^3,
    # m^3/m^3] at layer `k`, zero elsewhere.
    function ice_matrix(nlay; k::Int = 1, s::Int = 1,
                        rho_i::Float64 = 0.0, n_i::Float64 = 0.0,
                        a_i::Float64 = 0.0, c_i::Float64 = 0.0)
        M = zeros(Float64, nlay, 12)
        c0 = (1, 5, 9)[s]
        M[k, c0] = rho_i
        M[k, c0 + 1] = n_i
        M[k, c0 + 2] = a_i
        M[k, c0 + 3] = c_i
        return M
    end

    # Build one ISHMAEL species' RAW density moments (rho_i, n_i, a_i, c_i) at
    # deltastr = 1 (spherical, cni = ani) and bulk density rhobar = 920 (ISHMAEL_RHOI)
    # by INVERTING the mass relation `_ice_effective`/`ishmael_var_check` use:
    #   q = n * rhobar * (4/3)*pi * ani^3 * Gamma(nu+3) / Gamma(nu)      (mass, delta=1)
    #   a = ani^2 * cni * n = ani^3 * n                                  (area-axis moment)
    #   c = cni^2 * ani * n = ani^3 * n                                  (length-axis moment)
    # then multiplies the mixing ratios back up by rho_d to get the model's own
    # DENSITY-unit prognostic slots (what cloud_optics_column! actually takes).
    # This exactly round-trips through `_ice_effective` back to `ani` (see the
    # docstring's algebra): ani_calc = ((a^2)/(c*n))^(1/3) = ani when a = c = ani^3*n.
    NU = Scythe.ISHMAEL_NU
    function ice_species_densities(ani::Float64, n_mix::Float64, rho_d::Float64;
                                   rhobar::Float64 = 920.0)
        gam_spherical = gamma(NU + 3.0)   # Gamma(nu+2+delta), delta=1
        gammnu = gamma(NU)
        q_mix = n_mix * rhobar * (4.0 / 3.0) * pi * ani^3 * gam_spherical / gammnu
        a_mix = ani^3 * n_mix
        c_mix = ani^3 * n_mix
        return (rho_i = q_mix * rho_d, n_i = n_mix * rho_d,
                a_i = a_mix * rho_d, c_i = c_mix * rho_d)
    end

    # ──────────────────────────────────────────────
    # 1. CloudOpticsParams defaults
    # ──────────────────────────────────────────────
    @testset "CloudOpticsParams defaults" begin
        P = Scythe.CloudOpticsParams(Dict{Symbol,Any}(), Dict{Symbol,Float64}())
        @test P.N_c == 100.0
        @test P.k_factor == 0.8
        @test P.q_min == 1.0e-6
        @test P.rain_in_cloud == false
        @test P.ice_on == false

        P2 = Scythe.CloudOpticsParams(
            Dict{Symbol,Any}(:radiation_rain_in_cloud => true, :ice_microphysics => :ishmael),
            Dict{Symbol,Float64}(:max_N_c => 200.0, :cloud_k_factor => 0.7,
                                 :radiation_q_min => 1.0e-5))
        @test P2.N_c == 200.0
        @test P2.k_factor == 0.7
        @test P2.q_min == 1.0e-5
        @test P2.rain_in_cloud == true
        @test P2.ice_on == true
    end

    # ──────────────────────────────────────────────
    # 2. Liquid effective radius: hand-computed against the documented formula
    # ──────────────────────────────────────────────
    @testset "liquid r_eff, hand-computed" begin
        N_c = 100.0    # #/cm^3
        q_c = 1.0e-3   # kg/kg
        rho_d = 1.0    # kg/m^3
        k_factor = 0.8

        # cloud_droplet_radius's own documented formula (src/microphysics.jl:502-527),
        # NOT a call to it -- an independent hand computation.
        rho_c_val = q_c * rho_d
        kg_drop = rho_c_val / (N_c * 1.0e6)
        r_vol_expected = 1.0e6 * (kg_drop * 3.0 / (4000.0 * pi))^(1.0 / 3.0)
        r_eff_expected = r_vol_expected / cbrt(k_factor)
        @test 2.5 < r_eff_expected < 21.5   # sanity: this test case must not clamp

        cld = Scythe.CloudOpticsColumn(1)
        P = Scythe.CloudOpticsParams(Dict{Symbol,Any}(), Dict{Symbol,Float64}(:max_N_c => N_c))
        n_liq, n_ice = Scythe.cloud_optics_column!(cld, P, [rho_d], [rho_c_val], [0.0],
                                                    nothing, [10.0])
        @test cld.re_liq[1] ≈ r_eff_expected rtol=1e-12
        @test n_liq == 0
        @test n_ice == 0
        @test cld.lwp[1] ≈ 1000.0 * rho_c_val * 10.0 rtol=1e-12
        @test cld.iwp[1] == 0.0
    end

    # ──────────────────────────────────────────────
    # 3. Liquid clamps at both ends
    # ──────────────────────────────────────────────
    @testset "liquid r_eff clamps" begin
        P = Scythe.CloudOpticsParams(Dict{Symbol,Any}(), Dict{Symbol,Float64}())

        # Tiny cloud water at huge droplet number -> tiny droplets -> clamp low.
        cld = Scythe.CloudOpticsColumn(1)
        n_liq, _ = Scythe.cloud_optics_column!(cld, P, [1.0], [1.0e-9], [0.0], nothing, [10.0])
        @test n_liq == 1
        @test cld.re_liq[1] == 2.5

        # Large cloud water, small droplet number -> big droplets -> clamp high.
        cld2 = Scythe.CloudOpticsColumn(1)
        n_liq2, _ = Scythe.cloud_optics_column!(cld2, P, [1.0], [5.0e-2], [0.0], nothing, [10.0])
        @test n_liq2 == 1
        @test cld2.re_liq[1] == 21.5

        # A column mixing both directions counts both.
        P3 = P
        cld3 = Scythe.CloudOpticsColumn(2)
        n_liq3, _ = Scythe.cloud_optics_column!(cld3, P3, [1.0, 1.0], [1.0e-9, 5.0e-2],
                                                 [0.0, 0.0], nothing, [10.0, 10.0])
        @test n_liq3 == 2
        @test cld3.re_liq == [2.5, 21.5]
    end

    # ──────────────────────────────────────────────
    # 4. Ice: single species, spherical (delta=1) round-trip, re_ice == 6*rni
    # ──────────────────────────────────────────────
    @testset "ice r_eff spherical round-trip (delta=1, nu=4 -> 6*rni)" begin
        rho_d = 1.2
        ani = 10.0e-6     # 10 micron a-axis scale -> r_eff = 6*10 = 60 micron, in-range
        n_mix = 1.0e6     # #/kg, arbitrary positive
        dens = ice_species_densities(ani, n_mix, rho_d)

        # What _ice_effective itself returns for these densities' mixing ratios --
        # the independent reference this test checks cloud_optics_column! against.
        eff = Scythe._ice_effective(dens.rho_i / rho_d, dens.n_i / rho_d,
                                    dens.a_i / rho_d, dens.c_i / rho_d, 1)
        @test eff.deltastr ≈ 1.0 atol=1e-10   # construction must land exactly on delta=1
        expected_re_ice = 6.0 * eff.rni * 1.0e6   # micron
        @test 5.0 < expected_re_ice < 90.0        # sanity: must not clamp

        M = ice_matrix(1; k = 1, s = 1, rho_i = dens.rho_i, n_i = dens.n_i,
                       a_i = dens.a_i, c_i = dens.c_i)
        cld = Scythe.CloudOpticsColumn(1)
        P = Scythe.CloudOpticsParams(Dict{Symbol,Any}(), Dict{Symbol,Float64}())
        n_liq, n_ice = Scythe.cloud_optics_column!(cld, P, [rho_d], [0.0], [0.0], M, [10.0])

        @test n_ice == 0
        @test cld.re_ice[1] ≈ expected_re_ice rtol=1e-9
        @test cld.iwp[1] ≈ 1000.0 * dens.rho_i * 10.0 rtol=1e-12
    end

    # ──────────────────────────────────────────────
    # 5. Ice: two-species harmonic combination against a hand computation
    # ──────────────────────────────────────────────
    @testset "ice two-species harmonic combination" begin
        rho_d = 1.0
        dens1 = ice_species_densities(5.0e-6, 5.0e5, rho_d)    # r1 = 6*5  = 30 micron
        dens2 = ice_species_densities(8.0e-6, 2.0e5, rho_d)    # r2 = 6*8  = 48 micron

        eff1 = Scythe._ice_effective(dens1.rho_i / rho_d, dens1.n_i / rho_d,
                                     dens1.a_i / rho_d, dens1.c_i / rho_d, 1)
        eff2 = Scythe._ice_effective(dens2.rho_i / rho_d, dens2.n_i / rho_d,
                                     dens2.a_i / rho_d, dens2.c_i / rho_d, 2)
        r1 = 6.0 * eff1.rni * 1.0e6
        r2 = 6.0 * eff2.rni * 1.0e6
        iwc1 = dens1.rho_i
        iwc2 = dens2.rho_i
        expected = (iwc1 + iwc2) / (iwc1 / r1 + iwc2 / r2)
        @test 5.0 < expected < 90.0

        M = zeros(Float64, 1, 12)
        M[1, 1:4] = [dens1.rho_i, dens1.n_i, dens1.a_i, dens1.c_i]
        M[1, 5:8] = [dens2.rho_i, dens2.n_i, dens2.a_i, dens2.c_i]

        cld = Scythe.CloudOpticsColumn(1)
        P = Scythe.CloudOpticsParams(Dict{Symbol,Any}(), Dict{Symbol,Float64}())
        n_liq, n_ice = Scythe.cloud_optics_column!(cld, P, [rho_d], [0.0], [0.0], M, [1.0])
        @test n_ice == 0
        @test cld.re_ice[1] ≈ expected rtol=1e-10
        # The mixed radius must lie strictly between the two species' own radii --
        # the defining property of a mass-preserving harmonic mean.
        @test min(r1, r2) <= cld.re_ice[1] <= max(r1, r2)
    end

    # ──────────────────────────────────────────────
    # 6. Ice clamps: high end reachable, low end is NOT (a D5 finding -- see below)
    # ──────────────────────────────────────────────
    @testset "ice r_eff clamps" begin
        rho_d = 1.0
        P = Scythe.CloudOpticsParams(Dict{Symbol,Any}(), Dict{Symbol,Float64}())

        # Large a-axis scale (near ISHMAEL's 1 mm axis cap) -> large r_eff -> clamp high.
        densB = ice_species_densities(9.0e-4, 1.0, rho_d)
        MB = zeros(Float64, 1, 12)
        MB[1, 1:4] = [densB.rho_i, densB.n_i, densB.a_i, densB.c_i]
        cldB = Scythe.CloudOpticsColumn(1)
        _, n_iceB = Scythe.cloud_optics_column!(cldB, P, [rho_d], [0.0], [0.0], MB, [1.0])
        @test n_iceB == 1
        @test cldB.re_ice[1] == 90.0

        # FINDING (reported to the orchestrator): the RAD_RE_ICE_MIN = 5 micron floor is
        # UNREACHABLE through any legitimate `_ice_effective` output. `ishmael_var_check`
        # floors `rni` at ISHMAEL_RMIN = 2 micron and clamps `deltastr` to [0.55, 1.3]; the
        # gamma-moment factor Gamma(nu+2+delta)/Gamma(nu+(4+2*delta)/3) is minimized at
        # delta=0.55 (factor ~4.342, nu=4), so the absolute floor of a single species'
        # r_eff,k is ~2*4.342 = 8.68 micron, and the harmonic combination across species is
        # bounded within [min_k r_eff,k, max_k r_eff,k] (see the docstring), so it can never
        # go lower either. This test exercises the closest approach found by direct search
        # (an extreme, physically degenerate population: tiny mass, huge number, strongly
        # oblate axes) and confirms it still clamps to NEITHER bound, staying comfortably
        # above 5 micron -- i.e. the low clamp's counter can be nonzero only if a future
        # change to ISHMAEL's own floors (RMIN, the deltastr range) or a coding error moved
        # rni or delta outside their current ranges; the clamp is retained as that defensive
        # bound, not because it is expected to fire under ISHMAEL's current physics.
        ani, cni, n_mix, qi_mix = 1.0e-5, 1.0e-7, 1.0e8, 1.0e-10
        a_mix = ani^2 * cni * n_mix
        c_mix = cni^2 * ani * n_mix
        MC = zeros(Float64, 1, 12)
        MC[1, 1:4] = [qi_mix * rho_d, n_mix * rho_d, a_mix * rho_d, c_mix * rho_d]
        cldC = Scythe.CloudOpticsColumn(1)
        _, n_iceC = Scythe.cloud_optics_column!(cldC, P, [rho_d], [0.0], [0.0], MC, [1.0])
        @test n_iceC == 0
        @test 5.0 < cldC.re_ice[1] < 15.0   # near the theoretical floor, but not clamped

        # Both the reachable high clamp and the unreachable-low case in one column.
        Mboth = zeros(Float64, 2, 12)
        Mboth[1, 1:4] = [densB.rho_i, densB.n_i, densB.a_i, densB.c_i]
        Mboth[2, 1:4] = MC[1, 1:4]
        cldBoth = Scythe.CloudOpticsColumn(2)
        _, n_iceBoth = Scythe.cloud_optics_column!(cldBoth, P, [rho_d, rho_d], [0.0, 0.0],
                                                    [0.0, 0.0], Mboth, [1.0, 1.0])
        @test n_iceBoth == 1
        @test cldBoth.re_ice[1] == 90.0
        @test cldBoth.re_ice[2] == cldC.re_ice[1]
    end

    # ──────────────────────────────────────────────
    # 7. Zero condensate: exact zero paths, cf = 0, benign radii
    # ──────────────────────────────────────────────
    @testset "zero condensate" begin
        P = Scythe.CloudOpticsParams(Dict{Symbol,Any}(), Dict{Symbol,Float64}())

        cld = Scythe.CloudOpticsColumn(3)
        n_liq, n_ice = Scythe.cloud_optics_column!(cld, P, [1.0, 1.1, 1.2],
                                                    [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                                                    nothing, [10.0, 10.0, 10.0])
        @test cld.lwp == [0.0, 0.0, 0.0]
        @test cld.iwp == [0.0, 0.0, 0.0]
        @test cld.cf == [0.0, 0.0, 0.0]
        @test cld.re_liq == fill(10.0, 3)
        @test cld.re_ice == fill(30.0, 3)
        @test n_liq == 0
        @test n_ice == 0

        # Same, with an explicit all-zero ice matrix rather than `nothing`.
        M = zeros(Float64, 3, 12)
        cld2 = Scythe.CloudOpticsColumn(3)
        Scythe.cloud_optics_column!(cld2, P, [1.0, 1.1, 1.2], [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0], M, [10.0, 10.0, 10.0])
        @test cld2.iwp == [0.0, 0.0, 0.0]
        @test cld2.re_ice == fill(30.0, 3)
    end

    # ──────────────────────────────────────────────
    # 8. Cloud fraction: strict threshold on both sides
    # ──────────────────────────────────────────────
    @testset "cf strict threshold" begin
        P = Scythe.CloudOpticsParams(Dict{Symbol,Any}(), Dict{Symbol,Float64}(:radiation_q_min => 1.0e-6))
        rho_d = 1.0

        # Exactly at q_min: cf must be 0 (strict >).
        cld = Scythe.CloudOpticsColumn(1)
        Scythe.cloud_optics_column!(cld, P, [rho_d], [1.0e-6], [0.0], nothing, [10.0])
        @test cld.cf[1] == 0.0

        # Just above q_min: cf must be 1.
        cld2 = Scythe.CloudOpticsColumn(1)
        Scythe.cloud_optics_column!(cld2, P, [rho_d], [1.0e-6 + 1.0e-12], [0.0], nothing, [10.0])
        @test cld2.cf[1] == 1.0

        # Just below: cf must be 0.
        cld3 = Scythe.CloudOpticsColumn(1)
        Scythe.cloud_optics_column!(cld3, P, [rho_d], [1.0e-6 - 1.0e-12], [0.0], nothing, [10.0])
        @test cld3.cf[1] == 0.0

        # Ice-only condensate crossing the threshold.
        M = ice_matrix(1; rho_i = 2.0e-6, n_i = 1.0e6, a_i = 1.0e-6, c_i = 1.0e-6)
        cld4 = Scythe.CloudOpticsColumn(1)
        Scythe.cloud_optics_column!(cld4, P, [rho_d], [0.0], [0.0], M, [10.0])
        @test cld4.cf[1] == 1.0
    end

    # ──────────────────────────────────────────────
    # 9. Rain excluded from lwp unless rain_in_cloud
    # ──────────────────────────────────────────────
    @testset "rain excluded from lwp by default" begin
        rho_d = 1.0
        rho_c = 1.0e-4
        rho_r = 5.0e-4
        dz = 10.0

        P_off = Scythe.CloudOpticsParams(Dict{Symbol,Any}(), Dict{Symbol,Float64}())
        cld_off = Scythe.CloudOpticsColumn(1)
        Scythe.cloud_optics_column!(cld_off, P_off, [rho_d], [rho_c], [rho_r], nothing, [dz])
        @test cld_off.lwp[1] ≈ 1000.0 * rho_c * dz rtol=1e-12

        P_on = Scythe.CloudOpticsParams(Dict{Symbol,Any}(:radiation_rain_in_cloud => true),
                                        Dict{Symbol,Float64}())
        cld_on = Scythe.CloudOpticsColumn(1)
        Scythe.cloud_optics_column!(cld_on, P_on, [rho_d], [rho_c], [rho_r], nothing, [dz])
        @test cld_on.lwp[1] ≈ 1000.0 * (rho_c + rho_r) * dz rtol=1e-12
        @test cld_on.lwp[1] > cld_off.lwp[1]
    end

    # ──────────────────────────────────────────────
    # 10. Negative condensate never produces a negative path
    # ──────────────────────────────────────────────
    @testset "negative condensate never produces negative paths" begin
        P = Scythe.CloudOpticsParams(Dict{Symbol,Any}(), Dict{Symbol,Float64}())

        # Negative rho_c alone -> lwp exactly 0, benign re_liq, no clamp.
        cld = Scythe.CloudOpticsColumn(1)
        n_liq, _ = Scythe.cloud_optics_column!(cld, P, [1.0], [-3.0], [0.0], nothing, [10.0])
        @test cld.lwp[1] == 0.0
        @test cld.re_liq[1] == 10.0
        @test n_liq == 0

        # Negative rain with rain_in_cloud can partially offset positive rho_c but
        # the SUM is floored, never a maxed-then-summed negative slipping through.
        P_on = Scythe.CloudOpticsParams(Dict{Symbol,Any}(:radiation_rain_in_cloud => true),
                                        Dict{Symbol,Float64}())
        cld2 = Scythe.CloudOpticsColumn(1)
        Scythe.cloud_optics_column!(cld2, P_on, [1.0], [1.0e-4], [-5.0e-4], nothing, [10.0])
        @test cld2.lwp[1] == 0.0   # 1e-4 - 5e-4 < 0

        # Ice: mixed-sign species sum BEFORE flooring (not sum of per-species maxes) --
        # species1 = +1.0e-3, species2 = -3.0e-4, species3 = 0 -> raw sum = 7.0e-4,
        # not 1.0e-3 (which is what summing individual maxes would give).
        M = zeros(Float64, 1, 12)
        M[1, 1] = 1.0e-3     # rho_i1, n_i1 = 0 so it's excluded from the re_ice weighting
        M[1, 5] = -3.0e-4    # rho_i2
        cld3 = Scythe.CloudOpticsColumn(1)
        Scythe.cloud_optics_column!(cld3, P, [1.0], [0.0], [0.0], M, [10.0])
        @test cld3.iwp[1] ≈ 1000.0 * 7.0e-4 * 10.0 rtol=1e-12
        @test cld3.iwp[1] > 0.0

        # A single species that is ALL negative -> iwp exactly 0, benign re_ice.
        M2 = zeros(Float64, 1, 12)
        M2[1, 1] = -2.0e-3
        cld4 = Scythe.CloudOpticsColumn(1)
        _, n_ice4 = Scythe.cloud_optics_column!(cld4, P, [1.0], [0.0], [0.0], M2, [10.0])
        @test cld4.iwp[1] == 0.0
        @test cld4.re_ice[1] == 30.0
        @test n_ice4 == 0
    end

    # ──────────────────────────────────────────────
    # 11. Ice species with mass but no carried number is excluded from the
    #     re_ice weighting (the model's own "population gate") but its mass
    #     still counts toward iwp and cf.
    # ──────────────────────────────────────────────
    @testset "mass without number excluded from re_ice weighting" begin
        rho_d = 1.0
        M = zeros(Float64, 1, 12)
        M[1, 1] = 5.0e-4   # rho_i1 > 0
        M[1, 2] = 0.0      # n_i1 == 0 -- no population
        P = Scythe.CloudOpticsParams(Dict{Symbol,Any}(), Dict{Symbol,Float64}())
        cld = Scythe.CloudOpticsColumn(1)
        _, n_ice = Scythe.cloud_optics_column!(cld, P, [rho_d], [0.0], [0.0], M, [10.0])
        @test cld.iwp[1] ≈ 1000.0 * 5.0e-4 * 10.0 rtol=1e-12   # mass still in the water path
        @test cld.re_ice[1] == 30.0   # but no valid PSD to size it with -> benign default
        @test cld.cf[1] == 1.0        # and it still counts as cloud
        @test n_ice == 0
    end

    # ──────────────────────────────────────────────
    # 12. `ice === nothing` path: never crashes, ice outputs are the clear-sky default
    # ──────────────────────────────────────────────
    @testset "ice === nothing" begin
        P = Scythe.CloudOpticsParams(Dict{Symbol,Any}(), Dict{Symbol,Float64}())
        rho_d = fill(1.1, 5)
        rho_c = [0.0, 1.0e-4, 0.0, 2.0e-4, 0.0]
        rho_r = zeros(5)
        dz = fill(200.0, 5)

        cld = Scythe.CloudOpticsColumn(5)
        n_liq, n_ice = Scythe.cloud_optics_column!(cld, P, rho_d, rho_c, rho_r, nothing, dz)
        @test cld.iwp == zeros(5)
        @test all(cld.re_ice .== 30.0)
        @test n_ice == 0
        # Liquid path is unaffected by ice being absent.
        @test cld.lwp[2] ≈ 1000.0 * rho_c[2] * dz[2] rtol=1e-12
        @test cld.cf[2] == 1.0
        @test cld.cf[1] == 0.0
    end
end
