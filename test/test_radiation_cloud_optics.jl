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
        # A high droplet number is what puts a CLOUDY layer below the 2.5 um table floor:
        # from S4 the clamp counters are gated on `cf == 1`, so the low-end case has to
        # carry real cloud (q_c > q_min) as well as tiny droplets. 5000 /cm^3 with
        # 1e-5 kg/m^3 of liquid gives r_eff ~0.84 um at q_c = 1e-5 > q_min = 1e-6.
        P_hi = Scythe.CloudOpticsParams(Dict{Symbol,Any}(),
                                        Dict{Symbol,Float64}(:max_N_c => 5000.0))

        cld = Scythe.CloudOpticsColumn(1)
        n_liq, _ = Scythe.cloud_optics_column!(cld, P_hi, [1.0], [1.0e-5], [0.0],
                                                nothing, [10.0])
        @test cld.cf[1] == 1.0
        @test n_liq == 1
        @test cld.re_liq[1] == 2.5

        # Large cloud water, small droplet number -> big droplets -> clamp high.
        cld2 = Scythe.CloudOpticsColumn(1)
        n_liq2, _ = Scythe.cloud_optics_column!(cld2, P, [1.0], [5.0e-2], [0.0], nothing, [10.0])
        @test n_liq2 == 1
        @test cld2.re_liq[1] == 21.5

        # A column mixing both directions counts both, at the DEFAULT droplet number:
        # 3e-6 kg/m^3 is inside the `q_min` window (cloudy, but below the 2.5 um floor --
        # testset 14) and 5e-2 kg/m^3 is over the 21.5 um ceiling.
        cld3 = Scythe.CloudOpticsColumn(2)
        n_liq3, _ = Scythe.cloud_optics_column!(cld3, P, [1.0, 1.0], [3.0e-6, 5.0e-2],
                                                 [0.0, 0.0], nothing, [10.0, 10.0])
        @test cld3.cf == [1.0, 1.0]
        @test n_liq3 == 2
        @test cld3.re_liq == [2.5, 21.5]
    end

    # ──────────────────────────────────────────────
    # 4. Ice: single species, spherical (delta=1) round-trip.
    #    SOLID ice (rhobar = RHOI = 920) -> re_ice == 6*rni exactly; a low-density
    #    aggregate (rhobar = 50) -> the same 6*rni scaled by rhobar/RHOI (S4, A1).
    # ──────────────────────────────────────────────
    @testset "ice r_eff spherical round-trip (delta=1, nu=4 -> 6*rni at rhobar=RHOI)" begin
        rho_d = 1.2
        ani = 10.0e-6     # 10 micron a-axis scale -> r_eff = 6*10 = 60 micron, in-range
        n_mix = 1.0e6     # #/kg, arbitrary positive
        dens = ice_species_densities(ani, n_mix, rho_d)

        # What _ice_effective itself returns for these densities' mixing ratios --
        # the independent reference this test checks cloud_optics_column! against.
        eff = Scythe._ice_effective(dens.rho_i / rho_d, dens.n_i / rho_d,
                                    dens.a_i / rho_d, dens.c_i / rho_d, 1)
        @test eff.deltastr ≈ 1.0 atol=1e-10   # construction must land exactly on delta=1
        # Solid ice: the bulk-density factor rhobar/RHOI is EXACTLY 1, so this case still
        # pins the pure gamma-moment factor Gamma(nu+2+delta)/Gamma(nu+(4+2delta)/3) = 6.
        @test eff.rhobar == Scythe.ISHMAEL_RHOI
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

    @testset "ice r_eff bulk-density correction (rhobar = 50 -> 6*rni*50/920)" begin
        # Same spherical construction, but a LOW-density particle: the same axis scale
        # holds 50/920 of the solid-ice mass, so Fu (1996)'s generalized effective size --
        # built on the SOLID-ICE volume m/rho_ice over the (unchanged) projected area --
        # is smaller by exactly that ratio. `rni` is the spheroid radius at the PARTICLE
        # density rhobar, so the correction is the linear factor rhobar/RHOI, not its
        # cube root: the volume ratio is what changes, the area is not.
        rho_d = 1.2
        ani = 1.5e-4      # 150 micron a-axis scale: 6*rni = 900 um uncorrected,
        n_mix = 1.0e6     # 48.9 um once corrected -- i.e. inside [5, 90] ONLY when the
                          # correction is applied, so this test cannot pass by accident.
        dens = ice_species_densities(ani, n_mix, rho_d; rhobar = 50.0)
        eff = Scythe._ice_effective(dens.rho_i / rho_d, dens.n_i / rho_d,
                                    dens.a_i / rho_d, dens.c_i / rho_d, 1)
        @test eff.deltastr ≈ 1.0 atol=1e-10
        @test eff.rhobar ≈ 50.0 rtol=1e-8      # var_check re-derives it from (q, n, ani)

        uncorrected = 6.0 * eff.rni * 1.0e6
        expected = uncorrected * eff.rhobar / Scythe.ISHMAEL_RHOI
        @test uncorrected > 90.0               # the OLD formula would have clamped here
        @test 5.0 < expected < 90.0

        M = ice_matrix(1; k = 1, s = 3, rho_i = dens.rho_i, n_i = dens.n_i,
                       a_i = dens.a_i, c_i = dens.c_i)
        cld = Scythe.CloudOpticsColumn(1)
        P = Scythe.CloudOpticsParams(Dict{Symbol,Any}(), Dict{Symbol,Float64}())
        _, n_ice = Scythe.cloud_optics_column!(cld, P, [rho_d], [0.0], [0.0], M, [10.0])
        @test n_ice == 0
        @test cld.re_ice[1] ≈ expected rtol=1e-9
        # And the ratio to the solid-ice answer is the bulk-density ratio itself.
        @test cld.re_ice[1] / uncorrected ≈ 50.0 / 920.0 rtol=1e-7
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

        # FINDING, REVISED IN S4. Before the bulk-density correction the RAD_RE_ICE_MIN =
        # 5 micron floor was UNREACHABLE: `ishmael_var_check` floors `rni` at
        # ISHMAEL_RMIN = 2 micron and clamps `deltastr` to [0.55, 1.3], the gamma-moment
        # factor Gamma(nu+2+delta)/Gamma(nu+(4+2*delta)/3) is minimized at delta = 0.55
        # (~4.342 at nu = 4), so a single species' r_eff,k could not fall below
        # 2*4.342 = 8.68 micron, and the harmonic combination is bounded within
        # [min_k r_eff,k, max_k r_eff,k]. The rhobar/RHOI factor moves that floor DOWN by
        # up to 50/920 = 0.0543, to ~0.47 micron, so the low clamp is now genuinely
        # reachable -- by exactly the population it should be reachable by: a
        # near-massless, hugely numerous, strongly oblate low-density crystal, whose
        # solid-ice volume per unit projected area really is tiny. The same degenerate
        # population used before now clamps LOW, and this test asserts that (it is the
        # regression that would fail if the correction were dropped again).
        ani, cni, n_mix, qi_mix = 1.0e-5, 1.0e-7, 1.0e8, 1.0e-10
        a_mix = ani^2 * cni * n_mix
        c_mix = cni^2 * ani * n_mix
        MC = zeros(Float64, 1, 12)
        MC[1, 1:4] = [qi_mix * rho_d, n_mix * rho_d, a_mix * rho_d, c_mix * rho_d]
        # The crystal's own mass (1e-10 kg/kg) is far below `q_min`, and from S4 clamps
        # are counted only in CLOUDY layers -- so the layer is made cloudy by liquid
        # water alongside it, which is exactly the situation an anvil edge is in.
        cldC = Scythe.CloudOpticsColumn(1)
        _, n_iceC = Scythe.cloud_optics_column!(cldC, P, [rho_d], [1.0e-3], [0.0], MC, [1.0])
        @test cldC.cf[1] == 1.0
        @test n_iceC == 1
        @test cldC.re_ice[1] == 5.0
        # The uncorrected value is what the pre-S4 code returned, and it did NOT clamp:
        # the low clamp exists because of the density factor, not in spite of it.
        effC = Scythe._ice_effective(qi_mix, n_mix, a_mix, c_mix, 1)
        uncorrC = effC.rni * 1.0e6 *
                  gamma(NU + 2.0 + effC.deltastr) / gamma(NU + (4.0 + 2.0 * effC.deltastr) / 3.0)
        @test 5.0 < uncorrC < 15.0
        @test effC.rhobar == 50.0

        # Both clamps, one column: the high end from solid ice, the low end from the
        # low-density degenerate population.
        Mboth = zeros(Float64, 2, 12)
        Mboth[1, 1:4] = [densB.rho_i, densB.n_i, densB.a_i, densB.c_i]
        Mboth[2, 1:4] = MC[1, 1:4]
        cldBoth = Scythe.CloudOpticsColumn(2)
        _, n_iceBoth = Scythe.cloud_optics_column!(cldBoth, P, [rho_d, rho_d],
                                                    [0.0, 1.0e-3], [0.0, 0.0],
                                                    Mboth, [1.0, 1.0])
        @test n_iceBoth == 2
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

    # ──────────────────────────────────────────────
    # 13. S4/A2: clamps are counted ONLY in cloudy layers (cf == 1)
    # ──────────────────────────────────────────────
    #
    # RRTMGP masks the cloud optics of every layer with cf == 0, so an effective radius
    # written there is never read: counting its clamp saturates the counter with layers
    # that are radiatively inert. On the S3b warm arm that was about a third of all
    # gridpoints. `cf` is therefore computed FIRST and the counters are gated on it; the
    # radii themselves are still written (RRTMGP wants a finite in-table value in every
    # cell), just not counted.
    @testset "clamps counted only where cf == 1" begin
        P = Scythe.CloudOpticsParams(Dict{Symbol,Any}(), Dict{Symbol,Float64}())
        rho_d = 1.0

        # Trace liquid, BELOW the cf threshold (q_min = 1e-6 kg/kg at rho_d = 1 means
        # rho_c = 1e-6 kg/m^3): the droplet radius is far below 2.5 um and IS clamped in
        # the array, but the layer is not cloud, so nothing is counted.
        cld = Scythe.CloudOpticsColumn(1)
        n_liq, n_ice = Scythe.cloud_optics_column!(cld, P, [rho_d], [5.0e-7], [0.0],
                                                    nothing, [10.0])
        @test cld.cf[1] == 0.0
        @test cld.lwp[1] > 0.0          # the path is still built: mass is mass
        @test cld.re_liq[1] == 2.5      # still written in range for the solver
        @test n_liq == 0                # ... but NOT counted
        @test n_ice == 0

        # Same trace ice: a low-density degenerate crystal that would clamp low, at a
        # mass below the cloud-fraction threshold.
        M = zeros(Float64, 1, 12)
        ani, cni, n_mix, qi_mix = 1.0e-5, 1.0e-7, 1.0e8, 1.0e-10
        M[1, 1:4] = [qi_mix * rho_d, n_mix * rho_d,
                     ani^2 * cni * n_mix * rho_d, cni^2 * ani * n_mix * rho_d]
        @test qi_mix < 1.0e-6           # below q_min: this layer is not cloud
        cld2 = Scythe.CloudOpticsColumn(1)
        _, n_ice2 = Scythe.cloud_optics_column!(cld2, P, [rho_d], [0.0], [0.0], M, [1.0])
        @test cld2.cf[1] == 0.0
        @test cld2.re_ice[1] == 5.0     # clamped in the array
        @test n_ice2 == 0               # not counted

        # The SAME crystal in a layer made cloudy by liquid water alongside it IS counted:
        # the gate is the layer's cloud fraction, not the species' own mass.
        cld3 = Scythe.CloudOpticsColumn(1)
        _, n_ice3 = Scythe.cloud_optics_column!(cld3, P, [rho_d], [1.0e-3], [0.0], M, [1.0])
        @test cld3.cf[1] == 1.0
        @test n_ice3 == 1
    end

    # ──────────────────────────────────────────────
    # 14. S4/A3: q_min admits cloudy layers whose uncorrected r_eff < 2.5 um
    # ──────────────────────────────────────────────
    #
    # At N_c = 100 /cm^3 the monodisperse volume radius reaches RAD_RE_LIQ_MIN / cbrt(k)
    # = 2.5*cbrt(0.8) = 2.321 um at rho_c = 5.24e-6 kg/m^3, while the cloud-fraction
    # threshold q_min = 1e-6 kg/kg admits cloud from rho_c = 1.2e-6 kg/m^3 (rho_d = 1.2).
    # The window in between is CLOUDY BY DEFINITION and clamped: that is intended, and
    # the count it produces is a real, reported number rather than a defect.
    @testset "q_min admits clamped sub-2.5 um cloudy layers (intended)" begin
        rho_d = 1.2
        P = Scythe.CloudOpticsParams(Dict{Symbol,Any}(), Dict{Symbol,Float64}())

        # The exact rho_c at which the UNCLAMPED effective radius equals 2.5 um, from
        # cloud_droplet_radius's own formula inverted by hand.
        r_vol_at_min = 2.5 * cbrt(0.8) * 1.0e-6              # m
        kg_drop = (4.0 / 3.0) * pi * r_vol_at_min^3 * 1000.0  # kg, rho_water = 1000
        rho_c_at_min = kg_drop * (100.0 * 1.0e6)              # N_c = 100 /cm^3 -> /m^3
        @test rho_c_at_min ≈ 5.24e-6 rtol = 2.0e-2

        rho_c_cf = P.q_min * rho_d                            # 1.2e-6 kg/m^3
        @test rho_c_cf < rho_c_at_min                         # the window is non-empty

        # A layer inside the window: cloudy, clamped, and COUNTED.
        rho_c_mid = 0.5 * (rho_c_cf + rho_c_at_min)
        cld = Scythe.CloudOpticsColumn(1)
        n_liq, _ = Scythe.cloud_optics_column!(cld, P, [rho_d], [rho_c_mid], [0.0],
                                                nothing, [10.0])
        @test cld.cf[1] == 1.0
        @test cld.re_liq[1] == Scythe.RAD_RE_LIQ_MIN
        @test n_liq == 1
        # Optically negligible, which is why it is kept rather than screened out: over
        # the 10 m layer used here that is 0.03 g/m^2 of liquid water path, and even at
        # O01's 250 m spacing it is under 1 g/m^2 -- an optical depth of order 1e-3.
        @test cld.lwp[1] < 5.0e-2

        # Just above the window: cloudy, in range, not counted.
        cld2 = Scythe.CloudOpticsColumn(1)
        n_liq2, _ = Scythe.cloud_optics_column!(cld2, P, [rho_d], [2.0 * rho_c_at_min],
                                                 [0.0], nothing, [10.0])
        @test cld2.cf[1] == 1.0
        @test cld2.re_liq[1] > Scythe.RAD_RE_LIQ_MIN
        @test n_liq2 == 0
    end
end
