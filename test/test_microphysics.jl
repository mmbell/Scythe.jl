using Test
using Scythe
using SpecialFunctions: gamma

@testset "Microphysics" begin

    # ──────────────────────────────────────────────
    # Helper: set up typical thermodynamic state
    # ──────────────────────────────────────────────
    # Warm, moist conditions near surface
    Tk_std = 293.15        # 20 C
    rho_d_std = 1.1        # kg/m^3
    q_v_std = 0.012        # 12 g/kg
    p_std = 1013.25        # hPa

    # Compute entropy-based coordinates for saturation_adjustment tests
    s_std = Scythe.entropy(Tk_std, rho_d_std, q_v_std)
    xi_std = Scythe.log_dry_density(rho_d_std)

    # ──────────────────────────────────────────────
    # 1. Saturation adjustment
    # ──────────────────────────────────────────────
    @testset "Saturation adjustment" begin

        tol = 1.0e-8

        # 1a. No water (q_v = 0): should return (0, 0)
        s_dry = Scythe.entropy(Tk_std, rho_d_std, 0.0)
        xi_dry = Scythe.log_dry_density(rho_d_std)
        mu_dry = Scythe.mu_transform(0.0)
        mu_l_dry = Scythe.mu_transform(0.0)
        dq, dT = Scythe.saturation_adjustment(s_dry, xi_dry, mu_dry, mu_l_dry, tol)
        @test dq == 0.0
        @test dT == 0.0

        # 1b. Already saturated: q_v = q_sat should return (0, 0)
        q_sat = Scythe.q_sat_liquid(Tk_std, p_std)
        s_sat = Scythe.entropy(Tk_std, rho_d_std, q_sat)
        mu_sat = Scythe.mu_transform(q_sat)
        mu_l_zero = Scythe.mu_transform(0.0)
        dq_sat, dT_sat = Scythe.saturation_adjustment(s_sat, xi_std, mu_sat, mu_l_zero, tol)
        @test abs(dq_sat) < 1.0e-6
        @test abs(dT_sat) < 1.0e-3

        # 1c. Supersaturated: q_v > q_sat, should condense (dq < 0, meaning vapor decreases)
        #     Use higher rho_d to get pressure high enough that q_v actually exceeds q_sat
        rho_d_super = 1.225  # sea-level density
        xi_super = Scythe.log_dry_density(rho_d_super)
        q_v_super = q_sat * 1.05  # 5% above original q_sat
        s_super = Scythe.entropy(Tk_std, rho_d_super, q_v_super)
        mu_super = Scythe.mu_transform(q_v_super)
        dq_super, dT_super = Scythe.saturation_adjustment(s_super, xi_super, mu_super, mu_l_zero, tol)
        # dq should be negative (condensation removes vapor) -- dq = q_sat - q_v
        @test dq_super < 0.0

        # 1d. Subsaturated with liquid water: should evaporate (dq > 0)
        q_v_sub = q_sat * 0.80  # 80% RH
        q_l_sub = 0.001         # some cloud water available
        s_sub = Scythe.entropy(Tk_std, rho_d_std, q_v_sub)
        mu_sub = Scythe.mu_transform(q_v_sub)
        mu_l_sub = Scythe.mu_transform(q_l_sub)
        dq_evap, dT_evap = Scythe.saturation_adjustment(s_sub, xi_std, mu_sub, mu_l_sub, tol)
        # dq should be positive (evaporation adds vapor)
        @test dq_evap > 0.0

        # 1e. Conservation: result should not produce negative q_v or q_l
        q_v_test = 0.001
        q_l_test = 0.0005
        s_test = Scythe.entropy(Tk_std, rho_d_std, q_v_test)
        mu_test = Scythe.mu_transform(q_v_test)
        mu_l_test = Scythe.mu_transform(q_l_test)
        dq_cons, _ = Scythe.saturation_adjustment(s_test, xi_std, mu_test, mu_l_test, tol)
        @test q_v_test + dq_cons >= -1.0e-15
        @test q_l_test - dq_cons >= -1.0e-15
    end

    # ──────────────────────────────────────────────
    # 2. Cloud droplet geometry
    # ──────────────────────────────────────────────
    @testset "Cloud droplet geometry" begin

        # 2a. Typical conditions: N_c=100 #/cm^3, q_c=0.001 kg/kg, rho_d=1.0
        #     Should give radius around 10 microns
        r_c = Scythe.cloud_droplet_radius(100.0, 0.001, 1.0)
        @test 5.0 < r_c < 20.0  # roughly 10 micron range

        # 2b. Inverse consistency: radius -> number -> radius
        N_c_orig = 100.0
        q_c_test = 0.001
        rho_d_test = 1.0
        r_from_N = Scythe.cloud_droplet_radius(N_c_orig, q_c_test, rho_d_test)
        N_from_r = Scythe.cloud_droplet_number(r_from_N, q_c_test, rho_d_test)
        @test N_from_r ≈ N_c_orig rtol=1.0e-6

        # 2c. Zero q_c gives zero radius
        r_zero = Scythe.cloud_droplet_radius(100.0, 0.0, 1.0)
        @test r_zero == 0.0

        # 2d. Zero r_c in cloud_droplet_number gives zero
        N_zero = Scythe.cloud_droplet_number(0.0, 0.001, 1.0)
        @test N_zero == 0.0
    end

    # ──────────────────────────────────────────────
    # 3. q_condensation
    # ──────────────────────────────────────────────
    @testset "q_condensation" begin

        Tk = 293.15
        p = 1013.25
        rho_d = 1.1
        max_N_c = 100.0

        # 3a. Subsaturated with no cloud: returns 0.0
        q_cond_none = Scythe.q_condensation(0.95, Tk, p, rho_d, 0.012, 0.0, max_N_c)
        @test q_cond_none == 0.0

        # 3b. Supersaturated (sat_ratio > 1.0001): positive condensation
        q_cond_pos = Scythe.q_condensation(1.01, Tk, p, rho_d, 0.015, 0.001, max_N_c)
        @test q_cond_pos > 0.0

        # 3c. Subsaturated with existing cloud (sat_ratio < 1): negative (evaporation)
        q_cond_neg = Scythe.q_condensation(0.95, Tk, p, rho_d, 0.010, 0.002, max_N_c)
        @test q_cond_neg < 0.0

        # 3d. Bounded by available water: condensation cannot exceed q_v
        q_v_small = 1.0e-6
        q_cond_bound = Scythe.q_condensation(1.05, Tk, p, rho_d, q_v_small, 0.001, max_N_c)
        @test q_cond_bound <= q_v_small

        # 3e. Nucleation threshold: no condensation just above saturation (sat_ratio <= 1.0001)
        q_cond_nuc = Scythe.q_condensation(1.00005, Tk, p, rho_d, 0.015, 0.0, max_N_c)
        @test q_cond_nuc == 0.0
    end

    # ──────────────────────────────────────────────
    # 4. q_evaporation
    # ──────────────────────────────────────────────
    @testset "q_evaporation" begin

        Tk = 293.15
        p = 1013.25
        rho_d = 1.1
        mean_r = 500.0  # 500 micron mean rain drop radius

        # 4a. No rain (q_r = 0): returns 0
        q_evap_norain = Scythe.q_evaporation(0.90, Tk, p, rho_d, 0.010, 0.0, mean_r)
        @test q_evap_norain == 0.0

        # 4b. Supersaturated: returns 0 (no evaporation when saturated)
        q_evap_super = Scythe.q_evaporation(1.05, Tk, p, rho_d, 0.015, 0.001, mean_r)
        @test q_evap_super == 0.0

        # 4c. Subsaturated with rain: positive evaporation
        q_evap_pos = Scythe.q_evaporation(0.90, Tk, p, rho_d, 0.010, 0.001, mean_r)
        @test q_evap_pos > 0.0
    end

    # ──────────────────────────────────────────────
    # 5. Autoconversion, collection, sedimentation
    # ──────────────────────────────────────────────
    @testset "Autoconversion, collection, sedimentation" begin

        rho_d = 1.0
        Tk = 293.15

        # 5a. Autoconversion threshold at q_c = 0.001
        q_auto_at = Scythe.autoconversion(0.001, rho_d)
        @test q_auto_at == 0.0

        # 5b. Autoconversion returns 0 below threshold
        q_auto_below = Scythe.autoconversion(0.0005, rho_d)
        @test q_auto_below == 0.0

        # 5c. Autoconversion positive above threshold
        q_auto_above = Scythe.autoconversion(0.002, rho_d)
        @test q_auto_above > 0.0

        # 5d. Collection non-negative
        q_coll = Scythe.collection(0.001, 0.001, rho_d, Tk)
        @test q_coll >= 0.0

        # 5e. Collection returns 0 for zero cloud or rain
        @test Scythe.collection(0.0, 0.001, rho_d, Tk) == 0.0

        # 5f. Sedimentation negative (downward velocity) for positive q_r
        Vt = Scythe.sedimentation(0.001, rho_d, Tk)
        @test Vt < 0.0
    end

    # ──────────────────────────────────────────────
    # 5.5 Density-form warm-rain rates for the total-energy set
    # ──────────────────────────────────────────────
    @testset "Density-form warm-rain rates (Ooyama 2001 App. A)" begin
        Tk = 290.0
        p_hPa = 950.0
        rho_d = 1.1

        # Autoconversion: Q_auto = 0.001*(rho_c - 0.001*rho_d), clamped >= 0.
        # Consistent with the mixing-ratio form: rho_d * autoconversion(q_c) at q_c = rho_c/rho_d.
        @test Scythe.autoconversion_density(2.0e-3, 1.0) ≈ 1.0e-6
        @test Scythe.autoconversion_density(0.5e-3, 1.0) == 0.0
        @test Scythe.autoconversion_density(3.0e-3, rho_d) ≈
              rho_d * Scythe.autoconversion(3.0e-3 / rho_d, rho_d)

        # Collection: Q_coll = 2.20*rho_c*(rho_r/rho_d)^0.875*f_ice = rho_d * q-form
        @test Scythe.collection_density(1.0e-3, 1.0e-3, rho_d, Tk) ≈
              rho_d * Scythe.collection(1.0e-3 / rho_d, 1.0e-3 / rho_d, rho_d, Tk)
        @test Scythe.collection_density(0.0, 1.0e-3, rho_d, Tk) == 0.0

        # Terminal velocity: same fall speed as the mixing-ratio form, <= 0
        @test Scythe.rain_terminal_velocity(1.0e-3, rho_d, Tk) ≈
              Scythe.sedimentation(1.0e-3 / rho_d, rho_d, Tk)
        @test Scythe.rain_terminal_velocity(1.0e-3, rho_d, Tk) < 0.0
        @test Scythe.rain_terminal_velocity(0.0, rho_d, Tk) == 0.0

        # Ventilation factor: same as the q-form at rho_r = q_r*rho_d
        @test Scythe.f_ventilation_density(1.0e-3, Tk) ≈
              Scythe.f_ventilation(1.0e-3 / rho_d, rho_d, Tk)
        @test Scythe.f_ventilation_density(0.0, Tk) ≈ 1.6

        # Monodisperse rain drop radius [microns]: rho_r = N_r*1e6*(4/3)*pi*rho_l*(r*1e-6)^3.
        # Round trip through the cloud geometry helper (same monodisperse assumption).
        N_r = 1.0e-3   # #/cm^3 (~1000 per m^3)
        r_r = Scythe.rain_drop_radius(N_r, 1.0e-3)
        @test r_r ≈ Scythe.cloud_droplet_radius(N_r, 1.0e-3 / rho_d, rho_d)
        @test 500.0 < r_r < 800.0            # ~620 microns at 1 g/m^3, N_r = 1e-3/cm^3
        @test Scythe.rain_drop_radius(0.0, 1.0e-3) == 0.0

        # Rain relaxation timescale: tau_r = (4*pi*Dv*N_r*<r_r*f(r)>)^-1, with the bulk
        # ventilation factor. Same units convention as invtau_condensation.
        invtau_r = Scythe.invtau_rain(Tk, p_hPa, N_r, 1.0e-3)
        @test invtau_r ≈ Scythe.invtau_condensation(Tk, p_hPa, N_r, r_r) *
                         Scythe.f_ventilation_density(1.0e-3, Tk)
        @test 5.0e-4 < invtau_r < 5.0e-3     # ~1.7e-3 1/s at 1 g/m^3
        # No rain (or no drops): no relaxation through the rain channel
        @test Scythe.invtau_rain(Tk, p_hPa, N_r, 0.0) == 0.0
        @test Scythe.invtau_rain(Tk, p_hPa, N_r, 0.5e-8) == 0.0   # below RHO_R_MIN
        @test Scythe.invtau_rain(Tk, p_hPa, 0.0, 1.0e-3) == 0.0

        # Spline undershoots: negative rho_r to a fractional power is NaN unless guarded.
        for f in (rho_r -> Scythe.autoconversion_density(rho_r, rho_d),
                  rho_r -> Scythe.collection_density(1.0e-3, rho_r, rho_d, Tk),
                  rho_r -> Scythe.rain_terminal_velocity(rho_r, rho_d, Tk),
                  rho_r -> Scythe.f_ventilation_density(rho_r, Tk),
                  rho_r -> Scythe.rain_drop_radius(N_r, rho_r),
                  rho_r -> Scythe.invtau_rain(Tk, p_hPa, N_r, rho_r))
            @test isfinite(f(-1.0e-12))
        end
        @test Scythe.rain_terminal_velocity(-1.0e-12, rho_d, Tk) == 0.0
        @test Scythe.invtau_rain(Tk, p_hPa, N_r, -1.0e-12) == 0.0

        # rho_d undershoots: vigorous convection can momentarily drive the dry
        # density negative at cloud edges (this killed the first 6-h TC run at
        # rain onset: (rho_d0/rho_d)^0.25 with rho_d = -0.08). Every rate that
        # divides by rho_d or raises a rho_d ratio to a fractional power must
        # stay finite there too.
        for f in (rd -> Scythe.rain_terminal_velocity(1.0e-3, rd, Tk),
                  rd -> Scythe.collection_density(1.0e-3, 1.0e-3, rd, Tk),
                  rd -> Scythe.f_ventilation_mp(2000.0, rd, Tk),
                  rd -> Scythe.invtau_rain_mp(Tk, p_hPa, 8.0e6, 1.0e-3, rd),
                  rd -> Scythe.autoconversion_density(1.0e-3, rd))
            @test isfinite(f(-0.08))
            @test isfinite(f(0.0))
            @test isfinite(f(1.0e-6))
        end
        # and the guarded factors stay bounded (no blowup from a tiny floor)
        @test abs(Scythe.rain_terminal_velocity(1.0e-3, -0.08, Tk)) < 100.0
        @test Scythe.f_ventilation_mp(2000.0, -0.08, Tk) < 100.0
    end

    # ──────────────────────────────────────────────
    # 5b. Exponential (Marshall-Palmer) rain DSD: tau-only closure
    # ──────────────────────────────────────────────
    @testset "Marshall-Palmer rain DSD (tau-only)" begin
        Tk = 293.15
        p_hPa = 1013.25
        rho_d = 1.1
        N_0 = 8.0e6            # MP intercept [m^-4]
        rho_r = 1.0e-3         # 1 g/m^3

        # Slope roundtrip: rho_r = pi rho_l N_0 / lambda^4
        lam = Scythe.mp_slope(rho_r, N_0)
        @test lam ≈ (pi * Scythe.rho_l * N_0 / rho_r)^0.25
        @test pi * Scythe.rho_l * N_0 / lam^4 ≈ rho_r
        @test 2000.0 < lam < 2500.0          # ~2239 m^-1 at 1 g/m^3

        # The DSD-integrated diffusional moment 2*pi*Dv*N_0/lambda^2 equals the
        # monodisperse formula 4*pi*Dv*N_T*r evaluated at the DSD total number
        # N_T = N_0/lambda and mean radius r = 1/(2*lambda), so invtau_rain_mp is
        # exactly the composition of the existing plumbing with the DSD-mean
        # ventilation factor (units convention: #/cm^3 and microns).
        N_eff_cm3 = 1.0e-6 * N_0 / lam
        r_eff_um = 1.0e6 / (2.0 * lam)
        @test (N_eff_cm3 * 1.0e6) * (r_eff_um * 1.0e-6) ≈ N_0 / (2.0 * lam^2)
        invtau_mp = Scythe.invtau_rain_mp(Tk, p_hPa, N_0, rho_r, rho_d)
        @test invtau_mp ≈ Scythe.invtau_condensation(Tk, p_hPa, N_eff_cm3, r_eff_um) *
                          Scythe.f_ventilation_mp(lam, rho_d, Tk)

        # DSD-mean ventilation: quiescent limit 0.78, enhancement ∝ lambda^{-3/4}
        # (~4.6 at 1 g/m^3, about half of Ooyama's bulk 9.0 — the intended slowdown);
        # thinner air ventilates more (faster fall speed).
        fbar = Scythe.f_ventilation_mp(lam, rho_d, Tk)
        @test 4.0 < fbar < 5.5
        @test Scythe.f_ventilation_mp(Inf, rho_d, Tk) ≈ 0.78
        @test Scythe.f_ventilation_mp(lam, 0.7, Tk) > fbar

        # Guards match the monodisperse channel: inert below RHO_R_MIN, for N_0 <= 0,
        # and for spline undershoots (fractional powers of negative rho_r).
        @test Scythe.invtau_rain_mp(Tk, p_hPa, N_0, 0.0, rho_d) == 0.0
        @test Scythe.invtau_rain_mp(Tk, p_hPa, N_0, 0.5e-8, rho_d) == 0.0   # < RHO_R_MIN
        @test Scythe.invtau_rain_mp(Tk, p_hPa, 0.0, rho_r, rho_d) == 0.0
        @test Scythe.invtau_rain_mp(Tk, p_hPa, N_0, -1.0e-12, rho_d) == 0.0

        # Monotone increasing in rho_r: more rain, faster relaxation
        for rr in (1.0e-6, 1.0e-4, 1.0e-3, 5.0e-3)
            @test Scythe.invtau_rain_mp(Tk, p_hPa, N_0, 1.01 * rr, rho_d) >
                  Scythe.invtau_rain_mp(Tk, p_hPa, N_0, rr, rho_d)
        end

        # Consistency of the retained Ooyama bulk sedimentation with the MP DSD: the
        # MP mass-weighted fall speed a_v*(Γ(4.5)/Γ(4))/sqrt(lambda) matches the bulk
        # terminal velocity within 10% at moderate rain content (Γ(4.5)/Γ(4) = 1.938622).
        Vmp = Scythe.MP_AV * 1.938622 / sqrt(lam)
        @test isapprox(Vmp, -Scythe.rain_terminal_velocity(rho_r, Scythe.rho_d0, Tk),
                       rtol=0.1)

        # Slower phase change than the monodisperse closure at the model defaults —
        # fewer large drops means less integrated surface area x ventilation.
        @test invtau_mp < Scythe.invtau_rain(Tk, p_hPa, 1.0e-3, rho_r)
    end

    # ──────────────────────────────────────────────
    # 6. s_condensation (entropy source from condensation)
    # ──────────────────────────────────────────────
    @testset "s_condensation" begin

        Tk = 293.15
        rho_d = 1.1
        q_v = 0.012
        q_l = 0.001
        p = 1013.25
        q_cond = 0.0001   # condensation
        q_evap = 0.00005  # evaporation

        # 6a. Sign consistent: condensation and evaporation give opposite sign entropy changes
        ds_cond = Scythe.s_condensation(q_cond, Tk, rho_d, q_v, q_l, p)
        ds_evap = Scythe.s_condensation(-q_cond, Tk, rho_d, q_v, q_l, p)
        @test sign(ds_cond) == -sign(ds_evap)

        # 6b. 7-arg version consistency: when q_evap=0, should match 6-arg version
        ds_6arg = Scythe.s_condensation(q_cond, Tk, rho_d, q_v, q_l, p)
        ds_7arg = Scythe.s_condensation(0.0, q_cond, Tk, rho_d, q_v, q_l, p)
        @test ds_6arg ≈ ds_7arg atol=1.0e-15

        # 6c. Returns finite values for typical inputs
        @test isfinite(ds_cond)
        @test isfinite(ds_7arg)
    end

    # ──────────────────────────────────────────────
    # 7. Utility functions
    # ──────────────────────────────────────────────
    @testset "Utility functions" begin

        Tk = 293.15
        p = 1013.25

        # 7a. Vapor diffusivity positive at standard conditions
        Dv = Scythe.vapor_diffusivity(Tk, p)
        @test Dv > 0.0

        # 7b. Droplet growth rate positive at standard conditions
        G = Scythe.droplet_growth_rate(Tk, p)
        @test G > 0.0

        # 7c. Q_s_factor positive for typical conditions
        Qs = Scythe.Q_s_factor(Tk, p, 0.012, 0.001)
        @test Qs > 0.0

        # 7d. invtau_condensation positive for typical conditions
        invtau = Scythe.invtau_condensation(Tk, p, 100.0, 10.0)
        @test invtau > 0.0
    end

    # ──────────────────────────────────────────────
    # 8. Two-moment warm rain (ISHMAEL / Morrison exponential DSD)
    # ──────────────────────────────────────────────
    #
    # These are the density-form ports of module_mp_jensen_ishmael.F lines 1707-1785 and
    # 2244-2247. Each testset states the Fortran form independently (the "twin"), in the
    # Fortran's own MIXING-RATIO variables, and checks that the density-form function agrees
    # exactly-to-roundoff. A twin that merely restated the Julia would test nothing; these
    # restate the FORTRAN, so a units slip in the conversion is what they catch.
    @testset "Two-moment warm rain" begin

        RHOW = Scythe.ISHMAEL_RHOW
        PI_I = Scythe.ISHMAEL_PI
        AR = Scythe.ISHMAEL_AR
        BR = Scythe.ISHMAEL_BR
        R0 = Scythe.ISHMAEL_R0

        # A representative rainy state: 1 g/m^3 of rain in 1000 drops per m^3
        # (mean drop mass 1e-6 kg, D ~ 1.24 mm), 1 g/m^3 of cloud, near-surface air.
        rho_d = 1.1
        rho_r = 1.0e-3
        n_r = 1.0e3
        rho_c = 1.0e-3
        Tk = 293.15
        p_hPa = 1000.0

        @testset "DSD slope: density form == mixing-ratio form" begin
            q_r = rho_r / rho_d
            nr_kg = n_r / rho_d
            twin = Scythe.ishmael_rain_lambda(q_r, nr_kg)
            dsd = Scythe.rain_dsd_2m(rho_r, n_r, rho_d)

            # lamr is conversion-INVARIANT: the Fortran forms it from nr/qr, which is n_r/rho_r
            @test dsd.lamr == twin.lamr
            @test dsd.lamr ≈ (PI_I * RHOW * n_r / rho_r)^(1 / 3) rtol=1e-12
            # n0rr and n_r carry exactly one factor of the air density
            @test dsd.n0rr ≈ twin.n0rr * rho_d rtol=1e-14
            @test dsd.n_r ≈ twin.nr * rho_d rtol=1e-14
            @test dsd.q_r ≈ q_r rtol=1e-14

            # Empty distribution: no NaN, no Inf leaking into the callers
            empty = Scythe.rain_dsd_2m(0.0, 0.0, rho_d)
            @test empty.n0rr == 0.0
            @test empty.n_r == 0.0
            @test isfinite(empty.lamr)
            @test Scythe.rain_dsd_2m(-1.0e-6, -5.0, rho_d).n0rr == 0.0
        end

        @testset "KK2000 autoconversion (lines 1726-1734)" begin
            N_c = 100.0                                  # #/cm^3
            q_c = rho_c / rho_d
            # Fortran: PRC = 1350 qc^2.47 (nc/1e6*rhoair)^(-1.79), and (nc/1e6*rhoair) IS
            # the droplet number in #/cm^3.
            PRC = 1350.0 * q_c^2.47 * N_c^(-1.79)
            NPRC1 = PRC / (4.0 / 3.0 * PI_I * RHOW * (25.0e-6)^3)

            Qdot, Ndot = Scythe.rain_autoconversion_2m(rho_c, rho_d, N_c)
            @test Qdot ≈ PRC * rho_d rtol=1e-13
            @test Ndot ≈ NPRC1 * rho_d rtol=1e-13
            # The number source is the mass source divided by a 25 micron-RADIUS drop
            @test Ndot ≈ Qdot / Scythe.RAIN_2M_M_AUTO rtol=1e-14
            @test Scythe.RAIN_2M_M_AUTO ≈ (4 / 3) * pi * 1000.0 * (25.0e-6)^3 rtol=1e-6

            # Threshold and negative-argument safety: exact zeros, not small numbers
            below = 0.5 * Scythe.RAIN_2M_QC_AUTO * rho_d
            @test Scythe.rain_autoconversion_2m(below, rho_d, N_c) === (0.0, 0.0)
            @test Scythe.rain_autoconversion_2m(-1.0e-3, rho_d, N_c) === (0.0, 0.0)
            @test Scythe.rain_autoconversion_2m(rho_c, rho_d, 0.0) === (0.0, 0.0)
            # Monotone in cloud, and steeply so (exponent 2.47)
            @test Scythe.rain_autoconversion_2m(2 * rho_c, rho_d, N_c)[1] >
                  4 * Scythe.rain_autoconversion_2m(rho_c, rho_d, N_c)[1]
            # More droplets for the same water => less autoconversion (exponent -1.79)
            @test Scythe.rain_autoconversion_2m(rho_c, rho_d, 200.0)[1] <
                  Scythe.rain_autoconversion_2m(rho_c, rho_d, 100.0)[1]
        end

        @testset "KK2000 accretion (lines 1736-1742)" begin
            q_c = rho_c / rho_d
            q_r = rho_r / rho_d
            PRA = 67.0 * (q_c * q_r)^1.15
            @test Scythe.rain_accretion_2m(rho_c, rho_r, rho_d) ≈ PRA * rho_d rtol=1e-13

            below = 0.5 * Scythe.RAIN_2M_Q_MIN * rho_d
            @test Scythe.rain_accretion_2m(below, rho_r, rho_d) === 0.0
            @test Scythe.rain_accretion_2m(rho_c, below, rho_d) === 0.0
            @test Scythe.rain_accretion_2m(-1.0e-3, rho_r, rho_d) === 0.0
            @test Scythe.rain_accretion_2m(rho_c, -1.0e-3, rho_d) === 0.0
        end

        @testset "Beheng self-collection / Verlinde-Cotton breakup (lines 1744-1753)" begin
            # Small drops (1/lamr < 300 micron): dum = 1, pure self-collection (a SINK).
            # 0.1 g/m^3 in 1e5 drops/m^3 => mean mass 1e-9 kg, D ~ 124 micron.
            rho_r_s = 1.0e-4
            n_r_s = 1.0e5
            dsd_s = Scythe.rain_dsd_2m(rho_r_s, n_r_s, rho_d)
            @test 1.0 / dsd_s.lamr < 300.0e-6
            twin_s = -5.78 * 1.0 * (n_r_s / rho_d) * (rho_r_s / rho_d) * rho_d * rho_d
            got_s = Scythe.rain_selfcollection_2m(rho_r_s, n_r_s, rho_d)
            @test got_s ≈ twin_s rtol=1e-12
            @test got_s < 0.0

            # Large drops (1/lamr >= 300 micron): breakup drives dum negative, so the term
            # becomes a number SOURCE. This is the intended Verlinde-Cotton behaviour.
            dsd_l = Scythe.rain_dsd_2m(rho_r, n_r, rho_d)
            @test 1.0 / dsd_l.lamr >= 300.0e-6
            dum_l = 2.0 - exp(2300.0 * ((1.0 / dsd_l.lamr) - 300.0e-6))
            @test dum_l < 0.0
            twin_l = -5.78 * dum_l * dsd_l.n_r * rho_r
            @test Scythe.rain_selfcollection_2m(rho_r, n_r, rho_d) ≈ twin_l rtol=1e-12
            @test Scythe.rain_selfcollection_2m(rho_r, n_r, rho_d) > 0.0

            below = 0.5 * Scythe.RAIN_2M_Q_MIN * rho_d
            @test Scythe.rain_selfcollection_2m(below, n_r, rho_d) === 0.0
            @test Scythe.rain_selfcollection_2m(-1.0e-3, n_r, rho_d) === 0.0
        end

        @testset "Fall speeds and size sorting (lines 2244-2247)" begin
            dsd = Scythe.rain_dsd_2m(rho_r, n_r, rho_d)
            arn = AR * (R0 / rho_d)^0.5
            vtrn = min(arn * gamma(1.0 + BR) / dsd.lamr^BR, 9.1)
            vtrm = min(arn * gamma(4.0 + BR) / 6.0 / dsd.lamr^BR, 9.1)

            w_m, w_n = Scythe.rain_fall_speeds_2m(rho_r, n_r, rho_d)
            @test w_m ≈ -vtrm rtol=1e-13
            @test w_n ≈ -vtrn rtol=1e-13

            # SIZE SORTING: mass always outruns number. Γ(4.5)/6 = 1.6684 > Γ(1.5) = 0.8862.
            @test abs(w_m) > abs(w_n)
            @test w_m < w_n < 0.0
            @test Scythe.RAIN_2M_GAMMA_4BR / 6.0 > Scythe.RAIN_2M_GAMMA_1BR

            # Sorting holds across the whole size range the DSD clamp admits
            for (rr, nn) in ((1.0e-5, 1.0e6), (1.0e-4, 1.0e5), (5.0e-3, 5.0e2))
                a, b = Scythe.rain_fall_speeds_2m(rr, nn, rho_d)
                @test abs(a) >= abs(b)
                @test a <= 0.0 && b <= 0.0
            end

            # The 9.1 m/s cap binds at the large-drop end (LAMMINR = 1/2800 micron)
            big_m, big_n = Scythe.rain_fall_speeds_2m(1.0e-2, 1.0, rho_d)
            @test abs(big_m) <= Scythe.RAIN_2M_VT_MAX
            @test abs(big_n) <= Scythe.RAIN_2M_VT_MAX
            @test abs(big_m) ≈ Scythe.RAIN_2M_VT_MAX rtol=1e-12

            # Rain-free and negative-argument: exactly still air
            @test Scythe.rain_fall_speeds_2m(0.0, 0.0, rho_d) === (0.0, 0.0)
            @test Scythe.rain_fall_speeds_2m(-1.0e-6, -5.0, rho_d) === (0.0, 0.0)
            # Thinner air => faster fall (the (R0/rho)^0.5 density correction)
            @test abs(Scythe.rain_fall_speeds_2m(rho_r, n_r, 0.4)[1]) >
                  abs(Scythe.rain_fall_speeds_2m(rho_r, n_r, 1.1)[1])
        end

        @testset "invtau_rain_2m: ventilated exponential DSD (lines 1755-1763)" begin
            dsd = Scythe.rain_dsd_2m(rho_r, n_r, rho_d)
            p_Pa = p_hPa * 100.0
            # ISHMAEL's own air properties, lines 1015-1019
            mu_air = 1.496e-6 * Tk^1.5 / (Tk + 120.0)
            dv = 8.794e-5 * Tk^1.81 / p_Pa
            nsch = mu_air / (rho_d * dv)
            arn = AR * (R0 / rho_d)^0.5
            # n0rr in the Fortran is per kg, and the formula carries an explicit rhoair;
            # the density form folds the two into n0rr [m^-4].
            twin = 2.0 * PI_I * dsd.n0rr * dv *
                   (Scythe.ISHMAEL_F1R / (dsd.lamr * dsd.lamr) +
                    Scythe.ISHMAEL_F2R * (arn * rho_d / mu_air)^0.5 *
                    nsch^(1 / 3) * gamma(2.5 + BR / 2) / dsd.lamr^(2.5 + BR / 2))

            got = Scythe.invtau_rain_2m(Tk, p_hPa, rho_r, n_r, rho_d)
            @test got ≈ twin rtol=1e-12
            @test got > 0.0
            # An inverse timescale: seconds, and a physically sane one for 1 g/m^3 of rain
            @test 1.0e-5 < got < 1.0e-1

            # More rain in the same number of drops (bigger drops) => slower relaxation per
            # unit... no: more surface area overall, so a FASTER exchange. Monotone in mass.
            @test Scythe.invtau_rain_2m(Tk, p_hPa, 2 * rho_r, n_r, rho_d) > got
            # More drops holding the same mass => more surface area => faster still
            @test Scythe.invtau_rain_2m(Tk, p_hPa, rho_r, 4 * n_r, rho_d) > got

            below = 0.5 * Scythe.RAIN_2M_Q_MIN * rho_d
            @test Scythe.invtau_rain_2m(Tk, p_hPa, below, n_r, rho_d) === 0.0
            @test Scythe.invtau_rain_2m(Tk, p_hPa, -1.0e-3, n_r, rho_d) === 0.0
            @test Scythe.invtau_rain_2m(Tk, p_hPa, rho_r, -5.0, rho_d) > 0.0   # floored n_r
            # Same order of magnitude as the fixed-intercept Marshall-Palmer closure it
            # replaces, at the classic N_0 (a units sanity check, not an identity).
            mp = Scythe.invtau_rain_mp(Tk, p_hPa, 8.0e6, rho_r, rho_d)
            @test 0.05 < got / mp < 20.0
        end

        @testset "Rain number loss to evaporation (lines 1780-1785)" begin
            # Proportional to the mass loss, at fixed mean drop mass
            Qdot_r = -1.0e-6
            got = Scythe.rain_number_evaporation_2m(Qdot_r, rho_r, n_r)
            @test got ≈ Qdot_r * n_r / rho_r rtol=1e-14
            @test got < 0.0
            # The fractional loss rates of mass and number are equal: mean mass invariant
            @test got / n_r ≈ Qdot_r / rho_r rtol=1e-14

            # Condensation ONTO rain makes no new drops
            @test Scythe.rain_number_evaporation_2m(1.0e-6, rho_r, n_r) === 0.0
            @test Scythe.rain_number_evaporation_2m(0.0, rho_r, n_r) === 0.0
            # The RHO_R_MIN floor keeps a vanishing-mass column finite
            @test isfinite(Scythe.rain_number_evaporation_2m(-1.0e-12, 0.0, n_r))
            @test Scythe.rain_number_evaporation_2m(-1.0e-6, rho_r, -5.0) === 0.0
        end

        @testset "qss_condensation_rates rain channel is switchable" begin
            # The two-moment channel replaces invtau_rain / invtau_rain_mp and nothing else:
            # same split arithmetic, same cloud gate.
            Q_ss = 1.0e-4; rho_v = 0.02; Q_s = 0.5
            args = (Q_ss, rho_v, rho_c, rho_r, rho_d, Tk, p_hPa, Q_s, 0.5, 1.0e-3, 100.0)

            one_m = Scythe.qss_condensation_rates(args...)
            two_m = Scythe.qss_condensation_rates(args...; rain_2m = true, n_r_density = n_r)

            # Default keywords are inert: rain_2m = false reproduces the 1-moment call
            @test Scythe.qss_condensation_rates(args...; rain_2m = false,
                                                n_r_density = n_r) === one_m
            # And the 2-moment arm uses invtau_rain_2m for its rain channel
            @test two_m[4] ≈ Scythe.invtau_rain_2m(Tk, p_hPa, rho_r, n_r, rho_d) rtol=1e-14
            @test two_m[3] == one_m[3]          # cloud channel untouched
            @test two_m[4] != one_m[4]          # rain channel replaced

            # Dry, rain-free, cloud-free air stays exactly inert on both arms
            dry = Scythe.qss_condensation_rates(0.0, 1.0e-6, 0.0, 0.0, rho_d, Tk, p_hPa,
                                                Q_s, 0.5, 1.0e-3, 100.0;
                                                rain_2m = true, n_r_density = 0.0)
            @test dry === (0.0, 0.0, 0.0, 0.0)
        end
    end

end
