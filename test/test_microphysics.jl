using Test
using Scythe

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

end
