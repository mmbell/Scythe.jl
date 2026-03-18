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
