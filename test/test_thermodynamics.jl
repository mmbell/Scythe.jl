using Test
using Scythe

@testset "Thermodynamics" begin

    # Shorthand aliases for constants
    Rd  = Scythe.Rd
    Rv  = Scythe.Rv
    Eps = Scythe.Eps
    Cvd = Scythe.Cvd
    Cvv = Scythe.Cvv
    Cpd = Scythe.Cpd
    Cpv = Scythe.Cpv
    Cl  = Scythe.Cl
    T_0 = Scythe.T_0
    p_0 = Scythe.p_0
    q0  = Scythe.q0
    L_v0 = Scythe.L_v0
    rho_d0 = Scythe.rho_d0
    rho_v0 = Scythe.rho_v0
    gravity = Scythe.gravity

    # ──────────────────────────────────────────────
    # 1. Constants consistency
    # ──────────────────────────────────────────────
    @testset "Constants consistency" begin
        @test Cpd == Cvd + Rd
        @test Cpv == Cvv + Rv
        @test Eps ≈ Rd / Rv
        @test rho_d0 ≈ 100.0 * p_0 / (T_0 * Rd)
        @test rho_v0 ≈ 100.0 * Scythe.sat_pressure_liquid(T_0) / (T_0 * Rv)
    end

    # ──────────────────────────────────────────────
    # 2. Saturation pressure
    # ──────────────────────────────────────────────
    @testset "Saturation pressure" begin
        # Triple point of water: es ~ 6.112 hPa at 273.16 K
        @test Scythe.sat_pressure_liquid(T_0) ≈ 6.112 atol=0.02

        # Boiling point: es ~ 1013.25 hPa at 373.15 K (within 5%)
        es_boil = Scythe.sat_pressure_liquid(373.15)
        @test es_boil ≈ 1013.25 rtol=0.05

        # Ice saturation at triple point should also be near 6.112 hPa
        @test Scythe.sat_pressure_ice(T_0) ≈ 6.112 atol=0.05

        # Buck formula should agree with simple Bolton near T_0
        es_buck = Scythe.sat_pressure_liquid_buck(T_0, 1013.25)
        es_simple = Scythe.sat_pressure_liquid(T_0)
        @test es_buck ≈ es_simple rtol=0.01

        # Buck derivative via finite difference
        dT = 1.0e-5
        Tk_test = 290.0
        p_test = 1013.25
        fd = (Scythe.sat_pressure_liquid_buck(Tk_test + dT, p_test) -
              Scythe.sat_pressure_liquid_buck(Tk_test - dT, p_test)) / (2.0 * dT)
        analytic = Scythe.sat_pressure_liquid_buck_dT(Tk_test, p_test)
        @test analytic ≈ fd rtol=1.0e-4

        # q_sat_liquid should give reasonable values at standard conditions
        qs = Scythe.q_sat_liquid(300.0, 1013.25)
        @test 0.01 < qs < 0.05  # roughly 10-50 g/kg at 300 K

        # q_sat_ice should be less than q_sat_liquid at same sub-freezing T
        T_cold = 260.0
        @test Scythe.q_sat_ice(T_cold, 500.0) < Scythe.q_sat_liquid(T_cold, 500.0)

        # Buck ice saturation near triple point
        ei_buck = Scythe.sat_pressure_ice_buck(T_0, 1013.25)
        @test ei_buck ≈ 6.112 atol=0.1
    end

    # ──────────────────────────────────────────────
    # 3. Latent heat
    # ──────────────────────────────────────────────
    @testset "Latent heat" begin
        @test Scythe.L_v(T_0) == L_v0
        # L_v should decrease with temperature (Cpv - Cl < 0)
        @test Scythe.L_v(300.0) < Scythe.L_v(280.0)
    end

    # ──────────────────────────────────────────────
    # 4. Entropy / temperature round-trip
    # ──────────────────────────────────────────────
    @testset "Entropy-temperature round-trip" begin
        test_cases = [
            (T_0,   rho_d0, 0.010),   # standard triple-point conditions
            (300.0, 1.1,    0.015),    # warm moist
            (250.0, 1.4,    0.002),    # cold dry-ish
            (280.0, 1.0,    0.005),    # moderate
            (310.0, 0.9,    0.020),    # hot humid
        ]
        for (Tk, rho_d, qv) in test_cases
            s = Scythe.entropy(Tk, rho_d, qv)
            Tk_rec = Scythe.temperature(s, rho_d, qv)
            @test Tk_rec ≈ Tk atol=1.0e-10
        end
    end

    # ──────────────────────────────────────────────
    # 5. Pressure
    # ──────────────────────────────────────────────
    @testset "Pressure" begin
        # Ideal gas consistency: p = 0.01*(Rd + q_v*Rv)*Tk*rho_d
        Tk = 300.0; rho_d = 1.1; qv = 0.012
        s = Scythe.entropy(Tk, rho_d, qv)
        p = Scythe.pressure(s, rho_d, qv)
        p_ideal = 0.01 * (Rd + qv * Rv) * Tk * rho_d
        @test p ≈ p_ideal atol=1.0e-10

        # Standard-ish atmosphere: pressure in reasonable range
        @test 800.0 < p < 1200.0

        # Dry case: no vapor contribution
        s_dry = Scythe.entropy(288.0, 1.2, 0.0)
        p_dry = Scythe.pressure(s_dry, 1.2, 0.0)
        p_dry_ideal = 0.01 * Rd * 288.0 * 1.2
        @test p_dry ≈ p_dry_ideal atol=1.0e-10
    end

    # ──────────────────────────────────────────────
    # 6. Transform round-trips
    # ──────────────────────────────────────────────
    @testset "Transform round-trips" begin
        # mu_transform / inv_mu_transform
        q_test = 0.012
        mu = Scythe.mu_transform(q_test)
        @test Scythe.inv_mu_transform(mu) ≈ q_test atol=1.0e-15

        # inv_mu_transform returns 0 for negative input
        @test Scythe.inv_mu_transform(-1.0) == 0.0

        # dry_density / log_dry_density round-trip
        xi_test = 0.05
        rho = Scythe.dry_density(xi_test)
        @test Scythe.log_dry_density(rho) ≈ xi_test atol=1.0e-12

        # log_dry_density / dry_density round-trip (other direction)
        rho_test = 1.15
        xi = Scythe.log_dry_density(rho_test)
        @test Scythe.dry_density(xi) ≈ rho_test atol=1.0e-10

        # ahyp / bhyp round-trip
        qv_orig = 0.015
        mu_hyp = Scythe.bhyp(qv_orig)
        qv_rec = Scythe.ahyp(mu_hyp)
        @test qv_rec ≈ qv_orig atol=1.0e-10

        # ahyp returns 0 for negative input
        @test Scythe.ahyp(-0.001) == 0.0
    end

    # ──────────────────────────────────────────────
    # 7. Thermodynamic tuple
    # ──────────────────────────────────────────────
    @testset "Thermodynamic tuple" begin
        s_val = 100.0
        xi_val = 0.01
        mu_val = Scythe.mu_transform(0.010)

        qv, rho_d, Tk, p = Scythe.thermodynamic_tuple(s_val, xi_val, mu_val)

        # q_v should match inv_mu_transform
        @test qv ≈ Scythe.inv_mu_transform(mu_val) atol=1.0e-15

        # rho_d should match dry_density
        @test rho_d ≈ Scythe.dry_density(xi_val) atol=1.0e-15

        # Tk should match temperature
        Tk_direct = Scythe.temperature(s_val, rho_d, qv)
        @test Tk ≈ Tk_direct atol=1.0e-12
    end

    # ──────────────────────────────────────────────
    # 8. Pressure derivatives (finite-difference check)
    # ──────────────────────────────────────────────
    @testset "Pressure derivatives" begin
        Tk = 290.0; rho_d = 1.1; qv = 0.010
        eps_fd = 1.0e-6

        # All partial derivatives are at constant (s, xi, qv) coordinates
        # pressure() returns hPa, P_s/P_xi/P_qv return Pa-scale values
        s0 = Scythe.entropy(Tk, rho_d, qv)
        xi0 = Scythe.log_dry_density(rho_d)

        # P_s: dp/ds at constant xi, q_v
        p_plus  = Scythe.pressure(s0 + eps_fd, rho_d, qv)
        p_minus = Scythe.pressure(s0 - eps_fd, rho_d, qv)
        dP_ds_fd = (p_plus - p_minus) / (2.0 * eps_fd)
        Ps_analytic = Scythe.P_s(Tk, rho_d, qv) * 0.01
        @test Ps_analytic ≈ dP_ds_fd rtol=1.0e-4

        # P_xi: dp/d(xi) at constant s, q_v
        rho_plus  = Scythe.dry_density(xi0 + eps_fd)
        rho_minus = Scythe.dry_density(xi0 - eps_fd)
        p_xi_plus  = Scythe.pressure(s0, rho_plus, qv)
        p_xi_minus = Scythe.pressure(s0, rho_minus, qv)
        dP_dxi_fd = (p_xi_plus - p_xi_minus) / (2.0 * eps_fd)
        Pxi_analytic = Scythe.P_xi(Tk, rho_d, qv) * 0.01
        @test Pxi_analytic ≈ dP_dxi_fd rtol=2.0e-3

        # P_qv: dp/d(q_v) at constant s, xi
        p_qplus  = Scythe.pressure(s0, rho_d, qv + eps_fd)
        p_qminus = Scythe.pressure(s0, rho_d, qv - eps_fd)
        dP_dqv_fd = (p_qplus - p_qminus) / (2.0 * eps_fd)
        Pqv_analytic = Scythe.P_qv(Tk, rho_d, qv) * 0.01
        @test Pqv_analytic ≈ dP_dqv_fd rtol=1.0e-3

        # P_s should be positive (higher entropy -> higher pressure)
        @test Scythe.P_s(Tk, rho_d, qv) > 0.0

        # pressure_gradient: linear combination of partials
        s_x = 0.001; xi_x = 0.002; qv_x = 0.0001
        pg = Scythe.pressure_gradient(Tk, rho_d, qv, s_x, xi_x, qv_x)
        pg_manual = Scythe.P_s(Tk, rho_d, qv) * s_x +
                    Scythe.P_xi(Tk, rho_d, qv) * xi_x +
                    Scythe.P_qv(Tk, rho_d, qv) * qv_x
        @test pg ≈ pg_manual atol=1.0e-10
    end

    # ──────────────────────────────────────────────
    # 9. Vapor pressure and mixing ratio
    # ──────────────────────────────────────────────
    @testset "Vapor pressure and mixing ratio" begin
        p_total = 1013.25
        qv = 0.012
        e = Scythe.vapor_pressure(p_total, qv)
        qv_rec = Scythe.mixing_ratio(p_total, e)
        @test qv_rec ≈ qv atol=1.0e-12

        # Dewpoint should be below or equal to temperature for unsaturated air
        Td = Scythe.dewpoint(p_total, qv)
        @test Td < 300.0   # well below typical warm temperature

        # Dewpoint should be positive K for reasonable moisture
        @test Td > 200.0
    end

    # ──────────────────────────────────────────────
    # 10. Rayleigh damping
    # ──────────────────────────────────────────────
    @testset "Rayleigh damping" begin
        # Full Durran & Klemp (1983) eq. 29 profile (originally Klemp & Lilly 1978):
        # half-cosine ramp over the lower half of the layer, linear continuation over
        # the upper half, C1-continuous at the junction.
        alpha = 0.01
        z_d = 10000.0
        z_t = 15000.0
        depth = z_t - z_d

        # Zero at or below z_d
        @test Scythe.Rayleigh_damping(alpha, z_d, z_d, z_t) == 0.0
        @test Scythe.Rayleigh_damping(alpha, 5000.0, z_d, z_t) == 0.0

        # Lower half: -(alpha/2)(1 - cos(norm_z*pi))
        z_q1 = z_d + 0.25 * depth
        @test Scythe.Rayleigh_damping(alpha, z_q1, z_d, z_t) ≈
              -0.5 * alpha * (1.0 - cos(0.25 * pi)) atol = 1.0e-12

        # Junction at norm_z = 1/2: -alpha/2
        z_mid = z_d + 0.5 * depth
        @test Scythe.Rayleigh_damping(alpha, z_mid, z_d, z_t) ≈ -0.5 * alpha atol = 1.0e-12

        # Upper half: -(alpha/2)[1 + (norm_z - 1/2)*pi]
        z_q3 = z_d + 0.75 * depth
        @test Scythe.Rayleigh_damping(alpha, z_q3, z_d, z_t) ≈
              -0.5 * alpha * (1.0 + 0.25 * pi) atol = 1.0e-12

        # Maximum magnitude at z_t equals -(alpha/2)(1 + pi/2)
        @test Scythe.Rayleigh_damping(alpha, z_t, z_d, z_t) ≈
              -0.5 * alpha * (1.0 + 0.5 * pi) atol = 1.0e-12

        # Monotonically increasing magnitude with height
        tau1 = Scythe.Rayleigh_damping(alpha, 11000.0, z_d, z_t)
        tau2 = Scythe.Rayleigh_damping(alpha, 13000.0, z_d, z_t)
        tau3 = Scythe.Rayleigh_damping(alpha, z_t, z_d, z_t)
        @test tau1 > tau2 > tau3   # more negative = stronger damping

        # C1 continuity at the junction: one-sided finite-difference slopes agree
        # (both branches have slope -(alpha/2)*pi/depth there, and the cosine branch
        # has zero curvature at norm_z = 1/2, so the match is tight)
        eps_z = 1.0e-6 * depth
        slope_lo = (Scythe.Rayleigh_damping(alpha, z_mid, z_d, z_t) -
                    Scythe.Rayleigh_damping(alpha, z_mid - eps_z, z_d, z_t)) / eps_z
        slope_hi = (Scythe.Rayleigh_damping(alpha, z_mid + eps_z, z_d, z_t) -
                    Scythe.Rayleigh_damping(alpha, z_mid, z_d, z_t)) / eps_z
        @test slope_lo ≈ slope_hi rtol = 1.0e-6
        @test slope_lo ≈ -0.5 * alpha * pi / depth rtol = 1.0e-6
    end

    # ──────────────────────────────────────────────
    # 11. Thermal conductivity
    # ──────────────────────────────────────────────
    @testset "Thermal conductivity" begin
        k = Scythe.thermal_conductivity(293.15)  # 20 C
        @test k > 0.0
    end

    # ──────────────────────────────────────────────
    # 12. Edge cases: dry air (q_v = 0)
    # ──────────────────────────────────────────────
    @testset "Edge cases - dry air" begin
        Tk = 280.0; rho_d = 1.2

        # Entropy with q_v = 0 should not error
        s_dry = Scythe.entropy(Tk, rho_d, 0.0)
        @test isfinite(s_dry)

        # Temperature round-trip with q_v = 0
        Tk_rec = Scythe.temperature(s_dry, rho_d, 0.0)
        @test Tk_rec ≈ Tk atol=1.0e-10

        # P_qv returns 0 for q_v = 0
        @test Scythe.P_qv(Tk, rho_d, 0.0) == 0.0
    end

end
