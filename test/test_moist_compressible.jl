using Test
using Scythe
using Springsteel

# Tests for the total-energy moist compressible equation set moist_compressible_XZ
# (src/moist_compressible.jl): prognostic p, rho_d, rho_t, u, w, E_t, Q_ss, rho_r
# with T diagnosed from a univariate Newton retrieval and rho_v = Q_ss + rho_vs(T,p),
# rho_c = rho_t - rho_d - rho_v - rho_r recovered diagnostically.

@testset "moist_compressible (total energy)" begin

    import Springsteel.Thermodynamics: rho_v_sat, internal_energy_bf02, L_v,
        Rd, Rv, Cvd, Cvv, Cpd, Cpv, Cl, gravity

    # ──────────────────────────────────────────────
    # 1. Density-form saturation partials vs finite differences
    # ──────────────────────────────────────────────
    @testset "rho_v_sat partial derivatives" begin
        for (Tk, p_hPa) in ((300.0, 1000.0), (270.0, 700.0), (240.0, 400.0))
            hT = 1.0e-3
            fd_T = (rho_v_sat(Tk + hT, p_hPa) - rho_v_sat(Tk - hT, p_hPa)) / (2.0 * hT)
            @test Scythe.drho_vsat_dT(Tk, p_hPa) ≈ fd_T rtol = 1e-6

            hp_Pa = 10.0
            fd_p = (rho_v_sat(Tk, (100.0 * p_hPa + hp_Pa) / 100.0) -
                    rho_v_sat(Tk, (100.0 * p_hPa - hp_Pa) / 100.0)) / (2.0 * hp_Pa)
            @test Scythe.drho_vsat_dp(Tk, p_hPa) ≈ fd_p rtol = 1e-6
            @test Scythe.drho_vsat_dT(Tk, p_hPa) > 0.0
            @test Scythe.drho_vsat_dp(Tk, p_hPa) > 0.0    # Buck enhancement factor only
        end
    end

    # ──────────────────────────────────────────────
    # 2. Temperature retrieval round-trips forward-built states
    # ──────────────────────────────────────────────
    # Build a state from (T, p, saturation fraction, q_l, winds, z), compute the
    # prognostic set (p, rho_d, rho_t, E_t, Q_ss), and invert for T.
    function forward_state(Tk, p_Pa, sat_frac, q_l, u, w, z)
        rho_v = sat_frac * rho_v_sat(Tk, p_Pa / 100.0)
        rho_d = (p_Pa - Rv * Tk * rho_v) / (Rd * Tk)     # EOS with the chosen vapor
        rho_c = q_l * rho_d
        rho_t = rho_d + rho_v + rho_c
        q_v = rho_v / rho_d
        ke = 0.5 * (u^2 + w^2)
        E_t = rho_d * internal_energy_bf02(Tk, q_v, rho_c / rho_d) +
              rho_t * (ke + gravity * z)
        Q_ss = rho_v - rho_v_sat(Tk, p_Pa / 100.0)
        M = p_Pa + E_t - rho_t * (ke + gravity * z)
        return (; M, rho_d, rho_t, Q_ss, p_Pa, rho_v, rho_c)
    end

    @testset "retrieve_temperature round-trip" begin
        cases = (
            (Tk=300.0, p=100000.0, sat=0.0, q_l=0.0,    u=0.0,  w=0.0,  z=0.0),     # dry
            (Tk=285.0, p=90000.0,  sat=1.0, q_l=1.0e-3, u=0.0,  w=0.0,  z=1000.0),  # saturated cloudy
            (Tk=230.0, p=40000.0,  sat=0.5, q_l=0.0,    u=0.0,  w=0.0,  z=8000.0),  # cold aloft
            (Tk=295.0, p=95000.0,  sat=0.9, q_l=5.0e-4, u=15.0, w=5.0,  z=500.0),   # windy
        )
        for c in cases
            st = forward_state(c.Tk, c.p, c.sat, c.q_l, c.u, c.w, c.z)
            for guess in (c.Tk - 20.0, c.Tk + 20.0, 273.0)
                Tret = Scythe.retrieve_temperature(st.M, st.rho_d, st.rho_t, st.Q_ss,
                                                   st.p_Pa, guess)
                @test Tret ≈ c.Tk rtol = 1e-8
            end
            # Diagnostic recovery of the water partition
            Tret = Scythe.retrieve_temperature(st.M, st.rho_d, st.rho_t, st.Q_ss,
                                               st.p_Pa, c.Tk)
            rho_v = st.Q_ss + rho_v_sat(Tret, st.p_Pa / 100.0)
            rho_c = st.rho_t - st.rho_d - rho_v
            @test rho_v ≈ st.rho_v rtol = 1e-8 atol = 1e-14
            @test rho_c ≈ st.rho_c rtol = 1e-6 atol = 1e-10
        end

        # F(T) is monotone increasing over the physical range for a saturated state
        st = forward_state(285.0, 90000.0, 1.0, 1.0e-3, 0.0, 0.0, 1000.0)
        F(T) = (st.rho_d * Cpd + (st.rho_t - st.rho_d) * Cpv) * T +
               (st.Q_ss + st.rho_d + rho_v_sat(T, st.p_Pa / 100.0) - st.rho_t) * L_v(T) -
               st.M
        Ts = 200.0:5.0:330.0
        @test all(diff(F.(Ts)) .> 0.0)
    end

    # ──────────────────────────────────────────────
    # 3. Q_s identity: condensation-induced (dT, dp) reproduce the rho_vs change
    # ──────────────────────────────────────────────
    # This test locks the SIGN of the pressure-equation condensation coefficient:
    # dp|cond = (R_m/C_vt)*(L_v − R_v*C_pt*T/R_m)*delta (minus, from the corrected TeX).
    @testset "Q_s energy-consistent psychrometric identity" begin
        for (Tk, p_Pa, q_v, q_l) in ((285.0, 90000.0, 0.010, 1.0e-3),
                                     (300.0, 100000.0, 0.020, 0.0),
                                     (250.0, 50000.0, 0.001, 5.0e-4))
            rho_d = p_Pa / ((Rd + q_v * Rv) * Tk)
            C_vt = Cvd + q_v * Cvv + q_l * Cl
            R_m = Rd + q_v * Rv
            C_pt = C_vt + R_m
            delta = 1.0e-7                               # condensed mass [kg/m^3]
            dT = (L_v(Tk) - Rv * Tk) * delta / (rho_d * C_vt)
            dp = (R_m / C_vt) * (L_v(Tk) - Rv * C_pt * Tk / R_m) * delta
            drho_vs = rho_v_sat(Tk + dT, (p_Pa + dp) / 100.0) - rho_v_sat(Tk, p_Pa / 100.0)
            Q_s = Scythe.Q_s_energy(Tk, p_Pa, rho_d, q_v, q_l)
            @test drho_vs ≈ Q_s * delta rtol = 1e-4
            @test Q_s > 0.0
        end
    end

    # ──────────────────────────────────────────────
    # 4. Condensation rate limiter
    # ──────────────────────────────────────────────
    @testset "qss_condensation_rate limits" begin
        Tk = 285.0; p_hPa = 900.0; ts = 0.1
        rho_vs = rho_v_sat(Tk, p_hPa)
        rho_d = (100.0 * p_hPa - Rv * Tk * rho_vs) / (Rd * Tk)
        Q_s = Scythe.Q_s_energy(Tk, 100.0 * p_hPa, rho_d, rho_vs / rho_d, 1.0e-3)

        # Subsaturated, cloud-free: no droplets to evaporate -> zero
        @test Scythe.qss_condensation_rate(-0.5 * rho_vs, 0.5 * rho_vs, 0.0, rho_d, Tk,
                                           p_hPa, Q_s, ts) == 0.0
        # Strongly subsaturated with a little cloud: evaporation clamped by rho_c/ts
        rho_c = 1.0e-6 * rho_d
        rate = Scythe.qss_condensation_rate(-0.5 * rho_vs, 0.5 * rho_vs, rho_c, rho_d, Tk,
                                            p_hPa, Q_s, ts)
        @test rate ≈ -rho_c / ts
        # Mildly subsaturated with plenty of cloud: physical evaporation, not clamped
        rho_c = 2.0e-3 * rho_d
        rate = Scythe.qss_condensation_rate(-1.0e-4 * rho_vs, (1.0 - 1.0e-4) * rho_vs, rho_c,
                                            rho_d, Tk, p_hPa, Q_s, ts)
        @test -rho_c / ts < rate < 0.0
        # Supersaturated with cloud: condensation
        @test Scythe.qss_condensation_rate(1.0e-3 * rho_vs, (1.0 + 1.0e-3) * rho_vs, rho_c,
                                           rho_d, Tk, p_hPa, Q_s, ts) > 0.0
        # Supersaturated, cloud-free: Twomey nucleation kicks in
        @test Scythe.qss_condensation_rate(1.0e-3 * rho_vs, (1.0 + 1.0e-3) * rho_vs, 0.0,
                                           rho_d, Tk, p_hPa, Q_s, ts) > 0.0
        # Nearly saturated, cloud-free (below nucleation threshold): zero
        @test Scythe.qss_condensation_rate(1.0e-5 * rho_vs, (1.0 + 1.0e-5) * rho_vs, 0.0,
                                           rho_d, Tk, p_hPa, Q_s, ts) == 0.0
        # Dry air with spurious positive Q_ss drift (no actual vapor): the clamped
        # rho_v = 0 kills phantom condensation entirely
        @test Scythe.qss_condensation_rate(0.5 * rho_vs, 0.0, 0.0, rho_d, Tk, p_hPa,
                                           Q_s, ts) == 0.0
    end

    @testset "qss_condensation_rates cloud/rain split" begin
        Tk = 285.0; p_hPa = 900.0; ts = 0.1
        N_r = 1.0e-3   # #/cm^3
        rho_vs = rho_v_sat(Tk, p_hPa)
        rho_d = (100.0 * p_hPa - Rv * Tk * rho_vs) / (Rd * Tk)
        Q_s = Scythe.Q_s_energy(Tk, 100.0 * p_hPa, rho_d, rho_vs / rho_d, 1.0e-3)

        # With no rain (or no rain channel), the cloud rate is BIT-identical to the
        # single-category closure across the branch space.
        for (Q_ss, rho_v, rho_c) in (
                (-0.5 * rho_vs, 0.5 * rho_vs, 0.0),
                (-0.5 * rho_vs, 0.5 * rho_vs, 1.0e-6 * rho_d),
                (-1.0e-4 * rho_vs, (1.0 - 1.0e-4) * rho_vs, 2.0e-3 * rho_d),
                (1.0e-3 * rho_vs, (1.0 + 1.0e-3) * rho_vs, 2.0e-3 * rho_d),
                (1.0e-3 * rho_vs, (1.0 + 1.0e-3) * rho_vs, 0.0),
                (1.0e-5 * rho_vs, (1.0 + 1.0e-5) * rho_vs, 0.0),
                (0.5 * rho_vs, 0.0, 0.0))
            old = Scythe.qss_condensation_rate(Q_ss, rho_v, rho_c, rho_d, Tk, p_hPa, Q_s, ts)
            c0, r0 = Scythe.qss_condensation_rates(Q_ss, rho_v, rho_c, 0.0, rho_d, Tk,
                                                   p_hPa, Q_s, ts, N_r)
            @test c0 === old
            @test r0 === 0.0
            cN, rN = Scythe.qss_condensation_rates(Q_ss, rho_v, rho_c, 1.0e-3, rho_d, Tk,
                                                   p_hPa, Q_s, ts, 0.0)
            @test cN === old
            @test rN === 0.0
        end

        # Seamless split: with cloud AND rain, the unlimited rates divide in proportion
        # to the channel timescales, both directions.
        rho_c = 2.0e-3 * rho_d
        rho_r = 1.0e-3
        q_c = rho_c / rho_d
        invtau_c = Scythe.invtau_condensation(Tk, p_hPa, 100.0,
                                              Scythe.cloud_droplet_radius(100.0, q_c, rho_d))
        invtau_r = Scythe.invtau_rain(Tk, p_hPa, N_r, rho_r)
        for Q_ss in (1.0e-3 * rho_vs, -1.0e-4 * rho_vs)
            rho_v = rho_vs + Q_ss
            Qc, Qr = Scythe.qss_condensation_rates(Q_ss, rho_v, rho_c, rho_r, rho_d, Tk,
                                                   p_hPa, Q_s, ts, N_r)
            @test sign(Qc) == sign(Q_ss)
            @test sign(Qr) == sign(Q_ss)
            @test Qc / Qr ≈ invtau_c / invtau_r
            @test Qc + Qr ≈ Q_ss * (invtau_c + invtau_r) / (1.0 + Q_s)
        end

        # Rain evaporates in subsaturated cloud-free air (the O01 Qevap analogue). The
        # relaxation timescale is ~10 min, so a model step never clamps; the physical
        # rate is negative and bounded by the available rain.
        Qc, Qr = Scythe.qss_condensation_rates(-0.5 * rho_vs, 0.5 * rho_vs, 0.0, 1.0e-3,
                                               rho_d, Tk, p_hPa, Q_s, ts, N_r)
        @test Qc == 0.0
        @test -1.0e-3 / ts < Qr < 0.0
        @test Qr ≈ -0.5 * rho_vs * Scythe.invtau_rain(Tk, p_hPa, N_r, 1.0e-3) / (1.0 + Q_s)
        # With a long enough step the rho_r/ts clamp engages: no negative rain
        ts_long = 1.0e6
        Qc, Qr = Scythe.qss_condensation_rates(-0.5 * rho_vs, 0.5 * rho_vs, 0.0, 1.0e-3,
                                               rho_d, Tk, p_hPa, Q_s, ts_long, N_r)
        @test Qc == 0.0
        @test Qr ≈ -1.0e-3 / ts_long

        # No negative rain: evaporation never exceeds the available rho_r even when
        # cloud has more to give.
        Qc, Qr = Scythe.qss_condensation_rates(-0.5 * rho_vs, 0.5 * rho_vs, rho_c, 1.0e-9,
                                               rho_d, Tk, p_hPa, Q_s, ts, N_r)
        @test Qr == 0.0                # below RHO_R_MIN: channel inactive
        Qc, Qr = Scythe.qss_condensation_rates(-0.5 * rho_vs, 0.5 * rho_vs, rho_c, 2.0e-8,
                                               rho_d, Tk, p_hPa, Q_s, ts, N_r)
        @test Qr >= -2.0e-8 / ts

        # Vapor cap rescales both channels proportionally: tiny vapor, strong drive
        rho_v_tiny = 1.0e-9
        Qc, Qr = Scythe.qss_condensation_rates(0.5 * rho_vs, rho_v_tiny, rho_c, rho_r,
                                               rho_d, Tk, p_hPa, Q_s, ts, N_r)
        @test Qc + Qr ≈ rho_v_tiny / ts
        @test Qc / Qr ≈ invtau_c / invtau_r
    end

    @testset "Marshall-Palmer keyword selects the rain-channel closure" begin
        Tk = 285.0; p_hPa = 900.0; ts = 0.1
        N_r = 1.0e-3   # #/cm^3 (monodisperse)
        N_0 = 8.0e6    # m^-4 (MP intercept)
        rho_vs = rho_v_sat(Tk, p_hPa)
        rho_d = (100.0 * p_hPa - Rv * Tk * rho_vs) / (Rd * Tk)
        Q_s = Scythe.Q_s_energy(Tk, 100.0 * p_hPa, rho_d, rho_vs / rho_d, 1.0e-3)
        rho_c = 2.0e-3 * rho_d
        rho_r = 1.0e-3
        invtau_c = Scythe.invtau_condensation(Tk, p_hPa, 100.0,
                                              Scythe.cloud_droplet_radius(100.0, rho_c / rho_d, rho_d))
        invtau_mp = Scythe.invtau_rain_mp(Tk, p_hPa, N_0, rho_r, rho_d)

        for Q_ss in (1.0e-3 * rho_vs, -1.0e-4 * rho_vs)
            rho_v = rho_vs + Q_ss
            # Default keyword (N_0 = 0) is BIT-identical to the monodisperse call
            base = Scythe.qss_condensation_rates(Q_ss, rho_v, rho_c, rho_r, rho_d, Tk,
                                                 p_hPa, Q_s, ts, N_r)
            kw = Scythe.qss_condensation_rates(Q_ss, rho_v, rho_c, rho_r, rho_d, Tk,
                                               p_hPa, Q_s, ts, N_r; N_0=0.0)
            @test kw[1] === base[1]
            @test kw[2] === base[2]

            # N_0 > 0 swaps ONLY the rain-channel timescale to MP; the cloud channel
            # and the proportional split arithmetic are untouched.
            Qc, Qr = Scythe.qss_condensation_rates(Q_ss, rho_v, rho_c, rho_r, rho_d, Tk,
                                                   p_hPa, Q_s, ts, N_r; N_0=N_0)
            @test Qc / Qr ≈ invtau_c / invtau_mp
            @test Qc + Qr ≈ Q_ss * (invtau_c + invtau_mp) / (1.0 + Q_s)
        end

        # The cloud gate applies identically to the MP channel: no deposition on rain
        # in supersaturated cloud-free air; evaporation stays unconditional.
        Q_ss = 1.0e-3 * rho_vs
        Qc, Qr = Scythe.qss_condensation_rates(Q_ss, rho_vs + Q_ss, 0.0, rho_r,
                                               rho_d, Tk, p_hPa, Q_s, ts, N_r; N_0=N_0)
        @test Qr === 0.0
        Qc, Qr = Scythe.qss_condensation_rates(-0.5 * rho_vs, 0.5 * rho_vs, 0.0, rho_r,
                                               rho_d, Tk, p_hPa, Q_s, ts, N_r; N_0=N_0)
        @test Qc == 0.0
        @test Qr ≈ -0.5 * rho_vs * invtau_mp / (1.0 + Q_s)
    end

    @testset "rain condensation is gated on cloud presence" begin
        # The physical pathway to rain is condensation -> cloud -> autoconversion:
        # direct vapor deposition onto rain in CLOUD-FREE air is unphysically fast
        # under the monodisperse fixed-N_r closure (rate ∝ rho_r^{1/3}, non-Lipschitz
        # at zero), and is exactly the O01 spurious-blob pathway (ringing-seeded rain
        # growing in wave-driven supersaturation at the lid). The rain channel must
        # therefore be inert for CONDENSATION unless cloud coexists; EVAPORATION in
        # subsaturated air stays unconditional.
        Tk = 285.0; p_hPa = 900.0; ts = 0.1
        N_r = 1.0e-3   # #/cm^3
        rho_vs = rho_v_sat(Tk, p_hPa)
        rho_d = (100.0 * p_hPa - Rv * Tk * rho_vs) / (Rd * Tk)
        Q_s = Scythe.Q_s_energy(Tk, 100.0 * p_hPa, rho_d, rho_vs / rho_d, 1.0e-3)
        rho_r = 1.0e-6   # well above RHO_R_MIN

        # Supersaturated cloud-free air with rain present: NO condensation onto rain;
        # nucleation routes the full rate to the cloud channel.
        Q_ss = 1.0e-3 * rho_vs
        Qc, Qr = Scythe.qss_condensation_rates(Q_ss, rho_vs + Q_ss, 0.0, rho_r,
                                               rho_d, Tk, p_hPa, Q_s, ts, N_r)
        @test Qr === 0.0
        @test Qc > 0.0

        # Below the nucleation threshold, cloud-free + rain: nothing condenses at all
        Q_ss = 1.0e-5 * rho_vs
        Qc, Qr = Scythe.qss_condensation_rates(Q_ss, rho_vs + Q_ss, 0.0, rho_r,
                                               rho_d, Tk, p_hPa, Q_s, ts, N_r)
        @test Qc === 0.0
        @test Qr === 0.0

        # Trace cloud below the q_c = 1e-8 existence threshold counts as cloud-free
        Q_ss = 1.0e-3 * rho_vs
        Qc, Qr = Scythe.qss_condensation_rates(Q_ss, rho_vs + Q_ss, 0.5e-8 * rho_d, rho_r,
                                               rho_d, Tk, p_hPa, Q_s, ts, N_r)
        @test Qr === 0.0
        @test Qc > 0.0

        # Cloudy air: the split is un-gated (proportional rates, both channels)
        rho_c = 2.0e-3 * rho_d
        invtau_c = Scythe.invtau_condensation(Tk, p_hPa, 100.0,
                                              Scythe.cloud_droplet_radius(100.0, rho_c / rho_d, rho_d))
        invtau_r = Scythe.invtau_rain(Tk, p_hPa, N_r, rho_r)
        Qc, Qr = Scythe.qss_condensation_rates(Q_ss, rho_vs + Q_ss, rho_c, rho_r,
                                               rho_d, Tk, p_hPa, Q_s, ts, N_r)
        @test Qr > 0.0
        @test Qc / Qr ≈ invtau_c / invtau_r

        # Subsaturated cloud-free rain evaporation is NOT gated
        Qc, Qr = Scythe.qss_condensation_rates(-0.5 * rho_vs, 0.5 * rho_vs, 0.0, rho_r,
                                               rho_d, Tk, p_hPa, Q_s, ts, N_r)
        @test Qc == 0.0
        @test Qr < 0.0
    end

    # ──────────────────────────────────────────────
    # 5. Integration: equation set on a ModelTile
    # ──────────────────────────────────────────────

    using SparseArrays

    """Saturated cloudy column exactly on the Q_ss = 0 manifold (density form)."""
    function saturated_cloudy_column_mc(z; q_l=1.0e-3)
        n = length(z)
        Tk = @. 290.0 - 0.005 * z
        p_Pa = @. 90000.0 * exp(-z / 8000.0)
        rho_v = rho_v_sat.(Tk, p_Pa ./ 100.0)
        rho_d = (p_Pa .- (Rv .* Tk .* rho_v)) ./ (Rd .* Tk)
        rho_c = q_l .* rho_d
        return (; z, Tk, p_Pa, rho_d, rho_v, rho_c)
    end

    """Dry, neutrally stable (theta = 300 K) analytic adiabat -- the Straka/BF02 base."""
    function dry_adiabatic_column_mc(z; theta0=300.0)
        n = length(z)
        exner = @. 1.0 - (Scythe.gravity * z) / (Cpd * theta0)
        Tk = theta0 .* exner
        p_Pa = @. 100000.0 * exner^(Cpd / Rd)
        rho_d = p_Pa ./ (Rd .* Tk)
        return (; z, Tk, p_Pa, rho_d, rho_v = zeros(n), rho_c = zeros(n))
    end

    function make_mc_mtile(tmpdir; num_cells=8, kDim=16, semiimplicit=true, ts=0.1,
                           dry=false, Khdiff=0.0, Kvdiff=0.0, Kvdiff_heat=nothing,
                           Kvdiff_water=0.0, tau_qss=10.0,
                           u_side_bc=DirichletBC(), precipitation=false, N_r=1.0e-3,
                           q_l=1.0e-3, alpha=0.0, z_damp=20.0e3,
                           equation_set="moist_compressible_XZ",
                           iMin=0.0, iMax=2000.0, f=0.0)
        varlist = equation_set == "moist_compressible_XZ" ? Scythe.MC_VARS :
                                                            Scythe.MC_VARS_CYL
        vars = Dict(v => i for (i, v) in enumerate(varlist))
        scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
        wall_bc = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))
        # A z-only u profile is only representable when the side walls let u be nonzero;
        # with a Dirichlet u the spline fit forces u -> 0 at x = 0, L and u_x (hence the
        # divergence) swamps any diffusive tendency.
        side_bc = merge(scalar_bc, Dict("u" => u_side_bc, "w" => DirichletBC()))
        gp = GridParameters(
            geometry = "RZ", num_cells = num_cells,
            iMin = iMin, iMax = iMax, kMin = 0.0, kMax = 2000.0, kDim = kDim,
            BCL = side_bc, BCR = side_bc, BCB = wall_bc, BCT = wall_bc, vars = vars,
        )
        patch = createGrid(gp)
        gridpoints = Scythe.getGridpoints(patch)
        z = gridpoints[1:kDim, 2]
        col = dry ? dry_adiabatic_column_mc(z) : saturated_cloudy_column_mc(z; q_l)
        ref_file = joinpath(tmpdir, "mc_pressure.ref")
        Scythe.write_exact_ref_mc(ref_file, z, col.p_Pa, col.rho_d, col.rho_v, col.rho_c)
        model = ModelParameters(
            ts = ts, integration_time = 1.0, output_interval = 1.0,
            equation_set = equation_set,
            ref_state_file = ref_file, grid_params = gp,
            physical_params = Dict(:Khdiff => Khdiff, :Kvdiff => Kvdiff,
                                   :Kvdiff_heat => (Kvdiff_heat === nothing ? Kvdiff : Kvdiff_heat),
                                   :Kvdiff_water => Kvdiff_water,
                                   :Kv_mudiff => 0.0, :tau_qss => tau_qss, :N_r => N_r,
                                   :alpha => alpha, :z_damp => z_damp, :f => f),
            options = Dict(:semiimplicit => semiimplicit, :exact_reference_state => true,
                           :precipitation => precipitation, :vertical_mixing => false),
        )
        patch.physical .= 0.0
        spectralTransform!(patch)
        gridTransform!(patch)
        haloReceiveMap = sparse(Int64[], Int64[], Float64[],
                                size(patch.spectral, 1), size(patch.spectral, 2))
        mtile = createModelTile(patch, patch, model, haloReceiveMap)
        return mtile, patch, model, col
    end

    """Advance every column of `mtile` for `nsteps` steps, refreshing the transforms."""
    function step_mc!(mtile, patch, model, nsteps)
        kDim = model.grid_params.kDim
        ncols = div(size(patch.physical, 1), kDim)
        for t in 1:nsteps
            for c in 1:ncols
                Scythe.advance_column(mtile, c, t)
            end
            Scythe.calcTendency(mtile)
            gridTransform!(patch)
        end
        return patch
    end

    @testset "reference routing" begin
        mktempdir() do tmpdir
            mtile, patch, model, col = make_mc_mtile(tmpdir)
            rs = mtile.ref_state
            @test rs isa Springsteel.PressureReferenceState
            @test 250.0 < sqrt(Springsteel.sound_speed_sq(rs)) < 400.0
            # Saturated base: Q_ssbar = 0 identically before smoothing
            @test maximum(abs.(Springsteel.ref_qss(rs)[:, 1])) < 1e-10
        end
    end

    @testset "resting cloudy base preserved" begin
        mktempdir() do tmpdir
            mtile, patch, model, col = make_mc_mtile(tmpdir)
            kDim = model.grid_params.kDim
            ncols = div(size(patch.physical, 1), kDim)
            for c in 1:ncols
                Scythe.advance_column(mtile, c, 1)
            end
            # Per-slot tendency tolerances scaled to the slot magnitudes
            # (p ~ 1e5 Pa, E_t ~ 2e8 J/m^3, densities ~ 1)
            scales = Dict(1 => 1.0e5, 2 => 1.0, 3 => 1.0, 4 => 1.0, 5 => 1.0,
                          6 => 2.0e8, 7 => 1.0e-2, 8 => 1.0)
            for v in 1:8
                @test maximum(abs.(mtile.expdot_n[:, v])) / scales[v] < 1.0e-9
                @test maximum(abs.(mtile.var_np1[:, v])) / scales[v] < 1.0e-9
            end
            @test all(isfinite.(mtile.var_np1))
        end
    end

    @testset "condensation closure at rest" begin
        mktempdir() do tmpdir
            mtile, patch, model, col = make_mc_mtile(tmpdir)
            kDim = model.grid_params.kDim
            vars = model.grid_params.vars
            qss_i = vars["Q_ss"]

            # Supersaturate the lower half of every column by a small, realistic amount
            rho_dbar = Springsteel.ref_rho_d(mtile.ref_state)[:, 1]
            dq = 5.0e-5
            npts = size(patch.physical, 1)
            for i in 1:npts
                k = mod1(i, kDim)
                if k <= div(kDim, 2)
                    patch.physical[i, qss_i, 1] = rho_dbar[k] * dq
                end
            end
            spectralTransform!(patch)
            gridTransform!(patch)

            ncols = div(npts, kDim)
            for c in 1:ncols
                Scythe.advance_column(mtile, c, 1)
            end

            perturbed = [i for i in 1:npts if mod1(i, kDim) <= div(kDim, 2)]
            # Exact first law: no condensation source in E_t or rho_t at rest
            @test maximum(abs.(mtile.expdot_n[:, vars["E_t"]])) == 0.0
            @test maximum(abs.(mtile.expdot_n[:, vars["rho_t"]])) == 0.0
            @test maximum(abs.(mtile.expdot_n[:, vars["rho_d"]])) == 0.0
            # Latent heating raises pressure; supersaturation relaxes
            @test all(mtile.expdot_n[perturbed, vars["p"]] .> 0.0)
            @test all(mtile.expdot_n[perturbed, qss_i] .< 0.0)
            @test all(isfinite.(mtile.var_np1))
        end
    end

    @testset "post-step retrieval physical" begin
        mktempdir() do tmpdir
            mtile, patch, model, col = make_mc_mtile(tmpdir)
            kDim = model.grid_params.kDim
            vars = model.grid_params.vars
            # Small warm perturbation via E_t' in the lower half
            E_tbar = Springsteel.ref_total_energy(mtile.ref_state)[:, 1]
            npts = size(patch.physical, 1)
            for i in 1:npts
                k = mod1(i, kDim)
                if k <= div(kDim, 2)
                    patch.physical[i, vars["E_t"], 1] = 1.0e-4 * E_tbar[k]
                end
            end
            spectralTransform!(patch)
            gridTransform!(patch)
            ncols = div(npts, kDim)
            for c in 1:ncols
                Scythe.advance_column(mtile, c, 1)
            end
            # Re-retrieve T from the advanced state
            rs = mtile.ref_state
            pbar = Springsteel.ref_pressure(rs)[:, 1]
            rho_dbar = Springsteel.ref_rho_d(rs)[:, 1]
            rho_tbar = Springsteel.ref_rho_t(rs)[:, 1]
            Q_ssbar = Springsteel.ref_qss(rs)[:, 1]
            Tbar = Springsteel.reference_temperature(rs)
            z = Scythe.getGridpoints(patch)[1:kDim, 2]
            for i in 1:npts
                k = mod1(i, kDim)
                v = mtile.var_np1
                p = v[i, vars["p"]] + pbar[k]
                rho_d = v[i, vars["rho_d"]] + rho_dbar[k]
                rho_t = v[i, vars["rho_t"]] + rho_tbar[k]
                E_t = v[i, vars["E_t"]] + E_tbar[k]
                Q_ss = v[i, vars["Q_ss"]] + Q_ssbar[k]
                ke = 0.5 * (v[i, vars["u"]]^2 + v[i, vars["w"]]^2)
                M = p + E_t - rho_t * (ke + Scythe.gravity * z[k])
                Tk = Scythe.retrieve_temperature(M, rho_d, rho_t, Q_ss, p, Tbar[k])
                @test isfinite(Tk) && 200.0 < Tk < 320.0
            end
        end
    end

    @testset "semi-implicit acoustic stability" begin
        mktempdir() do tmpdir
            # ts = 0.1 s is ~7x the explicit VERTICAL acoustic limit for the
            # boundary-clustered Chebyshev levels (kDim=32: dz_min ~ 5 m, c ~ 340 m/s
            # => ~0.015 s; the fully explicit scheme NaNs by step ~19 at this ts) while
            # staying below the HORIZONTAL limit (dx ~ 80 m), since the semi-implicit
            # adjustment is vertical-only.
            mtile, patch, model, col = make_mc_mtile(tmpdir; semiimplicit=true,
                                                     ts=0.1, kDim=32)
            kDim = model.grid_params.kDim
            vars = model.grid_params.vars
            p_i = vars["p"]; rhot_i = vars["rho_t"]
            npts = size(patch.physical, 1)
            gridpoints = Scythe.getGridpoints(patch)

            # Small pressure pulse in the domain center
            for i in 1:npts
                x = gridpoints[i, 1]; z = gridpoints[i, 2]
                L = sqrt(((x - 1000.0) / 500.0)^2 + ((z - 1000.0) / 500.0)^2)
                patch.physical[i, p_i, 1] = L <= 1.0 ? 10.0 * (cos(pi * L / 2.0))^2 : 0.0
            end
            spectralTransform!(patch)
            gridTransform!(patch)

            p0_max = maximum(abs.(patch.physical[:, p_i, 1]))
            ncols = div(npts, kDim)
            for t in 1:200
                for c in 1:ncols
                    Scythe.advance_column(mtile, c, t)
                end
                Scythe.calcTendency(mtile)
                gridTransform!(patch)
            end
            @test all(isfinite.(patch.physical[:, :, 1]))
            # Acoustic energy must not grow: the dispersing pulse DECAYS below its
            # initial amplitude after 200 vertically-stiff steps
            @test maximum(abs.(patch.physical[:, p_i, 1])) < p0_max
            @test maximum(abs.(patch.physical[:, vars["w"], 1])) < 1.0
        end
    end

    @testset "SI vertical-acoustic ceiling removed (dry isothermal, RiRk)" begin
        # Regression gate for the operator-consistent AI2* staging: the old
        # subtract-AB3 scheme mixed the pointwise product-rule acoustic operator
        # (history levels) with the fitted Galerkin operator (implicit level), leaving
        # a grid-scale residual under explicit weights that blew up above a VERTICAL
        # acoustic Courant of ~2.1-2.4 on the RiRk spline vertical (grid-scale,
        # top-boundary-localized mode; tc/SI_VERTICAL_CEILING.md). With the
        # spline-consistent history staging (single fitted-φ chain for the slaved
        # legs, stored applied increment for the w leg) the measured ceiling moves to
        # Co_z ≈ 9-18, past the explicit HORIZONTAL acoustic limit of any realistic
        # grid aspect ratio: a broadband w seed on a resting, statically stable,
        # bone-dry isothermal base must DECAY at the previously-fatal Co_z ≈ 4.5 and
        # 9.0 (the old scheme e-folded in ~65 s at Co_z 2.4 and NaN'd within ~50 s at
        # Co_z 4.5). Horizontal cells widen with ts so the explicit horizontal
        # acoustic Courant stays <= 0.35 and cannot bind.
        # The stratified case guards the reference-state (SHB78) instability on top
        # of the operator-consistency one: with a Dunion-like lapse to a 195-K
        # tropopause, a DOMAIN-MEAN Pxi in the acoustic linearization left the
        # local c² deviation explicit and NaN'd at Co_z 9 within 240 s — the local
        # Pxi_prof profile must keep it decaying.
        function ceiling_column_mc(z; kind, p0=101325.0)
            n = length(z)
            if kind == :isothermal
                T0 = 250.0
                Tk = fill(T0, n)
                p_Pa = @. p0 * exp(-gravity * z / (Rd * T0))
            else
                gam = (300.0 - 195.0) / 17000.0
                Tk = [zz <= 17000.0 ? 300.0 - gam * zz :
                      195.0 + 2.0e-3 * (zz - 17000.0) for zz in z]
                p_trop = p0 * (195.0 / 300.0)^(gravity / (Rd * gam))
                p_Pa = [zz <= 17000.0 ?
                        p0 * ((300.0 - gam * zz) / 300.0)^(gravity / (Rd * gam)) :
                        p_trop * ((195.0 + 2.0e-3 * (zz - 17000.0)) / 195.0)^(-gravity / (Rd * 2.0e-3))
                        for zz in z]
            end
            return (; z, Tk, p_Pa, rho_d = p_Pa ./ (Rd .* Tk),
                    rho_v = zeros(n), rho_c = zeros(n))
        end
        for (kind, ts) in ((:isothermal, 0.75), (:isothermal, 1.5), (:stratified, 1.5))
            # Co_z ≈ 4.5, 9.0 on dz_min = 0.2254 * 250 m, c ≈ 340 m/s
            mktempdir() do tmpdir
                dx_cell = max(3200.0, 340.0 * ts / (0.35 * 0.2254))
                vars = Dict(v => i for (i, v) in enumerate(Scythe.MC_VARS))
                scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
                side_bc = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))
                wall_bc = merge(scalar_bc, Dict("w" => DirichletBC()))
                kMax = kind == :isothermal ? 10.0e3 : 25.0e3       # dz_cell = 250 m
                gp = GridParameters(geometry = "RiRk",
                    iMin = 0.0, iMax = 4.0 * dx_cell, num_cells_i = 4,
                    kMin = 0.0, kMax = kMax, num_cells_k = round(Int, kMax / 250.0),
                    BCL = side_bc, BCR = side_bc, BCB = wall_bc, BCT = wall_bc,
                    vars = vars)
                ref_file = joinpath(tmpdir, "ceiling_pressure.ref")
                model = ModelParameters(
                    ts = ts, integration_time = 600.0, output_interval = 600.0,
                    equation_set = "moist_compressible_XZ",
                    ref_state_file = ref_file, grid_params = gp,
                    physical_params = Dict(:Khdiff => 0.0, :Kvdiff => 0.0,
                                           :Kvdiff_heat => 0.0, :Kvdiff_water => 0.0,
                                           :tau_qss => 10.0, :alpha => 0.0,
                                           :z_damp => 20.0e3, :f => 0.0),
                    options = Dict(:semiimplicit => true,
                                   :exact_reference_state => true,
                                   :precipitation => false))
                gp = model.grid_params
                patch = createGrid(gp)
                gridpoints = Scythe.getGridpoints(patch)
                kDim = gp.kDim
                z = gridpoints[1:kDim, end]
                col = ceiling_column_mc(z; kind)
                Scythe.write_exact_ref_mc(ref_file, z, col.p_Pa, col.rho_d,
                                          col.rho_v, col.rho_c)
                patch.physical .= 0.0
                # Deterministic broadband w seed (irrational chirp sweeps every
                # vertical wavenumber, grid scale included)
                w_i = gp.vars["w"]
                npts = size(patch.physical, 1)
                for i in 1:npts
                    patch.physical[i, w_i, 1] = 1.0e-4 * sin(0.5 * sqrt(2.0) * i^2)
                end
                spectralTransform!(patch)
                gridTransform!(patch)
                haloReceiveMap = sparse(Int64[], Int64[], Float64[],
                                        size(patch.spectral, 1),
                                        size(patch.spectral, 2))
                mtile = createModelTile(patch, patch, model, haloReceiveMap)

                w0_max = maximum(abs.(patch.physical[:, w_i, 1]))
                ncols = div(npts, kDim)
                nsteps = round(Int, 600.0 / ts)
                for t in 1:nsteps
                    for c in 1:ncols
                        Scythe.advance_column(mtile, c, t)
                    end
                    Scythe.calcTendency(mtile)
                    gridTransform!(patch)
                end
                @test all(isfinite.(patch.physical[:, :, 1]))
                # The seed must decay, not grow (the old scheme grew 1e-4 -> ~1 m/s
                # here); the stable reference behavior is a x40-50 decay over 600 s.
                @test maximum(abs.(patch.physical[:, w_i, 1])) < w0_max
            end
        end
    end

    @testset "SI horizontal-acoustic ceiling removed (resting XZ, RiRk)" begin
        # Regression gate for the Phase-1 horizontal semi-implicit
        # (options[:horizontal_semiimplicit], src/horizontal_si.jl): a broadband
        # u+w seed on a resting base must DECAY over 300 s at a horizontal
        # acoustic Courant of 3.0 on dx_min — 4x past the explicit AB3 limit
        # (flag-off control blows up above Co_h ≈ 0.7; the measured flag-on
        # envelope with the default hsi_u_history = "none" is clean decay
        # through Co_h 3, marginal at 4.5 on the stratified base, unstable at
        # 9 — model_tests/hsi_ceiling_sweep.jl). The vertical is 500-m cells so
        # this is simultaneously a combined-Courant case (Co_z ≈ 3.0 x
        # Co_h 3.0). Stratified case included per the SHB78 lesson: an
        # isothermal base cannot see reference-state errors in the
        # linearization coefficients.
        for kind in (:isothermal, :stratified)
            mktempdir() do tmpdir
                ts = 1.0
                co_h = 3.0
                dx_cell = 340.0 * ts / (co_h * 0.2254)
                vars = Dict(v => i for (i, v) in enumerate(Scythe.MC_VARS))
                scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
                side_bc = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))
                wall_bc = merge(scalar_bc, Dict("w" => DirichletBC()))
                gp = GridParameters(geometry = "RiRk",
                    iMin = 0.0, iMax = 50.0 * dx_cell, num_cells_i = 50,
                    kMin = 0.0, kMax = 25.0e3, num_cells_k = 50,
                    BCL = side_bc, BCR = side_bc, BCB = wall_bc, BCT = wall_bc,
                    vars = vars)
                ref_file = joinpath(tmpdir, "hsi_ceiling.ref")
                model = ModelParameters(
                    ts = ts, integration_time = 300.0, output_interval = 300.0,
                    equation_set = "moist_compressible_XZ",
                    ref_state_file = ref_file, grid_params = gp,
                    physical_params = Dict(:Khdiff => 0.0, :Kvdiff => 0.0,
                                           :Kvdiff_heat => 0.0, :Kvdiff_water => 0.0,
                                           :tau_qss => 10.0, :alpha => 0.0,
                                           :z_damp => 30.0e3, :f => 0.0),
                    options = Dict(:semiimplicit => true,
                                   :exact_reference_state => true,
                                   :precipitation => false,
                                   :horizontal_semiimplicit => true))
                gp = model.grid_params
                patch = createGrid(gp)
                kDim = gp.kDim
                z = Scythe.getGridpoints(patch)[1:kDim, end]
                if kind == :isothermal
                    T0 = 250.0
                    p_Pa = @. 101325.0 * exp(-gravity * z / (Rd * T0))
                else
                    gam = (300.0 - 195.0) / 17000.0
                    p_trop = 101325.0 * (195.0 / 300.0)^(gravity / (Rd * gam))
                    p_Pa = [zz <= 17000.0 ?
                            101325.0 * ((300.0 - gam * zz) / 300.0)^(gravity / (Rd * gam)) :
                            p_trop * ((195.0 + 2.0e-3 * (zz - 17000.0)) / 195.0)^(-gravity / (Rd * 2.0e-3))
                            for zz in z]
                end
                Tk = kind == :isothermal ? fill(250.0, kDim) :
                     [zz <= 17000.0 ? 300.0 - (300.0 - 195.0) / 17000.0 * zz :
                      195.0 + 2.0e-3 * (zz - 17000.0) for zz in z]
                Scythe.write_exact_ref_mc(ref_file, z, p_Pa, p_Pa ./ (Rd .* Tk),
                                          zeros(kDim), zeros(kDim))
                patch.physical .= 0.0
                u_i = gp.vars["u"]; w_i = gp.vars["w"]
                npts = size(patch.physical, 1)
                for i in 1:npts
                    patch.physical[i, u_i, 1] = 1.0e-4 * sin(0.5 * sqrt(2.0) * i^2)
                    patch.physical[i, w_i, 1] = 1.0e-4 * sin(0.4 * sqrt(3.0) * i^2 + 1.0)
                end
                spectralTransform!(patch)
                gridTransform!(patch)
                haloReceiveMap = sparse(Int64[], Int64[], Float64[],
                                        size(patch.spectral, 1),
                                        size(patch.spectral, 2))
                mtile = createModelTile(patch, patch, model, haloReceiveMap)
                hsd = Scythe.create_horizontal_solve_data(patch, model,
                    mtile.mc_ref_diag.Pxi_prof,
                    collect(view(Springsteel.ref_rho_t(mtile.ref_state), :, 1)),
                    collect(view(Springsteel.ref_rho_d(mtile.ref_state), :, 1)),
                    collect(view(Springsteel.ref_total_energy(mtile.ref_state), :, 1) .+
                            view(Springsteel.ref_pressure(mtile.ref_state), :, 1)))
                seed = max(maximum(abs.(patch.physical[:, u_i, 1])),
                           maximum(abs.(patch.physical[:, w_i, 1])))
                ncols = div(npts, kDim)
                u_incr = zeros(npts, 6)
                for t in 1:round(Int, 300.0 / ts)
                    if t > 1
                        Scythe.horizontal_si_load_increment!(mtile, u_incr, t, 1)
                    end
                    for c in 1:ncols
                        Scythe.advance_column(mtile, c, t)
                    end
                    Scythe.calcTendency(mtile)
                    u_incr .= Scythe.horizontal_si_correct!(patch.spectral, patch,
                                                            model, hsd, t)
                    gridTransform!(patch)
                end
                @test all(isfinite.(patch.physical[:, :, 1]))
                amp = max(maximum(abs.(patch.physical[:, u_i, 1])),
                          maximum(abs.(patch.physical[:, w_i, 1])))
                @test amp < seed
            end
        end
    end

    @testset "exact SI (unsplit 2-D solve, RiRk XZ)" begin
        # Stage-2 gates for options[:exact_si] (src/exact_si.jl): the unsplit
        # p′-form weighted-mass 2-D Helmholtz replacing the per-column vertical
        # solve, every fast leg recovered from the one solved coefficient set
        # (reference/exact_si_derivation.md; von Neumann part 3(a) — the
        # ε-insensitive consistent-weak composition).
        function xsi_build(tmpdir; opts_extra=Dict{Symbol,Any}(), dx_cell=503.0,
                           dz_cells=50, ts=1.0, kind=:isothermal, seed_u=true)
            vars = Dict(v => i for (i, v) in enumerate(Scythe.MC_VARS))
            scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
            side_bc = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))
            wall_bc = merge(scalar_bc, Dict("w" => DirichletBC()))
            gp = GridParameters(geometry = "RiRk",
                iMin = 0.0, iMax = 50.0 * dx_cell, num_cells_i = 50,
                kMin = 0.0, kMax = 25.0e3, num_cells_k = dz_cells,
                BCL = side_bc, BCR = side_bc, BCB = wall_bc, BCT = wall_bc,
                vars = vars)
            ref_file = joinpath(tmpdir, "xsi_$(kind)_$(dz_cells)_$(round(dx_cell)).ref")
            model = ModelParameters(
                ts = ts, integration_time = 600.0, output_interval = 600.0,
                equation_set = "moist_compressible_XZ",
                ref_state_file = ref_file, grid_params = gp,
                physical_params = Dict(:Khdiff => 0.0, :Kvdiff => 0.0,
                                       :Kvdiff_heat => 0.0, :Kvdiff_water => 0.0,
                                       :tau_qss => 10.0, :alpha => 0.0,
                                       :z_damp => 30.0e3, :f => 0.0),
                options = merge(Dict{Symbol,Any}(:semiimplicit => true,
                                :exact_reference_state => true,
                                :precipitation => false), opts_extra))
            gp = model.grid_params
            patch = createGrid(gp)
            z = Scythe.getGridpoints(patch)[1:gp.kDim, end]
            if kind == :isothermal
                T0 = 250.0
                p_Pa = @. 101325.0 * exp(-gravity * z / (Rd * T0))
                rho_prof = p_Pa ./ (Rd * T0)
            else
                gam = (300.0 - 195.0) / 17000.0
                p_trop = 101325.0 * (195.0 / 300.0)^(gravity / (Rd * gam))
                Tk = [zz <= 17000.0 ? 300.0 - gam * zz :
                      195.0 + 2.0e-3 * (zz - 17000.0) for zz in z]
                p_Pa = [zz <= 17000.0 ?
                        101325.0 * ((300.0 - gam * zz) / 300.0)^(gravity / (Rd * gam)) :
                        p_trop * ((195.0 + 2.0e-3 * (zz - 17000.0)) / 195.0)^(-gravity / (Rd * 2.0e-3))
                        for zz in z]
                rho_prof = p_Pa ./ (Rd .* Tk)
            end
            Scythe.write_exact_ref_mc(ref_file, z, p_Pa, rho_prof,
                                      zeros(gp.kDim), zeros(gp.kDim))
            patch.physical .= 0.0
            u_i = gp.vars["u"]; w_i = gp.vars["w"]
            for i in 1:size(patch.physical, 1)
                seed_u && (patch.physical[i, u_i, 1] = 1.0e-4 * sin(0.5 * sqrt(2.0) * i^2))
                patch.physical[i, w_i, 1] = 1.0e-4 * sin(0.4 * sqrt(3.0) * i^2 + 1.0)
            end
            spectralTransform!(patch)
            gridTransform!(patch)
            haloReceiveMap = sparse(Int64[], Int64[], Float64[],
                                    size(patch.spectral, 1), size(patch.spectral, 2))
            mtile = Scythe.createModelTile(patch, patch, model, haloReceiveMap)
            esd = nothing
            if get(model.options, :exact_si, false) === true
                esd = Scythe.create_exact_si_data(patch, model,
                    mtile.mc_ref_diag.Pxi_prof,
                    collect(view(Springsteel.ref_rho_t(mtile.ref_state), :, 1:2)),
                    collect(view(Springsteel.ref_rho_d(mtile.ref_state), :, 1:2)),
                    collect(view(Springsteel.ref_total_energy(mtile.ref_state), :, 1:2) .+
                            view(Springsteel.ref_pressure(mtile.ref_state), :, 1:2)))
            end
            return mtile, patch, model, gp, esd
        end
        function xsi_run!(mtile, patch, model, gp, esd; nsteps)
            kDim = gp.kDim
            u_i = gp.vars["u"]; w_i = gp.vars["w"]; p_i = gp.vars["p"]
            npts = size(patch.physical, 1)
            ncols = div(npts, kDim)
            xsi = get(model.options, :exact_si, false) === true
            xstar = zeros(npts, 3)
            hfields = zeros(npts, Scythe.XSI_NPLANES)
            for t in 1:nsteps
                xsi && t > 1 && Scythe.exact_si_load_history!(mtile, hfields, t, 1)
                for c in 1:ncols
                    Scythe.advance_column(mtile, c, t)
                end
                if xsi
                    xstar[:, 1] .= view(mtile.var_np1, :, u_i)
                    xstar[:, 2] .= view(mtile.var_np1, :, w_i)
                    xstar[:, 3] .= view(mtile.var_np1, :, p_i)
                    Scythe.exact_si_solve!(hfields, esd, patch, model, t, xstar)
                    for c in 1:ncols
                        cs = (c - 1) * kDim + 1
                        Scythe.exact_si_apply_column!(mtile, cs, cs + kDim - 1, t,
                                                      hfields, 1)
                    end
                end
                Scythe.calcTendency(mtile)
                gridTransform!(patch)
            end
        end

        mktempdir() do tmpdir
            # 1. Flag validation: the structurally incompatible combinations
            # error at createModelTile time.
            for bad in (Dict{Symbol,Any}(:exact_si => true, :state_dependent_si => true),
                        Dict{Symbol,Any}(:exact_si => true, :horizontal_semiimplicit => true))
                @test_throws ErrorException xsi_build(tmpdir; opts_extra=bad)
            end

            # 2. A≡0 bypass equivalence: with the horizontal coupling zeroed
            # (options[:exact_si_zero_x]) the two-phase exact-SI path must
            # reproduce the production vertical-only path to ≤ 1e-10 (measured
            # bitwise 0.0) — the per-column vertical solve runs on identical
            # data with identical operations, only split across the phases.
            mref, pref, modref, gpref, _ = xsi_build(tmpdir)
            xsi_run!(mref, pref, modref, gpref, nothing; nsteps=20)
            mzx, pzx, modzx, gpzx, ezx = xsi_build(tmpdir;
                opts_extra=Dict{Symbol,Any}(:exact_si => true, :exact_si_zero_x => true))
            xsi_run!(mzx, pzx, modzx, gpzx, ezx; nsteps=20)
            @test maximum(abs.(pref.physical[:, :, 1] .- pzx.physical[:, :, 1])) <= 1.0e-10

            # 3. Ceiling gate: broadband u+w seed on the resting base must
            # DECAY over 300 s at Co_h 3.0 (4x past the explicit AB3 limit),
            # both bases (the SHB78 lesson), with Co_z ≈ 3 simultaneously
            # (dz 500-m cells) — the combined-Courant case. The measured
            # envelope (model_tests/hsi_ceiling_sweep.jl --exact-si) is clean
            # decay through Co_h 18 on both bases.
            for kind in (:isothermal, :stratified)
                m3, p3, mod3, gp3, e3 = xsi_build(tmpdir; kind,
                    opts_extra=Dict{Symbol,Any}(:exact_si => true))
                u_i = gp3.vars["u"]; w_i = gp3.vars["w"]
                seed = max(maximum(abs.(p3.physical[:, u_i, 1])),
                           maximum(abs.(p3.physical[:, w_i, 1])))
                xsi_run!(m3, p3, mod3, gp3, e3; nsteps=300)
                @test all(isfinite.(p3.physical[:, :, 1]))
                amp = max(maximum(abs.(p3.physical[:, u_i, 1])),
                          maximum(abs.(p3.physical[:, w_i, 1])))
                @test amp < seed
            end

            # 4. The p′-primary VERTICAL ceiling must not regress: Co_z ≈ 9 on
            # the stratified base (the SHB78 case) with the 2-D solve active
            # must decay — 250-m cells, ts 1.5, wide horizontal cells so Co_h
            # never binds (measured: decay ×226 over 600 s, matching the
            # φ-solve; this is a NEW measured quantity, not inherited).
            mv, pv, modv, gpv, ev = xsi_build(tmpdir; kind=:stratified,
                dx_cell=6800.0, dz_cells=100, ts=1.5,
                opts_extra=Dict{Symbol,Any}(:exact_si => true), seed_u=false)
            w_i = gpv.vars["w"]
            w0 = maximum(abs.(pv.physical[:, w_i, 1]))
            xsi_run!(mv, pv, modv, gpv, ev; nsteps=round(Int, 600.0 / 1.5))
            @test all(isfinite.(pv.physical[:, :, 1]))
            @test maximum(abs.(pv.physical[:, w_i, 1])) < w0
        end
    end

    @testset "exact SI (unsplit 2-D solve, axisym r–z)" begin
        # Stage-3 gates for options[:exact_si] on the axisymmetric (cylindrical
        # r–z) geometry (reference/exact_si_stage3_plan.md): the SAME unsplit
        # p′-form solve with the radial Galerkin blocks re-weighted by r (the
        # cylindrical volume element r dr dz) and the divergence load carrying
        # the radial metric u*/r. The r = 0 axis needs no explicit row — the
        # r-weight kills the by-parts axis flux (cylindrical regularity).
        function axi_build(tmpdir; opts_extra=Dict{Symbol,Any}(), dx_cell=503.0,
                           dz_cells=50, ts=1.0, kind=:isothermal, seed_u=true,
                           iMin=0.0)
            vars = Dict(v => i for (i, v) in enumerate(Scythe.MC_VARS_CYL))
            scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
            side_bc = merge(scalar_bc, Dict("u" => DirichletBC(),
                            "v" => DirichletBC(), "w" => DirichletBC()))
            wall_bc = merge(scalar_bc, Dict("w" => DirichletBC()))
            gp = GridParameters(geometry = "RiRk",
                iMin = iMin, iMax = iMin + 50.0 * dx_cell, num_cells_i = 50,
                kMin = 0.0, kMax = 25.0e3, num_cells_k = dz_cells,
                BCL = side_bc, BCR = side_bc, BCB = wall_bc, BCT = wall_bc,
                vars = vars)
            ref_file = joinpath(tmpdir, "axi_$(kind)_$(dz_cells)_$(round(dx_cell)).ref")
            model = ModelParameters(
                ts = ts, integration_time = 600.0, output_interval = 600.0,
                equation_set = "moist_compressible_axisym",
                ref_state_file = ref_file, grid_params = gp,
                physical_params = Dict(:Khdiff => 0.0, :Kvdiff => 0.0,
                                       :Kvdiff_heat => 0.0, :Kvdiff_water => 0.0,
                                       :tau_qss => 10.0, :alpha => 0.0,
                                       :z_damp => 30.0e3, :f => 0.0),
                options = merge(Dict{Symbol,Any}(:semiimplicit => true,
                                :exact_reference_state => true,
                                :precipitation => false), opts_extra))
            gp = model.grid_params
            patch = createGrid(gp)
            z = Scythe.getGridpoints(patch)[1:gp.kDim, end]
            if kind == :isothermal
                T0 = 250.0
                p_Pa = @. 101325.0 * exp(-gravity * z / (Rd * T0))
                rho_prof = p_Pa ./ (Rd * T0)
            else
                gam = (300.0 - 195.0) / 17000.0
                p_trop = 101325.0 * (195.0 / 300.0)^(gravity / (Rd * gam))
                Tk = [zz <= 17000.0 ? 300.0 - gam * zz :
                      195.0 + 2.0e-3 * (zz - 17000.0) for zz in z]
                p_Pa = [zz <= 17000.0 ?
                        101325.0 * ((300.0 - gam * zz) / 300.0)^(gravity / (Rd * gam)) :
                        p_trop * ((195.0 + 2.0e-3 * (zz - 17000.0)) / 195.0)^(-gravity / (Rd * 2.0e-3))
                        for zz in z]
                rho_prof = p_Pa ./ (Rd .* Tk)
            end
            Scythe.write_exact_ref_mc(ref_file, z, p_Pa, rho_prof,
                                      zeros(gp.kDim), zeros(gp.kDim))
            patch.physical .= 0.0
            u_i = gp.vars["u"]; w_i = gp.vars["w"]
            for i in 1:size(patch.physical, 1)
                seed_u && (patch.physical[i, u_i, 1] = 1.0e-4 * sin(0.5 * sqrt(2.0) * i^2))
                patch.physical[i, w_i, 1] = 1.0e-4 * sin(0.4 * sqrt(3.0) * i^2 + 1.0)
            end
            spectralTransform!(patch)
            gridTransform!(patch)
            haloReceiveMap = sparse(Int64[], Int64[], Float64[],
                                    size(patch.spectral, 1), size(patch.spectral, 2))
            mtile = Scythe.createModelTile(patch, patch, model, haloReceiveMap)
            esd = nothing
            if get(model.options, :exact_si, false) === true
                esd = Scythe.create_exact_si_data(patch, model,
                    mtile.mc_ref_diag.Pxi_prof,
                    collect(view(Springsteel.ref_rho_t(mtile.ref_state), :, 1:2)),
                    collect(view(Springsteel.ref_rho_d(mtile.ref_state), :, 1:2)),
                    collect(view(Springsteel.ref_total_energy(mtile.ref_state), :, 1:2) .+
                            view(Springsteel.ref_pressure(mtile.ref_state), :, 1:2)))
            end
            return mtile, patch, model, gp, esd
        end
        function axi_run!(mtile, patch, model, gp, esd; nsteps)
            kDim = gp.kDim
            u_i = gp.vars["u"]; w_i = gp.vars["w"]; p_i = gp.vars["p"]
            npts = size(patch.physical, 1)
            ncols = div(npts, kDim)
            xsi = get(model.options, :exact_si, false) === true
            xstar = zeros(npts, 3)
            hfields = zeros(npts, Scythe.XSI_NPLANES)
            for t in 1:nsteps
                xsi && t > 1 && Scythe.exact_si_load_history!(mtile, hfields, t, 1)
                for c in 1:ncols
                    Scythe.advance_column(mtile, c, t)
                end
                if xsi
                    xstar[:, 1] .= view(mtile.var_np1, :, u_i)
                    xstar[:, 2] .= view(mtile.var_np1, :, w_i)
                    xstar[:, 3] .= view(mtile.var_np1, :, p_i)
                    Scythe.exact_si_solve!(hfields, esd, patch, model, t, xstar)
                    for c in 1:ncols
                        cs = (c - 1) * kDim + 1
                        Scythe.exact_si_apply_column!(mtile, cs, cs + kDim - 1, t,
                                                      hfields, 1)
                    end
                end
                Scythe.calcTendency(mtile)
                gridTransform!(patch)
            end
        end

        mktempdir() do tmpdir
            # 1. Flag validation: the RLR set is Stage 4 and still errors; the
            # structurally incompatible flags error on the axisym set too.
            for bad in (Dict{Symbol,Any}(:exact_si => true, :state_dependent_si => true),
                        Dict{Symbol,Any}(:exact_si => true, :horizontal_semiimplicit => true))
                @test_throws ErrorException axi_build(tmpdir; opts_extra=bad)
            end

            # 2. A≡0 bypass equivalence (G3-unit): with the horizontal coupling
            # zeroed (options[:exact_si_zero_x]) the two-phase exact-SI path must
            # reproduce the production vertical-only path to ≤ 1e-10 on the resting
            # isothermal axisym tile — the vertical solve is geometry-agnostic, so
            # this holds bitwise exactly as in XZ.
            mref, pref, modref, gpref, _ = axi_build(tmpdir)
            axi_run!(mref, pref, modref, gpref, nothing; nsteps=20)
            mzx, pzx, modzx, gpzx, ezx = axi_build(tmpdir;
                opts_extra=Dict{Symbol,Any}(:exact_si => true, :exact_si_zero_x => true))
            axi_run!(mzx, pzx, modzx, gpzx, ezx; nsteps=20)
            @test maximum(abs.(pref.physical[:, :, 1] .- pzx.physical[:, :, 1])) <= 1.0e-10

            # 3. Ceiling gate (G3-stability): broadband u+w seed on the resting
            # axisym base (iMin = 0, so the r = 0 axis column is exercised) must
            # DECAY over 300 s at Co_h 3.0, both bases (the radial metric must not
            # destabilize the acoustic solve).
            for kind in (:isothermal, :stratified)
                m3, p3, mod3, gp3, e3 = axi_build(tmpdir; kind,
                    opts_extra=Dict{Symbol,Any}(:exact_si => true))
                u_i = gp3.vars["u"]; w_i = gp3.vars["w"]
                seed = max(maximum(abs.(p3.physical[:, u_i, 1])),
                           maximum(abs.(p3.physical[:, w_i, 1])))
                axi_run!(m3, p3, mod3, gp3, e3; nsteps=300)
                @test all(isfinite.(p3.physical[:, :, 1]))
                amp = max(maximum(abs.(p3.physical[:, u_i, 1])),
                          maximum(abs.(p3.physical[:, w_i, 1])))
                @test amp < seed
            end

            # 4. The p′-primary VERTICAL ceiling must not regress on the axisym
            # path: Co_z ≈ 9 stratified with the 2-D radial solve active, wide
            # radial cells (Co_h never binds), must decay — the radial re-weight
            # leaves the vertical block untouched.
            mv, pv, modv, gpv, ev = axi_build(tmpdir; kind=:stratified,
                dx_cell=6800.0, dz_cells=100, ts=1.5, iMin=50.0e3,
                opts_extra=Dict{Symbol,Any}(:exact_si => true), seed_u=false)
            w_i = gpv.vars["w"]
            w0 = maximum(abs.(pv.physical[:, w_i, 1]))
            axi_run!(mv, pv, modv, gpv, ev; nsteps=round(Int, 600.0 / 1.5))
            @test all(isfinite.(pv.physical[:, :, 1]))
            @test maximum(abs.(pv.physical[:, w_i, 1])) < w0
        end
    end

    # ──────────────────────────────────────────────
    # 6. Diffusion: theta_d, heating consistency, dissipation
    # ──────────────────────────────────────────────

    @testset "potential_temperature(p_Pa, rho_d)" begin
        # Dry air: exactly Straka's theta = T (p_0/p)^kappa
        for (Tk, p_Pa) in ((300.0, 100000.0), (280.0, 85000.0), (250.0, 50000.0))
            rho_d = p_Pa / (Rd * Tk)
            theta = Tk * ((100.0 * Scythe.p_0) / p_Pa)^(Rd / Cpd)
            @test Scythe.potential_temperature(p_Pa, rho_d) ≈ theta rtol=1e-14
        end
        # Moist air: (R_m/R_d) T (p_0/p)^kappa
        Tk, p_Pa, q_v = 295.0, 95000.0, 0.015
        R_m = Rd + (q_v * Rv)
        rho_d = p_Pa / (R_m * Tk)
        expected = (R_m / Rd) * Tk * ((100.0 * Scythe.p_0) / p_Pa)^(Rd / Cpd)
        @test Scythe.potential_temperature(p_Pa, rho_d) ≈ expected rtol=1e-14

        # The mc heating coefficient rho_d*C_vt*(Cpd/Cvd)*(T/theta_d) reduces to the
        # classical rho*C_p*pi in dry air, so Q_therm = rho*Cp*pi*K*Lap(theta) exactly.
        Tk, p_Pa = 290.0, 90000.0
        rho_d = p_Pa / (Rd * Tk)
        theta_d = Scythe.potential_temperature(p_Pa, rho_d)
        exner = (p_Pa / (100.0 * Scythe.p_0))^(Rd / Cpd)
        @test rho_d * Cvd * (Cpd / Cvd) * (Tk / theta_d) ≈ rho_d * Cpd * exner rtol=1e-12
    end

    @testset "qss admissible bounds and relaxation" begin
        Tk, p_hPa = 290.0, 900.0
        rho_vs = rho_v_sat(Tk, p_hPa)
        rho_d = 1.1
        tau = 10.0

        # Dry air: rho_t == rho_d, so the interval collapses to the point -rho_vs
        lo, hi = Scythe.qss_admissible_bounds(rho_d, rho_d, 0.0, rho_vs)
        @test lo == -rho_vs && hi == -rho_vs
        # ... and the relaxation drives Q_ss there from either side, at rate 1/tau
        @test Scythe.qss_relaxation(0.0, rho_d, rho_d, 0.0, rho_vs, tau) ≈ -rho_vs / tau
        @test Scythe.qss_relaxation(-2rho_vs, rho_d, rho_d, 0.0, rho_vs, tau) ≈ rho_vs / tau

        # Cloudy air: Q_ss strictly interior => exactly zero, no nudge at all
        rho_t = rho_d + rho_vs + 1.0e-3          # 1 g/m^3 of cloud
        @test Scythe.qss_relaxation(0.0, rho_d, rho_t, 0.0, rho_vs, tau) == 0.0
        @test Scythe.qss_relaxation(1.0e-4, rho_d, rho_t, 0.0, rho_vs, tau) == 0.0

        # Supersaturated cloud-free air sits exactly at the ceiling Q_hi > 0: the
        # relaxation is zero there, so nucleation is not suppressed.
        rho_w = 1.05 * rho_vs
        rho_t = rho_d + rho_w
        _, Q_hi = Scythe.qss_admissible_bounds(rho_d, rho_t, 0.0, rho_vs)
        @test Q_hi > 0.0
        @test Scythe.qss_relaxation(Q_hi, rho_d, rho_t, 0.0, rho_vs, tau) == 0.0
        Q_s = Scythe.Q_s_energy(Tk, 100.0 * p_hPa, rho_d, rho_w / rho_d, 0.0)
        @test Scythe.qss_condensation_rate(Q_hi, rho_w, 0.0, rho_d, Tk, p_hPa, Q_s, 0.1) > 0.0
        # Drift above the ceiling is pulled back
        @test Scythe.qss_relaxation(Q_hi + 0.02, rho_d, rho_t, 0.0, rho_vs, tau) ≈ -0.02 / tau

        # Rain is liquid: the ceiling excludes it
        rho_r = 5.0e-4
        rho_t = rho_d + rho_vs + rho_r
        _, Q_hi_rain = Scythe.qss_admissible_bounds(rho_d, rho_t, rho_r, rho_vs)
        @test Q_hi_rain ≈ 0.0 atol=1e-15
    end

    @testset "retrieval clamp keeps rho_c nonnegative with rain" begin
        # Total water is all rain: vapor must clamp to zero, not to rho_w, or the
        # residual cloud rho_c = rho_t - rho_d - rho_v - rho_r goes negative.
        Tk, p_Pa = 290.0, 90000.0
        rho_d = p_Pa / (Rd * Tk)
        rho_r = 1.0e-3
        rho_t = rho_d + rho_r
        rho_vs = rho_v_sat(Tk, p_Pa / 100.0)
        Q_ss = -rho_vs                                   # rho_v = 0
        M = (rho_d * Cpd * Tk) + (-rho_r * Scythe.L_v(Tk))
        T_ret = Scythe.retrieve_temperature(M + p_Pa - p_Pa, rho_d, rho_t, Q_ss, p_Pa,
                                            Tk, rho_r)
        rho_v = clamp(Q_ss + rho_v_sat(T_ret, p_Pa / 100.0), 0.0,
                      max(rho_t - rho_d - rho_r, 0.0))
        @test rho_v ≈ 0.0 atol=1e-14
        @test rho_t - rho_d - rho_v - rho_r >= -1e-14    # rho_c >= 0
    end

    @testset "resting dry base is untouched by diffusion" begin
        # s_d' = 0 identically on the reference (s_d = C_vd ln p - C_pd ln rho_d is an
        # explicit function of p, rho_d, so at rest it equals s_dbar bit-for-bit), so every
        # diffusive tendency must vanish. A K = 75 run has to be BIT-IDENTICAL to a K = 0
        # run at rest: this catches a bad s_dbar subtraction that would cook the base.
        mktempdir() do tmpdir
            m0, p0, mod0, _ = make_mc_mtile(tmpdir; dry=true, Kvdiff=0.0, Khdiff=0.0)
            step_mc!(m0, p0, mod0, 3)
        end
        mktempdir() do tmpdir
            mK, pK, modK, _ = make_mc_mtile(tmpdir; dry=true, Kvdiff=75.0, Khdiff=75.0)
            step_mc!(mK, pK, modK, 3)
            # Nothing was seeded, so the base must stay exactly at zero perturbation
            @test maximum(abs.(pK.physical[:, 4, 1])) == 0.0   # u
            @test maximum(abs.(pK.physical[:, 5, 1])) == 0.0   # w
            @test maximum(abs.(pK.physical[:, 1, 1])) == 0.0   # p'
            @test maximum(abs.(pK.physical[:, 6, 1])) == 0.0   # E_t'
            @test all(isfinite.(pK.physical[:, :, 1]))
        end
    end

    @testset "diffusive heating sources p and E_t consistently" begin
        # In dry air the retrieval Jacobian is F_T = rho_d*C_pt, so a heating Qdot that
        # sources dE_t = Qdot and dp = (R_m/C_vt)*Qdot yields dT = dp/(rho_d*R_m) exactly.
        # Diffing a Kvdiff = 75 step against a Kvdiff = 0 step isolates the split from
        # the O(ts^2) truncation error of the rest of the scheme. Khdiff = 0 so the two
        # runs share an identical expdot.
        mktempdir() do tmpdir
            args = (; dry=true, Khdiff=0.0, kDim=16, num_cells=8)
            m0, patch0, mod0, _ = make_mc_mtile(tmpdir; args..., Kvdiff=0.0)
            gp0 = Scythe.getGridpoints(patch0)
            ref0 = m0.ref_state
            Scythe.theta_bubble_mc!(patch0, gp0, ref0;
                                    xc=1000.0, xr=400.0, zc=1000.0, zr=400.0, dtheta_max=2.0)
            spectralTransform!(patch0); gridTransform!(patch0)
            ncols = div(size(patch0.physical, 1), mod0.grid_params.kDim)
            for c in 1:ncols; Scythe.advance_column(m0, c, 1); end

            mK, patchK, modK, _ = make_mc_mtile(tmpdir; args..., Kvdiff=75.0)
            gpK = Scythe.getGridpoints(patchK)
            Scythe.theta_bubble_mc!(patchK, gpK, mK.ref_state;
                                    xc=1000.0, xr=400.0, zc=1000.0, zr=400.0, dtheta_max=2.0)
            spectralTransform!(patchK); gridTransform!(patchK)
            for c in 1:ncols; Scythe.advance_column(mK, c, 1); end

            kDim = mod0.grid_params.kDim
            pbar = Springsteel.ref_pressure(ref0)[:, 1]
            rho_dbar = Springsteel.ref_rho_d(ref0)[:, 1]
            rho_tbar = Springsteel.ref_rho_t(ref0)[:, 1]
            E_tbar = Springsteel.ref_total_energy(ref0)[:, 1]
            Q_ssbar = Springsteel.ref_qss(ref0)[:, 1]
            Tbar = Springsteel.reference_temperature(ref0)
            zs = gp0[:, 2]

            function retrieved(mt, i, k)
                p = mt.var_np1[i, 1] + pbar[k]
                rho_d = mt.var_np1[i, 2] + rho_dbar[k]
                rho_t = mt.var_np1[i, 3] + rho_tbar[k]
                E_t = mt.var_np1[i, 6] + E_tbar[k]
                Q_ss = mt.var_np1[i, 7] + Q_ssbar[k]
                ke = 0.5 * (mt.var_np1[i, 4]^2 + mt.var_np1[i, 5]^2)
                M = p + E_t - rho_t * (ke + Scythe.gravity * zs[i])
                return Scythe.retrieve_temperature(M, rho_d, rho_t, Q_ss, p, Tbar[k],
                                                   mt.var_np1[i, 8]), p, rho_d
            end

            dp_max = 0.0
            worst = 0.0
            for i in 1:size(patch0.physical, 1)
                k = mod1(i, kDim)
                T0, p0v, rho_d0 = retrieved(m0, i, k)
                TK, pKv, rho_dK = retrieved(mK, i, k)
                dp = pKv - p0v
                dp_max = max(dp_max, abs(dp))
                # Dry air: R_m = Rd, and rho_d is untouched by the diffusion split
                @test rho_dK ≈ rho_d0 rtol=1e-14
                dT_expected = dp / (rho_dK * Rd)
                worst = max(worst, abs((TK - T0) - dT_expected))
            end
            # The diffusion split actually did something...
            @test dp_max > 1.0e-6
            # ...and the retrieved temperature increment matches the EOS-slaved pressure
            # increment to machine precision. Sourcing p or E_t alone breaks this.
            @test worst < 1.0e-9
        end
    end

    @testset "momentum diffusion is a resolved-KE sink" begin
        # Eddy friction removes resolved KE to the subgrid (the future TKE shear
        # production), so E_t follows the KE DOWN and the internal energy is HELD: T and p
        # are unchanged (no dissipative heating — the review fix). The E_t decrease equals
        # the resolved KE removed, dE_t = rho_t*dke.
        mktempdir() do tmpdir
            mtile, patch, model, _ = make_mc_mtile(tmpdir; dry=true, Khdiff=0.0,
                                                   Kvdiff=75.0, kDim=16,
                                                   u_side_bc=NeumannBC())
            kDim = model.grid_params.kDim
            gridpoints = Scythe.getGridpoints(patch)
            rho_tbar = Springsteel.ref_rho_t(mtile.ref_state)[:, 1]
            # u = U0 sin(pi z/H): vanishes at the no-slip lids, u_zz != 0 in the interior,
            # and is x-independent so the divergence (hence every other tendency) is zero.
            # E_t carries the kinetic energy in this set, so a consistent shear IC must
            # seed E_t' = rho_t*ke too — without it the retrieval sees a phantom cold
            # anomaly and the (now moist-entropy) heat path fires on it.
            for i in 1:size(patch.physical, 1)
                k = mod1(i, kDim)
                u0 = 10.0 * sin(pi * gridpoints[i, 2] / 2000.0)
                patch.physical[i, 4, 1] = u0
                patch.physical[i, 6, 1] = rho_tbar[k] * 0.5 * u0^2
            end
            spectralTransform!(patch); gridTransform!(patch)

            E_t_before = copy(patch.physical[:, 6, 1])
            p_before = copy(patch.physical[:, 1, 1])
            u_before = copy(patch.physical[:, 4, 1])
            ncols = div(size(patch.physical, 1), kDim)
            for c in 1:ncols; Scythe.advance_column(mtile, c, 1); end

            @test all(isfinite.(mtile.var_np1))
            dke = 0.5 .* (mtile.var_np1[:, 4] .^ 2 .+ mtile.var_np1[:, 5] .^ 2 .-
                          u_before .^ 2)
            # Net kinetic energy falls
            @test sum(dke) < 0.0

            sheared = findall(x -> x < -1.0e-9, dke)
            @test !isempty(sheared)
            for i in sheared
                k = mod1(i, kDim)
                # E_t follows the KE down (the sink), and p is untouched (internal energy
                # held — no frictional heating; the fit-residual s_t' leaves only a
                # negligible heat increment).
                @test isapprox(mtile.var_np1[i, 6] - E_t_before[i], rho_tbar[k] * dke[i];
                               rtol=1e-4, atol=1e-7)
                @test mtile.var_np1[i, 1] ≈ p_before[i] atol=1e-6
            end
        end
    end

    @testset "Prandtl/Schmidt parameters are rejected" begin
        # Eddy mixing coefficients are not molecular ratios: heat and water diffusivities
        # are specified directly (:Khdiff_heat/:Kvdiff_heat, :Khdiff_water/:Kvdiff_water).
        vars = Dict(v => i for (i, v) in enumerate(Scythe.MC_VARS))
        scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
        gp = GridParameters(
            geometry = "RZ", num_cells = 4,
            iMin = 0.0, iMax = 1000.0, kMin = 0.0, kMax = 1000.0, kDim = 8,
            BCL = scalar_bc, BCR = scalar_bc, BCB = scalar_bc, BCT = scalar_bc, vars = vars,
        )
        for bad in (:Prandtl, :Schmidt)
            @test_throws ErrorException ModelParameters(
                ts = 0.1, equation_set = "moist_compressible_XZ", grid_params = gp,
                physical_params = Dict(:Khdiff => 0.0, :Kvdiff => 0.0, bad => 1.0))
        end
    end

    @testset "heat-only vertical diffusion leaves momentum untouched" begin
        # Kvdiff = 0, Kvdiff_heat > 0: the momentum solve must be skipped entirely (a K = 0
        # solve is not the identity — it refits and refilters the column), so u and w must
        # be BIT-identical to a no-diffusion run, while the heat path sources p/E_t/Q_ss.
        mktempdir() do tmpdir
            args = (; dry=true, Khdiff=0.0, kDim=16, num_cells=8)
            function run_bubble(; kwargs...)
                mtile, patch, model, _ = make_mc_mtile(tmpdir; args..., kwargs...)
                gp = Scythe.getGridpoints(patch)
                Scythe.theta_bubble_mc!(patch, gp, mtile.ref_state;
                                        xc=1000.0, xr=400.0, zc=1000.0, zr=400.0,
                                        dtheta_max=2.0)
                spectralTransform!(patch); gridTransform!(patch)
                ncols = div(size(patch.physical, 1), model.grid_params.kDim)
                for c in 1:ncols; Scythe.advance_column(mtile, c, 1); end
                return mtile
            end
            m0 = run_bubble(; Kvdiff=0.0)
            mH = run_bubble(; Kvdiff=0.0, Kvdiff_heat=75.0)
            @test mH.var_np1[:, 4] == m0.var_np1[:, 4]   # u bit-identical
            @test mH.var_np1[:, 5] == m0.var_np1[:, 5]   # w bit-identical
            # ...but the heat path actually did something to p, E_t and Q_ss
            @test maximum(abs.(mH.var_np1[:, 1] .- m0.var_np1[:, 1])) > 1.0e-6
            @test maximum(abs.(mH.var_np1[:, 6] .- m0.var_np1[:, 6])) > 1.0e-6
        end
    end

    @testset "momentum-only vertical diffusion leaves thermodynamics untouched" begin
        # Kvdiff > 0, Kvdiff_heat = 0: the heat solve must be skipped, so p and Q_ss are
        # BIT-identical to a no-diffusion run and E_t changes only by the resolved-KE sink
        # dE_visc = rho_t*dke. The x-independent shear IC keeps the divergence (and hence
        # every explicit tendency) identical between the runs.
        mktempdir() do tmpdir
            args = (; dry=true, Khdiff=0.0, kDim=16, num_cells=8, u_side_bc=NeumannBC())
            function run_shear(; kwargs...)
                mtile, patch, model, _ = make_mc_mtile(tmpdir; args..., kwargs...)
                gridpoints = Scythe.getGridpoints(patch)
                for i in 1:size(patch.physical, 1)
                    patch.physical[i, 4, 1] = 10.0 * sin(pi * gridpoints[i, 2] / 2000.0)
                end
                spectralTransform!(patch); gridTransform!(patch)
                ncols = div(size(patch.physical, 1), model.grid_params.kDim)
                for c in 1:ncols; Scythe.advance_column(mtile, c, 1); end
                return mtile
            end
            m0 = run_shear(; Kvdiff=0.0)
            mM = run_shear(; Kvdiff=75.0, Kvdiff_heat=0.0)
            @test mM.var_np1[:, 1] == m0.var_np1[:, 1]   # p bit-identical
            @test mM.var_np1[:, 7] == m0.var_np1[:, 7]   # Q_ss bit-identical
            # Momentum diffusion acted, and E_t moved by exactly the resolved-KE sink
            @test maximum(abs.(mM.var_np1[:, 4] .- m0.var_np1[:, 4])) > 1.0e-6
            kDim = 16
            rho_tbar = Springsteel.ref_rho_t(mM.ref_state)[:, 1]
            dke = 0.5 .* (mM.var_np1[:, 4] .^ 2 .+ mM.var_np1[:, 5] .^ 2 .-
                          m0.var_np1[:, 4] .^ 2 .- m0.var_np1[:, 5] .^ 2)
            for i in findall(x -> abs(x) > 1.0e-9, dke)
                k = mod1(i, kDim)
                @test (mM.var_np1[i, 6] - m0.var_np1[i, 6]) ≈ rho_tbar[k] * dke[i] rtol=1e-6
            end
        end
    end

    @testset "Q_ss relaxation is thermodynamically inert in dry air" begin
        # rho_w = 0 => rho_v == 0 regardless of Q_ss, so the relaxation cannot move T, p,
        # E_t or the densities. It must, however, pull the drifting Q_ss back to -rho_vs.
        mktempdir() do tmpdir
            function run_dry(qss_offset)
                mtile, patch, model, _ = make_mc_mtile(tmpdir; dry=true, tau_qss=10.0)
                gridpoints = Scythe.getGridpoints(patch)
                Scythe.theta_bubble_mc!(patch, gridpoints, mtile.ref_state;
                                        xc=1000.0, xr=400.0, zc=1000.0, zr=400.0,
                                        dtheta_max=2.0)
                patch.physical[:, 7, 1] .+= qss_offset
                spectralTransform!(patch); gridTransform!(patch)
                step_mc!(mtile, patch, model, 5)
                return copy(patch.physical[:, :, 1])
            end
            base = run_dry(0.0)
            drifted = run_dry(0.02)

            # Every slot except Q_ss is bit-identical
            for v in (1, 2, 3, 4, 5, 6, 8)
                @test base[:, v] == drifted[:, v]
            end
            # ... and the drift decays toward the base at the relaxation rate
            @test maximum(abs.(drifted[:, 7] .- base[:, 7])) < 0.02
            @test maximum(abs.(drifted[:, 7] .- base[:, 7])) > 0.0
        end
    end

    # ──────────────────────────────────────────────
    # 7. Warm-rain microphysics (density form)
    # ──────────────────────────────────────────────

    """Retrieved temperature at every point of a stepped mtile (var_np1 + reference)."""
    function retrieved_T_mc(mtile, kDim, zs)
        ref = mtile.ref_state
        pbar = Springsteel.ref_pressure(ref)[:, 1]
        rho_dbar = Springsteel.ref_rho_d(ref)[:, 1]
        rho_tbar = Springsteel.ref_rho_t(ref)[:, 1]
        E_tbar = Springsteel.ref_total_energy(ref)[:, 1]
        Q_ssbar = Springsteel.ref_qss(ref)[:, 1]
        Tbar = Springsteel.reference_temperature(ref)
        n = size(mtile.var_np1, 1)
        T = zeros(n)
        for i in 1:n
            k = mod1(i, kDim)
            p = mtile.var_np1[i, 1] + pbar[k]
            rho_d = mtile.var_np1[i, 2] + rho_dbar[k]
            rho_t = mtile.var_np1[i, 3] + rho_tbar[k]
            E_t = mtile.var_np1[i, 6] + E_tbar[k]
            Q_ss = mtile.var_np1[i, 7] + Q_ssbar[k]
            ke = 0.5 * (mtile.var_np1[i, 4]^2 + mtile.var_np1[i, 5]^2)
            M = p + E_t - rho_t * (ke + Scythe.gravity * zs[i])
            T[i] = Scythe.retrieve_temperature(M, rho_d, rho_t, Q_ss, p, Tbar[k],
                                               mtile.var_np1[i, 8])
        end
        return T
    end

    """Seed a T-invariant rain bump: rho_r, rho_t and E_t move together so the
    retrieval is unchanged (delta_E_t = (C_pv*T - L_v + g*z)*delta_rho at fixed p, Q_ss)."""
    function seed_rain_bump!(patch, gridpoints, col, kDim; rho_r0=1.0e-3, zc=1200.0, zr=300.0)
        for i in 1:size(patch.physical, 1)
            k = mod1(i, kDim)
            z = gridpoints[i, 2]
            seed = rho_r0 * exp(-((z - zc) / zr)^2)
            Tref = col.Tk[k]
            patch.physical[i, 8, 1] += seed
            patch.physical[i, 3, 1] += seed
            patch.physical[i, 6, 1] += seed * ((Cpv * Tref) - L_v(Tref) +
                                               (Scythe.gravity * z))
        end
        spectralTransform!(patch)
        gridTransform!(patch)
    end

    @testset "autoconversion is thermodynamically inert" begin
        # Cloud above the 1 g/kg threshold, no rain yet: autoconversion is the only
        # active process (collection needs rain, sedimentation and the rain relaxation
        # channel need rho_r >= RHO_R_MIN). Liquid -> liquid conversion must leave every
        # thermodynamic slot BIT-identical to a precipitation-off run; only rho_r grows
        # (cloud is the diagnostic residual, so it shrinks automatically).
        mktempdir() do tmpdir
            args = (; q_l=3.0e-3, kDim=16, num_cells=8)
            function run_once(precip)
                mtile, patch, model, _ = make_mc_mtile(tmpdir; args..., precipitation=precip)
                ncols = div(size(patch.physical, 1), model.grid_params.kDim)
                for c in 1:ncols; Scythe.advance_column(mtile, c, 1); end
                return mtile
            end
            m_off = run_once(false)
            m_on = run_once(true)
            for slot in (1, 2, 3, 4, 5, 6, 7)
                @test m_on.var_np1[:, slot] == m_off.var_np1[:, slot]
            end
            @test all(m_on.var_np1[:, 8] .> 0.0)
            @test maximum(m_on.var_np1[:, 8]) < 3.0e-3   # bounded by the available cloud
            @test all(m_off.var_np1[:, 8] .== 0.0)
        end
    end

    @testset "sedimentation moves rain and total mass together" begin
        # Cloud-free saturated column with a T-invariant rain bump aloft: the rain
        # channel is quiet (Q_ss = 0) and there is no cloud to convert, so the on/off
        # difference isolates the sedimentation flux. rho_r and rho_t must receive the
        # SAME fitted divergence, the bump must fall, and the energy coupling must keep
        # the retrieved temperature unchanged to leading order.
        mktempdir() do tmpdir
            args = (; q_l=0.0, kDim=32, num_cells=8, ts=0.05)
            function run_once(precip)
                mtile, patch, model, col = make_mc_mtile(tmpdir; args..., precipitation=precip)
                gp = Scythe.getGridpoints(patch)
                seed_rain_bump!(patch, gp, col, model.grid_params.kDim)
                rho_r0 = copy(patch.physical[:, 8, 1])
                ncols = div(size(patch.physical, 1), model.grid_params.kDim)
                for c in 1:ncols; Scythe.advance_column(mtile, c, 1); end
                return mtile, gp, rho_r0
            end
            m_off, gp, rho_r0 = run_once(false)
            m_on, _, _ = run_once(true)
            @test all(isfinite.(m_on.var_np1))

            d3 = m_on.var_np1[:, 3] .- m_off.var_np1[:, 3]
            d8 = m_on.var_np1[:, 8] .- m_off.var_np1[:, 8]
            # Sedimentation actually moved mass...
            @test maximum(abs.(d8)) > 1.0e-8
            # ...and rho_t tracks rho_r exactly (identical -dF/dz in both slots)
            @test all(isapprox.(d3, d8; atol=1.0e-14))
            # The bump falls: rain-weighted mean height decreases
            kDim = 32
            zs = gp[:, 2]
            com(r) = sum(max.(r, 0.0) .* zs) / sum(max.(r, 0.0))
            @test com(m_on.var_np1[:, 8]) < com(rho_r0)
            # Full energy coupling: the local exchange leaves the retrieval unchanged;
            # only the genuine transport term (~1e-5 K per step) remains. A missing or
            # wrong e_l/gz coupling shows up at the 0.1 K level.
            dT = retrieved_T_mc(m_on, kDim, zs) .- retrieved_T_mc(m_off, kDim, zs)
            @test maximum(abs.(dT)) < 1.0e-3
        end
    end

    @testset "rain evaporation in subsaturated air" begin
        # Rain falling through dry air: the rain channel of the supersaturation
        # relaxation evaporates it (no separate Qevap parameterization). Vapor is
        # added (Q_ss rises toward saturation), rain is lost from rho_r but NOT from
        # rho_t (the vapor stays in the column), and the retrieval cools.
        mktempdir() do tmpdir
            args = (; dry=true, kDim=32, num_cells=8, ts=0.05)
            function run_once(precip)
                mtile, patch, model, col = make_mc_mtile(tmpdir; args..., precipitation=precip)
                gp = Scythe.getGridpoints(patch)
                seed_rain_bump!(patch, gp, col, model.grid_params.kDim)
                ncols = div(size(patch.physical, 1), model.grid_params.kDim)
                for c in 1:ncols; Scythe.advance_column(mtile, c, 1); end
                return mtile, gp
            end
            m_off, gp = run_once(false)
            m_on, _ = run_once(true)
            @test all(isfinite.(m_on.var_np1))

            d3 = m_on.var_np1[:, 3] .- m_off.var_np1[:, 3]
            d7 = m_on.var_np1[:, 7] .- m_off.var_np1[:, 7]
            d8 = m_on.var_np1[:, 8] .- m_off.var_np1[:, 8]
            # Evaporation added vapor somewhere in the rain shaft
            @test maximum(d7) > 0.0
            # Rain lost beyond what sedimentation moved: d8 - d3 = ts * Qdot_r < 0.
            # The upper bound is acoustic cross-talk, not evaporation: the (always-on)
            # SI solve slaves rho_t' to the mass flux but not rho_r, and the precip
            # switch perturbs the solve, so the difference is no longer exactly
            # ts * Qdot_r (measured ~1e-9 vs the >1e-6 evaporation signal).
            @test minimum(d8 .- d3) < 0.0
            @test maximum(d8 .- d3) <= 1.0e-8
            # Evaporative cooling with E_t held: the retrieval must cool, never warm
            kDim = 32
            zs = gp[:, 2]
            dT = retrieved_T_mc(m_on, kDim, zs) .- retrieved_T_mc(m_off, kDim, zs)
            @test minimum(dT) < 0.0
            @test maximum(dT) < 1.0e-3
        end
    end

    @testset "no condensational rain growth in cloud-free air" begin
        # The O01 spurious-blob mechanism in miniature: rain seeds in supersaturated
        # CLOUD-FREE air (spectral ringing + gravity-wave cooling at the lid) must not
        # grow by direct vapor deposition — the physical pathway is condensation ->
        # cloud -> autoconversion. With no cloud in the column, rain may only move
        # (sedimentation, where d_rho_t tracks d_rho_r exactly) or evaporate, so the
        # precipitation on/off difference must satisfy d8 - d3 <= 0 EVERYWHERE.
        mktempdir() do tmpdir
            args = (; q_l=0.0, kDim=32, num_cells=8, ts=0.05)
            function run_once(precip)
                mtile, patch, model, col = make_mc_mtile(tmpdir; args..., precipitation=precip)
                gp = Scythe.getGridpoints(patch)
                kDim = model.grid_params.kDim
                seed_rain_bump!(patch, gp, col, kDim)
                # Co-located supersaturation bump (S ~ 1e-3 at the center, well above
                # the nucleation threshold). Prognostic Q_ss only: both runs carry the
                # identical seed, so the on/off difference isolates the microphysics.
                for i in 1:size(patch.physical, 1)
                    k = mod1(i, kDim)
                    z = gp[i, 2]
                    rho_vs_k = rho_v_sat(col.Tk[k], col.p_Pa[k] / 100.0)
                    patch.physical[i, 7, 1] += 1.0e-3 * rho_vs_k *
                                               exp(-((z - 1200.0) / 300.0)^2)
                end
                spectralTransform!(patch)
                gridTransform!(patch)
                ncols = div(size(patch.physical, 1), kDim)
                for c in 1:ncols; Scythe.advance_column(mtile, c, 1); end
                return mtile
            end
            m_off = run_once(false)
            m_on = run_once(true)
            @test all(isfinite.(m_on.var_np1))
            d3 = m_on.var_np1[:, 3] .- m_off.var_np1[:, 3]
            d8 = m_on.var_np1[:, 8] .- m_off.var_np1[:, 8]
            # Sedimentation is active (the rain actually moves)...
            @test maximum(abs.(d8)) > 1.0e-8
            # ...but no rain is created by condensation anywhere in the column
            @test maximum(d8 .- d3) <= 1.0e-14
        end
    end

    # ──────────────────────────────────────────────
    # 7b. Rayleigh sponge (upper-boundary absorbing layer)
    # ──────────────────────────────────────────────

    @testset "Rayleigh sponge damps momentum and routes KE to E_t" begin
        # Momentum-only Durran-Klemp sponge: u and w are damped toward the resting
        # base state above z_damp, the destroyed resolved KE follows into E_t (the
        # FRIC_KE invariant: dE = 2*rho_t*tau*ke, T/p/Q_ss held), and everything
        # below the onset height — and every other slot — is bit-identical to an
        # alpha = 0 run.
        mktempdir() do tmpdir
            alpha = 0.2
            z_damp = 1000.0
            args = (; dry=true, kDim=32, num_cells=8, ts=0.1, u_side_bc=NeumannBC())
            function run_once(a)
                mtile, patch, model, _ = make_mc_mtile(tmpdir; args..., alpha=a,
                                                       z_damp=z_damp)
                gp = Scythe.getGridpoints(patch)
                kDim = model.grid_params.kDim
                for i in 1:size(patch.physical, 1)
                    x, z = gp[i, 1], gp[i, 2]
                    patch.physical[i, 4, 1] += 20.0 * sin(0.5 * pi * z / 2000.0)
                    patch.physical[i, 5, 1] += 5.0 * exp(-((x - 1000.0) / 300.0)^2 -
                                                         ((z - 1500.0) / 200.0)^2)
                end
                spectralTransform!(patch)
                gridTransform!(patch)
                # The filtered pre-step state the tendency actually sees
                u0 = copy(patch.physical[:, 4, 1])
                w0 = copy(patch.physical[:, 5, 1])
                rho_tp0 = copy(patch.physical[:, 3, 1])
                ncols = div(size(patch.physical, 1), kDim)
                for c in 1:ncols; Scythe.advance_column(mtile, c, 1); end
                return mtile, gp, u0, w0, rho_tp0
            end
            m_off, gp, u0, w0, rho_tp0 = run_once(0.0)
            m_on, _, _, _, _ = run_once(alpha)

            kDim = 32
            zs = gp[:, 2]
            ztop = gp[kDim, 2]
            RAY = Scythe.Rayleigh_damping.(alpha, zs, z_damp, ztop)
            rho_tbar = Springsteel.ref_rho_t(m_on.ref_state)[:, 1]
            rho_t0 = rho_tp0 .+ [rho_tbar[mod1(i, kDim)] for i in eachindex(zs)]
            ke0 = 0.5 .* ((u0 .^ 2) .+ (w0 .^ 2))

            d4 = m_on.expdot_n[:, 4] .- m_off.expdot_n[:, 4]
            d5 = m_on.expdot_n[:, 5] .- m_off.expdot_n[:, 5]
            d6 = m_on.expdot_n[:, 6] .- m_off.expdot_n[:, 6]
            below = zs .<= z_damp
            @test all(d4[below] .== 0.0)
            @test all(d5[below] .== 0.0)
            @test all(d6[below] .== 0.0)
            @test any(.!below)
            @test maximum(abs.(d4 .- (RAY .* u0))) < 1.0e-10
            @test maximum(abs.(d5 .- (RAY .* w0))) < 1.0e-10
            expected6 = 2.0 .* rho_t0 .* RAY .* ke0
            @test maximum(abs.(d6 .- expected6)) <
                  1.0e-10 * maximum(abs.(expected6)) + 1.0e-10
            # T, p, Q_ss, masses: no sponge term at all
            for slot in (1, 2, 3, 7, 8)
                @test m_on.expdot_n[:, slot] == m_off.expdot_n[:, slot]
            end
            # After the step the retrieved temperature is unchanged to O(ts^2): a
            # missing E_t coupling shows up at the ~1e-2 K level with this seed.
            # (3e-3: the always-on SI solve routes the sponge's w difference through
            # the acoustic E_t slaving, adding ~1.4e-3 K on top of the old bound —
            # still 3x below the missing-coupling discriminant.)
            dT = retrieved_T_mc(m_on, kDim, zs) .- retrieved_T_mc(m_off, kDim, zs)
            @test maximum(abs.(dT)) < 3.0e-3
        end

        # Configs without :alpha/:z_damp keys must run (the sponge defaults to off)
        mktempdir() do tmpdir
            mtile, patch, model, _ = make_mc_mtile(tmpdir; dry=true)
            delete!(model.physical_params, :alpha)
            delete!(model.physical_params, :z_damp)
            ncols = div(size(patch.physical, 1), model.grid_params.kDim)
            for c in 1:ncols; Scythe.advance_column(mtile, c, 1); end
            @test all(isfinite.(mtile.var_np1))
        end
    end

    # ──────────────────────────────────────────────
    # 8. Vertical moist diffusion (s_t heat + water species)
    # ──────────────────────────────────────────────

    @testset "resting base is untouched by full moist diffusion" begin
        # s_tbar and rho_vbar come from mc_reference_diagnostics — the SAME retrieval
        # pipeline the equation set runs — so at rest s_t' = 0 and rho_v' = 0 BIT-exactly.
        # On the DRY base every vertical diffusive tendency (heat AND water) must then
        # vanish bit-for-bit; this catches a reference profile built from the (not
        # bit-identical) file Tbar instead.
        mktempdir() do tmpdir
            mK, pK, modK, _ = make_mc_mtile(tmpdir; dry=true,
                                            Kvdiff=75.0, Kvdiff_water=75.0, Khdiff=0.0)
            step_mc!(mK, pK, modK, 3)
            for slot in (1, 2, 3, 4, 5, 6, 8)
                @test maximum(abs.(pK.physical[:, slot, 1])) == 0.0
            end
            # Q_ss carries a pre-existing ~1e-18 tracking crumb (file Tbar vs retrieved-T
            # saturation in the relaxation limiters) even with all diffusion off
            @test maximum(abs.(pK.physical[:, 7, 1])) < 1.0e-16
        end
        # The CLOUDY base carries a pre-existing machine-precision condensation crumb
        # (the file-derived Q_ssbar vs the retrieved-T saturation differ at ~1e-17, so
        # Qdot != 0 at rest even with diffusion off — verified on the pre-diffusion
        # code). The diffusion solves see that crumb through the star state, so exact
        # zero is unattainable; the base must still be preserved to noise level.
        mktempdir() do tmpdir
            mK, pK, modK, _ = make_mc_mtile(tmpdir; dry=false, q_l=1.0e-3,
                                            Kvdiff=75.0, Kvdiff_water=75.0, Khdiff=0.0)
            step_mc!(mK, pK, modK, 3)
            @test maximum(abs.(pK.physical[:, 1, 1])) < 1.0e-9    # p [Pa]
            @test maximum(abs.(pK.physical[:, 6, 1])) < 1.0e-8    # E_t [J/m^3]
            for slot in (2, 3, 4, 5, 7, 8)
                @test maximum(abs.(pK.physical[:, slot, 1])) < 1.0e-10
            end
        end
    end

    @testset "water diffusion: rain bump conserves mass and holds T" begin
        # Rain bump aloft in a cloud-free saturated column, ONLY Kvdiff_water active.
        # The on/off difference isolates the water solves: rho_d untouched bit-exactly,
        # rho_t tracks rho_r (rain is the only water moving), the column integral of the
        # Neumann solve is conserved, and the fixed-T increment map leaves the retrieval
        # unchanged to splitting-error tolerance.
        mktempdir() do tmpdir
            args = (; q_l=0.0, kDim=32, num_cells=8, ts=0.05)
            function run_once(Kw)
                mtile, patch, model, col = make_mc_mtile(tmpdir; args..., Kvdiff_water=Kw)
                gp = Scythe.getGridpoints(patch)
                seed_rain_bump!(patch, gp, col, model.grid_params.kDim)
                ncols = div(size(patch.physical, 1), model.grid_params.kDim)
                for c in 1:ncols; Scythe.advance_column(mtile, c, 1); end
                return mtile, gp
            end
            m_off, gp = run_once(0.0)
            m_on, _ = run_once(75.0)
            @test all(isfinite.(m_on.var_np1))

            # rho_d carries no water diffusion: bit-identical
            @test m_on.var_np1[:, 2] == m_off.var_np1[:, 2]
            d3 = m_on.var_np1[:, 3] .- m_off.var_np1[:, 3]
            d8 = m_on.var_np1[:, 8] .- m_off.var_np1[:, 8]
            @test maximum(abs.(d8)) > 1.0e-8            # diffusion acted on the bump
            # Rain is the only water species moving: rho_w' and rho_r' get the same
            # increment (same Neumann operator, same data). Tolerance: the always-on
            # SI solve slaves an acoustic increment onto rho_t' (moist base: c_d < 1)
            # but not rho_r, so the two solves' inputs differ at the ~1e-12 level
            # (signal > 1e-8).
            @test maximum(abs.(d3 .- d8)) < 1.0e-10
            # Neumann solve conserves the column integral (trapezoid per column)
            kDim = 32
            zs = gp[1:kDim, 2]
            ncols = div(length(d8), kDim)
            for c in 1:ncols
                seg = d8[(c-1)*kDim+1:c*kDim]
                integral = sum(0.5 .* (seg[1:end-1] .+ seg[2:end]) .* diff(zs))
                mass = sum(0.5 .* (m_on.var_np1[(c-1)*kDim+1:c*kDim, 8][1:end-1] .+
                                   m_on.var_np1[(c-1)*kDim+1:c*kDim, 8][2:end]) .* diff(zs))
                @test abs(integral) < 1.0e-6 * max(abs(mass), 1.0e-3)
            end
            # Fixed-T map: the retrieval is invariant under the water increments
            dT = retrieved_T_mc(m_on, kDim, gp[:, 2]) .- retrieved_T_mc(m_off, kDim, gp[:, 2])
            @test maximum(abs.(dT)) < 1.0e-4
        end
    end

    @testset "water diffusion: vapor bump keeps the cloud residual zero" begin
        # Subsaturated vapor bump in dry air, ONLY Kvdiff_water active: rho_w' and
        # rho_v' diffuse through the SAME operator, so the implied cloud increment
        # delta_rho_c = delta_rho_w - delta_rho_v must stay ~0 — vapor diffusion cannot
        # manufacture cloud. Rain stays exactly zero, and the fixed-T map holds T.
        mktempdir() do tmpdir
            args = (; dry=true, kDim=32, num_cells=8, ts=0.05)
            function run_once(Kw)
                mtile, patch, model, col = make_mc_mtile(tmpdir; args..., Kvdiff_water=Kw)
                gp = Scythe.getGridpoints(patch)
                kDim = model.grid_params.kDim
                # T-invariant vapor seed: dQ_ss = drho_t = seed, dp = Rv*T*seed,
                # dE_t = (Cvv*T + g*z)*seed (the water map's drho_w = drho_v case)
                for i in 1:size(patch.physical, 1)
                    k = mod1(i, kDim)
                    zi = gp[i, 2]
                    seed = 2.0e-3 * exp(-((zi - 1000.0) / 300.0)^2)
                    Tref = col.Tk[k]
                    patch.physical[i, 7, 1] += seed
                    patch.physical[i, 3, 1] += seed
                    patch.physical[i, 1, 1] += Rv * Tref * seed
                    patch.physical[i, 6, 1] += ((Cvv * Tref) + (Scythe.gravity * zi)) * seed
                end
                spectralTransform!(patch)
                gridTransform!(patch)
                ncols = div(size(patch.physical, 1), kDim)
                for c in 1:ncols; Scythe.advance_column(mtile, c, 1); end
                return mtile, gp
            end
            m_off, gp = run_once(0.0)
            m_on, _ = run_once(75.0)
            @test all(isfinite.(m_on.var_np1))

            d3 = m_on.var_np1[:, 3] .- m_off.var_np1[:, 3]
            d7 = m_on.var_np1[:, 7] .- m_off.var_np1[:, 7]
            @test maximum(abs.(d3)) > 1.0e-8            # diffusion acted
            # Vapor-only water: the rho_w and rho_v increments must agree (no cloud
            # manufactured); the two fields ride the same operator but different
            # staging paths (prognostic zz slots vs a column refit), plus the acoustic
            # slaving on rho_t' from the always-on SI solve, hence the tolerance
            @test maximum(abs.(d3 .- d7)) < 5.0e-9
            # No rain appears from water diffusion of a rain-free column
            @test m_on.var_np1[:, 8] == m_off.var_np1[:, 8]
            # Fixed-T map holds the retrieval
            kDim = 32
            dT = retrieved_T_mc(m_on, kDim, gp[:, 2]) .- retrieved_T_mc(m_off, kDim, gp[:, 2])
            @test maximum(abs.(dT)) < 1.0e-4
        end
    end

    # ── Axisymmetric cylinder (moist_compressible_axisym): the same kernel on the
    #    same 2D grid with x reinterpreted as radius, metric/curvature terms, and
    #    the prognostic tangential wind v (slot 9) ──
    @testset "axisym: resting cloudy base preserved" begin
        mktempdir() do tmpdir
            # Small radius on purpose: the metric terms are O(1/r), so this exercises
            # them as strongly as the domain allows; at rest they must all vanish.
            mtile, patch, model, col = make_mc_mtile(tmpdir;
                equation_set = "moist_compressible_axisym",
                iMin = 10.0e3, iMax = 12.0e3, f = 5.0e-5)
            kDim = model.grid_params.kDim
            ncols = div(size(patch.physical, 1), kDim)
            for c in 1:ncols
                Scythe.advance_column(mtile, c, 1)
            end
            scales = Dict(1 => 1.0e5, 2 => 1.0, 3 => 1.0, 4 => 1.0, 5 => 1.0,
                          6 => 2.0e8, 7 => 1.0e-2, 8 => 1.0, 9 => 1.0)
            for v in 1:9
                @test maximum(abs.(mtile.expdot_n[:, v])) / scales[v] < 1.0e-9
                @test maximum(abs.(mtile.var_np1[:, v])) / scales[v] < 1.0e-9
            end
            @test all(isfinite.(mtile.var_np1))
        end
    end

    @testset "axisym: v stays exactly zero without rotation" begin
        mktempdir() do tmpdir
            mtile, patch, model, col = make_mc_mtile(tmpdir;
                equation_set = "moist_compressible_axisym",
                iMin = 10.0e3, iMax = 12.0e3, Khdiff = 25.0, Kvdiff = 25.0, f = 0.0)
            kDim = model.grid_params.kDim
            vars = model.grid_params.vars
            npts = size(patch.physical, 1)
            ncols = div(npts, kDim)
            # A column-varying pressure anomaly drives u through the PGF from step 1
            p_i = vars["p"]
            for i in 1:npts
                c = div(i - 1, kDim)
                patch.physical[i, p_i, 1] = 10.0 * sinpi((c + 0.5) / ncols)
            end
            spectralTransform!(patch)
            gridTransform!(patch)
            step_mc!(mtile, patch, model, 5)
            # The flow spins up ...
            @test maximum(abs.(patch.physical[:, vars["u"], 1])) > 0.0
            # ... but with f = 0 and v0 = 0 the tangential wind has no source at all
            @test maximum(abs.(patch.physical[:, vars["v"], 1])) == 0.0
            @test maximum(abs.(mtile.expdot_n[:, vars["v"]])) == 0.0
            @test all(isfinite.(patch.physical))
        end
    end

    @testset "axisym: large-radius tendencies match the Cartesian slice" begin
        mktempdir() do tmpdir
            # At r ~ 1e9 m the metric terms are O(dx/r) ~ 1e-6 relative: the axisym
            # tendencies of the shared 8 slots must converge to XZ's on identical fields.
            R0 = 1.0e9
            mt_xz, p_xz, model_xz, _ = make_mc_mtile(tmpdir; Khdiff = 25.0)
            mt_ax, p_ax, model_ax, _ = make_mc_mtile(tmpdir;
                equation_set = "moist_compressible_axisym",
                iMin = R0, iMax = R0 + 2000.0, Khdiff = 25.0, f = 0.0)
            kDim = model_xz.grid_params.kDim
            vars = model_xz.grid_params.vars
            npts = size(p_xz.physical, 1)
            for (patch, nv) in ((p_xz, 8), (p_ax, 9))
                for i in 1:npts, v in 1:8    # identical fields on the shared slots; v = 0
                    k = mod1(i, kDim)
                    c = div(i - 1, kDim)
                    amp = v in (4, 5) ? 0.5 : (v == 1 ? 10.0 : (v == 6 ? 100.0 : 1.0e-4))
                    patch.physical[i, v, 1] = amp * sinpi(0.25 * k / kDim) *
                                              sinpi(0.5 * (c + 1) / 9.0)
                end
                spectralTransform!(patch)
                gridTransform!(patch)
            end
            ncols = div(npts, kDim)
            for c in 1:ncols
                Scythe.advance_column(mt_xz, c, 1)
                Scythe.advance_column(mt_ax, c, 1)
            end
            for v in 1:8
                exz = mt_xz.expdot_n[:, v]
                eax = mt_ax.expdot_n[:, v]
                scale = max(maximum(abs.(exz)), 1.0e-12)
                @test maximum(abs.(eax .- exz)) / scale < 1.0e-5
            end
            # v is untouched by the passive dynamics
            @test maximum(abs.(mt_ax.expdot_n[:, 9])) == 0.0
        end
    end

    @testset "axisym: Coriolis and curvature couple u and v" begin
        f = 5.0e-5
        run_once(vseed) = mktempdir() do dir
            mtile, patch, model, _ = make_mc_mtile(dir;
                equation_set = "moist_compressible_axisym",
                iMin = 100.0e3, iMax = 102.0e3, u_side_bc = NeumannBC(), f = f)
            kDim = model.grid_params.kDim
            vars = model.grid_params.vars
            patch.physical[:, vars["u"], 1] .= 2.0
            patch.physical[:, vars["v"], 1] .= vseed
            spectralTransform!(patch)
            gridTransform!(patch)
            for c in 1:div(size(patch.physical, 1), kDim)
                Scythe.advance_column(mtile, c, 1)
            end
            return (expdot = copy(mtile.expdot_n),
                    u = copy(patch.physical[:, vars["u"], 1]),
                    v = copy(patch.physical[:, vars["v"], 1]),
                    r = Scythe.getGridpoints(patch)[:, 1],
                    vars = vars)
        end

        # v = 0: the v equation reduces to dv/dt = -u(f + 0/r) = -f u exactly
        # (advection and curvature of a zero field vanish, no diffusion), pointwise
        # in the spline-fit u.
        r0 = run_once(0.0)
        @test all(isapprox.(r0.expdot[:, r0.vars["v"]], -f .* r0.u; atol = 1.0e-12))

        # v = 1: dv/dt = -u(f + v/r) (a constant v is fit exactly by the Neumann
        # spline, so its derivatives vanish), and relative to the v = 0 run the
        # u tendency gains exactly the absolute-rotation force +(f + v/r)v — the
        # thermodynamic and advective terms are v-independent and cancel in the
        # difference.
        r1 = run_once(1.0)
        @test all(isapprox.(r1.expdot[:, r1.vars["v"]],
                            -(f .+ r1.v ./ r1.r) .* r1.u; rtol = 1.0e-6, atol = 1.0e-10))
        du = r1.expdot[:, r1.vars["u"]] .- r0.expdot[:, r0.vars["u"]]
        @test all(isapprox.(du, (f .+ r1.v ./ r1.r) .* r1.v;
                            rtol = 1.0e-6, atol = 1.0e-10))
        @test all(isfinite.(r1.expdot))
    end

    # ── 3D cylinder (moist_compressible_RLR): the same kernel on the RLR grid
    #    (spline-r, Fourier-λ, spline-z). The radial and vertical mish nodes of an
    #    RLR grid are bitwise identical to the matched RiRk grid's, so a WN0
    #    (axisymmetric) state can be compared POINTWISE against the axisym set. ──
    @testset "RLR: WN0 tendencies match the axisymmetric set" begin
        mktempdir() do tmpdir
            R = 24.0e3
            H = 2000.0
            f = 5.0e-5

            varlist = Scythe.MC_VARS_CYL
            vars = Dict(v => i for (i, v) in enumerate(varlist))
            scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
            axis_bc = merge(scalar_bc, Dict("u" => DirichletBC(), "v" => DirichletBC()))
            wall_bc = merge(scalar_bc, Dict("u" => DirichletBC()))
            topbot_bc = merge(scalar_bc, Dict("w" => DirichletBC()))

            gp_rlr = GridParameters(
                geometry = "RLR", iMin = 0.0, iMax = R, num_cells_i = 6,
                kMin = 0.0, kMax = H, num_cells_k = 8,
                max_wavenumber = Dict(v => 2 for v in keys(vars)),
                BCL = axis_bc, BCR = wall_bc, BCB = topbot_bc, BCT = topbot_bc,
                vars = vars)
            gp_ax = GridParameters(
                geometry = "RiRk", iMin = 0.0, iMax = R, num_cells_i = 6,
                kMin = 0.0, kMax = H, num_cells_k = 8,
                BCL = axis_bc, BCR = wall_bc, BCB = topbot_bc, BCT = topbot_bc,
                vars = vars)

            function build(gp, eqset, zc)
                ref_file = joinpath(tmpdir, "rlr_$(eqset).ref")
                model = ModelParameters(
                    ts = 0.1, integration_time = 1.0, output_interval = 1.0,
                    equation_set = eqset, ref_state_file = ref_file, grid_params = gp,
                    physical_params = Dict(:Khdiff => 25.0, :Kvdiff => 0.0,
                                           :Kvdiff_water => 0.0, :Kv_mudiff => 0.0,
                                           :tau_qss => 10.0, :N_r => 1.0e-3,
                                           :alpha => 0.0, :z_damp => H, :f => f),
                    options = Dict(:semiimplicit => true,
                                   :exact_reference_state => true,
                                   :precipitation => true,
                                   :vertical_mixing => false))
                # kDim is DERIVED for spline-vertical grids — read it off the model's
                # recomputed grid_params, not the input gp
                kD = model.grid_params.kDim
                patch = createGrid(model.grid_params)
                gpts = Scythe.getGridpoints(patch)
                z = gpts[1:kD, zc]
                col = saturated_cloudy_column_mc(z; q_l = 1.0e-3)
                Scythe.write_exact_ref_mc(ref_file, z, col.p_Pa, col.rho_d,
                                          col.rho_v, col.rho_c)
                patch.physical .= 0.0
                # Smooth axisymmetric seeds on every prognostic slot (functions of
                # r and z only, so the RLR state is exactly WN0)
                npts = size(patch.physical, 1)
                amps = Dict(1 => 10.0, 2 => 1.0e-4, 3 => 1.0e-4, 4 => 0.5,
                            5 => 0.25, 6 => 100.0, 7 => 1.0e-5, 8 => 1.0e-5,
                            9 => 0.5)
                for i in 1:npts
                    r = gpts[i, 1]
                    z_i = gpts[i, zc]
                    bump = exp(-((r - 12.0e3) / 6.0e3)^2 - ((z_i - 1000.0) / 500.0)^2)
                    for (v, amp) in amps
                        patch.physical[i, v, 1] = amp * bump
                    end
                end
                spectralTransform!(patch)
                gridTransform!(patch)
                mtile = createModelTile(patch, patch, model,
                                        sparse(Int64[], Int64[], Float64[],
                                               size(patch.spectral, 1),
                                               size(patch.spectral, 2)))
                for c in 1:div(npts, kD)
                    Scythe.advance_column(mtile, c, 1)
                end
                return mtile, gpts, kD
            end

            mt_rlr, gp1, kDim = build(gp_rlr, "moist_compressible_RLR", 3)
            mt_ax, gp2, kD_ax = build(gp_ax, "moist_compressible_axisym", 2)
            @test kDim == kD_ax

            # Map each RLR column to the axisym column at the same radius
            r_ax = gp2[1:kDim:end, 1]
            ncols_rlr = div(size(gp1, 1), kDim)
            worst = 0.0
            scales = Dict(1 => 1.0e2, 2 => 1.0e-5, 3 => 1.0e-5, 4 => 1.0e-2,
                          5 => 1.0e-2, 6 => 1.0e3, 7 => 1.0e-6, 8 => 1.0e-6,
                          9 => 1.0e-2)
            for c in 1:ncols_rlr
                i0 = (c - 1) * kDim
                r_c = gp1[i0 + 1, 1]
                a = findfirst(x -> x == r_c, r_ax)
                @test a !== nothing
                j0 = (a - 1) * kDim
                for v in 1:9
                    d = maximum(abs.(mt_rlr.expdot_n[i0+1:i0+kDim, v] .-
                                     mt_ax.expdot_n[j0+1:j0+kDim, v])) / scales[v]
                    worst = max(worst, d)
                end
            end
            @info "RLR WN0 vs axisym: worst scaled tendency mismatch = $worst"
            @test worst < 1.0e-6
            @test all(isfinite.(mt_rlr.expdot_n))
        end
    end

    # ── 3D Cartesian box (moist_compressible_RRR): slots 4/5 are the full ∂y/∂yy,
    #    v is the y-wind, rotation is a plain f-plane. A y-invariant state on RRR
    #    lives on the same x/z mish nodes as the matched RiRk slice, so the shared
    #    8 slots must match the XZ set pointwise; v feels only -f u. ──
    @testset "RRR: y-invariant tendencies match the XZ slice" begin
        mktempdir() do tmpdir
            L = 24.0e3
            H = 2000.0
            f = 5.0e-5

            varlist9 = Scythe.MC_VARS_CYL
            vars9 = Dict(v => i for (i, v) in enumerate(varlist9))
            vars8 = Dict(v => i for (i, v) in enumerate(Scythe.MC_VARS))
            mkbc(vars) = Dict(v => NeumannBC() for v in keys(vars))

            gp_rrr = GridParameters(
                geometry = "RRR", iMin = 0.0, iMax = L, num_cells_i = 6,
                jMin = 0.0, jMax = L,
                kMin = 0.0, kMax = H, num_cells_k = 8,
                BCL = merge(mkbc(vars9), Dict("u" => DirichletBC())),
                BCR = merge(mkbc(vars9), Dict("u" => DirichletBC())),
                BCU = merge(mkbc(vars9), Dict("v" => DirichletBC())),
                BCD = merge(mkbc(vars9), Dict("v" => DirichletBC())),
                BCB = merge(mkbc(vars9), Dict("w" => DirichletBC())),
                BCT = merge(mkbc(vars9), Dict("w" => DirichletBC())),
                vars = vars9)
            gp_xz = GridParameters(
                geometry = "RiRk", iMin = 0.0, iMax = L, num_cells_i = 6,
                kMin = 0.0, kMax = H, num_cells_k = 8,
                BCL = merge(mkbc(vars8), Dict("u" => DirichletBC())),
                BCR = merge(mkbc(vars8), Dict("u" => DirichletBC())),
                BCB = merge(mkbc(vars8), Dict("w" => DirichletBC())),
                BCT = merge(mkbc(vars8), Dict("w" => DirichletBC())),
                vars = vars8)

            function build(gp, eqset, zc)
                ref_file = joinpath(tmpdir, "rrr_$(eqset).ref")
                model = ModelParameters(
                    ts = 0.1, integration_time = 1.0, output_interval = 1.0,
                    equation_set = eqset, ref_state_file = ref_file, grid_params = gp,
                    physical_params = Dict(:Khdiff => 25.0, :Kvdiff => 0.0,
                                           :Kvdiff_water => 0.0, :Kv_mudiff => 0.0,
                                           :tau_qss => 10.0, :N_r => 1.0e-3,
                                           :alpha => 0.0, :z_damp => H, :f => f),
                    options = Dict(:semiimplicit => true,
                                   :exact_reference_state => true,
                                   :precipitation => true,
                                   :vertical_mixing => false))
                kD = model.grid_params.kDim
                patch = createGrid(model.grid_params)
                gpts = Scythe.getGridpoints(patch)
                z = gpts[1:kD, zc]
                col = saturated_cloudy_column_mc(z; q_l = 1.0e-3)
                Scythe.write_exact_ref_mc(ref_file, z, col.p_Pa, col.rho_d,
                                          col.rho_v, col.rho_c)
                patch.physical .= 0.0
                # Smooth y-invariant seeds on the shared 8 slots (v stays 0)
                npts = size(patch.physical, 1)
                amps = Dict(1 => 10.0, 2 => 1.0e-4, 3 => 1.0e-4, 4 => 0.5,
                            5 => 0.25, 6 => 100.0, 7 => 1.0e-5, 8 => 1.0e-5)
                for i in 1:npts
                    x = gpts[i, 1]
                    z_i = gpts[i, zc]
                    bump = exp(-((x - 12.0e3) / 6.0e3)^2 - ((z_i - 1000.0) / 500.0)^2)
                    for (v, amp) in amps
                        patch.physical[i, v, 1] = amp * bump
                    end
                end
                spectralTransform!(patch)
                gridTransform!(patch)
                mtile = createModelTile(patch, patch, model,
                                        sparse(Int64[], Int64[], Float64[],
                                               size(patch.spectral, 1),
                                               size(patch.spectral, 2)))
                for c in 1:div(npts, kD)
                    Scythe.advance_column(mtile, c, 1)
                end
                return mtile, gpts, kD
            end

            mt_rrr, gp3, kDim = build(gp_rrr, "moist_compressible_RRR", 3)
            mt_xz, gp2, kD_xz = build(gp_xz, "moist_compressible_XZ", 2)
            @test kDim == kD_xz

            # Map each RRR column to the XZ column at the same x
            x_xz = gp2[1:kDim:end, 1]
            ncols_rrr = div(size(gp3, 1), kDim)
            worst = 0.0
            scales = Dict(1 => 1.0e2, 2 => 1.0e-5, 3 => 1.0e-5, 4 => 1.0e-2,
                          5 => 1.0e-2, 6 => 1.0e3, 7 => 1.0e-6, 8 => 1.0e-6)
            worst_v = 0.0
            for c in 1:ncols_rrr
                i0 = (c - 1) * kDim
                x_c = gp3[i0 + 1, 1]
                a = findfirst(==(x_c), x_xz)
                @test a !== nothing
                j0 = (a - 1) * kDim
                for v in 1:8
                    d = maximum(abs.(mt_rrr.expdot_n[i0+1:i0+kDim, v] .-
                                     mt_xz.expdot_n[j0+1:j0+kDim, v])) / scales[v]
                    worst = max(worst, d)
                end
                # v feels exactly -f u (advection/PGF/diffusion of a zero field
                # vanish; pp_y of a y-invariant field is spline roundoff)
                worst_v = max(worst_v,
                    maximum(abs.(mt_rrr.expdot_n[i0+1:i0+kDim, 9] .+
                                 f .* mt_rrr.tile.physical[i0+1:i0+kDim, 4, 1])))
            end
            @info "RRR y-invariant vs XZ: worst scaled mismatch = $worst; " *
                  "max |dv/dt + f u| = $worst_v"
            @test worst < 1.0e-6
            @test worst_v < 1.0e-10
            @test all(isfinite.(mt_rrr.expdot_n))
        end
    end

    # ── 3D spherical shell (moist_compressible_SLR): a zonally-symmetric (WN0)
    #    state on a narrow equatorial band converges to the Cartesian XZ slice
    #    with x = a·(θ − θ₀): at θ ≈ π/2 the metric terms (cotθ, 1 − sinθ) are
    #    O(half-width/a) ≈ 2e-6 relative, and the θ-mish maps affinely onto the
    #    x-mish so columns correspond one-to-one. ──
    @testset "SLR: equatorial WN0 tendencies converge to the XZ slice" begin
        mktempdir() do tmpdir
            a_sphere = 6.371e6
            L = 24.0e3
            H = 2000.0
            th0 = pi / 2 - (L / (2.0 * a_sphere))

            varlist9 = Scythe.MC_VARS_CYL
            vars9 = Dict(v => i for (i, v) in enumerate(varlist9))
            vars8 = Dict(v => i for (i, v) in enumerate(Scythe.MC_VARS))
            mkbc(vars) = Dict(v => NeumannBC() for v in keys(vars))
            side9 = merge(mkbc(vars9), Dict("u" => DirichletBC()))
            side8 = merge(mkbc(vars8), Dict("u" => DirichletBC()))
            tb9 = merge(mkbc(vars9), Dict("w" => DirichletBC()))
            tb8 = merge(mkbc(vars8), Dict("w" => DirichletBC()))

            gp_slr = GridParameters(
                geometry = "SLR", iMin = th0, iMax = th0 + L / a_sphere,
                num_cells_i = 6,
                kMin = 0.0, kMax = H, num_cells_k = 8,
                max_wavenumber = Dict(v => 2 for v in keys(vars9)),
                BCL = side9, BCR = side9, BCB = tb9, BCT = tb9, vars = vars9)
            gp_xz = GridParameters(
                geometry = "RiRk", iMin = 0.0, iMax = L, num_cells_i = 6,
                kMin = 0.0, kMax = H, num_cells_k = 8,
                BCL = side8, BCR = side8, BCB = tb8, BCT = tb8, vars = vars8)

            function build(gp, eqset, zc, xmap)
                ref_file = joinpath(tmpdir, "slr_$(eqset).ref")
                model = ModelParameters(
                    ts = 0.1, integration_time = 1.0, output_interval = 1.0,
                    equation_set = eqset, ref_state_file = ref_file, grid_params = gp,
                    physical_params = Dict(:Khdiff => 25.0, :Kvdiff => 0.0,
                                           :Kvdiff_water => 0.0, :Kv_mudiff => 0.0,
                                           :tau_qss => 10.0, :N_r => 1.0e-3,
                                           :alpha => 0.0, :z_damp => H,
                                           :Omega => 0.0,
                                           :sphere_radius => a_sphere),
                    options = Dict(:semiimplicit => true,
                                   :exact_reference_state => true,
                                   :precipitation => true,
                                   :vertical_mixing => false))
                kD = model.grid_params.kDim
                patch = createGrid(model.grid_params)
                gpts = Scythe.getGridpoints(patch)
                z = gpts[1:kD, zc]
                col = saturated_cloudy_column_mc(z; q_l = 1.0e-3)
                Scythe.write_exact_ref_mc(ref_file, z, col.p_Pa, col.rho_d,
                                          col.rho_v, col.rho_c)
                patch.physical .= 0.0
                npts = size(patch.physical, 1)
                amps = Dict(1 => 10.0, 2 => 1.0e-4, 3 => 1.0e-4, 4 => 0.5,
                            5 => 0.25, 6 => 100.0, 7 => 1.0e-5, 8 => 1.0e-5)
                for i in 1:npts
                    x = xmap(gpts[i, 1])           # arc length from the band edge
                    z_i = gpts[i, zc]
                    bump = exp(-((x - 12.0e3) / 6.0e3)^2 - ((z_i - 1000.0) / 500.0)^2)
                    for (v, amp) in amps
                        patch.physical[i, v, 1] = amp * bump
                    end
                end
                spectralTransform!(patch)
                gridTransform!(patch)
                mtile = createModelTile(patch, patch, model,
                                        sparse(Int64[], Int64[], Float64[],
                                               size(patch.spectral, 1),
                                               size(patch.spectral, 2)))
                for c in 1:div(npts, kD)
                    Scythe.advance_column(mtile, c, 1)
                end
                return mtile, gpts, kD
            end

            mt_slr, gp1, kDim = build(gp_slr, "moist_compressible_SLR", 3,
                                      th -> a_sphere * (th - th0))
            mt_xz, gp2, kD_xz = build(gp_xz, "moist_compressible_XZ", 2, identity)
            @test kDim == kD_xz

            # Map each SLR column to the XZ column at the corresponding arc length
            x_xz = gp2[1:kDim:end, 1]
            ncols_slr = div(size(gp1, 1), kDim)
            worst = 0.0
            scales = Dict(1 => 1.0e2, 2 => 1.0e-5, 3 => 1.0e-5, 4 => 1.0e-2,
                          5 => 1.0e-2, 6 => 1.0e3, 7 => 1.0e-6, 8 => 1.0e-6)
            for c in 1:ncols_slr
                i0 = (c - 1) * kDim
                x_c = a_sphere * (gp1[i0 + 1, 1] - th0)
                d, a_idx = findmin(abs.(x_xz .- x_c))
                @test d < 1.0e-3            # affine mish correspondence (roundoff)
                j0 = (a_idx - 1) * kDim
                for v in 1:8
                    dd = maximum(abs.(mt_slr.expdot_n[i0+1:i0+kDim, v] .-
                                      mt_xz.expdot_n[j0+1:j0+kDim, v])) / scales[v]
                    worst = max(worst, dd)
                end
            end
            @info "SLR equatorial WN0 vs XZ: worst scaled mismatch = $worst " *
                  "(metric terms are O(L/2a) ≈ 2e-6)"
            @test worst < 1.0e-4
            @test all(isfinite.(mt_slr.expdot_n))
        end
    end
end
