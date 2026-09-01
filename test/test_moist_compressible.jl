using Test
using Scythe
using Springsteel

# Tests for the total-energy moist compressible equation set moist_compressible_XZ
# (src/moist_compressible.jl): prognostic p, rho_d, rho_t, u, w, E_t, Q_ss, rho_r, rho_c
# and rho_v — every water species is a slot — with T from the closed-form BF02 retrieval
# on the condensed masses, and the two redundancies (Q_ss against rho_v, rho_v against the
# rho_t budget) removed by the slow nudges of the reconciliation chain.

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
            # The condensate is prognostic, so rho_liq is an INPUT and the retrieval is
            # closed form: no guess, no iteration, and T is exact to rounding rather
            # than to a Newton tolerance.
            Tret = Scythe.retrieve_temperature(st.M, st.rho_d, st.rho_t, st.rho_c)
            @test Tret ≈ c.Tk rtol = 1e-12
            # Diagnostic recovery of the water partition: vapor is the residual
            rho_v = st.rho_t - st.rho_d - st.rho_c
            @test rho_v ≈ st.rho_v rtol = 1e-12 atol = 1e-16
            # ... and the supersaturation the microphysics reads round-trips too
            @test (rho_v - rho_v_sat(Tret, st.p_Pa / 100.0)) ≈ st.Q_ss rtol = 1e-8 atol = 1e-14
        end

        # T is INDEPENDENT of the vapor/cloud split at fixed total water and fixed
        # liquid: it reads only (M, rho_d, rho_t, rho_liq). This is what decouples the
        # microphysics driver from the thermodynamics.
        st = forward_state(285.0, 90000.0, 1.0, 1.0e-3, 0.0, 0.0, 1000.0)
        @test Scythe.retrieve_temperature(st.M, st.rho_d, st.rho_t, st.rho_c) ==
              Scythe.retrieve_temperature(st.M, st.rho_d, st.rho_t, st.rho_c)
        # F(T) = Cfactor*T - rho_liq*L_v(T) - M is linear with a positive slope, so the
        # root is unique and the closed form is exact (not merely convergent).
        F(T) = (st.rho_d * Cpd + (st.rho_t - st.rho_d) * Cpv) * T -
               (st.rho_c * L_v(T)) - st.M
        Ts = 200.0:5.0:330.0
        @test all(diff(F.(Ts)) .> 0.0)
        @test abs(F(Scythe.retrieve_temperature(st.M, st.rho_d, st.rho_t, st.rho_c))) <
              1.0e-6 * abs(st.M)
        # Linearity: the second difference of F vanishes to rounding
        @test maximum(abs.(diff(diff(F.(Ts))))) < 1.0e-9 * abs(F(Ts[1]))
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
    # 4. Condensation rate: gates and regimes (there is no limiter)
    # ──────────────────────────────────────────────
    @testset "qss_condensation_rate gates" begin
        Tk = 285.0; p_hPa = 900.0; ts = 0.1
        rho_vs = rho_v_sat(Tk, p_hPa)
        rho_d = (100.0 * p_hPa - Rv * Tk * rho_vs) / (Rd * Tk)
        Q_s = Scythe.Q_s_energy(Tk, 100.0 * p_hPa, rho_d, rho_vs / rho_d, 1.0e-3)

        # Subsaturated, cloud-free: no droplets to evaporate -> zero
        @test Scythe.qss_condensation_rate(-0.5 * rho_vs, 0.5 * rho_vs, 0.0, rho_d, Tk,
                                           p_hPa, Q_s, ts) == 0.0
        # Strongly subsaturated with a little cloud: the rate is the PHYSICS, and it is
        # allowed to exceed rho_c/ts. That used to be clamped; the clamp made the rate a
        # function of ts and was removed (see qss_condensation_rates).
        rho_c = 1.0e-6 * rho_d
        rate = Scythe.qss_condensation_rate(-0.5 * rho_vs, 0.5 * rho_vs, rho_c, rho_d, Tk,
                                            p_hPa, Q_s, ts)
        @test rate < -rho_c / ts
        # ...and it is bitwise independent of ts, which the clamped version was not.
        for ts2 in (1.0e-3, 1.0, 1.0e6)
            @test Scythe.qss_condensation_rate(-0.5 * rho_vs, 0.5 * rho_vs, rho_c, rho_d, Tk,
                                               p_hPa, Q_s, ts2) === rate
        end
        # Mildly subsaturated with plenty of cloud: slow physical evaporation
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
        # Dry air with spurious positive Q_ss drift: still exactly zero, but now because the
        # DRIVE is clipped to min(Q_ss, rho_v - rho_vs) = -rho_vs < 0, not because a
        # ts-sized ceiling clipped the rate. See the "clipped drive" testset below.
        @test Scythe.qss_condensation_rate(0.5 * rho_vs, 0.0, 0.0, rho_d, Tk, p_hPa,
                                           Q_s, ts) == 0.0
    end

    @testset "AB3-sized water depletion bounds" begin
        # The defect these exist to fix: every water limiter used to be written as a
        # forward-Euler budget while the integrator is AB3. reference/
        # FINDINGS_NEGATIVE_WATER_ATTRIBUTION.md STAGE 3.
        @testset "the forward-Euler bound overshoots under AB3 by 23/12" begin
            ts = 0.3
            avail = 2.0e-3                       # kg/m^3 of cloud
            euler_sink = -avail / ts             # the historical cap
            landed = avail + Scythe._ab3_increment(ts, 5, euler_sink, 0.0, 0.0)
            @test landed ≈ -avail * (23.0 / 12.0 - 1.0) rtol = 1.0e-12
            @test landed / avail ≈ -0.9166666666 rtol = 1.0e-8
        end

        @testset "_ab3_sink_bound solves the increment exactly" begin
            # The bound is defined by "the species lands exactly at zero", at every
            # integrator branch and for arbitrary history.
            for t in 1:5, ts in (0.075, 0.3, 1.0)
                for (avail, s1, s2) in ((1.0e-3, 0.0, 0.0),
                                        (1.0e-3, -2.0e-3, 5.0e-4),
                                        (4.0e-2, 1.0e-2, -3.0e-3),
                                        (0.0, -1.0e-4, 2.0e-4),
                                        (7.5e-5, 1.0e-6, 1.0e-6))
                    b = Scythe._ab3_sink_bound(ts, t, avail, s1, s2)
                    landed = avail + Scythe._ab3_increment(ts, t, b, s1, s2)
                    # Scale the tolerance by the terms actually being cancelled, not by
                    # `avail` alone: with avail = 0 the balance is between the two history
                    # contributions, whose magnitude is ts*|s|.
                    scale = max(avail, ts * abs(s1), ts * abs(s2), 1.0e-12)
                    @test isapprox(landed, 0.0; atol = 1.0e-13 * scale)
                end
            end
        end

        @testset "t = 1 reproduces the forward-Euler bound bitwise" begin
            # The `t = 1` branch must reproduce the forward-Euler budget BITWISE: it is what
            # the census reports the removed caps against, and what runs A-H of
            # reference/FINDINGS_NEGATIVE_WATER_ATTRIBUTION.md were measured under.
            for ts in (0.075, 0.15, 0.3), cf in (1.0, 12.0 / 23.0, 0.5)
                for rho in (0.0, 1.0e-9, 3.7e-3, 2.5)
                    avail = cf * max(rho, 0.0)
                    @test Scythe._ab3_sink_bound(ts, 1, avail, 1.0, -2.0) ===
                          -cf * max(rho, 0.0) / ts
                end
            end
        end

        @testset "a positive bound means the history alone is inadmissible" begin
            # `_ab3_sink_bound` returns the raw solve; callers clamp at 0. The clamp is only
            # ever reached when even a ZERO current-level sink lands the species negative,
            # which is what `:d_*_infeas` counts.
            ts = 0.3
            avail = 1.0e-5
            s1 = 5.0e-4          # a large sink two levels back, weighted -16/12 at t >= 3
            b = Scythe._ab3_sink_bound(ts, 5, avail, s1, 0.0)
            @test b > 0.0
            @test avail + Scythe._ab3_increment(ts, 5, 0.0, s1, 0.0) < 0.0
            @test min(b, 0.0) == 0.0
        end

        @testset "the depletion caps are GONE: rates are pure and ts-independent" begin
            # The caps (`Qdot_c >= -rho_c/ts`, `Qdot_r >= -rho_r/ts`, and the joint vapor
            # ceiling `Qdot_c + Qdot_r <= rho_v/ts`, later their AB3-exact forms) made the
            # RATE — and hence the converged solution — a function of the time step. They were
            # removed; reference/Scythe_moist_compressible.tex §"Departures from the ISHMAEL
            # implementation" (b). This testset replaces the one that pinned their defaults.
            Tk = 285.0; p_hPa = 900.0
            rho_vs = rho_v_sat(Tk, p_hPa)
            rho_d = (100.0 * p_hPa - Rv * Tk * rho_vs) / (Rd * Tk)
            Q_s = Scythe.Q_s_energy(Tk, 100.0 * p_hPa, rho_d, rho_vs / rho_d, 1.0e-3)
            N_r = 1.0e-3

            # Every one of these states is a case the OLD floor/ceiling would have bound at
            # ts = 0.1: a huge drive on a small reservoir, in each channel and in both signs.
            for (Q_ss, rho_v, rho_c, rho_r) in (
                    (-0.5 * rho_vs, 0.5 * rho_vs, 1.0e-8 * rho_d, 1.0e-6),   # tiny cloud+rain
                    (-0.5 * rho_vs, 0.5 * rho_vs, 2.0e-3 * rho_d, 1.0e-9),   # rain below RHO_R_MIN
                    (0.5 * rho_vs, 1.0e-9, 2.0e-3 * rho_d, 1.0e-3),          # no vapor to give
                    (-0.9 * rho_vs, 0.1 * rho_vs, 1.0e-7 * rho_d, 1.0e-3))
                # 1. The returned rate IS drive*invtau/(1+Q_s), split by channel, exactly,
                #    where `drive = min(Q_ss, rho_v - rho_vs)` is the ts-FREE clip.
                Qc, Qr, itc, itr = Scythe.qss_condensation_rates(Q_ss, rho_v, rho_c, rho_r,
                                                                 rho_d, Tk, p_hPa, Q_s,
                                                                 0.1, N_r)
                drive = min(Q_ss, rho_v - rho_vs)
                invtau = itc + itr
                if invtau > 0.0
                    Qdot = drive * invtau / (1.0 + Q_s)
                    @test Qc === (itc == 0.0 ? 0.0 : Qdot * (itc / invtau))
                    @test Qr === (itr == 0.0 ? 0.0 : Qdot * (itr / invtau))
                    @test Qc + Qr ≈ Qdot
                else
                    @test Qc === 0.0 && Qr === 0.0
                end

                # 2. The rate does not depend on ts AT ALL — bitwise, across four decades.
                for ts in (1.0e-3, 0.075, 0.3, 1.0, 1.0e6)
                    r = Scythe.qss_condensation_rates(Q_ss, rho_v, rho_c, rho_r, rho_d, Tk,
                                                      p_hPa, Q_s, ts, N_r)
                    @test r[1] === Qc
                    @test r[2] === Qr
                    @test r[3] === itc
                    @test r[4] === itr
                end
            end

            # 3. The unfloored evaporation genuinely EXCEEDS the old cap where the cap used to
            #    bind — i.e. this is a behaviour change, not a no-op rename.
            rho_c_small = 1.0e-7 * rho_d       # just above the q_c = 1e-8 existence gate
            Qc, = Scythe.qss_condensation_rates(-0.5 * rho_vs, 0.5 * rho_vs, rho_c_small, 0.0,
                                                rho_d, Tk, p_hPa, Q_s, 0.1, N_r)
            @test Qc < -rho_c_small / 0.1        # past the old forward-Euler floor
            @test Qc < 0.0

            # 4. The returned inverse timescales are the channel closures themselves.
            rho_c = 2.0e-3 * rho_d
            rho_r = 1.0e-3
            Qc, Qr, itc, itr = Scythe.qss_condensation_rates(1.0e-3 * rho_vs,
                                                             rho_vs + 1.0e-3 * rho_vs,
                                                             rho_c, rho_r, rho_d, Tk, p_hPa,
                                                             Q_s, 0.1, N_r)
            @test itc === Scythe.invtau_condensation(Tk, p_hPa, 100.0,
                              Scythe.cloud_droplet_radius(100.0, rho_c / rho_d, rho_d))
            @test itr === Scythe.invtau_rain(Tk, p_hPa, N_r, rho_r)
            # ...and the MP closure comes back through the same slot.
            _, _, _, itr_mp = Scythe.qss_condensation_rates(1.0e-3 * rho_vs,
                                                            rho_vs + 1.0e-3 * rho_vs,
                                                            rho_c, rho_r, rho_d, Tk, p_hPa,
                                                            Q_s, 0.1, N_r; N_0 = 8.0e6)
            @test itr_mp === Scythe.invtau_rain_mp(Tk, p_hPa, 8.0e6, rho_r, rho_d)
            # An inactive channel reports an exactly-zero timescale, not a small one.
            _, _, itc0, itr0 = Scythe.qss_condensation_rates(1.0e-5 * rho_vs,
                                                             rho_vs + 1.0e-5 * rho_vs,
                                                             0.0, 0.0, rho_d, Tk, p_hPa,
                                                             Q_s, 0.1, N_r)
            @test itc0 === 0.0 && itr0 === 0.0
        end
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

        # Rain evaporates in subsaturated cloud-free air (the O01 Qevap analogue): the
        # physical rate, with no bound of any kind on it.
        Qc, Qr = Scythe.qss_condensation_rates(-0.5 * rho_vs, 0.5 * rho_vs, 0.0, 1.0e-3,
                                               rho_d, Tk, p_hPa, Q_s, ts, N_r)
        @test Qc == 0.0
        @test Qr < 0.0
        @test Qr ≈ -0.5 * rho_vs * Scythe.invtau_rain(Tk, p_hPa, N_r, 1.0e-3) / (1.0 + Q_s)
        # A long step no longer changes the rate: the `rho_r/ts` clamp that used to engage
        # here is gone, so the same state gives the same rate bitwise at any ts.
        for ts_long in (1.0, 1.0e6)
            c2, r2 = Scythe.qss_condensation_rates(-0.5 * rho_vs, 0.5 * rho_vs, 0.0, 1.0e-3,
                                                   rho_d, Tk, p_hPa, Q_s, ts_long, N_r)
            @test c2 === Qc
            @test r2 === Qr
        end

        # The rain channel's EXISTENCE threshold still gates it (that is a closure property,
        # not a limiter): below RHO_R_MIN there is no channel at all.
        Qc, Qr = Scythe.qss_condensation_rates(-0.5 * rho_vs, 0.5 * rho_vs, rho_c, 1.0e-9,
                                               rho_d, Tk, p_hPa, Q_s, ts, N_r)
        @test Qr == 0.0
        # Just above it the channel is live, and its rate is the closure's — which for a
        # nearly-empty reservoir is far faster than `rho_r/ts`. That is the reported
        # stiffness, not something to clip.
        Qc, Qr = Scythe.qss_condensation_rates(-0.5 * rho_vs, 0.5 * rho_vs, rho_c, 2.0e-8,
                                               rho_d, Tk, p_hPa, Q_s, ts, N_r)
        # The split is proportional, so each channel's rate is just Q_ss/(1+Q_s) times its
        # OWN inverse timescale — the (invtau_c + invtau_r) factors cancel.
        @test Qr ≈ -0.5 * rho_vs * Scythe.invtau_rain(Tk, p_hPa, N_r, 2.0e-8) / (1.0 + Q_s)

        # The vapor enters through the DRIVE clip, not through a ts-sized ceiling. With a
        # consistent vapor (rho_v = rho_vs + Q_ss) the clip is inert and the split is the
        # proportional one; with almost no vapor the drive goes negative and the SAME
        # condensate evaporates instead of growing. Both directions, same arithmetic.
        Qc, Qr = Scythe.qss_condensation_rates(0.5 * rho_vs, 1.5 * rho_vs, rho_c, rho_r,
                                               rho_d, Tk, p_hPa, Q_s, ts, N_r)
        @test Qc / Qr ≈ invtau_c / invtau_r
        @test Qc + Qr ≈ 0.5 * rho_vs * (invtau_c + invtau_r) / (1.0 + Q_s)
        rho_v_tiny = 1.0e-9
        Qc, Qr = Scythe.qss_condensation_rates(0.5 * rho_vs, rho_v_tiny, rho_c, rho_r,
                                               rho_d, Tk, p_hPa, Q_s, ts, N_r)
        @test Qc < 0.0 && Qr < 0.0
        @test Qc + Qr ≈ (rho_v_tiny - rho_vs) * (invtau_c + invtau_r) / (1.0 + Q_s)
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

    @testset "the closure is driven by the CLIPPED drive min(Q_ss, max(rho_v,0) - rho_vs)" begin
        # PROVISIONAL (pending author ratification): the ts-free replacement for the removed
        # vapor ceiling. In the continuum Q_ss IS rho_v - rho_vs and the min never binds;
        # discretely the two prognostics can detach, and this is the instantaneous form of
        # the reconciliation `qss_relaxation` applies on tau_qss.
        #
        # `rho_v` IS THE PROGNOSTIC VAPOR SLOT now, not a residual of the density budget.
        # Nothing about the clip changed with it -- these are the same gates against the
        # same argument -- but what they mean did: the clip compares two transported fields
        # rather than a transported one against a reassembled one, and `MC_QSS_GAP` is the
        # run-time census of exactly the quantity it acts on.
        Tk = 285.0; p_hPa = 900.0; ts = 0.1
        N_r = 1.0e-3
        rho_vs = rho_v_sat(Tk, p_hPa)
        rho_d = (100.0 * p_hPa - Rv * Tk * rho_vs) / (Rd * Tk)
        Q_s = Scythe.Q_s_energy(Tk, 100.0 * p_hPa, rho_d, rho_vs / rho_d, 1.0e-3)

        # (i) DRY AIR with a spuriously positive Q_ss: zero rates AND zero nucleation. THIS
        #     GATE IS LOAD-BEARING and survives the prognostic vapor unchanged — if S came
        #     from the raw Q_ss the Twomey branch would open (invtau_c > 0) and the clipped,
        #     negative drive would then evaporate cloud that does not exist and manufacture
        #     vapor.
        for qss in (1.0e-4 * rho_vs, 0.02, 0.5 * rho_vs, 5.0 * rho_vs)
            Qc, Qr, itc, itr = Scythe.qss_condensation_rates(qss, 0.0, 0.0, 0.0, rho_d, Tk,
                                                             p_hPa, Q_s, ts, N_r)
            @test Qc === 0.0
            @test Qr === 0.0
            @test itc === 0.0        # the nucleation branch never opened
            @test itr === 0.0
        end
        # ...and with rain present in that dry column, still no deposition onto rain.
        Qc, Qr, itc, _ = Scythe.qss_condensation_rates(0.5 * rho_vs, 0.0, 0.0, 1.0e-3, rho_d,
                                                       Tk, p_hPa, Q_s, ts, N_r)
        @test itc === 0.0
        @test Qc === 0.0
        @test Qr < 0.0               # the clipped drive is EVAPORATIVE, which is correct:
                                     # rain in genuinely dry air does evaporate.

        # (ii) INERT where the two prognostics agree — which is every consistent state,
        #      so the clip changes nothing in a well-behaved column. Compared against the
        #      unclipped formula in cloud and in clear air, both signs. NOT bitwise, and it
        #      cannot be: the clip evaluates `(rho_vs + q) - rho_vs`, which differs from `q`
        #      by one rounding of the LARGER number. The relative size of that error is
        #      `eps*rho_vs/|Q_ss|`, so it GROWS as the supersaturation shrinks — 2e-16 at
        #      |Q_ss| ~ rho_vs, but 2e-11 at |Q_ss| = 1e-5*rho_vs, which is the loosest case
        #      swept here. That scaling is the real price of the clip in a consistent column
        #      and is why the tolerance is 1e-9 rather than machine epsilon.
        rho_c = 2.0e-3 * rho_d
        rho_r = 1.0e-3
        for qss in (1.0e-3 * rho_vs, 1.0e-5 * rho_vs, -1.0e-4 * rho_vs, -0.5 * rho_vs)
            for (rc, rr) in ((rho_c, rho_r), (rho_c, 0.0), (0.0, rho_r))
                Qc, Qr, itc, itr = Scythe.qss_condensation_rates(qss, rho_vs + qss, rc, rr,
                                                                 rho_d, Tk, p_hPa, Q_s,
                                                                 ts, N_r)
                invtau = itc + itr
                if invtau > 0.0
                    unclipped = qss * invtau / (1.0 + Q_s)
                    @test Qc ≈ (itc == 0.0 ? 0.0 : unclipped * (itc / invtau)) rtol = 1.0e-9
                    @test Qr ≈ (itr == 0.0 ? 0.0 : unclipped * (itr / invtau)) rtol = 1.0e-9
                    # An inactive channel is still EXACTLY zero, not merely small.
                    itc == 0.0 && @test Qc === 0.0
                    itr == 0.0 && @test Qr === 0.0
                end
            end
        end

        # (iii) Q_ss > rho_v - rho_vs > 0: BOTH drives condense, but the rate must use the
        #       SMALLER one. A detached Q_ss cannot buy condensation the vapor cannot fund.
        gap = 1.0e-3 * rho_vs                       # the honest supersaturation
        detached = 0.5 * rho_vs                     # what the prognostic slot claims
        @test detached > gap > 0.0
        Qc, Qr, itc, itr = Scythe.qss_condensation_rates(detached, rho_vs + gap, rho_c,
                                                         rho_r, rho_d, Tk, p_hPa, Q_s,
                                                         ts, N_r)
        @test Qc > 0.0 && Qr > 0.0                  # still condensing
        @test Qc + Qr ≈ gap * (itc + itr) / (1.0 + Q_s)
        honest = Scythe.qss_condensation_rates(gap, rho_vs + gap, rho_c, rho_r, rho_d, Tk,
                                               p_hPa, Q_s, ts, N_r)
        @test Qc ≈ honest[1] rtol = 1.0e-12         # the honest-drive rate (see (ii) on ULPs)
        @test Qr ≈ honest[2] rtol = 1.0e-12
        # ...and it is strictly less than the detached slot would have bought.
        @test Qc + Qr < detached * (itc + itr) / (1.0 + Q_s)
        clip_c = Qc; clip_r = Qr

        # (iv) The EVAPORATION side is untouched whenever rho_v - rho_vs >= Q_ss, which is
        #      every subsaturated state the density budget agrees with — including one where
        #      the vapor is far ABOVE what Q_ss claims (the min then picks Q_ss).
        for (qss, rho_v) in ((-0.5 * rho_vs, 0.5 * rho_vs),      # consistent
                             (-0.5 * rho_vs, 3.0 * rho_vs),      # vapor-rich, Q_ss binds
                             (-1.0e-4 * rho_vs, rho_vs))         # marginal
            Qc, Qr, itc, itr = Scythe.qss_condensation_rates(qss, rho_v, rho_c, rho_r, rho_d,
                                                             Tk, p_hPa, Q_s, ts, N_r)
            @test Qc + Qr ≈ qss * (itc + itr) / (1.0 + Q_s)
            @test Qc < 0.0 && Qr < 0.0
        end

        # (v) NEGATIVE vapor (an unresolved spike's undershoot, not a physical state): the
        #     max(rho_v, 0) inside the clip floors the drive at -rho_vs, the maximum-dryness
        #     evaporation drive — the rate must NOT scale with the size of the partition error.
        for rho_v_neg in (-1.0e-6, -7.2e-4, -0.5)
            Qc, Qr, itc, itr = Scythe.qss_condensation_rates(0.5 * rho_vs, rho_v_neg, rho_c,
                                                             rho_r, rho_d, Tk, p_hPa, Q_s,
                                                             ts, N_r)
            invtau = itc + itr
            @test invtau > 0.0                       # condensate present, channels open
            @test Qc + Qr ≈ -rho_vs * invtau / (1.0 + Q_s) rtol = 1.0e-12
            @test Qc < 0.0 && Qr < 0.0               # evaporative, independent of |rho_v_neg|
        end
        # ...identical rates for wildly different partition errors: the floor, bitwise.
        r_small = Scythe.qss_condensation_rates(0.5 * rho_vs, -1.0e-9, rho_c, rho_r, rho_d,
                                                Tk, p_hPa, Q_s, ts, N_r)
        r_large = Scythe.qss_condensation_rates(0.5 * rho_vs, -1.0e3, rho_c, rho_r, rho_d,
                                                Tk, p_hPa, Q_s, ts, N_r)
        @test r_small[1] === r_large[1] && r_small[2] === r_large[2]
        # ...but a prognostic Q_ss below -rho_vs still passes through the min unfloored (the
        # prognostic is not the pathology the vapor floor addresses).
        Qc, Qr, itc, itr = Scythe.qss_condensation_rates(-2.0 * rho_vs, 0.0, rho_c, rho_r,
                                                         rho_d, Tk, p_hPa, Q_s, ts, N_r)
        @test Qc + Qr ≈ -2.0 * rho_vs * (itc + itr) / (1.0 + Q_s) rtol = 1.0e-12

        # The clip is ts-FREE: that is the whole point of preferring it to the ceiling. Taken
        # on the BINDING state of (iii), where a ts-sized ceiling would have varied wildly.
        for ts2 in (1.0e-3, 1.0, 1.0e6)
            r = Scythe.qss_condensation_rates(detached, rho_vs + gap, rho_c, rho_r, rho_d,
                                              Tk, p_hPa, Q_s, ts2, N_r)
            @test r[1] === clip_c                   # BITWISE across four decades of ts
            @test r[2] === clip_r
        end
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
                           iMin=0.0, iMax=2000.0, f=0.0,
                           consistent_qss=false, extra_options=Dict{Symbol,Any}(),
                           extra_params=Dict{Symbol,Float64}())
        # From `mc_var_names`, not the raw constant, so an option that APPENDS a slot (the
        # two-moment rain number) brings its name along. With no such option declared this
        # returns exactly MC_VARS / MC_VARS_CYL, so every existing caller is unchanged.
        varlist = Scythe.mc_var_names(extra_options;
                                      cyl = equation_set != "moist_compressible_XZ")
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
            physical_params = merge(Dict(:Khdiff => Khdiff, :Kvdiff => Kvdiff,
                                   :Kvdiff_heat => (Kvdiff_heat === nothing ? Kvdiff : Kvdiff_heat),
                                   :Kvdiff_water => Kvdiff_water,
                                   :Kv_mudiff => 0.0, :tau_qss => tau_qss, :N_r => N_r,
                                   :alpha => alpha, :z_damp => z_damp, :f => f),
                                   extra_params),
            options = merge(Dict{Symbol,Any}(:semiimplicit => semiimplicit,
                           :exact_reference_state => true,
                           :consistent_qss_reference => consistent_qss,
                           :precipitation => precipitation, :vertical_mixing => false),
                           extra_options),
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

    @testset "mc_stiffness_census! reports stiffness and changes nothing" begin
        # The census is what stands in place of the removed depletion caps: it measures
        # ts/tau per relaxation channel and REPORTS under-resolution instead of absorbing it.
        mktempdir() do tmpdir
            mtile, patch, model, _ = make_mc_mtile(tmpdir)
            st = mtile.mc_water_stats
            st .= 0.0
            crow = Scythe.MC_STIFF_FIRST + Scythe.MC_STIFF_N * (Scythe.MC_STIFF_C - 1)
            rrow = Scythe.MC_STIFF_FIRST + Scythe.MC_STIFF_N * (Scythe.MC_STIFF_R - 1)
            tid = Threads.threadid()

            # A BENIGN state: ts/tau under 1 everywhere. The max is recorded; nothing is
            # counted as an exceedance.
            itc = [0.1, 0.5, 0.9]
            itr = [0.0, 0.2, 0.4]
            before = copy(mtile.expdot_n)
            Scythe.mc_stiffness_census!(mtile, 1.0, itc, itr)
            @test st[crow, tid] == 0.9
            @test st[crow + 1, tid] == 0.0
            @test st[rrow, tid] == 0.4
            @test st[rrow + 1, tid] == 0.0
            # It is a MEASUREMENT: neither the rates it reads nor the tendencies move.
            @test itc == [0.1, 0.5, 0.9]
            @test itr == [0.0, 0.2, 0.4]
            @test mtile.expdot_n == before

            # A STIFF state: ts = 2 puts two cloud points past ts/tau = 1.
            Scythe.mc_stiffness_census!(mtile, 2.0, [0.6, 0.51, 0.1], [0.05, 0.1, 0.2])
            @test st[crow, tid] == 1.2            # running max over the run
            @test st[crow + 1, tid] == 2.0        # 1.2 and 1.02 exceed; 0.2 does not
            @test st[rrow, tid] == 0.4            # 2.0*0.2 ties the previous max
            @test st[rrow + 1, tid] == 0.0        # rain still resolved

            # Cumulative, and the max never regresses on a subsequent quiet step.
            Scythe.mc_stiffness_census!(mtile, 1.0e-3, [0.6, 0.51, 0.1], [0.05, 0.1, 0.2])
            @test st[crow, tid] == 1.2
            @test st[crow + 1, tid] == 2.0

            # The warning is UNCONDITIONAL (no :stiffness_trace needed) and fires ONCE.
            model.options[:stiffness_trace] = 0
            st[Scythe.MC_STIFF_WARNED, 1] = 0.0
            st[crow, 1] = 5.0
            @test_logs (:warn,) match_mode = :any Scythe.mc_stiffness_trace(mtile, 3)
            @test st[Scythe.MC_STIFF_WARNED, 1] == 1.0
            @test_logs Scythe.mc_stiffness_trace(mtile, 4)      # silent from here on
            # ...and the periodic report is the separate, opt-in channel.
            model.options[:stiffness_trace] = 2
            @test_logs (:info,) match_mode = :any Scythe.mc_stiffness_trace(mtile, 4)
            @test_logs Scythe.mc_stiffness_trace(mtile, 5)      # off-interval: nothing
            delete!(model.options, :stiffness_trace)

            # A benign run never warns at all.
            st .= 0.0
            Scythe.mc_stiffness_census!(mtile, 0.1, [0.5, 1.0], [0.1, 2.0])
            @test_logs Scythe.mc_stiffness_trace(mtile, 6)
            @test st[Scythe.MC_STIFF_WARNED, 1] == 0.0
        end
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
            # (p ~ 1e5 Pa, E_t ~ 2e8 J/m^3, densities ~ 1). Slot 10 is the prognostic
            # vapor, which is a density like slots 2/3/8/9.
            scales = Dict(1 => 1.0e5, 2 => 1.0, 3 => 1.0, 4 => 1.0, 5 => 1.0,
                          6 => 2.0e8, 7 => 1.0e-2, 8 => 1.0, 9 => 1.0, 10 => 1.0)
            for v in 1:10
                @test maximum(abs.(mtile.expdot_n[:, v])) / scales[v] < 1.0e-9
                @test maximum(abs.(mtile.var_np1[:, v])) / scales[v] < 1.0e-9
            end
            @test all(isfinite.(mtile.var_np1))

            # THE RECONCILIATION GAP IS IDENTICALLY ZERO AT REST, BITWISE — not small,
            # zero. That is the property the whole prognostic-vapor construction rests on:
            # the slot is carried against the DERIVED reference rho_tbar - rho_dbar -
            # rho_cbar (`Scythe.vapor_slot`), which is the same expression `res_rho_t`
            # reassembles from the same three fitted columns, so `rho_v_reconcile` returns
            # an exact 0.0 and the resting base stays a discrete fixed point. Reading the
            # per-thread scratch is legitimate here: the loop above ran serially on this
            # thread, so these columns hold the last column's state.
            S = mtile.mc_scratch[Threads.threadid()]
            @test S.res_rho_t == S.rho_v          # `==` on Float64, deliberately
            @test all(iszero, S.VREC)
            # The vapor slot's tendency is then the phase change and nothing else. On this
            # SATURATED base that is not identically zero: the fitted rho_vbar and
            # rho_vs(T_retrieved) differ at the fit level, so the closure runs at ~1e-17 —
            # the same reference-state crumb the "resting base is untouched by full moist
            # diffusion" testset documents, and the one `consistent_qss_reference` removes.
            # What must be exactly zero is the TRANSPORT and the NUDGE, and they are.
            rv_i = mtile.mc_slots.rho_v
            @test maximum(abs.(mtile.expdot_n[:, rv_i])) < 1.0e-16
            @test mtile.expdot_n[:, rv_i] == -mtile.expdot_n[:, 9]   # phase change only
        end
    end

    """
    Supersaturate the lower half of every column by `dq` (a mixing-ratio increment), as a
    PHYSICALLY CONSISTENT water perturbation.

    A Q_ss-only bump is not one and no longer does anything: Q_ss is the microphysics'
    tracker, the retrieval never reads it, and the condensation drive is clipped by what the
    VAPOR can actually fund — `min(Q_ss, max(rho_v,0) - rho_vs)`. Adding supersaturation
    means adding water, so the seed moves rho_v and rho_t together and compensates p and E_t
    (`dp = R_v T dm`, `dE_t = (C_vv T + g z) dm`) so the temperature retrieval is untouched
    and the state stays at rest. This is the same T-invariant vapor seed the water-diffusion
    and boundary-layer testsets use.
    """
    function supersaturate_lower_half!(patch, mtile, model, col, dq)
        kDim = model.grid_params.kDim
        vars = model.grid_params.vars
        rho_dbar = Springsteel.ref_rho_d(mtile.ref_state)[:, 1]
        z = Scythe.getGridpoints(patch)[1:kDim, 2]
        for i in 1:size(patch.physical, 1)
            k = mod1(i, kDim)
            k <= div(kDim, 2) || continue
            dm = rho_dbar[k] * dq
            Tref = col.Tk[k]
            patch.physical[i, vars["Q_ss"], 1] = dm
            patch.physical[i, vars["rho_v"], 1] = dm
            patch.physical[i, vars["rho_t"], 1] = dm
            patch.physical[i, vars["p"], 1] = Rv * Tref * dm
            patch.physical[i, vars["E_t"], 1] = ((Cvv * Tref) + (Scythe.gravity * z[k])) * dm
        end
        spectralTransform!(patch)
        gridTransform!(patch)
        return patch
    end

    @testset "condensation closure at rest" begin
        mktempdir() do tmpdir
            mtile, patch, model, col = make_mc_mtile(tmpdir)
            kDim = model.grid_params.kDim
            vars = model.grid_params.vars
            qss_i = vars["Q_ss"]

            npts = size(patch.physical, 1)
            supersaturate_lower_half!(patch, mtile, model, col, 5.0e-5)

            ncols = div(npts, kDim)
            for c in 1:ncols
                Scythe.advance_column(mtile, c, 1)
            end

            perturbed = [i for i in 1:npts if mod1(i, kDim) <= div(kDim, 2)]
            # Exact first law: no condensation source in E_t or rho_t at rest
            @test maximum(abs.(mtile.expdot_n[:, vars["E_t"]])) == 0.0
            @test maximum(abs.(mtile.expdot_n[:, vars["rho_t"]])) == 0.0
            @test maximum(abs.(mtile.expdot_n[:, vars["rho_d"]])) == 0.0
            # Latent heating raises pressure; supersaturation relaxes.
            #
            # THE HEATING IS NO LONGER A TENDENCY TERM. Condensation relaxes `Q_ss`, so it is
            # withheld from the multistep and applied as a direct increment at the step-mean
            # rate (`relaxation_adjustment_qss!`), carrying the same `(R_m/C_vt)(L_v −
            # R_vC_ptT/R_m)` bracket slot 1 carried it in. The physical statement is unchanged
            # and is read off that increment instead. The scratch holds the LAST column
            # advanced, and every column carries the same lower-half seed, so its lower half is
            # the perturbed set.
            S = mtile.mc_scratch[Threads.threadid()]
            half = div(kDim, 2)
            @test all(S.Qdot_bar[1:half] .> 0.0)         # condensing
            @test all(S.etd_d1[1:half] .> 0.0)           # ...and that raises the pressure
            # The supersaturation relaxes: downward in what the multistep still carries, and
            # downward in the value the exponential update actually writes.
            @test all(mtile.expdot_n[perturbed, qss_i] .< 0.0)
            @test all(mtile.var_np1[perturbed, qss_i] .<
                      mtile.tile.physical[perturbed, qss_i, 1])
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
            rho_cbar = Springsteel.ref_rho_c(rs)[:, 1]
            Tbar = Springsteel.reference_temperature(rs)
            z = Scythe.getGridpoints(patch)[1:kDim, 2]
            for i in 1:npts
                k = mod1(i, kDim)
                v = mtile.var_np1
                p = v[i, vars["p"]] + pbar[k]
                rho_d = v[i, vars["rho_d"]] + rho_dbar[k]
                rho_t = v[i, vars["rho_t"]] + rho_tbar[k]
                E_t = v[i, vars["E_t"]] + E_tbar[k]
                rho_liq = (v[i, vars["rho_c"]] + rho_cbar[k]) + v[i, vars["rho_r"]]
                ke = 0.5 * (v[i, vars["u"]]^2 + v[i, vars["w"]]^2)
                M = p + E_t - rho_t * (ke + Scythe.gravity * z[k])
                Tk = Scythe.retrieve_temperature(M, rho_d, rho_t, rho_liq)
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
        # top-boundary-localized mode; reference/SI_VERTICAL_CEILING.md). With the
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
            #
            # NOTE: this configuration (ts = 1.0, Co_h 3, 503 m cells) DIVERGES on both
            # paths -- max|u| runs 7e-5 -> 0.055 -> 12.6 -> 1542 m/s over 20 steps, and
            # p' reaches 7.6 bar. That is pre-existing (measured identically on
            # development) and is a property of the configuration, not of either solve.
            # The equivalence is still a valid plumbing check -- two paths, same
            # arithmetic -- but it must be made at a step count where the state is still
            # physical, or it degenerates into comparing two piles of garbage. So: the
            # quantitative comparison runs at 5 steps, and the 20-step comparison is kept
            # as an exact bitwise identity via isequal (NaN-tolerant).
            mstab, pstab, modstab, gpstab, _ = axi_build(tmpdir)
            axi_run!(mstab, pstab, modstab, gpstab, nothing; nsteps=5)
            mstabx, pstabx, modstabx, gpstabx, estabx = axi_build(tmpdir;
                opts_extra=Dict{Symbol,Any}(:exact_si => true, :exact_si_zero_x => true))
            axi_run!(mstabx, pstabx, modstabx, gpstabx, estabx; nsteps=5)
            @test maximum(abs.(pstab.physical[:, :, 1])) < 1.0e3      # still physical
            @test maximum(abs.(pstab.physical[:, :, 1] .-
                               pstabx.physical[:, :, 1])) <= 1.0e-10

            mref, pref, modref, gpref, _ = axi_build(tmpdir)
            axi_run!(mref, pref, modref, gpref, nothing; nsteps=20)
            mzx, pzx, modzx, gpzx, ezx = axi_build(tmpdir;
                opts_extra=Dict{Symbol,Any}(:exact_si => true, :exact_si_zero_x => true))
            axi_run!(mzx, pzx, modzx, gpzx, ezx; nsteps=20)
            @test all(isequal.(pref.physical[:, :, 1], pzx.physical[:, :, 1]))

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

    @testset "qss reconciliation drives Q_ss to the prognostic vapor's supersaturation" begin
        # Q_ss is redundant now that EVERY water species is prognostic: rho_v - rho_vs(T,p)
        # already fixes it. `qss_relaxation` is what keeps the prognostic tracker and the
        # prognostic vapor from drifting apart (and it is thermodynamically INERT, because
        # retrieve_temperature does not read Q_ss at all). Its `rho_v` argument is the
        # PROGNOSTIC SLOT — feeding it the retrieval was once forbidden, because one branch
        # of the retrieval was built out of Q_ss + rho_vs and the term self-annihilated; a
        # transported slot is not that by construction, so the anchor is real.
        Tk, p_hPa = 290.0, 900.0
        rho_vs = rho_v_sat(Tk, p_hPa)
        tau = 10.0

        # On target: zero. Exactly zero when the target is representable; otherwise
        # to the rounding of (rho_vs + d) - rho_vs, which is the best a difference of
        # two large numbers can do — and precisely why Q_ss is carried prognostically
        # rather than diagnosed from one.
        @test Scythe.qss_relaxation(0.0, rho_vs, rho_vs, tau) == 0.0
        @test Scythe.qss_relaxation(-0.002, rho_vs - 0.002, rho_vs, tau) ≈ 0.0 atol=1e-18
        @test Scythe.qss_relaxation(0.003, rho_vs + 0.003, rho_vs, tau) ≈ 0.0 atol=1e-18

        # Off target: first-order relaxation at rate 1/tau, toward rho_v - rho_vs
        @test Scythe.qss_relaxation(0.0, rho_vs - 0.002, rho_vs, tau) ≈ -0.002 / tau
        @test Scythe.qss_relaxation(0.002, rho_vs, rho_vs, tau) ≈ -0.002 / tau
        # Dry air: the target is -rho_vs (rho_v = 0), which is where Q_ss belongs
        @test Scythe.qss_relaxation(0.0, 0.0, rho_vs, tau) ≈ -rho_vs / tau

        # It is a pure tracker correction: sign always opposes the discrepancy, and the
        # magnitude is the discrepancy over tau — nothing state-dependent hides in it.
        for (Q_ss, rho_v) in ((0.001, 0.02), (-0.005, 0.01), (0.0, 0.0))
            d = Q_ss - (rho_v - rho_vs)
            @test Scythe.qss_relaxation(Q_ss, rho_v, rho_vs, tau) ≈ -d / tau
        end
    end

    @testset "rho_v_reconcile: the second link of the chain" begin
        # The vapor nudge, `(res_rho_t - rho_v)/tau_rec`. rho_t stays prognostic as the
        # conservation anchor, so the set carries one redundancy and its whole content is
        # the gap delta = res_rho_t - rho_v. This removes it at rate 1/tau_rec — a drift
        # correction, never a projection back onto the budget.
        tau_rec = 10.0

        # On target: EXACTLY zero, whatever the value. Nothing to remove and no rounding to
        # introduce, because the two arguments are the same double.
        for x in (0.0, 1.0e-8, 1.2e-2, -3.0e-4)
            @test Scythe.rho_v_reconcile(x, x, tau_rec) === 0.0
        end

        # Off target: first-order relaxation at 1/tau_rec, signed toward res_rho_t.
        @test Scythe.rho_v_reconcile(1.2e-2, 1.1e-2, tau_rec) ≈ 1.0e-3 / tau_rec
        @test Scythe.rho_v_reconcile(1.1e-2, 1.2e-2, tau_rec) ≈ -1.0e-3 / tau_rec
        for (res, rv) in ((1.0e-2, 9.0e-3), (5.0e-3, 6.0e-3), (0.0, -1.0e-6))
            @test Scythe.rho_v_reconcile(res, rv, tau_rec) ≈ (res - rv) / tau_rec
        end

        # The RATE is 1/tau_rec and nothing else: halving tau doubles it exactly, and the
        # law is ts-free (no timestep appears in it anywhere).
        d = 1.0e-3
        @test Scythe.rho_v_reconcile(1.2e-2, 1.2e-2 - d, 5.0) ===
              2.0 * Scythe.rho_v_reconcile(1.2e-2, 1.2e-2 - d, 10.0)

        # It is ODD in the gap and carries no floor: a negative vapor is nudged UP toward
        # the budget rather than clamped, which is the standing rule that negative water is
        # a resolution diagnostic and never a state to enforce.
        @test Scythe.rho_v_reconcile(0.0, -1.0e-3, tau_rec) > 0.0
        @test Scythe.rho_v_reconcile(1.0e-3, 0.0, tau_rec) ===
              -Scythe.rho_v_reconcile(-1.0e-3, 0.0, tau_rec)
    end

    @testset "the vapor nudge decays a seeded gap at 1/tau_rec and is inert in T" begin
        # The nudge ON A COLUMN: seed delta != 0 by moving the vapor slot alone (which
        # leaves rho_t, rho_d and rho_c untouched, so `res_rho_t` is unchanged and the whole
        # of delta is the seed), then check the tendency it produces and that the
        # temperature retrieval does not notice.
        mktempdir() do tmpdir
            mtile, patch, model, col = make_mc_mtile(tmpdir)
            kDim = model.grid_params.kDim
            npts = size(patch.physical, 1)
            ncols = div(npts, kDim)
            rv_i = mtile.mc_slots.rho_v
            tau_rec = get(model.physical_params, :tau_rho_v_rec, 10.0)

            # The resting temperature field, for the inertness check below.
            for c in 1:ncols
                Scythe.advance_column(mtile, c, 1)
            end
            T_rest = copy(mtile.mc_scratch[Threads.threadid()].Tk)

            # A uniform seed: rho_v' -> rho_v' - d everywhere. Written to the mish and NOT
            # refit, so the seed is exactly d at every point (a refit would smooth it and
            # the expected tendency would stop being a single number).
            d = 1.0e-5
            for i in 1:npts
                patch.physical[i, rv_i, 1] -= d
            end

            for c in 1:ncols
                Scythe.advance_column(mtile, c, 2)
            end
            S = mtile.mc_scratch[Threads.threadid()]

            # delta is the seed, to the last bit: nothing else moved.
            @test all(x -> x ≈ d, S.res_rho_t .- S.rho_v)
            # ...and the nudge is delta/tau_rec, restoring (positive: vapor was removed).
            @test all(x -> x ≈ d / tau_rec, S.VREC)
            @test all(>(0.0), S.VREC)

            # THERMODYNAMICALLY INERT. retrieve_temperature reads rho_t, the condensed
            # masses and E_t — never the vapor — so a gap of this size must not move T at
            # all. Bitwise, because the retrieval's inputs are the same doubles.
            @test S.Tk == T_rest
            # The conserved slots take nothing from it either.
            @test maximum(abs.(mtile.expdot_n[:, model.grid_params.vars["rho_t"]])) == 0.0
            @test maximum(abs.(mtile.expdot_n[:, model.grid_params.vars["rho_d"]])) == 0.0
            @test maximum(abs.(mtile.expdot_n[:, model.grid_params.vars["E_t"]])) == 0.0
        end
    end

    @testset "clamp_water! is a conservative phase change, not a mass source" begin
        # The floor is opt-in (options[:clamp_water]); by default clamp_water! only
        # MEASURES. These two testsets exercise the floor itself, so they enable it.
        # The positivity floor moves the deficit out of the prognostic condensates while
        # leaving rho_t, rho_d and E_t untouched, so it only ever moves the PARTITION and
        # the retrieval supplies exactly the latent heat of the implied phase change. This
        # is the property that makes flooring negative water legitimate rather than a fudge.
        # The vapor's half of that phase change now arrives through the reconciliation
        # nudge (the vapor is prognostic and this function does not touch it), so the
        # partition closes on tau_rec instead of within the step; what is asserted below —
        # total water conserved exactly — is unaffected, and is the load-bearing claim.
        mktempdir() do tmpdir
            mtile, patch, model, _ = make_mc_mtile(tmpdir)
            model.options[:clamp_water] = true
            vars = model.grid_params.vars
            kDim = model.grid_params.kDim
            rc_i = vars["rho_c"]; rr_i = vars["rho_r"]
            rt_i = vars["rho_t"]; rd_i = vars["rho_d"]
            rho_cbar = Springsteel.ref_rho_c(mtile.ref_state)[:, 1]
            rho_tbar = Springsteel.ref_rho_t(mtile.ref_state)[:, 1]
            rho_dbar = Springsteel.ref_rho_d(mtile.ref_state)[:, 1]

            mtile.var_np1 .= 0.0
            # Seed a negative cloud and a negative rain in the first column
            deficit_c = -1.0e-6
            deficit_r = -2.0e-7
            mtile.var_np1[1, rc_i] = deficit_c - rho_cbar[1]
            mtile.var_np1[1, rr_i] = deficit_r
            mtile.var_np1[2, rc_i] = -rho_cbar[2]          # exactly zero cloud
            rw_before = [(mtile.var_np1[i, rt_i] + rho_tbar[mod1(i, kDim)]) -
                         (mtile.var_np1[i, rd_i] + rho_dbar[mod1(i, kDim)]) for i in 1:2]
            before = Scythe.water_negativity_report(mtile).total

            Scythe.clamp_water!(mtile, 1, kDim)

            # 1. Nothing negative survives
            @test mtile.var_np1[1, rc_i] + rho_cbar[1] == 0.0
            @test mtile.var_np1[1, rr_i] == 0.0
            # 2. TOTAL WATER is untouched — that is what makes the floor conservative
            rw_after = [(mtile.var_np1[i, rt_i] + rho_tbar[mod1(i, kDim)]) -
                        (mtile.var_np1[i, rd_i] + rho_dbar[mod1(i, kDim)]) for i in 1:2]
            @test rw_after == rw_before
            # 3. An already-admissible point is left EXACTLY alone
            @test mtile.var_np1[2, rc_i] == -rho_cbar[2]
            # 4. The moved mass is accounted, not silent
            @test Scythe.water_negativity_report(mtile).total - before ≈
                  abs(deficit_c) + abs(deficit_r) rtol=1e-12
        end
    end

    @testset "clamp_water! caps condensate by the water present" begin
        # Rule 2: rho_c + rho_r may not exceed rho_w (i.e. the water budget may not imply a
        # negative vapor). The excess comes out of cloud first, then rain.
        mktempdir() do tmpdir
            mtile, patch, model, _ = make_mc_mtile(tmpdir)
            model.options[:clamp_water] = true
            vars = model.grid_params.vars
            kDim = model.grid_params.kDim
            rc_i = vars["rho_c"]; rr_i = vars["rho_r"]
            rt_i = vars["rho_t"]; rd_i = vars["rho_d"]
            rho_cbar = Springsteel.ref_rho_c(mtile.ref_state)[:, 1]
            rho_tbar = Springsteel.ref_rho_t(mtile.ref_state)[:, 1]
            rho_dbar = Springsteel.ref_rho_d(mtile.ref_state)[:, 1]
            rho_w1 = rho_tbar[1] - rho_dbar[1]

            mtile.var_np1 .= 0.0
            # Ask for twice the available water as cloud, plus some rain
            mtile.var_np1[1, rc_i] = (2.0 * rho_w1) - rho_cbar[1]
            mtile.var_np1[1, rr_i] = 0.1 * rho_w1
            Scythe.clamp_water!(mtile, 1, kDim)

            rho_c = mtile.var_np1[1, rc_i] + rho_cbar[1]
            rho_r = mtile.var_np1[1, rr_i]
            @test rho_c >= 0.0 && rho_r >= 0.0
            @test rho_c + rho_r <= rho_w1 + 1.0e-18
            # The implied vapor is exactly zero, not negative: the cap binds
            @test (rho_w1 - rho_c - rho_r) ≈ 0.0 atol=1e-18
        end
    end

    @testset "positivity bounds are available and inert for rho_d/rho_t" begin
        # WHY THIS IS OFF BY DEFAULT, and what this testset is for.
        #
        # `install_positivity_bounds!` already understands "rho_d" and "rho_t" — both are
        # carried as perturbations from a nonzero reference, so their bound is the
        # reference-offset one (-ρ̄ through the support-minimum rule on the k-leg, the
        # negated SB coefficients on the i-leg). Nothing about the machinery is
        # species-specific. But NOTHING ENABLES IT, and that is deliberate:
        #
        #   * rho_d has no rate sinks at all, and rho_t's only sink is the fitted
        #     sedimentation flux divergence, which is not a rate. The AB3 depletion-bound
        #     machinery that sizes the rho_c/rho_r caps (`AB3_REAL_LIMIT` and friends) is
        #     therefore vacuous for these two — there is no cap to size.
        #   * The constraint that actually matters physically is rho_w = rho_t - rho_d >= 0,
        #     a DIFFERENCE of two fields. It is not expressible as a per-field bound on
        #     either one's spline coefficients, so bounding rho_d and rho_t individually
        #     does not buy it.
        #   * Both fields sit ~5 orders of magnitude away from zero in every configuration
        #     run to date (min rho_d/ρ̄_d ≈ 1 - 5e-6 here).
        #
        # So the doctrine holds: measure, don't clamp. What this testset locks is that the
        # option WORKS and is BIT-INERT when it does not bind, so that a future strongly
        # convective run can turn it on (via `GridParameters.positivity`) without first
        # having to debug the plumbing. See benchmarks/FUTURE_WORK.md.
        function positivity_rirk(tmpdir, tag, positivity)
            vars = Dict(v => i for (i, v) in enumerate(Scythe.MC_VARS))
            # Scalars take NeumannBC on all four sides — R1T1, which is bound-safe (the
            # mirror a[1] = a[3] copies rather than combines, so a box constraint on the
            # free coefficients implies it on the slaved ones). Springsteel's
            # `set_lower_bound!` throws for anything else.
            scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
            side_bc = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))
            wall_bc = merge(scalar_bc, Dict("w" => DirichletBC()))
            gp = GridParameters(geometry = "RiRk",
                iMin = 0.0, iMax = 8.0e3, num_cells_i = 4,
                kMin = 0.0, kMax = 4.0e3, num_cells_k = 8,
                positivity = positivity,
                BCL = side_bc, BCR = side_bc, BCB = wall_bc, BCT = wall_bc,
                vars = vars)
            ref_file = joinpath(tmpdir, "positivity_$(tag).ref")
            model = ModelParameters(
                ts = 0.25, integration_time = 5.0, output_interval = 5.0,
                equation_set = "moist_compressible_XZ",
                ref_state_file = ref_file, grid_params = gp,
                physical_params = Dict(:Khdiff => 0.0, :Kvdiff => 0.0,
                                       :Kvdiff_heat => 0.0, :Kvdiff_water => 0.0,
                                       :tau_qss => 10.0, :alpha => 0.0,
                                       :z_damp => 20.0e3, :f => 0.0),
                options = Dict{Symbol,Any}(:semiimplicit => true,
                                           :exact_reference_state => true,
                                           :precipitation => false))
            gp = model.grid_params
            patch = createGrid(gp)
            z = Scythe.getGridpoints(patch)[1:gp.kDim, end]
            col = saturated_cloudy_column_mc(z)
            Scythe.write_exact_ref_mc(ref_file, z, col.p_Pa, col.rho_d, col.rho_v, col.rho_c)
            patch.physical .= 0.0
            # Deterministic broadband w seed (irrational chirp), amplitude large enough to
            # move every prognostic but far too small to threaten positive-definiteness.
            w_i = gp.vars["w"]
            for i in 1:size(patch.physical, 1)
                patch.physical[i, w_i, 1] = 1.0e-2 * sin(0.5 * sqrt(2.0) * i^2)
            end
            spectralTransform!(patch)
            gridTransform!(patch)
            hrm = sparse(Int64[], Int64[], Float64[],
                         size(patch.spectral, 1), size(patch.spectral, 2))
            mtile = createModelTile(patch, patch, model, hrm)
            return mtile, patch, model, gp
        end

        function positivity_run!(mtile, patch, model, gp, nsteps)
            kDim = gp.kDim
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

        mktempdir() do tmpdir
            pos = Dict("rho_d" => Dict(:i => 0.0, :k => 0.0),
                       "rho_t" => Dict(:i => 0.0, :k => 0.0))
            mB, pB, moB, gpB = positivity_rirk(tmpdir, "on", pos)
            mO, pO, moO, gpO = positivity_rirk(tmpdir, "off",
                                               Dict{String,Dict{Symbol,Float64}}())
            vars = gpB.vars
            kDim = gpB.kDim
            rho_dbar = view(Springsteel.ref_rho_d(mB.ref_state), :, 1)
            rho_tbar = view(Springsteel.ref_rho_t(mB.ref_state), :, 1)

            # 1. The bounds installed at all, and installed as REFERENCE-OFFSET bounds
            #    rather than the factory's constant 0 (which on a perturbation would pin
            #    the field at or above its own reference).
            for (name, bar) in (("rho_d", rho_dbar), ("rho_t", rho_tbar))
                v = vars[name]
                kcol = pB.kbasis.data[v]
                @test length(kcol.lower) == kcol.params.bDim
                # Support-minimum rule: L[i] = -min(ρ̄ over the 4 cells Bᵢ covers), so every
                # entry lies in [-max ρ̄, -min ρ̄] and none of them is the factory's 0.
                @test all(-maximum(bar) - 1e-12 .<= kcol.lower .<= -minimum(bar) + 1e-12)
                @test maximum(kcol.lower) < 0.0
                for z in axes(pB.ibasis.data, 1)
                    @test length(pB.ibasis.data[z, v].lower) == gpB.b_iDim
                end
                # Control run: no bound anywhere.
                @test isempty(pO.kbasis.data[v].lower)
                @test isempty(pO.ibasis.data[1, v].lower)
            end

            nsteps = 20
            # Exact 2-D mass integral: the mish points are Gauss nodes in BOTH directions,
            # so the Gauss-weight quadrature on them integrates the spline representation
            # exactly (the same rule the Galerkin solver uses). An unweighted point sum
            # would confuse redistribution among unequally weighted nodes with a real drift.
            cellw = (npts, ncells, len) -> begin
                _, qw = Springsteel.CubicBSpline._quadrature_rule(div(npts, ncells),
                                                                 gpB.quadrature)
                repeat(qw .* (len / ncells), outer = ncells)
            end
            Wi = cellw(gpB.iDim, gpB.num_cells_i, gpB.iMax - gpB.iMin)
            Wk = cellw(gpB.kDim, gpB.num_cells_k, gpB.kMax - gpB.kMin)
            massfun = (p, name, bar) -> sum(Wi[div(i - 1, kDim) + 1] * Wk[mod1(i, kDim)] *
                                            (p.physical[i, vars[name], 1] +
                                             bar[mod1(i, kDim)])
                                            for i in axes(p.physical, 1))
            md0 = massfun(pB, "rho_d", rho_dbar)
            mt0 = massfun(pB, "rho_t", rho_tbar)
            positivity_run!(mB, pB, moB, gpB, nsteps)
            positivity_run!(mO, pO, moO, gpO, nsteps)

            @test all(isfinite.(pB.physical[:, :, 1]))

            # 2. The limiter never had to act. A nonzero shortfall would mean a column was
            #    INFEASIBLE (its total mass below what any admissible spline can carry), so
            #    the limiter created mass instead of redistributing it. Read per leg: the
            #    k-leg alone hid the i-leg entirely in the rho_c attribution (see
            #    reference/FINDINGS_NEGATIVE_WATER_ATTRIBUTION.md).
            for name in ("rho_d", "rho_t")
                v = vars[name]
                @test Springsteel.CubicBSpline.bound_shortfall(pB.kbasis.data[v]) == 0.0
                @test sum(Springsteel.CubicBSpline.bound_shortfall(pB.ibasis.data[z, v])
                          for z in axes(pB.ibasis.data, 1)) == 0.0
            end
            # ... because both fields stay ~5 orders of magnitude clear of zero.
            @test minimum(pB.physical[i, vars["rho_d"], 1] / rho_dbar[mod1(i, kDim)] + 1.0
                          for i in axes(pB.physical, 1)) > 0.99
            @test minimum(pB.physical[i, vars["rho_t"], 1] / rho_tbar[mod1(i, kDim)] + 1.0
                          for i in axes(pB.physical, 1)) > 0.99

            # 3. BITWISE inert. `SAtransform_bounded!` is a conservative clip-and-shrink; a
            #    non-binding bound must be the identity, not "almost" the identity. Any
            #    drift here means the bounded solve is doing arithmetic the unbounded one
            #    is not, which would silently change every run that enables it.
            @test pB.physical == pO.physical
            @test pB.spectral == pO.spectral

            # 4. Mass. rho_d has no sinks whatsoever and rho_t's only sink (sedimentation)
            #    is off here, so both domain integrals are conserved to rounding — and the
            #    bounds neither created nor destroyed any of it.
            @test abs(massfun(pB, "rho_d", rho_dbar) - md0) / md0 < 1e-12
            @test abs(massfun(pB, "rho_t", rho_tbar) - mt0) / mt0 < 1e-12
        end
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
            rho_cbar = Springsteel.ref_rho_c(ref0)[:, 1]
            Tbar = Springsteel.reference_temperature(ref0)
            zs = gp0[:, 2]

            function retrieved(mt, i, k)
                p = mt.var_np1[i, 1] + pbar[k]
                rho_d = mt.var_np1[i, 2] + rho_dbar[k]
                rho_t = mt.var_np1[i, 3] + rho_tbar[k]
                E_t = mt.var_np1[i, 6] + E_tbar[k]
                rho_liq = (mt.var_np1[i, 9] + rho_cbar[k]) + mt.var_np1[i, 8]
                ke = 0.5 * (mt.var_np1[i, 4]^2 + mt.var_np1[i, 5]^2)
                M = p + E_t - rho_t * (ke + Scythe.gravity * zs[i])
                return Scythe.retrieve_temperature(M, rho_d, rho_t, rho_liq), p, rho_d
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
        rho_cbar = Springsteel.ref_rho_c(ref)[:, 1]
        Tbar = Springsteel.reference_temperature(ref)
        n = size(mtile.var_np1, 1)
        T = zeros(n)
        for i in 1:n
            k = mod1(i, kDim)
            p = mtile.var_np1[i, 1] + pbar[k]
            rho_d = mtile.var_np1[i, 2] + rho_dbar[k]
            rho_t = mtile.var_np1[i, 3] + rho_tbar[k]
            E_t = mtile.var_np1[i, 6] + E_tbar[k]
            rho_liq = (mtile.var_np1[i, 9] + rho_cbar[k]) + mtile.var_np1[i, 8]
            ke = 0.5 * (mtile.var_np1[i, 4]^2 + mtile.var_np1[i, 5]^2)
            M = p + E_t - rho_t * (ke + Scythe.gravity * zs[i])
            T[i] = Scythe.retrieve_temperature(M, rho_d, rho_t, rho_liq)
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
            # Cloud is PROGNOSTIC now, so the conversion is an explicit equal and
            # opposite pair rather than something the residual absorbs silently:
            # every gram slot 8 gains, slot 9 loses.
            drc = m_on.var_np1[:, 9] .- m_off.var_np1[:, 9]
            drr = m_on.var_np1[:, 8] .- m_off.var_np1[:, 8]
            @test maximum(abs.(drc .+ drr)) < 1.0e-18
            @test all(drc .<= 0.0)
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
            # ...and rho_t tracks rho_r through the sedimentation flux (identical
            # -dF/dz in both slots) EXCEPT where the positivity floor fired. The
            # spline fit of the Gaussian bump undershoots negative at its edges, and
            # clamp_water! converts that negative rain to vapor: rho_r moves, rho_t
            # (total water) deliberately does not. So the residual is bounded by the
            # clamped mass, which is itself accounted rather than silent.
            @test maximum(abs.(d3 .- d8)) <= 1.0e-14
            # The fitted bump undershoots negative on its flanks (the spline cannot
            # represent a Gaussian spike exactly). That is MEASURED, not repaired:
            # options[:clamp_water] is off by default precisely because flooring it
            # would rectify the undershoot into one-signed latent heating.
            @test Scythe.water_negativity_report(m_on).count > 0
            @test Scythe.water_negativity_report(m_on).worst_dT < 1.0e-3
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
            # Slot 10 is the prognostic VAPOR, which now has a diffusion leg of its own
            # in `_diffusion_water_step!` — its own AI2*/AM2 staging and its own
            # factorization — instead of being the implied remainder of the other three.
            # Its resting perturbation is zero for the same reason theirs is, and the leg
            # must not disturb that.
            for slot in (1, 2, 3, 4, 5, 6, 8, 10)
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
            for slot in (2, 3, 4, 5, 7, 8, 10)
                @test maximum(abs.(pK.physical[:, slot, 1])) < 1.0e-10
            end
        end
    end

    @testset "consistent_qss_reference: the resting base is an EXACT fixed point" begin
        # THE GATE of reference/HANDOFF_REFERENCE_STATE.md. Springsteel builds Q_ssbar
        # pointwise from the EOS temperature and then FITS it, but the equation set
        # retrieves T from the FITTED (pbar, E_tbar, rho_tbar) -- so rho_v* differs,
        # the clamped partition leaves a residual cloud, and qss_condensation_rates
        # fires on a state that is supposed to be at rest. On the TC reference that
        # was expdot[p] = 2.3 Pa/s with zero perturbation and zero physics.
        #
        # The crumb this removes is the one the "resting base is untouched by full
        # moist diffusion" testset documents above ("the file-derived Q_ssbar vs the
        # retrieved-T saturation differ at ~1e-17, so Qdot != 0 at rest even with
        # diffusion off"). With the flag on it is not ~1e-17, it is EXACTLY zero.
        for (dry, q_l) in ((true, 0.0), (false, 1.0e-3))
            # OFF: the crumb is present (this is the defect, asserted so the test
            # fails loudly if someone "fixes" it by changing the default).
            mktempdir() do tmpdir
                m0, p0, mod0, _ = make_mc_mtile(tmpdir; dry=dry, q_l=q_l,
                                                consistent_qss=false)
                step_mc!(m0, p0, mod0, 5)
                @test maximum(abs.(p0.physical)) > 0.0
            end
            # ON, CLOUD-FREE: every prognostic slot stays bit-exactly zero. Not
            # "small" -- zero. The construction runs the driver's own expressions, so
            # Q_ssbar IS the diagnosed supersaturation bit-for-bit, both condensation
            # gates are shut, and the reconciliation term is exactly 0.0.
            #
            # ON, CLOUDY: exact zero is not attainable and should not be asserted. The
            # saturated branch has to solve g(rho_c) = rho_c - (rho_w - rho_vs(T(rho_c)))
            # by Newton, and a root found to 1e-15 in rho_c leaves the state ~1e-16 off
            # the saturation manifold -- so either Qdot or the Q_ss reconciliation is
            # nonzero at the last bit, whichever the branch chooses to zero exactly.
            # What matters is that it is at rounding and does not accumulate.
            mktempdir() do tmpdir
                m1, p1, mod1, _ = make_mc_mtile(tmpdir; dry=dry, q_l=q_l,
                                                consistent_qss=true)
                step_mc!(m1, p1, mod1, 5)
                if q_l == 0.0
                    @test maximum(abs.(p1.physical)) == 0.0
                else
                    vars1 = mod1.grid_params.vars
                    @test maximum(abs.(p1.physical[:, vars1["p"], 1])) < 1.0e-9
                    @test maximum(abs.(p1.physical[:, vars1["E_t"], 1])) < 1.0e-6
                    for nm in ("rho_d", "rho_t", "u", "w", "Q_ss", "rho_r", "rho_c",
                               "rho_v")
                        @test maximum(abs.(p1.physical[:, vars1[nm], 1])) < 1.0e-12
                    end
                end
            end
        end
    end

    @testset "consistent_qss_reference: a saturated base keeps its cloud" begin
        # The cloudy branch must put the level on the Q_ss = 0 manifold rather than
        # forcing all the water into vapor -- that would destroy the base cloud which
        # is exactly what makes a BF02-style saturated base neutrally buoyant.
        mktempdir() do tmpdir
            m, p, mod, _ = make_mc_mtile(tmpdir; dry=false, q_l=1.0e-3,
                                         consistent_qss=true)
            ref = m.ref_state
            @test all(Springsteel.ref_qss(ref)[:, 1] .== 0.0)          # on the manifold
            @test all(Springsteel.ref_rho_c(ref)[:, 1] .> 0.0)         # cloud retained
            # ...and the dry base takes the other branch: subsaturated, so Q_ssbar < 0.
        end
        mktempdir() do tmpdir
            m, p, mod, _ = make_mc_mtile(tmpdir; dry=true, consistent_qss=true)
            @test all(Springsteel.ref_qss(m.ref_state)[:, 1] .< 0.0)
        end
    end

    # ──────────────────────────────────────────────
    # 8b. A running cloudy column: the shared harness
    # ──────────────────────────────────────────────
    # This section used to hold the regime-blended vapor retrieval's model-level gates. The
    # blend is retired — the vapor is a prognostic slot, so there is no diagnostic left to
    # choose a representation for — and what survives it is the HARNESS: a small saturated
    # RiRk patch that actually makes cloud, used by the condensate-floor, condensate- and
    # rain-transform, and prognostic-vapor testsets below.

    """
    Small RiRk moist patch on a saturated cloudy base, with a deterministic broadband w
    seed. The geometry is RiRk in both directions because the mish points are then Gauss
    nodes on both legs, which is what makes the exact mass quadrature in the conservation
    testsets possible (same argument as the positivity testset's `cellw`).

    `extra_opts` is merged into `options` — how a caller asks for a transform, a floor, or a
    retired key it expects to be refused.
    """
    function cloudy_rirk(tmpdir, tag; extra_opts = Dict{Symbol,Any}())
        vars = Dict(v => i for (i, v) in enumerate(Scythe.MC_VARS))
        scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
        side_bc = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))
        wall_bc = merge(scalar_bc, Dict("w" => DirichletBC()))
        gp = GridParameters(geometry = "RiRk",
            iMin = 0.0, iMax = 8.0e3, num_cells_i = 4,
            kMin = 0.0, kMax = 4.0e3, num_cells_k = 8,
            BCL = side_bc, BCR = side_bc, BCB = wall_bc, BCT = wall_bc,
            vars = vars)
        ref_file = joinpath(tmpdir, "cloudy_$(tag).ref")
        opts = merge(Dict{Symbol,Any}(:semiimplicit => true,
                                      :exact_reference_state => true,
                                      :precipitation => false), extra_opts)
        model = ModelParameters(
            ts = 0.25, integration_time = 5.0, output_interval = 5.0,
            equation_set = "moist_compressible_XZ",
            ref_state_file = ref_file, grid_params = gp,
            physical_params = Dict(:Khdiff => 0.0, :Kvdiff => 0.0,
                                   :Kvdiff_heat => 0.0, :Kvdiff_water => 0.0,
                                   :tau_qss => 10.0, :alpha => 0.0,
                                   :z_damp => 20.0e3, :f => 0.0),
            options = opts)
        gp = model.grid_params
        patch = createGrid(gp)
        z = Scythe.getGridpoints(patch)[1:gp.kDim, end]
        col = saturated_cloudy_column_mc(z)
        Scythe.write_exact_ref_mc(ref_file, z, col.p_Pa, col.rho_d, col.rho_v, col.rho_c)
        patch.physical .= 0.0
        # Deterministic broadband w seed (irrational chirp) — the same one the positivity
        # testset uses. Large enough to move every prognostic and to make cloud, far too
        # small to threaten positive-definiteness.
        w_i = gp.vars["w"]
        for i in 1:size(patch.physical, 1)
            patch.physical[i, w_i, 1] = 1.0e-2 * sin(0.5 * sqrt(2.0) * i^2)
        end
        spectralTransform!(patch)
        gridTransform!(patch)
        hrm = sparse(Int64[], Int64[], Float64[],
                     size(patch.spectral, 1), size(patch.spectral, 2))
        mtile = createModelTile(patch, patch, model, hrm)
        return mtile, patch, model, gp
    end

    """Advance every column of an RiRk cloudy patch for `nsteps` steps."""
    function cloudy_run!(mtile, patch, gp, nsteps)
        kDim = gp.kDim
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


    # ──────────────────────────────────────────────
    # 8c. The PROGNOSTIC VAPOR (Stage A)
    # ──────────────────────────────────────────────
    # The vapor is a slot, not a retrieval. What that has to buy, and what it must not cost:
    #
    #   * registration — it is in the CONSTANTS, so every list built by enumerating them
    #     carries it, and a `vars` dict built from a stale name list must fail LOUDLY at
    #     tile creation rather than writing the vapor tendency into whatever is at that index;
    #   * conservation — condensation is now an equal and opposite pair of SLOT sources, and
    #     rho_t must not move at all;
    #   * the nudge — a drift correction on tau_rec that is thermodynamically inert (its own
    #     testset, up with `qss_relaxation`);
    #   * the tombstone — a configuration that still selects a retired retrieval must stop.

    @testset "rho_v is registered in the canonical slot lists" begin
        # In `MC_VARS` / `MC_VARS_CYL`, NOT in `mc_var_names`'s optional block: the vapor is
        # prognostic in every configuration of this set, and the model_tests / tc configs
        # that enumerate the constants literally have to pick it up with no edit.
        @test Scythe.MC_VARS[10] == "rho_v"
        @test Scythe.MC_VARS_CYL[11] == "rho_v"
        @test length(Scythe.MC_VARS) == 10
        @test length(Scythe.MC_VARS_CYL) == 11
        # The appended OPTIONAL slots shift out by one behind it.
        base = Dict{Symbol,Any}()
        two = Dict{Symbol,Any}(:rain_moments => 2)
        @test Scythe.mc_var_names(base) == Scythe.MC_VARS
        @test Scythe.mc_var_names(base; cyl = true) == Scythe.MC_VARS_CYL
        @test findfirst(==("n_r"), Scythe.mc_var_names(two)) == 11
        @test findfirst(==("n_r"), Scythe.mc_var_names(two; cyl = true)) == 12

        # `mc_slot` resolves it by name on both geometries.
        vars_xz = Dict(v => i for (i, v) in enumerate(Scythe.MC_VARS))
        vars_cyl = Dict(v => i for (i, v) in enumerate(Scythe.MC_VARS_CYL))
        @test Scythe.mc_slot(vars_xz, "rho_v") == 10
        @test Scythe.mc_slot(vars_cyl, "rho_v") == 11
        # ...and a STALE list has no slot for it, which must throw rather than resolve to 0.
        stale = Dict(v => i for (i, v) in
                     enumerate(["p", "rho_d", "rho_t", "u", "w", "E_t", "Q_ss",
                                "rho_r", "rho_c"]))
        @test_throws ErrorException Scythe.mc_slot(stale, "rho_v")
    end

    @testset "a stale vars dict fails at tile creation, not later" begin
        # `mc_slots` resolves rho_v with the THROWING lookup for every mc set, so a
        # configuration built from a pre-Stage-A name list stops at `createModelTile`. The
        # alternative — a 0 index, the optional-slot convention — would have the vapor
        # tendency written nowhere and the run proceed.
        mktempdir() do tmpdir
            vars = Dict(v => i for (i, v) in
                        enumerate(["p", "rho_d", "rho_t", "u", "w", "E_t", "Q_ss",
                                   "rho_r", "rho_c"]))
            scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
            wall_bc = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))
            gp = GridParameters(geometry = "RZ", num_cells = 8,
                iMin = 0.0, iMax = 2000.0, kMin = 0.0, kMax = 2000.0, kDim = 16,
                BCL = wall_bc, BCR = wall_bc, BCB = wall_bc, BCT = wall_bc, vars = vars)
            patch = createGrid(gp)
            z = Scythe.getGridpoints(patch)[1:gp.kDim, 2]
            col = dry_adiabatic_column_mc(z)
            ref_file = joinpath(tmpdir, "stale.ref")
            Scythe.write_exact_ref_mc(ref_file, z, col.p_Pa, col.rho_d, col.rho_v, col.rho_c)
            model = ModelParameters(
                ts = 0.1, integration_time = 1.0, output_interval = 1.0,
                equation_set = "moist_compressible_XZ",
                ref_state_file = ref_file, grid_params = gp,
                physical_params = Dict(:Khdiff => 0.0, :Kvdiff => 0.0),
                options = Dict{Symbol,Any}(:semiimplicit => true,
                                           :exact_reference_state => true))
            patch.physical .= 0.0
            spectralTransform!(patch)
            gridTransform!(patch)
            hrm = sparse(Int64[], Int64[], Float64[],
                         size(patch.spectral, 1), size(patch.spectral, 2))
            @test_throws ErrorException createModelTile(patch, patch, model, hrm)
        end
    end

    @testset "positivity_reference_profile serves rho_v the DERIVED profile" begin
        # The vapor is a PERTURBATION, so a constant zero bound on it would pin the field at
        # or above its reference — the same trap rho_c is protected from. Its offset is the
        # DERIVED rho_tbar - rho_dbar - rho_cbar, which is what the slot is carried against;
        # reading Springsteel's independently fitted ref_rho_v here would put a fit-level
        # disagreement into the bound.
        mktempdir() do tmpdir
            mtile, patch, model, col = make_mc_mtile(tmpdir)
            ref = mtile.ref_state
            prof = Scythe.positivity_reference_profile("rho_v", ref)
            @test prof !== nothing
            expected = Springsteel.ref_rho_t(ref)[:, 1] .-
                       Springsteel.ref_rho_d(ref)[:, 1] .-
                       Springsteel.ref_rho_c(ref)[:, 1]
            @test collect(prof) == expected                # bitwise, not approximately
            # ...and it is the SAME profile mc_reference_diagnostics built for the driver.
            @test collect(prof) == mtile.mc_ref_diag.rho_vbar
        end
    end

    @testset "options[:vapor_retrieval] is a tombstone" begin
        # The retired retrieval selector. A configuration that still sets it was tuned
        # against a representation this equation set no longer has, so it must stop rather
        # than run under a silently different one.
        mktempdir() do tmpdir
            m, p, mod, gp = cloudy_rirk(tmpdir, "tomb";
                                        extra_opts = Dict{Symbol,Any}(
                                            :vapor_retrieval => :blend))
            @test_throws ErrorException cloudy_run!(m, p, gp, 1)
        end
        mktempdir() do tmpdir
            m, p, mod, gp = cloudy_rirk(tmpdir, "tomb_res";
                                        extra_opts = Dict{Symbol,Any}(
                                            :vapor_retrieval => :residual))
            @test_throws ErrorException cloudy_run!(m, p, gp, 1)
        end
        # The blend itself is gone with the option.
        @test !isdefined(Scythe, :vapor_retrieval_blend)
        @test !isdefined(Scythe, :_vapor_blend_weight)
        @test !isdefined(Scythe, :_blend_saturate)
        @test !isdefined(Scythe, :_blend_smoothstep)
    end

    @testset "condensation moves rho_v and rho_c equal and opposite, rho_t untouched" begin
        # THE CONSERVATION STATEMENT of the prognostic vapor. Phase change used to move one
        # slot and let the residual absorb the other side; it is now a pair of slot sources,
        # and rho_t — the conservation anchor — takes NOTHING from it.
        mktempdir() do tmpdir
            mtile, patch, model, col = make_mc_mtile(tmpdir)
            kDim = model.grid_params.kDim
            vars = model.grid_params.vars
            npts = size(patch.physical, 1)
            ncols = div(npts, kDim)
            rv_i = mtile.mc_slots.rho_v
            rc_i = vars["rho_c"]

            # The T-invariant supersaturation seed — a live cloud-channel condensation with
            # no rain, no sedimentation and no wind.
            supersaturate_lower_half!(patch, mtile, model, col, 5.0e-5)
            for c in 1:ncols
                Scythe.advance_column(mtile, c, 1)
            end

            perturbed = [i for i in 1:npts if mod1(i, kDim) <= div(kDim, 2)]
            # MEASURED ON THE APPLIED INCREMENT, not on `expdot`. Condensation is one half of
            # the `Q_ss` relaxation pair, so it no longer travels through the multistep at all:
            # it is applied directly, at the step-mean rate, by `relaxation_adjustment_qss!`.
            # `var_np1 − physical` is the whole advance these two slots receive over the step —
            # the multistep part plus that increment — and nothing else touches them here (no
            # acoustic leg, no diffusion, and the positivity clamp is inert on positive water).
            # The conservation statement is the same one, integrated: what the cloud gains, the
            # vapor loses.
            d_rc = mtile.var_np1[:, rc_i] .- mtile.tile.physical[:, rc_i, 1]
            d_rv = mtile.var_np1[:, rv_i] .- mtile.tile.physical[:, rv_i, 1]
            @test all(d_rc[perturbed] .> 0.0)     # cloud is being made
            @test all(d_rv[perturbed] .< 0.0)     # ...out of the vapor
            # EQUAL AND OPPOSITE. The vapor's increment is -Qdot_bar*ts and the cloud's is
            # +Qdot_bar*ts, both built from the SAME step-mean rate, so they cancel exactly;
            # the only thing left in the sum is the reconciliation nudge, and on this
            # consistent seed that is twelve decades below the phase change itself.
            scale = maximum(abs.(d_rc))
            @test scale > 0.0
            @test maximum(abs.(d_rv .+ d_rc)) < 1.0e-8 * scale
            # The rate-level statement behind it, exact by construction: every consumer is fed
            # from one step-mean rate per channel.
            S = mtile.mc_scratch[Threads.threadid()]
            @test S.VAPOR_SRC == -((S.Qdot_bar .+ S.Qdot_r_bar) .+ S.Qdep_bar)
            # The conserved anchor does not move.
            @test maximum(abs.(mtile.expdot_n[:, vars["rho_t"]])) == 0.0
            @test maximum(abs.(mtile.expdot_n[:, vars["rho_d"]])) == 0.0
            @test maximum(abs.(mtile.expdot_n[:, vars["E_t"]])) == 0.0
        end
    end

    @testset "sedimentation leaves the reconciliation gap unchanged" begin
        # Falling rain moves rho_r AND rho_t by the SAME fitted flux divergence, and moves
        # neither the vapor nor the cloud. `res_rho_t = rho_t - rho_d - rho_liq - rho_ice`
        # is therefore invariant under it, and so is delta = res_rho_t - rho_v: a
        # sedimentation-only step must not feed the nudge at all.
        mktempdir() do tmpdir
            mtile, patch, model, col = make_mc_mtile(tmpdir; precipitation=true, q_l=0.0,
                extra_options = Dict{Symbol,Any}(:condensation => false))
            kDim = model.grid_params.kDim
            vars = model.grid_params.vars
            npts = size(patch.physical, 1)
            ncols = div(npts, kDim)
            rv_i = mtile.mc_slots.rho_v
            rr_i = vars["rho_r"]

            # A T-invariant rain bump: rho_r, rho_t and E_t move together, so the water
            # PARTITION is consistent to begin with and delta starts at the fit floor.
            seed_rain_bump!(patch, Scythe.getGridpoints(patch), col, kDim)
            for c in 1:ncols
                Scythe.advance_column(mtile, c, 1)
            end

            # The run is live: rain is falling.
            @test maximum(abs.(mtile.expdot_n[:, rr_i])) > 0.0
            # rho_t receives the same flux divergence, so it moves too...
            @test maximum(abs.(mtile.expdot_n[:, vars["rho_t"]])) > 0.0
            # ...and slot 3 and slot 8 receive IDENTICAL discrete numbers from it, which is
            # what makes res_rho_t invariant under sedimentation. (With condensation off,
            # both slots carry only -Fr_z here: nothing is moving and there is no phase
            # change, so the two expressions reduce to the same term.)
            @test mtile.expdot_n[:, vars["rho_t"]] == mtile.expdot_n[:, rr_i]
            # THE POINT: the vapor takes nothing from sedimentation. No phase change (the
            # closure is switched off), and no nudge to feed, because the falling mass
            # leaves the reconciliation gap where it was. What is left is the fit-level
            # disagreement between the separately fitted rho_t and rho_r bumps, twelve
            # decades under the rain tendency itself.
            @test maximum(abs.(mtile.expdot_n[:, rv_i])) <
                  1.0e-10 * maximum(abs.(mtile.expdot_n[:, rr_i]))
        end
    end

    @testset "the vapor census reads the slot, ice included" begin
        # `_vapor_census!` used to assemble the vapor tendency as f_3 - f_2 - f_8 - f_9,
        # which was exact for the liquid-only set and had NO ice term — so with ice
        # registered it silently attributed the deposition sink to nothing. It reads the
        # slot now, so the depletion fraction it reports is the one the integrator applied,
        # whatever the phase changes were.
        mktempdir() do tmpdir
            mtile, patch, model, col = make_mc_mtile(tmpdir;
                extra_options = Dict{Symbol,Any}(:water_budget_trace => 1))
            kDim = model.grid_params.kDim
            vars = model.grid_params.vars
            npts = size(patch.physical, 1)
            ncols = div(npts, kDim)
            rv_i = mtile.mc_slots.rho_v

            supersaturate_lower_half!(patch, mtile, model, col, 5.0e-5)
            for c in 1:ncols
                Scythe.advance_column(mtile, c, 1)
            end

            st = mtile.mc_water_stats
            # The census counted the vapor points and measured a nonzero depletion, and the
            # Euler fraction it reports is the slot's own tendency over the field.
            @test sum(view(st, Scythe.MC_DEPLETION_V, :)) > 0.0
            @test maximum(view(st, Scythe.MC_DEPLETION_V + 3, :)) > 0.0
            # Both reconciliation-chain gaps are recorded, and on a CONSISTENT seed both
            # are at rounding: the seed moves rho_v and rho_t together (so the density
            # budget still implies the vapor carried) and moves Q_ss with them (so the
            # tracker still implies the supersaturation carried). A seed that moved only
            # one of the three would show up here as a gap of its own size, which is
            # exactly what this census exists to report.
            # `dm`, the water the seed moved, is the scale both are measured against.
            dm = 5.0e-5 * maximum(Springsteel.ref_rho_d(mtile.ref_state)[:, 1])
            # The vapor gap is at ROUNDING: rho_v and rho_t took the same increment.
            @test maximum(view(st, Scythe.MC_VAPOR_GAP, :)) < 1.0e-8 * dm
            # The Q_ss gap is not zero and should not be: the seed's pressure compensation
            # moves rho_vs(T, p) a little, so Q_ss = dm and rho_v - rho_vs differ by that
            # saturation shift. It is four decades under the water moved, which is the
            # statement that the seed is consistent — a seed that moved only one of the
            # three would put its whole size here.
            @test maximum(view(st, Scythe.MC_QSS_GAP, :)) < 1.0e-3 * dm
        end
    end


    # ── Diagnostic flooring of rho_liq ──────────────────────────────────────────────────
    #
    # `options[:condensate_floor]` (reference/HANDOFF_CONDENSATE_REPRESENTATION.md,
    # Experiment 1). The spline undershoot puts rho_c a few 1e-4 BELOW zero on the flanks
    # of a cloud; the raw value then enters the closed-form temperature retrieval, where
    # dT/d rho_liq = L_v/D turns it into a cold anomaly reaching -54 K, and through
    # rho_vs(T) into false supersaturation and false nucleation.
    #
    # `:diagnostic` floors rho_liq at the DIAGNOSTIC INTERFACE only -- the retrieval, q_l
    # (hence C_vt / R_m / gamma_m), Q_s_energy, the entropy and the sedimentation energy --
    # leaving the STATE and the CONTINUITY terms raw. That is what separates it from
    # `clamp_water!`: it is memoryless, converts no mass, and cannot pump latent heat,
    # because nothing it does is written back to a prognostic slot. The measured cost of
    # NOT doing it is in reference/FINDINGS_CONDENSATE_STAGE1.md.
    #
    # Three claims below, and the third is the one that makes this not-a-clamp:
    #   (a) an ABSENT option is `:none` bitwise -- every existing configuration is unmoved;
    #   (b) where no liquid is negative the floor is INACTIVE, also bitwise, so it cannot
    #       drift a healthy run;
    #   (c) where liquid IS negative it changes the run, and the prognostic rho_c stays
    #       negative afterwards -- the state was not repaired.

    """
    A `cloudy_rirk` patch whose rho_c perturbation is seeded NEGATIVE enough to drive
    rho_liq below zero somewhere, which is what the floor exists for. `amp = 0.0` leaves the
    healthy state (the floor is then inactive and must be bitwise inert).
    """
    function condensate_floor_rirk(tmpdir, tag; floor = nothing, amp = 2.0e-3)
        mtile, patch, model, gp = cloudy_rirk(tmpdir, tag)
        floor === nothing || (model.options[:condensate_floor] = floor)
        if amp != 0.0
            rc = gp.vars["rho_c"]
            for i in axes(patch.physical, 1)
                patch.physical[i, rc, 1] = -amp * (1.0 + sin(0.5 * sqrt(3.0) * i^2))
            end
            spectralTransform!(patch)
            gridTransform!(patch)
        end
        return mtile, patch, model, gp
    end

    @testset "condensate floor: absent is bitwise :none, and an unknown value errors" begin
        have = isdefined(Scythe, :condensate_floor_mode)
        @test have                   # ← the Stage 2 gate
        # The reader/validator in isolation, before any column runs it.
        @test Scythe.condensate_floor_mode(Dict{Symbol,Any}()) == false
        @test Scythe.condensate_floor_mode(Dict{Symbol,Any}(:condensate_floor => :none)) == false
        @test Scythe.condensate_floor_mode(
            Dict{Symbol,Any}(:condensate_floor => :diagnostic)) == true
        @test_throws ErrorException Scythe.condensate_floor_mode(
            Dict{Symbol,Any}(:condensate_floor => :clamp))
        mktempdir() do tmpdir
            mA, pA, moA, gpA = condensate_floor_rirk(tmpdir, "cf_absent")
            mN, pN, moN, gpN = condensate_floor_rirk(tmpdir, "cf_none"; floor = :none)
            @test !haskey(moA.options, :condensate_floor)
            @test moN.options[:condensate_floor] === :none

            cloudy_run!(mA, pA, gpA, 10)
            cloudy_run!(mN, pN, gpN, 10)
            @test all(isfinite.(pA.physical))
            # Absent === :none, to the last bit.
            @test pA.physical == pN.physical
            @test pA.spectral == pN.spectral

            # A value that is neither must be rejected at the first column, not silently
            # taken as one of them.
            mX, pX, moX, gpX = condensate_floor_rirk(tmpdir, "cf_bad"; floor = :clamp)
            @test_throws ErrorException Scythe.advance_column(mX, 1, 1)
        end
    end

    @testset "condensate floor: inert where no liquid is negative" begin
        # The guarantee that keeps this from being a tuning knob: on a state whose liquid is
        # non-negative everywhere the floor never binds, and `:diagnostic` is then the same
        # integration as `:none` BITWISE -- not "to 1e-14".
        mktempdir() do tmpdir
            mN, pN, moN, gpN = condensate_floor_rirk(tmpdir, "cf_pos_none";
                                                     floor = :none, amp = 0.0)
            mD, pD, moD, gpD = condensate_floor_rirk(tmpdir, "cf_pos_diag";
                                                     floor = :diagnostic, amp = 0.0)
            rc, rr = gpN.vars["rho_c"], gpN.vars["rho_r"]
            rho_cbar = view(Springsteel.ref_rho_c(mN.ref_state), :, 1)
            kDim = gpN.kDim
            liq = [pN.physical[i, rc, 1] + rho_cbar[mod1(i, kDim)] + pN.physical[i, rr, 1]
                   for i in axes(pN.physical, 1)]
            @test minimum(liq) >= 0.0            # the premise: nothing to floor

            cloudy_run!(mN, pN, gpN, 10)
            cloudy_run!(mD, pD, gpD, 10)
            @test all(isfinite.(pN.physical))
            @test pN.physical == pD.physical
            @test pN.spectral == pD.spectral
        end
    end

    @testset "condensate floor: it changes the run without repairing the state" begin
        mktempdir() do tmpdir
            mN, pN, moN, gpN = condensate_floor_rirk(tmpdir, "cf_neg_none"; floor = :none)
            mD, pD, moD, gpD = condensate_floor_rirk(tmpdir, "cf_neg_diag";
                                                     floor = :diagnostic)
            rc, rr = gpN.vars["rho_c"], gpN.vars["rho_r"]
            rho_cbar = view(Springsteel.ref_rho_c(mN.ref_state), :, 1)
            kDim = gpN.kDim
            liqfun = pp -> [pp.physical[i, rc, 1] + rho_cbar[mod1(i, kDim)] +
                            pp.physical[i, rr, 1] for i in axes(pp.physical, 1)]
            @test minimum(liqfun(pN)) < 0.0      # the premise: there IS something to floor

            cloudy_run!(mN, pN, gpN, 10)
            cloudy_run!(mD, pD, gpD, 10)
            @test all(isfinite.(pN.physical))
            @test all(isfinite.(pD.physical))
            # (c1) it is a different integration ...
            @test pN.physical != pD.physical
            # (c2) ... and the STATE was not repaired: the prognostic liquid is still
            #      negative afterwards. A floor that had been written back would show a
            #      non-negative rho_liq here, and that is the clamp_water! failure mode --
            #      one-signed, cumulative, and non-finite in 23 min on O01.
            @test minimum(liqfun(pD)) < 0.0
        end
    end

    # ── The condensate control-variable transform ───────────────────────────────────────
    #
    # `options[:condensate_transform]` — Ooyama (2001) Eq. 4.19/4.20/4.23, the class-level fix
    # for the negative-condensate reservoir. Slot 9 carries n = bhyp(rho_c) and the density is
    # recovered by ahyp; nothing is ever repaired, and n is never modified. The measurement
    # that chose this family over softplus and the square-root class (both of which fit
    # equally well and detonate on first nucleation) is in
    # reference/FINDINGS_CONDENSATE_STAGE1.md §3.

    @testset "bhyp/ahyp: the algebra Ooyama's Eq. 4.19-4.23 promises" begin
        have = isdefined(Scythe, :bhyp) && isdefined(Scythe, :ahyp) &&
               isdefined(Scythe, :dbhyp) && isdefined(Scythe, :ahyp_smooth)
        @test have                   # ← the Stage 3 gate
        mu = 1.0e-7

        # The origin is exact in BOTH directions -- this is what makes a cloud-free initial
        # condition need no conversion at all, and it is not an approximation.
        @test Scythe.bhyp(0.0, mu) === 0.0
        @test Scythe.ahyp(0.0, mu) === 0.0
        @test Scythe.ahyp_smooth(0.0, mu) === 0.0

        # Round trip over ten decades spanning the knee, both inverses.
        for rho in (1.0e-12, 1.0e-9, mu, 1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2, 1.0)
            n = Scythe.bhyp(rho, mu)
            @test isapprox(Scythe.ahyp_smooth(n, mu), rho; rtol = 1.0e-14, atol = 1.0e-300)
            @test isapprox(Scythe.ahyp(n, mu), rho; rtol = 1.0e-14, atol = 1.0e-300)
            # J is the derivative of the forward map, checked against a central difference
            # in the linear regime where the step is well conditioned.
            if rho >= 1.0e-6
                h = 1.0e-4 * rho
                fd = (Scythe.bhyp(rho + h, mu) - Scythe.bhyp(rho - h, mu)) / (2h)
                @test isapprox(Scythe.dbhyp(rho, mu), fd; rtol = 1.0e-8)
            end
        end

        # THE property the whole design rests on: J is bounded in [0.5, 1] on rho >= 0, so
        # the source quotient cannot blow up at the cloud edge. softplus reaches 1e12 here
        # and the square-root class 1e6; both detonate on first nucleation.
        for rho in (0.0, 1.0e-12, mu, 1.0e-3, 1.0, 1.0e3)
            J = Scythe.dbhyp(rho, mu)
            @test 0.5 <= J <= 1.0
        end
        @test Scythe.dbhyp(0.0, mu) == 1.0          # the maximum, attained exactly at zero

        # f' = 1/J, the identity the LOUIS BL's cloud flux rests on: it recovers the
        # density gradient as dz(nu)/J, so f'(nu) had better be exactly the reciprocal of
        # the forward map's Jacobian. Bounded in [1, 2] by J in [0.5, 1] -- and that
        # boundedness is why the BL admits an exact fix where the horizontal Laplacian does
        # not (which needs f'', unbounded as 1/mu).
        for rho in (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0)
            n = Scythe.bhyp(rho, mu)
            h = 1.0e-5 * abs(n)
            fp = (Scythe.ahyp_smooth(n + h, mu) - Scythe.ahyp_smooth(n - h, mu)) / (2h)
            @test isapprox(fp, 1.0 / Scythe.dbhyp(rho, mu); rtol = 1.0e-7)
            @test 1.0 <= fp <= 2.0 + 1.0e-9
        end

        # bhyp is EXACTLY AFFINE above the knee -- algebra, not an expansion:
        #     bhyp(rho) = (rho + mu)/2 - mu^2/(2(rho + mu))
        # This is why mixing the CONTROL VARIABLE with a Laplacian is not an approximation
        # to mixing the density: a spline fit is linear, so K*grad^2(nu) is K*grad^2(rho)/2
        # to relative O((mu/rho)^2), and the recovered density rate f'*K*grad^2(nu) is
        # K*grad^2(rho). See the Khdiff_water block in src/moist_compressible.jl.
        for rho in (1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2, 1.0)
            @test isapprox(Scythe.bhyp(rho, mu),
                           (0.5 * (rho + mu)) - ((mu * mu) / (2.0 * (rho + mu)));
                           rtol = 1.0e-15)
            # The DEVIATION from exact affinity is bounded by mu^2/(rho+mu), i.e. it
            # vanishes quadratically in mu/rho -- the bound the ν-space mixing relies on.
            # The +8eps is round-off headroom, not slack in the identity: forming the
            # difference of two O(rho) quantities cancels away all but ~eps*rho, and at
            # rho >= 1e-2 that round-off is itself larger than the analytic bound.
            @test abs((2.0 * Scythe.bhyp(rho, mu)) - (rho + mu)) <=
                  ((mu * mu) / (rho + mu)) + (8.0 * eps(rho))
            @test isapprox(Scythe.dbhyp(rho, mu), 0.5; rtol = 2.0 * (mu / rho)^2 + 1.0e-15)
        end

        # Range. The published quasi-inverse is non-negative everywhere; the strict inverse
        # is bounded below by -mu, and approaches it rather than crossing it.
        for n in (-1.0, -1.0e-2, -1.0e-4, -mu, -1.0e-12, 0.0, 1.0e-12, 1.0e-3, 1.0)
            @test Scythe.ahyp(n, mu) >= 0.0
            @test Scythe.ahyp_smooth(n, mu) > -mu
        end
        @test Scythe.ahyp_smooth(-1.0e6 * mu, mu) > -mu
        # ... and the two agree exactly where Ooyama does not clip.
        for n in (1.0e-9, 1.0e-6, 1.0e-3, 1.0)
            @test Scythe.ahyp(n, mu) === Scythe.ahyp_smooth(n, mu)
        end
        # Monotonicity across the knee, which is what makes the map a change of variables.
        ns = [-1.0e-3, -1.0e-5, -mu, 0.0, mu, 1.0e-5, 1.0e-3]
        @test issorted([Scythe.ahyp_smooth(n, mu) for n in ns])

        # Linear well above the knee: n -> rho/2 (Ooyama's factor 0.5 is what removes the
        # coefficient from the inverse), so the ringing in n is HALF the ringing in rho --
        # which is why the measured excursion moves by exactly 2.15x and not more. The
        # approach is from ABOVE at relative distance mu/rho, so the tolerance has to be
        # loose where the knee is still close and can tighten as it recedes.
        @test isapprox(Scythe.bhyp(1.0e-3, mu) / 1.0e-3, 0.5; rtol = 2.0e-4)
        @test isapprox(Scythe.bhyp(1.0, mu), 0.5; rtol = 1.0e-6)
        @test isapprox(Scythe.dbhyp(1.0e-3, mu), 0.5; rtol = 1.0e-6)
        @test Scythe.bhyp(1.0e-3, mu) > 0.5e-3          # ... and from above, not below

        # condensate_slot: the initial-condition conversion, and the reason a cloud-free
        # state on a cloud-free reference needs no threading anywhere.
        @test Scythe.condensate_slot(0.0, 0.0, :none, mu) === 0.0
        @test Scythe.condensate_slot(0.0, 0.0, :bhyp, mu) === 0.0
        @test Scythe.condensate_slot(1.0e-3, 0.0, :none, mu) === 1.0e-3
        @test Scythe.condensate_slot(1.0e-3, 0.0, :bhyp, mu) === Scythe.bhyp(1.0e-3, mu)
    end

    @testset "condensate transform: option gating" begin
        @test Scythe.condensate_transform_mode(Dict{Symbol,Any}()) === :none
        @test Scythe.condensate_transform_mode(
            Dict{Symbol,Any}(:condensate_transform => :none)) === :none
        @test Scythe.condensate_transform_mode(
            Dict{Symbol,Any}(:condensate_transform => :bhyp)) === :bhyp
        @test Scythe.condensate_transform_mode(
            Dict{Symbol,Any}(:condensate_transform => :bhyp_smooth)) === :bhyp_smooth
        @test_throws ErrorException Scythe.condensate_transform_mode(
            Dict{Symbol,Any}(:condensate_transform => :ooyama))
    end

    """
    A small RiRk moist patch on a CLOUD-FREE reference (which is what every benchmark
    configuration has, and what the transform is validated on -- a cloudy reference is
    refused by `check_condensate_transform_ic` until the initializers are threaded). The
    rho_c slot is seeded with a broadband oscillation big enough that an untransformed run
    carries a clearly negative cloud density.
    """
    function ctrans_rirk(tmpdir, tag; transform = nothing, rain_transform = nothing,
                         positivity = nothing, amp = 3.0e-3, precipitation = false,
                         rain_amp = 0.0, xmod = false, Khdiff_water = 0.0, Sc_t = 1.0)
        ref_file = joinpath(tmpdir, "ctrans_$(tag).ref")
        opts = Dict{Symbol,Any}(:semiimplicit => true, :exact_reference_state => true,
                                :precipitation => precipitation)
        transform === nothing || (opts[:condensate_transform] = transform)
        rain_transform === nothing || (opts[:rain_transform] = rain_transform)
        # The slot NAMES follow the declared transforms, and every name-keyed dict has to
        # agree — `check_mc_var_names` refuses the configuration otherwise.
        varnames = Scythe.mc_var_names(opts)
        vars = Dict(v => i for (i, v) in enumerate(varnames))
        scalar_bc = Dict(v => NeumannBC() for v in varnames)
        side_bc = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))
        wall_bc = merge(scalar_bc, Dict("w" => DirichletBC()))
        gp = GridParameters(geometry = "RiRk",
            iMin = 0.0, iMax = 8.0e3, num_cells_i = 4,
            kMin = 0.0, kMax = 4.0e3, num_cells_k = 8,
            positivity = positivity === nothing ?
                Dict{String,Dict{Symbol,Float64}}() : positivity,
            BCL = side_bc, BCR = side_bc, BCB = wall_bc, BCT = wall_bc,
            vars = vars)
        model = ModelParameters(
            ts = 0.25, integration_time = 5.0, output_interval = 5.0,
            equation_set = "moist_compressible_XZ",
            ref_state_file = ref_file, grid_params = gp,
            physical_params = Dict(:Khdiff => 0.0, :Kvdiff => 0.0,
                                   :Kvdiff_heat => 0.0, :Kvdiff_water => 0.0,
                                   :tau_qss => 10.0, :alpha => 0.0,
                                   :Khdiff_water => Khdiff_water, :Sc_t => Sc_t,
                                   :z_damp => 20.0e3, :f => 0.0),
            options = opts)
        gp = model.grid_params
        patch = createGrid(gp)
        z = Scythe.getGridpoints(patch)[1:gp.kDim, end]
        col = saturated_cloudy_column_mc(z)
        # CLOUD-FREE reference: the cloud lives entirely in the perturbation, so the slot
        # convention is bhyp(rho_c) - bhyp(0) = bhyp(rho_c) and nothing has to be threaded.
        Scythe.write_exact_ref_mc(ref_file, z, col.p_Pa, col.rho_d, col.rho_v,
                                  zeros(gp.kDim))
        patch.physical .= 0.0
        w_i = gp.vars["w"]
        rc_i = Scythe.mc_slot(gp.vars, "rho_c")
        rr_i = Scythe.mc_slot(gp.vars, "rho_r")
        gpts = Scythe.getGridpoints(patch)
        tf = transform === nothing ? :none : transform
        rtf = rain_transform === nothing ? :none : rain_transform
        # A SUB-CELL cloud spike: sigma = 250 m against a 500 m cell. That is the physical
        # situation -- a convective condensate spike the column cannot resolve -- and it is
        # what makes the fit undershoot on the flanks. A smooth seed does not ring at all on
        # this coarse a column, and then the test asserts nothing.
        # `xmod` gives the seed HORIZONTAL structure. Off by default and then bitwise the
        # z-only seed this fixture has always used. It is needed only by the Khdiff_water
        # testset: `mc_w_kdiff!` on the Cartesian slice is `K*f_xx`, which is ~0 for a
        # z-only field, so without it a horizontal-mixing test asserts nothing. One full
        # cosine over the 8 km domain is well resolved on 4 cells' worth of nodes, so the
        # two arms' fits stay comparable.
        for i in 1:size(patch.physical, 1)
            patch.physical[i, w_i, 1] = 1.0e-2 * sin(0.5 * sqrt(2.0) * i^2)
            zi = gpts[i, end]
            xfac = xmod ? 1.0 + (0.5 * cos(2.0 * pi * gpts[i, 1] / 8.0e3)) : 1.0
            rho_c = xfac * amp * exp(-((zi - 2000.0) / 250.0)^2)    # >= 0 by construction
            patch.physical[i, rc_i, 1] = Scythe.condensate_slot(rho_c, 0.0, tf, 1.0e-7)
            # An equally sub-cell RAIN spike, sited lower so it is a distinct feature. Rain
            # has no reference profile, so its slot is the control variable itself.
            if rain_amp > 0.0
                rho_r = xfac * rain_amp * exp(-((zi - 1200.0) / 250.0)^2)
                patch.physical[i, rr_i, 1] = Scythe.rain_slot(rho_r, rtf, 1.0e-7)
            end
        end
        spectralTransform!(patch)
        gridTransform!(patch)
        hrm = sparse(Int64[], Int64[], Float64[],
                     size(patch.spectral, 1), size(patch.spectral, 2))
        mtile = createModelTile(patch, patch, model, hrm)
        return mtile, patch, model, gp
    end

    """Recovered cloud density from a patch's slot 9, under `transform`."""
    function ctrans_rho_c(patch, gp, transform)
        rc = Scythe.mc_slot(gp.vars, "rho_c")
        s = @view patch.physical[:, rc, 1]
        transform === :none && return collect(s)
        transform === :bhyp_smooth && return [Scythe.ahyp_smooth(x, 1.0e-7) for x in s]
        return [Scythe.ahyp(x, 1.0e-7) for x in s]
    end

    """Recovered rain density from a patch's slot 8, under `transform` (no reference)."""
    function ctrans_rho_r(patch, gp, transform)
        rr = Scythe.mc_slot(gp.vars, "rho_r")
        s = @view patch.physical[:, rr, 1]
        return [Scythe.recover_rho_r(x, transform, 1.0e-7) for x in s]
    end

    @testset "condensate transform: absent is bitwise :none" begin
        mktempdir() do tmpdir
            mA, pA, moA, gpA = ctrans_rirk(tmpdir, "ct_absent")
            mN, pN, moN, gpN = ctrans_rirk(tmpdir, "ct_none"; transform = :none)
            @test !haskey(moA.options, :condensate_transform)
            cloudy_run!(mA, pA, gpA, 10)
            cloudy_run!(mN, pN, gpN, 10)
            @test all(isfinite.(pA.physical))
            @test pA.physical == pN.physical
            @test pA.spectral == pN.spectral
        end
    end

    @testset "condensate transform: the recovered density stays non-negative" begin
        # THE GATE. The untransformed run's cloud goes clearly negative on this seed --
        # that is the premise, and without it the test asserts nothing. Under `:bhyp` the
        # recovered density is non-negative at every point and every step by construction,
        # and under `:bhyp_smooth` it is bounded below by -mu.
        mktempdir() do tmpdir
            mu = 1.0e-7
            mN, pN, moN, gpN = ctrans_rirk(tmpdir, "ct_g_none"; transform = :none)
            mB, pB, moB, gpB = ctrans_rirk(tmpdir, "ct_g_bhyp"; transform = :bhyp)
            mS, pS, moS, gpS = ctrans_rirk(tmpdir, "ct_g_smooth"; transform = :bhyp_smooth)

            cloudy_run!(mN, pN, gpN, 20)
            cloudy_run!(mB, pB, gpB, 20)
            cloudy_run!(mS, pS, gpS, 20)

            @test all(isfinite.(pN.physical))
            @test all(isfinite.(pB.physical))
            @test all(isfinite.(pS.physical))

            rN = ctrans_rho_c(pN, gpN, :none)
            rB = ctrans_rho_c(pB, gpB, :bhyp)
            rS = ctrans_rho_c(pS, gpS, :bhyp_smooth)

            @test minimum(rN) < -1.0e-6          # the premise: it really does go negative
            @test minimum(rB) >= 0.0             # exactly, by construction
            @test minimum(rS) > -mu              # bounded, and it does not reach the bound
            # There is still cloud: a transform that annihilated the field would pass the
            # bounds above and mean nothing.
            @test maximum(rB) > 1.0e-4
            @test maximum(rS) > 1.0e-4
            # The two variants are two different INTEGRATIONS, not two readings of one field:
            # they differ wherever n < 0 (by at most mu, which is all Ooyama's clip can move)
            # and that difference then advects into the cloud. So the claim is not equality --
            # the map-level identity `ahyp === ahyp_smooth` for n > 0 is asserted in the
            # kernel testset above -- but that 20 steps of divergence stays bounded by the
            # only thing that seeds it. Measured here: 5.4e-9, i.e. 0.05 mu.
            @test maximum(abs, [b - s for (b, s) in zip(rB, rS) if b > 1.0e-6]) < mu
        end
    end

    @testset "condensate transform: positivity on rho_c is refused" begin
        # Two ways of enforcing the same constraint, and a coefficient bound would constrain
        # the CONTROL variable rather than the density. That is a configuration error, not
        # something to resolve silently by precedence.
        mktempdir() do tmpdir
            @test_throws ErrorException ctrans_rirk(tmpdir, "ct_pos"; transform = :bhyp,
                positivity = Dict("rho_c" => Dict(:k => 0.0)))
            # ... while rain keeps its bound alongside the transform: that asymmetry is
            # deliberate, and rho_r is the control that shows the transform did no harm.
            m, p, mo, gp = ctrans_rirk(tmpdir, "ct_pos_r"; transform = :bhyp,
                positivity = Dict("rho_r" => Dict(:i => 0.0, :k => 0.0)))
            @test mo.grid_params.positivity["rho_r"] == Dict(:i => 0.0, :k => 0.0)
            # ... and the mirror: once RAIN is transformed too, its bound is refused for
            # exactly the same reason.
            @test_throws ErrorException ctrans_rirk(tmpdir, "rt_pos"; rain_transform = :bhyp,
                positivity = Dict("rho_r" => Dict(:k => 0.0)))
        end
    end

    @testset "water transforms: slot names and the configuration guard" begin
        # A transformed slot no longer holds what its name says, and the name is what every
        # consumer keys off -- Springsteel builds the output header from `vars`. The names
        # therefore move with the transforms.
        none = Dict{Symbol,Any}()
        both = Dict{Symbol,Any}(:condensate_transform => :bhyp, :rain_transform => :bhyp)
        cloud_only = Dict{Symbol,Any}(:condensate_transform => :bhyp_smooth)
        rain_only = Dict{Symbol,Any}(:rain_transform => :bhyp)
        @test Scythe.mc_var_names(none) == Scythe.MC_VARS
        @test Scythe.mc_var_names(none; cyl = true) == Scythe.MC_VARS_CYL
        @test Scythe.mc_var_names(both)[8:9] == ["nu_r", "nu_c"]
        @test Scythe.mc_var_names(cloud_only)[8:9] == ["rho_r", "nu_c"]
        @test Scythe.mc_var_names(rain_only)[8:9] == ["nu_r", "rho_c"]
        @test Scythe.mc_var_names(both; cyl = true)[10] == "v"
        @test Scythe.condensate_var_name(none) == "rho_c"
        @test Scythe.rain_var_name(both) == "nu_r"
        @test Scythe.rain_transform_mode(none) === :none
        @test Scythe.rain_transform_mode(rain_only) === :bhyp
        @test_throws ErrorException Scythe.rain_transform_mode(
            Dict{Symbol,Any}(:rain_transform => :ooyama))

        # `mc_slot` accepts either convention and raises rather than returning a wrong slot.
        @test Scythe.mc_slot(Dict("rho_c" => 9, "rho_r" => 8), "rho_c") == 9
        @test Scythe.mc_slot(Dict("nu_c" => 9, "nu_r" => 8), "rho_c") == 9
        @test Scythe.mc_slot(Dict("nu_c" => 9, "nu_r" => 8), "rho_r") == 8
        @test_throws ErrorException Scythe.mc_slot(Dict("p" => 1), "rho_c")

        # THE SILENT-MISS GUARD. A stale name key in one of the four BC dicts, l_q or
        # positivity is IGNORED by Springsteel's `_resolve_spline_filter` rather than raised,
        # so a leftover "rho_r" => NaturalBC() under the rain transform would quietly put slot
        # 8 back on the default Neumann fit -- which forces a zero boundary flux derivative
        # and traps falling rain at the surface. That must be a configuration error.
        mktempdir() do tmpdir
            # Sanity: a correctly-named transformed configuration builds.
            m, p, mo, gp = ctrans_rirk(tmpdir, "names_ok";
                                       transform = :bhyp, rain_transform = :bhyp)
            @test haskey(gp.vars, "nu_c") && haskey(gp.vars, "nu_r")
            @test !haskey(gp.vars, "rho_c") && !haskey(gp.vars, "rho_r")
            @test Scythe.check_mc_var_names(mo) === nothing

            # ... and one with a stale l_q key does not.
            stale = ModelParameters(
                ts = mo.ts, integration_time = mo.integration_time,
                output_interval = mo.output_interval,
                equation_set = mo.equation_set, ref_state_file = mo.ref_state_file,
                grid_params = GridParameters(geometry = "RiRk",
                    iMin = 0.0, iMax = 8.0e3, num_cells_i = 4,
                    kMin = 0.0, kMax = 4.0e3, num_cells_k = 8,
                    l_q = Dict("default" => 2.0, "rho_r" => 1.0),
                    BCL = gp.BCL, BCR = gp.BCR, BCB = gp.BCB, BCT = gp.BCT,
                    vars = gp.vars),
                physical_params = mo.physical_params, options = mo.options)
            @test_throws ErrorException Scythe.check_mc_var_names(stale)
            # With no transform declared the guard is a no-op, whatever the names are.
            @test Scythe.check_mc_var_names(ModelParameters(
                ts = mo.ts, integration_time = mo.integration_time,
                output_interval = mo.output_interval,
                equation_set = mo.equation_set, ref_state_file = mo.ref_state_file,
                grid_params = stale.grid_params,
                physical_params = mo.physical_params,
                options = Dict{Symbol,Any}())) === nothing
        end
    end

    @testset "rain transform: absent is bitwise :none" begin
        mktempdir() do tmpdir
            mA, pA, moA, gpA = ctrans_rirk(tmpdir, "rt_absent"; rain_amp = 2.0e-3,
                                           precipitation = true)
            mN, pN, moN, gpN = ctrans_rirk(tmpdir, "rt_none"; rain_transform = :none,
                                           rain_amp = 2.0e-3, precipitation = true)
            @test !haskey(moA.options, :rain_transform)
            cloudy_run!(mA, pA, gpA, 10)
            cloudy_run!(mN, pN, gpN, 10)
            @test all(isfinite.(pA.physical))
            @test pA.physical == pN.physical
            @test pA.spectral == pN.spectral
        end
    end

    @testset "rain transform: the recovered rain density stays non-negative" begin
        # The rain analogue of the condensate gate, and it exercises the one term rain has
        # that cloud does not: the sedimentation flux divergence, which reaches slot 8
        # through the Jacobian while rho_t and E_t keep receiving it untransformed.
        mktempdir() do tmpdir
            mu = 1.0e-7
            mN, pN, moN, gpN = ctrans_rirk(tmpdir, "rt_g_none"; rain_transform = :none,
                                           rain_amp = 2.0e-3, precipitation = true)
            mB, pB, moB, gpB = ctrans_rirk(tmpdir, "rt_g_bhyp"; rain_transform = :bhyp,
                                           rain_amp = 2.0e-3, precipitation = true)
            mS, pS, moS, gpS = ctrans_rirk(tmpdir, "rt_g_smooth";
                                           rain_transform = :bhyp_smooth,
                                           rain_amp = 2.0e-3, precipitation = true)
            cloudy_run!(mN, pN, gpN, 20)
            cloudy_run!(mB, pB, gpB, 20)
            cloudy_run!(mS, pS, gpS, 20)

            @test all(isfinite.(pN.physical))
            @test all(isfinite.(pB.physical))
            @test all(isfinite.(pS.physical))

            rN = ctrans_rho_r(pN, gpN, :none)
            rB = ctrans_rho_r(pB, gpB, :bhyp)
            rS = ctrans_rho_r(pS, gpS, :bhyp_smooth)

            @test minimum(rN) < -1.0e-6          # the premise: it really does go negative
            @test minimum(rB) >= 0.0             # exactly, by construction
            @test minimum(rS) > -mu
            # There is still rain: a transform that annihilated the field would pass both
            # bounds and mean nothing.
            @test maximum(rB) > 1.0e-4
            @test maximum(rS) > 1.0e-4
        end
    end

    @testset "water transforms: the refused combinations" begin
        # Each of these would run and produce plausible numbers while solving the wrong
        # equation for a transformed slot. They fail loudly instead.
        mktempdir() do tmpdir
            for (tag, kw) in (("wt_kv_c", (; transform = :bhyp)),
                              ("wt_kv_r", (; rain_transform = :bhyp)))
                m, p, mo, gp = ctrans_rirk(tmpdir, tag; kw...)
                # The implicit vertical water diffusion solves slots 8/9 as densities.
                mo.physical_params[:Kvdiff_water] = 1.0
                @test_throws ErrorException Scythe.diffusion_timestep_mc(
                    m, 1, gp.kDim, 1, Scythe.MCCartesianXZ())
                mo.physical_params[:Kvdiff_water] = 0.0
                # The per-term production budget would mix control-variable and density units.
                mo.options[:water_budget_trace] = 1
                @test_throws ErrorException cloudy_run!(m, p, gp, 1)
                delete!(mo.options, :water_budget_trace)
                # clamp_water! would be a state repair on the control variable.
                mo.options[:clamp_water] = true
                @test_throws ErrorException Scythe.clamp_water!(m, 1, gp.kDim)
                delete!(mo.options, :clamp_water)
            end
        end
    end

    @testset "moist_entropy_total survives a negative vapor" begin
        # `entropy` takes log(q_v*rho_d/rho_v0), and the vapor is a RESIDUAL, so it goes
        # negative wherever the difference of two independently fitted densities exceeds
        # the vapor present -- i.e. at the tropopause. Measured on the 3-nest TC initial
        # state: 362 of 6750 nest-1 points between 14.4 and 17.3 km, worst rho_v = -4.5e-6
        # against a reference rho_vbar of +1.5e-6 there. It threw a DomainError out of the
        # first Louis-BL call of the first timestep.
        Tk = 210.0; rho_d = 0.18
        @test isfinite(Scythe.moist_entropy_total(Tk, rho_d, -2.5e-5, 0.0))
        @test Scythe.moist_entropy_total(Tk, rho_d, -2.5e-5, 0.0) ==
              Scythe.moist_entropy_total(Tk, rho_d, 0.0, 0.0)
        # CONTINUOUS at zero, so the clamped points join the dry limit smoothly rather
        # than stepping: q*ln(q) -> 0 as q -> 0+, so entropy has no kink there.
        s0 = Scythe.moist_entropy_total(Tk, rho_d, 0.0, 0.0)
        # The gap closes like q*|ln q| -- superlinearly in q but not linearly, so the
        # tolerance has to carry the log factor rather than being a constant.
        for q in (1.0e-6, 1.0e-8, 1.0e-10, 1.0e-12)
            @test isapprox(Scythe.moist_entropy_total(Tk, rho_d, q, 0.0), s0;
                           atol = 1.0e5 * q)
        end
        # and it is monotone: smaller q is closer to the dry limit
        gaps = [abs(Scythe.moist_entropy_total(Tk, rho_d, q, 0.0) - s0)
                for q in (1.0e-6, 1.0e-8, 1.0e-10, 1.0e-12)]
        @test issorted(gaps; rev = true)
        # q_l is NOT clamped: nothing takes its log, and a negative condensate has to stay
        # visible in the entropy budget. It passes through as exactly its own term --
        # note that term is POSITIVE for a negative q_l here, because Tk < T_0 makes
        # log(Tk/T_0) negative; the claim is pass-through, not a sign.
        @test Scythe.moist_entropy_total(Tk, rho_d, 1.0e-4, -1.0e-4) -
              Scythe.moist_entropy_total(Tk, rho_d, 1.0e-4, 0.0) ≈
              -1.0e-4 * Scythe.Cl * log(Tk / Scythe.T_0)
        # Positive vapor is untouched -- the clamp is inert wherever the physics is sane.
        @test Scythe.moist_entropy_total(300.0, 1.1, 0.015, 1.0e-3) ==
              Scythe.entropy(300.0, 1.1, 0.015) +
              (1.0e-3 * Scythe.Cl * log(300.0 / Scythe.T_0))
    end

    @testset "Khdiff_water under a water transform mixes the control variable" begin
        # It used to REFUSE to run here. It no longer does, and the reason is measured in
        # the bhyp algebra testset above: bhyp is exactly affine for rho >> mu, so
        # K*grad^2(nu) IS K*grad^2(rho)/2 to relative O((mu/rho)^2), and the density rate
        # the slot implies is the density mixing the term intends. The exact chain rule was
        # rejected on f''(0) = 1/mu -- see the block comment in src/moist_compressible.jl.
        #
        # Isolated exactly by flipping Khdiff_water between two runs of the SAME tile:
        # every other term cancels in the difference.
        mktempdir() do tmpdir
            K = 5.0
            function kh_increment(tag; kw...)
                m0, p0, mo0, gp0 = ctrans_rirk(tmpdir, "$(tag)_off";
                                               xmod = true, Khdiff_water = 0.0, kw...)
                m1, p1, _, gp1 = ctrans_rirk(tmpdir, "$(tag)_on";
                                             xmod = true, Khdiff_water = K, kw...)
                for (m, p, gp) in ((m0, p0, gp0), (m1, p1, gp1))
                    ncols = div(size(p.physical, 1), gp.kDim)
                    for c in 1:ncols
                        Scythe.advance_column(m, c, 1)
                    end
                end
                return m1.expdot_n .- m0.expdot_n, m1, p1, gp1
            end

            Dn, _, _, _ = kh_increment("khw_none"; rain_amp = 2.0e-3)
            Db, mb, pb, gpb = kh_increment("khw_bhyp"; transform = :bhyp,
                                           rain_transform = :bhyp, rain_amp = 2.0e-3)

            # It runs at all — this is the gate the old error() closed.
            @test all(isfinite.(Db))
            @test maximum(abs.(Dn[:, 9])) > 0.0
            @test maximum(abs.(Dn[:, 8])) > 0.0

            # The transformed slots receive HALF the density Laplacian, which is the affine
            # identity measured through the model rather than in isolation. The residual is
            # the two arms' different spline fits of rho vs nu, not the analytic gap.
            for s in (8, 9)
                sc = maximum(abs.(Dn[:, s]))
                @test isapprox(Db[:, s], 0.5 .* Dn[:, s]; atol = 1.0e-3 * sc)
            end
            # Slots 3 (total water) and 7 (Q_ss) are untransformed and must be untouched by
            # the change of variables.
            for s in (3, 7)
                sc = maximum(abs.(Dn[:, s]))
                @test isapprox(Db[:, s], Dn[:, s]; atol = 1.0e-3 * sc + 1.0e-300)
            end

            # Positivity survives the mixing, because it is a property of the RECOVERY, not
            # of the operator. Step the transformed tile and check the recovered densities.
            m, p, _, gp = ctrans_rirk(tmpdir, "khw_run"; transform = :bhyp,
                                      rain_transform = :bhyp, rain_amp = 2.0e-3,
                                      xmod = true, Khdiff_water = K)
            cloudy_run!(m, p, gp, 20)
            @test all(isfinite.(p.physical[:, :, 1]))
            @test minimum(ctrans_rho_c(p, gp, :bhyp)) >= 0.0
            @test minimum(ctrans_rho_r(p, gp, :bhyp)) >= 0.0

            # The Smagorinsky sentinel still needs Ls > 0 — untested until now.
            mk, pk, mok, gpk = ctrans_rirk(tmpdir, "khw_smag"; Khdiff_water = -1.0)
            @test_throws ErrorException Scythe.advance_column(mk, 1, 1)
        end
    end

    @testset "condensate transform: a cloudy reference warns" begin
        # The one silent-misreading hazard: an initial condition written in DENSITY on a
        # cloudy reference is reinterpreted as a control variable, off by roughly a factor
        # of two in the linear regime and undetectable from the run. The model cannot check
        # it -- both conventions give exactly 0.0 where there is no cloud, and neither is
        # distinguishable from the other where there is -- so it names the requirement and
        # runs. It was an ERROR until 2026-07-30, which turned out to block bf02_moist,
        # whose reference is legitimately cloudy and which threads the conversion.
        mktempdir() do tmpdir
            # `:condensate_transform => :bhyp` below, so slot 9 is named nu_c.
            varnames = Scythe.mc_var_names(Dict{Symbol,Any}(:condensate_transform => :bhyp))
            vars = Dict(v => i for (i, v) in enumerate(varnames))
            scalar_bc = Dict(v => NeumannBC() for v in varnames)
            side_bc = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))
            wall_bc = merge(scalar_bc, Dict("w" => DirichletBC()))
            gp0 = GridParameters(geometry = "RiRk",
                iMin = 0.0, iMax = 8.0e3, num_cells_i = 4,
                kMin = 0.0, kMax = 4.0e3, num_cells_k = 8,
                BCL = side_bc, BCR = side_bc, BCB = wall_bc, BCT = wall_bc, vars = vars)
            ref_file = joinpath(tmpdir, "ctrans_cloudy.ref")
            model = ModelParameters(
                ts = 0.25, integration_time = 5.0, output_interval = 5.0,
                equation_set = "moist_compressible_XZ",
                ref_state_file = ref_file, grid_params = gp0,
                physical_params = Dict(:Khdiff => 0.0, :Kvdiff => 0.0,
                                       :Kvdiff_heat => 0.0, :Kvdiff_water => 0.0,
                                       :tau_qss => 10.0, :alpha => 0.0,
                                       :z_damp => 20.0e3, :f => 0.0),
                options = Dict{Symbol,Any}(:semiimplicit => true,
                                           :exact_reference_state => true,
                                           :precipitation => false,
                                           :condensate_transform => :bhyp))
            gp = model.grid_params
            patch = createGrid(gp)
            z = Scythe.getGridpoints(patch)[1:gp.kDim, end]
            col = saturated_cloudy_column_mc(z)
            @test maximum(col.rho_c) > 0.0        # the premise: the reference IS cloudy
            Scythe.write_exact_ref_mc(ref_file, z, col.p_Pa, col.rho_d, col.rho_v, col.rho_c)
            patch.physical .= 0.0
            spectralTransform!(patch); gridTransform!(patch)
            hrm = sparse(Int64[], Int64[], Float64[],
                         size(patch.spectral, 1), size(patch.spectral, 2))
            mt = @test_logs (:warn, r"reference state is CLOUDY") match_mode=:any createModelTile(patch, patch, model, hrm)
            @test mt isa Scythe.ModelTile
            # ... and a CLOUD-FREE reference must stay silent, or the warning is noise.
            ref2 = joinpath(tmpdir, "ctrans_clear.ref")
            Scythe.write_exact_ref_mc(ref2, z, col.p_Pa, col.rho_d, col.rho_v,
                                      zeros(gp.kDim))
            m2 = ModelParameters(
                ts = 0.25, integration_time = 5.0, output_interval = 5.0,
                equation_set = "moist_compressible_XZ",
                ref_state_file = ref2, grid_params = gp0,
                physical_params = model.physical_params, options = model.options)
            p2 = createGrid(m2.grid_params)
            p2.physical .= 0.0
            spectralTransform!(p2); gridTransform!(p2)
            @test Scythe.check_condensate_transform_ic(
                Springsteel.exact_pressure_reference_state(
                    ref2, z, Scythe.reference_column(p2, m2.grid_params)), m2) === nothing
        end
    end

    @testset "balanced_vortex_native!: the vortex is a discrete steady state" begin
        # THE GATE of reference/HANDOFF_INITIALIZATION.md, and the direct analogue of the
        # resting-fixed-point gate above: a balanced vortex is a steady state of the
        # equation set, so at t = 0 the u tendency and the reconstructed w tendency
        # should vanish. They cannot vanish exactly -- see below -- but they must be
        # decisively better than the analytic construction they replace.
        #
        # WHY A RATIO AND NOT AN ABSOLUTE TOLERANCE. The achievable residual is set
        # by how well the spline basis can hold the balanced state on THIS grid, so
        # any absolute number would be a property of the test's own resolution and
        # would have to be re-tuned for every grid. Both inits are built on the same
        # patch from the same reference here, so the ratio isolates the thing under
        # test. (On the shipped TC nest the same comparison gives 5.1x on the
        # gradient-wind leg and 4.4x on the hydrostatic one.)
        #
        # WHY THE COLUMN IS MOIST. With rho_v == 0 the vapor field is degenerate --
        # rho_v = rho_t - rho_d is then a difference of two nearly equal fitted
        # fields, so ANY fit error is infinitely large in relative terms and the
        # Q_ss/pressure tendencies stop measuring anything. A subsaturated moist
        # column is both the realistic case and the discriminating one.
        vortex = (; v_m = 15.0, r_m = 40.0e3, r_0 = 200.0e3, v_top = 10.0e3,
                    fcor = 5.0e-5)

        function moist_adiabatic_column(z; theta0 = 300.0)
            exner = @. 1.0 - (gravity * z) / (Cpd * theta0)
            Tk = theta0 .* exner
            p_Pa = @. 100000.0 * exner^(Cpd / Rd)
            rho_d = p_Pa ./ (Rd .* Tk)
            q_v = @. 0.012 * exp(-z / 3000.0)
            return (; z, Tk, p_Pa, rho_d, rho_v = rho_d .* q_v, rho_c = zeros(length(z)))
        end

        function make_vortex_patch(tmpdir, tag)
            vars = Dict(v => i for (i, v) in enumerate(Scythe.MC_VARS_CYL))
            scalar = Dict(v => NeumannBC() for v in keys(vars))
            axis = merge(scalar, Dict("u" => DirichletBC(), "v" => DirichletBC()))
            wall = merge(scalar, Dict("u" => DirichletBC()))
            vert = Dict{String,Any}(v => SecondDerivativeBC() for v in keys(vars))
            bot = merge(vert, Dict{String,Any}("w" => DirichletBC(),
                                               "rho_r" => NaturalBC()))
            top = merge(vert, Dict{String,Any}("w" => DirichletBC()))
            gp0 = GridParameters(geometry = "RiRk", num_cells_i = 12,
                                 iMin = 0.0, iMax = 300.0e3,
                                 kMin = 0.0, kMax = 10.0e3, num_cells_k = 20,
                                 BCL = axis, BCR = wall, BCB = bot, BCT = top,
                                 vars = vars)
            patch = createGrid(gp0)
            gp = patch.params        # createGrid is what reconciles iDim/kDim/num_cells
            z = Scythe.getGridpoints(patch)[1:gp.kDim, 2]
            col = moist_adiabatic_column(z)
            ref_file = joinpath(tmpdir, "vortex_$(tag).ref")
            Scythe.write_exact_ref_mc(ref_file, z, col.p_Pa, col.rho_d, col.rho_v,
                                      col.rho_c)
            model = ModelParameters(
                ts = 0.5, integration_time = 1.0, output_interval = 1.0,
                equation_set = "moist_compressible_axisym",
                ref_state_file = ref_file, grid_params = gp,
                physical_params = Dict(:Khdiff => 0.0, :Kvdiff => 0.0,
                                       :Kvdiff_heat => 0.0, :Kvdiff_water => 0.0,
                                       :Kv_mudiff => 0.0, :tau_qss => 10.0,
                                       :N_r => 1.0e-3, :alpha => 0.0,
                                       :z_damp => 20.0e3, :f => vortex.fcor),
                options = Dict{Symbol,Any}(:semiimplicit => true,
                                           :exact_reference_state => true,
                                           :state_dependent_si => true,
                                           :precipitation => false,
                                           :vertical_mixing => false))
            ref = Springsteel.exact_pressure_reference_state(
                      ref_file, z, Scythe.reference_column(patch, gp))
            return gp, patch, model, ref, z
        end

        "The two balance residuals the equation set actually computes, plus every
        other prognostic slot's tendency, after the SINGLE fit the model performs."
        function vortex_residuals(gp, patch, model, ref)
            spectralTransform!(patch)          # == load_initial_conditions!: ONE fit.
            gridTransform!(patch)              # A second one would measure F∘F.
            hrm = sparse(Int64[], Int64[], Float64[],
                         size(patch.spectral, 1), size(patch.spectral, 2))
            mtile = createModelTile(patch, patch, model, hrm)
            kDim = gp.kDim
            ncol = Scythe.num_columns(patch)
            npts = ncol * kDim
            # The EFFECTIVE slot-1 tendency, accumulated per column. The condensation and
            # deposition heating is no longer a term in `expdot[.,1]`: it is withheld with the
            # rest of the relaxation pair and applied as the direct increment `etd_d1` over the
            # step (`relaxation_adjustment_qss!`). The spurious-condensation channel this
            # residual exists to see therefore lives in `expdot + etd_d1/ts`, and the scratch
            # column has to be read inside the loop because it is overwritten per column.
            p_eff = 0.0
            for c in 1:ncol
                cs = ((c - 1) * kDim) + 1
                Scythe.physical_model(mtile, cs, cs + kDim - 1, 1)
                Ssc = mtile.mc_scratch[Threads.threadid()]
                for k in 1:kDim
                    p_eff = max(p_eff, abs(mtile.expdot_n[cs + k - 1, gp.vars["p"]] +
                                           (Ssc.etd_d1[k] / model.ts)))
                end
            end
            rho_tbar = Springsteel.ref_rho_t(ref)[:, 1]
            p_i = gp.vars["p"]; rt_i = gp.vars["rho_t"]
            # Slot 4 IS the gradient-wind residual at rest: ADV = KDIFF = 0, so
            # expdot[u] = -pp_x/rho_t + (f + v/r) v.
            gw = maximum(abs, view(mtile.expdot_n, 1:npts, gp.vars["u"]))
            # The w tendency has to be reconstructed: expdot slot 5 carries only the
            # buoyancy half, the -pp_z/rho_t leg lives in the implicit acoustic solve.
            hy = 0.0
            i = 1
            for _ in 1:ncol, k in 1:kDim
                rtp = patch.physical[i, rt_i, 1]
                hy = max(hy, abs(patch.physical[i, p_i, 4] + (gravity * rtp)) /
                             (rtp + rho_tbar[k]))
                i += 1
            end
            slots = [maximum(abs, view(mtile.expdot_n, 1:npts, s))
                     for s in 1:size(mtile.expdot_n, 2)]
            slots[gp.vars["p"]] = p_eff
            return gw, hy, slots
        end

        mktempdir() do tmpdir
            # A single patch, so the nest topology is trivial: no interfaces, hence
            # no R3X payloads to inherit. (The nested path is exercised by
            # model_tests/tc_discrete_balance.jl, which needs a real 3-patch nest.)
            topo = Scythe.NestTopology(Scythe.NestInterface[], [Int[]], [Int[]],
                                       [1], [0.5])

            gpN, pN, mN, refN, _ = make_vortex_patch(tmpdir, "native")
            Scythe.balanced_vortex_native!([pN], topo, refN; zcol = 2,
                                           fcor = vortex.fcor, vortex_profile = :re87,
                                           v_m = vortex.v_m, r_m = vortex.r_m,
                                           r_0 = vortex.r_0, v_top = vortex.v_top,
                                           verbose = false)
            gwN, hyN, slotsN = vortex_residuals(gpN, pN, mN, refN)

            gpL, pL, mL, refL, zL = make_vortex_patch(tmpdir, "analytic")
            r_axis = collect(0.0:1000.0:300.0e3)
            flds = Scythe.balanced_vortex_fields(
                       r_axis, zL, Springsteel.ref_pressure(refL)[:, 1],
                       Springsteel.ref_rho_d(refL)[:, 1],
                       Springsteel.ref_rho_v(refL)[:, 1];
                       vortex_profile = :re87, v_m = vortex.v_m, r_m = vortex.r_m,
                       r_0 = vortex.r_0, v_top = vortex.v_top, fcor = vortex.fcor)
            pL.physical .= 0.0
            Scythe.balanced_vortex_mc!(pL, Scythe.getGridpoints(pL), refL, flds,
                                       r_axis; zcol = 2)
            gwL, hyL, slotsL = vortex_residuals(gpL, pL, mL, refL)

            # Both balance legs, decisively better than the analytic construction.
            @test gwN < gwL / 1.4
            @test hyN < hyL / 3.0
            # The thermodynamic slots must not be traded away for the balance: the
            # p and Q_ss tendencies at rest are the spurious-condensation channel,
            # and an init whose Q_ss is inconsistent with the fitted rho_d/rho_t
            # lights them up (that inconsistency cost a factor of 4 here before the
            # targets were chained off settled FITTED values).
            @test slotsN[gpN.vars["p"]] < slotsL[gpL.vars["p"]]
            @test slotsN[gpN.vars["Q_ss"]] < slotsL[gpL.vars["Q_ss"]]
            # Slots with no term that survives u = w = 0 must be at round-off.
            for name in ("rho_d", "rho_t", "rho_r")
                @test slotsN[gpN.vars[name]] == 0.0
            end
            @test slotsN[gpN.vars["E_t"]] < 1.0e-9
            @test slotsN[gpN.vars["v"]] < 1.0e-6
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
            # (signal > 1e-8); and the positivity floor converts the fitted bump's
            # negative edges to vapor, which moves rho_r but deliberately not rho_t.
            @test maximum(abs.(d3 .- d8)) < 1.0e-10
            # Cloud is untouched: rain diffusion must not manufacture condensate
            @test maximum(abs.(m_on.var_np1[:, 9] .- m_off.var_np1[:, 9])) < 1.0e-18
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

    @testset "water diffusion: vapor bump manufactures no cloud" begin
        # Subsaturated vapor bump in dry air, ONLY Kvdiff_water active. Four species are
        # solved now — rho_w', rho_v', rho_c' and rho_r — and the vapor increment is SOLVED
        # rather than implied. With no cloud and no rain to diffuse, delta_rho_c and
        # delta_rho_r are identically zero, so the whole total-water increment must be
        # vapor: cloud cannot appear. rho_w and rho_v see the same profile through the same
        # Neumann operator here, so the two increments agree and no reconciliation gap
        # opens — which is what makes `d3 ≈ d7` still the right assertion.
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
                    # ...and the VAPOR SLOT: `drho_t = drho_v = seed` is what makes this a
                    # vapor bump rather than a reconciliation gap.
                    patch.physical[i, 10, 1] += seed
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
            # THE assertion: the cloud slot does not move at all. Where there is no
            # condensate to diffuse there is no condensate increment -- not "small",
            # exactly zero -- so vapor diffusion cannot manufacture cloud. Under the
            # old residual partition this had to be inferred from two large increments
            # cancelling; now it is read straight off the prognostic slot.
            @test m_on.var_np1[:, 9] == m_off.var_np1[:, 9]
            # ... hence the whole total-water increment is vapor. Tolerance: the
            # always-on SI solve slaves an acoustic increment onto rho_t' but not onto
            # Q_ss, and Q_ss also carries its (inert) reconciliation term.
            @test maximum(abs.(d3 .- d7)) < 5.0e-9
            # The VAPOR SLOT took that increment directly, and it is the same one rho_w
            # took: same profile, same Neumann operator, same data.
            d10 = m_on.var_np1[:, 10] .- m_off.var_np1[:, 10]
            @test maximum(abs.(d10)) > 1.0e-8           # the vapor leg ran
            @test maximum(abs.(d3 .- d10)) < 5.0e-9
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
                          6 => 2.0e8, 7 => 1.0e-2, 8 => 1.0, 9 => 1.0, 10 => 1.0,
                          11 => 1.0)
            for v in 1:11               # ...including v (10) and the vapor (11)
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
            for v in 1:9
                exz = mt_xz.expdot_n[:, v]
                eax = mt_ax.expdot_n[:, v]
                scale = max(maximum(abs.(exz)), 1.0e-12)
                @test maximum(abs.(eax .- exz)) / scale < 1.0e-5
            end
            # v (slot 10 since rho_c was appended at 9) is untouched by the passive
            # dynamics
            @test maximum(abs.(mt_ax.expdot_n[:, 10])) == 0.0
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
            # Nominal per-slot magnitudes, kept as a FLOOR so a slot whose tendency is
            # genuinely tiny is still held to an absolute standard rather than to a
            # meaningless ratio of two near-zeros.
            # Slot 11 is the prognostic vapor on both cylinders (RLR and axisym share
            # MC_VARS_CYL), so the whole list lines up index for index.
            scales = Dict(1 => 1.0e2, 2 => 1.0e-5, 3 => 1.0e-5, 4 => 1.0e-2,
                          5 => 1.0e-2, 6 => 1.0e3, 7 => 1.0e-6, 8 => 1.0e-6,
                          9 => 1.0e-6, 10 => 1.0e-2, 11 => 1.0e-6)
            # ...but the DENOMINATOR is the larger of that floor and the tendency actually
            # produced, so this is a RELATIVE agreement test. It has to be: the seed puts
            # rho_c' at 0.5 kg/m^3, three decades above anything physical here, and the
            # closure's evaporation branch then runs at a drive nothing about the seeded
            # Q_ss scale predicts. Judging the resulting tendency against a hardcoded 1e-6
            # scale measures nothing about whether the two geometries agree — and they agree
            # here to roundoff RELATIVE. Keeping the fixed scale would have made this gate a
            # hostage to the magnitude of whatever state the seed happens to produce.
            per_slot = zeros(11)
            denom = Dict(v => max(scales[v], maximum(abs.(mt_rlr.expdot_n[:, v])),
                                  maximum(abs.(mt_ax.expdot_n[:, v]))) for v in 1:11)
            for c in 1:ncols_rlr
                i0 = (c - 1) * kDim
                r_c = gp1[i0 + 1, 1]
                a = findfirst(x -> x == r_c, r_ax)
                @test a !== nothing
                j0 = (a - 1) * kDim
                for v in 1:11
                    d = maximum(abs.(mt_rlr.expdot_n[i0+1:i0+kDim, v] .-
                                     mt_ax.expdot_n[j0+1:j0+kDim, v])) / denom[v]
                    per_slot[v] = max(per_slot[v], d)
                    worst = max(worst, d)
                end
            end
            @info "RLR WN0 vs axisym: worst scaled tendency mismatch = $worst" *
                  "\n  per slot: " * join(("$v=$(per_slot[v])" for v in 1:11), " ")
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
                          5 => 1.0e-2, 6 => 1.0e3, 7 => 1.0e-6, 8 => 1.0e-6,
                          9 => 1.0e-6)
            worst_v = 0.0
            for c in 1:ncols_rrr
                i0 = (c - 1) * kDim
                x_c = gp3[i0 + 1, 1]
                a = findfirst(==(x_c), x_xz)
                @test a !== nothing
                j0 = (a - 1) * kDim
                for v in 1:9
                    d = maximum(abs.(mt_rrr.expdot_n[i0+1:i0+kDim, v] .-
                                     mt_xz.expdot_n[j0+1:j0+kDim, v])) / scales[v]
                    worst = max(worst, d)
                end
                # v feels exactly -f u (advection/PGF/diffusion of a zero field
                # vanish; pp_y of a y-invariant field is spline roundoff)
                worst_v = max(worst_v,
                    maximum(abs.(mt_rrr.expdot_n[i0+1:i0+kDim, 10] .+
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
                          5 => 1.0e-2, 6 => 1.0e3, 7 => 1.0e-6, 8 => 1.0e-6,
                          9 => 1.0e-6)
            for c in 1:ncols_slr
                i0 = (c - 1) * kDim
                x_c = a_sphere * (gp1[i0 + 1, 1] - th0)
                d, a_idx = findmin(abs.(x_xz .- x_c))
                @test d < 1.0e-3            # affine mish correspondence (roundoff)
                j0 = (a_idx - 1) * kDim
                for v in 1:9
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

    # ──────────────────────────────────────────────────────────────────────────
    # Two-moment rain: slot registration and the column integration
    # ──────────────────────────────────────────────────────────────────────────
    #
    # `options[:rain_moments] = 2` is the FIRST option that adds a prognostic slot to this
    # equation set. Slots 1-9 (+v) are hardcoded literals in the kernel, so the new one is
    # APPENDED and its index is geometry-dependent: 10 on the XZ slice, 11 wherever v is
    # carried. The registration pattern the ice categories will reuse is what these test.

    @testset "rain_moments: slot registration" begin
        base = Dict{Symbol,Any}()
        two = Dict{Symbol,Any}(:rain_moments => 2)

        # Default: the option's very existence changes no name list anywhere.
        @test Scythe.rain_moments(base) == 1
        @test Scythe.mc_var_names(base) == Scythe.MC_VARS
        @test Scythe.mc_var_names(base; cyl = true) == Scythe.MC_VARS_CYL
        @test Scythe.rain_number_var_name(base) == "n_r"
        @test_throws ErrorException Scythe.rain_moments(Dict{Symbol,Any}(:rain_moments => 3))

        # Two moments: APPENDED, never inserted. The 1-9 (+v) prefix is untouched.
        names_xz = Scythe.mc_var_names(two)
        names_cyl = Scythe.mc_var_names(two; cyl = true)
        @test names_xz == vcat(Scythe.MC_VARS, "n_r")
        @test names_cyl == vcat(Scythe.MC_VARS_CYL, "n_r")
        @test names_xz[1:10] == Scythe.MC_VARS
        @test names_cyl[1:11] == Scythe.MC_VARS_CYL
        # The index the driver must resolve by NAME, precisely because it moves
        @test findfirst(==("n_r"), names_xz) == 11
        @test findfirst(==("n_r"), names_cyl) == 12

        # The number's own control-variable transform renames the slot, like the others
        tr = Dict{Symbol,Any}(:rain_moments => 2, :rain_number_transform => :bhyp)
        @test Scythe.rain_number_var_name(tr) == "nu_nr"
        @test Scythe.mc_var_names(tr)[11] == "nu_nr"
        @test Scythe.MC_NU_ALIAS["n_r"] == "nu_nr"
        @test_throws ErrorException Scythe.rain_number_transform_mode(
            Dict{Symbol,Any}(:rain_number_transform => :bogus))

        # mc_slot resolves by ROLE through either name; mc_optional_slot answers 0 for
        # a configuration that never registered it.
        vars_xz = Dict(v => i for (i, v) in enumerate(names_xz))
        vars_tr = Dict(v => i for (i, v) in enumerate(Scythe.mc_var_names(tr)))
        vars_1m = Dict(v => i for (i, v) in enumerate(Scythe.MC_VARS))
        @test Scythe.mc_slot(vars_xz, "n_r") == 11
        @test Scythe.mc_slot(vars_tr, "n_r") == 11       # via the "nu_nr" alias
        @test Scythe.mc_optional_slot(vars_xz, "n_r") == 11
        @test Scythe.mc_optional_slot(vars_tr, "n_r") == 11
        @test Scythe.mc_optional_slot(vars_1m, "n_r") == 0
        @test_throws ErrorException Scythe.mc_slot(vars_1m, "n_r")

        # n_r is a TOTAL: no reference profile to offset a positivity bound against, like
        # rho_r and unlike rho_c. The transformed name is refused, like nu_c/nu_r.
        @test Scythe.positivity_reference_profile("n_r", nothing) === nothing
        @test_throws ErrorException Scythe.positivity_reference_profile("nu_nr", nothing)

        # The total-form transform pair is the SHARED helper, not a third copy
        @test Scythe.rain_number_slot(0.0, :none, 1.0) === 0.0
        @test Scythe.rain_number_slot(0.0, :bhyp, 1.0) === 0.0     # bhyp(0) == 0 exactly
        for mode in (:none, :bhyp, :bhyp_smooth)
            @test Scythe.rain_number_slot(1.0e3, mode, 1.0) ==
                  Scythe.total_slot(1.0e3, mode, 1.0)
            @test Scythe.rain_slot(1.0e-3, mode, 1.0e-7) ==
                  Scythe.total_slot(1.0e-3, mode, 1.0e-7)
            n = Scythe.rain_number_slot(1.0e3, mode, 1.0)
            @test Scythe.recover_n_r(n, mode, 1.0) ≈ 1.0e3 rtol=1e-12
        end
    end

    @testset "rain_moments: a tile caches the appended slot by name" begin
        mktempdir() do tmpdir
            m1, _, _, _ = make_mc_mtile(tmpdir)
            @test m1.mc_slots.n_r == 0                      # absent, and says so

            m2, _, mod2, _ = make_mc_mtile(tmpdir; precipitation = true,
                extra_options = Dict{Symbol,Any}(:rain_moments => 2))
            # 11, not 10: the prognostic vapor is the unconditional appended slot at 10.
            @test m2.mc_slots.rho_v == 10
            @test m2.mc_slots.n_r == 11
            @test length(mod2.grid_params.vars) == 11
            @test mod2.grid_params.vars["n_r"] == 11
            # Concrete field: the resolution must not cost the ModelTile its type stability
            @test isconcretetype(fieldtype(typeof(m2), :mc_slots))
            # And the per-variable scratch columns grew with the slot, so the number flux
            # has a column of its own to be fitted on.
            @test size(m2.scratch_columns, 2) == 11

            # Cylindrical: the same slot, one index further out, resolved by name
            m3, _, mod3, _ = make_mc_mtile(tmpdir; precipitation = true, iMin = 100.0,
                iMax = 2100.0, equation_set = "moist_compressible_axisym",
                extra_options = Dict{Symbol,Any}(:rain_moments => 2))
            @test m3.mc_slots.rho_v == 11
            @test m3.mc_slots.n_r == 12
            @test mod3.grid_params.vars["v"] == 10
        end
    end

    @testset "rain_moments: check_mc_var_names and the deferred-support guards" begin
        mktempdir() do tmpdir
            two = Dict{Symbol,Any}(:rain_moments => 2)

            # A `vars` built from a STALE name list has no slot for the number tendency.
            # `check_mc_var_names` says so; without it the failure is a KeyError in
            # createModelTile with nothing to explain it.
            m, _, mod, _ = make_mc_mtile(tmpdir; precipitation = true, extra_options = two)
            gp_bad = deepcopy(mod.grid_params)
            delete!(gp_bad.vars, "n_r")
            bad = ModelParameters(ts = mod.ts, equation_set = mod.equation_set,
                                  ref_state_file = mod.ref_state_file,
                                  grid_params = gp_bad,
                                  physical_params = mod.physical_params,
                                  options = mod.options)
            @test_throws ErrorException Scythe.check_mc_var_names(bad)
            @test Scythe.check_mc_var_names(mod) === nothing

            kDim = mod.grid_params.kDim
            # Each of these would run and produce plausible numbers while moving rain MASS
            # without its NUMBER (or reporting a budget with no row for it). They refuse.
            mod.options[:clamp_water] = true
            @test_throws ErrorException Scythe.clamp_water!(m, 1, kDim)
            delete!(mod.options, :clamp_water)
            # ... but the MEASUREMENT still runs: it reads rho_c/rho_r only.
            @test Scythe.clamp_water!(m, 1, kDim) === nothing

            mod.options[:water_budget_trace] = 1
            @test_throws ErrorException Scythe.advance_column(m, 1, 1)
            delete!(mod.options, :water_budget_trace)

            mod.physical_params[:Kvdiff_water] = 1.0
            @test_throws ErrorException Scythe.advance_column(m, 1, 1)
            mod.physical_params[:Kvdiff_water] = 0.0
            @test Scythe.advance_column(m, 1, 1) === nothing
        end
    end

    @testset "rain_moments = 2: rain and its number form together" begin
        # A cloudy column with no rain: KK2000 autoconversion is the only active process
        # (accretion and self-collection need rain, sedimentation needs a fall speed), and
        # it is the ONLY source of rain number in the scheme.
        mktempdir() do tmpdir
            args = (; q_l = 3.0e-3, kDim = 16, num_cells = 8)
            two = Dict{Symbol,Any}(:rain_moments => 2)
            function run_once(precip, opts)
                m, _, mod, _ = make_mc_mtile(tmpdir; args..., precipitation = precip,
                                             extra_options = opts)
                ncols = div(size(m.tile.physical, 1), mod.grid_params.kDim)
                for c in 1:ncols; Scythe.advance_column(m, c, 1); end
                return m
            end
            m1 = run_once(true, Dict{Symbol,Any}())     # single moment, Ooyama
            m2 = run_once(true, two)                    # two moments, KK2000
            m2_off = run_once(false, two)               # same slots, no warm rain at all
            # BY NAME: the number slot is APPENDED after the unconditional vapor slot, so
            # its index is 11 here and there is no literal that stays right.
            nr = m2.mc_slots.n_r

            @test all(isfinite.(m2.var_np1))
            @test all(isfinite.(m2.expdot_n))
            # Rain mass appears, and so does rain NUMBER, everywhere the cloud is
            @test all(m2.var_np1[:, 8] .> 0.0)
            @test all(m2.var_np1[:, nr] .> 0.0)
            @test all(m2_off.var_np1[:, 8] .== 0.0)
            @test all(m2_off.var_np1[:, nr] .== 0.0)

            # Isolate the conversion by differencing against the precipitation-off arm, so
            # the (unrelated) condensation source on slot 9 cancels out.
            drr = m2.var_np1[:, 8] .- m2_off.var_np1[:, 8]
            drc = m2.var_np1[:, 9] .- m2_off.var_np1[:, 9]
            dnr = m2.var_np1[:, nr] .- m2_off.var_np1[:, nr]
            # Cloud pays for the rain gram for gram (the equal-and-opposite AUTO_COLL pair)
            @test all(drr .> 0.0)
            @test all(drc .< 0.0)
            @test maximum(abs.(drc .+ drr)) < 1.0e-18
            # The number produced is the mass produced divided by a 25 micron drop
            @test dnr[1] ≈ drr[1] / Scythe.RAIN_2M_M_AUTO rtol=1e-10
            # Number is not mass and not energy: rho_t and E_t receive nothing from it,
            # and autoconversion stays thermodynamically inert on both arms.
            for slot in 1:7
                @test m2.var_np1[:, slot] == m1.var_np1[:, slot]
                @test m2.var_np1[:, slot] == m2_off.var_np1[:, slot]
            end
            # The closure really did change: KK2000 is not Ooyama's threshold form
            @test m2.var_np1[1, 8] != m1.var_np1[1, 8]

            # The single-moment arm has no number slot at all (10 = the fixed nine plus
            # the unconditional vapor)
            @test size(m1.var_np1, 2) == 10
        end
    end

    @testset "rain_moments = 2: mass sediments faster than number (size sorting)" begin
        # THE point of the second moment. vtrm/vtrn = Γ(4+BR)/6/Γ(1+BR) = 1.88 for BR = 0.5,
        # so a falling shaft leaves its drop COUNT behind and the mean drop size sorts with
        # height. The single-moment closure cannot represent this at all: it has one fall
        # speed and everything falls at it.
        mktempdir() do tmpdir
            kDim = 48
            m, patch, mod, col = make_mc_mtile(tmpdir; q_l = 0.0, kDim = kDim,
                num_cells = 8, ts = 0.05, precipitation = true,
                extra_options = Dict{Symbol,Any}(:rain_moments => 2))
            gpts = Scythe.getGridpoints(patch)
            nr = m.mc_slots.n_r                          # BY NAME: 11, after the vapor slot
            seed_rain_bump!(patch, gpts, col, kDim; rho_r0 = 1.0e-3, zc = 1200.0, zr = 300.0)
            # Rain number PROPORTIONAL to the rain mass, so the two profiles start with the
            # same shape and the same centroid: any later difference is the sorting.
            for i in 1:size(patch.physical, 1)
                patch.physical[i, nr, 1] = 1.0e6 * patch.physical[i, 8, 1]
            end
            spectralTransform!(patch)
            gridTransform!(patch)

            z = gpts[:, 2]
            centroid(f) = sum(max.(f, 0.0) .* z) / sum(max.(f, 0.0))
            z_mass_0 = centroid(patch.physical[:, 8, 1])
            z_num_0 = centroid(patch.physical[:, nr, 1])
            @test z_mass_0 ≈ z_num_0 rtol=1e-10          # same shape to begin with

            # Fall speeds at the seeded state: |vtrm| > |vtrn|, which is the mechanism
            w_m, w_n = Scythe.rain_fall_speeds_2m(1.0e-3, 1.0e3, col.rho_d[1])
            @test abs(w_m) > abs(w_n) > 0.0

            step_mc!(m, patch, mod, 120)                  # 6 s of sedimentation

            @test all(isfinite.(patch.physical))
            z_mass_1 = centroid(patch.physical[:, 8, 1])
            z_num_1 = centroid(patch.physical[:, nr, 1])
            # Both fall...
            @test z_mass_1 < z_mass_0
            @test z_num_1 < z_num_0
            # ...and the mass falls FURTHER. That is size sorting, visible in the state.
            @test z_mass_1 < z_num_1
            @test (z_num_1 - z_mass_1) > 1.0
        end
    end

    # ──────────────────────────────────────────────────────────────────────────
    # Ice microphysics: slot registration, thermodynamics, transport
    # ──────────────────────────────────────────────────────────────────────────
    #
    # `options[:ice_microphysics] = :ishmael` appends TWELVE slots after `n_r` — three
    # species x (mass, number, and the two spheroid volume moments). At this stage every
    # process rate and every fall speed is zero, so what is under test is the state's
    # EXISTENCE, its TRANSPORT, and the ice thermodynamics reducing exactly to the liquid
    # one at zero ice.

    @testset "ice_microphysics: slot registration" begin
        base = Dict{Symbol,Any}()
        two = Dict{Symbol,Any}(:rain_moments => 2)
        ice = Dict{Symbol,Any}(:rain_moments => 2, :ice_microphysics => :ishmael)

        # Default: the option's existence changes no name list anywhere.
        @test Scythe.ice_microphysics(base) === :none
        @test Scythe.ice_microphysics(two) === :none
        @test Scythe.mc_var_names(base) == Scythe.MC_VARS
        @test Scythe.mc_var_names(two) == vcat(Scythe.MC_VARS, "n_r")
        @test_throws ErrorException Scythe.ice_microphysics(
            Dict{Symbol,Any}(:ice_microphysics => :morrison))

        # THE VALIDATION: :ishmael against single-moment rain is refused at config time.
        @test_throws ErrorException Scythe.ice_microphysics(
            Dict{Symbol,Any}(:ice_microphysics => :ishmael))
        @test Scythe.ice_microphysics(ice) === :ishmael

        # APPENDED after n_r, species-major, never inserted.
        names_xz = Scythe.mc_var_names(ice)
        names_cyl = Scythe.mc_var_names(ice; cyl = true)
        @test names_xz == vcat(Scythe.MC_VARS, "n_r", collect(Scythe.MC_ICE_VARS))
        @test names_cyl == vcat(Scythe.MC_VARS_CYL, "n_r", collect(Scythe.MC_ICE_VARS))
        @test names_xz[1:11] == vcat(Scythe.MC_VARS, "n_r")
        @test length(names_xz) == 23 && length(names_cyl) == 24
        # The geometry-dependent indices the driver must resolve BY NAME
        @test findfirst(==("rho_i1"), names_xz) == 12
        @test findfirst(==("c_i3"), names_xz) == 23
        @test findfirst(==("rho_i1"), names_cyl) == 13
        @test findfirst(==("c_i3"), names_cyl) == 24
        # Species-major: a species' four moments are contiguous and in (q, n, a, c) order
        @test Scythe.MC_ICE_VARS[1:4] == ("rho_i1", "n_i1", "a_i1", "c_i1")
        @test Scythe.MC_ICE_VARS[5:8] == ("rho_i2", "n_i2", "a_i2", "c_i2")
        @test Scythe.MC_ICE_VARS[9:12] == ("rho_i3", "n_i3", "a_i3", "c_i3")

        # ONE transform family covering all twelve; FOUR widths, because mu is dimensional
        tr = merge(ice, Dict{Symbol,Any}(:ice_transform => :bhyp))
        @test Scythe.ice_transform_mode(ice) === :none
        @test Scythe.ice_transform_mode(tr) === :bhyp
        @test_throws ErrorException Scythe.ice_transform_mode(
            Dict{Symbol,Any}(:ice_transform => :bogus))
        @test Scythe.ice_var_names(ice) == Scythe.MC_ICE_VARS
        @test Scythe.ice_var_names(tr) == ("nu_i1", "nu_ni1", "nu_ai1", "nu_ci1",
                                           "nu_i2", "nu_ni2", "nu_ai2", "nu_ci2",
                                           "nu_i3", "nu_ni3", "nu_ai3", "nu_ci3")
        @test Scythe.mc_var_names(tr)[12:23] == collect(Scythe.ice_var_names(tr))
        for nm in Scythe.MC_ICE_VARS
            @test haskey(Scythe.MC_NU_ALIAS, nm)
        end
        @test length(unique(values(Scythe.MC_NU_ALIAS))) == length(Scythe.MC_NU_ALIAS)
        # Widths: mass, number, a, c — and the defaults are per KIND, not per slot.
        # Each sits two to five decades BELOW the moment it transforms, so `bhyp` is a
        # change of variables in its affine regime rather than a bare positivity floor;
        # see `MC_ICE_MU_DEFAULTS` for the sizing principle and the measurement (at the
        # former mass width of 1e-7 against a 2.7e-10 kg/m^3 field, `bhyp(rho)/rho` was
        # 0.9987 and `J` 0.9973 — the identity to 0.3 %, i.e. no transform at all).
        pp = Dict{Symbol,Float64}()
        @test [Scythe.ice_mu(pp, j) for j in 1:12] ==
              [1.0e-12, 1.0e-2, 1.0e-16, 1.0e-16,
               1.0e-12, 1.0e-2, 1.0e-16, 1.0e-16,
               1.0e-12, 1.0e-2, 1.0e-16, 1.0e-16]
        @test [Scythe.ice_mu(pp, j) for j in 1:12] == collect(Scythe.MC_ICE_MU_DEFAULTS)
        @test Scythe.ice_mu(Dict(:mu_ice_a => 3.0e-15), 3) == 3.0e-15
        # The mass and number widths are below their field scales; the two volume widths
        # were already correctly sized and are deliberately unmoved.
        @test Scythe.ice_mu(pp, 1) < 1.0e-9      # vs ~1e-9 kg/m^3 of species-1 ice
        @test Scythe.ice_mu(pp, 2) < 1.0e3       # vs ~1e3 #/m^3
        @test Scythe.ice_mu(pp, 3) < 8.0e-12     # vs n*r^3 = 1e3*(2e-5)^3

        # All twelve are TOTALS: no reference profile to offset a bound against. The
        # transformed names are refused, like nu_c/nu_r/nu_nr.
        for nm in Scythe.MC_ICE_VARS
            @test Scythe.positivity_reference_profile(nm, nothing) === nothing
            @test_throws ErrorException Scythe.positivity_reference_profile(
                Scythe.MC_NU_ALIAS[nm], nothing)
        end

        # mc_slot resolves by ROLE through either name; mc_optional_slot answers 0.
        vars_ice = Dict(v => i for (i, v) in enumerate(names_xz))
        vars_tr = Dict(v => i for (i, v) in enumerate(Scythe.mc_var_names(tr)))
        vars_1m = Dict(v => i for (i, v) in enumerate(Scythe.MC_VARS))
        @test Scythe.mc_slot(vars_ice, "rho_i2") == 16
        @test Scythe.mc_slot(vars_tr, "rho_i2") == 16          # via the "nu_i2" alias
        @test Scythe.mc_ice_slot_indices(vars_ice) == ntuple(j -> 11 + j, 12)
        @test Scythe.mc_ice_slot_indices(vars_tr) == ntuple(j -> 11 + j, 12)
        @test Scythe.mc_ice_slot_indices(vars_1m) == ntuple(_ -> 0, 12)
        @test_throws ErrorException Scythe.mc_slot(vars_1m, "a_i3")

        # The TOTAL-form transform pair is the shared helper for every mu kind, and
        # bhyp(0) == 0 exactly is what lets every initializer seed a literal 0.0.
        for (j, mu) in enumerate((1.0e-7, 1.0e2, 1.0e-16, 1.0e-16))
            for mode in (:none, :bhyp, :bhyp_smooth)
                @test Scythe.total_slot(0.0, mode, mu) === 0.0
                x = (1.0e-4, 5.0e3, 2.0e-12, 1.0e-12)[j]
                @test Scythe.recover_total(Scythe.total_slot(x, mode, mu), mode, mu) ≈ x rtol=1e-12
            end
        end
    end

    @testset "ice thermodynamics: the ice terms vanish EXACTLY at zero ice" begin
        # The whole "ice-on with zero ice is inert" claim rests on this: the added terms are
        # `+0.0` in the numerator and `-(-0.0)` in the denominator, both exact identities in
        # IEEE — but only if the associativity puts them LAST. `===` is the test, not `≈`.
        for M in (2.0e5, 3.5e5, 6.0e5, -1.0e4)
            for rho_d in (0.3, 0.8, 1.2)
                for rho_t in (rho_d, rho_d + 1.0e-6, rho_d + 0.02)
                    for rho_liq in (0.0, 1.0e-9, 1.0e-4, 3.0e-3, -1.0e-6)
                        @test Scythe.retrieve_temperature(M, rho_d, rho_t, rho_liq) ===
                              Scythe.retrieve_temperature(M, rho_d, rho_t, rho_liq, 0.0)
                    end
                end
            end
        end
        for Tk in (200.0, 253.15, 273.15, 300.0)
            for rho_d in (0.3, 1.2)
                for q_v in (0.0, 1.0e-6, 0.02)
                    for q_l in (0.0, 1.0e-5, 3.0e-3)
                        @test Scythe.moist_entropy_total(Tk, rho_d, q_v, q_l) ===
                              Scythe.moist_entropy_total(Tk, rho_d, q_v, q_l, 0.0)
                        @test Scythe.Q_s_energy(Tk, 8.0e4, rho_d, q_v, q_l) ===
                              Scythe.Q_s_energy(Tk, 8.0e4, rho_d, q_v, q_l, 0.0)
                    end
                end
            end
        end
    end

    @testset "ice thermodynamics: the retrieval at rho_ice > 0" begin
        # Independent evaluation of Eq. T_closed_form_ice — written out from the TeX, not
        # refactored from the implementation — plus the two structural properties the
        # derivation claims: dT/d rho_i = L_s/D_i, and freezing at fixed total condensate
        # warms by L_f/D_i.
        Cpd = Scythe.Cpd; Cpv = Scythe.Cpv; Cl = Scythe.Cl; Ci = Scythe.Ci
        L_v0 = Scythe.L_v0; L_s0 = Scythe.L_s0; T_0 = Scythe.T_0
        tex_T(M, rho_d, rho_t, rl, ri) =
            (M + rl * (L_v0 - (Cpv - Cl) * T_0) + ri * (L_s0 - (Cpv - Ci) * T_0)) /
            ((rho_d * Cpd + (rho_t - rho_d) * Cpv) - rl * (Cpv - Cl) - ri * (Cpv - Ci))
        for M in (2.0e5, 3.5e5, 6.0e5)
            for rho_d in (0.4, 1.2)
                for rl in (0.0, 1.0e-4, 2.0e-3)
                    for ri in (1.0e-7, 1.0e-4, 5.0e-3)
                        rho_t = rho_d + 0.015
                        T = Scythe.retrieve_temperature(M, rho_d, rho_t, rl, ri)
                        @test T ≈ tex_T(M, rho_d, rho_t, rl, ri) rtol=1e-14
                        # The denominator D_i = C_f + rl(Cl-Cpv) + ri(Ci-Cpv) is strictly
                        # positive for any admissible state, so the root never fails.
                        D = (rho_d * Cpd + (rho_t - rho_d) * Cpv) +
                            rl * (Cl - Cpv) + ri * (Ci - Cpv)
                        @test D > 0.0
                        # dT/d rho_i = L_s(T)/D_i (TeX Eq. dTdrhoi), by finite difference
                        h = 1.0e-9
                        dT = (Scythe.retrieve_temperature(M, rho_d, rho_t, rl, ri + h) -
                              Scythe.retrieve_temperature(M, rho_d, rho_t, rl, ri - h)) / (2h)
                        @test dT ≈ Scythe.L_s(T) / D rtol=1e-5
                        # FREEZING moves mass between the two condensed slots at fixed total
                        # water, so the retrieval warms by L_f/D_i and by nothing else.
                        dTf = (Scythe.retrieve_temperature(M, rho_d, rho_t, rl - h, ri + h) -
                               Scythe.retrieve_temperature(M, rho_d, rho_t, rl + h, ri - h)) / (2h)
                        @test dTf ≈ Scythe.L_f(T) / D rtol=1e-5
                    end
                end
            end
        end
        # Ice makes the column WARMER than the same mass of liquid would (it released more
        # latent heat getting there), and the two agree in the L_f -> 0 sense nowhere else.
        T_liq = Scythe.retrieve_temperature(3.5e5, 1.0, 1.02, 3.0e-3, 0.0)
        T_ice = Scythe.retrieve_temperature(3.5e5, 1.0, 1.02, 0.0, 3.0e-3)
        @test T_ice > T_liq
        # Kirchhoff: L_s == L_v + L_f identically, which is what makes the three
        # linearizations mutually consistent.
        for Tk in (220.0, 273.15, 305.0)
            @test Scythe.L_s(Tk) ≈ Scythe.L_v(Tk) + Scythe.L_f(Tk) rtol=1e-14
        end
        # The mixture heat capacity takes ice through the same q*C as liquid
        @test Scythe.Q_s_energy(260.0, 8.0e4, 1.0, 0.004, 0.0, 0.001) !=
              Scythe.Q_s_energy(260.0, 8.0e4, 1.0, 0.004, 0.0, 0.0)
        @test Scythe.moist_entropy_total(260.0, 1.0, 0.004, 0.0, 0.001) ≈
              Scythe.moist_entropy_total(260.0, 1.0, 0.004, 0.0, 0.0) +
              0.001 * Ci * log(260.0 / T_0) rtol=1e-14
    end

    @testset "ice: a tile caches the twelve slots by name" begin
        mktempdir() do tmpdir
            ice = Dict{Symbol,Any}(:rain_moments => 2, :ice_microphysics => :ishmael)

            m0, _, _, _ = make_mc_mtile(tmpdir)
            @test !Scythe.ice_registered(m0.mc_slots)          # absent, and says so
            @test Scythe.ice_slots(m0.mc_slots, 2) == (0, 0, 0, 0)

            m, _, mod, _ = make_mc_mtile(tmpdir; precipitation = true, extra_options = ice)
            @test Scythe.ice_registered(m.mc_slots)
            @test m.mc_slots.rho_v == 10
            @test m.mc_slots.n_r == 11
            @test Scythe.ice_slots(m.mc_slots, 1) == (12, 13, 14, 15)
            @test Scythe.ice_slots(m.mc_slots, 2) == (16, 17, 18, 19)
            @test Scythe.ice_slots(m.mc_slots, 3) == (20, 21, 22, 23)
            @test length(mod.grid_params.vars) == 23
            # Concrete: resolving twelve more indices must not cost the tile its typing
            @test isconcretetype(fieldtype(typeof(m), :mc_slots))
            # Every ice slot got a scratch column of its own, so each flux is fitted on its
            # own basis and its own BCs.
            @test size(m.scratch_columns, 2) == 23

            # Cylindrical: the same twelve, one index further out, resolved by name
            m3, _, mod3, _ = make_mc_mtile(tmpdir; precipitation = true, iMin = 100.0,
                iMax = 2100.0, equation_set = "moist_compressible_axisym",
                extra_options = ice)
            @test mod3.grid_params.vars["v"] == 10
            @test m3.mc_slots.rho_v == 11
            @test m3.mc_slots.n_r == 12
            @test Scythe.ice_slots(m3.mc_slots, 1) == (13, 14, 15, 16)
            @test Scythe.ice_slots(m3.mc_slots, 3) == (21, 22, 23, 24)
        end
    end

    @testset "ice: check_mc_var_names and the deferred-support guards" begin
        mktempdir() do tmpdir
            ice = Dict{Symbol,Any}(:rain_moments => 2, :ice_microphysics => :ishmael)
            m, _, mod, _ = make_mc_mtile(tmpdir; precipitation = true, extra_options = ice)
            @test Scythe.check_mc_var_names(mod) === nothing

            # A `vars` built from a stale name list has no slot for the tendency to land in.
            gp_bad = deepcopy(mod.grid_params)
            delete!(gp_bad.vars, "a_i2")
            bad = ModelParameters(ts = mod.ts, equation_set = mod.equation_set,
                                  ref_state_file = mod.ref_state_file,
                                  grid_params = gp_bad,
                                  physical_params = mod.physical_params,
                                  options = mod.options)
            @test_throws ErrorException Scythe.check_mc_var_names(bad)

            # ... and the rain_moments requirement fires from check_mc_var_names too, which
            # is what makes it a CONFIGURATION-time error rather than a first-use one.
            gp_1m = deepcopy(mod.grid_params)
            bad_1m = ModelParameters(ts = mod.ts, equation_set = mod.equation_set,
                                     ref_state_file = mod.ref_state_file,
                                     grid_params = gp_1m,
                                     physical_params = mod.physical_params,
                                     options = merge(mod.options,
                                         Dict{Symbol,Any}(:rain_moments => 1)))
            @test_throws ErrorException Scythe.check_mc_var_names(bad_1m)

            kDim = mod.grid_params.kDim
            # Each of these would run and produce plausible numbers while moving water mass
            # that the ice is part of, without the ice. They refuse.
            mod.options[:clamp_water] = true
            @test_throws ErrorException Scythe.clamp_water!(m, 1, kDim)
            delete!(mod.options, :clamp_water)
            @test Scythe.clamp_water!(m, 1, kDim) === nothing   # measurement still runs

            mod.options[:water_budget_trace] = 1
            @test_throws ErrorException Scythe.advance_column(m, 1, 1)
            delete!(mod.options, :water_budget_trace)

            mod.physical_params[:Kvdiff_water] = 1.0
            @test_throws ErrorException Scythe.advance_column(m, 1, 1)
            mod.physical_params[:Kvdiff_water] = 0.0

            mod.physical_params[:Khdiff_water] = 1.0
            @test_throws ErrorException Scythe.advance_column(m, 1, 1)
            mod.physical_params[:Khdiff_water] = 0.0

            mod.options[:louis_bl] = true
            @test_throws ErrorException Scythe.advance_column(m, 1, 1)
            delete!(mod.options, :louis_bl)

            @test Scythe.advance_column(m, 1, 1) === nothing

            # A transformed ice slot may not ALSO carry a coefficient bound.
            gp_pos = deepcopy(mod.grid_params)
            gp_pos.positivity["rho_i1"] = Dict(:k => 0.0)
            pos_model = ModelParameters(ts = mod.ts, equation_set = mod.equation_set,
                                        ref_state_file = mod.ref_state_file,
                                        grid_params = gp_pos,
                                        physical_params = mod.physical_params,
                                        options = merge(mod.options,
                                            Dict{Symbol,Any}(:ice_transform => :bhyp)))
            @test_throws ErrorException Scythe.install_positivity_bounds!(
                m.tile, m.ref_state, pos_model)
        end
    end

    @testset "ice: a seeded blob advects, all four moments together" begin
        # TRANSPORT ONLY is the whole claim of this stage, so this is the test of it: seed a
        # smooth Gaussian in all four moments of species 1, with the RAIN MASS seeded to the
        # identical profile as a control, and integrate a few hundred steps of real flow with
        # every process rate off.
        #
        # Slot 8 and the ice mass slot then obey the SAME discrete equation — pure continuity
        # in a transform-free total — so they must stay equal to round-off. And the four ice
        # moments differ only by their seeded constant, so their RATIOS must not move: that is
        # the statement that mass, number and volume are being transported by one operator and
        # not by four slightly different ones.
        mktempdir() do tmpdir
            kDim = 32
            opts = Dict{Symbol,Any}(:rain_moments => 2, :ice_microphysics => :ishmael,
                                    :condensation => false)
            m, patch, mod, col = make_mc_mtile(tmpdir; q_l = 0.0, kDim = kDim,
                num_cells = 8, ts = 0.05, precipitation = false, extra_options = opts)
            s = m.mc_slots
            gpts = Scythe.getGridpoints(patch)
            # Four constants nine decades apart, as the real moments are: mass [kg/m³],
            # number [#/m³], and the two volume moments [m³/m³].
            scale = (1.0e-4, 5.0e3, 2.0e-12, 1.0e-12)
            for i in 1:size(patch.physical, 1)
                z = gpts[i, 2]
                g = exp(-((z - 1200.0) / 300.0)^2)
                patch.physical[i, s.i1_q, 1] = scale[1] * g
                patch.physical[i, s.i1_n, 1] = scale[2] * g
                patch.physical[i, s.i1_a, 1] = scale[3] * g
                patch.physical[i, s.i1_c, 1] = scale[4] * g
                patch.physical[i, 8, 1] = scale[1] * g       # rho_r: the control tracer
                patch.physical[i, 5, 1] = 5.0                # w: the flow that moves it
            end
            spectralTransform!(patch)
            gridTransform!(patch)

            q0 = copy(patch.physical[:, s.i1_q, 1])
            sum0 = sum(q0)
            @test sum0 > 0.0

            step_mc!(m, patch, mod, 300)                     # 15 s of advection

            @test all(isfinite.(patch.physical))
            @test all(isfinite.(m.expdot_n))

            q1 = @view patch.physical[:, s.i1_q, 1]
            n1 = @view patch.physical[:, s.i1_n, 1]
            a1 = @view patch.physical[:, s.i1_a, 1]
            c1 = @view patch.physical[:, s.i1_c, 1]
            rr = @view patch.physical[:, 8, 1]

            # It actually MOVED: a test that passed on a stationary blob would prove nothing.
            @test maximum(abs.(q1 .- q0)) > 0.01 * maximum(q0)

            # The ice mass and the rain mass obeyed the same equation and stayed together.
            @test maximum(abs.(q1 .- rr)) < 1.0e-12 * maximum(abs.(q1))

            # RATIOS at the blob peak — and everywhere the blob is resolved — are fixed.
            pk = argmax(q1)
            @test n1[pk] / q1[pk] ≈ scale[2] / scale[1] rtol=1e-10
            @test a1[pk] / q1[pk] ≈ scale[3] / scale[1] rtol=1e-10
            @test c1[pk] / q1[pk] ≈ scale[4] / scale[1] rtol=1e-10
            big = findall(>(0.05 * maximum(q1)), q1)
            @test maximum(abs.((n1[big] ./ q1[big]) .- (scale[2] / scale[1]))) <
                  1.0e-9 * (scale[2] / scale[1])
            @test maximum(abs.((a1[big] ./ q1[big]) .- (scale[3] / scale[1]))) <
                  1.0e-9 * (scale[3] / scale[1])

            # The column integral is carried, not created: the advective product-rule form
            # conserves it to the same order the rain does, which is the bar set here.
            @test abs(sum(q1) - sum0) < 0.02 * sum0
            @test sum(n1) / sum(q1) ≈ scale[2] / scale[1] rtol=1e-9

            # The other ELEVEN slots are finite and the ice slots produced no NaN anywhere.
            for slot in 1:(s.i1_q - 1)
                @test all(isfinite.(patch.physical[:, slot, 1]))
            end
            for slot in s.i1_q:(s.i1_q + 11)
                @test all(isfinite.(patch.physical[:, slot, 1]))
            end
            # Species 2 and 3 were never seeded and no process can create them.
            for slot in s.i2_q:(s.i1_q + 11)
                @test all(patch.physical[:, slot, 1] .== 0.0)
            end
        end
    end

    @testset "ice: zero ice is inert on the column" begin
        # The gate the O01 regression makes on a real storm, made here on a column: an ice-ON
        # run with zero ice must reproduce an ice-OFF run's ten common slots, because the
        # rho_i = 0 algebra is exact everywhere it was added.
        mktempdir() do tmpdir
            args = (; q_l = 3.0e-3, kDim = 16, num_cells = 8, precipitation = true)
            two = Dict{Symbol,Any}(:rain_moments => 2)
            ice = Dict{Symbol,Any}(:rain_moments => 2, :ice_microphysics => :ishmael)
            function run_once(opts)
                m, patch, mod, _ = make_mc_mtile(tmpdir; args..., extra_options = opts)
                ncols = div(size(m.tile.physical, 1), mod.grid_params.kDim)
                for t in 1:5, c in 1:ncols
                    Scythe.advance_column(m, c, t)
                end
                return m
            end
            m_off = run_once(two)
            m_on = run_once(ice)
            i1q = m_on.mc_slots.i1_q                    # 12: the first ice slot
            for slot in 1:(i1q - 1)                     # the eleven common slots
                @test m_on.var_np1[:, slot] == m_off.var_np1[:, slot]
                @test m_on.expdot_n[:, slot] == m_off.expdot_n[:, slot]
            end
            # ...and the twelve ice slots are EXACTLY zero, not merely small.
            for slot in i1q:(i1q + 11)
                @test all(m_on.var_np1[:, slot] .== 0.0)
                @test all(m_on.expdot_n[:, slot] .== 0.0)
            end
        end
    end

    # ══════════════════════════════════════════════════════════════════════════
    # S8: the LIVE ice physics — deposition through the shared Q_ss, the ISHMAEL
    # process set, the fall speeds, and the pressure/Q_ss/energy coupling.
    # reference/Scythe_moist_compressible.tex §Ice Processes is the specification.
    # ══════════════════════════════════════════════════════════════════════════

    """A supercooled mixed-phase column: liquid-saturated at about -15 C."""
    function supercooled_column_mc(z; q_l = 1.0e-3, Tsurf = 258.15)
        Tk = @. Tsurf - 0.005 * z
        p_Pa = @. 70000.0 * exp(-z / 8000.0)
        rho_v = rho_v_sat.(Tk, p_Pa ./ 100.0)
        rho_d = (p_Pa .- (Rv .* Tk .* rho_v)) ./ (Rd .* Tk)
        rho_c = q_l .* rho_d
        return (; z, Tk, p_Pa, rho_d, rho_v, rho_c)
    end

    """
    A resting mixed-phase tile: `make_mc_mtile`'s machinery on the supercooled column, with
    ice species 1 seeded to `rho_i`/`n_i` and monodisperse-equivalent volume moments.
    Species 2 (columnar) is seeded the same way from `rho_i2`/`n_i2`/`r_i2`, which default to
    an EMPTY population: the aggregation kernel's cross-species pairs need a second live
    species, and DeMott nucleation only ever fills one of the two habits (the one the
    inherent growth ratio selects at the column's temperature).
    """
    # `n_i_levels`: restrict species 1's NUMBER (and its two volume moments) to those
    # vertical levels of every column, leaving its MASS uniform. That is a column with a
    # live population over part of its depth and number-less mass over the rest — the
    # fixture the `:local` seeding needs, and the shape the transport's ringing actually
    # makes. `nothing` (the default) is the uniform column every other test uses.
    function make_ice_mtile(tmpdir; Tsurf = 258.15, q_l = 1.0e-3, rho_i = 1.0e-5,
                            n_i = 5.0e4, r_i = 30.0e-6, rho_r = 0.0, n_r = 0.0,
                            rho_i2 = 0.0, n_i2 = 0.0, r_i2 = 30.0e-6, n_i_levels = nothing,
                            kDim = 16, ts = 0.1, extra_options = Dict{Symbol,Any}(),
                            extra_params = Dict{Symbol,Float64}())
        # No `:vapor_retrieval` here: it was the ice arm's opt-out from the regime-blended
        # retrieval, and there is no retrieval left to opt out of — the vapor is a slot.
        opts = merge(Dict{Symbol,Any}(:rain_moments => 2, :ice_microphysics => :ishmael),
                     extra_options)
        varlist = Scythe.mc_var_names(opts; cyl = false)
        vars = Dict(v => i for (i, v) in enumerate(varlist))
        scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
        wall_bc = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))
        gp = GridParameters(geometry = "RZ", num_cells = 8,
            iMin = 0.0, iMax = 2000.0, kMin = 0.0, kMax = 2000.0, kDim = kDim,
            BCL = wall_bc, BCR = wall_bc, BCB = wall_bc, BCT = wall_bc, vars = vars)
        patch = createGrid(gp)
        z = Scythe.getGridpoints(patch)[1:kDim, 2]
        col = supercooled_column_mc(z; q_l = q_l, Tsurf = Tsurf)
        ref = joinpath(tmpdir, "ice_$(Tsurf)_$(q_l).ref")
        Scythe.write_exact_ref_mc(ref, z, col.p_Pa, col.rho_d, col.rho_v, col.rho_c)
        model = ModelParameters(ts = ts, integration_time = 1.0, output_interval = 1.0,
            equation_set = "moist_compressible_XZ", ref_state_file = ref, grid_params = gp,
            physical_params = merge(Dict(:Khdiff => 0.0, :Kvdiff => 0.0, :Kvdiff_heat => 0.0,
                :Kvdiff_water => 0.0, :Kv_mudiff => 0.0, :tau_qss => 10.0, :N_r => 1.0e-3,
                :alpha => 0.0, :z_damp => 20.0e3, :f => 0.0), extra_params),
            options = merge(Dict{Symbol,Any}(:semiimplicit => true,
                :exact_reference_state => true, :precipitation => true,
                :vertical_mixing => false), opts))
        patch.physical .= 0.0
        s = Dict(v => i for (i, v) in vars)
        for i in 1:size(patch.physical, 1)
            n_here = (n_i_levels === nothing || mod1(i, kDim) in n_i_levels) ? n_i : 0.0
            patch.physical[i, vars["rho_i1"], 1] = rho_i
            patch.physical[i, vars["n_i1"], 1] = n_here
            patch.physical[i, vars["a_i1"], 1] = n_here * r_i^3
            patch.physical[i, vars["c_i1"], 1] = n_here * r_i^3
            patch.physical[i, vars["rho_i2"], 1] = rho_i2
            patch.physical[i, vars["n_i2"], 1] = n_i2
            patch.physical[i, vars["a_i2"], 1] = n_i2 * r_i2^3
            patch.physical[i, vars["c_i2"], 1] = n_i2 * r_i2^3
            patch.physical[i, 8, 1] = rho_r
            patch.physical[i, vars["n_r"], 1] = n_r
            # The seeded condensate comes OUT OF THE VAPOR, which the prognostic slot has
            # to be told: rho_t is untouched here, so leaving rho_v at its reference would
            # make the seed a reconciliation gap of its own size rather than a partition.
            # (Under the retired residual retrieval this subtraction was implicit — the
            # vapor WAS rho_t minus the condensates.)
            patch.physical[i, vars["rho_v"], 1] = -((rho_i + rho_i2) + rho_r)
        end
        spectralTransform!(patch)
        gridTransform!(patch)
        mtile = createModelTile(patch, patch, model, sparse(Int64[], Int64[], Float64[],
            size(patch.spectral, 1), size(patch.spectral, 2)))
        return mtile, patch, model, col
    end

    @testset "ice: 𝒟 and the deposition drive clip" begin
        # TeX Eqs. Dwi and qss_ice_shift, and the ice mirror of the liquid clip in
        # `qss_condensation_rates`.
        p_hPa = 700.0

        # 𝒟 = max(rho_vs - rho_i_sat, 0): EXACTLY zero above the triple point (the `max`
        # closes it there), negligible AT it, positive below, peaking near -12 C (the
        # classical WBF maximum). At T_0 itself the two Buck branches differ by 6e-5 of
        # rho_vs rather than by nothing -- they are separate fits, not one function evaluated
        # twice -- so this is a statement about the FORMULATION's internal consistency, which
        # is the accuracy 𝒟 is only ever as good as (TeX §Departures (c)).
        @test Scythe.ice_supersaturation_gap(Scythe.T_0, p_hPa) <
              1.0e-4 * rho_v_sat(Scythe.T_0, p_hPa)
        @test Scythe.ice_supersaturation_gap(Scythe.T_0 + 10.0, p_hPa) == 0.0
        @test Scythe.ice_supersaturation_gap(Scythe.T_0 - 15.0, p_hPa) > 0.0
        gaps = [Scythe.ice_supersaturation_gap(Scythe.T_0 + dT, p_hPa) for dT in -40.0:0.5:0.0]
        peakT = (-40.0:0.5:0.0)[argmax(gaps)]
        @test -16.0 < peakT < -8.0

        for Tk in (268.15, 258.15, 243.15)
            rvs = rho_v_sat(Tk, p_hPa)
            rvsi = Springsteel.Thermodynamics.rho_i_sat(Tk, p_hPa)
            D = rvs - rvsi
            @test D > 0.0

            # THE CONTINUUM IDENTITY. With Q_ss on the density budget the clip never binds
            # and the drive is exactly the over-ice supersaturation density.
            rho_v = 1.05 * rvs
            @test Scythe.ice_deposition_drive(rho_v - rvs, rho_v, Tk, p_hPa) ≈
                  rho_v - rvsi rtol = 1e-14

            # DRY AIR IS INERT the same way the liquid channel is: whatever the detached
            # prognostic says, the drive is the vapor-free-air maximum sublimation drive and
            # no more, so nothing can deposit vapor that is not there.
            @test Scythe.ice_deposition_drive(1.0, 0.0, Tk, p_hPa) == -rvsi

            # THE NEGATIVE-VAPOR FLOOR: a partition error must not set the rate.
            @test Scythe.ice_deposition_drive(1.0, -0.5, Tk, p_hPa) == -rvsi
            # ...but a prognostic that has detached DOWNWARD passes through unfloored: it is
            # advanced by the smooth multistep integrator and is not the pathology the floor
            # addresses (the liquid clip's own rule).
            @test Scythe.ice_deposition_drive(-0.9, -0.5, Tk, p_hPa) == -0.9 + D

            # SUBLIMATION is the same expression, negative, with no case distinction.
            sub = Scythe.ice_deposition_drive(0.5 * rvsi - rvs, 0.5 * rvsi, Tk, p_hPa)
            @test sub < 0.0
            @test sub ≈ -0.5 * rvsi rtol = 1e-14
        end

        # Above T_0 the gap closes, so the drive collapses onto the liquid one.
        Tw = 290.0
        rvsw = rho_v_sat(Tw, p_hPa)
        @test Scythe.ice_deposition_drive(0.02 * rvsw, 1.02 * rvsw, Tw, p_hPa) ≈
              1.02 * rvsw - Springsteel.Thermodynamics.rho_i_sat(Tw, p_hPa) rtol = 1e-14
    end

    @testset "Q_s_energy_ice is Q_s_energy with L_v -> L_s and nothing else" begin
        # Independent transcription of TeX Eq. Qs_ice, written out rather than reusing the
        # implementation's own factorization.
        for (Tk, p_Pa, q_v, q_l, q_i) in ((258.15, 70000.0, 2.0e-3, 1.0e-3, 5.0e-4),
                                          (243.15, 40000.0, 5.0e-4, 0.0, 2.0e-4),
                                          (268.15, 85000.0, 3.5e-3, 2.0e-3, 0.0))
            rho_d = p_Pa / (Rd * Tk)
            C_vt = Cvd + q_v * Cvv + q_l * Cl + q_i * Scythe.Ci
            R_m = Rd + q_v * Rv
            C_pt = C_vt + R_m
            Ls = Springsteel.Thermodynamics.L_s(Tk)
            dT = Scythe.drho_vsat_dT(Tk, p_Pa / 100.0)
            dp = Scythe.drho_vsat_dp(Tk, p_Pa / 100.0)
            expect = (dT * (Ls - Rv * Tk) / rho_d +
                      dp * R_m * (Ls - Rv * C_pt * Tk / R_m)) / C_vt
            @test Scythe.Q_s_energy_ice(Tk, p_Pa, rho_d, q_v, q_l, q_i) ≈ expect rtol = 1e-14

            # It is NOT Q_s re-evaluated over ice: the two saturation derivatives are the
            # WATER ones in both terms. The only difference from the liquid factor is L.
            Qs = Scythe.Q_s_energy(Tk, p_Pa, rho_d, q_v, q_l, q_i)
            Lv = L_v(Tk)
            # Both factors share the same two derivative slots, so their difference is the
            # latent-heat difference times the same bracket coefficients -- exactly L_f.
            Lf = Springsteel.Thermodynamics.L_f(Tk)
            @test Scythe.Q_s_energy_ice(Tk, p_Pa, rho_d, q_v, q_l, q_i) - Qs ≈
                  ((dT / rho_d) + (dp * R_m)) * Lf / C_vt rtol = 1e-12
            @test Ls ≈ Lv + Lf rtol = 1e-14
        end
    end

    @testset "ice: warm air with ice present is exactly inert" begin
        # The state gate that keeps the bitwise inertness claim alive now that the physics is
        # live: above T_0 the deposition channel is closed and aggregation does not run, so a
        # WARM column with ice in it still produces exact zeros from those two channels.
        mktempdir() do tmpdir
            m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 290.0, q_l = 1.0e-3)
            Scythe.advance_column(m, 1, 2)
            S = m.mc_scratch[Threads.threadid()]
            @test all(S.Tk .> Scythe.T_0)
            # Deposition is identically zero above freezing (Eq. dep_rate is closed there).
            @test all(S.Qdot_i1 .== 0.0)
            @test all(S.Qdot_i2 .== 0.0)
            @test all(S.Qdot_i3 .== 0.0)
            # Species 2 and 3 have no ice, so every one of their rates is an EXACT zero and
            # their fall speeds are too -- which is what lets `_ice_flux!` skip the fit.
            for nm in (:SRC_i2q, :SRC_i2n, :SRC_i2a, :SRC_i2c,
                       :SRC_i3q, :SRC_i3n, :SRC_i3a, :SRC_i3c,
                       :F_i2q_z, :F_i2n_z, :F_i3q_z, :F_i3n_z, :Vi2m, :Vi2n, :Vi3m, :Vi3n)
                @test all(getproperty(S, nm) .== 0.0)
            end
        end
    end

    @testset "ice: zero ice in WARM air is bitwise the ice-free run" begin
        # The strict gate. Every rate is gated on a STATE test, so with no ice and T > T_0
        # throughout, the ten common slots must match the ice-off run to the last bit --
        # not to 1e-10.
        mktempdir() do tmpdir
            args = (; q_l = 3.0e-3, kDim = 16, num_cells = 8, precipitation = true)
            two = Dict{Symbol,Any}(:rain_moments => 2)
            ice = Dict{Symbol,Any}(:rain_moments => 2, :ice_microphysics => :ishmael)
            function run_once(opts)
                m, patch, mod, _ = make_mc_mtile(tmpdir; args..., extra_options = opts)
                ncols = div(size(m.tile.physical, 1), mod.grid_params.kDim)
                for t in 1:5, c in 1:ncols
                    Scythe.advance_column(m, c, t)
                end
                return m
            end
            m_off = run_once(two)
            m_on = run_once(ice)
            i1q = m_on.mc_slots.i1_q                    # 12: the first ice slot, by name
            for slot in 1:(i1q - 1)                     # the eleven common slots
                @test m_on.var_np1[:, slot] == m_off.var_np1[:, slot]
                @test m_on.expdot_n[:, slot] == m_off.expdot_n[:, slot]
            end
            for slot in i1q:(i1q + 11)
                @test all(m_on.expdot_n[:, slot] .== 0.0)
            end
        end
    end

    @testset "ice: the attribution census is inert — warm path and default-off" begin
        # Stage 0a. The block REPORTS and never limits, so two independent statements have
        # to hold: (1) with the option ON, a WARM ice-free column is still bitwise the
        # ice-free run AND every attribution row is an exact 0.0 — every write in the block
        # is gated on a state test that column fails identically; (2) with ice actually
        # running, switching the option on changes no tendency and no state at all, because
        # the only thing it writes is `mc_water_stats`.
        mktempdir() do tmpdir
            # (1) the warm gate, the ":5216 pattern" with the census switched on in BOTH.
            args = (; q_l = 3.0e-3, kDim = 16, num_cells = 8, precipitation = true)
            two = Dict{Symbol,Any}(:rain_moments => 2, :ice_attr_census => true)
            ice = Dict{Symbol,Any}(:rain_moments => 2, :ice_microphysics => :ishmael,
                                   :ice_attr_census => true)
            function run_warm(opts)
                m, patch, mod, _ = make_mc_mtile(tmpdir; args..., extra_options = opts)
                ncols = div(size(m.tile.physical, 1), mod.grid_params.kDim)
                for t in 1:5, c in 1:ncols
                    Scythe.advance_column(m, c, t)
                end
                return m
            end
            m_off = run_warm(two)
            m_on = run_warm(ice)
            i1q = m_on.mc_slots.i1_q
            for slot in 1:(i1q - 1)
                @test m_on.var_np1[:, slot] == m_off.var_np1[:, slot]
                @test m_on.expdot_n[:, slot] == m_off.expdot_n[:, slot]
            end
            @test all(view(m_on.mc_water_stats,
                           Scythe.MC_ATTR_FIRST:Scythe.MC_ATTR_LAST, :) .== 0.0)
            @test all(view(m_off.mc_water_stats,
                           Scythe.MC_ATTR_FIRST:Scythe.MC_ATTR_LAST, :) .== 0.0)
        end

        # (2) default-off on a COLD ice column: the census writes only into the stats.
        mktempdir() do tmpdir
            function run_cold(opts)
                m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 250.15, rho_i = 1.0e-3,
                                                  n_i = 1.0e6, rho_r = 1.0e-4, n_r = 1.0e3,
                                                  ts = 1.0, extra_options = opts)
                ncols = div(size(m.tile.physical, 1), mod.grid_params.kDim)
                for t in 1:4, c in 1:ncols
                    Scythe.advance_column(m, c, t)
                end
                return m
            end
            m_plain = run_cold(Dict{Symbol,Any}())
            m_attr = run_cold(Dict{Symbol,Any}(:ice_attr_census => true))
            @test m_plain.expdot_n == m_attr.expdot_n
            @test m_plain.var_np1 == m_attr.var_np1
            # ...and the default really is off: nothing was written.
            @test all(view(m_plain.mc_water_stats,
                           Scythe.MC_ATTR_FIRST:Scythe.MC_ATTR_LAST, :) .== 0.0)
        end
    end

    @testset "ice: the attribution census partitions the q_r donor breach" begin
        # The identity block A is built on: at every breach point exactly ONE of the five
        # q_r debit channels is credited (the largest debit), so their counts sum to the
        # `MC_DONOR_QR` count and no leg is blamed twice. And the two bounds that say the
        # shares are shares: no single leg exceeds the net depletion, and the sum of the
        # four true debits (`MC_ATTR_QR_ALL`) is at least the largest of them.
        #
        # `MC_ATTR_QR_ALL >= MC_DONOR_QR` was the third bound here until Stage 2b, on the
        # reading that the four debits are the net ice draw PLUS the melt credit. That
        # census row is no longer the ice draw: it is the COMBINED draw, ice legs plus the
        # realized evaporation (TeX §donor_relax), and block A decomposes only the ice half
        # — evaporation has block B to itself, at the applied step-mean. The two rows are
        # therefore no longer comparable in either direction at a point (`QR_ALL` drops the
        # melt credit, `MC_DONOR_QR` adds the evaporation), and the bound that survives is
        # the one internal to block A.
        mktempdir() do tmpdir
            m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 250.15, rho_i = 1.0e-3,
                                              n_i = 1.0e6, rho_r = 1.0e-4, n_r = 1.0e3,
                                              ts = 1.0,
                                              extra_options = Dict{Symbol,Any}(
                                                  :ice_attr_census => true))
            ncols = div(size(m.tile.physical, 1), mod.grid_params.kDim)
            m.mc_water_stats .= 0.0
            for t in 1:6, c in 1:ncols
                Scythe.advance_column(m, c, t)
            end
            st = m.mc_water_stats
            drow = Scythe.MC_DONOR_FIRST + Scythe.MC_DONOR_N * (Scythe.MC_DONOR_QR - 1)
            arow(ch) = Scythe.MC_ATTR_FIRST + Scythe.MC_ATTR_N * (ch - 1)
            dmax = maximum(view(st, drow, :))
            dcount = sum(view(st, drow + 1, :))
            # PRECONDITION: this state actually breaches, or the identities below are
            # trivially true and prove nothing. Measured on this fixture: 78 breach points
            # at max 1.0000000000000002 -- the SATURATED rain donor, `1 - exp(-kappa*dt)`
            # rounding to 1.0 and the division landing one ULP over it, and Bigg freezing
            # carrying the largest debit at all 78. That is exactly why block A's gate has
            # to be `mc_donor_census!`'s expression character for character: at one ULP the
            # two forms disagree about which points are breach points, and the partition
            # below then misses by a handful of counts.
            @test dcount > 0.0
            @test dmax > 1.0
            legs = Scythe.MC_ATTR_QR_HOM:Scythe.MC_ATTR_QR_COLL
            acount = sum(sum(view(st, arow(ch) + 1, :)) for ch in legs)
            amax = maximum(maximum(view(st, arow(ch), :)) for ch in legs)
            @test acount == dcount                       # the partition
            @test amax <= dmax + 1.0e-12                 # a share is a share
            # The sum of the four debits is at least the largest single one — pointwise, so
            # the run-maxima inherit it whichever points they were taken at.
            @test maximum(view(st, arow(Scythe.MC_ATTR_QR_ALL), :)) >= amax - 1.0e-12
            # The melt CREDIT is never counted (it adds rain; it cannot be the largest
            # debit), and the sub-part channel keeps its own run-max.
            @test sum(view(st, arow(Scythe.MC_ATTR_QR_MELT) + 1, :)) == 0.0
            @test maximum(view(st, arow(Scythe.MC_ATTR_QR_RIMEX), :)) <=
                  maximum(view(st, arow(Scythe.MC_ATTR_QR_RIME), :)) + 1.0e-12
        end
    end

    @testset "ice: aggregation is realized against the ice donors" begin
        # Stage 1. Aggregation used to be counted in the ice donors' conductance and then
        # applied UNSCALED, on the claim that `qagg3` could not be decomposed into the two
        # donors it came from. It can: the recipient gains exactly what the donors lose. So
        # `mc_ice_sources!` now runs the kernel TWICE — once at unit factors to form the
        # conductance, once at `f_ice1`/`f_ice2` to apply it — and forms `qagg3` in the
        # caller as `-(qagg1 + qagg2)`. Two claims: the donor census is back under 1, and
        # the cross-species exchange closes on the REALIZED numbers.
        mktempdir() do tmpdir
            drow(ch) = Scythe.MC_DONOR_FIRST + Scythe.MC_DONOR_N * (ch - 1)
            arow(ch) = Scythe.MC_ATTR_FIRST + Scythe.MC_ATTR_N * (ch - 1)

            # (1) The STIFF cold driver: both habits carrying ice at 1e-3 kg/m^3 and
            # 1e6 /m^3 (the ice-number cap), a full second of timestep, cloud water present
            # so riming and the anchor are live too. Aggregation is ACTIVE here — the
            # planar/columnar cross pairs and both self-collections all fire — which is what
            # makes the bound a statement rather than a tautology.
            m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 259.15,
                                              rho_i = 1.0e-3, n_i = 1.0e6,
                                              rho_i2 = 1.0e-3, n_i2 = 1.0e6, ts = 1.0,
                                              extra_options = Dict{Symbol,Any}(
                                                  :ice_attr_census => true))
            ncols = div(size(m.tile.physical, 1), mod.grid_params.kDim)
            m.mc_water_stats .= 0.0
            for t in 1:4, c in 1:ncols
                Scythe.advance_column(m, c, t)
            end
            st = m.mc_water_stats
            i1max = maximum(view(st, drow(Scythe.MC_DONOR_I1), :))
            i2max = maximum(view(st, drow(Scythe.MC_DONOR_I2), :))
            # ACTIVE: the melt legs are exactly zero in this column (it is below `T_0`
            # throughout), so these numbers ARE the realized aggregation draw.
            @test i1max > 0.0
            @test i2max > 0.0
            # ...and BOUNDED, which is the whole of Stage 1.
            @test i1max <= 1.0 + 1.0e-9
            @test i2max <= 1.0 + 1.0e-9
            @test maximum(view(st, drow(Scythe.MC_DONOR_I3), :)) <= 1.0 + 1.0e-9
            # The attribution census reads the same realized increment, so its aggregation
            # channels are under the bound too (they were the ones that convicted it).
            @test maximum(view(st, arow(Scythe.MC_ATTR_AGG1), :)) > 0.0
            @test maximum(view(st, arow(Scythe.MC_ATTR_AGG1), :)) <= 1.0 + 1.0e-9
            @test maximum(view(st, arow(Scythe.MC_ATTR_AGG2), :)) <= 1.0 + 1.0e-9
        end

        # (2) The EXCHANGE, on a column where aggregation is the only thing moving ice mass:
        # no cloud and no rain (so no riming), below `T_0` (so no melting), and the
        # deposition mass is withheld from the slot sources by construction. Species 1's and
        # 2's mass sources are then their realized aggregation losses exactly, and species
        # 3's is the gain — which must be their exact negative, because the caller forms it
        # that way rather than taking the kernel's own six-term sum.
        mktempdir() do tmpdir
            m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 259.15, q_l = 0.0,
                                              rho_i = 1.0e-5, n_i = 1.0e6,
                                              rho_i2 = 1.0e-5, n_i2 = 1.0e6, ts = 1.0)
            ncols = div(size(m.tile.physical, 1), mod.grid_params.kDim)
            for t in 1:4, c in 1:ncols
                Scythe.advance_column(m, c, t)
            end
            S = m.mc_scratch[Threads.threadid()]
            s1 = S.SRC_i1q; s2 = S.SRC_i2q; s3 = S.SRC_i3q
            scale = max(maximum(abs, s1), maximum(abs, s2), maximum(abs, s3))
            @test scale > 0.0                            # aggregation ran
            # The donors only lose, the recipient only gains, and species 3 gains WHEREVER
            # species 1's aggregation part is negative.
            @test all(s1 .<= 0.0)
            @test all(s2 .<= 0.0)
            @test all(s3 .>= 0.0)
            for i in eachindex(s1)
                s1[i] < 0.0 || continue
                @test s3[i] > 0.0
            end
            # The closure itself. Not "to round-off in the kernel's summation order" — the
            # caller subtracts what the donors lost, so this is zero to the last bit of the
            # three multiplications that carry it.
            @test maximum(abs, s1 .+ s2 .+ s3) <= 1.0e-14 * scale
        end
    end


    @testset "rain evaporation joins the rain donor's realization (Stage 2b)" begin
        # TeX §donor_relax, "The rain reservoir has a third sink that lived outside its
        # conductance". Rain evaporation is integrated on the `Q_ss` relaxation pair, whose
        # propagator bounds the SUPERSATURATION and not the rain: the vapor deficit sets the
        # step-mean and nothing set it by the rain that is there. The ice legs were realized
        # on a conductance that did not contain it, so two draws on ONE reservoir summed
        # without either knowing of the other. The cure is the one sublimation already had:
        # `kappa_ev = max(-Qdot_r, 0)/rho_r` joins the rain donor's total, one factor is formed
        # over all of it, and the realized `f_r/tau_r` replaces `1/tau_r` in lambda, in N and
        # in the rain transfer at once.

        # ── (1) THE FACTOR ITSELF, at the function ────────────────────────────────────────
        # `kev = 0` is the pre-Stage-2b construction BITWISE — which is the whole content of
        # `options[:rain_evap_realization] = false` — and a positive `kev` shrinks BOTH rain
        # moments' factors while leaving the cloud's alone.
        let hf = (mim = 1.0e-4, mimr = 2.0e-4, nimr = 3.0),
            bg = (mbiggr = 5.0e-4, nbiggr = 7.0),
            pz = Scythe._ice_empty_pre(),
            qc = 1.0e-3, qr = 5.0e-4, nr = 100.0, dt = 1.0

            f0 = Scythe._ice_donor_factors(hf, bg, pz, pz, pz, qc, qr, nr, 0.0, dt)
            @test f0[1] === Scythe.relaxation_realization(hf.mim, qc, dt)
            @test f0[2] === Scythe.relaxation_realization(hf.mimr + bg.mbiggr, qr, dt)
            @test f0[3] === Scythe.relaxation_realization(hf.nimr + bg.nbiggr, nr, dt)

            kev = 0.3
            f1 = Scythe._ice_donor_factors(hf, bg, pz, pz, pz, qc, qr, nr, kev, dt)
            @test f1[1] === f0[1]                 # the CLOUD channel is not touched
            @test f1[2] < f0[2]                   # the rain MASS donor now sees the third sink
            @test f1[3] < f0[3]                   # ...and so does the rain NUMBER donor
            # BOUNDED BY CONSTRUCTION: the combined realized draw of every sink on the
            # reservoir, evaporation included, is `1 - exp(-kappa_tot dt) < 1`.
            @test ((hf.mimr + bg.mbiggr) + kev * qr) * f1[2] * dt / qr <= 1.0
            @test ((hf.nimr + bg.nbiggr) + kev * nr) * f1[3] * dt / nr <= 1.0

            # With evaporation the ONLY sink the factor is `J_0(kappa_ev dt)`, and it is the
            # SAME number for mass and number: `rain_number_evaporation_2m` is
            # `Qdot_r n_r / max(rho_r, RHO_R_MIN)`, i.e. the identical conductance acting on
            # the number, unlike Bigg (whose number conductance is a twentieth of its mass
            # one) which is why the two moments need separate factors at all.
            hz = (mim = 0.0, mimr = 0.0, nimr = 0.0)
            bz = (mbiggr = 0.0, nbiggr = 0.0)
            f2 = Scythe._ice_donor_factors(hz, bz, pz, pz, pz, qc, qr, nr, kev, dt)
            @test f2[1] === 1.0
            @test f2[2] ≈ -expm1(-kev * dt) / (kev * dt) rtol = 1e-14
            @test f2[3] ≈ f2[2] rtol = 1e-14
            # ...and a dead reservoir keeps an exact 1.0, whatever the conductance.
            @test Scythe._ice_donor_factors(hz, bz, pz, pz, pz, 0.0, 0.0, 0.0,
                                            kev, dt) === (1.0, 1.0, 1.0)
        end

        # The evaporation number sink is EXACTLY proportional to the mass sink, which is what
        # licenses one conductance for two moments (`_ice_donor_factors`, the note beside
        # `ev_n`).
        for (Qr, rr, nn) in ((-3.0e-6, 1.0e-3, 5.0e2), (-1.0e-5, 2.0e-4, 1.0e3))
            @test Scythe.rain_number_evaporation_2m(Qr, rr, nn) ≈
                  (Qr / rr) * nn rtol = 1e-14
        end

        """One step of a warm rain shaft evaporating into the dry adiabat."""
        function evap_run(tmpdir, ts; evapreal = true, nsteps = 1)
            eo = Dict{Symbol,Any}(:rain_evap_realization => evapreal)
            m, patch, model, col = make_mc_mtile(tmpdir; dry = true, kDim = 32,
                                                 num_cells = 8, ts = ts,
                                                 precipitation = true, extra_options = eo)
            gp = Scythe.getGridpoints(patch)
            seed_rain_bump!(patch, gp, col, model.grid_params.kDim)
            ncols = div(size(patch.physical, 1), model.grid_params.kDim)
            for t in 1:nsteps
                for c in 1:ncols; Scythe.advance_column(m, c, t); end
                t < nsteps && (Scythe.calcTendency(m); gridTransform!(patch))
            end
            return m, patch, model
        end

        # ── (2) THE DRY PATH IS BITWISE ───────────────────────────────────────────────────
        # No rain, no conductance, no factor: `kappa_ev` is an exact 0.0 and `f_rain` an exact
        # 1.0 everywhere, so the fold into `invtau_r`/`Qdot_r` is `1.0 * x === x` and slot 7
        # still takes the classical Euler branch it took before the stage.
        mktempdir() do tmpdir
            m, patch, model, _ = make_mc_mtile(tmpdir; dry = true)
            npts = size(patch.physical, 1)
            ncols = div(npts, model.grid_params.kDim)
            for c in 1:ncols
                Scythe.advance_column(m, c, 1)
            end
            S = m.mc_scratch[Threads.threadid()]
            @test all(S.kappa_ev .=== 0.0)
            @test all(S.f_rain .=== 1.0)
            @test all(S.Qdot_r .=== 0.0)
            @test all(S.etd_lam .== 0.0)
            for i in 1:npts
                @test m.var_np1[i, 7] ===
                      patch.physical[i, 7, 1] + (model.ts * m.expdot_n[i, 7])
            end
        end

        # ── (3) THE KNOB OFF IS THE PREVIOUS BEHAVIOUR, BITWISE ───────────────────────────
        # `options[:rain_evap_realization] = false` forces `kappa_ev = 0` and NOTHING else.
        # Every consequence then collapses to an identity: the rain donor's factor is an
        # exact 1.0, so the fold of `f_rain` into `invtau_r`/`Qdot_r` is a multiplication by
        # one (`1.0 * x === x` for every finite `x`, `-0.0` included), and
        # `_ice_donor_factors` adds an exact 0.0 to a sum of non-negative rates. That is the
        # bitwise-restoration claim, checked here as the identity it rests on; the
        # end-to-end comparison against the pre-stage tree was run offline on this same
        # fixture and reproduced `var_np1` exactly.
        mktempdir() do tmpdir
            m, patch, model = evap_run(tmpdir, 0.05; evapreal = false, nsteps = 3)
            S = m.mc_scratch[Threads.threadid()]
            @test all(S.kappa_ev .=== 0.0)
            @test all(S.f_rain .=== 1.0)
            @test all(x -> 1.0 * x === x, S.invtau_r)
            @test all(x -> 1.0 * x === x, S.Qdot_r)
            # ...and this column really does evaporate, or the claim is about nothing.
            @test minimum(S.Qdot_r) < 0.0
        end

        # ── (4) THE WARM ANSWER MOVES, BOUNDED AND CONVERGENT IN ts ──────────────────────
        # "The factor is not ice-gated: `kappa_ev` is a warm-path quantity, so `f_r < 1`
        # wherever rain evaporates, and the warm answer moves by the amount `J_0` differs
        # from one at the stiffest evaporating points" (TeX). `J_0 -> 1` as `dt -> 0`, so the
        # move is an integrator coefficient and not a rate law: it must shrink with the step,
        # at first order, exactly as the freezing realization's does.
        #
        # Measured on this fixture, one step, `d` the relative change in the slot increment:
        #
        #   ts        d(rho_r)   d(Q_ss)    d(qr_bar)   1 - f_min   kappa_ev*ts
        #   0.05      2.47e-4    1.10e-3    7.96e-4     0.105       0.227
        #   0.005     2.64e-5    1.18e-4    8.51e-5     0.0113      0.0227
        #   0.0005    2.66e-6    1.19e-5    8.56e-6     0.00113     0.00227
        #
        # — a per-mille-level move at the production-sized step, shrinking by ten for ten,
        # which is `1 - J_0 ~ kappa_ev*dt/2` and nothing else. (This shaft is far stiffer
        # than the O01 production configuration, where the TeX records the rain channel's
        # `dt/tau_r` at no more than 0.04.)
        mktempdir() do tmpdir
            reldiff(x, y) = maximum(abs.(x .- y)) / max(maximum(abs.(x)), 1e-300)
            function shift_at(ts)
                # OFF first, then ON, so the per-thread scratch left behind belongs to the
                # realized run and `f_rain` below is the factor that actually acted.
                m_of, _, _ = evap_run(tmpdir, ts; evapreal = false)
                m_on, _, _ = evap_run(tmpdir, ts; evapreal = true)
                S_on = m_on.mc_scratch[Threads.threadid()]
                # The two runs' own slot INCREMENTS — what the rain and the supersaturation
                # actually received over the step — rather than the states, whose difference
                # would be swamped by the base profile.
                return (d8 = reldiff(m_of.var_np1[:, 8] .- m_of.tile.physical[:, 8, 1],
                                     m_on.var_np1[:, 8] .- m_on.tile.physical[:, 8, 1]),
                        d7 = reldiff(m_of.var_np1[:, 7] .- m_of.tile.physical[:, 7, 1],
                                     m_on.var_np1[:, 7] .- m_on.tile.physical[:, 7, 1]),
                        fmin = minimum(S_on.f_rain))
            end
            coarse = shift_at(0.05)
            mid    = shift_at(0.005)
            fine   = shift_at(0.0005)
            # IT MOVES: the factor is strictly below one somewhere, and both the rain
            # increment and the supersaturation increment differ.
            @test coarse.d8 > 0.0
            @test coarse.d7 > 0.0
            @test coarse.fmin < 1.0                # `f_r < 1` wherever rain evaporates
            @test coarse.fmin > 0.0
            @test fine.fmin > coarse.fmin          # ...and `J_0 -> 1` as the step is refined
            # BOUNDED: `1 - J_0` is at most `kappa_ev dt / 2`, which at this configuration is
            # a per-cent-level effect and nowhere near a reservoir.
            @test coarse.d8 < 0.05
            @test coarse.d7 < 0.05
            # CONVERGENT: refining the step by ten shrinks the disagreement by at least five,
            # which a `dt` written into a rate law would not do.
            @test mid.d8 <= 0.2 * coarse.d8 + 1.0e-14
            @test fine.d8 <= 0.2 * mid.d8 + 1.0e-14
            @test mid.d7 <= 0.2 * coarse.d7 + 1.0e-14
            @test fine.d7 <= 0.2 * mid.d7 + 1.0e-14
        end

        # ── (5) THE INVARIANT THE STAGE EXISTS FOR ───────────────────────────────────────
        # On the Stage 0a attribution fixture the two draws on the rain used to sum to 1.0012
        # reservoirs per step over 1296 gridpoint-steps while `MC_DONOR_QR`, which saw the ice
        # half alone, reported a dutiful 1.0. Both readings must now be under one: the census
        # row because it measures the COMBINED draw, and the independent block-B reading —
        # taken at the APPLIED step-mean rather than at the frozen rate the factor was formed
        # on — because the two draws are shares of one exponential depletion.
        mktempdir() do tmpdir
            m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 250.15, rho_i = 1.0e-3,
                                              n_i = 1.0e6, rho_r = 1.0e-4, n_r = 1.0e3,
                                              ts = 1.0,
                                              extra_options = Dict{Symbol,Any}(
                                                  :ice_attr_census => true))
            ncols = div(size(m.tile.physical, 1), mod.grid_params.kDim)
            m.mc_water_stats .= 0.0
            for t in 1:6, c in 1:ncols
                Scythe.advance_column(m, c, t)
            end
            st = m.mc_water_stats
            drow(ch) = Scythe.MC_DONOR_FIRST + Scythe.MC_DONOR_N * (ch - 1)
            arow(ch) = Scythe.MC_ATTR_FIRST + Scythe.MC_ATTR_N * (ch - 1)
            # ACTIVE: the rain donor is genuinely being drawn on here, and evaporation is
            # part of the draw — otherwise the bounds below are vacuous.
            @test maximum(view(st, drow(Scythe.MC_DONOR_QR), :)) > 0.5
            @test maximum(view(st, arow(Scythe.MC_ATTR_QR_EVAP), :)) > 0.0
            # ...and BOUNDED, which is the whole of Stage 2b.
            @test maximum(view(st, drow(Scythe.MC_DONOR_QR), :)) <= 1.0 + 1.0e-9
            @test maximum(view(st, arow(Scythe.MC_ATTR_QR_COMB), :)) <= 1.0 + 1.0e-9
            # The number donor is under the same bound with the number sink in ITS total.
            @test maximum(view(st, drow(Scythe.MC_DONOR_NR), :)) <= 1.0 + 1.0e-9
        end
    end

    @testset "ice: the ice NUMBER is a donor reservoir of its own" begin
        # STAGE 1b. A species' NUMBER and its MASS are two reservoirs, and the three legs
        # that draw on the number are not proportional to the three that draw on the mass:
        # aggregation's `deltan` comes from `coltabn` and its `colamt` from `coltab` (two
        # offline tables, two moments of one kernel), the melt number carries `dNmltri` — the
        # collected drops, a number sink with NO mass partner — and only sublimation's two
        # conductances agree, exactly (`niq = n/q`). Until this stage all three number legs
        # rode the MASS factor, on the claim that a pair's `(colamt, deltan)` is one collision
        # count. It is; that does not make the two RESERVOIR FRACTIONS equal.
        #
        # Four claims: (1) the kernel takes the two moments' factors separately and the
        # defaults are the old single-factor form bitwise; (2) the option is OFF by default
        # and absent-vs-false is bitwise; (3) the new census rows read the number draw in both
        # modes; (4) on a state where the number leg genuinely over-draws, the switch is what
        # puts it back under one reservoir per step.
        drow(ch) = Scythe.MC_DONOR_FIRST + Scythe.MC_DONOR_N * (ch - 1)

        # ── (1) THE KERNEL HOOK ─────────────────────────────────────────────────────────
        # `ishmael_aggregation` now scales `colamt` by `f_agg<k>` and `deltan` by
        # `f_aggn<k>`. Mass and number must move independently, and the defaults
        # (`f_aggn<k> = f_agg<k>`, `f_aggn3 = 1`) must reproduce the shipped behaviour to the
        # bit — that is what makes the whole stage opt-in.
        let
            tab = Scythe.load_ishmael_tables_or_error()
            rhoair = 0.95; dt = 1.0; temp = 259.15
            q = 1.0e-3 / rhoair; n = 1.0e6 / rhoair; a = 1.0e6 * (30.0e-6)^3 / rhoair
            e1 = Scythe._ice_effective(q, n, a, a, 1)
            e2 = Scythe._ice_effective(q, n, a, a, 2)
            e3 = Scythe._ice_effective(0.0, 0.0, 0.0, 0.0, 3)
            d1 = clamp(2.0 * ((e1.ai^2) / (e1.ci * e1.ni))^0.333333333333, 1.0e-6, 1.0e-2)
            d2 = clamp(2.0 * ((e2.ci^2) / (e2.ai * e2.ni))^0.333333333333, 1.0e-6, 1.0e-2)
            phi1 = clamp(e1.ci / e1.ai * Scythe.gamma(Scythe.ISHMAEL_NU - 1.0 + e1.deltastr) *
                         Scythe.ISHMAEL_I_GAMMNU, 0.01, 100.0)
            phi2 = clamp(e2.ci / e2.ai * Scythe.gamma(Scythe.ISHMAEL_NU - 1.0 + e2.deltastr) *
                         Scythe.ISHMAEL_I_GAMMNU, 0.01, 100.0)
            agg(; kw...) = Scythe.ishmael_aggregation(dt, rhoair, temp,
                                q, e1.ni, d1, q, e2.ni, d2, 0.0, 0.0, 1.0e-4,
                                e1.rhobar, e2.rhobar, phi1, phi2,
                                tab.coltab, tab.coltabn; kw...)
            base = agg()
            # ACTIVE: this state aggregates in both moments, or nothing below is a statement.
            @test base.qagg1 < 0.0
            @test base.nagg1 < 0.0
            # THE MEASUREMENT the stage rests on: the two conductances are different numbers.
            # `kappa_n/kappa_q` here is ~0.46 — the number kernel is the SLOWER one at this
            # state, so the mass factor OVER-realizes the number transfer by a factor of two.
            # It is the INEQUALITY that matters, not its direction: two reservoirs, two
            # conductances, and neither bounds the other.
            kq = -base.qagg1 / dt / q
            kn = -base.nagg1 / dt / n
            @test !(kq ≈ kn)
            @test 0.1 < kn / kq < 0.9
            # The defaults ride the mass factor, bitwise — the pre-Stage-1b hook.
            @test agg(f_agg1 = 0.4, f_agg2 = 0.7) ==
                  agg(f_agg1 = 0.4, f_agg2 = 0.7, f_aggn1 = 0.4, f_aggn2 = 0.7,
                      f_aggn3 = 1.0)
            # ...and the two moments then separate: halving ONLY the number factor halves
            # `nagg1` and leaves `qagg1` untouched to the bit.
            half = agg(f_aggn1 = 0.5)
            @test half.qagg1 == base.qagg1
            @test half.nagg1 ≈ 0.5 * base.nagg1 rtol=1.0e-14
            # ...and the mass factor alone moves the mass and not the number.
            hq = agg(f_agg1 = 0.5, f_aggn1 = 1.0)
            @test hq.qagg1 ≈ 0.5 * base.qagg1 rtol=1.0e-14
            @test hq.nagg1 == base.nagg1
            # Unit factors are the Fortran, to the bit.
            @test agg(f_agg1 = 1.0, f_agg2 = 1.0, f_aggn1 = 1.0, f_aggn2 = 1.0,
                      f_aggn3 = 1.0) == base
        end

        # ── (2) DEFAULT OFF, AND BITWISE ────────────────────────────────────────────────
        # The option absent and the option explicitly `false` must be the same run, and both
        # must be the run that existed before the number factors did. The second half of that
        # cannot be asserted from inside the suite, so what is asserted is the part that can
        # be: the switch is inert until it is thrown, and throwing it changes the answer.
        mktempdir() do tmpdir
            function run_cold(opts)
                m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 259.15,
                                                  rho_i = 1.0e-3, n_i = 1.0e6,
                                                  rho_i2 = 1.0e-3, n_i2 = 1.0e6, ts = 1.0,
                                                  extra_options = opts)
                ncols = div(size(m.tile.physical, 1), mod.grid_params.kDim)
                m.mc_water_stats .= 0.0
                for t in 1:4, c in 1:ncols
                    Scythe.advance_column(m, c, t)
                end
                return m
            end
            m_absent = run_cold(Dict{Symbol,Any}())
            m_false  = run_cold(Dict{Symbol,Any}(:ice_number_realization => false))
            m_true   = run_cold(Dict{Symbol,Any}(:ice_number_realization => true))
            @test m_absent.var_np1 == m_false.var_np1
            @test m_absent.expdot_n == m_false.expdot_n
            # ...and it is not a no-op switch: with it on the number legs carry a different
            # factor, so the answer moves.
            @test m_absent.var_np1 != m_true.var_np1
            # The residual sublimation-number factor column is an exact 1.0 with the switch
            # off — that is what keeps the ETD site bitwise — and live with it on.
            S_off = m_absent.mc_scratch[Threads.threadid()]
            @test all(S_off.f_isn1 .== 1.0)
            @test all(S_off.f_isn2 .== 1.0)
            @test all(S_off.f_isn3 .== 1.0)
            S_on = m_true.mc_scratch[Threads.threadid()]
            @test any(S_on.f_isn1 .!= 1.0)
            @test all(S_on.f_isn1 .> 0.0)
            # It is a RATIO, `f_n/f_i`, not a factor, and it is ABOVE ONE on this fixture
            # (measured 1.0000004 to 1.00003): below the melting level the number's only
            # legs are aggregation — whose conductance here is the SMALLER one, κ_n/κ_q =
            # 0.46 — and sublimation, whose two conductances are equal, so κ_n < κ_q, the
            # mass factor UNDER-realizes the number, and the residual has to make that up.
            # The invariant is not on the ratio; it is on what the leg ends up carrying,
            # which is the product, and that is `J₀(κ_{n,k}Δt) ≤ 1` by construction.
            @test all(S_on.f_isn1 .* S_on.f_ice1 .<= 1.0)
            @test all(S_on.f_isn2 .* S_on.f_ice2 .<= 1.0)
            @test maximum(S_on.f_isn1) > 1.0

            # ── (3) THE CENSUS READS THE NUMBER DRAW, IN BOTH MODES ──────────────────────
            # On the Stage-1 stiff-cold fixture the number row is the LARGE one: measured
            # 0.0694 reservoirs per step against 1.19e-4 for the melt+aggregation mass row
            # and 0.0605 for the sublimation mass row. Nothing here breaches — the point is
            # that the number is being drawn on at a rate the mass rows do not report, which
            # is what "the ice number reservoirs are not censused at all" meant.
            for m in (m_absent, m_true)
                st = m.mc_water_stats
                n1 = maximum(view(st, drow(Scythe.MC_DONOR_N1), :))
                n2 = maximum(view(st, drow(Scythe.MC_DONOR_N2), :))
                @test n1 > 0.0
                @test n2 > 0.0
                @test n1 <= 1.0 + 1.0e-9
                # An order of magnitude above BOTH mass rows for the same species.
                @test n1 > 10.0 * maximum(view(st, drow(Scythe.MC_DONOR_I1), :))
                @test n1 > maximum(view(st, drow(Scythe.MC_DONOR_S1), :))
                # Species 3 is empty on this fixture, so its row must be an exact zero.
                @test maximum(view(st, drow(Scythe.MC_DONOR_N3), :)) == 0.0
            end
        end

        # The SMALL-CRYSTAL variant: same mass, a thousand times the number. The number row
        # goes to 0.9986 reservoirs per step — the whole population, every step — while the
        # melt+aggregation mass row is 7.9e-4. Bounded in both modes here (the draw is
        # dominated by sublimation, whose two conductances are equal by construction), and
        # this is the state that says WHICH reservoir is the binding one.
        mktempdir() do tmpdir
            m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 259.15,
                                              rho_i = 1.0e-3, n_i = 1.0e9,
                                              rho_i2 = 1.0e-3, n_i2 = 1.0e9, ts = 1.0)
            ncols = div(size(m.tile.physical, 1), mod.grid_params.kDim)
            m.mc_water_stats .= 0.0
            for t in 1:4, c in 1:ncols
                Scythe.advance_column(m, c, t)
            end
            st = m.mc_water_stats
            @test maximum(view(st, drow(Scythe.MC_DONOR_N1), :)) > 0.9
            @test maximum(view(st, drow(Scythe.MC_DONOR_N1), :)) <= 1.0 + 1.0e-9
            @test maximum(view(st, drow(Scythe.MC_DONOR_I1), :)) < 0.01
        end

        # ── (4) THE BREACH, AND WHAT THE SWITCH DOES TO IT ──────────────────────────────
        # Above the melting level with rain present, `nmlt = q̇_mlt·(n/q) − dNmltri` and the
        # second term has no mass partner: the number conductance is then STRICTLY larger
        # than the mass one, the mass factor under-realizes it, and the realized number draw
        # runs past the reservoir. Measured on this fixture with the switch off: 1.0000091
        # reservoirs per step over 672 gridpoint-steps, while the MASS row sits at the
        # saturated `1 + 1 ULP`. With the switch on the row is `1 − exp(−κ_n Δt)`, which
        # rounds to that same one-ULP saturation and nowhere above it.
        mktempdir() do tmpdir
            function run_melt(opts)
                m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 278.15,
                                                  rho_i = 1.0e-3, n_i = 1.0e9,
                                                  rho_i2 = 1.0e-3, n_i2 = 1.0e9,
                                                  rho_r = 1.0e-4, n_r = 1.0e3, ts = 1.0,
                                                  extra_options = opts)
                ncols = div(size(m.tile.physical, 1), mod.grid_params.kDim)
                m.mc_water_stats .= 0.0
                for t in 1:4, c in 1:ncols
                    Scythe.advance_column(m, c, t)
                end
                return m.mc_water_stats
            end
            st_off = run_melt(Dict{Symbol,Any}())
            st_on  = run_melt(Dict{Symbol,Any}(:ice_number_realization => true))
            off_max = maximum(view(st_off, drow(Scythe.MC_DONOR_N1), :))
            on_max  = maximum(view(st_on,  drow(Scythe.MC_DONOR_N1), :))
            off_n = sum(view(st_off, drow(Scythe.MC_DONOR_N1) + 1, :))
            on_n  = sum(view(st_on,  drow(Scythe.MC_DONOR_N1) + 1, :))
            # IT BREACHES with the switch off — by more than the one-ULP saturation the mass
            # row shows, which is what makes it a real over-draw rather than rounding.
            @test off_max > 1.0 + 1.0e-6
            @test off_n > 100.0
            # ...and the switch bounds it. `1 + 1e-9` is the same tolerance the rain donor's
            # test uses, for the same reason: `1 − exp(−κΔt)` rounds to 1.0 at large `κΔt`
            # and the division in the census lands one ULP over.
            @test on_max <= 1.0 + 1.0e-9
            @test on_n < off_n
            # The MASS rows are untouched by the number factors: they are formed on, and
            # realized at, the species' mass conductance in both modes.
            @test maximum(view(st_off, drow(Scythe.MC_DONOR_I1), :)) ==
                  maximum(view(st_on,  drow(Scythe.MC_DONOR_I1), :))
        end
    end

    @testset "ice: anchor reconciliation — inert when admissible, proportional when not" begin
        # Stage C (TeX §Reconciliation of the condensate partition). Three driver-level
        # claims: (1) on an ADMISSIBLE mixed-phase column the device is bitwise absent —
        # every tendency identical with the source on or off, and the census silent;
        # (2) on a DETACHED column (ice beyond the anchor headroom) it removes with ONE
        # shared factor across a species' four moments and touches no other slot's
        # tendency assembly; (3) the census records the defect whether or not the source
        # is applied — the off switch is a forensic tool, not a blindfold.
        mktempdir() do tmpdir
            on = Dict{Symbol,Any}()                                   # default: source on
            off = Dict{Symbol,Any}(:ice_anchor_source => false)
            function run_ice(opts; rho_i)
                m, patch, mod, _ = make_ice_mtile(tmpdir; rho_i = rho_i,
                                                  extra_options = opts)
                Scythe.advance_column(m, 1, 2)
                return m
            end

            # (1) admissible: the default seed sits far inside the headroom.
            m_on = run_ice(on; rho_i = 1.0e-5)
            m_off = run_ice(off; rho_i = 1.0e-5)
            @test m_on.expdot_n == m_off.expdot_n
            @test m_on.var_np1 == m_off.var_np1
            st = m_on.mc_water_stats
            @test all(view(st, Scythe.MC_ANCHOR_GAP, :) .== 0.0)
            @test all(view(st, Scythe.MC_ANCHOR_PTS, :) .== 0.0)
            @test all(view(st, Scythe.MC_ANCHOR_REMOVED, :) .== 0.0)

            # (2) detached: ice at ~2x the anchor headroom (the headroom here is the
            # reference vapor plus cloud, ~2.4e-3 kg/m^3).
            m_on = run_ice(on; rho_i = 5.0e-3)
            m_off = run_ice(off; rho_i = 5.0e-3)
            tid = Threads.threadid()
            S_on = m_on.mc_scratch[tid]
            S_off = m_off.mc_scratch[tid]
            dq = S_off.SRC_i1q .- S_on.SRC_i1q          # what the device removed: phi*max(m,0)
            dn = S_off.SRC_i1n .- S_on.SRC_i1n
            da = S_off.SRC_i1a .- S_on.SRC_i1a
            dc = S_off.SRC_i1c .- S_on.SRC_i1c
            @test maximum(dq) > 0.0                     # it removes, somewhere
            @test all(dq .>= 0.0)                       # and ONLY removes
            @test all(dn .>= 0.0)
            for i in eachindex(dq)
                dq[i] > 0.0 || continue
                q = max(S_on.i1q[i], 0.0)
                phi = dq[i] / q
                # one factor, four moments — the proportional-carriage statement
                @test dn[i] ≈ phi * max(S_on.i1n[i], 0.0) rtol = 1e-12
                @test da[i] ≈ phi * max(S_on.i1a[i], 0.0) rtol = 1e-12
                @test dc[i] ≈ phi * max(S_on.i1c[i], 0.0) rtol = 1e-12
                # bounded by dt/tau at the default tau = 10 s
                @test phi <= 1.0 / 10.0 + eps()
            end
            # No other slot's tendency is assembled differently in the same step: the
            # device writes the twelve SRC_i* and nothing else.
            i1q_slot = m_on.mc_slots.i1_q
            for slot in 1:(i1q_slot - 1)
                @test m_on.expdot_n[:, slot] == m_off.expdot_n[:, slot]
            end

            # (3) the census saw the same defect in BOTH runs; only the removal differs.
            st_on = m_on.mc_water_stats
            st_off = m_off.mc_water_stats
            @test maximum(view(st_on, Scythe.MC_ANCHOR_GAP, :)) > 0.0
            @test maximum(view(st_on, Scythe.MC_ANCHOR_GAP, :)) ==
                  maximum(view(st_off, Scythe.MC_ANCHOR_GAP, :))
            @test sum(view(st_on, Scythe.MC_ANCHOR_PTS, :)) ==
                  sum(view(st_off, Scythe.MC_ANCHOR_PTS, :)) > 0.0
            @test sum(view(st_on, Scythe.MC_ANCHOR_REMOVED, :)) > 0.0
            @test sum(view(st_off, Scythe.MC_ANCHOR_REMOVED, :)) == 0.0

            # (4) the READER-SIDE CAP: on the detached column the retrieval reads the
            # anchor-capped partition and the phantom L_s credit is gone; with the cap
            # ablated it reads the full credit. On the admissible column the cap never
            # binds and the two configurations are bitwise identical.
            m_nf = run_ice(Dict{Symbol,Any}(:ice_anchor_floor => false); rho_i = 5.0e-3)
            S_nf = m_nf.mc_scratch[tid]
            @test maximum(S_nf.Tk) - maximum(S_on.Tk) > 5.0
            @test all(S_on.Tk .<= S_nf.Tk .+ 1.0e-12)
            m_nf_ok = run_ice(Dict{Symbol,Any}(:ice_anchor_floor => false); rho_i = 1.0e-5)
            m_ok = run_ice(on; rho_i = 1.0e-5)
            @test m_ok.expdot_n == m_nf_ok.expdot_n
            @test m_ok.var_np1 == m_nf_ok.var_np1

            # (5) the SEDIMENTATION anchor share: on the detached column the flux
            # assemblies transport the anchor-supported part — one factor, all moments,
            # so F ratios across a species' moments are those of the uncapped fluxes —
            # and on the admissible column the factor is exactly 1.0 and the fluxes are
            # bitwise identical.
            m_nx = run_ice(Dict{Symbol,Any}(:ice_anchor_flux => false); rho_i = 5.0e-3)
            S_nx = m_nx.mc_scratch[tid]
            m_x = run_ice(on; rho_i = 5.0e-3)
            S_x = m_x.mc_scratch[tid]
            @test any(S_x.anchor_f .< 1.0)      # it binds somewhere on this column
            @test all(0.0 .<= S_x.anchor_f .<= 1.0)
            for i in eachindex(S_x.anchor_f)
                f = S_x.anchor_f[i]
                @test S_x.F_i1q[i] ≈ f * S_nx.F_i1q[i] rtol = 1e-12
                @test S_x.F_i1n[i] ≈ f * S_nx.F_i1n[i] rtol = 1e-12
            end
            m_nx_ok = run_ice(Dict{Symbol,Any}(:ice_anchor_flux => false); rho_i = 1.0e-5)
            @test m_ok.expdot_n == m_nx_ok.expdot_n
            @test m_ok.var_np1 == m_nx_ok.var_np1

            # (6) the RATE-SIDE share: on the detached column the process rates see the
            # anchor-supported population (deposition on the shared surface differs from
            # the raw-population rates); on the admissible column the factor is 1.0 and
            # the configurations are bitwise identical.
            m_nr = run_ice(Dict{Symbol,Any}(:ice_anchor_rates => false); rho_i = 5.0e-3)
            S_nr = m_nr.mc_scratch[tid]
            @test any(S_x.Qdot_i1 .!= S_nr.Qdot_i1)
            m_nr_ok = run_ice(Dict{Symbol,Any}(:ice_anchor_rates => false); rho_i = 1.0e-5)
            @test m_ok.expdot_n == m_nr_ok.expdot_n
            @test m_ok.var_np1 == m_nr_ok.var_np1
        end
    end

    @testset "ice: Wegener-Bergeron-Findeisen emerges from the shared Q_ss" begin
        # TeX Eq. wbf_qs. A supercooled liquid cloud at -15 C with seeded ice: with the
        # forcing F between the two thresholds, the ice grows while the liquid evaporates to
        # feed it. Nothing arbitrates this -- both phases relax the SAME Q_ss, and the
        # quasi-steady value sits between the water and ice saturations by construction.
        mktempdir() do tmpdir
            m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 258.15, q_l = 1.0e-3,
                                              rho_i = 1.0e-5, n_i = 5.0e4)
            kDim = mod.grid_params.kDim
            ncols = div(size(patch.physical, 1), kDim)
            # Let Q_ss find its quasi-steady value against the two relaxations.
            for t in 1:60
                for c in 1:ncols
                    Scythe.advance_column(m, c, t)
                end
                Scythe.calcTendency(m)
                gridTransform!(patch)
            end
            Scythe.advance_column(m, 1, 61)
            S = m.mc_scratch[Threads.threadid()]

            @test all(S.Tk .< Scythe.T_0 - 10.0)
            @test all(isfinite, S.Qdot_i1)
            # THE SIGNATURE: ice grows, liquid evaporates, at the same gridpoints.
            @test all(S.Qdot_i1 .> 0.0)
            @test all(S.Qdot .< 0.0)
            # ...and the quasi-steady state sits BETWEEN the two saturations, which is the
            # same statement written on the state: subsaturated over water, supersaturated
            # over ice, one reservoir, offset by 𝒟 (Eq. qss_ice_shift).
            #
            # Read off the two DRIVES, not the raw prognostic. Both closures are driven by
            # the clipped drive, and here the prognostic Q_ss has detached upward from the
            # density budget by ~7e-6 kg/m³ — which is exactly the detachment the clip exists
            # to catch, and which the census would report if it mattered.
            for i in eachindex(S.Tk)
                D = Scythe.ice_supersaturation_gap(S.Tk[i], S.p_hPa[i])
                drive_i = Scythe.ice_deposition_drive(S.Q_ss[i], S.rho_v[i], S.Tk[i],
                                                      S.p_hPa[i])
                drive_w = drive_i - D
                @test D > 0.0
                @test drive_w < 0.0          # the liquid is subsaturated and evaporating
                @test drive_i > 0.0          # the ice is supersaturated and growing
            end
            # The two relaxations are both live -- the competition is between conductances,
            # not between an active and an inert channel (Eq. wbf_qs).
            @test all(S.invtau_c .> 0.0)
            @test all(S.invtau_i1 .> 0.0)
        end
    end

    @testset "ice: freezing heats T and p with the bare L_f" begin
        # TeX Eqs. dThat_ice / dphat_ice / thermo_p_ice. `L_f·Q̇_freeze` sits inside BOTH
        # non-condensation tendencies and inside the pressure equation's (R_m/C_vt) bracket,
        # with NO -R_v C_pt T/R_m companion -- that companion is the work done by the vapor a
        # phase change removes from the gas, and freezing removes none.
        mktempdir() do tmpdir
            # Rain at -15 C freezes by Bigg, and rimes onto the seeded ice: Q̇_freeze > 0.
            m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 258.15, q_l = 1.0e-3,
                                              rho_i = 1.0e-5, n_i = 5.0e4,
                                              rho_r = 1.0e-4, n_r = 1.0e3)
            Scythe.advance_column(m, 1, 2)
            S = m.mc_scratch[Threads.threadid()]

            # It fired, and in the freezing direction.
            @test maximum(S.FRZ_NET) > 0.0
            # Q̇_freeze IS minus the liquid mass sink, exactly -- the structural statement
            # that freezing sources neither rho_t nor rho_v.
            @test S.FRZ_NET == -(S.ICE_C .+ S.ICE_R)

            # dT_nc / dp_nc carry exactly L_f·Q̇_freeze over the ice-free expressions.
            baseT = @. ((-S.p * S.div) + S.QDOT_TH) / (S.rho_d * S.C_vt)
            basep = @. (-S.gamma_m * S.p * S.div) + ((S.R_m / S.C_vt) * S.QDOT_TH)
            Lf = Springsteel.Thermodynamics.L_f.(S.Tk)
            @test S.dT_nc .- baseT ≈ (Lf .* S.FRZ_NET) ./ (S.rho_d .* S.C_vt) rtol = 1e-12
            @test S.dp_nc .- basep ≈ (S.R_m ./ S.C_vt) .* (Lf .* S.FRZ_NET) rtol = 1e-12

            # The pressure equation itself, which is now in TWO pieces and the split is the
            # point of the test. FREEZING relaxes nothing and moves no vapor, so it stays on
            # the multistep and is the ONLY phase change left in `expdot[.,1]`; CONDENSATION
            # and DEPOSITION relax the shared `Q_ss`, are withheld with it, and come back as
            # the direct increment `etd_d1` at the step-mean rate. The column is at rest
            # (u = w = 0), so the advective part of slot 1 is identically zero, the divergence
            # work vanishes and the acoustic staging term is proportional to w.
            kDim = mod.grid_params.kDim
            Ls = Springsteel.Thermodynamics.L_s.(S.Tk)
            thermal = @. (S.R_m / S.C_vt) * S.QDOT_TH
            frz = @. (S.R_m / S.C_vt) * (Lf * S.FRZ_NET)
            @test m.expdot_n[1:kDim, 1] ≈ thermal .+ frz rtol = 1e-8 atol = 1e-12
            # The freezing piece carries the BARE L_f — no -R_v C_pt T/R_m companion — and is
            # a pure heating that RAISES the pressure.
            @test all(frz[S.FRZ_NET .> 0.0] .> 0.0)
            # The withheld heating, at the step-mean rate, with the companion that freezing
            # does not get: deposition's coefficient is condensation's with L_v -> L_s.
            ts = mod.ts
            want_d1 = @. ts * (S.R_m / S.C_vt) *
                          (((S.Lv - (Rv * S.C_pt * S.Tk / S.R_m)) *
                            (S.Qdot_bar + S.Qdot_r_bar)) +
                           ((Ls - (Rv * S.C_pt * S.Tk / S.R_m)) * S.Qdep_bar))
            @test S.etd_d1 ≈ want_d1 rtol = 1e-12 atol = 1e-300
        end
    end

    @testset "ice: total energy carries no phase-change source" begin
        # TeX Eq. Et_ice: condensation, deposition and freezing appear NOWHERE in the total
        # energy equation. The test is a perturbation one: change only Q_ss, which no other
        # part of the state reads (the retrieval is independent of it, and `:residual` vapor
        # is the density budget), and every phase-change rate moves while E_t's tendency must
        # not move at all.
        mktempdir() do tmpdir
            function run_with(qss_scale)
                m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 258.15, q_l = 1.0e-3,
                                                  rho_i = 1.0e-5, n_i = 5.0e4,
                                                  rho_r = 1.0e-4, n_r = 1.0e3)
                kDim = mod.grid_params.kDim
                for i in 1:size(patch.physical, 1)
                    patch.physical[i, 7, 1] = qss_scale
                end
                spectralTransform!(patch)
                gridTransform!(patch)
                Scythe.advance_column(m, 1, 2)
                return m, kDim
            end
            m_a, kDim = run_with(+2.0e-4)     # strongly supersaturated: deposit and condense
            m_b, _ = run_with(-2.0e-4)        # strongly subsaturated: sublimate and evaporate
            Sa = m_a.mc_scratch[Threads.threadid()]

            # The rates really did move, and in the right direction: the more supersaturated
            # state condenses more (equivalently, evaporates less) and deposits more. Read off
            # the ADVANCE rather than off `expdot`: the two phase changes are withheld from the
            # multistep and applied directly at the step-mean rate, and the two runs differ
            # only in slot 7 — which nothing else on these slots reads — so the difference in
            # `var_np1` IS the difference in the phase change. (`m_a` and `m_b` are separate
            # tiles with separate scratch, so the step-mean columns are comparable too.)
            i1q = m_a.mc_slots.i1_q                                         # 12, by name
            @test all(m_a.var_np1[1:kDim, 9] .> m_b.var_np1[1:kDim, 9])     # cloud
            @test all(m_a.var_np1[1:kDim, i1q] .> m_b.var_np1[1:kDim, i1q]) # ice mass
            Sb = m_b.mc_scratch[Threads.threadid()]
            @test all(Sa.Qdot_bar .> Sb.Qdot_bar)
            @test all(Sa.Qdep_bar .> Sb.Qdep_bar)

            # ...and E_t's tendency is BITWISE identical, because no phase change is a source
            # of it. (rho_t likewise: phase changes are internal to the water.)
            @test m_a.expdot_n[1:kDim, 6] == m_b.expdot_n[1:kDim, 6]
            @test m_a.expdot_n[1:kDim, 3] == m_b.expdot_n[1:kDim, 3]
        end
    end

    @testset "ice: the mass exchange closes exactly" begin
        # The property Scythe's RESIDUAL vapor makes load-bearing: everything the ice gains
        # that is not deposition must come out of the liquid, or the difference is silently
        # manufactured vapor. This is the check that ISHMAEL's own splinter-mass leak (which
        # the port closes deliberately) has not come back.
        mktempdir() do tmpdir
            m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 265.15, q_l = 2.0e-3,
                                              rho_i = 2.0e-5, n_i = 1.0e5,
                                              rho_r = 3.0e-4, n_r = 2.0e3)
            Scythe.advance_column(m, 1, 2)
            S = m.mc_scratch[Threads.threadid()]
            dep = S.Qdot_i1 .+ S.Qdot_i2 .+ S.Qdot_i3
            ice = S.SRC_i1q .+ S.SRC_i2q .+ S.SRC_i3q
            liq = S.ICE_C .+ S.ICE_R
            scale = maximum(abs.(ice)) + maximum(abs.(dep))
            @test scale > 0.0
            # Riming, splintering, freezing, melting and aggregation are all live here.
            @test maximum(abs.(ice)) > 0.0
            # THE CLOSURE IS NOW TWO STATEMENTS, because the deposition leg no longer travels
            # through `SRC_i*q`: it is the ice half of the `Q_ss` relaxation pair, withheld
            # from the multistep and applied at the step-mean rate. So `ice` here is exactly
            # the NON-deposition ice mass source, and
            #   (i) everything the ice gains that is not deposition comes out of the liquid;
            @test maximum(abs.(ice .+ liq)) < 1.0e-12 * scale
            #   (ii) the vapor pays exactly what the condensation and deposition channels gain,
            #        at the SAME step-mean rate every consumer is fed from — which is what
            #        makes the transfer identical for all of them by construction.
            @test S.VAPOR_SRC == -((S.Qdot_bar .+ S.Qdot_r_bar) .+ S.Qdep_bar)
            @test maximum(abs.(S.Qdep_bar)) > 0.0
        end
    end

    # ══════════════════════════════════════════════════════════════════════════
    # The STIFF RELAXATION INTEGRATOR at the driver level. TeX §"Integration of the
    # relaxation pair in the stiff limit"; the scalar coefficient math is exercised
    # on its own in test_etd_relaxation.jl.
    # ══════════════════════════════════════════════════════════════════════════

    @testset "stiff relaxation: dry air takes the classical multistep, bitwise" begin
        # "wherever λ = 0 — air with no droplets, no rain, and no ice, which is the entire dry
        # path — Eq. etd_ab3 IS the third-order multistep, bitwise, so the scheme change
        # touches nothing outside the condensate-bearing points" (TeX, first property). This
        # is that sentence as a gate: zero λ, zero increments, and a slot-7 advance
        # reconstructed independently and compared with `===`.
        mktempdir() do tmpdir
            mtile, patch, model, _ = make_mc_mtile(tmpdir; dry = true)
            kDim = model.grid_params.kDim
            npts = size(patch.physical, 1)
            ncols = div(npts, kDim)
            for c in 1:ncols
                Scythe.advance_column(mtile, c, 1)
            end
            S = mtile.mc_scratch[Threads.threadid()]
            @test all(S.etd_lam .== 0.0)
            @test all(S.Qdot_bar .== 0.0)
            @test all(S.Qdot_r_bar .== 0.0)
            @test all(S.Qdep_bar .== 0.0)
            for col in (S.etd_d1, S.etd_d8, S.etd_d9, S.etd_dv)
                @test all(col .== 0.0)
            end
            # The Euler branch of the first step, rebuilt from the tendency the integrator
            # was handed. Bitwise, because the adjustment must not touch this slot at all.
            for i in 1:npts
                @test mtile.var_np1[i, 7] ===
                      patch.physical[i, 7, 1] + (model.ts * mtile.expdot_n[i, 7])
            end
        end
    end

    @testset "stiff relaxation: the fixed point IS the quasi-steady state (Eq. wbf_qs)" begin
        # "its fixed point is N/λ = Q_ss^qs of Eq. wbf_qs at every x, not merely
        # asymptotically" (TeX, third property), with `λ = τ^{-1} + τ_i^{-1}` the denominator
        # of Eq. wbf_qs and `N = F − 𝒟/τ_i` its numerator. A big step on a numerous, tiny ice
        # population puts `x = λΔt` far into the stiff regime, where the classical multistep
        # (real-axis limit 0.545) could not have taken a step at all.
        mktempdir() do tmpdir
            m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 258.15, q_l = 1.0e-3,
                                              rho_i = 1.0e-6, n_i = 5.0e9, r_i = 10.0e-6,
                                              rho_r = 1.0e-4, n_r = 1.0e3, ts = 100.0)
            kDim = mod.grid_params.kDim
            # OFF THE CLIP BOUNDARY, deliberately. The reference column is exactly
            # liquid-saturated, so `Q_ss` and `ρ_v − ρ_v*` are the same number to within the
            # retrieval's own round-off, and the vapor-availability clip binds or not by
            # accident. A CLIPPED channel is a constant flux that contributes nothing to λ
            # (TeX), which is precisely NOT the regime under test, so the column is given a
            # large vapor excess — 1e-3 against a saturation density of ~1.6e-3 — which leaves
            # `min(Q_ss, ρ_v − ρ_v*) = Q_ss` by two orders of magnitude on both channels. The
            # liquid then sits at zero drive while the ice sees `Q_ss + 𝒟 > 0`: the
            # Wegener-Bergeron-Findeisen window, with both conductances live in λ.
            rv_i = m.mc_slots.rho_v
            for i in 1:size(patch.physical, 1)
                patch.physical[i, rv_i, 1] = 1.0e-3
            end
            spectralTransform!(patch)
            gridTransform!(patch)
            Scythe.advance_column(m, 1, 1)
            S = m.mc_scratch[Threads.threadid()]
            ts = mod.ts
            Q_ssbar = Springsteel.ref_qss(m.ref_state)[:, 1]

            # λ is the sum of the UNCLIPPED conductances; on this vapor-rich state nothing is
            # clipped, so it is all of them, assembled in the same order the driver does.
            @test all(S.etd_dep_a .== 1.0)
            lam_all = S.invtau_c .+ S.invtau_r .+
                      (S.invtau_i1 .+ S.invtau_i2 .+ S.invtau_i3)
            @test S.etd_lam == lam_all
            x = S.etd_lam .* ts
            @test minimum(x) > 0.545         # every point is past AB3's real-axis limit
            @test maximum(x) > 30.0          # and the worst is deep in the stiff regime

            # At t = 1 the extrapolant is the constant N^n, so the update is exactly
            # Q^{n+1} − N/λ = e^{−x}(Q^n − N/λ): the fixed point is N/λ at EVERY x, and the
            # approach to it is the exact propagator's. `N/λ` here IS Eq. wbf_qs — `N` is the
            # tendency slot 7 carries (F − 𝒟/τ_i, the two relaxations withheld) and `λ` is its
            # denominator.
            qn = patch.physical[1:kDim, 7, 1] .+ Q_ssbar
            qnp1 = m.var_np1[1:kDim, 7] .+ Q_ssbar
            nn = m.expdot_n[1:kDim, 7]
            qqs = nn ./ S.etd_lam
            @test maximum(abs.((qnp1 .- qqs) .- (exp.(-x) .* (qn .- qqs)))) <=
                  1.0e-9 * maximum(abs.(qqs))
            # ...and where the step is genuinely stiff the exponential has closed the gap
            # completely: one step lands on the Korolev-Mazin quasi-steady state regardless of
            # history, which the classical multistep could not have reached at all.
            stiff = findall(x .> 30.0)
            @test !isempty(stiff)
            @test qnp1[stiff] ≈ qqs[stiff] rtol = 1e-6
            # The step-MEAN the consumers are fed approaches the same state, but
            # ALGEBRAICALLY rather than exponentially, and that difference is the physics: a
            # trajectory that starts at Q^n and relaxes onto N/λ inside the step spends O(1/x)
            # of that step away from it, so with the t = 1 constant extrapolant
            #
            #     Q̄ = J0 Q^n + Δt J1 N = (N/λ)(1 − 1/x) + Q^n/x + O(1/x²) .
            #
            # It is the MEAN of the trajectory, not its endpoint, and it is the mean that every
            # consumer must see for Eq. qss_stepmean to close — feeding them the endpoint would
            # over-count the transfer by exactly this 1/x.
            @test S.etd_qbar[stiff] ≈
                  (qqs[stiff] .* (1.0 .- (1.0 ./ x[stiff]))) .+ (qn[stiff] ./ x[stiff]) rtol = 5e-3
            @test S.etd_qbar[stiff] ≈ qqs[stiff] rtol = 5.0 / minimum(x[stiff])
        end
    end

    @testset "stiff relaxation: the step-mean closes the exchange with ice present" begin
        # Every consumer is fed from ONE step-mean rate per channel, so "the vapor removed,
        # the mass deposited, and the heat released are identical by construction" (TeX,
        # "The step-mean closes the exchange"). Checked on the INCREMENTS the adjustment
        # actually applies, in density units — the transformed slots carry their own Jacobian,
        # so the density increment of a slot is its slot increment divided by that Jacobian.
        mktempdir() do tmpdir
            m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 258.15, q_l = 1.0e-3,
                                              rho_i = 1.0e-5, n_i = 5.0e4,
                                              rho_r = 1.0e-4, n_r = 1.0e3)
            Scythe.advance_column(m, 1, 2)
            S = m.mc_scratch[Threads.threadid()]
            ts = mod.ts
            @test maximum(abs.(S.Qdot_bar)) > 0.0
            @test maximum(abs.(S.Qdep_bar)) > 0.0

            gained = (S.etd_d9 ./ S.Jc) .+ (S.etd_d8 ./ S.Jr) .+
                     (S.etd_di1 ./ S.J_i1q) .+ (S.etd_di2 ./ S.J_i2q) .+
                     (S.etd_di3 ./ S.J_i3q)
            lost = S.etd_dv
            scale = maximum(abs.(lost))
            @test scale > 0.0
            # VAPOR REMOVED = Σ CONDENSATE GAINED. Equivalently: the sum of every water
            # increment the adjustment applies is zero, i.e. ρ_t is untouched by it — which
            # is what keeps ρ_t the conservation anchor. (ρ_d is not a water slot at all and
            # the adjustment never names it.)
            @test maximum(abs.(gained .+ lost)) < 1.0e-12 * scale
            # HEAT CONSISTENCY: the slot-1 increment is the same three rates through the
            # pressure equation's own bracket, and nothing else.
            Ls = Springsteel.Thermodynamics.L_s.(S.Tk)
            want = @. ts * (S.R_m / S.C_vt) *
                       (((S.Lv - (Rv * S.C_pt * S.Tk / S.R_m)) *
                         (S.Qdot_bar + S.Qdot_r_bar)) +
                        ((Ls - (Rv * S.C_pt * S.Tk / S.R_m)) * S.Qdep_bar))
            @test S.etd_d1 == want

            # MOMENT/MASS CONSISTENCY of the habit partition: the a/c (and sublimation-n)
            # increments are the partition of the SAME realized mass increment the mass slot
            # received — recomputed here, independently, from the stashed state-n inputs and
            # the applied mass increment itself. Zero mass increment ⇒ zero moment increment.
            habs = ((S.hab1_ani, S.hab1_cni, S.hab1_rni, S.hab1_ds, S.hab1_rb,
                     S.hab1_nim3, S.hab1_vt, S.hab1_cg, S.hab1_niq,
                     S.etd_di1, S.J_i1q, S.etd_da1, S.J_i1a, S.etd_dc1, S.J_i1c,
                     S.etd_dn1, S.J_i1n),
                    (S.hab2_ani, S.hab2_cni, S.hab2_rni, S.hab2_ds, S.hab2_rb,
                     S.hab2_nim3, S.hab2_vt, S.hab2_cg, S.hab2_niq,
                     S.etd_di2, S.J_i2q, S.etd_da2, S.J_i2a, S.etd_dc2, S.J_i2c,
                     S.etd_dn2, S.J_i2n),
                    (S.hab3_ani, S.hab3_cni, S.hab3_rni, S.hab3_ds, S.hab3_rb,
                     S.hab3_nim3, S.hab3_vt, S.hab3_cg, S.hab3_niq,
                     S.etd_di3, S.J_i3q, S.etd_da3, S.J_i3a, S.etd_dc3, S.J_i3c,
                     S.etd_dn3, S.J_i3n))
            any_partition = false
            for (ani, cni, rni, ds, rb, nim3, vt, cg, niq,
                 dq, Jq, da, Ja, dc, Jc_, dn, Jn) in habs
                for i in eachindex(dq)
                    qk = dq[i] / (ts * Jq[i])              # the realized mass rate
                    if dq[i] == 0.0 || cg[i] <= 0.0
                        @test da[i] == 0.0
                        @test dc[i] == 0.0
                        @test dn[i] == 0.0
                        continue
                    end
                    any_partition = true
                    afn = qk / (4.0 * pi * nim3[i] * cg[i])
                    dp = Scythe.ishmael_deposition_partition(ts, ani[i], cni[i], rni[i],
                            ds[i], rb[i], nim3[i], S.hab_igr[i], afn, S.hab_maxsui[i],
                            vt[i], qk < 0.0, cg[i], S.hab_dv[i], S.Tk[i],
                            Scythe.ISHMAEL_AO, Scythe.ISHMAEL_NU, Scythe.ISHMAEL_GAMMNU,
                            Scythe.ISHMAEL_I_GAMMNU, Scythe.ISHMAEL_FOURTHIRDSPI)
                    @test da[i] ≈ ts * Ja[i] * dp.ard rtol = 1e-12
                    @test dc[i] ≈ ts * Jc_[i] * dp.crd rtol = 1e-12
                    if qk < 0.0
                        @test dn[i] ≈ ts * Jn[i] * (qk * niq[i]) rtol = 1e-12
                    else
                        @test dn[i] == 0.0
                    end
                end
            end
            @test any_partition                      # the case actually exercised the path
        end
    end

    @testset "stiff relaxation: a one-step glaciation burst does not ring the moments" begin
        # The measured death of the first ETD ice arm (t = 1031.4 s): the nucleation burst
        # makes the deposition conductance jump from ~1e-6 to ~70 ts/tau IN ONE STEP; with
        # the habit-moment sources on the multistep at the instantaneous rate, AB3 applied
        # +23/12 of the spike and then -16/12 of it, ringing the freshly created a/c moments
        # negative and driving the effective-axis construction to a negative Reynolds number.
        # The partition now rides the step-mean direct increments, so a burst-scale
        # conductance must produce moments that stay finite and non-negative through the
        # steps that follow the spike. The state: a dense, small-crystal population (the
        # post-homogeneous-freezing signature) in supersaturated air at -35 C.
        mktempdir() do tmpdir
            m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = Scythe.T_0 - 35.0,
                                              q_l = 2.0e-3, rho_i = 5.0e-4, n_i = 1.0e8,
                                              rho_r = 0.0, n_r = 0.0)
            IS = m.mc_slots
            for step in 1:3
                Scythe.advance_column(m, 1, step)
                for sl in (IS.i1_q, IS.i1_a, IS.i1_c, IS.i1_n,
                           IS.i2_q, IS.i2_a, IS.i2_c, IS.i2_n,
                           IS.i3_q, IS.i3_a, IS.i3_c, IS.i3_n, 7, IS.rho_v)
                    col = view(m.var_np1, :, sl)
                    @test all(isfinite, col)
                end
                # The moments the partition feeds must not be rung negative by the burst:
                # the increments are one-sided (deposition) and the multistep carries only
                # the bounded non-deposition legs, so any negative here is transport-scale
                # ringing, orders below the burst amplitude. Bound it at a tiny fraction of
                # the species mass actually present.
                for (qs, asl, csl) in ((IS.i1_q, IS.i1_a, IS.i1_c),
                                       (IS.i2_q, IS.i2_a, IS.i2_c),
                                       (IS.i3_q, IS.i3_a, IS.i3_c))
                    qmax = maximum(view(m.var_np1, :, qs))
                    qmax <= 0.0 && continue
                    amin = minimum(view(m.var_np1, :, asl))
                    cmin = minimum(view(m.var_np1, :, csl))
                    ascale = maximum(abs.(view(m.var_np1, :, asl)))
                    if ascale > 0.0
                        @test amin > -1.0e-6 * ascale
                        @test cmin > -1.0e-6 * max(ascale, maximum(abs.(view(m.var_np1, :, csl))))
                    end
                end
            end
        end
    end

    @testset "ice: one realization factor per donor reservoir, and it bounds the donor" begin
        # THE ACCEPTANCE PROPERTY of the per-donor construction. Three statements, in the
        # order they have to hold.

        # 1. `J₀` is the coefficient it claims to be: exactly 1 with no sink, strictly
        #    bounding at any conductance, and converging to 1 as the step is refined.
        @test Scythe.relaxation_realization(0.0, 1.0e-3, 0.3) === 1.0
        # The ZERO-RESERVOIR guard, and it is DELIBERATE: a factor answers "how much of the
        # reservoir may this sink take", and with nothing there the answer is vacuous, so an
        # exact 1.0 is what keeps the conversion-free path bitwise. It is also the reason a
        # number bound may never be built out of a factor: at `n_r = 0` this returns 1.0 for
        # the number leg while the mass leg's factor clamps to 1e-19, which is the seventeen-
        # order gap FINDINGS §5i measured inside Bigg. What protects that case is the
        # MINIMUM-CRYSTAL BOUND at the slot assembly (`nbig ≤ mbig/ISHMAEL_M_MIN`), tested
        # below in "every realized ice-number source is bounded by its mass partner" — not
        # this function, which is behaving exactly as specified here.
        @test Scythe.relaxation_realization(1.0, 0.0, 0.3) === 1.0
        for (rate, res) in ((1.0e-9, 1.0e-3), (1.0e-4, 1.0e-4), (48.4, 1.88e-5),
                            (2.9e9, 2.84e-7), (1.2e7, 6.0e-11))
            f = Scythe.relaxation_realization(rate, res, 0.3)
            # The converted mass never exceeds the reservoir, at any conductance. Two
            # floating-point caveats, and both are why the bound is stated as `1 + eps`:
            # the mathematical statement is strict (`1 − e^{−x} < 1`) but SATURATES at exactly
            # 1.0 in Float64 once x ≳ 37 — the whole reservoir converts, which is right — and
            # re-forming `rate·f·Δt/q` redoes the same divide and multiply that went into `f`,
            # so the round trip can land a couple of ulp above. The census uses the same
            # `1 + 1e-9` for the same reason.
            @test 0.0 < rate * f * 0.3 / res <= 1.0 + 1.0e-9
            # ...and it IS the exact exponential depletion, not a clamp
            kdt = rate / res * 0.3
            @test rate * f * 0.3 / res ≈ -expm1(-kdt) rtol = 1e-12
        end
        # Δt → 0 recovers the bare rate: an integrator coefficient, not a rate law.
        for dt in (1.0e-2, 1.0e-4, 1.0e-6)
            # κ = 1 here, so J₀(dt) = 1 − dt/2 + dt²/6 − …: first order in the step, and the
            # residual shrinks WITH the step, which is what "converges to the same equation"
            # means. Asserted as such rather than against a fixed tolerance.
            f = Scythe.relaxation_realization(1.0e-4, 1.0e-4, dt)
            @test abs(f - (1.0 - 0.5 * dt)) < dt * dt
            @test f < 1.0
        end
        @test Scythe.relaxation_realization(1.0e-4, 1.0e-4, 1.0e-6) >
              Scythe.relaxation_realization(1.0e-4, 1.0e-4, 1.0e-2)

        # 2. THE SHARE RULE. Several sinks on one donor take one exponential depletion between
        #    them, in proportion to their rates — so the SUM is bounded, which is what
        #    per-leg factors could not do (measured: realized rain depletion pinned at 3.0).
        rates = (0.7, 0.2, 5.0, 0.05)
        q = 1.0e-5
        ktot = sum(rates) / q
        f = Scythe.relaxation_realization(sum(rates), q, 0.3)
        @test sum(r * f for r in rates) * 0.3 / q ≈ -expm1(-ktot * 0.3) rtol = 1e-12
        @test sum(r * f for r in rates) * 0.3 / q <= 1.0 + 1.0e-9
        # each sink keeps its share of the total
        for r in rates
            @test (r * f) / (sum(rates) * f) ≈ r / sum(rates) rtol = 1e-14
        end

        # 3. THE TWO-PASS RIMING ARGUMENT. `prdr ∝ rnfr³ − rni³`, so bounding the RADIUS
        #    increment against the linear-limit conductance leaks through the cubic (measured
        #    2.0 reservoirs at p99, 13.0 at max). Forming κ from the pass-one NONLINEAR
        #    `prdr₀` puts the second pass in the linear-or-below regime, where the response is
        #    sublinear in the scale factor and therefore `prdr(f) ≤ f·prdr₀` — which is what
        #    makes the realized riming mass bounded by its share of the reservoir.
        base = (dt = 0.3, rni = 2.0e-5, deltastr = 1.0, rbdum = 400.0, nidum = 5.0e4,
                ani = 2.0e-5, cni = 2.0e-5, temp = 258.15)
        rim = (qc = 1.0e-3, nc = 2.0e8, nrm = 3.0e-9, nrd = 3.0e-10,
               qr = 5.0e-5, nr = 1.0e3)
        growth(fc, fr) = Scythe.ishmael_riming_growth(
            base.dt, base.rni, base.deltastr, base.rbdum, base.nidum, base.ani, base.cni,
            base.temp, rim.qc, rim.nc, rim.nrm, rim.nrd, 5.0e-4,
            rim.qr, rim.nr, rim.nrm, rim.nrd, 5.0e-4, 1.0, true,
            Scythe.ISHMAEL_NU, Scythe.ISHMAEL_AO, Scythe.ISHMAEL_GAMMNU,
            Scythe.ISHMAEL_I_GAMMNU, Scythe.ISHMAEL_FOURTHIRDSPI;
            f_rime_c = fc, f_rime_r = fr)

        prdr0 = growth(1.0, 1.0).prdr
        @test prdr0 > 0.0
        # sublinearity of the realized mass in the scale factor, across four decades
        for f in (0.5, 0.1, 1.0e-2, 1.0e-3, 1.0e-4)
            @test growth(f, f).prdr <= f * prdr0 * (1.0 + 1.0e-9)
        end
        # ...hence the bound: with κ formed from prdr₀ the realized mass is under the share
        qdon = 1.0e-6                                   # a donor far too small for prdr0*dt
        fq = Scythe.relaxation_realization(prdr0, qdon, base.dt)
        @test growth(fq, fq).prdr * base.dt <= qdon * (1.0 + 1.0e-9)
        # and the axis partners come from the SAME realized growth, so they move with it
        g1 = growth(1.0, 1.0); gf = growth(1.0e-3, 1.0e-3)
        @test gf.ardr <= g1.ardr && gf.crdr <= g1.crdr
        @test (gf.prdr == 0.0) == (gf.ardr == 0.0 && gf.crdr == 0.0)
        # unit factors are the Fortran path, bitwise
        @test growth(1.0, 1.0).prdr === Scythe.ishmael_riming_growth(
            base.dt, base.rni, base.deltastr, base.rbdum, base.nidum, base.ani, base.cni,
            base.temp, rim.qc, rim.nc, rim.nrm, rim.nrd, 5.0e-4,
            rim.qr, rim.nr, rim.nrm, rim.nrd, 5.0e-4, 1.0, true,
            Scythe.ISHMAEL_NU, Scythe.ISHMAEL_AO, Scythe.ISHMAEL_GAMMNU,
            Scythe.ISHMAEL_I_GAMMNU, Scythe.ISHMAEL_FOURTHIRDSPI).prdr
    end

    @testset "ice: the impulse-form rates are finite rates, and NO rate sees ts" begin
        # ISHMAEL writes homogeneous freezing and DeMott activation as `reservoir/Δt` -- the
        # whole reservoir converted in one step. That is the depletion caps' defect with the
        # sign reversed: the rate, and hence the converged solution, is a function of the step
        # size. Both are relaxations on physical timescales here.
        T_cold = Scythe.T_0 - 40.0            # below -35 C: homogeneous freezing is open
        qc, nc, qr, nr = 1.0e-3, 4.0e8, 5.0e-4, 1.0e4

        # ── Homogeneous freezing ──
        # `qc = 1e-3` in `nc = 4e8` droplets is 8.4 micron drops, comfortably above r_min, so
        # the mass-consistency bound does NOT bind and the drop-preserving transfer stands.
        h1 = Scythe._ice_homogeneous_rates(T_cold, qc, nc, qr, nr, 5.0)
        h2 = Scythe._ice_homogeneous_rates(T_cold, qc, nc, qr, nr, 10.0)
        @test h1.mim ≈ qc / 5.0 rtol = 1e-14
        @test h1.nim ≈ nc / 5.0 rtol = 1e-14
        @test h1.mimr ≈ qr / 5.0 rtol = 1e-14
        @test h1.nimr ≈ nr / 5.0 rtol = 1e-14
        # Exact 1/tau scaling -- doubling the timescale halves every leg.
        @test h2.mim ≈ 0.5 * h1.mim rtol = 1e-14
        @test h2.nimr ≈ 0.5 * h1.nimr rtol = 1e-14
        # Mass and number stay in the mean-droplet-mass ratio, so a frozen droplet becomes
        # exactly one crystal of its own mass at ANY timescale.
        @test h1.mim / h1.nim ≈ qc / nc rtol = 1e-14
        @test h2.mim / h2.nim ≈ qc / nc rtol = 1e-14

        # ── The seeded crystal is never below the smallest size the scheme resolves ──
        # `ISHMAEL_RMIN` is `var_check`'s mean-radius floor AND the bottom of the itab/itabr
        # table domain; a number source that seeds below it hands the habit laws a particle
        # they were never fitted for, which is what detonated the first live ice arm.
        rad(m, n) = (m / n / (Scythe.ISHMAEL_FOURTHIRDSPI * Scythe.ISHMAEL_RHOI))^(1 / 3)
        @test Scythe.ISHMAEL_M_MIN ≈
              Scythe.ISHMAEL_FOURTHIRDSPI * Scythe.ISHMAEL_RHOI * Scythe.ISHMAEL_RMIN^3 rtol = 1e-14
        # A THIN anvil cloud is the case that violates it: the closure carries a FIXED droplet
        # number whatever the content, so 1e-7 kg/kg in 1e8 droplets/m^3 is 0.2 micron.
        thin = Scythe._ice_homogeneous_rates(T_cold, 1.0e-7, nc, 0.0, 0.0, 5.0)
        @test thin.mim ≈ 1.0e-7 / 5.0 rtol = 1e-14          # the MASS transfers in full
        @test thin.nim < nc / 5.0                            # ...the number is bounded
        @test thin.nim ≈ thin.mim / Scythe.ISHMAEL_M_MIN rtol = 1e-14
        @test rad(thin.mim, thin.nim) ≈ Scythe.ISHMAEL_RMIN rtol = 1e-9
        # ...and over a sweep of cloud contents the implied crystal is NEVER sub-r_min, while
        # the bound stays inactive wherever the droplets are genuinely resolvable.
        bound_ever = false
        for q in (1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 5.0e-3)
            h = Scythe._ice_homogeneous_rates(T_cold, q, nc, 0.0, 0.0, 5.0)
            @test rad(h.mim, h.nim) >= Scythe.ISHMAEL_RMIN * (1 - 1e-9)
            @test h.mim ≈ q / 5.0 rtol = 1e-14               # mass is untouched throughout
            h.nim < nc / 5.0 * (1 - 1e-12) && (bound_ever = true)
        end
        @test bound_ever                                      # it does bind somewhere
        # Large droplets: the bound is exactly inactive, bitwise.
        big = Scythe._ice_homogeneous_rates(T_cold, 5.0e-3, nc, 0.0, 0.0, 5.0)
        @test big.nim == nc / 5.0

        # THE RAIN LEG'S GUARD MUST NEVER BIND at any state the rain closure can be in:
        # raindrops are two to three decades above r_min. It is a safety net, not physics.
        for (q, n) in ((1.0e-6, 1.0e2), (1.0e-5, 1.0e3), (1.0e-4, 2.5e3),
                       (1.0e-3, 2.5e4), (5.0e-3, 1.0e5), (1.0e-5, 2.5e5))
            h = Scythe._ice_homogeneous_rates(T_cold, 0.0, 0.0, q, n, 5.0)
            @test h.nimr == n / 5.0                           # unbound, bitwise
            @test rad(h.mimr, h.nimr) > 10.0 * Scythe.ISHMAEL_RMIN
        end

        # VOLUME SEEDING is consistent with the seeded size, and is shared by every channel:
        # `_ice_nucleation_volume` reads the SUMMED mass/number ratio, so bounding that ratio
        # bounds the seeded axis. Below the r_min ratio it would clamp; at or above it, it
        # tracks the mass.
        vmin = Scythe._ice_nucleation_volume(thin.mim, thin.nim)
        @test vmin > 0.0
        @test Scythe._ice_nucleation_volume(2.0 * thin.mim, 2.0 * thin.nim) ≈
              2.0 * vmin rtol = 1e-12                         # homogeneous of degree one
        @test Scythe._ice_nucleation_volume(0.0, thin.nim) == 0.0
        @test Scythe._ice_nucleation_volume(thin.mim, 0.0) == 0.0
        # The -35 C threshold is a STATE test and still shuts it off exactly.
        for T in (Scythe.T_0 - 35.0, Scythe.T_0 - 10.0, Scythe.T_0 + 5.0)
            z = Scythe._ice_homogeneous_rates(T, qc, nc, qr, nr, 5.0)
            @test z.mim == 0.0 && z.nim == 0.0 && z.mimr == 0.0 && z.nimr == 0.0
        end
        # ...and so does an empty reservoir.
        z = Scythe._ice_homogeneous_rates(T_cold, 0.0, nc, 0.0, nr, 5.0)
        @test z.mim == 0.0 && z.nim == 0.0 && z.mimr == 0.0 && z.nimr == 0.0

        # ── DeMott activation ──
        rhoair = 0.4
        d1 = Scythe._ice_demott_rates(T_cold, 0.05, rhoair, 0.0, 1.0)
        d2 = Scythe._ice_demott_rates(T_cold, 0.05, rhoair, 0.0, 2.0)
        @test d1.nnuccd > 0.0
        @test d2.nnuccd ≈ 0.5 * d1.nnuccd rtol = 1e-14          # 1/tau_act scaling
        # New particles are 2 micron spheres of density RHOI, so the mass follows the number.
        @test d1.mnuccd / d1.nnuccd ≈
              Scythe.ISHMAEL_FOURTHIRDSPI * Scythe.ISHMAEL_RHOI * (2.0e-6)^3 rtol = 1e-14
        # It is a DEFICIT rate: activation stops once the ice number reaches the available
        # nuclei, which is what makes it self-limiting without any cap on the tendency.
        n_target = d1.nnuccd * 1.0 * rhoair                      # back out n_IN [m^-3]
        @test Scythe._ice_demott_rates(T_cold, 0.05, rhoair, n_target, 1.0).nnuccd == 0.0
        @test Scythe._ice_demott_rates(T_cold, 0.05, rhoair, 2 * n_target, 1.0).nnuccd == 0.0
        @test Scythe._ice_demott_rates(T_cold, 0.05, rhoair, 0.5 * n_target, 1.0).nnuccd ≈
              0.5 * d1.nnuccd rtol = 1e-12
        # The existing-ice CEILING is a measured concentration, not a Δt limiter, and stays:
        # very cold air would activate past it, and does not.
        cold = Scythe._ice_demott_rates(Scythe.T_0 - 70.0, 0.05, rhoair, 0.0, 1.0)
        @test cold.nnuccd <= Scythe.ISHMAEL_IN_CEILING / (1.0 * rhoair) * (1 + 1e-12)
        # The nucleation window is a state test.
        @test Scythe._ice_demott_rates(Scythe.T_0 + 1.0, 0.05, rhoair, 0.0, 1.0).nnuccd == 0.0
        @test Scythe._ice_demott_rates(T_cold, -0.01, rhoair, 0.0, 1.0).nnuccd == 0.0

        # ── Neither function can see a timestep: it is not in the signature ──
        @test !any(m -> :dt in Base.method_argnames(m),
                   methods(Scythe._ice_homogeneous_rates))
        @test !any(m -> :dt in Base.method_argnames(m), methods(Scythe._ice_demott_rates))

        # ── Driver level: change ts by 10x on the SAME state ──
        # With any `reservoir/Δt` left anywhere the nucleation-dominated sources would differ
        # by a factor of ~10, and the sources would not converge as the step is refined. Two
        # things legitimately remain, and neither is a rate law:
        #
        #   * the habit PARTITION (`ard`/`crd`/`prdr`/`qagg`), an increment divided by the same
        #     step -- a consistent first-order discretization;
        #   * the REALIZATION of the two freezing relaxations (`_freeze_realization`, the
        #     `J₀(κΔt)` factor on `mim`/`mimr`/`mbig`). `κ` is the parameterization's own
        #     conductance and carries no step; `J₀` is an integrator coefficient of exactly the
        #     kind `b₁₋₃` are, and it is applied TO a rate rather than written INTO one.
        #
        # So the contract this block enforces is not "the sources are step-independent" -- that
        # was only ever true because the stiff freezing channel was being integrated wrongly --
        # but the two statements that actually distinguish a `Δt`-free rate law from a
        # depletion cap: the sources CONVERGE as the step is refined, at first order, and the
        # DEPOSITION rate (which the exponential integrator handles elsewhere, not here) is
        # bitwise independent of the step.
        mktempdir() do tmpdir
            function rates_at(ts)
                m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 258.15, q_l = 1.0e-3,
                                                  rho_i = 1.0e-5, n_i = 5.0e4,
                                                  rho_r = 1.0e-4, n_r = 1.0e3, ts = ts)
                Scythe.advance_column(m, 1, 2)
                S = m.mc_scratch[Threads.threadid()]
                return (Qdot_i = copy(S.Qdot_i1), q = copy(S.SRC_i1q),
                        n = copy(S.SRC_i1n), c = copy(S.ICE_C), r = copy(S.ICE_R),
                        f = copy(S.FRZ_NET))
            end
            a = rates_at(0.1)
            b = rates_at(0.01)
            c = rates_at(0.001)
            # DEPOSITION contains no Δt at all: bitwise, at both refinements.
            @test a.Qdot_i == b.Qdot_i
            @test b.Qdot_i == c.Qdot_i
            reldiff(x, y) = maximum(abs.(x .- y)) / max(maximum(abs.(x)), 1e-300)
            for (nm, x, y, z) in (("SRC_i1q", a.q, b.q, c.q), ("SRC_i1n", a.n, b.n, c.n),
                                  ("ICE_C", a.c, b.c, c.c), ("ICE_R", a.r, b.r, c.r),
                                  ("FRZ_NET", a.f, b.f, c.f))
                coarse = reldiff(x, y)
                fine = reldiff(y, z)
                # Bounded at every step (no `reservoir/Δt` anywhere)...
                @test coarse < 0.2
                # ...and CONVERGENT: refining the step by 10 shrinks the disagreement by
                # roughly the same factor, which a `Δt` in a rate law would not do.
                @test fine <= 0.2 * coarse + 1.0e-12
            end
        end
    end

    @testset "ice: the var_check consistency source restores the moments" begin
        # ISHMAEL's `var_check` writes back; this port carries the raw moments and reinstates
        # the write-back as a source. Seed a MASS/NUMBER pair that no population can have --
        # 5e9 crystals per m^3 holding 1e-9 kg/m^3, i.e. 9 nm "crystals" -- and the number
        # source must be a large NEGATIVE one pulling it back toward the effective value.
        mktempdir() do tmpdir
            m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 258.15, q_l = 1.0e-3,
                                              rho_i = 1.0e-9, n_i = 5.0e9, r_i = 10.0e-6)
            Scythe.advance_column(m, 1, 2)
            S = m.mc_scratch[Threads.threadid()]
            @test all(S.SRC_i1n .< 0.0)
            # It is a RESTORATION on a physical timescale, not a one-step wipe: the target is
            # var_check's own re-diagnosis and the approach is `(n_eff − n)/tau_varcheck`, so
            # a `tau_varcheck`-sized excursion removes an e-folding, not the whole excess.
            tau_vc = get(mod.physical_params, :tau_varcheck, 5.0)
            n_new = 5.0e9 .+ (tau_vc .* S.SRC_i1n)
            @test all(n_new .>= 0.0)
            @test maximum(n_new) < 1.0e8
            # ...and it carries NO Δt: doubling tau halves the source, and the step is inert.
            @test all(abs.(S.SRC_i1n) .< 5.0e9 / tau_vc * (1 + 1e-9))
            # The MASS is untouched by it -- var_check never changes the mass, so the water
            # exchange closure of the previous testset is unaffected.
            dep = S.Qdot_i1 .+ S.Qdot_i2 .+ S.Qdot_i3
            ice = S.SRC_i1q .+ S.SRC_i2q .+ S.SRC_i3q
            scale = maximum(abs.(ice)) + maximum(abs.(dep))
            # `SRC_i*q` is the non-deposition ice mass source (the deposition leg is withheld
            # with the rest of the relaxation pair), so the liquid must balance it alone.
            @test maximum(abs.(ice .+ (S.ICE_C .+ S.ICE_R))) < 1.0e-12 * scale
        end
        # ...and switching it off leaves the moments alone. Same state, no restoration.
        mktempdir() do tmpdir
            m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 258.15, q_l = 1.0e-3,
                                              rho_i = 1.0e-9, n_i = 5.0e9, r_i = 10.0e-6,
                                              extra_options =
                                                  Dict{Symbol,Any}(:ice_var_check => false))
            Scythe.advance_column(m, 1, 2)
            S = m.mc_scratch[Threads.threadid()]
            # Without it the only number sink is sublimation, which here is orders of
            # magnitude smaller than the inconsistency.
            @test maximum(abs.(S.SRC_i1n)) < 1.0e8
        end
        # A LONGER tau is a weaker source, exactly proportionally.
        mktempdir() do tmpdir
            function src_at(tau)
                m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 258.15, q_l = 1.0e-3,
                    rho_i = 1.0e-9, n_i = 5.0e9, r_i = 10.0e-6,
                    extra_params = Dict{Symbol,Float64}(:tau_varcheck => tau))
                Scythe.advance_column(m, 1, 2)
                return copy(m.mc_scratch[Threads.threadid()].SRC_i1n)
            end
            s5 = src_at(5.0)
            s10 = src_at(10.0)
            @test maximum(abs.(s10 .- 0.5 .* s5)) < 1.0e-6 * maximum(abs.(s5))
        end
    end

    @testset "ice: the stiffness census gains three channels" begin
        @test Scythe.MC_STIFF_CHANNELS == 5
        @test Scythe.MC_STIFF_NAMES == ("cloud", "rain", "ice1", "ice2", "ice3")
        @test Scythe.MC_STIFF_WARNED ==
              Scythe.MC_STIFF_FIRST + Scythe.MC_STIFF_N * Scythe.MC_STIFF_CHANNELS
        # The stiffness block is no longer last: the DONOR-DEPLETION block appends after it,
        # and the ANCHOR-RECONCILIATION census (Stage C) after that.
        @test Scythe.MC_DONOR_FIRST == Scythe.MC_STIFF_WARNED + 1
        @test Scythe.MC_ANCHOR_GAP == Scythe.MC_DONOR_WARNED + 1
        # ...and the per-channel ATTRIBUTION block (Stage 0a) after THAT, with the
        # POPULATION-RECONCILIATION census (Stage 3a) last, which is what the final row of
        # `MC_WATER_STATS` now is.
        @test Scythe.MC_ATTR_FIRST == Scythe.MC_ANCHOR_WARNED + 1
        @test Scythe.MC_ATTR_CHANNELS == length(Scythe.MC_ATTR_NAMES)
        @test Scythe.MC_ATTR_WARM_PTS ==
              Scythe.MC_ATTR_FIRST + Scythe.MC_ATTR_N * Scythe.MC_ATTR_CHANNELS
        @test Scythe.MC_ATTR_LAST == Scythe.MC_ATTR_WARM_PTS + 4
        @test Scythe.MC_POP_MAX == Scythe.MC_ATTR_LAST + 1
        @test Scythe.MC_POP_LAST == Scythe.MC_POP_MAX + 4
        @test length(Scythe.MC_WATER_STATS) == Scythe.MC_POP_LAST
        @test Scythe.MC_WATER_STATS[Scythe.MC_ANCHOR_GAP] == :a_gap
        @test Scythe.MC_WATER_STATS[Scythe.MC_ANCHOR_REMOVED] == :a_rem
        @test Scythe.MC_WATER_STATS[Scythe.MC_ATTR_FIRST] == :x_hom
        @test Scythe.MC_WATER_STATS[Scythe.MC_ATTR_WARM_HELD] == :x_wheld
        @test Scythe.MC_WATER_STATS[Scythe.MC_POP_MAX] == :o_max
        @test Scythe.MC_WATER_STATS[Scythe.MC_POP_SEED] == :o_seed
        @test Scythe.MC_WATER_STATS[Scythe.MC_POP_WARNED] == :o_warned
        @test Scythe.MC_DONOR_CHANNELS == length(Scythe.MC_DONOR_NAMES)
        # STAGE 1b: the three ICE NUMBER donors append after the three sublimation rows, so
        # the donor block is twelve channels and everything downstream of it shifted again.
        @test Scythe.MC_DONOR_CHANNELS == 12
        @test (Scythe.MC_DONOR_N1, Scythe.MC_DONOR_N2, Scythe.MC_DONOR_N3) == (10, 11, 12)
        @test Scythe.MC_DONOR_N3 == Scythe.MC_DONOR_CHANNELS
        @test Scythe.MC_WATER_STATS[Scythe.MC_DONOR_FIRST +
                                    Scythe.MC_DONOR_N * (Scythe.MC_DONOR_N1 - 1)] == :p_n1_max
        @test Scythe.MC_WATER_STATS[Scythe.MC_DONOR_FIRST +
                                    Scythe.MC_DONOR_N * (Scythe.MC_DONOR_N3 - 1) + 1] == :p_n3_n
        @test Scythe.MC_WATER_STATS[Scythe.MC_DONOR_WARNED] == :p_warned
        mktempdir() do tmpdir
            m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 258.15, rho_i = 1.0e-5,
                                              n_i = 5.0e4)
            m.mc_water_stats .= 0.0
            Scythe.advance_column(m, 1, 2)
            row1 = Scythe.MC_STIFF_FIRST + Scythe.MC_STIFF_N * (Scythe.MC_STIFF_I1 - 1)
            row3 = Scythe.MC_STIFF_FIRST + Scythe.MC_STIFF_N * (Scythe.MC_STIFF_I3 - 1)
            @test maximum(view(m.mc_water_stats, row1, :)) > 0.0      # ice1 is loaded
            @test maximum(view(m.mc_water_stats, row3, :)) == 0.0     # ice3 is empty
        end
    end

    # ── The POPULATION GATE ──────────────────────────────────────────────────────
    #
    # Growth without activation is impossible: deposition needs crystals to deposit onto,
    # riming needs crystals to rime, and a fall speed is a property of particles. A slot
    # holding MASS WITH NO NUMBER is not a population — it is a state four independent
    # spline fits can produce and nothing physical can — so every RATE and every FALL SPEED
    # must be an exact zero there, while NUCLEATION (which creates number and mass together)
    # must still fire. See the gate comment in `mc_ice_sources!`.
    #
    # `:ice_population_source => false` in the two dead-species fixtures below is not a
    # loosening of the claim, it is what keeps the claim about RATES. Stage 3a's fourth
    # reconciliation tier is now the one and only thing that writes into a dead species'
    # slots — deliberately, because the gate leaves that mass exempt from every device that
    # could remove it — and it is not a rate: it seeds crystals so that the rates can own
    # the species again from the next step on. It has its own testset below.
    @testset "ice: mass with no number is inert, but nucleation still fires" begin
        no_pop = Dict{Symbol,Any}(:ice_population_source => false)
        mktempdir() do tmpdir
            # A column carrying ice MASS with the number slot at exactly zero. Warm enough
            # that no nucleation fires, so the only thing that could move the slots is a
            # rate read off the phantom population var_check would manufacture.
            m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 258.15, q_l = 0.0,
                                              rho_i = 1.0e-5, n_i = 0.0,
                                              extra_options = no_pop)
            Scythe.advance_column(m, 1, 2)
            S = m.mc_scratch[Threads.threadid()]
            # Every species-1 SOURCE is an exact zero: no rate may read a population that
            # is not carried.
            @test all(iszero, S.SRC_i1q)
            @test all(iszero, S.SRC_i1n)
            @test all(iszero, S.SRC_i1a)
            @test all(iszero, S.SRC_i1c)
            # The fall speeds the flux would have used are exactly zero, so nothing
            # sediments off unsupported mass -- and `_ice_flux!` then short-circuits.
            @test all(iszero, S.Vi1m)
            @test all(iszero, S.Vi1n)
            @test all(iszero, S.F_i1q_z)
            # ...and no deposition timescale is ever formed on it.
            @test all(iszero, S.invtau_i1)
            @test all(iszero, S.Qdot_i1)
        end

        mktempdir() do tmpdir
            # The SAME mass, now with a consistent number: the species is alive and the
            # rates must be nonzero, or the gate would be vacuous.
            m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 258.15, q_l = 0.0,
                                              rho_i = 1.0e-5, n_i = 5.0e4)
            Scythe.advance_column(m, 1, 2)
            S = m.mc_scratch[Threads.threadid()]
            @test any(!iszero, S.Vi1m)                    # a real population falls
            @test any(!iszero, S.invtau_i1)               # and can exchange vapor
        end

        mktempdir() do tmpdir
            # ACTIVATION IS NOT GATED: cold, supersaturated, and with NO ice at all, the
            # nucleation channels must still create ice — number and mass together.
            m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 233.15, q_l = 1.0e-3,
                                              rho_i = 0.0, n_i = 0.0)
            Scythe.advance_column(m, 1, 2)
            S = m.mc_scratch[Threads.threadid()]
            tgt = any(>(0.0), S.SRC_i1n) ? (S.SRC_i1q, S.SRC_i1n) : (S.SRC_i2q, S.SRC_i2n)
            @test any(>(0.0), tgt[2])                     # crystals are created
            @test any(>(0.0), tgt[1])                     # with mass, in the same cells
            k = argmax(tgt[2])
            @test tgt[1][k] > 0.0                         # never number without mass
        end
    end

    @testset "ice: a decorrelated slot cannot grow, at any supersaturation" begin
        # The defect state, driven as hard as the drive clip allows: mass present, number
        # exactly zero, air strongly supersaturated over ice. Under the phantom population
        # this grew without bound; gated, it must not move at all.
        mktempdir() do tmpdir
            m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 243.15, q_l = 5.0e-3,
                                              rho_i = 1.0e-4, n_i = 0.0,
                                              extra_options = Dict{Symbol,Any}(
                                                  :ice_population_source => false))
            for step in 1:5
                Scythe.advance_column(m, 1, step)
            end
            S = m.mc_scratch[Threads.threadid()]
            @test all(iszero, S.Qdot_i1)                  # no deposition on nothing
            @test all(iszero, S.invtau_i1)                # the timescale is never formed
            @test all(iszero, S.Vi1m)                     # and no sedimentation either
            @test all(iszero, S.F_i1q_z)
            # The slot is NOT frozen solid: this air is cold and supersaturated, so
            # NUCLEATION fires into species 1 and legitimately adds mass. What the gate
            # guarantees is the author's principle itself -- mass never arrives without
            # the number that carries it. Every cell that gains ice mass gains crystals.
            for i in eachindex(S.SRC_i1q)
                S.SRC_i1q[i] > 0.0 && @test S.SRC_i1n[i] > 0.0
            end
            # ...and the only mass source present IS the nucleation one: every
            # population-dependent channel is an exact zero.
            @test all(iszero, S.Qdot_i1)                  # deposition
            @test all(iszero, S.SRC_i1a .* iszero.(S.SRC_i1n))   # no volume without number
        end
    end

    # ── STAGE 3e: THE MINIMUM-CRYSTAL BOUND, AND THE RAIN POPULATION GATE ────────
    #
    # reference/FINDINGS_ISHMAEL_S8S9.md §5i. At the anvil top the two rain moments
    # decorrelate on their independent spline fits: `n_r` rings to EXACTLY zero while `q_r`
    # keeps ~1e-8 kg/kg at 192–203 K. `ishmael_rain_lambda` floors the number at `QNSMALL`
    # and clamps the slope to `LAMMINR`, handing every DSD consumer a PHANTOM population of
    # 2800 μm drops; Bigg, 30–40 K outside its validity where `exp(0.66ΔT) ~ 1e21`, then
    # freezes it — and its number leg is realized at `relaxation_realization(rate, 0, dt) =
    # 1.0` while its mass leg's factor clamps to 1e-19. Two devices close it, and they are
    # the two principles already ratified: the number source is bounded by its REALIZED mass
    # partner at `ISHMAEL_M_MIN`, and no kernel may read a rain DSD the carried number does
    # not support.
    @testset "ice: every realized ice-number source is bounded by its mass partner" begin
        M = Scythe.ISHMAEL_M_MIN
        gate_off = Dict{Symbol,Any}(:rain_population_gate => false)
        ivars = Dict(v => i for (i, v) in enumerate(
            Scythe.mc_var_names(Dict{Symbol,Any}(:rain_moments => 2,
                                                 :ice_microphysics => :ishmael); cyl = false)))

        # THE §5i STATE, reconstructed: rain mass with the number slot at exactly zero, at
        # 202 K, with no ice anywhere. Nothing else can fire — the cloud is empty so the
        # homogeneous cloud leg is zero, DeMott needs `sup ≥ 0` and the column is
        # sub-saturated, Hallett-Mossop is outside its window and there is no ice to rime or
        # collect with — so species 1's two sources are exactly Bigg plus the homogeneous
        # RAIN leg, and the slots read the assembly directly.
        mktempdir() do tmpdir
            m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 203.0, q_l = 0.0,
                                              rho_i = 0.0, n_i = 0.0,
                                              rho_r = 1.2e-8, n_r = 0.0,
                                              extra_options = gate_off)
            Scythe.advance_column(m, 1, 1)
            S = m.mc_scratch[Threads.threadid()]
            i = 4
            rhoair = S.rho_d[i]; temp = S.Tk[i]
            qr = max(S.rho_r[i], 0.0) / rhoair
            nr = max(S.n_r[i], 0.0) / rhoair
            qc = max(S.rho_c[i], 0.0) / rhoair
            @test 192.0 < temp < 203.0                    # the measured window
            @test nr == 0.0                               # the carried number rang to zero
            @test qr > Scythe.ISHMAEL_QSMALL              # ...and the mass did not

            # What the unbounded assembly would have produced, rebuilt from the same pieces
            # the loop uses.
            bg = Scythe.ishmael_bigg_freezing(temp, qr, nr, mod.ts; reservoir_caps = false)
            hf = Scythe._ice_homogeneous_rates(temp, qc, 1.0e6 / rhoair, qr, nr, 5.0)
            pe = Scythe._ice_empty_pre()
            (f_qc, f_qr, f_nr) = Scythe._ice_donor_factors(hf, bg, pe, pe, pe, qc, qr, nr,
                                                           S.kappa_ev[i], mod.ts)
            # The two factors, seventeen orders apart on the SAME collisions — `f_nr` is
            # `relaxation_realization`'s zero-reservoir 1.0, `f_qr` is the clamp.
            @test f_nr === 1.0
            @test f_qr < 1.0e-15
            mbig = f_qr * bg.mbiggr
            nbig_raw = f_nr * bg.nbiggr
            nbig_bound = mbig / M
            @test nbig_raw > 1.0e4 * nbig_bound           # measured 7.4e4x on this fixture

            # THE BOUND, on the slot the step actually applied. Species 1's whole number
            # source is Bigg, and it is the realized Bigg MASS at the minimum crystal — not
            # the unbounded rate.
            @test S.SRC_i1n[i] ≈ rhoair * nbig_bound rtol = 1e-12
            @test S.SRC_i1n[i] < 1.0e-4 * rhoair * nbig_raw
            # ...and the invariant it enforces, stated on the slots: no more crystals than
            # the mass that arrived with them supports.
            @test S.SRC_i1n[i] <= S.SRC_i1q[i] / M * (1.0 + 1.0e-12)
            @test S.SRC_i1q[i] > 0.0                      # the mass leg is untouched
            # Every species obeys it, at every gridpoint of the column, in both directions
            # of the habit selector.
            for (nsrc, qsrc) in ((S.SRC_i1n, S.SRC_i1q), (S.SRC_i2n, S.SRC_i2q),
                                 (S.SRC_i3n, S.SRC_i3q))
                @test all(k -> nsrc[k] <= max(qsrc[k], 0.0) / M * (1.0 + 1.0e-12),
                          eachindex(nsrc))
            end
        end

        # WITH THE GATE ON (the default) the kernel never runs at all, so the bound has
        # nothing left to bind: no crystals, and no Bigg mass either.
        mktempdir() do tmpdir
            m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 203.0, q_l = 0.0,
                                              rho_i = 0.0, n_i = 0.0,
                                              rho_r = 1.0e-6, n_r = 0.0)
            Scythe.advance_column(m, 1, 1)
            S = m.mc_scratch[Threads.threadid()]
            @test all(iszero, S.SRC_i1n)
            @test all(iszero, S.SRC_i2n)
            # The homogeneous RAIN leg is NOT gated — it needs no size distribution — so the
            # mass channel is still alive and this is a gate on the DSD, not on the rain.
            @test any(>(0.0), S.SRC_i1q)
        end

        # THE SPIKE DOES NOT GROW. Under the unbounded term this state made ~1e15 m⁻³ s⁻¹ of
        # immortal crystals; here the number the column can accumulate is capped by the rain
        # mass it has to make them out of, and with the gate on nothing is made at all.
        mktempdir() do tmpdir
            m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 203.0, q_l = 0.0,
                                              rho_i = 0.0, n_i = 0.0,
                                              rho_r = 1.2e-8, n_r = 0.0,
                                              extra_options = gate_off)
            for step in 1:20
                Scythe.advance_column(m, 1, step)
            end
            # Everything the rain could possibly become, one crystal per `M_MIN`: the whole
            # seeded rain mass. The unbounded term passed this in a SINGLE step.
            ceiling = 1.2e-8 / M
            n1 = maximum(view(patch.physical, :, ivars["n_i1"], 1))
            @test n1 < 100.0 * ceiling
            @test n1 < 1.0e8
        end

        mktempdir() do tmpdir
            m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 203.0, q_l = 0.0,
                                              rho_i = 0.0, n_i = 0.0,
                                              rho_r = 1.2e-8, n_r = 0.0)
            for step in 1:20
                Scythe.advance_column(m, 1, step)
            end
            @test all(iszero, view(patch.physical, :, ivars["n_i1"], 1))
        end
    end

    @testset "ice: the rain population gate, and which kernels needed it" begin
        # WHICH KERNEL NEEDED THE GATE, at the kernel boundary. `ishmael_ice_rain_riming`
        # SELF-GATES: every rate it returns carries a factor of the CARRIED `nr` it was
        # handed, so at `nr = 0` the whole tuple is exact zeros and the gate would be
        # redundant. `ishmael_bigg_freezing` does NOT: its only existence test is on `q_r`,
        # and the number it freezes is `ishmael_rain_lambda`'s floored, clamped one.
        itabr = mktempdir() do tmpdir
            m, _, _, _ = make_ice_mtile(tmpdir; Tsurf = 258.15)
            m.ishmael_tables.itabr
        end
        rr0 = Scythe.ishmael_ice_rain_riming(itabr, 3.0e-5, 1.0e-8, 0.0, 0.6, 500.0,
                                             5.0e4, 1.0, 202.0, 1.0e-5)
        @test rr0.rimesumr == 0.0
        @test rr0.numrateri == 0.0 && rr0.rainrateri == 0.0 && rr0.icerateri == 0.0
        @test rr0.dQRfzri == 0.0 && rr0.dQIfzri == 0.0 && rr0.dNfzri == 0.0
        @test rr0.qi_qr_nrm == 0.0 && rr0.qi_qr_nrd == 0.0 && rr0.qi_qr_nrn == 0.0
        # The phantom DSD the gate exists to forbid: at `nr = 0` the slope clamps to the
        # 2800 μm drop, the largest the scheme admits, and Bigg returns a large rate for a
        # rain population that is not there.
        dsd = Scythe.ishmael_rain_lambda(1.0e-8, 0.0)
        @test dsd.lamr == Scythe.ISHMAEL_LAMMINR
        @test 0.5 / dsd.lamr ≈ 1400.0e-6 rtol = 1e-12     # 2800 μm diameter
        bg0 = Scythe.ishmael_bigg_freezing(202.0, 1.0e-8, 0.0, 0.1; reservoir_caps = false)
        @test bg0.nbiggr > 1.0e10
        @test bg0.mbiggr > 0.0

        # BITWISE INERT WHEREVER THE RAIN IS HEALTHY. A column with real rain — mass AND
        # number — must give the identical field under both settings, so the gate is a
        # statement about a defect state and not a change to the storm.
        mktempdir() do tmpdir
            fields = map((true, false)) do g
                m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 258.15, q_l = 1.0e-3,
                    rho_i = 1.0e-5, n_i = 5.0e4, rho_r = 1.0e-4, n_r = 1.0e3,
                    extra_options = Dict{Symbol,Any}(:rain_population_gate => g))
                for step in 1:3
                    Scythe.advance_column(m, 1, step)
                end
                copy(m.tile.physical[:, :, 1])
            end
            @test fields[1] == fields[2]
        end

        # ...and it is NOT vacuous: the same column with the number rung out is where the
        # two answers separate.
        mktempdir() do tmpdir
            src = map((true, false)) do g
                m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 203.0, q_l = 0.0,
                    rho_i = 0.0, n_i = 0.0, rho_r = 1.2e-8, n_r = 0.0,
                    extra_options = Dict{Symbol,Any}(:rain_population_gate => g))
                Scythe.advance_column(m, 1, 1)
                S = m.mc_scratch[Threads.threadid()]
                (maximum(S.SRC_i1n), maximum(S.SRC_i1q))
            end
            @test src[1][1] == 0.0                        # gate on: no crystals
            @test src[2][1] > 0.0                         # gate off: the phantom fires
            @test src[2][2] > src[1][2]                   # and freezes rain mass with it
        end

        # ── STAGE 3f: THE WARM SIDE OF THE SAME AUDIT ────────────────────────────────
        # `rain_selfcollection_2m` is the warm two-moment closure that reads the DIAGNOSED
        # number rather than the carried one, and at the `LAMMINR` clamp its breakup rolloff
        # (`dum = 2 - e^5.75 = -312`) flips sign and makes it a number SOURCE — the way a
        # rung-out rain slot gets a positive number back, and with it a population for Bigg
        # to freeze on the next step. It now tests the carried number first
        # (`microphysics.jl`, and see the per-consumer classification in test_microphysics).
        # HERE, at the host: a column with rain mass, no rain number and no cloud must write
        # an EXACTLY zero rain-number source. Every leg is zero for its own reason —
        # autoconversion has no cloud, the evaporation number sink self-gates at `n_r <= 0`,
        # and self-collection is gated — so the assertion is an exact one.
        mktempdir() do tmpdir
            m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 203.0, q_l = 0.0,
                                              rho_i = 0.0, n_i = 0.0,
                                              rho_r = 1.2e-8, n_r = 0.0)
            Scythe.advance_column(m, 1, 1)
            S = m.mc_scratch[Threads.threadid()]
            # The state the audit is about: rain mass above the closures' threshold with no
            # carried drops anywhere.
            @test all(<=(0.0), S.n_r)
            @test any(S.rho_r .> Scythe.RAIN_2M_Q_MIN .* S.rho_d)
            @test all(iszero, S.NR_SRC)
            # ...and it is not vacuous: the same column WITH drops has a live number source.
            m2, _, _, _ = make_ice_mtile(tmpdir; Tsurf = 203.0, q_l = 0.0,
                                         rho_i = 0.0, n_i = 0.0,
                                         rho_r = 1.0e-6, n_r = 1.0e3)
            Scythe.advance_column(m2, 1, 1)
            @test any(!iszero, m2.mc_scratch[Threads.threadid()].NR_SRC)
        end
    end

    # ── STAGE 3a: THE RECONCILIATION OF THE POPULATION ───────────────────────────
    #
    # TeX §"Reconciliation of the population". The gate above is right, and the same
    # measurement that justified it records its consequence: mass with no number is exempt
    # from every device that could REMOVE it too — below the melting level of the O01 ice
    # column the ice is number-less at 97–99.9% by mass over the final half hour, sitting at
    # the surface at 300 K, with an anchor share of exactly one. The fourth reconciliation
    # tier returns it to a representation the equations can act on: rain with L_f absorbed
    # above T_0, 2 um crystals below it, on tau_pop, one-sided and censused.
    @testset "ice: the population reconciliation returns number-less mass" begin
        tau = 10.0                                    # physical_params[:tau_ice_population]

        # ── (1) BELOW T_0: number only. No mass moves, nothing thermodynamic happens, and
        # the seeded population is the smallest crystal the scheme resolves.
        mktempdir() do tmpdir
            m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 258.15, q_l = 0.0,
                                              rho_i = 1.0e-5, n_i = 0.0)
            m.mc_water_stats .= 0.0
            Scythe.advance_column(m, 1, 2)
            S = m.mc_scratch[Threads.threadid()]
            @test all(S.Tk .< Scythe.T_0)
            hits = 0
            for i in eachindex(S.SRC_i1n)
                (S.i1q[i] > Scythe.ISHMAEL_QSMALL * S.rho_d[i] && S.i1n[i] <= 0.0) || continue
                hits += 1
                rate = S.i1q[i] / tau
                @test S.SRC_i1n[i] ≈ rate / Scythe.ISHMAEL_M_MIN rtol = 1e-13
                @test S.SRC_i1a[i] ≈ Scythe._ice_nucleation_volume(rate,
                          rate / Scythe.ISHMAEL_M_MIN) rtol = 1e-13
                @test S.SRC_i1c[i] == S.SRC_i1a[i]
                # THE MASS IS NOT TOUCHED, and no latent heat accompanies the seeding.
                @test S.SRC_i1q[i] == 0.0
                @test S.FRZ_NET[i] == 0.0
                @test S.ICE_R[i] == 0.0
                @test S.ICE_NR[i] == 0.0
            end
            @test hits > 0                            # the fixture really is dead ice
            # Nothing in lambda or N: this branch is not a vapor exchange.
            @test all(iszero, S.invtau_i1)
            @test all(iszero, S.Qdot_i1)
            # ...and the census recorded the defect, its support and the number seeded.
            st = m.mc_water_stats
            @test maximum(view(st, Scythe.MC_POP_MAX, :)) > 0.0
            @test sum(view(st, Scythe.MC_POP_PTS, :)) == Float64(hits)
            @test sum(view(st, Scythe.MC_POP_SEED, :)) > 0.0
            @test sum(view(st, Scythe.MC_POP_RAIN, :)) == 0.0
        end

        # ── (2) The point of the seeding: the species comes ALIVE, and the rates, the
        # consistency source and the sedimentation own it from then on.
        #
        # Sampled over several steps rather than read off the last one. The source is gated
        # on a STATE test (`n <= 0`), like every other rate in the block, so under the
        # multistep it switches on and off as the number crosses zero — one step of seeding,
        # then the AB3 history's negative coefficient pulling back, then more seeding. The
        # claim is that the species HAS a population and rates again, not that it does at
        # some particular step; the cumulative census is the monotone statement.
        mktempdir() do tmpdir
            # `advance_column` alone only forms tendencies; the state advances through
            # `calcTendency` + `gridTransform!` (what `step_mc!` does), and this test is
            # about the STATE, so the loop is written out to sample it every step.
            function run_pop(opts)
                m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 258.15, q_l = 0.0,
                                                  rho_i = 1.0e-5, n_i = 0.0,
                                                  extra_options = opts)
                m.mc_water_stats .= 0.0
                s = m.mc_slots
                ncols = div(size(patch.physical, 1), mod.grid_params.kDim)
                @test all(patch.physical[:, s.i1_n, 1] .== 0.0)      # dead at t = 0
                numbered = false
                alive = false
                fell = false
                for t in 1:12
                    for c in 1:ncols
                        Scythe.advance_column(m, c, t)
                    end
                    S = m.mc_scratch[Threads.threadid()]
                    any(!iszero, S.invtau_i1) && (alive = true)   # it can exchange vapor
                    any(!iszero, S.Vi1m) && (fell = true)         # ...and it falls
                    Scythe.calcTendency(m)
                    gridTransform!(patch)
                    maximum(patch.physical[:, s.i1_n, 1]) > 0.0 && (numbered = true)
                end
                return (m, patch, numbered, alive, fell)
            end

            m, patch, numbered, alive, fell = run_pop(Dict{Symbol,Any}())
            @test numbered                            # crystals, where there were none
            @test alive                               # ...and rates, where there were none
            @test fell
            @test sum(view(m.mc_water_stats, Scythe.MC_POP_SEED, :)) > 0.0

            # The same run with the source OFF stays dead forever: zero number, zero rates.
            m_off, patch_off, numbered_off, alive_off, fell_off =
                run_pop(Dict{Symbol,Any}(:ice_population_source => false))
            @test !numbered_off
            @test !alive_off
            @test !fell_off
            @test all(patch_off.physical[:, m_off.mc_slots.i1_n, 1] .== 0.0)
        end

        # ── (3) ABOVE T_0: the mass returns to the rain, total water closes to the last
        # bit, and L_f goes with it (FRZ_NET < 0 — the transfer ABSORBS the latent heat of
        # fusion, which is what melting ice does).
        mktempdir() do tmpdir
            m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 290.0, q_l = 0.0,
                                              rho_i = 1.0e-5, n_i = 0.0)
            m.mc_water_stats .= 0.0
            Scythe.advance_column(m, 1, 2)
            S = m.mc_scratch[Threads.threadid()]
            @test all(S.Tk .> Scythe.T_0)
            hits = 0
            for i in eachindex(S.SRC_i1q)
                (S.i1q[i] > Scythe.ISHMAEL_QSMALL * S.rho_d[i] && S.i1n[i] <= 0.0) || continue
                hits += 1
                rate = S.i1q[i] / tau
                @test S.SRC_i1q[i] ≈ -rate rtol = 1e-13
                @test S.ICE_R[i] ≈ rate rtol = 1e-13
                # CONSERVATION, exactly: one number, two slots, opposite signs.
                @test S.SRC_i1q[i] + S.ICE_R[i] === 0.0
                # The identity FRZ_NET = -(ICE_C + ICE_R) survives the increment...
                @test S.FRZ_NET[i] === -(S.ICE_C[i] + S.ICE_R[i])
                @test S.FRZ_NET[i] < 0.0
                # ...and the rain number is seeded at autoconversion's 25 um drop.
                @test S.ICE_NR[i] ≈ rate / Scythe.RAIN_2M_M_AUTO rtol = 1e-13
                # Both volume moments leave at the same fraction as the mass.
                @test S.SRC_i1a[i] ≈ -max(S.i1a[i], 0.0) / tau rtol = 1e-13
                @test S.SRC_i1c[i] ≈ -max(S.i1c[i], 0.0) / tau rtol = 1e-13
                # Bounded by dt/tau: no realization factor is needed and none is applied.
                @test -S.SRC_i1q[i] * mod.ts / S.i1q[i] <= mod.ts / tau + eps()
            end
            @test hits > 0
            st = m.mc_water_stats
            @test sum(view(st, Scythe.MC_POP_RAIN, :)) > 0.0
            @test sum(view(st, Scythe.MC_POP_SEED, :)) == 0.0
        end

        # ── (4) THE OFF SWITCH reproduces the pre-Stage-3a tree bitwise, on both branches,
        # while the census keeps measuring: a forensic tool, not a blindfold.
        mktempdir() do tmpdir
            off = Dict{Symbol,Any}(:ice_population_source => false)
            for Tsurf in (258.15, 290.0)
                m_on, _, mod, _ = make_ice_mtile(tmpdir; Tsurf = Tsurf, q_l = 0.0,
                                                 rho_i = 1.0e-5, n_i = 0.0)
                m_off, _, _, _ = make_ice_mtile(tmpdir; Tsurf = Tsurf, q_l = 0.0,
                                                rho_i = 1.0e-5, n_i = 0.0,
                                                extra_options = off)
                m_on.mc_water_stats .= 0.0
                m_off.mc_water_stats .= 0.0
                Scythe.advance_column(m_on, 1, 2)
                Scythe.advance_column(m_off, 1, 2)
                # The device MOVES something with it on...
                @test m_on.expdot_n != m_off.expdot_n
                # ...and with it off, every ice source is the exact zero the gate left.
                S_off = m_off.mc_scratch[Threads.threadid()]
                for nm in (:SRC_i1q, :SRC_i1n, :SRC_i1a, :SRC_i1c,
                           :ICE_C, :ICE_R, :ICE_NR, :FRZ_NET)
                    # `iszero`, not `=== 0.0`: several of these are formed as `-ρ_a·0.0`
                    # and land on the signed zero, which is the same number.
                    @test all(iszero, getproperty(S_off, nm))
                end
                # ...but the census saw the same defect in BOTH runs.
                @test maximum(view(m_on.mc_water_stats, Scythe.MC_POP_MAX, :)) ==
                      maximum(view(m_off.mc_water_stats, Scythe.MC_POP_MAX, :)) > 0.0
                @test sum(view(m_on.mc_water_stats, Scythe.MC_POP_PTS, :)) ==
                      sum(view(m_off.mc_water_stats, Scythe.MC_POP_PTS, :))
            end
        end

        # ── (5) ONE-SIDED AND BITWISE ABSENT from healthy air: a fully live ice column and
        # a warm ice-free one are identical with the source on or off, and the census is
        # silent. This is the property that keeps every existing reference bit-for-bit.
        mktempdir() do tmpdir
            off = Dict{Symbol,Any}(:ice_population_source => false)
            m_on, _, mod, _ = make_ice_mtile(tmpdir; Tsurf = 258.15, q_l = 1.0e-3,
                                             rho_i = 1.0e-5, n_i = 5.0e4)
            m_off, _, _, _ = make_ice_mtile(tmpdir; Tsurf = 258.15, q_l = 1.0e-3,
                                            rho_i = 1.0e-5, n_i = 5.0e4,
                                            extra_options = off)
            m_on.mc_water_stats .= 0.0
            for t in 2:4
                Scythe.advance_column(m_on, 1, t)
                Scythe.advance_column(m_off, 1, t)
            end
            @test m_on.expdot_n == m_off.expdot_n
            @test m_on.var_np1 == m_off.var_np1
            @test all(view(m_on.mc_water_stats,
                           Scythe.MC_POP_MAX:Scythe.MC_POP_LAST, :) .== 0.0)
        end
        mktempdir() do tmpdir
            # The DRY/warm path: ice registered, no ice present, both Stage 3 options on —
            # still bitwise the ice-free run in every common slot.
            args = (; q_l = 3.0e-3, kDim = 16, num_cells = 8, precipitation = true)
            function run_warm(opts)
                m, patch, mod, _ = make_mc_mtile(tmpdir; args..., extra_options = opts)
                ncols = div(size(m.tile.physical, 1), mod.grid_params.kDim)
                for t in 1:5, c in 1:ncols
                    Scythe.advance_column(m, c, t)
                end
                return m
            end
            m_off = run_warm(Dict{Symbol,Any}(:rain_moments => 2))
            m_on = run_warm(Dict{Symbol,Any}(:rain_moments => 2,
                                             :ice_microphysics => :ishmael,
                                             :ice_population_source => true,
                                             :ice_shed_above_t0 => true))
            for slot in 1:(m_on.mc_slots.i1_q - 1)
                @test m_on.var_np1[:, slot] == m_off.var_np1[:, slot]
                @test m_on.expdot_n[:, slot] == m_off.expdot_n[:, slot]
            end
            @test all(view(m_on.mc_water_stats,
                           Scythe.MC_POP_MAX:Scythe.MC_POP_LAST, :) .== 0.0)
        end

        # ── (6) STAGE 3c: WHICH CRYSTAL the below-T_0 branch seeds.
        # `options[:ice_population_seed]`: `:min` (the 2 um sphere), `:large` (var_check's
        # own re-diagnosis of the dead mass at the floor number -- the size-sorted particles
        # the number-less mass actually is) or `:local` (the neighbour's crystals, (7)
        # below). The DEFAULT is `:local`, chosen by measurement (author decision
        # 2026-08-31; reference/FINDINGS_ISHMAEL_S8S9.md 5f-5j). `:min` and `:large` remain
        # selectable and their arithmetic is untouched.
        mktempdir() do tmpdir
            fixture(opts) = make_ice_mtile(tmpdir; Tsurf = 258.15, q_l = 0.0,
                                           rho_i = 1.0e-5, n_i = 0.0,
                                           extra_options = opts)[1]

            # THE DEFAULT IS `:local`, BITWISE.
            m_abs = fixture(Dict{Symbol,Any}())
            m_loc = fixture(Dict{Symbol,Any}(:ice_population_seed => :local))
            for t in 2:4
                Scythe.advance_column(m_abs, 1, t)
                Scythe.advance_column(m_loc, 1, t)
            end
            @test m_abs.var_np1 == m_loc.var_np1
            @test m_abs.expdot_n == m_loc.expdot_n

            # AND `:min` IS STILL SELECTABLE. On THIS fixture species 1 is dead at every
            # level, which is `:local`'s documented pure-dead fallback (asserted in (7)), so
            # the two agree bitwise HERE and for that reason alone. The mixed column of (7)
            # is where the default separates from `:min`, and that is asserted there.
            m_min = fixture(Dict{Symbol,Any}(:ice_population_seed => :min))
            for t in 2:4
                Scythe.advance_column(m_min, 1, t)
            end
            @test m_min.var_np1 == m_abs.var_np1
            @test m_min.expdot_n == m_abs.expdot_n

            # THE `:large` ARM on the same dead-species-below-T_0 fixture: the species
            # comes alive on a number ten orders of magnitude smaller. The `:min` column
            # is re-run to the same step first, so the two are read off one state.
            m_min = fixture(Dict{Symbol,Any}(:ice_population_seed => :min))
            Scythe.advance_column(m_min, 1, 2)
            S_min = m_min.mc_scratch[Threads.threadid()]
            hits = 0
            for i in eachindex(S_min.SRC_i1n)
                (S_min.i1q[i] > Scythe.ISHMAEL_QSMALL * S_min.rho_d[i] &&
                 S_min.i1n[i] <= 0.0) || continue
                hits += 1
                @test S_min.SRC_i1n[i] ≈ (S_min.i1q[i] / 10.0) / Scythe.ISHMAEL_M_MIN rtol = 1e-13
                @test S_min.SRC_i1a[i] > 0.0
            end
            @test hits > 0

            m_lrg2 = fixture(Dict{Symbol,Any}(:ice_population_seed => :large))
            Scythe.advance_column(m_lrg2, 1, 2)
            S_lrg = m_lrg2.mc_scratch[Threads.threadid()]
            checked = 0
            for i in eachindex(S_lrg.SRC_i1n)
                (S_lrg.i1q[i] > Scythe.ISHMAEL_QSMALL * S_lrg.rho_d[i] &&
                 S_lrg.i1n[i] <= 0.0) || continue
                checked += 1
                n, a, c = Scythe._ice_population_seed_large(S_lrg.i1q[i] / S_lrg.rho_d[i], 1)
                @test S_lrg.SRC_i1n[i] ≈ n * S_lrg.rho_d[i] / 10.0 rtol = 1e-13
                @test S_lrg.SRC_i1a[i] ≈ a * S_lrg.rho_d[i] / 10.0 rtol = 1e-13
                @test S_lrg.SRC_i1c[i] === S_lrg.SRC_i1a[i]
                # ORDERS OF MAGNITUDE fewer crystals for the identical mass...
                @test S_lrg.SRC_i1n[i] < 1.0e-8 * S_min.SRC_i1n[i]
                # ...and STILL no mass, no latent heat: the branch creates number only.
                @test S_lrg.SRC_i1q[i] == 0.0
                @test S_lrg.FRZ_NET[i] == 0.0
                @test S_lrg.ICE_R[i] == 0.0
            end
            @test checked == hits
            @test sum(view(m_lrg2.mc_water_stats, Scythe.MC_POP_SEED, :)) > 0.0
            @test sum(view(m_lrg2.mc_water_stats, Scythe.MC_POP_SEED, :)) <
                  1.0e-8 * sum(view(m_min.mc_water_stats, Scythe.MC_POP_SEED, :))
            # The census sees the SAME defect: the seeding changes the answer, not the
            # measurement of the defect that provoked it.
            @test maximum(view(m_lrg2.mc_water_stats, Scythe.MC_POP_MAX, :)) ==
                  maximum(view(m_min.mc_water_stats, Scythe.MC_POP_MAX, :))

            # THE POPULATION IS REALIZABLE AND FALLS: stepped forward, the species
            # acquires number, exchanges vapor, and has a FINITE fall speed -- 1 mm
            # particles at 920 kg/m^3 are the fastest the scheme can represent, so
            # "finite" is the claim that matters and is not automatic.
            m_run = fixture(Dict{Symbol,Any}(:ice_population_seed => :large))
            patch = m_run.tile
            s_ = m_run.mc_slots
            ncols = div(size(patch.physical, 1), m_run.model.grid_params.kDim)
            numbered = false
            alive = false
            fell = false
            for t in 1:12
                for c in 1:ncols
                    Scythe.advance_column(m_run, c, t)
                end
                Sr = m_run.mc_scratch[Threads.threadid()]
                @test all(isfinite, Sr.Vi1m)
                @test all(isfinite, Sr.Vi1n)
                @test all(isfinite, Sr.SRC_i1n)
                any(!iszero, Sr.invtau_i1) && (alive = true)
                any(!iszero, Sr.Vi1m) && (fell = true)
                Scythe.calcTendency(m_run)
                gridTransform!(patch)
                maximum(patch.physical[:, s_.i1_n, 1]) > 0.0 && (numbered = true)
            end
            @test numbered
            @test alive
            @test fell
            @test all(isfinite, m_run.var_np1)

            # An unrecognized seeding is refused rather than silently defaulted.
            m_bad = fixture(Dict{Symbol,Any}(:ice_population_seed => :medium))
            @test_throws ErrorException Scythe.advance_column(m_bad, 1, 2)
        end

        # ── (7) STAGE 3d: `:local`, THE THIRD SEEDING. Neither of the other two reads
        # the DEFECT: number-less mass is the negative lobe of the number moment's
        # spline ringing at cloud edges and gradients, so the crystals it lost are the
        # crystals of the same species one gridpoint away, not a size to be derived
        # from its own mass. The fixture is a column with a live population over part
        # of its depth and number-less mass over the rest, which is that shape.
        mktempdir() do tmpdir
            mixed(opts) = make_ice_mtile(tmpdir; Tsurf = 258.15, q_l = 0.0,
                                         rho_i = 1.0e-5, n_i = 5.0e4, r_i = 30.0e-6,
                                         n_i_levels = 11:16, extra_options = opts)[1]
            seeded(m) = sum(view(m.mc_water_stats, Scythe.MC_POP_SEED, :))

            m_loc = mixed(Dict{Symbol,Any}(:ice_population_seed => :local))
            m_min = mixed(Dict{Symbol,Any}(:ice_population_seed => :min))
            m_lrg = mixed(Dict{Symbol,Any}(:ice_population_seed => :large))
            # THE DEFAULT, ON A COLUMN WHERE THE THREE SEEDS DISAGREE: the optionless tree
            # is the `:local` one bitwise, and it is NOT the `:min` one. This is the
            # separation (6) cannot see, because there species 1 is dead everywhere and
            # `:local` falls back to `:min` by construction.
            m_def = mixed(Dict{Symbol,Any}())
            for m in (m_loc, m_min, m_lrg, m_def)
                m.mc_water_stats .= 0.0
                Scythe.advance_column(m, 1, 2)
            end
            @test m_def.var_np1 == m_loc.var_np1
            @test m_def.expdot_n == m_loc.expdot_n
            @test m_def.var_np1 != m_min.var_np1
            @test m_def.var_np1 != m_lrg.var_np1
            S = m_loc.mc_scratch[Threads.threadid()]
            @test all(S.Tk .< Scythe.T_0)

            # PRECONDITION: the fixture really is mixed -- dead points AND live ones.
            live(i) = S.i1q[i] > Scythe.ISHMAEL_QSMALL * S.rho_d[i] && S.i1n[i] > 0.0
            dead(i) = S.i1q[i] > Scythe.ISHMAEL_QSMALL * S.rho_d[i] && S.i1n[i] <= 0.0
            idx = eachindex(S.Tk)
            @test count(live, idx) > 0
            @test count(dead, idx) > 0

            hits = 0
            for i in idx
                dead(i) || continue
                hits += 1
                # The nearest live point of this species IN THIS COLUMN, found here by an
                # independent outward walk rather than by the code under test.
                j = 0
                for d in 1:length(idx)
                    if i - d >= first(idx) && live(i - d)
                        j = i - d
                        break
                    elseif i + d <= last(idx) && live(i + d)
                        j = i + d
                        break
                    end
                end
                @test j != 0
                nj = S.i1n[j]
                nl, al, cl = Scythe._ice_population_seed_local(S.i1q[i] / S.rho_d[i],
                                 S.i1q[j] / nj, S.i1a[j] / nj, S.i1c[j] / nj, 1)
                @test S.SRC_i1n[i] ≈ nl * S.rho_d[i] / 10.0 rtol = 1e-13
                @test S.SRC_i1a[i] ≈ al * S.rho_d[i] / 10.0 rtol = 1e-13
                @test S.SRC_i1c[i] ≈ cl * S.rho_d[i] / 10.0 rtol = 1e-13
                # ...which is `rho_empty/(m_loc tau_pop)` with `m_loc` the neighbour's
                # per-crystal mass, and that mass is a 30 um crystal: orders of magnitude
                # heavier than the 2 um sphere and lighter than the 1 mm one, so the
                # seeded number sits STRICTLY BETWEEN the two ends.
                Smn = m_min.mc_scratch[Threads.threadid()]
                Slg = m_lrg.mc_scratch[Threads.threadid()]
                @test Slg.SRC_i1n[i] < S.SRC_i1n[i] < Smn.SRC_i1n[i]
                @test S.SRC_i1n[i] > 1.0e2 * Slg.SRC_i1n[i]
                @test S.SRC_i1n[i] < 1.0e-2 * Smn.SRC_i1n[i]
                # STILL NUMBER ONLY: no mass, no latent heat, nothing in the rain.
                @test S.SRC_i1q[i] == 0.0
                @test S.FRZ_NET[i] == 0.0
                @test S.ICE_R[i] == 0.0
                @test S.ICE_NR[i] == 0.0
            end
            @test hits > 0
            # ONE-SIDEDNESS at the live points. Their `SRC_i1n` is NOT zero -- ISHMAEL's own
            # aggregation is taking number out of a live population there, which is the
            # physics working -- so the statement is that the SEEDING did not touch it: all
            # three arms leave the live points bit for bit identical.
            for i in idx
                live(i) || continue
                @test S.SRC_i1n[i] === m_min.mc_scratch[Threads.threadid()].SRC_i1n[i]
                @test S.SRC_i1n[i] === m_lrg.mc_scratch[Threads.threadid()].SRC_i1n[i]
                @test S.SRC_i1a[i] === m_min.mc_scratch[Threads.threadid()].SRC_i1a[i]
            end
            # The census sees the SAME defect in all three; only the seeded number moves.
            @test maximum(view(m_loc.mc_water_stats, Scythe.MC_POP_MAX, :)) ==
                  maximum(view(m_min.mc_water_stats, Scythe.MC_POP_MAX, :)) ==
                  maximum(view(m_lrg.mc_water_stats, Scythe.MC_POP_MAX, :))
            @test sum(view(m_loc.mc_water_stats, Scythe.MC_POP_PTS, :)) ==
                  sum(view(m_min.mc_water_stats, Scythe.MC_POP_PTS, :)) == Float64(hits)
            @test seeded(m_lrg) < seeded(m_loc) < seeded(m_min)

            # THE FALLBACK. A column in which species 1 is live NOWHERE is the pure-dead
            # case: there is no habit anywhere to inherit and `:local` is `:min` BITWISE.
            allmin = make_ice_mtile(tmpdir; Tsurf = 258.15, q_l = 0.0, rho_i = 1.0e-5,
                                    n_i = 0.0,
                                    extra_options = Dict{Symbol,Any}(
                                        :ice_population_seed => :min))[1]
            allloc = make_ice_mtile(tmpdir; Tsurf = 258.15, q_l = 0.0, rho_i = 1.0e-5,
                                    n_i = 0.0,
                                    extra_options = Dict{Symbol,Any}(
                                        :ice_population_seed => :local))[1]
            for t in 2:4
                Scythe.advance_column(allmin, 1, t)
                Scythe.advance_column(allloc, 1, t)
            end
            @test allmin.var_np1 == allloc.var_np1
            @test allmin.expdot_n == allloc.expdot_n

            # AND IT RUNS: stepped forward, the seeded species acquires number, exchanges
            # vapor, falls, and every moment stays finite.
            m_run = mixed(Dict{Symbol,Any}(:ice_population_seed => :local))
            patch = m_run.tile
            s_ = m_run.mc_slots
            ncols = div(size(patch.physical, 1), m_run.model.grid_params.kDim)
            numbered = false
            alive = false
            fell = false
            for t in 1:12
                for c in 1:ncols
                    Scythe.advance_column(m_run, c, t)
                end
                Sr = m_run.mc_scratch[Threads.threadid()]
                @test all(isfinite, Sr.SRC_i1n)
                @test all(isfinite, Sr.Vi1m)
                any(!iszero, Sr.invtau_i1) && (alive = true)
                any(!iszero, Sr.Vi1m) && (fell = true)
                Scythe.calcTendency(m_run)
                gridTransform!(patch)
                maximum(patch.physical[:, s_.i1_n, 1]) > 0.0 && (numbered = true)
            end
            @test numbered
            @test alive
            @test fell
            @test all(isfinite, m_run.var_np1)
        end
    end

    # ── STAGE 3b: NO LIQUID IS CONVERTED TO ICE ABOVE THE FREEZING LEVEL ─────────
    #
    # TeX §Departures (e). ISHMAEL's collection kernels carry no temperature gate and the
    # wet-growth branch adds the collected liquid to the ice mass, on the understanding that
    # the melting rate will return it — free in a host whose freezing latent heat is gated to
    # T <= T_0, and not free here, where FRZ_NET is signed by the partition and the transfer
    # releases L_f wherever it happens. Above T_0 a crystal that collects liquid SHEDS it.
    @testset "ice: above T_0 the collected liquid is shed, not frozen" begin
        # A riming state ABOVE the melting level: cloud, rain and a LIVE ice population, all
        # at 290 K, which is where ISHMAEL's wet-growth branch runs.
        warm_rimer(tmpdir, opts) = make_ice_mtile(tmpdir; Tsurf = 290.0, q_l = 1.0e-3,
                                                 rho_i = 1.0e-5, n_i = 5.0e4,
                                                 rho_r = 1.0e-4, n_r = 1.0e3, ts = 1.0,
                                                 extra_options = opts)[1]
        mktempdir() do tmpdir
            m_on = warm_rimer(tmpdir, Dict{Symbol,Any}())                       # default: shed
            m_off = warm_rimer(tmpdir, Dict{Symbol,Any}(:ice_shed_above_t0 => false))
            Scythe.advance_column(m_on, 1, 2)
            Scythe.advance_column(m_off, 1, 2)
            S_on = m_on.mc_scratch[Threads.threadid()]
            S_off = m_off.mc_scratch[Threads.threadid()]
            @test all(S_on.Tk .> Scythe.T_0)

            # PRECONDITION: the ungated kernel really does convert liquid to ice here, or
            # the statements below are vacuous. Riming is the ONLY liquid->ice channel left
            # above T_0 (homogeneous freezing is gated to T < T_0-35, Bigg to T < T_0-4,
            # and ishmael_ice_rain_riming routes dQRfzri/dQIfzri/dNfzri to zero there), so
            # a cloud debit above freezing IS the wet-growth transfer.
            @test minimum(S_off.ICE_C) < 0.0
            # ...and it releases L_f where it happens. `FRZ_NET` itself stays negative in
            # this column because the melt credit dominates it; the statement is that the
            # riming transfer PUSHES IT UP, which is exactly the loan the departure removes.
            @test maximum(S_off.FRZ_NET .- S_on.FRZ_NET) > 0.0

            # WITH THE DEPARTURE: no mass leaves the cloud or the rain to the ice.
            @test all(S_on.ICE_C .== 0.0)             # nothing debits the cloud at all
            @test all(S_on.ICE_R .>= 0.0)             # the rain only GAINS (melt credit)
            # ...so FRZ_NET above T_0 contains melting alone, and melting is negative.
            @test all(S_on.FRZ_NET .<= 0.0)
            @test minimum(S_on.FRZ_NET) < 0.0         # the melt is live, not switched off
            # The ice mass source carries no riming gain either: the only mass channel left
            # for a species above T_0 is melting, which removes.
            @test all(S_on.SRC_i1q .<= 0.0)
            @test minimum(S_on.SRC_i1q) < 0.0
            # And the axes grow on no rime.
            @test all(S_on.SRC_i1a .<= S_off.SRC_i1a .+ 1.0e-30)

            # THE LOOP IS THE DIFFERENCE: melting still carries the sensible heat of the
            # liquid that struck the crystal, so the collection kernels DID run — the melt
            # rate is not the shed-free one.
            @test S_on.SRC_i1q != S_off.SRC_i1q
        end

        # BITWISE INERT BELOW T_0: the branch is a strict `temp > T_0` state test, exactly
        # like the melt's, so a subfreezing riming column is unchanged to the last bit.
        mktempdir() do tmpdir
            function run_cold(opts)
                m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 258.15, q_l = 1.0e-3,
                                                  rho_i = 1.0e-5, n_i = 5.0e4,
                                                  rho_r = 1.0e-4, n_r = 1.0e3, ts = 1.0,
                                                  extra_options = opts)
                for t in 2:4
                    Scythe.advance_column(m, 1, t)
                end
                return m
            end
            m_on = run_cold(Dict{Symbol,Any}())
            m_off = run_cold(Dict{Symbol,Any}(:ice_shed_above_t0 => false))
            @test m_on.expdot_n == m_off.expdot_n
            @test m_on.var_np1 == m_off.var_np1
            S_on = m_on.mc_scratch[Threads.threadid()]
            @test all(S_on.Tk .< Scythe.T_0)
            @test minimum(S_on.ICE_C) < 0.0           # riming IS running down here
        end
    end

    @testset "ice: each moment sediments at its own weighted speed" begin
        # Mass falls faster than number (size sorting), and the a/c volume moments ride the
        # mass-weighted speed. The number/mass ratio therefore CANNOT stay fixed the way it
        # does under transport alone -- which is the point of giving each moment its own flux.
        mktempdir() do tmpdir
            m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 258.15, q_l = 0.0,
                                              rho_i = 1.0e-4, n_i = 2.0e4, r_i = 80.0e-6)
            Scythe.advance_column(m, 1, 2)
            S = m.mc_scratch[Threads.threadid()]
            @test all(S.Vi1m .< 0.0)                     # downward
            @test all(S.Vi1n .< 0.0)
            @test all(abs.(S.Vi1m) .> abs.(S.Vi1n))      # mass-weighted is the faster
            @test all(abs.(S.Vi1m) .<= 25.0)             # the ported cap
            # The two flux divergences are genuinely different fields.
            @test S.F_i1q_z != S.F_i1n_z
            # The empty species produce EXACT zeros and skip their spline fits entirely.
            @test all(S.F_i2q_z .== 0.0)
            @test all(S.F_i3c_z .== 0.0)
        end
    end

end
