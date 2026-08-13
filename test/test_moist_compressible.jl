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

    # ──────────────────────────────────────────────
    # 4b. Regime-blended vapor retrieval — WRITTEN BEFORE THE IMPLEMENTATION
    # ──────────────────────────────────────────────
    # The vapor is retrievable two ways, and the two ways fail in DISJOINT regimes
    # (reference/HANDOFF_VAPOR_RETRIEVAL.md; both measured sweeps are recorded in the
    # header of benchmarks/vapor_blend_diagnostic.jl):
    #
    #   res_rho_t = rho_t - rho_d - rho_c - rho_r    the DENSITY-BUDGET residual. Cancels
    #                                                catastrophically IN CLOUD, where
    #                                                rho_c/rho_w reaches 1.0011.
    #   res_qss   = Q_ss + rho_vs(T, p)              the SUPERSATURATION residual. Cancels
    #                                                catastrophically IN DRY AIR, where
    #                                                rho_v -> 0 forces Q_ss -> -rho_vs.
    #
    # Neither wins outright — a straight swap to res_qss is a NET REGRESSION on the shipped
    # O01 default (1.11 % -> 2.61 % negative points, because the dry population dominates
    # there) while being 5.7x better in the NOPRECIP storm's cloud. So the retrieval is a
    # C¹ BLEND of the two, selected on the two variables that separate the populations by
    # ~8 decades in rho_liq:
    #
    #   w = w_cloud(rho_liq; l0, l1) * w_trust(s; t0, t1),      s = |Q_ss| / rho_vs
    #   rho_v = w*res_qss + (1 - w)*res_rho_t
    #
    # w_cloud smoothsteps UP in rho_liq (there IS condensate, so res_rho_t is the route that
    # cancels) and w_trust smoothsteps DOWN in s. `s` is exactly the conditioning number of
    # the res_qss route — the ratio of the difference to the terms being differenced — and it
    # is SELF-PROTECTING: when Q_ss detaches from the density budget (the POSITIVITY=1 run
    # reaches s ~ 2e26) the weight goes to zero and the point is handed back to the density
    # residual with no special-case logic.
    #
    # THESE TESTSETS ARE THE SPECIFICATION, and they run before `src/` knows any of it. Each
    # is gated on an `isdefined` probe so that the missing Stage 2 API costs ONE countable
    # failing test per testset instead of erroring the rest of the file out from under the
    # suite. `@test_broken` is deliberately not used: these are requirements, not known bugs.

    @testset "vapor blend weight: limits, range and monotonicity" begin
        have = isdefined(Scythe, :_vapor_blend_weight)
        @test have                    # ← the Stage 2 gate; see the block comment above
        if have
            # The shipped defaults (physical_params :vapor_blend_l0/_l1/_t0/_t1). They are
            # passed EXPLICITLY here so that a later retuning of the defaults cannot silently
            # change what these assertions mean.
            l0, l1, t0, t1 = 1.0e-6, 1.0e-4, 2.0, 5.0
            W = (rho_liq, s) -> Scythe._vapor_blend_weight(rho_liq, s, l0, l1, t0, t1)

            # 1. Full trust in res_qss: thick cloud AND a supersaturation small compared with
            #    the saturation density it is differenced against. EXACTLY 1.0, not 1 - eps —
            #    the blend has to degenerate to a pure copy so the regime limits below can be
            #    asserted with ===.
            for rho_liq in (l1, 2.0e-4, 1.0e-3, 1.0), s in (0.0, 0.5, 1.0, t0)
                @test W(rho_liq, s) === 1.0
            end

            # 2. No condensate -> no trust, at ANY s. NOTE THE NEGATIVES: spline ringing puts
            #    rho_liq a few times 1e-9 BELOW zero in clear air routinely, and the clamp
            #    inside the smoothstep is what stops that from becoming a negative weight
            #    (i.e. an extrapolation past res_rho_t, not a blend).
            for rho_liq in (-1.0e-3, -3.0e-9, 0.0, 1.0e-9, l0),
                s in (0.0, 1.0, t0, 4.0, 1.0e10)
                @test W(rho_liq, s) === 0.0
            end

            # 3. Detached Q_ss -> no trust, however cloudy the point is.
            for s in (t1, 6.0, 1.0e10, 2.0e26), rho_liq in (0.0, 1.0e-5, 1.0e-3, 1.0)
                @test W(rho_liq, s) === 0.0
            end

            # 4. A weight is a weight: in [0,1] and finite everywhere, including at the two
            #    pathological inputs above.
            rls = collect(range(-1.0e-5, 1.0e-3; length = 257))
            ss = collect(range(0.0, 8.0; length = 257))
            @test all(0.0 <= W(rl, s) <= 1.0 for rl in rls, s in ss)
            @test all(isfinite(W(rl, s)) for rl in rls, s in ss)

            # 5. Monotone in both arguments, on grids fine enough to resolve the smoothstep
            #    interiors (the band is [l0, l1], four decades below the coarse grid's span).
            #    The tolerance is one rounding of the cubic x^2(3-2x), not slack: the exact
            #    function is monotone, and a non-monotone weight would mean the retrieval
            #    could move AWAY from the better-conditioned route as the point gets cloudier.
            rls_band = collect(range(-2.0e-6, 2.0e-4; length = 513))
            for s in (0.0, 1.0, 2.5, 3.5, 4.5)
                for grid in (rls, rls_band)
                    @test all(diff([W(rl, s) for rl in grid]) .>= -1.0e-16)
                end
            end
            for rl in (0.0, 5.0e-6, 5.0e-5, 1.0e-4, 1.0e-3)
                @test all(diff([W(rl, s) for s in ss]) .<= 1.0e-16)
            end
        end
    end

    @testset "vapor blend weight: C¹ across all four thresholds" begin
        # WHY C¹ AND NOT MERELY CONTINUOUS. rho_v feeds q_v, hence C_vt, R_m, C_pt and
        # gamma_m, hence the acoustic coefficient gamma_m*p/rho_t that the semi-implicit
        # solve linearizes about. A hard regime switch would put a JUMP in the sound speed
        # across a surface moving through the flow; a C⁰-only blend would put a kink in it.
        # The smoothstep x^2(3-2x) is flat at both ends, so the composite weight has a
        # continuous gradient across every one of the four thresholds.
        have = isdefined(Scythe, :_vapor_blend_weight)
        @test have
        if have
            l0, l1, t0, t1 = 1.0e-6, 1.0e-4, 2.0, 5.0
            W = (rho_liq, s) -> Scythe._vapor_blend_weight(rho_liq, s, l0, l1, t0, t1)
            dl = l1 - l0
            dt = t1 - t0

            # ── the rho_liq thresholds, probed at s = 0 where w_trust ≡ 1 so w = w_cloud ──
            hl = 1.0e-4 * dl
            dWl = rl -> (W(rl + hl, 0.0) - W(rl - hl, 0.0)) / (2.0 * hl)
            peak_l = 1.5 / dl                       # max |w'| of a unit smoothstep of width dl
            for knot in (l0, l1)
                # Flat AT the knot: both one-sided derivatives vanish there, which is what
                # "C¹ across the threshold" means for a clamped smoothstep.
                @test abs(dWl(knot)) < 1.0e-3 * peak_l
                # ... and no jump in the derivative in a neighbourhood of it.
                ds = [dWl(knot + j * 5.0e-4 * dl) for j in -5:5]
                @test maximum(abs.(diff(ds))) < 1.0e-2 * peak_l
            end
            # Interior: the closed form 6x(1-x)/Δ, not merely "something smooth".
            for x in (0.25, 0.5, 0.75)
                @test dWl(l0 + x * dl) ≈ 6.0 * x * (1.0 - x) / dl rtol = 1.0e-6
            end

            # ── the s thresholds, probed at rho_liq >> l1 where w_cloud ≡ 1 so w = w_trust ──
            ht = 1.0e-4 * dt
            dWt = s -> (W(1.0e-3, s + ht) - W(1.0e-3, s - ht)) / (2.0 * ht)
            peak_t = 1.5 / dt
            for knot in (t0, t1)
                @test abs(dWt(knot)) < 1.0e-3 * peak_t
                ds = [dWt(knot + j * 5.0e-4 * dt) for j in -5:5]
                @test maximum(abs.(diff(ds))) < 1.0e-2 * peak_t
            end
            for x in (0.25, 0.5, 0.75)
                # w_trust steps DOWN, so the sign is negative and x = (t1 - s)/Δ.
                @test dWt(t1 - x * dt) ≈ -6.0 * x * (1.0 - x) / dt rtol = 1.0e-6
            end
        end
    end

    @testset "vapor blend retrieval: the regime limits are exact copies" begin
        have = isdefined(Scythe, :vapor_retrieval_blend)
        @test have
        if have
            l0, l1, t0, t1 = 1.0e-6, 1.0e-4, 2.0, 5.0
            B = (Q_ss, rho_vs, rho_liq, res_rho_t) ->
                Scythe.vapor_retrieval_blend(Q_ss, rho_vs, rho_liq, res_rho_t, l0, l1, t0, t1)
            rho_vs = rho_v_sat(290.0, 900.0)        # ~1.4e-2 kg/m^3
            # The two candidates are deliberately DIFFERENT, by 4e-5 kg/m^3 — the order of
            # the in-cloud partition gap the sweep measured (median 1.0e-7, p99 2.0e-4) — so
            # "returns one of them exactly" is a real assertion rather than a tautology.
            #
            # AMENDED WITH THE CAP (Stage 2b): the separation must sit inside the saturator's
            # IDENTITY REGION, |c| <= dcap*rho_vs/2 = 0.01*rho_vs = 1.44e-4 kg/m^3, because
            # that is where the exact-copy guarantee lives and where the corrections that
            # constitute the fix actually are (median ~1.5e-3*rho_vs). It was 4e-4 before the
            # cap shipped, i.e. 2.8x outside; the assertions below prove the same property
            # about the same code path, on a separation the shipped default reaches.
            # `_blend_saturate`'s own testset covers the saturating tail.
            Q_ss = -2.0e-5                          # s = 0.0014 << t0: res_qss is trusted
            res_qss = Q_ss + rho_vs
            res_rho_t = rho_vs + 4.0e-5             # |res_qss - res_rho_t| = 6.0e-5 < 1.44e-4

            # 1. Thick cloud near saturation -> the supersaturation residual, EXACTLY.
            for rho_liq in (l1, 5.0e-4, 8.0e-3)
                @test B(Q_ss, rho_vs, rho_liq, res_rho_t) === res_qss
            end
            # 2. Condensate-free -> the density residual, EXACTLY, including at the small
            #    negative rho_liq that spline ringing makes routine in clear air.
            for rho_liq in (0.0, -3.0e-9, -1.0e-6, l0)
                @test B(Q_ss, rho_vs, rho_liq, res_rho_t) === res_rho_t
            end
            # 3. Detached Q_ss -> the density residual, EXACTLY, however cloudy the point is.
            #    Either sign: s is a magnitude.
            @test B(1.0e10 * rho_vs, rho_vs, 8.0e-3, res_rho_t) === res_rho_t
            @test B(-1.0e10 * rho_vs, rho_vs, 8.0e-3, res_rho_t) === res_rho_t

            # 4. THE BLEND IS NOT A CLAMP. `res_qss < 0` is ALGEBRAICALLY `s > 1` (measured
            #    min s over the res_qss-negative points: 1.0000 on both runs), and t0 = 2 sits
            #    ABOVE that ceiling, so a fully trusted point can and must import a NEGATIVE
            #    vapor. Negative water is a RESOLUTION DIAGNOSTIC (reference/
            #    FINDINGS_NEGATIVE_WATER_ATTRIBUTION.md): flooring it here would destroy the
            #    measurement and manufacture water on the way.
            #
            #    AMENDED WITH THE CAP (Stage 2b): a negative res_qss can only be imported
            #    EXACTLY where the correction is inside the identity region, so the density
            #    residual here is itself negative — which is the configuration the rescue
            #    actually runs in (NOPRECIP's rescued points are res_rho_t < 0 <= res_qss).
            #    It was `Q_neg = -1.5*rho_vs` against `res_rho_t = rho_vs + 4e-4` before,
            #    a correction of 2.1e-2 = 74x the whole cap; the property under test —
            #    s > 1 imports at FULL weight and the result is NEGATIVE, i.e. the retrieval
            #    is not a non-negativity constraint — is unchanged, and §6 below adds the
            #    statement that the SATURATOR does not rectify it either.
            res_rho_t_neg = -1.0e-4                 # the anvil configuration
            Q_neg = -2.0e-4 - rho_vs                # res_qss = -2.0e-4; s = 1.014 in (1, t0)
            @test abs(Q_neg) / rho_vs > 1.0         # ... genuinely past the res_qss ceiling
            @test abs(Q_neg) / rho_vs < t0          # ... and still at full trust
            @test B(Q_neg, rho_vs, 8.0e-3, res_rho_t_neg) === Q_neg + rho_vs
            @test B(Q_neg, rho_vs, 8.0e-3, res_rho_t_neg) < 0.0

            # 5. In between, it is the convex combination the weight function reports — no
            #    third representation of the vapor anywhere in the transition.
            #
            #    AMENDED WITH THE CAP (Stage 2b): `s` and the correction are not independent
            #    (s = Q/rho_vs, so a large s is a large res_qss), so the separation is now
            #    taken RELATIVE to res_qss — a fixed 4e-5 gap at every s — which keeps every
            #    point inside the identity region while sweeping exactly the same weights.
            #    Previously `res_rho_t = rho_vs + 4e-4` was shared with §1, which at s = 4.5
            #    made the correction 4.2e-3, 15x the cap.
            if isdefined(Scythe, :_vapor_blend_weight)
                for rho_liq in (5.0e-6, 2.0e-5, 8.0e-5), s in (0.0, 2.5, 3.5, 4.5)
                    Q = s * rho_vs
                    w = Scythe._vapor_blend_weight(rho_liq, s, l0, l1, t0, t1)
                    rrt = (Q + rho_vs) + 4.0e-5     # |c| <= 4.0e-5 < 1.44e-4 for every w
                    @test B(Q, rho_vs, rho_liq, rrt) ≈
                          (w * (Q + rho_vs)) + ((1.0 - w) * rrt) rtol = 1.0e-14
                end
            end

            # 6. THE CAP IS ODD, SO IT IS NOT A RECTIFIER. Push the same fully trusted,
            #    negative res_qss far outside the identity region: the result is SATURATED
            #    (it no longer equals res_qss) but it is still on the res_qss side of the
            #    density residual and still negative. A one-sided floor would show up here.
            Q_far = -1.5 * rho_vs                   # res_qss = -0.5*rho_vs; s = 1.5 < t0
            v_far = B(Q_far, rho_vs, 8.0e-3, res_rho_t_neg)
            @test v_far != Q_far + rho_vs                       # saturated, not copied
            @test v_far < res_rho_t_neg                         # ... on the res_qss side
            @test res_rho_t_neg - v_far < 0.02 * rho_vs         # ... within the shipped cap
            @test v_far < 0.0                                   # ... and still negative
            # The mirror image: the Q_ss whose res_qss sits the same distance on the OTHER
            # side of res_rho_t. Equal and opposite correction, so the cap has no sign bias.
            Q_mirror = 2.0 * res_rho_t_neg - Q_far - 2.0 * rho_vs
            @test abs(Q_mirror) / rho_vs < t0       # still fully trusted
            @test (B(Q_mirror, rho_vs, 8.0e-3, res_rho_t_neg) - res_rho_t_neg) ≈
                  -(v_far - res_rho_t_neg) rtol = 1.0e-14
        end
    end

    # ──────────────────────────────────────────────
    # 4c. The blend's C¹ CORRECTION SATURATOR (Stage 2b)
    # ──────────────────────────────────────────────
    # WHY THIS EXISTS. The uncapped blend of §4b DETONATES the NOPRECIP storm
    # (SCYTHE_O01_PRECIP=0 SCYTHE_O01_VAPOR=blend), at the identical step 6828 in two runs
    # differing only in the last bits of the state — a threshold crossing, not noise. The
    # mechanism is a MISSING GUARD, not a bad weight:
    #
    #   * the partition gap `res_qss - res_rho_t` is a property of the Q_ss splitting and is
    #     IDENTICAL in a :blend run and a :residual run, ~1e-4 kg/m^3 absolute;
    #   * but it is UNBOUNDED RELATIVE to rho_vs, and rho_vs is not a constant: in the rising
    #     anvil the air cools, rho_vs collapses, and gap/rho_vs climbed 0.08 (t = 1980 s) ->
    #     0.46 (t = 2040 s) -> ~0.8 through the mature phase;
    #   * `w_trust` is blind to that. Its variable s = |Q_ss|/rho_vs is the SUPERSATURATION
    #     magnitude, measured 0.02-0.46 at the worst-detached points, so w_trust ≡ 1 and the
    #     guard fired on ZERO points in 3600 s — and during growth s is ANTI-CORRELATED with
    #     detachment. `s` is not a detachment measure;
    #   * `res_qss - res_rho_t = -tau_qss*QSSREL` is an IDENTITY, not a bound. It relates the
    #     gap to the reconciliation RATE and constrains neither.
    #
    # The bound is therefore imposed directly, on the correction, relative to rho_vs:
    # |rho_v - res_rho_t| <= dcap*rho_vs with dcap = 0.02 shipped. That was validated
    # CAUSALLY: 0 % of points touched before t ~ 1680 s, ~10 % of active cloud after, the
    # detonation gone, the stability ladder matching the :residual control, and the median
    # correction (~1.5e-3*rho_vs, an order of magnitude under the cap) untouched.
    #
    # It must be C¹ for the same reason the weight is (rho_v -> q_v -> gamma_m -> the acoustic
    # coefficient the semi-implicit solve linearizes about), and EXACTLY the identity on the
    # small corrections, because those are the fix. Both requirements together FORCE the
    # junction to half the cap: matching value and slope of the Huber tail g(x) = m - k/x to
    # the identity at x = a gives k = a(m - a) and k = a^2 simultaneously, i.e. a = m/2.
    @testset "vapor blend saturator: identity region, bound, C¹, odd" begin
        have = isdefined(Scythe, :_blend_saturate)
        @test have
        if have
            F = Scythe._blend_saturate
            rho_vs = rho_v_sat(290.0, 900.0)        # ~1.4393e-2 kg/m^3
            m = 0.02 * rho_vs                       # the shipped cap, 2.8787e-4
            a = 0.5 * m                             # 1.4393e-4

            # 1. THE IDENTITY REGION IS EXACT, not accurate. The corrections that constitute
            #    the fix live here (median ~1.5e-3*rho_vs = 2.2e-5, an order of magnitude
            #    under the cap), so `===`: they must pass through with no rounding at all, which is what
            #    keeps the regime limits of §4b bit-for-bit copies.
            for c in (0.0, -0.0, 1.0e-18, -1.0e-18, 1.0e-6, -1.0e-6, 2.2e-5, -2.2e-5,
                      0.5 * a, -0.5 * a, prevfloat(a), -prevfloat(a), a, -a)
                @test F(c, m) === c
            end
            # ... on a fine sweep of the whole inner half, at three cap sizes.
            for mm in (m, 1.0e-3, 1.0e-8), c in range(-0.5 * mm, 0.5 * mm; length = 401)
                @test F(c, mm) === c
            end

            # 2. THE BOUND IS NEVER EXCEEDED, and is approached ASYMPTOTICALLY — no corner.
            #    f -> m - m^2/(4|c|) in exact arithmetic, so the cap is a supremum rather than
            #    a value the map attains. The bound asserted is `<=`, which is what the
            #    retrieval needs and what holds in floating point: past |c| ~ m/(4*eps) the
            #    subtrahend falls below one ulp of m and the tail ROUNDS ONTO the cap. That is
            #    the correct rounding of a strict supremum, not a corner — the derivative is
            #    still ~m^2/(4c^2) and vanishing, and the limit is approached from below.
            for c in (nextfloat(a), 2.0 * a, 10.0 * m, 1.0e3 * m, 1.0e12 * m, 1.0e300)
                @test abs(F(c, m)) <= m
                @test abs(F(-c, m)) <= m
            end
            for c in (nextfloat(a), 2.0 * a, 10.0 * m, 1.0e3 * m, 1.0e12 * m)
                @test abs(F(c, m)) < m                # resolvable: strictly under the cap
                @test abs(F(-c, m)) < m
            end
            @test F(1.0e12 * m, m) ≈ m rtol = 1.0e-11
            # ... and everywhere on a wide sweep spanning both regions.
            cs = sort(vcat(range(-50.0 * m, 50.0 * m; length = 2001),
                           [-1.0e6 * m, 1.0e6 * m]))
            @test all(abs(F(c, m)) <= m for c in cs)
            # The closed form itself, on the tail: sign(c)*(m - (m/2)^2/|c|).
            for c in (1.3 * a, 2.0 * a, 7.0 * a, 1.0e4 * a)
                @test F(c, m) ≈ m - (a * a) / c rtol = 1.0e-15
                @test F(-c, m) ≈ -(m - (a * a) / c) rtol = 1.0e-15
            end

            # 3. C¹ AT THE JUNCTION ±m/2, by finite difference. The one-sided slopes must
            #    BOTH be 1 there: the identity side by construction, the tail side because
            #    g'(a) = a^2/a^2 = 1 is exactly what fixed the junction at half the cap.
            #    A C⁰ clamp — the diagnostic form this replaces — fails this: its slope drops
            #    from 1 to 0 across |c| = m.
            h = 1.0e-6 * a
            for knot in (a, -a)
                left  = (F(knot, m) - F(knot - h, m)) / h
                right = (F(knot + h, m) - F(knot, m)) / h
                @test left ≈ 1.0 rtol = 1.0e-5
                @test right ≈ 1.0 rtol = 1.0e-5
                @test abs(right - left) < 1.0e-4          # no jump in the derivative
            end
            # No jump anywhere: the FD derivative is continuous across a neighbourhood of the
            # junction, and matches the closed form a^2/c^2 on the tail.
            for x in (1.5, 2.0, 4.0, 20.0)
                c = x * a
                fd = (F(c + h, m) - F(c - h, m)) / (2.0 * h)
                @test fd ≈ (a * a) / (c * c) rtol = 1.0e-6
            end
            ds = [(F(a + (j + 1) * 1.0e-3 * a, m) - F(a + j * 1.0e-3 * a, m)) / (1.0e-3 * a)
                  for j in -50:50]
            @test maximum(abs.(diff(ds))) < 1.0e-2        # slope walks, never jumps

            # 4. ODD, so the cap cannot rectify the sign of the correction — the property
            #    that keeps it a bound on the DISTANCE BETWEEN TWO REPRESENTATIONS rather
            #    than a non-negativity constraint on the vapor (which reference/
            #    FINDINGS_NEGATIVE_WATER_ATTRIBUTION.md forbids).
            for c in vcat(collect(range(1.0e-9, 50.0 * m; length = 997)), [1.0e6 * m])
                @test F(-c, m) === -F(c, m)
            end

            # 5. MONOTONE — a non-monotone saturator would make rho_v move the WRONG WAY as
            #    the two representations separate further.
            @test all(diff([F(c, m) for c in cs]) .>= 0.0)

            # 6. dcap = Inf IS THE UNCAPPED BLEND, bitwise: the identity region is the whole
            #    line. dcap = 0 is the density residual: the correction is annihilated.
            for c in (0.0, 1.0e-9, -1.0e-9, 1.0, -1.0, 1.0e300, -1.0e300)
                @test F(c, Inf) === c
                @test F(c, 0.0) == 0.0
            end
        end
    end

    @testset "vapor blend retrieval: the cap through vapor_retrieval_blend" begin
        have = isdefined(Scythe, :vapor_retrieval_blend) && isdefined(Scythe, :_blend_saturate)
        @test have
        if have
            l0, l1, t0, t1 = 1.0e-6, 1.0e-4, 2.0, 5.0
            rho_vs = rho_v_sat(290.0, 900.0)
            dcap = 0.02                              # the shipped physical_params default
            B = (Q_ss, rho_liq, res_rho_t, args...) ->
                Scythe.vapor_retrieval_blend(Q_ss, rho_vs, rho_liq, res_rho_t,
                                             l0, l1, t0, t1, args...)

            # 1. THE DEFAULT IS THE SHIPPED CAP. The trailing argument's default and
            #    `physical_params[:vapor_blend_dcap]` are the same number, so the unit-scale
            #    calls above and the model are testing one configuration.
            @test B(-1.0e-3, 8.0e-3, rho_vs + 5.0e-2) ===
                  B(-1.0e-3, 8.0e-3, rho_vs + 5.0e-2, dcap)

            # 2. dcap = Inf RECOVERS THE UNCAPPED BLEND BITWISE, at every weight — including
            #    at corrections far outside any finite cap. This is the A/B lever: the
            #    detonating configuration is still reachable, exactly.
            for rho_liq in (0.0, 5.0e-6, 2.0e-5, 8.0e-5, 8.0e-3), s in (0.0, 0.5, 2.5, 4.5)
                Q = s * rho_vs
                w = Scythe._vapor_blend_weight(rho_liq, s, l0, l1, t0, t1)
                for rrt in (rho_vs + 4.0e-5, rho_vs + 4.0e-1, -1.0e-3)
                    uncapped = w == 0.0 ? rrt :
                               w == 1.0 ? Q + rho_vs :
                               (w * (Q + rho_vs)) + ((1.0 - w) * rrt)
                    @test B(Q, rho_liq, rrt, Inf) === uncapped
                end
            end

            # 3. THE BOUND HOLDS POINTWISE, over a sweep that deliberately includes
            #    separations of the order the NOPRECIP anvil reached (0.46*rho_vs, 23x the
            #    cap) and the gross detachment w_trust is retained for.
            for rho_liq in (-3.0e-9, 0.0, 5.0e-6, 2.0e-5, 8.0e-5, 8.0e-3),
                s in (0.0, 0.5, 1.0, 1.5, 2.5, 3.5, 4.5, 6.0, 1.0e10, 2.0e26),
                d in (-0.46 * rho_vs, -0.02 * rho_vs, -1.0e-7, 0.0,
                      1.0e-7, 0.02 * rho_vs, 0.46 * rho_vs, 5.0)

                Q = s * rho_vs
                rrt = (Q + rho_vs) - d               # so the raw separation is exactly d
                v = B(Q, rho_liq, rrt)
                @test abs(v - rrt) <= dcap * rho_vs
                @test isfinite(v)
            end

            # 4. GROSS DETACHMENT IS STILL BITWISE res_rho_t. The cap does not displace
            #    `w_trust`'s one retained job: above t1 the weight is identically zero, so the
            #    point is handed back to the density residual with no correction at all — not
            #    a capped one.
            for rrt in (rho_vs, -1.0e-3, 1.0e-2, 0.0)
                @test B(1.0e10 * rho_vs, 8.0e-3, rrt) === rrt
                @test B(-2.0e26 * rho_vs, 8.0e-3, rrt) === rrt
                @test B(6.0 * rho_vs, 8.0e-3, rrt) === rrt
            end

            # 5. dcap = 0 DEGENERATES TO :residual, at every weight. Not a special case in the
            #    code — the identity region collapses to the origin and the tail evaluates to
            #    ±0.0 — but it is the statement that the cap family spans both endpoints.
            for rho_liq in (0.0, 2.0e-5, 8.0e-3), s in (0.0, 1.5, 4.5)
                @test B(s * rho_vs, rho_liq, rho_vs + 4.0e-4, 0.0) === rho_vs + 4.0e-4
            end

            # 6. THE CAP SCALES WITH rho_vs, not with an absolute density. That is the whole
            #    point: the failure was a gap that was small in kg/m^3 and large relative to a
            #    COLLAPSING rho_vs in the rising anvil.
            for rvs in (1.0e-3, 1.0e-2, 3.0e-2)
                # s = 0.5 (full trust), res_qss = 0.5*rvs, res_rho_t = -0.5*rvs, so the raw
                # correction is 1.0*rvs — 100x the cap at every one of these scales.
                v = Scythe.vapor_retrieval_blend(-0.5 * rvs, rvs, 8.0e-3, -0.5 * rvs,
                                                 l0, l1, t0, t1, dcap)
                @test abs(v + 0.5 * rvs) <= dcap * rvs
                @test abs(v + 0.5 * rvs) > 0.9 * dcap * rvs     # saturated, not merely small
            end
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
        # discretely the prognostic Q_ss can detach from the density budget, and this is the
        # instantaneous form of the reconciliation qss_relaxation applies on tau_qss.
        Tk = 285.0; p_hPa = 900.0; ts = 0.1
        N_r = 1.0e-3
        rho_vs = rho_v_sat(Tk, p_hPa)
        rho_d = (100.0 * p_hPa - Rv * Tk * rho_vs) / (Rd * Tk)
        Q_s = Scythe.Q_s_energy(Tk, 100.0 * p_hPa, rho_d, rho_vs / rho_d, 1.0e-3)

        # (i) DRY AIR with a spuriously positive Q_ss: zero rates AND zero nucleation. The
        #     second half is the load-bearing one — if S came from the raw Q_ss the Twomey
        #     branch would open (invtau_c > 0) and the clipped, negative drive would then
        #     evaporate cloud that does not exist and manufacture vapor.
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

        # (ii) INERT where the two representations agree — which is every consistent state,
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

        # (v) NEGATIVE retrieved vapor (a partition error, not a physical state): the
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
            # (p ~ 1e5 Pa, E_t ~ 2e8 J/m^3, densities ~ 1)
            scales = Dict(1 => 1.0e5, 2 => 1.0, 3 => 1.0, 4 => 1.0, 5 => 1.0,
                          6 => 2.0e8, 7 => 1.0e-2, 8 => 1.0, 9 => 1.0)
            for v in 1:9
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

    @testset "qss reconciliation drives Q_ss to the diagnosed supersaturation" begin
        # Q_ss is redundant now that rho_c is prognostic: the water masses already fix
        # the vapor. qss_relaxation is what keeps the prognostic tracker and the
        # mass-implied value from drifting apart (and it is thermodynamically INERT,
        # because retrieve_temperature does not read Q_ss at all).
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

    @testset "clamp_water! is a conservative phase change, not a mass source" begin
        # The floor is opt-in (options[:clamp_water]); by default clamp_water! only
        # MEASURES. These two testsets exercise the floor itself, so they enable it.
        # The positivity floor moves the deficit between the prognostic condensate and
        # the RESIDUAL vapor, so total water and E_t are untouched and the retrieval
        # supplies exactly the latent heat of the implied phase change. This is the
        # property that makes flooring negative water legitimate rather than a fudge.
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
        # Rule 2: rho_c + rho_r may not exceed rho_w (i.e. the residual vapor may not go
        # negative). The excess comes out of cloud first, then rain.
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
            # Vapor is exactly zero, not negative: the cap binds
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
                    for nm in ("rho_d", "rho_t", "u", "w", "Q_ss", "rho_r", "rho_c")
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
    # 8b. Regime-blended vapor retrieval, ON THE MODEL — still written before src/ knows it
    # ──────────────────────────────────────────────
    # The unit-level specification is in section 4b. These four testsets pin the parts that
    # only a running column can show: that the blend is what an unconfigured model runs (and
    # that `:residual` still opts out of it), that the reconciliation is anchored to the
    # DENSITY residual and not to the blended vapor, that the resting fixed point survives,
    # and that the blend is a PARTITION rather than a mass source.

    """
    Small RiRk moist patch on a saturated cloudy base, with a deterministic broadband w
    seed. `retrieval` is the `options[:vapor_retrieval]` value, or `nothing` to leave the
    option ABSENT — absent and `:blend` must be the same run bitwise (see the testset
    below), since `:blend` is the default. The
    geometry is RiRk in both directions because the mish points are then Gauss nodes on
    both legs, which is what makes the exact mass quadrature in the conservation testset
    possible (same argument as the positivity testset's `cellw`).
    """
    function vapor_blend_rirk(tmpdir, tag; retrieval = nothing)
        vars = Dict(v => i for (i, v) in enumerate(Scythe.MC_VARS))
        scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
        side_bc = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))
        wall_bc = merge(scalar_bc, Dict("w" => DirichletBC()))
        gp = GridParameters(geometry = "RiRk",
            iMin = 0.0, iMax = 8.0e3, num_cells_i = 4,
            kMin = 0.0, kMax = 4.0e3, num_cells_k = 8,
            BCL = side_bc, BCR = side_bc, BCB = wall_bc, BCT = wall_bc,
            vars = vars)
        ref_file = joinpath(tmpdir, "vapor_blend_$(tag).ref")
        opts = Dict{Symbol,Any}(:semiimplicit => true,
                                :exact_reference_state => true,
                                :precipitation => false)
        # ABSENT is not a synonym for either value here: "no option" is the state every
        # existing configuration file is in, and it is the one that has to stay bitwise
        # equal to the shipped default (`:blend` since 2026-07-29).
        retrieval === nothing || (opts[:vapor_retrieval] = retrieval)
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

    """Advance every column of an RiRk vapor-blend patch for `nsteps` steps."""
    function vapor_blend_run!(mtile, patch, gp, nsteps)
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

    @testset "vapor retrieval: an absent option is bitwise :blend, and :residual opts out" begin
        # The gate on the DEFAULT (flipped to `:blend` 2026-07-29, Stage 5). Two claims, and
        # the second is what keeps the first from being vacuous:
        #
        #   (a) leaving the key ABSENT — the state of every configuration file that says
        #       nothing — is the blend to the LAST BIT, not "agrees to 1e-14".
        #   (b) `:residual` still selects the pre-blend density-budget route, and on THIS
        #       state that is a genuinely different run. The base is saturated with
        #       rho_c ~ 1.1e-3 kg/m^3, three decades above l1 = 1e-4, so w_cloud = 1 and the
        #       blend is fully active; asserting equality here would be asserting the option
        #       does nothing.
        have = isdefined(Scythe, :vapor_retrieval_blend)
        @test have                   # ← the Stage 2 gate
        mktempdir() do tmpdir
            mA, pA, moA, gpA = vapor_blend_rirk(tmpdir, "absent")
            mB, pB, moB, gpB = vapor_blend_rirk(tmpdir, "blend"; retrieval = :blend)
            mR, pR, moR, gpR = vapor_blend_rirk(tmpdir, "residual"; retrieval = :residual)
            @test !haskey(moA.options, :vapor_retrieval)
            @test moB.options[:vapor_retrieval] === :blend
            @test moR.options[:vapor_retrieval] === :residual

            nsteps = 12
            vapor_blend_run!(mA, pA, gpA, nsteps)
            vapor_blend_run!(mB, pB, gpB, nsteps)
            vapor_blend_run!(mR, pR, gpR, nsteps)

            @test all(isfinite.(pA.physical))
            @test all(isfinite.(pR.physical))
            # The seed actually did something — otherwise "bitwise equal" is vacuous.
            @test maximum(abs.(pA.physical[:, gpA.vars["Q_ss"], 1])) > 0.0
            # (a) absent === :blend, bitwise
            @test pA.physical == pB.physical
            @test pA.spectral == pB.spectral
            # (b) :residual is a different integration on this cloudy state, bounded by the
            #     blend's own correction cap.
            #
            #     This bound USED to be "within 1% of |Q_ss|", i.e. the two retrievals were a
            #     nearby pair. That is no longer true, and the reason is load-bearing: the
            #     condensation closure now reads the retrieved vapor through the DRIVE clip
            #     `min(Q_ss, rho_v - rho_vs)` (see qss_condensation_rates), where before it
            #     read it only through a vapor ceiling that was almost never active. So the
            #     retrieval choice is now FIRST-ORDER in the rate, and in near-saturated air
            #     — where |Q_ss| is small and the partition gap is not — the gap dominates the
            #     drive outright. Here |Q_ss| ~ 2.3e-7 while the blend may open a gap of
            #     dcap*rho_vs ~ 2.9e-4, three decades larger, so the two runs' Q_ss fields
            #     diverge by O(|Q_ss|) and a 1% bound is unmeetable BY CONSTRUCTION.
            #
            #     What still holds — and is the honest statement of "a different integration,
            #     not a different equation set" — is that the divergence stays inside the
            #     correction the blend is permitted to make.
            @test pA.physical != pR.physical
            qi = gpA.vars["Q_ss"]
            dq = maximum(abs.(pA.physical[:, qi, 1] .- pR.physical[:, qi, 1]))
            @test dq > 0.0
            z_col = Scythe.getGridpoints(pA)[1:gpA.kDim, end]
            rvs_max = maximum(saturated_cloudy_column_mc(z_col).rho_v)
            dcap = get(moA.physical_params, :vapor_blend_dcap, 0.02)
            @test dq <= dcap * rvs_max
        end
    end

    @testset "vapor retrieval: the reconciliation stays anchored to res_rho_t" begin
        # THE ONE PLACE THE BLENDED VAPOR MUST NOT BE USED. qss_relaxation pulls the
        # prognostic Q_ss toward the vapor the WATER MASSES imply:
        #
        #     QSSREL = -(Q_ss - (rho_v - rho_vs)) / tau_qss
        #
        # If `rho_v` there is the BLENDED vapor then wherever the blend trusts res_qss the
        # term reads -(Q_ss - ((Q_ss + rho_vs) - rho_vs))/tau ≡ 0. It SELF-ANNIHILATES, and
        # the single mechanism tying Q_ss to the density budget vanishes exactly where it is
        # load-bearing — the POSITIVITY=1 run reaches |Q_ss|/rho_vs ~ 2e26 WITH the
        # reconciliation running. So slot 7 is fed S.res_rho_t, never S.rho_v.
        #
        # THE DISCRIMINATOR. On a consistent-qss saturated base at rest with condensation
        # switched off, slot 7's ENTIRE tendency is QSSREL: u = w = 0 so ADV = 0 and div = 0,
        # which kills -Q_ss*div and every term of SATF; the diffusivities are zero so
        # QDOT_TH = 0; and Qdot = Qdot_r = 0. Perturb Q_ss by dQ on a base whose cloud is
        # thick (rho_liq ~ 9e-4 >> l1) and whose s = dQ/rho_vs stays under t0, so w = 1, and
        # the two candidate anchors are 100 % apart:
        #
        #     anchored to res_rho_t (= rho_vs) : -(dQ - 0) /tau = -dQ/tau
        #     anchored to the blend (= rho_vs + dQ) : -(dQ - dQ)/tau = 0
        #
        # The assertion is on the tendency itself, so it holds whatever helper Stage 2
        # factors the retrieval into.
        tau = 10.0
        dQ = 5.0e-3        # kg/m^3; s = dQ/rho_vs runs 0.35 (bottom) to 0.65 (top) < t0 = 2
        function qssrel_probe(tmpdir, mode)
            mtile, patch, model, col = make_mc_mtile(tmpdir; consistent_qss = true,
                                                     q_l = 1.0e-3, tau_qss = tau)
            # Not physics — a diagnostic switch, and the whole point here: with the phase
            # change off, slot 7 carries nothing but the reconciliation.
            model.options[:condensation] = false
            mode === nothing || (model.options[:vapor_retrieval] = mode)
            gp = model.grid_params
            qss_i = gp.vars["Q_ss"]
            patch.physical[:, qss_i, 1] .= dQ
            spectralTransform!(patch)
            gridTransform!(patch)
            ncols = div(size(patch.physical, 1), gp.kDim)
            for c in 1:ncols
                Scythe.advance_column(mtile, c, 1)
            end
            # Read Q_ss BACK from the mish: the spline round-trip is what the kernel saw,
            # so the expected QSSREL is pointwise exact rather than "dQ up to the fit".
            return (Q_ss = copy(patch.physical[:, qss_i, 1]),
                    e7 = copy(mtile.expdot_n[:, qss_i]), col = col)
        end

        # The shipped route first: this is the calibration of the assertion, and it must
        # pass BEFORE Stage 2 as well as after.
        mktempdir() do tmpdir
            r = qssrel_probe(tmpdir, :residual)
            @test maximum(abs.(r.e7 .+ (r.Q_ss ./ tau))) < 1.0e-6 * dQ / tau
            @test minimum(abs.(r.e7)) > 0.5 * dQ / tau          # nowhere near annihilated
        end

        have = isdefined(Scythe, :vapor_retrieval_blend) &&
               isdefined(Scythe, :_vapor_blend_weight)
        @test have                   # ← the Stage 2 gate
        if have
            l0, l1, t0, t1 = 1.0e-6, 1.0e-4, 2.0, 5.0
            mktempdir() do tmpdir
                r = qssrel_probe(tmpdir, :blend)
                # 1. The state really is in the full-trust regime, so the blend really does
                #    differ from res_rho_t here. Without this the assertion below could pass
                #    for the wrong reason (a blend that quietly returned res_rho_t).
                k = div(length(r.col.z), 2)
                rho_vs_k = rho_v_sat(r.col.Tk[k], r.col.p_Pa[k] / 100.0)
                rho_liq_k = r.col.rho_c[k]
                s_k = dQ / rho_vs_k
                @test rho_liq_k > l1
                @test s_k < t0
                @test Scythe._vapor_blend_weight(rho_liq_k, s_k, l0, l1, t0, t1) === 1.0
                # AMENDED WITH THE CAP (Stage 2b): dQ = 5e-3 is ~35x the shipped cap
                # 0.02*rho_vs_k, so the shipped retrieval SATURATES here rather than copying
                # res_qss. Both halves of the original claim are kept and separated: the
                # uncapped blend still returns res_qss bit-for-bit, and the shipped one still
                # makes a real import (it is NOT quietly returning res_rho_t, which is the
                # failure mode this probe exists to exclude).
                @test Scythe.vapor_retrieval_blend(dQ, rho_vs_k, rho_liq_k, rho_vs_k,
                                                   l0, l1, t0, t1, Inf) === dQ + rho_vs_k
                v_cap = Scythe.vapor_retrieval_blend(dQ, rho_vs_k, rho_liq_k, rho_vs_k,
                                                     l0, l1, t0, t1)
                @test v_cap > rho_vs_k                          # a real import
                @test v_cap - rho_vs_k <= 0.02 * rho_vs_k       # ... bounded by the cap
                @test v_cap - rho_vs_k > 0.9 * 0.02 * rho_vs_k  # ... and saturated
                # 2. ... and the reconciliation is unmoved by that: still -dQ/tau, still not
                #    the self-annihilating zero.
                @test maximum(abs.(r.e7 .+ (r.Q_ss ./ tau))) < 1.0e-6 * dQ / tau
                @test minimum(abs.(r.e7)) > 0.5 * dQ / tau
            end
        end
    end

    @testset "vapor retrieval: the resting fixed point is blend-invariant" begin
        # The companion of the consistent_qss gate above. At rest the reference construction
        # makes Q_ssbar EXACTLY the diagnosed supersaturation, i.e. res_qss = Q_ssbar + rho_vs
        # is res_rho_t up to the rounding of one add — so the blend is the IDENTITY there,
        # whatever weight it computes, and a base that was a discrete fixed point stays one.
        # If enabling :blend moves the resting state, the blend is not a partition of the
        # same vapor and everything downstream of it is suspect.
        have = isdefined(Scythe, :vapor_retrieval_blend)
        @test have                   # ← the Stage 2 gate
        for (dry, q_l) in ((true, 0.0), (false, 1.0e-3))
            res = mktempdir() do tmpdir
                m, p, mod, _ = make_mc_mtile(tmpdir; dry = dry, q_l = q_l,
                                             consistent_qss = true)
                mod.options[:vapor_retrieval] = :residual
                step_mc!(m, p, mod, 5)
                (copy(p.physical), mod.grid_params.vars)
            end
            blend = mktempdir() do tmpdir
                m, p, mod, _ = make_mc_mtile(tmpdir; dry = dry, q_l = q_l,
                                             consistent_qss = true)
                mod.options[:vapor_retrieval] = :blend
                step_mc!(m, p, mod, 5)
                copy(p.physical)
            end
            phys_res, vars = res
            if q_l == 0.0
                # DRY: rho_liq = 0 identically, so w_cloud = 0 and the blend is not merely
                # the identity in value — it never evaluates res_qss at all. BITWISE, and
                # the exact-zero fixed point is preserved.
                @test blend == phys_res
                @test maximum(abs.(blend)) == 0.0
            else
                # CLOUDY: exact zero is not attainable (the saturated branch solves for
                # rho_c by Newton and lands ~1e-16 off the manifold), so the standard is the
                # one the consistent_qss gate already sets — and enabling the blend must not
                # loosen it. res_qss and res_rho_t differ here only by the rounding of
                # (rho_v - rho_vs) + rho_vs, ~1 ulp of rho_vs.
                @test maximum(abs.(blend[:, vars["p"], 1])) < 1.0e-9
                @test maximum(abs.(blend[:, vars["E_t"], 1])) < 1.0e-6
                for nm in ("rho_d", "rho_t", "u", "w", "Q_ss", "rho_r", "rho_c")
                    @test maximum(abs.(blend[:, vars[nm], 1])) < 1.0e-12
                end
                # ... and the two modes agree with each other to the same standard.
                @test maximum(abs.(blend[:, vars["p"], 1] .- phys_res[:, vars["p"], 1])) < 1.0e-9
                @test maximum(abs.(blend[:, vars["E_t"], 1] .- phys_res[:, vars["E_t"], 1])) < 1.0e-6
                for nm in ("rho_d", "rho_t", "u", "w", "Q_ss", "rho_r", "rho_c")
                    @test maximum(abs.(blend[:, vars[nm], 1] .-
                                       phys_res[:, vars[nm], 1])) < 1.0e-12
                end
            end
        end
    end

    @testset "vapor retrieval: :blend is a partition, not a mass source" begin
        # WHAT THE BLEND IS ALLOWED TO CHANGE. rho_d + rho_v + rho_c + rho_r no longer equals
        # rho_t pointwise once the vapor is blended — that gap is deliberate, it is what the
        # reconciliation absorbs, and on the measured snapshots it is at most 9.4e-4 kg/m^3
        # in cloud. What it may NOT change is the CONSERVED masses: rho_d has no sinks at all
        # and rho_t's only sink (sedimentation) is off here, so both domain integrals must
        # still be conserved to rounding. A blend that moved either one would not be a
        # partition of the water, it would be a source of it.
        have = isdefined(Scythe, :vapor_retrieval_blend)
        @test have                   # ← the Stage 2 gate
        mktempdir() do tmpdir
            m, p, mod, gp = vapor_blend_rirk(tmpdir, "blend"; retrieval = :blend)
            vars = gp.vars
            kDim = gp.kDim
            rho_dbar = view(Springsteel.ref_rho_d(m.ref_state), :, 1)
            rho_tbar = view(Springsteel.ref_rho_t(m.ref_state), :, 1)

            # Exact 2-D mass integral on the Gauss nodes (the positivity testset's argument:
            # an unweighted point sum would confuse redistribution among unequally weighted
            # nodes with a real drift).
            cellw = (npts, ncells, len) -> begin
                _, qw = Springsteel.CubicBSpline._quadrature_rule(div(npts, ncells),
                                                                  gp.quadrature)
                repeat(qw .* (len / ncells), outer = ncells)
            end
            Wi = cellw(gp.iDim, gp.num_cells_i, gp.iMax - gp.iMin)
            Wk = cellw(gp.kDim, gp.num_cells_k, gp.kMax - gp.kMin)
            massfun = (pp, name, bar) -> sum(Wi[div(i - 1, kDim) + 1] * Wk[mod1(i, kDim)] *
                                             (pp.physical[i, vars[name], 1] +
                                              bar[mod1(i, kDim)])
                                             for i in axes(pp.physical, 1))
            md0 = massfun(p, "rho_d", rho_dbar)
            mt0 = massfun(p, "rho_t", rho_tbar)

            vapor_blend_run!(m, p, gp, 20)

            @test all(isfinite.(p.physical))
            @test maximum(abs.(p.physical[:, vars["Q_ss"], 1])) > 0.0   # the run is live
            @test abs(massfun(p, "rho_d", rho_dbar) - md0) / md0 < 1.0e-12
            @test abs(massfun(p, "rho_t", rho_tbar) - mt0) / mt0 < 1.0e-12
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
    A `vapor_blend_rirk` patch whose rho_c perturbation is seeded NEGATIVE enough to drive
    rho_liq below zero somewhere, which is what the floor exists for. `amp = 0.0` leaves the
    healthy state (the floor is then inactive and must be bitwise inert).
    """
    function condensate_floor_rirk(tmpdir, tag; floor = nothing, amp = 2.0e-3)
        mtile, patch, model, gp = vapor_blend_rirk(tmpdir, tag)
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

            vapor_blend_run!(mA, pA, gpA, 10)
            vapor_blend_run!(mN, pN, gpN, 10)
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

            vapor_blend_run!(mN, pN, gpN, 10)
            vapor_blend_run!(mD, pD, gpD, 10)
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

            vapor_blend_run!(mN, pN, gpN, 10)
            vapor_blend_run!(mD, pD, gpD, 10)
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
            vapor_blend_run!(mA, pA, gpA, 10)
            vapor_blend_run!(mN, pN, gpN, 10)
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

            vapor_blend_run!(mN, pN, gpN, 20)
            vapor_blend_run!(mB, pB, gpB, 20)
            vapor_blend_run!(mS, pS, gpS, 20)

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
            vapor_blend_run!(mA, pA, gpA, 10)
            vapor_blend_run!(mN, pN, gpN, 10)
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
            vapor_blend_run!(mN, pN, gpN, 20)
            vapor_blend_run!(mB, pB, gpB, 20)
            vapor_blend_run!(mS, pS, gpS, 20)

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
                @test_throws ErrorException vapor_blend_run!(m, p, gp, 1)
                delete!(mo.options, :water_budget_trace)
                # clamp_water! would be a state repair on the control variable.
                mo.options[:clamp_water] = true
                @test_throws ErrorException Scythe.clamp_water!(m, 1, gp.kDim)
                delete!(mo.options, :clamp_water)
            end
        end
    end

    @testset "moist_entropy_total survives a negative vapor residual" begin
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
            vapor_blend_run!(m, p, gp, 20)
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
            for c in 1:ncol
                cs = ((c - 1) * kDim) + 1
                Scythe.physical_model(mtile, cs, cs + kDim - 1, 1)
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
        # Subsaturated vapor bump in dry air, ONLY Kvdiff_water active. The solved
        # species are now rho_w', rho_c' and rho_r, and the VAPOR increment is the
        # implied remainder delta_rho_v = delta_rho_w - delta_rho_c - delta_rho_r. With
        # no cloud and no rain to diffuse, delta_rho_c and delta_rho_r are identically
        # zero, so all of delta_rho_w must land in the vapor: cloud cannot appear.
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
            scales = Dict(1 => 1.0e2, 2 => 1.0e-5, 3 => 1.0e-5, 4 => 1.0e-2,
                          5 => 1.0e-2, 6 => 1.0e3, 7 => 1.0e-6, 8 => 1.0e-6,
                          9 => 1.0e-6, 10 => 1.0e-2)
            # ...but the DENOMINATOR is the larger of that floor and the tendency actually
            # produced, so this is a RELATIVE agreement test. It has to be: the seeded state
            # opens the closure's evaporation branch at full maximum-dryness drive (the bump
            # puts rho_c' at 0.5 kg/m^3, making the residual vapor about -0.5, so the clip's
            # vapor floor pins the drive at -rho_vs — far beyond the seeded Q_ss scale).
            # Judging the resulting tendency against a hardcoded 1e-6 scale measures nothing
            # about whether the two geometries agree — and they agree here to roundoff
            # RELATIVE. Keeping the fixed scale would have made this gate a hostage to the
            # magnitude of whatever state the seed happens to produce.
            per_slot = zeros(10)
            denom = Dict(v => max(scales[v], maximum(abs.(mt_rlr.expdot_n[:, v])),
                                  maximum(abs.(mt_ax.expdot_n[:, v]))) for v in 1:10)
            for c in 1:ncols_rlr
                i0 = (c - 1) * kDim
                r_c = gp1[i0 + 1, 1]
                a = findfirst(x -> x == r_c, r_ax)
                @test a !== nothing
                j0 = (a - 1) * kDim
                for v in 1:10
                    d = maximum(abs.(mt_rlr.expdot_n[i0+1:i0+kDim, v] .-
                                     mt_ax.expdot_n[j0+1:j0+kDim, v])) / denom[v]
                    per_slot[v] = max(per_slot[v], d)
                    worst = max(worst, d)
                end
            end
            @info "RLR WN0 vs axisym: worst scaled tendency mismatch = $worst" *
                  "\n  per slot: " * join(("$v=$(per_slot[v])" for v in 1:10), " ")
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
        @test names_xz[1:9] == Scythe.MC_VARS
        @test names_cyl[1:10] == Scythe.MC_VARS_CYL
        # The index the driver must resolve by NAME, precisely because it moves
        @test findfirst(==("n_r"), names_xz) == 10
        @test findfirst(==("n_r"), names_cyl) == 11

        # The number's own control-variable transform renames the slot, like the others
        tr = Dict{Symbol,Any}(:rain_moments => 2, :rain_number_transform => :bhyp)
        @test Scythe.rain_number_var_name(tr) == "nu_nr"
        @test Scythe.mc_var_names(tr)[10] == "nu_nr"
        @test Scythe.MC_NU_ALIAS["n_r"] == "nu_nr"
        @test_throws ErrorException Scythe.rain_number_transform_mode(
            Dict{Symbol,Any}(:rain_number_transform => :bogus))

        # mc_slot resolves by ROLE through either name; mc_optional_slot answers 0 for
        # a configuration that never registered it.
        vars_xz = Dict(v => i for (i, v) in enumerate(names_xz))
        vars_tr = Dict(v => i for (i, v) in enumerate(Scythe.mc_var_names(tr)))
        vars_1m = Dict(v => i for (i, v) in enumerate(Scythe.MC_VARS))
        @test Scythe.mc_slot(vars_xz, "n_r") == 10
        @test Scythe.mc_slot(vars_tr, "n_r") == 10       # via the "nu_nr" alias
        @test Scythe.mc_optional_slot(vars_xz, "n_r") == 10
        @test Scythe.mc_optional_slot(vars_tr, "n_r") == 10
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
            @test m2.mc_slots.n_r == 10
            @test length(mod2.grid_params.vars) == 10
            @test mod2.grid_params.vars["n_r"] == 10
            # Concrete field: the resolution must not cost the ModelTile its type stability
            @test isconcretetype(fieldtype(typeof(m2), :mc_slots))
            # And the per-variable scratch columns grew with the slot, so the number flux
            # has a column of its own to be fitted on.
            @test size(m2.scratch_columns, 2) == 10

            # Cylindrical: the same slot, one index further out, resolved by name
            m3, _, mod3, _ = make_mc_mtile(tmpdir; precipitation = true, iMin = 100.0,
                iMax = 2100.0, equation_set = "moist_compressible_axisym",
                extra_options = Dict{Symbol,Any}(:rain_moments => 2))
            @test m3.mc_slots.n_r == 11
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

            @test all(isfinite.(m2.var_np1))
            @test all(isfinite.(m2.expdot_n))
            # Rain mass appears, and so does rain NUMBER, everywhere the cloud is
            @test all(m2.var_np1[:, 8] .> 0.0)
            @test all(m2.var_np1[:, 10] .> 0.0)
            @test all(m2_off.var_np1[:, 8] .== 0.0)
            @test all(m2_off.var_np1[:, 10] .== 0.0)

            # Isolate the conversion by differencing against the precipitation-off arm, so
            # the (unrelated) condensation source on slot 9 cancels out.
            drr = m2.var_np1[:, 8] .- m2_off.var_np1[:, 8]
            drc = m2.var_np1[:, 9] .- m2_off.var_np1[:, 9]
            dnr = m2.var_np1[:, 10] .- m2_off.var_np1[:, 10]
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

            # The single-moment arm has no number slot at all
            @test size(m1.var_np1, 2) == 9
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
            seed_rain_bump!(patch, gpts, col, kDim; rho_r0 = 1.0e-3, zc = 1200.0, zr = 300.0)
            # Rain number PROPORTIONAL to the rain mass, so the two profiles start with the
            # same shape and the same centroid: any later difference is the sorting.
            for i in 1:size(patch.physical, 1)
                patch.physical[i, 10, 1] = 1.0e6 * patch.physical[i, 8, 1]
            end
            spectralTransform!(patch)
            gridTransform!(patch)

            z = gpts[:, 2]
            centroid(f) = sum(max.(f, 0.0) .* z) / sum(max.(f, 0.0))
            z_mass_0 = centroid(patch.physical[:, 8, 1])
            z_num_0 = centroid(patch.physical[:, 10, 1])
            @test z_mass_0 ≈ z_num_0 rtol=1e-10          # same shape to begin with

            # Fall speeds at the seeded state: |vtrm| > |vtrn|, which is the mechanism
            w_m, w_n = Scythe.rain_fall_speeds_2m(1.0e-3, 1.0e3, col.rho_d[1])
            @test abs(w_m) > abs(w_n) > 0.0

            step_mc!(m, patch, mod, 120)                  # 6 s of sedimentation

            @test all(isfinite.(patch.physical))
            z_mass_1 = centroid(patch.physical[:, 8, 1])
            z_num_1 = centroid(patch.physical[:, 10, 1])
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
        @test names_xz[1:10] == vcat(Scythe.MC_VARS, "n_r")
        @test length(names_xz) == 22 && length(names_cyl) == 23
        # The geometry-dependent indices the driver must resolve BY NAME
        @test findfirst(==("rho_i1"), names_xz) == 11
        @test findfirst(==("c_i3"), names_xz) == 22
        @test findfirst(==("rho_i1"), names_cyl) == 12
        @test findfirst(==("c_i3"), names_cyl) == 23
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
        @test Scythe.mc_var_names(tr)[11:22] == collect(Scythe.ice_var_names(tr))
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
        @test Scythe.mc_slot(vars_ice, "rho_i2") == 15
        @test Scythe.mc_slot(vars_tr, "rho_i2") == 15          # via the "nu_i2" alias
        @test Scythe.mc_ice_slot_indices(vars_ice) == ntuple(j -> 10 + j, 12)
        @test Scythe.mc_ice_slot_indices(vars_tr) == ntuple(j -> 10 + j, 12)
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
            @test m.mc_slots.n_r == 10
            @test Scythe.ice_slots(m.mc_slots, 1) == (11, 12, 13, 14)
            @test Scythe.ice_slots(m.mc_slots, 2) == (15, 16, 17, 18)
            @test Scythe.ice_slots(m.mc_slots, 3) == (19, 20, 21, 22)
            @test length(mod.grid_params.vars) == 22
            # Concrete: resolving twelve more indices must not cost the tile its typing
            @test isconcretetype(fieldtype(typeof(m), :mc_slots))
            # Every ice slot got a scratch column of its own, so each flux is fitted on its
            # own basis and its own BCs.
            @test size(m.scratch_columns, 2) == 22

            # Cylindrical: the same twelve, one index further out, resolved by name
            m3, _, mod3, _ = make_mc_mtile(tmpdir; precipitation = true, iMin = 100.0,
                iMax = 2100.0, equation_set = "moist_compressible_axisym",
                extra_options = ice)
            @test mod3.grid_params.vars["v"] == 10
            @test m3.mc_slots.n_r == 11
            @test Scythe.ice_slots(m3.mc_slots, 1) == (12, 13, 14, 15)
            @test Scythe.ice_slots(m3.mc_slots, 3) == (20, 21, 22, 23)
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

            # The other TEN slots are finite and the ice slots produced no NaN anywhere.
            for slot in 1:10
                @test all(isfinite.(patch.physical[:, slot, 1]))
            end
            for slot in 11:22
                @test all(isfinite.(patch.physical[:, slot, 1]))
            end
            # Species 2 and 3 were never seeded and no process can create them.
            for slot in 15:22
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
            for slot in 1:10
                @test m_on.var_np1[:, slot] == m_off.var_np1[:, slot]
                @test m_on.expdot_n[:, slot] == m_off.expdot_n[:, slot]
            end
            # ...and the twelve ice slots are EXACTLY zero, not merely small.
            for slot in 11:22
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
    """
    function make_ice_mtile(tmpdir; Tsurf = 258.15, q_l = 1.0e-3, rho_i = 1.0e-5,
                            n_i = 5.0e4, r_i = 30.0e-6, rho_r = 0.0, n_r = 0.0,
                            kDim = 16, ts = 0.1, extra_options = Dict{Symbol,Any}(),
                            extra_params = Dict{Symbol,Float64}())
        opts = merge(Dict{Symbol,Any}(:rain_moments => 2, :ice_microphysics => :ishmael,
                                      :vapor_retrieval => :residual), extra_options)
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
            patch.physical[i, vars["rho_i1"], 1] = rho_i
            patch.physical[i, vars["n_i1"], 1] = n_i
            patch.physical[i, vars["a_i1"], 1] = n_i * r_i^3
            patch.physical[i, vars["c_i1"], 1] = n_i * r_i^3
            patch.physical[i, 8, 1] = rho_r
            patch.physical[i, vars["n_r"], 1] = n_r
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
            for slot in 1:10
                @test m_on.var_np1[:, slot] == m_off.var_np1[:, slot]
                @test m_on.expdot_n[:, slot] == m_off.expdot_n[:, slot]
            end
            for slot in 11:22
                @test all(m_on.expdot_n[:, slot] .== 0.0)
            end
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

            # The pressure equation itself. The column is at rest (u = w = 0), so the
            # advective part of slot 1 is identically zero and expdot IS the forcing.
            kDim = mod.grid_params.kDim
            Ls = Springsteel.Thermodynamics.L_s.(S.Tk)
            qdep = S.Qdot_i1 .+ S.Qdot_i2 .+ S.Qdot_i3
            liquid = @. (S.R_m / S.C_vt) *
                        (((S.Lv - (Rv * S.C_pt * S.Tk / S.R_m)) * (S.Qdot + S.Qdot_r)) +
                         S.QDOT_TH)
            icepart = @. (S.R_m / S.C_vt) *
                         (((Ls - (Rv * S.C_pt * S.Tk / S.R_m)) * qdep) + (Lf * S.FRZ_NET))
            # Everything else in the slot-1 forcing is the divergence work, which is zero at
            # rest, plus the acoustic staging term, which is proportional to w.
            @test m.expdot_n[1:kDim, 1] ≈ liquid .+ icepart rtol = 1e-8 atol = 1e-12
            # The freezing piece is a pure heating and RAISES the pressure.
            @test all(icepart[S.FRZ_NET .> 0.0] .> 0.0)
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
            # state condenses more (equivalently, evaporates less) and deposits more.
            @test all(m_a.expdot_n[1:kDim, 9] .> m_b.expdot_n[1:kDim, 9])   # cloud
            @test all(m_a.expdot_n[1:kDim, 11] .> m_b.expdot_n[1:kDim, 11]) # ice mass

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
            @test maximum(abs.(ice .- dep)) > 0.0
            @test maximum(abs.((ice .- dep) .+ liq)) < 1.0e-12 * scale
        end
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
        # by a factor of ~10. What remains is the habit PARTITION (`ard`/`crd`/`prdr`/`qagg`),
        # which is an increment divided by the same step -- a consistent first-order
        # discretization -- so the agreement is close but not bitwise.
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
            # DEPOSITION contains no Δt at all: bitwise.
            @test a.Qdot_i == b.Qdot_i
            reldiff(x, y) = maximum(abs.(x .- y)) / max(maximum(abs.(x)), 1e-300)
            for (nm, x, y) in (("SRC_i1q", a.q, b.q), ("SRC_i1n", a.n, b.n),
                               ("ICE_C", a.c, b.c), ("ICE_R", a.r, b.r),
                               ("FRZ_NET", a.f, b.f))
                @test reldiff(x, y) < 0.05
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
            @test maximum(abs.((ice .- dep) .+ (S.ICE_C .+ S.ICE_R))) < 1.0e-12 * scale
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
        @test length(Scythe.MC_WATER_STATS) == Scythe.MC_STIFF_WARNED
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
    @testset "ice: mass with no number is inert, but nucleation still fires" begin
        mktempdir() do tmpdir
            # A column carrying ice MASS with the number slot at exactly zero. Warm enough
            # that no nucleation fires, so the only thing that could move the slots is a
            # rate read off the phantom population var_check would manufacture.
            m, patch, mod, _ = make_ice_mtile(tmpdir; Tsurf = 258.15, q_l = 0.0,
                                              rho_i = 1.0e-5, n_i = 0.0)
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
                                              rho_i = 1.0e-4, n_i = 0.0)
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
