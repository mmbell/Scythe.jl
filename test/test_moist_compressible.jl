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
            # This is what `water_cap_mode = :euler` pins to at every step, and it has to be
            # BIT-identical or the A/B against runs A-H is not an A/B.
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

        @testset "qss_condensation_rates default bounds are the historical ones" begin
            Tk = 285.0; p_hPa = 900.0; ts = 0.1
            rho_vs = rho_v_sat(Tk, p_hPa)
            rho_d = (100.0 * p_hPa - Rv * Tk * rho_vs) / (Rd * Tk)
            Q_s = Scythe.Q_s_energy(Tk, 100.0 * p_hPa, rho_d, rho_vs / rho_d, 1.0e-3)
            for (Q_ss, rho_v, rho_c, rho_r) in (
                    (-0.5 * rho_vs, 0.5 * rho_vs, 1.0e-4, 1.0e-3),
                    (0.5 * rho_vs, 1.5 * rho_vs, 1.0e-4, 1.0e-3),
                    (0.2 * rho_vs, 1.0e-9, 1.0e-4, 0.0),
                    (-0.9 * rho_vs, 0.1 * rho_vs, 0.0, 1.0e-3))
                for cf in (1.0, 12.0 / 23.0)
                    base = Scythe.qss_condensation_rates(Q_ss, rho_v, rho_c, rho_r, rho_d,
                                                         Tk, p_hPa, Q_s, ts, 1.0e-3;
                                                         cap_factor = cf)
                    expl = Scythe.qss_condensation_rates(Q_ss, rho_v, rho_c, rho_r, rho_d,
                                                         Tk, p_hPa, Q_s, ts, 1.0e-3;
                                                         cap_factor = cf,
                                                         floor_c = -cf * max(rho_c, 0.0) / ts,
                                                         floor_r = -cf * max(rho_r, 0.0) / ts,
                                                         ceil_v = max(rho_v, 0.0) / ts)
                    @test base[1] === expl[1]
                    @test base[2] === expl[2]
                end
            end
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
                           iMin=0.0, iMax=2000.0, f=0.0,
                           consistent_qss=false)
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
            options = Dict{Symbol,Any}(:semiimplicit => semiimplicit,
                           :exact_reference_state => true,
                           :consistent_qss_reference => consistent_qss,
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
        #       nothing — is the blend to the LAST BIT, not "agrees to 1e-14". The same
        #       standard `water_cap_mode = :euler` is held to.
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
            # (b) :residual is a different integration on this cloudy state — and a nearby
            #     one, since the two retrievals differ by the bounded partition gap and not
            #     by a change of equation set.
            @test pA.physical != pR.physical
            qi = gpA.vars["Q_ss"]
            dq = maximum(abs.(pA.physical[:, qi, 1] .- pR.physical[:, qi, 1]))
            @test dq > 0.0
            @test dq < 1.0e-2 * maximum(abs.(pA.physical[:, qi, 1]))
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
                         rain_amp = 0.0)
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
        for i in 1:size(patch.physical, 1)
            patch.physical[i, w_i, 1] = 1.0e-2 * sin(0.5 * sqrt(2.0) * i^2)
            zi = gpts[i, end]
            rho_c = amp * exp(-((zi - 2000.0) / 250.0)^2)          # >= 0 by construction
            patch.physical[i, rc_i, 1] = Scythe.condensate_slot(rho_c, 0.0, tf, 1.0e-7)
            # An equally sub-cell RAIN spike, sited lower so it is a distinct feature. Rain
            # has no reference profile, so its slot is the control variable itself.
            if rain_amp > 0.0
                rho_r = rain_amp * exp(-((zi - 1200.0) / 250.0)^2)
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
            scales = Dict(1 => 1.0e2, 2 => 1.0e-5, 3 => 1.0e-5, 4 => 1.0e-2,
                          5 => 1.0e-2, 6 => 1.0e3, 7 => 1.0e-6, 8 => 1.0e-6,
                          9 => 1.0e-6, 10 => 1.0e-2)
            for c in 1:ncols_rlr
                i0 = (c - 1) * kDim
                r_c = gp1[i0 + 1, 1]
                a = findfirst(x -> x == r_c, r_ax)
                @test a !== nothing
                j0 = (a - 1) * kDim
                for v in 1:10
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
end
