# ─────────────────────────────────────────────────────────────────────────────
# The anchor reconciliation of the condensate partition (Stage C).
# reference/Scythe_moist_compressible.tex §"Reconciliation of the condensate
# partition" is the specification; `ice_anchor_rate` is the rate law under test.
#
# The driver-level gates (bitwise inertness through mc_driver!, the off-switch,
# the census rows) live in test_moist_compressible.jl beside the other ice
# integration tests, which own the ModelTile harness.
# ─────────────────────────────────────────────────────────────────────────────

@testset "Anchor reconciliation of the condensate partition (Stage C)" begin

    @testset "one-sidedness: exactly 0.0 wherever the partition is admissible" begin
        # The device exists for ONE state: ice mass beyond the anchor headroom. On
        # every admissible state — and every empty one — both the rate and the
        # recorded defect are EXACT zeros, which is what makes the warm and dry
        # paths bitwise and the census silent for the pre-glaciation half hour.
        tau = 10.0
        # (rho_t, rho_d, rho_liq, rho_ice) admissible states:
        for (rt, rd, rl, ri) in (
                (1.0, 0.98, 0.0, 0.0),          # moist air, no condensate
                (1.0, 0.98, 0.01, 0.0),         # liquid only, within headroom
                (1.0, 0.98, 0.005, 0.01),       # liquid + ice exactly filling less
                (1.0, 0.98, 0.0, 0.02),         # ice exactly AT the headroom
                (1.0, 0.98, -1.0e-4, 0.019),    # ringing negative liquid, ice within
                (1.0, 0.98, 0.0, -1.0e-12),     # ice slot ringing at the -mu floor
                (1.0, 1.0, 0.0, 0.0),           # dry: zero headroom, zero ice
                (0.4, 0.4081, 0.0, 0.0))        # rho_t < rho_d (fit noise), no ice
            phi, delta = Scythe.ice_anchor_rate(rt, rd, rl, ri, tau)
            @test phi === 0.0
            @test delta === 0.0
        end
    end

    @testset "the defect and the rate on detached states" begin
        tau = 10.0
        # The §2c measured point: ice at 4.1x the anchor water.
        rt, rd, ri = 0.4080435348613092, 0.408042550918894, 8.8166805353408e-6
        phi, delta = Scythe.ice_anchor_rate(rt, rd, 0.0, ri, tau)
        head = max(rt - rd, 0.0)
        @test delta ≈ ri - head rtol = 1e-14
        @test phi ≈ delta / (ri * tau) rtol = 1e-14
        # A ringing NEGATIVE liquid must not enlarge the ice's debit: the headroom
        # floors the liquid at zero, so these two states carry the same defect.
        phi_a, delta_a = Scythe.ice_anchor_rate(1.0, 0.99, 0.0, 0.02, tau)
        phi_b, delta_b = Scythe.ice_anchor_rate(1.0, 0.99, -5.0e-3, 0.02, tau)
        @test delta_a == delta_b
        @test phi_a == phi_b
        # A POSITIVE liquid does take its share of the headroom first.
        _, delta_c = Scythe.ice_anchor_rate(1.0, 0.99, 4.0e-3, 0.02, tau)
        @test delta_c == delta_a + 4.0e-3
    end

    @testset "the removed fraction is bounded by dt/tau — never the reservoir" begin
        # phi <= 1/tau identically (delta <= rho_ice by construction), so the
        # per-step removed fraction phi*dt <= dt/tau << 1 at every state the
        # transport can manufacture. No realization factor is needed and none is
        # applied; this is the bound the TeX's fourth property states.
        tau = 10.0
        for rt in (0.2, 0.4, 1.0), rd in (0.19, 0.4, 1.1), rl in (-1.0e-3, 0.0, 0.05),
            ri in (1.0e-12, 1.0e-6, 2.75e-3, 1.123e-2, 1.0)
            phi, delta = Scythe.ice_anchor_rate(rt, rd, rl, ri, tau)
            @test phi >= 0.0
            @test phi <= 1.0 / tau + eps()
            @test delta >= 0.0
            for ts in (0.03, 0.15, 0.3)
                @test phi * ts < 1.0
            end
        end
        # And the rate law carries no timestep at all: the same state gives the
        # same rate whatever the step — Delta-t-freedom by signature, checked by
        # value across the tau sweep's linearity instead.
        phi3, d3 = Scythe.ice_anchor_rate(1.0, 0.99, 0.0, 0.02, 3.0)
        phi30, d30 = Scythe.ice_anchor_rate(1.0, 0.99, 0.0, 0.02, 30.0)
        @test d3 == d30                       # the defect is a state, not a rate
        @test phi3 ≈ 10.0 * phi30 rtol = 1e-14
    end

    @testset "proportional carriage preserves the per-particle state" begin
        # One factor on all four moments: the recovered per-particle axes, the
        # effective density and the aspect ratio must not move when a species is
        # scaled — the moment consistency that blocked a bare mass floor.
        #
        # The state must be REALIZABLE (var_check a no-op): on a non-realizable
        # state var_check's repair branches are legitimately not scale-invariant,
        # which is var_check's business, not the device's — the device only
        # guarantees the CARRIED moments keep their ratios. Monodisperse spheres,
        # r = 30 um at effective density 500 kg/m^3, the make_ice_mtile seed.
        r_i = 30.0e-6
        ni = 5.0e4
        ai = ni * r_i^3
        ci = ni * r_i^3
        qi = ni * (4.0 / 3.0) * pi * r_i^3 * 500.0
        eff0 = Scythe._ice_effective(qi, ni, ai, ci, 1)
        # one forward-Euler application of the device at a hostile fraction
        f = 1.0 - 0.015                      # phi*ts at tau = 10 s, ts = 0.15 s
        eff1 = Scythe._ice_effective(f * qi, f * ni, f * ai, f * ci, 1)
        @test eff1.ani ≈ eff0.ani rtol = 1e-12
        @test eff1.cni ≈ eff0.cni rtol = 1e-12
        @test eff1.rhobar ≈ eff0.rhobar rtol = 1e-12
        @test eff1.deltastr ≈ eff0.deltastr rtol = 1e-12
        # The mass shares among species are equally invariant: the same factor on
        # each species' four moments is the same factor on the sums.
        q = (1.0e-4, 3.0e-5, 7.0e-6)
        @test (f * q[1]) / sum(f .* q) ≈ q[1] / sum(q) rtol = 1e-14

        # The n = 0 corner — the gate-orphaned state the device exists for: the
        # removal drains the mass while the carried zero number STAYS an exact
        # zero, because -phi * max(0.0, 0.0) is an exact zero.
        phi, _ = Scythe.ice_anchor_rate(0.4, 0.399, 0.0, 2.0e-3, 10.0)
        @test phi > 0.0
        @test -phi * max(0.0, 0.0) === -0.0 || -phi * max(0.0, 0.0) === 0.0
        # ...and a moment ringing NEGATIVE contributes no source at all (the
        # device may only remove): the increment reads max(moment, 0).
        @test -phi * max(-1.0e-13, 0.0) == 0.0
    end

    @testset "the reader-side cap breaks the retrieval amplifier (fixture: step 8180)" begin
        # The recorded worst-class point of the ts=0.3 death window (RETDUMP, run
        # stageBlam_probe2, step 8180, g=33448, z=12278 m): the per-step fit oscillation
        # has put 24.2 g/m^3 of ice against 5.9 g/m^3 of anchor water, and the raw
        # retrieval credits the phantom with L_s.
        M     = 59179.85549936144
        rho_d = 0.2597466555954982
        rho_t = 0.2656140614709738
        rho_i = 0.024247399759469938
        T_un = Scythe.retrieve_temperature(M, rho_d, rho_t, 0.0, rho_i)
        @test T_un ≈ 466.6543722427886 rtol = 1e-12       # reproduces the recorded T
        # Capped at the anchor headroom, the same point retrieves a PHYSICAL temperature:
        # the excursion was entirely the inadmissible partition's latent credit.
        head = max(rho_t - rho_d, 0.0)
        T_cap = Scythe.retrieve_temperature(M, rho_d, rho_t, 0.0, min(rho_i, head))
        @test T_cap < 280.0
        @test T_un - T_cap > 180.0
        # And the cap is STRICTLY one-sided: an admissible partition retrieves the
        # identical bits, because min(rho_i, head) IS rho_i there.
        @test min(1.0e-5, head) === 1.0e-5
    end

    @testset "resolution generality: pointwise, stateless, no grid constants" begin
        # The rate is a pure function of the local state — the same numbers on any
        # grid, any nest, any resolution. (The census rows are per-thread
        # accumulators; the driver-level test exercises them.)
        args = (0.4080435348613092, 0.408042550918894, 0.0, 8.8166805353408e-6, 10.0)
        @test Scythe.ice_anchor_rate(args...) === Scythe.ice_anchor_rate(args...)
        @test length(methods(Scythe.ice_anchor_rate)) == 1
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# The population reconciliation (Stage 3a).
# reference/Scythe_moist_compressible.tex §"Reconciliation of the population" is
# the specification; `ice_population_rate` is the rate law and
# `_ice_population_reconcile!` the one-column pass under test.
#
# The driver-level gates (bitwise inertness through mc_driver!, the off-switch,
# the census rows, the multi-step behaviour) live in test_moist_compressible.jl
# beside the other ice integration tests, which own the ModelTile harness.
# ─────────────────────────────────────────────────────────────────────────────

@testset "Population reconciliation of the ice moments (Stage 3a)" begin

    # A bare `mc_scratch` column: the same NamedTuple `_allocate_mc_scratch` builds, so the
    # pass under test sees exactly the fields the driver hands it and nothing is mocked.
    make_S(n) = NamedTuple{Scythe.MC_SCRATCH_SLOTS}(
        ntuple(_ -> zeros(Float64, n), length(Scythe.MC_SCRATCH_SLOTS)))

    @testset "the rate law: one-sided, and the exact complement of the population gate" begin
        tau = 10.0
        rho_a = 1.0
        # LIVE species (mass AND number): the gate lets every rate act, so the
        # reconciliation must not — exact zeros, which is what makes the interior of a
        # healthy ice cloud bitwise.
        for (q, n) in ((1.0e-5, 5.0e4), (1.0e-3, 1.0e6), (1.0e-12 + 1.0e-15, 1.0e-30))
            rate, rho0 = Scythe.ice_population_rate(q, n, rho_a, tau)
            @test rate === 0.0
            @test rho0 === 0.0
        end
        # EMPTY species (no mass): nothing to return, whatever the number does.
        for (q, n) in ((0.0, 0.0), (0.0, -1.0), (-1.0e-9, 0.0), (1.0e-13, 0.0),
                       (Scythe.ISHMAEL_QSMALL, 0.0))
            rate, rho0 = Scythe.ice_population_rate(q, n, rho_a, tau)
            @test rate === 0.0
            @test rho0 === 0.0
        end
        # DEAD species: mass past the gate's own mixing-ratio threshold, number not
        # positive. This is the whole support of the device.
        for n in (0.0, -1.0e-30, -5.0e3)
            rate, rho0 = Scythe.ice_population_rate(1.0e-5, n, rho_a, tau)
            @test rho0 == 1.0e-5
            @test rate ≈ 1.0e-5 / tau rtol = 1e-15
        end
        # The mass test is the GATE's, in densities: `q > QSMALL` with `q = rho_i/rho_a`.
        @test Scythe.ice_population_rate(2.0e-12, 0.0, 1.0, tau)[2] == 2.0e-12
        @test Scythe.ice_population_rate(2.0e-12, 0.0, 4.0, tau)[2] === 0.0
        # Delta-t free and pointwise: the per-step fraction is dt/tau and nothing else.
        @test Scythe.ice_population_rate(1.0e-5, 0.0, 1.0, 10.0)[1] ==
              10.0 * Scythe.ice_population_rate(1.0e-5, 0.0, 1.0, 100.0)[1]
        @test length(methods(Scythe.ice_population_rate)) == 1
    end

    @testset "below T_0: number only — no mass, no latent heat, 2 um crystals" begin
        tau = 10.0
        ts = 0.5
        S = make_S(3)
        S.i1q .= 1.0e-5                       # mass with no crystals
        S.i1n .= 0.0
        S.i1a .= 0.0
        S.i1c .= 0.0
        S.Tk .= 258.15
        S.rho_d .= 1.0
        st = zeros(Float64, length(Scythe.MC_WATER_STATS), 1)
        Scythe._ice_population_reconcile!(S, st, 1, S.Tk, S.rho_d, tau, true, ts)

        rate = 1.0e-5 / tau
        nseed = rate / Scythe.ISHMAEL_M_MIN
        # THE NUMBER SOURCE IS EXACTLY rho_empty/(m_min tau) — the TeX's own expression.
        @test all(S.SRC_i1n .≈ nseed)
        @test nseed > 0.0
        # NO MASS MOVES, and nothing thermodynamic happens: this branch creates number and
        # nothing else, so the freezing net and both liquid back-reactions stay untouched.
        @test all(iszero, S.SRC_i1q)
        @test all(iszero, S.FRZ_NET)
        @test all(iszero, S.ICE_R)
        @test all(iszero, S.ICE_NR)
        @test all(iszero, S.ICE_C)
        # The volume moments are seeded the way every nucleation channel seeds them, so the
        # population that results is 2 um spheres and both axes get the same number.
        vseed = Scythe._ice_nucleation_volume(rate, nseed)
        @test vseed > 0.0
        @test all(S.SRC_i1a .≈ vseed)
        @test all(S.SRC_i1c .≈ vseed)
        # ...which is the SMALLEST crystal the scheme resolves, not the largest: the
        # characteristic axis the seeding implies sits at the `ISHMAEL_RMIN` clamp.
        gam3 = Scythe.gamma(Scythe.ISHMAEL_NU + 3.0)
        @test cbrt((rate * Scythe.ISHMAEL_GAMMNU) /
                   (Scythe.ISHMAEL_RHOI * nseed * Scythe.ISHMAEL_FOURTHIRDSPI * gam3)) <=
              Scythe.ISHMAEL_RMIN
        # The census: the defect, its support, and the number seeded.
        @test st[Scythe.MC_POP_MAX, 1] == 1.0e-5
        @test st[Scythe.MC_POP_PTS, 1] == 3.0
        @test st[Scythe.MC_POP_RAIN, 1] == 0.0
        @test st[Scythe.MC_POP_SEED, 1] ≈ 3.0 * nseed * ts rtol = 1e-14
        # Species 2 and 3 are empty, so their slots are EXACT zeros — the inertness the
        # whole zero-ice gate rests on, and `x + 0.0 === x` is what preserves it.
        for nm in (:SRC_i2q, :SRC_i2n, :SRC_i2a, :SRC_i2c,
                   :SRC_i3q, :SRC_i3n, :SRC_i3a, :SRC_i3c)
            @test all(getproperty(S, nm) .=== 0.0)
        end
    end

    # ── STAGE 3c: the SECOND seeding, `options[:ice_population_seed] = :large` ────
    #
    # The `:min` branch above seeds 2 um spheres, which is the fastest-responding
    # population and the slowest-falling one. It is NOT what the dead mass is: mass
    # with no number is what SIZE SORTING leaves behind, so the particles sitting
    # there are the big end of a distribution whose number went somewhere else. The
    # `:large` branch seeds what ISHMAEL's own moment checker re-diagnoses from that
    # mass at the floor number -- the 1 mm large-ice limit at the species' bulk
    # density -- so the seeded triple is realizable on the next step by construction.
    @testset "below T_0, :large -- var_check's own re-diagnosis of the dead mass" begin

        # The Fortran's LARGE-ICE LIMIT (module_mp_jensen_ishmael.F lines 3171-3193),
        # re-derived here from the formula and NOT from the code under test: at
        # deltastr = 1 (the 2 um spherical axes the QNSMALL/QASMALL floors imply) the
        # capped axis is a_n = c_n = 1 mm and
        #
        #     n = q GAMMA(nu) / ((4/3) pi rhobar a_n^3 GAMMA(nu+3)),  a_i = c_i = n a_n^3
        #
        # with rhobar the species ceiling var_check clamps to (lines 3131-3154):
        # RHOI = 920 for the planar and columnar species, 50 for the aggregates
        # (line 2603, the "Final check on aggregates" block).
        gam3 = Scythe.gamma(Scythe.ISHMAEL_NU + 3.0)
        amax = 1.0e-3
        function large_ref(q, k)
            rb = k == 3 ? Scythe.ISHMAEL_RHO_AGG : Scythe.ISHMAEL_RHOI
            n = q * Scythe.ISHMAEL_GAMMNU /
                (Scythe.ISHMAEL_FOURTHIRDSPI * rb * amax^3 * gam3)
            return (n, n * amax^3, n * amax^3)
        end

        @testset "the rule, for all three species" begin
            for k in 1:3, q in (1.0e-7, 1.0e-5, 6.0e-3, 6.0e-2)
                n, a, c = Scythe._ice_population_seed_large(q, k)
                nr, ar, cr = large_ref(q, k)
                @test n ≈ nr rtol = 1e-12
                @test a ≈ ar rtol = 1e-12
                @test c ≈ cr rtol = 1e-12
                # SPHERES: the floors give deltastr = 1 exactly, so both axes are the
                # same 1 mm and the two volume moments coincide.
                @test a === c
                # ...and it really is the FEW-LARGE end: 1.5e10 times fewer particles
                # than the 2 um seeding gives the same mass (8.2e8 for aggregates).
                @test n < 1.0e-8 * (q / Scythe.ISHMAEL_M_MIN)
                # Aggregates are the only species that differs, and only through the
                # 50 kg/m^3 ceiling: 920/50 = 18.4x more of them for the same mass.
                k == 3 && @test n ≈ 18.4 * large_ref(q, 1)[1] rtol = 1e-12
            end
            # Below QSMALL there is no population to diagnose: var_check divides by
            # the mass, so the helper refuses rather than dividing by nothing.
            for k in 1:3
                @test Scythe._ice_population_seed_large(Scythe.ISHMAEL_QSMALL, k) ===
                      (0.0, 0.0, 0.0)
                @test Scythe._ice_population_seed_large(0.0, k) === (0.0, 0.0, 0.0)
            end
            @test length(methods(Scythe._ice_population_seed_large)) == 1
        end

        @testset "it is exactly `_ice_effective` at the floors, and a fixed point of it" begin
            for k in 1:3, q in (1.0e-11, 1.0e-7, 1.0e-5, 6.0e-3)
                n, a, c = Scythe._ice_population_seed_large(q, k)
                # For the two 920 kg/m^3 species the helper IS `_ice_effective` at the
                # floors, to the last bit -- nothing is re-derived.
                if k != 3
                    e0 = Scythe._ice_effective(q, 0.0, 0.0, 0.0, k)
                    @test (n, a, c) === (e0.ni, e0.ai, e0.ci)
                end
                # THE POINT OF THE SEEDING: var_check does not change it. The seeded
                # triple is already realizable, so the very next step's re-diagnosis
                # returns it -- to 2e-11 where the axes come back through
                # `^0.333333333333` and to 2e-9 where the small-volume branch's
                # `^0.3333333333` is the one that fires. Both are the PORT'S OWN
                # truncated cube-root exponents, carried into `n` as `a_n^-3`; nothing
                # here is a repair, and a genuine re-diagnosis moves the number by ten
                # orders of magnitude, not by nine decimal places.
                e = Scythe._ice_effective(q, n, a, c, k)
                @test e.ni ≈ n rtol = 1e-8
                @test e.ai ≈ a rtol = 1e-8
                @test e.ci ≈ c rtol = 1e-8
                @test e.deltastr ≈ 1.0 rtol = 1e-12
                # ...and the population is LIVE: a positive number is exactly what the
                # population gate in `mc_ice_sources!` tests for.
                @test n > 0.0
                @test Scythe.ice_population_rate(q, n, 1.0, 10.0) === (0.0, 0.0)
            end
            # Below the 1 mm cap's threshold the LARGE-ICE limit does not bind and the
            # QNSMALL floor is itself the answer -- the same statement (the fewest
            # particles the scheme admits for the mass), by the other clamp.
            @test Scythe._ice_population_seed_large(1.0e-11, 1)[1] ===
                  Scythe.ISHMAEL_QNSMALL
        end

        @testset "in the pass: number only, and orders of magnitude fewer crystals" begin
            tau = 10.0
            ts = 0.5
            rho_a = 1.0
            q = 1.0e-5
            S = make_S(3)
            S.i1q .= q; S.i2q .= q; S.i3q .= q
            S.Tk .= 258.15
            S.rho_d .= rho_a
            st = zeros(Float64, length(Scythe.MC_WATER_STATS), 1)
            Scythe._ice_population_reconcile!(S, st, 1, S.Tk, S.rho_d, tau, true, ts, :large)

            for (k, SRCn, SRCa, SRCc, SRCq) in ((1, S.SRC_i1n, S.SRC_i1a, S.SRC_i1c, S.SRC_i1q),
                                                (2, S.SRC_i2n, S.SRC_i2a, S.SRC_i2c, S.SRC_i2q),
                                                (3, S.SRC_i3n, S.SRC_i3a, S.SRC_i3c, S.SRC_i3q))
                n, a, c = large_ref(q / rho_a, k)
                @test all(SRCn .≈ n * rho_a / tau)
                @test all(SRCa .≈ a * rho_a / tau)
                @test all(SRCc .≈ c * rho_a / tau)
                # NO MASS MOVES -- the same statement the `:min` branch makes.
                @test all(iszero, SRCq)
            end
            @test all(iszero, S.FRZ_NET)
            @test all(iszero, S.ICE_R)
            @test all(iszero, S.ICE_NR)
            @test all(iszero, S.ICE_C)

            # Against the `:min` seeding on the identical column: same defect, same
            # census support, a number smaller by ten orders of magnitude.
            Smin = make_S(3)
            Smin.i1q .= q; Smin.i2q .= q; Smin.i3q .= q
            Smin.Tk .= 258.15
            Smin.rho_d .= rho_a
            stmin = zeros(Float64, length(Scythe.MC_WATER_STATS), 1)
            Scythe._ice_population_reconcile!(Smin, stmin, 1, Smin.Tk, Smin.rho_d,
                                              tau, true, ts)
            @test stmin[Scythe.MC_POP_MAX, 1] == st[Scythe.MC_POP_MAX, 1]
            @test stmin[Scythe.MC_POP_PTS, 1] == st[Scythe.MC_POP_PTS, 1]
            @test st[Scythe.MC_POP_SEED, 1] < 1.0e-8 * stmin[Scythe.MC_POP_SEED, 1]
            @test st[Scythe.MC_POP_SEED, 1] > 0.0

            # The DEFAULT is `:min`: the trailing argument absent is the committed
            # behaviour to the last bit.
            Sdef = make_S(3)
            Sdef.i1q .= q; Sdef.i2q .= q; Sdef.i3q .= q
            Sdef.Tk .= 258.15
            Sdef.rho_d .= rho_a
            stdef = zeros(Float64, length(Scythe.MC_WATER_STATS), 1)
            Scythe._ice_population_reconcile!(Sdef, stdef, 1, Sdef.Tk, Sdef.rho_d,
                                              tau, true, ts, :min)
            for nm in (:SRC_i1n, :SRC_i1a, :SRC_i1c, :SRC_i2n, :SRC_i3n)
                @test getproperty(Sdef, nm) == getproperty(Smin, nm)
            end
            @test stdef == stmin

            # ABOVE T_0 the seeding is not consulted at all: that branch moves mass to
            # the rain and seeds no ice number, whichever crystal is selected.
            Swarm = make_S(2)
            Swarm.i1q .= q; Swarm.Tk .= 290.0; Swarm.rho_d .= rho_a
            stw = zeros(Float64, length(Scythe.MC_WATER_STATS), 1)
            Scythe._ice_population_reconcile!(Swarm, stw, 1, Swarm.Tk, Swarm.rho_d,
                                              tau, true, ts, :large)
            @test all(Swarm.SRC_i1q .≈ -q / tau)
            @test all(iszero, Swarm.SRC_i1n)
            @test stw[Scythe.MC_POP_SEED, 1] == 0.0

            # Neither branch allocates: the seed is a pure function of (q, k) and the
            # pass stays the same non-allocating gridpoint loop it was.
            call(Sx, stx, sd) = Scythe._ice_population_reconcile!(Sx, stx, 1, Sx.Tk,
                                     Sx.rho_d, tau, true, ts, sd)
            call(S, st, :large); call(Smin, stmin, :min)
            @test @allocated(call(S, st, :large)) == 0
            @test @allocated(call(Smin, stmin, :min)) == 0
        end
    end

    # ── STAGE 3d: the THIRD seeding, `options[:ice_population_seed] = :local` ─────
    #
    # Neither of the two above reads the DEFECT. Number-less mass is the negative
    # lobe of the number moment's spline ringing at cloud edges and gradients -- the
    # mass-without-number face of the transport-decorrelation family whose other face
    # is the number spikes sitting right beside it -- so the crystals the lobe lost
    # are not a size to be derived from its mass at all: they are the crystals of the
    # same species one gridpoint away. `:local` seeds THOSE.
    @testset "below T_0, :local -- the crystals of the neighbours the lobe took them from" begin

        gam3 = Scythe.gamma(Scythe.ISHMAEL_NU + 3.0)
        # The per-crystal volume moment of a SPHERE of mass `m` at bulk density `rb`:
        # a_pc = c_pc = a_n^3 with m = rb (4/3)pi a_n^3 GAMMA(nu+3)/GAMMA(nu).
        sphere_pc(m, rb) = m * Scythe.ISHMAEL_GAMMNU /
                           (rb * Scythe.ISHMAEL_FOURTHIRDSPI * gam3)
        # A LIVE, CHECKED population: whatever raw moments go in, what comes out is a
        # fixed point of `var_check`, which is what a healthy gridpoint carries.
        function live_pc(q, n, r, k)
            e = Scythe._ice_effective(q, n, n * r^3, n * r^3, k)
            return (q / e.ni, e.ai / e.ni, e.ci / e.ni)      # (m, a, c) per crystal
        end

        @testset "the column search: nearest live point, both directions" begin
            rho_a = ones(7)
            Q = fill(1.0e-5, 7)                       # mass everywhere...
            N = zeros(7)
            idx = eachindex(Q)
            # ...and no crystals anywhere: the pure-dead column has no span at all.
            @test Scythe._ice_live_span(Q, N, rho_a, idx) === (0, 0)
            for i in idx
                @test Scythe._ice_live_neighbour(Q, N, rho_a, i, 0, 0) == 0
            end
            # Live at 2 and 6.
            N[2] = 1.0e4; N[6] = 1.0e4
            f, l = Scythe._ice_live_span(Q, N, rho_a, idx)
            @test (f, l) == (2, 6)
            # Outside the span the answer is the span end, with no walk at all.
            @test Scythe._ice_live_neighbour(Q, N, rho_a, 1, f, l) == 2
            @test Scythe._ice_live_neighbour(Q, N, rho_a, 7, f, l) == 6
            # A live point is its own neighbour (never consulted: it is not dead).
            @test Scythe._ice_live_neighbour(Q, N, rho_a, 2, f, l) == 2
            @test Scythe._ice_live_neighbour(Q, N, rho_a, 6, f, l) == 6
            # Inside it, the nearest -- and a TIE goes to the lower index.
            @test Scythe._ice_live_neighbour(Q, N, rho_a, 3, f, l) == 2
            @test Scythe._ice_live_neighbour(Q, N, rho_a, 4, f, l) == 2      # tie, d = 2
            @test Scythe._ice_live_neighbour(Q, N, rho_a, 5, f, l) == 6
            # The liveness test is the population gate's own, in densities: a point with
            # number but no mass past QSMALL*rho_a is NOT a live neighbour.
            Q2 = copy(Q); Q2[2] = 1.0e-13
            @test Scythe._ice_live_span(Q2, N, rho_a, idx) === (6, 6)
            rho_b = fill(1.0, 7); rho_b[6] = 1.0e8       # QSMALL*rho_a now exceeds the mass
            @test Scythe._ice_live_span(Q, N, rho_b, idx) === (2, 2)
            # ...and so is one whose number has rung negative.
            N2 = copy(N); N2[2] = -1.0e4
            @test Scythe._ice_live_span(Q, N2, rho_a, idx) === (6, 6)
            @test length(methods(Scythe._ice_live_neighbour)) == 1
            @test length(methods(Scythe._ice_live_span)) == 1
        end

        @testset "the seed IS the neighbour's crystal: same mass, same habit" begin
            for k in 1:3, (q_nb, n_nb, r_nb) in ((1.0e-5, 1.0e4, 30.0e-6),
                                                 (3.0e-4, 2.0e5, 80.0e-6),
                                                 (1.0e-6, 5.0e3, 12.0e-6))
                m, apc, cpc = live_pc(q_nb, n_nb, r_nb, k)
                # The clamps must not be binding, or this is a test of them instead.
                @test Scythe.ISHMAEL_M_MIN < m < Scythe.ISHMAEL_M_LARGE[k]
                for q in (1.0e-5, 4.0e-5, 6.0e-3)
                    n, a, c = Scythe._ice_population_seed_local(q, m, apc, cpc, k)
                    # THE NUMBER IS rho_empty/m_loc, and the per-crystal state is the
                    # neighbour's. "Is" to a few parts in 1e11: the axes round-trip
                    # a^2/c -> cube root -> back through the port's own truncated
                    # `^0.333333333333` exponents, which is the entire discrepancy and
                    # is nine orders below any re-diagnosis.
                    @test n ≈ q / m rtol = 1e-9
                    @test a / n ≈ apc rtol = 1e-9
                    @test c / n ≈ cpc rtol = 1e-9
                    # ...so the seeded population is LIVE, which is the whole point.
                    @test n > 0.0
                    @test Scythe.ice_population_rate(q, n, 1.0, 10.0) === (0.0, 0.0)
                end
            end
            @test Scythe._ice_population_seed_local(Scythe.ISHMAEL_QSMALL, 1.0e-10,
                                                    1.0e-14, 1.0e-14, 1) === (0.0, 0.0, 0.0)
            @test Scythe._ice_population_seed_local(1.0e-5, 0.0, 0.0, 0.0, 1) ===
                  (0.0, 0.0, 0.0)
            @test length(methods(Scythe._ice_population_seed_local)) == 1
        end

        @testset "REALIZABLE: var_check is the identity on what it seeds" begin
            # The seeding's contract, and the one `:large` states for itself: the triple
            # that goes into the SRC accumulators is what the next step's re-diagnosis
            # returns, not something it has to repair.
            for k in 1:3, (q_nb, n_nb, r_nb) in ((1.0e-5, 1.0e4, 30.0e-6),
                                                 (2.0e-4, 8.0e5, 150.0e-6),
                                                 (5.0e-6, 1.0e6, 5.0e-6))
                m, apc, cpc = live_pc(q_nb, n_nb, r_nb, k)
                for q in (1.0e-7, 1.0e-5, 6.0e-3)
                    n, a, c = Scythe._ice_population_seed_local(q, m, apc, cpc, k)
                    e = Scythe._ice_effective(q, n, a, c, k)
                    @test e.ni ≈ n rtol = 1e-9
                    @test e.ai ≈ a rtol = 1e-9
                    @test e.ci ≈ c rtol = 1e-9
                    # ...and it sits inside every one of var_check's admissible ranges.
                    @test 0.55 <= e.deltastr <= 1.3
                    @test 50.0 <= e.rhobar <= Scythe.ISHMAEL_RHOI
                    @test e.rni >= Scythe.ISHMAEL_RMIN * (1.0 - 1.0e-9)
                    @test max(e.ani, e.cni) <= 1.0e-3 * (1.0 + 1.0e-9)
                end
            end
        end

        @testset "the clamps bind at both ends, and bracket :min and :large" begin
            q = 1.0e-5
            for k in 1:3
                rb = k == 3 ? Scythe.ISHMAEL_RHO_AGG : Scythe.ISHMAEL_RHOI
                # TOO COARSE: a per-crystal mass ten times `:large`'s, which is what a
                # ringing-corrupted rho/n ratio at the neighbour looks like. Clamped to
                # `:large`'s own crystal -- and with the volume moments carried down by
                # the same factor, the seed IS `:large`'s, to the last bit of var_check.
                mbig = 10.0 * Scythe.ISHMAEL_M_LARGE[k]
                nb, ab, cb = Scythe._ice_population_seed_local(q, mbig,
                                 sphere_pc(mbig, rb), sphere_pc(mbig, rb), k)
                nl, al, cl = Scythe._ice_population_seed_large(q, k)
                @test nb ≈ q / Scythe.ISHMAEL_M_LARGE[k] rtol = 1e-9
                @test nb ≈ nl rtol = 1e-9
                @test ab ≈ al rtol = 1e-9
                @test cb ≈ cl rtol = 1e-9
                # TOO FINE: a per-crystal mass below the 2 um sphere's. Clamped to
                # `ISHMAEL_M_MIN`, and the clamp SATURATES -- a tenth and a hundredth of
                # it seed the identical population, bit for bit.
                seed_sml(m) = Scythe._ice_population_seed_local(q, m,
                                  sphere_pc(m, Scythe.ISHMAEL_RHOI),
                                  sphere_pc(m, Scythe.ISHMAEL_RHOI), k)
                ns, as_, cs = seed_sml(0.1 * Scythe.ISHMAEL_M_MIN)
                @test (ns, as_, cs) === seed_sml(0.01 * Scythe.ISHMAEL_M_MIN)
                @test (ns, as_, cs) === seed_sml(Scythe.ISHMAEL_M_MIN)
                @test as_ > 0.0 && cs > 0.0
                # ...and at that end `var_check` is the STRICTER of the two limits, which
                # is what routing the clamped triple through it is for. `ISHMAEL_M_MIN` is
                # the mass of ONE 2 um sphere; the checker's own small-ice floor is on the
                # distribution's mean radius, so the smallest per-crystal mass it admits at
                # `RHOI` is `M_MIN GAMMA(nu+3)/GAMMA(nu)` -- 120x larger. `:local` seeds
                # THAT, i.e. a realizable population, where `:min` seeds `q/M_MIN` and
                # leaves the next read to repair it.
                @test ns ≈ q / (Scythe.ISHMAEL_M_MIN * gam3 * Scythe.ISHMAEL_I_GAMMNU) rtol = 1e-9
                @test ns < q / Scythe.ISHMAEL_M_MIN
                # THE BRACKET, which is what the two clamps are for: whatever the raw
                # neighbour ratio says, `:local` never seeds more crystals than `:min`
                # nor fewer than `:large`.
                for m in (1.0e-20, 0.1 * Scythe.ISHMAEL_M_MIN, 1.0e-10, mbig, 1.0e3)
                    n, _, _ = Scythe._ice_population_seed_local(q, m,
                                  sphere_pc(m, rb), sphere_pc(m, rb), k)
                    @test n <= q / Scythe.ISHMAEL_M_MIN * (1.0 + 1.0e-9)
                    @test n >= Scythe._ice_population_seed_large(q, k)[1] * (1.0 - 1.0e-9)
                end
            end
        end

        @testset "in the pass: the dead points take the live point's crystal" begin
            tau = 10.0
            ts = 0.5
            rho_a = 1.0
            q = 1.0e-5
            npts = 7
            # A column with mass everywhere and ONE live population, at point 5.
            e = Scythe._ice_effective(q, 1.0e4, 1.0e4 * (30.0e-6)^3,
                                      1.0e4 * (30.0e-6)^3, 1)
            function fixture()
                Sx = make_S(npts)
                Sx.i1q .= q
                Sx.Tk .= 258.15
                Sx.rho_d .= rho_a
                Sx.i1n[5] = e.ni * rho_a
                Sx.i1a[5] = e.ai * rho_a
                Sx.i1c[5] = e.ci * rho_a
                return Sx, zeros(Float64, length(Scythe.MC_WATER_STATS), 1)
            end

            S, st = fixture()
            Scythe._ice_population_reconcile!(S, st, 1, S.Tk, S.rho_d, tau, true, ts, :local)
            nl, al, cl = Scythe._ice_population_seed_local(q / rho_a, q / e.ni,
                                                           e.ai / e.ni, e.ci / e.ni, 1)
            for i in 1:npts
                if i == 5
                    # The live point is not dead: the device is exactly absent there.
                    @test S.SRC_i1n[i] === 0.0
                    @test S.SRC_i1a[i] === 0.0
                    continue
                end
                @test S.SRC_i1n[i] ≈ (nl * rho_a) / tau rtol = 1e-13
                @test S.SRC_i1a[i] ≈ (al * rho_a) / tau rtol = 1e-13
                @test S.SRC_i1c[i] ≈ (cl * rho_a) / tau rtol = 1e-13
                # STILL NUMBER ONLY: no mass moves, no latent heat, nothing in lambda.
                @test S.SRC_i1q[i] == 0.0
            end
            @test all(iszero, S.FRZ_NET)
            @test all(iszero, S.ICE_R)
            @test all(iszero, S.ICE_NR)
            @test all(iszero, S.ICE_C)
            @test st[Scythe.MC_POP_PTS, 1] == Float64(npts - 1)
            @test st[Scythe.MC_POP_SEED, 1] ≈ (npts - 1) * (nl * rho_a) / tau * ts rtol = 1e-13

            # ORDERS BETWEEN `:min` AND `:large`: a 30 um crystal is far above the 2 um
            # sphere and far below the 1 mm one, so the seeded number sits strictly
            # between the two ends -- which is the statement the whole option makes.
            Smin, stmin = fixture()
            Scythe._ice_population_reconcile!(Smin, stmin, 1, Smin.Tk, Smin.rho_d,
                                              tau, true, ts, :min)
            Slrg, stlrg = fixture()
            Scythe._ice_population_reconcile!(Slrg, stlrg, 1, Slrg.Tk, Slrg.rho_d,
                                              tau, true, ts, :large)
            @test stlrg[Scythe.MC_POP_SEED, 1] < st[Scythe.MC_POP_SEED, 1] <
                  stmin[Scythe.MC_POP_SEED, 1]
            @test st[Scythe.MC_POP_SEED, 1] > 1.0e3 * stlrg[Scythe.MC_POP_SEED, 1]
            @test st[Scythe.MC_POP_SEED, 1] < 1.0e-3 * stmin[Scythe.MC_POP_SEED, 1]
            # The census of the DEFECT is the same in all three: the seeding changes the
            # answer, never the measurement that provoked it.
            @test st[Scythe.MC_POP_MAX, 1] == stmin[Scythe.MC_POP_MAX, 1] ==
                  stlrg[Scythe.MC_POP_MAX, 1]
            @test st[Scythe.MC_POP_PTS, 1] == stmin[Scythe.MC_POP_PTS, 1]

            # THE FALLBACK. A column in which the species is live NOWHERE is the pure-dead
            # case: there is no habit to inherit, and `:local` is `:min` there BITWISE.
            Sdead, stdead = fixture()
            Sdead.i1n .= 0.0; Sdead.i1a .= 0.0; Sdead.i1c .= 0.0
            Scythe._ice_population_reconcile!(Sdead, stdead, 1, Sdead.Tk, Sdead.rho_d,
                                              tau, true, ts, :local)
            Sdm, stdm = fixture()
            Sdm.i1n .= 0.0; Sdm.i1a .= 0.0; Sdm.i1c .= 0.0
            Scythe._ice_population_reconcile!(Sdm, stdm, 1, Sdm.Tk, Sdm.rho_d,
                                              tau, true, ts, :min)
            for nm in (:SRC_i1n, :SRC_i1a, :SRC_i1c, :SRC_i1q)
                @test getproperty(Sdead, nm) == getproperty(Sdm, nm)
            end
            @test stdead == stdm

            # TWO live points, with DIFFERENT crystals: the seed follows the NEARER one,
            # and a tie goes to the lower index. This is what makes it a local rule.
            e2 = Scythe._ice_effective(q, 1.0e6, 1.0e6 * (8.0e-6)^3,
                                       1.0e6 * (8.0e-6)^3, 1)
            S2 = make_S(npts)
            S2.i1q .= q; S2.Tk .= 258.15; S2.rho_d .= rho_a
            S2.i1n[2] = e.ni * rho_a;  S2.i1a[2] = e.ai * rho_a;  S2.i1c[2] = e.ci * rho_a
            S2.i1n[6] = e2.ni * rho_a; S2.i1a[6] = e2.ai * rho_a; S2.i1c[6] = e2.ci * rho_a
            st2 = zeros(Float64, length(Scythe.MC_WATER_STATS), 1)
            Scythe._ice_population_reconcile!(S2, st2, 1, S2.Tk, S2.rho_d, tau, true,
                                              ts, :local)
            n_a = Scythe._ice_population_seed_local(q / rho_a, q / e.ni,
                                                     e.ai / e.ni, e.ci / e.ni, 1)[1]
            n_b = Scythe._ice_population_seed_local(q / rho_a, q / e2.ni,
                                                     e2.ai / e2.ni, e2.ci / e2.ni, 1)[1]
            @test !(n_a ≈ n_b)                          # the two crystals really differ
            for (i, nref) in ((1, n_a), (3, n_a), (4, n_a), (5, n_b), (7, n_b))
                @test S2.SRC_i1n[i] ≈ (nref * rho_a) / tau rtol = 1e-13
            end

            # ABOVE T_0 the seeding is not consulted: that branch moves mass to the rain.
            Sw = make_S(2)
            Sw.i1q .= q; Sw.Tk .= 290.0; Sw.rho_d .= rho_a
            stw = zeros(Float64, length(Scythe.MC_WATER_STATS), 1)
            Scythe._ice_population_reconcile!(Sw, stw, 1, Sw.Tk, Sw.rho_d, tau, true,
                                              ts, :local)
            @test all(Sw.SRC_i1q .≈ -q / tau)
            @test all(iszero, Sw.SRC_i1n)
            @test stw[Scythe.MC_POP_SEED, 1] == 0.0

            # THE SPAN SWEEP AND THE WALK ALLOCATE NOTHING: three scalar sweeps and a
            # bounded index walk, with no per-column workspace of their own.
            callL(Sx, stx) = Scythe._ice_population_reconcile!(Sx, stx, 1, Sx.Tk,
                                 Sx.rho_d, tau, true, ts, :local)
            callL(S, st); callL(Sdead, stdead)
            @test @allocated(callL(S, st)) == 0
            @test @allocated(callL(Sdead, stdead)) == 0
        end
    end

    @testset "above T_0: the mass returns to the rain, and L_f goes with it" begin
        tau = 10.0
        ts = 0.5
        S = make_S(2)
        S.i1q .= 4.0e-5
        S.i1n .= 0.0
        S.i1a .= 7.0e-13
        S.i1c .= 3.0e-13
        S.Tk .= 290.0
        S.rho_d .= 1.0
        st = zeros(Float64, length(Scythe.MC_WATER_STATS), 1)
        Scythe._ice_population_reconcile!(S, st, 1, S.Tk, S.rho_d, tau, true, ts)

        rate = 4.0e-5 / tau
        # TOTAL WATER IS CONSERVED TO THE LAST BIT: what the ice slot loses the rain slot
        # gains, formed from the one number, so the transfer cannot manufacture vapor
        # through `res_rho_t` the way an unbalanced exchange would.
        @test all(S.SRC_i1q .≈ -rate)
        @test all(S.ICE_R .≈ rate)
        for i in eachindex(S.ICE_R)
            @test S.SRC_i1q[i] + S.ICE_R[i] === 0.0
        end
        # `FRZ_NET ≡ −(ICE_C + ICE_R)` still holds exactly, and it is NEGATIVE — ice
        # becoming rain ABSORBS the latent heat of fusion, which is what a melt does.
        for i in eachindex(S.FRZ_NET)
            @test S.FRZ_NET[i] === -(S.ICE_C[i] + S.ICE_R[i])
            @test S.FRZ_NET[i] < 0.0
        end
        # The rain NUMBER is seeded the way the two-moment closure seeds a mass source with
        # no number of its own: autoconversion's 25 um drop (`RAIN_2M_M_AUTO`).
        @test all(S.ICE_NR .≈ rate / Scythe.RAIN_2M_M_AUTO)
        # The dead species leaves in one piece: both volume moments relax by the SAME
        # fraction 1/tau as the mass (leg A's shared-factor rule), so the per-particle
        # state of what is left is exactly invariant. The number is already <= 0 and has
        # nothing to relax.
        @test all(S.SRC_i1a .≈ -7.0e-13 / tau)
        @test all(S.SRC_i1c .≈ -3.0e-13 / tau)
        @test all(iszero, S.SRC_i1n)
        @test st[Scythe.MC_POP_RAIN, 1] ≈ 2.0 * rate * ts rtol = 1e-14
        @test st[Scythe.MC_POP_SEED, 1] == 0.0
    end

    @testset "the off switch measures and does not remove" begin
        tau = 10.0
        S = make_S(2)
        S.i1q .= 1.0e-5; S.i1n .= 0.0
        S.i2q .= 2.0e-5; S.i2n .= 0.0
        S.Tk .= [258.15, 290.0]               # one of each branch
        S.rho_d .= 1.0
        st = zeros(Float64, length(Scythe.MC_WATER_STATS), 1)
        Scythe._ice_population_reconcile!(S, st, 1, S.Tk, S.rho_d, tau, false, 0.5)
        for nm in (:SRC_i1q, :SRC_i1n, :SRC_i1a, :SRC_i1c,
                   :SRC_i2q, :SRC_i2n, :SRC_i2a, :SRC_i2c,
                   :ICE_R, :ICE_NR, :ICE_C, :FRZ_NET)
            @test all(getproperty(S, nm) .=== 0.0)
        end
        # ...but the defect was still measured: the off switch is a forensic tool, not a
        # blindfold, exactly as `options[:ice_anchor_source] = false` is.
        @test st[Scythe.MC_POP_MAX, 1] == 2.0e-5
        @test st[Scythe.MC_POP_PTS, 1] == 4.0        # two species x two gridpoints
        @test st[Scythe.MC_POP_RAIN, 1] == 0.0
        @test st[Scythe.MC_POP_SEED, 1] == 0.0
    end

    @testset "one-sidedness: a fully live column is untouched, bitwise" begin
        tau = 10.0
        S = make_S(4)
        S.i1q .= 1.0e-5; S.i1n .= 5.0e4; S.i1a .= 1.0e-12; S.i1c .= 1.0e-12
        S.i2q .= 3.0e-6; S.i2n .= 2.0e4; S.i2a .= 5.0e-13; S.i2c .= 5.0e-13
        S.Tk .= [250.0, 265.0, 280.0, 300.0]      # both branches represented
        S.rho_d .= 1.0
        st = zeros(Float64, length(Scythe.MC_WATER_STATS), 1)
        Scythe._ice_population_reconcile!(S, st, 1, S.Tk, S.rho_d, tau, true, 0.5)
        for nm in (:SRC_i1q, :SRC_i1n, :SRC_i1a, :SRC_i1c,
                   :SRC_i2q, :SRC_i2n, :SRC_i2a, :SRC_i2c,
                   :SRC_i3q, :SRC_i3n, :SRC_i3a, :SRC_i3c,
                   :ICE_R, :ICE_NR, :ICE_C, :FRZ_NET)
            @test all(getproperty(S, nm) .=== 0.0)
        end
        @test all(view(st, Scythe.MC_POP_MAX:Scythe.MC_POP_LAST, 1) .== 0.0)
    end
end
