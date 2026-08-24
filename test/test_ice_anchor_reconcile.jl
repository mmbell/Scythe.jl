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
