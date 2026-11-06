# Scalar tests for the exponential (ETD) Adams-Bashforth integrator of the stiff
# supersaturation relaxation pair — TeX §"Integration of the relaxation pair in the stiff
# limit", implemented by `etd_step_weights` / `relaxation_adjustment_qss!` in
# src/moist_compressible.jl.
#
# Everything here exercises the PURE coefficient math on scalars, so the properties the TeX
# argues the scheme has are checked at the one place they are actually decided, without a
# model tile in the way. The driver-level gates (exchange closure, the dry bitwise path, the
# wbf_qs fixed point on a real column) live in test_moist_compressible.jl.

using Test
using Scythe

# The reference implementations, in BigFloat with 256 bits of precision, straight from the
# TeX's definitions. `J_k(x) = ∫₀¹ θ^k e^{−x(1−θ)} dθ`; `D = (1/3 − J2)/x`.
_bigJ(x::BigFloat) = begin
    e = exp(-x)
    j0 = (one(x) - e) / x
    j1 = (x - one(x) + e) / x^2
    j2 = (x^2 - 2 * x + 2 - 2 * e) / x^3
    d = (BigFloat(1) / 3 - j2) / x
    (j0, j1, j2, d)
end

@testset "ETD-AB3 relaxation integrator" begin

    @testset "kernel moments match BigFloat across the series/closed-form switch" begin
        # The switch is at x = 1. Straddle it densely, and go far out on both sides: the
        # small-x branch has to survive x -> 0 (where the closed forms cancel catastrophically)
        # and the closed forms have to survive x -> huge (where the series diverges).
        setprecision(BigFloat, 256) do
            for x in (1.0e-12, 1.0e-8, 1.0e-6, 1.0e-4, 0.01, 0.1, 0.5, 0.9, 0.99,
                      0.999, 0.9999, 1.0, 1.0001, 1.001, 1.01, 1.1, 2.0, 5.0,
                      10.0, 100.0, 1.0e4, 1.0e6)
                got = Scythe._etd_moments(x)
                ref = _bigJ(BigFloat(x))
                for k in 1:4
                    @test abs(got[k] - Float64(ref[k])) <=
                          1.0e-14 * max(abs(Float64(ref[k])), 1.0e-300)
                end
            end
        end
    end

    @testset "x = 0 reduces to the classical multistep EXACTLY" begin
        # "wherever λ = 0 ... Eq. etd_ab3 IS the third-order multistep, bitwise" (TeX, first
        # property). `===` deliberately: one ulp is a failure here, because the dry path is
        # supposed to be bit-for-bit the code that had no exponential in it.
        (emx, b1, b2, b3, j0, g1, g2, g3) = Scythe.etd_step_weights(0.0, 3)
        @test emx === 1.0
        @test b1 === 23.0 / 12.0
        @test b2 === -16.0 / 12.0
        @test b3 === 5.0 / 12.0
        # The step-mean weights at λ = 0 are ∫₀¹(1−θ)N(θ)dθ for the same quadratic extrapolant.
        @test j0 === 1.0
        @test g1 === 19.0 / 24.0
        @test g2 === -5.0 / 12.0
        @test g3 === 1.0 / 8.0

        (emx2, b1_2, b2_2, b3_2, j0_2, g1_2, g2_2, g3_2) = Scythe.etd_step_weights(0.0, 2)
        @test (emx2, b1_2, b2_2, b3_2) === (1.0, 1.5, -0.5, 0.0)
        @test (j0_2, g1_2, g2_2, g3_2) === (1.0, 2.0 / 3.0, -1.0 / 6.0, 0.0)

        (emx1, b1_1, b2_1, b3_1, j0_1, g1_1, g2_1, g3_1) = Scythe.etd_step_weights(0.0, 1)
        @test (emx1, b1_1, b2_1, b3_1) === (1.0, 1.0, 0.0, 0.0)
        @test (j0_1, g1_1, g2_1, g3_1) === (1.0, 0.5, 0.0, 0.0)
    end

    @testset "the classical weights are the x -> 0 LIMIT as well as the branch" begin
        # Continuity across the branch: at x = 1e-10 the computed weights must already be the
        # classical ones to rounding, so the `x == 0` special case is a statement about the
        # last ulp and not a discontinuity.
        (_, b1, b2, b3, j0, g1, g2, g3) = Scythe.etd_step_weights(1.0e-10, 3)
        @test b1 ≈ 23.0 / 12.0 atol = 1.0e-9
        @test b2 ≈ -16.0 / 12.0 atol = 1.0e-9
        @test b3 ≈ 5.0 / 12.0 atol = 1.0e-9
        @test j0 ≈ 1.0 atol = 1.0e-9
        @test g1 ≈ 19.0 / 24.0 atol = 1.0e-9
        @test g2 ≈ -5.0 / 12.0 atol = 1.0e-9
        @test g3 ≈ 1.0 / 8.0 atol = 1.0e-9
    end

    @testset "constant N is integrated EXACTLY at every x" begin
        # With N constant the backward differences vanish and Eq. etd_ab3 must collapse to
        # Q^{n+1} = e^{−x}Q^n + (1−e^{−x})N/λ — the exact solution — which is the statement
        # b1+b2+b3 == J0. Checked at every startup branch too, since all three extrapolants
        # reproduce a constant.
        dt = 0.5
        for x in (1.0e-6, 0.1, 1.0, 10.0, 1.0e6)
            lam = x / dt
            for t in (1, 2, 3)
                (emx, b1, b2, b3, j0, g1, g2, g3) = Scythe.etd_step_weights(x, t)
                for (q0, nval) in ((3.0e-4, -7.0e-5), (-2.0e-4, 1.5e-4), (0.0, 1.0))
                    got = (emx * q0) + dt * ((b1 + b2 + b3) * nval)
                    want = exp(-x) * q0 + (1.0 - exp(-x)) * (nval / lam)
                    @test isapprox(got, want; rtol = 1.0e-13,
                                   atol = 1.0e-13 * max(abs(q0), abs(nval / lam)))
                    # ... and the step-mean of the same trajectory, Q̄ = J0 Q^n + Δt J1 N,
                    # which is what every consumer of the phase-change rates is fed.
                    qbar = (j0 * q0) + dt * ((g1 + g2 + g3) * nval)
                    (rj0, rj1, _, _) = Scythe._etd_moments(x)
                    @test isapprox(qbar, rj0 * q0 + dt * rj1 * nval; rtol = 1.0e-13,
                                   atol = 1.0e-16)
                end
            end
        end
    end

    @testset "the stiff limit lands on the quasi-steady state N/λ" begin
        # "As x -> ∞ the update reaches the quasi-steady state in a single step regardless of
        # history" (TeX, third property). Not asymptotically — at machine precision.
        dt = 1.0
        for x in (1.0e3, 1.0e6, 1.0e12)
            lam = x / dt
            (emx, b1, b2, b3, j0, g1, g2, g3) = Scythe.etd_step_weights(x, 3)
            nval = 4.0e-5
            # Constant N, and a wildly wrong starting point: the answer must be N/λ either way.
            for q0 in (0.0, 1.0, -1.0e3)
                got = (emx * q0) + dt * ((b1 + b2 + b3) * nval)
                @test isapprox(got, nval / lam; rtol = 1.0e-12)
            end
            # The STEP-MEAN reaches the same limit, with the classical weights divided by x —
            # which is what makes every consumer see Q_ss^qs rather than Q_ss^n. The approach
            # is O(1/x) in the relative sense (the next term of each g is O(1/x²)), so the
            # tolerance is written as such rather than as a fixed number.
            rt = 50.0 / x
            @test isapprox(g1, (23.0 / 12.0) / x; rtol = rt)
            @test isapprox(g2, (-16.0 / 12.0) / x; rtol = rt)
            @test isapprox(g3, (5.0 / 12.0) / x; rtol = rt)
        end
    end

    @testset "the step-mean satisfies the discrete budget identity" begin
        # Eq. qss_stepmean: (Q^{n+1} − Q^n)/Δt = N̄ − λ Q̄, with N̄ the CLASSICAL-weight
        # combination of the same N history. This is the identity that makes the exchange
        # close, and the g-weights are the removable-singularity form of it.
        dt = 0.4
        n0, n1, n2 = 3.1e-5, -1.7e-5, 8.0e-6
        for x in (1.0e-3, 0.3, 1.0, 3.0, 50.0)
            lam = x / dt
            (emx, b1, b2, b3, j0, g1, g2, g3) = Scythe.etd_step_weights(x, 3)
            q0 = 2.4e-4
            qnp1 = (emx * q0) + dt * ((b1 * n0) + (b2 * n1) + (b3 * n2))
            qbar = (j0 * q0) + dt * ((g1 * n0) + (g2 * n1) + (g3 * n2))
            nbar = ((23.0 * n0) - (16.0 * n1) + (5.0 * n2)) / 12.0
            @test isapprox((qnp1 - q0) / dt, nbar - lam * qbar;
                           rtol = 1.0e-11, atol = 1.0e-18)
        end
        # ... and at the two startup branches, with their own classical weights.
        for (t, nbar) in ((2, ((3.0 * n0) - n1) / 2.0), (1, n0))
            for x in (1.0e-3, 0.7, 20.0)
                lam = x / dt
                (emx, b1, b2, b3, j0, g1, g2, g3) = Scythe.etd_step_weights(x, t)
                q0 = -1.1e-4
                qnp1 = (emx * q0) + dt * ((b1 * n0) + (b2 * n1) + (b3 * n2))
                qbar = (j0 * q0) + dt * ((g1 * n0) + (g2 * n1) + (g3 * n2))
                @test isapprox((qnp1 - q0) / dt, nbar - lam * qbar;
                               rtol = 1.0e-11, atol = 1.0e-18)
            end
        end
    end

    @testset "third-order convergence on a smooth N(t)" begin
        # dQ/dt = sin(t) − λQ, Q(0) = Q0, integrated to t = 2 with the same Euler/AB2/AB3
        # startup ladder `explicit_timestep` uses. λ moderate on purpose: the exponential is
        # exact, so what is being measured is the order of the EXTRAPOLANT, which is the part
        # that has to stay third order for the scheme to converge to the same equation.
        lam = 2.0
        q0 = 0.3
        exact(tt) = ((lam * sin(tt) - cos(tt)) / (1.0 + lam^2)) +
                    (q0 + 1.0 / (1.0 + lam^2)) * exp(-lam * tt)
        # EXACT history: `N` at the two fictitious previous levels is known analytically, so
        # this measures the order of the ETD-AB3 step itself. (The `t = 1` Euler branch of the
        # startup ladder has O(Δt²) local error and would cap the observed order of a
        # cold-started run at 2 no matter how good the AB3 step is — that is a property of the
        # startup, is shared with `explicit_timestep`'s own ladder, and is checked separately
        # below.)
        function integrate(nsteps; cold::Bool)
            dt = 2.0 / nsteps
            x = lam * dt
            q = q0
            nm1 = cold ? 0.0 : sin(-dt)
            nm2 = cold ? 0.0 : sin(-2.0 * dt)
            for step in 1:nsteps
                tn = (step - 1) * dt
                n0 = sin(tn)
                branch = cold ? min(step, 3) : 3
                (emx, b1, b2, b3, _, _, _, _) = Scythe.etd_step_weights(x, branch)
                q = (emx * q) + dt * ((b1 * n0) + (b2 * nm1) + (b3 * nm2))
                nm2 = nm1
                nm1 = n0
            end
            return abs(q - exact(2.0))
        end
        e1 = integrate(80; cold = false)
        e2 = integrate(160; cold = false)
        e3 = integrate(320; cold = false)
        @test log2(e1 / e2) >= 2.7
        @test log2(e2 / e3) >= 2.7
        # The cold-started ladder converges too, at the order its Euler first step allows.
        c1 = integrate(80; cold = true)
        c2 = integrate(160; cold = true)
        @test log2(c1 / c2) >= 1.8
    end

    @testset "the exponential is exact where the extrapolant is trivial" begin
        # The homogeneous problem (N ≡ 0) must be integrated with NO error at any step size:
        # that is the property the multistep does not have and the whole point of the change.
        lam = 500.0
        dt = 1.0
        q = 1.0
        for step in 1:5
            (emx, b1, b2, b3, _, _, _, _) = Scythe.etd_step_weights(lam * dt, min(step, 3))
            q = (emx * q) + dt * ((b1 * 0.0) + (b2 * 0.0) + (b3 * 0.0))
        end
        @test isapprox(q, exp(-lam * dt * 5); rtol = 1.0e-12, atol = 1.0e-300)
        # The classical multistep at the same λΔt = 500 would have diverged by ~10^13; state
        # the contrast so the test is not merely "a small number is small".
        @test q < 1.0e-300
    end
end
