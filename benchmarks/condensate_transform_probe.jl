# =============================================================================
# Does the transformed condensate SURVIVE BEING INTEGRATED?
#
# reference/HANDOFF_CONDENSATE_REPRESENTATION.md, Stage 1b. The companion script
# benchmarks/condensate_transform_diagnostic.jl scores REPRESENTATION -- what one
# fit->reconstruct round trip does to a saved field. It cannot see the two things
# that would actually sink the design:
#
#   (a) the RATCHET, which needs the physics re-adding the spike between refits;
#   (b) SOURCE STIFFNESS, which is a property of the time step, not of the fit.
#
# This script runs a single column of the model's own vertical basis through the
# model's own AB3 increment with a refit every step, under prescribed transport
# and the model's own warm-rain rate kernels. It runs no model and changes
# nothing in src/.
#
# ── What is being integrated ─────────────────────────────────────────────────
#
# The rho-space (baseline) arm is the slot-9 continuity of
# src/moist_compressible.jl:2628 restricted to one column:
#
#     d rho / dt = -w dz(rho) - rho dz(w) + S
#
# The transformed arm predicts n = g(rho) and recovers rho = f(n):
#
#     d n / dt   = -w dz(n)  + J(rho) * ( -rho dz(w) + S )
#
# Advection is transform-invariant, so it is taken from n's OWN spline gradient
# -- one fit, one field, no chain rule. Only divergence and sources pick up the
# Jacobian, and J is evaluated at the RECOVERED DENSITY clamped at zero for the
# argument only (Ooyama 2001 Eq. 4.21-4.23), which is what keeps it bounded.
#
# ── The arms ─────────────────────────────────────────────────────────────────
#
#   TRANSPORT   prescribed updraft, no sources. Reproduces the per-refit ringing
#               and shows whether it accumulates under transport alone.
#   DRAIN       transport + the model's own autoconversion_density and
#               collection_density as the cloud sink. This is the G2 mechanism:
#               the rates are max(rho,0)-guarded, so they drain the POSITIVE
#               cloud and cannot touch the negative part. If the reservoir is
#               real, it appears here and not in TRANSPORT.
#   NUCLEATION  rho = 0 everywhere, a constant source switched on in three
#               adjacent cells. Isolates S/f' -- the pole that disqualifies any
#               family whose Jacobian is unbounded at the cloud edge. The source
#               is PRESCRIBED on purpose: the question is how the control
#               variable responds to a source at rho = 0, and the answer must
#               not depend on which closure produced it.
#
# ── Falsification criteria (any one => the candidate is out) ─────────────────
#
#   F1  overshoot: the recovered rho step exceeds the rho-space arm's step by
#       more than 2x at any point and time.
#   F2  blow-up: the integration fails to complete, or the peak exceeds 3x the
#       rho-space arm's peak.
#   F3  nucleation lag: time from S > 0 to rho > 1e-6 exceeds 10 s.
#   F4  mass: per-refit leak beyond the one-time sign correction.
#
# ── WHAT IT MEASURED (2026-07-29, mu = 1e-7, 300 mish points, dz 250 m,
#    ts 0.3 s, 12 000 steps = 3600 s) ────────────────────────────────────────
#
# THE DISCRIMINATOR IS NUCLEATION, AND IT IS BRUTAL.
#
#   candidate         min_rho      max_rho     neg_mass   mass_drift   verdict
#   identity       -1.2812e-03   1.2294e-02   8.6354e-01   +2.300e+01   baseline
#   H-smooth       -9.9991e-08   1.2167e-02   6.2883e-04   +2.369e+01   PASSES
#   Q-quad          0.0000e+00   9.6636e+24   0.0000e+00   +1.905e+28   F1 F2
#   softplus        0.0000e+00   9.9899e+06   0.0000e+00   +1.950e+10   F1 F2
#
# Q-quad and softplus are INDISTINGUISHABLE from H on the transport and drain
# arms (mass drift 1.077 and -1.645 for all three, to four digits). They fail
# ONLY when a source acts at rho = 0. That is the S/f' pole, isolated: any
# monotone f whose range is (0, inf) has f' -> 0 there, so the source quotient is
# unbounded, and an explicit step in the control variable overshoots by 25 orders
# of magnitude (Q-quad) or 9 (softplus). H's Jacobian is bounded in [0.5, 1] by
# Ooyama Eq. 4.23 and its peak lands within 1 % of the rho-space arm's.
#
# The offline representation study could not have found this: it scores fits, and
# all three families fit equally well. It is a property of the time step.
#
# THE RESERVOIR IS REPRODUCED, AND SO IS ITS ASYMMETRY.
#   TRANSPORT: identity accumulates neg_mass 4.24e-1 against pos_mass 2.42e+0
#              (17.5 %); H holds min_rho at -mu exactly and neg_mass at 8.1e-4,
#              520x smaller, with the peak identical to five digits (2.9376e-3).
#   DRAIN:     with the model's own autoconversion_density + collection_density
#              draining the cloud, identity's neg_mass is 30 % of what positive
#              cloud remains. The rates are max(rho,0)-guarded, so they consume
#              the positive part and cannot touch the negative part. That is the
#              G2 mechanism, in one column.
#
# F4 IS THE ONE REAL COST, AND IT IS BOUNDED. f is convex near zero, so a
# zero-mean oscillation in n maps to a positive-mean perturbation in rho: the
# transform RECTIFIES ringing into mass, the mirror image of the reservoir.
# Measured on the transport arm, cumulative mass added at 900/1800/2700/3600 s:
#
#     +4.942e-01   +7.567e-01   +9.367e-01   +1.077e+00
#
# The quarterly increments are 0.494, 0.263, 0.180, 0.140 -- DECELERATING, not
# linear, because once the spike relaxes to something the basis can represent the
# ringing stops. Contrast the identity arm's negative reservoir, which the state
# census in condensate_transform_diagnostic.jl shows growing monotonically over
# the whole run. And on the nucleation arm, where a real source dominates, H's
# total mass tracks the rho-space arm to 3 % (+2.369e+01 vs +2.300e+01).
#
# Caveat on the drain arm: it carries autoconversion and collection only. The
# real model also EVAPORATES subsaturated cloud through qss_condensation_rates,
# which is the dominant sink for thin cloud and which this probe cannot exercise
# without a full thermodynamic state. The spurious positive mass H retains is
# therefore removed by more physics in the model than it is here -- another
# reason the F4 numbers above are an upper bound.
#
# THE LIMITER PASSES THIS PROBE TOO -- AND THAT IS THE IMPORTANT NEGATIVE RESULT.
# POSITIVITY=k, integrated on the same three arms:
#
#   arm         min_rho      max_rho     neg_mass   mass_drift   shortfall
#   TRANSPORT  1.99e-46    2.9267e-03   0.0000e+00   +4.273e-07   0
#   DRAIN      1.99e-46    2.9260e-03   0.0000e+00   -1.663e+00   0
#   NUCLEATION 0.00e+00    1.0537e-02   0.0000e+00   +2.300e+01   0
#
# In a single column under prescribed transport the limiter is not merely
# adequate, it is BETTER than the transform: positivity exact, mass conserved to
# 4.3e-7 against the transform's +1.077, no blow-up anywhere, and it creates no
# mass at all (shortfall 0 on every arm). Offline, on one refit, the two were
# already indistinguishable (max|drho| 4.46e-5 vs 4.45e-5, 0.134 K each).
#
# So NOTHING measurable offline discriminates the limiter from the transform, and
# the campaign's premise -- that the POSITIVITY=1 detonation lives in the manner
# of enforcement -- is NOT supported by any measurement available without a run.
#
# The reason is structural and worth stating: this probe PRESCRIBES w. The loop
# that actually detonates the model is
#     d rho_c -> d T (via L_v/D) -> d rho_vs -> d Q_ss -> d condensation
#             -> d buoyancy -> d w -> d rho_c
# and a probe with imposed transport cannot close it. Both schemes inject into
# that loop; the only measured difference is HOW MUCH temperature each moves per
# refit, where the transform is 2.25x smaller than the two-leg limiter (0.134 K
# vs 0.301 K) and indistinguishable from the one-leg limiter.
#
# THE TWO H VARIANTS ARE INTERCHANGEABLE (added 2026-07-30, after the first
# write-up singled out H-smooth on an argument rather than a measurement).
# H-Ooyama is the PUBLISHED quasi-inverse (Eq. 4.20): same forward map, same
# Jacobian, but rho pinned at exactly 0 for n <= 0 instead of relaxing to -mu,
# at the cost of a kink in f sited exactly at the cloud edge. Run on all three
# arms it is indistinguishable from H-smooth:
#
#   arm         metric        H-smooth      H-Ooyama
#   TRANSPORT   max_rho      2.9376e-03    2.9376e-03
#               mass_drift    +1.077e+00    +1.077e+00
#               min_rho      -9.9975e-08    0.0000e+00
#               neg_mass      8.0914e-04    0.0000e+00
#   DRAIN       pos_mass      3.4900e-01    3.4911e-01
#               mass_drift    -1.646e+00    -1.645e+00
#   NUCLEATION  max_rho      1.2167e-02    1.2166e-02
#               lag              0.30 s        0.30 s
#
# The kink causes no dynamical trouble at all. Both pass F1-F4 on every arm.
#
# THE CONTROL-VARIABLE EXCURSION -- the transform's own possible failure mode, and
# the only quantity that could have separated the two. min(n) over the run:
#
#   arm         identity     H-smooth     H-Ooyama       Q-quad     softplus   POS=k
#   TRANSPORT  -4.3339e-4   -2.0125e-4   -2.0123e-4   -4.0256e-4   -4.0521e-4  +2e-46
#   DRAIN      -2.1568e-4   -1.0071e-4   -1.0072e-4   -2.0152e-4   -2.0160e-4  +2e-46
#   NUCLEATION -1.2812e-3   -5.6572e-4   -5.6563e-4   -2.4795e+24  -2.4800e+06 +0.0
#
# The two H variants agree to four or five significant figures on every arm, so
# the choice between them is free and must be made on other grounds.
#
# Two things this table says that matter beyond that choice:
#
#   * THE RATCHET DOES NOT RELOCATE INTO n -- IT IS THE SAME EXCURSION, HALVED BY
#     OOYAMA'S FACTOR 0.5 (n ~ rho/2 in the linear regime). identity's rho reaches
#     -4.334e-4; H's n reaches -2.013e-4, i.e. 2.15x smaller, which is exactly that
#     factor. In rho-equivalent terms the excursion is UNCHANGED. The transform does
#     not reduce the ringing -- it makes the ringing harmless, because rho = f(n) is
#     bounded below by -mu whatever n does. That is the whole claim, and it should
#     not be overstated into a claim about amplitude.
#   * THE NUCLEATION LAG IS REAL AND MUST BE CENSUSED IN THE MODEL. The F3 number
#     above (0.30 s, one step) is measured from n = 0 and is therefore a best case.
#     Once n has ratcheted to -5.66e-4, a source of 1e-5 kg/m^3/s needs ~57 s to
#     bring the recovered density back above 1e-6. For contrast the identity arm's
#     rho reaches -1.28e-3 and needs ~128 s to climb back -- 2.2x longer, and
#     carrying a -54 K-class cold anomaly the whole time, where the transformed
#     point simply reads zero cloud. Better, but not free.
#   * POSITIVITY=k holds min(n) at +2e-46: the limiter genuinely REMOVES the
#     excursion rather than relocating it. That is its one structural advantage,
#     and the o01 runs price it at a 250-1000x irreversible entropy source.
#
# VERDICT. Among the TRANSFORMS, the OOYAMA BIASED HYPERBOLIC FAMILY is the only
# survivor: Q-quad and softplus are out on source stiffness, log was already out
# on representation. Within the family H-smooth and H-Ooyama are measurably
# equivalent; H-Ooyama is preferred as the shipped form because it delivers
# rho >= 0 exactly at no measured dynamical cost and is the published, validated
# map, with H-smooth retained as a one-branch variant for any future consumer that
# needs f'' at the cloud edge (the water Laplacians, currently disabled).
# Whether a transform is needed AT ALL over the existing limiter is a question no
# offline measurement here can answer, and it must be settled by running the
# limiter under the current head -- the o01 harness already supports it
# (SCYTHE_O01_POSITIVITY=ck and =1). The comment at benchmarks/o01_rainfall.jl:237
# recording "max_w 2.7 -> 36 with either leg alone" predates BOTH the AB3-exact
# depletion caps (402da30) and the blended vapor retrieval (2a90b0e), so it is
# not evidence about the code as it stands.
#
# Usage:
#   julia --project=. benchmarks/condensate_transform_probe.jl [outdir]
# =============================================================================

using Scythe, Springsteel, CSV, DataFrames, Printf, Statistics
using Springsteel.CubicBSpline

const OUTDIR = length(ARGS) >= 1 ? ARGS[1] :
    joinpath(@__DIR__, "output", "o01_rainfall_quick_mc_rirk")

# The model's own vertical basis for o01 (benchmarks/o01_rainfall.jl:112-131,
# 273-288): 25 km lid, 100 cells, mubar 3 => 300 mish points, l_q 2.0, and
# rho_c's BCB/BCT are NeumannBC == R1T1.
const NC   = 100
const KMAX = 25.0e3
const TS   = 0.3            # quick-mode timestep
const NSTEP = 12_000        # 3600 s, the full quick run

spline_params(; l_q = 2.0) =
    SplineParameters(xmin = 0.0, xmax = KMAX, num_cells = NC, mubar = 3, l_q = l_q,
                     BCL = CubicBSpline.R1T1, BCR = CubicBSpline.R1T1)

# ── the candidate transforms (identical algebra to the diagnostic script) ─────

struct Transform
    name::String
    g::Function; f::Function; J::Function
    bounded::Bool          # use the spline positivity limiter instead of a transform
end
Transform(name, g, f, J) = Transform(name, g, f, J, false)

# The rival enforcement: rho-space dynamics with Springsteel's coefficient box
# constraint on the k leg (the only leg a single column has). This is handoff
# Experiment 2 done DYNAMICALLY, which is the form that discriminates -- offline,
# on one refit of a clean field, the limiter and the transform make
# indistinguishable modifications (max|drho| 4.46e-5 vs 4.45e-5, 0.134 K each;
# see condensate_transform_diagnostic.jl's EXPERIMENT 2 table). The difference,
# if there is one, is in what 12 000 of those do to a trajectory.
limiter_tf() = Transform("POSITIVITY=k", r -> r, n -> n, _ -> 1.0, true)

bhyp(rho, mu)  = 0.5 * ((rho + mu) - (mu * mu) / (rho + mu))
ahyp(n, mu)    = sqrt(n * n + mu * mu) + n - mu
dbhyp(rho, mu) = 0.5 * (1.0 + (mu * mu) / ((rho + mu) * (rho + mu)))

identity_tf() = Transform("identity", r -> r, n -> n, _ -> 1.0)
hyp(mu)       = Transform("H-smooth(mu=$mu)", r -> bhyp(r, mu), n -> ahyp(n, mu),
                          r -> dbhyp(max(r, 0.0), mu))
# Ooyama's PUBLISHED quasi-inverse (Eq. 4.20): the strict inverse above for n >= 0,
# identically zero below. Same forward map, same Jacobian; the only difference is
# that rho is pinned at exactly 0 instead of relaxing to -mu, at the cost of a kink
# in f at n = 0 (f'(0-) = 0, f'(0+) = 1) sited exactly at the cloud edge. It is NOT
# a state repair -- the state is n, and n is never modified ("n itself is untouched,
# so that the effect of m adjustments does not accumulate in the predicted n").
hyp_ooyama(mu) = Transform("H-Ooyama(mu=$mu)", r -> bhyp(r, mu),
                           n -> (n <= 0.0 ? 0.0 : ahyp(n, mu)),
                           r -> dbhyp(max(r, 0.0), mu))
quad_f(n, mu) = n <= -mu ? 0.0 : (n >= mu ? n : (n + mu)^2 / (4mu))
quad_g(r, mu) = r >= mu ? r : 2.0 * sqrt(mu * max(r, 0.0)) - mu
quad_J(r, mu) = (x = max(r, 0.0); x >= mu ? 1.0 : (x <= 0.0 ? 1e30 : sqrt(mu / x)))
quadratic(mu) = Transform("Q-quad(mu=$mu)", r -> quad_g(r, mu), n -> quad_f(n, mu),
                          r -> quad_J(r, mu))
softplus_g(r, mu) = (x = r / mu; x < 1e-12 ? mu * log(1e-12) :
                                 (x > 40.0 ? r : mu * log(expm1(x))))
softplus_f(n, mu) = (x = n / mu; x > 40.0 ? n : mu * log1p(exp(x)))
softplus_J(r, mu) = (x = max(r, 0.0) / mu; x < 1e-12 ? 1e12 : 1.0 / (-expm1(-x)))
softplus(mu) = Transform("softplus(mu=$mu)", r -> softplus_g(r, mu),
                         n -> softplus_f(n, mu), r -> softplus_J(r, mu))

# ── the column the probe runs on ─────────────────────────────────────────────

"""
    background(outdir, z) -> (rho_d, Tk)

Dry density and temperature of the run's own reference column, interpolated to
the probe's mish. Only the warm-rain rate kernels read these, and they read them
weakly, but taking them from the run keeps the drain rates the model's own.
"""
function background(outdir, z)
    snaps = sort([f for f in readdir(outdir; join = true) if endswith(f, "_physical.csv")])
    df0 = CSV.read(first(snaps), DataFrame)
    ncols = length(unique(df0.r)); kDim = nrow(df0) ÷ ncols
    vars = Scythe.MC_VARS
    sb = Dict(v => NeumannBC() for v in vars)
    gp = GridParameters(; geometry = "RiRk",
        iMin = 0.0, iMax = 150.0e3, num_cells_i = ncols ÷ 3,
        kMin = 0.0, kMax = KMAX, num_cells_k = kDim ÷ 3, l_q = Dict("default" => 2.0),
        BCL = merge(sb, Dict("u" => DirichletBC(), "w" => DirichletBC())),
        BCR = merge(sb, Dict("u" => DirichletBC(), "w" => DirichletBC())),
        BCB = merge(sb, Dict("w" => DirichletBC(), "rho_r" => NaturalBC())),
        BCT = merge(sb, Dict("w" => DirichletBC(), "rho_r" => NaturalBC())),
        vars = Dict(v => i for (i, v) in enumerate(vars)))
    patch = createGrid(gp)
    zref = Scythe.getGridpoints(patch)[1:kDim, 2]
    ref = Springsteel.exact_pressure_reference_state(joinpath(outdir, "o01_exact.ref"),
                                                     zref, Scythe.reference_column(patch, gp))
    rho_d = Springsteel.ref_rho_d(ref)[:, 1]
    rho_t = Springsteel.ref_rho_t(ref)[:, 1]
    rho_c = Springsteel.ref_rho_c(ref)[:, 1]
    p     = Springsteel.ref_pressure(ref)[:, 1]
    E_t   = Springsteel.ref_total_energy(ref)[:, 1]
    # The reference temperature: the model's own retrieval at rest (ke = 0).
    M = p .+ E_t .- (rho_t .* (Scythe.gravity .* zref))
    Tk = Scythe.retrieve_temperature.(M, rho_d, rho_t, rho_c)
    # The probe's mish is the same 300 points when kDim == 300; otherwise take the
    # nearest reference level, which is ample for a weakly-read background.
    idx = [argmin(abs.(zref .- zi)) for zi in z]
    return rho_d[idx], Tk[idx]
end

# ── the spline column, and one refit ─────────────────────────────────────────

"""
    refit!(sp, u) -> (u_fitted, du_dz)

One fit->reconstruct round trip on the model's own basis, exactly as the
per-timestep path does: values -> b -> a -> values and first derivative.
"""
function refit!(sp, u; bounded = false)
    sp.uMish .= u
    SBtransform!(sp)
    bounded ? SAtransform_bounded!(sp) : SAtransform!(sp)
    SItransform!(sp)
    return copy(sp.uMish), SIxtransform(sp)
end

"""Mass on the mish in the model's own Gauss metric."""
function mass_operator()
    _, qw = CubicBSpline._quadrature_rule(3, :gauss)
    W = repeat(qw .* (KMAX / NC), outer = NC)
    return u -> sum(W .* u)
end

# ── the prescribed transport ─────────────────────────────────────────────────

"""
A 10 m/s updraft centred at 5 km, vanishing at both walls so it is compatible
with w's Dirichlet BCs. The divergence is taken from the SAME spline as
everything else, so the discretization the probe sees is the model's.
"""
function updraft(sp, z; w0 = 10.0, zc = 5000.0, half = 3000.0)
    w = @. w0 * exp(-((z - zc) / half)^2) * sin(pi * z / KMAX)
    wf, wz = refit!(sp, w)
    return wf, wz
end

# ── one integration ──────────────────────────────────────────────────────────

"""
    integrate(tf, arm, z, sp, w, wz, rho_d, Tk, mass; mu, nstep) -> NamedTuple

Advance one column for `nstep` AB3 steps under `arm`, in the control variable of
`tf`, refitting every step. Returns the diagnostics the falsification criteria
need. `tf.name == "identity"` is the rho-space baseline.
"""
function integrate(tf::Transform, arm::Symbol, z, sp, w, wz, rho_d, Tk, mass;
                   nstep = NSTEP, src_amp = 0.0, src_mask = falses(length(z)))
    npt = length(z)
    rho0 = zeros(npt)
    if arm !== :nucleation
        # A spike the column cannot resolve: sigma = 1.5 cells. This is the input
        # branch (b) of water_projection_diagnostic.jl -- strictly non-negative,
        # with a kink at its foot, so the ringing it provokes is an upper bound.
        dz = KMAX / NC
        @. rho0 = 3.0e-3 * exp(-((z - 5000.0) / (1.5 * dz))^2)
    end

    # The state is the CONTROL variable at the mish; it is what AB3 advances and
    # what the refit fits. rho is only ever recovered from it.
    u = tf.g.(rho0)
    u, _ = refit!(sp, u; bounded = tf.bounded)
    dot_n = zeros(npt); dot_nm1 = zeros(npt); dot_nm2 = zeros(npt)

    rho_r = zeros(npt)                     # rain accumulates from the drain arm
    max_step_ratio = 0.0
    lag = NaN
    mass0 = mass(tf.f.(u))
    peak = -Inf; worst_min = Inf
    # How far the CONTROL variable ratchets below zero. This is the transform's own
    # possible failure mode -- the reservoir relocating into n, where it is invisible
    # in rho but sets the nucleation lag |n|/S -- and it is the only quantity that
    # distinguishes H-smooth from Ooyama's quasi-inverse, since on the n < 0 branch
    # the two recovered densities differ by at most mu.
    worst_n = Inf
    finished = true
    # F4 needs to know whether the mass injection SATURATES or keeps growing, so
    # the trajectory is sampled rather than only differenced end to end.
    checkpoints = [nstep ÷ 4, nstep ÷ 2, 3 * nstep ÷ 4, nstep]
    mass_track = Float64[]

    for t in 1:nstep
        uf, uz = refit!(sp, u; bounded = tf.bounded)
        u .= uf
        rho = tf.f.(uf)
        J = tf.J.(rho)

        # Sources, all evaluated on the RECOVERED density, exactly as the kernel's
        # max(rho,0)-guarded rates are.
        S = zeros(npt)
        if arm === :drain
            for i in 1:npt
                a = Scythe.autoconversion_density(max(rho[i], 0.0), rho_d[i])
                c = Scythe.collection_density(max(rho[i], 0.0), rho_r[i], rho_d[i], Tk[i])
                S[i] = -(a + c)
                rho_r[i] += TS * (a + c)     # rain is a passive accumulator here
            end
        elseif arm === :nucleation
            @. S = ifelse(src_mask, src_amp, 0.0)
        end

        # The tendency. Advection from the control variable's own gradient;
        # divergence and sources through the Jacobian.
        rhs = @. (-w * uz) + (J * ((-rho * wz) + S))
        !all(isfinite, rhs) && (finished = false; break)

        dot_nm2 .= dot_nm1; dot_nm1 .= dot_n; dot_n .= rhs
        # `_ab3_increment` already carries ts (Euler at t=1, AB2 at t=2, AB3 after).
        u .+= Scythe._ab3_increment.(TS, t, dot_n, dot_nm1, dot_nm2)
        !all(isfinite, u) && (finished = false; break)

        rho_new = tf.f.(u)
        worst_n = min(worst_n, minimum(u))
        peak = max(peak, maximum(rho_new))
        worst_min = min(worst_min, minimum(rho_new))
        max_step_ratio = max(max_step_ratio, maximum(abs, rho_new .- rho))
        if isnan(lag) && arm === :nucleation && maximum(rho_new) > 1e-6
            lag = t * TS
        end
        t in checkpoints && push!(mass_track, mass(rho_new) - mass0)
    end

    ufin, _ = refit!(sp, u; bounded = tf.bounded)
    rho_fin = tf.f.(ufin)
    return (; name = tf.name, arm, finished,
            min_rho = worst_min, max_rho = peak,
            final_min = minimum(rho_fin), final_max = maximum(rho_fin),
            neg_mass = mass(max.(-rho_fin, 0.0)), pos_mass = mass(max.(rho_fin, 0.0)),
            mass_drift = mass(rho_fin) - mass0, max_step = max_step_ratio, lag,
            mass_track, shortfall = CubicBSpline.bound_shortfall(sp), worst_n)
end

# ── driver ───────────────────────────────────────────────────────────────────

function main()
    sp = Spline1D(spline_params())
    z = copy(sp.mishPoints)
    mass = mass_operator()
    rho_d, Tk = background(OUTDIR, z)
    w, wz = updraft(sp, z)
    @printf("probe column: %d mish points, dz = %.1f m, ts = %.2f s, %d steps (%.0f s)\n",
            length(z), KMAX / NC, TS, NSTEP, NSTEP * TS)
    @printf("updraft: max w = %.2f m/s, max |dw/dz| = %.3e 1/s\n",
            maximum(w), maximum(abs, wz))

    # A nucleation source sized so that it would build 1 g/m^3 in 100 s if
    # nothing removed it -- vigorous, and squarely in the regime that matters.
    dzc = KMAX / NC
    src_mask = @. abs(z - 5000.0) < 1.5 * dzc
    src_amp = 1.0e-3 / 100.0

    MU = 1e-7
    cands = [identity_tf(), hyp(MU), hyp_ooyama(MU), quadratic(MU), softplus(MU),
             limiter_tf()]

    for arm in (:transport, :drain, :nucleation)
        println("\n" * "="^92)
        println("ARM $(uppercase(String(arm)))")
        println("="^92)
        @printf("%-22s %6s %12s %12s %12s %12s %11s %8s\n",
                "candidate", "fin", "min_rho", "max_rho", "neg_mass", "pos_mass",
                "mass_drift", "lag(s)")
        base = nothing
        for tf in cands
            col = tf.bounded ? Spline1D(spline_params(); lower = zeros(NC + 3)) :
                               Spline1D(spline_params())
            r = integrate(tf, arm, z, col, w, wz, rho_d, Tk, mass;
                          src_amp = src_amp, src_mask = src_mask)
            tf.name == "identity" && (base = r)
            flag = ""
            if base !== nothing && tf.name != "identity"
                r.max_step > 2 * base.max_step && (flag *= " F1")
                (!r.finished || r.max_rho > 3 * base.max_rho) && (flag *= " F2")
            end
            !r.finished && (flag *= " F2")
            arm === :nucleation && (isnan(r.lag) || r.lag > 10.0) && (flag *= " F3")
            @printf("%-22s %6s %+12.4e %12.4e %12.4e %12.4e %+11.3e %8s%s\n",
                    tf.name, r.finished ? "yes" : "NO", r.min_rho, r.max_rho,
                    r.neg_mass, r.pos_mass, r.mass_drift,
                    isnan(r.lag) ? "-" : @sprintf("%.2f", r.lag), flag)
            r.shortfall == 0.0 ||
                @printf("%-22s        bound_shortfall (mass the limiter CREATED): %.4e\n",
                        "", r.shortfall)
            # F4: does the mass injection saturate, or keep growing linearly?
            isempty(r.mass_track) ||
                @printf("%-22s        mass at 900/1800/2700/3600 s: %s   min(n) = %+.4e\n", "",
                        join((@sprintf("%+.3e", m) for m in r.mass_track), "  "), r.worst_n)
        end
    end
end

main()
