# =============================================================================
# Which POSITIVE-DEFINITE CONTROL VARIABLE should the condensate be fitted in?
#
# reference/HANDOFF_CONDENSATE_REPRESENTATION.md, Stage 1a. This script runs no
# model and changes nothing in src/. It measures, offline on saved snapshots,
# what one cubic-B-spline fit->reconstruct round trip does to the cloud density
# under each candidate change of variable.
#
# ── The problem being measured ───────────────────────────────────────────────
#
# rho_c is positive definite and spatially sharp; a spline column with too few
# nodes to resolve the spike undershoots on its flanks. The undershoot is not
# inert: rho_liq = rho_c + rho_r enters the closed-form temperature retrieval
# directly (dT/d rho_liq = L_v/D, src/moist_compressible.jl:235), so a negative
# lobe of magnitude d carries a temperature anomaly L_v*d/D -- reaching -54 K on
# the O01 benchmark. Flooring the STATE rectifies a two-signed oscillation and
# detonates the run in 23 min; bounding the spline coefficients (POSITIVITY=1)
# drives max_w 9.87 -> 46.5 because the limiter CREATES mass (bound_shortfall
# 7.9e6) wherever a whole column arrives with negative total mass.
#
# The class-level alternative is to fit a TRANSFORMED variable n = g(rho) whose
# range is all of R, and recover rho = f(n) >= -mu pointwise. Nothing is ever
# clamped, so nothing rectifies; the ringing lands in n, where it cannot make
# the recovered density meaningfully negative.
#
# ── The candidates ───────────────────────────────────────────────────────────
#
# Each is a triple (g, f, J) with f = g^-1 (or a quasi-inverse), J = dg/drho:
#
#   (I)  identity        -- the baseline; this is what the model does today.
#   (H)  Ooyama 2001 biased hyperbolic, reference/ooyama_jas2001.pdf Eq. 4.19-4.20:
#            n = bhyp(rho) = 0.5[(rho+mu) - mu^2/(rho+mu)]
#            rho = ahyp(n) = sqrt(n^2 + mu^2) + n - mu
#        with J = dn/drho = 0.5[1 + mu^2/(rho+mu)^2] in [0.5, 1] for rho >= 0
#        (Ooyama's Eq. 4.23). Two variants are scored:
#          H-smooth : the strict inverse everywhere; range (-mu, inf), C-infinity.
#          H-Ooyama : the published quasi-inverse, rho = 0 for n <= 0; C^0 only.
#        Ooyama's own note: the strict inverse reaches at worst rho = -mu, so the
#        quasi-inverse's adjustment never exceeds mu. mu = 1e-7 is his typical
#        value; he reports no discernible difference between 1e-7 and 1e-8 and
#        "very slight differences" at 1e-6.
#   (S)  softplus        -- n = mu*ln(exp(rho/mu) - 1), rho = mu*ln(1 + exp(n/mu)).
#        Strictly positive range, and therefore J = 1/(1 - exp(-rho/mu)) is
#        UNBOUNDED at the cloud edge. Carried so that is measured, not asserted.
#   (Q)  quadratic touchdown -- the monotone member of the square-root class
#        (naive rho = n^2 is not invertible). C^1, identity above the knee:
#            f(n) = 0 | (n+mu)^2/(4mu) | n     on n <= -mu | -mu<=n<=mu | n >= mu
#            g(rho) = 2*sqrt(mu*rho) - mu | rho
#        J = sqrt(mu/rho). Quadratic suppression of BOTH signs of sub-knee
#        ringing, and the only family whose g is total on [0, inf).
#   (L)  log             -- the known-failed control (prior art, Chebyshev era).
#
# ── Two honest caveats about what this can and cannot say ────────────────────
#
# 1. The input field is rho+ = max(rho_c, 0). It has a kink at the zero contour
#    and is therefore if anything HARDER to represent than the true pre-fit
#    field, so every ringing number here is an upper bound. (Same argument as
#    benchmarks/water_projection_diagnostic.jl's branch (b).)
# 2. Iterating a round trip offline does NOT reproduce the model's ratchet. The
#    model's accumulation comes from the physics re-adding the spike between
#    refits, which only the dynamical probe (Stage 1b,
#    benchmarks/condensate_transform_probe.jl) can exercise. What the iteration
#    here DOES measure is the round trip's own idempotency defect -- and that is
#    not zero, because of a fact measured while writing this script and worth
#    recording:
#
#      MEASURED (2026-07-29, NOPRECIP t = 1200 s, this basis):
#        l_q = 0.0 : |P(saved) - saved|/peak = 1.2e-15, and 100 iterations drift
#                    by 7.4e-14. The refit is an EXACT projection, and the saved
#                    state is exactly a spline reconstruction.
#        l_q = 2.0 : |P(saved) - saved|/peak = 1.0e-3, and 100 iterations drift
#                    by 3.5e-2 of peak with the per-iteration step still 2.6e-4
#                    at iteration 100 -- decaying the peak, not converging.
#
#    The shipped l_q = 2.0 spline low-pass therefore makes the per-step refit a
#    genuine SMOOTHER rather than a projection, applied 12 000 times over an O01
#    quick run. That is a separate lever from the transform, and it is why the
#    l_q sweep below is a first-class arm rather than a footnote.
#
# ── WHAT IT MEASURED (2026-07-29; full tables in
#    benchmarks/output/condensate_transform_sweep.csv and
#    benchmarks/output/condensate_state_census.csv) ───────────────────────────
#
# THE STATE. The accumulated reservoir is not a rounding artifact, and PRECIPITATION
# is what lets it take over. Gauss-weighted, as a fraction of the positive cloud mass
# actually present:
#
#   t (s)          1200   1800   2400   3000   3600
#   NOPRECIP      0.373  0.388  0.302  0.187  0.151     dT_neg 2.1 -> 12.6 K
#   G2 (precip)   0.555  1.167  2.751  2.958  2.683     dT_neg 1.7 ->  4.3 K
#
# In NOPRECIP the cloud keeps growing (pos_mass 7.9e4 -> 1.0e6) so the reservoir stays
# a bounded fraction. In G2 autoconversion and collection drain the POSITIVE cloud into
# rain (pos_mass stays 3-7e4) while the negative part -- which every max(rho,0)-guarded
# rate function cannot see, and which therefore has NO SINK -- keeps accumulating
# (2.8e4 -> 1.8e5). By t = 2400 s the shipped run's condensate field is 2.75x more
# spurious negative mass than real cloud. That is the mechanism, measured.
#
# ONE REFIT vs THE STATE. One round trip of a strictly non-negative field injects
# min_rho = -5.8e-5 and neg_mass = 1.4e3 (G2, t = 3600); the state carries -2.1e-3 and
# 1.8e5. So the accumulation is ~36x in amplitude and ~130x in mass over one refit --
# the ratchet, quantified.
#
# THE ASYMMETRY, IN ONE NUMBER PAIR. Per refit the undershoot is -5.8e-5 and the
# spurious POSITIVE cloud where the input was exactly zero is +1.4e-4 -- 2.4x LARGER.
# The positive error is the bigger one and it does not accumulate, because it looks
# like cloud and the physics consumes it. Only the negative side is immortal.
#
# THE RANKING (mu = 1e-7, l_q = 2.0, NOPRECIP t = 3600; every candidate scored on the
# identical round trip):
#
#   candidate     min_rho     dT_neg    halo_amp  edge_grad_err   J range      src_amp
#   identity    -5.77e-05   7.19e-01   1.4393e-4     0 (ref)      [1, 1]       1.4e-4
#   H-smooth    -9.98e-08   2.12e-03   1.4388e-4    3.94e-10      [0.5, 1]     1.4e-4
#   H-Ooyama     0.00e+00   0.00e+00   1.4388e-4    3.94e-10      [0.5, 1]     1.4e-4
#   softplus     ~1e-270    0.00e+00   1.4239e-4    1.79e-08      [1, 1e12]    1.4e+08
#   Q-quad       0.00e+00   0.00e+00   1.4388e-4    4.64e-10      [1, 1.5e6]   2.2e+02
#   log          5.93e-06   0.00e+00   5.9254e-6    5.69e-06      [210, 1e7]   1.4e+03
#
#   * H (both variants) is the ONLY family with a bounded Jacobian: [0.5, 1] exactly,
#     as Ooyama's Eq. 4.23 predicts, so its source amplification is 1.4e-4 -- i.e. none.
#     softplus amplifies by 1e12 and Q-quad by up to 1e6. Those two are disqualified on
#     dynamical grounds, not representational ones: they are fine in this table and
#     would be stiff in the model.
#   * H's edge-gradient error is 3.9e-10 against a 4.5e-3 peak (relative 1e-7) -- the
#     best of the family, and 45x better than softplus, 14000x better than log.
#   * THE HALO IS PRE-EXISTING AND UNCHANGED: identity 1.4393e-4 vs H 1.4388e-4, and
#     the point counts agree to 1.5%. The transform neither creates nor removes
#     spurious positive cloud. The handoff's halo worry was a Chebyshev GLOBAL-ringing
#     artifact and does not transfer to this basis. There is no halo trade to make.
#   * mass_err for a transform is NOT a leak: at every snapshot it equals the identity
#     row's neg_mass to ~2% (e.g. 1.326e3 vs 1.372e3). It is exactly the reservoir being
#     handed back. log's is 11x too big, which is the known failure.
#   * Idempotency drift under 20 further round trips is 1.5666e-4 for identity and
#     1.5667e-4 for H -- the transform adds nothing; the drift is the l_q filter's.
#     log drifts 20x more.
#
# EXPERIMENT 2 DOES NOT DISCRIMINATE -- record this before anyone re-litigates it.
# Scored on the identical round trip of the identical non-negative input (G2, t = 1200),
# what each enforcement of rho_c >= 0 actually MOVES:
#
#   scheme      min_rho     mass_err   shortfall   max|drho|   dT_per_refit
#   H-smooth  -9.98e-08     +4.558e+2      --      4.451e-05      0.1341 K
#   POS=k      0.00e+00     +2.743e+2   4.07e-01   4.461e-05      0.1344 K
#   POS=i     -2.43e-05     -7.3e-12    0.00e+00   1.196e-04      0.2988 K
#   POS=ik     1.05e-26     -7.3e-12    0.00e+00   1.252e-04      0.3007 K
#
# H-smooth and the one-leg limiter make the SAME modification to three digits. The
# two-leg limiter moves 2.8x more (the i leg redistributes horizontally) but conserves
# mass EXACTLY (-7.3e-12) and creates none (shortfall 0), where the transform adds 456
# and POS=k adds 274. On a clean non-negative input the limiter is therefore at least as
# good as the transform on every axis measurable here, and better on mass.
#
# (The POS=k shortfall and its mass creation are the same number: 4.07e-01 in the k leg's
# 1-D coefficient metric times the ~667 m horizontal quadrature weight is 2.7e+2. That
# consistency is a useful check that `bound_shortfall` means what it says.)
#
# The 7.9e6 shortfall recorded for the real POSITIVITY=1 run therefore does NOT come from
# the limiter's per-refit action on a good field. It comes from columns that arrive
# already carrying negative total mass -- a property of the trajectory, not of the fit.
# Nothing offline settles which scheme is better; see the VERDICT block in
# condensate_transform_probe.jl.
#
# THE M2 GATE (does the vapor partition survive handing the reservoir back?). Replacing
# the state's rho_c by the recovery moves min_rho_v from -8.25e-5 to -1.17e-4 (G2,
# t = 3600), i.e. ~40% worse, and the transform arms differ from the identity arm by
# ~0.1%. For scale, the measured POSITIVITY=1 run reached -1.35e-3 -- an order of
# magnitude further. So the CONSTRAINT costs ~40% instantaneously while the LIMITER's
# trajectory cost 10x that: evidence that the detonation lives in the manner of
# enforcement, not in the constraint. It is not proof -- this is a one-step
# substitution on a trajectory grown WITH the reservoir; only a run settles it.
#
# mu COSTS (M8, worst point in the domain, through the model's own retrieval):
#   1e-8 -> 7.2e-4 K   1e-7 -> 7.2e-3 K   1e-6 -> 7.2e-2 K   1e-5 -> 0.72 K
# The worst point is the model TOP (25 km), where rho_d ~ 0.04 and the retrieval
# denominator is ~30x smaller than at the surface; near the ground mu = 1e-7 costs
# ~2.5e-4 K. Ooyama's mu is a MIXING RATIO, so his bias shrinks with rho_d and his
# cost is flat in z; ours is a constant density and its cost is worst where there is
# no cloud. mu = 1e-7 kg/m^3 is admissible as it stands; mu(z) = mu_q * rho_dbar(z) is
# available if a configuration ever cares about the stratosphere.
#
# l_q IS NOT A CO-FIX. Sweeping the low-pass (identity arm, NOPRECIP t = 3600):
#   l_q = 0 -> min_rho -6.22e-5, halo 1.298e-4
#   l_q = 2 -> min_rho -5.77e-5, halo 1.439e-4      (shipped)
#   l_q = 4 -> min_rho -9.21e-5, halo 2.769e-4
# There is no free win: more filtering makes both worse. H holds min_rho = -mu exactly
# at every l_q.
#
# Usage:
#   julia --project=. benchmarks/condensate_transform_diagnostic.jl [LABEL=dir ...]
# Default snapshot sets: the NOPRECIP stage0 run (2 km, cloud only) and the
# shipped quick run. Add `FULL=benchmarks/output/o01_rainfall_full_mc_rirk` for
# the 500 m resolution-independence arm.
# =============================================================================

using Scythe, Springsteel, CSV, DataFrames, Printf, Statistics
using Springsteel.CubicBSpline

const OUT = joinpath(@__DIR__, "output")

# Snapshot times to score. The cloud is well developed by 1200 s on the quick
# grid and the negative reservoir has taken over by 2400 s (see the header table
# in the campaign plan), so the set spans both regimes.
const TIMES = [1200.0, 1800.0, 2400.0, 3000.0, 3600.0]

# Ooyama's typical value is 1e-7 (as a MIXING RATIO, kg/kg). Our prognostic is a
# partial DENSITY, kg/m^3, and rho_d spans ~1.2 -> 0.05 over the column, so the
# two coincide near the surface and differ by ~20x at the model top. The bias is
# carried here as a density; M8 reports what each value costs in temperature
# through the model's own retrieval, which is the test that actually matters.
const MU_GRID = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]

# The spline low-pass. It now filters the CONTROL variable rather than the
# density, so it has to be swept: 2.0 is the shipped value (o01_rainfall.jl:215).
const LQ_GRID = [0.0, 2.0, 4.0]

# "Cloud" for the edge-band statistics: fraction of the column peak.
const EDGE_FRAC = 0.1
# A recovered density above this counts as spurious condensate where there was none.
const HALO_THRESH = 1e-6

# ── the candidate transforms ─────────────────────────────────────────────────
# Each is (name, g(rho), f(n), J(rho) = dg/drho). J is evaluated at the RECOVERED
# DENSITY, clamped at zero for the argument only -- Ooyama Eq. 4.21-4.23. That is
# what keeps the Jacobian bounded for (H); for (S) and (Q) it is unbounded anyway
# and M6 measures by how much.

struct Transform
    name::String
    g::Function       # rho -> n
    f::Function       # n -> rho
    J::Function       # rho -> dn/drho
end

identity_tf() = Transform("identity", r -> r, n -> n, _ -> 1.0)

# Ooyama 2001 Eq. 4.19 / 4.20 / 4.23.
bhyp(rho, mu) = 0.5 * ((rho + mu) - (mu * mu) / (rho + mu))
ahyp_smooth(n, mu) = sqrt(n * n + mu * mu) + n - mu
ahyp_ooyama(n, mu) = n <= 0.0 ? 0.0 : ahyp_smooth(n, mu)
dbhyp(rho, mu) = 0.5 * (1.0 + (mu * mu) / ((rho + mu) * (rho + mu)))

hyp_smooth(mu) = Transform("H-smooth(mu=$(mu))", r -> bhyp(r, mu),
                           n -> ahyp_smooth(n, mu), r -> dbhyp(max(r, 0.0), mu))
hyp_ooyama(mu) = Transform("H-Ooyama(mu=$(mu))", r -> bhyp(r, mu),
                           n -> ahyp_ooyama(n, mu), r -> dbhyp(max(r, 0.0), mu))

# Softplus. g(0) = -Inf, so a cloud-free point has no finite representative; the
# forward map is guarded at a value that maps to the smallest representable n.
function softplus_g(rho, mu)
    x = rho / mu
    x < 1e-12 && return mu * log(1e-12)            # guarded: g(0) is -Inf
    return x > 40.0 ? rho : mu * log(expm1(x))     # expm1 for small x
end
softplus_f(n, mu) = (x = n / mu; x > 40.0 ? n : mu * log1p(exp(x)))
softplus_J(rho, mu) = (x = max(rho, 0.0) / mu; x < 1e-12 ? 1.0 / 1e-12 : 1.0 / (-expm1(-x)))
softplus(mu) = Transform("softplus(mu=$(mu))", r -> softplus_g(r, mu),
                         n -> softplus_f(n, mu), r -> softplus_J(r, mu))

# Quadratic touchdown: C^1, identity above the knee, quadratic below it.
quad_f(n, mu) = n <= -mu ? 0.0 : (n >= mu ? n : (n + mu)^2 / (4mu))
quad_g(rho, mu) = rho >= mu ? rho : 2.0 * sqrt(mu * max(rho, 0.0)) - mu
quad_J(rho, mu) = (r = max(rho, 0.0); r >= mu ? 1.0 : (r <= 0.0 ? Inf : sqrt(mu / r)))
quadratic(mu) = Transform("Q-quad(mu=$(mu))", r -> quad_g(r, mu),
                          n -> quad_f(n, mu), r -> quad_J(r, mu))

# Log, the known-failed control.
log_tf(mu) = Transform("log(mu=$(mu))", r -> log(max(r, mu) / mu),
                       n -> mu * exp(min(n, 700.0)), r -> 1.0 / max(r, mu))

candidates(mu) = [hyp_smooth(mu), hyp_ooyama(mu), softplus(mu), quadratic(mu), log_tf(mu)]

# ── grid + reference rebuild (mirrors benchmarks/vapor_blend_diagnostic.jl:185) ──

"""
    build_patch(outdir, df0; l_q) -> (patch, grid_params, R)

The run's own grid and exact reference state, inferred from a snapshot's shape.
`l_q` overrides the spline low-pass so it can be swept; everything else must
match `benchmarks/o01_rainfall.jl` exactly or the round trip is not idempotent
on the saved field (which the `P(saved)` check below verifies).
"""
function build_patch(outdir, df0; l_q = 2.0, positivity = Dict{String,Dict{Symbol,Float64}}())
    ncols = length(unique(df0.r))
    kDim = nrow(df0) ÷ ncols
    vars = Scythe.MC_VARS
    scalar_bc = Dict(v => NeumannBC() for v in vars)
    side_bc   = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))
    topbot_bc = merge(scalar_bc, Dict("w" => DirichletBC(), "rho_r" => NaturalBC()))
    grid_params = GridParameters(;
        geometry = "RiRk",
        iMin = 0.0, iMax = 150.0e3, num_cells_i = ncols ÷ 3,
        kMin = 0.0, kMax = 25.0e3, num_cells_k = kDim ÷ 3,
        l_q = Dict("default" => l_q),
        positivity = positivity,
        BCL = side_bc, BCR = side_bc, BCB = topbot_bc, BCT = topbot_bc,
        vars = Dict(v => i for (i, v) in enumerate(vars)),
    )
    patch = createGrid(grid_params)
    z = Scythe.getGridpoints(patch)[1:kDim, 2]
    column = Scythe.reference_column(patch, grid_params)
    ref = Springsteel.exact_pressure_reference_state(joinpath(outdir, "o01_exact.ref"),
                                                     z, column)
    R = (; ncols, kDim,
         pbar     = Springsteel.ref_pressure(ref)[:, 1],
         rho_dbar = Springsteel.ref_rho_d(ref)[:, 1],
         rho_tbar = Springsteel.ref_rho_t(ref)[:, 1],
         rho_cbar = Springsteel.ref_rho_c(ref)[:, 1],
         E_tbar   = Springsteel.ref_total_energy(ref)[:, 1],
         Q_ssbar  = Springsteel.ref_qss(ref)[:, 1])
    return patch, grid_params, R
end

"""
    project(patch, field, slot) -> (value, dz)

One fit->reconstruct round trip on the model's own basis, returning the
reconstructed field and its VERTICAL derivative (physical slot 4 of the RiRk
5-slot layout `[f, di, dii, dk, dkk]`). Other slots are left untouched, so the
caller must not read them.
"""
function project(patch, field, slot)
    patch.physical .= 0.0
    patch.physical[:, slot, 1] .= field
    spectralTransform!(patch)
    gridTransform!(patch)
    return copy(patch.physical[:, slot, 1]), copy(patch.physical[:, slot, 4])
end

"""Gauss-weight quadrature on the mish, both directions (cf. `gauss_cell_weights`)."""
function mass_operator(gp, ncols, kDim)
    _, qw = CubicBSpline._quadrature_rule(gp.mubar, gp.quadrature)
    nk = gp.num_cells_k
    ni = gp.num_cells_i
    Wv = repeat(qw .* ((gp.kMax - gp.kMin) / nk), outer = nk)
    Wh = repeat(qw .* ((gp.iMax - gp.iMin) / ni), outer = ni)
    return function (f)
        F = reshape(f, kDim, ncols)
        return sum(Wh[c] * sum(Wv .* @view(F[:, c])) for c in 1:ncols)
    end
end

# ── the scored quantities ────────────────────────────────────────────────────

"""
Everything one (snapshot, transform, mu, l_q) cell contributes to the table.
`ref_*` fields carry the identity arm's value so each row is self-contained.
"""
struct Row
    label::String; t::Float64; tf::String; lq::Float64
    min_rho::Float64          # M1: worst recovered density after ONE refit
    max_rho::Float64
    mass::Float64             # M1: Gauss-weighted domain integral of the recovery
    mass_in::Float64          #     ... of the non-negative input
    neg_mass::Float64         # M1: integral of max(-rho,0) after ONE refit
    dT_neg::Float64           # M1: |L_v * min(rho,0) / D|, the anomaly the lobe carries
    min_rho_v::Float64        # M2: vapor after substituting the recovery
    n_rho_v_neg::Int          # M2
    halo_amp::Float64         # M3: max recovery where the input was exactly zero
    halo_n::Int               # M3
    edge_grad_err::Float64    # M4: |d/dz recovery - d/dz direct fit| on the edge band
    idem_drift::Float64       # M5: max |x - (f.P.g)^20(x)|, the smoother+transform defect
    drho_vs_free::Float64     # E2: max |recovery - unbounded fit|, the modification made
    dT_vs_free::Float64       # E2: the temperature that modification moves, per refit
    J_min::Float64            # M6
    J_max::Float64
    J_p99::Float64
    src_amp::Float64          # M6: ts * S * J_p99, the implied source amplification
end

"""
Per-snapshot facts that do not depend on the transform: what the ACCUMULATED
state carries, against which one refit's injection is measured. The ratio of
`neg_mass` here to `neg_mass` in a `Row` is the ratchet factor.
"""
struct StateRow
    label::String; t::Float64
    min_rho::Float64; max_rho::Float64
    mass::Float64; pos_mass::Float64; neg_mass::Float64
    dT_neg::Float64
    min_rho_v::Float64; n_rho_v_neg::Int
end

"""
    thermo(R, df) -> (M, rho_d, rho_t, rho_r, rho_c)

The retrieval's arguments on FULL fields, so a caller can substitute any
condensate and re-read the temperature and the residual vapor exactly as the
kernel does. Everything but `rho_r` is a perturbation from the exact reference.
"""
function thermo(R, df)
    tile(v) = repeat(v, R.ncols)
    p     = df.p     .+ tile(R.pbar)
    rho_d = df.rho_d .+ tile(R.rho_dbar)
    rho_t = df.rho_t .+ tile(R.rho_tbar)
    E_t   = df.E_t   .+ tile(R.E_tbar)
    rho_c = df.rho_c .+ tile(R.rho_cbar)
    ke = 0.5 .* ((df.u .^ 2) .+ (df.w .^ 2))
    M  = p .+ E_t .- (rho_t .* (ke .+ (Scythe.gravity .* df.z)))
    return M, rho_d, rho_t, Float64.(df.rho_r), rho_c
end

"""
    dT_from_negative(M, rho_d, rho_t, liq) -> Kelvin

The temperature anomaly the NEGATIVE part of the liquid density carries, read
through the model's own retrieval: T(liq) - T(max(liq,0)). This is the -54 K
class number the campaign exists to remove (TeX, "Negative condensate and
vertical resolution").
"""
function dT_from_negative(M, rho_d, rho_t, liq)
    T_raw = Scythe.retrieve_temperature.(M, rho_d, rho_t, liq)
    T_pos = Scythe.retrieve_temperature.(M, rho_d, rho_t, max.(liq, 0.0))
    return maximum(abs, T_raw .- T_pos)
end

"""
    state_row(R, df, label, t, mass) -> StateRow

What the ACCUMULATED saved state carries, with no round trip at all.
"""
function state_row(R, df, label, t, mass)
    M, rho_d, rho_t, rho_r, rho_c = thermo(R, df)
    liq = rho_c .+ rho_r
    rho_v = rho_t .- rho_d .- liq
    return StateRow(label, t, minimum(rho_c), maximum(rho_c),
                    mass(rho_c), mass(max.(rho_c, 0.0)), mass(max.(-rho_c, 0.0)),
                    dT_from_negative(M, rho_d, rho_t, liq),
                    minimum(rho_v), count(<(0.0), rho_v))
end

"""
    score(patch, gp, R, df, tf, mu, lq, label, t) -> Row

Round-trip `rho+ = max(rho_c + rho_cbar, 0)` through `tf` and measure. The
identity transform reproduces `water_projection_diagnostic.jl`'s branch (b)
exactly, which is the cross-check that this script's basis matches the run's.
"""
function score(patch, gp, R, df, tf::Transform, mu, lq, label, t, mass; n_idem = 0)
    slot = gp.vars["rho_c"]
    M, rho_d, rho_t, rho_r, rho_c = thermo(R, df)
    rho_p = max.(rho_c, 0.0)                     # the non-negative surrogate input

    # Forward map, one round trip, inverse map. The recovered vertical gradient is
    # f'(n)*dn/dz = (dn/dz)/J(rho), i.e. exactly what the model would form -- one
    # fit, one field, no differencing.
    n_in = tf.g.(rho_p)
    n_out, n_dz = project(patch, n_in, slot)
    rho_out = tf.f.(n_out)
    grad_out = n_dz ./ tf.J.(rho_out)

    # The identity arm's direct fit is the gradient baseline for M4.
    fit_out, fit_dz = project(patch, rho_p, slot)

    # ── M1: the reservoir this ONE refit injects, and what it costs in temperature ──
    neg_mass = mass(max.(-rho_out, 0.0))
    liq_out = rho_out .+ rho_r
    dT_neg = dT_from_negative(M, rho_d, rho_t, liq_out)

    # ── E2: the modification this enforcement makes, against the SAME unbounded
    # fit the limiter arms are compared to. This is the apples-to-apples
    # rectification measure: how much condensate the scheme moves per refit, and
    # how much temperature that carries through dT/d rho_liq = L_v/D.
    drho_vs_free = maximum(abs, rho_out .- fit_out)
    T_free = Scythe.retrieve_temperature.(M, rho_d, rho_t, fit_out .+ rho_r)
    T_out  = Scythe.retrieve_temperature.(M, rho_d, rho_t, liq_out)
    dT_vs_free = maximum(abs, T_out .- T_free)

    # ── M2: what the vapor residual has to absorb ──
    rho_v_out = rho_t .- rho_d .- liq_out
    min_rho_v = minimum(rho_v_out)
    n_rho_v_neg = count(<(0.0), rho_v_out)

    # ── M3: spurious condensate where the input had none ──
    zero_in = rho_p .== 0.0
    halo_amp = any(zero_in) ? maximum(rho_out[zero_in]) : 0.0
    halo_n   = count(x -> x > HALO_THRESH, rho_out[zero_in])

    # ── M4: cloud-edge gradient fidelity, against the DIRECT fit ──
    pk = maximum(rho_p)
    edge = (rho_p .> 0.0) .& (rho_p .< EDGE_FRAC * pk)
    edge_grad_err = any(edge) ? maximum(abs, grad_out[edge] .- fit_dz[edge]) : 0.0

    # ── M5: idempotency defect of f.P.g (see caveat 2 in the header) ──
    # Costly (one round trip per iteration), so it is enabled only for the
    # headline configuration; `n_idem = 0` reports NaN rather than a wrong zero.
    idem_drift = NaN
    if n_idem > 0
        x = copy(rho_out)
        for _ in 1:n_idem
            y, _ = project(patch, tf.g.(max.(x, 0.0)), slot)
            x = tf.f.(y)
        end
        idem_drift = maximum(abs, x .- rho_out)
    end

    # ── M6: Jacobian census and the implied source amplification ──
    Jv = tf.J.(rho_out)
    Jf = filter(isfinite, Jv)
    J_min = isempty(Jf) ? NaN : minimum(Jf)
    J_max = isempty(Jf) ? NaN : maximum(Jf)
    J_p99 = isempty(Jf) ? NaN : quantile(Jf, 0.99)
    # A representative source magnitude from the snapshot itself: the largest
    # condensation rate the cloud is carrying, sized as rho_c/tau_c.
    S = maximum(rho_p) / 10.0
    src_amp = isempty(Jf) ? NaN : 0.3 * S * J_p99

    return Row(label, t, tf.name, lq,
               minimum(rho_out), maximum(rho_out), mass(rho_out), mass(rho_p),
               neg_mass, dT_neg, min_rho_v, n_rho_v_neg,
               halo_amp, halo_n, edge_grad_err, idem_drift, drho_vs_free, dT_vs_free,
               J_min, J_max, J_p99, src_amp)
end

# ── The LIMITER as a rival way of enforcing rho >= 0 (handoff Experiment 2) ──
#
# The transform and the spline positivity limiter enforce the SAME constraint by
# different means, so the honest comparison scores them on the identical round
# trip. That is what decides the campaign's GO/NO-GO question: does the measured
# POSITIVITY=1 detonation (max_w 9.87 -> 46.5) live in the MANNER of enforcement
# (a per-step, one-signed clip-and-shrink that CREATES mass wherever a column
# arrives with negative total mass) or in the CONSTRAINT itself (the residual
# vapor cannot absorb the relocation)? If the manner, the transform fixes it; if
# the constraint, the transform reproduces it and the real fix is prognostic
# rho_v.

const LIMITER_ARMS = ["POS=k"  => Dict("rho_c" => Dict(:k => 0.0)),
                      "POS=i"  => Dict("rho_c" => Dict(:i => 0.0)),
                      "POS=ik" => Dict("rho_c" => Dict(:i => 0.0, :k => 0.0))]

"""
    limiter_shortfall(patch, gp) -> Float64

Mass the limiter CREATED on this round trip -- the accumulated `D - H` of
Springsteel's `_bound_free_space!` infeasible branch, summed over both legs. A
fresh patch is used per measurement because the accumulator never resets.
"""
function limiter_shortfall(patch, gp)
    v = gp.vars["rho_c"]
    total = CubicBSpline.bound_shortfall(patch.kbasis.data[v])
    for z in 1:gp.b_kDim
        total += CubicBSpline.bound_shortfall(patch.ibasis.data[z, v])
    end
    return total
end

# ── M8: what a given mu costs in temperature, through the model's retrieval ───

"""
    mu_cost(R, df, mu) -> Kelvin

The temperature difference between `mu` of water being vapor and being fully
condensed, evaluated with the model's own retrieval denominator. This is
Ooyama's negligibility test (he reports d(theta) < 3e-4 K at mu = 1e-7 kg/kg)
repeated in this model's units, where the prognostic is a partial density.
"""
function mu_cost(R, df, mu)
    tile(v) = repeat(v, R.ncols)
    p     = df.p     .+ tile(R.pbar)
    rho_d = df.rho_d .+ tile(R.rho_dbar)
    rho_t = df.rho_t .+ tile(R.rho_tbar)
    E_t   = df.E_t   .+ tile(R.E_tbar)
    ke = 0.5 .* ((df.u .^ 2) .+ (df.w .^ 2))
    M  = p .+ E_t .- (rho_t .* (ke .+ (Scythe.gravity .* df.z)))
    liq = max.(df.rho_c .+ tile(R.rho_cbar), 0.0) .+ df.rho_r
    T0 = Scythe.retrieve_temperature.(M, rho_d, rho_t, liq)
    T1 = Scythe.retrieve_temperature.(M, rho_d, rho_t, liq .+ mu)
    return maximum(abs, T1 .- T0)
end

# ── driver ───────────────────────────────────────────────────────────────────

function snapshot_path(dir, t)
    p = joinpath(dir, "$(t)_physical.csv")
    return isfile(p) ? p : nothing
end

function run_set(label, dir, rows, staterows)
    println("\n" * "="^78)
    println("SET $label  ($dir)")
    println("="^78)
    df0path = snapshot_path(dir, TIMES[1])
    df0path === nothing && (println("  no snapshot at t = $(TIMES[1]); skipping"); return)
    df0 = CSV.read(df0path, DataFrame)

    # Basis validation. The TRUE test is at l_q = 0, where the refit is an exact
    # projection and the saved state is already in the spline space: a mismatch
    # there means this script's basis is not the run's and nothing below counts.
    # At the shipped l_q = 2.0 the same round trip is a smoother, so its residual
    # is a MEASUREMENT (of the filter), not an error.
    patch0, gp0, R = build_patch(dir, df0; l_q = 0.0)
    saved = Float64.(df0.rho_c) .+ repeat(R.rho_cbar, R.ncols)
    P0, _ = project(patch0, saved, gp0.vars["rho_c"])
    rel0 = maximum(abs, P0 .- saved) / max(maximum(abs, saved), eps())
    patch2, gp2, _ = build_patch(dir, df0; l_q = 2.0)
    P2, _ = project(patch2, saved, gp2.vars["rho_c"])
    rel2 = maximum(abs, P2 .- saved) / max(maximum(abs, saved), eps())
    @printf("  basis: ncols=%d kDim=%d   |P(saved)-saved|/peak: l_q=0 %.3e %s ; l_q=2 %.3e (filter)\n",
            R.ncols, R.kDim, rel0, rel0 < 1e-10 ? "OK" : "*** BASIS MISMATCH ***", rel2)

    # M8: mu-only, and it sets the admissible range for everything else.
    println("\n  M8 -- meteorological cost of the bias (max |dT| over the domain, t=$(TIMES[1])):")
    for mu in MU_GRID
        @printf("      mu = %8.1e kg/m^3   ->  |dT| = %.3e K\n", mu, mu_cost(R, df0, mu))
    end

    # What the accumulated state carries, per snapshot, with no round trip.
    mass0 = mass_operator(gp0, R.ncols, R.kDim)
    for t in TIMES
        path = snapshot_path(dir, t)
        path === nothing && continue
        df = CSV.read(path, DataFrame)
        all(isfinite, df.rho_c) || continue
        push!(staterows, state_row(R, df, label, t, mass0))
    end

    # The limiter arms, at the shipped l_q, on the same round trip as everything
    # else. A FRESH patch per snapshot: `bound_shortfall` never resets.
    println("\n  Experiment 2 -- the LIMITER as a rival enforcement of rho_c >= 0")
    @printf("  %-8s %6s %12s %12s %12s %12s %12s %12s\n", "arm", "t", "min_rho",
            "neg_mass", "mass_err", "shortfall", "max|drho|", "dT_lim")
    for t in TIMES
        path = snapshot_path(dir, t)
        path === nothing && continue
        df = CSV.read(path, DataFrame)
        all(isfinite, df.rho_c) || continue
        M, rho_d, rho_t, rho_r, rho_c = thermo(R, df)
        rho_p = max.(rho_c, 0.0)
        pfree, gfree, _ = build_patch(dir, df0; l_q = 2.0)
        mfree = mass_operator(gfree, R.ncols, R.kDim)
        free, _ = project(pfree, rho_p, gfree.vars["rho_c"])
        for (name, pos) in LIMITER_ARMS
            pb, gb, _ = build_patch(dir, df0; l_q = 2.0, positivity = pos)
            bounded, _ = project(pb, rho_p, gb.vars["rho_c"])
            # The limiter's own pointwise modification, and the one-signed latent
            # heat it implies through dT/d rho_liq = L_v/D. `dT_lim` is the
            # temperature the limiter MOVES, not an error in the field: it is the
            # rectification channel, measured.
            d = bounded .- free
            T_b = Scythe.retrieve_temperature.(M, rho_d, rho_t, bounded .+ rho_r)
            T_f = Scythe.retrieve_temperature.(M, rho_d, rho_t, free .+ rho_r)
            @printf("  %-8s %6.0f %+12.4e %12.4e %+12.3e %12.4e %12.4e %12.4e\n",
                    name, t, minimum(bounded), mfree(max.(-bounded, 0.0)),
                    mfree(bounded) - mfree(rho_p), limiter_shortfall(pb, gb),
                    maximum(abs, d), maximum(abs, T_b .- T_f))
        end
    end

    for lq in LQ_GRID
        patch, gp, R = build_patch(dir, df0; l_q = lq)
        mass = mass_operator(gp, R.ncols, R.kDim)
        for t in TIMES
            path = snapshot_path(dir, t)
            path === nothing && continue
            df = CSV.read(path, DataFrame)
            all(isfinite, df.rho_c) || (println("  t=$t: non-finite rho_c, skipped"); continue)
            # The idempotency iteration is 20 extra round trips per cell, so it
            # runs only on the headline configuration (shipped l_q, final time).
            ni = (lq == 2.0 && t == TIMES[end]) ? 20 : 0
            push!(rows, score(patch, gp, R, df, identity_tf(), 0.0, lq, label, t, mass;
                              n_idem = ni))
            for mu in MU_GRID, tf in candidates(mu)
                push!(rows, score(patch, gp, R, df, tf, mu, lq, label, t, mass; n_idem = ni))
            end
        end
    end
end

function report(rows, staterows)
    println("\n" * "="^78)
    println("THE ACCUMULATED STATE (no round trip) -- this is what has to be fixed")
    println("neg/pos is the reservoir as a fraction of the cloud mass actually present;")
    println("dT_neg is the anomaly the negative liquid carries through the retrieval.")
    println("="^78)
    @printf("%-10s %6s %+11s %11s %11s %11s %7s %10s %11s %7s\n",
            "set", "t", "min_rho", "max_rho", "pos_mass", "neg_mass", "neg/pos",
            "dT_neg(K)", "min_rho_v", "n_v<0")
    for s in staterows
        @printf("%-10s %6.0f %+11.4e %11.4e %11.4e %11.4e %7.3f %10.3f %+11.4e %7d\n",
                s.label, s.t, s.min_rho, s.max_rho, s.pos_mass, s.neg_mass,
                s.neg_mass / max(s.pos_mass, eps()), s.dT_neg, s.min_rho_v, s.n_rho_v_neg)
    end

    println("\n" * "="^78)
    println("M1/M2 -- ONE refit: the reservoir injected, and what the vapor absorbs")
    println("(l_q = 2.0, the shipped value; identity row first in each block)")
    println("mass_err for a transform is NOT a leak -- it is the reservoir being handed")
    println("back, and it should equal the identity row's neg_mass.")
    println("="^78)
    @printf("%-22s %6s %11s %11s %11s %10s %11s %7s\n",
            "transform", "t", "min_rho", "neg_mass", "mass_err", "dT_neg",
            "min_rho_v", "n_v<0")
    for r in rows
        r.lq == 2.0 || continue
        @printf("%-22s %6.0f %+11.4e %11.4e %+11.3e %10.3e %+11.4e %7d\n",
                r.tf, r.t, r.min_rho, r.neg_mass, r.mass - r.mass_in,
                r.dT_neg, r.min_rho_v, r.n_rho_v_neg)
    end

    println("\n" * "="^78)
    println("EXPERIMENT 2 -- what each enforcement MOVES per refit, vs the same")
    println("unbounded fit. Compare these two columns directly with the POS=* table")
    println("printed per set above: same baseline, same round trip, same metric.")
    println("="^78)
    @printf("%-22s %6s %14s %14s\n", "scheme", "t", "max|drho|", "dT_per_refit(K)")
    for r in rows
        (r.lq == 2.0 && occursin("1.0e-7", r.tf)) || r.tf == "identity" || continue
        r.lq == 2.0 || continue
        @printf("%-22s %6.0f %14.4e %14.4e\n", r.tf, r.t, r.drho_vs_free, r.dT_vs_free)
    end

    println("\n" * "="^78)
    println("M3/M4/M5/M6 -- halo, edge gradient, idempotency defect, Jacobian")
    println("="^78)
    @printf("%-22s %6s %11s %6s %12s %11s %9s %9s %10s\n",
            "transform", "t", "halo_amp", "halo_n", "edge_grad_err", "idem_drift",
            "J_min", "J_p99", "src_amp")
    for r in rows
        r.lq == 2.0 || continue
        @printf("%-22s %6.0f %11.4e %6d %12.4e %11.4e %9.3g %9.3g %10.3g\n",
                r.tf, r.t, r.halo_amp, r.halo_n, r.edge_grad_err, r.idem_drift,
                r.J_min, r.J_p99, r.src_amp)
    end

    println("\n" * "="^78)
    println("l_q sensitivity (t = $(TIMES[end]), the low-pass now filters the CONTROL variable)")
    println("="^78)
    @printf("%-22s %5s %11s %11s %12s\n", "transform", "l_q", "min_rho", "halo_amp",
            "edge_grad_err")
    for r in rows
        r.t == TIMES[end] || continue
        @printf("%-22s %5.1f %+11.4e %11.4e %12.4e\n",
                r.tf, r.lq, r.min_rho, r.halo_amp, r.edge_grad_err)
    end
end

function main()
    specs = isempty(ARGS) ?
        ["NOPRECIP=" * joinpath(OUT, "o01_rainfall_quick_mc_rirk_noprecip_stage0"),
         "G2="       * joinpath(OUT, "o01_rainfall_quick_mc_rirk")] : ARGS
    rows = Row[]
    staterows = StateRow[]
    for spec in specs
        label, dir = split(spec, "="; limit = 2)
        isdir(dir) || (println("skipping $label: $dir does not exist"); continue)
        run_set(String(label), String(dir), rows, staterows)
    end
    report(rows, staterows)

    csv = joinpath(OUT, "condensate_transform_sweep.csv")
    CSV.write(csv, DataFrame(rows))
    CSV.write(joinpath(OUT, "condensate_state_census.csv"), DataFrame(staterows))
    println("\nmachine-readable dumps: $csv and condensate_state_census.csv")
end

main()
