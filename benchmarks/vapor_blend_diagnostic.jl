# =============================================================================
# Which BLEND of the two vapor retrievals is best conditioned, everywhere?
#
# The model retrieves the vapor as the density-budget residual
#
#     res_rho_t = rho_t - rho_d - rho_c - rho_r
#
# and reference/HANDOFF_VAPOR_RETRIEVAL.md measured that this route cancels
# catastrophically IN CLOUD (rho_c/rho_w = 1.0011 at the worst point), while the
# supersaturation residual
#
#     res_qss = Q_ss + rho_vs(T, p)          (Q_ss is prognostic; T is closed form
#                                             in rho_liq and never reads Q_ss)
#
# cancels catastrophically in DRY AIR, where Q_ss -> -rho_vs. Neither route wins
# outright: a straight swap to res_qss is a net regression on the shipped O01
# default (1.11 % -> 2.61 % negative points) because the dry population dominates
# there, but res_qss is 5.7x better in the NOPRECIP storm's cloud.
#
# The blend variable is
#
#     s = |Q_ss| / rho_vs
#
# which is EXACTLY the conditioning number of the res_qss route: it is the ratio
# of the difference to the terms being differenced, so the relative error of
# res_qss is the relative error of rho_vs divided by s. It is dimensionless,
# already computed every step by the kernel, and it is SELF-PROTECTING: if Q_ss
# detaches from the density budget (the POSITIVITY=1 run reaches s ~ 2e26), s
# blows up and the blend weight goes to zero, handing the point back to the
# density residual with no special-case logic.
#
# Weight (C^1 smoothstep, so q_v -- hence C_vt, R_m, gamma_m and the acoustic
# coefficient -- stays continuously differentiable; a hard switch would put a
# jump into the sound speed):
#
#     x = clamp((s1 - s) / (s1 - s0), 0, 1)
#     w = x^2 (3 - 2x)
#     rho_v = w*res_qss + (1 - w)*res_rho_t
#
# so w = 1 (pure res_qss) where s <= s0 (near saturation / in cloud) and w = 0
# (pure density residual) where s >= s1 (dry, condensate-free).
#
# This script measures, OFFLINE on saved snapshots, which (s0, s1) works. It
# changes nothing in src/ and runs no model of its own. Per point it mirrors the
# kernel's diagnostic block (src/moist_compressible.jl:2002-2009) on FULL fields
# (perturbation + reference profile), so the retrieval it evaluates is the one
# the model would see.
#
# WHAT IT MEASURED (2026-07-28, quick O01, t = 3600 s; full table in
# benchmarks/output/vapor_blend_sweep.{txt,csv}). The premise above does not
# survive the measurement, and the reason is structural:
#
#   * `res_qss < 0` is ALGEBRAICALLY `s > 1` (Q_ss < -rho_vs). Measured min s over
#     the res_qss-negative points: 1.0000 on both G2 and NOPRECIP. So any band with
#     s1 <= 1 cannot import a single res_qss negative -- a stronger and cheaper
#     guarantee than the detachment self-protection the variable was chosen for.
#   * The negatives are a 1-2 % tail that sits AGAINST that ceiling in BOTH regimes,
#     not split between them. G2: every res_rho_t < 0 point has s >= 0.9993 (median
#     1.28). NOPRECIP: s in [0.876, 1.057], median 0.996, and the 719 points res_qss
#     would rescue all have s in [0.876, 1.000]. The HANDOFF's regime contrast
#     (in-cloud s = 0.081 vs dry s = 0.73) is a statement about the MEANS; the tail
#     that actually goes negative does not follow it.
#   * Consequently the assigned grid (s0 <= 0.2, s1 <= 0.8) is an exact no-op on the
#     negatives of both runs: it reproduces the res_rho_t counts and minima to the
#     last digit. The only bands that do anything are pushed up against s = 1, which
#     is where res_qss is WORST conditioned -- they are selecting on the structural
#     non-negativity of res_qss, not on conditioning.
#
# S1_GRID_EXT and the near-ceiling probe below were added to quantify that, and the
# selection block reports a do-no-harm frontier when nothing dominates.
#
# THE COMPOSITE BLEND DOES WORK (second sweep, same snapshots). What `s` cannot do,
# CLOUDINESS can: with w = w_cloud(rho_liq) * w_trust(s), every band on the whole
# (l0, l1, t0, t1) grid gives
#
#   G2       750 neg (0 cloud, 750 dry), min -8.2546e-05  -- the shipped route, unchanged
#   NOPRECIP 474 neg (472 cloud, 2 dry), min -8.6932e-07  -- the res_qss in-cloud count
#                                                            and minimum EXACTLY, with
#                                                            the residual's dry behaviour
#
# because the two populations are separated by ~8 decades in rho_liq: NOPRECIP's 719
# rescued points have median rho_liq 3.95e-03, its 425 harmed points 2.3e-14, and G2's
# 1017 harmed points sit at |rho_liq| < 3e-09. The result is insensitive to every
# threshold in the grid, which is what a real regime separation looks like.
#
# The cost is a partition gap in cloud where there is nothing to gain: on G2 all 13632
# in-cloud points acquire one (max 9.39e-04, p99 1.95e-04, median 1.02e-07 kg/m^3,
# signed sum -0.22 kg/m^3). That is a genuine 4.3 % disagreement about the vapor at the
# worst point (r = 75 km, z = 28 m, rho_liq = 8.0e-03: res_rho_t = 2.184e-02 vs
# res_qss = 2.090e-02), and it is what the reconciliation has to absorb.
#
# Requires `<outdir>/o01_exact.ref`, which every o01 run writes beside its output.
#
# Usage:
#   julia --project=. benchmarks/vapor_blend_diagnostic.jl [label=]outdir ...
#
# With no arguments it uses the shipped-default quick O01 run. Each argument may
# be a bare directory or `label=directory`; the label is only cosmetic.
# =============================================================================

using Scythe, Springsteel, CSV, DataFrames, Printf

# `detect_transforms` / `mc_water` / `water_column`: the shared inverse-map plumbing, so this
# script cannot read a control variable as a density.
include(joinpath(@__DIR__, "common", "diagnostics.jl"))

const G2_DIR = joinpath(@__DIR__, "output", "o01_rainfall_quick_mc_rirk")

# In-cloud threshold. 1e-6 kg/m^3 = 1e-3 g/m^3 is three orders below the cloud
# water this run actually makes, and above the spline ringing floor.
const CLOUD_RHO_LIQ = 1.0e-6

const S0_GRID = [0.02, 0.05, 0.1, 0.15, 0.2]
const S1_GRID = [0.3, 0.4, 0.5, 0.6, 0.8]

# Extended upper edge. The band above is where the CONDITIONING argument says the
# crossover belongs, but s is bounded above by 1 wherever rho_v >= 0 (Q_ss >= -rho_vs),
# so a run's negative points pile up against s = 1 and a band that stops at 0.8 can
# never see them. This second grid measures what a crossover placed at or above the
# physical ceiling would buy -- it is a MEASUREMENT of the ceiling, not a proposal.
const S1_GRID_EXT = [0.9, 0.95, 0.99, 1.0, 1.05, 1.2]

# Near-ceiling probe. `res_qss < 0` is ALGEBRAICALLY equivalent to s > 1 (Q_ss < -rho_vs),
# and the residual route's negatives turn out to sit just below that same ceiling, so the
# only bands that give the rescuing route real weight where it rescues are ones pushed up
# against s = 1. This block measures the CEILING of what any smoothstep in s can achieve.
const S0_PROBE = [0.8, 0.85, 0.9, 0.92, 0.95, 0.99, 1.0]
const S1_PROBE = [1.0, 1.02, 1.05, 1.1, 1.2, 1.5]

"""Every (s0, s1) band scored, deduplicated: the assigned grid, its extended upper
edge, and the near-ceiling probe."""
const ALL_BANDS = sort(unique(vcat([(a, b) for a in S0_GRID for b in S1_GRID if b > a],
                                   [(a, b) for a in S0_GRID for b in S1_GRID_EXT if b > a],
                                   [(a, b) for a in S0_PROBE for b in S1_PROBE if b > a])))

# ── COMPOSITE weight: cloudiness selects, s only rejects detachment ──────────
# The s-only sweep above refutes s as the SELECTOR, but not the regime hypothesis
# itself: NOPRECIP's 719 rescued points are 717 in cloud, and G2's 750 negatives
# are all dry. Cloudiness may separate what s cannot. So
#
#     w = w_cloud(rho_liq) * w_trust(s)
#
# with w_cloud a C^1 smoothstep UP in rho_liq (0 below l0, 1 above l1) doing the
# REGIME SELECTION, and w_trust a C^1 smoothstep DOWN in s (1 below t0, 0 above t1)
# doing nothing but rejecting a Q_ss that has detached from the density budget.
#
# The trust thresholds are deliberately placed WELL ABOVE 1 (t0 = 2 or 5). Anything
# at or below 1 would make w_trust a sign selector -- res_qss < 0 is algebraically
# s > 1 -- and the blend would become a soft clamp on negative vapor rather than an
# honest choice between two representations. With t0 >= 2 the res_qss negatives at
# s in (1, t0] import at FULL cloud weight, which is exactly the property that keeps
# this a representation choice; `n_qssneg_w` below counts them, and a nonzero count
# is the evidence.
const L0_GRID = [1.0e-7, 1.0e-6, 1.0e-5]
const L1_GRID = [1.0e-5, 1.0e-4, 5.0e-4, 1.0e-3]
const TRUST_BANDS = [(2.0, 5.0), (5.0, 20.0)]

@inline function smoothstep(x)
    y = clamp(x, 0.0, 1.0)
    return y * y * (3.0 - 2.0 * y)
end

"""Weight for the s-only blend: 1 where s <= s0, 0 where s >= s1."""
@inline blend_weight(s, s0, s1) = smoothstep((s1 - s) / (s1 - s0))

"""Cloud selector: 0 where rho_liq <= l0, 1 where rho_liq >= l1."""
@inline w_cloud(rho_liq, l0, l1) = smoothstep((rho_liq - l0) / (l1 - l0))

"""Detachment rejector: 1 where s <= t0, 0 where s >= t1."""
@inline w_trust(s, t0, t1) = smoothstep((t1 - s) / (t1 - t0))

# ── one snapshot, reduced to the fields the decision needs ───────────────────
struct Snapshot
    label::String
    dir::String
    t::Float64
    res_rho_t::Vector{Float64}   # rho_t - rho_d - rho_c - rho_r
    res_qss::Vector{Float64}     # Q_ss + rho_vs(T, p)
    s::Vector{Float64}           # |Q_ss| / rho_vs
    rho_liq::Vector{Float64}     # rho_c + rho_r, the composite blend's first variable
    rho_vs::Vector{Float64}      # rho_v_sat(T, p); Q_ss = res_qss - rho_vs
    kDim::Int                    # levels per column, z fastest (for column roughness)
    incloud::BitVector           # rho_liq > CLOUD_RHO_LIQ
    r::Vector{Float64}
    z::Vector{Float64}
end

"""Build the grid + exact reference state the run used, from a snapshot's shape."""
function reference_profiles(outdir, df0)
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
        l_q = Dict("default" => 2.0),
        BCL = side_bc, BCR = side_bc, BCB = topbot_bc, BCT = topbot_bc,
        vars = Dict(v => i for (i, v) in enumerate(vars)),
    )
    patch = createGrid(grid_params)
    z = Scythe.getGridpoints(patch)[1:kDim, 2]
    column = Scythe.reference_column(patch, grid_params)
    ref = Springsteel.exact_pressure_reference_state(joinpath(outdir, "o01_exact.ref"),
                                                     z, column)
    return (; ncols, kDim,
            pbar     = Springsteel.ref_pressure(ref)[:, 1],
            rho_dbar = Springsteel.ref_rho_d(ref)[:, 1],
            rho_tbar = Springsteel.ref_rho_t(ref)[:, 1],
            rho_cbar = Springsteel.ref_rho_c(ref)[:, 1],
            E_tbar   = Springsteel.ref_total_energy(ref)[:, 1],
            Q_ssbar  = Springsteel.ref_qss(ref)[:, 1],
            # What slots 8/9 hold in THIS run (see `detect_transforms`).
            trans    = detect_transforms(outdir))
end

"""
Mirror of the kernel's diagnostic block on FULL fields. Everything except rho_r
is carried as a perturbation from the exact reference profile.
"""
function analyze(df, R)
    tile(v) = repeat(v, R.ncols)
    p     = df.p     .+ tile(R.pbar)
    rho_d = df.rho_d .+ tile(R.rho_dbar)
    rho_t = df.rho_t .+ tile(R.rho_tbar)
    E_t   = df.E_t   .+ tile(R.E_tbar)
    Q_ss  = df.Q_ss  .+ tile(R.Q_ssbar)
    # Slots 8/9 may hold Ooyama control variables rather than densities; the column NAMES
    # say which, and `R.trans` carries the variant and bias read from the run's own log.
    rho_c, rho_r = mc_water(df, R.rho_cbar, R.ncols;
                            ctrans = R.trans.ctrans, cmu = R.trans.cmu,
                            rtrans = R.trans.rtrans, rmu = R.trans.rmu)

    ke = 0.5 .* ((df.u .^ 2) .+ (df.w .^ 2))          # moist_compressible_XZ
    M = p .+ E_t .- (rho_t .* (ke .+ (Scythe.gravity .* df.z)))
    rho_liq = rho_c .+ rho_r
    Tk = Scythe.retrieve_temperature.(M, rho_d, rho_t, rho_liq)
    rho_vs = Springsteel.rho_v_sat.(Tk, p ./ 100.0)

    res_rho_t = rho_t .- rho_d .- rho_liq
    res_qss   = Q_ss .+ rho_vs
    s = abs.(Q_ss) ./ max.(rho_vs, eps())
    return res_rho_t, res_qss, s, rho_liq, rho_vs, (rho_liq .> CLOUD_RHO_LIQ)
end

"""
Load the LAST usable snapshot at or before `tmax`. "Usable" = every field the
retrieval reads is finite; the POSITIVITY=1 run detonates, so its useful
snapshots stop before the file list does.
"""
function load_snapshot(label, outdir; tmax = 3600.0)
    snaps = sort([(parse(Float64, replace(basename(f), "_physical.csv" => "")), f)
                  for f in readdir(outdir; join = true) if endswith(f, "_physical.csv")];
                 by = first)
    isempty(snaps) && error("no *_physical.csv snapshots in $outdir")
    R = reference_profiles(outdir, CSV.read(snaps[1][2], DataFrame))

    for (t, path) in Iterators.reverse(snaps)
        t > tmax && continue
        df = CSV.read(path, DataFrame)
        cols = (df.p, df.rho_d, df.rho_t, df.E_t, df.Q_ss, df.u, df.w,
                first(water_column(df, "rho_c")), first(water_column(df, "rho_r")))
        all(c -> all(isfinite, c), cols) || continue
        res_rho_t, res_qss, s, rho_liq, rho_vs, incloud = analyze(df, R)
        all(isfinite, res_qss) && all(isfinite, s) || continue
        return Snapshot(label, outdir, t, res_rho_t, res_qss, s, rho_liq, rho_vs,
                        R.kDim, incloud, df.r, df.z)
    end
    error("no finite snapshot at or before t = $tmax in $outdir")
end

# ── scoring ──────────────────────────────────────────────────────────────────
struct Score
    neg::Int
    neg_cloud::Int
    neg_dry::Int
    minv::Float64
    gap::Float64          # max |blend - res_rho_t|, the partition over-determination
    gap_p99::Float64      # 99th pct of that gap over the ACTIVE set (w > 0); the max
                          # alone can be one outlier, and over the whole domain the
                          # percentile is swamped by the w = 0 points that gap exactly 0
    n_active::Int         # points with w > 0
    n_qssneg_w::Int       # res_qss < 0 points that receive w > 0 -- the "this is not a
                          # clamp" counter: a clamp would import none of them
    signed_gap::Float64   # sum(blend - res_rho_t): the net mass the reconciliation
                          # (qss_relaxation) has to absorb, in kg/m^3 summed over points
end

"""Score an arbitrary pointwise weight field. Every route is one of these."""
function score(snap::Snapshot, w::Vector{Float64})
    n = length(snap.res_rho_t)
    neg = 0; negc = 0; minv = Inf; gap = 0.0; nact = 0; nqw = 0; sgap = 0.0
    active_gaps = Float64[]
    @inbounds for i in 1:n
        wi = w[i]
        d = wi * (snap.res_qss[i] - snap.res_rho_t[i])   # blend - res_rho_t, exactly
        v = snap.res_rho_t[i] + d
        v < minv && (minv = v)
        gap = max(gap, abs(d))
        sgap += d
        if wi > 0.0
            nact += 1
            push!(active_gaps, abs(d))
            snap.res_qss[i] < 0.0 && (nqw += 1)
        end
        if v < 0.0
            neg += 1
            snap.incloud[i] && (negc += 1)
        end
    end
    p99 = isempty(active_gaps) ? 0.0 :
          (sort!(active_gaps); active_gaps[max(1, cld(99 * length(active_gaps), 100))])
    return Score(neg, negc, neg - negc, minv, gap, p99, nact, nqw, sgap)
end

s_weights(snap::Snapshot, s0, s1) = [blend_weight(x, s0, s1) for x in snap.s]
composite_weights(snap::Snapshot, l0, l1, t0, t1) =
    [w_cloud(snap.rho_liq[i], l0, l1) * w_trust(snap.s[i], t0, t1)
     for i in eachindex(snap.s)]

score(snap::Snapshot, s0::Float64, s1::Float64) = score(snap, s_weights(snap, s0, s1))

"""Pure routes, expressed through the same scorer: w == 0 or w == 1 everywhere."""
pure_score(snap::Snapshot, which::Symbol) =
    score(snap, fill(which === :qss ? 1.0 : 0.0, length(snap.s)))

# ── STAGE 0b: tau_qss sensitivity of the partition gap ───────────────────────
# The adopted composite band. `qss_relaxation` is
#     QSSREL = -(Q_ss - (rho_v - rho_vs))/tau   with rho_v the DENSITY residual,
# and since res_qss - res_rho_t = Q_ss - (rho_v - rho_vs) exactly, the partition gap
# the blend opens IS the relaxation residual times tau:
#     gap = res_qss - res_rho_t = -tau * QSSREL.
# So shortening tau shrinks the gap mechanically. The question this report answers is
# whether it shrinks the BENEFIT (NOPRECIP's in-cloud rescue) just as fast, because the
# same relaxation drags Q_ss onto the residual that is corrupted in exactly that cloud.
# If gap and rescue scale together, tau is not a lever and the gap is the price.
const ADOPTED_BAND = (1.0e-6, 1.0e-4, 2.0, 5.0)

"""Fraction of vertically adjacent in-cloud pairs across which `f` changes sign.
A smooth field gives a small number; a field carrying fit-level noise tends to 0.5.
This is the roughness proxy for "is fast relaxation transferring the fitted residual's
noise into the transported Q_ss"."""
function column_sign_flips(snap::Snapshot, f::Vector{Float64})
    kD = snap.kDim
    ncol = length(f) ÷ kD
    flips = 0; pairs = 0
    for c in 0:(ncol - 1), k in 1:(kD - 1)
        i = c * kD + k; j = i + 1
        (snap.incloud[i] && snap.incloud[j]) || continue
        pairs += 1
        (f[i] * f[j] < 0.0) && (flips += 1)
    end
    return pairs == 0 ? NaN : flips / pairs, pairs
end

"""Scalar diagnostics the run wrote (max_w, accum_rainfall_mm, ...), or missing."""
function run_diagnostics(dir)
    path = joinpath(dir, "diagnostics.csv")
    isfile(path) || return Dict{String,Float64}()
    df = CSV.read(path, DataFrame)
    nm = names(df)
    kcol = "name" in nm ? :name : Symbol(nm[1])
    vcol = "value" in nm ? :value : Symbol(nm[end])
    return Dict(String(r[kcol]) => Float64(r[vcol]) for r in eachrow(df))
end

function tau_report(snaps)
    l0, l1, t0, t1 = ADOPTED_BAND
    println("="^118)
    @printf("STAGE 0b  tau_qss SENSITIVITY   composite band l0=%.0e l1=%.0e t0=%.0f t1=%.0f\n",
            l0, l1, t0, t1)
    println("  identity: gap = res_qss - res_rho_t = -tau_qss * QSSREL, so |QSSREL| = |gap|/tau")
    println("="^118)
    for snap in snaps
        d = run_diagnostics(snap.dir)
        w = composite_weights(snap, l0, l1, t0, t1)
        b_r = pure_score(snap, :rho_t); b_q = pure_score(snap, :qss)
        sc = score(snap, w)
        Q_ss = snap.res_qss .- snap.rho_vs
        gapall = snap.res_qss .- snap.res_rho_t          # independent of the weight
        cloud = snap.incloud
        gc = abs.(gapall[cloud]); sort!(gc)
        m = length(gc)
        signed_cloud = sum(gapall[cloud])
        # gap actually applied by the blend (weighted), in cloud
        dblend = [w[i] * gapall[i] for i in eachindex(w) if cloud[i]]
        adb = sort(abs.(dblend))
        ss = Q_ss ./ max.(snap.rho_vs, eps())            # SIGNED supersaturation ratio
        fq, np = column_sign_flips(snap, Q_ss)
        fr, _  = column_sign_flips(snap, snap.res_rho_t .- snap.rho_vs)

        @printf("\n%-24s t = %.1f s   (%d points, %d in cloud)\n",
                snap.label, snap.t, length(w), count(cloud))
        @printf("  pure res_rho_t : %6d neg (%5d cloud, %5d dry)  min %+.4e\n",
                b_r.neg, b_r.neg_cloud, b_r.neg_dry, b_r.minv)
        @printf("  pure res_qss   : %6d neg (%5d cloud, %5d dry)  min %+.4e\n",
                b_q.neg, b_q.neg_cloud, b_q.neg_dry, b_q.minv)
        @printf("  COMPOSITE      : %6d neg (%5d cloud, %5d dry)  min %+.4e   n_qssneg_w %d\n",
                sc.neg, sc.neg_cloud, sc.neg_dry, sc.minv, sc.n_qssneg_w)
        @printf("  in-cloud gap (raw res_qss-res_rho_t): max %.4e  p99 %.4e  median %.4e  SIGNED SUM %+.6e kg/m^3\n",
                m == 0 ? NaN : gc[end], m == 0 ? NaN : gc[max(1, cld(99m, 100))],
                m == 0 ? NaN : gc[cld(m, 2)], signed_cloud)
        @printf("  in-cloud gap AS BLENDED (w-weighted): max %.4e  p99 %.4e  SIGNED SUM %+.6e kg/m^3\n",
                isempty(adb) ? NaN : adb[end],
                isempty(adb) ? NaN : adb[max(1, cld(99 * length(adb), 100))], sum(dblend))
        @printf("  supersaturation Q_ss/rho_vs : max %+.4e  min %+.4e   (in cloud: max %+.4e  min %+.4e)\n",
                maximum(ss), minimum(ss),
                count(cloud) == 0 ? NaN : maximum(ss[cloud]),
                count(cloud) == 0 ? NaN : minimum(ss[cloud]))
        @printf("  Q_ss column sign-flip fraction in cloud : %.4f   (diagnosed rho_v-rho_vs: %.4f)  over %d pairs\n",
                fq, fr, np)
        for (k, lbl) in (("max_w", "max_w"), ("accum_rainfall_mm", "accum_rainfall_mm"),
                         ("min_rho_v_gm3", "min_rho_v_gm3"), ("max_rho_r_gm3", "max_rho_r_gm3"))
            haskey(d, k) && @printf("  %-20s = %.6g\n", lbl, d[k])
        end
    end
end

# ── main ─────────────────────────────────────────────────────────────────────
tau_only = "--tau-report" in ARGS
specs = filter(!startswith("--"), ARGS)
isempty(specs) && (specs = ["G2=$(G2_DIR)"])
snaps = Snapshot[]
for spec in specs
    label, dir = occursin('=', spec) ? split(spec, '=', limit = 2) :
                 (basename(rstrip(spec, '/')), spec)
    push!(snaps, load_snapshot(String(label), String(dir)))
end

if tau_only
    tau_report(snaps)
    exit(0)
end

println("="^100)
println("VAPOR RETRIEVAL BLEND SWEEP   res_rho_t = rho_t-rho_d-rho_c-rho_r   " *
        "res_qss = Q_ss+rho_vs   s = |Q_ss|/rho_vs")
println("="^100)

for snap in snaps
    n = length(snap.s)
    nc = count(snap.incloud)
    @printf("\n%-10s %s\n", snap.label, snap.dir)
    @printf("  snapshot t = %.1f s, %d points, %d in cloud (%.2f %%), %d dry\n",
            snap.t, n, nc, 100 * nc / n, n - nc)
    if nc > 0
        sc = snap.s[snap.incloud]
        @printf("  in cloud: mean s = %.4f, median s = %.4f, max s = %.4g\n",
                sum(sc) / nc, sort(sc)[cld(nc, 2)], maximum(sc))
    end
    sd = snap.s[.!snap.incloud]
    @printf("  dry     : mean s = %.4f, median s = %.4f, max s = %.4g\n",
            sum(sd) / length(sd), sort(sd)[cld(length(sd), 2)], maximum(sd))
    # Self-protection: how much of the domain is so detached that every candidate
    # band gives it w = 0 regardless of (s0, s1)?
    detached = count(>(maximum(S1_GRID)), snap.s)
    @printf("  s > %.2f (w = 0 under every candidate band): %d points (%.2f %%)\n",
            maximum(S1_GRID), detached, 100 * detached / n)

    # WHERE THE NEGATIVES SIT IN s. This is what decides whether ANY band can help:
    # the blend can only change a point whose s falls inside [s0, s1]. Q_ss >= -rho_vs
    # is equivalent to rho_v >= 0 on the res_qss route, so a point that route calls
    # negative necessarily has s > 1, and a point the DENSITY route calls negative has
    # rho_v ~ 0 there, which -- to the extent qss_relaxation (tau_qss = 10 s, i.e. 360
    # e-foldings over this hour) has slaved Q_ss to that same residual -- drags s toward
    # 1 as well. If the negatives really do pile up at s ~ 1, no band with s1 < 1 can
    # touch them, and that is a property of the two representations, not of the grid.
    function s_at(mask, what)
        m = count(mask)
        if m == 0
            @printf("  s at %-22s : none\n", what)
            return
        end
        v = sort(snap.s[mask])
        @printf("  s at %-22s : n = %5d  min %.4f  p10 %.4f  median %.4f  p90 %.4f  max %.4g\n",
                what, m, v[1], v[max(1, cld(m, 10))], v[cld(m, 2)],
                v[max(1, cld(9m, 10))], v[end])
    end
    negr = snap.res_rho_t .< 0.0
    negq = snap.res_qss .< 0.0
    s_at(negr, "res_rho_t < 0")
    s_at(negr .& snap.incloud, "res_rho_t < 0, in cloud")
    s_at(negq, "res_qss < 0")
    @printf("  negative-set overlap: both %d, res_rho_t only %d, res_qss only %d\n",
            count(negr .& negq), count(negr .& .!negq), count(negq .& .!negr))
    @printf("  res_qss RESCUES (res_rho_t < 0 <= res_qss) at %d points, %d of them in cloud\n",
            count(negr .& .!negq), count(negr .& .!negq .& snap.incloud))
    resc = negr .& .!negq
    if count(resc) > 0
        v = sort(snap.s[resc])
        @printf("    s over the rescued points: min %.4f  median %.4f  max %.4f  (%d of %d have s <= %.2f, reachable by the candidate grid)\n",
                v[1], v[cld(length(v), 2)], v[end],
                count(<=(maximum(S1_GRID)), v), length(v), maximum(S1_GRID))
    end
end

for snap in snaps
    @printf("\n%s\n", "-"^100)
    @printf("%s  (t = %.1f s)\n", snap.label, snap.t)
    @printf("%s\n", "-"^100)
    @printf("  %-6s %-6s | %8s %8s %8s  %12s  %12s\n",
            "s0", "s1", "neg", "in cloud", "dry", "min rho_v", "max gap")
    for (name, which) in (("res_rho_t", :rho_t), ("res_qss", :qss))
        sc = pure_score(snap, which)
        @printf("  %-13s | %8d %8d %8d  %+12.4e  %12.4e\n",
                name, sc.neg, sc.neg_cloud, sc.neg_dry, sc.minv, sc.gap)
    end
    for s0 in S0_GRID, s1 in S1_GRID
        s1 > s0 || continue
        sc = score(snap, s0, s1)
        @printf("  %-6.2f %-6.2f | %8d %8d %8d  %+12.4e  %12.4e\n",
                s0, s1, sc.neg, sc.neg_cloud, sc.neg_dry, sc.minv, sc.gap)
    end
    println("  -- extended upper edge (s1 at/above the physical s <= 1 ceiling) --")
    for s0 in S0_GRID, s1 in S1_GRID_EXT
        s1 > s0 || continue
        sc = score(snap, s0, s1)
        @printf("  %-6.2f %-6.2f | %8d %8d %8d  %+12.4e  %12.4e\n",
                s0, s1, sc.neg, sc.neg_cloud, sc.neg_dry, sc.minv, sc.gap)
    end
    println("  -- near-ceiling probe: the best any smoothstep in s can do --")
    for s0 in S0_PROBE, s1 in S1_PROBE
        s1 > s0 || continue
        sc = score(snap, s0, s1)
        @printf("  %-6.2f %-6.2f | %8d %8d %8d  %+12.4e  %12.4e\n",
                s0, s1, sc.neg, sc.neg_cloud, sc.neg_dry, sc.minv, sc.gap)
    end
end

# ── COMPOSITE SWEEP: w = w_cloud(rho_liq) * w_trust(s) ───────────────────────
println("\n" * "="^100)
println("COMPOSITE BLEND   w = w_cloud(rho_liq; l0, l1) * w_trust(s; t0, t1)")
println("  w_cloud selects the regime (0 below l0, 1 above l1); w_trust only rejects")
println("  detached Q_ss (1 below t0, 0 above t1), with t0 >= 2 so it can never act as")
println("  a sign selector. n_qssneg_w > 0 is the evidence this is not a soft clamp.")
println("="^100)

"""Quantiles of `v` reported the same way throughout."""
function qline(tag, v)
    m = length(v)
    if m == 0
        @printf("    %-34s n =     0\n", tag)
        return
    end
    u = sort(v)
    @printf("    %-34s n = %5d  min %.3e  p10 %.3e  median %.3e  p90 %.3e  max %.3e\n",
            tag, m, u[1], u[max(1, cld(m, 10))], u[cld(m, 2)],
            u[max(1, cld(9m, 10))], u[end])
end

# Where the two populations the composite must separate actually live in rho_liq.
# RESCUED = res_rho_t < 0 <= res_qss (the blend should reach these).
# HARMED  = res_qss < 0 <= res_rho_t (the blend imports a negative here -- honest, but
#           it is the cost side of the ledger, and it is what w_cloud must exclude).
for snap in snaps
    negr = snap.res_rho_t .< 0.0
    negq = snap.res_qss .< 0.0
    @printf("\n  %s: rho_liq [kg/m^3] of the populations the cloud selector must separate\n",
            snap.label)
    qline("RESCUED (res_rho_t<0<=res_qss)", snap.rho_liq[negr .& .!negq])
    qline("HARMED  (res_qss<0<=res_rho_t)", snap.rho_liq[negq .& .!negr])
    qline("both negative", snap.rho_liq[negr .& negq])
    qline("neither negative", snap.rho_liq[.!negr .& .!negq])
end

const COMPOSITE_BANDS = [(l0, l1, t0, t1) for l0 in L0_GRID for l1 in L1_GRID
                         for (t0, t1) in TRUST_BANDS if l1 > l0]

for snap in snaps
    @printf("\n%s\n", "-"^118)
    @printf("%s  (t = %.1f s)   COMPOSITE\n", snap.label, snap.t)
    @printf("%s\n", "-"^118)
    @printf("  %-8s %-8s %-3s %-4s | %8s %8s %8s  %12s  %11s %11s  %7s %8s\n",
            "l0", "l1", "t0", "t1", "neg", "in cloud", "dry", "min rho_v",
            "gap max", "gap p99", "n_act", "n_qssneg")
    for (name, which) in (("res_rho_t", :rho_t), ("res_qss", :qss))
        sc = pure_score(snap, which)
        @printf("  %-26s | %8d %8d %8d  %+12.4e  %11.4e %11.4e  %7d %8d\n",
                name, sc.neg, sc.neg_cloud, sc.neg_dry, sc.minv, sc.gap, sc.gap_p99,
                sc.n_active, sc.n_qssneg_w)
    end
    for (l0, l1, t0, t1) in COMPOSITE_BANDS
        sc = score(snap, composite_weights(snap, l0, l1, t0, t1))
        @printf("  %-8.0e %-8.0e %-3.0f %-4.0f | %8d %8d %8d  %+12.4e  %11.4e %11.4e  %7d %8d\n",
                l0, l1, t0, t1, sc.neg, sc.neg_cloud, sc.neg_dry, sc.minv,
                sc.gap, sc.gap_p99, sc.n_active, sc.n_qssneg_w)
    end
end

# ── what the reconciliation must absorb, IN G2's CLOUD ───────────────────────
# G2 has zero negatives from either route in cloud, so every composite band is pure
# cost there: it buys nothing and opens a partition gap that qss_relaxation has to
# carry. This block sizes that cost on whichever supplied run has no in-cloud
# negatives to gain (G2 by construction).
println("\n" * "="^100)
println("PARTITION GAP IN CLOUD WHERE THERE IS NOTHING TO GAIN (G2's cloud: 0 negatives, both routes)")
println("="^100)
for snap in snaps
    b_r = pure_score(snap, :rho_t)
    b_q = pure_score(snap, :qss)
    (b_r.neg_cloud == 0 && b_q.neg_cloud == 0) || continue
    @printf("  %s: %d in-cloud points, both routes 0 negatives there\n",
            snap.label, count(snap.incloud))
    for (l0, l1, t0, t1) in COMPOSITE_BANDS
        w = composite_weights(snap, l0, l1, t0, t1)
        act = snap.incloud .& (w .> 0.0)
        ca = count(act)
        ca == 0 && continue
        d = [w[i] * (snap.res_qss[i] - snap.res_rho_t[i]) for i in eachindex(w) if act[i]]
        ad = sort(abs.(d))
        @printf("    l0=%.0e l1=%.0e t0=%.0f t1=%.0f : %5d cloud points gap != 0  max %.3e  p99 %.3e  median %.3e  signed sum %+.4e kg/m^3\n",
                l0, l1, t0, t1, ca, ad[end], ad[max(1, cld(99 * ca, 100))],
                ad[cld(ca, 2)], sum(d))
    end
    # WHERE the worst of that gap sits, for the first band (they all share the same
    # max: it is one point, and it is the same point for every threshold pair).
    l0, l1, t0, t1 = COMPOSITE_BANDS[1]
    w = composite_weights(snap, l0, l1, t0, t1)
    dall = [w[i] * (snap.res_qss[i] - snap.res_rho_t[i]) for i in eachindex(w)]
    j = argmax(abs.(dall) .* snap.incloud)
    @printf("    worst in-cloud gap point: r = %.2f km, z = %.3f km, rho_liq = %.4e, s = %.4f, w = %.4f\n",
            snap.r[j] / 1e3, snap.z[j] / 1e3, snap.rho_liq[j], snap.s[j], w[j])
    @printf("      res_rho_t = %+.6e   res_qss = %+.6e   blend - res_rho_t = %+.6e kg/m^3\n",
            snap.res_rho_t[j], snap.res_qss[j], dall[j])
    # How many in-cloud points carry a gap that MATTERS, i.e. is a material fraction
    # of the local vapor the two routes are arguing about.
    for frac in (0.01, 0.1)
        c = count(i -> snap.incloud[i] && abs(dall[i]) >
                       frac * max(abs(snap.res_rho_t[i]), eps()), eachindex(w))
        @printf("      in-cloud points where |gap| > %.0f %% of res_rho_t: %d (%.2f %% of cloud)\n",
                100 * frac, c, 100 * c / count(snap.incloud))
    end
end

# ── self-protection against Q_ss detachment ──────────────────────────────────
# The clamp in `blend_weight` makes w EXACTLY zero for every s >= s1, so a point
# whose Q_ss has detached from the density budget is handed back to the density
# residual bitwise -- no special case, no tolerance. This block verifies that
# numerically on whatever runs were supplied, and reports how detached they get.
println("\n" * "="^100)
println("SELF-PROTECTION: points whose Q_ss has detached (s >> 1) must get w = 0 and blend == res_rho_t")
println("="^100)
for snap in snaps
    n = length(snap.s)
    for thresh in (10.0, 1.0e3, 1.0e6)
        m = snap.s .> thresh
        cm = count(m)
        cm == 0 && continue
        # Worst case over the widest band scored (largest s1): if w is 0 there it is
        # 0 for every narrower band too, since w is nonincreasing in s and in s1.
        s1max = maximum(b[2] for b in ALL_BANDS)
        s0max = maximum(b[1] for b in ALL_BANDS if b[2] == s1max)
        wmax = maximum(blend_weight(x, s0max, s1max) for x in snap.s[m])
        dev = 0.0
        for i in 1:n
            m[i] || continue
            w = blend_weight(snap.s[i], s0max, s1max)
            dev = max(dev, abs((w * snap.res_qss[i]) + ((1.0 - w) * snap.res_rho_t[i]) -
                               snap.res_rho_t[i]))
        end
        @printf("  %-10s s > %-8.0e : %6d points (%5.2f %%)  max w = %.3e  max |blend - res_rho_t| = %.3e\n",
                snap.label, thresh, cm, 100 * cm / n, wmax, dev)
    end
    @printf("  %-10s max s over the domain = %.4g\n", snap.label, maximum(snap.s))
    # COMPOSITE weight: the guarantee now comes from w_trust, whose clamp puts w
    # exactly 0 above t1 (5 or 20). Verified for the LOOSEST trust band -- if it is
    # exact there it is exact for the tighter one, since w_trust is nonincreasing
    # in both s and t1.
    # BOTH trust bands: the guarantee is w == 0 above t1, so the (10, 1e3, 1e6)
    # thresholds are exact for t0/t1 = 2/5 but NOT at s = 10 for 5/20, where the
    # smoothstep is still open. `s > t1` is the row that states the guarantee itself.
    l0w = minimum(L0_GRID); l1w = minimum(l for l in L1_GRID if l > l0w)
    for (t0w, t1w) in TRUST_BANDS, thresh in (10.0, 1.0e3, 1.0e6, t1w)
        m = snap.s .> thresh
        cm = count(m)
        cm == 0 && continue
        wmax = 0.0; dev = 0.0
        for i in 1:n
            m[i] || continue
            w = w_cloud(snap.rho_liq[i], l0w, l1w) * w_trust(snap.s[i], t0w, t1w)
            wmax = max(wmax, w)
            dev = max(dev, abs(w * (snap.res_qss[i] - snap.res_rho_t[i])))
        end
        @printf("  %-10s composite t0=%-3.0f t1=%-3.0f  s > %-8.0e : %6d points  max w = %.3e  max |blend - res_rho_t| = %.3e%s\n",
                snap.label, t0w, t1w, thresh, cm, wmax, dev,
                thresh == t1w ? "   <- the guarantee" : "")
    end
end

# ── machine-readable dump ────────────────────────────────────────────────────
rows = DataFrame(run = String[], t = Float64[], s0 = Float64[], s1 = Float64[],
                 neg = Int[], neg_cloud = Int[], neg_dry = Int[],
                 min_rho_v = Float64[], max_gap = Float64[], npoints = Int[],
                 ncloud = Int[])
for snap in snaps
    n = length(snap.s); nc = count(snap.incloud)
    entries = vcat([(NaN, 0.0, pure_score(snap, :rho_t)),
                    (NaN, 1.0, pure_score(snap, :qss))],
                   [(s0, s1, score(snap, s0, s1)) for (s0, s1) in ALL_BANDS])
    for (s0, s1, sc) in entries
        push!(rows, (snap.label, snap.t, s0, s1, sc.neg, sc.neg_cloud, sc.neg_dry,
                     sc.minv, sc.gap, n, nc))
    end
end
outcsv = joinpath(@__DIR__, "output", "vapor_blend_sweep.csv")
CSV.write(outcsv, rows)
@printf("\nfull s-only sweep written to %s (%d rows)\n", outcsv, nrow(rows))

crows = DataFrame(run = String[], t = Float64[], l0 = Float64[], l1 = Float64[],
                  t0 = Float64[], t1 = Float64[], neg = Int[], neg_cloud = Int[],
                  neg_dry = Int[], min_rho_v = Float64[], max_gap = Float64[],
                  gap_p99_active = Float64[], n_active = Int[], n_qssneg_weighted = Int[],
                  signed_gap = Float64[], npoints = Int[], ncloud = Int[])
for snap in snaps
    n = length(snap.s); nc = count(snap.incloud)
    entries = vcat([(NaN, NaN, NaN, 0.0, pure_score(snap, :rho_t)),
                    (NaN, NaN, NaN, 1.0, pure_score(snap, :qss))],
                   [(l0, l1, t0, t1, score(snap, composite_weights(snap, l0, l1, t0, t1)))
                    for (l0, l1, t0, t1) in COMPOSITE_BANDS])
    for (l0, l1, t0, t1, sc) in entries
        push!(crows, (snap.label, snap.t, l0, l1, t0, t1, sc.neg, sc.neg_cloud,
                      sc.neg_dry, sc.minv, sc.gap, sc.gap_p99, sc.n_active,
                      sc.n_qssneg_w, sc.signed_gap, n, nc))
    end
end
ccsv = joinpath(@__DIR__, "output", "vapor_blend_composite_sweep.csv")
CSV.write(ccsv, crows)
@printf("composite sweep written to %s (%d rows)\n", ccsv, nrow(crows))

# ── selection ────────────────────────────────────────────────────────────────
# The bar from the plan: a candidate must beat BOTH pure routes in the regime
# where each is weak, on EVERY run scored here (the POSITIVITY run is excluded --
# it is the self-protection check, not a quality target).
println("\n" * "="^100)
println("SELECTION  (dry negatives <= res_rho_t baseline AND in-cloud negatives <= res_qss baseline)")
println("="^100)
targets = [s for s in snaps if !occursin("POS", uppercase(s.label))]
if isempty(targets)
    println("  no quality-target runs supplied")
else
    base = Dict(s.label => (pure_score(s, :rho_t), pure_score(s, :qss)) for s in targets)
    for s in targets
        b_r, b_q = base[s.label]
        @printf("  %-10s baseline dry (res_rho_t) = %d, in-cloud (res_qss) = %d\n",
                s.label, b_r.neg_dry, b_q.neg_cloud)
    end
    winners = Tuple{Float64,Float64}[]
    for (s0, s1) in ALL_BANDS
        ok = all(targets) do s
            b_r, b_q = base[s.label]
            sc = score(s, s0, s1)
            sc.neg_dry <= b_r.neg_dry && sc.neg_cloud <= b_q.neg_cloud
        end
        ok && push!(winners, (s0, s1))
    end
    if isempty(winners)
        println("\n  NO candidate dominates both pure routes on every run.")
        # Fallback bar: DO NO HARM anywhere (no run's negatives, in either regime,
        # exceed what the shipped res_rho_t route already produces) and improve where
        # it can. Ranked by in-cloud negatives, tie-break toward the widest band.
        println("\n  DO-NO-HARM frontier (every run's neg/cloud/dry <= its res_rho_t baseline),")
        println("  ranked by summed in-cloud negatives, then by band width:")
        harmless = filter(ALL_BANDS) do (s0, s1)
            all(targets) do s
                b_r, _ = base[s.label]
                sc = score(s, s0, s1)
                sc.neg <= b_r.neg && sc.neg_cloud <= b_r.neg_cloud && sc.neg_dry <= b_r.neg_dry
            end
        end
        if isempty(harmless)
            println("    none -- every band makes at least one run worse than the shipped route.")
        else
            sort!(harmless; by = p -> (sum(score(s, p[1], p[2]).neg_cloud for s in targets),
                                       -(p[2] - p[1])))
            for (s0, s1) in harmless[1:min(10, end)]
                @printf("    s0 = %.2f  s1 = %.2f  width %.2f", s0, s1, s1 - s0)
                for s in targets
                    sc = score(s, s0, s1)
                    @printf("   | %s: %d neg (%d cloud, %d dry) min %+.3e",
                            s.label, sc.neg, sc.neg_cloud, sc.neg_dry, sc.minv)
                end
                println()
            end
            @printf("\n  BEST DO-NO-HARM BAND: s0 = %.2f, s1 = %.2f  (NOT a dominating candidate --\n",
                    harmless[1][1], harmless[1][2])
            println("  it sits against the s = 1 ceiling, where res_qss >= 0 holds structurally,")
            println("  so it acts as a smooth non-negativity selector rather than a conditioning")
            println("  selector. Treat that as a physics decision, not a numerics one.)")
        end
    else
        # Tie-break toward the widest band, per the plan.
        sort!(winners; by = p -> -(p[2] - p[1]))
        println("\n  candidates that dominate (widest band first):")
        for (s0, s1) in winners
            @printf("    s0 = %.2f  s1 = %.2f   width %.2f", s0, s1, s1 - s0)
            for s in targets
                sc = score(s, s0, s1)
                @printf("   | %s: %d neg (%d cloud, %d dry)",
                        s.label, sc.neg, sc.neg_cloud, sc.neg_dry)
            end
            println()
        end
        @printf("\n  RECOMMENDED: s0 = %.2f, s1 = %.2f\n", winners[1][1], winners[1][2])
    end
end

# ── composite selection ──────────────────────────────────────────────────────
# Same bar, applied to the cloudiness-selected blend:
#   G2       negatives and minimum no worse than the shipped res_rho_t route
#   NOPRECIP in-cloud negatives as close as possible to the res_qss baseline
#   POS1     detached points bitwise residual (checked above, not re-tested here)
println("\n" * "="^100)
println("COMPOSITE SELECTION")
println("="^100)
if isempty(targets)
    println("  no quality-target runs supplied")
else
    for s in targets
        b_r, b_q = base[s.label]
        @printf("  %-10s res_rho_t baseline %d neg (%d cloud, %d dry) min %+.4e | res_qss %d neg (%d cloud, %d dry) min %+.4e\n",
                s.label, b_r.neg, b_r.neg_cloud, b_r.neg_dry, b_r.minv,
                b_q.neg, b_q.neg_cloud, b_q.neg_dry, b_q.minv)
    end
    # No-harm filter on every target run, then rank by in-cloud negatives.
    ok_bands = filter(COMPOSITE_BANDS) do (l0, l1, t0, t1)
        all(targets) do s
            b_r, _ = base[s.label]
            sc = score(s, composite_weights(s, l0, l1, t0, t1))
            sc.neg <= b_r.neg && sc.neg_cloud <= b_r.neg_cloud &&
                sc.neg_dry <= b_r.neg_dry && sc.minv >= b_r.minv
        end
    end
    if isempty(ok_bands)
        println("\n  NO composite band leaves every target run no worse than the shipped route.")
        println("  Ranked by summed in-cloud negatives anyway:")
        ranked = sort([(b, sum(score(s, composite_weights(s, b...)).neg_cloud
                               for s in targets)) for b in COMPOSITE_BANDS]; by = last)
        for ((l0, l1, t0, t1), tot) in ranked[1:min(8, end)]
            @printf("    l0=%.0e l1=%.0e t0=%.0f t1=%.0f  summed in-cloud negatives = %d\n",
                    l0, l1, t0, t1, tot)
        end
    else
        sort!(ok_bands; by = b -> sum(score(s, composite_weights(s, b...)).neg_cloud
                                      for s in targets))
        println("\n  DO-NO-HARM composite bands, best in-cloud first:")
        for (l0, l1, t0, t1) in ok_bands[1:min(10, end)]
            @printf("    l0=%.0e l1=%.0e t0=%.0f t1=%.0f", l0, l1, t0, t1)
            for s in targets
                sc = score(s, composite_weights(s, l0, l1, t0, t1))
                @printf("  | %s: %d neg (%d cloud, %d dry) min %+.3e gapp99 %.2e qssneg_w %d",
                        s.label, sc.neg, sc.neg_cloud, sc.neg_dry, sc.minv,
                        sc.gap_p99, sc.n_qssneg_w)
            end
            println()
        end
        b = ok_bands[1]
        @printf("\n  RECOMMENDED COMPOSITE: l0 = %.0e, l1 = %.0e, t0 = %.0f, t1 = %.0f\n",
                b[1], b[2], b[3], b[4])
    end
end
