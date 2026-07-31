# =============================================================================
# Where does the negative RESIDUAL VAPOR come from?
#
# `rho_v = rho_t - rho_d - rho_c - rho_r` is the one water field with no
# prognostic equation and no bound anywhere in the model, so every error the four
# densities cannot absorb lands in it. STAGE 4 of
# reference/FINDINGS_NEGATIVE_WATER_ATTRIBUTION.md measured min_rho_v = -0.72 g/m^3
# in the shipped default over a full hour, and -1.85 with rho_c bounded.
#
# This sweeps the saved snapshots of a completed run - no model run of its own -
# and answers the questions that decide WHICH candidate to instrument next:
#
#   (1) WHEN does rho_v first go negative, and how does the minimum grow?
#       Linear in time  => a constant per-step deposit (the refit, or the
#                          positivity limiter acting every step).
#       Tracking the convection => a rate problem (the condensation bound).
#   (2) WHERE is it, in (r, z)?
#   (3) Does it COINCIDE with the condensate the positivity limiter had to lift?
#       The limiter raises rho_r (and rho_c under POSITIVITY=1) with no
#       compensating change in rho_t, so every gram it adds at a point is a gram
#       taken out of rho_v AT THAT POINT. If the two structures coincide, that
#       candidate is implicated; if they do not, it is ruled out.
#
# Requires `<outdir>/o01_exact.ref`, which every o01 run writes beside its output.
#
# Usage:
#   julia --project=. benchmarks/water_residual_diagnostic.jl [outdir]
# =============================================================================

using Scythe, Springsteel, CSV, DataFrames, Printf

outdir = length(ARGS) >= 1 ? ARGS[1] :
         joinpath(@__DIR__, "output", "o01_rainfall_quick_mc_rirk")

snaps = sort([(parse(Float64, replace(basename(f), "_physical.csv" => "")), f)
              for f in readdir(outdir; join = true) if endswith(f, "_physical.csv")];
             by = first)
isempty(snaps) && error("no *_physical.csv snapshots in $outdir")

df0 = CSV.read(snaps[1][2], DataFrame)
ncols = length(unique(df0.r))
kDim  = nrow(df0) ÷ ncols
@printf("%s: %d snapshots, %d columns x %d levels\n", outdir, length(snaps), ncols, kDim)

# Same basis as o01_rainfall.jl, so `reference_column` sees the grid the run used.
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

rho_dbar = Springsteel.ref_rho_d(ref)[:, 1]
rho_tbar = Springsteel.ref_rho_t(ref)[:, 1]
rho_cbar = Springsteel.ref_rho_c(ref)[:, 1]

# Slots 8/9 may hold Ooyama control variables rather than densities; the column NAMES say
# which, and the run's log supplies the variant and the bias. Reading them raw is silently
# wrong by a factor of two in the linear regime and invents negative water that is not there.
include(joinpath(@__DIR__, "common", "diagnostics.jl"))
const WTRANS = detect_transforms(outdir)

"""Totals from a perturbation snapshot: (rho_v, rho_c, rho_r), flat, z fastest."""
function water(df)
    rho_d = df.rho_d .+ repeat(rho_dbar, ncols)
    rho_t = df.rho_t .+ repeat(rho_tbar, ncols)
    rho_c, rho_r = mc_water(df, rho_cbar, ncols;
                            ctrans = WTRANS.ctrans, cmu = WTRANS.cmu,
                            rtrans = WTRANS.rtrans, rmu = WTRANS.rmu)
    return (rho_t .- rho_d .- rho_c .- rho_r), rho_c, rho_r
end

println("\n  time      min rho_v      max rho_v   |   at r [km]   z [km]  |  " *
        "rho_c there   rho_r there  |  min rho_c   min rho_r   [g/m^3]")
first_negative = NaN
series = Tuple{Float64,Float64}[]
for (t, path) in snaps
    df = CSV.read(path, DataFrame)
    rho_v, rho_c, rho_r = water(df)
    j = argmin(rho_v)
    vmin = rho_v[j]
    push!(series, (t, vmin))
    isnan(first_negative) && vmin < 0.0 && (global first_negative = t)
    @printf("%7.1f  %+11.4f  %11.4f   |  %8.2f  %7.3f  |  %10.4f  %12.4f  |  %9.4f  %10.4f\n",
            t, 1e3 * vmin, 1e3 * maximum(rho_v), df.r[j] / 1e3, df.z[j] / 1e3,
            1e3 * rho_c[j], 1e3 * rho_r[j], 1e3 * minimum(rho_c), 1e3 * minimum(rho_r))
end

@printf("\nfirst snapshot with rho_v < 0: %s\n",
        isnan(first_negative) ? "never" : "t = $(first_negative) s")

# Growth diagnosis. A constant per-step deposit is LINEAR in t; a rate problem that
# switches on with the convection is not. Fit a line over the negative tail and
# report how well it holds -- the same test that identified the refit deposit as
# linear (~2.4e-7 kg/m^3/step) in the 2026-07-26 CORRECTION.
neg = [(t, v) for (t, v) in series if v < 0.0]
if length(neg) >= 4
    tt = [p[1] for p in neg]; vv = [p[2] for p in neg]
    tbar = sum(tt) / length(tt); vbar = sum(vv) / length(vv)
    slope = sum((tt .- tbar) .* (vv .- vbar)) / sum((tt .- tbar) .^ 2)
    icept = vbar - slope * tbar
    pred = icept .+ slope .* tt
    ss_res = sum((vv .- pred) .^ 2); ss_tot = sum((vv .- vbar) .^ 2)
    @printf("linear fit over the negative tail (%d points): %.4e g/m^3/s, R^2 = %.4f\n",
            length(neg), 1e3 * slope, 1.0 - ss_res / ss_tot)
    println("  R^2 near 1 => a constant per-step deposit (refit / limiter);" *
            " well below 1 => a rate that tracks the storm.")
end

# Coincidence test. If the limiter is the source, the vapor deficit sits where the
# condensate was lifted -- i.e. where the UNBOUNDED field would have been negative
# and the bounded one reads exactly 0.
dfN = CSV.read(snaps[end][2], DataFrame)
rho_v, rho_c, rho_r = water(dfN)
neg_v = rho_v .< 0.0
@printf("\nfinal snapshot: %d of %d points have rho_v < 0 (%.2f %%)\n",
        count(neg_v), length(rho_v), 100 * count(neg_v) / length(rho_v))
if count(neg_v) > 0
    @printf("  of those, rho_r == 0 exactly (limiter-flattened) at %d (%.1f %%)\n",
            count(neg_v .& (rho_r .== 0.0)), 100 * count(neg_v .& (rho_r .== 0.0)) / count(neg_v))
    @printf("  of those, rho_c <= 0 at %d (%.1f %%)\n",
            count(neg_v .& (rho_c .<= 0.0)), 100 * count(neg_v .& (rho_c .<= 0.0)) / count(neg_v))
    @printf("  rho_v deficit total = %.6e kg/m^3 summed over points\n", -sum(rho_v[neg_v]))
    @printf("  negative rho_c total = %.6e kg/m^3 summed over points\n",
            -sum(rho_c[rho_c .< 0.0]))
    zneg = dfN.z[neg_v]; rneg = dfN.r[neg_v]
    @printf("  z range %.2f - %.2f km, r range %.1f - %.1f km\n",
            minimum(zneg) / 1e3, maximum(zneg) / 1e3, minimum(rneg) / 1e3, maximum(rneg) / 1e3)
end

# ── Is the deficit in the PARTITION or in the total water? ────────────────────
# rho_v = (rho_t - rho_d) - rho_c - rho_r. Negative condensate RAISES rho_v, so
# wherever rho_v < 0 AND rho_c <= 0 the deficit must already be in the total water
# rho_w = rho_t - rho_d -- a difference of two large, nearly equal fitted fields
# whose true difference is O(1e-5) in the upper troposphere. That is the same
# catastrophic-cancellation pathology the equation-set note describes for the OLD
# diagnosed-rho_c formulation, relocated: it is not a microphysics or limiter
# problem at all, and neither the depletion bounds nor the positivity limiter can
# touch it.
rho_w = dfN.rho_t .- dfN.rho_d .+ repeat(rho_tbar .- rho_dbar, ncols)
negw = rho_w .< 0.0
@printf("\ntotal water rho_w = rho_t - rho_d: min = %+.4f g/m^3, %d points < 0 (%.2f %%)\n",
        1e3 * minimum(rho_w), count(negw), 100 * count(negw) / length(rho_w))
if count(neg_v) > 0
    @printf("  of the %d rho_v < 0 points, rho_w < 0 at %d (%.1f %%)\n",
            count(neg_v), count(neg_v .& negw), 100 * count(neg_v .& negw) / count(neg_v))
end
# The decomposition AT THE WORST VAPOR POINT, which is the quantity of interest and is
# NOT in general the worst rho_w point: rho_w < 0 is usually just the shadow of a
# negative rho_c (the same spline undershoot appears in rho_t, which carries the same
# cloud water), and there the two cancel and the vapor is fine. The vapor fails where
# rho_c + rho_r EXCEEDS a healthy rho_w.
jv = argmin(rho_v)
kv = mod1(jv, kDim)
@printf("\nworst rho_v point: r = %.2f km, z = %.3f km\n", dfN.r[jv] / 1e3, dfN.z[jv] / 1e3)
@printf("  reference vapor rho_vbar = %.6e kg/m^3\n", rho_tbar[kv] - rho_dbar[kv])
@printf("  rho_w = %+.6e   rho_c = %+.6e   rho_r = %+.6e   =>  rho_v = %+.6e\n",
        rho_w[jv], rho_c[jv], rho_r[jv], rho_v[jv])
@printf("  condensate/total-water ratio (rho_c+rho_r)/rho_w = %+.4f\n",
        (rho_c[jv] + rho_r[jv]) / rho_w[jv])

j = argmin(rho_w)
k = mod1(j, kDim)
@printf("\nworst rho_w point: r = %.2f km, z = %.3f km\n", dfN.r[j] / 1e3, dfN.z[j] / 1e3)
@printf("  reference   : rho_tbar = %.6e  rho_dbar = %.6e  rho_vbar = %.6e kg/m^3\n",
        rho_tbar[k], rho_dbar[k], rho_tbar[k] - rho_dbar[k])
@printf("  perturbation: rho_t' = %+.6e  rho_d' = %+.6e  (difference %+.6e)\n",
        dfN.rho_t[j], dfN.rho_d[j], dfN.rho_t[j] - dfN.rho_d[j])
@printf("  totals      : rho_w = %+.6e  rho_c = %+.6e  rho_r = %+.6e  rho_v = %+.6e\n",
        rho_w[j], rho_c[j], rho_r[j], rho_v[j])
@printf("  the perturbations are %.1fx the reference water they must not overshoot\n",
        abs(dfN.rho_t[j]) / max(rho_tbar[k] - rho_dbar[k], eps()))

# ── Does the deficit scale with the LARGE fields or with the water? ───────────
# The prediction if this is truncation error differenced out of two large nearly
# equal transports: the discrepancy tracks |rho_t'| (the field whose high
# derivatives set the truncation), NOT the local water content. If instead it
# tracked the water, a water-side rate or limiter would be implicated.
resid = dfN.rho_t .- dfN.rho_d                # = rho_v' + rho_c' + rho_r' (perturbation water)
scale = abs.(dfN.rho_t)
water_ref = repeat(rho_tbar .- rho_dbar, ncols)
function pearson(a, b)
    abar = sum(a) / length(a); bbar = sum(b) / length(b)
    return sum((a .- abar) .* (b .- bbar)) /
           sqrt(sum((a .- abar) .^ 2) * sum((b .- bbar) .^ 2))
end
if count(negw) > 2
    d = -rho_w[negw]                          # the deficit, positive
    @printf("\nover the %d rho_w < 0 points:\n", count(negw))
    @printf("  corr(deficit, |rho_t'|)      = %+.4f\n", pearson(d, scale[negw]))
    @printf("  corr(deficit, reference water) = %+.4f\n", pearson(d, water_ref[negw]))
    @printf("  median deficit / |rho_t'|      = %.4e\n",
            sum(d ./ max.(scale[negw], eps())) / count(negw))
end
