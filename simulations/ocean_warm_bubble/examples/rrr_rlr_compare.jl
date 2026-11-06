#!/usr/bin/env julia
# RRR box vs RLR cylinder, solution level (close-out Stream C2, stage R2e).
#
#   julia --project=<scythe> rrr_rlr_compare.jl <rrr_root> <rlr_root> [--gate k=v ...]
#       [--strict-until <s>] [--radius <m>]
#
# <rrr_root>, <rlr_root>: the output.dir of rrr_120km.toml and rlr_60km.toml runs (each
# holds scythe_config.toml and seg_01/<t>.nc, the comprehensive files). The same bubble is
# centred in the box and on the axis of the cylinder, so inside r < R/2 of the bubble centre
# (R = the cylinder radius; --radius overrides R/2) the two solutions should agree at the
# solution level. The discretizations differ (Cartesian x-y mish vs polar r-lambda mish), so
# this is NOT a roundoff comparison: the gates are relative agreement of
#
#   max w          run-max of w over r < R/2 up to t                       [m/s]
#   max T'         max of T - Tbar(z) over r < R/2 AT t (a run-max would be the initial 3 K) [K]
#   rain onset     first output time with surface rho_r > onset_rho_r over r < R/2   [s]
#
# Also printed per output time (informational): the rel L2 of w and T' along the x-axis row
# through the bubble centre for r < R/2 (the RRR row nearest y = yc against the RLR ring
# mean interpolated to the same radii), and the box's four vertical edges at the lowest level
# (min T', min/max p') beside the whole box's min T' -- the edge instability of the
# w-Dirichlet walls (STATUS.md, R2c) shows there first.
#
# at t = 1200 s (gate early_rel, 2 %) and t = 3600 s (gate late_rel, 5 %). The first passing
# run seeds the README numbers. With ice on, ISHMAEL amplifies roundoff by ~1e10 within
# ~10 min of glaciation (C1, r1_rlr/README.md), so a check past glaciation is a statement
# about the chaotic solution, not about the 3-D code: `--strict-until <s>` makes every
# check at a time > s informational ([info pass] / [info exceeds]) instead of a failure, as
# in owb3d_compare.jl. Exit status 0 = every strict check passes.

using NCDatasets
using TOML
using Printf

const GATES = Dict{String,Float64}(
    "early_t"     => 1200.0,   # [s]
    "early_rel"   => 0.02,
    "late_t"      => 3600.0,   # [s]
    "late_rel"    => 0.05,
    "onset_rho_r" => 1.0e-5)   # [kg/m^3] surface rho_r that defines rain onset

const STRICT_UNTIL = Ref(Inf)
const RADIUS = Ref(NaN)

function parse_args(args)
    pos = String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--gate"
            k, v = split(args[i + 1], "="; limit = 2)
            haskey(GATES, k) || error("unknown gate $k; gates: $(join(sort(collect(keys(GATES))), ", "))")
            GATES[k] = parse(Float64, v); i += 2
        elseif a == "--strict-until"
            STRICT_UNTIL[] = parse(Float64, args[i + 1]); i += 2
        elseif a == "--radius"
            RADIUS[] = parse(Float64, args[i + 1]); i += 2
        else
            push!(pos, a); i += 1
        end
    end
    length(pos) == 2 || error("usage: rrr_rlr_compare.jl <rrr_root> <rlr_root> [--gate k=v] [--strict-until s] [--radius m]")
    return pos[1], pos[2]
end

seg(root) = joinpath(root, "seg_01")
function output_times(root)
    ts = Float64[]
    for f in readdir(seg(root))
        m = match(r"^([0-9.]+)\.nc$", f)
        m === nothing || push!(ts, parse(Float64, m.captures[1]))
    end
    return sort(ts)
end
ncfile(root, t) = joinpath(seg(root), "$(t).nc")

"Per-time (max w, max T', max surface rho_r) over the points within `rmax` of the centre."
function region_stats(path, geom, xc, yc, rmax)
    NCDataset(path, "r") do ds
        ds.attrib["scythe_file_kind"] == "comprehensive" ||
            error("$path is not a comprehensive file (scythe_file_kind = $(ds.attrib["scythe_file_kind"]))")
        w = Array(ds["w"])[1, :, :, :]
        T = Array(ds["T"])[1, :, :, :]
        rr = Array(ds["rho_r"])[1, :, :, :]
        Tbar = Array(ds["Tbar"])
        if geom == "RRR"
            x = Array(ds["x"]); y = Array(ds["y"])
            inside = [hypot(x[i] - xc, y[j] - yc) < rmax for i in eachindex(x), j in eachindex(y)]
        else
            r = Array(ds["radius"]); nj = ds.dim["azimuth"]
            inside = [r[i] < rmax for i in eachindex(r), j in 1:nj]
        end
        mw = -Inf; mT = -Inf; ms = 0.0
        for i in axes(w, 1), j in axes(w, 2)
            inside[i, j] || continue
            for k in axes(w, 3)
                mw = max(mw, w[i, j, k])
                mT = max(mT, T[i, j, k] - Tbar[k])
            end
            ms = max(ms, rr[i, j, 1])
        end
        return (; maxw = mw, maxT = mT, sfc_rho_r = ms)
    end
end

function centre(root)
    c = TOML.parsefile(joinpath(root, "scythe_config.toml"))
    g = c["grid"]["geometry"]
    b = c["initial_conditions"]["temperature_bubble"]
    return g, Float64(b["xc"]), Float64(get(b, "yc", 0.0)), Float64(c["grid"]["boundaries"][end])
end

relerr(a, b) = abs(a - b) / max(abs(a), abs(b), eps())

"Linear interpolation of the columns of `f` (rows on the abscissa `r`) to the points `rq`."
function interp_rows(r, f, rq)
    out = zeros(length(rq), size(f, 2))
    for (n, x) in enumerate(rq)
        i = clamp(searchsortedlast(r, x), 1, length(r) - 1)
        a = (x - r[i]) / (r[i + 1] - r[i])
        out[n, :] .= (1 - a) .* f[i, :] .+ a .* f[i + 1, :]
    end
    return out
end

"""
rel L2 of w and T' along the x-axis row through the bubble centre, x - xc in [0, rmax):
the RRR regular-grid row nearest y = yc against the RLR ring mean (azimuthal average)
interpolated linearly in r to the same radii, all heights.
"""
function row_relL2(prrr, prlr, xc, yc, rmax)
    rowv(ds) = begin
        x = Array(ds["x"]); y = Array(ds["y"]); j = argmin(abs.(y .- yc))
        sel = findall(xx -> 0.0 <= xx - xc < rmax, x)
        T = Array(ds["T"])[1, :, j, :]; Tb = Array(ds["Tbar"])
        (x[sel] .- xc, Array(ds["w"])[1, sel, j, :], T[sel, :] .- Tb', abs(y[j] - yc))
    end
    rq, wa, Ta, dy = NCDataset(rowv, prrr, "r")
    wb, Tb2 = NCDataset(prlr, "r") do ds
        r = Array(ds["radius"])
        T = Array(ds["T"])[1, :, :, :]; Tb = Array(ds["Tbar"])
        wm = dropdims(sum(Array(ds["w"])[1, :, :, :]; dims = 2); dims = 2) ./ ds.dim["azimuth"]
        Tm = dropdims(sum(T; dims = 2); dims = 2) ./ ds.dim["azimuth"] .- Tb'
        (interp_rows(r, wm, rq), interp_rows(r, Tm, rq))
    end
    l2(a, b) = sqrt(sum(abs2, a .- b) / max(sum(abs2, b), eps()))
    return (; w = l2(wa, wb), T = l2(Ta, Tb2), dy = dy)
end

"Per-time min T' and min/max p' over the four vertical edges of the box at the lowest level."
function corner_stats(path)
    NCDataset(path, "r") do ds
        T = Array(ds["T"])[1, :, :, :]; Tb = Array(ds["Tbar"]); p = Array(ds["p_prime"])[1, :, :, :]
        ni, nj = size(T, 1), size(T, 2)
        cs = ((1, 1), (1, nj), (ni, 1), (ni, nj))
        Tp = [T[i, j, 1] - Tb[1] for (i, j) in cs]
        pp = [p[i, j, 1] for (i, j) in cs]
        Tall = T .- reshape(Tb, 1, 1, :)
        return (; minT = minimum(Tp), minp = minimum(pp), maxp = maximum(pp), domain_minT = minimum(Tall))
    end
end

function main(args)
    rrr, rlr = parse_args(args)
    g1, xc, yc, _ = centre(rrr)
    g2, _, _, R = centre(rlr)
    (g1 == "RRR" && g2 == "RLR") || error("expected an RRR run and an RLR run, got $g1 and $g2")
    rmax = isnan(RADIUS[]) ? R / 2 : RADIUS[]
    times = intersect(output_times(rrr), output_times(rlr))
    isempty(times) && error("no common output times")
    println("RRR $rrr\nRLR $rlr\nregion: r < $(rmax / 1e3) km of the bubble centre ($(xc / 1e3), $(yc / 1e3)) km; " *
            "strict until $(STRICT_UNTIL[]) s")
    run = Dict(:rrr => Dict(:maxw => -Inf, :maxT => -Inf, :onset => NaN),
               :rlr => Dict(:maxw => -Inf, :maxT => -Inf, :onset => NaN))
    @printf("%8s | %10s %10s | %9s %9s | %12s %12s | %8s %8s | %8s %9s %9s %8s\n", "t [s]", "max w RRR", "RLR",
            "max T' RRR", "RLR", "sfc rho_r RRR", "RLR", "row w L2", "row T' L2", "edge T'", "edge minp'",
            "edge maxp'", "dom minT'")
    rows = Dict{Float64,Any}()
    rowl2 = Dict{Float64,Any}()
    for t in times
        a = region_stats(ncfile(rrr, t), "RRR", xc, yc, rmax)
        b = region_stats(ncfile(rlr, t), "RLR", 0.0, 0.0, rmax)
        for (k, s) in ((:rrr, a), (:rlr, b))
            run[k][:maxw] = max(run[k][:maxw], s.maxw)
            run[k][:maxT] = s.maxT        # at t: the run-max would be the initial bubble
            isnan(run[k][:onset]) && s.sfc_rho_r > GATES["onset_rho_r"] && (run[k][:onset] = t)
        end
        rows[t] = (deepcopy(run[:rrr]), deepcopy(run[:rlr]))
        rl = row_relL2(ncfile(rrr, t), ncfile(rlr, t), xc, yc, rmax)
        cs = corner_stats(ncfile(rrr, t))
        rowl2[t] = rl
        @printf("%8.0f | %10.4f %10.4f | %9.4f %9.4f | %12.3e %12.3e | %8.3g %8.3g | %8.3f %9.1f %9.1f %8.3f\n",
                t, a.maxw, b.maxw, a.maxT, b.maxT, a.sfc_rho_r, b.sfc_rho_r, rl.w, rl.T,
                cs.minT, cs.minp, cs.maxp, cs.domain_minT)
    end
    println("(row: the RRR row at |y - yc| = $(round(first(values(rowl2)).dy; digits = 1)) m vs the RLR ring mean; " *
            "edges: the four vertical box edges at the lowest level; dom min T': the whole box)")
    ok = true
    println("\nchecks (max w and onset: run values up to t; max T': at t; relative agreement):")
    for (tg, rel) in ((GATES["early_t"], GATES["early_rel"]), (GATES["late_t"], GATES["late_rel"]))
        haskey(rows, tg) || (println("  t = $tg s: not in the output (skipped)"); continue)
        x, y = rows[tg]
        strict = tg <= STRICT_UNTIL[]
        for (lbl, kx) in (("max w", :maxw), ("max T'", :maxT), ("rain onset", :onset))
            a, b = x[kx], y[kx]
            pass = kx === :onset ? ((isnan(a) && isnan(b)) || (!isnan(a) && !isnan(b) && relerr(a, b) <= rel)) :
                                   relerr(a, b) <= rel
            tag = strict ? (pass ? "PASS" : "FAIL") : (pass ? "info pass" : "info exceeds")
            strict && !pass && (ok = false)
            @printf("  t = %5.0f s  %-10s RRR %-12s RLR %-12s rel %-10s gate %.3g  [%s]\n", tg, lbl,
                    @sprintf("%.5g", a), @sprintf("%.5g", b),
                    (isnan(a) || isnan(b)) ? "-" : @sprintf("%.3g", relerr(a, b)), rel, tag)
        end
    end
    println(ok ? "\nOVERALL PASS" : "\nOVERALL FAIL")
    return ok
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS) ? 0 : 1)
end
