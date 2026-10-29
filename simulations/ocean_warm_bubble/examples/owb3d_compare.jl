#!/usr/bin/env julia
# Compare the 3-D RLR warm-bubble smoke against its axisymmetric control (Stream C1).
#
#   julia --project=<scythe> owb3d_compare.jl <rlr_root> <axisym_root> [<rlr_2worker_root>]
#       [--gate name=value ...] [--strict-until <s>]
#
# Each root is a run's output.dir (the `<t>_physical.csv` / `<t>_spectral.csv` files are read
# from its seg_01/). Both runs need output.formats to include "csv" (rlr_smoke.toml and
# axisym_control.toml do). Per output time it reports, for every prognostic slot:
#   * ring statistics of the RLR run: per (r, z) ring the mean, min and max over azimuth;
#     the SYMMETRY SPREAD is max over rings of (max - min), scaled by the slot's max |.|;
#     (v, identically 0 in the exact answer, is scaled by the wind max(|u|, |w|) throughout);
#   * the WN>=1 ENERGY FRACTION from `<t>_spectral.csv`: sum of squared spectral coefficients
#     in the wavenumber >= 1 blocks over the sum over all blocks (the RLR layout, Springsteel
#     transforms_cylindrical.jl: per vertical B-spline mode a WN0 radial block, then real and
#     imaginary radial blocks per wavenumber);
#   * the WN0 ERROR vs the axisym run: the ring mean minus the axisym value at the same
#     (r, z) mish node (the nodes are bitwise identical), as a scaled pointwise max and as a
#     relative L2 over the domain;
# plus max |v|, max w, rain onset (first output time with surface rho_r > ONSET_RHO_R), peak
# surface rho_r and the max rain path, for both runs; and, with a third root, the 2-worker vs
# 1-worker RLR difference. Prints a PASS/FAIL table per output time and a summary; see
# STRICT_UNTIL below for which gates decide PASS/FAIL when.
#
# rho_r is recovered from the nu_r slot with Scythe.recover_rho_r(slot, transform, mu) using
# the run's own physics.microphysics.rain_transform / rain_mu (read from the resolved
# <root>/scythe_config.toml).

using Scythe
using CSV
using DataFrames
using Printf

# ── Gates (script parameters; override with --gate name=value) ─────────────────────────────
const GATES = Dict{String,Float64}(
    "spread_max"        => 1.0e-6,   # scaled ring spread of w, rho_r slot, E_t (every time)
    "wn1_energy_max"    => 1.0e-20,  # WN>=1 energy fraction of w, rho_r slot, E_t (every time)
    "vmax"              => 1.0e-9,   # max |v| [m/s] (every time)
    "wn0_max"           => 1.0e-8,   # WN0 vs axisym, scaled pointwise max (every strict time)
    "wn0_relL2"         => 1.0e-3,   # WN0 vs axisym relative L2 (every strict time; provisional)
    "bulk_rel"          => 0.02,     # max w, onset, peak surface rain: relative agreement
    "workers_max"       => 1.0e-12,  # 2-worker vs 1-worker RLR, scaled max (every time)
    "scale_floor"       => 1.0e-10,  # denominator floor for a slot that is ~0 everywhere
    "onset_rho_r"       => 1.0e-5,   # [kg/m^3] surface rho_r that defines rain onset
)
# STRICT window (--strict-until <s>, default: the whole run). The ROUNDOFF gates -- spread,
# WN>=1 energy, max|v|, WN0 vs axisym (pointwise and rel L2) and 2-worker vs 1-worker -- are
# STRICT (they decide PASS/FAIL) at output times t <= strict_until and INFORMATIONAL after it
# (printed as [info ...], never failing). The ice-on pair is run with --strict-until 1500
# (pre-glaciation): ISHMAEL amplifies roundoff to O(1) within ~600 s of glaciation, in 2-D as
# well (reference/session_2026-10_scratch/closeout_2026-10-27/r1_rlr/README.md).
# SOLUTION-LEVEL gates: finite (every time), run-max w and rain onset within bulk_rel are
# always strict; peak surface rho_r within bulk_rel is strict only when the strict window
# covers the whole run, informational otherwise.
const STRICT_UNTIL = Ref(Inf)

# The slots the spread / WN>=1 gates are applied to (the plan's w, rho_r, E_t'); every slot
# is REPORTED. The WN0 gates apply to every slot except v (gated by vmax).
const SYM_GATE_VARS = ["w", "nu_r", "rho_r", "E_t"]

function parse_args(args)
    roots = String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--gate"
            k, v = split(args[i + 1], '='; limit = 2)
            haskey(GATES, k) || error("unknown gate $(k); known: $(join(sort(collect(keys(GATES))), ", "))")
            GATES[k] = parse(Float64, v)
            i += 2
        elseif a == "--strict-until"
            STRICT_UNTIL[] = parse(Float64, args[i + 1])
            i += 2
        else
            push!(roots, a)
            i += 1
        end
    end
    2 <= length(roots) <= 3 || error("usage: owb3d_compare.jl <rlr_root> <axisym_root> [<rlr_2worker_root>] [--gate k=v ...] [--strict-until s]")
    return roots
end

segdir(root) = joinpath(root, "seg_01")
function output_times(root)
    fs = filter(f -> endswith(f, "_physical.csv"), readdir(segdir(root)))
    return sort([parse(Float64, replace(f, "_physical.csv" => "")) for f in fs])
end
function tagfile(root, t, kind)
    for f in readdir(segdir(root))
        endswith(f, "_$(kind).csv") || continue
        parse(Float64, replace(f, "_$(kind).csv" => "")) == t && return joinpath(segdir(root), f)
    end
    error("no $(kind) CSV for t = $t in $(segdir(root))")
end

"Prognostic slot names, rain transform and mu, and the B-spline sizes of a run."
function run_info(root)
    cfg = load_config(joinpath(root, "scythe_config.toml"); set = ["output.dir=$(root)"])
    model = build_model(cfg)
    vars = first.(sort(collect(model.grid_params.vars), by = last))
    rtrans = Scythe.rain_transform_mode(model.options)
    rmu = get(model.physical_params, :rain_mu, 1.0e-7)
    return (cfg = cfg, vars = vars, rtrans = rtrans, rmu = rmu,
            rslot = Scythe.rain_var_name(model.options),
            b_iDim = sum(cfg.grid.num_cells) + 3, b_kDim = cfg.grid.num_cells_k + 3)
end

read_phys(path, vars, coords) = CSV.read(path, DataFrame; select = [coords; vars])

"Sorted (r, z) ring keys, and per var the ring (mean, min, max) vectors over azimuth."
function ring_stats(df, vars)
    idx = Dict{Tuple{Float64,Float64},Vector{Int}}()
    for (i, (r, z)) in enumerate(zip(df.r, df.z))
        push!(get!(idx, (r, z), Int[]), i)
    end
    keys_ = sort(collect(keys(idx)))
    st = Dict{String,NamedTuple}()
    for v in vars
        col = df[!, v]
        nk = length(keys_)
        mean_, min_, max_ = zeros(nk), zeros(nk), zeros(nk)
        for (n, k) in enumerate(keys_)
            x = view(col, idx[k])
            mean_[n] = sum(x) / length(x)
            min_[n] = minimum(x)
            max_[n] = maximum(x)
        end
        st[v] = (mean = mean_, min = min_, max = max_)
    end
    return keys_, st
end

"WN>=1 share of the squared spectral coefficients, per var (RLR layout)."
function wn1_fraction(path, vars, b_iDim, b_kDim)
    df = CSV.read(path, DataFrame; select = vars)
    n = nrow(df)
    stride = n ÷ b_kDim
    (n % b_kDim == 0 && stride % b_iDim == 0 && isodd(stride ÷ b_iDim)) ||
        error("spectral CSV $(path): $(n) rows do not fit the RLR layout " *
              "(b_kDim = $(b_kDim), b_iDim = $(b_iDim))")
    out = Dict{String,Float64}()
    for v in vars
        c = df[!, v]
        e0 = 0.0
        e1 = 0.0
        for zb in 1:b_kDim
            base = (zb - 1) * stride
            for j in 1:stride
                x = c[base + j]^2
                j <= b_iDim ? (e0 += x) : (e1 += x)
            end
        end
        out[v] = (e0 + e1) > 0 ? e1 / (e0 + e1) : 0.0
    end
    return out
end

"Surface rho_r max, column rain path max [kg/m^2] from (r, z) fields of rho_r."
function rain_metrics(r, z, rho_r)
    zs = sort(unique(z))
    rs = sort(unique(r))
    zi = Dict(x => i for (i, x) in enumerate(zs))
    ri = Dict(x => i for (i, x) in enumerate(rs))
    F = zeros(length(rs), length(zs))
    for (a, b, x) in zip(r, z, rho_r)
        F[ri[a], zi[b]] = x
    end
    surf = maximum(F[:, 1])
    # Trapezoid over the mish z nodes (the column's own sample points; not the spline
    # quadrature, so a path is comparable between the two runs, not an exact integral).
    path = maximum(sum(0.5 * (F[i, k] + F[i, k + 1]) * (zs[k + 1] - zs[k])
                       for k in 1:length(zs) - 1) for i in 1:length(rs))
    return surf, path
end

relerr(a, b) = abs(a - b) / max(abs(a), abs(b), 1.0e-30)
fmt(x) = x isa AbstractFloat ? @sprintf("%.3e", x) : string(x)
pf(ok) = ok ? "PASS" : "FAIL"
tag(ok, gating) = gating ? pf(ok) : (ok ? "info pass" : "info exceeds")

function main(args)
    roots = parse_args(args)
    rlr_root, ax_root = roots[1], roots[2]
    w2_root = length(roots) > 2 ? last(roots) : nothing
    ri = run_info(rlr_root)
    ai = run_info(ax_root)
    ri.vars == ai.vars || error("slot lists differ: $(ri.vars) vs $(ai.vars)")
    vars = ri.vars
    rslot = ri.rslot
    times = intersect(output_times(rlr_root), output_times(ax_root))
    w2_root === nothing || (times = intersect(times, output_times(w2_root)))
    floor_ = GATES["scale_floor"]
    symvars = filter(v -> v in SYM_GATE_VARS, vars)
    wn0vars = filter(!=("v"), vars)

    println("="^100)
    println("owb3d_compare: RLR $(rlr_root)\n               axisym $(ax_root)" *
            (w2_root === nothing ? "" : "\n               RLR 2-worker $(w2_root)"))
    println("gates: ", join(("$k=$(GATES[k])" for k in sort(collect(keys(GATES)))), "  "))
    println("rain slot $(rslot) (transform $(ri.rtrans), mu $(ri.rmu)); symmetry gates on " *
            "$(symvars); WN0 gates on every slot but v")
    println("="^100)

    bulk = Dict(:rlr => Dict(:maxw => -Inf, :onset => NaN, :psurf => 0.0, :path => 0.0),
                :ax => Dict(:maxw => -Inf, :onset => NaN, :psurf => 0.0, :path => 0.0))
    history = []
    allpass = true
    for t in times
        drl = read_phys(tagfile(rlr_root, t, "physical"), vars, ["r", "l", "z"])
        dax = read_phys(tagfile(ax_root, t, "physical"), vars, ["r", "z"])
        all(isfinite, Matrix(drl[:, vars])) || println("  !! non-finite values in the RLR run at t = $t")
        all(isfinite, Matrix(dax[:, vars])) || println("  !! non-finite values in the axisym run at t = $t")
        fin = all(isfinite, Matrix(drl[:, vars])) && all(isfinite, Matrix(dax[:, vars]))
        keys_, st = ring_stats(drl, vars)
        axidx = Dict((r, z) => i for (i, (r, z)) in enumerate(zip(dax.r, dax.z)))
        length(keys_) == nrow(dax) && all(k -> haskey(axidx, k), keys_) ||
            error("RLR rings and axisym nodes do not match at t = $t")
        perm = [axidx[k] for k in keys_]
        wn1 = wn1_fraction(tagfile(rlr_root, t, "spectral"), vars, ri.b_iDim, ri.b_kDim)

        # v is identically 0 in the exact answer, so its own max is roundoff: v is scaled by
        # the wind (max |u|, |w| of the RLR run), where the roundoff it carries comes from.
        wind = max(maximum(abs, drl.u), maximum(abs, drl.w), floor_)
        rows = []
        for v in vars
            axv = dax[!, v][perm]
            sc = v == "v" ? wind : max(maximum(abs, drl[!, v]), maximum(abs, axv), floor_)
            sp = st[v].max .- st[v].min
            ispr = argmax(sp)
            d = st[v].mean .- axv
            iw = argmax(abs.(d))
            l2 = sqrt(sum(abs2, d)) / max(sqrt(sum(abs2, axv)), floor_ * sqrt(length(axv)))
            push!(rows, (v = v, scale = sc, spread = sp[ispr] / sc, spread_at = keys_[ispr],
                         wn1 = wn1[v], wn0 = abs(d[iw]) / sc, wn0_at = keys_[iw], relL2 = l2))
        end
        vmax = maximum(abs, drl.v)
        vmax_ax = maximum(abs, dax.v)

        # bulk: max w, surface rain, rain path (RLR: from ring means)
        rr_rl = Scythe.recover_rho_r.(st[rslot].mean, ri.rtrans, ri.rmu)
        rr_ax = Scythe.recover_rho_r.(dax[!, rslot][perm], ai.rtrans, ai.rmu)
        rk = first.(keys_); zk = last.(keys_)
        s_rl, p_rl = rain_metrics(rk, zk, rr_rl)
        s_ax, p_ax = rain_metrics(rk, zk, rr_ax)
        for (who, mw, s, p) in ((:rlr, maximum(drl.w), s_rl, p_rl), (:ax, maximum(dax.w), s_ax, p_ax))
            b = bulk[who]
            b[:maxw] = max(b[:maxw], mw)
            b[:psurf] = max(b[:psurf], s)
            b[:path] = max(b[:path], p)
            isnan(b[:onset]) && s > GATES["onset_rho_r"] && (b[:onset] = t)
        end

        w12 = NaN
        if w2_root !== nothing
            d2 = read_phys(tagfile(w2_root, t, "physical"), vars, ["r", "l", "z"])
            (d2.r == drl.r && d2.l == drl.l && d2.z == drl.z) || error("2-worker grid differs")
            w12 = maximum(maximum(abs, drl[!, v] .- d2[!, v]) /
                          (v == "v" ? wind : max(maximum(abs, drl[!, v]), floor_)) for v in vars)
        end

        g = Dict(r.v => r for r in rows)
        spread_g = maximum(g[v].spread for v in symvars)
        wn1_g = maximum(g[v].wn1 for v in symvars)
        wn0_g = maximum(g[v].wn0 for v in wn0vars)
        l2_g = maximum(g[v].relL2 for v in wn0vars)
        strict = t <= STRICT_UNTIL[]
        checks = [("finite (all slots, ice incl.)", fin, "", true),
                  ("spread(w, rho_r, E_t) <= $(GATES["spread_max"])", spread_g <= GATES["spread_max"], fmt(spread_g), strict),
                  ("WN>=1 energy (w, rho_r, E_t) <= $(GATES["wn1_energy_max"])", wn1_g <= GATES["wn1_energy_max"], fmt(wn1_g), strict),
                  ("max|v| <= $(GATES["vmax"])", vmax <= GATES["vmax"], fmt(vmax), strict),
                  ("WN0 vs axisym scaled max <= $(GATES["wn0_max"])", wn0_g <= GATES["wn0_max"], fmt(wn0_g), strict),
                  ("WN0 vs axisym relL2 <= $(GATES["wn0_relL2"])", l2_g <= GATES["wn0_relL2"], fmt(l2_g), strict)]
        w2_root === nothing ||
            push!(checks, ("2-worker vs 1-worker <= $(GATES["workers_max"])", w12 <= GATES["workers_max"], fmt(w12), strict))

        println("\n── t = $(t) s ", "─"^80)
        @printf("  %-8s %10s %10s %16s %10s %10s %16s %10s\n", "slot", "scale", "spread",
                "spread@(r,z)", "WN>=1 E", "WN0 max", "WN0@(r,z)", "WN0 relL2")
        for r in rows
            @printf("  %-8s %10.3e %10.3e %16s %10.3e %10.3e %16s %10.3e\n", r.v, r.scale,
                    r.spread, @sprintf("(%.0f,%.0f)", r.spread_at...), r.wn1, r.wn0,
                    @sprintf("(%.0f,%.0f)", r.wn0_at...), r.relL2)
        end
        @printf("  max|v| RLR %.3e  axisym %.3e;  max w RLR %.4f axisym %.4f;  surface rho_r RLR %.4e axisym %.4e;  rain path RLR %.4e axisym %.4e\n",
                vmax, vmax_ax, maximum(drl.w), maximum(dax.w), s_rl, s_ax, p_rl, p_ax)
        @printf("  WN0 vs axisym (all slots but v): scaled max %.3e, max relL2 %.3e\n", wn0_g, l2_g)
        strict || println("  (t > strict_until = $(STRICT_UNTIL[]) s: roundoff gates INFORMATIONAL)")
        for (name, ok, val, gating) in checks
            @printf("  [%s] %-48s %s\n", tag(ok, gating), name, val)
            gating && (allpass &= ok)
        end
        push!(history, (t = t, wn0 = wn0_g, relL2 = l2_g, spread = spread_g, wn1 = wn1_g,
                        vmax = vmax, w12 = w12))
    end

    # final-time gates
    t_end = times[end]
    h_end = history[end]
    br, ba = bulk[:rlr], bulk[:ax]
    onset_ok = (isnan(br[:onset]) && isnan(ba[:onset])) ||
               (!isnan(br[:onset]) && !isnan(ba[:onset]) &&
                relerr(br[:onset], ba[:onset]) <= GATES["bulk_rel"])
    whole = t_end <= STRICT_UNTIL[]
    final = [("WN0 vs axisym relL2 at t=$(t_end) <= $(GATES["wn0_relL2"]) (provisional)",
              h_end.relL2 <= GATES["wn0_relL2"], fmt(h_end.relL2), whole),
             ("run-max w within $(GATES["bulk_rel"])", relerr(br[:maxw], ba[:maxw]) <= GATES["bulk_rel"],
              "RLR $(fmt(br[:maxw])) ax $(fmt(ba[:maxw])) rel $(fmt(relerr(br[:maxw], ba[:maxw])))", true),
             ("rain onset within $(GATES["bulk_rel"])", onset_ok, "RLR $(br[:onset]) ax $(ba[:onset])", true),
             ("peak surface rho_r within $(GATES["bulk_rel"])", relerr(br[:psurf], ba[:psurf]) <= GATES["bulk_rel"],
              "RLR $(fmt(br[:psurf])) ax $(fmt(ba[:psurf])) rel $(fmt(relerr(br[:psurf], ba[:psurf])))", whole),
             ("max rain path within $(GATES["bulk_rel"])", relerr(br[:path], ba[:path]) <= GATES["bulk_rel"],
              "RLR $(fmt(br[:path])) ax $(fmt(ba[:path])) rel $(fmt(relerr(br[:path], ba[:path])))", false)]
    println("\n", "="^100)
    println("WN0-vs-axisym growth (all slots but v)   and symmetry history")
    @printf("  %8s %12s %12s %12s %12s %12s %12s\n", "t [s]", "WN0 max", "WN0 relL2", "spread", "WN>=1 E", "max|v|", "2w vs 1w")
    for h in history
        @printf("  %8.1f %12.3e %12.3e %12.3e %12.3e %12.3e %12.3e\n", h.t, h.wn0, h.relL2, h.spread, h.wn1, h.vmax, h.w12)
    end
    @printf("  max rain path RLR %.4e axisym %.4e kg/m^2\n", br[:path], ba[:path])
    println("final gates (strict window t <= $(STRICT_UNTIL[]) s; [PASS]/[FAIL] gate, [info ...] is recorded only):")
    for (name, ok, val, gating) in final
        @printf("  [%s] %-60s %s\n", tag(ok, gating), name, val)
        gating && (allpass &= ok)
    end
    println("\nOVERALL: ", pf(allpass))
    return allpass
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS) || exit(1)
end
