#!/usr/bin/env julia
# tc_bl_assess.jl — automated boundary-layer assessment from the comprehensive TC NetCDF.
#
# Reads every `<t>.nc` written directly by the model in `<indir>/nest<N>/` (one file per
# output time, `scythe_file_kind = "comprehensive"`; see `src/netcdf_output.jl` and
# `tc/MYNN_RUNS.md`) and prints (or writes as Markdown) the same tables that
# `tc/output/S10_BL_ASSESSMENT.md` was built by hand: an hourly nest table, a provenance
# block, and — with two or more `--indir`s — an options diff and a final-time comparison.
#
# This script reads NetCDF and log files only. It never runs the model.
#
# Usage:
#   julia --project=. tc/tc_bl_assess.jl --indir <run> [--indir <run> ...] \
#       [--nest 1] [--out <md path>] [--clamp-hist]
#
# With one --indir: single-run hourly table + provenance.
# With two+ --indirs: each run's table, an options/physical_params diff between the first
#   two runs' ModelParameters lines, and a final-time side-by-side comparison table.
# --out writes the report as Markdown to the given path; omitted, the report goes to stdout.
# --clamp-hist adds a per-height histogram of the fraction of points sitting at the file's
#   own min(e) * 1.05 — a cheap, approximate proxy for where the MYNN TKE floor clamp
#   (`mynn_n_clamp_e`) is accumulating. It is a proxy only: the census counter itself is
#   accumulated on the scheme's OWN mish over every internal sub-step, not on this
#   regridded, single-snapshot field (see the `mynn_regridding` attribute).

using NCDatasets
using Printf
using Statistics

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

# NCDatasets returns Union{Missing,Float64} arrays for variables with a _FillValue; turn
# missing into NaN so the rest of the script can use plain isnan/isfinite logic.
denan(x) = Float64.(coalesce.(x, NaN))

nanfinite(x) = isfinite(x)

function nanmax(A)
    m = -Inf
    idx = CartesianIndex{ndims(A)}(ntuple(_ -> 1, ndims(A)))
    found = false
    for I in CartesianIndices(A)
        v = A[I]
        if isfinite(v) && (!found || v > m)
            m = v; idx = I; found = true
        end
    end
    return found ? (m, idx) : (NaN, nothing)
end

function nanmin(A)
    m = Inf
    idx = CartesianIndex{ndims(A)}(ntuple(_ -> 1, ndims(A)))
    found = false
    for I in CartesianIndices(A)
        v = A[I]
        if isfinite(v) && (!found || v < m)
            m = v; idx = I; found = true
        end
    end
    return found ? (m, idx) : (NaN, nothing)
end

nanmean(x) = (v = filter(isfinite, x); isempty(v) ? NaN : mean(v))
allnan(x) = all(v -> !isfinite(v), x)

fmt(x; digits=2) = isfinite(x) ? @sprintf("%.*f", digits, x) : "n/a"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

function parse_args(argv)
    indirs = String[]
    nest = 1
    out = nothing
    clamp_hist = false
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--indir"
            i += 1
            push!(indirs, argv[i])
        elseif a == "--nest"
            i += 1
            nest = parse(Int, argv[i])
        elseif a == "--out"
            i += 1
            out = argv[i]
        elseif a == "--clamp-hist"
            clamp_hist = true
        else
            error("tc_bl_assess.jl: unrecognized argument '$a'. " *
                  "Usage: --indir <run> [--indir <run> ...] [--nest N] [--out path] [--clamp-hist]")
        end
        i += 1
    end
    isempty(indirs) && error("tc_bl_assess.jl: at least one --indir is required")
    return indirs, nest, out, clamp_hist
end

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

# Comprehensive per-time files are named exactly "<t>.nc" (t a plain float, e.g. "1800.0.nc").
# Opt-in sidecars are named "<t>_mynn_i<k>.nc" / "<t>_radiation_i<k>.nc" and must NOT match.
const NC_RE = r"^([0-9]+(?:\.[0-9]+)?)\.nc$"

function list_nc_files(nestdir)
    out = Tuple{Float64,String}[]
    isdir(nestdir) || return out
    for f in readdir(nestdir)
        m = match(NC_RE, f)
        m === nothing && continue
        push!(out, (parse(Float64, m.captures[1]), joinpath(nestdir, f)))
    end
    sort!(out, by = x -> x[1])
    return out
end

nestdir(indir, nest) = joinpath(indir, "nest$(nest)")

# ---------------------------------------------------------------------------
# Per-file statistics
# ---------------------------------------------------------------------------

struct FileStats
    t::Float64
    path::String
    physics_groups::String
    has_mynn::Bool
    has_surface::Bool
    max_v::Float64
    max_v_r_km::Float64
    max_v_z_km::Float64
    min_pp_hpa::Float64
    min_pp_r_km::Float64
    max_inflow::Float64
    max_inflow_r_km::Float64
    max_outflow::Float64
    max_outflow_r_km::Float64
    max_w::Float64
    max_w_r_km::Float64
    max_w_z_km::Float64
    max_refl::Float64          # NaN => n/a (all-NaN field)
    mean_precip::Float64
    max_PW::Float64
    max_Km::Float64
    max_Kh::Float64
    pblh_rmw::Float64
    rmw_r_km::Float64
    max_e::Float64
    max_Fsh::Float64
    max_Fq::Float64
    max_U10::Float64
    max_ust::Float64
    # last-file-only plume census (NaN elsewhere)
    n_clamp_e::Float64
    n_cap_K::Float64
    n_diffnum::Float64
    n_gate::Float64
    n_stall::Float64
    n_plume::Float64
end

# Attribute lookup that never throws; returns NaN (numeric) for a missing/non-numeric attr.
attrnum(ds, k) = haskey(ds.attrib, k) ? Float64(ds.attrib[k]) : NaN

function file_stats(path)
    ds = NCDataset(path)
    try
        physics_groups = get(ds.attrib, "physics_groups", "")
        has_mynn = occursin("mynn", physics_groups)
        has_surface = occursin("surface", physics_groups)

        t = Float64(ds["time"][1])
        x_km = denan(ds["x"][:]) ./ 1000.0
        z_km = denan(ds["z"][:]) ./ 1000.0
        z_m = denan(ds["z"][:])

        v = denan(ds["v"][1, :, :])          # (x, z)
        max_v, ivz = nanmax(v)
        max_v_r_km = ivz === nothing ? NaN : x_km[ivz[1]]
        max_v_z_km = ivz === nothing ? NaN : z_km[ivz[2]]

        pp = denan(ds["p_prime"][1, :, :])
        iz0 = argmin(z_m)                     # lowest output level (should be z = 0)
        pp_surf = pp[:, iz0]
        min_pp, ipp = nanmin(pp_surf)
        min_pp_hpa = isfinite(min_pp) ? min_pp / 100.0 : NaN
        min_pp_r_km = ipp === nothing ? NaN : x_km[ipp[1]]

        u = denan(ds["u"][1, :, :])
        low_mask = z_m .<= 2000.0
        hi_mask = z_m .>= 8000.0
        u_low = u[:, low_mask]
        u_hi = u[:, hi_mask]
        max_inflow, iin = nanmin(u_low)       # most negative u = strongest inflow
        max_inflow_r_km = iin === nothing ? NaN : x_km[iin[1]]
        max_outflow, iout = nanmax(u_hi)
        max_outflow_r_km = iout === nothing ? NaN : x_km[iout[1]]

        w = denan(ds["w"][1, :, :])
        max_w, iw = nanmax(w)
        max_w_r_km = iw === nothing ? NaN : x_km[iw[1]]
        max_w_z_km = iw === nothing ? NaN : z_km[iw[2]]

        refl = denan(ds["reflectivity"][1, :, :])
        max_refl = allnan(refl) ? NaN : nanmax(refl)[1]

        precip = denan(ds["precip_rate"][1, :])
        mean_precip = nanmean(precip)

        PW = denan(ds["PW"][1, :])
        max_PW = nanmax(PW)[1]

        max_Km = max_Kh = pblh_rmw = rmw_r_km = max_e = NaN
        if has_mynn
            Km = denan(ds["K_m"][1, :, :])
            Kh = denan(ds["K_h"][1, :, :])
            e = denan(ds["e"][1, :, :])
            max_Km = nanmax(Km)[1]
            max_Kh = nanmax(Kh)[1]
            max_e = nanmax(e)[1]
            v_lowlev = v[:, iz0]
            _, irmw = nanmax(v_lowlev)
            if irmw !== nothing
                rmw_r_km = x_km[irmw[1]]
                pblh = denan(ds["mynn_pblh"][1, :])
                pblh_rmw = pblh[irmw[1]]
            end
        end

        max_Fsh = max_Fq = max_U10 = max_ust = NaN
        if has_surface
            max_Fsh = nanmax(denan(ds["F_sh"][1, :]))[1]
            max_Fq = nanmax(denan(ds["F_q"][1, :]))[1]
            max_U10 = nanmax(denan(ds["U10"][1, :]))[1]
            max_ust = nanmax(denan(ds["ust"][1, :]))[1]
        end

        n_clamp_e = attrnum(ds, "mynn_n_clamp_e")
        n_cap_K = attrnum(ds, "mynn_n_cap_K")
        n_diffnum = attrnum(ds, "mynn_n_diffnum")
        n_gate = attrnum(ds, "mynn_n_gate")
        n_stall = attrnum(ds, "mynn_n_stall")
        n_plume = attrnum(ds, "mynn_n_plume")

        return FileStats(t, path, physics_groups, has_mynn, has_surface,
            max_v, max_v_r_km, max_v_z_km,
            min_pp_hpa, min_pp_r_km,
            max_inflow, max_inflow_r_km, max_outflow, max_outflow_r_km,
            max_w, max_w_r_km, max_w_z_km,
            max_refl, mean_precip, max_PW,
            max_Km, max_Kh, pblh_rmw, rmw_r_km, max_e,
            max_Fsh, max_Fq, max_U10, max_ust,
            n_clamp_e, n_cap_K, n_diffnum, n_gate, n_stall, n_plume)
    finally
        close(ds)
    end
end

# ---------------------------------------------------------------------------
# Clamp histogram (optional)
# ---------------------------------------------------------------------------

function clamp_hist_table(path)
    ds = NCDataset(path)
    local lines
    try
        haskey(ds, "e") || return "  (no `e` variable in $(basename(path)) — not a MYNN file)\n"
        e = denan(ds["e"][1, :, :])           # (x, z)
        z_km = denan(ds["z"][:]) ./ 1000.0
        emin = nanmin(e)[1]
        if !isfinite(emin)
            return "  (no finite `e` values in $(basename(path)))\n"
        end
        # Widen the threshold 5% of |emin| away from the floor value, toward the bulk of the
        # distribution, regardless of sign (the plain `1.05 * emin` from the spec only works
        # when the floor is a small positive epsilon; the regridded field here can dip
        # slightly negative — see the caveat below — in which case `1.05 * emin` would move
        # further from zero and match nothing).
        floor_thresh = emin + 0.05 * abs(emin)
        lines = String[]
        push!(lines, "| z [km] | frac(e <= floor+5%|floor|) | n finite |")
        push!(lines, "|---|---|---|")
        for iz in 1:size(e, 2)
            col = @view e[:, iz]
            fin = filter(isfinite, col)
            n = length(fin)
            n == 0 && continue
            frac = count(v -> v <= floor_thresh, fin) / n
            push!(lines, "| $(fmt(z_km[iz]; digits=2)) | $(fmt(frac; digits=3)) | $n |")
        end
        push!(lines, "")
        push!(lines, "min(e) over the file = $(fmt(emin; digits=5)) m2/s2 (regridded field; " *
                      "the closure's own mish min may differ and can be negative here due to " *
                      "the linear-interpolation regridding described in the `mynn_regridding` " *
                      "attribute — this is a proxy, not the census counter itself).")
    finally
        close(ds)
    end
    return join(lines, "\n") * "\n"
end

# ---------------------------------------------------------------------------
# ModelParameters-line parsing (textual, not a real Julia parse — see module docstring)
# ---------------------------------------------------------------------------

# Split a `Dict(...)`/`Dict{Symbol, Any}(...)` inner-content string into key => value-text
# pairs by finding `:key => ` boundaries. Robust to nested `[...]` (e.g. `[:csv, :netcdf]`)
# because it never splits on a bare comma, only at a recognized `:key => ` boundary.
function split_dict_text(s)
    d = Dict{String,String}()
    ms = collect(eachmatch(r"(?:^|, )(:\w+) => ", s))
    for (i, m) in enumerate(ms)
        key = m.captures[1][2:end]
        vstart = m.offset + length(m.match)
        vend = i < length(ms) ? ms[i+1].offset - 1 : lastindex(s)
        val = rstrip(strip(s[vstart:vend]), [',', ' '])
        d[key] = val
    end
    return d
end

# Returns (physical_params::Dict{String,String}, options::Dict{String,String}) parsed from
# line 2 of `nest<N>/scythe_out.log` (the `ModelParameters(...)` dump). Returns (nothing,
# nothing) if the line doesn't have the expected shape (e.g. missing/short log).
function parse_model_parameters_line(line)
    l = rstrip(line)
    (isempty(l) || !endswith(l, "))")) && return (nothing, nothing)
    body = l[1:end-2]
    marker = "Dict{Symbol, Any}("
    idx = findlast(marker, body)
    idx === nothing && return (nothing, nothing)
    options_str = body[idx[end]+1:end]
    before = rstrip(body[1:idx[1]-1], [',', ' '])
    ms = collect(eachmatch(r"Dict\(:", before))
    isempty(ms) && return (nothing, split_dict_text(options_str))
    idx2 = ms[end].offset
    phys_full = before[idx2:end]
    # strip leading "Dict(" (5 chars) and trailing ")"
    phys_str = phys_full[6:end-1]
    return (split_dict_text(phys_str), split_dict_text(options_str))
end

function read_model_parameters_line(indir, nest)
    p = joinpath(nestdir(indir, nest), "scythe_out.log")
    isfile(p) || return nothing
    lines = readlines(p)
    length(lines) < 2 && return nothing
    return lines[2]
end

function dict_diff_table(labelA, dA, labelB, dB)
    allkeys = sort(collect(union(keys(dA), keys(dB))))
    lines = String[]
    push!(lines, "| key | $labelA | $labelB |")
    push!(lines, "|---|---|---|")
    ndiff = 0
    for k in allkeys
        va = get(dA, k, "—")
        vb = get(dB, k, "—")
        if va != vb
            ndiff += 1
            push!(lines, "| `$k` | $va | $vb |")
        end
    end
    if ndiff == 0
        push!(lines, "| *(no differing keys)* |  |  |")
    end
    return join(lines, "\n") * "\n"
end

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

function job_stdout_provenance(indir)
    p = joinpath(indir, "job_stdout.log")
    isfile(p) || return nothing
    lines = readlines(p)
    keep = filter(l -> occursin(r"HEAD|dirty|SCYTHE_TC|threads|wall clock|TC boundary layer:|TC radiation:"i, l), lines)
    return keep
end

function nest_log_wallclock_line(indir, nest)
    p = joinpath(nestdir(indir, nest), "scythe_out.log")
    isfile(p) || return nothing
    for l in readlines(p)
        occursin("wall clock", l) && return l
    end
    return nothing
end

function mtime_wallclock(indir, nest, files)
    isempty(files) && return NaN
    icscsv = joinpath(indir, "nest$(nest)_tc_ics.csv")
    t0 = isfile(icscsv) ? mtime(icscsv) : mtime(files[1][2])
    t1 = mtime(files[end][2])
    return t1 - t0
end

function census_line(indir, nest)
    p = joinpath(nestdir(indir, nest), "scythe_out.log")
    isfile(p) || return nothing
    line = nothing
    for l in readlines(p)
        occursin("mynn census", l) && (line = l)
    end
    return line
end

function provenance_block(indir, nest, files)
    lines = String[]
    push!(lines, "**Run dir:** `$indir`, nest $nest")
    jp = job_stdout_provenance(indir)
    if jp === nothing
        push!(lines, "**`job_stdout.log`:** not present (laptop run without the sbatch wrapper).")
    else
        push!(lines, "**`job_stdout.log`` provenance lines:**")
        push!(lines, "```")
        append!(lines, jp)
        push!(lines, "```")
    end
    wl = nest_log_wallclock_line(indir, nest)
    if jp !== nothing && any(l -> occursin("wall clock", l), jp)
        push!(lines, "**Wall clock:** from `job_stdout.log` (above).")
    elseif wl !== nothing
        push!(lines, "**Wall clock:** from `nest$(nest)/scythe_out.log`: `$wl`")
    else
        wc = mtime_wallclock(indir, nest, files)
        push!(lines, "**Wall clock:** no driver-reported wall-clock line found in " *
                      "`job_stdout.log` or `nest$(nest)/scythe_out.log`; estimated from " *
                      "file mtimes (`nest$(nest)_tc_ics.csv` -> last `<t>.nc`) as a fallback " *
                      "proxy: **$(fmt(wc; digits=1)) s** ($(fmt(wc/60; digits=1)) min). " *
                      "This is a filesystem-timestamp proxy, not the driver's own `@elapsed` " *
                      "figure (which prints to the top-level driver stdout, not captured " *
                      "in this run's logs).")
    end
    cl = census_line(indir, nest)
    if cl !== nothing
        push!(lines, "**MYNN census (end of run):** `$cl`")
    end
    mp = read_model_parameters_line(indir, nest)
    if mp === nothing
        push!(lines, "**`ModelParameters` line:** not found (need `nest$(nest)/scythe_out.log` " *
                      "line 2).")
    else
        physd, optd = parse_model_parameters_line(mp)
        if optd === nothing
            push!(lines, "**`ModelParameters` line:** present but did not match the expected " *
                          "shape (textual parse failed) — dumping the raw options substring is " *
                          "skipped; see the raw log line directly.")
        else
            push!(lines, "**`options` keys ($(length(optd))):** " *
                          join(["`$k` => $v" for (k, v) in sort(collect(optd))], "; "))
        end
    end
    return join(lines, "\n\n") * "\n"
end

# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

function hourly_table(stats::Vector{FileStats})
    any_mynn = any(s -> s.has_mynn, stats)
    any_sfc = any(s -> s.has_surface, stats)

    header = ["t [h]", "max v (r,z) [km]", "min p' [hPa] @ r", "max inflow [m/s] @ r",
              "max outflow [m/s] @ r", "max w [m/s] @ r,z", "max refl [dBZ]",
              "mean precip [mm/hr]", "max PW [kg/m2]"]
    if any_mynn
        append!(header, ["max K_m", "max K_h", "pblh@RMW [m]", "max e"])
    end
    if any_sfc
        append!(header, ["max F_sh", "max F_q", "max U10", "max ust"])
    end

    lines = String[]
    push!(lines, "| " * join(header, " | ") * " |")
    push!(lines, "|" * repeat("---|", length(header)))

    for s in stats
        row = String[]
        push!(row, fmt(s.t / 3600.0; digits=2))
        push!(row, "$(fmt(s.max_v; digits=2)) (@ $(fmt(s.max_v_r_km; digits=0)), $(fmt(s.max_v_z_km; digits=2)))")
        push!(row, "$(fmt(s.min_pp_hpa; digits=1)) (@ $(fmt(s.min_pp_r_km; digits=0)))")
        push!(row, "$(fmt(s.max_inflow; digits=3)) (@ $(fmt(s.max_inflow_r_km; digits=0)))")
        push!(row, "$(fmt(s.max_outflow; digits=3)) (@ $(fmt(s.max_outflow_r_km; digits=0)))")
        push!(row, "$(fmt(s.max_w; digits=2)) (@ $(fmt(s.max_w_r_km; digits=0)), $(fmt(s.max_w_z_km; digits=2)))")
        push!(row, isfinite(s.max_refl) ? fmt(s.max_refl; digits=1) : "n/a (all-NaN)")
        push!(row, fmt(s.mean_precip; digits=4))
        push!(row, fmt(s.max_PW; digits=2))
        if any_mynn
            if s.has_mynn
                push!(row, fmt(s.max_Km; digits=2))
                push!(row, fmt(s.max_Kh; digits=2))
                push!(row, "$(fmt(s.pblh_rmw; digits=1)) (r=$(fmt(s.rmw_r_km; digits=0)))")
                push!(row, fmt(s.max_e; digits=3))
            else
                append!(row, ["n/a", "n/a", "n/a", "n/a"])
            end
        end
        if any_sfc
            if s.has_surface
                push!(row, fmt(s.max_Fsh; digits=1))
                push!(row, fmt(s.max_Fq; digits=6))
                push!(row, fmt(s.max_U10; digits=2))
                push!(row, fmt(s.max_ust; digits=3))
            else
                append!(row, ["n/a", "n/a", "n/a", "n/a"])
            end
        end
        push!(lines, "| " * join(row, " | ") * " |")
    end
    footnote = String[]
    if !any_mynn
        push!(footnote, "(no MYNN columns: no file in this run carries `mynn` in `physics_groups`)")
    end
    if !any_sfc
        push!(footnote, "(no surface columns: no file in this run carries `surface` in `physics_groups`)")
    end
    push!(lines, "")
    push!(lines, "`min p'` and the inflow/outflow search use the lowest output level as the " *
                  "surface diagnostic (matching `tc/output/S10_BL_ASSESSMENT.md`); inflow " *
                  "searches z <= 2 km, outflow z >= 8 km. " * join(footnote, " "))
    return join(lines, "\n") * "\n"
end

function final_comparison_table(labels, laststats::Vector{FileStats})
    lines = String[]
    metrics = [
        ("max v [m/s]", s -> s.max_v),
        ("min p' [hPa]", s -> s.min_pp_hpa),
        ("max inflow [m/s]", s -> s.max_inflow),
        ("max outflow [m/s]", s -> s.max_outflow),
        ("max w [m/s]", s -> s.max_w),
        ("mean precip [mm/hr]", s -> s.mean_precip),
        ("max PW [kg/m2]", s -> s.max_PW),
        ("max K_m", s -> s.max_Km),
        ("max K_h", s -> s.max_Kh),
        ("pblh@RMW [m]", s -> s.pblh_rmw),
        ("max e", s -> s.max_e),
        ("max F_sh", s -> s.max_Fsh),
        ("max F_q", s -> s.max_Fq),
        ("max U10", s -> s.max_U10),
    ]
    push!(lines, "| metric | " * join(labels, " | ") * " |")
    push!(lines, "|---|" * repeat("---|", length(labels)))
    for (name, f) in metrics
        vals = [fmt(f(s); digits=4) for s in laststats]
        push!(lines, "| $name | " * join(vals, " | ") * " |")
    end
    return join(lines, "\n") * "\n"
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main(argv)
    indirs, nest, out, clamp_hist = parse_args(argv)

    report = IOBuffer()
    println(report, "# TC boundary-layer assessment (auto-generated by tc/tc_bl_assess.jl)")
    println(report)
    println(report, "Runs: " * join(["`$d`" for d in indirs], ", ") * "; nest $nest.")
    println(report)

    all_stats = Vector{Vector{FileStats}}()
    for indir in indirs
        nd = nestdir(indir, nest)
        files = list_nc_files(nd)
        if isempty(files)
            println(report, "## `$indir`\n")
            println(report, "**No `<t>.nc` files found in `$nd`.**\n")
            push!(all_stats, FileStats[])
            continue
        end
        stats = [file_stats(p) for (_, p) in files]
        push!(all_stats, stats)

        println(report, "## `$indir` (nest $nest)")
        println(report)
        println(report, hourly_table(stats))
        println(report)
        println(report, "### Provenance")
        println(report)
        println(report, provenance_block(indir, nest, files))
        println(report)

        if clamp_hist
            last_mynn = findlast(s -> s.has_mynn, stats)
            if last_mynn !== nothing
                println(report, "### Clamp-floor histogram (proxy), last MYNN file: " *
                                 "`$(basename(stats[last_mynn].path))`")
                println(report)
                println(report, clamp_hist_table(stats[last_mynn].path))
            else
                println(report, "### Clamp-floor histogram (proxy)")
                println(report)
                println(report, "(no MYNN file in this run — skipped)")
            end
            println(report)
        end
    end

    if length(indirs) >= 2
        println(report, "## Options / physical_params diff (first two runs)")
        println(report)
        mpA = read_model_parameters_line(indirs[1], nest)
        mpB = read_model_parameters_line(indirs[2], nest)
        if mpA === nothing || mpB === nothing
            println(report, "(could not read `ModelParameters` line for one or both runs)")
        else
            physA, optA = parse_model_parameters_line(mpA)
            physB, optB = parse_model_parameters_line(mpB)
            labelA, labelB = basename(indirs[1]), basename(indirs[2])
            if optA !== nothing && optB !== nothing
                println(report, "**options:**")
                println(report)
                println(report, dict_diff_table(labelA, optA, labelB, optB))
                println(report)
            end
            if physA !== nothing && physB !== nothing
                println(report, "**physical_params:**")
                println(report)
                println(report, dict_diff_table(labelA, physA, labelB, physB))
                println(report)
            end
        end

        println(report, "## Final-time side-by-side comparison")
        println(report)
        labels = [basename(d) for d in indirs]
        laststats = FileStats[]
        keep_labels = String[]
        for (lab, stats) in zip(labels, all_stats)
            if !isempty(stats)
                push!(laststats, stats[end])
                push!(keep_labels, lab)
            end
        end
        if !isempty(laststats)
            println(report, final_comparison_table(keep_labels, laststats))
        else
            println(report, "(no runs had output files)")
        end
    end

    text = String(take!(report))
    if out === nothing
        print(text)
    else
        mkpath(dirname(out))
        write(out, text)
        println("Wrote $out")
    end
end

main(ARGS)
