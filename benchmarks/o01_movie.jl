#!/usr/bin/env julia
# Cloud-and-rain evolution movie for an existing o01_rainfall run (no re-run):
# a color fill of the condensate rho_c with rho_r contours, one frame per output
# snapshot, assembled with ffmpeg. Works for BOTH the single-grid run and the
# 3-level / 5-patch two-way nest (--nests 3): the nested patches abut in x and
# are drawn on one shared axis, finest last so it wins in the collar overlaps.
#
#   julia --project=. benchmarks/o01_movie.jl [--mode full] [--grid rirk]
#         [--nests 1|3] [--dir NAME_OR_PATH] [--field rho_c|rho_r]
#         [--xlim lo,hi] [--zlim lo,hi] [--fps 8]
#         [--ctrans none|bhyp|bhyp_smooth] [--rtrans ...] [--cmu X] [--rmu X]
#
# Reads benchmarks/output/o01_rainfall_<mode>_mc<gridsuffix><nestsuffix>/ and
# writes o01_rainfall_<mode>_<grid><nestsuffix>_<field>.mp4 there. Grid
# dimensions are inferred per patch from the snapshots (3 Gauss points per cell),
# so this works at any resolution the benchmark was run at. --dir overrides the
# constructed path with a directory name under benchmarks/output (or an absolute
# path), which is how the hand-labelled sweep directories — _ladder_bhyp and the
# rest, which benchmark_output_dir has no knob to emit — are reached.
#
# THE CONTROL VARIABLE. Under options[:condensate_transform] / [:rain_transform]
# the corresponding slot carries Ooyama's biased hyperbolic control variable
# rather than a density, and the output column is NAMED for it (nu_c / nu_r).
# This script reads the names, applies the inverse map, and plots densities. It
# did not always: against a transformed run it used to draw half-amplitude cloud
# with a spurious −0.46 g/m³ minimum, over a run whose own diagnostics.csv
# recorded min_rho_c = 0 exactly. The exact variant and the bias are read from
# scythe_out.log when present and can be overridden with --ctrans/--cmu etc.
#
# The default view is the original: rho_c filled (viridis), rho_r as line
# contours. Untransformed, rho_c and rho_r are prognostic densities that can
# undershoot NEGATIVE where the vertical B-spline basis cannot resolve a
# positive-definite spike (see clamp_water! in src/moist_compressible.jl).
# Negative water is drawn as dashed contours over the fill — magenta for rain,
# cyan for cloud — and both fields' per-frame minima are printed in the title.
# Under a transform the recovered density cannot go below −mu, so those contours
# fall silent by construction and their absence is the result, not a bug.
# --field rho_r instead fills rain with a diverging map so its (larger)
# undershoots read blue directly. --xlim 64,86 zooms to the fine nest, where the
# undershoots concentrate.

using CSV
using DataFrames
using CairoMakie
using Printf
using Scythe
using Springsteel

# ── Arguments ───────────────────────────────────────────────────────────────
mode = "full"
grid = "rirk"
nests = 0                       # 0 = auto-detect from the output directory
field = "rho_c"                 # fill field: rho_c or rho_r
fps = 8
xlim = nothing                  # (lo, hi) km, or nothing for the full domain
zlim = (0.0, 17.0)              # km
rundir = nothing                # explicit output directory (name under output/, or a path)
ctrans_arg = nothing            # transform overrides; nothing = detect (see below)
rtrans_arg = nothing
cmu_arg = nothing
rmu_arg = nothing
let i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--mode"
            global mode = ARGS[i+1]; i += 2
        elseif ARGS[i] == "--grid"
            global grid = ARGS[i+1]; i += 2
        elseif ARGS[i] == "--nests"
            global nests = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--dir"
            global rundir = ARGS[i+1]; i += 2
        elseif ARGS[i] == "--field"
            global field = ARGS[i+1]; i += 2
        elseif ARGS[i] == "--xlim"
            global xlim = Tuple(parse.(Float64, split(ARGS[i+1], ","))); i += 2
        elseif ARGS[i] == "--zlim"
            global zlim = Tuple(parse.(Float64, split(ARGS[i+1], ","))); i += 2
        elseif ARGS[i] == "--fps"
            global fps = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--ctrans"
            global ctrans_arg = Symbol(ARGS[i+1]); i += 2
        elseif ARGS[i] == "--rtrans"
            global rtrans_arg = Symbol(ARGS[i+1]); i += 2
        elseif ARGS[i] == "--cmu"
            global cmu_arg = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--rmu"
            global rmu_arg = parse(Float64, ARGS[i+1]); i += 2
        else
            error("Unknown argument: $(ARGS[i])")
        end
    end
end
field in ("rho_c", "rho_r") || error("--field must be rho_c or rho_r")

# ── Locate the run, and its patches ─────────────────────────────────────────
suffix = grid == "rz" ? "" : "_$(grid)"
# Auto-detect nesting: prefer an explicit --nests, else look for a _n3 dir with
# nest*/ subdirectories, else fall back to the single-grid layout.
function resolve_dir()
    base(nsfx) = joinpath(@__DIR__, "output", "o01_rainfall_$(mode)_mc$(suffix)$(nsfx)")
    # --dir wins: `benchmark_output_dir` has no run-label knob, so every sweep directory
    # (_ladder_bhyp, _lq3_none, ...) is a manual rename and cannot be reconstructed here.
    if rundir !== nothing
        d = isdir(rundir) ? rundir : joinpath(@__DIR__, "output", rundir)
        isdir(d) || error("No run output at $d")
        nested_here = !isempty(filter(f -> startswith(f, "nest") &&
                                           isdir(joinpath(d, f)), readdir(d)))
        return d, (nests > 1 || nested_here)
    end
    if nests > 1
        d = base("_n$(nests)")
        isdir(d) || error("No nested run output at $d")
        return d, true
    elseif nests == 1
        d = base("")
        isdir(d) || error("No single-grid run output at $d")
        return d, false
    end
    # auto
    for n in (3, 2)
        d = base("_n$(n)")
        isdir(d) && !isempty(filter(f -> startswith(f, "nest"), readdir(d))) &&
            return d, true
    end
    d = base("")
    isdir(d) || error("No run output at $d (single-grid) or its _n* nested siblings")
    return d, false
end
dir, nested = resolve_dir()

# Patch directories, inner→outer is irrelevant; we sort so the FINEST is drawn
# last (below). For the single grid there is one "patch" (dir itself).
patch_dirs = nested ?
    sort(filter(d -> isdir(joinpath(dir, d)) && startswith(d, "nest"), readdir(dir)),
         by = s -> parse(Int, replace(s, "nest" => ""))) : ["."]
npatch = length(patch_dirs)
println("Run: $dir  ($(nested ? "$npatch-patch nest" : "single grid"))")

# ── Shared vertical reference (identical column on every patch) ─────────────
# The nest patches share one vertical grid and one o01_exact.ref at the base
# directory, so rho_cbar(z) is built once and reused. The reference cloud is
# zero on the subsaturated Dunion sounding, but add it for correctness.
vars = ["p", "rho_d", "rho_t", "u", "w", "E_t", "Q_ss", "rho_r", "rho_c"]
scalar_bc = Dict(v => NeumannBC() for v in vars)

# Build rho_cbar from the FIRST patch's kDim (all patches share it). The throwaway
# grid supplies only reference VALUES; its BCs are unused.
first_path = let d = joinpath(dir, patch_dirs[1])
    fs = filter(f -> endswith(f, "_physical.csv"), readdir(d))
    joinpath(d, first(sort(fs, by = f -> parse(Float64, replace(f, "_physical.csv" => "")))))
end

# ── What the water columns hold ─────────────────────────────────────────────
# Two sources, in this order. The COLUMN NAMES decide whether a transform is on: they come
# from the run's own `GridParameters.vars`, so they cannot disagree with the file. The log
# supplies only what a name cannot carry — the variant (:bhyp vs :bhyp_smooth) and the bias —
# and the command line overrides both. `mc_water` cross-checks the two and raises if they
# disagree, which is the guard that stops this script from ever again reading a control
# variable as a density.
include(joinpath(@__DIR__, "common", "diagnostics.jl"))
transform_cfg = let det = detect_transforms(dir),
                    hdr = names(CSV.read(first_path, DataFrame; limit = 1))
    # A nu_* column always means transformed (the log then supplies only the variant). Absent
    # one, the log still decides: the runs made before the slots were renamed carry the
    # control variable in a column named rho_c, and their log is the only record of it.
    ct = ctrans_arg !== nothing ? ctrans_arg :
         ("nu_c" in hdr ? (det.ctrans === :none ? :bhyp : det.ctrans) : det.ctrans)
    rt = rtrans_arg !== nothing ? rtrans_arg :
         ("nu_r" in hdr ? (det.rtrans === :none ? :bhyp : det.rtrans) : det.rtrans)
    (ctrans = ct, cmu = something(cmu_arg, det.cmu),
     rtrans = rt, rmu = something(rmu_arg, det.rmu))
end
water_desc(col, mode, mu) = mode === :none ? "$col (density)" :
    "$col (control variable, $(mode), mu = $(mu)" *
    (startswith(col, "nu_") ? ")" : "; pre-rename layout)")
let hdr = names(CSV.read(first_path, DataFrame; limit = 1))
    println("Water columns: cloud = " *
            water_desc("nu_c" in hdr ? "nu_c" : "rho_c",
                       transform_cfg.ctrans, transform_cfg.cmu) *
            ", rain = " *
            water_desc("nu_r" in hdr ? "nu_r" : "rho_r",
                       transform_cfg.rtrans, transform_cfg.rmu))
end

"""Read one patch snapshot; return (x[km] (ncol), z[km] (kDim), field2d, rho_c2d, rho_r2d)."""
function read_patch(path, rho_cbar)
    df = CSV.read(path, DataFrame)
    zc = df.z
    kDim = findfirst(i -> zc[i+1] < zc[i], 1:length(zc)-1)   # z resets per column
    ncol = div(nrow(df), kDim)
    x = reshape(df.r, kDim, ncol)[1, :] ./ 1000.0
    z = reshape(df.z, kDim, ncol)[:, 1] ./ 1000.0
    # The plotted fields are DENSITIES, recovered through the run's own inverse maps.
    rc, rr = mc_water(df, rho_cbar, ncol;
                      ctrans = transform_cfg.ctrans, cmu = transform_cfg.cmu,
                      rtrans = transform_cfg.rtrans, rmu = transform_cfg.rmu)
    return (x = x, z = z, kDim = kDim, ncol = ncol,
            rho_c = reshape(1000.0 .* rc, kDim, ncol),
            rho_r = reshape(1000.0 .* rr, kDim, ncol))
end
kDim0 = let zc = CSV.read(first_path, DataFrame).z
    findfirst(i -> zc[i+1] < zc[i], 1:length(zc)-1)
end
ztop = let df = CSV.read(first_path, DataFrame)
    zc = reshape(df.z, kDim0, div(nrow(df), kDim0))[:, 1]
    zc[1] + zc[end]                       # mish symmetric about the mid-height
end
gp = Scythe.compute_derived_params(GridParameters(;
    geometry = grid == "rz" ? "RZ" : "RiRk",
    iMin = 0.0, iMax = 150.0e3, num_cells_i = 8,
    kMin = 0.0, kMax = ztop,
    (grid == "rz" ? (; kDim = kDim0) : (; num_cells_k = div(kDim0, 3)))...,
    BCL = scalar_bc, BCR = scalar_bc, BCB = scalar_bc, BCT = scalar_bc,
    vars = Dict(v => i for (i, v) in enumerate(vars))))
refpatch = createGrid(gp)
column = Scythe.reference_column(refpatch, gp)
ref = Springsteel.exact_pressure_reference_state(joinpath(dir, "o01_exact.ref"),
                                                 Scythe.getGridpoints(refpatch)[1:kDim0, 2],
                                                 column)
rho_cbar = Springsteel.ref_rho_c(ref)[:, 1]

# ── Snapshot list (canonical: the first patch; all patches share output times) ─
snap_times = let d = joinpath(dir, patch_dirs[1])
    fs = filter(f -> endswith(f, "_physical.csv"), readdir(d))
    sort(parse.(Float64, replace.(fs, "_physical.csv" => "")))
end
isempty(snap_times) && error("No *_physical.csv snapshots under $dir")
println("Found $(length(snap_times)) snapshots, field = $field")

# ── Colour scaling ──────────────────────────────────────────────────────────
# One pass over every patch and snapshot to fix a stable, shared colour range so
# the movie does not re-scale frame to frame. The FILLED field sets it. rho_c
# fills with a sequential viridis (vivid cloud, the original look); rho_r fills
# with a diverging map centred on zero so its (large) undershoots read blue.
snap_path(p, t) = joinpath(dir, patch_dirs[p], "$(t)_physical.csv")
fld(pat) = field == "rho_c" ? pat.rho_c : pat.rho_r
vmax_pos = 0.0
vmin_neg = 0.0
for t in snap_times, p in 1:npatch
    f = fld(read_patch(snap_path(p, t), rho_cbar))
    global vmax_pos = max(vmax_pos, maximum(f))
    global vmin_neg = min(vmin_neg, minimum(f))
end
if field == "rho_c"
    # Sequential, positive; the anvil saturates via extendhigh so the low cloud
    # is not crushed. The fill STARTS at a small cloud threshold (not zero), so
    # the sub-threshold representation noise that the prognostic cloud carries
    # across the whole domain stays blank white rather than flickering across the
    # first band (the zebra artefact of a fill boundary sitting on zero). Negative
    # cloud is shown by its own dashed contours, below.
    cthresh = 0.05                          # g/m³ — below this is "no cloud"
    cmax = max(0.7 * vmax_pos, 0.4)
    levels = range(cthresh, cmax; length = 30)
    fill_cmap = :viridis
else
    # Diverging, symmetric, clipped a little tighter than the positive peak so
    # the smaller negative rain still reads at contrast.
    cmax = max(0.6 * vmax_pos, -vmin_neg, 0.2)
    levels = range(-cmax, cmax; length = 41)
    fill_cmap = :balance
end
rain_pos = [0.5, 1.0, 2.0, 4.0, 8.0]        # rho_r POSITIVE contours [g/m³]
rain_neg = [-4.0, -2.0, -1.0, -0.5]         # rho_r NEGATIVE (unphysical) contours
cloud_neg = [-0.6, -0.3, -0.1]              # rho_c NEGATIVE (unphysical) contours
println(@sprintf("fill = %s, range %s  (peak +%.2f, min %.3f)", field,
                 field == "rho_c" ? "0..$(round(cmax; digits=2))" :
                                    "±$(round(cmax; digits=2))", vmax_pos, vmin_neg))

# ── Frames ──────────────────────────────────────────────────────────────────
framedir = joinpath(dir, "movie_frames")
mkpath(framedir)
rm.(joinpath.(framedir, readdir(framedir)); force = true)

# Draw order: sort patches by horizontal cell size so the FINEST is plotted last
# (on top) in the abutting collars. Cell size ≈ x-span / ncol.
draw_order = let widths = map(1:npatch) do p
        pat = read_patch(snap_path(p, snap_times[1]), rho_cbar)
        (pat.x[end] - pat.x[1]) / pat.ncol
    end
    sortperm(widths; rev = true)             # coarse first, fine last
end

xspan = xlim === nothing ? (0.0, 150.0) : xlim

# Positive rain contours read best in a colour that contrasts the fill: white on
# the dark viridis cloud (the original convention), black on the light diverging
# rain map. Negative water is always dashed — magenta for rain, cyan for cloud.
rain_pos_color = field == "rho_c" ? :black : :white

conv_note = let bits = String[]
    transform_cfg.ctrans === :none || push!(bits, "ρ_c via $(transform_cfg.ctrans)⁻¹")
    transform_cfg.rtrans === :none || push!(bits, "ρ_r via $(transform_cfg.rtrans)⁻¹")
    isempty(bits) ? "" : "   [" * join(bits, ", ") * "]"
end

for (i, t) in enumerate(snap_times)
    fig = Figure(size = (1100, 460))
    ax = Axis(fig[1, 1], xlabel = "x (km)", ylabel = "z (km)")
    local cf = nothing
    minc = Inf; minr = Inf
    for p in draw_order
        pat = read_patch(snap_path(p, t), rho_cbar)
        minc = min(minc, minimum(pat.rho_c)); minr = min(minr, minimum(pat.rho_r))
        cf = contourf!(ax, pat.x, pat.z, fld(pat)',
                       levels = levels, extendhigh = :auto,
                       extendlow = field == "rho_c" ? nothing : :auto,
                       colormap = fill_cmap)
        # Positive rain as line contours over the filled cloud (the original view).
        contour!(ax, pat.x, pat.z, pat.rho_r', levels = rain_pos,
                 color = rain_pos_color, linewidth = 1.0)
        # Negative water, dashed: cloud in cyan first, then rain in magenta ON TOP
        # (rain carries the larger undershoots and is the primary thing to see).
        # These are the unphysical B-spline undershoots this movie exists to show.
        # Guard on the LEAST-negative level (…[end]) so a field that only dips to,
        # say, −3 still draws its −2/−1/−0.5 contours.
        any(pat.rho_c .< cloud_neg[end]) &&
            contour!(ax, pat.x, pat.z, pat.rho_c', levels = cloud_neg,
                     color = :cyan, linewidth = 1.2, linestyle = :dash)
        any(pat.rho_r .< rain_neg[end]) &&
            contour!(ax, pat.x, pat.z, pat.rho_r', levels = rain_neg,
                     color = :black, linewidth = 1.4, linestyle = :dash)
        # Mark patch seams faintly so the nest layout is legible.
        nested && vlines!(ax, [pat.x[1], pat.x[end]]; color = (:gray, 0.3),
                          linewidth = 0.5)
    end
    # State the CONVENTION alongside the minima. Under a transform these read 0.000 by
    # construction, and a reader has to be able to tell that from a run that simply had no
    # undershoot.
    ax.title = @sprintf("O01 warm rain — t = %d min    min ρ_c = %.3f, min ρ_r = %.3f g/m³%s",
                        round(Int, t / 60), minc, minr, conv_note)
    xlims!(ax, xspan...); ylims!(ax, zlim...)
    lbl = field == "rho_c" ?
        "ρ_c (g/m³)   [lines: ρ_r>0; dashed magenta ρ_r<0, cyan ρ_c<0]" :
        "ρ_r (g/m³)   [blue<0; lines ρ_r>0; dashed magenta ρ_r<0, cyan ρ_c<0]"
    Colorbar(fig[1, 2], cf, label = lbl)
    frame = joinpath(framedir, "frame_" * lpad(i - 1, 4, '0') * ".png")
    save(frame, fig)
    i % 10 == 0 && println("  frame $i / $(length(snap_times))")
end
println("Rendered $(length(snap_times)) frames")

# ── Movie ───────────────────────────────────────────────────────────────────
nsfx = nested ? "_n$(npatch == 5 ? 3 : npatch)" : ""
# Name the file after the DIRECTORY when one was given explicitly, so a sweep's movies do not
# all land on the same name and get hand-renamed (which is how the mp4 in
# o01_rainfall_full_mc_rirk_ladder_bhyp/ came to sit beside a differently-named run).
stem = rundir === nothing ? "o01_rainfall_$(mode)_$(grid)$(nsfx)" : basename(rstrip(dir, '/'))
movie = joinpath(dir, "$(stem)_$(field).mp4")
run(`ffmpeg -y -loglevel error -framerate $fps -i $(joinpath(framedir, "frame_%04d.png"))
     -c:v libx264 -pix_fmt yuv420p -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" $movie`)
println("Wrote $movie")
