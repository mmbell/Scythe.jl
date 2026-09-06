#!/usr/bin/env julia
# Cloud-and-rain evolution movie for an existing o01_rainfall run (no re-run):
# a color fill of the condensate rho_c with rho_r contours, one frame per output
# snapshot, assembled with ffmpeg. Works for BOTH the single-grid run and the
# 3-level / 5-patch two-way nest (--nests 3): the nested patches abut in x and
# are drawn on one shared axis, finest last so it wins in the collar overlaps.
#
#   julia --project=. benchmarks/o01_movie.jl [--mode full] [--grid rirk]
#         [--nests 1|3] [--dir NAME_OR_PATH] [--field rho_c|rho_r|ice|rad]
#         [--xlim lo,hi] [--zlim lo,hi] [--fps 8]
#         [--ctrans none|bhyp|bhyp_smooth] [--rtrans ...] [--cmu X] [--rmu X]
#         [--icetrans none|bhyp|bhyp_smooth] [--icemu X]
#
# --field rad (S5) needs a radiation sidecar (<tag>_radiation_i*.nc, Scythe.read_radiation
# -- src/radiation_io.jl) in every patch directory, i.e. the run was made with
# SCYTHE_O01_RAD set. Three panels: a thin top row of domain OLR vs x (line); bottom-left
# is the SAME condensate view as --field rho_c/ice (rho_c fill, rho_r contours, and the
# three ISHMAEL ice-mass contours WHEN the run carries ice -- same "mass vars only,
# distinct colours per species" convention, drawn automatically rather than gated behind
# a separate flag); bottom-right is the net radiative heating `dT_lw + sw_scale*dT_sw`
# [K/day] on the radiation mish (no regridding), a diverging colormap with symmetric
# limits fixed over the whole run. Sign convention: negative (blue) is cooling.
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
#
# --field ice keeps the rho_c fill and rho_r contours of the default view — same
# thresholds, same levels — and adds the three ISHMAEL ice MASS densities (the
# planar/columnar/aggregate species of Scythe.MC_ICE_VARS: i1/i2/i3) as their own line
# contours, in a colour family chosen to stay legible against both viridis and white:
# dodgerblue (planar), purple (columnar), darkorange (aggregate). No number or
# axis-moment slot is drawn — the point is "where is ice vs liquid", not the DSD. The
# ice slots carry their own control variable under options[:ice_transform] (default
# :bhyp), one family key for all twelve slots but a per-moment mu (Scythe.MC_ICE_MU_KEYS/
# MC_ICE_MU_DEFAULTS); the mass moment's mu is the SAME key (:mu_ice) for all three
# species, so this script reads and applies just the one. Ice mass is a TOTAL (no
# reference profile, unlike cloud), so recovery is the bare inverse map. Contour levels
# are a fixed g/m³ ladder (0.01/0.05/0.2/0.5/1/2), clipped per species to that species'
# own domain max over the whole run — a species that never reaches the first rung draws
# nothing, which is the answer ("this species is absent"), not a missing feature. A run
# with no ice columns (options[:ice_microphysics] = :none) errors out by name rather
# than silently drawing an empty ice view.

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
field = "rho_c"                 # fill field: rho_c, rho_r, or ice (rho_c fill + ice contours)
fps = 8
xlim = nothing                  # (lo, hi) km, or nothing for the full domain
zlim = (0.0, 17.0)              # km
rundir = nothing                # explicit output directory (name under output/, or a path)
ctrans_arg = nothing            # transform overrides; nothing = detect (see below)
rtrans_arg = nothing
cmu_arg = nothing
rmu_arg = nothing
icetrans_arg = nothing          # ice-mass transform override; nothing = detect
icemu_arg = nothing
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
        elseif ARGS[i] == "--icetrans"
            global icetrans_arg = Symbol(ARGS[i+1]); i += 2
        elseif ARGS[i] == "--icemu"
            global icemu_arg = parse(Float64, ARGS[i+1]); i += 2
        else
            error("Unknown argument: $(ARGS[i])")
        end
    end
end
field in ("rho_c", "rho_r", "ice", "rad") ||
    error("--field must be rho_c, rho_r, ice or rad")
is_rad = field == "rad"

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
transform_cfg = let det = detect_transforms(dir), det_ice = detect_ice_transform(dir),
                    hdr = names(CSV.read(first_path, DataFrame; limit = 1))
    # A nu_* column always means transformed (the log then supplies only the variant). Absent
    # one, the log still decides: the runs made before the slots were renamed carry the
    # control variable in a column named rho_c, and their log is the only record of it.
    ct = ctrans_arg !== nothing ? ctrans_arg :
         ("nu_c" in hdr ? (det.ctrans === :none ? :bhyp : det.ctrans) : det.ctrans)
    rt = rtrans_arg !== nothing ? rtrans_arg :
         ("nu_r" in hdr ? (det.rtrans === :none ? :bhyp : det.rtrans) : det.rtrans)
    # Ice mirrors cloud/rain: the nu_i1 name is authoritative, the log only supplies the
    # variant and the mass mu (Scythe.MC_ICE_MU_KEYS[1] == :mu_ice for all three species).
    it = icetrans_arg !== nothing ? icetrans_arg :
         ("nu_i1" in hdr ? (det_ice.itrans === :none ? :bhyp : det_ice.itrans) : det_ice.itrans)
    (ctrans = ct, cmu = something(cmu_arg, det.cmu),
     rtrans = rt, rmu = something(rmu_arg, det.rmu),
     itrans = it, imu = something(icemu_arg, det_ice.imu))
end
# Ice-mass column presence, by NAME (either the density name or its nu_* alias) --
# --field ice REQUIRES the three columns (name exactly what is missing rather than
# letting the failure surface deep inside the render loop as an opaque KeyError);
# --field rad draws the ice contours only WHEN they are present, same convention as the
# left panel of --field ice, but does not require ice (a warm-rain radiation run is a
# perfectly good --field rad target).
ice_present = let hdr = names(CSV.read(first_path, DataFrame; limit = 1))
    missing_ice = [ "$role/$alias" for (role, alias) in
                    (("rho_i1", "nu_i1"), ("rho_i2", "nu_i2"), ("rho_i3", "nu_i3"))
                    if !(role in hdr) && !(alias in hdr) ]
    if field == "ice" && !isempty(missing_ice)
        error("--field ice needs the three ISHMAEL ice-mass columns, but this run has none " *
              "of " * join(missing_ice, ", ") * " in $dir — it was very likely run with " *
              "options[:ice_microphysics] = :none (or an older layout). Columns present: " *
              "$hdr")
    end
    isempty(missing_ice)
end
draw_ice = field == "ice" || (is_rad && ice_present)
water_desc(col, mode, mu) = mode === :none ? "$col (density)" :
    "$col (control variable, $(mode), mu = $(mu)" *
    (startswith(col, "nu_") ? ")" : "; pre-rename layout)")
let hdr = names(CSV.read(first_path, DataFrame; limit = 1))
    ice_desc = draw_ice ? ", ice mass (i1/i2/i3) = " *
        water_desc("nu_i1" in hdr ? "nu_i1" : "rho_i1", transform_cfg.itrans,
                   transform_cfg.imu) : ""
    println("Water columns: cloud = " *
            water_desc("nu_c" in hdr ? "nu_c" : "rho_c",
                       transform_cfg.ctrans, transform_cfg.cmu) *
            ", rain = " *
            water_desc("nu_r" in hdr ? "nu_r" : "rho_r",
                       transform_cfg.rtrans, transform_cfg.rmu) * ice_desc)
end

"""Read one patch snapshot; return (x[km] (ncol), z[km] (kDim), field2d, rho_c2d, rho_r2d,
i1/i2/i3 2d). The three ice-mass fields are `nothing` unless ice contours are being drawn
(`--field ice`, or `--field rad` on a run that carries ice) — the liquid-only runs this
script also serves carry no ice columns to recover them from."""
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
    i1 = i2 = i3 = nothing
    if draw_ice
        ri1, ri2, ri3 = mc_ice(df; itrans = transform_cfg.itrans, imu = transform_cfg.imu)
        i1 = reshape(1000.0 .* ri1, kDim, ncol)
        i2 = reshape(1000.0 .* ri2, kDim, ncol)
        i3 = reshape(1000.0 .* ri3, kDim, ncol)
    end
    return (x = x, z = z, kDim = kDim, ncol = ncol,
            rho_c = reshape(1000.0 .* rc, kDim, ncol),
            rho_r = reshape(1000.0 .* rr, kDim, ncol),
            i1 = i1, i2 = i2, i3 = i3)
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
# The run's own exact reference column: o01_rainfall writes o01_exact.ref, ocean_warm_bubble
# writes owb_exact.ref -- any single "*_exact.ref" in the run directory is the one.
reffiles = filter(f -> endswith(f, "_exact.ref"), readdir(dir))
length(reffiles) == 1 || error("expected exactly one *_exact.ref in $dir, found $(reffiles)")
ref = Springsteel.exact_pressure_reference_state(joinpath(dir, reffiles[1]),
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
# with a diverging map centred on zero so its (large) undershoots read blue. The
# ice view fills with rho_c too (same thresholds/levels as --field rho_c) and, in
# the same pass, tracks each ice species' own domain max — the ABSOLUTE g/m³ ladder
# below is then clipped per species so it is stable across frames and an absent
# species draws nothing.
snap_path(p, t) = joinpath(dir, patch_dirs[p], "$(t)_physical.csv")
fld(pat) = field == "rho_r" ? pat.rho_r : pat.rho_c
vmax_pos = 0.0
vmin_neg = 0.0
imax1 = imax2 = imax3 = 0.0
for t in snap_times, p in 1:npatch
    pat = read_patch(snap_path(p, t), rho_cbar)
    f = fld(pat)
    global vmax_pos = max(vmax_pos, maximum(f))
    global vmin_neg = min(vmin_neg, minimum(f))
    if draw_ice
        global imax1 = max(imax1, maximum(pat.i1))
        global imax2 = max(imax2, maximum(pat.i2))
        global imax3 = max(imax3, maximum(pat.i3))
    end
end

# ── Radiation colour/axis scaling (--field rad only) ─────────────────────────
# A second pass, over the sidecar (Scythe.read_radiation, on the radiation mish -- no
# regridding): the net-heating diverging colour range (symmetric, fixed over the whole
# run, same rationale as rho_r's above) and the OLR line panel's y-range (so it does not
# rescale frame to frame either).
rad_cmax = 0.1
olr_lo = Inf
olr_hi = -Inf
if is_rad
    for p in 1:npatch
        pdir = joinpath(dir, patch_dirs[p])
        isempty(Scythe.radiation_snapshots(pdir)) &&
            error("--field rad needs a radiation sidecar (<tag>_radiation_i*.nc, " *
                  "Scythe.read_radiation) in $pdir — this run was very likely made with " *
                  "radiation off. Re-run o01_rainfall.jl with SCYTHE_O01_RAD set.")
    end
    for t in snap_times, p in 1:npatch
        rad = Scythe.read_radiation(joinpath(dir, patch_dirs[p]), string(t))
        net = rad.dT_lw .+ rad.sw_scale .* rad.dT_sw
        global rad_cmax = max(rad_cmax, maximum(abs.(net)))
        global olr_lo = min(olr_lo, minimum(rad.olr))
        global olr_hi = max(olr_hi, maximum(rad.olr))
    end
end
if field == "rho_r"
    # Diverging, symmetric, clipped a little tighter than the positive peak so
    # the smaller negative rain still reads at contrast.
    cmax = max(0.6 * vmax_pos, -vmin_neg, 0.2)
    levels = range(-cmax, cmax; length = 41)
    fill_cmap = :balance
else
    # Sequential, positive; the anvil saturates via extendhigh so the low cloud
    # is not crushed. The fill STARTS at a small cloud threshold (not zero), so
    # the sub-threshold representation noise that the prognostic cloud carries
    # across the whole domain stays blank white rather than flickering across the
    # first band (the zebra artefact of a fill boundary sitting on zero). Negative
    # cloud is shown by its own dashed contours, below. Shared verbatim by --field
    # ice, which fills the same rho_c and only adds the ice line contours on top.
    cthresh = 0.05                          # g/m³ — below this is "no cloud"
    cmax = max(0.7 * vmax_pos, 0.4)
    levels = range(cthresh, cmax; length = 30)
    fill_cmap = :viridis
end
rain_pos = [0.5, 1.0, 2.0, 4.0, 8.0]        # rho_r POSITIVE contours [g/m³]
rain_neg = [-4.0, -2.0, -1.0, -0.5]         # rho_r NEGATIVE (unphysical) contours
cloud_neg = [-0.6, -0.3, -0.1]              # rho_c NEGATIVE (unphysical) contours
# Ice-mass contour levels: one fixed ladder, clipped to each species' OWN domain max so a
# species that never reaches the first rung draws nothing rather than an empty legend entry.
ice_ladder = [0.01, 0.05, 0.2, 0.5, 1.0, 2.0]           # g/m³
ice_levels1 = filter(l -> l <= imax1, ice_ladder)       # i1 = planar
ice_levels2 = filter(l -> l <= imax2, ice_ladder)       # i2 = columnar
ice_levels3 = filter(l -> l <= imax3, ice_ladder)       # i3 = aggregate
println(@sprintf("fill = %s, range %s  (peak +%.2f, min %.3f)", field,
                 field == "rho_r" ? "±$(round(cmax; digits=2))" :
                                    "0..$(round(cmax; digits=2))", vmax_pos, vmin_neg))
draw_ice && println(@sprintf(
    "ice species max (g/m³) over the run: i1(planar) %.4f, i2(columnar) %.4f, i3(aggregate) %.4f",
    imax1, imax2, imax3))
is_rad && println(@sprintf(
    "radiation: net heating range ±%.2f K/day, OLR range %.1f..%.1f W/m²",
    rad_cmax, olr_lo, olr_hi))

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

# Full-domain x span from the data (the last mish point sits inside the last cell, so round
# the far edge up to the next 10 km): o01_rainfall is 150 km, ocean_warm_bubble 300 km.
xspan = xlim === nothing ?
    (0.0, 10.0 * ceil(maximum(read_patch(snap_path(p, snap_times[1]), rho_cbar).x[end]
                              for p in 1:npatch) / 10.0)) : xlim
# Case label for the title: the run directory's benchmark name, humanised.
case_label = startswith(basename(dir), "ocean_warm_bubble") ? "Ocean warm bubble" : "O01"

# Positive rain contours read best in a colour that contrasts the fill: white on
# the dark viridis cloud (the original convention, shared by --field ice since it
# fills the same rho_c), black on the light diverging rain map. Negative water is
# always dashed — magenta for rain, cyan for cloud.
rain_pos_color = field == "rho_r" ? :white : :black

conv_note = let bits = String[]
    transform_cfg.ctrans === :none || push!(bits, "ρ_c via $(transform_cfg.ctrans)⁻¹")
    transform_cfg.rtrans === :none || push!(bits, "ρ_r via $(transform_cfg.rtrans)⁻¹")
    draw_ice && transform_cfg.itrans !== :none &&
        push!(bits, "ρ_i via $(transform_cfg.itrans)⁻¹")
    isempty(bits) ? "" : "   [" * join(bits, ", ") * "]"
end

# `--field rad`'s "late frame" PNG deliverable: 90% through the run, so it shows
# established cloud/heating structure rather than the still-quiescent early state.
late_frame_idx = max(1, round(Int, 0.9 * length(snap_times)))
late_frame_path = joinpath(dir, "$(rundir === nothing ?
    "o01_rainfall_$(mode)_$(grid)$(nested ? "_n$(npatch == 5 ? 3 : npatch)" : "")" :
    basename(rstrip(dir, '/')))_$(field)_frame_late.png")

for (i, t) in enumerate(snap_times)
    fig = Figure(size = is_rad ? (1500, 560) : (1100, 460))
    ax = is_rad ? Axis(fig[2, 1], xlabel = "x (km)", ylabel = "z (km)") :
                  Axis(fig[1, 1], xlabel = "x (km)", ylabel = "z (km)")
    ax_olr = is_rad ? Axis(fig[1, 1:4], xlabel = "", ylabel = "OLR (W/m²)") : nothing
    ax_rad = is_rad ? Axis(fig[2, 3], xlabel = "x (km)", ylabel = "z (km)") : nothing
    is_rad && rowsize!(fig.layout, 1, Relative(0.18))   # thin top row, per the S5 spec
    local cf = nothing
    local cf_rad = nothing
    minc = Inf; minr = Inf
    max1 = max2 = max3 = 0.0
    for p in draw_order
        pat = read_patch(snap_path(p, t), rho_cbar)
        minc = min(minc, minimum(pat.rho_c)); minr = min(minr, minimum(pat.rho_r))
        cf = contourf!(ax, pat.x, pat.z, fld(pat)',
                       levels = levels, extendhigh = :auto,
                       extendlow = field == "rho_r" ? :auto : nothing,
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
        # The three ISHMAEL ice-mass species, each its own colour family, drawn LAST so
        # they sit on top of everything else — the point of this view is "where is ice",
        # so it has to read at a glance over the liquid fill and the undershoot contours.
        if draw_ice
            max1 = max(max1, maximum(pat.i1)); max2 = max(max2, maximum(pat.i2))
            max3 = max(max3, maximum(pat.i3))
            isempty(ice_levels1) ||
                contour!(ax, pat.x, pat.z, pat.i1', levels = ice_levels1,
                        color = :dodgerblue, linewidth = 1.3)
            isempty(ice_levels2) ||
                contour!(ax, pat.x, pat.z, pat.i2', levels = ice_levels2,
                        color = :purple, linewidth = 1.3)
            isempty(ice_levels3) ||
                contour!(ax, pat.x, pat.z, pat.i3', levels = ice_levels3,
                        color = :darkorange, linewidth = 1.3)
        end
        # Mark patch seams faintly so the nest layout is legible.
        nested && vlines!(ax, [pat.x[1], pat.x[end]]; color = (:gray, 0.3),
                          linewidth = 0.5)
    end

    # ── Radiation panel (--field rad): net heating on the RIGHT, OLR-vs-x on top.
    # Both read straight off the sidecar mish (Scythe.read_radiation) -- no regridding.
    if is_rad
        for p in draw_order
            rad = Scythe.read_radiation(joinpath(dir, patch_dirs[p]), string(t))
            xr = rad.x ./ 1000.0
            zr = rad.z ./ 1000.0
            net = rad.dT_lw .+ rad.sw_scale .* rad.dT_sw
            cf_rad = contourf!(ax_rad, xr, zr, net,
                               levels = range(-rad_cmax, rad_cmax; length = 41),
                               extendlow = :auto, extendhigh = :auto,
                               colormap = Reverse(:RdBu))   # blue = cooling, red = warming
            lines!(ax_olr, xr, rad.olr, color = :black, linewidth = 1.5)
            nested && vlines!(ax_rad, [xr[1], xr[end]]; color = (:gray, 0.3),
                              linewidth = 0.5)
        end
        xlims!(ax_rad, xspan...); ylims!(ax_rad, zlim...)
        ax_rad.title = "net heating  dT_lw + sw_scale·dT_sw  (blue = cooling)"
        xlims!(ax_olr, xspan...)
        ylims!(ax_olr, olr_lo - 0.05 * abs(olr_lo), olr_hi + 0.05 * abs(olr_hi) + 1.0)
        Colorbar(fig[2, 4], cf_rad, label = "K/day")
    end

    # State the CONVENTION alongside the minima. Under a transform these read 0.000 by
    # construction, and a reader has to be able to tell that from a run that simply had no
    # undershoot. The ice view (and --field rad on an ice run) adds each species' current-
    # frame max alongside them.
    ice_bit = draw_ice ? @sprintf("   max ρ_i1 = %.3f, ρ_i2 = %.3f, ρ_i3 = %.3f g/m³",
                                  max1, max2, max3) : ""
    # Name the ARM, not the warm-rain default: an ice run's frames said "warm rain".
    title_str = @sprintf("%s — t = %d min    min ρ_c = %.3f, min ρ_r = %.3f g/m³%s%s",
                         ice_present ? "$case_label ice" : "$case_label warm rain",
                         round(Int, t / 60), minc, minr, ice_bit, conv_note)
    if is_rad
        ax_olr.title = title_str
    else
        ax.title = title_str
    end
    xlims!(ax, xspan...); ylims!(ax, zlim...)
    # The colorbar label carries the quantity AND the contour legend. Keep the two
    # separable: under --field rad the fill colorbar sits in row 2, which is only ~70% of
    # the figure height, and the ice legend rotated vertically is taller than that — it
    # ran up into row 1 and struck through the OLR panel's title. There, label the bar
    # with the quantity alone and hang the legend off the cross-section's own title.
    lbl_q = field == "rho_r" ? "ρ_r (g/m³)" : "ρ_c (g/m³)"
    lbl_legend = if field == "rho_r"
        "[blue<0; lines ρ_r>0; dashed magenta ρ_r<0, cyan ρ_c<0]"
    elseif draw_ice
        "[lines: ρ_r>0 black; ice mass i1 planar dodgerblue, i2 columnar " *
        "purple, i3 aggregate darkorange; dashed magenta ρ_r<0, cyan ρ_c<0]"
    else
        "[lines: ρ_r>0; dashed magenta ρ_r<0, cyan ρ_c<0]"
    end
    if is_rad
        ax.title = lbl_legend
        ax.titlesize = 11
    end
    Colorbar(fig[is_rad ? 2 : 1, 2], cf,
             label = is_rad ? lbl_q : lbl_q * "   " * lbl_legend)
    frame = joinpath(framedir, "frame_" * lpad(i - 1, 4, '0') * ".png")
    save(frame, fig)
    is_rad && i == late_frame_idx && save(late_frame_path, fig)
    i % 10 == 0 && println("  frame $i / $(length(snap_times))")
end
println("Rendered $(length(snap_times)) frames")
is_rad && println("Late frame: $late_frame_path")

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
