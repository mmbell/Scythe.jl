#!/usr/bin/env julia
# Radius-height movie of a nested TC run from its NetCDF output.
#
#   julia --project=. tc/tc_movie.jl [--indir DIR] [--nests n1,n2,...] [--fps 4]
#          [--rmax KM] [--zmax KM] [--vec-scale S] [--w-scale S] [--rad]
#
# Reads the comprehensive `<t>.nc` the model itself now writes per nest
# (options[:output_formats] default [:netcdf], src/netcdf_output.jl -- global attribute
# `scythe_file_kind = "comprehensive"`) directly: no postprocessing step needed. A run
# made BEFORE that stage (no such attribute on its `<t>.nc`, prognostic primes only)
# falls back to the `<t>_derived.nc` tc/tc_postprocess.jl produces for it, if present;
# with neither, this script errors and names tc/tc_postprocess.jl as the fix. Draws one
# frame per output time, overlaying every nest on a single radius-height axis:
#
#   • reflectivity as a filled contour (shared colour scale, one colorbar)
#   • tangential wind v as line contours (negative dashed)
#   • the (u, w) secondary circulation as vectors
#   • dashed vertical lines at the nest boundaries
#
# The nests are NOT stitched: each is contoured on its own grid, coarsest first so
# the finer inner nests draw on top. The coarser outer nests therefore look
# blockier, and the boundary lines mark where the resolution changes. Frames are
# assembled into an mp4 with ffmpeg. The input directory is configurable and
# defaults to the axisymmetric TC run.
#
# --rad adds the RADIATION view: a thin OLR-vs-radius line panel across the top and,
# beside the reflectivity cross-section, the net radiative heating
# dT_net = dT_lw + sw_scale*dT_sw as a diverging filled contour. A comprehensive file
# carries these fields directly when the run had radiation on (`physics_groups`
# includes it; sidecars are opt-in via options[:radiation_output] and are NOT read
# here); a legacy derived file carries them only if tc/tc_postprocess.jl merged the
# S5 sidecar into it. Both overlay every nest coarsest-first exactly as the main panel
# does. The flag only ADDS: without it the figure, the frames and the movie name are
# what they always were, so the old view stays reproducible. The two views write to
# separate frame directories and separate mp4s (_rz.mp4 vs _rz_rad.mp4), so rendering
# one never clobbers the other.
#
# --bl mirrors --rad for the MYNN-EDMF boundary layer (S9): a thin PBL-height-vs-radius
# line panel across the top (the OLR panel's role) and, beside the reflectivity
# cross-section, K_h -- log-scaled, a boundary-layer diffusivity spans orders of
# magnitude -- as a filled contour. A comprehensive file carries these fields directly
# when the run had `:mynn => true` on; a legacy derived file needs tc/tc_postprocess.jl
# to have merged the S9 sidecar. Writes to its own frame directory and its own mp4
# (_rz_bl.mp4), so it never clobbers --rad or the plain view, and --rad/--bl are
# mutually exclusive (one movie, one extra view).

using NCDatasets
using CairoMakie
using Printf

# ── Arguments ────────────────────────────────────────────────────────────────
indir = joinpath(@__DIR__, "output", "tc_axisym")
nests = String[]
fps = 4
rmax = nothing         # [km] outer radius of the plot (default: outermost nest)
zmax = 20.0            # [km] model top is 25 km; the storm layer is below ~18 km
vec_scale = 2.0        # arrow length = vec_scale * speed[m/s], in km of the r-axis
w_scale = 5.0          # vertical velocity exaggeration for the arrows only
showrad = false        # --rad: add the OLR line panel and the dT_net cross-section
showbl = false         # --bl: add the PBL-height line panel and the K_h cross-section
let i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--indir";       global indir = ARGS[i+1];  i += 2
        elseif a == "--nests";   global nests = String.(split(ARGS[i+1], ",")); i += 2
        elseif a == "--fps";     global fps = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--rmax";    global rmax = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--zmax";    global zmax = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--vec-scale"; global vec_scale = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--w-scale"; global w_scale = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--rad";     global showrad = true; i += 1
        elseif a == "--bl";      global showbl = true; i += 1
        else error("Unknown argument: $a")
        end
    end
end
showrad && showbl && error("--rad and --bl are mutually exclusive (one extra view per movie)")
isdir(indir) || error("Input directory not found: $indir")

# A raw snapshot name: the whole name is a number plus the extension (tc/tc_postprocess.jl's
# RAW_SNAPSHOT_RE, kept in sync by inspection -- both scripts need the same "is this a
# snapshot, not a sidecar or a derived file" test).
const RAW_SNAPSHOT_RE = r"^[0-9]+(\.[0-9]+)?\.nc$"
snaptime_raw(f) = parse(Float64, replace(f, ".nc" => ""))
snaptime_derived(f) = parse(Float64, replace(f, "_derived.nc" => ""))

"""Does `path` carry the model's own `scythe_file_kind = "comprehensive"` global
attribute? False for both a legacy prognostic-only raw snapshot (no such attribute) and
a `<t>_derived.nc` (tc/tc_postprocess.jl's own `source` attribute, not this one)."""
iscomprehensive(path) = NCDataset(path, "r") do ds
    get(ds.attrib, "scythe_file_kind", nothing) == "comprehensive"
end

# Auto-detect nests with either a comprehensive snapshot or a legacy derived file.
if isempty(nests)
    for d in sort(readdir(indir))
        full = joinpath(indir, d)
        isdir(full) || continue
        files = readdir(full)
        has_snap = any(f -> occursin(RAW_SNAPSHOT_RE, f), files)
        has_derived = any(f -> endswith(f, "_derived.nc"), files)
        (has_snap || has_derived) && push!(nests, d)
    end
    isempty(nests) &&
        error("No nests with a comprehensive <t>.nc or a legacy <t>_derived.nc under $indir")
end

# ── Catalogue snapshots: {time => Dict(nest => path)} ─────────────────────────
# A comprehensive `<t>.nc` is read directly and wins over a same-time derived file (the
# derived file would just be a stale re-derivation of the same run). A legacy raw
# snapshot (no scythe_file_kind attribute) needs its `<t>_derived.nc` companion --
# without one this errors immediately rather than silently dropping that time from
# every downstream nest-intersection.
catalog = Dict{Float64,Dict{String,String}}()
for nest in nests
    ndir = joinpath(indir, nest)
    files = readdir(ndir)
    for f in filter(x -> occursin(RAW_SNAPSHOT_RE, x), files)
        path = joinpath(ndir, f)
        t = snaptime_raw(f)
        if iscomprehensive(path)
            get!(catalog, t, Dict{String,String}())[nest] = path
        else
            dpath = joinpath(ndir, replace(f, ".nc" => "_derived.nc"))
            isfile(dpath) ||
                error("$path is a legacy prognostic-only snapshot (no " *
                      "scythe_file_kind = \"comprehensive\" attribute) with no " *
                      "$(basename(dpath)) alongside it. Run tc/tc_postprocess.jl " *
                      "--indir $indir on this old run first.")
            get!(catalog, t, Dict{String,String}())[nest] = dpath
        end
    end
    # Any standalone *_derived.nc not already reached via its raw snapshot above.
    for f in filter(x -> endswith(x, "_derived.nc"), files)
        t = snaptime_derived(f)
        d = get!(catalog, t, Dict{String,String}())
        haskey(d, nest) || (d[nest] = joinpath(ndir, f))
    end
end
# Keep only times present in every requested nest, sorted.
times = sort([t for (t, d) in catalog if length(d) == length(nests)])
isempty(times) && error("No output times common to all nests $(nests)")
println("Movie: $(length(nests)) nests, $(length(times)) frames from $indir")

# ── Nest geometry: read once (grids are static) ──────────────────────────────
# For each nest keep r [km], z [km], and its outer edge (the boundary marker).
geom = Dict{String,NamedTuple}()
for nest in nests
    NCDataset(catalog[times[1]][nest], "r") do ds
        r = ds["x"][:] ./ 1000.0; z = ds["z"][:] ./ 1000.0
        geom[nest] = (; r, z, redge = r[end])
    end
end
# Order nests coarsest→finest by radial spacing so the finest draws last (on top).
draw_order = sort(nests, by = n -> -(geom[n].r[2] - geom[n].r[1]))
rmax === nothing && (rmax = maximum(geom[n].redge for n in nests))
# Boundary lines = outer edge of every nest except the outermost one.
boundaries = sort([geom[n].redge for n in nests if geom[n].redge < rmax - 1e-6])

readfield(ds, name) = coalesce.(Array(ds[name])[1, :, :], NaN)   # (n_r, n_z) → NaN fill

"""Error text for a `--rad`/`--bl` view that finds `path` missing `varname`. A
comprehensive file (has scythe_file_kind attribute) simply had that physics group off --
point at `physics_groups` rather than at tc/tc_postprocess.jl, which such a file never
needs. A legacy derived file keeps the old wording: either the run had it off, or the
postprocessor ran before the sidecar merge existed."""
function missing_physics_error(path, varname, flag, groupname)
    kind, pg = NCDataset(path, "r") do ds
        (get(ds.attrib, "scythe_file_kind", nothing), get(ds.attrib, "physics_groups", "none"))
    end
    if kind == "comprehensive"
        error("$flag needs `$varname`, and $path has none (physics_groups = " *
              "\"$pg\"). This run had $groupname off.")
    else
        error("$flag needs the $groupname sidecar merged into the derived files, and " *
              "$path has no $varname. Either the run was made with $groupname off, or " *
              "tc/tc_postprocess.jl was run on it before the merge existed — re-run " *
              "tc/tc_postprocess.jl --indir $indir.")
    end
end

"""`true` when `ds` carries `varname`; `false` when this is a comprehensive frame written
before the scheme's first call (the t = 0 file: `physics_groups` says the schemes have not
run yet), which is drawn without the overlay; otherwise the run had the scheme off and
`missing_physics_error` fires."""
function physics_available(ds, path, varname, flag, groupname)
    haskey(ds, varname) && return true
    pg = get(ds.attrib, "physics_groups", "none")
    startswith(pg, "none (schemes not yet run)") && return false
    missing_physics_error(path, varname, flag, groupname)
end

# ── Fixed contour/colour scales (shared across nests and frames) ──────────────
refl_levels = -15.0:5.0:60.0                 # dBZ
p_levels = -1500.0:50:250.0
u_levels = -10.0:1.0:5.0
v_levels    = vcat(-40.0:5.0:-5.0, 5.0:5.0:40.0)   # m/s (0 omitted)

# ── Radiation scales (--rad only): one pass over every frame and nest ─────────
# Both scales are fixed for the WHOLE movie, so a frame's colours and the OLR line's
# height mean the same thing throughout and the eye can read change rather than
# rescaling. The heating limit is the 99th percentile of |dT_net| rather than its max:
# the maximum is set by a handful of points at a sharp cloud top, and keying the colour
# range to those flattens the entire rest of the field to white.
rad_cmax = 0.1
olr_lo = Inf; olr_hi = -Inf
if showrad
    absnet = Float64[]
    for t in times, nest in nests
        NCDataset(catalog[t][nest], "r") do ds
            physics_available(ds, catalog[t][nest], "dT_net", "--rad", "radiation") ||
                return
            net = readfield(ds, "dT_net")
            append!(absnet, abs.(filter(isfinite, vec(net))))
            fo = filter(isfinite, coalesce.(Array(ds["olr"])[1, :], NaN))
            if !isempty(fo)
                global olr_lo = min(olr_lo, minimum(fo))
                global olr_hi = max(olr_hi, maximum(fo))
            end
        end
    end
    sort!(absnet)
    isempty(absnet) ||
        (rad_cmax = max(absnet[max(1, ceil(Int, 0.99 * length(absnet)))], 0.1))
    if !isfinite(olr_lo)
        olr_lo = 0.0; olr_hi = 300.0
    end
    println(@sprintf("  radiation: net heating ±%.2f K/day (99th pct), OLR %.1f..%.1f W/m²",
                     rad_cmax, olr_lo, olr_hi))
end
rad_levels = range(-rad_cmax, rad_cmax; length = 41)

# ── MYNN scales (--bl only): one pass over every frame and nest ───────────────
# K_h's log-scale range (a positive floor for the axis, and the domain max) and the
# PBL-height line panel's y-range, both fixed for the whole movie -- the `showrad`
# pre-pass above, mirrored.
kh_floor = 1.0e-3       # m^2/s -- below this reads as "no mixing" on the log scale
kh_max = kh_floor
pblh_lo = Inf; pblh_hi = -Inf
if showbl
    for t in times, nest in nests
        NCDataset(catalog[t][nest], "r") do ds
            physics_available(ds, catalog[t][nest], "K_h", "--bl", "MYNN") ||
                return
            Kh = readfield(ds, "K_h")
            global kh_max = max(kh_max, maximum(x -> isnan(x) ? -Inf : x, Kh))
            fp = filter(isfinite, coalesce.(Array(ds["mynn_pblh"])[1, :], NaN))
            if !isempty(fp)
                global pblh_lo = min(pblh_lo, minimum(fp))
                global pblh_hi = max(pblh_hi, maximum(fp))
            end
        end
    end
    kh_max = max(kh_max, 10.0 * kh_floor)   # widen a decade if K_h is identically ~0
    if !isfinite(pblh_lo)
        pblh_lo = 0.0; pblh_hi = 1000.0
    end
    println(@sprintf("  MYNN: K_h range %.1e..%.2f m²/s (log scale), PBLH %.0f..%.0f m",
                     kh_floor, kh_max, pblh_lo, pblh_hi))
end
kh_levels = 10.0 .^ range(log10(kh_floor), log10(kh_max); length = 31)

# ── Frame rendering ───────────────────────────────────────────────────────────
framedir = joinpath(indir, showrad ? "movie_frames_rad" : showbl ? "movie_frames_bl" :
                          "movie_frames")
mkpath(framedir)
rm.(joinpath.(framedir, readdir(framedir)); force = true)

function draw_frame(t, framepath)
    # --rad/--bl grow the figure to two rows: a thin line-diagnostic strip across the top
    # and, in row 2, the reflectivity cross-section (col 1, colorbar col 2) beside the
    # extra cross-section (col 3, colorbar col 4). Without either the layout is the
    # original single-axis figure, unchanged.
    extra = showrad || showbl
    fig = Figure(size = extra ? (1500, 620) : (1100, 520))
    title_str = @sprintf("Nested TC — reflectivity / v / (u,w),  t = %.1f h", t / 3600)
    ax = extra ? Axis(fig[2, 1]; xlabel = "radius (km)", ylabel = "height (km)") :
                 Axis(fig[1, 1]; title = title_str,
                      xlabel = "radius (km)", ylabel = "height (km)")
    ax_line = showrad ? Axis(fig[1, 1:4]; title = title_str, ylabel = "OLR (W/m²)") :
              showbl ? Axis(fig[1, 1:4]; title = title_str, ylabel = "PBL height (m)") :
              nothing
    ax_rad = showrad ? Axis(fig[2, 3]; xlabel = "radius (km)", ylabel = "height (km)",
                            title = "net radiative heating  dT_lw + sw_scale·dT_sw") :
                       nothing
    ax_bl = showbl ? Axis(fig[2, 3]; xlabel = "radius (km)", ylabel = "height (km)",
                          title = "K_h (log scale)") :
                     nothing
    extra && rowsize!(fig.layout, 1, Relative(0.20))
    local cf
    local cf_rad = nothing
    local cf_bl = nothing
    for nest in draw_order
        g = geom[nest]
        NCDataset(catalog[t][nest], "r") do ds
            refl = readfield(ds, "reflectivity")
            p_prime = readfield(ds, "p_prime")
            v    = readfield(ds, "v")
            u    = readfield(ds, "u")
            w    = readfield(ds, "w")

            # Reflectivity fill (NaN below the echo floor → transparent)
            cf = contourf!(ax, g.r, g.z, refl; levels = refl_levels,
                           colormap = :turbo, extendhigh = :auto)
            #cf = contourf!(ax, g.r, g.z, p_prime; levels = p_levels,
            #               colormap = :turbo, extendhigh = :auto)
            #cf = contourf!(ax, g.r, g.z, u; levels = u_levels,
            #               colormap = :turbo, extendhigh = :auto)
            # Tangential wind line contours (negative dashed, as in the papers)
            contour!(ax, g.r, g.z, v; levels = filter(l -> l > 0, v_levels),
                     color = :black, linewidth = 0.8)
            contour!(ax, g.r, g.z, v; levels = filter(l -> l < 0, v_levels),
                     color = :black, linewidth = 0.8, linestyle = :dash)

            # (u,w) vectors, subsampled to ~20 km radial / ~1.5 km vertical spacing
            dr = g.r[2] - g.r[1]; dz = g.z[2] - g.z[1]
            si = max(1, round(Int, 20.0 / dr)); sk = max(1, round(Int, 1.5 / dz))
            px = Float64[]; py = Float64[]; pu = Float64[]; pw = Float64[]
            for i in 1:si:length(g.r), k in 1:sk:length(g.z)
                (g.r[i] > rmax || g.z[k] > zmax) && continue
                (isnan(u[i, k]) || isnan(w[i, k])) && continue
                push!(px, g.r[i]); push!(py, g.z[k])
                push!(pu, u[i, k] * vec_scale); push!(pw, w[i, k] * vec_scale * w_scale)
            end
            isempty(px) || arrows2d!(ax, Point2f.(px, py), Vec2f.(pu, pw);
                                     lengthscale = 0.5, shaftwidth = 1.0,
                                     tipwidth = 6.0, tiplength = 5.0,
                                     color = (:gray20, 0.7))

            # Radiation, from the merged sidecar fields in the same derived file. Same
            # coarsest-first overlay as everything else, so the fine nests draw on top.
            if showrad && haskey(ds, "dT_net")   # absent only on the t = 0 frame
                net = readfield(ds, "dT_net")
                cf_rad = contourf!(ax_rad, g.r, g.z, net; levels = rad_levels,
                                   extendlow = :auto, extendhigh = :auto,
                                   colormap = Reverse(:RdBu))  # blue cools, red warms
                lines!(ax_line, g.r, coalesce.(Array(ds["olr"])[1, :], NaN);
                       color = :black, linewidth = 1.4)
            end

            # MYNN, from the merged sidecar fields in the same derived file. Same
            # coarsest-first overlay as everything else, so the fine nests draw on top.
            if showbl && haskey(ds, "K_h")       # absent only on the t = 0 frame
                Kh = max.(readfield(ds, "K_h"), kh_floor)   # floored for log10
                cf_bl = contourf!(ax_bl, g.r, g.z, Kh; levels = kh_levels,
                                  colorscale = log10, extendlow = :auto, extendhigh = :auto,
                                  colormap = :viridis)
                lines!(ax_line, g.r, coalesce.(Array(ds["mynn_pblh"])[1, :], NaN);
                       color = :black, linewidth = 1.4)
            end
        end
    end
    for b in boundaries
        vlines!(ax, b; color = :black, linestyle = :dashdot, linewidth = 1.2)
        if showrad
            vlines!(ax_rad, b; color = :black, linestyle = :dash, linewidth = 1.2)
            vlines!(ax_line, b; color = :black, linestyle = :dash, linewidth = 1.0)
        end
        if showbl
            vlines!(ax_bl, b; color = :black, linestyle = :dash, linewidth = 1.2)
            vlines!(ax_line, b; color = :black, linestyle = :dash, linewidth = 1.0)
        end
    end
    xlims!(ax, 0, rmax); ylims!(ax, 0, zmax)
    Colorbar(fig[extra ? 2 : 1, 2], cf; label = "reflectivity (dBZ)")
    if showrad
        xlims!(ax_rad, 0, rmax); ylims!(ax_rad, 0, zmax)
        xlims!(ax_line, 0, rmax)
        pad = 0.05 * max(olr_hi - olr_lo, 1.0)
        ylims!(ax_line, olr_lo - pad, olr_hi + pad)
        cf_rad === nothing || Colorbar(fig[2, 4], cf_rad; label = "K/day")   # t = 0 frame has none
    end
    if showbl
        xlims!(ax_bl, 0, rmax); ylims!(ax_bl, 0, zmax)
        xlims!(ax_line, 0, rmax)
        pad = 0.05 * max(pblh_hi - pblh_lo, 1.0)
        ylims!(ax_line, pblh_lo - pad, pblh_hi + pad)
        cf_bl === nothing || Colorbar(fig[2, 4], cf_bl; label = "K_h (m²/s)")   # t = 0 frame has none
    end
    save(framepath, fig)
end

stem = basename(rstrip(indir, '/')) * "_rz" * (showrad ? "_rad" : showbl ? "_bl" : "")
for (i, t) in enumerate(times)
    frame = joinpath(framedir, "frame_" * lpad(i - 1, 4, '0') * ".png")
    draw_frame(t, frame)
    println("  frame $i/$(length(times))  (t = $(round(t/3600; digits=2)) h)")
    # Keep the LAST frame as a still, outside the frame directory the next render
    # clears: it is the one picture a report wants, and it survives re-runs.
    i == length(times) && cp(frame, joinpath(indir, "$(stem)_late.png"); force = true)
end

# ── Assemble movie ────────────────────────────────────────────────────────────
movie = joinpath(indir, "$(stem).mp4")
run(`ffmpeg -y -loglevel error -framerate $fps -i $(joinpath(framedir, "frame_%04d.png"))
     -c:v libx264 -pix_fmt yuv420p -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" $movie`)
println("Wrote $movie")
