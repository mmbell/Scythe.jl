#!/usr/bin/env julia
# Radius-height movie of a nested TC run from the postprocessed derived NetCDF.
#
#   julia --project=. tc/tc_movie.jl [--indir DIR] [--nests n1,n2,...] [--fps 4]
#          [--rmax KM] [--zmax KM] [--vec-scale S] [--w-scale S] [--rad]
#
# Reads <t>_derived.nc from each nest (produced by tc/tc_postprocess.jl) and draws
# one frame per output time, overlaying every nest on a single radius-height axis:
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
# --rad adds the RADIATION view, which needs a run whose derived files carry the
# merged radiation sidecar (tc/tc_postprocess.jl on a run made with SCYTHE_TC_RAD):
# a thin OLR-vs-radius line panel across the top and, beside the reflectivity
# cross-section, the net radiative heating dT_net = dT_lw + sw_scale*dT_sw as a
# diverging filled contour. Both overlay every nest coarsest-first exactly as the
# main panel does. The flag only ADDS: without it the figure, the frames and the
# movie name are what they always were, so the old view stays reproducible. The two
# views write to separate frame directories and separate mp4s (_rz.mp4 vs
# _rz_rad.mp4), so rendering one never clobbers the other.

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
        else error("Unknown argument: $a")
        end
    end
end
isdir(indir) || error("Input directory not found: $indir")

# Auto-detect nests with derived snapshots.
if isempty(nests)
    for d in sort(readdir(indir))
        full = joinpath(indir, d)
        isdir(full) && any(f -> endswith(f, "_derived.nc"), readdir(full)) && push!(nests, d)
    end
    isempty(nests) &&
        error("No nests with *_derived.nc under $indir — run tc/tc_postprocess.jl first")
end

# ── Catalogue snapshots: {time => Dict(nest => path)} ─────────────────────────
snaptime(f) = parse(Float64, replace(f, "_derived.nc" => ""))
catalog = Dict{Float64,Dict{String,String}}()
for nest in nests
    ndir = joinpath(indir, nest)
    for f in filter(x -> endswith(x, "_derived.nc"), readdir(ndir))
        t = snaptime(f)
        get!(catalog, t, Dict{String,String}())[nest] = joinpath(ndir, f)
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
            haskey(ds, "dT_net") || error(
                "--rad needs the radiation sidecar merged into the derived files, and " *
                "$(catalog[t][nest]) has no dT_net. Either the run was made with " *
                "radiation off, or tc/tc_postprocess.jl was run on it before the merge " *
                "existed — re-run tc/tc_postprocess.jl --indir $indir.")
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

# ── Frame rendering ───────────────────────────────────────────────────────────
framedir = joinpath(indir, showrad ? "movie_frames_rad" : "movie_frames")
mkpath(framedir)
rm.(joinpath.(framedir, readdir(framedir)); force = true)

function draw_frame(t, framepath)
    # --rad grows the figure to two rows: a thin OLR-vs-r strip across the top and, in
    # row 2, the reflectivity cross-section (col 1, colorbar col 2) beside the net-heating
    # cross-section (col 3, colorbar col 4). Without it the layout is the original
    # single-axis figure, unchanged.
    fig = Figure(size = showrad ? (1500, 620) : (1100, 520))
    title_str = @sprintf("Nested TC — reflectivity / v / (u,w),  t = %.1f h", t / 3600)
    ax = showrad ? Axis(fig[2, 1]; xlabel = "radius (km)", ylabel = "height (km)") :
                   Axis(fig[1, 1]; title = title_str,
                        xlabel = "radius (km)", ylabel = "height (km)")
    ax_olr = showrad ? Axis(fig[1, 1:4]; title = title_str, ylabel = "OLR (W/m²)") : nothing
    ax_rad = showrad ? Axis(fig[2, 3]; xlabel = "radius (km)", ylabel = "height (km)",
                            title = "net radiative heating  dT_lw + sw_scale·dT_sw") :
                       nothing
    showrad && rowsize!(fig.layout, 1, Relative(0.20))
    local cf
    local cf_rad = nothing
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
            if showrad
                net = readfield(ds, "dT_net")
                cf_rad = contourf!(ax_rad, g.r, g.z, net; levels = rad_levels,
                                   extendlow = :auto, extendhigh = :auto,
                                   colormap = Reverse(:RdBu))  # blue cools, red warms
                lines!(ax_olr, g.r, coalesce.(Array(ds["olr"])[1, :], NaN);
                       color = :black, linewidth = 1.4)
            end
        end
    end
    for b in boundaries
        vlines!(ax, b; color = :black, linestyle = :dashdot, linewidth = 1.2)
        if showrad
            vlines!(ax_rad, b; color = :black, linestyle = :dash, linewidth = 1.2)
            vlines!(ax_olr, b; color = :black, linestyle = :dash, linewidth = 1.0)
        end
    end
    xlims!(ax, 0, rmax); ylims!(ax, 0, zmax)
    Colorbar(fig[showrad ? 2 : 1, 2], cf; label = "reflectivity (dBZ)")
    if showrad
        xlims!(ax_rad, 0, rmax); ylims!(ax_rad, 0, zmax)
        xlims!(ax_olr, 0, rmax)
        pad = 0.05 * max(olr_hi - olr_lo, 1.0)
        ylims!(ax_olr, olr_lo - pad, olr_hi + pad)
        Colorbar(fig[2, 4], cf_rad; label = "K/day")
    end
    save(framepath, fig)
end

stem = basename(rstrip(indir, '/')) * "_rz" * (showrad ? "_rad" : "")
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
