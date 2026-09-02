#!/usr/bin/env julia
# Radius-height movie of a nested TC run from the postprocessed derived NetCDF.
#
#   julia --project=. tc/tc_movie.jl [--indir DIR] [--nests n1,n2,...] [--fps 4]
#          [--rmax KM] [--zmax KM] [--vec-scale S] [--w-scale S]
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

# ── Frame rendering ───────────────────────────────────────────────────────────
framedir = joinpath(indir, "movie_frames")
mkpath(framedir)
rm.(joinpath.(framedir, readdir(framedir)); force = true)

function draw_frame(t, framepath)
    fig = Figure(size = (1100, 520))
    ax = Axis(fig[1, 1];
              title = @sprintf("Nested TC — reflectivity / v / (u,w),  t = %.1f h", t / 3600),
              xlabel = "radius (km)", ylabel = "height (km)")
    local cf
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
        end
    end
    for b in boundaries
        vlines!(ax, b; color = :black, linestyle = :dashdot, linewidth = 1.2)
    end
    xlims!(ax, 0, rmax); ylims!(ax, 0, zmax)
    Colorbar(fig[1, 2], cf; label = "reflectivity (dBZ)")
    save(framepath, fig)
end

for (i, t) in enumerate(times)
    frame = joinpath(framedir, "frame_" * lpad(i - 1, 4, '0') * ".png")
    draw_frame(t, frame)
    println("  frame $i/$(length(times))  (t = $(round(t/3600; digits=2)) h)")
end

# ── Assemble movie ────────────────────────────────────────────────────────────
movie = joinpath(indir, "$(basename(indir))_rz.mp4")
run(`ffmpeg -y -loglevel error -framerate $fps -i $(joinpath(framedir, "frame_%04d.png"))
     -c:v libx264 -pix_fmt yuv420p -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" $movie`)
println("Wrote $movie")
