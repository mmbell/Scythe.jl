#!/usr/bin/env julia
# Movie of a 3-D ocean warm bubble run on the RRR box, from its comprehensive NetCDF output
# (close-out Stream C2, stage R2e).
#
#   julia --project=. simulations/ocean_warm_bubble/examples/owb3d_movie.jl --root <run root>
#          [--z 1000] [--fps 4] [--yc <m>]
#
# (--project=., the root Scythe project: it carries CairoMakie and NCDatasets.) `--root` is a
# run root written by run.jl for an RRR config (rrr_120km.toml beside this file, or
# ../../ocean_warm_bubble_3d/scythe_config.toml): `<root>/seg_NN/<t>.nc`, stitched onto one
# time axis through `<root>/segments.csv` (t = t_offset_s + t_local; each later segment's
# duplicate t = 0 dropped). Every `<t>.nc` must be the comprehensive file
# (`scythe_file_kind = "comprehensive"`, dims time, x, y, z; src/netcdf_output.jl), which
# carries T, reflectivity and the surface rain rate. One frame per output time:
#
#   left   horizontal slice at height --z (default 1 km): reflectivity (filled), w (lines,
#          negative dashed) and the surface rain rate (white contours)
#   right  vertical x-z slice through the bubble centre (the y row nearest --yc, default the
#          bubble's yc from the run's scythe_config.toml): reflectivity (filled), w (lines)
#
# Frames go to <root>/owb3d_frames/, the last frame is kept as <root>/owb3d_late.png and the
# frames are assembled into <root>/owb3d.mp4 with ffmpeg (as tc_axisym/tc_movie.jl does).

using NCDatasets
using CairoMakie
using Printf
using TOML

"A variable as a plain Float64 array (fill values -> NaN), for Makie."
nc(ds, name) = Float64.(coalesce.(Array(ds[name]), NaN))

function parse_args(args)
    o = Dict{String,Any}("root" => "", "z" => 1000.0, "fps" => 4, "yc" => NaN)
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--root"; o["root"] = args[i + 1]
        elseif a == "--z"; o["z"] = parse(Float64, args[i + 1])
        elseif a == "--fps"; o["fps"] = parse(Int, args[i + 1])
        elseif a == "--yc"; o["yc"] = parse(Float64, args[i + 1])
        else error("unknown argument $a; usage: owb3d_movie.jl --root <run root> [--z m] [--fps n] [--yc m]")
        end
        i += 2
    end
    isempty(o["root"]) && error("--root <run root> is required")
    return o
end

"(time [s], file) for every comprehensive snapshot of the run, on one time axis."
function snapshots(root)
    segs = Tuple{Float64,String}[]
    csvp = joinpath(root, "segments.csv")
    if isfile(csvp)
        lines = readlines(csvp)
        hdr = split(lines[1], ",")
        id = findfirst(==("dir"), hdr); io = findfirst(==("t_offset_s"), hdr)
        for l in lines[2:end]
            f = split(l, ",")
            push!(segs, (parse(Float64, f[io]), String(f[id])))
        end
    else
        for d in sort(filter(d -> startswith(d, "seg_"), readdir(root)))
            push!(segs, (0.0, d))
        end
    end
    out = Tuple{Float64,String}[]
    for (n, (off, d)) in enumerate(segs)
        dir = isabspath(d) ? d : joinpath(root, d)
        isdir(dir) || continue
        for f in readdir(dir)
            m = match(r"^([0-9.]+)\.nc$", f)
            m === nothing && continue
            tl = parse(Float64, m.captures[1])
            (n > 1 && tl == 0.0) && continue
            push!(out, (off + tl, joinpath(dir, f)))
        end
    end
    sort!(out; by = first)
    return out
end

o = parse_args(ARGS)
root = abspath(o["root"])
snaps = snapshots(root)
isempty(snaps) && error("no <t>.nc under $root")
yc = o["yc"]
if isnan(yc)
    cfgp = joinpath(root, "scythe_config.toml")
    c = isfile(cfgp) ? TOML.parsefile(cfgp) : Dict()
    yc = Float64(get(get(get(c, "initial_conditions", Dict()), "temperature_bubble", Dict()), "yc", NaN))
end

x = y = z = nothing
zk = jy = 0
NCDataset(last(snaps[1]), "r") do ds
    get(ds.attrib, "scythe_file_kind", "") == "comprehensive" ||
        error("$(last(snaps[1])) is not a comprehensive file (needs T and reflectivity)")
    haskey(ds.dim, "y") || error("$(last(snaps[1])) has no y dimension: not an RRR box file")
    global x = nc(ds, "x") ./ 1e3; global y = nc(ds, "y") ./ 1e3; global z = nc(ds, "z")
end
isnan(yc) && (yc = 1e3 * (y[1] + y[end]) / 2)
zk = argmin(abs.(z .- o["z"]))
jy = argmin(abs.(1e3 .* y .- yc))
println("owb3d_movie: $(length(snaps)) frames; horizontal slice z = $(z[zk]) m, vertical slice y = $(y[jy]) km")

framedir = joinpath(root, "owb3d_frames")
mkpath(framedir)
foreach(f -> rm(joinpath(framedir, f)), filter(f -> endswith(f, ".png"), readdir(framedir)))
dbz_levels = -10:5:65

function frame(t, path, out)
    NCDataset(path, "r") do ds
        dbz = nc(ds, "reflectivity")[1, :, :, :]
        w = nc(ds, "w")[1, :, :, :]
        pr = nc(ds, "precip_rate")[1, :, :]
        dz(a) = replace(a, NaN => -30.0)
        fig = Figure(size = (1400, 600))
        ax1 = Axis(fig[1, 1]; title = @sprintf("z = %.1f km, t = %.1f min", z[zk] / 1e3, t / 60),
                   xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect())
        cf = contourf!(ax1, x, y, dz(dbz[:, :, zk]); levels = dbz_levels, colormap = :turbo,
                       extendlow = :auto)
        wl = maximum(abs, w[:, :, zk]) > 0.05 ? (-5:0.5:5) : (-0.5:0.05:0.5)
        contour!(ax1, x, y, w[:, :, zk]; levels = filter(>(0), collect(wl)), color = :black, linewidth = 0.8)
        contour!(ax1, x, y, w[:, :, zk]; levels = filter(<(0), collect(wl)), color = :black,
                 linewidth = 0.8, linestyle = :dash)
        maximum(pr) > 0.1 && contour!(ax1, x, y, pr; levels = [0.1, 1, 10, 50], color = :white, linewidth = 1.5)
        hlines!(ax1, [y[jy]]; color = :gray, linestyle = :dot)
        ax2 = Axis(fig[1, 2]; title = @sprintf("y = %.1f km (max w %.2f m/s, max sfc rain %.2f mm/h)",
                                              y[jy], maximum(w), maximum(pr)),
                   xlabel = "x (km)", ylabel = "z (km)")
        contourf!(ax2, x, z ./ 1e3, dz(dbz[:, jy, :]); levels = dbz_levels, colormap = :turbo,
                  extendlow = :auto)
        ws = w[:, jy, :]
        wl2 = maximum(abs, ws) > 0.05 ? (-20:1:20) : (-0.5:0.05:0.5)
        contour!(ax2, x, z ./ 1e3, ws; levels = filter(>(0), collect(wl2)), color = :black, linewidth = 0.8)
        contour!(ax2, x, z ./ 1e3, ws; levels = filter(<(0), collect(wl2)), color = :black,
                 linewidth = 0.8, linestyle = :dash)
        Colorbar(fig[1, end + 1], cf; label = "reflectivity (dBZ)")
        save(out, fig)
    end
end

for (i, (t, path)) in enumerate(snaps)
    out = joinpath(framedir, "frame_" * lpad(i - 1, 4, '0') * ".png")
    frame(t, path, out)
    println("  frame $i/$(length(snaps))  (t = $(round(t / 60; digits = 1)) min)")
    i == length(snaps) && cp(out, joinpath(root, "owb3d_late.png"); force = true)
end
movie = joinpath(root, "owb3d.mp4")
run(`ffmpeg -y -loglevel error -framerate $(o["fps"]) -i $(joinpath(framedir, "frame_%04d.png"))
     -c:v libx264 -pix_fmt yuv420p -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" $movie`)
println("Wrote $movie")
