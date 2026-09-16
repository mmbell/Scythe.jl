#!/usr/bin/env julia
# x-z movie of an benchmarks/ocean_rce.jl output tree: one frame per stitched output
# time (or every `n`-th, with `--every n`), reading ONLY the comprehensive `<t>.nc` files
# the model itself writes (src/netcdf_output.jl). Never runs the model.
#
#   julia --project=. benchmarks/rce_movie.jl --dir <root> [--out <root>/rce.mp4]
#         [--every n] [--fps 4]
#
# Four x-z panels per frame:
#   - potential-temperature anomaly: theta(x,z) - mean_x(theta(z)) FOR THIS FRAME (a
#     per-frame domain-mean removal, not a departure from the reference-state Tbar-
#     derived theta -- simpler, and what's needed to see convective cells/cold pools;
#     stated here since the plan allowed either).
#   - condensate MASS variables only, one flat colour per species (the user's standing
#     rule for an ice view): cloud liquid (rho_c), rain (rho_r), and each ISHMAEL ice
#     species (rho_i1/2/3) as its own colour, overlaid.
#   - vertical velocity w.
#   - relative humidity RH.
# Across the top, a strip with the FULL (not --every-subsampled) domain-mean time series
# of precip_rate and olr, each frame showing the series up to that frame's time.
#
# A t=0 file (physics_groups = "none (schemes not yet run)") has no radiation group: its
# frame draws the four main panels from the always-present regular-grid fields and simply
# shows a flat/empty olr line up to that point (NaN, not plotted) rather than erroring.

using CairoMakie
using Printf
using Statistics

include(joinpath(@__DIR__, "common", "rce_io.jl"))

# ── CLI ──────────────────────────────────────────────────────────────────────
root = nothing
out = nothing
every = 1
fps = 4
let i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--dir"; global root = ARGS[i + 1]; i += 2
        elseif a == "--out"; global out = ARGS[i + 1]; i += 2
        elseif a == "--every"; global every = parse(Int, ARGS[i + 1]); i += 2
        elseif a == "--fps"; global fps = parse(Int, ARGS[i + 1]); i += 2
        elseif a == "--fields"; i += 2   # accepted, currently a no-op: the four panels
                                         # below are fixed by the plan; kept as a CLI
                                         # stub so a caller passing it doesn't error.
        else
            error("rce_movie.jl: unrecognized argument '$a'. Usage: --dir <root> " *
                  "[--out path] [--every n] [--fps N] [--fields ...]")
        end
    end
end
root === nothing && error("rce_movie.jl: --dir <root> is required")
isdir(root) || error("rce_movie.jl: --dir $root is not a directory")
out === nothing && (out = joinpath(root, "rce.mp4"))

const t0 = time()

snaps = catalog_snapshots(root)
println("rce_movie: $(length(snaps)) snapshots under $root")

readxz(ds, name) = field_xz(ds, name)   # (x, z)

# ── Prepass: full-resolution time series (precip, olr) for the top strip, and global
# color scales for theta-anomaly and w (99th percentile of |value|, fixed for the whole
# movie so colour means the same thing frame to frame -- tc/tc_movie.jl's convention). ──
t_h = Float64[]; precip_ts = Float64[]; olr_ts = Float64[]
abs_theta_anom = Float64[]; abs_w = Float64[]
x_km = nothing; z_km = nothing; interior = nothing
for snap in snaps
    NCDataset(snap.path, "r") do ds
        if x_km === nothing
            x, z, itr = xz_coords(ds)
            global x_km = x ./ 1000.0; global z_km = z ./ 1000.0; global interior = itr
        end
        push!(t_h, snap.t_abs / 3600.0)
        push!(precip_ts, xmean(field_x(ds, "precip_rate"), interior))
        push!(olr_ts, has_group(ds, "radiation") ? xmean(field_x(ds, "olr"), interior) : NaN)
        theta = readxz(ds, "theta")
        anom = theta .- mean(theta[interior, :]; dims = 1)
        append!(abs_theta_anom, abs.(vec(anom[interior, :])))
        w = readxz(ds, "w")
        append!(abs_w, abs.(vec(w[interior, :])))
    end
end
sort!(abs_theta_anom); sort!(abs_w)
pct99(v) = isempty(v) ? 1.0 : max(v[max(1, ceil(Int, 0.99 * length(v)))], 1.0e-6)
theta_cmax = pct99(abs_theta_anom)
w_cmax = pct99(abs_w)
theta_levels = range(-theta_cmax, theta_cmax; length = 31)
w_levels = range(-w_cmax, w_cmax; length = 31)
rh_levels = range(0.0, 1.2; length = 25)
println(@sprintf("  scales: theta' +-%.2f K, w +-%.2f m/s (99th pct)", theta_cmax, w_cmax))

const SPECIES = (("rho_c", "cloud liquid", :cyan), ("rho_r", "rain", :blue),
                 ("rho_i1", "ice 1 (planar)", :gold), ("rho_i2", "ice 2 (columnar)", :orange),
                 ("rho_i3", "ice 3 (aggregates)", :firebrick))
const MASS_THRESH = 1.0e-5   # kg/m^3

framedir = joinpath(root, "movie_frames_rce")
mkpath(framedir)
rm.(joinpath.(framedir, readdir(framedir)); force = true)

frame_idx = collect(1:every:length(snaps))

function draw_frame(k, fidx, framepath)
    snap = snaps[fidx]
    NCDataset(snap.path, "r") do ds
        title_str = @sprintf("Ocean RCE  t = %.2f h  (segment %d)", snap.t_abs / 3600.0, snap.segment)
        fig = Figure(size = (1300, 820))
        ax_precip = Axis(fig[1, 1:2]; ylabel = "precip (mm/hr)", title = title_str)
        ax_olr = Axis(fig[1, 3:4]; ylabel = "OLR (W/m^2)")
        ax_theta = Axis(fig[2, 1]; xlabel = "x (km)", ylabel = "z (km)", title = "theta' (x-mean removed)")
        ax_cond = Axis(fig[2, 3]; xlabel = "x (km)", ylabel = "z (km)", title = "condensate mass (species)")
        ax_w = Axis(fig[3, 1]; xlabel = "x (km)", ylabel = "z (km)", title = "w (m/s)")
        ax_rh = Axis(fig[3, 3]; xlabel = "x (km)", ylabel = "z (km)", title = "RH")

        lines!(ax_precip, t_h[1:fidx], precip_ts[1:fidx]; color = :black)
        xlims!(ax_precip, t_h[1], t_h[end])
        finite_olr = filter(isfinite, olr_ts)
        if !isempty(finite_olr)
            lines!(ax_olr, t_h[1:fidx], olr_ts[1:fidx]; color = :black)
            ylims!(ax_olr, minimum(finite_olr) - 5, maximum(finite_olr) + 5)
        end
        xlims!(ax_olr, t_h[1], t_h[end])

        theta = readxz(ds, "theta")
        anom = theta .- mean(theta[interior, :]; dims = 1)
        cf_t = contourf!(ax_theta, x_km, z_km, anom; levels = theta_levels,
                         colormap = Reverse(:RdBu), extendlow = :auto, extendhigh = :auto)
        Colorbar(fig[2, 2], cf_t; label = "K")

        for (nm, _, color) in SPECIES
            has_var(ds, nm) || continue
            A = readxz(ds, nm)
            mx = maximum(A)
            mx > MASS_THRESH || continue
            contourf!(ax_cond, x_km, z_km, A; levels = [MASS_THRESH, max(mx, 2 * MASS_THRESH)],
                     colormap = cgrad([color, color]), extendhigh = :auto, alpha = 0.6)
        end
        Legend(fig[2, 4], [PolyElement(color = c) for (_, _, c) in SPECIES],
              [lbl for (_, lbl, _) in SPECIES]; framevisible = false, labelsize = 11)

        w = readxz(ds, "w")
        cf_w = contourf!(ax_w, x_km, z_km, w; levels = w_levels, colormap = Reverse(:RdBu),
                         extendlow = :auto, extendhigh = :auto)
        Colorbar(fig[3, 2], cf_w; label = "m/s")

        rh = readxz(ds, "RH")
        cf_rh = contourf!(ax_rh, x_km, z_km, rh; levels = rh_levels, colormap = :viridis,
                          extendhigh = :auto)
        Colorbar(fig[3, 4], cf_rh; label = "RH")

        for ax in (ax_theta, ax_cond, ax_w, ax_rh)
            xlims!(ax, x_km[1], x_km[end]); ylims!(ax, z_km[1], z_km[end])
        end
        save(framepath, fig)
    end
end

for (k, fidx) in enumerate(frame_idx)
    frame = joinpath(framedir, "frame_" * lpad(k - 1, 4, '0') * ".png")
    draw_frame(k, fidx, frame)
    println("  frame $k/$(length(frame_idx))  (t = $(round(t_h[fidx]; digits = 2)) h)")
    k == length(frame_idx) && cp(frame, joinpath(root, "rce_late.png"); force = true)
end

run(`ffmpeg -y -loglevel error -framerate $fps -i $(joinpath(framedir, "frame_%04d.png"))
     -c:v libx264 -pix_fmt yuv420p -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" $out`)

elapsed = time() - t0
println(@sprintf("rce_movie.jl: wrote %s (%d frames) in %.1f s", out, length(frame_idx), elapsed))
