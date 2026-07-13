#!/usr/bin/env julia
# Cloud-and-rain evolution movie for an existing o01_rainfall run (no re-run):
# a color fill of the diagnosed condensate rho_c with white rho_r contours, one
# frame per output snapshot, assembled with ffmpeg.
#
#   julia --project=. benchmarks/o01_movie.jl [--mode full] [--grid rirk] [--fps 8]
#
# Reads benchmarks/output/o01_rainfall_<mode>_mc<gridsuffix>/ and writes
# o01_rainfall_<mode>_<grid>.mp4 there. The grid dimensions are inferred from
# the snapshots (mubar = 3 Gauss points per cell), so this works for any
# resolution the benchmark was run at.

using CSV
using DataFrames
using CairoMakie
using Scythe
using Springsteel

# ── Arguments ───────────────────────────────────────────────────────────────
mode = "full"
grid = "rirk"
fps = 8
let i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--mode"
            global mode = ARGS[i+1]; i += 2
        elseif ARGS[i] == "--grid"
            global grid = ARGS[i+1]; i += 2
        elseif ARGS[i] == "--fps"
            global fps = parse(Int, ARGS[i+1]); i += 2
        else
            error("Unknown argument: $(ARGS[i])")
        end
    end
end

suffix = grid == "rz" ? "" : "_$(grid)"
dir = joinpath(@__DIR__, "output", "o01_rainfall_$(mode)_mc$(suffix)")
isdir(dir) || error("No run output at $dir — run the benchmark first")

# ── Snapshots and grid shape (inferred from the output itself) ──────────────
files = filter(f -> endswith(f, "_physical.csv"), readdir(dir))
snaps = sort([(parse(Float64, replace(f, "_physical.csv" => "")), joinpath(dir, f))
              for f in files], by = first)
isempty(snaps) && error("No *_physical.csv snapshots in $dir")

df0 = CSV.read(snaps[1][2], DataFrame)
zvals = df0.z
kDim = findfirst(i -> zvals[i+1] < zvals[i], 1:length(zvals)-1)   # z resets per column
ncols = div(nrow(df0), kDim)
x = reshape(df0.r, kDim, ncols)[1, :]
z = reshape(df0.z, kDim, ncols)[:, 1]
println("Found $(length(snaps)) snapshots, kDim=$kDim, ncols=$ncols")

# ── Reference state (values only; the BCs of this throwaway grid are unused) ─
vars = ["p", "rho_d", "rho_t", "u", "w", "E_t", "Q_ss", "rho_r"]
scalar_bc = Dict(v => NeumannBC() for v in vars)
gp = Scythe.compute_derived_params(GridParameters(;
    geometry = grid == "rz" ? "RZ" : "RiRk",
    iMin = 0.0, iMax = 150.0e3, num_cells_i = div(ncols, 3),
    kMin = 0.0, kMax = 20.0e3,
    (grid == "rz" ? (; kDim = kDim) : (; num_cells_k = div(kDim, 3)))...,
    BCL = scalar_bc, BCR = scalar_bc, BCB = scalar_bc, BCT = scalar_bc,
    vars = Dict(v => i for (i, v) in enumerate(vars))))
patch = createGrid(gp)
column = Scythe.reference_column(patch, gp)
ref = Springsteel.exact_pressure_reference_state(joinpath(dir, "o01_exact.ref"),
                                                 Scythe.getGridpoints(patch)[1:kDim, 2],
                                                 column)
pbar = Springsteel.ref_pressure(ref)[:, 1]
rho_dbar = Springsteel.ref_rho_d(ref)[:, 1]
rho_tbar = Springsteel.ref_rho_t(ref)[:, 1]
E_tbar = Springsteel.ref_total_energy(ref)[:, 1]
Q_ssbar = Springsteel.ref_qss(ref)[:, 1]
Tbar = Springsteel.reference_temperature(ref)

"""Diagnosed condensate and rain [g/m³] of one snapshot, (kDim, ncols)."""
function cloud_and_rain(path)
    df = CSV.read(path, DataFrame)
    nc = div(nrow(df), kDim)
    p = df.p .+ repeat(pbar, nc)
    rho_d = df.rho_d .+ repeat(rho_dbar, nc)
    rho_t = df.rho_t .+ repeat(rho_tbar, nc)
    E_t = df.E_t .+ repeat(E_tbar, nc)
    Q_ss = df.Q_ss .+ repeat(Q_ssbar, nc)
    ke = 0.5 .* (df.u .^ 2 .+ df.w .^ 2)
    M = p .+ E_t .- rho_t .* (ke .+ Scythe.gravity .* df.z)
    Tk = Scythe.retrieve_temperature.(M, rho_d, rho_t, Q_ss, p, repeat(Tbar, nc), df.rho_r)
    rho_vs = Springsteel.Thermodynamics.rho_v_sat.(Tk, p ./ 100.0)
    rho_v = clamp.(Q_ss .+ rho_vs, 0.0, max.(rho_t .- rho_d .- df.rho_r, 0.0))
    rho_c = max.(rho_t .- rho_d .- rho_v .- df.rho_r, 0.0)
    return reshape(1000.0 .* rho_c, kDim, ncols), reshape(1000.0 .* df.rho_r, kDim, ncols)
end

# ── Frames ──────────────────────────────────────────────────────────────────
framedir = joinpath(dir, "movie_frames")
mkpath(framedir)
rm.(joinpath.(framedir, readdir(framedir)))   # stale frames would leak into the movie

cloud_levels = 0.0:0.2:3.0                     # rho_c fill [g/m³]
rain_levels = [0.5, 1.0, 2.0, 4.0, 8.0]       # rho_r white contours [g/m³]

for (i, (t, path)) in enumerate(snaps)
    rho_c, rho_r = cloud_and_rain(path)
    fig = Figure(size = (950, 420))
    ax = Axis(fig[1, 1],
              title = "O01 warm rain — t = $(round(Int, t / 60)) min",
              xlabel = "x (km)", ylabel = "z (km)")
    cf = contourf!(ax, x ./ 1000.0, z ./ 1000.0, rho_c',
                   levels = cloud_levels, extendhigh = :auto, colormap = :viridis)
    contour!(ax, x ./ 1000.0, z ./ 1000.0, rho_r',
             levels = rain_levels, color = :white, linewidth = 1.2)
    ylims!(ax, 0, 16)                          # storm layer; lid at 20 km is quiet
    Colorbar(fig[1, 2], cf, label = "ρ_c (g/m³)   [white: ρ_r]")
    frame = joinpath(framedir, "frame_" * lpad(i - 1, 4, '0') * ".png")
    save(frame, fig)
    i % 10 == 0 && println("  frame $i / $(length(snaps))")
end
println("Rendered $(length(snaps)) frames")

# ── Movie ───────────────────────────────────────────────────────────────────
movie = joinpath(dir, "o01_rainfall_$(mode)_$(grid).mp4")
run(`ffmpeg -y -loglevel error -framerate $fps -i $(joinpath(framedir, "frame_%04d.png"))
     -c:v libx264 -pix_fmt yuv420p -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" $movie`)
println("Wrote $movie")
