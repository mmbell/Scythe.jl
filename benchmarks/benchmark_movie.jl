#!/usr/bin/env julia
# Time-evolution movie for an existing Straka93 / BF02 full_mc_rirk benchmark run
# (no re-run): the same two-panel θ′(or θ_e′) + w figure the benchmark draws at
# the end state, one frame per output snapshot, assembled with ffmpeg.
#
#   julia --project=. benchmarks/benchmark_movie.jl --test straka93 [--fps 2]
#   julia --project=. benchmarks/benchmark_movie.jl --test bf02_dry
#   julia --project=. benchmarks/benchmark_movie.jl --test bf02_moist
#
# Reads benchmarks/output/<test>_full_mc_rirk/ and writes <test>_full_mc_rirk.mp4
# there. Only the full_mc_rirk configuration (the best benchmark) is supported;
# grid dimensions are inferred from the snapshots (mubar = 3 Gauss points/cell),
# so this works for any resolution the benchmark was run at.

using CSV
using DataFrames
using CairoMakie
using Scythe
using Springsteel

# Reconstruction (mc_state, theta_perturbation) and the figure renderer used by the
# benchmark diagnostics themselves — reused verbatim so each frame matches the
# committed end-state figure.
include(joinpath(@__DIR__, "common", "diagnostics.jl"))
include(joinpath(@__DIR__, "common", "plots.jl"))

# ── Per-test configuration ──────────────────────────────────────────────────
# Panel levels/labels are copied from each driver's own plotter (straka93.jl,
# bf02_dry.jl, bf02_moist.jl) so the movie frames match the *_final.png figures.
struct MovieConfig
    dirbase::String
    reffile::String
    iMax::Float64
    kMax::Float64
    title::String
    moist::Bool
    theta_label::String
    theta_levels::StepRangeLen
    w_levels::StepRangeLen
end

const CONFIGS = Dict(
    "straka93" => MovieConfig(
        "straka93_full_mc_rirk", "straka93.ref", 25.6e3, 6.4e3,
        "Straka93 density current", false,
        "θ′ (K)", -15.5:1.0:-0.5, -16.0:2.0:14.0),
    "bf02_dry" => MovieConfig(
        "bf02_dry_full_mc_rirk", "bf02_dry.ref", 20.0e3, 10.0e3,
        "BF02 dry thermal", false,
        "θ′ (K)", -0.2:0.2:2.2, -10.0:2.0:16.0),
    "bf02_moist" => MovieConfig(
        "bf02_moist_full_mc_rirk", "bf02_moist_exact.ref", 20.0e3, 10.0e3,
        "BF02 moist thermal", true,
        "θ_e′ (K)", -0.5:0.5:4.5, -10.0:2.0:16.0),
)

# ── Arguments ───────────────────────────────────────────────────────────────
test = nothing
fps = 2
let i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--test"
            global test = ARGS[i+1]; i += 2
        elseif ARGS[i] == "--fps"
            global fps = parse(Int, ARGS[i+1]); i += 2
        else
            error("Unknown argument: $(ARGS[i])")
        end
    end
end
test === nothing && error("--test is required (one of $(join(sort(collect(keys(CONFIGS))), ", ")))")
haskey(CONFIGS, test) || error("Unknown --test $test (one of $(join(sort(collect(keys(CONFIGS))), ", ")))")
cfg = CONFIGS[test]

dir = joinpath(@__DIR__, "output", cfg.dirbase)
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

# ── Reference state (RiRk / pressure reference; BCs of this throwaway grid unused) ─
vars = ["p", "rho_d", "rho_t", "u", "w", "E_t", "Q_ss", "rho_r"]
scalar_bc = Dict(v => NeumannBC() for v in vars)
gp = Scythe.compute_derived_params(GridParameters(;
    geometry = "RiRk",
    iMin = 0.0, iMax = cfg.iMax, num_cells_i = div(ncols, 3),
    kMin = 0.0, kMax = cfg.kMax, num_cells_k = div(kDim, 3),
    BCL = scalar_bc, BCR = scalar_bc, BCB = scalar_bc, BCT = scalar_bc,
    vars = Dict(v => i for (i, v) in enumerate(vars))))
patch = createGrid(gp)
column = Scythe.reference_column(patch, gp)
ref = Springsteel.exact_pressure_reference_state(joinpath(dir, cfg.reffile),
                                                 Scythe.getGridpoints(patch)[1:kDim, 2],
                                                 column)

# Baseline θ_e profile for the moist perturbation (written by bf02_moist.jl)
base = cfg.moist ? CSV.read(joinpath(dir, "base_profile.csv"), DataFrame) : nothing

"""θ′ (dry) or θ_e′ (moist) and w for one snapshot, both (kDim, ncols)."""
function frame_fields(path)
    df = CSV.read(path, DataFrame)
    nc = div(nrow(df), kDim)
    w = reshape(df.w, kDim, nc)
    if !cfg.moist
        theta_p, _ = theta_perturbation(df, ref, kDim)
        return theta_p, w
    end
    # Moist θ_e′: mc branch of bf02_moist.jl's moist_fields, gated to stage mc.
    Tk, p, rho_d, rho_v, rho_c, rho_t = mc_state(df, ref, kDim, nc)
    q_v = max.(rho_v, 0.0) ./ rho_d      # entropy()/theta_e take log(q_v)
    q_l = (max.(rho_c, 0.0) .+ df.rho_r) ./ rho_d
    s = Scythe.entropy.(Tk, rho_d, q_v)
    xi = Scythe.log_dry_density.(rho_d)
    mu = Scythe.mu_transform.(q_v)
    mu_liq = Scythe.mu_transform.(q_l)
    theta_e = Scythe.reversible_theta_e.(s, xi, mu, mu_liq)
    theta_e_p = reshape(theta_e .- repeat(base.theta_e, nc), kDim, nc)
    return theta_e_p, w
end

# ── Frames ──────────────────────────────────────────────────────────────────
framedir = joinpath(dir, "movie_frames")
mkpath(framedir)
rm.(joinpath.(framedir, readdir(framedir)))   # stale frames would leak into the movie

for (i, (t, path)) in enumerate(snaps)
    theta, w = frame_fields(path)
    frame = joinpath(framedir, "frame_" * lpad(i - 1, 4, '0') * ".png")
    save_benchmark_figure(frame, x, z,
        [(theta, cfg.theta_label, cfg.theta_levels),
         (w, "w (m/s)", cfg.w_levels)];
        title = "$(cfg.title), t = $(round(Int, t)) s")
end
println("Rendered $(length(snaps)) frames")

# ── Movie ───────────────────────────────────────────────────────────────────
movie = joinpath(dir, "$(cfg.dirbase).mp4")
run(`ffmpeg -y -loglevel error -framerate $fps -i $(joinpath(framedir, "frame_%04d.png"))
     -c:v libx264 -pix_fmt yuv420p -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" $movie`)
println("Wrote $movie")
