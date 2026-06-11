#!/usr/bin/env julia
# Compare the legacy and primitive-equation solutions of a benchmark case:
# relative L2 differences of the shared prognostic fields at the final time.
#
#   julia --project=. benchmarks/compare_stages.jl <case> [quick|full]
#
# Both stages must have been run (their output directories must contain the
# final physical CSV). The shared variables are s, xi, mu, u, w; liquid water
# is compared as the sum of the stage's liquid variables when both stages are
# moist.

using CSV
using DataFrames
using Printf
using LinearAlgebra

case = isempty(ARGS) ? error("usage: compare_stages.jl <case> [quick|full]") : ARGS[1]
mode = length(ARGS) >= 2 ? ARGS[2] : "quick"

const BENCHMARKS_DIR = @__DIR__
legacy_dir = joinpath(BENCHMARKS_DIR, "output", "$(case)_$(mode)_legacy")
pe_dir = joinpath(BENCHMARKS_DIR, "output", "$(case)_$(mode)_pe")

final_csv(dir) = begin
    csvs = filter(f -> endswith(f, "_physical.csv"), readdir(dir))
    times = [parse(Float64, replace(f, "_physical.csv" => "")) for f in csvs]
    joinpath(dir, csvs[argmax(times)])
end

legacy = CSV.read(final_csv(legacy_dir), DataFrame)
pe = CSV.read(final_csv(pe_dir), DataFrame)
nrow(legacy) == nrow(pe) || error("Grid mismatch: $(nrow(legacy)) vs $(nrow(pe)) points")

println("── $(case) [$(mode)] legacy vs primitive_equation_XZ " * "─"^20)
@printf("%-10s %12s %12s %12s\n", "field", "rel_L2", "max_abs", "legacy_scale")

function compare(name, a, b)
    scale = max(norm(b), eps())
    rel = norm(a .- b) / scale
    @printf("%-10s %12.3e %12.3e %12.3e\n", name, rel, maximum(abs.(a .- b)), scale)
end

for var in ["s", "xi", "mu", "u", "w"]
    compare(var, pe[!, var], legacy[!, var])
end

# Liquid water: legacy mu_l vs PE mu_c + mu_r (linear transform sums exactly)
if "mu_l" in names(legacy) && "mu_c" in names(pe)
    compare("liquid", pe.mu_c .+ pe.mu_r, legacy.mu_l)
end
