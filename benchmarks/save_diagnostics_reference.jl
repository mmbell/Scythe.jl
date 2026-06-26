#!/usr/bin/env julia
# Write a committed regression reference from an ALREADY-COMPLETED benchmark run,
# without re-running the simulation.
#
# `run_benchmark --update-reference` re-runs init! + integrate_model before writing
# the reference, which is wasteful for an expensive run that has already produced
# good output. This script instead takes the diagnostics that the completed run
# already wrote to its output directory (`<output_dir>/diagnostics.csv`) and emits
# the full-mode regression reference via the harness's own path + writer, so the
# result is byte-for-byte what `--update-reference` would have produced.
#
# Usage:
#   julia --project=. benchmarks/save_diagnostics_reference.jl <name> \
#       --mode full --stage pe-rho_d-pd --grid rirk
#
# Full mode only: full-mode references store the scalar target diagnostics. Quick
# mode stores full final fields, which are cheap to regenerate with a normal
# `--update-reference` run.

include(joinpath(@__DIR__, "common", "harness.jl"))

length(ARGS) >= 1 || error("Usage: save_diagnostics_reference.jl <name> [--mode full --stage ... --grid ...]")
name = ARGS[1]
opts = parse_benchmark_args(ARGS[2:end])
opts.mode == :full || error("This script handles full-mode (diagnostics) references; " *
                            "quick mode stores full fields — re-run with --update-reference.")

output_dir = benchmark_output_dir(name, opts)
diag_csv = joinpath(output_dir, "diagnostics.csv")
isfile(diag_csv) || error("No diagnostics.csv in $(output_dir) — run the benchmark first.")

# Read the diagnostics the completed run wrote (diagnostic,value,target,atol,pass,source)
df = CSV.read(diag_csv, DataFrame)
diags = Dict{String,Float64}(string(r.diagnostic) => Float64(r.value) for r in eachrow(df))

targets = load_targets(name, opts)
missing_targets = [t.name for t in targets if !haskey(diags, t.name)]
isempty(missing_targets) || error("diagnostics.csv is missing target(s): $(missing_targets)")

ref_path = reference_csv_path(name, opts)
write_diagnostics_reference(ref_path, diags, targets)

println("Wrote regression reference (no re-run): $(ref_path)")
for t in sort(targets, by = x -> x.name)
    @printf("  %-16s = %.10g\n", t.name, diags[t.name])
end
