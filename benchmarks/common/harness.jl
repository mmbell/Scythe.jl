# Shared harness for Scythe benchmark scripts.
#
# Each benchmark script builds a ModelParameters configuration for the requested
# mode/stage, provides an init! function that generates the reference sounding
# and initial conditions, and a diagnostics function that reduces the final
# output to scalar values. The harness runs the model, checks the diagnostics
# against published target values, compares the output fields against committed
# reference data, and appends a JSON record with timing and provenance.

using Distributed
using Printf
using Dates
using CSV
using DataFrames

const BENCHMARKS_DIR = normpath(joinpath(@__DIR__, ".."))
const REFERENCE_DATA_DIR = joinpath(BENCHMARKS_DIR, "reference_data")
const RESULTS_DIR = joinpath(BENCHMARKS_DIR, "results")

include(joinpath(REFERENCE_DATA_DIR, "expected_values.jl"))

struct BenchmarkOptions
    mode::Symbol        # :quick | :full
    stage::Symbol       # :legacy | :pe
    workers::Int
    update_reference::Bool
    plot::Bool
end

struct Target
    name::String
    value::Float64
    atol::Float64
    source::String
end

"""
    parse_benchmark_args(args) -> BenchmarkOptions

Parse benchmark command line arguments:
--mode quick|full (default quick), --stage legacy|pe (default legacy),
--workers N (default 2), --update-reference, --plot.
"""
function parse_benchmark_args(args::Vector{String})
    mode = :quick
    stage = :legacy
    nworkers = 2
    update_reference = false
    plot = false
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--mode"
            mode = Symbol(args[i+1]); i += 2
        elseif arg == "--stage"
            stage = Symbol(args[i+1]); i += 2
        elseif arg == "--workers"
            nworkers = parse(Int, args[i+1]); i += 2
        elseif arg == "--update-reference"
            update_reference = true; i += 1
        elseif arg == "--plot"
            plot = true; i += 1
        elseif arg in ("--help", "-h")
            println("Usage: julia --project=. benchmarks/<case>.jl " *
                    "[--mode quick|full] [--stage legacy|pe] [--workers N] [--update-reference] [--plot]")
            exit(0)
        else
            error("Unknown argument: $arg")
        end
    end
    mode in (:quick, :full) || error("--mode must be quick or full")
    stage in (:legacy, :pe) || error("--stage must be legacy or pe")
    nworkers >= 1 || error("--workers must be >= 1")
    return BenchmarkOptions(mode, stage, nworkers, update_reference, plot)
end

"""Output directory for a benchmark variant (created if missing)."""
function benchmark_output_dir(name::String, opts::BenchmarkOptions)
    dir = joinpath(BENCHMARKS_DIR, "output", "$(name)_$(opts.mode)_$(opts.stage)")
    mkpath(dir)
    return dir * "/"   # integrate_model concatenates paths with *
end

"""Committed regression reference CSV path for a benchmark variant."""
function reference_csv_path(name::String, opts::BenchmarkOptions)
    return joinpath(REFERENCE_DATA_DIR, name, "$(opts.mode)_$(opts.stage)_final.csv")
end

"""
    load_targets(name, opts) -> Vector{Target}

Load published target values for a benchmark. Quick mode and the primitive
equation stage use the wider quick tolerances.
"""
function load_targets(name::String, opts::BenchmarkOptions)
    haskey(BENCHMARK_EXPECTED, name) || error("No expected values defined for $name")
    use_quick_tol = opts.mode == :quick || opts.stage == :pe
    targets = Target[]
    for (diag, (value, atol_full, atol_quick, source)) in BENCHMARK_EXPECTED[name]
        atol = use_quick_tol ? atol_quick : atol_full
        push!(targets, Target(diag, value, atol, source))
    end
    sort!(targets, by = t -> t.name)
    return targets
end

"""Check diagnostics against targets. Returns Dict of name => pass."""
function check_targets(diags::Dict{String,Float64}, targets::Vector{Target})
    results = Dict{String,Bool}()
    for t in targets
        if haskey(diags, t.name)
            results[t.name] = abs(diags[t.name] - t.value) <= t.atol
        else
            results[t.name] = false
        end
    end
    return results
end

"""
    compare_reference(output_csv, reference_csv, varnames; rtol=1.0e-6)

Compare the value columns of a model output CSV against a committed reference.
Returns (pass, Dict of var => (rel_l2, max_abs_diff)).
"""
function compare_reference(output_csv::String, reference_csv::String,
                           varnames::Vector{String}; rtol=1.0e-6)
    out = CSV.read(output_csv, DataFrame)
    ref = CSV.read(reference_csv, DataFrame)
    if nrow(out) != nrow(ref)
        println("  Reference comparison: grid size mismatch " *
                "($(nrow(out)) vs $(nrow(ref)) points)")
        return false, Dict{String,Tuple{Float64,Float64}}()
    end
    stats = Dict{String,Tuple{Float64,Float64}}()
    pass = true
    for var in varnames
        a = out[!, var]
        b = ref[!, var]
        scale = max(sqrt(sum(abs2, b)), eps())
        rel_l2 = sqrt(sum(abs2, a .- b)) / scale
        max_abs = maximum(abs.(a .- b))
        stats[var] = (rel_l2, max_abs)
        pass &= rel_l2 <= rtol
    end
    return pass, stats
end

"""
    write_reference(output_csv, reference_csv, varnames)

Write a committed regression reference from a model output CSV, keeping only
the coordinate and value columns (dropping derivative columns) at 10
significant digits.
"""
function write_reference(output_csv::String, reference_csv::String,
                         varnames::Vector{String})
    out = CSV.read(output_csv, DataFrame)
    coords = intersect(["r", "l", "z"], names(out))
    cols = vcat(coords, varnames)
    ref = select(out, cols)
    for col in cols
        ref[!, col] = round.(ref[!, col], sigdigits=10)
    end
    mkpath(dirname(reference_csv))
    CSV.write(reference_csv, ref)
    return reference_csv
end

"""Git SHA and dirty flag for a repository directory, or ("unknown", false)."""
function git_info(dir::String)
    try
        sha = strip(read(`git -C $dir rev-parse --short HEAD`, String))
        dirty = !isempty(strip(read(`git -C $dir status --porcelain`, String)))
        return String(sha), dirty
    catch
        return "unknown", false
    end
end

"""Minimal JSON encoding for flat values, dicts, and arrays."""
function to_json(x)
    if x isa AbstractDict
        entries = ["\"$(k)\":$(to_json(v))" for (k, v) in sort(collect(x), by=first)]
        return "{" * join(entries, ",") * "}"
    elseif x isa AbstractVector
        return "[" * join(to_json.(x), ",") * "]"
    elseif x isa AbstractString || x isa Symbol
        return "\"$(x)\""
    elseif x isa Bool
        return string(x)
    elseif x isa Real
        return isfinite(x) ? string(x) : "null"
    elseif isnothing(x)
        return "null"
    else
        return "\"$(x)\""
    end
end

"""Append a JSON line describing this run to benchmarks/results/<name>.jsonl."""
function record_result(name::String, record::Dict)
    mkpath(RESULTS_DIR)
    open(joinpath(RESULTS_DIR, "$(name).jsonl"), "a") do f
        println(f, to_json(record))
    end
end

"""Print the verification table for a benchmark run."""
function report_table(name::String, opts::BenchmarkOptions, diags::Dict{String,Float64},
                      targets::Vector{Target}, target_pass::Dict{String,Bool})
    println()
    println("── $(name) [$(opts.mode)/$(opts.stage)] verification " * "─"^30)
    @printf("%-18s %12s %12s %8s %6s  %s\n",
            "diagnostic", "value", "target", "atol", "pass", "source")
    for t in targets
        value = get(diags, t.name, NaN)
        pass = target_pass[t.name] ? "PASS" : "FAIL"
        @printf("%-18s %12.4f %12.4f %8.3f %6s  %s\n",
                t.name, value, t.value, t.atol, pass, t.source)
    end
    for key in sort(collect(keys(diags)))
        if !any(t -> t.name == key, targets)
            @printf("%-18s %12.5g %35s\n", key, diags[key], "(informational)")
        end
    end
end

"""
    run_benchmark(name, opts; model, init!, diagnostics, varnames) -> Bool

Run a benchmark end-to-end: initialize, integrate, verify, report, record.
Returns true if all targets pass and the regression comparison (when a
committed reference exists) is within tolerance.
"""
function run_benchmark(name::String, opts::BenchmarkOptions;
                       model, init!::Function, diagnostics::Function,
                       varnames::Vector{String}, plotter=nothing)

    println("═"^70)
    println("Benchmark: $name  mode=$(opts.mode)  stage=$(opts.stage)  " *
            "equation_set=$(model.equation_set)")
    nsteps = round(Int, model.integration_time / model.ts)
    println("Grid: num_cells=$(model.grid_params.num_cells) kDim=$(model.grid_params.kDim)  " *
            "ts=$(model.ts) s  steps=$nsteps")
    println("═"^70)

    println("Generating reference state and initial conditions...")
    init!(model)

    println("Integrating...")
    wallclock = @elapsed integrate_model(model)
    steps_per_sec = nsteps / wallclock
    @printf("Wall clock: %.1f s  (%.1f steps/s)\n", wallclock, steps_per_sec)

    println("Computing diagnostics...")
    diags = diagnostics(model)

    if plotter !== nothing
        println("Generating figures...")
        plotter(model)
    end

    targets = load_targets(name, opts)
    target_pass = check_targets(diags, targets)
    report_table(name, opts, diags, targets, target_pass)
    targets_ok = all(values(target_pass))

    # Regression comparison against committed reference output
    final_tag = string(round(model.integration_time; digits=2))
    output_csv = joinpath(model.output_dir, "$(final_tag)_physical.csv")
    ref_csv = reference_csv_path(name, opts)
    regression_ok = nothing
    regression_stats = Dict{String,Tuple{Float64,Float64}}()
    if opts.update_reference
        write_reference(output_csv, ref_csv, varnames)
        println("\nUpdated regression reference: $ref_csv")
    elseif isfile(ref_csv)
        regression_ok, regression_stats = compare_reference(output_csv, ref_csv, varnames)
        println("\nRegression vs committed reference ($(basename(ref_csv))):")
        for var in varnames
            rel_l2, max_abs = regression_stats[var]
            @printf("  %-8s rel_L2 = %.3e  max_abs = %.3e  %s\n",
                    var, rel_l2, max_abs, rel_l2 <= 1.0e-6 ? "PASS" : "FAIL")
        end
    else
        println("\nNo committed reference at $ref_csv")
        println("(re-run with --update-reference after accepting this result)")
    end

    scythe_sha, scythe_dirty = git_info(normpath(joinpath(@__DIR__, "..", "..")))
    springsteel_sha, _ = git_info(pkgdir(Springsteel))
    worker_threads = fetch(@spawnat workers()[1] Threads.nthreads())

    passed = targets_ok && (isnothing(regression_ok) || regression_ok)
    record = Dict(
        "timestamp" => string(now()),
        "case" => name,
        "mode" => opts.mode,
        "stage" => opts.stage,
        "equation_set" => model.equation_set,
        "scythe_sha" => scythe_sha,
        "scythe_dirty" => scythe_dirty,
        "springsteel_sha" => springsteel_sha,
        "julia_version" => string(VERSION),
        "hostname" => gethostname(),
        "nworkers" => nworkers(),
        "worker_threads" => worker_threads,
        "ts" => model.ts,
        "num_cells" => model.grid_params.num_cells,
        "kDim" => model.grid_params.kDim,
        "integration_time" => model.integration_time,
        "wallclock_s" => round(wallclock, digits=2),
        "steps_per_sec" => round(steps_per_sec, digits=2),
        "diagnostics" => Dict(k => v for (k, v) in diags),
        "target_pass" => target_pass,
        "regression_pass" => regression_ok,
        "passed" => passed,
    )
    record_result(name, record)

    println()
    println(passed ? "✓ $name [$(opts.mode)/$(opts.stage)] PASSED" :
                     "✗ $name [$(opts.mode)/$(opts.stage)] FAILED")
    return passed
end
