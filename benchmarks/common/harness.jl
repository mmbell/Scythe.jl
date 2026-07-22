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
    stage::Symbol       # :legacy | :pe | Symbol("pe-rho_d")
    grid::Symbol        # :rz (Chebyshev vertical) | :rirk (B-spline vertical)
    workers::Int
    update_reference::Bool
    plot::Bool
    ts_factor::Float64  # RiRk timestep scale (--ts-factor, default 1.0)
    nests::Int          # --nests N: grid-nesting levels (1 = single grid, default)
    hsi::Bool           # --hsi: horizontal semi-implicit (mc stage, RiRk grid only)
    xsi::Bool           # --exact-si: unsplit 2-D acoustic SI (mc stage, RiRk grid only)
end

# Primitive-equation stage carrying the linear dry-air density rho_d' (slot 2)
# in place of the log-density xi. Spelled with a hyphen for external clarity;
# the hyphen precludes a `:pe-rho_d` symbol literal, so reference this constant.
const STAGE_PE_RHOD = Symbol("pe-rho_d")

# Partial-density moisture stage: as STAGE_PE_RHOD but the moisture variables are the
# partial densities rho_v/rho_c/rho_r (conserving the physical water mass under spline
# smoothing). Uses a condensate-bearing physical reference state.
const STAGE_PE_RHOD_PD = Symbol("pe-rho_d-pd")

# Entropy-density stage (Stage 2): as STAGE_PE_RHOD_PD but slot 1 carries the extensive
# entropy density sigma = rho_d*s instead of the intensive specific entropy s, so the spline
# smoothing conserves the entropy-density integral ∫sigma (equation set PE_SIGMA_XZ).
const STAGE_PE_SIGMA = Symbol("pe-sigma")

# Total-energy stage: prognostic p / rho_d / rho_t / E_t / Q_ss on a pressure-based
# reference (equation set moist_compressible_XZ). All conserved quantities are extensive
# flux-form prognostics; T is diagnosed and the vapor/cloud partition follows from Q_ss.
const STAGE_MC = Symbol("mc")

"""
    reference_state_options() -> Dict{Symbol,Any}

Opt-in reference-state fixes for the pressure-reference (`mc`) stages, selected by the
environment variable `SCYTHE_REFSTATE`:

    SCYTHE_REFSTATE=qss          options[:consistent_qss_reference]
    SCYTHE_REFSTATE=hydro        options[:hydrostatic_reference]
    SCYTHE_REFSTATE=qss,hydro    both  (also spelled "both")

Empty by default, so every committed baseline is reproduced BITWISE unless the variable
is set. See `reference/HANDOFF_REFERENCE_STATE.md`: these make the resting reference a discrete
steady state (it condensed at rest, and its stored dp̄/dz was up to 17 % off hydrostatic
balance), and they necessarily move the mc-stage baselines — which is what running the
benchmarks under them is meant to quantify.

`reference_state_hydrostatic()` reports just the hydrostatic half, for the benchmark's
own `calculate_pressure_reference_state` call that writes the `.ref` file: the file
carries values only, so the converged triple has to be written for the balance to
survive the round trip.
"""
function reference_state_options()
    spec = get(ENV, "SCYTHE_REFSTATE", "")
    isempty(spec) && return Dict{Symbol,Any}()
    parts = strip.(split(lowercase(spec), ','))
    both = "both" in parts
    opts = Dict{Symbol,Any}()
    (both || "qss" in parts) && (opts[:consistent_qss_reference] = true)
    (both || "hydro" in parts) && (opts[:hydrostatic_reference] = true)
    isempty(opts) && error("SCYTHE_REFSTATE='$spec' matched nothing; use qss, hydro, " *
                           "qss,hydro or both")
    return opts
end

reference_state_hydrostatic() = get(reference_state_options(), :hydrostatic_reference, false)

struct Target
    name::String
    value::Float64
    atol::Float64
    source::String
end

"""
    parse_benchmark_args(args) -> BenchmarkOptions

Parse benchmark command line arguments:
--mode quick|full (default quick), --stage legacy|pe|pe-rho_d (default legacy),
--grid rz|rirk (default rz), --workers N (default 2), --ts-factor F (RiRk timestep
scale, default 1.0), --update-reference, --plot.
"""
function parse_benchmark_args(args::Vector{String})
    mode = :quick
    stage = :legacy
    grid = :rz
    nworkers = 2
    update_reference = false
    plot = false
    ts_factor = 1.0
    nests = 1
    hsi = false
    xsi = false
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--mode"
            mode = Symbol(args[i+1]); i += 2
        elseif arg == "--stage"
            stage = Symbol(args[i+1]); i += 2
        elseif arg == "--grid"
            grid = Symbol(args[i+1]); i += 2
        elseif arg == "--workers"
            nworkers = parse(Int, args[i+1]); i += 2
        elseif arg == "--ts-factor"
            ts_factor = parse(Float64, args[i+1]); i += 2
        elseif arg == "--nests"
            nests = parse(Int, args[i+1]); i += 2
        elseif arg == "--hsi"
            hsi = true; i += 1
        elseif arg == "--exact-si"
            xsi = true; i += 1
        elseif arg == "--update-reference"
            update_reference = true; i += 1
        elseif arg == "--plot"
            plot = true; i += 1
        elseif arg in ("--help", "-h")
            println("Usage: julia --project=. benchmarks/<case>.jl " *
                    "[--mode quick|full] [--stage legacy|pe|pe-rho_d] [--grid rz|rirk] " *
                    "[--workers N] [--ts-factor F] [--nests N] [--hsi] [--exact-si] [--update-reference] [--plot]")
            exit(0)
        else
            error("Unknown argument: $arg")
        end
    end
    mode in (:quick, :full) || error("--mode must be quick or full")
    stage in (:legacy, :pe, STAGE_PE_RHOD, STAGE_PE_RHOD_PD, STAGE_PE_SIGMA, STAGE_MC) ||
        error("--stage must be legacy, pe, pe-rho_d, pe-rho_d-pd, pe-sigma, or mc")
    grid in (:rz, :rirk) || error("--grid must be rz or rirk")
    nworkers >= 1 || error("--workers must be >= 1")
    ts_factor > 0.0 || error("--ts-factor must be > 0")
    nests >= 1 || error("--nests must be >= 1")
    hsi && (stage == STAGE_MC && grid == :rirk ||
        error("--hsi requires --stage mc --grid rirk"))
    xsi && (stage == STAGE_MC && grid == :rirk ||
        error("--exact-si requires --stage mc --grid rirk"))
    (hsi && xsi) && error("--hsi and --exact-si are mutually exclusive")
    return BenchmarkOptions(mode, stage, grid, nworkers, update_reference, plot, ts_factor, nests, hsi, xsi)
end

"""Geometry string for the configured vertical basis (RZ Chebyshev vs RiRk B-spline)."""
benchmark_geometry(opts::BenchmarkOptions) = opts.grid == :rirk ? "RiRk" : "RZ"

"""
    add_benchmark_workers(opts)

Add the worker processes, dividing the machine's threads *among* them rather than giving
each one the whole box.

`--threads=auto` hands every worker `Sys.CPU_THREADS` threads, so the default 2 workers on a
12-core machine spawn 24 compute threads for 12 cores. They then all allocate into a shared
GC from inside the `Threads.@threads` column loop, which shows up as heavy GC time and
millions of lock conflicts — and is a suspected contributor to the intermittent worker death
in long moist_compressible runs.
"""
function add_benchmark_workers(opts::BenchmarkOptions; count::Int = opts.workers)
    nthreads = max(1, Sys.CPU_THREADS ÷ count)
    println("Adding $(count) worker(s) with $(nthreads) thread(s) each " *
            "($(Sys.CPU_THREADS) CPU threads available)")
    return addprocs(count, exeflags = "--threads=$(nthreads)")
end

"""
    vertical_ts(ts, opts) -> Float64

Timestep for the configured grid. On the cubic B-spline (RiRk) grid the base
timestep is scaled by `opts.ts_factor` (the `--ts-factor` flag, default 1.0); the
Chebyshev (RZ) grid is unaffected. RiRk historically used 0.5 because its explicit
acoustic solve has a tighter stability limit than the Chebyshev pseudospectral
solve, but the semi-implicit solver treats vertical acoustics implicitly, so the
reduction is now opt-in via the flag (e.g. `--ts-factor 0.5` for explicit runs).
The physical times — and hence output and diagnostics — are unchanged; only the
step count changes.
"""
vertical_ts(ts::Float64, opts::BenchmarkOptions) =
    opts.grid == :rirk ? ts * opts.ts_factor : ts

"""
    vertical_size(opts; num_cells_k, kDim) -> NamedTuple

Vertical sizing keywords for the configured geometry, to be splatted into
`GridParameters(...)`. The two vertical bases are sized by *different*, independent numbers:

- **RiRk** (cubic B-spline vertical) is sized by its **cell count** `num_cells_k`. That is the
  formal definition of a spline axis (`kDim = num_cells_k * mubar`, `b_kDim = num_cells_k + 3`),
  and it is what lets the vertical cell width be matched to the horizontal one.
- **RZ** (Chebyshev vertical) is sized by its **gridpoint count** `kDim`. A Chebyshev axis has
  no cells: Springsteel leaves `num_cells_k = 0` for it, so passing a cell count to an RZ grid
  is silently ignored and yields a zero-height column.

The two are deliberately *not* tied together (e.g. `kDim = num_cells_k * mubar`), for two
reasons. First, `mubar` is a B-spline parameter; there is no reason a Chebyshev axis should
carry 3x the spline cell count. Second, and decisively, Chebyshev points cluster at the walls,
so `dz_min` shrinks roughly as `1/kDim^2` (measured over 6.4 km: `kDim` 32/48/64/96 gives
`dz_min` 16.4/7.2/4.0/1.8 m). The *explicit* acoustic CFL therefore tightens quadratically, and
straka's `legacy`/`pe` stages integrate explicitly on RZ: sizing them to `3 * num_cells_k = 96`
instead of 64 shrinks `dz_min` by 2.3x and they go non-finite at t = 0.688 s. Sizing the two
bases independently keeps the spline vertical physically correct without destabilising the
Chebyshev one — raising RZ `kDim` requires cutting `ts` to match.

This replaces the old `vertical_kdim`, whose only job was to snap a gridpoint count to a
multiple of `mubar`; sizing the spline axis by cells makes that rounding unnecessary (and the
rounding was what left the RiRk vertical 3x coarser in cells than the horizontal).

Splat it into the *keyword* section of the call — note the leading `;`, without which Julia
splats the NamedTuple positionally and the constructor fails:

```julia
grid_params = GridParameters(;
    geometry = benchmark_geometry(opts),
    iMin = 0.0, iMax = 25.6e3, num_cells_i = num_cells_i,
    kMin = 0.0, kMax = 6.4e3,
    vertical_size(opts; num_cells_k = num_cells_k, kDim = kDim)...,
    ...)
```
"""
vertical_size(opts::BenchmarkOptions; num_cells_k::Int, kDim::Int) =
    opts.grid == :rirk ? (; num_cells_k = num_cells_k) : (; kDim = kDim)

"""Path suffix distinguishing non-default grids (empty for the default RZ grid)."""
grid_suffix(opts::BenchmarkOptions) = opts.grid == :rz ? "" : "_$(opts.grid)"

"""Suffix distinguishing nested-run artifacts (`_n3`); empty for single-grid runs."""
nest_suffix(opts::BenchmarkOptions) = opts.nests > 1 ? "_n$(opts.nests)" : ""

"""Output directory for a benchmark variant (created if missing)."""
function benchmark_output_dir(name::String, opts::BenchmarkOptions)
    dir = joinpath(BENCHMARKS_DIR, "output",
                   "$(name)_$(opts.mode)_$(opts.stage)$(grid_suffix(opts))$(nest_suffix(opts))")
    mkpath(dir)
    return dir * "/"   # integrate_model concatenates paths with *
end

"""
Committed regression reference path for a benchmark variant. Quick mode
references store the full final-time fields (small grids); full mode stores
the final scalar diagnostics (full fields are tens of MB).
"""
function reference_csv_path(name::String, opts::BenchmarkOptions)
    if opts.mode == :full
        return joinpath(REFERENCE_DATA_DIR, name,
                        "full_$(opts.stage)$(grid_suffix(opts))_diagnostics.csv")
    end
    return joinpath(REFERENCE_DATA_DIR, name,
                    "$(opts.mode)_$(opts.stage)$(grid_suffix(opts))_final.csv")
end

"""Write a full-mode regression reference: target diagnostics as name,value rows."""
function write_diagnostics_reference(path::String, diags::Dict{String,Float64},
                                     targets::Vector{Target})
    names = sort([t.name for t in targets])
    mkpath(dirname(path))
    open(path, "w") do f
        println(f, "diagnostic,value")
        for n in names
            println(f, "$(n),$(diags[n])")
        end
    end
    return path
end

"""
Compare target diagnostics against a committed full-mode reference. Only the
published-target diagnostics are compared (conservation drifts are
machine-precision residuals whose relative run-to-run change is meaningless).
"""
function compare_diagnostics_reference(path::String, diags::Dict{String,Float64};
                                       rtol=1.0e-6)
    ref = CSV.read(path, DataFrame)
    stats = Dict{String,Tuple{Float64,Float64}}()
    pass = true
    for row in eachrow(ref)
        name = row.diagnostic
        if haskey(diags, name)
            diff = abs(diags[name] - row.value)
            rel = diff / max(abs(row.value), eps())
            stats[name] = (rel, diff)
            pass &= rel <= rtol
        else
            stats[name] = (Inf, Inf)
            pass = false
        end
    end
    return pass, stats
end

"""
    load_targets(name, opts) -> Vector{Target}

Load published target values for a benchmark. Quick mode and the primitive
equation stage use the wider quick tolerances.
"""
function load_targets(name::String, opts::BenchmarkOptions)
    haskey(BENCHMARK_EXPECTED, name) || error("No expected values defined for $name")
    use_quick_tol = opts.mode == :quick ||
                    opts.stage in (:pe, STAGE_PE_RHOD, STAGE_PE_RHOD_PD, STAGE_MC)
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
    write_diagnostics_csv(path, diags, targets, target_pass)

Write every computed diagnostic (targets with their target/atol/pass/source,
then informational ones) to a CSV, mirroring [`report_table`](@ref). Written on
every run — independent of `--update-reference` — so min/max w and the other
metrics can be assessed after the fact.
"""
function write_diagnostics_csv(path::String, diags::Dict{String,Float64},
                               targets::Vector{Target}, target_pass::Dict{String,Bool})
    target_names = Set(t.name for t in targets)
    open(path, "w") do f
        println(f, "diagnostic,value,target,atol,pass,source")
        for t in targets
            value = get(diags, t.name, NaN)
            pass = target_pass[t.name] ? "PASS" : "FAIL"
            println(f, "$(t.name),$(value),$(t.value),$(t.atol),$(pass),\"$(t.source)\"")
        end
        for key in sort(collect(keys(diags)))
            if !(key in target_names)
                println(f, "$(key),$(diags[key]),,,,informational")
            end
        end
    end
    return path
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
    println("Benchmark: $name  mode=$(opts.mode)  stage=$(opts.stage)  grid=$(opts.grid)  " *
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
    diag_csv = write_diagnostics_csv(joinpath(model.output_dir, "diagnostics.csv"),
                                     diags, targets, target_pass)
    println("\nSaved diagnostics: $diag_csv")
    targets_ok = all(values(target_pass))

    # Regression comparison against committed reference output
    final_tag = string(round(model.integration_time; digits=2))
    output_csv = joinpath(model.output_dir, "$(final_tag)_physical.csv")
    ref_csv = reference_csv_path(name, opts)
    regression_ok = nothing
    regression_stats = Dict{String,Tuple{Float64,Float64}}()
    if opts.update_reference
        if opts.mode == :full
            write_diagnostics_reference(ref_csv, diags, targets)
        else
            write_reference(output_csv, ref_csv, varnames)
        end
        println("\nUpdated regression reference: $ref_csv")
    elseif isfile(ref_csv)
        if opts.mode == :full
            regression_ok, regression_stats = compare_diagnostics_reference(ref_csv, diags)
            println("\nRegression vs committed diagnostics ($(basename(ref_csv))):")
            for (name, (rel, diff)) in sort(collect(regression_stats), by = first)
                @printf("  %-16s rel = %.3e  abs = %.3e  %s\n",
                        name, rel, diff, rel <= 1.0e-6 ? "PASS" : "FAIL")
            end
        else
            regression_ok, regression_stats = compare_reference(output_csv, ref_csv, varnames)
            println("\nRegression vs committed reference ($(basename(ref_csv))):")
            if isempty(regression_stats)
                println("  FAIL (incomparable grids — no per-variable stats)")
            else
                for var in varnames
                    rel_l2, max_abs = regression_stats[var]
                    @printf("  %-8s rel_L2 = %.3e  max_abs = %.3e  %s\n",
                            var, rel_l2, max_abs, rel_l2 <= 1.0e-6 ? "PASS" : "FAIL")
                end
            end
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
        "grid" => opts.grid,
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

"""
    run_nested_benchmark(name, opts; nest, init!, diagnostics) -> Bool

Nested-grid counterpart of [`run_benchmark`](@ref): builds the nest, runs
`init!(models, topo)`, integrates via `integrate_nested_model`, and checks
`diagnostics(models, topo)` against the same published target windows.
Regression uses the scalar-diagnostics reference (per-nest field CSVs are not
committed); the reference file carries the `_n<nests>` suffix so single-grid
references are untouched.
"""
function run_nested_benchmark(name::String, opts::BenchmarkOptions;
                              nest, init!::Function, diagnostics::Function)

    models, topo = build_nest(nest)
    n = length(models)

    println("═"^70)
    println("Benchmark: $name  mode=$(opts.mode)  stage=$(opts.stage)  grid=$(opts.grid)  " *
            "nests=$(opts.nests) ($(n) patches)  equation_set=$(nest.base.equation_set)")
    for (i, m) in enumerate(models)
        nsteps = round(Int, m.integration_time / m.ts)
        println("  nest$i: [$(m.grid_params.iMin/1000), $(m.grid_params.iMax/1000)] km  " *
                "num_cells=$(m.grid_params.num_cells)  ts=$(m.ts) s (n_sub=$(topo.n_sub[i]))  steps=$nsteps")
    end
    println("═"^70)

    println("Generating reference state and initial conditions...")
    init!(models, topo)

    println("Integrating...")
    wallclock = @elapsed integrate_nested_model(nest)
    @printf("Wall clock: %.1f s\n", wallclock)

    println("Computing diagnostics...")
    diags = diagnostics(models, topo)

    targets = load_targets(name, opts)
    target_pass = check_targets(diags, targets)
    report_table(name, opts, diags, targets, target_pass)
    diag_csv = write_diagnostics_csv(joinpath(nest.base.output_dir, "diagnostics.csv"),
                                     diags, targets, target_pass)
    println("\nSaved diagnostics: $diag_csv")
    targets_ok = all(values(target_pass))

    ref_csv = joinpath(REFERENCE_DATA_DIR, name,
                       "$(opts.mode)_$(opts.stage)$(grid_suffix(opts))$(nest_suffix(opts))_diagnostics.csv")
    regression_ok = nothing
    if opts.update_reference
        write_diagnostics_reference(ref_csv, diags, targets)
        println("\nUpdated regression reference: $ref_csv")
    elseif isfile(ref_csv)
        regression_ok, regression_stats = compare_diagnostics_reference(ref_csv, diags)
        println("\nRegression vs committed diagnostics ($(basename(ref_csv))):")
        for (dname, (rel, diff)) in sort(collect(regression_stats), by = first)
            @printf("  %-16s rel = %.3e  abs = %.3e  %s\n",
                    dname, rel, diff, rel <= 1.0e-6 ? "PASS" : "FAIL")
        end
    else
        println("\nNo committed reference at $ref_csv")
        println("(re-run with --update-reference after accepting this result)")
    end

    scythe_sha, scythe_dirty = git_info(normpath(joinpath(@__DIR__, "..", "..")))
    springsteel_sha, _ = git_info(pkgdir(Springsteel))

    passed = targets_ok && (isnothing(regression_ok) || regression_ok)
    record = Dict(
        "timestamp" => string(now()),
        "case" => name,
        "mode" => opts.mode,
        "stage" => opts.stage,
        "grid" => opts.grid,
        "nests" => opts.nests,
        "equation_set" => nest.base.equation_set,
        "scythe_sha" => scythe_sha,
        "scythe_dirty" => scythe_dirty,
        "springsteel_sha" => springsteel_sha,
        "julia_version" => string(VERSION),
        "hostname" => gethostname(),
        "nworkers" => nworkers(),
        "ts" => join(topo.ts_actual, "|"),
        "num_cells" => join([m.grid_params.num_cells for m in models], "|"),
        "kDim" => nest.base.grid_params.kDim,
        "integration_time" => nest.base.integration_time,
        "wallclock_s" => round(wallclock, digits=2),
        "diagnostics" => Dict(k => v for (k, v) in diags),
        "target_pass" => target_pass,
        "regression_pass" => regression_ok,
        "passed" => passed,
    )
    record_result(name, record)

    println()
    println(passed ? "✓ $name [$(opts.mode)/$(opts.stage)/n$(opts.nests)] PASSED" :
                     "✗ $name [$(opts.mode)/$(opts.stage)/n$(opts.nests)] FAILED")
    return passed
end
