# Shared harness for Scythe benchmark scripts.
#
# Each benchmark script builds a ModelParameters configuration for the requested
# mode/stage, provides an init! function that generates the reference sounding
# and initial conditions, and a diagnostics function that reduces the final
# output to scalar values. The harness runs the model, checks the diagnostics
# against published target values, compares the output fields against committed
# reference data, and appends a JSON record with timing and provenance.

using Distributed
using LinearAlgebra: BLAS
using Printf
using Dates
using CSV
using DataFrames

# WHY EVERY BENCHMARK PINS CSV.
#
# The model's default `options[:output_formats]` is `[:netcdf]` -- one comprehensive
# `<t>.nc` per output time, written by the model itself. The benchmark harness does not read
# that: `read_final_output`, `compare_reference`, `write_reference`, `conservation_drift` and
# every per-case diagnostic parse `<t>_physical.csv` / `<t>_spectral.csv`, and the committed
# regression references ARE CSVs. A benchmark that inherited the default would produce no
# CSV at all and fail with a missing file, or -- far worse for a regression suite -- quietly
# compare against a stale one left over from an earlier run.
#
# So the format is PINNED here and spliced into every benchmark's options Dict, rather than
# left to the default. It is a list so a benchmark that also wants the comprehensive file
# can write `[BENCHMARK_OUTPUT_FORMATS..., :netcdf]` without guessing what the harness needs.
const BENCHMARK_OUTPUT_FORMATS = [:csv]

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
    worker_threads::Int # --worker-threads N: Julia threads per worker (0 = CPU_THREADS ÷ workers)
end

# Positional 10-argument form, kept for callers that predate `worker_threads`
# (test/test_benchmark_smoke.jl builds options this way).
BenchmarkOptions(mode, stage, grid, workers, update_reference, plot, ts_factor, nests, hsi, xsi) =
    BenchmarkOptions(mode, stage, grid, workers, update_reference, plot, ts_factor, nests, hsi,
                     xsi, 0)

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
--grid rz|rirk (default rz), --workers N (default 2), --worker-threads N (Julia threads
per worker; default 0 = `Sys.CPU_THREADS ÷ workers`; env fallback `SCYTHE_BENCH_THREADS`),
--ts-factor F (RiRk timestep scale, default 1.0), --update-reference, --plot.
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
    worker_threads = parse(Int, get(ENV, "SCYTHE_BENCH_THREADS", "0"))
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
        elseif arg == "--worker-threads"
            worker_threads = parse(Int, args[i+1]); i += 2
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
                    "[--workers N] [--worker-threads N] [--ts-factor F] [--nests N] [--hsi] " *
                    "[--exact-si] [--update-reference] [--plot]")
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
    worker_threads >= 0 || error("--worker-threads must be >= 0 (0 = CPU_THREADS ÷ workers)")
    ts_factor > 0.0 || error("--ts-factor must be > 0")
    nests >= 1 || error("--nests must be >= 1")
    hsi && (stage == STAGE_MC && grid == :rirk ||
        error("--hsi requires --stage mc --grid rirk"))
    xsi && (stage == STAGE_MC && grid == :rirk ||
        error("--exact-si requires --stage mc --grid rirk"))
    (hsi && xsi) && error("--hsi and --exact-si are mutually exclusive")
    return BenchmarkOptions(mode, stage, grid, nworkers, update_reference, plot, ts_factor, nests,
                            hsi, xsi, worker_threads)
end

"""Geometry string for the configured vertical basis (RZ Chebyshev vs RiRk B-spline)."""
benchmark_geometry(opts::BenchmarkOptions) = opts.grid == :rirk ? "RiRk" : "RZ"

"""
    add_benchmark_workers(opts; count = opts.workers)

Add the worker processes, dividing the machine's threads *among* them rather than giving
each one the whole box, and pin each worker's OpenBLAS to one thread.

`--threads=auto` hands every worker `Sys.CPU_THREADS` threads, so the default 2 workers on a
12-core machine spawn 24 compute threads for 12 cores. They then all allocate into a shared
GC from inside the `Threads.@threads` column loop, which shows up as heavy GC time and
millions of lock conflicts — and is a suspected contributor to the intermittent worker death
in long moist_compressible runs. `opts.worker_threads > 0` overrides the split (the
reproducibility ladder needs 1 worker × 1 thread, which the split alone cannot express).

OpenBLAS defaults to `Sys.CPU_THREADS` threads per process on top of the Julia threads:
2 workers × 6 Julia threads × 12 BLAS threads = 144 runnable threads on 12 cores. The
periodic-BC spline fit on the RiRk grid solves through LAPACK `potrs` (Springsteel
`DenseSplineFactor`), so BLAS IS on the fit path, and a threaded BLAS partitioning is the one
timing-dependent reduction order a fixed-thread-count run could still carry. The pin is
applied HERE, never at the top level of this file (test/test_benchmark_smoke.jl includes it).
`SCYTHE_BENCH_BLAS_THREADS=0` leaves OpenBLAS at its default for before/after checks.
"""
function add_benchmark_workers(opts::BenchmarkOptions; count::Int = opts.workers)
    nthreads = opts.worker_threads > 0 ? opts.worker_threads : max(1, Sys.CPU_THREADS ÷ count)
    blas = parse(Int, get(ENV, "SCYTHE_BENCH_BLAS_THREADS", "1"))
    println("Adding $(count) worker(s) with $(nthreads) thread(s) each " *
            "($(Sys.CPU_THREADS) CPU threads available); BLAS threads per worker: " *
            (blas > 0 ? string(blas) : "OpenBLAS default"))
    pids = addprocs(count, exeflags = "--threads=$(nthreads)")
    if blas > 0
        BLAS.set_num_threads(blas)
        # Quoted expressions rather than `@everywhere ... using ...`: a `using` inside a
        # macro call in a function body is rejected as a non-toplevel expression.
        Distributed.remotecall_eval(Main, pids, :(using LinearAlgebra))
        Distributed.remotecall_eval(Main, pids,
                                    :(LinearAlgebra.BLAS.set_num_threads($blas)))
    end
    return pids
end

"""
    worker_provenance() -> Dict

Thread/BLAS/bounds-check settings as the FIRST WORKER sees them (the master's values are not
what the model ran under), for the JSONL record. Every entry is a plain Int; -1 means the
worker could not report it (no workers, or LinearAlgebra not loaded there).
"""
function worker_provenance()
    isempty(workers()) && return Dict("worker_threads" => -1, "blas_threads" => -1,
                                      "check_bounds" => -1)
    w = workers()[1]
    wt = remotecall_fetch(Threads.nthreads, w)
    bt = remotecall_fetch(w) do
        isdefined(Main, :LinearAlgebra) ? Main.LinearAlgebra.BLAS.get_num_threads() : -1
    end
    cb = remotecall_fetch(() -> Int(Base.JLOptions().check_bounds), w)
    return Dict("worker_threads" => wt, "blas_threads" => bt, "check_bounds" => cb)
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

"""
Path/key suffix carrying an explicit ARM token (e.g. `"ice"`), derived by the caller from
the resolved MODEL CONFIG -- never from `SCYTHE_BENCH_TAG`, which is a free-text run label
for `output_dir` only and has nothing to do with which reference file or target windows a
run is checked against. Empty when no arm is given (the default everywhere it is accepted),
which is what keeps the plain (unarmed) path and target lookup exactly what they always
were -- byte-identical, not merely equivalent.
"""
arm_suffix(arm::String) = isempty(arm) ? "" : "_$(arm)"

"""
Output directory for a benchmark variant (created if missing).

`SCYTHE_BENCH_TAG` appends a run label, e.g. `_ladder_bhyp`. Sweeps otherwise all write to
the same directory and have to be `mv`d afterwards — which is how every archived sweep here
was made, and why the `output_dir` recorded inside those runs' own logs points somewhere
else. Unset, the path is exactly what it always was, so no committed reference moves.
"""
function benchmark_output_dir(name::String, opts::BenchmarkOptions)
    tag = get(ENV, "SCYTHE_BENCH_TAG", "")
    dir = joinpath(BENCHMARKS_DIR, "output",
                   "$(name)_$(opts.mode)_$(opts.stage)$(grid_suffix(opts))$(nest_suffix(opts))$(tag)")
    mkpath(dir)
    return dir * "/"   # integrate_model concatenates paths with *
end

"""
Committed regression reference path for a benchmark variant. Quick mode
references store the full final-time fields (small grids); full mode stores
the final scalar diagnostics (full fields are tens of MB).

`arm` (default `""`, meaning "no arm") is an explicit token for a benchmark variant that
carries genuinely different prognostics/physics from the case's default configuration --
e.g. `"ice"` for `o01_rainfall`'s ISHMAEL arm, which appends 12 ice slots plus a prognostic
rain number and turns on transforms the default doesn't carry. It inserts an `arm_suffix`
segment ahead of `_final`/`_diagnostics` (e.g. `quick_mc_rirk_ice_final.csv`), so an armed
run reads and writes its OWN committed reference and can never collide with, or
`--update-reference`-overwrite, the unarmed default's file. With the default `arm=""` this
function returns EXACTLY what it always has -- `arm_suffix("")` is the empty string, so the
unarmed path is untouched, byte-for-byte.
"""
function reference_csv_path(name::String, opts::BenchmarkOptions; arm::String="")
    if opts.mode == :full
        return joinpath(REFERENCE_DATA_DIR, name,
                        "full_$(opts.stage)$(grid_suffix(opts))$(arm_suffix(arm))_diagnostics.csv")
    end
    return joinpath(REFERENCE_DATA_DIR, name,
                    "$(opts.mode)_$(opts.stage)$(grid_suffix(opts))$(arm_suffix(arm))_final.csv")
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
    load_targets(name, opts; arm="") -> Vector{Target}

Load published target values for a benchmark. Quick mode and the primitive
equation stage use the wider quick tolerances.

`arm` (default `""`) looks the targets up under the ARM-QUALIFIED key `"\$(name)_\$(arm)"`
instead of `name` -- e.g. `"o01_rainfall_ice"` for the ISHMAEL arm of `o01_rainfall`. There
is deliberately NO fallback to the base `name` entry when the arm-qualified key is missing
or its Dict is empty: an armed run's diagnostics differ physically from the unarmed case
(different closures, different prognostics), so borrowing the warm windows would silently
check the wrong physics against the wrong numbers. Absent arm-specific windows, an armed
run gets NO targets at all -- an empty `Vector{Target}` -- rather than the warm case's. With
the default `arm=""` this looks up `name` exactly as before and still errors if it is
undefined, so the unarmed path's behavior is unchanged.
"""
function load_targets(name::String, opts::BenchmarkOptions; arm::String="")
    key = isempty(arm) ? name : "$(name)_$(arm)"
    # A mode-qualified window set wins where one exists (the nested-arm precedent): the
    # full-resolution ice storm is not the quick storm re-run finer (accum 0.90 vs 3.7 on
    # the accepted 2026-09-01 run -- more water held in the anvil, hour-1 rain delayed),
    # so its windows carry their own centers. The ice arm's rtol-1e-6 diagnostics
    # reference is seeded like any other: the run-to-run spread once recorded against it
    # (FINDINGS 4d') was two code states across a commit, and the arm reproduces bitwise
    # on the same code (FINDINGS 5n).
    if opts.mode == :full && haskey(BENCHMARK_EXPECTED, "$(key)_full")
        key = "$(key)_full"
    end
    if !haskey(BENCHMARK_EXPECTED, key)
        isempty(arm) && error("No expected values defined for $name")
        return Target[]
    end
    use_quick_tol = opts.mode == :quick ||
                    opts.stage in (:pe, STAGE_PE_RHOD, STAGE_PE_RHOD_PD, STAGE_MC)
    targets = Target[]
    for (diag, (value, atol_full, atol_quick, source)) in BENCHMARK_EXPECTED[key]
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
Returns (pass, Dict of var => (rel_l2, max_abs_diff)), or `(nothing, Dict())` when the two
files do not carry the same variables at all — which is not a failure but an absence of a
comparison, and must not be reported as either a pass or a regression.
"""
function compare_reference(output_csv::String, reference_csv::String,
                           varnames::Vector{String}; rtol=1.0e-6)
    out = CSV.read(output_csv, DataFrame)
    ref = CSV.read(reference_csv, DataFrame)
    # A run that declares a control-variable transform carries DIFFERENT prognostics from the
    # committed (untransformed) reference — `nu_c` where the reference has `rho_c` — and the
    # names say so. Comparing them column by column is meaningless: even where the physics is
    # identical the values differ by the map. Skip, and say why.
    missing_cols = setdiff(varnames, names(ref))
    if !isempty(missing_cols)
        println("  Reference comparison SKIPPED: this run carries $(missing_cols), which the " *
                "committed reference does not. A transformed run is not comparable to an " *
                "untransformed reference; judge it on its diagnostics and targets.")
        return nothing, Dict{String,Tuple{Float64,Float64}}()
    end
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

`varnames` already carries the run's OWN slot names (e.g. `o01_rainfall.jl` resolves them
via `Scythe.mc_var_names(model.options)`), so a control-variable transform's renamed slots
-- `nu_c`/`nu_r`/the twelve `nu_i*` ice moments under the `MC_NU_ALIAS` naming -- are written
under those names, not the untransformed `rho_*` names. This is deliberate: the ice arm's
reference stores the TRANSFORMED control variables it actually integrated, and
`compare_reference` above already handles this correctly with no changes needed here --
it compares `varnames` by name against whatever columns the committed reference has, and
skips (rather than false-failing) when a run's `nu_*` names aren't present in an untransformed
reference.
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
        # Tracked modifications only. Untracked scratch (notebooks, Finder files) cannot
        # enter a run -- Scythe.jl includes its files explicitly -- and counting it made
        # every record on file read dirty=true, which hid the one distinction that matters
        # for provenance (FINDINGS_ISHMAEL_S8S9 §4d′ compared two dirty trees across a commit).
        dirty = !isempty(strip(read(`git -C $dir status --porcelain --untracked-files=no`,
                                    String)))
        return String(sha), dirty
    catch
        return "unknown", false
    end
end

"""Number of untracked (non-ignored) paths in `dir`, recorded beside `scythe_dirty`."""
function git_untracked_count(dir::String)
    try
        out = read(`git -C $dir status --porcelain --untracked-files=all`, String)
        return count(l -> startswith(l, "??"), split(out, '\n'))
    catch
        return -1
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
    run_benchmark(name, opts; model, init!, diagnostics, varnames, arm="") -> Bool

Run a benchmark end-to-end: initialize, integrate, verify, report, record.
Returns true if all targets pass and the regression comparison (when a
committed reference exists) is within tolerance.

`arm` (default `""`) is threaded straight through to [`reference_csv_path`](@ref) and
[`load_targets`](@ref): it is the caller's job to derive it from the RESOLVED model config
(e.g. `Scythe.ice_microphysics(model.options) === :ishmael ? "ice" : ""`), never from
`SCYTHE_BENCH_TAG`. Because both of those default to `arm=""` and return their unarmed
result unchanged in that case, an unarmed call to this function (the default) reads/writes
exactly the reference file and target windows it always did -- an armed call can never see,
let alone overwrite with `--update-reference`, the unarmed default's committed reference.
"""
function run_benchmark(name::String, opts::BenchmarkOptions;
                       model, init!::Function, diagnostics::Function,
                       varnames::Vector{String}, plotter=nothing, arm::String="")

    println("═"^70)
    arm_tag = isempty(arm) ? "" : "  arm=$(arm)"
    println("Benchmark: $name  mode=$(opts.mode)  stage=$(opts.stage)  grid=$(opts.grid)  " *
            "equation_set=$(model.equation_set)$(arm_tag)")
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

    targets = load_targets(name, opts; arm=arm)
    target_pass = check_targets(diags, targets)
    report_table(name, opts, diags, targets, target_pass)
    diag_csv = write_diagnostics_csv(joinpath(model.output_dir, "diagnostics.csv"),
                                     diags, targets, target_pass)
    println("\nSaved diagnostics: $diag_csv")
    targets_ok = all(values(target_pass))

    # Regression comparison against committed reference output
    final_tag = string(round(model.integration_time; digits=2))
    output_csv = joinpath(model.output_dir, "$(final_tag)_physical.csv")
    ref_csv = reference_csv_path(name, opts; arm=arm)
    regression_ok = nothing
    regression_stats = Dict{String,Tuple{Float64,Float64}}()
    if opts.update_reference
        # ref_csv already carries the arm suffix (empty for an unarmed run), so
        # `--update-reference` on an ARMED run writes ONLY its own arm-qualified path and
        # can never clobber the unarmed default's committed reference -- this follows
        # automatically from `reference_csv_path`'s `arm` keyword, not from any check here.
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
            println("\nRegression vs committed reference ($(basename(ref_csv))):")
            regression_ok, regression_stats = compare_reference(output_csv, ref_csv, varnames)
            if isempty(regression_stats)
                regression_ok === nothing ||
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

    scythe_root = normpath(joinpath(@__DIR__, "..", ".."))
    scythe_sha, scythe_dirty = git_info(scythe_root)
    springsteel_sha, _ = git_info(pkgdir(Springsteel))
    prov = worker_provenance()

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
        "scythe_untracked" => git_untracked_count(scythe_root),
        "springsteel_sha" => springsteel_sha,
        "julia_version" => string(VERSION),
        "hostname" => gethostname(),
        "nworkers" => nworkers(),
        "worker_threads" => prov["worker_threads"],
        "master_threads" => Threads.nthreads(),
        "blas_threads" => prov["blas_threads"],
        "check_bounds" => prov["check_bounds"],
        "bench_tag" => get(ENV, "SCYTHE_BENCH_TAG", ""),
        "tstop_env" => get(ENV, "SCYTHE_O01_TSTOP", ""),
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
    run_nested_benchmark(name, opts; nest, init!, diagnostics, arm="") -> Bool

Nested-grid counterpart of [`run_benchmark`](@ref): builds the nest, runs
`init!(models, topo)`, integrates via `integrate_nested_model`, and checks
`diagnostics(models, topo)` against the same published target windows.
Regression uses the scalar-diagnostics reference (per-nest field CSVs are not
committed); the reference file carries the `_n<nests>` suffix so single-grid
references are untouched.

`arm` (default `""`) behaves exactly as in [`run_benchmark`](@ref): it is threaded through
to [`load_targets`](@ref) and appended (via `arm_suffix`) to this function's own inline
reference-path construction below, ahead of `_diagnostics.csv`. Unarmed calls (`arm=""`,
the default) are unaffected -- byte-identical path and target lookup.
"""
function run_nested_benchmark(name::String, opts::BenchmarkOptions;
                              nest, init!::Function, diagnostics::Function, arm::String="")

    models, topo = build_nest(nest)
    n = length(models)

    println("═"^70)
    arm_tag = isempty(arm) ? "" : "  arm=$(arm)"
    println("Benchmark: $name  mode=$(opts.mode)  stage=$(opts.stage)  grid=$(opts.grid)  " *
            "nests=$(opts.nests) ($(n) patches)  equation_set=$(nest.base.equation_set)$(arm_tag)")
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

    # A nested run is judged against ITS OWN window key when one is seeded
    # (e.g. "o01_rainfall_n3"), falling back to the base case's windows otherwise.
    # The nested diagnostics define max_w/min_w as the extrema over EVERY output
    # snapshot (user decision 2026-07-14), while the single-grid diagnostics report
    # the final snapshot only — so windows derived from single-grid values are a
    # different unit and the nested arm could never pass them regardless of the
    # model (measured 2026-08-20: the single grid's own run-maximum w is 65.1 m/s
    # against its reported final-time 16.5).
    nested_key = "$(name)$(nest_suffix(opts))"
    targets_name = haskey(BENCHMARK_EXPECTED,
                          isempty(arm) ? nested_key : "$(nested_key)_$(arm)") ?
                   nested_key : name
    targets = load_targets(targets_name, opts; arm=arm)
    target_pass = check_targets(diags, targets)
    report_table(name, opts, diags, targets, target_pass)
    diag_csv = write_diagnostics_csv(joinpath(nest.base.output_dir, "diagnostics.csv"),
                                     diags, targets, target_pass)
    println("\nSaved diagnostics: $diag_csv")
    targets_ok = all(values(target_pass))

    ref_csv = joinpath(REFERENCE_DATA_DIR, name,
                       "$(opts.mode)_$(opts.stage)$(grid_suffix(opts))$(nest_suffix(opts))$(arm_suffix(arm))_diagnostics.csv")
    regression_ok = nothing
    if opts.update_reference
        # As in run_benchmark: ref_csv already carries the arm suffix (empty when unarmed),
        # so this can never write the unarmed default's path -- it follows from arm_suffix,
        # not from a check here.
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

    scythe_root = normpath(joinpath(@__DIR__, "..", ".."))
    scythe_sha, scythe_dirty = git_info(scythe_root)
    springsteel_sha, _ = git_info(pkgdir(Springsteel))
    prov = worker_provenance()

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
        "scythe_untracked" => git_untracked_count(scythe_root),
        "springsteel_sha" => springsteel_sha,
        "julia_version" => string(VERSION),
        "hostname" => gethostname(),
        "nworkers" => nworkers(),
        "worker_threads" => prov["worker_threads"],
        "master_threads" => Threads.nthreads(),
        "blas_threads" => prov["blas_threads"],
        "check_bounds" => prov["check_bounds"],
        "bench_tag" => get(ENV, "SCYTHE_BENCH_TAG", ""),
        "tstop_env" => get(ENV, "SCYTHE_O01_TSTOP", ""),
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
