# Run driver: execution-environment helpers (worker setup, provenance, git state) and the
# pure segment arithmetic, and the segmented run orchestrator built on them
# (`run_simulation`, `segment_main`, `parse_launcher_args`).
#
# The worker helpers moved here from benchmarks/common/harness.jl, which keeps thin
# forwarders under the old names so every benchmark and test/test_benchmark_smoke.jl works
# unchanged.

using ClusterManagers: addprocs_slurm
import ArgParse
using Dates

export ExecutionEnv, setup_workers!, checkpoint_path
export run_simulation, segment_main, parse_launcher_args

"""
    ExecutionEnv(; workers_per_patch = [1], worker_threads = 0, blas_threads = 1, slurm = false)

How a run is executed, as opposed to what it computes: it comes from the launcher's command
line, is recorded in `segments.csv` for provenance, and is never written to `config.toml`
(a run's physics must not depend on it).

# Fields
- `workers_per_patch::Vector{Int}` — distributed worker group size per patch (one entry for
  a single-patch run); [`setup_workers!`](@ref) adds `sum(workers_per_patch)` workers.
- `worker_threads::Int` — Julia threads per worker; `0` splits the machine,
  `Sys.CPU_THREADS ÷ nworkers` (see [`add_workers`](@ref)).
- `blas_threads::Int` — OpenBLAS threads per process; `0` leaves the OpenBLAS default.
- `slurm::Bool` — add the workers through `ClusterManagers.addprocs_slurm` (one per SLURM
  task in the allocation) instead of local `addprocs`.
"""
Base.@kwdef struct ExecutionEnv
    workers_per_patch::Vector{Int} = [1]
    worker_threads::Int = 0
    blas_threads::Int = 1
    slurm::Bool = false
end

"""
    _pin_blas_and_load(pids, blas_threads)

Pin OpenBLAS to `blas_threads` on the master and on `pids` (skipped when `blas_threads <= 0`),
then load Springsteel and Scythe on the new workers (plus `RRTMGP, ClimaComms` when the
master has the `ScytheRRTMGPExt` extension loaded). Quoted expressions through
`remotecall_eval` rather than `@everywhere using ...`: a `using` inside a macro call in a
function body is rejected as a non-toplevel expression.
"""
function _pin_blas_and_load(pids, blas_threads::Int)
    if blas_threads > 0
        LinearAlgebra.BLAS.set_num_threads(blas_threads)
        Distributed.remotecall_eval(Main, pids, :(using LinearAlgebra))
        Distributed.remotecall_eval(Main, pids,
                                    :(LinearAlgebra.BLAS.set_num_threads($blas_threads)))
    end
    Distributed.remotecall_eval(Main, pids, :(using Springsteel, Scythe))
    # RRTMGP/ClimaComms are WEAK dependencies: the RRTMGP radiation path lives in the
    # package extension `ScytheRRTMGPExt`, which Julia loads per process only once that
    # process has imported the trigger packages. The master having it loaded does not load
    # it on the workers, and every worker that builds a radiating tile needs its own copy,
    # so mirror the master. `get_extension` returns `nothing` when the extension is not
    # loaded (or does not exist in this build), in which case the workers skip it.
    if Base.get_extension(Scythe, :ScytheRRTMGPExt) !== nothing
        Distributed.remotecall_eval(Main, pids, :(import RRTMGP, ClimaComms))
    end
    return pids
end

"""
    add_workers(count::Int; worker_threads::Int = 0, blas_threads::Int = 1) -> Vector{Int}

Add `count` local worker processes, dividing the machine's threads *among* them rather than
giving each one the whole box, pin each worker's OpenBLAS, and load Springsteel and Scythe
on them. Returns the new worker pids.

`--threads=auto` hands every worker `Sys.CPU_THREADS` threads, so 2 workers on a 12-core
machine spawn 24 compute threads for 12 cores. They then all allocate into a shared GC from
inside the `Threads.@threads` column loop, which shows up as heavy GC time and millions of
lock conflicts — and is a suspected contributor to the intermittent worker death in long
moist_compressible runs. `worker_threads > 0` overrides the split
(`Sys.CPU_THREADS ÷ count`, at least 1); the reproducibility ladder needs 1 worker ×
1 thread, which the split alone cannot express.

OpenBLAS defaults to `Sys.CPU_THREADS` threads per process on top of the Julia threads:
2 workers × 6 Julia threads × 12 BLAS threads = 144 runnable threads on 12 cores. The
periodic-BC spline fit on the RiRk grid solves through LAPACK `potrs` (Springsteel
`DenseSplineFactor`), so BLAS IS on the fit path, and a threaded BLAS partitioning is the one
timing-dependent reduction order a fixed-thread-count run could still carry. `blas_threads`
(default 1) pins the master and every worker; `blas_threads = 0` leaves OpenBLAS at its
default for before/after checks.
"""
function add_workers(count::Int; worker_threads::Int = 0, blas_threads::Int = 1)
    count >= 1 || throw(ArgumentError("add_workers: count must be >= 1, got $count"))
    worker_threads >= 0 || throw(ArgumentError(
        "add_workers: worker_threads must be >= 0 (0 = CPU_THREADS ÷ workers)"))
    nthreads = worker_threads > 0 ? worker_threads : max(1, Sys.CPU_THREADS ÷ count)
    println("Adding $(count) worker(s) with $(nthreads) thread(s) each " *
            "($(Sys.CPU_THREADS) CPU threads available); BLAS threads per worker: " *
            (blas_threads > 0 ? string(blas_threads) : "OpenBLAS default"))
    pids = addprocs(count, exeflags = "--threads=$(nthreads)")
    return _pin_blas_and_load(pids, blas_threads)
end

"""
    add_slurm_workers(count::Int; worker_threads::Int = 0, blas_threads::Int = 1,
                      kwargs...) -> Vector{Int}

Add `count` workers through `ClusterManagers.addprocs_slurm` (inside a SLURM allocation; the
workers land on the allocated nodes via `srun`), started in the master's active project, then
pin BLAS and load Springsteel and Scythe exactly as [`add_workers`](@ref) does.
`worker_threads > 0` adds `--threads=N` to the worker command line; `0` leaves the thread
count to the environment (`JULIA_NUM_THREADS` from the batch script). Extra `kwargs` pass
through to `addprocs_slurm` (e.g. `partition`, `time`).

`integrate_nested_model` partitions `workers()` in order, so with one worker per patch and
one task per node the node↔patch mapping is automatic.
"""
function add_slurm_workers(count::Int; worker_threads::Int = 0, blas_threads::Int = 1,
                           kwargs...)
    count >= 1 || throw(ArgumentError("add_slurm_workers: count must be >= 1, got $count"))
    worker_threads >= 0 || throw(ArgumentError(
        "add_slurm_workers: worker_threads must be >= 0 (0 = environment default)"))
    exeflags = "--project=$(Base.active_project())"
    worker_threads > 0 && (exeflags *= " --threads=$(worker_threads)")
    println("Adding $(count) SLURM worker(s) " *
            (worker_threads > 0 ? "with $(worker_threads) thread(s) each" :
                                  "with the environment's thread count") *
            "; BLAS threads per worker: " *
            (blas_threads > 0 ? string(blas_threads) : "OpenBLAS default"))
    pids = addprocs_slurm(count; exeflags = exeflags, kwargs...)
    return _pin_blas_and_load(pids, blas_threads)
end

"""
    setup_workers!(exec::ExecutionEnv) -> Vector{Int}

Add `sum(exec.workers_per_patch)` workers for a run — through [`add_slurm_workers`](@ref)
when `exec.slurm`, else [`add_workers`](@ref) — with `exec`'s thread and BLAS settings,
and load Springsteel and Scythe on them. Returns the new worker pids.
"""
function setup_workers!(exec::ExecutionEnv)
    isempty(exec.workers_per_patch) && throw(ArgumentError(
        "ExecutionEnv.workers_per_patch must not be empty"))
    all(>=(1), exec.workers_per_patch) || throw(ArgumentError(
        "ExecutionEnv.workers_per_patch entries must be >= 1, got $(exec.workers_per_patch)"))
    count = sum(exec.workers_per_patch)
    return exec.slurm ?
        add_slurm_workers(count; worker_threads = exec.worker_threads,
                          blas_threads = exec.blas_threads) :
        add_workers(count; worker_threads = exec.worker_threads,
                    blas_threads = exec.blas_threads)
end

"""
    worker_provenance() -> Dict

Thread/BLAS/bounds-check settings as the FIRST WORKER sees them (the master's values are not
what the model ran under), for provenance records. Every entry is a plain Int; -1 means the
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
    git_info(dir::AbstractString) -> (sha::String, dirty::Bool)

Short git SHA and dirty flag for a repository directory, or `("unknown", false)`.

Dirty counts tracked modifications only. Untracked scratch (notebooks, Finder files) cannot
enter a run -- Scythe.jl includes its files explicitly -- and counting it made every record
on file read dirty=true, which hid the one distinction that matters for provenance
(FINDINGS_ISHMAEL_S8S9 §4d′ compared two dirty trees across a commit). The untracked count
is recorded separately by [`git_untracked_count`](@ref).
"""
function git_info(dir::AbstractString)
    try
        sha = strip(read(`git -C $dir rev-parse --short HEAD`, String))
        dirty = !isempty(strip(read(`git -C $dir status --porcelain --untracked-files=no`,
                                    String)))
        return String(sha), dirty
    catch
        return "unknown", false
    end
end

"""
    git_untracked_count(dir::AbstractString) -> Int

Number of untracked (non-ignored) paths in `dir`, recorded beside the dirty flag; -1 when
git cannot report it.
"""
function git_untracked_count(dir::AbstractString)
    try
        out = read(`git -C $dir status --porcelain --untracked-files=all`, String)
        return count(l -> startswith(l, "??"), split(out, '\n'))
    catch
        return -1
    end
end

# ── Segment arithmetic ──────────────────────────────────────────────────────────────────

"""
    seg_length(j::Int, N::Real, D::Real) -> Float64

Length of segment `j` of a run of total length `N` split into segments of length `D` (any
consistent unit): `D` for every full segment, the remainder `N - (j-1)·D` for the last one
when `N/D` is not an integer.
"""
seg_length(j::Int, N::Real, D::Real) = Float64(min(D, N - (j - 1) * D))

"""
    seg_offset(k::Int, N::Real, D::Real) -> Float64

Cumulative run time before segment `k` starts: the sum of [`seg_length`](@ref) over
segments `1:k-1` (0 for the first segment).
"""
seg_offset(k::Int, N::Real, D::Real) =
    sum(seg_length(j, N, D) for j in 1:(k - 1); init = 0.0)

"""
    whole_steps(len, dt) -> Bool

Whether `len` is a whole number of `dt` steps, to an absolute tolerance of 1e-9 on the
ratio. `integrate_model` rounds `integration_time/ts` to the step count, so a fractional
segment integrates past its nominal length while the checkpoint name and the next segment's
offset carry the nominal value — a silent clock skew. The segment driver holds segment
lengths, output and restart intervals to this rule for every patch.
"""
whole_steps(len, dt) = abs(len / dt - round(len / dt)) < 1.0e-9

# ── Segment plan and on-disk layout ─────────────────────────────────────────────────────
#
#   <root>/config.toml              resolved config (fresh start) + provenance header
#   <root>/reference.ref, ics.csv | nest{i}_ics.csv
#   <root>/segments.csv             one row per segment (SEGMENTS_CSV_COLS)
#   <root>/seg_NN/config.toml       the same config, with this segment's values in the header
#   <root>/seg_NN/segment_stdout.log
#   <root>/seg_NN/[nest{i}/]<t>.nc | <t>.jld2 | scythe_{out,err}.log
#
# Always the segmented layout, even for one segment, so readers handle one shape.

"""
    segment_plan(cfg::RunConfig) -> Vector{NamedTuple{(:index, :t_offset, :integration_time)}}

The segments of a run, in seconds: `ceil(integration_time / segment_length)` segments of
`segment_length` (the last one the remainder when the ratio is not an integer), or one
segment when `run.segment_length == 0`. A ratio within 1e-9 of an integer counts as that
integer, so floating-point noise cannot add a sliver segment.
"""
function segment_plan(cfg::RunConfig)
    N = cfg.run.integration_time
    D = cfg.run.segment_length
    D > 0.0 || return [(index = 1, t_offset = 0.0, integration_time = N)]
    r = N / D
    nseg = abs(r - round(r)) < 1.0e-9 ? max(1, round(Int, r)) : ceil(Int, r)
    return [(index = k, t_offset = seg_offset(k, N, D),
             integration_time = k == nseg ? N - seg_offset(k, N, D) : seg_length(k, N, D))
            for k in 1:nseg]
end

"`seg_NN` (two digits) — the directory name of segment `k`."
segment_dirname(k::Int) = "seg_" * lpad(k, 2, '0')

"Per-patch output directories of segment `k`: `<root>/seg_NN` (one patch) or `<root>/seg_NN/nest{i}`."
segment_patch_dirs(root::AbstractString, k::Int, npatches::Int) =
    npatches == 1 ? [joinpath(root, segment_dirname(k))] :
                    [joinpath(root, segment_dirname(k), "nest$i") for i in 1:npatches]

"Segment-1 initial-condition paths: `<root>/ics.csv` or `<root>/nest{i}_ics.csv` (the names `build_nest` derives)."
initial_ics_paths(root::AbstractString, npatches::Int) =
    npatches == 1 ? [joinpath(root, "ics.csv")] :
                    [joinpath(root, "nest$(i)_ics.csv") for i in 1:npatches]

"Model time [s] a checkpoint file name encodes (`<t>.jld2`), or `nothing`."
_checkpoint_time(path::AbstractString) =
    tryparse(Float64, replace(basename(path), r"\.jld2$" => ""))

"""
    find_segment_checkpoints(dirs::Vector{String}, t::Real) -> Vector{String}

The checkpoint each patch directory in `dirs` holds for the end of a segment of length `t`:
[`checkpoint_path`](@ref)`(dir, t)` when it exists. Otherwise the newest `.jld2` in that
directory (by mtime) with a WARNING — the naming rule did not hold, which is worth knowing
but not worth losing a finished segment over. Errors when a directory has no checkpoint,
and when the patches' chosen checkpoints disagree on the model time they encode (chaining
patches from different times would silently desynchronize the nest).
"""
function find_segment_checkpoints(dirs::AbstractVector{<:AbstractString}, t::Real)
    paths = String[]
    for dir in dirs
        expected = checkpoint_path(dir, t)
        if isfile(expected)
            push!(paths, expected)
            continue
        end
        candidates = isdir(dir) ? filter(f -> endswith(f, ".jld2"), readdir(dir)) : String[]
        isempty(candidates) && error("segment driver: no .jld2 checkpoint in $dir to restart " *
                                     "from (expected $expected)")
        newest = joinpath(dir, candidates[argmax([mtime(joinpath(dir, c)) for c in candidates])])
        @warn "segment driver: expected checkpoint $expected not found; using the newest " *
              "checkpoint in $dir instead: $newest"
        push!(paths, newest)
    end
    times = map(_checkpoint_time, paths)
    (any(isnothing, times) || !all(==(times[1]), times)) && error(
        "segment driver: the patches' restart checkpoints disagree on the model time: " *
        join(basename.(paths), ", ") * " (in " * join(dirs, ", ") * ")")
    return paths
end

# ── segments.csv ────────────────────────────────────────────────────────────────────────

"""
Columns of `<root>/segments.csv`: the legacy RCE columns, then `n_patches`, `ts_actual`
(per patch, `;`-joined), `start_hour` (the segment's solar clock), `workers`
(`workers_per_patch`, `;`-joined), `threads` (Julia threads per worker as the first worker
reports them; the requested value until the child records it), `host`, and `note`
(e.g. an integration-time extension on `--resume`).
"""
const SEGMENTS_CSV_COLS = ["segment", "dir", "t_offset_s", "integration_time_s", "ts",
                           "output_interval_s", "restart_file", "status", "scythe_sha",
                           "started", "finished", "n_patches", "ts_actual", "start_hour",
                           "workers", "threads", "host", "note"]

segments_csv_path(root::AbstractString) = joinpath(root, "segments.csv")

"""
    read_segments_csv(root) -> Dict{Int,Dict{String,String}}

The rows of `<root>/segments.csv` keyed by segment number, every value a String (`""` for
an empty cell); empty when the file does not exist.
"""
function read_segments_csv(root::AbstractString)
    path = segments_csv_path(root)
    rows = Dict{Int,Dict{String,String}}()
    isfile(path) || return rows
    for r in CSV.File(path; types = String, pool = false, strict = true)
        d = Dict{String,String}()
        for c in SEGMENTS_CSV_COLS
            v = Symbol(c) in propertynames(r) ? getproperty(r, Symbol(c)) : missing
            d[c] = v === missing ? "" : String(v)
        end
        rows[parse(Int, d["segment"])] = d
    end
    return rows
end

"""
    update_segment_row!(root, k, fields::AbstractDict) -> path

Merge `fields` (column => value, any value `string`-able) into segment `k`'s row of
`<root>/segments.csv` and rewrite the whole file sorted by segment. The one writer used by
both the orchestrator (`running` before launch, `failed` if the child died without saying)
and the child (`running` with the resolved values, then `done`/`failed`). The file is
small, so a read-modify-write per transition keeps it always complete on disk.
"""
function update_segment_row!(root::AbstractString, k::Int, fields::AbstractDict)
    rows = read_segments_csv(root)
    row = get!(() -> Dict{String,String}(c => "" for c in SEGMENTS_CSV_COLS), rows, k)
    for (c, v) in fields
        c = string(c)
        c in SEGMENTS_CSV_COLS || throw(ArgumentError("segments.csv has no column $(repr(c))"))
        row[c] = string(v)
    end
    row["segment"] = string(k)
    ks = sort(collect(keys(rows)))
    df = DataFrame([Symbol(c) => String[rows[j][c] for j in ks] for c in SEGMENTS_CSV_COLS])
    mkpath(root)
    CSV.write(segments_csv_path(root), df)
    return segments_csv_path(root)
end

"Recorded status of segment `k` in `<root>/segments.csv` (`\"\"` when absent)."
segment_status(root::AbstractString, k::Int) =
    get(get(read_segments_csv(root), k, Dict{String,String}()), "status", "")

# ── Guards and provenance ───────────────────────────────────────────────────────────────

"Model output files in a tree: `.nc`, `.jld2`, `*_physical.csv`, `*_gridded.csv` (first `limit`)."
function _model_output_files(root::AbstractString; limit::Int = 4)
    found = String[]
    isdir(root) || return found
    for (dir, _, files) in walkdir(root), f in files
        (endswith(f, ".nc") || endswith(f, ".jld2") ||
         endswith(f, "_physical.csv") || endswith(f, "_gridded.csv")) || continue
        push!(found, joinpath(relpath(dir, root), f))
        length(found) >= limit && break
    end
    return found
end

"""
    check_fresh_root(root)

Refuse a fresh start into a tree that already holds a run: `<root>/seg_01` exists, or the
tree holds model output. Model output is experimental data; the fix is to `mv` the tree
aside (never `rm`) or to continue it with `--resume`.
"""
function check_fresh_root(root::AbstractString)
    found = _model_output_files(root)
    seg1 = isdir(joinpath(root, segment_dirname(1)))
    (seg1 || !isempty(found)) || return nothing
    what = seg1 ? "$(joinpath(root, segment_dirname(1))) exists" :
        "it already holds model output ($(join(first(found, 3), ", "))" *
        "$(length(found) > 3 ? ", ..." : ""))"
    error("""
        refusing a fresh start in $(root): $what.

        Model output is experimental data and a fresh start would overwrite it file by file.
        Either continue this run with --resume, or mv the tree aside under a descriptive
        name (never rm it) and start again.""")
end

_exec_string(exec::ExecutionEnv) =
    "workers_per_patch = $(join(exec.workers_per_patch, ",")), worker_threads = " *
    "$(exec.worker_threads), blas_threads = $(exec.blas_threads), slurm = $(exec.slurm)"

function _git_string()
    dir = pkgdir(@__MODULE__)
    sha, dirty = git_info(dir)
    return "$sha (dirty = $dirty, untracked = $(git_untracked_count(dir)))"
end

"Short SHA with a `-dirty` suffix when tracked files are modified, for segments.csv."
function _sha_tag()
    sha, dirty = git_info(pkgdir(@__MODULE__))
    return dirty ? sha * "-dirty" : sha
end

function _provenance_header(exec::ExecutionEnv, sources, set)
    lines = ["Scythe run configuration: resolved, every field (defaults included).",
             "written:   $(Dates.now()) on $(gethostname())",
             "scythe:    $(pkgdir(@__MODULE__)) @ $(_git_string())",
             "sources:   " * (isempty(sources) ? "(constructed in Julia)" :
                              join(abspath.(sources), ", ")),
             "--set:     " * (isempty(set) ? "(none)" : join(set, " ")),
             "execution: $(_exec_string(exec))",
             "           (provenance only: the execution environment is never configuration)"]
    return join(lines, "\n")
end

"The leading `#` comment block of a config file (without the `# `), for rewriting it."
function _read_header(path::AbstractString)
    lines = String[]
    for l in eachline(path)
        startswith(l, "#") || break
        push!(lines, replace(l, r"^# ?" => ""))
    end
    return join(lines, "\n")
end

"The command-line flags that reproduce `exec` for `parse_launcher_args` (child launch)."
function exec_flags(exec::ExecutionEnv)
    f = ["--workers-per-patch", join(exec.workers_per_patch, ","),
         "--worker-threads", string(exec.worker_threads),
         "--blas-threads", string(exec.blas_threads)]
    exec.slurm && push!(f, "--slurm")
    return f
end

"""
    _load_rrtmgp!()

Import `RRTMGP, ClimaComms` into `Main` at run time (loading the `ScytheRRTMGPExt`
extension in this process) when the active environment can load them; a clear error when it
cannot. Callers continue through `Base.invokelatest` so the extension's methods are visible.
"""
function _load_rrtmgp!()
    Base.get_extension(@__MODULE__, :ScytheRRTMGPExt) === nothing || return nothing
    (Base.identify_package("RRTMGP") === nothing ||
     Base.identify_package("ClimaComms") === nothing) && error(
        "physics.radiation.scheme = \"rrtmgp\" but RRTMGP/ClimaComms cannot be loaded from " *
        "the active project $(Base.active_project()). Launch with " *
        "`julia --project=<Scythe>/simulations` (the workspace member that lists them).")
    Core.eval(Main, :(import RRTMGP, ClimaComms))
    return nothing
end

# ── Per-segment model and whole-step guards ─────────────────────────────────────────────

"""
    segment_model(cfg, seg, exec) -> (model, ts_actual::Vector{Float64})

`build_model(cfg; segment = seg)` with the execution environment's worker groups (a nest's
`workers_per_patch` comes from `exec`, never from the config), plus the whole-step guards
every patch must pass before any worker is spawned: `D/ts_actual`, `output.interval/
ts_actual`, `output.restart_interval/ts_actual` (the last two through
`build_nest(...; strict_intervals = true)` for a nest) and `D/output.interval`, where `D`
is the segment length. A single patch takes `workers_per_patch = [N]` (N workers on the
one patch).
"""
function segment_model(cfg::RunConfig, seg::NamedTuple, exec::ExecutionEnv)
    model = build_model(cfg; segment = seg)
    D = Float64(seg.integration_time)
    k = seg.index
    wpp = exec.workers_per_patch
    if model isa NestedModelParameters
        n = length(model.num_cells)
        length(wpp) == n || error("segment driver: the nest has $n patches but the " *
            "execution environment gives workers_per_patch = $(wpp); pass " *
            "--workers-per-patch with $n entries")
        model = NestedModelParameters(boundaries = model.boundaries, num_cells = model.num_cells,
                                      ts = model.ts, workers_per_patch = copy(wpp),
                                      base = model.base,
                                      initial_conditions = model.initial_conditions)
        _, topo = try
            build_nest(model; strict_intervals = true)
        catch e
            e isa ArgumentError || rethrow()
            error("segment driver, segment $k: $(e.msg)")
        end
        ts_actual = Float64.(topo.ts_actual)
    else
        length(wpp) == 1 || error("segment driver: a single-patch run takes one " *
            "workers_per_patch entry (--workers N), got $(wpp)")
        ts_actual = [model.ts]
        for (name, v) in (("output.interval", cfg.output.interval),
                          ("output.restart_interval", cfg.output.restart_interval))
            (v > 0.0 && !whole_steps(v, model.ts)) && error("segment driver: $name = $v s " *
                "is not a whole number of time steps (ts = $(model.ts) s)")
        end
    end
    for (i, ts) in enumerate(ts_actual)
        whole_steps(D, ts) || error("segment driver: segment $k length $D s is not a whole " *
            "number of patch $i's time steps (ts_actual = $ts s); the checkpoint name and " *
            "the next segment's offset would drift from the model clock")
    end
    whole_steps(D, cfg.output.interval) || error("segment driver: segment $k length $D s " *
        "is not a whole number of output intervals (output.interval = " *
        "$(cfg.output.interval) s)")
    return model, ts_actual
end

npatches(cfg::RunConfig) = length(cfg.grid.num_cells)

# ── Orchestrator ────────────────────────────────────────────────────────────────────────

"""
    run_simulation(cfg::RunConfig, exec::ExecutionEnv; resume = false,
                   config_sources = String[], set = String[]) -> nothing

Run `cfg` as a chain of segments under `root = cfg.output.dir`, one child Julia process per
segment ([`segment_main`](@ref)), in the layout `root/seg_NN/[nest{i}/]`.

Fresh start: refuses a root that already holds a run ([`check_fresh_root`](@ref)); writes
`root/config.toml` (the resolved config, with a provenance header: git SHA and dirty state,
date, host, `config_sources`, `set`, and `exec`, which is provenance only); builds segment
1, writes the reference state and initial conditions (`initialize_case!`); then runs every
segment.

`resume = true`: re-reads `root/config.toml` and uses THAT configuration (the `cfg`
argument supplies only the root). `set` may only hold `run.integration_time=...` (to extend
a run; config.toml is rewritten and the new segments carry a note in segments.csv).
Segments recorded `done` are skipped, `failed`/`running`/missing ones re-run. The reference
state and initial conditions must already exist. Execution flags may differ freely.

Each child's stdout+stderr is tee'd to `root/seg_NN/segment_stdout.log`; the first failure
stops the chain with an error naming the segment and its log.
"""
function run_simulation(cfg::RunConfig, exec::ExecutionEnv; resume::Bool = false,
                        config_sources::Vector{String} = String[],
                        set::Vector{String} = String[])
    root = abspath(cfg.output.dir)
    cfgpath = joinpath(root, "config.toml")
    notes = Dict{Int,String}()

    if resume
        for s in set
            key = strip(first(split(s, '='; limit = 2)))
            key == "run.integration_time" || error("--resume: --set $(repr(s)) refused: a " *
                "resumed run keeps its recorded configuration; only run.integration_time " *
                "may change (to extend the run)")
        end
        isfile(cfgpath) || error("--resume: $cfgpath does not exist (nothing to resume)")
        here = "output.dir=" * root
        old = load_config(cfgpath; set = [here])
        cfg = load_config(cfgpath; set = [here; set])
        _load_if_rrtmgp(cfg)
        oldplan, plan = segment_plan(old), segment_plan(cfg)
        rows = read_segments_csv(root)
        for (k, row) in rows
            row["status"] == "done" || continue
            (k <= length(plan) &&
             isapprox(parse(Float64, row["t_offset_s"]), plan[k].t_offset; atol = 1e-6) &&
             isapprox(parse(Float64, row["integration_time_s"]), plan[k].integration_time;
                      atol = 1e-6)) || error("--resume: segment $k is done but " *
                "run.integration_time = $(cfg.run.integration_time) would change it; a " *
                "resumed run may only add segments after the completed ones")
        end
        if cfg.run.integration_time != old.run.integration_time
            msg = "run.integration_time $(old.run.integration_time) -> " *
                  "$(cfg.run.integration_time) s on --resume $(Dates.now())"
            write_config(cfg, cfgpath; header = _read_header(cfgpath) * "\nresumed: " * msg)
            for s in plan
                (s.index > length(oldplan) || oldplan[s.index] != s) && (notes[s.index] = msg)
            end
            println("Scythe: $msg")
        end
        n = npatches(cfg)
        for p in [joinpath(root, "reference.ref"); initial_ics_paths(root, n)]
            isfile(p) || error("--resume: $p is missing; the reference state and initial " *
                               "conditions are written by a fresh start only")
        end
    else
        check_fresh_root(root)
        _load_if_rrtmgp(cfg)
    end
    return Base.invokelatest(_orchestrate, cfg, exec, root, resume, notes, config_sources, set)
end

_load_if_rrtmgp(cfg::RunConfig) =
    cfg.physics.radiation.scheme === :rrtmgp ? _load_rrtmgp!() : nothing

function _orchestrate(cfg::RunConfig, exec::ExecutionEnv, root::String, resume::Bool,
                      notes::Dict{Int,String}, sources::Vector{String}, set::Vector{String})
    cfgpath = joinpath(root, "config.toml")
    plan = segment_plan(cfg)
    nseg = length(plan)
    n = npatches(cfg)

    # Every segment's guards before anything is written or launched (IC paths are
    # placeholders: build_model only records them).
    for s in plan
        segment_model(cfg, merge(s, (initial_conditions = initial_ics_paths(root, n),)), exec)
    end

    if !resume
        mkpath(root)
        write_config(cfg, cfgpath; header = _provenance_header(exec, sources, set))
        seg1 = merge(plan[1], (initial_conditions = initial_ics_paths(root, n),))
        println("Scythe: writing the reference state and initial conditions into $root")
        initialize_case!(cfg, build_model(cfg; segment = seg1))
    end

    println("Scythe: $(cfg.run.name): $(cfg.run.integration_time) s in $nseg segment(s); " *
            "root = $root")
    proj = dirname(Base.active_project())
    code = "using Scythe; Scythe.segment_main(ARGS)"
    for s in plan
        k = s.index
        if resume && segment_status(root, k) == "done"
            println("Scythe: segment $k/$nseg already done -- skipping (--resume)")
            continue
        end
        segdir = joinpath(root, segment_dirname(k))
        mkpath(segdir)
        update_segment_row!(root, k, Dict(
            "dir" => segdir, "t_offset_s" => s.t_offset,
            "integration_time_s" => s.integration_time,
            "output_interval_s" => cfg.output.interval, "status" => "running",
            "scythe_sha" => _sha_tag(), "started" => Dates.now(), "finished" => "",
            "n_patches" => n, "workers" => join(exec.workers_per_patch, ";"),
            "threads" => exec.worker_threads, "host" => gethostname(),
            "note" => get(notes, k, "")))
        child_args = ["--config", cfgpath, "--segment", string(k), exec_flags(exec)...]
        # Interpolate the Vector WITHOUT a splat: `$(v...)` in a command literal concatenates
        # every element into ONE argument; a bare `$v` expands each element as its own.
        cmd = `$(Base.julia_cmd()) --project=$(proj) -e $code -- $child_args`
        log = joinpath(segdir, "segment_stdout.log")
        println("Scythe: launching segment $k/$nseg\n  $cmd")
        open(log, "a") do io
            println(io, "=== Scythe segment $k/$nseg launched $(Dates.now()) ===")
        end
        tee = open(`tee -a $log`, "w", stdout)
        ok = try
            success(pipeline(cmd; stdout = tee, stderr = tee))
        finally
            close(tee)
            wait(tee)
        end
        if !ok
            segment_status(root, k) == "failed" ||
                update_segment_row!(root, k, Dict("status" => "failed",
                                                  "finished" => Dates.now()))
            error("Scythe: segment $k/$nseg FAILED (see $log). Fix the cause and re-run " *
                  "with --resume to continue from segment $k.")
        end
        println("Scythe: segment $k/$nseg complete")
    end
    println("Scythe: all $nseg segment(s) complete. Root: $root")
    return nothing
end

# ── Child: one segment ──────────────────────────────────────────────────────────────────

"""
    segment_main(args::Vector{String})

Entry point of a segment's child process (launched by [`run_simulation`](@ref)):
`--config <root>/config.toml --segment k` plus the execution flags
(`--workers-per-patch`, `--worker-threads`, `--blas-threads`, `--slurm`). Loads the
config (its `output.dir` taken as the config file's directory, so a moved tree still
resumes in place), imports RRTMGP/ClimaComms first when radiation is on, and chains the
initial conditions: segment 1 reads `root/ics.csv` (nests: `root/nest{i}_ics.csv`),
segment k > 1 each patch's [`find_segment_checkpoints`](@ref) in `seg_{k-1}`. Builds the
model with the whole-step guards ([`segment_model`](@ref)) before any worker is spawned,
writes `seg_NN/config.toml` with the segment's values in the header, adds the workers, and
integrates; the segment's row in segments.csv ends `done` or `failed` (the error is
rethrown).
"""
function segment_main(args::Vector{String})
    a = parse_launcher_args(args)
    a.segment >= 1 || error("segment_main: --segment k (k >= 1) is required")
    length(a.config_paths) == 1 || error("segment_main: exactly one config (the run's " *
        "config.toml) is expected, got $(a.config_paths)")
    isempty(a.set) || error("segment_main: --set is not accepted by a segment child")
    config = abspath(a.config_paths[1])
    cfg = load_config(config; set = ["output.dir=" * dirname(config)])
    _load_if_rrtmgp(cfg)
    return Base.invokelatest(_run_segment, cfg, a.segment, a.exec)
end

function _run_segment(cfg::RunConfig, k::Int, exec::ExecutionEnv)
    root = abspath(cfg.output.dir)
    plan = segment_plan(cfg)
    nseg = length(plan)
    1 <= k <= nseg || error("segment_main: --segment $k is out of range ($nseg segment(s))")
    n = npatches(cfg)
    s = plan[k]
    ics = k == 1 ? initial_ics_paths(root, n) :
        find_segment_checkpoints(segment_patch_dirs(root, k - 1, n),
                                 plan[k - 1].integration_time)
    for p in ics
        isfile(p) || error("segment_main: initial condition $p does not exist")
    end
    isfile(joinpath(root, "reference.ref")) || error("segment_main: " *
        "$(joinpath(root, "reference.ref")) is missing; segment 1 of a fresh start writes it")
    seg = merge(s, (initial_conditions = ics,))
    start_hour = cfg.physics.radiation.start_hour + s.t_offset / 3600.0
    segdir = joinpath(root, segment_dirname(k))
    row = Dict{String,Any}("status" => "running", "dir" => segdir,
                           "t_offset_s" => s.t_offset, "integration_time_s" => s.integration_time,
                           "output_interval_s" => cfg.output.interval,
                           "n_patches" => n, "start_hour" => start_hour,
                           "workers" => join(exec.workers_per_patch, ";"),
                           "host" => gethostname(), "scythe_sha" => _sha_tag())
    try
        model, ts_actual = segment_model(cfg, seg, exec)
        dirs = segment_patch_dirs(root, k, n)
        restart = join([checkpoint_path(d, s.integration_time) for d in dirs], ";")
        row["ts"] = model isa NestedModelParameters ? model.base.ts : model.ts
        row["ts_actual"] = join(ts_actual, ";")
        row["restart_file"] = restart
        update_segment_row!(root, k, row)

        header = join(["Scythe segment $k of $nseg: the run's resolved configuration; the " *
                       "driver applies the segment values below.",
                       "written = $(Dates.now()) on $(gethostname())",
                       "scythe = $(_git_string())",
                       "segment = $k",
                       "t_offset_s = $(s.t_offset)",
                       "integration_time_s = $(s.integration_time)",
                       "start_hour = $(start_hour)",
                       "initial_conditions = $(join(ics, ";"))",
                       "ts_actual = $(join(ts_actual, ";"))",
                       "restart_file = $restart",
                       "execution: $(_exec_string(exec))"], "\n")
        mkpath(segdir)
        write_config(cfg, joinpath(segdir, "config.toml"); header = header)

        println("Scythe: segment $k/$nseg, t_offset = $(s.t_offset) s, length = " *
                "$(s.integration_time) s, start_hour = $start_hour, ICs = $(join(ics, ", "))")
        setup_workers!(exec)
        row["threads"] = worker_provenance()["worker_threads"]
        update_segment_row!(root, k, row)
        wall = @elapsed (model isa NestedModelParameters ? integrate_nested_model(model) :
                                                           integrate_model(model))
        println("Scythe: segment $k/$nseg done in $(round(wall; digits = 1)) s wall clock")
        row["status"] = "done"
        row["finished"] = Dates.now()
        update_segment_row!(root, k, row)
    catch e
        row["status"] = "failed"
        row["finished"] = Dates.now()
        update_segment_row!(root, k, row)
        rethrow()
    end
    return nothing
end

# ── Launcher command line ───────────────────────────────────────────────────────────────

"""
    parse_launcher_args(args) -> (config_paths, set, resume, exec::ExecutionEnv, segment)

The shared command line of the `simulations/<case>/run.jl` launchers (and of the segment
child):

- `--config PATH` — the case file (default `config.toml` beside `PROGRAM_FILE`), then any
  positional overlay files, later ones winning: `config_paths = [config; overlays...]`;
- `--set key.path=value` (repeatable), `--resume`;
- `--workers N` (N workers on a single patch) or `--workers-per-patch a,b,c` (a nest), not
  both; `--worker-threads N` (0 = split the machine), `--blas-threads N` (0 = OpenBLAS
  default), `--slurm`;
- `--segment k` — internal (the orchestrator's child); 0 = orchestrator.

Throws `ArgParse.ArgParseError` (not `exit`) on a malformed command line.
"""
function parse_launcher_args(args::AbstractVector{<:AbstractString})
    s = ArgParse.ArgParseSettings(prog = "run.jl", exc_handler = ArgParse.debug_handler,
                                  description = "Run a Scythe simulation from a config file.")
    ArgParse.@add_arg_table! s begin
        "--config"
            help = "case config file (default: config.toml beside the launcher script)"
            arg_type = String
            default = ""
        "overlays"
            help = "overlay config files applied after --config, in order"
            nargs = '*'
            arg_type = String
        "--set"
            help = "key.path=value override (repeatable)"
            action = :append_arg
            arg_type = String
        "--resume"
            help = "continue the run in output.dir from its segments.csv"
            action = :store_true
        "--workers"
            help = "worker processes for a single-patch run"
            arg_type = Int
            default = 0
        "--workers-per-patch"
            help = "comma-separated worker counts, one per nest patch"
            arg_type = String
            default = ""
        "--worker-threads"
            help = "Julia threads per worker (0 = split the machine)"
            arg_type = Int
            default = 0
        "--blas-threads"
            help = "OpenBLAS threads per process (0 = OpenBLAS default)"
            arg_type = Int
            default = 1
        "--slurm"
            help = "add workers through SLURM (addprocs_slurm)"
            action = :store_true
        "--segment"
            help = "internal: run only this segment (child process)"
            arg_type = Int
            default = 0
    end
    p = ArgParse.parse_args(collect(String, args), s)
    config = p["config"]
    isempty(config) && (config = joinpath(dirname(abspath(PROGRAM_FILE)), "config.toml"))
    nw, wpp = p["workers"], p["workers-per-patch"]
    (nw > 0 && !isempty(wpp)) && throw(ArgumentError(
        "give --workers N (single patch) or --workers-per-patch a,b,c (nest), not both"))
    nw < 0 && throw(ArgumentError("--workers must be >= 1"))
    workers_per_patch = !isempty(wpp) ? parse.(Int, strip.(split(wpp, ','))) :
                        nw > 0 ? [nw] : [1]
    all(>=(1), workers_per_patch) || throw(ArgumentError(
        "--workers-per-patch entries must be >= 1, got $wpp"))
    exec = ExecutionEnv(workers_per_patch = workers_per_patch,
                        worker_threads = p["worker-threads"],
                        blas_threads = p["blas-threads"], slurm = p["slurm"])
    p["segment"] >= 0 || throw(ArgumentError("--segment must be >= 1"))
    return (config_paths = String[config; String.(p["overlays"])],
            set = String.(p["set"]), resume = p["resume"], exec = exec,
            segment = p["segment"])
end
