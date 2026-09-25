# Run driver: execution-environment helpers (worker setup, provenance, git state) and the
# pure segment arithmetic the segmented run orchestrator uses. The orchestrator itself
# (`run_simulation`, `segment_main`) builds on these.
#
# The worker helpers moved here from benchmarks/common/harness.jl, which keeps thin
# forwarders under the old names so every benchmark and test/test_benchmark_smoke.jl works
# unchanged.

using ClusterManagers: addprocs_slurm

export ExecutionEnv, setup_workers!, checkpoint_path

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
