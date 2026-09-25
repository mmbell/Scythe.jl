#!/usr/bin/env julia
# tools/capture_legacy_models.jl -- Stage 0 "golden capture" of the config-consolidation
# plan (see /Users/mmbell/.claude/plans/the-last-several-sessions-shimmering-lighthouse.md,
# §6 row "0", §7 "Equivalence"/"IC bytes"). Three legacy driver scripts each translate
# environment variables into a `ModelParameters`/`NestedModelParameters`:
#
#   tc/tc_params.jl + tc/tc_init.jl   (SCYTHE_TC_*,  make_base/make_nest/init_tc!)
#   benchmarks/ocean_rce.jl           (SCYTHE_RCE_*, rce_model/rce_init!)
#   benchmarks/ocean_warm_bubble.jl   (SCYTHE_OWB_*, owb_model/owb_init!)
#
# They will be retired in favor of a typed config + `build_model(cfg)`. This script records
# an exact fingerprint of what each one builds, for a fixed set of "arms" (env-variable
# combinations), plus SHA-256 hashes of the initial-condition files their own `init!`
# writes -- so the replacement can be proven equivalent key by key, and byte-identical on
# the initial conditions, before the legacy drivers are deleted.
#
#   julia --project=. tools/capture_legacy_models.jl [arm1,arm2,...]
#
# With no argument, every arm below is captured. Output:
#   test/reference_data/legacy_models/<arm>.toml           model/nest fingerprint
#   test/reference_data/legacy_models/<arm>.hashes.toml     basename => SHA-256 (IC arms only)
#   test/reference_data/legacy_models/README.md             what these are, env, git SHA
#
# EACH ARM RUNS IN ITS OWN FRESH `julia --project=.` SUBPROCESS. This is not incidental:
# tc/tc_params.jl's TC_BL_CHOICE/TC_VARS/OUTPUT_DIR and ocean_rce.jl's DUNION_SOUNDING-
# adjacent globals are read from ENV (or built from ARGS-like vectors) once at file-include
# time, so two arms sharing one process would either collide on `const` redefinition or
# silently inherit the first arm's configuration. No integration timestep is ever run --
# only model construction and (for some arms) `init!`, which calls `createGrid` on the
# master and needs no worker processes.
#
# SCRATCH OUTPUT: TC arms redirect via SCYTHE_TC_OUTDIR, RCE arms via `rce_model`'s `root`
# keyword, and OWB arms via SCYTHE_BENCH_TAG (a uniquely tagged, never-before-existing
# directory under the ordinarily-gitignored benchmarks/output/) -- never into an existing
# output tree, per the project convention that model output is experimental data.

using Dates

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const OUT_DIR = joinpath(REPO_ROOT, "test", "reference_data", "legacy_models")
mkpath(OUT_DIR)

# One scratch root for the whole capture run; each arm gets its own subdirectory. Left in
# place after the run (harmless temp-dir litter) so a failed arm's output can be inspected;
# it is NOT under the repo, so nothing here can ever show up in `git status`.
const SCRATCH_ROOT = mktempdir(; prefix = "scythe_capture_")
println("Scratch root (not part of the repo): $SCRATCH_ROOT")

struct Arm
    name::String
    kind::Symbol            # :tc | :rce | :owb
    env::Dict{String,String}   # exactly the legacy variable names for this arm
    body::String             # raw Julia source specific to this arm (see below)
end

# ── Shared header/preamble, common to every arm's generated script ─────────────────────
# `raw` throughout: none of this is meant to be interpolated by THIS (orchestrator) file --
# it becomes literal Julia source for the CHILD process, which does its own interpolation
# when it runs.
const PREAMBLE = raw"""
using Scythe, Springsteel
using TOML, SHA

include(joinpath(REPO_ROOT, "test", "config_fingerprint.jl"))

function sha256_file(path::AbstractString)
    open(path, "r") do io
        return bytes2hex(SHA.sha256(io))
    end
end

function write_toml(data, path::AbstractString)
    open(path, "w") do io
        TOML.print(io, data; sorted = true)
    end
    return path
end

# Basename => hex SHA-256 for every path in `paths` (each must already exist).
function hash_files(paths::Vector{String})
    h = Dict{String,Any}()
    for p in paths
        isfile(p) || error("expected file not found: $p")
        h[basename(p)] = sha256_file(p)
    end
    return h
end

mkpath(SCRATCH)
"""

# ── Per-arm bodies ───────────────────────────────────────────────────────────────────────
# Each references the header consts ARM/REPO_ROOT/SCRATCH/OUT_TOML/OUT_HASHES (defined
# before PREAMBLE in the generated file) and the PREAMBLE helpers above. `raw` for the same
# reason as PREAMBLE.

const BODY_TC_DEFAULT = raw"""
include(joinpath(REPO_ROOT, "tc", "tc_params.jl"))
include(joinpath(REPO_ROOT, "tc", "tc_init.jl"))

base = make_base(1800.0)
nest = make_nest(base)
write_toml(nest_fingerprint(nest), OUT_TOML)

models, topo = init_tc!(nest)
ic_paths = String[m.initial_conditions for m in models]
push!(ic_paths, base.ref_state_file)
write_toml(hash_files(ic_paths), OUT_HASHES)
"""

# tc_mynn_rad_stab is not marked "fingerprint only" in the plan, so it gets ICs too (worth
# hashing separately: MYNN appends the "rho_e" TKE slot, so the written IC CSV has a
# different column layout than tc_default's).
const BODY_TC_MYNN_RAD_STAB = BODY_TC_DEFAULT

const BODY_TC_RLR = raw"""
include(joinpath(REPO_ROOT, "tc", "tc_params.jl"))
include(joinpath(REPO_ROOT, "tc", "tc_init.jl"))

base = make_base(1800.0; geometry = "RLR")
nest = make_nest(base)
write_toml(nest_fingerprint(nest), OUT_TOML)
"""

const BODY_TC_R1T1X = raw"""
include(joinpath(REPO_ROOT, "tc", "tc_params.jl"))
include(joinpath(REPO_ROOT, "tc", "tc_init.jl"))

base = make_base(1800.0)
nest = make_nest(base)
write_toml(nest_fingerprint(nest), OUT_TOML)
"""

const BODY_RCE_DEFAULT_SEG1 = raw"""
include(joinpath(REPO_ROOT, "benchmarks", "ocean_rce.jl"))

rce = parse_rce_args(["--mode", "quick", "--stage", "mc", "--grid", "rirk",
                      "--days", "20", "--segment-days", "2"])
opts = parse_benchmark_args(rce.rest)
model = rce_model(rce, opts, 1; root = SCRATCH)
write_toml(model_fingerprint(model), OUT_TOML)

rce_init!(model)
write_toml(hash_files([model.initial_conditions, model.ref_state_file]), OUT_HASHES)
"""

const BODY_RCE_DEFAULT_SEG3 = raw"""
include(joinpath(REPO_ROOT, "benchmarks", "ocean_rce.jl"))

rce = parse_rce_args(["--mode", "quick", "--stage", "mc", "--grid", "rirk",
                      "--days", "20", "--segment-days", "2"])
opts = parse_benchmark_args(rce.rest)

# rce_model's restart-chain lookup for segment k > 1 expects an existing checkpoint from
# segment k-1 under `root`. The golden capture wants only segment 3's CONFIGURATION (grid,
# timestep, the radiation start_hour offset carried from t_offset_days) -- never an actual
# multi-segment run -- so a placeholder checkpoint is created at exactly the path/name the
# restart-chain logic looks for. Its bytes are never read: ModelParameters.initial_conditions
# is fingerprinted only by basename.
Dprev = seg_length(2, rce.days, rce.segment_days)
prev_seg_dir = joinpath(SCRATCH, "seg_02")
mkpath(prev_seg_dir)
touch(joinpath(prev_seg_dir, "$(string(round(Dprev * 86400.0; digits = 2))).jld2"))

model = rce_model(rce, opts, 3; root = SCRATCH)
write_toml(model_fingerprint(model), OUT_TOML)
"""

const BODY_RCE_ICE0_LOUIS = raw"""
include(joinpath(REPO_ROOT, "benchmarks", "ocean_rce.jl"))

rce = parse_rce_args(["--mode", "quick", "--stage", "mc", "--grid", "rirk",
                      "--days", "20", "--segment-days", "2"])
opts = parse_benchmark_args(rce.rest)
model = rce_model(rce, opts, 1; root = SCRATCH)
write_toml(model_fingerprint(model), OUT_TOML)
"""

const BODY_OWB_DEFAULT = raw"""
include(joinpath(REPO_ROOT, "benchmarks", "ocean_warm_bubble.jl"))

opts = parse_benchmark_args(["--mode", "quick", "--stage", "mc", "--grid", "rirk"])
model = owb_model(opts)
write_toml(model_fingerprint(model), OUT_TOML)

owb_init!(model)
write_toml(hash_files([model.initial_conditions, model.ref_state_file]), OUT_HASHES)
"""

const BODY_OWB_MYNN_ICE_RAD = raw"""
include(joinpath(REPO_ROOT, "benchmarks", "ocean_warm_bubble.jl"))

opts = parse_benchmark_args(["--mode", "quick", "--stage", "mc", "--grid", "rirk"])
model = owb_model(opts)
write_toml(model_fingerprint(model), OUT_TOML)
"""

# ── Arm table ────────────────────────────────────────────────────────────────────────────

const ARMS = Arm[
    Arm("tc_default", :tc, Dict{String,String}(), BODY_TC_DEFAULT),
    Arm("tc_mynn_rad_stab", :tc,
        Dict("SCYTHE_TC_BL" => "mynn", "SCYTHE_TC_RAD" => "diurnal",
             "SCYTHE_TC_SFC_STAB" => "1", "SCYTHE_TC_SIDECARS" => "1"),
        BODY_TC_MYNN_RAD_STAB),
    Arm("tc_rlr", :tc, Dict{String,String}(), BODY_TC_RLR),
    Arm("tc_r1t1x", :tc, Dict("SCYTHE_TC_WALL_BC" => "r1t1x"), BODY_TC_R1T1X),
    Arm("rce_default_seg1", :rce, Dict{String,String}(), BODY_RCE_DEFAULT_SEG1),
    Arm("rce_default_seg3", :rce, Dict{String,String}(), BODY_RCE_DEFAULT_SEG3),
    Arm("rce_ice0_louis", :rce,
        Dict("SCYTHE_RCE_ICE" => "0", "SCYTHE_RCE_BL" => "louis", "SCYTHE_RCE_RAD" => "0"),
        BODY_RCE_ICE0_LOUIS),
    Arm("owb_default", :owb, Dict{String,String}(), BODY_OWB_DEFAULT),
    Arm("owb_mynn_ice_rad", :owb,
        Dict("SCYTHE_OWB_BL" => "mynn", "SCYTHE_OWB_ICE" => "1",
             "SCYTHE_OWB_RAD" => "diurnal", "SCYTHE_OWB_SFC_STAB" => "1"),
        BODY_OWB_MYNN_ICE_RAD),
]

# ── Driver ───────────────────────────────────────────────────────────────────────────────

"""Build the full generated script text and the process-environment overlay for `arm`."""
function generate(arm::Arm)
    scratch = joinpath(SCRATCH_ROOT, arm.name)
    out_toml = joinpath(OUT_DIR, "$(arm.name).toml")
    out_hashes = joinpath(OUT_DIR, "$(arm.name).hashes.toml")

    header = """
    const ARM = "$(arm.name)"
    const REPO_ROOT = "$(REPO_ROOT)"
    const SCRATCH = "$(scratch)"
    const OUT_TOML = "$(out_toml)"
    const OUT_HASHES = "$(out_hashes)"
    """

    script = header * PREAMBLE * arm.body * "\nprintln(\"CAPTURE_OK: \$ARM\")\n"

    env_overlay = copy(arm.env)
    if arm.kind == :tc
        env_overlay["SCYTHE_TC_OUTDIR"] = scratch
    elseif arm.kind == :owb
        env_overlay["SCYTHE_BENCH_TAG"] = "_stage0capture_$(arm.name)"
    end
    # RCE redirects entirely through `rce_model`'s `root` keyword (SCRATCH), which needs no
    # env var -- RCE_ROOT is never referenced by the generated script.

    return script, env_overlay, out_toml, out_hashes
end

function run_arm(arm::Arm)
    script, env_overlay, out_toml, out_hashes = generate(arm)
    script_path = joinpath(SCRATCH_ROOT, "$(arm.name).jl")
    write(script_path, script)

    child_env = merge(Dict{String,String}(string(k) => string(v) for (k, v) in ENV),
                      env_overlay)
    cmd = setenv(`$(Base.julia_cmd()) --project=$(REPO_ROOT) $(script_path)`, child_env)

    println("── $(arm.name) ", "─"^max(1, 60 - length(arm.name)))
    isempty(arm.env) || println("  env: ", join(["$k=$v" for (k, v) in sort(collect(arm.env))], " "))
    t0 = time()
    out = IOBuffer()
    ok = try
        run(pipeline(cmd; stdout = out, stderr = out))
        true
    catch e
        false
    end
    elapsed = round(time() - t0; digits = 1)
    log = String(take!(out))
    print(log)
    ok = ok && occursin("CAPTURE_OK: $(arm.name)", log)
    println(ok ? "  OK ($(elapsed) s)" : "  FAILED ($(elapsed) s)")
    return (arm = arm.name, ok = ok, toml = out_toml, hashes = out_hashes,
            has_hashes = isfile(out_hashes), elapsed = elapsed)
end

function write_readme(arms, results)
    lines = String[]
    push!(lines, "# Legacy-driver golden capture (Stage 0)")
    push!(lines, "")
    push!(lines, "Generated by `tools/capture_legacy_models.jl` at git SHA `$(strip(read(`git -C $REPO_ROOT rev-parse HEAD`, String)))`" *
                 (isempty(strip(read(`git -C $REPO_ROOT status --porcelain`, String))) ? "" : " (dirty working tree)") *
                 ", $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS")).")
    push!(lines, "")
    push!(lines, "Each `<arm>.toml` is a fingerprint (`test/config_fingerprint.jl`) of the " *
                 "`ModelParameters`/`NestedModelParameters` the corresponding legacy driver " *
                 "(`tc/tc_params.jl`+`tc/tc_init.jl`, `benchmarks/ocean_rce.jl`, " *
                 "`benchmarks/ocean_warm_bubble.jl`) builds for that arm's environment. Each " *
                 "`<arm>.hashes.toml` (where present) is basename => SHA-256 of every initial-" *
                 "condition CSV and the reference-state `.ref` file that arm's own `init!` " *
                 "writes on a scratch grid. No integration timestep was run to produce any of " *
                 "this. These are the values a future `build_model(cfg)` replacement must " *
                 "reproduce key-by-key (plan §7 \"Equivalence\"/\"IC bytes\").")
    push!(lines, "")
    push!(lines, "## Arms")
    push!(lines, "")
    for arm in arms
        push!(lines, "### `$(arm.name)`")
        push!(lines, "")
        push!(lines, isempty(arm.env) ? "Env: (none -- driver defaults)" :
                     "Env: `" * join(["$k=$v" for (k, v) in sort(collect(arm.env))], " ") * "`")
        push!(lines, "")
        scratch_note = arm.kind == :tc ?
            "Scratch redirection: `SCYTHE_TC_OUTDIR` -> a per-arm `mktempdir()` (never `tc/output/`)." :
            arm.kind == :owb ?
            "Scratch redirection: `SCYTHE_BENCH_TAG=_stage0capture_$(arm.name)` -> a freshly-tagged, " *
            "never-before-existing directory under `benchmarks/output/` (gitignored)." :
            "Scratch redirection: `rce_model`'s `root` keyword -> a per-arm `mktempdir()` " *
            "(never `benchmarks/output/`)."
        push!(lines, scratch_note)
        push!(lines, "")
    end
    push!(lines, "## Files")
    push!(lines, "")
    for r in results
        push!(lines, "- `$(r.arm).toml`" * (r.has_hashes ? ", `$(r.arm).hashes.toml`" : "") *
                     " -- capture $(r.ok ? "OK" : "FAILED") ($(r.elapsed) s)")
    end
    write(joinpath(OUT_DIR, "README.md"), join(lines, "\n") * "\n")
end

function main()
    wanted = isempty(ARGS) ? nothing : Set(split(ARGS[1], ','))
    arms = wanted === nothing ? ARMS : filter(a -> a.name in wanted, ARMS)
    isempty(arms) && error("no arms matched $(ARGS[1]); known arms: " *
                           join([a.name for a in ARMS], ", "))
    results = [run_arm(a) for a in arms]
    write_readme(arms, results)
    println()
    nfail = count(r -> !r.ok, results)
    println("Capture complete: $(length(results) - nfail)/$(length(results)) arms OK.")
    nfail > 0 && exit(1)
end

main()
