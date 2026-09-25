#!/usr/bin/env julia
# Preflight for the simulations/<case>/*.sbatch launchers: fail in seconds on the login or
# compute node, not twenty minutes into a queued job.
#
#   julia --project=simulations simulations/_preflight.jl <config.toml> [overlays...] \
#         [--set key.path=value ...] [any other run.jl flag, ignored]
#
# (`--config <config.toml>` works too; the arguments are the launcher's own, so an sbatch
# script passes exactly what it will pass to run.jl.) Checks, in order:
#
#   1. `Pkg.instantiate()` + `Pkg.precompile()` of the simulations environment.
#   2. Springsteel is new enough. Springsteel is a REGISTERED package, so a fresh checkout
#      can resolve an old release that lacks `_hydrostatic_pressure_profile`,
#      `CubicBSpline.R1T1X` and the `hydrostatic` keyword of
#      `calculate_pressure_reference_state` -- and the run then dies on an unknown-keyword
#      MethodError deep into the job, with nothing saying "wrong Springsteel". The
#      `[sources]` entry points at the sibling checkout; this catches that checkout being on
#      the wrong branch.
#   3. The config (plus overlays and --set) loads and validates.
#   4. When `physics.radiation.scheme == "rrtmgp"`: RRTMGP loads, the ScytheRRTMGPExt
#      extension activates, and the RRTMGP lookup-table artifacts are ALREADY local. This is
#      check-only -- it never downloads. Compute nodes have no network, and RRTMGP fetches
#      the tables lazily on the first radiation call; run tools/rrtmgp_prewarm.jl on a login
#      node first.
#
# Prints `run root: <output.dir>` on success (the sbatch scripts read it to place
# job_stdout.log). Exits 3 with a message naming the failure otherwise.

function preflight_fail(msg)
    println(stderr, "\nPREFLIGHT FAILED - not starting the run.\n", msg)
    exit(3)
end

import Pkg
# No automatic precompile inside instantiate: it would also try the environment itself
# (ScytheSimulations has a name but no source) and print a spurious failure.
ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
try
    Pkg.instantiate()
    # Named: the environment itself (ScytheSimulations) has a name but no source to precompile.
    Pkg.precompile(["Scythe", "Springsteel", "RRTMGP", "ClimaComms"])
catch e
    preflight_fail("Pkg.instantiate/precompile of $(Base.active_project()) failed " *
                   "(the launch environment is julia --project=<Scythe>/simulations):\n" *
                   sprint(showerror, e))
end

using Springsteel
println("  Springsteel      : ", pkgdir(Springsteel))
miss = String[]
isdefined(Springsteel, :_hydrostatic_pressure_profile) || push!(miss, "_hydrostatic_pressure_profile")
isdefined(Springsteel.CubicBSpline, :R1T1X)            || push!(miss, "CubicBSpline.R1T1X")
any(m -> :hydrostatic in Base.kwarg_decl(m),
    methods(Springsteel.calculate_pressure_reference_state)) ||
    push!(miss, "calculate_pressure_reference_state(; hydrostatic)")
isempty(miss) || preflight_fail("""
    This Springsteel is too old for Scythe's moist-compressible configurations.
    Missing: $(join(miss, ", ")).
    Check the branch of the Springsteel.jl checked out beside Scythe.jl (the [sources]
    entry of simulations/Project.toml); a registered release resolved from the General
    registry is not evidence of anything.""")
println("  Springsteel      : required symbols present")

using Scythe

# The launcher's own argument vector. A bare leading path is the case config (the default
# `--config`, config.toml beside PROGRAM_FILE, is simulations/config.toml, which is not a case).
a = try
    Scythe.parse_launcher_args(ARGS)
catch e
    preflight_fail("could not parse the launcher arguments $(ARGS):\n" * sprint(showerror, e))
end
paths = a.config_paths
if !isfile(paths[1]) && length(paths) > 1
    paths = paths[2:end]
end
isempty(paths) && preflight_fail("no config file given")
cfg = try
    load_config(paths...; set = a.set)
catch e
    preflight_fail("the configuration $(join(paths, " + ")) does not load:\n" *
                   sprint(showerror, e))
end
println("  config           : ", join(paths, " + "),
        isempty(a.set) ? "" : "  (--set " * join(a.set, " ") * ")")

# Three top-level statements, not one block: under Julia 1.12's world-age-partitioned
# bindings, a name imported inside a block is not visible to the rest of that same block.
const RADIATION = cfg.physics.radiation.scheme == :rrtmgp
RADIATION && Base.identify_package("RRTMGP") === nothing && preflight_fail(
    "physics.radiation.scheme = \"rrtmgp\" but RRTMGP cannot be loaded from " *
    "$(Base.active_project()); launch with julia --project=<Scythe>/simulations")
RADIATION && @eval import RRTMGP, ClimaComms
if RADIATION
    Base.get_extension(Scythe, :ScytheRRTMGPExt) === nothing &&
        preflight_fail("RRTMGP loaded but the ScytheRRTMGPExt extension did not activate")
    println("  RRTMGP           : ", pkgversion(RRTMGP), ", ScytheRRTMGPExt loaded")
end
# validate(cfg) needs the extension loaded when radiation is on, hence after the import.
try
    Scythe.validate(cfg)
catch e
    preflight_fail("the configuration does not validate:\n" * sprint(showerror, e))
end
if RADIATION
    missing_artifacts = String[]
    for optics_type in (:gas, :cloud), lam in (:lw, :sw)
        path = RRTMGP.ArtifactPaths.get_lookup_filename(optics_type, lam)
        isfile(path) || push!(missing_artifacts, "$(optics_type)/$(lam) -> $path")
    end
    isempty(missing_artifacts) || preflight_fail("""
        RRTMGP lookup-table artifacts are missing (radiation is on in this config):
          $(join(missing_artifacts, "\n  "))
        Run tools/rrtmgp_prewarm.jl on a LOGIN NODE (with network access) first:
          julia --project=<Scythe.jl checkout>/simulations tools/rrtmgp_prewarm.jl
        Compute nodes have no network, so a lazy download at the first radiation call
        would hang or fail deep into a queued job.""")
    println("  RRTMGP artifacts : present (gas/lw, gas/sw, cloud/lw, cloud/sw)")
end

println("  preflight        : OK")
println("run root: ", cfg.output.dir)
