#!/usr/bin/env julia
# Local launcher for the ocean_2D_rce case: parses the shared launcher command line
# (src/run_driver.jl `parse_launcher_args`), loads the config, then either runs one segment
# (the orchestrator's own child process, `--segment k`) or the whole segment chain.
#
#   julia --project=simulations simulations/ocean_2D_rce/run.jl [overlay.toml ...] \
#       [--set key.path=value ...] [--resume] [--workers N | --workers-per-patch a,b,c] \
#       [--worker-threads N] [--blas-threads N] [--slurm]
#
# radiation.scheme = "rrtmgp" (this case's default) needs RRTMGP/ClimaComms, WEAK
# dependencies of Scythe (ext/ScytheRRTMGPExt.jl) loadable only from an environment that
# lists them -- simulations/Project.toml, a workspace member sharing the root
# Manifest.toml. Launch with `--project=simulations`, not the root project.
#
# Examples:
#   julia --project=simulations simulations/ocean_2D_rce/run.jl --workers 1
#   julia --project=simulations simulations/ocean_2D_rce/run.jl --resume
#   julia --project=simulations simulations/ocean_2D_rce/run.jl full.toml --workers 1 \
#       --worker-threads 18

using Scythe

paths, set, resume, exec, segment = Scythe.parse_launcher_args(ARGS)
cfg = load_config(paths...; set = set)

if cfg.physics.radiation.scheme == :rrtmgp
    Base.identify_package("RRTMGP") !== nothing || error(
        "physics.radiation.scheme = \"rrtmgp\" but RRTMGP cannot be loaded from the " *
        "active project $(Base.active_project()). Launch with " *
        "`julia --project=simulations`.")
    @eval Main import RRTMGP, ClimaComms
end

if segment > 0
    Scythe.segment_main(ARGS)
else
    run_simulation(cfg, exec; resume = resume, config_sources = paths, set = set)
end
