#!/usr/bin/env julia
# Local launcher for the axisymmetric TC (simulations/tc_axisym/config.toml).
#
#   julia --project=simulations simulations/tc_axisym/run.jl [overlays...] \
#       [--set key.path=value ...] [--resume] \
#       --workers-per-patch 1,1,1 [--worker-threads N] [--blas-threads N] [--slurm]
#
# --project=simulations, not the root project: radiation is on (RRTMGP), and RRTMGP and
# ClimaComms are WEAK dependencies of Scythe that load only from an environment listing them.
#
# Workers: this case is a 3-patch nest (the rlr.toml overlay too), so pass
# `--workers-per-patch 1,1,1` -- one worker per nest (the default, a single entry, is refused
# for a nest). `--worker-threads N` threads per worker (0 = split the machine).
#
# Examples (from the repository root; a `--set` path resolves against the working directory):
#   R=simulations/tc_axisym
#   julia --project=simulations $R/run.jl --workers-per-patch 1,1,1 --worker-threads 4
#   julia --project=simulations $R/run.jl $R/fine.toml --workers-per-patch 1,1,1 \
#         --set run.integration_time=86400 --set output.dir=$R/output/tc_fine_24h
#   julia --project=simulations $R/run.jl --workers-per-patch 1,1,1 --resume \
#         --set output.dir=$R/output/tc_fine_24h                      # continue a run
#   julia --project=simulations $R/run.jl --workers-per-patch 1,1,1 --resume \
#         --set output.dir=$R/output/tc_fine_24h --set run.integration_time=172800   # extend it
#
# Output goes to <output.dir>/seg_NN/nest{i}/ (see README.md). A fresh start refuses a tree
# that already holds output: mv it aside (never rm model output), or continue with --resume.

using Scythe

paths, set, resume, exec, segment = Scythe.parse_launcher_args(ARGS)
if segment > 0                     # internal: a segment child launched by run_simulation
    Scythe.segment_main(ARGS)
    exit(0)
end
cfg = load_config(paths...; set)
if cfg.physics.radiation.scheme == :rrtmgp
    Base.identify_package("RRTMGP") === nothing &&
        error("radiation is on (physics.radiation.scheme = \"rrtmgp\") but RRTMGP cannot " *
              "be loaded here: launch with julia --project=simulations")
    @eval Main import RRTMGP, ClimaComms
end
# On --resume the run's own <root>/config.toml is authoritative: run_simulation accepts
# `--set output.dir=...` (it locates the root) and `run.integration_time` (extends the run)
# and refuses everything else.
run_simulation(cfg, exec; resume, config_sources = paths, set)
