#!/usr/bin/env julia
# Launcher for the axisymmetric TC (scythe_config.toml beside this script).
#
#   julia --project=<env> run.jl [--config scythe_config.toml] [--set key.path=value ...]
#       [--resume] --workers-per-patch 1,1,1 [--worker-threads N] [--blas-threads N] [--slurm]
#
# <env> is any Julia environment where Scythe is installed (a Scythe checkout itself works:
# `--project=/path/to/Scythe.jl`). This file needs nothing else from the repository: copy it
# and scythe_config.toml to any directory and run it there. Radiation (RRTMGP) is part of
# Scythe; no special environment is needed.
#
# A run is ONE config file: edit scythe_config.toml (examples/fine.toml and examples/rlr.toml
# are alternate grids to copy into it). `--set` is for run control -- output.dir, extending
# run.integration_time on --resume -- and is recorded in the header of the resolved
# <output.dir>/scythe_config.toml. A relative output.dir resolves against the config file's
# directory; a relative --set path against the working directory.
#
# Workers: this case is a 3-patch nest, so pass `--workers-per-patch 1,1,1` -- one worker per
# nest (the default, a single entry, is refused for a nest). `--worker-threads N` threads per
# worker (0 = split the machine).
#
# Examples (from this directory):
#   julia --project=<env> run.jl --workers-per-patch 1,1,1 --worker-threads 4
#   julia --project=<env> run.jl --workers-per-patch 1,1,1 --set output.dir=output/tc_24h \
#         --set run.integration_time=86400
#   julia --project=<env> run.jl --workers-per-patch 1,1,1 --resume \
#         --set output.dir=output/tc_24h                                      # continue a run
#   julia --project=<env> run.jl --workers-per-patch 1,1,1 --resume \
#         --set output.dir=output/tc_24h --set run.integration_time=172800    # extend it
#
# Output goes to <output.dir>/seg_NN/nest{i}/ (see README.md). A fresh start refuses a tree
# that already holds output: mv it aside (never rm model output), or continue with --resume.
# On --resume the run's own <root>/scythe_config.toml is authoritative: run_simulation
# accepts `--set output.dir=...` (it locates the root) and `run.integration_time` (extends
# the run) and refuses everything else.

using Scythe

path, set, resume, exec, segment = Scythe.parse_launcher_args(ARGS)
cfg = load_config(path; set)
segment > 0 ? Scythe.segment_main(ARGS) :
              run_simulation(cfg, exec; resume, config_sources = [path], set)
