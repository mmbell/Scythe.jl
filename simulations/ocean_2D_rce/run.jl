#!/usr/bin/env julia
# Launcher for the open-ocean 2-D RCE case (scythe_config.toml beside this script).
#
#   julia --project=<env> run.jl [--config scythe_config.toml] [--set key.path=value ...]
#       [--resume] [--workers N] [--worker-threads N] [--blas-threads N] [--slurm]
#
# <env> is any Julia environment where Scythe is installed (a Scythe checkout itself works:
# `--project=/path/to/Scythe.jl`). This file needs nothing else from the repository: copy it
# and scythe_config.toml to any directory and run it there. This case runs diurnal all-sky
# RRTMGP by default; radiation is part of Scythe and needs no special environment.
#
# A run is ONE config file: edit scythe_config.toml (examples/full.toml is the
# full-resolution grid to copy into it). `--set` is for run control -- output.dir, extending
# run.integration_time on --resume -- and is recorded in the header of the resolved
# <output.dir>/scythe_config.toml.
#
# Examples (from this directory):
#   julia --project=<env> run.jl --workers 1
#   julia --project=<env> run.jl --workers 1 --resume
#   julia --project=<env> run.jl --workers 1 --worker-threads 18 --set output.dir=output/ctrl1

using Scythe

path, set, resume, exec, segment = Scythe.parse_launcher_args(ARGS)
cfg = load_config(path; set)
segment > 0 ? Scythe.segment_main(ARGS) :
              run_simulation(cfg, exec; resume, config_sources = [path], set)
