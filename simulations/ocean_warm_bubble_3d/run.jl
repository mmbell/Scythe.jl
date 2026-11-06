#!/usr/bin/env julia
# Launcher for the 3-D ocean warm bubble (the RRR box, 300 km x 300 km x 25 km, two one-hour
# segments, walls, MYNN + ice, radiation off; scythe_config.toml beside this script).
#
#   julia --project=<env> run.jl [--config scythe_config.toml] [--set k.path=v ...] [--resume]
#       [--workers N] [--worker-threads N] [--blas-threads N] [--slurm]
#
# <env> is any Julia environment where Scythe is installed (a Scythe checkout itself works).
# This file needs nothing else from the repository: copy it and scythe_config.toml anywhere.
# The production run is for the cluster (scythe_owb3d.sbatch: 11 workers x 20 threads, one
# x tile per node); the Mac development size is ../ocean_warm_bubble/examples/rrr_120km.toml.
#
# Flags (Scythe.parse_launcher_args, src/run_driver.jl):
#   --config PATH              the run's one config file (default: scythe_config.toml beside
#                              this script); there are no overlay files
#   --set k.path=v             run control, e.g. --set output.dir=output/test (repeatable)
#   --resume                   continue an existing run from its segments.csv
#   --workers N                worker processes (the box is tiled in x, >= 3 cells per tile)
#   --worker-threads N         Julia threads per worker (0 = split the machine)
#   --blas-threads N           OpenBLAS threads per process
#   --slurm                    add workers through SLURM (addprocs_slurm)

using Scythe

path, set, resume, exec, segment = Scythe.parse_launcher_args(ARGS)
cfg = load_config(path; set)
if segment > 0
    Scythe.segment_main(ARGS)
else
    run_simulation(cfg, exec; resume, config_sources = [path], set)
end
