#!/usr/bin/env julia
# Launcher for the ocean warm bubble case (2 h, walls, MYNN + ice, radiation off;
# scythe_config.toml beside this script).
#
#   julia --project=<env> run.jl [--config scythe_config.toml] [--set k.path=v ...] [--resume]
#       [--workers N] [--worker-threads N] [--blas-threads N] [--slurm]
#
# <env> is any Julia environment where Scythe is installed (a Scythe checkout itself works:
# `--project=/path/to/Scythe.jl`). This file needs nothing else from the repository: copy it
# and scythe_config.toml to any directory and run it there.
#
# Flags (Scythe.parse_launcher_args, src/run_driver.jl):
#   --config PATH              the run's one config file (default: scythe_config.toml beside
#                              this script); there are no overlay files
#   --set k.path=v             run control, e.g. --set output.dir=output/test (repeatable;
#                              recorded in the resolved <output.dir>/scythe_config.toml)
#   --resume                   continue an existing run from its segments.csv
#   --workers N                worker processes (OWB is single-patch)
#   --worker-threads N         Julia threads per worker (0 = split the machine)
#   --blas-threads N           OpenBLAS threads per process
#   --slurm                    add workers through SLURM (addprocs_slurm)
#
# Radiation is off (physics.radiation.scheme = "none"); to turn it on, edit
# scythe_config.toml -- radiation is part of Scythe and needs no special environment.

using Scythe

path, set, resume, exec, segment = Scythe.parse_launcher_args(ARGS)
cfg = load_config(path; set)
if segment > 0
    Scythe.segment_main(ARGS)
else
    run_simulation(cfg, exec; resume, config_sources = [path], set)

    # Case-specific post-step, ONLY inside a Scythe checkout: the OWB diagnostic table
    # (owb_diagnostics, benchmarks/ocean_warm_bubble.jl), reused rather than duplicated --
    # that file's own top level is guarded, so including it only defines the functions. A
    # copied run.jl (no benchmarks/ beside it) skips it with a one-line note. The table
    # reads `<t>_physical.csv`, which the production config does not write
    # (`output.formats = ["netcdf"]`), so it is also a no-op unless the config asks for CSV.
    # Only single-segment runs are covered: build_model(cfg) with no segment override
    # targets seg_01, which is the run's only (and therefore final) segment here.
    owb = joinpath(@__DIR__, "..", "..", "benchmarks", "ocean_warm_bubble.jl")
    if !isfile(owb)
        println("\nScythe: benchmarks/ocean_warm_bubble.jl not found relative to this " *
                "script -- skipping the OWB diagnostics table")
    elseif cfg.run.segment_length == 0.0
        model = build_model(cfg)
        if isdir(model.output_dir) &&
           any(f -> endswith(f, "_physical.csv"), readdir(model.output_dir))
            include(owb)
            diags = Base.@invokelatest owb_diagnostics(model)   # defined by the include above
            println("\n=== ocean_warm_bubble diagnostics ($(model.output_dir)) ===")
            for k in sort(collect(keys(diags)))
                println("  ", k, " = ", diags[k])
            end
        else
            println("\nScythe: no *_physical.csv in $(model.output_dir) -- skipping the " *
                    "OWB diagnostics table (set output.formats = [\"netcdf\", \"csv\"] in " *
                    "scythe_config.toml to get one)")
        end
    end
end
