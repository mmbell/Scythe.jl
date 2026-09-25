#!/usr/bin/env julia
# ocean_warm_bubble/run.jl -- local launcher for the ocean warm bubble case (2 h, walls,
# MYNN + ice, radiation off; simulations/ocean_warm_bubble/config.toml).
#
#   julia --project=simulations simulations/ocean_warm_bubble/run.jl [overlay.toml ...] \
#       [--set k.path=v ...] [--resume] [--workers N | --workers-per-patch a,b,c] \
#       [--worker-threads N] [--blas-threads N] [--slurm] [--segment k]
#
# Flags (Scythe.parse_launcher_args, src/run_driver.jl):
#   --config PATH             case config file (default: config.toml beside this script)
#   overlays                   positional overlay TOML files, applied in order after --config
#   --set k.path=v             override one key (repeatable), e.g.
#                               --set physics.boundary_layer.scheme=louis
#   --resume                   continue an existing run from its segments.csv
#   --workers N                worker processes (OWB is single-patch)
#   --workers-per-patch a,b,c  per-patch worker counts (nests only; unused here)
#   --worker-threads N         Julia threads per worker (0 = split the machine)
#   --blas-threads N           OpenBLAS threads per process
#   --slurm                    add workers through SLURM (addprocs_slurm)
#   --segment k                internal: run only segment k (child process)
#
# Radiation is off by default (physics.radiation.scheme = "none"); turning it on with
# --set physics.radiation.scheme=rrtmgp needs `--project=simulations` (RRTMGP/ClimaComms
# are weak dependencies of Scythe loaded only from that workspace member -- the standard
# launch environment for every case).

using Scythe

paths, set, resume, exec, segment = Scythe.parse_launcher_args(ARGS)
cfg = load_config(paths...; set)

if cfg.physics.radiation.scheme == :rrtmgp
    Base.identify_package("RRTMGP") === nothing &&
        error("physics.radiation.scheme = rrtmgp but RRTMGP cannot be loaded from the " *
              "active project $(Base.active_project()); launch with " *
              "`julia --project=simulations`")
    @eval Main import RRTMGP, ClimaComms
end

if segment > 0
    Scythe.segment_main(ARGS)
else
    run_simulation(cfg, exec; resume, config_sources = paths, set)

    # Case-specific post-step: the OWB diagnostic table (owb_diagnostics,
    # benchmarks/ocean_warm_bubble.jl), reused rather than duplicated -- that file's own
    # top level is guarded (`abspath(PROGRAM_FILE) == @__FILE__`), so including it here
    # only defines the functions. They read `<t>_physical.csv`, which the production
    # config does not write (`output.formats = ["netcdf"]`); this step is a no-op unless
    # the run (or a --set) also asked for CSV, e.g. --set output.formats='["netcdf","csv"]'.
    # Only single-segment runs are covered: build_model(cfg) with no segment override
    # targets seg_01, which is the run's only (and therefore final) segment here.
    if cfg.run.segment_length == 0.0
        model = build_model(cfg)
        if isdir(model.output_dir) &&
           any(f -> endswith(f, "_physical.csv"), readdir(model.output_dir))
            include(joinpath(@__DIR__, "..", "..", "benchmarks", "ocean_warm_bubble.jl"))
            diags = owb_diagnostics(model)
            println("\n=== ocean_warm_bubble diagnostics ($(model.output_dir)) ===")
            for k in sort(collect(keys(diags)))
                println("  ", k, " = ", diags[k])
            end
        else
            println("\nScythe: no *_physical.csv in $(model.output_dir) -- skipping the " *
                     "OWB diagnostics table (add --set output.formats='[\"netcdf\",\"csv\"]' " *
                     "to get one)")
        end
    end
end
