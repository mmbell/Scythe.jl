# Ocean warm bubble

A mesoscale warm bubble over a fixed-SST ocean, at production-like spacing (the TC coarse
grid: 6 km x 500 m cells, 300 km x 25 km, lateral walls), with surface fluxes and drag on.
The boundary-layer / full-physics-suite case: an RH-preserving 3 K bubble (30 km
half-width) over open ocean, MYNN + ISHMAEL ice, radiation off (a 2-hour run does not feel
it). See `config.toml` for the full resolved case (domain, grid, sounding, initial
condition and environment are spelled out there in full; physics/numerics stay delta-only
against the `RunConfig` struct defaults in `src/config.jl`).

This is a SINGLE SEGMENT run (`run.segment_length = 0`): 2 hours is short enough that it
never needs restart chaining, so it runs to completion in one `julia` process.

## Running locally

```sh
# The production case, unmodified:
julia --project=simulations simulations/ocean_warm_bubble/run.jl --workers 1 --worker-threads 6

# Radiation is off in the case config; the root project is enough as long as it stays off.
julia --project=. simulations/ocean_warm_bubble/run.jl --workers 1 --worker-threads 6
```

Every variation is a `--set key.path=value` (repeatable) or an overlay TOML file --
`simulations/ocean_warm_bubble/run.jl`'s header lists every flag. There is no `full.toml`
overlay for this case (unlike `ocean_2D_rce`); grid changes go through `--set grid.*` /
`--set timestep.*` directly.

```sh
# Louis instead of MYNN (surface fluxes stay on; MYNN + ice is the only ice-supporting
# boundary-layer pair -- ice + Louis is refused by validate(cfg)):
julia --project=simulations simulations/ocean_warm_bubble/run.jl \
    --set physics.boundary_layer.scheme=louis --set physics.microphysics.ice=none

# Warm rain only (no ice), MYNN kept:
julia --project=simulations simulations/ocean_warm_bubble/run.jl \
    --set physics.microphysics.ice=none --set physics.microphysics.rain_moments=1

# Radiation on (diurnal all-sky RRTMGP) -- needs --project=simulations, the workspace
# member that can load RRTMGP/ClimaComms (Scythe's weak deps, ext/ScytheRRTMGPExt.jl):
julia --project=simulations simulations/ocean_warm_bubble/run.jl \
    --set physics.radiation.scheme=rrtmgp

# CSV alongside the comprehensive NetCDF, for the diagnostics post-step and the replay
# tools (the production config writes NetCDF only):
julia --project=simulations simulations/ocean_warm_bubble/run.jl \
    --set 'output.formats=["netcdf","csv"]'

# A different output tree (never re-run into an existing one -- see "Output layout" below):
julia --project=simulations simulations/ocean_warm_bubble/run.jl \
    --set output.dir=/path/to/scratch/owb_ctrl1
```

When `physics.radiation.scheme` (from the config, an overlay, or `--set`) resolves to
`rrtmgp`, `run.jl` checks that RRTMGP/ClimaComms can be loaded and errors immediately
("launch with `julia --project=simulations`") if they can't, rather than failing deep
inside model construction.

### Diagnostics post-step

After the run, `run.jl` reuses `owb_diagnostics` (`benchmarks/ocean_warm_bubble.jl`, its
own top level is guarded so this `include` only defines the functions) to print the O01
rain/ice/radiation diagnostics plus the surface-flux budget and MYNN rows -- but only when
`<t>_physical.csv` files exist in the run's output (they read those, not the NetCDF). The
production config writes NetCDF only, so this step is a no-op by default; add
`--set 'output.formats=["netcdf","csv"]'` to get the table. This only covers single-segment
runs (`run.segment_length = 0`, this case's default); a config with segments would need the
diagnostics pointed at a specific `seg_NN` by hand.

## Running via SLURM

```sh
sbatch simulations/ocean_warm_bubble/scythe_owb.sbatch
sbatch simulations/ocean_warm_bubble/scythe_owb.sbatch --set physics.boundary_layer.scheme=louis
sbatch simulations/ocean_warm_bubble/scythe_owb.sbatch --set physics.radiation.scheme=rrtmgp
```

One node, one process (`--workers 1 --worker-threads $JULIA_NUM_THREADS`); every argument
after the script name is forwarded verbatim to `run.jl` (overlays and `--set`, in order).
The wrapper runs `simulations/_preflight.jl` first (Springsteel-version and, when radiation
resolves to `rrtmgp`, RRTMGP-artifact checks against the SAME resolved config), writes a
provenance header (date, hostname, git HEAD/dirty count, thread count, forwarded args), and
tees everything to `<output.dir>/job_stdout.log`. No environment-variable knobs: every
option is `--set` or an overlay, exactly as the local launcher.

## Output layout

```
<output.dir>/                   default: simulations/ocean_warm_bubble/output/
├── config.toml                 resolved config (every key, defaults included) +
│                                provenance header (git SHA, dirty count, date, sources,
│                                --set)
├── reference.ref, ics.csv      reference-state sounding + the bubble initial condition
├── segments.csv                one row (this case is single-segment): dir, time offset,
│                                integration length, ts, status, scythe_sha, started/
│                                finished timestamps, workers, threads, host
├── seg_01/
│   ├── config.toml              per-segment resolved config
│   ├── segment_stdout.log
│   └── <t>.nc [+ <t>_physical.csv, <t>_spectral.csv with CSV output] + scythe_{out,err}.log
└── job_stdout.log               sbatch wrapper only
```

Model output is experimental data: a fresh start refuses to write into an existing tree
(`seg_01` present, or any model output already there) -- `mv` it aside under a descriptive
name, never `rm` it, then start again.

## Resuming

This case never needs to resume in the ordinary sense (2 hours, one segment, no restart
chaining) -- `--resume` exists for extending the run:

```sh
julia --project=simulations simulations/ocean_warm_bubble/run.jl --resume \
    --set run.integration_time=10800.0   # extend 2 h -> 3 h
```

`--resume` re-reads `<output.dir>/config.toml`; every other `--set` is refused (a resumed
run keeps its recorded configuration).

## Benchmark harness

`benchmarks/ocean_warm_bubble.jl` is a thin `run_benchmark` wrapper around this case (so
the harness's diagnostics table, target checks and committed regression CSVs keep
working): `julia --project=. benchmarks/ocean_warm_bubble.jl --mode quick --stage mc --grid rirk`.
It always builds `config.toml` unmodified (plus `--mode full`'s grid, `--ts-factor`, and
the harness's own CSV/output-directory `--set`s) -- it takes no physics `--set` of its own.
Vary the physics with `simulations/ocean_warm_bubble/run.jl` instead.
