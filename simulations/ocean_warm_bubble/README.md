# Ocean warm bubble

A mesoscale warm bubble over a fixed-SST ocean, at production-like spacing (the TC coarse
grid: 6 km x 500 m cells, 300 km x 25 km, lateral walls), with surface fluxes and drag on.
The boundary-layer / full-physics-suite case: an RH-preserving 3 K bubble (30 km
half-width) over open ocean, MYNN + ISHMAEL ice, radiation off (a 2-hour run does not feel
it). See `scythe_config.toml` for the full resolved case (domain, grid, sounding, initial
condition and environment are spelled out there in full; physics/numerics stay delta-only
against the `RunConfig` struct defaults in `src/config.jl`).

This is a SINGLE SEGMENT run (`run.segment_length = 0`): 2 hours is short enough that it
never needs restart chaining, so it runs to completion in one `julia` process.

## Running locally

`run.jl` is `using Scythe` plus the launcher call, so it runs from any directory under any
Julia environment where Scythe is installed (`<env>`; the Scythe checkout itself works):

```sh
# The production case, unmodified, from the checkout:
julia --project=. simulations/ocean_warm_bubble/run.jl --workers 1 --worker-threads 6

# The same case anywhere: copy the two files and run there.
mkdir -p ~/runs/owb && cp simulations/ocean_warm_bubble/{scythe_config.toml,run.jl} ~/runs/owb
cd ~/runs/owb && julia --project=<env> run.jl --workers 1 --worker-threads 6
```

A run is ONE config file: every physics or grid variation is an edit of (a copy of)
`scythe_config.toml`, so the resolved `scythe_config.toml` written into the run root is the
whole record. For example, in the copy:

```toml
# Louis instead of MYNN (surface fluxes stay on; MYNN + ice is the only ice-supporting
# boundary-layer pair -- ice + Louis is refused by validate(cfg)):
[physics.boundary_layer]
scheme = "louis"
[physics.microphysics]
ice = "none"

# Radiation on (diurnal all-sky RRTMGP; part of Scythe, no special environment):
[physics.radiation]
scheme = "rrtmgp"

# CSV alongside the comprehensive NetCDF, for the diagnostics post-step and the replay
# tools (the production config writes NetCDF only):
[output]
formats = ["netcdf", "csv"]
```

`--set` is for run control -- where the output goes, how long to run -- and every `--set`
is recorded in the header of the resolved config:

```sh
# A different output tree (never re-run into an existing one -- see "Output layout" below):
julia --project=. simulations/ocean_warm_bubble/run.jl --workers 1 \
    --set output.dir=/path/to/scratch/owb_ctrl1
```

A relative `output.dir` in the config file resolves against the config file's directory
(the default `output` lands beside `scythe_config.toml`); a relative `--set` path against
the working directory. `reference_state.sounding = "@scythe/..."` resolves against the
installed Scythe package, so a copied config finds its sounding anywhere.

Before anything is written, `run_simulation` runs `preflight_checks` (Springsteel version;
the RRTMGP lookup tables when radiation is on) and fails in seconds with a message naming
the fix.

### Diagnostics post-step

After the run, `run.jl` reuses `owb_diagnostics` (`benchmarks/ocean_warm_bubble.jl`, its
own top level is guarded so this `include` only defines the functions) to print the O01
rain/ice/radiation diagnostics plus the surface-flux budget and MYNN rows -- but only when
the script sits in a Scythe checkout (a copied `run.jl` prints a one-line note and skips
it) and `<t>_physical.csv` files exist in the run's output (they read those, not the
NetCDF). The production config writes NetCDF only, so this step is a no-op by default; set
`output.formats = ["netcdf", "csv"]` in the config to get the table. This only covers
single-segment runs (`run.segment_length = 0`, this case's default); a config with segments
would need the diagnostics pointed at a specific `seg_NN` by hand.

## Running via SLURM

```sh
sbatch simulations/ocean_warm_bubble/scythe_owb.sbatch
SCYTHE_PROJECT=/path/to/env sbatch simulations/ocean_warm_bubble/scythe_owb.sbatch
```

One node, one process (`--workers 1 --worker-threads $JULIA_NUM_THREADS`) running
`julia --project=${SCYTHE_PROJECT:-$SCYTHE} $SCYTHE/simulations/ocean_warm_bubble/run.jl`,
where `SCYTHE` (from `SCYTHE_DIR`) is the checkout holding `scythe_config.toml` and
`SCYTHE_PROJECT` any environment where Scythe is installed. Every argument after the script
name (`--set` for run control, `--resume`) is forwarded verbatim to `run.jl`. The wrapper
writes a provenance header (date, hostname, git HEAD/dirty count, thread count, forwarded
args) and tees everything to `<output.dir>/job_stdout.log`. No environment-variable knobs:
a different arm is an edited `scythe_config.toml`.

## Output layout

```
<output.dir>/                   default: simulations/ocean_warm_bubble/output/
├── scythe_config.toml          resolved config (every key, defaults included) +
│                                provenance header (git SHA, dirty count, date, sources,
│                                --set)
├── reference.ref, ics.csv      reference-state sounding + the bubble initial condition
├── segments.csv                one row (this case is single-segment): dir, time offset,
│                                integration length, ts, status, scythe_sha, started/
│                                finished timestamps, workers, threads, host
├── seg_01/
│   ├── scythe_config.toml       per-segment resolved config
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
julia --project=. simulations/ocean_warm_bubble/run.jl --resume \
    --set run.integration_time=10800.0   # extend 2 h -> 3 h
```

`--resume` re-reads `<output.dir>/scythe_config.toml`; every other `--set` (besides
`output.dir`, which only locates the tree) is refused (a resumed
run keeps its recorded configuration).

## Benchmark harness

`benchmarks/ocean_warm_bubble.jl` is a thin `run_benchmark` wrapper around this case (so
the harness's diagnostics table, target checks and committed regression CSVs keep
working): `julia --project=. benchmarks/ocean_warm_bubble.jl --mode quick --stage mc --grid rirk`.
It always builds `scythe_config.toml` unmodified (plus `--mode full`'s grid, `--ts-factor`, and
the harness's own CSV/output-directory `--set`s) -- it takes no physics `--set` of its own.
Vary the physics in a copy of `scythe_config.toml` and run it with `run.jl` instead.
