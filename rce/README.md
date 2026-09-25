# Open-ocean RCE (radiative-convective equilibrium)

`benchmarks/ocean_rce.jl` is a standalone, multi-day driver for a 2D (x-z), periodic,
open-ocean radiative-convective-equilibrium run: MYNN-EDMF boundary layer (`:scythe`
fidelity bundle + stability-aware surface layer by default), ISHMAEL ice, diurnal all-sky
RRTMGP radiation, seeded low-level temperature noise instead of a bubble. It is built from
`benchmarks/ocean_warm_bubble.jl` but is **not** a `run_benchmark` case: there is no
published target and no committed regression reference (an RCE equilibrium is a many-day
statistical statement, not a bitwise one).

See the header of `benchmarks/ocean_rce.jl` for the full physics/grid/CLI reference (every
`SCYTHE_RCE_*` env knob is documented there).

## Why segments

A 20-day run at ocean-RCE resolution is many hours of wall clock. Rather than one long
`julia` process (which loses everything to a walltime kill or a crash), the run is split
into SEGMENTS: one fresh `julia` process per segment, each warm-restarting from the
previous segment's JLD2 checkpoint (`write_restart`'s guaranteed final checkpoint). Running
`ocean_rce.jl` with no `--segment` flag is the ORCHESTRATOR -- it spawns one child
`julia ... ocean_rce.jl ... --segment k` per segment, in order, waiting for each to finish
(and stopping with a clear message if one fails) before starting the next. A `--segment k`
invocation is a normal, standalone, reproducible run of exactly that segment -- useful for
re-running one segment by hand after a crash.

## Launching

`--project=simulations`, not the root project: the RCE runs diurnal RRTMGP by default, and
RRTMGP/ClimaComms are WEAK dependencies of Scythe (`ext/ScytheRRTMGPExt.jl`) that load only
from an environment listing them. `simulations/Project.toml` is a workspace member sharing
the root `Manifest.toml`; the orchestrator launches its segment children in the same
environment (`Base.active_project()`).

```sh
# Local, foreground, the default 20 days in 2-day segments (10 children):
julia --project=simulations benchmarks/ocean_rce.jl --mode quick --stage mc --grid rirk

# On the cluster (single node, 20 cores; see rce/scythe_rce.sbatch's header for knobs):
sbatch rce/scythe_rce.sbatch                 # 20-day, quick, default physics
sbatch rce/scythe_rce.sbatch 40 4            # 40-day, 4-day segments
sbatch --export=ALL,SCYTHE_RCE_TAG=ctrl1,SCYTHE_RCE_ICE=0 rce/scythe_rce.sbatch
                                              # tagged, warm-rain-only control
```

Every physics/grid knob is a `SCYTHE_RCE_*` environment variable (passed to `sbatch` with
`--export=ALL,VAR=value`, or exported before a local run) -- all are no-ops when unset, so
the un-tagged default run is always the same configuration described in
`benchmarks/ocean_rce.jl`'s header comment.

## Output layout

```
benchmarks/output/ocean_rce_<mode>_mc_rirk<tag>/
├── rce_exact.ref     reference-state sounding (written once, by segment 1's init)
├── rce_ics.csv       segment 1's noise-seeded initial conditions
├── segments.csv      one row per segment: dir, time offset, integration length, ts,
│                     restart file, status (running/done/failed), scythe_sha, started/
│                     finished timestamps
├── seg_01/           comprehensive-NetCDF <t>.nc output tree + scythe_out.log/
│                     scythe_err.log (written by the model itself) + the final <T>.jld2
│                     checkpoint that seg_02 restarts from
├── seg_02/
└── ...
```

`<tag>` is `SCYTHE_BENCH_TAG` (as usual) combined with `--tag <s>` / `SCYTHE_RCE_TAG` (the
sbatch wrapper maps its own `SCYTHE_RCE_TAG` onto `SCYTHE_BENCH_TAG`); both may be used
together. `<mode>` is `quick` or `full` (`--mode`).

## How `segments.csv` stitches time together

Each segment's `ModelParameters.integration_time` is just that segment's own length in
seconds (`t=0` at the start of every child process) -- `run_model`/`finalize_model` know
nothing about the segments before or after. Two things carry continuity across the
segment boundary:

- **The restart file**: segment `k`'s `initial_conditions` is segment `k-1`'s final JLD2
  checkpoint (`seg_<k-1>/<round(length_{k-1}*86400; digits=2)>.jld2`); this is a WARM
  restart (spectral coefficients only -- the two-step AB3 tendency history is not
  checkpointed, so the first step after every segment boundary is Euler, not AB3 -- see
  `write_restart`'s docstring in `src/io.jl`).
- **The diurnal solar clock**: `physical_params[:start_hour]` is set to the CUMULATIVE
  elapsed time in hours before this segment starts (`t_offset_days * 24`), because
  `solar_state` (`src/radiation_state.jl`) computes `doy = start_doy +
  (start_hour*3600 + t_model)/86400` with `t_model` zero-based within the current
  process. Without this offset every segment would restart the diurnal cycle at the same
  local time of day it started at.

`segments.csv`'s `t_offset_s` column records that same cumulative offset (in seconds) for
each segment, so the whole run's wall-clock-to-model-time mapping can be reconstructed
from the file alone, without re-deriving it from `--days`/`--segment-days`.

## Resuming

```sh
julia --project=simulations benchmarks/ocean_rce.jl --mode quick --stage mc --grid rirk --resume
# or: sbatch --export=ALL,RESUME=1,SCYTHE_RCE_TAG=ctrl1 rce/scythe_rce.sbatch
```

`--resume` re-runs the SAME orchestrator command (same `--mode`/`--tag`/`--days`/
`--segment-days`/env knobs, so it resolves to the same output root) and skips every
segment whose `segments.csv` row already reads `status = done`. A segment marked `failed`
(or missing entirely) is re-run. A FRESH start (no `--resume`) refuses to run at all if the
root already contains `seg_01/` -- move the tree aside (`mv`, never `rm`) instead.
