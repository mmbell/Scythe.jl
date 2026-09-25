# Open-ocean RCE (radiative-convective equilibrium)

`config.toml` is a 2-D (x-z), periodic, open-ocean radiative-convective-equilibrium case:
MYNN-EDMF boundary layer (`:scythe` fidelity bundle + stability-aware surface layer,
struct defaults), ISHMAEL ice, diurnal all-sky RRTMGP radiation, seeded low-level
temperature noise instead of a bubble. See `config.toml`'s header for the full physics/grid
reference -- every key is spelled out there, even where it equals the `RunConfig` struct
default (`src/config.jl`), so the file alone describes what runs. `full.toml` overlays the
full-resolution grid (3 km x 300 m cells); every other case config key is unchanged.

This is not a `run_benchmark` case: there is no published target and no committed
regression reference (an RCE equilibrium is a many-day statistical statement, not a
bitwise one). `test/test_config_equivalence.jl` checks `build_model(load_config(...))`
against the legacy driver's fingerprint instead.

## Why segments

A 20-day run at ocean-RCE resolution is many hours of wall clock. Rather than one long
`julia` process (which loses everything to a walltime kill or a crash), a run is split into
SEGMENTS: one fresh `julia` process per segment, each warm-restarting from the previous
segment's JLD2 checkpoint (`write_restart`'s guaranteed final checkpoint,
`finalize_model`). `run.jl` with no `--segment` flag is the ORCHESTRATOR
(`Scythe.run_simulation`, `src/run_driver.jl`) -- it spawns one child
`julia ... run.jl --segment k` per segment, in order, waiting for each to finish (and
stopping with a clear message naming the segment and its log if one fails) before starting
the next. A `--segment k` invocation (`Scythe.segment_main`) is a normal, standalone,
reproducible run of exactly that segment -- useful for re-running one segment by hand.

## Launching

`--project=simulations`, not the root project: this case runs diurnal RRTMGP by default,
and RRTMGP/ClimaComms are WEAK dependencies of Scythe (`ext/ScytheRRTMGPExt.jl`) that load
only from an environment listing them. `simulations/Project.toml` is a workspace member
sharing the root `Manifest.toml`; the orchestrator launches its segment children in the
same environment (`Base.active_project()`).

```sh
# Local, foreground, the default 20 days in 2-day segments (10 children):
julia --project=simulations simulations/ocean_2D_rce/run.jl --workers 1

# An overlay (the full-resolution grid) plus an override:
julia --project=simulations simulations/ocean_2D_rce/run.jl full.toml --workers 1 \
    --set run.integration_time=864000

# On the cluster (single node, 20 cores; see scythe_rce.sbatch's header for more examples):
sbatch simulations/ocean_2D_rce/scythe_rce.sbatch                    # 20-day, default physics
sbatch simulations/ocean_2D_rce/scythe_rce.sbatch \
    --set run.integration_time=3456000 --set run.segment_length=345600   # 40 d / 4-d segments
sbatch simulations/ocean_2D_rce/scythe_rce.sbatch --set output.dir=output/ctrl1
                                                                       # tagged output tree
```

Every physics/grid knob that used to be a `SCYTHE_RCE_*` environment variable is now a
config key: change it with a repeatable `--set key.path=value` or an overlay TOML file
(`config_paths = [config.toml, overlay1.toml, ...]`, later files winning). There are no
`SCYTHE_RCE_*` or `SCYTHE_BENCH_TAG` environment knobs left -- an un-tagged run with no
extra arguments is always exactly `config.toml` as committed.

## Output layout

```
<output.dir>/                        default: simulations/ocean_2D_rce/output/
├── config.toml       the resolved configuration (every field, defaults included) +
│                      a provenance header (git SHA/dirty, date, host, sources, --set)
├── reference.ref      reference-state sounding (written once, by segment 1's init)
├── ics.csv             segment 1's noise-seeded initial conditions
├── segments.csv        one row per segment (src/run_driver.jl SEGMENTS_CSV_COLS):
│                        segment, dir, t_offset_s, integration_time_s, ts,
│                        output_interval_s, restart_file, status (running/done/failed),
│                        scythe_sha, started, finished, n_patches, ts_actual, start_hour,
│                        workers, threads, host, note
├── seg_01/
│   ├── config.toml      this segment's resolved config (t_offset, start_hour, IC paths
│   │                    in the header)
│   ├── segment_stdout.log
│   └── <t>.nc, <T>.jld2   comprehensive NetCDF output + the final checkpoint seg_02
│                          restarts from
├── seg_02/
└── ...
```

(A nested case would add one more level, `seg_NN/nest{i}/`; this case has one patch.)

## How `segments.csv` stitches time together

Each segment's own `ModelParameters.integration_time` is just that segment's length in
seconds (`t = 0` at the start of every child process) -- the model loop knows nothing about
the segments before or after. Two things carry continuity across a segment boundary:

- **The restart file**: segment `k`'s initial condition is segment `k - 1`'s final JLD2
  checkpoint (`Scythe.checkpoint_path`, `src/io.jl`) -- a WARM restart (spectral
  coefficients only; the AB3 tendency history is not checkpointed, so the first step after
  every segment boundary is Euler, not AB3).
- **The diurnal solar clock**: each segment's `start_hour` (recorded in its own row of
  `segments.csv`, and in the header of its `seg_NN/config.toml`) is
  `physics.radiation.start_hour + t_offset_s / 3600`, the CUMULATIVE elapsed time in hours
  before that segment starts -- `solar_state` (`src/radiation_state.jl`) computes
  `doy = start_doy + (start_hour*3600 + t_model)/86400` with `t_model` zero-based within the
  current process. Without this offset every segment would restart the diurnal cycle at the
  same local time of day it started at.

`segments.csv`'s `t_offset_s` column records the same cumulative offset (in seconds) for
every segment, so the whole run's wall-clock-to-model-time mapping can be reconstructed
from the file alone.

## Resuming

```sh
# The default tree:
julia --project=simulations simulations/ocean_2D_rce/run.jl --workers 1 --resume

# A tagged tree: point --config at THAT tree's OWN resolved config.toml (already an
# absolute output.dir) -- NOT the case file with --set output.dir=... again, which
# --resume refuses (see below):
julia --project=simulations simulations/ocean_2D_rce/run.jl \
    --config output/ctrl1/config.toml --workers 1 --resume
# or: sbatch simulations/ocean_2D_rce/scythe_rce.sbatch --config output/ctrl1/config.toml --resume
```

`--resume` re-reads `<output.dir>/config.toml` (not the case file on disk -- the recorded
configuration) and skips every segment whose `segments.csv` row already reads
`status = done`; a segment marked `failed` (or missing entirely) is re-run. The `cfg`
built from `--config`/overlays/`--set` supplies ONLY the root (`output.dir`) to resume;
the only `--set` a resumed run accepts is `run.integration_time=...`, to extend the run
with more segments -- any OTHER `--set` (including `output.dir=...`, even to the tree's
own current value) is refused, because a resumed run keeps its recorded configuration.
That is why a tagged tree resumes through `--config <root>/config.toml`, not through
repeating the `--set output.dir=...` that created it. A FRESH start (no `--resume`)
refuses to run at all if
the root already holds `seg_01/` or any model output -- move the tree aside (`mv`, never
`rm`) instead.

## Post-processing

Three scripts, moved here from `benchmarks/`, read an existing output root and never run the
model (`benchmarks/common/rce_io.jl` holds their shared NetCDF/segment-stitching helpers):

```sh
julia --project=. simulations/ocean_2D_rce/rce_diagnostics.jl --dir <root> [--avg-days M]
julia --project=. simulations/ocean_2D_rce/rce_movie.jl --dir <root> [--out <root>/rce.mp4]
julia --project=. simulations/ocean_2D_rce/rce_sounding.jl --dir <root> [--avg-days 5]
```

`rce_sounding.jl` turns the time- and domain-mean state of a run into a sounding file in
the format Springsteel's reference-state builder reads, for seeding a TC run from a
model-consistent, SST-tuned environment; it reads the SST and rebuilds the model's own grid
from `<root>/config.toml` (`load_config` + `build_model`), so it needs
`--project=simulations` whenever the run's `physics.radiation.scheme` is `"rrtmgp"` (the
case default).
