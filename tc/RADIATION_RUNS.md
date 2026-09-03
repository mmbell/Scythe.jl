# TC radiation runs (S6)

`SCYTHE_TC_RAD` selects the arm (wired in `tc/tc_init.jl` `make_base`, constants in
`tc/tc_params.jl`), in the style of the benchmark's `SCYTHE_O01_RAD`:

| value            | meaning                                                              |
|-------------------|----------------------------------------------------------------------|
| unset / `0`        | no radiation keys at all — bit-identical to a pre-radiation TC run   |
| `lw`                | RRTMGP clear-sky longwave only, no sun (`:clearsky`, `:solar=:none`) |
| `allsky`            | cloud-optics-coupled all-sky method, still no sun                    |
| `diurnal`           | allsky + full diurnal cycle at `RAD_LATITUDE`/`RAD_START_DOY`/`RAD_START_HOUR` (20 N, day 240, local midnight at lon 0) — the S6 production arm |

Tuning knobs (env, no file edit needed — a restart re-reads `tc_params.jl`/`tc_init.jl`):
`SCYTHE_TC_RAD_FORCING` (`full` default | `anomaly`), `SCYTHE_TC_RAD_INTERVAL` (seconds,
default 300 = `RAD_INTERVAL`), `SCYTHE_TC_RAD_ZMAX` (metres, default 17000 =
`RAD_ZMAX` = `Z_DAMP`, the DK83 sponge onset).

`physical_params[:SST]` (already set to `SST_K` = 302.65 K) becomes the radiative surface
temperature automatically — `radiation_surface_temperature`'s `:T_sfc` > `:SST` >
extrapolated-air-temperature precedence, and no `:T_sfc` is set anywhere in the TC config.

RRTMGP lookup-table artifacts must be prewarmed on a login node before a compute-node job
requests any arm (`tools/rrtmgp_prewarm.jl`); `tc/_preflight.sh`'s `tc_preflight` now
CHECK-ONLY-asserts they are present whenever `SCYTHE_TC_RAD` is set, and fails loudly with
that instruction if not.

## S6 smoke run (this stage, laptop, 2026-09-02)

30-min 3-nest axisym coarse smoke, `SCYTHE_TC_RAD=diurnal`:

```
JULIA_NUM_THREADS=4 SCYTHE_TC_RAD=diurnal \
SCYTHE_TC_OUTDIR=$PWD/tc/output/tc_s6_rad_smoke_diurnal \
julia --project=. tc/tc_run_axisym.jl 1800 --csv
```

Radiation-off control (unset `SCYTHE_TC_RAD`), 5-min, confirms no radiation setup line, no
sidecar files, no per-call trace (`rs.active == false`, `EMPTY_RADIATION`):

```
JULIA_NUM_THREADS=2 \
SCYTHE_TC_OUTDIR=$PWD/tc/output/tc_s6_rad_smoke_control \
julia --project=. tc/tc_run_axisym.jl 300 --csv
```

Both completed cleanly (`finite=true` on every nest); see the S6 report for the trace
excerpts, OLR/cooling numbers and the `read_radiation` reassembly check.

## Production runs (NOT launched by S6 — commands for the next session/user)

### 6 h, laptop, `:full` forcing

```
JULIA_NUM_THREADS=4 SCYTHE_TC_RAD=diurnal \
SCYTHE_TC_OUTDIR=$PWD/tc/output/tc_rad_6h_full \
julia --project=. tc/tc_run_axisym.jl 21600 --csv
```

### 6 h, laptop, `:anomaly` forcing (for comparison against `:full`)

```
JULIA_NUM_THREADS=4 SCYTHE_TC_RAD=diurnal SCYTHE_TC_RAD_FORCING=anomaly \
SCYTHE_TC_OUTDIR=$PWD/tc/output/tc_rad_6h_anomaly \
julia --project=. tc/tc_run_axisym.jl 21600 --csv
```

### 24 h, laptop, `:full` forcing (the plan's first production diurnal target)

```
JULIA_NUM_THREADS=4 SCYTHE_TC_RAD=diurnal \
SCYTHE_TC_OUTDIR=$PWD/tc/output/tc_rad_24h_full \
julia --project=. tc/tc_run_axisym.jl 86400 --csv
```

### 24 h, single-node sbatch shape (`scythe_tc.sbatch`)

```
sbatch --export=ALL,SCYTHE_TC_RAD=diurnal,SCYTHE_TC_OUTDIR=$PWD/tc/output/tc_rad_24h_full \
       tc/scythe_tc.sbatch 86400
```

`:anomaly` variant: add `SCYTHE_TC_RAD_FORCING=anomaly` to the `--export=ALL,...` list and
point `SCYTHE_TC_OUTDIR` at a separate tree.

### 24 h, node-per-patch sbatch shape (`scythe_tc_multinode.sbatch`)

```
sbatch --export=ALL,SCYTHE_TC_RAD=diurnal,SCYTHE_TC_OUTDIR=$PWD/tc/output/tc_rad_24h_full_mn \
       tc/scythe_tc_multinode.sbatch 86400
```

Both sbatch wrappers already `tc_preflight` before launching, which now includes the
RRTMGP artifact check whenever `SCYTHE_TC_RAD` is exported — run
`julia --project=<Scythe.jl checkout> tools/rrtmgp_prewarm.jl` on the login node first if
that check has never been run on the cluster.

## What to look at (S6 acceptance, per the plan)

- **OLR** — per-nest domain-mean/min/max at the full column (70 km, includes the
  stratospheric extension) and at the model top (25 km face); should sit in the low/mid
  260s W/m^2 range for this clear-sky tropical column and drift slowly as the vortex
  moistens/dries columns near the core.
- **Diurnal heating cycle** — starting at local midnight (`RAD_START_HOUR=0` at lon 0),
  the first ~12 h of a run stay `cos_zenith = 0` (SW identically zero — confirmed on the
  S6 smoke run at every call); sunrise near h12 should show `cos_zenith` and `q_sw`
  turning on together with the per-step `sw_scale` rescale (turn on
  `options[:radiation_trace_sw]` via `extra_options` for the per-step window if the
  4-intermediate-point-per-interval view is wanted — off by default on the TC, unlike the
  O01 `diurnal` arm).
- **Sponge-layer drift** — compare the state above `Z_DAMP` (17 km, where the taper zeroes
  the held heating, S4's O01 finding) between a radiation-on and radiation-off run over the
  same window; the taper exists precisely so radiation should not need to fight the DK83
  sponge.
- **Cost** — per-call wall time and the fraction of total wall clock it represents; the S6
  smoke run measured ~80-160 ms/call (steady state) per nest at 45-48 columns x 150 layers
  (+15 extension) at a 300 s cadence, negligible against the run's total wall clock. If a
  24 h production run's radiation cost becomes material, `options[:radiation_layer_stride]`
  (e.g. 3, cell-centre layers) is the lever — untested on the TC as of S6.
- **`:anomaly` vs `:full`** — `:anomaly` should read near machine-zero heating in the
  undisturbed far field (RE87's compact vortex vanishes by `r_0` = 800 km) and isolate the
  vortex's own departure from the reference column; `:full` carries the whole clear-sky
  cooling everywhere. Run both 6 h arms above and diff the sidecar `q_lw`/`q_sw` fields
  (`Scythe.read_radiation`) outside vs inside the vortex.
