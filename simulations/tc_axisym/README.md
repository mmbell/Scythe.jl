# Axisymmetric tropical cyclone (`tc_axisym`)

A 3-nest axisymmetric (radius-height, RiRk B-spline) tropical cyclone on an f-plane at
15 N over a 29.5 C ocean: a Rotunno & Emanuel (1987) eq. (37) vortex, gradient-wind and
hydrostatically balanced on the model's own grid, with a moistened inner core, spun up by
surface fluxes under the full production physics — ISHMAEL ice with two-moment rain,
MYNN level-2.5 boundary layer (`scythe` fidelity, EDMF off), Monin-Obukhov surface layer
with Komori drag, diurnal all-sky RRTMGP radiation, horizontal Smagorinsky mixing, the
DK83 sponge, and the state-dependent vertical semi-implicit solver.

| file | what |
|---|---|
| `config.toml` | the case: coarse grid (10/20/40 km radial x 500 m vertical, nests at 150/450/1050 km, 25 km lid), ts 0.5 s, 5 days in 1-day segments. The case definition is spelled out in full with the physical reasons for each value; physics is delta-only |
| `fine.toml` | overlay: the original fine grid, 3/6/12 km radial x 300 m vertical, ts 0.25 s (structure runs; ~1x real time) |
| `rlr.toml` | overlay: the 3-D cylindrical RLR geometry on the coarse grid (nests at 160/480/1120 km: RLR needs each junction a whole number of parent cells from the origin) |
| `run.jl` | local launcher |
| `scythe_tc.sbatch` | SLURM, one 20-core node, 3 workers x 6 threads |
| `scythe_tc_multinode.sbatch` | SLURM, node per patch (3 nodes), 18 threads per worker |
| `tc_movie.jl`, `tc_bl_assess.jl`, `tc_postprocess.jl`, `tc_run_io.jl` | output readers (below) |

The full configuration reference (every key, default and allowed value) is
[`../README.md`](../README.md).

## Running

Always `julia --project=simulations` (radiation is on; RRTMGP/ClimaComms are weak
dependencies of Scythe that load only from that environment). The nest needs one worker
group per patch: `--workers-per-patch 1,1,1` (the RLR overlay is the same three patches).

### Laptop

From the repository root:

```
R=simulations/tc_axisym
# the production 5-day run, into simulations/tc_axisym/output (gitignored)
julia --project=simulations $R/run.jl --workers-per-patch 1,1,1 --worker-threads 4

# a named 24 h run
julia --project=simulations $R/run.jl --workers-per-patch 1,1,1 --worker-threads 4 \
      --set run.integration_time=86400 --set output.dir=$R/output/tc_ice_24h

# continue it after a failure or a kill (runs every segment not recorded `done`)
julia --project=simulations $R/run.jl --workers-per-patch 1,1,1 --worker-threads 4 \
      --resume --set output.dir=$R/output/tc_ice_24h

# extend it to 48 h (only run.integration_time may change on --resume)
julia --project=simulations $R/run.jl --workers-per-patch 1,1,1 --worker-threads 4 \
      --resume --set output.dir=$R/output/tc_ice_24h --set run.integration_time=172800
```

A fresh start refuses a tree that already holds a run. Model output is experimental data:
`mv` an old tree aside, never delete it. Run one heavy job at a time on a laptop.

Laptop cost (coarse, 3 workers x 2 threads, 2026-09-25, this configuration): ~6 min wall
per 10 simulated minutes, i.e. ~36x slower than real time on a shared machine; the 5-day
run is a cluster job.

### Cluster

```
sbatch simulations/tc_axisym/scythe_tc.sbatch                       # 5 days, coarse, 1 node
sbatch simulations/tc_axisym/scythe_tc_multinode.sbatch             # 5 days, node per patch
sbatch simulations/tc_axisym/scythe_tc.sbatch simulations/tc_axisym/fine.toml \
       --set run.integration_time=86400                             # 1 day, fine grid
sbatch simulations/tc_axisym/scythe_tc.sbatch --resume \
       --set output.dir=$PWD/simulations/tc_axisym/output/tc_12345   # continue job 12345's tree
```

Each fresh job writes `simulations/tc_axisym/output/tc_<jobid>` (`tc_mn_<jobid>` for the
multinode shape) unless `--set output.dir=...` names a tree. Both scripts run
`simulations/_preflight.jl` first (Springsteel version, the config loads and validates,
RRTMGP lookup tables present — run `julia --project=simulations tools/rrtmgp_prewarm.jl`
once on a login node; compute nodes have no network) and tee all output to
`<run root>/job_stdout.log` after a provenance header. Before a 5-day multinode run,
measure a 1 h job against the single-node shape: at 15 x 50 cells per coarse patch the
cross-node exchange may not amortize.

The 1-day segments are the restart granularity (`run.segment_length`); checkpoints are
written every 6 h (`output.restart_interval`). A walltime-killed job is continued with
`--resume` on the same tree, from the start of the segment it was in.

## From the old environment variables

`tc/tc_run_axisym.jl` was configured by `SCYTHE_TC_*` environment variables and the
`tc/` sbatch knobs. None of them are read any more; each is a `--set` or an overlay:

| old | now |
|---|---|
| `SCYTHE_TC_RES=fine` | overlay `fine.toml` |
| RLR geometry | overlay `rlr.toml` |
| `SCYTHE_TC_TS_SCALE=s` | `--set timestep.scale=s` |
| `SCYTHE_TC_OUTDIR=p` | `--set output.dir=p` |
| positional `integration_time` | `--set run.integration_time=s` |
| `RESTART_T=t` (+ `SCYTHE_TC_OUTDIR`) | `--resume --set output.dir=<tree>` (restarts at segment boundaries) |
| `CSV=1` / `--csv` | `--set 'output.formats=["netcdf","csv"]'` |
| `TRACE=n` / `--trace n` | `--set diagnostics.state_minima_trace=n` |
| `SCYTHE_TC_SIDECARS=1` | `--set output.sidecars=true` |
| `SCYTHE_TC_SOUNDING=p` | `--set reference_state.sounding=p` |
| `SCYTHE_TC_CTRANS` / `SCYTHE_TC_RTRANS` | `--set physics.microphysics.condensate_transform=...` / `rain_transform=...` |

### Debug restart from a checkpoint

`tc/tc_debug_restart.jl` (warm-restarting each nest from a crashed run's mid-run CSV state,
with a short trace cadence) is gone; its use case is `initial_conditions.method = "file"`
on a fresh, short-`integration_time` config pointed at the per-nest checkpoint or CSV
snapshot, one path per patch, in nest order:

```
R=simulations/tc_axisym/output/tc_ice_24h    # the tree that crashed/stopped
O=simulations/tc_axisym/output/tc_debug
julia --project=simulations $(dirname $R)/../run.jl \
      --workers-per-patch 1,1,1 --worker-threads 4 \
      --set run.integration_time=3600 --set output.dir=$O \
      --set diagnostics.state_minima_trace=40 \
      --set initial_conditions.method=file \
      --set 'initial_conditions.file.paths=["'$R'/seg_01/nest1/18000.0.jld2","'$R'/seg_01/nest2/18000.0.jld2","'$R'/seg_01/nest3/18000.0.jld2"]'
```

A `.jld2` checkpoint gives a warm restart (the AB3 tendency history is rebuilt, a small
transient); a `<t>_physical.csv` snapshot works too (cold restart from that state). The
reference state (`<R>/reference.ref`) is still required and is written automatically for
`method = "file"`, so nothing needs to be copied by hand.

### Boundary layer (was `tc/MYNN_RUNS.md`)

| old | now |
|---|---|
| `SCYTHE_TC_BL=mynn` | the default (`physics.boundary_layer.scheme = "mynn"`) |
| `SCYTHE_TC_BL=louis` | `--set physics.boundary_layer.scheme=louis --set physics.microphysics.ice=none` (Louis is refused with ice: it books the ice flux as vapour) |
| `SCYTHE_TC_MYNN_INTERVAL=s` | `--set physics.boundary_layer.mynn_interval=s` |
| `SCYTHE_TC_MYNN_EDMF=1` | `--set physics.boundary_layer.mynn_edmf=true` |
| `SCYTHE_TC_MYNN_FIDELITY=x` | `--set physics.boundary_layer.mynn_fidelity=x` (`scythe` is the default; `fortran`, or a comma list such as `gtr_local,pdk1`) |
| `SCYTHE_TC_SFC=gfdl_v7` / `charnock` | `--set physics.surface.roughness=gfdl_v7` (default `komori`) |
| `SCYTHE_TC_SFC_STAB=0` | `--set physics.surface.stability=false` (stability is ON by default now; the old TC default was off) |

MYNN knobs without a typed key (`:mynn_edmf_mom`, `:mynn_mix_numbers`,
`:mynn_water_carry`, ...) go in `[extra.options]` of an overlay file.

The `scythe` fidelity is the bundle `gtr_local, exner_single, pdk1` plus `rmol_sfc`
whenever the stability option is on (it is, by default). It became the model default on
2026-09-23 after the coarse 12 h hurricane pair (near-surface TKE −80 %, peak K −70 %,
spin-up unchanged); a verbatim-Fortran control is `mynn_fidelity=fortran`. EDMF plumes
fire only where the boundary layer is unstable AND the first model layer is thinner than
~300 m; the S10 assessment found them a no-op on the TC, so EDMF stays off.

The S10 definitive-pair arms (was `tc/submit_s10_pair.sh`), node-per-patch, fine grid,
24 h, CSV for the replay-harness anchor. Radiation was off in S10; add
`--set physics.radiation.scheme=none` to reproduce that. Run the 1 h calibration first
and measure the per-simulated-hour cost:

```
M=simulations/tc_axisym/scythe_tc_multinode.sbatch
F=simulations/tc_axisym/fine.toml
O=$PWD/simulations/tc_axisym/output
CSV='output.formats=["netcdf","csv"]'
# 1 h fine calibration
sbatch $M $F --set run.integration_time=3600 --set "$CSV" --set output.dir=$O/tc_s10_fine_cal
# the arms
sbatch $M $F --set run.integration_time=86400 --set "$CSV" --set output.dir=$O/tc_s10_24h_louis \
       --set physics.boundary_layer.scheme=louis --set physics.microphysics.ice=none
sbatch $M $F --set run.integration_time=86400 --set "$CSV" --set output.dir=$O/tc_s10_24h_mynn \
       --set physics.boundary_layer.mynn_fidelity=fortran --set physics.surface.stability=false
sbatch $M $F --set run.integration_time=86400 --set "$CSV" --set output.dir=$O/tc_s10_24h_mynn_edmf \
       --set physics.boundary_layer.mynn_fidelity=fortran --set physics.surface.stability=false \
       --set physics.boundary_layer.mynn_edmf=true
sbatch $M $F --set run.integration_time=86400 --set "$CSV" --set output.dir=$O/tc_s10_24h_mynn_scythe
```

(The old arms ran without ice; add `--set physics.microphysics.ice=none` to every arm
for a like-for-like repeat.) Wall time: the coarse 6 h pair took 40/52 min (Louis/MYNN)
at 4 threads on a laptop without ice or radiation; fine is ~11x the per-simulated-hour
cost of coarse.

### Radiation (was `tc/RADIATION_RUNS.md`)

Radiation is on by default: RRTMGP all-sky with the diurnal sun at 20 N, day 240, the
run starting at local midnight at longitude 0 (so sunrise near h12 puts both halves of
the diurnal cycle in the first day). The SST (302.65 K) is the radiative surface
temperature. The held heating is tapered to zero at the DK83 sponge onset
(`radiation.z_max = 0` means `physics.sponge.z_damp`, 17 km), so radiation does not fight
the sponge.

| old `SCYTHE_TC_RAD` | now |
|---|---|
| unset / `0` | `--set physics.radiation.scheme=none` |
| `lw` | `--set physics.radiation.method=clearsky --set physics.radiation.solar=none` |
| `allsky` | `--set physics.radiation.solar=none` |
| `diurnal` | the default |
| `SCYTHE_TC_RAD_FORCING=anomaly` | `--set physics.radiation.forcing=anomaly` |
| `SCYTHE_TC_RAD_INTERVAL=s` | `--set physics.radiation.interval=s` |
| `SCYTHE_TC_RAD_ZMAX=z` | `--set physics.radiation.z_max=z` |
| per-step SW trace | `--set diagnostics.radiation_trace_sw=true` |

Gas concentrations take the radiation defaults (420 ppm CO2, ...); override them with
`[extra.physical_params]` (`co2_ppm`, `ch4_ppb`, ...). Across segments the solar clock is
carried: segment k runs with `start_hour = physics.radiation.start_hour + t_offset/3600`.
If radiation cost becomes material, `[extra.options] radiation_layer_stride = 3` is the
lever (untested on the TC).

What to look at: OLR in the low/mid 260s W/m² in the clear environment, drifting slowly
as the vortex moistens or dries columns; `cos_zenith = 0` and zero SW for the first
~12 h, then SW heating turning on at sunrise; the state above 17 km with radiation on vs
off (the taper exists so the sponge layer does not drift); per-call cost (~80-160 ms per
nest at 45 columns x 150 layers at the 300 s cadence in S6); `anomaly` forcing should read
near machine-zero heating in the undisturbed far field (the vortex vanishes at r_0 =
800 km).

## Output

```
<root>/config.toml            resolved config (every field) + provenance header
<root>/reference.ref          the reference column (z p rho_d rho_v rho_c)
<root>/nest{1,2,3}_ics.csv    initial conditions
<root>/segments.csv           one row per segment: dir, t_offset_s, integration_time_s,
                              status, ts_actual, start_hour, workers, threads, host, ...
<root>/job_stdout.log         (sbatch only)
<root>/seg_NN/config.toml     the segment's values in the header (t_offset, start_hour, ICs)
<root>/seg_NN/segment_stdout.log
<root>/seg_NN/nest{i}/<t>.nc  comprehensive NetCDF (t local to the segment)
<root>/seg_NN/nest{i}/<t>.jld2, scythe_{out,err}.log
```

Each `<t>.nc` is the comprehensive file: primes, totals, retrieved thermodynamics,
hydrometeors (cloud, rain, and the ice species), reflectivity and precipitation,
column integrals, and the physics groups the run carries (MYNN `K_m, K_h, e, mynn_*`,
radiation, surface), listed in the global attribute `physics_groups`. The per-tile
`<t>_mynn_i*.nc` / `<t>_radiation_i*.nc` sidecars are opt-in (`output.sidecars = true`).
Segment k > 1 starts from `seg_{k-1}/nest{i}/<D>.jld2` for every nest and writes its own
t = 0 file, a duplicate of the previous segment's last one; the readers drop it.

## Readers

All three take a run ROOT, stitch the segments onto one time axis
(`t = t_offset_s + t_local`, via `tc_run_io.jl`), read the number of nests from
`<root>/config.toml`, and still read a legacy `tc/tc_run_axisym.jl` tree
(`<root>/nest{i}/<t>.nc`, `tc_exact.ref`) as one segment. They run in the ROOT project,
which carries CairoMakie and NCDatasets:

```
# radius-height movie: reflectivity, v contours, (u,w) vectors, all nests overlaid
julia --project=. simulations/tc_axisym/tc_movie.jl --root <root> [--rad | --bl] [--fps 4]
# -> <root>/<name>_rz[_rad|_bl].mp4 and a _late.png still

# boundary-layer assessment: hourly nest table, provenance (config header, segments,
# per-segment wall clock), and with two or more roots a config diff + final-time table
julia --project=. simulations/tc_axisym/tc_bl_assess.jl --root <A> [--root <B>] [--nest 1] [--out report.md]

# LEGACY: derived products from raw prognostic-only snapshots (<t>_raw.nc from
# output.formats "netcdf_raw", or an old tree's <t>.nc); refuses comprehensive files
julia --project=. simulations/tc_axisym/tc_postprocess.jl --root <root>
```

The RCE-derived sounding (`rce_sounding.jl`, see `simulations/ocean_2D_rce/`) is used
with `--set reference_state.sounding=<path>`; the written `theta` is rebuilt from `(T, p)`
as `calculate_pressure_reference_state` assumes. Check SST − T_air at the surface is
positive for a new sounding: the vortex's own warm anomaly must not reverse the air-sea
fluxes.
