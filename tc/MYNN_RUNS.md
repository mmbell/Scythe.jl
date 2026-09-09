# TC boundary-layer runs (MYNN-EDMF S10)

`SCYTHE_TC_BL` selects the boundary-layer closure (wired in `tc/tc_init.jl`; the choice is
resolved at include time because MYNN's `rho_e` is an appended prognostic slot and `TC_VARS`
and the BC dicts are built once from the name list):

| value            | meaning                                                                 |
|------------------|-------------------------------------------------------------------------|
| unset / `louis`  | the Louis scheme + bulk surface fluxes — today's options dict byte-for-byte |
| `mynn`           | MYNN Level-2.5 closure (`options[:mynn]`), same surface fluxes, TKE slot `rho_e` |

MYNN knobs (env, no file edit): `SCYTHE_TC_MYNN_INTERVAL` (seconds the held closure —
mixing length, stability functions, cloud PDF, PBL height, plumes — is reused; K = l q S is
recomputed every step from the prognostic TKE; default 20), `SCYTHE_TC_MYNN_EDMF` (`0` default
| `1` mass-flux plumes; they fire only where the boundary layer is unstable AND the first
model layer is thinner than ~300 m — see the S7 commit d4a27d2 resolution study).
Other MYNN options (`:mynn_edmf_mom`, `:mynn_mix_numbers`, `:mynn_water_carry`,
`:mynn_K_max`, `:mynn_output`) take their defaults; add them to `bl_options` in
`tc_init.jl` when a sweep needs them.

Surface layer (shared by BOTH `louis` and `mynn` — resolved OUTSIDE the BL branch in
`tc_init.jl`'s `make_base`, so it is common to either scheme):

| value                    | meaning                                                                 |
|--------------------------|---------------------------------------------------------------------------|
| `SCYTHE_TC_SFC=komori` (default, unset) | `options[:sfc_z0]` left UNSET — the Komori et al. (2018) Cd(U) fit + constant Ck, byte-identical to today |
| `SCYTHE_TC_SFC=gfdl_v7`  | `options[:sfc_z0] = :gfdl_v7` — the HWRF/HAFS z0m(U10)/z0t(U10) polynomial fits |
| `SCYTHE_TC_SFC=charnock` | `options[:sfc_z0] = :charnock` — Charnock + Zeng et al. (1998) thermal roughness |
| `SCYTHE_TC_SFC_STAB=0` (default) / `1` | `options[:sfc_stability]` — Monin–Obukhov stability functions + Beljaars gustiness over the neutral coefficients above |
| `SCYTHE_TC_SIDECARS` | `0` (default) \| `1` | `1` also writes the per-tile `<t>_mynn_i*.nc` / `<t>_radiation_i*.nc` sidecars (mish-native; the legacy `tc_postprocess.jl` path, the replay tools, and the face-based radiation flux profiles on `zf` need them). Default: the comprehensive `<t>.nc` only. |

`SCYTHE_TC_SFC` and `SCYTHE_TC_SFC_STAB` are validated at setup (`Scythe.SFC_Z0_MODES`,
`src/mc_surface_layer.jl`) and printed on the `TC boundary layer:` line.
`SCYTHE_TC_SFC=komori` (the default) deliberately leaves `bl_options` untouched so the
Louis path's options dict stays byte-identical to a pre-S1b run; only a non-default value
adds a key.

`SCYTHE_TC_MYNN_FIDELITY` — the named deviations from the verbatim-Fortran closure
(`Scythe.MYNN_DEVIATIONS`), comma separated, e.g.
`SCYTHE_TC_MYNN_FIDELITY=gtr_local,pdk1`. Unset or `fortran` (the default) adds no
`:mynn_fidelity` key at all, so the standard run is byte-identical to a pre-F1 one. The
seven names are `gtr_local` (`g/theta_v` per level instead of `g/300 K`), `K_interface`
(`K` from the wall average of `el*S` instead of the colocated product), `sqfac1`
(`K_e = K_m` instead of `3 K_m`), `pdk1` (the Fortran log-layer surface TKE production
instead of the drag work, with the difference routed to heating), `exner_single`
(`th_sfc = SST/exner(1)`, not divided twice), `rmol_sfc` (`1/L` from
`surface_exchange`'s Monin–Obukhov solve — REQUIRES `SCYTHE_TC_SFC_STAB=1`, and setup
refuses it otherwise) and `flux_clip` (the wrapper's `hfx`/`qfx` limits on what the
closure sees; the model's own surface delivery stays unclipped). The value is validated
at setup and printed on the `TC boundary layer:` line, and it is carried into every
output file's `mynn_fidelity` attribute. The clip counters `mynn_n_hfx_clip` /
`mynn_n_qfx_clip` are reported on the census line on EVERY run, `flux_clip` or not.

Ice: the MYNN closure runs with ISHMAEL (ice legs, S8 e007c81); the Louis scheme is still
refused with ice. The TC has never run ice with either.

## Runs made (2026-09-07, commits f2055ae / d4a27d2 + S8 edits in progress, laptop)

30-min coarse smoke pair (`tc/output/tc_s10_smoke_{louis,mynn}`, 445 / 386 s wall) and the
6 h coarse pair (`tc/output/tc_s10_6h_{louis,mynn}`, 40 / 52 min wall), radiation off:

```
JULIA_NUM_THREADS=4 SCYTHE_TC_BL=louis \
SCYTHE_TC_OUTDIR=$PWD/tc/output/tc_s10_6h_louis \
julia --project=. tc/tc_run_axisym.jl 21600 --csv

JULIA_NUM_THREADS=4 SCYTHE_TC_BL=mynn SCYTHE_TC_MYNN_INTERVAL=20 SCYTHE_TC_MYNN_EDMF=0 \
SCYTHE_TC_OUTDIR=$PWD/tc/output/tc_s10_6h_mynn \
julia --project=. tc/tc_run_axisym.jl 21600 --csv
```

Assessment: `tc/output/S10_BL_ASSESSMENT.md` — at 6 h the two vortices are statistically
indistinguishable (both still in the initial adjustment, v 13.85 → 11.4 m/s, no jet
organised); MYNN +30 % wall time; nest-1 max K_m 164, K_h 262 m²/s, pblh 595 m, no K cap or
diffusion-number hits. CAVEAT: the pair ran on a working tree carrying uncommitted S8 edits
(no recompile mid-run, but not a commit) — the definitive pair is below.

## The definitive pair (NOT run yet — next session)

Fine resolution (`SCYTHE_TC_RES=fine`: 3/6/12 km nests, 300 m cells, ts 0.25), long enough
for the flux-driven spin-up (the Louis vortex intensifies after ~11 h; 24 h matches
`tc/output/tc_bhyp_24h`), radiation as in `tc/RADIATION_RUNS.md` if wanted, committed code.

### Cluster (preferred — `tc/submit_s10_pair.sh`, node-per-patch)

`tc/submit_s10_pair.sh` prints (dry run, default) or submits (`--calibrate` / `--pair` /
both) the calibration job and the three 24 h arms on `tc/scythe_tc_multinode.sbatch`:

```
tc/submit_s10_pair.sh                    # dry run: print the sbatch lines, submit nothing
tc/submit_s10_pair.sh --calibrate        # submit ONLY the 1 h fine calibration job
tc/submit_s10_pair.sh --pair             # submit ONLY the three 24 h arms
```

which resolves to:

```
# 1 h fine MYNN calibration (measure per-simulated-hour wall time and thread scaling first)
sbatch --export=ALL,SCYTHE_TC_RES=fine,SCYTHE_TC_BL=mynn,CSV=1,SCYTHE_TC_OUTDIR=$PWD/tc/output/tc_s10_fine_cal tc/scythe_tc_multinode.sbatch 3600

# the definitive pair (+ EDMF arm), 24 h, fine, --csv for the replay-harness anchor
sbatch --export=ALL,SCYTHE_TC_RES=fine,SCYTHE_TC_BL=louis,CSV=1,SCYTHE_TC_OUTDIR=$PWD/tc/output/tc_s10_24h_louis tc/scythe_tc_multinode.sbatch 86400
sbatch --export=ALL,SCYTHE_TC_RES=fine,SCYTHE_TC_BL=mynn,SCYTHE_TC_MYNN_EDMF=0,CSV=1,SCYTHE_TC_OUTDIR=$PWD/tc/output/tc_s10_24h_mynn tc/scythe_tc_multinode.sbatch 86400
sbatch --export=ALL,SCYTHE_TC_RES=fine,SCYTHE_TC_BL=mynn,SCYTHE_TC_MYNN_EDMF=1,CSV=1,SCYTHE_TC_OUTDIR=$PWD/tc/output/tc_s10_24h_mynn_edmf tc/scythe_tc_multinode.sbatch 86400
```

Wall time: the coarse 6 h pair above took 40/52 min (louis/mynn) at 4 threads on a laptop;
fine is roughly 11x the per-simulated-hour cost of coarse, so a naive scale-up puts a 24 h
fine arm at ~30-38 h at 4 threads — RUN THE CALIBRATION JOB FIRST and measure the actual
per-simulated-hour cost and thread scaling rather than trusting that estimate. At 18
threads per patch (this shape: one 20-core node per nest) expect roughly 9-15 h per 24 h
arm. The three arms run concurrently on separate node allocations without contention — the
"never run more than one heavy job at once" rule is a laptop shared-memory rule, not a
cluster one. `-t 72:00:00` (the sbatch default) covers either estimate for one submission;
`RESTART_INTERVAL` (`tc_params.jl`, 21600 s = 6 h) checkpoints often enough that a
walltime-killed or dead arm can be chained with `RESTART_T=<seconds>` and the same
`SCYTHE_TC_OUTDIR` rather than restarted from scratch. Each job's stdout (including the
`TC boundary layer:`/`TC radiation:` setup lines and the provenance header) now lands in
`$SCYTHE_TC_OUTDIR/job_stdout.log`, not just the submit-cwd `%j.log`.

### Laptop (alternative — same arms, single machine)

```
JULIA_NUM_THREADS=4 SCYTHE_TC_RES=fine SCYTHE_TC_BL=louis \
SCYTHE_TC_OUTDIR=$PWD/tc/output/tc_s10_24h_louis \
julia --project=. tc/tc_run_axisym.jl 86400 --csv

JULIA_NUM_THREADS=4 SCYTHE_TC_RES=fine SCYTHE_TC_BL=mynn SCYTHE_TC_MYNN_EDMF=0 \
SCYTHE_TC_OUTDIR=$PWD/tc/output/tc_s10_24h_mynn \
julia --project=. tc/tc_run_axisym.jl 86400 --csv

# EDMF arm (plumes fire where the BL is unstable; the first fine-grid layer is ~100 m)
JULIA_NUM_THREADS=4 SCYTHE_TC_RES=fine SCYTHE_TC_BL=mynn SCYTHE_TC_MYNN_EDMF=1 \
SCYTHE_TC_OUTDIR=$PWD/tc/output/tc_s10_24h_mynn_edmf \
julia --project=. tc/tc_run_axisym.jl 86400 --csv
```

## Output

Each nest's `<t>.nc` is now the COMPREHENSIVE NetCDF the model writes directly
(`options[:output_formats]` default `[:netcdf]`, `src/netcdf_output.jl`): primes, totals,
derived thermodynamics, and — when the run carries them — the BL group (`K_m, K_h,
mynn_*`) and radiation group, all in the SAME file (global attr `physics_groups` lists
which). The `<t>_mynn_i*.nc` / `<t>_radiation_i*.nc` sidecars are now OPT-IN
(`options[:mynn_output]` / `options[:radiation_output]`, both default `false`) — most runs
no longer produce them. `tc/tc_movie.jl` reads the comprehensive file directly; no
postprocessing step. `tc/tc_postprocess.jl` is now LEGACY: it is for runs made BEFORE this
stage (raw `<t>.nc` with no `scythe_file_kind` attribute) and REFUSES to touch a
comprehensive file.

Postprocess / views for an OLD run only: `julia --project=. tc/tc_postprocess.jl --indir
tc/output/<run>` merges the MYNN sidecar (`<t>_mynn_i*.nc`: K_m, K_h, e, l, the budget
columns P_s, P_b, ε, the plume sums, pblh, ust, 1/L) into a derived NetCDF;
`tc/tc_movie.jl --bl` then draws the PBL height and the K_h cross-section from that derived
file exactly as it would from a comprehensive one. For a NEW run, skip straight to
`tc/tc_movie.jl --bl` — no postprocessing step, and no `--mynn_output` needed unless the
sidecar itself (rather than the merged fields) is wanted. Assess as `S10_BL_ASSESSMENT.md`
did (hourly nest-1 table, census lines, cost), n = 1 in a ±10 m/s oscillating regime is not
quotable without a repeat.
