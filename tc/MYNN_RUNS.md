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
`:mynn_fidelity`, `:mynn_K_max`, `:mynn_output`) take their defaults; add them to
`bl_options` in `tc_init.jl` when a sweep needs them. The surface layer is shared with Louis:
`options[:sfc_z0] = :komori (default) | :gfdl_v7 | :charnock` and `options[:sfc_stability]`
(Monin–Obukhov; default false) — also not yet exposed as TC env knobs.

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
`tc/output/tc_bhyp_24h`), radiation as in `tc/RADIATION_RUNS.md` if wanted, committed code:

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

Postprocess / views: `julia --project=. tc/tc_postprocess.jl --indir tc/output/<run>` merges
the MYNN sidecar (`<t>_mynn_i*.nc`: K_m, K_h, e, l, the budget columns P_s, P_b, ε, the
plume sums, pblh, ust, 1/L) into the derived NetCDF; `tc/tc_movie.jl --bl` draws the PBL
height and the K_h cross-section. Assess as `S10_BL_ASSESSMENT.md` did (hourly nest-1 table,
census lines, cost), n = 1 in a ±10 m/s oscillating regime is not quotable without a repeat.
