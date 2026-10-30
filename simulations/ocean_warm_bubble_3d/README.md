# Ocean warm bubble, 3-D

The ocean warm bubble case ([`../ocean_warm_bubble/`](../ocean_warm_bubble/README.md)) on
the 3-D Cartesian box: `moist_compressible_RRR`, B-splines in x, y and z.

| | |
|---|---|
| domain | 300 km × 300 km × 25 km |
| cells | 50 × 50 × 50 (6 km × 6 km × 500 m, the TC coarse spacing) |
| columns | 22,500 (150 × 150 mish columns × 150 levels) |
| lateral BC | FREE-SLIP walls on all four faces (the box's `walls` rule): the normal velocity Dirichlet (`u` on the x faces, `v` on the y faces), `w` and every other variable Neumann |
| initial condition | a round RH-preserving 3 K bubble, `xc = yc = 150 km`, `xr = 30 km` in x and y, `zc = 500 m`, `zr = 3 km` |
| environment | f-plane at 15°N (the 2-D case's latitude, which the box does feel), SST 302.65 K |
| physics | the production defaults: ISHMAEL ice, MYNN, two-moment rain, surface fluxes and drag; radiation off |
| run | 2 h as two one-hour segments (`run.segment_length = 3600`), output every 300 s, comprehensive NetCDF |

`scythe_config.toml` is the whole case. Domain, grid, sounding, initial condition and
environment are spelled out in full; physics, numerics and diagnostics are delta-only
against the `RunConfig` defaults.

## Running

This is a **cluster** run. `scythe_owb3d.sbatch` uses 11 workers, one per 20-core node,
each holding one x tile of the box:

```sh
sbatch simulations/ocean_warm_bubble_3d/scythe_owb3d.sbatch                  # fresh: output/owb3d_<jobid>
sbatch simulations/ocean_warm_bubble_3d/scythe_owb3d.sbatch --resume \
       --set output.dir=$PWD/simulations/ocean_warm_bubble_3d/output/owb3d_12345
```

- The 50 x cells split 4–5 per tile.
- The tiles exchange their 3-row spectral halos (every j and k mode block) once per step.
- The y and z axes are never decomposed.

The Mac development size of the same case is
[`../ocean_warm_bubble/examples/rrr_120km.toml`](../ocean_warm_bubble/examples/rrr_120km.toml):
a 120 km box, 20 × 20 × 50 cells, 1 h. Run it with `../ocean_warm_bubble/run.jl --config ...`.

## Viewing

```sh
julia --project=. simulations/ocean_warm_bubble/examples/owb3d_movie.jl --root <run root>
```

The movie script reads the comprehensive `<t>.nc` files and writes `owb3d.mp4`. Each frame
has two panels:
- a horizontal slice at 1 km: reflectivity, `w`, and the surface rain rate;
- a vertical x–z slice through the bubble centre.

Physics groups (MYNN, surface) are not yet written into the 3-D comprehensive file. Set
`output.sidecars = true` for the per-tile sidecars.

## Acceptance

The 3-D code is verified on small boxes. All three tests run under the default physics,
f = 0, for 40 steps at 0.5 s (`test/test_3d_rrr_timestepping.jl`):

| check | gate | measured |
|---|---|---|
| y-invariant box started from the XZ slice's state vs the XZ run (24 × 12 × 4 km) | ≤ 1e-9 scaled | 8.0e-14 |
| centred bubble on a square box, x ↔ y with u ↔ v (24 × 24 × 4 km) | ≤ 1e-10 | 4.3e-14 |
| 2 workers vs 1 worker | ≤ 1e-12 | 4.8e-14 |

The case itself is compared against the RLR cylinder at the Mac size:
- `rrr_120km.toml` vs `rlr_60km.toml`;
- compare script `../ocean_warm_bubble/examples/rrr_rlr_compare.jl`;
- region: within R/2 = 30 km of the bubble centre;
- quantities: max w to time t, max T′ at t, and rain onset.

These are solution-level gates. They are not roundoff gates, because the two grids
discretize the same bubble differently. With ice on, ISHMAEL amplifies roundoff after
glaciation (~25 min; reference/…/r1_rlr/README.md), so the 1 h numbers describe the
chaotic solution.

The acceptance numbers are **pending the 3 km pair** (close-out 2026-10-29, Stream C2,
`reference/session_2026-10_scratch/closeout_2026-10-27/r2_rrr/STATUS.md`). What is known:

- At the Mac size (6 km cells, `xr = 20 km`: 3.3 cells per bubble half-width) the box's bubble
  is weaker than the cylinder's (20 min: max w 0.45 vs 1.78 m/s, max T′ 1.27 vs 2.10 K; the box
  never passes 1 m/s in the hour while the cylinder convects to 24 m/s). This is resolution, not
  a defect: on the dry 300 s convergence study (`xr = 10 km`) the box-to-axisymmetric gap in
  w at 120 s is 60 % at 6 km, 24 % at 3 km and 1.0 % at 1.5 km (6.7 cells per half-width),
  the RLR cylinder equals the axisymmetric run exactly, and with the horizontal spline filter
  off both grids hold the bubble's T′ in the first 30 s where the filtered box loses half of it.
  The box filters a near-grid-scale round bubble along x and y, the cylinder only along r.
- Proposed gate (to be seeded by the first 3 km pair): ≥ 6.7 cells per bubble half-width, then
  at 20 min (pre-glaciation) max w, max T′ and rain onset within 2 % of the RLR control inside
  r < R/2; the 1 h numbers are informational (5 %), the solution being chaotic after glaciation.
- The 6 km pair stays as the Mac smoke (walls clean all hour: box-edge T′ ≤ 0.056 K, edge p′
  within −4.2 … +3.0 Pa). This production case keeps the plan's 6 km spacing with `xr = 30 km`
  (5 cells per half-width); whether to run it at 3 km (100 × 100 × 50, 90,000 columns) is the
  author's call once the 3 km pair has been measured.
