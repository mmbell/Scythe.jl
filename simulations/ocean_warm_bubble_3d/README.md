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

The acceptance numbers come from the box/cylinder pairs of the close-out (2026-10-29/30,
Stream C2, `reference/session_2026-10_scratch/closeout_2026-10-27/r2_rrr/STATUS.md`): the Mac
pair at 6 km (`rrr_120km.toml` vs `rlr_60km.toml` as committed, 1 h) and the same pair at 3 km
(`--set grid.num_cells=[40] --set grid.num_cells_j=[40]` on the box, `--set grid.num_cells=[20]`
on the cylinder, 20 min), both with the default physics and free-slip walls. Max w is the run
maximum up to t, max T′ the value at t, both inside r < R/2 = 30 km of the bubble centre; the row
L2 compares the box's x-axis row with the cylinder's ring mean.

| spacing (cells per half-width) | t | max w box / cyl [m/s] | gap | max T′ box / cyl [K] | gap | row L2 w / T′ |
|---|---|---|---|---|---|---|
| 6 km (3.3) | 300 s | 0.427 / 0.676 | 37 % | 1.79 / 2.23 | 20 % | 0.33 / — |
| 6 km (3.3) | 600 s | 0.450 / 0.853 | 47 % | 1.56 / 2.13 | 27 % | 0.44 / — |
| 6 km (3.3) | 900 s | 0.423 / 1.209 | 65 % | 1.41 / 1.63 | 13 % | 0.65 / — |
| 6 km (3.3) | 1200 s | 0.447 / 1.778 | 75 % | 1.27 / 2.10 | 40 % | 0.79 / — |
| 6 km (3.3) | 3600 s | 0.97 / 24.2 | 96 % | — | — | — |
| 3 km (6.7) | 300 s | 1.220 / 1.289 | 5.3 % | 2.678 / 2.693 | 0.6 % | 0.043 / 0.010 |
| 3 km (6.7) | 600 s | 2.465 / 3.191 | 23 % | 2.537 / 2.778 | 8.7 % | 0.247 / 0.121 |
| 3 km (6.7) | 900 s | 5.271 / 9.130 | 42 % | 3.750 / 4.992 | 25 % | 0.463 / 0.427 |
| 3 km (6.7) | 1200 s | 9.835 / 21.33 | 54 % | 5.685 / 9.836 | 42 % | 0.644 / 0.901 |

Rain onset: none by 1200 s at 6 km in either run; 1200 s in both runs at 3 km. The box's edges
stay clean in every run (3 km: edge T′ ≤ 0.036 K, edge p′ ≤ 10.1 Pa; 6 km hour: edge T′ ≤ 0.056 K,
edge p′ within −4.2 … +3.0 Pa). Mac cost at 3 km: 13 s of wall per simulated second for either
grid (box 2 workers × 5 threads, cylinder 1 worker × 4 threads, run concurrently); the 6 km box hour
was 3.2 s per simulated second.

What the numbers say:

- **The early (pre-convective) gap is resolution.** Halving the spacing takes the 300 s gap from
  37 % to 5.3 % in w and from 20 % to 0.6 % in T′. The dry 300 s convergence study (`xr = 10 km`;
  box, cylinder and axisymmetric at 6, 3 and 1.5 km) puts the box-to-axisymmetric w gap at 120 s
  at 60 %, 24 % and 1.0 %, every output through 180 s within 1.3 % at 1.5 km, and the RLR cylinder
  equal to the axisymmetric run exactly. A defect in the y-terms would leave a fixed fraction; it
  does not. The mechanism is the horizontal spline filter (l_q 2.0 in every direction on both
  grids): the box filters a near-grid-scale round bubble along x and y, the cylinder along r only;
  with the filter off both grids hold T′ in the first 30 s.
- **The convective phase does not close at 3 km, and the cylinder is not the converged answer
  there.** From 600 s the cylinder's updraft accelerates on its axis (3.2 → 9.1 → 21.3 m/s) and the
  box's grows about half as fast (2.5 → 5.3 → 9.8 m/s). The dry study shows the same split after
  200 s, where the axisymmetric run focuses an axis jet that is itself unconverged (2.0 m/s at
  3 km, 8.7 m/s at 1.5 km at 300 s). So a tight 20-min gate is not an honest gate at 3 km, nor at
  1.5 km; it would need a resolution series run until the cylinder's own max w stops changing.
- **The 2 % gate at 20 min proposed before the 3 km pair was run is therefore NOT the
  acceptance.** The correctness evidence is the three roundoff tests above, the dry 1.5 km
  agreement and the 300 s pair agreement at ≥ 6.7 cells per half-width.

Acceptance, as seeded by the 3 km pair (the author may tighten or replace it):

- ≥ 6.7 cells per bubble half-width (`xr / Δx`), and at 300 s max w and max T′ within 10 % of the
  RLR control inside r < R/2 (measured 5.3 % and 0.6 %; the 6 km pair is at 37 % and 20 %):
  `rrr_rlr_compare.jl <box> <cyl> --gate early_t=300 --gate early_rel=0.1 --strict-until 300`.
- The 20 min and 1 h numbers are recorded as the measured convective-phase difference, not gated.
- Pending, cluster: a 1.5 km moist pair (8× the 3 km cost; `stream_f/LAUNCH.md` §4c) extends the
  series if a converged convective-phase comparison is wanted.

Production spacing: this case keeps the plan's 6 km with `xr = 30 km` (5 cells per half-width),
between the Mac smoke (3.3) and the acceptance spacing (6.7); the faithful control is
`../ocean_warm_bubble/examples/rlr_150km.toml`. Whether to run it at 3 km (100 × 100 × 50, 90,000
columns, ~4× the cost) is the author's call.
