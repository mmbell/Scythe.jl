# Scythe Benchmarks

Reproducible, self-contained benchmark scripts that serve as both numerical
correctness checks and performance regression tests. Each script generates its
own reference sounding and initial conditions (nothing the model reads is a
hand-edited artifact), runs the distributed model, verifies scalar diagnostics
against published values, compares output fields against committed reference
data, and appends a provenance record with wall-clock timing.

## Running

```sh
julia --project=. benchmarks/straka93.jl   --mode quick --stage legacy
julia --project=. benchmarks/bf02_dry.jl   --mode full  --stage legacy --plot
julia --project=. benchmarks/bf02_moist.jl --mode quick --stage legacy
```

| Flag | Values | Meaning |
|------|--------|---------|
| `--mode` | `quick` (default) \| `full` | quick: coarse grid for routine regression; full: paper-grade resolution |
| `--stage` | `legacy` (default) \| `pe` | legacy: original equation sets (`Euler_test`/`BF02_test`); pe: `primitive_equation_XZ` (Stage 2) |
| `--workers` | integer (default 2) | distributed worker count (recorded; results are worker-count independent to ~1e-10) |
| `--update-reference` | | write the committed regression reference from this run |
| `--plot` | | save final-time contour figures to the output directory |

Exit code 0 = all targets and the regression comparison passed (CI-friendly).
Run records append to `results/<case>.jsonl` (gitignored) with git SHAs,
timing, worker/thread counts, and all diagnostics.

## Cases

| Case | Equation set (legacy) | Domain | Full / quick cells | Reference |
|------|----------------------|--------|--------------------|-----------|
| `straka93` | `Euler_test` | 25.6 × 6.4 km, K=75 m²/s | 25 m / 100 m | Straka et al. (1993), `reference/Straka.pdf` |
| `bf02_dry` | `Euler_test` | 20 × 10 km, K=0 | 100 m / 200 m | Bryan & Fritsch (2002) Fig. 1, `reference/bryan_fritsch_mwr2002.pdf` |
| `bf02_moist` | `BF02_test` | 20 × 10 km, K=0, θ_e=320 K, r_t=0.020 | 100 m / 200 m | Bryan & Fritsch (2002) Fig. 3 |
| `o01_rainfall` | `moist_compressible_XZ` (mc only) | 150 × 20 km, warm rain, inviscid (spline filter only), humidified Dunion sounding | 500 m / 2 km (Δz 500 m both) | Ooyama (2001) Fig. 6 magnitudes, `reference/ooyama_jas2001.pdf` |

Published target values and tolerances live in
`reference_data/expected_values.jl` with their sources. Full-mode tolerances
are wider than the inter-reference-model spread reported in the papers
(Scythe is spectral; the references are finite-difference codes) — they catch
"physics broke". The committed reference comparison (rel-L2 ≤ 1e-6 per
variable) catches "the code changed the answer".

### Known physics differences vs the published benchmarks

- **bf02_moist**: `BF02_test` carries a prognostic supersaturation variable
  (`qss`) rather than the paper's instantaneous saturation adjustment.
  Allowing supersaturation production/consumption softens latent heat release,
  which lowers the w extrema relative to Fig. 3 while avoiding the stiffness
  of strict adjustment. The structure comparison (`--plot`: rotors, arch,
  thermal top near 8 km) is the primary check.
- **Spectral vs finite-difference amplitudes**: with K=0 the spectral method
  retains dispersive ripples (±0.1–0.4 K in θ′/θ_e′) that the papers'
  odd-order FD advection schemes implicitly diffuse, so the BF02 cases
  over/undershoot the strict Fig. 1/3 extrema. Full-resolution legacy results
  (2026-06, baseline in `reference_data/*/full_legacy_diagnostics.csv`):

  | Diagnostic | bf02_dry (paper) | bf02_moist (paper) |
  |------------|------------------|--------------------|
  | max θ′ / θ_e′ | 2.26 (2.07) | 5.57 (4.10) |
  | min θ′ / θ_e′ | −0.39 (−0.14) | −0.95 (−0.31) |
  | max w | 13.0 (14.5) | 13.0 (15.7) |
  | min w | −8.50 (−8.58) ✓ | −12.1 (−9.93) |

  Structure (rotors, arch, thermal top ≈ 8 km) matches the unapproximated
  reference solution; the ~10–17% updraft-maximum deficit is an open question
  for the Stage 2 equation-set comparison. Conservation in the dry case is at
  the paper's own level (mass 3.5e-6 %, energy 2.0e-4 %, entropy 1.5e-4 %).

### Stage 2 (primitive_equation_XZ) findings (2026-06, quick resolution)

- The PE set runs all three benchmarks in **reduced form**: the benchmark
  configs set `options[:precipitation] = false` (BF02 spec: no fallout; the
  saturated base state would otherwise autoconvert to rain domain-wide) and
  `options[:vertical_mixing] = false` (no turbulence in the benchmark specs;
  the Louis shear mixing is also explicitly unstable where arch shear meets
  the ~10 m Chebyshev spacing at the lid).
- **Dry equivalence**: with the reduced form the PE solution matches
  `Euler_test` to rel-L2 ~1e-5 across all fields — the comprehensive set
  collapses to the dry Euler equations
  (`julia benchmarks/compare_stages.jl bf02_dry quick`).
- **Moist**: the PE's explicit droplet-growth condensation recovers the
  paper's updraft (max w 15.0 vs 15.7 at 200 m) where the legacy qss
  relaxation softened it to 10.5. θ_e′ extrema overshoot more (5.9 vs 4.1)
  with the stronger arch — the Chebyshev overshoot question (see
  FUTURE_WORK.md).
- **Straka PE** passes the same targets as legacy with a sharper front
  (max u 42 vs legacy 37 at 100 m), but with notably larger conservation
  drifts (mass 0.11 %, total entropy −0.89 % vs legacy's ~1e-4 %) traced to
  the implicit `Kvdiff` vertical diffusion path — under investigation.
- Conservation diagnostics (informational): total mass, BF02 eq.-29 total
  energy, and total entropy including the condensate term
  `q_l·Cl·ln(T/T₀)` are spectrally integrated over the domain; percent drift
  between t=0 and the final time is reported and recorded. The prognostic
  entropy (dry + vapor) has condensation sources by construction; the total
  should be conserved.

## Layout

```
benchmarks/
├── common/            harness (CLI, verification, timing, JSONL), diagnostics, plots
├── reference_data/    committed: expected_values.jl + per-case regression CSVs
├── output/            gitignored: per-run model output, figures, generated soundings
├── results/           gitignored: <case>.jsonl run records
└── <case>.jl          self-contained scripts
```

To accept a new result as the regression reference (after intentional physics
changes), re-run with `--update-reference` and commit the CSV under
`reference_data/`.
