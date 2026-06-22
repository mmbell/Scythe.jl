# Phase 3 plan — Conserve water mass via moisture partial densities

## Context

Phase 1 made the **dry-air** mass exactly conserved (carry linear `ρ_d`, not `ξ`). With
that fixed, the leading conservation residual in the moist benchmark moved to the
**water** mass: `water_mass_drift_pct` went from 2×10⁻⁵ % (`ξ` run) to 3.4×10⁻³ %
(`ρ_d` run), and is now essentially the whole `mass_drift_pct` (6.6×10⁻⁵ %).

Cause — same convexity issue, one level down. Moisture is advected as **mixing ratios**
(`mu = q_v·10⁵`, `mu_c`, `mu_r`), so the `l_q` spline smoothing conserves `∫mu` (a
linear function of `q`) but **not** `∫ρ_d q_v` (the physical water mass), because `ρ_d`
varies across the smoothing stencil. The conserved quantity is the **partial density**
`ρ_v = ρ_d q_v`, not the intensive `q_v`.

## Fix: carry partial densities

Make the prognostic moisture variables the partial densities
`ρ_v = ρ_d q_v`, `ρ_c = ρ_d q_c`, `ρ_r = ρ_d q_r` (extensive, linear in mass), so the
smoothing preserves each `∫ρ_x`. Their continuity is the same advective product-rule
form already used for `ρ_d`:

```
∂ρ_v/∂t = −v·∇ρ_v − ρ_v ∇·v + ρ_d(q̇_evap − q̇_cond)     (and analogues for ρ_c, ρ_r)
```

Total-water mass `∫(ρ_v+ρ_c+ρ_r)` is then conserved by smoothing, and the
condensation source terms cancel in the sum exactly as the mixing-ratio versions do
today (`primitive_equations.jl` MU/Q_C/Q_R forcings).

## Implementation steps

1. **New equation set `primitive_equation_XZ_rhod_pd`** (extends the Phase-1 set):
   slots `rho_v`, `rho_c`, `rho_r` replace `mu`, `mu_c`, `mu_r`. Continuity in
   product-rule form (mirror the `ρ_d` block). Reference: add `rho_vbar = ρ̄_d q̄_v`
   (and zero `rho_cbar`/`rho_rbar` for the dry-reference convention, or the saturated
   base cloud as `rho_cbar`).

2. **Thermodynamics (`src/thermodynamics.jl`):** reconstruct `q_v = ρ_v/ρ_d` wherever
   the tuple currently uses `mu`; the pressure-gradient `P_qv` term uses the chain rule
   `∂q_v/∂x = (ρ_v_x − q_v ρ_d_x)/ρ_d`. Add a `thermodynamic_tuple` variant taking
   `(s, rho_d, rho_v)`.

3. **Microphysics (`src/microphysics.jl`):** condensation/evaporation move mass between
   `ρ_v` and `ρ_c` (`Δρ_v = −Δρ_c`), conserving `ρ_v+ρ_c+ρ_r`. Update
   `condensation_adjustment_new_rhod` → a partial-density variant; saturation variable
   `mu_sat` likewise reconstructs `sat_ratio` from `q_v = ρ_v/ρ_d`.

4. **ICs / benchmark / diagnostics:** write `ρ_v′`, `ρ_c`, `ρ_r` initial fields; add a
   benchmark stage; diagnostics read partial densities directly (`water_mass = ∫(ρ_v+ρ_c+ρ_r)`).

## Lighter alternatives (if full partial-density rework is deferred)

- **Smooth `ρ_d q` instead of `q`** in the transform for the moisture slots only
  (apply the spline projection to the partial density formed on the fly). Smaller code
  surface but couples the transform to a second field — evaluate feasibility against
  Springsteel's per-variable transform.
- **Mass borrower/fixer** post-step to redistribute the `∫ρ_d q_t` residual. Cheap but
  non-local and not structurally clean; acceptable only as a stopgap.

## Verification

- `water_mass_drift_pct` → machine precision; `mass_drift_pct` (total) → machine
  precision (dry already exact from Phase 1).
- Dynamics (min_w, supersat, θ_e′) unchanged vs the Phase-1 `ρ_d` run.
- Total-water uniformity: `q_t` stays at its initial constant to round-off in the
  reversible no-precip benchmark.

## Note

This is a larger change than Phase 1 (touches every moisture equation, the condensation
microphysics, the saturation variable, and `q_v` reconstruction in the thermodynamics).
Sequence it after Phase 2, or independently of it — the two are orthogonal (acoustics vs
moisture). The dry-air conservation win from Phase 1 stands regardless.
