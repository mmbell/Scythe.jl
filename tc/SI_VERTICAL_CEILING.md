# The vertical-acoustic timestep ceiling of the mc semi-implicit solver

Handoff note, 2026-07-16. Status: **diagnosed and measured, mitigated by
timestep choice (ts = 0.3 s at 250-m cells), not yet fixed.** This documents
why the semi-implicit solver is *not* unconditionally stable for vertical
acoustics, the experimental evidence, and the candidate fixes.

## Symptom

With `options[:semiimplicit] => true`, mc runs blow up above a **vertical**
acoustic Courant number of ≈ 2.1–2.4 on the minimum Gauss spacing
(`Co = c·ts/dz_min`, `dz_min ≈ 0.2254·dz_cell` for 3-point Gauss). At 250-m
vertical cells (`dz_min = 56.35 m`) that is `ts ≈ 0.35–0.40 s`; at the O01
500-m cells it reproduces the previously observed 0.6-s-stable / 1.2-s-unstable
behavior. The nested TC configuration at ts = [0.75, 1.5, 3.0] died within
minutes (`rho_d → −164` via the rain fall-speed sqrt). Since all nest patches
share the vertical grid, the ceiling binds every patch equally — per-nest ts
scaling with DX buys nothing, and the nesting speedup is purely column count.

## The scheme as implemented

`semiimplicit_adjustment_p` (src/moist_compressible.jl:976):

- The **reference-linear vertical acoustic pair** `∂φ/∂t = −∂z p′`,
  `∂p′/∂t = −c̄² ∂z φ` (φ = ρ̄_t w, reference coefficients `Pxi_bar`, ρ̄
  profiles) is staged into `impdot` in the driver and integrated with the
  off-centered **AI2\*** weights `+1.25 L^{n+1} − 1.0 L^n + 0.75 L^{n−1}`
  via the pre-factorized Helmholtz solve `(I − Δτ² c̄² ∂zz) φ`, Δτ = 1.25·ts.
- Everything else (advection, buoyancy/gravity modes, horizontal acoustics,
  physics) runs AB3.
- The adjustment works subtractively: the predictor is advanced with AB3 of
  the FULL tendency including `impdot`, then the AB3 treatment of `impdot`
  is subtracted and the AI2\* implicit terms added (lines 1010–1028).
- The density and energy slots are slaved to the flux in flux form with
  reference coefficients.

The AI2\*+AB3 pairing is formally fine: the amplification factor of AI2\* for
a pure oscillation `L = iω` satisfies |A| ≤ 1 for **all** ω·Δt (checked
analytically; |A| → 0.775 as ω·Δt → ∞). The instability is not in the
implicit scheme.

## Root cause

**The subtract-AB3 / add-implicit algebra assumes the explicit and implicit
halves apply the SAME discrete operator, and they do not:**

- `impdot` is built in `mc_driver!` in **pointwise product-rule form on the
  tile's derivative slots** (`−Pξ̄(ρ̄_t w_z + ρ̄_t,z w)` etc.). This is
  deliberate — see the comment at src/moist_compressible.jl:664: a column
  refit reapplies the spectral filter and lets ρ_d and ρ_t drift apart in
  dry air.
- The adjustment's RHS (`∂z p′*`), the recovered `p′` update, and the
  Helmholtz operator itself all go through **refit column transforms that DO
  apply the spline filter**, plus the spline-basis ∂zz with its boundary
  rows.

For well-resolved vertical modes the two agree and the cancellation is clean.
For the **highest vertical wavenumbers** — exactly where the filter acts and
where the boundary rows of the spline solve differ from the pointwise form —
the subtracted AB3 term is not the operator the implicit solve adds back. The
residual is a fraction of the grid-scale acoustic operator left effectively
explicit under AB3, whose imaginary-axis stability limit is ω·Δt ≈ 0.72.
That converts "unconditional" into a Courant-type ceiling on the residual;
the measured ceiling (Co ≈ 2.1–2.4 vs the pure-explicit ≈ 0.7) implies the
residual fraction at the grid scale is roughly 30%. The SI still buys ~3×
over pure explicit.

## Evidence

Resting, **bone-dry isothermal (statically stable) base**, all physics off
(precipitation off, all K = 0, sponge off), broadband 1e-4 w-noise seed,
XZ 8×100 cells (25.6 km × 25 km, dz = 250 m), 600 s integration:

| ts [s] | Co (dz_min) | result |
|--------|-------------|--------------------------------------|
| 0.30   | 1.81        | seed decays ×40                       |
| 0.35   | 2.11        | seed decays ×50                       |
| 0.40   | 2.41        | grows 1e-4 → 1 m/s, e-fold ≈ 65 s     |
| 0.50   | 3.0         | growth ×87 per 30 s                   |
| 0.75   | 4.5         | NaN in ~50 s                          |

The growing mode is **grid-scale in z and localized at the TOP boundary**
(w swinging sign point-to-point over the last ~5 Gauss points below the lid)
— consistent with the filter/boundary-row mismatch and impossible physically
on this base.

Two pitfalls when reproducing:
- On the **humidified Dunion sounding** the same seed grows slowly even at
  ts = 0.3 — that is *physical* conditional instability of the profile, not
  numerics. Use the dry isothermal base to isolate the numerical mode.
- Run ≥ 600 s; 90-s runs miss the slow marginal growth near the ceiling.

Reproduction recipe: build a resting XZ mc tile (pattern of
`test/test_louis_bl.jl`'s builder) with an analytic isothermal reference
(`write_exact_ref_mc` with `p = p₀ exp(−gz/RT₀)`), seed `w` with uniform
noise, and step with `advance_column` + `calcTendency` + `gridTransform!`,
tracking `max|w|`.

## Candidate fixes (in rough order of promise)

1. **Divergence damping on the vertical acoustic mode** (Skamarock & Klemp
   1992-style): add a small damper `∝ ∂z(δ·D)` to the w (or φ) equation
   targeting 3D/vertical divergence. Cheap, standard in compressible NWP,
   acts exactly on the grid-scale acoustic residual, does not touch the SI
   structure. Would likely lift the ceiling to Co ~ 4–6. Must be added
   consistently to E_t (KE-sink convention, like the sponge/FRIC_KE).
2. **Make the two halves discretely consistent**: stage `impdot` through the
   SAME refit+filtered column derivatives the adjustment uses. The :664
   comment explains the historical cost — both densities (ρ_d′, ρ_t′) must
   then take the same refit so they cannot drift apart in dry air; solvable
   but delicate, and it changes answers at the filter level (re-baseline
   regressions).
3. **Build the Helmholtz operator from the pointwise discrete ∂z** (the
   inverse of 2): assemble `(I − Δτ²c̄²∂zz)` in mish space with the exact
   product-rule/derivative-slot operators. Bigger surgery to the vertical
   solve machinery.
4. **Inspect the top-boundary rows first** regardless: the unstable mode is
   top-localized, so the w-Dirichlet boundary row of the spline solve vs the
   pointwise staging near the lid is probably the single largest mismatch —
   a targeted fix there might buy a useful margin alone.

## Current mitigation

`tc/tc_params.jl` sets `NEST_TS = [0.3, 0.3, 0.3]` (Co 1.81, ~15% margin
below the measured ceiling). ts = 0.35 is stable on the resting base but the
margin is too thin for a 5-day production run. The advisory warning heuristic
in `src/reference_state.jl` (~:191) still uses a conservative Co 0.5 target;
it could be updated to reflect the measured SI ceiling.
