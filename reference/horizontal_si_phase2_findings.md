# Horizontal SI Phase 2 findings — the ADI cross term is not benign (decision point)

Working note, 2026-07-17. Status: **Phase-1 prototype works and is committed (opt-in);
Phase-2 measurements trip the plan's ADI-rejection metric 2 — escalation decision needed.**
Companion to the approved plan and `reference/horizontal_acoustic_si_handoff.md`.

## What was measured

All on the XZ prototype (`src/horizontal_si.jl`), resting-base sweeps
(`model_tests/hsi_ceiling_sweep.jl`, isothermal + stratified, 600 s), growth probes
(`model_tests/hsi_growth_probe.jl`), and the BF02 dry/moist quick benchmarks on RiRk
(`--hsi` harness flag; quick = 200-m cells, ts 0.15 at `--ts-factor 1.5`).

**Stability (resting sweeps), by history variant:**

| u history          | p/ρ/E histories | result |
|--------------------|-----------------|--------|
| stored increment   | fresh (u_x slots) | no hard ceiling to Co_h 18, but a slow leak: e-fold ≈ 90–110 s, ~Courant-independent, smooth oblique mode (λ_z ≈ 6.6 km) |
| stored increment   | stored increments | same leak, same rate — the fresh-vs-applied mismatch is NOT the cause |
| pointwise fresh    | —               | NaN in minutes (the vertical w-leg lesson, reconfirmed horizontally) |
| none (θ = 1 u leg) | fresh           | stable: clean decay Co_h ≤ 3, marginal 4.5 stratified, unstable ≥ 9 |
| AM2 (trapezoidal x-channel) | —      | same envelope as "none" |

Correcting the vertical w history to the final-state tendency made the leak WORSE
(e-fold 50 s): the stored histories are self-consistent (history = applied operator);
do not "fix" them toward analytic tendencies.

**Accuracy (BF02 dry bubble, deterministic, flag-on vs flag-off at the SAME ts):**

| ts-factor (Co_h) | flag-off max_w / min_w / max\|u\| | flag-on (θ=1) | flag-on (all-stored) |
|------------------|-----------------------------------|----------------|----------------------|
| 0.5  (0.19)      | 9.76 / −5.08 / 5.36               | 9.00 / −4.01 / 4.26 | — |
| 1.5  (0.57)      | 10.38 / −5.89 / 5.73              | 8.76 / −2.78 / 4.19 | 8.37 / −2.73 / 3.84 |

The flag-on deficit (8–20% at Co_h 0.19, 20–55% at 0.57) shrinks only LINEARLY with ts
— a first-order cumulative error, present for BOTH history variants, i.e. NOT the θ=1
damping and NOT the leak: it is the **uncompensated ADI cross term**. The sequential
factorization applies (I−νA)⁻¹(I−νB)⁻¹ instead of (I−ν(A+B))⁻¹; the defect ν²AB is
O(ts²) per step, O(ts) cumulative, and acts on ALL pressure-coupled modes — the
resolved bubble circulation, not just grid-scale oblique acoustics. The handoff's
"splitting only over-damps oblique acoustics" argument held for the eliminated scalar
Helmholtz, not for the primitive-variable sequential composition; the same term is the
likely source of the slow leak (it is non-normal — damps some oblique modes, amplifies
others).

The moist BF02 comparisons show the same signature on top of that case's intrinsic
sensitivity. Estimate of the per-step cross-term size for the bubble:
(cΔτ)²·k_x·k_z ≈ 1%/step at ts-factor 1.5 — matches the observed cumulative deficit.

## Verdict against the plan's ADI-rejection metrics

Metric 2 (benchmark drift attributable to the split, present at flag-on small-ts) is
**decisively tripped** — >5% resolved-field drift at every tested ts, ∝ ts. The
history-variant A/B space is exhausted. Per the plan, this is the escalation branch.

## Options (in recommended order)

1. **Cross-term-compensated / delta-form ADI (Douglas–Gunn).** Restructure the two
   implicit applications into the standard second-order-consistent split (each sweep
   solves for an increment with the other operator's lagged contribution on the RHS).
   Reuses the entire Phase-1 machinery (per-level solves, staging, increment
   plumbing); the vertical column solve needs its RHS extended by the lagged
   horizontal term (a pointwise product on grid slots — cheap), and the horizontal
   sweep solves the delta form. One derivation + implementation round; the leak and
   the damping should BOTH vanish if the cross term is truly the cause (they share
   the fingerprint). Verify against Ikawa (1988) for the ADI-split SI precedent
   (house rule: pull the original).
2. **Variant 1 — vertical-normal-mode exact 3-D solve** (the plan's designated
   escalation): no splitting at all, unconditionally clean composition. More
   machinery (vertical eigenproblem, p′-form boundary rows in both directions,
   replaces the validated vertical solve). Reuses the staging/histories/harness.
3. Accept θ=1 ("none") for configurations where the damping is tolerable — NOT
   acceptable for the production benchmarks measured here.

Recommendation: attempt (1); it is the smallest change that addresses the measured
mechanism, and (2) remains available unchanged if (1) disappoints.

## State of the tree

Committed and green (suite 7646/7646): opt-in Phase-1 sweep (code defaults
`hsi_u_history="none"`, `hsi_x_history="fresh"` — the stable, over-damping variant),
regression gate at Co_h 3 (code defaults), multi-limit `warn_timestep_stability`,
worker-invariance smoke (9e-12), `--hsi` benchmark flag, A/B levers
(`hsi_u_history="stored"`, `hsi_x_history="stored"`, `hsi_scheme="am2"`), 5-plane
increment feed (all applied increments available to any history scheme — DG will
want them too).
