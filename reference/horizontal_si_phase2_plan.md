# Horizontal SI, Phases 2–5: self-contained plan (delta-form Douglas–Gunn first)

Plan handoff, 2026-07-17, for a FRESH session. Approved direction (user, 2026-07-17):
attempt the **delta-form Douglas–Gunn (DG) split**; if it fails the measured gates,
**scrap the ADI composition and implement variant 1** (the vertical-normal-mode exact
3-D solve) — a major overhaul, which is why the DG attempt gets one well-scoped
session first. Nothing below requires re-derivation: read, in order,
`reference/horizontal_acoustic_si_handoff.md` (the original assessment),
`reference/horizontal_si_phase2_findings.md` (Phase-1/2 measurements: why the naive
sequential ADI is rejected, the cross-term algebra, why the lagged compensation is
unstable, and the exact DG form), and `tc/SI_VERTICAL_CEILING.md` +
`tc/SI_CONVECTIVE_CEILING.md` (the staging-discipline and state-dependence lessons).

## State you inherit (all committed on `development`)

- Phase-1 XZ prototype WORKS structurally: `src/horizontal_si.jl` (per-physical-level
  u-form Helmholtz sweep on the merged patch B coefficients, operator-consistent
  read/write chains, 6.5 ms/step), `hacdot` history channel, opt-in
  `options[:horizontal_semiimplicit]`, nesting guard, worker-count invariance 9e-12,
  suite 76xx green with flag off.
- Levers: `:hsi_u_history = stored|none`, `:hsi_x_history = stored|fresh`,
  `:hsi_scheme = ai2s|am2`, `:hsi_cross_comp` (lagged compensation — UNSTABLE, off).
  The sweep already returns all five applied increments + the cross-term field G
  (6-plane feed) — DG consumes the same plumbing.
- Harnesses: `model_tests/hsi_ceiling_sweep.jl` (Co_h sweeps, both bases, levers),
  `model_tests/hsi_growth_probe.jl` (single-case histories + field dumps, `--ts=`),
  `model_tests/hsi_distributed_smoke.jl`, `model_tests/step_cost_profile.jl`,
  `benchmarks/bf02_{dry,moist}.jl --stage mc --grid rirk --hsi [--ts-factor F]`.
- Regression gate: `test/test_moist_compressible.jl` "SI horizontal-acoustic ceiling
  removed" (code-default config, Co_h 3).
- The vertical solve now has TWO paths: reference-profile (precomputed `h_matrix`)
  and `options[:state_dependent_si]` (per-column per-step `_assemble_sd_helmholtz`
  with the current-state Pξⁿ — the TC production path). **The DG restructure must
  handle both, or explicitly gate to one and error on the other.**

## Why the naive composition failed (one paragraph)

Sequential (I−νA)⁻¹(I−νB)⁻¹ solves (I−νL+ν²BA)X = S: the uncompensated ν²BA cross
term is O(ts²) per step ⇒ O(ts) cumulative on ALL pressure-coupled modes — measured
as 8–20% resolved-flow damping on the BF02 dry bubble even at Co_h 0.19, plus a slow
non-normal leak (e-fold ~100 s) on oblique noise. Adding +ν²BA·X(lagged) explicitly
is unstable where ν²|BA| ≈ Co_x·Co_z ≳ 1 (measured NaN). The correction must sit
INSIDE the implicit factors ⇒ delta form.

## Phase 2-DG: the delta-form split

Target scheme (ν = 1.25·ts; 0.5·ts on the AM2 first step), per step:

    R    = [AB3 remainder increment] + [AI2* explicit history terms] + ν·L(Xⁿ)
    (I − νB) δ¹ = R          — per-column z-solve, SOLVING FOR AN INCREMENT
    (I − νA) δ  = δ¹         — patch-level x-sweep, delta form
    X^{n+1} = Xⁿ + δ

The splitting error is ν²BA·δ = O(ts³)/step (2nd-order cumulative), and the
correction is regularized by both factors — the stability failure mode of the lagged
compensation cannot occur. ν·L(Xⁿ) = ν·(A+B)(Xⁿ) is a pointwise product on the grid
slots (the same expressions as the remainder staging, opposite sign), available per
column — including the x-legs (u_x, pp_x slots).

Design work items (budget a derivation pass BEFORE coding, in this order):

1. **Pull Ikawa (1988, JMSJ) into ~/Downloads and verify the delta-form ADI-SI
   scheme + its stability claims against the original** (house rule after the DK83
   miscopy). Also re-check Durran & Blossey (2012) for how AI2* off-centering is
   expressed in delta form — the history terms in R must reproduce the AI2* weights
   exactly when A ≡ 0 (the current vertical-only scheme must be recoverable as the
   B-only special case; that is the first unit check).
2. **Restructure `semiimplicit_adjustment_p` to delta form** so its output is
   δ¹ = the z-implicit increment of the FULL R (which now includes ν·A(Xⁿ) and the
   x-history terms). Key discipline items from 2026-07-16 that must survive the
   restructure: the w-leg stored-applied-increment history; fresh histories
   evaluated on the carried state; the l_q fit inside every column transform; the
   state-dependent coefficient path. The elimination algebra (φ-form Helmholtz)
   applies to δ the same way it applied to the state — rederive the boundary rows
   for the increment (homogeneous w-Dirichlet on δ, since w^{n+1} and wⁿ both
   satisfy the BC).
3. **The x-sweep in delta form**: `horizontal_si_correct!` already computes
   increments; it changes from "solve for the state, subtract" to "solve
   (I−νA)δ = δ¹ directly" — the per-level Helmholtz on the δ¹_u field with rhs
   δ¹_u − (Δτ/ρ̄_t)∂x δ¹_p… re-derive the elimination for the delta variables
   (identical structure; the star is δ¹).
4. **Histories under DG**: with the split now 2nd-order consistent, re-run the
   history A/B (stored vs fresh on each channel) — the prediction from the findings
   note is that BOTH become stable once the cross term is gone; prefer whichever
   the sweep says, defaulting to the discipline that mirrors the vertical
   (stored u-leg, fresh others).

### Gates for Phase 2-DG (all must pass, else fall back to variant 1)

- G1 stability: `hsi_ceiling_sweep` decay at Co_h ≤ 4.5 BOTH bases over 600 s; no
  slow leak on the `hsi_growth_probe` (flat to 600 s at Co_h 3); report Co_h 9.
- G2 accuracy: BF02 dry bubble flag-on vs flag-off at `--ts-factor 0.5` and `1.5`:
  extrema (max_w, min_w, max|u|) within 2% at 0.5 and 5% at 1.5. (The naive ADI
  measured 8–20% and 20–55% — these gates are the rejection line the user set,
  metric 2 of the approved plan.)
- G3 regression: full suite green flag-off (bitwise) and flag-on gate updated;
  2-worker invariance ≤ 1e-10.
- G4 combined: sweep case Co_z 9 × Co_h 4.5 (stratified) decays — and one case
  with `:state_dependent_si` on (the TC production vertical path).

If any gate fails after the history A/B round: **stop, do not iterate further** —
write the failure into the findings note and proceed to variant 1 (below).

## Fallback: variant 1 (vertical-normal-mode exact 3-D solve)

Everything except the solve loop is reused (staging, hacdot, feeds, harnesses,
gates). New machinery: precompute the generalized symmetric eigenproblem of the
vertical Galerkin (Stiff, Mass) pair per column basis in `createModelTile`; per
step project the combined rhs onto vertical modes, solve one banded 1-D horizontal
Helmholtz per mode `(1 + Δτ²Pξ̄μ_m − Δτ²Pξ̄∂xx)`, project back. p′-form boundary
rows both directions (inhomogeneous Neumann from the rigid lid: ∂z p′ = φ*/Δτ —
the "session-eating detail" flagged in the original handoff; budget the same A/B
experiments). References: Tanguay, Robert & Laprise (1990, MWR 118) for the 3-D SI
normal-mode structure; Robert (1981). NOTE: variant 1 replaces the VERTICAL solve
too — reconcile with `:state_dependent_si` (a state-dependent vertical operator
has state-dependent eigenmodes; either freeze modes at the reference and carry the
deviation explicitly — reintroducing the convective ceiling — or re-eigen-solve on
a cadence; this tension is a real design issue to raise before implementing).

## Phases 3–5 (unchanged from the approved plan, gated on Phase 2 passing)

- **Phase 3 axisym**: radial metric operator (u-form vector-Laplacian vs p′-form
  scalar Laplacian — decide by fit-chain consistency + a resting axisym sweep);
  `mc_linear_div!` axisym method (`u_x + u/r`); axis row (u Dirichlet exists) and
  outer wall; gate: axisym sweep decay at Co_h 4.5 stratified.
- **Phase 4 RLR**: p′-form per (level, azimuthal wavenumber n) radial solves
  `(I − Δτ²Pξ̄[(1/r)∂r(r∂r·) − n²/r²])p̂′_n` over the contiguous b_iDim blocks of
  the RLR spectral layout; v-leg staging (slot 9: `+pp_l/(r·ρ̄_t)`; `v_l/r` in the
  linear divergence); per-n axis regularity from the existing radial bases;
  stored-increment leg = p′ (A/B-verify). Gate: axisym/RLR-n0 equivalence
  (`model_tests/nested_rlr_equivalence.jl` pattern) + RLR sweep.
- **Phase 5 nesting + defaults**: sweep in `run_nested_patch` AFTER the parent
  payload application; R3X interfaces freeze-parent-value (δu = 0) first (collar
  margins are 12–24× — measured, `collar_cadence_check`); NEST_TS retune; flip the
  flag default, re-baseline every mc regression, remove the flag (explicit-false
  errors, like the vertical), update `warn_timestep_stability` and docs.

## Practical notes for the fresh session

- Production TC needs only Co_h ≲ 1.1 today, so there is no schedule pressure on
  this work; correctness > speed of delivery (user's standing preference).
- The per-session discipline that paid off: measure every variant on the sweeps
  before believing it; test stratified AND finite-amplitude states; keep every
  experiment behind an option so flag-off stays bitwise; `git add` specific files
  only (model_tests/ contains large untracked user files).
- Cost budget: the naive sweep is 9% of a step; DG adds one pointwise ν·L(Xⁿ)
  evaluation and swaps solve targets — expect no material change. Variant 1's modal
  projections are the only cost risk (measure with `step_cost_profile.jl`).
