# Variant 1: unsplit 3-D semi-implicit solve (exact composition) — plan + Courant assessment

Plan of 2026-07-17 (Fable planning session; implementation intended for **Opus** with
hard measured gates and stop rules). Supersedes the "Fallback: variant 1" sketch in
`reference/horizontal_si_phase2_plan.md` where they differ (see Design Resolution).

## Context — the Courant assessment the user asked for

Current commit `fbe249e`, current TC config (`tc/tc_params.jl`: 3/6/12-km nests,
300-m vertical cells → dz_min 67.6 m mish, nest1 dx_min 676.2 m; `NEST_TS
[0.5,0.5,0.5]`; `:state_dependent_si => true` in tc_init.jl; horizontal SI **off**).
The advisor printout (`warn_timestep_stability`, src/reference_state.jl:174) is
`Co = current/target`. The ladder, as max-allowed ts on nest1:

| constraint | ceiling | max ts |
|---|---|---|
| vertical acoustic, CONVECTIVE SI ceiling (δ=0.25) | Co_z ≤ 0.72/δ = 2.88 | **0.573 s ← binding** |
| horizontal acoustic, explicit AB3 | Co_h ≤ 0.5 advisory / ≈0.72 hard | 0.99 / 1.43 s |
| w advection (w_max 25 m/s, on dz_min) | Co ≤ 0.5 | 1.35 s |
| vertical acoustic, resting SI ceiling | Co_z ≈ 9–18 measured | 1.79–3.58 s |

Clarifications settled during planning:

- The printed `horizontal acoustic: Co=0.25/0.5` means *current 0.25 vs advisory
  target 0.5* — horizontal is at 2× margin, NOT binding. The phase-2 plan's
  "Co_h ≲ 1.1" is what the TC *would* need if the vertical ran at its resting
  ceiling (~Co_z 13 → ts ≈ 2.2 s → Co_h ≈ 1.1); no contradiction.
- The binding constraint today is the **convective** vertical ceiling
  (tc/SI_CONVECTIVE_CEILING.md), not the resting one and not horizontal.
- `:state_dependent_si` measurably buys **nothing**: the sd run died at +616 s vs
  +1580 s reference at the fatal ts (experiment table in SI_CONVECTIVE_CEILING.md);
  the amplifier is the fit-chain grid-scale residual, not coefficient locality. It
  costs a per-column Helmholtz refactorization every step. It stays available as an
  opt-in lever but variant 1 will NOT support it (user decision, 2026-07-17).
- The current 12-h run (ts 0.5, Co_z 2.51, margin 1.15× under the δ=0.25 advisory,
  sd on) is inside the validated envelope (E3 completed the crash hour at Co_z ≈
  2.05–4.5 across nests). No action needed on the running job.
- Consequence stated plainly: **variant 1 buys zero TC wall-clock until the
  convective ceiling is fixed** (0.57 s binds below the 0.99-s horizontal advisory).
  Its value is structural: at a future 1-km nest the explicit horizontal limit is
  ts ≤ 0.33 s — below even the convective ceiling — so variant 1 is the
  prerequisite for ANY horizontal refinement, and it is the ε-tolerant solver class
  (state-form, unsplit) that the DG round proved is the only viable composition.
  User chose: variant 1 now; the convective/axis-column investigation
  (SI_CONVECTIVE_CEILING.md probes i–iv) is the follow-on Fable session.

User decisions recorded (2026-07-17): **(1)** plan variant 1 now; **(2)** variant 1
is reference-linearized ONLY — `:state_dependent_si` + variant 1 is an error;
sd_si remains an opt-in lever on the existing vertical-only path; **(3)** Opus
implements from this spec; gate failure ⇒ stop and write findings, do not iterate
(one history-A/B round allowed, matching prior session discipline).

## Design resolution: direct 2-D solve (1b), not the normal-mode eigenproblem (1a)

The handoff's variant-1 sketch (`reference/horizontal_acoustic_si_handoff.md:68`)
precomputes a vertical eigenproblem and solves per-mode 1-D horizontal Helmholtz
problems. **That form requires Pξ̄ constant in z for exact separability** (the
identity term, the Pξ̄-weighted ∂xx term, and the vertical operator cannot be
simultaneously diagonalized when Pξ̄ varies — two distinct mass weights). A
constant/mean Pξ̄ is exactly the SHB78 reference-state configuration that
SI_VERTICAL_CEILING.md ("Second mechanism") measured as fatal: ±20–25% of the
grid-scale vertical operator explicit, NaN at Co_z 9 on the stratified base; the
fix was the LOCAL profile Pξ̄(z), now load-bearing. Do not reintroduce it.

**Chosen form (1b): one unsplit variable-coefficient 2-D (x,z) p′-form Helmholtz
solve on the merged patch B coefficients, with the banded LU factorization
precomputed once at setup.** This is possible precisely because the path is
reference-linearized: the operator is time-independent. Literature precedent:
Ikawa (1988) E-HI-VI (iterated full 2-D implicit solve) — already identified in
the findings note as the variant-1 class. Sizing (axisym TC patch, ~54×88 B
coefficients): N ≈ 4.8k unknowns, bandwidth ≈ 4·b_kDim ≈ 350 → banded LU ≈ 6e8
flop once; back-substitution ≈ 2e6 flop/step — cheap next to the transforms. On
RLR: one factorization per azimuthal wavenumber n (same 2-D (r,z) structure),
parallel over n; memory ≈ N·bw·8 B ≈ 13 MB per n — Stage 0 sizes this for the
production nDim before committing (fallback if prohibitive: 1a with a
warm-reference Pξ̄* ≥ max_z Pξ̄, gated by a stratified-base sweep at Co_z 9).

Why unsplit at all: the Phase-2-DG round proved every two-factor split of this
pair is structurally blocked (ε ≈ 0.1–0.2 chain mismatch ⇒ hard ceiling Co 3–6;
`reference/horizontal_si_phase2_findings.md`), while state-form unsplit solves are
ε-insensitive (von Neumann part 2). Variant 1 is that class.

Note what variant 1 does NOT fix: the convective ceiling (state deviation δ from
the reference is explicit in BOTH directions; δ·Co_z ≲ 0.72 still governs). Gates
below therefore target the resting/stratified ceilings and accuracy, same as the
DG round.

## Stages (each ends at a committed, green state)

### Stage 0 — derivation + sizing pass (NO code changes to src/; one session)

**STATUS 2026-07-17: COMPLETE — GO.** Deliverables in
`reference/exact_si_derivation.md` (A: p′-form elimination, weighted-stiffness
2-D assembly, RHS/recovery, boundary rows, ∂xx→0 vertical-equivalence unit-test
statement, staging audit) and `model_tests/hsi_dg_von_neumann.jl` part 3 (B).
Go/no-go 3(a): the unsplit 2-D solve is ε-insensitive — flat max|G| ≤ 1.003 at
every ε ∈ [0,0.2], Co ≤ 30² — variant 1's premise holds. 3(b): 1a rejected (the
in-mode SHB78 residual is benign rank-1, but inter-mode coupling + slaved legs
give the measured r·Co_z ≲ 0.72; warm-reference max_z Pξ̄* fails Co_z 9 by ~4.5×).
Sizing: b_iDim 53 × b_kDim 87, N 4611, i-fast half-bandwidth 162, LU 2.4e8 flop
once / 4.5e6 flop-per-step / 18 MB; RLR ≤ 2.5 GB all-n per node — feasible, no 1a
fallback. Operator form (REVISED in Stage-0 review — the first draft's
"φ-solve weighted stiffness on p′" solved neither elimination, off by the
commutator [D,P]D ≈ (∂zPξ̄)∂z; script part 3(c) verifies): the recommendation is
the P⁻¹-scaled weighted-MASS form `M0ᵀ(W/Pξ̄)M0 + Δτ²·plain stiffness` — the
EXACT p′ elimination AND self-adjoint (SPD; all BCs natural, Pξ̄-free loads
+Δτ·φ* lid / +Δτ·ρ̄_t u* wall). Consequence for the Stage-1 first unit test:
the A≡0 ≤1e-10 equivalence with the vertical solve holds on the ISOTHERMAL base
(proportional operators + resolvent identity); on the stratified base agreement
is truncation-level only, and the p′-primary stratified vertical ceiling is a
NEW measured quantity (G1's Co_z 9 stratified sweep establishes it). Six
Stage-0 items below all addressed.

1. Derive the p′-form elimination for the XZ reference-linearized acoustic pair
   (u, w/φ, p′ with local Pξ̄(z), ρ̄_t(z)) into a single weak-Galerkin 2-D
   Helmholtz on the patch B-coefficient tensor basis. Reuse the assembly patterns
   that exist: vertical `M1ᵀ(W·Pξ̄)M1` weighted-stiffness (built for the SHB78 fix,
   src/moist_compressible.jl) and the per-level horizontal spline Helmholtz of
   `src/horizontal_si.jl`. Deliverable: a short derivation note
   (`reference/exact_si_derivation.md`) with the discrete operator, RHS staging,
   and leg-recovery expressions.
2. **Boundary rows** (the budgeted "session-eating detail"): rigid lid/surface
   w^{n+1}=0 ⇒ inhomogeneous Neumann `∂z p′^{n+1} = φ*/Δτ` rows; side-wall
   u-Dirichlet ⇒ `∂x p′^{n+1} = ρ̄_t u*/Δτ`; corners get both. Write them as
   modified rows of the banded system + RHS contributions. Sanity anchor: with
   the ∂xx block zeroed, the 2-D solve restricted to one column must reproduce
   the existing φ-form vertical solve's answer to solver roundoff (~1e-12) —
   derive this equivalence explicitly (it is the first unit test of Stage 1).
3. **Von Neumann part 3**: extend `model_tests/hsi_dg_von_neumann.jl` with (a)
   the unsplit 2-D solve + the ε chain-imperfection knob — confirm ε-insensitivity
   (expect the phase-1-like flat |G| ≤ 1 at every ε); (b) a two-layer c² profile
   with a constant-Pξ̄* implicit operator (AI2*+AB3), to document quantitatively
   why 1a is rejected (and under what warm-reference margin it would survive, for
   the record). Go/no-go: (a) must pass; if it fails, stop — the premise is wrong.
4. **Staging audit** (findings tension #3): legs recovered THROUGH the weak solve
   (w-leg) use the stored-applied-increment history; legs recovered by strong
   derivatives of solved coefficients (u from ∂x p′, slaved ρ/E slots) stay
   fresh-staged. Nothing staged outside the solve at predictor level. One page in
   the derivation note.
5. Size the RLR per-n factorization memory for the production grid; record the
   1a fallback decision rule.
6. Optional reference check: Tanguay, Robert & Laprise (1990) is in
   `~/Downloads/mwre-1520-0493_1990_118_1970_asislf_2_0_co_2.pdf` — consult only
   if 1a becomes live (house rule: originals before implementing).

### Stage 1 — XZ implementation behind `options[:exact_si]` (opt-in)

- New `src/exact_si.jl` (mirroring `horizontal_si.jl`'s structure): operator
  assembly + banded LU at `createModelTile` time; per-step patch-level solve
  after `calcTendency` + merge (where the phase-1 sweep sits — communication
  already paid).
- When `:exact_si` is on, the per-column vertical implicit adjustment in
  `semiimplicit_adjustment_p` is bypassed (the 2-D solve replaces it); the
  expdot remainder staging and AI2* history staging in `mc_driver!` are reused
  unchanged; the w-leg history becomes the applied increment of the 2-D solve
  (same convention as today: `(w^{n+1} − w*)/Δτ`).
- Flag interactions: `:exact_si` + `:state_dependent_si` ⇒ error at setup;
  `:exact_si` + `:horizontal_semiimplicit` ⇒ error. Flag-off must be bitwise
  (suite green unchanged) — keep every change behind the option.
- First unit test: A≡0 recoverability — `:exact_si` with the ∂xx block disabled
  matches the vertical-only path to ≤1e-10 on one step of the resting
  ISOTHERMAL base. (Stage-0 review correction: on a stratified base the p′- and
  φ-primary discretizations are different exact eliminations — agreement is
  truncation-level, NOT roundoff; the stratified vertical check is the Co_z 9
  sweep in G1, and the p′-primary stratified ceiling is a new measured
  quantity. See exact_si_derivation.md §3.)
- History A/B: start with the discipline that mirrors the vertical (stored w-leg,
  fresh others); one A/B round on the sweep if the default disappoints.

### Stage 2 — gates (all measured; fail ⇒ stop + findings note, no iteration)

- **G1 stability**: `model_tests/hsi_ceiling_sweep.jl` (add `--exact-si` lever)
  decay at Co_h ≤ 4.5, BOTH bases (isothermal + stratified), 600 s; no slow leak
  on `hsi_growth_probe.jl` (flat to 600 s at Co_h 3); report Co_h 9. ALSO the
  vertical must not regress: Co_z 9 decay on the stratified base with the 2-D
  solve active (this replaces the validated vertical solve — it must inherit its
  ceiling).
- **G2 accuracy**: BF02 dry bubble `--exact-si` vs off at `--ts-factor 0.5` and
  `1.5`: extrema (max_w, min_w, max|u|) within 2% / 5% (the user's rejection line
  from the approved plan; naive ADI measured 8–20% / 20–55%).
- **G3 regression**: full suite green, flag-off bitwise; 2-worker invariance
  ≤ 1e-10 (`hsi_distributed_smoke.jl` pattern); the sd_si/exact_si error test.
- **G4 combined**: Co_z 9 × Co_h 4.5 stratified sweep decays.
- **Cost**: `model_tests/step_cost_profile.jl` before/after — the solve should be
  ≲10% of a step (the phase-1 sweep was 9%).

### Stage 3 — follow-ons (separate sessions, NOT in this plan's scope)

Axisym radial metric (the TC actually needs this — XZ is the prototype), RLR
per-n solves, nesting integration + collar-width vs c·Δt check, default flip:
follow `reference/horizontal_si_phase2_plan.md` Phases 3–5, adapted to the 2-D
direct solve. And the **convective/axis-column investigation** (the actual
binding constraint; SI_CONVECTIVE_CEILING.md probes i–iv) — Fable session.

## Files touched

- `src/exact_si.jl` (new), `src/Scythe.jl` (include), `src/moist_compressible.jl`
  (bypass gating + flag validation), `src/horizontal_si.jl` (only if shared
  helpers are lifted), `warn_timestep_stability` in `src/reference_state.jl`
  (advisory target when `:exact_si` on — use measured envelope after G1).
- `model_tests/hsi_ceiling_sweep.jl`, `hsi_growth_probe.jl`,
  `hsi_dg_von_neumann.jl` (part 3), `benchmarks/bf02_{dry,moist}.jl`
  (`--exact-si`).
- `test/test_moist_compressible.jl` (new gate testset; keep the existing
  vertical + phase-1 gates untouched).
- `reference/exact_si_derivation.md` (new), findings/plan notes updated with
  outcomes; memory update at completion.

## Verification (end-to-end)

1. Suite: `julia --project test/runtests.jl` — full pass count reported, flag-off
   bitwise vs `fbe249e` baselines.
2. Sweeps/gates G1–G4 above, with the numbers written into the findings note.
3. TC smoke: 1-h axisym run at current NEST_TS with `:exact_si` **off** (bitwise
   guard on the production path — XZ-only stage must not perturb the TC).
4. Discipline from prior sessions: measure every variant on the sweeps before
   believing it; stratified base always included; `git add` specific files only
   (model_tests/ holds large untracked user files); no Claude footers in commits.
