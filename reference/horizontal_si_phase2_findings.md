# Horizontal SI Phase 2 findings — the ADI cross term is not benign (decision point)

Working note, 2026-07-17. Status: **Phase-2-DG round complete (same day, second
session): the delta-form Douglas–Gunn split was implemented, measured, and
REJECTED — G1 fails structurally, mechanism identified and quantified (see "The
Phase-2-DG round" at the end). The plan's stopping rule applies: proceed to
variant 1. The tree reverts the sweep to the Phase-1 composition (opt-in,
rejected-for-production) pending that decision.**
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

## Addendum (2026-07-17, same session): the compensated factorization, derived

The factored solve applies X = (I−νA)⁻¹(I−νB)⁻¹S, i.e. (I−νB)(I−νA)X = S, so the
effective system is (I−νL)X = S − ν²BA·X — the defect is the single cross term
ν²BA·X^{n+1}. For this acoustic pair the product has ONE nonzero row: A touches
only (u, p) and returns nothing in w; B's w-row is the only row that reads p; so

    (BA·X)_w = (1/ρ̄_t) ∂z( Pξ̄ ρ̄_t ∂x u ),      every other row exactly 0.

The Douglas–Gunn-style repair is therefore one term: solve the factored system
with S′ = S + ν²·(BA·X̃)_w, X̃ a lagged state — the effective system becomes
(I−νL)X = S − ν²BA(X − X̃) = S + O(ν³) per step, restoring second-order
consistency of the split (cumulative O(ts²) instead of O(ts)). Concretely: the
sweep evaluates G = (1/ρ̄_t)∂z(Pξ̄ρ̄_t ∂x u^{n+1}) from the solved coefficients
(∂x u is already in hand; the ∂z chain existed for the reverted w-history
experiment) and ships it as plane 2 of the increment feed; the next step adds
+ν²·G to the w predictor before the vertical solve. Sign cross-check: the
reverted "correct the w history" experiment applied this SAME field with net
weight −0.25·ts (through the AI2* history weights) — the opposite sign — and
made the leak worse (e-fold 110 → 50 s), exactly as this algebra predicts.

**Measured (same session): the LAGGED compensation is unstable.** Implemented as
+ν²·G(lagged) on the w predictor (plane 6 of the feed, `options[:hsi_cross_comp]`,
now default OFF), the Co_h 3 probe NaNs within 200 s. The reason is structural: in
the exact composition the defect −ν²BA·X sits INSIDE the implicit bracketing, so
for grid-scale oblique modes with ν²|BA| ≈ Co_x·Co_z ≳ 1 it is large but
regularized; a lagged explicit source of the same size is not. The consistent
repair is the true delta-form Douglas–Gunn split — solve for the INCREMENT with
each factor applied implicitly around the correction:

    (I − νB) δ¹ = [AB3 remainder + histories] + ν·L(Xⁿ)      (per column; the
                   x-leg ν·A(Xⁿ) is a pointwise grid-slot product, available
                   in the column step)
    (I − νA) δ  = δ¹                                          (patch sweep)
    X^{n+1} = Xⁿ + δ

whose splitting error acts on δ = O(ts) instead of X = O(1). This RESTRUCTURES
the vertical solve's RHS (the plan had deliberately kept it untouched) and its
stability for the wave-type system at large Courant must be verified against the
ADI-SI literature — Ikawa (1988, JMSJ) is the designated reference (house rule:
pull the original before implementing). The alternative remains variant 1 (the
vertical-normal-mode exact 3-D solve, no splitting at all).

## State of the tree

Committed and green (suite 7646/7646): opt-in Phase-1 sweep (code defaults
`hsi_u_history="none"`, `hsi_x_history="fresh"` — the stable, over-damping variant),
regression gate at Co_h 3 (code defaults), multi-limit `warn_timestep_stability`,
worker-invariance smoke (9e-12), `--hsi` benchmark flag, A/B levers
(`hsi_u_history="stored"`, `hsi_x_history="stored"`, `hsi_scheme="am2"`), 5-plane
increment feed (all applied increments available to any history scheme — DG will
want them too).

## The Phase-2-DG round (2026-07-17, second session): delta form implemented, measured, rejected

### Derivation-pass corrections to the plan

- **Ikawa (1988, JMSJ) contains NO ADI-split semi-implicit scheme** — the plan's
  citation was a false lead (text hits are e.g. "r**adi**ation"). Its schemes are
  AE, E-HI-VI (a FULL 2-D implicit elliptic solve, iterated direct method — the
  class of variant 1), and E-HE-VI (split-explicit). Useful content: the
  Kurihara (1965)/SHB78 partially-implicit instability discussion, consistent
  with the state-dependence lessons. The two-factor delta form stands on
  Douglas & Gunn (1964) / Beam & Warming (1978). Caution recorded: the
  THREE-factor delta form is known unstable for pure wave systems — future 3-D
  compositions must stay at two factors.
- AI2* weights verified against Durran & Blossey (2012) eq. (30): P₃ = I −
  (5/4)LΔt etc. expand to exactly the implemented +1.25 L^{n+1} − 1.0 Lⁿ +
  0.75 L^{n−1}.
- Von Neumann analysis (`model_tests/hsi_dg_von_neumann.jl`, part 1), exact
  arithmetic, AI2* weights, 2-D acoustic pair, Co ∈ [0,30]²: unsplit 1.0;
  naive sequential 1.019 at oblique Co ≈ 0.5 (the measured leak) with 0.85
  over-damping at Co 2 (the measured accuracy deficit); **delta-form DG 1.0 to
  1e-8 for both u-history variants** — the premise of the round was sound.
- Key algebraic identities found during design: (i) with Y ≡ Xⁿ + δ¹ the DG
  z factor is the EXISTING vertical state solve applied to X* + ν·A(Xⁿ) — no
  restructure of `semiimplicit_adjustment_p`, both vertical coefficient paths
  covered, B-only case exact; (ii) the whole DG scheme is algebraically the
  naive composition plus the single source +ν²(BA·Xⁿ)_w — i.e. the shelved
  `hsi_cross_comp` lever evaluated at the CURRENT state was already the DG
  scheme in disguise.

### What was implemented and measured (all on the 120-s resting sweeps, both bases)

The delta sweep (coefficient-difference δ¹ against a snapshot baseline,
per-level Helmholtz on δ_u, write-back δ − δ¹, 5-plane operator feed) plus the
per-column predictor addition ν·A(Xⁿ), in successive variants as instability
drivers were identified:

1. **Predictor staged through the grid-slot chain** (A_grid): growth e-fold
   ≈ 20 s at Co_h 3, all history variants (stored/stored, stored/fresh,
   none/fresh — history-independent).
2. **Predictor through the sweep-chain feed** (A_sweep evaluated from the final
   coefficients): statistically identical growth — chain-of-staging was not the
   dominant driver.
3. **u-row via an accumulated applied-increment recursion** (au ← au + A_u(δ)):
   much worse (growth at Co_h 1.5). A stored-increment field used as a
   PREDICTOR is a free integrator coupled to the state — secular growth. The
   vertical w-leg stored increment works only because it is a HISTORY
   (net −0.25 weight), refreshed each step from a state-form solve.
4. **Round-tripped baseline fix** (the real first-order bug): δ¹ = Y − Xⁿ_raw
   leaks (P − I)Xⁿ — the l_q refit residual of the CARRIED STATE, grid-scale
   and state-amplitude — into the sweep, which the acoustic coupling re-injects
   across variables every step. Fix: snapshot the baseline through the same
   eval + fit round trip the state undergoes (P = fit∘eval), so
   δ¹ = fit(step increment) exactly. This removed the violent low-Co growth
   (Co 1.5 clean; Co 3 growth slowed to e-fold ≈ 25–40 s) — but growth at
   Co 3–4.5 persists in every history variant.
5. **Isolation control** (predictor additions disabled): worse (NaN at Co 3 in
   ~90 s) — the DG structure is net stabilizing; the residual growth is the
   imperfect recombination.

### The mechanism, quantified (the structural verdict)

`model_tests/hsi_dg_von_neumann.jl` part 2 adds a chain-imperfection knob: the
sweep's weak Helmholtz symbol differs from the strong read chain by a factor
(1 − ε) on k². Results:

- **phase-1 state-form: ε-INSENSITIVE** (max|G| = 1.063 from the cross term at
  every ε) — state-form solves absorb chain mismatch as a consistent shift of
  the implicit operator.
- **DG delta form: ε-fragile** — the split requires ν·A(Xⁿ) (staged outside
  the weak solve) to recombine with the weak-solve increment term exactly; the
  mismatch leaves ε·ν·A of the STATE-amplitude fast operator explicit.
  Growth onset: Co ≈ 3 at ε = 0.1, Co ≈ 4.5 at ε = 0.05, Co 9 unstable even at
  ε = 0.02. Measured onset (Co 1.5–3) ⇒ effective ε ≈ 0.1–0.2 at grid scale —
  the expected weak-Galerkin vs strong-chain + l_q-fit symbol difference for
  the cubic-spline basis.
- **naive + ν²BA(Xⁿ) source** (the DG-equivalent arrangement): the source is
  O(Co²·X) at grid scale, so its ε-fraction is explicit at ε·Co² ⇒ ceiling
  Co ≈ √(0.7/ε) ≈ 2 — this retroactively explains why the `hsi_cross_comp`
  lever NaN'd at Co 3 despite being (at the correct time level) algebraically
  the DG scheme.
- The z-factor-last ordering moves the same intrinsic split to the w row,
  whose pointwise-staged ceiling is the MEASURED vertical Co_z ≈ 2–3 of
  2026-07-16 — both orderings are blocked by the same wall.

**Conclusion: every two-factor composition of this pair requires staging one
weak-solve leg's linear operator at state amplitude outside its own solve, and
the ε ≈ 0.1–0.2 chain mismatch of this discretization converts that into a
hard Courant ceiling ≈ 0.7/(1.25ε) ≈ 3–6. G1 (stable 4.5 both bases, report
9) needs ε ≲ 0.02–0.05: not achievable. The delta-form DG split is REJECTED —
not by tuning, but structurally.** Per the plan's stopping rule the next step
is variant 1 (the vertical-normal-mode exact 3-D solve) — an UNSPLIT
state-form solve, i.e. exactly the ε-tolerant class (Ikawa's E-HI-VI is the
literature precedent).

### Variant-1 design tensions to resolve BEFORE implementing (user decisions)

1. **`:state_dependent_si` is structurally incompatible with a precomputed
   vertical-normal-mode decomposition** (state-dependent operator ⇒
   state-dependent eigenmodes). Freezing modes at the reference reintroduces
   the convective ceiling (tc/SI_CONVECTIVE_CEILING.md); re-eigen-solving per
   column per step erases the cost advantage. Realistically variant 1 forces
   the two-path structure the user was already weighing: keep the current
   per-column vertical-only SI (with sd_si) as one path, add the full 3-D
   normal-mode solve (reference-linearized) as the other. The maintenance
   concern is real but the split is intrinsic, not optional.
2. The p′-form boundary rows (inhomogeneous Neumann ∂z p′ = φ*/Δτ at the lid,
   and the u-Dirichlet side walls) remain the budgeted "session-eating detail".
3. The ε-analysis methodology (part 2 of the script) should be reused to check
   variant 1's staging choices before implementation: any leg staged outside
   the 3-D solve must be a history (small net weight), never a predictor-level
   term.

### Stage-0 follow-up (2026-07-17): variant 1 derivation done, GO

`reference/exact_si_derivation.md` + `model_tests/hsi_dg_von_neumann.jl` part 3.
The ε-analysis methodology of part 2 (tension #3 above) was reused on the unsplit
2-D solve: it is **ε-INSENSITIVE** (flat max|G| ≤ 1.003 at every ε ∈ [0,0.2],
Co ≤ 30²) — exactly the ε-tolerant state-form class this note predicted, and the
opposite of the DG split (max|G| 1.10–1.54, ceiling Co 3–6). Requirement surfaced
by the derivation: the leg recoveries must reuse the stiffness's M1 (weak/weak);
staging them strong against the weak Helmholtz reintroduces ε-growth (max|G| 1.10
at ε 0.2 — part 3(a-cautionary)). The normal-mode form 1a is rejected: the in-mode
SHB78 residual is a benign rank-1 perturbation, but inter-mode coupling of δc²∂zz
+ the slaved legs give the measured r·Co_z ≲ 0.72, and a warm-reference constant
Pξ̄* fails Co_z 9 by ~4.5× on the Dunion sounding — the local-profile operator
stays load-bearing. Review correction (same day): the derivation's first-draft
vertical block (the φ-solve's DPD weighted stiffness applied to p′) solved
NEITHER elimination — off by the commutator [D,P]D ≈ (∂zPξ̄)∂z (script part 3(c):
misses by the full commutator scale while the exact p′-elimination and the
corrected P⁻¹-scaled weighted-MASS form `M0ᵀ(W/Pξ̄)M0 + Δτ²·plain stiffness`
match the φ-solve to ~1e-16). The A≡0 roundoff unit test is therefore
ISOTHERMAL-base; the p′-primary stratified vertical ceiling is a new measured
quantity for Stage 1's sweeps. Proceeding to Stage 1 (opt-in `:exact_si`).

### Tree state after this round

The DG implementation is preserved in the commit history (one commit, gates
marked skipped) and then REVERTED to the Phase-1 composition (opt-in, its own
documented limits), so the live `options[:horizontal_semiimplicit]` behavior is
unchanged from the 2026-07-17 morning state. Kept: this note,
`model_tests/hsi_dg_von_neumann.jl`, and the harness `--no-dg-pred` lever
documentation in the history.
