# Variant 1 (unsplit XZ acoustic SI): p′-form derivation, assembly, boundary rows, staging, sizing

Stage-0 derivation note, 2026-07-17. Deliverable A/B/C of the Stage-0 section of
`reference/exact_si_plan.md`. NO src/ changes are made in Stage 0; this note plus
the extended `model_tests/hsi_dg_von_neumann.jl` (part 3) are the whole output.
**Revised same day after Stage-0 review**: the first draft's recommended vertical
block (the φ-solve's weighted stiffness applied to p′) solved neither elimination
— off by the commutator `[D,P]·D ≈ (∂zPξ̄)∂z`. Corrected to the P⁻¹-scaled
weighted-MASS form (§2, exact and self-adjoint; symbolic verification in script
part 3(c)); the boundary loads (§3-BC) and the Stage-1 unit test (§3) are
re-derived accordingly. The GO verdict (part 3(a)) is form-independent and
unaffected.
Companion to `reference/horizontal_acoustic_si_handoff.md` (the assessment),
`reference/horizontal_si_phase2_findings.md` (why every split was rejected), and
`tc/SI_VERTICAL_CEILING.md` (the operator-consistency and SHB78 lessons that
govern every formula here). Where the handoff sketch and the code disagree, the
code + SI_VERTICAL_CEILING win, and the disagreement is flagged.

Notation follows the existing reference notes: `Pξ̄(z)` the LOCAL reference sound
speed squared `γ̄_m(z)·p̄(z)/ρ̄_t(z)` (`mc_ref_diag.Pxi_prof`), `φ = ρ̄_t w` the
vertical mass flux, `Δτ = 1.25·Δt` the off-centred AI2* implicit coefficient
(`0.5·Δt` on the AM2 first step), `p′` the pressure perturbation, `∂x`/`∂z` the
horizontal/vertical spline derivatives. `X* = (u*, φ*, p′*, ρ_d′*, ρ_t′*, E_t′*)`
is the predictor after AB3 of the acoustic remainder AND the AI2* explicit
history levels have been applied (`mc_driver!`, `horizontal_si_history!`).

---

## 0. The reference-linearized XZ acoustic pair (from the code, not the sketch)

The pair the vertical solve (`semiimplicit_adjustment_p`,
src/moist_compressible.jl:1139) and the horizontal sweep (`src/horizontal_si.jl`)
integrate, XZ slice, `u` the fast horizontal leg, `φ = ρ̄_t w`:

    ∂u/∂t   = −(1/ρ̄_t) ∂x p′
    ∂φ/∂t   = −∂z p′
    ∂p′/∂t  = −Pξ̄(z) ( ρ̄_t ∂x u + ∂z φ )
    slaved:  ∂ρ_t′/∂t = −( ρ̄_t ∂x u + ∂z φ )
             ∂ρ_d′/∂t = −( ρ̄_d ∂x u + ∂z(c_d φ) ),   c_d = ρ̄_d/ρ̄_t
             ∂E_t′/∂t = −( (Ē_t+p̄) ∂x u + ∂z(c_e φ) ), c_e = (Ē_t+p̄)/ρ̄_t

All reference coefficients are z-only, so along x they are constants (the reason
the horizontal operator is constant-coefficient along its solve direction —
handoff §"Why this model wants the implicit route"). The vertical form carries
the local profile Pξ̄(z), ρ̄_t(z), and their z-derivatives; the domain-mean scalar
is the SHB78 instability (§4 below, and SI_VERTICAL_CEILING "Second mechanism").

The AI2* implicit level (Durran & Blossey 2012; verified in the phase-2 findings
against eq. (30)) is `X^{n+1} = X* + Δτ·L X^{n+1}`, i.e. per leg:

    u^{n+1}  = u*  − (Δτ/ρ̄_t) ∂x p′^{n+1}                       … (I)
    φ^{n+1}  = φ*  − Δτ ∂z p′^{n+1}                             … (II)
    p′^{n+1} = p′* − Δτ Pξ̄ ( ρ̄_t ∂x u^{n+1} + ∂z φ^{n+1} )     … (III)

The explicit history levels `−1.0 Lⁿ + 0.75 Lⁿ⁻¹` are already inside X* (they are
staged per column in `mc_driver!`/`horizontal_si_history!`; §5). Both dimensions'
histories belong to X* before either implicit direction acts.

---

## 1. p′-form elimination → one 2-D Helmholtz

Substitute (I) and (II) into (III). ρ̄_t is z-only, so `∂x(ρ̄_t⁻¹ · ) = ρ̄_t⁻¹ ∂x`
and `ρ̄_t ∂x(ρ̄_t⁻¹ ∂x p′) = ∂xx p′`:

    p′^{n+1} = p′* − Δτ Pξ̄ ρ̄_t ∂x u* − Δτ Pξ̄ ∂z φ*
                   + Δτ² Pξ̄ ∂xx p′^{n+1} + Δτ² Pξ̄ ∂zz p′^{n+1}

giving the scalar **p′-form Helmholtz** (the handoff form, `handoff:56`):

    ( I − Δτ² Pξ̄(z) (∂xx + ∂zz) ) p′^{n+1} = p′* − Δτ Pξ̄ ( ρ̄_t ∂x u* + ∂z φ* )   … (H)

Pξ̄(z) **left-multiplies** the whole Laplacian. This is the exact elimination of
(I)–(III) into p′. Two consequences drive the assembly and the equivalence test:

- The horizontal block `Pξ̄ ∂xx` has Pξ̄ constant along x (z-only), so per
  physical level z_k it is `Pξ̄(z_k)·∂xx` — exactly the per-level operator the
  phase-1 sweep already assembles (`src/horizontal_si.jl:126`).
- The vertical block `Pξ̄ ∂zz` has Pξ̄ **outside** the derivatives — non-self-
  adjoint when Pξ̄ varies, and NOT the operator of the validated φ-solve (whose
  elimination is the self-adjoint `∂z(Pξ̄ ∂z·)` acting on φ). The two operators
  belong to DIFFERENT eliminations of the same pair: with D the discrete ∂z and
  P = diag(Pξ̄), the φ-form is `(I − Δτ²·DPD)φ` and the p′-form is
  `(I − Δτ²·PDD)p′`. Putting DPD on p′ with the p′ RHS solves NEITHER — it
  differs from the exact p′ system by the commutator `[D,P]·D ≈ (∂zPξ̄)∂z`, a
  reference-stratification-scale term (this was the error in the first draft of
  this note, caught in Stage-0 review; verified numerically in the script's
  part 3(c): the hybrid misses the true solution by exactly the commutator
  scale, while the exact p′-elimination and the §2 scaled form match the
  φ-solve to ~1e-16). The resolution — exact AND self-adjoint at identical
  cost — is to multiply the p′ elimination through by 1/Pξ̄; §2.

---

## 2. Discrete operator: reuse the two validated weak-Galerkin patterns

The B-coefficient tensor basis on a patch is `p̂[(k_b−1)·b_iDim + i_b]` — i_b the
i-spline coefficient (contiguous, fast), k_b the vertical block (slow). This is
the existing patch coefficient layout (`src/horizontal_si.jl:152`,
`r1 = (z−1)·b_iDim`). Keep it: it minimizes the bandwidth (§6).

Both 1-D building blocks already exist and are cubic-B-spline weak Galerkin forms
built from the SAME mish operators `M0`(kDim×b), `M1`(kDim×b) and physical
quadrature weights `W` the explicit `gridTransform` tendencies use — the
operator-consistency requirement (SI_VERTICAL_CEILING; block comment
src/semiimplicit.jl:1880). Measured half-bandwidth of `M1ᵀW M1` and `M0ᵀW M0` is
**3** in coefficient index (cubic-spline support; verified this session).

The existing 1-D patterns being reused (as PATTERNS — the profile-weighted
quadrature and the plain Galerkin blocks; the new form composes them
differently, see below):

- **Profile-weighted quadrature pattern** `M1ᵀ(W·prof)M1`
  (`_assemble_spline_matrix` profile method, src/semiimplicit.jl:1966; used by
  `calc_Helmholtz_semiimplicit_matrix` profile, :2126, where prof = Δτ²Pξ̄ — the
  φ-solve's weighted stiffness, validated to Co_z ≈ 9–18). The p′ form applies
  the SAME `(W .* prof)` weighting to `M0` instead (`prof = 1/Pξ̄` — the
  weighted mass).
- **Plain stiffness** `M1ᵀW M1` and **plain mass** `M0ᵀW M0` on u's/p's i-basis
  and the vertical basis (`_rirk_solve_data`, :1909; the phase-1 `build_set`,
  src/horizontal_si.jl:128, assembles per-level combinations of exactly these).

### The operator we assemble (recommended: the P⁻¹-scaled weighted-MASS form)

*(Revised after Stage-0 review — the first draft recommended DPD-on-p′, which
solves neither elimination; §1 and script part 3(c).)*

Multiply the exact p′ elimination (H) through by `1/Pξ̄(z)`:

    ( 1/Pξ̄(z) − Δτ² (∂xx + ∂zz) ) p′^{n+1}
        = p′*/Pξ̄(z) − Δτ ( ρ̄_t ∂x u* + ∂z φ* )                     … (H′)

This is **algebraically the exact p′ elimination** (no operator modification —
part 3(c) shows it matches the φ-solve + recovery to ~1e-16 with a shared
discrete D even on a two-level P) and it is **self-adjoint**: the Pξ̄ weight now
sits on the identity (a weighted mass), and the whole Laplacian is plain. Weak
Galerkin on the tensor basis, with `Sx = (M1ᵀWM1)_i`, `Sz = (M1ᵀWM1)_k`,
`Mx = (M0ᵀWM0)_i`, `Mz = (M0ᵀWM0)_k`, `⊗` the tensor product in the (i_b, k_b)
layout:

    A = (Mx ⊗ Mz_weighted)                [weighted MASS, Mz_weighted = M0ᵀ(W/Pξ̄)M0]
      + Δτ² [ (Sx ⊗ Mz) + (Mx ⊗ Sz) ]     [PLAIN stiffness, both directions]

The `1/Pξ̄(z)` weight goes inside the VERTICAL quadrature (the vertical mish
index is the quadrature point; Pξ̄ is z-only so the weight is exact per level and
the horizontal factor stays the plain `Mx`). `Mz_weighted` is the mass-matrix
analogue of the existing profile-weighted stiffness pattern
(`_assemble_spline_matrix` profile method, src/semiimplicit.jl:1966 — same
`(W .* prof)` quadrature weighting, applied to M0 instead of M1); one new
NamedTuple entry beside `d.Mass` in `_rirk_solve_data`.

**Symmetry / definiteness.** All four blocks are symmetric; the weighted mass is
positive definite (Pξ̄ > 0) and the stiffnesses positive semi-definite, so `A` is
SPD. Moreover the boundary conditions of §3-BC are all *natural* (load-only, no
row replacement), so — unlike the 1-D solves, whose Dirichlet rows break
symmetry — the assembled 2-D system stays SPD. Stage 1 plans a banded LU (the
plan's sizing, §6, assumes it and it needs no symmetry); a banded Cholesky is an
optional ~2× memory/flop saving, not load-bearing.

**Per-level reuse of the phase-1 assembly.** Under the scaling, the horizontal
per-level operator is no longer `Mass + Δτ²Pξ̄(z_k)·Stiff` (the phase-1
`build_set`, src/horizontal_si.jl:128) but its exact multiple by `1/Pξ̄(z_k)`:
`(1/Pξ̄(z_k))·Mass_x + Δτ²·Stiff_x`. So the phase-1 BUILDING BLOCKS (`du`/`dp` =
`_rirk_solve_data` on u's/p's i-basis: M0, M1, W) are reused unchanged, but the
per-level factorization set is not (the 2-D solve is one banded LU), and the
level dependence moves from the per-level stiffness coefficient into the
weighted vertical mass. Signs: (H′) is naturally `+wtdMass + Δτ²Stiff = load`
(all positive); the existing vertical solve's convention is the negated
equivalent (`−Stiff − Mass`, `calc_Helmholtz_semiimplicit_matrix`:2114). One
global convention, fixed at implementation (§7).

### The literal PDD form (documented fallback)

The unscaled exact p′ elimination `(I − Δτ²Pξ̄(∂xx+∂zz))p′ = p′* − Δτ Pξ̄(ρ̄_t ∂x
u* + ∂z φ*)` — Pξ̄ outside, non-self-adjoint — is equally exact (part 3(c)) and a
banded LU does not need symmetry, so it remains a valid fallback if the
weighted-mass quadrature exposes an unexpected obstruction. It gives up the SPD
structure and the natural-BC symmetry, gains nothing; not recommended.

**What must NOT be assembled: the DPD-on-p′ hybrid** (the vertical φ-solve's
weighted stiffness `M1ᵀ(W·Pξ̄)M1` acting on p′ with the p′ RHS). It differs from
every exact elimination by `[D,P]·D ≈ (∂zPξ̄)∂z` — measured in part 3(c) at the
full commutator scale — and would (a) fail the A≡0 equivalence deterministically
on any stratified base and (b) silently disagree with the staged
remainder/histories by that term.

---

## 3. RHS staging, leg recovery, and the ∂xx→0 vertical-equivalence

### RHS

The load of the scaled form (H′) splits into two differently-weighted pieces —
write them out carefully, they do NOT share a quadrature weight:

    load = M0ᵀ(W/Pξ̄)·p′*  −  Δτ · M0ᵀW·( ρ̄_t ∂x u* + ∂z φ* )

i.e. the `p′*` term enters through the SAME `1/Pξ̄`-weighted quadrature as the
weighted mass (operator/RHS consistency on the identity term), while the
divergence term enters **unweighted** (the Pξ̄ that multiplied it in (H) is
cancelled by the P⁻¹ scaling). In the tensor layout both weights sit in the
vertical quadrature factor; the horizontal factor is plain `M0ᵀW` for both
pieces. Fields are formed on the patch coefficients through the model's read
chain (`_hsi_to_levels!`, src/horizontal_si.jl:145 — per-z_b i-fit→evaluate,
then per-i vertical fit→evaluate, the `gridTransform` chain, so the solve sees
the same discrete state the grid slots see). The two divergence pieces are
exactly the existing reads: `∂x u*` (as in the phase-1 load, :247) and `∂z φ*`
(the vertical solve's `p_nstar_z`/`φ*` chain, moist_compressible.jl:1231–1238).
Predictor X* already carries the AB3 remainder and BOTH dimensions' AI2*
explicit histories.

### Leg recovery (all from the ONE solved p′^{n+1})

    u^{n+1}   = u*  − (Δτ/ρ̄_t) ∂x p′^{n+1}        (I): strong ∂x of solved p′
    φ^{n+1}   = φ*  − Δτ ∂z p′^{n+1};  w = φ/ρ̄_t   (II): strong ∂z of solved p′
    ρ_t′^{n+1}= ρ_t′* − Δτ ( ρ̄_t ∂x u^{n+1} + ∂z φ^{n+1} )
    ρ_d′/E_t′ : the c_d/c_e flux-form analogues (moist_compressible.jl:1270–1301)

Operator consistency (SI_VERTICAL_CEILING): ∂x and ∂z of p′^{n+1} in the
recoveries MUST reuse the SAME `M1` the stiffness uses (the phase-1 sweep does
this — `dxu += du.M1[i,j]·acoef[j]`, src/horizontal_si.jl:265 — recovering ∂x u
from the solved coefficients through the stiffness's own M1). The corrections are
written back as B-coefficient INCREMENTS through the forward chain
(`_hsi_add_increment!`, :184) so they get the same fit treatment `calcTendency`
gives the state. The von Neumann part-3(a-cautionary) table quantifies the cost
of breaking this: strong recoveries against a weak Helmholtz reintroduce
ε-growth (max|G| 1.10 at ε 0.2 vs 1.003 consistent).

### ∂xx→0 equivalence with the vertical φ-solve (Stage-1 unit test #1, REVISED)

*(Revised after Stage-0 review: the first draft claimed roundoff equivalence on
any base; that claim was tied to the erroneous DPD-on-p′ operator and is false.)*

Zero the horizontal block (Sx→0). Column-by-column, (H′) becomes the 1-D
weighted-mass system `(M0ᵀ(W/Pξ̄)M0 + Δτ²M1ᵀWM1)·p̂′ = load`. How this relates to
the validated φ-solve (`−M1ᵀ(W·Δτ²Pξ̄)M1 − M0ᵀWM0` on φ) splits by base:

- **(i) Isothermal base (Pξ̄ constant): roundoff equivalence — THE A≡0 unit
  test.** With P constant the weighted mass is `(1/P)·M0ᵀWM0`, so the p′
  operator is `(1/P)·[M0ᵀWM0 + Δτ²P·M1ᵀWM1]` — proportional to the φ-solve's
  matrix. Part 3(c) (constant-P case) confirms: with the same discrete D chain,
  p′-solve + φ-recovery ≡ φ-solve + p′-recovery to machine roundoff (the
  resolvent identity — both are exact Schur complements of the same 2×2 block
  system). **Stage-1 unit test #1: `:exact_si` with ∂xx disabled vs the
  vertical-only path, ≤1e-10 on one step of the resting ISOTHERMAL base.**
- **(ii) Stratified base: truncation-level agreement, NOT roundoff.** Both
  eliminations are exact in exact arithmetic (part 3(c): ~1e-16 with one shared
  square D), but the CODE discretizes them in different variables through
  different weak chains: the φ-solve works in w's Dirichlet column basis with
  the Pξ̄-weighted stiffness; the p′-solve works in p's basis with the
  1/Pξ̄-weighted mass and natural rows, and the recoveries pass through separate
  fit-evaluate chains. Their discrete D's are not one shared matrix, so on a
  stratified base the two answers agree only at the spline-truncation level of
  the (∂zPξ̄)-bearing terms — O(l_q-filtered truncation of Pξ̄(z) over a cell),
  expected orders of magnitude above 1e-12 but shrinking with resolution. Do
  NOT assert coefficient equivalence on the stratified base. The stratified
  vertical gate is the **Co_z 9 stratified sweep** (G1's "vertical must not
  regress" clause), not equivalence.

**Consequence, stated plainly: the p′-primary stratified vertical ceiling is a
NEW measured quantity that Stage 1's sweeps must establish.** The expectation is
that it matches the φ-solve's Co_z ≈ 9–18 (both are consistent self-adjoint
discretizations of exact eliminations of the same pair, and part 3(a) is
ε-insensitive at the symbol level), but it is measured, not inherited.

---

## 3-BC. Boundary rows (the budgeted session-eating detail)

The vertical φ-solve works in **w's Dirichlet basis** (φ = 0 rows via `d.Nb`,
`_assemble_spline_matrix`:1972). The p′-form needs **inhomogeneous Neumann**
conditions instead — the physical wall conditions expressed on p′ through the
recoveries. Under the scaled form (H′) the stiffness is PLAIN (no Pξ̄ factor), so
integrating by parts yields the boundary flux `[ψ · Δτ² ∂n p′]` **without a Pξ̄
factor**, and the loads are correspondingly Pξ̄-free (revised from the first
draft, whose `+Δτ Pξ̄ (·)` loads belonged to the unscaled stiffness):

- **Rigid lid / surface** (z = z_t, z_b): `w^{n+1} = 0 ⇒ φ^{n+1} = 0`. By (II),
  `∂z p′^{n+1} = φ*/Δτ` at the boundary. Natural condition: the by-parts flux of
  the plain vertical stiffness is `[ψ · Δτ² ∂z p′]`, so the inhomogeneous value
  enters the LOAD as `+ Δτ · φ*` evaluated at the lid/surface node (no row
  replacement; NO Pξ̄ factor). Recovering φ then gives
  `φ^{n+1} = φ* − Δτ(φ*/Δτ) = 0` at the wall exactly — which is why the ∂xx→0
  restriction reproduces the Dirichlet-φ answer (§3(i)) even though the p′
  operator carries a Neumann condition. **Note vs the handoff:** the lid BC is a
  load contribution, not a replaced row; the row stays the natural weak row.
- **Side walls / axis** (x = x_l, x_r; r = 0 axis on axisym): u-Dirichlet
  `u^{n+1} = 0 ⇒ ∂x p′^{n+1} = ρ̄_t u*/Δτ` by (I). Natural condition on the
  plain horizontal stiffness; load contribution `+ Δτ · ρ̄_t(z_k) u*` at the wall
  node per level (ρ̄_t from the recovery relation (I), NOT a quadrature weight;
  no Pξ̄). (For Neumann u walls — the other case `horizontal_si` supports, :120 —
  `∂x u = 0` gives a homogeneous natural condition and no load term.) On axisym
  the r = 0 axis is u-Dirichlet (`tc_init.jl` axis_bc), same structure; RLR adds
  per-wavenumber axis-regularity rows in a later phase (Stage 3).
- **Corners** (lid ∩ wall): both load contributions add on the single corner
  coefficient; no interaction term (the two are independent first-order fluxes).
  Structure unchanged from the first draft.

Because every BC is natural (load-only), the §2 SPD structure survives the
boundary treatment intact.

All boundary contributions are LOAD terms at state amplitude (`φ*`, `u*`),
recomputed each step from the predictor — they are not staged histories and carry
no accumulation risk. The banded structure is preserved: the lid/wall rows couple
only within ±3 coefficients of the boundary (cubic support), inside the §6
bandwidth.

---

## 4. Von Neumann part 3 — the go/no-go and the SHB78 quantification

`model_tests/hsi_dg_von_neumann.jl` part 3 (exact arithmetic, AI2* weights, same
ε chain-imperfection knob as part 2). Run: `julia --project
model_tests/hsi_dg_von_neumann.jl`.

### 3(a) unsplit 2-D solve — GO/NO-GO = ε-insensitivity → **PASS**

Consistent weak solve+recovery (L_weak = √(1−ε)(A+B), recovery reuses the same
weak operator — the M1/M1 discipline of §3), strong AI2* histories, both u-leg
disciplines (θ=1 and fresh):

| ε    | max|G| over Co ∈ [0,30]² |
|------|--------------------------|
| 0.0  | 1.00000000               |
| 0.02 | 1.00000000               |
| 0.05 | 1.00000000               |
| 0.10 | 1.00007800               |
| 0.20 | 1.00305872               |

Flat max|G| ≤ 1.003 at every ε — the ε-insensitive state-form class the DG round
proved is the only viable composition (part 2: the DG split hit max|G| 1.10–1.54
at ε 0.1–0.2, ceiling Co 3–6). **The premise of variant 1 holds; proceed.**
Cautionary contrast (weak Helmholtz + STRONG recovery, breaking M1/M1): max|G|
climbs to 1.10 at ε 0.2 — the Stage-1 rule that the recovery derivative and the
stiffness must share M1.

**This GO verdict is form-independent**: the von Neumann analysis is at the
symbol level with locally constant P, where the weighted-mass (H′), literal PDD,
and φ-form eliminations coincide exactly — so the §2 operator revision after
Stage-0 review does not touch it. The discrete-elimination equivalence and the
DPD-on-p′ commutator error are checked separately in **part 3(c)** (two-level P,
shared discrete D): exact-p′ and weighted-mass match the φ-solve to ~1e-16;
DPD-on-p′ misses by the full commutator scale.

### 3(b) the constant-Pξ̄* normal-mode form (1a) — why it is rejected

1a needs a CONSTANT implicit Pξ̄* to separate variables (plan "Design
resolution"). The local deviation `δc²(z) = c²(z) − Pξ̄*`, fraction `r = δc²/Pξ̄*`,
is left under AB3. Two findings:

1. **The in-mode residual is benign.** In a single mode the residual `δc² ∂zφ` is
   a rank-1, nilpotent perturbation of the p←w coupling (the PGF −∂z p/ρ is
   Pξ̄-free and stays implicit, `Lres² = 0`), so the implicit AI2* damping of the
   same mode absorbs it up to r ≈ 1 (a DOUBLING of local c²): max|G| ≤ 1 for
   r ≤ 0.75 at every Co*, first exceeding 1 near r ≈ 1 (measured table in the
   script). This is FAR above the measured stratified-base NaN (r ≈ 0.25 at Co_z
   9, SI_VERTICAL_CEILING). So 1a's instability is **not** the in-mode residual.
2. **The operative danger is inter-mode coupling + the slaved legs.** `δc²(z) ∂zz`
   couples the precomputed constant-Pξ̄* eigenmodes (the very
   non-simultaneous-diagonalizability that forces the constant reference), and
   the ρ_d/ρ_t/E_t legs add more explicit acoustic pathways — neither visible to a
   single-mode symbol. The MEASURED threshold is linear in the deviation,
   `r·Co_z ≲ 0.72` (SI_VERTICAL_CEILING: r·Co_z ≈ 0.25·1.8 < 0.72 stable at
   ts 0.3; ≈ 2 at Co_z 9, fatal).

**Warm-reference margin.** To survive Co_z = 9 with a constant Pξ̄* needs
`r ≲ 0.72/9 = 0.08` — Pξ̄* within ~8% of local c² everywhere. The warm choice
Pξ̄* = max_z c² makes δc² one-signed but the largest deviation is
`(max−min)/max ≈ 0.36` on the Dunion sounding (c² ∈ [0.78,1.22]·mean), giving
`r·Co_z ≈ 3.3 ≫ 0.72` — **1a fails by ~4.5×**. It would survive only for a
near-isothermal reference, not the TC. **The LOCAL-profile operator (the
1/Pξ̄-weighted mass of §2 — the p′-form carrier of the same locality the φ-solve
gets from its weighted stiffness) is load-bearing for variant 1 exactly as for
the vertical solve — 1a is rejected, 1b (the direct 2-D local-profile solve) is
the design.** (Tanguay, Robert &
Laprise 1990, `~/Downloads/mwre-...1970...pdf`, need only be consulted if 1a
becomes live; it does not.)

---

## 5. Staging audit (findings tension #3; plan item 4)

Predictor level (X*): AB3 of the acoustic REMAINDER (`expdot`, unchanged), plus
the AI2* explicit histories `−1.0 Lⁿ + 0.75 Lⁿ⁻¹` applied per column
(`mc_driver!` for the fresh legs, `horizontal_si_history!` for the horizontal
channel). **Nothing at STATE amplitude is staged outside the 2-D solve as a
predictor term** — that was the structural DG failure (part 2: `ν·A(Xⁿ)` outside
the weak solve left ε·ν·A of the state-amplitude fast operator explicit; the
ε-analysis §4 is the check the plan's item 4 asks be reused).

| leg | recovered by | history discipline | reason |
|-----|--------------|--------------------|--------|
| p′  | the weak 2-D solve (solve variable) | fresh, staged through the same weak divergence chain (mc_driver impdot pattern, :834–883) | the solve variable's own operator; z-only x-coeffs make it clean |
| w (φ) | strong ∂z p′^{n+1} = the applied increment (II) | **stored applied increment** `(w^{n+1}−w*)/Δτ` (moist_compressible:1307) | vertical, z-dependent Pξ̄ + lid Dirichlet — the 2026-07-16 w-leg locus; the stored increment is self-consistent (history = operator the solve applied) so `−∂z p′^{n+1}/ρ̄_t` here is the GOOD case, unlike the mismatched fresh `−∂z p′ⁿ` staging (NaN Co_z 4.5) |
| u   | strong ∂x p′^{n+1} (I) | fresh (from the `pp_x`/`u_x` grid slot) | horizontal, z-ONLY coefficients — staging is exact from the grid slots by construction (handoff §Option A); A/B against stored-increment is a Stage-1 lever, not a correctness issue |
| ρ_d, ρ_t, E_t | strong derivatives of solved coefficients (flux form) | fresh (slaved slots) | z-only x-part; the flux-form slaving mirrors the solve pointwise |

The self-consistency point is the crux and resolves an apparent contradiction:
storing `(w^{n+1}−w*)/Δτ = −∂z p′^{n+1}/ρ̄_t` looks like the fatal φ-form staging,
but here it IS the operator the unsplit solve applied to w (there is no separate
weak φ-solve to mismatch against), so it is the stored-increment discipline, not a
re-staging. The division of labor 3(a) establishes: the HISTORY levels tolerate
strong (grid-slot) staging — the passing 3(a) configuration uses strong histories
at every ε — while the RECOVERIES must stay weak-consistent with the solve
operator (the cautionary strong-recovery variant re-grows to max|G| 1.10 at
ε 0.2) — the M1/M1 rule of §3. Note this audit is variable-level and did not
depend on the first draft's (erroneous) claim that the p′ vertical block is
byte-identical to the φ-solve's matrix; under the corrected §2 form the table
stands unchanged.

---

## 6. Sizing (deliverable C) — production dimensions from the code

Spline dims (Springsteel `factory.jl`: `_spline_k_dims`/i-dims give `Dim =
cells·mubar`, `bDim = cells + 3`, `mubar = 3`). Production TC (`tc/tc_params.jl`):
50 radial cells × 84 vertical cells per patch ⇒ **b_iDim = 53, b_kDim = 87**,
`N = b_iDim·b_kDim = 4611` unknowns per patch (XZ prototype and each axisym/RLR
patch). Confirmed by building the nest: axisym RiRk and every RLR patch report
b_kDim = 87, b_iDim ∈ {53,54,55}.

**Bandwidth.** Cubic-spline `M1ᵀWM1`/`M0ᵀWM0` half-bandwidth = 3 (measured). In
the existing i-fast coefficient layout (`(k_b−1)·b_iDim + i_b`) the vertical block
couples k_b±3 ⇒ flat ±3·b_iDim, the horizontal couples i_b±3 ⇒ flat ±3, so the
**half-bandwidth = 3·b_iDim + 3 = 162**. (The plan assumed the k-fast layout,
bw ≈ 3·b_kDim = 264; keeping i-fast is a free ~40% bandwidth reduction and is
adopted.)

| quantity (i-fast, bw=162) | value |
|---------------------------|-------|
| unknowns N | 4611 |
| half-bandwidth bw | 162 |
| banded-LU factorization (once, setup) | ≈ 2.4e8 flop |
| back-substitution (per step, one RHS) | ≈ 4.5e6 flop |
| LU storage, pivoted `(2·bw+bw+1)·N·8` | ≈ 18.0 MB |

Back-substitution ≈ 4.5e6 flop/step is negligible next to a transform sweep
(the phase-1 sweep was 9% of a step). Factorization is a one-time setup cost.
(The plan's ~6e8 factorization / ~2e6 solve figures were for the k-fast bw≈350
and an un-pivoted count; the i-fast pivoted numbers above supersede them.)

### RLR per-wavenumber memory verdict

RLR azimuthal is Fourier per ring; ring ri supports wavenumbers 0..ri
(`_cyl_ring_dims`, factory.jl:158). Per patch the max wavenumber = patchOffsetL +
iDim (max global ring). Built from the production nest:

| patch | b_iDim | bw | max wavenumber n | per-n LU | all-n stored (full) | ragged (~½) |
|-------|--------|----|------------------|----------|---------------------|-------------|
| 1 (inner, 3 km) | 53 | 162 | 150 (151 n) | 18.0 MB | 2.71 GB | ~1.4 GB |
| 2 (6 km) | 55 | 168 | 228 (229 n) | 19.3 MB | 4.43 GB | ~2.2 GB |
| 3 (outer, 12 km) | 54 | 165 | 264 (265 n) | 18.6 MB | 4.94 GB | ~2.5 GB |

Each wavenumber n is an independent 2-D (r,z) banded LU of the same N and bw
(radial B-spline × vertical B-spline), parallel over n. Storing ALL factorizations
up front costs ≤ 4.9 GB on the busiest patch (outer, full-size upper bound); the
ragged truncation (mode n lives only on rings ri ≥ n, so its radial extent
shrinks) roughly halves this to ~2.5 GB. **Verdict: feasible.** The TC design is
node-per-patch (20-core SLURM nodes, `tc/scythe_tc_multinode.sbatch`), so ~2.5 GB
of precomputed factorizations per node sits comfortably in node RAM. No 1a
fallback is triggered on memory grounds.

**1a-fallback decision rule (recorded):** adopt 1b (this note). Trigger the 1a
fallback (constant warm-reference Pξ̄* ≥ max_z Pξ̄, per-mode 1-D horizontal solves)
ONLY if BOTH (i) the RLR all-n factorization storage exceeds a node's budget
(here ~2.5 GB ≪ budget — not triggered) AND (ii) a stratified-base sweep at Co_z 9
shows the warm-reference 1a is stable (§4: it is NOT — fails ~4.5×). Both gates
fail for 1a, so 1b stands unconditionally for the production grid.

---

## 7. Open questions carried to Stage 1

1. **Operator form — SETTLED by review + part 3(c)**: the P⁻¹-scaled
   weighted-mass form (§2) is the recommendation — exact AND self-adjoint. The
   literal PDD form remains the documented fallback (banded LU needs no
   symmetry); the DPD-on-p′ hybrid is forbidden (commutator error, part 3(c)).
   The A≡0 unit test is on the ISOTHERMAL base (§3(i), ≤1e-10); the stratified
   comparison is truncation-level only.
2. **The stratified p′-primary vertical ceiling is a NEW measured quantity**:
   Stage 1's sweeps (G1's Co_z 9 stratified case with the 2-D solve active)
   establish it — expectation: matches the φ-solve's ≈ 9–18 (both consistent
   self-adjoint exact eliminations; 3(a) ε-insensitive at the symbol level),
   but measured, not inherited.
3. **u-leg history** — fresh (grid slot, plan default) vs stored applied
   increment. Both are correctness-neutral per §5; one A/B round on the resting
   sweep if the default disappoints (mirrors the vertical/phase-1 discipline).
4. **Sign convention** — (H′) is naturally all-positive
   (`+wtdMass + Δτ²Stiff = load`, SPD); the existing vertical solve uses the
   negated equivalent (`−Stiff − Mass`). One global convention, fixed at
   implementation before the A≡0 test is meaningful. Mechanical.
5. **Boundary-load sign/scale** — the natural-BC load terms `+Δτ·φ*` (lid) and
   `+Δτ·ρ̄_t u*` (wall) — Pξ̄-FREE under the scaled form (§3-BC) — must be
   verified against the by-parts boundary flux of the chosen global sign
   convention (a one-column analytic check before wiring the 2-D solve).
6. **Weighted-mass quadrature accuracy** — `M0ᵀ(W/Pξ̄)M0` integrates `1/Pξ̄(z)`
   with the same 3-point Gauss rule the weighted stiffness uses for Pξ̄(z);
   same-order accuracy is expected (smooth reference profile), but confirm the
   isothermal A≡0 test and the stratified sweep see no quadrature artifact.
7. **Axisym metric** — this note is XZ (RiRk). The axisym radial metric (the TC
   actually needs it) and the RLR per-n solves are Stage 3; the boundary-row
   structure (§3-BC) already anticipates the axis-regularity rows.

---

## 8. Stage 1–2 implementation outcomes (2026-07-17; code wins, documented)

`src/exact_si.jl` implements the above behind `options[:exact_si]`. All Stage-2
gates pass. Five places where the CODE corrected or refined this note:

1. **Every fast leg must be recovered from the ONE 2-D solve** (not a hybrid).
   §5's audit envisioned the vertical legs completing through the existing
   per-column φ-solve with the horizontal absorbed into its predictors. Measured:
   every composite that routed some legs through a SEPARATE solve (the φ-solve,
   or u through its own per-level Helmholtz threaded via the z-solve, or u from a
   strong ∂x of the merged p) left an O(state) fraction of the fast operator
   staged outside a solve and grew at e-fold 12–40 s for Co_h ≥ 3 × Co_z ≈ 2 —
   the ε-chain mechanism of the DG round in a new guise, invisible to the
   symbol-level VN because it lives in the fit-chain/boundary mismatch. The
   fix — u, w, p, and the slaved ρ_d/ρ_t/E_t ALL recovered from the single
   solved p̂ (u,w by M1x/M1z; the slaved legs by the pointwise divergence
   identity `div^{n+1} = −δp/(Δτ Pξ̄)` plus the fitted φ^{n+1}) — is the
   consistent-weak composition VN 3(a) proved ε-insensitive, and decays cleanly
   to Co_h 18 on both bases.
2. **The u-leg history default is `"stored"`, not `"none"`** (§5/§7 item resolved
   by the G2 A/B round). θ=1 ("none") is stable but its first-order component
   leaves an **~11% max|u| bias** on the BF02 dry bubble at BOTH ts-factors
   (constant in ts ⇒ not truncation). The stored applied-increment history (full
   2nd-order AI2*) brings every extremum to ≤ 1.1%/0.6% of the explicit
   reference. Unlike the rejected ADI sweep, the stored u-history in the UNSPLIT
   solve carries no leak (VN 3(a): the fresh strong u-history is ε-insensitive).
3. **Round-trip the u*/p′* predictors through their own vertical fit chains
   before the solve** (P = eval∘fit). The raw AB3 predictors carry z-grid-scale
   content the model's read chains never see; feeding it to the solve leaks the
   (I−P) l_q-refit residual into the implicit operator at state amplitude — the
   DG round's "round-tripped baseline" lesson, and it recurs here (top-localized
   growth, e-fold ≈ 30 s) until the predictors are round-tripped.
4. **Lid rows: strong ∂z-value rows + Neumann-basis lid data** (§3-BC refined).
   The natural (load-only) lid of §3-BC left the p̂ lid weakly controlled and fed
   a slow surface/lid u–p̂ inconsistency. Adopted: replace the first/last vertical
   coefficient planes with strong rows `Σ ∂zψ_kb(z_bnd) p̂ = φ*(bnd)/Δτ`
   (`options[:xsi_lid_rows] = "strong"`, default), and evaluate the lid φ* VALUES
   through p's NEUMANN column basis (w's Dirichlet basis forces them to zero,
   losing the inhomogeneous-Neumann data). Both cut the surface growth.
5. **A≡0 is a plumbing bypass, not operator equivalence; §3(i) roundoff-on-
   isothermal was optimistic.** The implemented A≡0 test (`options[:exact_si_zero_x]`)
   bypasses the 2-D solve to the per-column φ-solve and is **bitwise 0.0** — it
   validates the two-phase orchestration and flag-off, and IS the ≤1e-10 gate.
   But the OPERATOR-level check (`options[:exact_si_ax0]`: the 2-D operator with
   ∂xx disabled vs the φ-solve) is **0.012 on isothermal**, surface-localized
   (k=1–4: 0.025→0.014, decaying to 0.007 mid-domain) — NOT roundoff. Cause: the
   p′-primary form carries inhomogeneous-Neumann lid/surface rows while the
   φ-solve carries Dirichlet-φ rows, so §3(i)'s "same discrete D + BCs ⇒
   roundoff" premise does not hold (the BCs differ by construction). The p′-primary
   vertical scheme is therefore validated NOT by equivalence but by its OWN
   measured ceiling (below) — the "new measured quantity" §3 flagged. This is a
   genuine deviation from the Stage-0 §3(i) prediction; §3(i) should be read as
   "roundoff-equal ONLY if the boundary rows are also identical", which they are
   not for the p′-form.

### Measured gate numbers (600-s sweeps unless noted; RiRk XZ)

- **A≡0 plumbing** (isothermal, ∂xx bypassed vs vertical-only, 20 steps):
  **0.0** (bitwise). Flag-off bitwise vs 2ca710c: **0.0**.
- **G1 stability** (`hsi_ceiling_sweep.jl --exact-si`, both bases): decay at every
  Co_h ∈ {1.5, 3, 4.5, 9, 18} — e.g. stratified Co_h 4.5 → 1.5e-5, Co_h 9 →
  1.7e-5 from seed 6.5e-5. Growth probe (Co_h 3, 600 s): flat, t20 3.8e-5 → t600
  1.3e-5 (no leak). **PASS.**
- **p′-primary vertical ceiling (NEW measured quantity)** (`si_ceiling_sweep.jl
  --exact-si`, stratified): Co_z 9.05 → max|w| 3.2e-7 from seed 7.3e-5 (decay
  ×226), matching the φ-solve's Co_z 9–18. **The p′-primary form inherits the
  vertical ceiling** — measured, not assumed.
- **G2 accuracy** (BF02 dry, exact-si stored vs off): ts-factor 0.5 →
  max_w +1.1%, min_w −0.35%, max|u| +0.44%; ts-factor 1.5 → +1.8% / +0.36% /
  +0.58%. All within 2%. **PASS** (the ADI measured 8–20% / 20–55%).
- **G4 combined** (Co_z 9 × Co_h 4.5, stratified): decay ×5.2. **PASS.**
- **Cost** (`step_cost_profile.jl`): the solve is 8.6 ms (TC-nest1 50×100) /
  11.6 ms (o01 128×50) per step ≈ **11% of a step** — comparable to and net
  cheaper than the phase-1 ADI sweep (12–21% here). Uses a **sparse (banded) LU**
  (the plan's banded form, brought forward from Stage 3 to meet the gate; the
  dense factorization was 40–60% of a step).
- **G3 regression**: full suite **7655/7655**; 2-worker invariance 2.1e-12
  (≤ 1e-10); the two flag-error tests pass.

---

## 9. Stage 3 — axisymmetric (cylindrical r–z) radial metric (2026-07-18)

The axisymmetric TC (`moist_compressible_axisym`, RiRk grid, i-coordinate = r)
extends the XZ solve by the cylindrical radial Laplacian `(1/r)∂r(r∂r·)` and the
cylindrical volume element `r dr dz`. The elimination §1 and the P⁻¹-scaled
weighted-mass form §2 are unchanged in structure; the ONLY geometry switch is an
`r`-weight on the RADIAL Galerkin blocks and load. Everything vertical is
byte-identical (the vertical solve is geometry-agnostic).

**Operator.** Re-weight only the radial (i) blocks by r (the radial coordinate at
the i-mish points, which are Gauss points so never at r = 0):

    Mᵣ = M0ᵣᵀ (W·r) M0ᵣ        (r-weighted radial mass)
    Sᵣ = M1ᵣᵀ (W·r) M1ᵣ        (r-weighted radial stiffness — the cylindrical
                                Laplacian by-parts ∫(∂rψ)(∂rφ) r dr)
    A_axisym = kron(Mzw, Mᵣ) + Δτ²( kron(Sz, Mᵣ) + kron(Mz, Sᵣ) )

The vertical blocks (Mzw = M0ᵀ(W/Pξ̄)M0, Sz, Mz) are unchanged; the r-weight lives
entirely in the radial factor of every tensor block (the volume element factors
as (r dr)·dz). The build function, the strong lid rows, and the whole
solve/recovery structure are geometry-agnostic once Mᵣ/Sᵣ replace Mx/Sx.

**Load.** The radial quadrature carries the cylindrical volume weight (Wr → Wr·r);
the divergence gains the radial metric term u*/r (the strong linear radial
divergence `lindiv_r u* = ∂r u* + u*/r`, = `mc_linear_div!` on `MCAxisymRZ`). Both
the p′* identity term and the divergence term are multiplied through by r.

**Boundary (§3-BC, r-weighted).** The radial-Laplacian by-parts flux is
`[ψ·Δτ²·r·∂r p′]`, so the u-Dirichlet wall load is r-weighted by the wall radius:

- **r = 0 axis: NO explicit row.** The by-parts flux `[ψ·r·∂r p′]_{r=0} = 0`
  vanishes by the r-weight (cylindrical regularity for the axisym/n=0 problem) —
  a simplification vs Cartesian, which needed a wall row at both ends. In code the
  inner-wall load is weighted by r = iMin, which is 0 on the axis and kills it
  automatically (and correctly r-weights a nested inner wall at iMin > 0).
- **Outer wall r = R: u-Dirichlet** load r-weighted by R = iMax
  (`+Δτ·R·ρ̄_t u*`).
- **Lid/surface z: unchanged from Cartesian** (strong ∂z-value rows).

**Recovery legs are geometry-free.** u = u* − (Δτ/ρ̄_t)∂r p′ (the M1ᵣ operator, no
1/r); w/φ vertical recovery unchanged; and the slaved ρ_t/ρ_d/E_t legs use the
pointwise pressure identity `δρ_t = δp/Pξ̄ = −Δτ·D` — because the solved p′ came
from the cylindrical operator, δp/Pξ̄ automatically equals −Δτ times the
*cylindrical* mass-flux divergence D, with ZERO code change to the recovery. (The
plan's note that the slaved legs "use the cylindrical divergence" is realized
through this identity, not an explicit divergence recompute.)

**Cartesian bitwise.** The XZ path is a clean `else` branch in each of the three
touched spots; the wall-load r-weights are 1.0 (`ts_term·1.0 ≡ ts_term`), so
`options[:exact_si]` on XZ is byte-identical to `37740a4`.

### Measured Stage-3 gate numbers (axisym, RiRk r–z)

- **G3-unit / A≡0 plumbing** (`exact_si_zero_x`, isothermal axisym, iMin = 0,
  20 steps vs vertical-only): **0.0** (bitwise) — the vertical solve is
  geometry-agnostic, so this holds exactly as in XZ.
- **G3-stability** (broadband u+w seed, resting axisym tile, iMin = 0 so the r = 0
  axis column is exercised, Co_h 3, 300 s): decay on BOTH bases — isothermal
  7.4e-5 → 2.1e-5, stratified 7.4e-5 → 2.9e-5. Vertical-ceiling no-regress
  (Co_z ≈ 9 stratified, wide radial cells, iMin = 50 km): 5.5e-5 → 7.3e-6
  (decay ×7.5). **PASS.**
- **Operator correctness — large-R convergence to XZ** (one-step p′ increment,
  identical seed/reference): the axisym solve → the XZ Cartesian solve as the
  metric vanishes, with clean O(1/r) scaling: rel diff (du/dw/dp) ≈
  6.6e-5/4.4e-5/1.2e-4 at R₀ = 1e6 m, ≈ 6.6e-7/4.4e-7/1.2e-6 at R₀ = 1e8,
  ≈ 5.9e-9/5.5e-9/1.4e-8 at R₀ = 1e10 — a wrong r-factor or sign would not
  converge. This is the strongest correctness check (not in the plan's gate list;
  added this session).
- **G3-regression**: full suite **7664/7664** (7655 baseline + 9 new axisym
  exact-SI assertions); XZ `exact_si` gates unchanged (bitwise `else` branches);
  flag-off untouched (all changes behind `options[:exact_si]` / axisym-only
  methods).
- **G3-payoff (nested axisym TC restart, `exact_si` on vs off through a CAPE
  release): DEFERRED to Stage 5.** `exact_si` is blocked in nested runs
  (`nesting.jl:187-188`) until Stage 5 wires the nested master solve, so the
  literal nested-restart payoff cannot run at Stage 3; Stage 5's readiness gate
  (nested `exact_si` TC through a CAPE release) IS this comparison. The user also
  flagged (2026-07-18) that the comparison is confounded by a TC-initialization
  moisture inconsistency (spurious near-surface/top condensate, noisy Q_ss at
  t = 0 from the first-order thermal-wind integration and a T/E_t/Q_ss retrieval
  mismatch), so a crash reproduction would not cleanly attribute to the acoustic
  solver regardless. Diagnosis of that is the user's separate deferred session.
