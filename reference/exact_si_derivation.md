# Variant 1 (unsplit XZ acoustic SI): p′-form derivation, assembly, boundary rows, staging, sizing

Stage-0 derivation note, 2026-07-17. Deliverable A/B/C of the Stage-0 section of
`reference/exact_si_plan.md`. NO src/ changes are made in Stage 0; this note plus
the extended `model_tests/hsi_dg_von_neumann.jl` (part 3) are the whole output.
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
- The vertical block `Pξ̄ ∂zz` has Pξ̄ **outside** the derivatives. This is NOT
  the self-adjoint `∂z(Pξ̄ ∂z·)` the validated vertical solve uses
  (`M1ᵀ(W·Δτ²Pξ̄)M1`, §2). The two differ by the term `(∂z Pξ̄) ∂z p′`. **This is
  the central deviation from the handoff sketch** and is resolved in §2/§3.

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

- **Vertical weighted stiffness** `−M1ᵀ(W·Δτ²Pξ̄)M1` — the self-adjoint
  `∂z(Δτ²Pξ̄(z) ∂z·)` (`_assemble_spline_matrix` profile method,
  src/semiimplicit.jl:1966; used by `calc_Helmholtz_semiimplicit_matrix` profile,
  :2126). This is the operator the vertical φ-solve is built on and validated to
  Co_z ≈ 9–18.
- **Horizontal (per level) stiffness** `Δτ²Pξ̄(z_k)·M1ᵀW M1` on u's i-basis
  (`build_set`, src/horizontal_si.jl:128 → `_assemble_spline_matrix` scalar,
  :1928), assembled once per physical level with the level's Pξ̄(z_k).
- **Mass** `M0ᵀW M0` (the `−I`).

### The operator we assemble (recommended, operator-consistent)

Rather than the literal `Pξ̄ ∂zz` of (H), assemble the vertical block as the
**self-adjoint** `∂z(Δτ²Pξ̄∂z·)` — the vertical solve's exact matrix — and the
horizontal block as `Δτ²Pξ̄(z_k) ∂xx`. On the tensor basis, with `Sx = (M1ᵀWM1)_i`,
`Sz = (M1ᵀWM1)_k`, `Mx = (M0ᵀWM0)_i`, `Mz = (M0ᵀWM0)_k`, and `⊗` the tensor
product in the (i_b, k_b) layout:

    A = (Mx ⊗ Mz)                         [the −I mass; sign per _assemble* below]
      + Δτ² [ (Sx ⊗ Mz)·Pξ̄_level        [horizontal Pξ̄(z_k) ∂xx]
            + (Mx ⊗ Sz_weighted) ]        [vertical ∂z(Pξ̄ ∂z·) = weighted Sz]

where `Sz_weighted = M1ᵀ(W·Δτ²Pξ̄)M1` carries Pξ̄(z) INSIDE the vertical quadrature
(the level index is the quadrature point), and `Sx ⊗ Mz` is scaled per vertical
block by `Δτ²Pξ̄(z_k)` (Pξ̄ constant along x within a level). Signs and the `−I`
follow `_assemble_spline_matrix` exactly: vertical `A = −Stiff_weighted − Mass`
(β = −1, `calc_Helmholtz_semiimplicit_matrix`, :2114/:2129); horizontal
`A = +Δτ²Pξ̄·Stiff + Mass` (β = +1, `build_set`, :128). The 2-D operator is the
consistent superposition on the shared p′ coefficients (one global sign
convention, fixed at implementation to match the vertical solve's residual).

**Symmetry.** `Sx, Sz_weighted, Mx, Mz` are symmetric; the tensor operator is
symmetric before boundary rows. The inhomogeneous-Neumann/Dirichlet boundary
rows (§3) break symmetry on those rows only (as the 1-D solves already do —
`factorize(A)` is a general banded LU, not Cholesky; `_assemble_spline_matrix`
returns `factorize(A)`), so Stage 1 uses a banded LU, not a symmetric factor.

### Why this operator and not literal `Pξ̄ ∂zz` (deviation flagged)

The literal (H) vertical block `Pξ̄ ∂zz` (Pξ̄ outside) is an equally-consistent
O(Δz²) discretization, but it is NOT the operator the vertical solve validated,
and it is non-self-adjoint when Pξ̄ varies. Choosing the self-adjoint weighted
stiffness (a) makes the ∂xx→0 restriction reproduce the vertical solve exactly
(§3, the Stage-1 unit test), (b) inherits the measured Co_z ≈ 9 ceiling by
construction, and (c) reuses `M1ᵀ(W·Pξ̄)M1` unchanged. The price is that the
assembled operator differs from the pure p′-elimination by the vertical
`(∂z Pξ̄) ∂z p′` term — physically the reference-stratification advective piece
the φ-form already carries and validated. **Recommendation: assemble the vertical
block as the weighted stiffness. The literal-p′ form is the documented
fallback if the Stage-1 unit test (§3) exposes an unexpected obstruction.**

---

## 3. RHS staging, leg recovery, and the ∂xx→0 vertical-equivalence

### RHS

The load of (H) is `p′* − Δτ Pξ̄ (ρ̄_t ∂x u* + ∂z φ*)`, formed on the patch
coefficients through the model's read chain (`_hsi_to_levels!`,
src/horizontal_si.jl:145 — per-z_b i-fit→evaluate, then per-i vertical
fit→evaluate, the `gridTransform` chain, so the solve sees the same discrete
state the grid slots see) then reduced to the Galerkin load `M0ᵀW·rhs` with the
boundary rows treated per §3-BC. The two divergence pieces are exactly the
existing reads: `∂x u*` (as in the phase-1 load, :247) and `∂z φ*` (the vertical
solve's `p_nstar_z`/`φ*` chain, moist_compressible.jl:1231–1238). Predictor X*
already carries the AB3 remainder and BOTH dimensions' AI2* explicit histories.

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

### ∂xx→0 reproduces the vertical φ-solve (Stage-1 unit test #1)

Zero the horizontal block (Sx→0). Column-by-column, (H) becomes
`(I − Δτ²∂z(Pξ̄∂z)) p′^{n+1} = p′* − Δτ Pξ̄ ∂z φ*` with the vertical operator
matrix = `−M1ᵀ(W·Δτ²Pξ̄)M1 − M0ᵀWM0` — **identical** to the vertical solve's
factorized matrix (`calc_Helmholtz_semiimplicit_matrix` profile). The two forms
differ only in the solved variable (p′ here, φ in `semiimplicit_adjustment_p`)
and are related by the exact recovery (II): solving the p′ system and recovering
`φ = φ* − Δτ ∂z p′` yields the same discrete φ^{n+1} as the φ-solve **provided the
lid boundary rows match** (below). Because the operator matrix, the mish
operators, and the l_q fit are byte-identical, the recovered
`(u,w,p′,ρ_d,ρ_t,E_t)` must agree to solver roundoff (~1e-12). **This exactness is
the property Stage 1 asserts as its first unit test** (`:exact_si` with ∂xx
disabled vs the vertical-only path, ≤1e-10 on the resting stratified base). If the
test shows O(∂z Pξ̄·Δz) drift instead, it means the p′-primary vs φ-primary
variable change does not commute with the discrete boundary handling at roundoff,
and the vertical block must be kept in φ-primary form with the horizontal added as
a correction (the fallback noted in §2).

---

## 3-BC. Boundary rows (the budgeted session-eating detail)

The vertical φ-solve works in **w's Dirichlet basis** (φ = 0 rows via `d.Nb`,
`_assemble_spline_matrix`:1972). The p′-form needs **inhomogeneous Neumann** rows
instead — the physical wall conditions expressed on p′ through the recoveries:

- **Rigid lid / surface** (z = z_t, z_b): `w^{n+1} = 0 ⇒ φ^{n+1} = 0`. By (II),
  `∂z p′^{n+1} = φ*/Δτ` at the boundary. In the weak Galerkin form this is a
  **natural** condition: integrating the vertical stiffness by parts yields a
  boundary flux `[ψ · Δτ²Pξ̄ ∂z p′]`, so the inhomogeneous value enters the LOAD
  as `+ Δτ Pξ̄ φ*` evaluated at the lid/surface node (no row replacement).
  Recovering φ then gives `φ^{n+1} = φ* − Δτ(φ*/Δτ) = 0` at the wall exactly —
  which is why the ∂xx→0 equivalence (§3) reproduces the Dirichlet-φ answer even
  though the p′ operator carries a Neumann row. **Note vs the handoff:** the lid
  BC is a load contribution, not a replaced row; the row stays the natural
  weak-stiffness row.
- **Side walls / axis** (x = x_l, x_r; r = 0 axis on axisym): u-Dirichlet
  `u^{n+1} = 0 ⇒ ∂x p′^{n+1} = ρ̄_t u*/Δτ` by (I). Natural weak condition; load
  contribution `+ Δτ Pξ̄ ρ̄_t u*` at the wall node. (For Neumann u walls — the
  other case `horizontal_si` supports, :120 — `∂x u = 0` gives a homogeneous
  natural row and no load term.) On axisym the r = 0 axis is u-Dirichlet
  (`tc_init.jl` axis_bc), same structure; RLR adds per-wavenumber axis-regularity
  rows in a later phase (Stage 3).
- **Corners** (lid ∩ wall): both load contributions add on the single corner
  coefficient; no interaction term (the two are independent first-order fluxes).

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
near-isothermal reference, not the TC. **The LOCAL-profile weighted-stiffness solve
is load-bearing for variant 1 exactly as for the vertical solve — 1a is rejected,
1b (the direct 2-D local-profile solve) is the design.** (Tanguay, Robert &
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
re-staging. 3(a)'s ε-insensitivity is the linear evidence that the weak-solve /
strong-recovery chain mismatch on this leg is tolerated in the unsplit form.

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

1. **Vertical block form** — assemble as the self-adjoint weighted stiffness
   (§2, recommended, exact ∂xx→0 equivalence) vs literal `Pξ̄ ∂zz`. The Stage-1
   unit test (§3, A≡0 vs the vertical solve, ≤1e-10) decides; expected to pass
   with the weighted stiffness.
2. **u-leg history** — fresh (grid slot, plan default) vs stored applied
   increment. Both are correctness-neutral per §5; one A/B round on the resting
   sweep if the default disappoints (mirrors the vertical/phase-1 discipline).
3. **Sign convention** — the vertical (`β=−1`) and horizontal (`β=+1`) 1-D
   assemblies use opposite mass signs; the merged 2-D operator needs one global
   convention chosen to match the vertical solve's residual. Mechanical, but must
   be fixed before the A≡0 test is meaningful.
4. **Boundary-load sign/scale** — the inhomogeneous-Neumann load terms
   `+Δτ Pξ̄ φ*` (lid) and `+Δτ Pξ̄ ρ̄_t u*` (wall) must be verified against the
   by-parts boundary flux of the chosen global sign convention (a one-column
   analytic check before wiring the 2-D solve).
5. **Axisym metric** — this note is XZ (RiRk). The axisym radial metric (the TC
   actually needs it) and the RLR per-n solves are Stage 3; the boundary-row
   structure (§3-BC) already anticipates the axis-regularity rows.
