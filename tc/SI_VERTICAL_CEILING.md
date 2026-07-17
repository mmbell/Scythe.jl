# The vertical-acoustic timestep ceiling of the mc semi-implicit solver

Handoff note, 2026-07-16. Status: **FIXED (same day)** — the explicit acoustic
mode was removed and the AI2* staging made operator-consistent, moving the
measured ceiling from Co_z ≈ 2.1–2.4 to ≈ 9–18. See "The fix" at the end; the
diagnosis and measurement history below is kept for the record (its
"candidate fixes" section is superseded).

## The fix (2026-07-16)

The subtract-AB3/add-implicit structure was removed entirely (`:semiimplicit
=> false` is now an error for the mc sets):

- `expdot` stages the acoustic REMAINDER (full tendency minus the pointwise
  reference-linear term, cancelling analytically at grid scale), so AB3 never
  integrates any part of the linear vertical acoustic operator.
- The AI2* histories for p/ρ_d/ρ_t/E_t are staged in `mc_driver!` from ONE
  fit of φⁿ = ρ̄_t wⁿ in w's column basis — the same discrete chain the
  Helmholtz solve and the slaved recoveries apply — freshly evaluated each
  step (robust to diffusion/collar/refit between steps). A dry reference
  gives bitwise-identical ρ_d/ρ_t histories (c_d = 1, c_d,z = 0 exactly), so
  the historical drift objection to fitted staging is void.
- The w-leg history is the increment the Helmholtz elimination ACTUALLY
  applied, stored by the adjustment ((w^{n+1} − w*)/Δτ): the weak-Galerkin
  elimination is not expressible as a pointwise chain, and staging −∂z p′/ρ̄_t
  instead leaves its residual under explicit weights (ceiling stays ≈ 2–3;
  measured). Recovering w from a refit ∂z p′^{n+1} (moving the residual to the
  p leg) is WORSE (NaN at Co_z 4.5 in ~20 s; measured).
- Corrections to this note's original framing, found during the fix: the
  spline filter `_filter_mish!` runs inside `spectralTransform!` (every step,
  not on the output cadence), and it was configured OFF in these runs anyway —
  the "filter" that matters is the Ooyama `l_q` regularization inside EVERY
  `SAtransform!` fit. It is load-bearing: with `l_q = 0` every variant blows
  up (the scheme's damping composition relies on it), and histories must be
  evaluated on the post-refit state (all-stored histories go unstable at
  Co_z 9 where fresh staging survives).

Measured after the fix (`model_tests/si_ceiling_sweep.jl`, same dry isothermal
base, horizontal Courant held ≤ 0.35): Co_z 3.0/4.5/9.05 decay (×100–300 over
600 s); Co_z 18 grows slowly (e-fold ≈ 62 s); Co_z 36 NaNs. The vertical
ceiling (≈ 9 with solid margin) now sits ABOVE the explicit horizontal
acoustic limit (AB3, ω·Δt ≈ 0.72) for every realistic grid aspect ratio, so
the horizontal Courant is the binding constraint — per-nest timestep scaling
with DX is meaningful again (`tc/tc_params.jl` NEST_TS = [0.75, 1.5, 1.5]).
The regression gate lives in test/test_moist_compressible.jl ("SI
vertical-acoustic ceiling removed"), asserting decay at Co_z 4.5 and 9.
`warn_timestep_stability` now checks the horizontal Courant for mc runs.

## Second mechanism, found immediately after (2026-07-16, same day): the
## reference-state (SHB78) instability of the mean-c² linearization

The first TC run at NEST_TS [0.75, 1.5, 1.5] died within ~1 min (ρ_d → −0.9 on
nest2, surfacing as a `log` DomainError in the Louis-BL entropy staging). The
operator-consistent scheme above was validated on an ISOTHERMAL base — uniform
c — which is structurally blind to a second explicit residual: `Pxi_bar` was
Springsteel's DOMAIN-MEAN `mean(γp̄/ρ̄_t)`, so wherever the local reference c²
deviates from that mean (±20–25% on the Dunion sounding between the 300-K
surface and the 195-K tropopause) that fraction of the grid-scale vertical
acoustic operator stays EXPLICIT under AB3. This is the classic
reference-state semi-implicit instability (Simmons, Hoskins & Burridge 1978).
It never bit before because ε·Co_z ≈ 0.25·1.8 < 0.72 at ts = 0.3; at ts = 1.5
(Co_z 9) it is ≈ 2, fatal in tens of steps. Minimal reproduction: the sweep's
`:stratified` base (Dunion-like lapse) NaN'd at Co_z 9 in 232 s while the
isothermal base decayed — no physics, no nesting, single patch.

Fix (same day): the acoustic linearization uses the LOCAL profile
`Pξ̄(z) = γ̄_m(z)·p̄(z)/ρ̄_t(z)`, computed in `mc_reference_diagnostics` through
the model's own retrieval pipeline and used everywhere the scalar was: the
expdot remainder, the impdot staging, the p′ recovery, and the Helmholtz
operator — which becomes `∂z(Δτ²Pξ̄(z)∂z·) − I`, assembled on RiRk as the
symmetric weighted-stiffness Galerkin form `M1ᵀ(W·Pξ̄)M1` (a new
profile-coefficient method; the scalar path is untouched so the legacy sets
stay bitwise) and on RZ as the exact collocation product `M1·(M0\(Pξ̄·M1))`.
After the fix the stratified base decays at Co_z 9 exactly like the isothermal
one; the regression testset gained the stratified case. Lesson recorded: any
future SI stability claim must be tested on a STRATIFIED base — an isothermal
base cannot see reference-state errors in the linearization coefficients.

---

Original note (2026-07-16, pre-fix) follows.

## Session-close state of the TC effort (2026-07-16)

All seven stages of the TC plan are CODE-COMPLETE and committed on
`development` (MP rain DSD `aaeb981`; Louis BL + Smagorinsky `f013c3e`;
surface fluxes `cfc1a09`; balanced vortex init + tc/ `352120e`; ts fix +
sbatch `dcf1247`; RLR nesting `9e144bf`, `8455660`; rho_d rate guards
`23cf85b`; this doc `e824642`). Springsteel `feature/rlr-tiling` carries
`5f02f11` (RLR collar evaluator, unstructured-eval sine-sign fix,
coupled-border cache) and `f988eff` (tiled RLR splineTransform! R3X ahat
reload — required by Scythe's RLR nesting). Suites at close: Scythe
7637/7637, Springsteel 41220/41220. The nested-RLR vs nested-axisym
equivalence gate (`model_tests/nested_rlr_equivalence.jl`) passes at ~1e-10
pointwise.

**Deliberately deferred, in order:**

1. **Fix this SI ceiling** (below) so the TC runs at a larger timestep before
   burning wall clock on long integrations.
2. **Rerun the 6-h laptop validation** (`julia --project tc/tc_run_axisym.jl
   21600 --csv`). The first attempt at ts = 0.3 reached t ≈ 5 h and showed a
   healthy quiet spin-up (BL inflow to −1.3 m/s, v 15 → 12 m/s under drag)
   followed by a vigorous but resolved first CAPE release (updraft annulus at
   r ≈ 10 km, axis downdraft −28 m/s, smooth in r and z — NOT an axis
   instability) — it died at rain onset from an unguarded ρ_d undershoot in
   the rain rates, now fixed (`23cf85b`, RHO_D_MIN floors). A rerun with the
   guards was killed by choice to fix the SI first; expect the eruption
   near t ≈ 4.5–5 h.
3. **Server production**: 5-day axisym via `tc/scythe_tc.sbatch` (single
   20-core node) or `tc/scythe_tc_multinode.sbatch` (node per patch —
   run a 1-h test job first); then the 3D run via `--rlr`.

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
