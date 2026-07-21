# The wall boundary condition sets the SI timestep ceiling (2026-07-21)

> # ⚠ CORRECTED 2026-07-21 (later): THE CEILING BELOW IS AN ARTIFACT
>
> **The d2 ceiling is not ts ≈ 0.75 s. It is between ts = 1.0 and ts = 1.25 s.**
> Every measurement in this note was taken on a reference state that CONDENSED AT
> REST — `expdot[p] = 2.3 Pa/s` everywhere, continuously, with zero perturbation and
> zero physics (see `tc/HANDOFF_REFERENCE_STATE.md`). That was a persistent
> grid-scale source feeding exactly the acoustic modes this note characterises, and
> the acoustic solve amplified it faster at larger ts. The instability attributed to
> the wall condition was largely the reference driving it.
>
> Note this document ALREADY CONTAINS the defect, in the section "A SECOND, smaller
> defect: a ts- and BC-independent resting drift" (`-2.3 Pa at 900 s, -4.5 Pa at
> 1800 s`). It was not smaller. It was the whole thing.
>
> Re-measured with `options[:consistent_qss_reference]` and
> `options[:hydrostatic_reference]` on, using a SEEDED resting column
> (`model_tests/tc_lid_drift_probe.jl`, `d2@<ts>+qss+hydro+seed`, 1e-4 w seed):
>
> | ts   | before (flags off) | after (flags on) |
> |------|--------------------|------------------|
> | 0.5  | max\|w\| 1.4e-3, drifting | **decays** 1.3e-5 → 1.0e-6 (1 h) |
> | 0.85 | **3.5e-1, unstable**      | **decays** 6.9e-6 → 9.5e-7 (1 h) |
> | 1.0  | —                         | **decays** 5.0e-6 → 7.6e-8 (3 h) |
> | 1.25 | —                         | 6.98 m/s @ 900 s → NON-FINITE @ 1909 s |
> | 1.5  | —                         | NON-FINITE @ 798 s |
> | 2.0  | —                         | NON-FINITE @ 52 s |
>
> **The UNSEEDED probe is worthless for this question and was what misled the
> re-measurement at first.** Once the reference is a discrete fixed point the
> unseeded column is bit-exact zero at EVERY ts tested, including 2.0 — there is
> nothing to grow, so it measures preservation, not stability. Always use `+seed`.
>
> Consequences for the rest of this note:
> - `NEST_TS = 0.5` has a 2× margin, not a 0.75× deficit. It can likely go to 1.0,
>   but a resting probe cannot license a production timestep — validate on
>   `tc_balance_holdtest.jl 12 nophysics` first (finite amplitude).
> - **The R1T1X rejection is void.** It was measured at ts = 1.0, which was above the
>   CONTAMINATED ceiling and below the true one, i.e. in a regime that no longer
>   exists. Re-take it.
> - The "This is the spurious cooling" section is still directionally right — but the
>   cooling had two contributors and the reference was one of them; the split has not
>   been re-measured.
>
> What survives unchanged: the quartet must share one wall condition (the slaving
> argument), the acoustic-vs-balance conflict analysis, and the R1T1X rationale.

Companion to `tc/SI_VERTICAL_CEILING.md` (resting-base operator consistency, fixed)
and `tc/SI_CONVECTIVE_CEILING.md` (the state-dependent ceiling). This note documents
a **third** ceiling, upstream of both, and identifies the spurious cooling aloft as
its symptom.

**Headline (SUPERSEDED — see the correction above): `SecondDerivativeBC` — the
2026-07-21 fix for the vortex drain — lowers the vertical-acoustic timestep ceiling
from ts ≈ 2.0 s to ts ≈ 0.75 s. The production configuration runs at ts = 1.0 s,
i.e. ABOVE its own stability ceiling.**

---

## The measurement

`model_tests/tc_lid_drift_probe.jl`. ONE axisymmetric patch on the production TC
vertical grid (50 cells to 25 km), the production Dunion reference state,
**zero perturbation** (the state is exactly the reference — an exact hydrostatic
steady state), no vortex, no physics, no diffusion. Anything that grows is numerical.

All four acoustic-quartet variables (`p`, `rho_t`, `rho_d`, `E_t`) carry the same
wall condition unless stated. 30-60 min of model time.

### Wall condition vs. stability

| walls (whole quartet) | ts   | max&#124;w&#124; from zero | verdict |
|---|---|---|---|
| `NeumannBC`          | 1.0  | 1.3e-3  | quiet |
| `NeumannBC`          | 1.5  | 1.5e-3  | quiet |
| `NeumannBC`          | 2.0  | 1.0e-2  | quiet |
| `NeumannBC`          | 2.5  | —       | **NON-FINITE at 625 s** |
| `NeumannBC`          | 3.0  | —       | **NaN at 246 s** |
| `SecondDerivativeBC` | 0.5  | 1.4e-3  | quiet |
| `SecondDerivativeBC` | 0.7  | 1.8e-3  | quiet |
| `SecondDerivativeBC` | 0.85 | 4.3e-1  | **unstable** |
| `SecondDerivativeBC` | 1.0  | 7.8e-1  | **unstable (production)** |

The sponge is *suppressing* the instability, not causing it: at `d2 @ ts = 1.0` with
`alpha = 0` the same run reaches max&#124;w&#124; = 3.0 m/s and is still growing at 1 h.
Turning off `:state_dependent_si` does not help either (1.08 m/s) — this is not the
convective/state-dependent ceiling.

### Mixing conditions across the quartet is far WORSE than either uniformly

At ts = 1.0, `d2` applied to a SUBSET with the rest Neumann:

| `d2` applied to | outcome |
|---|---|
| `p` only            | unstable, max&#124;w&#124; 0.21 |
| `rho_t` only        | **NON-FINITE at 26 s** |
| `rho_d`, `E_t`      | **NON-FINITE at 45 s** |
| `v` only            | quiet — identical to all-Neumann |
| all four (production) | unstable but survives (0.78 m/s) |

`v` is irrelevant; the quartet is everything. The semi-implicit slaves `rho_t'`,
`rho_d'` and `E_t'` to the SAME solved `phi` through one discrete chain
(`semiimplicit_adjustment_p`, `moist_compressible.jl:1386-1412`). If the four are
refit into bases with DIFFERENT wall constraints, the slaving relation is broken
differently for each and the coupled system desynchronizes immediately. **Whatever
condition is chosen, the quartet must share it.**

## Why Neumann is the acoustically consistent one

`w` carries `DirichletBC` at both walls, so `phi = rho_tbar w = 0` there for all
time, and the Helmholtz solve enforces exactly that (`_assemble_sd_helmholtz`,
`dirichlet = (true, true)`). The implicit pair is

    d(phi)/dt = -dp'/dz ,    dp'/dt = -Pxi(z) d(phi)/dz

Evaluated at the wall, `phi = 0` for all time forces **`dp'/dz = 0` at the wall for
the IMPLICIT ACOUSTIC PART**. That is precisely `NeumannBC` / R1T1. Gravity is not
in the implicit pair — the buoyancy term `-g rho_t'/rho_t` sits in the explicit
remainder — so the acoustic subsystem genuinely wants a homogeneous Neumann wall.

`SecondDerivativeBC` leaves `dp'/dz` free at the wall. The refit can therefore inject
a nonzero wall derivative every step into exactly the mode the Helmholtz solve cannot
see, leaving that component under AB3's explicit weights — the same mechanism as
`SI_VERTICAL_CEILING.md`, relocated to the wall, and with the same consequence: a
Courant-type ceiling where there should be none.

## Why the balanced vortex wants the opposite

The FULL `w` equation at the wall (where `w = 0`, so advection vanishes) gives the
exact compatibility condition

    dp'/dz |_wall  =  -g rho_t' |_wall

which is **nonzero and state-dependent** — it *is* the surface pressure deficit.
`NeumannBC` is the special case `rho_t' = 0`, which is why it destroys the balanced
vortex (0.10 m/s² hydrostatic residual, the drain of `HANDOFF_2026-07-20.md`).

**So the two requirements genuinely conflict for any HOMOGENEOUS condition:** the
acoustic solve needs the time-varying part of `dp'/dz` to vanish at the wall; the
balanced state needs a static nonzero `dp'/dz` there. `SecondDerivativeBC` buys the
balance and pays for it in timestep; `NeumannBC` does the reverse. Neither is right.

The resolution is the **inhomogeneous Neumann condition** `dp'/dz|_wall = -g rho_t'|_wall`,
which satisfies both: it is R1T1 (acoustically consistent, ceiling ts ≈ 2.0) with the
hydrostatic wall value instead of zero.

## This is the spurious cooling

`tc/output/tc_holdtest_nophysics` (12 h, `d2` walls, ts = 1.0, no diabatic physics,
max&#124;w&#124; = 0.04 m/s) cools ~1 K/h at 25 km and GAINS 2.7 %/h of dry-air density
at the top level, uniformly in radius and identically in all three nests:

    nest1, patch-mean dT over 12 h:  -12.6 K @ 25 km, -5.7 @ 21, -2.0 @ 15-13 km
    top-level mean rho_d: +32.1 %,  E_t: +35.7 %

Radius-independent and nest-independent = 1-D. `-5.7 K at 19.5 km in 11 h` in the
full-physics run (`HANDOFF_2026-07-20.md`) is the same number. The probe reproduces
the sign and the top-localization from a resting column at the production ts.

## A SECOND, smaller defect: a ts- and BC-independent resting drift

Every stable case above still drifts at the lid at the same rate:

    p'(top)      -2.3 Pa at 900 s, -4.5 Pa at 1800 s      (linear, ~ -9 Pa/h)
    rho_d'(top)  -0.09 % at 1800 s

identical for `d2@0.5`, `d2@0.7`, `neumann@1.0` and `neumann@1.5`. Independent of the
timestep => this is a **spatial** discretization error, not a time-integration one:
the exact hydrostatic reference is not a discrete steady state at the lid. It is
~15x too small to explain the 12 h run and is a separate, lower-priority item.

## What this does NOT change

- The 2026-07-21 diagnosis stands: `NeumannBC` really does destroy the balanced
  vortex, and that really was the dominant cause of the drain. The error was
  believing a homogeneous condition could serve both roles.
- Do not simply revert to `NeumannBC`. That trades the cooling back for the drain.

## Reproduce

    julia --project=. model_tests/tc_lid_drift_probe.jl 1.0 base nosponge neumann noSI halfts
    julia --project=. model_tests/tc_lid_drift_probe.jl 0.3 d2@0.7 d2@0.85 neumann@2.0 neumann@2.5
    julia --project=. model_tests/tc_lid_drift_probe.jl 0.3 "d2:rho_t@1.0" "d2:v@1.0"

    julia --project=. tc/tc_postprocess.jl --indir tc/output/tc_holdtest_nophysics
    julia --project=. model_tests/tc_cooling_probe.jl

---

## CONFIRMED in the full nested run (2026-07-21)

12 h `tc_balance_holdtest.jl nophysics`, identical in every respect except the
timestep (`SCYTHE_TC_TS_SCALE`), nest 3 patch-mean, first 7 h:

    ts = 1.0 (ABOVE the d2 ceiling)     ts = 0.5 (below it)
    t[h]  T@25km   rho_d(lid)           T@25km   rho_d(lid)
     0    231.98   3.3832e-02            231.98   3.3832e-02
     7    224.99   3.9440e-02            230.34   3.3327e-02
          -7.0 K    +16.6 %              -1.6 K    -1.5 %

**~77 % of the spurious cooling, and ALL of the lid mass gain, is the timestep
instability.** The sign of the mass drift even reverses. What survives at ts = 0.5
is the small ts- and BC-independent drift of the second defect above.

Output preserved: `tc/output/tc_holdtest_nophysics_ts05/`.

---

## The fix: R1T1X, and the feedback trap it walks into

`CubicBSpline.R1T1X` (Springsteel) is a rank-1 inhomogeneous Neumann condition:
the SAME `gammaBC` as R1T1 — hence the same admissible subspace and the same
solver stability — with the boundary derivative carried in the affine `ahat`
offset, set per column by `set_ahat_neumann!` / `set_wall_derivatives!`.

Verified (Springsteel suite 41237/41237, `test/r1t1x.jl`):
- the prescribed wall derivative is attained to ~1e-14;
- `du = 0` reproduces the homogeneous R1T1 fit BITWISE;
- `gammaBC` is element-wise identical to R1T1's;
- the response is exactly linear in `du` (it is affine);
- end-to-end through the RiRk grid transform, the near-wall vertical derivative
  improves **409x** over homogeneous Neumann on a test field with a known nonzero
  wall slope.

Scythe side: `mc_wall_bc_active` / `update_mc_wall_bc!` (moist_compressible.jl),
called from `advanceTimestep` and `load_initial_conditions!`.

### The trap — a state-tracking wall derivative is unstable

Setting `∂p'/∂z|wall = -g rho_t'|wall` from the CURRENT state each step closes a
loop: the wall condition moves p' near the wall, the acoustic solve moves rho_t'
there, which resets the wall condition. Measured on the resting column:

| wall-derivative source | outcome |
|---|---|
| pinned to zero (`ahat` frozen at 0) | quiet indefinitely — R1T1X == R1T1 |
| 3-point Lagrange extrapolation to the wall | non-finite in ~26 steps |
| 2nd-order Taylor off the fitted derivatives | non-finite in ~30 steps |
| cell-mean + 300 s relaxation | non-finite in ~900 steps |

Smoothing and relaxation slow it; neither removes it, because the TARGET is what
grows. Two mechanisms compound: the extrapolations weight near-wall curvature by
`d²/2 ~ 1.6e3 m²`, amplifying grid-scale content (the cell mean fixes that part);
and, more fundamentally, imposing `∂p'/∂z = -g rho_t'` makes the net vertical
force at the wall exactly zero, removing the restoring force that would otherwise
oppose a growing boundary mode. Neutral, not damped — so any numerical
amplification is unopposed.

### What ships, and what is open

`tc/tc_init.jl` sets `:wall_bc_tau => Inf`, which FREEZES the wall derivative at
the value `load_initial_conditions!` computes from the balanced vortex. A frozen
offset is affine and provably (and measurably) as stable as R1T1, so the ceiling
stays at ts ~ 2.0 while the balanced state remains representable.

**Open:** a frozen value goes stale as the storm deepens and its true surface
deficit grows. Candidate resolutions, none yet tested:
1. refresh from a heavily time-averaged state on a slow (~hourly) cadence, far off
   the acoustic timescale, accepting some drift;
2. add an explicit damping term to the near-wall pressure so the boundary mode is
   damped rather than neutral, then allow tracking;
3. carry `p'' = p' + g ∫ rho_t' dz` as the prognostic pressure, whose wall
   derivative vanishes identically — plain R1T1 then becomes exactly right and no
   inhomogeneous machinery (or feedback) exists at all. Cleanest; largest change.

---

## The resting ceiling does NOT transfer to the balanced vortex

`prod@2.0` (R1T1X walls) is quiet indefinitely on the resting column, but the
12 h nophysics hold test at `SCYTHE_TC_TS_SCALE=2.0` **died at t = 9840 s
(2.73 h)** with a non-finite spectral coefficient on nest 3.

This is `SI_CONVECTIVE_CEILING.md`'s standing lesson applied to the wall ceiling:
**an SI stability claim measured on a resting base is an upper bound, not a
timestep.** The run's own advisory called it correctly before it failed —
"Consider ts <~ 1.5911 s", from the δ = 0.15 convective heuristic.

So the resting-column table above brackets the WALL-CONDITION ceiling (which is
what it was built to isolate, and where d2 at 0.75 s vs Neumann/R1T1X at 2.0 s is
the real comparison). The production timestep is then set by the convective
ceiling on top of it, which is a separate and lower limit.

Failed run preserved at `tc/output/tc_holdtest_nophysics_r1t1x_ts20/`.


---

## THE R1T1X CONFIGURATION IS WORSE THAN d2. DO NOT SHIP IT. (2026-07-21)

Retracting an earlier claim in this note. R1T1X was reported as cutting the
nest-1 discrete hydrostatic residual 0.1053 -> 0.00993 m/s^2. **That baseline was
not the d2 configuration.** It was the half-applied R1T1X configuration itself,
silently degraded to plain Neumann because `evaluate_tendencies` (and, at the
time, `advanceTimestep` on its first step) builds a FRESH TILE whose `wall_du` is
zero. The number being beaten was Neumann's, not d2's.

Like for like, under the model's own operators at t = 0, nest 1:

    config          gradient-wind      hydrostatic
    all d2           4.670e-05          5.091e-03
    R1T1X            3.062e-03          9.932e-03      65x and 2x WORSE

and in flight, 12 h nophysics hold test, nest 1:

    d2    @ ts 0.5    15.8 % of the deficit filled,   4.1 % of the wind lost
    d2    @ ts 1.0    34 %                           20 %
    R1T1X @ ts 1.0    95 %                           46 %

The static and dynamic measurements agree once the baseline is right. The earlier
"a better t = 0 residual bought a worse trajectory" framing was an artefact of the
bad baseline and is withdrawn.

**Why R1T1X loses.** Keeping the acoustic set operator-consistent forced `rho_d`,
`rho_t`, `E_t` and `Q_ss` from `SecondDerivativeBC` onto homogeneous Neumann,
which pins `d(rho_t')/dz = 0` at the ground. That is precisely the projection
damage `HANDOFF_2026-07-21.md` identified for `p`, relocated to the densities —
and it costs far more than the correct `p` wall condition gains. Second, untested
suspect: the FROZEN wall derivative pinning `dp'/dz` to its t = 0 value while the
vortex adjusts.

**Shipped state:** `tc_init.jl` is back to `SecondDerivativeBC` on every scalar,
with `NEST_TS = 0.5` (the ceiling result stands and is independent of all this).
The R1T1X machinery is kept, tested and OFF.

**Lesson for whoever picks this up.** Every one of the three integration bugs
(state feedback, the `dr` axis, R3X contamination) and the bad baseline shared one
signature: a fresh or nested grid silently carrying zero/foreign wall data while
every static check still looked plausible. Before believing any future wall-BC
measurement, assert that the object being measured actually has the wall data
installed.

---

## ROOT CAUSE OF THE RESIDUAL DRIFT: the reference state (2026-07-21)

The "second, smaller defect" above — the ts- and BC-independent ~-9 Pa/h lid drift —
comes **entirely from the reference state**. Not the BCs, not the SI, not the
dynamics. TWO separate reference defects contribute; see the corrected control
table below.

### The control

`model_tests/tc_lid_drift_probe.jl`, resting column, zero perturbation, identical
in every respect except which reference is written:

    reference                              p'(top) @ 1800 s   max|w|
    analytic, exactly hydrostatic, DRY      0.000000e+00      0.0e+00
    analytic, exactly hydrostatic, MOIST   -5.818e-01        5.4e-04
    sounding-derived (moist, inconsistent) -4.733e+00        1.2e-03

**Bit-exact zero** for the dry, hydrostatically consistent reference. The equation
set, the d2 wall conditions, the semi-implicit solve and the sponge are therefore
all exonerated: they form a perfect discrete fixed point when the reference is one.

But an EXACTLY hydrostatic reference still breaks that fixed point as soon as
moisture is added (row 2), so the two defects below are independent and BOTH real.
An earlier version of this section claimed the drift was "entirely" the hydrostatic
inconsistency, on the strength of the dry control alone. That was wrong: the dry
control cannot distinguish "hydrostatically consistent" from "dry", because
expdot[w] is identically zero in EVERY configuration (buoyancy is perturbation-only)
and the only nonzero tendency at rest is expdot[p], which the condensation closure
feeds. The moist analytic control is what separates them.

Attribution as it now stands: the moist/Q_ss defect accounts for ~12 % of the
sounding case's drift at RH ~ 0.5; the remaining ~88 % is the sounding reference's
additional problems (its hydrostatic inconsistency and its much higher RH, not yet
separated from each other).

### The inconsistency

`-(d(pbar)/dz + g*rho_tbar)/rho_tbar` on the stored reference, TC grid:

    z = 0.06 km   -2.0e-01        z = 17.6 km   +4.0e-01
    z = 2.56 km   +2.2e-02        z = 20.1 km   +9.3e-01
    z = 15.1 km   -1.7e-02        z = 22.6 km   +1.67e+00

For scale, the balanced vortex's own hydrostatic residual is 5.09e-03 — the
REFERENCE is ~300x worse than the state it carries. And it is not the spline: a
centred finite difference of the same pbar agrees with the spline derivative to
<1 % and disagrees with -g*rho_tbar identically (-16.1 % vs -17.0 % at 22.6 km).
The stored analytic profiles are simply not in hydrostatic balance.

### Why

`Springsteel/src/reference_state.jl:572-581`, the refinement loop, closes only one
way:

    column.uMish .= -gravity .* rho_t
    Btransform!(column); Atransform!(column)      # a FIT, with l_q smoothing
    p_Pa = IInttransform(column, p_sfc)           # pbar := antiderivative of the FIT
    Tk    = theta ./ (p_0 ./ p).^(Rd/Cpd)
    rho_d = 100 .* (p .- e_v) ./ (Tk .* Rd)       # rho from p and THETA, not from dpbar/dz

`pbar` is the antiderivative of the FITTED `-g*rho_t`, but `rho_t` is then
recomputed from theta through the EOS. Nothing forces `d(pbar)/dz = -g*rho_tbar`
discretely, so the two separate wherever the l_q-regularised fit cannot follow
rho_t — above the tropopause, where the Dunion sounding has knots at only 14.2,
16.6, 20.7 and 32 km against a 500 m model grid, and where rho is small so a fixed
absolute fit error is a large relative one. The error switches on at 16.6 km and
grows monotonically to the lid, which is exactly the vertical structure of the
spurious cooling.

### A SECOND reference-consistency defect

At exact rest the equation set returns

    expdot[p]    = 2.32e+00        Qdot   = 2.96e-06   (condensation ACTIVE)
    expdot[Q_ss] = 1.30e-05        rho_c  = 1.48e-06   (cloud water EXISTS)
    every other slot = 0.000000e+00

Unchanged with `tau_qss = 1e9`, so it is the saturation closure, not the
relaxation: the reference's `Q_ss` is not consistent with the T the model's own
retrieval returns, so the resting state condenses. `mc_ref_diag` already does this
kind of retrieval-consistent reference diagnosis for the DIFFUSION path; the
condensation path has no equivalent.

### The fix

Make the reference a discrete fixed point by construction — the native-operator
route. After `IInttransform`, derive `rho_tbar` from the SPLINE DERIVATIVE of the
fitted `pbar` instead of recomputing it from theta, so `d(pbar)/dz = -g*rho_tbar`
holds to round-off; then distribute that `rho_t` over `rho_d`/`rho_v` consistently
with `q_v`. Separately, define `Q_ss_bar` through the model's retrieval so `Qdot`
vanishes at rest.

Caveat on the control: the analytic reference above is DRY, so it conflates
"hydrostatically consistent" with "no moisture". An analytic MOIST reference is the
sharper control and would separate the two defects. Both live in the same function.

**This is the target for a multi-day run.** It is timestep-independent, so it does
not trade against the timestep, and if the lid is where the marginal mode lives,
removing its forcing may also lift the wall ceiling.
