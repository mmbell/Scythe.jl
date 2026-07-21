# The reference state is now a discrete steady state (2026-07-21, resolved)

Supersedes the working document of the same name. Both defects it identified are
fixed, both behind opt-in flags, and **the gate passes**: the resting column is
BIT-EXACT zero on the production sounding-derived reference.

    julia --project=. model_tests/tc_lid_drift_probe.jl 3.0 d2@0.5+qss+hydro
    #  p'(top) = 0.000000e+00,  max|w| = 0.000000e+00  at every report to 3 h

---

## WHAT THE FIX IS

Two independent opt-ins, both bitwise-off by default (they move every
pressure-reference baseline: BF02, O01/rainfall, Straka). The TC configuration
sets both (`tc/tc_init.jl`, `make_base`).

### `options[:consistent_qss_reference]` — defect 1, the reference condensed at rest

`Scythe.consistent_qss_reference` (`src/moist_compressible.jl`) rebuilds `Q̄_ss`
through the model's OWN retrieval, following the `mc_reference_diagnostics`
precedent:

    ρ_v,max = max(ρ̄_t − ρ̄_d, 0)
    T       = retrieve_temperature(p̄ + Ē_t − ρ̄_t·g·z, ρ̄_d, ρ̄_t, ·, p̄, T̄, 0)
    Q̄_ss    = ρ_v,max − ρ_v*(T, p̄)

The retrieval is independent of `Q_ss` while the diagnostic clamp is active
(the clamped `wfactor` is `−ρ_r = 0`), so it is a one-shot solve. The resulting
partition gives `ρ_c = 0` to within one ulp and `S < 0`, which shuts both gates in
`qss_condensation_rates`; it returns `(0.0, 0.0)` exactly and slot 1's condensation
source vanishes. Both gates are asserted per level, so a saturated sounding fails
loudly instead of silently condensing.

Applied at the single reference-construction call site
(`src/semiimplicit.jl`, `createModelTile`) and, so the initial conditions are
differenced against the same `Q̄_ss` the model adds back, in `init_tc!`.

### `options[:hydrostatic_reference]` — defect 2, the reference was not balanced

Two parts, in `Springsteel/src/reference_state.jl`:

1. **`calculate_pressure_reference_state` iterates to convergence.** The hydrostatic
   sweep OSCILLATES before it settles, and the historical `for _ in 1:5` stopped
   mid-swing. On the TC grid with the Dunion MT sounding the lid pressure runs

       it     1      2      3      4      5      6      7      8    ...   50
       p   1912   4010   1512   3550   2273   2897   2642   2724    ...  2707 Pa

   so the stored `p` was **16 % below** the converged value and the returned
   `(p, ρ_d)` pair was mutually inconsistent by **27 %**. That truncation is the
   whole of defect 2 — not the sounding's vertical resolution, not `l_q`, not the
   spline operator. Now it sweeps to `1e-12` (≈10-50 passes) or throws.

2. **`_hydrostatic_pressure_profile` builds `p̄` from the antiderivative, then snaps.**
   The value slot is the `IInttransform` antiderivative of the fitted `−g·ρ̄_t`
   (previously that spline was discarded and its VALUES re-fitted — and a 0.03 %
   value-fit error on 1e5 Pa is ~30 Pa, i.e. ~0.06 Pa/m across a 500 m cell, which
   is 10-17 % of `g·ρ_t` where `p` is small). The derivative slots are then set to
   `−g·ρ̄_t` and `−g·dρ̄_t/dz` exactly. Step 1 is what makes that snap a 0.03 %
   adjustment rather than a 17 % one.

Measured on the TC grid, `−(dp̄/dz + g·ρ̄_t)/ρ̄_t`:

    before   −2.0e-01 at 0.06 km  ...  +1.67e+00 at 22.6 km
    after     0.0 at EVERY level (exact)

The `.ref` file carries values only, so `init_tc!` writes the CONVERGED
`(p, ρ_d, ρ_v)` triple and `exact_pressure_reference_state` re-integrates
`dp/dz = −g·ρ_t` from it. No file-format change was needed. A reconstruction that
disagrees with the input pressure by more than 1 % throws — the coarse-sounding
gate.

---

## THE 88/12 SPLIT WAS AN ARTIFACT — RETRACTED

The previous document flagged it as "inferred by difference, not isolated." It is
now isolated, and it is **100/0**:

    d2@0.5              p'(top) @ 1800 s = −4.733e+00     (neither fix)
    d2@0.5+hydro                          −7.400e+00      (defect 2 fixed only)
    d2@0.5+qss                             0.000000e+00   (defect 1 fixed only)
    d2@0.5+qss+hydro                       0.000000e+00   (both)

**Defect 1 accounts for all of the resting drift; defect 2 contributes none of it.**
The reason is structural and visible in the equation set: at exact rest every
reference-derivative term in the `moist_compressible` tendencies multiplies `w` or
`w_z`, the `w` forcing is perturbation-only (`−g·ρ_t'/ρ_t`), and
`semiimplicit_adjustment_p` acts only on perturbations. So the reference's
hydrostatic imbalance produces **exactly zero tendency at rest** — it cannot show up
in this probe at all. The earlier 88/12 came from comparing a moist ANALYTIC
reference against the SOUNDING reference, which differ in more than one way.

Defect 2 still had to be fixed, for the opposite reason: because the imbalance is
invisible to the `w` equation, the equation set carries `dp̄/dz = −g·ρ̄_t` as an
unstated assumption, and where it failed the model **silently omitted a forcing of
up to 1.67 m/s²** while every reference gradient the perturbations advect
(`p_z = pp_z + p̄_z`) was wrong by that amount.

---

## STATE OF THE REPO

Suites green with the flags OFF (bitwise inert, as required):
**Springsteel 41240/41240**, **Scythe 7665/7665**.

Changed:

    Springsteel  src/reference_state.jl   converge the sweep; antiderivative + snap
    Scythe       src/moist_compressible.jl  consistent_qss_reference
    Scythe       src/semiimplicit.jl        both opt-ins at the one call site
    Scythe       tc/tc_init.jl              both flags ON for the TC run
    Scythe       tc/tc_postprocess.jl       match the run's Q̄_ss (--legacy-qss to opt out)
    Scythe       model_tests/tc_lid_drift_probe.jl   "+qss" / "+hydro" cases

`tc/tc_postprocess.jl` reconstructs `Q̄_ss` independently of the model and had to be
updated to match, or adding it back to the stored `Q_ss'` recovers the wrong total
and corrupts the retrieved `T` and the water partition. Runs made before this change
need `--legacy-qss`.

---

## WHAT IS STILL OPEN

- **Promote the gate into `test/`** as a standing regression (one patch, zero
  perturbation, sounding-derived reference, assert bit-exact zero).
- **Re-baseline BF02 and O01/rainfall with the flags ON** and check whether the fix
  improves them or merely changes them. Not yet run.
- **The balanced vortex's own discrete balance** is a separate defect and is NOT
  addressed here (see `project_tc_discrete_balance_root_cause`): its t = 0
  hydrostatic residual was 5.09e-3 m/s², built by a trapezoid rule on the work grid
  and then projected onto splines. Now that the reference it sits on is exact, that
  residual is the next-largest term and can finally be measured cleanly.
- **`R1T1X` re-measurement.** The comparisons that rejected it
  (`tc/SI_WALL_BC_CEILING.md`) were taken on the unbalanced reference. They can be
  re-taken now. `R1T1X` remains built, tested and OFF.
- **`Ē_t`'s derivative slot** has the same large-value / small-derivative character
  as `p̄` did. It carries no balance constraint, so it was left alone — but it has
  not been measured.

---

## WRONG TURNS FROM THE PREVIOUS SESSION — STILL WORTH READING

They share one shape: **a control was run that could CONFIRM the hypothesis, when
what was needed was one that could SEPARATE it.** The 88/12 split above is the
fourth instance and was caught only by tracing which terms survive at `w = 0`.

1. **R1T1X was built on a broken baseline** — a "10.6x better hydrostatic residual"
   measured against a config that had silently degraded to plain Neumann. Like for
   like it was 65x and 2x WORSE than d2.
2. **"Mass created at the lid, +32 %"** was a trapezoid rule on overlapping
   regular-grid nests. On the model's own quadrature mass conserves to ~1e-18.
   Retracted.
3. **"The drift is entirely the hydrostatic inconsistency"** rested on a DRY analytic
   control, which cannot separate "hydrostatically consistent" from "dry". Retracted.
4. **A 0.3 h probe declared a configuration stable**; it went non-finite at 2678 s.
   Run >= 600 s.

---

## REPRODUCTION

    # the gate, and the four-way split that isolates the two defects
    julia --project=. model_tests/tc_lid_drift_probe.jl 3.0 \
        d2@0.5 d2@0.5+hydro d2@0.5+qss d2@0.5+qss+hydro

    # the reference's hydrostatic residual, with and without the fix
    #   -(ref_pressure(ref)[:,2] + gravity*ref_rho_t(ref)[:,1]) ./ ref_rho_t(ref)[:,1]
    # build the TC grid, calculate_pressure_reference_state(SOUNDING, z, column;
    # hydrostatic=true/false), write_exact_ref_mc, exact_pressure_reference_state

    # 12 h nested vortex hold test, never judge on a 6 h window
    env JULIA_NUM_THREADS=4 SCYTHE_TC_TAG=_tag julia --project=. \
        model_tests/tc_balance_holdtest.jl 12 nophysics
