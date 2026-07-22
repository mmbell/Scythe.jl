# HANDOFF — solve the balanced vortex on the NATIVE grid (2026-07-21)

**Start here.** Supersedes the priorities of `tc/HANDOFF_REFERENCE_STATE.md`, which is
now a RESOLVED document — read it for what the reference state does and why, not for
what to do next.

---

## THE TASK

**Rebuild the TC vortex initialization so the balance is solved on the model's own
mish with its own spline operators, and the initial state has no spurious `u`/`w`
forcing.**

The precedent is BF02: its base state is constructed so that, without the warm bubble,
the system is INERT. The vortex should reach the same standard — a balanced vortex is
a steady state of the equation set, so `∂u/∂t` and `∂w/∂t` should vanish to round-off
at `t = 0`, and the vortex should simply sit there under `nophysics`.

It does not currently. The vortex is balanced in a metric the model never uses.

---

## WHY THIS IS THE RIGHT TARGET NOW

The background state is no longer a suspect. As of the previous session the reference
is an EXACT discrete steady state: the resting column is bit-exact zero for 3 h on the
production sounding-derived reference, `-(dp̄/dz + g·ρ̄_t)/ρ̄_t = 0` at every level, and
there is a standing regression in `test/test_moist_compressible.jl`
("the resting base is an EXACT fixed point"). **There is no longer a ghost forcing
driving the system out of balance.** Whatever remains is the vortex or the dynamics.

The model is also stable at ts = 1.0 s over 12 h — it does not blow up and does not
diverge catastrophically. So this is an accuracy problem, not a stability problem.

---

## WHAT IS WRONG WITH THE CURRENT INITIALIZATION

`init_tc!` (`tc/tc_init.jl`) calls `Scythe.balanced_vortex_fields` on a foreign work
grid and then `Scythe.balanced_vortex_mc!` to interpolate onto the model. **Six
separate non-native discretizations**, all in `src/idealized.jl`:

| # | step | current | native equivalent |
|---|------|---------|-------------------|
| 1 | radial grid | `r_axis = 0:DR_WORK:r_outer`, uniform 500 m (`tc_init.jl:243`) | the model's Gauss-quadrature mish |
| 2 | radial transfer | piecewise-**LINEAR** interpolation (`:1015-1017`) | spline projection onto the basis |
| 3 | `∂/∂z` in the thermal-wind march | `_ddz_nonuniform!`, 3-point **FD** (`:730-742`) | `Ixtransform` on the column basis |
| 4 | radial march | explicit **Heun** predictor-corrector in `dr` (`:772-781`) | spectral/implicit solve in the radial basis |
| 5 | `p'` hydrostatic integration | **trapezoid** rule on the mish (`:874-877`) | `IInttransform` (the antiderivative spline) |
| 6 | the reported residual | centred **FD** in `r` on the WORK grid (`:974-978`) | the model's own operators |

Item 6 matters more than it looks: the initialization reports
`gradient-wind residual = 3.58e-4` and that number **does not measure what the model
sees**. `model_tests/tc_discrete_balance.jl`, which evaluates under the model's own
operators, puts the hydrostatic residual at **5.09e-3** — 14x larger. That gap IS the
interpolation-plus-FD mismatch, and it is now the largest imbalance in the system by a
wide margin, because the reference beneath it is at 0.0.

Item 5 is exactly the defect that was just fixed in the reference state, where the
difference between "trapezoid/re-fit" and "the spline antiderivative" was the ENTIRE
17 % hydrostatic error. Expect it to matter here too.

---

## THE ACCEPTANCE TEST

1. **Balance residuals on the mish, under the model's own operators, near zero.** Both
   the gradient-wind residual and the hydrostatic residual. `tc_discrete_balance.jl`
   already computes these — use it, do not invent a new metric, and do NOT trust
   `balanced_vortex_fields`' own `residual` field (item 6).

2. **No initial forcing.** Load the vortex, call the equation set once at `t = 1`, and
   require `expdot[u]` and `expdot[w]` at round-off. This is the direct analogue of
   the resting-column gate that settled the reference state, and it is a sharper
   instrument than any multi-hour run.

3. **Then, and only then, the hold test.** `tc_balance_holdtest.jl 12 nophysics`.

Do (2) before (3). The previous two sessions repeatedly drew conclusions from
multi-hour integrations when a one-step tendency evaluation would have separated the
question in seconds.

---

## THE 12 h HOLD TEST, FOR THE RECORD (fixed reference, nophysics, 3 nests)

Both runs start from BIT-IDENTICAL `tc_exact.ref` and nest1/2/3 IC files (verified by
md5); they differ ONLY in `NEST_TS`.

    t[h] |   dp_0.5    v_0.5 |   dp_1.0    v_1.0
       0 |   -4.289   13.848 |   -4.289   13.848
       2 |   -3.637   13.392 |   -3.510   12.739
       4 |   -3.000   12.286 |   -3.334   12.322
       6 |   -2.448   11.063 |   -3.699   12.863
       8 |   -2.015    9.873 |   -4.019   13.327
      10 |   -1.687    8.941 |   -4.300   13.861
      12 |   -1.469    8.283 |   -4.784   14.436

    ts = 0.5   65.8 % of the deficit filled, 40.2 % of the wind lost
    ts = 1.0   11.5 % DEEPER and 4.2 % STRONGER than t = 0

**ts = 0.5 drains monotonically; ts = 1.0 troughs at 4 h and then re-deepens PAST its
initial value.** A 2x timestep change flips the sign of the behaviour, so neither run
is timestep-converged and **the hold test cannot currently rank configurations.**

**Two things to be honest about before using these numbers.**

*The ts = 0.5 drain is much WORSE than it was before the reference fix.* The
pre-fix figure for the same configuration was 15.8 % filled / 4.1 % lost at 12 h
(`tc/SI_WALL_BC_CEILING.md`); it is now 65.8 % / 40.2 %. The reference is
demonstrably better in isolation — bit-exact resting fixed point, exactly zero
hydrostatic residual, and o01 improves on every conservation metric — so the most
likely reading is that the old ghost forcing was partly MASKING the vortex's own
adjustment rather than that the fix hurt. **That is a hypothesis, not a measurement.**
It has not been tested and it should not be assumed. A clean way to test it once the
native-grid init exists: run the old and new reference against the SAME init and
compare the adjustment, rather than comparing across two different initial states.

*The drain is DECELERATING, which is what an adjustment looks like.* Hourly `dp`
increments run -0.39, -0.26, -0.32, -0.32, -0.29, -0.26, -0.23, -0.20, -0.18, -0.15,
-0.12, -0.09 hPa/h — asymptoting toward roughly -1.4 hPa / 8 m/s rather than
collapsing. That is consistent with an initial state that is not a discrete balanced
state relaxing to one, and with the size of the imbalance (5.09e-3 residual). It is
the main quantitative reason to believe the initialization is the right target.

**Do not read this as "ts = 1.0 is better."** The AI2* scheme is deliberately
off-centred (`ts_term = 1.25*ts`, explicit weights `-1.0 Lⁿ + 0.75 Lⁿ⁻¹`) and is
therefore DAMPING by construction, with damping per unit time increasing with `ts`.
The larger timestep may simply be dissipating the adjustment harder — looking better
while being less accurate. This was NOT tested; a ts = 0.25 convergence run was
started and deliberately abandoned, because with six coupled non-native knobs in the
initialization the response may be nonlinear and no single-knob attribution would be
trustworthy. **Fix the initialization first, then re-run the convergence study.**

`NEST_TS` stays at 0.5. See the ceiling note below.

---

## STATE OF THE REPO

Suites green: **Scythe 7665/7665 + the new gate**, **Springsteel 41240/41240**
(both with the reference-state flags OFF; they are opt-in and bitwise inert).

    Springsteel a749f27  converge the hydrostatic sweep, keep the antiderivative
    Scythe 9cc9498  the reference condensed at rest; make it a discrete fixed point
    Scythe a8b5f07  probe and benchmark opt-ins
    Scythe b771487  TC: both fixes on; postprocessor updated
    Scythe c6609fe  correct three ceiling notes
    Scythe a4ed70e  promote the resting-fixed-point gate into the suite

**Shipped TC configuration:** `SecondDerivativeBC` on every scalar (`w` Dirichlet,
`rho_r` natural at the ground), `NEST_TS = 0.5`,
`options[:consistent_qss_reference] = true`, `options[:hydrostatic_reference] = true`.

**Benchmarks with the fixes ON:** `o01_rainfall` passes all 5 published targets and
improves every conservation drift metric (energy 0.1008 → 0.0989 %, entropy
-1.490 → -1.426 %, mass -0.01374 → -0.01344 %, water -2.262 → -2.217 %).
`bf02_moist` `+qss` is a near-no-op (its base is already saturated to 1.6e-9).
Both reproduce their committed references with the flags off (o01 to rel_L2 1e-10).

---

## CORRECTIONS TO EARLIER DOCS — DO NOT ACT ON THE ORIGINALS

Three notes now carry correction headers. Read the headers, not the bodies.

1. **`tc/SI_WALL_BC_CEILING.md` — the d2 ceiling of ts ≈ 0.75 s was an ARTIFACT.**
   The true resting-column ceiling is between ts = 1.0 and 1.25 s. Every measurement
   in that note was taken on a reference that condensed at rest at `expdot[p] = 2.3
   Pa/s`, feeding exactly the acoustic modes it characterises. **The R1T1X rejection
   is void** — it was measured at ts = 1.0, above the contaminated ceiling and below
   the true one. Note this does NOT license raising `NEST_TS`: a resting-column
   ceiling is linear stability about rest, not accuracy on a finite-amplitude vortex,
   and the hold test above shows the two are different questions.

2. **`tc/SI_CONVECTIVE_CEILING.md` — E4 did not test what it claims.** `tau_qss`
   appears in exactly ONE place in the source, `qss_relaxation`, which is "exactly
   zero wherever cloud exists". So the "condensation inert" run left condensation
   fully active in the updraft and its "NOT condensation heating" conclusion does not
   follow. All of E1-E4 also ran on the condensing reference.

3. **`tc/HANDOFF_REFERENCE_STATE.md` retracts its own 88/12 split — it is 100/0.**
   Defect 1 (condensation at rest) was all of the resting drift; the hydrostatic
   imbalance contributed none of it, because at exact rest every reference-derivative
   term multiplies `w` or `w_z`.

---

## METHOD — THE PATTERN THAT KEEPS COSTING SESSIONS

Every wrong turn in this campaign has one shape: **a control was run that could
CONFIRM the hypothesis, when what was needed was one that could SEPARATE it.** Recent
instances, all from the last two sessions:

- The 88/12 split compared a moist ANALYTIC reference against the SOUNDING reference —
  two states differing in more than one way — instead of toggling one defect.
- The wall-BC ceiling was measured on an UNSEEDED resting column. Once the reference
  is a true fixed point that column is bit-exact zero at EVERY timestep including
  2.0 s, where the SEEDED run dies in 52 s. **Unseeded measures preservation, not
  stability.** A 4x speedup was nearly reported from this.
- E4 toggled a knob (`tau_qss`) that does not control the process it was named for.
- The hold-test comparison above was set up as a genuine separator (adjustment should
  be timestep-independent) and it returned the OPPOSITE of the prediction — which is
  the useful outcome, and the reason we are not now chasing a five-knob attribution.

For the initialization work specifically: the one-step tendency test (acceptance test
2) is a separator. A 12 h hold test is not — it confounds init error, timestep error,
scheme damping and the sponge. Use the cheap sharp instrument first.

---

## REPRODUCTION

    # the reference-state gate (should be bit-exact zero)
    julia --project=. model_tests/tc_lid_drift_probe.jl 3.0 d2@0.5+qss+hydro

    # the wall ceiling -- ALWAYS with +seed
    julia --project=. model_tests/tc_lid_drift_probe.jl 1.0 \
        d2@1.0+qss+hydro+seed d2@1.25+qss+hydro+seed

    # balance residuals under the MODEL's operators
    julia --project=. model_tests/tc_discrete_balance.jl

    # 12 h hold test (tag it -- never overwrite preserved model output)
    env JULIA_NUM_THREADS=4 SCYTHE_TC_TS_SCALE=1.0 SCYTHE_TC_TAG=_tag \
        julia --project=. model_tests/tc_balance_holdtest.jl 12 nophysics

    # benchmarks with the fixes on
    SCYTHE_REFSTATE=both julia --project=. benchmarks/o01_rainfall.jl \
        --mode quick --stage mc --grid rirk

Preserved output from this session:
`tc/output/tc_holdtest_nophysics_ts05_reffix/`, `..._ts10_reffix/`.

---

## SMALLER ITEMS, NOT ON THE CRITICAL PATH

- **BF02's mc base is not in balance in the mc metric** (1.41 % over 10 km), so
  `:hydrostatic_reference` correctly rejects it. It balances the entropy/log-density
  chain-rule form `P_s·s_z + P_xi·xi_z + P_qv·q_v,z` (`src/idealized.jl:398`) to
  3.8e-8, but the total-energy set uses `dp̄/dz = -g·ρ̄_t`. Not an artifact of the
  reconstruction: the `rho_t` fit is good to 1.3e-6 and a plain trapezoid gives the
  same -1.36 %. Same class of defect, different metric — and worth fixing alongside
  the vortex work since it is the same skill.
- `Ē_t`'s derivative slot has the same large-value / small-derivative character `p̄`
  had. No balance constraint on it, so it was left alone, and it has not been measured.
- `tc/tc_postprocess.jl` reconstructs `Q̄_ss` independently and now matches the run.
  Output from runs made BEFORE 2026-07-21 needs `--legacy-qss`.
- `R1T1X` is built, tested and OFF. Re-enabling is one edit in
  `tc_boundary_conditions`; the measurements that rejected it need re-taking first.
