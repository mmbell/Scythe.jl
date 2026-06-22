# Benchmark future work

Ideas deferred until after Stage 2 (primitive equation benchmarks).

## Straka resolution scaling / convergence test

Replace the single full-resolution Straka run with a convergence study in the
style of the CM1 gravity current test
(https://www2.mmm.ucar.edu/people/bryan/cm1/test_gravity_current/):

- Run a sequence of doubling resolutions starting from the quick configuration
  (e.g. 400 m, 200 m, 100 m, 50 m, 25 m cells, with kDim scaled to match).
- Compute the L2 norm of theta' at t = 900 s against the finest solution
  (interpolated to a common grid, or sampled at 200/400 m as in Straka et al.
  Table IV) and plot L2 error vs resolution alongside the published
  convergence figure.
- Record the wall clock per resolution: this doubles as the model's scaling
  benchmark. Note that a 1024x256 spectral solution is effectively much higher
  resolution than the paper's 25-m finite difference reference, and the
  Chebyshev vertical grid forces a very small timestep at high kDim, so a
  single "full" run is less informative than the convergence curve.

## Vertical basis experiments (Chebyshev vs cubic B-spline) — DONE

The theta'/theta_e' ripple in the BF02 benchmarks is spectral overshoot
associated with the Chebyshev vertical basis.

- [x] Switch the vertical basis to cubic B-splines and rerun
  bf02_dry/bf02_moist/straka93. Implemented as the new `RiRk` Springsteel
  geometry (spline-i × spline-k, vertical kept in the k-slot) with a `--grid
  rz|rirk` benchmark option. All three benchmarks run; moist RiRk passes all
  targets. The B-spline smoothing removes the Chebyshev Gibbs ripple, but at
  matched *point count* the spline has far fewer vertical DOF (`nc+3` vs the
  Chebyshev mode count), so the sharp extrema (theta' peak, density-current
  front) are damped rather than just de-rippled — a resolution effect, see the
  kDim item below. Two reformulations were needed and are documented in
  `reference/rirk_vertical_solver.tex`:
  - The semi-implicit vertical acoustic solve: Chebyshev's square pseudospectral
    collocation does not transfer (`b_kDim != kDim`); replaced with a symmetric
    Galerkin assembly consistent with the explicit mish-point tendencies.
  - The saturated moist base state: its iteration converges on Chebyshev but not
    on the low-DOF spline (it assumes the first guess is close rather than doing
    true root finding, and the saturation vapor pressure aloft is hypersensitive);
    worked around by constructing it on a Chebyshev column and spectrally
    interpolating to the spline levels (`interpolate_base_state`).
- Observed: RiRk runs at ~half the Chebyshev timestep (tighter acoustic
  stability of the coefficient-space implicit solve), the opposite of the
  near-boundary-clustering expectation — worth understanding.
- [ ] Alternative cheaper mitigations on the Chebyshev basis (stronger spectral
  filtering, small explicit diffusion) — still untried.

## RiRk follow-ups

- [ ] **Refactor the saturated base-state iteration into a proper root finder.**
  `saturated_hydrostatic_profile` currently assumes the first guess is close
  enough and is not a Newton/root-finding scheme, which is why it converges for
  Chebyshev but diverges on the B-spline. A well-posed root-finding formulation
  should converge directly on any vertical basis and remove the need for the
  Chebyshev-construct-then-interpolate workaround.
- [ ] **Determine the RiRk vertical resolution (kDim) for dry/straka.** At the
  current matched-point-count `kDim`, the spline under-resolves the sharp
  extrema. Pick a `kDim` (more DOF) that brings dry theta' and the Straka front
  close to the published values, and weigh it against the runtime cost
  (`vertical_kdim` snaps kDim to a multiple of `mubar`).
- [ ] **Commit RiRk regression references** (via `--update-reference`) once the
  resolution above is settled, to lock in the dry/straka/moist RiRk behavior as
  a guard alongside the RZ references.

## Open questions from the Stage 1 baselines

- Updraft maximum ~10% low at 100 m in BOTH dry cases (Euler_test, no qss
  involved) while min_w matches to 1%: localized to the thin arch. Candidate
  explanations: vertical resolution at the arch, dispersion error, theta
  overshoot redistributing buoyancy. Compare against the Stage 2 PE solution.
  [Stage 2 partial answer: the moist PE recovers max w to within 5% of the
  paper, so the legacy moist deficit was mostly the qss relaxation timescale;
  the dry arch deficit persists in both equation sets.]
- Moist energy drift (-0.009 %) is ~50x the dry value: possibly tied to the
  s_condensation formulation (the +Rv term question) or the qss relaxation
  energy budget. Track across Stage 2. [Stage 2: moist PE energy drift is
  -0.013 %, same order — points at the shared adjustment/thermodynamics
  rather than the qss scheme specifically.]

## Open questions from Stage 2

- Straka PE conservation drifts (mass 0.11 %, total entropy -0.89 % over
  900 s) are ~1000x the legacy values and persist with vertical mixing off,
  implicating the implicit Kvdiff vertical diffusion path
  (calc_Helmholtz_diffusion_matrix / diffusion_timestep): the implicit solve
  appears to diffuse all variables including xi (mass) and is applied in
  advective (non-flux) form. Investigate whether xi should be excluded and
  whether a flux-form discretization conserves better.
- The +Rv term in s_condensation (added in ccf16c0, absent from the restored
  legacy path): the user reviewed the math and suspects it is a development
  artifact — remove it and rerun the moist PE benchmark to quantify the
  effect on theta_e' and energy drift.
- theta_e' Gibbs overshoot at the arch grows with updraft strength (5.9 K at
  200 m in the PE moist run vs 4.1 published). Ties into the
  Chebyshev-vs-spline vertical basis experiment above.
