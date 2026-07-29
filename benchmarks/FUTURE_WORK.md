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

## Positivity bounds for rho_d / rho_t: available, tested, and OFF

The spline positivity limiter already supports the two prognostic densities.
`positivity_reference_profile` (src/moist_compressible.jl) recognizes `"rho_d"`
and `"rho_t"`, so `install_positivity_bounds!` installs the reference-offset
bound on both legs (`-ρ̄(z)` via the support-minimum rule on the k-leg, the
negated SB coefficients of ρ̄ on the i-leg), exactly as it does for `rho_c`. The
MC scalar BCs are Neumann (R1T1), which is bound-safe, so nothing blocks it.

**Nothing enables it, by design.** Three reasons:

1. `rho_d` has no rate sinks whatsoever, and `rho_t`'s only sink is the fitted
   sedimentation flux divergence — not a rate. The AB3 depletion-bound machinery
   that had to be built for `rho_c`/`rho_r` (the forward-Euler cap that AB3's
   23/12 leading weight overshoots by ~2.3x) is therefore vacuous here: there is
   no cap to size, and no per-step depletion to bound.
2. The constraint that actually matters is `rho_w = rho_t - rho_d >= 0`. It is a
   DIFFERENCE of two independently fitted fields, so it is not expressible as a
   bound on either one's spline coefficients. Bounding `rho_d` and `rho_t`
   separately does not deliver it and never will.
3. Both fields sit ~5 orders of magnitude from zero in every configuration run
   to date. There is nothing to protect yet.

Consistent with the project doctrine — measure, don't clamp — the state is
monitored instead:

- `options[:state_minima_trace]` (src/semiimplicit.jl) prints per-step minima of
  `rho_d/ρ̄_d`, `rho_t/ρ̄_t` and `rho_w`, flagging the first two below half
  reference and `rho_w` on any negative value.
- `min_rho_d_frac` and `min_rho_w_gm3` are reported by `o01_rainfall.jl` (both
  the single-grid and nested diagnostics) and by `bf02_moist.jl` at the MC
  stage. Informational — neither has a pass/fail target.

**To turn the bounds on** when strong convection makes them relevant, add the
species to `GridParameters.positivity`, e.g.

```julia
positivity = Dict("rho_d" => Dict(:i => 0.0, :k => 0.0),
                  "rho_t" => Dict(:i => 0.0, :k => 0.0))
```

(only `0.0` is accepted for a reference-carried variable; the offset is supplied
automatically). The testset *"positivity bounds are available and inert for
rho_d/rho_t"* in `test/test_moist_compressible.jl` locks that this installs
cleanly, produces zero `bound_shortfall`, conserves both domain masses to
rounding, and is **bitwise** identical to the unbounded run when it does not
bind. Watch `bound_shortfall` after enabling: a nonzero value means a column was
infeasible and the limiter created mass rather than redistributing it.

The `rho_w >= 0` constraint stays a monitored diagnostic regardless. If it ever
needs enforcing it has to be done in the water partition (see the negative-water
attribution work), not in the spline fit.

## Dead implicit vertical momentum diffusion in the pe/pd/sigma sets

`impdot[u] = Kvdiff * u_zz` is written by `primitive_equation_XZ`
(src/primitive_equations.jl:275), `_rhod` (:593), `_rhod_pd` (:943) and
`_sigma` (:1306), but `diffusion_timestep` (src/semiimplicit.jl:1019) and
`diffusion_timestep_pd` (:1142) only solve the `s`/`sigma` and moisture slots,
and `explicit_timestep` (:1252) reads only `expdot`. The write is therefore
dropped: those sets apply HORIZONTAL momentum diffusion only, and `w` never had
vertical diffusion at all (its `impdot` slot is the acoustic PGF).

straka93 `--stage pe` and `--stage pe-rho_d` have been passing their targets in
that state. `moist_compressible_XZ` was fixed (it now carries its own
`diffdot_n`/`diffdot_nm1` channel and `diffusion_timestep_mc`, which diffuses
u, w and theta_d'); the older sets were left bit-for-bit alone so no baseline
moved.

Fixing them means adding `u` and `w` to the `diffusion_timestep*` variable lists
— `w` needs the `diffdot` channel, since `impdot[w]` is the acoustic tendency —
and will change straka93 `pe` / `pe-rho_d` results plus anything in
`tcblModels` that has tuned around the current behavior. Re-seed those
baselines deliberately when it is done.
