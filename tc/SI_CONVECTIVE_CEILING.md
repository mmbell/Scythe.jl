# The state-dependent (convective) ceiling of the mc semi-implicit — TC crash diagnosis

Working note, 2026-07-16/17. Status: **DIAGNOSED, fix decision pending user review.**
Companion to `tc/SI_VERTICAL_CEILING.md` (the resting-base operator-consistency
ceiling, FIXED) — this note documents the *next* ceiling out, found by the first
6-h nested TC run at NEST_TS [0.75, 1.5, 1.5].

## Symptom

The 6-h axisym nested run (`tc/tc_run_axisym.jl 21600 --csv`) died between
18000 s and 21600 s on nest1 with `log(−0.123)` in the Louis-BL entropy staging
(`entropy` ← `moist_entropy_total` ← `mc_louis_bl!`) — a ρ_d gone negative,
surfacing at the first `log`. Courant numbers in the log were unremarkable
(advective w/u Co ≤ 0.05 at the 18000-s diagnostic; acoustic Co_z 4.1 on nest1,
8.2 on nests 2–3 — inside the measured resting-base ceiling of 9–18). The run
was NOT conditionally unstable in the classic sense.

## Diagnosis chain (all runs restart from the 18000-s output; per-step
## min-ρ_d tracing via the new `options[:state_minima_trace]`)

1. **Reference rerun** (`tc/tc_debug_restart.jl`, physics unchanged):
   reproduces the death at t ≈ +1580 s. The trace shows a sequence of
   explosive convective bubbles (updraft cores rising 5 → 15 km at ~25 m/s,
   igniting at successive radii ~0.3/22/45/72 km), each carrying a ρ_d deficit
   of 25–40% of the local reference — far beyond any physical warm anomaly
   (an undiluted parcel gives ~5–10%). The terminal event is a surface
   (z = 28 m) collapse under the convection: a ~600-m-deep dense cold dome
   (ρ_d′ +43%, p′ +88 hPa, implied T′ ≈ −65 K, u swinging −62/+83 m/s across
   25 km, negative ρ_r over r = 5–15 km) that grows quasi-exponentially
   (surface p′ 165 → 1030 → 2530 → 8850 Pa per 300 s, e-fold 3–4 min), ending
   in a grid-scale step-to-step oscillation that drives ρ_d from +0.4 to
   −1.4 kg/m³ in ~7 s.
2. **E1, precipitation off**: dies FASTER (t ≈ +511 s), in the axis updraft at
   z ≈ 8–10 km, via a violent grid-scale VERTICAL oscillation (the traced
   minimum flickers between adjacent mish levels step to step, ρ_d ratio
   0.15 → −0.34 in ~6 s). **Rain is not the root cause** — the microphysics
   per-evaluation limiters check out (evaporation capped by available rain,
   `Fr = max(ρ_r,0)·V_t`, AUTO_COLL capped) — though small negative ρ_r
   appears early at the gust front (fitted flux-divergence overshoot + AB3
   overdraw; rain has no positivity restoration analogous to
   `qss_relaxation`), and the giant cold dome's thermodynamics (Q_ss ≈ −0.07,
   ~100× below the −ρ_vs admissible floor despite the τ = 10 s relaxation)
   show the moist bookkeeping deep outside its design envelope in the
   runaway region.
3. **E2, precipitation off at half timestep** (ts_scale = 0.5, Co_z 2.05 on
   nest1): the SAME axis bubble (same location, same event) rises with a
   4–5% ρ_d deficit — a physically sensible updraft anomaly — while E1 showed
   40%+ at the same stage; the deficit then deepens over minutes and the run
   dies at t ≈ +868 s in the same grid-scale collapse, **1.7× later than E1
   (511 s) from identical state and physics**. Growth rate of the anomaly is
   strongly Courant-dependent at fixed physics → the amplification is
   numerical, not physical CAPE release. Caveat: the 18000-s restart state
   itself evolved 5 h at Co_z 4.1 (contaminated by the same mechanism), so
   even E2's "physical" early anomalies inherit prior amplification; the
   clean ts = 0.3 (Co_z 1.64) run survived its full violent CAPE release.
   Note both E1/E2 are precip-off, which removes the physical brakes
   (loading, evaporative cooling) — the full-physics reference died later
   (+1580 s) than either.

4. **E3, FULL physics at half timestep** (ts_scale = 0.5, NEST_TS
   [0.375, 0.75, 0.75]): completes the full crash hour 18000 → 21600 s
   cleanly — **the 6-h integration is done** (`tc/output/tc_debug_fullslowts/`,
   300-s CSV output). Zero min-ρ_d flags; final state healthy and storm-like
   (max|w| 4.4 m/s, max|u| 29 m/s, min ρ_d/ref ≈ 0.94). Wall clock 31 min for
   the hour on the laptop. This validates ts_scale 0.5 as the production
   STOPGAP while the durable fix is decided, and provides the reference run
   for the horizontal-SI ADI-damping metrics.

## Mechanism (proposed): the nonlinear/state-dependent SHB78 residual

The SI integrates the acoustic operator linearized about the RESTING reference
(ρ̄_t(z), Pξ̄(z) — the local-profile fix of SI_VERTICAL_CEILING.md removed the
*reference-profile* error). In a violent convective core the true state deviates
from that reference by a fraction δ (ρ_t′/ρ̄_t, and the c² anomaly from the
warm/cold anomaly); that fraction of the grid-scale acoustic operator sits in
the REMAINDER and is integrated explicitly by AB3 (imag-axis limit ω·Δt ≈ 0.72).
Local instability onset therefore at

    δ · Co_z ≳ 0.72        (Co_z on dz_min = 0.2254·dz_cell)

- nest1 at ts 0.75 → Co_z 4.1 → tolerance δ ≈ 0.18. The traced bubbles
  transition from smooth growth to grid-scale collapse at deficits ≈ 20–50% —
  consistent.
- The earlier ts = 0.3 validation run (Co_z 1.6, tolerance δ ≈ 0.44) survived
  an equally violent CAPE release ("vigorous but resolved", −28 m/s downdraft)
  — consistent.
- The resting-base sweep ceiling (9–18) measured δ → 0 noise — blind to this
  by construction, exactly as the isothermal base was blind to the SHB78
  profile error. **Lesson: SI stability claims must also be tested against
  finite-amplitude states, not only resting bases.**
- The instability amplifies the very anomaly that enables it (grid-scale
  acoustic growth → larger δ → faster growth), which explains the
  bubble-deficit runaway to unphysical amplitudes and the accelerating
  surface-dome collapse (the dome: δ ≈ +0.43 in ρ, −25% in c², at the w = 0
  stagnation where vertical acoustic energy reflects).

The rain path and the BL are then victims/accelerants, not causes: negative
ρ_r and the impossible Q_ss excursions appear inside regions the acoustic
runaway has already taken far outside physical bounds (E1 removes rain and the
crash persists, faster).

## Candidate fixes (for user review — equation-set decisions)

1. **State-dependent linearization** (the principled one): evaluate the
   acoustic coefficients from the CURRENT column state each step (Pξ(z,t),
   ρ_t(z,t) column profiles through the same retrieval), refactorize the
   per-column Helmholtz each step. Cost is small (≈ b_kDim³/3 per column per
   step ≈ 5×10⁷ flop/step for a nest — trivial next to the transforms; the
   profile-coefficient assembly path already exists). The staging/remainder
   and fresh histories are pointwise and update naturally; the stored w-leg
   increment already mirrors whatever the solve applied. Delicacy: AI2*
   formally assumes a fixed L across its three time levels — slowly varying
   coefficients are standard practice (time-varying-reference SI) but the
   consistency experiments must be rerun on a CONVECTIVE test state.
2. **Divergence damping** on the acoustic modes (Skamarock–Klemp): mops up
   the nonlinear residual; standard in compressible NWP; previously
   disfavored on formulation grounds.
3. **Adaptive/split timestep**: drop ts when max local δ approaches
   0.72/Co_z. Cheap insurance, ugly as the only fix.
4. **Reduce NEST_TS** to Co_z ≈ 2 (ts ≈ 0.35): abandons most of the vertical
   SI gain; not a fix.

A separate, smaller item regardless of the above: **ρ_r positivity
restoration** (a `qss_relaxation`-style term or floor-with-bookkeeping), since
fitted-flux overshoot + AB3 can and does produce persistent negative rain at
gust fronts even in healthy runs.

## Verification assets

- `tc/tc_debug_restart.jl` — parameterized restart driver (tag, option
  overrides, ts_scale) from the 18000-s state, 300-s CSV output.
- `options[:state_minima_trace] = N` — per-step tile min-ρ_d trace
  (`state_minima_trace` in semiimplicit.jl), prints on cadence and immediately
  below half reference.
- Outputs under `tc/output/tc_debug*/`.
