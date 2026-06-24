# Springsteel: opt-in mixed derivative ∂²f/∂r∂λ for the RL grid (FUTURE WORK)

> Status: planned, not started. Prerequisite for the corrected anisotropic turbulent flux
> divergence in Scythe's `Twoway_PV_mixing` (see `scythe_pv_mixing_diffusion_plan.md`).
> Line numbers are approximate as of 2026-06-23 — re-grep the named functions/strings.

## Context & goal
The Springsteel.jl RL grid (cubic B-spline in radius `r` × Fourier in azimuth `λ`) produces a
physical array `grid.physical[:, var, slot]` with exactly **5 derivative slots**: 1=value,
2=∂/∂r, 3=∂²/∂r², 4=∂/∂λ, 5=∂²/∂λ². There is no mixed `∂²/∂r∂λ`. Downstream models (Scythe's
anisotropic turbulent-mixing scheme) need it to form `∇K` for a strain-dependent eddy viscosity.
The mixed derivative is required even for the *isotropic* Smagorinsky case (because `|S|` couples
both directions via `S_λλ`, `S_rλ`), so it is a hard prerequisite for the corrected flux
divergence in the expanded physical-space form.

Add the mixed derivative as an **opt-in slot 6**, with the **default behavior unchanged** (5-slot
RL output, Springsteel 1.0 API preserved, every generic `gridTransform!`/`tileTransform!`/
`physical_transform!` call site untouched). Enable only by grid configuration.

**Forward-looking note (refine before/while implementing):** rather than a one-off `Bool`,
consider designing this as the first step of a **general configurable derivative set** — e.g.
`derivative_set::Symbol = :standard` (or a small spec of which (∂r,∂λ) orders to emit), so the same
machinery can later (a) add the mixed term, (b) emit first-derivatives-only to save memory where
second derivatives are unused, and (c) extend to RLZ/other families. The `Bool` below is the
minimal version; a `Symbol`/spec is the recommended generalization. Either way the **default must
reproduce today's output exactly.**

## Why it is cheap (mechanism)
In the existing **∂f/∂r pass** of the RL transform, each ring's b-coefficients are reconstructed
from the spline radial-derivative coefficients, then `FAtransform!` + `FItransform!` produce ∂f/∂r
(slot 2). After those calls the ring's `.a` holds the **Fourier amplitudes of ∂f/∂r**, so one extra
`FIxtransform` on that ring yields `∂/∂λ(∂f/∂r) = ∂²f/∂r∂λ` — no new spline work, one inverse FFT
per ring. This mirrors how slot 4 (∂f/∂λ) is produced from the value ring.

## Hard constraint
The slot count must become a **grid property** consulted by the transforms, so the same generic
functions fill 5 or 6 slots with **no new dispatch** and **no call-site changes**. Write slot 6
inside the transform gated on `size(physical,3) >= 6` (robust to custom-sized arrays).

## Ordered implementation (paths in the Springsteel.jl repo)
1. **`src/Springsteel.jl`** (`SpringsteelGridParameters` @kwdef struct): add
   `mixed_derivatives::Bool = false` (or the `derivative_set::Symbol = :standard` generalization).
2. **`src/types.jl`** (`num_deriv_slots` dispatch): add params-aware
   `num_deriv_slots(gp, jbasis, kbasis)` returning `base + 1` **only** when mixed derivatives are
   requested **and** the basis combination is the RL family (`jbasis isa FourierBasisArray &&
   kbasis isa NoBasisArray`); else `base`. Guard makes the flag a no-op on non-RL grids.
3. **`src/factory.jl`** (RL allocation `_create_cylindrical_2d_rl`, the `zeros(…, nvars, 5)`): size
   via `num_deriv_slots(gp, jbasis, kbasis)`.
4. **`src/transform_scratch.jl`** (`_ScratchRL` + builder): add `mixed_scratch::Vector{Float64}`
   sized to the outer ring `4 + 4*iDim` (or reuse `jring.uMish`, free after slot 2 is copied out).
5. **`src/transforms_cylindrical.jl`** (`gridTransform(grid::_RLGrid, …)`, ∂f/∂r ring loop): if
   `size(physical,3) >= 6`, `FIxtransform` the radial-derivative ring into scratch and `copyto!`
   slot 6 (after slot 2 is written).
6. **`src/tiling.jl`** (`tileTransform!(…, tile::_RLGrid, …)`, ∂f/∂r write): if `size(physical,3) >=
   6`, write `FIxtransform(ring)` (allocating 1-arg form — preserves `@threads`-over-variable
   safety; do NOT share scratch across threads).
7. **Propagate the new field through EVERY params reconstructor (HIGHEST RISK):** `_update_gp`
   (`src/factory.jl`, on the `createGrid` path — if missed, the feature is silently disabled
   globally; same class as the `i_regular_out` propagation comment), `_create_tile_from_patch`
   (`src/tiling.jl`), the `withVariables`-style clone (`src/factory.jl`), and the inline
   `tile_gp = SpringsteelGridParameters(...)` constructors (`src/tiling.jl`, 2 sites). Grep every
   `SpringsteelGridParameters(` and `_update_gp` call.
8. **`src/io.jl`** (`write_grid` for the RL / 2D-with-j path): set `nderiv = size(grid.physical,3)`
   and extend the `suffix` label array to 6 entries (append `"_rl"`).
9. **`src/transforms_cylindrical.jl`** (`regularGridTransform(grid::_RLGrid, …)`): size output via
   `num_deriv_slots(gp,…)` and emit slot 6 from the radial-derivative amplitudes `ak_r` (∂/∂λ
   series). If deferring, zero-fill slot 6 AND gate `write_grid`'s gridded output so `_physical.csv`
   and `_gridded.csv` keep matching slot dimensions.
10. **Docs:** update the slot tables in `transforms_cylindrical.jl` and `types.jl`.

`src/interpolation.jl` already loops `2:size(grid.physical,3)` → handles 6 slots. Forward
`spectralTransform` reads only slot 1 → unaffected. Slot 6 is **appended**, never reordered, so all
existing fixed-index reads of slots ≤5 remain valid.

## Risks
- **R1 (highest) — flag propagation:** any missed reconstructor (esp. `_update_gp`) silently
  reverts to 5 slots. Covered by default-vs-enabled and multi-tile tests.
- **R2 — patch/regular/tile slot divergence:** size all three from `num_deriv_slots(gp,…)`.
- **R3 — io BoundsError / dropped column:** `suffix` must grow with `nderiv` (step 8).
- **R4 — scratch aliasing:** copy slot 2 out before the mixed `FIxtransform`.
- **R5 — thread safety:** tile path must not share `mixed_scratch` across the `@threads` loop.
- **R6 — non-RL + flag set:** must be a no-op via the basis guard (step 2); test it.

## Verification (TDD; `test/`)
- **Analytic mixed derivative:** flag on; `f = r·cos(2λ)` (⇒ `∂²f/∂r∂λ = −2 sin(2λ)`) and
  `f = r²·sin(3λ)`; assert slot 6 matches analytic in the interior (~1e-2 rel, same as existing
  derivative tests).
- **Default unchanged:** flag off ⇒ RL allocates 5 slots; slot-1..5 outputs bit-identical to the
  pre-change transform (regression guard).
- **Multi-tile:** 2-tile slot 6 matches single-patch slot 6 at shared points (R1 + R5).
- **Non-RL no-op:** Cartesian/SL grid with flag set keeps its normal slot count (R6).
- **I/O round-trip:** `write_grid` with 6 slots writes 6 derivative columns incl. `_rl` (R3).
