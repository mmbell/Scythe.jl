# Scythe: anisotropic turbulent flux divergence in `Twoway_PV_mixing` (FUTURE WORK)

> Status: planned, not started.
> Prerequisite: the companion Springsteel plan (`springsteel_mixed_derivative_plan.md`) — opt-in
> slot 6 = `∂²/∂r∂λ`. The diffusivity floor parameterization (`:K_min_free`, `:K_min_bl`) is
> ALREADY DONE (shipped separately). This plan covers the remaining two items: anisotropic
> radial/azimuthal length scales and the corrected `∇·(K∇u)` flux divergence.

## Context & goal
`Twoway_PV_mixing` (`src/shallowWaterModels.jl`) currently writes diffusion as `K·∇²u` with `K`
**outside** the derivative, dropping the `∇K·∇u` contribution — only correct for spatially constant
`K`, but `K = Ls²|S|` varies. Fix: (a) split the mixing length into **radial and azimuthal**
components (azimuthal as **arclength in meters**); (b) correctly formulate `∇·(K∇u)` including
`∇K·∇u`, with anisotropic `K_r`, `K_λ`. Use the **per-component + curvature** operator form (not
the full stress tensor).

**Prerequisite assertion:** the equation set requires `grid.physical[:, var, 6] = ∂²/∂r∂λ`. Assert
`size(grid.physical,3) ≥ 6` at entry with an error pointing to building the grid with
`mixed_derivatives=true`; the driver must enable that flag.

## Coordinates / fields / slots (self-contained)
- Fields: radial `u`, azimuthal `v`. Free layer `(ug,vg)`, BL `(ub,vb)`; height `h`; diagnostic `w`.
- Slots: `1=value, 2=∂/∂r, 3=∂²/∂r², 4=∂/∂λ, 5=∂²/∂λ², 6=∂²/∂r∂λ`.
- Strain tensor (already in code): `S_rr = u_r`, `S_λλ = v_l/r + u/r`, `S_rλ = ½(u_l/r + v_r − v/r)`,
  `|S| = √(2(S_rr² + S_λλ² + 2 S_rλ²))`.

## Parameters
Replace the single `:Ls_free`, `:Ls_bl` with anisotropic length scales (all in meters; azimuthal =
arclength), keeping the already-added floors:
- `:Ls_r_free, :Ls_l_free, :Ls_r_bl, :Ls_l_bl` (m)
- `:K_min_free, :K_min_bl` (m²/s) — already implemented; reuse.
Document in the `# Required physical parameters` block; index directly (no defaults mechanism in
Scythe). Update the PV-mixing driver(s).

## Diffusivities (anisotropic via length scale; shared isotropic |S|)
```
K_r = max(ℓ_r² |S|, K_min)      K_λ = max(ℓ_λ² |S|, K_min)
```

## Operator: per-component flux divergence + curvature corrections
```
D_u = K_r u_rr + (∂_rK_r + K_r/r) u_r  +  K_λ u_ll/r² + (∂_λK_λ) u_l/r²  +  K_λ(−u/r² − 2 v_l/r²)
D_v = K_r v_rr + (∂_rK_r + K_r/r) v_r  +  K_λ v_ll/r² + (∂_λK_λ) v_l/r²  +  K_λ(−v/r² + 2 u_l/r²)
```
Curvature terms (Batchelor 1967 / Shapiro 1983) scaled by `K_λ` (azimuthal/metric origin) — a
per-component modeling choice. **Validation:** `K_r=K_λ=K` const, `∂K=0` ⇒ reduces EXACTLY to the
current `K·(vector Laplacian)` lines. Use as a test.

## ∇K via slot 6 (`∂_rK_r = ℓ_r²∂_r|S|`, `∂_λK_λ = ℓ_λ²∂_λ|S|`)
```
∂_r|S| = (2/|S|)(S_rr·u_rr + S_λλ·∂_rS_λλ + 2 S_rλ·∂_rS_rλ)
   ∂_rS_λλ = v_rl/r − v_l/r² + u_r/r − u/r²              # uses v_rl  (slot 6)
   ∂_rS_rλ = ½(u_rl/r − u_l/r² + v_rr − v_r/r + v/r²)    # uses u_rl  (slot 6)
∂_λ|S| = (2/|S|)(S_rr·u_rl + S_λλ·(v_ll/r + u_l/r) + 2 S_rλ·∂_λS_rλ)
   ∂_λS_rλ = ½(u_ll/r + v_rl − v_l/r)                    # uses v_rl  (slot 6)
```
Both directional `K` share `|S|`, hence the same `∇|S|` — the mixed derivative is required even for
isotropic `ℓ_r=ℓ_λ`; anisotropy only changes the `ℓ²` prefactor. Free layer uses `ug,vg` + slot 6;
BL uses `ub,vb` + slot 6.

## Numerical guards
- **Floor knee:** where `ℓ²|S| < K_min`, `K` is constant ⇒ `∂K = 0`. Compute the unfloored gradient
  and mask to zero where the floor is active (per direction; knee at `|S| = K_min/ℓ²`). Removes the
  `1/|S|` singularity in floored regions; still guard `1/|S|` for `|S|→0` elsewhere.
- Slot-6 assertion at entry. `h` needs no diffusion (unchanged); only `ug,vg,ub,vb` use slot 6.

## Structure (TDD)
Factor the per-component operator into a pure helper
`aniso_smag_diffusion(K_r,K_λ,dKr_dr,dKl_dl, φ,φ_r,φ_rr,φ_l,φ_ll, cross_l, r; curv_sign)`
returning `KDIFF`, unit-testable independently. Keep the `@turbo` style.

## Anisotropy / literature note (read before committing)
Anisotropic radial/azimuthal mixing lengths are physically motivated (vortex anisotropy + variable
metric) but **not well-established** in the TC literature (most schemes use a single `l_h`, e.g.
Bryan 2012). Conservative fallbacks needing the SAME slot-6 machinery and SAME code path:
- **Isotropic specialization:** `ℓ_r=ℓ_λ ≡ Ls` ⇒ standard isotropic `∇·(K∇u)` with the correct
  `∇K` term — well-grounded, a strict improvement over today's `K·∇²u`. Recommended default until
  the anisotropy is justified.
- **Near-axis cap:** azimuthal spacing `r·Δλ → 0` at the axis, so a fixed arclength `ℓ_λ` can exceed
  the resolvable scale near `r=0`; optionally cap `ℓ_λ ≤ c·r·Δλ`.

## ALTERNATIVE that avoids the Springsteel change (flux-form)
`∇·(K∇u)` can instead be computed WITHOUT slot 6 by assembling the flux as a field and
differentiating it spectrally — the idiom Scythe already uses for VERTICAL diffusion
(`tcblModels.jl` / `primitive_equations.jl`: `col.uMish = K .* ∂u/∂z` → transform → differentiate).
Radial: build `G = r·K_r·∂u/∂r`, take `∂G/∂r` via the spline radial transform, divide by `r`.
Azimuthal: build `H = K_λ·(1/r)∂u/∂λ`, take `∂H/∂λ` via the Fourier transform, divide by `r`. Exact
and conservative for isotropic or anisotropic K, no mixed slot, no Springsteel change — at the cost
of several extra per-step transforms (more expensive than a slot-6 read). This is the more
established full-strain-tensor Smagorinsky stress-divergence; the params/math above are identical
either way. Choose slot-6 vs flux-form on the engine-change-vs-runtime-cost tradeoff.

## Verification (TDD; new `test/test_pv_mixing_diffusion.jl`)
- **Isotropic constant limit:** `ℓ_r=ℓ_λ`, uniform strain ⇒ `K` const, `∂K=0`; assert `D_u,D_v`
  equal the legacy `K·(vector Laplacian)` to machine precision.
- **Manufactured solution:** analytic `ug,vg` with closed-form `∇·(K∇φ)`; assert the helper matches.
- **Floor:** where `ℓ²|S| < K_min`, assert `K==K_min` and the `∇K` contribution is exactly 0.
- **End-to-end:** driver with `Ls_*_free=0, K_min_free=0` ⇒ inviscid free atmosphere (recovers
  `Twoway_ShallowWater_Slab` WN2 behavior); nonzero free mixing ⇒ asymmetry damping; check
  momentum/energy budgets closer to conserved than the old `K·∇²u` form.
