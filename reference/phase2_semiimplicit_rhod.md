# Phase 2 plan — Semi-implicit acoustics for the linear `ρ_d` set

> Math reference: `reference/Semiimplicit_linear_rhod.tex` (`.pdf`). Equation numbers
> below refer to that document.

## Context

Phase 1 (`primitive_equation_XZ_rhod`, branch `linear_rho_d`) made the dry-air
continuity mass-conserving by carrying the **linear** dry-air density `ρ_d` instead of
the log-density `ξ`, so the per-step `l_q` spline smoothing preserves `∫ρ_d`
(dry_mass drift went from −3×10⁻³ % to −4×10⁻¹⁴ %; see
`reference/mass_conservation_advective.tex`). That prototype runs **fully explicit**:
its `impdot` slots still carry the `ξ`-form placeholders
(`primitive_equations.jl:578` `-w_z`, `:595` `-(Pxi_bar*xi_z)`) and the si branch is
gated off (`:638`, "not validated for rho_d; keep disabled"). Phase 2 restores the
semi-implicit acoustic solver for the `ρ_d` variable, required for the dry/production
cases (si on) and to take acoustic-stable timesteps.

## The linearized acoustic system for `ρ_d`

Linearizing about the (time-constant) hydrostatic reference `(ρ̂_d, c̄)` couples the
dry-acoustic pair `(ρ_d′, w)` — doc Eqs. (5)–(6):

```
∂ρ_d′/∂t = [·]^ex − ∂_z(ρ̂_d w)              (continuity, implicit part)
∂w/∂t    = [·]^ex − c̄_ρ · ∂_z ρ_d′           (w-momentum, implicit part)
```

with the reference acoustic coefficient (doc Eq. 7)

```
c̄_ρ ≡ P̄_ξ / ρ̂_d ,    P̄_ξ ≡ ⟨ ρ_d P_{ρ_d} / ρ ⟩   (domain mean = code `Pxi_bar`).
```

`P̄_ξ` is the same scalar the `ξ`-form uses; the `ρ_d` form differs only by carrying
`ρ̂_d(z)`. With `ξ′ ≈ ρ_d′/ρ̂_d`, both terms reduce exactly to the `ξ`-form terms when
`ρ̂_d` is constant.

## The mass-flux reduction (no variable-coefficient solver needed)

The earlier draft of this plan proposed a variable-coefficient self-adjoint B-spline
Galerkin assembler and flagged it as the delicate risk. **That assembler is dropped.**
Introduce the mass flux `φ ≡ ρ̂_d w`. Because `ρ̂_d · c̄_ρ = P̄_ξ` is **constant**, the
implicit update (doc Eqs. 8–9) collapses to a constant-coefficient modified-Helmholtz
problem (doc Eq. 10):

```
φ^{n+1} − Δτ² P̄_ξ ∂_zz φ^{n+1} = ρ̂_d w^* − Δτ P̄_ξ ∂_z ρ_d′^*
```

The operator `(I − Δτ² P̄_ξ ∂_zz)` is **identical** to the existing `w`-form solve
`calc_Helmholtz_semiimplicit_matrix(grid, model, Pxi_bar, ts_term)` (builds
`c ∂_zz − 1`, `c = Δτ² P̄_ξ`, Dirichlet rows). Rigid lid/floor `w=0` ⇒ `φ=0` ⇒ the
same homogeneous Dirichlet BCs. **Reuse the already-factorized `h_matrix`**, on either
the RZ (Chebyshev) or RiRk (B-spline) basis — no new matrix code.

Recovery (doc Eqs. 11–12):

```
w^{n+1}    = φ^{n+1} / ρ̂_d
ρ_d′^{n+1} = ρ_d′^* − Δτ ∂_z φ^{n+1}      (flux form ⇒ ∫ρ_d′ conserved at rigid walls)
```

## Implementation steps

1. **New `semiimplicit_adjustment_rhod` (`src/semiimplicit.jl`)** — mirror
   `semiimplicit_adjustment` (`:752–825`). The AI2* predictor block (`:778–789`) and
   the n−1/n−2 shuffle carry over verbatim, operating on the `rho_d` and `w` slots.
   Three deltas from the `ξ`-form:
   - **RHS** (doc Eq. 10): `rhs = Δτ P̄_ξ ∂_z ρ_d′^* − ρ̂_d ⊙ w^*` — take the spline
     derivative of the `ρ_d′` predictor (in place of `ξ`), and multiply the `w`
     predictor by `rhobar` at the mish points (`mtile.ref_state.rhobar[:,1]`).
   - **Solve** with the existing `mtile.h_matrix`; the solved coefficient field is `φ`.
   - **Recovery** (doc Eqs. 11–12): `w^{n+1} = Itransform(φ_col) ./ rhobar`;
     `ρ_d′^{n+1} = ρ_d′^* − ts_term · Ixtransform(φ_col)`.

2. **`impdot` terms in `primitive_equation_XZ_rhod` (`src/primitive_equations.jl`)** —
   write the implicit acoustic tendencies in the flux form consistent with the solve
   (doc §7):
   - continuity (`:578`): `impdot[:,2] = −∂_z(ρ̂_d w)`, formed as a **single** spline
     derivative of the mass flux `rhobar .* w` (`−Ixtransform(spline(rhobar.*w))`) —
     **not** the product rule `−(ρ̂_d w_z + ρ̂_d,z w)` (the two differ discretely; only
     the single-derivative form is operator-consistent with the solve).
   - w-momentum (`:595`): `impdot[:,5] = −(Pxi_bar ./ rhobar) .* rho_dp_z`.
   - dispatch (`:638–640`): call `semiimplicit_adjustment_rhod` when `:semiimplicit`;
     remove the "keep disabled" comment.

3. **No changes** to `_assemble_vertical_matrix`, `_assemble_spline_matrix`,
   `_vertical_solve!`, `calc_Helmholtz_*`, `thermodynamics.jl`, or
   `reference_state.jl`. `rhobar`/`rhobar_z` already exist (Phase 1); a `P_rho`
   thermodynamic helper is **not** needed (the coefficient enters only as `P̄_ξ/ρ̂_d`).

4. **Factorization caching:** `P̄_ξ` is time-constant, so the matrix is still
   built/factorized **once** in `createModelTile` (`semiimplicit.jl:114`). The per-step
   solve path and `_vertical_solve!` are unchanged.

## Verification

1. **1-D standing-wave reduction (unit):** constant `ρ̂_d` ⇒ the `ρ_d` adjustment must
   reproduce the `ξ`-form result to machine precision, on both RZ and RiRk bases. This
   replaces the old self-adjoint-assembly validation.
2. **straka93 ρ_d stage, si on vs off (`--mode full`):** physics (min/max θ′, min/max
   w, density-current front location) agree si-on vs si-off; dry-mass drift stays
   ~machine precision; si-on is stable at a timestep the explicit set cannot take
   (acoustic Courant).
3. **bf02 (dry + moist), si on vs off:** extrema (min_w, supersaturation) agree, and
   dry-mass drift stays at the Phase-1 level. Dry first, then moist.

Run the full Julia test suite afterward; report pass/fail counts.

## Risk / notes

- The only consistency hazard is step 2: build `impdot[:,2]` with the **flux-form
  single derivative** of `ρ̂_d w`, identical to the recovery in doc Eq. (12). The
  product rule desyncs the explicit/implicit operators (the AI2* failure mode the
  `rirk_vertical_solver.tex` note documents) and reintroduces mass drift.
- Relies on `P̄_ξ` being the domain-mean scalar (as the `ξ`-form already assumes). A
  z-local sound speed would require the variable-coefficient assembler; deferred, not
  needed for consistency with the existing scheme.
- Do not start until the running straka93 (si off, linear `ρ_d`) passes — it
  establishes the explicit baseline the si-on run is checked against.
