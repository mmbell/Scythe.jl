# Phase 2 plan — Semi-implicit acoustics for the linear `ρ_d` set

## Context

Phase 1 (`primitive_equation_XZ_rhod`, branch `linear_rho_d`) made the dry-air
continuity mass-conserving by carrying the **linear** dry-air density `ρ_d` instead of
the log-density `ξ`, so the per-step `l_q` spline smoothing preserves `∫ρ_d`
(dry_mass drift went from −3×10⁻³ % to −4×10⁻¹⁴ %). That prototype runs **fully
explicit** (`:semiimplicit => false`). Phase 2 restores the semi-implicit acoustic
solver for the `ρ_d` variable, which is required for the dry / production cases (where
si is on) and to take acoustic-stable timesteps.

The difficulty is structural: the `ξ`-form acoustic operator is **constant-coefficient**
*because* `ξ` is the log — its linearized continuity is exactly `∂ξ/∂t = −w_z`, with no
density weighting (`src/semiimplicit.jl`, `primitive_equations.jl:264,281`). For `ρ_d`
the linearized continuity carries the reference density `ρ̄(z)`, making the operator
variable-coefficient.

## The linearized acoustic system for `ρ_d`

Linearizing about the (time-constant) reference state `(ρ̄, c̄)`:

```
∂ρ_d′/∂t = −∂(ρ̄ w)/∂z              (continuity)
∂w/∂t    = −(c_ρ̄ / 1) · ∂ρ_d′/∂z    with c_ρ̄ ≡ (∂p/∂ρ_d)/ρ_total
```

Relationship to the existing mean sound speed: `P_ξ = ∂p/∂ξ = ρ_d ∂p/∂ρ_d`, so
`∂p/∂ρ_d = P_ξ/ρ_d` and the per-level coefficient is `c_ρ̄(z) = P_ξ/(ρ_d ρ_total)` —
i.e. the current `Pxi_bar = mean(P_ξ/ρ_total)` divided by `ρ̄`. Eliminating `ρ_d′`
between the two half-steps yields the implicit operator for `w`:

```
w − Δτ² · c_ρ̄(z) · ∂_z[ ∂_z(ρ̄ w) ] = RHS
```

a **variable-coefficient** modified-Helmholtz operator (the constant `α∂_zz + β` becomes
`α(z)∂_zz + γ(z)∂_z + β(z)`). Design choice: write it in the self-adjoint flux form
`∂_z(ρ̄ c̄² ∂_z w)` so the B-spline Galerkin assembly stays symmetric (positive-definite,
banded-Cholesky-friendly), rather than a non-symmetric collocation of the expanded form.

## Implementation steps

1. **Thermodynamics / reference (`src/thermodynamics.jl`, `src/reference_state.jl`):**
   add `P_rho(Tk, rho_d, q_v) = P_xi(...) / rho_d` and a reference mean
   `P_rho_bar` (or reuse `Pxi_bar` with the stored `rhobar`). Keep `rhobar`,`rhobar_z`
   (already added in Phase 1) as the operator coefficients.

2. **Variable-coefficient assembler (`src/semiimplicit.jl:1265–1304`):** generalize
   `_assemble_vertical_matrix` / `_assemble_spline_matrix` from scalar `(α, β)` to
   z-varying coefficients.
   - Chebyshev (RZ): diagonal-scale the operator matrices — `Diagonal(coef) * M2`, etc.
   - B-spline Galerkin (RiRk): put the coefficient **inside** the quadrature,
     `M1ᵀ·(W .* a(z) .* M1)`, derived from the flux form `∂_z(a ∂_z w)` so symmetry is
     preserved. The block comment at `semiimplicit.jl:1211–1233` (implicit/explicit
     operator consistency for AI2*) must still hold: sample `ρ̄`, `c̄` on the **mish
     points**, not re-derived independently.

3. **`ρ_d`-form solver:** add `calc_Helmholtz_semiimplicit_matrix_rhod` and a
   `semiimplicit_adjustment_rhod` mirroring `semiimplicit_adjustment`
   (`semiimplicit.jl:752–825`) with:
   - RHS / vertical-derivative construction using `ρ̄ c̄²` instead of `Pxi_bar`.
   - back-substitution `ρ_d′ⁿ⁺¹ = ρ_d′* − Δτ · ∂_z(ρ̄ wⁿ⁺¹)` (replacing the plain
     `−Δτ ∂_z w` at line 824/900).
   - keep Dirichlet BC on `w` (rigid lid/floor) — unchanged.

4. **`impdot` terms in `primitive_equation_XZ_rhod` (`src/primitive_equations.jl`):**
   set the continuity implicit term to `−∂_z(ρ̄ w) = −(ρ̄ w_z + ρ̄_z w)` and the
   `w`-momentum implicit term to `−(P_rho_bar)·ρ_d′_z` (currently they retain the
   `xi`-form placeholders `−w_z`, `−Pxi_bar·xi_z`). Dispatch the `rhod` solver when
   `:semiimplicit => true` for this set.

5. **Factorization caching:** all coefficients are reference-state (time-constant), so
   the matrix is still built/factorized **once** in `createModelTile`
   (`semiimplicit.jl:114`). The per-step solve path and `_vertical_solve!` are unchanged.

## Verification

- `bf02_dry --stage perhod` (add a dry `ρ_d` stage) with `:semiimplicit => true`:
  dry_mass drift stays ~machine precision and the run is stable at the `xi`-form
  timestep.
- `bf02_moist --stage perhod` with si **on** vs si **off**: extrema (min_w, supersat)
  agree, confirming the implicit acoustics are consistent with the explicit ones.
- Acoustic Courant: confirm si-on permits the larger timestep the explicit set cannot.

## Risk / notes

- The self-adjoint B-spline assembly is the only genuinely delicate piece; validate the
  variable-coefficient operator on a 1-D standing-acoustic-wave test (constant `ρ̄`
  should reduce exactly to the current `xi`-form result) before wiring into the model.
- Confirm `semiimplicit_adjustment_xi` vs the w-form matrix mismatch noted in
  `semiimplicit.jl` is not copied into the `rhod` variant — pick the canonical form
  deliberately.
