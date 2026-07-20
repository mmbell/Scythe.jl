# exact_si fails on the balanced TC vortex (2026-07-19)

`options[:exact_si]` drives `rho_d` negative within **~4 timesteps** (2 s of model
time) on the 3-nest axisymmetric TC, while the vertical-only SI runs the *same*
initial conditions cleanly. This is an immediate blow-up, categorically different
from the slow convective failure at 5–6 h that motivated Stages 3–5a.

The TC therefore ships on the vertical-only SI (`:state_dependent_si`).
`tc_run_axisym.jl --exact-si` re-enables the exact solve for debugging.

## Evidence

Both runs: `julia --project=. tc/tc_run_axisym.jl 600`, 3 nests, `NEST_TS =
[0.5,0.5,0.5]`, identical ICs (fixed init: perturbation-form hydrostatic +
moist core), `precipitation`, `louis_bl`, `surface_fluxes` all on.

| run | outcome |
|---|---|
| `--exact-si` | `DomainError` in `entropy` at ~step 4, worker 3 (nest2): `log(-0.007017326924991579)` via `moist_entropy_total` ← `mc_louis_bl!` (`mc_boundary_layer.jl:194`) |
| default (vertical-only SI) | no error; runs past the crash point by orders of magnitude in wall clock |

The negative argument is a collapsed dry density, i.e. the state blew up and the
Louis BL was merely the first consumer to hit a `log`.

## Leading hypothesis (NOT yet proven)

exact_si has only ever been validated on states where this cannot bite:
a dry-isothermal resting base, a moist resting base, and a moist warm bubble —
all with `f = 0` and no boundary layer or surface fluxes
(`model_tests/nested_exact_si_axisym.jl`).

The TC adds a **gradient-wind balanced vortex**, where two large terms nearly
cancel in the radial momentum equation (`mc_u_forcing!`, `mc_geometry.jl:274`):

    du/dt = -p_x/rho_t + (f + v/r) v

exact_si defers the reference-linear PGF to the implicit solve and adds it back to
the explicit remainder (`moist_compressible.jl`, the `hsi_like` block):

    expdot[u] += p_x / rho_tbar

so the explicit part carries `-p_x(1/rho_t - 1/rho_tbar) + (f + v/r)v` while the
implicit solve supplies `-p_x/rho_tbar`. **The near-cancelling balance is thereby
split across the implicit and explicit halves**, which are advanced with different
time weightings (off-centered AI2*). Any inconsistency between the two no longer
cancels against a large opposing term — it appears as a net radial acceleration on
top of an ~8.5 hPa pressure gradient. That is the natural suspect for an
O(few-step) failure, and it is invisible to every existing gate because they all
have `v = 0` and `f = 0`.

## Suggested next probes (cheapest first)

1. `--exact-si` with `VMAX = 0` (no vortex, everything else identical). If it
   survives, the vortex is implicated rather than the BL/surface-flux physics.
2. `--exact-si` with `VMAX = 15` but `F_COR = 0`. Separates Coriolis from
   curvature in `(f + v/r)v`.
3. `--exact-si` with `louis_bl` and `surface_fluxes` off, vortex on. Rules the
   physics coupling in or out.
4. If the vortex is implicated: check whether the u-leg's implicit PGF and the
   explicit centrifugal term are advanced with matching time weights, and whether
   the balanced state is a fixed point of the combined update at `t = 1` (the
   trapezoidal first step) — `exact_si_load_history!` is only called for `t > 1`.

A genuinely balanced vortex should be a **steady state**: the strongest gate to
add is that exact_si holds a balanced vortex at rest to roundoff, the rotating
analogue of the existing dry-isothermal resting-base gate.

## Note on payoff

Fixing this does not buy timestep at the current configuration. The run's own
Courant ladder at `ts = 0.5 s` (`dx_min = 676.2 m`, `dz_min = 67.6 m`) reports:

    horizontal acoustic (semi-implicit): Co = 0.25/3.0
    vertical acoustic (convective SI ceiling, delta=0.25): Co = 2.51/2.88  <-- binding
    u advection: Co = 0.07/0.5 ;  w advection: Co = 0.18/0.5

The horizontal acoustic mode that exact_si makes implicit has a 12x margin; the
binding constraint is the vertical convective ceiling, which exact_si does not
address (it is reference-linearized, and is in fact mutually exclusive with
`:state_dependent_si`). The value of exact_si here is finer grids and solver
robustness, not wall clock.
