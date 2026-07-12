# Reference state functions
using Statistics

"""
    ReferenceState

Holds the thermodynamic reference (base) state for the model, including vertical profiles
and their first and second derivatives.

# Fields
- `sbar::Array{Float64}`: moist entropy profile, size `(nlevels, 3)` with columns for value, first derivative, and second derivative [J/(kg K)]
- `xibar::Array{Float64}`: log dry air density profile, size `(nlevels, 3)` with columns for value, first derivative, and second derivative [log(kg/m^3)]
- `rhobar::Array{Float64}`: dry air density profile, size `(nlevels, 3)` with columns for value, first derivative, and second derivative [kg/m^3]. Derived from `xibar` for the linear-`rho_d` equation set; the spectral derivatives are computed on the model basis (not via the chain rule) so they are consistent with the prognostic field's auto-derivatives.
- `mubar::Array{Float64}`: transformed water vapor mixing ratio profile, size `(nlevels, 3)` with columns for value, first derivative, and second derivative
- `satbar::Array{Float64}`: transformed saturation ratio profile, size `(nlevels, 3)` with columns for value, first derivative, and second derivative
- `Pxi_bar::Float64`: domain-mean speed of sound squared [m^2/s^2]
"""
struct ReferenceState <: AbstractReferenceState
    sbar::Array{Float64}
    xibar::Array{Float64}
    rhobar::Array{Float64}
    mubar::Array{Float64}
    satbar::Array{Float64}
    Pxi_bar::Float64
end

# ── Reference-state accessor interface ─────────────────────────────────────────
# Equation sets and microphysics read reference profiles through these accessors
# rather than reaching into struct fields directly. The accessor generics are shared
# with Springsteel (which defines them for the physical-density AbstractReferenceState
# subtypes); here we extend them for the legacy transformed-variable ReferenceState so
# both reference representations dispatch through one interface. `ref_xi`/`ref_mu` are
# Scythe-only (transformed-variable) accessors.
import Springsteel: ref_entropy, ref_sigma, ref_rho_d, ref_rho_v, ref_rho_c, ref_sat, sound_speed_sq

# The vertical reference column and the spectral derivative fill are Springsteel's, not
# Scythe's. Importing (rather than redefining) them is what keeps the two packages from
# drifting: Springsteel exports only the reference-state *types*, so a same-named definition
# here would silently become a separate generic function with no ambiguity error — which is
# exactly how Scythe's copy came to size a spline column as `kDim ÷ mubar` while Springsteel's
# had moved to the canonical `num_cells_k`.
import Springsteel: reference_column, natural_column, transform_reference_state!

"""Moist entropy reference profile `(nlevels, 3)` [J/(kg K)]."""
ref_entropy(rs::ReferenceState) = rs.sbar

"""Log dry-air density reference profile `(nlevels, 3)`."""
ref_xi(rs::ReferenceState) = rs.xibar

"""Dry-air density reference profile `(nlevels, 3)` [kg/m^3]."""
ref_rho_d(rs::ReferenceState) = rs.rhobar

"""Vapor partial-density reference profile derived from the transformed mubar."""
ref_rho_v(rs::ReferenceState) = rs.rhobar .* inv_mu_transform.(rs.mubar)

"""Legacy reference carries no separate condensate profile."""
ref_rho_c(rs::ReferenceState) = 0.0

"""Transformed water-vapor mixing ratio reference profile `(nlevels, 3)`."""
ref_mu(rs::ReferenceState) = rs.mubar

"""Transformed saturation-ratio reference profile `(nlevels, 3)`."""
ref_sat(rs::ReferenceState) = rs.satbar

"""Domain-mean speed of sound squared [m^2/s^2]."""
sound_speed_sq(rs::ReferenceState) = rs.Pxi_bar

"""
    legacy_reference_view(rs::AbstractReferenceState, column) -> ReferenceState

Materialize a legacy `ReferenceState` (transformed `xi`/`mu` profiles) from a physical
Springsteel reference state, so the existing `xi`/`mu` equation sets consume the new
physical-density reference without change. Derived profiles are
`xibar = log_dry_density(rho_dbar)`, `mubar = mu_transform(rho_vbar/rho_dbar)`, and
`satbar = mu_transform(saturation ratio)`, with spectral derivatives recomputed on
`column`. `rhobar` is the physical dry-air density profile carried through directly.
"""
function legacy_reference_view(rs::AbstractReferenceState, column)
    sbar = copy(Springsteel.ref_entropy(rs))
    rhobar = copy(Springsteel.ref_rho_d(rs))
    n = size(rhobar, 1)
    rho_d = rhobar[:, 1]

    xibar = zeros(Float64, n, 3)
    xibar[:, 1] .= log_dry_density.(rho_d)
    transform_reference_state!(column, xibar)

    rv = Springsteel.ref_rho_v(rs)
    q_v = rv === 0.0 ? zeros(Float64, n) : rv[:, 1] ./ rho_d
    mubar = zeros(Float64, n, 3)
    mubar[:, 1] .= mu_transform.(q_v)
    transform_reference_state!(column, mubar)

    # satbar = mu_transform(saturation ratio). The physical satbar profile is already
    # spectrally smoothed with derivatives; in linear mode mu_transform is a constant
    # scaling that commutes with smoothing, so scale the profile directly rather than
    # re-smoothing (which would double-filter and perturb the derivatives). Fall back to
    # transform-then-smooth for the nonlinear hyperbolic transform.
    satbar = zeros(Float64, n, 3)
    sat = Springsteel.ref_sat(rs)
    if sat !== 0.0
        if _MU_HYPERBOLIC[]
            satbar[:, 1] .= mu_transform.(sat[:, 1])
            transform_reference_state!(column, satbar)
        else
            satbar .= mu_transform.(sat)
        end
    end

    return ReferenceState(sbar, xibar, rhobar, mubar, satbar, Springsteel.sound_speed_sq(rs))
end

"""
    rhobar_from_xibar(xibar, column) -> Array{Float64}

Build the dry-air density reference profile `rhobar` (value + first/second vertical
derivative, shape `(nlevels, 3)`) from the log-density reference `xibar` by applying
`dry_density` and then computing the derivatives spectrally on `column` (via
[`transform_reference_state!`](@ref)). Used by the linear-`rho_d` equation set so the
reference density and its gradients are consistent with the model basis.
"""
function rhobar_from_xibar(xibar::Array{Float64}, column)
    rhobar = zeros(Float64, size(xibar, 1), 3)
    rhobar[:, 1] .= dry_density.(xibar[:, 1])
    transform_reference_state!(column, rhobar)
    return rhobar
end

"""
    empty_reference_state()

Create an empty [`ReferenceState`](@ref) with undefined arrays and `Pxi_bar = 0.0`.

Useful as a placeholder when a reference state is not needed (e.g., for simple test models).

# Returns
- `ReferenceState`: a reference state with uninitialized array fields.
"""
function empty_reference_state()

    ReferenceState(Array{Float64}(undef), Array{Float64}(undef), Array{Float64}(undef), Array{Float64}(undef), Array{Float64}(undef), 0.0)
end

"""
    warn_timestep_stability(grid_params, ts; c_nominal=340.0, target_courant=0.5)

Advisory startup check that the timestep is consistent with the vertical
resolution, aimed at catching the common mistake of raising `kDim` without
lowering `ts`. Builds a lightweight 1-D B-spline column from `grid_params` to get
the true minimum mish-point spacing `dz_min`, estimates the acoustic Courant
number `c_nominal * ts / dz_min`, and emits an `@warn` (never aborts) if it
exceeds `target_courant`. Returns the estimated Courant number, or `nothing`
when skipped.

**RiRk only.** The check is restricted to the cubic B-spline (`"RiRk"`) vertical
geometry, whose mish points are near-uniform so a single `dz_min`-based Courant
number is meaningful. It is a deliberate no-op for the Chebyshev (`"RZ"`)
geometry: boundary clustering makes `dz_min` tiny, so stable RZ runs routinely
sit at an acoustic Courant of several (the semi-implicit scheme treats vertical
acoustics implicitly), and a nominal-sound-speed threshold would fire on every
run.

The default `target_courant=0.5` is a conservative *empirical* tripwire, not a
derived stability bound: the BF02 RiRk case is stable near Co≈0.25 (kDim=100) but
blew up near Co≈0.75 (kDim=300). The true RiRk semi-implicit margin is still
under investigation, so treat this as advisory.
"""
function warn_timestep_stability(grid_params, ts::Float64;
                                 c_nominal::Float64=340.0, target_courant::Float64=0.5)
    grid_params.geometry == "RiRk" || return nothing
    # `num_cells_k` is the canonical cell count for the spline vertical, resolved by
    # `compute_derived_params` when `ModelParameters` is built (so it is populated whether the
    # caller supplied cells or gridpoints). This is a lightweight mish-spacing probe, not a
    # reference column, so it builds SplineParameters directly rather than via `natural_column`
    # (which dispatches on an existing column instance we do not have here).
    z = try
        sp = SplineParameters(
            xmin = grid_params.kMin, xmax = grid_params.kMax,
            num_cells = grid_params.num_cells_k,
            mubar = grid_params.mubar, quadrature = grid_params.quadrature,
            BCL = CubicBSpline.R0, BCR = CubicBSpline.R0)
        Spline1D(sp).mishPoints
    catch
        return nothing       # best-effort: never let a startup check break a run
    end

    dz_min = minimum(diff(sort(z)))
    courant = c_nominal * ts / dz_min
    if courant > target_courant
        suggested = target_courant * dz_min / c_nominal
        @warn "Timestep may be too large for the RiRk vertical resolution " *
              "(kDim=$(grid_params.kDim)): estimated acoustic Courant " *
              "≈ $(round(courant; digits=2)) [c≈$(round(c_nominal)) m/s, " *
              "dz_min=$(round(dz_min; digits=2)) m, ts=$(ts) s] exceeds target " *
              "$(target_courant). Consider ts ≲ $(round(suggested; digits=4)) s. " *
              "(Advisory empirical heuristic — the semi-implicit solver treats vertical acoustics implicitly.)"
    end
    return courant
end

"""
    calculate_reference_state(model::ModelParameters, z::Array{Float64}, column)

Hydrostatic reference state from the sounding file named in `model.ref_state_file`, as a
legacy transformed-variable [`ReferenceState`](@ref).

Thin adapter over `Springsteel.calculate_reference_state` (the shared physical-density
builder: sounding interpolation, spectral re-integration to hydrostatic balance, Newton
refinement), viewed back into `xi`/`mu` control variables via [`legacy_reference_view`](@ref).
The numerics live in Springsteel so there is exactly one hydrostatic builder; Scythe only
supplies the `ModelParameters` -> file-path adaptation and the transformed view.

This is the same path `initialize_model` takes at run time, so a benchmark that builds its
initial condition against this reference is now consistent with the reference the solver
itself uses.
"""
calculate_reference_state(model::ModelParameters, z::Array{Float64}, column) =
    legacy_reference_view(
        Springsteel.calculate_reference_state(model.ref_state_file, z, column; moisture=true),
        column)

"""
    uses_physical_reference(equation_set) -> Bool

True for equation sets that consume the *physical* (partial-density, condensate-bearing)
Springsteel reference state directly — `ref_rho_v`/`ref_rho_c` profiles rather than the legacy
xi/mu derived view. These are the partial-density `_pd` sets and the `_sigma`
family (which carries the entropy density σ=ρ_d·s on the same physical reference).
"""
uses_physical_reference(equation_set::AbstractString) =
    endswith(equation_set, "_pd") || endswith(equation_set, "_sigma")

"""
    uses_pressure_reference(equation_set) -> Bool

True for equation sets that consume the pressure-based `PressureReferenceState`
(prognostic p / E_t / Q_ss), i.e. the `moist_compressible` family.
"""
uses_pressure_reference(equation_set::AbstractString) =
    startswith(equation_set, "moist_compressible")

"""
    exact_reference_state(model::ModelParameters, z::Array{Float64}, column)

Read a pre-computed reference state from a file that has already been adjusted to
hydrostatic balance. Useful for highly idealized simulations and benchmarking.

The file must contain one line per model level with columns: altitude, entropy, log density,
and transformed moisture. Vertical derivatives are computed via [`transform_reference_state!`](@ref).

!!! note "Deliberately not delegated to Springsteel"
    Unlike [`calculate_reference_state`](@ref), this is *not* a thin adapter over
    `Springsteel.exact_reference_state`, and the difference is load-bearing rather than
    cosmetic. Two things differ:

    1. **File schema.** This reads the transformed `z s xi mu` written by
       [`write_exact_ref`](@ref); Springsteel reads the physical `z s rho_d rho_v rho_c`
       written by [`write_exact_ref_pd`](@ref).
    2. **`satbar` convention.** This returns `satbar = 0`, so the `mu_sat` prognostic carries
       the *full* saturation ratio. Springsteel's builder returns a nonzero `satbar` (≈1, the
       base `q_v/q_sat`), against which `mu_sat` is a *perturbation*.

    The legacy/`pe`/`pe-rho_d` stages of `bf02_moist` depend on convention (2): switching them
    to the Springsteel builder without also re-splitting `mu_sat` would double-count the
    saturation ratio (it would start at ≈2 domain-wide and drive spurious condensation). Doing
    that re-split changes the advected field and hence those benchmarks' results, so it is
    tracked as follow-up work rather than folded into the reference-state deduplication.

# Arguments
- `model::ModelParameters`: model configuration containing the reference state file path and grid parameters.
- `z::Array{Float64}`: vertical coordinate array of model levels [m].
- `column`: a 1D spectral basis object (e.g., `Chebyshev1D` or `Spline1D`) for computing vertical derivatives.

# Returns
- `ReferenceState`: the reference state read from file with computed vertical derivatives.

# Throws
- `DomainError` if a model level does not match the corresponding level in the file.
"""
function exact_reference_state(model::ModelParameters, z::Array{Float64}, column)

    # Read a reference state file that has already been adjusted to hydrostatic balance
    # This function is useful for highly idealized simulations and benchmarking

    # Open the file with sounding information
    ref = open(model.ref_state_file,"r")

    # Allocate some empty arrays
    sbar = zeros(Float64,length(z),3)
    xibar = zeros(Float64,length(z),3)
    mubar = zeros(Float64,length(z),3)

    # Read the file
    for i = 1:length(z)
        lineparts = split(readline(ref))
        if lineparts[1] != string(z[i])
            throw(DomainError(i, "Model level does not match reference level"))
        end
        sbar[i,1] = parse(Float64,lineparts[2])
        xibar[i,1] = parse(Float64,lineparts[3])
        mubar[i,1] = parse(Float64,lineparts[4])
    end

    # Calculate the derivatives
    transform_reference_state!(column, sbar)
    transform_reference_state!(column, xibar)
    transform_reference_state!(column, mubar)

    # Get the mean speed of sound squared
    Pxi =  P_xi_from_s.(sbar[:,1], xibar[:,1], mubar[:,1])
    rho_bar = dry_density.(xibar[:,1])
    q_bar = inv_mu_transform.(mubar[:,1])
    Pxi_bar = mean(Pxi ./ (rho_bar .* (1.0 .+ q_bar)))

    satbar = zeros(Float64,length(z),3)
    rhobar = rhobar_from_xibar(xibar, column)
    ref_state = ReferenceState(sbar, xibar, rhobar, mubar, satbar, Pxi_bar)
    return ref_state
end
