# ── MYNN-EDMF state container and option validation ───────────────────────────
#
# This file is the PLUMBING half of the MYNN-EDMF boundary layer: the struct `ModelTile`
# carries, the option table that resolves a configuration, and nothing else. It names no
# closure routine at all -- every call into `mym_turbulence!`, `mym_predict!` and friends
# will live in src/mc_mynn_bl.jl (S5), which is included after mc_boundary_layer.jl. The
# split is the same one radiation_state.jl / radiation.jl makes, for the same two reasons:
#
#   1. `ModelTile` holds its `MYNNState` CONCRETELY (the `EMPTY_RADIATION` argument, and
#      before it `EMPTY_ISHMAEL_TABLES`): a `Union{...,Nothing}` field makes
#      `mtile.mynn.K_m` a type-unstable load in the per-column driver's hot path, so the
#      struct definition has to exist before semiimplicit.jl, which is where `ModelTile`
#      is defined.
#   2. Everything here is testable with no closure call and no boundary-layer physics at
#      all, which is what test/test_mynn_driver.jl exercises at this stage.
#
# STAGE S4 SCOPE. The closure is OFF. What exists here is the container, the validated
# option table and the setup line; the arrays are allocated and zeroed and NOTHING writes
# them. The prognostic TKE-density slot `rho_e` (moist_compressible.jl, `mc_var_names`)
# is registered and TRANSPORTED, with zero sources. A configuration without
# `options[:mynn]` must be BYTE-identical to the code that had no MYNN at all, which is
# why every fold added by this stage is inside an `if mynn_on` and never an `x + 0.0`
# (which is not the identity for `x = -0.0`).
#
# Conventions used throughout the MYNN coupling:
#   * The prognostic slot is the TKE DENSITY `rho_e = rho_t * e` [J/m^3], a TOTAL with
#     `rho_e_bar == 0`, carried so the transport is the same continuity equation every
#     other total in the set takes. The closure's own variable is `qke = 2e` [m^2/s^2];
#     the conversion is S5's business, not this file's.
#   * Gridpoint-indexed arrays are laid out exactly like `expdot` rows
#     (row `j = (c-1)*kDim + k`, layer 1 at the ground), so a driver fold is a straight
#     indexed read with no reshape.
#   * `K_m`, `K_h` are the exchange coefficients [m^2/s] at the LAYER points, which is
#     where `retrieve_exchange_coeffs!` leaves them.

"""
    MYNNState

Everything one tile needs to run and hold the MYNN-EDMF boundary layer.

The struct is `mutable` but nearly every field is `const`: the arrays are written in place
and only the three clamp/cap counters change identity between calls. The `const`
annotations are not decoration -- they let the compiler hoist the array loads in the
per-column fold.

# Resolved configuration
`closure`, `edmf`, `scale_aware`, `init_mode`, `fidelity`, `water_carry`,
`interval_steps`, `K_max`, `output`, `trace`, `check_values` are what
[`validate_mynn_options`](@ref) returned; `active` is `options[:mynn] == true`.

# Held column state (GRIDPOINT-indexed, `expdot` layout)
`el` (mixing length [m]), `sm`/`sh` (the stability functions), `vt`/`vq` (the buoyancy
coefficients of `mym_condensation!`), `sgm` (the cloud-PDF width), `cldfra_bl`/`qc_bl`/
`qi_bl` (the subgrid cloud), and `K_m`/`K_h` [m^2/s]. These are the fields
`MYNNColumnState` carries per column in the pure port (src/mynn_closure.jl), flattened
onto the tile. `qke` is NOT among them: the TKE is the PROGNOSTIC slot `rho_e`, which is
the whole point of the coupling -- a second copy carried here could disagree with it.

# Plume sums (S7)
`s_aw`, `s_aw_st`, `s_aw_qw`, `s_aw_qv`, `s_aw_u`, `s_aw_v`, `s_aw_e` are the mass-flux
sums `DMP_mf` will fill (the Fortran's `s_aw`, `s_awthl`, `s_awqt`, `s_awqv`, `s_awu`,
`s_awv`, `s_awqke`). Allocated and zeroed HERE, at the same gridpoint length as everything
else, so the S5 tendency assembly can read them unconditionally and the S7 plume stage is
a fill rather than a re-plumb. Nothing writes them before S7.

# Per-column diagnostics
`pblh` [m], `kpbl` (the layer index of the PBL top), `ust` [m/s], `rmol` (1/L), and
`last_update_step`, all of length `ncol`. `last_update_step` starts at `typemin(Int)` for
the same reason `RadiationState.last_call_step` does: "no call has happened yet" has to be
distinguishable from "called at step 0", so the first step forces an update whatever the
cadence is.

# Work and constants
`work[t]` is thread `t`'s [`MYNNWork`](@ref) -- the preallocated scratch that keeps the
closure routines allocation-free -- indexed by `threadid()` under the same
`@threads :static` ownership rule `scratch_columns` uses. `constants` is the host constant
set, built once from Springsteel.

# Counters
`n_clamp_e`, `n_cap_K`, `n_diffnum` accumulate how often the TKE had to be floored, an
exchange coefficient hit `physical_params[:mynn_K_max]`, and the vertical diffusion number
exceeded its stability bound. They are NOT `const`: they are the reason a clamp is never
silent.
"""
mutable struct MYNNState
    # ── resolved configuration (see validate_mynn_options) ──
    const active::Bool
    const closure::Float64          # 2.5 (2.6 arrives later)
    const edmf::Int                 # 0 | 1 (1 arrives at S7)
    const scale_aware::Bool
    const init_mode::Symbol         # :taper | :zero
    const fidelity::Symbol          # :fortran
    const water_carry::Symbol       # :flux | :fixed_T
    const interval_steps::Int       # BL cadence, in model steps
    const ncol::Int
    const kDim::Int
    const K_max::Float64
    const output::Bool
    const trace::Bool
    const check_values::Bool
    # ── held column state, GRIDPOINT-indexed ──
    const el::Vector{Float64}
    const sm::Vector{Float64}
    const sh::Vector{Float64}
    const vt::Vector{Float64}
    const vq::Vector{Float64}
    const sgm::Vector{Float64}
    const cldfra_bl::Vector{Float64}
    const qc_bl::Vector{Float64}
    const qi_bl::Vector{Float64}
    const K_m::Vector{Float64}
    const K_h::Vector{Float64}
    # ── mass-flux plume sums (S7); zero at this stage ──
    const s_aw::Vector{Float64}
    const s_aw_st::Vector{Float64}
    const s_aw_qw::Vector{Float64}
    const s_aw_qv::Vector{Float64}
    const s_aw_u::Vector{Float64}
    const s_aw_v::Vector{Float64}
    const s_aw_e::Vector{Float64}
    # ── per-column ──
    const pblh::Vector{Float64}
    const kpbl::Vector{Int}
    const ust::Vector{Float64}
    const rmol::Vector{Float64}
    const last_update_step::Vector{Int}
    # ── per-thread scratch and the host constants ──
    const work::Vector{MYNNWork}
    const constants::MYNNConstants
    # ── clamp/cap counters (never silent) ──
    n_clamp_e::Int
    n_cap_K::Int
    n_diffnum::Int
end

"""
    MYNNState(; kwargs...)

Keyword constructor with the INACTIVE value as its default for every field, so
[`mc_mynn_state`](@ref) names only what a live configuration actually changes. The
positional list above is long enough that a positional call is a bug waiting to happen.
"""
function MYNNState(;
        active::Bool = false,
        closure::Float64 = 2.5,
        edmf::Int = 0,
        scale_aware::Bool = true,
        init_mode::Symbol = :taper,
        fidelity::Symbol = :fortran,
        water_carry::Symbol = :flux,
        interval_steps::Int = 1,
        ncol::Int = 0,
        kDim::Int = 0,
        K_max::Float64 = Inf,
        output::Bool = false,
        trace::Bool = false,
        check_values::Bool = false,
        el::Vector{Float64} = Float64[],
        sm::Vector{Float64} = Float64[],
        sh::Vector{Float64} = Float64[],
        vt::Vector{Float64} = Float64[],
        vq::Vector{Float64} = Float64[],
        sgm::Vector{Float64} = Float64[],
        cldfra_bl::Vector{Float64} = Float64[],
        qc_bl::Vector{Float64} = Float64[],
        qi_bl::Vector{Float64} = Float64[],
        K_m::Vector{Float64} = Float64[],
        K_h::Vector{Float64} = Float64[],
        s_aw::Vector{Float64} = Float64[],
        s_aw_st::Vector{Float64} = Float64[],
        s_aw_qw::Vector{Float64} = Float64[],
        s_aw_qv::Vector{Float64} = Float64[],
        s_aw_u::Vector{Float64} = Float64[],
        s_aw_v::Vector{Float64} = Float64[],
        s_aw_e::Vector{Float64} = Float64[],
        pblh::Vector{Float64} = Float64[],
        kpbl::Vector{Int} = Int[],
        ust::Vector{Float64} = Float64[],
        rmol::Vector{Float64} = Float64[],
        last_update_step::Vector{Int} = Int[],
        work::Vector{MYNNWork} = MYNNWork[],
        constants::MYNNConstants = MYNNConstants(),
        n_clamp_e::Int = 0,
        n_cap_K::Int = 0,
        n_diffnum::Int = 0)
    return MYNNState(active, closure, edmf, scale_aware, init_mode, fidelity, water_carry,
                     interval_steps, ncol, kDim, K_max, output, trace, check_values,
                     el, sm, sh, vt, vq, sgm, cldfra_bl, qc_bl, qi_bl, K_m, K_h,
                     s_aw, s_aw_st, s_aw_qw, s_aw_qv, s_aw_u, s_aw_v, s_aw_e,
                     pblh, kpbl, ust, rmol, last_update_step,
                     work, constants, n_clamp_e, n_cap_K, n_diffnum)
end

"""
The MYNN-OFF value: inactive, every array empty, no per-thread work.

`ModelTile` carries its `MYNNState` concretely for the same reason it carries
`RadiationState` concretely (see [`EMPTY_RADIATION`](@ref)) -- a `Union` field would make
`mtile.mynn.active` a type-unstable load in `mc_driver!`'s preamble. A run with the
closure off never reads past `.active`, so the empty arrays are unreachable rather than
merely harmless.

Shared `const` across tiles, like `EMPTY_RADIATION` and for the same reason: it is a
`mutable struct`, so sharing is safe only because nothing ever writes to the off value
(the driver gate is `.active` / the slot index), and it keeps tile construction
allocation-free on the common path.
"""
const EMPTY_MYNN = MYNNState()

# ── Option validation ─────────────────────────────────────────────────────────

# Every options key this module reads. Anything else starting with "mynn" is a typo and is
# refused: a silently ignored `:mynn_intervals` would look like a working run at the
# default cadence, which is exactly the kind of failure that costs a campaign (cf. the
# Louis-BL and Khdiff_water blockers, and `RADIATION_OPTION_KEYS`).
const MYNN_OPTION_KEYS = Set{Symbol}((
    :mynn, :mynn_interval, :mynn_edmf, :mynn_closure, :mynn_scale_aware, :mynn_init,
    :mynn_fidelity, :mynn_water_carry, :mynn_output, :mynn_trace, :mynn_check_values))

const MYNN_INIT_MODES = (:taper, :zero)
const MYNN_FIDELITIES = (:fortran,)
const MYNN_WATER_CARRIES = (:flux, :fixed_T)

_mynn_check(value, allowed, key) = value in allowed || error(
    "options[:$key] = :$(value) is not recognized; use " *
    join(string.(":", allowed), ", ", " or "))

"""
    validate_mynn_options(options, physical_params, equation_set, ts, kDim)

Check the MYNN-EDMF configuration LOUDLY and return the resolved settings as a NamedTuple
`(active, closure, edmf, scale_aware, init_mode, fidelity, water_carry, interval_steps,
K_max, output, trace, check_values)`.

Called once per tile from [`mc_mynn_state`](@ref), before any array is allocated, so a
misconfigured run dies at setup rather than several minutes in -- or, worse, runs to
completion with the wrong closure.

`options[:mynn]` absent or `false` short-circuits after the type check: a run with the
closure off must never be able to fail on a MYNN rule, so a configuration carrying a stale
`:mynn_interval` still starts.

Errors:
- an unknown `:mynn_*` key, or a non-`Bool` `:mynn`;
- `options[:mynn]` on an equation set that is not a pressure-reference
  (`moist_compressible_*`) set -- nothing else carries the `rho_e` slot;
- `:mynn_closure` other than 2.5 (2.6 is a later stage);
- `:mynn_edmf = 1` (the mass-flux plumes arrive at S7), or any value but 0/1;
- an unrecognized `:mynn_init`, `:mynn_fidelity` or `:mynn_water_carry`;
- a non-positive `:mynn_interval`, or one SHORTER than the model timestep (the cadence is
  in seconds so it is nest-invariant; a cadence below one step is not a cadence);
- a non-positive `physical_params[:mynn_K_max]`.
"""
function validate_mynn_options(options, physical_params, equation_set, ts, kDim)
    on = get(options, :mynn, false)
    isa(on, Bool) || error(
        "options[:mynn] must be a Bool (got $(repr(on))); the closure variant is chosen " *
        "with options[:mynn_closure], not here")

    # Resolve the whole table first so the off path returns the same NamedTuple shape.
    closure = Float64(get(options, :mynn_closure, 2.5))
    edmf = Int(get(options, :mynn_edmf, 0))
    scale_aware = get(options, :mynn_scale_aware, true)::Bool
    init_mode = get(options, :mynn_init, :taper)
    fidelity = get(options, :mynn_fidelity, :fortran)
    water_carry = get(options, :mynn_water_carry, :flux)
    interval_sec = Float64(get(options, :mynn_interval, 20.0))
    output = get(options, :mynn_output, false)::Bool
    trace = get(options, :mynn_trace, true)::Bool
    check_values = get(options, :mynn_check_values, false)::Bool
    K_max = Float64(get(physical_params, :mynn_K_max, Inf))

    resolved = (; active = on, closure, edmf, scale_aware, init_mode, fidelity,
                water_carry, interval_steps = 1, K_max, output, trace, check_values)
    on || return resolved

    for key in keys(options)
        (startswith(String(key), "mynn") && !(key in MYNN_OPTION_KEYS)) && error(
            "options[:$key] is not a recognized MYNN option. The MYNN keys are " *
            join(string.(":", sort!(collect(MYNN_OPTION_KEYS); by = String)), ", "))
    end

    uses_pressure_reference(equation_set) || error(
        "options[:mynn] needs a pressure-reference (moist_compressible) equation set — " *
        "the TKE density is an APPENDED prognostic slot of that set and the tendencies " *
        "enter through its momentum/heat/water channels (got equation_set = " *
        "\"$(equation_set)\")")

    closure == 2.5 || error(
        "options[:mynn_closure] = $closure is not available; the port is Level 2.5. " *
        "Level 2.6 (the prognostic tsq/qsq/cov set) arrives later — it needs three more " *
        "carried fields and a second predictor")
    (edmf == 0 || edmf == 1) || error(
        "options[:mynn_edmf] must be 0 (no mass flux) or 1, got $edmf")
    edmf == 0 || error(
        "options[:mynn_edmf] = 1 arrives at S7: the mass-flux plumes (DMP_mf, " *
        "module_bl_mynn.F90 :5700-6820) are not ported. Run with 0, which leaves every " *
        "plume sum zero — the same column the Fortran produces for ktop_plume = 0")

    _mynn_check(init_mode, MYNN_INIT_MODES, "mynn_init")
    _mynn_check(fidelity, MYNN_FIDELITIES, "mynn_fidelity")
    _mynn_check(water_carry, MYNN_WATER_CARRIES, "mynn_water_carry")

    ts > 0.0 || error("validate_mynn_options: the model timestep must be positive, " *
                      "got ts = $ts")
    interval_sec > 0.0 || error(
        "options[:mynn_interval] must be positive seconds, got $interval_sec")
    interval_sec >= ts || error(
        "options[:mynn_interval] = $interval_sec s is shorter than the model timestep " *
        "ts = $ts s. The cadence is in SECONDS so it is nest-invariant (like " *
        ":output_interval and :radiation_interval); a boundary layer cannot be called " *
        "more often than the model steps")
    # Seconds -> steps, exactly once, here.
    interval_steps = max(1, round(Int, interval_sec / ts))

    (isinf(K_max) || K_max > 0.0) || error(
        "physical_params[:mynn_K_max] must be positive (or Inf for no cap), got $K_max")
    kDim > 0 || error("options[:mynn] needs a vertical dimension (grid_params.kDim = 0); " *
                      "the boundary layer is a column process")

    return (; active = on, closure, edmf, scale_aware, init_mode, fidelity, water_carry,
            interval_steps, K_max, output, trace, check_values)
end

# ── Setup ─────────────────────────────────────────────────────────────────────

"""
    mynn_setup_line(model, cfg, ncol, kDim)

One line, `mynn: ...`, stating the resolved configuration of this tile's boundary layer:
closure level, the EDMF switch, scale awareness, the initialization mode, the water-carry
convention, the cadence in seconds and in this patch's steps (nests differ in `ts`), the
`K` cap and the column geometry. Printed to stdout so it lands in the per-nest
`scythe_out.log`, beside the `radiation:` line (see [`radiation_setup_line`](@ref) for why
the tile constructor and not the run script prints it).
"""
function mynn_setup_line(model::ModelParameters, cfg, ncol::Int, kDim::Int)
    cfg.trace || return nothing
    interval_s = Float64(get(model.options, :mynn_interval, 20.0))
    println("mynn: closure=$(cfg.closure) edmf=$(cfg.edmf) " *
            "scale_aware=$(cfg.scale_aware) init=:$(cfg.init_mode) " *
            "water_carry=:$(cfg.water_carry) fidelity=:$(cfg.fidelity) " *
            "interval=$(interval_s) s (= $(cfg.interval_steps) steps at ts=$(model.ts) s) " *
            "K_max=$(cfg.K_max) m^2/s columns=$(ncol) layers=$(kDim); " *
            "TRANSPORT ONLY at this stage — the closure tendencies arrive at S5")
    return nothing
end

"""
    mc_mynn_state(model, tile, tilepoints) -> MYNNState

The tile's MYNN-EDMF state, or [`EMPTY_MYNN`](@ref) when `options[:mynn]` is absent or
`false`.

Setup path only — once per tile, never per column, and the exact analogue of
[`mc_radiation_state`](@ref) beside it in `createModelTile`. Everything that can be wrong
about a MYNN configuration is diagnosed HERE, by [`validate_mynn_options`](@ref), rather
than at the first boundary-layer call.

The vertical basis is checked here rather than in the validator because it is a property
of the GRID, not of the options: the closure's mixing length, its wall treatment and the
`rho_e` fit all assume the cubic B-spline (RiRk) column, and a Chebyshev vertical would
run it silently on a basis it was never derived for.
"""
function mc_mynn_state(model::ModelParameters, tile, tilepoints)

    get(model.options, :mynn, false) === true || return EMPTY_MYNN

    kDim = model.grid_params.kDim
    cfg = validate_mynn_options(model.options, model.physical_params,
                                model.equation_set, model.ts, kDim)

    ncol = num_columns(tile)
    ncol > 0 || error(
        "options[:mynn] needs a grid with vertical columns (num_columns(tile) = 0); the " *
        "boundary layer is a column process")
    tile.kbasis isa Springsteel.SplineBasisArray || error(
        "options[:mynn] requires the cubic B-spline (RiRk) vertical — the closure's " *
        "mixing length and wall treatment are derived on that column, and the TKE " *
        "density slot is fitted on it")

    npoints = size(tile.physical, 1)
    zpt() = zeros(Float64, npoints)
    zcol() = zeros(Float64, ncol)

    st = MYNNState(; active = true, closure = cfg.closure, edmf = cfg.edmf,
        scale_aware = cfg.scale_aware, init_mode = cfg.init_mode,
        fidelity = cfg.fidelity, water_carry = cfg.water_carry,
        interval_steps = cfg.interval_steps, ncol, kDim, K_max = cfg.K_max,
        output = cfg.output, trace = cfg.trace, check_values = cfg.check_values,
        el = zpt(), sm = zpt(), sh = zpt(), vt = zpt(), vq = zpt(), sgm = zpt(),
        cldfra_bl = zpt(), qc_bl = zpt(), qi_bl = zpt(), K_m = zpt(), K_h = zpt(),
        s_aw = zpt(), s_aw_st = zpt(), s_aw_qw = zpt(), s_aw_qv = zpt(),
        s_aw_u = zpt(), s_aw_v = zpt(), s_aw_e = zpt(),
        pblh = zcol(), kpbl = zeros(Int, ncol), ust = zcol(), rmol = zcol(),
        # `typemin(Int)` rather than 0: "no call yet" must be distinguishable from
        # "called at step 0" so the first step forces an update whatever the cadence is
        # (the `RadiationState.last_call_step` argument).
        last_update_step = fill(typemin(Int), ncol),
        # Per-thread, indexed by `threadid()` under the same `@threads :static` ownership
        # rule `scratch_columns` uses. One `MYNNWork` is ~40 kDim-length columns.
        work = [MYNNWork(kDim) for _ in 1:Threads.maxthreadid()],
        constants = MYNNConstants())

    mynn_setup_line(model, cfg, ncol, kDim)
    return st
end
