# ── Comprehensive NetCDF analysis output (Stage N1) ────────────────────────────
#
# One file per output time per patch directory, `<tag>.nc`, written DIRECTLY by the model
# at `output_interval` and holding everything a reader of a moist-compressible run wants:
# the prognostic primes, the totals with the reference added back, the recovered
# hydrometeor densities, the retrieved thermodynamics, the radar/precipitation products,
# the column integrals, and the 1-D reference profiles the totals were built from.
#
# This SUBSUMES `tc/tc_postprocess.jl`. That script existed only because the raw
# Springsteel writer emits the PROGNOSTIC control variables and nothing else, so every
# consumer had to rebuild the reference state from the run's `.ref` file, guess the
# control-variable transforms out of `scythe_out.log`, and redo the retrieval. All of that
# is known exactly HERE, inside the model, at zero reconstruction risk: the reference is
# the object `createModelTile` built, the transforms are `model.options`, and the retrieval
# is the same `retrieve_temperature` the column driver calls.
#
# THREE LAYERS, deliberately separated:
#
#   1. A PURE ARRAY layer (`mc_derived_fields`, `regular_reference_profiles`, the regrid
#      helpers, `reflectivity_dBZ`) that knows nothing about `ModelTile`, workers or
#      NCDatasets. Everything physical is testable here on hand-built matrices.
#   2. A MASTER-SIDE context + writer (`netcdf_output_context`,
#      `write_netcdf_comprehensive`) that runs on the patch the master already holds.
#      `patch.physical`/`patch.spectral` and `model` are all it reads -- no `mtile`, which
#      lives only on the workers.
#   3. `write_output` (src/io.jl) dispatching on `options[:output_formats]`.
#
# PHYSICS GROUPS (BL / radiation / surface) are LAYER 2b, added in stage N2: the master
# gathers each worker's held MYNN / radiation / surface diagnostics at output steps only
# (`physics_snapshot` -> `gather_physics` -> `assemble_physics`) and the writer resamples
# them from their own mishes onto this file's regular grid. The global attribute
# `physics_groups` names the groups a file actually carries. Nothing is added to the
# per-step path; the only per-step cost anywhere is the gated store `surface_record!`
# (src/mc_surface_layer.jl) at the two boundary-layer call sites.
#
# `import`, not `using`, for NCDatasets -- the same narrower-blast-radius choice
# radiation_io.jl and mynn_io.jl make. Included in src/Scythe.jl right after mynn_io.jl.
import NCDatasets

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — pure arrays
# ══════════════════════════════════════════════════════════════════════════════

# ── Separable linear regridding (mish → regular) ──────────────────────────────
# Moved VERBATIM from tc/tc_postprocess.jl, which now calls these instead of carrying its
# own copies. Stage N2 needs them to fold the physics sidecars (radiation, MYNN), whose
# fields live on their own mishes, into this file; they are here in N1 so there is exactly
# ONE implementation from the start and the postprocessor and the model can never drift.

"""
    interp_weights(xs, xt) -> (i0, i1, w0)

Linear-interpolation weights from ascending source nodes `xs` onto targets `xt`: the
bracketing indices and the weight on the left one. Targets outside `xs` clamp to the edge.
"""
function interp_weights(xs::AbstractVector{<:Real}, xt::AbstractVector{<:Real})
    n = length(xs)
    i0 = Vector{Int}(undef, length(xt)); i1 = similar(i0)
    w0 = Vector{Float64}(undef, length(xt))
    for (j, x) in enumerate(xt)
        if n == 1 || x <= xs[1]
            i0[j] = 1; i1[j] = 1; w0[j] = 1.0
        elseif x >= xs[n]
            i0[j] = n; i1[j] = n; w0[j] = 1.0
        else
            k = clamp(searchsortedlast(xs, x), 1, n - 1)
            i0[j] = k; i1[j] = k + 1
            w0[j] = (xs[k + 1] - x) / (xs[k + 1] - xs[k])
        end
    end
    return (i0, i1, w0)
end

"""
    regrid2d(A, wx, wz) -> Matrix{Float64}

Separable linear resample of the (x, z) matrix `A` using [`interp_weights`](@ref) in each
axis.
"""
function regrid2d(A::AbstractMatrix, wx, wz)
    (ix0, ix1, wx0) = wx; (iz0, iz1, wz0) = wz
    B = Matrix{Float64}(undef, length(ix0), length(iz0))
    @inbounds for j in eachindex(iz0), i in eachindex(ix0)
        a = wx0[i] * A[ix0[i], iz0[j]] + (1 - wx0[i]) * A[ix1[i], iz0[j]]
        b = wx0[i] * A[ix0[i], iz1[j]] + (1 - wx0[i]) * A[ix1[i], iz1[j]]
        B[i, j] = wz0[j] * a + (1 - wz0[j]) * b
    end
    return B
end

"""
    regrid1d(v, wx) -> Vector{Float64}

Linear resample of a per-column (x-only) vector.
"""
function regrid1d(v::AbstractVector, wx)
    (ix0, ix1, wx0) = wx
    return [wx0[i] * v[ix0[i]] + (1 - wx0[i]) * v[ix1[i]] for i in eachindex(ix0)]
end

# ── S-band Rayleigh reflectivity ──────────────────────────────────────────────
# Moved VERBATIM from tc/tc_postprocess.jl (which now calls this one).
#
# Equivalent reflectivity factor Z = ∫ N(D) D^6 dD, converted to mm^6/m^3, then
# dBZ = 10 log10(Z). Rain: exponential DSD n(D)=N0 exp(-λD) with the model's slope
# λ = (π ρ_l N0 / ρ_r)^(1/4) (`mp_slope`), giving Z_r = 720 N0 / λ^7. Cloud: N_c identical
# droplets of diameter D_c set by the mass, Z_c = N_c D_c^6. Cloud is negligible in dBZ but
# included for completeness. Below the floor → NaN (no echo), so a plot shows blank sky
# rather than a -200 dBZ background.

"Reflectivity floor [dBZ]; below it `reflectivity_dBZ` returns `NaN` (no echo)."
const REFL_FLOOR_DBZ = -30.0

"""
    reflectivity_dBZ(rho_c, rho_r, N0, Nc_m3) -> Float64

S-band (Rayleigh) equivalent radar reflectivity [dBZ] from the cloud and rain densities
[kg/m³], the rain-DSD intercept `N0` [m⁻⁴] and the cloud droplet number `Nc_m3` [m⁻³].
`NaN` below [`REFL_FLOOR_DBZ`](@ref).
"""
function reflectivity_dBZ(rho_c, rho_r, N0, Nc_m3)
    Z = 0.0                                                   # [m^6/m^3]
    if rho_r > 1.0e-8
        λ = (π * rho_l * N0 / rho_r)^0.25                     # [1/m]
        Z += 720.0 * N0 / λ^7
    end
    if rho_c > 1.0e-8
        Dc = (6.0 * rho_c / (π * rho_l * Nc_m3))^(1.0 / 3.0)  # [m]
        Z += Nc_m3 * Dc^6
    end
    Z *= 1.0e18                                               # m^6/m^3 → mm^6/m^3
    dBZ = 10.0 * log10(Z)
    return dBZ < REFL_FLOOR_DBZ ? NaN : dBZ
end

# ── Derived-field configuration ───────────────────────────────────────────────

"""
    DerivedConfig

Everything [`mc_derived_fields`](@ref) needs to know about a run that a slot MATRIX cannot
carry: which control variable each water slot holds and with what width, how many rain
moments, whether ice is on, whether the geometry carries `v`, the reflectivity DSD
parameters, and whether the thermodynamic interface reads a floored condensate.

Resolved ONCE per run by [`derived_config`](@ref) (a `ModelParameters` read, so it must not
appear per output time, let alone per gridpoint) and carried in the
[`NetCDFOutputContext`](@ref).

The transform modes come from the same accessors the equation set uses
(`condensate_transform_mode`, `rain_transform_mode`, `rain_number_transform_mode`,
`ice_transform_mode`), so the file can never disagree with the run about what its slots
hold -- the disagreement `tc/tc_postprocess.jl` had to guess its way out of by parsing
`scythe_out.log`.
"""
struct DerivedConfig
    ctrans::Symbol            # condensate (slot 9) control variable
    cmu::Float64
    rtrans::Symbol            # rain density (slot 8) control variable
    rmu::Float64
    nrtrans::Symbol           # rain number (n_r) control variable
    nrmu::Float64
    itrans::Symbol            # the twelve ice moments (one family key)
    imu::NTuple{4,Float64}    # ice widths by moment kind: mass, number, a, c
    rain_moments::Int
    ice::Bool
    has_v::Bool
    N0::Float64               # rain DSD intercept used for reflectivity [m^-4]
    Nc_cm3::Float64           # cloud droplet number for reflectivity [cm^-3]
    N0_source::String         # provenance of N0, written as a global attribute
    cond_floor::Bool          # options[:condensate_floor] === :diagnostic
end

"Default Marshall-Palmer rain intercept [m^-4] when the run declares none."
const REFL_N0_DEFAULT = 8.0e6

"""
    derived_config(model::ModelParameters) -> DerivedConfig

Resolve the run's water representation, ice configuration and reflectivity DSD parameters
into the [`DerivedConfig`](@ref) the derived-field layer consumes.

`physical_params[:N_0]` is the run's OWN Marshall-Palmer intercept; `0.0` (the default)
means the single-moment closure used the monodisperse `:N_r` path instead and there is no
run-declared exponential intercept, in which case the reflectivity falls back to
[`REFL_N0_DEFAULT`](@ref) -- the value `tc/tc_postprocess.jl` used for every TC run to date,
so the products are continuous across this change. Which of the two was used is recorded in
the file (`reflectivity_source`), because a reflectivity is only as meaningful as its
assumed DSD.
"""
function derived_config(model::ModelParameters)
    opts = model.options
    pp = model.physical_params
    n0 = get(pp, :N_0, 0.0)
    n0_on = n0 > 0.0
    ice = ice_microphysics(opts) === :ishmael
    nmom = rain_moments(opts)
    return DerivedConfig(
        condensate_transform_mode(opts), get(pp, :condensate_mu, 1.0e-7),
        rain_transform_mode(opts), get(pp, :rain_mu, 1.0e-7),
        nmom == 2 ? rain_number_transform_mode(opts) : :none, get(pp, :mu_rain_n, 1.0),
        ice ? ice_transform_mode(opts) : :none, ntuple(j -> ice_mu(pp, j), 4),
        nmom, ice,
        haskey(model.grid_params.vars, "v"),
        n0_on ? n0 : REFL_N0_DEFAULT,
        get(pp, :max_N_c, 100.0),
        n0_on ? "physical_params[:N_0]" : "default $(REFL_N0_DEFAULT)",
        condensate_floor_mode(opts))
end

# ── Reference profiles on the regular output grid ─────────────────────────────

"""
    RegularReferenceProfiles

The run's OWN hydrostatic reference profiles, evaluated on the regular output `z`. Each
field is the model's reference spline for that quantity -- the natural-column B-spline fit
of the mish values that `createModelTile` built and every column of the run integrates
against -- sampled at the output levels. NOT an ad-hoc interpolation of the stored profile,
and NOT re-derived from a sounding: the same spline, evaluated somewhere else.

All eight are STORED profiles of `Springsteel.PressureReferenceState`
(`pbar, rho_dbar, rho_vbar, rho_cbar, rho_tbar, Tbar, E_tbar, Q_ssbar`), so none of them is
reconstructed here and `consistent_qss_reference`'s rebuilt `Q̄_ss`/`ρ̄_c`/`ρ̄_v` come
through as the run actually carries them. That was the single largest correctness trap in
`tc/tc_postprocess.jl`, which had to re-derive `Q̄_ss` from a sounding and gate it on a
`--legacy-qss` flag the user had to remember.
"""
struct RegularReferenceProfiles
    z::Vector{Float64}
    pbar::Vector{Float64}
    rho_dbar::Vector{Float64}
    rho_vbar::Vector{Float64}
    rho_cbar::Vector{Float64}
    rho_tbar::Vector{Float64}
    E_tbar::Vector{Float64}
    Q_ssbar::Vector{Float64}
    Tbar::Vector{Float64}
end

"""
    eval_reference_profile!(col, vals_mish, z_reg, buf) -> Vector{Float64}

Fit `vals_mish` (in reference-column mish order) to the vertical basis column `col` and
evaluate the resulting spline at `z_reg`, writing into `buf` and returning it.

This is `tc/tc_postprocess.jl`'s `eval_ref`, moved: `Btransform!`/`Atransform!` are the
same pair `transform_reference_state!` uses to build the reference's derivative columns, so
the curve sampled here IS the model's reference profile.

`col` is MUTATED (its `uMish`/`b`/`a` are overwritten), which is why the caller passes a
column it owns -- `netcdf_output_context` builds a fresh `reference_column`.
"""
function eval_reference_profile!(col, vals_mish::AbstractVector{Float64},
                                 z_reg::Vector{Float64}, buf::Vector{Float64})
    col.uMish .= vals_mish
    Springsteel.Btransform!(col)
    Springsteel.Atransform!(col)
    return Springsteel.SItransform(col, z_reg, buf)
end

"""
    regular_reference_profiles(ref, col, z_reg) -> RegularReferenceProfiles

Evaluate each STORED profile of the run's `PressureReferenceState` at the regular output
levels `z_reg`, through the reference column `col`.

Nothing is re-derived: `Tbar`, `E_tbar` and `Q_ssbar` are fields of the reference object
(see [`RegularReferenceProfiles`](@ref)), so a run with `:consistent_qss_reference` or
`:hydrostatic_reference` on gets exactly the profiles it integrated against.
"""
function regular_reference_profiles(ref::Springsteel.PressureReferenceState, col,
                                    z_reg::Vector{Float64})
    ev(prof) = eval_reference_profile!(col, view(prof, :, 1), z_reg,
                                       zeros(Float64, length(z_reg)))
    return RegularReferenceProfiles(
        copy(z_reg),
        ev(Springsteel.ref_pressure(ref)),
        ev(Springsteel.ref_rho_d(ref)),
        ev(Springsteel.ref_rho_v(ref)),
        ev(Springsteel.ref_rho_c(ref)),
        ev(Springsteel.ref_rho_t(ref)),
        ev(Springsteel.ref_total_energy(ref)),
        ev(Springsteel.ref_qss(ref)),
        ev(ref.Tbar))
end

# ── Derived fields ────────────────────────────────────────────────────────────

"""
    _column_integral(A, z) -> Vector{Float64}

Trapezoid integral of the (x, z) matrix `A` over `z`, one value per column. Exact for a
constant profile, which is what the column-water tests pin.
"""
function _column_integral(A::AbstractMatrix, z::AbstractVector)
    n_i, n_k = size(A)
    out = zeros(Float64, n_i)
    @inbounds for k in 1:(n_k - 1)
        dz = z[k + 1] - z[k]
        for i in 1:n_i
            out[i] += 0.5 * (A[i, k] + A[i, k + 1]) * dz
        end
    end
    return out
end

"""
    mc_derived_fields(z, slots, ref, cfg) -> NamedTuple

The whole derived-product set of a moist-compressible snapshot, from the RAW slot matrices
on the regular grid.

`z` is the regular output level vector; `slots` is a `NamedTuple` of `(n_i, n_k)` matrices
keyed by ROLE, not by slot name -- `p, rho_d, rho_t, E_t, Q_ss, u, w, rain, cloud, rho_v`
are required (the reference-relative ones hold PRIMES, `u`/`w`/`rain`/`cloud` hold whatever
control variable the run declares), and `v`, `n_r`, `ice` (an `NTuple{12,Matrix}` in
[`MC_ICE_VARS`](@ref) order) and `rho_e` are supplied when the configuration has them.

The order is the model's own:

 1. **recover** the transformed species -- `rho_r` and the rain number are TOTALS
    ([`recover_total`](@ref)), the cloud is a PERTURBATION off `ρ̄_c`
    ([`recover_rho_c`](@ref)), the twelve ice moments are totals with the width of their
    moment KIND ([`ice_mu`](@ref));
 2. **add the reference back** for the five reference-relative controls, and for the vapor,
    whose reference is the DERIVED `ρ̄_t − ρ̄_d − ρ̄_c` of [`vapor_slot`](@ref) rather than
    `ref_rho_v` -- the same expression `mc_driver!` adds back;
 3. **retrieve the temperature** with [`retrieve_temperature`](@ref) from the available
    enthalpy `M = p + E_t − ρ_t(ke + gz)`, reading the FLOORED condensate exactly where
    `cfg.cond_floor` says the run's thermodynamics reads it (`options[:condensate_floor]`);
 4. saturation, potential temperature, mixing ratios, entropy and θ_e;
 5. reflectivity, the rain fall speed of the run's OWN closure (two-moment when
    `rain_moments == 2`), the sedimentation rain rate and its surface value;
 6. trapezoid column integrals.

Every quantity here is a DIAGNOSTIC read of the state; nothing is written back and no
tolerance or clamp is applied to the state itself. The `max(·, 0)` calls in the entropy and
θ_e arguments guard `log` of a negative mixing ratio at an under-resolved point (negative
water is a resolution diagnostic in this model, never a state to repair -- see
reference/FINDINGS_NEGATIVE_WATER_ATTRIBUTION.md); `q_v`, `q_c`, `q_r` and the densities
themselves are reported RAW so that diagnostic stays visible.
"""
function mc_derived_fields(z::AbstractVector{Float64}, slots::NamedTuple,
                           ref::RegularReferenceProfiles, cfg::DerivedConfig)

    n_i, n_k = size(slots.p)
    length(z) == n_k || error("mc_derived_fields: z has $(length(z)) levels but the slot " *
                              "matrices have $n_k")
    length(ref.pbar) == n_k || error("mc_derived_fields: the reference profiles have " *
                                     "$(length(ref.pbar)) levels but the slot matrices " *
                                     "have $n_k")
    col(a) = reshape(a, 1, n_k)          # a z-profile broadcast across x

    # ── (1) Species recovery ────────────────────────────────────────────────
    rho_r = recover_rho_r.(slots.rain, cfg.rtrans, cfg.rmu)
    rho_c = recover_rho_c.(slots.cloud, col(ref.rho_cbar), cfg.ctrans, cfg.cmu)
    n_r = haskey(slots, :n_r) ? recover_n_r.(slots.n_r, cfg.nrtrans, cfg.nrmu) : nothing
    # The twelve ice moments, each with the width of its moment KIND (mass, number, a, c
    # cycle with period 4 through MC_ICE_VARS -- see MC_ICE_MU_KEYS).
    ice = if cfg.ice && haskey(slots, :ice)
        ntuple(j -> recover_total.(slots.ice[j], cfg.itrans, cfg.imu[((j - 1) % 4) + 1]), 12)
    else
        nothing
    end

    # ── (2) Totals ──────────────────────────────────────────────────────────
    p     = slots.p     .+ col(ref.pbar)
    rho_d = slots.rho_d .+ col(ref.rho_dbar)
    rho_t = slots.rho_t .+ col(ref.rho_tbar)
    E_t   = slots.E_t   .+ col(ref.E_tbar)
    Q_ss  = slots.Q_ss  .+ col(ref.Q_ssbar)
    rho_v = slots.rho_v .+ col(ref.rho_tbar .- ref.rho_dbar .- ref.rho_cbar)

    # ── (3) Retrieval ───────────────────────────────────────────────────────
    u = slots.u; w = slots.w
    v = cfg.has_v && haskey(slots, :v) ? slots.v : nothing
    ke = v === nothing ? 0.5 .* (u .^ 2 .+ w .^ 2) :
                         0.5 .* (u .^ 2 .+ v .^ 2 .+ w .^ 2)
    M = p .+ E_t .- rho_t .* (ke .+ (gravity .* col(z)))
    rho_liq = rho_c .+ rho_r
    rho_liq_t = cfg.cond_floor ? max.(rho_c, 0.0) .+ max.(rho_r, 0.0) : rho_liq
    rho_ice = ice === nothing ? zeros(Float64, n_i, n_k) : (ice[1] .+ ice[5]) .+ ice[9]
    rho_ice_t = if ice === nothing
        rho_ice
    elseif cfg.cond_floor
        (max.(ice[1], 0.0) .+ max.(ice[5], 0.0)) .+ max.(ice[9], 0.0)
    else
        rho_ice
    end
    T = retrieve_temperature.(M, rho_d, rho_t, rho_liq_t, rho_ice_t)

    # ── (4) Saturation and the thermodynamic diagnostics ────────────────────
    p_hPa = p ./ 100.0
    # Liquid saturation at ALL temperatures: that is what Q_ss saturates against in this
    # equation set (`rho_vs = rho_v_sat(Tk, p_hPa)` in mc_driver!), so an RH built on the
    # ice curve below freezing would not be the model's own supersaturation.
    RH = rho_v ./ Springsteel.Thermodynamics.rho_v_sat.(T, p_hPa)
    RH_ice = cfg.ice ? rho_v ./ Springsteel.Thermodynamics.rho_i_sat.(T, p_hPa) : nothing
    theta = Springsteel.Thermodynamics.potential_temperature.(p, rho_d)   # p in Pa
    q_v = rho_v ./ rho_d
    q_c = rho_c ./ rho_d
    q_r = rho_r ./ rho_d
    q_i = ice === nothing ? nothing : rho_ice_t ./ rho_d
    s = Springsteel.Thermodynamics.entropy.(T, rho_d, max.(q_v, 0.0))
    theta_e = Springsteel.Thermodynamics.reversible_theta_e.(s, rho_d, max.(q_v, 0.0),
                                                             max.(q_c, 0.0))

    # ── (5) Radar and precipitation ─────────────────────────────────────────
    reflectivity = reflectivity_dBZ.(rho_c, rho_r, cfg.N0, cfg.Nc_cm3 * 1.0e6)
    # The fall speed of the run's OWN rain closure: the two-moment (ρ_r, n_r) exponential
    # DSD when the run carries a rain number, Ooyama's single-moment power law otherwise.
    # Reading the wrong one is a factor-of-two error in the rain rate wherever the drop
    # size differs from the single-moment closure's implied one.
    Vt = if cfg.rain_moments == 2 && n_r !== nothing
        first.(rain_fall_speeds_2m.(rho_r, n_r, rho_d))
    else
        rain_terminal_velocity.(rho_r, rho_d, T)
    end
    rain_rate = .-rho_r .* Vt .* 3600.0          # [mm/hr] (ρ_l = 1000 kg/m³)
    precip_rate = rain_rate[:, 1]                # surface value, one per column

    # ── (6) Column integrals ────────────────────────────────────────────────
    PW = _column_integral(rho_v, z)
    column_cloud_water = _column_integral(rho_c, z)
    column_rain_water = _column_integral(rho_r, z)
    column_ice_water = ice === nothing ? nothing : _column_integral(rho_ice_t, z)

    out = (; p, rho_d, rho_t, E_t, Q_ss, rho_v, rho_c, rho_r, rho_liq, rho_ice,
             u, w, ke, M, T, theta, theta_e, s, RH, q_v, q_c, q_r,
             reflectivity, Vt, rain_rate, precip_rate,
             PW, column_cloud_water, column_rain_water)
    v === nothing || (out = merge(out, (; v)))
    n_r === nothing || (out = merge(out, (; n_r)))
    if ice !== nothing
        out = merge(out, (; ice, q_i, RH_ice, column_ice_water))
    end
    if haskey(slots, :rho_e)
        out = merge(out, (; rho_e = slots.rho_e, e = slots.rho_e ./ rho_t))
    end
    return out
end

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — master-side context and writer
# ══════════════════════════════════════════════════════════════════════════════

# ── Option validation ─────────────────────────────────────────────────────────
# The `MYNN_OPTION_KEYS` pattern (src/mynn_state.jl): a closed key set, a prefix scan that
# refuses an unknown `:netcdf_*` key, and errors that spell out the fix. An output option
# is exactly the kind of thing a typo hides in -- `:netcdf_derivative` silently doing
# nothing costs a rerun, and `:output_formats => :csv` (a bare Symbol) used to iterate the
# CHARACTERS of the symbol and fail somewhere unrecognizable.

"The `:netcdf_*` option keys `write_output` understands. Anything else with that prefix is a typo."
const NETCDF_OPTION_KEYS = Set{Symbol}((:netcdf_grid, :netcdf_derivatives))

"The analysis formats `options[:output_formats]` may list."
const OUTPUT_FORMAT_SYMBOLS = (:csv, :netcdf, :netcdf_raw)

"""
    validate_output_options(model::ModelParameters) -> Nothing

Check `options[:output_formats]` and the `:netcdf_*` keys LOUDLY, and return nothing.

Called once from [`initialize_model`](@ref) before any worker is set up -- a run whose
output configuration is wrong must die at setup, not after the first output interval -- and
again (cheaply) at the top of [`write_output`](@ref), so an in-process or REPL call that
never touches `initialize_model` is checked too.

Errors:
- `:output_formats` that is not a `Vector{Symbol}` (a bare `Symbol`, a `Vector{String}`);
- an entry outside `(:csv, :netcdf, :netcdf_raw)`, with `:jld2` keeping its own message
  pointing at `restart_interval`;
- `options[:netcdf_grid]` other than `:regular` (`:mish` is reserved, not implemented);
- an unrecognized `:netcdf_*` key.

`:netcdf_derivatives` applies to `:netcdf_raw` ONLY: the comprehensive file's variables are
derived quantities with no spline derivative slots at all, so a derivative knob on it would
be a promise the layout cannot keep.
"""
function validate_output_options(model::ModelParameters)
    opts = model.options

    if haskey(opts, :output_formats)
        formats = opts[:output_formats]
        isa(formats, AbstractVector) || error(
            "options[:output_formats] must be a Vector{Symbol}, got $(repr(formats)). " *
            "Write it as a LIST even for one format: [:netcdf], not :netcdf.")
        all(f -> isa(f, Symbol), formats) || error(
            "options[:output_formats] must hold Symbols, got $(repr(formats)). " *
            "Write [:csv], not [\"csv\"].")
        for fmt in formats
            fmt === :jld2 && error(
                "`:jld2` is not an analysis output format — it is the restart " *
                "format, written at model.restart_interval by write_restart. " *
                "Set restart_interval (e.g. = output_interval) instead of listing " *
                ":jld2 in options[:output_formats].")
            (fmt in OUTPUT_FORMAT_SYMBOLS) || error(
                "Unknown output format :$(fmt) in options[:output_formats]; supported: " *
                join(string.(":", OUTPUT_FORMAT_SYMBOLS), ", ") *
                " (:netcdf is the comprehensive <t>.nc, :netcdf_raw the legacy " *
                "prognostic-only <t>_raw.nc)")
        end
    end

    grid_mode = get(opts, :netcdf_grid, :regular)
    if grid_mode !== :regular
        grid_mode === :mish && error(
            "options[:netcdf_grid] = :mish is RESERVED and not implemented: the " *
            "comprehensive file is written on the regular output grid " *
            "(i_regular_out x k_regular_out). Use :regular (the default).")
        error("options[:netcdf_grid] = $(repr(grid_mode)) is not recognized; use " *
              ":regular (the default; :mish is reserved and not implemented)")
    end

    haskey(opts, :netcdf_derivatives) &&
        (opts[:netcdf_derivatives] isa Bool || error(
            "options[:netcdf_derivatives] must be a Bool, got " *
            "$(repr(opts[:netcdf_derivatives])). It applies to :netcdf_raw only — the " *
            "comprehensive :netcdf file carries derived products, which have no spline " *
            "derivative slots."))

    for key in keys(opts)
        (startswith(String(key), "netcdf_") && !(key in NETCDF_OPTION_KEYS)) && error(
            "options[:$key] is not a recognized NetCDF output option. The NetCDF keys " *
            "are " * join(string.(":", sort!(collect(NETCDF_OPTION_KEYS); by = String)),
                          ", ") * " (the format list itself is :output_formats)")
    end
    return nothing
end

# ── Output context ────────────────────────────────────────────────────────────

"""
    NetCDFOutputContext

Everything the comprehensive writer needs that does not change between output times: the
regular output coordinates, the reference profiles evaluated on them, the resolved
[`DerivedConfig`](@ref), and the static global attributes.

Built ONCE per patch (`netcdf_output_context`) and threaded through `run_model` /
`model_loop` / `finalize_model` / `run_nested_patch`, because building it means
constructing the reference state — reading and fitting the run's `.ref` file — which must
not happen every output interval.

`active` is false for a run whose formats do not include `:netcdf`, and for an equation set
that is not a pressure-reference (`moist_compressible`) one: there is no reference state to
add back and no water partition to recover, so those runs get the prognostic file instead.

`physics_weights` caches the interpolation weights that map each physics group's own mish
onto this grid (`interp_weights`, above), keyed by group name. The mishes are fixed for the
life of a run, so the weights are built on the first output time that carries a group and
reused for every one after. `nothing` until then.
"""
mutable struct NetCDFOutputContext
    active::Bool
    x_reg::Vector{Float64}
    z_reg::Vector{Float64}
    ref::Union{Nothing,RegularReferenceProfiles}
    cfg::DerivedConfig
    static_attrs::Dict{String,Any}
    physics_weights::Any
end

"""
    comprehensive_netcdf_eligible(grid, model) -> Bool

True when [`write_netcdf_comprehensive`](@ref) can write this grid: a pressure-reference
(moist-compressible) equation set on a 2-D k-active grid (RiRk / RZ — the XZ slice and the
axisymmetric cylinder), with a reference state to add back.

3-D moist-compressible grids (RLR, RRR, SLR) are NOT eligible in stage N1 and fall through
to the prognostic writer; see [`write_netcdf_comprehensive`](@ref) for what that costs.
"""
function comprehensive_netcdf_eligible(grid::AbstractGrid, model::ModelParameters)
    uses_pressure_reference(model.equation_set) || return false
    isempty(model.ref_state_file) && return false
    grid.jbasis isa Springsteel.NoBasisArray || return false
    grid.kbasis isa Springsteel.NoBasisArray && return false
    return true
end

"Cylindrical (r, z) rather than Cartesian (x, z): the coordinate long_name follows."
_mc_is_cylindrical(equation_set::AbstractString) =
    endswith(equation_set, "_axisym") || endswith(equation_set, "_RLR")

"""
    netcdf_output_context(grid, model) -> NetCDFOutputContext

Build the per-patch comprehensive-output context: the regular coordinates, the run's own
reference state evaluated on them, the resolved derived-field configuration and the static
global attributes.

The regular grid is constructed EXACTLY as Springsteel's own `write_netcdf` builds it
(`LinRange(iMin, iMax, i_regular_out)` and the matching `k`), and the values come from the
same `getRegularGridpoints`/`regularGridTransform` pair, so a variable that appears in both
the comprehensive file and the legacy `:netcdf_raw` one carries bit-identical numbers.

The reference state is built through [`build_reference_state`](@ref) — the same function
`createModelTile` calls, with the same arguments — so the profiles here are the ones the
run integrates against, including every opt-in (`:hydrostatic_reference`,
`:consistent_qss_reference`, `:exact_reference_state`).
"""
function netcdf_output_context(grid::AbstractGrid, model::ModelParameters)
    gp = grid.params
    cfg = derived_config(model)
    # A run that never asks for :netcdf must not pay for a reference-state construction
    # here -- `integrate_model` builds this context unconditionally, before it knows or
    # cares which formats the run wants.
    formats = get(model.options, :output_formats, [:netcdf])
    wants_netcdf = any(f -> f === :netcdf, formats)
    active = wants_netcdf && comprehensive_netcdf_eligible(grid, model)
    x_reg = collect(LinRange(gp.iMin, gp.iMax, gp.i_regular_out))
    z_reg = active ? collect(LinRange(gp.kMin, gp.kMax, gp.k_regular_out)) : Float64[]

    ref_profiles = nothing
    if active
        z_values = getGridpoints(grid)[1:gp.kDim, end]
        ref_column = reference_column(grid, gp)
        ref_state = build_reference_state(model, z_values, ref_column)
        ref_state isa Springsteel.PressureReferenceState || error(
            "netcdf_output_context: equation_set \"$(model.equation_set)\" is a " *
            "pressure-reference set but build_reference_state returned a " *
            "$(typeof(ref_state)); the comprehensive NetCDF writer needs the stored " *
            "p/rho/E_t/Q_ss profiles")
        ref_profiles = regular_reference_profiles(ref_state, ref_column, z_reg)
    end

    attrs = Dict{String,Any}(
        "Conventions" => "CF-1.12",
        "title" => "Scythe.jl moist-compressible comprehensive output",
        "source" => "Scythe.jl",
        "scythe_file_kind" => "comprehensive",
        "netcdf_grid" => "regular",
        "equation_set" => model.equation_set,
        "geometry" => gp.geometry,
        "ts" => model.ts,
        "output_interval" => model.output_interval,
        "integration_time" => model.integration_time,
        "ref_state_file" => model.ref_state_file,
        "exact_reference_state" => Int(get(model.options, :exact_reference_state, false) === true),
        "consistent_qss_reference" => Int(get(model.options, :consistent_qss_reference, false) === true),
        "hydrostatic_reference" => Int(get(model.options, :hydrostatic_reference, false) === true),
        "condensate_transform" => String(cfg.ctrans),
        "condensate_mu" => cfg.cmu,
        "rain_transform" => String(cfg.rtrans),
        "rain_mu" => cfg.rmu,
        "rain_number_transform" => String(cfg.nrtrans),
        "rain_number_mu" => cfg.nrmu,
        "rain_moments" => cfg.rain_moments,
        "ice_microphysics" => cfg.ice ? "ishmael" : "none",
        "ice_transform" => String(cfg.itrans),
        "ice_mu_mass" => cfg.imu[1],
        "ice_mu_number" => cfg.imu[2],
        "ice_mu_a" => cfg.imu[3],
        "ice_mu_c" => cfg.imu[4],
        "reflectivity_N0_m4" => cfg.N0,
        "reflectivity_Nc_cm3" => cfg.Nc_cm3,
        "reflectivity_source" => cfg.N0_source,
        "temperature_retrieval" =>
            "T = (M + rho_liq*(L_v0 - (Cpv-Cl)T_0) + rho_ice*(L_s0 - (Cpv-Ci)T_0)) / " *
            "((rho_d*Cpd + (rho_t-rho_d)*Cpv) - rho_liq*(Cpv-Cl) - rho_ice*(Cpv-Ci)), " *
            "M = p + E_t - rho_t*(ke + g z); condensate read " *
            (cfg.cond_floor ? "FLOORED at zero per species (options[:condensate_floor] " *
                              "= :diagnostic), state left raw" :
                              "RAW (options[:condensate_floor] = :none)"),
        "condensate_floor" => cfg.cond_floor ? "diagnostic" : "none")

    return NetCDFOutputContext(active, x_reg, z_reg, ref_profiles, cfg, attrs, nothing)
end

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2b — physics groups (stage N2)
# ══════════════════════════════════════════════════════════════════════════════
#
# The BL (MYNN), radiation and surface diagnostics live on the WORKERS, in
# `mtile.mynn` / `mtile.radiation` / `mtile.surface`, on each tile's own mish. The
# comprehensive file is written by the MASTER, which holds the patch and no `ModelTile` at
# all. So at an output step -- and ONLY at an output step; nothing here touches the
# per-step path -- the master asks every worker for a snapshot of its held physics
# (`gather_physics`), stitches the tiles back into one patch-wide set (`assemble_physics`)
# and hands it to the writer, which regrids it onto the regular output grid with the same
# separable linear interpolation `tc/tc_postprocess.jl` used to merge the sidecars.
#
# TIMING. The held physics a worker returns at output step `t` is what that step's column
# loop last wrote, which is the same alignment the sidecars have (see `radiation_write!`'s
# docstring): the schemes hold their forcing between calls, so the snapshot tagged `t` is
# the forcing the state at `t` was advanced with.
#
# OMITTED, NEVER ZERO-FILLED. A group is `nothing` when its scheme is off OR has not run
# yet, and a `nothing` group writes no variables at all. A file with a `K_h` of all zeros
# would be indistinguishable from a run whose boundary layer genuinely did nothing; a file
# with no `K_h` cannot be misread.

"""
    surface_diagnostics(mtile::ModelTile) -> NamedTuple

The tile's held per-column surface exchange ([`SurfaceDiag`](@ref)) as plain arrays: `x`
(the tile's own horizontal mish coordinate, the radiation/MYNN convention), the ten
per-column fields of [`SURFACE_FIELDS_1D`](@ref), and `attrs`/`sum_attrs`.

The attributes record which layer produced them: `boundary_layer` (`"louis"`/`"mynn"`),
`surface_fluxes`, and the two knobs that decide the formulas
([`surface_layer_params`](@ref)) -- `sfc_z0` and `sfc_stability`. A flux is only
interpretable against the closure that made it, and this run's is not recoverable from
the numbers.
"""
function surface_diagnostics(mtile::ModelTile)
    sd = mtile.surface
    ncol = length(sd.F_sh)
    kDim = ncol > 0 ? div(size(mtile.tile.physical, 1), ncol) : 0
    x = Vector{Float64}(undef, ncol)
    @inbounds for c in 1:ncol
        x[c] = mtile.tilepoints[(c - 1) * kDim + 1, 1]
    end
    opts = mtile.model.options
    attrs = Dict{String,Any}(
        "boundary_layer" => get(opts, :mynn, false) === true ? "mynn" :
                            get(opts, :louis_bl, false) === true ? "louis" : "none",
        "surface_fluxes" => Int(get(opts, :surface_fluxes, false) === true),
        "sfc_z0" => string(get(opts, :sfc_z0, :komori)),
        "sfc_stability" => Int(get(opts, :sfc_stability, false) === true))
    return (; x, F_sh = copy(sd.F_sh), F_q = copy(sd.F_q),
              tau_u = copy(sd.tau_u), tau_v = copy(sd.tau_v),
              ust = copy(sd.ust), inv_L = copy(sd.inv_L), U10 = copy(sd.U10),
              Cd = copy(sd.Cd), Ch = copy(sd.Ch), z0m = copy(sd.z0m),
              attrs = attrs, sum_attrs = String[])
end

"""
    SURFACE_FIELDS_1D

The `(name, units, long_name)` of every per-column surface-exchange diagnostic, written
under these exact names. `ust` and `inv_L` are the SURFACE LAYER's own, straight out of
[`surface_exchange`](@ref); MYNN's copies of the same two quantities are written
alongside as `mynn_ust`/`mynn_inv_L`.
"""
const SURFACE_FIELDS_1D = (
    ("F_sh", "W m-2", "surface sensible heat flux"),
    ("F_q", "kg m-2 s-1", "surface moisture flux"),
    ("tau_u", "Pa", "surface stress, u component"),
    ("tau_v", "Pa", "surface stress, v component"),
    ("ust", "m s-1", "friction velocity"),
    ("inv_L", "m-1", "inverse Obukhov length, 1/L"),
    ("U10", "m s-1", "10 m wind speed of the roughness fit (the exchange wind on :komori)"),
    ("Cd", "1", "surface drag coefficient"),
    ("Ch", "1", "surface enthalpy/moisture exchange coefficient"),
    ("z0m", "m", "momentum roughness length (0 on :komori, which forms none)"))

"""
    physics_snapshot(mtile::ModelTile) -> NamedTuple

One tile's physics groups, for the master to assemble. Evaluated ON THE WORKER (through
[`gather_physics`](@ref)); the return value is plain arrays and `Dict`s, so it serializes.

`(; offset, ncol, kDim, mynn, radiation, surface)`. `offset` is
`mtile.tile.params.patchOffsetL`, the left-edge gridpoint offset of this tile within its
patch — the same quantity the sidecar filenames carry and the key the tiles are sorted by.

Each group is [`mynn_diagnostics`](@ref) / [`radiation_diagnostics`](@ref) /
[`surface_diagnostics`](@ref), or `nothing` when that scheme is off OR HAS NOT RUN YET.
The three "has run" witnesses are the states' own:

- MYNN: some column has `last_update_step != typemin(Int)` (its cold-start sentinel);
- radiation: `rs.last_call_step != typemin(Int)` (the same sentinel, per tile);
- surface: `sd.n_calls[1] > 0`.

The distinction matters at `t = 0`: `run_model` writes an output file BEFORE the first
timestep, and at that point every held field is an allocation-time zero rather than a
computed one. Writing them would put a physically meaningless `K_h = 0`, `pblh = 0`,
`F_sh = 0` field in the file under the same names a spun-up snapshot uses.
"""
function physics_snapshot(mtile::ModelTile)
    MY = mtile.mynn
    rs = mtile.radiation
    sd = mtile.surface
    kDim = mtile.model.grid_params.kDim
    ncol = kDim > 0 ? div(size(mtile.tile.physical, 1), kDim) : 0
    mynn = (MY.active && any(!=(typemin(Int)), MY.last_update_step)) ?
           mynn_diagnostics(mtile) : nothing
    radiation = (rs.active && rs.last_call_step != typemin(Int)) ?
                radiation_diagnostics(mtile) : nothing
    surface = (sd.active && sd.n_calls[1] > 0) ? surface_diagnostics(mtile) : nothing
    return (; offset = mtile.tile.params.patchOffsetL, ncol, kDim,
              mynn, radiation, surface)
end

"""
    gather_physics(workerids) -> Vector

Ask every worker in `workerids` for its [`physics_snapshot`](@ref). Master-side; the
`get_val_from` pattern `run_model` uses for the reference profiles.

Called once per output time per patch, from [`write_output`](@ref)'s `:netcdf` branch.
"""
gather_physics(workerids::Vector{Int64}) =
    [get_val_from(w, :(Scythe.physics_snapshot(mtile))) for w in workerids]

"The physics group names, in the order [`write_netcdf_comprehensive`](@ref) writes them."
const PHYSICS_GROUPS = (:mynn, :radiation, :surface)

"""
    assemble_physics(snaps) -> Union{Nothing,NamedTuple}

Stitch per-tile [`physics_snapshot`](@ref)s into one patch-wide set, or `nothing` when no
group is present on any tile.

Tiles are sorted by `offset` and every array is concatenated along `x` (dimension 1 of the
`(x, z)` matrices, the whole of the per-column vectors) — the domain decomposition is in
`x` only, so this is the same reassembly `read_mynn`/`read_radiation` do from the sidecar
files, done in memory instead of through the filesystem. `z` is taken from the first tile
(every tile of a patch shares the vertical mish) and CHECKED against the rest.

A group must be present on EVERY tile or on NONE: the schemes are configured per MODEL,
not per tile, so a mixed answer means a tile failed to run one and the file would silently
carry a partial domain. That is an error, loudly.

Attributes come from the first tile — they are the resolved configuration, identical on
every tile by construction — EXCEPT each group's `sum_attrs` (the clamp/plume counters),
which are SUMMED across tiles to give the domain-wide census, exactly as `read_mynn` sums
them.
"""
function assemble_physics(snaps::AbstractVector)
    isempty(snaps) && return nothing
    order = sortperm([s.offset for s in snaps])
    ss = snaps[order]

    out = Dict{Symbol,Any}()
    for g in PHYSICS_GROUPS
        parts = [getfield(s, g) for s in ss]
        present = count(p -> p !== nothing, parts)
        if present == 0
            out[g] = nothing
            continue
        end
        present == length(parts) || error(
            "assemble_physics: the :$g group is present on $present of $(length(parts)) " *
            "tiles. A physics scheme is configured per MODEL, so it must be held by every " *
            "tile of a patch or by none; a partial answer would write a file covering " *
            "only part of the domain under full-domain variable names. Offsets with the " *
            "group: $([s.offset for (s, p) in zip(ss, parts) if p !== nothing]).")
        out[g] = _concat_group(g, parts)
    end
    all(g -> out[g] === nothing, PHYSICS_GROUPS) && return nothing
    return (; mynn = out[:mynn], radiation = out[:radiation], surface = out[:surface])
end

"Concatenate one group's per-tile diagnostics along `x`; see [`assemble_physics`](@ref)."
function _concat_group(g::Symbol, parts::Vector)
    a1 = parts[1]
    length(parts) == 1 && return a1
    for (n, p) in enumerate(parts[2:end])
        keys(p) == keys(a1) || error(
            "assemble_physics: tile $(n + 1)'s :$g group has fields $(keys(p)), the " *
            "first tile's has $(keys(a1)) — the tiles disagree about the diagnostic set")
    end
    vals = map(keys(a1)) do k
        v1 = getfield(a1, k)
        if k === :attrs
            merged = copy(v1)
            for nm in a1.sum_attrs
                merged[nm] = sum(getfield(p, :attrs)[nm] for p in parts)
            end
            merged
        elseif k === :sum_attrs
            v1
        elseif k === :z || k === :zf || k === :q_lw_ref || k === :q_sw_ref
            # Per-LAYER (or per-face) profiles shared by every column of the patch, not
            # per-column fields: concatenating them would multiply the vertical.
            for p in parts
                getfield(p, k) == v1 || error(
                    "assemble_physics: the :$g group's `$k` differs between tiles of the " *
                    "same patch; every tile shares the patch's vertical mish")
            end
            v1
        elseif v1 isa AbstractArray
            vcat((getfield(p, k) for p in parts)...)
        else
            v1
        end
    end
    return NamedTuple{keys(a1)}(vals)
end

# ── Writers ───────────────────────────────────────────────────────────────────

"Reshape the flat regular-transform array to (n_i, n_k) for one variable's VALUE slot."
function _reshape_regular(reg_phys, var_idx::Int, n_i::Int, n_k::Int)
    data = Matrix{Float64}(undef, n_i, n_k)
    @inbounds for i in 1:n_i, k in 1:n_k
        data[i, k] = reg_phys[(i - 1) * n_k + k, var_idx, 1]
    end
    return data
end

"The regridding note both physics groups carry, so a reader of either knows the fields
were computed somewhere else and linearly resampled here."
const PHYSICS_REGRID_NOTE =
    "These fields were computed on the scheme's OWN mish and are resampled onto this " *
    "file's regular (x, z) grid by separable LINEAR interpolation, x then z, with edge " *
    "clamping outside the mish hull (the outermost Gauss points lie inside the " *
    "first/last cell, so the wall and the model top are outside it by up to half a cell " *
    "and hold the edge value)."

"""
    _physics_weights!(ctx, key, xs, zs) -> (wx, wz)

The interpolation weights from a physics group's mish (`xs`, `zs`) onto the context's
regular grid, built on first use and CACHED on `ctx.physics_weights` under `key`.

The mish does not move for the life of a run, so the weights are computed once per group
per patch however many output times the run writes. `zs === nothing` (the surface group,
which is per-column only) gives `wz === nothing`.
"""
function _physics_weights!(ctx::NetCDFOutputContext, key::Symbol,
                           xs::AbstractVector, zs::Union{Nothing,AbstractVector})
    ctx.physics_weights === nothing && (ctx.physics_weights = Dict{Symbol,Any}())
    W = ctx.physics_weights::Dict{Symbol,Any}
    return get!(W, key) do
        (interp_weights(xs, ctx.x_reg),
         zs === nothing ? nothing : interp_weights(zs, ctx.z_reg))
    end
end

"""
    _write_group_attrs!(ds, prefix, attrs) -> Nothing

Write one physics group's global attributes under `prefix`, leaving alone any name that
already carries it (`mynn_interval_s`, `radiation_interval_s`) so the result is
`mynn_interval_s` and not `mynn_mynn_interval_s`. Sorted, so the file's attribute order is
reproducible rather than `Dict`-iteration order.
"""
function _write_group_attrs!(ds, prefix::String, attrs::AbstractDict)
    for k in sort!(collect(keys(attrs)))
        name = (isempty(prefix) || startswith(k, prefix)) ? k : prefix * k
        ds.attrib[name] = attrs[k]
    end
    return nothing
end

"""
    write_netcdf_comprehensive(path, grid, model, t, ctx, physics = nothing) -> Nothing

Write ONE comprehensive `<tag>.nc` snapshot: prognostic primes, raw transformed control
variables where a transform is on, the reconstructed totals, the recovered hydrometeors,
the retrieved thermodynamics, the radar/precipitation products, the column integrals and
the 1-D reference profiles, all on the regular (x, z) output grid.

`ctx` is the [`NetCDFOutputContext`](@ref) built once per patch. `physics` is the
patch-wide physics-group set [`assemble_physics`](@ref) built from the workers' snapshots:
the BL (MYNN), radiation and surface groups, each on its own mish and each resampled here
onto the regular output grid. `nothing` — a run with no physics, or one whose schemes have
not run yet — writes no physics variables and records that in `physics_groups`.

!!! note "2-D only in stage N1"
    Only the 2-D k-active moist-compressible grids (RiRk / RZ: the XZ slice and the
    axisymmetric cylinder) are written comprehensively. A 3-D moist-compressible grid
    (RLR, RRR, SLR) falls through to [`write_netcdf_prognostic`](@ref) and gets the
    prognostic-only layout — the derived set on `(time, r, azimuth, z)` is a layout change
    rather than a physics change and is deferred so it can be tested against a real 3-D
    run. `comprehensive_netcdf_eligible` is the gate.
"""
function write_netcdf_comprehensive(path::String, grid::AbstractGrid,
                                    model::ModelParameters, t::Float64,
                                    ctx::NetCDFOutputContext,
                                    physics::Union{Nothing,NamedTuple} = nothing)

    ref = ctx.ref
    ref === nothing && error("write_netcdf_comprehensive: the output context carries no " *
                             "reference profiles; call netcdf_output_context first")
    cfg = ctx.cfg
    gp = grid.params
    vars = gp.vars
    n_i = length(ctx.x_reg); n_k = length(ctx.z_reg)

    reg_pts = Springsteel.getRegularGridpoints(grid)
    reg_phys = Springsteel.regularGridTransform(grid, reg_pts)
    slot(idx::Int) = _reshape_regular(reg_phys, idx, n_i, n_k)

    # ── Slot roles. Resolved BY NAME through `mc_slot`/`mc_optional_slot`, so a
    # transformed run's `nu_c`/`nu_r`/`nu_i*` columns are found under their own names and
    # a configuration that never registered a slot returns 0 rather than throwing.
    rain_i = mc_slot(vars, "rho_r")
    cloud_i = mc_slot(vars, "rho_c")
    nr_i = mc_optional_slot(vars, "n_r")
    ice_idx = mc_ice_slot_indices(vars)
    v_i = get(vars, "v", 0)
    rho_e_i = get(vars, "rho_e", 0)

    p_p = slot(vars["p"]); rho_d_p = slot(vars["rho_d"]); rho_t_p = slot(vars["rho_t"])
    E_t_p = slot(vars["E_t"]); Q_ss_p = slot(vars["Q_ss"]); rho_v_p = slot(vars["rho_v"])
    u = slot(vars["u"]); w = slot(vars["w"])
    rain_raw = slot(rain_i); cloud_raw = slot(cloud_i)

    slots = (; p = p_p, rho_d = rho_d_p, rho_t = rho_t_p, E_t = E_t_p, Q_ss = Q_ss_p,
               rho_v = rho_v_p, u = u, w = w, rain = rain_raw, cloud = cloud_raw)
    v_i > 0 && (slots = merge(slots, (; v = slot(v_i))))
    nr_i > 0 && (slots = merge(slots, (; n_r = slot(nr_i))))
    ice_on = cfg.ice && all(>(0), ice_idx)
    ice_raw = ice_on ? ntuple(j -> slot(ice_idx[j]), 12) : nothing
    ice_on && (slots = merge(slots, (; ice = ice_raw)))
    rho_e_i > 0 && (slots = merge(slots, (; rho_e = slot(rho_e_i))))

    d = mc_derived_fields(ctx.z_reg, slots, ref, cfg)

    cyl = _mc_is_cylindrical(model.equation_set)

    NCDatasets.NCDataset(path, "c") do ds
        for (k, v) in ctx.static_attrs
            ds.attrib[k] = v
        end
        # `Libc.strftime` rather than `Dates.format`: Dates is not a Scythe dependency
        # (it is a test-only extra), and a timestamp is not worth adding one for.
        ds.attrib["history"] = "Created " *
            Libc.strftime("%Y-%m-%d %H:%M:%S", time()) * " by Scythe.write_output"

        NCDatasets.defDim(ds, "time", 1)
        NCDatasets.defDim(ds, "x", n_i)
        NCDatasets.defDim(ds, "z", n_k)

        tv = NCDatasets.defVar(ds, "time", Float64, ("time",))
        tv.attrib["units"] = "seconds"
        tv.attrib["long_name"] = "simulation time"
        tv[1] = t

        xv = NCDatasets.defVar(ds, "x", Float64, ("x",))
        xv.attrib["units"] = "m"
        xv.attrib["long_name"] = cyl ? "radius" : "horizontal coordinate"
        xv[:] = ctx.x_reg

        zv = NCDatasets.defVar(ds, "z", Float64, ("z",))
        zv.attrib["units"] = "m"
        zv.attrib["long_name"] = "height"
        zv[:] = ctx.z_reg

        wr(name, data, units, long) = begin
            dv = NCDatasets.defVar(ds, name, Float64, ("time", "x", "z");
                                   deflatelevel = 4, fillvalue = NaN)
            dv.attrib["units"] = units; dv.attrib["long_name"] = long
            dv[1, :, :] = data
        end
        wrx(name, data, units, long) = begin
            dv = NCDatasets.defVar(ds, name, Float64, ("time", "x");
                                   deflatelevel = 4, fillvalue = NaN)
            dv.attrib["units"] = units; dv.attrib["long_name"] = long
            dv[1, :] = data
        end
        wrz(name, data, units, long) = begin
            dv = NCDatasets.defVar(ds, name, Float64, ("z",); fillvalue = NaN)
            dv.attrib["units"] = units; dv.attrib["long_name"] = long
            dv[:] = data
        end

        # ── Prognostic perturbations off the hydrostatic reference ──
        wr("p_prime",     p_p,     "Pa",     "pressure perturbation")
        wr("rho_d_prime", rho_d_p, "kg m-3", "dry-air density perturbation")
        wr("rho_t_prime", rho_t_p, "kg m-3", "total density perturbation")
        wr("E_t_prime",   E_t_p,   "J m-3",  "total energy density perturbation")
        wr("Q_ss_prime",  Q_ss_p,  "kg m-3", "supersaturation density perturbation")
        wr("rho_v_prime", rho_v_p, "kg m-3",
           "vapor density perturbation off rho_tbar - rho_dbar - rho_cbar")

        # ── Raw transformed control variables, under their OWN slot names, and only
        # where a transform is actually on. Under :none the slot IS the density and is
        # already written as a total below; a duplicate column named `rho_c` would be a
        # second copy of the same numbers.
        cfg.ctrans === :none ||
            wr(condensate_var_name(model.options), cloud_raw, "1",
               "cloud control variable ($(cfg.ctrans), mu = $(cfg.cmu))")
        cfg.rtrans === :none ||
            wr(rain_var_name(model.options), rain_raw, "1",
               "rain control variable ($(cfg.rtrans), mu = $(cfg.rmu))")
        if nr_i > 0 && cfg.nrtrans !== :none
            wr(rain_number_var_name(model.options), slots.n_r, "1",
               "rain number control variable ($(cfg.nrtrans), mu = $(cfg.nrmu))")
        end
        if ice_on && cfg.itrans !== :none
            inames = ice_var_names(model.options)
            for j in 1:12
                wr(inames[j], ice_raw[j], "1",
                   "ice control variable for $(MC_ICE_VARS[j]) ($(cfg.itrans))")
            end
        end

        # ── Totals ──
        wr("p",     d.p,     "Pa",     "total pressure")
        wr("rho_d", d.rho_d, "kg m-3", "total dry-air density")
        wr("rho_t", d.rho_t, "kg m-3", "total density")
        wr("E_t",   d.E_t,   "J m-3",  "total energy density")
        wr("Q_ss",  d.Q_ss,  "kg m-3", "supersaturation density")
        wr("rho_v", d.rho_v, "kg m-3", "water vapor density")
        wr("rho_c", d.rho_c, "kg m-3", "cloud water density")
        wr("rho_r", d.rho_r, "kg m-3", "rain water density")
        nr_i > 0 && wr("n_r", d.n_r, "m-3", "rain number density")

        # ── Winds ──
        wr("u", u, "m s-1", cyl ? "radial velocity" : "horizontal velocity")
        wr("w", w, "m s-1", "vertical velocity")
        v_i > 0 && wr("v", d.v, "m s-1", cyl ? "tangential velocity" : "y velocity")

        # ── Ice moments ──
        if ice_on
            for (j, nm) in enumerate(MC_ICE_VARS)
                units = j % 4 == 1 ? "kg m-3" : j % 4 == 2 ? "m-3" : "m3 m-3"
                wr(nm, d.ice[j], units, "ISHMAEL ice moment $nm")
            end
            wr("rho_ice", d.rho_ice, "kg m-3", "total ice density, sum over species")
        end

        # ── Retrieved thermodynamics ──
        wr("T", d.T, "K", "temperature (closed-form nonlinear retrieval)")
        wr("theta", d.theta, "K", "potential temperature")
        wr("theta_e", d.theta_e, "K", "reversible equivalent potential temperature")
        wr("RH", d.RH, "1", "relative humidity over liquid water, rho_v / rho_v_sat(T, p)")
        ice_on && wr("RH_ice", d.RH_ice, "1", "relative humidity over ice")
        wr("q_v", d.q_v, "kg kg-1", "vapor mixing ratio, rho_v / rho_d")
        wr("q_c", d.q_c, "kg kg-1", "cloud mixing ratio, rho_c / rho_d")
        wr("q_r", d.q_r, "kg kg-1", "rain mixing ratio, rho_r / rho_d")
        ice_on && wr("q_i", d.q_i, "kg kg-1", "ice mixing ratio, rho_ice / rho_d")

        # ── MYNN TKE, when the run carries the rho_e slot ──
        if rho_e_i > 0
            wr("rho_e", d.rho_e, "J m-3", "turbulence kinetic energy density")
            wr("e", d.e, "m2 s-2", "mass-specific turbulence kinetic energy, rho_e / rho_t")
        end

        # ── Radar and precipitation ──
        wr("reflectivity", d.reflectivity, "dBZ", "S-band equivalent radar reflectivity")
        wr("rain_rate", d.rain_rate, "mm hr-1", "rain rate from the sedimentation flux")
        wrx("precip_rate", d.precip_rate, "mm hr-1", "surface rain rate")

        # ── Column integrals ──
        wrx("PW", d.PW, "kg m-2", "precipitable water, column integral of rho_v")
        wrx("column_cloud_water", d.column_cloud_water, "kg m-2",
            "column integral of rho_c")
        wrx("column_rain_water", d.column_rain_water, "kg m-2", "column integral of rho_r")
        ice_on && wrx("column_ice_water", d.column_ice_water, "kg m-2",
                      "column integral of rho_ice")

        # ── Physics groups (N2): BL, radiation, surface ──
        # Each group lives on its OWN mish and is resampled onto this file's regular
        # (x, z) grid by separable LINEAR interpolation, x then z, edge-clamped outside
        # the mish hull (the outermost Gauss points lie inside the first/last cell, so
        # the wall and the model top are outside it by up to half a cell and hold the
        # edge value). That is the convention `tc/tc_postprocess.jl` established when it
        # merged the sidecars, kept here verbatim so the two files agree.
        groups = String[]
        if physics !== nothing
            mynn = physics.mynn
            if mynn !== nothing
                push!(groups, "mynn")
                wxm, wzm = _physics_weights!(ctx, :mynn, mynn.x, mynn.z)
                for (nm, units, long) in MYNN_FIELDS_2D
                    # `K_m`/`K_h` keep their bare names -- an exchange coefficient is
                    # unambiguous and the prefix would only make every downstream reader
                    # spell it out again -- everything else is prefixed.
                    out = nm in ("K_m", "K_h") ? nm : "mynn_" * nm
                    lbl = nm in ("K_m", "K_h") ? "MYNN " * long : long
                    wr(out, regrid2d(getfield(mynn, Symbol(nm)), wxm, wzm), units, lbl)
                end
                for (nm, units, long) in MYNN_FIELDS_1D
                    wrx("mynn_" * nm, regrid1d(getfield(mynn, Symbol(nm)), wxm),
                        units, long)
                end
                _write_group_attrs!(ds, "mynn_", mynn.attrs)
                ds.attrib["mynn_regridding"] = PHYSICS_REGRID_NOTE
            end

            rad = physics.radiation
            if rad !== nothing
                push!(groups, "radiation")
                wxr, wzr = _physics_weights!(ctx, :radiation, rad.x, rad.z)
                for (nm, units, long) in RADIATION_FIELDS_2D
                    wr(nm, regrid2d(getfield(rad, Symbol(nm)), wxr, wzr), units, long)
                end
                # The rate a column actually felt, formed on the mish and regridded once
                # (not from the two regridded fields, which would interpolate twice).
                wr("dT_net", regrid2d(rad.dT_lw .+ rad.sw_scale .* rad.dT_sw, wxr, wzr),
                   "K day-1",
                   "net radiative heating at constant pressure, dT_lw + sw_scale*dT_sw")
                for (nm, units, long) in RADIATION_FIELDS_1D
                    lbl = nm == "cloudy" ?
                        "cloudy-column indicator, linearly interpolated off the mish: " *
                        "0 or 1 at a mish column, fractional between two" : long
                    wrx(nm, regrid1d(getfield(rad, Symbol(nm)), wxr), units, lbl)
                end
                _write_group_attrs!(ds, "radiation_", rad.attrs)
                # The snapshot's own model time, the name `tc/tc_postprocess.jl` gave it.
                ds.attrib["radiation_time"] = t
                ds.attrib["radiation_regridding"] = PHYSICS_REGRID_NOTE * " " *
                    "The face-based flux profiles (flux_lw_*, flux_sw_*, on zf, which " *
                    "runs to the top of the stratospheric extension) are NOT written: " *
                    "they have no counterpart on this regular z grid and remain in the " *
                    "radiation sidecar (options[:radiation_output]). The per-column " *
                    "boundary values derived from them (olr, olr_model_top, " *
                    "sw_sfc_dn, sw_toa_dn, lw_sfc_dn, ...) are here."
            end

            sfc = physics.surface
            if sfc !== nothing
                push!(groups, "surface")
                wxs, _ = _physics_weights!(ctx, :surface, sfc.x, nothing)
                for (nm, units, long) in SURFACE_FIELDS_1D
                    wrx(nm, regrid1d(getfield(sfc, Symbol(nm)), wxs), units, long)
                end
                _write_group_attrs!(ds, "", sfc.attrs)
            end
        end
        ds.attrib["physics_groups"] =
            isempty(groups) ? "none (schemes not yet run)" : join(groups, ",")

        # ── 1-D reference profiles the totals were built from ──
        for (nm, prof, units) in (("pbar", ref.pbar, "Pa"),
                                  ("rho_dbar", ref.rho_dbar, "kg m-3"),
                                  ("rho_vbar", ref.rho_vbar, "kg m-3"),
                                  ("rho_cbar", ref.rho_cbar, "kg m-3"),
                                  ("rho_tbar", ref.rho_tbar, "kg m-3"),
                                  ("E_tbar", ref.E_tbar, "J m-3"),
                                  ("Q_ssbar", ref.Q_ssbar, "kg m-3"),
                                  ("Tbar", ref.Tbar, "K"))
            wrz(nm, prof, units, "reference-state $nm")
        end
    end
    return nothing
end

"""
    write_netcdf_prognostic(path, grid, model, t) -> Nothing

Prognostic-only NetCDF for a grid the comprehensive writer does not cover: every equation
set that is not a pressure-reference one, and the 3-D moist-compressible grids.

This is Springsteel's `write_netcdf` with real coordinate units attached (it defaults every
coordinate to `"1"`) and a `scythe_file_kind = "prognostic"` marker so a reader can tell at
a glance which layout it has, without inspecting the variable list.
"""
function write_netcdf_prognostic(path::String, grid::AbstractGrid,
                                 model::ModelParameters, t::Float64)
    include_derivs = get(model.options, :netcdf_derivatives, false)::Bool
    coord_attrs = Dict{String,Dict{String,Any}}(
        "x" => Dict{String,Any}("units" => "m"),
        "y" => Dict{String,Any}("units" => "m"),
        "z" => Dict{String,Any}("units" => "m"),
        "radius" => Dict{String,Any}("units" => "m"),
        "height" => Dict{String,Any}("units" => "m"),
        "azimuth" => Dict{String,Any}("units" => "degrees"),
        "latitude" => Dict{String,Any}("units" => "degrees_north"),
        "longitude" => Dict{String,Any}("units" => "degrees_east"))
    global_attrs = Dict{String,Any}(
        "source" => "Scythe.jl",
        "scythe_file_kind" => "prognostic",
        "equation_set" => model.equation_set,
        "geometry" => grid.params.geometry,
        "ts" => model.ts,
        "output_interval" => model.output_interval,
        "integration_time" => model.integration_time,
        "physics_groups" => "none (prognostic layout)")
    Springsteel.write_netcdf(path, grid; include_derivatives = include_derivs, time = t,
                             coordinate_attributes = coord_attrs,
                             global_attributes = global_attrs)
    return nothing
end
