# ── Radiation state containers and the RRTMGP-independent column geometry ─────
#
# This file is the LEAF half of the radiation driver: the structs `ModelTile`
# carries, and the pure arithmetic that turns a Scythe column into the shape a
# radiative-transfer solver wants (layer faces, level pressures/temperatures,
# solar geometry, the sponge taper, the option table). It names no radiation
# library at all -- every RRTMGP reference lives in src/radiation.jl, which is
# included after moist_compressible.jl. The split exists for two reasons:
#
#   1. `ModelTile` holds its `RadiationState` CONCRETELY (same argument as
#      `IshmaelTables`, src/ishmael_tables.jl: a `Union{...,Nothing}` field makes
#      `mtile.radiation.q_lw` a type-unstable load in the per-column driver's hot
#      path), so the struct definition has to exist before semiimplicit.jl.
#   2. Everything here is testable with no artifacts, no network and no solver,
#      which is what test/test_radiation.jl exercises.
#
# Sign and unit conventions used throughout the radiation code:
#   * `q_lw`, `q_sw` are HEATING RATES in W/m^3 (a flux divergence), never K/s,
#     so they add straight into `QDOT_TH` and Scythe's g/Cp constants never meet
#     the radiation code's. Cooling is negative.
#   * Layer index 1 is the SURFACE and z ascends with the index, matching both
#     Scythe's column layout and RRTMGP's level-1-is-surface convention.
#   * Faces (RRTMGP "levels") are indexed 1..nlay+1, face k below layer k.

"""
    CloudOpticsColumn(nlay)

Per-column cloud-optics inputs for one radiation call: liquid and ice water path
`lwp`/`iwp` [g/m^2 per layer], effective radii `re_liq`/`re_ice` [micron], and cloud
fraction `cf`. Filled once per column by `cloud_optics_column!` (S3a) from the
prognostic condensate, called from `radiation_assemble!`; zero-initialized so the
clear-sky path can leave it alone.

`cf` is strictly 0 or 1 -- Scythe resolves its clouds, has no subgrid cloud scheme,
and McICA overlap is deterministic for a 0/1 fraction, which is what makes an all-sky
run bitwise reproducible.
"""
struct CloudOpticsColumn
    lwp::Vector{Float64}
    iwp::Vector{Float64}
    re_liq::Vector{Float64}
    re_ice::Vector{Float64}
    cf::Vector{Float64}
end

CloudOpticsColumn(nlay::Int) = CloudOpticsColumn(zeros(Float64, nlay), zeros(Float64, nlay),
                                                 zeros(Float64, nlay), zeros(Float64, nlay),
                                                 zeros(Float64, nlay))

"""
    StratosphericExtension

The layers stacked ABOVE the model top for the radiative transfer only. Scythe's top is
25 km (~25 hPa); a solver integrates exactly the layers it is handed, so without an
extension the outgoing longwave radiation is missing the entire stratospheric column and
the ozone that absorbs most of the solar ultraviolet.

Fields are the same shape as the model-layer arrays: `nlay` layers with `z`, `T`, `p`,
`vmr_h2o`, `vmr_o3` at layer midpoints, `z_face`/`T_face`/`p_face` of length `nlay+1`
(face 1 coincides with the model top face), and `dz` of length `nlay`.

This struct is the CONTAINER only -- the standard-atmosphere fill (profile choice,
log-p spacing, per-column pressure rescaling, the blend to the model top) is S1b.
"""
struct StratosphericExtension
    nlay::Int
    z::Vector{Float64}
    z_face::Vector{Float64}
    dz::Vector{Float64}
    T::Vector{Float64}
    T_face::Vector{Float64}
    p::Vector{Float64}
    p_face::Vector{Float64}
    vmr_h2o::Vector{Float64}
    vmr_o3::Vector{Float64}
end

"""
The no-extension value: zero layers, empty vectors. Used by every configuration with
`options[:radiation_extension] = :none` and by [`EMPTY_RADIATION`](@ref), and shared
`const` across tiles because it is immutable and never indexed.
"""
const EMPTY_EXTENSION = StratosphericExtension(0, Float64[], Float64[], Float64[],
                                               Float64[], Float64[], Float64[], Float64[],
                                               Float64[], Float64[])

"""
    RadiationWork(nlay)

Per-column scratch for one radiation call: the reconstructed thermodynamic column
(`p` [Pa], `Tk` [K], the dry/vapor/liquid/ice partial densities [kg/m^3], the kinetic
energy `ke` [J/kg] and the enthalpy-like combination `M` that `retrieve_temperature`
inverts), plus the face values `p_face`/`T_face` that [`radiation_levels!`](@ref) fills.

One instance per tile, reused by every column of every call: the radiation update runs
serially, OUTSIDE the `Threads.@threads :static` column loop, so a single buffer is
safe and the whole update allocates nothing.
"""
struct RadiationWork
    p::Vector{Float64}
    Tk::Vector{Float64}
    rho_d::Vector{Float64}
    rho_v::Vector{Float64}
    rho_liq::Vector{Float64}
    rho_ice::Vector{Float64}
    ke::Vector{Float64}
    M::Vector{Float64}
    p_face::Vector{Float64}
    T_face::Vector{Float64}
end

RadiationWork(nlay::Int) = RadiationWork(zeros(Float64, nlay), zeros(Float64, nlay),
                                         zeros(Float64, nlay), zeros(Float64, nlay),
                                         zeros(Float64, nlay), zeros(Float64, nlay),
                                         zeros(Float64, nlay), zeros(Float64, nlay),
                                         zeros(Float64, nlay + 1), zeros(Float64, nlay + 1))

"""
    RadiationState

Everything one tile needs to compute and hold a radiative heating field.

The struct is `mutable` but nearly every field is `const`: the arrays are written in
place and only five scalars plus the solver handle actually change between calls
(`solver`, `last_call_step`, `cos_zenith`, `toa_flux`, `sw_scale`). The `const`
annotations are not decoration -- they let the compiler hoist the array loads in the
`QDOT_TH` fold, which is inside the per-column hot path.

`solver::Any` is deliberately untyped: it holds the RRTMGP solver object, and typing it
would drag an RRTMGP name into this leaf file (and into `ModelTile`, and therefore into
every equation set). It is read exactly once per radiation call, from serial code, so
the dynamic dispatch is unmeasurable.

# Held forcing
`q_lw`, `q_sw` are GRIDPOINT-indexed [W/m^3], laid out exactly like `expdot`
(row `j = (c-1)*kDim + k`), so the driver's fold is a straight indexed add with no
reshape. `q_lw_ref`, `q_sw_ref` are the RESTING REFERENCE COLUMN's heating profiles (length
`nlay`) that `options[:radiation_forcing] = :anomaly` subtracts. They are recomputed on
EVERY radiation call, not captured once: on `:rrtmgp` the reference column is solved as
one extra column of the same batch, on `:prescribed` it is the same local product
evaluated on the reference profiles. That makes the reference bubble-free, identical on
every tile and nest patch, and independent of the initial condition -- and bitwise equal
to what a RESTING model column produces, which is what makes the far-field anomaly
exactly zero. (S2a/S2b used the per-tile horizontal mean at t=0 and had none of those
properties.) They stay populated between calls so the trace and the S5 sidecar can report
the reference profile itself.

# Counters
`n_clamp_tk`, `n_clamp_re_liq`, `n_clamp_re_ice`, `n_neg_rho_v` accumulate how often an
input had to be forced into the solver's valid range. They are NOT `const` because they
are incremented per call; they are the reason a clamp is never silent. They are written
only from the serial radiation update, never from a threaded column loop.
"""
mutable struct RadiationState
    # ── resolved configuration (see validate_radiation_options) ──
    const active::Bool
    const scheme::Symbol            # :none | :prescribed | :rrtmgp
    const forcing::Symbol           # :full | :anomaly
    const method::Symbol            # :gray | :clearsky | :allsky | :allsky_clear
    const solar::Symbol             # :none | :fixed | :diurnal
    const level_interp::Symbol      # :hydrostatic | :bestfit
    const extension_kind::Symbol    # :tropical | :midlatitude_summer | :none
    const interval_steps::Int       # radiation cadence, in model steps
    const stride::Int               # model gridpoints per radiation layer
    const nlay::Int                 # radiation layers per column (model layers only)
    const ncol::Int                 # columns on this tile
    const kDim::Int                 # model gridpoints per column
    const z_max::Float64            # taper the heating to zero at this height [m]
    const sw_rescale::Bool
    const rain_in_cloud::Bool
    const output::Bool
    const check_values::Bool
    # ── held radiative forcing [W/m^3] ──
    const q_lw::Vector{Float64}
    const q_sw::Vector{Float64}
    const q_lw_ref::Vector{Float64}
    const q_sw_ref::Vector{Float64}
    # ── fixed column geometry (identical for every column of a tile) ──
    const z::Vector{Float64}
    const z_face::Vector{Float64}
    const dz::Vector{Float64}
    const o3::Vector{Float64}       # ozone VMR on the MODEL layers (Scythe has none)
    const taper::Vector{Float64}    # the z_max cosine taper, precomputed on the layers
    # ── flux diagnostics, (nlev_tot, ncol) with the extension included ──
    const flux_lw_up::Matrix{Float64}
    const flux_lw_dn::Matrix{Float64}
    const flux_lw_net::Matrix{Float64}
    const flux_sw_up::Matrix{Float64}
    const flux_sw_dn::Matrix{Float64}
    const flux_sw_net::Matrix{Float64}
    # ── sub-objects ──
    const extension::StratosphericExtension
    const work::RadiationWork
    const cloud::CloudOpticsColumn
    const gases::Dict{String,Float64}
    # ── the five things that change between calls ──
    solver::Any
    last_call_step::Int
    cos_zenith::Float64
    toa_flux::Float64
    sw_scale::Float64
    # ── clamp/saturation counters (never silent) ──
    n_clamp_tk::Int
    n_clamp_re_liq::Int
    n_clamp_re_ice::Int
    n_neg_rho_v::Int
end

"""
    RadiationState(; kwargs...)

Keyword constructor with the INACTIVE value as its default for every field, so a caller
(`mc_radiation_state`, S2a) names only what a live radiation configuration actually
changes. The positional order above is long enough that a positional call is a bug
waiting to happen.
"""
function RadiationState(;
        active::Bool = false,
        scheme::Symbol = :none,
        forcing::Symbol = :full,
        method::Symbol = :allsky,
        solar::Symbol = :none,
        level_interp::Symbol = :hydrostatic,
        extension_kind::Symbol = :none,
        interval_steps::Int = 1,
        stride::Int = 1,
        nlay::Int = 0,
        ncol::Int = 0,
        kDim::Int = 0,
        z_max::Float64 = Inf,
        sw_rescale::Bool = false,
        rain_in_cloud::Bool = false,
        output::Bool = true,
        check_values::Bool = false,
        q_lw::Vector{Float64} = Float64[],
        q_sw::Vector{Float64} = Float64[],
        q_lw_ref::Vector{Float64} = Float64[],
        q_sw_ref::Vector{Float64} = Float64[],
        z::Vector{Float64} = Float64[],
        z_face::Vector{Float64} = Float64[],
        dz::Vector{Float64} = Float64[],
        o3::Vector{Float64} = Float64[],
        taper::Vector{Float64} = Float64[],
        flux_lw_up::Matrix{Float64} = zeros(Float64, 0, 0),
        flux_lw_dn::Matrix{Float64} = zeros(Float64, 0, 0),
        flux_lw_net::Matrix{Float64} = zeros(Float64, 0, 0),
        flux_sw_up::Matrix{Float64} = zeros(Float64, 0, 0),
        flux_sw_dn::Matrix{Float64} = zeros(Float64, 0, 0),
        flux_sw_net::Matrix{Float64} = zeros(Float64, 0, 0),
        extension::StratosphericExtension = EMPTY_EXTENSION,
        work::RadiationWork = RadiationWork(0),
        cloud::CloudOpticsColumn = CloudOpticsColumn(0),
        gases::Dict{String,Float64} = Dict{String,Float64}(),
        solver = nothing,
        # `typemin(Int)` rather than 0 or -1: the very first step of a run is t = 1, and
        # "no call has happened yet" has to be distinguishable from "called at step 0"
        # so the pre-pass can force the first solve whatever the cadence is.
        last_call_step::Int = typemin(Int),
        cos_zenith::Float64 = 0.0,
        toa_flux::Float64 = 0.0,
        sw_scale::Float64 = 1.0,
        n_clamp_tk::Int = 0,
        n_clamp_re_liq::Int = 0,
        n_clamp_re_ice::Int = 0,
        n_neg_rho_v::Int = 0)
    return RadiationState(active, scheme, forcing, method, solar, level_interp,
                          extension_kind, interval_steps, stride, nlay, ncol, kDim, z_max,
                          sw_rescale, rain_in_cloud, output, check_values,
                          q_lw, q_sw, q_lw_ref, q_sw_ref,
                          z, z_face, dz, o3, taper,
                          flux_lw_up, flux_lw_dn, flux_lw_net,
                          flux_sw_up, flux_sw_dn, flux_sw_net,
                          extension, work, cloud, gases,
                          solver, last_call_step, cos_zenith, toa_flux, sw_scale,
                          n_clamp_tk, n_clamp_re_liq, n_clamp_re_ice, n_neg_rho_v)
end

"""
The RADIATION-OFF value: inactive, every array empty, no solver.

`ModelTile` carries its `RadiationState` concretely for the same reason it carries
`IshmaelTables` concretely (see [`EMPTY_ISHMAEL_TABLES`](@ref)) -- a `Union` field would
make `mtile.radiation.active` a type-unstable load in `mc_driver!`'s preamble. A run
with radiation off never reads past `.active`, so the empty arrays are unreachable
rather than merely harmless.

NOT `const`-shared in the way `EMPTY_ISHMAEL_TABLES` is: it is a `mutable struct`, so
every tile that wants the off value must call `RadiationState()` for a fresh instance if
it ever writes to it. Nothing writes to the off value (the driver gate is `.active`),
so the shared singleton below is safe, and it keeps tile construction allocation-free
on the common path.
"""
const EMPTY_RADIATION = RadiationState()

# ── Column geometry ───────────────────────────────────────────────────────────

"""
    radiation_faces(z, z_bottom, z_top) -> (z_face, dz)

Layer faces for a column whose layer midpoints are the model gridpoints `z` (strictly
increasing, `z_bottom < z[1]`, `z[end] < z_top`). Face 1 is the ground, face `k` for
`2 <= k <= nlay` is the midpoint of `z[k-1]`/`z[k]`, and face `nlay+1` is the model top.

The midpoint rule is chosen because on the cubic-B-spline Gauss mish (three nodes per
cell at 0.1127/0.5/0.8873 dX) the INTER-CELL midpoint lands exactly on the cell
boundary, so the faces tile the column with no gaps and `sum(dz) == z_top - z_bottom`
holds to the last bit. That matters twice over: the same `dz` weights the layer water
paths and divides the flux difference, so the discrete heating telescopes to the column
net flux difference exactly, whatever the vertical basis does.

These are NOT the Gauss cell weights (`gauss_cell_weights`,
benchmarks/common/diagnostics.jl) -- those integrate the spline basis and remain the
right tool for the offline energy-budget cross-check, but they are not a face-based
control-volume partition and cannot divide a flux difference.
"""
function radiation_faces(z::AbstractVector, z_bottom, z_top)
    nlay = length(z)
    nlay >= 2 || error("radiation_faces needs at least 2 layer points, got $nlay")
    @inbounds for k in 1:(nlay - 1)
        z[k] < z[k + 1] || error(
            "radiation_faces: the layer heights must be strictly increasing, but " *
            "z[$k] = $(z[k]) >= z[$(k+1)] = $(z[k+1])")
    end
    z_bottom < z[1] || error(
        "radiation_faces: z_bottom = $z_bottom is not below the first layer z[1] = $(z[1])")
    z[nlay] < z_top || error(
        "radiation_faces: z_top = $z_top is not above the last layer z[end] = $(z[nlay])")

    z_face = Vector{Float64}(undef, nlay + 1)
    z_face[1] = z_bottom
    @inbounds for k in 2:nlay
        z_face[k] = 0.5 * (z[k - 1] + z[k])
    end
    z_face[nlay + 1] = z_top
    return z_face, diff(z_face)
end

"""
    radiation_levels!(work::RadiationWork, z, z_face)

Fill `work.p_face` and `work.T_face` [Pa, K] from the layer-point `work.p`, `work.Tk`,
`work.rho_d`, `work.rho_v`.

INTERIOR faces (`2 <= k <= nlay`) sit between two layer points, so they are interpolated:
`ln p` linear in z and `T` linear in z. Interpolating the LOGARITHM of pressure is the
whole point -- pressure is exponential in height, so a linear-in-p interpolation carries
a second-order error of order `(dz/H)^2/8` (~0.4 Pa at dz = 250 m, H = 8 km, but growing
with the layer thickness), while linear-in-ln-p is EXACT for any isothermal layer and
second-order in the lapse rate otherwise. Temperature is close to piecewise linear in z,
so it is interpolated directly.

The BOTTOM and TOP faces lie outside the layer-point range, where interpolation is not
available. They are extrapolated HYDROSTATICALLY,

    p_face = p_k * exp(-(z_face - z_k) / H_k),   H_k = R_m T_k / g,
    R_m    = Rd + q_v Rv,   q_v = rho_v / rho_d,

rather than by continuing the ln-p line through the two nearest layers. Two reasons:
the hydrostatic form uses the LOCAL state (the top two layers of a sponge-damped
stratosphere are not a reliable slope, and the bottom two straddle the boundary layer's
strongest curvature), and it reproduces an isothermal hydrostatic column to round-off,
which is the analytic check in test/test_radiation.jl. It also guarantees
`p_face > 0` and the monotonicity the solver requires. Temperature at the ends is
linearly extrapolated from the nearest two layer points, matching the interior rule.

`R_m` uses `max(rho_v, 0)`: negative vapor is a resolution diagnostic that the model
itself never alters (see reference/HANDOFF_CONDENSATE_REPRESENTATION.md), but a negative
`q_v` here would give a scale height on the wrong side of dry, which is a different
error from the one being diagnosed. The model state is untouched; only the extrapolant
is floored.
"""
function radiation_levels!(work::RadiationWork, z::AbstractVector, z_face::AbstractVector)
    p = work.p; Tk = work.Tk; rho_d = work.rho_d; rho_v = work.rho_v
    p_face = work.p_face; T_face = work.T_face
    nlay = length(p)
    length(z) == nlay || error(
        "radiation_levels!: length(z) = $(length(z)) does not match the work column ($nlay)")
    length(z_face) == nlay + 1 || error(
        "radiation_levels!: length(z_face) = $(length(z_face)) should be $(nlay + 1)")
    nlay >= 2 || error("radiation_levels! needs at least 2 layers, got $nlay")

    @inbounds for k in 2:nlay
        dzk = z[k] - z[k - 1]
        w = (z_face[k] - z[k - 1]) / dzk
        p_face[k] = exp((1.0 - w) * log(p[k - 1]) + w * log(p[k]))
        T_face[k] = (1.0 - w) * Tk[k - 1] + w * Tk[k]
    end

    # Hydrostatic ends. `_moist_scale_height` folds in the rho_d <= 0 guard so an empty
    # or not-yet-filled column cannot divide by zero.
    @inbounds begin
        H_bot = _moist_scale_height(rho_d[1], rho_v[1], Tk[1])
        p_face[1] = p[1] * exp(-(z_face[1] - z[1]) / H_bot)
        T_face[1] = Tk[1] + (Tk[2] - Tk[1]) * (z_face[1] - z[1]) / (z[2] - z[1])

        H_top = _moist_scale_height(rho_d[nlay], rho_v[nlay], Tk[nlay])
        p_face[nlay + 1] = p[nlay] * exp(-(z_face[nlay + 1] - z[nlay]) / H_top)
        T_face[nlay + 1] = Tk[nlay] +
            (Tk[nlay] - Tk[nlay - 1]) * (z_face[nlay + 1] - z[nlay]) /
            (z[nlay] - z[nlay - 1])
    end
    return nothing
end

"Local moist scale height H = (Rd + q_v Rv) T / g, with the dry value as the fallback."
@inline function _moist_scale_height(rho_d, rho_v, Tk)
    q_v = rho_d > 0.0 ? max(rho_v, 0.0) / rho_d : 0.0
    return (Rd + q_v * Rv) * Tk / gravity
end

# ── Solar geometry ────────────────────────────────────────────────────────────

"""
    solar_geometry(lat_deg, lon_deg, doy, solar_constant) -> (cos_zenith, toa_flux)

Cosine of the solar zenith angle and the top-of-atmosphere insolation for a fractional
day-of-year `doy` in UTC (`doy = 80.5` is 12:00 UTC on the 80th day).

Declination uses the first-order cosine approximation

    delta = -23.44 deg * cos(2 pi (doy + 10) / 365.25)

(Cooper 1969, in the form given by Hartmann, *Global Physical Climatology*, §2.5),
which is accurate to about 0.5 deg -- an error of the same order as ignoring the
equation of time, and far below the uncertainty in anything else in the column. The
hour angle is local apparent solar time from the UTC fraction of the day plus
`lon_deg/15` hours, with NO equation-of-time correction (up to +/-16 minutes of phase;
irrelevant for an idealized run, and documented here so it is not rediscovered as a bug).

    cos_z = max(0, sin(phi) sin(delta) + cos(phi) cos(delta) cos(h))

The eccentricity factor `E0 = 1 + 0.033 cos(2 pi doy / 365.25)` (Duffie & Beckman 1980,
first term of the Spencer series) scales the solar constant.

`toa_flux` is the flux NORMAL TO THE BEAM, i.e. `solar_constant * E0` BEFORE any
`cos_zenith` factor -- the horizontal TOA downward flux is `toa_flux * cos_zenith`.
Which of the two a solver wants is a library convention (S0 confirms RRTMGP's); this
function returns the beam-normal value and the caller multiplies if needed.
"""
function solar_geometry(lat_deg, lon_deg, doy::Float64, solar_constant)
    decl = deg2rad(-23.44) * cos(2π * (doy + 10.0) / 365.25)
    # Fractional part of the day in UTC hours, shifted to local solar time.
    utc_hour = 24.0 * (doy - floor(doy))
    local_hour = utc_hour + lon_deg / 15.0
    hour_angle = π * (local_hour / 12.0 - 1.0)      # 0 at local noon
    phi = deg2rad(lat_deg)
    cos_zenith = max(0.0, sin(phi) * sin(decl) + cos(phi) * cos(decl) * cos(hour_angle))
    E0 = 1.0 + 0.033 * cos(2π * doy / 365.25)
    return cos_zenith, solar_constant * E0
end

"""
    solar_state(options, physical_params, t_model) -> (cos_zenith, toa_flux)

Resolve `options[:solar]` into the two numbers the shortwave boundary condition needs, at
model time `t_model` [s] (`t_model = (t-1)*ts`, zero-based and identical across nest
patches).

- `:none` -- longwave only; both zero, so the shortwave solve can be skipped entirely.
- `:fixed` -- `physical_params[:cos_zenith]` (default 0.2588) and `[:sw_toa_flux]`
  (default 551.58). That pair gives a 142.7 W/m^2 daily-mean insolation, the standard
  idealized-RCE forcing, without a diurnal cycle.
- `:diurnal` -- [`solar_geometry`](@ref) at
  `doy = start_doy + (start_hour*3600 + t_model)/86400`, with
  `physical_params[:latitude]` REQUIRED (an implicit equator would be a silently wrong
  answer, not a default), `[:longitude]` 0, `[:start_doy]` 172 (northern solstice),
  `[:start_hour]` 0 and `[:solar_constant]` 1360.8.
"""
function solar_state(options, physical_params, t_model)
    solar = get(options, :solar, :none)
    if solar === :none
        return (0.0, 0.0)
    elseif solar === :fixed
        return (get(physical_params, :cos_zenith, 0.2588),
                get(physical_params, :sw_toa_flux, 551.58))
    elseif solar === :diurnal
        haskey(physical_params, :latitude) || error(
            "options[:solar] = :diurnal requires physical_params[:latitude] (degrees " *
            "north); there is no sensible default latitude for a diurnal cycle")
        lat = physical_params[:latitude]
        lon = get(physical_params, :longitude, 0.0)
        doy = get(physical_params, :start_doy, 172.0) +
              (get(physical_params, :start_hour, 0.0) * 3600.0 + t_model) / 86400.0
        return solar_geometry(lat, lon, doy, get(physical_params, :solar_constant, 1360.8))
    end
    error("options[:solar] = :$(solar) is not recognized; use :none, :fixed or :diurnal")
end

# ── Sponge taper ──────────────────────────────────────────────────────────────

"""
    radiation_taper(z, z_max, width = 2000.0) -> Vector{Float64}

Weights that switch the radiative heating off above `z_max`: 1 below `z_max - width`, a
raised cosine from 1 to 0 across `[z_max - width, z_max]`, and 0 above.

The sponge layer relaxes toward the reference state, so a radiative tendency there is
fighting the damping rather than doing physics, and the top of an extended column is
where the layer thicknesses and the extension blend are least trustworthy. Tapering the
FORCING is the first-choice fix (the alternative, adding the radiative equilibrium to the
sponge's relaxation target, is deferred until a `:full` run shows it is needed).

`z_max = Inf` (the default configuration) returns all ones -- an identity that costs one
multiply per gridpoint and keeps the store path branch-free.
"""
function radiation_taper(z::AbstractVector, z_max, width = 2000.0)
    w = ones(Float64, length(z))
    isinf(z_max) && return w
    width > 0.0 || error("radiation_taper: the taper width must be positive, got $width")
    z_start = z_max - width
    @inbounds for k in eachindex(z)
        zk = z[k]
        if zk >= z_max
            w[k] = 0.0
        elseif zk > z_start
            w[k] = 0.5 * (1.0 + cos(π * (zk - z_start) / width))
        end
    end
    return w
end

# ── Option validation ─────────────────────────────────────────────────────────

# Every options key this module reads. Anything else starting with "radiation" is a typo
# and is refused: a silently ignored `:radiation_intervals` would look like a working
# run with the default cadence, which is exactly the kind of failure that costs a
# campaign (cf. the Louis-BL and Khdiff_water blockers).
const RADIATION_OPTION_KEYS = Set{Symbol}((
    :radiation, :radiation_forcing, :radiation_interval, :radiation_method,
    :radiation_sw_rescale, :radiation_level_interp, :radiation_extension,
    :radiation_extension_layers, :radiation_layer_stride, :radiation_z_max,
    :radiation_rain_in_cloud, :radiation_gases, :radiation_output,
    :radiation_check_values, :radiation_trace, :radiation_trace_sw))

const RADIATION_SCHEMES = (:none, :prescribed, :rrtmgp)
const RADIATION_FORCINGS = (:full, :anomaly)
const RADIATION_METHODS = (:gray, :clearsky, :allsky, :allsky_clear)
const RADIATION_EXTENSIONS = (:tropical, :midlatitude_summer, :none)
const RADIATION_LEVEL_INTERPS = (:hydrostatic, :bestfit)
const RADIATION_SOLAR_MODES = (:none, :fixed, :diurnal)

_rad_check(value, allowed, key) = value in allowed || error(
    "options[:$key] = :$(value) is not recognized; use " *
    join(string.(":", allowed), ", ", " or "))

"""
    validate_radiation_options(options, physical_params, equation_set, ts, kDim)

Check the radiation configuration LOUDLY and return the resolved settings as a
NamedTuple `(scheme, forcing, interval_steps, method, solar, sw_rescale, level_interp,
extension, n_ext, stride, z_max, rain_in_cloud, output, check_values)`.

Called once per tile from `mc_radiation_state` (S2a), before any array is allocated, so
a misconfigured run dies at setup rather than after the first radiation call — or, worse,
runs to completion with the wrong forcing.

`options[:radiation] = :none` (or absent) short-circuits after the value check: a run
with radiation off must never be able to fail on a radiation rule, so an old
configuration that happens to carry a stale `:sfc_albedo` still starts.

Errors:
- an unrecognized value for any radiation symbol, or an unknown `:radiation_*` key;
- radiation on an equation set that is not a pressure-reference (`moist_compressible`)
  set — nothing else has the `QDOT_TH` insertion point;
- `:solar = :diurnal` without `physical_params[:latitude]`;
- `:SST <= 200` (the Celsius footgun, same guard as the surface fluxes);
- `:sfc_albedo` outside [0, 1] or `:sfc_emissivity` outside (0, 1];
- a non-positive radiation interval or `z_max`, or a stride that does not divide `kDim`.

Warnings: a cadence below ten timesteps (the held-forcing pattern buys nothing there,
and the cost is per call); `:gray` with ice microphysics on (the gray solver has no
cloud optics at all, so the ice the run is spending its time on is radiatively invisible).
"""
function validate_radiation_options(options, physical_params, equation_set, ts, kDim)
    scheme = get(options, :radiation, :none)
    _rad_check(scheme, RADIATION_SCHEMES, "radiation")

    # Resolve the whole table first so the off path returns the same NamedTuple shape.
    forcing = get(options, :radiation_forcing, :full)
    method = get(options, :radiation_method, :allsky)
    solar = get(options, :solar, :none)
    level_interp = get(options, :radiation_level_interp, :hydrostatic)
    extension = get(options, :radiation_extension, :tropical)
    n_ext = Int(get(options, :radiation_extension_layers, 15))
    stride = Int(get(options, :radiation_layer_stride, 1))
    z_max = Float64(get(options, :radiation_z_max, Inf))
    interval_sec = Float64(get(options, :radiation_interval, 300.0))
    rain_in_cloud = get(options, :radiation_rain_in_cloud, false)::Bool
    output = get(options, :radiation_output, true)::Bool
    check_values = get(options, :radiation_check_values, false)::Bool
    # The per-step zenith rescale exists to make a DIURNAL shortwave forcing continuous
    # between calls; with a fixed sun it is identically 1, so it defaults on only there.
    sw_rescale = get(options, :radiation_sw_rescale, solar === :diurnal)::Bool

    resolved = (; scheme, forcing, interval_steps = 1, method, solar, sw_rescale,
                level_interp, extension, n_ext, stride, z_max, rain_in_cloud, output,
                check_values)
    scheme === :none && return resolved

    for key in keys(options)
        (startswith(String(key), "radiation") && !(key in RADIATION_OPTION_KEYS)) && error(
            "options[:$key] is not a recognized radiation option. The radiation keys are " *
            join(string.(":", sort!(collect(RADIATION_OPTION_KEYS); by = String)), ", "))
    end

    _rad_check(forcing, RADIATION_FORCINGS, "radiation_forcing")
    _rad_check(method, RADIATION_METHODS, "radiation_method")
    _rad_check(extension, RADIATION_EXTENSIONS, "radiation_extension")
    _rad_check(level_interp, RADIATION_LEVEL_INTERPS, "radiation_level_interp")
    _rad_check(solar, RADIATION_SOLAR_MODES, "solar")

    uses_pressure_reference(equation_set) || error(
        "options[:radiation] = :$(scheme) needs a pressure-reference (moist_compressible) " *
        "equation set — the radiative heating enters through QDOT_TH, which only that " *
        "family has (got equation_set = \"$(equation_set)\")")

    if solar === :diurnal && !haskey(physical_params, :latitude)
        error("options[:solar] = :diurnal requires physical_params[:latitude] (degrees " *
              "north); there is no sensible default latitude for a diurnal cycle")
    end

    if haskey(physical_params, :SST) && physical_params[:SST] <= 200.0
        error("physical_params[:SST] must be in Kelvin (got $(physical_params[:SST]) — " *
              "28 C is 301.15); radiation uses it as the surface temperature")
    end

    albedo = get(physical_params, :sfc_albedo, 0.06)
    (0.0 <= albedo <= 1.0) || error(
        "physical_params[:sfc_albedo] = $albedo is outside [0, 1] (ocean is ~0.06)")
    emis = get(physical_params, :sfc_emissivity, 0.98)
    (0.0 < emis <= 1.0) || error(
        "physical_params[:sfc_emissivity] = $emis is outside (0, 1] (ocean is ~0.98)")

    interval_sec > 0.0 || error(
        "options[:radiation_interval] must be positive seconds, got $interval_sec")
    ts > 0.0 || error("validate_radiation_options: the model timestep must be positive, " *
                      "got ts = $ts")
    # Seconds, like :output_interval and :cfl_interval, so the cadence is nest-invariant;
    # converted to steps exactly once, here.
    interval_steps = max(1, round(Int, interval_sec / ts))

    n_ext >= 0 || error(
        "options[:radiation_extension_layers] must be non-negative, got $n_ext")
    stride >= 1 || error("options[:radiation_layer_stride] must be >= 1, got $stride")
    (kDim > 0 && mod(kDim, stride) != 0) && error(
        "options[:radiation_layer_stride] = $stride does not divide kDim = $kDim; the " *
        "radiation layers must tile the column exactly or the heating is not conservative")
    (isinf(z_max) || z_max > 0.0) || error(
        "options[:radiation_z_max] must be positive (or Inf for no taper), got $z_max")

    if interval_steps < 10
        @warn "options[:radiation_interval] = $interval_sec s is only $interval_steps " *
              "timestep(s) at ts = $ts s. The held-forcing pattern exists because a " *
              "radiative-transfer call costs ~10 ms per column; consider a longer cadence."
    end
    if method === :gray && get(options, :ice_microphysics, :none) !== :none
        @warn "options[:radiation_method] = :gray with ice microphysics on: the gray " *
              "solver has no cloud optics, so the ice the run is computing is " *
              "radiatively invisible. Use :allsky to see it."
    end

    # `:none` means no extension layers at all, whatever the layer count says.
    n_ext_resolved = extension === :none ? 0 : n_ext
    return (; scheme, forcing, interval_steps, method, solar, sw_rescale, level_interp,
            extension, n_ext = n_ext_resolved, stride, z_max, rain_in_cloud, output,
            check_values)
end
