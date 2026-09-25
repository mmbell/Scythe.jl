module ScytheRRTMGPExt

# ── ScytheRRTMGPExt: the RRTMGP half of the radiation driver ─────────────────
#
# A package extension of Scythe, loaded automatically when RRTMGP and ClimaComms are
# both loaded alongside Scythe (`using RRTMGP, ClimaComms`). This is the ONLY code in
# Scythe that names RRTMGP or ClimaComms (and, of the files that name NCDatasets, the
# one that names it only to trigger RRTMGP's own NCDatasets extension — see the
# import-style note below — never to read or write a file itself; src/radiation_io.jl,
# S5, is the file that actually does that).
# Everything above it (src/radiation_state.jl, src/radiation.jl,
# src/radiation_profiles.jl) works on plain arrays and knows nothing about the
# radiative-transfer library, which is what keeps `ModelTile` free of an RRTMGP type
# parameter and is what made the move of these dependencies into this extension a
# mechanical edit of one file.
#
# Every function here is a METHOD of a stub declared in src/radiation.jl
# (`function rrtmgp_solver end`, ...), so the driver calls them by their Scythe names
# and never has to know whether the extension is loaded — `mc_radiation_state` asks
# (`require_rrtmgp_extension`) before the first call and refuses with a message naming
# the missing `using`.
#
# What lives here:
#   * the per-process context and lookup-table caches (artifacts are read once
#     per worker, not once per tile; the cache containers themselves are declared in
#     src/radiation_profiles.jl),
#   * the standard-atmosphere sampling (`_standard_profile`) that
#     `standard_extension` and `model_ozone` (src/radiation_profiles.jl) interpolate
#     to build the stratospheric extension and the ozone profile Scythe does not carry,
#   * solver construction (preallocated, host-written, allocation-free solves),
#   * `rrtmgp_solve_columns!`, the one call that writes a whole tile's columns
#     into the solver and runs it,
#   * `rrtmgp_fluxes`, the aliasing flux views,
#   * `radiation_offline_column`, the S1 offline anchor that S2b's first
#     `:full` run has to reproduce to round-off.
#
# Conventions inherited from src/radiation_state.jl and repeated because they
# are easy to get backwards:
#   * layer index 1 is the SURFACE and z ascends with the index. RRTMGP uses
#     the same convention (verified in S0), so nothing is ever flipped.
#   * `flux_net = up - down`, so a longwave-cooling layer has
#     `flux_net[k+1] > flux_net[k]` and `q = (F[k] - F[k+1])/dz < 0`.
#   * heating is W/m^3, never K/s.
#
# NOTE ON THE IMPORT STYLE. The plan calls for `using NCDatasets`; this file
# uses `import NCDatasets` instead. The only reason NCDatasets is needed at all
# is that RRTMGP's spectral `lookup_tables` methods live in an RRTMGP package
# EXTENSION that is triggered by NCDatasets being LOADED -- `import` triggers it
# exactly as `using` does, and does not splat NCDatasets' exports (`Dataset`,
# `defVar`, `dimnames`, ...) into this module (formerly into `Scythe` itself), where
# they would collide with Springsteel's and Scythe's own I/O names. Functionally identical, strictly
# narrower blast radius.

using Scythe
# The Scythe names the bodies below use unqualified. Importing them keeps every body
# byte-identical to its pre-extension form in src/radiation_rrtmgp.jl; the stubs
# among them are extended by the qualified `function Scythe.<name>` definitions.
import Scythe: rrtmgp_context, rrtmgp_method, rrtmgp_lookups, _standard_profile,
               rrtmgp_solver, rrtmgp_solve_columns!, rrtmgp_fluxes,
               radiation_offline_column,
               StratosphericExtension, default_gases, standard_extension, model_ozone,
               radiation_divergence!, radiation_faces, radiation_levels!, RadiationWork,
               _interp_linear, _seed_boundary_row!, _check_o3_shape,
               _write_thermo!, _write_gases!, _write_clouds!,
               _RRTMGP_CACHE_LOCK, _RRTMGP_CONTEXT, _RRTMGP_LOOKUPS, _RRTMGP_PROFILES,
               RRTMGP_EXTENSION_TOP, RRTMGP_PROFILE_SAMPLES,
               RRTMGP_BOUNDARY_P, RRTMGP_BOUNDARY_T,
               Rd, Rv, Cpd, Cpv, Cl, Ci

import ClimaComms
ClimaComms.@import_required_backends
import RRTMGP
import NCDatasets    # loads RRTMGPNCDatasetsExt; see the note above

# ── Context, method and lookup tables ─────────────────────────────────────────

"""
    rrtmgp_context()

The ClimaComms context every RRTMGP object on this process shares, built once
and cached.

The device is pinned to `ClimaComms.CPUMultiThreaded()` EXPLICITLY rather than
taken from `ClimaComms.device()`, which reads the `CLIMACOMMS_DEVICE`
environment variable and would silently select CUDA on a GPU-equipped compute
node. Scythe's radiation is a host-side, thread-over-columns calculation; a
solver that quietly moved to a GPU would fail on the first host `view` write.

`CPUMultiThreaded` threads RRTMGP's own column loops over `Threads.nthreads()`.
That is safe precisely because the radiation update runs OUTSIDE Scythe's
`Threads.@threads :static` column loop (D3), never nested inside it.
"""
function Scythe.rrtmgp_context()
    ctx = _RRTMGP_CONTEXT[]
    ctx === nothing || return ctx
    return lock(_RRTMGP_CACHE_LOCK) do
        cached = _RRTMGP_CONTEXT[]
        cached === nothing || return cached
        fresh = ClimaComms.context(ClimaComms.CPUMultiThreaded())
        _RRTMGP_CONTEXT[] = fresh
        return fresh
    end
end

"""
    rrtmgp_method(method::Symbol)

Map a Scythe `options[:radiation_method]` symbol onto the RRTMGP method object.

- `:gray` -- `GrayRadiation()`. Needs no artifacts and no lookup tables at all
  (the optical depth is an analytic function of `p/p_sfc`), which makes it the
  only method that can be exercised on a machine with no network. It ignores
  the model's water vapor entirely, so it is a plumbing test, not physics.
- `:clearsky` -- `ClearSkyRadiation(false)` (the `false` is `aerosol_radiation`).
- `:allsky` -- `AllSkyRadiation(false, false)` (`aerosol_radiation`,
  `reset_rng_seed`). `reset_rng_seed = false` is deliberate: reseeding per call
  would make the McICA sampling depend on call count. Scythe's cloud fraction is
  strictly 0/1, for which McICA is deterministic anyway (D5), so the RNG never
  enters the answer.
- `:allsky_clear` -- `AllSkyRadiationWithClearSkyDiagnostics(false, false)`,
  which runs BOTH a clear-sky and an all-sky solve (roughly twice the cost) and
  retains the clear-sky fluxes for the cloud-radiative-effect diagnostic.
"""
function Scythe.rrtmgp_method(method::Symbol)
    method === :gray && return RRTMGP.GrayRadiation()
    method === :clearsky && return RRTMGP.ClearSkyRadiation(false)
    method === :allsky && return RRTMGP.AllSkyRadiation(false, false)
    method === :allsky_clear &&
        return RRTMGP.AllSkyRadiationWithClearSkyDiagnostics(false, false)
    error("rrtmgp_method: unknown radiation method :$(method); use one of " *
          ":gray, :clearsky, :allsky, :allsky_clear")
end

"""
    rrtmgp_lookups(method::Symbol)

The `LookupBundle` for `method`, built once per process per method and cached.

Keying on the method ALONE (not on `nlay`/`ncol`) is correct, not a shortcut:
RRTMGP's `lookup_tables(grid_params, method)` forwards to
`lookup_tables(method, ClimaComms.device(grid_params), eltype(grid_params))`
(ext/RRTMGPNCDatasetsExt.jl), so the grid parameters contribute only the device
and the float type, both of which are fixed here (CPU, `Float64`). The tables
are spectral data, not grid data. A one-layer, one-column `RRTMGPGridParams` is
therefore an adequate carrier, and the resulting bundle is shared by every
solver on the process -- which is the point, since the tables are ~50 MB of
NetCDF and every tile would otherwise re-read them.

The first call for a spectral method downloads the RRTMGP artifacts if they are
not already in the depot. On a compute node with no network that throws; callers
that must degrade gracefully (the test file, the preflight check) catch it.
"""
function Scythe.rrtmgp_lookups(method::Symbol)
    rm = rrtmgp_method(method)
    return lock(_RRTMGP_CACHE_LOCK) do
        get!(_RRTMGP_LOOKUPS, method) do
            gp = RRTMGP.RRTMGPGridParams(Float64; context = rrtmgp_context(),
                                         domain_nlay = 1, ncol = 1,
                                         isothermal_boundary_layer = false)
            RRTMGP.lookup_tables(gp, rm)
        end
    end
end

# ── Standard-atmosphere sampling ──────────────────────────────────────────────

"""
    _standard_profile(kind) -> NamedTuple

The RRTMGP idealized standard atmosphere `kind`, sampled on a uniform 50 m
altitude grid from 0 to `RRTMGP_EXTENSION_TOP`, cached per process.

The underlying profile (`RRTMGP.standard_atmosphere`) is analytic -- a
two-segment temperature profile, its exact hydrostatic pressure, an exponential
water vapor with a 4 ppmv floor and a log-pressure Gaussian ozone layer peaking
near 30 km -- so sampling it densely and interpolating is equivalent to
evaluating it, and it uses only the public API (the analytic functions
themselves are private to RRTMGP).

Fields: `z_lev`, `p_lev`, `lnp_lev`, `t_lev` on the 1401 sample faces;
`z_lay`, `vmr_h2o`, `vmr_o3` on the 1400 sample layers; `z_rev`, `lnp_rev` the
reversed (ascending-in-`ln p`) pair for inverting `p -> z`; and the profile's
own `t_sfc`, `lat` and `well_mixed` dictionary.
"""
function Scythe._standard_profile(kind::Symbol)
    return lock(_RRTMGP_CACHE_LOCK) do
        get!(_RRTMGP_PROFILES, kind) do
            prof = RRTMGP.standard_atmosphere(Float64; kind = kind,
                                              nlay = RRTMGP_PROFILE_SAMPLES,
                                              ncol = 1,
                                              z_top = RRTMGP_EXTENSION_TOP)
            z_lev = prof.z_lev[:, 1]
            p_lev = prof.p_lev[:, 1]
            lnp_lev = log.(p_lev)
            (; z_lev = z_lev,
               p_lev = p_lev,
               lnp_lev = lnp_lev,
               t_lev = prof.t_lev[:, 1],
               z_lay = 0.5 .* (z_lev[1:(end - 1)] .+ z_lev[2:end]),
               vmr_h2o = prof.vmr_h2o[:, 1],
               vmr_o3 = prof.vmr_o3[:, 1],
               z_rev = reverse(z_lev),
               lnp_rev = reverse(lnp_lev),
               t_sfc = prof.t_sfc[1],
               lat = prof.lat[1],
               well_mixed = prof.well_mixed_vmr)
        end
    end
end

# ── Solver construction ───────────────────────────────────────────────────────

"""
    rrtmgp_solver(nlay_model, ext, ncol, method, solar, check_values;
                  sfc_emissivity = 0.98, sfc_albedo = 0.06,
                  gases = default_gases(), z_face = nothing, latitude = 0.0)
        -> RRTMGP.RRTMGPSolver

Build and preallocate the solver for one tile: `nlay_model` model layers plus
`ext.nlay` extension layers, `ncol` columns.

The solver owns every input and output buffer for the life of the run. The host
writes the inputs through the named getter views and calls
[`rrtmgp_solve_columns!`](@ref); nothing is allocated per call.

# NaN placeholders
Every array the host is REQUIRED to write each call -- layer and level pressure
and temperature, the water-vapor and ozone mixing ratios, the surface
temperature -- is initialized to `NaN`, the trick ClimaAtmos uses for the same
purpose. A field the driver forgets to write then produces a loud, immediate
failure (a `NaN` flux, or an explicit `validate_inputs` error when
`check_values` is on) instead of a plausible answer computed from a stale or
zeroed profile. The two exceptions are deliberate:
  * the isothermal boundary row, seeded (see `_seed_boundary_row!`), and
  * the cloud fields, zeroed, so that a clear-sky configuration and the
    `cloud_optics_clear!` stub of S1 both give an honest cloud-free answer
    without the driver having to write five all-zero arrays every call.

# Physical parameters
`RRTMGP.default_parameters(Float64)`, unmodified. Its `grav = 9.81` is EXACTLY
Springsteel's `gravity`, which is the constant that matters: `col_dry`, the
absorber amount every optical depth scales with, is `Δp/(g m_air)`. Its
`molmass_dryair = 0.02897` implies `R_d = 8.314462618/0.02897 = 287.0025`
against Springsteel's `Rd = 287.04` -- 1.3e-4 relative. That difference enters
in exactly two places: `col_dry` (so every absorber amount is scaled by
1.00013, worth of order 0.03 W/m^2 in OLR, two orders below the spread between
radiation schemes) and the relative humidity that drives the humidity-dependent
AEROSOL optics, which are not enabled. Matching it exactly would mean
constructing `RRTMGP.Parameters.RRTMGPParameters` directly, which is not part of
RRTMGP's public API (`PUBLIC_NAMES`) and could change in a patch release. The
public constructor and the documented 1.3e-4 is the better trade; this is the
place to change it if a future budget closure ever needs the last digit.

# Latitude
The atmospheric state is built with `lat = nothing` ON PURPOSE. When `lat` is an
array RRTMGP applies the Helmert latitude correction to gravity inside
`col_dry` (`g0 = grav - 0.02586 cos(2 phi)`, a +/-0.26% swing). Scythe's
`gravity` is a single constant everywhere, and a radiation calculation using a
different g from the dynamics that produced the pressure field is an
inconsistency, not a refinement. `:gray` is the exception: its optical thickness
is explicitly latitude-dependent, so it gets the `latitude` keyword.

# Arguments
- `method` -- see [`rrtmgp_method`](@ref).
- `solar` -- `:none`, `:fixed` or `:diurnal`. Only `:none` changes anything
  here: it is recorded by initializing the shortwave boundary condition to zero.
  The actual `cos_zenith`/`toa_flux` are written every call.
- `check_values` -- sets the GLOBAL `RRTMGP.check_values[]`. It is a process-wide
  `Ref` inside RRTMGP, not a per-solver flag, so the last solver built on a
  worker wins. Every tile on a worker is configured from the same options, so
  this is only a hazard for a test that builds two solvers with different
  settings (and the tests here set it explicitly when they care).
- `z_face` -- the model's `nlay_model + 1` face heights [m]. Required by
  `:gray`, which needs level altitudes; ignored by the spectral methods, which
  work in pressure.
"""
function Scythe.rrtmgp_solver(nlay_model::Int, ext::StratosphericExtension, ncol::Int,
                       method::Symbol, solar::Symbol, check_values::Bool;
                       sfc_emissivity::Float64 = 0.98,
                       sfc_albedo::Float64 = 0.06,
                       gases::Dict{String,Float64} = default_gases(),
                       z_face::Union{Nothing,AbstractVector} = nothing,
                       latitude::Float64 = 0.0)
    nlay_model >= 2 || error("rrtmgp_solver: nlay_model = $nlay_model must be >= 2")
    ncol >= 1 || error("rrtmgp_solver: ncol = $ncol must be >= 1")
    0.0 <= sfc_emissivity <= 1.0 || error(
        "rrtmgp_solver: sfc_emissivity = $sfc_emissivity is outside [0, 1]")
    0.0 <= sfc_albedo <= 1.0 || error(
        "rrtmgp_solver: sfc_albedo = $sfc_albedo is outside [0, 1]")

    domain_nlay = nlay_model + ext.nlay
    rm = rrtmgp_method(method)
    ctx = rrtmgp_context()
    grid_params = RRTMGP.RRTMGPGridParams(Float64; context = ctx,
                                          domain_nlay = domain_nlay, ncol = ncol,
                                          isothermal_boundary_layer = true)
    lookups = rrtmgp_lookups(method)
    nlay = grid_params.nlay          # == domain_nlay + 1 (the boundary pad)
    nlev = nlay + 1
    params = RRTMGP.default_parameters(Float64)

    bcs_lw = RRTMGP.BCs.LwBCs(fill(sfc_emissivity, lookups.nbnd_lw, ncol), nothing)
    # SwBCs field order verified in S0:
    # (cos_zenith, toa_flux, sfc_alb_direct, inc_flux_diffuse, sfc_alb_diffuse).
    # Direct and diffuse albedo are equal: Scythe has no surface BRDF model, and
    # the ocean's direct-beam albedo depends on the zenith angle in a way that a
    # single number cannot express honestly either way.
    bcs_sw = RRTMGP.BCs.SwBCs(zeros(Float64, ncol),
                              zeros(Float64, ncol),
                              fill(sfc_albedo, lookups.nbnd_sw, ncol),
                              nothing,
                              fill(sfc_albedo, lookups.nbnd_sw, ncol))

    p_lev = fill(NaN, nlev, ncol)
    t_lev = fill(NaN, nlev, ncol)
    t_sfc = fill(NaN, ncol)

    if rm isa RRTMGP.GrayRadiation
        z_face === nothing && error(
            "rrtmgp_solver: method = :gray needs the model face heights; pass " *
            "z_face = <nlay_model+1 vector> (gray optical depth is defined on a " *
            "level-altitude grid)")
        length(z_face) == nlay_model + 1 || error(
            "rrtmgp_solver: length(z_face) = $(length(z_face)) should be " *
            "$(nlay_model + 1) for nlay_model = $nlay_model")
        z_lev = Array{Float64}(undef, nlev, ncol)
        @views for c in 1:ncol
            z_lev[1:(nlay_model + 1), c] .= z_face
            for j in 1:(ext.nlay)
                z_lev[nlay_model + 1 + j, c] = ext.z_face[j + 1]
            end
            # the boundary pad has no altitude of its own; reuse the top face
            z_lev[nlev, c] = z_lev[nlev - 1, c]
        end
        p_lay = fill(NaN, nlay, ncol)
        t_lay = fill(NaN, nlay, ncol)
        _seed_boundary_row!(nothing, p_lev, t_lev, nothing, nothing, nlay)
        p_lay[nlay, :] .= RRTMGP_BOUNDARY_P
        t_lay[nlay, :] .= RRTMGP_BOUNDARY_T
        as = RRTMGP.AtmosphericStates.GrayAtmosphericState(
            fill(latitude, ncol), p_lay, p_lev, t_lay, t_lev, z_lev, t_sfc,
            RRTMGP.AtmosphericStates.GrayOpticalThicknessOGorman2008(Float64))
        solver = RRTMGP.RRTMGPSolver(grid_params, rm, params, bcs_lw, bcs_sw, as;
                                     lookups = lookups)
        RRTMGP.check_values[] = check_values
        return solver
    end

    # ── spectral state ──
    # layerdata rows: 1 col_dry (RRTMGP computes it), 2 p_lay, 3 t_lay,
    # 4 relative humidity (aerosol optics only; zeroed so no NaN can reach the
    # transposed state cache).
    layerdata = fill(NaN, 4, nlay, ncol)
    @views layerdata[1, :, :] .= 0.0
    @views layerdata[4, :, :] .= 0.0

    ngas = max(lookups.ngas_lw, lookups.ngas_sw)
    vmr_wm = zeros(Float64, ngas)
    vmr_h2o = fill(NaN, nlay, ncol)
    vmr_o3 = fill(NaN, nlay, ncol)
    vmr = RRTMGP.VolumeMixingRatios.VmrGM(vmr_h2o, vmr_o3, vmr_wm)

    _seed_boundary_row!(layerdata, p_lev, t_lev, vmr_h2o, vmr_o3, nlay)

    cloud_state = if rm isa RRTMGP.AllSkyRadiation ||
                     rm isa RRTMGP.AllSkyRadiationWithClearSkyDiagnostics
        RRTMGP.AtmosphericStates.CloudState(
            zeros(Float64, nlay, ncol),   # cld_r_eff_liq [micron]
            zeros(Float64, nlay, ncol),   # cld_r_eff_ice [micron]
            zeros(Float64, nlay, ncol),   # cld_path_liq  [g/m^2]
            zeros(Float64, nlay, ncol),   # cld_path_ice  [g/m^2]
            zeros(Float64, nlay, ncol),   # cld_frac
            zeros(Bool, nlay, ncol),      # mask_lw
            zeros(Bool, nlay, ncol),      # mask_sw
            RRTMGP.AtmosphericStates.MaxRandomOverlap(),
            2,                            # ice_rgh = medium (Yang et al. 2013)
        )
    else
        nothing
    end

    as = RRTMGP.AtmosphericStates.AtmosphericState(
        nothing,        # lon (unused)
        nothing,        # lat: constant gravity in col_dry -- see the docstring
        layerdata, p_lev, t_lev, t_sfc, vmr, cloud_state, nothing)

    solver = RRTMGP.RRTMGPSolver(grid_params, rm, params, bcs_lw, bcs_sw, as;
                                 lookups = lookups)

    # The well-mixed gases are scalars in `VmrGM` and never change during a run,
    # so they are set once here rather than every call. `set_volume_mixing_ratio!`
    # indexes `idx_gases_sw`; the packed `vmr` vector is shared with the longwave
    # kernels, so assert the two index maps agree rather than discover a silently
    # mis-assigned gas from a 5 W/m^2 OLR offset.
    if lookups.idx_gases_lw != lookups.idx_gases_sw
        error("rrtmgp_solver: the longwave and shortwave gas index maps differ " *
              "in this RRTMGP build; the well-mixed VMR vector is shared between " *
              "them, so setting a gas through the shortwave map would mis-assign " *
              "it in the longwave. Report this -- the code assumes they match.")
    end
    skipped = String[]
    for name in sort(collect(keys(gases)))
        if haskey(lookups.idx_gases_sw, name)
            RRTMGP.set_volume_mixing_ratio!(solver, name, gases[name])
        else
            push!(skipped, name)
        end
    end
    isempty(skipped) || @info(
        "rrtmgp_solver: the $(method) lookup tables carry no absorption data " *
        "for $(join(skipped, ", ")); those gases are omitted from this solve " *
        "(available: $(join(sort(collect(keys(lookups.idx_gases_sw))), ", ")))")

    RRTMGP.check_values[] = check_values
    return solver
end

# ── The per-call column write + solve ─────────────────────────────────────────

"""
    rrtmgp_solve_columns!(solver, T_lay, p_lay, T_lev, p_lev, vmr_h2o, o3, cld,
                          ext, t_sfc, cos_zenith, toa_flux) -> nothing

Write one tile's worth of columns into `solver` and run it.

# Shapes (index 1 = surface, z ascending)
- `T_lay`, `p_lay`, `vmr_h2o` -- `(nlay_model, ncol)` [K, Pa, -]
- `T_lev`, `p_lev` -- `(nlay_model + 1, ncol)` [K, Pa]; row `nlay_model+1` is
  the model top face, the SEAM the extension is attached to.
- `o3` -- either `(nlay_model,)`, one profile shared by every column, or
  `(nlay_model, ncol)`, one per column. [`model_ozone`](@ref) produces the
  vector form (Scythe's prescribed ozone is a function of height only); the
  matrix form is accepted so that a driver which already stages ozone into its
  per-column input batch does not have to special-case it, and so that a future
  per-column, log-p ozone needs no signature change here.
- `cld` -- `(lwp, iwp, re_liq, re_ice, cf)`, each `(nlay_model, ncol)`,
  [g/m^2, g/m^2, micron, micron, -]. Ignored unless the method carries a cloud
  state (`:allsky`, `:allsky_clear`).
- `t_sfc` -- `(ncol,)` [K]. The RADIATIVE surface temperature: SST where there
  is one, otherwise the extrapolated surface air temperature.
- `cos_zenith`, `toa_flux` -- scalars broadcast to all columns. `toa_flux` is
  the BEAM-NORMAL flux; RRTMGP multiplies by `cos_zenith` internally (S0
  measured `sw_flux_dn(TOA) = 142.15` against `toa_flux * cos_zenith = 142.75`,
  a 0.4% difference that is the solar spectral rescaling inside the lookup
  tables, not a convention mismatch).

# Seam corrections
The extension is a single climatological profile shared by the whole tile, but
it has to join each column's own state continuously, or the discontinuity at the
seam radiates as a spurious thin layer. Three corrections, all cheap and all
per-column:

  * **Pressure** is rescaled multiplicatively, `p_ext * (p_lev[end,c] /
    ext.p_face[1])`, so the extension's bottom face is EXACTLY the column's top
    face pressure. Multiplicative rather than additive because pressure is
    exponential in height: a constant offset would distort the layer thicknesses
    in `ln p` (and hence the absorber distribution) far more than a constant
    scaling, which preserves them exactly.
  * **Temperature** takes the seam mismatch `T_lev[end,c] - ext.T_face[1]` and
    blends it out with `exp(-(z - z_top)/5 km)`, so the extension's bottom face
    matches the column exactly and its top is the unmodified climatology. The
    model's near-top temperature is sponge-damped, not free, so carrying its
    anomaly all the way to 70 km would be trusting it well past where it means
    anything.
  * **Water vapor** is scaled by the ratio at the seam, then floored at
    `RRTMGP_VMR_H2O_FLOOR` (2 ppmv). A column whose top layer happens to be very
    dry must not drive the whole stratosphere to zero vapor -- and a wet one
    must not flood it. When either side of the ratio is not usable (a
    non-positive extension value, a non-finite model value) the ratio falls back
    to 1.

Ozone is used VERBATIM: it is prescribed climatology on both sides of the seam,
so there is nothing to blend. Extension cloud fields are zeroed -- there is no
cloud above 25 km worth representing, and a stale value from a previous call
would be a phantom.
"""
function Scythe.rrtmgp_solve_columns!(solver, T_lay::AbstractMatrix, p_lay::AbstractMatrix,
                               T_lev::AbstractMatrix, p_lev::AbstractMatrix,
                               vmr_h2o::AbstractMatrix, o3::AbstractVecOrMat,
                               cld::NTuple{5,<:AbstractMatrix},
                               ext::StratosphericExtension,
                               t_sfc::AbstractVector,
                               cos_zenith::Float64, toa_flux::Float64)
    nlay_model, ncol = size(T_lay)
    rm = RRTMGP.radiation_method(solver)
    is_gray = rm isa RRTMGP.GrayRadiation
    has_cloud = rm isa RRTMGP.AllSkyRadiation ||
                rm isa RRTMGP.AllSkyRadiationWithClearSkyDiagnostics

    sol_t_lay = RRTMGP.layer_temperature(solver)
    sol_p_lay = RRTMGP.layer_pressure(solver)
    sol_t_lev = RRTMGP.level_temperature(solver)
    sol_p_lev = RRTMGP.level_pressure(solver)

    size(sol_t_lay, 1) == nlay_model + ext.nlay || error(
        "rrtmgp_solve_columns!: the solver has $(size(sol_t_lay, 1)) domain " *
        "layers but nlay_model + ext.nlay = $(nlay_model + ext.nlay)")
    size(sol_t_lay, 2) == ncol || error(
        "rrtmgp_solve_columns!: the solver has $(size(sol_t_lay, 2)) columns, " *
        "the state has $ncol")
    size(p_lay) == (nlay_model, ncol) || error("rrtmgp_solve_columns!: p_lay shape")
    size(T_lev) == (nlay_model + 1, ncol) || error("rrtmgp_solve_columns!: T_lev shape")
    size(p_lev) == (nlay_model + 1, ncol) || error("rrtmgp_solve_columns!: p_lev shape")
    size(vmr_h2o) == (nlay_model, ncol) || error("rrtmgp_solve_columns!: vmr_h2o shape")
    _check_o3_shape(o3, nlay_model, ncol)
    length(t_sfc) == ncol || error("rrtmgp_solve_columns!: t_sfc length")

    _write_thermo!(sol_t_lay, sol_p_lay, sol_t_lev, sol_p_lev,
                   T_lay, p_lay, T_lev, p_lev, ext, nlay_model, ncol)

    if !is_gray
        _write_gases!(RRTMGP.volume_mixing_ratio(solver, "h2o"),
                      RRTMGP.volume_mixing_ratio(solver, "o3"),
                      vmr_h2o, o3, p_lev, ext, nlay_model, ncol)
    end

    if has_cloud
        _write_clouds!(RRTMGP.cloud_liquid_water_path(solver),
                       RRTMGP.cloud_ice_water_path(solver),
                       RRTMGP.cloud_liquid_effective_radius(solver),
                       RRTMGP.cloud_ice_effective_radius(solver),
                       RRTMGP.cloud_fraction(solver),
                       cld, nlay_model, ncol)
    end

    RRTMGP.surface_temperature(solver) .= t_sfc
    RRTMGP.cos_zenith(solver) .= cos_zenith
    RRTMGP.toa_sw_flux_dn(solver) .= toa_flux

    # There is no way to skip the shortwave when the sun is down. RRTMGP's
    # kernels guard the RTE solve on `cos_zenith > 0` and zero those columns
    # (`set_flux_to_zero!`), so the ANSWER is exactly zero -- but the shortwave
    # gas optics are still computed for every g-point before the guard, so a
    # night-time or `:solar = :none` call costs nearly as much as a daytime one.
    # Splitting `update_fluxes!` to run only the longwave is not safe through
    # the Layer-2 API: `update_net_fluxes!` reads the shortwave solver's
    # internal compute buffer, which is `undef` until the first shortwave solve.
    RRTMGP.update_fluxes!(solver)
    return nothing
end

"""
    rrtmgp_fluxes(solver) -> NamedTuple

The six flux fields after a solve: `(lw_up, lw_dn, lw_net, sw_up, sw_dn,
sw_net)`, each an `(nlev_total, ncol)` array indexed from the SURFACE
(`[1, c]`) to the top of the extension (`[end, c]`), where
`nlev_total = nlay_model + ext.nlay + 1`. `net = up - down`.

**These alias the solver's own memory.** They are views into the buffers the
next [`rrtmgp_solve_columns!`](@ref) overwrites, and they are not copies: read
them, reduce them, or copy what must outlive the call (`RadiationState`'s
`flux_lw_up` and friends exist for exactly that). Nothing here is writable in
any meaningful sense -- writing would corrupt the diagnostic without changing
the physics.

The isothermal boundary layer is already masked off by RRTMGP's getters, so
`nlev_total` is the physical count, with no pad row to skip.
"""
function Scythe.rrtmgp_fluxes(solver)
    return (lw_up = RRTMGP.lw_flux_up(solver),
            lw_dn = RRTMGP.lw_flux_dn(solver),
            lw_net = RRTMGP.lw_flux_net(solver),
            sw_up = RRTMGP.sw_flux_up(solver),
            sw_dn = RRTMGP.sw_flux_dn(solver),
            sw_net = RRTMGP.sw_flux_net(solver))
end

# ── The offline single-column anchor ──────────────────────────────────────────

"""
    radiation_offline_column(; kind = :tropical, nlay = 60, n_ext = 15,
                             z_top = 25.0e3, method = :clearsky, solar = :fixed,
                             cos_zenith = 0.2588, toa_flux = 551.58,
                             sfc_emissivity = 0.98, sfc_albedo = 0.06,
                             t_sfc = nothing, gases = default_gases(),
                             check_values = false)

Run one model-shaped column end to end and return everything needed to check it.

This is the S1 ANCHOR. It exercises exactly the code path a live tile uses --
`radiation_faces` -> `radiation_levels!` -> [`standard_extension`](@ref) ->
[`model_ozone`](@ref) -> [`rrtmgp_solver`](@ref) ->
[`rrtmgp_solve_columns!`](@ref) -> [`rrtmgp_fluxes`](@ref) ->
[`radiation_divergence!`](@ref) -- differing only in where the thermodynamic
state comes from (here, the standard atmosphere; there, the prognostic state).
S2b's first `:full` run on a horizontally homogeneous sounding must reproduce
this answer to round-off, because it is the same arithmetic on the same numbers.

# The column
`nlay` uniform layers from the ground to `z_top`, layer midpoints at
`(k-1/2) dz`, so `radiation_faces` returns exactly the cell boundaries and
`sum(dz) == z_top` to the last bit. The thermodynamic state is the standard
atmosphere `kind` evaluated at those heights, converted to Scythe's variables:

    p_d = p / (1 + vmr),   rho_d = p_d / (Rd T),   rho_v = (p - p_d) / (Rv T)

(`vmr = n_v/n_d`, so `p = p_d (1 + vmr)`), and back the other way for RRTMGP,
`vmr = (rho_v/rho_d) (Rv/Rd)`, which is the same conversion `radiation_assemble!`
will use in S3. Level pressures and temperatures come from `radiation_levels!`,
NOT from the standard atmosphere's own levels -- the point is to test Scythe's
interpolation and hydrostatic end extrapolation, not RRTMGP's analytic profile.

`t_sfc` defaults to the profile's own surface temperature (300 K for
`:tropical`), which is the surface the sounding is in equilibrium with.

# Solar
`solar = :none` forces `cos_zenith = toa_flux = 0` and gives an identically zero
shortwave field. Any other value uses the `cos_zenith`/`toa_flux` arguments
as given; the defaults (0.2588, 551.58) are the standard idealized-RCE pair,
a 142.7 W/m^2 daily-mean insolation.

# Returns
A `NamedTuple`. The contract fields are `z`, `dz`, `q_lw`, `q_sw`, `fluxes`
(the [`rrtmgp_fluxes`](@ref) named tuple, still aliasing the solver) and `olr`.
It also carries the column state the caller needs to convert `q` [W/m^3] into
K/day without rebuilding it -- `p`, `Tk`, `rho`, `cp`, `vmr_h2o`, `o3`,
`z_face`, `T_lev`, `p_lev`, `t_sfc`, `cos_zenith`, `toa_flux` -- plus `ext` and
`solver` for inspection.
"""
function Scythe.radiation_offline_column(; kind::Symbol = :tropical,
                                  nlay::Int = 60,
                                  n_ext::Int = 15,
                                  z_top::Float64 = 25.0e3,
                                  method::Symbol = :clearsky,
                                  solar::Symbol = :fixed,
                                  cos_zenith::Float64 = 0.2588,
                                  toa_flux::Float64 = 551.58,
                                  sfc_emissivity::Float64 = 0.98,
                                  sfc_albedo::Float64 = 0.06,
                                  t_sfc::Union{Nothing,Float64} = nothing,
                                  gases::Dict{String,Float64} = default_gases(),
                                  check_values::Bool = false)
    prof = _standard_profile(kind === :none ? :tropical : kind)

    dzc = z_top / nlay
    z = [(k - 0.5) * dzc for k in 1:nlay]
    z_face, dz = radiation_faces(z, 0.0, z_top)

    work = RadiationWork(nlay)
    vmr_h2o_col = Vector{Float64}(undef, nlay)
    @inbounds for k in 1:nlay
        Tk = _interp_linear(prof.z_lev, prof.t_lev, z[k])
        p = exp(_interp_linear(prof.z_lev, prof.lnp_lev, z[k]))
        vmr = _interp_linear(prof.z_lay, prof.vmr_h2o, z[k])
        p_d = p / (1.0 + vmr)
        work.p[k] = p
        work.Tk[k] = Tk
        work.rho_d[k] = p_d / (Rd * Tk)
        work.rho_v[k] = (p - p_d) / (Rv * Tk)
        work.rho_liq[k] = 0.0
        work.rho_ice[k] = 0.0
        # Round-trip through Scythe's variables, exactly as the live driver will.
        vmr_h2o_col[k] = (work.rho_v[k] / work.rho_d[k]) * (Rv / Rd)
    end
    radiation_levels!(work, z, z_face)

    ext = standard_extension(kind, n_ext, z_top)
    o3 = model_ozone(kind, z)

    cosz, toa = solar === :none ? (0.0, 0.0) : (cos_zenith, toa_flux)
    ts = t_sfc === nothing ? prof.t_sfc : t_sfc

    solver = rrtmgp_solver(nlay, ext, 1, method, solar, check_values;
                           sfc_emissivity = sfc_emissivity,
                           sfc_albedo = sfc_albedo, gases = gases,
                           z_face = vcat(z_face), latitude = prof.lat)

    T_lay = reshape(work.Tk, nlay, 1)
    p_lay = reshape(work.p, nlay, 1)
    T_lev = reshape(work.T_face, nlay + 1, 1)
    p_lev = reshape(work.p_face, nlay + 1, 1)
    vmr_h2o = reshape(vmr_h2o_col, nlay, 1)
    zero_cld = zeros(Float64, nlay, 1)
    cld = (zero_cld, copy(zero_cld), copy(zero_cld), copy(zero_cld), copy(zero_cld))

    rrtmgp_solve_columns!(solver, T_lay, p_lay, T_lev, p_lev, vmr_h2o, o3, cld,
                          ext, [ts], cosz, toa)

    fluxes = rrtmgp_fluxes(solver)
    q_lw = Vector{Float64}(undef, nlay)
    q_sw = Vector{Float64}(undef, nlay)
    radiation_divergence!(q_lw, fluxes.lw_net, dz, nlay, 1, nlay, 1)
    radiation_divergence!(q_sw, fluxes.sw_net, dz, nlay, 1, nlay, 1)

    # The K/day conversion, in exactly the form `_rad_kday` (src/radiation.jl) uses, so
    # this anchor and a live run's trace are the same number with no factor between them:
    # `rho` is the TOTAL mass density and `rho * cp = rho_d C_pd + rho_v C_pv +
    # rho_liq C_l + rho_ice C_i`. The condensate terms are identically zero on this
    # clear-sky standard-atmosphere column; they are written out anyway so the definition
    # here is literally the shared one rather than a special case of it.
    rho = work.rho_d .+ work.rho_v .+ work.rho_liq .+ work.rho_ice
    cp = (Cpd .* work.rho_d .+ Cpv .* work.rho_v .+
          Cl .* work.rho_liq .+ Ci .* work.rho_ice) ./ rho

    return (z = z, dz = dz, z_face = z_face,
            q_lw = q_lw, q_sw = q_sw,
            fluxes = fluxes,
            olr = fluxes.lw_up[end, 1],
            p = copy(work.p), Tk = copy(work.Tk),
            rho_d = copy(work.rho_d), rho_v = copy(work.rho_v),
            rho = rho, cp = cp,
            vmr_h2o = vmr_h2o_col, o3 = o3,
            T_lev = copy(work.T_face), p_lev = copy(work.p_face),
            t_sfc = ts, cos_zenith = cosz, toa_flux = toa,
            ext = ext, solver = solver)
end


end # module ScytheRRTMGPExt
