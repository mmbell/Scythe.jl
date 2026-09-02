# ── Radiation driver: setup, cadence, held forcing ────────────────────────────
#
# The half of the radiation code that knows about `ModelTile`. It is included AFTER
# moist_compressible.jl because it reads the driver's own thermodynamic helpers
# (`recover_rho_c`, `recover_total`, `retrieve_temperature`, the transform-mode
# accessors) rather than re-deriving them: a radiation column that disagreed with
# `mc_driver!` about what the cloud IS would put that disagreement straight into the
# heating field.
#
# It names NO radiative-transfer library. Every RRTMGP/ClimaComms/NCDatasets reference
# lives in src/radiation_rrtmgp.jl, which this file calls through a small fixed
# interface (`rrtmgp_solver`, `rrtmgp_solve_columns!`, `rrtmgp_fluxes`,
# `radiation_divergence!`, `standard_extension`, `model_ozone`, `default_gases`). The
# split keeps the dependency at ONE file and makes the eventual conversion of RRTMGP to
# a package extension mechanical.
#
# What S2a ships (this stage): the plumbing. `ModelTile` carries a `RadiationState`, the
# pre-pass runs once per step before the threaded column loop, and the held heating folds
# into `QDOT_TH` inside `mc_driver!`. The `:prescribed` scheme — a constant K/day cooling
# converted to W/m^3 through the local `rho_d C_vt` — is the artifact-free regression
# target for all of that. The `:rrtmgp` path is written here but is not exercised until
# S2b.
#
# Sign and unit conventions are those of radiation_state.jl: `q_lw`/`q_sw` are flux
# divergences in W/m^3, cooling is negative, layer 1 is the surface, and gridpoint row
# `j = (c-1)*kDim + k` matches `expdot` exactly so the driver's fold needs no reshape.

# ── Per-tile batch of solver inputs ───────────────────────────────────────────

"""
    RadiationBatch(nlay, ncol)

The `(nlay, ncol)` input matrices one radiation call hands to the solver: layer and level
temperature/pressure, the water-vapour and ozone volume mixing ratios, the five
cloud-optics fields, and the per-column surface temperature.

Held in `RadiationState.solver` as the NamedTuple
`(solver = <solver>, batch = <this>, cloud_params = <CloudOpticsParams>)` rather than as
its own `RadiationState` field. Two reasons: `RadiationState` is the LEAF struct
(radiation_state.jl), so a field of this type would have to be defined there, where the
batch has no meaning; and `solver::Any` already exists precisely because that slot is
read once per radiation call from serial code, which makes the extra field lookup free.
`mc_radiation_state` is the only writer and always stores the triple, so a `:rrtmgp`
state's `solver` is a NamedTuple and a `:prescribed` state's is `nothing`.

`cloud_params` rides in the same bundle for the same reason and is resolved ONCE at
setup (S3b): `CloudOpticsParams(options, physical_params)` reads `:max_N_c`,
`:cloud_k_factor`, `:radiation_q_min` and `:radiation_rain_in_cloud` out of the model
tables, and re-reading a `Dict` per column per call would be both slower and a place for
a mid-run parameter change to half-apply.

Column `c` of every matrix is the model column `c` of the tile; row `k` is radiation
layer `k`, counting up from the surface.
"""
struct RadiationBatch
    T_lay::Matrix{Float64}
    p_lay::Matrix{Float64}
    T_lev::Matrix{Float64}
    p_lev::Matrix{Float64}
    vmr_h2o::Matrix{Float64}
    o3::Matrix{Float64}
    lwp::Matrix{Float64}
    iwp::Matrix{Float64}
    re_liq::Matrix{Float64}
    re_ice::Matrix{Float64}
    cf::Matrix{Float64}
    t_sfc::Vector{Float64}
    # ── per-COLUMN condensate scratch (S3b), overwritten once per column ──
    rho_c::Vector{Float64}
    rho_r::Vector{Float64}
    ice::Matrix{Float64}
    ice_on::Bool
end

"""
    RadiationBatch(nlay, ncol; ice_on = false)

`ice_on` sizes the `(nlay, 12)` ice-moment scratch: `(nlay, 12)` when the configuration
registered the twelve ISHMAEL slots, `(0, 12)` when it did not (the assembly then hands
`nothing` to `cloud_optics_column!`, which is its documented ice-off path, rather than a
block of zeros that would cost a per-layer loop over three species to discover).
"""
RadiationBatch(nlay::Int, ncol::Int; ice_on::Bool = false) =
    RadiationBatch(zeros(Float64, nlay, ncol), zeros(Float64, nlay, ncol),
                   zeros(Float64, nlay + 1, ncol), zeros(Float64, nlay + 1, ncol),
                   zeros(Float64, nlay, ncol), zeros(Float64, nlay, ncol),
                   zeros(Float64, nlay, ncol), zeros(Float64, nlay, ncol),
                   zeros(Float64, nlay, ncol), zeros(Float64, nlay, ncol),
                   zeros(Float64, nlay, ncol), zeros(Float64, ncol),
                   zeros(Float64, nlay), zeros(Float64, nlay),
                   zeros(Float64, ice_on ? nlay : 0, 12), ice_on)

# ── Setup ─────────────────────────────────────────────────────────────────────

"""
    mc_radiation_state(model, tile, tilepoints) -> RadiationState

The tile's radiation state, or [`EMPTY_RADIATION`](@ref) when `options[:radiation]` is
absent or `:none`.

Setup path only — once per tile, never per column, and the exact analogue of
[`mc_ishmael_tables`](@ref) beside it in `createModelTile`. Everything that can be wrong
about a radiation configuration is diagnosed HERE, by
[`validate_radiation_options`](@ref) (including radiation asked for on an equation set
with no `QDOT_TH` to fold into), rather than at the first radiation call several minutes
into a run.

The column geometry is taken from ONE column of `tilepoints`: the vertical mish is
identical for every column of a tile, which is what lets a single `z`/`z_face`/`dz`
triple and a single taper serve the whole tile. `z_bottom`/`z_top` are the grid's own
`kMin`/`kMax`, so `sum(dz) == kMax - kMin` exactly (see [`radiation_faces`](@ref)).

`options[:radiation_layer_stride] > 1` groups `stride` model gridpoints into one
radiation layer. The taper and the `:anomaly` reference profile are indexed by LAYER and
the held `q_lw`/`q_sw` by GRIDPOINT, which is the mapping `radiation_store!` applies; at
the default stride 1 the two indexings coincide. The `:rrtmgp` column packing is written
for stride 1 only and refuses anything else here rather than silently mis-stacking a
column.
"""
function mc_radiation_state(model::ModelParameters, tile, tilepoints)

    scheme = get(model.options, :radiation, :none)
    scheme === :none && return EMPTY_RADIATION

    kDim = model.grid_params.kDim
    cfg = validate_radiation_options(model.options, model.physical_params,
                                     model.equation_set, model.ts, kDim)

    ncol = num_columns(tile)
    ncol > 0 || error(
        "options[:radiation] = :$(scheme) needs a grid with vertical columns " *
        "(num_columns(tile) = 0); radiation is a column process")
    kDim > 0 || error("options[:radiation] = :$(scheme) needs a vertical dimension " *
                      "(grid_params.kDim = 0)")

    # One column of the vertical mish. `tilepoints[:, end]` is the z coordinate on every
    # moist-compressible geometry (2-D: column 2; 3-D: column 3), the same expression
    # `createModelTile` uses for `mc_reference_diagnostics`.
    z = collect(Float64, tilepoints[1:kDim, end])
    z_bottom = Float64(model.grid_params.kMin)
    z_top = Float64(model.grid_params.kMax)

    stride = cfg.stride
    nlay = div(kDim, stride)
    # Radiation-layer geometry. At stride 1 the layers ARE the gridpoints; at stride s the
    # layer faces are every s-th gridpoint face, so the layers still tile the column
    # exactly and `sum(dz)` is unchanged (the conservation argument of `radiation_faces`).
    zf_gp, _ = radiation_faces(z, z_bottom, z_top)
    if stride == 1
        z_lay = z
        z_face = zf_gp
    else
        z_face = zf_gp[1:stride:(kDim + 1)]
        z_lay = [sum(@view z[((j - 1) * stride + 1):(j * stride)]) / stride
                 for j in 1:nlay]
    end
    dz = diff(z_face)

    taper = radiation_taper(z_lay, cfg.z_max)

    npoints = size(tile.physical, 1)
    q_lw = zeros(Float64, npoints)
    q_sw = zeros(Float64, npoints)
    q_lw_ref = zeros(Float64, nlay)
    q_sw_ref = zeros(Float64, nlay)

    work = RadiationWork(kDim)
    cloud = CloudOpticsColumn(kDim)

    if scheme === :prescribed
        # No solver, no extension, no ozone: the prescribed heating is a local
        # thermodynamic product, not a radiative transfer. The flux diagnostics are still
        # allocated (at the model's own face count) so the output writer of S5 has one
        # shape to write whatever the scheme is.
        nlev = nlay + 1
        return RadiationState(; active = true, scheme, forcing = cfg.forcing,
            method = cfg.method, solar = cfg.solar, level_interp = cfg.level_interp,
            extension_kind = :none, interval_steps = cfg.interval_steps, stride, nlay,
            ncol, kDim, z_max = cfg.z_max, sw_rescale = cfg.sw_rescale,
            rain_in_cloud = cfg.rain_in_cloud, output = cfg.output,
            check_values = cfg.check_values,
            q_lw, q_sw, q_lw_ref, q_sw_ref, z = z_lay, z_face, dz,
            o3 = Float64[], taper,
            flux_lw_up = zeros(Float64, nlev, ncol),
            flux_lw_dn = zeros(Float64, nlev, ncol),
            flux_lw_net = zeros(Float64, nlev, ncol),
            flux_sw_up = zeros(Float64, nlev, ncol),
            flux_sw_dn = zeros(Float64, nlev, ncol),
            flux_sw_net = zeros(Float64, nlev, ncol),
            extension = EMPTY_EXTENSION, work, cloud,
            gases = Dict{String,Float64}(), solver = nothing)
    end

    # ── :rrtmgp ───────────────────────────────────────────────────────────────
    stride == 1 || error(
        "options[:radiation_layer_stride] = $stride is not wired for " *
        "options[:radiation] = :rrtmgp yet (S2a ships the stride-1 column packing); " *
        "the stride lever arrives with the cost measurement it exists for")

    ext = standard_extension(cfg.extension, cfg.n_ext, z_top)
    o3 = model_ozone(cfg.extension, z_lay)
    gases = merge(default_gases(model.physical_params),
                  Dict{String,Float64}(get(model.options, :radiation_gases,
                                           Dict{String,Float64}())))
    sfc_emis = get(model.physical_params, :sfc_emissivity, 0.98)
    sfc_alb = get(model.physical_params, :sfc_albedo, 0.06)

    solver = try
        # `z_face` and `latitude` are read only by the `:gray` method (level altitudes and
        # the latitude-dependent optical thickness); the spectral methods work in pressure
        # and ignore both. Passed unconditionally so a `:gray` run is configured, not
        # refused, and so nothing here has to know which method needs what.
        rrtmgp_solver(nlay, ext, ncol, cfg.method, cfg.solar, cfg.check_values;
                      sfc_emissivity = sfc_emis, sfc_albedo = sfc_alb, gases = gases,
                      z_face = z_face,
                      latitude = Float64(get(model.physical_params, :latitude, 0.0)))
    catch err
        error("mc_radiation_state: building the RRTMGP solver failed " *
              "(method = :$(cfg.method), solar = :$(cfg.solar), nlay = $nlay, " *
              "n_ext = $(ext.nlay), ncol = $ncol). The usual cause is a missing lookup " *
              "artifact on a node with no network — run tools/rrtmgp_prewarm.jl on a " *
              "login node first. Original error: $(sprint(showerror, err))")
    end
    # The twelve ISHMAEL slots are registered by NAME (and aliased under a transform), so
    # "does this configuration carry ice" is asked of `vars` here, exactly as
    # `mc_ice_slot_indices` asks it in `createModelTile`; `mtile.mc_slots` does not exist
    # yet at this point in the tile construction.
    ice_on = all(>(0), mc_ice_slot_indices(model.grid_params.vars))
    batch = RadiationBatch(nlay, ncol; ice_on = ice_on)
    cloud_params = CloudOpticsParams(model.options, model.physical_params)
    nlev_tot = nlay + 1 + ext.nlay

    return RadiationState(; active = true, scheme, forcing = cfg.forcing,
        method = cfg.method, solar = cfg.solar, level_interp = cfg.level_interp,
        extension_kind = cfg.extension, interval_steps = cfg.interval_steps, stride,
        nlay, ncol, kDim, z_max = cfg.z_max, sw_rescale = cfg.sw_rescale,
        rain_in_cloud = cfg.rain_in_cloud, output = cfg.output,
        check_values = cfg.check_values,
        q_lw, q_sw, q_lw_ref, q_sw_ref, z = z_lay, z_face, dz, o3, taper,
        flux_lw_up = zeros(Float64, nlev_tot, ncol),
        flux_lw_dn = zeros(Float64, nlev_tot, ncol),
        flux_lw_net = zeros(Float64, nlev_tot, ncol),
        flux_sw_up = zeros(Float64, nlev_tot, ncol),
        flux_sw_dn = zeros(Float64, nlev_tot, ncol),
        flux_sw_net = zeros(Float64, nlev_tot, ncol),
        extension = ext, work, cloud, gases,
        solver = (solver = solver, batch = batch, cloud_params = cloud_params))
end

# ── The per-step pre-pass ─────────────────────────────────────────────────────

"""
    radiation_prepass!(mtile, t)

The once-per-step radiation entry point, called from every tile advance IMMEDIATELY
BEFORE the `Threads.@threads :static` column loop (`advanceTimestep`,
`advance_nested_timestep`, `advance_nested_timestepA`, `advanceTimestepA`). At that
point `tile.physical` has just been materialized, the CFL check has passed and no
worker thread is running, so this can read the whole tile state serially and write the
held forcing that every column will then read.

Three things happen, in this order:

 1. `sw_scale` is refreshed EVERY step (never only on a radiation call). The longwave
    heating is held piecewise-constant between calls, but the shortwave has a diurnal
    factor that moves on the timestep, so the held SW profile is rescaled by
    `cos_z(now)/cos_z(call)`. The ratio is clamped to `[0, 4]` — near sunrise the
    denominator is small and an unclamped ratio would amplify the held profile without
    bound — and is exactly 0 at night. With `:solar = :fixed` (or `:none`) the rescale
    is off by default and `sw_scale` stays 1.
 2. The solve itself, when the cadence has elapsed. The FIRST call is forced regardless
    (`last_call_step` starts at `typemin(Int)`), so step 1 always has a heating field.
 3. `radiation_write!`, the sidecar diagnostic output — a stub until S5.
"""
function radiation_prepass!(mtile::ModelTile, t::Int64)

    rs = mtile.radiation
    rs.active || return nothing

    model = mtile.model
    # Zero-based model time, identical across nest patches (D6).
    t_model = (t - 1) * model.ts

    if rs.sw_rescale
        cz_now, _ = solar_state(model.options, model.physical_params, t_model)
        rs.sw_scale = rs.cos_zenith > 0.0 ?
            clamp(cz_now / rs.cos_zenith, 0.0, 4.0) : 0.0
    end

    if rs.last_call_step == typemin(Int) || (t - rs.last_call_step) >= rs.interval_steps
        radiation_update!(mtile, t)
    end

    radiation_write!(mtile, t)
    return nothing
end

"""
    radiation_update!(mtile, t)

Recompute the held radiative heating for the whole tile at step `t`.

Solar geometry is evaluated at the INTERVAL MIDPOINT, `((t-1) + interval/2)*ts`, not at
the call time: the resulting profile is held for the whole interval, so the midpoint is
the value whose time integral over the interval is right to second order. `sw_scale` is
reset to 1 here because the held profile has just been recomputed at the new zenith
angle — the rescale measures drift SINCE the call.

`last_call_step` is updated LAST, after [`radiation_store!`](@ref), because the
`:anomaly` reference profile is captured on the first call and "is this the first call"
is exactly `last_call_step == typemin(Int)`.
"""
function radiation_update!(mtile::ModelTile, t::Int64)

    rs = mtile.radiation
    model = mtile.model
    t_start = time_ns()

    t_mid = ((t - 1) + 0.5 * rs.interval_steps) * model.ts
    cz, toa = solar_state(model.options, model.physical_params, t_mid)
    rs.cos_zenith = cz
    rs.toa_flux = toa
    rs.sw_scale = 1.0

    if rs.scheme === :prescribed
        radiation_prescribed!(mtile)
    else
        radiation_rrtmgp_update!(mtile)
    end

    radiation_store!(mtile)
    rs.last_call_step = t
    # The wall time of the WHOLE call (column reconstruction, level build, cloud optics,
    # the solve and the store), which is the number the cadence has to be chosen against.
    # `RadiationState` has no field to park it in (its layout is fixed in the leaf file),
    # so the trace is where it is reported.
    radiation_trace!(mtile, t, (time_ns() - t_start) * 1.0e-6)
    return nothing
end

"""
    radiation_write!(mtile, t)

Sidecar radiation output (plan D9). A STUB in S2a: the writer, `read_radiation` and the
final-snapshot hook are S5. It is called from the pre-pass rather than from
`write_output` because the fluxes live on faces and on the extension, neither of which is
on the prognostic variable grid.
"""
radiation_write!(::ModelTile, ::Int64) = nothing

# ── The per-call trace ────────────────────────────────────────────────────────

"""
The heights [m] the trace reports the domain-mean heating at: the boundary layer, the
mid-troposphere, the outflow level, the O01 anvil/glaciation level (13 km), the upper
troposphere, and two levels inside the DK83 sponge (17-25 km), where a radiative tendency
is fighting the damping rather than doing physics and is therefore the thing to watch.
"""
const RADIATION_TRACE_HEIGHTS = (1.0e3, 5.0e3, 10.0e3, 13.0e3, 15.0e3, 20.0e3, 24.0e3)

"""
    radiation_trace!(mtile, t, wall_ms)

Print ONE compact block per radiation call, in the style of
[`mc_stiffness_trace`](@ref) and `water_budget_trace` (an `@info` on the worker, so the
output lands in `<output_dir>/scythe_err.log`, not on the master's console).

On by default (`options[:radiation_trace]`, default `true`) because the sidecar NetCDF
writer is S5: until then this block is the ONLY window onto what the radiation is doing
inside a run, and a radiation stage with no window is a stage whose numbers cannot be
reported. Set it to `false` for a run that wants the log quiet.

What it reports, and why each line is there:

- **OLR twice.** `flux_lw_up` at the top of the FULL column (the extension's top, ~70 km)
  is the OLR a satellite would see and the number the S1 offline anchor is quoted at; at
  the MODEL-top face (row `nlay+1`, 25 km) it is the same flux before the extension's
  ~25 hPa of ozone and CO2 have acted on it. The two differ by the extension's own
  emission, so labelling them apart is the difference between a check and a coincidence.
- **The heating profile in K/day**, converted with the column's OWN `rho_d C_vt` —
  the same mixture heat capacity `mc_driver!` uses — because W/m^3 is the unit the
  forcing is applied in and K/day is the unit it can be judged in. That is a CONSTANT
  VOLUME rate: it is the model's own internal-energy response to `QDOT_TH`, and it is
  `C_p/C_v ~ 1.4` times the constant-pressure rate that textbook and offline-column
  numbers are quoted at (`radiation_offline_column` returns `cp`, not `C_vt`). Both
  conventions are defensible; only silently mixing them is not, hence the label.
- **The extreme of `q_lw` over the whole tile with its height**, which is where a
  cloud-top cooling maximum (S3) or a bad extension seam would announce itself.
- **The cloud census** (S3b), read out of the batch the assembly has just filled: how
  many columns carry cloud anywhere, the domain mean and maximum COLUMN-INTEGRATED liquid
  and ice water path [g/m^2], the water-path-weighted effective radii over cloudy layers
  (the average an optical depth feels, not an average over layers), and the `q_lw`
  extremes restricted to CLOUDY columns -- which is where cloud-top cooling and cloud-base
  warming live and where a domain-wide extreme, dominated by the clear-sky model top,
  would hide them.
- **The four clamp/saturation counters**, cumulative over the run. They are the reason a
  clamp is never silent.

`work` is reused for the K/day conversion, which means the column reconstruction runs a
second time. That costs a few milliseconds against a solve that costs seconds, and it
keeps the trace from needing a per-gridpoint `rho_d C_vt` buffer that nothing else wants.
`radiation_column_state!` COUNTS its temperature clamps, though, so the counter is
snapshotted and restored around the loop: a diagnostic must not inflate the census it
reports.
"""
function radiation_trace!(mtile::ModelTile, t::Int64, wall_ms::Float64)

    rs = mtile.radiation
    (rs.active && get(mtile.model.options, :radiation_trace, true)::Bool) ||
        return nothing
    ncol = rs.ncol
    (ncol > 0 && rs.kDim > 0) || return nothing

    model = mtile.model
    kDim = rs.kDim
    stride = rs.stride
    work = rs.work

    # The reported counters are the state as the UPDATE left it; the reconstruction loop
    # below would otherwise re-count every temperature clamp.
    n_tk = rs.n_clamp_tk
    n_rel = rs.n_clamp_re_liq
    n_rei = rs.n_clamp_re_ice
    n_negv = rs.n_neg_rho_v

    # Gridpoint index of the layer nearest each reported height (the layer's first
    # gridpoint; identical at the default stride 1, which is the only stride :rrtmgp runs).
    nh = length(RADIATION_TRACE_HEIGHTS)
    kidx = ntuple(h -> begin
                      zt = RADIATION_TRACE_HEIGHTS[h]
                      lay = argmin(abs.(rs.z .- zt))
                      (lay - 1) * stride + 1
                  end, nh)

    sum_lw = zeros(Float64, nh)
    sum_sw = zeros(Float64, nh)
    qmin = Inf; qmax = -Inf
    kmin = 0; kmax = 0
    qmin_c = Inf; qmax_c = -Inf
    kmin_c = 0; kmax_c = 0

    # ── The cloud census (S3b), over the batch the assembly has just filled ──
    # The batch is the ONLY place the cloud-optics answer survives a call (the per-column
    # `CloudOpticsColumn` is overwritten column by column), so the census reads it rather
    # than recomputing the mapping: a diagnostic that recomputed could report a cloud the
    # solver never saw. `:prescribed` has no batch and the block below says so.
    #
    # The water paths are COLUMN SUMS (the per-layer `lwp`/`iwp` are g/m^2 already), and
    # the effective radii are weighted by the water path over CLOUDY layers only, which is
    # the average an optical depth actually feels -- an unweighted mean over layers would
    # be dominated by the thin edges of the cloud.
    B_ = rs.solver isa NamedTuple ? (rs.solver::NamedTuple).batch::RadiationBatch : nothing
    cloudy = falses(ncol)
    lwp_mean = 0.0; lwp_max = 0.0; iwp_mean = 0.0; iwp_max = 0.0
    n_cloudy = 0
    w_liq = 0.0; s_liq = 0.0; w_ice = 0.0; s_ice = 0.0
    if B_ !== nothing
        @inbounds for c in 1:ncol
            lc = 0.0; ic = 0.0; anycf = false
            for k in 1:rs.nlay
                lk = B_.lwp[k, c]; ik = B_.iwp[k, c]
                lc += lk; ic += ik
                if B_.cf[k, c] == 1.0
                    anycf = true
                    w_liq += lk; s_liq += lk * B_.re_liq[k, c]
                    w_ice += ik; s_ice += ik * B_.re_ice[k, c]
                end
            end
            lwp_mean += lc; iwp_mean += ic
            lc > lwp_max && (lwp_max = lc)
            ic > iwp_max && (iwp_max = ic)
            if anycf
                n_cloudy += 1
                cloudy[c] = true
            end
        end
        lwp_mean /= ncol; iwp_mean /= ncol
    end

    @inbounds for c in 1:ncol
        colstart = (c - 1) * kDim + 1
        radiation_column_state!(work, mtile, colstart, colstart + kDim - 1)
        for h in 1:nh
            k = kidx[h]
            rho_d = work.rho_d[k]
            q_v = work.rho_v[k] / rho_d
            q_l = work.rho_liq[k] / rho_d
            q_i = work.rho_ice[k] / rho_d
            f = 86400.0 / (rho_d * (Cvd + (q_v * Cvv) + (q_l * Cl) + (q_i * Ci)))
            sum_lw[h] += rs.q_lw[colstart + k - 1] * f
            sum_sw[h] += rs.q_sw[colstart + k - 1] * f
        end
        for k in 1:kDim
            rho_d = work.rho_d[k]
            q_v = work.rho_v[k] / rho_d
            q_l = work.rho_liq[k] / rho_d
            q_i = work.rho_ice[k] / rho_d
            f = 86400.0 / (rho_d * (Cvd + (q_v * Cvv) + (q_l * Cl) + (q_i * Ci)))
            v = rs.q_lw[colstart + k - 1] * f
            if v < qmin
                qmin = v; kmin = k
            end
            if v > qmax
                qmax = v; kmax = k
            end
            if cloudy[c]
                if v < qmin_c
                    qmin_c = v; kmin_c = k
                end
                if v > qmax_c
                    qmax_c = v; kmax_c = k
                end
            end
        end
    end
    rs.n_clamp_tk = n_tk

    # No Printf: Scythe does not depend on it anywhere else, and `round` to a fixed
    # number of digits with `lpad` reads the same in a log line.
    _fx(v, d, w) = lpad(string(round(v; digits = d)), w)
    prof = join((_fx(RADIATION_TRACE_HEIGHTS[h] / 1000.0, 1, 5) * " km: lw " *
                 _fx(sum_lw[h] / ncol, 4, 10) * "  sw " * _fx(sum_sw[h] / ncol, 4, 10)
                 for h in 1:nh), "\n  ")

    # `flux_lw_up` row 1 is the surface, row `nlay+1` the model top face, row `end` the
    # top of the stratospheric extension. On :prescribed these are identically zero
    # (there is no radiative transfer to have fluxes from) and the line says so.
    lwup = rs.flux_lw_up
    nlev_tot = size(lwup, 1)
    olr_line = if nlev_tot >= rs.nlay + 1
        top = view(lwup, nlev_tot, :)
        mid = view(lwup, rs.nlay + 1, :)
        z_ext = (rs.extension.nlay > 0 ? rs.extension.z_face[end] : rs.z_face[end]) / 1000.0
        "full column (" * string(round(z_ext; digits = 1)) * " km incl. extension): " *
        "mean " * _fx(sum(top) / ncol, 3, 9) * "  min " * _fx(minimum(top), 3, 9) *
        "  max " * _fx(maximum(top), 3, 9) * "\n  " *
        "model top (" * string(round(rs.z_face[end] / 1000.0; digits = 1)) * " km face): " *
        "mean " * _fx(sum(mid) / ncol, 3, 9) * "  min " * _fx(minimum(mid), 3, 9) *
        "  max " * _fx(maximum(mid), 3, 9)
    else
        "no flux diagnostic (scheme = :$(rs.scheme))"
    end

    zk(k) = (k >= 1 && k <= length(rs.z) * stride) ?
            rs.z[div(k - 1, stride) + 1] / 1000.0 : NaN

    cloud_line = if B_ === nothing
        "no cloud diagnostic (scheme = :$(rs.scheme))"
    else
        re_l = w_liq > 0.0 ? s_liq / w_liq : NaN
        re_i = w_ice > 0.0 ? s_ice / w_ice : NaN
        "cloudy columns " * string(n_cloudy) * "/" * string(ncol) *
        ", LWP mean " * _fx(lwp_mean, 4, 10) * " max " * _fx(lwp_max, 4, 10) *
        " g/m^2, IWP mean " * _fx(iwp_mean, 4, 10) * " max " * _fx(iwp_max, 4, 10) *
        " g/m^2\n  " *
        "path-weighted radii over cloudy layers: re_liq " * _fx(re_l, 3, 8) *
        " um, re_ice " * _fx(re_i, 3, 8) * " um\n  " *
        (n_cloudy == 0 ?
         "q_lw extremes over CLOUDY columns: none (no cloudy column this call)" :
         "q_lw extremes over CLOUDY columns: min " *
         _fx(qmin_c, 4, 10) * " K/day at " *
         string(round(zk(kmin_c); digits = 2)) * " km, max " *
         _fx(qmax_c, 4, 10) * " K/day at " *
         string(round(zk(kmax_c); digits = 2)) * " km")
    end

    @info """radiation call, step $t (t = $(round((t - 1) * model.ts; digits = 1)) s), scheme :$(rs.scheme)/:$(rs.method), forcing :$(rs.forcing)
  wall $(round(wall_ms; digits = 1)) ms for $ncol columns x $(rs.nlay) layers (+$(rs.extension.nlay) extension), cos_zenith = $(round(rs.cos_zenith; digits = 5)), toa_flux = $(round(rs.toa_flux; digits = 2)) W/m^2, sw_scale = $(rs.sw_scale)
  OLR [W/m^2], LW up at
  $olr_line
  domain-mean heating [K/day AT CONSTANT VOLUME, q/(rho_d C_vt) -- the model's own internal-energy response; divide by C_p/C_v ~ 1.4 for the conventional c_p rate the literature quotes] at the nearest layer to
  $prof
  q_lw extremes over the tile: min $(round(qmin; digits = 4)) K/day at $(round(zk(kmin); digits = 2)) km, max $(round(qmax; digits = 4)) K/day at $(round(zk(kmax); digits = 2)) km
  $cloud_line
  counters (cumulative): Tk clamps $n_tk, re_liq clamps $n_rel, re_ice clamps $n_rei, negative rho_v floored $n_negv"""
    return nothing
end

# ── :prescribed ───────────────────────────────────────────────────────────────

"""
    radiation_prescribed!(mtile)

`options[:radiation] = :prescribed`: a uniform radiative TEMPERATURE tendency
`physical_params[:radiation_prescribed_rate]` [K/day, negative = cooling], converted to
the W/m^3 flux divergence the driver's `QDOT_TH` wants,

    q_lw = rho_d * C_vt * rate / 86400,    q_sw = 0.

The conversion is local, not global: `C_vt = C_vd + q_v C_vv + q_l C_l + q_i C_i` is the
mixture heat capacity of the column state (reference/Scythe_moist_compressible.tex,
Eq. mixture_C), written here exactly as `mc_driver!` writes it, so a prescribed −1.5 K/day
really is −1.5 K/day of the model's own internal energy at every gridpoint rather than of
a dry-air surrogate.

This is the artifact-free regression target of the whole held-forcing mechanism: the
answer is a deterministic function of the state, so the pre-pass cadence, the taper, the
`:anomaly` subtraction and the `QDOT_TH` fold can all be tested with no lookup tables,
no network and no solver.
"""
function radiation_prescribed!(mtile::ModelTile)

    rs = mtile.radiation
    rate = get(mtile.model.physical_params, :radiation_prescribed_rate, -1.5)
    f = rate / 86400.0
    kDim = rs.kDim
    work = rs.work

    fill!(rs.q_sw, 0.0)
    @inbounds for c in 1:rs.ncol
        colstart = (c - 1) * kDim + 1
        colend = colstart + kDim - 1
        radiation_column_state!(work, mtile, colstart, colend)
        for k in 1:kDim
            rho_d = work.rho_d[k]
            q_v = work.rho_v[k] / rho_d
            q_l = work.rho_liq[k] / rho_d
            q_i = work.rho_ice[k] / rho_d
            C_vt = Cvd + (q_v * Cvv) + (q_l * Cl) + (q_i * Ci)
            rs.q_lw[colstart + k - 1] = rho_d * C_vt * f
        end
    end
    return nothing
end

# ── :rrtmgp ───────────────────────────────────────────────────────────────────

"""
    radiation_rrtmgp_update!(mtile)

One radiative-transfer call for the whole tile: reconstruct each column's thermodynamic
state, build the level (face) values, fill the cloud-optics fields, hand the batch to the
solver and turn the returned net fluxes into the held `q_lw`/`q_sw` divergences.

The column loop is SERIAL and runs outside the model's own `Threads.@threads :static`
loop; the solver threads over columns internally through its own context. Nesting the two
would oversubscribe the machine and, worse, break the `threadid()` ownership rule that
`mtile.scratch_columns` depends on.

Cloud optics are the real microphysics mapping from S3a
([`cloud_optics_column!`](@ref)), reached through [`radiation_assemble!`](@ref), which is
the whole per-column half of this function and is separated from the solve so it can be
tested with no lookup artifacts.
"""
function radiation_rrtmgp_update!(mtile::ModelTile)

    rs = mtile.radiation
    sb = rs.solver::NamedTuple
    solver = sb.solver
    B = sb.batch::RadiationBatch

    radiation_assemble!(rs, mtile, B, sb.cloud_params::CloudOpticsParams)

    rrtmgp_solve_columns!(solver, B.T_lay, B.p_lay, B.T_lev, B.p_lev, B.vmr_h2o, B.o3,
                          (B.lwp, B.iwp, B.re_liq, B.re_ice, B.cf), rs.extension,
                          B.t_sfc, rs.cos_zenith, rs.toa_flux)
    lw_up, lw_dn, lw_net, sw_up, sw_dn, sw_net = rrtmgp_fluxes(solver)

    radiation_divergence!(rs.q_lw, lw_net, rs.dz, rs.nlay, rs.ncol, rs.kDim, rs.stride)
    radiation_divergence!(rs.q_sw, sw_net, rs.dz, rs.nlay, rs.ncol, rs.kDim, rs.stride)

    copyto!(rs.flux_lw_up, lw_up); copyto!(rs.flux_lw_dn, lw_dn)
    copyto!(rs.flux_lw_net, lw_net)
    copyto!(rs.flux_sw_up, sw_up); copyto!(rs.flux_sw_dn, sw_dn)
    copyto!(rs.flux_sw_net, sw_net)
    return nothing
end

"""
    radiation_surface_temperature(model, T_face_bottom) -> Float64

The radiative surface temperature, by precedence

    physical_params[:T_sfc]  >  physical_params[:SST]  >  the extrapolated T_face[1]

An explicit `:T_sfc` wins because it is unambiguous. `:SST` is next: on a configuration
built around air–sea disequilibrium (the TC) the surface the radiation sees must be the
same ocean the enthalpy fluxes see, or the two parameterizations describe different
lower boundaries. The fallback is the column's own extrapolated surface AIR temperature,
which is what an idealized case with no ocean (O01) wants: it is radiatively consistent
with the sounding it was initialized from and introduces no disequilibrium the run was
never given.
"""
function radiation_surface_temperature(model::ModelParameters, T_face_bottom::Float64)
    pp = model.physical_params
    haskey(pp, :T_sfc) && return pp[:T_sfc]
    haskey(pp, :SST) && return pp[:SST]
    return T_face_bottom
end

"""
    radiation_assemble!(rs, mtile)
    radiation_assemble!(rs, mtile, B::RadiationBatch, P::CloudOpticsParams)

Fill the whole tile's solver inputs from the model state: one pass over the columns, each
reconstructed by [`radiation_column_state!`](@ref), given faces by
[`radiation_levels!`](@ref) and clouds by [`cloud_optics_column!`](@ref), written into the
`(nlay, ncol)` matrices of `B` that [`rrtmgp_solve_columns!`](@ref) then consumes.

This is the ASSEMBLY half of [`radiation_rrtmgp_update!`](@ref), split out for two
reasons. It is the half that is pure Scythe -- prognostic slots in, batch matrices out,
no radiative transfer anywhere in it -- so it is the half that can be tested on a
`:prescribed` tile with a hand-built batch, with no lookup artifacts, no network and no
solver (`test/test_radiation_driver.jl`). And it is the half whose allocation behaviour is
Scythe's responsibility: it allocates NOTHING, while the solve underneath it allocates
hundreds of kilobytes per call spawning RRTMGP's own tasks, so the two numbers have to be
measurable apart.

The column loop is SERIAL, outside the model's `Threads.@threads :static` loop, which is
what makes the single per-tile `work`/`cloud`/condensate scratch safe.

Three counters are accumulated here, never reset: `n_neg_rho_v` (a negative vapour is
floored on the way into the solver and NOWHERE else -- in the model it is a resolution
diagnostic, reference/HANDOFF_CONDENSATE_REPRESENTATION.md) and the two effective-radius
clamp counts returned by [`cloud_optics_column!`](@ref), which are how a saturated cloud
optics table announces itself instead of silently flattening the size distribution.

`rs.o3` is empty on a `:prescribed` state (there is no radiative transfer to have an ozone
profile for), so the ozone column is written as zero there rather than indexing an empty
vector: the assembly must run on any active state, because that is what makes it testable
without a solver.
"""
radiation_assemble!(rs::RadiationState, mtile::ModelTile) =
    let sb = rs.solver::NamedTuple
        radiation_assemble!(rs, mtile, sb.batch::RadiationBatch,
                            sb.cloud_params::CloudOpticsParams)
    end

function radiation_assemble!(rs::RadiationState, mtile::ModelTile, B::RadiationBatch,
                             P::CloudOpticsParams)

    kDim = rs.kDim
    work = rs.work
    cloud = rs.cloud
    # `Union{Nothing,Matrix{Float64}}` is a pointer-only union, so this costs no box; the
    # ice-off path hands `cloud_optics_column!` its documented `nothing`.
    ice = B.ice_on ? B.ice : nothing
    have_o3 = !isempty(rs.o3)

    @inbounds for c in 1:rs.ncol
        colstart = (c - 1) * kDim + 1
        colend = colstart + kDim - 1
        radiation_column_state!(work, mtile, colstart, colend, B)
        radiation_levels!(work, rs.z, rs.z_face)
        n_liq, n_ice = cloud_optics_column!(cloud, P, work.rho_d, B.rho_c, B.rho_r,
                                            ice, rs.dz)
        rs.n_clamp_re_liq += n_liq
        rs.n_clamp_re_ice += n_ice
        for k in 1:rs.nlay
            B.T_lay[k, c] = work.Tk[k]
            B.p_lay[k, c] = work.p[k]
            # The model never alters a negative vapour (it is a resolution diagnostic,
            # see reference/HANDOFF_CONDENSATE_REPRESENTATION.md); only this INPUT is
            # floored, and the flooring is counted.
            rv = work.rho_v[k]
            rv < 0.0 && (rs.n_neg_rho_v += 1)
            B.vmr_h2o[k, c] = (max(rv, 0.0) / work.rho_d[k]) * (Rv / Rd)
            B.o3[k, c] = have_o3 ? rs.o3[k] : 0.0
            B.lwp[k, c] = cloud.lwp[k]
            B.iwp[k, c] = cloud.iwp[k]
            B.re_liq[k, c] = cloud.re_liq[k]
            B.re_ice[k, c] = cloud.re_ice[k]
            B.cf[k, c] = cloud.cf[k]
        end
        for k in 1:(rs.nlay + 1)
            B.T_lev[k, c] = work.T_face[k]
            B.p_lev[k, c] = work.p_face[k]
        end
        B.t_sfc[c] = radiation_surface_temperature(mtile.model, work.T_face[1])
    end
    return nothing
end

"""
    cloud_optics_clear!(cloud::CloudOpticsColumn)

The all-clear cloud-optics stub: zero water paths, zero effective radii, zero cloud
fraction. Used by S2a/S2b so the clear-sky path can ship before the microphysics mapping
(plan D5, S3) exists. Zeroing rather than leaving the buffer alone matters: the buffer is
shared by every column of the tile, so a stale cloud from column `c-1` would otherwise
appear in a clear column `c`.
"""
function cloud_optics_clear!(cloud::CloudOpticsColumn)
    fill!(cloud.lwp, 0.0); fill!(cloud.iwp, 0.0)
    fill!(cloud.re_liq, 0.0); fill!(cloud.re_ice, 0.0)
    fill!(cloud.cf, 0.0)
    return nothing
end

# ── Column reconstruction ─────────────────────────────────────────────────────

"""
    radiation_column_state!(work, mtile, colstart, colend, cond = nothing)

Reconstruct one column's thermodynamic state into `work` from the prognostic slots and
the reference profiles: `p`, `Tk`, `rho_d`, `rho_v`, `rho_liq`, `rho_ice`, plus the
kinetic energy `ke` and the retrieval argument `M`.

This mirrors `mc_driver!`'s own thermodynamic block term for term, and deliberately
reuses its helpers rather than restating them:

- totals are `perturbation + reference` (`p = p' + p̄` and so on), with the reference
  taken from the same `ref_*` views the driver reads;
- the vapour is the PROGNOSTIC slot plus the DERIVED reference `ρ̄_v` from
  `mtile.mc_ref_diag`, never Springsteel's independently fitted `ref_rho_v`, so the
  radiation column and the model column agree bit for bit at rest;
- the cloud comes through [`recover_rho_c`](@ref) and the rain and the three ice masses
  through [`recover_total`](@ref), so a run carrying control-variable transforms
  (`:bhyp`) is reconstructed, not read raw;
- `ke = ½(u² + v² + w²)` is INCLUDED. It is tempting to drop as a small term; at 90 m/s
  it is worth about 4 K in the retrieval, which is a 1 K/day error in the cooling profile;
- `M = p + E_t − ρ_t(ke + gz)` and `Tk = retrieve_temperature(M, ρ_d, ρ_t, ρ_liq, ρ_ice)`
  are the driver's closed-form retrieval.

The retrieved temperature is CLAMPED to [170, 350] K and every clamp is COUNTED in
`rs.n_clamp_tk`. A solver fed a temperature outside its lookup range either errors or
silently extrapolates; a clamp that is never counted is the second failure mode of that
pair, so the count is the point.

`work` is a single per-tile buffer, safe because the radiation update is serial (see
[`RadiationWork`](@ref)).

The optional `cond` argument (S3b) is the [`RadiationBatch`](@ref) whose per-column
condensate scratch (`rho_c`, `rho_r`, the `(nlay, 12)` ice-moment matrix) this fills on
the way past. Omitted -- as `radiation_prescribed!` and `radiation_trace!` omit it -- the
function is exactly what S2b measured: the argument's type is known at each call site, so
the `cond !== nothing` tests are compile-time constants and the ice-moment views are never
formed.
"""
function radiation_column_state!(work::RadiationWork, mtile::ModelTile,
                                 colstart::Int64, colend::Int64,
                                 cond::Union{Nothing,RadiationBatch} = nothing)

    model = mtile.model
    rs = mtile.radiation
    grid = mtile.tile
    phys = grid.physical
    vars = model.grid_params.vars
    refstate = mtile.ref_state
    slots = mtile.mc_slots
    n = colend - colstart + 1

    pbar = view(ref_pressure(refstate), :, 1)
    rho_dbar = view(ref_rho_d(refstate), :, 1)
    rho_tbar = view(ref_rho_t(refstate), :, 1)
    E_tbar = view(ref_total_energy(refstate), :, 1)
    rho_cbar = view(Springsteel.ref_rho_c(refstate), :, 1)
    rho_vbar = mtile.mc_ref_diag.rho_vbar

    # Slots 1-9 are hardcoded literals throughout the driver (see `MCSlots`); the vapour
    # and the ice masses are APPENDED and are resolved by name once, at tile creation.
    pp = view(phys, colstart:colend, 1, 1)
    rho_dp = view(phys, colstart:colend, 2, 1)
    rho_tp = view(phys, colstart:colend, 3, 1)
    u = view(phys, colstart:colend, 4, 1)
    w = view(phys, colstart:colend, 5, 1)
    E_tp = view(phys, colstart:colend, 6, 1)
    nu_r = view(phys, colstart:colend, 8, 1)
    nu_c = view(phys, colstart:colend, 9, 1)
    rho_vp = view(phys, colstart:colend, slots.rho_v, 1)
    has_v = haskey(vars, "v")
    v = has_v ? view(phys, colstart:colend, vars["v"], 1) : nothing

    ctrans = condensate_transform_mode(model.options)
    cmu = get(model.physical_params, :condensate_mu, 1.0e-7)
    rtrans = rain_transform_mode(model.options)
    rmu = get(model.physical_params, :rain_mu, 1.0e-7)
    ice_on = ice_registered(slots)
    itrans = ice_transform_mode(model.options)
    imu = ice_mu(model.physical_params, 1)
    # The ice views are formed UNCONDITIONALLY, against slot 1 (pressure, which every
    # configuration has) when ice is off, and READ only under `ice_on`. The obvious
    # `ice_on ? view(...) : nothing` costs an allocation per column per view: a
    # `Union{Nothing,SubArray}` is not a pointer-only union — `SubArray` is an immutable
    # struct, so the union has to box it — and the twelve ice views cost 2.3 kB of garbage
    # per column per radiation call that way (measured). Formed this way every view is
    # concretely typed and stack-allocated, and the whole assembly allocates exactly zero.
    s_i1q = ice_on ? slots.i1_q : 1
    s_i2q = ice_on ? slots.i2_q : 1
    s_i3q = ice_on ? slots.i3_q : 1
    i1q = view(phys, colstart:colend, s_i1q, 1)
    i2q = view(phys, colstart:colend, s_i2q, 1)
    i3q = view(phys, colstart:colend, s_i3q, 1)

    # ── The condensate hand-off (S3b) ──
    # `cond === nothing` is the S2a/S2b call: nothing below this line exists, the ternaries
    # fold away at compile time (the argument type is known at every call site) and the
    # reconstruction is the one S2b measured. With a batch, the SAME traversal that already
    # recovers `rho_c`, `rho_r` and the three ice masses for the temperature retrieval also
    # publishes them, so the cloud optics can never disagree with the thermodynamics about
    # what the condensate is. The nine remaining ice moments (number and the two spheroid
    # volumes) are read only here, and only through `recover_total` with the moment's OWN
    # width -- `mu` carries the units of what it transforms, so one width for all twelve
    # would be wrong by orders of magnitude on `n`.
    want_cond = cond !== nothing
    ice_cond = want_cond && ice_on
    imu_n = ice_cond ? ice_mu(model.physical_params, 2) : 0.0
    imu_a = ice_cond ? ice_mu(model.physical_params, 3) : 0.0
    imu_c = ice_cond ? ice_mu(model.physical_params, 4) : 0.0
    i1n = view(phys, colstart:colend, ice_cond ? slots.i1_n : 1, 1)
    i1a = view(phys, colstart:colend, ice_cond ? slots.i1_a : 1, 1)
    i1c = view(phys, colstart:colend, ice_cond ? slots.i1_c : 1, 1)
    i2n = view(phys, colstart:colend, ice_cond ? slots.i2_n : 1, 1)
    i2a = view(phys, colstart:colend, ice_cond ? slots.i2_a : 1, 1)
    i2c = view(phys, colstart:colend, ice_cond ? slots.i2_c : 1, 1)
    i3n = view(phys, colstart:colend, ice_cond ? slots.i3_n : 1, 1)
    i3a = view(phys, colstart:colend, ice_cond ? slots.i3_a : 1, 1)
    i3c = view(phys, colstart:colend, ice_cond ? slots.i3_c : 1, 1)

    z = view(mtile.tilepoints, colstart:colend, size(mtile.tilepoints, 2))

    @inbounds for k in 1:n
        p = pp[k] + pbar[k]
        rho_d = rho_dp[k] + rho_dbar[k]
        rho_t = rho_tp[k] + rho_tbar[k]
        E_t = E_tp[k] + E_tbar[k]
        rho_v = rho_vp[k] + rho_vbar[k]
        rho_c = recover_rho_c(nu_c[k], rho_cbar[k], ctrans, cmu)
        rho_r = recover_rho_r(nu_r[k], rtrans, rmu)
        # The three ice MASSES, kept as named values so the same recovery serves both the
        # temperature retrieval and (below) the cloud-optics hand-off. The association is
        # `(i1 + i2) + i3`, bit for bit what the single expression it replaces computed.
        ri1 = ice_on ? recover_total(i1q[k], itrans, imu) : 0.0
        ri2 = ice_on ? recover_total(i2q[k], itrans, imu) : 0.0
        ri3 = ice_on ? recover_total(i3q[k], itrans, imu) : 0.0
        rho_ice = ice_on ? ((ri1 + ri2) + ri3) : 0.0
        vk = has_v ? v[k] : 0.0
        ke = 0.5 * ((u[k] * u[k]) + (vk * vk) + (w[k] * w[k]))
        M = p + E_t - (rho_t * (ke + (gravity * z[k])))
        rho_liq = rho_c + rho_r
        Tk = retrieve_temperature(M, rho_d, rho_t, rho_liq, rho_ice)
        if Tk < 170.0
            Tk = 170.0; rs.n_clamp_tk += 1
        elseif Tk > 350.0
            Tk = 350.0; rs.n_clamp_tk += 1
        end

        work.p[k] = p
        work.Tk[k] = Tk
        work.rho_d[k] = rho_d
        work.rho_v[k] = rho_v
        work.rho_liq[k] = rho_liq
        work.rho_ice[k] = rho_ice
        work.ke[k] = ke
        work.M[k] = M

        if cond !== nothing
            cond.rho_c[k] = rho_c
            cond.rho_r[k] = rho_r
            if ice_cond
                ic = cond.ice
                ic[k, 1] = ri1
                ic[k, 2] = recover_total(i1n[k], itrans, imu_n)
                ic[k, 3] = recover_total(i1a[k], itrans, imu_a)
                ic[k, 4] = recover_total(i1c[k], itrans, imu_c)
                ic[k, 5] = ri2
                ic[k, 6] = recover_total(i2n[k], itrans, imu_n)
                ic[k, 7] = recover_total(i2a[k], itrans, imu_a)
                ic[k, 8] = recover_total(i2c[k], itrans, imu_c)
                ic[k, 9] = ri3
                ic[k, 10] = recover_total(i3n[k], itrans, imu_n)
                ic[k, 11] = recover_total(i3a[k], itrans, imu_a)
                ic[k, 12] = recover_total(i3c[k], itrans, imu_c)
            end
        end
    end
    return nothing
end

# ── Taper, anomaly reference, storage ─────────────────────────────────────────

"""
    radiation_store!(mtile)

Post-process the freshly computed `q_lw`/`q_sw` in place: apply the `z_max` taper, and,
under `options[:radiation_forcing] = :anomaly`, subtract the horizontal-mean profile.

**Taper first** (the user's decision, plan D3/D5). The sponge layer relaxes the state
toward the reference, so a radiative tendency up there is fighting the damping rather
than doing physics, and the extension blend at the model top is the least trustworthy
part of the column. `taper` is precomputed per LAYER and `q` is per GRIDPOINT, hence the
`div(k-1, stride)+1` map; at the default stride 1 it is the identity. `z_max = Inf` makes
the taper all ones, so this costs one multiply per gridpoint and never branches.

**Anomaly reference.** `:anomaly` holds `q − q̄(z)` where `q̄` is the horizontal mean over
this tile at the FIRST radiation call, captured here and never recomputed. On a
horizontally homogeneous initial state — the O01 sounding, exactly — that makes the
forcing identically zero at t = 0, which is the sharp integration test: the run must stay
bitwise identical to the radiation-off run until something breaks the homogeneity.

The "first call" test is `last_call_step == typemin(Int)`, which is why
[`radiation_update!`](@ref) updates `last_call_step` only after calling this.

On a NESTED run each patch captures its own mean, so patches with different cloud
populations hold different references. The plan flags that as an open item (the coarsest
patch's mean broadcast to all is the intended fix); for a single patch, and for the
homogeneous t = 0 state every patch shares, the two agree.
"""
function radiation_store!(mtile::ModelTile)

    rs = mtile.radiation
    kDim = rs.kDim; ncol = rs.ncol; stride = rs.stride; nlay = rs.nlay
    q_lw = rs.q_lw; q_sw = rs.q_sw; taper = rs.taper

    @inbounds for c in 1:ncol
        base = (c - 1) * kDim
        for k in 1:kDim
            wgt = taper[div(k - 1, stride) + 1]
            q_lw[base + k] *= wgt
            q_sw[base + k] *= wgt
        end
    end

    rs.forcing === :anomaly || return nothing

    if rs.last_call_step == typemin(Int) && ncol > 0
        @inbounds for lay in 1:nlay
            s_lw = 0.0; s_sw = 0.0; cnt = 0
            for c in 1:ncol
                base = (c - 1) * kDim
                for k in ((lay - 1) * stride + 1):(lay * stride)
                    s_lw += q_lw[base + k]
                    s_sw += q_sw[base + k]
                    cnt += 1
                end
            end
            rs.q_lw_ref[lay] = s_lw / cnt
            rs.q_sw_ref[lay] = s_sw / cnt
        end
    end

    @inbounds for c in 1:ncol
        base = (c - 1) * kDim
        for k in 1:kDim
            lay = div(k - 1, stride) + 1
            q_lw[base + k] -= rs.q_lw_ref[lay]
            q_sw[base + k] -= rs.q_sw_ref[lay]
        end
    end
    return nothing
end
