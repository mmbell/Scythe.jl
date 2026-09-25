# ── Radiation sidecar I/O (Stage S5) ───────────────────────────────────────────
#
# The radiation fluxes and heating rates live on the MISH grid (faces, and the
# stratospheric extension above the model top), neither of which is on the prognostic
# variable grid that `write_output` (src/io.jl) writes -- that is why radiation gets its
# own file rather than riding along in the analysis CSV/NetCDF. No regridding: every
# array here is written exactly on the columns and layers `mtile.radiation` computed on.
#
# This is the SECOND (and last) file in Scythe that names NCDatasets -- alongside
# src/radiation_rrtmgp.jl, which needs it only to trigger RRTMGP's NCDatasets extension
# (see the note at the top of that file) and never calls it directly. This file is the one
# that actually reads and writes files with it, so it is named fully-qualified throughout
# (`NCDatasets.NCDataset`, `NCDatasets.defVar`, ...) rather than splatted with `using`, for
# the same "narrower blast radius" reason radiation_rrtmgp.jl gives.
#
# Included in src/Scythe.jl immediately after radiation_rrtmgp.jl. `import`, not `using`,
# for the reason above.
import NCDatasets

# ── Naming ─────────────────────────────────────────────────────────────────────
#
# `<tag>_radiation_i<offset>.nc`, one file per TILE per output time, in `model.output_dir`
# (which is already the per-patch `nest<i>` directory on a nested run -- see
# `ModelParameters.output_dir` construction in `src/nesting.jl`). `tag` is exactly
# `write_output`'s tag (`string(round(t_model; digits=2))`) so the two families of files
# line up by name for a snapshot; `offset` is `mtile.tile.params.patchOffsetL`, the
# left-edge Fourier/gridpoint offset of this WORKER's tile within its patch (0 for the
# first/only tile, and the same quantity `run_nested_patch`'s collar injection already
# uses to identify a tile -- see the `patchOffsetL` grep in the S5 plan). A single-tile
# run (the common case: one worker per patch) still writes the `_i0` suffix; `read_radiation`
# hides it by globbing.

"The sidecar's own tag, so `radiation_write!`/`radiation_write_final!` and a caller that
wants to open a companion `<tag>_physical.csv` compute it identically."
_radiation_tag(t_model::Float64) = string(round(t_model; digits = 2))

# ── Writer ───────────────────────────────────────────────────────────────────────

"""
    radiation_write!(mtile, t)

Sidecar radiation output. No-op unless `rs.active && rs.output`.

The tile's held forcing is piecewise-constant between radiation calls and the pre-pass
runs BEFORE the column advance, so at pre-pass step `t` the tile's PHYSICAL state (and
therefore the column reconstruction this writes) is the state at `t_model = (t-1)*ts` --
one step behind the tag `write_output` gives the state that step `t`'s OWN advance
produces. Firing on `mod(t-1, out_int) == 0` (not `mod(t, out_int)`) is what lines the
sidecar tag up with the physical snapshot at the SAME model time: at `t = out_int + 1` the
pre-pass sees the state at `t_model = out_int*ts = output_interval`, which is exactly the
time step `t = out_int`'s own `write_output` call wrote under the tag `output_interval`.
The step-1 call (t_model = 0) matches `run_model`'s own initial `write_output(patch, model,
0.0)`, made once before the timestep loop starts.

One consequence: the LAST physical snapshot, written when `mod(t, out_int) == 0` at
`t = num_ts` (model time `integration_time`), has no periodic radiation counterpart --
that would need a pre-pass at step `num_ts + 1`, which never runs. See
[`radiation_write_final!`](@ref).
"""
function radiation_write!(mtile::ModelTile, t::Int64)
    rs = mtile.radiation
    (rs.active && rs.output) || return nothing
    model = mtile.model
    out_int = max(1, round(Int, model.output_interval / model.ts))
    mod(t - 1, out_int) == 0 || return nothing
    _radiation_write_snapshot!(mtile, (t - 1) * model.ts)
    return nothing
end

"""
    radiation_write_final!(mtile, t_end)

Write the sidecar for the FINAL output time, `t_end` (seconds; callers pass
`model.integration_time`), which [`radiation_write!`](@ref)'s periodic cadence never
reaches (see its docstring). Called once, after the time loop, from the same two sites
`finalize_model`/`model_loop` (src/semiimplicit.jl) and `run_nested_patch`
(src/nesting.jl) invoke the final `write_output`/`write_restart`. No-op unless
`rs.active && rs.output`; a run with radiation off (or `radiation_output = false`) pays
nothing extra at finalize.

Uses the CURRENTLY HELD `q_lw`/`q_sw` -- the last radiation call's profile, exactly what
every column has been folding into `QDOT_TH` since -- reconstructed against the tile's
state AT `t_end` (the state the final `advanceTimestep` left it in), so the sidecar's
`dT_lw`/`dT_sw` and the flux/cloud diagnostics reflect the actual final column, not a
stale one.
"""
function radiation_write_final!(mtile::ModelTile, t_end::Float64)
    rs = mtile.radiation
    (rs.active && rs.output) || return nothing
    _radiation_write_snapshot!(mtile, t_end)
    return nothing
end

function _radiation_write_snapshot!(mtile::ModelTile, t_model::Float64)
    model = mtile.model
    isdir(model.output_dir) || mkpath(model.output_dir)
    tag = _radiation_tag(t_model)
    offset = mtile.tile.params.patchOffsetL
    path = joinpath(model.output_dir, "$(tag)_radiation_i$(offset).nc")
    _radiation_write_file!(path, mtile, t_model)
    return nothing
end

"The radiation counter attributes [`assemble_physics`](@ref) must SUM across tiles rather
than take from the first one."
const RADIATION_COUNTER_ATTRS = ("n_clamp_tk", "n_clamp_re_liq", "n_clamp_re_ice",
                                 "n_neg_rho_v")

"""
    RADIATION_FIELDS_2D

The `(name, units, long_name)` of every layer-resolved radiation field, in the order the
sidecar writes them. [`radiation_diagnostics`](@ref) returns each under `Symbol(name)`.

ONE table, read by BOTH the sidecar and [`write_netcdf_comprehensive`](@ref); the latter
also writes the derived `dT_net = dT_lw + sw_scale * dT_sw`, the rate a column actually
felt, which the sidecar leaves to the reader because it holds `sw_scale` as an attribute.
"""
const RADIATION_FIELDS_2D = (
    ("q_lw", "W m-3", "longwave heating rate (flux divergence)"),
    ("q_sw", "W m-3", "shortwave heating rate (flux divergence), held profile"),
    ("q_sw_applied", "W m-3",
     "sw_scale * q_sw, the rate actually folded into QDOT_TH this step"),
    ("dT_lw", "K day-1", RADIATION_KDAY_LABEL * " -- longwave"),
    ("dT_sw", "K day-1",
     RADIATION_KDAY_LABEL * " -- shortwave, UNSCALED (multiply by sw_scale " *
     "for the rate actually applied)"))

"""
    RADIATION_FIELDS_FACE

The `(name, units, long_name)` of the six FACE-based flux profiles, on `zf` (which runs
to the top of the stratospheric extension).

Sidecar ONLY. They are deliberately NOT merged into the comprehensive file: `zf` has no
counterpart on the regular output z grid, which stops at the model top. The per-column
boundary values derived from them ([`RADIATION_FIELDS_1D`](@ref): `olr`, `sw_sfc_dn`,
...) ARE in both files, so nothing a column-integrated budget needs is lost — only the
profiles themselves, which stay here.
"""
const RADIATION_FIELDS_FACE = (
    ("flux_lw_up", "W m-2", "longwave flux, upward"),
    ("flux_lw_dn", "W m-2", "longwave flux, downward"),
    ("flux_lw_net", "W m-2", "longwave net flux (up - down)"),
    ("flux_sw_up", "W m-2", "shortwave flux, upward"),
    ("flux_sw_dn", "W m-2", "shortwave flux, downward"),
    ("flux_sw_net", "W m-2", "shortwave net flux (up - down)"))

"""
    RADIATION_FIELDS_1D

The `(name, units, long_name)` of every per-column radiation diagnostic, in sidecar
order. Written under these exact names in BOTH files.
"""
const RADIATION_FIELDS_1D = (
    ("olr", "W m-2",
     "outgoing longwave radiation at the top of the full column (incl. extension)"),
    ("olr_model_top", "W m-2",
     "longwave flux up at the model top face, before the stratospheric extension"),
    ("lw_sfc_dn", "W m-2", "longwave flux down at the surface"),
    ("lw_sfc_up", "W m-2", "longwave flux up at the surface"),
    ("sw_sfc_dn", "W m-2", "shortwave flux down at the surface"),
    ("sw_sfc_up", "W m-2", "shortwave flux up at the surface"),
    ("sw_toa_dn", "W m-2", "shortwave flux down at the top of the full column"),
    ("sw_toa_up", "W m-2", "shortwave flux up at the top of the full column"),
    ("lwp", "g m-2", "column liquid water path"),
    ("iwp", "g m-2", "column ice water path"),
    ("cloudy", "1", "1 if any layer of this column has cf == 1, else 0"))

"""
    radiation_global_attrs(rs::RadiationState, ts::Float64) -> Vector{Pair{String,Any}}

The radiation run configuration, sun state and clamp census as NetCDF global attributes,
in ONE place. `ts` is the model timestep, which the last-call time and the call interval
are expressed in.

Consumed by BOTH writers: the sidecar writes these names verbatim, the comprehensive file
writes each under a `radiation_` prefix (names that already carry it — `radiation_interval_s`
— are left alone), which is how `tc/tc_postprocess.jl` named them when it merged the
sidecar. See [`mynn_global_attrs`](@ref), which does the same for MYNN.
"""
function radiation_global_attrs(rs::RadiationState, ts::Float64)
    return Pair{String,Any}[
        "t_last_call" => rs.last_call_step == typemin(Int) ? NaN :
                         (rs.last_call_step - 1) * ts,
        "cos_zenith" => rs.cos_zenith,
        "toa_flux" => rs.toa_flux,
        "sw_scale" => rs.sw_scale,
        "scheme" => string(rs.scheme),
        "method" => string(rs.method),
        "solar" => string(rs.solar),
        "forcing" => string(rs.forcing),
        "z_max" => isinf(rs.z_max) ? 1.0e30 : rs.z_max,
        "radiation_interval_s" => rs.interval_steps * ts,
        "n_clamp_tk" => rs.n_clamp_tk,
        "n_clamp_re_liq" => rs.n_clamp_re_liq,
        "n_clamp_re_ice" => rs.n_clamp_re_ice,
        "n_neg_rho_v" => rs.n_neg_rho_v]
end

"""
    radiation_diagnostics(mtile::ModelTile) -> NamedTuple

The tile's radiative forcing, fluxes and cloud diagnostics on the RADIATION MISH (Gauss
points in `x`, layer centres in `z`, faces in `zf`), as plain arrays: `x`, `z`, `zf`,
every layer field of [`RADIATION_FIELDS_2D`](@ref) as an `(ncol, nlay)` matrix, every
face field of [`RADIATION_FIELDS_FACE`](@ref) as an `(ncol, nlev_tot)` matrix, every
per-column field of [`RADIATION_FIELDS_1D`](@ref) as a length-`ncol` vector, the
`sw_scale` the applied rate is built with, `forcing` and (on `:anomaly`) the two
reference heating profiles, plus `attrs` ([`radiation_global_attrs`](@ref)) and
`sum_attrs` ([`RADIATION_COUNTER_ATTRS`](@ref)).

The ONE reader of `mtile.radiation`'s held state for output: the sidecar
([`_radiation_write_file!`](@ref)) writes it unchanged, the comprehensive writer regrids
the layer and per-column parts onto the regular output grid and drops the faces (see
[`RADIATION_FIELDS_FACE`](@ref)).

`dT_lw`/`dT_sw` are the K/day-at-constant-pressure rates, from the SAME `_rad_kday` the
trace uses. The column reconstruction is re-run (as the trace's is) and the
temperature-clamp counter is snapshotted and restored around it for the same reason: a
diagnostic must not inflate the census it reports.

Requires `options[:radiation_layer_stride] == 1` (`rs.kDim == rs.nlay`): the held
`q_lw`/`q_sw` are GRIDPOINT-indexed while the layer geometry (`rs.z`, the taper, the
`:anomaly` reference) is LAYER-indexed, and the two coincide only at stride 1 -- the only
stride `:rrtmgp` runs anyway (`mc_radiation_state` refuses anything else). A `:prescribed`
run at stride > 1 is the one configuration this could in principle serve and does not;
that combination is untested elsewhere in the radiation stack and errors here rather than
silently mis-shaping the `z` dimension.

ALLOCATES (reshapes, transposes, the K/day matrices) on every call. That is a deliberate
S5 choice, not an oversight: this fires at the OUTPUT cadence (60-300 s of model time),
three to four orders of magnitude coarser than the acoustic timestep the rest of the
driver is allocation-disciplined for.
"""
function radiation_diagnostics(mtile::ModelTile)
    rs = mtile.radiation
    kDim = rs.kDim
    nlay = rs.nlay
    ncol = rs.ncol
    kDim == nlay || error(
        "radiation output needs options[:radiation_layer_stride] = 1 " *
        "(kDim = $kDim, nlay = $nlay); the held q_lw/q_sw are gridpoint-indexed and " *
        "only coincide with the layer-indexed geometry at stride 1")

    # ── Coordinates ──
    x = Vector{Float64}(undef, ncol)
    @inbounds for c in 1:ncol
        x[c] = mtile.tilepoints[(c - 1) * kDim + 1, 1]
    end
    zf = rs.extension.nlay > 0 ? vcat(rs.z_face, rs.extension.z_face[2:end]) :
                                 copy(rs.z_face)
    nlev_tot = length(zf)
    size(rs.flux_lw_up, 1) == nlev_tot || error(
        "radiation output: flux matrix has $(size(rs.flux_lw_up, 1)) rows, expected " *
        "nlev_tot = $nlev_tot from z_face/extension -- geometry mismatch")

    # ── Held forcing, reshaped (x, z) ──
    q_lw = permutedims(reshape(rs.q_lw, kDim, ncol))
    q_sw = permutedims(reshape(rs.q_sw, kDim, ncol))
    q_sw_applied = rs.sw_scale .* q_sw

    # ── K/day at constant pressure (see the docstring) ──
    dT_lw = zeros(Float64, ncol, nlay)
    dT_sw = zeros(Float64, ncol, nlay)
    n_tk_save = rs.n_clamp_tk
    work = rs.work
    @inbounds for c in 1:ncol
        cs = (c - 1) * kDim + 1
        radiation_column_state!(work, mtile, cs, cs + kDim - 1)
        for k in 1:nlay
            f = _rad_kday(work, k)
            dT_lw[c, k] = rs.q_lw[cs + k - 1] * f
            dT_sw[c, k] = rs.q_sw[cs + k - 1] * f
        end
    end
    rs.n_clamp_tk = n_tk_save

    # ── Fluxes, reshaped (x, zf) ──
    flux_lw_up = permutedims(rs.flux_lw_up); flux_lw_dn = permutedims(rs.flux_lw_dn)
    flux_lw_net = permutedims(rs.flux_lw_net)
    flux_sw_up = permutedims(rs.flux_sw_up); flux_sw_dn = permutedims(rs.flux_sw_dn)
    flux_sw_net = permutedims(rs.flux_sw_net)

    # ── Per-column boundary diagnostics. Row 1 of every flux matrix is the surface, row
    # nlay+1 the model-top face, row end the top of the full column (incl. extension) --
    # the same rows `radiation_trace!`'s OLR/SW lines read. On `:prescribed` the flux
    # matrices are identically zero (no radiative transfer computed them), and these
    # columns are zero too -- which is the honest answer, not a missing feature. ──
    olr = rs.flux_lw_up[end, :]
    olr_model_top = rs.flux_lw_up[nlay + 1, :]
    lw_sfc_dn = rs.flux_lw_dn[1, :]; lw_sfc_up = rs.flux_lw_up[1, :]
    sw_sfc_dn = rs.flux_sw_dn[1, :]; sw_sfc_up = rs.flux_sw_up[1, :]
    sw_toa_dn = rs.flux_sw_dn[end, :]; sw_toa_up = rs.flux_sw_up[end, :]

    # ── Cloud column sums, from the batch (`:rrtmgp` only -- `:prescribed` has none and
    # the columns stay zero, matching `radiation_trace!`'s "no cloud diagnostic" branch). ──
    lwp = zeros(Float64, ncol); iwp = zeros(Float64, ncol); cloudy = zeros(Float64, ncol)
    B_ = rs.solver isa NamedTuple ? (rs.solver::NamedTuple).batch::RadiationBatch : nothing
    if B_ !== nothing
        @inbounds for c in 1:ncol
            lc = 0.0; ic = 0.0; anycf = false
            for k in 1:nlay
                lc += B_.lwp[k, c]; ic += B_.iwp[k, c]
                B_.cf[k, c] == 1.0 && (anycf = true)
            end
            lwp[c] = lc; iwp[c] = ic; cloudy[c] = anycf ? 1.0 : 0.0
        end
    end

    return (; x, z = copy(rs.z), zf,
              q_lw, q_sw, q_sw_applied, dT_lw, dT_sw,
              flux_lw_up, flux_lw_dn, flux_lw_net,
              flux_sw_up, flux_sw_dn, flux_sw_net,
              olr, olr_model_top, lw_sfc_dn, lw_sfc_up,
              sw_sfc_dn, sw_sfc_up, sw_toa_dn, sw_toa_up, lwp, iwp, cloudy,
              sw_scale = rs.sw_scale, forcing = rs.forcing,
              q_lw_ref = copy(rs.q_lw_ref), q_sw_ref = copy(rs.q_sw_ref),
              attrs = Dict{String,Any}(radiation_global_attrs(rs, mtile.model.ts)),
              sum_attrs = collect(RADIATION_COUNTER_ATTRS))
end

"""
    _radiation_write_file!(path, mtile, t_model)

Write one tile's sidecar to `path` from [`radiation_diagnostics`](@ref). Internal (not
part of the S5 API); `radiation_write!` and `radiation_write_final!` are the entry
points, and tests reach this directly (with a hand-chosen `path`) to write two "tiles" at
different offsets without a real multi-worker run -- see test/test_radiation_io.jl.

Every array written here comes from `radiation_diagnostics`, the SAME extraction the
comprehensive `<t>.nc` reads (src/netcdf_output.jl), so this file and that one can never
disagree about what a radiation field is. The face-based flux profiles and the `:anomaly`
reference columns are written here ONLY (see [`RADIATION_FIELDS_FACE`](@ref)).
"""
function _radiation_write_file!(path::String, mtile::ModelTile, t_model::Float64)
    rs = mtile.radiation
    d = radiation_diagnostics(mtile)

    NCDatasets.NCDataset(path, "c") do ds
        ds.attrib["Conventions"] = "CF-1.12"
        ds.attrib["source"] = "Scythe radiation sidecar"
        # The resolved configuration, sun state and clamp census, from the SAME function
        # the comprehensive writer reads (N2): one list, two files.
        for (k, v) in radiation_global_attrs(rs, mtile.model.ts)
            ds.attrib[k] = v
        end
        ds.attrib["patch_offset_l"] = mtile.tile.params.patchOffsetL

        NCDatasets.defDim(ds, "time", 1)
        NCDatasets.defDim(ds, "x", rs.ncol)
        NCDatasets.defDim(ds, "z", rs.nlay)
        NCDatasets.defDim(ds, "zf", length(d.zf))

        tv = NCDatasets.defVar(ds, "time", Float64, ("time",))
        tv.attrib["units"] = "seconds"; tv.attrib["long_name"] = "simulation time"
        tv[1] = t_model

        xv = NCDatasets.defVar(ds, "x", Float64, ("x",))
        xv.attrib["units"] = "m"
        xv.attrib["long_name"] = "tile horizontal mish coordinate"
        xv[:] = d.x

        zv = NCDatasets.defVar(ds, "z", Float64, ("z",))
        zv.attrib["units"] = "m"; zv.attrib["long_name"] = "radiation layer height"
        zv[:] = d.z

        zfv = NCDatasets.defVar(ds, "zf", Float64, ("zf",))
        zfv.attrib["units"] = "m"
        zfv.attrib["long_name"] =
            "radiation layer face height (incl. stratospheric extension)"
        zfv[:] = d.zf

        wrz(name, data, units, long) = begin
            dv = NCDatasets.defVar(ds, name, Float64, ("time", "x", "z");
                                   deflatelevel = 4, fillvalue = NaN)
            dv.attrib["units"] = units; dv.attrib["long_name"] = long
            dv[1, :, :] = data
        end
        wrzf(name, data, units, long) = begin
            dv = NCDatasets.defVar(ds, name, Float64, ("time", "x", "zf");
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

        for (nm, units, long) in RADIATION_FIELDS_2D
            wrz(nm, getfield(d, Symbol(nm)), units, long)
        end
        for (nm, units, long) in RADIATION_FIELDS_FACE
            wrzf(nm, getfield(d, Symbol(nm)), units, long)
        end
        for (nm, units, long) in RADIATION_FIELDS_1D
            wrx(nm, getfield(d, Symbol(nm)), units, long)
        end

        if rs.forcing === :anomaly
            qlr = NCDatasets.defVar(ds, "q_lw_ref", Float64, ("time", "z");
                                    deflatelevel = 4)
            qlr.attrib["units"] = "W m-3"
            qlr.attrib["long_name"] =
                "resting reference column longwave heating (subtracted for :anomaly forcing)"
            qlr[1, :] = d.q_lw_ref
            qsr = NCDatasets.defVar(ds, "q_sw_ref", Float64, ("time", "z");
                                    deflatelevel = 4)
            qsr.attrib["units"] = "W m-3"
            qsr.attrib["long_name"] =
                "resting reference column shortwave heating (subtracted for :anomaly forcing)"
            qsr[1, :] = d.q_sw_ref
        end
    end
    return nothing
end


# ── Reader ───────────────────────────────────────────────────────────────────────

"""
    radiation_snapshots(dir) -> Vector{String}

Every output tag `dir` holds a radiation sidecar for (at least one
`<tag>_radiation_i*.nc` file), sorted by model time. Empty when the run carries no
sidecars (radiation off, or `:radiation_output = false`) -- callers (e.g. the O01
radiation diagnostics) use that to decide whether to add radiation rows at all.
"""
function radiation_snapshots(dir::AbstractString)
    isdir(dir) || return String[]
    tags = String[]
    for f in readdir(dir)
        m = match(r"^(.*)_radiation_i[0-9]+\.nc$", f)
        m === nothing && continue
        (m.captures[1] in tags) || push!(tags, m.captures[1])
    end
    return sort(tags; by = s -> parse(Float64, s))
end

_rd_squeeze(ds, name) = dropdims(coalesce.(Array(ds[name]), NaN); dims = 1)

"""
    read_radiation(dir, tag) -> NamedTuple

Read every tile's sidecar for output tag `tag` (`write_output`'s tag string, e.g.
`"300.0"`) in `dir` and concatenate them along `x` in ascending `patchOffsetL` order,
reconstructing one domain-wide snapshot on the mish grid -- the inverse of what
`radiation_write!` writes per tile. Errors if no file matches.

Returns a `NamedTuple` with the coordinates (`t`, `x`, `z`, `zf`), every `(x, z)`/`(x, zf)`
field as a plain `Matrix{Float64}` and every per-column field as a `Vector{Float64}`
(field names match the NetCDF variable names), the `:anomaly` reference profiles
`q_lw_ref`/`q_sw_ref` (empty vectors under `:full` forcing), and the scalar attributes
(`cos_zenith`, `toa_flux`, `sw_scale`, `scheme`, `method`, `solar`, `forcing`, `z_max`,
`radiation_interval_s`) read off the FIRST tile (they are, by construction, the same on
every tile of a run) plus the four clamp/saturation counters SUMMED across tiles (the
domain-wide census, matching what a multi-tile `radiation_trace!` would report if it
pooled tiles).
"""
function read_radiation(dir::AbstractString, tag::AbstractString)
    pat = Regex("^" * Base.escape_string(String(tag)) * raw"_radiation_i([0-9]+)\.nc$")
    files = filter(f -> occursin(pat, f), readdir(dir))
    isempty(files) && error(
        "read_radiation: no sidecar files matching $(tag)_radiation_i*.nc in $dir")
    offs = [parse(Int, match(pat, f).captures[1]) for f in files]
    files = files[sortperm(offs)]

    snaps = map(files) do f
        NCDatasets.NCDataset(joinpath(dir, f), "r") do ds
            (
                x = coalesce.(Array(ds["x"]), NaN)::Vector{Float64},   # no time dim
                q_lw = _rd_squeeze(ds, "q_lw"), q_sw = _rd_squeeze(ds, "q_sw"),
                q_sw_applied = _rd_squeeze(ds, "q_sw_applied"),
                dT_lw = _rd_squeeze(ds, "dT_lw"), dT_sw = _rd_squeeze(ds, "dT_sw"),
                flux_lw_up = _rd_squeeze(ds, "flux_lw_up"),
                flux_lw_dn = _rd_squeeze(ds, "flux_lw_dn"),
                flux_lw_net = _rd_squeeze(ds, "flux_lw_net"),
                flux_sw_up = _rd_squeeze(ds, "flux_sw_up"),
                flux_sw_dn = _rd_squeeze(ds, "flux_sw_dn"),
                flux_sw_net = _rd_squeeze(ds, "flux_sw_net"),
                olr = _rd_squeeze(ds, "olr"), olr_model_top = _rd_squeeze(ds, "olr_model_top"),
                lw_sfc_dn = _rd_squeeze(ds, "lw_sfc_dn"),
                lw_sfc_up = _rd_squeeze(ds, "lw_sfc_up"),
                sw_sfc_dn = _rd_squeeze(ds, "sw_sfc_dn"),
                sw_sfc_up = _rd_squeeze(ds, "sw_sfc_up"),
                sw_toa_dn = _rd_squeeze(ds, "sw_toa_dn"),
                sw_toa_up = _rd_squeeze(ds, "sw_toa_up"),
                lwp = _rd_squeeze(ds, "lwp"), iwp = _rd_squeeze(ds, "iwp"),
                cloudy = _rd_squeeze(ds, "cloudy"),
                z = Array(ds["z"])::Vector{Float64}, zf = Array(ds["zf"])::Vector{Float64},
                t = Float64(Array(ds["time"])[1]),
                q_lw_ref = haskey(ds, "q_lw_ref") ?
                    Array(ds["q_lw_ref"])[1, :]::Vector{Float64} : Float64[],
                q_sw_ref = haskey(ds, "q_sw_ref") ?
                    Array(ds["q_sw_ref"])[1, :]::Vector{Float64} : Float64[],
                attrs = Dict{String,Any}(k => ds.attrib[k] for k in keys(ds.attrib)),
            )
        end
    end

    x = vcat((s.x for s in snaps)...)
    vcat2(field) = vcat((getfield(s, field) for s in snaps)...)
    a1 = snaps[1]
    return (;
        t = a1.t, x, z = a1.z, zf = a1.zf,
        q_lw = vcat2(:q_lw), q_sw = vcat2(:q_sw), q_sw_applied = vcat2(:q_sw_applied),
        dT_lw = vcat2(:dT_lw), dT_sw = vcat2(:dT_sw),
        flux_lw_up = vcat2(:flux_lw_up), flux_lw_dn = vcat2(:flux_lw_dn),
        flux_lw_net = vcat2(:flux_lw_net),
        flux_sw_up = vcat2(:flux_sw_up), flux_sw_dn = vcat2(:flux_sw_dn),
        flux_sw_net = vcat2(:flux_sw_net),
        olr = vcat2(:olr), olr_model_top = vcat2(:olr_model_top),
        lw_sfc_dn = vcat2(:lw_sfc_dn), lw_sfc_up = vcat2(:lw_sfc_up),
        sw_sfc_dn = vcat2(:sw_sfc_dn), sw_sfc_up = vcat2(:sw_sfc_up),
        sw_toa_dn = vcat2(:sw_toa_dn), sw_toa_up = vcat2(:sw_toa_up),
        lwp = vcat2(:lwp), iwp = vcat2(:iwp), cloudy = vcat2(:cloudy),
        q_lw_ref = a1.q_lw_ref, q_sw_ref = a1.q_sw_ref,
        cos_zenith = get(a1.attrs, "cos_zenith", NaN),
        toa_flux = get(a1.attrs, "toa_flux", NaN),
        sw_scale = get(a1.attrs, "sw_scale", NaN),
        scheme = get(a1.attrs, "scheme", ""),
        method = get(a1.attrs, "method", ""),
        solar = get(a1.attrs, "solar", ""),
        forcing = get(a1.attrs, "forcing", ""),
        z_max = get(a1.attrs, "z_max", NaN),
        radiation_interval_s = get(a1.attrs, "radiation_interval_s", NaN),
        n_clamp_tk = sum(Int(get(s.attrs, "n_clamp_tk", 0)) for s in snaps),
        n_clamp_re_liq = sum(Int(get(s.attrs, "n_clamp_re_liq", 0)) for s in snaps),
        n_clamp_re_ice = sum(Int(get(s.attrs, "n_clamp_re_ice", 0)) for s in snaps),
        n_neg_rho_v = sum(Int(get(s.attrs, "n_neg_rho_v", 0)) for s in snaps),
    )
end
