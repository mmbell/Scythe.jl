# ── MYNN-EDMF sidecar I/O (Stage S9) ────────────────────────────────────────────
#
# The exact `radiation_io.jl` pattern (read that file first), applied to the MYNN-EDMF
# closure's held column state: a periodic `<tag>_mynn_i<offset>.nc` per tile, on the same
# output cadence and the same tag convention (`write_output`'s `round(t_model; digits=2)`),
# reassembled across tiles by `read_mynn` in ascending `patchOffsetL` order.
#
# Unlike radiation, MYNN's held fields (`el`, `sm`, `sh`, `K_m`, `K_h`, ...) are ALREADY
# gridpoint-indexed in the `expdot` layout (`MYNNState`, src/mynn_state.jl) -- there is no
# layer/gridpoint stride mismatch to guard against, and no reconstruction of a whole
# column's thermodynamics: everything here is either a direct field of `mtile.mynn` or (for
# `e = rho_e/rho_t`) one division against the tile's own physical slots.
#
# Included in src/Scythe.jl immediately after radiation_io.jl. `import`, not `using`, for
# the same "narrower blast radius" reason that file gives.
import NCDatasets

# ── Naming ─────────────────────────────────────────────────────────────────────
#
# `<tag>_mynn_i<offset>.nc`, one file per TILE per output time, in `model.output_dir`.
# `tag` is `write_output`'s tag; `offset` is `mtile.tile.params.patchOffsetL`. See
# `radiation_io.jl`'s "Naming" section for the full rationale -- it applies verbatim.

"The sidecar's own tag; see `_radiation_tag`, which this mirrors exactly."
_mynn_tag(t_model::Float64) = string(round(t_model; digits = 2))

# ── Writer ───────────────────────────────────────────────────────────────────────

"""
    mynn_write!(mtile, t)

Sidecar MYNN-EDMF output. No-op unless `MY.active && MY.output`.

Fires at the SAME point in the timestep and on the SAME cadence rule as
[`radiation_write!`](@ref) (`mod(t-1, out_int) == 0`, `out_int` from
`model.output_interval / model.ts`): called from `semiimplicit.jl`/`nesting.jl` right
after `radiation_prepass!(mtile, t)`, i.e. BEFORE this step's column loop advances the
tile. At that point `mtile.mynn`'s held fields are exactly what the PREVIOUS step's
column loop last wrote (the closure updates on its own cadence and holds between calls,
same as radiation's `q_lw`/`q_sw`), so the snapshot tagged `t_model = (t-1)*ts` lines up
with the physical snapshot `write_output` wrote for that same model time -- the identical
alignment argument `radiation_write!`'s docstring gives, and for the identical reason
(both are read BEFORE the state they tag has been advanced again).
"""
function mynn_write!(mtile::ModelTile, t::Int64)
    MY = mtile.mynn
    (MY.active && MY.output) || return nothing
    model = mtile.model
    out_int = max(1, round(Int, model.output_interval / model.ts))
    mod(t - 1, out_int) == 0 || return nothing
    _mynn_write_snapshot!(mtile, (t - 1) * model.ts)
    return nothing
end

"""
    mynn_write_final!(mtile)

Run-end hook: folds the per-column counters into the tile scalars, prints the census line
(`mynn_census_line`, src/mc_mynn_bl.jl) and -- S9 -- writes the FINAL sidecar snapshot, at
`model.integration_time`, which `mynn_write!`'s periodic cadence never reaches on its own
(the same gap `radiation_write_final!` closes for radiation, at the same call sites:
`finalize_model`/`model_loop` in src/semiimplicit.jl and `run_nested_patch` in
src/nesting.jl, both of which already call `Scythe.mynn_write_final!(mtile)` with no
further argument -- unlike radiation's `t_end` this needs none, because
`model.integration_time` IS the run's own final time).

Uses the CURRENTLY HELD closure state (the last update's `el`/`K_m`/... and the last
column advance's per-gridpoint budget terms), exactly what every column has been folding
into its tendencies since -- the `radiation_write_final!` argument, restated for MYNN.
"""
function mynn_write_final!(mtile::ModelTile)
    MY = mtile.mynn
    MY.active || return nothing
    MY.n_clamp_e = sum(MY.n_clamp_col)
    MY.n_cap_K = sum(MY.n_capK_col)
    MY.n_diffnum = sum(MY.n_diffnum_col)
    MY.n_gate = sum(MY.n_gate_col)
    MY.n_stall = sum(MY.n_stall_col)
    MY.n_plume = sum(MY.n_plume_col)
    MY.trace && println(mynn_census_line(MY))
    MY.output && _mynn_write_snapshot!(mtile, mtile.model.integration_time)
    return nothing
end

function _mynn_write_snapshot!(mtile::ModelTile, t_model::Float64)
    model = mtile.model
    isdir(model.output_dir) || mkpath(model.output_dir)
    tag = _mynn_tag(t_model)
    offset = mtile.tile.params.patchOffsetL
    path = joinpath(model.output_dir, "$(tag)_mynn_i$(offset).nc")
    _mynn_write_file!(path, mtile, t_model)
    return nothing
end

"""
    _mynn_write_file!(path, mtile, t_model)

Write one tile's sidecar to `path`. Internal (not part of the S9 API); `mynn_write!` and
`mynn_write_final!` are the entry points, and tests reach this directly (with a
hand-chosen `path`) to write two "tiles" at different offsets -- see test/test_mynn_io.jl.

ALLOCATION: this allocates (a fresh `NCDataset`, the reshapes below, the `e` column) on
every call, the same deliberate S5 choice `_radiation_write_file!` documents: the sidecar
fires at the OUTPUT cadence, orders of magnitude coarser than the column driver's
allocation-disciplined hot path.
"""
function _mynn_write_file!(path::String, mtile::ModelTile, t_model::Float64)
    MY = mtile.mynn
    ncol = MY.ncol
    kDim = MY.kDim
    npoints = ncol * kDim

    # ── e = rho_e/rho_t, from the tile's own physical slot ──
    # `rho_e` (the appended TKE-density total) is `mtile.mc_slots.rho_e`; `rho_t` is the
    # SAME reconstruction `mc_driver!` uses every column (`rho_tp + rho_tbar`): slot 3
    # ("rho_t" is the third entry of both `MC_VARS` and `MC_VARS_CYL` -- see `MCSlots`)
    # plus the 1-D reference profile, which (like every reference profile) is shared by
    # every horizontal column and indexed by the WITHIN-COLUMN layer `k`, not the flat
    # gridpoint `j`.
    phys = mtile.tile.physical
    rho_e_slot = mtile.mc_slots.rho_e
    rho_tbar = view(ref_rho_t(mtile.ref_state), :, 1)
    rho_tp = view(phys, :, 3, 1)
    rho_e = view(phys, :, rho_e_slot, 1)
    e = Vector{Float64}(undef, npoints)
    @inbounds for c in 1:ncol, k in 1:kDim
        j = (c - 1) * kDim + k
        e[j] = rho_e[j] / (rho_tp[j] + rho_tbar[k])
    end

    # ── Coordinates: the exact radiation_io.jl convention (used for every MC geometry,
    # cylindrical included -- the tile's own `x` regardless of what physical coordinate
    # it represents). ──
    x = Vector{Float64}(undef, ncol)
    @inbounds for c in 1:ncol
        x[c] = mtile.tilepoints[(c - 1) * kDim + 1, 1]
    end

    reshape2(v) = permutedims(reshape(v, kDim, ncol))    # gridpoint vector -> (x, z)
    zerorc() = zeros(Float64, ncol, kDim)
    mx(v) = isempty(v) ? 0.0 : maximum(v)

    NCDatasets.NCDataset(path, "c") do ds
        ds.attrib["Conventions"] = "CF-1.12"
        ds.attrib["source"] = "Scythe MYNN-EDMF sidecar"
        ds.attrib["closure"] = MY.closure
        ds.attrib["edmf"] = MY.edmf
        # NCDatasets has no NetCDF attribute type for `Bool` -- store as 0/1 `Int`, the
        # same convention `cloudy` (radiation_io.jl) uses for a per-column flag.
        ds.attrib["edmf_mom"] = Int(MY.edmf_mom)
        ds.attrib["scale_aware"] = Int(MY.scale_aware)
        ds.attrib["init_mode"] = string(MY.init_mode)
        ds.attrib["fidelity"] = string(MY.fidelity)
        ds.attrib["water_carry"] = string(MY.water_carry)
        ds.attrib["mix_numbers"] = Int(MY.mix_numbers)
        ds.attrib["mynn_interval_s"] = MY.interval_steps * MY.ts
        ds.attrib["K_max"] = isinf(MY.K_max) ? 1.0e30 : MY.K_max
        ds.attrib["n_clamp_e"] = MY.n_clamp_e
        ds.attrib["n_cap_K"] = MY.n_cap_K
        ds.attrib["n_diffnum"] = MY.n_diffnum
        ds.attrib["n_gate"] = MY.n_gate
        ds.attrib["n_stall"] = MY.n_stall
        ds.attrib["n_plume"] = MY.n_plume
        ds.attrib["census_max_K_m"] = mx(MY.K_m_max)
        ds.attrib["census_max_K_h"] = mx(MY.K_h_max)
        ds.attrib["census_max_D_gal"] = mx(MY.D_gal)
        ds.attrib["census_max_D_mish"] = mx(MY.D_mish)
        ds.attrib["census_max_ts_tau"] = mx(MY.ts_tau)
        ds.attrib["census_max_pblh"] = mx(MY.pblh)
        ds.attrib["patch_offset_l"] = mtile.tile.params.patchOffsetL

        NCDatasets.defDim(ds, "time", 1)
        NCDatasets.defDim(ds, "x", ncol)
        NCDatasets.defDim(ds, "z", kDim)

        tv = NCDatasets.defVar(ds, "time", Float64, ("time",))
        tv.attrib["units"] = "seconds"; tv.attrib["long_name"] = "simulation time"
        tv[1] = t_model

        xv = NCDatasets.defVar(ds, "x", Float64, ("x",))
        xv.attrib["units"] = "m"
        xv.attrib["long_name"] = "tile horizontal mish coordinate"
        xv[:] = x

        zv = NCDatasets.defVar(ds, "z", Float64, ("z",))
        zv.attrib["units"] = "m"
        zv.attrib["long_name"] = "MYNN column height above ground"
        zv[:] = MY.z_lay

        wrz(name, data, units, long) = begin
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

        wrz("K_m", reshape2(MY.K_m), "m2 s-1", "momentum exchange coefficient")
        wrz("K_h", reshape2(MY.K_h), "m2 s-1", "heat/moisture exchange coefficient")
        wrz("e", reshape2(e), "m2 s-2", "mass-specific TKE, rho_e/rho_t")
        wrz("el", reshape2(MY.el), "m", "mixing length")
        wrz("sm", reshape2(MY.sm), "1", "momentum stability function")
        wrz("sh", reshape2(MY.sh), "1", "heat stability function")
        wrz("cldfra_bl", reshape2(MY.cldfra_bl), "1", "subgrid cloud fraction")
        wrz("qc_bl", reshape2(MY.qc_bl), "kg kg-1", "subgrid cloud liquid mixing ratio")
        wrz("qi_bl", reshape2(MY.qi_bl), "kg kg-1", "subgrid cloud ice mixing ratio")
        wrz("vt", reshape2(MY.vt), "1", "condensation buoyancy coefficient (temperature)")
        wrz("vq", reshape2(MY.vq), "1", "condensation buoyancy coefficient (moisture)")
        wrz("P_s", reshape2(MY.g_Ps), "W m-3", "discrete shear production")
        wrz("P_s_mynn", reshape2(MY.g_Ps_mynn), "W m-3",
            "closure's own shear production, rho_t K_m G_M")
        wrz("P_b", reshape2(MY.g_Pb), "W m-3",
            "buoyancy production/consumption, rho_t K_h G_H")
        wrz("eps", reshape2(MY.g_eps), "W m-3", "dissipation, rho_t q^3/(B1 l)")
        wrz("tke_transport", reshape2(MY.g_tke_transport), "W m-3",
            "fitted TKE turbulent-transport divergence, dz(S_e)")
        wrz("s_aw", reshape2(MY.s_aw), "m s-1", "plume mass-flux sum, sum_i a_i w_i")
        wrz("edmf_a", zerorc(), "1",
            "plume area-fraction sum (not separately held by MYNNState; zero)")
        wrz("edmf_w", zerorc(), "m s-1",
            "plume vertical-velocity sum (not separately held by MYNNState; zero)")

        wrx("pblh", MY.pblh, "m", "boundary layer height")
        wrx("kpbl", Float64.(MY.kpbl), "1", "PBL top layer index")
        wrx("ust", MY.ust, "m s-1", "friction velocity")
        wrx("inv_L", MY.rmol, "m-1", "inverse Obukhov length, 1/L")
        wrx("plume_ktop", Float64.(MY.plume_ktop), "1",
            "running-max plume top layer index")
        wrx("plume_ztop", MY.plume_ztop, "m", "running-max plume top height")
        wrx("aw_max", MY.aw_max, "m s-1", "running-max plume mass flux Sigma_aw")
        wrx("bdry_E", MY.bdry_E, "W m-2", "D3 boundary/surface energy input")
    end
    return nothing
end

# ── Reader ───────────────────────────────────────────────────────────────────────

"""
    mynn_snapshots(dir) -> Vector{String}

Every output tag `dir` holds a MYNN sidecar for, sorted by model time. Empty when the run
carries no sidecars (`:mynn` off, or `:mynn_output = false`). The `radiation_snapshots`
pattern exactly.
"""
function mynn_snapshots(dir::AbstractString)
    isdir(dir) || return String[]
    tags = String[]
    for f in readdir(dir)
        m = match(r"^(.*)_mynn_i[0-9]+\.nc$", f)
        m === nothing && continue
        (m.captures[1] in tags) || push!(tags, m.captures[1])
    end
    return sort(tags; by = s -> parse(Float64, s))
end

_mn_squeeze(ds, name) = dropdims(coalesce.(Array(ds[name]), NaN); dims = 1)

"""
    read_mynn(dir, tag) -> NamedTuple

Read every tile's MYNN sidecar for output tag `tag` in `dir` and concatenate them along
`x` in ascending `patchOffsetL` order, reconstructing one domain-wide snapshot -- the
`read_radiation` pattern exactly. Errors if no file matches.

Returns a `NamedTuple` with the coordinates (`t`, `x`, `z`), every `(x, z)` field as a
plain `Matrix{Float64}` and every per-column field as a `Vector{Float64}` (field names
match the NetCDF variable names, with `inv_L` for the `rmol` state field), plus the
resolved-option and counter attributes read off the FIRST tile (identical on every tile of
a run by construction) with the six counters SUMMED across tiles (the domain-wide census).
"""
function read_mynn(dir::AbstractString, tag::AbstractString)
    pat = Regex("^" * Base.escape_string(String(tag)) * raw"_mynn_i([0-9]+)\.nc$")
    files = filter(f -> occursin(pat, f), readdir(dir))
    isempty(files) && error(
        "read_mynn: no sidecar files matching $(tag)_mynn_i*.nc in $dir")
    offs = [parse(Int, match(pat, f).captures[1]) for f in files]
    files = files[sortperm(offs)]

    snaps = map(files) do f
        NCDatasets.NCDataset(joinpath(dir, f), "r") do ds
            (
                x = coalesce.(Array(ds["x"]), NaN)::Vector{Float64},
                K_m = _mn_squeeze(ds, "K_m"), K_h = _mn_squeeze(ds, "K_h"),
                e = _mn_squeeze(ds, "e"),
                el = _mn_squeeze(ds, "el"), sm = _mn_squeeze(ds, "sm"),
                sh = _mn_squeeze(ds, "sh"),
                cldfra_bl = _mn_squeeze(ds, "cldfra_bl"),
                qc_bl = _mn_squeeze(ds, "qc_bl"), qi_bl = _mn_squeeze(ds, "qi_bl"),
                vt = _mn_squeeze(ds, "vt"), vq = _mn_squeeze(ds, "vq"),
                P_s = _mn_squeeze(ds, "P_s"), P_s_mynn = _mn_squeeze(ds, "P_s_mynn"),
                P_b = _mn_squeeze(ds, "P_b"), eps = _mn_squeeze(ds, "eps"),
                tke_transport = _mn_squeeze(ds, "tke_transport"),
                s_aw = _mn_squeeze(ds, "s_aw"), edmf_a = _mn_squeeze(ds, "edmf_a"),
                edmf_w = _mn_squeeze(ds, "edmf_w"),
                pblh = _mn_squeeze(ds, "pblh"), kpbl = _mn_squeeze(ds, "kpbl"),
                ust = _mn_squeeze(ds, "ust"), inv_L = _mn_squeeze(ds, "inv_L"),
                plume_ktop = _mn_squeeze(ds, "plume_ktop"),
                plume_ztop = _mn_squeeze(ds, "plume_ztop"),
                aw_max = _mn_squeeze(ds, "aw_max"), bdry_E = _mn_squeeze(ds, "bdry_E"),
                z = Array(ds["z"])::Vector{Float64},
                t = Float64(Array(ds["time"])[1]),
                attrs = Dict{String,Any}(k => ds.attrib[k] for k in keys(ds.attrib)),
            )
        end
    end

    x = vcat((s.x for s in snaps)...)
    vcat2(field) = vcat((getfield(s, field) for s in snaps)...)
    a1 = snaps[1]
    return (;
        t = a1.t, x, z = a1.z,
        K_m = vcat2(:K_m), K_h = vcat2(:K_h), e = vcat2(:e),
        el = vcat2(:el), sm = vcat2(:sm), sh = vcat2(:sh),
        cldfra_bl = vcat2(:cldfra_bl), qc_bl = vcat2(:qc_bl), qi_bl = vcat2(:qi_bl),
        vt = vcat2(:vt), vq = vcat2(:vq),
        P_s = vcat2(:P_s), P_s_mynn = vcat2(:P_s_mynn), P_b = vcat2(:P_b),
        eps = vcat2(:eps), tke_transport = vcat2(:tke_transport),
        s_aw = vcat2(:s_aw), edmf_a = vcat2(:edmf_a), edmf_w = vcat2(:edmf_w),
        pblh = vcat2(:pblh), kpbl = vcat2(:kpbl), ust = vcat2(:ust),
        inv_L = vcat2(:inv_L), plume_ktop = vcat2(:plume_ktop),
        plume_ztop = vcat2(:plume_ztop), aw_max = vcat2(:aw_max),
        bdry_E = vcat2(:bdry_E),
        closure = get(a1.attrs, "closure", NaN), edmf = get(a1.attrs, "edmf", 0),
        edmf_mom = Bool(get(a1.attrs, "edmf_mom", 0)),
        scale_aware = Bool(get(a1.attrs, "scale_aware", 0)),
        init_mode = get(a1.attrs, "init_mode", ""),
        fidelity = get(a1.attrs, "fidelity", ""),
        water_carry = get(a1.attrs, "water_carry", ""),
        mix_numbers = Bool(get(a1.attrs, "mix_numbers", 0)),
        mynn_interval_s = get(a1.attrs, "mynn_interval_s", NaN),
        K_max = get(a1.attrs, "K_max", NaN),
        n_clamp_e = sum(Int(get(s.attrs, "n_clamp_e", 0)) for s in snaps),
        n_cap_K = sum(Int(get(s.attrs, "n_cap_K", 0)) for s in snaps),
        n_diffnum = sum(Int(get(s.attrs, "n_diffnum", 0)) for s in snaps),
        n_gate = sum(Int(get(s.attrs, "n_gate", 0)) for s in snaps),
        n_stall = sum(Int(get(s.attrs, "n_stall", 0)) for s in snaps),
        n_plume = sum(Int(get(s.attrs, "n_plume", 0)) for s in snaps),
        census_max_K_m = maximum(Float64(get(s.attrs, "census_max_K_m", 0.0)) for s in snaps),
        census_max_K_h = maximum(Float64(get(s.attrs, "census_max_K_h", 0.0)) for s in snaps),
        census_max_pblh = maximum(Float64(get(s.attrs, "census_max_pblh", 0.0)) for s in snaps),
    )
end
