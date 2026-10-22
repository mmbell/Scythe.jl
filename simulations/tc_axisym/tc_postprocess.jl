#!/usr/bin/env julia
# LEGACY. Postprocess PRE-COMPREHENSIVE moist-compressible NetCDF output (raw <t>.nc
# snapshots carrying only prognostic control variables, no scythe_file_kind attribute --
# runs made before the stage that added src/netcdf_output.jl) into derived physical
# products. A run made since then writes a comprehensive `<t>.nc` per nest directly
# (options[:output_formats] default [:netcdf]) with everything this script derives
# already in it (same names/units) plus the physics groups (BL, radiation, surface) --
# nothing to run here, and tc_movie.jl reads that file natively. This script now
# REFUSES to touch a comprehensive file (see the scythe_file_kind check below); it stays
# for the runs on disk from before the change.
#
#   julia --project=. simulations/tc_axisym/tc_postprocess.jl --root <run root>
#         [--nests 1,2,...] [--ref REFFILE] [--N0 8e6] [--Nc 100] [--legacy-qss]
#
# `--root` is a run root: a simulations/tc_axisym/run.jl tree (`<root>/segments.csv`,
# `<root>/seg_NN/nest{i}/`, `<root>/reference.ref`; raw snapshots there are `<t>_raw.nc`,
# written when `output.formats` includes "netcdf_raw") or a legacy tc/tc_run_axisym.jl tree
# (`<root>/nest{i}/<t>.nc`, `<root>/tc_exact.ref`). `--indir` is an alias of `--root`.
# Segments are processed in order; each derived file's `time` is the run's ABSOLUTE time
# (t_offset_s + the segment-local t), and each later segment's duplicate t = 0 snapshot is
# skipped (tc_run_io.jl). The legacy reconstruction below predates the prognostic-rho_c
# and ice control variables; on a current-physics raw file expect it to need updating.
#
# The raw <t>.nc snapshots written by a run carry only the PROGNOSTIC control
# variables, and the moist-compressible set stores most of them as PERTURBATIONS
# off a hydrostatic PressureReferenceState (p, rho_d, rho_t, E_t, Q_ss are primes;
# u, w, v, rho_r are full). This script, for every snapshot in each nest:
#
#   1. reconstructs the z-only reference profiles on the netCDF's regular z grid
#      by fitting the run's own reference column (from <root>/reference.ref) and
#      evaluating the B-spline with SItransform — so background(z_reg) is exactly
#      the model's reference spline, not an ad-hoc interpolation;
#   2. adds the background back to recover the TOTAL control fields;
#   3. retrieves temperature T with the model's closed-form retrieval
#      (retrieve_temperature) from the PROGNOSTIC condensate, then diagnoses the
#      vapor from its own prognostic slot, exactly as the model does;
#   4. derives a simple S-band (Rayleigh) radar reflectivity from rho_c
#      (monodisperse cloud) and rho_r (exponential Marshall-Palmer rain), and a
#      rain rate from rho_r and the Ooyama (2001) terminal fall speed.
#
# Each raw <t>.nc (<t>_raw.nc) becomes <t>_derived.nc in the same nest directory, holding the
# primes, the reconstructed totals, the carried full fields, and the derived
# products (T, rho_v, rho_c, reflectivity, rain_rate). Derivatives of the
# control variables are passed through if present; derived products carry none.
#
# When the run carried radiation (S5 sidecars <t>_radiation_i*.nc alongside the
# snapshot), the sidecar is merged into the same derived file: the heating rates
# (q_lw, q_sw, q_sw_applied, dT_lw, dT_sw, dT_net) and the per-column boundary
# diagnostics (olr, sw_sfc_dn, lwp, ...), resampled from the radiation mish onto
# the regular grid, plus the solar/scheme/counter scalars as `radiation_*` global
# attributes. See the "Radiation sidecar regridding" block below for what that
# interpolation does and which fields deliberately stay in the sidecar. A run
# without sidecars produces exactly the file it always did.
#
# When the run carried the MYNN-EDMF boundary layer (S9 sidecars <t>_mynn_i*.nc), that
# sidecar is merged the SAME way: the held closure fields (K_m, K_h, mynn_e, mynn_el,
# mynn_P_s, mynn_P_b, mynn_eps, mynn_tke_transport, mynn_cldfra_bl, ...) and the
# per-column diagnostics (pblh, ust, inv_L, ...), resampled from the closure's own mish
# onto the regular grid, plus the resolved-option/counter scalars as `mynn_*` global
# attributes. A run without MYNN sidecars produces exactly the file it always did.
#

using Scythe
using Springsteel
using NCDatasets
include(joinpath(@__DIR__, "tc_run_io.jl"))

# ── Arguments ────────────────────────────────────────────────────────────────
indir  = ""                             # the run root (--root / --indir)
nests  = Int[]                          # empty => every patch of the run
reffile = nothing                       # default: <root>/reference.ref (legacy: tc_exact.ref)
N0 = 8.0e6                               # [m^-4] Marshall-Palmer rain intercept
Nc_cm3 = 100.0                           # [cm^-3] monodisperse cloud droplet count
legacy_qss = false                       # --legacy-qss: for runs made BEFORE
                                         # options[:consistent_qss_reference]
let i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a in ("--root", "--indir"); global indir = ARGS[i+1];  i += 2
        elseif a == "--nests";  global nests = parse.(Int, replace.(split(ARGS[i+1], ","), "nest" => "")); i += 2
        elseif a == "--ref";    global reffile = ARGS[i+1]; i += 2
        elseif a == "--N0";     global N0 = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--Nc";     global Nc_cm3 = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--legacy-qss"; global legacy_qss = true; i += 1
        else error("Unknown argument: $a")
        end
    end
end
isempty(indir) && error("--root <run root> is required")
indir = abspath(indir)
isdir(indir) || error("Run root not found: $indir")
reffile === nothing && (reffile = tc_reference_file(indir))
isfile(reffile) || error("Reference-state file not found: $reffile")
isempty(nests) && (nests = collect(1:tc_npatches(indir)))

# A RAW snapshot: `<t>_raw.nc` (the driver's "netcdf_raw" format) or, in a legacy tree,
# `<t>.nc` -- the whole name a number plus the extension: a leading-digit test alone also
# matches this script's own `<t>_derived.nc` and the radiation/MYNN sidecars
# `<t>_radiation_i<offset>.nc`, which then reach `parse(Float64, ...)` as not-a-time.
# In a segment-driver tree the comprehensive file owns `<t>.nc`, so only `<t>_raw.nc` is raw
# there; a legacy tree's raw snapshots are `<t>.nc` (and a comprehensive one is refused below).
const RAW_SNAPSHOT_RE = is_segmented_root(indir) ? r"^([0-9]+(?:\.[0-9]+)?)_raw\.nc$" :
                                                   r"^([0-9]+(?:\.[0-9]+)?)\.nc$"
israw(f) = occursin(RAW_SNAPSHOT_RE, f)
snaptag(f) = match(RAW_SNAPSHOT_RE, f).captures[1]

# LIVE FOOTGUN without this: RAW_SNAPSHOT_RE also matches the model's own comprehensive
# <t>.nc (same "<number>.nc" name), and this script would then treat its `p` variable
# (already a TOTAL, background included) as a PRIME and add the reference background to
# it a second time -- silently corrupting every derived field. Check the marker the
# model writes (`scythe_file_kind`, src/netcdf_output.jl) before touching anything.
function check_not_comprehensive(path)
    kind = NCDataset(path, "r") do ds
        get(ds.attrib, "scythe_file_kind", nothing)
    end
    kind == "comprehensive" && error(
        "$(path) is already a comprehensive Scythe NetCDF file " *
        "(scythe_file_kind = \"comprehensive\") -- tc_postprocess.jl is LEGACY, for " *
        "pre-comprehensive raw snapshots only. Use $(path) directly, or point " *
        "tc_movie.jl at the run root; it reads comprehensive files natively and " *
        "needs no postprocessing step.")
end

# The sidecar family for one snapshot tag, so a tag with radiation output can be detected
# without opening anything (`Scythe.read_radiation` errors when nothing matches).
has_radiation(ndir, tag) =
    any(f -> occursin(Regex("^" * Base.escape_string(tag) * raw"_radiation_i[0-9]+\.nc$"), f),
        readdir(ndir))

# The MYNN sidecar family for one snapshot tag (S9), the `has_radiation` pattern exactly.
has_mynn(ndir, tag) =
    any(f -> occursin(Regex("^" * Base.escape_string(tag) * raw"_mynn_i[0-9]+\.nc$"), f),
        readdir(ndir))

# ── The run's water representation ────────────────────────────────────────────
# The transforms and the water anchor the RUN used, from its resolved config (a segmented
# tree's `<root>/scythe_config.toml`, through the same `build_physics` the run used); a legacy
# tree predates every option here and gets the code defaults of its day (no transforms,
# slot 3 the total density). Under `water_anchor = derived` slot 3 is a PASSIVE SHADOW: the
# total density the run read is `rho_d + sum_k alpha_k nu_k` (alpha = 2 transformed, 1
# density), which the comprehensive `<t>.nc` writes as `rho_t`; it is read from there when the
# run wrote one beside the raw snapshot on the same grid, and rebuilt from the raw slots
# otherwise (the model's member order: vapor, cloud, rain, the three ice masses).
function detect_water_transforms(root)
    if is_segmented_root(root)
        cfg = Scythe.load_config(Scythe.run_config_file(root); set = ["output.dir=" * root])
        o, pp = Scythe.build_physics(cfg)
        ice = Scythe.ice_microphysics(o) === :ishmael
        return (; ctrans = Scythe.condensate_transform_mode(o),
                  cmu = Float64(get(pp, :condensate_mu, 1.0e-7)),
                  rtrans = Scythe.rain_transform_mode(o),
                  rmu = Float64(get(pp, :rain_mu, 1.0e-7)),
                  vtrans = Scythe.vapor_transform_mode(o), vmu = Scythe.vapor_mu(pp),
                  itrans = ice ? Scythe.ice_transform_mode(o) : :none,
                  imu = ice ? Scythe.ice_mu(pp, 1) : 1.0,
                  anchor = Scythe.water_anchor_mode(o))
    end
    return (; ctrans = :none, cmu = 1.0e-7, rtrans = :none, rmu = 1.0e-7, vtrans = :none,
              vmu = 1.0e-9, itrans = :none, imu = 1.0, anchor = :slot)
end
const WTRANS = detect_water_transforms(indir)

println("Postprocessing nest(s) $(join(nests, ", ")) of $indir " *
        "($(length(tc_segments(indir))) segment(s)), reference $reffile")

# ── S-band Rayleigh reflectivity ──────────────────────────────────────────────
# The implementation now lives in the MODEL (`Scythe.reflectivity_dBZ`, src/netcdf_output.jl),
# which writes the same product straight into `<t>.nc`. This script keeps the name as an
# alias rather than a copy, so the two can never drift: a change to the assumed DSD has to
# happen in one place and both the live output and this postprocessor follow it.
const REFL_FLOOR_DBZ = Scythe.REFL_FLOOR_DBZ
const reflectivity_dBZ = Scythe.reflectivity_dBZ

# ── Reference column (shared: all nests use the same vertical grid) ───────────
# Read the run's own reference column values (z p rho_d rho_v rho_c, p in Pa) and
# fit the B-spline reference column so it can be evaluated on ANY z.
reflines = readlines(reffile)
nmish = length(reflines)
zm  = Vector{Float64}(undef, nmish); pm  = similar(zm)
rdm = similar(zm); rvm = similar(zm); rcm = similar(zm)
for (i, l) in enumerate(reflines)
    parts = split(l)
    zm[i]  = parse(Float64, parts[1]); pm[i]  = parse(Float64, parts[2])
    rdm[i] = parse(Float64, parts[3]); rvm[i] = parse(Float64, parts[4])
    rcm[i] = parse(Float64, parts[5])
end

"""Fit `vals_mish` (in reference-column mish order) to `col` and evaluate at `z`."""
function eval_ref(col, vals_mish, z)
    col.uMish .= vals_mish
    Springsteel.Btransform!(col)
    Springsteel.Atransform!(col)
    return Springsteel.SItransform(col, collect(z), zeros(Float64, length(z)))
end

"""
Build the reference column for a nest's (regular) vertical grid `z_reg` and return
the background profiles evaluated there: (pbar, rho_dbar, rho_tbar, E_tbar,
Q_ssbar, Tbar). `mubar = nmish / (length(z_reg) - 1)`.
"""
function reference_background(x, z_reg)
    n_i = length(x); n_k = length(z_reg)
    mubar = round(Int, nmish / (n_k - 1))
    mubar * (n_k - 1) == nmish ||
        error("Cannot infer mubar: $nmish mish levels vs $(n_k-1) cells")
    vars = ["p", "rho_d", "rho_t", "u", "w", "E_t", "Q_ss", "rho_r", "rho_c", "v",
            "rho_v"]
    bc = Dict(v => NeumannBC() for v in vars)
    gp = Scythe.compute_derived_params(GridParameters(; geometry = "RiRk",
        iMin = x[1], iMax = x[end], num_cells_i = max(n_i - 1, 1),
        kMin = z_reg[1], kMax = z_reg[end], num_cells_k = n_k - 1, mubar = mubar,
        BCL = bc, BCR = bc, BCB = bc, BCT = bc,
        vars = Dict(v => i for (i, v) in enumerate(vars))))
    patch = createGrid(gp)
    column = Scythe.reference_column(patch, gp)
    length(column.uMish) == nmish ||
        error("Reference column mish ($(length(column.uMish))) ≠ ref file ($nmish)")

    # Base profiles on the regular grid, then the derived reference exactly as
    # Springsteel's _pressure_reference builds it (moist EOS, Emanuel constants).
    pbar    = eval_ref(column, pm, z_reg)
    rho_dbar = eval_ref(column, rdm, z_reg)
    rho_vbar = eval_ref(column, rvm, z_reg)
    rho_cbar = eval_ref(column, rcm, z_reg)
    rho_tbar = rho_dbar .+ rho_vbar .+ rho_cbar
    Rd = Springsteel.Thermodynamics.Rd; Rv = Springsteel.Thermodynamics.Rv
    Tbar = pbar ./ ((rho_dbar .* Rd) .+ (rho_vbar .* Rv))
    q_v = rho_vbar ./ rho_dbar; q_l = rho_cbar ./ rho_dbar
    E_tbar = (rho_dbar .* Springsteel.Thermodynamics.internal_energy_bf02.(Tbar, q_v, q_l)) .+
             (rho_tbar .* Scythe.gravity .* z_reg)
    # Q̄_ss must be built the way the RUN built it, or adding it back to the stored
    # Q_ss' recovers the wrong total and corrupts T and the water partition. The TC
    # configuration sets options[:consistent_qss_reference], which defines Q̄_ss through
    # the model's own retrieval so the resting reference does not condense (see
    # Scythe.consistent_qss_reference); the pointwise EOS form below it is what
    # Springsteel stores by default and is NOT what the run used.
    M_bar = pbar .+ E_tbar .- (rho_tbar .* (Scythe.gravity .* z_reg))
    T_ret = Scythe.retrieve_temperature.(M_bar, rho_dbar, rho_tbar, rho_cbar)
    Q_ssbar = if legacy_qss
        rho_vbar .- Springsteel.Thermodynamics.rho_v_sat.(Tbar, pbar ./ 100.0)
    else
        (rho_tbar .- rho_dbar .- rho_cbar) .-
            Springsteel.Thermodynamics.rho_v_sat.(T_ret, pbar ./ 100.0)
    end
    return (; pbar, rho_dbar, rho_tbar, rho_cbar, E_tbar, Q_ssbar, Tbar)
end

# ── Radiation sidecar regridding (mish → regular) ─────────────────────────────
# The S5 sidecar (`<t>_radiation_i*.nc`, src/radiation_io.jl) is written MISH-NATIVE —
# deliberately, because the fluxes live on faces and on the stratospheric extension above
# the model top, neither of which is on the prognostic grid. The derived file here lives on
# the regular (x, z) grid the raw snapshot was written on, so merging the two means
# resampling: separable linear interpolation, x then z, from the radiation mish (Gauss
# points in x, layer centres in z) onto that regular grid.
#
# Outside the mish hull the value is CLAMPED to the edge, not extrapolated. The first and
# last Gauss points sit strictly inside the first and last cell, so the regular grid's end
# points — r = 0 and the outer wall, z = 0 and the model top — lie outside it by up to half
# a cell. Holding the edge value there is the conservative choice: linear extrapolation off
# a two-point slope at the wall would manufacture a gradient the radiation never computed.
#
# The face-based flux profiles (`flux_lw_*`, `flux_sw_*`, on `zf`, which extends to 70 km)
# are NOT merged: they have no counterpart on the derived file's 25 km regular z grid and
# regridding them would be a lie about where they live. They remain in the sidecar, which
# `Scythe.read_radiation` reads directly. The per-column boundary values derived FROM them
# (olr, sw_sfc_dn, ...) do come across, since those are scalars per column.

# The three regridding helpers below are the MODEL's (`Scythe.interp_weights`,
# `Scythe.regrid2d`, `Scythe.regrid1d`, src/netcdf_output.jl), aliased rather than copied
# for the reason given at `reflectivity_dBZ` above: stage N2 folds the physics sidecars into
# `<t>.nc` with these same weights, and a second copy here would be free to drift from the
# interpolation the model actually writes.
const interp_weights = Scythe.interp_weights
const regrid2d = Scythe.regrid2d
const regrid1d = Scythe.regrid1d

# ── Per-snapshot derivation and write ─────────────────────────────────────────
rd2d(ds, name, n_i, n_k) = coalesce.(Array(ds[name])[1, :, :], NaN)   # (n_i, n_k)

function process_snapshot(rawpath, outpath, bg, z_reg, tsec, rad = nothing, mynn = nothing)
    NCDataset(rawpath, "r") do ds
        x = ds["x"][:]; z = ds["z"][:]
        n_i = length(x); n_k = length(z)
        has_v = haskey(ds, "v")

        # Prognostic slots (primes for the reference-referenced ones; full otherwise)
        pp    = rd2d(ds, "p", n_i, n_k);     rdp = rd2d(ds, "rho_d", n_i, n_k)
        rtp   = rd2d(ds, "rho_t", n_i, n_k); Etp = rd2d(ds, "E_t", n_i, n_k)
        Qssp  = rd2d(ds, "Q_ss", n_i, n_k)
        u     = rd2d(ds, "u", n_i, n_k);     w   = rd2d(ds, "w", n_i, n_k)
        # Slots 8 and 9 may carry Ooyama control variables rather than densities, and the
        # netCDF variable NAME says which (Springsteel writes it from GridParameters.vars).
        # Recover the densities here so everything below -- the retrieval, the reflectivity
        # and the rain rate -- reads what it thinks it reads.
        rname = haskey(ds, "nu_r") ? "nu_r" : "rho_r"
        cname = haskey(ds, "nu_c") ? "nu_c" : "rho_c"
        rtrans = rname == "nu_r" ? WTRANS.rtrans : :none
        ctrans = cname == "nu_c" ? WTRANS.ctrans : :none
        nu_r = rd2d(ds, rname, n_i, n_k); rcp = rd2d(ds, cname, n_i, n_k)
        rho_r = Scythe.recover_rho_r.(nu_r, rtrans, WTRANS.rmu)
        v     = has_v ? rd2d(ds, "v", n_i, n_k) : zeros(n_i, n_k)

        # Totals = prime + background(z), broadcast over radius
        col(a) = reshape(a, 1, n_k)
        p    = pp   .+ col(bg.pbar);     rho_d = rdp .+ col(bg.rho_dbar)
        rho_t = rtp .+ col(bg.rho_tbar); E_t  = Etp .+ col(bg.E_tbar)
        Q_ss = Qssp .+ col(bg.Q_ssbar)
        rho_c = Scythe.recover_rho_c.(rcp, col(bg.rho_cbar), ctrans, WTRANS.cmu)
        # The vapor slot: a density perturbation off rho_tbar - rho_dbar - rho_cbar, or under
        # the vapor control variable `nu_v'` off bhyp of it (recovered as the model does).
        rho_vbar_d = bg.rho_tbar .- bg.rho_dbar .- bg.rho_cbar
        vname = haskey(ds, "nu_v") ? "nu_v" : (haskey(ds, "rho_v") ? "rho_v" : "")
        vtr = vname == "nu_v" ? WTRANS.vtrans : :none
        nvb = vtr === :none ? rho_vbar_d : Scythe.bhyp.(rho_vbar_d, WTRANS.vmu)
        vslot = isempty(vname) ? nothing : rd2d(ds, vname, n_i, n_k)
        # Ice masses (the first moment of each species), when the run carried ice
        inames = [n for n in ("nu_i1", "nu_i2", "nu_i3", "rho_i1", "rho_i2", "rho_i3")
                  if haskey(ds, n)]
        ice_slots = [rd2d(ds, n, n_i, n_k) for n in inames]
        itr = any(startswith("nu_"), inames) ? WTRANS.itrans : :none
        rho_ice = isempty(ice_slots) ? zeros(n_i, n_k) :
            reduce(+, (Scythe.recover_total.(sl, itr, WTRANS.imu) for sl in ice_slots))
        # THE DERIVED WATER ANCHOR: slot 3 is the passive shadow, not the total density.
        if WTRANS.anchor === :derived
            vslot === nothing && error("$(rawpath): water_anchor = derived but no vapor slot")
            compf = replace(rawpath, r"_raw\.nc$" => ".nc")
            from_comp = false
            if compf != rawpath && isfile(compf)
                NCDataset(compf, "r") do dc
                    if get(dc.attrib, "scythe_file_kind", nothing) == "comprehensive" &&
                       haskey(dc, "rho_t") && length(dc["x"]) == n_i && length(dc["z"]) == n_k
                        rho_t = rd2d(dc, "rho_t", n_i, n_k)
                        from_comp = true
                    end
                end
            end
            if !from_comp
                al(tr) = tr === :none ? 1.0 : 2.0
                ncb = ctrans === :none ? bg.rho_cbar : Scythe.bhyp.(bg.rho_cbar, WTRANS.cmu)
                W = (al(vtr) .* (vslot .+ col(nvb))) .+ (al(ctrans) .* (rcp .+ col(ncb)))
                W = W .+ (al(rtrans) .* nu_r)
                for sl in ice_slots
                    W = W .+ (al(itr) .* sl)
                end
                rho_t = rho_d .+ W
            end
        end

        # Closed-form temperature retrieval from the PROGNOSTIC condensate (M is the
        # enthalpy balance the model solves).
        ke = has_v ? 0.5 .* (u .^ 2 .+ v .^ 2 .+ w .^ 2) : 0.5 .* (u .^ 2 .+ w .^ 2)
        M  = p .+ E_t .- rho_t .* (ke .+ Scythe.gravity .* col(z))
        rho_liq = rho_c .+ rho_r
        T  = Scythe.retrieve_temperature.(M, rho_d, rho_t, rho_liq, rho_ice)
        rho_vs = Springsteel.Thermodynamics.rho_v_sat.(T, p ./ 100.0)
        # The vapor is its OWN prognostic slot, carried against the derived reference
        # rho_tbar - rho_dbar - rho_cbar (`Scythe.vapor_slot`). Files written before it
        # became prognostic have no such variable; those fall back to the density residual,
        # which is what the model then meant by the vapor. The two now differ by the
        # reconciliation gap, so the fallback must not be used on a new file.
        # (Under the derived anchor the residual fallback is refused above: there is always a
        # vapor slot.)
        rho_v = vslot === nothing ? rho_t .- rho_d .- rho_liq .- rho_ice :
            (vtr === :none ? vslot .+ col(rho_vbar_d) :
                             Scythe.recover_rho_v.(vslot, col(nvb), vtr, WTRANS.vmu))

        # Derived products
        refl = reflectivity_dBZ.(rho_c, rho_r, N0, Nc_cm3 * 1.0e6)
        Vt = Scythe.rain_terminal_velocity.(rho_r, rho_d, T)          # [m/s] ≤ 0
        rain_rate = .-rho_r .* Vt .* 3600.0                          # [mm/hr] (ρ_l=1000)

        # ── Write derived file ────────────────────────────────────────────────
        NCDataset(outpath, "c") do out
            out.attrib["Conventions"] = "CF-1.12"
            out.attrib["title"] = "Scythe.jl moist-compressible derived products"
            out.attrib["source"] = "simulations/tc_axisym/tc_postprocess.jl"
            haskey(ds.attrib, "history") && (out.attrib["source_history"] = ds.attrib["history"])
            out.attrib["reflectivity_N0_m4"] = N0
            out.attrib["reflectivity_Nc_cm3"] = Nc_cm3

            defDim(out, "time", 1); defDim(out, "x", n_i); defDim(out, "z", n_k)
            tv = defVar(out, "time", Float64, ("time",))
            tv.attrib["units"] = "seconds"; tv.attrib["long_name"] = "simulation time"
            tv[1] = tsec
            for (nm, src) in (("x", x), ("z", z))
                cv = defVar(out, nm, Float64, (nm,)); cv[:] = src
                for (k, vv) in ds[nm].attrib; cv.attrib[k] = vv; end
            end

            wr(name, data, units, long) = begin
                dv = defVar(out, name, Float64, ("time", "x", "z"); fillvalue = NaN)
                dv.attrib["units"] = units; dv.attrib["long_name"] = long
                dv[1, :, :] = data
            end
            wrx(name, data, units, long) = begin
                dv = defVar(out, name, Float64, ("time", "x"); fillvalue = NaN)
                dv.attrib["units"] = units; dv.attrib["long_name"] = long
                dv[1, :] = data
            end

            # Primes (perturbations off the hydrostatic reference)
            wr("p_prime",     pp,   "Pa",     "pressure perturbation")
            wr("rho_d_prime", rdp,  "kg m-3", "dry-air density perturbation")
            wr("rho_t_prime", rtp,  "kg m-3", "total density perturbation")
            wr("E_t_prime",   Etp,  "J m-3",  "total energy density perturbation")
            wr("Q_ss_prime",  Qssp, "kg m-3", "supersaturation density perturbation")
            # Totals (background added back)
            wr("p",     p,     "Pa",     "total pressure")
            wr("rho_d", rho_d, "kg m-3", "total dry-air density")
            wr("rho_t", rho_t, "kg m-3", "total density")
            wr("E_t",   E_t,   "J m-3",  "total energy density")
            wr("Q_ss",  Q_ss,  "kg m-3", "supersaturation density")
            # Carried full prognostic fields
            wr("u", u, "m s-1", "radial velocity")
            wr("w", w, "m s-1", "vertical velocity")
            has_v && wr("v", v, "m s-1", "tangential velocity")
            wr("rho_r", rho_r, "kg m-3", "rain water density")
            # Derived products
            wr("T",           T,         "K",        "temperature (nonlinear retrieval)")
            wr("rho_v",       rho_v,     "kg m-3",   "water vapor density")
            wr("rho_c",       rho_c,     "kg m-3",   "cloud water density")
            wr("reflectivity", refl,     "dBZ",      "S-band equivalent radar reflectivity")
            wr("rain_rate",   rain_rate, "mm hr-1",  "rain rate from sedimentation flux")

            # 1-D reference background profiles (z only)
            for (nm, prof, units) in (("pbar", bg.pbar, "Pa"),
                                       ("rho_dbar", bg.rho_dbar, "kg m-3"),
                                       ("rho_tbar", bg.rho_tbar, "kg m-3"),
                                       ("E_tbar", bg.E_tbar, "J m-3"),
                                       ("Q_ssbar", bg.Q_ssbar, "kg m-3"),
                                       ("Tbar", bg.Tbar, "K"))
                bv = defVar(out, nm, Float64, ("z",))
                bv.attrib["units"] = units
                bv.attrib["long_name"] = "reference-state $nm"
                bv[:] = prof
            end

            # ── Radiation, merged from the S5 sidecar (absent ⇒ nothing below is
            # written and the file is byte-for-byte what a radiation-off run produces) ──
            if rad !== nothing
                wx = interp_weights(rad.x, x); wz = interp_weights(rad.z, z)
                net = rad.dT_lw .+ rad.sw_scale .* rad.dT_sw

                wr("q_lw", regrid2d(rad.q_lw, wx, wz), "W m-3",
                   "longwave heating rate (flux divergence)")
                wr("q_sw", regrid2d(rad.q_sw, wx, wz), "W m-3",
                   "shortwave heating rate (flux divergence), held profile")
                wr("q_sw_applied", regrid2d(rad.q_sw_applied, wx, wz), "W m-3",
                   "sw_scale * q_sw, the rate actually folded into QDOT_TH")
                wr("dT_lw", regrid2d(rad.dT_lw, wx, wz), "K day-1",
                   "longwave heating rate at constant pressure")
                wr("dT_sw", regrid2d(rad.dT_sw, wx, wz), "K day-1",
                   "shortwave heating rate at constant pressure, UNSCALED " *
                   "(multiply by radiation_sw_scale for the rate actually applied)")
                wr("dT_net", regrid2d(net, wx, wz), "K day-1",
                   "net radiative heating at constant pressure, dT_lw + sw_scale*dT_sw")

                for (nm, v, units, long) in (
                        ("olr", rad.olr, "W m-2",
                         "outgoing longwave radiation at the top of the full column " *
                         "(incl. stratospheric extension)"),
                        ("olr_model_top", rad.olr_model_top, "W m-2",
                         "longwave flux up at the model top face"),
                        ("sw_sfc_dn", rad.sw_sfc_dn, "W m-2",
                         "shortwave flux down at the surface"),
                        ("sw_toa_dn", rad.sw_toa_dn, "W m-2",
                         "shortwave flux down at the top of the full column"),
                        ("lw_sfc_dn", rad.lw_sfc_dn, "W m-2",
                         "longwave flux down at the surface"),
                        ("lwp", rad.lwp, "g m-2", "column liquid water path"),
                        ("iwp", rad.iwp, "g m-2", "column ice water path"),
                        ("cloudy", rad.cloudy, "1",
                         "cloudy-column indicator, linearly interpolated off the mish: " *
                         "0 or 1 at a mish column, fractional between two — the sidecar " *
                         "holds the exact 0/1 mask"))
                    wrx(nm, regrid1d(v, wx), units, long)
                end

                out.attrib["radiation_source"] = "Scythe radiation sidecar " *
                    "<t>_radiation_i*.nc, read with Scythe.read_radiation"
                out.attrib["radiation_regridding"] =
                    "Radiation fields were computed on the radiation MISH (Gauss points " *
                    "in x, layer centres in z) and are resampled here onto this file's " *
                    "regular (x, z) grid by separable LINEAR interpolation, x then z, " *
                    "with edge clamping outside the mish hull (the outermost Gauss " *
                    "points lie inside the first/last cell, so the wall and the model " *
                    "top are outside it by up to half a cell and hold the edge value). " *
                    "The face-based flux profiles (flux_lw_*, flux_sw_*, on zf, which " *
                    "runs to the top of the stratospheric extension) are NOT merged: " *
                    "they have no counterpart on this 25 km regular z grid. They remain " *
                    "in the sidecar. The per-column boundary values derived from them " *
                    "(olr, olr_model_top, sw_sfc_dn, sw_toa_dn, lw_sfc_dn) are here."
                out.attrib["radiation_time"] = rad.t
                out.attrib["radiation_cos_zenith"] = rad.cos_zenith
                out.attrib["radiation_toa_flux"] = rad.toa_flux
                out.attrib["radiation_sw_scale"] = rad.sw_scale
                out.attrib["radiation_scheme"] = rad.scheme
                out.attrib["radiation_method"] = rad.method
                out.attrib["radiation_solar"] = rad.solar
                out.attrib["radiation_forcing"] = rad.forcing
                out.attrib["radiation_z_max"] = rad.z_max
                out.attrib["radiation_interval_s"] = rad.radiation_interval_s
                out.attrib["radiation_n_clamp_tk"] = rad.n_clamp_tk
                out.attrib["radiation_n_clamp_re_liq"] = rad.n_clamp_re_liq
                out.attrib["radiation_n_clamp_re_ice"] = rad.n_clamp_re_ice
                out.attrib["radiation_n_neg_rho_v"] = rad.n_neg_rho_v
            end

            # ── MYNN-EDMF, merged from the S9 sidecar (absent ⇒ nothing below is written
            # and the file is byte-for-byte what a MYNN-off run produces). Same regridding
            # as the radiation block above: separable linear interpolation, x then z, edge-
            # clamped outside the closure's own mish hull. Every (x, z) field is prefixed
            # `mynn_` except `K_m`/`K_h`, which keep their bare names (an exchange
            # coefficient is unambiguous and the prefix would only make every downstream
            # reader spell it out again).
            if mynn !== nothing
                wxm = interp_weights(mynn.x, x); wzm = interp_weights(mynn.z, z)

                wr("K_m", regrid2d(mynn.K_m, wxm, wzm), "m2 s-1",
                   "MYNN momentum exchange coefficient")
                wr("K_h", regrid2d(mynn.K_h, wxm, wzm), "m2 s-1",
                   "MYNN heat/moisture exchange coefficient")
                for (nm, A, units, long) in (
                        ("mynn_e", mynn.e, "m2 s-2", "mass-specific TKE, rho_e/rho_t"),
                        ("mynn_el", mynn.el, "m", "mixing length"),
                        ("mynn_sm", mynn.sm, "1", "momentum stability function"),
                        ("mynn_sh", mynn.sh, "1", "heat stability function"),
                        ("mynn_cldfra_bl", mynn.cldfra_bl, "1", "subgrid cloud fraction"),
                        ("mynn_qc_bl", mynn.qc_bl, "kg kg-1",
                         "subgrid cloud liquid mixing ratio"),
                        ("mynn_qi_bl", mynn.qi_bl, "kg kg-1",
                         "subgrid cloud ice mixing ratio"),
                        ("mynn_vt", mynn.vt, "1",
                         "condensation buoyancy coefficient (temperature)"),
                        ("mynn_vq", mynn.vq, "1",
                         "condensation buoyancy coefficient (moisture)"),
                        ("mynn_P_s", mynn.P_s, "W m-3", "discrete shear production"),
                        ("mynn_P_s_mynn", mynn.P_s_mynn, "W m-3",
                         "closure's own shear production, rho_t K_m G_M"),
                        ("mynn_P_b", mynn.P_b, "W m-3",
                         "buoyancy production/consumption, rho_t K_h G_H"),
                        ("mynn_eps", mynn.eps, "W m-3", "dissipation, rho_t q^3/(B1 l)"),
                        ("mynn_tke_transport", mynn.tke_transport, "W m-3",
                         "fitted TKE turbulent-transport divergence, dz(S_e)"),
                        ("mynn_s_aw", mynn.s_aw, "m s-1",
                         "plume mass-flux sum, sum_i a_i w_i"))
                    wr(nm, regrid2d(A, wxm, wzm), units, long)
                end

                for (nm, v, units, long) in (
                        ("mynn_pblh", mynn.pblh, "m", "boundary layer height"),
                        ("mynn_ust", mynn.ust, "m s-1", "friction velocity"),
                        ("mynn_inv_L", mynn.inv_L, "m-1", "inverse Obukhov length, 1/L"),
                        ("mynn_bdry_E", mynn.bdry_E, "W m-2",
                         "D3 boundary/surface energy input"))
                    wrx(nm, regrid1d(v, wxm), units, long)
                end

                out.attrib["mynn_source"] = "Scythe MYNN sidecar <t>_mynn_i*.nc, " *
                    "read with Scythe.read_mynn"
                out.attrib["mynn_regridding"] =
                    "MYNN fields were computed on the closure's own mish and are " *
                    "resampled here onto this file's regular (x, z) grid by separable " *
                    "LINEAR interpolation, x then z, with edge clamping outside the " *
                    "mish hull (the radiation_regridding convention, applied to the " *
                    "MYNN mish)."
                out.attrib["mynn_closure"] = mynn.closure
                out.attrib["mynn_edmf"] = mynn.edmf
                out.attrib["mynn_init_mode"] = mynn.init_mode
                out.attrib["mynn_water_carry"] = mynn.water_carry
                out.attrib["mynn_n_clamp_e"] = Float64(mynn.n_clamp_e)
                out.attrib["mynn_n_cap_K"] = Float64(mynn.n_cap_K)
                out.attrib["mynn_n_diffnum"] = Float64(mynn.n_diffnum)
                out.attrib["mynn_n_gate"] = Float64(mynn.n_gate)
                out.attrib["mynn_n_stall"] = Float64(mynn.n_stall)
                out.attrib["mynn_n_plume"] = Float64(mynn.n_plume)
            end

            # Pass through any control-variable derivative slots that were output
            # (derived products get none). These are extra data variables in the
            # source beyond the value slots handled above.
            handled = Set(["time", "x", "z", "p", "rho_d", "rho_t", "u", "w",
                           "E_t", "Q_ss", "rho_r", "rho_c", "nu_r", "nu_c", "v",
                           "rho_v"])
            for (vn, vv) in ds
                (vn in handled) && continue
                ndims(vv) == 3 || continue
                dv = defVar(out, vn, Float64, ("time", "x", "z"); fillvalue = NaN)
                for (k, av) in vv.attrib; dv.attrib[k] = av; end
                dv[1, :, :] = coalesce.(Array(vv)[1, :, :], NaN)
            end
        end
        return (; t = tsec, max_refl = maximum(x -> isnan(x) ? -Inf : x, refl),
                  max_rain = maximum(rain_rate), max_v = has_v ? maximum(v) : NaN,
                  Tmin = minimum(T), Tmax = maximum(T),
                  olr_mean = rad === nothing ? NaN : sum(rad.olr) / length(rad.olr))
    end
end

# ── Drive over nests, segments and snapshots ──────────────────────────────────
for nest in nests
    local bg, z_reg
    have_bg = false
    for (seg, ndir) in tc_nest_dirs(indir, nest)
        raws = sort(filter(israw, readdir(ndir)), by = f -> parse(Float64, snaptag(f)))
        # A later segment's t = 0 duplicates the previous segment's last snapshot.
        seg.segment > 1 && filter!(f -> parse(Float64, snaptag(f)) != 0.0, raws)
        isempty(raws) && (println("  nest$nest seg $(seg.segment): no raw snapshots, skipping"); continue)
        # All snapshots of one run share the same scythe_file_kind; checking the first is
        # enough to refuse a comprehensive-output run before any real work starts.
        check_not_comprehensive(joinpath(ndir, raws[1]))

        # Reference background from this nest's z grid (built once per nest)
        if !have_bg
            NCDataset(joinpath(ndir, raws[1]), "r") do ds
                z_reg = ds["z"][:]
                bg = reference_background(ds["x"][:], z_reg)
            end
            have_bg = true
        end

        println("  nest$nest seg $(seg.segment) (t_offset $(seg.t_offset) s): " *
                "$(length(raws)) snapshot(s)")
        for f in raws
            raw = joinpath(ndir, f)
            tag = snaptag(f)
            out = joinpath(ndir, tag * "_derived.nc")
            tsec = seg.t_offset + parse(Float64, tag)     # absolute run time
            # Merge the radiation sidecar when this snapshot has one (S5 runs only).
            rad = has_radiation(ndir, tag) ? Scythe.read_radiation(ndir, tag) : nothing
            # Merge the MYNN sidecar when this snapshot has one (S9 runs only).
            mynn = has_mynn(ndir, tag) ? Scythe.read_mynn(ndir, tag) : nothing
            s = process_snapshot(raw, out, bg, z_reg, tsec, rad, mynn)
            refl_str = isfinite(s.max_refl) ? "$(round(s.max_refl;digits=1)) dBZ" : "no echo"
            println("    t=$(round(Int, tsec)) s: " *
                    "T∈[$(round(s.Tmin;digits=1)),$(round(s.Tmax;digits=1))] K  " *
                    "max_refl=$(refl_str)  " *
                    "max_rain=$(round(s.max_rain;digits=2)) mm/hr  " *
                    "max_v=$(round(s.max_v;digits=1)) m/s" *
                    (isnan(s.olr_mean) ? "" :
                     "  OLR=$(round(s.olr_mean;digits=1)) W/m²"))
        end
    end
end
println("Done. Derived files written as <t>_derived.nc alongside each snapshot.")
