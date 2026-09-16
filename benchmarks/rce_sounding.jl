#!/usr/bin/env julia
# Turn the time- and domain-mean state of a 2D ocean_rce.jl run into a SOUNDING FILE in
# the format Springsteel's reference-state builder reads (`_read_sounding` /
# `calculate_pressure_reference_state`, src/reference_state.jl:231-320 in Springsteel.jl),
# so the TC driver (tc/tc_init.jl, via tc/tc_params.jl's SOUNDING constant) can start from
# a model-consistent, SST-tuned, near-moist-neutral environment instead of the Dunion MT
# sounding.
#
# Format (see benchmarks/reference_data/o01_rainfall/dunion_MT_hum90.ref for the exemplar):
#   line 1        p_sfc[hPa] theta[K] q_v[g/kg]              (z = 0)
#   line 2..N     z[m] theta[K] q_v[g/kg] u[m/s] v[m/s]       (one per level, u=v=0 here)
# `_interp_to_levels` is piecewise linear and THROWS if a model level is not bracketed, so
# the file must extend above the TC's Z_TOP (tc/tc_params.jl); this script splices on the
# high-altitude rows of a Dunion-format file (default: dunion_MT_hum90.ref) to guarantee that.
#
#   julia --project=. benchmarks/rce_sounding.jl --dir <root> [--avg-days 5] [--out <path>]
#       [--splice benchmarks/reference_data/o01_rainfall/dunion_MT_hum90.ref] [--sst <K>]
#
# `<root>` is an ocean_rce.jl output root (benchmarks/output/ocean_rce_<mode>_mc_rirk<tag>/),
# holding `segments.csv` and `seg_NN/<t>.nc` comprehensive NetCDF files. `--sst` is the last
# resort in the SST-resolution chain (netCDF attr -> "RCE:" setup line in
# seg_01/segment_stdout.log -> --sst) and is REQUIRED if neither of the first two is found.

using Printf

include(joinpath(@__DIR__, "common", "harness.jl"))

using NCDatasets
using Springsteel
using Scythe

const RCE_REF_DIR = joinpath(REFERENCE_DATA_DIR, "rce")
const DEFAULT_SPLICE = joinpath(REFERENCE_DATA_DIR, "o01_rainfall", "dunion_MT_hum90.ref")

# ── CLI ──────────────────────────────────────────────────────────────────────────────

function parse_sounding_args(args::Vector{String})
    dir = nothing
    avg_days = 5.0
    out = nothing
    splice = DEFAULT_SPLICE
    sst_cli = nothing
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--dir"
            dir = args[i+1]; i += 2
        elseif a == "--avg-days"
            avg_days = parse(Float64, args[i+1]); i += 2
        elseif a == "--out"
            out = args[i+1]; i += 2
        elseif a == "--splice"
            splice = args[i+1]; i += 2
        elseif a == "--sst"
            sst_cli = parse(Float64, args[i+1]); i += 2
        else
            error("rce_sounding.jl: unrecognized argument \"$a\"")
        end
    end
    dir === nothing && error("rce_sounding.jl: --dir <root> is required")
    avg_days > 0.0 || error("--avg-days must be > 0")
    return (dir = dir, avg_days = avg_days, out = out, splice = splice, sst_cli = sst_cli)
end

opts = parse_sounding_args(ARGS)
root = abspath(opts.dir)
isdir(root) || error("rce_sounding.jl: no such directory: $root")

# ── Locate every <t>.nc under segments.csv, in absolute (t_offset_s + t) time ────────

segments_csv = joinpath(root, "segments.csv")
isfile(segments_csv) || error("rce_sounding.jl: no segments.csv in $root")
segs = CSV.read(segments_csv, DataFrame)

struct RceSample
    abs_t::Float64
    path::String
    seg::Int
end

const SNAPSHOT_RE = r"^([0-9]+(\.[0-9]+)?)\.nc$"

samples = RceSample[]
for row in eachrow(segs)
    k = Int(row.segment)
    seg_dir = String(row.dir)
    if !isdir(seg_dir)
        println("WARNING: segment $k directory missing, skipping: $seg_dir")
        continue
    end
    t_off = Float64(row.t_offset_s)
    for f in readdir(seg_dir)
        m = match(SNAPSHOT_RE, f)
        m === nothing && continue
        t = parse(Float64, m.captures[1])
        # Each segment's 0.0.nc duplicates the PREVIOUS segment's final file (same absolute
        # time, identical data) -- drop it for every segment after the first so it is not
        # double-counted in the time mean. Segment 1's 0.0.nc is the true t=0 initial state.
        k > 1 && t == 0.0 && continue
        push!(samples, RceSample(t_off + t, joinpath(seg_dir, f), k))
    end
end
isempty(samples) && error("rce_sounding.jl: no <t>.nc snapshots found under $root")
sort!(samples, by = s -> s.abs_t)

# ── Averaging window: last `avg_days`, clamped to what exists ────────────────────────

t_end = maximum(s.abs_t for s in samples)
t_start_all = minimum(s.abs_t for s in samples)
available_days = (t_end - t_start_all) / 86400.0
window_s = opts.avg_days * 86400.0
t_start = t_end - window_s
if t_start < t_start_all - 1.0e-6
    println("WARNING: --avg-days $(opts.avg_days) requests more than the " *
            "$(round(available_days; digits = 4)) day(s) of output available under $root; " *
            "clamping the averaging window to all available output.")
    t_start = t_start_all
end
window_samples = filter(s -> s.abs_t >= t_start - 1.0e-6, samples)
isempty(window_samples) && error("rce_sounding.jl: no samples fall in the averaging window")
println("rce_sounding: averaging $(length(window_samples)) file(s) over absolute t in " *
        "[$(round(t_start; digits=2)), $(round(t_end; digits=2))] s " *
        "($(round((t_end - t_start) / 86400.0; digits = 4)) day(s) of a requested " *
        "$(opts.avg_days) day(s))")

# ── Read one snapshot's (x, z) fields ─────────────────────────────────────────────────

function read_rce_profile(path::String)
    return NCDataset(path, "r") do ds
        x = Array(ds["x"][:])
        z = Array(ds["z"][:])
        get1(name) = coalesce.(Array(ds[name])[1, :, :], NaN)   # (x, z)
        nx = length(x)
        # NOTE: the comprehensive file's own "theta" is NOT read here. It is written as
        # Springsteel.potential_temperature(p, rho_d) -- a DRY-density approximation, T_d :=
        # p_total/(Rd*rho_d), that never separates out the vapor partial pressure e. The
        # sounding-file reader (_read_sounding / calculate_pressure_reference_state) instead
        # assumes the textbook convention theta = T*(p0/p_total)^(Rd/Cpd) with T the TRUE
        # temperature, then derives rho_d = (p_total - e)/(Rd*T) separately. Writing the
        # model's "theta" straight into the file double-counts/mismatches the vapor
        # correction (rho_d comes out ~q_v/(Eps+q_v), a few percent, too LOW), which measured
        # ~4% in the round-trip p check before this fix. theta is instead rebuilt below from
        # the averaged (T, p) -- T is the model's accurate closed-form retrieval -- using
        # exactly the reader's own convention.
        p = get1("p"); T = get1("T"); q_v = get1("q_v")
        u = get1("u")
        v = haskey(ds, "v") ? get1("v") : zeros(nx, length(z))
        attrs = Dict{String,Any}(k => ds.attrib[k] for k in keys(ds.attrib))
        return (; x, z, p, T, q_v, u, v, attrs)
    end
end

# x-mean dropping the duplicate last column (regular x is a LinRange INCLUDING both ends
# under periodic BCs, so x[end] repeats x[1]).
xmean(field::AbstractMatrix) = vec(sum(view(field, 1:size(field, 1) - 1, :); dims = 1) ./
                                   (size(field, 1) - 1))

first_prof = read_rce_profile(window_samples[1].path)
z = first_prof.z
nz = length(z)
function accumulate_window(window_samples, first_prof, nz)
    acc_p = zeros(nz); acc_T = zeros(nz); acc_qv = zeros(nz)
    acc_u = zeros(nz); acc_v = zeros(nz)
    n = 0
    last_attrs = first_prof.attrs
    for s in window_samples
        prof = s === window_samples[1] ? first_prof : read_rce_profile(s.path)
        length(prof.z) == nz || error("rce_sounding.jl: z grid mismatch at $(s.path)")
        any(isnan, prof.p) && error("rce_sounding.jl: NaN in p at $(s.path) (nested " *
                                    "boundary fill leaking into a single-patch average?)")
        acc_p .+= xmean(prof.p)
        acc_T .+= xmean(prof.T)
        acc_qv .+= xmean(prof.q_v)
        acc_u .+= xmean(prof.u)
        acc_v .+= xmean(prof.v)
        n += 1
        last_attrs = prof.attrs
    end
    return (; p = acc_p ./ n, T = acc_T ./ n, qv = acc_qv ./ n,
            u = acc_u ./ n, v = acc_v ./ n, last_attrs)
end

wmean = accumulate_window(window_samples, first_prof, nz)
p_mean = wmean.p              # Pa
T_mean = wmean.T              # K
qv_mean = wmean.qv            # kg/kg
u_mean = wmean.u              # m/s
# Dry potential temperature in the sounding-file's OWN convention: theta = T*(p0/p)^(Rd/Cpd)
# with p0 = Springsteel.p_0 = 1000.0 hPa and p TOTAL pressure -- see the note in
# read_rce_profile above for why this is rebuilt from (T, p) rather than taken from the
# file's own "theta" diagnostic.
theta_mean = T_mean .* (Springsteel.p_0 ./ (p_mean ./ 100.0)) .^
             (Springsteel.Rd / Springsteel.Cpd)   # K
v_mean = wmean.v              # m/s
last_attrs = wmean.last_attrs

# ── Saturation cap ─────────────────────────────────────────────────────────────────────
# q_sat_liquid(Tk, phPa) is what calculate_pressure_reference_state itself uses (via
# vapor_pressure, same total-pressure-in-hPa convention) to define e/q_v, so capping against
# it keeps the written sounding consistent with how the model will read it back.

qsat_mean = Springsteel.q_sat_liquid.(T_mean, p_mean ./ 100.0)   # kg/kg
cap_mask = qv_mean .> qsat_mean
n_capped = count(cap_mask)
max_excess_gkg = 0.0
if n_capped > 0
    max_excess_gkg = maximum(qv_mean[cap_mask] .- qsat_mean[cap_mask]) * 1000.0
    println("Saturation cap: $n_capped level(s) capped at q_sat_liquid(T,p); " *
            "max excess = $(round(max_excess_gkg; digits = 4)) g/kg")
    qv_mean[cap_mask] .= qsat_mean[cap_mask]
else
    println("Saturation cap: 0 levels exceeded q_sat_liquid(T,p)")
end

# ── SST resolution: netCDF attr -> "RCE:" setup line -> --sst ─────────────────────────

function resolve_sst(root::String, last_nc::String, sst_cli)
    sst = nothing
    src = ""
    for key in ("sfc_sst", "SST", "sst")
        if haskey(last_attrs, key)
            sst = Float64(last_attrs[key]); src = "netCDF global attr \"$key\""
            break
        end
    end
    if sst === nothing
        logf = joinpath(root, "seg_01", "segment_stdout.log")
        if isfile(logf)
            for line in eachline(logf)
                if occursin("RCE:", line)
                    m = match(r"SST\s*=\s*([0-9.eE+-]+)\s*K", line)
                    if m !== nothing
                        sst = parse(Float64, m.captures[1])
                        src = "seg_01/segment_stdout.log \"RCE:\" setup line"
                        break
                    end
                end
            end
        end
    end
    if sst === nothing
        sst_cli === nothing && error("rce_sounding.jl: SST not found in the netCDF attrs " *
            "or seg_01/segment_stdout.log; pass --sst <K>")
        sst = sst_cli; src = "--sst"
    end
    return sst, src
end

sst, sst_src = resolve_sst(root, window_samples[end].path, opts.sst_cli)
println("SST resolved from $sst_src: $sst K")

# ── Output path ────────────────────────────────────────────────────────────────────────

mode_m = match(r"ocean_rce_([^_]+)_mc_rirk", basename(root))
mode_str = mode_m === nothing ? "unknown" : mode_m.captures[1]
default_out = joinpath(RCE_REF_DIR, "rce_$(round(Int, sst))K_$(mode_str).ref")
out_path = opts.out === nothing ? default_out : opts.out
mkpath(dirname(out_path))

# ── Write the sounding ───────────────────────────────────────────────────────────────

p_sfc_hpa = p_mean[1] / 100.0
theta0 = theta_mean[1]
qv0_gkg = qv_mean[1] * 1000.0

function read_dunion_rows(path::String)
    lines = readlines(path)
    length(lines) >= 2 || error("rce_sounding.jl: splice file $path has no data rows")
    rows = NTuple{5,Float64}[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        parts = split(line)
        u = length(parts) >= 4 ? parse(Float64, parts[4]) : 0.0
        v = length(parts) >= 5 ? parse(Float64, parts[5]) : 0.0
        push!(rows, (parse(Float64, parts[1]), parse(Float64, parts[2]),
                     parse(Float64, parts[3]), u, v))
    end
    return rows
end

isfile(opts.splice) || error("rce_sounding.jl: --splice file not found: $(opts.splice)")
dunion_rows = read_dunion_rows(opts.splice)
kMax = z[end]
splice_rows = filter(r -> r[1] > kMax, dunion_rows)
isempty(splice_rows) && error("rce_sounding.jl: no splice rows with alt > kMax = $kMax m " *
    "in $(opts.splice) -- the TC's Z_TOP would not be bracketed by this sounding")

open(out_path, "w") do io
    @printf(io, "%.4f %.4f %.4f\n", p_sfc_hpa, theta0, qv0_gkg)
    for i in 2:nz
        @printf(io, "%.4f %.4f %.5f %.2f %.2f\n",
                z[i], theta_mean[i], qv_mean[i] * 1000.0, u_mean[i], v_mean[i])
    end
    for r in splice_rows
        @printf(io, "%.4f %.4f %.5f %.2f %.2f\n", r[1], r[2], r[3], r[4], r[5])
    end
end
println("Wrote $(nz - 1 + 1 + length(splice_rows)) level(s) (1 surface + $(nz - 1) RCE + " *
        "$(length(splice_rows)) spliced) to $out_path")

z_full = vcat(z, [r[1] for r in splice_rows])
issorted(z_full) || error("rce_sounding.jl: sounding z is not monotone increasing after " *
                          "splicing (check --splice's kMax overlap)")
theta_full = vcat(theta_mean, [r[2] for r in splice_rows])
qv_full = vcat(qv_mean, [r[3] / 1000.0 for r in splice_rows])   # kg/kg

# ── Round-trip check: rebuild the RCE grid, re-read the file we just wrote ────────────

function options_from_attrs(attrs)
    o = Dict{Symbol,Any}()
    haskey(attrs, "rain_moments") && (o[:rain_moments] = Int(attrs["rain_moments"]))
    icemp = get(attrs, "ice_microphysics", "none")
    icemp != "none" && (o[:ice_microphysics] = Symbol(icemp))
    for k in ("condensate_transform", "rain_transform", "rain_number_transform", "ice_transform")
        haskey(attrs, k) && (o[Symbol(k)] = Symbol(attrs[k]))
    end
    o[:mynn] = get(attrs, "boundary_layer", "") == "mynn"
    return o
end

function rebuild_rce_grid(x, z, attrs)
    num_cells_i = length(x) - 1
    iMax = x[end]
    num_cells_k = length(z) - 1
    kMaxg = z[end]
    o = options_from_attrs(attrs)
    vars = Scythe.mc_var_names(o)
    rain_name = Scythe.rain_var_name(o)
    rain_number_name = Scythe.rain_number_var_name(o)
    two_moment = Scythe.rain_moments(o) == 2
    ice_on = Scythe.ice_microphysics(o) === :ishmael
    ice_names = ice_on ? Scythe.ice_var_names(o) : String[]

    vert_scalar = Dict{String,Any}(v => SecondDerivativeBC() for v in vars)
    topbot_bc = merge(vert_scalar,
                      Dict{String,Any}("w" => DirichletBC(), rain_name => NaturalBC()))
    two_moment && (topbot_bc[rain_number_name] = NaturalBC())
    if ice_on
        for nm in ice_names
            topbot_bc[nm] = NaturalBC()
        end
    end
    positivity = Dict{String,Any}()
    rain_name == "rho_r" && (positivity["rho_r"] = Dict(:k => 0.0))
    side_bc = Dict{String,Any}(v => PeriodicBC() for v in vars)

    # GridParameters (SpringsteelGridParameters) is immutable and leaves kDim/b_kDim etc.
    # at their auto (0) default until compute_derived_params resolves them -- normally done
    # for you by ModelParameters's inner constructor (Scythe.jl `Scythe.jl:73`), which is not
    # in play here since we only need a grid to build a reference_column.
    grid_params = Springsteel.compute_derived_params(GridParameters(;
        geometry = "RiRk",
        iMin = 0.0, iMax = iMax, num_cells_i = num_cells_i,
        kMin = 0.0, kMax = kMaxg, num_cells_k = num_cells_k,
        l_q = Dict("default" => 2.0),
        positivity = positivity,
        BCL = side_bc, BCR = side_bc, BCB = topbot_bc, BCT = topbot_bc,
        vars = Dict(v => i for (i, v) in enumerate(vars))))

    patch = createGrid(grid_params)
    kDim = grid_params.kDim
    z_mish = Scythe.getGridpoints(patch)[1:kDim, 2]
    column = Scythe.reference_column(patch, grid_params)
    return z_mish, column
end

function lininterp(x0::Vector{Float64}, y0::Vector{Float64}, xq::Vector{Float64})
    n0 = length(x0)
    out = zeros(Float64, length(xq))
    for (idx, xi) in enumerate(xq)
        if xi <= x0[1]
            out[idx] = y0[1]
        elseif xi >= x0[end]
            out[idx] = y0[end]
        else
            j = clamp(searchsortedlast(x0, xi), 1, n0 - 1)
            x1, x2 = x0[j], x0[j+1]
            y1, y2 = y0[j], y0[j+1]
            out[idx] = y1 + (xi - x1) * (y2 - y1) / (x2 - x1)
        end
    end
    return out
end

z_mish, column = rebuild_rce_grid(first_prof.x, z, last_attrs)
hydro = reference_state_hydrostatic()
ref = Springsteel.calculate_pressure_reference_state(out_path, z_mish, column;
                                                     hydrostatic = hydro)
pbar_mish = Springsteel.ref_pressure(ref)[:, 1]
rho_d_mish = Springsteel.ref_rho_d(ref)[:, 1]
rho_v_mish = Springsteel.ref_rho_v(ref)[:, 1]
Tbar_mish = Springsteel.reference_temperature(ref)
# Springsteel.p_0 = 1000.0 is in hPa (the Exner-function reference pressure), while
# ref_pressure returns Pa -- convert before applying the Exner relation.
theta_mish = Tbar_mish .* (Springsteel.p_0 ./ (pbar_mish ./ 100.0)) .^
             (Springsteel.Rd / Springsteel.Cpd)
qv_mish = rho_v_mish ./ rho_d_mish

theta_expected = lininterp(z_full, theta_full, z_mish)
qv_expected = lininterp(z_full, qv_full, z_mish)
p_expected = lininterp(z, p_mean, z_mish)   # only over the RCE's own measured range

max_dtheta = maximum(abs.(theta_mish .- theta_expected))
max_dqv_gkg = maximum(abs.(qv_mish .- qv_expected)) * 1000.0
max_dp_rel = maximum(abs.(pbar_mish .- p_expected) ./ p_expected)

println("Round trip (hydrostatic = $hydro): max|dtheta| = $(round(max_dtheta; sigdigits=3)) K, " *
        "max|dq_v| = $(round(max_dqv_gkg; sigdigits=3)) g/kg, " *
        "max|dp|/p = $(round(100.0 * max_dp_rel; sigdigits=3)) %")

# ── SST - T_air check ──────────────────────────────────────────────────────────────────

t_air_sfc = T_mean[1]
delta_sst = sst - t_air_sfc
println("Surface: sounding T(0) = $(round(t_air_sfc; digits = 3)) K, SST = $(round(sst; digits = 3)) K, " *
        "SST - T_air = $(round(delta_sst; digits = 3)) K")
if delta_sst <= 0.0
    println("!!!! WARNING: SST - T_air is NOT positive -- the TC's surface enthalpy/moisture " *
            "flux sign will REVERSE at this sounding's surface state (see the project's " *
            "surface-flux-reversal note). Do not ship this sounding for TC init as-is. !!!!")
end

# ── Metadata sidecar ─────────────────────────────────────────────────────────────────

window_segs = unique(s.seg for s in window_samples)
shas = unique(String.(segs.scythe_sha[[findfirst(==(k), segs.segment) for k in window_segs]]))
meta_path = out_path * ".meta.txt"
open(meta_path, "w") do io
    println(io, "source root: $root")
    println(io, "window: absolute t in [$(t_start), $(t_end)] s, $(length(window_samples)) file(s), " *
                "requested --avg-days $(opts.avg_days) (available $(round(available_days; digits=4)) d)")
    println(io, "scythe_sha: $(join(shas, ", "))")
    println(io, "splice file: $(opts.splice) ($(length(splice_rows)) row(s) with alt > $kMax m)")
    println(io, "saturation cap: $n_capped level(s), max excess $(round(max_excess_gkg; digits=4)) g/kg")
    println(io, "round trip (hydrostatic=$hydro): max|dtheta|=$(max_dtheta) K, " *
                "max|dq_v|=$(max_dqv_gkg) g/kg, max|dp|/p=$(100.0 * max_dp_rel) %")
    println(io, "surface: T(0)=$(t_air_sfc) K, SST=$sst K (source: $sst_src), " *
                "SST-T_air=$(delta_sst) K")
end
println("Wrote metadata to $meta_path")
