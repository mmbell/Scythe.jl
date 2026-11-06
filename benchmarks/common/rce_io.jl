# Shared NetCDF reading / segment-stitching helpers for the RCE diagnostics
# (simulations/ocean_2D_rce/rce_diagnostics.jl) and movie (rce_movie.jl) scripts. Both read
# the comprehensive per-time `<t>.nc` files simulations/ocean_2D_rce/run.jl writes under
# <output.dir>/seg_NN/[nest{i}/] (src/run_driver.jl's segmented layout; see
# src/netcdf_output.jl for the file contents). This file only READS existing output -- it
# never runs the model, and neither of its callers should either.
#
# Segment discovery goes through Scythe.read_segments_csv (src/run_driver.jl), the single
# writer of <root>/segments.csv, so a reader here can never drift from the columns the
# driver actually writes.

using NCDatasets
using Statistics
using Scythe

# NCDatasets returns Union{Missing,Float64} for a `_FillValue`'d variable; turn missing
# into NaN so the rest of the code can use plain isnan/isfinite logic (the convention
# tc/tc_bl_assess.jl already uses for the TC comprehensive files).
denan(x) = Float64.(coalesce.(x, NaN))

"""Comprehensive per-time files are named exactly `<t>.nc` (a plain float, e.g.
`864.0.nc`). Opt-in sidecars (`<t>_mynn_i<k>.nc`, `<t>_radiation_i<k>.nc`) must NOT
match -- same regex tc/tc_bl_assess.jl uses for the TC comprehensive files."""
const RCE_NC_RE = r"^([0-9]+(?:\.[0-9]+)?)\.nc$"

struct RceSegment
    segment::Int
    dir::String
    t_offset_s::Float64
    status::String
end

"""Read `<root>/segments.csv` (`src/run_driver.jl`'s bookkeeping file, one row per
segment), sorted by segment number, via `Scythe.read_segments_csv`."""
function read_segments(root::String)
    rows = Scythe.read_segments_csv(root)
    isempty(rows) && error("no segments.csv under $root -- is this a run_simulation " *
                           "(src/run_driver.jl) output root?")
    segs = [RceSegment(k, r["dir"], parse(Float64, r["t_offset_s"]), r["status"])
            for (k, r) in rows]
    sort!(segs, by = s -> s.segment)
    return segs
end

"One cataloged snapshot: absolute time [s], file path, owning segment number."
struct RceSnapshot
    t_abs::Float64
    path::String
    segment::Int
end

"""
    catalog_snapshots(root; patch = 0) -> Vector{RceSnapshot}

Every `<t>.nc` across every segment directory under `root`, stitched onto one absolute
time axis (`t_abs = t_offset_s + t_local`), sorted by `t_abs`. `segments.csv`'s `dir`
column is `<root>/seg_NN` (a single patch); `patch > 0` appends `nest\$patch` for a nested
run's patch `patch` (`src/run_driver.jl segment_patch_dirs`).

Segment 1's `0.0.nc` is the true initial condition (the noise-seeded state before the
first timestep) and is kept. Every LATER segment's `0.0.nc` duplicates the PREVIOUS
segment's final file (each segment warm-restarts from the prior segment's JLD2 checkpoint
and writes its own t=0 snapshot on the way in) and is dropped here, per the output-root
contract. A segment directory that does not exist yet (a run still in progress, or one
that failed before writing anything) is skipped with a warning rather than an error, so
this can be pointed at a live run.
"""
function catalog_snapshots(root::String; patch::Int = 0)
    segs = read_segments(root)
    isempty(segs) && error("segments.csv under $root has no rows")
    snaps = RceSnapshot[]
    for s in segs
        dir = patch > 0 ? joinpath(s.dir, "nest$patch") : s.dir
        if !isdir(dir)
            @warn "segment $(s.segment) directory missing, skipping" dir = dir
            continue
        end
        s.status == "done" ||
            @warn "segment $(s.segment) status = \"$(s.status)\" (not \"done\") -- " *
                  "including whatever <t>.nc files exist there"
        for f in readdir(dir)
            m = match(RCE_NC_RE, f)
            m === nothing && continue
            t_local = parse(Float64, m.captures[1])
            (s.segment > 1 && t_local == 0.0) && continue   # dup of prior segment's final file
            push!(snaps, RceSnapshot(s.t_offset_s + t_local, joinpath(dir, f), s.segment))
        end
    end
    sort!(snaps, by = x -> x.t_abs)
    isempty(snaps) && error("no <t>.nc files found under $root's segment directories")
    return snaps
end

# ── Field readers ──────────────────────────────────────────────────────────────
#
# Every comprehensive-file variable is declared in Julia with dims ("time","x","z") or
# ("time","x") or ("z",) (src/netcdf_output.jl `wr`/`wrx`/`wrz`). NCDatasets.jl keeps
# that JULIA-side declaration order for the array it hands back regardless of what
# `ncdump` prints (ncdump shows the reversed, C-storage-order dim list for a
# column-major writer -- verified against src/netcdf_output.jl and the convention
# tc/tc_bl_assess.jl / tc/tc_movie.jl already rely on: `ds[v][1,:,:]` is `(x,z)`).
field_xz(ds, name) = denan(Array(ds[name])[1, :, :])     # (n_x, n_z)
field_x(ds, name)  = denan(Array(ds[name])[1, :])        # (n_x,)
field_z(ds, name)  = denan(Array(ds[name])[:])           # (n_z,)  reference/bar profiles

has_var(ds, name) = haskey(ds, name)
physics_groups(ds) = get(ds.attrib, "physics_groups", "none")
has_group(ds, name) = occursin(name, physics_groups(ds))

"""x/z regular coordinates [m], and the INTERIOR x index range: `1:(n_x - 1)`, dropping
the last column, which duplicates the first physically under the case's periodic lateral
BCs (the file's `x = LinRange(iMin, iMax, n)` includes both ends -- see the comprehensive-
file contract). The grid is uniform in both x and z (both `LinRange`), so plain means and
trapezoidal integrals with the raw coordinate spacing are exact, not an approximation."""
function xz_coords(ds)
    x = denan(ds["x"][:]); z = denan(ds["z"][:])
    return x, z, 1:(length(x) - 1)
end

"Domain mean over x of a length-n_x vector, dropping the last (periodic-duplicate) column."
xmean(v::AbstractVector, interior) = mean(view(v, interior))

"""Mean of `v` skipping non-finite entries (NaN for a physics-group field on a t=0
snapshot, where `physics_groups = \"none (schemes not yet run)\"` -- see
`benchmarks/rce_diagnostics.jl`). NaN if every entry is non-finite."""
function nanmean(v)
    f = Iterators.filter(isfinite, v)
    n = 0; s = 0.0
    for x in f
        s += x; n += 1
    end
    return n == 0 ? NaN : s / n
end

"""Domain mean over x of an `(x, z)` field at every z, dropping the last column. Returns
a length-n_z vector (a horizontally averaged profile)."""
function xmean_profile(A::AbstractMatrix, interior)
    nz = size(A, 2)
    out = Vector{Float64}(undef, nz)
    @inbounds for k in 1:nz
        out[k] = mean(view(A, interior, k))
    end
    return out
end

"Trapezoidal integral of `f(z)` over a (not necessarily uniform, though here it is) z grid."
function trapz(z::AbstractVector, f::AbstractVector)
    n = length(z)
    n == length(f) || error("trapz: z and f must be the same length ($n vs $(length(f)))")
    s = 0.0
    @inbounds for k in 1:(n - 1)
        s += 0.5 * (f[k] + f[k + 1]) * (z[k + 1] - z[k])
    end
    return s
end

"Ordinary least-squares slope of `y` against `x` (both equal length), by the normal
equations. Used for the equilibrium-check trend lines (column energy vs time, PW vs
time); NaN rows are dropped first."
function ols_slope(x::AbstractVector, y::AbstractVector)
    keep = isfinite.(x) .& isfinite.(y)
    xs = Float64.(x[keep]); ys = Float64.(y[keep])
    n = length(xs)
    n < 2 && return NaN
    xbar = mean(xs); ybar = mean(ys)
    num = sum((xs .- xbar) .* (ys .- ybar))
    den = sum((xs .- xbar) .^ 2)
    den == 0.0 && return NaN
    return num / den
end
