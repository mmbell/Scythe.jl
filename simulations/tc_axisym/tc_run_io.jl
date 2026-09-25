# Shared run-tree reading for the TC readers (tc_movie.jl, tc_bl_assess.jl,
# tc_postprocess.jl). READS ONLY -- it never runs the model.
#
# A run ROOT written by the segment driver (src/run_driver.jl) looks like
#
#   <root>/scythe_config.toml     resolved config + provenance header (config.toml in
#                                 trees written before the rename; Scythe.run_config_file)
#   <root>/reference.ref          the reference column (z p rho_d rho_v rho_c)
#   <root>/nest{i}_ics.csv        initial conditions
#   <root>/segments.csv           one row per segment (columns incl. segment, dir,
#                                 t_offset_s, integration_time_s, status)
#   <root>/seg_NN/nest{i}/<t>.nc  analysis output, t LOCAL to the segment
#   <root>/seg_NN/nest{i}/<t>.jld2, scythe_{out,err}.log
#
# Each segment restarts its clock at 0 and writes its own t = 0 snapshot on the way in,
# which duplicates the previous segment's final file; `tc_catalog` stitches the segments
# onto one absolute time axis (t = t_offset_s + t_local) and drops those duplicates
# (the pattern of benchmarks/common/rce_io.jl `catalog_snapshots`, made patch-aware).
#
# A LEGACY tree (tc/tc_run_axisym.jl: `<root>/nest{i}/<t>.nc`, `<root>/tc_exact.ref`, no
# segments.csv) is read as one segment at offset 0, so old runs stay viewable.

using CSV
using DataFrames

"One segment of a run tree: number, directory (resolved), time offset [s], status."
struct TCSegment
    segment::Int
    dir::String
    t_offset::Float64
    status::String
end

"Is `root` a segment-driver tree (has segments.csv) rather than a legacy flat one?"
is_segmented_root(root::AbstractString) = isfile(joinpath(root, "segments.csv"))

"""
    tc_segments(root) -> Vector{TCSegment}

The segments of a run, sorted. The recorded `dir` is absolute; when it does not exist
(the tree was moved) `<root>/<basename(dir)>` is used instead. A legacy tree is one
segment, `dir = root`, offset 0.
"""
function tc_segments(root::AbstractString)
    is_segmented_root(root) || return [TCSegment(1, root, 0.0, "legacy")]
    df = CSV.read(joinpath(root, "segments.csv"), DataFrame; types = String, pool = false)
    segs = TCSegment[]
    for r in eachrow(df)
        dir = coalesce(r.dir, "")
        isdir(dir) || (dir = joinpath(root, basename(dir)))
        push!(segs, TCSegment(parse(Int, r.segment), dir,
                              parse(Float64, coalesce(r.t_offset_s, "0")),
                              coalesce(r.status, "")))
    end
    return sort!(segs, by = s -> s.segment)
end

"""
    tc_npatches(root) -> Int

Number of nest patches: `length(grid.num_cells)` of `<root>/scythe_config.toml` (read with
`Scythe.load_config`) for a segmented tree; the count of `nest{i}` directories for a
legacy one.
"""
function tc_npatches(root::AbstractString)
    if is_segmented_root(root)
        # output.dir overridden to the tree's own location, so a moved tree still loads.
        cfg = Scythe.load_config(Scythe.run_config_file(root); set = ["output.dir=" * root])
        return length(cfg.grid.num_cells)
    end
    n = 0
    while isdir(joinpath(root, "nest$(n + 1)"))
        n += 1
    end
    n > 0 || error("$root is neither a segment-driver run root (no segments.csv) nor a " *
                   "legacy tree (no nest1/)")
    return n
end

"`<root>/reference.ref`, or a legacy tree's `tc_exact.ref`."
function tc_reference_file(root::AbstractString)
    p = joinpath(root, "reference.ref")
    isfile(p) && return p
    legacy = joinpath(root, "tc_exact.ref")
    isfile(legacy) && return legacy
    error("no reference.ref (or legacy tc_exact.ref) under $root")
end

"The comprehensive per-time file name: `<t>.nc` exactly (sidecars and derived files do not match)."
const TC_NC_RE = r"^([0-9]+(?:\.[0-9]+)?)\.nc$"

"""
    tc_catalog(root; nests = 1:tc_npatches(root), re = TC_NC_RE)
        -> Dict{Float64,Dict{Int,String}}

Absolute time [s] => (nest => file) for every file matching `re` (first capture = the
segment-local time) in every segment's `nest{i}/` directory. Segment 1's t = 0 is the
initial condition and is kept; every later segment's t = 0 duplicates the previous
segment's last file and is dropped. A segment directory that does not exist yet (a run
in progress) is skipped with a warning.
"""
function tc_catalog(root::AbstractString; nests = 1:tc_npatches(root), re::Regex = TC_NC_RE)
    cat = Dict{Float64,Dict{Int,String}}()
    for s in tc_segments(root)
        if !isdir(s.dir)
            @warn "segment $(s.segment) directory missing, skipping" dir = s.dir
            continue
        end
        for i in nests
            ndir = joinpath(s.dir, "nest$i")
            isdir(ndir) || continue
            for f in readdir(ndir)
                m = match(re, f)
                m === nothing && continue
                t_local = parse(Float64, m.captures[1])
                (s.segment > 1 && t_local == 0.0) && continue
                get!(cat, s.t_offset + t_local, Dict{Int,String}())[i] = joinpath(ndir, f)
            end
        end
    end
    return cat
end

"Every `nest{i}` directory of the run, one per segment, in segment order: (segment, dir)."
tc_nest_dirs(root::AbstractString, i::Int) =
    [(s, joinpath(s.dir, "nest$i")) for s in tc_segments(root) if isdir(joinpath(s.dir, "nest$i"))]
