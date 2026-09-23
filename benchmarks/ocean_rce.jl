#!/usr/bin/env julia
# Open-ocean radiative-convective-equilibrium (RCE) case on a 2D (x-z) periodic domain.
# Built from benchmarks/ocean_warm_bubble.jl (the MYNN-EDMF / boundary-layer benchmark):
# same grid family (RiRk), same physics stack, but PERIODIC lateral BCs, seeded low-level
# temperature noise instead of a warm bubble, diurnal all-sky RRTMGP radiation, ISHMAEL ice,
# and MYNN-EDMF with the :scythe fidelity bundle + stability-aware surface layer as the
# default (not opt-in) boundary layer. There is no bubble, no wave, and no lateral wall to
# trigger convection from -- the noise IS the trigger (Scythe.moist_temperature_noise_mc!).
#
# This is NOT a benchmarks/common `run_benchmark` case: there is no published target and no
# committed regression reference (an RCE equilibrium is a many-day statistical statement, not
# a bitwise one). It reuses only a few pieces of benchmarks/common/harness.jl -- CLI parsing,
# grid/timestep sizing helpers, worker setup, REFERENCE_DATA_DIR -- and drives the model
# itself with its own multi-day, multi-process SEGMENT loop.
#
#   julia --project=. benchmarks/ocean_rce.jl --mode quick --stage mc --grid rirk \
#       --days 20 --segment-days 2
#
# Multi-day integration is expensive and long-running, so it is split into SEGMENTS: one
# fresh Julia process per segment, each warm-restarting from the previous segment's JLD2
# checkpoint (write_restart's guaranteed final checkpoint, src/io.jl / src/semiimplicit.jl
# finalize_model). Running with no --segment flag is the ORCHESTRATOR: it spawns one child
# `julia ... ocean_rce.jl ... --segment k` per segment and waits for each to finish before
# starting the next. Running WITH --segment k is a CHILD: it builds and integrates exactly
# that one segment, then exits. The orchestrator and the child share one CLI/env parse, so
# a child invocation is reproducible standalone (e.g. to re-run a failed segment by hand).
#
# Segment CLI (in addition to the benchmarks/common/harness.jl flags --mode/--stage/--grid/
# --workers/--worker-threads/--ts-factor/--nests/--hsi/--exact-si):
#   --days N            total integration [days] (default 20.0)
#   --segment-days D     length of one segment [days] (default 2.0)
#   --segment k          (internal) run segment k only, as a child process
#   --resume             skip segments already marked "done" in segments.csv
#   --csv                append :csv to options[:output_formats] (netcdf is the default and
#                        is always on; CSV is for a replay/diagnostics harness, not this file)
#   --tag <s>            appended to the output root's directory name (also honours the
#                        usual SCYTHE_BENCH_TAG env var; both may be combined)
#
# Output root: benchmarks/output/ocean_rce_<mode>_mc_rirk<tag>/, holding:
#   rce_exact.ref     the reference-state sounding (written once, by segment 1's init)
#   rce_ics.csv       segment 1's initial conditions (noise-seeded)
#   segments.csv      one row per segment: dir, time offset, status, provenance
#   seg_01/ seg_02/ … one comprehensive-NetCDF <t>.nc output tree + JLD2 checkpoints per segment
#
# Physics knobs (every one is a NO-OP when unset, mirroring ocean_warm_bubble's SCYTHE_OWB_*
# names as SCYTHE_RCE_*; see that file's header for the physics each one turns on):
#   SCYTHE_RCE_BL=mynn|louis|none     boundary layer (default mynn, NOT louis -- ice is the
#                                     RCE default and Louis + ice is refused by the driver)
#   SCYTHE_RCE_MYNN_EDMF=1            EDMF mass-flux plumes (needs BL=mynn)
#   SCYTHE_RCE_MYNN_FIDELITY=scythe   named deviation(s) or a fidelity name (default scythe,
#                                     which is also the MODEL default since 2026-09-23;
#                                     "fortran" is the verbatim-Fortran control). Set here
#                                     explicitly either way, so every RCE output file
#                                     records the fidelity it ran with.
#   SCYTHE_RCE_MYNN_INTERVAL=20       MYNN closure cadence [s]
#   SCYTHE_RCE_SFC=komori|gfdl_v7|charnock   surface roughness closure (default komori)
#   SCYTHE_RCE_SFC_STAB=0             stability functions are ON by default HERE (unlike OWB);
#                                     set to 0 to turn them off. :scythe fidelity with
#                                     stability off is allowed -- the bundle then drops
#                                     :rmol_sfc and has three members, not an error.
#   SCYTHE_RCE_RAD=diurnal|lw|allsky|sw|allsky_sw|0   RRTMGP arm (default diurnal; 0 = off)
#   SCYTHE_RCE_ICE=0                  warm rain instead of ISHMAEL ice (default ON here)
#   SCYTHE_RCE_SST=302.65             sea surface temperature [K]
#   SCYTHE_RCE_WIDTH=300.0e3          domain width [m] (periodic; 600e3 was the shakedown width)
#   SCYTHE_RCE_TS, _NI, _NK           grid/timestep overrides
#   SCYTHE_RCE_OUTPUT=3600.0          analysis output cadence [s]
#   SCYTHE_RCE_NOISE_DT=0.1           noise amplitude [K]
#   SCYTHE_RCE_NOISE_ZMAX=1000.0      noise vertical taper depth [m]
#   SCYTHE_RCE_NOISE_SEED=2026        noise RNG seed (segment 1 only)
#   SCYTHE_RCE_WALL_BC=d2|neumann     ground/lid scalar condition (default d2)
#   SCYTHE_RCE_SIDECARS=1             turn on the old MYNN/radiation sidecar NetCDFs (the
#                                     comprehensive <t>.nc always carries the BL/radiation/
#                                     surface physics groups regardless of this)
#   SCYTHE_BENCH_TAG                  free-text output-root suffix (combines with --tag)

using Distributed
using Dates

include(joinpath(@__DIR__, "common", "harness.jl"))

const DUNION_SOUNDING = joinpath(REFERENCE_DATA_DIR, "o01_rainfall", "dunion_MT_hum90.ref")

envflag(k) = get(ENV, k, "0") in ("1", "true", "yes")
env_on_unless_zero(k, default) = get(ENV, k, default) != "0"

# ── RCE-specific CLI (segments, resume, csv, tag) ───────────────────────────────
# Parsed BEFORE the harness's parse_benchmark_args, which never sees these flags.

function parse_rce_args(args::Vector{String})
    days = 20.0
    segment_days = 2.0
    segment = 0          # 0 = orchestrator; >=1 = child running that segment
    resume = false
    csv = false
    tag = ""
    rest = String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--days"
            days = parse(Float64, args[i+1]); i += 2
        elseif a == "--segment-days"
            segment_days = parse(Float64, args[i+1]); i += 2
        elseif a == "--segment"
            segment = parse(Int, args[i+1]); i += 2
        elseif a == "--resume"
            resume = true; i += 1
        elseif a == "--csv"
            csv = true; i += 1
        elseif a == "--tag"
            tag = args[i+1]; i += 2
        else
            push!(rest, a); i += 1
        end
    end
    days > 0.0 || error("--days must be > 0")
    segment_days > 0.0 || error("--segment-days must be > 0")
    segment >= 0 || error("--segment must be >= 1 (0 is reserved for the orchestrator)")
    return (days = days, segment_days = segment_days, segment = segment,
            resume = resume, csv = csv, tag = tag, rest = rest)
end

"""Length [days] of segment `j` out of a `N`-day run split into `D`-day segments."""
seg_length(j::Int, N::Float64, D::Float64) = min(D, N - (j - 1) * D)

"""Cumulative model-time offset [days] before segment `k` starts."""
seg_offset(k::Int, N::Float64, D::Float64) = sum(seg_length(j, N, D) for j in 1:(k - 1); init = 0.0)

rce = parse_rce_args(ARGS)
opts = parse_benchmark_args(rce.rest)
opts.stage == STAGE_MC || error("ocean_rce supports only --stage mc (the total-energy set)")
opts.nests == 1 || error("ocean_rce supports only --nests 1 (nesting is out of scope here)")
opts.grid == :rirk || error("ocean_rce targets the RiRk (B-spline vertical) grid")

nseg_total = ceil(Int, rce.days / rce.segment_days)

tag_suffix = get(ENV, "SCYTHE_BENCH_TAG", "") * (isempty(rce.tag) ? "" : "_$(rce.tag)")
const RCE_ROOT = joinpath(BENCHMARKS_DIR, "output",
                          "ocean_rce_$(opts.mode)_$(opts.stage)$(grid_suffix(opts))" *
                          "$(nest_suffix(opts))$(tag_suffix)")

# ── segments.csv bookkeeping (shared by orchestrator and child) ────────────────

const SEGMENTS_CSV_COLS = ["segment", "dir", "t_offset_s", "integration_time_s", "ts",
                           "output_interval_s", "restart_file", "status", "scythe_sha",
                           "started", "finished"]

function segments_csv_path()
    joinpath(RCE_ROOT, "segments.csv")
end

"""Read the recorded status of segment `k` from `segments.csv`, or `""` if absent."""
function segment_status(k::Int)
    path = segments_csv_path()
    isfile(path) || return ""
    df = CSV.read(path, DataFrame)
    idx = findfirst(==(k), df.segment)
    idx === nothing ? "" : String(df.status[idx])
end

"""Insert or replace the row for `row["segment"]` in `segments.csv`, rewriting the whole
file sorted by segment number. Small (one row per segment of a multi-day run), so a
read-modify-write on every status transition (running -> done/failed) is simplest and
cheapest way to keep the file always valid to read (no half-written append)."""
function upsert_segment_row!(row::Dict{String,Any})
    path = segments_csv_path()
    rows = Dict{Int,Dict{String,Any}}()
    if isfile(path)
        df = CSV.read(path, DataFrame)
        for r in eachrow(df)
            rows[Int(r.segment)] = Dict{String,Any}(c => r[Symbol(c)] for c in SEGMENTS_CSV_COLS)
        end
    end
    rows[Int(row["segment"])] = row
    ks = sort(collect(keys(rows)))
    out = DataFrame(; (Symbol(c) => [rows[k][c] for k in ks] for c in SEGMENTS_CSV_COLS)...)
    CSV.write(path, out)
    return path
end

# ── Orchestrator ─────────────────────────────────────────────────────────────

if rce.segment == 0
    mkpath(RCE_ROOT)
    if !rce.resume && isdir(joinpath(RCE_ROOT, "seg_01"))
        error("refusing a fresh start: $(RCE_ROOT)/seg_01 already exists. Move the whole " *
              "tree aside (mv, never rm) if you want a clean run, or pass --resume to " *
              "continue it from segments.csv.")
    end

    println("RCE orchestrator: $(rce.days) day(s) in $(nseg_total) segment(s) of " *
            "$(rce.segment_days) day(s); root = $RCE_ROOT")

    project_dir = dirname(Base.active_project())
    thisfile = abspath(@__FILE__)
    for k in 1:nseg_total
        if rce.resume && segment_status(k) == "done"
            println("RCE: segment $k/$nseg_total already done -- skipping (--resume)")
            continue
        end
        child_args = vcat(ARGS, ["--segment", string(k)])
        # NOTE: interpolate the Vector{String} WITHOUT a `...` splat -- backtick command
        # literals treat `$(v...)` as concatenating every element into ONE argument with
        # no separator (verified: `` `julia foo.jl $(["a","b"]...)` `` prints as a single
        # "ab" argument), while a bare `$v` expands each element as its own argument.
        cmd = `$(Base.julia_cmd()) --project=$(project_dir) $(thisfile) $child_args`
        println("RCE: launching segment $k/$nseg_total")
        println("  $cmd")
        # The child's stdout (setup line, POSITIVITY line, the workers' "mynn:" / "mynn
        # census:" lines) is tee'd into the segment's own directory -- the model's
        # scythe_out.log only captures the worker's redirected stdout after integrate_model
        # takes over, so without this the master-side prints of a segment are lost once the
        # sbatch log scrolls. stderr goes to the same file.
        seg_dir_k = joinpath(RCE_ROOT, "seg_$(lpad(k, 2, '0'))")
        mkpath(seg_dir_k)
        seg_log = joinpath(seg_dir_k, "segment_stdout.log")
        ok = open(seg_log, "a") do errio
            println(errio, "=== RCE segment $k/$nseg_total launched $(Dates.now()) ===")
            flush(errio)
            success(pipeline(pipeline(cmd; stderr = errio), `tee -a $seg_log`))
        end
        if !ok
            error("RCE: segment $k/$nseg_total FAILED (see $seg_log). " *
                  "Fix the issue and re-run this same command with --resume to continue " *
                  "from segment $k.")
        end
        println("RCE: segment $k/$nseg_total complete.")
    end
    println("RCE: all $nseg_total segment(s) complete. Root: $RCE_ROOT")
    exit(0)
end

# ── Child: build and integrate exactly one segment ──────────────────────────

k = rce.segment
(1 <= k <= nseg_total) || error("--segment $k is out of range for --days $(rce.days) " *
                                "--segment-days $(rce.segment_days) ($(nseg_total) segment(s))")

add_benchmark_workers(opts; count = opts.workers)
@everywhere using Springsteel
@everywhere using Scythe

mkpath(RCE_ROOT)
seg_dir = joinpath(RCE_ROOT, "seg_$(lpad(k, 2, '0'))")
mkpath(seg_dir)

Dk = seg_length(k, rce.days, rce.segment_days)
t_offset_days = seg_offset(k, rce.days, rce.segment_days)
integration_time = Dk * 86400.0
# The segment must be a whole number of steps: integrate_model rounds
# integration_time/ts to the step count, so a fractional segment integrates PAST the nominal
# length (the final output lands at n_steps*ts) while the checkpoint file and the next
# segment's t_offset carry the nominal value -- a silent clock skew. Checked after ts and
# output_interval are known (below), where the output interval is held to the same rule.
whole_steps(len, dt) = abs(len / dt - round(len / dt)) < 1.0e-9

ref_state_file = joinpath(RCE_ROOT, "rce_exact.ref")
if k == 1
    initial_conditions = joinpath(RCE_ROOT, "rce_ics.csv")
else
    Dprev = seg_length(k - 1, rce.days, rce.segment_days)
    prev_seg_dir = joinpath(RCE_ROOT, "seg_$(lpad(k - 1, 2, '0'))")
    expected = joinpath(prev_seg_dir, "$(string(round(Dprev * 86400.0; digits = 2))).jld2")
    if isfile(expected)
        initial_conditions = expected
    else
        candidates = filter(f -> endswith(f, ".jld2"), readdir(prev_seg_dir))
        isempty(candidates) &&
            error("no .jld2 checkpoint found in $prev_seg_dir to restart segment $k from")
        newest = candidates[argmax([mtime(joinpath(prev_seg_dir, c)) for c in candidates])]
        initial_conditions = joinpath(prev_seg_dir, newest)
        println("RCE: WARNING expected checkpoint $expected not found; the restart-file " *
                "naming assumption (round(integration_time; digits=2)) did not hold for " *
                "segment $(k - 1). Using the newest checkpoint in $prev_seg_dir instead: " *
                "$initial_conditions")
    end
end

# ── Grid / timestep sizing ───────────────────────────────────────────────────

WIDTH = parse(Float64, get(ENV, "SCYTHE_RCE_WIDTH", "300.0e3"))   # user decision 2026-09-22: 300 km for wall clock
if opts.mode == :full
    cellsize_i = 3000.0
    num_cells_k = 84
    kMax = 25.2e3
    ts = 0.25
else
    cellsize_i = 6000.0
    num_cells_k = 50
    kMax = 25.0e3
    ts = 0.5
end
num_cells_i = round(Int, WIDTH / cellsize_i)
haskey(ENV, "SCYTHE_RCE_NI") && (num_cells_i = parse(Int, ENV["SCYTHE_RCE_NI"]))
haskey(ENV, "SCYTHE_RCE_NK") && (num_cells_k = parse(Int, ENV["SCYTHE_RCE_NK"]))
haskey(ENV, "SCYTHE_RCE_TS") && (ts = parse(Float64, ENV["SCYTHE_RCE_TS"]))
kDim = 3 * num_cells_k
ts = vertical_ts(ts, opts)
output_interval = parse(Float64, get(ENV, "SCYTHE_RCE_OUTPUT", "3600.0"))

SST = parse(Float64, get(ENV, "SCYTHE_RCE_SST", "302.65"))
physical_params = Dict{Symbol,Any}(:Khdiff => 0.0, :Kvdiff => 0.0,
                                   :Khdiff_heat => 0.0, :Kvdiff_heat => 0.0,
                                   :Kvdiff_water => 0.0,
                                   :tau_qss => 10.0, :N_r => 1.0e-3, :N_0 => 8.0e6,
                                   :alpha => 0.02, :z_damp => 17.0e3,
                                   :Cd => -1.0, :Ck => 1.0e-3, :U_min => 2.0,
                                   :l_inf => 80.0, :SST => SST)

options = merge(Dict{Symbol,Any}(:semiimplicit => true, :exact_reference_state => true,
                                 :precipitation => true, :vertical_mixing => false,
                                 :output_formats => rce.csv ? [:netcdf, :csv] : [:netcdf]),
                reference_state_options())

# ── Boundary layer ───────────────────────────────────────────────────────────

bl = get(ENV, "SCYTHE_RCE_BL", "mynn")
if bl == "louis"
    options[:louis_bl] = true
    options[:surface_fluxes] = true
elseif bl == "none"
    # no boundary layer at all
elseif bl == "mynn"
    options[:mynn] = true
    options[:surface_fluxes] = true
    haskey(ENV, "SCYTHE_RCE_MYNN_INTERVAL") &&
        (options[:mynn_interval] = parse(Float64, ENV["SCYTHE_RCE_MYNN_INTERVAL"]))
    envflag("SCYTHE_RCE_MYNN_EDMF") && (options[:mynn_edmf] = 1)
    options[:mynn_fidelity] =
        Scythe.parse_mynn_fidelity(get(ENV, "SCYTHE_RCE_MYNN_FIDELITY", "scythe"))
else
    error("SCYTHE_RCE_BL must be louis | none | mynn, got \"$bl\"")
end

sfc_arm = get(ENV, "SCYTHE_RCE_SFC", "komori")
sfc_arm in ("komori", "gfdl_v7", "charnock") ||
    error("SCYTHE_RCE_SFC must be komori | gfdl_v7 | charnock, got \"$sfc_arm\"")
sfc_arm == "komori" || (options[:sfc_z0] = Symbol(sfc_arm))
# Stability-aware surface layer is ON BY DEFAULT here (unlike ocean_warm_bubble, where it is
# opt-in): SCYTHE_RCE_SFC_STAB=0 turns it off. :scythe fidelity with it off is not an error --
# _mynn_resolve_fidelity (src/mynn_state.jl) only folds :rmol_sfc into the bundle when
# options[:sfc_stability] === true, so the bundle just has its other three members.
env_on_unless_zero("SCYTHE_RCE_SFC_STAB", "1") && (options[:sfc_stability] = true)

# ── Ice / rain microphysics (default ON here, unlike ocean_warm_bubble) ─────

if env_on_unless_zero("SCYTHE_RCE_ICE", "1")
    options[:ice_microphysics] = :ishmael
    options[:rain_moments] = 2
    options[:condensate_transform] = :bhyp
    options[:rain_transform] = :bhyp
    options[:rain_number_transform] = :bhyp
    options[:ice_transform] = :bhyp
end

# ── Radiation (default diurnal here, unlike ocean_warm_bubble's default off) ─

rad_arm = get(ENV, "SCYTHE_RCE_RAD", "diurnal")
if rad_arm != "0" && rad_arm != ""
    options[:radiation] = :rrtmgp
    options[:radiation_forcing] = :full
    options[:radiation_interval] = 300.0
    options[:radiation_z_max] = 17.0e3
    physical_params[:sfc_albedo] = 0.06
    physical_params[:sfc_emissivity] = 0.98
    if rad_arm == "lw"
        options[:radiation_method] = :clearsky; options[:solar] = :none
    elseif rad_arm == "allsky"
        options[:radiation_method] = :allsky; options[:solar] = :none
    elseif rad_arm == "sw"
        options[:radiation_method] = :clearsky; options[:solar] = :fixed
    elseif rad_arm == "allsky_sw"
        options[:radiation_method] = :allsky; options[:solar] = :fixed
    elseif rad_arm == "diurnal"
        options[:radiation_method] = :allsky; options[:solar] = :diurnal
        physical_params[:latitude] = 20.0
        physical_params[:start_doy] = 240.0
        # Only start_hour*3600 + t_model enters the solar clock (src/radiation_state.jl
        # solar_state), and t_model is zero-based WITHIN this segment's own integration --
        # so the offset in hours from segment 1's start is what keeps the diurnal cycle
        # continuous across the segment boundary.
        physical_params[:start_hour] = t_offset_days * 24.0
    else
        error("SCYTHE_RCE_RAD = \"$rad_arm\" is not an arm; use diurnal, lw, allsky, sw, " *
              "allsky_sw, or 0/unset")
    end
end

sidecars = envflag("SCYTHE_RCE_SIDECARS")
options[:mynn_output] = sidecars
options[:radiation_output] = sidecars

vars = Scythe.mc_var_names(options)
rain_name = Scythe.rain_var_name(options)
rain_number_name = Scythe.rain_number_var_name(options)
two_moment = Scythe.rain_moments(options) == 2
ice_on = Scythe.ice_microphysics(options) === :ishmael
ice_names = Scythe.ice_var_names(options)

wall_mode = get(ENV, "SCYTHE_RCE_WALL_BC", "d2")
vert_scalar = wall_mode == "d2" ? Dict{String,Any}(v => SecondDerivativeBC() for v in vars) :
              wall_mode == "neumann" ? Dict{String,Any}(v => NeumannBC() for v in vars) :
              error("SCYTHE_RCE_WALL_BC must be d2 | neumann, got \"$wall_mode\"")
topbot_bc = merge(vert_scalar, Dict{String,Any}("w" => DirichletBC(), rain_name => NaturalBC()))
two_moment && (topbot_bc[rain_number_name] = NaturalBC())
if ice_on
    for nm in ice_names
        topbot_bc[nm] = NaturalBC()
    end
end

# Periodic on BOTH lateral sides, for EVERY variable including u and w -- an open-ocean RCE
# domain has no lateral wall at all.
side_bc = Dict{String,Any}(v => PeriodicBC() for v in vars)

# Springsteel refuses an :i positivity bound under a periodic BC, so the OWB two-sided
# ("rho_r" => Dict(:i=>0.0, :k=>0.0)) becomes :k-only here. With ice on (the RCE default)
# the rain slot is a bhyp control variable, not "rho_r", and positivity is dropped entirely
# -- exactly as ocean_warm_bubble's ice arm already does.
positivity = Dict{String,Any}()
if rain_name == "rho_r"
    positivity["rho_r"] = Dict(:k => 0.0)
else
    println("POSITIVITY: dropped \"rho_r\" -- it is carried as a control variable")
end

grid_params = GridParameters(;
    geometry = benchmark_geometry(opts),
    iMin = 0.0,
    iMax = WIDTH,
    num_cells_i = num_cells_i,
    kMin = 0.0,
    kMax = kMax,
    vertical_size(opts; num_cells_k = num_cells_k, kDim = kDim)...,
    l_q = Dict("default" => 2.0),
    positivity = positivity,
    BCL = side_bc,
    BCR = side_bc,
    BCB = topbot_bc,
    BCT = topbot_bc,
    vars = Dict(v => i for (i, v) in enumerate(vars)),
)

whole_steps(integration_time, ts) || error("RCE: segment length $(integration_time) s is not a " *
    "whole number of steps at ts = $(ts) s; choose --segment-days (and --days) so that " *
    "D*86400/ts is an integer")
whole_steps(output_interval, ts) || error("RCE: output interval $(output_interval) s is not a " *
    "whole number of steps at ts = $(ts) s (SCYTHE_RCE_OUTPUT)")
whole_steps(integration_time, output_interval) || error("RCE: segment length " *
    "$(integration_time) s is not a whole number of output intervals ($(output_interval) s); " *
    "the stitched time axis needs each segment to end on an output time")

model = ModelParameters(
    ts = ts,
    integration_time = integration_time,
    output_interval = output_interval,
    restart_interval = 21600.0,
    equation_set = "moist_compressible_XZ",
    initial_conditions = initial_conditions,
    output_dir = seg_dir,
    ref_state_file = ref_state_file,
    grid_params = grid_params,
    physical_params = physical_params,
    options = options,
)

scythe_root = normpath(joinpath(@__DIR__, ".."))
scythe_sha, scythe_dirty = git_info(scythe_root)

println("RCE: segment $k/$(nseg_total)  $(num_cells_i) x $(num_cells_k) cells " *
        "($(WIDTH / num_cells_i / 1000) km x $(kMax / num_cells_k) m), ts = $ts s, " *
        "t_offset = $(t_offset_days) d, length = $Dk d ($integration_time s), " *
        "output every $output_interval s; BL = $bl, ICE = $(ice_on), " *
        "RAD = $rad_arm, SST = $SST K, wall BC = $wall_mode, " *
        "sfc_z0 = $(get(options, :sfc_z0, :komori)), " *
        "sfc_stability = $(get(options, :sfc_stability, false)), " *
        "mynn_fidelity = $(get(options, :mynn_fidelity, :none)); " *
        "vars = $(join(vars, ' '))")
println("RCE: git SHA $scythe_sha$(scythe_dirty ? " (dirty)" : ""); root = $RCE_ROOT")

row = Dict{String,Any}(
    "segment" => k, "dir" => seg_dir, "t_offset_s" => t_offset_days * 86400.0,
    "integration_time_s" => integration_time, "ts" => ts,
    "output_interval_s" => output_interval,
    "restart_file" => joinpath(seg_dir, "$(string(round(integration_time; digits = 2))).jld2"),
    "status" => "running", "scythe_sha" => scythe_sha,
    "started" => string(now()), "finished" => "",
)
upsert_segment_row!(row)

# ── Init (segment 1 only): reference sounding + noise-seeded initial conditions ────

function rce_init!(model::ModelParameters)
    patch = createGrid(model.grid_params)
    gridpoints = Scythe.getGridpoints(patch)
    kDim = model.grid_params.kDim
    z = gridpoints[1:kDim, 2]
    column = Scythe.reference_column(patch, model.grid_params)

    hydro = reference_state_hydrostatic()
    ref_phys = Springsteel.calculate_pressure_reference_state(DUNION_SOUNDING, z, column;
                                                             hydrostatic = hydro)
    pbar = Springsteel.ref_pressure(ref_phys)[:, 1]
    rho_dbar = Springsteel.ref_rho_d(ref_phys)[:, 1]
    rho_vbar = Springsteel.ref_rho_v(ref_phys)[:, 1]
    Scythe.write_exact_ref_mc(model.ref_state_file, z, pbar, rho_dbar, rho_vbar, zeros(kDim))
    ref = Springsteel.exact_pressure_reference_state(model.ref_state_file, z, column;
                                                     hydrostatic = hydro)
    if get(reference_state_options(), :consistent_qss_reference, false)
        ref = Scythe.consistent_qss_reference(ref, z, column)
    end

    pp = model.physical_params
    z_max = parse(Float64, get(ENV, "SCYTHE_RCE_NOISE_ZMAX", "1000.0"))
    dT_max = parse(Float64, get(ENV, "SCYTHE_RCE_NOISE_DT", "0.1"))
    seed = parse(Int, get(ENV, "SCYTHE_RCE_NOISE_SEED", "2026"))
    println("RCE init: reference sounding $DUNION_SOUNDING, noise dT_max=$dT_max K, " *
            "z_max=$z_max m, seed=$seed")

    patch.physical .= 0.0
    Scythe.moist_temperature_noise_mc!(patch, gridpoints, ref;
        z_max = z_max, dT_max = dT_max, seed = seed,
        condensate_transform = Scythe.condensate_transform_mode(model.options),
        condensate_mu = get(pp, :condensate_mu, 1.0e-7),
        rain_transform = Scythe.rain_transform_mode(model.options),
        rain_mu = get(pp, :rain_mu, 1.0e-7))
    Scythe.write_ics_csv(model.initial_conditions, patch, gridpoints)
end

if k == 1
    println("Generating reference state and initial conditions (segment 1)...")
    rce_init!(model)
end

# ── Integrate ────────────────────────────────────────────────────────────────

try
    println("Integrating segment $k/$(nseg_total)...")
    wallclock = @elapsed integrate_model(model)
    @printf("Wall clock: %.1f s\n", wallclock)
    row["status"] = "done"
    row["finished"] = string(now())
    upsert_segment_row!(row)
    println("RCE: segment $k/$(nseg_total) done.")
catch e
    row["status"] = "failed"
    row["finished"] = string(now())
    upsert_segment_row!(row)
    rethrow(e)
end
