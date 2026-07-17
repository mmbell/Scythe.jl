#!/usr/bin/env julia
# Diagnostic rerun of the final hour of the 2026-07-16 6-h axisym TC run, which
# died between 18000 s and 21600 s with a log DomainError in the Louis-BL
# entropy staging (a rho_d positive-definiteness undershoot; the state at the
# 18000-s output was still healthy). Restarts each nest from its 18000-s
# physical CSV (warm restart: the AB3 history is rebuilt, small transient) into
# a separate output dir, with 300-s CSV output and the per-step rho_d minimum
# trace enabled so the runaway's onset time, location, and growth are captured.
#
#   julia --project=. tc/tc_debug_restart.jl [integration_time_s] [tag] [opt=val ...]
#
# `tag` names the output dir suffix (tc_debug_<tag>; default plain tc_debug).
# Remaining args toggle model options for process-bisection experiments, e.g.
#   julia --project=. tc/tc_debug_restart.jl 1800 noprecip precipitation=false
#
# Reads from tc/output/tc_axisym (must contain the 18000-s CSVs and
# tc_exact.ref); writes to tc/output/tc_debug[_<tag>].

using Distributed

integration_time = isempty(ARGS) ? 3600.0 : parse(Float64, ARGS[1])
run_tag = length(ARGS) >= 2 ? ARGS[2] : ""
opt_overrides = Dict{Symbol,Any}()
ts_scale = 1.0
for a in ARGS[3:end]
    k, v = split(a, "=")
    if k == "ts_scale"
        global ts_scale = parse(Float64, v)
    else
        opt_overrides[Symbol(k)] = v == "true" ? true : v == "false" ? false :
                                   something(tryparse(Int, v), tryparse(Float64, v))
    end
end

include(joinpath(@__DIR__, "tc_params.jl"))
addprocs(sum(NEST_WORKERS))
@everywhere using Springsteel
@everywhere using Scythe
using CSV, DataFrames

include(joinpath(@__DIR__, "tc_init.jl"))

const SRC_DIR = OUTPUT_DIR                                   # tc_axisym
const DEBUG_DIR = replace(OUTPUT_DIR, "tc_axisym" =>
                          isempty(run_tag) ? "tc_debug" : "tc_debug_$(run_tag)")
const RESTART_T = "18000.0"

mkpath(DEBUG_DIR)
# The reference file and the per-nest 18000-s states from the crashed run.
cp(joinpath(SRC_DIR, "tc_exact.ref"), joinpath(DEBUG_DIR, "tc_exact.ref");
   force = true)
for i in 1:length(NEST_CELLS)
    src = joinpath(SRC_DIR, "nest$(i)", "$(RESTART_T)_physical.csv")
    isfile(src) || error("restart state not found: $src")
    cp(src, joinpath(DEBUG_DIR, "nest$(i)_tc_restart.csv"); force = true)
end

base = make_base(integration_time;
                 output_formats = [:csv],
                 output_dir = DEBUG_DIR,
                 initial_conditions = joinpath(DEBUG_DIR, "tc_restart.csv"),
                 output_interval = 300.0,
                 # Trace cadence in steps: 40 steps = 30 s on nest1 (ts 0.75),
                 # 60 s on nests 2-3 (ts 1.5); any tile min rho_d below half its
                 # reference prints immediately regardless of cadence.
                 extra_options = merge(Dict{Symbol,Any}(:state_minima_trace => 40),
                                       opt_overrides))
# ts_scale: uniform timestep scaling for the Courant-dependence experiments
# (sub-cycle ratios are preserved since every patch scales together). The root
# (outer) ts on `base` must scale with it.
if ts_scale != 1.0
    base = ModelParameters(; (f => f == :ts ? base.ts * ts_scale : getfield(base, f)
                              for f in fieldnames(ModelParameters))...)
end
nest = make_nest(base)
if ts_scale != 1.0
    nest = NestedModelParameters(boundaries = nest.boundaries, num_cells = nest.num_cells,
                                 ts = nest.ts .* ts_scale,
                                 workers_per_patch = nest.workers_per_patch, base = base)
end
models, _ = build_nest(nest)

t_wall = @elapsed integrate_nested_model(nest)
println("Debug restart ($(integration_time) s from t=$(RESTART_T)) wall clock: " *
        "$(round(t_wall, digits=1)) s")
