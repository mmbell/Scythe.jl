#!/usr/bin/env julia
# 3-nest axisymmetric TC simulation (milestone 1): balanced modified-Rankine
# vortex on the humidified Dunion MT sounding over a 28 C ocean, with the
# Louis boundary layer, surface enthalpy/moisture fluxes, Smagorinsky Kh and
# Marshall-Palmer warm rain. Nests: 0-150-450-1050 km at 3|6|12 km cells,
# 250 m vertical to 25 km.
#
#   julia --project=. tc/tc_run_axisym.jl [integration_time_s] [--csv] [--restart T]
#
#   integration_time_s  default 1800 (laptop smoke); production 432000 (5 days)
#   --csv               add CSV output alongside NetCDF (short runs/diagnostics)
#   --rlr               run the 3D cylindrical (RLR) nest instead of axisym:
#                       ring-native azimuthal truncation, output in tc_rlr/
#   --restart T         continue from the T-second JLD2 restarts in each nest's
#                       output dir (warm restart: the AB3 history is rebuilt)
#   --exact-si          use the exact 2-D SI instead of the vertical-only SI
#                       (DEBUG ONLY: blows up on the balanced vortex, see
#                       reference/EXACT_SI_VORTEX_FAILURE.md)
#   --trace N           report the min rho_d/rho_dbar column every N steps (and
#                       immediately whenever it drops below 0.5), so a blow-up
#                       leaves a located trace instead of a bare log(negative)

using Distributed

integration_time = 1800.0
csv = false
rlr = false
restart_t = nothing
trace = 0
use_xsi = false
let args = copy(ARGS)
    i = 1
    while i <= length(args)
        if args[i] == "--csv"
            global csv = true
        elseif args[i] == "--rlr"
            global rlr = true
        elseif args[i] == "--exact-si"
            global use_xsi = true
        elseif args[i] == "--trace"
            global trace = parse(Int, args[i+1])
            i += 1
        elseif args[i] == "--restart"
            global restart_t = args[i+1]
            i += 1
        else
            global integration_time = parse(Float64, args[i])
        end
        i += 1
    end
end

include(joinpath(@__DIR__, "tc_params.jl"))
if haskey(ENV, "SCYTHE_SLURM_WORKERS")
    # Node-per-patch shape (scythe_tc_multinode.sbatch): workers land on the
    # allocated nodes via srun; integrate_nested_model partitions workers() in
    # order, so with 1 worker per patch the node<->patch mapping is automatic.
    using ClusterManagers
    addprocs_slurm(parse(Int, ENV["SCYTHE_SLURM_WORKERS"]);
                   exeflags = "--project=$(Base.active_project())")
else
    addprocs(sum(NEST_WORKERS))
end
@everywhere using Springsteel
@everywhere using Scythe
using CSV, DataFrames

include(joinpath(@__DIR__, "tc_init.jl"))

geometry = rlr ? "RLR" : "RiRk"
run_outdir = rlr ? replace(OUTPUT_DIR, "tc_axisym" => "tc_rlr") : OUTPUT_DIR
base = make_base(integration_time;
                 output_formats = csv ? [:csv, :netcdf] : OUTPUT_FORMATS,
                 output_dir = run_outdir, geometry = geometry,
                 extra_options = merge(
                     trace > 0 ? Dict{Symbol,Any}(:state_minima_trace => trace) :
                                 Dict{Symbol,Any}(),
                     # A/B lever: swap the vertical-only state-dependent SI for
                     # the exact 2-D solve (they are mutually exclusive) so the
                     # two can be compared at identical initial conditions.
                     use_xsi ? Dict{Symbol,Any}(:exact_si => true,
                                                :state_dependent_si => false) :
                               Dict{Symbol,Any}()))
nest = make_nest(base)

if restart_t === nothing
    models, topo = init_tc!(nest)
else
    # Chain from the per-nest JLD2 checkpoints: copy them to the IC paths
    # build_nest derives (nest$(i)_<basename>) and point the base at .jld2.
    # Warm restart: the AB3 tendency history is rebuilt (small transient).
    models, _ = build_nest(nest)
    ics_dir = dirname(base.initial_conditions)
    for (i, m) in enumerate(models)
        src = joinpath(m.output_dir, "$(restart_t).jld2")
        isfile(src) || error("restart file not found: $src")
        cp(src, joinpath(ics_dir, "nest$(i)_tc_restart.jld2"); force = true)
    end
    base = make_base(integration_time;
                     output_formats = csv ? [:csv, :netcdf] : OUTPUT_FORMATS,
                     output_dir = run_outdir, geometry = geometry,
                     initial_conditions = joinpath(ics_dir, "tc_restart.jld2"))
    nest = make_nest(base)
end

t_wall = @elapsed integrate_nested_model(nest)
println("Nested TC integration ($(integration_time) s) wall clock: " *
        "$(round(t_wall, digits=1)) s")

# ── Post-run diagnostics (CSV mode only) ─────────────────────────────────────
if csv
    kDim = models[1].grid_params.kDim
    for (i, m) in enumerate(models)
        files = filter(f -> endswith(f, "_physical.csv"), readdir(m.output_dir))
        isempty(files) && continue
        times = sort([parse(Float64, replace(f, "_physical.csv" => "")) for f in files])
        df = CSV.read(joinpath(m.output_dir, "$(times[end])_physical.csv"), DataFrame)
        finite = all(isfinite, Matrix(df[:, TC_VARS]))
        println("nest$i t=$(times[end]): finite=$(finite)" *
                "  max|w|=$(round(maximum(abs.(df.w)); digits=3))" *
                "  max|u|=$(round(maximum(abs.(df.u)); digits=3))" *
                "  max v=$(round(maximum(df.v); digits=2))" *
                "  max rho_r=$(round(1e3 * maximum(df.rho_r); digits=3)) g/m3")
    end
end
