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
#   --restart T         continue from the T-second JLD2 restarts in each nest's
#                       output dir (warm restart: the AB3 history is rebuilt)

using Distributed

integration_time = 1800.0
csv = false
restart_t = nothing
let args = copy(ARGS)
    i = 1
    while i <= length(args)
        if args[i] == "--csv"
            global csv = true
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
addprocs(sum(NEST_WORKERS))
@everywhere using Springsteel
@everywhere using Scythe
using CSV, DataFrames

include(joinpath(@__DIR__, "tc_init.jl"))

base = make_base(integration_time;
                 output_formats = csv ? [:csv, :netcdf] : OUTPUT_FORMATS)
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
