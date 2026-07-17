__precompile__()
module Scythe

# Infrastructure for model grid representation
using Springsteel

# Re-export Springsteel's basis-agnostic BoundaryConditions type system
export BoundaryConditions, NaturalBC, DirichletBC, NeumannBC, SecondDerivativeBC
export RobinBC, PeriodicBC, CauchyBC, ExponentialBC, SymmetricBC, AntisymmetricBC

"""
    ModelParameters

Main configuration struct for Scythe model runs. Uses `Base.@kwdef` for keyword construction.

# Fields
- `ts::Float64`: model timestep [s] (default: `0.0`)
- `integration_time::Float64`: total integration duration [s] (default: `1.0`)
- `output_interval::Float64`: time between analysis output writes [s] (default: `1.0`)
- `restart_interval::Float64`: time between JLD2 restart-checkpoint writes [s] (default: `0.0` = no checkpoints). Written by [`write_restart`](@ref); independent of `output_interval` so checkpoints can be less frequent than analysis output. A resulting `<t>.jld2` can be used as `initial_conditions` to restart. The restart is WARM, not bit-identical: the checkpoint stores the grid state exactly but not the AB3 integrator's tendency history (see [`write_restart`](@ref)).
- `equation_set::String`: name of the equation set to solve (default: `"LinearAdvection1D"`)
- `initial_conditions::String`: path to the initial conditions file (default: `"ic.csv"`)
- `output_dir::String`: path to the output directory (default: `"./output/"`)
- `ref_state_file::String`: path to the reference state sounding file (default: `""`)
- `grid_params::SpringsteelGridParameters`: Springsteel grid configuration (required, no default)
- `physical_params::Dict{Symbol,Float64}`: physical parameters for the equation set (default: empty)
- `options::Dict{Symbol,Any}`: solver and output options (default: `Dict(:semiimplicit => false, :exact_reference_state => false)`). `:semiimplicit` is an opt-in for the LEGACY equation sets only; the moist_compressible (pressure-reference) sets are ALWAYS semi-implicit — the key may be omitted or `true`, and an explicit `false` is an error (the explicit acoustic mode was removed). Output-format keys read by [`write_output`](@ref):
    - `:output_formats::Vector{Symbol}` (default `[:csv]`) — which ANALYSIS formats to write each `output_interval`. `:csv` writes the `<t>_spectral.csv`/`<t>_physical.csv`/`<t>_gridded.csv` trio (Springsteel `write_grid`); `:netcdf` writes the gridded representation to `<t>.nc` (`write_netcdf`). List several to write several, e.g. `[:csv, :netcdf]`. JLD2 is not listed here — it is the restart format, written at `restart_interval` (see `write_restart`). NOTE: the ICs reader, regression references, and benchmark harnesses parse the CSVs, so drop `:csv` only for runs that don't feed them.
    - `:netcdf_derivatives::Bool` (default `false`) — when `:netcdf` is selected, whether to write derivative slots alongside field values.

`grid_params` is passed through Springsteel's `compute_derived_params` on construction, so a
cubic B-spline axis may be sized by *either* its cell count (`num_cells_i`/`num_cells_k`, the
canonical form) or its gridpoint count (`iDim`/`kDim`), and both fields are populated and
mutually consistent afterwards. Without this, supplying only `num_cells_k` would leave
`grid_params.kDim == 0` for every consumer that reads it (the grid factory resolves the counts
onto the grid it returns, not onto the caller's parameter struct).
"""
Base.@kwdef struct ModelParameters
    ts::Float64 = 0.0
    integration_time::Float64 = 1.0
    output_interval::Float64 = 1.0
    # JLD2 restart-checkpoint cadence [s]. 0.0 = no checkpoints. Independent of
    # output_interval so checkpoints (for restart) can be less frequent
    # than analysis output (CSV/NetCDF); see write_restart.
    restart_interval::Float64 = 0.0
    equation_set::String = "LinearAdvection1D"
    initial_conditions::String = "ic.csv"
    output_dir::String = "./output/"
    ref_state_file::String = ""
    grid_params::SpringsteelGridParameters
    # Concretely typed so that `model.physical_params[:Khdiff]` inside a per-column function
    # returns a `Float64` instead of boxing an `Any`. The inner constructor's `new` converts,
    # so a caller passing an integer value (`:K => 75`) still works.
    physical_params::Dict{Symbol,Float64} = Dict{Symbol,Float64}()
    # Deliberately left with an `Any` value type: options are not all Bool (e.g. a numeric
    # :cfl_interval). The handful of option reads that sit inside per-column functions carry
    # a `::Bool` assertion at the read site instead, which is what type-stability needs.
    options::Dict{Symbol,Any} = Dict{Symbol,Any}(
        :semiimplicit => false,
        :exact_reference_state => false)

    function ModelParameters(ts, integration_time, output_interval, restart_interval,
                             equation_set, initial_conditions, output_dir, ref_state_file,
                             grid_params, physical_params, options)
        # Eddy diffusivities are specified directly per quantity, not as molecular-style
        # ratios of the momentum coefficient: :Khdiff/:Kvdiff (momentum),
        # :Khdiff_heat/:Kvdiff_heat (heat, default = momentum values), and
        # :Khdiff_water/:Kvdiff_water (water species, default 0).
        for bad in (:Prandtl, :Schmidt)
            haskey(physical_params, bad) && error(
                "physical_params[:$bad] was removed: eddy mixing coefficients are not " *
                "molecular ratios. Set :Khdiff_heat/:Kvdiff_heat (heat) and " *
                ":Khdiff_water/:Kvdiff_water (water species) directly.")
        end
        new(ts, integration_time, output_interval, restart_interval, equation_set,
            initial_conditions, output_dir, ref_state_file,
            compute_derived_params(grid_params), physical_params, options)
    end
end

# Files for model integration
include("thermodynamics.jl")
include("reference_state.jl")
include("semiimplicit.jl")
include("testModels.jl")
include("shallowWaterModels.jl")
include("tcblModels.jl")
include("primitive_equations.jl")
include("io.jl")
include("microphysics.jl")
include("mc_geometry.jl")
include("mc_boundary_layer.jl")
include("moist_compressible.jl")
include("horizontal_si.jl")
include("idealized.jl")
include("nesting.jl")

# Export the primary driver function and the ModelParameters
export integrate_model
export ModelParameters

"""
    integrate_model(model::ModelParameters)

Main entry point for running a Scythe simulation. Initializes worker processes,
redirects stdout/stderr to log files in the output directory, then runs the model
through its initialize, run, and finalize stages.

Requires at least one additional Julia worker process (added via `addprocs`).

# Arguments
- `model::ModelParameters`: the model configuration specifying equation set, grid, timing, and output options.

# Throws
- `ErrorException` if no worker processes are available.
"""
function integrate_model(model::ModelParameters)

    if workers()[1] == 1
        throw(ErrorException("Need to add at least 1 worker process"))
    end
    
    println("Starting model...")

    # Advisory startup check (master console, before the worker stdout redirect):
    # catch a timestep that is too large for the vertical resolution (e.g. raising
    # kDim without lowering ts). Warn-only; never aborts.
    warn_timestep_stability(model.grid_params, model.ts; equation_set=model.equation_set)

    if !isdir(model.output_dir)
        mkdir(model.output_dir)
    end
    outfile = model.output_dir * "/scythe_out.log"
    errfile = model.output_dir * "/scythe_err.log"
    wait(save_at(workers()[1], :out, :(open($(outfile),"w"))))
    wait(save_at(workers()[1], :err, :(open($(errfile),"w"))))
    wait(get_from(workers()[1], :(redirect_stdout(out))))
    wait(get_from(workers()[1], :(redirect_stderr(err))))
    
    wait(save_at(workers()[1], :patch, :(initialize_model($(model),workers()))))
    wait(get_from(workers()[1], :(@time run_model(patch, model, workers()))))
    wait(get_from(workers()[1], :(finalize_model(patch,model))))
    
    wait(get_from(workers()[1], :(close(out))))
    wait(get_from(workers()[1], :(close(err))))
    println("All done!")
end

# Module end
end