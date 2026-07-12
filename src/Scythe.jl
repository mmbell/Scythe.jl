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
- `output_interval::Float64`: time between output writes [s] (default: `1.0`)
- `equation_set`: name of the equation set to solve (default: `"LinearAdvection1D"`)
- `initial_conditions`: path to the initial conditions file (default: `"ic.csv"`)
- `output_dir`: path to the output directory (default: `"./output/"`)
- `ref_state_file`: path to the reference state sounding file (default: `""`)
- `grid_params::SpringsteelGridParameters`: Springsteel grid configuration (required, no default)
- `physical_params::Dict`: dictionary of physical parameters for the equation set (default: empty `Dict`)
- `options::Dict`: dictionary of solver options (default: `Dict(:semiimplicit => false, :exact_reference_state => false)`)

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
    equation_set = "LinearAdvection1D"
    initial_conditions = "ic.csv"
    output_dir = "./output/"
    ref_state_file = ""
    grid_params::SpringsteelGridParameters
    physical_params::Dict = Dict()
    options::Dict = Dict(
        :semiimplicit => false,
        :exact_reference_state => false)

    function ModelParameters(ts, integration_time, output_interval, equation_set,
                             initial_conditions, output_dir, ref_state_file,
                             grid_params, physical_params, options)
        new(ts, integration_time, output_interval, equation_set, initial_conditions,
            output_dir, ref_state_file, compute_derived_params(grid_params),
            physical_params, options)
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
include("moist_compressible.jl")
include("idealized.jl")

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
    warn_timestep_stability(model.grid_params, model.ts)

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