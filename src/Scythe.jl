__precompile__()
module Scythe

# Infrastructure for model grid representation
using Springsteel

# Structure to define model parameters
Base.@kwdef struct ModelParameters
    ts::Float64 = 0.0
    integration_time::Float64 = 1.0
    output_interval::Float64 = 1.0
    equation_set = "LinearAdvection1D"
    initial_conditions = "ic.csv"
    output_dir = "./output/"
    ref_state_file = ""
    grid_params::GridParameters
    physical_params::Dict = Dict()
    options::Dict = Dict(
        :semiimplicit => false,
        :exact_reference_state => false)
end

# Files for model integration
include("thermodynamics.jl")
include("reference_state.jl")
include("semiimplicit.jl")
include("testModels.jl")
include("shallowWaterModels.jl")
include("tcblModels.jl")
include("io.jl")
include("microphysics.jl")

# Export the primary driver function and the ModelParameters
export integrate_model
export ModelParameters

function integrate_model(model::ModelParameters)

    if workers()[1] == 1
        throw(ErrorException("Need to add at least 1 worker process"))
    end
    
    println("Starting model...")
    
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