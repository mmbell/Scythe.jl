__precompile__()
module Scythe

using Distributed
using DistributedData
using Statistics

include("CubicBSpline.jl")
include("Fourier.jl")
include("Chebyshev.jl")
include("spectralGrid.jl")
include("numericalModels.jl")
include("thermodynamics.jl")
include("reference_state.jl")
#include("integrator.jl")
include("semiimplicit.jl")
include("io.jl")
include("euler.jl")

export integrate_model
export CubicBSpline, Chebyshev

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