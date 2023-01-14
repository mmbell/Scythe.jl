__precompile__()
module Scythe

using Distributed
using DistributedData

include("CubicBSpline.jl")
include("Fourier.jl")
include("Chebyshev.jl")
include("SpectralGrid.jl")
include("NumericalModels.jl")
include("Integrator.jl")

export integrate_model
export CubicBSpline, Chebyshev

function integrate_model(model::ModelParameters)

    if workers()[1] == 1
        throw(ErrorException("Need to add at least 1 worker process"))
    end
    
    println("Starting model...")
    wait(save_at(workers()[1], :out, :(open("scythe_out.log","w"))))
    wait(save_at(workers()[1], :err, :(open("scythe_err.log","w"))))
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