module AdamsBashforth

using CubicBSpline
using Parameters
using CSV
using DataFrames

include 'AdamsBashforth_1D_1var_O2.jl'
include 'AdamsBashforth_1D_multivar_O2.jl'

export initialize, run, finalize, integrate_model

#Define some convenient aliases
const real = Float64
const int = Int64
const uint = UInt64

@with_kw struct ModelParameters
    ts::real = 0.0
    num_ts::int = 1
    output_interval::int = 1
    xmin::real = 0.0
    xmax::real = 0.0
    num_nodes::int = 1
    l_q::real = 2.0
    BCL::Dict = R0
    BCR::Dict = R0
    variables::Dict = Dict("u" => 1)
    equation_set = "1dLinearAdvection"
    initial_conditions = "ic.csv"
end

function integrate_model()
    
    model = ModelParameters(
    ts = 0.2,
    num_ts = 480,
    output_interval = 25,
    xmin = -48.0,
    xmax = 48.0,
    num_nodes = 96,
    BCL = PERIODIC,
    BCR = PERIODIC,
    equation_set = "1dLinearAdvection",
    initial_conditions = "testcase.csv"
    )
   
    spline = initialize(model)
    spline = run(spline, model)
    finalize(spline, model)
end


end