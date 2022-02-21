__precompile__()
module AdamsBashforth

using CubicBSpline
using Parameters
using CSV
using DataFrames

#Define some convenient aliases
const real = Float64
const int = Int64
const uint = UInt64

@with_kw struct ModelParameters
    ts::real = 0.0
    integration_time::real = 1.0
    #num_ts::int = 0
    output_interval::real = 1.0
    #output_interval::int = 0
    xmin::real = 0.0
    xmax::real = 0.0
    num_nodes::int = 1
    l_q::real = 2.0
    BCL::Dict = R0
    BCR::Dict = R0
    vars::Dict = Dict("u" => 1)
    equation_set = "1dLinearAdvection"
    initial_conditions = "ic.csv"
end

#include("AdamsBashforth_1D_1var_O2.jl")
#include("AdamsBashforth_1D_1var_O3.jl")
include("AdamsBashforth_1D_multivar_O3.jl")

export initialize, run, finalize, integrate_1dLinearAdvection, integrate_WilliamsSlabTCBL, integrate_model

function integrate_1dLinearAdvection()
    
    model = ModelParameters(
    ts = 0.2,
    num_ts = 480,
    output_interval = 60,
    xmin = -48.0,
    xmax = 48.0,
    num_nodes = 96,
    BCL = CubicBSpline.PERIODIC,
    BCR = CubicBSpline.PERIODIC,
    equation_set = "1dLinearAdvection",
    initial_conditions = "testcase.csv"
    )
   
    spline = initialize(model)
    spline = run(spline, model)
    finalize(spline, model)
end

function integrate_WilliamsSlabTCBL(ics_csv::String)
    
    model = ModelParameters(
        ts = 2.0,
        integration_time = 500.0,
        output_interval = 100.0,
        xmin = 0.0,
        xmax = 1.0e6,
        num_nodes = 2000,
        BCL = Dict("vgr" => CubicBSpline.R0, 
            "u" => CubicBSpline.R1T0, 
            "v" => CubicBSpline.R1T0, 
            "w" => CubicBSpline.R1T0),
        BCR = Dict("vgr" => CubicBSpline.R0, 
            "u" => CubicBSpline.R1T1, 
            "v" => CubicBSpline.R1T1, 
            "w" => CubicBSpline.R1T1),
        equation_set = "Williams2013_TCBL",
        initial_conditions = ics_csv,
        vars = Dict("vgr" => 1, "u" => 2, "v" => 3, "w" => 4)    
    )
   
    splines = initialize(model, 4)
    splines = run(splines, model)
    finalize(splines, model)
end

end
