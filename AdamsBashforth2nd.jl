# 2nd order Adams-Bashforth implementation

module AdamsBashforth2nd

using CubicBSpline
using Parameters
using CSV
using DataFrames

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

function first_timestep(spline::Spline1D, bdot::Vector{real}, ts::real)
   
    # Use Euler method for first step
    b_nxt = spline.b + (ts * bdot)
    bdot_svd = bdot
    return b_nxt, bdot_svd
end

function timestep(spline::Spline1D, bdot::Vector{real}, bdot_svd::Vector{real}, 
        bdot_delay::Vector{real}, ts::real)
    
    b_nxt = spline.b + (0.5 * ts) * ((3.0 * bdot) - bdot_svd) + (ts * bdot_delay)
    bdot_svd = bdot
    return b_nxt, bdot_svd
end

function calcTendency(spline::Spline1D, model::ModelParameters, t::int)
    
    a = SItransform!(spline)
    
    if mod(t,model.output_interval) == 0
        write_output(spline, model, t)
    end
    
    u = SItransform!(spline)
    ux = SIxtransform(spline)
    uxx = SIxxtransform(spline)
    
    #b_now is held in place
    SBtransform!(spline)
    
    # Feed to physical equations
    udot, F = physical_model(model,u,ux,uxx)
    
    # Do something with F later. May need to be a new spline because of BCs?
    # SBxtransform(spline,F,BCL,BCR)
    bdot_delay = zeros(real,spline.bDim)
    
    bdot = SBtransform(spline,udot)
    return bdot,bdot_delay
end

function physical_model(model::ModelParameters,u::Vector{real},ux::Vector{real},uxx::Vector{real})
    
    if model.equation_set == "1dLinearAdvection"
        #1D Linear advection to test
        c_0 = 1.0
    
        udot = -c_0 .* ux
        F = 0
    elseif model.equation_set == "1dNonlinearAdvection"
        c_0 = 1.0
        K = 0.048
        
        udot = -(c_0 .+ u) .* ux + (K .* uxx)
        F = 0
    else
        throw(MethodError(model.equation_set, "equation set not implemented"))
    end
    
    return udot, F 
end

function write_output(spline::Spline1D, model::ModelParameters, t::int)
   
    afile = open("model_a_$t","w")
    ufile = open("model_u_$t","w")

    a = spline.a    
    for i = 1:spline.aDim
        a_i = a[i]
        write(afile,"$i, $a_i\n")
    end        
    
    SItransform!(spline)
    u = spline.uMish
    mishPoints = spline.mishPoints
    for i = 1:spline.mishDim
        mp_i = mishPoints[i]
        u_i = u[i]
        write(ufile,"$i, $mp_i, $u_i\n")
    end
    close(afile)
    close(ufile)
end

function initialize(model::ModelParameters)
        
    spline = Spline1D(SplineParameters(xmin = model.xmin, xmax = model.xmax,
            num_nodes = model.num_nodes, BCL = model.BCL, BCR = model.BCR))
    
    #ic = CSV.read(model.initial_conditions, DataFrame, "x", "u")
    #if (spline.mishPoints ≉ ic.x)
    #    throw(DomainError(ic.x[1], "mish from IC does not match model parameters"))
    #end
    
    # Hard-code IC for testing
    for i = 1:spline.mishDim
        spline.uMish[i] = exp(-(spline.mishPoints[i])^2 / (2 * 4^2))
    end
        
    #setMishValues(spline,ic.u)
    SBtransform!(spline)
    SAtransform!(spline)
    SItransform!(spline)
    write_output(spline, model, 0)
    
    return spline
end

function run(spline::Spline1D, model::ModelParameters)
    
    println("Model starting up...")

    # Advance the first timestep
    bdot,bdot_delay = calcTendency(spline, model, 1)
    b_nxt, bdot_svd = first_timestep(spline,bdot,model.ts)
    spline.b .= b_nxt
    SAtransform!(spline)
    SItransform!(spline)
        
    # Keep going!
    for t = 2:model.num_ts
        bdot,bdot_delay = calcTendency(spline, model, t)
        b_nxt, bdot_svd = timestep(spline, bdot, bdot_svd, bdot_delay, model.ts)
        spline.b .= b_nxt
        SAtransform!(spline)
        SItransform!(spline)
    end
    
    println("Done with time integration")
    return spline
end
    

function finalize(spline::Spline1D, model::ModelParameters)
    
    write_output(spline, model, model.num_ts)
    println("Model complete!")
end

function integrate_model()
    
    model = ModelParameters(
    ts = 0.1,
    num_ts = 500,
    output_interval = 25,
    xmin = -48.0,
    xmax = 48.0,
    num_nodes = 192,
    BCL = R0,
    BCR = R0,
    equation_set = "1dNonlinearAdvection",
    initial_conditions = "testcase.csv"
    )
   
    spline = initialize(model)
    spline = run(spline, model)
    finalize(spline, model)
end

end
