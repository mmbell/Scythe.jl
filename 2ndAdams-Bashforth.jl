# 2nd order Adams-Bashforth implementation

push!(LOAD_PATH, pwd())
using CubicBSpline

using CairoMakie
using CSV
using DataFrames

#Define some convenient aliases
const real = Float64
const int = Int64
const uint = UInt64

struct NumericalModel
    ts::real
    num_ts::int
    output_interval::int
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

function calcTendency(spline::Spline1D, model::NumericalModel, t::int)
    
    a = SItransform!(spline)
    
    if mod(t,model.output_interval) == 0
        write_output(spline, model, t)
    end
    
    u = SItransform!(spline)
    ux = SIxtransform(spline)
    
    #b_now is held in place
    SBtransform!(spline)
    
    # Feed to physical equations
    udot, F = physical_model(u,ux)
    
    # Do something with F later. May need to be a new spline because of BCs?
    # SBxtransform(spline,F,BCL,BCR)
    bdot_delay = zeros(real,spline.bDim)
    
    bdot = SBtransform(spline,udot)
    return bdot,bdot_delay
end

function physical_model(u::Vector{real},ux::Vector{real})
    
    #1D Linear advection to test
    c_0 = 1.0
    
    udot = -c_0 * ux
    F = 0
    
    return udot, F 
end

function write_output(spline::Spline1D, model::NumericalModel, t::int)
   
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

function initialize()
    
    model = NumericalModel(0.1, 500, 25)
    
    spline = Spline1D(SplineParameters(xmin = -48.0, xmax = 48.0,num_nodes = 96,BCL = R0,BCR = R0))
    
    testfunction = zeros(Float64,spline.mishDim)
    for i = 1:spline.mishDim
        testfunction[i] = exp(-(spline.mishPoints[i])^2 / (2 * 8^2))
    end
    setMishValues(spline,testfunction)
    SBtransform!(spline)
    SAtransform!(spline)
    SItransform!(spline)
    write_output(spline, model, 0)
    
    return spline, model
end

function run(spline::Spline1D, model::NumericalModel)
    
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
    

function finalize(spline::Spline1D, model::NumericalModel)
    
    write_output(spline, model, model.num_ts)
    println("Model complete!")
end

function integrate_model()
   
    spline, model = initialize()
    spline = run(spline, model)
    finalize(spline, model)
end

integrate_model()

u0 = CSV.read("model_u_0", DataFrame, header=["i", "x", "u"])
u50 = CSV.read("model_u_50", DataFrame, header=["i", "x", "u"])
u100 = CSV.read("model_u_100", DataFrame, header=["i", "x", "u"])
u200 = CSV.read("model_u_200", DataFrame, header=["i", "x", "u"])
u500 = CSV.read("model_u_500", DataFrame, header=["i", "x", "u"])

lines(u0.x,u0.u)
lines!(u50.x,u50.u)
lines!(u100.x,u100.u)
lines!(u200.x,u200.u)
lines!(u500.x,u500.u)
current_figure()


