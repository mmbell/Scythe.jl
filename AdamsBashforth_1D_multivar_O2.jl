# 2nd order Adams-Bashforth implementation

module AdamsBashforth_1D_multivar_O2

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
    vars::Dict = Dict("u" => 1)
    equation_set = "1dLinearAdvection"
    initial_conditions = "ic.csv"
end

function first_timestep(splines::Vector{Spline1D}, bdot::Matrix{real}, ts::real)
   
    b_nxt = Matrix{real}(undef,splines[1].bDim,length(splines))
    bdot_svd = Matrix{real}(undef,splines[1].bDim,length(splines))
    
    # Use Euler method for first step
    for v in eachindex(splines)
        b_nxt[:,v] = splines[v].b + (ts * bdot[:,v])
        bdot_svd[:,v] = bdot[:,v]
    end
    return b_nxt, bdot_svd
end

function timestep(splines::Vector{Spline1D}, bdot::Matrix{real}, bdot_svd::Matrix{real}, 
        bdot_delay::Matrix{real}, ts::real)

    b_nxt = Matrix{real}(undef,splines[1].bDim,length(splines))
    
    for v in eachindex(splines)
        b_nxt[:,v] = splines[v].b + (0.5 * ts) * ((3.0 * bdot[:,v]) - bdot_svd[:,v]) + (ts * bdot_delay[:,v])
        bdot_svd[:,v] = bdot[:,v]
    end
    return b_nxt, bdot_svd
end

function calcTendency(splines::Vector{Spline1D}, model::ModelParameters, t::int)
    
    # Need to declare these before assigning
    # This works for multivar problem, but not for multidimensional since splines may have different dims
    u = Matrix{real}(undef,splines[1].mishDim,length(splines))
    ux = Matrix{real}(undef,splines[1].mishDim,length(splines))
    uxx = Matrix{real}(undef,splines[1].mishDim,length(splines))
    bdot_delay = Matrix{real}(undef,splines[1].bDim,length(splines))
    bdot = Matrix{real}(undef,splines[1].bDim,length(splines))
    
    for v in eachindex(splines)
        a = SItransform!(splines[v])

        u[:,v] = SItransform!(splines[v])
        ux[:,v] = SIxtransform(splines[v])
        uxx[:,v] = SIxxtransform(splines[v])

        #b_now is held in place
        SBtransform!(splines[v])
    end
    
    if mod(t,model.output_interval) == 0
        write_output(splines, model, t)
    end
    
    # Feed physical matrices to physical equations
    udot, F = physical_model(model,u,ux,uxx)

    for v in eachindex(splines)
        # Do something with F later. May need to be a new spline because of BCs?
        # SBxtransform(spline,F,BCL,BCR)
        bdot_delay[:,v] = zeros(real,splines[v].bDim)

        bdot[:,v] = SBtransform(splines[v],udot[:,v])
    end
    return bdot,bdot_delay
end

function physical_model(model::ModelParameters,u::Matrix{real},ux::Matrix{real},uxx::Matrix{real})
    
    udot = zeros(real,size(u, 1),size(u, 2))
    F = zeros(real,size(u, 1),size(u, 2))
    if model.equation_set == "1dLinearAdvection"
        #1D Linear advection to test
        c_0 = 1.0
        K = 0.003
        
        udot[:,1] = -c_0 .* ux[:,1] + (K .* uxx[:,1])
        udot[:,2] = -c_0 .* ux[:,2] + (K .* uxx[:,2])
        F = 0
    elseif model.equation_set == "1dNonlinearAdvection"
        c_0 = 1.0
        K = 0.048
        
        udot[:,1] = -(c_0 .+ u[:,1]) .* ux[:,1] + (K .* uxx[:,1])
        F = 0
    elseif model.equation_set == "1dLinearShallowWater"
        K = 0.003
        g = 9.81
        H = 1.0
        
        udot[:,1] = -g .* ux[:,2]
        udot[:,2] = -H .* ux[:,1]
        F = 0
    else
        throw(MethodError("Selected equation set not implemented"))
    end
    
    return udot, F 
end

function write_output(splines::Vector{Spline1D}, model::ModelParameters, t::int)
    
    for var in keys(model.vars)
        v = model.vars[var]
        afilename = string("model_a_", var , "_", t, ".csv")
        ufilename = string("model_", var , "_", t, ".csv")
        afile = open(afilename,"w")
        ufile = open(ufilename,"w")

        a = splines[v].a    
        for i = 1:splines[v].aDim
            a_i = a[i]
            write(afile,"$i, $a_i\n")
        end        

        SItransform!(splines[v])
        u = splines[v].uMish
        mishPoints = splines[v].mishPoints
        for i = 1:splines[v].mishDim
            mp_i = mishPoints[i]
            u_i = u[i]
            write(ufile,"$i, $mp_i, $u_i\n")
        end
        close(afile)
        close(ufile)
    end
end

function initialize(model::ModelParameters)
    
    splines = Vector{Spline1D}(undef,length(values(model.vars)))
    for key in keys(model.vars)
        splines[model.vars[key]] = Spline1D(SplineParameters(xmin = model.xmin, xmax = model.xmax,
                num_nodes = model.num_nodes, BCL = model.BCL[key], BCR = model.BCR[key]))
    end
    
    #ic = CSV.read(model.initial_conditions, DataFrame, "x", "u")
    #if (spline.mishPoints ≉ ic.x)
    #    throw(DomainError(ic.x[1], "mish from IC does not match model parameters"))
    #end
    
    # Hard-code IC for testing
    for i = 1:splines[1].mishDim
        splines[model.vars["u"]].uMish[i] = 0
        splines[model.vars["h"]].uMish[i] = exp(-(splines[1].mishPoints[i])^2 / (2 * 4^2))
    end
    #setMishValues(spline,ic.u)

    for spline in splines
        SBtransform!(spline)
        SAtransform!(spline)
        SItransform!(spline)
    end
    
    write_output(splines, model, 0)
    
    return splines
end

function run(splines::Vector{Spline1D}, model::ModelParameters)
    
    println("Model starting up...")

    # Advance the first timestep
    bdot,bdot_delay = calcTendency(splines, model, 1)
    b_nxt, bdot_svd = first_timestep(splines, bdot, model.ts)
    for v in eachindex(splines)
        splines[v].b .= b_nxt[:,v]
        SAtransform!(splines[v])
        SItransform!(splines[v])
    end
    
    # Keep going!
    for t = 2:model.num_ts
        bdot,bdot_delay = calcTendency(splines, model, t)
        b_nxt, bdot_svd = timestep(splines, bdot, bdot_svd, bdot_delay, model.ts)
        for v in eachindex(splines)
            splines[v].b .= b_nxt[:,v]
            SAtransform!(splines[v])
            SItransform!(splines[v])
        end
    end
    
    println("Done with time integration")
    return splines
end
    

function finalize(splines::Vector{Spline1D}, model::ModelParameters)
    
    write_output(splines, model, model.num_ts)
    println("Model complete!")
end

function integrate_model()
    
    model = ModelParameters(
        ts = 0.1,
        num_ts = 480,
        output_interval = 25,
        xmin = -48.0,
        xmax = 48.0,
        num_nodes = 96,
        BCL = Dict("u" => R2T20, "h" => R1T1),
        BCR = Dict("u" => R0, "h" => R0),
        equation_set = "1dLinearShallowWater",
        initial_conditions = "testcase.csv",
        vars = Dict("u" => 1, "h" => 2)    
    )
   
    splines = initialize(model)
    splines = run(splines, model)
    finalize(splines, model)
end

end
