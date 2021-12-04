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
    udot, F = physical_model(model,splines[1].mishPoints,u,ux,uxx)

    for v in eachindex(splines)
        bdot[:,v] = SBtransform(splines[v],udot[:,v])
        bdot_delay[:,v] = zeros(real,splines[v].bDim)
    end
    
    # Do something with F. May need to be a new spline because of BCs?
    Fspline = Spline1D(SplineParameters(xmin = model.xmin, xmax = model.xmax,
            num_nodes = model.num_nodes, BCL = R0, BCR = R0))
    # SBxtransform(spline,F,BCL,BCR) not working
    Fspline.uMish .= F[:,2]
    SBtransform!(Fspline)
    SAtransform!(Fspline)
    bdot_delay[:,2] = SBtransform(Fspline,SIxtransform(Fspline))
    Fspline.uMish .= F[:,3]
    SBtransform!(Fspline)
    SAtransform!(Fspline)
    bdot_delay[:,3] = SBtransform(Fspline,SIxtransform(Fspline))
   
    return bdot,bdot_delay
end

function physical_model(model::ModelParameters,x::Vector{real},var::Matrix{real},varx::Matrix{real},varxx::Matrix{real})
    
    vardot = zeros(real,size(var, 1),size(var, 2))
    F = zeros(real,size(var, 1),size(var, 2))
    if model.equation_set == "1dLinearAdvection"
        #1D Linear advection to test
        c_0 = 1.0
        K = 0.003
        
        vardot[:,1] = -c_0 .* varx[:,1] + (K .* varxx[:,1])
        vardot[:,2] = -c_0 .* varx[:,2] + (K .* varxx[:,2])
        F = 0
    elseif model.equation_set == "1dNonlinearAdvection"
        c_0 = 1.0
        K = 0.048
        
        udot[:,1] = -(c_0 .+ var[:,1]) .* varx[:,1] + (K .* varxx[:,1])
        F = 0
    elseif model.equation_set == "1dLinearShallowWater"
        K = 0.003
        g = 9.81
        H = 1.0
        
        vardot[:,1] = -g .* varx[:,2]
        vardot[:,2] = -H .* varx[:,1]
        F = 0
    elseif model.equation_set == "Williams2013_TCBL"
        
        #vardot, F = Williams2013_TBCL(model,x,var,varx,varxx)
        K = 1500.0
        Cd = 2.4e-3
        h = 1000.0
        f = 5.0e-5

        vgrdot = zeros(real,length(x))
        F = zeros(real,length(x),4)

        vgr = var[:,1]
        u = var[:,2]
        v = var[:,3]
        r = x
        U = 0.78 * sqrt.((u .* u) .+ (v .* v))

        ur = varx[:,2]
        vr = varx[:,3]

        urr = varxx[:,2]
        vrr = varxx[:,3]

        w = -h .* ((u ./ r) .+ ur)
        w_ = 0.5 .* abs.(w) .- w

        vardot[:,1] .= vgrdot
        
        UADV = -(u .* ur)
        UDRAG = -(Cd .* U .* u ./ h)
        UCOR = ((f .* v) .+ ((v .* v) ./ r))
        UPGF = -((f .* vgr) .+ ((vgr .* vgr) ./ r))
        UW = -(w_ .* (u ./ h))
        UKDIFF = K .* ((u ./ r) .+ ur)
        vardot[:,2] .= UADV .+ UDRAG .+ UCOR .+ UPGF .+ UW
        F[:,2] .= UKDIFF

        
        VADV = -u .* (f .+ (v ./ r) .+ vr)
        VDRAG = -(Cd .* U .* v ./ h)
        VW = w_ .* (vgr - v) ./ h
        VKDIFF = K .* ((v ./ r) .+ vr)
        vardot[:,3] .= VADV .+ VDRAG .+ UW
        F[:,3] .= VKDIFF
        
        vardot[:,4] .= w

    else
        error("Selected equation set not implemented")
    end
    
    return vardot, F 
end

function write_output(splines::Vector{Spline1D}, model::ModelParameters, t::int)
    
    println("Writing output at time $t")
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
    V0 = 50.0 / 20000.0
    for i = 1:splines[1].mishDim
        splines[model.vars["u"]].uMish[i] = 0
        #splines[model.vars["h"]].uMish[i] = exp(-(splines[1].mishPoints[i])^2 / (2 * 4^2))
        if (splines[1].mishPoints[i] < 20000.0)
            splines[model.vars["vgr"]].uMish[i] = V0 * splines[1].mishPoints[i]
            splines[model.vars["v"]].uMish[i] = V0 * splines[1].mishPoints[i]
        else
            splines[model.vars["vgr"]].uMish[i] = 4.0e8 * V0 / (splines[1].mishPoints[i])
            splines[model.vars["v"]].uMish[i] = 4.0e8 * V0 / (splines[1].mishPoints[i])
        end
        splines[model.vars["w"]].uMish[i] = 0
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
    for v in Base.OneTo(length(splines)-1)
        splines[v].b .= b_nxt[:,v]
        SAtransform!(splines[v])
        SItransform!(splines[v])
    end
    
    # Keep going!
    for t = 2:model.num_ts
        bdot,bdot_delay = calcTendency(splines, model, t)
                
        b_nxt, bdot_svd = timestep(splines, bdot, bdot_svd, bdot_delay, model.ts)
        for v in Base.OneTo(length(splines)-1)
            splines[v].b .= b_nxt[:,v]
            SAtransform!(splines[v])
            SItransform!(splines[v])
        end
        # Override diagnostic variables
        splines[4].b .= bdot[:,4]
        SAtransform!(splines[4])
        SItransform!(splines[4])
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
        ts = 1.0,
        num_ts = 10800,
        output_interval = 3600,
        xmin = 0.0,
        xmax = 1.0e6,
        num_nodes = 2000,
        BCL = Dict("vgr" => R0, "u" => R1T0, "v" => R1T0, "w" => R1T0),
        BCR = Dict("vgr" => R0, "u" => R1T1, "v" => R1T1, "w" => R1T1),
        equation_set = "Williams2013_TCBL",
        initial_conditions = "testcase.csv",
        vars = Dict("vgr" => 1, "u" => 2, "v" => 3, "w" => 4)    
    )
   
    splines = initialize(model)
    splines = run(splines, model)
    finalize(splines, model)
end

function Williams2013_TBCL(model::ModelParameters,x::Vector{real},
        var::Matrix{real},varx::Matrix{real},varxx::Matrix{real})

    K = 1500.0
    Cd = 2.4e-3
    h = 1000.0
    f = 5.0e-5

    udot = zeros(real,length(x))
    vdot = zeros(real,length(x))
    vgrdot = zeros(real,length(x))
    F = zeros(real,length(x),4)
    
    vgr = var[:,model.vars["vgr"]]
    u = var[:,model.vars["u"]]
    v = var[:,model.vars["v"]]
    r = x
    U = 0.78 * sqrt.(u .* u + v .* v)
    
    ur = varx[:,model.vars["u"]]
    vr = varx[:,model.vars["v"]]
    
    urr = varxx[:,model.vars["u"]]
    vrr = varxx[:,model.vars["v"]]
    
    w = -h * ((u ./ r) .+ ur)
    w_ = 0.5 * abs.(w) - w
    
    udot = (-u .* ur) .- (w_ .* (u ./ h)) .+ (f .+ (v .+ vgr) ./ r) .* (v .- vgr) 
        .- (Cd .* U .* u ./ h) .+ K .* ((ur ./ r) .+ (u ./ (r .* r)) .+ urr)
        
    vdot = (w_ .* (vgr - v) ./ h) .- (f .+ (v ./ r) .+ vr) .* u
        .- (Cd .* U .* v ./ h) .+ K .* ((vr ./ r) .+ (v ./ (r .* r)) .+ vrr)
    
    tendency = [vgrdot udot vdot w]
    return tendency, F
end

end
