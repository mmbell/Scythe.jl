module NumericalModels

using SpectralGrid
using Parameters
using LoopVectorization

export ModelParameters

#Define some convenient aliases
const real = Float64
const int = Int64
const uint = UInt64

@with_kw struct ModelParameters
    ts::Float64 = 0.0
    integration_time::Float64 = 1.0
    output_interval::Float64 = 1.0
    equation_set = "LinearAdvection1D"
    initial_conditions = "ic.csv"
    output_dir = "./output/"
    grid_params::GridParameters
end

function LinearAdvection1D(physical::Array{real}, 
            gridpoints::Array{real},
            vardot::Array{real},
            F::Array{real},
            model::ModelParameters)
   
    #1D Linear advection to test
    c_0 = 1.0
    K = 0.003

    vardot[:,1] .= -c_0 .* physical[:,1,2] .+ (K .* physical[:,1,3])        
    # F = 0
end

function Williams2013_slabTCBL(grid::R_Grid, 
            gridpoints::Array{real},
            vardot::Array{real},
            F::Array{real},
            model::ModelParameters)

    # Need to figure out how to assign these with symbols
    K = 1500.0
    Cd = 2.4e-3
    h = 1000.0
    f = 5.0e-5

    vgr = grid.physical[:,1,1]
    vardot[:,1] .= 0.0
    F[:,1] .= 0.0
    
    u = grid.physical[:,2,1]
    ur = grid.physical[:,2,2]
    urr = grid.physical[:,2,3]
    v = grid.physical[:,3,1]
    vr = grid.physical[:,3,2]
    vrr = grid.physical[:,3,3]
    r = gridpoints

    U = 0.78 * sqrt.((u .* u) .+ (v .* v))

    w = -h .* ((u ./ r) .+ ur)
    w_ = 0.5 .* abs.(w) .- w
    # W is diagnostic
    grid.physical[:,4,1] .= w
    vardot[:,4] .= 0.0
    F[:,4] .= 0.0

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
    vardot[:,3] .= VADV .+ VDRAG .+ VW
    F[:,3] .= VKDIFF

end

function LinearAdvectionRZ(physical::Array{real}, 
            gridpoints::Array{real},
            vardot::Array{real},
            F::Array{real},
            model::ModelParameters)
   
    #1D Linear advection to test
    c_0 = 5.0
    K = 0.003

    vardot[:,1] .= -c_0 .* physical[:,1,2] .+ (K .* physical[:,1,3])        
    # F = 0
end

function Kepert2017_TCBL(grid::RZ_Grid, 
            gridpoints::Array{real},
            udot::Array{real},
            F::Array{real},
            model::ModelParameters)

    # Need to figure out how to assign these with symbols
    K = 1500.0
    Cd = 2.4e-3
    f = 5.0e-5

    # No delayed diffusion
    #F = 0
    
    # Gradient wind doesn't change
    vgr = grid.physical[:,1,1]
    udot[:,1] .= 0.0
    
    u = grid.physical[:,2,1]
    ur = grid.physical[:,2,2]
    urr = grid.physical[:,2,3]
    uz = grid.physical[:,2,4]
    uzz = grid.physical[:,2,5]
    
    v = grid.physical[:,3,1]
    vr = grid.physical[:,3,2]
    vrr = grid.physical[:,3,3]
    vz = grid.physical[:,3,4]
    vzz = grid.physical[:,3,5]
    
    r = gridpoints[:,1]
    z = gridpoints[:,2]

    # Get the 10 meter wind (assuming 10 m @ z == 2)
    r1 = grid.params.rDim+1
    r2 = 2*grid.params.rDim
    u10 = grid.physical[r1:r2,2,1]
    v10 = grid.physical[r1:r2,3,1]
    U10 = sqrt.((u10 .* u10) .+ (v10 .* v10))
    
    # Calculate the vertical diffusivity and vertical velocity
    Kv = zeros(Float64, size(grid.physical))
    Kvspectral = zeros(Float64, size(grid.spectral))
    w = zeros(Float64, size(grid.physical[:,4,1]))
    
    S = sqrt.((uz .* uz) .+ (vz .* vz))

    # Surface drag
    r1 = 1
    r2 = grid.params.rDim
    Kv[r1:r2,1,1] = Cd .* U10 .* u10
    Kv[r1:r2,2,1] = Cd .* U10 .* v10
    
    # Go through each vertical level
    for z = 2:grid.params.zDim
        # Calculate Kv
        l = 1.0 / ((1.0 / (0.4 * gridpoints[z])) + (1.0 / 80.0))
        r1 = ((z-1)*grid.params.rDim)+1
        r2 = z*grid.params.rDim
        Kv[r1:r2,1,1] = (l * l) .* S[r1:r2] .* uz[r1:r2]
        Kv[r1:r2,2,1] = (l * l) .* S[r1:r2] .* vz[r1:r2]
    end
    
    # Use Kv[3] for convergence
    Kv[:,3,1] .= -((u ./ r) .+ ur)
    
    # Differentiate Ku and Kv
    spectralTransform(grid, Kv, Kvspectral)
    gridTransform_noBCs(grid, Kv, Kvspectral)

    # Integrate divergence to get W    
    w = integrateUp(grid, Kv[:,3,1], Kvspectral[:,3])
    grid.physical[:,4,1] .= w
    udot[:,4] .= 0.0

    UADV = -(u .* ur) 
    UCOR = ((f .* v) .+ ((v .* v) ./ r))
    UPGF = -((f .* vgr) .+ ((vgr .* vgr) ./ r))
    UW = -(w .* uz)
    UHDIFF = K .* ((ur ./ r) .+ urr .- (u ./ (r .* r)))
    UVDIFF = Kv[:,1,4]
    udot[:,2] .= UADV .+ UCOR .+ UPGF .+ UW .+ UHDIFF .+ UVDIFF

    VADV = -u .* (f .+ (v ./ r) .+ vr)
    VW = -(w .* vz)
    VHDIFF = K .* ((vr ./ r) .+ vrr .- (v ./ (r .* r)))
    VVDIFF = Kv[:,2,4]
    udot[:,3] .= VADV .+ UW .+ VHDIFF .+ VVDIFF

end

function LinearAdvectionRL(grid::RL_Grid, 
            gridpoints::Array{real},
            vardot::Array{real},
            F::Array{real},
            model::ModelParameters)
   
    #1D Linear advection to test
    c_0 = 5.0
    K = 0.003
    
    r = gridpoints[:,1]
    hr = grid.physical[:,1,2]
    hrr = grid.physical[:,1,3]
    hl = grid.physical[:,1,4]
    hll = grid.physical[:,1,5]
    u = grid.physical[:,2,1]
    v = grid.physical[:,3,1]
    
    vardot[:,1] .= (-u .* hr) .- (v .* (hl ./ r)) .+ (K .* ((hr ./ r) .+ hrr .+ (hll ./ (r .* r))))   
    # F = 0
end

function LinearShallowWaterRL(grid::RL_Grid, 
            gridpoints::Array{real},
            vardot::Array{real},
            F::Array{real},
            model::ModelParameters)
   
    #Linear shallow water equations
    g = 9.81
    H = 1000.0
    K = 0.003
    
    r = gridpoints[:,1]
    h = grid.physical[:,1,1]
    hr = grid.physical[:,1,2]
    hrr = grid.physical[:,1,3]
    hl = grid.physical[:,1,4]
    hll = grid.physical[:,1,5]
    u = grid.physical[:,2,1]
    ur = grid.physical[:,2,2]
    urr = grid.physical[:,2,3]
    ul = grid.physical[:,2,4]
    ull = grid.physical[:,2,5]
    v = grid.physical[:,3,1]
    vr = grid.physical[:,3,2]
    vrr = grid.physical[:,3,3]
    vl = grid.physical[:,3,4]
    vll = grid.physical[:,3,5]
    
    vardot[:,1] .= -H * ((u ./ r) .+ ur .+ (vl ./ r)) .+ (K .* ((hr ./ r) .+ hrr .+ (hll ./ (r .* r))))
    vardot[:,2] .= (-g .* hr) .+ (K .* ((ur ./ r) .+ urr .+ (ull ./ (r .* r)))) 
    vardot[:,3] .= (-g .* (hl ./ r)) .+ (K .* ((vr ./ r) .+ vrr .+ (vll ./ (r .* r))))
    # F = 0
end

function ShallowWaterRL(grid::RL_Grid, 
            gridpoints::Array{real},
            vardot::Array{real},
            F::Array{real},
            model::ModelParameters)
   
    #Nonlinear shallow water equations
    g = 9.81
    f = 5.0e-5
    H = 2000.0
    
    r = gridpoints[:,1]
    h = grid.physical[:,1,1]
    hr = grid.physical[:,1,2]
    hrr = grid.physical[:,1,3]
    hl = grid.physical[:,1,4]
    hll = grid.physical[:,1,5]
    u = grid.physical[:,2,1]
    ur = grid.physical[:,2,2]
    urr = grid.physical[:,2,3]
    ul = grid.physical[:,2,4]
    ull = grid.physical[:,2,5]
    v = grid.physical[:,3,1]
    vr = grid.physical[:,3,2]
    vrr = grid.physical[:,3,3]
    vl = grid.physical[:,3,4]
    vll = grid.physical[:,3,5]
    
    K = 0.0
    
    vardot[:,1] .= ((-v .* hl ./ r) .+ (-u .* hr) .+
        (-(H .+ h) .* ((u ./ r) .+ ur .+ (vl ./ r))) .+
        (K .* ((hr ./ r) .+ hrr .+ (hll ./ (r .* r)))))
    
    vardot[:,2] .= ((-v .* ul ./ r) .+ (-u .* ur) .+
        (-g .* hr) .+
        (v .* (f .+ (v ./ r))) .+
        (K .* ((ur ./ r) .+ urr .+ (ull ./ (r .* r)))))
    
    vardot[:,3] .= ((-v .* vl ./ r) .+ (-u .* vr) .+
        (-g .* (hl ./ r)) .+
        (-u .* (f .+ (v ./ r))) .+
        (K .* ((vr ./ r) .+ vrr .+ (vll ./ (r .* r)))))
    # F = 0
end

function Oneway_ShallowWater_Slab(grid::RL_Grid, 
            gridpoints::Array{real},
            vardot::Array{real},
            F::Array{real},
            model::ModelParameters)
   
    #Nonlinear shallow water equations
    g = 9.81
    f = 5.0e-5
    Cd = 2.4e-3
    Hfree = 2000.0
    Kfree = 0.0
    
    Hblayer = 500.0
    
    r = gridpoints[:,1]
    h = grid.physical[:,1,1]
    hr = grid.physical[:,1,2]
    hrr = grid.physical[:,1,3]
    hl = grid.physical[:,1,4]
    hll = grid.physical[:,1,5]
    
    u = grid.physical[:,2,1]
    ur = grid.physical[:,2,2]
    urr = grid.physical[:,2,3]
    ul = grid.physical[:,2,4]
    ull = grid.physical[:,2,5]
    
    v = grid.physical[:,3,1]
    vr = grid.physical[:,3,2]
    vrr = grid.physical[:,3,3]
    vl = grid.physical[:,3,4]
    vll = grid.physical[:,3,5]
    
    ub = grid.physical[:,4,1]
    ubr = grid.physical[:,4,2]
    ubrr = grid.physical[:,4,3]
    ubl = grid.physical[:,4,4]
    ubll = grid.physical[:,4,5]
    
    vb = grid.physical[:,5,1]
    vbr = grid.physical[:,5,2]
    vbrr = grid.physical[:,5,3]
    vbl = grid.physical[:,5,4]
    vbll = grid.physical[:,5,5]
    
    U = 0.78 * sqrt.((ub .* ub) .+ (vb .* vb))
    Kblayer = 1.0
    
    # w in boundary layer
    w = -Hblayer .* ((ub ./ r) .+ ubr .+ (vbl ./ r))
    w_ = 0.5 .* abs.(w) .- w
    # W is diagnostic
    grid.physical[:,6,1] .= w
    vardot[:,6] .= 0.0
    
    # h in free atmosphere
    @turbo vardot[:,1] .= ((-v .* hl ./ r) .+ (-u .* hr) .+
        (-(Hfree .+ h) .* ((u ./ r) .+ ur .+ (vl ./ r))))# .+
        #(Kfree .* ((hr ./ r) .+ hrr .+ (hll ./ (r .* r)))))
    
    # u in free atmosphere
    @turbo vardot[:,2] .= ((-v .* ul ./ r) .+ (-u .* ur) .+
        (-g .* hr) .+
        (v .* (f .+ (v ./ r)))) #.+
        #(Kfree .* ((ur ./ r) .+ urr .+ (ull ./ (r .* r)))))
    
    # v in free atmosphere
    @turbo vardot[:,3] .= ((-v .* vl ./ r) .+ (-u .* vr) .+
        (-g .* (hl ./ r)) .+
        (-u .* (f .+ (v ./ r))))# .+
        #(Kfree .* ((vr ./ r) .+ vrr .+ (vll ./ (r .* r)))))
    
    # u in boundary layer
    @turbo vardot[:,4] .= ((-vb .* ubl ./ r) .+ (-ub .* ubr) .+
        (-g .* hr) .+
        (-Cd .* U .* ub ./ Hblayer) .+
        (-w_ .* (u .- ub) ./ Hblayer) .+
        (vb .* (f .+ (vb ./ r))))
    @turbo F[:,4] = Kblayer .* ((ubr ./ r) .+ ubrr .+ (ubll ./ (r .* r)))
    
    # v in boundary layer
    @turbo vardot[:,5] .= ((-vb .* vbl ./ r) .+ (-ub .* vbr) .+
        (-g .* (hl ./ r)) .+
        (-Cd .* U .* vb ./ Hblayer) .+
        (-w_ .* (v .- vb) ./ Hblayer) .+
        (-ub .* (f .+ (vb ./ r))))
    @turbo F[:,5] = Kblayer .* ((vbr ./ r) .+ vbrr .+ (vbll ./ (r .* r)))
    
end

# Module end
end