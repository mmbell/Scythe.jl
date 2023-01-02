module NumericalModels

using SpectralGrid
using LoopVectorization

export ModelParameters

#Define some convenient aliases
const real = Float64
const int = Int64
const uint = UInt64

Base.@kwdef struct ModelParameters
    ts::Float64 = 0.0
    integration_time::Float64 = 1.0
    output_interval::Float64 = 1.0
    equation_set = "LinearAdvection1D"
    initial_conditions = "ic.csv"
    output_dir = "./output/"
    grid_params::GridParameters
    physical_params::Dict
end

function LinearAdvection1D(grid::R_Grid,
            gridpoints::Array{real},
            vardot::Array{real},
            F::Array{real},
            model::ModelParameters)
   
    #1D Linear advection to test
    c_0 = model.physical_params[:c_0]
    K = model.physical_params[:K]

    u = grid.physical[:,1,1]
    ur = grid.physical[:,1,2]
    urr = grid.physical[:,1,3]

    vardot[:,1] .= -(c_0 .* ur) .+ (K .* urr)

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
    #UKDIFF = K .* ((u ./ r) .+ ur)
    UKDIFF = K .* ((ur ./ r) .+ urr .- (u ./ (r .* r)))
    vardot[:,2] .= UADV .+ UDRAG .+ UCOR .+ UPGF .+ UW .+ UKDIFF
    #F[:,2] .= UKDIFF
    F[:,2] .= 0.0
    
    VADV = -u .* (f .+ (v ./ r) .+ vr)
    VDRAG = -(Cd .* U .* v ./ h)
    VW = w_ .* (vgr - v) ./ h
    #VKDIFF = K .* ((v ./ r) .+ vr)
    VKDIFF = K .* ((vr ./ r) .+ vrr .- (v ./ (r .* r)))
    vardot[:,3] .= VADV .+ VDRAG .+ VW .+ UKDIFF
    #F[:,3] .= VKDIFF
    F[:,3] .= 0.0

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
   
    #2D Linear advection to test
    K = model.physical_params[:K]
    
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

function LinearAdvectionRLZ(grid::RLZ_Grid, 
            gridpoints::Array{real},
            vardot::Array{real},
            F::Array{real},
            model::ModelParameters)

    #2D Linear advection to test
    K = model.physical_params[:K]

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
    
    vardot[:,1] .= -H * ((u ./ r) .+ ur .+ (vl ./ r))
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
    f = 0.0
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
        (-(H .+ h) .* ((u ./ r) .+ ur .+ (vl ./ r))))

    vardot[:,2] .= ((-v .* ul ./ r) .+ (-u .* ur) .+
        (-g .* hr) .+
        (v .* (f .+ (v ./ r))) .+
        (K .* ((ur ./ r) .+ urr .+ (ull ./ (r .* r)) .- (u ./ (r .* r)))))
    
    vardot[:,3] .= ((-v .* vl ./ r) .+ (-u .* vr) .+
        (-g .* (hl ./ r)) .+
        (-u .* (f .+ (v ./ r))) .+
        (K .* ((vr ./ r) .+ vrr .+ (vll ./ (r .* r)) .- (v ./ (r .* r)))))
    # F = 0
end

function RL_SlabTCBL(grid::RL_Grid, 
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
    r = gridpoints[:,1]

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
    UW = -(w_ .* (u ./ h))
    UKDIFF = K .* ((ur ./ r) .+ urr .- (u ./ (r .* r)))
    vardot[:,2] .= UADV .+ UDRAG .+ UCOR .+ UPGF .+ UW .+ UKDIFF
    F[:,2] .= 0.0
    
    VADV = -u .* (f .+ (v ./ r) .+ vr)
    VDRAG = -(Cd .* U .* v ./ h)
    VW = w_ .* (vgr - v) ./ h
    VKDIFF = K .* ((vr ./ r) .+ vrr .- (v ./ (r .* r)))
    vardot[:,3] .= VADV .+ VDRAG .+ VW .+ VKDIFF
    F[:,3] .= 0.0
    
end

function Oneway_ShallowWater_Slab(grid::RL_Grid, 
            gridpoints::Array{real},
            vardot::Array{real},
            F::Array{real},
            model::ModelParameters)
   
    # Need to figure out how to assign these with symbols
    g = 9.81
    K = 1500.0
    Cd = 2.4e-3
    Hfree = 2000.0
    Hb = 1000.0
    f = 5.0e-5

    r = gridpoints[:,1]

    h = grid.physical[:,1,1]
    hr = grid.physical[:,1,2]
    hrr = grid.physical[:,1,3]
    hl = grid.physical[:,1,4]
    hll = grid.physical[:,1,5]
    
    ug = grid.physical[:,2,1]
    ugr = grid.physical[:,2,2]
    ugrr = grid.physical[:,2,3]
    ugl = grid.physical[:,2,4]
    ugll = grid.physical[:,2,5]
        
    vg = grid.physical[:,3,1]
    vgr = grid.physical[:,3,2]
    vgrr = grid.physical[:,3,3]
    vgl = grid.physical[:,3,4]
    vgll = grid.physical[:,3,5]
    
    # h tendency
    vardot[:,1] .= ((-vg .* hl ./ r) .+ (-ug .* hr) .+
        (-(Hfree .+ h) .* ((ug ./ r) .+ ugr .+ (vgl ./ r))))
    F[:,1] .= 0.0

    # ug tendency
    vardot[:,2] .= ((-vg .* ugl ./ r) .+ (-ug .* ugr) .+
        (-g .* hr) .+
        (vg .* (f .+ (vg ./ r))))
    F[:,2] .= 0.0
    
    # vg tendency
    vardot[:,3] .= ((-vg .* vgl ./ r) .+ (-ug .* vgr) .+
        (-g .* (hl ./ r)) .+
        (-ug .* (f .+ (vg ./ r))))
    F[:,3] .= 0.0
    
    u = grid.physical[:,4,1]
    ur = grid.physical[:,4,2]
    urr = grid.physical[:,4,3]
    ul = grid.physical[:,4,4]
    ull = grid.physical[:,4,5]
    
    v = grid.physical[:,5,1]
    vr = grid.physical[:,5,2]
    vrr = grid.physical[:,5,3]
    vl = grid.physical[:,5,4]
    vll = grid.physical[:,5,5]

    U = 0.78 * sqrt.((u .* u) .+ (v .* v))

    # W is diagnostic
    w = -Hb .* ((u ./ r) .+ ur)
    w_ = 0.5 .* abs.(w) .- w
    grid.physical[:,6,1] .= w
    vardot[:,6] .= 0.0
    F[:,6] .= 0.0

    UADV = (-(u .* ur)) .+ (-v .* ul ./ r)
    UDRAG = -(Cd .* U .* u ./ Hb)
    UCOR = ((f .* v) .+ ((v .* v) ./ r))
    UPGF = (-g .* hr)
    UW = -(w_ .* (u ./ Hb))
    UKDIFF = K .* ((ur ./ r) .+ urr .- (u ./ (r .* r)) .+ (ull ./ (r .* r)) .- (2.0 .* vl ./ (r .* r)))
    vardot[:,4] .= UADV .+ UDRAG .+ UCOR .+ UPGF .+ UW .+ UKDIFF
    F[:,4] .= 0.0
    
    VADV = (-u .* (f .+ (v ./ r) .+ vr)) .+ (-v .* vl ./ r)
    VDRAG = -(Cd .* U .* v ./ Hb)
    VPGF = (-g .* (hl ./ r))
    VW = w_ .* (vg - v) ./ Hb
    VKDIFF = K .* ((vr ./ r) .+ vrr .- (v ./ (r .* r)) .+ (vll ./ (r .* r)) .+ (2.0 .* ul ./ (r .* r)))
    vardot[:,5] .= VADV .+ VDRAG .+ VPGF .+ VW .+ VKDIFF
    F[:,5] .= 0.0
    
end

function Twoway_ShallowWater_Slab(grid::RL_Grid, 
            gridpoints::Array{real},
            vardot::Array{real},
            F::Array{real},
            model::ModelParameters)
   
    # Need to figure out how to assign these with symbols
    g = 9.81
    K = 1500.0
    Cd = 2.4e-3
    Hfree = 2000.0
    Hb = 1000.0
    f = 5.0e-5
    S1 = 1.0e-5
    
    r = gridpoints[:,1]

    h = grid.physical[:,1,1]
    hr = grid.physical[:,1,2]
    hrr = grid.physical[:,1,3]
    hl = grid.physical[:,1,4]
    hll = grid.physical[:,1,5]
    
    ug = grid.physical[:,2,1]
    ugr = grid.physical[:,2,2]
    ugrr = grid.physical[:,2,3]
    ugl = grid.physical[:,2,4]
    ugll = grid.physical[:,2,5]
        
    vg = grid.physical[:,3,1]
    vgr = grid.physical[:,3,2]
    vgrr = grid.physical[:,3,3]
    vgl = grid.physical[:,3,4]
    vgll = grid.physical[:,3,5]
    
    # ug tendency
    vardot[:,2] .= ((-vg .* ugl ./ r) .+ (-ug .* ugr) .+
        (-g .* hr) .+
        (vg .* (f .+ (vg ./ r))))
    F[:,2] .= 0.0
    
    # vg tendency
    vardot[:,3] .= ((-vg .* vgl ./ r) .+ (-ug .* vgr) .+
        (-g .* (hl ./ r)) .+
        (-ug .* (f .+ (vg ./ r))))
    F[:,3] .= 0.0
    
    u = grid.physical[:,4,1]
    ur = grid.physical[:,4,2]
    urr = grid.physical[:,4,3]
    ul = grid.physical[:,4,4]
    ull = grid.physical[:,4,5]
    
    v = grid.physical[:,5,1]
    vr = grid.physical[:,5,2]
    vrr = grid.physical[:,5,3]
    vl = grid.physical[:,5,4]
    vll = grid.physical[:,5,5]

    U = 0.78 * sqrt.((u .* u) .+ (v .* v))

    # W is diagnostic
    w = -Hb .* ((u ./ r) .+ ur .+ (vl ./ r))
    w_ = 0.5 .* abs.(w) .- w
    grid.physical[:,6,1] .= w
    vardot[:,6] .= 0.0
    F[:,6] .= 0.0

    # h tendency includes W as mass sink/source
    S = w * S1
    vardot[:,1] .= ((-vg .* hl ./ r) .+ (-ug .* hr) .+
        (-(Hfree .+ h) .* ((ug ./ r) .+ ugr .+ (vgl ./ r)))) .+
        -(Hfree .+ h) .* S
    F[:,1] .= 0.0
    
    UADV = (-(u .* ur)) .+ (-v .* ul ./ r)
    UDRAG = -(Cd .* U .* u ./ Hb)
    UCOR = ((f .* v) .+ ((v .* v) ./ r))
    UPGF = (-g .* hr)
    UW = -(w_ .* (u ./ Hb))
    UKDIFF = K .* ((ur ./ r) .+ urr .- (u ./ (r .* r)) .+ (ull ./ (r .* r)) .- (2.0 .* vl ./ (r .* r)))
    vardot[:,4] .= UADV .+ UDRAG .+ UCOR .+ UPGF .+ UW .+ UKDIFF
    F[:,4] .= 0.0
    
    VADV = (-u .* (f .+ (v ./ r) .+ vr)) .+ (-v .* vl ./ r)
    VDRAG = -(Cd .* U .* v ./ Hb)
    VPGF = (-g .* (hl ./ r))
    VW = w_ .* (vg - v) ./ Hb
    VKDIFF = K .* ((vr ./ r) .+ vrr .- (v ./ (r .* r)) .+ (vll ./ (r .* r)) .+ (2.0 .* ul ./ (r .* r)))
    vardot[:,5] .= VADV .+ VDRAG .+ VPGF .+ VW .+ VKDIFF
    F[:,5] .= 0.0
    
end

# Module end
end
