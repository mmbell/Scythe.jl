# Functions that define the model parameters and physical models

using LoopVectorization

#Define some convenient aliases
const real = Float64
const int = Int64
const uint = UInt64

function LinearAdvectionRZ(grid::RZ_Grid,
            gridpoints::Array{real},
            vardot::Array{real},
            F::Array{real},
            model::ModelParameters)

    #Simple Linear advection to test
    K = model.physical_params[:K]

    r = gridpoints[:,1]
    hr = grid.physical[:,1,2]
    hrr = grid.physical[:,1,3]
    hz = grid.physical[:,1,4]
    hzz = grid.physical[:,1,5]
    u = grid.physical[:,2,1]
    v = grid.physical[:,3,1]
    w = grid.physical[:,4,1]

    @turbo vardot[:,1] .= @. (-u * hr) + (-w * hz) + (K * ((hr / r) + hrr + hzz))

    # F = 0
end

function LinearAdvectionRL(grid::RL_Grid,
            gridpoints::Array{real},
            vardot::Array{real},
            F::Array{real},
            model::ModelParameters)

    #2D Linear advection to test
    K = model.physical_params[:K]
    r = view(gridpoints,:,1)
    hr = view(grid.physical,:,1,2)
    hl = view(grid.physical,:,1,4)
    u = view(grid.physical,:,2,1)
    v = view(grid.physical,:,3,1)

    #@turbo vardot[:,1] .= (-u .* hr) .- (v .* (hl ./ r)) .+ (K .* ((hr ./ r) .+ hrr .+ (hll ./ (r .* r))))
    if K > 0.0
        hrr = view(grid.physical,:,1,3)
        hll = view(grid.physical,:,1,5)
        @turbo vardot[:,1] .= @. (-u * hr) - (v * (hl / r)) + (K * ((hr / r) + hrr + (hll / (r * r))))
    else
        @turbo vardot[:,1] .= @. (-u * hr) - (v * (hl / r))
    end
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

function Oneway_ShallowWater_Slab_old(grid::RL_Grid,
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

function Oneway_ShallowWater_Slab(grid::RL_Grid,
            gridpoints::Array{Float64},
            vardot::Array{Float64},
            F::Array{Float64},
            model::ModelParameters)

    # Need to figure out how to assign these with symbols
    g = model.physical_params[:g]
    K = model.physical_params[:K]
    Cd = model.physical_params[:Cd]
    Hfree = model.physical_params[:Hfree]
    Hb = model.physical_params[:Hb]
    f = model.physical_params[:f]

    r = view(gridpoints,:,1)

    h = view(grid.physical,:,1,1)
    hr = view(grid.physical,:,1,2)
    hrr = view(grid.physical,:,1,3)
    hl = view(grid.physical,:,1,4)
    hll = view(grid.physical,:,1,5)

    ug = view(grid.physical,:,2,1)
    ugr = view(grid.physical,:,2,2)
    ugrr = view(grid.physical,:,2,3)
    ugl = view(grid.physical,:,2,4)
    ugll = view(grid.physical,:,2,5)

    vg = view(grid.physical,:,3,1)
    vgr = view(grid.physical,:,3,2)
    vgrr = view(grid.physical,:,3,3)
    vgl = view(grid.physical,:,3,4)
    vgll = view(grid.physical,:,3,5)

    ADV = similar(r)
    DRAG = similar(r)
    COR = similar(r)
    PGF = similar(r)
    W_ = similar(r)
    KDIFF = similar(r)

    @turbo ADV .= @. (-vg * hl / r) + (-ug * hr) #HADV
    @turbo PGF .= @. (-(Hfree + h) * ((ug / r) + ugr + (vgl / r))) #HPGF (actually just divergence but use PGF array)

    # h tendency
    @turbo vardot[:,1] .= @. ADV + PGF
    #F[:,1] .= 0.0

    @turbo ADV .= @. (-vg * ugl / r) + (-ug * ugr) #UGADV
    @turbo PGF .= @. (-g * hr) #UGPGF
    @turbo COR .= @. (vg * (f + (vg / r))) #COR

    # ug tendency
    vardot[:,2] .= @. ADV + PGF + COR
    #F[:,2] .= 0.0

    @turbo ADV .= @. (-vg * vgl / r) + (-ug * vgr) #UGADV
    @turbo PGF .= @. (-g * (hl / r)) #UGPGF
    @turbo COR .= @. (-ug * (f + (vg / r))) #COR

    # vg tendency
    @turbo vardot[:,3] .= @. ADV + PGF + COR
    #F[:,3] .= 0.0

    u = view(grid.physical,:,4,1)
    ur = view(grid.physical,:,4,2)
    urr = view(grid.physical,:,4,3)
    ul = view(grid.physical,:,4,4)
    ull = view(grid.physical,:,4,5)

    v = view(grid.physical,:,5,1)
    vr = view(grid.physical,:,5,2)
    vrr = view(grid.physical,:,5,3)
    vl = view(grid.physical,:,5,4)
    vll = view(grid.physical,:,5,5)

    sfc_factor = 0.78
    U = similar(u)
    @turbo U .= @. sfc_factor * sqrt((u * u) + (v * v))

    # W is diagnostic
    w = view(grid.physical,:,6,1)
    #w_ = similar(w)
    @turbo w .= @. -Hb * ((u / r) + ur)
    w_ = @. 0.5 * abs(w) - w
    @turbo vardot[:,6] .= 0.0
    #F[:,6] .= 0.0

    @turbo ADV .= @. (-(u * ur)) + (-v * ul / r) #UADV
    @turbo DRAG .= @. -(Cd * U * u / Hb) #UDRAG
    @turbo COR .= @. ((f * v) + ((v * v) / r)) #UCOR
    @turbo PGF .= @. (-g * hr) #UPGF
    @turbo W_ .= @. -(w_ * (u / Hb)) #UW
    @turbo KDIFF .= @. K * ((ur / r) + urr - (u / (r * r)) + (ull / (r * r)) - (2.0 * vl / (r * r))) #UKDIFF

    @turbo vardot[:,4] .= @. ADV + DRAG + COR + PGF + W_ + KDIFF
    #F[:,4] .= 0.0

    @turbo ADV .= @. (-u * (f + (v / r) + vr)) + (-v * vl / r) #VADV
    @turbo DRAG .= @. -(Cd * U * v / Hb) #VDRAG
    @turbo COR .= @. ((f * u) + ((u * v) / r)) #UCOR
    @turbo PGF .= @. (-g * (hl / r)) #VPGF
    @turbo W_ .= @. w_ * (vg - v) / Hb #VW
    @turbo KDIFF .= @. K * ((vr / r) + vrr - (v / (r * r)) + (vll / (r * r)) + (2.0 * ul / (r * r))) #VKDIFF

    @turbo vardot[:,5] .= @. ADV + COR + DRAG + PGF + W_ + KDIFF
    #F[:,5] .= 0.0
end

function Twoway_ShallowWater_Slab(grid::RL_Grid,
            gridpoints::Array{Float64},
            vardot::Array{Float64},
            F::Array{Float64},
            model::ModelParameters)

    # Physical parameters
    g = model.physical_params[:g]
    K = model.physical_params[:K]
    Cd = model.physical_params[:Cd]
    Hfree = model.physical_params[:Hfree]
    Hb = model.physical_params[:Hb]
    f = model.physical_params[:f]
    S1 = model.physical_params[:S1]

    r = view(gridpoints,:,1)

    h = view(grid.physical,:,1,1)
    hr = view(grid.physical,:,1,2)
    hrr = view(grid.physical,:,1,3)
    hl = view(grid.physical,:,1,4)
    hll = view(grid.physical,:,1,5)

    ug = view(grid.physical,:,2,1)
    ugr = view(grid.physical,:,2,2)
    ugrr = view(grid.physical,:,2,3)
    ugl = view(grid.physical,:,2,4)
    ugll = view(grid.physical,:,2,5)

    vg = view(grid.physical,:,3,1)
    vgr = view(grid.physical,:,3,2)
    vgrr = view(grid.physical,:,3,3)
    vgl = view(grid.physical,:,3,4)
    vgll = view(grid.physical,:,3,5)

    ADV = similar(r)
    DRAG = similar(r)
    COR = similar(r)
    PGF = similar(r)
    W_ = similar(r)
    KDIFF = similar(r)

    @turbo ADV .= @. (-vg * hl / r) + (-ug * hr) #HADV

    #Divergence but use PGF array to reduce memory allocations
    @turbo PGF .= @. (-(Hfree + h) * ((ug / r) + ugr + (vgl / r)))

    # h tendency includes W as mass sink/source
    # S = w * S1
    # Use COR array to reduce memory allocations
    @turbo COR .= @. -(Hfree + h) * w * S1

    @turbo vardot[:,1] .= @. ADV + PGF + COR
    #F[:,1] .= 0.0

    @turbo ADV .= @. (-vg * ugl / r) + (-ug * ugr) #UGADV
    @turbo PGF .= @. (-g * hr) #UGPGF
    @turbo COR .= @. (vg * (f + (vg / r))) #COR

    # ug tendency
    vardot[:,2] .= @. ADV + PGF + COR
    #F[:,2] .= 0.0

    @turbo ADV .= @. (-vg * vgl / r) + (-ug * vgr) #UGADV
    @turbo PGF .= @. (-g * (hl / r)) #UGPGF
    @turbo COR .= @. (-ug * (f + (vg / r))) #COR

    # vg tendency
    @turbo vardot[:,3] .= @. ADV + PGF + COR
    #F[:,3] .= 0.0

    u = view(grid.physical,:,4,1)
    ur = view(grid.physical,:,4,2)
    urr = view(grid.physical,:,4,3)
    ul = view(grid.physical,:,4,4)
    ull = view(grid.physical,:,4,5)

    v = view(grid.physical,:,5,1)
    vr = view(grid.physical,:,5,2)
    vrr = view(grid.physical,:,5,3)
    vl = view(grid.physical,:,5,4)
    vll = view(grid.physical,:,5,5)

    sfc_factor = 0.78
    U = similar(u)
    @turbo U .= @. sfc_factor * sqrt((u * u) + (v * v))

    # W is diagnostic
    w = view(grid.physical,:,6,1)
    #w_ = similar(w)
    @turbo w .= @. -Hb * ((u / r) + ur)
    w_ = @. 0.5 * abs(w) - w
    @turbo vardot[:,6] .= 0.0
    #F[:,6] .= 0.0

    @turbo ADV .= @. (-(u * ur)) + (-v * ul / r) #UADV
    @turbo DRAG .= @. -(Cd * U * u / Hb) #UDRAG
    @turbo COR .= @. ((f * v) + ((v * v) / r)) #UCOR
    @turbo PGF .= @. (-g * hr) #UPGF
    @turbo W_ .= @. -(w_ * (u / Hb)) #UW
    @turbo KDIFF .= @. K * ((ur / r) + urr - (u / (r * r)) + (ull / (r * r)) - (2.0 * vl / (r * r))) #UKDIFF

    @turbo vardot[:,4] .= @. ADV + DRAG + COR + PGF + W_ + KDIFF
    #F[:,4] .= 0.0

    @turbo ADV .= @. (-u * (f + (v / r) + vr)) + (-v * vl / r) #VADV
    @turbo DRAG .= @. -(Cd * U * v / Hb) #VDRAG
    @turbo COR .= @. ((f * u) + ((u * v) / r)) #UCOR
    @turbo PGF .= @. (-g * (hl / r)) #VPGF
    @turbo W_ .= @. w_ * (vg - v) / Hb #VW
    @turbo KDIFF .= @. K * ((vr / r) + vrr - (v / (r * r)) + (vll / (r * r)) + (2.0 * ul / (r * r))) #VKDIFF

    @turbo vardot[:,5] .= @. ADV + COR + DRAG + PGF + W_ + KDIFF
    #F[:,5] .= 0.0
end

function Twoway_ShallowWater_Slab_old(grid::RL_Grid,
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

function Straka_old(grid::RZ_Grid,
            gridpoints::Array{Float64},
            vardot::Array{Float64},
            F::Array{Float64},
            model::ModelParameters)

    # Physical parameters
    Cp = model.physical_params[:Cp]
    Rd = model.physical_params[:Rd]
    Cv = model.physical_params[:Cv]
    p0 = model.physical_params[:p0]
    g = model.physical_params[:g]
    K = model.physical_params[:K]
    Ts = model.physical_params[:Ts]

    x = view(gridpoints,:,1)
    z = view(gridpoints,:,2)

    # Basic state
    theta0 = Ts
    theta0z = 0.0
    T0 = @. Ts - (g / Cp) * z
    #pbar = p0 * (Tbar / Ts)^(Rd / Cp)
    exner0 = T0 ./ theta0
    exner0z = -(g / Cp) / theta0

    # Variables
    u = view(grid.physical,:,1,1)
    ux = view(grid.physical,:,1,2)
    uxx = view(grid.physical,:,1,3)
    uz = view(grid.physical,:,1,4)
    uzz = view(grid.physical,:,1,5)

    w = view(grid.physical,:,2,1)
    wx = view(grid.physical,:,2,2)
    wxx = view(grid.physical,:,2,3)
    wz = view(grid.physical,:,2,4)
    wzz = view(grid.physical,:,2,5)

    theta = view(grid.physical,:,3,1)
    thetax = view(grid.physical,:,3,2)
    thetaxx = view(grid.physical,:,3,3)
    thetaz = view(grid.physical,:,3,4)
    thetazz = view(grid.physical,:,3,5)
    
    exner = view(grid.physical,:,4,1)
    exnerx = view(grid.physical,:,4,2)
    exnerxx = view(grid.physical,:,4,3)
    exnerz = view(grid.physical,:,4,4)
    exnerzz = view(grid.physical,:,4,5)
    
    ADV = similar(x)
    PGF = similar(x)
    KDIFF = similar(x)
    
    @turbo ADV .= @. (-u * ux) + (-w * uz) #UADV
    @turbo PGF .= @. -Cp * (theta + theta0) * exnerx #UPGF
    @turbo KDIFF .= @. K * (uxx + uzz)
    @turbo vardot[:,1] .= @. ADV + PGF + KDIFF

    @turbo ADV .= @. (-u * wx) + (-w * wz) #WADV
    @turbo PGF .= @. (-Cp * (theta + theta0) * exnerz) + (g * theta / theta0)  #WPGF
    @turbo KDIFF .= @. K * (wxx + wzz)
    @turbo vardot[:,2] .= @. ADV + PGF + KDIFF

    @turbo ADV .= @. (-u * thetax) + (-w * (thetaz + theta0z)) #THETAADV 
    #@turbo PGF .= 0.0
    @turbo KDIFF .= @. K * (thetaxx + thetazz)
    @turbo vardot[:,3] .= @. ADV + KDIFF
    
    @turbo ADV .= @. (-u * exnerx) + (-w * (exnerz + exner0z)) #SADV
    @turbo PGF .= @. -(exner + exner0) * (Rd / Cv) * (ux + wz)
    #@turbo KDIFF .= 0.0
    @turbo vardot[:,4] .= @. ADV + PGF

end

# Module end
#end
