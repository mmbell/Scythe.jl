# Functions that define the model parameters and physical models

using LoopVectorization

#Define some convenient aliases
const real = Float64
const int = Int64
const uint = UInt64


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
