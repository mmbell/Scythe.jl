function Oneway_ShallowWater_Slab(mtile::ModelTile, colstart::Int64, colend::Int64)

    # One-way Shallow Water model on top of slab BL
    grid = mtile.tile
    gridpoints = mtile.tilepoints
    expdot = mtile.expdot_n
    model = mtile.model

    # Physical parameters
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
    @turbo expdot[:,1] .= @. ADV + PGF

    @turbo ADV .= @. (-vg * ugl / r) + (-ug * ugr) #UGADV
    @turbo PGF .= @. (-g * hr) #UGPGF
    @turbo COR .= @. (vg * (f + (vg / r))) #COR

    # ug tendency
    expdot[:,2] .= @. ADV + PGF + COR

    @turbo ADV .= @. (-vg * vgl / r) + (-ug * vgr) #UGADV
    @turbo PGF .= @. (-g * (hl / r)) #UGPGF
    @turbo COR .= @. (-ug * (f + (vg / r))) #COR

    # vg tendency
    @turbo expdot[:,3] .= @. ADV + PGF + COR

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
    @turbo w .= @. -Hb * ((u / r) + ur)
    w_ = @. 0.5 * abs(w) - w
    @turbo expdot[:,6] .= 0.0

    @turbo ADV .= @. (-(u * ur)) + (-v * ul / r) #UADV
    @turbo DRAG .= @. -(Cd * U * u / Hb) #UDRAG
    @turbo COR .= @. ((f * v) + ((v * v) / r)) #UCOR
    @turbo PGF .= @. (-g * hr) #UPGF
    @turbo W_ .= @. -(w_ * (u / Hb)) #UW
    @turbo KDIFF .= @. K * ((ur / r) + urr - (u / (r * r)) + (ull / (r * r)) - (2.0 * vl / (r * r))) #UKDIFF

    @turbo expdot[:,4] .= @. ADV + DRAG + COR + PGF + W_ + KDIFF

    @turbo ADV .= @. (-u * (f + (v / r) + vr)) + (-v * vl / r) #VADV
    @turbo DRAG .= @. -(Cd * U * v / Hb) #VDRAG
    @turbo COR .= @. ((f * u) + ((u * v) / r)) #UCOR
    @turbo PGF .= @. (-g * (hl / r)) #VPGF
    @turbo W_ .= @. w_ * (vg - v) / Hb #VW
    @turbo KDIFF .= @. K * ((vr / r) + vrr - (v / (r * r)) + (vll / (r * r)) + (2.0 * ul / (r * r))) #VKDIFF

    @turbo expdot[:,5] .= @. ADV + COR + DRAG + PGF + W_ + KDIFF
end

function Twoway_ShallowWater_Slab(mtile::ModelTile, colstart::Int64, colend::Int64)

    # Two-way Shallow Water model on top of slab BL
    grid = mtile.tile
    gridpoints = mtile.tilepoints
    expdot = mtile.expdot_n
    model = mtile.model

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

    @turbo expdot[:,1] .= @. ADV + PGF + COR

    @turbo ADV .= @. (-vg * ugl / r) + (-ug * ugr) #UGADV
    @turbo PGF .= @. (-g * hr) #UGPGF
    @turbo COR .= @. (vg * (f + (vg / r))) #COR

    # ug tendency
    expdot[:,2] .= @. ADV + PGF + COR

    @turbo ADV .= @. (-vg * vgl / r) + (-ug * vgr) #UGADV
    @turbo PGF .= @. (-g * (hl / r)) #UGPGF
    @turbo COR .= @. (-ug * (f + (vg / r))) #COR

    # vg tendency
    @turbo expdot[:,3] .= @. ADV + PGF + COR

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
    @turbo w .= @. -Hb * ((u / r) + ur)
    w_ = @. 0.5 * abs(w) - w
    @turbo expdot[:,6] .= 0.0

    @turbo ADV .= @. (-(u * ur)) + (-v * ul / r) #UADV
    @turbo DRAG .= @. -(Cd * U * u / Hb) #UDRAG
    @turbo COR .= @. ((f * v) + ((v * v) / r)) #UCOR
    @turbo PGF .= @. (-g * hr) #UPGF
    @turbo W_ .= @. -(w_ * (u / Hb)) #UW
    @turbo KDIFF .= @. K * ((ur / r) + urr - (u / (r * r)) + (ull / (r * r)) - (2.0 * vl / (r * r))) #UKDIFF

    @turbo expdot[:,4] .= @. ADV + DRAG + COR + PGF + W_ + KDIFF

    @turbo ADV .= @. (-u * (f + (v / r) + vr)) + (-v * vl / r) #VADV
    @turbo DRAG .= @. -(Cd * U * v / Hb) #VDRAG
    @turbo COR .= @. ((f * u) + ((u * v) / r)) #UCOR
    @turbo PGF .= @. (-g * (hl / r)) #VPGF
    @turbo W_ .= @. w_ * (vg - v) / Hb #VW
    @turbo KDIFF .= @. K * ((vr / r) + vrr - (v / (r * r)) + (vll / (r * r)) + (2.0 * ul / (r * r))) #VKDIFF

    @turbo expdot[:,5] .= @. ADV + COR + DRAG + PGF + W_ + KDIFF

end

function LinearShallowWaterRL(mtile::ModelTile, colstart::Int64, colend::Int64)

    #Linear shallow water equations
    grid = mtile.tile
    gridpoints = mtile.tilepoints
    expdot = mtile.expdot_n
    model = mtile.model

    # Physical parameters
    g = model.physical_params[:g]
    K = model.physical_params[:K]
    H = model.physical_params[:H]
    
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
    
    expdot[:,1] .= -H * ((u ./ r) .+ ur .+ (vl ./ r))
    expdot[:,2] .= (-g .* hr) .+ (K .* ((ur ./ r) .+ urr .+ (ull ./ (r .* r)))) 
    expdot[:,3] .= (-g .* (hl ./ r)) .+ (K .* ((vr ./ r) .+ vrr .+ (vll ./ (r .* r))))

end

function ShallowWaterRL(grid::RL_Grid, 
            gridpoints::Array{real},
            expdot::Array{real},
            F::Array{real},
            model::ModelParameters)
   
    #Nonlinear shallow water equations
    grid = mtile.tile
    gridpoints = mtile.tilepoints
    expdot = mtile.expdot_n
    model = mtile.model

    # Physical parameters
    g = model.physical_params[:g]
    K = model.physical_params[:K]
    H = model.physical_params[:H]
    f = model.physical_params[:f]
    
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
    
    expdot[:,1] .= ((-v .* hl ./ r) .+ (-u .* hr) .+
        (-(H .+ h) .* ((u ./ r) .+ ur .+ (vl ./ r))))

    expdot[:,2] .= ((-v .* ul ./ r) .+ (-u .* ur) .+
        (-g .* hr) .+
        (v .* (f .+ (v ./ r))) .+
        (K .* ((ur ./ r) .+ urr .+ (ull ./ (r .* r)) .- (u ./ (r .* r)))))
    
    expdot[:,3] .= ((-v .* vl ./ r) .+ (-u .* vr) .+
        (-g .* (hl ./ r)) .+
        (-u .* (f .+ (v ./ r))) .+
        (K .* ((vr ./ r) .+ vrr .+ (vll ./ (r .* r)) .- (v ./ (r .* r)))))

end
