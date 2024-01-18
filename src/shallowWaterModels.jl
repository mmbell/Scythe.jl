function Oneway_ShallowWater_Slab(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    # One-way Shallow Water model on top of slab BL

    # Local helper variables
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

    # Assign local variables with views
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

    ub = view(grid.physical,:,4,1)
    ubr = view(grid.physical,:,4,2)
    ubrr = view(grid.physical,:,4,3)
    ubl = view(grid.physical,:,4,4)
    ubll = view(grid.physical,:,4,5)

    vb = view(grid.physical,:,5,1)
    vbr = view(grid.physical,:,5,2)
    vbrr = view(grid.physical,:,5,3)
    vbl = view(grid.physical,:,5,4)
    vbll = view(grid.physical,:,5,5)

    # Helper arrays to reduce memory allocations
    ADV = similar(r)
    DRAG = similar(r)
    COR = similar(r)
    PGF = similar(r)
    W_ = similar(r)
    KDIFF = similar(r)

    # Parameterized surface wind speed
    sfc_factor = 0.78
    U = similar(ub)
    @turbo U .= @. sfc_factor * sqrt((ub * ub) + (vb * vb))

    # W is diagnostic and is needed first for other calculations
    w = view(grid.physical,:,6,1)
    @turbo w .= @. -Hb * ((ub / r) + ubr + (vbl / r))
    w_ = @. 0.5 * abs(w) - w
    @turbo expdot[:,6] .= 0.0

    # h tendency
    @turbo ADV .= @. (-vg * hl / r) + (-ug * hr) #HADV
    @turbo PGF .= @. (-(Hfree + h) * ((ug / r) + ugr + (vgl / r))) # Divergence but use PGF array to reduce memory allocations
    @turbo expdot[:,1] .= @. ADV + PGF

    # ug tendency
    @turbo ADV .= @. (-vg * ugl / r) + (-ug * ugr) #UGADV
    @turbo PGF .= @. (-g * hr) #UGPGF
    @turbo COR .= @. (vg * (f + (vg / r))) #UCOR
    @turbo expdot[:,2] .= @. ADV + PGF + COR

    # vg tendency
    @turbo ADV .= @. (-vg * vgl / r) + (-ug * vgr) #VGADV
    @turbo PGF .= @. (-g * (hl / r)) #VGPGF
    @turbo COR .= @. (-ug * (f + (vg / r))) #VCOR
    @turbo expdot[:,3] .= @. ADV + PGF + COR

    # ub tendency
    @turbo ADV .= @. (-vb * ubl / r) + (-ub * ubr) #UBADV
    @turbo PGF .= @. (-g * hr) #UBPGF
    @turbo COR .= @. (vb * (f + (vb / r))) #UBCOR
    @turbo DRAG .= @. -(Cd * U * ub / Hb) #UDRAG
    @turbo W_ .= @. w_ * (ug - ub) / Hb #UW
    @turbo KDIFF .= @. K * ((ubr / r) + ubrr - (ub / (r * r)) + (ubll / (r * r)) - (2.0 * vbl / (r * r))) #UKDIFF
    # The following is just the Laplacian term without the curvature terms from Batchelor (1967) and Shapiro (1983)
    #@turbo KDIFF .= @. K * ((ur / r) + urr + (ull / (r * r))) #UKDIFF
    @turbo expdot[:,4] .= @. ADV + PGF + COR + DRAG + W_ + KDIFF

    # vb tendency
    @turbo ADV .= @. (-vb * vbl / r) + (-ub * vbr) #VBADV
    @turbo PGF .= @. (-g * (hl / r)) #VBPGF
    @turbo COR .= @. (-ub * (f + (vb / r))) #VBCOR
    @turbo DRAG .= @. -(Cd * U * vb / Hb) #VDRAG
    @turbo W_ .= @. w_ * (vg - vb) / Hb #VW
    @turbo KDIFF .= @. K * ((vbr / r) + vbrr - (vb / (r * r)) + (vbll / (r * r)) + (2.0 * ubl / (r * r))) #VKDIFF
    # The following is just the Laplacian term without the curvature terms from Batchelor (1967) and Shapiro (1983)
    #@turbo KDIFF .= @. K * ((vr / r) + vrr + (vll / (r * r))) #VKDIFF
    @turbo expdot[:,5] .= @. ADV + PGF + COR + DRAG + W_ + KDIFF

    # Advance the explicit terms
    explicit_timestep(mtile, colstart, colend, t)

end

function Twoway_ShallowWater_Slab(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    # Two-way Shallow Water model on top of slab BL
    # This equation set includes feedbacks back to SWM via the mass sink/source

    # Local helper variables
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

    # Assign local variables with views
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

    ub = view(grid.physical,:,4,1)
    ubr = view(grid.physical,:,4,2)
    ubrr = view(grid.physical,:,4,3)
    ubl = view(grid.physical,:,4,4)
    ubll = view(grid.physical,:,4,5)

    vb = view(grid.physical,:,5,1)
    vbr = view(grid.physical,:,5,2)
    vbrr = view(grid.physical,:,5,3)
    vbl = view(grid.physical,:,5,4)
    vbll = view(grid.physical,:,5,5)

    # Helper arrays to reduce memory allocations
    ADV = similar(r)
    DRAG = similar(r)
    COR = similar(r)
    PGF = similar(r)
    W_ = similar(r)
    KDIFF = similar(r)

    # Parameterized surface wind speed
    sfc_factor = 0.78
    U = similar(ub)
    @turbo U .= @. sfc_factor * sqrt((ub * ub) + (vb * vb))

    # W is diagnostic and is needed first for other calculations
    w = view(grid.physical,:,6,1)
    @turbo w .= @. -Hb * ((ub / r) + ubr + (vbl / r))
    w_ = @. 0.5 * abs(w) - w
    @turbo expdot[:,6] .= 0.0

    # h tendency
    @turbo ADV .= @. (-vg * hl / r) + (-ug * hr) #HADV
    @turbo PGF .= @. (-(Hfree + h) * ((ug / r) + ugr + (vgl / r))) # Divergence but use PGF array 
    # In two-way case, h tendency includes W as mass sink/source
    # S = w * S1
    # Use COR array to reduce memory allocations
    @turbo COR .= @. -(Hfree + h) * w * S1
    @turbo expdot[:,1] .= @. ADV + PGF + COR

    # ug tendency
    @turbo ADV .= @. (-vg * ugl / r) + (-ug * ugr) #UGADV
    @turbo PGF .= @. (-g * hr) #UGPGF
    @turbo COR .= @. (vg * (f + (vg / r))) #UCOR
    @turbo expdot[:,2] .= @. ADV + PGF + COR

    # vg tendency
    @turbo ADV .= @. (-vg * vgl / r) + (-ug * vgr) #VGADV
    @turbo PGF .= @. (-g * (hl / r)) #VGPGF
    @turbo COR .= @. (-ug * (f + (vg / r))) #VCOR
    @turbo expdot[:,3] .= @. ADV + PGF + COR

    # ub tendency
    @turbo ADV .= @. (-vb * ubl / r) + (-ub * ubr) #UBADV
    @turbo PGF .= @. (-g * hr) #UBPGF
    @turbo COR .= @. (vb * (f + (vb / r))) #UBCOR
    @turbo DRAG .= @. -(Cd * U * ub / Hb) #UBDRAG
    @turbo W_ .= @. w_ * (ug - ub) / Hb #UW
    @turbo KDIFF .= @. K * ((ubr / r) + ubrr - (ub / (r * r)) + (ubll / (r * r)) - (2.0 * vbl / (r * r))) #UKDIFF
    # The following is just the Laplacian term without the curvature terms from Batchelor (1967) and Shapiro (1983)
    #@turbo KDIFF .= @. K * ((ur / r) + urr + (ull / (r * r))) #UKDIFF
    @turbo expdot[:,4] .= @. ADV + PGF + COR + DRAG + W_ + KDIFF

    # vb tendency
    @turbo ADV .= @. (-vb * vbl / r) + (-ub * vbr) #VBADV
    @turbo PGF .= @. (-g * (hl / r)) #VBPGF
    @turbo COR .= @. (-ub * (f + (vb / r))) #VBCOR
    @turbo DRAG .= @. -(Cd * U * vb / Hb) #VDRAG
    @turbo W_ .= @. w_ * (vg - vb) / Hb #VW
    @turbo KDIFF .= @. K * ((vbr / r) + vbrr - (vb / (r * r)) + (vbll / (r * r)) + (2.0 * ubl / (r * r))) #VKDIFF
    # The following is just the Laplacian term without the curvature terms from Batchelor (1967) and Shapiro (1983)
    #@turbo KDIFF .= @. K * ((vr / r) + vrr + (vll / (r * r))) #VKDIFF
    @turbo expdot[:,5] .= @. ADV + PGF + COR + DRAG + W_ + KDIFF

    # Advance the explicit terms
    explicit_timestep(mtile, colstart, colend, t)

end

function LinearShallowWaterRL(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

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

    # Advance the explicit terms
    explicit_timestep(mtile, colstart, colend, t)

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

