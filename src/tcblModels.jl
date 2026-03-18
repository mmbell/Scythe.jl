# TCBL models

"""
    Williams2013_slabTCBL(mtile, colstart, colend)

Williams (2013) slab tropical cyclone boundary layer model in radial coordinates.
Deprecated: may produce incorrect results.
"""
function Williams2013_slabTCBL(mtile::ModelTile, colstart::Int64, colend::Int64)

    @warn "Williams2013_slabTCBL is deprecated and may produce incorrect results" maxlog=1

    # Williams (2013) slab TCBL
    grid = mtile.tile
    gridpoints = mtile.tilepoints
    expdot = mtile.expdot_n
    model = mtile.model

    # Physical parameters
    K = model.physical_params[:K]
    Cd = model.physical_params[:Cd]
    h = model.physical_params[:h]
    f = model.physical_params[:f]

    # Example values
    #K = 1500.0
    #Cd = 2.4e-3
    #h = 1000.0
    #f = 5.0e-5

    vgr = grid.physical[:,1,1]
    expdot[:,1] .= 0.0
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
    expdot[:,4] .= 0.0

    UADV = -(u .* ur)
    UDRAG = -(Cd .* U .* u ./ h)
    UCOR = ((f .* v) .+ ((v .* v) ./ r))
    UPGF = -((f .* vgr) .+ ((vgr .* vgr) ./ r))
    UW = -(w_ .* (u ./ h))
    UKDIFF = K .* ((ur ./ r) .+ urr .- (u ./ (r .* r)))
    expdot[:,2] .= UADV .+ UDRAG .+ UCOR .+ UPGF .+ UW .+ UKDIFF
    
    VADV = -u .* (f .+ (v ./ r) .+ vr)
    VDRAG = -(Cd .* U .* v ./ h)
    VW = w_ .* (vgr - v) ./ h
    VKDIFF = K .* ((vr ./ r) .+ vrr .- (v ./ (r .* r)))
    expdot[:,3] .= VADV .+ VDRAG .+ VW .+ UKDIFF

end

"""
    RL_SlabTCBL(mtile, colstart, colend)

Slab TCBL in r-lambda polar coordinates based on Williams (2013).
Deprecated: may produce incorrect results.
"""
function RL_SlabTCBL(mtile::ModelTile, colstart::Int64, colend::Int64)

    @warn "RL_SlabTCBL is deprecated and may produce incorrect results" maxlog=1

    # Williams (2013) slab TCBL in polar coordinates
    grid = mtile.tile
    gridpoints = mtile.tilepoints
    expdot = mtile.expdot_n
    model = mtile.model
   
    # Physical parameters
    K = model.physical_params[:K]
    Cd = model.physical_params[:Cd]
    h = model.physical_params[:h]
    f = model.physical_params[:f]

    vgr = grid.physical[:,1,1]
    expdot[:,1] .= 0.0
    
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
    expdot[:,4] .= 0.0

    UADV = -(u .* ur)
    UDRAG = -(Cd .* U .* u ./ h)
    UCOR = ((f .* v) .+ ((v .* v) ./ r))
    UW = -(w_ .* (u ./ h))
    UKDIFF = K .* ((ur ./ r) .+ urr .- (u ./ (r .* r)))
    expdot[:,2] .= UADV .+ UDRAG .+ UCOR .+ UPGF .+ UW .+ UKDIFF
    
    VADV = -u .* (f .+ (v ./ r) .+ vr)
    VDRAG = -(Cd .* U .* v ./ h)
    VW = w_ .* (vgr - v) ./ h
    VKDIFF = K .* ((vr ./ r) .+ vrr .- (v ./ (r .* r)))
    expdot[:,3] .= VADV .+ VDRAG .+ VW .+ VKDIFF
    
end

"""
    Kepert2017_TCBL(mtile, colstart, colend)

Kepert (2017) height-resolved TCBL in r-z coordinates with Louis mixing length.
Deprecated: incomplete, differentiation and integration of K and W need rework.
"""
function Kepert2017_TCBL(mtile::ModelTile, colstart::Int64, colend::Int64)

    @warn "Kepert2017_TCBL is deprecated and may produce incorrect results" maxlog=1

    # This code won't work now! Need to re-do the differentiation and integration of K and W
    
    # Kepert (2017) height-resolved TCBL
    grid = mtile.tile
    gridpoints = mtile.tilepoints
    expdot = mtile.expdot_n
    model = mtile.model

    # Physical parameters
    K = model.physical_params[:K]
    Cd = model.physical_params[:Cd]
    f = model.physical_params[:f]

    # Example parameters
    #K = 1500.0
    #Cd = 2.4e-3
    #f = 5.0e-5
    
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

"""
    Kepert2017_HeightResolvedTCBL(mtile, colstart, colend, t)

Height-resolved axisymmetric TC boundary layer model from Williams (2017) with Louis
mixing length vertical diffusivity and wind-speed-dependent surface drag.
"""
function Kepert2017_HeightResolvedTCBL(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    # Equation set for an axisymmetric, height-resolved TC boundary layer model reproduced 
    # from Williams (2017): Time and Space Scales in the Tropical Cyclone Boundary Layer,
    # and the Location of the Eyewall Updraft
    
    # Local helper variables
    grid = mtile.tile
    gridpoints = mtile.tilepoints
    expdot = mtile.expdot_n
    model = mtile.model

    # Physical parameters
    g = model.physical_params[:g]
    Kh = model.physical_params[:Kh]
    Cd = model.physical_params[:Cd]
    Hfree = model.physical_params[:Hfree]
    f = model.physical_params[:f]

    # Assign local variables with views
    r = view(gridpoints,colstart:colend,1)
    z = view(gridpoints,colstart:colend,2)   
    h = view(grid.physical,colstart:colend,1,1)
    hr = view(grid.physical,colstart:colend,1,2)
    hrr = view(grid.physical,colstart:colend,1,3)
    ug = view(grid.physical,colstart:colend,2,1)
    ugr = view(grid.physical,colstart:colend,2,2)
    ugrr = view(grid.physical,colstart:colend,2,3)

    vg = view(grid.physical,colstart:colend,3,1)
    vgr = view(grid.physical,colstart:colend,3,2)
    vgrr = view(grid.physical,colstart:colend,3,3)
    
    ub = view(grid.physical,colstart:colend,4,1)
    ubr = view(grid.physical,colstart:colend,4,2)
    ubrr = view(grid.physical,colstart:colend,4,3)
    ubz = view(grid.physical,colstart:colend,4,4)
    ubzz = view(grid.physical,colstart:colend,4,5)
    vb = view(grid.physical,colstart:colend,5,1)
    vbr = view(grid.physical,colstart:colend,5,2)
    vbrr = view(grid.physical,colstart:colend,5,3)
    vbz = view(grid.physical,colstart:colend,5,4)
    vbzz = view(grid.physical,colstart:colend,5,5)

    # Helper arrays to reduce memory allocations
    zDim = mtile.model.grid_params.zDim
    ADV = similar(r)
    COR = similar(r)
    PGF = similar(r)
    HDIFF = similar(r)
    VDIFF = similar(r)
    
    # Calculate the vertical diffusivity
    # Mixing length based on Louis parameterization
    S = sqrt.((ubz .* ubz) .+ (vbz .* vbz))
    l = 1.0 ./ ((1.0 ./ (0.4 .* z)) .+ (1.0 ./ 80.0))
    Kv = (l.^2) .* S

    # W is diagnostic and is needed first for other calculations
    wb = view(grid.physical,colstart:colend,6,1)

    # Integrate divergence to get W
    # Use h since it doesn't have any boundary conditions in the vertical
    h_col = deepcopy(mtile.tile.columns[mtile.model.grid_params.vars["h"]])
    col = Chebyshev1D(h_col.params,h_col.mishPoints,h_col.gammaBC,
        h_col.fftPlan,h_col.filter,h_col.uMish,h_col.b,h_col.a,h_col.ax)
    col.uMish .= @. -((ub / r) + ubr)
    CBtransform!(col)
    CAtransform!(col)
    wb .= CIInttransform(col)
    expdot[colstart:colend,6] .= 0.0 # TB: wb tendency = 0 because wb is diagnostic

    # h tendency
    expdot[colstart:colend,1] .= 0.0 
    # ug tendency
    expdot[colstart:colend,2] .= 0.0 
    # vg tendency
    expdot[colstart:colend,3] .= 0.0 

    # ub tendency
    ADV .= @. (-ub * ubr) + (-wb * ubz) #UBADV
    # PGF .= @. (-g * hr) # TB: UBPGF using h 
    PGF .= @. (-vg * (f + (vg / r))) # TB: UBPGF using v
    COR .= @. (vb * (f + (vb / r))) #UBCOR

    # Horizontal diffusion
    HDIFF .= @. Kh * ((ubr / r) + ubrr - (ub / (r * r))) #UHDIFF
    # The following is just the Laplacian term without the curvature terms from Batchelor (1967) and Shapiro (1983)
    #HDIFF .= @. K * ((ur / r) + urr + (ull / (r * r))) #UKDIFF

    # Get the 10 meter wind (assuming 10 m @ z == 2)
    u10 = ub[2]
    v10 = vb[2]
    U10 = sqrt(u10^2 + v10^2)

    # Differentiate Kv * du/dz
    col.uMish .= Kv .* ubz
    
    # Drag applies at z = 0
    if Cd < 0.0
        # Negative parameter so use a wind speed dependent drag
        # From Komori et al. (2018)
        if U10 < 5.2
            Cd = 1.0e-3
        elseif U10 < 33.6
            Cd = 4.4e-4 * U10^0.5
        else
            Cd = 2.55e-3
        end
    end
    col.uMish[1] = Cd * U10 * u10 #UDRAG

    CBtransform!(col)
    CAtransform!(col)
    VDIFF .= CIxtransform(col)

    expdot[colstart:colend,4] .= @. ADV + PGF + COR + VDIFF + HDIFF

    # vb tendency
    ADV .= @. (-ub * vbr) + (-wb * vbz) #VBADV
    PGF .= 0.0 #VBPGF # TB: there is no L pressure gradient in an axisymmetric storm
    COR .= @. (-ub * (f + (vb / r))) #VBCOR

    # Horizontal diffusion
    HDIFF .= @. Kh * ((vbr / r) + vbrr - (vb / (r * r))) #VHDIFF
    # The following is just the Laplacian term without the curvature terms from Batchelor (1967) and Shapiro (1983)
    #HDIFF .= @. K * ((vr / r) + vrr + (vll / (r * r))) #VKDIFF

    # TB: Vertical diffusion
    # TB: Kv * vbz is differentiated wrt z in the TCBL equations
    # TB: First assign to col and then do the transform, then assign to VDIFF variable
    # Differentiate Kv * dv/dz
    col.uMish .= Kv .* vbz

    # Drag only applies at z = 0
    col.uMish[1] = Cd * U10 * v10 #VDRAG

    CBtransform!(col)
    CAtransform!(col)
    VDIFF .= CIxtransform(col)
    
    expdot[colstart:colend,5] .= @. ADV + PGF + COR + VDIFF + HDIFF

    # Advance the explicit terms
    explicit_timestep(mtile, colstart, colend, t)

end

"""
    RLZ_HeightResolvedBL(mtile, colstart, colend, t)

Height-resolved boundary layer in r-lambda-z coordinates with fixed pressure gradient
from the gradient wind. Uses Louis mixing length and wind-speed-dependent drag.
"""
function RLZ_HeightResolvedBL(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    # Height resolved boundary layer with fixed pressure gradient from shallow water layer

    # Local helper variables
    grid = mtile.tile
    gridpoints = mtile.tilepoints
    expdot = mtile.expdot_n
    model = mtile.model

    # Physical parameters
    g = model.physical_params[:g]
    Kh = model.physical_params[:Kh]
    Cd = model.physical_params[:Cd]
    Hfree = model.physical_params[:Hfree]
    f = model.physical_params[:f]
    Um = model.physical_params[:Um]
    Vm = model.physical_params[:Vm]

    # Assign local variables with views
    r = view(gridpoints,colstart:colend,1)
    lambda = view(gridpoints,colstart:colend,2)
    z = view(gridpoints,colstart:colend,3)

    vg = view(grid.physical,colstart:colend,1,1)
    vgr = view(grid.physical,colstart:colend,1,2)
    vgrr = view(grid.physical,colstart:colend,1,3)
    vgl = view(grid.physical,colstart:colend,1,4)
    vgll = view(grid.physical,colstart:colend,1,5)

    ub = view(grid.physical,colstart:colend,2,1)
    ubr = view(grid.physical,colstart:colend,2,2)
    ubrr = view(grid.physical,colstart:colend,2,3)
    ubl = view(grid.physical,colstart:colend,2,4)
    ubll = view(grid.physical,colstart:colend,2,5)
    ubz = view(grid.physical,colstart:colend,2,6)
    ubzz = view(grid.physical,colstart:colend,2,7)

    vb = view(grid.physical,colstart:colend,3,1)
    vbr = view(grid.physical,colstart:colend,3,2)
    vbrr = view(grid.physical,colstart:colend,3,3)
    vbl = view(grid.physical,colstart:colend,3,4)
    vbll = view(grid.physical,colstart:colend,3,5)
    vbz = view(grid.physical,colstart:colend,3,6)
    vbzz = view(grid.physical,colstart:colend,3,7)

    # Helper arrays to reduce memory allocations
    zDim = mtile.model.grid_params.zDim
    ADV = similar(r)
    COR = similar(r)
    PGF = similar(r)
    HDIFF = similar(r)
    VDIFF = similar(r)

    # Calculate the vertical diffusivity
    # Mixing length based on Louis parameterization
    S = sqrt.((ubz .* ubz) .+ (vbz .* vbz))
    l = 1.0 ./ ((1.0 ./ (0.4 .* z)) .+ (1.0 ./ 80.0))
    Kv = (l.^2) .* S

    # W is diagnostic and is needed first for other calculations
    wb = view(grid.physical,colstart:colend,4,1)

    # Integrate divergence to get W
    # Use vg since it doesn't have any boundary conditions in the vertical
    col = deepcopy(mtile.tile.columns[mtile.model.grid_params.vars["vg"]])
    #col = Chebyshev1D(h_col.params,h_col.mishPoints,h_col.gammaBC,
    #    h_col.fftPlan,h_col.filter,h_col.uMish,h_col.b,h_col.a,h_col.ax)
    col.uMish .= @. -((ub / r) + ubr + (vbl / r))
    CBtransform!(col)
    CAtransform!(col)
    wb .= CIInttransform(col)
    expdot[colstart:colend,4] .= 0.0

    # vg tendency
    expdot[colstart:colend,1] .= 0.0

    # ub tendency
    ADV .= @. (-vb * ubl / r) + (-ub * ubr) + (-wb * ubz) #UBADV
    #PGF .= @. (-g * hr) #UBPGF
    PGF .= @. (-vg * (f + (vg / r)))
    COR .= @. (vb * (f + (vb / r))) #UBCOR

    # Horizontal diffusion
    HDIFF .= @. Kh * ((ubr / r) + ubrr - (ub / (r * r)) + (ubll / (r * r)) - (2.0 * vbl / (r * r))) #UHDIFF
    # The following is just the Laplacian term without the curvature terms from Batchelor (1967) and Shapiro (1983)
    #HDIFF .= @. K * ((ur / r) + urr + (ull / (r * r))) #UKDIFF

    # Surface wind speed based on storm motion
    sfcu = (Um * cos(lambda[1])) + (Vm * sin(lambda[1]))
    sfcv = (Vm * cos(lambda[1])) - (Um * sin(lambda[1]))

    # Get the 10 meter wind (assuming 10 m @ z == 2)
    u10 = ub[2] + sfcu
    v10 = vb[2] + sfcv
    U10 = sqrt(u10^2 + v10^2)

    # Differentiate Kv * du/dz
    col.uMish .= Kv .* ubz

    # Drag applies at z = 0
    if Cd < 0.0
        # Use a wind speed dependent drag
        if U10 < 5.2
            Cd = 1.0e-3
        elseif U10 < 33.6
            Cd = 4.4e-4 * U10^0.5
        else
            Cd = 2.55e-3
        end
    end
    col.uMish[1] = Cd * U10 * u10 #UDRAG

    CBtransform!(col)
    CAtransform!(col)
    VDIFF .= CIxtransform(col)

    expdot[colstart:colend,2] .= @. ADV + PGF + COR + VDIFF + HDIFF

    # vb tendency
    ADV .= @. (-vb * vbl / r) + (-ub * vbr) + (-wb * vbz)#VBADV
    PGF .= @. 0.0 #(-g * (hl / r)) #VBPGF
    COR .= @. (-ub * (f + (vb / r))) #VBCOR

    # Horizontal diffusion
    HDIFF .= @. Kh * ((vbr / r) + vbrr - (vb / (r * r)) + (vbll / (r * r)) + (2.0 * ubl / (r * r))) #VHDIFF
    # The following is just the Laplacian term without the curvature terms from Batchelor (1967) and Shapiro (1983)
    #HDIFF .= @. K * ((vr / r) + vrr + (vll / (r * r))) #VKDIFF

    # Differentiate Kv * dv/dz
    col.uMish .= Kv .* vbz

    # Drag only applies at z = 0
    col.uMish[1] = Cd * U10 * v10 #VDRAG

    CBtransform!(col)
    CAtransform!(col)
    VDIFF .= CIxtransform(col)

    expdot[colstart:colend,3] .= @. ADV + PGF + COR + VDIFF + HDIFF

    # Advance the explicit terms
    explicit_timestep(mtile, colstart, colend, t)

end
