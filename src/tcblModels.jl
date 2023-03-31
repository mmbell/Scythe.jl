# TCBL models

function Williams2013_slabTCBL(mtile::ModelTile, colstart::Int64, colend::Int64)

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

function RL_SlabTCBL(mtile::ModelTile, colstart::Int64, colend::Int64)

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

function Kepert2017_TCBL(mtile::ModelTile, colstart::Int64, colend::Int64)

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


