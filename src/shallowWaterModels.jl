"""
    SW_SCRATCH_SLOTS

Names of the reusable work vectors `Twoway_PV_mixing` needs, one full-tile-length
`Vector{Float64}` each. The set used to allocate all of these on every call — 13 heap arrays per
timestep, ~19 MB at `num_cells = 100`, ~230 GB of churn over a 10-hour run.

Keyed by NAME, not by index, for the same reason as [`MC_SCRATCH_SLOTS`](@ref): an index-numbered
pool makes it easy to hand one buffer to two temporaries that are live at once, and the resulting
corruption is silent. A `NamedTuple` cannot hold a duplicate field, so that bug cannot be written.

Note `:W_` and `:w_up` are DIFFERENT arrays and both are live at the same time: `:W_` is the
BL/free-layer momentum exchange term, `:w_up` is the upward-mass-flux limiter `0.5|w| - w`. The
source spells them `W_` and `w_`, which differ by one character — hence the distinct slot names.
"""
const SW_SCRATCH_SLOTS = (:ADV, :DRAG, :COR, :PGF, :W_, :KDIFF,
                          :S_rr, :S_ll, :S_rl, :K_free, :K_bl, :U, :w_up)

"""
    _allocate_sw_scratch(tile, model)

Work vectors for `Twoway_PV_mixing`; an empty `NamedTuple` for every other equation set.

ONE workspace per tile, not one per thread (unlike `mc_scratch`). `advanceTimestep` only reaches
its `Threads.@threads :static` column loop when `num_columns(tile) > 0`; on an RL grid that is 0,
so it takes the `else` branch and calls the equation set exactly once per timestep, serially, over
the whole tile. There is no concurrency to guard against, and per-thread copies would cost
`maxthreadid()` × 19 MB for nothing. `test/test_allocations.jl` asserts `num_columns == 0` for this
grid so the invariant this relies on cannot silently lapse.

Length is the tile's GRIDPOINT count, not `kDim`: the shallow-water sets have no vertical basis and
treat the entire tile as a single column.
"""
function _allocate_sw_scratch(tile::AbstractGrid, model::ModelParameters)
    model.equation_set == "Twoway_PV_mixing" || return NamedTuple()
    n = size(tile.physical, 1)
    return NamedTuple{SW_SCRATCH_SLOTS}(
        ntuple(_ -> zeros(Float64, n), length(SW_SCRATCH_SLOTS)))
end

"""
    Oneway_ShallowWater_Slab(mtile, colstart, colend, t)

One-way coupled shallow water model with a slab boundary layer in r-lambda coordinates.
The free-atmosphere shallow water layer provides the pressure gradient to the BL, but
the BL does not feed back to the shallow water layer.
"""
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

"""
    Twoway_ShallowWater_Slab(mtile, colstart, colend, t)

Two-way coupled shallow water model with a slab boundary layer in r-lambda coordinates.
Includes mass sink/source feedback from the BL vertical velocity back to the shallow water layer.
"""
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

"""
    Twoway_PV_mixing(mtile, colstart, colend, t)

Two-way shallow water model on top of a slab boundary layer with Smagorinsky-style
turbulent diffusion in both layers. Extends [`Twoway_ShallowWater_Slab`](@ref) by adding
flow-dependent horizontal diffusion to the free-atmosphere winds (ug, vg) and replacing
the constant-K diffusion in the boundary layer with a Smagorinsky parameterization.

The diffusion coefficient is ``K = L_s^2 \\, |S|`` where ``L_s`` is a configurable
length scale and ``|S| = \\sqrt{2 S_{ij} S_{ij}}`` is the strain rate magnitude
computed from the full cylindrical strain rate tensor.

# Required physical parameters
- `:Ls_free`: Smagorinsky length scale for the free-atmosphere (SW) layer (m).
- `:Ls_bl`: Smagorinsky length scale for the boundary layer (m).
- `:K_min_free`: lower bound on the free-atmosphere diffusivity ``K`` (m²/s); set `0`
  to make the free atmosphere inviscid where the strain-dependent `K` is small.
- `:K_min_bl`: lower bound on the boundary-layer diffusivity ``K`` (m²/s).
- `:g`, `:Cd`, `:Hfree`, `:Hb`, `:f`, `:S1`: same as `Twoway_ShallowWater_Slab`.
"""
function Twoway_PV_mixing(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    # Local helper variables
    grid = mtile.tile
    gridpoints = mtile.tilepoints
    expdot = mtile.expdot_n
    model = mtile.model

    # Physical parameters
    g = model.physical_params[:g]
    Ls_free = model.physical_params[:Ls_free]
    Ls_bl = model.physical_params[:Ls_bl]
    K_min_free = model.physical_params[:K_min_free]
    K_min_bl = model.physical_params[:K_min_bl]
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

    # Reusable work vectors, allocated once per tile (see `_allocate_sw_scratch`). These used to be
    # `similar(r)` — a fresh heap array per call, per timestep, the whole tile long.
    #
    # Every slot below is the destination of a whole-array `.=` before it is ever read, so the stale
    # contents left by the previous timestep are unobservable.
    sw = mtile.sw_scratch
    ADV = sw.ADV
    DRAG = sw.DRAG
    COR = sw.COR
    PGF = sw.PGF
    W_ = sw.W_
    KDIFF = sw.KDIFF
    S_rr = sw.S_rr
    S_ll = sw.S_ll
    S_rl = sw.S_rl

    # Compute Smagorinsky diffusion coefficient for the free-atmosphere layer
    # Full cylindrical strain rate tensor using (ug, vg)
    @turbo S_rr .= ugr
    @turbo S_ll .= @. (vgl / r) + (ug / r)
    @turbo S_rl .= @. 0.5 * ((ugl / r) + vgr - (vg / r))
    K_free = sw.K_free
    @. K_free = max(Ls_free * Ls_free * sqrt(2.0 * (S_rr * S_rr + S_ll * S_ll + 2.0 * S_rl * S_rl)), K_min_free)

    # Compute Smagorinsky diffusion coefficient for the boundary layer
    # Full cylindrical strain rate tensor using (ub, vb)
    @turbo S_rr .= ubr
    @turbo S_ll .= @. (vbl / r) + (ub / r)
    @turbo S_rl .= @. 0.5 * ((ubl / r) + vbr - (vb / r))
    K_bl = sw.K_bl
    @. K_bl = max(Ls_bl * Ls_bl * sqrt(2.0 * (S_rr * S_rr + S_ll * S_ll + 2.0 * S_rl * S_rl)), K_min_bl)

    # Parameterized surface wind speed
    sfc_factor = 0.78
    U = sw.U
    @turbo U .= @. sfc_factor * sqrt((ub * ub) + (vb * vb))

    # W is diagnostic and is needed first for other calculations
    w = view(grid.physical,:,6,1)
    @turbo w .= @. -Hb * ((ub / r) + ubr + (vbl / r))
    w_ = sw.w_up
    @. w_ = 0.5 * abs(w) - w
    @turbo expdot[:,6] .= 0.0

    # h tendency (no diffusion on height field)
    @turbo ADV .= @. (-vg * hl / r) + (-ug * hr) #HADV
    @turbo PGF .= @. (-(Hfree + h) * ((ug / r) + ugr + (vgl / r))) # Divergence but use PGF array
    @turbo COR .= @. -(Hfree + h) * w * S1
    @turbo expdot[:,1] .= @. ADV + PGF + COR

    # ug tendency (with Smagorinsky diffusion)
    @turbo ADV .= @. (-vg * ugl / r) + (-ug * ugr) #UGADV
    @turbo PGF .= @. (-g * hr) #UGPGF
    @turbo COR .= @. (vg * (f + (vg / r))) #UCOR
    @turbo KDIFF .= @. K_free * ((ugr / r) + ugrr - (ug / (r * r)) + (ugll / (r * r)) - (2.0 * vgl / (r * r)))
    @turbo expdot[:,2] .= @. ADV + PGF + COR + KDIFF

    # vg tendency (with Smagorinsky diffusion)
    @turbo ADV .= @. (-vg * vgl / r) + (-ug * vgr) #VGADV
    @turbo PGF .= @. (-g * (hl / r)) #VGPGF
    @turbo COR .= @. (-ug * (f + (vg / r))) #VCOR
    @turbo KDIFF .= @. K_free * ((vgr / r) + vgrr - (vg / (r * r)) + (vgll / (r * r)) + (2.0 * ugl / (r * r)))
    @turbo expdot[:,3] .= @. ADV + PGF + COR + KDIFF

    # ub tendency (with Smagorinsky diffusion)
    @turbo ADV .= @. (-vb * ubl / r) + (-ub * ubr) #UBADV
    @turbo PGF .= @. (-g * hr) #UBPGF
    @turbo COR .= @. (vb * (f + (vb / r))) #UBCOR
    @turbo DRAG .= @. -(Cd * U * ub / Hb) #UBDRAG
    @turbo W_ .= @. w_ * (ug - ub) / Hb #UW
    @turbo KDIFF .= @. K_bl * ((ubr / r) + ubrr - (ub / (r * r)) + (ubll / (r * r)) - (2.0 * vbl / (r * r)))
    @turbo expdot[:,4] .= @. ADV + PGF + COR + DRAG + W_ + KDIFF

    # vb tendency (with Smagorinsky diffusion)
    @turbo ADV .= @. (-vb * vbl / r) + (-ub * vbr) #VBADV
    @turbo PGF .= @. (-g * (hl / r)) #VBPGF
    @turbo COR .= @. (-ub * (f + (vb / r))) #VBCOR
    @turbo DRAG .= @. -(Cd * U * vb / Hb) #VDRAG
    @turbo W_ .= @. w_ * (vg - vb) / Hb #VW
    @turbo KDIFF .= @. K_bl * ((vbr / r) + vbrr - (vb / (r * r)) + (vbll / (r * r)) + (2.0 * ubl / (r * r)))
    @turbo expdot[:,5] .= @. ADV + PGF + COR + DRAG + W_ + KDIFF

    # Advance the explicit terms
    explicit_timestep(mtile, colstart, colend, t)

end

"""
    LinearShallowWater1D(mtile, colstart, colend, t)

1D linear shallow water equations with diffusion in the radial direction.
"""
function LinearShallowWater1D(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

    #Linear shallow water equations in 1D
    grid = mtile.tile
    expdot = mtile.expdot_n
    model = mtile.model

    g = model.physical_params[:g]
    K = model.physical_params[:K]
    H = model.physical_params[:H]

    #h = view(grid.physical,:,1,1)
    h_r = view(grid.physical,:,1,2)
    #hrr = view(grid.physical,:,1,3)
    #u = view(grid.physical,:,2,1)
    u_r = view(grid.physical,:,2,2)
    u_rr = view(grid.physical,:,2,3)

    expdot[:,1] .= @. -H * u_r
    expdot[:,2] .= @. (-g * h_r) + (K * u_rr)

    # Advance the explicit terms
    explicit_timestep(mtile, colstart, colend, t)

end

"""
    LinearShallowWaterRL(mtile, colstart, colend, t)

2D linear shallow water equations in r-lambda polar coordinates with diffusion.
"""
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

"""
    ShallowWaterRL(mtile, colstart, colend, t)

Nonlinear shallow water equations in r-lambda polar coordinates with Coriolis, nonlinear
advection, and diffusion.
"""
function ShallowWaterRL(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)
   
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

"""
    Oneway_ShallowWater_HeightResolvedBL(mtile, colstart, colend, t)

One-way coupled shallow water model with a height-resolved boundary layer in r-lambda-z
coordinates. Uses Louis mixing length parameterization for vertical diffusivity and
wind-speed-dependent surface drag.
"""
function Oneway_ShallowWater_HeightResolvedBL(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

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
    
    h = view(grid.physical,colstart:colend,1,1)
    hr = view(grid.physical,colstart:colend,1,2)
    hrr = view(grid.physical,colstart:colend,1,3)
    hl = view(grid.physical,colstart:colend,1,4)
    hll = view(grid.physical,colstart:colend,1,5)

    ug = view(grid.physical,colstart:colend,2,1)
    ugr = view(grid.physical,colstart:colend,2,2)
    ugrr = view(grid.physical,colstart:colend,2,3)
    ugl = view(grid.physical,colstart:colend,2,4)
    ugll = view(grid.physical,colstart:colend,2,5)

    vg = view(grid.physical,colstart:colend,3,1)
    vgr = view(grid.physical,colstart:colend,3,2)
    vgrr = view(grid.physical,colstart:colend,3,3)
    vgl = view(grid.physical,colstart:colend,3,4)
    vgll = view(grid.physical,colstart:colend,3,5)

    ub = view(grid.physical,colstart:colend,4,1)
    ubr = view(grid.physical,colstart:colend,4,2)
    ubrr = view(grid.physical,colstart:colend,4,3)
    ubl = view(grid.physical,colstart:colend,4,4)
    ubll = view(grid.physical,colstart:colend,4,5)
    ubz = view(grid.physical,colstart:colend,4,6)
    ubzz = view(grid.physical,colstart:colend,4,7)

    vb = view(grid.physical,colstart:colend,5,1)
    vbr = view(grid.physical,colstart:colend,5,2)
    vbrr = view(grid.physical,colstart:colend,5,3)
    vbl = view(grid.physical,colstart:colend,5,4)
    vbll = view(grid.physical,colstart:colend,5,5)
    vbz = view(grid.physical,colstart:colend,5,6)
    vbzz = view(grid.physical,colstart:colend,5,7)

    # Helper arrays to reduce memory allocations
    zDim = mtile.model.grid_params.kDim
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
    col = deepcopy(mtile.tile.kbasis.data[mtile.model.grid_params.vars["h"]])
    col.uMish .= @. -((ub / r) + ubr + (vbl / r))
    Btransform!(col)
    Atransform!(col)
    wb .= IInttransform(col)
    expdot[colstart:colend,6] .= 0.0

    # h tendency
    ADV .= @. (-vg * hl / r) + (-ug * hr) #HADV
    PGF .= @. (-(Hfree + h) * ((ug / r) + ugr + (vgl / r))) # Divergence but use PGF array to reduce memory allocations
    expdot[colstart:colend,1] .= @. ADV + PGF

    # ug tendency
    ADV .= @. (-vg * ugl / r) + (-ug * ugr) #UGADV
    PGF .= @. (-g * hr) #UGPGF
    COR .= @. (vg * (f + (vg / r))) #UCOR
    expdot[colstart:colend,2] .= @. ADV + PGF + COR

    # vg tendency
    ADV .= @. (-vg * vgl / r) + (-ug * vgr) #VGADV
    PGF .= @. (-g * (hl / r)) #VGPGF
    COR .= @. (-ug * (f + (vg / r))) #VCOR
    expdot[colstart:colend,3] .= @. ADV + PGF + COR

    # ub tendency
    ADV .= @. (-vb * ubl / r) + (-ub * ubr) + (-wb * ubz) #UBADV
    PGF .= @. (-g * hr) #UBPGF
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
    # Use a wind speed dependent drag
    if U10 < 5.2
        Cd = 1.0e-3
    elseif U10 < 33.6
        Cd = 4.4e-4 * U10^0.5
    end
    col.uMish[1] = Cd * U10 * u10 #UDRAG

    Btransform!(col)
    Atransform!(col)
    VDIFF .= Ixtransform(col)

    expdot[colstart:colend,4] .= @. ADV + PGF + COR + VDIFF + HDIFF

    # vb tendency
    ADV .= @. (-vb * vbl / r) + (-ub * vbr) + (-wb * vbz)#VBADV
    PGF .= @. (-g * (hl / r)) #VBPGF
    COR .= @. (-ub * (f + (vb / r))) #VBCOR

    # Horizontal diffusion
    HDIFF .= @. Kh * ((vbr / r) + vbrr - (vb / (r * r)) + (vbll / (r * r)) + (2.0 * ubl / (r * r))) #VHDIFF
    # The following is just the Laplacian term without the curvature terms from Batchelor (1967) and Shapiro (1983)
    #HDIFF .= @. K * ((vr / r) + vrr + (vll / (r * r))) #VKDIFF

    # Differentiate Kv * dv/dz
    col.uMish .= Kv .* vbz

    # Drag only applies at z = 0
    col.uMish[1] = Cd * U10 * v10 #VDRAG

    Btransform!(col)
    Atransform!(col)
    VDIFF .= Ixtransform(col)
    
    expdot[colstart:colend,5] .= @. ADV + PGF + COR + VDIFF + HDIFF

    # Advance the explicit terms
    explicit_timestep(mtile, colstart, colend, t)

end

"""
    Oneway_ShallowWater_Slab_Uniform_Flow(mtile, colstart, colend, t)

One-way coupled shallow water model with a slab boundary layer in r-lambda coordinates,
including a uniform environmental flow specified by storm motion parameters (Um, Vm).
"""
function Oneway_ShallowWater_Slab_Uniform_Flow(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64)

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
    Um = model.physical_params[:Um]
    Vm = model.physical_params[:Vm]

    # Assign local variables with views
    r = view(gridpoints,:,1)
    lambda = view(gridpoints,:,2)

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

    # Environmental wind speed based on storm motion
    u_env = @. (Um * cos(lambda)) + (Vm * sin(lambda))
    v_env = @. (Vm * cos(lambda)) - (Um * sin(lambda))

    # Parameterized 10-m surface wind speed
    sfc_factor = 0.78
    U10 = similar(ub)
    u10 = ub #+ u_env
    v10 = vb #+ v_env
    @turbo U10 .= @. sfc_factor *  sqrt(u10^2 + v10^2)

    # Parameterized surface wind speed
    #sfc_factor = 0.78
    #U = similar(ub)
    #@turbo U .= @. sfc_factor * sqrt((ub * ub) + (vb * vb))

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
    @turbo COR .= @. (vg * (f + (vg / r))) - v_env * f #UCOR
    @turbo expdot[:,2] .= @. ADV + PGF + COR

    # vg tendency
    @turbo ADV .= @. (-vg * vgl / r) + (-ug * vgr) #VGADV
    @turbo PGF .= @. (-g * (hl / r)) #VGPGF
    @turbo COR .= @. (-ug * (f + (vg / r))) + u_env * f #VCOR
    @turbo expdot[:,3] .= @. ADV + PGF + COR

    # ub tendency
    @turbo ADV .= @. (-vb * ubl / r) + (-ub * ubr) #UBADV
    @turbo PGF .= @. (-g * hr) #UBPGF
    @turbo COR .= @. (vb * (f + (vb / r))) - v_env * f #UBCOR
    @turbo DRAG .= @. -(Cd * U10 * ub / Hb) #UDRAG
    @turbo W_ .= @. w_ * (ug - ub) / Hb #UW
    @turbo KDIFF .= @. K * ((ubr / r) + ubrr - (ub / (r * r)) + (ubll / (r * r)) - (2.0 * vbl / (r * r))) #UKDIFF
    # The following is just the Laplacian term without the curvature terms from Batchelor (1967) and Shapiro (1983)
    #@turbo KDIFF .= @. K * ((ur / r) + urr + (ull / (r * r))) #UKDIFF
    @turbo expdot[:,4] .= @. ADV + PGF + COR + DRAG + W_ + KDIFF

    # vb tendency
    @turbo ADV .= @. (-vb * vbl / r) + (-ub * vbr) #VBADV
    @turbo PGF .= @. (-g * (hl / r)) #VBPGF
    @turbo COR .= @. (-ub * (f + (vb / r))) + u_env * f #VBCOR
    @turbo DRAG .= @. -(Cd * U10 * vb / Hb) #VDRAG
    @turbo W_ .= @. w_ * (vg - vb) / Hb #VW
    @turbo KDIFF .= @. K * ((vbr / r) + vbrr - (vb / (r * r)) + (vbll / (r * r)) + (2.0 * ubl / (r * r))) #VKDIFF
    # The following is just the Laplacian term without the curvature terms from Batchelor (1967) and Shapiro (1983)
    #@turbo KDIFF .= @. K * ((vr / r) + vrr + (vll / (r * r))) #VKDIFF
    @turbo expdot[:,5] .= @. ADV + PGF + COR + DRAG + W_ + KDIFF

    # Advance the explicit terms
    explicit_timestep(mtile, colstart, colend, t)

end

