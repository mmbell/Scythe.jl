# Shared helpers for the staged 1D nested-advection tests (one-way, two-way,
# subcycling). Single-process: each nest patch is its own grid + ModelTile
# (patch used as its own tile, no halo exchange), stepped by direct
# advance_column/calcTendency calls as in test_linear_advection_integration.jl.
#
# Coupling follows DeMaria et al. (1992): coarse→fine via the R3X border trio
# at the interface node (the coarse patch carries a one-cell collar past the
# interface so the trio is interior), fine→coarse via collar-point injection
# of the fine representation into the coarse physical array.

using SparseArrays

const NESTED_ADV_C0 = 1.0
const NESTED_ADV_K  = 0.0

"""One nest patch: grid, model tile, and bookkeeping for coupling."""
struct AdvectionNestPatch
    patch::Springsteel.SpringsteelGrid
    mtile::Scythe.ModelTile
    pts::Vector{Float64}          # mish points
end

function make_advection_patch(iMin, iMax, num_cells, bcl, bcr; ts, nsteps)
    gp = GridParameters(
        geometry = "R",
        num_cells = num_cells,
        iMin = iMin,
        iMax = iMax,
        BCL = Dict("u" => bcl),
        BCR = Dict("u" => bcr),
        vars = Dict("u" => 1),
    )
    model = ModelParameters(
        ts = ts,
        integration_time = nsteps * ts,
        output_interval = nsteps * ts,
        equation_set = "LinearAdvection1D",
        initial_conditions = "",
        grid_params = gp,
        physical_params = Dict(:c_0 => NESTED_ADV_C0, :K => NESTED_ADV_K),
    )
    patch = createGrid(gp)
    haloReceiveMap = sparse(Int64[], Int64[], Float64[],
                            size(patch.spectral, 1), size(patch.spectral, 2))
    mtile = createModelTile(patch, patch, model, haloReceiveMap)
    return AdvectionNestPatch(patch, mtile, vec(getGridpoints(patch)))
end

"""
Set the initial condition u(x) on a patch and compute its spectral (b)
state. Does NOT run gridTransform!: a patch with an R3X interface side must
receive its ahat (via update_interface!) before its first reconstruction,
otherwise the border is pinned to zero and the error is baked into the fit.
Callers run gridTransform! after any needed interface exchange.
"""
function set_advection_ic!(np::AdvectionNestPatch, f)
    for i in eachindex(np.pts)
        np.patch.physical[i, 1, 1] = f(np.pts[i])
    end
    spectralTransform!(np.patch)
    return np
end

"""Advance one patch one step: tendencies + AB3 update + forward transform."""
function step_patch!(np::AdvectionNestPatch, t::Int)
    Scythe.advance_column(np.mtile, -1, t)
    Scythe.calcTendency(np.mtile)
    return np
end

"""
Mish x-points of `parent` lying in its collar (past `interface_x`, toward the
child). `side` is the side of the PARENT the collar is on (:right = parent
extends rightward past the interface).
"""
function collar_points(parent::AdvectionNestPatch, interface_x, side::Symbol)
    if side == :right
        idx = findall(x -> x > interface_x + 1e-12, parent.pts)
    else
        idx = findall(x -> x < interface_x - 1e-12, parent.pts)
    end
    return idx, parent.pts[idx]
end

"""
Inject the child's representation into the parent's physical array at the
parent's collar mish points (fine→coarse feedback, DeMaria eq. 2.22).
Overwrites all derivative slices.
"""
function inject_collar!(parent::AdvectionNestPatch, child::AdvectionNestPatch,
                        collar_idx::Vector{Int}, collar_x::Vector{Float64})
    vals = evaluate_grid_ipoints(child.patch, collar_x)
    for s in 1:3
        for (n, i) in enumerate(collar_idx)
            parent.patch.physical[i, 1, s] = vals[n, 1, s]
        end
    end
    return parent
end

"""L2 relative error against a function evaluated at the patch points."""
function l2_error(np::AdvectionNestPatch, f)
    num = 0.0
    den = 0.0
    for i in eachindex(np.pts)
        a = f(np.pts[i])
        num += (np.patch.physical[i, 1, 1] - a)^2
        den += a^2
    end
    return sqrt(num) / max(sqrt(den), eps())
end

gaussian_pulse(x0, sigma) = x -> exp(-(x - x0)^2 / (2 * sigma^2))

"""
Gauss-quadrature integral of u over the patch, optionally restricted to
`[xmin, xmax]` (bounds must coincide with cell edges — used to exclude collar
cells so overlapping strips are counted once across a nest).
"""
function patch_integral(np::AdvectionNestPatch; xmin=-Inf, xmax=Inf)
    gp = np.mtile.model.grid_params
    mubar = gp.mubar
    DX = (gp.iMax - gp.iMin) / gp.num_cells
    _, qw = Springsteel.CubicBSpline._quadrature_rule(mubar, :gauss)
    total = 0.0
    for i in eachindex(np.pts)
        x = np.pts[i]
        (xmin - 1e-12 <= x <= xmax + 1e-12) || continue
        w = qw[(i - 1) % mubar + 1] * DX
        total += w * np.patch.physical[i, 1, 1]
    end
    return total
end
