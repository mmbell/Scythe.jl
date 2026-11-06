# =============================================================================
# What does ONE cubic-B-spline fit->reconstruct round trip actually cost the
# water species?
#
# `reference/FINDINGS_NEGATIVE_WATER_ATTRIBUTION.md` attributed the negative
# rho_r/rho_c to Galerkin projection ringing. This diagnostic isolates a single
# projection from the time loop, offline, on a saved snapshot — no model run:
#
#   (a) P(saved) vs saved      — is the saved field already in the spline space?
#                                (P must be idempotent there; also validates that
#                                this script's basis matches the run's)
#   (b) P(max(f,0))            — what does P do to a STRICTLY NON-NEGATIVE input?
#                                This is the projection ringing, with nothing else
#                                mixed in. max(f,0) has a kink at the zero contour,
#                                so it is if anything harder to represent than the
#                                true pre-fit field — the number is an upper bound.
#
# Mass is reported in the model's own metric (Gauss-weight quadrature on the mish,
# the same one `domain_integral` uses), so projection conservation is checked too.
#
# Usage:
#   julia --project=. benchmarks/water_projection_diagnostic.jl [snapshot] [outdir]
# =============================================================================

using Scythe, Springsteel, CSV, DataFrames, Printf
using Springsteel.CubicBSpline

snap   = length(ARGS) >= 1 ? ARGS[1] : "900.0"
outdir = length(ARGS) >= 2 ? ARGS[2] :
         joinpath(@__DIR__, "output", "o01_rainfall_quick_mc_rirk")

df = CSV.read(joinpath(outdir, "$(snap)_physical.csv"), DataFrame)

ncols = length(unique(df.r))
kDim  = nrow(df) ÷ ncols
num_cells_i = ncols ÷ 3            # mubar = 3 in both directions
num_cells_k = kDim ÷ 3
@printf("snapshot %s: nrows=%d ncols=%d kDim=%d  num_cells_i=%d num_cells_k=%d\n",
        snap, nrow(df), ncols, kDim, num_cells_i, num_cells_k)

# Basis must match o01_rainfall.jl exactly, or (a) will not come back idempotent.
vars = Scythe.MC_VARS
scalar_bc = Dict(v => NeumannBC() for v in vars)
side_bc   = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))
topbot_bc = merge(scalar_bc, Dict("w" => DirichletBC(), "rho_r" => NaturalBC()))

grid_params = GridParameters(;
    geometry = "RiRk",
    iMin = 0.0, iMax = 150.0e3, num_cells_i = num_cells_i,
    kMin = 0.0, kMax = 25.0e3, num_cells_k = num_cells_k,
    l_q = Dict("default" => 2.0),
    BCL = side_bc, BCR = side_bc, BCB = topbot_bc, BCT = topbot_bc,
    vars = Dict(v => i for (i, v) in enumerate(vars)),
)

patch = createGrid(grid_params)
gp = patch.params
@printf("basis: b_iDim=%d (from %d i-points), b_kDim=%d (from %d k-points), mubar=%d\n",
        gp.b_iDim, ncols, gp.b_kDim, kDim, gp.mubar)

"""P = eval∘fit on the model's own basis, using the given variable slot's BC."""
function project(field_flat, slot)
    patch.physical .= 0.0
    patch.physical[:, slot, 1] .= field_flat
    spectralTransform!(patch)
    gridTransform!(patch)
    return copy(patch.physical[:, slot, 1])
end

# Gauss-weight quadrature on the mish, both directions (cf. `gauss_cell_weights`
# in benchmarks/common/diagnostics.jl; inlined here to keep this script standalone).
_, qw = CubicBSpline._quadrature_rule(gp.mubar, gp.quadrature)
Wv = repeat(qw .* ((gp.kMax - gp.kMin) / num_cells_k), outer = num_cells_k)
Wh = repeat(qw .* ((gp.iMax - gp.iMin) / num_cells_i), outer = num_cells_i)
function mass(f)
    F = reshape(f, kDim, ncols)
    return sum(Wh[c] * sum(Wv .* @view(F[:, c])) for c in 1:ncols)
end

report(tag, f) = @printf("  %-26s min=%+11.4e  max=%11.4e  min/max=%+7.4f  mass=%.10e\n",
                         tag, minimum(f), maximum(f), minimum(f) / maximum(f), mass(f))

for name in ("rho_r", "rho_c")
    slot = grid_params.vars[name]
    f = Float64.(df[!, name])
    println("\n=== $name (slot $slot) ===")
    report("saved field (post-fit)", f)

    Pf = project(f, slot)
    report("P(saved)", Pf)
    @printf("  %-26s %.4e (rel %.3e)\n", "|P(f) - f|_inf",
            maximum(abs, Pf .- f), maximum(abs, Pf .- f) / maximum(abs, f))

    fpos = max.(f, 0.0)
    report("max(f,0)   [input]", fpos)
    report("P(max(f,0))", project(fpos, slot))
end
