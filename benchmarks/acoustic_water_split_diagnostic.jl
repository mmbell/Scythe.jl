# =============================================================================
# Does the ACOUSTIC SOLVE drive rho_t and rho_d apart?
#
# `semiimplicit_adjustment_p` slaves the two densities to the same solved
# acoustic variable phi = rho_tbar*w, but through DIFFERENT discrete operators:
#
#   rho_t' -= dtau * D(phi)                          (c == 1 exactly)
#   rho_d' -= dtau * (c_d * D(phi) + c_d_z * phi)     (pointwise product rule)
#
# so the implicit increment to the total water rho_w = rho_t - rho_d is
#
#   d(rho_w) = -dtau * [ (1 - c_d) * D(phi) - c_d_z * phi ]
#            = -dtau * [ q_w * D(phi) + D(q_w) * phi ],   q_w = 1 - c_d = rho_w/rho_t
#
# In the CONTINUUM that is exactly -dtau * d/dz (q_w * phi): the acoustic mode's
# transport of the water, and it is proportional to q_w (~1e-5 aloft). In the
# DISCRETE spline representation the product rule does not hold -- D(fg) requires
# fitting the product, which is not in the space when f and g are -- so the two
# forms differ, and the difference is NOT proportional to q_w.
#
# This computes both, offline, on a saved snapshot with the model's own basis:
#
#   current  = q_w * D(phi) + D(q_w) * phi     (what the code applies)
#   fluxform = D(q_w * phi)                    (one fit of the flux, then differentiate)
#
# D is linear, so the flux form makes the rho_w increment EXACTLY D(q_w*phi) --
# the water fraction multiplies BEFORE the derivative and the increment inherits
# its smallness. The pointwise form does not.
#
# Usage:
#   julia --project=. benchmarks/acoustic_water_split_diagnostic.jl [outdir] [snapshot]
# =============================================================================

using Scythe, Springsteel, CSV, DataFrames, Printf
using Springsteel.CubicBSpline
import Springsteel.CubicBSpline: Btransform!, Atransform!, Ixtransform

outdir = length(ARGS) >= 1 ? ARGS[1] :
         joinpath(@__DIR__, "output", "o01_rainfall_quick_mc_rirk")
snap = length(ARGS) >= 2 ? ARGS[2] : "3600.0"

df = CSV.read(joinpath(outdir, "$(snap)_physical.csv"), DataFrame)
ncols = length(unique(df.r))
kDim = nrow(df) ÷ ncols

vars = Scythe.MC_VARS
scalar_bc = Dict(v => NeumannBC() for v in vars)
side_bc   = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))
topbot_bc = merge(scalar_bc, Dict("w" => DirichletBC(), "rho_r" => NaturalBC()))
grid_params = GridParameters(;
    geometry = "RiRk",
    iMin = 0.0, iMax = 150.0e3, num_cells_i = ncols ÷ 3,
    kMin = 0.0, kMax = 25.0e3, num_cells_k = kDim ÷ 3,
    l_q = Dict("default" => 2.0),
    BCL = side_bc, BCR = side_bc, BCB = topbot_bc, BCT = topbot_bc,
    vars = Dict(v => i for (i, v) in enumerate(vars)),
)
patch = createGrid(grid_params)
z = Scythe.getGridpoints(patch)[1:kDim, 2]
column = Scythe.reference_column(patch, grid_params)
ref = Springsteel.exact_pressure_reference_state(joinpath(outdir, "o01_exact.ref"), z, column)

rho_dbar   = Springsteel.ref_rho_d(ref)[:, 1]
rho_dbar_z = Springsteel.ref_rho_d(ref)[:, 2]
rho_tbar   = Springsteel.ref_rho_t(ref)[:, 1]
rho_tbar_z = Springsteel.ref_rho_t(ref)[:, 2]

# The solve fits phi in W's column basis (Dirichlet top and bottom) -- slot 5.
wcol = deepcopy(patch.kbasis.data[grid_params.vars["w"]])
out = zeros(kDim)
"""D(f): fit f in w's column basis and evaluate the spline derivative, exactly as
`semiimplicit_adjustment_p` obtains `phi_z` from `phi`."""
function D!(dst, f)
    wcol.uMish .= f
    Btransform!(wcol)
    Atransform!(wcol)
    Ixtransform(wcol, dst)
    return dst
end

# Reference-profile coefficients (o01 runs with sd_si = false).
c_d   = rho_dbar ./ rho_tbar
c_d_z = ((rho_dbar_z .* rho_tbar) .- (rho_dbar .* rho_tbar_z)) ./ (rho_tbar .^ 2)
q_w   = 1.0 .- c_d                      # reference water mass fraction
q_w_z = -c_d_z

@printf("snapshot %s: %d columns x %d levels\n", snap, ncols, kDim)
@printf("reference water fraction q_w: min %.3e at z=%.2f km, max %.3e at z=%.2f km\n",
        minimum(q_w), z[argmin(q_w)] / 1e3, maximum(q_w), z[argmax(q_w)] / 1e3)

worst = Ref((0.0, 0, 0, 0.0, 0.0, 0.0))
sum_cur = Ref(0.0); sum_flux = Ref(0.0); npts = Ref(0)
phi = zeros(kDim); phi_z = zeros(kDim); flux = zeros(kDim); flux_z = zeros(kDim)
for c in 1:ncols
    rows = ((c - 1) * kDim + 1):(c * kDim)
    w = Float64.(df.w[rows])
    @. phi = rho_tbar * w                       # the non-sd_si acoustic variable
    D!(phi_z, phi)
    @. flux = q_w * phi
    D!(flux_z, flux)
    for k in 1:kDim
        cur = q_w[k] * phi_z[k] + q_w_z[k] * phi[k]     # what the code applies
        flx = flux_z[k]                                  # the flux form
        sum_cur[] += abs(cur); sum_flux[] += abs(flx); npts[] += 1
        d = abs(cur - flx)
        if d > worst[][1]
            worst[] = (d, c, k, cur, flx, phi[k])
        end
    end
end

@printf("\nmean |increment| per gridpoint (per unit dtau):\n")
@printf("  current  (pointwise product rule) : %.6e\n", sum_cur[] / npts[])
@printf("  fluxform (one fit of q_w*phi)     : %.6e\n", sum_flux[] / npts[])
@printf("  ratio current/fluxform            : %.3f\n", sum_cur[] / max(sum_flux[], eps()))

(d, c, k, cur, flx, phik) = worst[]
@printf("\nworst pointwise disagreement: r = %.2f km, z = %.3f km\n",
        df.r[(c - 1) * kDim + k] / 1e3, z[k] / 1e3)
@printf("  q_w there                = %.6e   (reference water fraction)\n", q_w[k])
@printf("  phi = rho_tbar*w         = %.6e\n", phik)
@printf("  current  increment/dtau  = %+.6e\n", cur)
@printf("  fluxform increment/dtau  = %+.6e\n", flx)
@printf("  difference               = %+.6e   (%.1fx the flux-form value)\n",
        cur - flx, abs(cur - flx) / max(abs(flx), eps()))
@printf("\nAt ts = 0.3 s that difference is %.3e kg/m^3 per step, %.3e over 12000 steps\n",
        0.3 * abs(cur - flx), 0.3 * abs(cur - flx) * 12000)
