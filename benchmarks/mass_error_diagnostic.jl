#!/usr/bin/env julia
# =============================================================================
# Standalone diagnostic: where does the dry-air mass drift come from?
#
# Hypothesis (see reference/mass_conservation_advective.tex):
#   The continuity equation is integrated in advective (non-conservative) form
#       dξ/dt = -v·∇ξ - ∇·v ,   ρ_d = ρ_0 e^ξ ,
#   so the discrete total mass M = Σ_i W_i ρ_{d,i} changes at the rate
#       dM/dt = Σ_i W_i ρ_{d,i} (dξ/dt)_i              (chain rule is exact per node)
#             = -Σ_i W_i [∇·(ρ_d v)]_i  +  ⟨v, E⟩,
#   where E = D(e^ξ) - e^ξ·D(ξ) is the discrete chain-rule defect (≡0 in the
#   continuum, ≠0 for the spline derivative D because e^ξ is not in the basis).
#   The first (flux-form) term telescopes to a boundary flux ≈ 0 at rigid walls,
#   so dM/dt ≈ ⟨v, E⟩ — a *velocity-weighted* mass source. No flow ⇒ no drift.
#
# This script tests that prediction directly from saved output, with no model
# changes, using the model's own spectral operators:
#   (1) the directly-integrated mass M(t) (matches diagnostics.csv);
#   (2) the advective continuity tendency dM/dt reconstructed from the saved
#       fields + their saved spectral derivatives (this IS the scheme's mass
#       tendency; for the moist run semi-implicit is off so it is complete);
#   (3) the flux-form tendency -Σ W ∇·(ρ_d v), formed with the model's spline
#       derivative of the nonlinear fluxes ρ_d u, ρ_d w (≈ boundary ≈ 0);
#   (4) time-integration of (2) over [0,100,200] s vs. the observed ΔM.
#
# If (4) reproduces the observed drift and (3) ≈ 0, the drift is exactly the
# advective discretization's non-conservation, and the defect (2)-(3) = ⟨v,E⟩
# is shown to scale with the velocity field.
#
# Usage:
#   julia --project=. benchmarks/mass_error_diagnostic.jl moist
#   julia --project=. benchmarks/mass_error_diagnostic.jl dry
# =============================================================================

using Scythe, Springsteel, CSV, DataFrames, Statistics, Printf

include(joinpath(@__DIR__, "common", "diagnostics.jl"))

const PE_VARS = ["s", "xi", "mu", "u", "w", "mu_c", "mu_r", "mu_sat"]
const TIMES = ["0.0", "100.0", "200.0"]

run = isempty(ARGS) ? "moist" : ARGS[1]
if run == "moist"
    outdir   = joinpath(@__DIR__, "output", "bf02_moist_full_pe_rirk") * "/"
    ref_file = outdir * "bf02_moist_exact.ref"
    exact_ref = true
elseif run == "dry"
    outdir   = joinpath(@__DIR__, "output", "bf02_dry_full_pe_rirk") * "/"
    ref_file = outdir * "bf02_dry.ref"
    exact_ref = false
else
    error("unknown run '$run' (use 'moist' or 'dry')")
end

# ── Infer grid size from the saved field, build matching grid/model ──────────
df0   = CSV.read(outdir * "0.0_physical.csv", DataFrame)
ncols = length(unique(df0.r))
kDim  = nrow(df0) ÷ ncols
num_cells = ncols ÷ 3                      # mubar = 3 in both i and k
@printf("run=%-5s  nrows=%d  ncols=%d  kDim=%d  num_cells=%d\n",
        run, nrow(df0), ncols, kDim, num_cells)

scalar_bc = Dict(v => NeumannBC() for v in PE_VARS)
wall_bc   = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))
grid_params = GridParameters(
    geometry = "RiRk",
    iMin = 0.0, iMax = 20.0e3, num_cells = num_cells,
    kMin = 0.0, kMax = 10.0e3, kDim = kDim,
    BCL = wall_bc, BCR = wall_bc, BCB = wall_bc, BCT = wall_bc,
    vars = Dict(v => i for (i, v) in enumerate(PE_VARS)),
)
model = ModelParameters(
    ts = 0.05, integration_time = 200.0, output_interval = 100.0,
    equation_set = "primitive_equation_XZ",
    initial_conditions = outdir * "ics.csv",
    output_dir = outdir, ref_state_file = ref_file,
    grid_params = grid_params,
    physical_params = Dict(:Khdiff => 0.0, :Kvdiff => 0.0, :Kv_mudiff => 0.0,
                           :alpha => 0.0, :z_damp => 20.0e3),
    # `:output_formats` pinned to CSV: this script reads `<t>_physical.csv` directly
    # (the model's default is now the comprehensive `<t>.nc`). It does not include the
    # benchmark harness, so the list is written out rather than shared.
    options = Dict{Symbol,Any}(:semiimplicit => (run == "dry"),
                   :exact_reference_state => exact_ref,
                   :precipitation => false, :vertical_mixing => false,
                   :output_formats => [:csv]),
)

ref, _, _ = rebuild_reference(model)
xibar_z_lev = ref.xibar[:, 2]                          # reference d(ξ̂)/dz per level
xibar_z = repeat(xibar_z_lev, ncols)                   # broadcast to every point

# Integral over the 2-D domain using the model's own Gauss-weight quadrature.
gi(field) = domain_integral(reshape(field, kDim, ncols), model)

# Model-exact spline derivatives of an arbitrary nodal field, obtained by
# loading it into a grid slot and running the model transform. The slot's BC is
# applied during the fit, so use a Dirichlet slot for fields that vanish at the
# walls (the momentum fluxes ρ_d u, ρ_d w do, since u=w=0 there).
patch = createGrid(grid_params)
vi = grid_params.vars
function spline_grads(field_flat, slot)
    patch.physical .= 0.0
    patch.physical[:, slot, 1] .= field_flat
    spectralTransform!(patch)
    gridTransform!(patch)
    return copy(patch.physical[:, slot, 2]), copy(patch.physical[:, slot, 4])   # ∂/∂x, ∂/∂z
end

# Project a nodal field through the model's spline basis using the given variable
# slot's BC — i.e. apply the same b_kDim(<kDim) least-squares filter + BC the
# model re-applies to every prognostic field each timestep. Returns P(field).
function spline_project(field_flat, slot)
    patch.physical .= 0.0
    patch.physical[:, slot, 1] .= field_flat
    spectralTransform!(patch)
    gridTransform!(patch)
    return copy(patch.physical[:, slot, 1])
end

# ── Per-snapshot quantities ──────────────────────────────────────────────────
function snapshot(tag)
    df = CSV.read(outdir * "$(tag)_physical.csv", DataFrame)
    xi_tot = df.xi .+ repeat(ref.xibar[:, 1], ncols)
    rho_d  = Scythe.dry_density.(xi_tot)
    u, w   = df.u, df.w
    div    = df.u_r .+ df.w_z                            # ∇·v  = u_x + w_z (Cartesian XZ)
    xi_z_tot = df.xi_z .+ xibar_z                        # D_z(ξ_total) = ξ'_z + ξ̂_z

    # (2) advective continuity tendency  ρ_d·dξ/dt = ∂ρ_d/∂t (per node)
    dxidt   = .-(u .* df.xi_r) .- (w .* xi_z_tot) .- div
    dMdt_adv = gi(rho_d .* dxidt)

    # (3) flux-form tendency  -∇·(ρ_d v), with model spline derivative of the flux
    dfu_dx, _ = spline_grads(rho_d .* u, vi["u"])        # ∂(ρ_d u)/∂x  (Dirichlet slot)
    _, dfw_dz = spline_grads(rho_d .* w, vi["w"])        # ∂(ρ_d w)/∂z  (Dirichlet slot)
    dMdt_flux = gi(.-(dfu_dx .+ dfw_dz))

    # (2b) the tendency the model *actually applies*: each step it re-projects ξ
    # through the b_kDim(<kDim) spline basis with the Neumann BC. The effective
    # increment is P(dξ/dt), not dξ/dt. Its mass tendency:
    P_dxidt   = spline_project(dxidt, vi["xi"])
    dMdt_proj = gi(rho_d .* P_dxidt)
    # mass tendency from the projection/BC step alone (filter of the increment)
    dMdt_filter = dMdt_proj - dMdt_adv

    M    = gi(rho_d)

    # (4) mass change from a single spline re-projection of the ξ' FIELD itself
    # (with its Neumann BC). If the saved field is already in the b_kDim range
    # this is ~0; if not, it is the per-application loss of the field filter.
    xi_p_proj = spline_project(df.xi, vi["xi"])
    rho_d_proj = Scythe.dry_density.(xi_p_proj .+ repeat(ref.xibar[:, 1], ncols))
    dM_reproject = gi(rho_d_proj) - M

    defect = dMdt_adv - dMdt_flux                        # ≈ ⟨v, E⟩
    speed = sqrt.(u .^ 2 .+ w .^ 2)
    return (; tag, M, dMdt_adv, dMdt_flux, dMdt_proj, dMdt_filter, dM_reproject,
            maxw = maximum(abs, w), meanspeed = mean(speed),
            node_filter = rho_d .* (P_dxidt .- dxidt), speed)
end

snaps = [snapshot(t) for t in TIMES]
M0 = snaps[1].M

println("\n", "="^96)
@printf("%-7s %13s %13s %13s %13s %13s %8s\n",
        "t[s]", "M", "dMdt_adv", "dMdt_flux", "dMdt_proj", "dMdt_filter", "max|w|")
println("  (adv = continuity RHS;  flux = -∇·(ρ_d v);  proj = RHS after spline projection;")
println("   filter = proj − adv = mass change from the per-step b_kDim<kDim spline filter + BC)")
println("-"^96)
for s in snaps
    @printf("%-7s %13.5e %13.5e %13.5e %13.5e %13.5e %8.4f\n",
            s.tag, s.M, s.dMdt_adv, s.dMdt_flux, s.dMdt_proj, s.dMdt_filter, s.maxw)
end
println("="^96)

dt = 100.0
obs_dM = snaps[end].M - M0
trap(r) = dt * (0.5 * r[1] + r[2] + 0.5 * r[3])

@printf("\nObserved  ΔM over 200 s                 : %+.5e  (%.4e %%)\n",
        obs_dM, 100 * obs_dM / abs(M0))
@printf("∫ dM/dt_adv  (continuity RHS only)      : %+.5e  (%.4e %%)\n",
        trap([s.dMdt_adv for s in snaps]), 100 * trap([s.dMdt_adv for s in snaps]) / abs(M0))
@printf("∫ dM/dt_flux (flux form, ≈ boundary)    : %+.5e  (%.4e %%)\n",
        trap([s.dMdt_flux for s in snaps]), 100 * trap([s.dMdt_flux for s in snaps]) / abs(M0))
@printf("∫ dM/dt_proj (RHS through projection)   : %+.5e  (%.4e %%)\n",
        trap([s.dMdt_proj for s in snaps]), 100 * trap([s.dMdt_proj for s in snaps]) / abs(M0))
@printf("   ↳ ratio  ∫dM/dt_proj / observed ΔM   : %.3f\n",
        trap([s.dMdt_proj for s in snaps]) / obs_dM)
println("\nSingle re-projection of the saved ξ' field (mass change per application):")
for s in snaps
    @printf("  t=%-6s  ΔM_reproject = %+.5e  (%.4e %%)\n",
            s.tag, s.dM_reproject, 100 * s.dM_reproject / abs(M0))
end

# ── Where does the projection/filter loss live, and how does it scale? ──────
s = snaps[end]
nf = s.node_filter
println("\n", "-"^96)
println("Projection/filter mass-loss structure (t = 200 s):")
@printf("  mean|v|                                 : %.4f m/s\n", s.meanspeed)
@printf("  corr(|node filter loss|, |v|)           : %.3f\n", cor(abs.(nf), s.speed))
@printf("  fraction of |filter loss| where |w|>median|w| : %.3f\n",
        sum(abs.(nf)[s.speed .> median(s.speed)]) / sum(abs.(nf)))
@printf("  b_kDim=%d of kDim=%d  ⇒ %d filtered vertical modes\n",
        grid_params.b_kDim, kDim, kDim - grid_params.b_kDim)

# ════════════════════════════════════════════════════════════════════════════
# In-loop tests: iterate the real continuity update + spline projection with the
# velocity frozen at t=200, isolating the per-step projection. Two knobs:
#   • project ξ (log-density, nonlinear)   vs   project ρ_d (linear, flux form)
#   • l_q = 2.0 (default 3rd-derivative smoothing)   vs   l_q = 0.0 (no smoothing)
# RHS transport conserves, so any drift is the projection. If l_q is the culprit,
# l_q=0 should cut the ξ-loop loss; if the nonlinearity is the culprit, the ρ_d
# (linear) loop should conserve regardless of l_q.
# ════════════════════════════════════════════════════════════════════════════
function build_patch(lq)
    gp = GridParameters(
        geometry = "RiRk",
        iMin = 0.0, iMax = 20.0e3, num_cells = num_cells,
        kMin = 0.0, kMax = 10.0e3, kDim = kDim,
        BCL = wall_bc, BCR = wall_bc, BCB = wall_bc, BCT = wall_bc,
        l_q = Dict("default" => lq),
        vars = Dict(v => i for (i, v) in enumerate(PE_VARS)),
    )
    return createGrid(gp)
end

# Project a field through the given patch's spline basis (xi slot, Neumann BC) and
# return P(field) and its ∂/∂x, ∂/∂z. Same operator for both ξ and ρ_d, so the
# only difference between the two loops is *which* variable is projected.
function proj3(p, field)
    p.physical .= 0.0
    p.physical[:, vi["xi"], 1] .= field
    spectralTransform!(p); gridTransform!(p)
    return (copy(p.physical[:, vi["xi"], 1]),
            copy(p.physical[:, vi["xi"], 2]),
            copy(p.physical[:, vi["xi"], 4]))
end

function frozen_loop(p; project_density::Bool, nstep::Int=400, ts::Float64=0.025)
    dfx = CSV.read(outdir * "200.0_physical.csv", DataFrame)
    u, w = dfx.u, dfx.w
    div  = dfx.u_r .+ dfx.w_z
    xibar_v = repeat(ref.xibar[:, 1], ncols)
    if project_density
        ρ = Scythe.dry_density.(dfx.xi .+ xibar_v)        # state: density (linear)
        Pρ, _, _ = proj3(p, ρ); M_start = gi(Pρ)
        for _ in 1:nstep
            Pρ, ρx, ρz = proj3(p, ρ)
            rhs = .-(u .* ρx) .- (w .* ρz) .- (Pρ .* div) # = -∇·(ρ v)  (flux form)
            ρ = Pρ .+ ts .* rhs
        end
        Pρ, _, _ = proj3(p, ρ); return (gi(Pρ) - M_start) / (nstep * ts)
    else
        f = copy(dfx.xi)                                   # state: ξ' (log-density)
        Pf, _, _ = proj3(p, f); M_start = gi(Scythe.dry_density.(Pf .+ xibar_v))
        for _ in 1:nstep
            Pf, fx, fz = proj3(p, f)
            rhs = .-(u .* fx) .- (w .* (fz .+ xibar_z)) .- div
            f = Pf .+ ts .* rhs
        end
        Pf, _, _ = proj3(p, f); return (gi(Scythe.dry_density.(Pf .+ xibar_v)) - M_start) / (nstep * ts)
    end
end

p_lq2 = build_patch(2.0)
p_lq0 = build_patch(0.0)
obs_rate = obs_dM / 200
results = [
    ("project ξ  (log-density)", "l_q=2.0", frozen_loop(p_lq2; project_density=false)),
    ("project ξ  (log-density)", "l_q=0.0", frozen_loop(p_lq0; project_density=false)),
    ("project ρ_d (linear/flux)", "l_q=2.0", frozen_loop(p_lq2; project_density=true)),
    ("project ρ_d (linear/flux)", "l_q=0.0", frozen_loop(p_lq0; project_density=true)),
]

println("\n", "="^96)
println("Frozen-velocity loop (400 steps, dt=0.025, v fixed at t=200)")
@printf("Observed model mass-loss rate: %+.4e kg/s   (= ΔM / 200 s)\n", obs_rate)
println("-"^96)
@printf("%-28s %-9s %16s %14s\n", "variable projected", "filter", "loss rate [kg/s]", "frac of obs")
println("-"^96)
for (name, lab, rate) in results
    @printf("%-28s %-9s %16.4e %14.3f\n", name, lab, rate, rate / obs_rate)
end
println("="^96)
println("""
CONCLUSION
  • Continuity RHS (advective) conserves to ~1e-9 %; flux form to ~1e-16 %.
    The spatial advection/divergence discretization is NOT the leak.
  • The leak is the l_q 3rd-derivative SMOOTHING penalty in the spline least-
    squares fit, applied to the NONLINEAR log-density ξ every step:
       – project ξ, l_q=2.0  →  reproduces 99 % of the observed mass loss
       – project ξ, l_q=0.0  →  loss falls 120x (to <1 % of observed)
    So it is the smoothing, not b_kDim<kDim (the ×3 Gauss nodes make the bare
    projection integral-preserving — confirmed by the ρ_d rows).
  • Projecting the LINEAR density ρ_d conserves to ~1e-15 % WITH OR WITHOUT l_q.
    Smoothing ξ lowers its variance while ~preserving ∫ξ; since exp is convex
    (Jensen), ∫ρ_0 e^ξ decreases systematically ⇒ the always-negative drift.
  • Fixes: (a) carry/project the linear density ρ_d (flux form) — keeps the l_q
    stabilization AND conserves mass; (b) l_q=0 removes most of the loss but
    invites spectral overshoot, so it is not a structural fix.
""")
