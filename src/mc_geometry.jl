# Geometry traits and geometry-specific terms for the total-energy moist
# compressible equation set (`mc_driver!` in moist_compressible.jl).
#
# The equation set is one geometry-generic kernel: everything thermodynamic,
# microphysical and vertical (retrieval, condensation, sedimentation, the
# semi-implicit acoustic solve, the implicit vertical diffusion) is column-wise
# and identical across geometries. What varies is the horizontal structure —
# which derivative slots hold what, the metric terms on the divergence and
# Laplacians, the curvature/Coriolis terms, and the extra tangential-wind
# prognostic — and all of that lives here, dispatched on a singleton
# `MCGeometry` trait so the geometry choice is compile-time and the Cartesian
# path lowers to exactly the pre-refactor code (bit-for-bit).
#
# Bitwise rule for this file: the `MCCartesianXZ` method bodies are the
# original statements of moist_compressible_XZ VERBATIM — same expression, same
# `@turbo` vs plain `@.`, whole statements only (broadcast fusion boundaries
# are statements, so moving a full statement into an @inline method with the
# same argument types preserves the generated code). Geometry-only terms exist
# solely in the cylindrical methods; there is no multiply-by-zero metric trick
# (an unconditional `+ 0.0` term could flip -0.0 tendencies — see the sponge
# comment in moist_compressible.jl).
#
# Springsteel cylindrical transforms return the RAW angular derivatives
# (slot 4 = ∂f/∂λ, slot 5 = ∂²f/∂λ²): every 1/r metric factor is applied here,
# written `f_l / r` and `f_ll / (r * r)` exactly like the shallow-water sets.
# `/r` is unguarded on purpose: the radial B-spline mish (Gaussian quadrature)
# points never sit at r = 0, and the u/v Dirichlet axis BC is enforced at the
# domain edge, not at a mish point.

abstract type MCGeometry end

"2D Cartesian x–z slice (RiRk/RZ geometry), 8 prognostic vars — the original XZ set."
struct MCCartesianXZ <: MCGeometry end

"2D axisymmetric r–z cylinder on the same RiRk/RZ grid (x reinterpreted as r), 9 vars."
struct MCAxisymRZ <: MCGeometry end

"3D r–λ–z cylinder on the RLR grid (spline-r, Fourier-λ, spline-z), 9 vars."
struct MCCylindricalRLR <: MCGeometry end

const MCCylinder = Union{MCAxisymRZ, MCCylindricalRLR}

# Canonical slot order for the cylindrical variants. v is APPENDED (index 9), not
# inserted next to u/w: semiimplicit_adjustment_p and diffusion_timestep_mc index
# variables by name, but the RHS kernel's view slots, expdot/impdot indices and
# scratch_column keys are hardcoded literals 1–8 — appending keeps every one valid
# and the XZ layout untouched.
const MC_VARS_CYL = ["p", "rho_d", "rho_t", "u", "w", "E_t", "Q_ss", "rho_r", "v"]

@inline has_v(::MCCartesianXZ) = false
@inline has_v(::MCCylinder) = true
@inline has_lambda(::MCCylindricalRLR) = true
@inline has_lambda(::MCGeometry) = false

# Where the vertical derivative slots and the z gridpoint column live: 2D grids
# put ∂z/∂zz in slots 4/5 (z is gridpoint column 2); the 3D RLR grid puts the
# raw azimuthal ∂λ/∂λλ in 4/5 and ∂z/∂zz in 6/7 (z is gridpoint column 3).
@inline zslot(::Union{MCCartesianXZ, MCAxisymRZ}) = 4
@inline zslot(::MCCylindricalRLR) = 6
@inline zcoord(::Union{MCCartesianXZ, MCAxisymRZ}) = 2
@inline zcoord(::MCCylindricalRLR) = 3

"""
    mc_slot_views(grid, colstart, colend, var, geom) -> (f, f_x, f_xx, f_z, f_zz, f_l, f_ll)

The seven derivative views of one prognostic slot, geometry-mapped: `f_x`/`f_xx` are
the first horizontal pair (∂x on XZ, ∂r on the cylinders), `f_z`/`f_zz` sit at
`zslot(geom)`, and `f_l`/`f_ll` are the raw azimuthal pair (slots 4/5) on the 3D
grid, `nothing` on the 2D grids. The NamedTuple is concrete per trait, so the
kernel specializes cleanly.
"""
@inline function mc_slot_views(grid, colstart::Int64, colend::Int64, var::Int, geom::MCGeometry)
    zs = zslot(geom)
    return (f    = view(grid.physical, colstart:colend, var, 1),
            f_x  = view(grid.physical, colstart:colend, var, 2),
            f_xx = view(grid.physical, colstart:colend, var, 3),
            f_z  = view(grid.physical, colstart:colend, var, zs),
            f_zz = view(grid.physical, colstart:colend, var, zs + 1),
            f_l  = mc_lambda_view(geom, grid, colstart, colend, var, 4),
            f_ll = mc_lambda_view(geom, grid, colstart, colend, var, 5))
end

@inline mc_lambda_view(::MCGeometry, grid, colstart, colend, var, slot) = nothing
@inline mc_lambda_view(::MCCylindricalRLR, grid, colstart, colend, var, slot) =
    view(grid.physical, colstart:colend, var, slot)

"""
    mc_v_views(geom, grid, colstart, colend)

Derivative views of the tangential wind v (var 9, reference vbar ≡ 0 so the totals
are the perturbation views), or `nothing` on the Cartesian slice.
"""
@inline mc_v_views(::MCCartesianXZ, grid, colstart, colend) = nothing
@inline mc_v_views(geom::MCCylinder, grid, colstart, colend) =
    mc_slot_views(grid, colstart, colend, 9, geom)

"Radius view (gridpoint column 1: x reinterpreted as r on the 2D grid, true r on RLR)."
@inline mc_radius(::MCCartesianXZ, gridpoints, colstart, colend) = nothing
@inline mc_radius(::MCCylinder, gridpoints, colstart, colend) =
    view(gridpoints, colstart:colend, 1)

# ── Geometry-specific terms of mc_driver! ─────────────────────────────────────
# Argument convention: scratch destination first, then geom, then the operands.
# `uv`/`wv`/`vv` are the slot-view NamedTuples of u, w, v (vv === nothing on XZ).

"Kinetic energy per unit mass."
@inline function mc_ke!(ke, ::MCCartesianXZ, u, w, vv)
    @. ke = 0.5 * ((u * u) + (w * w))
    return nothing
end
@inline function mc_ke!(ke, ::MCCylinder, u, w, vv)
    v = vv.f
    @. ke = 0.5 * ((u * u) + (v * v) + (w * w))
    return nothing
end

"Velocity divergence ∇·v (cylindrical: (1/r)∂(ru)/∂r + (1/r)∂v/∂λ + ∂w/∂z)."
@inline function mc_divergence!(div, ::MCCartesianXZ, u, u_x, w_z, vv, r)
    @. div = u_x + w_z
    return nothing
end
@inline function mc_divergence!(div, ::MCAxisymRZ, u, u_x, w_z, vv, r)
    @. div = u_x + (u / r) + w_z
    return nothing
end
@inline function mc_divergence!(div, ::MCCylindricalRLR, u, u_x, w_z, vv, r)
    v_l = vv.f_l
    @. div = u_x + (u / r) + (v_l / r) + w_z
    return nothing
end

"Scalar advection -v·∇f (the 3D cylinder adds the azimuthal -(v/r)∂f/∂λ)."
@inline function mc_advect!(ADV, ::Union{MCCartesianXZ, MCAxisymRZ}, u, w, vv, r, f_x, f_z, f_l)
    @turbo ADV .= @. (-u * f_x) + (-w * f_z)
    return nothing
end
@inline function mc_advect!(ADV, ::MCCylindricalRLR, u, w, vv, r, f_x, f_z, f_l)
    v = vv.f
    @turbo ADV .= @. (-u * f_x) + (-(v * f_l) / r) + (-w * f_z)
    return nothing
end

"""
Horizontal Laplacian of the dry-exact entropy s_d = C_vd·ln p − C_pd·ln ρ_d by chain
rule on the p/ρ_d derivative slots (see dry_entropy_pd). The cylindrical scalar
Laplacian adds (1/r)∂s/∂r (and (1/r²)∂²s/∂λ² on the 3D grid), each by the same
chain rule.
"""
@inline function mc_sd_lap!(sd_xx, ::MCCartesianXZ, p, rho_d, pv, rdv, r)
    pp_x = pv.f_x; pp_xx = pv.f_xx
    rho_dp_x = rdv.f_x; rho_dp_xx = rdv.f_xx
    @. sd_xx = (Cvd * ((pp_xx / p) - (pp_x * pp_x / (p * p)))) -
               (Cpd * ((rho_dp_xx / rho_d) - (rho_dp_x * rho_dp_x / (rho_d * rho_d))))
    return nothing
end
@inline function mc_sd_lap!(sd_xx, ::MCAxisymRZ, p, rho_d, pv, rdv, r)
    pp_x = pv.f_x; pp_xx = pv.f_xx
    rho_dp_x = rdv.f_x; rho_dp_xx = rdv.f_xx
    @. sd_xx = (Cvd * ((pp_xx / p) - (pp_x * pp_x / (p * p)))) -
               (Cpd * ((rho_dp_xx / rho_d) - (rho_dp_x * rho_dp_x / (rho_d * rho_d)))) +
               (((Cvd * (pp_x / p)) - (Cpd * (rho_dp_x / rho_d))) / r)
    return nothing
end
@inline function mc_sd_lap!(sd_xx, ::MCCylindricalRLR, p, rho_d, pv, rdv, r)
    pp_x = pv.f_x; pp_xx = pv.f_xx; pp_l = pv.f_l; pp_ll = pv.f_ll
    rho_dp_x = rdv.f_x; rho_dp_xx = rdv.f_xx; rho_dp_l = rdv.f_l; rho_dp_ll = rdv.f_ll
    @. sd_xx = (Cvd * ((pp_xx / p) - (pp_x * pp_x / (p * p)))) -
               (Cpd * ((rho_dp_xx / rho_d) - (rho_dp_x * rho_dp_x / (rho_d * rho_d)))) +
               (((Cvd * (pp_x / p)) - (Cpd * (rho_dp_x / rho_d))) / r) +
               (((Cvd * ((pp_ll / p) - (pp_l * pp_l / (p * p)))) -
                 (Cpd * ((rho_dp_ll / rho_d) - (rho_dp_l * rho_dp_l / (rho_d * rho_d))))) /
                (r * r))
    return nothing
end

"""
Horizontal frictional resolved-KE tendency ρ_t·K_h·(u·Lu + v·Lv + w·Lw) added to
E_t (see the FRIC_KE comment in mc_driver!). The cylindrical operators are the SAME
vector-Laplacian components as mc_u_kdiff!/mc_v_kdiff!/mc_w_kdiff! — E_t must follow
the resolved KE with identical operators or the energy budget drifts.
"""
@inline function mc_fric_ke!(FRIC_KE, ::MCCartesianXZ, rho_t, Khdiff, uv, wv, vv, r)
    u = uv.f; u_xx = uv.f_xx
    w = wv.f; w_xx = wv.f_xx
    @. FRIC_KE = rho_t * Khdiff * ((u * u_xx) + (w * w_xx))
    return nothing
end
@inline function mc_fric_ke!(FRIC_KE, ::MCAxisymRZ, rho_t, Khdiff, uv, wv, vv, r)
    u = uv.f; u_x = uv.f_x; u_xx = uv.f_xx
    w = wv.f; w_x = wv.f_x; w_xx = wv.f_xx
    v = vv.f; v_x = vv.f_x; v_xx = vv.f_xx
    @. FRIC_KE = rho_t * Khdiff *
        ((u * ((u_x / r) + u_xx - (u / (r * r)))) +
         (v * ((v_x / r) + v_xx - (v / (r * r)))) +
         (w * ((w_x / r) + w_xx)))
    return nothing
end
@inline function mc_fric_ke!(FRIC_KE, ::MCCylindricalRLR, rho_t, Khdiff, uv, wv, vv, r)
    u = uv.f; u_x = uv.f_x; u_xx = uv.f_xx; u_l = uv.f_l; u_ll = uv.f_ll
    w = wv.f; w_x = wv.f_x; w_xx = wv.f_xx; w_ll = wv.f_ll
    v = vv.f; v_x = vv.f_x; v_xx = vv.f_xx; v_l = vv.f_l; v_ll = vv.f_ll
    @. FRIC_KE = rho_t * Khdiff *
        ((u * ((u_x / r) + u_xx - (u / (r * r)) + (u_ll / (r * r)) - ((2.0 * v_l) / (r * r)))) +
         (v * ((v_x / r) + v_xx - (v / (r * r)) + (v_ll / (r * r)) + ((2.0 * u_l) / (r * r)))) +
         (w * ((w_x / r) + w_xx + (w_ll / (r * r)))))
    return nothing
end

"u-momentum (radial) forcing: PGF plus the cylindrical Coriolis + curvature (f + v/r)·v."
@inline function mc_u_forcing!(FORCING, ::MCCartesianXZ, pp_x, rho_t, vv, r, fcor)
    @turbo FORCING .= @. -pp_x / rho_t
    return nothing
end
@inline function mc_u_forcing!(FORCING, ::MCCylinder, pp_x, rho_t, vv, r, fcor)
    v = vv.f
    @turbo FORCING .= @. (-pp_x / rho_t) + ((fcor + (v / r)) * v)
    return nothing
end

"""
Horizontal momentum diffusion of u: bare ∂xx on the slice, the r-component of the
cylindrical vector Laplacian on the cylinders (Batchelor 1967 / Shapiro 1983
curvature corrections, as in the shallow-water sets).
"""
@inline function mc_u_kdiff!(KDIFF, ::MCCartesianXZ, Khdiff, uv, vv, r)
    u_xx = uv.f_xx
    @turbo KDIFF .= @. Khdiff * u_xx
    return nothing
end
@inline function mc_u_kdiff!(KDIFF, ::MCAxisymRZ, Khdiff, uv, vv, r)
    u = uv.f; u_x = uv.f_x; u_xx = uv.f_xx
    @turbo KDIFF .= @. Khdiff * ((u_x / r) + u_xx - (u / (r * r)))
    return nothing
end
@inline function mc_u_kdiff!(KDIFF, ::MCCylindricalRLR, Khdiff, uv, vv, r)
    u = uv.f; u_x = uv.f_x; u_xx = uv.f_xx; u_ll = uv.f_ll
    v_l = vv.f_l
    @turbo KDIFF .= @. Khdiff * ((u_x / r) + u_xx - (u / (r * r)) +
                                 (u_ll / (r * r)) - ((2.0 * v_l) / (r * r)))
    return nothing
end

"Horizontal momentum diffusion of w (scalar Laplacian — w has no curvature terms)."
@inline function mc_w_kdiff!(KDIFF, ::MCCartesianXZ, Khdiff, wv, r)
    w_xx = wv.f_xx
    @turbo KDIFF .= @. Khdiff * w_xx
    return nothing
end
@inline function mc_w_kdiff!(KDIFF, ::MCAxisymRZ, Khdiff, wv, r)
    w_x = wv.f_x; w_xx = wv.f_xx
    @turbo KDIFF .= @. Khdiff * ((w_x / r) + w_xx)
    return nothing
end
@inline function mc_w_kdiff!(KDIFF, ::MCCylindricalRLR, Khdiff, wv, r)
    w_x = wv.f_x; w_xx = wv.f_xx; w_ll = wv.f_ll
    @turbo KDIFF .= @. Khdiff * ((w_x / r) + w_xx + (w_ll / (r * r)))
    return nothing
end

"E_t flux/pressure-work forcing (the 3D cylinder adds the azimuthal -(v/r)∂p'/∂λ work)."
@inline function mc_et_work!(FORCING, ::Union{MCCartesianXZ, MCAxisymRZ},
                             E_t, p, div, u, pp_x, w, p_z, E_sed_z, pv, vv, r)
    @turbo FORCING .= @. (-(E_t + p) * div) - (u * pp_x) - (w * p_z) - E_sed_z
    return nothing
end
@inline function mc_et_work!(FORCING, ::MCCylindricalRLR,
                             E_t, p, div, u, pp_x, w, p_z, E_sed_z, pv, vv, r)
    v = vv.f; pp_l = pv.f_l
    @turbo FORCING .= @. (-(E_t + p) * div) - (u * pp_x) - ((v * pp_l) / r) -
                         (w * p_z) - E_sed_z
    return nothing
end

"""
Tangential (v) momentum tendency, slot 9 — explicit-only (no impdot; vertical
diffusion is handled by diffusion_timestep_mc like u/w). Advection, azimuthal PGF
(3D only; pbar has no λ-dependence), Coriolis + curvature -u(f + v/r), and the
λ-component of the cylindrical vector Laplacian. Reuses the S.ADV/FORCING/KDIFF
accumulators after slot 8, like every other slot.
"""
@inline mc_v_tendency!(expdot, ::MCCartesianXZ, colstart, colend, S,
                       u, w, uv, vv, pv, rho_t, r, fcor, Khdiff) = nothing
@inline function mc_v_tendency!(expdot, ::MCAxisymRZ, colstart, colend, S,
                                u, w, uv, vv, pv, rho_t, r, fcor, Khdiff)
    v = vv.f; v_x = vv.f_x; v_xx = vv.f_xx; v_z = vv.f_z
    ADV = S.ADV
    FORCING = S.FORCING
    KDIFF = S.KDIFF
    @turbo ADV .= @. (-u * v_x) + (-w * v_z)
    @turbo FORCING .= @. -u * (fcor + (v / r))
    @turbo KDIFF .= @. Khdiff * ((v_x / r) + v_xx - (v / (r * r)))
    @turbo expdot[colstart:colend, 9] .= @. ADV + FORCING + KDIFF
    return nothing
end
@inline function mc_v_tendency!(expdot, ::MCCylindricalRLR, colstart, colend, S,
                                u, w, uv, vv, pv, rho_t, r, fcor, Khdiff)
    v = vv.f; v_x = vv.f_x; v_xx = vv.f_xx; v_z = vv.f_z; v_l = vv.f_l; v_ll = vv.f_ll
    u_l = uv.f_l
    pp_l = pv.f_l
    ADV = S.ADV
    FORCING = S.FORCING
    KDIFF = S.KDIFF
    @turbo ADV .= @. (-u * v_x) + (-(v * v_l) / r) + (-w * v_z)
    @turbo FORCING .= @. (-(pp_l / r) / rho_t) + (-u * (fcor + (v / r)))
    @turbo KDIFF .= @. Khdiff * ((v_x / r) + v_xx - (v / (r * r)) +
                                 (v_ll / (r * r)) + ((2.0 * u_l) / (r * r)))
    @turbo expdot[colstart:colend, 9] .= @. ADV + FORCING + KDIFF
    return nothing
end

"""
Rayleigh sponge (momentum-only, see the comment at the call site in mc_driver!):
u, w (and v on the cylinders) relax toward the resting base with coefficient ray(z),
and E_t follows the resolved KE down exactly (dE = 2·ρ_t·ray·ke, with ke already
including v on the cylinders so the invariant holds unchanged).
"""
@inline function mc_sponge!(expdot, ::MCCartesianXZ, colstart, alpha, z_damp, z,
                            u, w, vv, rho_t, ke)
    if alpha > 0.0
        z_top = z[end]
        @inbounds for i in eachindex(z)
            ray = Rayleigh_damping(alpha, z[i], z_damp, z_top)
            j = colstart + i - 1
            expdot[j,4] += ray * u[i]
            expdot[j,5] += ray * w[i]
            expdot[j,6] += 2.0 * rho_t[i] * ray * ke[i]
        end
    end
    return nothing
end
@inline function mc_sponge!(expdot, ::MCCylinder, colstart, alpha, z_damp, z,
                            u, w, vv, rho_t, ke)
    if alpha > 0.0
        v = vv.f
        z_top = z[end]
        @inbounds for i in eachindex(z)
            ray = Rayleigh_damping(alpha, z[i], z_damp, z_top)
            j = colstart + i - 1
            expdot[j,4] += ray * u[i]
            expdot[j,5] += ray * w[i]
            expdot[j,9] += ray * v[i]
            expdot[j,6] += 2.0 * rho_t[i] * ray * ke[i]
        end
    end
    return nothing
end

"Explicit vertical-diffusion staging of v (AI2* history channel), slot 9."
@inline mc_v_diffdot!(diffdot, ::MCCartesianXZ, colstart, colend, Kvdiff, vv) = nothing
@inline function mc_v_diffdot!(diffdot, ::MCCylinder, colstart, colend, Kvdiff, vv)
    v_zz = vv.f_zz
    @turbo diffdot[colstart:colend, 9] .= @. Kvdiff * v_zz
    return nothing
end

# ── Tangential-wind machinery for diffusion_timestep_mc ───────────────────────

"View of the tangential-wind slot of var_np1 (name-keyed), or nothing on the slice."
@inline mc_v_np1_view(::MCCartesianXZ, vnp1, colstart, colend, vars) = nothing
@inline mc_v_np1_view(::MCCylinder, vnp1, colstart, colend, vars) =
    view(vnp1, colstart:colend, vars["v"])

"Star-state copy of v (mutated by the solve below, so a copy — not a view), or nothing."
@inline mc_v_star!(::MCCartesianXZ, S, v_v) = nothing
@inline function mc_v_star!(::MCCylinder, S, v_v)
    v_star = S.df_v_star
    copyto!(v_star, v_v)
    return v_star
end

"Star-state kinetic energy (v included on the cylinders)."
@inline function mc_ke_star!(ke_star, ::MCCartesianXZ, u_star, w_star, v_star)
    @. ke_star = 0.5 * ((u_star * u_star) + (w_star * w_star))
    return nothing
end
@inline function mc_ke_star!(ke_star, ::MCCylinder, u_star, w_star, v_star)
    @. ke_star = 0.5 * ((u_star * u_star) + (v_star * v_star) + (w_star * w_star))
    return nothing
end

"""
Implicit vertical momentum diffusion of v (AM2/AI2* staging + Helmholtz solve on
the `mats.v` factorization), mirroring the u/w solves of diffusion_timestep_mc.
Returns the diffused column (S.df_v_np1), or nothing on the Cartesian slice.
"""
@inline mc_v_diffusion_solve!(::MCCartesianXZ, S, mtile, col, mats, ts, t,
                              colstart, colend, v_star) = nothing
@inline function mc_v_diffusion_solve!(::MCCylinder, S, mtile, col, mats, ts, t,
                                       colstart, colend, v_star)
    v_index = mtile.model.grid_params.vars["v"]
    vdot_n = view(mtile.diffdot_n, colstart:colend, v_index)
    vdot_nm1 = view(mtile.diffdot_nm1, colstart:colend, v_index)
    v_nstar = S.df_v_nstar
    if (t == 1)
        @. v_nstar = v_star + (ts * 0.5 * vdot_n)
    else
        @. v_nstar = v_star - (ts * vdot_n) + (ts * 0.75 * vdot_nm1)
    end
    vdot_nm1 .= vdot_n

    h_v = (t == 1) ? mats.v_first : mats.v
    v_np1 = S.df_v_np1
    _vertical_solve!(col, h_v, v_nstar, mtile)
    copyto!(v_np1, Itransform!(col))
    return v_np1
end

"Post-solve resolved-KE increment (v included on the cylinders)."
@inline function mc_dke!(dke, ::MCCartesianXZ, u_np1, w_np1, u_star, w_star, v_np1, v_star)
    @. dke = 0.5 * (((u_np1 * u_np1) + (w_np1 * w_np1)) -
                    ((u_star * u_star) + (w_star * w_star)))
    return nothing
end
@inline function mc_dke!(dke, ::MCCylinder, u_np1, w_np1, u_star, w_star, v_np1, v_star)
    @. dke = 0.5 * (((u_np1 * u_np1) + (v_np1 * v_np1) + (w_np1 * w_np1)) -
                    ((u_star * u_star) + (v_star * v_star) + (w_star * w_star)))
    return nothing
end

"Write the diffused v back to var_np1 (no-op on the slice)."
@inline mc_assign_v!(::MCCartesianXZ, v_v, v_np1) = nothing
@inline function mc_assign_v!(::MCCylinder, v_v, v_np1)
    v_v .= v_np1
    return nothing
end
