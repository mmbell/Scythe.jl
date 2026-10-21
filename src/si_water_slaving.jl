# ─────────────────────────────────────────────────────────────────────────────
# ν-flux slaving of the water slots in the acoustic semi-implicit solve — Stage 2 of the ρ_t
# retirement (reference/RHO_T_RETIREMENT_DESIGN.md §4.2 options (i′) and (iii), §6 Stage 2).
#
# The vertical acoustic solve advances ρ_t' by the implicit flux divergence −Δτ ∂z φ and ρ_d' by
# −Δτ ∂z(c_d φ) (φ = ρ̂_t w), while every water species took its vertical compression −J ρ ∂z w
# EXPLICITLY, under AB3. The density the SI slaves and the species that should add up to it
# therefore advanced under two different integrators, and the difference was the first feeder of
# the vapor gap `res_rho_t − ρ_v` (design §2). Here every water-mass slot k ∈ 𝒲 is slaved to the
# same φ in flux form,
#
#     Δν_k' = −Δτ (c_k φ_z + c_{k,z} φ),
#
# its AI2* history `impdot[k] = −(c_k φ_z + c_{k,z} φ)` is staged from the SAME `imp_phi_z`/`phi_n`
# fit the dry legs use, and the explicit tendency adds back the SAME linear operator evaluated on
# the state at n (the product-rule remainder pattern of slots 2 and 3; `_siw_addback!`), so the
# equation is unchanged and only the integrator of its vertical acoustic part moves. Under `options[:si_water_slots] = :none` (the library
# default) nothing here runs and every path is bitwise the code without it.
#
# THE COEFFICIENTS, and why they are not `ν_k/ρ_t` with the prognostic ρ_t.
# The design's consistency statement is `c_d + Σ_k α_k c_k = 1` (α_k = 2 for a transformed slot,
# 1 for a density slot), so `Δ(ρ_d' + Σ α_k ν_k') = −Δτ ∂z φ` = the slot-3 increment to rounding.
# That needs a common denominator equal to `ρ_d + Σ α_k ν_k`, which in Stage 2 is NOT the
# prognostic ρ_t: the two differ by the gap δ = ρ_t − ρ_d − Σ α_k ν_k (the vapor gap, the knee
# bias B and the lobe pool Λ). Two forms close it exactly:
#
#   * STATE-DEPENDENT path (`options[:state_dependent_si]`): the denominator is the DERIVED
#     working total ρ̂^w = ρ_d + Σ α_k ν_k of the state at n (design §4.2 "ρ̂_t = ρ_d + ρ_w,
#     recomputed each step from the slots"), for c_d as well as for every c_k:
#         c_d = ρ_d/ρ̂^w,  c_k = ν_k/ρ̂^w,  c_{·,z} by the quotient rule with ρ̂^w_z.
#     Every constituent then moves with the one velocity φ/ρ̂^w; the Helmholtz problem, φ = ρ_t w,
#     the w and p recoveries and the slot-3 update are untouched (so the acoustic operator — and
#     its ceiling — is today's). The ρ_d coefficient moves by O(δ/ρ) from `ρ_d/ρ_t`.
#   * REFERENCE paths (the resting-reference vertical solve, the horizontal ADI sweep, the exact
#     2-D solve): the coefficients are z-only, so design option (iii) applies — the water share
#     of the anchor's own increment, δρ_w = δρ_t − δρ_d, is split by the reference shares
#     s_k = ν̄_k/ρ̄_w (ρ̄_w = Σ α_k ν̄_k):
#         c_k = s_k (1 − c_d),  c_{k,z} = s_{k,z}(1 − c_d) − s_k c_{d,z},  c_d = ρ̄_d/ρ̄_t unchanged.
#     Then Σ α_k c_k = 1 − c_d by Σ α_k s_k = 1, so ρ_d, ρ_t, p, w and E_t are bitwise as before
#     and the water receives exactly the anchor's water flux. On a cloud-free base s_v = 1/α_v
#     exactly and s_{v,z} = 0 exactly; the `s_{k,z}` term is the flux-form correction the plain
#     split (iii) drops when the share varies in z (a cloudy reference) — kept, so the exact 2-D
#     solve's water legs are `c_k·drt − Δτ c_{k,z} φ`, the formula its ρ_d plane uses. Only the
#     vapor and the cloud have a reference (rain and ice are totals with ρ̄ ≡ 0), so the other
#     slots receive nothing on these paths, as their linearization about the resting base says.
#
# Energy: no partner (plan §4 Stage 2). The acoustic increments are linear in ν; their effect on
# the recovered partition at a clip point is a transport-class residual that the P3 budget
# (`options[:energy_budget]`) measures in its acoustic_si row.
#
# The `−0.0` select: every increment, history and add-back here is applied only where it is
# nonzero (`x + 0.0` is not the identity for `x = −0.0`), so a resting column and a dry one —
# every product an exact ±0 — are bitwise untouched.
#
# Compile cliff: no scratch columns (MC_SCRATCH_SLOTS is at the documented LLVM cliff). Every
# coefficient is formed inline, per point, from the slot views and the reference vectors, in
# `@noinline` helpers on plain arrays that never take the tile struct.
# ─────────────────────────────────────────────────────────────────────────────

"Allowed `options[:si_water_slots]` values (`numerics.si_water_slots`)."
const SI_WATER_SLOT_MODES = (:none, :mass, :all)

"""
    si_water_slots_mode(options) -> Symbol

Which water slots the acoustic semi-implicit solve slaves to its mass flux (Stage 2 of the ρ_t
retirement, design §4.2 (i′)):

- `:none` — the library default: no water slot is slaved; every water species takes its
  vertical (and horizontal) acoustic compression explicitly, as before this stage. Bitwise.
- `:mass` — the water MASS slots 𝒲 = {vapor, cloud, rain, the three ice masses} (the run-config
  default, `numerics.si_water_slots`).
- `:all` — 𝒲 plus the number and habit moments (rain number, the nine ice moments; design option
  (i-c)), each with `c = ν_n/ρ̂` and no weight in the water sum.

See `src/si_water_slaving.jl` for the coefficients on each path and why they close the identity
`Δ(ρ_d' + Σ α_k ν_k') = Δρ_t'` (slot 3) to rounding.
"""
@inline function si_water_slots_mode(options)
    mode = get(options, :si_water_slots, :none)::Symbol
    mode in SI_WATER_SLOT_MODES ||
        error("options[:si_water_slots] = :$(mode) is not recognized; use :none (no water slot " *
              "slaved to the acoustic solve: the library default), :mass (vapor, cloud, rain and " *
              "the three ice masses) or :all (:mass plus the number and habit moments)")
    return mode
end

"Whether `options[:si_water_slots]` slaves any slot (a function barrier for `mc_driver!`)."
@noinline _siw_enabled(options) = si_water_slots_mode(options) !== :none

"""
    SIWaterSet

The slaved water slots of one configuration, as fixed 16-wide tuples (slot index 0 = inactive),
so it is built per call without allocating: `slot[j]` the prognostic slot, `alpha[j]` its weight
in the water sum `ρ_w = Σ α ν` (2 transformed, 1 a density slot, 0 a number/habit moment),
`ref[j]` which reference it is carried against (0 none — a TOTAL slot; 1 the vapor's `ν̄_v`;
2 the cloud's `ν̄_c`), `ctrans`/`cmu` the cloud transform (its `ν̄_c = bhyp(ρ̄_c)` is formed
inline, exactly as `mc_driver!` forms it). `on` is false under `:none`.
Members, in this fixed order (the water sum's summation order): vapor, cloud, rain, the three
ice masses; then, under `:all`, the rain number and the nine ice number/habit moments.
"""
struct SIWaterSet
    on::Bool
    slot::NTuple{16,Int}
    alpha::NTuple{16,Float64}
    ref::NTuple{16,Int}
    ctrans::Bool
    cmu::Float64
end

# Literal 16-tuples: `ntuple(f, 16)` is past the unrolled range and allocates.
const _SIW_ZERO_I = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
const _SIW_ZERO_F = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

"""
    si_water_set(options, physical_params, IS::MCSlots) -> SIWaterSet

Resolve the slaved water set from the options and the tile's cached slots. Allocation-free.
"""
@noinline function si_water_set(options, physical_params, IS::MCSlots)
    mode = si_water_slots_mode(options)
    on = mode !== :none
    ice = ice_registered(IS)
    allm = mode === :all
    av = vapor_transform_mode(options) !== :none ? 2.0 : 1.0
    ctr = condensate_transform_mode(options) !== :none
    ac = ctr ? 2.0 : 1.0
    ar = rain_transform_mode(options) !== :none ? 2.0 : 1.0
    ai = (ice && ice_transform_mode(options) !== :none) ? 2.0 : 1.0
    cmu = Float64(get(physical_params, :condensate_mu, 1.0e-7))
    on || return SIWaterSet(false, _SIW_ZERO_I, _SIW_ZERO_F, _SIW_ZERO_I, ctr, cmu)
    im = ice ? 1 : 0
    nr = allm ? IS.n_r : 0
    imo = (allm && ice) ? 1 : 0
    slot = (IS.rho_v, 9, 8, im * IS.i1_q, im * IS.i2_q, im * IS.i3_q, nr,
            imo * IS.i1_n, imo * IS.i1_a, imo * IS.i1_c,
            imo * IS.i2_n, imo * IS.i2_a, imo * IS.i2_c,
            imo * IS.i3_n, imo * IS.i3_a, imo * IS.i3_c)
    alpha = (av, ac, ar, ai, ai, ai, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ref = (1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    return SIWaterSet(true, slot, alpha, ref, ctr, cmu)
end

"""
    si_water_member_names(model) -> Vector{String}

The `vars` names of the slaved water slots (empty under `:none`), for the setup checks.
"""
function si_water_member_names(model)
    uses_pressure_reference(model.equation_set) || return String[]
    ws = si_water_set(model.options, model.physical_params, mc_slots(model))
    ws.on || return String[]
    byindex = Dict(i => v for (v, i) in model.grid_params.vars)
    return [byindex[s] for s in ws.slot if s > 0]
end

"""
    validate_si_water_slots(model)

Setup-time refusal of MIXED vertical wall families across the slaved water set (design §8 risk 2;
plan Stage 2 test 5). The slaved increments are pointwise mish values that each slot then refits
on its OWN vertical basis; the identity `Δ(ρ_d' + Σ α_k ν_k') = Δρ_t'` survives the refit only if
every member is fitted with one operator — the same ground and lid condition and the same
vertical filter (`l_q`, `spline_filter`). Under the `d2` or `r1t1x` scalar families the
sedimenting slots stay natural at the ground while the vapor and cloud take the family, so the
rain/ice share of the water would carry a different wall fit (the "freeing rho_t alone" class of
`[boundary_conditions]`). Refused unless `options[:si_water_mixed_walls] = true`, the explicit
opt-in to MEASURE that combination (design §4.4 caveat 1: "refuse that combination, or measure
it"). A no-op under `si_water_slots = :none` and for non-mc equation sets.
"""
function validate_si_water_slots(model)
    names = si_water_member_names(model)
    isempty(names) && return nothing
    get(model.options, :si_water_mixed_walls, false)::Bool && return nothing
    gp = model.grid_params
    dflt(d, v) = haskey(d, v) ? d[v] : get(d, "default", nothing)
    lqk(v) = haskey(gp.l_q, v * "_k") ? gp.l_q[v * "_k"] :
             haskey(gp.l_q, v) ? gp.l_q[v] : get(gp.l_q, "default", nothing)
    sig(v) = (dflt(gp.BCB, v), dflt(gp.BCT, v), lqk(v), get(gp.spline_filter, v, nothing))
    ref = names[1]; s0 = sig(ref)
    for v in names[2:end]
        s = sig(v)
        s == s0 && continue
        what = s[1] != s0[1] ? "ground BC (BCB) $(s[1]) vs $(s0[1])" :
               s[2] != s0[2] ? "lid BC (BCT) $(s[2]) vs $(s0[2])" :
               s[3] != s0[3] ? "vertical l_q $(s[3]) vs $(s0[3])" : "spline_filter"
        error("options[:si_water_slots] = :$(si_water_slots_mode(model.options)) slaves the " *
              "water slots $(join(names, ", ")) to the acoustic mass flux, but \"$(v)\" and " *
              "\"$(ref)\" do not share one vertical operator ($(what)). The slaved increments " *
              "are refit slot by slot, so a mixed wall family breaks the water/anchor identity " *
              "at the wall (design risk 2: under d2/r1t1x families the sedimenting slots stay " *
              "natural at the ground). Use one family for every water slot (the natural " *
              "family: [boundary_conditions] vertical_scalars = vertical_scalars_lid = " *
              "\"natural\", sedimentation_lid = \"natural\"), or set numerics.si_water_slots = " *
              "\"none\", or set options[:si_water_mixed_walls] = true to run and MEASURE it.")
    end
    return nothing
end

# ── Per-point values ─────────────────────────────────────────────────────────

# The TOTAL control variable ν_k = ν_k' + ν̄_k of member j at row g (level i) and its total
# vertical derivative, from the slot views (`phys[:, slot, 1]` and `phys[:, slot, zs]`) plus the
# member's reference — the same `perturbation + reference` lines `mc_driver!` forms (the vapor
# against `mc_nu_vbar`, the cloud against `bhyp(ρ̄_c)` or `ρ̄_c`, every other slot a total).
@inline function _siw_nu(phys, g::Int, zs::Int, s::Int, ref::Int, i::Int, ctrans::Bool,
                         cmu::Float64, nvb, nvbz, rcb, rcbz)
    @inbounds begin
        v = phys[g, s, 1]; vz = phys[g, s, zs]
        if ref == 1
            return (v + nvb[i], vz + nvbz[i])
        elseif ref == 2
            if ctrans
                return (v + bhyp(rcb[i], cmu), vz + (dbhyp(rcb[i], cmu) * rcbz[i]))
            end
            return (v + rcb[i], vz + rcbz[i])
        end
        return (v, vz)
    end
end

# The member's REFERENCE ν̄_k and ν̄_{k,z} at level i (0 for a total slot).
@inline function _siw_nubar(ref::Int, i::Int, ctrans::Bool, cmu::Float64, nvb, nvbz, rcb, rcbz)
    @inbounds begin
        ref == 1 && return (nvb[i], nvbz[i])
        if ref == 2
            ctrans && return (bhyp(rcb[i], cmu), dbhyp(rcb[i], cmu) * rcbz[i])
            return (rcb[i], rcbz[i])
        end
        return (0.0, 0.0)
    end
end

# The water sum W = Σ α_k ν_k of the state at row g and its z-derivative, in the members' fixed
# order; the first term is assigned rather than added to 0.0, so a dry column's W is the exact
# product its first member gives (and ρ_d + W is ρ_d bitwise whatever the sign of that zero).
@inline function _siw_W(phys, g::Int, zs::Int, i::Int, ws::SIWaterSet, nvb, nvbz, rcb, rcbz)
    W = 0.0; Wz = 0.0; first = true
    @inbounds for j in 1:16
        s = ws.slot[j]; a = ws.alpha[j]
        (s == 0 || a == 0.0) && continue
        v, vz = _siw_nu(phys, g, zs, s, ws.ref[j], i, ws.ctrans, ws.cmu, nvb, nvbz, rcb, rcbz)
        if first
            W = a * v; Wz = a * vz; first = false
        else
            W = W + (a * v); Wz = Wz + (a * vz)
        end
    end
    return (W, Wz)
end

# The reference water sum ρ̄_w = Σ α_k ν̄_k (vapor and cloud: the only references) at level i.
@inline function _siw_Wbar(i::Int, ws::SIWaterSet, nvb, nvbz, rcb, rcbz)
    W = 0.0; Wz = 0.0; first = true
    @inbounds for j in 1:16
        (ws.slot[j] == 0 || ws.ref[j] == 0) && continue
        a = ws.alpha[j]
        v, vz = _siw_nubar(ws.ref[j], i, ws.ctrans, ws.cmu, nvb, nvbz, rcb, rcbz)
        if first
            W = a * v; Wz = a * vz; first = false
        else
            W = W + (a * v); Wz = Wz + (a * vz)
        end
    end
    return (W, Wz)
end

# The quotient-rule pair (c, c_z) of c = n/d.
@inline _siw_quot(n, nz, d, dz) = (n / d, ((nz * d) - (n * dz)) / (d * d))

# The reference share s_k = ν̄_k/ρ̄_w and its z-derivative; (0, 0) where ρ̄_w == 0 (a dry level:
# no water to share, and the anchor's water increment is then an exact zero too).
@inline function _siw_share(ref::Int, i::Int, ws::SIWaterSet, Wb, Wbz, nvb, nvbz, rcb, rcbz)
    Wb == 0.0 && return (0.0, 0.0)
    v, vz = _siw_nubar(ref, i, ws.ctrans, ws.cmu, nvb, nvbz, rcb, rcbz)
    return _siw_quot(v, vz, Wb, Wbz)
end

# The slaved coefficient pair (c_k, c_{k,z}) of member j at row g / level i, on either path:
# state-dependent (`sd`: ν_k/ρ̂^w with ρ̂^w = ρ_d + W of the state at n) or reference (the share
# form s_k(1 − c_d), with `cd`, `cdz` the reference ρ_d coefficient pair).
@inline function _siw_coef(sd::Bool, phys, g::Int, zs::Int, i::Int, j::Int, ws::SIWaterSet,
                           rh, rhz, cd, cdz, Wb, Wbz, nvb, nvbz, rcb, rcbz)
    @inbounds begin
        if sd
            v, vz = _siw_nu(phys, g, zs, ws.slot[j], ws.ref[j], i, ws.ctrans, ws.cmu,
                            nvb, nvbz, rcb, rcbz)
            return _siw_quot(v, vz, rh, rhz)
        end
        s, sz = _siw_share(ws.ref[j], i, ws, Wb, Wbz, nvb, nvbz, rcb, rcbz)
        om = 1.0 - cd
        return (s * om, (sz * om) - (s * cdz))
    end
end

# Whether member j takes part on this path: every active member on the state-dependent path,
# only the referenced ones (vapor, cloud) on the reference paths.
@inline _siw_active(ws::SIWaterSet, j::Int, sd::Bool) =
    @inbounds(ws.slot[j] != 0 && (sd || ws.ref[j] != 0))

# The vertical-derivative slot of a moist-compressible physical array: 4 on the 2-D grids
# (f, f_x, f_xx, f_z, f_zz), 6 on the 3-D ones (f_z/f_zz after the second horizontal pair) —
# `zslot(geom)` for callers that have no geometry trait in hand.
@inline siw_zslot(phys) = size(phys, 3) >= 7 ? 6 : 4

# ── The state-dependent ρ_d coefficient (driver staging and the solve) ───────

"""
    _siw_sd_cd!(c_d, c_dz, rho_d, rho_d_z, phys, colstart, zs, ws, nvb, nvbz, rcb, rcbz)

The state-dependent ρ_d coefficient pair under water slaving: `c_d = ρ_d/ρ̂^w` with the DERIVED
working total `ρ̂^w = ρ_d + Σ α_k ν_k` of the state at n, by the quotient rule — the denominator
every slaved coefficient shares, so `c_d + Σ α_k c_k = 1` to rounding. Called by the AI2*
staging in `mc_driver!` and by `semiimplicit_adjustment_p` with the same arguments, so the two
see one operator bit for bit.
"""
@noinline function _siw_sd_cd!(c_d::AbstractVector{Float64}, c_dz::AbstractVector{Float64},
                               rho_d::AbstractVector{Float64}, rho_d_z::AbstractVector{Float64},
                               phys::Array{Float64,3}, colstart::Int, zs::Int, ws::SIWaterSet,
                               nvb, nvbz, rcb, rcbz)
    @inbounds for i in eachindex(c_d)
        g = colstart + i - 1
        W, Wz = _siw_W(phys, g, zs, i, ws, nvb, nvbz, rcb, rcbz)
        rh = rho_d[i] + W
        rhz = rho_d_z[i] + Wz
        c, cz = _siw_quot(rho_d[i], rho_d_z[i], rh, rhz)
        c_d[i] = c
        c_dz[i] = cz
    end
    return nothing
end

# ── AI2* history staging (mc_driver!) ────────────────────────────────────────

"""
    _siw_stage_impdot!(impdot, colstart, ws, sd, phys, zs, rho_d, rho_d_z, cd, cdz,
                       phi_z, phi, nvb, nvbz, rcb, rcbz)

Stage `impdot[k] = −(c_k φ_z + c_{k,z} φ)` for every slaved water slot from the SAME fitted
`imp_phi_z` / `phi_n` the dry legs' histories use (design §4.2: every AI2* level sees one
operator). `cd`/`cdz` are the reference ρ_d coefficient pair (read on the reference path only).
"""
@noinline function _siw_stage_impdot!(impdot::Matrix{Float64}, colstart::Int, ws::SIWaterSet,
                                      sd::Bool, phys::Array{Float64,3}, zs::Int,
                                      rho_d::AbstractVector{Float64},
                                      rho_d_z::AbstractVector{Float64},
                                      cd::AbstractVector{Float64}, cdz::AbstractVector{Float64},
                                      phi_z::AbstractVector{Float64},
                                      phi::AbstractVector{Float64}, nvb, nvbz, rcb, rcbz)
    @inbounds for i in eachindex(phi)
        g = colstart + i - 1
        rh = 0.0; rhz = 0.0; Wb = 0.0; Wbz = 0.0
        if sd
            W, Wz = _siw_W(phys, g, zs, i, ws, nvb, nvbz, rcb, rcbz)
            rh = rho_d[i] + W; rhz = rho_d_z[i] + Wz
        else
            Wb, Wbz = _siw_Wbar(i, ws, nvb, nvbz, rcb, rcbz)
        end
        for j in 1:16
            _siw_active(ws, j, sd) || continue
            c, cz = _siw_coef(sd, phys, g, zs, i, j, ws, rh, rhz, cd[i], cdz[i], Wb, Wbz,
                              nvb, nvbz, rcb, rcbz)
            impdot[g, ws.slot[j]] = -((c * phi_z[i]) + (cz * phi[i]))
        end
    end
    return nothing
end

# ── The explicit add-back (mc_driver!, after the water slots are assembled) ──

"""
    _siw_addback!(expdot, colstart, ws, sd, phys, zs, w, w_z, rho_t, rho_t_z, rho_d, rho_d_z,
                  rho_tbar, rho_tbar_z, rho_dbar, rho_dbar_z, nvb, nvbz, rcb, rcbz,
                  hl, LDIV, hacdot, fresh)

Add the linear flux the solve now integrates back into each slaved slot's explicit tendency,
beside the existing `−J_k ρ_k ∇·v` (design §4.2 (i′): the ν continuity is `−w ν_z − Jρ w_z`; the
remainder leaves only the `+D_k w_z` knee term explicit, `D_k ≤ μ_k/2`).

THE ADD-BACK IS THE SOLVE'S OWN LINEAR OPERATOR evaluated on the state at n, pointwise:
`∂z(c_k φⁿ) = c_k φⁿ_z + c_{k,z} φⁿ` with `φⁿ = ρ̂_t w` — on the state-dependent path
`ρ̂_t = ρ_t`, `φⁿ_z = ρ_t w_z + ρ_{t,z} w`, the coefficients the solve slaves with
(`ν_k/ρ̂^w`); on the reference paths `c_k ρ̄_t = s_k(ρ̄_t − ρ̄_d)` (the slot-2 add-back
`ρ̄_d w_z + ρ̄_{d,z} w` is already this form for ρ_d). Where ρ̂^w = ρ_t this is the design's
`ν_k w_z + ν_{k,z} w`; where they differ by the partition gap δ = ρ_t − ρ_d − Σ α ν, adding back
`ν_k w_z + ν_{k,z} w` would leave `∂z(δ w)` in the explicit tendencies with no implicit partner —
the gap would lose its vertical transport (measured: a seeded gap stops being a passive tracer).
For the same reason ρ_d's state-dependent add-back is corrected here to the same operator (its
`c_d = ρ_d/ρ̂^w`): the slot-2 expression `ρ_d w_z + ρ_{d,z} w` is replaced by `∂z(c_d φⁿ)`, the
difference added. Then Σ over ρ_d and 𝒲 of the add-backs is slot 3's `∂z(ρ_t w)` to rounding, and
the gap is carried by `−∇·(δ v)` exactly as a passive tracer.

With `hl` (the horizontal SI or the exact 2-D solve, `hsi_like`; `LDIV` the linear horizontal
divergence) the horizontal remainder `+s_k(ρ̄_t − ρ̄_d)·LDIV` too, and under the fresh horizontal
history `hacdot[k] = −s_k(ρ̄_t − ρ̄_d)·LDIV` (the slot-2 pattern of the `hsi_like` block in
`mc_driver!`). Every write is skipped where its value is an exact zero.
"""
@noinline function _siw_addback!(expdot::Matrix{Float64}, colstart::Int, ws::SIWaterSet,
                                 sd::Bool, phys::Array{Float64,3}, zs::Int,
                                 w::AbstractVector{Float64}, w_z::AbstractVector{Float64},
                                 rho_t::AbstractVector{Float64},
                                 rho_t_z::AbstractVector{Float64},
                                 rho_d::AbstractVector{Float64},
                                 rho_d_z::AbstractVector{Float64},
                                 rho_tbar, rho_tbar_z, rho_dbar, rho_dbar_z,
                                 nvb, nvbz, rcb, rcbz, hl::Bool,
                                 LDIV::AbstractVector{Float64}, hacdot::Matrix{Float64},
                                 fresh::Bool)
    @inbounds for i in eachindex(w)
        g = colstart + i - 1
        Wb, Wbz = _siw_Wbar(i, ws, nvb, nvbz, rcb, rcbz)
        dw = rho_tbar[i] - rho_dbar[i]
        dwz = rho_tbar_z[i] - rho_dbar_z[i]
        rh = 0.0; rhz = 0.0; ph = 0.0; phz = 0.0
        if sd
            W, Wz = _siw_W(phys, g, zs, i, ws, nvb, nvbz, rcb, rcbz)
            rh = rho_d[i] + W; rhz = rho_d_z[i] + Wz
            ph = rho_t[i] * w[i]                                  # φⁿ = ρ_t w
            phz = (rho_t[i] * w_z[i]) + (rho_t_z[i] * w[i])       # φⁿ_z, pointwise
            cd, cdz = _siw_quot(rho_d[i], rho_d_z[i], rh, rhz)
            ad = ((cd * phz) + (cdz * ph)) - ((rho_d[i] * w_z[i]) + (rho_d_z[i] * w[i]))
            ad != 0.0 && (expdot[g, 2] += ad)
        end
        for j in 1:16
            s = ws.slot[j]
            s == 0 && continue
            if sd
                v, vz = _siw_nu(phys, g, zs, s, ws.ref[j], i, ws.ctrans, ws.cmu,
                                nvb, nvbz, rcb, rcbz)
                c, cz = _siw_quot(v, vz, rh, rhz)
                a = (c * phz) + (cz * ph)
                a != 0.0 && (expdot[g, s] += a)
            end
            ws.ref[j] == 0 && continue
            sh, shz = _siw_share(ws.ref[j], i, ws, Wb, Wbz, nvb, nvbz, rcb, rcbz)
            vt = sh * dw
            if !sd
                vtz = (shz * dw) + (sh * dwz)
                a = (vt * w_z[i]) + (vtz * w[i])
                a != 0.0 && (expdot[g, s] += a)
            end
            if hl
                h = vt * LDIV[i]
                h != 0.0 && (expdot[g, s] += h)
                fresh && (hacdot[g, s] = -h)
            end
        end
    end
    return nothing
end

"""
    _siw_stage!(mtile, colstart, colend)

The water slaving's per-column work on the EXPLICIT side, called from `explicit_timestep` (after
`mc_driver!` has assembled every slot's explicit tendency, before the AB3 combination reads it)
under `options[:si_water_slots] ≠ :none`:

1. on the state-dependent path, restage the ρ_d history with `c_d = ρ_d/ρ̂^w`
   (`_siw_sd_cd!` into `imp_c_d`/`imp_c_d_z`, then `impdot[2] = −(c_d φ_z + c_{d,z} φ)` — the
   driver's expression, same operations);
2. stage the slaved water slots' histories (`_siw_stage_impdot!`) from the SAME `imp_phi_z` /
   `phi_n` fit the dry legs used (the driver's `S.imp_phi_z` and the w column's mish values,
   `scratch_column(mtile, 5).uMish`, which nothing between the staging and the solve refits);
3. add the solve's linear operator back into their explicit tendencies (`_siw_addback!`; with the
   horizontal remainder and fresh history under the horizontal SI / exact 2-D solve, whose
   linear divergence is formed again into `S.ADV`).

WHY HERE and not in `mc_driver!`: that body's first-call compile sits at the LLVM cliff. Every
call site tried there moved it — the ~30-argument call +33..+75 s on the axisymmetric Louis tile,
a dispatch barrier nothing but 10-12 boxed arguments per column, the same arguments bundled into
one stack struct within noise on the cylinder but past 35 min on the SLR shell (2026-10-21).
`mc_driver!` is now byte-for-byte be03fe9's, and everything the slaving needs is read here from
the tile and the thread's scratch, which still hold the column the driver just assembled.
"""
@noinline function _siw_stage!(mtile::ModelTile, colstart::Int, colend::Int)
    model = mtile.model
    opts = model.options
    ws = si_water_set(opts, model.physical_params, mtile.mc_slots)
    ws.on || return nothing
    S = @inbounds mtile.mc_scratch[Threads.threadid()]
    sd = get(opts, :state_dependent_si, false)::Bool
    hsi = get(opts, :horizontal_semiimplicit, false)::Bool
    xsi = get(opts, :exact_si, false)::Bool
    hsi_like = hsi || (xsi && !(get(opts, :exact_si_zero_x, false)::Bool))
    phys = mtile.tile.physical
    zs = siw_zslot(phys)
    w_i = model.grid_params.vars["w"]
    w = view(phys, colstart:colend, w_i, 1)
    w_z = view(phys, colstart:colend, w_i, zs)
    rs = mtile.ref_state
    rtb = ref_rho_t(rs); rdb = ref_rho_d(rs); rc = Springsteel.ref_rho_c(rs)
    rcb = view(rc, :, 1); rcbz = view(rc, :, 2)
    nvb = mc_nu_vbar(mtile); nvbz = mc_nu_vbar_z(mtile)
    phi = scratch_column(mtile, w_i).uMish
    phi_z = S.imp_phi_z
    impdot = mtile.impdot_n
    if sd
        _siw_sd_cd!(S.imp_c_d, S.imp_c_d_z, S.rho_d, S.rho_d_z, phys, colstart, zs, ws,
                    nvb, nvbz, rcb, rcbz)
        _siw_impdot2!(impdot, colstart, S.imp_c_d, S.imp_c_d_z, phi_z, phi)
    end
    _siw_stage_impdot!(impdot, colstart, ws, sd, phys, zs, S.rho_d, S.rho_d_z, S.imp_c_d,
                       S.imp_c_d_z, phi_z, phi, nvb, nvbz, rcb, rcbz)
    LDIV = S.ADV
    hsi_like && _siw_ldiv!(LDIV, mtile, colstart, colend)
    fresh = hsi_like && (get(opts, :hsi_x_history, "fresh") == "fresh")::Bool
    _siw_addback!(mtile.expdot_n, colstart, ws, sd, phys, zs, w, w_z, S.rho_t, S.rho_t_z,
                  S.rho_d, S.rho_d_z, view(rtb, :, 1), view(rtb, :, 2), view(rdb, :, 1),
                  view(rdb, :, 2), nvb, nvbz, rcb, rcbz, hsi_like, LDIV, mtile.hacdot_n, fresh)
    return nothing
end

# `impdot[2] = −(c_d φ_z + c_{d,z} φ)`, the driver's broadcast, same operations.
@noinline function _siw_impdot2!(impdot::Matrix{Float64}, colstart::Int, c_d, c_dz, phi_z, phi)
    @inbounds for i in eachindex(phi)
        impdot[colstart + i - 1, 2] = -((c_d[i] * phi_z[i]) + (c_dz[i] * phi[i]))
    end
    return nothing
end

# The reference-linear horizontal divergence the `hsi_like` block of `mc_driver!` forms
# (`mc_linear_div!`), for the geometries the horizontal SI / exact 2-D solve run on.
function _siw_ldiv!(LDIV, mtile::ModelTile, colstart::Int, colend::Int)
    es = mtile.model.equation_set
    if es == "moist_compressible_XZ"
        _siw_ldiv_g!(LDIV, MCCartesianXZ(), mtile, colstart, colend)
    elseif es == "moist_compressible_axisym"
        _siw_ldiv_g!(LDIV, MCAxisymRZ(), mtile, colstart, colend)
    elseif es == "moist_compressible_RLR"
        _siw_ldiv_g!(LDIV, MCCylindricalRLR(), mtile, colstart, colend)
    else
        error("si_water_slots with the horizontal SI / exact 2-D solve: no linear horizontal " *
              "divergence for equation set $(es)")
    end
    return nothing
end
function _siw_ldiv_g!(LDIV, geom, mtile::ModelTile, colstart::Int, colend::Int)
    vars = mtile.model.grid_params.vars
    uv = mc_slot_views(mtile.tile, colstart, colend, vars["u"], geom)
    vv = has_v(geom) ? mc_slot_views(mtile.tile, colstart, colend, vars["v"], geom) : uv
    r = mc_metric(geom, mtile.model, mtile.tilepoints, colstart, colend)
    mc_linear_div!(LDIV, geom, uv, vv, r)
    return nothing
end

# ── The solve (semiimplicit_adjustment_p, apply_acoustic_histories!) ─────────

"""
    _siw_histories!(vnp1, dot_n, dot_nm1, colstart, colend, ws, sd, t, ts)

Apply the AI2* explicit history levels of the slaved water slots to the predictor in place
(`ts·(0.75 L^{n−1} − L^n)`, `+0.5 ts L^n` on the AM2 first step) and roll the history — the
loop `semiimplicit_adjustment_p` runs over (p, w, ρ_d, ρ_t, E_t), extended to 𝒲, and hoisted to
phase A under `options[:exact_si]` (`apply_acoustic_histories!`). Skipped where the level is 0.
"""
@noinline function _siw_histories!(vnp1::Matrix{Float64}, dot_n::Matrix{Float64},
                                   dot_nm1::Matrix{Float64}, colstart::Int, colend::Int,
                                   ws::SIWaterSet, sd::Bool, t::Int, ts::Float64)
    @inbounds for j in 1:16
        _siw_active(ws, j, sd) || continue
        s = ws.slot[j]
        for g in colstart:colend
            dn = dot_n[g, s]
            h = (t == 1) ? (ts * 0.5 * dn) : (ts * ((0.75 * dot_nm1[g, s]) - dn))
            h != 0.0 && (vnp1[g, s] += h)
            dot_nm1[g, s] = dn
        end
    end
    return nothing
end

"""
    _siw_slave!(vnp1, colstart, ws, sd, ts_term, phi, phi_z, phys, zs, rho_d, rho_d_z,
                cd, cdz, nvb, nvbz, rcb, rcbz)

The slaved ν-flux increments of the vertical solve: `ν_k' −= Δτ (c_k φ_z + c_{k,z} φ)` from the
solve's own φ and φ_z, the coefficients exactly those `_siw_stage_impdot!` staged the history
with. Skipped where the increment is 0.
"""
@noinline function _siw_slave!(vnp1::Matrix{Float64}, colstart::Int, ws::SIWaterSet, sd::Bool,
                               ts_term::Float64, phi::AbstractVector{Float64},
                               phi_z::AbstractVector{Float64}, phys::Array{Float64,3}, zs::Int,
                               rho_d::AbstractVector{Float64}, rho_d_z::AbstractVector{Float64},
                               cd::AbstractVector{Float64}, cdz::AbstractVector{Float64},
                               nvb, nvbz, rcb, rcbz)
    @inbounds for i in eachindex(phi)
        g = colstart + i - 1
        rh = 0.0; rhz = 0.0; Wb = 0.0; Wbz = 0.0
        if sd
            W, Wz = _siw_W(phys, g, zs, i, ws, nvb, nvbz, rcb, rcbz)
            rh = rho_d[i] + W; rhz = rho_d_z[i] + Wz
        else
            Wb, Wbz = _siw_Wbar(i, ws, nvb, nvbz, rcb, rcbz)
        end
        for j in 1:16
            _siw_active(ws, j, sd) || continue
            c, cz = _siw_coef(sd, phys, g, zs, i, j, ws, rh, rhz, cd[i], cdz[i], Wb, Wbz,
                              nvb, nvbz, rcb, rcbz)
            d = ts_term * ((c * phi_z[i]) + (cz * phi[i]))
            d != 0.0 && (vnp1[g, ws.slot[j]] -= d)
        end
    end
    return nothing
end

"""
    _siw_solve!(vnp1, dot_n, dot_nm1, colstart, colend, t, ts, ts_term, apply_histories, sd,
                phi, phi_z, phys, rho_d, rho_d_z, c_d, c_dz, rhod_nstar, rhod_index, options,
                physical_params, IS, nvb, nvbz, rho_cref)

`semiimplicit_adjustment_p`'s ONE call into the water slaving, after its slaved updates (the
solve's first-call compile is part of the same cliff as `mc_driver!`'s). Returns at once under
`:none`. Otherwise: on the state-dependent path re-advance ρ_d with `c_d = ρ_d/ρ̂^w`
(`_siw_sd_cd!` into `c_d`/`c_dz`, then the solve's own expression
`ρ_d' = ρ_d'* − Δτ (c_d φ_z + c_{d,z} φ)` from the saved predictor); apply the slaved slots' AI2*
histories (`apply_histories`); and the slaved increments (`_siw_slave!`). `rho_cref` is the
reference cloud (value, ∂z) matrix.
"""
@noinline function _siw_solve!(vnp1::Matrix{Float64}, dot_n::Matrix{Float64},
                               dot_nm1::Matrix{Float64}, colstart::Int, colend::Int, t::Int,
                               ts::Float64, ts_term::Float64, apply_histories::Bool, sd::Bool,
                               phi, phi_z, phys, rho_d, rho_d_z, c_d, c_dz, rhod_nstar,
                               rhod_index::Int, options, physical_params, IS::MCSlots,
                               nvb, nvbz, rho_cref)
    ws = si_water_set(options, physical_params, IS)
    ws.on || return nothing
    zs = siw_zslot(phys)
    rcb = view(rho_cref, :, 1); rcbz = view(rho_cref, :, 2)
    if sd
        _siw_sd_cd!(c_d, c_dz, rho_d, rho_d_z, phys, colstart, zs, ws, nvb, nvbz, rcb, rcbz)
        @inbounds for i in eachindex(phi)
            vnp1[colstart + i - 1, rhod_index] =
                rhod_nstar[i] - (ts_term * ((c_d[i] * phi_z[i]) + (c_dz[i] * phi[i])))
        end
    end
    apply_histories && _siw_histories!(vnp1, dot_n, dot_nm1, colstart, colend, ws, sd, t, ts)
    _siw_slave!(vnp1, colstart, ws, sd, ts_term, phi, phi_z, phys, zs, rho_d, rho_d_z,
                c_d, c_dz, nvb, nvbz, rcb, rcbz)
    return nothing
end

"""
    _siw_xsi_apply!(vnp1, hf, rows, colstart, ws, ts_term, w_index, rho_tbar, rho_dbar,
                    rho_dbar_z, rho_tbar_z, nvb, nvbz, rcb, rcbz)

Phase B of the exact 2-D solve (`exact_si_apply_column!`, XZ/axisym and RLR alike): the
slaved water increments from the anchor's own planes, `δν_k = s_k (δρ_t − δρ_d) −
Δτ s_{k,z}(1 − c_d) φ^{n+1}` (planes 5 and 4; `φ^{n+1} = ρ̄_t w^{n+1}` with the w increment
already applied) — algebraically `c_k·drt − Δτ c_{k,z} φ^{n+1}` with `c_k = s_k(1 − c_d)`, the
formula plane 4 uses for ρ_d, and `Σ α_k δν_k = δρ_t − δρ_d` to rounding. Called AFTER the
w/ρ_d/ρ_t planes have been added. Skipped where the increment is 0.
"""
@noinline function _siw_xsi_apply!(vnp1::Matrix{Float64}, hf::AbstractMatrix{Float64},
                                   rows::UnitRange{Int}, colstart::Int, ws::SIWaterSet,
                                   ts_term::Float64, w_index::Int, rho_tbar, rho_tbar_z,
                                   rho_dbar, rho_dbar_z, nvb, nvbz, rcb, rcbz)
    @inbounds for i in eachindex(rows)
        g = colstart + i - 1
        r = rows[i]
        Wb, Wbz = _siw_Wbar(i, ws, nvb, nvbz, rcb, rcbz)
        Wb == 0.0 && continue
        dw = hf[r, 5] - hf[r, 4]
        rt = rho_tbar[i]
        cd, cdz = _siw_quot(rho_dbar[i], rho_dbar_z[i], rt, rho_tbar_z[i])
        phin1 = rt * vnp1[g, w_index]
        for j in 1:16
            (ws.slot[j] == 0 || ws.ref[j] == 0) && continue
            s, sz = _siw_share(ws.ref[j], i, ws, Wb, Wbz, nvb, nvbz, rcb, rcbz)
            d = (s * dw) - (ts_term * ((sz * (1.0 - cd)) * phin1))
            d != 0.0 && (vnp1[g, ws.slot[j]] += d)
        end
    end
    return nothing
end

"""
    _siw_hsi_history!(vnp1, dot_n, dot_nm1, colstart, colend, ws, t, ts, am2)

The horizontal acoustic legs' explicit AI2* history levels for the referenced water slots (the
`horizontal_si_history!` loop extended to 𝒲); skipped where the level is 0.
"""
@noinline function _siw_hsi_history!(vnp1::Matrix{Float64}, dot_n::Matrix{Float64},
                                     dot_nm1::Matrix{Float64}, colstart::Int, colend::Int,
                                     ws::SIWaterSet, t::Int, ts::Float64, am2::Bool)
    @inbounds for j in 1:16
        (ws.slot[j] == 0 || ws.ref[j] == 0) && continue
        s = ws.slot[j]
        for g in colstart:colend
            dn = dot_n[g, s]
            h = (t == 1 || am2) ? ((0.5 * ts) * dn) : (ts * ((0.75 * dot_nm1[g, s]) - dn))
            h != 0.0 && (vnp1[g, s] += h)
            dot_nm1[g, s] = dn
        end
    end
    return nothing
end

"""
    si_water_hsi_profiles(mtile) -> (slots::Vector{Int}, share::Matrix{Float64})

The referenced slaved water slots and their reference shares `s_k(z) = ν̄_k/ρ̄_w` at the kDim
mish levels (`kDim × n`), for the patch-level horizontal sweep (`create_horizontal_solve_data`'s
`water` keyword): it splits the anchor's water increment `δρ_t − δρ_d` by these shares (design
option (iii), exact for the z-only horizontal coefficients). Empty under `:none` or on a dry base.
"""
function si_water_hsi_profiles(mtile)
    ws = si_water_set(mtile.model.options, mtile.model.physical_params, mtile.mc_slots)
    slots = Int[]
    ws.on || return (slots, zeros(0, 0))
    nvb = mc_nu_vbar(mtile); nvbz = mc_nu_vbar_z(mtile)
    rc = Springsteel.ref_rho_c(mtile.ref_state)
    rcb = view(rc, :, 1); rcbz = view(rc, :, 2)
    kDim = length(nvb)
    cols = Vector{Vector{Float64}}()
    for j in 1:16
        (ws.slot[j] == 0 || ws.ref[j] == 0) && continue
        sh = zeros(kDim)
        for i in 1:kDim
            Wb, Wbz = _siw_Wbar(i, ws, nvb, nvbz, rcb, rcbz)
            sh[i] = first(_siw_share(ws.ref[j], i, ws, Wb, Wbz, nvb, nvbz, rcb, rcbz))
        end
        all(iszero, sh) && continue
        push!(slots, ws.slot[j]); push!(cols, sh)
    end
    share = isempty(cols) ? zeros(kDim, 0) : reduce(hcat, cols)
    return (slots, share)
end

"""
    _siw_hsi_stored!(hacdot_n, hacdot_nm1, incr, rows, ws, t, nvb, nvbz, rcb, rcbz, kDim)

Under `options[:hsi_x_history] = "stored"`: the referenced water slots' horizontal histories are
the anchor's stored water increment `incr[:,4] − incr[:,3]` (ρ_t minus ρ_d, already divided by
Δτ) split by the reference shares — one stored δρ_w plane, design option (iii).
"""
function _siw_hsi_stored!(hacdot_n::Matrix{Float64}, hacdot_nm1::Matrix{Float64},
                          incr::AbstractMatrix{Float64}, rows::UnitRange{Int}, ws::SIWaterSet,
                          t::Int, nvb, nvbz, rcb, rcbz, kDim::Int)
    @inbounds for j in 1:16
        (ws.slot[j] == 0 || ws.ref[j] == 0) && continue
        s = ws.slot[j]
        for (l, r) in enumerate(rows)
            i = mod1(l, kDim)
            Wb, Wbz = _siw_Wbar(i, ws, nvb, nvbz, rcb, rcbz)
            sh = first(_siw_share(ws.ref[j], i, ws, Wb, Wbz, nvb, nvbz, rcb, rcbz))
            hacdot_n[l, s] = sh * (incr[r, 4] - incr[r, 3])
            t == 2 && (hacdot_nm1[l, s] = hacdot_n[l, s])
        end
    end
    return nothing
end
