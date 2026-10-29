# ─────────────────────────────────────────────────────────────────────────────
# THE WATER ANCHOR AND EVERYTHING THAT KEEPS IT — the ρ_t retirement's code
# (reference/RHO_T_RETIREMENT_DESIGN.md; reference/Scythe_moist_compressible.tex §"The derived
# total density"). This file hosts, in order:
#
#   1. the ν-flux SLAVING of the water slots in the acoustic semi-implicit solve (Stage 2,
#      `options[:si_water_slots]`, library default :none, run-config default :mass) — the rest
#      of this header;
#   2. the DERIVED WATER ANCHOR API (Stages 3-4): `water_anchor_set` (the members, weights
#      α_k = 2 transformed / 1 density, and the fixed summation order shared with the slaving),
#      `anchor_Wp`/`anchor_W_value`/`anchor_rho_tp`, the driver staging `_anchor_stage!` and its
#      readers (`_anchor_w_forcing!`, `_anchor_star!`): every reader of the total density takes
#      ρ_t = ρ_d + Σ_k α_k ν_k, ρ_t' = ρ_d' + Σ_k α_k ν_k', ρ̄_t^w = ρ̄_d + Σ_k α_k ν̄_k;
#      `validate_water_anchor`, the setup refusals of every retired device (the nudge, slot 3's
#      water diffusion, the clamp floor, the lobe attribution, the MYNN slot-anchor closures,
#      the four ice-anchor legs deleted at Stage 5);
#   3. the EXCHANGE CLOSURE (Stage 3b): `water_exchange_dv`, the ONE function every water
#      exchange's vapor increment comes from (four call sites: the ETD increments
#      `_etd_closure!`, the multistep sources `water_exchange_multistep_rate` on the ice branch
#      of `_water_exchange_multistep!` and `water_exchange_dv` directly on its warm branch, the
#      paired Stage L `_lobe_relax_paired!`), the sedimentation knee closure, and `_water_exchange_stage!`, which runs the closures and the
#      affine-defect refit partner from `explicit_timestep` (outside `mc_driver!`, for the
#      compile cliff; reference/HANDOFF_DRIVER_COMPILE_CLIFF.md).
#
# The energy side (the partner primitive `recovered_change_partner`, the refit and ETD partners)
# is in src/moist_compressible.jl; the per-stage budget of R and X in src/energy_budget.jl.
#
# ── 1. ν-flux slaving of the water slots in the acoustic semi-implicit solve — Stage 2
#    (reference/RHO_T_RETIREMENT_DESIGN.md §4.2 options (i′) and (iii), §6 Stage 2).
#
# The vertical acoustic solve advanced ρ_t' (slot 3 until Stage 4) by the implicit flux divergence
# −Δτ ∂z φ and ρ_d' by −Δτ ∂z(c_d φ) (φ = ρ̂_t w), while every water species took its vertical
# compression −J ρ ∂z w EXPLICITLY, under AB3. The density the SI slaved and the species that
# should add up to it therefore advanced under two different integrators, and the difference was
# the first feeder of the old vapor gap (design §2). Here every water-mass slot k ∈ 𝒲 is slaved to
# the same φ in flux form,
#
#     Δν_k' = −Δτ (c_k φ_z + c_{k,z} φ),
#
# its AI2* history `impdot[k] = −(c_k φ_z + c_{k,z} φ)` is staged from the SAME `imp_phi_z`/`phi_n`
# fit the dry legs use, and the explicit tendency adds back the SAME linear operator evaluated on
# the state at n (the product-rule remainder pattern of slot 2; `_siw_addback!`), so the
# equation is unchanged and only the integrator of its vertical acoustic part moves. Under `options[:si_water_slots] = :none` (the library
# default) nothing here runs and every path is bitwise the code without it.
#
# THE COEFFICIENTS, and why they are not `ν_k/ρ_t` with the prognostic ρ_t.
# The design's consistency statement is `c_d + Σ_k α_k c_k = 1` (α_k = 2 for a transformed slot,
# 1 for a density slot), so `Δ(ρ_d' + Σ α_k ν_k') = −Δτ ∂z φ` to rounding. That needs a common
# denominator equal to `ρ_d + Σ α_k ν_k`, which in Stage 2 was NOT the prognostic ρ_t (the two
# differed by the gap δ = ρ_t − ρ_d − Σ α_k ν_k: the vapor gap, the knee bias B and the lobe pool
# Λ); since Stage 3 it IS the derived total every reader takes. Two forms close it exactly:
#
#   * STATE-DEPENDENT path (`options[:state_dependent_si]`): the denominator is the DERIVED
#     working total ρ̂^w = ρ_d + Σ α_k ν_k of the state at n (design §4.2 "ρ̂_t = ρ_d + ρ_w,
#     recomputed each step from the slots"), for c_d as well as for every c_k:
#         c_d = ρ_d/ρ̂^w,  c_k = ν_k/ρ̂^w,  c_{·,z} by the quotient rule with ρ̂^w_z.
#     Every constituent then moves with the one velocity φ/ρ̂^w; the Helmholtz problem, φ = ρ_t w,
#     and the w and p recoveries are untouched (so the acoustic operator — and its ceiling — is
#     Stage 2's).
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
`Δ(ρ_d' + Σ α_k ν_k') = −Δτ ∂z φ` (the derived total's acoustic increment) to rounding.
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
    return _water_member_set(mode !== :none, mode === :all, options, physical_params, IS)
end

# The water members of one configuration, in the water sum's fixed order (vapor, cloud, rain,
# the three ice masses; then, with `allm`, the rain number and the nine ice moments, which
# carry weight 0). Shared by the SI slaving (`si_water_set`) and the derived water anchor
# (`water_anchor_set`), so the two see one set of weights and one summation order.
@inline function _water_member_set(on::Bool, allm::Bool, options, physical_params,
                                   IS::MCSlots)
    ice = ice_registered(IS)
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

Setup-time refusal of a slaved water set that does not share ONE fit operator with the anchor
(design §8 risk 2; plan Stage 2 test 5). The slaved increments are pointwise mish values that
each slot then refits on its OWN bases, so `Δ(ρ_d' + Σ α_k ν_k') = −Δτ ∂z φ` survives the refit
only if `ρ_d` and every slaved slot are fitted identically: the same wall conditions on every
side (`BCB`/`BCT`, `BCL`/`BCR`, and `BCU`/`BCD` on the 3-D box — the horizontal refit runs on
every geometry, not only under the horizontal SI), the same vertical AND horizontal filter
lengths (`l_q`, its `_k`/`_j` keys), the same `spline_filter`, and NO positivity bound (a bounded
fit is nonlinear, so the sum of the bounded refits is not the refit of the sum — an untransformed
`rho_r` under `numerics.positivity = auto` is the case that gets one). The sedimentation FLUX
columns' zero-flux lid is not compared: those fit the fluxes, not the slaved increments.

Under the `d2` or `r1t1x` scalar families the sedimenting slots stay natural at the ground while
the vapor and cloud take the family (and `r1t1x` gives ρ_d Neumann), so those are refused.
`options[:si_water_mixed_walls] = true` is the explicit opt-in to run and MEASURE such a
combination (design §4.4 caveat 1: "refuse that combination, or measure it"). A no-op under
`si_water_slots = :none` and for non-mc equation sets.
"""
function validate_si_water_slots(model)
    water = si_water_member_names(model)
    isempty(water) && return nothing
    get(model.options, :si_water_mixed_walls, false)::Bool && return nothing
    gp = model.grid_params
    mode = si_water_slots_mode(model.options)
    names = ["rho_d"; water]
    dflt(d, v) = haskey(d, v) ? d[v] : get(d, "default", nothing)
    lqh(v) = haskey(gp.l_q, v) ? gp.l_q[v] : get(gp.l_q, "default", nothing)
    lqv(v) = haskey(gp.l_q, v * "_k") ? gp.l_q[v * "_k"] : lqh(v)
    lqj(v) = haskey(gp.l_q, v * "_j") ? gp.l_q[v * "_j"] : lqh(v)
    aspects = (("ground BC (BCB)", v -> dflt(gp.BCB, v)),
               ("lid BC (BCT)", v -> dflt(gp.BCT, v)),
               ("inner/left BC (BCL)", v -> dflt(gp.BCL, v)),
               ("outer/right BC (BCR)", v -> dflt(gp.BCR, v)),
               ("y BC (BCU)", v -> dflt(gp.BCU, v)),
               ("y BC (BCD)", v -> dflt(gp.BCD, v)),
               ("vertical l_q", lqv),
               ("horizontal l_q", lqh),
               ("second horizontal l_q", lqj),
               ("spline_filter", v -> get(gp.spline_filter, v, nothing)))
    hint = "Use one family for every water slot and the anchor (the natural family: " *
           "[boundary_conditions] vertical_scalars = vertical_scalars_lid = \"natural\", " *
           "sedimentation_lid = \"natural\", no per-variable override or filter on these " *
           "slots), or set numerics.si_water_slots = \"none\", or set " *
           "options[:si_water_mixed_walls] = true to run and MEASURE it."
    for v in names
        if haskey(gp.positivity, v) && !isempty(gp.positivity[v])
            error("options[:si_water_slots] = :$(mode) slaves $(join(water, ", ")) to the " *
                  "acoustic mass flux, but \"$(v)\" carries a positivity bound " *
                  "($(gp.positivity[v])): a bounded fit is nonlinear, so the slaved increments " *
                  "would not sum through the refit to the anchor's (an untransformed rho_r " *
                  "under numerics.positivity = auto gets one). Transform the slot " *
                  "(physics.microphysics.rain_transform = \"bhyp\"), set numerics.positivity " *
                  "= \"off\", or " * hint)
        end
    end
    for (label, f) in aspects
        ref = names[1]; s0 = f(ref)
        for v in names[2:end]
            sv = f(v)
            isequal(sv, s0) && continue
            error("options[:si_water_slots] = :$(mode) slaves the water slots " *
                  "$(join(water, ", ")) to the acoustic mass flux, but \"$(v)\" and " *
                  "\"$(ref)\" do not share one fit operator ($(label): $(sv) vs $(s0)). The " *
                  "slaved increments are refit slot by slot, so a mixed family breaks the " *
                  "water/anchor identity Δ(ρ_d' + Σ α ν') = −Δτ ∂z φ at the refit (design risk 2: " *
                  "under d2/r1t1x families the sedimenting slots stay natural at the ground). " *
                  hint)
        end
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
difference added. Then Σ over ρ_d and 𝒲 of the add-backs is `∂z(ρ_t w)` of the derived total to
rounding, and
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
    rtb = mc_rho_tbar_ref(mtile); rdb = ref_rho_d(rs); rc = Springsteel.ref_rho_c(rs)
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
    elseif es == "moist_compressible_RRR"
        _siw_ldiv_g!(LDIV, MCCartesianRRR(), mtile, colstart, colend)
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

Note on φ^{n+1}: the `s_{k,z}` term takes it as `ρ̄_t · vnp1[w]` (the w plane applied), while
the solve's ρ_d plane used its own per-mode `phin1`. On XZ/axisym the two differ by rounding; on
RLR by the azimuthally truncated part of w* (the ragged per-n solve). The identity is unaffected
(Σ α_k s_{k,z} = 0 to rounding whatever φ is used); only the flux-form correction's accuracy is.
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

# ═════════════════════════════════════════════════════════════════════════════
# THE DERIVED WATER ANCHOR — Stages 3-4 of the ρ_t retirement (design §3.2-3.3, §6 Stages 3-4;
# plan §4 Stages 3-4, decision E9).
#
# Every reader of the total density reads
#
#     ρ_t  = ρ_d  + W ,   W  = Σ_k α_k ν_k      (the TOTAL control variables, ν_k' + ν̄_k)
#     ρ_t' = ρ_d' + W',   W' = Σ_k α_k ν_k'     (the slots as carried)
#     ρ̄_t^w = ρ̄_d + Σ_k α_k ν̄_k               (`mc_reference_diagnostics`)
#
# over the water-MASS members 𝒲 (vapor, cloud, rain, the three ice masses; α = 2 for a
# transformed slot, 1 for a density slot), in ONE fixed summation order — the order
# `_siw_W`/`_siw_Wbar` use, so the total is bitwise Stage 2's working total ρ̂^w and at rest
# (every slot 0.0) `ρ_t === ρ̄_t^w`. Stage 3 kept the prognostic ρ_t as a passive shadow in slot
# 3; Stage 4 deleted it, the vapor moving into slot 3. `options[:water_anchor]` is retired:
# `:derived` is the only behaviour (an explicit `:derived` is accepted, `:slot` refused).
# ═════════════════════════════════════════════════════════════════════════════

"""
    water_anchor_mode(options) -> Symbol

What the model reads as the total density: always `:derived` (`ρ_t = ρ_d + Σ_k α_k ν_k` over the
water-mass slots, its perturbation `ρ_d' + Σ α_k ν_k'` and the reference
`ρ̄_t^w = ρ̄_d + Σ α_k ν̄_k`, at every reader). `options[:water_anchor]` is RETIRED (Stage 4 of
the ρ_t retirement): an absent key or an explicit `:derived` is accepted, and `:slot` — the
prognostic ρ_t slot every reader took up to Stage 3 — is refused by name, as is anything else.
"""
@inline function water_anchor_mode(options)
    haskey(options, :water_anchor) || return :derived
    mode = options[:water_anchor]
    mode === :derived && return :derived
    mode === :slot && error("options[:water_anchor] = :slot is retired: the prognostic rho_t " *
        "slot it selected was deleted at Stage 4 of the rho_t retirement (slot 3 is the vapor; " *
        "rho_t is the derived rho_d + sum_k alpha_k nu_k). The :slot lineage is in git history " *
        "(Stage 3, development bf577b2). Remove the key.")
    error("options[:water_anchor] = $(repr(mode)) is not recognized: the key is retired (Stage " *
          "4 of the rho_t retirement) and the derived anchor is the only behaviour. Remove it.")
end

"""
    water_anchor_set(options, physical_params, IS::MCSlots) -> SIWaterSet

The water-MASS members of the derived anchor, with the weights and summation order of the SI
slaving's set (`_water_member_set`); never a number or habit moment. Allocation-free.
"""
@noinline water_anchor_set(options, physical_params, IS::MCSlots) =
    _water_member_set(true, false, options, physical_params, IS)

"""
    anchor_Wp(phys, g, l, ws) -> Float64

The water sum of the SLOTS as carried, `W' = Σ_k α_k phys[g, s_k, l]`, at row `g` and derivative
layer `l` (1 the value, `zslot` the vertical derivative, 2 the first horizontal one), in the
fixed member order; the first term is assigned (so a dry column's `W'` is the exact product its
first member gives). Every member's slot holds its perturbation `ν_k' = ν_k − ν̄_k` (the rain and
ice references are zero), so this is the perturbation of `W` and has no reference to add.
"""
@inline function anchor_Wp(phys, g::Int, l::Int, ws::SIWaterSet)
    W = 0.0; first = true
    @inbounds for j in 1:6
        s = ws.slot[j]; a = ws.alpha[j]
        (s == 0 || a == 0.0) && continue
        v = a * phys[g, s, l]
        if first
            W = v; first = false
        else
            W = W + v
        end
    end
    return W
end

"""
    anchor_rho_tp(rho_dp, Wp) -> Float64

The derived perturbation `ρ_t' = ρ_d' + W'` with the `−0.0` select of design §6: where the
water sum is an exact zero (a dry column, every slot `±0.0`) it is `ρ_d'` bitwise, so a dry run
reads the dry density's own bits.
"""
@inline anchor_rho_tp(rho_dp, Wp) = Wp == 0.0 ? rho_dp : rho_dp + Wp

"""
    _anchor_stage!(rho_t, rho_t_z, phys, colstart, zs, ws, rho_d, rho_d_z, nvb, nvbz, rcb, rcbz)

The driver staging of the derived anchor for one column: the totals `ρ_t = ρ_d + W`,
`ρ_{t,z} = ρ_{d,z} + W_z` (`_siw_W`, the members' totals from the slot views plus their
references — bitwise Stage 2's ρ̂^w). The perturbation `ρ_t' = ρ_d' + W'` is formed where its
one reader needs it ([`_anchor_w_forcing!`](@ref)). `@noinline` on plain arrays (the compile
cliff).
"""
@noinline function _anchor_stage!(rho_t::AbstractVector{Float64},
                                  rho_t_z::AbstractVector{Float64}, phys::Array{Float64,3},
                                  colstart::Int, zs::Int, ws::SIWaterSet,
                                  rho_d::AbstractVector{Float64},
                                  rho_d_z::AbstractVector{Float64}, nvb, nvbz, rcb, rcbz)
    @inbounds for i in eachindex(rho_t)
        g = colstart + i - 1
        W, Wz = _siw_W(phys, g, zs, i, ws, nvb, nvbz, rcb, rcbz)
        rho_t[i] = rho_d[i] + W
        rho_t_z[i] = rho_d_z[i] + Wz
    end
    return nothing
end

"""
    _anchor_w_forcing!(FORCING, rho_tp, phys, colstart, ws, rho_t, pp_z, rho_tbar, sd)

The w-equation's explicit buoyancy/PGF remainder on the derived perturbation
`ρ_t' = ρ_d' + W'` (`anchor_rho_tp`, the `−0.0` select: a dry column's ρ_t' is ρ_d' bitwise),
formed first into the scratch column `rho_tp` (the caller's free `KDIFF`): `−g ρ_t'/ρ_t`
(state-dependent SI) and `(−g ρ_t' − p'_z)/ρ_t + p'_z/ρ̄_t` (reference SI), with ρ_t the
derived total. `@noinline` on plain arrays (the compile cliff).
"""
@noinline function _anchor_w_forcing!(FORCING::Vector{Float64}, rho_tp::Vector{Float64},
                                      phys::Array{Float64,3}, colstart::Int, ws::SIWaterSet,
                                      rho_t::Vector{Float64}, pp_z, rho_tbar, sd::Bool)
    @inbounds for i in eachindex(rho_tp)
        g = colstart + i - 1
        rho_tp[i] = anchor_rho_tp(phys[g, 2, 1], anchor_Wp(phys, g, 1, ws))
    end
    # The driver's own `@turbo` expressions (the instruction selection, and so the bits on a
    # dry column where ρ_t' = ρ_d', are the pre-retirement prognostic-ρ_t path's).
    if sd
        @turbo FORCING .= @. (-gravity * rho_tp) / rho_t
    else
        @turbo FORCING .= @. (((-gravity * rho_tp) - pp_z) / rho_t) + (pp_z / rho_tbar)
    end
    return nothing
end

"""
    mc_rho_tbar_ref(mtile) -> Matrix{Float64}

The `(kDim, 3)` total-density reference every reader takes (value, ∂z, ∂zz): the derived
`ρ̄_t^w` and its gradient (the second derivative is Springsteel's fitted `ref_rho_t`'s, read by
nothing in the moist-compressible set). One `Matrix{Float64}`, so a `view(·, :, 1)` binding is
type-stable.
"""
@inline mc_rho_tbar_ref(mtile) = mtile.ebudget.rho_tbar_ref

"""
    anchor_W_value(A, g, i, ws, nvb, rcb) -> Float64

The water sum `W = Σ_k α_k ν_k` of the TOTALS at row `g` (level `i`) of a slot MATRIX `A`
(`var_np1`'s layout), in the fixed member order — `_siw_W`'s value for a state that has no
derivative layers (the post-acoustic star state, the census of `var_np1`).
"""
@inline function anchor_W_value(A::AbstractMatrix{Float64}, g::Int, i::Int, ws::SIWaterSet,
                                nvb, rcb)
    W = 0.0; first = true
    @inbounds for j in 1:6
        s = ws.slot[j]; a = ws.alpha[j]
        (s == 0 || a == 0.0) && continue
        r = ws.ref[j]
        v = A[g, s]
        v = r == 1 ? v + nvb[i] :
            r == 2 ? (ws.ctrans ? v + bhyp(rcb[i], ws.cmu) : v + rcb[i]) : v
        av = a * v
        if first
            W = av; first = false
        else
            W = W + av
        end
    end
    return W
end

"""
    _anchor_star!(rho_t_star, vnp1, colstart, ws, rho_d_star, nvb, rcb)

The derived total `ρ_d* + W*` of the post-acoustic star state (`diffusion_timestep_mc`), over
the slot-based column. `@noinline`, plain arrays.
"""
@noinline function _anchor_star!(rho_t_star::AbstractVector{Float64}, vnp1::Matrix{Float64},
                                 colstart::Int, ws::SIWaterSet,
                                 rho_d_star::AbstractVector{Float64}, nvb, rcb)
    @inbounds for i in eachindex(rho_t_star)
        rho_t_star[i] = rho_d_star[i] + anchor_W_value(vnp1, colstart + i - 1, i, ws, nvb, rcb)
    end
    return nothing
end

"""
    validate_water_anchor(model)

Setup-time refusals of what the derived water anchor — the only one since Stage 4 of the ρ_t
retirement — cannot run, each naming the reason:

- `options[:water_anchor]` other than `:derived` ([`water_anchor_mode`](@ref): `:slot` is
  retired with the prognostic ρ_t slot);
- `physical_params[:Kvdiff_water] > 0`: the implicit water diffusion solved `ρ_t' − ρ_d'` (the
  deleted slot 3) as the total water and implied the vapor from it (design §6 Stages 3-4: the
  ρ_w solve and its matrices are retired);
- `options[:water_lobe_attribution]`, `options[:vapor_nudge_partner]`,
  `physical_params[:tau_rho_v_rec]`: they configured the vapor reconciliation nudge and its
  residual, deleted with the prognostic ρ_t (the vapor is the ν-space closure of every
  exchange instead);
- an explicit `true` for the four ice-anchor legs (`ice_anchor_source/_floor/_flux/_rates`,
  deleted at Stage 5: under the derived anchor the headroom read through `ρ_w` is
  `α_v ν_v + X_cond`, which can be negative at a vapor lobe beside ice, so the headroom test
  had become a vapor-negativity device; the legs were made inert at setup (de822e5/a7d25dc),
  then deleted with the census reading PTS = REMOVED = 0 on the Stage 3 RCE arms) and the two
  MYNN slot-anchor water devices; `physical_params[:tau_ice_anchor]` (the removal's timescale);
- `options[:clamp_water] = true` (its floor balanced the partition against slot 3), and
  `options[:mynn_water_carry] = :fixed_T`;
- `options[:water_budget_trace] > 0` with any water transform, two-moment rain or ice (the
  driver's own reasons, raised here at setup: the budget rows are density-unit liquid
  channels).

A no-op for non-mc equation sets.
"""
function validate_water_anchor(model)
    uses_pressure_reference(model.equation_set) || return nothing
    o = model.options
    water_anchor_mode(o)
    # The energy bisection's forensic options.
    get(o, :energy_budget_functional, :model) in (:model, :derived) ||
        error("options[:energy_budget_functional] must be :model (default) or :derived")
    get(o, :refit_partner_form, :defect) in (:defect, :recovered) ||
        error("options[:refit_partner_form] must be :defect (default) or :recovered")
    pp = model.physical_params; opts = model.options
    why = "the derived water anchor (rho_t = rho_d + sum_k alpha_k nu_k; the prognostic rho_t " *
          "slot was deleted at Stage 4 of the rho_t retirement)"
    get(pp, :Kvdiff_water, 0.0) > 0.0 && error("physical_params[:Kvdiff_water] > 0 is " *
        "refused under " * why * ": the implicit water diffusion solved the deleted rho_t slot " *
        "as the total water. Set Kvdiff_water = 0 (the boundary layer mixes the water).")
    for (k, what) in ((:water_lobe_attribution, "the vapor residual of the prognostic rho_t"),
                      (:vapor_nudge_partner, "the vapor reconciliation nudge"))
        haskey(opts, k) && opts[k] !== false && error("options[:$(k)] configures $(what), " *
            "which is deleted under " * why * ". Remove the key.")
    end
    haskey(pp, :tau_rho_v_rec) && error("physical_params[:tau_rho_v_rec] configures the " *
        "vapor reconciliation nudge, which is deleted under " * why * ". Remove the key.")
    haskey(pp, :tau_ice_anchor) && error("physical_params[:tau_ice_anchor] configured the " *
        "ice-anchor removal source, which was deleted with the other three ice-anchor legs at " *
        "Stage 5 of the rho_t retirement (under " * why * " the headroom test had become a " *
        "vapor-negativity device). Remove the key.")
    # Devices the derived anchor SUBSUMES (review D1 and the BL sub-check), deleted: an
    # explicit `true` is refused rather than silently bypassed (`false` is harmless).
    for (k, what) in ((:ice_anchor_source, "the ice-anchor removal source"),
                      (:ice_anchor_floor, "the ice-anchor reader cap"),
                      (:ice_anchor_flux, "the ice-anchor sedimentation share"),
                      (:ice_anchor_rates, "the ice-anchor rate share"),
                      (:mynn_water_closure, "the MYNN total-water residual closure"),
                      (:mynn_exchange_realization, "the MYNN clip-gap (lobe) partner"))
        get(opts, k, nothing) === true && error("options[:$(k)] = true configures $(what), " *
            "which is retired under " * why * " (the ice-anchor headroom read through rho_w " *
            "is alpha_v nu_v + X_cond, a vapor-negativity test, so the four legs were deleted " *
            "at Stage 5; every MYNN water leg is a nu leg realized on its own recovered rate). " *
            "Remove the key.")
    end
    if get(opts, :water_budget_trace, 0) > 0
        (condensate_transform_mode(opts) === :none && rain_transform_mode(opts) === :none &&
         vapor_transform_mode(opts) === :none && rain_moments(opts) == 1 &&
         ice_microphysics(opts) === :none) || error("options[:water_budget_trace] needs the " *
            "density lineage (no water transform, single-moment rain, no ice): its budget " *
            "rows attribute density-unit tendencies of the three liquid channels.")
    end
    get(opts, :clamp_water, false) === true && error("options[:clamp_water] balanced the " *
        "partition against the prognostic rho_t (its rule 2), which is deleted under " * why *
        ". Remove the key; clamp_water! still measures the negative water.")
    get(opts, :mynn_water_carry, :flux) === :fixed_T && error("options[:mynn_water_carry] " *
        "= :fixed_T (the local fixed-temperature water map) is not implemented under " * why *
        ": the boundary layer's water legs are nu legs whose energy carry is the flux-form " *
        "column regrouped by species. Use the default :flux carry.")
    return nothing
end

# ═════════════════════════════════════════════════════════════════════════════
# 3. WATER EXCHANGE WITH THE VAPOR AS ν-SPACE CLOSURE — Stage 3b (design §5.2, plan E4/E5/E8)
#
# Every process that moves water between species applies its non-vapor ν increments as before
# and the vapor receives
#
#     Δν_v = −(Σ_{k≠v} α_k Δν_k)/α_v                         (`water_exchange_dv`)
#
# in ONE fixed order (cloud, rain, the three ice masses), so ρ_w = Σ α_k ν_k is invariant under
# every exchange to rounding (exactly: with all α = 2 the doublings and the halving are exact,
# and α_v Δν_v is the negated sum bit for bit). Exchange sites, all through `water_exchange_dv`
# (four call sites): the ETD step-mean increments (`_etd_closure!`), the multistep exchange
# sources (AUTO_COLL, the ice back-reactions ICE_C/ICE_R, the twelve ice SRC_* including the
# population reconcile) through `water_exchange_multistep_rate` on the ice branch of
# `_water_exchange_multistep!` and `water_exchange_dv` directly on its warm (autoconversion
# only) branch, and the paired Stage L (`_lobe_relax_paired!`). The sedimentation knee closure
# (`sedimentation_knee_closure` on the ice branch, the same expression for rain alone written
# inline on the warm branch) is the one other water term in
# the vapor slot: not an exchange between species but the fall's ν/ρ mismatch, closed so that
# ρ_w loses exactly the fitted flux divergence. Every other write to the vapor slot is a
# transport-class term (advection and −J_v ρ_v ∇·v, the acoustic slaving, the boundary-layer ν
# leg with the surface flux, horizontal mixing), linear in ν, not an exchange.
# ═════════════════════════════════════════════════════════════════════════════

"""
    water_exchange_dv(av, ac, ar, ai, dc, dr, d1, d2, d3) -> Δν_v

THE ONE CLOSURE, and the single entry point for the vapor's share of every water exchange (four
call sites: the ETD increments, the multistep sources on both branches of
`_water_exchange_multistep!`, the paired Stage L; see the section header): the vapor's
ν increment of a water exchange whose non-vapor ν increments are `dc` (cloud), `dr` (rain),
`d1..d3` (the ice masses), weights `α` (`water_anchor_set`):
`−((((α_c dc + α_r dr) + α_i d1) + α_i d2) + α_i d3)/α_v`. With every `α = 2` it is bitwise
`−((((dc + dr) + d1) + d2) + d3)` (doubling and halving are exact). An all-zero exchange gives
`−0.0`, which every caller skips, so a quiet point is untouched bitwise.
"""
@inline water_exchange_dv(av, ac, ar, ai, dc, dr, d1, d2, d3) =
    -(((((ac * dc) + (ar * dr)) + (ai * d1)) + (ai * d2)) + (ai * d3)) / av

"""
    water_exchange_multistep_rate(av, ac, ar, ai, Jc, Jr, J1, J2, J3, ac_rate, ice_c, ice_r,
                                  s1, s2, s3) -> ν̇_v

The vapor's ν RATE closing the multistep exchange sources (the ones that stay on AB3): the
cloud receives `J_c(ICE_C − AUTO_COLL)`, the rain `J_r(ICE_R + AUTO_COLL)`, each ice mass
`J_k SRC_k` (the ISHMAEL process sources with the population reconcile
folded in), and the vapor `water_exchange_dv` of those ν rates. Autoconversion alone gives the
vapor `(J_c − J_r)·AUTO_COLL`: zero above both knees (design §5.2).
"""
@inline water_exchange_multistep_rate(av, ac, ar, ai, Jc, Jr, J1, J2, J3, acr, ice_c, ice_r,
                                      s1, s2, s3) =
    water_exchange_dv(av, ac, ar, ai, Jc * (ice_c - acr), Jr * (ice_r + acr), J1 * s1,
                      J2 * s2, J3 * s3)

"""
    sedimentation_knee_closure(av, ar, ai, Jr, J1, J2, J3, Frz, F1z, F2z, F3z) -> ν̇_v

Plan E8 (design condition d): the falling species' slots receive `−J_k F_{k,z}`, so ρ_w would
lose `α_k J_k F_{k,z}` (`2J_k F_z`, not `F_z`, at a knee); the vapor takes the remainder,
`ν̇_v = Σ_k (α_k J_k − 1) F_{k,z}/α_v` (`(J_k − ½) F_{k,z}` with every α = 2), so ρ_w loses exactly
the fitted flux divergence `Σ_k F_{k,z}` and the rainfall double entry closes. An exact zero
where nothing falls (or above every knee, to the rounding of `2·½ − 1`).
"""
@inline sedimentation_knee_closure(av, ar, ai, Jr, J1, J2, J3, Frz, F1z, F2z, F3z) =
    (((((ar * Jr) - 1.0) * Frz) + (((ai * J1) - 1.0) * F1z)) +
     ((((ai * J2) - 1.0) * F2z) + (((ai * J3) - 1.0) * F3z))) / av

"""
    _water_exchange_multistep!(expdot, phys, S, colstart, rv_i, options, physical_params, IS,
                               ice, precip, nvb, vtr, vmu, ts)

Under the derived anchor, add to the vapor slot's tendency the closure of the multistep
exchange (`water_exchange_multistep_rate`) and of the sedimentation (`sedimentation_knee_closure`),
and book their energy ON THE REALIZED RECOVERED RATES (review B2, the sedimentation knee's form
for all of them): with `ν̇_c = J_c(ICE_C − AUTO_COLL)`, `ν̇_r = J_r(ICE_R + AUTO_COLL)`,
`ν̇_k = J_k SRC_k` and `ν̇_v` the two closures, each slot's recovered rate is
`ρ̇_k = [recover(ν_k + Δt ν̇_k) − recover(ν_k)]/Δt`, and the partner is

    recovered_change_partner(ρ̇_v, ρ̇_l − (ICE_C + ICE_R), ρ̇_i + (ICE_C + ICE_R), …)

— the whole exchange's energy less the liquid→ice part `FRZ_NET = −(ICE_C + ICE_R)` that the
process tendency already carries as `L_f` in `p` (the primitive is linear, so the remainder is
the same primitive with the freezing taken out of the two condensate arguments). WHY NOT THE
REQUESTED MASS: autoconversion at a rain knee (`J_r → 1`, `J_c = ½`) moves the vapor's ν by
`(J_c − J_r) AC` and its recovered density by ~`−AC` while the liquid's recovered total is
unchanged; nucleation or splinters into empty ice (`J_k = 1`) move the recovered vapor by `−2s`
for `s` of ice. The requested-mass form booked neither the first nor the second `−s`; the
realized form holds `R` to the second order of the rates (the E4 tolerance). Only the vapor
and the condensates' exchange rates enter — the sedimentation's own fall moves its energy
through the flux. `Q_ss` takes the primitive's share (`_qss_share!`). `@noinline`, the scratch
and plain arrays.
"""
@noinline function _water_exchange_multistep!(expdot::Matrix{Float64}, phys::Array{Float64,3},
                                              S, colstart::Int, rv_i::Int, options,
                                              physical_params, IS::MCSlots,
                                              ice::Bool, precip::Bool,
                                              nvb::Vector{Float64}, vtr::Symbol,
                                              vmu::Float64, ts::Float64)
    ws = water_anchor_set(options, physical_params, IS)
    av = ws.alpha[1]; ac = ws.alpha[2]; ar = ws.alpha[3]; ai = ws.alpha[4]
    ctr = condensate_transform_mode(options); rtr = rain_transform_mode(options)
    cmu = Float64(get(physical_params, :condensate_mu, 1.0e-7))
    rmu = Float64(get(physical_params, :rain_mu, 1.0e-7))
    itr = ice ? ice_transform_mode(options) : :none
    imu = ice ? ice_mu(physical_params, 1) : 1.0
    Jc = S.Jc; Jr = S.Jr; AC = S.AUTO_COLL; Frz = S.Fr_z
    @inbounds for i in eachindex(Jc)
        g = colstart + i - 1
        acr = precip ? AC[i] : 0.0
        if ice
            icc = S.ICE_C[i]; icr = S.ICE_R[i]
            s1 = S.SRC_i1q[i]; s2 = S.SRC_i2q[i]; s3 = S.SRC_i3q[i]
            x = water_exchange_multistep_rate(av, ac, ar, ai, Jc[i], Jr[i], S.J_i1q[i],
                    S.J_i2q[i], S.J_i3q[i], acr, icc, icr, s1, s2, s3)
            xs = sedimentation_knee_closure(av, ar, ai, Jr[i], S.J_i1q[i], S.J_i2q[i],
                    S.J_i3q[i], Frz[i], S.F_i1q_z[i], S.F_i2q_z[i], S.F_i3q_z[i])
            nc = Jc[i] * (icc - acr); nr = Jr[i] * (icr + acr)
            n1 = S.J_i1q[i] * s1; n2 = S.J_i2q[i] * s2; n3 = S.J_i3q[i] * s3
            frz = icc + icr
        else
            icc = 0.0; icr = 0.0
            x = acr == 0.0 ? 0.0 :
                water_exchange_dv(av, ac, ar, ai, -(Jc[i] * acr), Jr[i] * acr, 0.0, 0.0, 0.0)
            xs = Frz[i] == 0.0 ? 0.0 : (((ar * Jr[i]) - 1.0) * Frz[i]) / av
            nc = -(Jc[i] * acr); nr = Jr[i] * acr
            n1 = 0.0; n2 = 0.0; n3 = 0.0
            frz = 0.0
        end
        x != 0.0 && (expdot[g, rv_i] += x)
        xs != 0.0 && (expdot[g, rv_i] += xs)
        xv = x + xs
        # the realized recovered rates (exact zeros where a slot's rate is zero)
        rv = 0.0
        if xv != 0.0
            nu = phys[g, rv_i, 1] + nvb[i]
            rv = (recover_rho_v(ts * xv, nu, vtr, vmu) - recover_rho_v(0.0, nu, vtr, vmu)) / ts
        end
        rl = 0.0
        nc != 0.0 && (rl += _rec_rate(S.nu_c[i], nc, ctr, cmu, ts))
        nr != 0.0 && (rl += _rec_rate(S.nu_r[i], nr, rtr, rmu, ts))
        ri = 0.0
        if ice
            n1 != 0.0 && (ri += _rec_rate(S.nu_i1q[i], n1, itr, imu, ts))
            n2 != 0.0 && (ri += _rec_rate(S.nu_i2q[i], n2, itr, imu, ts))
            n3 != 0.0 && (ri += _rec_rate(S.nu_i3q[i], n3, itr, imu, ts))
        end
        # less the freezing FRZ_NET = -(ICE_C + ICE_R) already in p as L_f
        if frz != 0.0
            rl -= frz
            ri += frz
        end
        (rv == 0.0 && rl == 0.0 && ri == 0.0) && continue
        dp, dq = recovered_change_partner(rv, rl, ri, S.Tk[i], S.rho_d[i], S.R_m[i],
                                          S.C_vt[i], S.Lv[i], ice ? L_s(S.Tk[i]) : 0.0,
                                          S.drvs_dT[i], S.drvs_dp[i])
        expdot[g, 1] += dp
        _qss_share!(expdot, S, g, i, dq, ts)
    end
    return nothing
end

# The realized recovered RATE of a transformed slot at total `nu` under a ν rate `r` over `ts`
# (`r` itself under `:none`, where the slot is the density).
@inline _rec_rate(nu, r, tr::Symbol, mu, ts) =
    tr === :none ? r : (recover_total(nu + (ts * r), tr, mu) - recover_total(nu, tr, mu)) / ts

# A partner's Q_ss share: where the stiff propagator owns slot 7 this step (`λ ≠ 0`, its
# predictor is discarded and N^n was already consumed) as a direct increment of the
# propagated value, otherwise as a tendency of the multistep (which keeps the predictor).
@inline function _qss_share!(expdot, S, g::Int, i::Int, dq::Float64, ts::Float64)
    @inbounds if S.etd_lam[i] != 0.0
        S.etd_qnp1[i] += ts * dq
    else
        expdot[g, 7] += dq
    end
    return nothing
end

"""
    _lobe_relax_paired!(vnp1, phys, S, colstart, rv_i, i1, i2, i3, ice, ws, nvb, vtr, vmu)
        -> (Δν_c, Δν_r, Δν_i1, Δν_i2, Δν_i3) per point summed with `S`-independent weights by the caller

THE PAIRED STAGE L (plan E5, design risk 4): under the derived anchor the lobe relaxation's
capped increments `Δν_k = lobe_relaxation_cap(−ν_k(1 − e^{−Δt/τ}), ν*)` on the closed channels
are a water EXCHANGE with the vapor: `Δν_v = water_exchange_dv(…)`, so ρ_w is untouched. The
recovered condensate is 0 before and after (the cap keeps ν ≤ 0), the recovered vapor falls by
`ahyp(ν_v) − ahyp(ν_v + Δν_v)`, and the partner is the primitive of that vapor change ALONE — no
latent heat (`recovered_change_partner(rec_v, 0, 0, …)` into `p` and `Q_ss`). Returns the column
increments it applied per point through `out` (the caller's census). `@noinline`.
"""
@noinline function _lobe_relax_paired!(vnp1::Matrix{Float64}, phys::Array{Float64,3}, S,
                                       colstart::Int, rv_i::Int, i1::Int, i2::Int, i3::Int,
                                       ice::Bool, ws::SIWaterSet, nvb::Vector{Float64},
                                       vtr::Symbol, vmu::Float64, wz::Vector{Float64},
                                       cen::Bool)
    av = ws.alpha[1]; ac = ws.alpha[2]; ar = ws.alpha[3]; ai = ws.alpha[4]
    dlc = S.etd_dlc; dlr = S.etd_dlr; nuc = S.nu_c
    lc = 0.0; lr = 0.0; l1 = 0.0; l2 = 0.0; l3 = 0.0
    @inbounds for i in eachindex(dlc)
        g = colstart + i - 1
        dc = 0.0; dr = 0.0; d1 = 0.0; d2 = 0.0; d3 = 0.0
        if dlc[i] != 0.0
            dc = lobe_relaxation_cap(dlc[i], vnp1[g, 9] + (nuc[i] - phys[g, 9, 1]))
            dc != 0.0 && (vnp1[g, 9] += dc)
        end
        if dlr[i] != 0.0
            dr = lobe_relaxation_cap(dlr[i], vnp1[g, 8])
            dr != 0.0 && (vnp1[g, 8] += dr)
        end
        if ice
            if S.etd_dli1[i] != 0.0
                d1 = lobe_relaxation_cap(S.etd_dli1[i], vnp1[g, i1])
                d1 != 0.0 && (vnp1[g, i1] += d1)
            end
            if S.etd_dli2[i] != 0.0
                d2 = lobe_relaxation_cap(S.etd_dli2[i], vnp1[g, i2])
                d2 != 0.0 && (vnp1[g, i2] += d2)
            end
            if S.etd_dli3[i] != 0.0
                d3 = lobe_relaxation_cap(S.etd_dli3[i], vnp1[g, i3])
                d3 != 0.0 && (vnp1[g, i3] += d3)
            end
        end
        if cen
            w = wz[i]
            lc += w * dc; lr += w * dr; l1 += w * d1; l2 += w * d2; l3 += w * d3
        end
        (dc == 0.0 && dr == 0.0 && d1 == 0.0 && d2 == 0.0 && d3 == 0.0) && continue
        dv = water_exchange_dv(av, ac, ar, ai, dc, dr, d1, d2, d3)
        nu0 = vnp1[g, rv_i] + nvb[i]
        vnp1[g, rv_i] += dv
        rec = recover_rho_v(vnp1[g, rv_i], nvb[i], vtr, vmu) - recover_rho_v(0.0, nu0, vtr, vmu)
        if rec != 0.0
            dp, dq = recovered_change_partner(rec, 0.0, 0.0, S.Tk[i], S.rho_d[i], S.R_m[i],
                                              S.C_vt[i], S.Lv[i], 0.0, S.drvs_dT[i],
                                              S.drvs_dp[i])
            vnp1[g, 1] += dp
            vnp1[g, 7] += dq
        end
    end
    return (lc, lr, l1, l2, l3)
end


"""
    _etd_closure!(dv, d9, d8, d1, d2, d3, ice, options, physical_params, IS)

The ETD's vapor increment under the derived anchor: `dv[i] = water_exchange_dv(…, d9, d8,
d1, d2, d3)` at every point of the column (the cloud, rain and ice-mass step-mean increments
already formed), overwriting the density-space `J_v·Δt·vsrc` the pre-compute wrote. `@noinline`.
"""
@noinline function _etd_closure!(dv::Vector{Float64}, d9::Vector{Float64},
                                 d8::Vector{Float64}, d1::Vector{Float64},
                                 d2::Vector{Float64}, d3::Vector{Float64}, ice::Bool,
                                 options, physical_params, IS::MCSlots)
    ws = water_anchor_set(options, physical_params, IS)
    av = ws.alpha[1]; ac = ws.alpha[2]; ar = ws.alpha[3]; ai = ws.alpha[4]
    @inbounds for i in eachindex(dv)
        dv[i] = water_exchange_dv(av, ac, ar, ai, d9[i], d8[i], ice ? d1[i] : 0.0,
                                  ice ? d2[i] : 0.0, ice ? d3[i] : 0.0)
    end
    return nothing
end


"""
    _water_exchange_stage!(mtile, colstart, colend, t)

The derived anchor's per-column work on the explicit side, called from `explicit_timestep`
(after `mc_driver!` has assembled every tendency and the ETD increments, before the AB3
combination and before `var_np1` is overwritten), reading the tile and the thread's scratch,
which still hold the column the driver assembled:

1. the multistep exchange and sedimentation closures into the vapor's tendency
   (`_water_exchange_multistep!`);
2. the ETD closure `etd_dv = water_exchange_dv(…)` (`_etd_closure!`);
3. the refit partner as the affine-defect booking (`_refit_partner_defect!`, plan E2/E3) —
   `var_np1` still holds the pre-refit state and the slots the post-refit one — with the energy
   budget's partner marks and the `MC_REFIT_BOOKED` census.

WHY HERE: the multistep closure's call site in `mc_driver!` took the first-call compile of the
axisymmetric Louis tile from ~310 s to more than 40 min (the LLVM cliff of
reference/HANDOFF_DRIVER_COMPILE_CLIFF.md, measured 2026-10-21); the `_siw_stage!` precedent.
"""
@noinline function _water_exchange_stage!(mtile::ModelTile, colstart::Int, colend::Int, t::Int)
    model = mtile.model
    opts = model.options; pp = model.physical_params
    IS = mtile.mc_slots
    S = @inbounds mtile.mc_scratch[Threads.threadid()]
    ice = ice_registered(IS)
    precip = get(opts, :precipitation, false)::Bool
    vtr = vapor_transform_mode(opts); vmu = vapor_mu(pp)
    nvb = mc_nu_vbar(mtile)
    phys = mtile.tile.physical
    rv_i = IS.rho_v
    eb = mtile.ebudget
    # (the energy budget isolates the closures as their own explicit row, `explicit:wexch`)
    eb.active && eb_snap!(eb, mtile.expdot_n, colstart)
    _water_exchange_multistep!(mtile.expdot_n, phys, S, colstart, rv_i, opts, pp, IS, ice,
                               precip, nvb, vtr, vmu, model.ts)
    eb.active && eb_take!(eb, EB_OP_WEXCH, mtile.expdot_n, colstart)
    _etd_closure!(S.etd_dv, S.etd_d9, S.etd_d8, S.etd_di1, S.etd_di2, S.etd_di3, ice,
                  opts, pp, IS)
    ctr = condensate_transform_mode(opts); rtr = rain_transform_mode(opts)
    itr = ice ? ice_transform_mode(opts) : :none
    if t > 1 && get(opts, :refit_partner, REFIT_PARTNER_DEFAULT)::Bool &&
       (ctr !== :none || rtr !== :none || itr !== :none || vtr !== :none)
        eb.active && eb_partner_snap!(eb, S.etd_d1, S.etd_dv)
        itr_on = itr !== :none
        booked = _refit_partner_defect!(S.etd_d1, S.etd_qnp1, S.etd_lam, mtile.var_np1, phys,
            S.nu_c, S.nu_r, S.nu_i1q, S.nu_i2q, S.nu_i3q, S.Tk, S.R_m, S.C_vt, S.Lv, colstart,
            ctr, Float64(get(pp, :condensate_mu, 1.0e-7)), rtr, Float64(get(pp, :rain_mu, 1.0e-7)),
            itr, ice ? ice_mu(pp, 1) : 1.0,
            itr_on ? IS.i1_q : 0, itr_on ? IS.i2_q : 0, itr_on ? IS.i3_q : 0, eb.wz,
            S.rho_d, S.drvs_dT, S.drvs_dp, rv_i, nvb, vtr, vmu,
            get(opts, :refit_partner_form, :defect)::Symbol === :recovered,
            get(opts, :refit_partner_vapor, true)::Bool)
        eb.active && eb_partner_take!(eb, S.etd_d1, S.etd_dv)
        st = mtile.mc_water_stats
        if size(st, 2) > 0 && length(eb.wz) == colend - colstart + 1
            @inbounds st[MC_REFIT_BOOKED, Threads.threadid()] += eb.wxn[_eb_col(eb, colstart)] * booked
        end
    end
    return nothing
end
