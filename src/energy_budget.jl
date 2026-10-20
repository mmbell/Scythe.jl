# ── The per-operator energy and partition-gap budget (options[:energy_budget]) ──────────
#
# A permanent, option-gated re-creation of the closed per-stage budget that drove the 2026-10
# energy partners (the scratchpad "ebudget" instrument, lost with the session scratchpad; see
# reference/HANDOFF_2026-10-15_ENERGY_WATER_CONSISTENCY.md §7-8). It books, per column and per
# step, the change every stage of the column step makes to the two bookkeeping residuals of
# the moist-compressible set,
#
#     R = E_t − E_implied(T_eos, ρ_d, ρ_v, ρ_liq, ρ_ice, ρ_t·geo)        [J/m³]
#     X = ρ_t − ρ_d − ρ_v − ρ_liq − ρ_ice                                  [kg/m³]
#
# (`implied_total_energy`, the netCDF `E_residual`), integrated over the column with the
# mish's Gauss weights. Each stage is a FINITE DIFFERENCE of the column integral over the
# increment that stage wrote, the stages are taken in the order the column step applies them,
# and so their sum telescopes to the actual change of the column integral between two output
# times — to rounding. That sum is the instrument's own test (the `closure` row).
#
# LIMITS, stated rather than hidden. Under NESTING the exchange between steps (the parent's
# collar injection, `refit_reference_after_exchange!`'s re-take of `var_np1`) is not a marked
# stage: what it moves lands in the `refit` row, and what it writes into `var_np1` after the
# last mark shows in the `closure` row. `options[:exact_si]` is refused (its column step ends in
# phase B, outside the marked driver). Explicit operators that are assembled inline with the
# dynamics (advection, the work terms, horizontal heat diffusion, the explicit microphysics and
# sedimentation, the cylinders' `v` equation including its horizontal diffusion) are not
# isolated: they are the `explicit:dyn` row.
#
# This file is a LEAF in the include order (before semiimplicit.jl, because `ModelTile`
# carries an `EnergyBudget` concretely, like `SurfaceDiag`); the functions it calls from
# moist_compressible.jl are resolved at call time.

"""
The stages of [`EnergyBudget`](@ref), in the order of the column step. Rows with a `:` are
SUB-ROWS: a split of the stage named before the colon, not stages of their own; the top-level
rows ([`EB_TOP`](@ref)) are the ones that telescope.

| row | what it books |
|---|---|
| `refit` | the per-step refit between the column step's end and the next step's state `n`: the spline fit (`calcTendency`), the patch merge and halo, any horizontal-SI sweep, and the next `tileTransform!` |
| `explicit` | the AB3 (AB2/Euler at startup) explicit advance, `explicit_timestep` |
| `explicit:dyn` | everything in the explicit tendency not isolated below (advection, compressibility and work terms, buoyancy, the PGF remainder, horizontal heat diffusion, explicit microphysics, sedimentation, Coriolis/curvature, the `v` equation) |
| `explicit:rad` | radiative heating (`q_lw + sw_scale q_sw` into `p` and `E_t`) |
| `explicit:hmom` | horizontal momentum diffusion of `u`, `w` and its `FRIC_KE` (the cylinders' `v` diffusion is in `dyn`) |
| `explicit:sponge` | the Rayleigh sponge (`mc_sponge!`) |
| `explicit:bl` | the boundary layer (MYNN or Louis), less its surface exchange |
| `explicit:sfc` | the surface exchange: drag, sensible and moisture fluxes on the surface-layer profile |
| `explicit:vrec` | the vapor reconciliation nudge and its pressure partner |
| `explicit:hwater` | horizontal water mixing (`Khdiff_water`) |
| `explicit:nonlin` | `explicit` minus the eight rows above: the cross terms of a nonlinear R |
| `etd` | the ETD phase-change increments landing (`relaxation_adjustment_qss!`), refit partner included |
| `etd:refit_partner` | the refit partner's own increments (its `p` and vapor folds), at the post-explicit state |
| `etd:phase` | `etd` minus `etd:refit_partner` |
| `exch_real` | the exchange realization (`_etd_exchange_realization!`) |
| `lobe_relax` | the lobe relaxation (Stage L) |
| `hsi_hist` | the horizontal acoustic AI2* history levels (`horizontal_si_history!`) |
| `acoustic_si` | the vertical acoustic semi-implicit solve (`semiimplicit_adjustment_p`) |
| `vdiff` | the implicit vertical diffusion (`diffusion_timestep_mc`) |
| `clamp` | `clamp_water!` |

Each `explicit:<op>` row is `R(xⁿ + δ_op) − R(xⁿ)` with `δ_op` the operator's own AB3-weighted
increment (its tendency history kept here, rotated as `explicit_timestep` rotates `expdot`), so
the rows are evaluated with the state-`n` coefficients and do not telescope; `dyn`'s increment
is the actual increment minus the isolated ones, so the eight increments sum to the actual one
and `nonlin` is the interaction of a nonlinear `R` only.
"""
const EB_STAGES = ("refit", "explicit", "explicit:dyn", "explicit:rad", "explicit:hmom",
                   "explicit:sponge", "explicit:bl", "explicit:sfc", "explicit:vrec",
                   "explicit:hwater", "explicit:nonlin", "etd", "etd:refit_partner",
                   "etd:phase", "exch_real", "lobe_relax", "hsi_hist", "acoustic_si",
                   "vdiff", "clamp")
const EB_REFIT = 1
const EB_EXPLICIT = 2
const EB_X_DYN = 3
const EB_X_NONLIN = 11
const EB_ETD = 12
const EB_ETD_PARTNER = 13
const EB_ETD_PHASE = 14
const EB_EXCH = 15
const EB_LOBE = 16
const EB_HSI = 17
const EB_SI = 18
const EB_VDIFF = 19
const EB_CLAMP = 20
const EB_NSTAGE = length(EB_STAGES)
"The top-level rows of [`EB_STAGES`](@ref): the ones that telescope to the actual change."
const EB_TOP = (EB_REFIT, EB_EXPLICIT, EB_ETD, EB_EXCH, EB_LOBE, EB_HSI, EB_SI, EB_VDIFF,
                EB_CLAMP)

# The isolated explicit operators: their index in the tendency history and their row.
const EB_OP_RAD = 1
const EB_OP_HMOM = 2
const EB_OP_SPONGE = 3
const EB_OP_BL = 4
const EB_OP_SFC = 5
const EB_OP_VREC = 6
const EB_OP_HWATER = 7
const EB_NOP = 7
"Row of [`EB_STAGES`](@ref) for each isolated explicit operator `EB_OP_*`."
const EB_OP_ROW = (4, 5, 6, 7, 8, 9, 10)

"""
    EnergyBudget

The tile's budget state: the column geometry and reference profiles the residuals are
evaluated with (ALWAYS allocated for a moist-compressible tile with a reference, because the
`MC_REFIT_DR`/`MC_LOBE_*`/`MC_XGAP_*` census rows read them too), and — only when
`options[:energy_budget]` is on (`active`) — the per-column accumulators, the isolated
operators' tendency histories and the per-thread work columns.

A `mutable struct` on purpose: the hooks that receive it are called from the middle of
`mc_driver!`, and what costs LLVM time there is handing a large IMMUTABLE struct (the tile,
the grid, the view NamedTuples) to an opaque call (the `mc_water_mixing!` measurement). A
mutable struct crosses as one pointer, like the `Matrix` arguments beside it. It is held by
the tile (`mtile.ebudget`) and adds nothing to `MC_SCRATCH_SLOTS`.

`rs` lists the slots `R` and `X` read (`p, ρ_d, ρ_t, u, w, E_t, ρ_v, [v], cloud, rain, [the
three ice masses]`); the tendency histories and the work columns carry only those, and
`j*` is each role's position in `rs`.
"""
mutable struct EnergyBudget
    const active::Bool
    const kDim::Int
    const nvars::Int
    # Column quadrature: Gauss weights of the vertical mish [m], and per column the horizontal
    # weight (Gauss weight of the i-mish x the radius on the cylinders) — RAW (`wx`) for the
    # domain mean across tiles and normalized over this tile (`wxn`) for the census.
    const wz::Vector{Float64}
    const z::Vector{Float64}
    const wx::Vector{Float64}
    const wxn::Vector{Float64}
    # Reference profiles of the perturbation slots (kDim).
    const pbar::Vector{Float64}
    const rho_dbar::Vector{Float64}
    const rho_tbar::Vector{Float64}
    const E_tbar::Vector{Float64}
    const rho_vbar::Vector{Float64}
    const rho_cbar::Vector{Float64}
    const nu_cbar::Vector{Float64}
    # Slot indices (0 = absent).
    const ip::Int; const ird::Int; const irt::Int; const iu::Int; const iw::Int
    const iEt::Int; const irv::Int; const iv::Int; const irc::Int; const irr::Int
    const ii1::Int; const ii2::Int; const ii3::Int
    const rs::Vector{Int}
    const jp::Int; const jrt::Int; const ju::Int; const jw::Int; const jv::Int
    const jEt::Int; const jrv::Int
    # Transforms and the thermodynamic reading of the condensate.
    const ctr::Symbol; const cmu::Float64
    const rtr::Symbol; const rmu::Float64
    const itr::Symbol; const imu::Float64
    const cond_floor::Bool
    # The vapor control variable (`options[:vapor_transform]`): under `:bhyp` the vapor slot
    # is ν_v' against `nu_vbar` and the recovered `ahyp(ν_v)` is what R and X read.
    const vtr::Symbol; const vmu::Float64
    const nu_vbar::Vector{Float64}
    # ...its vertical gradient, and the derived total reference ρ̄_d + Σ α_k ν̄_k of the ρ_t
    # retirement (`mc_reference_diagnostics`; built, not yet read). These four are the TILE's
    # copies, read through `mc_nu_vbar` & co. by every consumer: they live here and not in
    # `ModelTile.mc_ref_diag` because widening that NamedTuple tips `moist_compressible_SLR`
    # over the LLVM compile cliff (measured 2026-10-20), and this struct already carries the
    # tile's reference profiles.
    const nu_vbar_z::Vector{Float64}
    const rho_tbar_w::Vector{Float64}
    const rho_tbar_w_z::Vector{Float64}
    # Accumulators, per column (active only).
    const dR::Matrix{Float64}          # (EB_NSTAGE, ncol) J/m^2 summed over steps
    const dX::Matrix{Float64}          # (EB_NSTAGE, ncol) kg/m^2 summed over steps
    const Rcur::Vector{Float64}        # column integral at the last mark
    const Xcur::Vector{Float64}
    const Rref::Vector{Float64}        # ... at the last flush (the closure's baseline)
    const Xref::Vector{Float64}
    const nsteps::Vector{Int}          # completed column steps since the last flush
    const init::Vector{Bool}           # the column has a baseline
    # Isolated-operator tendency histories (npts, length(rs), EB_NOP), three levels.
    const tn::Array{Float64,3}
    const tnm1::Array{Float64,3}
    const tnm2::Array{Float64,3}
    # Per-thread work: a full-width column for evaluating R(base + δ), the post-explicit
    # column (the refit partner's base), the expdot snapshot, and the partner's increments.
    const tmp::Array{Float64,3}        # (kDim, nvars, nthreads)
    const pre::Array{Float64,3}        # (kDim, nvars, nthreads)
    const snap::Array{Float64,3}       # (kDim, length(rs), nthreads)
    const pd1::Matrix{Float64}         # (kDim, nthreads)
    const pdv::Matrix{Float64}         # (kDim, nthreads)
end

"""
    EMPTY_ENERGY_BUDGET

The inactive [`EnergyBudget`](@ref) with no geometry: the value of every tile that is not a
moist-compressible one with a reference state. Concrete, not `nothing`, for the
`SurfaceDiag` reason: the driver reads `mtile.ebudget.active` once per column.
"""
const EMPTY_ENERGY_BUDGET = EnergyBudget(false, 0, 0, Float64[], Float64[], Float64[],
    Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], Float64[],
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Int[], 0, 0, 0, 0, 0, 0, 0,
    :none, 1.0, :none, 1.0, :none, 1.0, false, :none, 1.0, Float64[], Float64[], Float64[],
    Float64[],
    zeros(0, 0), zeros(0, 0), Float64[], Float64[], Float64[], Float64[], Int[], Bool[],
    zeros(0, 0, 0), zeros(0, 0, 0), zeros(0, 0, 0), zeros(0, 0, 0), zeros(0, 0, 0),
    zeros(0, 0, 0), zeros(0, 0), zeros(0, 0))

"Whether `options[:energy_budget]` is on (absent = off)."
energy_budget_on(options) = get(options, :energy_budget, false) === true

"""
    _eb_column_weights(gp, kDim, z) -> Vector{Float64}

The vertical quadrature of one column of the mish: the B-spline cell's Gauss rule times the
cell depth when the k-mish is one Gauss rule per cell (`kDim == num_cells_k·mubar`, the RiRk /
RLR / RRR vertical; the `mc_mynn_state` construction), otherwise the trapezoid rule on the
points (a Chebyshev column; adequate for a census, not exact).
"""
function _eb_column_weights(gp, kDim::Int, z::Vector{Float64})
    nk = gp.num_cells_k
    if nk > 0 && kDim == nk * gp.mubar
        dz_cell = (gp.kMax - gp.kMin) / nk
        qw = Springsteel.CubicBSpline._quadrature_rule(gp.mubar, gp.quadrature)[2]
        return repeat(qw .* dz_cell, outer = nk)
    end
    w = zeros(Float64, kDim)
    for k in 1:(kDim - 1)
        h = 0.5 * (z[k + 1] - z[k])
        w[k] += h; w[k + 1] += h
    end
    return w
end

"""
    _eb_horizontal_weight(gp, x, cyl) -> Float64

The horizontal quadrature weight of the column at `x`: the i-mish point's Gauss weight times
the cell width when the i-direction is a B-spline (one Gauss rule per cell), else 1; times
`x` (the radius) on the cylinders, so the domain mean is an area mean.
"""
function _eb_horizontal_weight(gp, x::Float64, cyl::Bool)
    w = 1.0
    nc = gp.num_cells
    if nc > 0 && gp.iMax > gp.iMin
        dx = (gp.iMax - gp.iMin) / nc
        nodes, qw = Springsteel.CubicBSpline._quadrature_rule(gp.mubar, gp.quadrature)
        xi = (x - gp.iMin) / dx
        xi = xi - clamp(floor(xi), 0.0, nc - 1.0)
        j = argmin(abs.(nodes .- xi))
        w = qw[j] * dx
    end
    return cyl ? w * x : w
end

"""
    mc_energy_budget(model, tile, tilepoints, ref_state, mc_ref_diag) -> EnergyBudget

The tile's [`EnergyBudget`](@ref), built once in `createModelTile`. Its geometry and reference
part exists for every moist-compressible tile with a reference (the census reads it); the
accumulators are allocated only under `options[:energy_budget] = true`, which is refused with
the exact (unsplit) semi-implicit option — its column step finishes in phase B, outside the
driver whose stages are marked here, so the budget could not close.
"""
function mc_energy_budget(model, tile, tilepoints, ref_state, mc_ref_diag)
    uses_pressure_reference(model.equation_set) || return EMPTY_ENERGY_BUDGET
    isempty(mc_ref_diag.rho_vbar) && return EMPTY_ENERGY_BUDGET
    opts = model.options
    pp = model.physical_params
    gp = model.grid_params
    vars = gp.vars
    kDim = gp.kDim
    nvars = length(vars)
    active = energy_budget_on(opts)
    active && get(opts, :exact_si, false) === true && error(
        "options[:energy_budget] = true is not supported with options[:exact_si] = true: the " *
        "unsplit solve completes the column step in phase B (exact_si_apply_column!), " *
        "outside the stages the budget marks, so the budget could not close. Run the budget " *
        "on the vertical-only semi-implicit path.")
    z = collect(Float64, tilepoints[1:kDim, end])
    wz = _eb_column_weights(gp, kDim, z)
    ncol = num_columns(tile)
    ncol = ncol > 0 ? ncol : 1
    cyl = endswith(model.equation_set, "_axisym") || endswith(model.equation_set, "_RLR")
    wx = [_eb_horizontal_weight(gp, Float64(tilepoints[(c - 1) * kDim + 1, 1]), cyl)
          for c in 1:ncol]
    wxn = wx ./ sum(wx)
    ctr = condensate_transform_mode(opts)
    cmu = get(pp, :condensate_mu, 1.0e-7)
    rtr = rain_transform_mode(opts)
    rmu = get(pp, :rain_mu, 1.0e-7)
    IS = mc_slots(model)
    ice = ice_registered(IS)
    itr = ice ? ice_transform_mode(opts) : :none
    imu = ice ? ice_mu(pp, 1) : 1.0
    rho_cbar = collect(Float64, view(Springsteel.ref_rho_c(ref_state), :, 1))
    nu_cbar = ctr === :none ? copy(rho_cbar) : bhyp.(rho_cbar, cmu)
    ip = vars["p"]; ird = vars["rho_d"]; irt = vars["rho_t"]; iu = vars["u"]
    iw = vars["w"]; iEt = vars["E_t"]; irv = IS.rho_v; iv = get(vars, "v", 0)
    irc = mc_slot(vars, "rho_c"); irr = mc_slot(vars, "rho_r")
    ii1 = ice ? IS.i1_q : 0; ii2 = ice ? IS.i2_q : 0; ii3 = ice ? IS.i3_q : 0
    rs = Int[ip, ird, irt, iu, iw, iEt, irv]
    iv > 0 && push!(rs, iv)
    push!(rs, irc, irr)
    ice && push!(rs, ii1, ii2, ii3)
    pos(s) = s == 0 ? 0 : findfirst(==(s), rs)
    nrs = length(rs)
    nt = Threads.maxthreadid()
    npts = size(tile.physical, 1)
    na = active ? 1 : 0                # zero-size accumulators when inactive
    return EnergyBudget(active, kDim, nvars, wz, z, wx, wxn,
        collect(Float64, view(ref_pressure(ref_state), :, 1)),
        collect(Float64, view(ref_rho_d(ref_state), :, 1)),
        collect(Float64, view(ref_rho_t(ref_state), :, 1)),
        collect(Float64, view(ref_total_energy(ref_state), :, 1)),
        copy(mc_ref_diag.rho_vbar), rho_cbar, nu_cbar,
        ip, ird, irt, iu, iw, iEt, irv, iv, irc, irr, ii1, ii2, ii3, rs,
        pos(ip), pos(irt), pos(iu), pos(iw), pos(iv), pos(iEt), pos(irv),
        ctr, cmu, rtr, rmu, itr, imu, condensate_floor_mode(opts),
        vapor_transform_mode(opts), vapor_mu(pp), copy(mc_ref_diag.nu_vbar),
        copy(mc_ref_diag.nu_vbar_z), copy(mc_ref_diag.rho_tbar_w),
        copy(mc_ref_diag.rho_tbar_w_z),
        zeros(EB_NSTAGE * na, ncol * na), zeros(EB_NSTAGE * na, ncol * na),
        zeros(ncol * na), zeros(ncol * na), zeros(ncol * na), zeros(ncol * na),
        zeros(Int, ncol * na), fill(false, ncol * na),
        zeros(npts * na, nrs * na, EB_NOP * na), zeros(npts * na, nrs * na, EB_NOP * na),
        zeros(npts * na, nrs * na, EB_NOP * na),
        zeros(kDim * na, nvars * na, nt * na), zeros(kDim * na, nvars * na, nt * na),
        zeros(kDim * na, nrs * na, nt * na), zeros(kDim * na, nt * na),
        zeros(kDim * na, nt * na))
end

# ── The residuals at one point and over one column ──────────────────────────────────────

@inline _eb_get(A::Matrix{Float64}, g::Int, v::Int, l::Int) = @inbounds A[g, v]
@inline _eb_get(A::Array{Float64,3}, g::Int, v::Int, l::Int) = @inbounds A[g, v, l]

"""
    _eb_point_RX(eb, A, g, l, k) -> (R, X, Λ)

`R`, `X` and the lobe pool `Λ = Σ_k 2 min(ν_k, 0)` (over the `:bhyp` mass slots) at row `g`
of the state `A` (`var_np1`, `tile.physical` with `l = 1`, or a work column with `l` the
thread), level `k`. The definitions are the netCDF writer's (`mc_derived_fields`): the
recovered condensate as the thermodynamics reads it (floored under
`options[:condensate_floor] = :diagnostic`), `T_eos` from the raw vapor, the PROGNOSTIC `ρ_t`
in the mechanical term.
"""
@inline function _eb_point_RX(eb::EnergyBudget, A, g::Int, l::Int, k::Int)
    @inbounds begin
        p = _eb_get(A, g, eb.ip, l) + eb.pbar[k]
        rd = _eb_get(A, g, eb.ird, l) + eb.rho_dbar[k]
        rt = _eb_get(A, g, eb.irt, l) + eb.rho_tbar[k]
        Et = _eb_get(A, g, eb.iEt, l) + eb.E_tbar[k]
        rv = eb.vtr === :none ? _eb_get(A, g, eb.irv, l) + eb.rho_vbar[k] :
                                ahyp(_eb_get(A, g, eb.irv, l) + eb.nu_vbar[k], eb.vmu)
        u = _eb_get(A, g, eb.iu, l)
        w = _eb_get(A, g, eb.iw, l)
        v = eb.iv > 0 ? _eb_get(A, g, eb.iv, l) : 0.0
        sc = _eb_get(A, g, eb.irc, l)
        sr = _eb_get(A, g, eb.irr, l)
        rc = recover_rho_c(sc, eb.rho_cbar[k], eb.ctr, eb.cmu)
        rr = recover_rho_r(sr, eb.rtr, eb.rmu)
        lam = 0.0
        eb.ctr === :bhyp && (lam += 2.0 * min(sc + eb.nu_cbar[k], 0.0))
        eb.rtr === :bhyp && (lam += 2.0 * min(sr, 0.0))
        ri = 0.0
        if eb.ii1 > 0
            s1 = _eb_get(A, g, eb.ii1, l); s2 = _eb_get(A, g, eb.ii2, l)
            s3 = _eb_get(A, g, eb.ii3, l)
            ri = (recover_total(s1, eb.itr, eb.imu) + recover_total(s2, eb.itr, eb.imu)) +
                 recover_total(s3, eb.itr, eb.imu)
            eb.itr === :bhyp &&
                (lam += 2.0 * ((min(s1, 0.0) + min(s2, 0.0)) + min(s3, 0.0)))
        end
        rl = eb.cond_floor ? max(rc, 0.0) + max(rr, 0.0) : rc + rr
        T = eos_temperature(p, rd, rv)
        geo = (0.5 * (((u * u) + (v * v)) + (w * w))) + (gravity * eb.z[k])
        R = Et - implied_total_energy(T, rd, rv, rl, ri, rt, geo)
        X = (((rt - rd) - rv) - rl) - ri
    end
    return R, X, lam
end

"""
    _eb_column_RX(eb, A, g0, l) -> (∫R dz, ∫X dz, max|X|, ∫Λ dz)

[`_eb_point_RX`](@ref) over the column whose level `k` is row `g0 + k` of `A`, integrated
with the mish's Gauss weights `eb.wz` [J/m², kg/m², kg/m³, kg/m²].
"""
@inline function _eb_column_RX(eb::EnergyBudget, A, g0::Int, l::Int)
    Ri = 0.0; Xi = 0.0; Xm = 0.0; Li = 0.0
    @inbounds for k in 1:eb.kDim
        r, x, lam = _eb_point_RX(eb, A, g0 + k, l, k)
        w = eb.wz[k]
        Ri += w * r; Xi += w * x; Li += w * lam
        ax = abs(x)
        ax > Xm && (Xm = ax)
    end
    return Ri, Xi, Xm, Li
end

@inline _eb_col(eb::EnergyBudget, colstart::Int) = div(colstart - 1, eb.kDim) + 1

# ── The hooks. Every one is @noinline and takes the budget (one pointer) and plain arrays;
#    mc_driver! calls them only behind `eb.active`, read once per column. ────────────────

"""
    eb_begin!(eb, colstart)

Zero this column's isolated-operator tendencies for the step about to be assembled.
"""
@noinline function eb_begin!(eb::EnergyBudget, colstart::Int)
    tn = eb.tn
    @inbounds for op in 1:EB_NOP, j in axes(tn, 2), k in 1:eb.kDim
        tn[colstart + k - 1, j, op] = 0.0
    end
    return nothing
end

"""
    eb_put!(eb, op, j, colstart, vec)

Add a tendency column `vec` (kDim) to operator `op`'s rs-slot `j` (`j = 0` is a no-op).
"""
@noinline function eb_put!(eb::EnergyBudget, op::Int, j::Int, colstart::Int,
                           vec::AbstractVector{Float64})
    j == 0 && return nothing
    tn = eb.tn
    @inbounds for k in 1:eb.kDim
        tn[colstart + k - 1, j, op] += vec[k]
    end
    return nothing
end

"""
    eb_rad!(eb, colstart, q_lw, q_sw, sw_s, R_m, C_vt)

The radiative heating's tendency: `(R_m/C_vt) q` in `p` and `q` in `E_t`, `q = q_lw + sw_s q_sw`
(the `QDOT_TH` route; its `Q_ss` share is not read by `R` or `X`).
"""
@noinline function eb_rad!(eb::EnergyBudget, colstart::Int, q_lw::Vector{Float64},
                           q_sw::Vector{Float64}, sw_s::Float64, R_m::Vector{Float64},
                           C_vt::Vector{Float64})
    tn = eb.tn
    @inbounds for k in 1:eb.kDim
        g = colstart + k - 1
        q = q_lw[g] + sw_s * q_sw[g]
        tn[g, eb.jp, EB_OP_RAD] += (R_m[k] / C_vt[k]) * q
        tn[g, eb.jEt, EB_OP_RAD] += q
    end
    return nothing
end

"""
    eb_vrec!(eb, colstart, VREC, VREAL, Tk, R_m, C_vt, partner)

The vapor nudge `VREC` into the vapor slot and, when `partner`, its pressure partner
`vapor_nudge_partner_rate` of `VREAL` into `p` — the two terms the driver writes. `VREAL` is
`VREC` itself under `vapor_transform = :none`; under `:bhyp` `VREC` is the ν-space rate and
`VREAL` the realized recovered vapor rate its partner is booked on (`_vapor_bhyp_tendency!`).
"""
@noinline function eb_vrec!(eb::EnergyBudget, colstart::Int, VREC::Vector{Float64},
                            VREAL::Vector{Float64},
                            Tk::Vector{Float64}, R_m::Vector{Float64}, C_vt::Vector{Float64},
                            partner::Bool)
    tn = eb.tn
    @inbounds for k in 1:eb.kDim
        g = colstart + k - 1
        vr = VREC[k]
        tn[g, eb.jrv, EB_OP_VREC] += vr
        va = VREAL[k]
        (partner && va != 0.0) &&
            (tn[g, eb.jp, EB_OP_VREC] += vapor_nudge_partner_rate(va, Tk[k], R_m[k], C_vt[k]))
    end
    return nothing
end

"""
    eb_snap!(eb, expdot, colstart)

Snapshot this column's `expdot` (the rs slots) before an operator that adds to it in place.
"""
@noinline function eb_snap!(eb::EnergyBudget, expdot::Matrix{Float64}, colstart::Int)
    sn = eb.snap
    tid = Threads.threadid()
    rs = eb.rs
    @inbounds for j in eachindex(rs), k in 1:eb.kDim
        sn[k, j, tid] = expdot[colstart + k - 1, rs[j]]
    end
    return nothing
end

"""
    eb_take!(eb, op, expdot, colstart)

Book `expdot − snapshot` (the operator's addition) as operator `op`'s tendency.
"""
@noinline function eb_take!(eb::EnergyBudget, op::Int, expdot::Matrix{Float64},
                            colstart::Int)
    sn = eb.snap
    tn = eb.tn
    tid = Threads.threadid()
    rs = eb.rs
    @inbounds for j in eachindex(rs), k in 1:eb.kDim
        g = colstart + k - 1
        tn[g, j, op] += expdot[g, rs[j]] - sn[k, j, tid]
    end
    return nothing
end

"""
    eb_sfc!(eb, phys, colstart, u1, v1, T1, rd1, rt1, rv1, p1, sfc, rho_t, Tk, R_m, C_vt, ke)

The surface exchange's share of the boundary layer's tendency, rebuilt from the same pure
`surface_exchange` call both closures make (identical inputs, identical outputs), on the
same surface profile `g(z) = 2/δ (1 − z/δ)`, `δ = 2 z₂`: drag `−τ g/ρ_t` on `u` (and `v`) and
its work `−(τ_u u + τ_v v) g` in `E_t`; the sensible flux `F_sh g` in `E_t` and `(R_m/C_vt)F_sh g`
in `p`; the moisture flux `F_q g` in `ρ_t` and `ρ_v`, its energy `(C_vv T + ke + gz) F_q g` in
`E_t` (the closures' `c_w + c_v`) and `R_v T F_q g` in `p`. It is moved from the `bl` row to
the `sfc` row (the closures write it inside their own loop). Under MYNN's `:pdk1` deviation the
drag-work-minus-production heat stays in `bl`.
"""
@noinline function eb_sfc!(eb::EnergyBudget, phys::Array{Float64,3}, colstart::Int,
                           u1::Float64, v1::Float64, T1::Float64, rd1::Float64, rt1::Float64,
                           rv1::Float64, p1::Float64, sfc, rho_t::Vector{Float64},
                           Tk::Vector{Float64}, R_m::Vector{Float64}, C_vt::Vector{Float64},
                           ke::Vector{Float64}, J_v::Vector{Float64})
    z = eb.z
    sx = surface_exchange(u1, v1, T1, rd1, rt1, rv1, p1, z[1], sfc.SST, sfc)
    F_sh = sx.F_sh; F_q = sx.F_q; tau_u = sx.tau_u; tau_v = sx.tau_v
    delta = 2.0 * z[2]
    inv_delta = 1.0 / delta
    tn = eb.tn
    @inbounds for k in 1:eb.kDim
        z[k] < delta || continue
        g = colstart + k - 1
        gz = 2.0 * inv_delta * (1.0 - (z[k] * inv_delta))
        u = phys[g, eb.iu, 1]
        v = eb.iv > 0 ? phys[g, eb.iv, 1] : 0.0
        cwv = (Cvv * Tk[k]) + ke[k] + (gravity * z[k])
        su = -(tau_u * gz) / rho_t[k]
        sE = (-((tau_u * u) + (tau_v * v)) * gz) + (F_sh * gz) + (cwv * F_q * gz)
        sp = ((R_m[k] / C_vt[k]) * F_sh * gz) + (Rv * Tk[k] * F_q * gz)
        sq = F_q * gz
        # the vapor slot holds ν under the vapor control variable: its share through J_v
        sv = eb.vtr === :none ? sq : J_v[k] * sq
        for (j, s) in ((eb.ju, su), (eb.jEt, sE), (eb.jp, sp), (eb.jrt, sq), (eb.jrv, sv))
            tn[g, j, EB_OP_SFC] += s
            tn[g, j, EB_OP_BL] -= s
        end
        if eb.jv > 0
            sv = -(tau_v * gz) / rho_t[k]
            tn[g, eb.jv, EB_OP_SFC] += sv
            tn[g, eb.jv, EB_OP_BL] -= sv
        end
    end
    return nothing
end

"""
    eb_partner_snap!(eb, d1, dv) / eb_partner_take!(eb, d1, dv)

Around `_refit_partner_fold!`: the partner's own `p` and vapor increments, `after − before`
of the ETD columns it folds into.
"""
@noinline function eb_partner_snap!(eb::EnergyBudget, d1::Vector{Float64},
                                    dv::Vector{Float64})
    tid = Threads.threadid()
    @inbounds for k in 1:eb.kDim
        eb.pd1[k, tid] = d1[k]
        eb.pdv[k, tid] = dv[k]
    end
    return nothing
end

@doc (@doc eb_partner_snap!)
@noinline function eb_partner_take!(eb::EnergyBudget, d1::Vector{Float64},
                                    dv::Vector{Float64})
    tid = Threads.threadid()
    @inbounds for k in 1:eb.kDim
        eb.pd1[k, tid] = d1[k] - eb.pd1[k, tid]
        eb.pdv[k, tid] = dv[k] - eb.pdv[k, tid]
    end
    return nothing
end

"Book the mark `(R, X)` into `stage` of column `ci` and make it the current state."
@inline function _eb_book!(eb::EnergyBudget, stage::Int, ci::Int, R::Float64, X::Float64)
    @inbounds begin
        eb.dR[stage, ci] += R - eb.Rcur[ci]
        eb.dX[stage, ci] += X - eb.Xcur[ci]
        eb.Rcur[ci] = R
        eb.Xcur[ci] = X
    end
    return nothing
end

"""
    eb_refit!(eb, phys, colstart)

The column step's first mark, at state `n` (the post-refit slots). With a baseline it books
the `refit` row — state `n` minus the previous step's last mark, which is the pre-refit
`var_np1` the previous step left — otherwise (the first step of a run) it takes state `n`
as the closure's baseline and books nothing (`var_np1` holds no fitted state at `t = 1`).
"""
@noinline function eb_refit!(eb::EnergyBudget, phys::Array{Float64,3}, colstart::Int)
    ci = _eb_col(eb, colstart)
    R, X, _, _ = _eb_column_RX(eb, phys, colstart - 1, 1)
    @inbounds if eb.init[ci]
        _eb_book!(eb, EB_REFIT, ci, R, X)
    else
        eb.Rcur[ci] = R; eb.Xcur[ci] = X
        eb.Rref[ci] = R; eb.Xref[ci] = X
        eb.init[ci] = true
    end
    return nothing
end

"""
    eb_explicit!(eb, vnp1, phys, colstart, t, ts)

After `explicit_timestep`: books `explicit` (the actual advance), splits it over the isolated
operators — each `R(xⁿ + δ_op) − R(xⁿ)` with `δ_op` its AB3-weighted increment (Euler at
`t = 1`, AB2 at `t = 2`, exactly `explicit_timestep`'s startup), `dyn` on the actual increment
less the isolated ones, `nonlin` the remainder — rotates the operators' histories as
`explicit_timestep` rotates `expdot`, and keeps the post-explicit column as the refit
partner's base.
"""
@noinline function eb_explicit!(eb::EnergyBudget, vnp1::Matrix{Float64},
                                phys::Array{Float64,3}, colstart::Int, t::Int, ts::Float64)
    ci = _eb_col(eb, colstart)
    tid = Threads.threadid()
    kDim = eb.kDim
    rs = eb.rs
    tmp = eb.tmp; pre = eb.pre
    tn = eb.tn; tnm1 = eb.tnm1; tnm2 = eb.tnm2
    Rn = @inbounds eb.Rcur[ci]; Xn = @inbounds eb.Xcur[ci]
    R1, X1, _, _ = _eb_column_RX(eb, vnp1, colstart - 1, 1)
    dRt = R1 - Rn; dXt = X1 - Xn
    a1, a2, a3 = t == 1 ? (ts, 0.0, 0.0) :
                 t == 2 ? (1.5 * ts, -0.5 * ts, 0.0) :
                          ((23.0 / 12.0) * ts, (-16.0 / 12.0) * ts, (5.0 / 12.0) * ts)
    sR = 0.0; sX = 0.0
    @inbounds begin
        # dyn first: the actual increment less every isolated one, on top of state n.
        for j in eachindex(rs), k in 1:kDim
            g = colstart + k - 1
            s = rs[j]
            d = vnp1[g, s] - phys[g, s, 1]
            for op in 1:EB_NOP
                d -= ((a1 * tn[g, j, op]) + (a2 * tnm1[g, j, op])) + (a3 * tnm2[g, j, op])
            end
            tmp[k, s, tid] = phys[g, s, 1] + d
        end
        r, x, _, _ = _eb_column_RX(eb, tmp, 0, tid)
        eb.dR[EB_X_DYN, ci] += r - Rn; eb.dX[EB_X_DYN, ci] += x - Xn
        sR += r - Rn; sX += x - Xn
        for op in 1:EB_NOP
            for j in eachindex(rs), k in 1:kDim
                g = colstart + k - 1
                s = rs[j]
                tmp[k, s, tid] = phys[g, s, 1] +
                    (((a1 * tn[g, j, op]) + (a2 * tnm1[g, j, op])) + (a3 * tnm2[g, j, op]))
            end
            r, x, _, _ = _eb_column_RX(eb, tmp, 0, tid)
            row = EB_OP_ROW[op]
            eb.dR[row, ci] += r - Rn; eb.dX[row, ci] += x - Xn
            sR += r - Rn; sX += x - Xn
        end
        eb.dR[EB_X_NONLIN, ci] += dRt - sR
        eb.dX[EB_X_NONLIN, ci] += dXt - sX
        # Rotate, as explicit_timestep rotates expdot.
        for op in 1:EB_NOP, j in eachindex(rs), k in 1:kDim
            g = colstart + k - 1
            if t == 1
                tnm1[g, j, op] = tn[g, j, op]
            else
                tnm2[g, j, op] = tnm1[g, j, op]
                tnm1[g, j, op] = tn[g, j, op]
            end
        end
        for s in rs, k in 1:kDim
            pre[k, s, tid] = vnp1[colstart + k - 1, s]
        end
    end
    _eb_book!(eb, EB_EXPLICIT, ci, R1, X1)
    return nothing
end

"""
    eb_etd!(eb, vnp1, colstart)

After the ETD increments land (`relaxation_adjustment_qss!`, before the exchange
realization): books `etd`, and splits off the refit partner's own increments evaluated on
the post-explicit column (`etd:refit_partner`; `etd:phase` is the rest).
"""
@noinline function eb_etd!(eb::EnergyBudget, vnp1::Matrix{Float64}, colstart::Int)
    ci = _eb_col(eb, colstart)
    tid = Threads.threadid()
    kDim = eb.kDim
    tmp = eb.tmp; pre = eb.pre
    R0 = @inbounds eb.Rcur[ci]; X0 = @inbounds eb.Xcur[ci]
    R2, X2, _, _ = _eb_column_RX(eb, vnp1, colstart - 1, 1)
    @inbounds begin
        for s in eb.rs, k in 1:kDim
            tmp[k, s, tid] = pre[k, s, tid]
        end
        for k in 1:kDim
            tmp[k, eb.ip, tid] += eb.pd1[k, tid]
            tmp[k, eb.irv, tid] += eb.pdv[k, tid]
        end
        rp, xp, _, _ = _eb_column_RX(eb, tmp, 0, tid)
        eb.dR[EB_ETD_PARTNER, ci] += rp - R0; eb.dX[EB_ETD_PARTNER, ci] += xp - X0
        eb.dR[EB_ETD_PHASE, ci] += (R2 - R0) - (rp - R0)
        eb.dX[EB_ETD_PHASE, ci] += (X2 - X0) - (xp - X0)
        # The partner's increments are this step's: clear them for a step that has none.
        for k in 1:kDim
            eb.pd1[k, tid] = 0.0; eb.pdv[k, tid] = 0.0
        end
    end
    _eb_book!(eb, EB_ETD, ci, R2, X2)
    return nothing
end

"""
    eb_mark!(eb, stage, vnp1, colstart)

Book `stage` as the change of the column integral since the last mark. With
`stage == EB_CLAMP` it is the column step's last mark, and the step is counted.
"""
@noinline function eb_mark!(eb::EnergyBudget, stage::Int, vnp1::Matrix{Float64},
                            colstart::Int)
    ci = _eb_col(eb, colstart)
    R, X, _, _ = _eb_column_RX(eb, vnp1, colstart - 1, 1)
    _eb_book!(eb, stage, ci, R, X)
    stage == EB_CLAMP && (@inbounds eb.nsteps[ci] += 1)
    return nothing
end

# ── The census rows (P4): always measured, plain arrays ─────────────────────────────────

"""
    refit_dr_census!(eb, st, tid, vnp1, phys, colstart, booked)

`MC_REFIT_DR` and `MC_REFIT_BOOKED`: the column-integrated change of `R` across the per-step
refit — `(E_t⁺ − E_t⁻) − (E_implied⁺ − E_implied⁻)`, `⁻` the pre-refit `var_np1`, `⁺` the
post-refit slots — and the energy the refit partner booked for it (`booked`, the column
integral of `δp·C_vt/R_m` the partner folded into `p`; 0 with the partner off), each weighted by
the column's share of the tile's horizontal quadrature, so the rows hold the tile-mean column
integral [J/m²] summed over steps.
"""
@noinline function refit_dr_census!(eb::EnergyBudget, st::Matrix{Float64}, tid::Int,
                                    vnp1::Matrix{Float64}, phys::Array{Float64,3},
                                    colstart::Int, booked::Float64)
    ci = _eb_col(eb, colstart)
    Rp, _, _, _ = _eb_column_RX(eb, phys, colstart - 1, 1)
    Rm, _, _, _ = _eb_column_RX(eb, vnp1, colstart - 1, 1)
    @inbounds begin
        st[MC_REFIT_DR, tid] += eb.wxn[ci] * (Rp - Rm)
        st[MC_REFIT_BOOKED, tid] += eb.wxn[ci] * booked
    end
    return nothing
end

"""
    _eb_vapor_negative_points(eb, phys) -> Int

The `MC_VNEG` census: how many gridpoints of the state `phys` (`tile.physical`) carry a
negative vapor CONTROL variable, `ν_v' + ν̄_v < 0` (under `:none` that is the density
`ρ_v' + ρ̄_v`, since `nu_vbar ≡ rho_vbar` there).
"""
@noinline function _eb_vapor_negative_points(eb::EnergyBudget, phys::Array{Float64,3})
    n = 0
    kDim = eb.kDim
    @inbounds for g in axes(phys, 1)
        k = mod1(g, kDim)
        (phys[g, eb.irv, 1] + eb.nu_vbar[k]) < 0.0 && (n += 1)
    end
    return n
end

# ── Master side: gather, assemble, write ────────────────────────────────────────────────

"""
    energy_budget_snapshot!(mtile) -> NamedTuple or nothing

ON THE WORKER: this tile's accumulated budget since the last snapshot, as plain arrays, and
the accumulators reset (the next interval starts at the current marks). `nothing` when the
budget is off or no column has stepped. `actualR`/`actualX` are each column's change of the
integral between the two snapshots (its last mark minus the previous snapshot's), the
`closure` row's reference.
"""
function energy_budget_snapshot!(mtile)
    eb = mtile.ebudget
    eb.active || return nothing
    nmax = isempty(eb.nsteps) ? 0 : maximum(eb.nsteps)
    nmax > 0 || return nothing
    out = (; offset = mtile.tile.params.patchOffsetL, dR = copy(eb.dR), dX = copy(eb.dX),
             actualR = eb.Rcur .- eb.Rref, actualX = eb.Xcur .- eb.Xref, wx = copy(eb.wx),
             nsteps = copy(eb.nsteps), ts = mtile.model.ts)
    fill!(eb.dR, 0.0); fill!(eb.dX, 0.0)
    eb.Rref .= eb.Rcur; eb.Xref .= eb.Xcur
    fill!(eb.nsteps, 0)
    return out
end

"`energy_budget_snapshot!` on every worker (master side; the `gather_physics` pattern)."
gather_energy_budget(workerids::Vector{Int64}) =
    [get_val_from(w, :(Scythe.energy_budget_snapshot!(mtile))) for w in workerids]

"""
    energy_budget_rows(snaps) -> Vector{NamedTuple}

Assemble per-tile snapshots into the CSV rows of one output time: per stage of
[`EB_STAGES`](@ref), then `total` (the sum of the top-level rows), `actual` (the change of
the column integral between the two output times) and `closure` (`actual − total`, the
instrument's own test: rounding). Each row is
`(stage, dR_dt_Wm2, dX_dt_kgm2h, dR_dt_colmax_Wm2, dX_dt_colmax_kgm2h)`: the domain mean
(horizontal quadrature weights; area weights on the cylinders) of the column rate, and the
signed column rate of largest magnitude. `R` in W/m² (J/m² per second), `X` in kg/m² per hour.
"""
function energy_budget_rows(snaps::AbstractVector)
    ss = [s for s in snaps if s !== nothing]
    isempty(ss) && return NamedTuple[]
    dR = reduce(hcat, (s.dR for s in ss))
    dX = reduce(hcat, (s.dX for s in ss))
    aR = reduce(vcat, (s.actualR for s in ss))
    aX = reduce(vcat, (s.actualX for s in ss))
    wx = reduce(vcat, (s.wx for s in ss))
    secs = maximum(maximum(s.nsteps) for s in ss) * ss[1].ts
    W = sum(wx)
    hours = secs / 3600.0
    function row(name, r, x)
        mr = sum(wx .* r) / W / secs
        mx = sum(wx .* x) / W / hours
        ir = argmax(abs.(r)); ix = argmax(abs.(x))
        return (; stage = name, dR_dt_Wm2 = mr, dX_dt_kgm2h = mx,
                  dR_dt_colmax_Wm2 = r[ir] / secs, dX_dt_colmax_kgm2h = x[ix] / hours)
    end
    rows = [row(EB_STAGES[s], dR[s, :], dX[s, :]) for s in 1:EB_NSTAGE]
    tR = vec(sum(dR[collect(EB_TOP), :]; dims = 1))
    tX = vec(sum(dX[collect(EB_TOP), :]; dims = 1))
    push!(rows, row("total", tR, tX))
    push!(rows, row("actual", aR, aX))
    push!(rows, row("closure", aR .- tR, aX .- tX))
    return rows
end

"The header of `energy_budget.csv`."
const EB_CSV_HEADER = "time,stage,dR_dt_Wm2,dX_dt_kgm2h,dR_dt_colmax_Wm2,dX_dt_colmax_kgm2h"

"""
    write_energy_budget(dir, t, rows)

Append one output time's [`energy_budget_rows`](@ref) to `dir/energy_budget.csv` (the header
is written when the file is new), and print the one-line summary into the run log.
"""
function write_energy_budget(dir::AbstractString, t::Float64, rows)
    isempty(rows) && return nothing
    path = joinpath(dir, "energy_budget.csv")
    fresh = !isfile(path)
    open(path, "a") do io
        fresh && println(io, EB_CSV_HEADER)
        for r in rows
            println(io, t, ",", r.stage, ",", r.dR_dt_Wm2, ",", r.dX_dt_kgm2h, ",",
                    r.dR_dt_colmax_Wm2, ",", r.dX_dt_colmax_kgm2h)
        end
    end
    get1(n) = (i = findfirst(r -> r.stage == n, rows); i === nothing ? NaN : rows[i].dR_dt_Wm2)
    f(x) = round(x; sigdigits = 4)
    println("Scythe: energy budget t=$(t) s: dR/dt total=$(f(get1("total"))) W/m2 " *
            "(closure $(f(get1("closure")))), refit=$(f(get1("refit"))), " *
            "explicit=$(f(get1("explicit"))) [dyn=$(f(get1("explicit:dyn"))), " *
            "bl=$(f(get1("explicit:bl"))), sfc=$(f(get1("explicit:sfc")))], " *
            "etd=$(f(get1("etd"))) [refit_partner=$(f(get1("etd:refit_partner")))], " *
            "exch_real=$(f(get1("exch_real"))), acoustic_si=$(f(get1("acoustic_si"))); " *
            "rows in $(path)")
    return nothing
end

"""
    energy_census_trace(mtile, t)

Report and reset the energy / partition census ([`MC_REFIT_DR`](@ref) and its rows) at the
output cadence: called once per step from the single-threaded pre-column-loop slot of
`advanceTimestep` (beside [`mc_stiffness_trace`](@ref)), it counts the completed steps and,
when step `t − 1` was an output step, takes the partition-gap rows from `tile.physical` (the
output state, already refit) and prints one line:

    Scythe: energy census t=… s: refit dR/dt=… W/m2, refit partner booked=… W/m2,
    Stage L dnu per step (c, r, i1, i2, i3)=… kg/m2, max|X|=… kg/m3, int X=… kg/m2,
    pool=… kg/m2

the refit rows as tile-mean rates over the interval, the Stage L rows as tile-mean column
integrals per step, the `X` rows at the output state. Then the accumulating rows are zeroed.
Nothing is printed for a tile with no reference (`mtile.ebudget` empty).
"""
function energy_census_trace(mtile, t::Int64)
    st = mtile.mc_water_stats
    eb = mtile.ebudget
    (size(st, 2) == 0 || isempty(eb.wz)) && return nothing
    t > 1 || return nothing
    model = mtile.model
    st[MC_ECEN_STEPS, 1] += 1.0
    out_int = max(1, round(Int, model.output_interval / model.ts))
    mod(t - 1, out_int) == 0 || return nothing
    nst = st[MC_ECEN_STEPS, 1]
    # The partition gap of the output state, over the tile.
    phys = mtile.tile.physical
    kDim = eb.kDim
    xm = 0.0; xi = 0.0; li = 0.0
    for ci in eachindex(eb.wxn)
        _, X, Xm, L = _eb_column_RX(eb, phys, (ci - 1) * kDim, 1)
        xm = max(xm, Xm); xi += eb.wxn[ci] * X; li += eb.wxn[ci] * L
    end
    fill!(view(st, MC_XGAP_MAX:MC_XGAP_POOL, :), 0.0)
    st[MC_XGAP_MAX, 1] = xm; st[MC_XGAP_INT, 1] = xi; st[MC_XGAP_POOL, 1] = li
    # MC_VNEG: the vapor control variable's negative points at the output state.
    nneg = _eb_vapor_negative_points(eb, phys)
    fill!(view(st, MC_VNEG, :), 0.0)
    st[MC_VNEG, 1] = Float64(nneg)
    npts = size(phys, 1)
    secs = nst * model.ts
    s(r) = sum(view(st, r, :))
    f(x) = round(x; sigdigits = 4)
    println("Scythe: energy census t=$(round((t - 1) * model.ts; digits = 2)) s: " *
            "refit dR/dt=$(f(s(MC_REFIT_DR) / secs)) W/m2, " *
            "refit partner booked=$(f(s(MC_REFIT_BOOKED) / secs)) W/m2, " *
            "Stage L dnu per step (c, r, i1, i2, i3)=($(f(s(MC_LOBE_C) / nst)), " *
            "$(f(s(MC_LOBE_R) / nst)), $(f(s(MC_LOBE_I1) / nst)), $(f(s(MC_LOBE_I2) / nst)), " *
            "$(f(s(MC_LOBE_I3) / nst))) kg/m2, max|X|=$(f(xm)) kg/m3, int X=$(f(xi)) kg/m2, " *
            "pool=$(f(li)) kg/m2, nu_v<0 points=$(nneg) " *
            "($(f(100.0 * nneg / max(npts, 1))) %)")
    fill!(view(st, MC_REFIT_DR:MC_LOBE_I3, :), 0.0)
    st[MC_ECEN_STEPS, 1] = 0.0
    return nothing
end
