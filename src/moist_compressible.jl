# Total-energy moist compressible equation set.
#
# Prognostic variables (XZ slice): p [Pa], rho_d, rho_t, u, w, E_t [J/m^3], Q_ss
# [kg/m^3], rho_r, rho_c, rho_v. EVERY WATER SPECIES IS PROGNOSTIC — vapor, cloud, rain
# and (under options[:ice_microphysics]) the twelve ice moments — and temperature
# follows in CLOSED FORM from the Bryan & Fritsch (2002) total energy (see
# retrieve_temperature), with no iteration and no dependence on the partition.
#
# ── The criterion this equation set converged on
#
# A DIAGNOSTIC IS SAFE WHEN THE RETRIEVED QUANTITY IS NOT SMALL COMPARED WITH THE TERMS
# BEING DIFFERENCED. That is the whole content of a long campaign, and it is worth
# stating before the history, because it is what decides the next such question too.
#
# T is safe: it is large, and it comes out of a linear retrieval whose inputs are the
# conserved p, E_t, rho_t. A small water species retrieved as the residual of large
# ones is not safe, and the failure is CONDITIONING — not a bug to be fixed inside the
# retrieval, but a property of which variable was chosen to be the leftover.
#
# ── How the vapor got here: residual -> blend -> prognostic
#
# 1. rho_c AS THE RESIDUAL of four fitted fields. In cloud-free air the true Q_ss sits
#    exactly ON its admissible ceiling, so fit-level error (~2e-6 kg/m^3) put rho_c on
#    the wrong side of zero, the closure evaporated cloud that was not there, and
#    because the residual was REGENERATED every step it was a sustained pressure sink
#    (3.19 Pa/s at t = 0 on the balanced TC vortex — all of the measured tendency).
#    See reference/HANDOFF_DIAGNOSED_CLOUD.md.
#
# 2. rho_c PROGNOSTIC, rho_v THE RESIDUAL. This put the same fit error where it was
#    thought to be harmless: 1e-6 against a vapor density of ~1e-2 is a 5e-5 relative
#    error, in a field that no longer feeds back into temperature at all. It was a large
#    improvement and it was not the end, because IN CLOUD the cancellation is not
#    against rho_t but against rho_c: rho_c/rho_w reached 1.0011 at the worst measured
#    point, and 1184 of 1186 in-cloud negatives came from that one difference.
#
# 3. THE C1 REGIME BLEND (2026-07-29 to Stage A). rho_v was retrieved as a smoothstep
#    blend of the density-budget residual and the supersaturation residual Q_ss + rho_vs,
#    each used where the other cancelled, with a cap on their difference. It was a real
#    fix for the liquid set and it is documented in reference/HANDOFF_VAPOR_RETRIEVAL.md.
#    ICE BROKE IT: a fifth independently-ringing term entered the density residual, the
#    ahyp cloud transform's clamp channel fed its discarded negative ringing into the
#    residual vapor (8.3 % deeper deficit, 23 % more points), and the ice physics reading
#    that vapor detonated the retrieval — `min rho_c == 0.0 exactly` was death across all
#    eight ablation arms.
#
# 4. rho_v PROGNOSTIC (Stage A, here). The chain ends: no water variable is a residual.
#    The vapor is transported, phase change is an equal and opposite pair of SLOT
#    sources, and the conditioning question does not arise for any of them.
#
# ── What rho_t is still for, and the reconciliation chain
#
# rho_t stays prognostic as the CONSERVATION ANCHOR (and the semi-implicit acoustic
# structure is built on it, untouched by Stage A). With rho_v carried beside it the set
# holds one redundancy — rho_t - rho_d - rho_c - rho_r - rho_i is a second statement
# about the vapor — and redundancy has to be reconciled or the two drift apart under
# splitting error. Q_ss is a second such redundancy, one link further out. Both are
# removed by slow nudges, never by projection:
#
#     Q_ss  --(tau_qss)-->  rho_v  --(tau_rec)-->  the rho_t density budget
#
# `qss_relaxation` pulls Q_ss onto the supersaturation the prognostic vapor implies;
# `rho_v_reconcile` pulls the vapor onto the vapor the conserved masses imply. Both
# timescales default to 10 s, long compared with the timestep, so transport keeps the
# smooth field and only the accumulated inconsistency is removed. rho_t is never nudged,
# so the conserved water mass is exactly conserved. The two gaps are measured every step
# (MC_VAPOR_GAP, MC_QSS_GAP) and are identically zero on a resting base by construction:
# the vapor slot is carried as a perturbation from the DERIVED profile
# rho_vbar = rho_tbar - rho_dbar - rho_cbar, which is the same expression res_rho_t
# reassembles from the same fitted columns (see vapor_slot, mc_reference_diagnostics).
#
# Q_ss is therefore not thermodynamic — it is purely the microphysics' supersaturation
# driver, smooth and advected rather than a pointwise difference against a saturation
# curve four decades above it.
#
# Whether rho_t should eventually be RETIRED instead (rho_t = sum of components,
# conservation by construction, no nudge at all) is parked in
# reference/HANDOFF_ISHMAEL_SESSION.md: it touches the semi-implicit solve.
#
# ── Negative water
#
# Negative water is a RESOLUTION DIAGNOSTIC and is never clamped
# (reference/FINDINGS_NEGATIVE_WATER_ATTRIBUTION.md): a positive-definite spike the
# vertical basis cannot resolve undershoots on the way in, and the size of that
# undershoot is the signal that the column wants more nodes. `clamp_water!` floors
# rho_c and rho_r after every column step; because rho_t is prognostic and the phase
# change is internal to the water, that floor is exactly a phase change — total water
# and E_t are untouched and the retrieval supplies the matching latent heat — so it
# conserves mass, water and energy by construction.
#
# ── Conservation and the first law
#
# The conserved quantities (rho_d, rho_t, E_t) are extensive flux-form prognostics,
# so the Galerkin low-pass filter preserves their integrals (no Jensen drift). The
# energy equation is the exact first law: phase change carries no energy source
# (the R_v*T*ln(H) term of the entropy identity is entropy production, which
# cancels exactly against the production hidden in T*ds_t; see
# reference/Scythe_moist_compressible.tex).
#
# All new functions for this equation set live in this file: thermodynamic
# helpers, condensation closure, the equation set RHS, its semi-implicit
# adjustment, initializers, and the reference-state writer.

using Springsteel.Thermodynamics: rho_v_sat, internal_energy_bf02

# ── Per-thread scratch for the equation-set RHS ────────────────────────────────

"""
The live broadcast temporaries of [`moist_compressible_XZ`](@ref). Each is a `kDim` column,
recomputed every column of every timestep — as fresh allocations they were ~150 of the
function's 163 per-call allocations.

Keyed by NAME, not by index. An index-numbered scratch pool (`view(pool, :, 7, tid)`) makes it
easy to hand the same buffer to two temporaries that are live at once, and the resulting
corruption is silent and hard to see. A `NamedTuple` cannot hold a duplicate field, so that
class of bug cannot be written here at all.
"""
const MC_SCRATCH_SLOTS = (
    # ── moist_compressible_XZ ──
    :p, :rho_d, :rho_t, :E_t, :Q_ss, :rho_c, :rho_r,                  # totals
    :p_z, :rho_d_z, :rho_t_z, :E_t_z, :Q_ss_z, :rho_c_z,              # total vertical gradients
    :ke, :geo, :M, :Tk, :p_hPa, :rho_vs, :rho_v, :res_rho_t, :rho_liq, :q_v, :q_l, # diagnostic state
    :rho_liq_t,   # what the THERMODYNAMICS reads; == rho_liq unless condensate_floor_mode
    :nu_c, :nu_c_z, :Jc,   # slot-9 control variable, its gradient, dnu/drho (see bhyp/ahyp)
    :nu_r, :nu_r_z, :Jr, # slot-8 control variable, its gradient, dnu/drho (rain_transform_mode)
    :C_vt, :R_m, :C_pt, :gamma_m, :Lv, :drvs_dT, :drvs_dp,            # mixture thermo
    :Q_s, :Qdot, :Qdot_r, :div,                                       # condensation, divergence
    :cap_c, :cap_r, :cap_v,      # AB3 depletion bounds, DIAGNOSTIC ONLY (see _ab3_sink_bound)
    :invtau_c, :invtau_r,        # per-channel relaxation rates, for `mc_stiffness_census!`
    # ── Stage 2b: the RAIN DONOR's evaporation conductance and its realization factor ──
    # `kappa_ev` is `max(−Q̇_r, 0)/ρ_r` at the frozen state — the rain's evaporation sink
    # expressed as a conductance on its own reservoir (TeX §donor_relax, "The rain reservoir
    # has a third sink"), written beside the condensation closure where `Q̇_r` and `ρ_r` are
    # both known and read again inside `mc_ice_sources!`, which needs it to complete the rain
    # donor's TOTAL. `f_rain` is that donor's factor `J₀(κ_tot Δt)`: the warm-path value
    # `J₀(κ_ev Δt)` where there is no ice, OVERWRITTEN with the full ice-plus-evaporation
    # factor at every gridpoint the ice loop visits. It is folded into `invtau_r`/`Qdot_r`
    # themselves before λ and N are formed, so the pair, the slot and the census all read the
    # one realized conductance. Exactly 1.0 wherever the rain does not evaporate.
    :kappa_ev, :f_rain,
    :AUTO_COLL, :Vt, :Fr, :Fr_z, :E_sed, :E_sed_z,                    # warm-rain microphysics
    # Two-moment rain number (options[:rain_moments] == 2). Same six-way shape the mass
    # slot has — control variable, its gradient, the Jacobian, the recovered density, the
    # NUMBER-weighted fall speed, its flux and the flux divergence — plus one accumulator
    # for the number sources. Allocated unconditionally, like every other name here: eight
    # kDim columns per thread against the ~90 already present.
    :n_r, :nu_nr, :nu_nr_z, :Jnr, :Vtn, :Fnr, :Fnr_z, :NR_SRC,
    # ── Ice microphysics (options[:ice_microphysics] === :ishmael) ──────────────────────
    # SEVEN columns per ice slot, in the order the slots are registered: the recovered
    # quantity, the control variable and its gradient, the Jacobian, the process-source
    # accumulator and the sedimentation flux with its divergence. That is the n_r shape with
    # the fall speed dropped (the ISHMAEL speeds come out of one per-species call, so they
    # are formed pointwise into the flux rather than staged in a column of their own).
    #
    # The names are `<species><moment>`: `i1q` is the MASS of species 1, `i1n` its number,
    # `i1a`/`i1c` the two spheroid volume moments. Allocated unconditionally like everything
    # else here — 90 kDim columns per thread on top of the ~100 already present, ~0.2 MB per
    # thread at kDim = 300, and NOT conditional on the option, because a NamedTuple whose
    # field set depended on a run-time option would make `ModelTile` non-concrete.
    :i1q, :nu_i1q, :nu_i1q_z, :J_i1q, :SRC_i1q, :F_i1q, :F_i1q_z,
    :i1n, :nu_i1n, :nu_i1n_z, :J_i1n, :SRC_i1n, :F_i1n, :F_i1n_z,
    :i1a, :nu_i1a, :nu_i1a_z, :J_i1a, :SRC_i1a, :F_i1a, :F_i1a_z,
    :i1c, :nu_i1c, :nu_i1c_z, :J_i1c, :SRC_i1c, :F_i1c, :F_i1c_z,
    :i2q, :nu_i2q, :nu_i2q_z, :J_i2q, :SRC_i2q, :F_i2q, :F_i2q_z,
    :i2n, :nu_i2n, :nu_i2n_z, :J_i2n, :SRC_i2n, :F_i2n, :F_i2n_z,
    :i2a, :nu_i2a, :nu_i2a_z, :J_i2a, :SRC_i2a, :F_i2a, :F_i2a_z,
    :i2c, :nu_i2c, :nu_i2c_z, :J_i2c, :SRC_i2c, :F_i2c, :F_i2c_z,
    :i3q, :nu_i3q, :nu_i3q_z, :J_i3q, :SRC_i3q, :F_i3q, :F_i3q_z,
    :i3n, :nu_i3n, :nu_i3n_z, :J_i3n, :SRC_i3n, :F_i3n, :F_i3n_z,
    :i3a, :nu_i3a, :nu_i3a_z, :J_i3a, :SRC_i3a, :F_i3a, :F_i3a_z,
    :i3c, :nu_i3c, :nu_i3c_z, :J_i3c, :SRC_i3c, :F_i3c, :F_i3c_z,
    # Ice AGGREGATES over the three species: the total ice density (the retrieval's ρ_i),
    # what the thermodynamics reads under `condensate_floor`, the mixing ratio q_i, the
    # summed MASS flux divergence (the only ice flux that reaches ρ_t) and the ice
    # sedimentation energy flux with its divergence (the only one that reaches E_t).
    # `anchor_f` is the SEDIMENTATION anchor share (TeX §Reconciliation of the condensate
    # partition, third leg): min(1, headroom/ρ_i) per gridpoint, exactly 1.0 where the
    # partition is admissible, applied to all twelve flux assemblies so the phantom part of
    # a detached ice mass does not fall — and does not move ρ_t or E_t.
    :rho_ice, :rho_ice_t, :q_i, :Fi_z, :E_sed_i, :E_sed_i_z, :anchor_f,
    # ── Ice PHYSICS (S8): the process rates the twelve slots and the shared thermodynamics
    # read. `Qdot_i<k>` is species k's deposition/sublimation rate [kg/m³/s] (TeX Eq.
    # dep_rate) and `invtau_i<k>` its relaxation rate for the stiffness census; `Q_s_i` is
    # the ice psychrometric factor `𝒬_{s,i}` (Eq. Qs_ice); `FRZ_NET` is the NET liquid→ice
    # conversion `Q̇_freeze` [kg/m³/s] that heats T and p with the bare `L_f` (Eqs.
    # dThat_ice/dphat_ice); `ICE_C`/`ICE_R`/`ICE_NR` are the back-reactions on the cloud
    # mass, rain mass and rain number; `Vi<k>m`/`Vi<k>n` are the MASS- and NUMBER-weighted
    # fall speeds (negative downward, like `Vt`), mass-weighted also carrying the `a`/`c`
    # volume moments.
    :Qdot_i1, :Qdot_i2, :Qdot_i3, :invtau_i1, :invtau_i2, :invtau_i3,
    :Q_s_i, :FRZ_NET, :ICE_C, :ICE_R, :ICE_NR,
    # `f_ice<k>` is species k's DONOR realization factor (`_ice_donor_factors`' ice sibling),
    # formed in `mc_ice_sources!` from that species' TOTAL mass sink and read again in the ETD
    # pre-compute, which is where the sublimation half of that sink is realized. One factor,
    # two application sites, because the two halves of the sink are computed in two places.
    :f_ice1, :f_ice2, :f_ice3,
    # `f_isn<k>` is the RESIDUAL number realization the sublimation number sink still has to
    # apply at its own site: the ice NUMBER donor's factor divided by whatever MASS factor
    # `invtau_i<k>` is already carrying there (`f_ice<k>` where the step-n classification
    # said the channel is a sink, `1.0` where it did not). One column rather than a second
    # division in the ETD pre-compute, and an exact `1.0` — hence bitwise inert — wherever
    # `options[:ice_number_realization]` is off. A RATIO, so it is not itself bounded by one
    # — below the melting level the number conductance is the SMALLER of the two and this
    # column reads slightly ABOVE 1, making up what the mass factor under-realized. What is
    # bounded is the product, which is the donor's own `J₀(κ_{n,k}Δt)`. See
    # `mc_ice_sources!`.
    :f_isn1, :f_isn2, :f_isn3,
    :Vi1m, :Vi1n, :Vi2m, :Vi2n, :Vi3m, :Vi3n,
    :sd_xx, :QDOT_TH, :FRIC_KE,                                       # horizontal diffusion
    :ADV, :FORCING, :KDIFF,                                           # per-slot accumulators
    :dT_nc, :dp_nc, :SATF, :QSSREL,                                   # Q_ss chain rule
    # The PROGNOSTIC VAPOR slot's own three columns: the TOTAL vertical gradient
    # ∂z ρ_v = ∂z ρ_v' + ∂z ρ̄_v the advection reads (the `nu_c_z` pattern), the
    # reconciliation nudge `VREC = (res_rho_t − ρ_v)/τ_rec`, and `VAPOR_SRC`, the summed
    # phase-change sink `−(Q̇_c + Q̇_r) − Σ_k Q̇_{i,k}` accumulated in the same loops that
    # write the microphysics history, so the census and the tendency read identical numbers.
    :rho_v_z, :VREC, :VAPOR_SRC,
    # ── ETD-AB3 stiff relaxation of the Q_ss pair (TeX §"Integration of the relaxation pair
    #    in the stiff limit"; see `relaxation_adjustment_qss!`). The two relaxations are
    #    WITHHELD from the multistep, so slot 7's `expdot` carries `N` alone and these columns
    #    carry everything the exponential propagator and the step-mean consumers need.
    #    `etd_lam` is the LIQUID conductance that survives the clip (λ's liquid half) and
    #    `etd_nl` its clipped-channel constant flux, the liquid `N` piece. Both are written
    #    BELOW the ice block rather than beside the condensation closure, because the rain
    #    channel's conductance is not final until the rain donor's realization factor is
    #    (Stage 2b; see `f_rain`). The ICE half is carried as the affine decomposition of its drive,
    #    `drive_i = etd_dep_a·Q_ss + etd_dep_b` (written by `mc_ice_sources!`, where the two
    #    drives live): `a = 1, b = 𝒟` unclipped, `a = 0, b = drive_i` clipped, `a = b = 0`
    #    above `T_0` where the channel is shut. λ's ice half is then `a·Σ_k τ_{i,k}^{-1}` and
    #    its `N` piece `−b·Σ_k τ_{i,k}^{-1}`, and the SAME two numbers re-form the step-mean
    #    deposition drive without a second classification. `etd_qbar` is the step-mean
    #    supersaturation `Q̄_ss` (Eq. qss_stepmean) and `etd_qnp1` the propagated `Q_ss^{n+1}`
    #    in SLOT units. `etd_d*` are the per-slot direct increments the step-mean rates apply,
    #    and `Qdot_bar`/`Qdot_r_bar`/`Qdep_bar` the step-mean rates themselves (the census and
    #    the equations then read the SAME numbers).
    :etd_lam, :etd_nl, :etd_dep_a, :etd_dep_b, :etd_qbar, :etd_qnp1,
    :etd_d1, :etd_d8, :etd_d9, :etd_dv, :etd_di1, :etd_di2, :etd_di3,
    :Qdot_bar, :Qdot_r_bar, :Qdep_bar,
    #    The HABIT PARTITION of the deposition increment (`ishmael_deposition_partition`) is
    #    evaluated ONCE per step, at the STEP-MEAN rate, alongside the mass it partitions —
    #    the a/c-moment sources and the sublimation number sink are the same realized
    #    increment the mass slots receive, distributed over the axes (TeX §Departures (d)).
    #    `hab*_*` stash the state-n inputs the partition needs (written by `mc_ice_sources!`
    #    at live points, zero elsewhere); `hab*_niq` is the carried-number/mass ratio the
    #    sublimation number sink is proportional to; `etd_da*/dc*/dn*` are the resulting
    #    direct increments to the a/c/n slots, siblings of `etd_di*`.
    :hab1_ani, :hab1_cni, :hab1_rni, :hab1_ds, :hab1_rb, :hab1_nim3, :hab1_vt, :hab1_cg, :hab1_niq,
    :hab2_ani, :hab2_cni, :hab2_rni, :hab2_ds, :hab2_rb, :hab2_nim3, :hab2_vt, :hab2_cg, :hab2_niq,
    :hab3_ani, :hab3_cni, :hab3_rni, :hab3_ds, :hab3_rb, :hab3_nim3, :hab3_vt, :hab3_cg, :hab3_niq,
    :hab_igr, :hab_maxsui, :hab_dv,
    :etd_da1, :etd_da2, :etd_da3, :etd_dc1, :etd_dc2, :etd_dc3, :etd_dn1, :etd_dn2, :etd_dn3,
    :s_t, :stage_zz,                                                             # moist entropy (vertical heat)
    :imp_phi_z, :imp_c_d, :imp_c_d_z, :imp_c_e, :imp_c_e_z,           # acoustic AI2* history staging
    :sd_pxi, :sd_alpha,                    # state-dependent acoustic linearization
    # ── Louis boundary layer + Smagorinsky closure (mc_boundary_layer.jl) ──
    :Kv, :K_smag, :VD_u, :VD_v, :VD_w, :QDOT_V, :VDOT_w, :VDOT_v,
    # `bl_rho_cp_z` is the Louis BL's perturbation CLOUD-DENSITY gradient ∂z ρ_c', staged by
    # `mc_driver!` only under a condensate transform (it is `rcv.f_z` otherwise). It reuses the
    # column that was the retired diagnosed-ρ_v gradient.
    :bl_s_z, :bl_rho_cp_z,
    # ── semiimplicit_adjustment_p (si_ prefix) ──
    # Deliberately NOT sharing the names above. The two functions' temporaries are not live at
    # the same time today, so sharing would work — but it would be an invisible coupling, and
    # the first person to read an mc_XZ value after the adjustment call would get silent
    # corruption. Distinct names cost ~200 KB and make that unwriteable.
    :si_p_nstar, :si_w_nstar, :si_rhod_nstar, :si_rhot_nstar, :si_et_nstar,
    :si_p_nstar_z, :si_phi_z, :si_rhs, :si_c_d, :si_c_d_z, :si_c_e, :si_c_e_z,
    # ── diffusion_timestep_mc (df_ prefix) ──
    :df_u_star, :df_w_star, :df_p_star, :df_rho_d_star, :df_rho_t_star,
    :df_E_t_star, :df_ke_star, :df_M_star,
    :df_T_star, :df_p_hPa_star, :df_drvs_dT, :df_drvs_dp,
    :df_rho_vs_star, :df_rho_v_star, :df_q_v_star, :df_q_l_star,
    :df_C_vt_star, :df_R_m_star, :df_Lv_star, :df_stp_star,
    :df_u_nstar, :df_w_nstar, :df_s_nstar, :df_u_np1, :df_w_np1,
    :df_dke, :df_dE_visc, :df_ds_t, :df_dT_h, :df_dE_h, :df_dp_h, :df_dQ_h,
    :df_rw_star, :df_rv_star, :df_rw_nstar, :df_rv_nstar, :df_rr_nstar,
    :df_rw_np1, :df_rv_np1, :df_drw, :df_drv, :df_drr, :df_dE_w,
    :df_rc_star, :df_rc_nstar, :df_rc_np1, :df_drc, :df_rho_c_star, :df_rho_r_star,
    :df_rho_liq_star,
    # The ice mirror of `df_rho_liq_star`: the raw star-state ice mass (which the water
    # PARTITION reads) and the one the thermodynamics reads (floored under
    # `condensate_floor`), plus its mixing ratio for `C_vt_star`/`stp_star`.
    :df_rho_ice_star, :df_rho_ice_t_star, :df_q_i_star,
    # ── tangential wind v (cylindrical geometries; inert columns on the XZ slice) ──
    :df_v_star, :df_v_nstar, :df_v_np1)

"""
    _allocate_mc_scratch(tile, model)

One `NamedTuple` of `kDim` work vectors per thread for the total-energy set; an empty vector
for every other equation set. Indexed by `threadid()`, which is a valid owner tag because the
column loop is `Threads.@threads :static` (same rule as `scratch_columns`).
"""
function _allocate_mc_scratch(tile::AbstractGrid, model::ModelParameters)

    uses_pressure_reference(model.equation_set) || return Vector{Nothing}(undef, 0)
    kDim = model.grid_params.kDim
    return [NamedTuple{MC_SCRATCH_SLOTS}(
                ntuple(_ -> zeros(Float64, kDim), length(MC_SCRATCH_SLOTS)))
            for _ in 1:Threads.maxthreadid()]
end

# ── Thermodynamic helpers ──────────────────────────────────────────────────────

"""
    drho_vsat_dT(Tk, p_hPa)

Temperature partial of the saturation vapor density ρ_v* = e*/(R_v T) [kg/m³/K]
at constant pressure, using the Buck (1981) saturation vapor pressure.
"""
function drho_vsat_dT(Tk, p_hPa)

    return (100.0 * sat_pressure_liquid_buck_dT(Tk, p_hPa) / (Rv * Tk)) -
           (rho_v_sat(Tk, p_hPa) / Tk)
end

"""
    drho_vsat_dp(Tk, p_hPa)

Pressure partial of the saturation vapor density ρ_v* [kg/m³ per Pa] at constant
temperature. Only the Buck (1981) pressure-enhancement factor depends on total
pressure, so this is small and positive.
"""
function drho_vsat_dp(Tk, p_hPa)

    Tc = Tk - 273.15
    B = 3.20e-6
    C = 5.9e-10
    a = 6.1121
    b = 18.729
    c = 257.87
    d = 227.3
    ew4 = a * exp((b - (Tc / d)) * Tc / (Tc + c))
    # d(rho_vs)/dp_Pa = (100 * ew4 * dfw4/dp_hPa / (Rv*T)) * (dp_hPa/dp_Pa = 1/100)
    return ew4 * (B + (C * Tc^2)) / (Rv * Tk)
end

"""
    retrieve_temperature(M, rho_d, rho_t, rho_liq)
    retrieve_temperature(M, rho_d, rho_t, rho_liq, rho_ice)

Diagnose the temperature from the prognostic variables of the total-energy set. With the
condensate prognostic the liquid density `ρ_liq = ρ_c + ρ_r` is KNOWN, so the Bryan &
Fritsch (2002) energy identity

    M = (ρ_d C_pd + ρ_w C_pv) T − ρ_liq L_v(T),    ρ_w = ρ_t − ρ_d

(where `M = p + E_t − ρ_t(v²/2 + gz)` [J/m³] is the available enthalpy density) contains no
saturation density, and `L_v(T) = L_v0 + (C_pv − C_l)(T − T_0)` is linear in T. The root is
therefore CLOSED FORM:

    T = [M + ρ_liq (L_v0 − (C_pv − C_l) T_0)] / [(ρ_d C_pd + ρ_w C_pv) − ρ_liq (C_pv − C_l)]

The denominator is `Cfactor + (C_l − C_pv) ρ_liq` with `C_l > C_pv`, hence strictly positive
for any admissible state: there is no iteration, no tolerance, no guess, and no failure mode.

(The liquid density is spelled `rho_liq`, not `rho_l`: `Springsteel.rho_l` is the bulk
density of liquid water, 1000 kg/m³, used throughout `microphysics.jl`.)

Note what is ABSENT. T depends only on `(M, ρ_d, ρ_t, ρ_liq)` — not on the vapor/cloud split,
not on `Q_ss`, and not on `ρ_vs`. That is what decouples the microphysics driver from the
thermodynamics: a `Q_ss` error is now thermodynamically inert. Latent heating still needs no
source term — condensation raises `ρ_l` at fixed `E_t` and `ρ_t`, and this expression turns
that into a warming of `δρ_l·L_v/Cfactor` automatically.

This replaces a univariate Newton iteration on the same identity carrying an extra
`ρ_vs(T,p)` term, whose clamped partition is what manufactured phantom cloud (see the file
header). The two agree to ~1e-12 K wherever the old clamp was inactive.

# The ice term

With the ice masses prognostic (`options[:ice_microphysics] = :ishmael`; see
[`ice_microphysics`](@ref)) the ice density `ρ_i = Σ_k ρ_{i,k}` is known too, and the SAME
identity closes with one more affine latent term — `L_s(T) = L_{s0} + (C_pv − C_i)(T − T_0)`,
the Kirchhoff linearization of the sublimation heat, added to the internal energy as
`−ρ_i L_s(T)`:

    T = [M + ρ_liq (L_v0 − (C_pv − C_l) T_0) + ρ_ice (L_s0 − (C_pv − C_i) T_0)]
        / [C_f − ρ_liq (C_pv − C_l) − ρ_ice (C_pv − C_i)]

`C_f = ρ_d C_pd + (ρ_t − ρ_d) C_pv` is UNCHANGED in form (`ρ_t` now includes the ice, and an
ice mass that displaces vapor moves `C_f` not at all — see the derivation in
reference/Scythe_moist_compressible.tex, Eq. T_retrieve_ice), and `C_i = 2106 > C_pv` exactly
as `C_l > C_pv`, so the denominator stays strictly positive and the root stays unique. `T`
still does not read `Q_ss`, the vapor/cloud split, or the distribution of the ice mass among
species — deposition raises `ρ_i` at fixed `E_t`, `ρ_t` and this expression turns that into
the warming `δρ_i L_s/D_i` by itself.

**The 4-argument method DELEGATES with `rho_ice = 0.0` and that is BITWISE the pre-ice code.**
The ice numerator term is `0.0 * (L_s0 − (C_pv−C_i)T_0)` = `+0.0`, added LAST (so the liquid
sum is formed first and `x + 0.0 === x` for every double except `-0.0`, which `M + ρ_liq(...)`
— an O(10⁵) J/m³ available enthalpy — cannot be); the denominator term is
`0.0 * (C_pv − C_i)` = `-0.0` SUBTRACTED, and `x - (-0.0) === x` exactly. The associativity is
therefore load-bearing, not cosmetic: `test_moist_compressible.jl` asserts the identity with
`===` over a state sweep.
"""
@inline function retrieve_temperature(M, rho_d, rho_t, rho_liq, rho_ice)

    Cfactor = (rho_d * Cpd) + ((rho_t - rho_d) * Cpv)
    return ((M + (rho_liq * (L_v0 - ((Cpv - Cl) * T_0)))) +
            (rho_ice * (L_s0 - ((Cpv - Ci) * T_0)))) /
           ((Cfactor - (rho_liq * (Cpv - Cl))) - (rho_ice * (Cpv - Ci)))
end

@doc (@doc retrieve_temperature)
@inline retrieve_temperature(M, rho_d, rho_t, rho_liq) =
    retrieve_temperature(M, rho_d, rho_t, rho_liq, 0.0)

"""
    moist_entropy_total(Tk, rho_d, q_v, q_l)
    moist_entropy_total(Tk, rho_d, q_v, q_l, q_i)

Specific moist entropy per unit dry-air mass [J/(kg·K)], INCLUDING the liquid contribution
that `entropy` omits:

    s_t = entropy(T, ρ_d, q_v) + q_l·Cl·log(T/T_0)

This is the heat control variable diffused by the turbulence scheme (the moist analogue of
Straka's dry-entropy diffusion), and the same integrand `conservation_drift` uses for the
total entropy. Because `∂s_t/∂T = C_vt/T` at fixed ρ_d and composition, a diffusive
increment δs_t maps to a heating ρ_d·T·δs_t (= ρ_d C_vt δT). In dry air it reduces to
`dry_entropy_pd(p, ρ_d)` up to a constant.

**`q_v` IS READ THROUGH `max(q_v, 0)`, AND IT HAS TO BE.** `entropy` evaluates
`q_v·R_v·ln(q_v ρ_d / ρ_v0)`, so a negative vapor mixing ratio is a `DomainError`, not a
number — and negative vapor is a RESOLUTION DIAGNOSTIC that is never clamped, so `ρ_v` does
go below zero wherever an unresolved spike undershoots. That is the tropopause: measured on
the 3-nest TC initial state (when the vapor was still the residual of two independently
fitted O(0.1 kg/m³) densities), 362 of 6750 nest-1 points between 14.4 and 17.3 km, worst
`ρ_v` = −4.5e-6 against a reference `ρ̄_v` of +1.5e-6 there. It killed the first Louis-BL call
of the first timestep. Making the vapor prognostic removes THAT mechanism — the cancellation
of large fitted densities — but not the undershoot of the vapor's own fit, so the guard
stays. The same clamp, for the same
stated reason, has always been on the diagnostics side of this (`benchmarks/common/diagnostics.jl`,
"in dry air the residual vapor sits at 0 ± roundoff, and entropy() takes log(q_v)").

This is the `condensate_floor = :diagnostic` pattern, not `clamp_water!`: nothing is written
back, no mass is converted, no latent heat is pumped, and `s_t` is only ever read as a
mixing control variable. It is also CONTINUOUS — `q ln q → 0` as `q → 0⁺`, so `entropy` has
no kink at zero and the clamped points join the dry limit smoothly. What it does NOT do is
make the negative vapor go away; that deficit is a property of fitting `ρ_t` and `ρ_d`
independently, is expressible as a constraint on neither (`benchmarks/FUTURE_WORK.md`), and
no condensate scheme reaches it.

`q_l` is deliberately NOT clamped: nothing takes its log, a negative condensate must stay
visible in the entropy budget, and under the water transforms it cannot be negative anyway.

# The ice term

Ice enters with the identical structure and for the identical reason, `s_i = C_i ln(T/T_0)`
integrated from `T ds_i = C_i dT` under the same incompressibility assumption used for the
liquid (reference/Scythe_moist_compressible.tex, Eq. entropy_ice):

    s_t = entropy(T, ρ_d, q_v) + q_l·C_l·log(T/T_0) + q_i·C_i·log(T/T_0)

The 4-argument method delegates with `q_i = 0.0`, and the added term is then
`0.0 * C_i * log(T/T_0)` = `±0.0` appended LAST, so `x + (±0.0) === x` for every double and
the pre-ice path is bitwise. `q_i` is not clamped either, for `q_l`'s reasons.
"""
function moist_entropy_total(Tk, rho_d, q_v, q_l, q_i)

    return (entropy(Tk, rho_d, max(q_v, 0.0)) + (q_l * Cl * log(Tk / T_0))) +
           (q_i * Ci * log(Tk / T_0))
end

@doc (@doc moist_entropy_total)
moist_entropy_total(Tk, rho_d, q_v, q_l) = moist_entropy_total(Tk, rho_d, q_v, q_l, 0.0)

"""
    mc_reference_diagnostics(ref_state, z) -> (s_tbar, rho_vbar, rho_vbar_z, Pxi_prof)

Consistently-RETRIEVED diagnostics of the resting reference for the vertical moist
diffusion: the moist entropy `s_tbar` and clamped vapor density `rho_vbar`, computed
through the exact pipeline `moist_compressible_XZ` runs each step (retrieval at rest,
clamped partition, `moist_entropy_total`). The reference's own `Tbar` is NOT
bit-identical to the retrieved temperature, so subtracting profiles built from it
would leave a spurious O(retrieval tolerance) perturbation that diffusion then acts
on; with these, a resting base has `s_t' ≡ 0` and `rho_v' ≡ 0` bit-for-bit and every
moist diffusive tendency vanishes exactly.

`rho_vbar` is ALSO the reference the prognostic vapor slot is carried against — the slot holds
`ρ_v' = ρ_v − ρ̄_v` with `ρ̄_v ≡ ρ̄_t − ρ̄_d − ρ̄_c` — and `rho_vbar_z` is that profile's vertical
derivative, assembled from the reference columns' own fitted derivatives rather than refitted,
so the total gradient `ρ_v'_z + ρ̄_v_z` the advection reads is the exact derived sum. Every
initializer subtracts the SAME expression, which is what makes the resting reconciliation
`δ̄ = res_rho_t − ρ_v` identically zero rather than zero to fit tolerance.
"""
function mc_reference_diagnostics(ref_state, z)

    pbar = ref_pressure(ref_state)
    rho_dbar = ref_rho_d(ref_state)
    rho_tbar = ref_rho_t(ref_state)
    rho_cbar = Springsteel.ref_rho_c(ref_state)
    E_tbar = ref_total_energy(ref_state)
    n = length(z)
    s_tbar = zeros(Float64, n)
    rho_vbar = zeros(Float64, n)
    rho_vbar_z = zeros(Float64, n)
    Pxi_prof = zeros(Float64, n)
    for k in 1:n
        p = pbar[k, 1]
        rho_d = rho_dbar[k, 1]
        rho_t = rho_tbar[k, 1]
        # The condensate is PROGNOSTIC, so the resting cloud is the reference's own
        # fitted profile (exactly 0.0 for a condensate-free base, which fits rho_c from
        # an all-zero vector; positive on a saturated base such as BF02).
        rho_c = rho_cbar[k, 1]
        # Mirror the equation set's per-point pipeline at rest (ke = 0, rho_r = 0, and
        # rho_i = 0 — there is no reference ice in any configuration or in the
        # reference-state format, which is exactly why the ice slots are TOTALS; the 4-arg
        # retrieval and entropy below are the ice ones with rho_i/q_i = 0.0, bitwise);
        # every expression must match mc_driver! bit-for-bit.
        geo = 0.5 * ((0.0 * 0.0) + (0.0 * 0.0)) + (gravity * z[k])
        M = p + (E_tbar[k, 1]) - (rho_t * geo)
        Tk = retrieve_temperature(M, rho_d, rho_t, rho_c)
        # The DERIVED reference vapor, ρ̄_v ≡ ρ̄_t − ρ̄_d − ρ̄_c. This is the profile the
        # prognostic vapor slot is carried against, so writing it in exactly this form (and
        # not from Springsteel's fitted `ref_rho_v`, which differs at fit level) is what makes
        # a resting base a discrete fixed point: `res_rho_t − ρ_v` is then identically 0.0.
        rho_v = rho_t - rho_d - rho_c
        q_v = rho_v / rho_d
        q_l = rho_c / rho_d
        s_tbar[k] = moist_entropy_total(Tk, rho_d, q_v, q_l)
        rho_vbar[k] = rho_v
        # ...and its vertical derivative, assembled from the reference columns' own fitted
        # derivative slots for the same reason: the advection's TOTAL vapor gradient is
        # `ρ_v'_z + ρ̄_v_z`, and building ρ̄_v_z by differencing the same three fits the value
        # comes from keeps the two consistent to the last bit.
        rho_vbar_z[k] = rho_tbar[k, 2] - rho_dbar[k, 2] - rho_cbar[k, 2]
        # Local reference sound speed squared γ_m(z)·p̄(z)/ρ̄_t(z) for the acoustic
        # linearization. A DOMAIN-MEAN c̄² (Springsteel's sound_speed_sq) leaves the
        # local deviation of the vertical acoustic operator EXPLICIT under AB3 —
        # the classic reference-state SI instability (Simmons, Hoskins & Burridge
        # 1978): on the Dunion sounding the ±20-25% c² deviations blow up above
        # Co_z ≈ 4.5 even though the operator-consistent scheme is stable to
        # Co_z ≈ 9-18 on a uniform-c base. The PROFILE makes the explicit acoustic
        # remainder O(perturbation) at every level.
        C_vt = Cvd + (q_v * Cvv) + (q_l * Cl)
        R_m = Rd + (q_v * Rv)
        Pxi_prof[k] = ((C_vt + R_m) / C_vt) * p / rho_t
    end
    return (s_tbar = s_tbar, rho_vbar = rho_vbar, rho_vbar_z = rho_vbar_z,
            Pxi_prof = Pxi_prof)
end

"""
    consistent_qss_reference(ref, z, column) -> PressureReferenceState

Rebuild the reference water partition (`Q_ssbar`, and on a cloudy base `rho_cbar`/`rho_vbar`)
so the RESTING state is a discrete fixed point of the equation set. Opt-in through
`options[:consistent_qss_reference]`; off by default and bitwise inert when off.

Springsteel builds its moisture profiles POINTWISE from the EOS temperature and then fits
them. The equation set never sees those pointwise values: at run time it retrieves T from the
FITTED `(p̄, Ē_t, ρ̄_t, ρ̄_c)` through [`retrieve_temperature`](@ref) and diagnoses
`ρ_v = ρ̄_t − ρ̄_d − ρ̄_c` from the fitted densities, so both differ from the pointwise
construction by the fit error. Two resting tendencies can survive that mismatch:

- `Q̇ = Q_ss·invtau/(1+Q_s)` — the phase change, wherever a gate in
  [`qss_condensation_rates`](@ref) is open;
- `QSSREL = −(Q_ss − (ρ_v − ρ_vs))/τ` — the [`qss_relaxation`](@ref) reconciliation, which is
  thermodynamically inert (T does not read `Q_ss`) but is still a nonzero tendency at rest.

Killing both means making `Q_ssbar` equal the model's OWN diagnosed supersaturation, which is
what this builds:

    ρ_c     = ρ̄_c (fitted)          M = p̄ + Ē_t − ρ̄_t·g·z        (rest: ke = 0, ρ_r = 0)
    T       = retrieve_temperature(M, ρ̄_d, ρ̄_t, ρ_c)
    Q_ssbar = (ρ̄_t − ρ̄_d − ρ_c) − ρ_v*(T, p̄)

The vapor here is the DERIVED profile `ρ̄_t − ρ̄_d − ρ̄_c`, which is exactly what the prognostic
vapor slot is carried against ([`vapor_slot`](@ref), [`mc_reference_diagnostics`](@ref)). So
this construction says `Q_ssbar = ρ̄_v − ρ_v*(T, p̄)` on the same numbers the equation set will
form at rest, and both resting tendencies — the phase change and `QSSREL` — vanish for the
prognostic vapor exactly as they did for the residual one.

**Cloud-free levels** (`ρ̄_c = 0` — a subsaturated sounding: O01, the TC) are done at that
point. `q_c = 0 < 1e-8` and `S = Q_ssbar/ρ_vs < 0` shut both condensation gates, `invtau_r = 0`
for `ρ_r = 0`, the closure returns `(0.0, 0.0)` EXACTLY, and `QSSREL` vanishes by construction.
Note how much weaker the requirement is than it used to be: the condensate is prognostic, so a
subsaturated base cannot manufacture cloud however the fit falls, and this is now a refinement
rather than the load-bearing repair it was before.

**Condensate-bearing levels** (`ρ̄_c > 0` — a saturated base: Bryan & Fritsch 2002) cannot be
fixed by `Q_ssbar` alone. The cloud gate is necessarily open there, so `Q̇` vanishes only for
`Q_ssbar = 0`, while `QSSREL` vanishes only for `Q_ssbar = ρ_v − ρ_vs`. Both hold together iff
the base is EXACTLY saturated in the model's own arithmetic, so the PARTITION is what has to
move: solve

    g(ρ_c) = ρ_c − (ρ̄_t − ρ̄_d − ρ_v*(T(ρ_c), p̄)) = 0

for `ρ_c` by Newton, using `dT/dρ_c = L_v(T)/(C_factor − ρ_c(C_pv − C_l))` (differentiate the
closed-form retrieval), then set `Q_ssbar = 0` and `ρ̄_v = ρ̄_t − ρ̄_d − ρ_c`. Fixed-point
iteration is NOT usable here: `d/dρ_c` of the map is `−∂ρ_vs/∂T · L_v/C_factor ≈ −2.5`, so the
naive iteration diverges. `ρ̄_t`, `ρ̄_d`, `p̄` and `Ē_t` are untouched, so this moves mass
between the vapor and cloud slots at fixed total density — the hydrostatic balance and the
buoyancy of the base are exactly preserved, which is the whole point of a BF02 base state.

The VALUE slots are the raw targets (not the filtered fit): the fixed point depends on them
exactly, while the derivative slots — which come from a spline fit of those values — multiply
`w ≡ 0` at rest and so cannot disturb it.

Errors per level if the intended outcome is not actually reached: a cloud-free level that is
actually supersaturated (which needs a condensate-bearing reference, not a condensing one), or
a cloudy level whose Newton solve leaves no cloud (the sounding is not saturated there).
"""
function consistent_qss_reference(ref::Springsteel.PressureReferenceState,
                                  z::AbstractVector{Float64}, column)

    pbar = ref_pressure(ref)
    rho_dbar = Springsteel.ref_rho_d(ref)
    rho_tbar = ref_rho_t(ref)
    rho_cbar = Springsteel.ref_rho_c(ref)
    E_tbar = ref_total_energy(ref)
    n = length(z)
    Q_ss_new = zeros(Float64, n)
    rho_c_new = zeros(Float64, n)
    rho_v_new = zeros(Float64, n)
    cloudy_levels = 0
    for k in 1:n
        p = pbar[k, 1]
        p_hPa = p / 100.0
        rho_d = rho_dbar[k, 1]
        rho_t = rho_tbar[k, 1]
        rho_w = rho_t - rho_d
        # Mirror the driver's per-point pipeline at rest (ke = 0, rho_r = 0).
        M = p + E_tbar[k, 1] - (rho_t * (gravity * z[k]))
        # A condensate-free reference fits rho_c from an all-zero vector, so this is
        # EXACTLY 0.0 there; a saturated base (BF02) carries a positive profile.
        rho_c = rho_cbar[k, 1]
        cloudy = rho_c > 0.0

        if cloudy
            cloudy_levels += 1
            # Newton on g(rho_c) = rho_c - (rho_w - rho_vs(T(rho_c), p)), whose derivative
            # is 1 + (drho_vs/dT)(dT/drho_c) with dT/drho_c = L_v(T)/D — see the docstring.
            Cfactor = (rho_d * Cpd) + (rho_w * Cpv)
            for _ in 1:50
                Tk = retrieve_temperature(M, rho_d, rho_t, rho_c)
                g = rho_c - (rho_w - rho_v_sat(Tk, p_hPa))
                D = Cfactor - (rho_c * (Cpv - Cl))
                gp = 1.0 + (drho_vsat_dT(Tk, p_hPa) * L_v(Tk) / D)
                drc = -g / gp
                rho_c += drc
                abs(drc) < 1.0e-15 && break
            end
            rho_c_new[k] = rho_c
            # On the saturation manifold BOTH the phase change and the reconciliation
            # vanish; Q_ssbar = 0 exactly rather than the ~1e-19 the solve would leave.
            Q_ss_new[k] = 0.0
        else
            rho_c_new[k] = rho_c            # exactly 0.0
            Tk = retrieve_temperature(M, rho_d, rho_t, rho_c)
            Q_ss_new[k] = (rho_w - rho_c) - rho_v_sat(Tk, p_hPa)
        end
        rho_v_new[k] = rho_w - rho_c_new[k]

        # VERIFY the run-time outcome through the driver's own expressions, rather than
        # assuming the algebra above reached it.
        Tk_run = retrieve_temperature(M, rho_d, rho_t, rho_c_new[k])
        rho_vs = rho_v_sat(Tk_run, p_hPa)
        q_c = rho_c_new[k] / rho_d
        S = Q_ss_new[k] / rho_vs
        # QSSREL is inert but must still vanish for a true discrete fixed point; it is
        # scaled by rho_vs so the tolerance means "a relative supersaturation of 1e-10".
        rel = (Q_ss_new[k] - (rho_v_new[k] - rho_vs)) / rho_vs
        if cloudy
            # What must not happen is losing the base cloud, which is what makes a
            # saturated base neutrally buoyant.
            rho_c_new[k] > 0.0 || error("consistent_qss_reference: the saturation solve " *
                "leaves no cloud at level $k (z = $(z[k]) m): rho_cbar = " *
                "$(rho_cbar[k, 1]) kg/m^3 in the reference but the saturated partition " *
                "gives rho_c = $(rho_c_new[k]) at T = $Tk_run K, p = $p_hPa hPa. The " *
                "reference's (p, rho_d, rho_t, E_t) are not consistent with a " *
                "saturated state.")
            abs(rel) <= 1.0e-10 || error("consistent_qss_reference: the saturation " *
                "solve did not converge at level $k (z = $(z[k]) m): rho_v - rho_vs = " *
                "$(rho_v_new[k] - rho_vs) kg/m^3 (relative $rel), so the Q_ss " *
                "reconciliation would fire on the resting base.")
        else
            # Exactly the two gates in `qss_condensation_rates`; failing either leaves
            # invtau_c > 0 and the reference condenses at rest.
            (q_c <= 1.0e-8 && S <= 1.0e-4) || error("consistent_qss_reference: the " *
                "reference still condenses at level $k (z = $(z[k]) m): q_c = $q_c " *
                "(needs <= 1e-8), S = $S (needs <= 1e-4), T = $Tk_run K, " *
                "p = $p_hPa hPa. A saturated base state needs a condensate-bearing " *
                "reference (rho_c > 0), not a condensing one.")
        end
    end

    # Value slots EXACT (the fixed point depends on them bit-for-bit); derivative slots
    # from a spline fit of those values (they multiply w == 0 at rest).
    fit3 = function (vals)
        prof = zeros(Float64, n, 3)
        column.uMish[:] .= vals
        Btransform!(column)
        Atransform!(column)
        prof[:, 1] .= vals
        prof[:, 2] .= Ixtransform(column)
        prof[:, 3] .= Ixxtransform(column)
        return prof
    end

    Q_ssbar = fit3(Q_ss_new)
    # A cloud-free base leaves the partition untouched, so its profiles stay the objects
    # they already were — no refit, hence no chance of perturbing a working reference.
    rho_cbar_out = cloudy_levels == 0 ? ref.rho_cbar : fit3(rho_c_new)
    rho_vbar_out = cloudy_levels == 0 ? ref.rho_vbar : fit3(rho_v_new)

    return Springsteel.PressureReferenceState(
        ref.pbar, ref.rho_dbar, rho_vbar_out, rho_cbar_out, ref.rho_tbar,
        ref.Tbar, ref.E_tbar, Q_ssbar, ref.sound_speed_sq)
end

"""
    dry_entropy_pd(p_Pa, rho_d)

Dry-air specific entropy written as an explicit function of the prognostic pressure and
dry-air density, `s_d = C_vd·log(p) − C_pd·log(ρ_d)` (an additive constant is dropped — it
does not affect the diffusion operator). Equal to `moist_entropy_total(T, ρ_d, 0, 0)` up to
that constant via the dry EOS `T = p/(R_d ρ_d)`. Used for the dry-exact horizontal heat
diffusion, whose Laplacian is the two-term chain rule

    s_d,x  = C_vd·p_x/p − C_pd·ρ_d,x/ρ_d
    s_d,xx = C_vd·(p_xx/p − p_x²/p²) − C_pd·(ρ_d,xx/ρ_d − ρ_d,x²/ρ_d²)

on the existing `p`/`ρ_d` derivative slots (no transform of a diagnosed field — see
reference/moist_compressible_diffusion_plan.md). The moist correction terms are deferred;
see reference/moist_compressible_diffusion_handoff.md.
"""
function dry_entropy_pd(p_Pa, rho_d)

    return (Cvd * log(p_Pa)) - (Cpd * log(rho_d))
end

"""
    Q_s_energy(Tk, p_Pa, rho_d, q_v, q_l)
    Q_s_energy(Tk, p_Pa, rho_d, q_v, q_l, q_i)

Energy-consistent psychrometric factor (dimensionless) for the supersaturation
relaxation, built from the condensation-induced temperature and pressure tendencies
of the total-energy equation set (density form, no ln(H) term):

    Q_s = [ ∂ρ_vs/∂T (L_v − R_v T)/ρ_d + ∂ρ_vs/∂p R_m (L_v − R_v C_pt T / R_m) ] / C_vt

The (1 + Q_s) factor cancels between the condensation rate and the saturation
chain-rule terms, leaving −Q_ss/τ as the net supersaturation forcing.

Ice reaches this only through the MIXTURE HEAT CAPACITY, `C_vt = C_vd + q_v C_vv + q_l C_l +
q_i C_i` (reference/Scythe_moist_compressible.tex, Eq. mixture_C): a condensed phase performs
no expansion work, so ice enters `C_vt` and `C_pt` exactly as liquid does, `R_m` is untouched
because ice exerts no partial pressure, and the identity `C_pt = C_vt + R_m` — which is what
collapses the pressure equation's coefficients — survives unchanged. The 5-argument method
delegates with `q_i = 0.0`; `+ 0.0*C_i` appended last is bitwise the pre-ice `C_vt`.

The OVER-ICE psychrometric factor of the deposition channel is a different quantity (it reads
`∂ρ_i*/∂T` and `L_s`) and is not this function; it arrives with the deposition rates.
"""
function Q_s_energy(Tk, p_Pa, rho_d, q_v, q_l, q_i)

    C_vt = Cvd + (q_v * Cvv) + (q_l * Cl) + (q_i * Ci)
    R_m = Rd + (q_v * Rv)
    C_pt = C_vt + R_m
    p_hPa = p_Pa / 100.0
    Lv = L_v(Tk)
    Q_s = ((drho_vsat_dT(Tk, p_hPa) * (Lv - (Rv * Tk)) / rho_d) +
           (drho_vsat_dp(Tk, p_hPa) * R_m * (Lv - (Rv * C_pt * Tk / R_m)))) / C_vt
    return Q_s
end

@doc (@doc Q_s_energy)
Q_s_energy(Tk, p_Pa, rho_d, q_v, q_l) = Q_s_energy(Tk, p_Pa, rho_d, q_v, q_l, 0.0)

"""
    Q_s_energy_ice(Tk, p_Pa, rho_d, q_v, q_l, q_i)

The ICE psychrometric factor `𝒬_{s,i}` (dimensionless), the deposition channel's counterpart
of [`Q_s_energy`](@ref) (reference/Scythe_moist_compressible.tex, Eq. Qs_ice):

    𝒬_{s,i} = [ ∂ρ_vs/∂T (L_s − R_v T)/ρ_d + ∂ρ_vs/∂p R_m (L_s − R_v C_pt T / R_m) ] / C_vt

It is `Q_s_energy` with `L_v → L_s` **in both latent-heat slots and nowhere else**.

**The two saturation derivatives stay OVER WATER.** `∂ρ_vs/∂T` and `∂ρ_vs/∂p` here are the
derivatives of `ρ_v*`, the plane-water saturation density, in both factors — this is NOT
`Q_s_energy` re-evaluated over ice. The factor measures how far the deposition heating moves
the saturation *that `Q_ss` is defined against*, and `Q_ss ≡ ρ_v − ρ_v*` is defined over water
at all temperatures (Eq. qss_ice_shift). The ice enters only through the latent heat that does
the moving. Substituting `∂ρ_i*/∂T` here would break the exact cancellation of `(1 + 𝒬_{s,i})`
between the deposition rate of Eq. dep_rate and the chain-rule terms of Eq. Qss_ice, which is
what makes the supersaturation forcing energy-consistent.

`C_vt` and `C_pt` are the ice-inclusive mixture capacities of Eq. mixture_C, exactly as in
`Q_s_energy`; `R_m` is untouched, since ice exerts no partial pressure.
"""
function Q_s_energy_ice(Tk, p_Pa, rho_d, q_v, q_l, q_i)

    C_vt = Cvd + (q_v * Cvv) + (q_l * Cl) + (q_i * Ci)
    R_m = Rd + (q_v * Rv)
    C_pt = C_vt + R_m
    p_hPa = p_Pa / 100.0
    Ls = L_s(Tk)
    return ((drho_vsat_dT(Tk, p_hPa) * (Ls - (Rv * Tk)) / rho_d) +
            (drho_vsat_dp(Tk, p_hPa) * R_m * (Ls - (Rv * C_pt * Tk / R_m)))) / C_vt
end

"""
    ice_supersaturation_gap(Tk, p_hPa) -> 𝒟

`𝒟(T,p) = max(ρ_v*(T,p) − ρ_{v,i}*(T,p), 0)` [kg/m³], the offset between the water and ice
saturation densities (reference/Scythe_moist_compressible.tex, Eq. Dwi).

This is what makes the over-ice supersaturation an ALGEBRAIC function of the one prognostic
`Q_ss` rather than a second prognostic: `ρ_v − ρ_{v,i}* = Q_ss + 𝒟` exactly (Eq. qss_ice_shift).
It vanishes at the triple point, is positive below it because the ice branch of
Clausius-Clapeyron is the steeper one, peaks near −12 °C, and the `max` carries it to zero
above `T_0` where the ice cannot persist.

BOTH branches come from the same Buck (1981) family with the same dry-air enhancement factor
(`rho_v_sat` and `rho_i_sat` in Springsteel), which is the whole point: `𝒟` is a small
difference of two large saturation densities and is only as good as the internal consistency
of the two formulations (TeX §"Departures", (c)).
"""
@inline ice_supersaturation_gap(Tk, p_hPa) =
    max(rho_v_sat(Tk, p_hPa) - rho_i_sat(Tk, p_hPa), 0.0)

"""
    ice_deposition_drive(Q_ss, rho_v, Tk, p_hPa) -> Q_ss_drive_i

The DEPOSITION drive [kg/m³], the ice mirror of the liquid clip in
[`qss_condensation_rates`](@ref):

    Q_ss_drive_i = min(Q_ss + 𝒟, max(ρ_v, 0) − ρ_{v,i}*)

In the continuum `Q_ss + 𝒟 ≡ ρ_v − ρ_{v,i}*` (Eq. qss_ice_shift) and the `min` never binds.
Discretely `Q_ss` is an independent prognostic that can detach from the density budget, and
this is the instantaneous, `Δt`-free form of the same reconciliation `qss_relaxation` applies
on the `tau_qss` timescale — the identical argument, term for term, that the liquid clip
rests on, applied to the identical vapor reservoir. Both phases must read the same clip or
the shared-reservoir competition of Eq. wbf_qs is arbitrated twice.

`max(ρ_v, 0)` is the same vapor floor for the same reason: a NEGATIVE retrieved `ρ_v` is a
partition error, not a physical state, and must not set a sublimation rate whose magnitude is
the size of that error. The floor puts the drive at `−ρ_{v,i}*`, the vapor-free-air maximum.

The expression is SIGNED and needs no case distinction: where it is negative the air is
subsaturated with respect to ice and the same formula returns sublimation (Eq. dep_rate).
"""
@inline function ice_deposition_drive(Q_ss, rho_v, Tk, p_hPa)

    rho_vs_i = rho_i_sat(Tk, p_hPa)
    return min(Q_ss + max(rho_v_sat(Tk, p_hPa) - rho_vs_i, 0.0),
               max(rho_v, 0.0) - rho_vs_i)
end

"""
    qss_condensation_rate(Q_ss, rho_v, rho_c, rho_d, Tk, p_hPa, Q_s, ts, max_N_c=100.0)

Cloud condensation/evaporation rate [kg/m³/s] from the prognostic supersaturation
density, with the droplet-growth timescale of [`q_condensation`](@ref) (Twomey-type
nucleation, minimum droplet radius) and the energy-consistent psychrometric factor:

    Q̇_cond = Q_ss (1/τ) / (1 + Q_s)

Subsaturated cloud-free air returns zero: there are no droplets to evaporate.
Supersaturated air nucleates (`S > 1e-4`) and condenses.

No depletion cap of any kind is applied, so the rate is not a function of `ts` — `ts` is
inert in the signature, kept because callers pass it positionally and the ice work will read
it. `rho_v` IS read: it clips the drive to `min(Q_ss, ρ_v − ρ_v*)`, which is an identity in
the continuum and is what stops a column whose prognostic `Q_ss` has detached from the density
budget condensing vapor that is not there. See [`qss_condensation_rates`](@ref).
"""
qss_condensation_rate(Q_ss, rho_v, rho_c, rho_d, Tk, p_hPa, Q_s, ts, max_N_c=100.0) =
    qss_condensation_rates(Q_ss, rho_v, rho_c, 0.0, rho_d, Tk, p_hPa, Q_s, ts, 0.0,
                           max_N_c)[1]

"""
    qss_condensation_rates(Q_ss, rho_v, rho_c, rho_r, rho_d, Tk, p_hPa, Q_s, ts, N_r,
                           max_N_c=100.0; N_0=0.0, rain_2m=false, n_r_density=0.0)
        -> (Qdot_c, Qdot_r, invtau_c, invtau_r)

Two-category condensation/evaporation rates [kg/m³/s] from the generalized
supersaturation relaxation `1/τ = 1/τ_c + 1/τ_r` (see
reference/Scythe_moist_compressible.tex): the total rate `Q_ss (1/τ)/(1+Q_s)` splits
between the cloud and rain channels in proportion to their inverse timescales. Rain
gains a (small, `N_r`-controlled) share of supersaturated condensation ONLY where
cloud coexists (`q_c > 1e-8`, the cloud channel's own existence threshold): the
physical pathway to rain is condensation → cloud → autoconversion, and an ungated
rain channel grows rain from arbitrarily small seeds in cloud-free supersaturated
air (rate ∝ `rho_r^{1/3}` under the monodisperse fixed-`N_r` closure, non-Lipschitz
at zero). The rain-channel timescale is [`invtau_rain`](@ref) by default; the
keyword `N_0 > 0` [m⁻⁴] selects the exponential Marshall-Palmer closure
[`invtau_rain_mp`](@ref) instead (`N_r` is then unused), and `rain_2m = true`
with `n_r_density` [#/m³] selects the two-moment [`invtau_rain_2m`](@ref), whose
DSD intercept comes from the prognostic mass/number pair rather than being
prescribed (`N_r` and `N_0` are then both unused; see [`rain_moments`](@ref)).
All three leave the split arithmetic and the cloud gate untouched. Rain
evaporation in subsaturated air is unconditional, so no separate
rain-evaporation parameterization (O01's `Q_evap`) is needed. The ventilation
enhancement lives inside [`invtau_rain`](@ref).

# No limiters

The rates are PURE. This function used to floor cloud evaporation at `−ρ_c/ts`, rain
evaporation at `−ρ_r/ts`, and rescale both channels when the combined condensation exceeded
`+ρ_v/ts` (later, the AB3-exact versions of the same three budgets, sized in `mc_driver!` by
[`_ab3_sink_bound`](@ref)). All three are gone.

The reason is stated in reference/Scythe_moist_compressible.tex, "Departures from the ISHMAEL
implementation" §(b): a cap of the form `Q̇ ≥ −ρ/Δt` makes the RATE — and therefore the
converged solution — a function of the time step, so the scheme is not consistent in the sense
that refining `Δt` approaches the differential equation, because the equation being solved
changes with `Δt`. The same argument that rejects ISHMAEL's semi-analytic step-mean growth
rate rejects these.

What replaces them:

  * robustness against a transient negative condensate comes from the CONTROL-VARIABLE
    transforms ([`condensate_transform_mode`](@ref), [`rain_transform_mode`](@ref)), which
    bound the recovered density below by construction rather than by repairing a rate, and
    from the prognostic-supersaturation recovery — a point driven past zero is subsaturated
    on the next step and the closure stops removing vapor from it on its own;
  * under-resolution of a stiff relaxation is REPORTED, by [`mc_stiffness_census!`](@ref),
    which accumulates `max ts/τ` per channel and counts the gridpoints past `ts/τ = 1`. A run
    with a large census is a run whose time step is wrong, and that is a statement for the
    diagnostics to make, not one for the rates to absorb silently.

The two channel inverse timescales are returned alongside the rates precisely so the census
can be taken on the rates that were actually used, at no extra cost.

`ts` no longer enters the returned rate at all. `rho_v` does — but through the DRIVE, not
through a `Δt`-sized budget. See below.

# The clipped drive (PROVISIONAL — pending author ratification)

The closure is driven by

    Q_ss_drive = min(Q_ss, max(ρ_v, 0) − ρ_v*)

and `Q_ss_drive`, never the raw `Q_ss`, sets the saturation ratio `S` that gates Twomey
nucleation and the droplet number, the rate `Q̇ = Q_ss_drive·(1/τ)/(1+Q_s)` and its split, and
the sign test that gates the rain channel. The `Q_ss` PROGNOSTIC EQUATION and
[`qss_relaxation`](@ref) are untouched — the clip lives only inside this rate closure.

In the continuum this is an IDENTITY: `Q_ss ≡ ρ_v − ρ_v*`, so the `min` never binds and the
clip is invisible. Discretely `Q_ss` is carried as an independent prognostic and can DETACH
from the density budget, and the clip is the INSTANTANEOUS form of exactly the reconciliation
`qss_relaxation` already applies on the `tau_qss` timescale. It is `ts`-free and depends on
the state alone, so it is not a depletion cap and does not reintroduce the defect §(b)
rejects: refining `Δt` still approaches the same differential equation.

`ρ_v` here is the BLENDED vapor — the same variable the removed `ceil_v` bounded — which
minimizes the behaviour change relative to the code this replaces.

Why it is needed, measured 2026-08-12 by removing the three caps and restoring one at a time:

  * removing `floor_c` and `floor_r` breaks NOTHING;
  * removing `ceil_v` breaks two gates — "Q_ss relaxation is thermodynamically inert in dry
    air", and the axisym `exact_si` ceiling gate, which detonates to a NEGATIVE TEMPERATURE
    inside 300 steps on a resting DRY base.

`ρ_v` appeared ONLY in the ceiling, so with it gone nothing coupled the closure to the vapor
actually present, and the nucleation branch fired on a detached `S = Q_ss/ρ_v*` to condense
vapor that was not there. The dry bases are the worst case precisely because `ρ_v*` is tiny
(1.7e−6 kg/m³ at 195 K, 8.3e−4 at 250 K), so a noise-level `Q_ss'` reads as a large
supersaturation. The ceiling's `Δt` dependence was incidental; its CONTENT — "you cannot
condense vapor that is not there" — is physical and belongs in the closure. This is that
content, written without `Δt`.

**`S` must come from the clipped drive too**, which is why the clip is applied before `S` and
not just at the rate: with `S` computed from the raw `Q_ss`, dry air with a positive `Q_ss'`
still NUCLEATES (raw `S > 1e-4` opens the Twomey branch and sets `invtau_c > 0`), and the
clipped, now-negative drive then evaporates the cloud that does not exist and manufactures
vapor — the same hole with the sign reversed. Taking `S` from the drive shuts the branch and
leaves dry air exactly inert.

An existence gate (`ρ_v > 0`) was considered and rejected: the rate does not scale with `ρ_v`,
so 1e−12 kg/m³ of vapor would still admit the full rate.

**The EVAPORATION side is bounded by the vapor floor.** Where the retrieved vapor is NEGATIVE
— which happens (the O01 quick run reaches `ρ_v = −7.2e−4 kg/m³` at ~750 points), it is a
PARTITION ERROR, not a physical state — the `max(ρ_v, 0)` inside the clip floors the drive at
`−ρ_v*`, the maximum-dryness (vapor-free air) evaporation drive. Without the floor the drive
would be `ρ_v − ρ_v* < −ρ_v*`: evaporation at a rate set by the size of the partition error,
which nothing bounds (a seeded `ρ_c' = 0.5 kg/m³` state put `ρ_v ≈ −0.5` and drove slots 7 and
9 to O(1e3) kg/m³/s through exactly that branch before the floor was added). The floor is the
evaporation-side mirror of the condensation clip — "air cannot be drier than vapor-free" next
to "you cannot condense vapor that is not there" — and is equally `ts`-free and state-only.
A prognostic `Q_ss` below `−ρ_v*` still passes through the `min` unfloored: the prognostic is
advanced by the smooth multistep integrator and is not the pathology this floor addresses.

The vapor's depletion remains CENSUSED (`:d_v_*` in [`water_depletion_probe!`](@ref)), which
is where a failure of this argument would show up.

[`water_depletion_probe!`](@ref) still compares the realized rates against the AB3-exact
bounds and reports where the old caps WOULD have bound; that is a measurement, and nothing in
the RHS path reads it.
"""
function qss_condensation_rates(Q_ss, rho_v, rho_c, rho_r, rho_d, Tk, p_hPa, Q_s, ts,
                                N_r, max_N_c=100.0; N_0=0.0,
                                rain_2m::Bool=false, n_r_density=0.0)

    rho_vs = rho_v_sat(Tk, p_hPa)
    # THE DRIVE, clipped to the vapor actually present. In the continuum Q_ss IS rho_v - rho_vs
    # and this is an identity; discretely the prognostic Q_ss can detach from the density
    # budget, and this is the instantaneous form of the reconciliation `qss_relaxation` applies
    # on the tau_qss timescale. `ts`-free and state-dependent only. The vapor is floored at
    # zero inside the clip: air cannot be drier than vapor-free, so a NEGATIVE retrieved rho_v
    # (a partition error) must not set the evaporation rate. See the docstring.
    Q_ss_drive = min(Q_ss, max(rho_v, 0.0) - rho_vs)
    S = Q_ss_drive / rho_vs              # supersaturation (ratio - 1)
    q_c = max(rho_c, 0.0) / rho_d

    # Cloud channel: droplet number and radius logic mirrors q_condensation
    invtau_c = 0.0
    N_c = max_N_c
    r_c = cloud_droplet_radius(N_c, q_c, rho_d)
    if q_c > 1.0e-8
        if S < 0.0 && r_c < 1.0
            # Evaporating small droplets: count the 1-micron drops available
            r_c = 1.0
            N_c = cloud_droplet_number(r_c, q_c, rho_d)
            if N_c < 1.0
                N_c = 0.0
            end
        end
        if N_c > 0.0 && r_c > 0.0
            invtau_c = invtau_condensation(Tk, p_hPa, N_c, r_c)
        end
    elseif S > 1.0e-4
        # Nucleation: linear interpolation of the Twomey relationship
        if r_c < 1.0
            r_c = 1.0
            N_c = min(1.0e4 * N_c * S, max_N_c)
        end
        if N_c > 0.0 && r_c > 0.0
            invtau_c = invtau_condensation(Tk, p_hPa, N_c, r_c)
        end
    end
    # (no cloud and not supersaturated: invtau_c stays 0 — rain may still evaporate)

    # Rain CONDENSATION is gated on cloud presence (same q_c > 1e-8 existence
    # threshold as the cloud channel): the physical pathway to rain is condensation
    # -> cloud -> autoconversion, and in cloud-free supersaturated air the ungated
    # channel grows rain from arbitrarily small seeds in finite time (the rate is
    # ∝ rho_r^{1/3} under the fixed-N_r monodisperse closure, non-Lipschitz at zero
    # — the O01 spurious-blob pathway). Evaporation (drive <= 0) is unconditional.
    # The channel timescale is monodisperse fixed-N_r by default; N_0 > 0 selects the
    # exponential (Marshall-Palmer) DSD closure. Both are non-Lipschitz at zero rain
    # (rho_r^{1/3} and rho_r^{1/2} respectively), so the gate applies to either.
    # The sign test reads the CLIPPED drive, like every other use: a column whose Q_ss has
    # detached upward is not condensing onto rain either.
    # `rain_2m` (options[:rain_moments] == 2) replaces the prescribed-DSD timescales with
    # `invtau_rain_2m`, whose intercept comes from the prognostic (rho_r, n_r) pair. It is
    # tested FIRST and defaults to `false`, so the single-moment arithmetic below — and the
    # gate above it, which is unchanged and applies to all three closures — is bit-identical.
    invtau_r = (Q_ss_drive > 0.0 && q_c <= 1.0e-8) ? 0.0 :
               rain_2m ? invtau_rain_2m(Tk, p_hPa, rho_r, n_r_density, rho_d) :
               (N_0 > 0.0 ? invtau_rain_mp(Tk, p_hPa, N_0, rho_r, rho_d) :
                            invtau_rain(Tk, p_hPa, N_r, rho_r))
    invtau = invtau_c + invtau_r
    if invtau == 0.0
        return (0.0, 0.0, 0.0, 0.0)
    end

    Qdot = Q_ss_drive * invtau / (1.0 + Q_s)
    # An inactive channel gets an exact 0.0 (Qdot * 0.0 would be -0.0 for evaporation);
    # a lone active channel gets Qdot exactly (invtau/invtau == 1.0), which keeps the
    # single-category delegate bit-identical.
    Qdot_c = invtau_c == 0.0 ? 0.0 : Qdot * (invtau_c / invtau)
    Qdot_r = invtau_r == 0.0 ? 0.0 : Qdot * (invtau_r / invtau)

    # No cap of any kind: the rate is the physics, and `ts` does not appear in it. The only
    # bound anywhere in this function is the DRIVE clip at the top, which is `ts`-free.
    # The channel timescales come back with the rates so `mc_stiffness_census!` can measure
    # the stiffness the step is actually being asked to integrate.
    return (Qdot_c, Qdot_r, invtau_c, invtau_r)
end

import Springsteel: ref_pressure, ref_rho_t, ref_total_energy, ref_qss

# Canonical slot order for the total-energy set (u=4, w=5 in the shared-machinery
# positions used by the other XZ sets). rho_c is APPENDED at 9 rather than placed
# next to rho_r: slots 1-8 appear as hardcoded literals throughout the kernel, the
# acoustic solvers and mc_boundary_layer.jl, and appending keeps every one valid
# (the same rule MC_VARS_CYL follows for v — see the comment there).
#
# rho_v is appended at 10 for the same reason, and it is in the CONSTANT rather than in
# `mc_var_names`'s optional block because it is NOT optional: the vapor is prognostic in
# every configuration of this set (see the file header). Every list built by enumerating
# this constant therefore picks it up with no per-caller edit, and the appended optional
# slots (n_r, then the twelve ice moments) simply shift one further out.
const MC_VARS = ["p", "rho_d", "rho_t", "u", "w", "E_t", "Q_ss", "rho_r", "rho_c",
                 "rho_v"]

# ── Rigid-wall pressure compatibility condition ───────────────────────────────

"""
    mc_wall_bc_active(grid) -> Bool

True if `p` declares the inhomogeneous Neumann condition (`CubicBSpline.R1T1X`)
at either vertical wall, i.e. this grid wants [`update_mc_wall_bc!`](@ref) called
before every fit. Cheap enough for the per-step path: two `haskey` lookups.
"""
function mc_wall_bc_active(grid)
    grid.kbasis isa Springsteel.SplineBasisArray || return false
    haskey(grid.params.vars, "p") || return false
    kcol = grid.kbasis.data[grid.params.vars["p"]]
    return haskey(kcol.params.BCL, "X1") || haskey(kcol.params.BCR, "X1")
end

"""
    update_mc_wall_bc!(grid, src)

Install the exact rigid-wall compatibility condition on `p'` before a fit.

`w` is Dirichlet at both vertical walls, so `w ≡ 0` there for ALL time. Along the
wall that also kills the advection terms (`∂w/∂r = 0` because `w` vanishes at every
radius, and `w ∂w/∂z = 0`), so the vertical momentum equation collapses to an
identity rather than an evolution equation:

    ∂p'/∂z |_wall  =  -g ρ_t' |_wall

This is NOT an assumption of hydrostatic balance in the interior — only at the two
rigid walls, where it is exact. (If vertical diffusion or a boundary-layer scheme is
ever given a nonzero `w` tendency AT the wall, its contribution `ρ_t · ∂w/∂t|_wall`
belongs on the right-hand side here; with `Kvdiff = 0` and `w` Dirichlet there is
none, and the assertion in `mc_wall_bc_requires_no_w_forcing` guards the assumption.)

Why it matters: a homogeneous `NeumannBC` is the special case `ρ_t' = 0`, which a
balanced vortex violates precisely where its surface pressure deficit lives — that
was the drain of `reference/HANDOFF_2026-07-20.md`. `SecondDerivativeBC` leaves `∂p'/∂z`
free instead, which restores the balance but costs a factor ~2.7 in stable timestep,
because the semi-implicit acoustic solve eliminates `w` (φ = ρ̄_t w is Dirichlet in
the Helmholtz solve) and therefore cannot see a wall derivative the refit injects —
see `reference/SI_WALL_BC_CEILING.md`. R1T1X resolves the conflict: the admissible subspace
stays R1T1's, so the solve stays operator-consistent and the ceiling stays high,
while the affine `ahat` offset carries the nonzero derivative the balance needs.

`src` is a `(npts, nvars, nderiv)` physical array carrying the fitted field AND its
z-derivatives (slots 1, 4, 5) — normally `grid.physical`. `ρ_t'` at the wall is a
second-order Taylor step off the nearest mish point, since the mish never lands on
the boundary itself. A 2-D `src` is accepted for tests but degrades to the value
alone.

`relax` ∈ (0, 1] is the fraction of the way the stored derivative moves toward the
target on this call, i.e. a first-order filter with timescale `Δt/relax`. **It must
be well below 1 in a run.** With `relax = 1` the wall derivative is recomputed from
scratch every acoustic step, which closes a feedback loop — `∂p'/∂z|_wall` sets the
wall pressure, the acoustic solve moves `ρ_t'` at the wall, which resets
`∂p'/∂z|_wall` — and it is unstable: the stored value alternates sign step to step
and the column goes non-finite in a few hundred steps (measured). The affine-offset
argument for R1T1X's stability holds for a FIXED `ahat`; a state-dependent one adds
an explicit path that has to be kept off the acoustic timescale. The physical
justification is the same fact: the wall gradient is a property of the BALANCED
vortex and evolves on hours, so filtering it over ~minutes loses nothing real.
`relax = 1` remains the right choice for a one-shot initialization, where there is
no loop to close.
"""
# Radial spline used only to differentiate the wall profile: same knots as the
# variable's i-basis but NATURAL boundary conditions, so neither a wall Dirichlet
# nor a nesting R3X payload can bend the wall data. Cached per (domain, cells);
# Springsteel caches the factorised template underneath, so a miss is cheap.
const _WALL_DERIV_SPLINES = Dict{NTuple{3,Any}, Springsteel.CubicBSpline.Spline1D}()
const _WALL_DERIV_LOCK = ReentrantLock()


function _wall_deriv_spline(grid, v::Int)
    isp = grid.ibasis.data[1, v]
    sp = isp.params
    key = (sp.xmin, sp.xmax, sp.num_cells)
    lock(_WALL_DERIV_LOCK) do
        get!(_WALL_DERIV_SPLINES, key) do
            Springsteel.CubicBSpline.Spline1D(
                Springsteel.CubicBSpline.SplineParameters(
                    xmin = sp.xmin, xmax = sp.xmax, num_cells = sp.num_cells,
                    BCL = Springsteel.CubicBSpline.R0,
                    BCR = Springsteel.CubicBSpline.R0))
        end
    end
end

function update_mc_wall_bc!(grid, src::AbstractArray; relax::Float64 = 1.0)
    gp = grid.params
    kDim = gp.kDim
    iDim = gp.iDim
    vars = gp.vars
    p_i = vars["p"]
    rhot_i = vars["rho_t"]
    kcol = grid.kbasis.data[p_i]
    wall_du = grid.kbasis.wall_du

    # rho_t' at the wall, estimated as the MEAN over the boundary cell (the three
    # nearest mish points).
    #
    # NOT an extrapolation to the wall itself, which is what this did first — both
    # a 3-point Lagrange fit through the mish values and a second-order Taylor step
    # off the fitted derivatives. Both blow up, and for the same reason: they weight
    # the near-wall curvature by d^2/2 ~ 1.6e3 m^2, which amplifies whatever
    # grid-scale content sits in rho_t' near the boundary. That amplified signal
    # becomes p's wall derivative, which drives the acoustic mode, which enlarges
    # the near-wall grid-scale content — a loop with gain > 1 that no amount of time
    # relaxation suppresses, because the TARGET is what is growing. Measured: the
    # resting column goes non-finite in ~30 steps with either extrapolation, and is
    # quiet indefinitely with `ahat` pinned to zero, so the feedback is the whole of
    # the instability and the R1T1X plumbing is exonerated.
    #
    # A cell mean is a smoother rather than an amplifier. It is biased by O(dz) times
    # the true gradient, which is the right trade: the wall condition needs to track
    # the BALANCED state's wall value, not resolve a grid-scale feature.
    if haskey(kcol.params.BCL, "X1")
        @inbounds for r in 1:iDim
            n = (r - 1) * kDim + 1
            rt = (src[n, rhot_i, 1] + src[n+1, rhot_i, 1] + src[n+2, rhot_i, 1]) / 3
            old = wall_du[r, p_i, 1, 1]
            wall_du[r, p_i, 1, 1] = old + relax * ((-gravity * rt) - old)
        end
    end
    if haskey(kcol.params.BCR, "X1")
        @inbounds for r in 1:iDim
            n = r * kDim
            rt = (src[n, rhot_i, 1] + src[n-1, rhot_i, 1] + src[n-2, rhot_i, 1]) / 3
            old = wall_du[r, p_i, 2, 1]
            wall_du[r, p_i, 2, 1] = old + relax * ((-gravity * rt) - old)
        end
    end

    # Radial derivatives of the wall data (levels 2 and 3). The 2-D transform
    # evaluates the i-derivative BEFORE fitting in k, so its dr = 1 / dr = 2
    # passes need d(wall)/dr and d2(wall)/dr2 as their `ahat`; feeding them
    # level 1 asserts dg/dr = g and wrecks the radial pressure gradient in the
    # boundary cell.
    #
    # Differentiated through a CLEAN natural-BC spline, NOT the variable's own
    # i-basis. A nested child patch carries an R3X junction condition whose
    # `ahat` holds the PARENT's payload; fitting the wall profile through that
    # spline would inject the parent's pressure data into the wall derivative and
    # constrain the profile to the junction trio. Both nested hold tests died on
    # nest 3, whose inner edge is exactly such a junction.
    isp = _wall_deriv_spline(grid, p_i)
    for s in 1:2
        (s == 1 ? haskey(kcol.params.BCL, "X1") : haskey(kcol.params.BCR, "X1")) || continue
        @inbounds for r in 1:iDim
            isp.uMish[r] = wall_du[r, p_i, s, 1]
        end
        Springsteel.CubicBSpline.SBtransform!(isp)
        Springsteel.CubicBSpline.SAtransform!(isp)
        d1 = Springsteel.CubicBSpline.SIxtransform(isp)
        d2 = Springsteel.CubicBSpline.SIxxtransform(isp)
        @inbounds for r in 1:iDim
            wall_du[r, p_i, s, 2] = d1[r]
            wall_du[r, p_i, s, 3] = d2[r]
        end
    end
    return grid
end

# ── Water positivity ───────────────────────────────────────────────────────────

"""
Row names of `ModelTile.mc_water_stats`, accumulated per thread by [`clamp_water!`](@ref):

| row | meaning |
|-----|---------|
| `:total`   | Σ negative water encountered [kg/m³, summed over gridpoints and steps] |
| `:min_c`   | most negative `ρ_c` seen [kg/m³] |
| `:min_r`   | most negative `ρ_r` seen [kg/m³] |
| `:worst_dT`| largest implied latent-heat kick of a single event [K] (see below) |
| `:count`   | number of gridpoint-steps carrying negative water |
| `:warned`  | internal: largest `worst_dT` already warned about (thread 1 only) |

Rows 7 onward are the per-step **production budget**, written by [`water_budget_probe!`](@ref)
and reset every step by [`water_budget_trace`](@ref); they answer which term is driving the
water negative, which rows 1-6 (cumulative extrema) structurally cannot. Each block records the
term-by-term tendency AT the gridpoint where that species is most negative this step:

| row | meaning |
|-----|---------|
| `:b_r_val`  | most negative `ρ_r` in this thread's columns this step [kg/m³] |
| `:b_r_adv`  | `-v·∇ρ_r'` there [kg/m³/s] |
| `:b_r_cdiv` | `-ρ_r ∇·v` there — the compressibility term, a sign-blind linear amplifier |
| `:b_r_src`  | `Qdot_r` (rain-channel condensation/evaporation) there |
| `:b_r_auto` | `AUTO_COLL` (autoconversion + collection from cloud) there |
| `:b_r_sed`  | `-∂F_r/∂z` (sedimentation flux divergence) there |
| `:b_r_div`  | `∇·v` there [1/s] |
| `:b_r_w`    | `w` there [m/s] |
| `:b_r_z`    | height of the point [m] |
| `:b_r_eul`  | the FORWARD-EULER projection `ρ + ts·f_n` at that same point [kg/m³] |
| `:b_r_now`  | the current value `ρ` at that same point [kg/m³] |
| `:b_c_*`    | the same for `ρ_c`, with `:b_c_src` = `Qdot` and `:b_c_auto` = `-AUTO_COLL` |

The final two blocks are the per-step **depletion census**, written by
[`water_depletion_probe!`](@ref) just before `explicit_timestep`. The budget block above says
what happens at ONE point; these say how widespread it is, and they are what separates a
limiter sized for the wrong integrator from a stiff source term:

| row | meaning |
|-----|---------|
| `:d_r_n`      | gridpoints in this thread's columns with `ρ_r > 0` |
| `:d_r_cevap`  | of those, how many have the depletion sink pinned at its bound |
| `:d_r_cauto`  | how many have `AUTO_COLL` pinned at `avail` (cloud block only; 0 for rain/vapor) |
| `:d_r_eul`    | max forward-Euler depletion fraction `-ts·f_n/ρ` |
| `:d_r_ab3`    | max ACTUAL depletion fraction `-(ρ^{n+1}-ρ)/ρ` under the run's AB3 weights |
| `:d_r_neg`    | how many points the step actually drives negative (`:d_r_ab3 > 1` pointwise) |
| `:d_r_stiff`  | how many exceed AB3's real-axis stability limit (0.545) with NO cap active |
| `:d_r_infeas` | how many have an INADMISSIBLE HISTORY — the two previous sink levels alone already carry the point negative, so no bound on the current level can fix it and [`_ab3_sink_bound`](@ref) has been clamped at 0 |
| `:d_r_mab3`   | max depletion fraction from the MICROPHYSICS ALONE, `-AB3(micro history)/ρ` |
| `:d_r_mneg`   | how many points the microphysics alone would drive negative (`:d_r_mab3` past `MICRO_DEPLETION_TOL`) |
| `:d_c_*`      | the same for `ρ_c` |
| `:d_v_*`      | the same for the PROGNOSTIC `ρ_v`, read straight off its own slot tendency, whose depletion sink is the condensation `-(Qdot + Qdot_r)` and (with ice on) the deposition |
| `:v_gap`      | max over the tile this step of the RECONCILIATION GAP `\\|res_rho_t − ρ_v\\|` [kg/m³] — the disagreement between the conserved `ρ_t` budget and the transported vapor, and `τ_rec` times the rate [`rho_v_reconcile`](@ref) is removing it at. Identically 0 on a resting base by construction; a load-bearing DRIFT diagnostic everywhere else |
| `:q_gap`      | max over the tile this step of the Q_ss DETACHMENT `\\|Q_ss − (ρ_v − ρ_vs)\\|` [kg/m³] — the other end of the reconciliation chain, and `τ_qss` times [`qss_relaxation`](@ref)'s rate. The S3 drive clip is a function of exactly this quantity, so it is what says whether the clip is doing anything |

**Read `:d_*_mab3`, not `:d_*_ab3`, to judge the depletion.** `:d_*_ab3` is the fraction
of the FULL slot tendency, so it is dominated by advection and `-ρ∇·v` at points carrying
almost no condensate — where any fixed tendency divided by a near-zero `ρ` is large, and no
microphysical limiter has, or should have, any purchase. `:d_*_mab3` is the part a
microphysical bound would govern; it is now a pure measurement (the depletion caps were
removed — see [`qss_condensation_rates`](@ref)), so nothing holds it at 1 and its excursions
above `MICRO_DEPLETION_TOL` are the reported cost of that removal rather than a defect.

The LAST block is the **stiffness census**, written by [`mc_stiffness_census!`](@ref) on every
step of every run (not gated on any trace) and reported by [`mc_stiffness_trace`](@ref):

| row | meaning |
|-----|---------|
| `:s_c_max`  | running max over the run of `ts·(1/τ_c)`, the cloud condensation channel |
| `:s_c_n`    | gridpoint-STEPS with `ts·(1/τ_c) > 1`, summed over the run |
| `:s_r_max`  | the same for the rain channel |
| `:s_r_n`    | the same for the rain channel |
| `:s_warned` | internal: 1.0 once the once-per-run stiffness `@warn` has been emitted (thread 1 only) |

Unlike every block above it, this one is CUMULATIVE — it sits after [`MC_QSS_GAP`](@ref
MC_QSS_GAP) and outside the `MC_BUDGET_FIRST:MC_QSS_GAP` range `water_budget_trace` resets
each step, because a stiffness excursion that happened at step 40 is still true at step 4000
and the run-end warning has to be able to see it.
"""
const MC_WATER_STATS = (:total, :min_c, :min_r, :worst_dT, :count, :warned,
                        :b_r_val, :b_r_adv, :b_r_cdiv, :b_r_src, :b_r_auto, :b_r_sed,
                        :b_r_div, :b_r_w, :b_r_z, :b_r_eul, :b_r_now,
                        :b_c_val, :b_c_adv, :b_c_cdiv, :b_c_src, :b_c_auto, :b_c_sed,
                        :b_c_div, :b_c_w, :b_c_z, :b_c_eul, :b_c_now,
                        :b_pre_r, :b_pre_c,
                        :d_r_n, :d_r_cevap, :d_r_cauto, :d_r_eul, :d_r_ab3,
                        :d_r_neg, :d_r_stiff, :d_r_infeas, :d_r_mab3, :d_r_mneg,
                        :d_c_n, :d_c_cevap, :d_c_cauto, :d_c_eul, :d_c_ab3,
                        :d_c_neg, :d_c_stiff, :d_c_infeas, :d_c_mab3, :d_c_mneg,
                        :d_v_n, :d_v_cevap, :d_v_cauto, :d_v_eul, :d_v_ab3,
                        :d_v_neg, :d_v_stiff, :d_v_infeas, :d_v_mab3, :d_v_mneg,
                        :v_gap, :q_gap,
                        :s_c_max, :s_c_n, :s_r_max, :s_r_n,
                        :s_i1_max, :s_i1_n, :s_i2_max, :s_i2_n, :s_i3_max, :s_i3_n,
                        :s_warned,
                        :p_qc_max, :p_qc_n, :p_qr_max, :p_qr_n, :p_nr_max, :p_nr_n,
                        :p_i1_max, :p_i1_n, :p_i2_max, :p_i2_n, :p_i3_max, :p_i3_n,
                        :p_s1_max, :p_s1_n, :p_s2_max, :p_s2_n, :p_s3_max, :p_s3_n,
                        :p_n1_max, :p_n1_n, :p_n2_max, :p_n2_n, :p_n3_max, :p_n3_n,
                        :p_warned,
                        :a_gap, :a_pts, :a_rem, :a_warned,
                        :x_hom, :x_hom_n, :x_bigg, :x_bigg_n,
                        :x_rime, :x_rime_n, :x_rimex, :x_rimex_n,
                        :x_coll, :x_coll_n, :x_melt, :x_melt_n,
                        :x_all, :x_all_n, :x_evap, :x_evap_n, :x_comb, :x_comb_n,
                        :x_mlt1, :x_mlt1_n, :x_agg1, :x_agg1_n,
                        :x_mlt2, :x_mlt2_n, :x_agg2, :x_agg2_n,
                        :x_asat, :x_asat_n,
                        :x_wrc, :x_wrc_n, :x_wrr, :x_wrr_n,
                        :x_wmlt, :x_wmlt_n, :x_wmri, :x_wmri_n,
                        :x_wpts, :x_wmass, :x_wfa, :x_wdfa, :x_wheld,
                        :o_max, :o_pts, :o_rain, :o_seed, :o_warned)

"""
First row of the per-step budget block for each species in `mc_water_stats`.

Both blocks are `MC_BUDGET_N` rows with the SAME layout, so one probe writes either. Cloud has
no sedimentation, so `:b_c_sed` is always zero and is not printed — giving the two blocks
different lengths silently walks off the end of the matrix into the next thread's column.
"""
const MC_BUDGET_R = 7
const MC_BUDGET_C = 18
const MC_BUDGET_N = 11
"""
First row of the per-step depletion census for each species; `MC_DEPLETION_N` rows each,
same layout, written by [`water_depletion_probe!`](@ref).
"""
const MC_DEPLETION_R = 31
const MC_DEPLETION_C = 41
const MC_DEPLETION_V = 51
const MC_DEPLETION_N = 10
"""
The two RECONCILIATION-CHAIN gaps, one row each, holding the per-step maximum over the tile.

| constant | quantity | link of the chain |
|---|---|---|
| `MC_VAPOR_GAP` | `max\\|res_rho_t − ρ_v\\|` | ρ_v ↔ the conserved ρ_t budget ([`rho_v_reconcile`](@ref)) |
| `MC_QSS_GAP`   | `max\\|Q_ss − (ρ_v − ρ_vs)\\|` | Q_ss ↔ ρ_v ([`qss_relaxation`](@ref)) |

Each is `τ` times the rate its nudge is removing it at, and each is identically zero on a
resting base. They are DIAGNOSTICS in the strict sense — nothing in the RHS reads either — but
they are load-bearing ones: `MC_VAPOR_GAP` is the drift between the conservation anchor and
the transported vapor, and `MC_QSS_GAP` is the detachment the S3 drive clip in
[`qss_condensation_rates`](@ref) is a function of, so it is what says whether that clip is
doing anything at all.

They sit AFTER the three `MC_DEPLETION_N`-row blocks, deliberately outside them: they are two
scalars, not a fourth species, and giving the blocks a ragged length silently walks the census
off the end of a thread's column. They live in the per-step region
(`MC_BUDGET_FIRST:MC_QSS_GAP`), so they are reset every step and always describe the step just
reported. `MC_QSS_GAP` is the LAST per-step row: everything after it is cumulative.
"""
const MC_VAPOR_GAP = 61
@doc (@doc MC_VAPOR_GAP)
const MC_QSS_GAP = 62
"""
The STIFFNESS CENSUS block of `mc_water_stats` — `MC_STIFF_N` rows per relaxation channel,
then the once-per-run warning flag. Written by [`mc_stiffness_census!`](@ref).

| constant | meaning |
|---|---|
| `MC_STIFF_C` / `MC_STIFF_R` | channel indices: cloud condensation, rain condensation/evaporation |
| `MC_STIFF_CHANNELS` | how many channels the block holds |
| `MC_STIFF_N` | rows per channel: `[max ts/τ, count of points past ts/τ = 1]` |
| `MC_STIFF_FIRST` | first row of channel 1 |
| `MC_STIFF_WARNED` | the flag row, one past the last channel |

The ice channels (`i1`, `i2`, `i3`) append here: give each an index, raise
`MC_STIFF_CHANNELS`, and add its name pair to `MC_WATER_STATS` and to
[`MC_STIFF_NAMES`](@ref MC_STIFF_NAMES). Nothing else is sized by hand — the row arithmetic,
the reduction and the report all run over `1:MC_STIFF_CHANNELS`.

CUMULATIVE, unlike every block before it: `water_budget_trace` resets
`MC_BUDGET_FIRST:MC_QSS_GAP` each step and this sits outside that range on purpose.
"""
const MC_STIFF_C = 1
const MC_STIFF_R = 2
const MC_STIFF_I1 = 3
const MC_STIFF_I2 = 4
const MC_STIFF_I3 = 5
const MC_STIFF_CHANNELS = 5
const MC_STIFF_N = 2
const MC_STIFF_FIRST = 63
const MC_STIFF_WARNED = MC_STIFF_FIRST + (MC_STIFF_N * MC_STIFF_CHANNELS)
"""Channel labels for the stiffness report, in `MC_STIFF_*` index order."""
const MC_STIFF_NAMES = ("cloud", "rain", "ice1", "ice2", "ice3")
"""
The DONOR-DEPLETION CENSUS block of `mc_water_stats`, laid out exactly like the stiffness block
(`MC_DONOR_N` rows per donor reservoir, then a once-per-run warning flag) and written by
[`mc_donor_census!`](@ref).

What it measures is not what the stiffness census measures. `MC_STIFF_*` records `ts/τ` — how
UNRESOLVED a relaxation is. This records the fraction of each donor reservoir the step
ACTUALLY REMOVES, after the realization factors, i.e. `Σ_j rate_j · J₀(κ_tot Δt) · Δt / q`.
Under the construction of [`_ice_donor_factors`](@ref) that is `1 − e^{−κ_tot Δt} < 1` for the
three liquid donors, identically, so a reading above `1` is not a resolution statement — it is
the construction failing, and the number to watch when a new sink is added to a reservoir.

The rain's row counts EVERY sink on the reservoir, not only the ice ones: rain EVAPORATION —
which is integrated on the `Q_ss` relaxation pair, whose propagator bounds the supersaturation
and not the rain — joined `κ_tot` in Stage 2b (TeX §donor_relax), so `MC_DONOR_QR` is the
combined realized draw and the bound above is a statement about the number that is actually
taken. Before that it read the ice half alone: a dutiful 1.0 on the Stage 0a fixture while the
two halves together took 1.0012 reservoirs, which `MC_ATTR_QR_COMB` had to be built to see.

The three ICE reservoirs are scaled by the same construction: `mc_ice_sources!` forms one
conductance per species over melting, aggregation and sublimation, and every one of those legs
is realized at it, so a reading above `1` there means the same thing it means for the liquid.
Aggregation joined that construction when the census convicted it at 1.33 reservoirs of `q_i1`
per step, in a SECOND pass through `ishmael_aggregation` at the donors' factors (TeX
§donor_relax); before that it was counted in `κ_tot` but applied unscaled.

The three ICE NUMBER rows ([`MC_DONOR_N1`](@ref)) are the newest and the only ones that are
not bounded by construction in the default configuration: a species' number is its own
reservoir with its own conductance, and until `options[:ice_number_realization]` is switched
on its three legs are realized at the species' MASS factor. Whether that over-draws is the
question the rows exist to answer, so they are censused in BOTH modes and the over-depletion
warning is gated on the switch rather than on the reading.

REPORTED, NEVER ENFORCED — as with every census in this file.
"""
const MC_DONOR_QC = 1
const MC_DONOR_QR = 2
const MC_DONOR_NR = 3
const MC_DONOR_I1 = 4
const MC_DONOR_I2 = 5
const MC_DONOR_I3 = 6
const MC_DONOR_S1 = 7
const MC_DONOR_S2 = 8
const MC_DONOR_S3 = 9
"""
The three ICE NUMBER reservoirs (Stage 1b). Each species' NUMBER is a donor in its own
right — three legs draw on it (aggregation's number transfer, the melt number `nmlt`, and
the sublimation number sink of the deposition channel's habit partition) — and until this
block none of them was censused anywhere. The mass rows above cannot stand in for them:
`colamt` and `colamtn` come from two different offline tables integrating two different
moments of the collection kernel (`mkcoltb`, ishmael_tables.jl), and the melt number leg
carries `dNmltri`, which has no mass partner at all, so the fraction of the number a step
removes is an independent state function of the fraction of the mass it removes.

REPORTED, NEVER ENFORCED, like every row of this block — and unlike the six above, these
three are NOT bounded by construction unless `options[:ice_number_realization]` is on. With
it off (the default) the number legs ride the MASS factor, which is the state the census
exists to measure; with it on each species' number carries its own `J₀(κ_{n,k}Δt)` and the
row is `1 − e^{−κ_{n,k}Δt} < 1` for the same reason the mass rows are.

Two caveats, both on species 3 and both stated rather than hidden. (a) With
`options[:ice_agg_caps] = false` the conductance is formed on the CAPPED unit pass and
realized on the uncapped one, exactly as the mass side already is. (b) `nagg3` is a NET
count — new aggregates from the planar/columnar pairs MINUS aggregate self-collection — and
only the self-collection half is species 3's own to realize (`f_aggn3`). The conductance is
formed on the net loss, so where the two halves nearly cancel the `n_i3` row is bounded by
the aggregation term alone rather than by `1 − e^{−κ_{n,3}Δt}`. Decomposing it would mean
returning the seven pairs' number transfers separately, which is a larger change to the
ported kernel than the reading justifies today.
"""
const MC_DONOR_N1 = 10
@doc (@doc MC_DONOR_N1)
const MC_DONOR_N2 = 11
@doc (@doc MC_DONOR_N1)
const MC_DONOR_N3 = 12
const MC_DONOR_CHANNELS = 12
const MC_DONOR_N = 2
const MC_DONOR_FIRST = MC_STIFF_WARNED + 1
const MC_DONOR_WARNED = MC_DONOR_FIRST + (MC_DONOR_N * MC_DONOR_CHANNELS)
"""Donor labels for the depletion report, in `MC_DONOR_*` index order."""
const MC_DONOR_NAMES = ("q_c", "q_r(ice+evap)", "n_r", "q_i1(melt+agg)", "q_i2(melt+agg)",
                        "q_i3(melt+agg)", "q_i1(subl)", "q_i2(subl)", "q_i3(subl)",
                        "n_i1(melt+agg+subl)", "n_i2(melt+agg+subl)",
                        "n_i3(melt+agg+subl)")
"""
The ANCHOR-RECONCILIATION census of `mc_water_stats` — the third reconciliation tier's
diagnostic (TeX §"Reconciliation of the condensate partition"), written by
[`_ice_anchor_reconcile!`](@ref) and reported by [`mc_stiffness_trace`](@ref).

| constant | quantity |
|---|---|
| `MC_ANCHOR_GAP` | run-maximum of the partition defect `δ_part` (Eq. partition_gap) [kg/m³] |
| `MC_ANCHOR_PTS` | cumulative gridpoint-steps with `δ_part > 0` |
| `MC_ANCHOR_REMOVED` | cumulative removed mass, `Σ φ·Σₖmax(ρ_{i,k},0)·Δt` over points and steps [kg/m³ · gridpoint-steps] |
| `MC_ANCHOR_WARNED` | internal: 1.0 once the once-per-run detachment `@info` has fired |

CUMULATIVE, like the stiffness and donor blocks and unlike `MC_VAPOR_GAP`: those two
per-step rows are written and reset only inside `water_budget_trace`, which refuses to run
under `:ishmael` — an ice-arm census gated there would never record anything. This one is
written on every ice step of every run, whether or not the reconciliation SOURCE is applied:
`options[:ice_anchor_source] = false` switches off the removal, never the measurement, so
the off switch is a forensic tool and not a blindfold.

`MC_ANCHOR_GAP` is what `MC_VAPOR_GAP` cannot see: during a detachment the vapor nudge
succeeds — `res_rho_t − ρ_v → 0` while the partition itself is wrong — so the first gap
census reads zero exactly when this one is the whole story. `MC_ANCHOR_REMOVED` is the
standing bias detector for the one-sided removal: an unweighted gridpoint-sum proxy (the
same weighting as every domain sum in this census family), to be read against the ice
water path — secular growth outside the glaciation front means `τ_anchor` is too short or
the attribution is wrong.
"""
const MC_ANCHOR_GAP = MC_DONOR_WARNED + 1
@doc (@doc MC_ANCHOR_GAP)
const MC_ANCHOR_PTS = MC_DONOR_WARNED + 2
@doc (@doc MC_ANCHOR_GAP)
const MC_ANCHOR_REMOVED = MC_DONOR_WARNED + 3
@doc (@doc MC_ANCHOR_GAP)
const MC_ANCHOR_WARNED = MC_DONOR_WARNED + 4
"""
The ATTRIBUTION CENSUS of `mc_water_stats` — the per-CHANNEL decomposition of the two
depletion breaches the `MC_DONOR_*` block convicts (TeX §"Donor relaxation and its
realization", §"Reconciliation of the condensate partition"), written by
[`mc_attr_census!`](@ref) and reported by [`mc_stiffness_trace`](@ref).

`MC_DONOR_*` says WHICH reservoir a step over-draws and by how much. It cannot say WHICH LEG
did it, and on the ice arm that is the whole question: the q_r breach could be riming, Bigg,
homogeneous freezing, ice-rain collection or (uncensused anywhere else) rain evaporation, and
the q_i1 breach could be melting or aggregation. This block answers "which one" and nothing
else. Same `[run-max, count]` layout as the two blocks above it, `MC_ATTR_N` rows per channel,
then a five-scalar TAIL that is not a channel.

The four measurement sites, in the order the step visits them:

| block | site | what it decomposes |
|---|---|---|
| A | `mc_ice_sources!`, at the q_r breach points only | the six legs of `−ICE_R` against `q_r` |
| B | the ETD pre-compute, beside the sublimation census | rain EVAPORATION at the APPLIED step-mean, alone and combined with the ice draw |
| C | `mc_ice_sources!` | the q_i1/q_i2 split between MELT and AGGREGATION |
| D | `mc_ice_sources!`, `T > T_0 + 2` K | the above-freezing riming/melting loop, and the anchor share's withholding |

Block A's counts PARTITION the `MC_DONOR_QR` count exactly: at every breach point (the
combined `(−ICE_R/ρ_a + f_r κ_ev q_r)·Δt/q_r > 1` of Stage 2b, `mc_donor_census!`'s own
expression character for character) exactly one of channels `MC_ATTR_QR_HOM … MC_ATTR_QR_COLL` is
credited, the one carrying the LARGEST debit, so `Σ` of those five counts is the number of
breach points and no leg can be blamed twice. `MC_ATTR_QR_RIMEX` is a SUB-PART of
`MC_ATTR_QR_RIME` — the rain rime the realized pass debits IN EXCESS of the share the
conductance was formed on (`f_qr·prdr0_r`), the split-plus-density cross term — so it can only
win that argmax where the conductance share is negative; it is in the block for its run-max,
and a nonzero count there is itself the finding. `MC_ATTR_QR_MELT` is a CREDIT, not a debit
(melting ADDS rain), so it records a run-max and is never counted; `MC_ATTR_QR_ALL` is the sum
of the four true debits, which is `MC_DONOR_QR`'s ice half plus the melt credit and therefore
an upper bound on that half, point by point.

Block B (`MC_ATTR_QR_EVAP`, `MC_ATTR_QR_COMB`) is unchanged in DEFINITION and changed in what
it is expected to read. It reports the APPLIED step-mean evaporation `q̄_r` — the number the
rain slot actually receives, which is the propagator's and not the frozen rate — alone and
added to the ice draw. Since Stage 2b the evaporation conductance is inside `κ_tot`, so the
combined number is expected `≤ 1`: the two draws are shares of one exponential depletion
rather than two independent ones. It was 1.0012 over 1296 gridpoint-steps on the Stage 0a
fixture before that, which is the measurement the stage exists to answer. It remains an
independent reading of the bound — taken at the applied step-mean rather than at the frozen
rate the factor was formed on — and therefore still reports rather than limits.

`MC_ATTR_AGG1_SAT` counts the gridpoint-steps at which species 1's aggregation increment takes
its ENTIRE mass (`−qagg1 ≥ q1(1−10⁻¹²)`) — the caller-side proxy for `ishmael_col1`'s per-pair
`min(colamt, rx)` cap binding, which is invisible from here because the seven pairs are summed
inside `ishmael_aggregation`. Its run-max row is by construction `MC_ATTR_AGG1`'s; the row
exists for its COUNT, which is what says whether removing those caps would move anything —
`options[:ice_agg_caps] = false` (env `SCYTHE_O01_AGGCAPS=0`) is the switch that removes them,
and this count is the measurement it is meant to be read against.

The five-scalar TAIL, over `T > T_0` gridpoints that carry ice (block D, the melting-level
question): gridpoint-steps, the summed RAW ice density (an unweighted gridpoint-sum proxy, the
same weighting as every domain sum in this census family), the count and run-max of the
rate-side anchor share `f_a` biting, and the WITHHELD MELT `Σ(1/f_a − 1)|Σ q̇_mlt| ρ_a Δt` —
melting is linear in the population at fixed per-particle state, so `q̇_mlt/f_a` is what the
RAW population would have melted and the difference is what the anchor share held back. Read
together they decide whether sub-melting-level ice is anchor-REAL (`f_a = 1`, the withheld
melt zero) or detached, which is the question Stage 3's leg-D exemption turns on.

CUMULATIVE, like the stiffness, donor and anchor blocks. OPT-IN, unlike all three:
`mc_water_stats` is always allocated, so the block is written only where
`options[:ice_attr_census]` is true (default false, env `SCYTHE_O01_ATTR=1`) and is otherwise
never touched — an explicit flag is what makes an eighteen-channel census zero-cost on the
runs that are not asking the question.

REPORTED, NEVER LIMITED, and exactly `0.0` on the warm/dry path: every write is gated on a
state test that a warm ice-free column fails identically (no breach, no ice mass, no
above-freezing ice), so switching the option on cannot move a rate, a slot or a bit.
"""
const MC_ATTR_QR_HOM = 1
@doc (@doc MC_ATTR_QR_HOM)
const MC_ATTR_QR_BIGG = 2
@doc (@doc MC_ATTR_QR_HOM)
const MC_ATTR_QR_RIME = 3
@doc (@doc MC_ATTR_QR_HOM)
const MC_ATTR_QR_RIMEX = 4
@doc (@doc MC_ATTR_QR_HOM)
const MC_ATTR_QR_COLL = 5
@doc (@doc MC_ATTR_QR_HOM)
const MC_ATTR_QR_MELT = 6
@doc (@doc MC_ATTR_QR_HOM)
const MC_ATTR_QR_ALL = 7
@doc (@doc MC_ATTR_QR_HOM)
const MC_ATTR_QR_EVAP = 8
@doc (@doc MC_ATTR_QR_HOM)
const MC_ATTR_QR_COMB = 9
@doc (@doc MC_ATTR_QR_HOM)
const MC_ATTR_MLT1 = 10
@doc (@doc MC_ATTR_QR_HOM)
const MC_ATTR_AGG1 = 11
@doc (@doc MC_ATTR_QR_HOM)
const MC_ATTR_MLT2 = 12
@doc (@doc MC_ATTR_QR_HOM)
const MC_ATTR_AGG2 = 13
@doc (@doc MC_ATTR_QR_HOM)
const MC_ATTR_AGG1_SAT = 14
@doc (@doc MC_ATTR_QR_HOM)
const MC_ATTR_WARM_RIME_C = 15
@doc (@doc MC_ATTR_QR_HOM)
const MC_ATTR_WARM_RIME_R = 16
@doc (@doc MC_ATTR_QR_HOM)
const MC_ATTR_WARM_MELT = 17
@doc (@doc MC_ATTR_QR_HOM)
const MC_ATTR_WARM_MLTRI = 18
@doc (@doc MC_ATTR_QR_HOM)
const MC_ATTR_CHANNELS = 18
@doc (@doc MC_ATTR_QR_HOM)
const MC_ATTR_N = 2
@doc (@doc MC_ATTR_QR_HOM)
const MC_ATTR_FIRST = MC_ANCHOR_WARNED + 1
"""
The five-scalar TAIL of the attribution census, over `T > T_0` gridpoints carrying ice.
Not channels: they are sums and counts of the POPULATION, not of a depletion fraction, so
they sit after the `MC_ATTR_CHANNELS` `[max, count]` pairs rather than inside them.

| constant | quantity |
|---|---|
| `MC_ATTR_WARM_PTS`  | gridpoint-steps with `T > T_0` and `Σ_k ρ_{i,k} > 0` |
| `MC_ATTR_WARM_MASS` | `Σ` of the RAW ice density over those points [kg/m³ · gridpoint-steps] |
| `MC_ATTR_WARM_FA`   | how many of them had the rate-side anchor share `f_a < 1` |
| `MC_ATTR_WARM_DFA`  | run-max of `1 − f_a` there |
| `MC_ATTR_WARM_HELD` | `Σ(1/f_a − 1)·\\|Σ_k q̇_mlt\\|·ρ_a·Δt`, the melt the anchor share withheld |
"""
const MC_ATTR_WARM_PTS = MC_ATTR_FIRST + (MC_ATTR_N * MC_ATTR_CHANNELS)
@doc (@doc MC_ATTR_WARM_PTS)
const MC_ATTR_WARM_MASS = MC_ATTR_WARM_PTS + 1
@doc (@doc MC_ATTR_WARM_PTS)
const MC_ATTR_WARM_FA = MC_ATTR_WARM_PTS + 2
@doc (@doc MC_ATTR_WARM_PTS)
const MC_ATTR_WARM_DFA = MC_ATTR_WARM_PTS + 3
@doc (@doc MC_ATTR_WARM_PTS)
const MC_ATTR_WARM_HELD = MC_ATTR_WARM_PTS + 4
@doc (@doc MC_ATTR_WARM_PTS)
const MC_ATTR_LAST = MC_ATTR_WARM_HELD
"""Channel labels for the attribution report, in `MC_ATTR_*` index order."""
const MC_ATTR_NAMES = ("q_r: homogeneous (mimr)", "q_r: Bigg (mbig)",
                       "q_r: rain riming (realized)",
                       "q_r: rain riming EXCESS over the conductance share",
                       "q_r: ice-rain collection", "q_r: melt CREDIT (max only)",
                       "q_r: all debits, no melt credit",
                       "q_r: rain evaporation (applied step-mean)",
                       "q_r: ice + evaporation combined",
                       "q_i1: melting", "q_i1: aggregation",
                       "q_i2: melting", "q_i2: aggregation",
                       "q_i1: aggregation SATURATES the species mass (cap-binding proxy)",
                       "T>T_0+2: cloud riming share of q_c",
                       "T>T_0+2: rain riming share of q_r",
                       "T>T_0+2: melting share of q_ice",
                       "T>T_0+2: dQImltri share of q_ice")
"""
The POPULATION-RECONCILIATION census of `mc_water_stats` — the FOURTH reconciliation tier's
diagnostic (TeX §"Reconciliation of the population"), written by
[`_ice_population_reconcile!`](@ref) and reported by [`mc_stiffness_trace`](@ref).

| constant | quantity |
|---|---|
| `MC_POP_MAX`    | run-maximum of the number-less mass `ρ^∅_{i,k}` over species and points [kg/m³] |
| `MC_POP_PTS`    | cumulative (gridpoint × species)-steps with `ρ^∅ > 0` |
| `MC_POP_RAIN`   | cumulative mass returned to the RAIN above `T_0`, `Σ (ρ^∅/τ_pop)·Δt` [kg/m³ · gridpoint-steps] |
| `MC_POP_SEED`   | cumulative ICE number seeded below `T_0`, `Σ (ρ^∅/(m_seed τ_pop))·Δt` [#/m³ · gridpoint-steps]; `m_seed` is `ISHMAEL_M_MIN` or the large-crystal mass, per `options[:ice_population_seed]` |
| `MC_POP_WARNED` | internal: 1.0 once the once-per-run number-less-mass `@info` has fired |

CUMULATIVE, and written on every ice step whether or not the SOURCE is applied, exactly as
the `MC_ANCHOR_*` block is: `options[:ice_population_source] = false` — and
`options[:condensation] = false`, which switches this transfer off with the other phase
changes — stop the transfer, never the measurement.

The rain NUMBER seeded above `T_0` is not a row of its own because it is not independent:
the warm branch seeds one drop per [`RAIN_2M_M_AUTO`](@ref) of returned mass, so it is
`MC_POP_RAIN / RAIN_2M_M_AUTO` exactly.

`MC_POP_MAX` is what neither `MC_VAPOR_GAP` nor `MC_ANCHOR_GAP` can see: the anchor share of
a number-less mass is measured to be exactly one (the water is real; only its phase
REPRESENTATION is unusable), so the partition census reads zero at precisely the points this
one convicts. Read `MC_POP_RAIN` against the surface rain accumulation and `MC_POP_SEED`
against the ice number path — a secular contribution from either says `τ_pop` is too short.
"""
const MC_POP_MAX = MC_ATTR_LAST + 1
@doc (@doc MC_POP_MAX)
const MC_POP_PTS = MC_ATTR_LAST + 2
@doc (@doc MC_POP_MAX)
const MC_POP_RAIN = MC_ATTR_LAST + 3
@doc (@doc MC_POP_MAX)
const MC_POP_SEED = MC_ATTR_LAST + 4
@doc (@doc MC_POP_MAX)
const MC_POP_WARNED = MC_ATTR_LAST + 5
"""Last row of `mc_water_stats`; `length(MC_WATER_STATS)` must equal it."""
const MC_POP_LAST = MC_POP_WARNED
"""
Reservoirs below this are numerical remnants, not physics: a depletion fraction formed on
1e-12 kg/kg of leftover rain is arithmetic on round-off and says nothing about the integrator.
Measured need: without it the census reported `κΔt ≈ 0.5` thousands of times per step at points
holding 1e-12 kg/kg, which buried the real signal.
"""
const MC_DONOR_QFLOOR = 1.0e-9
"""Number-density counterpart of [`MC_DONOR_QFLOOR`](@ref) (per kg)."""
const MC_DONOR_NFLOOR = 1.0e-3
"""
Threshold for `:d_*_mneg`, the count of points the MICROPHYSICS alone drives negative.

A point sitting exactly on an AB3 depletion bound lands at `1 + O(eps)`, not at 1: the `t ≥ 3`
branch of [`_ab3_sink_bound`](@ref) divides by 23, which is not representable, so the bound
cannot cancel the increment exactly. Counting `> 1.0` would therefore report every such point
as a violation. This threshold sits ~1e7 ULP above the rounding and ~1e9 below the `23/12` a
forward-Euler-sized bound produces, which is what made it able to separate the two depletion
modes when those existed. With the caps removed it is a pure reporting threshold on a pure
measurement.
"""
const MICRO_DEPLETION_TOL = 1.0 + 1.0e-9
"""
Per-STEP pre-fit minima (rows 29/30), written by `clamp_water!` and reset every step.

Rows 2/3 (`:min_c`, `:min_r`) are cumulative, so they cannot be differenced against the
post-fit reconstruction to separate what the column step did from what the refit did. These
can: `post_t - pre_t` is the refit's contribution and `pre_t - post_{t-1}` is the column
step's, and the two sum to the step's change exactly.
"""
const MC_PRE_R = 29
const MC_PRE_C = 30
"""First budget row; rows `MC_BUDGET_FIRST:end` are reset every step."""
const MC_BUDGET_FIRST = MC_BUDGET_R

"""
    positivity_reference_profile(name, ref_state) -> AbstractVector or nothing

The reference profile a positivity-bounded prognostic is carried against, or `nothing` when
the variable is a TOTAL and its bound needs no offset.

`u, w, rho_r, n_r` and all twelve ice slots are totals; `p, rho_d, rho_t, E_t, Q_ss, rho_c,
rho_v` are perturbations from the pressure reference (see the prognostic-slot semantics
above). A
zero bound on a PERTURBATION is not merely conservative, it is wrong: it would pin the field
at or above its reference and forbid the cloud from ever evaporating below `ρ̄_c`. Hence
anything not recognised here throws rather than silently taking the factory's constant bound.

The transformed names `nu_c`/`nu_r`/`nu_nr` and the twelve `nu_*` ice aliases deliberately
fall through to the error. A coefficient bound on a control variable is a different constraint
from a bound on the density, and [`install_positivity_bounds!`](@ref) refuses the combination
before reaching here; this is the backstop for a configuration that somehow gets past it.

`rho_v` is served the DERIVED profile `ρ̄_t − ρ̄_d − ρ̄_c`, which is the reference the slot is
actually carried against (see `mc_reference_diagnostics` and the file header) — NOT
Springsteel's fitted `ref_rho_v`. The two differ at fit level, and the difference is exactly
what would stop the resting base being a discrete fixed point.
"""
function positivity_reference_profile(name::AbstractString, ref_state)
    # `n_r` (the two-moment rain number) joins the totals: there is no reference number
    # density in any configuration or in the reference-state format, exactly as for rho_r.
    name in ("rho_r", "n_r", "u", "w", "v") && return nothing  # totals: no offset
    # ...and so do all twelve ice slots, for the same reason and one stronger: there is no
    # resting ice field in any configuration, so `f̄ ≡ 0` is a property of the equation set
    # rather than of the sounding (reference/Scythe_moist_compressible.tex, "The ice variables
    # are carried as totals").
    name in MC_ICE_VARS && return nothing
    (name in ("nu_c", "nu_r", "nu_nr") || name in values(MC_NU_ALIAS)) &&
        error("positivity is declared for \"$name\", which is a CONTROL VARIABLE, not a " *
              "density: a box constraint on its coefficients would bound the transform of " *
              "the field rather than the field. The transform already makes the recovered " *
              "density non-negative by construction — drop \"$name\" from positivity.")
    # The prognostic vapor's reference is DERIVED, not fitted: ρ̄_v ≡ ρ̄_t − ρ̄_d − ρ̄_c on the
    # fitted reference columns, the same expression `mc_reference_diagnostics` forms and the
    # same one every seeding site subtracts. Built here rather than read from
    # `Springsteel.ref_rho_v` so a bound and the slot semantics cannot disagree.
    if name == "rho_v"
        rt = ref_rho_t(ref_state)
        rd = ref_rho_d(ref_state)
        rc = Springsteel.ref_rho_c(ref_state)
        (rt isa Number || rd isa Number) && return nothing
        rcv = rc isa Number ? zero(view(rt, :, 1)) : view(rc, :, 1)
        return view(rt, :, 1) .- view(rd, :, 1) .- rcv
    end
    prof = name == "rho_c" ? Springsteel.ref_rho_c(ref_state) :
           name == "rho_d" ? ref_rho_d(ref_state) :
           name == "rho_t" ? ref_rho_t(ref_state) :
           error("positivity is declared for \"$name\", which is carried as a perturbation " *
                 "from the reference state and has no offset rule here. A constant bound on " *
                 "a perturbation pins the field at or above its reference — add the variable " *
                 "to positivity_reference_profile before enabling it.")
    # Springsteel returns the scalar 0.0 for reference states that carry no such profile.
    prof isa Number && return nothing
    return view(prof, :, 1)
end

"""
    bhyp(rho, mu) -> n
    ahyp(n, mu) -> rho
    ahyp_smooth(n, mu) -> rho
    dbhyp(rho, mu) -> dn/drho

Ooyama (2001, JAS 58, 2073–2102) **biased hyperbolic transform** and its inverses, Eqs.
4.19/4.20/4.23. `reference/ooyama_jas2001.pdf`.

    n   = bhyp(rho)  = ½[(rho + μ) − μ²/(rho + μ)]
    rho = ahyp(n)    = √(n² + μ²) + n − μ        (0 for n ≤ 0 — the published quasi-inverse)
    J   = dn/drho    = ½[1 + μ²/(rho + μ)²]

The map is linear (`n ≈ rho/2`; Ooyama's factor ½ is what removes the coefficient from the
inverse) for `rho ≫ μ`, and stretches `rho` only where it is small enough to be
meteorologically insignificant. `bhyp(0) = ahyp(0) = 0` exactly.

**Why this family and not another.** Every monotone `f: ℝ → (0, ∞)` has `f′ → 0` as `n → −∞`,
so the source quotient `S/f′` is unbounded at the cloud edge and an explicit step overshoots
there. Ooyama biases the hyperbola instead, giving range `(−μ, ∞)`; evaluating `J` at the
RECOVERED density (his Eqs. 4.21–4.22, `D_t n = (dn/drho)·[rho-space tendency]`) then keeps it
in `[0.5, 1]` for `rho ≥ 0`. Measured: the softplus and quadratic-touchdown alternatives fit
exactly as well and detonate on first nucleation with peaks of 1e7 and 1e25 against a baseline
1.2e-2. See `reference/FINDINGS_CONDENSATE_STAGE1.md` §3 and
`benchmarks/condensate_transform_probe.jl`.

`ahyp_smooth` is the strict inverse everywhere — C^∞, range `(−μ, ∞)` — where `ahyp` is
Ooyama's C⁰ quasi-inverse pinned at exactly 0 for `n ≤ 0`. Ooyama notes the strict inverse
reaches at worst `−μ`, so the clip's adjustment never exceeds `μ`. The two are measurably
equivalent (probe: transport peak 2.9376e-3 both, mass drift +1.077 both, nucleation peak
1.2167e-2 vs 1.2166e-2); `ahyp` ships because it gives `rho ≥ 0` exactly, `ahyp_smooth` is
retained for any consumer that needs `f″` at the cloud edge.

`μ` is a partial DENSITY here (Ooyama's is a mixing ratio). At `μ = 1e-7 kg/m³` the residual
negativity costs at most 7.2e-3 K, and that worst point is the model top where `rho_d ~ 0.04`
shrinks the retrieval denominator; near the ground it is ~2.5e-4 K.
"""
#
# **Both maps are written in cancellation-free form**, not as Eqs. 4.19/4.20 read. They are
# the same functions algebraically:
#
#     ½[(ρ+μ) − μ²/(ρ+μ)]  =  ρ(ρ + 2μ) / (2(ρ + μ))
#     √(n²+μ²) + n − μ      =  n + n² / (√(n²+μ²) + μ)
#
# The paper's forms subtract two nearly equal quantities when `ρ ≪ μ` and when `|n| ≪ μ`
# respectively, and that is exactly the regime the whole transform exists to represent.
# Measured over `ρ ∈ [1e-14, 1]`: the paper's forms give `bhyp(0) = 6.6e-24` instead of zero
# and a round-trip relative error reaching 1.7e-9 at `ρ = 1e-14`; these give `bhyp(0) = 0.0`
# exactly and a round trip good to 2.2e-16 everywhere. The exact zero is load-bearing --
# `condensate_slot` relies on it so a cloud-free initial condition converts to exactly 0.0.
@inline bhyp(rho, mu) = (rho * (rho + (2.0 * mu))) / (2.0 * (rho + mu))
@inline ahyp_smooth(n, mu) = n + ((n * n) / (sqrt((n * n) + (mu * mu)) + mu))
@inline ahyp(n, mu) = n <= 0.0 ? 0.0 : ahyp_smooth(n, mu)
@inline dbhyp(rho, mu) = 0.5 * (1.0 + (mu * mu) / ((rho + mu) * (rho + mu)))

"""
    condensate_slot(rho_c, rho_cbar, transform, mu) -> slot value

What an initial condition must write into slot 9 for a physical cloud density `rho_c` against
a reference `rho_cbar`. Under `:none` that is the perturbation `rho_c - rho_cbar`; under a
transform it is the CONTROL-variable deviation `bhyp(rho_c) - bhyp(rho_cbar)`, which is what
Ooyama predicts (§4d: "actual prediction of n is performed in terms of its deviation n' from
the background n̂ = bhyp(m̂)").

A cloud-free initial condition on a cloud-free reference gives exactly `0.0` under both,
because `bhyp(0) = 0` exactly — which is why every current benchmark configuration needs no
initial-condition change to run transformed.
"""
@inline function condensate_slot(rho_c, rho_cbar, transform::Symbol, mu)
    transform === :none && return rho_c - rho_cbar
    return bhyp(rho_c, mu) - bhyp(rho_cbar, mu)
end

"""
    vapor_slot(rho_v, rho_tbar, rho_dbar, rho_cbar) -> slot value

What an initial condition must write into the prognostic VAPOR slot for a physical vapor
density `rho_v`: the perturbation from the DERIVED reference `ρ̄_v ≡ ρ̄_t − ρ̄_d − ρ̄_c`,
evaluated on the fitted reference columns at this level.

There is no transform and no Jacobian — the slot is a plain untransformed perturbation, the
shape slot 7 has rather than slot 9's. A transform on the vapor would buy nothing: it is the
LARGEST water species, positivity is a resolution diagnostic here and not a state to enforce
(reference/FINDINGS_NEGATIVE_WATER_ATTRIBUTION.md), and the whole point of making it
prognostic is that no water variable is a residual any more.

The reference is DERIVED rather than read from `Springsteel.ref_rho_v` deliberately. The
reconciliation nudge differences this slot against the equation set's own density residual
`ρ_t − ρ_d − ρ_c − ρ_r − ρ_i`, and the two agree BIT FOR BIT — hence `δ̄ ≡ 0` on a resting
base, hence a discrete fixed point — only if the same three fitted columns are subtracted
here. [`mc_reference_diagnostics`](@ref) forms the identical expression, and its `rho_vbar` is
what `mc_driver!` adds back.
"""
@inline vapor_slot(rho_v, rho_tbar, rho_dbar, rho_cbar) =
    rho_v - (rho_tbar - rho_dbar - rho_cbar)

"""
    recover_rho_c(slot, rho_cbar, transform, mu) -> rho_c

The inverse of [`condensate_slot`](@ref): the physical cloud density a slot-9 value stands
for. `:none` is the plain `slot + rho_cbar` the code always did, bit for bit.

Module-level rather than a closure at the call sites: `clamp_water!` runs this once per
gridpoint per step inside the loop the allocation tests hold at zero, and a captured closure
there is exactly the kind of thing that stops eliding.
"""
@inline function recover_rho_c(slot, rho_cbar, transform::Symbol, mu)
    transform === :none && return slot + rho_cbar
    n = slot + bhyp(rho_cbar, mu)
    return transform === :bhyp ? ahyp(n, mu) : ahyp_smooth(n, mu)
end

"""
    total_slot(value, transform, mu) -> slot value
    recover_total(slot, transform, mu) -> value

The [`condensate_slot`](@ref) / [`recover_rho_c`](@ref) pair for a species carried as a
TOTAL — one with no reference profile (`f̄ ≡ 0` in every configuration and in the
reference-state format itself), so there is no background to subtract and the slot IS the
control variable rather than its deviation.

Every total-form transform in the set shares this ONE implementation:
[`rain_slot`](@ref)/[`recover_rho_r`](@ref) for the rain density and
[`rain_number_slot`](@ref)/[`recover_n_r`](@ref) for the rain number are named aliases of
it, not copies. Adding a species means adding the two aliases and a mode accessor, never
another copy of the branch.

`total_slot(0.0, …) == 0.0` exactly, because `bhyp(0) == 0` exactly, which is why no initial
condition needs changing to run transformed — none of the `*_mc!` initializers seeds rain or
rain number with anything but zero.
"""
@inline function total_slot(value, transform::Symbol, mu)
    transform === :none && return value
    return bhyp(value, mu)
end

@doc (@doc total_slot)
@inline function recover_total(slot, transform::Symbol, mu)
    transform === :none && return slot
    return transform === :bhyp ? ahyp(slot, mu) : ahyp_smooth(slot, mu)
end

"""
    rain_slot(rho_r, transform, mu) -> slot value
    recover_rho_r(slot, transform, mu) -> rho_r

The rain-DENSITY (slot 8) names of [`total_slot`](@ref) / [`recover_total`](@ref).
"""
@inline rain_slot(rho_r, transform::Symbol, mu) = total_slot(rho_r, transform, mu)

@doc (@doc rain_slot)
@inline recover_rho_r(slot, transform::Symbol, mu) = recover_total(slot, transform, mu)

"""
    rain_number_slot(n_r, transform, mu) -> slot value
    recover_n_r(slot, transform, mu) -> n_r

The rain-NUMBER names of [`total_slot`](@ref) / [`recover_total`](@ref), for the
`options[:rain_moments] = 2` slot. `mu` here is `physical_params[:mu_rain_n]`, which is in
#/m³ and therefore has nothing to do with the `rain_mu` of the density slot — the two
transforms are independent knobs on quantities that differ by ten orders of magnitude.
"""
@inline rain_number_slot(n_r, transform::Symbol, mu) = total_slot(n_r, transform, mu)

@doc (@doc rain_number_slot)
@inline recover_n_r(slot, transform::Symbol, mu) = recover_total(slot, transform, mu)

"""
    _load_total_slot!(x, nu, nu_z, J, v, trans_on, trans, mu) -> Nothing

Stage one TOTAL-form prognostic for the column: copy the control variable and its fitted
vertical gradient out of the grid views, recover the quantity itself, and form the Jacobian
`J = dν/dx` the source terms are multiplied by.

This is the block slots 8/9 and `n_r` write out inline, factored so the twelve ice slots do
not become twelve more copies of it. Under `:none` it is `copyto!` plus `fill!(J, 1.0)`, and
`1.0 * x === x` for every double, so a transform-free ice slot takes exactly the untransformed
continuity form.

`@inline` and allocation-free: `x`, `nu`, `nu_z`, `J` are the caller's scratch columns and
`v` is an `mc_slot_views` `NamedTuple` of `SubArray`s.
"""
@inline function _load_total_slot!(x, nu, nu_z, J, v, trans_on::Bool, trans::Symbol, mu)
    copyto!(nu, v.f)
    copyto!(nu_z, v.f_z)
    if trans_on
        if trans === :bhyp
            @. x = ahyp(nu, mu)
        else
            @. x = ahyp_smooth(nu, mu)
        end
        @. J = dbhyp(max(x, 0.0), mu)
    else
        copyto!(x, nu)
        fill!(J, 1.0)
    end
    return nothing
end

"""
    _ice_flux!(mtile, F, F_z, x, w_fall, f_anchor, slot) -> Nothing

Form one ice moment's sedimentation flux `F = f_anchor·max(x, 0)·W` and its fitted divergence
`∂F/∂z`, on THAT SLOT'S OWN spline column — its own basis and its own boundary conditions, so
a mass slot with a Natural bottom lets the crystals leave the domain while a moment with a
different fit is not forced to agree with it.

`f_anchor` is the sedimentation anchor share (the `anchor_f` scratch column): `min(1,
headroom/ρ_i)` per gridpoint, exactly `1.0` — and this function bitwise its pre-Stage-C
self — wherever the partition is admissible. One factor shared by all four moments of every
species, so the size sorting and the mass/number correlation of what falls are untouched;
the withheld phantom part stays in the slot for the reconciliation source to remove.

`w_fall` is the fall speed WEIGHTED BY THE MOMENT BEING TRANSPORTED (negative downward, like
`Vt`), which is the whole reason each of the four moments gets its own flux: mass, number and
the two volume moments fall at different rates and the distribution sorts as it descends.

`max(x, 0)` mirrors the rain flux for the same reason: a spline undershoot must not sediment
NEGATIVE ice upward. The rate functions guard themselves the same way.

`w_fall` is a COLUMN (the Mitchell-Heymsfield speeds of [`ishmael_fall_speeds`](@ref), already
capped at 25 m/s inside that function and negated to Scythe's downward-negative convention),
or a scalar `0.0` for a caller that wants the flux switched off.

# The empty-species short circuit

An ice species with no ice anywhere in the column has `w_fall ≡ 0` and `x ≡ 0`, and fitting a
zero vector through the spline chain costs the same as fitting a real one — measured at S7 as
1.7× the wall clock of the whole driver for twelve such fits per column. So the fit is GATED:
if no level has both a nonzero fall speed and a nonzero (positive) amount, `F` and `F_z` are
`fill!`ed with zeros and the transform is skipped. That is not an approximation — the product
`max(x,0)·w` is identically zero over the column, the spline of the zero function is zero, and
its derivative is zero — it is the same numbers reached without the arithmetic, which is what
keeps the zero-ice inertness gate BITWISE rather than merely small.
"""
@inline function _ice_flux!(mtile::ModelTile, F, F_z, x, w_fall, f_anchor, slot::Int)
    # `f_anchor` is the sedimentation anchor share (see the `anchor_f` scratch doc): the
    # transported amount is the anchor-supported part of the moment, with `f_anchor == 1.0`
    # exactly — hence this line bitwise the unfactored product — wherever the partition is
    # admissible. One shared factor across a species' four moments, so the size sorting and
    # the mass/number correlation of what falls are untouched.
    @. F = (f_anchor * max(x, 0.0)) * w_fall
    live = false
    @inbounds for i in eachindex(F)
        if F[i] != 0.0
            live = true
            break
        end
    end
    if !live
        fill!(F, 0.0)          # sweeps any -0.0 the product may have produced
        fill!(F_z, 0.0)
        return nothing
    end
    col = scratch_column(mtile, slot)
    col.uMish .= F
    Btransform!(col)
    Atransform!(col)
    Ixtransform(col, F_z)
    return nothing
end

"""
    condensate_transform_mode(options) -> Symbol

Which control variable slot 9 carries. `:none` (the default, and the state of every
configuration that does not set the key) means the slot IS the cloud density and the whole
transform is bitwise absent. `:bhyp` means it carries `n = bhyp(rho_c)` and the density is
recovered by [`ahyp`](@ref); `:bhyp_smooth` uses `ahyp_smooth` instead.

**What the transform buys, and what it does not.** It does not reduce the ringing: the
excursion simply happens in `n`, at the same amplitude in `rho`-equivalent terms (measured:
`min(n)` = −2.01e-4 against the untransformed `min(rho)` = −4.33e-4, exactly Ooyama's factor
½). What it buys is that the RECOVERED density is bounded below by `−μ` whatever `n` does, so
the ringing can no longer feed the retrieval. That matters because the negative region is a
**physics-free reservoir**: every rate function is `max(rho,0)`-guarded, so a negative point
has no evaporation, no autoconversion and no collection, while the positive overshoot at the
peak is consumed every step. On the shipped O01 quick run that reservoir reaches 2.75× the
real cloud mass by t = 2400 s (`reference/FINDINGS_CONDENSATE_STAGE1.md` §1).

**Why not the spline positivity limiter instead.** It removes the reservoir too, and offline
the two are indistinguishable. Run, it costs a 250–1000× rise in entropy production
(2.86e4 → 7.06e6 at `POSITIVITY=ck`, → 2.98e7 at `=1`) with the entropy drift flipping from
+0.08 % to −2.6 %, because a per-step coefficient repair is not a physical process and the
budget cannot account for it; `max_w` goes 9.87 → 17.4 → 46.5. The transform is a change of
variables: nothing is ever repaired, and `n` is never modified ("n itself is untouched, so
that the effect of m adjustments does not accumulate in the predicted n" — Ooyama §4d).

**Measured in the model (2026-07-30, quick O01, 3600 s).** `min_rho_c_gm3` is 0 exactly, with
no limiter and nothing repaired, and all five o01 target windows PASS (`max_w` 11.17,
`peak_rain_rate` 78.67, `accum_rainfall_mm` 2.167, `max_rho_r_gm3` 12.65, `rain_onset_min` 24).
Against the untransformed baseline: entropy production over the physical points 0.409 vs 0.520
(−21 %), `energy_drift_pct` 0.153 vs 0.178, `water_mass_drift_pct` −3.55 vs −4.09,
`min_rho_d_frac` 0.881 vs 0.884. For contrast, enforcing the SAME constraint with the spline
limiter gives `max_w` 46.52, `min_rho_d_frac` 0.546 and an entropy production of 203.2 —
390x the baseline. The constraint is affordable; the limiter's way of imposing it is not.

The cost is in the water partition, and it is an UNMASKING rather than a new error. `min_rho_v`
goes −0.724 → −1.199 and the negative-vapor count 750 → 4181. But under either enforcement the
vapor deficit collapses onto the TOTAL-WATER deficit (`min_rho_v` − `min_rho_w`: −0.059 here,
−0.034 under POSITIVITY=1, against **+0.353** in the baseline). The baseline's better-looking
vapor was the negative cloud reservoir cancelling part of the `rho_t` − `rho_d` deficit. That
deficit is a difference of two independently fitted fields, which `benchmarks/FUTURE_WORK.md`
already records as not expressible as a bound on either — no condensate scheme can fix it.

**Known costs, both measured.** `f` is convex near zero, so ringing rectifies into mass
(+1.077 over 3600 s on the probe's transport arm, decelerating, and 3 % of the source-driven
mass where a real source dominates); and a point whose `n` has ratcheted below zero carries a
NUCLEATION LAG of `|n|/(J·S)` before cloud reappears — ~57 s at the worst measured excursion,
against ~128 s for the untransformed field, which spends that time carrying a −54 K anomaly
instead of reading zero cloud.
"""
@inline function condensate_transform_mode(options)
    mode = get(options, :condensate_transform, :none)::Symbol
    (mode === :none || mode === :bhyp || mode === :bhyp_smooth) ||
        error("options[:condensate_transform] = :$(mode) is not recognized; use :none " *
              "(the default: slot 9 is the cloud density), :bhyp (Ooyama's biased " *
              "hyperbolic control variable with his quasi-inverse) or :bhyp_smooth " *
              "(the same forward map with the strict C^inf inverse)")
    return mode
end

"""
    rain_transform_mode(options) -> Symbol

Which control variable SLOT 8 carries, the rain analogue of
[`condensate_transform_mode`](@ref). `:none` (the default, and the state of every
configuration that does not set the key) means the slot IS the rain density.

**Why rain gets its own key.** Cloud was transformed first and alone, deliberately, so that a
rain failure could not be confused with a cloud failure. With cloud settled the same treatment
is right for rain, but the two knobs stay separate so `:bhyp`/`:none` and `:bhyp`/`:bhyp` remain
separable arms rather than an assumption.

**What rain needs that cloud did not.** Cloud has no sedimentation. Rain's dominant sink is the
flux divergence `-∂F_r/∂z`, and under the transform slot 8's tendency carries it inside the
Jacobian while `rho_t` and `E_t` continue to receive the untransformed `-∂F_r/∂z` — they are
still densities and energies. That is correct term by term, but it does mean slot 8 and slot 3
no longer receive the identical discrete number, so the exact telescoping between them is
weakened. The diagnostic for it already exists and is independent: `accum_rainfall_mm` comes
from the `rho_t - rho_d` water path while `accum_rainfall_flux_mm` comes from the `rho_r`
surface flux, and the two must agree as well under the transform as without it.

**Why the limiter is not the answer for rain either, despite working.** `GridParameters.positivity`
on `rho_r` is exact and free on a single grid — `min 0.0` with `bound_shortfall` 0 at every
output time of every run. It is not available on a NESTED run: a child patch's i-boundary is
R3X, which `set_lower_bound!` rejects, so `src/nesting.jl` gives children a k-only bound and
warns about the leak that admits. The transform has no such restriction, which is the design
reason to move rain onto it.
"""
@inline function rain_transform_mode(options)
    mode = get(options, :rain_transform, :none)::Symbol
    (mode === :none || mode === :bhyp || mode === :bhyp_smooth) ||
        error("options[:rain_transform] = :$(mode) is not recognized; use :none " *
              "(the default: slot 8 is the rain density), :bhyp (Ooyama's biased " *
              "hyperbolic control variable with his quasi-inverse) or :bhyp_smooth " *
              "(the same forward map with the strict C^inf inverse)")
    return mode
end

"""
    rain_moments(options) -> Int

How many moments of the rain size distribution the set carries. `1` (the default, and the
state of every configuration that does not set the key) is the single-moment Ooyama closure
— mass only, with the drop number fixed by `physical_params[:N_r]` or the intercept
`physical_params[:N_0]`. `2` adds the prognostic rain NUMBER density `n_r` [#/m³] as an
APPENDED slot and switches the whole rain closure to the ISHMAEL/Morrison exponential-DSD
family in microphysics.jl:

| process | 1 moment (Ooyama) | 2 moments (ISHMAEL/Morrison) |
|---|---|---|
| autoconversion | [`autoconversion_density`](@ref) | [`rain_autoconversion_2m`](@ref) (KK2000) + its number source |
| accretion | [`collection_density`](@ref) | [`rain_accretion_2m`](@ref) (KK2000), number-neutral |
| self-collection / breakup | — | [`rain_selfcollection_2m`](@ref) (Beheng / Verlinde-Cotton) |
| evaporation timescale | [`invtau_rain`](@ref) / [`invtau_rain_mp`](@ref) | [`invtau_rain_2m`](@ref) |
| evaporation number loss | — | [`rain_number_evaporation_2m`](@ref) |
| fall speed | [`rain_terminal_velocity`](@ref) | [`rain_fall_speeds_2m`](@ref), `vtrm` for mass and `vtrn` for number |

The single-moment path is bitwise untouched by the option's presence: with the key absent
the slot is not registered, no scratch is written, and every branch below tests a `Bool`
that is `false`.

**What the second moment buys.** `vtrm > vtrn` always (Γ(4+BR)/6 vs Γ(1+BR)), so mass
outruns number down the column and the mean drop size sorts with height — the process the
single-moment closure cannot represent at all, because it has only one number to fall with.

**What is deferred.** `clamp_water!`'s floor, `water_budget_trace` and the implicit vertical
water diffusion (`Kvdiff_water > 0`) all refuse to run with two moments rather than silently
moving mass without its number; each says so where it refuses.
"""
@inline function rain_moments(options)
    m = get(options, :rain_moments, 1)::Int
    (m == 1 || m == 2) ||
        error("options[:rain_moments] = $(m) is not recognized; use 1 (the default: " *
              "single-moment rain mass, the Ooyama closure) or 2 (prognostic rain number " *
              "n_r, the ISHMAEL/Morrison exponential-DSD closure)")
    return m
end

"""
    rain_number_transform_mode(options) -> Symbol

Which control variable the rain-NUMBER slot carries, the `n_r` sibling of
[`rain_transform_mode`](@ref) with the same three values and the same `:none` default. `n_r`
is a TOTAL (`n̄_r ≡ 0`), so the transform pair is [`rain_number_slot`](@ref) /
[`recover_n_r`](@ref) and its width is `physical_params[:mu_rain_n]` (default `1.0`, in
#/m³ — the scale below which a number density is meteorologically meaningless, ten orders
of magnitude away from the `rain_mu` of the density slot).

Independent of `rain_transform`: mass and number ring independently and the arms have to be
separable. Meaningless unless `rain_moments == 2`, and inert then unless set.
"""
@inline function rain_number_transform_mode(options)
    mode = get(options, :rain_number_transform, :none)::Symbol
    (mode === :none || mode === :bhyp || mode === :bhyp_smooth) ||
        error("options[:rain_number_transform] = :$(mode) is not recognized; use :none " *
              "(the default: the slot is the rain number density n_r), :bhyp (Ooyama's " *
              "biased hyperbolic control variable with his quasi-inverse) or :bhyp_smooth " *
              "(the same forward map with the strict C^inf inverse)")
    return mode
end

# ── Ice: the twelve appended species slots ───────────────────────────────────

"""
The ROLE names of the twelve ice prognostic slots, SPECIES-MAJOR and in registration order —
the order [`mc_var_names`](@ref) appends them in and therefore the order their indices run in.

Species `k ∈ {1, 2, 3}` are ISHMAEL's (Jensen et al. 2017) planar-nucleated crystals,
columnar-nucleated crystals and aggregates. Each carries four PER-UNIT-VOLUME moments:

| name | symbol | units | what it is |
|---|---|---|---|
| `rho_ik` | ρ_{i,k} | kg/m³ | mass density |
| `n_ik`   | n_{i,k} | #/m³  | number density |
| `a_ik`   | a_{i,k} = n⟨a²c⟩ | m³/m³ | spheroid volume moment; (4/3)π a is the volume fraction |
| `c_ik`   | c_{i,k} = n⟨c²a⟩ | m³/m³ | the second volume moment; φ = c/a is the aspect ratio |

The moments are carried multiplied by the number density, rather than as bare per-particle
averages, precisely so that all four are DENSITIES and all four take the same continuity form
as every other prognostic in the set (reference/Scythe_moist_compressible.tex, Eq.
ice_moments). Two volume moments rather than one because deposition density and aspect ratio
evolve independently: a crystal can gain mass without gaining volume, or change habit at
constant mass.
"""
const MC_ICE_VARS = ("rho_i1", "n_i1", "a_i1", "c_i1",
                     "rho_i2", "n_i2", "a_i2", "c_i2",
                     "rho_i3", "n_i3", "a_i3", "c_i3")

"""
The `physical_params` key holding the transform width `mu` for each entry of
[`MC_ICE_VARS`](@ref), positionally. `mu` is DIMENSIONAL — it carries the units of the
variable it transforms — so the four moment kinds cannot share one number the way the twelve
slots share one transform FAMILY: the mass, the number and the two volume moments are nine
and ten decades apart (kg/m³, #/m³, m³/m³). Each width is set two to five decades BELOW the
field it transforms — see [`MC_ICE_MU_DEFAULTS`](@ref) for the sizing principle, the
measurement behind it, and what goes wrong when `mu` is set at or above the field amplitude.
"""
const MC_ICE_MU_KEYS = (:mu_ice, :mu_ice_n, :mu_ice_a, :mu_ice_c,
                        :mu_ice,  :mu_ice_n, :mu_ice_a, :mu_ice_c,
                        :mu_ice,  :mu_ice_n, :mu_ice_a, :mu_ice_c)

"""
Defaults for [`MC_ICE_MU_KEYS`](@ref), positionally. See there for why they differ by moment
kind, and the SIZING PRINCIPLE below for why they are the values they are. All four remain
per-run tunables through `physical_params`.

# The sizing principle: `mu` must sit WELL BELOW the field amplitude

`bhyp` is a stretch that acts only where `rho ≲ mu`; above that it is affine
(`bhyp(rho) → rho/2`) and `J → 1/2`. Sized correctly, the transform buys a recovered value
bounded below by zero AND a genuine change of variables in the region that needs it. Sized
too WIDE — `mu` at or above the field amplitude — `bhyp` degenerates: measured at the O01 ice
arm's own amplitude (peak `rho_i1 = 2.7e-10` kg/m³ against the former `mu_ice = 1e-7`, i.e.
`mu` 370x ABOVE the field), `bhyp(rho)/rho = 0.9987` and `J = 0.9973`. The map is the IDENTITY
to 0.3 %, so nothing is stretched and nothing is smoothed; all that survives of the transform
is `ahyp`'s hard floor at zero. That is a bare positivity clip on a two-signed spline
excursion — exactly the flooring that `condensate_transform_mode` documents as rejected for
the cloud ("Flooring rectifies a two-signed excursion into one-signed"), applied to the ice
slot but NOT to slot 3, which keeps integrating the signed sum. The manufactured mass is the
difference, and it shows up as a reconciliation gap between slot 3 and the vapor slot.

So each width is set two to five decades BELOW the field scale it transforms:

| moment | field scale (O01 ice arm) | `mu` | field/`mu` |
|---|---|---|---|
| mass `rho_i` | ~1e-9 kg/m³ | 1e-12 | 1e3 |
| number `n_i` | ~1e3 #/m³ | 1e-2 | 1e5 |
| volume `a_i`, `c_i` | ~`n·r³` = 1e3·(2e-5)³ = 8e-12 m³/m³ | 1e-16 | 1e4 |

The two VOLUME widths are unchanged: at `n·r³` for a 20 µm crystal at 1e3 /m³ the field is
already four decades above 1e-16, so they were correctly sized from the start and moving them
for symmetry with the other two would be a change with no measurement behind it.
"""
const MC_ICE_MU_DEFAULTS = (1.0e-12, 1.0e-2, 1.0e-16, 1.0e-16,
                            1.0e-12, 1.0e-2, 1.0e-16, 1.0e-16,
                            1.0e-12, 1.0e-2, 1.0e-16, 1.0e-16)

"""
    ice_microphysics(options) -> Symbol

Which ice scheme the moist-compressible set carries. `:none` (the default, and the state of
every configuration that does not set the key) registers no ice slot at all and is BITWISE
the code that had no ice. `:ishmael` appends the twelve slots of [`MC_ICE_VARS`](@ref) and
turns on the ice thermodynamics — the `ρ_i` term of [`retrieve_temperature`](@ref), `q_i C_i`
in the mixture heat capacities and the entropy, and `ρ_i` in the `res_rho_t` water
budget the vapor is reconciled against.

**Validation: `:ishmael` REQUIRES `options[:rain_moments] == 2`.** ISHMAEL's ice-rain physics
— rain freezing, the collection of rain by ice, and the shedding and melting branches that run
backwards through them — is defined against a rain size distribution with a prognostic
intercept. Run against single-moment rain it would read a drop size that the rain closure does
not actually carry, and the disagreement would be silent. The check is here rather than at
first use so it fires at configuration time.

**The physics is LIVE.** Deposition and sublimation run through the shared prognostic `Q_ss`
(TeX Eqs. dep_rate, Qss_ice_final), so the Wegener-Bergeron-Findeisen competition is a
property of the equation set rather than an arbitration bolted onto it; the full ISHMAEL
process set — DeMott/homogeneous/Bigg nucleation, Hallett-Mossop splintering, ice-cloud and
ice-rain riming with the wet-growth branch, aggregation and melting — supplies the rest; the
Mitchell-Heymsfield fall speeds sediment each moment at its own weighted speed. See
[`mc_ice_sources!`](@ref).

**Zero ice is still exactly inert where no ice can form.** Every rate is gated on a STATE
test — `q_i ≤ QSMALL` for a species, `T ≤ T_0` for aggregation and deposition, the nucleation
windows for the rest — so in air that is warm throughout with no initial ice, every ice source
and every fall speed is an exact `0.0` and the ten common slots reproduce the ice-free run
BITWISE. In SUBFREEZING supersaturated air it is not inert and must not be: DeMott nucleation
fires, which is the physics.

The tables (`data/ishmael_tables.jld2`, 44.5 MB, gitignored) are loaded once per tile at
construction; a missing file is refused there rather than mid-run. See
[`load_ishmael_tables_or_error`](@ref).

**`options[:ice_var_check]` (default `true`) — FLAGGED FOR REVIEW.** ISHMAEL's `var_check`
writes its moment-consistency re-diagnosis back into the state every step; this port evaluates
the rates at the effective moments but carries the raw ones, and nothing else in the set
restores them. The write-back is therefore reinstated as a SOURCE on the number and the two
volume moments (never on the mass), which is a one-step state repair and carries a `Δt`.
Setting this `false` removes it and reproduces the divergence it was added for — see the block
in [`mc_ice_sources!`](@ref) for the measurement and the argument.
"""
@inline function ice_microphysics(options)
    mode = get(options, :ice_microphysics, :none)::Symbol
    (mode === :none || mode === :ishmael) ||
        error("options[:ice_microphysics] = :$(mode) is not recognized; use :none (the " *
              "default: no ice slots, bitwise the liquid-only set) or :ishmael (the twelve " *
              "ISHMAEL ice slots — three species x mass, number and two volume moments)")
    if mode === :ishmael && rain_moments(options) != 2
        error("options[:ice_microphysics] = :ishmael requires options[:rain_moments] = 2 " *
              "(it is $(rain_moments(options))). The ISHMAEL ice-rain physics — rain " *
              "freezing, ice collection of rain, shedding and melting — is defined against " *
              "a rain DSD whose intercept comes from a prognostic (rho_r, n_r) pair; run " *
              "against single-moment rain it would read a drop size the rain closure does " *
              "not carry, and nothing downstream would notice.")
    end
    return mode
end

"""
    ice_transform_mode(options) -> Symbol

Which control variable the TWELVE ice slots carry — ONE family key, `options[:ice_transform]`,
with the same three values and the same `:none` default as
[`rain_transform_mode`](@ref)/[`condensate_transform_mode`](@ref).

One key for twelve slots rather than twelve keys, because the moments of a species are not
independently meaningful: `φ_k = c_{i,k}/a_{i,k}` and `ρ_{i,k}/(4/3 π a_{i,k})` are ratios of
two slots, and transforming one member of a ratio while the other stays linear would put a
habit diagnostic through two different low-pass filters. The WIDTHS still differ per moment
kind — see [`MC_ICE_MU_KEYS`](@ref) — because `mu` is dimensional.

All twelve are TOTALS (`f̄ ≡ 0`; there is no resting ice field for a perturbation to be
measured against), so the transform pair is [`total_slot`](@ref)/[`recover_total`](@ref), the
same one rain and the rain number use.
"""
@inline function ice_transform_mode(options)
    mode = get(options, :ice_transform, :none)::Symbol
    (mode === :none || mode === :bhyp || mode === :bhyp_smooth) ||
        error("options[:ice_transform] = :$(mode) is not recognized; use :none (the " *
              "default: each slot is the ice density/number/volume moment itself), :bhyp " *
              "(Ooyama's biased hyperbolic control variable with his quasi-inverse) or " *
              ":bhyp_smooth (the same forward map with the strict C^inf inverse)")
    return mode
end

"""
    ice_mu(physical_params, j) -> Float64

The transform width for the `j`-th entry of [`MC_ICE_VARS`](@ref), from
[`MC_ICE_MU_KEYS`](@ref) with the [`MC_ICE_MU_DEFAULTS`](@ref) fallback. Setup-path only —
`mc_driver!` reads the four distinct values once per column, not twelve.
"""
@inline ice_mu(physical_params, j::Int) =
    get(physical_params, MC_ICE_MU_KEYS[j], MC_ICE_MU_DEFAULTS[j])

# ── Slot NAMES under a transform ──────────────────────────────────────────────
# A transformed slot no longer holds what its name says, and the name is what every consumer
# keys off: Springsteel builds the CSV/netCDF column headers straight from `GridParameters.vars`
# (`Springsteel/src/io.jl`), so the output file is the only thing a postprocessor sees. Keeping
# the name `rho_c` while the column held `bhyp(rho_c)` is exactly how `benchmarks/o01_movie.jl`
# came to render half-amplitude cloud with spurious negative lobes against a run whose own
# diagnostics recorded `min_rho_c = 0`. The transformed slots are therefore RENAMED, following
# Ooyama's own notation, in which `mu` is the mixing ratio and `nu` its transform. (`n` was
# considered and rejected: it reads as number concentration in a double-moment scheme.)
# The ice slots follow the same rule: `nu_` prefixed to a tag that says which moment of which
# species the column holds (`nu_i1` mass, `nu_ni1` number, `nu_ai1`/`nu_ci1` the two volume
# moments), so no two of the fifteen transformable slots can collide and no output column ever
# claims to hold a density it does not.
const MC_NU_ALIAS = Dict("rho_c" => "nu_c", "rho_r" => "nu_r", "n_r" => "nu_nr",
                         "rho_i1" => "nu_i1", "n_i1" => "nu_ni1",
                         "a_i1" => "nu_ai1", "c_i1" => "nu_ci1",
                         "rho_i2" => "nu_i2", "n_i2" => "nu_ni2",
                         "a_i2" => "nu_ai2", "c_i2" => "nu_ci2",
                         "rho_i3" => "nu_i3", "n_i3" => "nu_ni3",
                         "a_i3" => "nu_ai3", "c_i3" => "nu_ci3")

"""
    condensate_var_name(options) -> String
    rain_var_name(options) -> String
    rain_number_var_name(options) -> String

The name slot 9 / slot 8 / the appended rain-number slot carries under the current options:
`"rho_c"` / `"rho_r"` / `"n_r"` untransformed, `"nu_c"` / `"nu_r"` / `"nu_nr"` under a
transform. Use these — never a literal — when building the `vars`, `BCL`/`BCR`/`BCB`/`BCT`,
`l_q`, `positivity` or `spline_filter` dicts of a `GridParameters`, all five of which are
keyed by name.
"""
condensate_var_name(options) =
    condensate_transform_mode(options) === :none ? "rho_c" : "nu_c"
@doc (@doc condensate_var_name)
rain_var_name(options) = rain_transform_mode(options) === :none ? "rho_r" : "nu_r"
@doc (@doc condensate_var_name)
rain_number_var_name(options) =
    rain_number_transform_mode(options) === :none ? "n_r" : "nu_nr"

"""
    ice_var_names(options) -> NTuple{12,String}

The names the twelve ice slots carry under the current options: [`MC_ICE_VARS`](@ref) itself
when `options[:ice_transform]` is `:none`, their [`MC_NU_ALIAS`](@ref) aliases under a
transform. Empty of meaning unless `ice_microphysics(options) === :ishmael`, which is the only
thing that registers them.
"""
@inline function ice_var_names(options)
    on = ice_transform_mode(options) !== :none
    return ntuple(j -> on ? MC_NU_ALIAS[MC_ICE_VARS[j]] : MC_ICE_VARS[j], 12)
end

"""
    mc_var_names(options; cyl = false) -> Vector{String}

The ordered prognostic-slot names for the moist-compressible set under the current options —
[`MC_VARS`](@ref) (or `MC_VARS_CYL` for the 11-slot cylindrical/3-D variants) with slots 8 and
9 renamed by [`rain_var_name`](@ref) / [`condensate_var_name`](@ref). With no transform declared
this returns the canonical list unchanged, so every existing configuration is bit-identical.

**Appending a slot.** Slots 1-9 (and `v` at 10 on the cylinders) appear as HARDCODED LITERALS
throughout `mc_driver!`, the acoustic solvers and `mc_boundary_layer.jl`, so every later slot
is APPENDED after that block and never inserted. Its index is therefore geometry-dependent
(`rho_v` is 10 on XZ and 11 on the cylinders, `n_r` 11 and 12) and must be resolved BY NAME —
[`mc_slot`](@ref), cached once per tile in [`MCSlots`](@ref), never a per-column lookup. This
is the pattern the ice categories follow — and do follow: the twelve of
[`MC_ICE_VARS`](@ref) are appended AFTER `n_r`, species-major, so on the XZ slice they run 12
through 23 and on the cylinders 13 through 24.

`rho_v` is in the constants rather than in the optional block below because the vapor is
prognostic in every configuration of this set; only `n_r` and the ice family are conditional.
"""
function mc_var_names(options; cyl::Bool = false)
    names = copy(cyl ? MC_VARS_CYL : MC_VARS)
    names[8] = rain_var_name(options)
    names[9] = condensate_var_name(options)
    # ── APPENDED optional slots, in registration order. Absent by default, so the list
    #    every existing configuration gets back is bit-identical. ──
    rain_moments(options) == 2 && push!(names, rain_number_var_name(options))
    # The ice family, after the rain number (which `ice_microphysics` requires be present),
    # species-major so a species' four moments are contiguous.
    if ice_microphysics(options) === :ishmael
        for nm in ice_var_names(options)
            push!(names, nm)
        end
    end
    return names
end

"""
    mc_slot(vars, role) -> Int

The slot index of a water species by ROLE rather than by name, accepting either the density
name or its transformed alias. `role` is `"rho_v"`, `"rho_c"`, `"rho_r"`, `"n_r"`, or any
entry of [`MC_ICE_VARS`](@ref). (`rho_v` carries no transform, so for it this is a plain
`vars` lookup with a loud failure message.)

Every `vars["rho_c"]`-style lookup in the kernel goes through this, so that a transformed
configuration cannot produce a `KeyError` deep in a solver — and, more importantly, so that the
lookup can never silently *succeed* against the wrong convention.

This is a `Dict{String,Int}` lookup and must NOT appear in the per-column path: resolve
appended slots once per tile through [`MCSlots`](@ref) instead.
"""
@inline function mc_slot(vars, role::AbstractString)
    haskey(vars, role) && return vars[role]
    alias = get(MC_NU_ALIAS, role, "")
    (alias != "" && haskey(vars, alias)) && return vars[alias]
    error("no slot named \"$role\"" * (alias == "" ? "" : " or \"$alias\"") *
          " in grid_params.vars (it has $(sort(collect(keys(vars))))). Build the variable " *
          "list with `Scythe.mc_var_names(options)` so the transformed slots are named " *
          "consistently.")
end

"""
    mc_optional_slot(vars, role) -> Int

[`mc_slot`](@ref) for an APPENDED, optional slot: the index if the configuration registered
it, `0` if it did not. Non-throwing, because absence is a legitimate answer here — the
canonical nine-slot configuration has no `n_r` and no ice.

For initializers and other setup code, which must seed a slot only if it exists. The
per-column path uses [`MCSlots`](@ref) instead; this is a name lookup.
"""
@inline function mc_optional_slot(vars, role::AbstractString)
    haskey(vars, role) && return vars[role]
    alias = get(MC_NU_ALIAS, role, "")
    (alias != "" && haskey(vars, alias)) && return vars[alias]
    return 0
end

"""
    mc_ice_slot_indices(vars) -> NTuple{12,Int}

The indices of the twelve [`MC_ICE_VARS`](@ref) slots in registration order, each `0` when the
configuration did not register it — [`mc_optional_slot`](@ref) twelve times, resolved once so
an initializer does not do a name lookup per gridpoint.
"""
@inline mc_ice_slot_indices(vars) =
    ntuple(j -> mc_optional_slot(vars, MC_ICE_VARS[j]), 12)

"""
    seed_ice_zero!(physical, i, ice_i) -> Nothing

Seed every registered ice slot of gridpoint `i` with zero. A no-op for a configuration that
registered none, so every existing initializer is unaffected.

Zero is the right value under EVERY transform without threading a mode through, for the same
reason `rain_slot(0.0, …) === 0.0`: `bhyp(0) == 0` exactly. And zero is the right PHYSICS for
every idealized initializer in this file — a warm bubble and a balanced vortex have no ice to
start with, and (with the process rates live) nucleation is what will put it there.
"""
@inline function seed_ice_zero!(physical, i::Int, ice_i::NTuple{12,Int})
    @inbounds for s in ice_i
        s > 0 && (physical[i, s, 1] = 0.0)
    end
    return nothing
end

"""
    check_mc_var_names(model) -> Nothing

Refuse a configuration whose name-keyed `GridParameters` dicts disagree with the declared
transforms and optional slots. A no-op when no transform is on and the slot list is the
canonical one, so no existing configuration is affected.

This exists because the failure it catches is SILENT. `vars` is the only one of the five
name-keyed dicts whose miss is loud; `_resolve_spline_filter` returns `nothing` rather than
throwing, so a leftover `"rho_r" => NaturalBC()` under the rain transform would quietly leave
slot 8 on the default Neumann fit — which forces a zero flux derivative at the ground and traps
the falling rain at the surface instead of letting sedimentation carry it out of the domain.
Likewise a stale `l_q` key removes the water filter the O01 configuration depends on for
stability. Neither would raise anything; both would change the physics.

An APPENDED optional slot is checked in the other direction too: `vars` must actually declare
it. Nothing else can — the appended index is geometry-dependent, so a `vars` built from a
stale name list has no slot for the tendency to land in.
"""
function check_mc_var_names(model::ModelParameters)
    ctrans = condensate_transform_mode(model.options)
    rtrans = rain_transform_mode(model.options)
    nrtrans = rain_number_transform_mode(model.options)
    nmoments = rain_moments(model.options)
    # `ice_microphysics` is where the `:ishmael` ⇒ `rain_moments == 2` requirement is raised,
    # so call it even on the early-return path: a configuration that declares ice against
    # single-moment rain must fail HERE, at tile creation, and not at whatever later point
    # something first reads a drop size that is not carried.
    ice = ice_microphysics(model.options)
    itrans = ice_transform_mode(model.options)
    (ctrans === :none && rtrans === :none && nrtrans === :none && nmoments == 1 &&
     ice === :none) && return nothing

    gp = model.grid_params
    expected = Set(mc_var_names(model.options;
                                cyl = haskey(gp.vars, "v") &&
                                      gp.vars["v"] == 10))
    stale = String[]
    ctrans === :none || push!(stale, "rho_c")
    rtrans === :none || push!(stale, "rho_r")
    nrtrans === :none || push!(stale, "n_r")
    if itrans !== :none
        for nm in MC_ICE_VARS
            push!(stale, nm)
        end
    end

    # The appended slots, in the other direction: declared by the options, so `vars` must
    # carry them. (Only `vars` — the BC/l_q/positivity dicts legitimately fall back to
    # "default" for a name they do not mention.)
    if nmoments == 2
        nrname = rain_number_var_name(model.options)
        haskey(gp.vars, nrname) ||
            error("options[:rain_moments] = 2 declares the prognostic rain number slot " *
                  "\"$nrname\", but grid_params.vars does not have it (it has " *
                  "$(sort(collect(keys(gp.vars))))). The slot is APPENDED after the fixed " *
                  "1-9 (+v) block and after the vapor slot, so its index is " *
                  "geometry-dependent and only " *
                  "`Scythe.mc_var_names(options)` knows it — build `vars` from that.")
    end
    if ice === :ishmael
        for nm in ice_var_names(model.options)
            haskey(gp.vars, nm) ||
                error("options[:ice_microphysics] = :ishmael declares the ice slot " *
                      "\"$nm\", but grid_params.vars does not have it (it has " *
                      "$(sort(collect(keys(gp.vars))))). All twelve are APPENDED after " *
                      "n_r, so their indices are geometry-dependent and only " *
                      "`Scythe.mc_var_names(options)` knows them — build `vars` from that.")
        end
    end

    for (label, d) in (("vars", gp.vars), ("BCL", gp.BCL), ("BCR", gp.BCR),
                       ("BCB", gp.BCB), ("BCT", gp.BCT), ("l_q", gp.l_q),
                       ("positivity", gp.positivity),
                       ("spline_filter", gp.spline_filter))
        for key in keys(d)
            key == "default" && continue
            key in expected && continue
            hint = key in stale ?
                " — that species is transformed, so the key must be " *
                "\"$(MC_NU_ALIAS[key])\"" : ""
            error("grid_params.$label declares \"$key\", which is not a slot name of this " *
                  "configuration$(hint). Expected names: $(sort(collect(expected))). " *
                  "Build every name-keyed dict from `Scythe.mc_var_names(options)`; a stale " *
                  "key in $label would be ignored silently rather than raising.")
        end
    end
    return nothing
end

"""
    condensate_floor_mode(options) -> Bool

Whether `rho_liq` is floored at the DIAGNOSTIC INTERFACE. `true` for
`options[:condensate_floor] = :diagnostic`; `false` for `:none`, which is the default and the
state of every configuration that does not set the key.

**What the floor is, and what it is not.** The spline undershoot puts `rho_c` a few `1e-4`
below zero on the flanks of a cloud. The raw value then enters the closed-form temperature
retrieval, where `∂T/∂ρ_liq = L_v/D` (Eq. of [`retrieve_temperature`](@ref)) turns it into a
cold anomaly reaching −54 K on O01, and through `rho_vs(T)` into a collapsed saturation
density, false supersaturation and false nucleation. `:diagnostic` replaces `rho_liq` by
`max(rho_c, 0) + max(rho_r, 0)` at the retrieval, `q_l` (hence `C_vt`/`R_m`/`gamma_m`),
`Q_s_energy`, the entropy and the sedimentation energy — and NOWHERE else.

It is **not** [`clamp_water!`](@ref). Nothing is written back to a prognostic slot, so it is
memoryless, converts no mass, and cannot pump latent heat: the continuity terms, the rate
budgets and the state itself keep reading the raw value. `clamp_water!` rectifies a two-signed
oscillation one step at a time and takes O01 non-finite in 23 min; this changes what the
thermodynamics READS on the step it reads it, and leaves no trace.

What it does NOT fix: the negative condensate is still there, still in continuity, and still
accumulating (`reference/FINDINGS_CONDENSATE_STAGE1.md` §1 — the reservoir reaches 2.75x the
real cloud mass). This addresses the consequence, not the cause. The cause needs the
control-variable transform, [`condensate_transform_mode`](@ref).

**Measured (2026-07-30, quick O01, 3600 s).** The win is where the cold anomalies live:

| diagnostic | NOPRECIP `:none` | NOPRECIP `:diagnostic` | precip `:none` | precip `:diagnostic` |
|---|---|---|---|---|
| `min_rho_v_gm3` | −0.186 | **−1.58e-4** | −0.724 | −0.759 |
| `neg_rho_v_points` | 2033 | **0** | 750 | 716 |
| `entropy_prod_rate` (over rho_v>0) | 34.81 | 44.91 | 0.520 | 0.470 |
| `max_w` | 6.600 | 6.085 | 9.869 | 11.15 |
| `peak_rain_rate_gm2s` | — | — | 59.94 | 68.64 |
| `water_mass_drift_pct` | 1.07e-6 | 1.08e-6 | −4.093 | −3.674 |

On the cloud-only storm the floor removes NEGATIVE VAPOR ENTIRELY — 2033 points to zero, the
worst value improving 1180x — which is the mechanism confirmed: a negative liquid density
drives `rho_vs` down through `T`, manufacturing supersaturation, and (with the vapor still
retrieved as the residual, as it was when this was measured) that is where it landed. With precipitation active the effect is smaller and mixed (`max_w` +13 %,
rain rate +15 %, water and energy drift both improved), because rain removes the condensate
before the lobes grow as large.

**Correction (2026-07-30).** An earlier version of this note claimed the irreversible entropy
production fell 14.5x here. It did not. `mc_entropy_production` clamped `H` at `1e-12` where
the vapor is negative, so its raw value tracked the NEGATIVE-VAPOR COUNT rather than the
irreversibility: 94.7 % of the NOPRECIP total came from clamped points. Restricted to the
points where the quantity exists, the floor RAISES production slightly (34.81 → 44.91) while
eliminating every excluded point. The diagnostic now returns that count beside the rate so the
two cannot be read apart again.

Under a working [`condensate_transform_mode`](@ref) this floor should become INACTIVE — the
recovered density is already non-negative — so a transformed run is the test of whether the
transform subsumes it.
"""
@inline function condensate_floor_mode(options)
    mode = get(options, :condensate_floor, :none)::Symbol
    (mode === :none || mode === :diagnostic) ||
        error("options[:condensate_floor] = :$(mode) is not recognized; use :none " *
              "(the default: rho_liq is read raw everywhere) or :diagnostic (floor " *
              "rho_liq at the thermodynamic interface only, leaving state and continuity raw)")
    return mode === :diagnostic
end

"""
    install_positivity_bounds!(grid, ref_state, model) -> Nothing

Convert the declared physical positivity bounds into the reference-aware coefficient bounds
each spline leg actually needs, overriding what the factory installed.

The factory can only express a CONSTANT bound, which is exactly right for a total (`rho_r`,
bound 0) and for a perturbation whose reference is identically zero (`rho_c` on a cloud-free
base, which is every current configuration). It cannot express the bound for a perturbation
against a nonzero reference, and that bound differs by leg:

- **k-leg** — fits the field itself, so the bound is `-ρ̄(z)`, applied through the
  support-minimum rule (`set_lower_bound_from_profile!`).
- **i-leg** — fits the k-direction B-coefficients `⟨φ_z, u⟩`, so the bound is the CONSTANT
  `-⟨φ_z, ρ̄⟩` for each mode `z`, i.e. the negated SB coefficients of the reference. This is
  the scaling the RiRk factory refuses to guess at.

A no-op when the reference is zero, so the common path is untouched.
"""
function install_positivity_bounds!(grid, ref_state, model::ModelParameters)

    pos = model.grid_params.positivity
    # The transform supersedes the limiter for the species it acts on: a box constraint on
    # the CONTROL variable's coefficients would bound nu, not the density, and the two are not
    # the same constraint. Declaring both is a configuration error, not something to resolve
    # silently by precedence. Checked per species, since the two transforms are independent
    # knobs — a run may transform cloud while rain still takes the coefficient bound, which is
    # the arm that showed the cloud transform did no harm.
    for (name, opt, mode) in (("rho_c", :condensate_transform,
                               condensate_transform_mode(model.options)),
                              ("rho_r", :rain_transform,
                               rain_transform_mode(model.options)),
                              ("n_r", :rain_number_transform,
                               rain_number_transform_mode(model.options)))
        (mode !== :none && haskey(pos, name)) &&
            error("positivity is declared for \"$name\" while options[:$opt] is on. The " *
                  "transform makes the recovered density non-negative by construction; a " *
                  "coefficient bound would constrain the control variable instead, which is " *
                  "a different constraint. Drop \"$name\" from positivity.")
    end
    # The twelve ice slots share ONE transform key (`ice_transform`; see
    # `ice_transform_mode` for why the family is not twelve independent knobs), so the
    # exclusivity is one mode against twelve names.
    let imode = ice_transform_mode(model.options)
        if imode !== :none
            for name in MC_ICE_VARS
                haskey(pos, name) &&
                    error("positivity is declared for \"$name\" while " *
                          "options[:ice_transform] is on. The transform makes the recovered " *
                          "moment non-negative by construction; a coefficient bound would " *
                          "constrain the control variable instead, which is a different " *
                          "constraint. Drop \"$name\" from positivity.")
            end
        end
    end
    isempty(pos) && return nothing
    grid.kbasis isa Springsteel.SplineBasisArray || return nothing
    vars = model.grid_params.vars

    for (name, v) in vars
        spec = Springsteel._resolve_spline_filter(pos, name, :k)
        spec_i = Springsteel._resolve_spline_filter(pos, name, :i)
        (spec === nothing && spec_i === nothing) && continue
        prof = positivity_reference_profile(name, ref_state)
        prof === nothing && continue                 # total: the factory bound is correct
        all(iszero, prof) && continue                # zero reference: likewise

        if spec !== nothing
            spec == 0.0 || error("positivity[\"$name\"][:k] = $spec: only a bound of 0.0 is " *
                                 "supported for a variable carried against a reference.")
            set_lower_bound_from_profile!(grid.kbasis.data[v], prof)
        end
        if spec_i !== nothing
            spec_i == 0.0 || error("positivity[\"$name\"][:i] = $spec_i: only 0.0 is supported.")
            # b̄_z = ⟨φ_z, ρ̄⟩ on the k basis. Safe to borrow the k-column's buffers: this
            # runs at initialization, before any transform.
            kcol = grid.kbasis.data[v]
            kcol.uMish .= prof
            SBtransform!(kcol)
            bbar = copy(kcol.b)
            b_iDim = model.grid_params.b_iDim
            for z in 1:model.grid_params.b_kDim
                set_lower_bound!(grid.ibasis.data[z, v], fill(-bbar[z], b_iDim))
            end
        end
    end
    return nothing
end

"""
    check_condensate_transform_ic(ref_state, model) -> Nothing

Warn once when a transformed run starts on a CLOUDY reference state.

Slot 9 then carries `n' = bhyp(rho_c) - bhyp(ρ̄_c)`, and every producer of an initial
condition has to know that: [`condensate_slot`](@ref) is the conversion, and the `*_mc!`
initializers take `condensate_transform`/`condensate_mu` keywords for it. On a cloud-free
reference (`ρ̄_c ≡ 0`, which is every benchmark configuration to date) the conversion of a
cloud-free state is exactly `0.0` under both conventions, so nothing has to be threaded and
nothing can go wrong. On a cloudy reference an initial condition written in DENSITY would be
silently reinterpreted as a control variable — off by a factor of two in the linear regime,
and not detectable from the run.

It is a warning and not an error because a cloudy reference is a legitimate configuration --
`bf02_moist` is one, and it threads the conversion -- and because the condition cannot be
checked from the state: both conventions give exactly 0.0 where there is no cloud, and where
there is cloud neither is distinguishable from the other without knowing the intended density.
Refusing outright would block correct configurations to catch a mistake it cannot actually
detect; naming the requirement is the most the model can honestly do.
"""
function check_condensate_transform_ic(ref_state, model::ModelParameters)
    uses_pressure_reference(model.equation_set) || return nothing
    condensate_transform_mode(model.options) === :none && return nothing
    prof = Springsteel.ref_rho_c(ref_state)
    prof isa Number && return nothing
    all(iszero, view(prof, :, 1)) && return nothing
    @warn """options[:condensate_transform] is on and the reference state is CLOUDY
      (ρ̄_c is not identically zero). Slot 9 therefore carries bhyp(rho_c) - bhyp(ρ̄_c), NOT
      the density perturbation, and the initial condition must have been written that way:
      `condensate_slot` is the conversion and the `*_mc!` initializers take
      `condensate_transform` / `condensate_mu` keywords for it. An initial condition written
      in density is read here as a control variable and is wrong by roughly a factor of two
      in the linear regime, with nothing in the run to show it.

      This cannot be verified from the state -- both conventions give exactly 0.0 where
      there is no cloud, and neither is distinguishable from the other where there is. It is
      the configuration's responsibility. In-tree, `bf02_moist.jl` threads it; O01 and the
      TC configurations have cloud-free references and are unaffected."""
    return nothing
end

"""
    _warn_unbounded_master_output(ref_state, model)

Warn once when a reference-aware bound is in force on the workers but cannot be installed on
the master's patch.

The master builds its patch from `model.grid_params` alone (`initialize_model`), never
constructing a reference state — only `createModelTile` does, per worker. So when a bounded
perturbation has a NONZERO reference, the master's `gridTransform!` (output/CFL/restart
cadence only) reconstructs without the offset bound. The dynamics are unaffected: the state's
coefficients are bounded on the workers and it is those that are integrated. Only the written
diagnostics can show an undershoot the model itself never saw.

Silent for every current configuration, all of which have `ρ̄_c ≡ 0`.
"""
function _warn_unbounded_master_output(ref_state, model::ModelParameters)
    pos = model.grid_params.positivity
    isempty(pos) && return nothing
    for (name, _) in model.grid_params.vars
        Springsteel._resolve_spline_filter(pos, name, :k) === nothing &&
            Springsteel._resolve_spline_filter(pos, name, :i) === nothing && continue
        prof = positivity_reference_profile(name, ref_state)
        (prof === nothing || all(iszero, prof)) && continue
        @warn """Positivity bound on "$name" is reference-offset (ρ̄ is not identically zero).
          The worker tiles and patches carry the correct bound, so the INTEGRATED state is
          bounded. The master's patch is built without a reference state, so the output
          written by gridTransform! is reconstructed unbounded and may show an undershoot the
          model never integrated. See install_positivity_bounds!.""" maxlog = 1
    end
    return nothing
end

"""
    clamp_water!(mtile, colstart, colend)

MEASURE the negative water in one column, and — only under
`options[:clamp_water] = true` — floor it at zero.

# Why the measurement is the point

Negative condensate is unphysical, so the instinct to clamp it is right. But a floor
cannot manufacture resolution, and at this scheme's ringing amplitudes it does real
damage.

`ρ_c` and `ρ_r` are positive-definite fields with sharp, spiky structure — a rain shaft
or a cloud core. A cubic B-spline column with too few nodes to resolve that spike
UNDERSHOOTS on its flanks, and the undershoot scales with how badly the spike is
under-resolved, not with roundoff. Flooring it then rectifies a zero-mean oscillation:
only the negative lobes are touched, so each step converts `δ` of vapor to liquid and the
retrieval faithfully releases `L_v·δ` of latent heat, one-signed and accumulating.

The size of that kick follows from differentiating the closed-form retrieval
([`retrieve_temperature`](@ref)) — `∂T/∂ρ_liq = L_v(T)/D` with
`D = C_factor − ρ_liq(C_pv − C_l)` — so flooring `δ` implies

    ΔT = L_v·δ / D          (`:worst_dT`, estimated with L_v0 and D ≈ C_factor)

Two measured regimes, four orders of magnitude apart:

- **Fit-level noise.** The balanced TC vortex moves ~5e-7 kg/m³ per column-step:
  `ΔT ~ 1e-3 K`. A floor there is free and harmless.
- **An unresolved spike.** `o01_rainfall`'s rain shafts reach `min(ρ_r) ≈ -1.8 g/m³` at
  500 m vertical spacing — `ΔT ≈ 4.4 K` in a SINGLE step at low levels, and ~68 K in the
  thin air aloft where it actually detonated. With the floor on, that run goes non-finite
  at t ≈ 23 min as convection erupts (`T = 9 K`, `E_t < 0`, `ρ_vs = Inf`, `Qdot = NaN`);
  with it off, the same run completes.

Re-accounting the floor does not rescue the second regime, it only chooses which budget
absorbs it. Holding `p` and subtracting `L_v(T)·δ` from `E_t` alongside the partition
change leaves `T` exactly invariant — but that is a `L_v·δ ≈ 4.5 kJ/m³` energy sink per
event, ~2 % of the local `E_t` per step. A temperature bias becomes an equal-sized
conservation bias. The amplitude is the problem; the bookkeeping is not.

So the default is to MEASURE and WARN rather than to floor, and to read a large
`worst_dT` as what it is: a request for more vertical nodes.

**Not flooring is not the same as harmless.** The `max(ρ_c, 0)` guards live inside the RATE
functions only ([`qss_condensation_rates`](@ref), autoconversion/collection), so an
undershoot cannot manufacture condensation or precipitation. But the raw value feeds the
continuity/advection terms AND `ρ_liq`, hence [`retrieve_temperature`](@ref) — so it drags
the temperature with it by the same `L_v·δ/D`. Measured on `o01_rainfall` quick, the
temperature attributable to negative liquid reaches **8.1 K** on the cloud flank at
t = 50 min. (The pre-`ρ_c` formulation could not do this: its `ρ_v` was clamped to
`[0, ρ_w − ρ_r]`, so the effective liquid the retrieval saw was `ρ_w − ρ_v ≥ ρ_r ≥ 0`.)

The difference between flooring and not is therefore the SIGN STRUCTURE, not the presence
of an error: unfloored, the anomaly oscillates with the ringing (cold on the undershoot,
warm on the overshoot) and is dispersive; floored, only the negative lobes are touched and
it becomes one-signed and cumulative. That is why one detonates and the other does not, and
why the honest remedy for both is resolution.

# The floor itself, when enabled

Two rules, in order, on the TOTALS (`ρ_c = ρ_c' + ρ̄_c`, so the perturbation floor is
`−ρ̄_c`):

1. `ρ_c, ρ_r ← max(·, 0)` — the deficit is borrowed from vapor.
2. If `ρ_c + ρ_r > ρ_w` (i.e. the water budget implies `ρ_v < 0`), take the excess out of
   `ρ_c` first, then `ρ_r`.

Mass and total water are exactly conserved either way: `ρ_t` and `ρ_d` are untouched, so the
floor only ever moves the partition, and `E_t` is untouched, which is precisely why the
retrieval turns it into latent heat.

**Where the vapor's half of that phase change now arrives.** It used to be instantaneous and
implicit: the vapor was the residual of `ρ_t` and the condensates, so moving a condensate
moved it in the same instruction. With `ρ_v` prognostic this function does not touch it, and
the countermove comes through the reconciliation nudge instead — the floor opens a gap
`δ = res_rho_t − ρ_v` of exactly the moved mass, and [`rho_v_reconcile`](@ref) closes it on
`τ_rec` rather than within the step. The conservation statement is unchanged (`ρ_t` is still
the anchor and still exact); what changed is that the partition closes on a timescale instead
of pointwise, and `MC_VAPOR_GAP` reports the size of what is outstanding. That is a further
reason the default is to MEASURE: a floor whose countermove is spread over `τ_rec` is a
smaller intervention than one that was hidden inside an algebraic identity, but it is still an
intervention.

Applied to `var_np1` at the END of the column step (after the acoustic solve and the
vertical diffusion), so the values entering the patch-level fit are admissible.

# What the MEASUREMENT covers

`rho_c` and `rho_r` only, hence `min_c`/`min_r` and nothing else. The rain NUMBER and the
twelve ICE slots are not measured here: this function's statistics are in kg/m³ and are
converted to a latent-heat `worst_dT` through `L_v0/D`, which is a statement about mass. A
negative ice moment is a different quantity with a different consequence (a negative `a_{i,k}`
is a negative volume, not a negative heat), and reporting it in these rows would make
`moved`/`worst_dT` mean two things at once. The ice slots are inert while every process rate
is zero; a moment-aware negativity census belongs with the rates that can drive them there.
"""
function clamp_water!(mtile::ModelTile, colstart::Int64, colend::Int64)

    vars = mtile.model.grid_params.vars
    rhod_i = vars["rho_d"]
    rhot_i = vars["rho_t"]
    rhor_i = mc_slot(vars, "rho_r")
    rhoc_i = mc_slot(vars, "rho_c")
    vnp1 = mtile.var_np1
    rho_dbar = view(ref_rho_d(mtile.ref_state), :, 1)
    rho_tbar = view(ref_rho_t(mtile.ref_state), :, 1)
    rho_cbar = view(Springsteel.ref_rho_c(mtile.ref_state), :, 1)
    apply = get(mtile.model.options, :clamp_water, false)::Bool
    # Slots 8 and 9 are not necessarily densities. Under a transform the measurement must
    # report the RECOVERED species — that is the number that says whether the transform is
    # doing its job in the model (it should never fall below -mu) — and the floor cannot be
    # applied at all, because flooring the control variable is a different operation from
    # flooring the density and would be a state repair of exactly the kind the transform
    # exists to avoid.
    ctrans = condensate_transform_mode(mtile.model.options)
    cmu = get(mtile.model.physical_params, :condensate_mu, 1.0e-7)
    rtrans = rain_transform_mode(mtile.model.options)
    rmu = get(mtile.model.physical_params, :rain_mu, 1.0e-7)
    if apply && (ctrans !== :none || rtrans !== :none)
        error("options[:clamp_water] with a water transform on (condensate_transform = " *
              ":$(ctrans), rain_transform = :$(rtrans)): the transform already bounds the " *
              "recovered density below by -mu, and flooring the control variable instead " *
              "would be a state repair. Drop one.")
    end
    # Two-moment rain refuses the FLOOR for a different reason: raising rho_r to zero without
    # touching n_r hands the column an infinite drop count at zero mass (and lowering it would
    # be worse), so the repair would corrupt the very DSD the second moment exists to carry.
    # The MEASUREMENT below is unaffected — it reads rho_c and rho_r only — so a two-moment
    # run still gets its negative-water statistics.
    if apply && rain_moments(mtile.model.options) == 2
        error("options[:clamp_water] with options[:rain_moments] = 2: flooring rho_r " *
              "without flooring n_r alongside it leaves the column with drops that carry " *
              "no mass, which is a worse state than the negative one. A number-consistent " *
              "floor is deferred; drop one.")
    end
    # Ice refuses it for BOTH of the above reasons at once, and a third. Rule 2 below balances
    # the partition against `rho_w = rho_t - rho_d`, which with ice present is
    # `rho_v + rho_liq + rho_ice` — so a floor that does not see the ice mass would attribute
    # it to vapor and take the "excess" out of the liquid. And each ice species carries three
    # further moments that a mass repair would have to move with it, or leave the column with
    # crystals of no mass (or mass in no crystals).
    if apply && ice_microphysics(mtile.model.options) === :ishmael
        error("options[:clamp_water] with options[:ice_microphysics] = :ishmael: the floor " *
              "balances the partition against rho_t - rho_d, which now includes the ice, and " *
              "an ice mass repair would have to carry n/a/c with it. A moment-consistent " *
              "floor is deferred; drop one.")
    end

    moved = 0.0
    min_c = 0.0
    min_r = 0.0
    worst_dT = 0.0
    nneg = 0.0
    @inbounds for (k, i) in enumerate(colstart:colend)
        rho_c = recover_rho_c(vnp1[i, rhoc_i], rho_cbar[k], ctrans, cmu)
        rho_r = recover_rho_r(vnp1[i, rhor_i], rtrans, rmu)
        negative = (rho_c < 0.0) || (rho_r < 0.0)
        # The measurement fast-path: an admissible point costs two adds and a branch.
        (negative || apply) || continue

        rho_d = vnp1[i, rhod_i] + rho_dbar[k]
        rho_w = (vnp1[i, rhot_i] + rho_tbar[k]) - rho_d

        if negative
            deficit = max(-rho_c, 0.0) + max(-rho_r, 0.0)
            # Implied single-step latent-heat kick if this were floored
            # (dT/d rho_liq = L_v/D). L_v0 and D ~ C_factor are the cheap
            # conservative stand-ins; the point is the ORDER, which separates
            # 1e-3 K noise from a 4 K detonation.
            D = (rho_d * Cpd) + (rho_w * Cpv)
            moved += deficit
            nneg += 1.0
            min_c = min(min_c, rho_c)
            min_r = min(min_r, rho_r)
            worst_dT = max(worst_dT, L_v0 * deficit / D)
        end

        if apply
            # Rule 1: no negative water. Rule 2 runs REGARDLESS of rule 1 — a
            # condensate that exceeds the water present makes the density budget
            # imply a negative vapor, which is just as inadmissible and needs no
            # negative input to happen.
            rho_c = max(rho_c, 0.0)
            rho_r = max(rho_r, 0.0)
            excess = (rho_c + rho_r) - rho_w
            if excess > 0.0
                moved += excess
                take = min(excess, rho_c)
                rho_c -= take
                rho_r = max(rho_r - (excess - take), 0.0)
            end
            vnp1[i, rhoc_i] = rho_c - rho_cbar[k]
            vnp1[i, rhor_i] = rho_r
        end
    end

    if nneg > 0.0
        st = mtile.mc_water_stats
        tid = Threads.threadid()
        @inbounds begin
            st[1, tid] += moved
            st[2, tid] = min(st[2, tid], min_c)
            st[3, tid] = min(st[3, tid], min_r)
            st[4, tid] = max(st[4, tid], worst_dT)
            st[5, tid] += nneg
            # Per-STEP pre-fit minima for the production budget (reset each step by
            # `water_budget_trace`); rows 2/3 above are cumulative and cannot be differenced
            # against the post-fit reconstruction.
            st[MC_PRE_C, tid] = min(st[MC_PRE_C, tid], min_c)
            st[MC_PRE_R, tid] = min(st[MC_PRE_R, tid], min_r)
        end
    end
    return nothing
end

"""
    water_negativity_report(mtile) -> NamedTuple

Reduce `ModelTile.mc_water_stats` across threads: `(total, min_c, min_r, worst_dT,
count)`. `total` is Σ|negative water| in kg/m³ summed over gridpoints and steps; `min_c`
and `min_r` are the most negative condensate and rain densities the tile has held; and
`worst_dT` [K] is the largest `L_v·δ/D`, which is simultaneously the temperature anomaly
the negative liquid imposes through the retrieval and the kick a floor would apply — the
number that says whether the vertical resolution can represent the condensate spike (see
[`clamp_water!`](@ref)). All zero on a run that never went negative.
"""
function water_negativity_report(mtile::ModelTile)

    st = mtile.mc_water_stats
    size(st, 2) == 0 && return (total = 0.0, min_c = 0.0, min_r = 0.0,
                                worst_dT = 0.0, count = 0.0)
    return (total = sum(view(st, 1, :)),
            min_c = minimum(view(st, 2, :)),
            min_r = minimum(view(st, 3, :)),
            worst_dT = maximum(view(st, 4, :)),
            count = sum(view(st, 5, :)))
end

"""
    water_negativity_trace(mtile, t)

Warn, once per doubling, when the negative water in the condensate fields implies a
latent-heat kick large enough to matter — i.e. when the vertical basis is failing to
represent a positive-definite spike.

Emitted from the single-threaded pre-column-loop slot next to
[`state_minima_trace`](@ref), so printing is race-free. The threshold ladder starts at
`options[:water_warn_dT]` (default 0.05 K, comfortably above the ~1e-3 K fit-level noise
of a quiet column) and each subsequent warning needs a doubling, so a run that is simply
under-resolved reports O(10) lines rather than one per step.

The message names the remedy. It is NOT a floor -- flooring converts the undershoot into
one-signed latent heating (or, re-accounted, into an equal-sized energy sink); see
[`clamp_water!`](@ref) for the measured failure. It is also no longer "more vertical
nodes", which this docstring used to say: convective width collapses with the grid, so the
spike stays at grid scale and the basis rings at a fixed RELATIVE amplitude. The remedy
differs by species, and both are now measured
(`reference/FINDINGS_CONDENSATE_STAGE1.md`):

  * `rho_r` -- the spline coefficient bound (`GridParameters.positivity`), which is exact
    and free here: rain carries no negative mass at any output time of any run.
  * `rho_c` -- the control-variable transform ([`condensate_transform_mode`](@ref)). The
    same coefficient bound applied to the cloud drives the O01 peak updraft from 9.9 to
    46.5 m/s and the entropy production to 390x baseline, because it must repair the state
    at the updraft on every step. The transform changes the variable instead and repairs
    nothing.
"""
function water_negativity_trace(mtile::ModelTile, t::Int64)

    st = mtile.mc_water_stats
    size(st, 2) == 0 && return nothing
    rep = water_negativity_report(mtile)
    rep.worst_dT > 0.0 || return nothing
    warn_dT = get(mtile.model.options, :water_warn_dT, 0.05)
    @inbounds level = st[6, 1]
    threshold = level == 0.0 ? warn_dT : 2.0 * level
    rep.worst_dT >= threshold || return nothing
    @inbounds st[6, 1] = rep.worst_dT

    applied = get(mtile.model.options, :clamp_water, false)::Bool
    @warn """Negative water: the vertical basis is undershooting a condensate spike.
      step $t: min rho_c = $(rep.min_c) kg/m^3, min rho_r = $(rep.min_r) kg/m^3
      |dT| from the negative liquid (= the kick if floored): $(rep.worst_dT) K
      $(rep.count) gridpoint-steps so far, $(rep.total) kg/m^3 total
      $(applied ? "options[:clamp_water] is ON, so that kick IS being applied, one-signed." :
                  "options[:clamp_water] is off, so this is the size of the COLD anomaly the negative liquid is currently imposing through the retrieval (oscillatory, not cumulative). Rate functions are guarded; the retrieval and advection are not.")
      GENERATOR (attributed 2026-07-26): the refit deposits a small undershoot every step
      (~0.06% of peak) and NOTHING REMOVES IT. Refined 2026-07-30: what makes it permanent
      is that the negative region has no SINK — every rate is max(rho,0)-guarded, so a
      negative point cannot evaporate, autoconvert or collect, while the positive overshoot
      IS consumed every step. On the shipped O01 run the reservoir reaches 2.75x the real
      cloud mass by t = 2400 s.
      REMEDY, by species: rho_r takes the spline coefficient bound
      (GridParameters.positivity, e.g. Dict("rho_r" => Dict(:k => 0.0))), which is exact and
      free. rho_c takes options[:condensate_transform] = :bhyp — the SAME bound applied to
      the cloud takes max_w 9.9 -> 46.5 and the entropy production to 390x, because it
      repairs the state at the updraft every step. Flooring rectifies a two-signed excursion
      into one-signed latent heating. See reference/FINDINGS_CONDENSATE_STAGE1.md."""
    return nothing
end

"""
    water_budget_probe!(mtile, offset, slot, t, colstart, val, adv, div, src, aut, autsign,
                        sed, w, z)

Record the term-by-term tendency at the gridpoint where one water species is most negative.

Called from `mc_driver!` immediately after that species' `expdot` is assembled, while its
`ADV`/source scratch is still live (the buffers are reused by the next slot). Writes into the
calling thread's column of `mc_water_stats`, keeping the worst point seen so far this step —
`water_budget_trace` resets the block after printing, so the semantics are per-step, unlike
rows 1-6.

`offset` is `MC_BUDGET_R` or `MC_BUDGET_C`. `autsign` is `+1` for rain and `-1` for cloud (the
same `AUTO_COLL` array is a source for one and a sink for the other); `sed` is `nothing` for
cloud, which has no sedimentation. All arrays are column-block-local (1-based over
`colstart:colend`).

This exists because rows 1-6 record cumulative extrema and so cannot say WHICH operation
drives the water negative — the question left open by
`reference/FINDINGS_NEGATIVE_WATER_ATTRIBUTION.md`'s CORRECTION section.

The worst point is selected on the **AB3-projected** value `ρ + Δ`, where `Δ` is the exact
combination [`explicit_timestep`](@ref) will apply at this step (Euler at `t = 1`, AB2 at
`t = 2`, AB3 thereafter) — NOT on the forward-Euler projection. That distinction is the whole
reason this probe was blind: every water depletion limiter is sized so that `ρ + ts·f_n ≥ 0`
exactly, so the Euler projection of a capped point is `0.0`, `0.0 < 0.0` is false, and the
probe skipped precisely the points that produce the negative. Both projections are recorded
(`:b_*_eul` alongside the selecting value) so their gap is readable directly.

`Δ` is read from `expdot`/`expdot_nm1`/`expdot_nm2` at `slot`, which is exact at the call site
for every configuration where nothing is added to that slot afterwards. The horizontal
water mixing (`Khdiff_water`/Smagorinsky) and the Louis boundary layer both add to slot 8/9
*after* this runs, so with either enabled the projection here understates them; the
[`water_depletion_probe!`](@ref) census, which runs immediately before `explicit_timestep`,
is exact in every configuration.

**The phase change is one more such term now.** Condensation and evaporation are withheld
from `expdot` entirely (they are the `Q_ss` relaxation pair, applied as a direct increment at
the step-mean rate by [`relaxation_adjustment_qss!`](@ref)), and the step-mean is not formed
until after every slot has been assembled — so it cannot be read at this call site at all.
`Δ` here is therefore the MULTISTEP part only, and the `:b_*_src` row is the `n`-rate rather
than the step-mean. Both are the same number to `O(Δt/τ)` and the ATTRIBUTION this probe
exists for — which operation is driving the water negative — is unaffected, but the selecting
projection is no longer the exact one; [`water_depletion_probe!`](@ref) is, because it is
handed the increments explicitly.
"""
@inline function water_budget_probe!(mtile::ModelTile, offset::Int64, slot::Int64, t::Int64,
                                     colstart::Int64,
                                     val, adv, div, src, aut, autsign::Float64, sed, w, z)

    st = mtile.mc_water_stats
    size(st, 2) == 0 && return nothing
    ts = mtile.model.ts
    expdot_n = mtile.expdot_n
    expdot_nm1 = mtile.expdot_nm1
    expdot_nm2 = mtile.expdot_nm2
    j = 0
    vmin = 0.0
    dmin = 0.0
    @inbounds for i in eachindex(val)
        g = colstart + i - 1
        delta = _ab3_increment(ts, t, expdot_n[g, slot], expdot_nm1[g, slot],
                               expdot_nm2[g, slot])
        projected = val[i] + delta
        if projected < vmin
            vmin = projected
            dmin = delta
            j = i
        end
    end
    j == 0 && return nothing
    tid = Threads.threadid()
    @inbounds begin
        vmin < st[offset, tid] || return nothing
        st[offset,      tid] = vmin
        st[offset +  1, tid] = adv[j]
        st[offset +  2, tid] = -val[j] * div[j]
        st[offset +  3, tid] = src[j]
        st[offset +  4, tid] = autsign * aut[j]
        st[offset +  5, tid] = sed === nothing ? 0.0 : -sed[j]
        st[offset +  6, tid] = div[j]
        st[offset +  7, tid] = w[j]
        st[offset +  8, tid] = z[j]
        st[offset +  9, tid] = val[j] + (ts * expdot_n[colstart + j - 1, slot])
        st[offset + 10, tid] = val[j]
    end
    return nothing
end

"""
    _ab3_increment(ts, t, f_n, f_nm1, f_nm2) -> Float64

The increment [`explicit_timestep`](@ref) applies to a prognostic slot at step `t`.

One definition, so neither a diagnostic nor a limiter can drift from the integrator it is sized
for: Euler at `t = 1`, second-order Adams-Bashforth at `t = 2`, AB3 (Durran & Blossey 2012)
thereafter. The leading AB3 weight is **23/12 ≈ 1.917**, which is what a sink capped at `-ρ/ts`
— a forward-Euler budget — used to get multiplied by. [`_ab3_sink_bound`](@ref) is this same
expression solved for the current level, and is how the water limiters USED to be written.
"""
@inline _ab3_increment(ts::Float64, t::Int64, f_n::Float64, f_nm1::Float64, f_nm2::Float64) =
    t == 1 ? ts * f_n :
    t == 2 ? (0.5 * ts) * ((3.0 * f_n) - f_nm1) :
             (ts / 12.0) * ((23.0 * f_n) - (16.0 * f_nm1) + (5.0 * f_nm2))

"""
    _ab3_sink_bound(ts, t, avail, s_nm1, s_nm2) -> Float64

The most negative CURRENT-level sink [`explicit_timestep`](@ref) can apply at step `t` without
carrying a species that has `avail ≥ 0` of itself below zero, given the two previous levels of
that species' sink. **Returned UNCLAMPED** — see the clamp note below.

This is [`_ab3_increment`](@ref) solved for `s_n`, one branch per integrator branch:

| `t` | increment | admissible `s_n` |
|---|---|---|
| 1 | `ts·s_n` | `−avail/ts` |
| 2 | `(ts/2)(3 s_n − s_nm1)` | `(−2·avail/ts + s_nm1)/3` |
| ≥3 | `(ts/12)(23 s_n − 16 s_nm1 + 5 s_nm2)` | `(−12·avail/ts + 16 s_nm1 − 5 s_nm2)/23` |

**DIAGNOSTIC ONLY.** Nothing in the RHS path applies this to a rate any more: the depletion
caps were removed because a bound of the form `Q̇ ≥ −ρ/Δt` makes the physics a function of the
time step (see [`qss_condensation_rates`](@ref) and reference/Scythe_moist_compressible.tex
§"Departures from the ISHMAEL implementation" (b)). It survives because
[`_depletion_census!`](@ref) reports where the caps WOULD have bound, which is exactly the
instrument that measures what removing them cost. The description below is therefore of a
bound that is computed and compared against, never enforced.

The `t = 1` branch is exactly the forward-Euler budget `−avail/ts` that every water limiter in
this file used to be written with, at every step — and that is the defect. AB3's leading weight
is 23/12, so a sink sitting on the Euler bound is applied at ≈1.92× and lands the species at
≈ `−0.92·avail`. Measured on the quick O01 at 2.34–2.82× the local cloud, regenerated at
100–190 fresh gridpoints on EVERY step from the first cloudy one onward (STAGE 3 of
reference/FINDINGS_NEGATIVE_WATER_ATTRIBUTION.md). A constant `12/23` removes most of it
(ACTUAL depletion fraction 2.345 → 1.223, STAGE 3b) but provably not all: `12/23` is only the
right factor when the sink has no history, and the residual 0.223 is the history term — which
is what the `t ≥ 3` branch above carries and no constant can.

**Scope: the MICROPHYSICS' own contribution.** By linearity of the AB3 operator, what the
phase changes contribute to `ρ^{n+1}` is `_ab3_increment` evaluated on the sink history alone,
so bounding that bounds what the physics removes — nothing more. Transport, the spectral refit
and the acoustic solve can still carry a point negative; that is the positivity limiter's job,
and deliberately not this one's. Making the microphysics surrender its sink to compensate a
transport-generated negative would be a floor in all but name, at exactly the points STAGE 3b
showed the caps have no purchase on.

**Returned raw, clamped by the reader.** A bound `> 0` means the two history levels alone
already drive the species negative, which limiting the current level could not have repaired.
The census applies `min(bound, 0.0)` (or `max` on the mirrored vapor ceiling) before comparing
and keeps the raw value so [`_depletion_census!`](@ref) can count how often that happens — a
nonzero `:d_*_infeas` is a real signal, not noise.
"""
@inline _ab3_sink_bound(ts::Float64, t::Int64, avail::Float64,
                        s_nm1::Float64, s_nm2::Float64) =
    t == 1 ? -avail / ts :
    t == 2 ? ((-2.0 * avail / ts) + s_nm1) / 3.0 :
             ((-12.0 * avail / ts) + (16.0 * s_nm1) - (5.0 * s_nm2)) / 23.0

"""
Columns of `ModelTile.mc_micro_n`/`mc_micro_nm1`/`mc_micro_nm2` — the per-gridpoint history of
each water channel's NET MICROPHYSICS tendency.

DIAGNOSTIC ONLY, like [`_ab3_sink_bound`](@ref) itself: the history exists so the census can
evaluate the integrator's actual three-level combination on the microphysics alone
(`:d_*_mab3`, and the would-have-bound comparison). It used to size the depletion caps, which
are gone.

| column | channel | value |
|---|---|---|
| `MC_MICRO_C` | cloud | `−AUTO_COLL` (`+ ICE_C` with ice on) |
| `MC_MICRO_R` | rain | `0` (`AUTO_COLL` is a SOURCE for rain; omitting it only makes rain's budget more conservative) (`+ ICE_R` with ice on) |
| `MC_MICRO_V` | vapor | `0` |

**These are the MULTISTEP-CARRIED micro legs only.** The condensation and deposition channels
used to be here too, and are not any more: they are the relaxation pair the stiff integrator
withholds from the multistep and applies once, at the step-mean rate
([`relaxation_adjustment_qss!`](@ref)), so weighting them by a three-level combination would
have described an integration nobody performs — and it would have been the WRONG weights by up
to `23/12`. They reach the census as the step-mean rates `Qdot_bar`/`Qdot_r_bar`/`Qdep_bar`
instead, added over one step, which is exactly what the integrator does with them. The rain
and vapor rows are consequently empty on the warm path; the vapor's whole microphysical sink
is phase change, so `MC_MICRO_V` carries nothing at all.

Rotated in lockstep with `expdot_n/nm1/nm2` by [`_rotate_micro_history!`](@ref), which runs
immediately after `explicit_timestep`. Zero-initialized, so step 1 sees no history and its
budget reduces to the forward-Euler one this code used to write everywhere.
"""
const MC_MICRO_C = 1
const MC_MICRO_R = 2
const MC_MICRO_V = 3
const MC_MICRO_N = 3

"""
    _rotate_micro_history!(mtile, colstart, colend)

Advance the microphysics sink history one level for this column, mirroring
[`explicit_timestep`](@ref)'s rotation of `expdot_*` so the next step's cap sees exactly the two
levels the integrator will weight.

Called from `mc_driver!` immediately after `explicit_timestep` — which is once per column per
step on BOTH paths, because the `exact_si` branch returns from `mc_driver!` further down, after
this. A second rotation would silently shift the whole history by one step and mis-size every
cap, so the placement is load-bearing.
"""
@inline function _rotate_micro_history!(mtile::ModelTile, colstart::Int64, colend::Int64)
    n = mtile.mc_micro_n
    nm1 = mtile.mc_micro_nm1
    nm2 = mtile.mc_micro_nm2
    size(n, 2) == 0 && return nothing
    # Unconditional, unlike `explicit_timestep`'s `t == 1` branch which skips `nm2 .= nm1`:
    # both levels are zero at t = 1, so the two are identical and the branch is not worth it.
    @inbounds for c in 1:MC_MICRO_N, g in colstart:colend
        nm2[g, c] = nm1[g, c]
        nm1[g, c] = n[g, c]
    end
    return nothing
end

"""
    water_depletion_probe!(mtile, colstart, colend, t, precipitation,
                           rho_c, rho_r, rho_v, res_rho_t, Qdot, Qdot_r, AUTO_COLL,
                           cap_c, cap_r, cap_v)

Census, over every gridpoint holding water, of how hard the step is depleting it.

**PURE DIAGNOSTIC.** Nothing here is applied to any rate, and the caps it used to be built
around no longer exist (see [`qss_condensation_rates`](@ref)). Its job now is to report where
those caps WOULD have bound — which is the direct measurement of what removing them cost —
alongside the depletion fractions the integrator actually applies.

Runs immediately before [`explicit_timestep`](@ref), so `expdot` is final for the step and the
measured increment is the one actually applied. For each channel it records how many points
exist, how many have a sink at or past the AB3-exact bound, the largest forward-Euler and
largest ACTUAL depletion fractions, how many points the step drives negative outright, how many
exceed AB3's real-axis stability limit, and how many carry an inadmissible history.

What the counts now mean:

- **would-have-bound** — `:d_*_cevap` and `:d_*_cauto` are the population the removed caps used
  to act on, and `:d_*_mab3` (the microphysics' own depletion fraction) is free to run past 1
  where they did; before removal the bound held it at 1 by construction. The excursion is the
  cost, in the units the argument is made in;
- **stiff source** — `:d_*_stiff` counts points depleted faster than AB3 can stably integrate.
  Read it with [`mc_stiffness_census!`](@ref), which measures the same under-resolution at the
  RATE rather than at the slot tendency and does so on every run;
- **inadmissible history** — `:d_*_infeas` is nonzero, i.e. the previous two sink levels alone
  carry the point negative.

The VAPOR is censused alongside the two condensates, and now on the SAME footing: it is a
prognostic slot, so its tendency is read straight off `expdot` instead of being assembled from
four other slots, and its sink is the condensation (and, with ice on, the deposition) that
removes it. It has no bound of its own anywhere else in the model, which is exactly why it
needs measuring.

The two RECONCILIATION-CHAIN gaps are recorded here too: `max|res_rho_t − ρ_v|` into
[`MC_VAPOR_GAP`](@ref MC_VAPOR_GAP) and `max|Q_ss − (ρ_v − ρ_vs)|` into
[`MC_QSS_GAP`](@ref MC_QSS_GAP).

Gated on `options[:water_budget_trace]`; never called otherwise.
"""
function water_depletion_probe!(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64,
                                precipitation::Bool,
                                rho_c, rho_r, rho_v, res_rho_t, Q_ss, rho_vs,
                                Qdot, Qdot_r, AUTO_COLL, cap_c, cap_r, cap_v, rv_slot,
                                dir_c, dir_r, dir_v, vapor_rate)

    size(mtile.mc_water_stats, 2) == 0 && return nothing
    _depletion_census!(mtile, MC_DEPLETION_C, 9, MC_MICRO_C, colstart, t, rho_c, Qdot,
                       precipitation ? AUTO_COLL : nothing, Qdot, cap_c, dir_c, Qdot)
    _depletion_census!(mtile, MC_DEPLETION_R, 8, MC_MICRO_R, colstart, t, rho_r, Qdot_r,
                       nothing, Qdot, cap_r, dir_r, Qdot_r)
    _vapor_census!(mtile, colstart, t, rho_v, Qdot, Qdot_r, cap_v, rv_slot, dir_v, vapor_rate)
    _vapor_gap_census!(mtile, rho_v, res_rho_t, Q_ss, rho_vs)
    return nothing
end

"""
    _vapor_gap_census!(mtile, rho_v, res_rho_t, Q_ss, rho_vs)

Record the two RECONCILIATION GAPS over this column: `max|res_rho_t − ρ_v|` into
[`MC_VAPOR_GAP`](@ref MC_VAPOR_GAP) and `max|Q_ss − (ρ_v − ρ_vs)|` into
[`MC_QSS_GAP`](@ref MC_QSS_GAP).

These used to measure an artifact — the partition gap a retrieval OPTION opened, zero whenever
the option was off. They are load-bearing now. `ρ_t` is kept prognostic as the conservation
anchor while `ρ_v` is transported independently, so `ρ_d + ρ_v + ρ_c + ρ_r + ρ_i` does not sum
to `ρ_t` pointwise and the first gap IS that drift; [`rho_v_reconcile`](@ref) removes it on
`τ_rec`, and `τ_rec` times its rate is exactly this number. The second is the same statement one
link up the chain for `Q_ss` and [`qss_relaxation`](@ref), and it is the argument of the S3
drive clip in [`qss_condensation_rates`](@ref) — so it is what says whether that clip is doing
anything.

Both are identically `0.0` on a resting base, because the vapor slot is carried against the
DERIVED reference `ρ̄_t − ρ̄_d − ρ̄_c` that `res_rho_t` reassembles from the same fitted
columns (see [`vapor_slot`](@ref)) and the reference `Q_ssbar` is built to match. Growth in
either is a drift measurement, and neither is bounded by anything: that is the point.
"""
@inline function _vapor_gap_census!(mtile::ModelTile, rho_v, res_rho_t, Q_ss, rho_vs)

    st = mtile.mc_water_stats
    gap = 0.0
    qgap = 0.0
    @inbounds for i in eachindex(rho_v)
        d = abs(res_rho_t[i] - rho_v[i])
        d > gap && (gap = d)
        q = abs(Q_ss[i] - (rho_v[i] - rho_vs[i]))
        q > qgap && (qgap = q)
    end
    tid = Threads.threadid()
    @inbounds st[MC_VAPOR_GAP, tid] = max(st[MC_VAPOR_GAP, tid], gap)
    @inbounds st[MC_QSS_GAP, tid] = max(st[MC_QSS_GAP, tid], qgap)
    return nothing
end

"""
    _depletion_census!(mtile, species, slot, channel, colstart, t, val, evap, aut, Qdot, cap)

One species' pass for [`water_depletion_probe!`](@ref). Split out so each call specializes on
its own argument types (`rho_c` is a scratch `Vector`, `rho_r` a grid `SubArray`), rather than
being iterated as a heterogeneous tuple. `aut === nothing` skips the `AUTO_COLL` cap test,
which is the correct behaviour for rain (where `AUTO_COLL` is a source, not a sink) and with
precipitation off.

`cap` is the RAW (unclamped) [`_ab3_sink_bound`](@ref) vector for this species — a
MEASUREMENT, not a limit: `mc_driver!` fills it only when this probe is on and no rate reads
it. The tests below ask whether each rate is at or past its bound, i.e. where the removed caps
would have bound, and `:d_*_infeas` (`cap > 0`) counts the points whose sink history alone is
already inadmissible. `channel` is this species' `MC_MICRO_*` column, which gives the
microphysics-only depletion fraction.
"""
function _depletion_census!(mtile::ModelTile, species::Int64, slot::Int64, channel::Int64,
                            colstart::Int64, t::Int64, val, evap, aut, Qdot, cap,
                            dir, dir_micro)

    st = mtile.mc_water_stats
    ts = mtile.model.ts
    expdot_n = mtile.expdot_n
    expdot_nm1 = mtile.expdot_nm1
    expdot_nm2 = mtile.expdot_nm2
    micro_n = mtile.mc_micro_n
    micro_nm1 = mtile.mc_micro_nm1
    micro_nm2 = mtile.mc_micro_nm2
    # AB3's real-axis absolute-stability interval is (-0.545, 0]; a decay faster than that is
    # unstable under this integrator no matter how the sink is limited.
    AB3_REAL_LIMIT = 0.545

    n = 0.0; n_cevap = 0.0; n_cauto = 0.0; n_neg = 0.0; n_stiff = 0.0; n_infeas = 0.0
    n_mneg = 0.0
    max_eul = 0.0; max_ab3 = 0.0; max_mab3 = 0.0
    @inbounds for i in eachindex(val)
        rho = val[i]
        rho > 0.0 || continue
        n += 1.0
        g = colstart + i - 1
        f_n = expdot_n[g, slot]
        # THE EFFECTIVE INCREMENT. `expdot` no longer carries the phase change — the stiff
        # relaxation pair is withheld from the multistep and applied as the direct increment
        # `dir` at the step-mean rate — so the increment this step actually applies is the
        # three-level combination PLUS that increment, and the forward-Euler comparison
        # correspondingly reads `f_n + dir/ts`. Adding it here is what keeps the census meaning
        # exactly what it meant: the depletion the integrator applies, measured against the
        # water present.
        delta = _ab3_increment(ts, t, f_n, expdot_nm1[g, slot], expdot_nm2[g, slot]) + dir[i]
        dep_eul = -((ts * f_n) + dir[i]) / rho
        dep_ab3 = -delta / rho
        dep_eul > max_eul && (max_eul = dep_eul)
        dep_ab3 > max_ab3 && (max_ab3 = dep_ab3)
        dep_ab3 > 1.0 && (n_neg += 1.0)
        # The MICROPHYSICS' own share of the same increment — the part `_ab3_sink_bound`
        # governs, and the only fraction the depletion mode can be judged by. Same split: the
        # `MC_MICRO_*` history now carries only the legs still on the multistep (see the block
        # that writes it), and the phase change enters at its step-mean rate over one step.
        dep_mab3 = -(_ab3_increment(ts, t, micro_n[g, channel], micro_nm1[g, channel],
                                    micro_nm2[g, channel]) + (ts * dir_micro[i])) / rho
        dep_mab3 > max_mab3 && (max_mab3 = dep_mab3)
        # Tolerance, not `> 1.0`: a point sitting exactly ON the bound lands at 1 + O(eps)
        # (the bound divides by 23, so it cannot be exact), and counting those as violations
        # would report every capped point. 1e-9 is ~1e7 ULP above that and ~1e9 below the
        # 23/12 the `:euler` mode produces, so it separates the two without ambiguity.
        dep_mab3 > MICRO_DEPLETION_TOL && (n_mneg += 1.0)
        # WOULD-HAVE-BOUND detection. The caps are gone from the RHS (see
        # `qss_condensation_rates`), so this no longer tests "did the limiter clip here"
        # (bitwise `==` against the array the limiter read) but "would it have": the realized
        # rate is at or past the AB3-exact bound. That is the measurement of what removing the
        # caps costs, and the counts are directly comparable with the pre-removal ones, which
        # were the same population by construction.
        raw = cap[i]
        raw > 0.0 && (n_infeas += 1.0)
        floor_i = min(raw, 0.0)
        capped = evap[i] <= floor_i
        capped && (n_cevap += 1.0)
        if aut !== nothing
            # `aut > 0` excludes the degenerate no-conversion point: with the budget exhausted
            # (`avail == 0`) every nonzero conversion would have been clipped, but a zero one
            # would not have been touched at all.
            avail = max(Qdot[i] - floor_i, 0.0)
            if aut[i] > 0.0 && aut[i] >= avail
                n_cauto += 1.0
                capped = true
            end
        end
        (!capped && dep_eul > AB3_REAL_LIMIT) && (n_stiff += 1.0)
    end

    _census_reduce!(st, species, n, n_cevap, n_cauto, max_eul, max_ab3, n_neg, n_stiff,
                    n_infeas, max_mab3, n_mneg)
    return nothing
end

"""Common per-thread accumulation for the depletion census blocks (same layout for all three)."""
@inline function _census_reduce!(st, species::Int64, n, n_cevap, n_cauto, max_eul, max_ab3,
                                 n_neg, n_stiff, n_infeas, max_mab3, n_mneg)
    tid = Threads.threadid()
    @inbounds begin
        st[species,     tid] += n
        st[species + 1, tid] += n_cevap
        st[species + 2, tid] += n_cauto
        st[species + 3, tid] = max(st[species + 3, tid], max_eul)
        st[species + 4, tid] = max(st[species + 4, tid], max_ab3)
        st[species + 5, tid] += n_neg
        st[species + 6, tid] += n_stiff
        st[species + 7, tid] += n_infeas
        st[species + 8, tid] = max(st[species + 8, tid], max_mab3)
        st[species + 9, tid] += n_mneg
    end
    return nothing
end

"""
    _vapor_census!(mtile, colstart, t, rho_v, Qdot, Qdot_r, cap, slot)

The vapor's pass for [`water_depletion_probe!`](@ref), in the same layout as
[`_depletion_census!`](@ref)'s two condensate blocks.

The vapor is a PROGNOSTIC SLOT, so its three time levels are read straight out of `expdot`
like every other species'. That replaced an assembly from four other slots,
`f_v = f_3 - f_2 - f_8 - f_9`, which was exact for the liquid-only set (all four use the same
advective product-rule form with the same pointwise divergence, both `-v·∇ρ` and `-ρ∇·v` are
linear in ρ, the sedimentation divergence cancels between slots 3 and 8 and `AUTO_COLL`
between 8 and 9) but had NO ice term — so with ice registered it silently attributed the
deposition sink to nothing at all. Reading the slot fixes that at the same time as it
simplifies: there is one tendency, and it is the one the integrator applied.

What this census will NOT see is anything applied outside `expdot`: the acoustic solve moves
`rho_t` and `rho_d` but no water slot, and the positivity limiter moves the condensates but
not `rho_t`. Both now show up in the RECONCILIATION GAP instead (see
[`_vapor_gap_census!`](@ref)), which is where a partition drift belongs.
"""
function _vapor_census!(mtile::ModelTile, colstart::Int64, t::Int64, rho_v, Qdot, Qdot_r, cap,
                        slot::Int64, dir, dir_micro)

    st = mtile.mc_water_stats
    ts = mtile.model.ts
    e_n = mtile.expdot_n
    e_1 = mtile.expdot_nm1
    e_2 = mtile.expdot_nm2
    AB3_REAL_LIMIT = 0.545

    micro_n = mtile.mc_micro_n
    micro_nm1 = mtile.mc_micro_nm1
    micro_nm2 = mtile.mc_micro_nm2

    n = 0.0; n_cap = 0.0; n_neg = 0.0; n_stiff = 0.0; n_infeas = 0.0; n_mneg = 0.0
    max_eul = 0.0; max_ab3 = 0.0; max_mab3 = 0.0
    @inbounds for i in eachindex(rho_v)
        rho = rho_v[i]
        rho > 0.0 || continue
        n += 1.0
        g = colstart + i - 1
        f_n = e_n[g, slot]
        f_1 = e_1[g, slot]
        f_2 = e_2[g, slot]
        # The effective increment, phase change included — see the note in
        # `_depletion_census!`. For the vapor the WHOLE microphysical sink is now direct: every
        # term of `VAPOR_SRC` is a relaxation of `Q_ss`, so the `MC_MICRO_V` history is empty
        # and `dir_micro` carries all of it.
        delta = _ab3_increment(ts, t, f_n, f_1, f_2) + dir[i]
        dep_eul = -((ts * f_n) + dir[i]) / rho
        dep_ab3 = -delta / rho
        dep_eul > max_eul && (max_eul = dep_eul)
        dep_ab3 > max_ab3 && (max_ab3 = dep_ab3)
        dep_ab3 > 1.0 && (n_neg += 1.0)
        dep_mab3 = -(_ab3_increment(ts, t, micro_n[g, MC_MICRO_V], micro_nm1[g, MC_MICRO_V],
                                    micro_nm2[g, MC_MICRO_V]) + (ts * dir_micro[i])) / rho
        dep_mab3 > max_mab3 && (max_mab3 = dep_mab3)
        # Tolerance, not `> 1.0`: a point sitting exactly ON the AB3 bound lands at
        # 1 + O(eps) (the bound divides by 23, so it cannot be exact). See
        # `MICRO_DEPLETION_TOL`.
        dep_mab3 > MICRO_DEPLETION_TOL && (n_mneg += 1.0)
        raw = cap[i]
        raw > 0.0 && (n_infeas += 1.0)
        # The vapor's ceiling is the negated bound; this counts where the REMOVED ceiling
        # would have clipped the combined condensation.
        ceil_i = -min(raw, 0.0)
        capped = ceil_i > 0.0 && (Qdot[i] + Qdot_r[i]) >= ceil_i
        capped && (n_cap += 1.0)
        (!capped && dep_eul > AB3_REAL_LIMIT) && (n_stiff += 1.0)
    end

    _census_reduce!(st, MC_DEPLETION_V, n, n_cap, 0.0, max_eul, max_ab3, n_neg, n_stiff,
                    n_infeas, max_mab3, n_mneg)
    return nothing
end

"""
    mc_stiffness_census!(mtile, ts, invtau_c, invtau_r)

Accumulate the per-channel STIFFNESS of the supersaturation relaxation over this column:
the running maximum of `ts·(1/τ)` and the number of gridpoints where it exceeds 1.

This is what stands in place of the depletion caps. Those made the rate — hence the converged
solution — a function of `ts`; see [`qss_condensation_rates`](@ref) and
reference/Scythe_moist_compressible.tex §"Departures from the ISHMAEL implementation" (b).
Under-resolution of a stiff relaxation is now REPORTED here instead of being absorbed there.

`ts/τ > 1` is the honest threshold: the relaxation removes more than the whole supersaturation
over a step, so the fast dynamics is not resolved in time.

**What that reading MEANS depends on which channel it is on**, and the answer changed when the
`Q_ss` pair moved off the multistep (TeX §Departures (b), amended). For the cloud, rain and ice
DEPOSITION channels — the ones that relax the shared `Q_ss` — the propagator is exact at any
`ts/τ` and lands on the quasi-steady state of Eq. wbf_qs as `ts/τ → ∞`, so this is no longer a
stability watchdog for them: a large reading marks air in which the quasi-steady limit is being
taken inside a single step, correctly but unresolved in time. For the NUCLEATION relaxations
(`τ_hf`, `τ_act`, `τ_vc`), which remain on the multistep, it keeps its original meaning as a
stability warning. (The integrator's own real-axis limit for the multistep channels is tighter
still — AB3 loses absolute stability at 0.545 — and `:d_*_stiff` in the depletion census counts
against that one. Both are measured; neither is enforced.)

Runs on EVERY step of EVERY run, not behind a trace flag: the once-per-run warning in
[`mc_stiffness_trace`](@ref) is unconditional and has to have something to read. The cost is
two comparisons per gridpoint per channel and no allocation — the accumulators are rows of
the preallocated `mtile.mc_water_stats`, indexed by `threadid()` under the same
`@threads :static` ownership rule as every other probe in this file.

Sized by `MC_STIFF_CHANNELS`, so the ice channels are added by widening that constant rather
than by editing this function. NEVER touches a rate.
"""
@inline function mc_stiffness_census!(mtile::ModelTile, ts::Float64, invtau_c, invtau_r)

    st = mtile.mc_water_stats
    size(st, 2) == 0 && return nothing
    tid = Threads.threadid()
    _stiffness_channel!(st, tid, MC_STIFF_C, ts, invtau_c)
    _stiffness_channel!(st, tid, MC_STIFF_R, ts, invtau_r)
    return nothing
end

"""
    mc_stiffness_census!(mtile, ts, invtau_i1, invtau_i2, invtau_i3)

The ICE arm of [`mc_stiffness_census!`](@ref): the three deposition relaxations `1/τ_{i,k}` of
TeX Eq. tau_ice, measured on exactly the rates the step used, with the same `ts/τ` yardstick
and into the same block (`MC_STIFF_I1..I3`).

A SEPARATE method rather than three more arguments to the liquid one, so an ice-free run does
not pay three passes over three columns it would only find zeros in — and, more importantly,
so those three channels report an exact zero rather than a zero that had to be computed.
"""
@inline function mc_stiffness_census!(mtile::ModelTile, ts::Float64,
                                      invtau_i1, invtau_i2, invtau_i3)

    st = mtile.mc_water_stats
    size(st, 2) == 0 && return nothing
    tid = Threads.threadid()
    _stiffness_channel!(st, tid, MC_STIFF_I1, ts, invtau_i1)
    _stiffness_channel!(st, tid, MC_STIFF_I2, ts, invtau_i2)
    _stiffness_channel!(st, tid, MC_STIFF_I3, ts, invtau_i3)
    return nothing
end

"""
One channel's pass for [`mc_stiffness_census!`](@ref): reduce the column into thread-local
scalars first, then touch the shared matrix twice. Split out so each call specializes on its
own array type, as [`_depletion_census!`](@ref) is.
"""
@inline function _stiffness_channel!(st, tid::Int64, channel::Int64, ts::Float64, invtau)

    mx = 0.0
    n = 0.0
    @inbounds for i in eachindex(invtau)
        x = ts * invtau[i]
        x > mx && (mx = x)
        x > 1.0 && (n += 1.0)
    end
    row = MC_STIFF_FIRST + (MC_STIFF_N * (channel - 1))
    @inbounds begin
        st[row, tid] = max(st[row, tid], mx)
        st[row + 1, tid] += n
    end
    return nothing
end

"""
    mc_donor_census!(st, tid, donor, realized, reservoir, floor)

One donor's pass for the depletion census: record the fraction of the reservoir this step
actually removes, and count the points where it exceeds one. Reads the rates and the state;
changes neither. See [`MC_DONOR_QC`](@ref) for what the block means and why it is separate from
the stiffness census.
"""
@inline function mc_donor_census!(st, tid::Int64, donor::Int64, realized::Float64,
                                  reservoir::Float64, floor_::Float64, dt::Float64)
    reservoir > floor_ || return nothing
    x = realized * dt / reservoir
    row = MC_DONOR_FIRST + (MC_DONOR_N * (donor - 1))
    @inbounds begin
        x > st[row, tid] && (st[row, tid] = x)
        x > 1.0 && (st[row + 1, tid] += 1.0)
    end
    return nothing
end

"""
    mc_attr_census!(st, tid, channel, x, count_it)

One channel's write for the ATTRIBUTION census: raise the channel's run-max to `x`, and add a
gridpoint-step to its count when `count_it`. See [`MC_ATTR_QR_HOM`](@ref) for what the block
means and why the count is a separate argument rather than `x > 1`.

The two are decoupled BECAUSE the block A channels are counted by ARGMAX, not by threshold:
the count records which leg carried the largest debit at a breach point, so the five counts
partition the `MC_DONOR_QR` count, while every channel still reports the run-max of its own
share. `MC_ATTR_QR_MELT` (a credit) passes `false` always, and `MC_ATTR_AGG1_SAT` passes the
cap-binding test rather than a depletion test.

Allocation-free and loop-free: it runs once per channel per gridpoint inside
`mc_ice_sources!`'s hot loop, which `test_allocations.jl` holds at zero.
"""
@inline function mc_attr_census!(st, tid::Int64, channel::Int64, x::Float64,
                                 count_it::Bool)
    row = MC_ATTR_FIRST + (MC_ATTR_N * (channel - 1))
    @inbounds begin
        x > st[row, tid] && (st[row, tid] = x)
        count_it && (st[row + 1, tid] += 1.0)
    end
    return nothing
end

"""
    ice_anchor_rate(rho_t, rho_d, rho_liq, rho_ice, tau_anchor) -> (phi, delta)

The rate law of the ANCHOR RECONCILIATION — the third tier of the reconciliation chain
(TeX §"Reconciliation of the condensate partition", Eqs. partition_gap / partition_rec_eq).

The twelve ice moments advect on their own fitted fields while the total water advects as
one, so the summed ice mass and the anchor's headroom are two independent discretizations
of the same physical water; at a sharp glaciation front they detach, the vapor nudge
faithfully closes the budget onto the unphysical partition (`MC_VAPOR_GAP` reads zero), and
the retrieval takes an `L_s` credit the heat capacity does not contain — the §2c death
chain. The defect is

    delta = max(0, rho_ice - max((rho_t - rho_d) - max(rho_liq, 0), 0))

— the ice mass in excess of the headroom left after the POSITIVE liquid (a ringing negative
liquid must not enlarge the ice's debit) — and the returned rate is one shared fraction per
second, `phi = delta / (rho_ice · tau_anchor)`, applied by [`_ice_anchor_reconcile!`](@ref)
to every moment of every species so the per-particle state and the species shares are
exactly invariant.

Properties (each is a test in test_ice_anchor_reconcile.jl):
- ONE-SIDED and exactly `(0.0, 0.0)` on every admissible state — measured identically zero
  from initialization through glaciation onset on the O01 ice arm, so the device is absent
  from healthy air, bitwise;
- `phi ≤ 1/tau_anchor` identically (`delta ≤ rho_ice` by construction), so the per-step
  removed fraction is bounded by `Δt/τ_anchor ≪ 1` with no realization factor — and
  `relaxation_realization` is the ready escalation if `τ_anchor` is ever pushed toward the
  step;
- Δt-free: the physics supplies a rate, the integrator supplies the step.

`tau_anchor` is `physical_params[:tau_ice_anchor]` (default 10.0 s, the `tau_rec` class):
at the measured transport-phase feed (≤ 1.7e-6 kg/m³/s, FINDINGS §Stage C) the detachment
equilibrates at `F·τ ≈ 1.7e-5` kg/m³ — a 0.16 K retrieval excursion against the ~1e4
K/(kg/m³) sensitivity the ts=0.3 death window calibrates.
"""
@inline function ice_anchor_rate(rho_t::Float64, rho_d::Float64, rho_liq::Float64,
                                 rho_ice::Float64, tau_anchor::Float64)
    head = max((rho_t - rho_d) - max(rho_liq, 0.0), 0.0)
    delta = max(rho_ice - head, 0.0)
    phi = (delta > 0.0 && rho_ice > 0.0) ? delta / (rho_ice * tau_anchor) : 0.0
    return phi, delta
end

"""
    _ice_anchor_reconcile!(S, st, tid, rho_t, rho_d, rho_liq, rho_ice, tau_anchor, apply, ts)

One column's pass of the anchor reconciliation: measure the partition defect at every
gridpoint into the `MC_ANCHOR_*` census, and — when `apply` — subtract
`phi · max(moment, 0)` from all twelve ice `SRC_*` accumulators.

Runs AFTER [`mc_ice_sources!`](@ref) and before the sedimentation fluxes, so it composes
with the process sources on the multistep exactly as the `var_check` consistency source
does (`τ_vc = 5 s` there, `τ_anchor = 10 s` here — both continuous relaxations, no
burst-spike hazard of the §2a′ class). Three deliberate non-couplings, per the TeX's third
property: no term in `ρ_t`, `E_t`, or the retrieval (the removed mass was never in the
anchor, so no latent credit accompanies it — the vapor follows through
[`rho_v_reconcile`](@ref) as `res_rho_t` rises); no term in `invtau_i*`/`Qdot_i*` (it is a
bookkeeping projection between two representations of one water, not a vapor exchange, so
the `Q_ss` propagator's λ and N never see it); and no test of the population gate — the
device exists precisely for the gate-orphaned points, where an `n = 0` mass drains while
the carried zero number stays an exact zero.

`max(moment, 0.0)` in every increment is the only asymmetry: the device may ONLY remove,
so a moment ringing at its `-μ` transform floor contributes no (positive) source. Where the
moments are positive the increments share one factor and the per-particle axes, effective
density, aspect ratio and species shares are exactly invariant — the moment consistency the
deferred `clamp_water!` ice floor owed and could not provide.

`apply = false` (`options[:ice_anchor_source] = false`) keeps the census and drops the
removal: the defect reproduction stays bitwise available for forensics, measured.
"""
function _ice_anchor_reconcile!(S, st, tid::Int64, rho_t, rho_d, rho_liq, rho_ice,
                                tau_anchor::Float64, apply::Bool, ts::Float64)
    gap = 0.0
    pts = 0.0
    rem = 0.0
    @inbounds for i in eachindex(rho_ice)
        phi, delta = ice_anchor_rate(rho_t[i], rho_d[i], rho_liq[i], rho_ice[i],
                                     tau_anchor)
        delta > 0.0 || continue
        delta > gap && (gap = delta)
        pts += 1.0
        (apply && phi > 0.0) || continue
        q1 = max(S.i1q[i], 0.0); q2 = max(S.i2q[i], 0.0); q3 = max(S.i3q[i], 0.0)
        rem += phi * ((q1 + q2) + q3) * ts
        S.SRC_i1q[i] -= phi * q1
        S.SRC_i2q[i] -= phi * q2
        S.SRC_i3q[i] -= phi * q3
        S.SRC_i1n[i] -= phi * max(S.i1n[i], 0.0)
        S.SRC_i2n[i] -= phi * max(S.i2n[i], 0.0)
        S.SRC_i3n[i] -= phi * max(S.i3n[i], 0.0)
        S.SRC_i1a[i] -= phi * max(S.i1a[i], 0.0)
        S.SRC_i2a[i] -= phi * max(S.i2a[i], 0.0)
        S.SRC_i3a[i] -= phi * max(S.i3a[i], 0.0)
        S.SRC_i1c[i] -= phi * max(S.i1c[i], 0.0)
        S.SRC_i2c[i] -= phi * max(S.i2c[i], 0.0)
        S.SRC_i3c[i] -= phi * max(S.i3c[i], 0.0)
    end
    if size(st, 2) > 0
        @inbounds begin
            gap > st[MC_ANCHOR_GAP, tid] && (st[MC_ANCHOR_GAP, tid] = gap)
            st[MC_ANCHOR_PTS, tid] += pts
            st[MC_ANCHOR_REMOVED, tid] += rem
        end
    end
    return nothing
end

"""
    ice_population_rate(rho_ik, n_ik, rho_a, tau_pop) -> (rate, rho_empty)

The NUMBER-LESS MASS of one ice species at one gridpoint and the rate at which it is returned
to a representation the physics can act on — the fourth and last tier of the reconciliation
chain (TeX §"Reconciliation of the population", Eq. class `ρ^∅_{i,k}/τ_pop`).

The defect this measures is the exact complement of the POPULATION GATE in
[`mc_ice_sources!`](@ref). That gate says a species has a population only where its CARRIED
number is positive, and where it does not, no rate acts: not deposition, not riming, not
melting, and no fall speed, because a fall speed is a property of particles. The gate is
right — it is what stopped the manufactured populations of the transport-decorrelated state
from growing, riming and falling at speeds the mass never earned — and it has one consequence
the same measurement records: mass with no number is also exempt from every device that could
REMOVE it. Measured on the full-resolution O01 ice column, the ice below the melting level is
number-less at 97–99.9% by mass over the final half hour, the largest concentrations sitting
at the surface at 300 K; the timestep-limited quick configuration reproduces it at 99%.

The anchor share of that mass is exactly one throughout, so this is NOT the phantom of
[`ice_anchor_rate`](@ref): it is water the equations own, in a phase representation the
equations cannot act on. Hence a separate tier, and hence the RAW carried moments here —
`f_a` is the rate-side share for PROCESS rates, and the TeX's division of labour leaves the
transport and the reconciliation sources alone reading the raw slots.

    ρ^∅_{i,k} = max(ρ_{i,k}, 0)   where   ρ_{i,k} > QSMALL·ρ_a  and  n_{i,k} ≤ 0
              = 0                 otherwise

The mass test is the mixing-ratio one the gate itself uses (`q_k > ISHMAEL_QSMALL`), written
in densities so no division is needed; the TeX's `ρ_{i,k} > 0` is sharpened to it for exactly
that reason, and the difference is 1e-12 kg/kg of round-off either way.

Both branches of the transfer are `Δt`-free (a rate, not a state repair; the removed or
seeded fraction per step is bounded by `Δt/τ_pop ≪ 1`), one-sided, and exactly `(0.0, 0.0)`
wherever every species with mass carries number — which is every gridpoint of the warm and
dry paths and the interior of every healthy ice cloud, so those paths are untouched bitwise.
"""
@inline function ice_population_rate(rho_ik::Float64, n_ik::Float64, rho_a::Float64,
                                     tau_pop::Float64)
    (rho_ik > ISHMAEL_QSMALL * rho_a && n_ik <= 0.0) || return (0.0, 0.0)
    rho_empty = max(rho_ik, 0.0)
    return (rho_empty / tau_pop, rho_empty)
end

"""
    _ice_population_seed_large(q, k) -> (n, a, c)

The LARGEST-CRYSTAL, FEWEST-PARTICLE population the scheme admits for a species-`k` mass
mixing ratio `q` [kg/kg]: the number and the two volume moments (`# kg⁻¹`, `m³ kg⁻¹`,
`m³ kg⁻¹`) [`ishmael_var_check`](@ref) re-diagnoses when the carried number is at its floor
and the mass is all that is known.

This is the second of the two seedings [`_ice_population_reconcile!`](@ref) can give
number-less ice below `T_0` (`options[:ice_population_seed] = :large`), and it is ISHMAEL's
OWN answer to the question: the port never carries a `(q, n)` pair the moment checker would
reject, so a species that arrives with mass and no number is one `var_check` call away from a
realizable population, and this is that call. Nothing is re-derived here — the body is
[`_ice_effective`](@ref) at the floors, which is the Fortran host loop's incoming-moment
initialization (module_mp_jensen_ishmael.F lines 1013-1032) followed by `var_check` (lines
3101-3196) — so the seeded `(n, a, c)` is EXACTLY the state the next step diagnoses, and the
population is a FIXED POINT of the checker rather than something it has to repair.

# The rule the floors produce, and why it is the large end

At `n = QNSMALL`, `a = c = QASMALL` the incoming axes are the 2 μm sphere, so `δ* = 1` exactly
and the species is spherical. `var_check` then walks its clamps in order:

  * the bulk density `ρ̄ = q Γ(ν) / (n α_v a_n^{2+δ*} Γ(ν+2+δ*))` is enormous at that number
    and is CLAMPED to the species ceiling (lines 3131-3154) — `RHOI = 920 kg/m³` for the
    planar and columnar species, [`ISHMAEL_RHO_AGG`](@ref) `= 50 kg/m³` for the aggregates.
    That ceiling is the ONLY place the three species differ, and it is the whole of the
    "different density/axis relations" the aggregate block asserts by hand;
  * the characteristic axis is re-derived from the mass at that density and, for any mass
    past `q ≈ 5.8e-11` kg/kg (planar/columnar; `≈ 3.1e-12` for the lighter aggregates),
    exceeds the 1 mm cap, so the LARGE-ICE LIMIT fires (lines 3171-3193) and returns

        a_n = c_n = 1 mm,   n = q Γ(ν) / ((4/3)π ρ̄ a_n³ Γ(ν+3)),   a_i = c_i = n a_n³

    i.e. one crystal per `m_large = (4/3)π ρ̄ (1 mm)³ Γ(ν+3)/Γ(ν)` — 4.62e-4 kg for the
    planar and columnar species, 2.51e-5 kg for the aggregates, against
    [`ISHMAEL_M_MIN`](@ref) `= 3.08e-14 kg` for the 2 μm seeding: a factor of 1.5e10 and
    8.2e8 fewer particles for the same mass;
  * below that mass the large-ice cap does not bind and the `QNSMALL` floor is itself the
    answer — the same statement, the FEWEST particles the scheme admits for the mass, written
    by whichever clamp is the binding one.

The aggregate's OTHER axis rules — the 0.2 aspect ratio and the 0.5 mm implicit-breakup cap
of the "Final check on aggregates" block (lines 2600-2639) — are deliberately NOT applied.
They are that block's update of an EXISTING characteristic axis, they need an incoming `a_n`
this function does not have, and the Fortran closes the block with `var_check` (line 2644),
which is the authority the seed has to satisfy and is what is called here.

`(0.0, 0.0, 0.0)` for a mass at or below `QSMALL`: `var_check` divides by the mass and cannot
be run on a species that has none.
"""
@inline function _ice_population_seed_large(q::Float64, k::Int)

    q > ISHMAEL_QSMALL || return (0.0, 0.0, 0.0)
    eff = _ice_effective(q, 0.0, 0.0, 0.0, k,
                         k == 3 ? ISHMAEL_RHO_AGG : ISHMAEL_RHOI)
    return (eff.ni, eff.ai, eff.ci)
end

"""
    _ice_live_span(Qk, Nk, rho_d, idx) -> (first, last)

The first and last index of `idx` at which species `k` is LIVE — mass past the population
gate's own threshold and a positive carried number, on the RAW slots `Qk`/`Nk`, which is
exactly the complement of [`ice_population_rate`](@ref)'s support test.

One O(n) sweep per species per column, computed once by [`_ice_population_reconcile!`](@ref)
before its gridpoint loop, and it is what makes the `:local` seeding's nearest-live search
cheap. `(0, 0)` when the species is live NOWHERE in the column — the pure-dead column, which
the search then never enters.
"""
@inline function _ice_live_span(Qk, Nk, rho_d, idx)

    first = 0
    last = 0
    @inbounds for j in idx
        (Qk[j] > ISHMAEL_QSMALL * rho_d[j] && Nk[j] > 0.0) || continue
        first == 0 && (first = j)
        last = j
    end
    return (first, last)
end

"""
    _ice_live_neighbour(Qk, Nk, rho_d, i, first, last) -> j

The index of the NEAREST gridpoint to `i` IN THE SAME COLUMN at which species `k` is live,
or `0` when the species is live nowhere in it. `first`/`last` are that species'
[`_ice_live_span`](@ref).

The search walks outward from `i` in both directions along the column index — which is the
VERTICAL index, because the physics is called one column at a time (`advance_column` strides
`colstart:colend` by `kDim`, and every `mc_scratch` slot is one `kDim` column of that call).
Ties go to the LOWER index, arbitrarily but deterministically: the two neighbours of a
one-point negative lobe are the same population on either side of it, so there is nothing to
choose between them.

Cost. The span bounds it on three sides: a point below `first` or above `last` is answered
without a walk at all, and between them the walk terminates at the first live point, so the
work is the local gap between live points and not the column length. A column in which the
species has NO live point costs the span sweep alone. The pathological O(n) walk needs a
dead point in the interior of a gap the length of the column, which is a live population at
each end and nothing in between.
"""
@inline function _ice_live_neighbour(Qk, Nk, rho_d, i::Int, first::Int, last::Int)

    first == 0 && return 0
    i <= first && return first
    i >= last && return last
    @inbounds for d in 1:max(i - first, last - i)
        j = i - d
        if j >= first && Qk[j] > ISHMAEL_QSMALL * rho_d[j] && Nk[j] > 0.0
            return j
        end
        j = i + d
        if j <= last && Qk[j] > ISHMAEL_QSMALL * rho_d[j] && Nk[j] > 0.0
            return j
        end
    end
    return 0                      # unreachable: `first < i < last` guarantees a hit
end

"""
    _ice_population_seed_local(q, m_nb, a_nb, c_nb, k) -> (n, a, c)

The population a species-`k` mass mixing ratio `q` [kg/kg] has when its crystals are THE
CRYSTALS OF ITS OWN NEIGHBOURS: `m_nb`, `a_nb`, `c_nb` are the per-crystal mass [kg] and the
two per-crystal volume moments [m³] of the same species at the nearest gridpoint in the
column where it is live (`ρ_{i,k}/n_{i,k}`, `a_{i,k}/n_{i,k}`, `c_{i,k}/n_{i,k}` on the raw
slots), and the seeded population carries the same mass, habit and bulk density per particle.

This is the third of the three seedings [`_ice_population_reconcile!`](@ref) can give
number-less ice below `T_0` (`options[:ice_population_seed] = :local`), and the argument for
it is a measurement of what the dead mass IS. Number-less mass is the NEGATIVE LOBE of the
number moment's spline ringing at cloud edges and gradients — the mass-without-number face of
the transport-decorrelation family whose other face is the 3e13 /L number spikes sitting
right beside it. The crystals that lobe lost are the crystals next door, so the per-crystal
state next door is the seed, and neither of the two ends the other seedings pick is: `:min`
puts ~1e12 2 μm crystals/m³ on it and thins the cloud threefold, `:large` puts one 1 mm
crystal per 4.6e-4 kg on it, too sparse to survive the number field's own ringing.

# The clamps, and why the seed is realizable

`m_nb` is read off RAW transported slots, so it is not itself guaranteed admissible — the
same ringing that emptied this point can have corrupted the ratio at that one. It is clamped
to `[ISHMAEL_M_MIN, ISHMAEL_M_LARGE[k]]`: never below the 2 μm crystal, never above the
1 mm sphere `:large` would seed. The volume moments are scaled by the SAME factor, which is
the isotropic rescale — per-crystal volume is linear in per-crystal mass at fixed bulk
density — so a bound clamp changes the SIZE of the inherited crystal and neither its bulk
density nor its aspect ratio.

`ISHMAEL_M_MIN` is the mass of ONE 2 μm sphere, and it is the LOOSER of the two limits at
that end: `var_check`'s own small-ice floor is on the distribution's mean radius, so the
smallest per-crystal mass it admits at `RHOI` is `M_MIN·Γ(ν+3)/Γ(ν)`, 120 times larger. A
`:local` seed at the lower clamp therefore comes out at the checker's floor rather than at
`:min`'s number — a realizable population where `:min` seeds `ρ^∅/m_min` and leaves the next
read to repair it. That is the clamps' division of labour: they bound the RAW ratio into
something sane, and the checker has the last word on what is representable.

The triple is then put through [`_ice_effective`](@ref), exactly as
[`_ice_population_seed_large`](@ref) is, so what is seeded is what `var_check` will diagnose
from it and the population is a FIXED POINT of the checker rather than something it has to
repair. The default `rhomax` is used for ALL three species — unlike `:large`, which hands the
aggregate its own 50 kg/m³ ceiling — because the neighbour's own state came through the
default-ceiling path in [`mc_ice_sources!`](@ref), and re-clamping it here would change the
habit this seeding exists to inherit. Where the mass clamp does not bind and the neighbour is
itself a checked population, `var_check` is the identity on the seed to a few parts in 1e11
(the port's truncated `^0.333333333333` cube roots), so the seeded per-crystal state IS the
neighbour's.

`(0.0, 0.0, 0.0)` for a mass at or below `QSMALL`, on `_ice_population_seed_large`'s rule and
for its reason.
"""
@inline function _ice_population_seed_local(q::Float64, m_nb::Float64, a_nb::Float64,
                                            c_nb::Float64, k::Int)

    (q > ISHMAEL_QSMALL && m_nb > 0.0) || return (0.0, 0.0, 0.0)
    m_loc = clamp(m_nb, ISHMAEL_M_MIN, @inbounds ISHMAEL_M_LARGE[k])
    # The isotropic rescale under a bound clamp: per-crystal volume is linear in per-crystal
    # mass at fixed bulk density, so the two volume moments carry the same factor the mass
    # does and the crystal changes size without changing density or aspect ratio.
    vscale = m_loc / m_nb
    n = q / m_loc
    eff = _ice_effective(q, n, (n * vscale) * a_nb, (n * vscale) * c_nb, k)
    return (eff.ni, eff.ai, eff.ci)
end

"""
    _ice_population_reconcile!(S, st, tid, Tk, rho_d, tau_pop, apply, ts, seed = :min)

One column's pass of the POPULATION reconciliation: measure the number-less ice mass of every
species at every gridpoint into the `MC_POP_*` census, and — when `apply` — return it to the
representation the environment dictates.

Runs immediately after [`_ice_anchor_reconcile!`](@ref), in the same slot and for the same
reason: the process sources are assembled, nothing has read them yet, and the twelve `SRC_i*`
accumulators plus the three liquid back-reactions are all that either device writes. A
separate pass rather than a block inside `mc_ice_sources!`'s gridpoint loop, on three grounds:
it reads the RAW slots (`mc_ice_sources!`'s per-species locals are the `f_a`-shared ones, and
the TeX's division of labour puts the reconciliation sources on the raw side with the
transport); it is a reconciliation and not a rate, so its off-switch, its census and its
`τ` belong beside leg A's rather than inside the ISHMAEL port; and the dead species whose
slots it writes are exactly the ones `_ice_empty_rates()` fills with hard zeros, so a pass
that only ever ADDS to those zeros cannot disturb the bitwise inertness gates.

# The two branches, and why the environment picks between them

ABOVE `T_0` the water is liquid. The mass transfers to the RAIN with `L_f` absorbed exactly
as a melt would: `SRC_i<k>q` loses `ρ^∅/τ_pop`, `ICE_R` gains it with the melt legs' own sign
convention (`ice_r` is credited `−MLQ` there, and `MLQ ≤ 0`), and `FRZ_NET ≡ −(ICE_C + ICE_R)`
therefore falls by the same number — so the transfer appears in `dT_nc`/`dp_nc` as a cooling
of `L_f` per unit mass, which is what melting ice is. Mass and energy close by construction
because the identity is maintained as a difference, not accumulated independently.

The rain NUMBER has no melt analogue to copy: `MLN` maps melted crystals one-to-one onto
drops, and here there are no crystals. The rule the two-moment rain closure already uses for
a mass source that arrives with no number of its own is AUTOCONVERSION's
(`rain_autoconversion_2m`, `Ṅ = Q̇/RAIN_2M_M_AUTO`), so that is what is mirrored: one 25 μm
drop per [`RAIN_2M_M_AUTO`](@ref) of returned mass, seeded into `ICE_NR` — the same slot,
sign and units the melt credit uses. The a/c VOLUME moments of the dead species are relaxed
by the SAME fraction `1/τ_pop` (leg A's shared-factor rule), so the species leaves in one
piece; its number is already `≤ 0` and `max(n, 0) = 0` gives it nothing to relax.

BELOW `T_0` the water is ice that has lost its crystals to the transport, and it is given
them back — always as number alone, at one of TWO crystals. NO mass moves either way, no
latent heat is released, and nothing enters `λ` or `N`: this branch creates number and
nothing else.

`options[:ice_population_seed]` picks the crystal. `:min` and `:large` are the two ends of
the same statement about the dead mass; `:local` is the third answer, and it is the one the
DEFECT rather than the mass argues for.

`:min` — the SMALLEST crystal the scheme resolves,

    ṅ_{i,k} = ρ^∅_{i,k} / (m_min τ_pop),   m_min = ISHMAEL_M_MIN (the 2 μm sphere)

with the volume moments from [`_ice_nucleation_volume`](@ref) at that mass/number pair, so
`a` and `c` describe the 2 μm spheres every nucleation channel of the port already seeds. It
is the source-side statement of the principle the `min(n/τ, ṁ/m_min)` bound in
[`_ice_homogeneous_rates`](@ref) states at the nucleation channels — no ice number may be
created below the minimum resolved crystal — and it is the FASTEST RESPONSE: smallest
particles, slowest fall, largest surface per unit mass, so the population sublimates within
seconds if the air is subsaturated and grows if it is not.

`:large` — the particles the dead mass ACTUALLY IS. Number-less mass is what SIZE SORTING
leaves behind: the mass-weighted fall speed outran the number-weighted one, so what is
sitting there is the big end of a distribution whose number has gone somewhere else. Seeding
it as 2 μm spheres puts ~1e12 crystals/m³ on 6e-3 kg/m³ of it, a deposition surface stiff
enough to sublimate inside one step (measured `ts/τ` 31 on ice1, 0.36 → 9.7 on ice3) and to
thin the ice cloud threefold. [`_ice_population_seed_large`](@ref) instead asks `var_check`
what population that mass has when its number is at the floor — ISHMAEL's own re-diagnosis,
the 1 mm large-ice limit at the species' bulk density — and seeds THAT, `ṅ = ρ^∅/(m_large
τ_pop)` with `m_large = ρ^∅/n_large` and the `a`/`c` moments carried along at the same
per-crystal size, so the seeded triple is a fixed point of the checker.

`:local` — THE NEIGHBOURS' CRYSTALS. Both of the above read the dead mass and ask what
particles a mass of that size is; neither reads the DEFECT. The defect is the negative lobe
of the number moment's spline ringing at cloud edges and gradients — the mass-without-number
face of the transport-decorrelation family whose other face is the 3e13 /L number spikes
sitting right next to it — so the crystals this point has lost are not a size to be derived
from its mass at all: they are the crystals of the same species one gridpoint away. The seed
is therefore the per-crystal state of the NEAREST LIVE gridpoint of that species IN THIS
COLUMN ([`_ice_live_neighbour`](@ref)), `ṅ = ρ^∅/(m_loc τ_pop)` with `m_loc = ρ_{i,k}/n_{i,k}`
there, and the `a`/`c` moments at that neighbour's own per-crystal axes so the seeded
population has its habit ([`_ice_population_seed_local`](@ref), which clamps `m_loc` into
`[ISHMAEL_M_MIN, ISHMAEL_M_LARGE[k]]` and passes the triple through `var_check`). Where the
species is live NOWHERE in the column the seeding falls back to `:min`: that is the pure-dead
case, there is no habit anywhere to inherit, and the fastest-responding population is the
right one for a column that has to resolve itself from its own thermodynamics.

Whichever of the three is chosen, the moments are realizable again and the rates, the
consistency source and the sedimentation own the species from the next step on.

The DEFAULT is `:local`, and it is chosen BY MEASUREMENT and not by argument
(reference/FINDINGS_ISHMAEL_S8S9.md §§5f-5j, quick production, 3600 s). `:min` reconciles the
dead mass (whole-cloud dead fraction 70% → 3%) but pays for it with a deposition surface stiff
enough to sublimate the seeded population inside a step, and the cloud it leaves is a third of
the unreconciled one (IWP 3.43 → 1.07, ice top 18.8 → 15.4 km, 6/11 windows). `:large` seeds one
crystal per 4.6e-4 kg — 5.4e5 /m³ over the whole run against 4.5e12 for `:min` — which does not
survive the number field's own ringing: the mass is dead again the next step and the cloud
stays 72% dead, its windows passing only because nothing was reconciled. `:local` keeps the
cloud at its unreconciled magnitude (IWP 3.30, ice top 19.2 km) and reconciles it: with the
minimum-crystal bound and the rain population gate in place it reads 2.7% dead mass at 3600 s,
9/11 windows, and `max n_i1` 1.7e5 /L against 9.4e13 /L before Stage 3 (§5j). Its earlier
re-glaciation defect (69% dead at 3600 s with species 1's number lost through the glaciated
hour) was the Bigg number pump, not the seeding, and is gone with it. `:min` and `:large`
remain selectable and unchanged.

`apply = false` (`options[:ice_population_source] = false`, and also
`options[:condensation] = false`) keeps the census and drops both transfers, reproducing the
unreconciled tree bitwise while still reporting the defect.
"""
function _ice_population_reconcile!(S, st, tid::Int64, Tk, rho_d,
                                    tau_pop::Float64, apply::Bool, ts::Float64,
                                    seed::Symbol = :min)

    itau = 1.0 / tau_pop
    Q = (S.i1q, S.i2q, S.i3q)
    N = (S.i1n, S.i2n, S.i3n)
    A = (S.i1a, S.i2a, S.i3a)
    C = (S.i1c, S.i2c, S.i3c)
    SRCq = (S.SRC_i1q, S.SRC_i2q, S.SRC_i3q)
    SRCn = (S.SRC_i1n, S.SRC_i2n, S.SRC_i3n)
    SRCa = (S.SRC_i1a, S.SRC_i2a, S.SRC_i3a)
    SRCc = (S.SRC_i1c, S.SRC_i2c, S.SRC_i3c)
    seed_large = seed === :large
    seed_local = seed === :local
    # The `:local` seeding's LIVE SPANS, one sweep per species over this column, taken before
    # the gridpoint loop because every seeded point reads them and none of them writes the
    # raw slots. `(0, 0)` per species otherwise: the tuple is built unconditionally so the
    # loop below stays one shape, and the branch that reads it is the one `:local` enters.
    idx = eachindex(Tk)
    spans = seed_local ?
        (_ice_live_span(Q[1], N[1], rho_d, idx), _ice_live_span(Q[2], N[2], rho_d, idx),
         _ice_live_span(Q[3], N[3], rho_d, idx)) :
        ((0, 0), (0, 0), (0, 0))
    worst = 0.0
    pts = 0.0
    to_rain = 0.0
    seeded = 0.0
    @inbounds for i in eachindex(Tk)
        warm = Tk[i] > T_0
        rhoair = rho_d[i]
        for k in 1:3
            rate, rho_empty = ice_population_rate(Q[k][i], N[k][i], rhoair, tau_pop)
            rho_empty > 0.0 || continue
            rho_empty > worst && (worst = rho_empty)
            pts += 1.0
            apply || continue
            if warm
                # The species leaves as rain, one shared factor across the moments it has.
                SRCq[k][i] -= rate
                SRCa[k][i] -= itau * max(A[k][i], 0.0)
                SRCc[k][i] -= itau * max(C[k][i], 0.0)
                nseed = rate / RAIN_2M_M_AUTO
                S.ICE_R[i] += rate
                S.ICE_NR[i] += nseed
                # `FRZ_NET ≡ −(ICE_C + ICE_R)`, maintained as the difference it is defined
                # to be: the ice→rain transfer ABSORBS L_f, so the freeze net falls.
                S.FRZ_NET[i] -= rate
                to_rain += rate * ts
            else
                # Number only: no mass moves, so no latent heat and nothing in λ or N.
                nb = 0
                if seed_local
                    # The crystals next door: the nearest gridpoint of THIS column at which
                    # this species still has a population. `nb == 0` is the pure-dead column
                    # and falls through to the `:min` branch below.
                    sp = @inbounds spans[k]
                    nb = _ice_live_neighbour(Q[k], N[k], rho_d, i, sp[1], sp[2])
                end
                if seed_large || nb > 0
                    # `:large` — the size-sorted particles the dead mass IS: `var_check`'s own
                    # re-diagnosis of it at the floor number. `:local` — the same mass at the
                    # per-crystal mass, habit and bulk density of the live neighbour. Both
                    # rules are non-linear in the mass (the large-ice cap on one, the clamps
                    # and the checker on the other), so the MIXING RATIO is formed and the
                    # answer converted back rather than the rule rescaled.
                    q_empty = rho_empty / rhoair
                    if seed_large
                        nl, al, cl = _ice_population_seed_large(q_empty, k)
                    else
                        nnb = @inbounds N[k][nb]
                        nl, al, cl = _ice_population_seed_local(q_empty,
                                         (@inbounds Q[k][nb]) / nnb, (@inbounds A[k][nb]) / nnb,
                                         (@inbounds C[k][nb]) / nnb, k)
                    end
                    nseed = (nl * rhoair) * itau
                    aseed = (al * rhoair) * itau
                    cseed = (cl * rhoair) * itau
                else
                    nseed = rate / ISHMAEL_M_MIN
                    aseed = _ice_nucleation_volume(rate, nseed)
                    cseed = aseed
                end
                SRCn[k][i] += nseed
                SRCa[k][i] += aseed
                SRCc[k][i] += cseed
                seeded += nseed * ts
            end
        end
    end
    if size(st, 2) > 0
        @inbounds begin
            worst > st[MC_POP_MAX, tid] && (st[MC_POP_MAX, tid] = worst)
            st[MC_POP_PTS, tid] += pts
            st[MC_POP_RAIN, tid] += to_rain
            st[MC_POP_SEED, tid] += seeded
        end
    end
    return nothing
end

"""
    mc_stiffness_trace(mtile, t)

Report the [`mc_stiffness_census!`](@ref) accumulators, and warn ONCE per run if any channel
has gone past `ts/τ = 1`.

Called from the same single-threaded pre-column-loop slot as [`water_negativity_trace`](@ref)
and [`water_budget_trace`](@ref), so the reduction across threads is race-free.

Two independent outputs:

  * the `@warn` is UNCONDITIONAL — it does not depend on `options[:stiffness_trace]`, because a
    time step too large for the microphysics is not something a run should be able to hide by
    leaving a diagnostic switched off. It fires on the first step at which the census shows an
    exceedance and then never again (`:s_warned`, thread 1), so an under-resolved run reports
    one line rather than one per step;
  * `options[:stiffness_trace]::Int` is the periodic `@info`, in steps; absent or `0` disables
    it. The census itself always runs, so switching this on mid-diagnosis costs nothing and
    changes no number.

Neither path touches a rate. The remedy for a large census is a smaller `ts` (or, where the
stiffness is a resolved physical timescale rather than a numerical one, a formulation that
does not relax on it) — never a limiter: see [`qss_condensation_rates`](@ref).
"""
function mc_stiffness_trace(mtile::ModelTile, t::Int64)

    st = mtile.mc_water_stats
    size(st, 2) == 0 && return nothing

    worst = 0.0
    worst_ch = 0
    @inbounds for ch in 1:MC_STIFF_CHANNELS
        row = MC_STIFF_FIRST + (MC_STIFF_N * (ch - 1))
        m = maximum(view(st, row, :))
        m > worst && (worst = m; worst_ch = ch)
    end

    if worst > 1.0 && st[MC_STIFF_WARNED, 1] == 0.0
        st[MC_STIFF_WARNED, 1] = 1.0
        @warn """UNRESOLVED MICROPHYSICS: the supersaturation relaxation is faster than this time step.
          step $t (t = $(round(t * mtile.model.ts; digits=1)) s), ts = $(mtile.model.ts) s
          worst channel: $(MC_STIFF_NAMES[worst_ch]), max ts/tau = $worst
          This is a statement about RESOLUTION, not about stability. The cloud, rain and ice
          channels all relax the shared Q_ss, and that pair is integrated by its own exact
          propagator (relaxation_adjustment_qss!), which is exact at any ts/tau and lands on
          the quasi-steady state of the TeX's Eq. wbf_qs in a single step as ts/tau -> Inf.
          So a large reading here marks air in which the quasi-steady limit is being taken
          INSIDE one step -- correctly, but unresolved in time -- rather than air in which the
          integrator is about to fail. A run whose census is large is a run whose fast
          microphysics is unresolved, and that is a statement the diagnostics should make
          rather than one the rates should silently absorb.
          The nucleation-channel relaxations (tau_hf, tau_act, tau_vc) remain on the multistep
          and keep the census's ORIGINAL meaning: for those, > 1 is a stability warning and the
          remedy is a smaller ts.
          Set options[:stiffness_trace] = N to print the full per-channel census every N steps."""
    end

    # The DONOR-DEPLETION census: a separate statement, and a much sharper one. Above 1 here
    # is not "unresolved", it is the realization construction failing to bound a reservoir.
    dworst = 0.0
    dworst_ch = 0
    # The three ICE NUMBER rows join the CONVICTION only once their realization is on. With
    # it off those legs ride the species' MASS factor by construction, so a reading above 1
    # there is the measurement this stage was built to take and not a broken invariant — it
    # is reported in `dreport` either way, and warned about only where the construction
    # claims to bound it.
    ice_n_real = get(mtile.model.options, :ice_number_realization, false)::Bool
    dlast = ice_n_real ? MC_DONOR_CHANNELS : MC_DONOR_N1 - 1
    @inbounds for ch in 1:dlast
        m = maximum(view(st, MC_DONOR_FIRST + (MC_DONOR_N * (ch - 1)), :))
        m > dworst && (dworst = m; dworst_ch = ch)
    end
    if dworst > 1.0 + 1.0e-9 && st[MC_DONOR_WARNED, 1] == 0.0
        st[MC_DONOR_WARNED, 1] = 1.0
        @warn """OVER-DEPLETION: a step removed more of a reservoir than was in it.
          step $t (t = $(round(t * mtile.model.ts; digits=1)) s), ts = $(mtile.model.ts) s
          worst donor: $(MC_DONOR_NAMES[dworst_ch]), max realized depletion = $dworst
          For the three LIQUID donors this must not happen: every sink on a reservoir is
          realized on that reservoir's TOTAL conductance (_ice_donor_factors), so the applied
          fraction is 1 - exp(-kappa_tot*dt) < 1 identically. A reading above 1 there means a
          sink was added to a donor without being added to its conductance. The rain's total
          includes EVAPORATION as well as the ice legs (Stage 2b), which is why its row is
          the combined draw.
          The three ICE MASS donors are realized the same way -- melting, aggregation and
          sublimation are all shares of one per-species conductance (mc_ice_sources!) -- so a
          reading above 1 there says the same thing: a sink is drawing on a reservoir it was
          never added to the conductance of.
          The three ICE NUMBER donors (n_i1/n_i2/n_i3) are in this loop ONLY when
          options[:ice_number_realization] is on, which is what gives them a conductance of
          their own; with it off their legs ride the species' MASS factor and a reading above
          1 there is a measurement, not a defect. Both modes print in the census below.
          REPORTED, NOT LIMITED: no rate is clamped and no state is written back."""
    end

    # The ANCHOR census: announce the FIRST partition detachment once per run. Expected
    # physics at glaciation onset (the front is where the independently advected moments
    # and the anchor disagree), so an @info and not a warning; the reconciliation source
    # holds it at drift scale unless options[:ice_anchor_source] = false.
    agap = maximum(view(st, MC_ANCHOR_GAP, :))
    if agap > 0.0 && st[MC_ANCHOR_WARNED, 1] == 0.0
        st[MC_ANCHOR_WARNED, 1] = 1.0
        src_on = get(mtile.model.options, :ice_anchor_source, true)::Bool
        @info """PARTITION DETACHMENT: the advected ice moments exceed the rho_t anchor's headroom.
          step $t (t = $(round(t * mtile.model.ts; digits=1)) s), max delta_part so far = $agap kg/m^3
          This is the transport-side defect of the partition (TeX Eq. partition_gap): the ice
          mass and the anchor headroom are two discretizations of one water and have drifted
          apart at a sharp gradient. The reconciliation source is $(src_on ?
          "ON (tau_anchor = $(get(mtile.model.physical_params, :tau_ice_anchor, 10.0)) s) and holds it at drift scale" :
          "OFF (options[:ice_anchor_source] = false): the defect is measured and NOT removed").
          Cumulative census in the stiffness trace: max delta_part, gridpoint-steps, removed mass."""
    end

    # The POPULATION census: announce the FIRST number-less ice mass once per run. Like the
    # anchor's, an @info and not a warning — four moments on four independent fits at three
    # weighted fall speeds do not stay mutually realizable, and the reconciliation source is
    # what returns the orphaned mass to a representation the rates can act on.
    pmax = maximum(view(st, MC_POP_MAX, :))
    if pmax > 0.0 && st[MC_POP_WARNED, 1] == 0.0
        st[MC_POP_WARNED, 1] = 1.0
        psrc_on = get(mtile.model.options, :ice_population_source, true)::Bool
        pseed = get(mtile.model.options, :ice_population_seed, :local)::Symbol
        @info """NUMBER-LESS ICE MASS: a species carries mass with no crystals.
          step $t (t = $(round(t * mtile.model.ts; digits=1)) s), max rho_empty so far = $pmax kg/m^3
          The population gate (mc_ice_sources!) correctly refuses every rate and every fall
          speed on mass that carries no number; the same mass is then exempt from every
          device that could remove it. This is the transport-side defect of the population
          (TeX §Reconciliation of the population). The reconciliation source is $(psrc_on ?
          "ON (tau_pop = $(get(mtile.model.physical_params, :tau_ice_population, 10.0)) s): above T_0 the mass returns to the rain with L_f, below T_0 it is given crystals (options[:ice_population_seed] = :$(pseed) — $(pseed === :large ? "var_check's large-ice re-diagnosis of the dead mass, the size-sorted particles it came from" : pseed === :local ? "the per-crystal mass and habit of the nearest live gridpoint of the same species in the column, the crystals the ringing's negative lobe lost" : "the 2 um sphere, the smallest the scheme resolves"))" :
          "OFF (options[:ice_population_source] = false): the defect is measured and NOT removed").
          Cumulative census in the stiffness trace: max rho_empty, gridpoint-steps, mass to rain, number seeded."""
    end

    interval = get(mtile.model.options, :stiffness_trace, 0)::Int
    (interval > 0 && mod(t, interval) == 0) || return nothing

    report = join(("$(MC_STIFF_NAMES[ch]): max ts/tau = " *
                   "$(maximum(view(st, MC_STIFF_FIRST + MC_STIFF_N * (ch - 1), :)))" *
                   ", gridpoint-steps past 1: " *
                   "$(Int(sum(view(st, MC_STIFF_FIRST + MC_STIFF_N * (ch - 1) + 1, :))))"
                   for ch in 1:MC_STIFF_CHANNELS), "\n  ")
    dreport = join(("$(MC_DONOR_NAMES[ch]): max realized depletion = " *
                    "$(maximum(view(st, MC_DONOR_FIRST + MC_DONOR_N * (ch - 1), :)))" *
                    ", gridpoint-steps past 1: " *
                    "$(Int(sum(view(st, MC_DONOR_FIRST + MC_DONOR_N * (ch - 1) + 1, :))))"
                    for ch in 1:MC_DONOR_CHANNELS), "\n  ")
    # The per-channel ATTRIBUTION block, only where it was asked for. It ATTRIBUTES; it does
    # not convict, so there is no warning path here and nothing prints when the option is off
    # (the rows are then identically zero and printing them would be noise).
    areport = ""
    if get(mtile.model.options, :ice_attr_census, false)::Bool
        achan = join(("$(MC_ATTR_NAMES[ch]): max = " *
                      "$(maximum(view(st, MC_ATTR_FIRST + MC_ATTR_N * (ch - 1), :)))" *
                      ", gridpoint-steps: " *
                      "$(Int(sum(view(st, MC_ATTR_FIRST + MC_ATTR_N * (ch - 1) + 1, :))))"
                      for ch in 1:MC_ATTR_CHANNELS), "\n  ")
        areport = """\n  attribution of the donor breaches, per channel (block A's five q_r counts partition the q_r donor count):
  $achan
  above T_0 carrying ice: gridpoint-steps = $(Int(sum(view(st, MC_ATTR_WARM_PTS, :)))), sum rho_ice (raw slots) = $(sum(view(st, MC_ATTR_WARM_MASS, :))) kg/m^3, points with f_a < 1 = $(Int(sum(view(st, MC_ATTR_WARM_FA, :)))), max (1 - f_a) = $(maximum(view(st, MC_ATTR_WARM_DFA, :))), withheld melt = $(sum(view(st, MC_ATTR_WARM_HELD, :))) kg/m^3"""
    end
    @info """microphysics stiffness census step $t (t = $(round(t * mtile.model.ts; digits=1)) s), ts = $(mtile.model.ts) s
  cumulative over the run so far; > 1 means the relaxation is under-resolved
  $report
  donor depletion actually applied (must stay <= 1 for the liquid and ice-MASS donors; the
  three ice-NUMBER rows are bounded only under options[:ice_number_realization]):
  $dreport
  anchor reconciliation (partition defect delta_part; 0 until glaciation onset):
  max delta_part = $(maximum(view(st, MC_ANCHOR_GAP, :))) kg/m^3, gridpoint-steps = $(Int(sum(view(st, MC_ANCHOR_PTS, :)))), removed mass (gridpoint-sum) = $(sum(view(st, MC_ANCHOR_REMOVED, :)))
  population reconciliation (number-less ice mass rho_empty; 0 while every species with mass carries number):
  max rho_empty = $(maximum(view(st, MC_POP_MAX, :))) kg/m^3, (gridpoint x species)-steps = $(Int(sum(view(st, MC_POP_PTS, :)))), mass to rain = $(sum(view(st, MC_POP_RAIN, :))) kg/m^3, ice number seeded = $(sum(view(st, MC_POP_SEED, :))) /m^3 (seed = :$(get(mtile.model.options, :ice_population_seed, :local)))$areport"""
    return nothing
end

"""
    _ileg_shortfall(v) -> Float64

Accumulated positivity shortfall of variable slot `v`'s i-direction splines, or `NaN` when it
cannot be read.

The i-leg bound bites on the WORKER'S PATCH, not on the tile: every call site uses the 3-arg
`splineTransform!(sharedSpectral, patch, mtile.tile)` (`semiimplicit.jl:793`, `:968`,
`nesting.jl:740`, `:879`), which runs `SAtransform_bounded` on `patch.ibasis`. The tile's own
i-splines carry the same bound but are never the ones solved, so their shortfall is
identically zero and reporting it would be a lie of omission. `ModelTile` holds no reference
to the patch — it is a worker-scope binding — hence the introspection, which is confined to
this diagnostic and returns `NaN` (never a misleading `0.0`) whenever the lookup does not
find a real spline patch.
"""
function _ileg_shortfall(v::Int64)
    isdefined(Main, :patch) || return NaN
    p = getfield(Main, :patch)
    hasproperty(p, :ibasis) || return NaN
    ib = p.ibasis
    ib isa Springsteel.NoBasisArray && return 0.0
    eltype(ib.data) <: Springsteel.CubicBSpline.Spline1D || return NaN
    v <= size(ib.data, 2) || return NaN
    return sum(Springsteel.CubicBSpline.bound_shortfall(ib.data[z, v])
               for z in axes(ib.data, 1))
end

"""
    water_budget_trace(mtile, t)

Print the per-step water production budget, then reset it for the next step.

Emitted from the same single-threaded pre-column-loop slot as
[`water_negativity_trace`](@ref), so printing is race-free and the numbers describe the step
that just finished. Gated on `options[:water_budget_trace]::Int` — the print interval in steps;
absent or `0` disables it, and `water_budget_probe!` is then never called, so the hot path is
untouched.

Also reports the POST-reconstruction minimum from `tile.physical`, which is the step-matched
partner of `clamp_water!`'s pre-fit measurement. Their difference is what the fit contributes
per step, the quantity the previous attribution had no way to see (it read only the pre-fit
number, and only from the console, where the run's `@warn` never appears — the worker's stderr
goes to `<output_dir>/scythe_err.log`).
"""
function water_budget_trace(mtile::ModelTile, t::Int64)

    st = mtile.mc_water_stats
    size(st, 2) == 0 && return nothing
    interval = get(mtile.model.options, :water_budget_trace, 0)::Int
    interval > 0 || return nothing

    if mod(t, interval) == 0
        # Reduce across threads: the thread holding the most negative value owns the budget.
        vars = mtile.model.grid_params.vars
        # Positivity limiter health: nonzero shortfall means a column was infeasible (its
        # total mass below the minimum an admissible field can carry), so the limiter
        # created mass instead of redistributing it. This must stay at zero.
        #
        # PER SPECIES and per LEG. Summing over every variable hid which species was
        # infeasible, and reading only the k-basis hid the i-leg entirely — the two together
        # are why "bound_shortfall stays exactly 0.0" was recorded for a configuration whose
        # k-leg shortfall reached 8e3 (see the STAGE 2 section of
        # reference/FINDINGS_NEGATIVE_WATER_ATTRIBUTION.md).
        #
        # The arithmetic below reads slots 8 and 9 as DENSITIES. That is unconditionally
        # correct here because `mc_driver!` refuses `water_budget_trace` under either
        # transform — the probe's ADV column would be in control-variable units while its
        # source columns are densities, so the attribution would not close.
        kb = mtile.tile.kbasis
        kshort = (name) -> kb isa Springsteel.NoBasisArray ? 0.0 :
            Springsteel.CubicBSpline.bound_shortfall(kb.data[mc_slot(vars, name)])
        shortfall = join(("$name k=$(kshort(name)) i=$(_ileg_shortfall(mc_slot(vars, name)))"
                          for name in ("rho_r", "rho_c")), ", ")
        phys = mtile.tile.physical
        kDim = mtile.model.grid_params.kDim
        rho_cbar = view(Springsteel.ref_rho_c(mtile.ref_state), :, 1)
        rr = mc_slot(vars, "rho_r")
        rc = mc_slot(vars, "rho_c")
        rv = mc_slot(vars, "rho_v")
        rho_vbar = mtile.mc_ref_diag.rho_vbar
        post_r = 0.0
        post_c = 0.0
        # The VAPOR is a PROGNOSTIC SLOT, so this is the field itself rather than a residual
        # reassembled from four others. Negative water is a RESOLUTION DIAGNOSTIC and is never
        # clamped (reference/FINDINGS_NEGATIVE_WATER_ATTRIBUTION.md), so this minimum says
        # directly how far the transported vapor undershoots.
        post_v = Inf
        @inbounds for i in axes(phys, 1)
            k = mod1(i, kDim)
            post_r = min(post_r, phys[i, rr, 1])
            rho_c_tot = phys[i, rc, 1] + rho_cbar[k]
            post_c = min(post_c, rho_c_tot)
            post_v = min(post_v, phys[i, rv, 1] + rho_vbar[k])
        end

        # Cloud-top diagnostic: the highest gridpoint carrying condensate, and the coldest
        # temperature anywhere. A positivity limiter redistributes mass WITHIN a leg, so it
        # can only move condensate along that leg — this is what shows whether it is
        # depositing cloud at an altitude the energy budget cannot support.
        z_all = view(mtile.tilepoints, :, size(mtile.tilepoints, 2))
        z_cloud = -Inf
        @inbounds for i in axes(phys, 1)
            (phys[i, rc, 1] + rho_cbar[mod1(i, kDim)]) > 1.0e-6 && (z_cloud = max(z_cloud, z_all[i]))
        end

        # The summary always prints on the interval: with every species bounded the
        # per-species blocks below fall silent, and the limiter health has to stay visible.
        @info """water summary step $t (t = $(round(t * mtile.model.ts; digits=1)) s)
          positivity limiter shortfall (must be 0): $shortfall
          post-fit(reconstruction) minima: rho_r = $post_r, rho_c = $post_c, rho_v = $post_v
          pre-fit(var_np1) minima: rho_r = $(minimum(view(st, MC_PRE_R, :))), rho_c = $(minimum(view(st, MC_PRE_C, :)))
          highest gridpoint with rho_c > 1e-6: $(isfinite(z_cloud) ? z_cloud : NaN) m
          reconciliation gaps (0 at rest by construction; tau times the nudge rates): max|res_rho_t - rho_v| = $(maximum(view(st, MC_VAPOR_GAP, :))), max|Q_ss - (rho_v - rho_vs)| = $(maximum(view(st, MC_QSS_GAP, :))) kg/m^3"""

        # The depletion census: how widely, and how hard, the step is draining each species,
        # and where the REMOVED depletion caps would have bound. `euler` is the forward-Euler
        # fraction, `ab3` what the integrator actually applies; the gap between them is the
        # leading AB3 weight.
        for (name, off, sink) in (("rho_r", MC_DEPLETION_R, "evaporation"),
                                  ("rho_c", MC_DEPLETION_C, "evaporation"),
                                  ("rho_v", MC_DEPLETION_V, "condensation"))
            npts = sum(view(st, off, :))
            npts > 0.0 || continue
            @info """water depletion [$name] step $t (t = $(round(t * mtile.model.ts; digits=1)) s)
              gridpoints with $name > 0: $(Int(npts))
              sinks at or past the AB3 bound (where the REMOVED caps would have bound): $sink $(Int(sum(view(st, off + 1, :)))), auto+coll $(Int(sum(view(st, off + 2, :))))
              MICROPHYSICS depletion fraction (uncapped): max $(maximum(view(st, off + 8, :))), points > 1: $(Int(sum(view(st, off + 9, :))))
              FULL-tendency depletion fraction: Euler $(maximum(view(st, off + 3, :))), ACTUAL(AB3) $(maximum(view(st, off + 4, :)))
              points driven negative by the step: $(Int(sum(view(st, off + 5, :))))
              points past AB3's 0.545 stability limit: $(Int(sum(view(st, off + 6, :))))
              points whose sink HISTORY alone is inadmissible (bound clamped at 0): $(Int(sum(view(st, off + 7, :))))"""
        end

        for (name, offset) in (("rho_r", MC_BUDGET_R), ("rho_c", MC_BUDGET_C))
            tid = argmin(view(st, offset, :))
            val = st[offset, tid]
            val < 0.0 || continue
            trm = (name == "rho_r" ?
                   ("adv", "-rho*div", "Qdot_r", "auto+coll", "-dFr/dz") :
                   ("adv", "-rho*div", "Qdot", "-auto-coll"))
            budget = join(("$(trm[k])=$(st[offset + k, tid])" for k in 1:length(trm)), "  ")
            @info """water budget [$name] step $t (t = $(round(t * mtile.model.ts; digits=1)) s)
              worst AB3-projected point: $name $(st[offset + 10, tid]) -> $val kg/m^3 at z = $(st[offset + 8, tid]) m
              the FORWARD-EULER projection of the same point: $(st[offset + 9, tid]) kg/m^3
              tendency terms [kg/m^3/s]: $budget
              net explicit tendency: $(sum(st[offset + k, tid] for k in 1:5)) kg/m^3/s (x ts = $(mtile.model.ts * sum(st[offset + k, tid] for k in 1:5)))
              local flow: div(v) = $(st[offset + 6, tid]) 1/s, w = $(st[offset + 7, tid]) m/s"""
        end
    end

    # Reset the per-step block regardless of whether this was a print step, so the
    # recorded worst point always belongs to the step just finished. Stops at
    # MC_QSS_GAP: the stiffness census after it is cumulative by design (an excursion at
    # step 40 is still true at step 4000, and the once-per-run warning must be able to see
    # it) and must not be cleared by a diagnostic that happens to be switched on.
    @inbounds fill!(view(st, MC_BUDGET_FIRST:MC_QSS_GAP, :), 0.0)
    return nothing
end

# ── Ice microphysics: the host-side integration of the ISHMAEL process library ────
#
# Everything below turns the ported Fortran rate functions (src/ishmael.jl,
# src/ishmael_tables.jl) into the source terms of Eqs. ice_prog_mass..ice_prog_c of
# reference/Scythe_moist_compressible.tex. Three conventions run through all of it:
#
#   UNITS. The ISHMAEL library is MIXING-RATIO internally (per kg of dry air), exactly as
#   the Fortran is; Scythe is densities. The conversion happens at this boundary and only
#   here: `q = ρ/ρ_d` going in, `× ρ_d` coming out. The one exception is the DEPOSITION
#   rate, which is formed in density directly from `Q_ss` (Eq. dep_rate) because that is the
#   form in which the `(1 + 𝒬_{s,i})` cancellation with the supersaturation equation happens.
#
#   STATE. Every rate is evaluated at the VAR_CHECK-EFFECTIVE moments — the Fortran's own
#   `var_check` consistency clamps applied to the transported (ρ_i, n_i, a_i, c_i) — so the
#   rate functions never see a moment set that does not describe a realizable population.
#   `_ice_effective` is that map, and it is the only place the raw slots are read.
#
#   RATES, NOT UPDATES. The Fortran advances its state between process blocks and
#   re-diagnoses in between. Scythe hands a multistep integrator pure tendencies, so every
#   rate here is evaluated at the SAME state, summed, and returned. Where the Fortran's
#   bookkeeping is expressed as a state update (the aggregation and nucleation volume
#   moments) the increment is divided by the step and used as a rate, which is the same
#   thing to the order the scheme is written in.

# ── The impulse-form rates, converted to finite physical rates ────────────────
#
# Three of ISHMAEL's nucleation rates are written as `reservoir/Δt`: they convert the WHOLE
# available reservoir in exactly one time step. That is the same defect as the depletion caps
# the model already removed (TeX §Departures (b)) with the sign reversed — the rate, and hence
# the converged solution, is a function of the step size, so refining `Δt` does not approach
# any differential equation. It is also what detonated the first live ice arm: `n_c/Δt` at
# `Δt = 0.3 s` injects 3.3e8 sub-micron crystals per m³ per second into the anvil, and the
# resulting `1/τ_i ∝ n_i C̄` and ice-rain collection rate (∝ n_i n_r) both ran away. Measured:
# the divergence sat at the SAME PHYSICAL TIME (986 s at ts = 0.3, 1007 s at ts = 0.1), which
# is the signature of a state runaway rather than of an under-resolved step.
#
# Each is therefore rewritten as a relaxation on a physical timescale. The timescales are
# `physical_params`, they are `Δt`-free, and the stiffness census reports when a step fails to
# resolve them — which is the model's standing answer to a fast process.

"""
    ISHMAEL_IN_CEILING

The DeMott existing-ice ceiling, 1.0e7 m⁻³. This is the Fortran's own `10000` compared against
`n/1000` in m⁻³ (module_mp_jensen_ishmael.F line 1591) — a measured upper bound on activated
ice nuclei, i.e. a STATE ceiling on a concentration, not a `Δt` limiter on a rate, so it stays.
"""
const ISHMAEL_IN_CEILING = 1.0e7

"""
    ISHMAEL_M_MIN

The mass of the smallest ice crystal the scheme resolves: a sphere of radius
[`ISHMAEL_RMIN`](@ref) at the bulk ice density `RHOI = 920 kg/m³` the Fortran assigns to every
freshly nucleated or frozen particle (`rhobar = RHOI·nucfrac + (1−nucfrac)·rhobar`,
module_mp_jensen_ishmael.F line 2246). It is the same particle DeMott seeds
(`mnuccd/nnuccd = (4/3)π RHOI (2e-6)³`, line 1596), so the two channels agree by construction.
"""
const ISHMAEL_M_MIN = ISHMAEL_FOURTHIRDSPI * ISHMAEL_RHOI * ISHMAEL_RMIN^3

"""
    ISHMAEL_RHO_AGG

The bulk density of the AGGREGATE species, 50 kg/m³ — the Fortran's own forced value below
`T_0` (`rhobar(ICE3) = 50.`, module_mp_jensen_ishmael.F line 2603, opening the "Final check on
aggregates" block) and, not by coincidence, the lower bound `var_check` holds every species
above (lines 3131-3154). [`_ice_effective`](@ref) already hands it to an EMPTY aggregate;
[`_ice_population_seed_large`](@ref) is the one place a species that HAS mass is given it.
"""
const ISHMAEL_RHO_AGG = 50.0

"""
    ISHMAEL_M_LARGE

The per-crystal mass of the `:large` seeding, one entry per species [kg]: the mass of a
1 mm sphere at the species' bulk density, `m_large = (4/3)π ρ̄ a_max³ Γ(ν+3)/Γ(ν)` with
`a_max = 1 mm` — 4.62e-4 kg for the planar and columnar species at `RHOI`, 2.51e-5 kg for
the aggregates at [`ISHMAEL_RHO_AGG`](@ref).

This is [`_ice_population_seed_large`](@ref)'s own answer written as a mass, and it is the
LARGEST per-crystal mass `var_check` admits: the mass is maximized over the checker's
admissible set at the sphere (`ρ̄ = ρ_max`, `a_n = c_n = 1 mm`), because any aspect ratio
other than one puts the shorter axis inside the cap. [`_ice_population_seed_local`](@ref)
clamps its inherited per-crystal mass against it, so a crystal read out of ringing-corrupted
slots can never seed a coarser population than `:large` would — and `:local`'s number is
therefore bracketed by the other two seedings by construction.
"""
const ISHMAEL_M_LARGE = let a3 = 1.0e-3^3,
                            g = gamma(ISHMAEL_NU + 3.0) * ISHMAEL_I_GAMMNU
    (ISHMAEL_FOURTHIRDSPI * ISHMAEL_RHOI * a3 * g,
     ISHMAEL_FOURTHIRDSPI * ISHMAEL_RHOI * a3 * g,
     ISHMAEL_FOURTHIRDSPI * ISHMAEL_RHO_AGG * a3 * g)
end

"""
    _ice_donor_factors(hf, bg, p1, p2, p3, qc, qr, nr, kev, dt) -> (f_qc, f_qr, f_nr)

The realization factor of each LIQUID DONOR RESERVOIR at one gridpoint: cloud mass, rain mass,
rain number. One factor per reservoir, formed from that reservoir's TOTAL sink conductance over
every leg drawing on it, and shared out to those legs in proportion to their rates
([`relaxation_realization`](@ref) states the rule and why `J₀` is an integrator coefficient
rather than a rate law).

# Why a per-DONOR total and not a per-leg factor

Realizing legs one at a time bounds each leg at one reservoir and lets the SUM run past it —
measured on the O01 ice arm as a realized rain depletion pinned at exactly 3.0 reservoirs per
step. It is also whack-a-mole: the wall moved 1031.4 s → 2457.3 s when the freezing legs were
realized, then 2538.6 s when ice-rain collection was, with the next stiffest leg simply taking
over each time (nominal `κΔt` for the rain donor measured at 1.7e18 — 18 orders above 1, so no
step refinement ever reaches it). The donor total is the statement the ODE actually makes,
`dq/dt = −(Σ_j κ_j) q`, and it is the same combined-rate doctrine the TeX derivation uses for
the supersaturation pair, where `λ = Σ τ⁻¹` and each channel takes its share of one propagator.

# The legs, by reservoir

  * `q_c`  homogeneous freezing `ṁ_hf,c`, and the CLOUD half of every species' riming;
  * `q_r`  homogeneous freezing `ṁ_hf,r`, Bigg `ṁ_Bigg`, the RAIN half of every species'
           riming, every species' ice-rain collection `dQRfzri`, and — since Stage 2b —
           EVAPORATION, through the conductance `kev` the caller supplies;
  * `n_r`  the number partners of the same: `ṅ_hf,r`, `ṅ_Bigg`, `dNfzri`, the drops riming
           removes (`nrn_loss`), and the evaporation number sink at the same `kev`.

# EVAPORATION is a sink of the rain like any other (Stage 2b)

`kev = max(−Q̇_r, 0)/ρ_r` is the rain-channel evaporation of `qss_condensation_rates` written
as a conductance on the rain (TeX §donor_relax, "The rain reservoir has a third sink that
lived outside its conductance"). It was outside every conductance in this file: the
exponential propagator of the relaxation pair bounds `Q_ss`, not `ρ_r`, so the vapor DEFICIT
limited the applied evaporation and nothing limited it by the rain that is there. The ice legs
were then realized on a conductance that did not contain it, and the two draws on ONE
reservoir summed without either knowing of the other — measured on the Stage 0a fixture at a
combined `MC_ATTR_QR_COMB` of 1.0012 over 1296 gridpoint-steps while `MC_DONOR_QR` itself, which
saw only the ice half, read 1.0. Adding it to `κ_tot` is the closure sublimation already had on
the ice side, and the SAME realized number then goes into `λ`, into `N` and into the rain
transfer (the fold of `f_rain` into `invtau_r`/`Qdot_r` in `mc_driver!`).

The NUMBER sink is exactly proportional to the mass sink — `rain_number_evaporation_2m` is
`Q̇_r·n_r/max(ρ_r, RHO_R_MIN)`, i.e. the same `κ_ev` acting on `n_r` — so the number donor takes
the IDENTICAL conductance rather than a scaled one (contrast Bigg, whose number conductance is
a twentieth of its mass conductance, which is why the two moments need separate factors at
all). The two agree exactly above `RHO_R_MIN`; in the decade between the census floor and it
the number's true conductance is the SMALLER of the two, so `κ_ev` over-states it and the
factor errs toward realizing more — the safe direction, on a reservoir with no rain in it. It is added to `f_nr`'s total, which is what the ICE number legs are realized on; the
number sink itself is applied through `NR_SRC` on the multistep, beside self-collection, which
carries no conductance either.

`kev` is passed as a CONDUCTANCE, not as a rate: `κ_evΔt` is dimensionless in either unit
system, and multiplying by the caller's `qr`/`nr` here is what puts it in the same
mixing-ratio units as every other leg of the sum. `kev = 0.0` restores the pre-Stage-2b
factors bitwise — `relaxation_realization` returns an exact `1.0` at a zero rate, and adding
an exact `0.0` to a sum of non-negative rates is the identity — which is what
`options[:rain_evap_realization] = false` uses.

Mass and number are separate reservoirs with separate conductances, and get separate factors —
the ICE species are the same statement in the same shape (`options[:ice_number_realization]`,
Stage 1b; see the number-donor block in [`mc_ice_sources!`](@ref)), and this is where the rule
was first forced:
Bigg's number conductance is `1/20` of its mass conductance (`ṅ/ṁ = λ_r³/20πρ_w` against
`n_r/q_r = λ_r³/πρ_w`), so realizing the number on the MASS factor would freeze all of the
rain's mass while removing a twentieth of its drops — mass without number, precisely the state
the population gate exists to forbid. The cloud has no prognostic droplet number (`n_c` is the
closure's prescribed concentration) and `hf`'s two cloud legs share a conductance by
construction, so one factor serves both there.

`dQIfzri`, the ice that changes species because it collected rain, rides `f_qr`: it is the same
collision events, so it is a share of the rain reservoir's sink like its partners.

# The price of two factors, and where it is paid

Two factors on one pair of moments is right for the RESERVOIRS and wrong for the CRYSTALS: a
number rate and its mass partner scaled by `f_nr` and `f_qr` no longer stand in the ratio the
kernel wrote them in, so any bound of the form `ṅ ≤ ṁ/m_min` stated at the RATE does not
survive this function. `min` commutes with one shared factor and with nothing else. Measured at
the anvil top (reference/FINDINGS_ISHMAEL_S8S9.md §5i): `f_nr = 1.0` — `relaxation_realization`
at a zero reservoir, correct for a factor and catastrophic for a bound — against `f_qr` at
1.8e-19, the same Bigg collisions realized seventeen orders apart, ~1e15 m⁻³ s⁻¹ of crystals
created carrying a millionth of `ISHMAEL_M_MIN` each. The bound is therefore restated on the
REALIZED pair, in [`mc_ice_sources!`](@ref)'s slot assembly where both factors exist, and this
function is left to do the one job it is for. `f_nr` is NOT the place to fix it: clamping a
donor factor to its partner's would under-realize every healthy number sink on the reservoir.

# Riming enters through `prdr₀`, not through `rimesum`

The riming contribution is the phase-A **unit-realization mass gain** `prdr₀`, split cloud/rain
by the Fortran's own `qcrimefrac`, NOT the linear-limit collection rate `rimesum/ρ_a`. That is
the difference between a bound that holds and one that leaks: `prdr ∝ rnfr³ − rni³`, and
bounding the radius increment against the linear conductance still delivered a measured 2.0
reservoirs at p99 and 13.0 at max. With `κ` formed from `prdr₀` itself the second growth pass
lands in the linear-or-below regime, where `prdr(f) ≤ f·prdr₀`, so the realized riming mass is
bounded by its share of the reservoir along with everything else.

Every factor is an exact `1.0` where its reservoir has no sink, which is what keeps the
conversion-free path — and the whole warm and dry path, where this block never runs — bitwise.
"""
@inline function _ice_donor_factors(hf, bg, p1, p2, p3, qc::Float64, qr::Float64,
                                    nr::Float64, kev::Float64, dt::Float64)
    rim_c = (p1.prdr0_c + p2.prdr0_c) + p3.prdr0_c
    rim_r = (p1.prdr0_r + p2.prdr0_r) + p3.prdr0_r
    col_q = (p1.rr.dQRfzri + p2.rr.dQRfzri) + p3.rr.dQRfzri
    col_n = (p1.rr.dNfzri + p2.rr.dNfzri) + p3.rr.dNfzri
    nrn   = (p1.nrn_loss + p2.nrn_loss) + p3.nrn_loss
    # The two EVAPORATION legs, in the mixing-ratio units this sum is taken in. One
    # conductance, two moments: the number sink is `κ_ev n_r` because
    # `rain_number_evaporation_2m` is proportional to the mass loss with exactly this
    # constant of proportionality.
    ev_r = kev * qr
    ev_n = kev * nr
    f_qc = relaxation_realization(hf.mim + rim_c, qc, dt)
    f_qr = relaxation_realization((((hf.mimr + bg.mbiggr) + rim_r) + col_q) + ev_r, qr, dt)
    f_nr = relaxation_realization((((hf.nimr + bg.nbiggr) + nrn) + col_n) + ev_n, nr, dt)
    return (f_qc, f_qr, f_nr)
end

"""
    _ice_empty_pre() -> NamedTuple

The all-zero phase-A result for a species with no population, matching
[`_ice_species_pre`](@ref)'s shape. Every rate field is an EXACT `0.0`, so a dead species
contributes nothing to any donor conductance and the factors stay exactly `1.0`.
"""
@inline _ice_empty_pre() =
    (rc = (rimesum = 0.0, qi_qc_nrm = 0.0, qi_qc_nrd = 0.0),
     rr = (rimesumr = 0.0, qi_qr_nrm = 0.0, qi_qr_nrd = 0.0, qi_qr_nrn = 0.0,
           dQRfzri = 0.0, dQIfzri = 0.0, dNfzri = 0.0, dQImltri = 0.0, dNmltri = 0.0),
     vc = (Cbar = 0.0, fv = 0.0, fh = 0.0, vtrni1 = 0.0, vtrmi1 = 0.0, vtrzi1 = 0.0),
     rimesum = 0.0, rimesumr = 0.0, capgam = 0.0, fv = 0.0, fh = 0.0,
     invtau_i = 0.0, Qdot = 0.0, niq = 0.0, dry_growth_pre = false,
     prdr0_c = 0.0, prdr0_r = 0.0, nrn_loss = 0.0)


"""
    _ice_homogeneous_rates(temp, qc, nc, qr, nr, tau_hf) -> NamedTuple

Homogeneous freezing below −35 °C as a finite relaxation: everything liquid freezes on the
timescale `tau_hf` (`physical_params[:tau_homogeneous]`, default 5 s) rather than in one step.

    ṁ = ρ_c/τ_hf,   ṅ = min( n_c/τ_hf , ṁ/m_min )

and the same pair for the rain. The Fortran writes `mim = q_c/Δt`, `nim = n_c/Δt` (lines
1545-1553), which makes the rate literally `1/Δt`; this is the `Δt`-free statement of the same
physics — "all supercooled liquid freezes within seconds at −35 °C".

# Why the number is bounded and the mass is not

Every ice NUMBER source must seed crystals at or above the smallest size the scheme resolves,
[`ISHMAEL_RMIN`](@ref) — which is `var_check`'s mean-radius floor and the bottom of the
`itab`/`itabr` table domain at the same time. DeMott seeds 2 μm spheres and Hallett-Mossop
seeds 5 μm splinters, so both comply already. One-crystal-per-droplet does NOT: the cloud
closure carries a FIXED droplet number (`max_N_c`) regardless of the cloud content, so a thin
anvil cloud freezing at −35 °C would hand the scheme 10⁸ m⁻³ crystals of sub-micron size —
particles the habit laws and the collection tables were never fitted for, and whose
`rimedr ∝ 1/r_ni²` axis growth at the clamped floor is unbounded. Measured, before this
bound: the O01 ice arm made 6×10⁷ m⁻³ crystals of 0.13 μm at 990 s, and once the anvil met
them the ice deposition timescale went from `ts/τ = 1.8e-4` to `36.5` in a hundred seconds.

The `min` states the same physics ISHMAEL states with its `n_i ≤ 1000 L⁻¹` state cap, but at
the SOURCE rather than by re-diagnosing the population afterwards: the frozen MASS transfers
in full — no water is lost, and `Q̇_freeze` is untouched — and the number is what that mass
supports at the minimum resolved crystal. It is state-dependent and `Δt`-free, so it is not a
depletion cap: refining `Δt` still approaches the same differential equation.

Where the droplets are LARGER than `r_min` (the ordinary case: 100 cm⁻³ in 1 g/m³ of cloud is
13 μm) the bound does not bind and the drop-preserving transfer `ṅ = n_c/τ_hf` stands exactly.

The RAIN leg carries the same `min` for symmetry and safety. Raindrops are three orders of
magnitude above `r_min`, so it should never bind — `test_moist_compressible.jl` asserts it does
not at typical rain states, which is what makes it a guard rather than a parameterization.

# The bound is stated here and ENFORCED at the assembly

This `min` is a statement about two rates, and it holds for the rates this function returns.
It does NOT survive realization: `mimr` is scaled by the rain-MASS donor factor and `nimr` by
the rain-NUMBER one ([`_ice_donor_factors`](@ref) — two reservoirs, two conductances), and
`min` commutes with one shared factor and no more. The cloud pair shares `f_qc` and is safe by
that accident; the rain pair is not, and neither is Bigg, which carries no rate-level bound at
all. So the invariant this docstring argues for is re-applied to the REALIZED legs in
[`mc_ice_sources!`](@ref)'s slot assembly, over every ice number source at once
(reference/FINDINGS_ISHMAEL_S8S9.md §5i). What is written here still stands and still binds
first; the assembly is what makes it an invariant rather than a property of one call.

The volume-moment seeding needs no change: [`_ice_nucleation_volume`](@ref) derives the
characteristic axis from the summed mass/number ratio at density `RHOI` and is shared by every
nucleation channel, so bounding the ratio here bounds the seeded size there too.

The `-35 °C` threshold and the `QSMALL` existence gates are STATE tests and are unchanged.
"""
@inline function _ice_homogeneous_rates(temp::Float64, qc::Float64, nc::Float64,
                                        qr::Float64, nr::Float64, tau_hf::Float64)

    (temp < (T_0 - 35.0)) || return (mim = 0.0, nim = 0.0, mimr = 0.0, nimr = 0.0)
    itau = 1.0 / tau_hf
    cloud = qc > ISHMAEL_QSMALL
    rain = qr > ISHMAEL_QSMALL
    mim = cloud ? qc * itau : 0.0
    mimr = rain ? qr * itau : 0.0
    return (mim  = mim,
            nim  = cloud ? min(nc * itau, mim / ISHMAEL_M_MIN) : 0.0,
            mimr = mimr,
            nimr = rain ? min(nr * itau, mimr / ISHMAEL_M_MIN) : 0.0)
end

"""
    _ice_demott_rates(temp, sup, rhoair, n_ice, tau_act) -> NamedTuple

DeMott et al. (2010) heterogeneous nucleation as a finite ACTIVATION rate:

    ṅ = max( n_IN(T) − n_ice , 0 ) / τ_act

with `n_IN` the DeMott ice-nuclei concentration [m⁻³] capped at [`ISHMAEL_IN_CEILING`](@ref),
`n_ice` the ice number ALREADY present [m⁻³], and `τ_act` = `physical_params[:tau_activation]`
(default 1 s). The mass follows the number at ISHMAEL's own new-particle size — 2 μm spheres
of density `RHOI` (Fortran lines 1596-1597).

Two things change from [`ishmael_nucleation_demott`](@ref), which stays as the Fortran wrote
it for the reference harness:

  * the rate is `deficit/τ_act`, not `deficit/Δt`;
  * the deficit is taken in the UNCAPPED branch too. The Fortran activates the full `inrate`
    every step until the ceiling binds, so a column that already holds the DeMott
    concentration keeps nucleating; here activation stops when the ice number reaches it,
    which is what "the available nuclei have been used up" means.

The nucleation window (`T < T_0` and supersaturated over water) is a state test and is
unchanged; `n_ice` is the CARRIED ice number, for the reason given in `_ice_species_rates`.
"""
@inline function _ice_demott_rates(temp::Float64, sup::Float64, rhoair::Float64,
                                   n_ice::Float64, tau_act::Float64)

    (temp < T_0 && sup >= 0.0) || return (mnuccd = 0.0, nnuccd = 0.0)

    dT = 273.16 - temp
    # The DeMott fit, verbatim (Fortran lines 1583-1588). `0.03` is the Chagnon and Junge
    # (1962) large-aerosol number; the result is #/L and is converted to m⁻³ here.
    n_IN = 1000.0 * 0.0000594 * dT^3.33 * 0.03^((0.0264 * dT) + 0.0033)
    target = min(n_IN, ISHMAEL_IN_CEILING)
    deficit = max(target - n_ice, 0.0)               # m⁻³
    nnuccd = deficit / (tau_act * rhoair)            # # kg⁻¹ s⁻¹
    # New particles are `r_min` spheres at `RHOI` — [`ISHMAEL_M_MIN`](@ref), the SAME crystal
    # the homogeneous leg is bounded against, so the two channels seed one population.
    mnuccd = nnuccd * ISHMAEL_M_MIN
    return (mnuccd = mnuccd, nnuccd = nnuccd)
end

"""
    _ice_effective(qi, ni, ai, ci, k, rhomax = ISHMAEL_RHOI) -> NamedTuple

One ice species' EFFECTIVE moments: the incoming-volume floors of the Fortran host loop
(module_mp_jensen_ishmael.F lines 1013-1032) followed by [`ishmael_var_check`](@ref).

`qi`, `ni`, `ai`, `ci` are MIXING RATIOS (per kg dry air) and are assumed already floored at
zero by the caller. Returns `var_check`'s tuple — `(deltastr, ani, cni, rni, rhobar, ni, ai,
ci, alphstr, alphv, betam)` — which is what every rate function downstream takes.

An EMPTY species (`qi ≤ QSMALL`) returns the Fortran's own top-of-loop initialization instead
(lines 874-901): number and volumes at their floors, 2 μm spherical axes, `deltastr = 1`, and
`rhobar = RHOI` except for the aggregate species, where it is 50 kg/m³. `var_check` cannot be
run on an empty species — its density re-derivation divides by the mass — and the aggregation
block still needs a well-defined characteristic diameter for a species that has no ice yet,
because that is exactly the species aggregation CREATES.

`k` is the species index; only `k == 3` (aggregates) reads it.

`rhomax` is the CEILING `var_check` clamps the bulk density against (its `RHOI` argument),
and defaults to the bulk ice density every existing call site uses, so they are unchanged.
The one caller that passes anything else is [`_ice_population_seed_large`](@ref), which hands
the AGGREGATE species its own 50 kg/m³ (module_mp_jensen_ishmael.F line 2603, and the same
number this function's own empty-species fallback gives it two lines above).
"""
@inline function _ice_effective(qi::Float64, ni::Float64, ai::Float64, ci::Float64, k::Int,
                                rhomax::Float64 = ISHMAEL_RHOI)

    if !(qi > ISHMAEL_QSMALL)
        return (deltastr = 1.0, ani = 2.0e-6, cni = 2.0e-6, rni = 2.0e-6,
                rhobar = (k == 3 ? 50.0 : ISHMAEL_RHOI),
                ni = ISHMAEL_QNSMALL, ai = ISHMAEL_QASMALL, ci = ISHMAEL_QASMALL,
                alphstr = 1.0, alphv = ISHMAEL_FOURTHIRDSPI, betam = 3.0)
    end

    ni = max(ni, ISHMAEL_QNSMALL)
    ai = max(ai, ISHMAEL_QASMALL)
    ci = max(ci, ISHMAEL_QASMALL)
    ani = max(((ai^2) / (ci * ni))^0.333333333333, 2.0e-6)
    cni = max(((ci^2) / (ai * ni))^0.333333333333, 2.0e-6)
    # Smallest bulk volume for which shape is meaningful: below it, assume spherical.
    if ai < 1.0e-12 || ci < 1.0e-12
        ai = min(ai, ci)
        ci = ai
        ani = (ai / ni)^0.3333333333
        cni = (ci / ni)^0.3333333333
    end
    if ani < 2.0e-6 || cni < 2.0e-6
        ani = 2.0e-6
        cni = 2.0e-6
    end
    ci = cni^2 * ani * ni
    ai = ani^2 * cni * ni
    ds = (log(cni) - log(ISHMAEL_AO)) / (log(ani) - log(ISHMAEL_AO))
    # `rbdum` in is irrelevant: var_check's first act is to re-derive it from (qi, ni, ani).
    return ishmael_var_check(ISHMAEL_NU, ISHMAEL_AO, ISHMAEL_FOURTHIRDSPI, ISHMAEL_GAMMNU,
                             qi, ds, ani, cni, ISHMAEL_RHOI, ni, ai, ci; RHOI = rhomax)
end

"""
    _ice_species_pre(tab, dt, eff, qi, temp, rhoair, air, drive_i, Q_s_i, maxsui, igr,
                       qc, nc, qr, nr, qv, shed) -> NamedTuple

Every per-species ice process rate at one gridpoint, in ISHMAEL's mixing-ratio units except
where noted. `eff` is [`_ice_effective`](@ref)'s tuple, `qi` the species mass mixing ratio,
and `air` a tuple of the level's temperature-dependent air properties (see the caller).

The order is the Fortran's own, because several rates read each other: the two riming lookups
first (they supply `rimesum`/`rimesumr`, which the wet-growth check and the melting rate both
read), then the vapor coefficients, then deposition, riming growth, splintering, and melting.

# Deposition: where the Q_ss closure meets the habit physics

`invtau_i = 4π D_v n_i C̄ f_v` (TeX Eq. tau_ice, density form: `n_i` in #/m³, `C̄` in m, `D_v`
in m²/s) and `Q̇_i = drive · invtau_i / (1 + 𝒬_{s,i})` (Eq. dep_rate) — the rate is Scythe's,
not ISHMAEL's, which is departure (b) of the TeX: the semi-analytic step-mean integration is
replaced by the explicit rate, because `Q_ss` is prognostic here and is advanced by the same
multistep integrator as everything else.

The HABIT physics is then ISHMAEL's, applied as a PARTITION of that realized rate (departure
(d)). The seam is the Fortran's `afn`, whose relation to the mass rate is exact:
`vaporgrow` integrates `r² → r² + 2·afn·(C̄/r)/ρ_dep·(Γ(ν)/Γ(ν+2+δ*))·Δt`, whose volume
derivative is `dV/dt = 4π·afn·C̄/ρ_dep` per particle, so the species mass rate is
`4π n_i C̄ afn` and

    afn = Q̇_i / (4π n_i C̄)

is the inversion that hands `ishmael_deposition_partition` exactly the growth Scythe's own
supersaturation supports. (Substituting the Fortran's own `afn`, which is
`D_v f_v (ρ_v − ρ_{v,i}*)/(1 + 𝒬_{s,i})` written through its diagnosed `sui`, recovers
`Q̇_i = invtau_i·drive/(1+𝒬_{s,i})` identically — the two schemes' underlying mass rate is the
same capacitance-diffusion expression, differing only in the supersaturation supplied to it.)

`maxsui` — which of `{ρ_ice·igr, ρ_ice/igr, ρ_ice}` the deposition density blends toward — is
the caller's, formed from the same clipped drives (see `mc_ice_sources!`).

# What is deliberately NOT ported

The Fortran's `if(abs(prd*dt) < QSMALL*0.01) prd = nrd = ard = crd = 0` smallness cutoff is a
Δt-dependent modification of a rate and goes with the rest of them (TeX §Departures (b)). The
`qi ≤ QSMALL` gate above it is a STATE test and is kept — it is what makes an empty species
cost nothing and produce exact zeros.

# EFFECTIVE moments versus CARRIED moments

`eff` is the var_check-effective state and `ni_c` is the CARRIED number mixing ratio, and the
distinction is load-bearing. In the Fortran the two are the same array: `var_check` has
`INTENT(INOUT)` arguments and its re-diagnosis is WRITTEN BACK into the prognostic state, so
an inconsistent moment set is repaired once and never seen again. Scythe cannot do that — a
per-step state repair is exactly what this model rejects everywhere else — so `var_check` is
used here as a pure map and the carried moments keep whatever transport and the sources gave
them. They can therefore drift arbitrarily far from consistency, and nothing in the effective
state feels it.

The split that follows from that:

  * KERNEL quantities — the axes, the mean radius, the bulk density, the capacitance, the
    ventilation, the fall speeds, the collection lookups and aggregation — read the EFFECTIVE
    moments, because a growth or collision rate is only meaningful for a realizable population.
  * BUDGET quantities — how much NUMBER leaves when a given fraction of the MASS leaves — read
    the CARRIED number, because "sublimating half the ice removes half the crystals" is a
    statement about the crystals that are actually there. Reading the effective number here
    makes the number sink proportional to a quantity the number itself does not appear in,
    which leaves the carried number with no restoring channel at all: measured on the O01 ice
    arm, `n_i1` reached 2.2e10 m⁻³ against 6.7e-11 kg/m³ of mass (9 nm "crystals") and the run
    detonated at t ≈ 1050 s.

The same rule sends the CARRIED number to the DeMott existing-ice cap in `mc_ice_sources!`:
a cap that throttles nucleation on the ice already present must see the ice already present.
"""
function _ice_species_pre(tab::IshmaelTables, dt::Float64, eff, qi::Float64, ni_c::Float64,
                            temp::Float64, rhoair::Float64,
                            mu::Float64, dv::Float64, kt::Float64, nsch::Float64,
                            npr::Float64, xxlv::Float64, xxlf::Float64, qs0::Float64,
                            drive_i::Float64, Q_s_i::Float64, maxsui::Float64,
                            igr::Float64, qc::Float64, nc::Float64, qr::Float64,
                            nr::Float64, qv::Float64, shed::Bool)

    ani = eff.ani; cni = eff.cni; rni = eff.rni; ds = eff.deltastr
    rhobar = eff.rhobar; ni = eff.ni; alphstr = eff.alphstr
    nim3 = ni * rhoair                       # # m^-3, the Fortran's `nim3dum`

    # ── Collection lookups (the itab/itabr families) ──
    rc = ishmael_ice_cloud_riming(tab.itab, rni, qc, ds, rhobar, ni, nc, rhoair)
    rr = ishmael_ice_rain_riming(tab.itabr, rni, qr, nr, ds, rhobar, ni, rhoair, temp, qi)
    rimesum = rc.rimesum
    rimesumr = rr.rimesumr
    rimetotal = rimesum + rimesumr

    # ── Capacitance, ventilation and the (uncapped) fall speeds the growth math reads ──
    vc = ishmael_vapor_coefficients(ani, cni, ds, ISHMAEL_NU, ISHMAEL_I_GAMMNU, alphstr,
                                    rhobar, rhoair, mu, nsch, npr)
    capgam = vc.Cbar
    fv = vc.fv
    fh = vc.fh

    # ── Deposition / sublimation (TeX Eqs. tau_ice, dep_rate) ──
    invtau_i = 4.0 * pi * dv * nim3 * capgam * fv
    # Above T_0 the deposition channel is closed: 𝒟 = 0 there, the ice cannot persist, and
    # `vaporgrow` itself passes straight through (Fortran lines 3473-3479). Melting, not
    # sublimation, is what removes ice above freezing.
    Qdot = temp > T_0 ? 0.0 : drive_i * invtau_i / (1.0 + Q_s_i)
    # The HABIT PARTITION of this rate (`ishmael_deposition_partition` — the a/c-moment
    # sources and the sublimation number sink) is NOT evaluated here. Deposition is one of
    # the two withheld relaxations of the shared `Q_ss`, and its realized mass increment is
    # the STEP-MEAN one; the partition of that increment is evaluated once, at the step-mean
    # rate, in the ETD pre-compute (see `mc_driver!`), so the axes and the number receive
    # exactly the increment the mass slots do. Evaluating it here, at the instantaneous rate,
    # and integrating the result by the multistep put the glaciation burst's one-step spike
    # through the AB3 history (+23/12 then −16/12 of it), which rang the freshly nucleated
    # moments and was the measured death of the ice arm at t = 1031.4 s. This function
    # returns the two coefficients the partition needs that only exist here (`capgam`,
    # the uncapped `vtrmi1`) and the carried-number/mass ratio its sublimation sink is
    # proportional to (Fortran lines 1273-1276: number leaves IN PROPORTION to mass, of the
    # CARRIED number — the only sink the carried number has).
    niq = ni_c / qi

    # ── Riming, PASS ONE: the growth at unit realization ──────────────────────────────────
    # The nonlinear mass gain the collection would produce if nothing throttled it. This is
    # the number the donor's conductance has to be formed from — NOT the linear-limit
    # `rimesum/ρ_a`, because `prdr ∝ rnfr³ − rni³` and the cubic term is exactly what leaked:
    # bounding the radius increment against the linear conductance still delivered a MEASURED
    # 2.0 reservoirs at p99 and 13.0 at max. With `κ` formed from `prdr₀` itself the second
    # pass lands in the linear-or-below regime, where `prdr(f) ≤ f·prdr₀`, and the realized
    # conversion is then bounded by the reservoir. See `test_moist_compressible.jl`.
    dry_growth_pre = ishmael_wet_growth_check(ISHMAEL_NU, temp, rhoair, xxlv, xxlf, qv, dv,
                                              kt, qs0, fv, fh, rimetotal, rni, ni)
    rg0 = ishmael_riming_growth(dt, rni, ds, rhobar, nim3, ani, cni, temp,
                                qc, nc, rc.qi_qc_nrm, rc.qi_qc_nrd, rimesum,
                                qr, nr, rr.qi_qr_nrm, rr.qi_qr_nrd, rimesumr,
                                rhoair, dry_growth_pre, ISHMAEL_NU, ISHMAEL_AO,
                                ISHMAEL_GAMMNU, ISHMAEL_I_GAMMNU, ISHMAEL_FOURTHIRDSPI)
    # The unit-realization split, which is the Fortran's own `qcrimefrac`; it says how much of
    # `prdr₀` each donor is being asked for.
    qcf0 = rimetotal > 0.0 ? clamp(rimesum / rimetotal, 0.0, 1.0) : 0.0
    # SHED (TeX §Departures (e)). Above `T_0` a crystal that collects liquid sheds it, so
    # there is no liquid→ice conversion to put into either donor's conductance. The kernels
    # above still ran and `rimetotal` still reaches the melting rate's sensible-heat term
    # through `pre.rimesum`/`pre.rimesumr`; what is zeroed is the CONSUMER side. Exactly the
    # unshed values below `T_0` (branch not taken), so the sub-freezing path — and every
    # `ishmael.jl` fidelity test, which calls the ported kernels directly — is bitwise.
    shed_now = shed && (temp > T_0)

    return (rc = rc, rr = rr, vc = vc, rimesum = rimesum, rimesumr = rimesumr,
            capgam = capgam, fv = fv, fh = fh, invtau_i = invtau_i, Qdot = Qdot, niq = niq,
            dry_growth_pre = dry_growth_pre,
            prdr0_c = shed_now ? 0.0 : rg0.prdr * qcf0,
            prdr0_r = shed_now ? 0.0 : rg0.prdr * (1.0 - qcf0),
            nrn_loss = rr.qi_qr_nrn * ni * nr * rhoair)
end

"""
    _ice_species_post(tab, dt, eff, qi, ni_c, temp, rhoair, mu, dv, kt, xxlv, xxlf, qs0,
                      qc, nc, qr, nr, qv, pre, f_qc, f_qr) -> NamedTuple

PHASE C of one species' rates: the riming growth at the realized collection, and everything
downstream of it (splintering, melting, fall speeds). Takes the donor factors `f_qc`/`f_qr`
formed across all three species by [`_ice_donor_factors`](@ref) and the phase-A result `pre`.

Returns the same NamedTuple the single-pass `_ice_species_rates` used to, so `mc_ice_sources!`
downstream of the factors is unchanged.

`shed` (`options[:ice_shed_above_t0]`) is TeX §Departures (e): above `T_0` the collection
kernels still run — so `rimetotal` and `dQImltri` still reach the melting rate's sensible-heat
term — but every riming OUTPUT is zeroed, so no mass leaves the cloud or the rain, no axis
grows on rime and no `L_f` is released. Below `T_0` the branch is not taken and every
returned field is bit-for-bit the unshed one.
"""
function _ice_species_post(tab::IshmaelTables, dt::Float64, eff, qi::Float64, ni_c::Float64,
                           temp::Float64, rhoair::Float64, mu::Float64, dv::Float64,
                           kt::Float64, xxlv::Float64, xxlf::Float64, qs0::Float64,
                           qc::Float64, nc::Float64, qr::Float64, nr::Float64, qv::Float64,
                           pre, f_qc::Float64, f_qr::Float64, shed::Bool)

    ani = eff.ani; cni = eff.cni; rni = eff.rni; ds = eff.deltastr
    rhobar = eff.rhobar; ni = eff.ni; ai = eff.ai; ci = eff.ci; alphstr = eff.alphstr
    nim3 = ni * rhoair
    rc = pre.rc; rr = pre.rr
    rimetotal = pre.rimesum + pre.rimesumr

    # ── Riming, PASS TWO: the growth at the REALIZED collection ───────────────────────────
    rg = ishmael_riming_growth(dt, rni, ds, rhobar, nim3, ani, cni, temp,
                               qc, nc, rc.qi_qc_nrm, rc.qi_qc_nrd, pre.rimesum,
                               qr, nr, rr.qi_qr_nrm, rr.qi_qr_nrd, pre.rimesumr,
                               rhoair, pre.dry_growth_pre, ISHMAEL_NU, ISHMAEL_AO,
                               ISHMAEL_GAMMNU, ISHMAEL_I_GAMMNU, ISHMAEL_FOURTHIRDSPI;
                               f_rime_c = f_qc, f_rime_r = f_qr)
    # How the rime mass is split between the cloud and the rain reservoirs. The Fortran keeps
    # this as `qcrimefrac(cc)`; the growth forms the same ratio internally for the density
    # blend, from the REALIZED collections it grew the axes with, so the liquid debit is split
    # the way the mass was actually collected. With both factors 1.0 it is bitwise the
    # Fortran's `rimesum/rimetotal`.
    rimetotal_r = rg.rimesum_r + rg.rimesumr_r
    qcrimefrac = rimetotal_r > 0.0 ? clamp(rg.rimesum_r / rimetotal_r, 0.0, 1.0) : 0.0
    # Hallett-Mossop: splinter mass comes OUT of this species' rime gain and goes to the
    # nucleated species, so `prdr` shrinks while `prdr_pre` — the liquid that was actually
    # collected — does not. The liquid sink is written against `prdr_pre`; see `mc_ice_sources!`.
    sp = ishmael_rime_splintering(temp, rg.prdr)

    # ── SHED ABOVE THE FREEZING LEVEL (TeX §Departures (e)) ──────────────────────────────
    #
    # ISHMAEL's collection kernels carry no temperature gate: above `T_0` the riming branch
    # switches to wet growth (`dry_growth = dry_growth_pre && !(temp > T_0)` in
    # `ishmael_riming_growth`) and the collected liquid is added to the ice mass, on the
    # understanding that it sits as water on the crystal surface and that the melting rate,
    # which carries the sensible heat of the collected liquid, will return it. In ISHMAEL's
    # host that is a free bookkeeping loan: the latent-heat update for freezing is gated to
    # `T ≤ T_0`, so the transfer has no thermodynamic content.
    #
    # Here it is not free. `FRZ_NET ≡ −(ICE_C + ICE_R)` is SIGNED BY THE PARTITION, so mass
    # moving from the rain slot to an ice slot releases `L_f` wherever it happens, and the
    # census of the timestep-limited configuration found the wet-growth branch taking the
    # WHOLE of the rain into the ice in a single step two degrees above freezing, with the
    # melt returning it at its own realized rate and the latent heat of the exchange cycling
    # through the temperature (`MC_ATTR_WARM_RIME_R`/`MC_ATTR_WARM_MELT` are that loop).
    #
    # The departure states the physics directly: above `T_0` a crystal that collects liquid
    # SHEDS it, and there is no liquid-to-ice conversion at all. The collection rate is still
    # evaluated — `rimetotal` below is the unshed one, so `ishmael_melting` keeps the sensible
    # heat of the liquid that struck the crystal, and `dQImltri` is untouched — but no mass
    # leaves the rain or the cloud, no axis grows on rime, and no `L_f` is released. What is
    # then left in `FRZ_NET` above `T_0` is melting alone (negative): homogeneous freezing is
    # gated to `T < T_0 − 35`, Bigg to `T < T_0 − 4`, and `ishmael_ice_rain_riming` already
    # routes `dQRfzri/dQIfzri/dNfzri` to zero and `dQImltri/dNmltri` to the collection above
    # `T_0`. That is the sign property the positive-definiteness of the ice entropy production
    # rests on, and the one the wet-growth transfer had violated.
    #
    # Applied HERE, at the consumer, and not by gating the ported kernel: `ishmael.jl` is a
    # verbatim port with its own fidelity tests, and the reference harness must stay bitwise.
    # Below `T_0` the branch is not taken and every value is the unshed one, bit for bit.
    shed_now = shed && (temp > T_0)
    prdr_pre_v = shed_now ? 0.0 : rg.prdr
    prdr_v     = shed_now ? 0.0 : sp.prdr
    ardr_v     = shed_now ? 0.0 : rg.ardr
    crdr_v     = shed_now ? 0.0 : rg.crdr
    # Hallett-Mossop is already zero outside [265.16, 270.16] K and so above `T_0`; these two
    # are zeroed for the statement, not for the arithmetic.
    qmult_v    = shed_now ? 0.0 : sp.qmult
    nmult_v    = shed_now ? 0.0 : sp.nmult
    qcrimefrac_v = shed_now ? 0.0 : qcrimefrac

    # ── Melting ──
    # `reservoir_caps=false`: the three `Δt`-dependent clauses inside the ported melting rate
    # (the `-qi/Δt` floor, the `ai<1e-12` dump-it-all branch and the `-ni/Δt` number floor)
    # are depletion caps and go with the rest of them.
    ml = ishmael_melting(temp, ni, ani, cni, ds, rhobar, qi, ai, ci, kt, pre.fh, rhoair,
                         xxlv, dv, pre.fv, qs0, qv, xxlf, rimetotal, rr.dQImltri, rr.dNmltri,
                         dt, alphstr, ISHMAEL_GAMMNU, ISHMAEL_I_GAMMNU,
                         ISHMAEL_FOURTHIRDSPI; reservoir_caps = false)
    # The melting NUMBER loss is a budget leg, so it is re-formed on the CARRIED number —
    # the Fortran's own expression (line 1527) evaluated on the crystals that are actually
    # there rather than on the effective population the melting KERNEL (`ml.qmlt`, `ml.amlt`,
    # `ml.cmlt`, which need number and size to agree) reads. No `Δt` floor, for the same
    # reason the kernel above has none.
    nmlt = ml.qmlt < 0.0 ? ((ml.qmlt * ni_c / qi) - rr.dNmltri) : ml.nmlt

    # ── Fall speeds (the capped copy the host model keeps; size-sorting under melting) ──
    # Evaluated at the SAME effective state as every rate above. The Fortran recomputes these
    # after its sequential state update; there is no updated state here to recompute them on,
    # and taking them at the state the tendency is being formed at is what a rate-based
    # scheme means. Negated: `Vt` and every fall speed in this file is downward-NEGATIVE.
    fs = ishmael_fall_speeds(ani, cni, ds, ISHMAEL_NU, ISHMAEL_I_GAMMNU, alphstr, rhobar,
                             rhoair, mu; in_melting = (temp > T_0 && ml.qmlt < 0.0))

    return (invtau_i = pre.invtau_i, Qdot = pre.Qdot, niq = pre.niq, capgam = pre.capgam,
            vtrmi1 = pre.vc.vtrmi1,
            prdr_pre = prdr_pre_v, prdr = prdr_v, ardr = ardr_v, crdr = crdr_v,
            qmult = qmult_v, nmult = nmult_v, qcrimefrac = qcrimefrac_v,
            qmlt = ml.qmlt, nmlt = nmlt, amlt = ml.amlt, cmlt = ml.cmlt,
            dQRfzri = rr.dQRfzri, dQIfzri = rr.dQIfzri, dNfzri = rr.dNfzri,
            nrn_loss = pre.nrn_loss,
            vtrm = -fs.vtrmi1, vtrn = -fs.vtrni1)
end

"""
    _ice_empty_rates() -> NamedTuple

The all-zero counterpart of [`_ice_species_rates`](@ref) for a species with no ice. Every
field is an EXACT `0.0`, which is what keeps the warm/ice-free inertness gate bitwise: the
source accumulators are sums of these, `x + 0.0 === x`, and `_ice_flux!` short-circuits its
spline fit on a fall speed that is identically zero.
"""
@inline _ice_empty_rates() =
    (invtau_i = 0.0, Qdot = 0.0, niq = 0.0, capgam = 0.0, vtrmi1 = 0.0,
     prdr_pre = 0.0, prdr = 0.0, ardr = 0.0, crdr = 0.0,
     qmult = 0.0, nmult = 0.0, qcrimefrac = 0.0,
     qmlt = 0.0, nmlt = 0.0, amlt = 0.0, cmlt = 0.0,
     dQRfzri = 0.0, dQIfzri = 0.0, dNfzri = 0.0, nrn_loss = 0.0,
     vtrm = 0.0, vtrn = 0.0)

"""
    _ice_agg_moments(eff, qi, ni_new, qi_new, rhoair, dt, dnew3, k) -> (Ȧ, Ċ)

The `a`/`c` volume-moment rates [m³/m³/s] that accompany one species' aggregation mass and
number transfer.

ISHMAEL expresses this as a state update (Fortran lines 2404-2478), and it is the one place
the port has to differentiate one: given the post-aggregation mass and number, the axes are
re-diagnosed at CONSTANT `deltastr` and `rhobar` (the assumption stated in the Fortran's own
comment — the loss of ice1/ice2 to aggregates does not significantly change the shape or
density of what is left) and the moments follow by the product rule on `a_i = n⟨a²c⟩`,

    δa_i = 2 n a c δa + n a² δc + a² c δn

which is the Fortran's own three-term expression, divided here by the step.

The AGGREGATE species under `forced3` is different and is not a perturbation: its shape is
IMPOSED (ρ̄ = 50 kg/m³, aspect ratio 0.2, `a = ½·dnew3` from the collection kernel's own
updated characteristic diameter), so its moments are assigned outright and the rate is the
difference from where they were. `forced3` is set only when the aggregation block actually
ran (there is no `dnew3` otherwise); a species-3 mass change from the ice-rain freezing
transfer alone takes the ordinary constant-shape branch.

The same function serves the NUCLEATION TRANSFER — the mass and number a non-nucleated
species loses to the nucleated one when rain freezes onto it (Fortran lines 2330-2347 apply
the identical three-term update for the identical reason). Aggregation and the transfer are
combined into ONE new `(qi, ni)` by the caller and re-diagnosed once, rather than
sequentially: with a rate-based scheme there is no intermediate state for the second
re-diagnosis to be taken at.
"""
@inline function _ice_agg_moments(eff, ni_new::Float64, qi_new::Float64,
                                  rhoair::Float64, dt::Float64, dnew3::Float64,
                                  forced3::Bool)

    (qi_new > ISHMAEL_QSMALL && ni_new > 0.0) || return (0.0, 0.0)
    ds = eff.deltastr
    ni = eff.ni
    aniold = eff.ani
    cniold = eff.cni

    if forced3
        # Forced aggregate shape (Fortran lines 2406-2416).
        ani = 0.5 * dnew3
        cni = 0.2 * ani * ISHMAEL_GAMMNU / gamma(ISHMAEL_NU - 1.0 + ds)
        da = (ani^2 * cni * ni_new) - eff.ai
        dc = (cni^2 * ani * ni_new) - eff.ci
        return (rhoair * da / dt, rhoair * dc / dt)
    end

    alphstr = ISHMAEL_AO^(1.0 - ds)
    gam = gamma(ISHMAEL_NU + 2.0 + ds)
    ani = (qi_new / (ni_new * eff.rhobar * ISHMAEL_FOURTHIRDSPI * alphstr * gam *
                     ISHMAEL_I_GAMMNU))^(1.0 / (2.0 + ds))
    ani = max(ani, 2.0e-6)
    cni = alphstr * ani^ds
    da = (cniold * ni * 2.0 * aniold * (ani - aniold)) +
         (ni * aniold^2 * (cni - cniold)) +
         (aniold^2 * cniold * (ni_new - ni))
    dc = (aniold * ni * 2.0 * cniold * (cni - cniold)) +
         (ni * cniold^2 * (ani - aniold)) +
         (cniold^2 * aniold * (ni_new - ni))
    return (rhoair * da / dt, rhoair * dc / dt)
end

"""
    _ice_nucleation_volume(Qnuc, Nnuc) -> Ȧ

The `a` (and, for the spheres it makes, `c`) volume-moment source [m³/m³/s] that accompanies
a nucleation mass rate `Qnuc` [kg/m³/s] and number rate `Nnuc` [#/m³/s].

New ice particles are spheres of density `RHOI` (Fortran lines 1577-1601 for DeMott; frozen
drops arrive with their own mass and number and the same relation sizes them). Inverting
`Q = N·(4/3)π ρ_i a³ Γ(ν+3)/Γ(ν)` for the characteristic axis and forming `Ȧ = Ṅ⟨a²c⟩ =
Ṅ a³ Γ(ν+3)/Γ(ν)` gives, before the clamps, simply

    Ȧ = Q̇_nuc / ((4/3)π ρ_i)

which is the bulk volume of the nucleated ice — consistent with the TeX's own reading of the
moment, "⁴⁄₃π a_{i,k} is the volume fraction occupied by the species" (§Model variables with
ice). The Fortran's `anuc` clamps to `[2 μm, 3 mm]` are applied to the characteristic axis
before the moment is formed, which is why this is written the long way round rather than as
the one-line identity.

Zero when either rate is non-positive: a mass with no number, or a number with no mass, seeds
no realizable population.
"""
@inline function _ice_nucleation_volume(Qnuc::Float64, Nnuc::Float64)

    (Qnuc > 0.0 && Nnuc > 0.0) || return 0.0
    gam3 = gamma(ISHMAEL_NU + 3.0)
    a3 = (Qnuc * ISHMAEL_GAMMNU) / (ISHMAEL_RHOI * Nnuc * ISHMAEL_FOURTHIRDSPI * gam3)
    anuc = clamp(cbrt(a3), 2.0e-6, 3.0e-3)
    return Nnuc * anuc^3 * gam3 * ISHMAEL_I_GAMMNU
end

"""
    mc_ice_sources!(S, tab, ts, max_N_c) -> Nothing

Fill one column's ICE process sources. Reads and writes only the thread's `mc_scratch`
workspace `S`, so it takes no views and no grid: every input (`Tk`, `p_hPa`, `rho_d`,
`rho_v`, `rho_vs`, `Q_ss`, `rho_c`, `rho_r`, `n_r`, `Q_s_i`, and the twelve recovered ice
moments) is already staged there by `mc_driver!`, and every output goes back to it.

Written:

| scratch | meaning |
|---|---|
| `SRC_i<k><m>` | the twelve process sources `Q̇_{i,k}+Q̇^ρ_k`, `Ṅ_k`, `Ȧ_k`, `Ċ_k` of TeX Eqs. ice_prog_mass..ice_prog_c [density units per second] |
| `Qdot_i<k>` | species k's DEPOSITION rate alone [kg/m³/s] — slots 1 and 7 need it separately from the rest of the mass source |
| `invtau_i<k>` | `1/τ_{i,k}` (Eq. tau_ice), for the stiffness census |
| `FRZ_NET` | `Q̇_freeze`, the net liquid→ice conversion [kg/m³/s] |
| `ICE_C`, `ICE_R`, `ICE_NR` | the back-reactions on cloud mass, rain mass and rain number |
| `Vi<k>m`, `Vi<k>n` | mass- and number-weighted fall speeds [m/s, negative downward] |
| `f_ice<k>`, `f_isn<k>` | species k's MASS donor factor, and the residual factor its sublimation NUMBER sink still owes its own reservoir (Stage 1b) — both read again in the ETD pre-compute |

# Mass conservation, and the one place it forced a departure from the Fortran

Scythe's vapor is PROGNOSTIC (Stage A; TeX §Prognostic vapor), but the density budget still
implies one — `res_rho_t = ρ_t − ρ_d − ρ_l − ρ_i` — and the `rho_v_reconcile` nudge pulls the
transported vapor onto it. Any mismatch between the liquid mass this block removes and the
ice mass it creates therefore still becomes manufactured or destroyed vapor, on `τ_rec`
instead of instantly, with no latent heat accounting. So the freezing channels are written
to satisfy

    FRZ_NET ≡ −(ICE_C + ICE_R)

structurally: `FRZ_NET` is not accumulated independently, it is that difference, which makes
"freezing sources neither ρ_t nor ρ_v and cancels between the ρ_l and ρ_i equations" (TeX
§Water categories) true by construction rather than by inspection.

That required ONE change to ISHMAEL's own accounting. The Fortran subtracts the
Hallett-Mossop splinter mass from the riming rate (`prdr := prdr − qmult`), gives the
splinters to the nucleated species, and then debits the cloud and rain with the REDUCED
`prdr` — so the ice gains `prdr + qmult` while the liquid loses only `prdr`. Here the liquid
is debited with `prdr_pre`, the rime that was actually collected, so the two balance exactly.
Nothing else about the splinter physics changes: the same mass and the same number still move
to the same species.

Everything else in the block is an exact exchange already: aggregation's three transfers sum
to zero by construction (`qagg1 + qagg2 + qagg3 ≡ 0`), the ice-rain freezing transfer moves
mass between two ice species, and deposition moves vapor, which the vapor slot takes as
`-Σ_k Q̇_{i,k}` through `VAPOR_SRC`.

# Gates

`qi ≤ QSMALL` skips a species' whole rate block and returns exact zeros
([`_ice_empty_rates`](@ref)); aggregation runs only for `T ≤ T_0` and only when some species
has ice; deposition is zero above `T_0`. In air that is warm everywhere with no ice, every
number this function writes is an exact `0.0` — which is what the bitwise inertness gate
rests on.

The RAIN POPULATION GATE is the same statement about the liquid side
(`options[:rain_population_gate]`, default true): where the CARRIED rain number is not
positive, the kernels that read a rain SIZE DISTRIBUTION see no rain. `ishmael_rain_lambda`
floors the number at `QNSMALL` and clamps the slope, so a mass-without-number rain slot is
handed to every DSD consumer as 2800 μm drops — the largest the scheme admits, and the pump
that dead-ended the anvil through Bigg freezing (reference/FINDINGS_ISHMAEL_S8S9.md §5i).
`ishmael_ice_rain_riming` self-gates (every rate it returns carries a factor of the carried
`n_r`); Bigg does not, and is the one kernel the gate has to switch off.

The POPULATION GATE below (mass with no carried number is not a population, so no rate acts
on it) leaves that mass exempt from every device that could REMOVE it as well. That is a
representation error of the transport and not a rate, and it is repaired one level up by
[`_ice_population_reconcile!`](@ref), which runs after this function and is the only thing
that writes into a gated species' slots — the fourth tier of the reconciliation chain, TeX
§"Reconciliation of the population".

Above `T_0` the ISHMAEL collection kernels still run, but there is no liquid→ice conversion:
`shed` (`options[:ice_shed_above_t0]`, default true) zeroes the riming rates, their axis
partners, their liquid debits and their share of the donor conductances at the CONSUMER
(`_ice_species_pre`/`_ice_species_post`), so the melting rate keeps the sensible heat of the
liquid that struck the crystal while no mass and no `L_f` move — TeX §Departures (e).
"""
function mc_ice_sources!(S, tab::IshmaelTables, ts::Float64, max_N_c::Float64,
                         active::Bool, var_check_source::Bool, anchor_rates::Bool,
                         attr_census::Bool, agg_caps::Bool, rain_evap_real::Bool,
                         shed::Bool, ice_n_real::Bool, rain_pop_gate::Bool,
                         tau_hf::Float64, tau_act::Float64, tau_vc::Float64,
                         stats, stats_tid::Int64)

    # `dt` reaches ONLY the habit-partition calls (`ishmael_deposition_partition`,
    # `ishmael_riming_growth`, `ishmael_aggregation`, `_ice_agg_moments`), where it is a
    # one-step INCREMENT that is immediately divided by the same step: `ard = δa/Δt`,
    # `qagg/Δt`. Those are consistent first-order discretizations of a rate — they converge to
    # the derivative as `Δt → 0` — and the TeX retains the habit physics verbatim as a
    # PARTITION of a mass increment (§Departures (d)). NO rate law in this block is a function
    # of the step: the three that were (`q_c/Δt`, `n_c/Δt`, the DeMott deficit/Δt) are now
    # relaxations on `tau_hf`, `tau_act` and `tau_vc`, and every reservoir cap of the form
    # `min(rate, ρ/Δt)` inside the ported functions is switched off at the call site.
    dt = ts
    i_dt = 1.0 / dt
    i_tau_vc = 1.0 / tau_vc
    # Census handles, hoisted: `mc_water_stats` has zero columns when the water trace is off,
    # and the census is then skipped entirely rather than branching per gridpoint.
    stx = stats
    tid_x = stats_tid
    census_on = size(stx, 2) > 0
    # The per-channel ATTRIBUTION block (`MC_ATTR_*`) is OPT-IN on top of that: eighteen
    # channels and a five-scalar tail are not something a run that is not asking the question
    # should pay for, and `mc_water_stats` is allocated either way, so the flag is what makes
    # it free. Reported, never limiting; exactly zero on the warm/ice-free path.
    attr_on = census_on && attr_census
    Tk = S.Tk; p_hPa = S.p_hPa; rho_d = S.rho_d; rho_v = S.rho_v; rho_vs = S.rho_vs
    Q_ss = S.Q_ss; rho_c = S.rho_c; rho_r = S.rho_r; n_r = S.n_r; Q_s_i = S.Q_s_i

    # Per-species output columns, gathered once as homogeneous tuples so the species loop is
    # an index rather than three copies of the same block. All `Vector{Float64}`, so the
    # tuple is concrete and `SRCq[k]` costs nothing.
    SRCq = (S.SRC_i1q, S.SRC_i2q, S.SRC_i3q)
    SRCn = (S.SRC_i1n, S.SRC_i2n, S.SRC_i3n)
    # State-n inputs of the step-mean habit partition (see the ETD pre-compute in
    # `mc_driver!`): stashed here, where the effective moments and the vapor coefficients
    # exist, consumed there, where the step-mean deposition rate does.
    habANI = (S.hab1_ani, S.hab2_ani, S.hab3_ani)
    habCNI = (S.hab1_cni, S.hab2_cni, S.hab3_cni)
    habRNI = (S.hab1_rni, S.hab2_rni, S.hab3_rni)
    habDS  = (S.hab1_ds,  S.hab2_ds,  S.hab3_ds)
    habRB  = (S.hab1_rb,  S.hab2_rb,  S.hab3_rb)
    habNIM = (S.hab1_nim3, S.hab2_nim3, S.hab3_nim3)
    habVT  = (S.hab1_vt,  S.hab2_vt,  S.hab3_vt)
    habCG  = (S.hab1_cg,  S.hab2_cg,  S.hab3_cg)
    habNIQ = (S.hab1_niq, S.hab2_niq, S.hab3_niq)
    SRCa = (S.SRC_i1a, S.SRC_i2a, S.SRC_i3a)
    SRCc = (S.SRC_i1c, S.SRC_i2c, S.SRC_i3c)
    QDi  = (S.Qdot_i1, S.Qdot_i2, S.Qdot_i3)
    ITi  = (S.invtau_i1, S.invtau_i2, S.invtau_i3)
    Vm   = (S.Vi1m, S.Vi2m, S.Vi3m)
    Vn   = (S.Vi1n, S.Vi2n, S.Vi3n)

    if !active
        for k in 1:3
            fill!(SRCq[k], 0.0); fill!(SRCn[k], 0.0)
            fill!(SRCa[k], 0.0); fill!(SRCc[k], 0.0)
            fill!(QDi[k], 0.0);  fill!(ITi[k], 0.0)
            fill!(Vm[k], 0.0);   fill!(Vn[k], 0.0)
        end
        fill!(S.FRZ_NET, 0.0); fill!(S.ICE_C, 0.0)
        fill!(S.ICE_R, 0.0);   fill!(S.ICE_NR, 0.0)
        # 1.0, not 0.0: these MULTIPLY the sublimation leg in the ETD pre-compute, so the
        # inert value is the identity. (With `active = false` the conductances are zero and
        # the leg is zero anyway; this keeps the column meaningful rather than relying on it.)
        fill!(S.f_ice1, 1.0); fill!(S.f_ice2, 1.0); fill!(S.f_ice3, 1.0)
        fill!(S.f_isn1, 1.0); fill!(S.f_isn2, 1.0); fill!(S.f_isn3, 1.0)
        fill!(S.etd_dep_a, 0.0); fill!(S.etd_dep_b, 0.0)
        for k in 1:3
            fill!(habANI[k], 0.0); fill!(habCNI[k], 0.0); fill!(habRNI[k], 0.0)
            fill!(habDS[k], 0.0);  fill!(habRB[k], 0.0);  fill!(habNIM[k], 0.0)
            fill!(habVT[k], 0.0);  fill!(habCG[k], 0.0);  fill!(habNIQ[k], 0.0)
        end
        fill!(S.hab_igr, 0.0); fill!(S.hab_maxsui, 0.0); fill!(S.hab_dv, 0.0)
        return nothing
    end

    @inbounds for i in eachindex(Tk)

        temp = Tk[i]
        rhoair = rho_d[i]
        ph = p_hPa[i]

        # ── The level's air properties (Fortran lines 975-991), in SI ──
        mu = 1.496e-6 * temp^1.5 / (temp + 120.0)
        # `vapor_diffusivity` is the liquid channel's own D_v and returns cm²/s (the CGS the
        # `invtau_condensation` chain works in); the ice timescale is assembled in SI, so it
        # is converted here and NOWHERE else. Using the same function for both channels is
        # what makes the shared-reservoir competition of Eq. wbf_qs a competition between two
        # conductances rather than between two diffusivities.
        dv = 1.0e-4 * vapor_diffusivity(temp, ph)
        kt = 2.3823e-2 + (7.1177e-5 * (temp - T_0))
        nsch = mu / (rhoair * dv)
        npr  = mu / (rhoair * kt)
        xxlv = L_v(temp)
        xxlf = L_f(temp)
        qs0 = rho_v_sat(T_0, ph) / rhoair
        igr = get_igr(tab.igrdata, temp)
        (temp - T_0) < -20.0 && (igr = 0.7)   # planar below -20 C (Bailey and Hallett)

        # ── The liquid reservoirs, as ISHMAEL mixing ratios ──
        qc = max(rho_c[i], 0.0) / rhoair
        qr = max(rho_r[i], 0.0) / rhoair
        nr = max(n_r[i], 0.0) / rhoair
        qv = max(rho_v[i], 0.0) / rhoair
        # The cloud droplet number is the HOST's (`max_N_c`, #/cm³), not ISHMAEL's hardcoded
        # 200 cm⁻³: the same number the liquid condensation closure nucleates against, so the
        # riming kernel collects the droplets the cloud channel actually made.
        nc = 1.0e6 * max_N_c / rhoair
        # ── THE RAIN POPULATION GATE ───────────────────────────────────────────────────
        # The ice population gate's statement, applied to the RAIN: a rate may not act on a
        # size distribution the CARRIED number does not support. `nr` is `max(n_r, 0)/ρ_a`
        # just above, so `nr > 0` is exactly `n_r > 0` — the drops the transport is carrying,
        # not a floored or re-diagnosed stand-in for them.
        #
        # The measurement (reference/FINDINGS_ISHMAEL_S8S9.md §5i): at the anvil top the two
        # rain moments decorrelate on their independent spline fits — `n_r` rings to EXACTLY
        # zero while `ρ_r` keeps ~1e-8 kg/kg at 15 km and 192–203 K, rain mass without rain
        # number where there is no rain. `ishmael_rain_lambda` then FLOORS the number at
        # `QNSMALL` and the resulting slope falls into the `lamr < LAMMINR` clamp, so every
        # kernel that reads the DSD is handed a phantom population of 2800 μm drops — the
        # largest particle the scheme admits, the same failure the ice population gate exists
        # to forbid, arriving on the rain and from INSIDE a kernel. Bigg, evaluated 30–40 K
        # below its validity on that phantom, was the number pump that dead-ended the anvil.
        #
        # WHICH kernels need this gate, and which already self-gate:
        #   * `ishmael_bigg_freezing` NEEDS it. Its only existence test is `q_r > QSMALL`
        #     (plus `T < T_0 − 4`), and the floored `nr_adj` it freezes is the DSD's, not the
        #     carried number's, so at `n_r = 0` it returns a large rate rather than nothing.
        #   * `ishmael_ice_rain_riming` (`p_k.rr`) SELF-GATES and is left alone: every rate it
        #     returns carries a factor of the CARRIED `nr` it was passed (`procr·n_i·n_r·ρ_a`
        #     for the four collection moments, `procr[1]·n_i·n_r·ρ_a²` for `rimesumr`, which
        #     then trips its own `QSMALL` test and zeroes `qi_qr_nrm/nrd/nrn` with it), so at
        #     `nr = 0` the whole returned tuple is exact zeros — and with it `nrn_loss`, the
        #     rain-riming branch of `ishmael_riming_growth` (gated on `qi_qr_nrm > 0`) and
        #     every ice–rain leg of the donor conductances.
        #   * The WARM two-moment closures in `microphysics.jl` read the same DSD through
        #     `rain_dsd_2m` and were audited with it. `rain_selfcollection_2m` needed the same
        #     gate and carries its own (the clamped breakup rolloff `dum = −312` turns it into
        #     a number SOURCE on the phantom, which is how a gated rain slot gets a positive
        #     number back and un-gates Bigg on the next step); `rain_number_evaporation_2m`
        #     already tested `n_r <= 0`; `invtau_rain_2m` and `rain_fall_speeds_2m` are
        #     CLAMP-BOUNDED — `n0rr ∝ q_r` at `LAMMINR` makes the first a sink 1.7e3 too SLOW
        #     rather than a source, and the second is bounded to the 3.4x between the two ends
        #     of the clamp — and are deliberately left ungated, since gating either would
        #     strand rain mass that no reconciliation exists to return (their docstrings carry
        #     the measurements).
        #   * `_ice_homogeneous_rates` is NOT a DSD consumer and is out of this gate's scope:
        #     `ṁ = q_r/τ_hf` needs no size distribution, and its number leg is already
        #     `min(n_r/τ_hf, ṁ/m_min)`, which is exactly `0` at `n_r = 0`. The rain MASS it
        #     freezes without number is the ordinary mass-without-number defect the population
        #     reconciliation handles one level up, not a phantom-DSD rate.
        #
        # Bitwise inert wherever `n_r > 0`, which is all healthy rain — the branch is not
        # taken and `bg` is the same call it always was. `options[:rain_population_gate]`,
        # default TRUE; env `SCYTHE_O01_RAINGATE=0` is forensic only.
        live_r = (!rain_pop_gate) || nr > 0.0

        # ── The two drives (TeX Eqs. qss_ice_shift, Dwi) and the habit-density selector ──
        rvs = rho_vs[i]
        rvsi = rho_i_sat(temp, ph)
        Dgap = max(rvs - rvsi, 0.0)
        rv_floor = max(rho_v[i], 0.0)
        drive_w = min(Q_ss[i], rv_floor - rvs)
        drive_i = min(Q_ss[i] + Dgap, rv_floor - rvsi)
        sup = drive_w / rvs
        # `maxsui`, the Fortran's blend weight between the habit-limited and the solid-ice
        # deposition density (lines 3327-3335), rewritten in the TeX's own variables:
        # `sui·q_vi` is the over-ice supersaturation DENSITY (= drive_i) and `q_vs − q_vi` is
        # `𝒟`, so the whole selector is a function of the two drives and needs none of the
        # diagnosed-supersaturation pathway the port excluded.
        maxsui = sup >= 0.0 ? 1.0 :
                 (drive_i >= 0.0 && Dgap > 0.0) ? clamp(drive_i / Dgap, 0.0, 1.0) : 0.0
        Qsi = Q_s_i[i]
        # ── The ICE half of the stiff-relaxation split (TeX Eq. relax_linear) ──
        # The deposition drive written as an AFFINE function of the shared supersaturation,
        # `drive_i = a·Q_ss + b`, which is all the exponential integrator and the step-mean
        # consumers need from this block. Substituting Eq. dep_rate into Eq. Qss_ice, the ice
        # contribution to slot 7 is `−Σ_k Q̇_{i,k}(1+𝒬_{s,i}) = −drive_i·Σ_k τ_{i,k}^{-1}`
        # (the psychrometric factor cancels at the rate level, exactly as the liquid one does),
        # so `a` carries the λ part and `b` the `N` part with no further case analysis:
        #
        #   UNCLIPPED  a = 1, b = 𝒟          →  λ += Σ τ_{i,k}^{-1},  N −= 𝒟 Σ τ_{i,k}^{-1}
        #   CLIPPED    a = 0, b = drive_i     →  λ unchanged,          N −= drive_i Σ τ_{i,k}^{-1}
        #   T > T_0    a = 0, b = 0           →  the channel is shut (Q̇_{i,k} ≡ 0 below)
        #
        # `𝒟` in `N` is the whole content of the Wegener-Bergeron-Findeisen competition: it is
        # the offset that makes the ice relax the SHARED reservoir toward a different zero than
        # the liquid does, and Eq. wbf_qs is `N/λ` with exactly these two numbers in it.
        # The classification is FROZEN at state n, like every coefficient the step freezes.
        if temp > T_0
            S.etd_dep_a[i] = 0.0
            S.etd_dep_b[i] = 0.0
        elseif (rv_floor - rvsi) < (Q_ss[i] + Dgap)
            S.etd_dep_a[i] = 0.0
            S.etd_dep_b[i] = drive_i
        else
            S.etd_dep_a[i] = 1.0
            S.etd_dep_b[i] = Dgap
        end
        # Shared state-n inputs of the step-mean habit partition (per-species ones are
        # stashed in the assembly loop below).
        S.hab_igr[i] = igr
        S.hab_maxsui[i] = maxsui
        S.hab_dv[i] = dv

        # ── The three species: effective moments, then rates ──
        # `fa` is the RATE-SIDE anchor share (TeX §Reconciliation of the condensate
        # partition, the read-side completion): what every process rate sees is the
        # anchor-supported part of the population, one shared factor so the per-particle
        # state is untouched. Exactly 1.0 — these twelve lines bitwise their pre-Stage-C
        # selves — wherever the partition is admissible. Sinks computed on the shared
        # population and applied to the raw slots under-deplete relative to the raw mass,
        # so every reservoir bound tightens, never loosens. The melting level is the
        # channel this closes: melt on a phantom-laden q converts anchor-absent ice into
        # anchor-real rain with real L_f, which no reader-side cap downstream could undo.
        fa = anchor_rates ? S.anchor_f[i] : 1.0
        q1 = fa * max(S.i1q[i], 0.0) / rhoair; n1 = fa * max(S.i1n[i], 0.0) / rhoair
        a1 = fa * max(S.i1a[i], 0.0) / rhoair; c1 = fa * max(S.i1c[i], 0.0) / rhoair
        q2 = fa * max(S.i2q[i], 0.0) / rhoair; n2 = fa * max(S.i2n[i], 0.0) / rhoair
        a2 = fa * max(S.i2a[i], 0.0) / rhoair; c2 = fa * max(S.i2c[i], 0.0) / rhoair
        q3 = fa * max(S.i3q[i], 0.0) / rhoair; n3 = fa * max(S.i3n[i], 0.0) / rhoair
        a3 = fa * max(S.i3a[i], 0.0) / rhoair; c3 = fa * max(S.i3c[i], 0.0) / rhoair

        e1 = _ice_effective(q1, n1, a1, c1, 1)
        e2 = _ice_effective(q2, n2, a2, c2, 2)
        e3 = _ice_effective(q3, n3, a3, c3, 3)

        # ── THE POPULATION GATE ────────────────────────────────────────────────
        # A species has a POPULATION at this gridpoint only where the CARRIED number is
        # positive. Mass with no number is not a population: it is a state the transport can
        # produce (four moments on four independent spline fits do not stay mutually
        # realizable) and that nothing physical can. GROWTH WITHOUT ACTIVATION IS IMPOSSIBLE
        # — deposition needs crystals to deposit onto, riming needs crystals to rime, and a
        # fall speed is a property of particles. Ice begins by NUCLEATION, which creates
        # number and mass together, and those channels are deliberately NOT gated here (see
        # `q_nuc`/`n_nuc` below): activation is how ice is allowed to start.
        #
        # Why the CARRIED number and not `_ice_effective`'s. `_ice_effective` floors the
        # incoming number at `ISHMAEL_QNSMALL` and hands the result to `ishmael_var_check`,
        # whose small/large-ice limits then RE-DERIVE a number from the mass. At carried
        # n = 0 that manufactures a phantom population — measured on the O01 ice arm: 744 /m³
        # of 1 mm crystals at ρ_b = 50 kg/m³, the largest and lowest-density particle the
        # scheme can represent, hence the FASTEST. Its fall speed is 5.27 m/s against
        # 0.046 m/s for a physically consistent 20 µm population carrying the same mass
        # (115x; 1646x at 5 µm), and it is INDEPENDENT of the mass, so the sedimentation flux
        # `ρ_i·V` is linear in a density the number does not support. Measured consequence:
        # 42-48 gridpoints per column falling at up to the 25 m/s cap, a fall-speed field
        # that jumps 0 -> 5 -> 25 m/s between adjacent gridpoints, a spline fit of `ρ_i·V`
        # that rings at an amplitude far above the ice present, and — with `ahyp` keeping
        # only the positive lobes — a column ice mass that went 4.9e-8 -> 4.6e-5 -> 32.4
        # kg/m² in three steps at t = 999 s while the cumulative deposition at the worst
        # point was -3.5e-12 kg/m³. The mass was not deposited and did not arrive; it was
        # manufactured by rates and speeds read off a population that was not there.
        #
        # The floor itself stays inside `_ice_effective` — `var_check` divides by the number
        # and would fault without it — but NO RATE may see the population it implies, so the
        # gate is applied here, at the rate boundary, and `_ice_empty_rates()` is every field
        # EXACT `0.0`. That is what keeps the warm/ice-free inertness gate bitwise: a
        # configuration with no ice takes the same branch it always did.
        live1 = q1 > ISHMAEL_QSMALL && n1 > 0.0
        live2 = q2 > ISHMAEL_QSMALL && n2 > 0.0
        live3 = q3 > ISHMAEL_QSMALL && n3 > 0.0

        # ── PHASE A: every species' collection kernels and unit-realization growth ────────
        # Nothing here depends on the realization, and everything the DONOR conductances need
        # comes out of it. It has to run for all three species before any factor exists,
        # because a donor's conductance is a sum over the species drawing on it.
        p1 = live1 ?
             _ice_species_pre(tab, dt, e1, q1, n1, temp, rhoair, mu, dv, kt, nsch, npr,
                              xxlv, xxlf, qs0, drive_i, Qsi, maxsui, igr,
                              qc, nc, qr, nr, qv, shed) : _ice_empty_pre()
        p2 = live2 ?
             _ice_species_pre(tab, dt, e2, q2, n2, temp, rhoair, mu, dv, kt, nsch, npr,
                              xxlv, xxlf, qs0, drive_i, Qsi, maxsui, igr,
                              qc, nc, qr, nr, qv, shed) : _ice_empty_pre()
        p3 = live3 ?
             _ice_species_pre(tab, dt, e3, q3, n3, temp, rhoair, mu, dv, kt, nsch, npr,
                              xxlv, xxlf, qs0, drive_i, Qsi, maxsui, igr,
                              qc, nc, qr, nr, qv, shed) : _ice_empty_pre()

        # ── PHASE B: the three liquid donors' conductances, and their one factor each ──────
        # `hf` and `bg` are needed here, so they move ahead of the species loop's remains.
        hf = _ice_homogeneous_rates(temp, qc, nc, qr, nr, tau_hf)
        # Bigg, behind the RAIN POPULATION GATE (see `live_r` above): with no carried drops
        # there is no rain DSD to freeze, and the exact zeros keep it out of `f_qr`/`f_nr`
        # and out of `q_nuc`/`n_nuc` alike.
        bg = live_r ? ishmael_bigg_freezing(temp, qr, nr, dt; reservoir_caps = false) :
                      (mbiggr = 0.0, nbiggr = 0.0)
        # The rain's EVAPORATION conductance, staged beside the condensation closure where
        # `Q̇_r` and `ρ_r` are both known (Stage 2b; `kappa_ev` in the scratch doc). It is the
        # third sink of the rain reservoir and it is not ice-gated — this loop merely COMPLETES
        # the rain donor's total where ice legs also draw. Zero, hence inert, wherever the rain
        # is not evaporating or `options[:rain_evap_realization]` is off.
        kev = S.kappa_ev[i]
        (f_qc, f_qr, f_nr) = _ice_donor_factors(hf, bg, p1, p2, p3, qc, qr, nr, kev, dt)
        # The rain donor's factor now covers every sink on the reservoir, so it — not the
        # warm-path `J₀(κ_evΔt)` this overwrites — is what the fold below the ice block puts
        # into `invtau_r`/`Qdot_r`, and hence into λ, into `N` and into the rain transfer.
        # Written for EVERY gridpoint of the column, exactly as `f_ice<k>` is: the loop has no
        # ice gate, and a point with no ice contributes no ice legs, so the value written
        # there is bitwise the warm one it replaces.
        #
        # GATED on the stage's own switch, and this is the whole of what the switch does to
        # the fold. `rain_evap_real = false` leaves the column at the exact `1.0` the
        # condensation block wrote, so the rain CONDUCTANCE is unrealized exactly as it was
        # before Stage 2b, while `kev = 0` independently takes `f_qr`/`f_nr` back to their
        # ice-leg-only values for the legs that always carried them. The two together are the
        # bitwise restoration; realizing `τ_r^{-1}` at the ice legs' factor without the
        # evaporation in it would be neither the old behaviour nor the new construction.
        rain_evap_real && (S.f_rain[i] = f_qr)
        # The realized EVAPORATION draw, in the same mixing-ratio units and with the same
        # factor the ice legs carry. Formed here because the census below is the one place the
        # ice half and the evaporation half of one reservoir's draw can be added together.
        ev_real = f_qr * (kev * qr)

        # ── PHASE C: the growth at the realized collection, and everything downstream ─────
        r1 = live1 ?
             _ice_species_post(tab, dt, e1, q1, n1, temp, rhoair, mu, dv, kt,
                               xxlv, xxlf, qs0, qc, nc, qr, nr, qv, p1, f_qc, f_qr,
                               shed) :
             _ice_empty_rates()
        r2 = live2 ?
             _ice_species_post(tab, dt, e2, q2, n2, temp, rhoair, mu, dv, kt,
                               xxlv, xxlf, qs0, qc, nc, qr, nr, qv, p2, f_qc, f_qr,
                               shed) :
             _ice_empty_rates()
        r3 = live3 ?
             _ice_species_post(tab, dt, e3, q3, n3, temp, rhoair, mu, dv, kt,
                               xxlv, xxlf, qs0, qc, nc, qr, nr, qv, p3, f_qc, f_qr,
                               shed) :
             _ice_empty_rates()

        # ── Aggregation (T ≤ T_0 only, and only if some species has ice to aggregate) ──
        qagg1 = 0.0; qagg2 = 0.0; qagg3 = 0.0
        nagg1 = 0.0; nagg2 = 0.0; nagg3 = 0.0
        dnew3 = 0.0
        # The kernel's geometry arguments, hoisted: they are formed once, at state n, and
        # read by BOTH aggregation passes (the second one is below the donor factors).
        dn1 = 0.0; dn2 = 0.0; dn3 = 0.0; phi1 = 0.0; phi2 = 0.0
        # Aggregation collects crystals with crystals, so it too runs only on species that
        # HAVE a population: the gated masses and numbers below are exactly zero for a
        # species whose carried number is not positive, and a collection kernel with no
        # collector and no collectee returns nothing.
        agg_on = (temp <= T_0) && (live1 || live2 || live3)
        # PASS ONE, at unit factors. Its only consumer is `sink1`/`sink2` below: this is the
        # draw aggregation WOULD take, which is what the conductance has to be formed on.
        if agg_on
            dn1 = clamp(2.0 * ((e1.ai^2) / (e1.ci * e1.ni))^0.333333333333, 1.0e-6, 1.0e-2)
            dn2 = clamp(2.0 * ((e2.ci^2) / (e2.ai * e2.ni))^0.333333333333, 1.0e-6, 1.0e-2)
            dn3 = clamp(2.0 * ((e3.ai^2) / (e3.ci * e3.ni))^0.333333333333, 1.0e-6, 1.0e-2)
            phi1 = clamp(e1.ci / e1.ai * gamma(ISHMAEL_NU - 1.0 + e1.deltastr) *
                         ISHMAEL_I_GAMMNU, 0.01, 100.0)
            phi2 = clamp(e2.ci / e2.ai * gamma(ISHMAEL_NU - 1.0 + e2.deltastr) *
                         ISHMAEL_I_GAMMNU, 0.01, 100.0)
            ag1 = ishmael_aggregation(dt, rhoair, temp,
                                      live1 ? q1 : 0.0, live1 ? e1.ni : 0.0, dn1,
                                      live2 ? q2 : 0.0, live2 ? e2.ni : 0.0, dn2,
                                      live3 ? q3 : 0.0, live3 ? e3.ni : 0.0, dn3,
                                      e1.rhobar, e2.rhobar, phi1, phi2,
                                      tab.coltab, tab.coltabn)
            qagg1 = ag1.qagg1; qagg2 = ag1.qagg2; qagg3 = ag1.qagg3
            nagg1 = ag1.nagg1; nagg2 = ag1.nagg2; nagg3 = ag1.nagg3
            dnew3 = ag1.dnew3
        end

        # The activation deficit reads the CARRIED ice number, not the effective one: the
        # ceiling exists to stop nucleation once the ice number is already at the DeMott
        # concentration, and the number it has to compare against is the number being
        # transported. (In the Fortran the two are the same array, because `var_check` writes
        # back.) Species 3 is excluded, as the Fortran excludes it.
        dm = _ice_demott_rates(temp, sup, rhoair, (n1 + n2) * rhoair, tau_act)

        # WHICH species receives it: the inherent growth ratio decides the habit, so
        # `igr ≤ 1` (plate-like growth, which the -20 C override forces below that
        # temperature) nucleates into species 1 (planar) and `igr > 1` into species 2
        # (columnar). Aggregates are never nucleated into. Fortran lines 2153-2196.
        # ── The ICE donors' realization factors ────────────────────────────────────────────
        #
        # Convicted by the census this construction was built to run: `q_i1` reached a realized
        # depletion of 5.41 reservoirs per step from melting and aggregation and a further 2.63
        # from SUBLIMATION, `q_i2` 1.29. The same share rule applies, over each species' TOTAL
        # mass sink:
        #
        #   * MELTING, `−q̇_mlt`, an unbounded rate like every other conversion here;
        #   * AGGREGATION's transfer of species 1/2 into 3, already a per-step increment;
        #   * SUBLIMATION, the negative half of the deposition channel — and the one ice sink
        #     that lived outside every conductance in this file. The exponential propagator
        #     bounds `Q_ss`, not `ρ_i`, so the step-mean drive is limited by the vapor DEFICIT
        #     and nothing limited it by the ice present. Measured: 2.63 reservoirs per step.
        #
        # The conductance is formed here, at state `n` (`r_k.Qdot` is the instantaneous
        # deposition rate, the frozen-coefficient convention every other factor uses), and the
        # factor is stashed in `f_ice<k>` because its two halves are applied in two places: the
        # melting legs below, and the STEP-MEAN sublimation in the ETD pre-compute, which does
        # not exist until after this function returns. One factor per donor, two application
        # sites, so the ice a species loses to melting and the ice it loses to sublimation are
        # shares of one exponential depletion rather than two independent ones.
        #
        # AGGREGATION is counted in the conductance AND realized at the factor, in the SECOND
        # pass through `ishmael_aggregation` just below. This block used to argue that `qagg3`
        # is not decomposable into its donors, so scaling species 1 and 2 by different factors
        # would break the closure aggregation is built on. That was wrong twice over: the
        # routine's own recipient gain IS the sum of the two donors' losses, so re-forming it
        # HERE as `−(qagg1 + qagg2)` closes the exchange at whatever the donors realized; and
        # the claim that aggregation was bounded on its own did not survive the measurement
        # (`q_i1(melt+agg)` at 1.33 reservoirs per step on the Stage-C quick run, 814
        # gridpoint-steps past 1). Over-depleting species 1/2 while the `:bhyp` control
        # variable recovers the slot at −μ is mass CREATED in species 3.
        sink1 = (max(-r1.qmlt, 0.0) + max(-qagg1, 0.0) * i_dt) + max(-r1.Qdot, 0.0) / rhoair
        sink2 = (max(-r2.qmlt, 0.0) + max(-qagg2, 0.0) * i_dt) + max(-r2.Qdot, 0.0) / rhoair
        sink3 = (max(-r3.qmlt, 0.0) + max(-qagg3, 0.0) * i_dt) + max(-r3.Qdot, 0.0) / rhoair
        f_i1 = relaxation_realization(sink1, q1, dt)
        f_i2 = relaxation_realization(sink2, q2, dt)
        f_i3 = relaxation_realization(sink3, q3, dt)
        S.f_ice1[i] = f_i1; S.f_ice2[i] = f_i2; S.f_ice3[i] = f_i3
        FIC = (f_i1, f_i2, f_i3)

        # ── The ICE NUMBER donors: one conductance per species' NUMBER ─────────────────────
        #
        # A species' number is a SEPARATE RESERVOIR from its mass, and the three legs that
        # draw on it are not proportional to the three that draw on the mass:
        #
        #   * AGGREGATION's number transfer comes from `coltabn`, the mass transfer from
        #     `coltab` — two offline tables integrating two different MOMENTS of the same
        #     collision kernel (`mkcoltb`, ishmael_tables.jl). A pair's `(colamt, deltan)` is
        #     one collision count, which is what the single-factor hook was argued from, but
        #     the FRACTION OF THE RESERVOIR each moment loses is not the same number.
        #     Measured at unit factors on the stiff-cold two-habit fixture (ρ_i = 1e-3,
        #     n_i = 1e6, T = 259.15 K, Δt = 1 s): κ_q = 1.77e-4 /s against κ_n = 8.21e-5 /s,
        #     κ_n/κ_q = 0.464; 0.377 at n_i = 1e9; 0.201 on an anvil-like 1e-4 kg/m³,
        #     5e7 /m³, 240 K state. Here the number kernel is the SLOWER one, so the mass
        #     factor OVER-realizes the number transfer by up to 5x — and nothing in the
        #     tables promises it stays on that side of 1 elsewhere.
        #   * MELTING's number leg is `nmlt = q̇_mlt·(n/q) − dNmltri`. The first term is the
        #     mass leg's exact number partner; `dNmltri` — the drops the crystal collected
        #     and is now melting off — is a number sink with NO mass partner in this
        #     reservoir at all, so κ_{n,melt} > κ_{q,melt} strictly.
        #   * SUBLIMATION is the one leg where the two conductances agree exactly: the number
        #     leaves in proportion to the mass, `ṅ = q̇·(n/q)` (`niq`, Fortran lines
        #     1273-1276), so its contribution to κ_n is its contribution to κ_q identically.
        #
        # The construction is the rain's, exactly: the rain MASS and the rain NUMBER are
        # already two donors with two factors (`f_qr`/`f_nr`) precisely because Bigg's number
        # conductance is a twentieth of its mass conductance, and realizing the number on the
        # mass factor there would freeze all of the rain's mass while removing a twentieth of
        # its drops. This is the ice mirror of that argument, and it is the same failure mode
        # in the other direction: MASS WITHOUT NUMBER, which is the state the population gate
        # exists to forbid and which the population reconciliation then has to re-seed.
        #
        # The reservoir is the CARRIED number `n_k`, not `e_k.ni` — the budget/kernel split
        # of `_ice_species_pre`: a kernel rate is meaningful only on a realizable population,
        # but "how much of the number leaves" is a statement about the number that is there.
        # Aggregation's number rate is formed on the effective number and applied to the
        # carried slot, so the carried slot is what has to bound it.
        #
        # GATED, default OFF (`options[:ice_number_realization]`, env `SCYTHE_O01_ICENREAL=1`):
        # with the switch off every factor here is set to that species' MASS factor, which is
        # what each of these legs already carried, so the whole block is bitwise inert and the
        # census below measures the state as it stands.
        nsub1 = (max(-r1.Qdot, 0.0) / rhoair) * r1.niq
        nsub2 = (max(-r2.Qdot, 0.0) / rhoair) * r2.niq
        nsub3 = (max(-r3.Qdot, 0.0) / rhoair) * r3.niq
        nsink1 = (max(-nagg1, 0.0) * i_dt + max(-r1.nmlt, 0.0)) + nsub1
        nsink2 = (max(-nagg2, 0.0) * i_dt + max(-r2.nmlt, 0.0)) + nsub2
        nsink3 = (max(-nagg3, 0.0) * i_dt + max(-r3.nmlt, 0.0)) + nsub3
        f_n1 = ice_n_real ? relaxation_realization(nsink1, n1, dt) : f_i1
        f_n2 = ice_n_real ? relaxation_realization(nsink2, n2, dt) : f_i2
        f_n3 = ice_n_real ? relaxation_realization(nsink3, n3, dt) : f_i3

        # ── AGGREGATION, PASS TWO: the REALIZED collection ────────────────────────────────
        # The mirror of the riming construction. Pass one ran at unit factors and existed
        # only to put aggregation's draw into `sink1`/`sink2`; the factors are formed now, so
        # the collection is re-integrated with each donor species' three pairs scaled by ITS
        # factor, at the hook the Fortran's `ratioagg` occupied. Nothing downstream changes
        # text — the census, the a/c moment re-diagnosis and the slot assembly read these
        # same names — and the chain is not circular: pass one → `f_ik` → pass two → census
        # → assembly. `qagg3` is formed HERE as the exact negative of what the two donors
        # lost, rather than taken from the routine's own six-term sum (which closes to
        # round-off, not exactly), so the cross-species exchange closes on the realized
        # numbers. The NUMBER is scaled inside the routine at the NUMBER donors' factors
        # (`f_aggn<k>`), which are the mass ones exactly when `ice_number_realization` is off.
        if agg_on
            ag2 = ishmael_aggregation(dt, rhoair, temp,
                                      live1 ? q1 : 0.0, live1 ? e1.ni : 0.0, dn1,
                                      live2 ? q2 : 0.0, live2 ? e2.ni : 0.0, dn2,
                                      live3 ? q3 : 0.0, live3 ? e3.ni : 0.0, dn3,
                                      e1.rhobar, e2.rhobar, phi1, phi2,
                                      tab.coltab, tab.coltabn;
                                      f_agg1 = f_i1, f_agg2 = f_i2,
                                      f_aggn1 = f_n1, f_aggn2 = f_n2,
                                      # Species 3's own number factor reaches the kernel
                                      # ONLY through aggregate self-collection, which is the
                                      # one pair that carried no factor at all before this
                                      # stage — hence the explicit `1.0` with the switch off
                                      # rather than `f_n3`, which is `f_i3` there and would
                                      # not be the behaviour being reproduced.
                                      f_aggn3 = ice_n_real ? f_n3 : 1.0,
                                      reservoir_caps = agg_caps)
            qagg1 = ag2.qagg1; qagg2 = ag2.qagg2
            qagg3 = -(qagg1 + qagg2)
            nagg1 = ag2.nagg1; nagg2 = ag2.nagg2; nagg3 = ag2.nagg3
            dnew3 = ag2.dnew3
        end
        # Is the ice channel a SINK of ice at state n? Frozen here, like every other
        # classification the step freezes, and it is what decides whether the donor factor
        # belongs in the conductance at all: `f_ice<k>` bounds an ice sink, and deposition is
        # a source. Above `T_0` the channel is shut and the question does not arise.
        sub_now = (temp <= T_0) && (drive_i < 0.0)
        # The realized melting legs. ONE name per leg, used by the ice slot that loses the mass
        # AND by the rain slot that gains it, so the melt exchange closes exactly as the freeze
        # exchange does.
        MLQ = (f_i1 * r1.qmlt, f_i2 * r2.qmlt, f_i3 * r3.qmlt)
        # The melt NUMBER rides the NUMBER donor's factor, not the mass one — the two are
        # different reservoirs and `nmlt` carries `dNmltri`, a number sink with no mass
        # partner (see the number-donor block above). `f_n<k> = f_i<k>` with the switch off,
        # so this line is bitwise what it was. The rain gains exactly `−MLN[k]` drops through
        # `ice_nr` below, as it always did: the exchange still closes on ONE name per leg,
        # and it is the number of drops the melt actually produced. Mass conservation is
        # untouched either way — the number carries none.
        MLN = (f_n1 * r1.nmlt, f_n2 * r2.nmlt, f_n3 * r3.nmlt)
        # The two AXIS moments stay on the MASS factor. `amlt`/`cmlt` are volume moments of
        # the melting crystal and their number part (`a_i·nmlt/n_i`) is inseparable from
        # their mass part inside `ishmael_melting`; the moment set's consistency is repaired
        # by the `var_check` source rather than by splitting this rate.
        MLA = (f_i1 * r1.amlt, f_i2 * r2.amlt, f_i3 * r3.amlt)
        MLC = (f_i1 * r1.cmlt, f_i2 * r2.cmlt, f_i3 * r3.cmlt)
        # ── What the SUBLIMATION number sink still has to apply, at its own site ───────────
        # The habit partition's number sink lives in the ETD pre-compute, where the step-mean
        # deposition rate exists, and it is formed there as `q̄_k·(n_k/q_k)` — so it already
        # carries whatever factor `invtau_i<k>` carries, which is `f_i<k>` where `sub_now`
        # classified the channel as a sink and nothing where it did not. What is written here
        # is the RESIDUAL: multiply by this and the leg is realized at `f_n<k>` exactly, one
        # number for one reservoir. An exact `1.0` with the switch off, so that site is
        # bitwise. The guard is for the overflow corner of `J₀`: `relaxation_realization`
        # returns a true `0.0` once `κΔt` overflows to `Inf`, and there the leg it would
        # divide into is an exact zero anyway.
        S.f_isn1[i] = !ice_n_real ? 1.0 :
                      (sub_now ? (f_i1 > 0.0 ? f_n1 / f_i1 : 1.0) : f_n1)
        S.f_isn2[i] = !ice_n_real ? 1.0 :
                      (sub_now ? (f_i2 > 0.0 ? f_n2 / f_i2 : 1.0) : f_n2)
        S.f_isn3[i] = !ice_n_real ? 1.0 :
                      (sub_now ? (f_i3 > 0.0 ? f_n3 / f_i3 : 1.0) : f_n3)

        target = igr <= 1.0 ? 1 : 2
        # ── ONE realization factor per DONOR RESERVOIR, over ALL of that donor's legs ───────
        #
        # The Bigg pass realized the two nucleation-channel freezing legs; the measured wall
        # then moved to riming (cured inside `ishmael_riming_growth`, where the axis partners
        # are formed) and then to ICE-RAIN COLLECTION: `κΔt = 118` on `dQRfzri` at
        # t = 2101.5 s against `q_r = 1.7e-4 kg/kg`. Realizing legs one at a time is
        # whack-a-mole, and it also lets INDEPENDENTLY realized legs sum past the reservoir
        # (measured: `kr` sitting at 1.03 where the freezing factor had already saturated).
        # So the conductance is formed over the donor's WHOLE sink, which is what the ODE
        # `dq/dt = −κ_total q` actually says, and every leg drawing on that donor is
        # multiplied by the one factor. Riming is the exception and stays inside its own
        # routine: its mass rate is not known until after the growth integration that its
        # `ardr`/`crdr` partners come out of, and scaling it here would desynchronize them.
        # The step-mean realizations. ONE name per leg, formed once here and used by BOTH
        # sides of the exchange below, so the mass the ice gains and the mass the liquid loses
        # cannot come apart. `dQIfzri` is the ICE that moved species because it collected
        # rain — the same collision events — so it rides the rain-mass factor with its
        # partners rather than carrying one of its own.
        #
        # ── THE MINIMUM-CRYSTAL BOUND ON EVERY REALIZED ICE-NUMBER SOURCE ────────────────
        #
        # [`_ice_homogeneous_rates`](@ref) states the rule: every ice NUMBER source must seed
        # crystals at or above the smallest size the scheme resolves, so a number rate is
        # `min(n/τ, ṁ/m_min)` — the mass transfers in full and the number is what that mass
        # supports at [`ISHMAEL_M_MIN`](@ref). It is the SOURCE-side twin of the POPULATION
        # GATE above ("no rate may act on a population the CARRIED number does not support"),
        # and it is stated there at the RATE. The rate is not where it survives.
        #
        # A number leg and its mass partner are realized at DIFFERENT donor factors wherever
        # they draw on different reservoirs, and `min` does not commute with two of them:
        # `f·min(a, b) = min(f·a, f·b)` for ONE shared `f ≥ 0`, but `min(f_n·a, f_q·b)` is
        # bounded by neither when `f_n ≫ f_q`. That is exactly the anvil-top pathology
        # (reference/FINDINGS_ISHMAEL_S8S9.md §5i), four stacked failures in one term:
        # the rain slots decorrelate at 15 km — `n_r` rings to EXACTLY zero while `q_r` stays
        # ~1e-8 kg/kg at 192–203 K; `ishmael_bigg_freezing` gates on `q_r` alone, so
        # `ishmael_rain_lambda` floors the number at `QNSMALL` and lands in the
        # `lamr < LAMMINR` clamp — a PHANTOM DSD of 2800 μm drops, the largest the scheme
        # admits, where there is no rain at all; Bigg is then evaluated 30–40 K outside its
        # validity, `exp(0.66ΔT) ~ 1e21`; and the number leg is realized at
        # `f_nr = relaxation_realization(rate, 0.0, dt) = 1.0` — the zero-reservoir guard,
        # which is correct for a factor and catastrophic for a bound — while `f_qr` clamps
        # the mass leg to 1.8e-19…2.3e-17. The SAME collisions, realized seventeen to
        # nineteen orders apart. What reached species 1 was ~1e15 m⁻³ s⁻¹ of crystals
        # carrying 1e-6·`M_MIN` each (`n₁` at 1.5e16 m⁻³ against `q₁ = 0` exactly, immortal
        # because every sink is live-gated), and the ringing of that spike through the number
        # moment's spline fit is the mass-without-number lobe that made 70% of the quick
        # arm's ice cloud — and 99% of the rejected full-mode run's — dead mass.
        #
        # So the bound is restated HERE, on the REALIZED pair, where both factors exist:
        #
        #     n_leg ≤ m_leg / ISHMAEL_M_MIN
        #
        # Δt-FREE (a relation between two simultaneous rates, not a depletion cap — refining
        # the step still approaches the same differential equation), one-sided, and EXACTLY
        # INERT wherever the number source already respects its mass partner, which is
        # everywhere the two factors agree. It carries NO off-switch: like the population
        # gate it is an invariant of the representation, not a parameterization choice.
        #
        # Leg by leg:
        #   * `nim` (homogeneous CLOUD freezing) is bounded at the rate and realized at the
        #     SAME `f_qc` as `mim`, so the `min` cannot bind beyond rounding. Written anyway,
        #     so that the invariant does not rest on the two factors staying equal.
        #   * `nimr` (homogeneous RAIN freezing) carries `f_nr` against `f_qr`: two factors,
        #     so the rate-level `min(n_r/τ_hf, ṁ/m_min)` does NOT survive realization, and
        #     this is where it is restored. §5i measured this leg contributing exactly 0 at
        #     the pathological points — the rate-level bound doing its job at `n_r = 0`, the
        #     bound Bigg lacks — and this keeps that true after the factors.
        #   * `nbig` (Bigg) is the measured pump: no rate-level bound of any kind.
        #   * `FzN` (ice–rain collection) is the drops that froze onto a crystal and moved it
        #     to another species. The mass those `FzN[k]` particles carry to the destination
        #     is `RfzQ[k]`, the RAIN mass frozen (new ice), PLUS `IfzQ[k]`, that species' own
        #     ice — a species TRANSFER rather than new mass, but mass moving with the same
        #     particles, hence part of what they weigh once they arrive. Both ride `f_qr`, so
        #     their sum is the crystal mass the number must be supported by and it is what
        #     the bound divides. This leg was never the pathology: the kernel self-gates at
        #     `n_r = 0` (every `procr` rate carries a factor `nr`). It is bounded for the same
        #     reason the homogeneous ones are — an invariant may not rest on a kernel's
        #     internals.
        #   * `dm.nnuccd`/`dm.mnuccd` (DeMott) and `r_k.nmult`/`r_k.qmult` (Hallett-Mossop)
        #     need nothing. `mnuccd ≡ nnuccd·ISHMAEL_M_MIN` by construction and the splinter
        #     pair is 350 crystals per mg at the 5 μm mass; decisively, NEITHER pair is
        #     multiplied by a realization factor at all — both enter `q_nuc`/`n_nuc` bare —
        #     so there is no second factor to break the ratio.
        mim  = f_qc * hf.mim
        nim  = min(f_qc * hf.nim, mim / ISHMAEL_M_MIN)
        mimr = f_qr * hf.mimr
        nimr = min(f_nr * hf.nimr, mimr / ISHMAEL_M_MIN)
        mbig = f_qr * bg.mbiggr
        nbig = min(f_nr * bg.nbiggr, mbig / ISHMAEL_M_MIN)
        RfzQ = (f_qr * r1.dQRfzri, f_qr * r2.dQRfzri, f_qr * r3.dQRfzri)
        IfzQ = (f_qr * r1.dQIfzri, f_qr * r2.dQIfzri, f_qr * r3.dQIfzri)
        FzN  = (min(f_nr * r1.dNfzri, (RfzQ[1] + IfzQ[1]) / ISHMAEL_M_MIN),
                min(f_nr * r2.dNfzri, (RfzQ[2] + IfzQ[2]) / ISHMAEL_M_MIN),
                min(f_nr * r3.dNfzri, (RfzQ[3] + IfzQ[3]) / ISHMAEL_M_MIN))
        NRN  = (f_nr * r1.nrn_loss, f_nr * r2.nrn_loss, f_nr * r3.nrn_loss)
        qIfz_in = target == 1 ? (IfzQ[2] + IfzQ[3]) : (IfzQ[1] + IfzQ[3])
        nIfz_in = target == 1 ? (FzN[2] + FzN[3]) : (FzN[1] + FzN[3])
        q_nuc = dm.mnuccd + mim + mimr + mbig + qIfz_in +
                (r1.qmult + r2.qmult + r3.qmult) +
                ((RfzQ[1] + RfzQ[2]) + RfzQ[3])
        n_nuc = dm.nnuccd + nim + nimr + nbig + nIfz_in +
                (r1.nmult + r2.nmult + r3.nmult)

        # ── The liquid back-reactions, and Q̇_freeze as their exact negative ──
        ice_c = -rhoair * (mim + ((r1.prdr_pre * r1.qcrimefrac) +
                                  (r2.prdr_pre * r2.qcrimefrac) +
                                  (r3.prdr_pre * r3.qcrimefrac)))
        ice_r = rhoair * ((-mimr - mbig) +
                          (-(r1.prdr_pre * (1.0 - r1.qcrimefrac)) - MLQ[1] - RfzQ[1]) +
                          (-(r2.prdr_pre * (1.0 - r2.qcrimefrac)) - MLQ[2] - RfzQ[2]) +
                          (-(r3.prdr_pre * (1.0 - r3.qcrimefrac)) - MLQ[3] - RfzQ[3]))
        ice_nr = rhoair * ((-nimr - nbig) +
                           (-NRN[1] - MLN[1] - FzN[1]) +
                           (-NRN[2] - MLN[2] - FzN[2]) +
                           (-NRN[3] - MLN[3] - FzN[3]))
        S.ICE_C[i] = ice_c
        S.ICE_R[i] = ice_r
        S.ICE_NR[i] = ice_nr
        S.FRZ_NET[i] = -(ice_c + ice_r)

        # ── The DONOR-DEPLETION census, on the REALIZED rates the step is about to apply ──
        # `ice_c`/`ice_r`/`ice_nr` are exactly those sums with the sign the liquid sees, so the
        # census reads the applied numbers rather than a reconstruction of them. The ice
        # donors are censused on their own sinks (melting out, and aggregation's transfer of
        # species 1/2 into 3, which is already an increment). Reports; limits nothing.
        if census_on
            mc_donor_census!(stx, tid_x, MC_DONOR_QC, -ice_c / rhoair, qc,
                             MC_DONOR_QFLOOR, dt)
            # STAGE 2b: the COMBINED realized draw on the rain — the ice legs (`ice_r`)
            # PLUS the realized evaporation. That sum, not either half, is what
            # `1 − e^{−κ_totΔt} < 1` bounds, so censusing it is what makes the block's
            # invariant ("no realized donor may exceed one reservoir per step") the thing
            # actually measured. Before Stage 2b this row saw the ice half alone and read a
            # dutiful 1.0 while the two halves together took 1.0012 reservoirs.
            mc_donor_census!(stx, tid_x, MC_DONOR_QR, (-ice_r / rhoair) + ev_real, qr,
                             MC_DONOR_QFLOOR, dt)
            mc_donor_census!(stx, tid_x, MC_DONOR_NR, -ice_nr / rhoair, nr,
                             MC_DONOR_NFLOOR, dt)
            mc_donor_census!(stx, tid_x, MC_DONOR_I1, max(-MLQ[1], 0.0) - qagg1 * i_dt, q1,
                             MC_DONOR_QFLOOR, dt)
            mc_donor_census!(stx, tid_x, MC_DONOR_I2, max(-MLQ[2], 0.0) - qagg2 * i_dt, q2,
                             MC_DONOR_QFLOOR, dt)
            mc_donor_census!(stx, tid_x, MC_DONOR_I3, max(-MLQ[3], 0.0) - qagg3 * i_dt, q3,
                             MC_DONOR_QFLOOR, dt)
            # ── The three ICE NUMBER reservoirs (Stage 1b) ────────────────────────────────
            # The same three legs on the other moment, at the numbers the step applies:
            # aggregation's REALIZED number transfer (`nagg<k>` is pass two's, and already a
            # per-step increment, so it is divided by the reservoir directly while the two
            # rates carry a `Δt`), the realized melt number `MLN`, and the sublimation number
            # sink at whatever factor it will be applied with — `f_n<k>` under the switch,
            # and what `invtau_i<k>` alone carries without it.
            #
            # The sublimation term is the STATE-n instantaneous rate, not the step-mean one
            # the ETD site applies; that is the same approximation the ice MASS conductance
            # is formed on (`sink<k>` above), and its own step-mean reading is the
            # `MC_DONOR_S*` block. Reported, never limiting, in both modes.
            nsr1 = (ice_n_real ? f_n1 : (sub_now ? f_i1 : 1.0)) * nsub1
            nsr2 = (ice_n_real ? f_n2 : (sub_now ? f_i2 : 1.0)) * nsub2
            nsr3 = (ice_n_real ? f_n3 : (sub_now ? f_i3 : 1.0)) * nsub3
            mc_donor_census!(stx, tid_x, MC_DONOR_N1,
                             (max(-MLN[1], 0.0) + max(-nagg1, 0.0) * i_dt) + nsr1, n1,
                             MC_DONOR_NFLOOR, dt)
            mc_donor_census!(stx, tid_x, MC_DONOR_N2,
                             (max(-MLN[2], 0.0) + max(-nagg2, 0.0) * i_dt) + nsr2, n2,
                             MC_DONOR_NFLOOR, dt)
            mc_donor_census!(stx, tid_x, MC_DONOR_N3,
                             (max(-MLN[3], 0.0) + max(-nagg3, 0.0) * i_dt) + nsr3, n3,
                             MC_DONOR_NFLOOR, dt)
        end

        # ── The ATTRIBUTION CENSUS (opt-in; see `MC_ATTR_QR_HOM`) ─────────────────────────
        # Blocks A, C and D; block B is at the ETD pre-compute, where the APPLIED step-mean
        # evaporation exists. Nothing below writes anything but `stx`.
        if attr_on
            # ── BLOCK A: the six legs of the q_r debit, AT THE BREACH POINTS ONLY ─────────
            # The gate is the `MC_DONOR_QR` number itself, re-formed from the same `ice_r`
            # the census above read, so "breach point" means exactly what that block means
            # by it and the counts below partition its count.
            if qr > MC_DONOR_QFLOOR
                iqr = dt / qr
                # Formed EXPRESSION FOR EXPRESSION as `mc_donor_census!` forms it
                # (`realized * dt / reservoir`), so "breach point" is the same set of points
                # to the last bit and the counts below cannot drift off its count. Since
                # Stage 2b that expression is the COMBINED draw, so the gate carries the
                # realized evaporation too; the six legs decomposed below stay the ice ones
                # (evaporation has block B to itself, at the APPLIED step-mean), and with the
                # evaporation now inside `κ_tot` the gate is expected never to open at all.
                if ((-ice_r / rhoair) + ev_real) * dt / qr > 1.0
                    # The four TRUE debits, in the order `ice_r` accumulates them, plus the
                    # rime EXCESS (a sub-part of the third) and the melt CREDIT.
                    rime_r = ((r1.prdr_pre * (1.0 - r1.qcrimefrac)) +
                              (r2.prdr_pre * (1.0 - r2.qcrimefrac))) +
                             (r3.prdr_pre * (1.0 - r3.qcrimefrac))
                    d_hom = mimr * iqr
                    d_big = mbig * iqr
                    d_rim = rime_r * iqr
                    # The conductance was formed on the UNIT pass's split (`prdr0_r`,
                    # `qcf0 = rimesum/rimetotal`); the debit is taken from the REALIZED
                    # pass's (`prdr_pre`, `qcrimefrac`), whose split and rime-density blend
                    # move whenever `f_qc != f_qr`. This is that difference.
                    d_rix = (rime_r - f_qr * ((p1.prdr0_r + p2.prdr0_r) + p3.prdr0_r)) * iqr
                    d_col = ((RfzQ[1] + RfzQ[2]) + RfzQ[3]) * iqr
                    d_mlt = -((MLQ[1] + MLQ[2]) + MLQ[3]) * iqr
                    # WHICH leg is the largest debit here. Strict `>`, so a tie goes to the
                    # earlier channel and exactly one channel is credited per breach point.
                    bch = MC_ATTR_QR_HOM
                    best = d_hom
                    d_big > best && (best = d_big; bch = MC_ATTR_QR_BIGG)
                    d_rim > best && (best = d_rim; bch = MC_ATTR_QR_RIME)
                    d_rix > best && (best = d_rix; bch = MC_ATTR_QR_RIMEX)
                    d_col > best && (best = d_col; bch = MC_ATTR_QR_COLL)
                    mc_attr_census!(stx, tid_x, MC_ATTR_QR_HOM, d_hom,
                                    bch == MC_ATTR_QR_HOM)
                    mc_attr_census!(stx, tid_x, MC_ATTR_QR_BIGG, d_big,
                                    bch == MC_ATTR_QR_BIGG)
                    mc_attr_census!(stx, tid_x, MC_ATTR_QR_RIME, d_rim,
                                    bch == MC_ATTR_QR_RIME)
                    mc_attr_census!(stx, tid_x, MC_ATTR_QR_RIMEX, d_rix,
                                    bch == MC_ATTR_QR_RIMEX)
                    mc_attr_census!(stx, tid_x, MC_ATTR_QR_COLL, d_col,
                                    bch == MC_ATTR_QR_COLL)
                    mc_attr_census!(stx, tid_x, MC_ATTR_QR_MELT, d_mlt, false)
                    mc_attr_census!(stx, tid_x, MC_ATTR_QR_ALL,
                                    ((d_hom + d_big) + (d_rim + d_col)), false)
                end
            end

            # ── BLOCK C: the q_i1/q_i2 split between MELTING and AGGREGATION ─────────────
            # `qagg_k` is already a per-step INCREMENT (the Fortran's own construction), so
            # it is divided by the reservoir directly while the melting RATE carries a `dt`.
            if q1 > MC_DONOR_QFLOOR
                xm1 = max(-MLQ[1], 0.0) * dt / q1
                xa1 = -qagg1 / q1
                mc_attr_census!(stx, tid_x, MC_ATTR_MLT1, xm1, xm1 > 1.0)
                mc_attr_census!(stx, tid_x, MC_ATTR_AGG1, xa1, xa1 > 1.0)
                # The caller-side proxy for `ishmael_col1`'s per-pair `min(colamt, rx)`: the
                # seven pairs are summed inside `ishmael_aggregation`, so the only thing
                # visible from here is the SUM taking the whole species mass.
                mc_attr_census!(stx, tid_x, MC_ATTR_AGG1_SAT, xa1,
                                -qagg1 >= q1 * (1.0 - 1.0e-12))
            end
            if q2 > MC_DONOR_QFLOOR
                xm2 = max(-MLQ[2], 0.0) * dt / q2
                xa2 = -qagg2 / q2
                mc_attr_census!(stx, tid_x, MC_ATTR_MLT2, xm2, xm2 > 1.0)
                mc_attr_census!(stx, tid_x, MC_ATTR_AGG2, xa2, xa2 > 1.0)
            end

            # ── BLOCK D: what runs ABOVE the melting level ────────────────────────────────
            # Riming is the only liquid->ice channel with no temperature gate, and above
            # `T_0` riming -> ice -> melt is a closed loop: both legs are recorded, plus
            # `dQImltri`, which is NOT a transfer — it is the collected liquid's sensible
            # heat inside `ishmael_melting` (ishmael.jl) — so the loop can be read as a loop.
            # `T_0 + 2` K, not `T_0`: two degrees of margin puts the channels well clear of
            # the freezing level itself, where all of this is ordinary physics.
            if temp > T_0 + 2.0
                rime_c = ((r1.prdr_pre * r1.qcrimefrac) + (r2.prdr_pre * r2.qcrimefrac)) +
                         (r3.prdr_pre * r3.qcrimefrac)
                rime_rw = ((r1.prdr_pre * (1.0 - r1.qcrimefrac)) +
                           (r2.prdr_pre * (1.0 - r2.qcrimefrac))) +
                          (r3.prdr_pre * (1.0 - r3.qcrimefrac))
                if qc > MC_DONOR_QFLOOR
                    xwc = rime_c * dt / qc
                    mc_attr_census!(stx, tid_x, MC_ATTR_WARM_RIME_C, xwc, xwc > 1.0)
                end
                if qr > MC_DONOR_QFLOOR
                    xwr = rime_rw * dt / qr
                    mc_attr_census!(stx, tid_x, MC_ATTR_WARM_RIME_R, xwr, xwr > 1.0)
                end
                q_ice = (q1 + q2) + q3
                if q_ice > MC_DONOR_QFLOOR
                    xwm = -((MLQ[1] + MLQ[2]) + MLQ[3]) * dt / q_ice
                    xwi = ((p1.rr.dQImltri + p2.rr.dQImltri) + p3.rr.dQImltri) * dt / q_ice
                    mc_attr_census!(stx, tid_x, MC_ATTR_WARM_MELT, xwm, xwm > 1.0)
                    mc_attr_census!(stx, tid_x, MC_ATTR_WARM_MLTRI, xwi, xwi > 1.0)
                end
            end

            # ── BLOCK D's TAIL: the sub-melting-level population, and what `f_a` withheld ──
            # The RAW slots, not the anchor-shared `q_k`: the question is how much ice the
            # transport actually put below the melting level, and `f_a` is the second half
            # of the answer rather than part of the first.
            if temp > T_0
                rho_ice_raw = (max(S.i1q[i], 0.0) + max(S.i2q[i], 0.0)) + max(S.i3q[i], 0.0)
                if rho_ice_raw > 0.0
                    @inbounds begin
                        stx[MC_ATTR_WARM_PTS, tid_x] += 1.0
                        stx[MC_ATTR_WARM_MASS, tid_x] += rho_ice_raw
                        if fa < 1.0
                            stx[MC_ATTR_WARM_FA, tid_x] += 1.0
                            dfa = 1.0 - fa
                            dfa > stx[MC_ATTR_WARM_DFA, tid_x] &&
                                (stx[MC_ATTR_WARM_DFA, tid_x] = dfa)
                            # Melting is linear in the population at fixed per-particle
                            # state, so `q̇_mlt/f_a` is what the RAW population would melt.
                            fa > 0.0 && (stx[MC_ATTR_WARM_HELD, tid_x] +=
                                ((1.0 / fa) - 1.0) *
                                abs((MLQ[1] + MLQ[2]) + MLQ[3]) * rhoair * dt)
                        end
                    end
                end
            end
        end

        # ── Assemble the twelve slot sources ──
        rs = (r1, r2, r3)
        es = (e1, e2, e3)
        qs = (q1, q2, q3)
        ns = (n1, n2, n3)
        as = (a1, a2, a3)
        cs = (c1, c2, c3)
        aggq = (qagg1, qagg2, qagg3)
        aggn = (nagg1, nagg2, nagg3)
        lives = (live1, live2, live3)
        for k in 1:3
            rk = rs[k]
            ek = es[k]
            # Aggregation and the nucleation transfer both redistribute mass and number at
            # (assumed) fixed shape and density, so they are combined into ONE new state and
            # re-diagnosed once. The target species' nucleation GAIN is not here: new
            # particles arrive with a size of their own (`_ice_nucleation_volume`).
            is_target = k == target
            # The REALIZED transfers (`IfzQ`/`FzN`), not the nominal ones: the species that
            # loses the ice must lose exactly what the target gains.
            dq_re = aggq[k] - (is_target ? 0.0 : IfzQ[k] * dt)
            dn_re = aggn[k] - (is_target ? 0.0 : FzN[k] * dt)
            aagg = 0.0
            cagg = 0.0
            if dq_re != 0.0 || dn_re != 0.0
                aagg, cagg = _ice_agg_moments(ek, ek.ni + dn_re, qs[k] + dq_re,
                                              rhoair, dt, dnew3, k == 3 && agg_on)
            end

            # `rk.Qdot` — the DEPOSITION mass source — is deliberately NOT here. It is one of
            # the two relaxations of the shared `Q_ss` and is withheld from the multistep with
            # its partner, returning as a direct increment at the step-mean rate
            # (`relaxation_adjustment_qss!`). And its HABIT PARTITION — the a/c-moment sources
            # and the sublimation number sink — is not here EITHER: the partition distributes
            # the realized mass increment over the axes (TeX §Departures (d)), so it is
            # evaluated once, at the SAME step-mean rate the mass slots receive, in the ETD
            # pre-compute, from the state-n inputs stashed below. Leaving it here at the
            # instantaneous rate put the glaciation burst's one-step spike through the AB3
            # history (+23/12 of it, then −16/12 of it), rang the freshly nucleated moments
            # negative against a step-mean-bounded mass, and was the measured death of the ice
            # arm at t = 1031.4 s. Everything remaining in these four lines is riming, melting
            # and aggregation, which relax nothing and stay on the multistep.
            qsrc = rhoair * ((rk.prdr + MLQ[k]) + (aggq[k] * i_dt))
            nsrc = rhoair * ((MLN[k]) + (aggn[k] * i_dt))
            asrc = (rk.ardr + (rhoair * MLA[k])) + aagg
            csrc = (rk.crdr + (rhoair * MLC[k])) + cagg

            # State-n inputs of the step-mean partition. `capgam` is the live gate: it is an
            # exact 0.0 for a gated species (`_ice_empty_rates`), so the pre-compute calls the
            # partition only where a population exists and the zero-ice path stays bitwise.
            habANI[k][i] = ek.ani;    habCNI[k][i] = ek.cni;  habRNI[k][i] = ek.rni
            habDS[k][i]  = ek.deltastr; habRB[k][i] = ek.rhobar
            habNIM[k][i] = ek.ni * rhoair
            habVT[k][i]  = rk.vtrmi1; habCG[k][i] = rk.capgam
            habNIQ[k][i] = rk.niq

            # ── The var_check CONSISTENCY SOURCE ────────────────────────────────
            # `var_check` is ISHMAEL's own moment-consistency operator, and in the Fortran it
            # has `INTENT(INOUT)` arguments: its re-diagnosis is WRITTEN BACK into the state
            # every step. This port evaluates the rates at the effective moments but carries
            # the raw ones, and four moments fitted by four independent splines, sedimenting
            # at three different weighted speeds, do not stay a realizable population on their
            # own. Nothing else in the set restores them. Measured on the O01 ice arm without
            # this term: n_i1 reaches 2.2e9 m⁻³ against ~5e-10 kg/m³ of mass, the ice-rain
            # collection rate (∝ n_i·n_r) goes stiff, and slot 8 diverges from −4e-3 to −62
            # kg/m³ in six steps at t ≈ 1050 s.
            #
            # So the write-back is reinstated, as a SOURCE — a RELAXATION on a physical
            # timescale, `Ẋ = (X_eff − X)/τ_vc` with `τ_vc = physical_params[:tau_varcheck]`
            # (default 5 s), on the number and the two volume moments. It is `Δt`-FREE, like
            # every other rate in this block: `(X_eff − X)/Δt` would have been a one-step state
            # repair and would have made the tendency a function of the step size, which is
            # exactly the defect the depletion caps were removed for.
            #
            # The MASS is untouched — `var_check` never changes it — so this cannot move
            # water, cannot move energy, and cannot break the exchange closure above.
            # `options[:ice_var_check] = false` removes it and reproduces the drift.
            #
            # GATED ON THE SPECIES EXISTING, like every other rate. An empty species' `eff` is
            # the fallback initialization, not a re-diagnosis of anything, so restoring toward
            # it would inject `QNSMALL/Δt` into a slot that must produce an EXACT zero — and
            # the whole zero-ice inertness gate rests on that.
            # GATED ON THE POPULATION, not merely on the mass. `ek.ni` at carried n = 0 is
            # `var_check`'s re-derivation from the mass — the phantom population — so
            # relaxing the carried number TOWARD it would inject number that no nucleation
            # created, at 744/5 = 149 m⁻³ s⁻¹ on the O01 arm. That is the same
            # growth-without-activation this gate exists to forbid, arriving through the
            # consistency source instead of through a rate. A species that HAS a population
            # still gets the full relaxation, so the drift this term was added to stop
            # (n_i1 -> 2.2e9 m⁻³ against 5e-10 kg/m³) remains covered.
            if var_check_source && lives[k]
                nsrc += rhoair * (ek.ni - ns[k]) * i_tau_vc
                asrc += rhoair * (ek.ai - as[k]) * i_tau_vc
                csrc += rhoair * (ek.ci - cs[k]) * i_tau_vc
            end

            if is_target
                qn = rhoair * q_nuc
                nn = rhoair * n_nuc
                vn = _ice_nucleation_volume(qn, nn)
                qsrc += qn
                nsrc += nn
                asrc += vn
                csrc += vn
            else
                qsrc -= rhoair * IfzQ[k]
                nsrc -= rhoair * FzN[k]
            end

            SRCq[k][i] = qsrc
            SRCn[k][i] = nsrc
            SRCa[k][i] = asrc
            SRCc[k][i] = csrc
            # ── The ICE CONDUCTANCE λ SEES IS THE REALIZED ONE ────────────────────────
            #
            # `invtau_i` is consumed in three places that must agree: λ of the relaxation pair,
            # the `Q`-independent piece `−b·Σ_k τ_{i,k}^{-1}` of `N`, and the per-species
            # step-mean transfer `Q̇_{i,k} = drive·τ_{i,k}^{-1}/(1+𝒬_{s,i})`. When the ice donor
            # realization scales the TRANSFER but not the other two, the pair is integrating a
            # different exchange than the one the slots receive — exactly in the over-drawn ice
            # regions where the factor bites, and the mismatch is left for `τ_ss`/`τ_rec` to
            # absorb. That is an inconsistency, not an approximation.
            #
            # The consistent object follows from the same algebra the share rule does. The ice
            # contribution to Eq. Qss_ice is `−Σ_k Q̇_{i,k}(1+𝒬_{s,i})`; substituting the
            # REALIZED rate `Q̇_{i,k} = drive·f_k τ_{i,k}^{-1}/(1+𝒬_{s,i})` the psychrometric
            # factor cancels exactly as before and what is left is `−drive·Σ_k f_k τ_{i,k}^{-1}`.
            # So the effective conductance is `f_k τ_{i,k}^{-1}`, and writing it here puts it
            # into λ, into `N`, and into the transfer at once — one number, three consumers.
            #
            # The stiff limit is unchanged in form: `Q_ss^{qs} = N/λ` with `Σ_eff` in place of
            # `Σ`, so `drive = Q^{qs} + 𝒟 = F/λ + 𝒟 λ_l/λ` is Eq. wbf_qs with the realized ice
            # conductance — the Wegener-Bergeron-Findeisen competition weighted by the ice
            # surface that is actually active this step rather than by the surface a
            # reservoir-blind rate implies. `f_k = 1` wherever the ice is not over-drawn, which
            # is everywhere except the points the census flags, so this is inert in the bulk.
            QDi[k][i] = sub_now ? FIC[k] * rk.Qdot : rk.Qdot
            ITi[k][i] = sub_now ? FIC[k] * rk.invtau_i : rk.invtau_i
            Vm[k][i] = rk.vtrm
            Vn[k][i] = rk.vtrn
        end
    end
    return nothing
end

# ── Equation set ───────────────────────────────────────────────────────────────

"""
    mc_driver!(mtile, colstart, colend, t, geom)

Total-energy moist compressible equation set — the geometry-generic master driver.
Prognostic slots (perturbations vs the `PressureReferenceState` except u, w, rho_r,
and v on the cylindrical geometries): p' [Pa], rho_d', rho_t', u, w, E_t' [J/m^3],
Q_ss' [kg/m^3], rho_r, rho_c' (+ tangential v, slot 10, on the cylinders) and rho_v'.

Each step the CLOSED-FORM `retrieve_temperature` gives T from the prognostic condensed
masses; the vapor is its own prognostic slot; and the condensation rate is the (uncapped)
supersaturation relaxation, an equal and opposite pair of sources on rho_c and rho_v at
fixed rho_t and E_t. The energy equation carries no condensation source (exact first law);
the pressure equation's condensation coefficient is (L_v - R_v*C_pt*T/R_m). See
reference/Scythe_moist_compressible.tex, and the file header for why no water variable is a
residual any more.

Everything geometry-specific — the derivative-slot mapping, metric and curvature
terms, and the tangential-wind machinery — is dispatched on the singleton `geom`
trait (see mc_geometry.jl), so the Cartesian path compiles to exactly the
historical `moist_compressible_XZ` code. The name-dispatched equation sets are
thin wrappers below.
"""
function mc_driver!(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64,
                    geom::MCGeometry)

    grid = mtile.tile
    gridpoints = mtile.tilepoints
    expdot = mtile.expdot_n
    impdot = mtile.impdot_n
    model = mtile.model
    refstate = mtile.ref_state

    # Physical parameters. Momentum and the moist entropy s_t (heat) are the diffused
    # quantities: the masses, pressure, total energy and Q_ss carry no diffusive tendency
    # of their own (the heat/friction increments are slaved onto them). Momentum and heat
    # eddy coefficients are specified independently (eddy mixing is not a molecular-ratio
    # process); the heat coefficients default to the momentum values. Water-species mixing
    # is deferred (see the handoff doc).
    Khdiff = model.physical_params[:Khdiff]
    Kvdiff = model.physical_params[:Kvdiff]
    Khdiff_heat = get(model.physical_params, :Khdiff_heat, Khdiff)
    Kvdiff_heat = get(model.physical_params, :Kvdiff_heat, Kvdiff)
    Kvdiff_water = get(model.physical_params, :Kvdiff_water, 0.0)
    tau_qss = get(model.physical_params, :tau_qss, 10.0)
    # The vapor RECONCILIATION timescale (see `rho_v_reconcile`). ρ_t is still the conserved
    # anchor and ρ_v is now prognostic beside it, so the two carry a redundancy that has to be
    # removed or they drift apart under splitting error — the same role, and the same default,
    # `tau_qss` has for Q_ss. Long compared with the timestep on purpose: a drift correction,
    # never a shock.
    tau_rec = get(model.physical_params, :tau_rho_v_rec, 10.0)
    # The PARTITION reconciliation timescale — the third tier of the chain (TeX
    # §"Reconciliation of the condensate partition"): the advected ice moments are pulled
    # back inside the water the conserved ρ_t anchor supports, on the same class of
    # timescale as the two tiers above it, and only where they exceed it. `ice_anchor_on`
    # gates the SOURCE only — the `MC_ANCHOR_*` census measures the defect regardless, so
    # `false` reproduces the unreconciled configuration bitwise while still reporting it.
    tau_anchor = get(model.physical_params, :tau_ice_anchor, 10.0)
    ice_anchor_src = get(model.options, :ice_anchor_source, true)::Bool
    # The reader-side leg (see the rho_ice_t block): applied only with ice registered.
    ice_anchor_flr = get(model.options, :ice_anchor_floor, true)::Bool
    # The sedimentation leg (see the `anchor_f` scratch doc): applied only with ice on.
    ice_anchor_flx = get(model.options, :ice_anchor_flux, true)::Bool
    # The rate-side leg (see the read block in `mc_ice_sources!`): ditto.
    ice_anchor_rts = get(model.options, :ice_anchor_rates, true)::Bool
    # The POPULATION reconciliation — the FOURTH tier of the chain (TeX §"Reconciliation of
    # the population"): ice mass whose carried number is not positive is orphaned by the
    # population gate, exempt from every rate AND from every device that could remove it, and
    # is returned here to a representation the physics can act on — rain above `T_0` with
    # `L_f` absorbed, 2 μm crystals below it. `ice_pop_src` gates the SOURCE only; the
    # `MC_POP_*` census measures the defect either way. Env `SCYTHE_O01_POPSRC=0`.
    tau_pop = get(model.physical_params, :tau_ice_population, 10.0)
    ice_pop_src = get(model.options, :ice_population_source, true)::Bool
    # WHICH crystal the below-`T_0` branch seeds: `:min` (the 2 μm sphere, the fastest
    # response), `:large` (`var_check`'s own re-diagnosis of the dead mass at the floor
    # number — the size-sorted particles the mass came from; `_ice_population_seed_large`)
    # or `:local` (the per-crystal mass, habit and bulk density of the nearest live
    # gridpoint of the same species in the column — the crystals the ringing's negative
    # lobe lost, which are the ones next door; `_ice_population_seed_local`).
    # DEFAULT `:local`, chosen by measurement (reference/FINDINGS_ISHMAEL_S8S9.md §§5f-5j):
    # it is the only seed that both reconciles the dead mass and leaves the ice cloud at its
    # unreconciled magnitude. Env `SCYTHE_O01_POPSEED` selects `:min` or `:large`.
    pop_seed = get(model.options, :ice_population_seed, :local)::Symbol
    pop_seed in (:min, :large, :local) ||
        error("options[:ice_population_seed] = :$(pop_seed) is not recognized; use :min " *
              "(the 2 um sphere), :large (var_check's large-ice re-diagnosis) or :local " *
              "(the nearest live neighbour's crystals in the same column)")
    # TeX §Departures (e): above `T_0` a crystal that collects liquid SHEDS it. The collection
    # kernels still run — the melting rate keeps the sensible heat of the liquid that struck
    # the crystal — but no mass leaves the cloud or the rain and no `L_f` is released.
    # Applied at the CONSUMER (`_ice_species_pre`/`_ice_species_post`), never by gating the
    # ported kernel, so the reference harness stays bitwise. Env `SCYTHE_O01_SHED=0`.
    ice_shed = get(model.options, :ice_shed_above_t0, true)::Bool
    # The per-channel ATTRIBUTION census (`MC_ATTR_*`), opt-in and default OFF: it decomposes
    # the `MC_DONOR_*` breaches by LEG, which is a Stage-0 measurement rather than a standing
    # diagnostic. Writes only into `mc_water_stats`; no rate, state or slot is a function of
    # it. Env `SCYTHE_O01_ATTR=1` in benchmarks/o01_rainfall.jl.
    ice_attr = get(model.options, :ice_attr_census, false)::Bool
    # The PER-PAIR reservoir caps inside `ishmael_col1` (`min(colamt, q)`, `min(colamtn, n)`):
    # the `min(rate, ρ/Δt)` class every other ported call site switches off, still standing in
    # the aggregation kernel because the pair sums had nothing else bounding them. They do now
    # (the donor factors), so this is the switch that can retire them once the census has
    # measured what they move. Default ON — bitwise the caps the Fortran has. Env
    # `SCYTHE_O01_AGGCAPS=0` in benchmarks/o01_rainfall.jl.
    ice_agg_caps = get(model.options, :ice_agg_caps, true)::Bool
    # STAGE 2b: whether RAIN EVAPORATION joins the rain donor's conductance (TeX §donor_relax,
    # "The rain reservoir has a third sink that lived outside its conductance"). Default ON —
    # the committed construction, and the one that makes `MC_DONOR_QR`'s "≤ 1 by construction"
    # invariant true of the combined draw rather than of the ice half alone. `false` forces
    # `κ_ev = 0` and nothing else, which takes every rain factor back to the ice-leg-only
    # expression and the fold below the ice block back to a multiplication by an exact `1.0`
    # — i.e. it restores the pre-Stage-2b answer BITWISE, on the warm path as well as the ice
    # one. Forensics only. Env `SCYTHE_O01_EVAPREAL=0` in benchmarks/o01_rainfall.jl.
    rain_evap_real = get(model.options, :rain_evap_realization, true)::Bool
    # STAGE 1b: whether each ICE SPECIES' NUMBER is realized on its OWN donor conductance
    # rather than on its mass factor (TeX §donor_relax; see the number-donor block in
    # `mc_ice_sources!`). Default OFF — `false` sets every number factor to the species' mass
    # factor and every `f_isn<k>` to an exact `1.0`, which is what the three number legs
    # (aggregation's `deltan`, the melt number, the sublimation number sink) already carried,
    # so the answer is BITWISE the pre-Stage-1b one. The `MC_DONOR_N*` census rows are written
    # in BOTH modes: what they read with the switch off is the measurement the switch exists
    # to answer. Env `SCYTHE_O01_ICENREAL=1` in benchmarks/o01_rainfall.jl.
    ice_n_real = get(model.options, :ice_number_realization, false)::Bool
    # STAGE 3e: the RAIN POPULATION GATE — where the CARRIED rain number is not positive, the
    # kernels that read a rain SIZE DISTRIBUTION see no rain (see the `live_r` block in
    # `mc_ice_sources!`, and reference/FINDINGS_ISHMAEL_S8S9.md §5i for the 2800 μm phantom
    # DSD it forbids). Default ON: it is the rain's statement of the ice population gate, and
    # it is bitwise inert wherever `n_r > 0`. `false` restores the phantom-DSD answer for
    # forensics and NOTHING else — the minimum-crystal bound on the realized number sources
    # that ships with it has no switch, being an invariant rather than a parameterization.
    # Env `SCYTHE_O01_RAINGATE=0` in benchmarks/o01_rainfall.jl.
    rain_pop_gate = get(model.options, :rain_population_gate, true)::Bool
    # Turbulent Prandtl number for the Smagorinsky heat diffusion (Khdiff_heat < 0
    # sentinel, below). 1.0 = heat mixes with the same eddy diffusivity as momentum.
    Pr_t = get(model.physical_params, :Pr_t, 1.0)
    # Horizontal water-species mixing: 0.0 = off (default), < 0 = Smagorinsky
    # K_smag/Sc_t. DIAGNOSTIC and NOT energy consistent -- see the block by slot 8.
    Khdiff_water = get(model.physical_params, :Khdiff_water, 0.0)
    Sc_t = get(model.physical_params, :Sc_t, 1.0)

    # Coriolis parameter (constant f-plane) for the cylindrical geometries; the
    # Cartesian slice carries no rotation and its methods never read it.
    fcor = get(model.physical_params, :f, 0.0)

    # Rayleigh sponge (Durran-Klemp 1983 eq. 29 profile above z_damp): momentum-only,
    # damping u and w toward the resting base state. alpha = 0 (or absent keys)
    # disables it and skips the block entirely, keeping alpha = 0 configs bit-identical.
    alpha = get(model.physical_params, :alpha, 0.0)
    z_damp = get(model.physical_params, :z_damp, 0.0)

    # Warm-rain microphysics (autoconversion, collection, sedimentation, and the rain
    # channel of the supersaturation relaxation). N_r [#/cm^3] is the fixed rain-drop
    # number of the monodisperse tau_r closure; zeroing it (or the switch) makes the
    # rain channel inert and slot 8 advection-only. N_0 [m^-4] > 0 switches the
    # channel's timescale to the exponential (Marshall-Palmer) DSD closure (classic
    # value 8.0e6); absent, the monodisperse closure is bit-identical to before.
    precipitation = get(model.options, :precipitation, false)::Bool
    # Negative-water production attribution; see `water_budget_probe!`. Hoisted out of the
    # tendency block so the default path pays one Dict lookup per column, not two.
    budget_trace = get(model.options, :water_budget_trace, 0)::Int > 0
    # NOTE there is no depletion-cap plumbing here any more. `water_cap_factor` and
    # `water_cap_mode` are gone with the caps they sized: a rate floored at `-rho/ts` makes
    # the physics a function of the time step, which is the one thing a consistent scheme
    # cannot do. The AB3 bounds survive as a MEASUREMENT inside `water_depletion_probe!` (it
    # reports where they WOULD have bound), and the stiffness they used to hide is reported
    # by `mc_stiffness_census!`. See `qss_condensation_rates` for the full argument.
    # TOMBSTONE. `options[:vapor_retrieval]` selected between two DIAGNOSTIC vapors — the
    # density-budget residual and the supersaturation residual — and, from 2026-07-29, the C¹
    # regime blend of the two. All three are gone: the vapor is a PROGNOSTIC SLOT, so there is
    # nothing left to choose between. A configuration that still sets the key was tuned against
    # a representation this equation set no longer has, and must be looked at rather than
    # silently run.
    haskey(model.options, :vapor_retrieval) &&
        error("options[:vapor_retrieval] was retired: rho_v is prognostic (Stage A), so " *
              "there is no diagnostic vapor left to select a representation for. The blend " *
              "(vapor_retrieval_blend), its thresholds (:vapor_blend_l0/_l1/_t0/_t1) and its " *
              "correction cap (:vapor_blend_dcap) went with it. Remove the key; the " *
              "reconciliation of rho_v against the rho_t budget is now the nudge " *
              "physical_params[:tau_rho_v_rec] (see rho_v_reconcile).")
    # Whether the thermodynamic interface reads a floored rho_liq. `:none` (the default) is
    # bitwise the code that had no option; see `condensate_floor_mode` for why this is not a
    # clamp. In `options` for the same reason as the two above: a choice of what the
    # diagnostics READ, not a physical parameter.
    cond_floor = condensate_floor_mode(model.options)
    # Which control variable slot 9 carries. `:none` (the default) is bitwise the code that
    # had no option: the branch below then multiplies by an exact 1.0 and divides by an exact
    # 1.0, both of which are identity in IEEE. See `condensate_transform_mode`.
    ctrans = condensate_transform_mode(model.options)
    ctrans_on = ctrans !== :none
    cmu = get(model.physical_params, :condensate_mu, 1.0e-7)
    # The same for slot 8 (rain). Independent knob, same machinery; see `rain_transform_mode`
    # for why the two are separate and for what rain needs that cloud did not.
    rtrans = rain_transform_mode(model.options)
    rtrans_on = rtrans !== :none
    rmu = get(model.physical_params, :rain_mu, 1.0e-7)
    if (ctrans_on || rtrans_on) && budget_trace
        # `water_budget_probe!` attributes a slot's production to named channels and reports
        # them in kg/m^3/s. Under a transform the ADV column is in CONTROL-VARIABLE units
        # while the source columns are densities, so the rows would not sum to the tendency
        # and the mismatch would be invisible. Refuse rather than print a wrong budget --
        # this file has shipped one silently before (see `water_budget_trace`).
        error("options[:water_budget_trace] with a water transform on " *
              "(condensate_transform = :$(ctrans), rain_transform = :$(rtrans)) is not " *
              "implemented: the advective term is then in control-variable units and the " *
              "source terms in density units, so the attribution does not close. Rescale " *
              "the transformed slot's columns by the Jacobian before enabling it.")
    end
    # Two-moment rain (options[:rain_moments] == 2; see `rain_moments`). Read from the SLOT,
    # not from the option: `MCSlots` resolved the appended index by name once when the tile
    # was built, and `> 0` is the same test the writes below use, so "the slot exists" and
    # "the physics runs" cannot disagree. Its own transform is independent of slot 8's.
    nr_i = mtile.mc_slots.n_r
    rain_2m = nr_i > 0
    nrtrans = rain_2m ? rain_number_transform_mode(model.options) : :none
    nrtrans_on = nrtrans !== :none
    nrmu = get(model.physical_params, :mu_rain_n, 1.0)
    if rain_2m && budget_trace
        # `water_budget_probe!`/`water_depletion_probe!` write into fixed MASS rows
        # (MC_BUDGET_R/C/V) and attribute kg/m^3/s. A NUMBER slot has no row, and its sources
        # are #/m^3/s, so switching the closure to the two-moment rates would silently change
        # what the rain MASS rows mean (`auto+coll` becomes KK2000) while the number budget
        # went unreported entirely. Refuse rather than print a budget that does not close.
        error("options[:water_budget_trace] with options[:rain_moments] = 2 is not " *
              "implemented: the budget rows are the three MASS channels, so the rain " *
              "number slot has nowhere to report and its sources are in #/m^3/s. Add a " *
              "number block to `MC_WATER_STATS` before enabling it.")
    end
    if rain_2m && Kvdiff_water > 0.0
        # `_diffusion_water_step!` diffuses rho_t/rho_c/rho_r and implies the vapor; nothing
        # diffuses n_r. Mixing rain MASS without its NUMBER changes the mean drop size of
        # every column it touches — the DSD would be a function of the diffusivity — and the
        # fall speeds, the evaporation timescale and the self-collection all read that size.
        error("physical_params[:Kvdiff_water] > 0 with options[:rain_moments] = 2 is not " *
              "implemented: the implicit vertical water diffusion moves rho_r and would " *
              "leave n_r behind, silently rescaling the drop size of every diffused " *
              "column. Set Kvdiff_water = 0 or add the n_r solve to " *
              "`_diffusion_water_step!`.")
    end
    # ── Ice (options[:ice_microphysics] === :ishmael; see `ice_microphysics`) ──
    # Read from the SLOTS, on the same rule the rain number follows: `MCSlots` resolved the
    # twelve appended indices by name once when the tile was built, and `ice_registered` is
    # the same `> 0` test the writes below use, so "the slots exist" and "the ice runs"
    # cannot disagree. ONE transform family covers all twelve; FOUR widths, because `mu`
    # carries the units of what it transforms (see `ice_transform_mode`, `MC_ICE_MU_KEYS`).
    IS = mtile.mc_slots
    ice_on = ice_registered(IS)
    itrans = ice_on ? ice_transform_mode(model.options) : :none
    itrans_on = itrans !== :none
    # Through `ice_mu`, so the four widths have ONE source of truth
    # (`MC_ICE_MU_DEFAULTS`) and a default cannot be changed in one place and silently
    # overridden by a literal in another.
    imu_q = ice_mu(model.physical_params, 1)
    imu_n = ice_mu(model.physical_params, 2)
    imu_a = ice_mu(model.physical_params, 3)
    imu_c = ice_mu(model.physical_params, 4)
    if ice_on && budget_trace
        # Same refusal as the rain number's, one category further out: the budget rows are
        # the three LIQUID mass channels, so neither the ice mass nor its three moments has
        # anywhere to report, and `MC_BUDGET_V`'s vapor row would stop closing the moment
        # deposition existed.
        error("options[:water_budget_trace] with options[:ice_microphysics] = :ishmael is " *
              "not implemented: the budget rows are the liquid mass channels, so the ice " *
              "mass and its number/volume moments have nowhere to report. Add ice blocks to " *
              "`MC_WATER_STATS` before enabling it.")
    end
    if ice_on && Khdiff_water != 0.0
        # The horizontal counterpart, refused for the same reason and one more. The block
        # mixes rho_t, rho_d, rho_c and rho_r and takes the vapor Laplacian as the remainder
        # — which with ice inside rho_t is not the vapor — and it is explicitly NOT energy
        # consistent even for liquid (see the block itself). Mixing an ice MASS without its
        # number and volume moments would also rescale every crystal the operator touches.
        error("physical_params[:Khdiff_water] != 0 with options[:ice_microphysics] = " *
              ":ishmael is not implemented: the horizontal water mixing implies the vapor " *
              "from the liquid species, and mixing an ice mass without its n/a/c moments " *
              "rescales the crystals of every column it touches. Set Khdiff_water = 0 or " *
              "add the twelve ice Laplacians (and the energy consistency they need).")
    end
    if ice_on && Kvdiff_water > 0.0
        # `_diffusion_water_step!` diffuses rho_t/rho_c/rho_r and IMPLIES the vapor as the
        # remainder. With ice in rho_t that remainder is wrong by the ice mass, and nothing
        # diffuses the ice or its moments — the diffused column would gain vapor it does not
        # have and keep crystals whose volume moments the operator never touched.
        error("physical_params[:Kvdiff_water] > 0 with options[:ice_microphysics] = " *
              ":ishmael is not implemented: the implicit vertical water diffusion implies " *
              "the vapor from rho_t - rho_d - rho_c - rho_r, which with ice present is not " *
              "the vapor, and nothing diffuses the ice moments. Set Kvdiff_water = 0 or add " *
              "the ice solves to `_diffusion_water_step!`.")
    end
    if ice_on && get(model.options, :louis_bl, false)::Bool
        # The Louis BL mixes total water and cloud and takes the vapor as their difference
        # (`vdot_v = vdot_w - vdot_c` in `mc_boundary_layer.jl`). With ice inside rho_t that
        # difference attributes the ice flux to vapor, and the surface-flux ladder has no ice
        # leg at all. Refuse rather than mix a partition that does not add up.
        error("options[:louis_bl] with options[:ice_microphysics] = :ishmael is not " *
              "implemented: the boundary layer implies the vapor tendency as " *
              "rho_dot_w - rho_dot_c, which with ice in rho_t is not the vapor tendency, " *
              "and nothing mixes the ice moments. Add the ice legs to " *
              "`mc_boundary_layer_column!` before enabling it.")
    end
    N_r = precipitation ? get(model.physical_params, :N_r, 1.0e-3) : 0.0
    N_0 = precipitation ? get(model.physical_params, :N_0, 0.0) : 0.0
    # Cloud droplet number [#/cm^3] of the condensation closure. Passed explicitly (it was
    # the positional default before, at the same value, so the default path is unchanged) so
    # that the two-moment autoconversion can use the SAME number rather than a second,
    # disagreeing one. See `rain_autoconversion_2m`.
    max_N_c = get(model.physical_params, :max_N_c, 100.0)

    # Louis boundary layer (vertical mixing + surface drag; mc_boundary_layer.jl)
    # and Smagorinsky horizontal closure (Ls > 0 replaces the constant Khdiff in
    # the momentum diffusion). Both default OFF so existing configurations are
    # bit-identical.
    louis_bl = get(model.options, :louis_bl, false)::Bool
    # Radiative heating. `RAD.q_lw`/`q_sw` are the HELD flux divergences [W/m^3] the
    # pre-pass (`radiation_prepass!`, src/radiation.jl) recomputed on the radiation cadence
    # before this column loop started; nothing is solved here. `rad_on` is a plain field
    # load because `ModelTile` carries the state concretely (see `EMPTY_RADIATION`), and it
    # gates the fold below so a radiation-free run is BYTE-identical to the code that had
    # no radiation at all -- `x + 0.0` is not the identity for `x = -0.0`.
    RAD = mtile.radiation
    rad_on = RAD.active
    # Horizontal acoustic semi-implicit (the patch-level ADI sweep in
    # horizontal_si.jl). Default OFF so existing configurations are bit-identical.
    hsi = get(model.options, :horizontal_semiimplicit, false)::Bool
    # Exact (unsplit) 2-D acoustic semi-implicit (exact_si.jl). Shares the
    # horizontal remainder/staging blocks with hsi below; the driver exits at
    # the end of phase A (before the per-column implicit solve — the solve
    # happens at the patch level, then exact_si_apply_column! completes the
    # column). :exact_si_zero_x is the A≡0 TEST lever: it zeroes the whole
    # horizontal coupling so the path must be bitwise the vertical-only path.
    xsi = get(model.options, :exact_si, false)::Bool
    xsi_zero = xsi && (get(model.options, :exact_si_zero_x, false)::Bool)
    hsi_like = hsi || (xsi && !xsi_zero)
    # State-dependent vertical acoustic linearization: the implicit pair's
    # coefficients (Pξ, ρ̂_t and the slaved-leg chains) come from the CURRENT
    # column state each step instead of the resting reference, and the
    # Helmholtz matrix is refactorized per column per step. Removes the
    # convective (finite-amplitude) SI ceiling — the resting-reference form
    # leaves δ·Co_z of the grid-scale acoustic operator explicit in a core
    # whose state deviates by δ, fatal in TC deep convection at Co_z ≳ 4
    # (reference/SI_CONVECTIVE_CEILING.md). Default OFF so existing configurations
    # are bit-identical; the TC configs enable it. Spline (RiRk) vertical only.
    sd_si = get(model.options, :state_dependent_si, false)::Bool
    l_inf = get(model.physical_params, :l_inf, 80.0)
    Cd_param = get(model.physical_params, :Cd, -1.0)
    sfc_wind_factor = get(model.physical_params, :sfc_wind_factor, 1.0)
    Ls = get(model.physical_params, :Ls, 0.0)
    K_min = get(model.physical_params, :K_min, 0.0)
    use_smag = Ls > 0.0

    # Bulk surface enthalpy/moisture fluxes over a fixed-SST ocean: the flux
    # values are the bottom nodes of the Louis heat/water flux columns, so the
    # switch requires louis_bl. SST is in KELVIN (Float64 params carry no units;
    # the guard catches the Celsius footgun).
    surface_fluxes = get(model.options, :surface_fluxes, false)::Bool
    if surface_fluxes && !louis_bl
        error("options[:surface_fluxes] requires options[:louis_bl] — the fluxes " *
              "enter as the bottom nodes of the Louis boundary-layer flux columns")
    end
    Ck = get(model.physical_params, :Ck, 1.0e-3)
    SST = get(model.physical_params, :SST, 301.15)
    if surface_fluxes && SST <= 200.0
        error("physical_params[:SST] must be in Kelvin (got $SST — 28 C is 301.15)")
    end
    U_min = get(model.physical_params, :U_min, 0.0)

    # Gridpoints: z from the geometry's vertical column; r is the geometry metric
    # handle (radius view on the cylinders, colatitude/a/Omega on the sphere,
    # `nothing` on the Cartesian geometries — see mc_metric)
    z = view(gridpoints,colstart:colend,zcoord(geom))
    r = mc_metric(geom, model, gridpoints, colstart, colend)

    # Prognostic slot views, geometry-mapped (see mc_slot_views): `_x` is ∂x on the
    # slice and ∂r on the cylinders, `_z`/`_zz` sit at zslot(geom), and the raw
    # azimuthal `f_l`/`f_ll` pair exists only on the 3D grid (`nothing` on 2D).
    pv  = mc_slot_views(grid, colstart, colend, 1, geom)   # p' [Pa]
    rdv = mc_slot_views(grid, colstart, colend, 2, geom)   # rho_d'
    rtv = mc_slot_views(grid, colstart, colend, 3, geom)   # rho_t'
    uv  = mc_slot_views(grid, colstart, colend, 4, geom)   # u
    wv  = mc_slot_views(grid, colstart, colend, 5, geom)   # w
    etv = mc_slot_views(grid, colstart, colend, 6, geom)   # E_t'
    qsv = mc_slot_views(grid, colstart, colend, 7, geom)   # Q_ss'
    rrv = mc_slot_views(grid, colstart, colend, 8, geom)   # rho_r (rho_rbar = 0)
    rcv = mc_slot_views(grid, colstart, colend, 9, geom)   # rho_c'
    vv  = mc_v_views(geom, grid, colstart, colend)         # tangential v (cylinders)
    # The APPENDED but UNCONDITIONAL vapor slot (10 on XZ, 11 on the cylinders — see
    # `MCSlots`). Bound like the fixed nine: the index is a field load resolved once per tile,
    # and every configuration of this set carries it, so there is no branch here at all.
    rv_i = IS.rho_v
    rvv = mc_slot_views(grid, colstart, colend, rv_i, geom)   # rho_v'
    # The APPENDED rain-number slot (11 on XZ, 12 on the cylinders — see `MCSlots`). Bound
    # UNCONDITIONALLY, at slot 8 when the option is off, so the SubArray construction is the
    # same straight-line code every other slot's is and elides identically; nothing reads it
    # unless `rain_2m`. (Building it inside a runtime branch is what stops a view eliding.)
    nrv = mc_slot_views(grid, colstart, colend, rain_2m ? nr_i : 8, geom)
    # The twelve APPENDED ice slots (11-22 on XZ, 12-23 on the cylinders), bound the same
    # way and for the same reason: UNCONDITIONALLY, aliased to slot 8 when ice is off, so
    # every one of these is the same straight-line `mc_slot_views` call the fixed slots
    # make and elides identically. Nothing reads them unless `ice_on`.
    i1qv = mc_slot_views(grid, colstart, colend, ice_on ? IS.i1_q : 8, geom)
    i1nv = mc_slot_views(grid, colstart, colend, ice_on ? IS.i1_n : 8, geom)
    i1av = mc_slot_views(grid, colstart, colend, ice_on ? IS.i1_a : 8, geom)
    i1cv = mc_slot_views(grid, colstart, colend, ice_on ? IS.i1_c : 8, geom)
    i2qv = mc_slot_views(grid, colstart, colend, ice_on ? IS.i2_q : 8, geom)
    i2nv = mc_slot_views(grid, colstart, colend, ice_on ? IS.i2_n : 8, geom)
    i2av = mc_slot_views(grid, colstart, colend, ice_on ? IS.i2_a : 8, geom)
    i2cv = mc_slot_views(grid, colstart, colend, ice_on ? IS.i2_c : 8, geom)
    i3qv = mc_slot_views(grid, colstart, colend, ice_on ? IS.i3_q : 8, geom)
    i3nv = mc_slot_views(grid, colstart, colend, ice_on ? IS.i3_n : 8, geom)
    i3av = mc_slot_views(grid, colstart, colend, ice_on ? IS.i3_a : 8, geom)
    i3cv = mc_slot_views(grid, colstart, colend, ice_on ? IS.i3_c : 8, geom)

    pp = pv.f;      pp_x = pv.f_x;         pp_z = pv.f_z
    rho_dp = rdv.f; rho_dp_x = rdv.f_x;    rho_dp_z = rdv.f_z; rho_dp_zz = rdv.f_zz
    rho_tp = rtv.f; rho_tp_x = rtv.f_x;    rho_tp_z = rtv.f_z; rho_tp_zz = rtv.f_zz
    u = uv.f;       u_x = uv.f_x;          u_z = uv.f_z;       u_zz = uv.f_zz
    w = wv.f;       w_x = wv.f_x;          w_z = wv.f_z;       w_zz = wv.f_zz
    E_tp = etv.f;   E_tp_x = etv.f_x;      E_tp_z = etv.f_z
    Q_ssp = qsv.f;  Q_ssp_x = qsv.f_x;     Q_ssp_z = qsv.f_z
    rho_rp = rrv.f; rho_rp_x = rrv.f_x;    rho_rp_z = rrv.f_z; rho_rp_zz = rrv.f_zz
    rho_cp = rcv.f; rho_cp_x = rcv.f_x;    rho_cp_z = rcv.f_z; rho_cp_zz = rcv.f_zz
    rho_vp = rvv.f; rho_vp_x = rvv.f_x;    rho_vp_z = rvv.f_z; rho_vp_zz = rvv.f_zz

    # Reference state (pressure-based). Views, not `[:,1]` copies: these are read-only and
    # loop-invariant, so copying them allocated a fresh kDim vector per column per timestep.
    pbar = view(ref_pressure(refstate),:,1)
    pbar_z = view(ref_pressure(refstate),:,2)
    rho_dbar = view(ref_rho_d(refstate),:,1)
    rho_dbar_z = view(ref_rho_d(refstate),:,2)
    rho_tbar = view(ref_rho_t(refstate),:,1)
    rho_tbar_z = view(ref_rho_t(refstate),:,2)
    E_tbar = view(ref_total_energy(refstate),:,1)
    E_tbar_z = view(ref_total_energy(refstate),:,2)
    Q_ssbar = view(ref_qss(refstate),:,1)
    Q_ssbar_z = view(ref_qss(refstate),:,2)
    rho_cbar = view(Springsteel.ref_rho_c(refstate),:,1)
    rho_cbar_z = view(Springsteel.ref_rho_c(refstate),:,2)
    # LOCAL reference sound-speed-squared profile γ̄_m(z)·p̄/ρ̄_t (see
    # mc_reference_diagnostics) — the acoustic linearization must use the local
    # value so the explicit remainder is O(perturbation) at every level; the
    # domain-mean sound_speed_sq is unstable above Co_z ≈ 4.5 on a stratified base.
    Pxi_bar = mtile.mc_ref_diag.Pxi_prof
    # The DERIVED reference vapor ρ̄_v ≡ ρ̄_t − ρ̄_d − ρ̄_c and its vertical derivative, both
    # from `mc_reference_diagnostics` so that the profile the slot is carried against is
    # BIT-FOR-BIT the one `res_rho_t` reconstructs at rest — the whole reason the resting
    # nudge is identically zero rather than zero to fit tolerance. Not `Springsteel.ref_rho_v`,
    # which is an independent fit of the same quantity.
    rho_vbar = mtile.mc_ref_diag.rho_vbar
    rho_vbar_z = mtile.mc_ref_diag.rho_vbar_z

    # Per-thread work vectors for every temporary below (see `MC_SCRATCH_SLOTS`). Each `@.`
    # writes into a preallocated column instead of allocating a fresh one per column per step.
    S = @inbounds mtile.mc_scratch[Threads.threadid()]

    # Total fields (perturbation + reference)
    p = S.p;         @. p = pp + pbar
    rho_d = S.rho_d; @. rho_d = rho_dp + rho_dbar
    rho_t = S.rho_t; @. rho_t = rho_tp + rho_tbar
    E_t = S.E_t;     @. E_t = E_tp + E_tbar
    Q_ss = S.Q_ss;   @. Q_ss = Q_ssp + Q_ssbar
    # The VAPOR, staged here with the other totals rather than retrieved 150 lines down. It is
    # a prognostic, untransformed perturbation against the derived ρ̄_v, so this is the same
    # `perturbation + reference` line rho_t and Q_ss take — and everything downstream (the
    # temperature retrieval's partition, q_v, Q_s, the condensation closure, the surface
    # moisture flux) now reads STATE AT n, exactly as it does for rho_t.
    rho_v = S.rho_v; @. rho_v = rho_vp + rho_vbar
    # Slot 9. Under `:none` this is the cloud density and `Jc ≡ 1`; under a transform the slot
    # carries the control variable `ν' = ν − ν̄` with `ν̄ = bhyp(ρ̄_c)` (Ooyama predicts the
    # DEVIATION from the transformed background, §4d), and the density is recovered pointwise.
    # `ν̄` and `ν̄_z` are formed here rather than cached because ρ̄_c is identically zero in
    # every current configuration, making them exactly zero for two kDim-length operations.
    nu_c = S.nu_c; nu_c_z = S.nu_c_z; Jc = S.Jc
    rho_c = S.rho_c
    if ctrans_on
        @. nu_c = rho_cp + bhyp(rho_cbar, cmu)
        @. nu_c_z = rho_cp_z + (dbhyp(rho_cbar, cmu) * rho_cbar_z)
        if ctrans === :bhyp
            @. rho_c = ahyp(nu_c, cmu)
        else
            @. rho_c = ahyp_smooth(nu_c, cmu)
        end
        # Evaluated at the RECOVERED density, clamped at zero for the argument only: that is
        # what bounds J in [0.5, 1] and removes the source stiffness. Clamping the argument
        # changes no state, no mass and no energy — it perturbs the tendency by at most a
        # factor of two at points whose density is within μ of zero.
        @. Jc = dbhyp(max(rho_c, 0.0), cmu)
    else
        @. nu_c = rho_cp + rho_cbar
        @. nu_c_z = rho_cp_z + rho_cbar_z
        copyto!(rho_c, nu_c)
        fill!(Jc, 1.0)
    end
    # Slot 8, the same construction for rain — with one simplification: rain is a TOTAL with
    # `ρ̄_r ≡ 0`, so the slot IS `ν_r`, with no background to add and no `ν̄_r` to form. Under
    # `:none` the copy is of identical doubles and `Jr` is an exact 1.0, so the default path
    # stays bit-identical (the same construction slot 9 uses above).
    nu_r = S.nu_r; nu_r_z = S.nu_r_z; Jr = S.Jr
    rho_r = S.rho_r
    copyto!(nu_r, rho_rp)
    copyto!(nu_r_z, rho_rp_z)
    if rtrans_on
        if rtrans === :bhyp
            @. rho_r = ahyp(nu_r, rmu)
        else
            @. rho_r = ahyp_smooth(nu_r, rmu)
        end
        @. Jr = dbhyp(max(rho_r, 0.0), rmu)
    else
        copyto!(rho_r, nu_r)
        fill!(Jr, 1.0)
    end
    # The appended rain-NUMBER slot, identical construction to slot 8 above: also a TOTAL
    # (`n̄_r ≡ 0`), so the slot IS its control variable. Skipped entirely under the default
    # single-moment configuration, where none of `n_r`/`nu_nr`/`Jnr` is ever read.
    nu_nr = S.nu_nr; nu_nr_z = S.nu_nr_z; Jnr = S.Jnr
    n_r = S.n_r
    if rain_2m
        copyto!(nu_nr, nrv.f)
        copyto!(nu_nr_z, nrv.f_z)
        if nrtrans_on
            if nrtrans === :bhyp
                @. n_r = ahyp(nu_nr, nrmu)
            else
                @. n_r = ahyp_smooth(nu_nr, nrmu)
            end
            @. Jnr = dbhyp(max(n_r, 0.0), nrmu)
        else
            copyto!(n_r, nu_nr)
            fill!(Jnr, 1.0)
        end
    end
    # The twelve appended ICE slots, each the same construction through the shared
    # `_load_total_slot!`: all twelve are TOTALS (`f̄ ≡ 0`), so each slot IS its own control
    # variable. One transform family, four widths — `mu` is dimensional and mass, number and
    # the two volume moments are ten and nine decades apart (`MC_ICE_MU_KEYS`). Skipped
    # entirely when ice is off, where none of these columns is ever read.
    i1q = S.i1q; i1n = S.i1n; i1a = S.i1a; i1c = S.i1c
    i2q = S.i2q; i2n = S.i2n; i2a = S.i2a; i2c = S.i2c
    i3q = S.i3q; i3n = S.i3n; i3a = S.i3a; i3c = S.i3c
    if ice_on
        _load_total_slot!(i1q, S.nu_i1q, S.nu_i1q_z, S.J_i1q, i1qv, itrans_on, itrans, imu_q)
        _load_total_slot!(i1n, S.nu_i1n, S.nu_i1n_z, S.J_i1n, i1nv, itrans_on, itrans, imu_n)
        _load_total_slot!(i1a, S.nu_i1a, S.nu_i1a_z, S.J_i1a, i1av, itrans_on, itrans, imu_a)
        _load_total_slot!(i1c, S.nu_i1c, S.nu_i1c_z, S.J_i1c, i1cv, itrans_on, itrans, imu_c)
        _load_total_slot!(i2q, S.nu_i2q, S.nu_i2q_z, S.J_i2q, i2qv, itrans_on, itrans, imu_q)
        _load_total_slot!(i2n, S.nu_i2n, S.nu_i2n_z, S.J_i2n, i2nv, itrans_on, itrans, imu_n)
        _load_total_slot!(i2a, S.nu_i2a, S.nu_i2a_z, S.J_i2a, i2av, itrans_on, itrans, imu_a)
        _load_total_slot!(i2c, S.nu_i2c, S.nu_i2c_z, S.J_i2c, i2cv, itrans_on, itrans, imu_c)
        _load_total_slot!(i3q, S.nu_i3q, S.nu_i3q_z, S.J_i3q, i3qv, itrans_on, itrans, imu_q)
        _load_total_slot!(i3n, S.nu_i3n, S.nu_i3n_z, S.J_i3n, i3nv, itrans_on, itrans, imu_n)
        _load_total_slot!(i3a, S.nu_i3a, S.nu_i3a_z, S.J_i3a, i3av, itrans_on, itrans, imu_a)
        _load_total_slot!(i3c, S.nu_i3c, S.nu_i3c_z, S.J_i3c, i3cv, itrans_on, itrans, imu_c)
    end

    # Total vertical gradients (perturbation + reference)
    p_z = S.p_z;         @. p_z = pp_z + pbar_z
    rho_d_z = S.rho_d_z; @. rho_d_z = rho_dp_z + rho_dbar_z
    rho_t_z = S.rho_t_z; @. rho_t_z = rho_tp_z + rho_tbar_z
    E_t_z = S.E_t_z;     @. E_t_z = E_tp_z + E_tbar_z
    Q_ss_z = S.Q_ss_z;   @. Q_ss_z = Q_ssp_z + Q_ssbar_z
    # The DENSITY vertical gradient, f'(n)·∂z n = ∂z n / J — self-consistent with the recovered
    # density because both come from the same fit. Under `:none`, J is exactly 1.0 and this is
    # bitwise `rho_cp_z + rho_cbar_z`. Advection does NOT read this (it is transform-invariant
    # and reads `nu_c_z` directly); this exists for any consumer that wants dρ_c/dz.
    rho_c_z = S.rho_c_z; @. rho_c_z = nu_c_z / Jc
    # The vapor's TOTAL vertical gradient, perturbation slot plus derived reference — the
    # same `f'_z + f̄_z` the four lines above form, and what the advective product-rule term
    # below reads (the horizontal leg reads the perturbation gradient, because ρ̄_v has no
    # x-dependence).
    rho_v_z = S.rho_v_z; @. rho_v_z = rho_vp_z + rho_vbar_z

    # Diagnostic thermodynamic state. EVERY water species is prognostic now — vapor, cloud,
    # rain and the twelve ice moments — so the temperature retrieval is closed-form from the
    # condensed masses and nothing in this block is a difference of large nearly-equal fields.
    # See the file header for the chain that ended here.
    ke = S.ke;   mc_ke!(ke, geom, u, w, vv)
    geo = S.geo; @. geo = ke + (gravity * z)
    M = S.M;     @. M = p + E_t - (rho_t * geo)
    rho_liq = S.rho_liq; @. rho_liq = rho_c + rho_r
    # The total ICE density, ρ_i = Σ_k ρ_{i,k} — the retrieval's second condensed mass and the
    # third term of the vapor residual. Filled exactly where `rho_liq` is, and ZEROED (not
    # left stale) when ice is off, because every consumer below reads it unconditionally: the
    # `ρ_i = 0` arithmetic is EXACT (see `retrieve_temperature`), so an ice-free run is
    # bitwise the code that had no ice, and one `fill!` per column is the price of not
    # branching the whole thermodynamic block.
    rho_ice = S.rho_ice
    if ice_on
        @. rho_ice = (i1q + i2q) + i3q
    else
        fill!(rho_ice, 0.0)
    end
    # What the THERMODYNAMICS reads. Under the default `:none` this is a copy of identical
    # doubles, so the whole option is bitwise inert. Under `:diagnostic` it is the floored
    # liquid, and ONLY the consumers below see it: the retrieval, q_l -> C_vt/R_m/gamma_m,
    # Q_s_energy, the entropy and E_sed. `rho_liq` itself stays raw, because its other reader
    # must NOT be floored -- `res_rho_t` is the water PARTITION the reconciliation differences
    # the prognostic vapor against, and flooring a partition manufactures vapor.
    rho_liq_t = S.rho_liq_t
    if cond_floor
        @. rho_liq_t = max(rho_c, 0.0) + max(rho_r, 0.0)
    else
        copyto!(rho_liq_t, rho_liq)
    end
    # The ice mirror of `rho_liq_t`, floored per species under `:diagnostic` for the same
    # reason: a spline undershoot in one ice mass must not reach the retrieval as negative
    # latent heat. `rho_ice` itself stays raw, because `res_rho_t` must see the water
    # partition the continuity equations actually carry.
    #
    # AND capped at the ANCHOR HEADROOM below (`ice_anchor_floor`, default on with ice):
    # the reader-side leg of the partition reconciliation (TeX §Reconciliation of the
    # condensate partition). The reconciliation SOURCE holds the accumulated detachment at
    # drift scale, but the terminal §2c burst is a per-step grid-scale oscillation of the
    # ice-mass fit at an under-resolved front — measured swinging ±50% PER STEP while the
    # anchor water stays smooth — and no τ-relaxation outruns a per-step oscillation. What
    # made it lethal was never the ringing itself but the AMPLIFIER: the retrieval credits
    # each transient phantom with L_s (~1e4 K per kg/m³ measured), and the T spike closes
    # the sound-speed/updraft loop that sharpens the front further. Capping what the
    # thermodynamics READS at the water the anchor supports breaks that loop while state,
    # continuity, `res_rho_t` and the census all stay raw — nothing is written back and no
    # mass is converted, exactly the `condensate_floor` device class. Binds only where the
    # partition is inadmissible; bitwise inert everywhere else (strict branch, no scaling).
    rho_ice_t = S.rho_ice_t
    if cond_floor && ice_on
        @. rho_ice_t = (max(i1q, 0.0) + max(i2q, 0.0)) + max(i3q, 0.0)
    else
        copyto!(rho_ice_t, rho_ice)
    end
    if ice_anchor_flr && ice_on
        @inbounds for i in eachindex(rho_ice_t)
            head = max((rho_t[i] - rho_d[i]) - max(rho_liq_t[i], 0.0), 0.0)
            rho_ice_t[i] > head && (rho_ice_t[i] = head)
        end
    end
    Tk = S.Tk;   @. Tk = retrieve_temperature(M, rho_d, rho_t, rho_liq_t, rho_ice_t)
    p_hPa = S.p_hPa;   @. p_hPa = p / 100.0
    rho_vs = S.rho_vs; @. rho_vs = rho_v_sat(Tk, p_hPa)
    # The DENSITY-BUDGET residual — no longer A vapor, but the vapor the ρ_t budget IMPLIES.
    # With ρ_v prognostic the two are independent numbers, and the whole content of the
    # redundancy is their difference δ = res_rho_t − ρ_v: the nudge `rho_v_reconcile` removes
    # it on τ_rec, and `_vapor_gap_census!` records max|δ| as the drift diagnostic. It is
    # identically zero on a resting base by construction (see `mc_reference_diagnostics`).
    # ICE IS IN THE RESIDUAL: ρ_v = ρ_t − ρ_d − ρ_liq − ρ_ice (Eq. ice_mass of the TeX). With
    # ice off `rho_ice` is an exact zero column and `x - 0.0 === x`, so this is bitwise the
    # three-term residual it was.
    res_rho_t = S.res_rho_t; @. res_rho_t = rho_t - rho_d - rho_liq - rho_ice
    q_v = S.q_v;     @. q_v = rho_v / rho_d
    q_l = S.q_l;     @. q_l = rho_liq_t / rho_d      # thermodynamic reader: see rho_liq_t
    q_i = S.q_i;     @. q_i = rho_ice_t / rho_d      # ditto; exactly 0.0 with ice off
    # Ice enters the mixture heat capacity through `q_i C_i`, exactly as liquid does through
    # `q_l C_l` — a condensed phase performs no expansion work (TeX Eq. mixture_C). `R_m` is
    # untouched (ice exerts no partial pressure) and `C_pt = C_vt + R_m` still holds, so
    # gamma_m and the acoustic coefficient change VALUE with q_i but not FORM.
    C_vt = S.C_vt;   @. C_vt = Cvd + (q_v * Cvv) + (q_l * Cl) + (q_i * Ci)
    R_m = S.R_m;     @. R_m = Rd + (q_v * Rv)
    C_pt = S.C_pt;   @. C_pt = C_vt + R_m
    gamma_m = S.gamma_m; @. gamma_m = C_pt / C_vt
    # State-dependent acoustic coefficient Pξⁿ(z) = γ_m·p/ρ_t of the CURRENT
    # column state — used by the remainder/history staging below and by
    # semiimplicit_adjustment_p (same thread, same column, so the scratch
    # column persists into the adjustment call).
    if sd_si
        @. S.sd_pxi = gamma_m * p / rho_t
    end
    Lv = S.Lv;           @. Lv = L_v(Tk)
    drvs_dT = S.drvs_dT; @. drvs_dT = drho_vsat_dT(Tk, p_hPa)
    drvs_dp = S.drvs_dp; @. drvs_dp = drho_vsat_dp(Tk, p_hPa)

    # Condensation: supersaturation relaxation with the energy-consistent
    # psychrometric factor, split between the cloud and rain channels in proportion to
    # their inverse timescales (1/tau = 1/tau_c + 1/tau_r). Qdot moves mass between the
    # prognostic condensate (slot 9) and the prognostic vapor at fixed rho_t and E_t -- an
    # equal and opposite pair of slot sources now, with rho_t untouched -- so the latent heat
    # comes out of the retrieval and no adjustment step is needed after the timestep. With the
    # rain channel inert (N_r = 0), Qdot is bit-identical to the single-category closure and
    # Qdot_r is exactly zero.
    Q_s = S.Q_s;   @. Q_s = Q_s_energy(Tk, p, rho_d, q_v, q_l, q_i)
    Qdot = S.Qdot          # cloud channel
    Qdot_r = S.Qdot_r      # rain channel
    # `options[:condensation] = false` switches the phase change off ENTIRELY --
    # both channels, condensation and evaporation. It is a DIAGNOSTIC control, not
    # a physics option: `:precipitation => false` only zeroes N_r/N_0, which stops
    # rain but leaves the cloud channel running, so a run advertised as "no
    # physics" still evaporates cloud every step.
    #
    # It was introduced to diagnose the phantom-cloud drain (see the file header),
    # back when rho_c was the residual and a subsaturated column could carry
    # ~2e-6 kg/m^3 of cloud that was not there, evaporate it, and have it regenerated
    # from the same fit error on the next step. With rho_c prognostic that pathway is
    # closed -- a subsaturated column has rho_c = 0 exactly and both gates shut, so
    # this switch should now make NO difference to a cloud-free initial state. That is
    # a useful regression check in its own right (model_tests/tc_discrete_balance.jl),
    # and it remains the clean separator between "the initial state is not a discrete
    # steady state" and "the moisture is doing something".

    # ── Depletion budgets: DIAGNOSTIC ONLY ─────────────────────────────────────
    # These are the AB3-exact bounds the condensation rates used to be limited by. Nothing in
    # the RHS reads them any more (see `qss_condensation_rates` for why the caps were removed);
    # they are computed only when the depletion census is on, which then reports where they
    # WOULD have bound -- the instrument that measures what removing them cost.
    cap_c = S.cap_c
    cap_r = S.cap_r
    cap_v = S.cap_v
    if budget_trace
        micro_nm1 = mtile.mc_micro_nm1
        micro_nm2 = mtile.mc_micro_nm2
        @inbounds for i in eachindex(cap_c)
            g = colstart + i - 1
            cap_c[i] = _ab3_sink_bound(model.ts, t, max(rho_c[i], 0.0),
                                       micro_nm1[g, MC_MICRO_C], micro_nm2[g, MC_MICRO_C])
            cap_r[i] = _ab3_sink_bound(model.ts, t, max(rho_r[i], 0.0),
                                       micro_nm1[g, MC_MICRO_R], micro_nm2[g, MC_MICRO_R])
            # The vapor is a SINK channel too -- condensation removes it -- so its bound has
            # the same form and is negated into a ceiling on `Qdot_c + Qdot_r` at the use site.
            # The PROGNOSTIC `rho_v`, not `res_rho_t`: the measurement is of the vapor the
            # closure actually reads, and that is now a slot rather than a retrieval.
            cap_v[i] = _ab3_sink_bound(model.ts, t, max(rho_v[i], 0.0),
                                       micro_nm1[g, MC_MICRO_V], micro_nm2[g, MC_MICRO_V])
        end
    end
    invtau_c = S.invtau_c
    invtau_r = S.invtau_r
    # The LIQUID half of the stiff-relaxation split (TeX Eq. relax_linear; see
    # `relaxation_adjustment_qss!`). The withheld slot-7 term is
    # `−(Q̇_c + Q̇_r)(1 + 𝒬_s) = −Q_ss_drive·(τ_c^{-1} + τ_r^{-1})` — the psychrometric factor
    # cancels at the RATE level, before any integral is taken, which is what keeps the
    # cancellation intact under the exponential (TeX, second property). Splitting that term
    # into `−λ Q_ss` and a `Q`-independent remainder is therefore a statement about the DRIVE
    # CLIP and nothing else:
    #
    #   * UNCLIPPED (`Q_ss_drive == Q_ss`): the whole term is `−Q_ss·invtau`, i.e. entirely
    #     the λ part, and it contributes NOTHING to `N`;
    #   * CLIPPED (`Q_ss_drive == max(ρ_v,0) − ρ_v*`): the drive no longer depends on `Q_ss` at
    #     all, so the whole term is a constant flux, contributes NOTHING to λ, and belongs in
    #     `N` in full — exactly as the TeX states ("a channel whose drive is clipped by the
    #     vapor-availability bound contributes its (then constant) flux to N and nothing to λ").
    #
    # The classification is FROZEN at state `n`, like every other coefficient the step freezes.
    # The clip itself is recomputed here from the same three scratch columns
    # `qss_condensation_rates` reads, so the two cannot disagree by so much as an ulp.
    etd_lam = S.etd_lam
    etd_nl = S.etd_nl
    kappa_ev = S.kappa_ev
    f_rain = S.f_rain
    if get(model.options, :condensation, true)::Bool
        # The closure reads the PROGNOSTIC `rho_v` (the call site is unchanged; what it is
        # handed is now a transported slot rather than a difference of five fitted fields,
        # which is the whole point of Stage A).
        for i in 1:length(Qdot)
            Qdot[i], Qdot_r[i], invtau_c[i], invtau_r[i] =
                qss_condensation_rates(Q_ss[i], rho_v[i], rho_c[i], rho_r[i],
                                       rho_d[i], Tk[i], p_hPa[i], Q_s[i],
                                       model.ts, N_r, max_N_c; N_0=N_0,
                                       rain_2m=rain_2m, n_r_density=n_r[i])
            if isnan(Qdot[i]) || isnan(Qdot_r[i])
                error("Qdot is NaN at index $i, time $(t)!\n" *
                      "  T = $(Tk[i]) K, p = $(p[i]) Pa, rho_d = $(rho_d[i]), " *
                      "rho_t = $(rho_t[i]), rho_c = $(rho_c[i]), rho_r = $(rho_r[i])\n" *
                      "  rho_liq = $(rho_liq[i]), rho_v = $(rho_v[i]), " *
                      "rho_vs = $(rho_vs[i]), Q_ss = $(Q_ss[i]), Q_s = $(Q_s[i])\n" *
                      "  M = $(M[i]), ke = $(ke[i]), E_t = $(E_t[i]), " *
                      "Qdot = $(Qdot[i]), Qdot_r = $(Qdot_r[i])")
            end
            # ── The RAIN DONOR's EVAPORATION CONDUCTANCE (Stage 2b) ──────────────────
            # `κ_ev = max(−Q̇_r, 0)/ρ_r` at the frozen state: the rain's third sink, written
            # on its own reservoir so it can join the donor total `_ice_donor_factors` forms
            # (TeX §donor_relax). CONDENSATION onto rain is a source and contributes nothing,
            # so the `max` is a sign test and not a limiter; below the census floor there is
            # no reservoir to divide by and the conductance is an exact zero, which is what
            # keeps the rain-free path — and the whole dry path — bitwise.
            #
            # The λ/N classification has moved OUT of this loop, because `invtau_r` is not
            # final here. The rain donor's total needs the ice legs, which do not exist until
            # `mc_ice_sources!` has run, and λ, N and the rain transfer must all read the ONE
            # realized conductance. See the fold below the ice block.
            kev = 0.0
            if rain_evap_real && Qdot_r[i] < 0.0 &&
               rho_r[i] > MC_DONOR_QFLOOR * rho_d[i]
                kev = -Qdot_r[i] / rho_r[i]
            end
            kappa_ev[i] = kev
            # The WARM-PATH factor. Evaporation is not ice-gated, so the rain donor has a
            # factor wherever it evaporates whether or not any ice exists; where ice does
            # exist `mc_ice_sources!` overwrites this with the full-conductance one, and with
            # no ice legs the value it writes is bitwise this one. Formed through the same
            # `relaxation_realization(rate, reservoir, dt)` the ice loop calls, on the same
            # `q_r`, so the two expressions cannot drift apart.
            qr_i = max(rho_r[i], 0.0) / rho_d[i]
            f_rain[i] = relaxation_realization(kev * qr_i, qr_i, model.ts)
        end
    else
        fill!(Qdot, 0.0)
        fill!(Qdot_r, 0.0)
        fill!(invtau_c, 0.0)
        fill!(invtau_r, 0.0)
        fill!(kappa_ev, 0.0)
        fill!(f_rain, 1.0)
    end
    # Stiffness of the relaxation this step was asked to integrate, per channel. UNGATED: it
    # is what stands in place of the depletion caps, and the once-per-run warning in
    # `mc_stiffness_trace` has to be able to see an excursion whether or not anyone switched a
    # diagnostic on. Reads the rates; changes none of them.
    mc_stiffness_census!(mtile, model.ts, invtau_c, invtau_r)

    # Warm-rain conversion and sedimentation. Autoconversion + collection move cloud to
    # rain — a liquid-to-liquid exchange, thermodynamically inert (T, p, E_t, Q_ss all
    # unmoved; with the condensate prognostic it is now an equal and opposite pair of
    # sources on slots 9 and 8, and rho_liq — hence the retrieval — does not move). The
    # sedimentation flux F_r = rho_r*Vt (Vt <= 0) moves rain mass AND the energy it
    # carries: e_l(T) + ke + gz per kg of liquid, with e_l = C_pv*T - L_v(T) in the
    # BF02 internal-energy convention. Its divergence sources rho_r, rho_t and E_t with
    # the SAME fitted -dF/dz, so the two densities cannot drift apart, and the column
    # integral telescopes to the boundary fluxes — with a free (Natural) bottom BC on
    # rho_r the surface flux removes rain from the domain. T is invariant under the
    # local exchange (delta_E_t = (e_l+ke+gz)*delta_rho at delta_rho_r = delta_rho_t);
    # the residual flux terms are the physical energy transport by falling rain. No p
    # or Q_ss source: rain exerts no partial pressure, and the (small) sedimentation
    # dT/dt is omitted from the saturation chain rule below (absorbed by the
    # condensation relaxation).
    #
    # Under `rain_moments == 2` this whole closure switches to the ISHMAEL/Morrison
    # exponential-DSD family and gains a NUMBER budget alongside it: `NR_SRC` collects
    # autoconversion's number source, self-collection/breakup and the evaporation number
    # loss, and `Vtn`/`Fnr`/`Fnr_z` are the number's own fall speed, flux and flux
    # divergence, fitted on the n_r column (its own basis and BCs) rather than rho_r's.
    # `Vt` then carries the MASS-weighted `vtrm` in place of `rain_terminal_velocity`, so
    # rho_r, rho_t and E_t all still receive the same single mass flux divergence and the
    # telescoping between them is untouched. Number carries no mass and no energy: `Fnr_z`
    # sources the n_r slot and NOTHING else.
    AUTO_COLL = S.AUTO_COLL
    Fr_z = S.Fr_z
    E_sed_z = S.E_sed_z
    NR_SRC = S.NR_SRC
    Fnr_z = S.Fnr_z
    if precipitation && rain_2m
        Vt = S.Vt
        Vtn = S.Vtn
        for i in 1:length(AUTO_COLL)
            # KK2000 autoconversion (mass + rain number) and accretion (mass only), then
            # Beheng self-collection with Verlinde-Cotton breakup and the evaporation number
            # loss, at the FROZEN `Qdot_r` (the rain donor's realization factor is folded into
            # that column below the ice block, after this loop has read it; the number's own
            # factor `f_nr` realizes the ICE number legs, and this sink stays on the multistep
            # beside self-collection, which carries no conductance either — see
            # `_ice_donor_factors`).
            #
            # It is proportional to the `n`-RATE, not to the step-mean the mass now moves at
            # (the rain condensation/evaporation channel is half of the withheld `Q_ss`
            # relaxation pair). This is the same `O(Δt)` moment-consistency gap the ice habit
            # partition carries, and the same decision: a NUMBER is not mass, vapor or heat,
            # so the exchange closure is untouched, and the alternative — withholding this leg
            # too and giving the number slot its own step-mean increment — buys a second-order
            # correction to a diagnostic partition at the cost of another withheld channel.
            # The mean drop size it sets is bounded by the mass and number it is derived from.
            auto, n_auto = rain_autoconversion_2m(rho_c[i], rho_d[i], max_N_c)
            AUTO_COLL[i] = auto + rain_accretion_2m(rho_c[i], rho_r[i], rho_d[i])
            NR_SRC[i] = n_auto + rain_selfcollection_2m(rho_r[i], n_r[i], rho_d[i]) +
                        rain_number_evaporation_2m(Qdot_r[i], rho_r[i], n_r[i])
            w_m, w_n = rain_fall_speeds_2m(rho_r[i], n_r[i], rho_d[i])
            Vt[i] = w_m
            Vtn[i] = w_n
        end
        Fr = S.Fr;       @. Fr = max(rho_r, 0.0) * Vt
        E_sed = S.E_sed; @. E_sed = Fr * ((Cpv * Tk) - Lv + ke + (gravity * z))
        Fnr = S.Fnr;     @. Fnr = max(n_r, 0.0) * Vtn
        r_col = scratch_column(mtile, 8)
        r_col.uMish .= Fr
        Btransform!(r_col)
        Atransform!(r_col)
        Ixtransform(r_col, Fr_z)
        r_col.uMish .= E_sed
        Btransform!(r_col)
        Atransform!(r_col)
        Ixtransform(r_col, E_sed_z)
        # The number flux on the n_r column: its OWN spline fit and its OWN boundary
        # conditions, so a Natural bottom lets the drops leave the domain with the rain they
        # carry rather than piling up at the ground.
        nr_col = scratch_column(mtile, nr_i)
        nr_col.uMish .= Fnr
        Btransform!(nr_col)
        Atransform!(nr_col)
        Ixtransform(nr_col, Fnr_z)
    elseif precipitation
        for i in 1:length(AUTO_COLL)
            auto = autoconversion_density(max(rho_c[i], 0.0), rho_d[i])
            coll = collection_density(max(rho_c[i], 0.0), rho_r[i], rho_d[i], Tk[i])
            # The PURE conversion rate. This used to carry the joint cloud-depletion cap
            # `min(auto + coll, max(Qdot - min(cap_c, 0), 0))` -- cloud's two sinks,
            # evaporation and conversion to rain, sharing one `rho_c/ts`-sized budget. It went
            # with the rest of the caps: it is a `ts`-dependent modification of a rate, so the
            # equation being solved changed with the step size (see `qss_condensation_rates`).
            # The cap's own justification -- that bounding the two sinks SEPARATELY let them
            # sum to twice the budget -- was an argument for a joint budget over two
            # independent ones, not an argument that either should exist.
            #
            # `water_depletion_probe!` still counts where this would have been pinned
            # (`:d_c_cauto`), so the population the cap used to act on stays visible.
            AUTO_COLL[i] = auto + coll
        end
        Vt = S.Vt;       @. Vt = rain_terminal_velocity(rho_r, rho_d, Tk)
        Fr = S.Fr;       @. Fr = max(rho_r, 0.0) * Vt
        E_sed = S.E_sed; @. E_sed = Fr * ((Cpv * Tk) - Lv + ke + (gravity * z))
        # Fitted flux divergences on rho_r's column basis (its BCs decide whether the
        # surface flux is free to be nonzero), same discrete d/dz as the advection.
        r_col = scratch_column(mtile, 8)
        r_col.uMish .= Fr
        Btransform!(r_col)
        Atransform!(r_col)
        Ixtransform(r_col, Fr_z)
        r_col.uMish .= E_sed
        Btransform!(r_col)
        Atransform!(r_col)
        Ixtransform(r_col, E_sed_z)
    else
        fill!(AUTO_COLL, 0.0)
        fill!(Fr_z, 0.0)
        fill!(E_sed_z, 0.0)
    end
    # The number channels are zeroed OUTSIDE the branch above so that
    # `precipitation && !rain_2m` (which never touches them) and `!precipitation && rain_2m`
    # (advection-only rain number) both leave them at an exact zero. Two `fill!`s on a kDim
    # column; the single-moment path never reads them either way.
    if rain_2m && !precipitation
        fill!(NR_SRC, 0.0)
        fill!(Fnr_z, 0.0)
    end

    # ── Ice: process sources and sedimentation ─────────────────────────────────
    #
    # Each of the twelve slots gets a source accumulator `SRC_*` (the `Q̇_{i,k}`, `Ṅ_k`, `Ȧ_k`,
    # `Ċ_k` of Eqs. ice_prog_mass..ice_prog_c) and a sedimentation flux `F_X = X·W_X` fitted on
    # its OWN spline column, where `W_X` is the fall speed weighted by the very moment being
    # transported — which is what makes the size distribution sort as it falls, and is why the
    # four moments of a species cannot share one flux.
    #
    # ONLY THE MASS FLUXES LEAVE THIS BLOCK. `Fi_z = Σ_k ∂F_{ρ,k}/∂z` is the ice sedimentation
    # rate `Q̇_sed_i` that sources ρ_t, and `E_sed_i` is the energy it carries,
    # `(C_pv T − L_s + ke + gz)` per kg — the ice mirror of the liquid `E_sed`, with `L_s` in
    # place of `L_v` (TeX Eq. Et_ice). The number and volume fluxes carry neither mass nor
    # energy and appear in no other equation, exactly as the rain number's does not.
    Fi_z = S.Fi_z
    E_sed_i_z = S.E_sed_i_z
    if ice_on
        # The ice psychrometric factor (TeX Eq. Qs_ice) — `Q_s_energy` with L_v -> L_s in both
        # latent slots, the two saturation derivatives still over WATER. Formed here rather
        # than beside `Q_s` because nothing but the ice reads it.
        @. S.Q_s_i = Q_s_energy_ice(Tk, p, rho_d, q_v, q_l, q_i)
        # The ANCHOR SHARE column, computed once per column and read by three consumers:
        # the process-rate reads inside `mc_ice_sources!` (rate-side leg, `anchor_rates`),
        # the twelve flux assemblies below (sedimentation leg, `ice_anchor_flx`), and
        # nothing else — the reconciliation source computes its own defect from the same
        # states. Exactly 1.0 wherever the partition is admissible, which is what keeps
        # every leg bitwise absent from healthy air.
        af = S.anchor_f
        @inbounds for i in eachindex(af)
            head = max((rho_t[i] - rho_d[i]) - max(rho_liq[i], 0.0), 0.0)
            ri = rho_ice[i]
            af[i] = (ri > head && ri > 0.0) ? head / ri : 1.0
        end
        # Every process rate: deposition through the shared prognostic supersaturation, the
        # full ISHMAEL nucleation/riming/aggregation/melting set, and the twelve slot sources
        # they assemble into. Reads and writes the scratch only; see `mc_ice_sources!`.
        # `options[:condensation] = false` switches the ice phase changes off with the liquid
        # ones, for the same reason: a run advertised as "no physics" must not glaciate.
        mc_ice_sources!(S, mtile.ishmael_tables, model.ts, max_N_c,
                        get(model.options, :condensation, true)::Bool,
                        get(model.options, :ice_var_check, true)::Bool,
                        ice_anchor_rts, ice_attr, ice_agg_caps, rain_evap_real, ice_shed,
                        ice_n_real, rain_pop_gate,
                        get(model.physical_params, :tau_homogeneous, 5.0),
                        get(model.physical_params, :tau_activation, 1.0),
                        get(model.physical_params, :tau_varcheck, 5.0),
                        mtile.mc_water_stats, Threads.threadid())
        # The three deposition relaxations, on the rates the step actually used. Ungated, for
        # the reason the liquid census is: the once-per-run under-resolution warning has to be
        # able to see an excursion whether or not a diagnostic is switched on.
        mc_stiffness_census!(mtile, model.ts, S.invtau_i1, S.invtau_i2, S.invtau_i3)
        # The ANCHOR RECONCILIATION — the third tier of the chain, after the process sources
        # and before anything reads the SRC accumulators. It measures the partition defect
        # δ_part at every gridpoint (census: MC_ANCHOR_*), and where the advected ice mass
        # exceeds the anchor headroom it relaxes all twelve moments down by one shared
        # fraction δ/(ρ_i·τ_anchor) — per-particle state and species shares exactly
        # invariant, no latent heat, nothing in λ/N. See `ice_anchor_rate` for the law and
        # TeX §"Reconciliation of the condensate partition" for the derivation.
        _ice_anchor_reconcile!(S, mtile.mc_water_stats, Threads.threadid(),
                               rho_t, rho_d, rho_liq, rho_ice,
                               tau_anchor, ice_anchor_src, model.ts)

        # The POPULATION RECONCILIATION — the fourth and last tier, in the same slot and for
        # the same reason as the third: the process sources are assembled and nothing has
        # read them yet. It is the exact complement of the population gate above — mass whose
        # carried number is not positive, which no rate may act on and which no device could
        # therefore remove — returned to a representation the equations can use: rain above
        # `T_0` with `L_f` absorbed through `FRZ_NET`, 2 μm crystals below it, on `τ_pop`.
        # Census `MC_POP_*`; TeX §"Reconciliation of the population".
        # Switched off with the liquid phase changes by `options[:condensation] = false`,
        # for the same reason `mc_ice_sources!` is: BOTH branches move water between
        # categories (the warm one moves mass and `L_f`, the cold one creates crystals), and
        # a run advertised as "no physics" — the transport-only fixtures included — must not.
        # That is the one difference from leg A, which is a pure bookkeeping projection and
        # runs regardless. The CENSUS is unaffected either way.
        _ice_population_reconcile!(S, mtile.mc_water_stats, Threads.threadid(),
                                   Tk, rho_d, tau_pop,
                                   ice_pop_src && get(model.options, :condensation,
                                                      true)::Bool,
                                   model.ts, pop_seed)

        # The SEDIMENTATION ANCHOR SHARE (a leg of the partition reconciliation, TeX
        # §Reconciliation of the condensate partition). At a detached point the slots carry
        # more ice than the anchor holds; letting the phantom part FALL moves real ρ_t and
        # real L_s-laden E_t for mass that was never in the anchor's books — measured
        # pumping ~20 g/m³ of anchor water to the melting level and retrieving 454 K at
        # points whose own partition was admissible. `af` (computed once, above the
        # process-source call) is applied to ALL twelve flux assemblies below, so the ice
        # slots and ρ_t/E_t keep receiving the identical discrete flux (the telescoping is
        # preserved exactly) and what falls keeps its size sorting and mass/number
        # correlation. Exactly 1.0 — bitwise absent — wherever the partition is
        # admissible; the withheld phantom waits in the slot for the reconciliation
        # source. `ice_anchor_flux = false` passes the scalar 1.0 instead (forensics).
        afx = ice_anchor_flx ? af : 1.0
        # Each moment's flux at ITS OWN weighted fall speed — mass-weighted for the mass and
        # the two volume moments, number-weighted for the number. That is the whole reason
        # the four moments of a species cannot share one flux: the distribution SORTS as it
        # falls, and a number that sedimented at the mass-weighted speed would carry the small
        # crystals down as fast as the large ones. `ishmael_fall_speeds` has already applied
        # the 25 m/s cap and the melting size-sorting override.
        _ice_flux!(mtile, S.F_i1q, S.F_i1q_z, i1q, S.Vi1m, afx, IS.i1_q)
        _ice_flux!(mtile, S.F_i1n, S.F_i1n_z, i1n, S.Vi1n, afx, IS.i1_n)
        _ice_flux!(mtile, S.F_i1a, S.F_i1a_z, i1a, S.Vi1m, afx, IS.i1_a)
        _ice_flux!(mtile, S.F_i1c, S.F_i1c_z, i1c, S.Vi1m, afx, IS.i1_c)
        _ice_flux!(mtile, S.F_i2q, S.F_i2q_z, i2q, S.Vi2m, afx, IS.i2_q)
        _ice_flux!(mtile, S.F_i2n, S.F_i2n_z, i2n, S.Vi2n, afx, IS.i2_n)
        _ice_flux!(mtile, S.F_i2a, S.F_i2a_z, i2a, S.Vi2m, afx, IS.i2_a)
        _ice_flux!(mtile, S.F_i2c, S.F_i2c_z, i2c, S.Vi2m, afx, IS.i2_c)
        _ice_flux!(mtile, S.F_i3q, S.F_i3q_z, i3q, S.Vi3m, afx, IS.i3_q)
        _ice_flux!(mtile, S.F_i3n, S.F_i3n_z, i3n, S.Vi3n, afx, IS.i3_n)
        _ice_flux!(mtile, S.F_i3a, S.F_i3a_z, i3a, S.Vi3m, afx, IS.i3_a)
        _ice_flux!(mtile, S.F_i3c, S.F_i3c_z, i3c, S.Vi3m, afx, IS.i3_c)

        # The two AGGREGATES that reach the conserved variables. `Fi_z` is summed from the
        # three fitted mass-flux divergences rather than refitting their sum, so slot 3 and
        # the three ice mass slots receive exactly the same discrete numbers and cannot drift
        # apart — the telescoping argument slot 8 and slot 3 make for the rain.
        @. Fi_z = (S.F_i1q_z + S.F_i2q_z) + S.F_i3q_z
        E_sed_i = S.E_sed_i
        @. E_sed_i = ((S.F_i1q + S.F_i2q) + S.F_i3q) *
                     ((Cpv * Tk) - L_s(Tk) + ke + (gravity * z))
        # Fitted on species 1's mass column: it is a MASS flux and takes a mass slot's basis
        # and BCs (Natural at the bottom, so the energy can leave with the ice).
        ei_col = scratch_column(mtile, IS.i1_q)
        ei_col.uMish .= E_sed_i
        Btransform!(ei_col)
        Atransform!(ei_col)
        Ixtransform(ei_col, E_sed_i_z)
    end

    # ── The REALIZED RAIN CONDUCTANCE, and the liquid λ/N split taken on it (Stage 2b) ────
    #
    # HERE, and not beside the condensation closure where it used to be, because `f_rain` is
    # not final until `mc_ice_sources!` has completed the rain donor's total: the reservoir's
    # sinks are the ice legs AND the evaporation, and the TeX's closure is that ONE realized
    # conductance `f_r τ_r^{-1}` replaces `τ_r^{-1}` in λ, in N and in the rain-channel
    # transfer at once (§donor_relax, "The rain reservoir has a third sink"). Nothing between
    # the closure and this point reads either column except `mc_stiffness_census!`, which
    # measures the stiffness the step was ASKED to integrate and therefore wants the
    # unrealized one.
    #
    # The factor is folded into `invtau_r` and `Qdot_r` THEMSELVES, exactly as the ice folds
    # `f_ice<k>` into `invtau_i<k>`, so every consumer downstream — λ, the `N` remainder just
    # below, the step-mean split in the ETD pre-compute and the `qr_bar` the rain slot
    # receives — reads the one number with no opportunity to disagree. `f_rain` is an exact
    # `1.0` wherever the rain has no sink, and `1.0 * x === x` for every `x`, so the warm
    # non-evaporating path and the whole dry path are untouched bitwise.
    #
    # The split itself is UNCHANGED, term for term: it is a statement about the DRIVE CLIP
    # and nothing else (see the block beside the closure). A clipped channel's drive does not
    # depend on `Q_ss`, so its whole — now realized — flux belongs in `N`; an unclipped one's
    # whole realized conductance belongs in λ.
    #
    # `evap_now` is the exact mirror of the ice's `sub_now`, and for the same reason: a donor
    # factor bounds a SINK, and rain CONDENSATION is a source. Where the channel is condensing
    # at state `n` the factor does not belong in its conductance at all — `κ_ev` is already an
    # exact zero there by its own `max`, but the rain donor's factor also carries the ice legs,
    # and scaling a source by the rate at which freezing empties the reservoir would be a
    # bound applied to the wrong sign. FROZEN at state `n`, like every classification the step
    # freezes.
    if get(model.options, :condensation, true)::Bool
        @inbounds for i in eachindex(invtau_r)
            evap_now = Qdot_r[i] < 0.0
            fr = f_rain[i]
            invtau_r[i] = evap_now ? fr * invtau_r[i] : invtau_r[i]
            Qdot_r[i] = evap_now ? fr * Qdot_r[i] : Qdot_r[i]
            invtau = invtau_c[i] + invtau_r[i]
            drive_cap = max(rho_v[i], 0.0) - rho_vs[i]
            if drive_cap < Q_ss[i]
                etd_lam[i] = 0.0
                etd_nl[i] = -drive_cap * invtau
            else
                etd_lam[i] = invtau
                etd_nl[i] = 0.0
            end
        end
    else
        fill!(etd_lam, 0.0)
        fill!(etd_nl, 0.0)
    end

    # Record this step's net microphysics tendency per channel, so the NEXT step's depletion
    # budget can be written against the integrator's actual three-level combination. Here,
    # after the `precipitation` branch, because the cloud channel is not final until
    # AUTO_COLL is. See `MC_MICRO_C` for the three channels and `_rotate_micro_history!` for
    # when these become the history levels.
    #
    # The ICE arm is a SEPARATE loop, so the ice-free path keeps the short body it had.
    # The cloud and rain channels gain the riming/freezing/melting back-reactions, which is
    # what makes the census (still a pure measurement — nothing in the RHS reads it) describe
    # the water budget that ran.
    #
    # WHAT MOVED WHEN THE RELAXATION PAIR CAME OFF THE MULTISTEP. The condensation and
    # deposition legs are no longer weighted by the three-level combination at all — they are
    # applied once, at the step-mean rate, by `relaxation_adjustment_qss!` — so keeping them in
    # a MULTISTEP history would have made the census weight them `23/12, −16/12, 5/12` when the
    # integrator weights them `1, 0, 0`. These three channels therefore carry the legs that are
    # still on the multistep (autoconversion/collection and the ice back-reactions), and the
    # phase change reaches the census through `Qdot_bar`/`Qdot_r_bar`/`Qdep_bar` instead —
    # the SAME columns the tendencies are built from, which is the property this block exists
    # to guarantee. `water_depletion_probe!` adds the two together.
    micro_n = mtile.mc_micro_n
    @inbounds for i in eachindex(Qdot)
        g = colstart + i - 1
        micro_n[g, MC_MICRO_C] = -AUTO_COLL[i]
        micro_n[g, MC_MICRO_R] = 0.0
        micro_n[g, MC_MICRO_V] = 0.0
    end
    if ice_on
        @inbounds for i in eachindex(Qdot)
            g = colstart + i - 1
            micro_n[g, MC_MICRO_C] += S.ICE_C[i]
            micro_n[g, MC_MICRO_R] += S.ICE_R[i]
        end
    end

    div = S.div; mc_divergence!(div, geom, u, u_x, w_z, vv, r)

    # ── Horizontal diffusion ───────────────────────────────────────────────────
    # Turbulence diffuses momentum (u, w) and the moist entropy s_t (heat). Horizontally,
    # s_t is diffused by the chain rule on the DRY-exact form s_d = C_vd ln p - C_pd ln rho_d
    # (an explicit function of the prognostic p, rho_d). Its Laplacian uses only the p/rho_d
    # derivative slots — no transform of a diagnosed field, which the column decomposition
    # cannot do (see reference/moist_compressible_diffusion_plan.md). The MOIST horizontal
    # correction (through the retrieval's T sensitivities) is deferred to the rainfall
    # session (reference/moist_compressible_diffusion_handoff.md). pbar/rho_dbar have no
    # x-dependence, so the total x-derivatives are the perturbation slots.
    sd_xx = S.sd_xx
    mc_sd_lap!(sd_xx, geom, p, rho_d, pv, rdv, r)

    # Smagorinsky horizontal eddy viscosity (Ls > 0): flow-dependent K(strain)
    # replaces the constant Khdiff in the momentum diffusion and its FRIC_KE
    # energy sink below. Computed BEFORE the heat diffusion because heat can be
    # tied to it (the Khdiff_heat < 0 sentinel below).
    K_smag = S.K_smag
    if use_smag
        mc_smag_k!(K_smag, geom, uv, vv, r, Ls, K_min)
    end

    # Horizontal diabatic heating [W/m^3] from the entropy diffusion: the source to internal
    # energy is rho_d*T*(ds_t/dt)_diff = rho_d*T*K_heat*d2(s_t)/dx2. This is the
    # moist analogue of Straka's rho*T*ds_d source.
    #
    # Khdiff_heat < 0 is the SENTINEL (as with Cd < 0 for the wind-dependent drag) for
    # "mix heat with the Smagorinsky eddy diffusivity K_smag/Pr_t" instead of a constant.
    # Without it, a Smagorinsky run mixes MOMENTUM only and leaves the thermodynamic
    # fields with no horizontal mixing whatsoever -- which on an axisymmetric grid is
    # especially bad, since there are no asymmetries to provide radial mixing and the
    # strong radial gradients of a TC then support undamped grid-scale buoyancy
    # structure. Tying it to K_smag rather than a constant keeps it resolution-general
    # (K scales with the resolved deformation, so it follows a nest refinement).
    # NOTE the form is K*grad^2(s), not div(K grad s): the variable-K correction
    # grad(K).grad(s) is dropped, exactly as the existing momentum diffusion does with
    # K_smag (mc_u_kdiff!) -- consistent with the surrounding scheme, not a new
    # approximation.
    QDOT_TH = S.QDOT_TH
    if Khdiff_heat < 0.0
        use_smag || error("physical_params[:Khdiff_heat] < 0 selects the Smagorinsky " *
                          "heat diffusivity, which requires Ls > 0")
        @. QDOT_TH = rho_d * Tk * (K_smag / Pr_t) * sd_xx
    else
        @. QDOT_TH = rho_d * Tk * Khdiff_heat * sd_xx
    end

    # RADIATION enters the thermal source and nothing else. `QDOT_TH` is the single point
    # where a diabatic heating reaches the pressure equation (slot 1, through R_m/C_vt),
    # the total energy (slot 6, directly) and the supersaturation forcing (slot 7, through
    # the non-condensational dT_nc/dp_nc that feed SATF) -- which is exactly the set of
    # places radiative cooling has to reach: it creates supersaturation, and a radiation
    # term that skipped Q_ss would cool the air without making the cloud that cooling makes.
    # The Louis boundary layer adds its heating at the same place for the same reason.
    #
    # The field is HELD between radiation calls (the pre-pass owns the cadence), so this is
    # an indexed add and nothing more. `q_lw`/`q_sw` are gridpoint-indexed exactly like
    # `expdot`, hence `colstart + i - 1` and no reshape. The shortwave carries `sw_scale`,
    # the per-step zenith rescale that keeps a diurnal forcing continuous between calls
    # (plan D6); it is exactly 1 with a fixed sun and 0 at night.
    if rad_on
        q_lw = RAD.q_lw; q_sw = RAD.q_sw; sw_s = RAD.sw_scale
        @inbounds for i in eachindex(QDOT_TH)
            QDOT_TH[i] += q_lw[colstart + i - 1] + sw_s * q_sw[colstart + i - 1]
        end
    end

    # Horizontal frictional KE change [W/m^3]. Momentum diffusion is a resolved-KE SINK to
    # the subgrid (the future TKE shear production), NOT dissipative heating: with an eddy K
    # the resolved KE lost goes to unresolved scales, and the molecular heating (proportional
    # to the far-smaller kinematic viscosity) is negligible. So E_t follows the KE down and
    # internal energy (T, p) is held. FRIC_KE is d(rho_t*ke)/dt from horizontal momentum
    # diffusion, added to E_t; p and Q_ss get NO friction term.
    FRIC_KE = S.FRIC_KE
    if use_smag
        mc_fric_ke!(FRIC_KE, geom, rho_t, K_smag, uv, wv, vv, r)
    else
        mc_fric_ke!(FRIC_KE, geom, rho_t, Khdiff, uv, wv, vv, r)
    end

    # Placeholders for intermediate calculations
    ADV = S.ADV
    FORCING = S.FORCING
    KDIFF = S.KDIFF

    # Pressure (slot 1): -v·∇p - γp∇·v + (R_m/C_vt)[(L_v - R_v C_pt T/R_m) Q̇_cond + Q̇_therm].
    # Q̇_cond is the TOTAL phase-change rate (cloud + rain channels); only the THERMAL
    # diffusion sources pressure (friction holds T, hence p; sedimentation moves no
    # partial pressure).
    #
    # The acoustic slots (1, 2, 3, 5, 6) stage the REMAINDER: the full tendency minus the
    # reference-linear vertical acoustic term in the same pointwise product-rule form, so
    # the grid-scale vertical-acoustic content cancels analytically and what AB3 advances
    # is O(perturbation). The linear part is integrated by the AI2* acoustic solve alone
    # (see semiimplicit_adjustment_p and the impdot staging block below).
    mc_advect!(ADV, geom, u, w, vv, r, pp_x, p_z, pv.f_l)
    # sd_si: the added-back linear term uses the same frozen-at-n state
    # coefficients the implicit solve applies (Pξⁿ, ρ_tⁿ and its FULL vertical
    # gradient), so the grid-scale acoustic cancellation holds at any
    # perturbation amplitude — the point of the state-dependent linearization.
    #
    # THE CONDENSATION HEATING IS NOT HERE ANY MORE. `(L_v − R_v C_pt T/R_m)·Q̇_cond` is
    # proportional to a rate that the stiff-relaxation integrator withholds from the multistep,
    # so it returns as a DIRECT increment at the step-mean rate, carrying this same bracket,
    # in `relaxation_adjustment_qss!` — which runs before the acoustic solve, so the corrected
    # latent heating still enters that solve at exactly the point the rate-level heating
    # entered it before (TeX §stiff_integrator, "Placement in the step"). Only the THERMAL
    # diffusion source is left in the bracket here.
    if sd_si
        FORCING .= @. (-gamma_m * p * div) + (S.sd_pxi * ((rho_t * w_z) + (rho_t_z * w))) +
                      ((R_m / C_vt) * QDOT_TH)
    else
        FORCING .= @. (-gamma_m * p * div) + (Pxi_bar * ((rho_tbar * w_z) + (rho_tbar_z * w))) +
                      ((R_m / C_vt) * QDOT_TH)
    end
    @turbo expdot[colstart:colend,1] .= @. ADV + FORCING
    # The ICE phase changes, inside the same `(R_m/C_vt)[...]` bracket (TeX Eq. thermo_p_ice).
    # Deposition's coefficient is condensation's with `L_v -> L_s` and NOTHING else; freezing
    # carries the bare `L_f` with NO `-R_v C_pt T/R_m` companion, because that companion is the
    # work done by the vapor the phase change removes from the gas phase and freezing removes
    # none. Accumulated separately, so the ice-free `@turbo` expression above stays
    # byte-identical (the slot-3 rule); explicit loop, so the indexed update does not
    # materialize a column (the slot-3 rule again).
    # FREEZING stays on the multistep and DEPOSITION does not, which is the whole difference
    # between the two: `Q̇_freeze` relaxes nothing and moves no vapor, while `Q̇_dep` is one of
    # the two channels that relax `Q_ss`. So only the bare `L_f` term is accumulated here; the
    # `(L_s − R_v C_pt T/R_m)·Q̇_dep` companion arrives with the condensation heating as a
    # direct increment.
    if ice_on
        @inbounds for i in eachindex(Fi_z)
            expdot[colstart + i - 1, 1] +=
                (R_m[i] / C_vt[i]) * (L_f(Tk[i]) * S.FRZ_NET[i])
        end
    end

    # Dry-air mass continuity (slot 2, advective product-rule form; no mass diffusion)
    mc_advect!(ADV, geom, u, w, vv, r, rho_dp_x, rho_d_z, rdv.f_l)
    if sd_si
        @turbo FORCING .= @. (-rho_d * div) + ((rho_d * w_z) + (rho_d_z * w))
    else
        @turbo FORCING .= @. (-rho_d * div) + ((rho_dbar * w_z) + (rho_dbar_z * w))
    end
    @turbo expdot[colstart:colend,2] .= @. ADV + FORCING

    # Total mass continuity (slot 3): the only source is the sedimentation flux
    # divergence, identical to slot 8's so rain and total water cannot drift apart.
    mc_advect!(ADV, geom, u, w, vv, r, rho_tp_x, rho_t_z, rtv.f_l)
    if sd_si
        @turbo FORCING .= @. (-rho_t * div) + ((rho_t * w_z) + (rho_t_z * w)) - Fr_z
    else
        @turbo FORCING .= @. (-rho_t * div) + ((rho_tbar * w_z) + (rho_tbar_z * w)) - Fr_z
    end
    @turbo expdot[colstart:colend,3] .= @. ADV + FORCING
    # The ICE sedimentation rate `Q̇_sed_i = Σ_k ∂F_{ρ,k}/∂z`, the second mass flux out of the
    # total water (TeX Eq. ice_mass). Applied as a SEPARATE accumulation rather than folded
    # into the expression above, so the ice-free path's `@turbo` expression is untouched
    # byte-for-byte and the default run cannot move by an instruction-selection accident.
    # Explicit loop, not `expdot[colstart:colend,3] .-= Fi_z`: an indexed broadcast-update
    # reads its own target through `getindex`, which materializes a fresh column every call
    # (measured: 2 allocations per line). Same reason the horizontal water mixing and the
    # Louis BL write their additions this way.
    if ice_on
        @inbounds for i in eachindex(Fi_z)
            expdot[colstart + i - 1, 3] -= Fi_z[i]
        end
    end

    # u momentum (slot 4): PGF directly from the prognostic pressure
    mc_advect!(ADV, geom, u, w, vv, r, u_x, u_z, uv.f_l)
    mc_u_forcing!(FORCING, geom, pp_x, rho_t, vv, r, fcor)
    if use_smag
        mc_u_kdiff!(KDIFF, geom, K_smag, uv, vv, r)
    else
        mc_u_kdiff!(KDIFF, geom, Khdiff, uv, vv, r)
    end
    @turbo expdot[colstart:colend,4] .= @. ADV + FORCING + KDIFF

    # w momentum (slot 5): perturbation PGF + total-density buoyancy loading, minus the
    # reference-linear PGF -pp_z/ρ̄_t (the acoustic remainder; see the slot-1 comment)
    mc_advect!(ADV, geom, u, w, vv, r, w_x, w_z, wv.f_l)
    if sd_si
        # With ρ̂_t = ρ_tⁿ the linear PGF −pp_z/ρ̂_t cancels the full
        # perturbation PGF exactly: the explicit w remainder is pure buoyancy.
        @turbo FORCING .= @. (-gravity * rho_tp) / rho_t
    else
        @turbo FORCING .= @. (((-gravity * rho_tp) - pp_z) / rho_t) + (pp_z / rho_tbar)
    end
    if use_smag
        mc_w_kdiff!(KDIFF, geom, K_smag, wv, r)
    else
        mc_w_kdiff!(KDIFF, geom, Khdiff, wv, r)
    end
    @turbo expdot[colstart:colend,5] .= @. ADV + FORCING + KDIFF

    # Total energy (slot 6): -v·∇E_t - E_t∇·v - ∇·(pv) - ∇·(E_sed) + Q̇_therm + friction
    # sink. No condensation source (exact first law: phase change is not an energy source).
    # E_sed is the energy carried by falling rain (see the microphysics block above). The
    # THERMAL diffusion heats (Q̇_therm); the momentum diffusion adds FRIC_KE = d(rho_t*ke)/dt
    # so E_t follows the resolved KE down to the subgrid (internal energy held). pbar has no
    # x-dependence so u*p_x = u*pp_x.
    mc_advect!(ADV, geom, u, w, vv, r, E_tp_x, E_t_z, etv.f_l)
    mc_et_work!(FORCING, geom, E_t, p, div, u, pp_x, w, p_z, E_sed_z, pv, vv, r)
    if sd_si
        @turbo expdot[colstart:colend,6] .= @. ADV + FORCING + QDOT_TH + FRIC_KE +
                                               (((E_t + p) * w_z) + ((E_t_z + p_z) * w))
    else
        @turbo expdot[colstart:colend,6] .= @. ADV + FORCING + QDOT_TH + FRIC_KE +
                                               (((E_tbar + pbar) * w_z) + ((E_tbar_z + pbar_z) * w))
    end
    # The energy the falling ICE carries out of the parcel, `(C_pv T − L_s + ke + gz)` per kg
    # of ice (TeX Eq. Et_ice) — the ice mirror of `E_sed_z`, which `mc_et_work!` already
    # applied for the liquid, and with the same sign. Separate accumulation for the same
    # reason slot 3's is: the ice-free `@turbo` expressions above stay byte-identical.
    if ice_on
        @inbounds for i in eachindex(E_sed_i_z)      # explicit loop; see the slot-3 note
            expdot[colstart + i - 1, 6] -= E_sed_i_z[i]
        end
    end

    # ── AI2* acoustic history staging ──
    # The linear vertical acoustic operator, evaluated on the CURRENT state with the same
    # discrete chain the semi-implicit solve applies: fit φⁿ = ρ̄_t wⁿ once in w's column
    # basis (the basis the Helmholtz solve works in, Dirichlet rows included), take its
    # spline derivative, and form each slot's tendency with the same coefficient profiles
    # the slaved updates use. Every AI2* time level then sees the SAME discrete operator —
    # the pointwise product-rule staging used previously left the grid-scale difference
    # between the two operators under explicit weights, which imposed a vertical-acoustic
    # Courant ceiling (see reference/SI_VERTICAL_CEILING.md). The single-φ-fit structure also
    # keeps the rho_d and rho_t histories bitwise identical under a dry reference
    # (c_d = 1, c_d_z = 0 exactly), so the two densities cannot drift apart in dry air.
    # Fresh evaluation every step (not stored solve increments) makes the history robust
    # to whatever touches the state between steps (implicit diffusion, nesting collar
    # injection, the spectral refit).
    w_col = scratch_column(mtile, 5)
    # sd_si: φⁿ = ρ_tⁿ·w and every coefficient chain from the CURRENT column
    # state (frozen at n), matching the operator the adjustment's per-column
    # state-dependent solve applies; else the resting-reference profiles.
    if sd_si
        w_col.uMish .= rho_t .* w
    else
        w_col.uMish .= rho_tbar .* w
    end
    Btransform!(w_col)
    Atransform!(w_col)
    phi_n = Itransform!(w_col)
    imp_phi_z = S.imp_phi_z
    Ixtransform(w_col, imp_phi_z)

    imp_c_d = S.imp_c_d
    imp_c_d_z = S.imp_c_d_z
    imp_c_e = S.imp_c_e
    imp_c_e_z = S.imp_c_e_z
    if sd_si
        @. imp_c_d = rho_d / rho_t
        @. imp_c_d_z = ((rho_d_z * rho_t) - (rho_d * rho_t_z)) / (rho_t^2)
        @. imp_c_e = (E_t + p) / rho_t
        @. imp_c_e_z = (((E_t_z + p_z) * rho_t) - ((E_t + p) * rho_t_z)) / (rho_t^2)
        impdot[colstart:colend,1] .= @. -S.sd_pxi * imp_phi_z
    else
        @. imp_c_d = rho_dbar / rho_tbar
        @. imp_c_d_z = ((rho_dbar_z * rho_tbar) - (rho_dbar * rho_tbar_z)) / (rho_tbar^2)
        @. imp_c_e = (E_tbar + pbar) / rho_tbar
        @. imp_c_e_z = (((E_tbar_z + pbar_z) * rho_tbar) -
                        ((E_tbar + pbar) * rho_tbar_z)) / (rho_tbar^2)
        impdot[colstart:colend,1] .= @. -Pxi_bar * imp_phi_z
    end
    impdot[colstart:colend,2] .= @. -((imp_c_d * imp_phi_z) + (imp_c_d_z * phi_n))
    impdot[colstart:colend,3] .= @. -imp_phi_z
    impdot[colstart:colend,6] .= @. -((imp_c_e * imp_phi_z) + (imp_c_e_z * phi_n))
    # w's history is NOT staged here: the Helmholtz elimination's effective operator on
    # the w leg is not expressible as a pointwise chain on the state, so the adjustment
    # stores the increment it actually applied ((w_np1 - w*)/Δτ) as the next step's
    # history — the only exact mirror. Staging -pp_z/ρ̄_t here instead leaves the
    # weak-Galerkin elimination residual under the explicit AI2* weights (vertical
    # Courant ceiling at Co_z ≈ 2-3, measured).

    # ── Horizontal acoustic remainder + AI2* history staging (horizontal SI) ──
    # The horizontal analogue of the vertical staging above (see horizontal_si.jl):
    # the remainder additions cancel the reference-linear horizontal legs out of
    # the AB3 predictor, and the fresh histories are the same legs evaluated on
    # the carried state through the grid slots (u_x IS the spline-chain derivative
    # of the post-refit state, and the reference coefficients are z-only, so no
    # fit and no product-rule chain is needed). u's history is NOT staged here —
    # it is the stored applied increment of the patch-level sweep
    # (horizontal_si_load_increment!, the w-leg discipline).
    if hsi_like
        LDIV = S.ADV                     # free between slot 6 and slot 7
        mc_linear_div!(LDIV, geom, uv, vv, r)
        @turbo expdot[colstart:colend,1] .+= @. Pxi_bar * rho_tbar * LDIV
        @turbo expdot[colstart:colend,2] .+= @. rho_dbar * LDIV
        @turbo expdot[colstart:colend,3] .+= @. rho_tbar * LDIV
        @turbo expdot[colstart:colend,4] .+= @. pp_x / rho_tbar
        @turbo expdot[colstart:colend,6] .+= @. (E_tbar + pbar) * LDIV
        # Fresh history staging only under options[:hsi_x_history] = "fresh"
        # (the A/B alternative): the default "stored" histories are the
        # sweep's applied increments, loaded into hacdot_n at the top of the
        # step (horizontal_si_load_increment!) — writing here would clobber
        # them. See the load function for the measured trade-offs.
        if get(model.options, :hsi_x_history, "fresh") == "fresh"
            hacdot = mtile.hacdot_n
            hacdot[colstart:colend,1] .= @. -Pxi_bar * rho_tbar * LDIV
            hacdot[colstart:colend,2] .= @. -rho_dbar * LDIV
            hacdot[colstart:colend,3] .= @. -rho_tbar * LDIV
            hacdot[colstart:colend,6] .= @. -(E_tbar + pbar) * LDIV
        end
    end

    # Supersaturation density (slot 7): the saturation chain-rule terms use the
    # non-condensation T and p tendencies, which carry the horizontal THERMAL diffusive
    # heating as well as the divergence work (friction holds T, so it does not enter; the
    # VERTICAL diffusive heating is applied as a slaved δQ_ss inside diffusion_timestep_mc).
    # The condensation contribution is -Q̇_cond(1+Q_s) (= -Q_ss/τ when the rate is unlimited).
    # QSSREL reconciles the prognostic Q_ss with the supersaturation the prognostic VAPOR
    # implies (see qss_relaxation) — Q_ss is redundant now that every water species is
    # prognostic, and this is what keeps the two from drifting apart under splitting error.
    #
    # IT IS FED THE PROGNOSTIC `rho_v`. That was forbidden while the vapor was a retrieval —
    # a blended vapor built partly OUT OF `Q_ss + rho_vs` made this term read
    # `-(Q_ss - ((Q_ss + rho_vs) - rho_vs))/tau ≡ 0` and SELF-ANNIHILATE — but the hazard was
    # algebraic, not physical, and it is gone with the retrieval: a transported slot is not
    # `Q_ss + rho_vs` by construction, so `Q_ss - (rho_v - rho_vs)` is a genuine disagreement
    # between two independently carried fields.
    #
    # The reconciliation is now a CHAIN, and each link is a drift correction on its own slow
    # timescale rather than a shock:
    #
    #     Q_ss  --(tau_qss)-->  rho_v  --(tau_rec)-->  the rho_t density budget
    #
    # Q_ss is pulled onto the supersaturation the prognostic vapor implies (here), and the
    # prognostic vapor is pulled onto the vapor the conserved masses imply (`rho_v_reconcile`,
    # at the vapor slot below). rho_t is the anchor of the chain and is never nudged.
    #
    # FREEZING enters here and ONLY here on the chain-rule side. It moves no vapor, so it has
    # no `-Q̇` of its own and no `(1 + 𝒬)` grouping to cancel against; it reaches Q_ss purely
    # through the temperature and pressure it changes, which is why `L_f·Q̇_freeze` belongs
    # inside BOTH non-condensation tendencies (TeX Eqs. dThat_ice, dphat_ice) rather than
    # beside the relaxations. Physically: freezing in a mixed-phase cloud warms the air,
    # raises ρ_v*, and drives the LIQUID subsaturated — the mechanism by which riming
    # glaciates a cloud from the inside with no vapor having moved.
    dT_nc = S.dT_nc; @. dT_nc = ((-p * div) + QDOT_TH) / (rho_d * C_vt)
    dp_nc = S.dp_nc; @. dp_nc = (-gamma_m * p * div) + ((R_m / C_vt) * QDOT_TH)
    if ice_on
        @. dT_nc += (L_f(Tk) * S.FRZ_NET) / (rho_d * C_vt)
        @. dp_nc += (R_m / C_vt) * (L_f(Tk) * S.FRZ_NET)
    end
    SATF = S.SATF;   @. SATF = (-rho_vs * div) - (drvs_dT * dT_nc) - (drvs_dp * dp_nc)
    QSSREL = S.QSSREL
    @. QSSREL = qss_relaxation(Q_ss, rho_v, rho_vs, tau_qss)
    #
    # WHAT THIS SLOT'S `expdot` NOW CARRIES: `N`, not the tendency. The two relaxations
    # `−Q_ss/τ` and `−(Q_ss + 𝒟)/τ_i` are withheld and integrated by their own exact
    # propagator (TeX Eq. etd_ab3; `relaxation_adjustment_qss!`), so what is assembled here —
    # advection, the divergence forcing, the saturation chain rule, the flux terms and the
    # `qss_relaxation` reconciliation — is exactly the `F` of Eq. relax_linear. `τ_ss` is never
    # stiff, so `QSSREL` stays in `N` where the TeX puts it. The history rotates as usual and
    # supplies `N^{n−1}` and `N^{n−2}` at no cost, which is the whole reason the withholding is
    # done here rather than by unwinding the relaxation after the fact.
    mc_advect!(ADV, geom, u, w, vv, r, Q_ssp_x, Q_ss_z, qsv.f_l)
    FORCING .= @. (-Q_ss * div) + SATF + QSSREL
    @turbo expdot[colstart:colend,7] .= @. ADV + FORCING
    # The `Q`-INDEPENDENT remainder of the withheld LIQUID relaxation, which belongs in `N`:
    # zero wherever the drive is unclipped (there the whole term is `−λ Q_ss`), and the full
    # constant flux `−drive·(τ_c^{-1} + τ_r^{-1})` wherever it is clipped. Gated on an exact
    # nonzero so a point with nothing to add is not touched at all — `x + (±0.0)` is not the
    # identity for `x = ∓0.0`, and the dry path's bitwise gate rests on it. Explicit loop, not
    # a broadcast update: see the slot-3 note.
    @inbounds for i in eachindex(etd_nl)
        etd_nl[i] != 0.0 && (expdot[colstart + i - 1, 7] += etd_nl[i])
    end
    # The DEPOSITION relaxation, `-Σ_k Q̇_{i,k}(1 + 𝒬_{s,i})` (TeX Eq. Qss_ice). Substituting
    # Eq. dep_rate, the `(1 + 𝒬_{s,i})` cancels exactly as its liquid counterpart does and
    # what is left is `-(Q_ss + 𝒟)/τ_i`: the ice contributes one further relaxation of the
    # SAME shared supersaturation and no psychrometric factor survives into the prognostic
    # equation. That shared reservoir is what makes the Wegener-Bergeron-Findeisen competition
    # (Eq. wbf_qs) emergent rather than arbitrated — and is why ISHMAEL's mixed-phase
    # deposition cap is dropped (TeX §Departures (a)).
    #
    # Withheld with its liquid partner, and for the same reason: `−(Q_ss + 𝒟)/τ_i` is the
    # STIFF half of the pair. What is added back here is only its `Q`-independent piece,
    # `−b·Σ_k τ_{i,k}^{-1}` with `b` the constant term of the affine drive decomposition
    # (`mc_ice_sources!`): `𝒟 Σ τ_{i,k}^{-1}` where the drive is unclipped — the offset that
    # makes the ice relax the shared reservoir toward a different zero than the liquid, i.e.
    # the Wegener-Bergeron-Findeisen term of Eq. wbf_qs — and the whole constant flux where it
    # is clipped.
    if ice_on
        @inbounds for i in eachindex(Fi_z)
            n_i = -S.etd_dep_b[i] *
                  ((S.invtau_i1[i] + S.invtau_i2[i]) + S.invtau_i3[i])
            n_i != 0.0 && (expdot[colstart + i - 1, 7] += n_i)
        end
    end

    # Rain partial density (slot 8): rain-channel condensation/evaporation,
    # autoconversion + collection from cloud, and the sedimentation flux divergence
    # (no diffusion yet — water-species mixing arrives with the moist diffusion).
    #
    # Under `rain_transform_mode`, exactly as for slot 9: advection is transform-invariant and
    # reads the control variable's own fitted gradients (`nu_r_z` is bitwise `rho_rp_z` under
    # `:none`), while divergence and the sources pick up `Jr`, an exact 1.0 under `:none`.
    #
    # The sedimentation term is the one that has no cloud precedent. `-Fr_z` is a DENSITY flux
    # divergence, formed from the recovered density on slot 8's own spline column and BCs, and
    # slot 3 (rho_t) and slot 6 (E_t) below continue to receive it untransformed — they are
    # still a density and an energy. Only this slot, which is no longer a density, carries it
    # through `Jr`. That is correct term by term, but it does mean slot 8 and slot 3 stop
    # receiving the identical discrete number, so the exact telescoping between them is
    # weakened. The independent check is already in the diagnostics: `accum_rainfall_mm` comes
    # from the rho_t - rho_d water path and `accum_rainfall_flux_mm` from this surface flux.
    #
    # `Qdot_r` is withheld here with the rest of the relaxation pair and returns through `Jr`
    # as a direct increment at the step-mean rate; autoconversion, collection and sedimentation
    # relax nothing and keep the multistep.
    mc_advect!(ADV, geom, u, w, vv, r, rho_rp_x, nu_r_z, rrv.f_l)
    @turbo FORCING .= @. Jr * ((-rho_r * div) + AUTO_COLL - Fr_z)
    @turbo expdot[colstart:colend,8] .= @. ADV + FORCING
    # The ICE back-reaction on the rain: riming and Bigg/homogeneous freezing remove rain,
    # melting ice returns it. Through `Jr`, like every other source on this slot; separate
    # accumulation and explicit loop for the reasons slot 3 states.
    if ice_on
        @inbounds for i in eachindex(Fi_z)
            expdot[colstart + i - 1, 8] += Jr[i] * S.ICE_R[i]
        end
    end
    # Production attribution (off unless options[:water_budget_trace] > 0). Must run HERE:
    # ADV is reused by slot 9 two lines down.
    budget_trace && water_budget_probe!(mtile, MC_BUDGET_R, 8, t, colstart, rho_r, ADV, div,
                                        Qdot_r, AUTO_COLL, 1.0, Fr_z, w, z)

    # Cloud partial density (slot 9): the same advective product-rule continuity as every
    # other density, with the cloud-channel condensation as its source and autoconversion
    # + collection as its sink (the equal and opposite pair of slot 8's AUTO_COLL). No
    # sedimentation — cloud droplets do not fall in this scheme — and no rho_t source:
    # condensation is INTERNAL to the water, moving mass between this slot and the
    # prognostic vapor slot at fixed total density.
    #
    # This is the slot the whole formulation exists for. Because it is prognostic rather
    # than a residual of rho_t - rho_d - rho_v - rho_r, cloud in subsaturated air can only
    # arrive by advection or nucleation, and fit-level error in the density fields lands
    # in the vapor (where it is 5e-5 of the field) instead of manufacturing condensate
    # (where it was O(1) of the field). See reference/HANDOFF_DIAGNOSED_CLOUD.md.
    # ADVECTION IS TRANSFORM-INVARIANT: u·∇ρ_c = f'(n) u·∇n, and the tendency being formed is
    # dn/dt = J·(dρ_c/dt), so the advective part is just −u·∇n. It therefore reads the control
    # variable's OWN spline gradients — no chain rule, no f'', no reference-gradient round
    # trip. Under `:none`, `nu_c_z` is bitwise the old `rho_c_z` and `rho_cp_x` is unchanged.
    #
    # The rest picks up the Jacobian: dn/dt = J(ρ_c)·[−ρ_c ∇·u + Qdot − AUTO_COLL]. `Jc` is
    # exactly 1.0 under `:none`, and multiplication by 1.0 is the identity in IEEE, so the
    # default path is bit-for-bit the code that had no transform.
    #
    # `Qdot` is withheld here for the same reason `Qdot_r` is on slot 8, and returns through
    # `Jc` at the step-mean rate. The `-AUTO_COLL` sink is untouched.
    mc_advect!(ADV, geom, u, w, vv, r, rho_cp_x, nu_c_z, rcv.f_l)
    @turbo FORCING .= @. Jc * ((-rho_c * div) - AUTO_COLL)
    @turbo expdot[colstart:colend,9] .= @. ADV + FORCING
    # The ICE back-reaction on the cloud: riming collection and homogeneous freezing, both
    # sinks. There is no ice-to-cloud return channel — melting ice becomes rain, not cloud.
    if ice_on
        @inbounds for i in eachindex(Fi_z)
            expdot[colstart + i - 1, 9] += Jc[i] * S.ICE_C[i]
        end
    end
    # Cloud has no sedimentation channel, and its autoconversion sink is -AUTO_COLL.
    budget_trace && water_budget_probe!(mtile, MC_BUDGET_C, 9, t, colstart, rho_c, ADV, div,
                                        Qdot, AUTO_COLL, -1.0, nothing, w, z)

    # Vapor partial density (the appended slot, unconditional). The continuity equation every
    # other density in the set takes, with the phase changes as its only sources:
    #
    #     ∂ρ_v'/∂t = −u·∇ρ_v − ρ_v ∇·u − (Q̇_c + Q̇_r) − Σ_k Q̇_{i,k} + (res_rho_t − ρ_v)/τ_rec
    #
    # THE SLOT IS AN UNTRANSFORMED PERTURBATION against the derived ρ̄_v (see `vapor_slot`), so
    # this takes slot 7's shape and not slot 9's: no control variable, no Jacobian, and the
    # advection reads the perturbation gradient horizontally (ρ̄_v has no x-dependence) but the
    # TOTAL vertical gradient `rho_v_z`, exactly as slots 1-3 and 6-7 do.
    #
    #
    # `VREC` is the reconciliation nudge (`rho_v_reconcile`), the second link of the
    # Q_ss → ρ_v → ρ_t chain described at slot 7. It is thermodynamically INERT — the retrieval
    # reads ρ_t, ρ_liq and ρ_ice and never the vapor — so it moves the partition and nothing
    # else, and it is identically zero on a resting base.
    #
    # `VAPOR_SRC` is withheld from this tendency with the condensate sources it is the exact
    # negative of — all of it, because every one of its terms is a relaxation of `Q_ss`. It
    # returns as the vapor's direct increment, still the exact negative of what slots 8, 9 and
    # the three ice masses receive, so phase change remains INTERNAL to the water and `ρ_t`
    # remains the untouched conservation anchor. The column itself is now filled by the
    # pre-compute block below, at the STEP-MEAN rate, and is what the census reads.
    VREC = S.VREC
    mc_advect!(ADV, geom, u, w, vv, r, rho_vp_x, rho_v_z, rvv.f_l)
    @. VREC = rho_v_reconcile(res_rho_t, rho_v, tau_rec)
    @turbo FORCING .= @. (-rho_v * div) + VREC
    @turbo expdot[colstart:colend, rv_i] .= @. ADV + FORCING

    # Rain NUMBER density (the appended slot, `options[:rain_moments] = 2`). The same
    # continuity form every other total takes — transform-invariant advection of the control
    # variable, everything else through the Jacobian:
    #
    #     ∂n_r/∂t = −u·∇n_r − n_r ∇·u + NPRC1 + nragg + npre − ∂F_n/∂z
    #
    # `NR_SRC` is the sum of the three number sources (autoconversion's NPRC1, Beheng
    # self-collection/Verlinde-Cotton breakup, and the evaporation loss npre); `Fnr_z` is the
    # NUMBER-weighted sedimentation flux divergence, fitted on this slot's own column. Nothing
    # else in the set receives either term: a raindrop count is not mass and not energy, so
    # rho_t and E_t keep taking only the mass flux `Fr_z` assembled for slot 8.
    #
    # Accretion contributes nothing here on purpose — it grows the drops that exist. See
    # `rain_accretion_2m`, and `rain_moments` for the process table.
    if rain_2m
        mc_advect!(ADV, geom, u, w, vv, r, nrv.f_x, nu_nr_z, nrv.f_l)
        @turbo FORCING .= @. Jnr * ((-n_r * div) + NR_SRC - Fnr_z)
        @turbo expdot[colstart:colend, nr_i] .= @. ADV + FORCING
        # The ICE back-reaction on the raindrop COUNT: drops lost to ice-rain collection,
        # Bigg and homogeneous freezing, and drops gained from melting ice. This is the
        # channel `ice_microphysics` requires `rain_moments == 2` for — with a prescribed
        # rain DSD there is no number for the ice to remove.
        if ice_on
            @inbounds for i in eachindex(Fi_z)
                expdot[colstart + i - 1, nr_i] += Jnr[i] * S.ICE_NR[i]
            end
        end
    end

    # ── The twelve ICE slots ───────────────────────────────────────────────────
    #
    # All twelve take the SAME form, which is the form every other total in the set takes
    # (Eqs. ice_prog_mass..ice_prog_c):
    #
    #     ∂X/∂t = −u·∇X − X ∇·u + Ẋ − ∂F_X/∂z
    #
    # with the advection transform-invariant (it reads the control variable's own fitted
    # gradients; `nu_*_z` is bitwise the density gradient under `:none`) and everything else
    # through the Jacobian, exactly as slots 8 and 9 do. `SRC_*` carries the ISHMAEL process
    # sources plus the anchor-reconciliation removal, and `F_*_z` the moment-weighted
    # sedimentation flux divergence — both assembled in the ice block above.
    #
    # Nothing else in the set receives anything from a NUMBER or a VOLUME slot: only the three
    # MASS fluxes reach ρ_t and E_t, and they did so above.
    if ice_on
        mc_advect!(ADV, geom, u, w, vv, r, i1qv.f_x, S.nu_i1q_z, i1qv.f_l)
        @turbo FORCING .= @. S.J_i1q * ((-i1q * div) + S.SRC_i1q - S.F_i1q_z)
        @turbo expdot[colstart:colend, IS.i1_q] .= @. ADV + FORCING

        mc_advect!(ADV, geom, u, w, vv, r, i1nv.f_x, S.nu_i1n_z, i1nv.f_l)
        @turbo FORCING .= @. S.J_i1n * ((-i1n * div) + S.SRC_i1n - S.F_i1n_z)
        @turbo expdot[colstart:colend, IS.i1_n] .= @. ADV + FORCING

        mc_advect!(ADV, geom, u, w, vv, r, i1av.f_x, S.nu_i1a_z, i1av.f_l)
        @turbo FORCING .= @. S.J_i1a * ((-i1a * div) + S.SRC_i1a - S.F_i1a_z)
        @turbo expdot[colstart:colend, IS.i1_a] .= @. ADV + FORCING

        mc_advect!(ADV, geom, u, w, vv, r, i1cv.f_x, S.nu_i1c_z, i1cv.f_l)
        @turbo FORCING .= @. S.J_i1c * ((-i1c * div) + S.SRC_i1c - S.F_i1c_z)
        @turbo expdot[colstart:colend, IS.i1_c] .= @. ADV + FORCING

        mc_advect!(ADV, geom, u, w, vv, r, i2qv.f_x, S.nu_i2q_z, i2qv.f_l)
        @turbo FORCING .= @. S.J_i2q * ((-i2q * div) + S.SRC_i2q - S.F_i2q_z)
        @turbo expdot[colstart:colend, IS.i2_q] .= @. ADV + FORCING

        mc_advect!(ADV, geom, u, w, vv, r, i2nv.f_x, S.nu_i2n_z, i2nv.f_l)
        @turbo FORCING .= @. S.J_i2n * ((-i2n * div) + S.SRC_i2n - S.F_i2n_z)
        @turbo expdot[colstart:colend, IS.i2_n] .= @. ADV + FORCING

        mc_advect!(ADV, geom, u, w, vv, r, i2av.f_x, S.nu_i2a_z, i2av.f_l)
        @turbo FORCING .= @. S.J_i2a * ((-i2a * div) + S.SRC_i2a - S.F_i2a_z)
        @turbo expdot[colstart:colend, IS.i2_a] .= @. ADV + FORCING

        mc_advect!(ADV, geom, u, w, vv, r, i2cv.f_x, S.nu_i2c_z, i2cv.f_l)
        @turbo FORCING .= @. S.J_i2c * ((-i2c * div) + S.SRC_i2c - S.F_i2c_z)
        @turbo expdot[colstart:colend, IS.i2_c] .= @. ADV + FORCING

        mc_advect!(ADV, geom, u, w, vv, r, i3qv.f_x, S.nu_i3q_z, i3qv.f_l)
        @turbo FORCING .= @. S.J_i3q * ((-i3q * div) + S.SRC_i3q - S.F_i3q_z)
        @turbo expdot[colstart:colend, IS.i3_q] .= @. ADV + FORCING

        mc_advect!(ADV, geom, u, w, vv, r, i3nv.f_x, S.nu_i3n_z, i3nv.f_l)
        @turbo FORCING .= @. S.J_i3n * ((-i3n * div) + S.SRC_i3n - S.F_i3n_z)
        @turbo expdot[colstart:colend, IS.i3_n] .= @. ADV + FORCING

        mc_advect!(ADV, geom, u, w, vv, r, i3av.f_x, S.nu_i3a_z, i3av.f_l)
        @turbo FORCING .= @. S.J_i3a * ((-i3a * div) + S.SRC_i3a - S.F_i3a_z)
        @turbo expdot[colstart:colend, IS.i3_a] .= @. ADV + FORCING

        mc_advect!(ADV, geom, u, w, vv, r, i3cv.f_x, S.nu_i3c_z, i3cv.f_l)
        @turbo FORCING .= @. S.J_i3c * ((-i3c * div) + S.SRC_i3c - S.F_i3c_z)
        @turbo expdot[colstart:colend, IS.i3_c] .= @. ADV + FORCING
    end

    # ── Horizontal water-species mixing (Khdiff_water; 0.0 = OFF, the default) ──
    #
    # *** DIAGNOSTIC / TEMPORARY -- NOT ENERGY CONSISTENT. READ BEFORE USING. ***
    #
    # Why it exists: an axisymmetric domain has no asymmetries to provide radial
    # mixing, so the sharp moisture gradients a TC develops are damped by nothing at
    # all -- until now the water species had NO horizontal diffusion whatsoever
    # (slot 8's "no diffusion yet" note), while momentum had Smagorinsky. That
    # leaves grid-scale structure in the water field free to grow.
    #
    # What is wrong with it: this is a bare K*grad^2 on the water species. Diffusing
    # water MASS without transporting the internal energy and latent heat that mass
    # carries is exactly the moist coupling deferred in
    # reference/moist_compressible_diffusion_handoff.md. The proper form is a product
    # rule through the retrieval's T sensitivities, sourcing E_t and p alongside the
    # mass (the `M = p + E_t - rho_t(ke+gz)` enthalpy invariant: any transport must
    # source BOTH). Because this does not, it WILL drift energy -- watch
    # conservation_drift. It is a noise control to obtain a stable run, not physics,
    # and it must be replaced by the full moist form before any production science.
    #
    # Khdiff_water < 0 selects the Smagorinsky K_smag/Sc_t (the Khdiff_heat sentinel
    # convention); > 0 is a constant diffusivity. Same K*grad^2 (not div(K grad))
    # approximation as the momentum and heat terms.
    #
    # WHAT CHANGED WITH THE PROGNOSTIC VAPOR, and what deliberately did not. This block still
    # mixes total water, cloud, rain and Q_ss and gives the VAPOR SLOT no Laplacian of its own.
    # While the vapor was a residual that was automatic -- mixing rho_t and the condensates WAS
    # mixing the vapor. It is no longer: the vapor's countermove now arrives through the
    # reconciliation nudge, on tau_rec rather than instantaneously, so a diffused column's
    # partition closes on the nudge timescale instead of pointwise. That is a deliberate
    # deferral and not an oversight -- adding K*grad^2(rho_v') here is one more Laplacian, but
    # a water mixing that is honest about energy has to source E_t and p alongside the mass
    # (see above), and that rewrite is where the vapor leg belongs.
    if Khdiff_water != 0.0
        # UNDER A TRANSFORM THIS MIXES THE CONTROL VARIABLE, AND THAT IS THE INTENDED FORM.
        # The Laplacian is applied to the slot, so with `condensate_transform`/`rain_transform`
        # on it smooths ν rather than ρ. That is not an approximation to `K∇²ρ` — above the
        # knee it IS `K∇²ρ`, because `bhyp` is exactly affine there:
        #
        #     bhyp(ρ) = (ρ + μ)/2 − μ²/(2(ρ + μ))        (algebra, not an expansion)
        #
        # so `∇²ν = ∇²ρ/2 + O(μ²)` and `f'(ν) = 2 + O(μ²/ρ²)`, and the implied density rate
        # `f'(ν)·K∇²ν` agrees with `K∇²ρ` to relative O(μ²/ρ²). Measured on a Gaussian cloud at
        # μ = 1e-7: 3.7e-9 at ρ_c = 2.3e-3, 2.6e-7 at 3.2e-4, 6.2e-4 at 5.8e-6, and only
        # reaching 7.2e-2 at 3.7e-7 kg/m³ — a thousandth of the smallest meaningful cloud.
        #
        # WHY NOT THE EXACT CHAIN RULE `J·K·(f''(ν)|∇ν|² + f'(ν)∇²ν)`. Because `f''(0) = 1/μ`:
        #
        #     f'(ν)  = 2(ρ+μ)² / ((ρ+μ)² + μ²)          ∈ [1, 2]
        #     f''(ν) = 8μ²(ρ+μ)³ / ((ρ+μ)² + μ²)³       → 1/μ = 1e7 as ρ → 0
        #
        # and the region where that term matters is `μ/|∇ρ| ≈ 1.7 cm` wide against a 500 m
        # cell. Its magnitude there is `K·f''·|∇ν|²` ≈ 0.036 kg/m³/s at ρ = μ and 0.148 at
        # ρ = 1e-8 — twelve to fifty times the entire cloud amplitude, per second. Analytically
        # it is cancelled by `f'∇²ν`; discretely both come from a filtered, ringing spline fit
        # at the one place the fit is worst, and nothing makes that cancellation survive. It is
        # not a smoothing operator, it is a pointwise sample of an unresolvable knee, so it is
        # deliberately NOT offered as an option — a branch that is unrunnable at the cloud edge
        # is dead code that will be believed. (This also retires the stated reason for keeping
        # `:bhyp_smooth` in reference/FINDINGS_CONDENSATE_STAGE1.md — "for any future consumer
        # that needs f'' at the cloud edge, the water Laplacians". They do not need it.)
        #
        # Two consequences worth stating rather than leaving to be rediscovered:
        #   * slot 3 (total water) is untransformed and keeps mixing in DENSITY space while
        #     slots 8/9 mix in ν space, so the difference shows up as a reconciliation gap
        #     between slot 3 and the vapor slot. That is O(μ²/ρ²) above the knee — not a new
        #     inconsistency class.
        #   * positivity survives whatever the mixing does, because it is a property of the
        #     recovery, not of the operator: ν monotone in ρ and `ahyp` floored at 0 (`:bhyp`)
        #     or −μ (`:bhyp_smooth`). The coefficient limiter this replaced could not say that
        #     across a nest interface at all.
        #
        # Precedent inside the model: the Galerkin low-pass `l_q` (2.0 by default) and the
        # spline filter are ALREADY applied to the slot, i.e. already smooth ν directly, and
        # the whole transform validation ladder was run with them on.
        WLAP = S.KDIFF                    # both free again after slot 9
        WLAP2 = S.FORCING
        # ACCUMULATED THROUGH AN EXPLICIT LOOP, not `expdot[colstart:colend,s] .+= …`.
        # `.+=` on an indexed expression expands to `A[r] = A[r] .+ B`, and the READ
        # materializes the slice: this block was allocating four times per column (measured;
        # it is the only mc term that did, and it was never in the allocation gate because
        # Khdiff_water is 0.0 in every shipped configuration). Same convention as the Louis
        # BL's apply loop. It is also the only additive mc block, which is why `.=`
        # everywhere else was fine.
        # The two branches stay SEPARATE and each carries its own loops: hoisting the
        # diffusivity into one variable would give it type `Union{Vector{Float64},Float64}`
        # and box every use — the same trap the state-dependent SI hit.
        if Khdiff_water < 0.0
            use_smag || error("physical_params[:Khdiff_water] < 0 selects the " *
                              "Smagorinsky water diffusivity, which requires Ls > 0")
            # Total water rho_w' = rho_t' - rho_d' (dry air is NOT mixed: only the
            # water rides on rho_t here, so rho_d's own tendency is untouched).
            mc_w_kdiff!(WLAP,  geom, K_smag, rtv, r)
            mc_w_kdiff!(WLAP2, geom, K_smag, rdv, r)
            @inbounds for i in eachindex(WLAP)
                expdot[colstart + i - 1, 3] += (WLAP[i] - WLAP2[i]) / Sc_t
            end
            # Slots 8 and 9 are the transformed ones: under a transform `rrv`/`rcv` are ν,
            # and `K∇²ν` is the intended operator (see the block comment above). No
            # Jacobian — the tendency is already in the slot's own units.
            mc_w_kdiff!(WLAP, geom, K_smag, rrv, r)
            @inbounds for i in eachindex(WLAP)
                expdot[colstart + i - 1, 8] += WLAP[i] / Sc_t
            end
            mc_w_kdiff!(WLAP, geom, K_smag, rcv, r)
            @inbounds for i in eachindex(WLAP)
                expdot[colstart + i - 1, 9] += WLAP[i] / Sc_t
            end
            mc_w_kdiff!(WLAP, geom, K_smag, qsv, r)
            @inbounds for i in eachindex(WLAP)
                expdot[colstart + i - 1, 7] += WLAP[i] / Sc_t
            end
            # The rain NUMBER mixes with the same coefficient as the rain MASS, and must:
            # mixing one without the other rescales the mean drop size of every column the
            # operator touches, and the fall speeds, evaporation timescale and
            # self-collection all read that size. (This is the horizontal counterpart of the
            # refusal the VERTICAL water diffusion raises for the same reason — there the
            # solve is implicit and adding n_r to it is real work; here it is one more
            # Laplacian.)
            if rain_2m
                mc_w_kdiff!(WLAP, geom, K_smag, nrv, r)
                @inbounds for i in eachindex(WLAP)
                    expdot[colstart + i - 1, nr_i] += WLAP[i] / Sc_t
                end
            end
        else
            mc_w_kdiff!(WLAP,  geom, Khdiff_water, rtv, r)
            mc_w_kdiff!(WLAP2, geom, Khdiff_water, rdv, r)
            @inbounds for i in eachindex(WLAP)
                expdot[colstart + i - 1, 3] += WLAP[i] - WLAP2[i]
            end
            mc_w_kdiff!(WLAP, geom, Khdiff_water, rrv, r)
            @inbounds for i in eachindex(WLAP)
                expdot[colstart + i - 1, 8] += WLAP[i]
            end
            mc_w_kdiff!(WLAP, geom, Khdiff_water, rcv, r)
            @inbounds for i in eachindex(WLAP)
                expdot[colstart + i - 1, 9] += WLAP[i]
            end
            mc_w_kdiff!(WLAP, geom, Khdiff_water, qsv, r)
            @inbounds for i in eachindex(WLAP)
                expdot[colstart + i - 1, 7] += WLAP[i]
            end
            # Same coefficient as the rain mass — see the note in the Smagorinsky branch.
            if rain_2m
                mc_w_kdiff!(WLAP, geom, Khdiff_water, nrv, r)
                @inbounds for i in eachindex(WLAP)
                    expdot[colstart + i - 1, nr_i] += WLAP[i]
                end
            end
        end
    end

    # Tangential momentum (slot 9, cylindrical geometries only — no method body
    # executes on the Cartesian slice): advection, azimuthal PGF (3D), Coriolis +
    # curvature -u(f + v/r), and the λ-component of the cylindrical vector Laplacian.
    if use_smag
        mc_v_tendency!(expdot, geom, colstart, colend, S, u, w, uv, vv, pv, rho_t, r,
                       fcor, K_smag)
    else
        mc_v_tendency!(expdot, geom, colstart, colend, S, u, w, uv, vv, pv, rho_t, r,
                       fcor, Khdiff)
    end
    # Under the exact unsplit acoustic SI, cancel v's reference-linear azimuthal
    # PGF from the AB3 remainder (RLR only). The v-leg AI2* history is the STORED
    # applied increment (loaded in phase A, exact_si_load_history!) — the weak
    # operator the unsplit solve actually applied, self-consistent per §8; a fresh
    # pointwise −(1/ρ̄_t r)∂λp′ history would mismatch the weak solve (ε-chain).
    if hsi_like
        mc_stage_v_acoustic!(expdot, geom, colstart, colend, pv, rho_tbar, r)
    end

    # ── Rayleigh sponge (momentum-only) ──
    # Klemp-Durran absorbing layer against gravity-wave reflection off the rigid lid:
    # u and w (and v on the cylinders) relax toward the resting base state with the
    # (negative) coefficient ray(z), and E_t follows the resolved KE down exactly as
    # for friction (the FRIC_KE invariant: dE = rho_t*(u*ray*u + w*ray*w) =
    # 2*rho_t*ray*ke, with ke already carrying v on the cylinders). T, p and
    # Q_ss are held — a KE sink to the sponge, not diabatic heating. Explicit loop,
    # not `.+=` view broadcasts: the SubArray stops eliding at this function size
    # (the dE_w lesson). Behind `alpha > 0` so disabled configs are bit-identical
    # (an unconditional `+ 0.0` term could flip -0.0 tendencies).
    mc_sponge!(expdot, geom, colstart, alpha, z_damp, z, u, w, vv, rho_t, ke)

    # ── Louis boundary layer (explicit vertical mixing + surface drag) ──
    # Added AFTER every slot is written: all contributions are additive (the Q_ss
    # saturation chain rule is linear), so the lines above stay frozen and
    # louis_bl = false is bit-identical. See mc_boundary_layer.jl.
    if louis_bl
        if ctrans_on
            # The BL's cloud eddy flux is a MASS flux, so it has to be built from the
            # perturbation DENSITY gradient, not from the slot's. `rho_c_z` (line ~2703) is the
            # total ∂z ρ_c = ∂z ν / J, and this is its first consumer — it was written for one.
            #
            # Staged HERE and not inside `mc_louis_bl!` because `rho_cbar_z` is a SubArray, and
            # handing it to a @noinline callee is an escape that boxes once per column; the
            # zero-allocation gate in test/test_allocations.jl exists for exactly that.
            #
            # NOT used on the `:none` path: there `rho_c_z` is `rho_cp_z + rho_cbar_z` and
            # subtracting `rho_cbar_z` back off is not bitwise `rho_cp_z`. The callee keeps
            # reading `rcv.f_z` directly when the transform is off, which is.
            @. S.bl_rho_cp_z = rho_c_z - rho_cbar_z
        end
        mc_louis_bl!(mtile, S, geom, colstart, colend, z, uv, wv, vv, rtv, rdv, rcv,
                     expdot, l_inf, Cd_param, sfc_wind_factor,
                     surface_fluxes, Ck, SST, U_min, ctrans_on)
    end

    # ── Implicit vertical diffusion tendencies (AI2* history in the diffdot channel) ──
    # Vertical diffusion must be implicit on a Chebyshev column, where the spectral
    # second-derivative eigenvalues scale as N^4. The acoustic solver owns impdot[w],
    # impdot[p] and impdot[E_t], so these live in diffdot instead. Slot 6 (E_t) carries the
    # HEAT tendency in entropy space (not an energy tendency): diffusion_timestep_mc solves
    # for s_t' and maps the increment onto (p, E_t, Q_ss). Slots 3/8/9 carry the water
    # tendencies (total water rho_w' = rho_t' - rho_d', rain rho_r, cloud rho_c').
    #
    # The heat variable is the full moist entropy s_t (vapor + liquid contribution), a
    # retrieval-dependent diagnostic; the resting base is still bit-preserved because
    # s_tbar comes from mc_reference_diagnostics — the SAME retrieval pipeline — so at
    # rest s_t' == 0 exactly. (The horizontal heat diffusion keeps the dry-exact s_d
    # chain rule: transforming a diagnosed field horizontally needs halo machinery the
    # column decomposition doesn't have; see the handoff doc.)
    if Kvdiff > 0.0 || Kvdiff_heat > 0.0 || Kvdiff_water > 0.0
        diffdot = mtile.diffdot_n
        if Kvdiff > 0.0
            @turbo diffdot[colstart:colend,4] .= @. Kvdiff * u_zz
            @turbo diffdot[colstart:colend,5] .= @. Kvdiff * w_zz
            mc_v_diffdot!(diffdot, geom, colstart, colend, Kvdiff, vv)
        end
        if Kvdiff_heat > 0.0
            s_t = S.s_t
            # `q_i C_i ln(T/T_0)` joins the liquid term (TeX Eq. entropy_ice); `q_i` is an
            # exact 0.0 column with ice off, so this is bitwise the pre-ice `s_t`.
            @. s_t = moist_entropy_total(Tk, rho_d, q_v, q_l, q_i)
            # ∂zz(s_t') from the column basis, so the explicit AI2* tendency and the implicit
            # Helmholtz operator use the same discrete ∂zz.
            s_col = scratch_column(mtile, 6)
            s_col.uMish .= s_t .- mtile.mc_ref_diag.s_tbar
            Btransform!(s_col)
            Atransform!(s_col)
            stage_zz = S.stage_zz
            Ixxtransform(s_col, stage_zz)
            @turbo diffdot[colstart:colend,6] .= Kvdiff_heat .* stage_zz
        end
        if Kvdiff_water > 0.0
            # EVERY diffused water species is a prognostic slot, so every ∂zz comes straight
            # from the grid's derivative slots: total water rho_w' = rho_t' - rho_d', vapor,
            # cloud, rain. The vapor's ∂zz is now the SLOT's, not an assembled remainder —
            # which retires the column transform of the DIAGNOSED rho_v this block once
            # needed (a fit of a diagnosed field, with the reference-profile subtraction
            # required to keep the resting base quiet).
            @turbo diffdot[colstart:colend,3] .= @. Kvdiff_water * (rho_tp_zz - rho_dp_zz)
            @turbo diffdot[colstart:colend,8] .= @. Kvdiff_water * rho_rp_zz
            @turbo diffdot[colstart:colend,9] .= @. Kvdiff_water * rho_cp_zz
            @turbo diffdot[colstart:colend, rv_i] .= @. Kvdiff_water * rho_vp_zz
        end
    end

    # ── ETD-AB3 pre-compute: the stiff relaxation pair, and the step-mean it closes on ──────
    #
    # HERE for the same reason the depletion census is here, and one more. `expdot[.,7]` is
    # final for the step — every contribution to `N` has been added, including the horizontal
    # water mixing and the boundary layer — and `expdot_nm1`/`expdot_nm2` are still `N^{n−1}`
    # and `N^{n−2}`, which `explicit_timestep` is about to rotate away. `Q_ss` is still the
    # `n`-state. All four are needed together and this is the only window in which they exist
    # together, so the whole computation is done now and the APPLY (`relaxation_adjustment_qss!`,
    # after the explicit advance) is reduced to writes.
    #
    # Per gridpoint:
    #
    #   λ  = τ_c^{-1} + τ_r^{-1} + Σ_k τ_{i,k}^{-1}, counting only the channels whose drive is
    #        UNCLIPPED at state n — a clipped channel's flux is constant and lives in N;
    #        the rain and ice conductances entering here are the REALIZED ones (see below);
    #   x  = λΔt, and (e^{−x}, b, J0, g) = `etd_step_weights(x, t)`;
    #   Q_ss^{n+1} = e^{−x} Q_ss^n + Δt(b1 N^n + b2 N^{n−1} + b3 N^{n−2})     (Eq. etd_ab3)
    #   Q̄_ss      = J0 Q_ss^n + Δt(g1 N^n + g2 N^{n−1} + g3 N^{n−2})         (Eq. qss_stepmean)
    #
    # Then EVERY phase-change rate is evaluated ONCE at `Q̄_ss` — liquid drive and ice drive
    # recomputed at the step-mean with the SAME frozen clip classification — and handed to
    # every consumer as the same number over the step. That is what makes the vapor removed,
    # the mass condensed or deposited and the heat released identical by construction; the
    # supersaturation's own budget is the same statement rearranged through the psychrometric
    # cancellation.
    #
    # For a CLIPPED channel the step-mean drive IS the frozen drive, so its step-mean rate is
    # bitwise its `n`-rate: the scheme only ever changes the channels it actually integrates.
    etd_dep_a = S.etd_dep_a
    etd_dep_b = S.etd_dep_b
    etd_qbar = S.etd_qbar
    etd_qnp1 = S.etd_qnp1
    etd_d1 = S.etd_d1; etd_d8 = S.etd_d8; etd_d9 = S.etd_d9; etd_dv = S.etd_dv
    etd_di1 = S.etd_di1; etd_di2 = S.etd_di2; etd_di3 = S.etd_di3
    Qdot_bar = S.Qdot_bar; Qdot_r_bar = S.Qdot_r_bar; Qdep_bar = S.Qdep_bar
    VAPOR_SRC = S.VAPOR_SRC
    expdot_nm1 = mtile.expdot_nm1
    expdot_nm2 = mtile.expdot_nm2
    ts = model.ts
    # Census handles for the sublimation leg below (see `mc_donor_census!`); zero columns when
    # the water trace is off, in which case the census is skipped entirely.
    stx = mtile.mc_water_stats
    tid_x = Threads.threadid()
    census_on = size(stx, 2) > 0
    # Block B of the ATTRIBUTION census (`MC_ATTR_QR_EVAP`/`MC_ATTR_QR_COMB`), opt-in with
    # the other three blocks.
    attr_on = census_on && ice_attr
    @inbounds for i in eachindex(etd_lam)
        g = colstart + i - 1
        invtau_l = invtau_c[i] + invtau_r[i]
        # The liquid clip flag, re-derived from the two columns written beside the closure:
        # `etd_lam` holds the conductance only where the drive is unclipped, so a channel with
        # conductance and no λ is a clipped one. (An inactive channel has neither, and its
        # rates are exact zeros whichever branch it takes.)
        clipped_l = etd_lam[i] == 0.0 && invtau_l != 0.0
        sum_it = ice_on ?
            ((S.invtau_i1[i] + S.invtau_i2[i]) + S.invtau_i3[i]) : 0.0
        lam = etd_lam[i] + (ice_on ? etd_dep_a[i] * sum_it : 0.0)
        etd_lam[i] = lam

        (emx, b1, b2, b3, j0, g1, g2, g3) = etd_step_weights(lam * ts, t)
        n0 = expdot[g, 7]
        n1 = expdot_nm1[g, 7]
        n2 = expdot_nm2[g, 7]
        qn = Q_ss[i]
        qbar = (j0 * qn) + (ts * (((g1 * n0) + (g2 * n1)) + (g3 * n2)))
        etd_qbar[i] = qbar
        # The propagated total, carried back to the SLOT (a perturbation against `Q̄_ssbar`,
        # which is constant in time, so the propagator may be taken on the total and shifted).
        etd_qnp1[i] = ((emx * qn) + (ts * (((b1 * n0) + (b2 * n1)) + (b3 * n2)))) - Q_ssbar[i]

        # ── The step-mean LIQUID rates (TeX Eq. simple_cond at Q̄_ss) ──
        # The channel split is `qss_condensation_rates`' own, expression for expression, so a
        # clipped or inactive channel returns the same exact zeros it does.
        #
        # ── THE RAIN CONDUCTANCE λ SEES IS THE REALIZED ONE ──────────────────────────────
        # Nothing is scaled HERE, exactly as nothing is scaled on the sublimation leg below.
        # Rain evaporation is a sink of the RAIN, and the propagator above bounds `Q_ss` and
        # not `ρ_r`: the vapor deficit sets the step-mean, and before Stage 2b nothing set it
        # by the rain that is there — while the ice legs were realized on a conductance that
        # did not contain the evaporation, so the two draws on one reservoir summed without
        # either knowing of the other (1.0012 reservoirs per step, measured). The rain donor's
        # factor `f_rain` now covers every sink on the reservoir and is folded into
        # `invtau_r`/`Qdot_r` themselves, below the ice block, so the realized conductance
        # `f_r τ_r^{-1}` is ALREADY in the `λ` this step propagated with, in the `N` it
        # carried, and in the `invtau_r` the split below shares out — one number, three
        # consumers, and the psychrometric cancellation goes through unchanged because the
        # factor multiplies the already-cancelled conductance. The clipped/unclipped
        # classification is untouched: it is a statement about the drive, not about the rate,
        # and the fold applies only where the channel is a SINK of the rain at state `n` —
        # the `evap_now` mirror of `sub_now`, since a donor factor bounds a draw and rain
        # condensation is a source. (The CLOUD channel is not realized: `q_c` carries its own
        # donor factor for the ice legs and its evaporation is not part of this construction.)
        qc_bar = 0.0
        qr_bar = 0.0
        if clipped_l
            qc_bar = Qdot[i]
            qr_bar = Qdot_r[i]
        else
            qb = qbar * invtau_l / (1.0 + Q_s[i])
            qc_bar = invtau_c[i] == 0.0 ? 0.0 : qb * (invtau_c[i] / invtau_l)
            qr_bar = invtau_r[i] == 0.0 ? 0.0 : qb * (invtau_r[i] / invtau_l)
        end
        # ── The step-mean ICE rates (Eq. dep_rate at Q̄_ss + 𝒟) ──
        # `a·Q̄_ss + b` is the same affine drive `mc_ice_sources!` classified: `Q̄_ss + 𝒟` where
        # the drive is unclipped, the frozen constant where it is clipped, and an exact zero
        # above `T_0` where the channel is shut.
        qdep_bar = 0.0
        if ice_on
            drive_bar = (etd_dep_a[i] * qbar) + etd_dep_b[i]
            iq = drive_bar / (1.0 + S.Q_s_i[i])
            q1 = iq * S.invtau_i1[i]
            q2 = iq * S.invtau_i2[i]
            q3 = iq * S.invtau_i3[i]
            # ── SUBLIMATION is already realized, because `invtau_i` is ────────────────────
            # `drive_bar < 0` makes these ice SINKS, and the propagator above bounds `Q_ss`,
            # not `ρ_i`: nothing bounds the draw by the ice that is present, and an over-drawn
            # `:bhyp` ice slot recovers at −μ rather than at the debit it was given — the
            # created-mass mechanism of the earlier walls, running in reverse and making vapor
            # out of ice that was not there. Measured before the cure: 2.63 reservoirs of
            # `ρ_i1` per step.
            #
            # Nothing is scaled HERE. `mc_ice_sources!` folds the ice donor factor into
            # `invtau_i<k>` itself (see the comment beside `ITi[k][i]`), so the same realized
            # conductance is already in the `λ` this step propagated with, in the `N` it
            # carried, and in the `q_k` just formed — one number, three consumers, no
            # opportunity for the pair and the transfer to disagree. Everything downstream (the
            # mass increments, the vapor source, the latent heating, the habit partition) is
            # therefore consistent with it by construction.
            qdep_bar = (q1 + q2) + q3
            etd_di1[i] = ts * S.J_i1q[i] * q1
            etd_di2[i] = ts * S.J_i2q[i] * q2
            etd_di3[i] = ts * S.J_i3q[i] * q3
            # ── The SUBLIMATION side of the deposition channel, censused as an ICE SINK ─────
            # `Q̇_{i,k} < 0` removes ice mass, and it is the one ice sink that lives OUTSIDE
            # every conductance in this file: the exponential propagator bounds `Q_ss`, not
            # `ρ_i`, so the step-mean drive is limited by the vapor DEFICIT and nothing limits
            # it by the ice that is there. Under `:bhyp` an over-drawn ice slot recovers at −μ
            # rather than at the debit it was given, which is the created-mass mechanism of the
            # earlier walls running in reverse — vapor made from ice that was not there.
            # Censused here, where the step-mean increment exists; reported, never limited.
            if census_on
                mc_donor_census!(stx, tid_x, MC_DONOR_S1,
                                 max(-q1, 0.0) / rho_d[i], S.i1q[i] / rho_d[i],
                                 MC_DONOR_QFLOOR, ts)
                mc_donor_census!(stx, tid_x, MC_DONOR_S2,
                                 max(-q2, 0.0) / rho_d[i], S.i2q[i] / rho_d[i],
                                 MC_DONOR_QFLOOR, ts)
                mc_donor_census!(stx, tid_x, MC_DONOR_S3,
                                 max(-q3, 0.0) / rho_d[i], S.i3q[i] / rho_d[i],
                                 MC_DONOR_QFLOOR, ts)
                # ── BLOCK B of the attribution census: RAIN EVAPORATION on the same
                # reservoir the ice legs draw on. Its DEFINITION is unchanged by Stage 2b
                # and what it is expected to read is not: `_ice_donor_factors` now carries
                # the evaporation conductance, so this draw is inside `MC_DONOR_QR` and the
                # combined number is expected `≤ 1` — one exponential depletion shared out,
                # not two independent ones. It was the block that convicted the omission
                # (1.0012 reservoirs over 1296 gridpoint-steps on the Stage 0a fixture) and
                # it stays as the INDEPENDENT reading of the bound: taken at the APPLIED
                # step-mean `qr_bar` (the number the slot actually receives, already carrying
                # `f_rain`) rather than at the frozen `Qdot_r` the factor was formed on, and
                # in DENSITY units, which is what `ρ_r` and `ICE_R` are already in. The
                # combined channel is the total fraction of the rain the step removes.
                if attr_on
                    rr_i = rho_r[i]
                    if rr_i > MC_DONOR_QFLOOR * rho_d[i]
                        ev = max(-qr_bar, 0.0)
                        xe = ev * ts / rr_i
                        xa = (-S.ICE_R[i] + ev) * ts / rr_i
                        mc_attr_census!(stx, tid_x, MC_ATTR_QR_EVAP, xe, xe > 1.0)
                        mc_attr_census!(stx, tid_x, MC_ATTR_QR_COMB, xa, xa > 1.0)
                    end
                end
            end
            # ── The habit partition of the SAME realized increment (TeX §Departures (d)) ──
            # `ishmael_deposition_partition` distributes a mass increment over the two axes;
            # here it is handed the STEP-MEAN rate — the identical `q_k` the mass slot above
            # receives — through the same `afn` inversion the instantaneous path used, so the
            # axes and the number receive exactly the increment the mass does and the
            # glaciation burst's one-step spike never reaches an AB3 history. The `Δt` inside
            # the partition's sqrt-growth integration is the REALIZATION of the increment
            # (integrator-side, like the `ts` in every direct increment here), not a rate law:
            # the rate the physics supplied is `q_k`, and the partition is how one step's
            # worth of it is laid onto the spheroid. Gated on `capgam > 0` — an exact 0.0 for
            # a gated species — so the zero-ice path is untouched, bitwise.
            subl = drive_bar < 0.0
            # `f_isn<k>` is the ice NUMBER donor's residual factor (Stage 1b): the sublimation
            # number sink is `q̄_k·(n_k/q_k)`, so it arrives here already carrying the MASS
            # factor that `invtau_i<k>` folded in, and this is what takes it the rest of the
            # way to its own reservoir's `J₀(κ_{n,k}Δt)`. An exact `1.0` unless
            # `options[:ice_number_realization]` is on, so the three lines below are bitwise
            # what they were. See the number-donor block in `mc_ice_sources!`.
            cg1 = S.hab1_cg[i]
            if cg1 > 0.0 && q1 != 0.0
                afn1 = q1 / (4.0 * pi * S.hab1_nim3[i] * cg1)
                dp1 = ishmael_deposition_partition(ts, S.hab1_ani[i], S.hab1_cni[i],
                        S.hab1_rni[i], S.hab1_ds[i], S.hab1_rb[i], S.hab1_nim3[i],
                        S.hab_igr[i], afn1, S.hab_maxsui[i], S.hab1_vt[i], subl, cg1,
                        S.hab_dv[i], Tk[i], ISHMAEL_AO, ISHMAEL_NU, ISHMAEL_GAMMNU,
                        ISHMAEL_I_GAMMNU, ISHMAEL_FOURTHIRDSPI)
                S.etd_da1[i] = ts * S.J_i1a[i] * dp1.ard
                S.etd_dc1[i] = ts * S.J_i1c[i] * dp1.crd
                S.etd_dn1[i] = q1 < 0.0 ?
                    ts * S.J_i1n[i] * (q1 * S.hab1_niq[i] * S.f_isn1[i]) : 0.0
            else
                S.etd_da1[i] = 0.0; S.etd_dc1[i] = 0.0; S.etd_dn1[i] = 0.0
            end
            cg2 = S.hab2_cg[i]
            if cg2 > 0.0 && q2 != 0.0
                afn2 = q2 / (4.0 * pi * S.hab2_nim3[i] * cg2)
                dp2 = ishmael_deposition_partition(ts, S.hab2_ani[i], S.hab2_cni[i],
                        S.hab2_rni[i], S.hab2_ds[i], S.hab2_rb[i], S.hab2_nim3[i],
                        S.hab_igr[i], afn2, S.hab_maxsui[i], S.hab2_vt[i], subl, cg2,
                        S.hab_dv[i], Tk[i], ISHMAEL_AO, ISHMAEL_NU, ISHMAEL_GAMMNU,
                        ISHMAEL_I_GAMMNU, ISHMAEL_FOURTHIRDSPI)
                S.etd_da2[i] = ts * S.J_i2a[i] * dp2.ard
                S.etd_dc2[i] = ts * S.J_i2c[i] * dp2.crd
                S.etd_dn2[i] = q2 < 0.0 ?
                    ts * S.J_i2n[i] * (q2 * S.hab2_niq[i] * S.f_isn2[i]) : 0.0
            else
                S.etd_da2[i] = 0.0; S.etd_dc2[i] = 0.0; S.etd_dn2[i] = 0.0
            end
            cg3 = S.hab3_cg[i]
            if cg3 > 0.0 && q3 != 0.0
                afn3 = q3 / (4.0 * pi * S.hab3_nim3[i] * cg3)
                dp3 = ishmael_deposition_partition(ts, S.hab3_ani[i], S.hab3_cni[i],
                        S.hab3_rni[i], S.hab3_ds[i], S.hab3_rb[i], S.hab3_nim3[i],
                        S.hab_igr[i], afn3, S.hab_maxsui[i], S.hab3_vt[i], subl, cg3,
                        S.hab_dv[i], Tk[i], ISHMAEL_AO, ISHMAEL_NU, ISHMAEL_GAMMNU,
                        ISHMAEL_I_GAMMNU, ISHMAEL_FOURTHIRDSPI)
                S.etd_da3[i] = ts * S.J_i3a[i] * dp3.ard
                S.etd_dc3[i] = ts * S.J_i3c[i] * dp3.crd
                S.etd_dn3[i] = q3 < 0.0 ?
                    ts * S.J_i3n[i] * (q3 * S.hab3_niq[i] * S.f_isn3[i]) : 0.0
            else
                S.etd_da3[i] = 0.0; S.etd_dc3[i] = 0.0; S.etd_dn3[i] = 0.0
            end
        end

        Qdot_bar[i] = qc_bar
        Qdot_r_bar[i] = qr_bar
        Qdep_bar[i] = qdep_bar
        # The vapor source, at the step-mean and grouped so that it is the EXACT negative of
        # the sum the five condensate slots receive.
        vsrc = -((qc_bar + qr_bar) + qdep_bar)
        VAPOR_SRC[i] = vsrc
        etd_dv[i] = ts * vsrc
        etd_d8[i] = ts * Jr[i] * qr_bar
        etd_d9[i] = ts * Jc[i] * qc_bar
        # The latent heating of the pressure equation, in the bracket slot 1 carried it in
        # (TeX Eqs. thermo_p, thermo_p_ice): deposition's coefficient is condensation's with
        # `L_v → L_s` and nothing else, and freezing — which has no `(1+𝒬)` partner and stayed
        # on the multistep — is not here.
        heat = (Lv[i] - (Rv * C_pt[i] * Tk[i] / R_m[i])) * (qc_bar + qr_bar)
        if ice_on
            heat += (L_s(Tk[i]) - (Rv * C_pt[i] * Tk[i] / R_m[i])) * qdep_bar
        end
        etd_d1[i] = ts * (R_m[i] / C_vt[i]) * heat

    end

    # Depletion census of the water species. HERE, not next to the tendency assembly: this is
    # the last point at which `expdot` is still the tendency of the step about to be taken and
    # `expdot_nm1`/`expdot_nm2` are still the previous two levels (`explicit_timestep` rotates
    # them), so the increment measured is exactly the one applied — including the horizontal
    # water mixing and the boundary-layer contributions added after slot 8/9 were assembled.
    # The phase change no longer travels through `expdot` at all, so the DIRECT increments the
    # relaxation adjustment is about to apply are handed over beside it (`etd_d*`) together
    # with the step-mean rates themselves; see `water_depletion_probe!`.
    budget_trace && water_depletion_probe!(mtile, colstart, colend, t, precipitation,
                                           rho_c, rho_r, rho_v, res_rho_t, Q_ss, rho_vs,
                                           Qdot_bar, Qdot_r_bar, AUTO_COLL,
                                           cap_c, cap_r, cap_v, rv_i,
                                           etd_d9, etd_d8, etd_dv, VAPOR_SRC)

    # Advance the explicit terms
    explicit_timestep(mtile, colstart, colend, t)
    # ...and rotate the microphysics sink history with it, so the next step's depletion
    # budgets see the same two levels the integrator will weight. Placed HERE, not further
    # down, because the exact_si branch returns from this function below: this is the last
    # point common to both paths, and it is reached exactly once per column per step.
    _rotate_micro_history!(mtile, colstart, colend)

    # The stiff relaxation pair's own propagator, and the step-mean phase-change increments
    # that close the exchange against it. AFTER the explicit advance (whose slot-7 predictor
    # carried `N` alone and is discarded here) and BEFORE the acoustic solve, so the corrected
    # latent heating enters that solve exactly where the rate-level heating entered it before.
    # Placed beside `_rotate_micro_history!` for the same reason: it is the last point common
    # to the vertical-only and `exact_si` paths, reached once per column per step.
    relaxation_adjustment_qss!(mtile, colstart, colend, t)

    # Explicit AI2* history levels of the HORIZONTAL acoustic legs: both
    # dimensions' history levels belong to the star state before either implicit
    # solve (the ADI factorization applies the vertical solve below first, then
    # the horizontal patch-level sweep after the spectral merge; the exact_si
    # unsplit solve wants ALL explicit levels inside X* before it runs).
    if hsi_like
        horizontal_si_history!(mtile, colstart, colend, t)
    end

    # Exact (unsplit) 2-D semi-implicit: phase A ends here. The vertical AI2*
    # explicit history levels are applied now (the patch-level solve must see
    # the complete X*); the implicit solve and everything after it happen in
    # exact_si_apply_column! (phase B) once the patch solve has run.
    if xsi
        apply_acoustic_histories!(mtile, colstart, colend, t)
        return
    end

    # Semi-implicit (p', ρ̄_t w) acoustic solve — unconditional: the explicit acoustic
    # mode was removed (expdot carries only the remainder; the linear vertical acoustic
    # terms are integrated here and nowhere else).
    semiimplicit_adjustment_p(mtile, colstart, colend, t)

    # Implicit vertical diffusion of u, w (friction sink), the moist entropy s_t' (heat,
    # slaved onto p, E_t, Q_ss) and the water species (rho_w', rho_c', rho_r). Skipped
    # when every coefficient is zero: a vertical solve is not the identity there — it
    # refits the column and reapplies the spectral filter.
    if Kvdiff > 0.0 || Kvdiff_heat > 0.0 || Kvdiff_water > 0.0
        diffusion_timestep_mc(mtile, colstart, colend, t, geom)
    end

    # Negative water is not a representable state (see clamp_water!). LAST, so that
    # nothing downstream of it can reintroduce one: the acoustic solve and the vertical
    # diffusion have both had their say by here.
    clamp_water!(mtile, colstart, colend)

end

# ── Name-dispatched equation-set wrappers (physical_model resolves the config's
#    equation_set string to one of these by name; all keep the moist_compressible
#    prefix so uses_pressure_reference gates their scratch/reference plumbing) ──

"Total-energy moist compressible set on a Cartesian XZ slice (RiRk/RZ grid), 10 vars."
moist_compressible_XZ(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64) =
    mc_driver!(mtile, colstart, colend, t, MCCartesianXZ())

"""
Axisymmetric r–z cylinder on the RiRk/RZ grid (gridpoint column 1 reinterpreted as
radius, so the domain must sit at r > 0), with prognostic tangential wind v
(`MC_VARS_CYL`, 11 vars) and optional f-plane rotation (`physical_params[:f]`).
"""
moist_compressible_axisym(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64) =
    mc_driver!(mtile, colstart, colend, t, MCAxisymRZ())

"3D r–λ–z cylinder on the RLR grid (`MC_VARS_CYL`, 11 vars, f-plane rotation optional)."
moist_compressible_RLR(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64) =
    mc_driver!(mtile, colstart, colend, t, MCCylindricalRLR())

"""
3D Cartesian x–y–z box on the RRR grid (`MC_VARS_CYL`, 11 vars: v is the y-wind,
f-plane rotation optional — +f v / −f u with no curvature terms).
"""
moist_compressible_RRR(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64) =
    mc_driver!(mtile, colstart, colend, t, MCCartesianRRR())

"""
3D spherical θ–λ–z shell on the SLR grid (`MC_VARS_CYL`, 11 vars: u is the
θ-ward wind, v the zonal wind). Shallow atmosphere with metric radius
`physical_params[:sphere_radius]` (default Earth) and full latitude-dependent
Coriolis f = 2Ω cosθ from `physical_params[:Omega]` (NOT the cylinders' f-plane
`:f`). See `MCSphericalSLR`.
"""
moist_compressible_SLR(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64) =
    mc_driver!(mtile, colstart, colend, t, MCSphericalSLR())

"""
    qss_relaxation(Q_ss, rho_v, rho_vs, tau)

Reconciliation of the prognostic supersaturation density toward the supersaturation the
prognostic VAPOR implies, `-(Q_ss - (rho_v - rho_vs))/tau` [kg/m^3/s].

Every water species is prognostic now — vapor, cloud, rain and the twelve ice moments — so
`Q_ss = rho_v - rho_vs(T,p)` is formally REDUNDANT. It is carried prognostically anyway
because it is the quantity the condensation closure reads, and a prognostic, advected `Q_ss`
is smooth where a pointwise difference against a saturation curve is not: near saturation the
supersaturation is four to five decades below the vapor itself, right at the 1e-4 nucleation
gate, so the transported field is the better-conditioned representation of it.

Redundancy has to be reconciled or the two drift apart under splitting error, and this is
that reconciliation — the "extra adjustment step on a slower timescale than the timestep" of
reference/HANDOFF_DIAGNOSED_CLOUD.md. `tau = physical_params[:tau_qss]` (default 10 s) is
deliberately long compared with the timestep, so the transport keeps the smooth field and
only the accumulated inconsistency is removed.

It is free of any effect on the conserved rho_d, rho_t and E_t, and it is thermodynamically
inert unconditionally: [`retrieve_temperature`](@ref) does not read `Q_ss` at all.

# The reconciliation CHAIN

This is the first of two links, and neither is a shock:

    Q_ss  --(tau_qss)-->  rho_v  --(tau_rec)-->  the rho_t density budget

`Q_ss` is pulled onto the prognostic vapor here; the prognostic vapor is pulled onto the
vapor the conserved masses imply by [`rho_v_reconcile`](@ref). `rho_t` anchors the chain and
is never nudged, which is what keeps the conserved water mass exactly conserved.

**Feeding this the vapor was once forbidden, and is now correct.** While the vapor was a
RETRIEVAL, one branch of it was built out of `Q_ss + rho_vs`, and this term then read
`-(Q_ss - ((Q_ss + rho_vs) - rho_vs))/tau == 0`: it SELF-ANNIHILATED wherever the retrieval
trusted the supersaturation residual, i.e. exactly where the anchor was load-bearing (the
`POSITIVITY=1` run reached `|Q_ss|/rho_vs ~ 2e26` *with* the reconciliation running). That
hazard was ALGEBRAIC, not physical, and it is gone with the retrieval: a transported slot is
not `Q_ss + rho_vs` by construction, so the difference this term removes is a genuine
disagreement between two independently carried fields. The anchor to the conserved masses did
not disappear — it moved one link down the chain, onto `rho_v` itself.
"""
@inline function qss_relaxation(Q_ss, rho_v, rho_vs, tau)

    return -(Q_ss - (rho_v - rho_vs)) / tau
end

"""
    rho_v_reconcile(res_rho_t, rho_v, tau_rec)

Reconciliation of the prognostic vapor toward the vapor the conserved masses imply,
`(res_rho_t - rho_v)/tau_rec` [kg/m^3/s], with
`res_rho_t = rho_t - rho_d - rho_c - rho_r - rho_i`.

`rho_t` is kept prognostic as the CONSERVATION ANCHOR (and the semi-implicit structure is
built on it), so with `rho_v` prognostic beside it the set carries one redundancy, and the
whole content of that redundancy is the gap `delta = res_rho_t - rho_v`. This removes it on
`tau_rec = physical_params[:tau_rho_v_rec]` (default 10 s) — the `qss_relaxation` pattern, and
for the same reason: a drift correction on a timescale long compared with the step, never a
per-step projection back onto the budget. Projecting would throw away the transported vapor's
smoothness, which is the thing being bought.

It is thermodynamically INERT. [`retrieve_temperature`](@ref) reads `rho_t`, the condensed
masses and `E_t`, and never the vapor, so moving `rho_v` moves the water PARTITION and the
mixture gas constants that read it — not the temperature, not the energy, and not `rho_t`.
Mass, water and energy are all untouched by construction.

**It is identically zero on a resting base**, bitwise, and that is a construction rather than
a coincidence: the slot is carried against the DERIVED reference
`rho_vbar == rho_tbar - rho_dbar - rho_cbar` ([`mc_reference_diagnostics`](@ref),
[`vapor_slot`](@ref)), which is the same expression `res_rho_t` reassembles from the same
fitted columns. So a resting column has `delta == 0.0` exactly, the nudge contributes nothing,
and the reference state stays the discrete fixed point that
reference/HANDOFF_REFERENCE_STATE.md established it must be.

`max|delta|` over the tile is recorded every step as [`MC_VAPOR_GAP`](@ref MC_VAPOR_GAP) — a
load-bearing drift diagnostic now, not an artifact of a retrieval option: it is the size of
the disagreement between the conserved budget and the transported vapor, and `tau_rec` times
this rate is exactly that gap.

# What FEEDS the gap, and what does not

Every term that moves `rho_t` and `rho_v` by the same amount leaves `delta` alone, and the
tendency block is written so that most of them do: condensation and deposition are equal and
opposite pairs of slot sources with `rho_t` untouched; rain and ice sedimentation move
`rho_t` and the falling species by the SAME fitted flux divergence; the Louis boundary layer
applies `rho_dot_w` to slot 3 and `rho_dot_w - rho_dot_c` to this slot, out of the same two
column fits.

Four things do feed it, and they are the ones to watch:

1. **The semi-implicit acoustic solve.** It slaves `rho_t' -= dtau*dz(phi)` and
   `rho_d' -= dtau*dz(rho_dbar/rho_tbar * phi)`, whose difference is exactly the WATER the
   vertical acoustic mass flux carries. The vapor slot is not in that solve (the SI structure
   is deliberately untouched by Stage A), so that transport reaches the vapor only through
   this nudge. On a resting base both legs are zero and nothing is fed.
2. **The horizontal water mixing** (`Khdiff_water`, off by default), which mixes `rho_t` and
   the condensates and gives this slot no Laplacian.
3. **`clamp_water!`'s floor** (opt-in), which moves a condensate and leaves `rho_t` alone.
4. **The independent ADVECTION of the condensates themselves** — the §2c feeder, and the
   dominant one on the ice arm. Sedimentation telescopes (`rho_t` and the falling species
   receive the same fitted flux divergence), but the advective legs `−u·∇X − X∇·u` of the
   twelve ice moments are fitted on their own spline columns and do NOT telescope against
   slot 3's, so at a sharp glaciation front the summed ice mass detaches from the anchor's
   headroom and this nudge faithfully drags the vapor onto the deficit — the budget "closes"
   on an unphysical partition and `MC_VAPOR_GAP` reads zero. That defect is measured and
   removed one tier up, by [`ice_anchor_rate`](@ref) / [`_ice_anchor_reconcile!`](@ref) on
   `tau_ice_anchor` (census `MC_ANCHOR_GAP`, TeX §Reconciliation of the condensate
   partition).

Whether `rho_t` should eventually be RETIRED instead (with `rho_t = sum of components` by
construction, conservation exact and no nudge at all) is the open question parked in
reference/HANDOFF_ISHMAEL_SESSION.md — it touches the semi-implicit solve, which is where
item 1 above lives.

"""
@inline function rho_v_reconcile(res_rho_t, rho_v, tau_rec)

    return (res_rho_t - rho_v) / tau_rec
end

# ── ETD-AB3: the exponential multistep for the stiff Q_ss relaxation pair ──────
#
# The whole construction is TeX §"Integration of the relaxation pair in the stiff limit"
# (\\label{stiff_integrator}), and the amended departure (b). Glaciation makes the pair
#
#     dQ_ss/dt = N(t) − λ Q_ss ,   λ = 1/τ + 1/τ_i ,   N = F − 𝒟/τ_i
#
# stiff: `1/τ_i` reaches O(10²) s⁻¹ for gram-scale ice held at seeding size, so `Δt/τ` crosses
# AB3's real-axis stability bound (0.545) within tens of seconds of the first homogeneous
# freezing. The stiffness is PHYSICAL and the limit it approaches is the quasi-steady state of
# Eq. wbf_qs, so the integrator is made to land on that limit rather than fall over ahead of
# it: the two relaxations are withheld from the multistep (which then carries `N`, its history
# rotating as usual) and `Q_ss` is advanced by the exact propagator of the frozen-coefficient
# linear problem, with `N` represented by the SAME quadratic extrapolant that defines the AB3
# weights (Nørsett 1969; Hochbruck and Ostermann 2010).
#
# This is the acoustic semi-implicit treatment transposed to the microphysics — the fast linear
# dynamics advanced by its own exact propagator while the slow tendencies keep the multistep,
# the two meeting through an adjustment applied to the predictors — which is why
# `relaxation_adjustment_qss!` sits here, beside `semiimplicit_adjustment_p`, and runs in the
# same window.

"""
    _etd_horner(x, j) -> Float64

`Σ_{m=0}^{K} (−1)^m x^m j!/(m+j)!`, the rescaled series shared by all four kernel moments,
evaluated by the nested form `1 − x/(j+1)·[1 − x/(j+2)·[…]]`.

This is the small-`x` branch of [`_etd_moments`](@ref) and exists for one reason: the closed
forms of `J_1`, `J_2` are differences of terms that cancel to `O(x²)` and `O(x³)`, so their
relative error grows like `ε/x` — at `x = 10⁻⁴` even `expm1` leaves `J_1` good to only ~2e−12,
and `J_2` far worse. The series has no cancellation at all (every term is formed by a divide
and a multiply-add), so the two branches are joined where BOTH are accurate rather than where
either is merely finite. `K = 17` puts the truncation at `x^17/20! ≈ 4e−19` of the leading term
at the `x = 1` switch, i.e. below the roundoff of the closed form it hands over to.
"""
@inline function _etd_horner(x::Float64, j::Int64)

    r = 1.0
    @inbounds for m in 17:-1:1
        r = 1.0 - (x / (j + m)) * r
    end
    return r
end

"""
    _etd_moments(x) -> (J0, J1, J2, D)

The three kernel moments of the exponential Adams-Bashforth update at `x = λΔt`,

    J_k(x) = ∫₀¹ θ^k e^{−x(1−θ)} dθ ,
    J0 = (1 − e^{−x})/x ,  J1 = (x − 1 + e^{−x})/x² ,  J2 = (x² − 2x + 2 − 2e^{−x})/x³ ,

together with the fourth moment the STEP-MEAN needs,

    D(x) = (1/3 − J2)/x = (x³/3 − x² + 2x − 2 + 2e^{−x})/x⁴ ,

which is to `J2` what `J1` is to `J0` (`J1 ≡ (1 − J0)/x` and `J2/2 ≡ (1/2 − J1)/x` are the same
identity one and two rungs up). `D` is what removes the apparent `1/λ` singularity of
Eq. qss_stepmean analytically — see [`etd_step_weights`](@ref).

`x ≥ 0` always: λ is a sum of conductances. `x < 1` takes the series ([`_etd_horner`](@ref)),
`x ≥ 1` the closed forms, whose smallest numerator is `J2`'s `(x−1)² + 1 − 2e^{−x} ≥ 0.264` —
no cancellation anywhere above the switch.
"""
@inline function _etd_moments(x::Float64)

    if x < 1.0
        # j! divided out of the series: J0 = R(1), J1 = R(2)/2!, J2 = 2·R(3)/3!, D = 2·R(4)/4!.
        return (_etd_horner(x, 1),
                _etd_horner(x, 2) / 2.0,
                _etd_horner(x, 3) / 3.0,
                _etd_horner(x, 4) / 12.0)
    end
    em = exp(-x)
    x2 = x * x
    j0 = (1.0 - em) / x
    j1 = ((x - 1.0) + em) / x2
    j2 = (((x2 - (2.0 * x)) + 2.0) - (2.0 * em)) / (x2 * x)
    return (j0, j1, j2, ((1.0 / 3.0) - j2) / x)
end

"""
    etd_step_weights(x, t) -> (e^{−x}, b1, b2, b3, J0, g1, g2, g3)

Every coefficient one gridpoint needs, at `x = λΔt` and integrator step `t`. Pure function of
two numbers; the driver-level tests exercise it directly.


# The update (TeX Eq. etd_ab3)

    Q^{n+1} = e^{−x} Q^n + Δt [ b1 N^n + b2 N^{n−1} + b3 N^{n−2} ] ,
    b1 = J0 + J1 + ½(J1+J2) ,   b2 = −J1 − (J1+J2) ,   b3 = ½(J1+J2) ,

the variation-of-constants integral of `dQ/dt = N − λQ` with `N` represented by the backward
quadratic `N^n + θ∇N^n + ½θ(θ+1)∇²N^n` — the same extrapolant the classical weights integrate,
which is why `b → (23/12, −16/12, 5/12)` as `x → 0`. `t = 1` and `t = 2` mirror
[`explicit_timestep`](@ref)'s Euler and AB2 startup with the one- and two-point extrapolants
(`b1 = J0`; `b1 = J0+J1, b2 = −J1`, classical limits `1` and `3/2, −1/2`).

# The step-mean (TeX Eq. qss_stepmean)

Every consumer of the phase-change rates must see the SAME transfer or the exchange does not
close, and the discrete budget of the update defines it:

    (Q^{n+1} − Q^n)/Δt = N̄ − λ Q̄_ss  ⟺  Q̄_ss = (N̄ − (Q^{n+1} − Q^n)/Δt)/λ

with `N̄` the classical-weight combination of the SAME `N` history. Written that way the
expression is `0/0` at `λ = 0`; substituting the update and using `(1 − e^{−x})/Δt = λ J0`
removes the singularity ANALYTICALLY rather than by a guard:

    Q̄_ss = J0 Q^n + Δt [ g1 N^n + g2 N^{n−1} + g3 N^{n−2} ] ,   g_j ≡ (a_j − b_j)/x

and each `g_j` collapses onto the moments already computed —
`g1 = J1 + ¾J2 + ½D`, `g2 = −(J2 + D)`, `g3 = ¼J2 + ½D` at `t ≥ 3`;
`g1 = J1 + ½J2`, `g2 = −½J2` at `t = 2`; `g1 = J1` at `t = 1`.

Two limits worth checking against the TeX, and both are asserted in the tests:

  * `x → 0` gives `(J0; g) = (1; 19/24, −5/12, 1/8)`, which is `∫₀¹ (1−θ) N(θ) dθ` for the same
    quadratic — the exact step-mean of a non-relaxing trajectory;
  * `x → ∞` gives `g → (23, −16, 5)/(12x)`, i.e. `Q̄_ss → N̄/λ = Q_ss^{qs}`, the Korolev-Mazin
    quasi-steady state of Eq. wbf_qs. The scheme and the derivation agree about the fast limit
    at every `x`, not merely asymptotically, which is the third property the TeX requires.

# `x == 0` is a branch, not a limit

The reduction the TeX states is BITWISE, not asymptotic: "wherever λ = 0 — air with no
droplets, no rain, and no ice, which is the entire dry path — Eq. etd_ab3 IS the third-order
multistep". Evaluating `J0 + J1 + ½(J1+J2)` at `x = 0` in Float64 gives 1.9166666666666665, one
ulp below `23/12`, so the branch returns the literals instead. (The driver goes further and
does not touch the predictor at all where `λ = 0` — see `relaxation_adjustment_qss!` — so the
dry path is bitwise by construction rather than by arithmetic luck.)
"""
@inline function etd_step_weights(x::Float64, t::Int64)

    if x == 0.0
        return t == 1 ? (1.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0) :
               t == 2 ? (1.0, 1.5, -0.5, 0.0, 1.0, 2.0 / 3.0, -1.0 / 6.0, 0.0) :
                        (1.0, 23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0,
                         1.0, 19.0 / 24.0, -5.0 / 12.0, 1.0 / 8.0)
    end
    (j0, j1, j2, d) = _etd_moments(x)
    emx = exp(-x)
    if t == 1
        return (emx, j0, 0.0, 0.0, j0, j1, 0.0, 0.0)
    elseif t == 2
        return (emx, j0 + j1, -j1, 0.0, j0, j1 + (0.5 * j2), -0.5 * j2, 0.0)
    end
    s = 0.5 * (j1 + j2)
    return (emx, j0 + j1 + s, -j1 - (j1 + j2), s,
            j0, j1 + (0.75 * j2) + (0.5 * d), -(j2 + d), (0.25 * j2) + (0.5 * d))
end

"""
    relaxation_adjustment_qss!(mtile, colstart, colend, t)

Apply the exponential propagator of the supersaturation relaxation pair, and the step-mean
phase-change increments that close the exchange against it.

Runs immediately AFTER [`explicit_timestep`](@ref) and BEFORE
[`semiimplicit_adjustment_p`](@ref) — the placement the TeX prescribes, so that the corrected
latent heating enters the acoustic solve at exactly the point the rate-level heating entered it
before. It is the microphysical sibling of the acoustic adjustment in structure as well as in
position: a fast linear mode integrated by its own exact propagator, meeting the multistep
through a correction applied to the predictor.

# Why the arithmetic is not here

`explicit_timestep` ROTATES `expdot_n → expdot_nm1 → expdot_nm2` internally, so by the time
this function runs `N^{n−2}` no longer exists. Everything that needs the three unrotated levels
— `λ`, `x`, the weights, `Q_ss^{n+1}`, `Q̄_ss`, the step-mean rates and every consumer
increment — is therefore computed in `mc_driver!`'s own window, just before the explicit
advance, and stashed in the per-thread scratch columns (`etd_*`, `Qdot_bar`, `Qdot_r_bar`,
`Qdep_bar`; see [`MC_SCRATCH_SLOTS`](@ref)). Same thread, same column, same step, exactly as
`semiimplicit_adjustment_p` reads the `sd_pxi` column staged for it.

# What it applies

  * slot 7 is OVERWRITTEN with the propagated `Q_ss^{n+1}` (in slot units, `Q_ss` minus the
    reference `Q̄_ssbar`) — the multistep predictor for that slot carried `N` only and is
    discarded;
  * slots 1 (pressure), 8 (rain), 9 (cloud), the vapor slot and the three ice mass slots
    receive their withheld phase-change terms back as DIRECT increments over the step, each
    already carrying the same per-slot factor its rate-level term carried (the frozen-at-`n`
    Jacobians `Jr`, `Jc`, `J_i.q` for the transformed slots; the `(R_m/C_vt)(L − R_vC_pT/R_m)`
    bracket for the pressure). Because all of them are built from ONE step-mean rate per
    channel, the vapor removed, the mass condensed or deposited and the heat released are
    identical by construction.

Where `λ = 0` the predictor is left exactly as `explicit_timestep` wrote it, so the dry path —
and any point with no droplets, no rain and no ice — is bitwise the third-order multistep.
Where a channel's step-mean rate is exactly zero its increment is skipped rather than added, so
no slot can pick up a `±0.0` it did not have.
"""
function relaxation_adjustment_qss!(mtile::ModelTile, colstart::Int64, colend::Int64,
                                    t::Int64)
    # `t` is accepted and unused: the startup branch it selects was already applied when the
    # weights were formed, before the rotation. Kept in the signature so this reads as the
    # sibling of `semiimplicit_adjustment_p` at every call site.

    S = @inbounds mtile.mc_scratch[Threads.threadid()]
    IS = mtile.mc_slots
    ice_on = ice_registered(IS)
    vnp1 = mtile.var_np1

    lam = S.etd_lam
    qnp1 = S.etd_qnp1
    d1 = S.etd_d1; d8 = S.etd_d8; d9 = S.etd_d9; dv = S.etd_dv
    rv_i = IS.rho_v

    @inbounds for i in eachindex(lam)
        g = colstart + i - 1
        # λ = 0 is the ENTIRE dry path and every condensate-free point in a moist one: the
        # predictor `explicit_timestep` already wrote IS the exponential update there, so it is
        # left alone rather than recomputed (TeX: the reduction is bitwise, not asymptotic).
        lam[i] != 0.0 && (vnp1[g, 7] = qnp1[i])
        d1[i] != 0.0 && (vnp1[g, 1] += d1[i])
        d8[i] != 0.0 && (vnp1[g, 8] += d8[i])
        d9[i] != 0.0 && (vnp1[g, 9] += d9[i])
        dv[i] != 0.0 && (vnp1[g, rv_i] += dv[i])
    end
    if ice_on
        di1 = S.etd_di1; di2 = S.etd_di2; di3 = S.etd_di3
        da1 = S.etd_da1; da2 = S.etd_da2; da3 = S.etd_da3
        dc1 = S.etd_dc1; dc2 = S.etd_dc2; dc3 = S.etd_dc3
        dn1 = S.etd_dn1; dn2 = S.etd_dn2; dn3 = S.etd_dn3
        i1 = IS.i1_q; i2 = IS.i2_q; i3 = IS.i3_q
        a1 = IS.i1_a; a2 = IS.i2_a; a3 = IS.i3_a
        c1 = IS.i1_c; c2 = IS.i2_c; c3 = IS.i3_c
        n1 = IS.i1_n; n2 = IS.i2_n; n3 = IS.i3_n
        @inbounds for i in eachindex(lam)
            g = colstart + i - 1
            di1[i] != 0.0 && (vnp1[g, i1] += di1[i])
            di2[i] != 0.0 && (vnp1[g, i2] += di2[i])
            di3[i] != 0.0 && (vnp1[g, i3] += di3[i])
            # The habit partition of the same realized increments: axes and (under
            # sublimation) number, evaluated at the identical step-mean rates in the
            # pre-compute. Zero wherever the mass increment is zero, by construction.
            da1[i] != 0.0 && (vnp1[g, a1] += da1[i])
            da2[i] != 0.0 && (vnp1[g, a2] += da2[i])
            da3[i] != 0.0 && (vnp1[g, a3] += da3[i])
            dc1[i] != 0.0 && (vnp1[g, c1] += dc1[i])
            dc2[i] != 0.0 && (vnp1[g, c2] += dc2[i])
            dc3[i] != 0.0 && (vnp1[g, c3] += dc3[i])
            dn1[i] != 0.0 && (vnp1[g, n1] += dn1[i])
            dn2[i] != 0.0 && (vnp1[g, n2] += dn2[i])
            dn3[i] != 0.0 && (vnp1[g, n3] += dn3[i])
        end
    end
    return nothing
end

# ── Semi-implicit adjustment ───────────────────────────────────────────────────

"""
    semiimplicit_adjustment_p(mtile, colstart, colend, t)

Semi-implicit acoustic solve for the total-energy set — the ONLY integrator of the
reference-linear vertical acoustic terms (the explicit acoustic mode was removed; the
AB3 predictor advances the acoustic remainder). Applies the AI2* explicit history
weights (`-1.0 Lⁿ + 0.75 Lⁿ⁻¹`, staged spline-consistently in `mc_driver!`), then
solves the Helmholtz problem `∂z(Δτ² Pξ̄(z) ∂z φ) - φ` for the vertical mass flux
`φ = ρ̄_t w` from the implicit pair `∂φ/∂t = -∂z p'`, `∂p'/∂t = -Pξ̄(z) ∂z φ`, with
`Pξ̄(z)` the LOCAL reference sound speed squared (`mc_ref_diag.Pxi_prof` — a
domain-mean c̄² is the classic SHB78 reference-state instability above Co_z ≈ 4.5
on a stratified base), and recovers `w = φ/ρ̄_t` and `p' = p'* - Δτ Pξ̄(z) ∂z φ`. The density and energy slots are
slaved to the flux in flux form: `rho_t' -= Δτ ∂z φ` (conserves `∫rho_t'`),
`rho_d' -= Δτ ∂z(ρ̄_d/ρ̄_t φ)`, `E_t' -= Δτ ∂z((Ē_t+p̄)/ρ̄_t φ)`. Because the history
staging mirrors this operator chain exactly, no part of the linear acoustic operator
is left under explicit weights — the scheme has no vertical-acoustic Courant limit.
"""
function semiimplicit_adjustment_p(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64;
        apply_histories::Bool=true)
    # apply_histories = false: the exact_si path applies the AI2* explicit
    # history levels in phase A (apply_acoustic_histories!, so the patch-level
    # solve sees the complete X*); the predictors arriving here already carry
    # them and the history roll has been done. The default (true) is the
    # production vertical-only path, bitwise unchanged.

    vars = mtile.model.grid_params.vars
    p_index = vars["p"]
    rhod_index = vars["rho_d"]
    rhot_index = vars["rho_t"]
    w_index = vars["w"]
    et_index = vars["E_t"]
    ts = mtile.model.ts

    S = @inbounds mtile.mc_scratch[Threads.threadid()]

    # Predictors (copies — they are mutated below, so these must NOT be views onto var_np1)
    # and implicit tendency histories (views).
    p_nstar = S.si_p_nstar;       copyto!(p_nstar, view(mtile.var_np1,colstart:colend,p_index))
    w_nstar = S.si_w_nstar;       copyto!(w_nstar, view(mtile.var_np1,colstart:colend,w_index))
    rhod_nstar = S.si_rhod_nstar; copyto!(rhod_nstar, view(mtile.var_np1,colstart:colend,rhod_index))
    rhot_nstar = S.si_rhot_nstar; copyto!(rhot_nstar, view(mtile.var_np1,colstart:colend,rhot_index))
    et_nstar = S.si_et_nstar;     copyto!(et_nstar, view(mtile.var_np1,colstart:colend,et_index))

    # Reference profiles and the LOCAL sound-speed-squared profile (views/vectors —
    # read-only, loop-invariant; see the staging comment in mc_driver!)
    Pxi_bar = mtile.mc_ref_diag.Pxi_prof
    rho_dbar = view(ref_rho_d(mtile.ref_state),:,1)
    rho_dbar_z = view(ref_rho_d(mtile.ref_state),:,2)
    rho_tbar = view(ref_rho_t(mtile.ref_state),:,1)
    rho_tbar_z = view(ref_rho_t(mtile.ref_state),:,2)
    E_tbar = view(ref_total_energy(mtile.ref_state),:,1)
    E_tbar_z = view(ref_total_energy(mtile.ref_state),:,2)
    pbar = view(ref_pressure(mtile.ref_state),:,1)
    pbar_z = view(ref_pressure(mtile.ref_state),:,2)

    # State-dependent linearization (options[:state_dependent_si]): every
    # coefficient of the implicit pair — Pξ, the mass-flux density ρ̂_t, and
    # the slaved-leg chains — comes from the CURRENT column state (the driver's
    # scratch, filled this column on this thread, frozen at time n), and the
    # Helmholtz matrix is refactorized per column with that Pξ profile. The
    # remainder/history staging in mc_driver! uses the SAME coefficients, so
    # the operator-consistency cancellation holds at finite amplitude — the
    # resting-reference form leaves δ·Co_z of the grid-scale operator explicit
    # in a convective core with state deviation δ (reference/SI_CONVECTIVE_CEILING.md).
    sd_si = get(mtile.model.options, :state_dependent_si, false)::Bool
    if sd_si && mtile.solve_data === nothing
        error("options[:state_dependent_si] requires the cubic B-spline (RiRk) " *
              "vertical — the per-column profile Helmholtz is not implemented " *
              "for the Chebyshev vertical")
    end
    # (Pxi_bar and S.sd_pxi are both Vector{Float64}, so this binding is
    # type-stable; ρ̂ is branched at each use site instead — a `? S.rho_t :
    # rho_tbar` union of Vector and SubArray boxes every broadcast it touches,
    # which showed up as per-column allocations in the flag-OFF path.)
    Pxi_vec = sd_si ? S.sd_pxi : Pxi_bar

    # Off-centered AI2* history terms (Durran & Blossey 2012). The AB3 predictor carries
    # only the acoustic REMAINDER, so the linear terms enter purely here: the explicit
    # history levels -1.0 Lⁿ + 0.75 Lⁿ⁻¹ now, the implicit +1.25 L̃ⁿ⁺¹ through the
    # Helmholtz solve below. impdot is staged in mc_driver! with the same discrete
    # operator chain the solve applies, so every AI2* level sees ONE operator (the old
    # subtract-AB3/add-implicit form mixed pointwise and fitted operators, leaving a
    # grid-scale residual under explicit weights — the vertical-acoustic Courant ceiling
    # of reference/SI_VERTICAL_CEILING.md). The first step is AM2 trapezoidal (+0.5 Lⁿ with the
    # ts_term = 0.5·ts solve); its history seed gives t == 2 the full AI2* weights.
    ts_term = (t == 1) ? 0.5 * ts : 1.25 * ts
    if apply_histories
        for index in (p_index, w_index, rhod_index, rhot_index, et_index)
            nstar = index == p_index ? p_nstar :
                    index == w_index ? w_nstar :
                    index == rhod_index ? rhod_nstar :
                    index == rhot_index ? rhot_nstar : et_nstar
            dot_n = view(mtile.impdot_n,colstart:colend,index)
            dot_nm1 = view(mtile.impdot_nm1,colstart:colend,index)
            if (t == 1)
                nstar .= @. nstar + (ts * 0.5 * dot_n)
            else
                nstar .= @. nstar + (ts * ((0.75 * dot_nm1) - dot_n))
            end
            dot_nm1 .= dot_n
        end
    end

    # Take the vertical derivative of the p' predictor (coefficient 1: the pair is
    # ∂φ/∂t = -∂z p'; Pxi_bar enters in the p' update instead)
    # Distinct scratch columns per variable: `p_nstar` below aliases `p_col.uMish`, and is
    # still read after `phi_col` has been transformed — so these two must not be the same
    # object. Keyed on p_index vs w_index, they are not.
    p_col = scratch_column(mtile, p_index)
    p_col.uMish .= p_nstar
    Btransform!(p_col)
    Atransform!(p_col)
    p_nstar = Itransform!(p_col)
    p_nstar_z = S.si_p_nstar_z
    Ixtransform(p_col, p_nstar_z)
    p_nstar_z .*= ts_term

    # Mass-flux Helmholtz RHS (φ = ρ̄_t w): rhs = Δτ ∂z p'* - ρ̄_t w*.
    # Elimination gives (∂z(Δτ² Pξ̄(z) ∂z ·) - I) φ (profile coefficient — the
    # weighted-stiffness Galerkin form on RiRk).
    rhs = S.si_rhs
    if sd_si
        @. rhs = p_nstar_z - (S.rho_t * w_nstar)
    else
        @. rhs = p_nstar_z - (rho_tbar * w_nstar)
    end
    phi_col = scratch_column(mtile, w_index)
    if sd_si
        # Per-column, per-step factorization with the state's Pξⁿ profile
        # (w-Dirichlet rows, flags passed explicitly — no registry traffic).
        @. S.sd_alpha = (ts_term * ts_term) * Pxi_vec
        h_sd = _assemble_sd_helmholtz(mtile.solve_data, S.sd_alpha, true, true)
        _vertical_solve!(phi_col, h_sd, rhs, mtile; dirichlet=(true, true))
    elseif t == 1
        # Calculate the Helmholtz matrix for the first time step
        h_a = calc_Helmholtz_semiimplicit_matrix(mtile.tile, mtile.model, Pxi_bar, ts_term)
        _vertical_solve!(phi_col, h_a, rhs, mtile)
    else
        # Use the pre-calculated one
        _vertical_solve!(phi_col, mtile.h_matrix, rhs, mtile)
    end

    phi = Itransform!(phi_col)
    phi_z = S.si_phi_z
    Ixtransform(phi_col, phi_z)

    # Recover w_n+1 = φ_n+1 / ρ̂_t
    if sd_si
        view(mtile.var_np1,colstart:colend,w_index) .= phi ./ S.rho_t
    else
        view(mtile.var_np1,colstart:colend,w_index) .= phi ./ rho_tbar
    end

    # Recover p'_n+1 = p'* - Δτ Pξ ∂z φ_n+1
    view(mtile.var_np1,colstart:colend,p_index) .= p_nstar .- (ts_term .* Pxi_vec .* phi_z)

    # Slaved flux-form updates ∂z(c φ) = c φ_z + c_z φ, pointwise from the solve's
    # φ and φ_z — the exact form the AI2* history staging in mc_driver! mirrors, so
    # the slaved slots see one discrete operator across all time levels. rho_t' has
    # c = 1 exactly (conserves ∫rho_t'); with a dry reference the rho_d' update is
    # then IDENTICAL to rho_t''s, so the two densities cannot drift apart.
    view(mtile.var_np1,colstart:colend,rhot_index) .= rhot_nstar .- (ts_term .* phi_z)

    c_d = S.si_c_d
    c_d_z = S.si_c_d_z
    if sd_si
        @. c_d = S.rho_d / S.rho_t
        @. c_d_z = ((S.rho_d_z * S.rho_t) - (S.rho_d * S.rho_t_z)) / (S.rho_t^2)
    else
        @. c_d = rho_dbar / rho_tbar
        @. c_d_z = ((rho_dbar_z * rho_tbar) - (rho_dbar * rho_tbar_z)) / (rho_tbar^2)
    end
    view(mtile.var_np1,colstart:colend,rhod_index) .=
        rhod_nstar .- (ts_term .* ((c_d .* phi_z) .+ (c_d_z .* phi)))

    c_e = S.si_c_e
    c_e_z = S.si_c_e_z
    if sd_si
        @. c_e = (S.E_t + S.p) / S.rho_t
        @. c_e_z = (((S.E_t_z + S.p_z) * S.rho_t) -
                    ((S.E_t + S.p) * S.rho_t_z)) / (S.rho_t^2)
    else
        @. c_e = (E_tbar + pbar) / rho_tbar
        @. c_e_z = (((E_tbar_z + pbar_z) * rho_tbar) -
                    ((E_tbar + pbar) * rho_tbar_z)) / (rho_tbar^2)
    end
    view(mtile.var_np1,colstart:colend,et_index) .=
        et_nstar .- (ts_term .* ((c_e .* phi_z) .+ (c_e_z .* phi)))

    # Store the applied implicit w increment as the next step's w history (see the
    # staging comment in mc_driver!): the exact operator the Helmholtz elimination
    # applied to the w leg, which no pointwise staging can reproduce. On the first
    # step, seed the n-1 level too so t == 2 has a full AI2* history on the w leg.
    view(mtile.impdot_n,colstart:colend,w_index) .=
        (view(mtile.var_np1,colstart:colend,w_index) .- w_nstar) ./ ts_term
    if t == 1
        view(mtile.impdot_nm1,colstart:colend,w_index) .=
            view(mtile.impdot_n,colstart:colend,w_index)
    end
end

# ── Implicit vertical diffusion ────────────────────────────────────────────────

"""
    diffusion_timestep_mc(mtile, colstart, colend, t)

Implicit vertical diffusion of `u` and `w` (momentum, `Kvdiff`), the moist entropy `s_t'`
(heat, `Kvdiff_heat`) and the water species (`Kvdiff_water`: total water `rho_w'`, cloud
`rho_c'`, rain `rho_r`) for the total-energy set, using the AI2* off-centered weights
(`+1.25 N^{n+1} - 1.0 N^n + 0.75 N^{n-1}`) with the tendency history in `mtile.diffdot_*`.
It needs its own tendency channel because the acoustic solve owns `impdot[w/p/E_t]`.
The momentum, heat and water solves are gated independently on their coefficients: a K = 0
solve is not the identity (it refits the column and reapplies the spectral filter), so a
zero coefficient must skip its solve entirely, leaving the post-acoustic state untouched.

Runs AFTER [`semiimplicit_adjustment_p`](@ref). The density fields are NOT re-slaved to the
post-diffusion `w` — momentum diffusion is a force, not a mass flux, and they were already
advanced with the acoustic flux divergence (which conserves `∫rho_t'`). The residual is the
usual O(ts·Kv) splitting error.

The heat and water paths retrieve the post-acoustic temperature (the star state) through
the same closed-form retrieval `mc_driver!` uses; the resting base stays bit-preserved
because the reference profile `s_tbar` comes from the same pipeline
(`mc_reference_diagnostics`).

Energy routing:

- **Friction is a resolved-KE sink, not heat.** With an eddy diffusivity the resolved KE the
  momentum solve removes goes to the subgrid cascade (the future TKE shear production), and
  the molecular heating is negligible. So `E_t += rho_t δke` (E_t follows the KE down),
  holding internal energy — `T` and `p` are unchanged by friction.
- **Heat.** The `s_t` increment maps at fixed `rho_d` and composition via `∂s_t/∂T = C_vt/T`:
  `δT = (T/C_vt) δs_t`, `δE_t = rho_d T δs_t`, `δp = rho_d R_m δT`, and
  `δQ_ss = -(∂ρ_vs/∂T δT + ∂ρ_vs/∂p δp)` (so `Q_ss = ρ_v - ρ_vs` tracks the diabatic heating).
- **Water.** The species increments map at FIXED temperature (verified against the
  retrieval: `δF(T) ≡ 0` under this map), moving the partial pressure and the internal +
  potential energy the water carries:
  `δp = R_v T δρ_v`, `δE_t = (C_pv T − L_v + ke + gz) δρ_w + (L_v − R_v T) δρ_v`,
  `δQ_ss = δρ_v − ∂ρ_vs/∂p · δp`, with the VAPOR increment `δρ_v = δρ_w − δρ_c − δρ_r`
  implied. Each species' Neumann solve conserves its own `∫ρ`; `∫E_t` is conserved only
  approximately (the exact multicomponent enthalpy flux needs a flux form the per-column
  architecture excludes — see the handoff doc; the residual is O(K_water × vertical
  variation of e_l + gz) and shows up in the energy-drift diagnostic).
"""
diffusion_timestep_mc(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64) =
    diffusion_timestep_mc(mtile, colstart, colend, t, MCCartesianXZ())

function diffusion_timestep_mc(mtile::ModelTile, colstart::Int64, colend::Int64, t::Int64,
                               geom::MCGeometry)

    vars = mtile.model.grid_params.vars
    p_index = vars["p"]
    rhod_index = vars["rho_d"]
    rhot_index = vars["rho_t"]
    u_index = vars["u"]
    w_index = vars["w"]
    et_index = vars["E_t"]
    qss_index = vars["Q_ss"]
    rhor_index = mc_slot(vars, "rho_r")
    rhoc_index = mc_slot(vars, "rho_c")
    rhov_index = mc_slot(vars, "rho_v")

    ts = mtile.model.ts

    Kvdiff = mtile.model.physical_params[:Kvdiff]
    Kvdiff_heat = get(mtile.model.physical_params, :Kvdiff_heat, Kvdiff)
    Kvdiff_water = get(mtile.model.physical_params, :Kvdiff_water, 0.0)
    do_momentum = Kvdiff > 0.0
    do_heat = Kvdiff_heat > 0.0
    do_water = Kvdiff_water > 0.0

    pbar = view(ref_pressure(mtile.ref_state),:,1)
    rho_dbar = view(ref_rho_d(mtile.ref_state),:,1)
    rho_tbar = view(ref_rho_t(mtile.ref_state),:,1)
    E_tbar = view(ref_total_energy(mtile.ref_state),:,1)
    rho_cbar = view(Springsteel.ref_rho_c(mtile.ref_state),:,1)
    # The DERIVED reference the vapor slot is carried against; see `mc_reference_diagnostics`.
    rho_vbar = mtile.mc_ref_diag.rho_vbar
    z = view(mtile.tilepoints,colstart:colend,zcoord(geom))

    S = @inbounds mtile.mc_scratch[Threads.threadid()]

    # Post-acoustic totals (the star state)
    vnp1 = mtile.var_np1
    # Bind the views OUTSIDE the `@.` blocks below — `@.` dots every call in the expression,
    # `view` included, which broadcasts the view itself instead of its contents.
    u_v = view(vnp1,colstart:colend,u_index)
    w_v = view(vnp1,colstart:colend,w_index)
    p_v = view(vnp1,colstart:colend,p_index)
    rhod_v = view(vnp1,colstart:colend,rhod_index)
    rhot_v = view(vnp1,colstart:colend,rhot_index)
    et_v = view(vnp1,colstart:colend,et_index)
    qss_v = view(vnp1,colstart:colend,qss_index)
    rhor_v = view(vnp1,colstart:colend,rhor_index)
    rhoc_v = view(vnp1,colstart:colend,rhoc_index)
    rhov_v = view(vnp1,colstart:colend,rhov_index)

    u_star = S.df_u_star; copyto!(u_star, u_v)
    w_star = S.df_w_star; copyto!(w_star, w_v)
    v_v = mc_v_np1_view(geom, vnp1, colstart, colend, vars)
    v_star = mc_v_star!(geom, S, v_v)
    p_star = S.df_p_star;         @. p_star = p_v + pbar
    rho_d_star = S.df_rho_d_star; @. rho_d_star = rhod_v + rho_dbar
    rho_t_star = S.df_rho_t_star; @. rho_t_star = rhot_v + rho_tbar

    # The heat and water maps need the retrieved star-state thermodynamics; the
    # momentum-only path skips the retrieval entirely.
    T_star = S.df_T_star
    p_hPa_star = S.df_p_hPa_star
    drvs_dT = S.df_drvs_dT
    drvs_dp = S.df_drvs_dp
    rho_v_star = S.df_rho_v_star
    C_vt_star = S.df_C_vt_star
    R_m_star = S.df_R_m_star
    Lv_star = S.df_Lv_star
    ke_star = S.df_ke_star
    stp_star = S.df_stp_star
    rho_c_star = S.df_rho_c_star
    rho_r_star = S.df_rho_r_star
    rho_liq_star = S.df_rho_liq_star
    if do_heat || do_water
        E_t_star = S.df_E_t_star;   @. E_t_star = et_v + E_tbar
        mc_ke_star!(ke_star, geom, u_star, w_star, v_star)
        M_star = S.df_M_star
        @. M_star = p_star + E_t_star - (rho_t_star * (ke_star + (gravity * z)))
        # Same closed-form retrieval and residual partition as mc_driver!: the
        # condensate is prognostic, so the star state needs no iteration and no clamp.
        # Slot 9 recovery, same rule as mc_driver!: the slot holds the control variable and
        # the density is recovered pointwise. `:none` is bitwise `rhoc_v + rho_cbar`.
        ct = condensate_transform_mode(mtile.model.options)
        if ct === :none
            @. rho_c_star = rhoc_v + rho_cbar
        else
            cmu_s = get(mtile.model.physical_params, :condensate_mu, 1.0e-7)
            if ct === :bhyp
                @. rho_c_star = ahyp(rhoc_v + bhyp(rho_cbar, cmu_s), cmu_s)
            else
                @. rho_c_star = ahyp_smooth(rhoc_v + bhyp(rho_cbar, cmu_s), cmu_s)
            end
        end
        # Slot 8 likewise. Rain has no reference profile, so the slot IS the control variable
        # and `:none` is a copy of identical doubles — bitwise inert.
        rt = rain_transform_mode(mtile.model.options)
        if rt === :none
            copyto!(rho_r_star, rhor_v)
        else
            rmu_s = get(mtile.model.physical_params, :rain_mu, 1.0e-7)
            if rt === :bhyp
                @. rho_r_star = ahyp(rhor_v, rmu_s)
            else
                @. rho_r_star = ahyp_smooth(rhor_v, rmu_s)
            end
        end
        # Same thermodynamic interface as mc_driver!, so the same floor applies: T_star and
        # q_l_star are read through it, and a star state that disagreed with the n state
        # about what the liquid IS would put the difference into every increment built here.
        # `:none` (the default) leaves this a copy of identical doubles -- bitwise inert.
        # NOT applied to `rho_v_star` two lines down: that is the water partition, and
        # flooring a partition manufactures vapor. See `condensate_floor_mode`.
        if condensate_floor_mode(mtile.model.options)
            @. rho_liq_star = max(rho_c_star, 0.0) + max(rho_r_star, 0.0)
        else
            @. rho_liq_star = rho_c_star + rho_r_star
        end
        # The ICE mass of the star state, recovered from the three appended mass slots the
        # same way slot 8 is (all three are TOTALS under one transform family). `Kvdiff_water`
        # with ice is refused in `mc_driver!`, so nothing here DIFFUSES the ice — but the
        # star-state thermodynamics still has to SEE it, or `T_star`, `C_vt_star` and
        # `stp_star` would describe a column with the ice mass removed and the heat increment
        # built from them would carry that error. Exact zeros with ice off, hence bitwise.
        rho_ice_star = S.df_rho_ice_star
        rho_ice_t_star = S.df_rho_ice_t_star
        if ice_registered(mtile.mc_slots)
            it = ice_transform_mode(mtile.model.options)
            imu_s = ice_mu(mtile.model.physical_params, 1)
            i1q_v = view(vnp1, colstart:colend, mtile.mc_slots.i1_q)
            i2q_v = view(vnp1, colstart:colend, mtile.mc_slots.i2_q)
            i3q_v = view(vnp1, colstart:colend, mtile.mc_slots.i3_q)
            @. rho_ice_star = (recover_total(i1q_v, it, imu_s) +
                               recover_total(i2q_v, it, imu_s)) +
                              recover_total(i3q_v, it, imu_s)
            if condensate_floor_mode(mtile.model.options)
                @. rho_ice_t_star = (max(recover_total(i1q_v, it, imu_s), 0.0) +
                                     max(recover_total(i2q_v, it, imu_s), 0.0)) +
                                    max(recover_total(i3q_v, it, imu_s), 0.0)
            else
                copyto!(rho_ice_t_star, rho_ice_star)
            end
        else
            fill!(rho_ice_star, 0.0)
            fill!(rho_ice_t_star, 0.0)
        end
        @. T_star = retrieve_temperature(M_star, rho_d_star, rho_t_star, rho_liq_star,
                                         rho_ice_t_star)
        @. p_hPa_star = p_star / 100.0
        @. drvs_dT = drho_vsat_dT(T_star, p_hPa_star)
        @. drvs_dp = drho_vsat_dp(T_star, p_hPa_star)
        rho_vs_star = S.df_rho_vs_star
        @. rho_vs_star = rho_v_sat(T_star, p_hPa_star)
        # The PROGNOSTIC vapor of the star state, `ρ_v'* + ρ̄_v` — the same
        # `perturbation + derived reference` construction `mc_driver!` stages, so the star
        # state and the n state agree about what the vapor IS. It used to be reassembled here
        # as the density residual, which is now a DIFFERENT number by the reconciliation gap;
        # reading the slot is what keeps the two phases consistent.
        @. rho_v_star = rhov_v + rho_vbar
        q_v_star = S.df_q_v_star
        q_l_star = S.df_q_l_star
        q_i_star = S.df_q_i_star
        @. q_v_star = rho_v_star / rho_d_star
        @. q_l_star = rho_liq_star / rho_d_star     # thermodynamic reader: floored if enabled
        @. q_i_star = rho_ice_t_star / rho_d_star   # ditto; exactly 0.0 with ice off
        @. C_vt_star = Cvd + (q_v_star * Cvv) + (q_l_star * Cl) + (q_i_star * Ci)
        @. R_m_star = Rd + (q_v_star * Rv)
        @. Lv_star = L_v(T_star)
        @. stp_star = moist_entropy_total(T_star, rho_d_star, q_v_star, q_l_star, q_i_star)
        stp_star .-= mtile.mc_ref_diag.s_tbar
    end

    # Implicit tendency histories (slot 6 carries the s_t ENTROPY tendency, not an E_t
    # one; slots 3/7/8 carry the water tendencies)
    udot_n = view(mtile.diffdot_n,colstart:colend,u_index)
    udot_nm1 = view(mtile.diffdot_nm1,colstart:colend,u_index)
    wdot_n = view(mtile.diffdot_n,colstart:colend,w_index)
    wdot_nm1 = view(mtile.diffdot_nm1,colstart:colend,w_index)
    sdot_n = view(mtile.diffdot_n,colstart:colend,et_index)
    sdot_nm1 = view(mtile.diffdot_nm1,colstart:colend,et_index)

    # Pre-factorized in createModelTile, per variable and per timestep coefficient (the
    # first step uses the AM2 coefficient), so nothing is factorized per column here.
    mats = mtile.mc_diffusion_matrices

    # One scratch column serves all three solves: the boundary conditions live in the
    # factorization, and neither `_vertical_solve!` nor `Itransform!` consults the
    # column's own BCs (only Btransform!/Atransform! do). Same reuse as
    # `diffusion_timestep_pd`, which shares one column across five variables.
    col = scratch_column(mtile, u_index)

    # ── Momentum (Kvdiff): friction is a resolved-KE sink, E_t follows the KE down;
    #    T, p, Q_ss held. ──
    dE_visc = S.df_dE_visc
    if do_momentum
        u_nstar = S.df_u_nstar
        w_nstar = S.df_w_nstar
        if (t == 1)
            # Use trapezoidal method (AM2) for first step
            @. u_nstar = u_star + (ts * 0.5 * udot_n)
            @. w_nstar = w_star + (ts * 0.5 * wdot_n)
        else
            # Use AI2* for second step and beyond
            @. u_nstar = u_star - (ts * udot_n) + (ts * 0.75 * udot_nm1)
            @. w_nstar = w_star - (ts * wdot_n) + (ts * 0.75 * wdot_nm1)
        end
        udot_nm1 .= udot_n
        wdot_nm1 .= wdot_n

        h_u = (t == 1) ? mats.u_first : mats.u
        h_w = (t == 1) ? mats.w_first : mats.w
        # u_np1 and w_np1 must be copied out: `col` is reused by the next solve, and
        # Itransform! returns the column's own buffer.
        u_np1 = S.df_u_np1
        w_np1 = S.df_w_np1
        _vertical_solve!(col, h_u, u_nstar, mtile)
        copyto!(u_np1, Itransform!(col))

        _vertical_solve!(col, h_w, w_nstar, mtile)
        copyto!(w_np1, Itransform!(col))

        v_np1 = mc_v_diffusion_solve!(geom, S, mtile, col, mats, ts, t,
                                      colstart, colend, v_star)

        dke = S.df_dke
        mc_dke!(dke, geom, u_np1, w_np1, u_star, w_star, v_np1, v_star)
        @. dE_visc = rho_t_star * dke

        u_v .= u_np1
        w_v .= w_np1
        mc_assign_v!(geom, v_v, v_np1)
    end

    # ── Heat (Kvdiff_heat): moist entropy increment -> (T, p, E_t, Q_ss) at fixed
    #    rho_d and composition via ∂s_t/∂T = C_vt/T (dry limit: C_vt=C_vd, R_m=R_d). ──
    dE_h = S.df_dE_h
    if do_heat
        s_nstar = S.df_s_nstar
        if (t == 1)
            @. s_nstar = stp_star + (ts * 0.5 * sdot_n)
        else
            @. s_nstar = stp_star - (ts * sdot_n) + (ts * 0.75 * sdot_nm1)
        end
        sdot_nm1 .= sdot_n

        h_h = (t == 1) ? mats.heat_first : mats.heat
        _vertical_solve!(col, h_h, s_nstar, mtile)
        stp_np1 = Itransform!(col)

        ds_t = S.df_ds_t; @. ds_t = stp_np1 - stp_star
        dT_h = S.df_dT_h; @. dT_h = (T_star / C_vt_star) * ds_t
        @. dE_h = rho_d_star * T_star * ds_t
        dp_h = S.df_dp_h; @. dp_h = rho_d_star * R_m_star * dT_h
        dQ_h = S.df_dQ_h; @. dQ_h = -((drvs_dT * dT_h) + (drvs_dp * dp_h))

        p_v .+= dp_h
        qss_v .+= dQ_h
    end

    # ── Water (Kvdiff_water): rho_w' on the rho_t operator, rho_v', rho_c' and rho_r each
    #    on its own; increments map at FIXED temperature (the retrieval is invariant under
    #    them), moving the vapor partial pressure and the water's internal + potential
    #    energy. Nothing is implied any more -- every species is solved.
    #    Lives in its own function: inlined here it pushed diffusion_timestep_mc past
    #    the optimizer's budget and one broadcast-into-view stopped eliding its
    #    SubArray (one 64-byte allocation per column per step). ──
    dE_w = S.df_dE_w
    if do_water
        # `_diffusion_water_step!` solves the implicit vertical diffusion for slots 8 and 9 as
        # DENSITIES (`drc = rhoc_np1 - rhoc_star` feeds the mass and energy maps directly).
        # Under a transform the slot is a control variable, so the solve would have to run in
        # nu-space and the increment be `ahyp(nu_np1) - ahyp(nu_star)`. That is not written
        # yet, and `Kvdiff_water` is 0.0 in every shipped configuration (o01_rainfall.jl:89,
        # tc/tc_init.jl:205), so the combination has never run. Fail loudly rather than
        # silently solve the wrong equation for the water.
        (condensate_transform_mode(mtile.model.options) === :none &&
         rain_transform_mode(mtile.model.options) === :none) ||
            error("a water transform with physical_params[:Kvdiff_water] > 0 is not " *
                  "implemented: the implicit water diffusion solves slots 8 and 9 as " *
                  "densities, and under a transform the affected slot must be solved in the " *
                  "control variable with the increment mapped through ahyp. Set " *
                  "Kvdiff_water = 0 or implement the nu-space solve in " *
                  "_diffusion_water_step!.")
        _diffusion_water_step!(mtile, S, col, mats, ts, t, colstart, colend, z,
                               rhod_v, rhot_v, rhor_v, rhoc_v, rhov_v, p_v, qss_v)
    end

    # E_t update: keep the single fused add when the historical pair ran (bit-identical
    # to the pre-water combined update), and touch E_t only with computed terms.
    if do_momentum && do_heat
        et_v .+= dE_visc .+ dE_h
    elseif do_momentum
        et_v .+= dE_visc
    elseif do_heat
        et_v .+= dE_h
    end
    if do_water
        # Explicit loop, not broadcast: as the last branch of this large function
        # both the broadcast and @turbo forms stopped eliding the destination
        # SubArray wrapper (one 64-byte heap allocation per column per step)
        @inbounds for i in eachindex(dE_w)
            et_v[i] += dE_w[i]
        end
    end
end

"""
    _diffusion_water_step!(mtile, S, col, mats, ts, t, colstart, colend, z,
                           rhod_v, rhot_v, rhor_v, rhoc_v, rhov_v, p_v, qss_v)

The water-species half of [`diffusion_timestep_mc`](@ref): AM2/AI2* staging and
implicit solves for rho_w' (on the rho_t operator), rho_v', rho_c' and rho_r (each on its
own), then the fixed-temperature increment maps onto rho_t, rho_v, rho_c, rho_r, p and Q_ss.
The energy increment lands in `S.df_dE_w`; the CALLER adds it to E_t so the
historical dE_visc + dE_h fusion stays bit-identical. Star-state fields
(`df_T_star` etc.) are read from the scratch where the caller computed them.

**The vapor increment is SOLVED, not implied.** It used to be `drho_w - drho_c - drho_r`,
because the vapor was a residual and diffusing the other three WAS diffusing it. With rho_v
prognostic that identity is gone: it gets its own AI2*/AM2 staging and its own factorization
on its own boundary conditions, exactly like rho_c and rho_r. The four increments therefore no
longer close pointwise — `drho_w - (drho_v + drho_c + drho_r)` is a new contribution to the
reconciliation gap, which [`rho_v_reconcile`](@ref) removes on `tau_rec`. That is the same
trade the whole stage makes: rho_t stays the exactly-conserved anchor, and the partition
closes on the nudge timescale rather than instantaneously.

The pressure, Q_ss and latent-energy maps read the SOLVED `drv`, since that is the vapor mass
this step actually moved; `rho_t` keeps taking `drw`, so the conserved total water is
untouched by the change.

`@noinline` in its own function on purpose: inlined into diffusion_timestep_mc this
block pushed the function past the optimizer's budget and one broadcast-into-view
stopped eliding its SubArray — one 64-byte heap allocation per column per step.
"""
@noinline function _diffusion_water_step!(mtile::ModelTile, S, col, mats,
                                          ts::Float64, t::Int64,
                                          colstart::Int64, colend::Int64, z,
                                          rhod_v, rhot_v, rhor_v, rhoc_v, rhov_v, p_v, qss_v)
    vars = mtile.model.grid_params.vars
    rhot_index = vars["rho_t"]
    rhor_index = mc_slot(vars, "rho_r")
    rhoc_index = mc_slot(vars, "rho_c")
    rhov_index = mc_slot(vars, "rho_v")

    T_star = S.df_T_star
    Lv_star = S.df_Lv_star
    ke_star = S.df_ke_star
    drvs_dp = S.df_drvs_dp
    dE_w = S.df_dE_w

    # Every diffused species is prognostic: total water (on the rho_t operator), vapor,
    # cloud and rain. Four solves, four independent increments, nothing implied.
    rw_star = S.df_rw_star; @. rw_star = rhot_v - rhod_v
    rc_star = S.df_rc_star; copyto!(rc_star, rhoc_v)
    rv_star = S.df_rv_star; copyto!(rv_star, rhov_v)

    rwdot_n = view(mtile.diffdot_n,colstart:colend,rhot_index)
    rwdot_nm1 = view(mtile.diffdot_nm1,colstart:colend,rhot_index)
    rcdot_n = view(mtile.diffdot_n,colstart:colend,rhoc_index)
    rcdot_nm1 = view(mtile.diffdot_nm1,colstart:colend,rhoc_index)
    rrdot_n = view(mtile.diffdot_n,colstart:colend,rhor_index)
    rrdot_nm1 = view(mtile.diffdot_nm1,colstart:colend,rhor_index)
    rvdot_n = view(mtile.diffdot_n,colstart:colend,rhov_index)
    rvdot_nm1 = view(mtile.diffdot_nm1,colstart:colend,rhov_index)

    rw_nstar = S.df_rw_nstar
    rc_nstar = S.df_rc_nstar
    rr_nstar = S.df_rr_nstar
    rv_nstar = S.df_rv_nstar
    if (t == 1)
        @. rw_nstar = rw_star + (ts * 0.5 * rwdot_n)
        @. rc_nstar = rc_star + (ts * 0.5 * rcdot_n)
        @. rr_nstar = rhor_v + (ts * 0.5 * rrdot_n)
        @. rv_nstar = rv_star + (ts * 0.5 * rvdot_n)
    else
        @. rw_nstar = rw_star - (ts * rwdot_n) + (ts * 0.75 * rwdot_nm1)
        @. rc_nstar = rc_star - (ts * rcdot_n) + (ts * 0.75 * rcdot_nm1)
        @. rr_nstar = rhor_v - (ts * rrdot_n) + (ts * 0.75 * rrdot_nm1)
        @. rv_nstar = rv_star - (ts * rvdot_n) + (ts * 0.75 * rvdot_nm1)
    end
    rwdot_nm1 .= rwdot_n
    rcdot_nm1 .= rcdot_n
    rrdot_nm1 .= rrdot_n
    rvdot_nm1 .= rvdot_n

    h_rw = (t == 1) ? mats.water_first : mats.water
    h_rc = (t == 1) ? mats.water_c_first : mats.water_c
    h_rr = (t == 1) ? mats.water_r_first : mats.water_r
    h_rv = (t == 1) ? mats.water_v_first : mats.water_v
    rw_np1 = S.df_rw_np1
    rc_np1 = S.df_rc_np1
    rv_np1 = S.df_rv_np1
    _vertical_solve!(col, h_rw, rw_nstar, mtile)
    copyto!(rw_np1, Itransform!(col))
    _vertical_solve!(col, h_rc, rc_nstar, mtile)
    copyto!(rc_np1, Itransform!(col))
    _vertical_solve!(col, h_rv, rv_nstar, mtile)
    copyto!(rv_np1, Itransform!(col))
    _vertical_solve!(col, h_rr, rr_nstar, mtile)
    rr_np1 = Itransform!(col)

    drw = S.df_drw; @. drw = rw_np1 - rw_star
    drc = S.df_drc; @. drc = rc_np1 - rc_star
    drr = S.df_drr; @. drr = rr_np1 - rhor_v
    drv = S.df_drv; @. drv = rv_np1 - rv_star
    @. dE_w = (((Cpv * T_star) - Lv_star + ke_star + (gravity * z)) * drw) +
              ((Lv_star - (Rv * T_star)) * drv)

    rhot_v .+= drw
    rhoc_v .+= drc
    rhor_v .+= drr
    rhov_v .+= drv
    p_v .+= Rv .* T_star .* drv
    qss_v .+= drv .- (drvs_dp .* (Rv .* T_star .* drv))
    return nothing
end

# ── Initial conditions and reference writer ────────────────────────────────────

"""
    write_exact_ref_mc(path, z, p_Pa, rho_d, rho_v, rho_c)

Write a pressure-based exact reference state file (`z p rho_d rho_v rho_c` per line,
p in Pa) in the format read by `Springsteel.exact_pressure_reference_state`. `z`
values are written with `string()` so they match the model gridpoints exactly when
generated in-process.
"""
function write_exact_ref_mc(path::String, z::Vector{Float64}, p_Pa::Vector{Float64},
                            rho_d::Vector{Float64}, rho_v::Vector{Float64},
                            rho_c::Vector{Float64})
    open(path, "w") do f
        for i in 1:length(z)
            println(f, "$(z[i]) $(p_Pa[i]) $(rho_d[i]) $(rho_v[i]) $(rho_c[i])")
        end
    end
    return path
end

"""
    theta_bubble_mc!(patch, gridpoints, ref; xc, xr, zc, zr, dtheta_max)

Dry warm-bubble initial condition for the total-energy set on a
`PressureReferenceState`. The constant-pressure θ perturbation mirrors
[`theta_bubble_pd!`](@ref) but writes the moist_compressible slots: p' = 0 (constant
pressure), rho_d' = rho_t', E_t' from the BF02 internal energy at rest, and Q_ss'
tracking -ρ_v*(T, p̄) in the dry air.
"""
function theta_bubble_mc!(patch::AbstractGrid, gridpoints::Matrix{Float64},
                          ref::Springsteel.PressureReferenceState;
                          xc=10000.0, xr=2000.0, zc=2000.0, zr=2000.0, dtheta_max=2.0,
                          condensate_transform::Symbol=:none, condensate_mu=1.0e-7,
                          rain_transform::Symbol=:none, rain_mu=1.0e-7)
    vars = patch.params.vars
    p_i = vars["p"]; rho_d_i = vars["rho_d"]; rho_t_i = vars["rho_t"]
    et_i = vars["E_t"]; qss_i = vars["Q_ss"]; rho_r_i = mc_slot(vars, "rho_r")
    rho_c_i = mc_slot(vars, "rho_c")
    # The prognostic VAPOR slot, present in every configuration of this set (see `MC_VARS`).
    rho_v_i = mc_slot(vars, "rho_v")
    # APPENDED optional slots: seeded only where they exist. `rain_number_slot(0.0, ...)` is
    # exactly 0.0 under every transform (`bhyp(0) == 0`), so no transform keyword has to be
    # threaded here — the same reason `rain_slot` needed none.
    n_r_i = mc_optional_slot(vars, "n_r")
    ice_i = mc_ice_slot_indices(vars)
    kDim = patch.params.kDim
    pbar = ref_pressure(ref); rho_dbar = ref_rho_d(ref); rho_tbar = ref_rho_t(ref)
    rho_cbar = Springsteel.ref_rho_c(ref)
    E_tbar = ref_total_energy(ref); Q_ssbar = ref_qss(ref)

    i = 1
    for _ in 1:num_columns(patch)
        for k in 1:kDim
            x = gridpoints[i, 1]
            z = gridpoints[i, 2]
            L = sqrt(((x - xc) / xr)^2 + ((z - zc) / zr)^2)
            dtheta = L <= 1.0 ? dtheta_max * (cos(pi * L / 2.0))^2 : 0.0
            p_ref = pbar[k, 1]                              # Pa
            rho_dref = rho_dbar[k, 1]
            T_ref = p_ref / (Rd * rho_dref)                 # dry EOS
            exner = (p_0 * 100.0 / p_ref)^(Rd / Cpd)
            theta = (T_ref * exner) + dtheta
            Tk = theta / exner
            rho_d = p_ref / (Rd * Tk)                       # constant-pressure perturbation
            E_t = (rho_d * internal_energy_bf02(Tk, 0.0, 0.0)) + (rho_d * gravity * z)
            Q_ss = -rho_v_sat(Tk, p_ref / 100.0)
            patch.physical[i, p_i, 1] = 0.0
            patch.physical[i, rho_d_i, 1] = rho_d - rho_dref
            patch.physical[i, rho_t_i, 1] = rho_d - rho_tbar[k, 1]
            patch.physical[i, et_i, 1] = E_t - E_tbar[k, 1]
            patch.physical[i, qss_i, 1] = Q_ss - Q_ssbar[k, 1]
            patch.physical[i, rho_r_i, 1] = rain_slot(0.0, rain_transform, rain_mu)
            patch.physical[i, rho_c_i, 1] =
                condensate_slot(0.0, rho_cbar[k, 1], condensate_transform, condensate_mu)
            # Dry air: the state carries no vapor, so the slot is the negated reference.
            patch.physical[i, rho_v_i, 1] =
                vapor_slot(0.0, rho_tbar[k, 1], rho_dbar[k, 1], rho_cbar[k, 1])
            n_r_i > 0 && (patch.physical[i, n_r_i, 1] = 0.0)
            seed_ice_zero!(patch.physical, i, ice_i)
            i += 1
        end
    end
    return patch
end

"""
    temperature_bubble_mc!(patch, gridpoints, ref; xc, xr, zc, zr, dT_max)

Straka et al. (1993) cold-bubble initial condition for the total-energy set on a
`PressureReferenceState`. A constant-pressure temperature perturbation
`ΔT = dT_max (cos(πL)+1)/2` for `L ≤ 1` (identical to the `cos²(πL/2)` shape used by
[`theta_bubble_mc!`](@ref)) is applied to the dry reference profile: p' = 0,
rho_d' = rho_t' from the dry EOS at the perturbed temperature, E_t' from the BF02 internal
energy at rest, and Q_ss' tracking `-ρ_v*(T, p̄)` in the dry air.
"""
function temperature_bubble_mc!(patch::AbstractGrid, gridpoints::Matrix{Float64},
                                ref::Springsteel.PressureReferenceState;
                                xc=0.0, xr=4000.0, zc=3000.0, zr=2000.0, dT_max=-15.0,
                                condensate_transform::Symbol=:none, condensate_mu=1.0e-7,
                          rain_transform::Symbol=:none, rain_mu=1.0e-7)
    vars = patch.params.vars
    p_i = vars["p"]; rho_d_i = vars["rho_d"]; rho_t_i = vars["rho_t"]
    et_i = vars["E_t"]; qss_i = vars["Q_ss"]; rho_r_i = mc_slot(vars, "rho_r")
    rho_c_i = mc_slot(vars, "rho_c")
    # The prognostic VAPOR slot, present in every configuration of this set (see `MC_VARS`).
    rho_v_i = mc_slot(vars, "rho_v")
    # APPENDED optional slots: seeded only where they exist. `rain_number_slot(0.0, ...)` is
    # exactly 0.0 under every transform (`bhyp(0) == 0`), so no transform keyword has to be
    # threaded here — the same reason `rain_slot` needed none.
    n_r_i = mc_optional_slot(vars, "n_r")
    ice_i = mc_ice_slot_indices(vars)
    kDim = patch.params.kDim
    pbar = ref_pressure(ref); rho_dbar = ref_rho_d(ref); rho_tbar = ref_rho_t(ref)
    rho_cbar = Springsteel.ref_rho_c(ref)
    E_tbar = ref_total_energy(ref); Q_ssbar = ref_qss(ref)

    i = 1
    for _ in 1:num_columns(patch)
        for k in 1:kDim
            x = gridpoints[i, 1]
            z = gridpoints[i, 2]
            L = sqrt(((x - xc) / xr)^2 + ((z - zc) / zr)^2)
            dT = L <= 1.0 ? dT_max * (cos(pi * L / 2.0))^2 : 0.0
            p_ref = pbar[k, 1]                              # Pa
            rho_dref = rho_dbar[k, 1]
            Tk = (p_ref / (Rd * rho_dref)) + dT             # dry EOS reference T, perturbed
            rho_d = p_ref / (Rd * Tk)                       # constant-pressure perturbation
            E_t = (rho_d * internal_energy_bf02(Tk, 0.0, 0.0)) + (rho_d * gravity * z)
            Q_ss = -rho_v_sat(Tk, p_ref / 100.0)
            patch.physical[i, p_i, 1] = 0.0
            patch.physical[i, rho_d_i, 1] = rho_d - rho_dref
            patch.physical[i, rho_t_i, 1] = rho_d - rho_tbar[k, 1]
            patch.physical[i, et_i, 1] = E_t - E_tbar[k, 1]
            patch.physical[i, qss_i, 1] = Q_ss - Q_ssbar[k, 1]
            patch.physical[i, rho_r_i, 1] = rain_slot(0.0, rain_transform, rain_mu)
            patch.physical[i, rho_c_i, 1] =
                condensate_slot(0.0, rho_cbar[k, 1], condensate_transform, condensate_mu)
            # Dry air: the state carries no vapor, so the slot is the negated reference.
            patch.physical[i, rho_v_i, 1] =
                vapor_slot(0.0, rho_tbar[k, 1], rho_dbar[k, 1], rho_cbar[k, 1])
            n_r_i > 0 && (patch.physical[i, n_r_i, 1] = 0.0)
            seed_ice_zero!(patch.physical, i, ice_i)
            i += 1
        end
    end
    return patch
end

"""
    moist_temperature_bubble_mc!(patch, gridpoints, ref; xc, xr, zc, zr, dT_max)

Ooyama (2001)-style warm-rain trigger for the total-energy set on a moist (vapor-bearing)
`PressureReferenceState`: a constant-pressure temperature perturbation
`ΔT = dT_max cos²(πL/2)` for `L ≤ 1` with the vapor adjusted to PRESERVE the local
relative humidity at the perturbed temperature (`ρ_v = H·ρ_v*(T)` with
`H = ρ̄_v/ρ_v*(T̄)`), so the bubble carries extra moisture rather than drying out as it
warms. The dry density follows from the moist EOS at constant pressure; E_t and Q_ss
are computed pointwise before subtracting the reference; rho_r = 0.
"""
function moist_temperature_bubble_mc!(patch::AbstractGrid, gridpoints::Matrix{Float64},
                                      ref::Springsteel.PressureReferenceState;
                                      xc=75.0e3, xr=16.0e3, zc=500.0, zr=3000.0,
                                      dT_max=3.0, zcol=2,
                                      condensate_transform::Symbol=:none,
                                      condensate_mu=1.0e-7,
                                      rain_transform::Symbol=:none, rain_mu=1.0e-7)
    vars = patch.params.vars
    p_i = vars["p"]; rho_d_i = vars["rho_d"]; rho_t_i = vars["rho_t"]
    et_i = vars["E_t"]; qss_i = vars["Q_ss"]; rho_r_i = mc_slot(vars, "rho_r")
    rho_c_i = mc_slot(vars, "rho_c")
    # The prognostic VAPOR slot, present in every configuration of this set (see `MC_VARS`).
    rho_v_i = mc_slot(vars, "rho_v")
    # APPENDED optional slots: seeded only where they exist. `rain_number_slot(0.0, ...)` is
    # exactly 0.0 under every transform (`bhyp(0) == 0`), so no transform keyword has to be
    # threaded here — the same reason `rain_slot` needed none.
    n_r_i = mc_optional_slot(vars, "n_r")
    ice_i = mc_ice_slot_indices(vars)
    kDim = patch.params.kDim
    pbar = ref_pressure(ref); rho_dbar = ref_rho_d(ref); rho_tbar = ref_rho_t(ref)
    rho_cbar = Springsteel.ref_rho_c(ref)
    rho_vbar = Springsteel.ref_rho_v(ref)
    E_tbar = ref_total_energy(ref); Q_ssbar = ref_qss(ref)

    i = 1
    for _ in 1:num_columns(patch)
        for k in 1:kDim
            # On a 3D cylindrical grid (zcol = 3) this is a WN0 torus bubble:
            # column 1 is the radius and λ (column 2) does not enter.
            x = gridpoints[i, 1]
            z = gridpoints[i, zcol]
            L = sqrt(((x - xc) / xr)^2 + ((z - zc) / zr)^2)
            dT = L <= 1.0 ? dT_max * (cos(pi * L / 2.0))^2 : 0.0
            p_ref = pbar[k, 1]                              # Pa
            rho_dref = rho_dbar[k, 1]
            rho_vref = rho_vbar[k, 1]
            # Moist EOS reference temperature (matches the reference's own Tbar)
            T_ref = p_ref / ((rho_dref * Rd) + (rho_vref * Rv))
            Tk = T_ref + dT
            rho_v = rho_vref
            if dT > 0.0
                H = rho_vref / rho_v_sat(T_ref, p_ref / 100.0)
                rho_v = H * rho_v_sat(Tk, p_ref / 100.0)
            end
            rho_d = (p_ref - (rho_v * Rv * Tk)) / (Rd * Tk) # constant-pressure moist EOS
            rho_t = rho_d + rho_v
            q_v = rho_v / rho_d
            E_t = (rho_d * internal_energy_bf02(Tk, q_v, 0.0)) + (rho_t * gravity * z)
            Q_ss = rho_v - rho_v_sat(Tk, p_ref / 100.0)
            patch.physical[i, p_i, 1] = 0.0
            patch.physical[i, rho_d_i, 1] = rho_d - rho_dref
            patch.physical[i, rho_t_i, 1] = rho_t - rho_tbar[k, 1]
            patch.physical[i, et_i, 1] = E_t - E_tbar[k, 1]
            patch.physical[i, qss_i, 1] = Q_ss - Q_ssbar[k, 1]
            patch.physical[i, rho_r_i, 1] = rain_slot(0.0, rain_transform, rain_mu)
            patch.physical[i, rho_c_i, 1] =
                condensate_slot(0.0, rho_cbar[k, 1], condensate_transform, condensate_mu)
            # The moistened bubble's own vapor, against the DERIVED reference (the
            # cloud-free base makes rho_cbar exactly 0.0 here, but the term is written out
            # so this site cannot drift from the one the reconciliation differences).
            patch.physical[i, rho_v_i, 1] =
                vapor_slot(rho_v, rho_tbar[k, 1], rho_dbar[k, 1], rho_cbar[k, 1])
            n_r_i > 0 && (patch.physical[i, n_r_i, 1] = 0.0)
            seed_ice_zero!(patch.physical, i, ice_i)
            i += 1
        end
    end
    return patch
end

"""
    moist_buoyancy_bubble_mc!(patch, gridpoints, base, ref; q_t=0.02,
                              xc, xr, zc, zr, amp=2.0/300.0)

Total-energy variant of [`moist_buoyancy_bubble_pd!`](@ref): the identical Bryan &
Fritsch (2002) warm-bubble construction (θ_ρ inflation at constant pressure, reset to
exact saturation, re-converged with `saturation_adjustment`), written to the
moist_compressible slots. The final pressure comes from the EOS of the converged
(s, ρ_d, q_v) state so the initial temperature retrieval is exact; E_t and Q_ss are
computed pointwise from (T, p, ρ_d, ρ_v, ρ_c) before subtracting the reference.
"""
function moist_buoyancy_bubble_mc!(patch::AbstractGrid, gridpoints::Matrix{Float64},
                                   base, ref::Springsteel.PressureReferenceState; q_t=0.02,
                                   xc=10000.0, xr=2000.0, zc=2000.0, zr=2000.0,
                                   amp=2.0/300.0,
                                   condensate_transform::Symbol=:none, condensate_mu=1.0e-7,
                          rain_transform::Symbol=:none, rain_mu=1.0e-7)
    vars = patch.params.vars
    p_i = vars["p"]; rho_d_i = vars["rho_d"]; rho_t_i = vars["rho_t"]
    et_i = vars["E_t"]; qss_i = vars["Q_ss"]; rho_r_i = mc_slot(vars, "rho_r")
    rho_c_i = mc_slot(vars, "rho_c")
    # The prognostic VAPOR slot, present in every configuration of this set (see `MC_VARS`).
    rho_v_i = mc_slot(vars, "rho_v")
    # APPENDED optional slots: seeded only where they exist. `rain_number_slot(0.0, ...)` is
    # exactly 0.0 under every transform (`bhyp(0) == 0`), so no transform keyword has to be
    # threaded here — the same reason `rain_slot` needed none.
    n_r_i = mc_optional_slot(vars, "n_r")
    ice_i = mc_ice_slot_indices(vars)
    kDim = patch.params.kDim
    pbar = ref_pressure(ref); rho_dbar = ref_rho_d(ref); rho_tbar = ref_rho_t(ref)
    rho_cbar = Springsteel.ref_rho_c(ref)
    E_tbar = ref_total_energy(ref); Q_ssbar = ref_qss(ref)

    i = 1
    for _ in 1:num_columns(patch)
        for k in 1:kDim
            x = gridpoints[i, 1]
            z = gridpoints[i, 2]
            L = sqrt(((x - xc) / xr)^2 + ((z - zc) / zr)^2)
            b_incr = L <= 1.0 ? amp * (cos(pi * L / 2.0))^2 : 0.0

            new_s = base.s[k]
            new_rho_d = base.rho_d[k]
            new_q_v = base.q_v[k]
            new_q_l = base.q_l[k]
            if b_incr > 0.0
                p = base.p[k]
                new_theta = base.theta_rho[k] * (1.0 + b_incr) * (1.0 + q_t) /
                            (1.0 + (base.q_v[k] / Eps))
                new_T = new_theta / (p_0 / p)^(Rd / Cpd)
                new_q_v = q_sat_liquid(new_T, p)
                new_q_l = q_t - new_q_v
                new_T_rho = new_T * (1.0 + new_q_v / Eps) / (1.0 + q_t)
                new_rho_t = (p * 100.0) / (new_T_rho * Rd)
                new_rho_d = new_rho_t / (1.0 + q_t)
                new_xi = log_dry_density(new_rho_d)
                new_mu = mu_transform(new_q_v)
                new_mu_l = mu_transform(new_q_l)
                new_s = entropy(new_T, new_rho_d, new_q_v)
                dq, _ = saturation_adjustment(new_s, new_xi, new_mu, new_mu_l, eps())
                new_s += s_condensation_relaxation(-dq, new_T, new_rho_d, new_q_v, new_q_l, p)
                new_q_v += dq
                new_q_l = q_t - new_q_v
            end

            # Final consistent state: T from the entropy, p from the EOS, so that the
            # total-energy temperature retrieval reproduces T exactly at t = 0.
            Tk = temperature(new_s, new_rho_d, new_q_v)
            p_Pa = 100.0 * pressure(new_s, new_rho_d, new_q_v)
            rho_v = new_rho_d * new_q_v
            rho_c = new_rho_d * new_q_l
            rho_t = new_rho_d + rho_v + rho_c
            E_t = (new_rho_d * internal_energy_bf02(Tk, new_q_v, new_q_l)) +
                  (rho_t * gravity * z)
            Q_ss = rho_v - rho_v_sat(Tk, p_Pa / 100.0)

            patch.physical[i, p_i, 1] = p_Pa - pbar[k, 1]
            patch.physical[i, rho_d_i, 1] = new_rho_d - rho_dbar[k, 1]
            patch.physical[i, rho_t_i, 1] = rho_t - rho_tbar[k, 1]
            patch.physical[i, et_i, 1] = E_t - E_tbar[k, 1]
            patch.physical[i, qss_i, 1] = Q_ss - Q_ssbar[k, 1]
            patch.physical[i, rho_r_i, 1] = rain_slot(0.0, rain_transform, rain_mu)
            patch.physical[i, rho_c_i, 1] =
                condensate_slot(rho_c, rho_cbar[k, 1], condensate_transform, condensate_mu)
            # BF02's base IS cloudy, so rho_cbar is genuinely nonzero here and subtracting it
            # is what keeps the derived reference the one `res_rho_t` reconstructs.
            patch.physical[i, rho_v_i, 1] =
                vapor_slot(rho_v, rho_tbar[k, 1], rho_dbar[k, 1], rho_cbar[k, 1])
            n_r_i > 0 && (patch.physical[i, n_r_i, 1] = 0.0)
            seed_ice_zero!(patch.physical, i, ice_i)
            i += 1
        end
    end
    return patch
end
