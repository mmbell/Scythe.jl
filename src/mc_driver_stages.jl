# src/mc_driver_stages.jl -- the physical stages of `mc_driver!` (src/moist_compressible.jl),
# split out as `@noinline` functions on plain arrays (Stream B, direction A of
# reference/HANDOFF_DRIVER_COMPILE_CLIFF.md). Each stage takes the per-thread scratch
# NamedTuple `S` (one pointer), the tendency matrices and reference columns it reads, and its
# scalars as Ints/Float64s or an isbits NamedTuple of flags -- never the `ModelTile` (a helper
# that takes the tile costs 3-4x the compile of the same code on arrays) and never a
# slot-view NamedTuple. The bodies are the driver's statements moved VERBATIM, in order, so
# the answer is bitwise the inline driver's (test/test_driver_stages.jl is the gate).
#
# Included after moist_compressible.jl: the stages call helpers defined there and in
# ishmael.jl, and `mc_driver!` calls the stages; both resolve at the driver's first call.

"""
    _etd_precompute_stage!(S, expdot, expdot_nm1, expdot_nm2, Q_ssbar, J_v, stx,
                           colstart, t, ts, tid_x, f) -> nothing

Stage 16 of `mc_driver!`: the ETD-AB3 pre-compute of the stiff relaxation pair and the
step-mean phase-change rates it closes on, per gridpoint of the column `colstart`. Called in
the one window in which the `Q_ss` tendency in `expdot` is final for the step and
`expdot_nm1`/`expdot_nm2` still hold `N^{n-1}`/`N^{n-2}` (before `explicit_timestep` rotates
them).

Reads (written by the earlier stages): `S.etd_lam` (read-modify-write), `invtau_c/r`,
`invtau_i1..3`, `etd_dep_a/b`, `Q_ss`, `Q_s`, `Qdot`, `Qdot_r`, `f_cloud`, `Q_s_i`, the ice
Jacobians `J_i{1,2,3}{q,a,c,n}`, `f_isn1..3`, `rho_d`, `rho_r`, `i1q..i3q`, `ICE_C/ICE_R`, the
habit columns `hab*`, `Tk`, `Jr`, `Jc`, `Lv`, `C_pt`, `R_m`, `C_vt`, `AUTO_COLL`, `Fr_z`,
`nu_c`, `nu_r`, `SRC_i*q`, `F_i*q_z`, `nu_i*q`; the `Q_ss` (slot 7) column of `expdot` and its
two history levels; `Q_ssbar`; `J_v` (the vapor Jacobian column, today
`vapor_jacobian_column(S)`).

Writes: `S.etd_lam` (lambda including the ice conductances), `etd_qbar`, `etd_qnp1`,
`etd_di1..3`, `etd_da/dc/dn1..3` (ice only), `Qdot_bar`, `Qdot_r_bar`, `Qdep_bar`,
`VAPOR_SRC`, `etd_dv`, `etd_d8`, `etd_d9`, `etd_d1`, and `etd_dlc/dlr/dli1..3` (only under
`f.lobe_relax`). Read downstream by `relaxation_adjustment_qss!`, `_etd_exchange_closure!`,
`_water_exchange_stage!`/`_water_exchange_multistep!`, `_lobe_relax_paired!` and
`water_depletion_probe!`.

Census hooks inside (all on `stx = mtile.mc_water_stats`, thread column `tid_x`):
`_cloud_evap_census!`, `mc_donor_census!` (S1-S3, under `f.census_on`) and `mc_attr_census!`
(QR_EVAP/QR_COMB, under `f.attr_on`). No energy-budget hook sits inside this stage.

`f` is the isbits flag tuple `(; ice_on, vtrans_on, cloud_evap_real, census_on, attr_on,
lobe_relax, lr_c, lr_r, lr_i, a_lobe)` formed by the driver.
"""
@noinline function _etd_precompute_stage!(S, expdot::Matrix{Float64},
                                          expdot_nm1::Matrix{Float64},
                                          expdot_nm2::Matrix{Float64}, Q_ssbar,
                                          J_v::Vector{Float64}, stx::Matrix{Float64},
                                          colstart::Int, t::Int, ts::Float64, tid_x::Int,
                                          f::NamedTuple)
    (; ice_on, vtrans_on, cloud_evap_real, census_on, attr_on,
       lobe_relax, lr_c, lr_r, lr_i, a_lobe) = f
    # The columns the earlier stages wrote, under the names `mc_driver!` binds them to.
    etd_lam = S.etd_lam; invtau_c = S.invtau_c; invtau_r = S.invtau_r
    Q_ss = S.Q_ss; Q_s = S.Q_s; Qdot = S.Qdot; Qdot_r = S.Qdot_r; f_cloud = S.f_cloud
    rho_d = S.rho_d; rho_r = S.rho_r; Tk = S.Tk; Jr = S.Jr; Jc = S.Jc; Lv = S.Lv
    C_pt = S.C_pt; R_m = S.R_m; C_vt = S.C_vt; AUTO_COLL = S.AUTO_COLL; Fr_z = S.Fr_z
    nu_c = S.nu_c; nu_r = S.nu_r
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
    etd_dlc = S.etd_dlc; etd_dlr = S.etd_dlr
    etd_dli1 = S.etd_dli1; etd_dli2 = S.etd_dli2; etd_dli3 = S.etd_dli3
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
        # condensation is a source. (The CLOUD channel is not realized this way: its one donor
        # factor `f_cloud` — evaporation, `AUTO_COLL` and the ice legs on one `κ_tot` — is
        # applied to the step-mean just below, `cloud_evap_realized`.)
        qc_bar = 0.0
        qr_bar = 0.0
        if clipped_l
            # Already realized at the fold, where a clipped flux enters `N`.
            qc_bar = Qdot[i]
            qr_bar = Qdot_r[i]
        else
            qb = qbar * invtau_l / (1.0 + Q_s[i])
            qc_bar = invtau_c[i] == 0.0 ? 0.0 : qb * (invtau_c[i] / invtau_l)
            qr_bar = invtau_r[i] == 0.0 ? 0.0 : qb * (invtau_r[i] / invtau_l)
            # ── THE CLOUD DONOR'S REALIZATION of its evaporation (`cloud_evap_realized`) ──
            # The propagator bounds `Q_ss`, not `ρ_c`: the deficit can ask a thin cloud for
            # more than it holds, and `d9` alone cannot take the slot below `ν = 0` while
            # `dv`, `d1` and `Q_ss^{n+1}` would take the full draw. Realized HERE, at the cloud
            # donor's one factor `f_cloud = J₀(κ_tot Δt)` (the same number `AUTO_COLL` and the
            # ice legs carry), on the one number all of them are built from below, and
            # `Q_ss^{n+1}` gives back what was not realized: its liquid loss over the step is
            # `Δt(1 + 𝒬_s)(q̄_c + q̄_r)` (the psychrometric cancellation, `qb` above), so
            # `−Δt(1 + 𝒬_s)·Δq̄_c`. `λ ≠ 0` here whenever the cloud evaporates (an unclipped
            # channel's conductance IS `λ`), so `etd_qnp1` is the value
            # `relaxation_adjustment_qss!` writes. `J₀ < 1` at every evaporating point with
            # cloud; `===` the unrealized `qc_bar` only where it condenses or `f_cloud = 1.0`.
            qcb = cloud_evap_realized(qc_bar, f_cloud[i])
            if qcb != qc_bar
                _cloud_evap_census!(stx, tid_x, (qcb - qc_bar) * ts)
                if cloud_evap_real
                    etd_qnp1[i] -= ts * ((1.0 + Q_s[i]) * (qcb - qc_bar))
                    qc_bar = qcb
                end
            end
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
        # Under the vapor control variable the slot receives the ν-image `J_v·Δt·vsrc` (the
        # frozen-at-n Jacobian every transformed slot's increment carries).
        etd_dv[i] = vtrans_on ? J_v[i] * (ts * vsrc) : ts * vsrc
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

        # ── LOBE RELAXATION (Stage L): the closed-channel gate, read off the very terms the
        # slot is about to receive ──────────────────────────────────────────────────────────
        # HERE because this is the one window in which every term the gate needs is final for
        # the step and in hand together: the step-mean phase change just formed (`qc_bar`,
        # `qr_bar`, the deposition increments `etd_di*`), the conversion `AUTO_COLL`, the ice
        # back-reactions `ICE_C`/`ICE_R`, the ISHMAEL+anchor+population sources `SRC_*`, the
        # fitted sedimentation divergences `Fr_z`/`F_*_z` — and the state-`n` control variable,
        # which IS the post-refit value the step started from. A point is relaxed only where
        # `ν^n < 0` (recovered density 0 under `:bhyp`) and every one of those terms is an
        # exact `0.0`: the gate is the physics' own statement that nothing acts there, not a
        # threshold of ours. For the cloud, `qc_bar == 0` is exactly `qss_condensation_rates`'
        # `invtau_c = 0` branch (`q_c ≤ 1e-8` and `S ≤ 1e-4`): the CLIPPED branch above copies
        # `Qdot`, which is `invtau_c·drive` and an exact zero with it.
        #
        # The increments are applied by `relaxation_adjustment_qss!`, bounded there by the
        # post-advance value (`lobe_relaxation_cap`), so no recovered mass moves.
        if lobe_relax
            etd_dlc[i] = (lr_c && qc_bar == 0.0 && AUTO_COLL[i] == 0.0 &&
                          (!ice_on || S.ICE_C[i] == 0.0)) ?
                lobe_relaxation_increment(nu_c[i], a_lobe) : 0.0
            etd_dlr[i] = (lr_r && qr_bar == 0.0 && AUTO_COLL[i] == 0.0 && Fr_z[i] == 0.0 &&
                          (!ice_on || S.ICE_R[i] == 0.0)) ?
                lobe_relaxation_increment(nu_r[i], a_lobe) : 0.0
            if lr_i
                etd_dli1[i] = (S.SRC_i1q[i] == 0.0 && etd_di1[i] == 0.0 &&
                               S.F_i1q_z[i] == 0.0) ?
                    lobe_relaxation_increment(S.nu_i1q[i], a_lobe) : 0.0
                etd_dli2[i] = (S.SRC_i2q[i] == 0.0 && etd_di2[i] == 0.0 &&
                               S.F_i2q_z[i] == 0.0) ?
                    lobe_relaxation_increment(S.nu_i2q[i], a_lobe) : 0.0
                etd_dli3[i] = (S.SRC_i3q[i] == 0.0 && etd_di3[i] == 0.0 &&
                               S.F_i3q_z[i] == 0.0) ?
                    lobe_relaxation_increment(S.nu_i3q[i], a_lobe) : 0.0
            else
                etd_dli1[i] = 0.0; etd_dli2[i] = 0.0; etd_dli3[i] = 0.0
            end
        end
    end
    return nothing
end
