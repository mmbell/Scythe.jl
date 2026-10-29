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
    _thermo_stage!(S, geom, phys, gridpoints, u, w, pp, pbar, z, colstart, colend, t, ts,
                   output_dir, f) -> nothing

Stage 3 of `mc_driver!`: the diagnostic thermodynamic state of the column — `ke`, `geo`, `M`,
the liquid and ice totals and their thermodynamic (floored) readings, the model temperature
`Tk`, the located state guard, the saturation density and its partials, `q_v/q_l/q_i`, the
mixture coefficients `C_vt`, `R_m`, `C_pt`, `gamma_m` (and `S.sd_pxi` under `f.sd_si`), `Lv`.

`u`, `w`, `pp` are the slot views the driver binds (`view(phys, colstart:colend, slot, 1)`);
the tangential wind `vv` is rebuilt here from `phys` with `mc_v_views` (it is a slot-view
NamedTuple, never passed). `pbar` is the reference-pressure view, `z` the height view,
`gridpoints`/`output_dir` only feed the guard's location message.
`f = (; ice_on, cond_floor, teos, sd_si, column_state_guard)`.

Writes (S): `ke`, `geo`, `M`, `rho_liq`, `rho_ice`, `rho_liq_t`, `rho_ice_t`, `Tk`, `p_hPa`,
`rho_vs`, `q_v`, `q_l`, `q_i`, `C_vt`, `R_m`, `C_pt`, `gamma_m`, `sd_pxi`, `Lv`, `drvs_dT`,
`drvs_dp`. Readers: STAGE_MAP.md (driver_split scratch).
"""
@noinline function _thermo_stage!(S, geom::MCGeometry, phys::Array{Float64,3},
                                  gridpoints::Matrix{Float64}, u, w, pp, pbar, z,
                                  colstart::Int, colend::Int, t::Int, ts::Float64,
                                  output_dir::String, f::NamedTuple)
    (; ice_on, cond_floor, teos, sd_si, column_state_guard) = f
    # The tangential wind, rebuilt from the plain array (`nothing` on the Cartesian slice).
    vv = mc_v_views(geom, (physical = phys,), colstart, colend)
    # The columns the earlier stages wrote, under the names `mc_driver!` binds them to.
    p = S.p; rho_d = S.rho_d; rho_v = S.rho_v; rho_t = S.rho_t; E_t = S.E_t; Q_ss = S.Q_ss
    rho_c = S.rho_c; rho_r = S.rho_r; i1q = S.i1q; i2q = S.i2q; i3q = S.i3q
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
    # Q_s_energy, the entropy and E_sed. `rho_liq` itself stays raw: the census and the
    # sedimentation read the water partition as carried, and flooring a partition
    # manufactures water.
    rho_liq_t = S.rho_liq_t
    if cond_floor
        @. rho_liq_t = max(rho_c, 0.0) + max(rho_r, 0.0)
    else
        copyto!(rho_liq_t, rho_liq)
    end
    # The ice mirror of `rho_liq_t`, floored per species under `:diagnostic` for the same
    # reason: a spline undershoot in one ice mass must not reach the retrieval as negative
    # latent heat. `rho_ice` itself stays raw: the census must see the water partition the
    # continuity equations actually carry. (The reader-side cap at the ice-anchor headroom
    # that followed was deleted with the ice-anchor legs at Stage 5 of the ρ_t retirement.)
    rho_ice_t = S.rho_ice_t
    if cond_floor && ice_on
        @. rho_ice_t = (max(i1q, 0.0) + max(i2q, 0.0)) + max(i3q, 0.0)
    else
        copyto!(rho_ice_t, rho_ice)
    end
    # THE MODEL TEMPERATURE, and everything downstream in this step reads it: the guard below,
    # the saturation density and its partials, Q_s, the condensation closure and the ETD
    # pre-compute, the ice sources, the surface layer, the Louis/MYNN boundary layer, E_sed,
    # the entropy. Under `:eos` (default) it is the equation-of-state temperature of the
    # prognostic pressure and the two gas densities, with `rho_v` the PROGNOSTIC vapor staged
    # above (`rho_vp + rho_vbar`, unfloored — negative vapor is a resolution diagnostic, and the
    # EOS denominator is harmless to it; see `eos_temperature`). It reads NO condensate, so
    # the `condensate_floor` does not enter T any more; it still shapes `q_l`/`q_i` (hence
    # C_vt, R_m, γ_m, Q_s) below, as before. Under `:energy` it is the closed-form
    # total-energy retrieval on the floored condensate — the pre-switch line, bitwise. Why the switch: `eos_temperature` (the clipped-lobe phantom latent heat).
    Tk = S.Tk
    if teos
        @. Tk = eos_temperature(p, rho_d, rho_v)
    else
        @. Tk = retrieve_temperature(M, rho_d, rho_t, rho_liq_t, rho_ice_t)
    end
    # LOCATED STATE GUARD (see `check_column_state`): p, ρ_d, ρ_t and T must be positive and
    # finite before anything consumes them — the saturation lookups just below, the sd_si
    # coefficient γ_m p/ρ_t, the microphysics. The §2c death class (retrieval heating → w /
    # sound-speed runaway → vertical SI past its limit → p < 0) used to surface first as a bare
    # `DomainError` in the ice kernel, and the liquid path let it straight through. One
    # comparison pass per column; reads only, and the message is built only on failure.
    # `options[:column_state_guard] = false` disables it — forensics only (a test that
    # deliberately carries a diverged state bitwise through two paths).
    column_state_guard &&
    check_column_state(p, rho_d, rho_t, Tk, z,
        vv === nothing ?
            (; p, p_prime = pp, pbar, rho_d, rho_t, Tk, w, u, rho_v, rho_liq, rho_ice,
               rho_ice_t, rho_liq_t, Q_ss, E_t, M) :
            (; p, p_prime = pp, pbar, rho_d, rho_t, Tk, w, u, v = vv.f, rho_v, rho_liq,
               rho_ice, rho_ice_t, rho_liq_t, Q_ss, E_t, M),
        (; step = t, time_s = t * ts, tile = output_dir, tile_ts = ts,
           colstart, colend, column = Base.div(colstart - 1, colend - colstart + 1) + 1,
           coord1 = gridpoints[colstart, 1],
           coord2 = zcoord(geom) == 3 ? gridpoints[colstart, 2] : NaN))
    p_hPa = S.p_hPa;   @. p_hPa = p / 100.0
    rho_vs = S.rho_vs; @. rho_vs = rho_v_sat(Tk, p_hPa)
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
    return nothing
end

"""
    _warm_micro_stage!(S, stx, micro_nm1, micro_nm2, fcols, scols, z, colstart, t, ts,
                       N_r, N_0, max_N_c, nr_i, f) -> nothing

Stage 4 of `mc_driver!`: the psychrometric factor `Q_s`, the (diagnostic) AB3 sink bounds
`cap_*`, the two-channel condensation closure with the rain donor's evaporation conductance
and warm factor, the liquid stiffness census, the warm-rain conversion (single- or
two-moment) and its sedimentation-flux fits, and the cloud donor's warm conductance and
factor.

`stx` is `mtile.mc_water_stats`, `micro_nm1`/`micro_nm2` the micro history matrices (read
only under `f.budget_trace`), `fcols`/`scols` the per-thread flux/scratch spline columns
(`sedimentation_flux_column(fcols, scols, slot, f.sed_lid_dir)`), `nr_i` the rain-number
slot. `f = (; budget_trace, condensation, rain_evap_real, precipitation, rain_2m,
sed_lid_dir)`.

Writes (S): `Q_s`, `cap_c/r/v` (budget trace only), `Qdot`, `Qdot_r`, `invtau_c`, `invtau_r`,
`kappa_ev`, `f_rain`, `AUTO_COLL`, `NR_SRC`, `Vt`, `Vtn`, `Fr`, `E_sed`, `Fnr`, `Fr_z`, `E_sed_z`,
`Fnr_z`, `kappa_c`, `f_cloud`; the stiffness census rows of `stx`.
"""
@noinline function _warm_micro_stage!(S, stx::Matrix{Float64}, micro_nm1::Matrix{Float64},
                                      micro_nm2::Matrix{Float64}, fcols::AbstractMatrix,
                                      scols::AbstractMatrix, z, colstart::Int, t::Int,
                                      ts::Float64, N_r::Float64, N_0::Float64,
                                      max_N_c::Float64, nr_i::Int, f::NamedTuple)
    (; budget_trace, condensation, rain_evap_real, precipitation, rain_2m, sed_lid_dir) = f
    # The columns the earlier stages wrote, under the names `mc_driver!` binds them to.
    Tk = S.Tk; p = S.p; p_hPa = S.p_hPa; rho_d = S.rho_d; rho_v = S.rho_v; rho_vs = S.rho_vs
    rho_c = S.rho_c; rho_r = S.rho_r; n_r = S.n_r; Q_ss = S.Q_ss; rho_t = S.rho_t
    rho_liq = S.rho_liq; M = S.M; ke = S.ke; E_t = S.E_t; Lv = S.Lv
    q_v = S.q_v; q_l = S.q_l; q_i = S.q_i
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
        @inbounds for i in eachindex(cap_c)
            g = colstart + i - 1
            cap_c[i] = _ab3_sink_bound(ts, t, max(rho_c[i], 0.0),
                                       micro_nm1[g, MC_MICRO_C], micro_nm2[g, MC_MICRO_C])
            cap_r[i] = _ab3_sink_bound(ts, t, max(rho_r[i], 0.0),
                                       micro_nm1[g, MC_MICRO_R], micro_nm2[g, MC_MICRO_R])
            # The vapor is a SINK channel too -- condensation removes it -- so its bound has
            # the same form and is negated into a ceiling on `Qdot_c + Qdot_r` at the use site.
            # The PROGNOSTIC `rho_v`: the measurement is of the vapor the closure actually
            # reads, a slot rather than a retrieval.
            cap_v[i] = _ab3_sink_bound(ts, t, max(rho_v[i], 0.0),
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
    if condensation
        # The closure reads the PROGNOSTIC `rho_v` (the call site is unchanged; what it is
        # handed is now a transported slot rather than a difference of five fitted fields,
        # which is the whole point of Stage A).
        for i in 1:length(Qdot)
            Qdot[i], Qdot_r[i], invtau_c[i], invtau_r[i] =
                qss_condensation_rates(Q_ss[i], rho_v[i], rho_c[i], rho_r[i],
                                       rho_d[i], Tk[i], p_hPa[i], Q_s[i],
                                       ts, N_r, max_N_c; N_0=N_0,
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
            f_rain[i] = relaxation_realization(kev * qr_i, qr_i, ts)
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
    mc_stiffness_census!(stx, ts, invtau_c, invtau_r)

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
        r_col = sedimentation_flux_column(fcols, scols, 8, sed_lid_dir)
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
        nr_col = sedimentation_flux_column(fcols, scols, nr_i, sed_lid_dir)
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
        r_col = sedimentation_flux_column(fcols, scols, 8, sed_lid_dir)
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

    # ── The CLOUD DONOR's warm conductance and its factor (`cloud_evap_realized`) ──────────
    # HERE, the first point at which both warm sinks of the cloud exist: the evaporation
    # `Q̇_c` (frozen at state `n` by the closure above, the rain's `kappa_ev` convention) and
    # the conversion `AUTO_COLL`. `κ_c` is their sum over the reservoir, one conductance, and
    # `J₀(κ_c Δt)` is the cloud's factor wherever no ice leg draws on it; `mc_ice_sources!`
    # completes the total with `ṁ_hf,c` and the riming and overwrites the factor at every
    # point it visits. Formed in the ISHMAEL mixing-ratio units (`q_c = max(ρ_c, 0)/ρ_d`) the
    # ice loop forms it in, so the two expressions are bitwise the same number where the ice
    # legs are zero. Formed in BOTH modes — the census measures what the off arm would have
    # withheld — and applied only below (the fold), gated on `cloud_evap_real`. An exact
    # `κ_c = 0`, `f = 1.0` with no cloud.
    kappa_c = S.kappa_c
    f_cloud = S.f_cloud
    @inbounds for i in eachindex(kappa_c)
        kc = cloud_donor_conductance(Qdot[i], AUTO_COLL[i], rho_c[i])
        kappa_c[i] = kc
        qc_i = max(rho_c[i], 0.0) / rho_d[i]
        f_cloud[i] = relaxation_realization(kc * qc_i, qc_i, ts)
    end
    return nothing
end

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
two history levels; `Q_ssbar`; `J_v` (the vapor Jacobian column,
`jacobian_column(mtile.species, rv_i, tid)`).

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

"""
    _ice_stage!(S, tab, stx, fcols, scols, IS, z, ts, max_N_c, tau_pop, pop_seed,
                tau_hf, tau_act, tau_vc, f) -> nothing

Stage 5 of `mc_driver!`: the ISHMAEL ice process sources, the stiffness census of the three
deposition relaxations, the population reconciliation, the twelve moment sedimentation-flux
fits and the two mass-flux aggregates (`Fi_z`, `E_sed_i_z`). A no-op unless `f.ice_on`.

`tab` is `mtile.ishmael_tables`, `stx` the water census matrix `mtile.mc_water_stats`,
`fcols`/`scols` the per-thread flux (zero-flux lid) and scratch spline columns
`mtile.flux_columns`/`mtile.scratch_columns` (the fit column is chosen by
`sedimentation_flux_column(fcols, scols, slot, f.sed_lid_dir)`), `IS::MCSlots` the slot
indices, `z` the column's height view. `f = (; ice_on, condensation, ice_var_check, ice_attr,
ice_agg_caps, rain_evap_real, cloud_evap_real, ice_shed, ice_n_real, rain_pop_gate,
ice_pop_src, sed_lid_dir)`.

Writes (S): `Q_s_i`; through `mc_ice_sources!` the `SRC_i*`, `invtau_i1..3`, `Qdot_i1..3`,
`etd_dep_a/b`, `f_ice1..3`, `f_isn1..3`, the donor totals `f_rain`/`f_cloud`, `ICE_C`,
`ICE_R`, `ICE_NR`, `FRZ_NET`, the habit columns `hab*`, the fall speeds `Vi*m/n`; through
`_ice_population_reconcile!` the `SRC_*` it reconciles; `F_i{1,2,3}{q,n,a,c}` and their
`_z` divergences; `Fi_z`, `E_sed_i`, `E_sed_i_z`. Census rows in `stx` (stiffness, donor,
attribution, population). Downstream readers: STAGE_MAP.md (driver_split scratch).
"""
@noinline function _ice_stage!(S, tab::IshmaelTables, stx::Matrix{Float64},
                               fcols::AbstractMatrix, scols::AbstractMatrix, IS::MCSlots, z,
                               ts::Float64, max_N_c::Float64, tau_pop::Float64,
                               pop_seed::Symbol, tau_hf::Float64, tau_act::Float64,
                               tau_vc::Float64, f::NamedTuple)
    (; ice_on, condensation, ice_var_check, ice_attr, ice_agg_caps, rain_evap_real,
       cloud_evap_real, ice_shed, ice_n_real, rain_pop_gate, ice_pop_src, sed_lid_dir) = f
    # The columns the earlier stages wrote, under the names `mc_driver!` binds them to.
    Tk = S.Tk; p = S.p; rho_d = S.rho_d; q_v = S.q_v; q_l = S.q_l; q_i = S.q_i; ke = S.ke
    i1q = S.i1q; i1n = S.i1n; i1a = S.i1a; i1c = S.i1c
    i2q = S.i2q; i2n = S.i2n; i2a = S.i2a; i2c = S.i2c
    i3q = S.i3q; i3n = S.i3n; i3a = S.i3a; i3c = S.i3c
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
        # Every process rate: deposition through the shared prognostic supersaturation, the
        # full ISHMAEL nucleation/riming/aggregation/melting set, and the twelve slot sources
        # they assemble into. Reads and writes the scratch only; see `mc_ice_sources!`.
        # `options[:condensation] = false` switches the ice phase changes off with the liquid
        # ones, for the same reason: a run advertised as "no physics" must not glaciate.
        mc_ice_sources!(S, tab, ts, max_N_c,
                        condensation,
                        ice_var_check,
                        ice_attr, ice_agg_caps, rain_evap_real,
                        cloud_evap_real, ice_shed,
                        ice_n_real, rain_pop_gate,
                        tau_hf,
                        tau_act,
                        tau_vc,
                        stx, Threads.threadid())
        # The three deposition relaxations, on the rates the step actually used. Ungated, for
        # the reason the liquid census is: the once-per-run under-resolution warning has to be
        # able to see an excursion whether or not a diagnostic is switched on.
        mc_stiffness_census!(stx, ts, S.invtau_i1, S.invtau_i2, S.invtau_i3)
        # The POPULATION RECONCILIATION, after the process sources and before anything reads
        # the SRC accumulators: the process sources are assembled and nothing has read them
        # yet. (The ice-anchor reconciliation that ran here before it, with its reader cap,
        # rate share and sedimentation share, was deleted at Stage 5 of the ρ_t retirement:
        # its headroom test read through ρ_w had become a vapor-negativity device.)
        # It is the exact complement of the population gate above — mass whose
        # carried number is not positive, which no rate may act on and which no device could
        # therefore remove — returned to a representation the equations can use: rain above
        # `T_0` with `L_f` absorbed through `FRZ_NET`, 2 μm crystals below it, on `τ_pop`.
        # Census `MC_POP_*`; TeX §"Reconciliation of the population".
        # Switched off with the liquid phase changes by `options[:condensation] = false`,
        # for the same reason `mc_ice_sources!` is: BOTH branches move water between
        # categories (the warm one moves mass and `L_f`, the cold one creates crystals), and
        # a run advertised as "no physics" — the transport-only fixtures included — must not.
        # The CENSUS is unaffected either way.
        _ice_population_reconcile!(S, stx, Threads.threadid(),
                                   Tk, rho_d, tau_pop,
                                   ice_pop_src && condensation,
                                   ts, pop_seed)

        # Each moment's flux at ITS OWN weighted fall speed — mass-weighted for the mass and
        # the two volume moments, number-weighted for the number. That is the whole reason
        # the four moments of a species cannot share one flux: the distribution SORTS as it
        # falls, and a number that sedimented at the mass-weighted speed would carry the small
        # crystals down as fast as the large ones. `ishmael_fall_speeds` has already applied
        # the 25 m/s cap and the melting size-sorting override.
        _ice_flux!(fcols, scols, S.F_i1q, S.F_i1q_z, i1q, S.Vi1m, IS.i1_q, sed_lid_dir)
        _ice_flux!(fcols, scols, S.F_i1n, S.F_i1n_z, i1n, S.Vi1n, IS.i1_n, sed_lid_dir)
        _ice_flux!(fcols, scols, S.F_i1a, S.F_i1a_z, i1a, S.Vi1m, IS.i1_a, sed_lid_dir)
        _ice_flux!(fcols, scols, S.F_i1c, S.F_i1c_z, i1c, S.Vi1m, IS.i1_c, sed_lid_dir)
        _ice_flux!(fcols, scols, S.F_i2q, S.F_i2q_z, i2q, S.Vi2m, IS.i2_q, sed_lid_dir)
        _ice_flux!(fcols, scols, S.F_i2n, S.F_i2n_z, i2n, S.Vi2n, IS.i2_n, sed_lid_dir)
        _ice_flux!(fcols, scols, S.F_i2a, S.F_i2a_z, i2a, S.Vi2m, IS.i2_a, sed_lid_dir)
        _ice_flux!(fcols, scols, S.F_i2c, S.F_i2c_z, i2c, S.Vi2m, IS.i2_c, sed_lid_dir)
        _ice_flux!(fcols, scols, S.F_i3q, S.F_i3q_z, i3q, S.Vi3m, IS.i3_q, sed_lid_dir)
        _ice_flux!(fcols, scols, S.F_i3n, S.F_i3n_z, i3n, S.Vi3n, IS.i3_n, sed_lid_dir)
        _ice_flux!(fcols, scols, S.F_i3a, S.F_i3a_z, i3a, S.Vi3m, IS.i3_a, sed_lid_dir)
        _ice_flux!(fcols, scols, S.F_i3c, S.F_i3c_z, i3c, S.Vi3m, IS.i3_c, sed_lid_dir)

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
        ei_col = sedimentation_flux_column(fcols, scols, IS.i1_q, sed_lid_dir)
        ei_col.uMish .= E_sed_i
        Btransform!(ei_col)
        Atransform!(ei_col)
        Ixtransform(ei_col, E_sed_i_z)
    end
    return nothing
end

"""
    _liquid_fold_stage!(S, stx, micro_n, colstart, ts, max_N_c, f) -> nothing

Stage 6 of `mc_driver!`: the realized rain conductance folded into `invtau_r`/`Qdot_r`, the
realized cloud sinks (`AUTO_COLL` and its rain-number partner, the clipped-drive cloud
evaporation), the liquid λ/N split (`etd_lam`, `etd_nl`), and this step's microphysics
history row (`micro_n = mtile.mc_micro_n`, channels `MC_MICRO_C/R/V` at rows
`colstart:colend`). `stx` is `mtile.mc_water_stats` (the `MC_CEVAP_*` census).
`f = (; cloud_evap_real, precipitation, rain_2m, condensation, ice_on)`.

Writes (S): `AUTO_COLL`, `NR_SRC`, `invtau_r`, `Qdot_r`, `invtau_c`, `Qdot`, `etd_lam`,
`etd_nl`. Reads `f_cloud`, `f_rain` (final after `_ice_stage!`), `rho_c`, `rho_d`, `rho_v`,
`rho_vs`, `Q_ss`, `ICE_C`, `ICE_R`.
"""
@noinline function _liquid_fold_stage!(S, stx::Matrix{Float64}, micro_n::Matrix{Float64},
                                       colstart::Int, ts::Float64, max_N_c::Float64,
                                       f::NamedTuple)
    (; cloud_evap_real, precipitation, rain_2m, condensation, ice_on) = f
    # The columns the earlier stages wrote, under the names `mc_driver!` binds them to.
    AUTO_COLL = S.AUTO_COLL; f_cloud = S.f_cloud; f_rain = S.f_rain; NR_SRC = S.NR_SRC
    rho_c = S.rho_c; rho_d = S.rho_d; rho_v = S.rho_v; rho_vs = S.rho_vs; Q_ss = S.Q_ss
    invtau_c = S.invtau_c; invtau_r = S.invtau_r; Qdot = S.Qdot; Qdot_r = S.Qdot_r
    etd_lam = S.etd_lam; etd_nl = S.etd_nl
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
    #
    # ── The REALIZED CLOUD SINKS (`cloud_evap_realized`) ────────────────────────────────
    # `f_cloud` is final here for the same reason `f_rain` is: `mc_ice_sources!` has
    # completed the cloud donor's total. ONE factor, every sink on the reservoir:
    #
    #   * `AUTO_COLL` — the conversion to rain, which is the SAME array the rain slot reads
    #     as its source, so the liquid-to-liquid pair stays equal and opposite; under two
    #     moments its rain-number partner `n_auto` takes the same factor (the same collisions;
    #     accretion adds no drops), recomputed from the state-`n` arguments `AUTO_COLL` was
    #     formed from rather than staged in a column. Nothing between the closure and this
    #     point reads `AUTO_COLL` except `kappa_c` itself, which wants the unrealized rate.
    #   * the EVAPORATION, where the drive is CLIPPED (in the loop below): a clipped channel's
    #     step-mean IS its frozen rate and its flux lives in `N`, so the factor has to reach
    #     `Q̇_c` and `1/τ_c` at state `n`, before `N` is formed from them — scaling both by the
    #     one factor keeps `Q̇_c(1 + 𝒬_s) = drive·τ_c^{-1}`, which is what `etd_nl` is written
    #     in. Where the drive is unclipped the factor acts on the step-mean `qc_bar` in the ETD
    #     pre-compute instead. Only where the cloud EVAPORATES at state `n` — the `evap_now`
    #     mirror: a donor factor bounds a draw, and condensation is a source.
    #   * the ice legs already carry it, as `f_qc`.
    #
    # Gated on `cloud_evap_real`; the census of the evaporation is written in both modes.
    if cloud_evap_real && precipitation
        @inbounds for i in eachindex(AUTO_COLL)
            fc = f_cloud[i]
            ac = AUTO_COLL[i]
            (fc < 1.0 && ac > 0.0) || continue
            AUTO_COLL[i] = fc * ac
            if rain_2m
                n_auto = rain_autoconversion_2m(rho_c[i], rho_d[i], max_N_c)[2]
                NR_SRC[i] += (fc - 1.0) * n_auto
            end
        end
    end
    cev_st = stx
    cev_tid = Threads.threadid()
    if condensation
        @inbounds for i in eachindex(invtau_r)
            evap_now = Qdot_r[i] < 0.0
            fr = f_rain[i]
            invtau_r[i] = evap_now ? fr * invtau_r[i] : invtau_r[i]
            Qdot_r[i] = evap_now ? fr * Qdot_r[i] : Qdot_r[i]
            invtau = invtau_c[i] + invtau_r[i]
            drive_cap = max(rho_v[i], 0.0) - rho_vs[i]
            if drive_cap < Q_ss[i]
                qcb = cloud_evap_realized(Qdot[i], f_cloud[i])
                if qcb != Qdot[i]
                    _cloud_evap_census!(cev_st, cev_tid, (qcb - Qdot[i]) * ts)
                    if cloud_evap_real
                        invtau_c[i] = f_cloud[i] * invtau_c[i]
                        Qdot[i] = qcb
                        invtau = invtau_c[i] + invtau_r[i]
                    end
                end
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
    return nothing
end

"""
    MCDriverParams

Stage 0 of `mc_driver!` resolved once per driver call: every option, flag, width and
parameter the driver reads, formed by [`_mc_driver_params`](@ref) with the setup refusals it
carries. Immutable and concretely typed (the transform modes and the population seed are
Symbols, so it is not isbits); the driver destructures it into the local names it always used.
"""
struct MCDriverParams
    Khdiff::Float64
    Kvdiff::Float64
    Khdiff_heat::Float64
    Kvdiff_heat::Float64
    tau_qss::Float64
    lobe_relax::Bool
    tau_lobe::Float64
    lobe_tr::Bool
    teos::Bool
    sed_lid_dir::Bool
    tau_pop::Float64
    ice_pop_src::Bool
    pop_seed::Symbol
    ice_shed::Bool
    ice_attr::Bool
    ice_agg_caps::Bool
    rain_evap_real::Bool
    cloud_evap_real::Bool
    ice_n_real::Bool
    rain_pop_gate::Bool
    Pr_t::Float64
    Khdiff_water::Float64
    Sc_t::Float64
    fcor::Float64
    alpha::Float64
    z_damp::Float64
    precipitation::Bool
    budget_trace::Bool
    cond_floor::Bool
    ctrans::Symbol
    ctrans_on::Bool
    cmu::Float64
    rtrans::Symbol
    rtrans_on::Bool
    rmu::Float64
    vtrans::Symbol
    vtrans_on::Bool
    vmu::Float64
    nr_i::Int
    rain_2m::Bool
    nrtrans::Symbol
    nrtrans_on::Bool
    nrmu::Float64
    ice_on::Bool
    itrans::Symbol
    itrans_on::Bool
    imu_q::Float64
    imu_n::Float64
    imu_a::Float64
    imu_c::Float64
    N_r::Float64
    N_0::Float64
    max_N_c::Float64
    louis_bl::Bool
    re_i::Int
    mynn_on::Bool
    rad_on::Bool
    hsi::Bool
    xsi::Bool
    xsi_zero::Bool
    hsi_like::Bool
    sd_si::Bool
    l_inf::Float64
    Ls::Float64
    K_min::Float64
    use_smag::Bool
    surface_fluxes::Bool
    SST::Float64
end

"""
    _mc_driver_params(model, IS, water_carry, RAD) -> MCDriverParams

The options/refusals block of `mc_driver!` (stage 0), moved verbatim: `IS` is the tile's
`MCSlots`, `water_carry` the MYNN state's water-carry mode (read only by a refusal), `RAD`
the tile's `RadiationState` (its `active` flag).
"""
@noinline function _mc_driver_params(model::ModelParameters, IS::MCSlots, water_carry,
                                     RAD::RadiationState)::MCDriverParams

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
    tau_qss = get(model.physical_params, :tau_qss, 10.0)
    # LOBE RELAXATION (Stage L; see `lobe_relaxation_increment`): the pragmatic device that
    # relaxes a transformed water-mass slot's refit lobe toward ν = 0 on `tau_lobe` where the
    # species' channel is closed, the vapor paying the lobe's Δν (the paired form). `false` is
    # bitwise the code without it.
    lobe_relax = lobe_relaxation_on(model.options)
    tau_lobe = get(model.physical_params, :tau_lobe, 10.0)
    if lobe_relax
        tau_lobe > 0.0 || error("physical_params[:tau_lobe] = $(tau_lobe) must be > 0 " *
                                "(the lobe relaxation timescale, s)")
    end
    # LOBE TRANSPORT (see `lobe_transport_jacobian`): below ν = 0 the continuity terms of the
    # five transformed water-mass slots read the odd continuation of the recovery and its
    # Jacobian, so their transport is the ν-image of the linear ρ_t's there. Default ON;
    # bitwise inert wherever every slot is non-negative and whenever no transform is
    # declared. `false` restores the clipped-point forms (`J = dbhyp(0) = 1`, `ρ = 0`) bitwise,
    # for forensics. (Measured 2026-10-08: those forms are inconsistent but are NOT what grows
    # the 2-D RCE's negative control-variable pool — that is the per-step refit of the
    # condensation closure's kinked cloud, see `lobe_relaxation_increment`; the consistent
    # forms change the pool by a few per cent.)
    lobe_tr = get(model.options, :water_lobe_transport, true)::Bool
    # (THE REFIT PARTNER, `REFIT_PARTNER_DEFAULT`, is the affine-defect booking of
    # `_water_exchange_stage!`, called from `explicit_timestep`; nothing of it is here.)
    # THE MODEL TEMPERATURE (`temperature_mode`): `:eos`, T = p/(ρ_d R_d + ρ_v R_v), unless the
    # run asks for the `:energy` retrieval. Read once per column, like every switch here.
    teos = temperature_mode(model.options) === :eos
    # The LID of every sedimentation-flux fit (rain mass, its energy, rain number, the twelve
    # ice moments, the ice energy): `:dirichlet` (the default and the physics) fits the flux
    # on `flux_column` — the slot's basis with F = 0 at the lid, since no hydrometeor enters
    # through the top of the domain. `:natural` fits it on the slot's own column (its own lid
    # BC), the pre-2026-09-26 behaviour, whose extrapolated lid flux is inflow and a linear
    # instability wherever the fall speed increases upward. Forensics only.
    sed_lid_dir = sedimentation_flux_lid_dirichlet(model.options)
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
    # The CLOUD donor's realization (`cloud_evap_realized`): every sink on the cloud —
    # evaporation, AUTO_COLL, the ice legs — on ONE factor `J₀(κ_tot Δt)`. Default ON; `false`
    # restores the pre-9acd5b8 unrealized draw bitwise (the ice legs' `f_qc` without the warm
    # conductance, `AUTO_COLL` and the evaporation unscaled). The `MC_CEVAP_*` census is
    # written either way.
    cloud_evap_real = get(model.options, :cloud_evap_realization,
                          CLOUD_EVAP_REALIZATION_DEFAULT)::Bool
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
              "correction cap (:vapor_blend_dcap) went with it. Remove the key; the vapor is " *
              "the nu-space closure of the water sum (the derived water anchor).")
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
    # The VAPOR slot's control variable (`vapor_transform_mode`): `:none` (the library
    # default) is bitwise the code that had no option — every vapor branch below takes its
    # `else` arm there. Under `:bhyp` the slot is `ν_v' = bhyp(ρ_v) − ν̄_v`, recovered with
    # `ahyp`, and every density-space tendency reaches it through `J_v`.
    vtrans = vapor_transform_mode(model.options)
    vtrans_on = vtrans !== :none
    vmu = vapor_mu(model.physical_params)
    # THE WATER ANCHOR (Stages 3-4 of the ρ_t retirement): every reader below takes
    # ρ_t = ρ_d + Σ_k α_k ν_k (`_anchor_stage!`) and the reference ρ̄_t^w; the vapor carries no
    # reconciliation nudge and there is no ρ_t slot. The combinations it cannot run are refused
    # at setup (`validate_water_anchor`), `physical_params[:Kvdiff_water] > 0` among them (the
    # implicit water solve read the deleted slot 3 as the total water).
    if (ctrans_on || rtrans_on || vtrans_on) && budget_trace
        # `water_budget_probe!` attributes a slot's production to named channels and reports
        # them in kg/m^3/s. Under a transform the ADV column is in CONTROL-VARIABLE units
        # while the source columns are densities, so the rows would not sum to the tendency
        # and the mismatch would be invisible. Refuse rather than print a wrong budget --
        # this file has shipped one silently before (see `water_budget_trace`).
        error("options[:water_budget_trace] with a water transform on " *
              "(condensate_transform = :$(ctrans), rain_transform = :$(rtrans), " *
              "vapor_transform = :$(vtrans)) is not " *
              "implemented: the advective term is then in control-variable units and the " *
              "source terms in density units, so the attribution does not close. Rescale " *
              "the transformed slot's columns by the Jacobian before enabling it.")
    end
    # Two-moment rain (options[:rain_moments] == 2; see `rain_moments`). Read from the SLOT,
    # not from the option: `MCSlots` resolved the appended index by name once when the tile
    # was built, and `> 0` is the same test the writes below use, so "the slot exists" and
    # "the physics runs" cannot disagree. Its own transform is independent of slot 8's.
    nr_i = IS.n_r
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
    # ── Ice (options[:ice_microphysics] === :ishmael; see `ice_microphysics`) ──
    # Read from the SLOTS, on the same rule the rain number follows: `MCSlots` resolved the
    # twelve appended indices by name once when the tile was built, and `ice_registered` is
    # the same `> 0` test the writes below use, so "the slots exist" and "the ice runs"
    # cannot disagree. ONE transform family covers all twelve; FOUR widths, because `mu`
    # carries the units of what it transforms (see `ice_transform_mode`, `MC_ICE_MU_KEYS`).
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
    # and Smagorinsky horizontal closure (a length scale > 0 replaces the constant Khdiff in
    # the momentum diffusion). Both default OFF so existing configurations are
    # bit-identical.
    louis_bl = get(model.options, :louis_bl, false)::Bool
    # ── MYNN-EDMF boundary layer (options[:mynn]; src/mynn_state.jl, and mc_mynn_bl.jl
    #    from S5). Read from the SLOT, on the rule the rain number and the ice family
    #    follow: `MCSlots` resolved the appended `rho_e` index by name once when the tile
    #    was built, and `> 0` is the same test the transport below uses, so "the slot
    #    exists" and "the closure runs" cannot disagree. At THIS stage the closure itself
    #    is not written: `rho_e` is transported with zero sources and nothing else changes,
    #    so a run without the option is byte-identical to the code that had no MYNN.
    re_i = IS.rho_e
    mynn_on = re_i > 0
    if mynn_on && louis_bl
        # One boundary layer, or two schemes mix the same column twice — the fluxes are
        # additive and neither knows about the other, so the surface layer would be
        # over-mixed by an amount nothing reports.
        error("options[:mynn] with options[:louis_bl] is not allowed: both are complete " *
              "boundary-layer closures and their tendencies are additive, so enabling " *
              "both mixes every column twice. Choose one.")
    end
    # ICE + MYNN is the SUPPORTED pair (S8; the Louis refusal above stands, because its
    # `rho_dot_v = rho_dot_w - rho_dot_c` decomposition attributes the ice flux to vapour).
    # `mc_mynn_bl!` mixes all twelve ice moments species-wise, all four moments of a species
    # on the SAME K_h, and carries the ice's `-L_f` share of the water energy in the
    # flux-form `S_Ew` column. The one combination that is still refused is the LOCAL
    # fixed-temperature water map, which has no ice term:
    if mynn_on && ice_on && water_carry === :fixed_T
        # `:fixed_T` is the D10 fidelity comparison: E_t receives `c_w rho_dot_w +
        # c_v rho_dot_v` pointwise instead of the divergence of the energy column. It has no
        # ice coefficient, and the routine it mirrors — `_diffusion_water_step!` — has no ice
        # handling to copy (it refuses ice itself, above). Adding a third pointwise term here
        # would be inventing a convention rather than mirroring one, so it is refused until
        # the local map is derived with ice.
        error("options[:mynn_water_carry] = :fixed_T with options[:ice_microphysics] = " *
              ":ishmael is not implemented: the local fixed-temperature water map carries " *
              "`c_w rho_dot_w + c_v rho_dot_v` and has no ice coefficient, and " *
              "`_diffusion_water_step!`, the routine it mirrors, has no ice handling to " *
              "copy. Use the default :flux carry (where the ice rides the S_Ew column as " *
              "-L_f K_h dz(rho_i)) or set ice_microphysics = :none.")
    end
    # Radiative heating. `RAD.q_lw`/`q_sw` are the HELD flux divergences [W/m^3] the
    # pre-pass (`radiation_prepass!`, src/radiation.jl) recomputed on the radiation cadence
    # before this column loop started; nothing is solved here. `rad_on` is a plain field
    # load because `ModelTile` carries the state concretely (see `EMPTY_RADIATION`), and it
    # gates the fold below so a radiation-free run is BYTE-identical to the code that had
    # no radiation at all -- `x + 0.0` is not the identity for `x = -0.0`.
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
    Ls = smagorinsky_length(model)
    K_min = get(model.physical_params, :K_min, 0.0)
    use_smag = Ls > 0.0

    # Bulk surface enthalpy/moisture fluxes over a fixed-SST ocean: the flux
    # values are the bottom nodes of the Louis heat/water flux columns, so the
    # switch requires louis_bl. SST is in KELVIN (Float64 params carry no units;
    # the guard catches the Celsius footgun).
    surface_fluxes = get(model.options, :surface_fluxes, false)::Bool
    if surface_fluxes && !(louis_bl || mynn_on)
        error("options[:surface_fluxes] requires a boundary-layer closure " *
              "(options[:louis_bl] or options[:mynn]) — the fluxes enter as the bottom " *
              "nodes of that scheme's flux columns and have nowhere to go without one")
    end
    SST = get(model.physical_params, :SST, 301.15)
    if surface_fluxes && SST <= 200.0
        error("physical_params[:SST] must be in Kelvin (got $SST — 28 C is 301.15)")
    end
    return MCDriverParams(Khdiff, Kvdiff, Khdiff_heat, Kvdiff_heat, tau_qss, lobe_relax,
                          tau_lobe, lobe_tr, teos, sed_lid_dir, tau_pop, ice_pop_src, pop_seed,
                          ice_shed, ice_attr, ice_agg_caps, rain_evap_real, cloud_evap_real,
                          ice_n_real, rain_pop_gate, Pr_t, Khdiff_water, Sc_t, fcor, alpha,
                          z_damp, precipitation, budget_trace, cond_floor, ctrans, ctrans_on,
                          cmu, rtrans, rtrans_on, rmu, vtrans, vtrans_on, vmu, nr_i, rain_2m,
                          nrtrans, nrtrans_on, nrmu, ice_on, itrans, itrans_on, imu_q, imu_n,
                          imu_a, imu_c, N_r, N_0, max_N_c, louis_bl, re_i, mynn_on, rad_on,
                          hsi, xsi, xsi_zero, hsi_like, sd_si, l_inf, Ls, K_min, use_smag,
                          surface_fluxes, SST)
end
