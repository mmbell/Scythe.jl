#!/usr/bin/env julia
# Ooyama (2001)-style warm-rain benchmark for the total-energy set.
#
# A 3 K RH-preserving warm bubble (cos² profile, 16 km x 3 km radii, centered
# at 500 m over the domain midpoint) rises through a humidified Dunion
# moist-tropical sounding (see DUNION_SOUNDING below for why the humidification
# is required), condenses, converts cloud to rain (autoconversion + collection),
# and rains out through the surface: the rho_r bottom boundary is a free
# (Natural) fit so the sedimentation flux removes water and its energy through
# z = 0. Rain condensation/evaporation runs through the tau_r channel of the
# generalized supersaturation relaxation (1/tau = 1/tau_c + 1/tau_r) with the
# monodisperse N_r closure — no separate Qevap parameterization (this differs
# from Ooyama 2001, as do the equation set and the single non-nested grid with
# the bubble at the domain center, so the targets are comparable-magnitude
# sanity windows, not a reproduction of his figures). Rain CONDENSATION is
# gated on cloud presence (see qss_condensation_rates): without the gate,
# spectral-ringing rain seeds grow in the wave-driven supersaturation at the
# lid and sediment back down as a spurious upper-tropospheric rain blob.
#
#   julia --project=. benchmarks/o01_rainfall.jl --mode quick --stage mc --grid rirk
#
# Modes: quick (2 km cells) | full (500 m cells). Both use 500 m vertical
# nodal spacing to the 25 km rigid lid, with a Durran-Klemp Rayleigh sponge
# (momentum-only, KE routed to E_t) over the top 8 km — confined to the
# stratosphere (the sounding tropopause knot is at 16.59 km) so deep
# convective flow is never damped, only the radiated gravity waves.
# Only --stage mc is supported.
#
# Radiation (S2b/S3b/S4): SCYTHE_O01_RAD selects an RRTMGP arm -- `lw` (clear-sky longwave
# only, the S2b production arm), `allsky` (same solver, all-sky kernels, with the S3a
# microphysics -> cloud-optics mapping live: a real cloudy arm), `sw` (clear-sky with a
# fixed sun), `allsky_sw` (all-sky WITH the fixed sun: cloud-top shortwave warming
# against cloud-top longwave cooling) or `diurnal` (clear-sky, full diurnal cycle at
# 20 N, day 240, 12:00 local).
# Unset or `0` sets NO radiation key at all and the run is bit-identical to the committed
# reference. SCYTHE_O01_RAD_FORCING (`full` default | `anomaly`), SCYTHE_O01_RAD_INTERVAL
# (cadence in seconds, default 300), SCYTHE_O01_RAD_ZMAX (taper height, unset = no taper)
# and SCYTHE_O01_RAD_RAIN (1 = put rain mass in the liquid water path, default off) tune
# the arm; SCYTHE_O01_RAD_COSZ / SCYTHE_O01_RAD_TOA override the FIXED sun's geometry
# (used to reproduce a `:diurnal` first call with `:fixed`), and
# SCYTHE_O01_RAD_TRACE_SW=1 turns on the coarse per-step shortwave-rescale line (already
# on for the `diurnal` arm). The radiative surface temperature is deliberately left
# unset, so the driver uses the column's own extrapolated surface air temperature (O01
# has no ocean).
#
# Reference: Ooyama (2001), J. Atmos. Sci. 58, 2073-2102 (Fig. 6: peak ground
# precipitation ~75-125 g m^-2 s^-1 at ~35-40 min for a similar bubble).
# reference/ooyama_jas2001.pdf

using Distributed

include(joinpath(@__DIR__, "common", "harness.jl"))
opts = parse_benchmark_args(ARGS)
opts.stage == STAGE_MC ||
    error("o01_rainfall supports only --stage mc (the total-energy set carries rho_r)")
opts.nests in (1, 3) ||
    error("o01_rainfall supports --nests 1 (single grid) or 3 (5-patch two-way nest)")
opts.nests == 1 || opts.grid == :rirk ||
    error("--nests 3 requires --grid rirk (spline interface coupling)")

# 3-level nest = 5 abutting patches; the fine center patch gets the extra workers
# in full mode (it dominates the column-step cost).
nested_workers(opts) = opts.mode == :full ? [1, 1, 4, 1, 1] : [1, 1, 1, 1, 1]

add_benchmark_workers(opts;
                      count = opts.nests > 1 ? sum(nested_workers(opts)) : opts.workers)
@everywhere using Springsteel
@everywhere using Scythe

include(joinpath(@__DIR__, "common", "diagnostics.jl"))

# ── Configuration ──────────────────────────────────────────────────────────

const MC_VARS = ["p", "rho_d", "rho_t", "u", "w", "E_t", "Q_ss", "rho_r", "rho_c",
                 "rho_v"]

# HUMIDIFIED Dunion moist-tropical sounding (WRF input_sounding format): RH floors
# of 0.90 (z <= 1.6 km) / 0.88 (<= 3.2 km) / 0.85 (<= 4.5 km) applied to the
# original dunion_MT.ref (committed alongside), mirroring Ooyama's own "slightly
# humidified" Jordan sounding. The UNmodified Dunion profile (low-level RH ~83%,
# falling to 56% at 4.4 km) does NOT convect from this trigger within the hour at
# ANY resolution: the wide slab's linear ascent (~w < 1 m/s) saturates a thin core
# at ~14 min but dry entrainment and CIN kill it — the archived RZ notebook run
# with the same bubble only erupted at t ~ 2.7 h, and only with Kv_mudiff = 100
# background moistening. The RH floors give O01-comparable onset (~24 min) and
# ground rain rates (quick ~28, full ~78 g m^-2 s^-1 vs O01's 75-125).
const DUNION_SOUNDING = joinpath(REFERENCE_DATA_DIR, "o01_rainfall", "dunion_MT_hum90.ref")

# Rain-drop number concentration [#/cm^3] for the monodisperse tau_r closure
# (~1000 drops per m^3; the small number keeps the bulk of condensation on cloud).
const N_R = 1.0e-3

# Marshall-Palmer intercept [m^-4] for the exponential-DSD tau_r closure, for
# sensitivity runs only (SCYTHE_O01_N0=8.0e6 selects MP; the default 0.0 keeps
# the monodisperse closure the references were seeded with, bit-identical).
const N_0_MP = parse(Float64, get(ENV, "SCYTHE_O01_N0", "0.0"))

# Vertical eddy coefficients [m^2/s]. ZERO: the run is stable fully inviscid on
# the RiRk grid at both resolutions — the cubic B-spline Galerkin filter is the
# only dissipation, which is the near-inviscid goal. (Kv = 5 and 25 were tested
# and change the solution by < 1%; raise these if a future configuration needs
# damping, they feed the momentum/heat/water solves independently.) The rain
# shafts do ring: min(rho_r) undershoots reach ~ -1.8 g/m^3 at 500 m resolution
# (all rate functions are negative-safe; reported as min_rho_r_gm3).
const KV_MOM = 0.0
const KV_HEAT = 0.0
const KV_WATER = 0.0

function o01_model(opts::BenchmarkOptions)
    if opts.mode == :full
        num_cells_i = 300       # 500 m cells over 150 km
        ts = 0.15
    else
        num_cells_i = 75        # 2 km cells
        ts = 0.3
    end
    # Negative-water attribution sweeps (reference/HANDOFF_NEGATIVE_WATER.md). These
    # SCYTHE_O01_* knobs isolate one term at a time in the single-grid ablation matrix;
    # ALL of them are no-ops when unset, so the committed config is bit-identical.
    haskey(ENV, "SCYTHE_O01_NI") && (num_cells_i = parse(Int, ENV["SCYTHE_O01_NI"]))
    # Timestep override, for separating a CFL/SI-ceiling violation from a physics error:
    # if halving ts makes an unstable configuration stable, the instability is the ceiling,
    # not the formulation. The water species feed the acoustic coefficient through q_l, so
    # a change to the condensate CAN move that ceiling.
    haskey(ENV, "SCYTHE_O01_TS") && (ts = parse(Float64, ENV["SCYTHE_O01_TS"]))
    # The ice arm's production timestep is 0.15 s (author decision 2026-08-20). Stage B
    # integrates every microphysical relaxation consistently at any ts (per-donor J0
    # realization; the donor census verifies), and at ts=0.15 the ice arm runs the full
    # 3600 s through glaciation with every donor bounded. The quick default ts=0.3 still
    # dies at t=2454.9 s through a chain the integrator cannot own: water-partition
    # detachment at the 13 km glaciation front (independently advected ice moments at
    # 4.1x the conserved rho_t anchor, rho_v absorbing -8.4 g/m^3 as the closing
    # residual) -> 363 K retrieval -> vertical-acoustic SI ceiling breach (Co_w = 2.08
    # against the 1.59x margin). That detachment is a transport/representation problem,
    # the named next-stage target; evidence preserved under
    # benchmarks/output/o01_rainfall_quick_mc_rirkstageB* (2026-08-20). SCYTHE_O01_TS
    # overrides this default, which is how the ts=0.3 chain stays reproducible.
    if get(ENV, "SCYTHE_O01_ICE", "") in ("1", "true", "yes", "ishmael") &&
       !haskey(ENV, "SCYTHE_O01_TS")
        ts = 0.15
    end
    # Vertical: 500 m nodal spacing to 25 km in BOTH modes (the rain physics and
    # the sedimentation flux do not coarsen with the horizontal grid). The top
    # 8 km (17-25 km) is the Rayleigh sponge; the 25 km lid (vs the historical
    # 20 km) buys a stratosphere-confined absorber above the 16.59 km tropopause.
    num_cells_k = 100            # RiRk: 250 m cells over the 25 km lid (kDim = 300)
    haskey(ENV, "SCYTHE_O01_NK") && (num_cells_k = parse(Int, ENV["SCYTHE_O01_NK"]))
    kDim = 300                  # RZ: Chebyshev points (untargeted fallback)
    output_interval = 60.0

    ts = vertical_ts(ts, opts)

    # `vars` is resolved AFTER the transform options are parsed, below: a transformed slot is
    # renamed (`rho_c` -> `nu_c`, `rho_r` -> `nu_r`) so that the output columns say what they
    # hold. See `Scythe.mc_var_names`.
    # Rayleigh sponge (Durran-Klemp 1983 eq. 29 profile, momentum-only in mc):
    # onset at 17 km (just above the sounding's 16.59 km tropopause knot, so the
    # damping lives entirely in the high-static-stability stratosphere), 8 km =
    # 16 cells deep. alpha = 0.02 1/s puts the lid e-folding at ~39 s (the DK
    # profile peaks at 1.285*alpha) against stratospheric wave intrinsic
    # frequencies ~0.005-0.015 1/s — inside the Klemp-Lilly 2 <= alpha/omega <= 5
    # optimum for the dominant modes.
    physical_params = Dict(:Khdiff => 0.0, :Kvdiff => KV_MOM,
                           :Khdiff_heat => 0.0, :Kvdiff_heat => KV_HEAT,
                           :Kvdiff_water => KV_WATER,
                           :tau_qss => 10.0, :N_r => N_R, :N_0 => N_0_MP,
                           :alpha => 0.02, :z_damp => 17.0e3)
    options = merge(Dict{Symbol,Any}(:semiimplicit => true, :exact_reference_state => true,
                                     :precipitation => true, :vertical_mixing => false),
                    reference_state_options())
    # Attribution knobs: toggle the water source terms ("0"/"1"). Unset => defaults
    # (:precipitation on, :condensation on-by-absence), i.e. bit-identical.
    envflag(k) = ENV[k] in ("1", "true", "yes")
    haskey(ENV, "SCYTHE_O01_PRECIP") && (options[:precipitation] = envflag("SCYTHE_O01_PRECIP"))
    haskey(ENV, "SCYTHE_O01_COND")   && (options[:condensation]  = envflag("SCYTHE_O01_COND"))
    # Attribution: lower the negative-water warning threshold so the PRE-refit
    # (clamp_water!) ladder prints every doubling, to compare against the POST-refit
    # min_rho_r_gm3 from the output CSVs (the production-vs-ringing split).
    haskey(ENV, "SCYTHE_O01_WARNDT") && (options[:water_warn_dT] = parse(Float64, ENV["SCYTHE_O01_WARNDT"]))
    # Attribution: per-step term-by-term water production budget (see water_budget_probe!).
    # Value = print interval in steps; unset/0 => probe never runs, bit-identical.
    # NOTE the output lands in <output_dir>/scythe_err.log, NOT the console — Scythe.jl:136
    # redirects the worker's stderr for the whole run.
    haskey(ENV, "SCYTHE_O01_BUDGET") && (options[:water_budget_trace] = parse(Int, ENV["SCYTHE_O01_BUDGET"]))
    # Microphysics STIFFNESS census (see Scythe.mc_stiffness_census!): max ts/tau and the
    # gridpoint-step count past ts/tau = 1, per relaxation channel. Value = print interval in
    # steps; unset/0 => no printing. The census itself runs on EVERY run either way and warns
    # once if any channel exceeds 1, so this knob only controls the periodic report and is
    # bitwise inert. This is what replaced the SCYTHE_O01_CAPFAC/CAPMODE depletion-cap levers,
    # which went with the caps themselves: a rate floored at `rho/ts` makes the physics a
    # function of the time step. Output lands in <output_dir>/scythe_err.log, not the console.
    haskey(ENV, "SCYTHE_O01_STIFFNESS") &&
        (options[:stiffness_trace] = parse(Int, ENV["SCYTHE_O01_STIFFNESS"]))
    # First link of the reconciliation chain Q_ss -> rho_v -> the rho_t budget: the timescale
    # on which `Scythe.qss_relaxation` pulls the prognostic Q_ss onto the supersaturation the
    # PROGNOSTIC vapor implies. A shorter tau keeps the two tied together at the cost of the
    # advected Q_ss's smoothness -- and Q_ss is carried precisely because near saturation the
    # supersaturation is four decades below the vapor, right at the 1e-4 nucleation gate.
    # This knob sweeps that tradeoff. Unset => 10.0, the committed default, bitwise.
    haskey(ENV, "SCYTHE_O01_TAUQSS") &&
        (physical_params[:tau_qss] = parse(Float64, ENV["SCYTHE_O01_TAUQSS"]))
    # The second link of the same chain: the timescale on which the PROGNOSTIC vapor is
    # reconciled with the vapor the conserved rho_t budget implies (`Scythe.rho_v_reconcile`).
    # Unset => 10.0, the committed default, bitwise. Short tau_rec projects rho_v onto the
    # density budget every few steps and throws away the transported field's smoothness --
    # which is the thing the prognostic vapor was introduced to buy; long tau_rec lets the
    # partition drift (watch the reconciliation gap in the water summary). This knob sweeps
    # that tradeoff, and it is the one that replaced SCYTHE_O01_VAPOR/BLEND_T0T1/BLEND_DCAP:
    # those chose between two DIAGNOSTIC vapors, and there is no diagnostic vapor any more.
    haskey(ENV, "SCYTHE_O01_TAUREC") &&
        (physical_params[:tau_rho_v_rec] = parse(Float64, ENV["SCYTHE_O01_TAUREC"]))
    # The third link of the chain (Stage C): the timescale on which the advected ice
    # PARTITION is reconciled with the water the conserved rho_t anchor supports
    # (`Scythe.ice_anchor_rate`; TeX §Reconciliation of the condensate partition). Unset =>
    # 10.0, the committed default, bitwise. Sized by measurement: at the recorded
    # transport-phase feed (<= 1.7e-6 kg/m^3/s across the three preserved dying runs) the
    # detachment equilibrates at F*tau -- a ~0.16 K retrieval excursion at 10 s, thirty
    # times under the 5 K ship-the-cap criterion. Ice-arm only; inert (and the knob
    # meaningless) with ice off.
    haskey(ENV, "SCYTHE_O01_TAUANCHOR") &&
        (physical_params[:tau_ice_anchor] = parse(Float64, ENV["SCYTHE_O01_TAUANCHOR"]))
    # The anchor-reconciliation SOURCE switch. Unset => on (the committed default). `=0`
    # drops the removal while the MC_ANCHOR_* census keeps measuring the defect -- the
    # bitwise reproduction of the unreconciled §2c configuration, for forensics:
    #   SCYTHE_O01_ICE=1 SCYTHE_O01_RAIN_MOMENTS=2 SCYTHE_O01_ANCHOR=0 SCYTHE_O01_TS=0.3
    # reproduces the 2454.9 s detachment death.
    haskey(ENV, "SCYTHE_O01_ANCHOR") &&
        (options[:ice_anchor_source] = ENV["SCYTHE_O01_ANCHOR"] != "0")
    # The reader-side leg of the same reconciliation: the thermodynamic interface's
    # rho_ice_t capped at the anchor headroom (the `condensate_floor` device class —
    # readers only, no write-back, no mass conversion). Unset => on (the committed
    # default): the terminal §2c burst is a per-step fit oscillation at the front that no
    # tau-relaxation outruns, and the cap is what breaks its retrieval amplifier. `=0`
    # drops the cap (state and census unaffected) for forensics.
    haskey(ENV, "SCYTHE_O01_ANCHORFLOOR") &&
        (options[:ice_anchor_floor] = ENV["SCYTHE_O01_ANCHORFLOOR"] != "0")
    # The sedimentation leg of the same reconciliation: the flux assemblies transport the
    # anchor-supported share of each ice moment (one factor per gridpoint, all twelve
    # fluxes, telescoping preserved exactly), so phantom mass cannot move rho_t or E_t by
    # falling. Unset => on (the committed default). `=0` drops it for forensics.
    haskey(ENV, "SCYTHE_O01_ANCHORFLUX") &&
        (options[:ice_anchor_flux] = ENV["SCYTHE_O01_ANCHORFLUX"] != "0")
    # The rate-side leg: every ISHMAEL process rate reads the anchor-supported share of
    # the ice population (same single factor; per-particle state untouched; melt on
    # phantom ice is the channel it closes). Unset => on. `=0` drops it for forensics.
    haskey(ENV, "SCYTHE_O01_ANCHORRATES") &&
        (options[:ice_anchor_rates] = ENV["SCYTHE_O01_ANCHORRATES"] != "0")
    # The per-channel ATTRIBUTION census (`Scythe.MC_ATTR_*`): which LEG the `MC_DONOR_*`
    # breaches come from -- the six q_r legs at the breach points, rain evaporation at the
    # applied step-mean, the q_i melt/aggregation split, and what runs above the melting
    # level. Unset => off, and the run is bitwise the committed one: the block writes only
    # into `mc_water_stats` and every write is gated on a state test a warm ice-free column
    # fails identically. Read it in the stiffness trace (`options[:stiffness_trace]`).
    haskey(ENV, "SCYTHE_O01_ATTR") && (options[:ice_attr_census] = envflag("SCYTHE_O01_ATTR"))
    # The PER-PAIR reservoir caps inside the aggregation kernel (`Scythe.ishmael_col1`'s
    # `min(colamt, q)` / `min(colamtn, n)`): a `dt` inside a rate law, the `min(rate, rho/dt)`
    # class this port removes at every other ISHMAEL call site. They stood because nothing
    # else bounded the three-pair sums; the donor realization factors now do (Stage 1). Unset
    # => on, bitwise the Fortran's caps. `=0` retires the class on the aggregation kernel too
    # -- an answer-changing forensic arm, to be read against the MC_DONOR_I1/I2 census and
    # SCYTHE_O01_ATTR's MC_ATTR_AGG1_SAT count (which is what says whether they bind at all).
    haskey(ENV, "SCYTHE_O01_AGGCAPS") &&
        (options[:ice_agg_caps] = ENV["SCYTHE_O01_AGGCAPS"] != "0")
    # ICE NUMBER REALIZATION (Stage 1b): each species' NUMBER as a donor reservoir of its
    # own, with its own conductance over the three legs that draw on it (aggregation's
    # `deltan`, the melt number `nmlt`, the sublimation number sink) instead of riding the
    # species' MASS factor. The two are not proportional -- `colamt` and `colamtn` come from
    # two offline tables integrating two different moments of the collection kernel (measured
    # kappa_n/kappa_q = 0.46 on the stiff-cold fixture, 0.20 on an anvil-like state), and the
    # melt number carries `dNmltri`, a number sink with no mass partner. Unset => OFF, the
    # committed default and BITWISE the pre-Stage-1b answer on every path. `=1` turns it on.
    # Read it against the three new MC_DONOR_N1/N2/N3 census rows, which are written in BOTH
    # modes: what they read with this off is the measurement, what they read with it on is
    # `1 - exp(-kappa_n dt) <= 1` by construction. The question it was built for is the
    # glaciated hour's number-less ice mass -- the MC_POP_MAX/MC_POP_SEED rows.
    haskey(ENV, "SCYTHE_O01_ICENREAL") &&
        (options[:ice_number_realization] = envflag("SCYTHE_O01_ICENREAL"))
    # RAIN EVAPORATION in the rain donor's conductance (Stage 2b; TeX §donor_relax, "The rain
    # reservoir has a third sink that lived outside its conductance"). Unset => on, the
    # committed construction: `kappa_ev = max(-Qdot_r,0)/rho_r` joins `kappa_tot`, one factor
    # `J_0(kappa_tot dt)` is formed over all of it, and the realized `f_r/tau_r` replaces
    # `1/tau_r` in lambda, in N and in the rain transfer at once. It is NOT ice-gated -- the
    # conductance is a warm-path quantity -- so this knob moves the warm answer too. `=0`
    # forces `kappa_ev = 0` and nothing else, which is the pre-Stage-2b behaviour BITWISE on
    # every path: the factors go back to the ice-leg-only expression and the fold becomes a
    # multiplication by an exact 1.0. Forensic arm, to be read against `SCYTHE_O01_ATTR`'s
    # MC_ATTR_QR_COMB (1.0012 reservoirs over 1296 gridpoint-steps with it off) and the
    # MC_DONOR_QR row, which only measures the combined draw with it on.
    haskey(ENV, "SCYTHE_O01_EVAPREAL") &&
        (options[:rain_evap_realization] = ENV["SCYTHE_O01_EVAPREAL"] != "0")
    # The POPULATION-reconciliation timescale [s], the fourth tier of the reconciliation
    # chain (Stage 3a; TeX §"Reconciliation of the population"). Unset => 10.0, the
    # tau_anchor class, bitwise. Ice-arm only.
    haskey(ENV, "SCYTHE_O01_TAUPOP") &&
        (physical_params[:tau_ice_population] = parse(Float64, ENV["SCYTHE_O01_TAUPOP"]))
    # The population-reconciliation SOURCE switch. Unset => on (the committed default): ice
    # mass whose carried number is not positive is orphaned by the population gate — no rate
    # may act on it, so no device can remove it either — and is returned to a representation
    # the equations can use, as rain with L_f above T_0 and as 2 um crystals below it. `=0`
    # drops both transfers while the MC_POP_* census keeps measuring the defect: the bitwise
    # reproduction of the unreconciled tree, in which the sub-melting-level ice is 97-99.9%
    # number-less by mass over the final half hour.
    haskey(ENV, "SCYTHE_O01_POPSRC") &&
        (options[:ice_population_source] = ENV["SCYTHE_O01_POPSRC"] != "0")
    # WHICH crystal the below-T_0 branch of that source seeds. Unset => `:local`, the
    # committed default (author decision 2026-08-31, on the measurements below and in
    # reference/FINDINGS_ISHMAEL_S8S9.md 5f-5j). `=min` is the 2 um sphere, the smallest the
    # scheme resolves, the fastest-responding population and the slowest-falling one; it
    # reconciles the dead mass but pays a deposition surface stiff enough to sublimate the
    # seed inside a step and leaves a third of the cloud (IWP 3.43 -> 1.07, ice top 18.8 ->
    # 15.4 km). `=large` seeds instead what
    # var_check re-diagnoses from the dead mass at the floor number -- the 1 mm large-ice
    # limit at the species' bulk density, i.e. the SIZE-SORTED particles the number-less mass
    # actually is (its mass-weighted fall speed outran its number-weighted one). The 2 um
    # seeding puts ~1e12 crystals/m^3 on the 6e-3 kg/m^3 dead masses this run carries, a
    # deposition surface stiff enough to sublimate within a step in subsaturated air (ts/tau
    # 31 on ice1, 9.7 on ice3) and measured thinning the ice cloud 3x; `=large` is ~1.5e10
    # times fewer particles for the same mass, and was measured leaving the cloud 72% dead at
    # 3600 s -- too sparse to survive the number field's own transport ringing, so the mass is
    # dead again the next step, its windows passing only because nothing was reconciled. The
    # DEFAULT `:local` is the seed those two measurements motivate: the dead mass is the
    # NEGATIVE LOBE of the number moment's ringing (its other face is the 3e13 /L number
    # spikes beside it), so the crystals it lost are the crystals NEXT DOOR -- the per-crystal
    # mass, habit and bulk density of the nearest gridpoint in the same column where that
    # species is still live, clamped between the `:min` and `:large` crystals and put through
    # var_check. It keeps the cloud at its unreconciled magnitude (IWP 3.30, ice top 19.2 km)
    # and, with the Stage 3e minimum-crystal bound and rain population gate, reconciles it:
    # 2.7% dead mass at 3600 s, 9/11 windows, max n_i1 1.7e5 /L. Either non-default arm is
    # answer-changing on the ice path and bitwise inert on the warm and dry ones.
    haskey(ENV, "SCYTHE_O01_POPSEED") &&
        (options[:ice_population_seed] = Symbol(ENV["SCYTHE_O01_POPSEED"]))
    # The RAIN POPULATION GATE (Stage 3e; FINDINGS 5i). Unset => ON, the committed default:
    # where the CARRIED rain number is not positive, the kernels that need a rain SIZE
    # DISTRIBUTION see no rain. ishmael_rain_lambda floors nr at QNSMALL and clamps the slope
    # to LAMMINR, so a rain slot that rang to mass-without-number is handed to every DSD
    # consumer as 2800 um drops -- the largest particle the scheme admits, at 192-203 K where
    # Bigg's exp(0.66 dT) is 1e21. That was the number pump: ~1e15 /m^3/s of ice crystals at a
    # millionth of the minimum resolved mass, and the anvil 70-99% dead mass behind it.
    # Bigg is the only kernel this switches off -- ishmael_ice_rain_riming self-gates, every
    # rate it returns carrying a factor of the carried nr. Bitwise inert wherever n_r > 0, so
    # `=0` is a forensic arm and not an ablation of anything the healthy storm uses. The
    # minimum-crystal bound on the realized ice-number sources that ships with it has NO
    # switch: it is an invariant, like the ice population gate.
    haskey(ENV, "SCYTHE_O01_RAINGATE") &&
        (options[:rain_population_gate] = ENV["SCYTHE_O01_RAINGATE"] != "0")
    # SHED ABOVE THE FREEZING LEVEL (Stage 3b; TeX §Departures (e)): above T_0 a crystal that
    # collects liquid sheds it, so there is no liquid-to-ice conversion — the collection rate
    # is still evaluated, so the melting rate keeps the sensible heat of the liquid that
    # struck the crystal, but no mass leaves the cloud or the rain and no L_f is released.
    # Unset => on (the committed default). `=0` restores ISHMAEL's ungated wet-growth
    # transfer, which the census caught taking the whole of the rain into the ice in a single
    # step two degrees above freezing; read it against SCYTHE_O01_ATTR's MC_ATTR_WARM_RIME_R
    # and MC_ATTR_WARM_MELT, which are that loop. Inert below T_0, bitwise.
    haskey(ENV, "SCYTHE_O01_SHED") &&
        (options[:ice_shed_above_t0] = ENV["SCYTHE_O01_SHED"] != "0")
    # The attribution block is PRINTED only by the periodic stiffness trace, so asking for
    # the census without a trace interval would integrate an hour and report nothing: the
    # knob supplies the production interval (4000 steps) unless one was given explicitly.
    get(options, :ice_attr_census, false) && !haskey(options, :stiffness_trace) &&
        (options[:stiffness_trace] = 4000)
    # Cloud droplet number ceiling [#/cm^3] for the Twomey activation branch AND the KK2000
    # two-moment autoconversion (both read physical_params[:max_N_c]; the driver passes one
    # number so a column carries ONE droplet population). Unset => 100.0, the closure's
    # long-standing default, bitwise. The ISHMAEL Fortran hardcodes 200 cm^-3 in its KK2000
    # (module_mp_jensen_ishmael.F:1031): PRC ~ N_c^-1.79, so `=200` cuts cloud->rain
    # conversion ~3.4x and carries more cloud to the -35 C level — the S9 tuning arm
    # (SCYTHE_BENCH_TAG it; answer-changing for the whole warm path, never a default flip).
    haskey(ENV, "SCYTHE_O01_MAXNC") &&
        (physical_params[:max_N_c] = parse(Float64, ENV["SCYTHE_O01_MAXNC"]))
    # Whether the THERMODYNAMIC INTERFACE reads a floored rho_liq (Experiment 1 of
    # reference/HANDOFF_CONDENSATE_REPRESENTATION.md). Unset => `:none`, bitwise the code
    # that had no option. `=diagnostic` floors rho_liq at the retrieval, q_l, Q_s_energy,
    # the entropy and E_sed ONLY, leaving the state, the continuity terms and the water
    # partition raw -- it removes the -54 K-class cold anomaly the undershoot carries
    # without touching the undershoot. See `Scythe.condensate_floor_mode` for why it is not
    # `clamp_water!`, and reference/FINDINGS_CONDENSATE_STAGE1.md for what the anomaly costs.
    haskey(ENV, "SCYTHE_O01_CFLOOR") &&
        (options[:condensate_floor] = Symbol(ENV["SCYTHE_O01_CFLOOR"]))
    # Which CONTROL VARIABLE slot 9 carries. Unset => `:none`, bitwise the code that had no
    # option: the slot is the cloud density itself. `=bhyp` makes it Ooyama's biased
    # hyperbolic control variable, so the recovered density is non-negative by construction
    # and the spline undershoot can no longer reach the temperature retrieval -- the
    # class-level fix for the negative-condensate reservoir, which reaches 2.75x the real
    # cloud mass on this very run (reference/FINDINGS_CONDENSATE_STAGE1.md §1). `=bhyp_smooth`
    # is the C^inf variant, bounded below by -mu rather than 0. NOTE the default POSITIVITY=r
    # is rain-only, which is what the transform needs: declaring positivity for rho_c
    # alongside it is refused (see install_positivity_bounds!).
    haskey(ENV, "SCYTHE_O01_CTRANS") &&
        (options[:condensate_transform] = Symbol(ENV["SCYTHE_O01_CTRANS"]))
    # The transform's bias [kg/m^3]. Unset => 1e-7, Ooyama's own value, which costs at most
    # 7.2e-3 K through the retrieval and that at the model top. Only meaningful with CTRANS.
    haskey(ENV, "SCYTHE_O01_CMU") &&
        (physical_params[:condensate_mu] = parse(Float64, ENV["SCYTHE_O01_CMU"]))
    # The same pair for slot 8 (rain). Unset => `:none`, bitwise the code that had no option.
    # `=bhyp` gives rain the treatment cloud already has. The design reason is nesting: the
    # coefficient limiter below is exact and free for rain on a SINGLE grid, but a nested
    # child's i-boundary is R3X, which the bound rejects, so children run with a k-only bound
    # and a documented leak. The transform has no such restriction.
    haskey(ENV, "SCYTHE_O01_RTRANS") &&
        (options[:rain_transform] = Symbol(ENV["SCYTHE_O01_RTRANS"]))
    haskey(ENV, "SCYTHE_O01_RMU") &&
        (physical_params[:rain_mu] = parse(Float64, ENV["SCYTHE_O01_RMU"]))

    # How many moments of the rain DSD to carry. Unset => the key is absent => 1, the
    # single-moment Ooyama closure, bit-identical. "2" appends the prognostic rain number
    # density n_r and switches the whole rain closure to ISHMAEL/Morrison (KK2000
    # autoconversion + accretion, Beheng self-collection, the ventilated exponential-DSD
    # evaporation timescale, and mass/number-weighted fall speeds that size-sort).
    # See `Scythe.rain_moments`.
    haskey(ENV, "SCYTHE_O01_RAIN_MOMENTS") &&
        (options[:rain_moments] = parse(Int, ENV["SCYTHE_O01_RAIN_MOMENTS"]))
    # The n_r control-variable transform, the number sibling of RTRANS. Its width is in
    # #/m^3, not kg/m^3 — nothing to do with RMU.
    haskey(ENV, "SCYTHE_O01_NRTRANS") &&
        (options[:rain_number_transform] = Symbol(ENV["SCYTHE_O01_NRTRANS"]))
    haskey(ENV, "SCYTHE_O01_NRMU") &&
        (physical_params[:mu_rain_n] = parse(Float64, ENV["SCYTHE_O01_NRMU"]))

    # ICE. Unset => the key is absent => `:none` => no ice slots at all, bit-identical.
    # "1"/"ishmael" appends the twelve ISHMAEL slots (three species x mass, number and the two
    # spheroid volume moments) and turns on the ice thermodynamics: the rho_i term of the
    # closed-form temperature retrieval, q_i*Ci in the mixture heat capacities and the entropy,
    # and rho_i in the rho_t water budget. It REQUIRES SCYTHE_O01_RAIN_MOMENTS=2 and says so if it
    # is missing (`Scythe.ice_microphysics`). The physics is LIVE: deposition through the
    # shared prognostic Q_ss, the full ISHMAEL nucleation/riming/aggregation/melting set, and
    # the Mitchell-Heymsfield fall speeds. Zero initial ice does NOT mean an inert run -- ice
    # nucleates wherever the air is subfreezing and supersaturated, which above the bubble it
    # is. The inertness that survives is on a WARM state, where every rate is gated off by a
    # state test and the ten common slots reproduce the ice-off run bitwise.
    haskey(ENV, "SCYTHE_O01_ICE") &&
        (options[:ice_microphysics] =
             ENV["SCYTHE_O01_ICE"] in ("1", "true", "yes", "ishmael") ? :ishmael : :none)
    # The var_check consistency source (default ON; see `Scythe.ice_microphysics`). Set to 0
    # to carry the ice moments with no consistency restoration at all, which is the
    # configuration that diverges at t ~ 1050 s and is kept switchable for exactly that
    # measurement.
    haskey(ENV, "SCYTHE_O01_ICEVARCHECK") &&
        (options[:ice_var_check] = ENV["SCYTHE_O01_ICEVARCHECK"] in ("1", "true", "yes"))
    # The ice control-variable transform: ONE family key for all twelve slots (the moments of
    # a species are ratios of each other -- see `Scythe.ice_transform_mode`), the ice sibling
    # of RTRANS/NRTRANS.
    haskey(ENV, "SCYTHE_O01_ICETRANS") &&
        (options[:ice_transform] = Symbol(ENV["SCYTHE_O01_ICETRANS"]))
    # The transform widths, as "mass,number,a,c" -- FOUR numbers, because mu is dimensional
    # and the four moment kinds are ten and nine decades apart (kg/m^3, #/m^3, m^3/m^3).
    # Unset => Scythe.MC_ICE_MU_DEFAULTS = (1e-12, 1e-2, 1e-16, 1e-16), each two to five
    # decades below the moment it transforms (see there for the sizing principle and why
    # the former (1e-7, 1e2, ...) made bhyp a bare positivity floor on this arm).
    # Only meaningful with ICETRANS.
    if haskey(ENV, "SCYTHE_O01_ICEMU")
        imus = parse.(Float64, split(ENV["SCYTHE_O01_ICEMU"], ","))
        length(imus) == 4 || error("SCYTHE_O01_ICEMU takes four comma-separated widths " *
                                   "(mass,number,a,c); got \"$(ENV["SCYTHE_O01_ICEMU"])\"")
        physical_params[:mu_ice]   = imus[1]
        physical_params[:mu_ice_n] = imus[2]
        physical_params[:mu_ice_a] = imus[3]
        physical_params[:mu_ice_c] = imus[4]
    end

    # ── The ICE ARM'S PRODUCTION CONFIGURATION: ALL FOUR water transforms on, the cloud
    #    included (restored 2026-08-19 under the prognostic vapor; measurement below) ──
    #
    # `SCYTHE_O01_ICE=1` turns the cloud, rain-mass, rain-number and twelve-ice-moment
    # transforms on by DEFAULT, because that is the configuration the ice physics is meant
    # to be run in, not an option on top of it.
    #
    # The CLOUD transform was DELIBERATELY :none from 2026-08-14 to 2026-08-19. Measured
    # then (model_tests/ice_transform_ablation_compare.jl, pre- and post-gate ladders): the
    # ahyp clamp on the cloud slot discards the negative half of the spline ringing, the
    # RESIDUAL vapor absorbed the discarded mass (8.3% deeper deficit over 23% more points),
    # and the ice physics reading that vapor detonated the temperature retrieval at
    # t ~ 1000 s — `min rho_c == 0.0 exactly` separated dying from surviving runs in all
    # eight ladder arms, and cloud=:none with everything else transformed survived.
    #
    # STAGE A (prognostic rho_v, 0957496) removed that channel's medium — the vapor is no
    # longer a residual for the discarded mass to land in — and the RE-MEASUREMENT
    # (2026-08-19, arms stageA_r6_ice vs stageA_r8_icebhyp) confirms the channel is gone:
    # with the cloud transformed the arm no longer has a death mode of its own. Both arms
    # now die of the SAME finding-4 glaciation stiffness, with the same single
    # stiffness-census signature on channel ice1 (cloud :none at t = 1031.7 s, ts/tau 59.7;
    # cloud :bhyp at t = 1002.6 s, ts/tau 23.6 — the ~30 s offset is supply timing at the
    # -35 C level, not a mechanism difference). The stiffness itself is Stage B's problem;
    # this setting is about which representation the arm carries when it gets there.
    #
    # The reason is the glaciation itself. Freezing an anvil is genuinely fast: homogeneous
    # freezing, riming and ice-rain collection empty the liquid reservoirs on timescales of
    # seconds, and an UNTRANSFORMED slot integrated explicitly through that overshoots below
    # zero, at which point every `max(rho,0)`-guarded rate switches off and the negative
    # region becomes a physics-free reservoir that nothing can drain (the mechanism
    # reference/FINDINGS_CONDENSATE_STAGE1.md documents for the cloud). Measured on the first
    # live ice arm: rho_r ran from -5e-4 to -62 kg/m^3 in twenty steps at t ~ 990 s, and
    # thirding the step moved the divergence by 20 s, not by a factor of three.
    #
    # Under the biased hyperbolic transform the RECOVERED density is bounded below by -mu
    # whatever the control variable does, so the collection rates vanish smoothly as a
    # reservoir empties instead of changing sign, and the overshoot is returned through the
    # prognostic vapor and supersaturation. Nothing is repaired and nothing is
    # clamped; it is a change of variables (Ooyama 2001 Eq. 4.19-4.23, `Scythe.bhyp`).
    #
    # Each is still individually overridable by its own env knob above, and NONE of this fires
    # when ice is off, so the default path and the plain two-moment arm are untouched.
    if Scythe.ice_microphysics(options) === :ishmael
        haskey(ENV, "SCYTHE_O01_CTRANS")  || (options[:condensate_transform]   = :bhyp)
        haskey(ENV, "SCYTHE_O01_RTRANS")  || (options[:rain_transform]         = :bhyp)
        haskey(ENV, "SCYTHE_O01_NRTRANS") || (options[:rain_number_transform]  = :bhyp)
        haskey(ENV, "SCYTHE_O01_ICETRANS")|| (options[:ice_transform]          = :bhyp)
    end

    # ── RADIATION (S2b). Unset (or "0") => NOT ONE radiation key is set, so the committed
    #    configuration is bit-identical: `mc_radiation_state` sees no `:radiation` key,
    #    returns `Scythe.EMPTY_RADIATION`, and `mc_driver!`'s fold is gated off entirely
    #    (the gate is `if rad_on`, not `+ 0.0` — adding an exact zero is not the identity
    #    for -0.0, and every committed reference here rests on that).
    #
    #    SCYTHE_O01_RAD selects the ARM:
    #      lw      RRTMGP clear-sky longwave only (`:clearsky`, `:solar = :none`). The S2b
    #              production arm: no shortwave is solved for, the cloud-optics fields are
    #              the all-clear stub, and the answer is the Dunion column's own clear-sky
    #              cooling profile.
    #      allsky  the same solver with the all-sky method. From S3b this is a REAL cloudy
    #              arm: `Scythe.radiation_assemble!` runs the S3a microphysics mapping
    #              (`cloud_optics_column!`) on every column, so the liquid cloud of the
    #              warm arm and the ISHMAEL anvil of the ice arm are both seen by the
    #              radiation. (Until S3a landed it was the all-clear stub and differed
    #              from `lw` only in which kernels ran.)
    #      sw      clear-sky with a FIXED sun (`:solar = :fixed`, cos_z 0.2588 / beam
    #              551.58 => 142.7 W/m^2 daily-mean insolation).
    #      diurnal clear-sky with the full diurnal cycle (`:solar = :diurnal`) at
    #              latitude 20 N, day 240, starting at 12:00 local — i.e. local noon, so
    #              the one hour this benchmark integrates sits on the peak of the cycle
    #              where the per-step zenith rescale is measurable.
    #
    #    SCYTHE_O01_RAD_FORCING = full (default) | anomaly. `:anomaly` holds
    #    `q - q̄(z)`, the horizontal mean over the patch captured on the FIRST call, so on
    #    the horizontally homogeneous sounding the forcing starts at machine zero and only
    #    the bubble's own departure from the mean is felt.
    #
    #    SCYTHE_O01_RAD_INTERVAL is the cadence in SECONDS (default 300, i.e. 12 calls in
    #    the 3600 s run); SCYTHE_O01_RAD_ZMAX tapers the heating to zero below that height
    #    (unset => Inf => no taper; 17000 would confine it below the DK83 sponge).
    #
    #    SCYTHE_O01_RAD_RAIN = 1 puts the RAIN mass into the liquid water path
    #    (`options[:radiation_rain_in_cloud]`, default false). Off is the physical default:
    #    rain's per-mass extinction is far below cloud's and its size is far above the
    #    21.5 um edge of the liquid table, so including it both over-states the optical
    #    depth and saturates the effective-radius clamp. The knob exists so the difference
    #    can be MEASURED on this benchmark rather than argued about.
    #
    #    The radiative SURFACE temperature is left UNSET on purpose: O01 has no ocean, so
    #    `Scythe.radiation_surface_temperature` falls through `:T_sfc` and `:SST` to the
    #    column's own hydrostatically extrapolated surface AIR temperature, which is
    #    radiatively consistent with the sounding and introduces no air-sea
    #    disequilibrium the run was never given. The line printed below says so.
    rad_arm = get(ENV, "SCYTHE_O01_RAD", "0")
    if rad_arm != "0" && rad_arm != ""
        options[:radiation] = :rrtmgp
        if rad_arm == "lw"
            options[:radiation_method] = :clearsky
            options[:solar] = :none
        elseif rad_arm == "allsky"
            options[:radiation_method] = :allsky
            options[:solar] = :none
        elseif rad_arm == "sw"
            options[:radiation_method] = :clearsky
            options[:solar] = :fixed
        elseif rad_arm == "allsky_sw"
            # S4 D5: the cloudy arm WITH a sun. Cloud-top shortwave absorption partly
            # offsets the longwave cooling there, and the shadow under the cloud shows up
            # in the surface downward flux -- neither is visible on `allsky` (no sun) or
            # on `sw` (no cloud), which is why this is its own arm rather than a flag.
            options[:radiation_method] = :allsky
            options[:solar] = :fixed
        elseif rad_arm == "diurnal"
            options[:radiation_method] = :clearsky
            options[:solar] = :diurnal
            physical_params[:latitude] = 20.0
            physical_params[:start_doy] = 240.0
            physical_params[:start_hour] = 12.0
            # The per-step zenith rescale is the mechanism this arm exists to exercise,
            # and the per-call trace cannot see it (it prints only at the cadence), so
            # the coarse per-step shortwave line is on by default HERE and nowhere else.
            options[:radiation_trace_sw] = true
        else
            error("SCYTHE_O01_RAD = \"$rad_arm\" is not an arm; use lw, allsky, sw, " *
                  "allsky_sw, diurnal, or 0/unset for no radiation at all")
        end
        options[:radiation_forcing] = Symbol(get(ENV, "SCYTHE_O01_RAD_FORCING", "full"))
        options[:radiation_interval] =
            parse(Float64, get(ENV, "SCYTHE_O01_RAD_INTERVAL", "300.0"))
        haskey(ENV, "SCYTHE_O01_RAD_ZMAX") &&
            (options[:radiation_z_max] = parse(Float64, ENV["SCYTHE_O01_RAD_ZMAX"]))
        get(ENV, "SCYTHE_O01_RAD_RAIN", "0") == "1" &&
            (options[:radiation_rain_in_cloud] = true)
        # S4 D4: override the FIXED sun's geometry, so a `:fixed` arm can be run at
        # exactly the `cos_zenith`/`toa_flux` a `:diurnal` arm resolves at its first call
        # and the two shortwave answers compared directly. Both knobs are needed for that
        # comparison: `solar_geometry` returns the beam-normal flux
        # `solar_constant * (1 + 0.033 cos(2 pi doy/365.25))`, which is NOT the 551.58
        # default. They do nothing on the `:none`/`:diurnal` arms (`solar_state` reads
        # `:cos_zenith`/`:sw_toa_flux` only under `:fixed`), and setting one without the
        # other is a half-matched comparison, so both are read here together.
        haskey(ENV, "SCYTHE_O01_RAD_COSZ") &&
            (physical_params[:cos_zenith] = parse(Float64, ENV["SCYTHE_O01_RAD_COSZ"]))
        haskey(ENV, "SCYTHE_O01_RAD_TOA") &&
            (physical_params[:sw_toa_flux] = parse(Float64, ENV["SCYTHE_O01_RAD_TOA"]))
        get(ENV, "SCYTHE_O01_RAD_TRACE_SW", "0") == "1" &&
            (options[:radiation_trace_sw] = true)
        # One line, on the MASTER's console (the driver's own setup runs on a worker and
        # its output goes to <output_dir>/scythe_err.log): which of the three surface
        # temperatures `radiation_surface_temperature` will actually use.
        tsfc_src = haskey(physical_params, :T_sfc) ?
            "physical_params[:T_sfc] = $(physical_params[:T_sfc]) K" :
            haskey(physical_params, :SST) ?
            "physical_params[:SST] = $(physical_params[:SST]) K" :
            "the column's own extrapolated surface AIR temperature T_face[1] (no ocean)"
        println("O01 radiation: arm=$rad_arm method=$(options[:radiation_method]) " *
                "solar=$(options[:solar]) forcing=$(options[:radiation_forcing]) " *
                "interval=$(options[:radiation_interval]) s " *
                "z_max=$(get(options, :radiation_z_max, Inf)) m " *
                "rain_in_cloud=$(get(options, :radiation_rain_in_cloud, false)); " *
                "surface temperature source = $tsfc_src")
    end

    # Slot names, now that the transforms are known. Everything keyed by NAME below — the BC
    # dicts, l_q, positivity and vars itself — must use these, because a stale key is ignored
    # silently rather than raising (`Scythe.check_mc_var_names` is the backstop). `vars`
    # already carries the APPENDED n_r slot when RAIN_MOMENTS=2, so `scalar_bc` and every
    # dict merged off it pick it up with no further edits.
    vars = Scythe.mc_var_names(options)
    rain_name = Scythe.rain_var_name(options)
    rain_number_name = Scythe.rain_number_var_name(options)
    two_moment = Scythe.rain_moments(options) == 2
    cloud_name = Scythe.condensate_var_name(options)
    # The twelve ice slot names in registration order (mass, number, a, c per species), under
    # whatever ICETRANS declares. Empty of meaning unless ice is on; `ice_on` is the gate.
    ice_on = Scythe.ice_microphysics(options) === :ishmael
    # (Positionally, entries 1, 5, 9 are the three MASS slots -- the only ice slots whose flux
    # reaches rho_t and E_t. Everything below applies to all twelve alike.)
    ice_names = Scythe.ice_var_names(options)

    output_dir = benchmark_output_dir("o01_rainfall", opts)
    scalar_bc = Dict(v => NeumannBC() for v in vars)
    # Side walls: no normal flow (reflective); domain is wide enough that the
    # bubble's gravity waves arrive late.
    side_bc = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))
    # Top/bottom: rigid lids (w = 0), free-slip u. rho_r takes a FREE (Natural)
    # fit at both: a Neumann fit would force a zero boundary flux derivative and
    # trap the falling rain at the surface instead of letting the sedimentation
    # flux divergence remove it through z = 0.
    # The rain NUMBER takes the same free fit as the rain mass, for the same reason: the
    # number flux divergence has to be able to carry drops out through z = 0 with the water
    # they hold, or the count piles up at the ground while its mass leaves.
    topbot_bc = merge(scalar_bc, Dict("w" => DirichletBC(), rain_name => NaturalBC()))
    two_moment && (topbot_bc[rain_number_name] = NaturalBC())
    # Every ice slot takes the same free (Natural) fit, which is the rain mass's rule and the
    # rain number's. For the three MASS slots it is load-bearing: their flux divergence is the
    # only ice term that sources rho_t and E_t, so once the fall speeds are live it has to be
    # able to carry ice out through z = 0 rather than piling it up at the ground with its mass
    # still in the total-water budget. For the number and volume moments it is the n_r
    # argument -- a Neumann fit would force zero flux derivative and leave the crystal count
    # (and the volume it occupies) at the surface after its mass had left, which is a worse
    # state than either.
    if ice_on
        for nm in ice_names
            topbot_bc[nm] = NaturalBC()
        end
    end

    # Attribution: sweep the cubic-spline filter length only on the water species.
    # Unset => Dict("default" => 2.0), which equals the struct default, so bit-identical.
    lq = merge(Dict("default" => 2.0),
               haskey(ENV, "SCYTHE_O01_LQ") ?
                   let x = parse(Float64, ENV["SCYTHE_O01_LQ"])
                       # The rain number takes the rain mass's filter length: the two are
                       # fitted on the same spike and a different low-pass on each would
                       # change the mean drop size at every wavenumber it separated them by.
                       d = Dict(rain_name => x, cloud_name => x)
                       two_moment && (d[rain_number_name] = x)
                       # The ice slots take the water filter length too, and all twelve take
                       # the SAME one: a species' four moments are fitted on the same spike
                       # and separating them by wavenumber would change the recovered
                       # aspect ratio phi = c/a and effective density at every scale the
                       # filters differed on.
                       if ice_on
                           for nm in ice_names
                               d[nm] = x
                           end
                       end
                       d
                   end : Dict{String,Float64}())

    # Positivity of the rain density, imposed as a box constraint on the spline coefficients
    # in BOTH directions (Springsteel `SAtransform_bounded!`). rho_r is a TOTAL, so the bound
    # is 0 on each leg; its side/top/bottom BCs are Natural or Neumann, both of which leave
    # the conservative clip-and-shrink exact.
    #
    # Both legs, not just :k. The k-leg alone gives positivity (it is the last leg), but the
    # negative lobes are broad HORIZONTAL flanks, so whole columns can consist of nothing but
    # spurious negative rain — and no non-negative spline has negative mass, leaving the
    # k-leg no conservative fix. Bounding :i first removes those lobes where the donor mass
    # actually is and guarantees every k-column arrives with non-negative mass. See the
    # MULTI-DIMENSIONAL DESIGN note in Springsteel's CubicBSpline.jl and
    # reference/FINDINGS_NEGATIVE_WATER_ATTRIBUTION.md.
    # rho_c is carried as a PERTURBATION from rho_cbar, so its bound is -ρ̄_c, not 0. This
    # sounding has no reference cloud (write_exact_ref_mc gets zeros(kDim) below), so the two
    # coincide here; install_positivity_bounds! substitutes the reference-offset bound
    # automatically on any configuration whose base state IS cloudy.
    # DEFAULT IS RAIN ONLY. Bounding rho_c as well destabilizes this run — max_w 2.7 -> 36
    # at t = 750 s with either leg alone, 78 with both. The cause is upstream of the limiter
    # and is now measured (STAGE 3 of reference/FINDINGS_NEGATIVE_WATER_ATTRIBUTION.md): the
    # cloud EVAPORATION limiter is a forward-Euler budget (`Qdot_c >= -rho_c/ts`) but the
    # integrator is AB3 with a leading weight of 23/12, so 185-281 gridpoints per step sit on
    # that cap and are then depleted by 2.3-2.8x the cloud present — ~150 points per step
    # driven negative, forever. rho_r is exempt only because rain's sinks NEVER reach their
    # cap (0 points, every run), which is the whole rho_r/rho_c asymmetry. The limiter then
    # has to act at full strength every step, a floor in all but name.
    #
    # Two corrections to the Stage 2 note this replaces: `bound_shortfall` does NOT stay at
    # 0 under mode `1` (the rho_c i-leg reaches 7.9e6; the old report summed only the k-basis),
    # and the effect is INVARIANT under ts (2.339 at ts=0.075 vs 2.345 at ts=0.3), so it was
    # never a CFL/ceiling problem.
    #
    # STAGE 3b/4 (HISTORICAL — the levers named here no longer exist): with the cap made
    # AB3-sized, mode `1` ran the full 3600 s with min_rho_c = 0 and max_rho_r_gm3
    # 13.12 FAIL -> 9.63 PASS, but the error relocated into the RESIDUAL vapor (min_rho_v
    # -0.72 -> -1.85 g/m^3) and max_w ran to 29.6. The depletion caps have since been removed
    # outright (see Scythe.qss_condensation_rates), so the paragraph above describes a
    # configuration the code can no longer be put in; keep the default rain-only.
    #
    # Modes: 0 = off, k = rain vertical leg only, r = rain both legs (the default),
    # ck / ci = rain both legs plus cloud vertical / horizontal, 1 = both species both legs.
    positivity = let mode = get(ENV, "SCYTHE_O01_POSITIVITY", "r")
        mode == "0" ? Dict{String,Dict{Symbol,Float64}}() :
        mode == "k" ? Dict("rho_r" => Dict(:k => 0.0)) :
        mode == "ck" ? Dict("rho_r" => Dict(:i => 0.0, :k => 0.0),
                            "rho_c" => Dict(:k => 0.0)) :
        mode == "ci" ? Dict("rho_r" => Dict(:i => 0.0, :k => 0.0),
                            "rho_c" => Dict(:i => 0.0)) :
        mode == "1" ? Dict("rho_r" => Dict(:i => 0.0, :k => 0.0),
                           "rho_c" => Dict(:i => 0.0, :k => 0.0)) :
                      Dict("rho_r" => Dict(:i => 0.0, :k => 0.0))
    end
    # A transformed species may not also be bounded — the two impose different constraints
    # and declaring both is refused by `install_positivity_bounds!`. POSITIVITY defaults to
    # "r", so a transformed rain arm would otherwise need a second env var set in lockstep;
    # drop it here instead and SAY SO, rather than resolve it silently. Nothing is dropped
    # when no transform is declared, so the default configuration is bit-identical.
    for (name, transformed) in (("rho_c", cloud_name != "rho_c"),
                                ("rho_r", rain_name != "rho_r"))
        (transformed && haskey(positivity, name)) || continue
        delete!(positivity, name)
        println("POSITIVITY: dropped \"$name\" — it is carried as a control variable " *
                "(transform on), which a coefficient bound cannot constrain")
    end

    grid_params = GridParameters(;
        geometry = benchmark_geometry(opts),   # RZ (Chebyshev) or RiRk (B-spline) vertical
        iMin = 0.0,
        iMax = 150.0e3,
        num_cells_i = num_cells_i,
        kMin = 0.0,
        kMax = 25.0e3,
        vertical_size(opts; num_cells_k = num_cells_k, kDim = kDim)...,
        l_q = lq,
        positivity = positivity,
        BCL = side_bc,
        BCR = side_bc,
        BCB = topbot_bc,
        BCT = topbot_bc,
        vars = Dict(v => i for (i, v) in enumerate(vars)),
    )

    return ModelParameters(
        ts = ts,
        # Attribution: shorten the run (negative water appears once cloud/rain form,
        # ~24 min onset in quick mode). Unset => the full 3600 s, bit-identical.
        integration_time = haskey(ENV, "SCYTHE_O01_TSTOP") ?
                           parse(Float64, ENV["SCYTHE_O01_TSTOP"]) : 3600.0,
        output_interval = output_interval,
        equation_set = "moist_compressible_XZ",
        initial_conditions = joinpath(output_dir, "o01_ics.csv"),
        output_dir = output_dir,
        ref_state_file = joinpath(output_dir, "o01_exact.ref"),
        grid_params = grid_params,
        physical_params = physical_params,
        options = options,
    )
end

# ── Initial conditions ─────────────────────────────────────────────────────

"""
Balance the Dunion sounding hydrostatically on the model column, write it as the
run's exact pressure-based reference (one file for init, integration and
diagnostics), and add the RH-preserving 3 K warm bubble.
"""
function o01_init!(model)
    # The output directory is shared across runs of this variant and the rain
    # diagnostics sweep EVERY snapshot in it — stale files from a previous (e.g.
    # longer) run would silently contaminate the onset/accumulation numbers.
    for f in readdir(model.output_dir)
        endswith(f, "_physical.csv") && rm(joinpath(model.output_dir, f))
    end

    patch = createGrid(model.grid_params)
    gridpoints = Scythe.getGridpoints(patch)
    kDim = model.grid_params.kDim
    z = gridpoints[1:kDim, 2]
    column = Scythe.reference_column(patch, model.grid_params)

    # The .ref file carries VALUES only, so under :hydrostatic_reference the CONVERGED
    # (p, rho_d, rho_v) triple has to be written for the balance to survive the round
    # trip -- exact_pressure_reference_state re-integrates dp/dz = -g*rho_t from it.
    hydro = reference_state_hydrostatic()
    ref_phys = Springsteel.calculate_pressure_reference_state(DUNION_SOUNDING, z, column;
                                                             hydrostatic = hydro)
    pbar = Springsteel.ref_pressure(ref_phys)[:, 1]
    rho_dbar = Springsteel.ref_rho_d(ref_phys)[:, 1]
    rho_vbar = Springsteel.ref_rho_v(ref_phys)[:, 1]
    Scythe.write_exact_ref_mc(model.ref_state_file, z, pbar, rho_dbar, rho_vbar,
                              zeros(kDim))
    ref = Springsteel.exact_pressure_reference_state(model.ref_state_file, z, column;
                                                     hydrostatic = hydro)
    # The ICs are stored as perturbations from Q̄_ss, so the bubble must be differenced
    # against the SAME Q̄_ss createModelTile will add back.
    if get(reference_state_options(), :consistent_qss_reference, false)
        ref = Scythe.consistent_qss_reference(ref, z, column)
    end

    # Hydrostatic sanity: residual of the balanced reference on its own column
    rho_tbar = Springsteel.ref_rho_t(ref)[:, 1]
    pbar_z = Springsteel.ref_pressure(ref)[:, 2]
    residual = pbar_z .+ (Scythe.gravity .* rho_tbar)
    println("Reference: sfc p = $(round(pbar[1] / 100.0, digits=2)) hPa, ",
            "max hydrostatic residual = $(maximum(abs.(residual))) Pa/m")

    patch.physical .= 0.0
    Scythe.moist_temperature_bubble_mc!(patch, gridpoints, ref;
                                        xc = 75.0e3, xr = 16.0e3,
                                        zc = 500.0, zr = 3000.0, dT_max = 3.0)
    Scythe.write_ics_csv(model.initial_conditions, patch, gridpoints)
end

# ── Diagnostics ─────────────────────────────────────────────────────────────

# Snapshot enumeration, water path and the rain / ice / radiation diagnostic sets live in
# benchmarks/common/warm_bubble_diagnostics.jl (shared with ocean_warm_bubble.jl).
include(joinpath(@__DIR__, "common", "warm_bubble_diagnostics.jl"))

function o01_diagnostics(model)
    df = read_final_output(model)
    ref, _, kDim = rebuild_reference(model)

    diags = merge(o01_rain_diagnostics(model, ref, kDim),
                  o01_ice_diagnostics(model, ref, kDim),
                  o01_radiation_diagnostics(model, ref, kDim))
    diags["max_w"] = maximum(df.w)
    diags["min_w"] = minimum(df.w)

    # Exact water budget: the surface outflow is the only sink of domain water, so
    # the accumulated rainfall IS the water-path difference (independent of the
    # 60 s flux sampling above).
    snaps = output_snapshots(model)
    wp0 = water_path_mm(snaps[1][2], model, ref, kDim)
    wpN = water_path_mm(snaps[end][2], model, ref, kDim)
    diags["accum_rainfall_mm"] = wp0 - wpN

    drift = conservation_drift(model, ref)
    # Energy closure: rain leaving through the surface CHANGES the domain E_t by the
    # boundary flux (positive: e_l < 0 leaves, so E_t rises). Compare the raw drift
    # against the predicted precipitation gain; the residual is the actual
    # conservation error of the run.
    E0 = begin
        df0 = CSV.read(snaps[1][2], DataFrame)
        ncols = div(nrow(df0), kDim)
        E_t = df0.E_t .+ repeat(Springsteel.ref_total_energy(ref)[:, 1], ncols)
        domain_integral(reshape(E_t, kDim, ncols), model)
    end
    width = model.grid_params.iMax - model.grid_params.iMin
    predicted_gain_pct = 100.0 * diags["precip_energy_gain_Jm2"] * width / E0
    diags["energy_residual_pct"] = drift["energy_drift_pct"] - predicted_gain_pct

    return merge(diags, drift)
end

# ── Nested (3-level, 5-patch) configuration ─────────────────────────────────
#
# Grid nesting per DeMaria et al. (1992)/Ooyama (2001): fine center patch over
# the bubble, 2x-coarser abutting patches outward, two-way coupling (R3X trio
# down, collar tendency injection up), per-patch timesteps. The vertical grid
# is identical everywhere. Coarse-patch timesteps are capped at 0.6 s: the
# moist scheme's stability ceiling is set by the vertical acoustic Courant
# (c*ts/dz_min ≈ 1.8 at 0.6 s; a single-grid isolation run at ts = 1.2
# reproduced the blow-up with no nesting involved), so the outermost patches
# gain less than the full 2x-per-level — the nested speedup comes mostly from
# the column-count reduction.

"""Nest layout for the configured mode, wrapping the single-grid model as base."""
function o01_nest(opts::BenchmarkOptions)
    base = o01_model(opts)
    if opts.mode == :full
        #boundaries = [0.0, 50.0e3, 63.0e3, 87.0e3, 100.0e3, 150.0e3]
        #num_cells = [25, 13, 48, 13, 25]           # 2 | 1 | 0.5 | 1 | 2 km cells
        #ts = [0.6, 0.3, 0.15, 0.3, 0.6]
        boundaries = [0.0, 50.0e3, 64.0e3, 86.0e3, 100.0e3, 150.0e3]
        num_cells = [50, 28, 88, 28, 50]           # 1 | 0.5 | 0.25 | 0.5 | 1 km cells
        ts = [0.3, 0.15, 0.075, 0.15, 0.3]
    else
        boundaries = [0.0, 48.0e3, 64.0e3, 86.0e3, 102.0e3, 150.0e3]
        num_cells = [6, 4, 11, 4, 6]               # 8 | 4 | 2 | 4 | 8 km cells
        ts = [0.6, 0.6, 0.3, 0.6, 0.6]
    end
    ts = [vertical_ts(t, opts) for t in ts]
    return NestedModelParameters(
        boundaries = boundaries,
        num_cells = num_cells,
        ts = ts,
        workers_per_patch = nested_workers(opts),
        base = base)
end

"""Nominal (collar-excluded) x-bounds of nest patch `i`."""
function o01_nominal_bounds(models, topo, i)
    xlo = models[i].grid_params.iMin
    xhi = models[i].grid_params.iMax
    for k in topo.child_ifaces[i]
        ni = topo.interfaces[k]
        if ni.parent_side == :right
            xhi = ni.interface_x
        else
            xlo = ni.interface_x
        end
    end
    return xlo, xhi
end

"""Shared reference + per-patch RH-preserving bubble ICs (nested o01_init!)."""
function o01_init_nested!(models, topo)
    for m in models
        mkpath(m.output_dir)
        for f in readdir(m.output_dir)
            endswith(f, "_physical.csv") && rm(joinpath(m.output_dir, f))
        end
    end

    # The reference depends on z only; build it once and share the file.
    m1 = models[1]
    patch = createGrid(m1.grid_params)
    gridpoints = Scythe.getGridpoints(patch)
    kDim = m1.grid_params.kDim
    z = gridpoints[1:kDim, 2]
    column = Scythe.reference_column(patch, m1.grid_params)
    ref_phys = Springsteel.calculate_pressure_reference_state(DUNION_SOUNDING, z, column)
    Scythe.write_exact_ref_mc(m1.ref_state_file, z,
                              Springsteel.ref_pressure(ref_phys)[:, 1],
                              Springsteel.ref_rho_d(ref_phys)[:, 1],
                              Springsteel.ref_rho_v(ref_phys)[:, 1],
                              zeros(kDim))
    ref = Springsteel.exact_pressure_reference_state(m1.ref_state_file, z, column)
    rho_tbar = Springsteel.ref_rho_t(ref)[:, 1]
    pbar_z = Springsteel.ref_pressure(ref)[:, 2]
    residual = pbar_z .+ (Scythe.gravity .* rho_tbar)
    println("Reference: sfc p = $(round(Springsteel.ref_pressure(ref)[1, 1] / 100.0, digits=2)) hPa, ",
            "max hydrostatic residual = $(maximum(abs.(residual))) Pa/m")

    for m in models
        p = createGrid(m.grid_params)
        gpts = Scythe.getGridpoints(p)
        p.physical .= 0.0
        # Slots 8 and 9 hold whatever control variable the run declares; the initializer
        # writes through `rain_slot`/`condensate_slot`. This bubble is condensate-free on a
        # condensate-free reference, so every convention agrees on exactly 0.0 — threaded
        # anyway so the coupling is visible rather than a coincidence.
        Scythe.moist_temperature_bubble_mc!(p, gpts, ref;
                                            xc = 75.0e3, xr = 16.0e3,
                                            zc = 500.0, zr = 3000.0, dT_max = 3.0,
                                            condensate_transform =
                                                Scythe.condensate_transform_mode(m.options),
                                            condensate_mu =
                                                get(m.physical_params, :condensate_mu, 1.0e-7),
                                            rain_transform =
                                                Scythe.rain_transform_mode(m.options),
                                            rain_mu = get(m.physical_params, :rain_mu, 1.0e-7))
        Scythe.write_ics_csv(m.initial_conditions, p, gpts)
    end
end

"""
    o01_nest_shortfall!(diags, models)

Record each patch's accumulated k-leg positivity `bound_shortfall` — the mass the limiter
had to CREATE because a whole spline column arrived with negative total mass, which no
non-negative spline can represent. It must be watched on a nested run specifically: the child
patches carry the vertical bound only (their i-BC is R3X, which `set_lower_bound!` rejects —
see `build_nest`), and the i-leg is the one that removes the broad horizontal negative lobes
before the k-leg ever sees them. A nonzero value is the leak that k-only bounding admits.

Read from the live workers rather than the output CSVs: the shortfall lives on
`mtile.tile.kbasis`, one accumulator per worker tile (the k-leg bites on the tile, the i-leg
on the worker's patch — see `water_budget_trace`). Workers are matched to patches by the
`output_dir` their model carries, so no group bookkeeping has to be threaded through the
harness. Diagnostic only: no target, no gate.
"""
function o01_nest_shortfall!(diags, models)
    n = length(models)
    nest_of = Dict(models[i].output_dir => i for i in 1:n)
    bounded = sort(unique(String[name for m in models
                                 for (name, spec) in m.grid_params.positivity
                                 if haskey(spec, :k)]))
    isempty(bounded) && return diags
    totals = Dict{Tuple{String,Int},Float64}((name, i) => 0.0 for name in bounded, i in 1:n)
    for w in sort(workers())
        # Guarded: this runs after a multi-hour integration, and a worker that somehow holds
        # no tile must not take the whole diagnostic down with it — it is reported and
        # skipped (leaving that patch's total short, which the warning says).
        res = try
            @fetchfrom w begin
                mt = Main.mtile
                vars = mt.model.grid_params.vars
                kb = mt.tile.kbasis
                (mt.model.output_dir,
                 Dict{String,Float64}(
                     name => (kb isa Springsteel.NoBasisArray ? NaN :
                              Springsteel.CubicBSpline.bound_shortfall(kb.data[vars[name]]))
                     for (name, spec) in mt.model.grid_params.positivity if haskey(spec, :k)))
            end
        catch err
            @warn "bound_shortfall: worker $w could not be read; its patch's total is " *
                  "incomplete" exception = err
            continue
        end
        odir, sf = res
        i = get(nest_of, odir, 0)
        i == 0 && continue
        for (name, v) in sf
            haskey(totals, (name, i)) && (totals[(name, i)] += v)
        end
    end
    total = 0.0
    for name in bounded, i in 1:n
        haskey(models[i].grid_params.positivity, name) || continue
        v = totals[(name, i)]
        diags["bound_shortfall_$(name)_n$i"] = v
        total += v
    end
    diags["bound_shortfall_total"] = total
    return diags
end

"""
Nested rain + budget diagnostics: per-patch surface-rain series restricted to
each patch's nominal region (collar cells excluded so abutting patches
partition the domain exactly), summed/maxed across patches; masked domain
integrals for the water and energy budgets, and the per-patch k-leg positivity
shortfall (see `o01_nest_shortfall!`).
"""
function o01_nested_diagnostics(models, topo)
    ref, _, kDim = rebuild_reference(models[1])
    ctf = Scythe.condensate_transform_mode(models[1].options)
    cmu = get(models[1].physical_params, :condensate_mu, 1.0e-7)
    rtf = Scythe.rain_transform_mode(models[1].options)
    rmu = get(models[1].physical_params, :rain_mu, 1.0e-7)
    n = length(models)
    masks = [nominal_col_mask(models[i], o01_nominal_bounds(models, topo, i)...)
             for i in 1:n]
    total_width = models[n].grid_params.iMax - models[1].grid_params.iMin

    snaps = [output_snapshots(m) for m in models]
    ntimes = minimum(length.(snaps))
    times = [snaps[1][j][1] for j in 1:ntimes]

    peak_rate = 0.0
    onset = NaN
    max_rr = 0.0
    min_rr = 0.0
    max_rc = 0.0
    min_rc = 0.0
    min_rv = Inf
    min_rw = Inf
    min_rd_frac = Inf                         # see o01_rain_diagnostics for what this is for
    # w extrema over the WHOLE run (user decision 2026-07-14): the nested max_w
    # is defined as the run maximum, not the final-time value the single-grid
    # diagnostic reports. By the end of the hour the secondary cells have
    # propagated into the coarser outer nests where their intensity is
    # resolution-limited, so a final-time sample measures the outer-nest
    # resolution rather than the convection the benchmark targets; the run
    # maximum (reached in the fine nest) is the comparable quantity.
    max_w = -Inf
    min_w = Inf
    rate_int = zeros(ntimes)
    eflux_int = zeros(ntimes)
    Whs = [Float64[] for _ in 1:n]            # masked weights, filled lazily
    for j in 1:ntimes
        pk_t = 0.0
        for i in 1:n
            t, path = snaps[i][j]
            df = CSV.read(path, DataFrame)
            gp = models[i].grid_params
            ncols = div(nrow(df), kDim)
            surf = 1:kDim:nrow(df)
            Tk, _, rho_d, rho_v, rho_c, rho_t, rho_r =
                mc_state(df, ref, kDim, ncols; transform = ctf, mu = cmu,
                         rain_transform = rtf, rain_mu = rmu)
            mask = masks[i]
            colmask = repeat(mask, inner = kDim)
            max_rr = max(max_rr, maximum(rho_r[colmask]))
            min_rr = min(min_rr, minimum(rho_r[colmask]))
            max_rc = max(max_rc, maximum(rho_c[colmask]))
            min_rc = min(min_rc, minimum(rho_c[colmask]))
            min_rv = min(min_rv, minimum(rho_v[colmask]))
            min_rw = min(min_rw, minimum((rho_t .- rho_d)[colmask]))
            min_rd_frac = min(min_rd_frac,
                              minimum((rho_d ./
                                       repeat(Springsteel.ref_rho_d(ref)[:, 1],
                                              ncols))[colmask]))
            max_w = max(max_w, maximum(df.w[colmask]))
            min_w = min(min_w, minimum(df.w[colmask]))
            rr_s = max.(rho_r[surf], 0.0) .* mask
            Vt = Scythe.rain_terminal_velocity.(rr_s, rho_d[surf], Tk[surf])
            R = -rr_s .* Vt
            pk_t = max(pk_t, maximum(R))
            if isempty(Whs[i])
                Whs[i] = gauss_cell_weights(ncols, gp.num_cells, gp.iMax - gp.iMin,
                                            ncols ÷ gp.num_cells, gp.quadrature) .* mask
            end
            e_l = (Scythe.Cpv .* Tk[surf]) .- Scythe.L_v.(Tk[surf])
            rate_int[j] += sum(Whs[i] .* R)
            eflux_int[j] += sum(Whs[i] .* (rr_s .* Vt .* e_l))    # F_E(z≈0), > 0 for e_l < 0
        end
        peak_rate = max(peak_rate, pk_t)
        if isnan(onset) && pk_t > 1.0e-3
            onset = times[j]
        end
    end
    accum_flux = 0.0
    accum_E = 0.0
    for j in 1:(ntimes - 1)
        dt = times[j+1] - times[j]
        accum_flux += 0.5 * (rate_int[j] + rate_int[j+1]) * dt
        accum_E += 0.5 * (eflux_int[j] + eflux_int[j+1]) * dt
    end

    diags = Dict(
        "peak_rain_rate_gm2s" => 1000.0 * peak_rate,
        "rain_onset_min" => onset / 60.0,
        "max_rho_r_gm3" => 1000.0 * max_rr,
        "min_rho_r_gm3" => 1000.0 * min_rr,
        "max_rho_c_gm3" => 1000.0 * max_rc,
        "min_rho_c_gm3" => 1000.0 * min_rc,
        "min_rho_v_gm3" => 1000.0 * min_rv,
        "min_rho_w_gm3" => 1000.0 * min_rw,
        "min_rho_d_frac" => min_rd_frac,
        "accum_rainfall_flux_mm" => accum_flux / total_width,
        "precip_energy_gain_Jm2" => accum_E / total_width,
    )

    # Masked water/energy budgets across the nest
    rho_wbar = Springsteel.ref_rho_t(ref)[:, 1] .- Springsteel.ref_rho_d(ref)[:, 1]
    E_tbar = Springsteel.ref_total_energy(ref)[:, 1]
    function nest_budget(j)
        W = 0.0
        E = 0.0
        for i in 1:n
            df = CSV.read(snaps[i][j][2], DataFrame)
            ncols = div(nrow(df), kDim)
            rho_w = (df.rho_t .- df.rho_d) .+ repeat(rho_wbar, ncols)
            E_t = df.E_t .+ repeat(E_tbar, ncols)
            W += domain_integral(reshape(rho_w, kDim, ncols), models[i], masks[i])
            E += domain_integral(reshape(E_t, kDim, ncols), models[i], masks[i])
        end
        return W, E
    end
    W0, E0 = nest_budget(1)
    WN, EN = nest_budget(ntimes)
    diags["accum_rainfall_mm"] = (W0 - WN) / total_width
    diags["energy_drift_pct"] = 100.0 * (EN - E0) / E0
    predicted_gain_pct = 100.0 * diags["precip_energy_gain_Jm2"] * total_width / E0
    diags["energy_residual_pct"] = diags["energy_drift_pct"] - predicted_gain_pct

    diags["max_w"] = max_w
    diags["min_w"] = min_w

    o01_nest_shortfall!(diags, models)

    return diags
end

# ── Figures ────────────────────────────────────────────────────────────────

plotter = nothing
if opts.plot
    include(joinpath(@__DIR__, "common", "plots.jl"))
    plotter = function (model)
        df = read_final_output(model)
        ref, _, kDim = rebuild_reference(model)
        ncols = div(nrow(df), kDim)
        x = reshape(df.r, kDim, ncols)[1, :]
        z = reshape(df.z, kDim, ncols)[:, 1]
        w = reshape(df.w, kDim, ncols)
        # The plotted rain is the RECOVERED density, whatever slot 8 carries.
        rho_r = reshape(Scythe.recover_rho_r.(first(water_column(df, "rho_r")),
                                              Scythe.rain_transform_mode(model.options),
                                              get(model.physical_params, :rain_mu, 1.0e-7)) .*
                        1000.0, kDim, ncols)
        save_benchmark_figure(
            joinpath(model.output_dir, "o01_rainfall_$(opts.mode)_$(opts.stage)_final.png"),
            x, z,
            [(rho_r, "ρ_r (g/m³)", 0.0:0.25:4.0),
             (w, "w (m/s)", -12.0:2.0:24.0)];
            title = "O01 warm rain, t = $(model.integration_time) s")
    end
end

# ── Run ────────────────────────────────────────────────────────────────────


# Arm token, derived from the RESOLVED MODEL CONFIG -- never from SCYTHE_BENCH_TAG (a
# free-text output_dir label with no bearing on which reference/windows a run is checked
# against). The ISHMAEL ice arm carries 12 extra prognostic slots, a prognostic rain
# number and (by default, see the ice-arm production-config block above) rain/rain-number/
# ice control-variable transforms the warm default doesn't -- physically different output,
# so it gets its own committed reference and its own (currently unseeded) target windows
# via this token, rather than reading or overwriting the warm arm's. Empty for every other
# configuration, so the default path stays exactly what it always was.
if opts.nests == 1
    model = o01_model(opts)
    arm = Scythe.ice_microphysics(model.options) === :ishmael ? "ice" : ""
    passed = run_benchmark("o01_rainfall", opts;
                           model = model,
                           init! = o01_init!,
                           diagnostics = o01_diagnostics,
                           # The run's OWN slot names: a transform renames slots 8/9, and the
                           # regression then reports itself skipped rather than KeyError.
                           varnames = Scythe.mc_var_names(model.options),
                           plotter = plotter,
                           arm = arm)
else
    nest = o01_nest(opts)
    arm = Scythe.ice_microphysics(nest.base.options) === :ishmael ? "ice" : ""
    passed = run_nested_benchmark("o01_rainfall", opts;
                                  nest = nest,
                                  init! = o01_init_nested!,
                                  diagnostics = o01_nested_diagnostics,
                                  arm = arm)
end
exit(passed ? 0 : 1)
