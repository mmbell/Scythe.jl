# ── Tropical cyclone simulation parameters ──────────────────────────────────
# Every knob for the TC runs lives here; tc_init.jl and the run scripts
# include this file. Edit and rerun tc_init to regenerate the initial
# conditions.

# ── Vortex ──────────────────────────────────────────────────────────────────
# Rotunno & Emanuel (1987, JAS 44, 542-561) eq. (37), gradient-wind +
# hydrostatically balanced. See src/idealized.jl `re87_v`.
#
# WHY :re87 AND NOT MODIFIED RANKINE (2026-07-19). The modified-Rankine profile
# with alpha = 0.3 is not compact: v was still 12 m/s at r = 1050 km, the DOMAIN
# EDGE. With no true far field the thermal-wind warm anomaly integrates across the
# whole domain and reaches +6.91 K at the surface on the axis. The Dunion
# sounding/SST pairing offers only +0.52 K of air-sea disequilibrium, so that
# anomaly put the surface air ~4.8 K ABOVE the SST and REVERSED the surface
# enthalpy and moisture fluxes -- the ocean cooling and drying the boundary layer,
# the inverse of WISHE. The vortex could only decay.
#
# The previous fix for that was Z_BAROTROPIC = 2 km (v height-independent below,
# so no thermal-wind anomaly in the BL). It restored the flux sign but compressed
# the vortex decay into 13 km instead of 15, and the resulting |dv/dz| drove dry
# Ertel PV NEGATIVE over ~1350 work-grid points -- symmetrically unstable. The
# model then overturned it with an exponentially growing circulation that looked
# like a blow-up but was the physically correct response. A C2-smooth version was
# WORSE (higher peak shear), which is what proved shear magnitude was the driver
# rather than the piecewise-linear kink.
#
# Eq. (37)'s cubic falloff and EXACT zero at r_0 makes the vortex compact, so the
# warm anomaly is +1.56-1.91 K instead of +6.91 K, and the linear-in-z decay is
# PV-stable without any barotropic layer. Z_BAROTROPIC is retained only as a knob
# and MUST stay 0: it was the sole source of the symmetric instability.
const VORTEX_PROFILE = :re87
const V_M = 15.0             # [m/s] -> v_max ~ 12.9 m/s, matching RE87's ~12
const R_M = 82.5e3           # [m] radius of maximum wind (RE87 control value)
# Outer radius, WIDENED from RE87's 412.5 km. A compact vortex has strongly
# anticyclonic outer vorticity (zeta ~ -v_m/(r_0 - r_m)) and f + zeta goes negative
# when f is small. RE87 ran f = 5e-5 (20 N); F_COR here is 3.775e-5 (15 N), where
# r_0 = 412.5 km gives 4546 negative-PV points. Measured: r_0 = 800 km gives ZERO.
const R_0 = 800.0e3          # [m] v = 0 exactly at and beyond this radius
const V_TOP = 15.0e3         # [m] winds decay linearly to zero here
const Z_BAROTROPIC = 0.0     # [m] MUST be 0 -- see above; nonzero => unstable PV
# Legacy modified-Rankine knobs, used only when VORTEX_PROFILE = :rankine.
const VMAX = 30.0            # [m/s] maximum tangential wind
const RMW = 50.0e3           # [m] radius of maximum wind
const RANKINE_ALPHA = 0.3    # outer decay exponent, v = VMAX (RMW/r)^alpha

# Environment
const F_COR = 2.0 * 7.292e-5 * sind(15.0)   # [1/s] Coriolis at 15 N (3.775e-5)
# SST was lowered 28 -> 27 C to moderate the surface fluxes, then raised to 29.5 C
# (2026-07-19) once the flux SIGN turned out to be the real problem. The Dunion
# hum90 sounding's surface air is 299.63 K, so SST = 300.15 left only +0.52 K of
# environmental air-sea disequilibrium -- less than the vortex's own surface warm
# anomaly (+1.91 K with the RE87 profile), which flipped the flux sign at the axis.
# 302.65 K keeps SST - T_air POSITIVE at every radius with ~0.7 K of margin.
# (RE87 could run 26.3 C because their model-neutral Jordan sounding had
# correspondingly cooler surface air; ours is a different sounding.)
const SST_K = 302.65                        # [K] fixed sea surface temp (29.5 C)
const SOUNDING = joinpath(@__DIR__, "..", "benchmarks", "reference_data",
                          "o01_rainfall", "dunion_MT_hum90.ref")

# Inner-core moisture. The Dunion MT sounding is the OUTER-core / near-RCE
# environment; the initial perturbation makes the inner core much moister and
# closer to convectively neutral, as a real pre-depression with 15 m/s winds and
# ongoing convection would be. Without this the warm core is the DRIEST column in
# RH terms (holding q_v at its ambient profile while T rises), which loads the
# core with the most CIN and forces convection to wait for an explosive release.
# The repartition holds rho_t and p fixed, so it is exactly balance preserving.
# Measured at t=0 with model_tests/tc_init_diagnostics.jl: the CORE is the driest
# column, RH 0.745 at the surface and only 0.699 at 5 km. The knobs separate
# cleanly: RH_BL alone sets the surface parcel (hence CAPE and CIN), RH_CORE alone
# sets mid-level humidity at ZERO CAPE cost.
#
# RH_BL was first left OFF, on the reasoning that CIN was modest (52 J/kg at the
# axis) and that moistening the boundary layer nearly doubles CAPE (1536 -> 3115).
# The 8-h run at that setting was STABLE BUT COMPLETELY QUIESCENT -- max w reached
# only 0.05 m/s and no cloud or rain ever formed. The reasoning had missed the
# TRIGGER problem: an axisymmetric domain has no asymmetries to initiate
# convection, so parcels must be lifted mechanically to the LFC, and at RH_BL off
# the LFC sits at 1.95 km while the frictional secondary circulation supplies only
# ~2 cm/s. Surface fluxes were moistening the boundary layer correctly (RH 0.739 ->
# 0.776 over 7 h) but far too slowly to close that gap.
#
# RH_BL = 0.88 drops CIN to 12 J/kg and the LFC to 0.45 km, which the existing
# secondary circulation clears almost at once -- convection fires early and can
# adjust continuously instead of waiting for a large release. The higher initial
# CAPE is accepted deliberately: it is consumed from the start rather than
# accumulated. (The pre-fix runs' explosive release at 5-6 h was very likely
# triggered by the BROKEN initialization -- 40881 supersaturated points and
# spurious condensate are a large artificial trigger, which the hydrostatic fix
# removed along with the noise.)
const RH_CORE = 0.95         # free-troposphere target RH in the core
const RH_BL = 0.90           # boundary-layer target RH (sets CAPE/CIN and the LFC)
const Z_BL = 1.5e3           # [m] top of the boundary-layer target
const R_MOIST = 150.0e3      # [m] Gaussian radial e-folding of the moist core
const Z_MOIST = 10.0e3       # [m] cosine taper to the environment above this
const RH_INIT_MAX = 0.98     # hard cap: never initialize at/above saturation
# Radial shape of the initial moistening. :gaussian peaks ON THE AXIS, which puts
# maximum CAPE exactly where the 1/r geometry most easily triggers convection --
# measured 5861 J/kg at r=0 vs 3327 at 100 km. :vortex weights by the surface
# tangential wind instead (zero at the axis, peak at the RMW), co-locating the
# moisture with the strongest surface fluxes and leaving the eye dry so mass
# continuity can establish axis subsidence. Switch to :vortex if the axis column
# goes unstable. See src/idealized.jl for the full reasoning.
const MOIST_PROFILE = :vortex

# Surface exchange and boundary layer
const CK = 1.0e-3            # bulk enthalpy/moisture exchange coefficient
const CD = -1.0              # negative => wind-speed-dependent Komori et al. (2018)
const U_MIN = 2.0            # [m/s] gustiness floor on the exchange wind
const L_INF = 80.0           # [m] asymptotic Louis mixing length

# Horizontal turbulence (Smagorinsky) and microphysics.
# An axisymmetric TC has NO asymmetries to provide radial mixing, so the resolved
# radial gradients are damped only by this closure -- prior axisymmetric work
# (model_tests/Twoway_PV_mixing) used Ls_free = 500 m / Ls_bl = 2000 m with a
# K_min of 1000 m²/s in the boundary layer, far more than the 200 m / 5 m²/s this
# case had been running. KH_HEAT = -1.0 is the sentinel for "diffuse heat with the
# Smagorinsky K/Pr_t": before this, Smagorinsky mixed MOMENTUM ONLY and the
# thermodynamic fields had no horizontal mixing at all, leaving grid-scale
# buoyancy structure undamped on 3 km radial cells.
const LS_SMAG = 500.0        # [m] Smagorinsky length scale
const K_MIN = 25.0           # [m²/s] horizontal eddy viscosity floor
const KH_HEAT = -1.0         # < 0 => Smagorinsky K/Pr_t (not a constant K)
const PR_T = 1.0             # turbulent Prandtl number
# Horizontal water mixing. OFF (2026-07-31, user decision). It is a bare K*grad^2 --
# TEMPORARY / NOT ENERGY CONSISTENT (see the long note in moist_compressible.jl by
# slot 8): it diffuses water mass without transporting the energy that mass carries,
# so it WILL drift energy, and it is a very idealized turbulence closure that has to
# be replaced regardless. It was switched on as pure noise control, on the reasoning
# that the water species otherwise have no horizontal mixing at all; the spline filter
# (l_q = 2.0 by default on every variable here) already serves part of that role, and
# the negative-condensate reservoir that produced much of the grid-scale water
# structure is gone under the transforms below. Turn it back on ONLY if a gate run
# demonstrates that minimal diffusion is needed for computational stability, and say
# so when you do.
#
# It is no longer refused under a transform: the mixing is applied to the slot, and
# for rho >> mu, bhyp is exactly affine, so K*grad^2(nu) IS K*grad^2(rho)/2 to
# relative O((mu/rho)^2) -- measured 3.7e-9 at rho_c = 2.3e-3 kg/m^3. The exact chain
# rule was rejected: f''(0) = 1/mu over a 1.7 cm knee is not representable on a 500 m
# cell. See the Khdiff_water block in src/moist_compressible.jl.
const KH_WATER = 0.0         # < 0 => Smagorinsky K/Sc_t; 0.0 disables
const SC_T = 1.0             # turbulent Schmidt number
const N_0_MP = 8.0e6         # [m^-4] Marshall-Palmer intercept (exponential DSD)
const TAU_QSS = 10.0         # [s] supersaturation relaxation

# ── Water control variables (Ooyama 2001 Eq. 4.19-4.23) ─────────────────────
# Slots 8 and 9 carry nu = bhyp(rho) instead of the density, so the RECOVERED
# density is non-negative BY CONSTRUCTION -- no limiter, no clamp, nothing repaired.
# Both species, because that is the only configuration available on a NEST: a child
# patch's i-boundary is R3X, which the spline coefficient bound rejects, so a bounded
# child runs on a k-only bound with a measured mass-creation leak
# (bound_shortfall_total 573 on quick nested O01) while the transform has no such
# restriction and delivers min rho = 0 exactly across the interface.
#
# Why this matters here specifically: the negative condensate was a PHYSICS-FREE
# RESERVOIR (every microphysical rate is max(rho,0)-guarded, so a negative point has
# no sink while the compensating positive overshoot is consumed every step), and it
# fed the temperature retrieval directly through rho_liq. On quick O01 it reached
# 2.75x the real cloud mass. reference/FINDINGS_CONDENSATE_STAGE1.md has the ladder.
#
# EXPECT the water partition to look WORSE, not better: min rho_v tracks min rho_w
# once the negative condensate can no longer cancel part of the rho_t - rho_d
# deficit. That deficit is a difference of two independently fitted fields and no
# condensate scheme reaches it (benchmarks/FUTURE_WORK.md). It is an unmasking.
#
# mu stays at Ooyama's own 1e-7 kg/m^3: the sweep moved max_w by 0.34% between 1e-8
# and 1e-7, and the retrieval cost is at most 7.2e-3 K, at the model top.
const CONDENSATE_TRANSFORM = Symbol(get(ENV, "SCYTHE_TC_CTRANS", "bhyp"))
const RAIN_TRANSFORM       = Symbol(get(ENV, "SCYTHE_TC_RTRANS", "bhyp"))

# Sponge (Durran-Klemp 1983, stratosphere-confined)
const SPONGE_ALPHA = 0.02    # [1/s]
const Z_DAMP = 17.0e3        # [m] sponge onset (tropopause knot at 16.59 km)

# ── Grid resolution ─────────────────────────────────────────────────────────
# Two configurations, selected by SCYTHE_TC_RES (default "coarse").
#
# "coarse" = 10 km radial cells (3.33 km nodal in the inner nest) x 500 m
# vertical cells (167 m nodal), 25 km top. It exists because axisymmetric
# spin-up from a cold start takes DAYS, and the fine grid runs at ~1x real time,
# which makes a multi-day integration impossible. Relative to "fine" it has 5.6x
# fewer gridpoints per nest and raises dz_min 67.6 -> 112.7 m, relaxing the
# vertical convective SI ceiling 0.57 -> 0.96 s.
#
# The vertical spacing is a deliberate COMPROMISE, arrived at empirically. A
# 1 km-cell version (25 cells) was tried first, following RE87 (15 km radial /
# 1.25 km vertical): it ran ~40x real time and DID spin up -- convection at
# 5-6 h, rain at 7 h, vortex 21 -> 55 m/s -- but blew up at 8.1 h on the AXIS
# column (r = 1127 m, the first gridpoint), where w reached +20.9/-14.5 m/s
# while the legitimate eyewall updraft sat at r = 50 km. Output preserved in
# tc/output/tc_coarse_v30_spinup_axiscrash/. 500 m restores most of the vertical
# fidelity -- and note surface exchange is delivered as an analytic flux
# divergence over the LOWEST CELL, so cell depth directly scales how the surface
# fluxes are spread; the Louis BL and surface-flux schemes were built at
# 250-300 m, and 500 m stays within a factor of two of that.
#
# "fine" is the original 3 km / 300 m grid, kept for structure runs.
const TC_RES = get(ENV, "SCYTHE_TC_RES", "coarse")
const _tc_coarse = TC_RES == "coarse"

const NEST_BOUNDARIES = [0.0, 150.0e3, 450.0e3, 1050.0e3]
const NEST_CELLS = _tc_coarse ? [15, 15, 15] : [50, 50, 50]   # 10|20|40 vs 3|6|12 km
const Z_TOP = _tc_coarse ? 25.0e3 : 25.2e3
const NUM_CELLS_K = _tc_coarse ? 50 : 84                      # 500 m vs 300 m cells

# Timesteps. delta = 0.25 is the ONLY limit holding ts below 2.25 s here:
#   vertical convective SI, delta=0.25 : 0.96 s   <-- suspect, see below
#   vertical convective SI, delta=0.10 : 2.39 s
#   w advective (w_max 25)             : 2.25 s
#   horizontal acoustic (explicit AB3) : 3.31 s
#   u advective (u_max 90)             : 12.5 s
# ts = 1.0 s (Co_z 3.02) deliberately sits ~5% ABOVE the delta = 0.25 ceiling
# and comfortably inside every other limit and any delta <~ 0.16. NOTE delta =
# 0.25 was
# calibrated on the pre-fix TC run whose init carried 40881 supersaturated
# points (reference/SI_CONVECTIVE_CEILING.md derives it from that run, whose own note
# records rho_d deficits of 25-40% as "far beyond any physical warm anomaly").
# With the initialization fixed the observed deficit is 4.4%, so delta is likely
# nearer 0.05 and the ceiling correspondingly looser, so STATE_DEVIATION below
# is set to 0.15 -- conservative against the observed 0.044, far below the
# discredited 0.25 -- to keep the advisory honest rather than crying wolf.
# Beyond that the next ceiling is the EXPLICIT horizontal acoustic limit (3.31 s
# coarse), which is why RE87 (20 s) and BR09 (7.5 s) can step so much longer:
# they sub-step the acoustics. Split-explicit acoustics + BR09's weak divergence
# damper are the route past it.
#
# SCYTHE_TC_TS_SCALE multiplies every entry, so a diagnostic run can sweep the
# timestep WITHOUT editing this file -- which matters because a restart re-reads
# tc_params.jl, so an edit made while a run is in flight silently changes what a
# later restart does. See reference/SI_WALL_BC_CEILING.md: the measured ceiling is
# ts ~ 0.75 s with SecondDerivativeBC walls and ~2.0 s with Neumann, so ts = 1.0
# below is ABOVE its ceiling until the inhomogeneous-Neumann wall fix lands.
const TS_SCALE = parse(Float64, get(ENV, "SCYTHE_TC_TS_SCALE", "1.0"))
# 1.0 -> 0.5 (2026-07-21): ts = 1.0 is ABOVE the measured wall-condition ceiling
# of ~0.75 s for SecondDerivativeBC walls. 0.5 is validated over 12 h.
const NEST_TS = TS_SCALE .* (_tc_coarse ? [0.5, 0.5, 0.5] : [0.25, 0.25, 0.25])
const STATE_DEVIATION = 0.15  # convective state deviation for the ts advisory
const NEST_WORKERS = [1, 1, 1]

# RLR requires each junction to sit a whole number of PARENT cells from the
# origin (global ring numbering), so the RLR layout differs from the axisym one.
const NEST_BOUNDARIES_RLR = _tc_coarse ? [0.0, 160.0e3, 480.0e3, 1120.0e3] :
                                         [0.0, 150.0e3, 456.0e3, 1056.0e3]
const NEST_CELLS_RLR = _tc_coarse ? [16, 16, 16] : [50, 51, 50]

# Balance work grid (radial; the vertical axis is the model mish itself)
const DR_WORK = 500.0        # [m]

# Run control (override integration_time on the command line where supported)
const OUTPUT_INTERVAL = 3600.0      # [s] hourly gridded output
const RESTART_INTERVAL = 21600.0    # [s] 6-hourly JLD2 restart
const OUTPUT_FORMATS = [:netcdf]    # add :csv for short diagnostic runs
# Output directory. Overridable via SCYTHE_TC_OUTDIR so concurrent or successive
# runs (SLURM job arrays, resolution sweeps) each write to their own tree instead
# of clobbering the previous one. Model output is experimental data: give every
# run its own directory and MOVE old trees aside rather than deleting them.
const OUTPUT_DIR = get(ENV, "SCYTHE_TC_OUTDIR",
                       joinpath(@__DIR__, "output", "tc_axisym"))
