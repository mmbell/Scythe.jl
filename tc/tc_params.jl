# ── Tropical cyclone simulation parameters ──────────────────────────────────
# Every knob for the TC runs lives here; tc_init.jl and the run scripts
# include this file. Edit and rerun tc_init to regenerate the initial
# conditions.

# Vortex (modified Rankine, gradient-wind + hydrostatically balanced)
const VMAX = 15.0            # [m/s] maximum tangential wind
const RMW = 50.0e3           # [m] radius of maximum wind
const RANKINE_ALPHA = 0.3    # outer decay exponent, v = VMAX (RMW/r)^alpha
const V_TOP = 15.0e3         # [m] winds decay linearly to zero here

# Environment
const F_COR = 2.0 * 7.292e-5 * sind(15.0)   # [1/s] Coriolis at 15 N (3.775e-5)
const SST_K = 301.15                        # [K] fixed sea surface temp (28 C)
const SOUNDING = joinpath(@__DIR__, "..", "benchmarks", "reference_data",
                          "o01_rainfall", "dunion_MT_hum90.ref")

# Surface exchange and boundary layer
const CK = 1.0e-3            # bulk enthalpy/moisture exchange coefficient
const CD = -1.0              # negative => wind-speed-dependent Komori et al. (2018)
const U_MIN = 2.0            # [m/s] gustiness floor on the exchange wind
const L_INF = 80.0           # [m] asymptotic Louis mixing length

# Horizontal turbulence (Smagorinsky) and microphysics
const LS_SMAG = 200.0        # [m] Smagorinsky length scale
const K_MIN = 5.0            # [m²/s] horizontal eddy viscosity floor
const N_0_MP = 8.0e6         # [m^-4] Marshall-Palmer intercept (exponential DSD)
const TAU_QSS = 10.0         # [s] supersaturation relaxation

# Sponge (Durran-Klemp 1983, stratosphere-confined)
const SPONGE_ALPHA = 0.02    # [1/s]
const Z_DAMP = 17.0e3        # [m] sponge onset (tropopause knot at 16.59 km)

# Grid: 3 nests, 2:1 refinement chain (3 | 6 | 12 km), 250 m vertical to 25 km
const NEST_BOUNDARIES = [0.0, 150.0e3, 450.0e3, 1050.0e3]
const NEST_CELLS = [50, 50, 50]
# The mc timestep ceiling is VERTICAL even with semi-implicit on (empirical
# acoustic Courant ≈ 1.8 on the min Gauss spacing; ts = 0.75 at 250-m cells
# blows up within minutes). All patches share the vertical grid, so per-nest
# ts scaling with DX buys nothing here — the nesting speedup is the column
# count. 0.3 s = Courant 1.8 at dz_min = 56.35 m.
const NEST_TS = [0.3, 0.3, 0.3]           # [s]; outer patch is the root
const NEST_WORKERS = [1, 1, 1]
const Z_TOP = 25.0e3
const NUM_CELLS_K = 100

# Balance work grid (radial; the vertical axis is the model mish itself)
const DR_WORK = 500.0        # [m]

# Run control (override integration_time on the command line where supported)
const OUTPUT_INTERVAL = 3600.0      # [s] hourly gridded output
const RESTART_INTERVAL = 21600.0    # [s] 6-hourly JLD2 restart
const OUTPUT_FORMATS = [:netcdf]    # add :csv for short diagnostic runs
const OUTPUT_DIR = joinpath(@__DIR__, "output", "tc_axisym")
