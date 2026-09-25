# ── DEPRECATED shim: TC parameters, now derived from simulations/tc_axisym/config.toml ──
#
# tc/tc_params.jl + tc/tc_init.jl are a THIN, READ-ONLY shim over the consolidated run
# configuration (src/config.jl, src/config_build.jl, src/config_init.jl), kept ONLY for the
# gitignored model_tests/ scratch scripts that still `include` them (tc_balance_holdtest,
# tc_discrete_balance, tc_init_diagnostics, tc_drain_onset, tc_lid_drift_probe,
# tc_single_patch_holdtest, tc_balance_floor). For a real run use
# simulations/tc_axisym/run.jl -- see simulations/tc_axisym/README.md, in particular its
# "From the old environment variables" table: every SCYTHE_TC_* knob this file used to read
# is GONE (`--set key.path=value` on the launcher, or an overlay TOML such as fine.toml /
# rlr.toml, replaces each one) except SCYTHE_TC_RES=fine, kept below as the one legacy knob
# worth keeping for a scratch script that wants the fine grid without a `--set`.
#
# THE PHYSICS IS NOW THE PRODUCTION SET: ISHMAEL ice + two-moment rain, MYNN (fidelity
# "scythe", EDMF off), Monin-Obukhov surface stability, diurnal RRTMGP radiation, the DK83
# sponge, and the state-dependent vertical semi-implicit solver -- see
# simulations/tc_axisym/config.toml's header for the physical reasoning behind each default
# and simulations/README.md for the full key reference. This is NOT the old "Louis + no ice
# + no radiation" TC default: a caller that wants that has to ask for it explicitly (and
# physics.boundary_layer.scheme = "louis" refuses ice, so drop ice too).
#
# RADIATION IS ON BY DEFAULT (physics.radiation.scheme = "rrtmgp"), and RRTMGP/ClimaComms
# are WEAK dependencies of Scythe (a package extension, ext/ScytheRRTMGPExt.jl, that loads
# only from an environment listing them). `make_base`/`init_tc!` in tc_init.jl therefore
# WILL ERROR under `--project=.` with build_model's own "radiation = rrtmgp requested but
# the RRTMGP extension is not loaded" message -- this shim does not silently import RRTMGP
# for you. Launch under `--project=simulations`, or `import RRTMGP, ClimaComms` (on every
# worker, under Distributed) before calling `make_base`/`init_tc!`.
#
# RE87 vortex, F_COR, SST, moisture and wall-BC reasoning: see the header of
# simulations/tc_axisym/config.toml (the case) and src/idealized.jl (`re87_v`) -- not
# reproduced here.
#
# This file needs `RunConfig`/`load_config` (Scythe), so it loads Scythe ITSELF rather than
# relying on the including script's own `using Scythe` -- several model_tests scripts
# `include` tc_params.jl BEFORE `@everywhere using Scythe` (they read NEST_WORKERS to size
# `addprocs` first). `using Scythe` here is idempotent with that later `@everywhere using`.
using Scythe

const _TC_SIM_DIR = normpath(joinpath(@__DIR__, "..", "simulations", "tc_axisym"))
const _TC_RES = get(ENV, "SCYTHE_TC_RES", "coarse")
_TC_RES in ("coarse", "fine") ||
    error("SCYTHE_TC_RES = \"$(_TC_RES)\" is not recognized; use coarse or fine -- every " *
          "other SCYTHE_TC_* knob is gone, see simulations/tc_axisym/README.md")
const _TC_CONFIG_PATHS = _TC_RES == "fine" ?
    (joinpath(_TC_SIM_DIR, "config.toml"), joinpath(_TC_SIM_DIR, "fine.toml")) :
    (joinpath(_TC_SIM_DIR, "config.toml"),)

"The resolved production TC configuration (coarse or fine grid per SCYTHE_TC_RES)."
const TC_CFG::RunConfig = load_config(_TC_CONFIG_PATHS...)
"`TC_CFG` with the rlr.toml overlay: the 3-D cylindrical RLR geometry and its own nest layout."
const TC_CFG_RLR::RunConfig = load_config(_TC_CONFIG_PATHS..., joinpath(_TC_SIM_DIR, "rlr.toml"))

const _TC_WARNED = Set{Symbol}()
"Print ONE deprecation line per shim function/file, on first use."
function _tc_deprecated(fn::Symbol)
    fn in _TC_WARNED && return nothing
    push!(_TC_WARNED, fn)
    @warn "tc/tc_params.jl + tc/tc_init.jl are a DEPRECATED shim kept for model_tests/ " *
          "scripts; for a real run use simulations/tc_axisym/run.jl" caller = fn
    return nothing
end
_tc_deprecated(:tc_params)

# ── Vortex (RE87), environment, moisture, transforms, sponge: thin derivations from
# TC_CFG. See simulations/tc_axisym/config.toml for the physical reasoning.
const VORTEX_PROFILE = TC_CFG.initial_conditions.balanced_vortex.profile
const V_M             = TC_CFG.initial_conditions.balanced_vortex.v_m
const R_M             = TC_CFG.initial_conditions.balanced_vortex.r_m
const R_0             = TC_CFG.initial_conditions.balanced_vortex.r_0
const V_TOP           = TC_CFG.initial_conditions.balanced_vortex.v_top
const Z_BAROTROPIC    = TC_CFG.initial_conditions.balanced_vortex.z_barotropic
const RH_CORE         = TC_CFG.initial_conditions.balanced_vortex.rh_core
const RH_BL           = TC_CFG.initial_conditions.balanced_vortex.rh_bl
const Z_BL            = TC_CFG.initial_conditions.balanced_vortex.z_bl
const R_MOIST         = TC_CFG.initial_conditions.balanced_vortex.r_moist
const Z_MOIST         = TC_CFG.initial_conditions.balanced_vortex.z_moist
const RH_INIT_MAX     = TC_CFG.initial_conditions.balanced_vortex.rh_max
const MOIST_PROFILE   = TC_CFG.initial_conditions.balanced_vortex.moist_profile
# Legacy modified-Rankine knobs (profile = :rankine). Not part of the config schema --
# src/config_init.jl hardcodes these same literals for that branch ("the legacy
# tc_params.jl values, not config fields") -- kept here verbatim for model_tests that
# exercise the rankine fallback directly.
const VMAX          = 30.0
const RMW           = 50.0e3
const RANKINE_ALPHA = 0.3

const F_COR    = 2.0 * 7.292e-5 * sind(TC_CFG.environment.f_latitude)
const SST_K    = TC_CFG.environment.sst
const SOUNDING = TC_CFG.reference_state.sounding

const CONDENSATE_TRANSFORM = TC_CFG.physics.microphysics.condensate_transform
const RAIN_TRANSFORM       = TC_CFG.physics.microphysics.rain_transform
const TAU_QSS               = TC_CFG.physics.microphysics.tau_qss
const N_0_MP                = TC_CFG.physics.microphysics.N_0

const CD    = TC_CFG.physics.surface.Cd
const CK    = TC_CFG.physics.surface.Ck
const U_MIN = TC_CFG.physics.surface.U_min
const L_INF = TC_CFG.physics.boundary_layer.l_inf
const PR_T  = TC_CFG.physics.horizontal_mixing.Pr_t
const SC_T  = TC_CFG.physics.horizontal_mixing.Sc_t

const SPONGE_ALPHA = TC_CFG.physics.sponge.alpha
const Z_DAMP        = TC_CFG.physics.sponge.z_damp

# ── Grid, nest, timestep: thin derivations. TC_CFG_RLR carries its OWN (different) nest
# layout -- RLR needs each junction a whole number of PARENT cells from the origin.
const Z_TOP           = TC_CFG.grid.z_top
const NUM_CELLS_K     = TC_CFG.grid.num_cells_k
const NEST_BOUNDARIES = TC_CFG.grid.boundaries
const NEST_CELLS      = TC_CFG.grid.num_cells
const NEST_BOUNDARIES_RLR = TC_CFG_RLR.grid.boundaries
const NEST_CELLS_RLR      = TC_CFG_RLR.grid.num_cells
const TS_SCALE = TC_CFG.timestep.scale
const NEST_TS  = TS_SCALE .* (isempty(TC_CFG.timestep.ts_per_patch) ?
    fill(TC_CFG.timestep.ts, length(NEST_CELLS)) : TC_CFG.timestep.ts_per_patch)
const NEST_WORKERS = ones(Int, length(NEST_CELLS))
const STATE_DEVIATION = TC_CFG.diagnostics.timestep_advisory.state_deviation

# Balance work-grid spacing: not part of the config schema (build_model/initialize_case!
# only take the native-grid balance, balanced_vortex_native!, never the foreign work-grid
# path); kept literal for model_tests that still call balanced_vortex_fields directly.
const DR_WORK = 500.0

const OUTPUT_INTERVAL  = TC_CFG.output.interval
const RESTART_INTERVAL = TC_CFG.output.restart_interval
const OUTPUT_FORMATS   = TC_CFG.output.formats
const OUTPUT_DIR       = TC_CFG.output.dir
