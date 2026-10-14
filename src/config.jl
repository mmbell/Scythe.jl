# ── Consolidated run configuration: typed structs, TOML front end, key registries ──
#
# A moist-compressible run used to be assembled from four unrelated mechanisms: the
# `ModelParameters` Dicts, ~90 `get(options, :key, default)` reads scattered through
# `src/`, three driver scripts that each translated their own environment variables into
# those Dicts, and driver logic interleaved with model configuration. A typo in any
# options key outside the three closed sets (MYNN, radiation, NetCDF) was silently
# ignored, and the three drivers each carried their own copy of the physics expansion.
#
# This file is the FRONT HALF of the replacement: a typed `RunConfig` that describes WHAT
# is run, never HOW (workers, threads and the scheduler are launcher concerns -- two
# configs that differ only in launcher must produce identical output, so nothing about
# execution may live here). It knows nothing about grids, tiles or `ModelParameters`;
# `build_model` (src/config_build.jl) is the single place that turns a `RunConfig` into
# the `options`/`physical_params` Dicts the model reads.
#
# THREE LAYERS:
#
#   1. The section structs (`Base.@kwdef`, every field typed and defaulted) composed into
#      `RunConfig`. Their defaults ARE the preferred production moist-compressible physics
#      (ice + full microphysics, MYNN :scythe without EDMF, surface fluxes with stability,
#      both reference-state fixes, the DK83 sponge, diurnal RRTMGP) on a generic
#      single-patch periodic XZ domain; a case file states only what differs.
#   2. The TOML front end: `load_config` (parse, deep-merge files, `--set`, typed
#      conversion with path resolution) and `write_config` (the full resolved TOML,
#      defaults included, so an output directory records exactly what ran).
#   3. The key registries `KNOWN_OPTION_KEYS` / `KNOWN_PHYSICAL_KEYS`: every key `src/`
#      reads from `options`/`physical_params`, with the type its read site expects. They
#      validate the `[extra.*]` escape hatch, and test/test_config.jl greps `src/` so a
#      new `get(options, :key, ...)` cannot land without a registry line.
#
# VALIDATION LIVES IN THE `RunConfig` CONSTRUCTOR, not in the loader, so a config built in
# Julia (`RunConfig(physics = Physics(radiation = Radiation(method = :allskyy)))`) is held
# to the same rules as one read from TOML: enum values, the equation-set/geometry pairing,
# and the `[extra.*]` keys. Cross-SECTION physics rules (ice with Louis, exact_si with
# state_dependent_si, ...) are `validate(cfg)`'s job in src/config_build.jl, because they
# are about what a build can do, not about whether the file is well formed.
#
# PATHS ARE ABSOLUTE IN THE STRUCT. A relative path means "relative to the file that set
# it" in TOML, "relative to the working directory" for `--set` and for Julia construction,
# and `@scythe/` means the package root. Resolving once, at the boundary, is what makes
# `load_config(write_config(cfg)) == cfg` hold and what lets `<root>/scythe_config.toml` be
# re-read from any directory by `--resume`.

using TOML
using SHA

export RunConfig, load_config, write_config

"Supertype of `RunConfig` and every section struct: gives them field-wise `==`/`hash` and the generic TOML conversion."
abstract type AbstractConfigSection end

# ── Allowed values for the enum fields whose set is new here ──────────────────────
#
# The enum fields whose value set `src/` already defines reuse that tuple (see
# `ENUM_FIELDS` below) so a scheme added to, say, `RADIATION_METHODS` is accepted by the
# config with no edit here. Only the sets that are born with the config live here.

"Allowed `boundary_conditions.lateral` values: walls (cylindrical sets always take axis + outer wall) or periodic (XZ only; refused for the cylindrical sets at build)."
const LATERAL_BCS = (:walls, :periodic)
"Allowed `boundary_conditions.vertical_scalars` (ground) and `boundary_conditions.vertical_scalars_lid` (lid) values: the BC of every variable except `w` (and the sedimenting slots at the ground)."
const VERTICAL_SCALAR_BCS = (:d2, :natural, :r1t1x, :neumann)
"Allowed `boundary_conditions.sedimentation_lid` values: the lid BC of the rain/number/ice SLOTS (the sedimentation FLUX fit always has a zero-flux lid on a B-spline vertical)."
const SEDIMENTATION_LID_BCS = (:natural, :wall)
"Sides a `boundary_conditions.override` entry may name: `inner`/`outer` (cylindrical: axis/outer wall; XZ: left/right) and `bottom`/`top`."
const BC_OVERRIDE_SIDES = ("inner", "outer", "bottom", "top")
"Values a `boundary_conditions.override` entry may take: Springsteel's BC vocabulary (`r1t1x` = `CubicBSpline.R1T1X`, the inhomogeneous-Neumann pressure wall)."
const BC_OVERRIDE_VALUES = ("natural", "dirichlet", "neumann", "second_derivative", "periodic", "r1t1x")
"Allowed `initial_conditions.method` values."
const IC_METHODS = (:balanced_vortex, :temperature_noise, :temperature_bubble, :file)
"Allowed `initial_conditions.balanced_vortex.profile` values (the `vortex_profile` keyword of `balanced_vortex_native!`)."
const VORTEX_PROFILES = (:re87, :rankine)
"Allowed `initial_conditions.balanced_vortex.moist_profile` values."
const VORTEX_MOIST_PROFILES = (:vortex, :gaussian)
"Allowed `physics.microphysics.ice` values (`options[:ice_microphysics]`)."
const ICE_SCHEMES = (:ishmael, :none)
"Allowed `physics.microphysics.rain_moments` values; 0 = auto (2 under ice, else 1), resolved at build."
const RAIN_MOMENT_CHOICES = (0, 1, 2)
"Control-variable transforms the four `*_transform_mode` accessors accept (src/moist_compressible.jl)."
const WATER_TRANSFORMS = (:none, :bhyp, :bhyp_smooth)
"Allowed `physics.boundary_layer.scheme` values."
const BL_SCHEMES = (:mynn, :louis, :none)
"Allowed `physics.horizontal_mixing.scheme` values."
const HMIX_SCHEMES = (:smagorinsky, :constant, :none)
"Allowed `numerics.positivity` values: `auto` applies the positivity rule at build, `off` never does."
const POSITIVITY_MODES = (:auto, :off)
"Allowed `numerics.temperature` values (`options[:temperature]`; `TEMPERATURE_MODES` in src/moist_compressible.jl)."
const TEMPERATURE_CHOICES = (:eos, :energy)
"Allowed `output.netcdf_grid` values (`:mish` is reserved in `validate_output_options` and refused)."
const NETCDF_GRIDS = (:regular,)

"""
    EQUATION_SETS

Canonical Scythe equation-set name → `(geometries, cyl, zcol)`: the Springsteel geometry
names `grid.geometry` may take with that set, whether the set is cylindrical (axis + outer
wall, Coriolis as `f` on the radial/azimuthal winds), and the column index of the vertical
coordinate in the patch's gridpoint layout. Both names pass through `build_model`
unchanged, so no new monikers are introduced.
"""
const EQUATION_SETS = Dict{String,NamedTuple{(:geometries, :cyl, :zcol),Tuple{Tuple{Vararg{String}},Bool,Int}}}(
    "moist_compressible_axisym" => (geometries = ("RiRk", "RZ"), cyl = true,  zcol = 2),
    "moist_compressible_XZ"     => (geometries = ("RiRk", "RZ"), cyl = false, zcol = 2),
    "moist_compressible_RLR"    => (geometries = ("RLR",),       cyl = true,  zcol = 3),
    "moist_compressible_RRR"    => (geometries = ("RRR",),       cyl = false, zcol = 3))

# ── Section structs ───────────────────────────────────────────────────────────────
#
# Field order is the order `write_config` prints, so it follows the §2.1 listing: what a
# user touches most comes first. Defaults are the production physics; see the header.

"""
    RunSection

`[run]` — what is being run and for how long.

- `name::String` — run label, used in log lines and output naming.
- `equation_set::String` — canonical Scythe equation set, a key of `EQUATION_SETS`.
- `integration_time::Float64` — total simulated time [s].
- `segment_length::Float64` — restart granularity [s]; each segment is one process; 0 = one segment.
"""
Base.@kwdef struct RunSection <: AbstractConfigSection
    name::String = "scythe_run"
    equation_set::String = "moist_compressible_XZ"
    integration_time::Float64 = 432000.0
    segment_length::Float64 = 0.0
end

"""
    OutputSection

`[output]` — where and how often the run writes.

- `dir::String` — output root; relative to the file that set it; absolute in the struct.
- `interval::Float64` — analysis-output cadence [s].
- `restart_interval::Float64` — JLD2 checkpoint cadence [s]; must be > 0 when `run.segment_length > 0`.
- `formats::Vector{Symbol}` — analysis formats, from `OUTPUT_FORMAT_SYMBOLS` (`netcdf`, `csv`, `netcdf_raw`).
- `sidecars::Bool` — also write the per-tile MYNN and radiation NetCDF sidecars.
- `netcdf_grid::Symbol` — grid the NetCDF output is written on (`regular`).
- `netcdf_derivatives::Bool` — `netcdf_raw` also writes derivative slots.
"""
Base.@kwdef struct OutputSection <: AbstractConfigSection
    dir::String = "output"
    interval::Float64 = 3600.0
    restart_interval::Float64 = 21600.0
    formats::Vector{Symbol} = [:netcdf]
    sidecars::Bool = false
    netcdf_grid::Symbol = :regular
    netcdf_derivatives::Bool = false
end

"""
    GridSection

`[grid]` — the patches and the vertical axis.

- `geometry::String` — Springsteel geometry name (`RiRk`, `RZ`, `RLR`, `RRR`); must suit `run.equation_set`.
- `boundaries::Vector{Float64}` — patch edges [m], outermost last; length 2 = single patch.
- `num_cells::Vector{Int}` — horizontal B-spline cells per patch; length `length(boundaries) - 1`.
- `z_top::Float64` — model lid [m].
- `num_cells_k::Int` — vertical B-spline cells.
"""
Base.@kwdef struct GridSection <: AbstractConfigSection
    geometry::String = "RiRk"
    boundaries::Vector{Float64} = [0.0, 300.0e3]
    num_cells::Vector{Int} = [50]
    z_top::Float64 = 25.0e3
    num_cells_k::Int = 50
end

"""
    BoundaryConditionsSection

`[boundary_conditions]` — the boundary condition of every prognostic variable, set by four family rules plus per-variable overrides. The resolved per-variable table is written to `<output.dir>/boundary_conditions.toml` and printed at startup.

- `lateral::Symbol` — `walls` or `periodic` (XZ only). XZ walls: every variable Neumann (zero gradient) on both sides, except `u` and `w` Dirichlet (zero). Cylindrical sets (always walls): axis = Neumann except `u`, `v` Dirichlet; outer wall = Neumann except `u` Dirichlet (`v` and `w` are Neumann there). periodic: `PeriodicBC` on every variable. Nest interfaces are set by the nesting, not here.
- `vertical_scalars::Symbol` — GROUND BC of every variable except `w` (always Dirichlet zero) and the sedimenting slots (always natural at the ground): `d2` second derivative zero, `natural` unconstrained, `neumann` zero gradient, `r1t1x` = `d2` except `rho_d`, `rho_t`, `E_t`, `Q_ss` Neumann and `p` the rigid-wall pressure condition (`dp'/dz = -g rho_t'` installed before every fit). Natural by default: a surface layer is logarithmic, and `d2` forces the velocity and scalar gradients constant there. The whole family must share one condition (freeing `p` or `rho_t` alone blows up); with all of it natural the resting-column SI ceiling is 0.875-0.9375 s (0.9375-1.0 s under `d2`) and the 12 h no-physics TC hold is indistinguishable from `d2` (1.78% vs 1.73% of the deficit filled). Never `neumann`: on `p` at the ground it drains the pressure field.
- `vertical_scalars_lid::Symbol` — LID BC of the same family (every variable except `w`, always Dirichlet zero), same values as `vertical_scalars`. Natural by default: `d2` at a rigid lid over-constrains the perturbation fields and drifts warm over days (multi-day 2D RCE: lid T anomaly 0.5 K at 48 h, crash at 277 h; restarted with a natural lid the anomaly decays). Clean at ts = 0.75 s on the RCE case; ts = 1.0 s showed a transient adjustment that `d2` did not.
- `sedimentation_lid::Symbol` — lid BC of the sedimenting slots (rain, two-moment rain number, the twelve ice slots): `natural` (unconstrained) or `wall` (left on `vertical_scalars_lid`). Only the slot VALUES: on a B-spline vertical every sedimentation FLUX is fitted with a zero-flux lid either way, so no hydrometeor enters through the top.
- `override::Dict{String,Dict{String,String}}` — expert per-variable overrides applied after the family rules: `{<variable> = {<side> = <bc>}}` with side `inner`/`outer`/`bottom`/`top` and bc `natural`, `dirichlet`, `neumann`, `second_derivative`, `periodic` or `r1t1x`; `inner`/`outer` act on the domain's own edges (a nest interface is always the nesting's), refused under periodic.
"""
Base.@kwdef struct BoundaryConditionsSection <: AbstractConfigSection
    lateral::Symbol = :periodic
    vertical_scalars::Symbol = :natural
    vertical_scalars_lid::Symbol = :natural
    sedimentation_lid::Symbol = :natural
    override::Dict{String,Dict{String,String}} = Dict{String,Dict{String,String}}()
end

"""
    TimestepSection

`[timestep]` — the time step of every patch.

- `ts::Float64` — root (coarsest) patch time step [s].
- `ts_per_patch::Vector{Float64}` — per-patch steps [s]; empty = every patch at `ts`.
- `scale::Float64` — multiplies `ts` and every `ts_per_patch` entry.
"""
Base.@kwdef struct TimestepSection <: AbstractConfigSection
    ts::Float64 = 0.5
    ts_per_patch::Vector{Float64} = Float64[]
    scale::Float64 = 1.0
end

"""
    ReferenceStateSection

`[reference_state]` — the sounding and the reference-state construction.

- `sounding::String` — Dunion-format sounding; `@scythe/` = package root; absolute in the struct.
- `exact::Bool` — pre-balance the reference state on the model grid (`:exact_reference_state`).
- `consistent_qss::Bool` — reference `Q_ss` consistent with the discrete retrieval (`:consistent_qss_reference`).
- `hydrostatic::Bool` — discretely hydrostatic reference (`:hydrostatic_reference`).
"""
Base.@kwdef struct ReferenceStateSection <: AbstractConfigSection
    sounding::String = "@scythe/benchmarks/reference_data/o01_rainfall/dunion_MT_hum90.ref"
    exact::Bool = true
    consistent_qss::Bool = true
    hydrostatic::Bool = true
end

"""
    VortexIC

`[initial_conditions.balanced_vortex]` — the balanced axisymmetric vortex (`balanced_vortex_native!`).

- `profile::Symbol` — tangential-wind profile: `re87` (Rotunno & Emanuel 1987) or `rankine`.
- `v_m::Float64` — maximum tangential wind [m/s].
- `r_m::Float64` — radius of maximum wind [m].
- `r_0::Float64` — outer radius where the wind vanishes [m].
- `v_top::Float64` — height at which the wind has decayed to zero [m].
- `z_barotropic::Float64` — depth of the barotropic (height-invariant) layer [m].
- `rh_core::Float64` — relative humidity in the moist core.
- `rh_bl::Float64` — relative humidity in the boundary layer.
- `z_bl::Float64` — boundary-layer depth for `rh_bl` [m].
- `r_moist::Float64` — radius of the moist core [m].
- `z_moist::Float64` — depth of the moist core [m].
- `rh_max::Float64` — relative-humidity cap.
- `moist_profile::Symbol` — moist-core weighting: `vortex` (surface wind) or `gaussian`.
"""
Base.@kwdef struct VortexIC <: AbstractConfigSection
    profile::Symbol = :re87
    v_m::Float64 = 15.0
    r_m::Float64 = 82.5e3
    r_0::Float64 = 800.0e3
    v_top::Float64 = 15.0e3
    z_barotropic::Float64 = 0.0
    rh_core::Float64 = 0.95
    rh_bl::Float64 = 0.90
    z_bl::Float64 = 1.5e3
    r_moist::Float64 = 150.0e3
    z_moist::Float64 = 10.0e3
    rh_max::Float64 = 0.98
    moist_profile::Symbol = :vortex
end

"""
    NoiseIC

`[initial_conditions.temperature_noise]` — seeded, RH-preserving low-level noise (`moist_temperature_noise_mc!`).

- `z_max::Float64` — top of the perturbed layer [m].
- `dT_max::Float64` — noise amplitude [K].
- `seed::Int` — RNG seed.
"""
Base.@kwdef struct NoiseIC <: AbstractConfigSection
    z_max::Float64 = 1000.0
    dT_max::Float64 = 0.1
    seed::Int = 2026
end

"""
    BubbleIC

`[initial_conditions.temperature_bubble]` — a warm ellipsoidal bubble (`moist_temperature_bubble_mc!`).

- `xc::Float64` — bubble centre, horizontal [m].
- `xr::Float64` — horizontal radius [m].
- `zc::Float64` — bubble centre, vertical [m].
- `zr::Float64` — vertical radius [m].
- `dT_max::Float64` — peak temperature perturbation [K].
"""
Base.@kwdef struct BubbleIC <: AbstractConfigSection
    xc::Float64 = 150.0e3
    xr::Float64 = 30.0e3
    zc::Float64 = 500.0
    zr::Float64 = 3000.0
    dT_max::Float64 = 3.0
end

"""
    FileIC

`[initial_conditions.file]` — initial conditions read from files.

- `paths::Vector{String}` — one per patch, `.csv` or `.jld2`; relative to the file that set them; absolute in the struct.
"""
Base.@kwdef struct FileIC <: AbstractConfigSection
    paths::Vector{String} = String[]
end

"""
    InitialConditions

`[initial_conditions]` — which initial condition, and the parameters of all four.
All four sub-tables are always held so a later file or `--set` can switch `method`; only the active
one is written back by `write_config`.

- `method::Symbol` — `balanced_vortex`, `temperature_noise`, `temperature_bubble` or `file`.
- `balanced_vortex::VortexIC` — parameters for `method = balanced_vortex`.
- `temperature_noise::NoiseIC` — parameters for `method = temperature_noise`.
- `temperature_bubble::BubbleIC` — parameters for `method = temperature_bubble`.
- `file::FileIC` — parameters for `method = file`.
"""
Base.@kwdef struct InitialConditions <: AbstractConfigSection
    method::Symbol = :temperature_noise
    balanced_vortex::VortexIC = VortexIC()
    temperature_noise::NoiseIC = NoiseIC()
    temperature_bubble::BubbleIC = BubbleIC()
    file::FileIC = FileIC()
end

"""
    Environment

`[environment]` — the fixed external environment.

- `f_latitude::Float64` — latitude of the f-plane [deg]; `f = 2Ω sin φ` is computed at build.
- `sst::Float64` — sea-surface temperature [K].
"""
Base.@kwdef struct Environment <: AbstractConfigSection
    f_latitude::Float64 = 15.0
    sst::Float64 = 302.65
end

"""
    Microphysics

`[physics.microphysics]` — warm rain, ice and the water control variables.

- `ice::Symbol` — ice microphysics: `ishmael` or `none`.
- `rain_moments::Int` — 1 or 2; 0 = auto (2 under ice, else 1), resolved at build.
- `precipitation::Bool` — rain production and sedimentation.
- `condensation::Bool` — phase change (QSS condensation and the ice transfers).
- `condensate_transform::Symbol` — cloud-slot control variable: `none`, `bhyp`, `bhyp_smooth`.
- `rain_transform::Symbol` — rain-slot control variable: `none`, `bhyp`, `bhyp_smooth`.
- `rain_number_transform::Symbol` — rain-number-slot control variable (two-moment rain).
- `ice_transform::Symbol` — control variable of the twelve ice slots.
- `condensate_mu::Float64` — bhyp width for cloud [kg/m³].
- `rain_mu::Float64` — bhyp width for rain [kg/m³].
- `N_0::Float64` — rain DSD intercept [m⁻⁴].
- `N_r::Float64` — single-moment rain number parameter.
- `tau_qss::Float64` — QSS relaxation timescale [s].
"""
Base.@kwdef struct Microphysics <: AbstractConfigSection
    ice::Symbol = :ishmael
    rain_moments::Int = 0
    precipitation::Bool = true
    condensation::Bool = true
    condensate_transform::Symbol = :bhyp
    rain_transform::Symbol = :bhyp
    rain_number_transform::Symbol = :bhyp
    ice_transform::Symbol = :bhyp
    condensate_mu::Float64 = 1.0e-7
    rain_mu::Float64 = 1.0e-7
    N_0::Float64 = 8.0e6
    N_r::Float64 = 1.0e-3
    tau_qss::Float64 = 10.0
end

"""
    BoundaryLayer

`[physics.boundary_layer]` — vertical mixing and the surface layer underneath it.

- `scheme::Symbol` — `mynn`, `louis` or `none`.
- `mynn_interval::Float64` — MYNN call cadence [s]; must be ≥ the patch time step.
- `mynn_edmf::Bool` — MYNN eddy-diffusivity mass flux (becomes `:mynn_edmf` 0/1 at build).
- `mynn_fidelity::String` — `scythe`, `fortran`, or a comma list of deviations (`parse_mynn_fidelity`).
- `mynn_init::Symbol` — TKE initialisation: `taper` or `zero`.
- `K_max::Float64` — counted safety cap on the MYNN exchange coefficient [m²/s]; `inf` = none.
- `l_inf::Float64` — Louis asymptotic mixing length [m] (louis only).
"""
Base.@kwdef struct BoundaryLayer <: AbstractConfigSection
    scheme::Symbol = :mynn
    mynn_interval::Float64 = 20.0
    mynn_edmf::Bool = false
    mynn_fidelity::String = "scythe"
    mynn_init::Symbol = :taper
    K_max::Float64 = Inf
    l_inf::Float64 = 80.0
end

"""
    Surface

`[physics.surface]` — air–sea exchange.

- `fluxes::Bool` — surface momentum, heat and moisture fluxes.
- `roughness::Symbol` — neutral roughness model: `komori`, `gfdl_v7` or `charnock` (`SFC_Z0_MODES`).
- `stability::Bool` — Monin–Obukhov stability correction of the exchange coefficients.
- `Cd::Float64` — drag coefficient; negative = Komori wind-dependent drag.
- `Ck::Float64` — enthalpy/moisture exchange coefficient (when not stability-derived).
- `U_min::Float64` — gustiness floor on the surface wind speed [m/s].
"""
Base.@kwdef struct Surface <: AbstractConfigSection
    fluxes::Bool = true
    roughness::Symbol = :komori
    stability::Bool = true
    Cd::Float64 = -1.0
    Ck::Float64 = 1.0e-3
    U_min::Float64 = 2.0
end

"""
    Radiation

`[physics.radiation]` — radiative transfer.

- `scheme::Symbol` — `rrtmgp`, `prescribed` or `none` (`RADIATION_SCHEMES`).
- `method::Symbol` — `gray`, `clearsky`, `allsky` or `allsky_clear` (`RADIATION_METHODS`).
- `solar::Symbol` — shortwave forcing: `none`, `fixed` or `diurnal` (`RADIATION_SOLAR_MODES`).
- `forcing::Symbol` — `full` or `anomaly` (horizontal mean removed) (`RADIATION_FORCINGS`).
- `interval::Float64` — radiation call cadence [s].
- `z_max::Float64` — top of the radiative heating taper [m]; 0 = `physics.sponge.z_damp`, resolved at build.
- `latitude::Float64` — latitude for the solar geometry [deg].
- `start_doy::Float64` — day of year at t = 0.
- `start_hour::Float64` — UTC hour at t = 0 (advanced per segment by the driver).
- `albedo::Float64` — surface albedo.
- `emissivity::Float64` — surface emissivity.
"""
Base.@kwdef struct Radiation <: AbstractConfigSection
    scheme::Symbol = :rrtmgp
    method::Symbol = :allsky
    solar::Symbol = :diurnal
    forcing::Symbol = :full
    interval::Float64 = 300.0
    z_max::Float64 = 0.0
    latitude::Float64 = 20.0
    start_doy::Float64 = 240.0
    start_hour::Float64 = 0.0
    albedo::Float64 = 0.06
    emissivity::Float64 = 0.98
end

"""
    HorizontalMixing

`[physics.horizontal_mixing]` — HORIZONTAL eddy mixing only; vertical mixing belongs to `boundary_layer`.

- `scheme::Symbol` — `smagorinsky` (horizontal-plane strain, every geometry), `constant` or `none`.
- `Ls::Float64` — Smagorinsky length scale [m]; 0 (the default) = `C_s` × each patch's horizontal cell width, on any geometry; > 0 = that absolute length, the same on every patch.
- `C_s::Float64` — Smagorinsky constant; read ONLY when `Ls = 0`, then the length scale is `C_s·Δx` per patch (Δx = that patch's i-cell width).
- `K_min::Float64` — Smagorinsky viscosity floor [m²/s].
- `K::Float64` — constant momentum viscosity [m²/s] (`constant` only).
- `heat::Bool` — also mix entropy/heat with K/Pr_t.
- `water::Bool` — also mix every water species with K/Sc_t: total water, vapor, cloud, rain (+ number) and, with ice, every ice mass and moment, all with the one K (NOT energy consistent).
- `Pr_t::Float64` — turbulent Prandtl number.
- `Sc_t::Float64` — turbulent Schmidt number.

THE LENGTH SCALE RULE is one rule on every geometry: `Ls > 0` is an absolute length used
unchanged on every patch (and `C_s` is not read); `Ls = 0` makes it `C_s·Δx`, with Δx the
patch's own horizontal cell width `(iMax − iMin)/num_cells_i` (the same Δx the MYNN
scale-aware taper uses), so it refines with a nest. The default is `Ls = 0`: the grid-scaled
length is the closure's own definition and the only one that is resolution-general, so every
geometry gets it unless a configuration says otherwise. The axisymmetric TC was tuned with an
absolute `Ls = 500` m and pins that value in `simulations/tc_axisym/scythe_config.toml`, so it
is unchanged; any other configuration that relied on the old 500 m default must now set it.

What "strain" means: the HORIZONTAL-PLANE deformation only, `K = max(L² √(2 S_ij S_ij), K_min)`
over i, j in the horizontal plane. On the cylinders `S_rr = ∂u/∂r`, `S_λλ = (1/r)∂v/∂λ + u/r`,
`S_rλ`; on the XZ slice the plane has one direction and the one component is `S_xx = ∂u/∂x`,
so `K = max(L² √2 |∂u/∂x|, K_min)`. `∂w/∂x` and `∂u/∂z` are vertical shear and belong to the
boundary-layer scheme; `∂w/∂z` is `−∂u/∂x` by continuity and would count the same
deformation twice.
"""
Base.@kwdef struct HorizontalMixing <: AbstractConfigSection
    scheme::Symbol = :none
    Ls::Float64 = 0.0
    C_s::Float64 = 0.2
    K_min::Float64 = 25.0
    K::Float64 = 0.0
    heat::Bool = true
    water::Bool = false
    Pr_t::Float64 = 1.0
    Sc_t::Float64 = 1.0
end

"""
    Sponge

`[physics.sponge]` — the Durran & Klemp (1983) Rayleigh sponge.

- `alpha::Float64` — damping coefficient [s⁻¹].
- `z_damp::Float64` — bottom of the sponge [m].
"""
Base.@kwdef struct Sponge <: AbstractConfigSection
    alpha::Float64 = 0.02
    z_damp::Float64 = 17.0e3
end

"""
    Physics

`[physics]` — container for the physics sections (TOML `[physics.<section>]`).

- `microphysics::Microphysics` — warm rain, ice, water transforms.
- `boundary_layer::BoundaryLayer` — vertical mixing scheme.
- `surface::Surface` — air–sea fluxes.
- `radiation::Radiation` — radiative transfer.
- `horizontal_mixing::HorizontalMixing` — horizontal eddy mixing.
- `sponge::Sponge` — upper-level Rayleigh damping.
"""
Base.@kwdef struct Physics <: AbstractConfigSection
    microphysics::Microphysics = Microphysics()
    boundary_layer::BoundaryLayer = BoundaryLayer()
    surface::Surface = Surface()
    radiation::Radiation = Radiation()
    horizontal_mixing::HorizontalMixing = HorizontalMixing()
    sponge::Sponge = Sponge()
end

"""
    Numerics

`[numerics]` — solver choices; leave alone unless you know why (boundary conditions live in `[boundary_conditions]`).

- `positivity::Symbol` — `auto` (untransformed rain on k, plus i with walls) or `off`.
- `state_dependent_si::Bool` — vertical SI coefficients from the current column state.
- `exact_si::Bool` — exact coupled SI solve; incompatible with `state_dependent_si`.
- `horizontal_semiimplicit::Bool` — horizontal semi-implicit acoustic terms (XZ RiRk only).
- `wall_bc_tau::Float64` — relaxation timescale [s] of the `p` lid/ground condition under `boundary_conditions.vertical_scalars` or `vertical_scalars_lid` = `r1t1x`; `inf` = none.
- `l_q::Float64` — B-spline spectral filter length, every variable, every direction not overridden below.
- `l_q_horizontal::Float64` — filter length of the horizontal B-spline (x / r); 0 = inherit `l_q`.
- `l_q_vertical::Float64` — filter length of the vertical B-spline (z); 0 = inherit `l_q`; refused on `RZ` (Chebyshev vertical).
- `cfl_interval::Float64` — CFL report cadence [s]; 0 = `output.interval`, resolved at build.
- `temperature::Symbol` — the model temperature every consumer reads (microphysics, radiation, surface layer, boundary layer, output): `eos` = `p/(rho_d R_d + rho_v R_v)` from the prognostic pressure and vapor (default since 2026-10-14), `energy` = the closed-form total-energy retrieval, which reads the clipped condensate and warms by ~2.5 K per 1e-3 kg/m3 of clipped lobe. `E_t` stays prognostic either way; the output's `T_split` and `E_residual` measure the disagreement.

The filter lengths are separate because the two directions do different jobs: the
vertical `l_q` is load-bearing for the semi-implicit acoustic solve (its column operators
are built on the filtered vertical basis, reference/SI_VERTICAL_CEILING.md), while the
horizontal one only sets how hard the grid scale is damped along the slice. An azimuthal
(Fourier) axis has no filter length, so on `RLR` `l_q_horizontal` is the radial filter.
0 means "inherit" for both, so to filter one direction with `l_q = 0` (an interpolating
fit) set `l_q = 0` and give the OTHER direction its own value.
"""
Base.@kwdef struct Numerics <: AbstractConfigSection
    positivity::Symbol = :auto
    state_dependent_si::Bool = true
    exact_si::Bool = false
    horizontal_semiimplicit::Bool = false
    wall_bc_tau::Float64 = 300.0
    l_q::Float64 = 2.0
    l_q_horizontal::Float64 = 0.0
    l_q_vertical::Float64 = 0.0
    cfl_interval::Float64 = 0.0
    temperature::Symbol = :eos
end

"""
    TimestepAdvisory

`[diagnostics.timestep_advisory]` — inputs to the WARN-ONLY startup timestep report; they change nothing in the run.

- `u_max::Float64` — expected peak horizontal wind [m/s] for the advective Courant lines.
- `w_max::Float64` — expected peak vertical wind [m/s].
- `bl_K_ref::Float64` — reference BL eddy diffusivity [m²/s] for the explicit-diffusion line.
- `state_deviation::Float64` — convective-ceiling target δ; that line is omitted when `state_dependent_si`.
"""
Base.@kwdef struct TimestepAdvisory <: AbstractConfigSection
    u_max::Float64 = 90.0
    w_max::Float64 = 25.0
    bl_K_ref::Float64 = 3000.0
    state_deviation::Float64 = 0.25
end

"""
    Diagnostics

`[diagnostics]` — run-time traces printed to the log.

- `state_minima_trace::Int` — state-minima print interval [steps]; 0 = off.
- `water_budget_trace::Int` — water-budget print interval [steps]; 0 = off (refused with ice).
- `stiffness_trace::Int` — stiffness-census print interval [steps]; 0 = off.
- `mynn_trace::Bool` — MYNN per-call summary line.
- `radiation_trace::Bool` — radiation per-call summary line.
- `radiation_trace_sw::Bool` — shortwave detail in the radiation trace.
- `timestep_advisory::TimestepAdvisory` — inputs to the startup timestep report.
"""
Base.@kwdef struct Diagnostics <: AbstractConfigSection
    state_minima_trace::Int = 0
    water_budget_trace::Int = 0
    stiffness_trace::Int = 0
    mynn_trace::Bool = true
    radiation_trace::Bool = true
    radiation_trace_sw::Bool = false
    timestep_advisory::TimestepAdvisory = TimestepAdvisory()
end

"""
    Extra

`[extra]` — the escape hatch for keys no typed section owns.

- `options::Dict{Symbol,Any}` — merged into `options`; keys from `KNOWN_OPTION_KEYS`, not `OWNED_OPTION_KEYS`; values coerced to the registered type.
- `physical_params::Dict{Symbol,Float64}` — merged into `physical_params`; numbers only; keys from `KNOWN_PHYSICAL_KEYS`, not `OWNED_PHYSICAL_KEYS`.
"""
Base.@kwdef struct Extra <: AbstractConfigSection
    options::Dict{Symbol,Any} = Dict{Symbol,Any}()
    physical_params::Dict{Symbol,Float64} = Dict{Symbol,Float64}()
end

"""
    RunConfig

The complete, typed description of a moist-compressible run: WHAT is run, never how
(workers, threads and the scheduler belong to the launcher). Build it in Julia with
keywords (`RunConfig(run = RunSection(name = "x"), physics = Physics(...))`) or from TOML
with [`load_config`](@ref); write it with [`write_config`](@ref). The constructor checks
enum values, the equation-set/geometry pairing and the `[extra.*]` keys, and makes every
path absolute (relative paths resolve against the working directory; `@scythe/` against
the package root).

- `run::RunSection` — `[run]`: name, equation set, length, segmenting.
- `output::OutputSection` — `[output]`: directory, cadences, formats.
- `grid::GridSection` — `[grid]`: geometry, patches, vertical axis.
- `boundary_conditions::BoundaryConditionsSection` — `[boundary_conditions]`: lateral, vertical and sedimentation BC families, per-variable overrides.
- `timestep::TimestepSection` — `[timestep]`: root and per-patch steps.
- `reference_state::ReferenceStateSection` — `[reference_state]`: sounding and reference fixes.
- `initial_conditions::InitialConditions` — `[initial_conditions]`: method and its parameters.
- `environment::Environment` — `[environment]`: f-plane latitude, SST.
- `physics::Physics` — `[physics.*]`: microphysics, boundary layer, surface, radiation, horizontal mixing, sponge.
- `numerics::Numerics` — `[numerics]`: solver choices.
- `diagnostics::Diagnostics` — `[diagnostics]`: log traces and the timestep advisory.
- `extra::Extra` — `[extra]`: validated escape hatch into `options`/`physical_params`.
"""
Base.@kwdef struct RunConfig <: AbstractConfigSection
    run::RunSection = RunSection()
    output::OutputSection = OutputSection()
    grid::GridSection = GridSection()
    boundary_conditions::BoundaryConditionsSection = BoundaryConditionsSection()
    timestep::TimestepSection = TimestepSection()
    reference_state::ReferenceStateSection = ReferenceStateSection()
    initial_conditions::InitialConditions = InitialConditions()
    environment::Environment = Environment()
    physics::Physics = Physics()
    numerics::Numerics = Numerics()
    diagnostics::Diagnostics = Diagnostics()
    extra::Extra = Extra()

    function RunConfig(run, output, grid, boundary_conditions, timestep, reference_state, initial_conditions,
                       environment, physics, numerics, diagnostics, extra)
        # Paths first (the checks below never look at them), then the checks, then the
        # coerced extra. A config that got this far is well formed; whether it is
        # BUILDABLE is `validate`'s question (src/config_build.jl).
        base = pwd()
        output = _with(output; dir = _resolve_config_path(output.dir, base))
        reference_state = _with(reference_state;
            sounding = _resolve_config_path(reference_state.sounding, base))
        ic_file = initial_conditions.file
        initial_conditions = _with(initial_conditions; file = _with(ic_file;
            paths = String[_resolve_config_path(p, base) for p in ic_file.paths]))
        cfg = new(run, output, grid, boundary_conditions, timestep, reference_state, initial_conditions,
                  environment, physics, numerics, diagnostics, _checked_extra(extra))
        _check_enums(cfg, "")
        _check_equation_set(cfg)
        _check_bc_override(cfg.boundary_conditions.override)
        return cfg
    end
end

# ── Field-wise equality and hashing ───────────────────────────────────────────────
#
# The default `==` of an immutable struct falls back to `===`, which compares Vector
# fields by IDENTITY: two configs loaded from the same file would compare unequal. The
# round-trip test and `--resume`'s "is this the same run" check both need VALUE equality.

function Base.:(==)(a::T, b::T) where {T<:AbstractConfigSection}
    for f in fieldnames(T)
        getfield(a, f) == getfield(b, f) || return false
    end
    return true
end

function Base.hash(x::T, h::UInt) where {T<:AbstractConfigSection}
    h = hash(T, h)
    for f in fieldnames(T)
        h = hash(getfield(x, f), h)
    end
    return h
end

"Copy of section `x` with the given fields replaced (the structs are immutable)."
_with(x::T; kw...) where {T<:AbstractConfigSection} =
    T(; (f => (haskey(kw, f) ? kw[f] : getfield(x, f)) for f in fieldnames(T))...)

# ── Enum checks ───────────────────────────────────────────────────────────────────
#
# (section type, field) → allowed values. Where `src/` already defines the set, the SAME
# tuple is referenced, so the config and the model can never disagree about what a valid
# `:sfc_z0` is. Vector-valued fields (`output.formats`) are checked element by element.

"`(section type, field) => allowed values` for every enum-valued config field."
const ENUM_FIELDS = Dict{Tuple{DataType,Symbol},Tuple}(
    (OutputSection, :formats)                  => OUTPUT_FORMAT_SYMBOLS,
    (OutputSection, :netcdf_grid)              => NETCDF_GRIDS,
    (BoundaryConditionsSection, :lateral)          => LATERAL_BCS,
    (BoundaryConditionsSection, :vertical_scalars) => VERTICAL_SCALAR_BCS,
    (BoundaryConditionsSection, :vertical_scalars_lid) => VERTICAL_SCALAR_BCS,
    (BoundaryConditionsSection, :sedimentation_lid) => SEDIMENTATION_LID_BCS,
    (InitialConditions, :method)               => IC_METHODS,
    (VortexIC, :profile)                       => VORTEX_PROFILES,
    (VortexIC, :moist_profile)                 => VORTEX_MOIST_PROFILES,
    (Microphysics, :ice)                       => ICE_SCHEMES,
    (Microphysics, :rain_moments)              => RAIN_MOMENT_CHOICES,
    (Microphysics, :condensate_transform)      => WATER_TRANSFORMS,
    (Microphysics, :rain_transform)            => WATER_TRANSFORMS,
    (Microphysics, :rain_number_transform)     => WATER_TRANSFORMS,
    (Microphysics, :ice_transform)             => WATER_TRANSFORMS,
    (BoundaryLayer, :scheme)                   => BL_SCHEMES,
    (BoundaryLayer, :mynn_init)                => MYNN_INIT_MODES,
    (Surface, :roughness)                      => SFC_Z0_MODES,
    (Radiation, :scheme)                       => RADIATION_SCHEMES,
    (Radiation, :method)                       => RADIATION_METHODS,
    (Radiation, :solar)                        => RADIATION_SOLAR_MODES,
    (Radiation, :forcing)                      => RADIATION_FORCINGS,
    (HorizontalMixing, :scheme)                => HMIX_SCHEMES,
    (Numerics, :positivity)                    => POSITIVITY_MODES,
    (Numerics, :temperature)                   => TEMPERATURE_CHOICES)

_join_path(path::AbstractString, f) = isempty(path) ? string(f) : string(path, ".", f)

function _check_enums(x::T, path::AbstractString) where {T<:AbstractConfigSection}
    for f in fieldnames(T)
        v = getfield(x, f)
        p = _join_path(path, f)
        if v isa AbstractConfigSection
            _check_enums(v, p)
        elseif haskey(ENUM_FIELDS, (T, f))
            allowed = ENUM_FIELDS[(T, f)]
            for e in (v isa AbstractVector ? v : (v,))
                e in allowed || error("config: $p = $(repr(e)) is not recognized; use " *
                                      join(repr.(allowed), ", ", " or "))
            end
        end
    end
    T === BoundaryLayer && _check_mynn_fidelity(x.mynn_fidelity, path)
    return nothing
end

# `mynn_fidelity` is a String in the struct (it may be a comma list), parsed into a Symbol
# or a Vector{Symbol} by the same `parse_mynn_fidelity` the NetCDF reader uses. Checking
# the parsed members here means a misspelled deviation dies at load, not at the first
# MYNN call.
function _check_mynn_fidelity(s::AbstractString, path::AbstractString)
    parsed = parse_mynn_fidelity(s)
    allowed = (MYNN_FIDELITIES..., MYNN_DEVIATIONS...)
    for e in (parsed isa AbstractVector ? parsed : (parsed,))
        e in allowed || error("config: $(_join_path(path, :mynn_fidelity)) = $(repr(s)): " *
                              "$(repr(e)) is not recognized; use " *
                              join(repr.(allowed), ", ", " or ") * " (comma-separated)")
    end
    return nothing
end

# The side and value of every override entry are checked here (the file is malformed
# otherwise); whether the VARIABLE exists depends on the physics, so that is
# `build_boundary_conditions`'s check, made against the run's own variable list.
function _check_bc_override(ov::AbstractDict)
    for (var, sides) in ov, (side, val) in sides
        where = "boundary_conditions.override.$var.$side"
        side in BC_OVERRIDE_SIDES || error("config: $where: side $(repr(side)) is not " *
            "recognized; use " * join(repr.(BC_OVERRIDE_SIDES), ", ", " or "))
        val in BC_OVERRIDE_VALUES || error("config: $where = $(repr(val)) is not " *
            "recognized; use " * join(repr.(BC_OVERRIDE_VALUES), ", ", " or "))
    end
    return nothing
end

function _check_equation_set(cfg::RunConfig)
    es = cfg.run.equation_set
    haskey(EQUATION_SETS, es) || error("config: run.equation_set = $(repr(es)) is not " *
        "recognized; use " * join(repr.(sort(collect(keys(EQUATION_SETS)))), ", ", " or "))
    geoms = EQUATION_SETS[es].geometries
    cfg.grid.geometry in geoms || error("config: grid.geometry = $(repr(cfg.grid.geometry)) " *
        "is not a geometry of run.equation_set = $(repr(es)); use " *
        join(repr.(geoms), ", ", " or "))
    return nothing
end

# ── Key registries ────────────────────────────────────────────────────────────────
#
# Every key `src/` reads from `options` / `physical_params`, with the type its read site
# expects and where it is read. NO DEFAULTS: the same key defaults differently per
# equation set (`:precipitation` is true in primitive_equations.jl and false in
# moist_compressible.jl), so a registry default would be wrong for someone. The registry
# answers only "is this a real key, and what type does the code expect".
#
# test/test_config.jl greps `src/*.jl` for `get(…options, :key` / `get(…physical_params,
# :key` (and the `opts`/`pp` aliases) and fails on any key missing here, so a new read
# cannot land without a line in this table. The `OWNED_*` tables name the keys a typed
# section produces; those are refused in `[extra.*]` so there is exactly one way to set
# each of them.

"""
    KNOWN_OPTION_KEYS

Every `options` key read in `src/`: key → `(expected type, one-line description with the
reading file)`. No defaults (they differ per equation set).
"""
const KNOWN_OPTION_KEYS = Dict{Symbol,Tuple{Type,String}}(
    # ── solver / reference state
    :semiimplicit             => (Bool, "legacy equation sets: semi-implicit time stepping on/off; always on for moist_compressible (primitive_equations.jl, semiimplicit.jl)"),
    :exact_reference_state    => (Bool, "read the pre-balanced exact reference-state file (semiimplicit.jl)"),
    :consistent_qss_reference => (Bool, "reference Q_ss rebuilt consistently with the discrete retrieval (semiimplicit.jl, moist_compressible.jl)"),
    :hydrostatic_reference    => (Bool, "discretely hydrostatic reference state (semiimplicit.jl)"),
    :state_dependent_si       => (Bool, "vertical SI implicit coefficients from the current column state (moist_compressible.jl)"),
    :exact_si                 => (Bool, "exact coupled semi-implicit acoustic solve, variant 1 (exact_si.jl, moist_compressible.jl)"),
    :temperature              => (Symbol, "model temperature: :eos = p/(rho_d R_d + rho_v R_v) (default) or :energy = the closed-form total-energy retrieval (moist_compressible.jl, radiation.jl, semiimplicit.jl, netcdf_output.jl)"),
    :exact_si_ax0             => (Bool, "exact_si axisym: experimental axis-row treatment of the radial operator (exact_si.jl)"),
    :exact_si_zero_x          => (Bool, "exact_si diagnostic: zero the horizontal increments (exact_si.jl)"),
    :xsi_lid_rows             => (String, "exact_si lid boundary rows: \"strong\" or weak (exact_si.jl, exact_si_rlr.jl)"),
    :xsi_u_history            => (String, "exact_si u-leg history: \"stored\" or \"none\" (exact_si.jl)"),
    :xsi_rlr_experimental     => (Bool, "allow the WIP exact_si RLR path at wavenumber >= 1 (exact_si.jl, exact_si_rlr.jl)"),
    :xsi_rlr_axis_dirichlet   => (Bool, "exact_si RLR: Dirichlet axis rows in the per-wavenumber solve (exact_si_rlr.jl)"),
    :xsi_rlr_no_v             => (Bool, "exact_si RLR isolation lever: drop the p-v azimuthal coupling (exact_si_rlr.jl)"),
    :horizontal_semiimplicit  => (Bool, "horizontal semi-implicit acoustic terms, XZ RiRk only (horizontal_si.jl, exact_si.jl)"),
    :hsi_scheme               => (String, "horizontal SI weights: \"ai2s\" or \"am2\" (horizontal_si.jl)"),
    :hsi_u_history            => (String, "horizontal SI u history: \"none\" or \"stored\" (horizontal_si.jl)"),
    :hsi_x_history            => (String, "horizontal SI x history: \"fresh\" or lagged (horizontal_si.jl, moist_compressible.jl)"),
    :hsi_cross_comp           => (Bool, "horizontal SI factorization cross-term compensation, low-Courant experiments only (horizontal_si.jl)"),
    :wall_bc_tau              => (Float64, "relaxation timescale [s] of the R1T1X wall BC; Inf = none (semiimplicit.jl)"),
    :cfl_interval             => (Float64, "CFL report cadence [s]; default output_interval (semiimplicit.jl, nesting.jl)"),
    :vertical_mixing          => (Bool, "legacy primitive equations: vertical mixing on/off (primitive_equations.jl)"),
    :noise_seed               => (Int, "Twoway_PV_mixing stochastic forcing seed (shallowWaterModels.jl)"),
    # ── microphysics and water
    :precipitation            => (Bool, "rain production and sedimentation (moist_compressible.jl, primitive_equations.jl)"),
    :condensation             => (Bool, "phase change on/off: QSS condensation and the ice transfers (moist_compressible.jl)"),
    :rain_moments             => (Int, "1 = single-moment rain, 2 = prognostic rain number n_r (moist_compressible.jl)"),
    :condensate_transform     => (Symbol, "cloud-slot control variable: :none, :bhyp, :bhyp_smooth (moist_compressible.jl)"),
    :rain_transform           => (Symbol, "rain-slot control variable: :none, :bhyp, :bhyp_smooth (moist_compressible.jl)"),
    :rain_number_transform    => (Symbol, "rain-number-slot control variable: :none, :bhyp, :bhyp_smooth (moist_compressible.jl)"),
    :ice_transform            => (Symbol, "control variable of the twelve ice slots: :none, :bhyp, :bhyp_smooth (moist_compressible.jl)"),
    :condensate_floor         => (Symbol, "cloud flooring at the thermodynamic interface: :none or :diagnostic (moist_compressible.jl)"),
    :clamp_water              => (Bool, "floor negative untransformed water at zero after each step (moist_compressible.jl)"),
    :water_warn_dT            => (Float64, "temperature-mismatch warning threshold [K] of the water reconciliation (moist_compressible.jl)"),
    :rain_evap_realization    => (Bool, "per-donor realization bound on rain evaporation (moist_compressible.jl)"),
    :cloud_evap_realization   => (Bool, "donor bound on cloud evaporation: the step-mean draw applied to p, Q_ss, the vapor and the cloud slot never exceeds the cloud present; default true, false restores the unbounded draw (moist_compressible.jl)"),
    :rain_population_gate     => (Bool, "skip two-moment rain sinks where the carried rain number is not a population (moist_compressible.jl)"),
    :water_lobe_attribution   => (Bool, "the vapor residual and the ice anchor headroom read each transformed water-mass slot's clipped negative lobe through the odd continuation of ahyp; default FALSE until the slots' below-zero transport is consistent with rho_t (measured 2026-10-07: the slots hold a 4-6.5 kg/m2 negative pool rho_t does not, so attributing it inflates the vapor by ~10% of the column); true enables it (moist_compressible.jl)"),
    :lobe_relaxation          => (Bool, "relax a transformed water-mass slot's (cloud, rain, the three ice masses) refit lobe nu < 0 toward 0 on physical_params[:tau_lobe] where that species' channel is closed (recovered density 0, every process term of the slot exactly 0); moves no recovered mass, water-neutral while rho_t is prognostic; a pragmatic device, not physics (Stage L, reference/RHO_T_RETIREMENT_DESIGN.md); refused with water_lobe_attribution (moist_compressible.jl)"),
    :water_lobe_transport     => (Bool, "below nu = 0 the continuity terms of each transformed water-mass slot (cloud, rain, ice masses) read the odd continuation ahyp_signed and its Jacobian, so the slots are transported consistently with the linear rho_t; default true, false restores the clipped-point forms (J = 1, rho = 0) for forensics (moist_compressible.jl)"),
    :sedimentation_flux_lid   => (Symbol, "lid of every sedimentation-flux fit: :dirichlet (default, F = 0 at the lid) or :natural (forensics: the slot's own lid BC, lets hydrometeors fall in from above) (moist_compressible.jl)"),
    :ice_microphysics         => (Symbol, "ice scheme: :ishmael or :none (moist_compressible.jl, radiation_state.jl, mc_mynn_bl.jl)"),
    :ice_agg_caps             => (Bool, "ISHMAEL aggregation caps on the unit pass (moist_compressible.jl)"),
    :ice_anchor_floor         => (Bool, "ice anchor reconciliation: floor leg (moist_compressible.jl)"),
    :ice_anchor_flux          => (Bool, "ice anchor reconciliation: flux leg (moist_compressible.jl)"),
    :ice_anchor_rates         => (Bool, "ice anchor reconciliation: rates leg (moist_compressible.jl)"),
    :ice_anchor_source        => (Bool, "ice anchor removal source (the census is kept when off) (moist_compressible.jl)"),
    :ice_attr_census          => (Bool, "ice attribution census in the log (moist_compressible.jl)"),
    :ice_number_realization   => (Bool, "per-donor realization bound on the ice-number sinks (moist_compressible.jl)"),
    :ice_population_seed      => (Symbol, "crystal seeding of number-less ice below T_0: :local or :large (moist_compressible.jl)"),
    :ice_population_source    => (Bool, "ice population source for number-less ice (moist_compressible.jl)"),
    :ice_shed_above_t0        => (Bool, "shed riming to rain above T_0 (moist_compressible.jl)"),
    :ice_var_check            => (Bool, "ISHMAEL var_check write-back as a relaxation source (moist_compressible.jl)"),
    # ── boundary layer and surface
    :louis_bl                 => (Bool, "Louis first-order vertical mixing (moist_compressible.jl, mc_surface_layer.jl)"),
    :mynn                     => (Bool, "MYNN-EDMF boundary layer (mynn_state.jl, mc_surface_layer.jl, netcdf_output.jl)"),
    :mynn_interval            => (Float64, "MYNN call cadence [s] (mynn_state.jl)"),
    :mynn_edmf                => (Int, "MYNN mass flux: 0 off, 1 on (mynn_state.jl)"),
    :mynn_edmf_mom            => (Bool, "EDMF momentum transport (mynn_state.jl)"),
    :mynn_closure             => (Float64, "MYNN closure level; only 2.5 is ported (mynn_state.jl)"),
    :mynn_scale_aware         => (Bool, "MYNN scale-aware mixing length (mynn_state.jl)"),
    :mynn_init                => (Symbol, "MYNN TKE initialisation: :taper or :zero (mynn_state.jl)"),
    :mynn_fidelity            => (Union{Symbol,Vector{Symbol}}, "MYNN closure fidelity: :scythe, :fortran or a Vector of deviations (mynn_state.jl)"),
    :mynn_water_carry         => (Symbol, "MYNN water carry: :flux or :fixed_T (mynn_state.jl)"),
    :mynn_mix_numbers         => (Bool, "MYNN also mixes the number densities (mynn_state.jl)"),
    :mynn_output              => (Bool, "per-tile MYNN NetCDF sidecar (mynn_state.jl)"),
    :mynn_trace               => (Bool, "MYNN per-call summary line (mynn_state.jl)"),
    :mynn_check_values        => (Bool, "MYNN finite-value checks every call (mynn_state.jl)"),
    :surface_fluxes           => (Bool, "surface momentum/heat/moisture fluxes (moist_compressible.jl, mc_surface_layer.jl)"),
    :sfc_z0                   => (Symbol, "neutral roughness model: :komori, :gfdl_v7, :charnock (mc_surface_layer.jl)"),
    :sfc_stability            => (Bool, "Monin-Obukhov stability correction (mc_surface_layer.jl, mynn_state.jl)"),
    :sfc_stability_iterations => (Int, "fixed-point iterations of the stability solve, >= 1 (mc_surface_layer.jl)"),
    # ── radiation
    :radiation                => (Symbol, "radiation scheme: :none, :prescribed, :rrtmgp (radiation.jl, radiation_state.jl)"),
    :radiation_method         => (Symbol, "RRTMGP method: :gray, :clearsky, :allsky, :allsky_clear (radiation_state.jl)"),
    :radiation_forcing        => (Symbol, "heating applied: :full or :anomaly (radiation_state.jl)"),
    :radiation_interval       => (Float64, "radiation call cadence [s] (radiation_state.jl, radiation.jl)"),
    :radiation_z_max          => (Float64, "top of the radiative heating taper [m]; Inf = no taper (radiation_state.jl)"),
    :radiation_sw_rescale     => (Bool, "rescale held SW heating by the current cos(zenith) between calls (radiation_state.jl)"),
    :radiation_level_interp   => (Symbol, "interface-level construction: :hydrostatic or :bestfit (radiation_state.jl)"),
    :radiation_extension      => (Symbol, "above-lid profile extension: :tropical, :midlatitude_summer, :none (radiation_state.jl)"),
    :radiation_extension_layers => (Int, "number of above-lid extension layers (radiation_state.jl)"),
    :radiation_layer_stride   => (Int, "model gridpoints per radiation layer (radiation_state.jl, radiation.jl)"),
    :radiation_rain_in_cloud  => (Bool, "rain counted in the cloud optics (radiation_state.jl, radiation_cloud_optics.jl)"),
    :radiation_gases          => (Dict{String,Float64}, "override of the well-mixed gas volume mixing ratios (radiation.jl)"),
    :radiation_output         => (Bool, "per-tile radiation NetCDF sidecar (radiation_state.jl)"),
    :radiation_check_values   => (Bool, "radiation finite-value checks every call (radiation_state.jl)"),
    :radiation_trace          => (Bool, "radiation per-call summary line (radiation.jl)"),
    :radiation_trace_sw       => (Bool, "shortwave detail in the radiation trace (radiation.jl)"),
    :solar                    => (Symbol, "shortwave forcing: :none, :fixed, :diurnal (radiation_state.jl)"),
    # ── output
    :output_formats           => (Vector{Symbol}, "analysis formats: :netcdf, :csv, :netcdf_raw (io.jl, netcdf_output.jl)"),
    :netcdf_grid              => (Symbol, "NetCDF output grid: :regular (netcdf_output.jl)"),
    :netcdf_derivatives       => (Bool, ":netcdf_raw also writes derivative slots (io.jl, netcdf_output.jl)"),
    # ── diagnostics and the timestep advisory
    :state_minima_trace       => (Int, "state-minima print interval [steps]; 0 = off (semiimplicit.jl)"),
    :column_state_guard       => (Bool, "located error when p, rho_d, rho_t or T is not positive/finite after the retrieval; default true, false is forensics only (moist_compressible.jl)"),
    :water_budget_trace       => (Int, "water-budget print interval [steps]; 0 = off (moist_compressible.jl)"),
    :stiffness_trace          => (Int, "stiffness-census print interval [steps]; 0 = off (moist_compressible.jl)"),
    :u_max                    => (Float64, "timestep advisory: expected peak horizontal wind [m/s] (reference_state.jl)"),
    :w_max                    => (Float64, "timestep advisory: expected peak vertical wind [m/s] (reference_state.jl)"),
    :bl_K_ref                 => (Float64, "timestep advisory: reference BL eddy diffusivity [m^2/s] (reference_state.jl)"),
    :state_deviation          => (Float64, "timestep advisory: convective-ceiling target delta (reference_state.jl)"))

"""
    RETIRED_OPTION_KEYS

Option keys `src/` still looks for only to refuse them (tombstones): key → why. Refused in
`[extra.options]` with the reason, and allow-listed by the registry grep test.
"""
const RETIRED_OPTION_KEYS = Dict{Symbol,String}(
    :vapor_retrieval => "retired: rho_v is prognostic (moist_compressible.jl tombstone)")

"""
    KNOWN_PHYSICAL_KEYS

Every `physical_params` key read in `src/`: key → `(expected type, one-line description
with the reading file)`. `physical_params` is a `Dict{Symbol,Float64}`, so every settable
key is a number. No defaults (they differ per equation set).
"""
const KNOWN_PHYSICAL_KEYS = Dict{Symbol,Tuple{Type,String}}(
    # ── mixing
    :Khdiff        => (Float64, "horizontal momentum eddy viscosity [m^2/s]; required key (moist_compressible.jl, primitive_equations.jl)"),
    :Kvdiff        => (Float64, "vertical momentum eddy viscosity [m^2/s]; required key (moist_compressible.jl, exact_si.jl)"),
    :Khdiff_heat   => (Float64, "horizontal heat diffusivity [m^2/s]; < 0 = Smagorinsky K/Pr_t (moist_compressible.jl)"),
    :Kvdiff_heat   => (Float64, "vertical heat diffusivity [m^2/s]; default Kvdiff (moist_compressible.jl, exact_si.jl)"),
    :Khdiff_water  => (Float64, "horizontal water diffusivity [m^2/s], every water species and ice moment; < 0 = Smagorinsky K/Sc_t (moist_compressible.jl)"),
    :Kvdiff_water  => (Float64, "vertical water diffusivity [m^2/s]; refused with ice and two-moment rain (moist_compressible.jl, exact_si.jl)"),
    :Pr_t          => (Float64, "turbulent Prandtl number for Smagorinsky heat mixing (moist_compressible.jl)"),
    :Sc_t          => (Float64, "turbulent Schmidt number for Smagorinsky water mixing (moist_compressible.jl)"),
    :Ls            => (Float64, "Smagorinsky length scale [m]; > 0 = this absolute length on every patch, 0 = C_s * each patch's cell width (moist_compressible.jl)"),
    :C_s           => (Float64, "Smagorinsky constant; read only when Ls is 0 or absent, then > 0 turns Smagorinsky on with length C_s * (patch i-cell width) (moist_compressible.jl)"),
    :K_min         => (Float64, "Smagorinsky viscosity floor [m^2/s] (moist_compressible.jl)"),
    :l_inf         => (Float64, "Louis asymptotic mixing length [m] (moist_compressible.jl)"),
    :mynn_K_max    => (Float64, "counted safety cap on the MYNN exchange coefficient [m^2/s] (mynn_state.jl)"),
    # ── sponge, rotation, geometry
    :alpha         => (Float64, "Rayleigh sponge coefficient [1/s] (moist_compressible.jl, primitive_equations.jl)"),
    :z_damp        => (Float64, "bottom of the Rayleigh sponge [m] (moist_compressible.jl, primitive_equations.jl)"),
    :f             => (Float64, "Coriolis parameter [1/s] (moist_compressible.jl, primitive_equations.jl)"),
    :Omega         => (Float64, "rotation rate [1/s] for the RRR spherical-shell Coriolis (mc_geometry.jl)"),
    :sphere_radius => (Float64, "shallow-atmosphere metric radius [m] (mc_geometry.jl)"),
    # ── surface
    :SST           => (Float64, "sea-surface temperature [K] (moist_compressible.jl, mc_surface_layer.jl)"),
    :Cd            => (Float64, "drag coefficient; negative = Komori wind-dependent drag (mc_surface_layer.jl, primitive_equations.jl)"),
    :Ck            => (Float64, "enthalpy/moisture exchange coefficient (mc_surface_layer.jl, primitive_equations.jl)"),
    :U_min         => (Float64, "gustiness floor on the surface wind speed [m/s] (mc_surface_layer.jl)"),
    :sfc_wind_factor => (Float64, "multiplier from the lowest-level wind to the 10 m wind (mc_surface_layer.jl)"),
    # ── microphysics
    :tau_qss       => (Float64, "QSS relaxation timescale [s] (moist_compressible.jl)"),
    :tau_rho_v_rec => (Float64, "vapor reconciliation timescale [s] (moist_compressible.jl)"),
    :N_0           => (Float64, "rain DSD intercept [1/m^4] (moist_compressible.jl, netcdf_output.jl)"),
    :N_r           => (Float64, "single-moment rain number parameter (moist_compressible.jl)"),
    :condensate_mu => (Float64, "bhyp width for cloud [kg/m^3] (moist_compressible.jl, netcdf_output.jl)"),
    :rain_mu       => (Float64, "bhyp width for rain [kg/m^3] (moist_compressible.jl, netcdf_output.jl)"),
    :mu_rain_n     => (Float64, "bhyp width for rain number [1/m^3] (moist_compressible.jl, netcdf_output.jl)"),
    :max_N_c       => (Float64, "cloud droplet number [1/cm^3] for ISHMAEL autoconversion (moist_compressible.jl, netcdf_output.jl)"),
    :tau_activation   => (Float64, "ice activation timescale [s] (moist_compressible.jl)"),
    :tau_homogeneous  => (Float64, "homogeneous freezing timescale [s] (moist_compressible.jl)"),
    :tau_varcheck     => (Float64, "ISHMAEL var_check relaxation timescale [s] (moist_compressible.jl)"),
    :tau_ice_anchor   => (Float64, "ice anchor removal timescale [s] (moist_compressible.jl)"),
    :tau_lobe         => (Float64, "lobe relaxation timescale [s] of options[:lobe_relaxation]; default 10 (moist_compressible.jl)"),
    :tau_ice_population => (Float64, "ice population source timescale [s] (moist_compressible.jl)"),
    :ishmael_tables_path => (String, "ISHMAEL collection tables file; NOT settable: physical_params is Dict{Symbol,Float64} (semiimplicit.jl)"),
    # ── radiation
    :latitude      => (Float64, "latitude for the solar geometry [deg] (radiation_state.jl, radiation.jl)"),
    :longitude     => (Float64, "longitude for the solar geometry [deg] (radiation_state.jl)"),
    :start_doy     => (Float64, "day of year at t = 0 (radiation_state.jl)"),
    :start_hour    => (Float64, "UTC hour at t = 0 (radiation_state.jl)"),
    :solar_constant => (Float64, "solar constant [W/m^2] (radiation_state.jl)"),
    :cos_zenith    => (Float64, "fixed-sun cosine of the zenith angle (radiation_state.jl)"),
    :sw_toa_flux   => (Float64, "fixed-sun TOA shortwave flux [W/m^2] (radiation_state.jl)"),
    :sfc_albedo    => (Float64, "surface albedo (radiation_state.jl, radiation.jl)"),
    :sfc_emissivity => (Float64, "surface emissivity (radiation_state.jl, radiation.jl)"),
    :T_sfc         => (Float64, "radiative surface temperature [K]; default the SST (radiation.jl)"),
    :radiation_prescribed_rate => (Float64, "prescribed radiative heating [K/day], negative = cooling (radiation.jl)"),
    :radiation_q_min => (Float64, "cloud-optics mixing-ratio threshold [kg/kg] (radiation_cloud_optics.jl)"),
    :cloud_k_factor  => (Float64, "Martin et al. effective-radius k factor (radiation_cloud_optics.jl)"),
    :co2_ppm       => (Float64, "CO2 volume mixing ratio [ppm] (radiation_rrtmgp.jl)"),
    :ch4_ppb       => (Float64, "CH4 volume mixing ratio [ppb] (radiation_rrtmgp.jl)"),
    :n2o_ppb       => (Float64, "N2O volume mixing ratio [ppb] (radiation_rrtmgp.jl)"),
    :o2_vmr        => (Float64, "O2 volume mixing ratio (radiation_rrtmgp.jl)"),
    :n2_vmr        => (Float64, "N2 volume mixing ratio (radiation_rrtmgp.jl)"),
    :cfc11_ppt     => (Float64, "CFC-11 volume mixing ratio [ppt] (radiation_rrtmgp.jl)"),
    :cfc12_ppt     => (Float64, "CFC-12 volume mixing ratio [ppt] (radiation_rrtmgp.jl)"),
    # ── legacy equation sets (shallow water, TC boundary layer, primitive equations, test models)
    :c_0           => (Float64, "linear advection speed [m/s] (testModels.jl)"),
    :g             => (Float64, "gravity [m/s^2] (shallowWaterModels.jl, primitive_equations.jl)"),
    :h             => (Float64, "slab boundary-layer depth [m] (tcblModels.jl)"),
    :H             => (Float64, "shallow-water mean depth [m] (shallowWaterModels.jl)"),
    :Hb            => (Float64, "two-way model boundary-layer depth [m] (shallowWaterModels.jl)"),
    :Hfree         => (Float64, "two-way model free-atmosphere depth [m] (shallowWaterModels.jl)"),
    :K             => (Float64, "shallow-water eddy viscosity [m^2/s] (shallowWaterModels.jl)"),
    :Kh            => (Float64, "legacy horizontal diffusivity [m^2/s] (shallowWaterModels.jl, tcblModels.jl, primitive_equations.jl)"),
    :Kv_mudiff     => (Float64, "primitive equations: vertical mu diffusivity (primitive_equations.jl)"),
    :K_min_bl      => (Float64, "PV-mixing model: boundary-layer viscosity floor [m^2/s] (shallowWaterModels.jl)"),
    :K_min_free    => (Float64, "PV-mixing model: free-atmosphere viscosity floor [m^2/s] (shallowWaterModels.jl)"),
    :Ls_bl         => (Float64, "PV-mixing model: boundary-layer mixing length [m] (shallowWaterModels.jl)"),
    :Ls_free       => (Float64, "PV-mixing model: free-atmosphere mixing length [m] (shallowWaterModels.jl)"),
    :S1            => (Float64, "two-way model heating amplitude (shallowWaterModels.jl)"),
    :S1_sigma      => (Float64, "PV-mixing model: stochastic heating amplitude (shallowWaterModels.jl)"),
    :sst           => (Float64, "primitive equations: sea-surface temperature [K] (primitive_equations.jl)"),
    :Um            => (Float64, "translation speed, x [m/s] (shallowWaterModels.jl, primitive_equations.jl)"),
    :Vm            => (Float64, "translation speed, y [m/s] (shallowWaterModels.jl, primitive_equations.jl)"),
    :z_ref_level   => (Float64, "primitive equations: surface-layer reference height [m] (primitive_equations.jl)"))

"""
    OWNED_OPTION_KEYS

`options` keys a typed section produces at build: key → the config field that owns it.
Refused in `[extra.options]`, so there is exactly one way to set each.
"""
const OWNED_OPTION_KEYS = Dict{Symbol,String}(
    :semiimplicit              => "(always on for moist_compressible)",
    :exact_reference_state     => "reference_state.exact",
    :consistent_qss_reference  => "reference_state.consistent_qss",
    :hydrostatic_reference     => "reference_state.hydrostatic",
    :ice_microphysics          => "physics.microphysics.ice",
    :rain_moments              => "physics.microphysics.rain_moments",
    :precipitation             => "physics.microphysics.precipitation",
    :condensation              => "physics.microphysics.condensation",
    :condensate_transform      => "physics.microphysics.condensate_transform",
    :rain_transform            => "physics.microphysics.rain_transform",
    :rain_number_transform     => "physics.microphysics.rain_number_transform",
    :ice_transform             => "physics.microphysics.ice_transform",
    :mynn                      => "physics.boundary_layer.scheme",
    :louis_bl                  => "physics.boundary_layer.scheme",
    :vertical_mixing           => "physics.boundary_layer.scheme",
    :mynn_interval             => "physics.boundary_layer.mynn_interval",
    :mynn_edmf                 => "physics.boundary_layer.mynn_edmf",
    :mynn_fidelity             => "physics.boundary_layer.mynn_fidelity",
    :mynn_init                 => "physics.boundary_layer.mynn_init",
    :surface_fluxes            => "physics.surface.fluxes",
    :sfc_z0                    => "physics.surface.roughness",
    :sfc_stability             => "physics.surface.stability",
    :radiation                 => "physics.radiation.scheme",
    :radiation_method          => "physics.radiation.method",
    :radiation_forcing         => "physics.radiation.forcing",
    :radiation_interval        => "physics.radiation.interval",
    :radiation_z_max           => "physics.radiation.z_max",
    :solar                     => "physics.radiation.solar",
    :state_dependent_si        => "numerics.state_dependent_si",
    :exact_si                  => "numerics.exact_si",
    :horizontal_semiimplicit   => "numerics.horizontal_semiimplicit",
    :wall_bc_tau               => "numerics.wall_bc_tau",
    :cfl_interval              => "numerics.cfl_interval",
    :temperature               => "numerics.temperature",
    :output_formats            => "output.formats",
    :netcdf_grid               => "output.netcdf_grid",
    :netcdf_derivatives        => "output.netcdf_derivatives",
    :mynn_output               => "output.sidecars",
    :radiation_output          => "output.sidecars",
    :state_minima_trace        => "diagnostics.state_minima_trace",
    :water_budget_trace        => "diagnostics.water_budget_trace",
    :stiffness_trace           => "diagnostics.stiffness_trace",
    :mynn_trace                => "diagnostics.mynn_trace",
    :radiation_trace           => "diagnostics.radiation_trace",
    :radiation_trace_sw        => "diagnostics.radiation_trace_sw",
    :u_max                     => "diagnostics.timestep_advisory.u_max",
    :w_max                     => "diagnostics.timestep_advisory.w_max",
    :bl_K_ref                  => "diagnostics.timestep_advisory.bl_K_ref",
    :state_deviation           => "diagnostics.timestep_advisory.state_deviation")

"""
    OWNED_PHYSICAL_KEYS

`physical_params` keys a typed section produces at build: key → the config field that owns
it. Refused in `[extra.physical_params]`. `:Kvdiff`, `:Kvdiff_heat` and `:Kvdiff_water` are
deliberately NOT owned: the build writes them as 0.0 (vertical mixing belongs to the BL
scheme) and `[extra.physical_params]` is the documented way to override them.
"""
const OWNED_PHYSICAL_KEYS = Dict{Symbol,String}(
    :Khdiff         => "physics.horizontal_mixing.K",
    :Khdiff_heat    => "physics.horizontal_mixing.heat",
    :Khdiff_water   => "physics.horizontal_mixing.water",
    :Pr_t           => "physics.horizontal_mixing.Pr_t",
    :Sc_t           => "physics.horizontal_mixing.Sc_t",
    :Ls             => "physics.horizontal_mixing.Ls",
    :C_s            => "physics.horizontal_mixing.C_s",
    :K_min         => "physics.horizontal_mixing.K_min",
    :l_inf          => "physics.boundary_layer.l_inf",
    :mynn_K_max     => "physics.boundary_layer.K_max",
    :alpha          => "physics.sponge.alpha",
    :z_damp         => "physics.sponge.z_damp",
    :f              => "environment.f_latitude",
    :SST            => "environment.sst",
    :Cd             => "physics.surface.Cd",
    :Ck             => "physics.surface.Ck",
    :U_min          => "physics.surface.U_min",
    :tau_qss        => "physics.microphysics.tau_qss",
    :N_r            => "physics.microphysics.N_r",
    :N_0            => "physics.microphysics.N_0",
    :condensate_mu  => "physics.microphysics.condensate_mu",
    :rain_mu        => "physics.microphysics.rain_mu",
    :latitude       => "physics.radiation.latitude",
    :start_doy      => "physics.radiation.start_doy",
    :start_hour     => "physics.radiation.start_hour",
    :sfc_albedo     => "physics.radiation.albedo",
    :sfc_emissivity => "physics.radiation.emissivity")

# ── Value coercion ────────────────────────────────────────────────────────────────
#
# TOML has no Symbol, writes `inf` for Inf, and cannot tell a caller's `300` from `300.0`.
# `_coerce(T, v, where)` turns a parsed TOML value into the Julia type the struct field or
# registry entry wants, and errors naming the full key (`physics.radiation.interval`)
# otherwise. `Bool <: Integer` in Julia, so every numeric branch excludes it explicitly:
# `alpha = true` must be an error, not 1.0.

_is_number(v) = v isa Real && !(v isa Bool)

function _parse_inf(v::AbstractString)
    s = lowercase(strip(v))
    s in ("inf", "+inf") && return Inf
    s == "-inf" && return -Inf
    return nothing
end

_type_error(where, v, what) =
    error("config: $where = $(repr(v)) has the wrong type; expected $what")

function _coerce(::Type{Float64}, v, where)
    _is_number(v) && return Float64(v)
    if v isa AbstractString
        x = _parse_inf(v)
        x === nothing || return x
    end
    _type_error(where, v, "a number (\"inf\" allowed)")
end
_coerce(::Type{Int}, v, where) =
    (v isa Integer && !(v isa Bool)) ? Int(v) : _type_error(where, v, "an integer")
_coerce(::Type{Bool}, v, where) = v isa Bool ? v : _type_error(where, v, "true or false")
_coerce(::Type{String}, v, where) =
    (v isa AbstractString || v isa Symbol) ? String(string(v)) : _type_error(where, v, "a string")
_coerce(::Type{Symbol}, v, where) =
    (v isa AbstractString || v isa Symbol) ? Symbol(v) : _type_error(where, v, "a string")
function _coerce(::Type{Vector{T}}, v, where) where {T}
    v isa AbstractVector || _type_error(where, v, "an array")
    return T[_coerce(T, e, "$where[$i]") for (i, e) in enumerate(v)]
end
function _coerce(::Type{Dict{String,Dict{String,String}}}, v, where)
    v isa AbstractDict || _type_error(where, v, "a table of {side = \"bc\"} tables")
    out = Dict{String,Dict{String,String}}()
    for (var, sides) in v
        sides isa AbstractDict || _type_error("$where.$var", sides, "a table {side = \"bc\"}")
        out[string(var)] = Dict{String,String}(string(k) => _coerce(String, x, "$where.$var.$k")
                                               for (k, x) in sides)
    end
    return out
end
function _coerce(::Type{Dict{String,Float64}}, v, where)
    v isa AbstractDict || _type_error(where, v, "a table of numbers")
    return Dict{String,Float64}(string(k) => _coerce(Float64, x, "$where.$k") for (k, x) in v)
end
_coerce(::Type{Union{Symbol,Vector{Symbol}}}, v, where) =
    v isa AbstractVector ? _coerce(Vector{Symbol}, v, where) : _coerce(Symbol, v, where)
_coerce(::Type{T}, v, where) where {T<:AbstractConfigSection} =
    v isa AbstractDict ? from_dict(T, v, where) : _type_error(where, v, "a table [$where]")

# ── Extra: the validated escape hatch ─────────────────────────────────────────────

function _checked_extra(extra::Extra)
    opts = Dict{Symbol,Any}()
    for (k, v) in extra.options
        where = "extra.options.$k"
        haskey(RETIRED_OPTION_KEYS, k) &&
            error("config: $where is refused: $(RETIRED_OPTION_KEYS[k])")
        haskey(KNOWN_OPTION_KEYS, k) || error("config: $where is not an options key " *
            "Scythe reads (KNOWN_OPTION_KEYS); check the spelling")
        haskey(OWNED_OPTION_KEYS, k) && error("config: $where is set by the typed config " *
            "($(OWNED_OPTION_KEYS[k])); set it there, not in [extra.options]")
        opts[k] = _coerce(KNOWN_OPTION_KEYS[k][1], v, where)
    end
    for k in keys(extra.physical_params)
        where = "extra.physical_params.$k"
        haskey(KNOWN_PHYSICAL_KEYS, k) || error("config: $where is not a physical_params " *
            "key Scythe reads (KNOWN_PHYSICAL_KEYS); check the spelling")
        haskey(OWNED_PHYSICAL_KEYS, k) && error("config: $where is set by the typed config " *
            "($(OWNED_PHYSICAL_KEYS[k])); set it there, not in [extra.physical_params]")
    end
    return Extra(options = opts, physical_params = copy(extra.physical_params))
end

# ── Dict → struct ─────────────────────────────────────────────────────────────────

"""
    MOVED_KEYS

Config keys that moved to another section: old full key → new full key. `load_config`
refuses the old spelling with a pointer to the new one instead of a bare unknown-key error
(plus the `MOVED_KEY_NOTES` entry, if any).
"""
const MOVED_KEYS = Dict{String,String}(
    "grid.lateral"                  => "boundary_conditions.lateral",
    "numerics.wall_bc"              => "boundary_conditions.vertical_scalars",
    "numerics.sedimentation_top_bc" => "boundary_conditions.sedimentation_lid")

"What a `MOVED_KEYS` refusal adds when the new key does not carry the old meaning alone."
const MOVED_KEY_NOTES = Dict{String,String}(
    "numerics.wall_bc" => "; it set the ground AND the lid, `vertical_scalars` is the ground " *
        "only: set `boundary_conditions.vertical_scalars_lid` to the same value for the old " *
        "behaviour (the lid default is natural)")

"""
    from_dict(T, d, path = "") -> T

Convert a parsed TOML table into config section `T` (or `RunConfig`). Unknown keys are
errors naming the section; values are coerced to the field types (Int → Float64, String →
Symbol, `"inf"` → `Inf`); absent keys keep their defaults. Path fields must already be
absolute or are resolved against the working directory (`load_config` resolves them
against their source file first).
"""
function from_dict(::Type{T}, d::AbstractDict, path::AbstractString = "") where {T<:AbstractConfigSection}
    names = fieldnames(T)
    kw = Dict{Symbol,Any}()
    for (k, v) in d
        s = Symbol(k)
        full = _join_path(path, k)
        haskey(MOVED_KEYS, full) && error("config: `$full` has moved to " *
            "`$(MOVED_KEYS[full])` (same values); set it there" * get(MOVED_KEY_NOTES, full, ""))
        s in names || error("config: unknown key `$k` in [$(isempty(path) ? "top level" : path)]; " *
                            "valid keys: " * join(string.(names), ", "))
        kw[s] = _coerce(fieldtype(T, s), v, _join_path(path, k))
    end
    return T(; kw...)
end

function from_dict(::Type{Extra}, d::AbstractDict, path::AbstractString = "extra")
    opts = Dict{Symbol,Any}()
    phys = Dict{Symbol,Float64}()
    for (k, v) in d
        where = _join_path(path, k)
        v isa AbstractDict || _type_error(where, v, "a table [$where]")
        if k == "options"
            for (ok, ov) in v
                opts[Symbol(ok)] = ov      # typed by `_checked_extra` against the registry
            end
        elseif k == "physical_params"
            for (pk, pv) in v
                w = "$where.$pk"
                x = _is_number(pv) ? Float64(pv) :
                    pv isa AbstractString ? _parse_inf(pv) : nothing
                x === nothing && error("config: $w = $(repr(pv)) is not a number; " *
                                       "physical_params holds numbers only")
                phys[Symbol(pk)] = x
            end
        else
            error("config: unknown key `$k` in [$path]; valid keys: options, physical_params")
        end
    end
    return Extra(options = opts, physical_params = phys)
end

# ── Paths ─────────────────────────────────────────────────────────────────────────
#
# The three path-valued fields, as TOML key paths. `load_config` resolves them in each
# parsed file against THAT file's directory before merging, which is the same as tracking
# the source file of every key through the merge but cannot get out of step with it.

const PATH_FIELDS = (("output", "dir"), ("reference_state", "sounding"),
                     ("initial_conditions", "file", "paths"))

"""
    _resolve_config_path(p, base) -> String

`""` stays `""` (no file); `@scythe/…` expands to `pkgdir(Scythe)`; `~` expands; anything
else relative is joined to `base`. The result is normalized and absolute.
"""
function _resolve_config_path(p::AbstractString, base::AbstractString)
    isempty(p) && return ""
    q = startswith(p, "@scythe/") ? joinpath(pkgdir(Scythe), p[length("@scythe/")+1:end]) :
        expanduser(p)
    return normpath(isabspath(q) ? q : joinpath(base, q))
end

function _resolve_path_fields!(d::AbstractDict, base::AbstractString)
    for keypath in PATH_FIELDS
        t = d
        for k in keypath[1:end-1]
            t = get(t, k, nothing)
            t isa AbstractDict || break
        end
        t isa AbstractDict || continue
        leaf = keypath[end]
        haskey(t, leaf) || continue
        v = t[leaf]
        if v isa AbstractString
            t[leaf] = _resolve_config_path(v, base)
        elseif v isa AbstractVector && all(e -> e isa AbstractString, v)
            t[leaf] = String[_resolve_config_path(e, base) for e in v]
        end   # a wrong type is left for from_dict to refuse with the key named
    end
    return d
end

# ── TOML front end ────────────────────────────────────────────────────────────────

"Merge `b` into `a`: tables merge recursively, scalars and arrays replace."
function _deep_merge!(a::AbstractDict, b::AbstractDict)
    for (k, v) in b
        if v isa AbstractDict && get(a, k, nothing) isa AbstractDict
            _deep_merge!(a[k], v)
        else
            a[k] = deepcopy(v)
        end
    end
    return a
end

"""
    parse_set(s) -> Dict{String,Any}

One `--set a.b.c=rhs` string as a nested table. `rhs` is parsed as a TOML value
(`TOML.parse("v = " * rhs)`) and falls back to the bare string, so `scheme=louis`,
`ts=0.25`, `num_cells=[15,15,15]` and `K_max=inf` all work unquoted.
"""
function parse_set(s::AbstractString)
    i = findfirst('=', s)
    i === nothing && error("config: --set $(repr(s)) is not of the form key.path=value")
    key = strip(s[1:prevind(s, i)])
    rhs = strip(s[nextind(s, i):end])
    parts = split(key, '.')
    (isempty(key) || any(isempty, parts)) && error("config: --set $(repr(s)) has an empty key")
    value = try
        TOML.parse("v = " * rhs)["v"]
    catch
        String(rhs)
    end
    d = Dict{String,Any}(String(parts[end]) => value)
    for p in reverse(parts[1:end-1])
        d = Dict{String,Any}(String(p) => d)
    end
    return d
end

"""
    load_config(paths...; set = String[]) -> RunConfig

Read one or more TOML files, later files overlaying earlier ones (tables merge
recursively, scalars and arrays replace), then apply each `set` string (`"a.b.c=value"`,
see [`parse_set`](@ref)) in order, and convert to a validated [`RunConfig`](@ref). Keys
absent everywhere take the struct defaults. Path fields resolve against the directory of
the file that set them; `set` paths against the working directory; `@scythe/` against the
package root.

The launcher (`simulations/<case>/run.jl`, [`parse_launcher_args`](@ref)) takes ONE file:
a run is one config file, written out resolved into its root as `scythe_config.toml`.
Several `paths` are for programmatic use (the equivalence tests stack legacy-pin files).
"""
function load_config(paths::AbstractString...; set::AbstractVector{<:AbstractString} = String[])
    merged = Dict{String,Any}()
    for p in paths
        ap = abspath(p)
        isfile(ap) || error("config: file not found: $ap")
        d = TOML.parsefile(ap)
        _resolve_path_fields!(d, dirname(ap))
        _deep_merge!(merged, d)
    end
    for s in set
        _deep_merge!(merged, _resolve_path_fields!(parse_set(s), pwd()))
    end
    return from_dict(RunConfig, merged, "")
end

# ── Struct → TOML ─────────────────────────────────────────────────────────────────

_toml_value(v::Symbol) = string(v)
_toml_value(v::AbstractVector{Symbol}) = String[string(e) for e in v]
_toml_value(v::AbstractConfigSection) = to_dict(v)
_toml_value(v::AbstractDict) = Dict{String,Any}(string(k) => _toml_value(x) for (k, x) in v)
_toml_value(v) = v

"""
    to_dict(cfg) -> Dict{String,Any}

The TOML-ready table of a config or section: every field including defaults, Symbols as
strings, `Inf` kept as a Float64 (the stdlib printer writes `inf`). Of the four initial-
condition sub-tables only the active `method`'s is included.
"""
to_dict(x::T) where {T<:AbstractConfigSection} =
    Dict{String,Any}(string(f) => _toml_value(getfield(x, f)) for f in fieldnames(T))

function to_dict(ic::InitialConditions)
    return Dict{String,Any}("method" => string(ic.method),
                            string(ic.method) => to_dict(getfield(ic, ic.method)))
end

# Print keys in struct-field order, not alphabetically, so the written file reads like the
# §2.1 listing. `TOML.print`'s `by` sees only a key, not its table, and the sections reuse
# names (`interval`, `method`, `scheme`, `z_max`), so one global rank cannot order every
# table. Each top-level section -- and each `[physics.*]` subsection -- is printed on its
# own, with a rank built from THAT section's fields (depth-first, so a nested table such
# as `[diagnostics.timestep_advisory]` ranks its own keys). Keys not in the structs (the
# `[extra.*]` contents) sort after, by name. Deterministic, so the text is canonical and
# `config_hash` is stable.
function _field_rank!(rank::Dict{String,Int}, x::AbstractConfigSection)
    for f in fieldnames(typeof(x))
        # Only the active initial-condition table is printed; ranking the other three would
        # let their shared names (`dT_max`) jump the active table's order.
        (x isa InitialConditions && f !== :method && f !== x.method) && continue
        haskey(rank, string(f)) || (rank[string(f)] = length(rank) + 1)
        v = getfield(x, f)
        v isa AbstractConfigSection && _field_rank!(rank, v)
    end
    return rank
end

function _print_table(io::IO, keypath::Vector{String}, x::AbstractConfigSection)
    rank = _field_rank!(Dict{String,Int}(), x)
    t = Dict{String,Any}(keypath[end] => to_dict(x))
    for k in reverse(keypath[1:end-1])
        t = Dict{String,Any}(k => t)
    end
    TOML.print(io, t; sorted = true,
               by = k -> (get(rank, string(k), typemax(Int)), string(k)))
    return nothing
end

"The canonical TOML text of `cfg` (no header): what `write_config` writes and `config_hash` hashes."
function canonical_toml(cfg::RunConfig)
    # The printer ends a nested table with a blank line of its own; collapse the doubles.
    text = sprint() do io
        first = true
        for f in fieldnames(RunConfig)
            sec = getfield(cfg, f)
            subs = sec isa Physics ?
                [([string(f), string(g)], getfield(sec, g)) for g in fieldnames(Physics)] :
                [([string(f)], sec)]
            for (kp, x) in subs
                first || println(io)
                first = false
                _print_table(io, kp, x)
            end
        end
    end
    return replace(text, r"\n{3,}" => "\n\n")
end

"""
    write_config(cfg, path; header = "") -> path

Write the full resolved TOML of `cfg` (every field, defaults included; absolute paths;
`inf` for `Inf`; only the active initial-condition sub-table) to `path`, with each line
of `header` written first as a `#` comment. `load_config(path) == cfg` holds whenever the
inactive initial-condition sub-tables are at their defaults.
"""
function write_config(cfg::RunConfig, path::AbstractString; header::AbstractString = "")
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        if !isempty(header)
            for line in split(chomp(header), '\n')
                println(io, isempty(line) ? "#" : "# " * line)
            end
            println(io)
        end
        print(io, canonical_toml(cfg))
    end
    return path
end

"SHA-256 (hex) of the canonical TOML text of `cfg`: equal configs hash equal, whatever file they came from."
config_hash(cfg::RunConfig) = bytes2hex(sha256(canonical_toml(cfg)))
