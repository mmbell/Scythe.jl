# ── DEPRECATED shim: TC nest construction and initialization ────────────────────────
# Include AFTER tc_params.jl. Provides, over `build_model`/`initialize_case!`
# (src/config_build.jl, src/config_init.jl) applied to TC_CFG / TC_CFG_RLR:
#   make_base(integration_time; ...)  -> ModelParameters template (3-nest placeholder grid)
#   make_nest(base)                   -> NestedModelParameters (production nest layout)
#   init_tc!(nest)                    -> (models, topo): writes the shared reference state
#                                         and the native-grid balanced-vortex ICs
#   tc_boundary_conditions()          -> (axis_bc, wall_bc, bot_bc, top_bc)
#   tc_output_dir(path; allow_existing) -> path (refuses to overwrite preserved output)
#   TC_VARS, TC_RAIN_VAR, TC_CLOUD_VAR, TC_NAME_OPTS, TC_WATER_OPTS
#
# See tc_params.jl's header for what changed (every SCYTHE_TC_* knob but SCYTHE_TC_RES is
# gone; the physics is now the production set; radiation is on by default and needs the
# RRTMGP extension loaded). `using Scythe` here too, for the same reason as tc_params.jl:
# this file must not depend on being included after the caller's own `using Scythe`.
using Scythe

# The water transforms decide what slots 8 and 9 are NAMED (rho_r/rho_c untransformed,
# nu_r/nu_c under a transform), and the name is the only thing a consumer sees --
# Springsteel builds the CSV/netCDF header straight from GridParameters.vars. So every
# name-keyed dict here (vars, the four BC dicts) has to be built from `mc_var_names`, never
# from a literal. `TC_WATER_OPTS` stays the MINIMAL water-transform dict (matching the
# pre-config constant exactly: model_tests build small hand-rolled option sets, e.g.
# tc_lid_drift_probe.jl's resting-column probe, that merge it with only a few other keys
# and must NOT pick up ice/MYNN/radiation along the way).
const TC_WATER_OPTS = Dict{Symbol,Any}(:condensate_transform => CONDENSATE_TRANSFORM,
                                       :rain_transform => RAIN_TRANSFORM)

# The FULL production options (ice, two-moment rain, MYNN's rho_e, ...) via the same
# build_physics the model itself is built from -- TC_VARS must reflect every
# physics-appended slot, not just the water transforms, now that the shim runs the
# production physics by default (see tc_params.jl's header). `TC_NAME_OPTS` is this same
# dict, kept under its pre-config name for model_tests that merge partial option sets
# against it.
const _TC_OPTIONS, _TC_PHYSICAL = Scythe.build_physics(TC_CFG)
const TC_NAME_OPTS = _TC_OPTIONS
const TC_VARS      = Scythe.mc_var_names(_TC_OPTIONS; cyl = true)
const TC_RAIN_VAR  = Scythe.rain_var_name(_TC_OPTIONS)
const TC_CLOUD_VAR = Scythe.condensate_var_name(_TC_OPTIONS)

"""
    tc_boundary_conditions() -> (axis_bc, wall_bc, bot_bc, top_bc)

DEPRECATED shim over `Scythe.build_boundary_conditions(TC_CFG, ...)`: the axis + wall
lateral conditions and the `boundary_conditions.vertical_scalars` (currently `"d2"`,
`SecondDerivativeBC` on every variable but `w`) vertical conditions of the production TC configuration, keyed by `TC_VARS`.
The old `SCYTHE_TC_WALL_BC` switch (d2 | natural | r1t1x) is gone: a diagnostic that needs
a non-default wall condition should build its own BC dicts directly (as
model_tests/tc_lid_drift_probe.jl does) or load a `TC_CFG`-like config with
`boundary_conditions.vertical_scalars` overridden.
"""
function tc_boundary_conditions()
    _tc_deprecated(:tc_boundary_conditions)
    return Scythe.build_boundary_conditions(TC_CFG, _TC_OPTIONS, TC_VARS)
end

"""
    tc_output_dir(path; allow_existing = false) -> path

Refuse to start a run that would OVERWRITE a preserved one, and say how to fix it.

Model output is experimental data: every run gets its own tree, and old ones are `mv`'d
aside, never `rm -rf`'d. This looks for real output (`.nc`, `.jld2`, `*_physical.csv`,
`*_gridded.csv`) anywhere under `path` and raises if any is found (the initial conditions,
the reference file and the logs are not counted -- every run legitimately rewrites those).

`allow_existing = true` is for a restart, which writes into its own tree by design.
`SCYTHE_TC_FORCE_OUTDIR=1` overrides it for a deliberate re-run.
"""
function tc_output_dir(path::AbstractString; allow_existing::Bool = false)
    _tc_deprecated(:tc_output_dir)
    (allow_existing || get(ENV, "SCYTHE_TC_FORCE_OUTDIR", "0") == "1") && return path
    isdir(path) || return path
    found = String[]
    for (root, _, files) in walkdir(path), f in files
        (endswith(f, ".nc") || endswith(f, ".jld2") ||
         endswith(f, "_physical.csv") || endswith(f, "_gridded.csv")) || continue
        push!(found, joinpath(relpath(root, path), f))
        length(found) >= 4 && break
    end
    isempty(found) && return path
    error("""
        refusing to write into $(path): it already holds model output
        ($(join(first(found, 3), ", "))$(length(found) > 3 ? ", ..." : "")).

        Model output is experimental data -- a run that took hours is not this script's to
        overwrite, and the writer would clobber it file by file without a word. Either

          * give this run its own tree, or
          * mv $(basename(path)) aside under a descriptive name first, or
          * set SCYTHE_TC_FORCE_OUTDIR=1 if you really do mean to replace it.

        Never rm -rf an output tree.""")
end

"Reconstruct `m::ModelParameters` with the given fields replaced (the struct is immutable)."
_tc_mp_with(m::ModelParameters; kw...) =
    ModelParameters(; (f => (haskey(kw, f) ? kw[f] : getfield(m, f))
                       for f in fieldnames(ModelParameters))...)

"""
    make_base(integration_time; output_formats=OUTPUT_FORMATS, output_dir=OUTPUT_DIR,
              initial_conditions=joinpath(output_dir, "tc_ics.csv"), geometry="RiRk",
              output_interval=OUTPUT_INTERVAL, extra_options=Dict{Symbol,Any}(),
              extra_physical=Dict{Symbol,Any}()) -> ModelParameters

DEPRECATED shim over `build_model`. Builds the single-patch TEMPLATE `ModelParameters` for
the production 3-nest TC (the role the pre-config `make_base` played): a full-domain
placeholder grid carrying `TC_CFG`'s (or `TC_CFG_RLR`'s, when `geometry == "RLR"`)
production options/physical_params. `output_dir` and `initial_conditions` are applied
directly to the built struct via a keyword-copy -- bypassing `build_model`'s own `seg_01`
segment-directory layout, which the model_tests scripts that call this do not expect (they
read/write `<output_dir>/nest{i}/<t>.nc` directly). `extra_options`/`extra_physical` are
merged into the built `options`/`physical_params` Dicts LAST and are NOT validated against
the config registries -- the same unchecked escape hatch the pre-config `extra_options` was,
for a diagnostic run that wants an option `RunConfig` has no field for.

Radiation is on by default (`TC_CFG.physics.radiation.scheme == :rrtmgp`); RRTMGP is an
ordinary dependency of Scythe, so nothing needs importing first.
"""
function make_base(integration_time; output_formats = OUTPUT_FORMATS,
                   output_dir::AbstractString = OUTPUT_DIR,
                   initial_conditions::AbstractString = joinpath(output_dir, "tc_ics.csv"),
                   geometry::AbstractString = "RiRk",
                   output_interval = OUTPUT_INTERVAL,
                   extra_options::AbstractDict = Dict{Symbol,Any}(),
                   extra_physical::AbstractDict = Dict{Symbol,Any}())
    _tc_deprecated(:make_base)
    mkpath(output_dir)
    src = geometry == "RLR" ? TC_CFG_RLR : TC_CFG
    cfg = Scythe._with(src;
        run = Scythe._with(src.run; integration_time = Float64(integration_time)),
        output = Scythe._with(src.output; dir = String(output_dir),
                              interval = Float64(output_interval),
                              formats = Symbol.(collect(output_formats))))
    built = build_model(cfg)
    base = built isa NestedModelParameters ? built.base : built
    base = _tc_mp_with(base; output_dir = String(output_dir),
                       initial_conditions = String(initial_conditions))
    isempty(extra_options) || merge!(base.options, extra_options)
    isempty(extra_physical) || merge!(base.physical_params, extra_physical)
    return base
end

"""
    make_nest(base) -> NestedModelParameters

DEPRECATED shim, unchanged in shape from the pre-config version: wraps the `base` template
from `make_base` in the production 3-nest layout (`NEST_BOUNDARIES`/`NEST_CELLS`/
`NEST_TS`/`NEST_WORKERS`, or the `_RLR` variants when `base`'s geometry is `"RLR"`).
"""
function make_nest(base)
    _tc_deprecated(:make_nest)
    rlr = base.grid_params.geometry == "RLR"
    return NestedModelParameters(
        boundaries = rlr ? NEST_BOUNDARIES_RLR : NEST_BOUNDARIES,
        num_cells = rlr ? NEST_CELLS_RLR : NEST_CELLS,
        ts = NEST_TS,
        workers_per_patch = NEST_WORKERS,
        base = base)
end

"""
    init_tc!(nest) -> (models, topo)

DEPRECATED shim over `initialize_case!`: builds the nest, `mkpath`s every patch's
`output_dir`, writes the shared reference state and the native-grid balanced-vortex initial
conditions (`initialize_case!` on `TC_CFG`/`TC_CFG_RLR`, matching `nest.base`'s geometry),
and returns `build_nest(nest)` -- the per-patch `ModelParameters` and the `NestTopology` --
matching the pre-config `init_tc!` return shape.
"""
function init_tc!(nest)
    _tc_deprecated(:init_tc!)
    models, topo = build_nest(nest)
    for m in models
        mkpath(m.output_dir)
    end
    println("TC nest: ts_actual = $(topo.ts_actual), n_sub = $(topo.n_sub)")
    rlr = nest.base.grid_params.geometry == "RLR"
    src = rlr ? TC_CFG_RLR : TC_CFG
    cfg = Scythe._with(src; output = Scythe._with(src.output; dir = nest.base.output_dir))
    initialize_case!(cfg, nest)
    return models, topo
end
