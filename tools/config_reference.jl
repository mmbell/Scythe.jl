#!/usr/bin/env julia
# Generate simulations/README.md: the launcher reference and the complete run-configuration
# reference, from the code itself -- the `RunConfig` section structs (their docstrings carry
# one `- `field::Type` — description` line per field), their defaults, the enum value sets
# (`Scythe.ENUM_FIELDS`), `EQUATION_SETS`, the `[extra.*]` key registries, and the resolved
# boundary condition of every variable of every case (`build_model` + `boundary_condition_table`
# on each simulations/<case>/scythe_config.toml, so the tables cannot drift from the code).
#
#   julia --project=. tools/config_reference.jl [--check]
#
# Deterministic: every table is in struct-field or sorted-key order and nothing depends on
# the machine, the date or the checkout path, so regenerating an unchanged tree is a no-op.
# `--check` writes nothing and exits 1 when simulations/README.md is out of date.
# Regenerate and commit it whenever a config field, default, docstring or registry changes.

using Scythe
using TOML

const OUT = normpath(joinpath(@__DIR__, "..", "simulations", "README.md"))
const S = Scythe

# ── Docstring parsing ─────────────────────────────────────────────────────────────────

"The raw docstring text of `T` (a type defined in Scythe)."
function rawdoc(T::DataType)
    b = Base.Docs.Binding(S, nameof(T))
    d = Base.Docs.meta(S)[b]
    return join(string.(d.docs[Union{}].text))
end

"`field => description` from the `- `field::Type` — description` lines of `T`'s docstring."
function field_docs(T::DataType)
    out = Dict{Symbol,String}()
    for line in split(rawdoc(T), '\n')
        m = match(r"^- `(\w+)::[^`]+` — (.*)$", line)
        m === nothing || (out[Symbol(m.captures[1])] = strip(m.captures[2]))
    end
    for f in fieldnames(T)
        haskey(out, f) || error("config_reference: $T has no docstring line for field `$f`")
    end
    return out
end

"The one-line summary of a section docstring (the line after the signature)."
function section_summary(T::DataType)
    for line in split(rawdoc(T), '\n')
        m = match(r"^`\[[^\]]+\]` — (.*)$", line)
        m === nothing || return strip(m.captures[1])
    end
    return ""
end

# ── Value formatting ──────────────────────────────────────────────────────────────────

"A default value as it would be written in TOML (`inf` for Inf, strings quoted)."
function toml_text(v)
    v isa AbstractFloat && isinf(v) && return v > 0 ? "inf" : "-inf"
    tv = S._toml_value(v)
    tv isa AbstractDict && return isempty(tv) ? "{}" : "(table)"
    s = sprint(io -> TOML.print(io, Dict("v" => tv)))
    return strip(split(s, " = "; limit = 2)[2])
end

typename(T) = T === Vector{Symbol} ? "[string]" : T === Vector{String} ? "[string]" :
              T === Vector{Float64} ? "[float]" : T === Vector{Int} ? "[int]" :
              T === Symbol ? "string (enum)" : T === String ? "string" :
              T === Float64 ? "float" : T === Int ? "int" : T === Bool ? "bool" :
              T <: AbstractDict ? "table" : string(T)

md(s) = replace(string(s), "|" => "\\|")

# ── Sections in write_config order ───────────────────────────────────────────────────

"""(TOML table path, struct type, default instance) for every table, in write_config order.
Each default is the section struct's own `T()`, NOT a field of `RunConfig()`: the
`RunConfig` constructor makes paths absolute against the working directory, which would
put this machine's checkout path into the page."""
function sections()
    out = Tuple{String,DataType,Any}[]
    for f in fieldnames(RunConfig)
        f === :extra && continue
        T = fieldtype(RunConfig, f)
        if T === S.Physics
            for g in fieldnames(S.Physics)
                U = fieldtype(S.Physics, g)
                push!(out, ("physics.$g", U, U()))
            end
        else
            push!(out, (string(f), T, T()))
            for g in fieldnames(T)
                U = fieldtype(T, g)
                U <: S.AbstractConfigSection && push!(out, ("$f.$g", U, U()))
            end
        end
    end
    return out
end

function section_table(io, path, T, inst)
    docs = field_docs(T)
    println(io, "### `[$path]`\n")
    summ = section_summary(T)
    isempty(summ) || println(io, md(summ), "\n")
    println(io, "| key | type | default | description |")
    println(io, "|---|---|---|---|")
    for f in fieldnames(T)
        v = getfield(inst, f)
        v isa S.AbstractConfigSection && continue      # its own table follows
        println(io, "| `$f` | ", typename(fieldtype(T, f)), " | `", md(toml_text(v)), "` | ",
                md(docs[f]), " |")
    end
    println(io)
end

function enum_table(io)
    rows = Tuple{String,String}[]
    for (path, T, _) in sections(), f in fieldnames(T)
        haskey(S.ENUM_FIELDS, (T, f)) || continue
        push!(rows, ("$path.$f", join(("`$(e)`" for e in S.ENUM_FIELDS[(T, f)]), ", ")))
    end
    push!(rows, ("physics.boundary_layer.mynn_fidelity",
                 "a fidelity name, " * join(("`$(e)`" for e in S.MYNN_FIDELITIES), " or ") *
                 ", or a comma list of deviations from " *
                 join(("`$(e)`" for e in S.MYNN_DEVIATIONS), ", ")))
    sort!(rows)
    println(io, "| key | allowed values |")
    println(io, "|---|---|")
    for (k, v) in rows
        println(io, "| `$k` | $v |")
    end
    println(io)
end

function equation_set_table(io)
    println(io, "| `run.equation_set` | allowed `grid.geometry` | cylindrical | vertical column index |")
    println(io, "|---|---|---|---|")
    for k in sort(collect(keys(S.EQUATION_SETS)))
        e = S.EQUATION_SETS[k]
        println(io, "| `$k` | ", join(("`$g`" for g in e.geometries), ", "), " | ",
                e.cyl ? "yes" : "no", " | ", e.zcol, " |")
    end
    println(io)
end

function registry_table(io, known, owned, what)
    println(io, "| key | type | owned by | description |")
    println(io, "|---|---|---|---|")
    for k in sort(collect(keys(known)); by = string)
        T, desc = known[k]
        own = haskey(owned, k) ? "`" * md(owned[k]) * "` (refused in `[extra.$what]`)" : ""
        println(io, "| `$k` | ", md(typename(T)), " | ", own, " | ", md(desc), " |")
    end
    println(io)
end

# ── Boundary conditions ───────────────────────────────────────────────────────────────

"The resolved BC table of every case, from the code (build_model + boundary_condition_table)."
function bc_section(io)
    print(io, """
    ## Boundary conditions

    Every prognostic variable has a boundary condition on each of its four sides: `inner` and
    `outer` (the axis and the outer wall on the cylindrical sets; left and right on XZ), and
    `bottom` (the ground) and `top` (the lid). They are NOT listed one by one in the config:
    `[boundary_conditions]` holds three family rules that `build_boundary_conditions`
    (src/config_build.jl) applies to the run's variable list, then the per-variable
    `override` entries, then `build_nest` sets the nest interfaces. What each family value
    does, exactly:

    | key = value | rule applied |
    |---|---|
    | `lateral = "walls"`, XZ | inner and outer: `u`, `w` dirichlet (zero); every other variable neumann (zero gradient) |
    | `lateral = "walls"`, cylindrical (the only choice) | inner (axis): `u`, `v` dirichlet; every other variable, `w` included, neumann. outer (wall): `u` dirichlet; every other variable, `v` and `w` included, neumann |
    | `lateral = "periodic"` (XZ, single patch only) | inner and outer: periodic on every variable |
    | `vertical_scalars = "d2"` | bottom and top of every variable: second_derivative (zero); then the `w` and sedimentation rules below |
    | `vertical_scalars = "natural"` | bottom and top of every variable: natural (unconstrained); then the rules below |
    | `vertical_scalars = "neumann"` | bottom and top of every variable: neumann; then the rules below |
    | `vertical_scalars = "r1t1x"` | as `d2`, except `rho_d`, `rho_t`, `E_t`, `Q_ss` neumann and `p` r1t1x (the rigid-wall pressure condition `dp'/dz = -g rho_t'`, installed before every fit; relaxed on `numerics.wall_bc_tau`) |
    | (always) | `w` dirichlet (zero) at bottom and top, whatever `vertical_scalars` |
    | (always) | the sedimenting slots (rain; the rain number under two-moment rain; the twelve ice slots under ISHMAEL) natural at the bottom, so they fall out |
    | `sedimentation_lid = "natural"` | the sedimenting slots natural at the top too |
    | `sedimentation_lid = "wall"` | the sedimenting slots keep the `vertical_scalars` condition at the top |
    | (always, B-spline vertical) | the sedimentation FLUX of every sedimenting slot is fitted with a zero-flux (dirichlet) lid, whatever its slot's own top BC: nothing falls in from above (`[extra.options] sedimentation_flux_lid = "natural"` undoes this, forensics only) |
    | nests (not configurable) | at each junction the finer patch's side is r3x (its border value, slope and curvature pinned to the coarser parent at run time) and the coarser patch's side is natural (its free collar termination, one parent cell inside the child) |

    `[boundary_conditions.override]` sets single entries after these rules, for experiments:

    ```toml
    [boundary_conditions.override]
    nu_r = { top = "dirichlet" }            # or a [boundary_conditions.override.nu_r] table
    ```

    Sides: $(join(("`$(x)`" for x in S.BC_OVERRIDE_SIDES), ", ")). Values: $(join(("`$(x)`" for x in S.BC_OVERRIDE_VALUES), ", ")).
    The variable must be one the run has (an error lists them); `inner`/`outer` act on the
    domain's own edges (the first patch's inner side, the last patch's outer side), never on a
    nest interface, and are refused under `lateral = "periodic"`.

    Every run writes the resolved table to `<output.dir>/boundary_conditions.toml` (one
    `[[patch]]` per patch, one `[[patch.variable]]` per variable) and prints its compact
    form at startup. The tables of the cases here, generated from their `scythe_config.toml`
    (a single lateral name = both sides; `p<i>` = patch `i`):

    """)
    for c in cases()
        cfg = load_config(joinpath(dirname(OUT), c, "scythe_config.toml"))
        bcs = cfg.boundary_conditions
        println(io, "### `$c`\n")
        println(io, "`lateral = \"$(bcs.lateral)\"`, `vertical_scalars = \"$(bcs.vertical_scalars)\"`, ",
                "`sedimentation_lid = \"$(bcs.sedimentation_lid)\"`",
                isempty(bcs.override) ? "" : ", plus $(length(bcs.override)) override(s)", ".\n")
        table = redirect_stderr(devnull) do
            boundary_condition_table(build_model(cfg), cfg)
        end
        println(io, "```")
        print_boundary_conditions(io, table)
        println(io, "```\n")
    end
end

# ── The page ──────────────────────────────────────────────────────────────────────────

function cases()
    dir = dirname(OUT)
    return sort([d for d in readdir(dir)
                 if isdir(joinpath(dir, d)) && isfile(joinpath(dir, d, "scythe_config.toml"))])
end

function page()
    io = IOBuffer()
    print(io, """
    # Scythe simulations

    <!-- GENERATED by tools/config_reference.jl from src/config.jl and src/run_driver.jl.
         Do not edit by hand: change the code (docstrings, defaults, registries) and run
         `julia --project=. tools/config_reference.jl`. -->

    Each directory here is one case: a `scythe_config.toml` (what is run), a launcher
    `run.jl`, SLURM launchers `*.sbatch`, a `README.md` for the case, and optionally an
    `examples/` directory of alternate grids. A run is ONE config file: the example files are
    copied into `scythe_config.toml` by hand, never layered on the command line, so the
    resolved `scythe_config.toml` written into the run root is the whole record. The
    configuration describes WHAT is run, never how: workers, threads and the scheduler are
    launcher flags, recorded for provenance in `segments.csv` but never in the config, so two
    runs that differ only in launcher produce identical output.

    ## Cases

    """)
    for c in cases()
        println(io, "- [`$c/`]($c/README.md) — `$c/scythe_config.toml`")
    end
    print(io, """

    ## Launching

    A `run.jl` is `using Scythe` plus the launcher call and nothing else, so it runs from any
    directory under any Julia environment where Scythe is installed (`<env>` below; a Scythe
    checkout itself works, `--project=/path/to/Scythe.jl`). Radiation (RRTMGP) is part of
    Scythe and needs no special environment. To run a case outside the repository, copy its
    `scythe_config.toml` and `run.jl` to a directory and run it there.

    ### Local: `run.jl`

    ```
    julia --project=<env> run.jl [--config scythe_config.toml] [--set key.path=value ...]
          [--resume] [--workers N | --workers-per-patch a,b,c] [--worker-threads N]
          [--blas-threads N] [--slurm]
    ```

    | flag | meaning |
    |---|---|
    | `--config PATH` | the run's one config file (default: `scythe_config.toml` beside `run.jl`); a positional argument is refused |
    | `--set key.path=value` | run control: `output.dir`, `run.integration_time`, ... (repeatable, applied in order; recorded in the resolved config's header) |
    | `--resume` | continue the run in `output.dir` from its `segments.csv` |
    | `--workers N` | N workers on a single-patch run |
    | `--workers-per-patch a,b,c` | worker group size per nest patch (one entry per patch) |
    | `--worker-threads N` | Julia threads per worker (0 = split the machine) |
    | `--blas-threads N` | OpenBLAS threads per process (default 1; 0 = OpenBLAS default) |
    | `--slurm` | add the workers with `ClusterManagers.addprocs_slurm` (inside an allocation) |
    | `--segment k` | internal: the orchestrator's per-segment child |

    Before anything is written (fresh start or `--resume`), `run_simulation` runs
    `preflight_checks`: Springsteel new enough, the config validates, and, when
    `physics.radiation.scheme = "rrtmgp"`, the RRTMGP lookup tables already local (run
    `tools/rrtmgp_prewarm.jl` once on a login node; compute nodes have no network). A fresh
    start refuses an `output.dir` that already holds a run (model output is experimental
    data: `mv` the tree aside, never `rm` it). `--resume` re-reads
    `<output.dir>/scythe_config.toml` and runs the segments not recorded `done`; the only `--set` it
    accepts is `run.integration_time` (to extend a run); give the tree to resume with
    `--set output.dir=...`, which the launchers use only to locate it.

    ### Cluster: `*.sbatch`

    ```
    sbatch simulations/<case>/<launcher>.sbatch [--set key.path=value ...] [--resume]
    ```

    The sbatch scripts run `julia --project=\${SCYTHE_PROJECT:-\$SCYTHE}
    \$SCYTHE/simulations/<case>/run.jl ...`, where `SCYTHE` (from `SCYTHE_DIR`) is the checkout
    or run directory holding `simulations/<case>/scythe_config.toml` and `SCYTHE_PROJECT` is
    any environment where Scythe is installed. They set the node shape and
    `JULIA_NUM_THREADS`, give a fresh job its own `output.dir` unless one is `--set` (TC),
    tee everything to `<output.dir>/job_stdout.log`, and pass every argument through to
    `run.jl`. Environment variables no longer configure a run.

    ## The config file and `--set`

    - The case file is read, then each `--set` applies in order. Keys absent from the file
      take the struct defaults below, so a case file lists only what it needs. `--set` is
      for run control (where the output goes, how long to run); a physics or grid change
      belongs in the config file, so that the file IS the record.
    - `--set` values are parsed as TOML (`TOML.parse("v = " * rhs)`) and fall back to a bare
      string: `--set physics.boundary_layer.scheme=louis`, `--set timestep.ts=0.25`,
      `--set grid.num_cells=[15,15,15]`, `--set physics.boundary_layer.K_max=inf`,
      `--set 'output.formats=["netcdf","csv"]'` all work.
    - Unknown keys anywhere are errors naming the section; enum values are checked against
      the sets below; `inf` round-trips.
    - Paths (`output.dir`, `reference_state.sounding`, `initial_conditions.file.paths`)
      resolve against the directory of the file that set them; `--set` paths against the
      working directory; `@scythe/` is the package root. The written config holds absolute
      paths.
    - Implied values are resolved at build: `physics.microphysics.rain_moments = 0` (2 under
      ice, else 1), `physics.radiation.z_max = 0` (= `physics.sponge.z_damp`),
      `timestep.ts_per_patch = []` (every patch at `ts`), `numerics.cfl_interval = 0`
      (= `output.interval`).
    - Cross-section rules (ice with Louis, `exact_si` with `state_dependent_si`, Smagorinsky
      on XZ, periodic cylindrical grids, segments without checkpoints, ...) are checked by
      `validate(cfg)` at build.

    ## Output layout

    Always the segmented layout, even for one segment, so readers handle one shape:

    ```
    <output.dir>/scythe_config.toml       resolved config (every field) + provenance header
    <output.dir>/reference.ref            reference column: z p rho_d rho_v rho_c
    <output.dir>/ics.csv | nest{i}_ics.csv  initial conditions
    <output.dir>/segments.csv             one row per segment (columns below)
    <output.dir>/boundary_conditions.toml the BC every variable actually has, per patch (see below)
    <output.dir>/job_stdout.log           (sbatch launchers only)
    <output.dir>/seg_NN/scythe_config.toml  the same config; the segment's values in the header
    <output.dir>/seg_NN/segment_stdout.log
    <output.dir>/seg_NN/[nest{i}/]<t>.nc | <t>.jld2 | scythe_{out,err}.log
    ```

    `<t>` is LOCAL to the segment: each segment starts at t = 0 from the previous segment's
    final checkpoint and writes its own t = 0 snapshot (a duplicate of the previous
    segment's last one). Absolute time is `t_offset_s + t`. The radiation clock is carried:
    each segment runs with `start_hour = physics.radiation.start_hour + t_offset_s / 3600`.

    `segments.csv` columns:

    """)
    for c in S.SEGMENTS_CSV_COLS
        println(io, "- `$c`")
    end
    print(io, """

    ## Configuration reference

    Tables in the order `write_config` writes them. Only the ACTIVE
    `[initial_conditions.<method>]` table is written back to an output `scythe_config.toml`;
    all four are held so a `--set` can switch `method`.

    """)
    for (path, T, inst) in sections()
        section_table(io, path, T, inst)
    end
    bc_section(io)
    println(io, "### Equation sets and geometries\n")
    equation_set_table(io)
    println(io, "### Enum values\n")
    enum_table(io)
    print(io, """
    ## The `[extra.*]` escape hatch

    `[extra.options]` and `[extra.physical_params]` pass keys straight into the model's
    `options` / `physical_params` dicts, for the knobs no typed section owns. Every key is
    checked against the registries below (a misspelled key is an error, not silently
    ignored) and coerced to the registered type; `physical_params` holds numbers only. A key
    a typed section produces ("owned by") is refused there, so each setting has exactly one
    spelling. `Kvdiff`, `Kvdiff_heat` and `Kvdiff_water` are deliberately not owned: the
    build writes them as 0 (vertical mixing belongs to the boundary-layer scheme) and
    `[extra.physical_params]` is the way to override them.

    ```toml
    [extra.options]
    mynn_edmf_mom = false
    [extra.physical_params]
    co2_ppm = 280.0
    ```

    ### `[extra.options]` keys (`KNOWN_OPTION_KEYS`)

    """)
    registry_table(io, S.KNOWN_OPTION_KEYS, S.OWNED_OPTION_KEYS, "options")
    if !isempty(S.RETIRED_OPTION_KEYS)
        println(io, "Retired (refused with a message):\n")
        for k in sort(collect(keys(S.RETIRED_OPTION_KEYS)); by = string)
            println(io, "- `$k` — ", md(S.RETIRED_OPTION_KEYS[k]))
        end
        println(io)
    end
    println(io, "### `[extra.physical_params]` keys (`KNOWN_PHYSICAL_KEYS`)\n")
    registry_table(io, S.KNOWN_PHYSICAL_KEYS, S.OWNED_PHYSICAL_KEYS, "physical_params")
    return replace(String(take!(io)), r"\n{3,}" => "\n\n")
end

text = page()
if "--check" in ARGS
    current = isfile(OUT) ? read(OUT, String) : ""
    current == text || (println(stderr, "$OUT is out of date; run tools/config_reference.jl"); exit(1))
    println("$OUT is up to date")
else
    write(OUT, text)
    println("wrote $OUT ($(count(==('\n'), text)) lines)")
end
