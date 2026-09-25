#!/usr/bin/env julia
# Generate simulations/README.md: the launcher reference and the complete run-configuration
# reference, from the code itself -- the `RunConfig` section structs (their docstrings carry
# one `- `field::Type` — description` line per field), their defaults, the enum value sets
# (`Scythe.ENUM_FIELDS`), `EQUATION_SETS`, and the `[extra.*]` key registries.
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

# ── The page ──────────────────────────────────────────────────────────────────────────

function cases()
    dir = dirname(OUT)
    return sort([d for d in readdir(dir)
                 if isdir(joinpath(dir, d)) && isfile(joinpath(dir, d, "config.toml"))])
end

function page()
    io = IOBuffer()
    print(io, """
    # Scythe simulations

    <!-- GENERATED by tools/config_reference.jl from src/config.jl and src/run_driver.jl.
         Do not edit by hand: change the code (docstrings, defaults, registries) and run
         `julia --project=. tools/config_reference.jl`. -->

    Each directory here is one case: a `config.toml` (what is run), optional overlay files, a
    local launcher `run.jl`, SLURM launchers `*.sbatch`, and a `README.md` for the case.
    The configuration describes WHAT is run, never how: workers, threads and the scheduler are
    launcher flags, recorded for provenance in `segments.csv` but never in `config.toml`, so
    two runs that differ only in launcher produce identical output.

    ## Cases

    """)
    for c in cases()
        println(io, "- [`$c/`]($c/README.md) — `$c/config.toml`")
    end
    print(io, """

    ## Launching

    Every run goes through `julia --project=simulations`, NOT the root project: RRTMGP and
    ClimaComms are *weak* dependencies of Scythe (`ext/ScytheRRTMGPExt.jl`) and load only from
    an environment that lists them. `simulations/Project.toml` is a workspace member that
    shares the root `Manifest.toml`. (`julia --project=.` is for the test suite and for the
    output readers.)

    ### Local: `run.jl`

    ```
    julia --project=simulations simulations/<case>/run.jl [overlays...] [--set key.path=value ...]
          [--resume] [--workers N | --workers-per-patch a,b,c] [--worker-threads N]
          [--blas-threads N] [--slurm]
    ```

    | flag | meaning |
    |---|---|
    | `--config PATH` | the case file (default: `config.toml` beside `run.jl`) |
    | positional `overlays...` | overlay TOML files applied after `--config`, in order |
    | `--set key.path=value` | override one key (repeatable; applied after the overlays, in order) |
    | `--resume` | continue the run in `output.dir` from its `segments.csv` |
    | `--workers N` | N workers on a single-patch run |
    | `--workers-per-patch a,b,c` | worker group size per nest patch (one entry per patch) |
    | `--worker-threads N` | Julia threads per worker (0 = split the machine) |
    | `--blas-threads N` | OpenBLAS threads per process (default 1; 0 = OpenBLAS default) |
    | `--slurm` | add the workers with `ClusterManagers.addprocs_slurm` (inside an allocation) |
    | `--segment k` | internal: the orchestrator's per-segment child |

    A fresh start refuses an `output.dir` that already holds a run (model output is
    experimental data: `mv` the tree aside, never `rm` it). `--resume` re-reads
    `<output.dir>/config.toml` and runs the segments not recorded `done`; the only `--set` it
    accepts is `run.integration_time` (to extend a run); give the tree to resume with
    `--set output.dir=...`, which the launchers use only to locate it.

    ### Cluster: `*.sbatch`

    ```
    sbatch simulations/<case>/<launcher>.sbatch [overlays...] [--set key.path=value ...]
    ```

    The sbatch scripts set the node shape and `JULIA_NUM_THREADS`, run
    `simulations/_preflight.jl` with the same arguments (Springsteel version, the config
    loads and validates, RRTMGP artifacts present when radiation is on — run
    `tools/rrtmgp_prewarm.jl` once on a login node), give a fresh job its own
    `output.dir` unless one is `--set`, tee everything to `<output.dir>/job_stdout.log`, and
    pass every argument through to `run.jl`. Environment variables no longer configure a run.

    ## Overlays and `--set`

    - Files load in order (`--config`, then each overlay); tables merge recursively, scalars
      and arrays replace. Then each `--set` applies in order. Keys absent everywhere take the
      struct defaults below, so a case file lists only what it needs.
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
    <output.dir>/config.toml              resolved config (every field) + provenance header
    <output.dir>/reference.ref            reference column: z p rho_d rho_v rho_c
    <output.dir>/ics.csv | nest{i}_ics.csv  initial conditions
    <output.dir>/segments.csv             one row per segment (columns below)
    <output.dir>/job_stdout.log           (sbatch launchers only)
    <output.dir>/seg_NN/config.toml       the same config; the segment's values in the header
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
    `[initial_conditions.<method>]` table is written back to an output `config.toml`; all
    four are held so an overlay can switch `method`.

    """)
    for (path, T, inst) in sections()
        section_table(io, path, T, inst)
    end
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
