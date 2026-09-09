#Functions for I/O

"""
    write_output(grid::AbstractGrid, model::ModelParameters, t::Float64;
                 ctx = nothing, workerids::Vector{Int64} = Int64[])

Write the current grid state to the output directory specified in `model.output_dir`,
using a time-stamped filename stem (`round(t; digits=2)`).

The set of ANALYSIS formats is selected by `model.options[:output_formats]`, a
list of symbols (default `[:netcdf]`). Each listed format writes its own file(s),
all sharing the time stem:

| Symbol        | Writer                          | Files                                              |
|:------------- |:------------------------------- |:-------------------------------------------------- |
| `:netcdf`     | [`write_netcdf_comprehensive`](@ref) | `<t>.nc` — ONE comprehensive file: primes, totals, hydrometeors, derived thermodynamics, radar/precipitation products, column integrals and the reference profiles |
| `:csv`        | `write_grid`                    | `<t>_spectral.csv`, `<t>_physical.csv`, `<t>_gridded.csv` |
| `:netcdf_raw` | Springsteel `write_netcdf`      | `<t>_raw.nc` — the legacy prognostic-slot-only gridded file |

`:netcdf` writes the comprehensive file only for the runs it is defined for: a
pressure-reference (`moist_compressible`) equation set on a 2-D k-active grid, with a
reference state to add back ([`comprehensive_netcdf_eligible`](@ref)). Every other run —
the legacy equation sets, the 3-D moist-compressible grids — gets
[`write_netcdf_prognostic`](@ref) under the same `<t>.nc` name, which is Springsteel's
prognostic layout with real coordinate units and a `scythe_file_kind` marker. A reader can
always tell which it has from that global attribute.

For `:netcdf_raw`, `model.options[:netcdf_derivatives]::Bool` (default `false`) controls
whether derivative slots are written alongside the field values. It does NOT apply to
`:netcdf`: the comprehensive file's variables are derived products, which have no spline
derivative slots.

JLD2 is NOT an analysis format: it is the restart/checkpoint format, written by
[`write_restart`](@ref) at `model.restart_interval`, not here. Passing
`:jld2` in `:output_formats` errors and points at `restart_interval` — set
`restart_interval = output_interval` to checkpoint every analysis step.

!!! note "CSV is opt-in; benchmarks pin it"
    The ICs reader (`read_physical_grid`), the regression references, and the benchmark
    harnesses all parse `<t>_physical.csv` / `<t>_spectral.csv`, so every benchmark under
    `benchmarks/` pins `:output_formats => [:csv]` explicitly rather than relying on the
    default. A NEW run that must feed one of those consumers has to list `:csv` itself;
    the default no longer does it for you.

# Arguments
- `grid::AbstractGrid`: the Springsteel grid containing the current model state.
- `model::ModelParameters`: model configuration providing the output directory
  and (via `options`) the output format selection.
- `t::Float64`: current simulation time [s], used to label the output file.

# Keywords
- `ctx`: the [`NetCDFOutputContext`](@ref) built once per patch by the driver. `nothing`
  (the default) builds one HERE, which costs a reference-state construction per call — fine
  for a test or a REPL call, wrong for a run, which is why `run_model`, `model_loop`,
  `finalize_model` and `run_nested_patch` all thread one through.
- `workerids`: the workers holding this patch's tiles. `:netcdf` asks each of them for its
  held physics diagnostics ([`gather_physics`](@ref)) and writes the assembled BL /
  radiation / surface groups into the comprehensive file. Empty (the default) writes the
  file with no physics groups, which is what an in-process caller gets.
"""
function write_output(grid::AbstractGrid, model::ModelParameters, t::Float64;
                      ctx = nothing, workerids::Vector{Int64} = Int64[])

    validate_output_options(model)

    tag = string(round(t; digits=2))
    isdir(model.output_dir) || mkpath(model.output_dir)

    formats = get(model.options, :output_formats, [:netcdf])
    include_derivs = get(model.options, :netcdf_derivatives, false)::Bool

    for fmt in formats
        if fmt === :csv
            write_grid(grid, model.output_dir, tag)
        elseif fmt === :netcdf
            # Lazy context so an in-process caller (test, REPL) needs no setup. The drivers
            # pass one in; see the docstring.
            ctx === nothing && (ctx = netcdf_output_context(grid, model))
            path = joinpath(model.output_dir, "$(tag).nc")
            if ctx.active
                # The physics groups (BL, radiation, surface) live on the WORKERS, on
                # each tile's own mish. Ask for them HERE, at the output cadence and
                # nowhere else, and stitch the tiles back together
                # (src/netcdf_output.jl). An in-process caller with no workers (a test,
                # the REPL) passes none and gets a file with no physics groups.
                physics = isempty(workerids) ? nothing :
                          assemble_physics(gather_physics(workerids))
                write_netcdf_comprehensive(path, grid, model, t, ctx, physics)
            else
                write_netcdf_prognostic(path, grid, model, t)
            end
        elseif fmt === :netcdf_raw
            write_netcdf(joinpath(model.output_dir, "$(tag)_raw.nc"), grid;
                         include_derivatives = include_derivs, time = t)
        elseif fmt === :jld2
            error("`:jld2` is not an analysis output format — it is the restart " *
                  "format, written at model.restart_interval by write_restart. " *
                  "Set restart_interval (e.g. = output_interval) instead of listing " *
                  ":jld2 in options[:output_formats].")
        else
            error("Unknown output format $(fmt) in options[:output_formats]; " *
                  "supported: :csv, :netcdf, :netcdf_raw")
        end
    end

    return nothing
end

"""
    write_restart(grid::AbstractGrid, model::ModelParameters, t::Float64)

Write a restart checkpoint of `grid` to `model.output_dir/<t>.jld2` via Springsteel's
[`save_grid`](@ref) (params + spectral + physical, reloaded exactly
by `load_grid`). Called at `model.restart_interval` and once more at the end of
the run; a run with `restart_interval == 0` writes no checkpoints.

A `.jld2` checkpoint is a valid `initial_conditions` for a later run: the loader
([`load_initial_conditions!`](@ref)) copies the archived spectral coefficients
directly. This is a WARM restart, not bit-identical: the AB3 explicit integrator
also carries two steps of tendency history (`expdot_nm1/nm2`) in the per-tile
`ModelTile`, which is not part of the grid and so is not checkpointed, and the
restarted run's step counter restarts (its first step is Euler, not AB3). The
resumed trajectory is physically continuous — the startup transient is ~0.1-0.3%
of the base fields and decays — but differs from an uninterrupted run at that
level. (Stochastic runs additionally resume with a fresh noise stream — the
per-tile RNG re-seeds at init.)
"""
function write_restart(grid::AbstractGrid, model::ModelParameters, t::Float64)
    isdir(model.output_dir) || mkpath(model.output_dir)
    save_grid(joinpath(model.output_dir, "$(string(round(t; digits=2))).jld2"), grid)
    return nothing
end
