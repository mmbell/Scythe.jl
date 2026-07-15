#Functions for I/O

"""
    write_output(grid::AbstractGrid, model::ModelParameters, t::Float64)

Write the current grid state to the output directory specified in `model.output_dir`,
using a time-stamped filename stem (`round(t; digits=2)`).

The set of ANALYSIS formats is selected by `model.options[:output_formats]`, a
list of symbols (default `[:csv]`). Each listed format writes its own file(s) via
the matching Springsteel writer, all sharing the time stem:

| Symbol    | Writer               | Files                                              |
|:--------- |:-------------------- |:-------------------------------------------------- |
| `:csv`    | `write_grid`         | `<t>_spectral.csv`, `<t>_physical.csv`, `<t>_gridded.csv` |
| `:netcdf` | `write_netcdf`       | `<t>.nc` — regular/gridded representation (CF, with a `time` coordinate) |

For `:netcdf`, `model.options[:netcdf_derivatives]::Bool` (default `false`)
controls whether derivative slots are written alongside the field values.

JLD2 is NOT an analysis format: it is the restart/checkpoint format, written by
[`write_restart`](@ref) at `model.restart_interval`, not here. Passing
`:jld2` in `:output_formats` errors and points at `restart_interval` — set
`restart_interval = output_interval` to checkpoint every analysis step.

!!! note "CSV is the default for a reason"
    The ICs reader (`read_physical_grid`), the regression references, and the
    benchmark harnesses all parse `<t>_physical.csv` / `<t>_spectral.csv`. A run
    whose `:output_formats` omits `:csv` will not feed those consumers — keep
    `:csv` in the list when a run must also produce CSV. See also
    [`Twoway_PV_mixing`](@ref) for the `:noise_seed` option pattern this follows.

# Arguments
- `grid::AbstractGrid`: the Springsteel grid containing the current model state.
- `model::ModelParameters`: model configuration providing the output directory
  and (via `options`) the output format selection.
- `t::Float64`: current simulation time [s], used to label the output file.
"""
function write_output(grid::AbstractGrid, model::ModelParameters, t::Float64)

    tag = string(round(t; digits=2))
    isdir(model.output_dir) || mkpath(model.output_dir)

    formats = get(model.options, :output_formats, [:csv])
    include_derivs = get(model.options, :netcdf_derivatives, false)::Bool

    for fmt in formats
        if fmt === :csv
            write_grid(grid, model.output_dir, tag)
        elseif fmt === :netcdf
            write_netcdf(joinpath(model.output_dir, "$(tag).nc"), grid;
                         include_derivatives = include_derivs, time = t)
        elseif fmt === :jld2
            error("`:jld2` is not an analysis output format — it is the restart " *
                  "format, written at model.restart_interval by write_restart. " *
                  "Set restart_interval (e.g. = output_interval) instead of listing " *
                  ":jld2 in options[:output_formats].")
        else
            error("Unknown output format $(fmt) in options[:output_formats]; " *
                  "supported: :csv, :netcdf")
        end
    end

    return nothing
end

"""
    write_restart(grid::AbstractGrid, model::ModelParameters, t::Float64)

Write a restart checkpoint of `grid` to `model.output_dir/<t>.jld2` via
Springsteel's [`save_grid`](@ref) (params + spectral + physical, reloaded exactly
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
