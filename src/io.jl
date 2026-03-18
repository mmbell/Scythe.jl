#Functions for I/O

"""
    write_output(grid::AbstractGrid, model::ModelParameters, t::Float64)

Write the current grid state to the output directory specified in `model.output_dir`.

Creates the output directory if it does not already exist, then delegates to the
Springsteel `write_grid` function with a time-stamped filename.

# Arguments
- `grid::AbstractGrid`: the Springsteel grid containing the current model state.
- `model::ModelParameters`: model configuration providing the output directory path.
- `t::Float64`: current simulation time [s], used to label the output file.
"""
function write_output(grid::AbstractGrid, model::ModelParameters, t::Float64)
    
    time = string(round(t; digits=2))
    if !isdir(model.output_dir)
        mkdir(model.output_dir)
    end

    # Calls Springsteel grid functions
    write_grid(grid, model.output_dir, time)

end
