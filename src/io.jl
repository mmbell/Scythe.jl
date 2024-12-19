#Functions for I/O

function write_output(grid::AbstractGrid, model::ModelParameters, t::Float64)
    
    time = string(round(t; digits=2))
    if !isdir(model.output_dir)
        mkdir(model.output_dir)
    end

    # Calls Springsteel grid functions
    write_grid(grid, model.output_dir, time)

end
