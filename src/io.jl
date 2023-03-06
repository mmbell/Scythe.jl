#Functions for I/O

function read_initialconditions(ic::String, grid::AbstractGrid)

    # Initialize the patch on each process
    initialconditions = CSV.read(ic, DataFrame, header=1)
    
    # Check that the dimensions are correct
    check_initialconditions(initialconditions, grid)
    
    # Check for all the variables
    for key in keys(grid.params.vars)
        foundkey = false
        for name in names(initialconditions)
            if (name == key)
                foundkey = true
            end
        end
        if foundkey == false
            throw(DomainError(key, "IC missing data"))
        end
    end
    
    # Assign variables
    for (key, value) in pairs(grid.params.vars)
        grid.physical[:,value,1] .= select(initialconditions, key)
    end
end

function check_initialconditions(initialconditions::DataFrame, grid::R_Grid)
    
    # Check for match to grid
    if grid.params.rDim != length(initialconditions.r)
        throw(DomainError(length(initialconditions.r), 
                "IC radius dimension does not match model parameters"))
    end
end

function check_initialconditions(initialconditions::DataFrame, grid::RL_Grid)
    
    # Check for match to grid
    # Can be more sophisticated here, just checking matching dimensions for now
    if (grid.params.lDim) != length(initialconditions.r)
        throw(DomainError(length(initialconditions.r), 
                "IC dimensions do not match model parameters"))
    end
end

function check_initialconditions(initialconditions::DataFrame, grid::RZ_Grid)
    
    # Check for match to grid
    # Can be more sophisticated here, just checking matching dimensions for now
    if (grid.params.rDim * grid.params.zDim) != length(initialconditions.r)
        throw(DomainError(length(initialconditions.r), 
                "IC dimensions do not match model parameters"))
    end
end

function check_initialconditions(initialconditions::DataFrame, grid::RLZ_Grid)

    # Check for match to grid
    # Can be more sophisticated here, just checking matching dimensions for now
    if (grid.params.zDim * grid.params.lDim) != length(initialconditions.r)
        throw(DomainError(length(initialconditions.r), 
                "IC dimensions do not match model parameters"))
    end
end

function write_output(grid::R_Grid, model::ModelParameters, t::real)
    
    time = round(t; digits=2)
    if !isdir(model.output_dir)
        mkdir(model.output_dir)
    end
    
    println("Writing output to $(model.output_dir) at time $time")
    dir = model.output_dir
    afilename = string(dir, "spectral_out_", time, ".csv")
    ufilename = string(dir, "physical_out_", time, ".csv")
    afile = open(afilename,"w")
    ufile = open(ufilename,"w")

    aheader = "r,"
    uheader = "r,"
    suffix = ["","_r","_rr"]
    for d = 1:3
        for var in keys(grid.params.vars)
            if (d == 1)
                aheader *= "$var,"
            end
            varout = var * suffix[d]
            uheader *= "$varout,"
        end
    end
    aheader = chop(aheader) * "\n"
    uheader = chop(uheader) * "\n"
    write(afile,aheader)
    write(ufile,uheader)
    
    for r = 1:grid.params.b_rDim
        astring = "$r,"
        for var in keys(grid.params.vars)
            v = grid.params.vars[var]
            a = grid.spectral[r,v]
            astring *= "$(a),"
        end
        astring = chop(astring) * "\n"
        write(afile,astring)
    end
    close(afile)
    
    for r = 1:grid.params.rDim
        radii = grid.splines[1].mishPoints
        ustring = "$(radii[r]),"
        for d = 1:3
            for var in keys(grid.params.vars)
                v = grid.params.vars[var]
                u = grid.physical[r,v,d]
                ustring *= "$u,"
            end
        end
        ustring = chop(ustring) * "\n"
        write(ufile,ustring)
    end
    close(ufile)
end

function write_output(grid::RZ_Grid, model::ModelParameters, t::real)
    
    time = round(t; digits=2)
    if !isdir(model.output_dir)
        mkdir(model.output_dir)
    end
    
    println("Writing output to $(model.output_dir) at time $time")
    dir = model.output_dir
    afilename = string(dir, "spectral_out_", time, ".csv")
    ufilename = string(dir, "physical_out_", time, ".csv")
    afile = open(afilename,"w")
    ufile = open(ufilename,"w")

    aheader = "r,z,"
    uheader = "r,z,"
    suffix = ["","_r","_rr","_z","_zz"]
    for d = 1:5
        for var in keys(grid.params.vars)
            if (d == 1)
                aheader *= "$var,"
            end
            varout = var * suffix[d]
            uheader *= "$varout,"
        end
    end
    aheader = chop(aheader) * "\n"
    uheader = chop(uheader) * "\n"
    write(afile,aheader)
    write(ufile,uheader)
    
    i = 1
    for z = 1:grid.params.b_zDim
        for r = 1:grid.params.b_rDim
            astring = "$r,$z,"
            for var in keys(grid.params.vars)
                v = grid.params.vars[var]
                z1 = ((r-1)*grid.params.b_zDim)
                a = grid.spectral[i,v]
                astring *= "$(a),"
            end
            i += 1
            astring = chop(astring) * "\n"
            write(afile,astring)
        end
    end
    close(afile)
    
    i = 1
    for r = 1:grid.params.rDim
        for z = 1:grid.params.zDim
            radii = grid.splines[1].mishPoints
            levels = grid.columns[1].mishPoints
            ustring = "$(radii[r]),$(levels[z]),"
            for d = 1:5
                for var in keys(grid.params.vars)
                    v = grid.params.vars[var]
                    u = grid.physical[i,v,d]
                    ustring *= "$u,"
                end
            end
            i += 1
            ustring = chop(ustring) * "\n"
            write(ufile,ustring)
        end
    end
    close(ufile)
end

function write_output(grid::RL_Grid, model::ModelParameters, t::real)
    
    time = round(t; digits=2)
    if !isdir(model.output_dir)
        mkdir(model.output_dir)
    end
    
    println("Writing output to $(model.output_dir) at time $time")
    dir = model.output_dir
    afilename = string(dir, "spectral_out_", time, ".csv")
    ufilename = string(dir, "physical_out_", time, ".csv")
    rfilename = string(dir, "gridded_out_", time, ".csv")
    afile = open(afilename,"w")
    ufile = open(ufilename,"w")
    rfile = open(rfilename,"w")

    aheader = "r,k,"
    uheader = "r,l,x,y,"
    rheader = "r,l,x,y,"
    suffix = ["","_r","_rr","_l","_ll"]
    for d = 1:5
        for var in keys(grid.params.vars)
            if (d == 1)
                aheader *= "$var,"
            end
            varout = var * suffix[d]
            uheader *= "$varout,"
            rheader *= "$varout,"
        end
    end
    aheader = chop(aheader) * "\n"
    uheader = chop(uheader) * "\n"
    rheader = chop(rheader) * "\n"
    write(afile,aheader)
    write(ufile,uheader)
    write(rfile,rheader)
        
    # Wave 0
    for r = 1:grid.params.b_rDim
        astring = "$r,0,"
        for var in keys(grid.params.vars)
            v = grid.params.vars[var]
            a = grid.spectral[r,v]
            astring *= "$(a),"
        end
        astring = chop(astring) * "\n"
        write(afile,astring)
    end
    
    # Higher wavenumbers
    for k = 1:grid.params.rDim
        for r = 1:grid.params.b_rDim
            astring = "$r,$(k)r,"
            kr = ((k*2-1)*grid.params.b_rDim)+r
            for var in keys(grid.params.vars)
                v = grid.params.vars[var]
                a = grid.spectral[kr,v]
                astring *= "$(a),"
            end
            astring = chop(astring) * "\n"
            write(afile,astring)
        end
        for r = 1:grid.params.b_rDim
            astring = "$r,$(k)i,"
            ki = ((k*2+1)*grid.params.b_rDim)+r
            for var in keys(grid.params.vars)
                v = grid.params.vars[var]
                a = grid.spectral[ki,v]
                astring *= "$(a),"
            end
            astring = chop(astring) * "\n"
            write(afile,astring)
        end
    end
    close(afile)
    
    l1 = 0
    l2 = 0
    gridpoints = getGridpoints(grid)
    cartesianpoints = getCartesianGridpoints(grid)
    for r = 1:grid.params.rDim
        l1 = l2 + 1
        l2 = l1 + 3 + (4*r)
        for l = l1:l2
            ustring = "$(gridpoints[l,1]),$(gridpoints[l,2]),$(cartesianpoints[l,1]),$(cartesianpoints[l,2]),"
            for d = 1:5
                for var in keys(grid.params.vars)
                    v = grid.params.vars[var]
                    u = grid.physical[l,v,d]
                    ustring *= "$u,"
                end
            end
            ustring = chop(ustring) * "\n"
            write(ufile,ustring)
        end
    end
    close(ufile)
    
    # Get regular grid
    regular_grid = regularGridTransform(grid)
    gridpoints = getRegularGridpoints(grid)
    for r = 1:grid.params.num_cells
        for l = 1:(grid.params.rDim*2+1)
            rstring = "$(gridpoints[r,l,1]),$(gridpoints[r,l,2]),$(gridpoints[r,l,3]),$(gridpoints[r,l,4]),"
            for d = 1:5
                for var in keys(grid.params.vars)
                    v = grid.params.vars[var]
                    u = regular_grid[r,l,v,d]
                    rstring *= "$u,"
                end
            end
            rstring = chop(rstring) * "\n"
            write(rfile,rstring)
        end
    end
    close(rfile)
    
end

function write_output(grid::RLZ_Grid, model::ModelParameters, t::real)

    time = round(t; digits=2)
    if !isdir(model.output_dir)
        mkdir(model.output_dir)
    end

    println("Writing output to $(model.output_dir) at time $time")
    dir = model.output_dir
    afilename = string(dir, "spectral_out_", time, ".csv")
    ufilename = string(dir, "physical_out_", time, ".csv")
    rfilename = string(dir, "gridded_out_", time, ".csv")
    afile = open(afilename,"w")
    ufile = open(ufilename,"w")
    rfile = open(rfilename,"w")

    aheader = "z,r,k,"
    uheader = "r,l,z,x,y,"
    rheader = "r,l,z,x,y,"
    suffix = ["","_r","_rr","_l","_ll","_z","_zz"]
    for d = 1:7
        for var in keys(grid.params.vars)
            if (d == 1)
                aheader *= "$var,"
            end
            varout = var * suffix[d]
            uheader *= "$varout,"
            rheader *= "$varout,"
        end
    end
    aheader = chop(aheader) * "\n"
    uheader = chop(uheader) * "\n"
    rheader = chop(rheader) * "\n"
    write(afile,aheader)
    write(ufile,uheader)
    write(rfile,rheader)

    kDim = grid.params.rDim + grid.params.patchOffsetL

    for z = 1:grid.params.b_zDim
        r1 = ((z-1) * grid.params.b_rDim * (1 + (kDim * 2))) + 1
        r2 = r1 + grid.params.b_rDim - 1
        # Wave 0
        r = 1
        for ri = r1:r2
            astring = "$z,$r,0,"
            for var in keys(grid.params.vars)
                v = grid.params.vars[var]
                a = grid.spectral[ri,v]
                astring *= "$(a),"
            end
            astring = chop(astring) * "\n"
            write(afile,astring)
            r += 1
        end

        # Higher wavenumbers
        for k = 1:kDim
            p = (k-1)*2
            p1 = r2 + 1 + (p*grid.params.b_rDim)
            p2 = p1 + grid.params.b_rDim - 1
            r = 1
            for kr = p1:p2
                astring = "$z,$r,$(k)r,"
                for var in keys(grid.params.vars)
                    v = grid.params.vars[var]
                    a = grid.spectral[kr,v]
                    astring *= "$(a),"
                end
                astring = chop(astring) * "\n"
                write(afile,astring)
                r += 1
            end
            p1 = p2 + 1
            p2 = p1 + grid.params.b_rDim - 1
            r = 1
            for ki = p1:p2
                astring = "$z,$ki,$(k)i,"
                for var in keys(grid.params.vars)
                    v = grid.params.vars[var]
                    a = grid.spectral[ki,v]
                    astring *= "$(a),"
                end
                astring = chop(astring) * "\n"
                write(afile,astring)
                r += 1
            end
        end
    end
    close(afile)

    gridpoints = getGridpoints(grid)
    cartesianpoints = getCartesianGridpoints(grid)
    for i in eachindex(gridpoints[:,1])
        ustring = "$(gridpoints[i,1]),$(gridpoints[i,2]),$(gridpoints[i,3]),$(cartesianpoints[i,1]),$(cartesianpoints[i,2]),"
        for d = 1:7
            for var in keys(grid.params.vars)
                v = grid.params.vars[var]
                u = grid.physical[i,v,d]
                ustring *= "$u,"
            end
        end
        ustring = chop(ustring) * "\n"
        write(ufile,ustring)
    end
    close(ufile)

    # Get regular grid
    regular_grid = regularGridTransform(grid)
    gridpoints = getRegularGridpoints(grid)
    for i in eachindex(gridpoints[:,1])
        rstring = "$(gridpoints[i,1]),$(gridpoints[i,2]),$(gridpoints[i,3]),$(gridpoints[i,4]),$(gridpoints[i,5]),"
        for d = 1:7
            for var in keys(grid.params.vars)
                v = grid.params.vars[var]
                u = regular_grid[i,v,d]
                rstring *= "$u,"
            end
        end
        rstring = chop(rstring) * "\n"
        write(rfile,rstring)
    end
    close(rfile)

end
