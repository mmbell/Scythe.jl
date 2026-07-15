using Test
using Scythe
using Springsteel
using NCDatasets

# Configurable analysis output format (write_output on options[:output_formats])
# and JLD2 restart checkpointing (write_restart / load_initial_conditions!).
# In-process, no workers: these functions take a grid and a ModelParameters, so
# the dispatch and the restart round-trip are exercised directly without the
# distributed machinery. The underlying writers (write_grid/save_grid/
# write_netcdf) are covered by Springsteel's test/io.jl — here we test that
# Scythe selects the right ones, names the files, and the checkpoint reloads
# the grid state exactly.

@testset "Output format selection" begin

    # A small RL 2-D grid with a non-trivial, azimuthally-varying field so the
    # checkpoint reload and the gridded NetCDF have real content to compare.
    gp = SpringsteelGridParameters(
        geometry = "RL", num_cells = 6,
        iMin = 0.0, iMax = 60.0,
        vars = Dict("u" => 1, "v" => 2),
        BCL = Dict("u" => Springsteel.CubicBSpline.R0,
                   "v" => Springsteel.CubicBSpline.R0),
        BCR = Dict("u" => Springsteel.CubicBSpline.R0,
                   "v" => Springsteel.CubicBSpline.R0))

    make_grid() = begin
        g = createGrid(gp)
        pts = getGridpoints(g)
        for p in 1:size(pts, 1)
            r = pts[p, 1]; λ = pts[p, 2]
            g.physical[p, 1, 1] = sin(r / 60.0) * cos(λ)
            g.physical[p, 2, 1] = cos(r / 60.0) * sin(2λ)
        end
        spectralTransform!(g)
        gridTransform!(g)
        g
    end

    model(dir; formats = nothing, derivs = nothing, ics = "") = begin
        opts = Dict{Symbol,Any}()
        formats === nothing || (opts[:output_formats] = formats)
        derivs === nothing || (opts[:netcdf_derivatives] = derivs)
        ModelParameters(
            equation_set = "Twoway_PV_mixing",
            initial_conditions = ics,
            output_dir = dir,
            grid_params = gp,
            options = opts)
    end

    @testset "default is CSV (backward compatible)" begin
        dir = mktempdir()
        Scythe.write_output(make_grid(), model(dir), 0.0)
        @test isfile(joinpath(dir, "0.0_spectral.csv"))
        @test isfile(joinpath(dir, "0.0_physical.csv"))
        @test isfile(joinpath(dir, "0.0_gridded.csv"))
        # No opt-in formats requested → no nc emitted.
        @test !isfile(joinpath(dir, "0.0.nc"))
    end

    @testset "multi-format [:csv, :netcdf] all emitted" begin
        dir = mktempdir()
        Scythe.write_output(make_grid(), model(dir; formats = [:csv, :netcdf]), 0.0)
        for f in ("0.0_spectral.csv", "0.0_physical.csv", "0.0_gridded.csv", "0.0.nc")
            @test isfile(joinpath(dir, f))
        end
    end

    @testset ":jld2 in output_formats is rejected (it is the restart format)" begin
        dir = mktempdir()
        @test_throws ErrorException Scythe.write_output(
            make_grid(), model(dir; formats = [:jld2]), 0.0)
    end

    @testset "NetCDF values-only by default" begin
        dir = mktempdir()
        Scythe.write_output(make_grid(), model(dir; formats = [:netcdf]), 0.0)
        NCDataset(joinpath(dir, "0.0.nc"), "r") do ds
            @test haskey(ds, "u")
            @test haskey(ds, "v")
            # time coordinate carries the snapshot time. Read via `.var` to get
            # the raw stored value (the CF units decode it to a DateTime otherwise).
            @test ds["time"].var[1] == 0.0
            # derivative slots absent when include_derivatives=false (RL suffix _r)
            @test !haskey(ds, "u_r")
            @test !haskey(ds, "u_rr")
        end
    end

    @testset "NetCDF derivatives knob" begin
        dir = mktempdir()
        Scythe.write_output(make_grid(),
                            model(dir; formats = [:netcdf], derivs = true), 0.0)
        NCDataset(joinpath(dir, "0.0.nc"), "r") do ds
            @test haskey(ds, "u")
            @test haskey(ds, "u_r")     # radial derivative slot now present
            @test haskey(ds, "u_az")    # azimuthal derivative slot
        end
    end

    @testset "unknown format errors" begin
        dir = mktempdir()
        @test_throws ErrorException Scythe.write_output(
            make_grid(), model(dir; formats = [:parquet]), 0.0)
    end

    # ── Restart checkpointing (write_restart / load_initial_conditions!) ──────
    # NOTE: the checkpoint FILE round-trips the grid state bit-for-bit (asserted
    # below). A full run CONTINUATION from a checkpoint is only a warm restart
    # (the AB3 integrator's tendency history is not in the grid) — that is not
    # exercised here (it needs a distributed run).
    @testset "restart checkpoint reloads the grid state exactly" begin
        dir = mktempdir()
        g = make_grid()
        Scythe.write_restart(g, model(dir), 3600.0)
        ckpt = joinpath(dir, "3600.0.jld2")
        @test isfile(ckpt)

        # Load the checkpoint into a fresh grid via the IC loader and confirm the
        # prognostic (spectral) state is bit-identical to what was written.
        patch = createGrid(gp)
        Scythe.load_initial_conditions!(patch, model(dir; ics = ckpt))
        @test patch.spectral == g.spectral
        @test all(isfinite, patch.physical)
    end

    @testset "CSV initial_conditions still load" begin
        dir = mktempdir()
        g = make_grid()
        # Write a CSV IC (physical only) and read it back through the loader.
        write_grid(g, dir, "ic")
        patch = createGrid(gp)
        Scythe.load_initial_conditions!(patch, model(dir; ics = joinpath(dir, "ic_physical.csv")))
        # CSV stores base values only; spectralTransform! re-fits, so the reload
        # is close but not exact (the .jld2 checkpoint stores the grid verbatim).
        @test all(isfinite, patch.spectral)
        @test maximum(abs.(patch.physical[:, 1, 1] .- g.physical[:, 1, 1])) < 1e-3
    end

    @testset "incompatible restart archive errors" begin
        dir = mktempdir()
        Scythe.write_restart(make_grid(), model(dir), 0.0)
        # A grid with a different variable map is not restart-compatible.
        other_gp = SpringsteelGridParameters(
            geometry = "RL", num_cells = 6, iMin = 0.0, iMax = 60.0,
            vars = Dict("u" => 1),
            BCL = Dict("u" => Springsteel.CubicBSpline.R0),
            BCR = Dict("u" => Springsteel.CubicBSpline.R0))
        other = ModelParameters(equation_set = "Twoway_PV_mixing",
            initial_conditions = joinpath(dir, "0.0.jld2"),
            output_dir = dir, grid_params = other_gp, options = Dict{Symbol,Any}())
        @test_throws ErrorException Scythe.load_initial_conditions!(createGrid(other_gp), other)
    end
end
