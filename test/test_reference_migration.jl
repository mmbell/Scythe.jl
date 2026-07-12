using Test
using Scythe
using Springsteel

# Guards the Scythe/Springsteel reference-state boundary.
#
# Springsteel exports only the reference-state *types*, not the functions, so a same-named
# definition in Scythe silently becomes a *separate* generic function — no method merge, no
# ambiguity error, no compiler signal. That is exactly how Scythe's copy of `natural_column`
# came to size a spline column as `kDim ÷ mubar` while Springsteel's had already moved to the
# canonical `num_cells_k`, and it is why supplying `num_cells_k` (with no `kDim`) used to build
# a degenerate zero-cell / zero-point column.
#
# These tests assert (1) Scythe has not re-forked those functions, and (2) sizing a grid by cell
# count and by gridpoint count are interchangeable.

@testset "Reference state: Springsteel is the single source" begin

    @testset "No forked definitions" begin
        # Identity, not just equality of results: if someone re-defines any of these inside
        # Scythe, the binding stops being Springsteel's and this fails immediately.
        @test Scythe.natural_column === Springsteel.natural_column
        @test Scythe.reference_column === Springsteel.reference_column
        @test Scythe.transform_reference_state! === Springsteel.transform_reference_state!
    end

    vars = Dict("s" => 1, "xi" => 2, "mu" => 3, "u" => 4, "w" => 5)
    bc = Dict(v => NeumannBC() for v in keys(vars))

    grid_params(; kw...) = GridParameters(; iMin=0.0, iMax=8000.0, kMin=0.0, kMax=6000.0,
                                          BCL=bc, BCR=bc, BCB=bc, BCT=bc, vars=vars, kw...)

    model_for(gp, sounding) = ModelParameters(ts=0.1, equation_set="Euler_test",
        ref_state_file=sounding, grid_params=gp, physical_params=Dict(:K => 0.0))

    @testset "ModelParameters resolves cell and gridpoint sizing identically" begin
        # RiRk: spline vertical. mubar = 3, so 16 cells <=> kDim 48.
        by_cells = model_for(grid_params(geometry="RiRk", num_cells_i=16, num_cells_k=16), "")
        by_dim   = model_for(grid_params(geometry="RiRk", num_cells_i=16, kDim=48), "")

        for gp in (by_cells.grid_params, by_dim.grid_params)
            @test gp.kDim == 48
            @test gp.num_cells_k == 16
        end

        # RZ: Chebyshev vertical has no cells, so kDim is the only way to size it and
        # num_cells_k stays 0. Sizing it by cell count would silently give a zero-height column.
        rz = model_for(grid_params(geometry="RZ", num_cells_i=16, kDim=48), "")
        @test rz.grid_params.kDim == 48
        @test rz.grid_params.num_cells_k == 0
    end

    @testset "Cell- and gridpoint-sized grids give the same reference state" begin
        mktempdir() do tmp
            sounding = Scythe.write_dry_sounding(joinpath(tmp, "dry.ref");
                                                 theta=300.0, zmax=6000.0)

            function build(gp)
                model = model_for(gp, sounding)
                patch = createGrid(model.grid_params)
                z = getGridpoints(patch)[1:model.grid_params.kDim, 2]
                column = Scythe.reference_column(patch, model.grid_params)
                return Scythe.calculate_reference_state(model, z, column)
            end

            a = build(grid_params(geometry="RiRk", num_cells_i=16, num_cells_k=16))
            b = build(grid_params(geometry="RiRk", num_cells_i=16, kDim=48))

            for f in (:sbar, :xibar, :rhobar, :mubar, :satbar)
                @test getfield(a, f) == getfield(b, f)
            end
            @test a.Pxi_bar == b.Pxi_bar
        end
    end

    @testset "calculate_reference_state delegates to the shared builder" begin
        mktempdir() do tmp
            gp = grid_params(geometry="RZ", num_cells_i=16, kDim=48)
            sounding = Scythe.write_dry_sounding(joinpath(tmp, "dry.ref");
                                                 theta=300.0, zmax=6000.0)
            model = model_for(gp, sounding)
            patch = createGrid(model.grid_params)
            z = getGridpoints(patch)[1:gp.kDim, 2]
            column = Scythe.reference_column(patch, model.grid_params)

            got = Scythe.calculate_reference_state(model, z, column)
            want = Scythe.legacy_reference_view(
                Springsteel.calculate_reference_state(sounding, z, column; moisture=true),
                column)

            for f in (:sbar, :xibar, :rhobar, :mubar, :satbar)
                @test getfield(got, f) == getfield(want, f)
            end
            @test got.Pxi_bar == want.Pxi_bar
        end
    end
end
