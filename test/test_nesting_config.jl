using Test
using Scythe
using Springsteel

# Stage 5 of the grid-nesting plan: NestedModelParameters + build_nest —
# per-patch grids with collar extensions and interface BCs, derived
# per-patch timesteps, and the NestTopology consumed by the drivers.

@testset "Nest configuration (build_nest)" begin

    base_1d(; ts=0.1) = ModelParameters(
        ts = ts,
        integration_time = 80.0,
        output_interval = 80.0,
        equation_set = "LinearAdvection1D",
        initial_conditions = "",
        output_dir = "./output_nesttest/",
        grid_params = GridParameters(
            geometry = "R",
            num_cells = 10, iMin = 0.0, iMax = 10.0,   # placeholder, ignored
            BCL = Dict("u" => DirichletBC()),
            BCR = Dict("u" => NaturalBC()),
            vars = Dict("u" => 1)),
        physical_params = Dict(:c_0 => 1.0, :K => 0.0),
    )

    @testset "3-patch 1D chain" begin
        nest = NestedModelParameters(
            boundaries = [-60.0, -20.0, 20.0, 60.0],
            num_cells = [40, 80, 40],
            ts = [0.1, 0.05, 0.1],
            workers_per_patch = [1, 1, 1],
            base = base_1d())
        models, topo = build_nest(nest)

        @test length(models) == 3
        # Parent collars: patch 1 extends to −19, patch 3 to +19
        @test models[1].grid_params.iMin == -60.0
        @test models[1].grid_params.iMax == -19.0
        @test models[1].grid_params.num_cells == 41
        @test models[2].grid_params.iMin == -20.0
        @test models[2].grid_params.iMax == 20.0
        @test models[2].grid_params.num_cells == 80
        @test models[3].grid_params.iMin == 19.0
        @test models[3].grid_params.iMax == 60.0
        @test models[3].grid_params.num_cells == 41

        # BCs: outer from base; collar terminations natural; child sides R3X
        @test models[1].grid_params.BCL["u"] == DirichletBC()
        @test models[1].grid_params.BCR["u"] == NaturalBC()
        @test models[2].grid_params.BCL["u"] == FixedBC()
        @test models[2].grid_params.BCR["u"] == FixedBC()
        @test models[3].grid_params.BCL["u"] == NaturalBC()
        @test models[3].grid_params.BCR["u"] == NaturalBC()

        # Timesteps and subcycling
        @test topo.ts_actual == [0.1, 0.05, 0.1]
        @test topo.n_sub == [1, 2, 1]
        @test models[2].ts == 0.05

        # Output dirs
        @test endswith(models[1].output_dir, joinpath("output_nesttest", "nest1"))
        @test endswith(models[3].output_dir, joinpath("output_nesttest", "nest3"))

        # Topology
        @test length(topo.interfaces) == 2
        i1, i2 = topo.interfaces
        @test (i1.parent, i1.child, i1.parent_side) == (1, 2, :right)
        @test (i2.parent, i2.child, i2.parent_side) == (3, 2, :left)
        @test i1.meta.coupling_matrix == Springsteel.COUPLING_MATRIX_2X
        @test i1.meta.is_stacked
        mubar = models[1].grid_params.mubar
        @test length(i1.collar_x) == mubar
        @test all(x -> -20.0 < x < -19.0, i1.collar_x)
        @test i1.collar_rows == collect(41 * mubar - mubar + 1 : 41 * mubar)
        @test all(x -> 19.0 < x < 20.0, i2.collar_x)
        @test i1.nslices == 3
        @test topo.parent_ifaces == [[], [1, 2], []]
        @test topo.child_ifaces == [[1], [], [2]]
    end

    @testset "float-ratio timestep derivation" begin
        nest = NestedModelParameters(
            boundaries = [-60.0, -20.0, 20.0, 60.0],
            num_cells = [40, 80, 40],
            ts = [0.09, 0.05, 0.09],
            workers_per_patch = [1, 1, 1],
            base = base_1d(ts=0.09))
        _, topo = build_nest(nest)
        @test topo.n_sub == [1, 2, 1]
        @test topo.ts_actual[2] ≈ 0.045
    end

    @testset "5-patch O01-style RiRk chain" begin
        base = ModelParameters(
            ts = 0.6,
            integration_time = 3600.0,
            output_interval = 60.0,
            equation_set = "moist_compressible_XZ",
            initial_conditions = "",
            output_dir = "./output_nesttest_o01/",
            grid_params = GridParameters(
                geometry = "RiRk",
                num_cells = 10, iMin = 0.0, iMax = 10.0,   # placeholder
                kMin = 0.0, kMax = 20000.0, num_cells_k = 40,
                BCL = Dict("u" => DirichletBC(), "w" => DirichletBC()),
                BCR = Dict("u" => DirichletBC(), "w" => DirichletBC()),
                BCB = Dict("u" => NeumannBC(), "w" => DirichletBC()),
                BCT = Dict("u" => NeumannBC(), "w" => DirichletBC()),
                vars = Dict("u" => 1, "w" => 2)),
        )
        nest = NestedModelParameters(
            boundaries = [0.0, 50.0e3, 63.0e3, 87.0e3, 100.0e3, 150.0e3],
            num_cells = [25, 13, 48, 13, 25],
            ts = [0.6, 0.3, 0.15, 0.3, 0.6],
            workers_per_patch = [1, 1, 4, 1, 1],
            base = base)
        models, topo = build_nest(nest)

        @test topo.n_sub == [1, 2, 2, 2, 1]
        @test topo.ts_actual == [0.6, 0.3, 0.15, 0.3, 0.6]
        # Middle 1-km patches are both child (outer side) and parent (inner)
        @test models[2].grid_params.iMin == 50.0e3
        @test models[2].grid_params.iMax == 64.0e3          # +1 km collar into N3
        @test models[2].grid_params.num_cells == 14
        @test models[2].grid_params.BCL["u"] == FixedBC()
        @test models[2].grid_params.BCR["u"] == NaturalBC()
        @test models[3].grid_params.iMin == 63.0e3
        @test models[3].grid_params.iMax == 87.0e3
        @test models[3].grid_params.BCL["u"] == FixedBC()
        @test models[3].grid_params.BCR["u"] == FixedBC()
        @test models[4].grid_params.iMin == 86.0e3
        @test models[4].grid_params.iMax == 100.0e3

        # Vertical inherited everywhere
        for m in models
            @test m.grid_params.num_cells_k == 40
            @test m.grid_params.kMax == 20000.0
        end

        # RiRk collar rows: one column of kDim rows per collar mish point
        kDim = models[2].grid_params.kDim
        i23 = topo.interfaces[2]
        @test (i23.parent, i23.child) == (2, 3)
        @test i23.nslices == 5
        @test length(i23.collar_rows) == length(i23.collar_x) * kDim
        @test all(x -> 63.0e3 < x < 64.0e3, i23.collar_x)
    end

    @testset "validation errors" begin
        # 4:1 junction ratio
        nest_bad = NestedModelParameters(
            boundaries = [-60.0, -20.0, 20.0, 60.0],
            num_cells = [10, 80, 10],
            ts = [0.1, 0.05, 0.1],
            workers_per_patch = [1, 1, 1],
            base = base_1d())
        @test_throws ArgumentError build_nest(nest_bad)

        # Parentless patches with different ts
        nest_ts = NestedModelParameters(
            boundaries = [-60.0, -20.0, 20.0, 60.0],
            num_cells = [40, 80, 40],
            ts = [0.1, 0.05, 0.2],
            workers_per_patch = [1, 1, 1],
            base = base_1d())
        @test_throws ArgumentError build_nest(nest_ts)

        # Child too narrow for its parents' collars
        nest_narrow = NestedModelParameters(
            boundaries = [-60.0, -1.0, 1.0, 60.0],
            num_cells = [59, 4, 59],
            ts = [0.1, 0.05, 0.1],
            workers_per_patch = [1, 1, 1],
            base = base_1d())
        @test_throws ArgumentError build_nest(nest_narrow)
    end
end
