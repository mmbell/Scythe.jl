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

    @testset "RL radial nest (Twoway_PV_mixing layout)" begin
        pv_vars = Dict("h" => 1, "u" => 2, "v" => 3, "ub" => 4, "vb" => 5, "wb" => 6)
        base_rl = ModelParameters(
            ts = 3.0,
            integration_time = 1800.0,
            output_interval = 1800.0,
            equation_set = "Twoway_PV_mixing",
            initial_conditions = "",
            output_dir = "./output_nesttest_rl/",
            grid_params = GridParameters(
                geometry = "RL",
                num_cells = 100, iMin = 0.0, iMax = 3.0e5,   # placeholder
                BCL = Dict(v => (v in ("h", "wb") ? NeumannBC() : DirichletBC()) for v in keys(pv_vars)),
                BCR = Dict(v => NaturalBC() for v in keys(pv_vars)),
                vars = pv_vars),
            physical_params = Dict(:g => 9.81),
        )

        # 1:1 split (same resolution both sides; left patch is parent)
        nest11 = NestedModelParameters(
            boundaries = [0.0, 1.5e5, 3.0e5],
            num_cells = [50, 50],
            ts = [3.0, 3.0],
            workers_per_patch = [1, 1],
            base = base_rl)
        models, topo = build_nest(nest11)
        @test topo.n_sub == [1, 1]
        @test models[1].grid_params.iMax == 1.53e5          # +1 own cell collar
        @test models[1].grid_params.num_cells == 51
        @test models[1].grid_params.patchOffsetL == 0
        @test models[2].grid_params.iMin == 1.5e5
        @test models[2].grid_params.patchOffsetL == 50 * models[2].grid_params.mubar
        @test models[2].grid_params.BCL["h"] == FixedBC()
        @test models[2].grid_params.BCR["h"] == NaturalBC()  # outer from base
        ni = topo.interfaces[1]
        @test (ni.parent, ni.child) == (1, 2)
        @test ni.meta.coupling_matrix == Springsteel.COUPLING_MATRIX_1X
        @test ni.nslices == 5
        # Collar = mubar rings past 150 km; ragged rows with per-ring kmax
        mubar = models[1].grid_params.mubar
        gp1 = models[1].grid_params
        expected_rows = sum(4 + 4 * ri for ri in (gp1.iDim - mubar + 1):gp1.iDim)
        @test length(ni.collar_rows) == expected_rows
        @test size(ni.collar_pts, 1) == expected_rows
        @test length(ni.collar_kmax) == expected_rows
        @test all(ni.collar_pts[:, 1] .> 1.5e5)
        @test minimum(ni.collar_kmax) == gp1.iDim - mubar + 1

        # 2:1: fine inner disc, coarse outer annulus (parent on the right)
        nest21 = NestedModelParameters(
            boundaries = [0.0, 1.5e5, 3.0e5],
            num_cells = [50, 25],
            ts = [3.0, 6.0],
            workers_per_patch = [1, 1],
            base = base_rl)
        models2, topo2 = build_nest(nest21)
        ni2 = topo2.interfaces[1]
        @test (ni2.parent, ni2.child) == (2, 1)
        @test topo2.n_sub == [2, 1]
        @test topo2.ts_actual == [3.0, 6.0]
        @test models2[2].grid_params.iMin == 1.44e5          # −1 own (6 km) cell collar
        @test models2[2].grid_params.num_cells == 26
        @test models2[2].grid_params.patchOffsetL == 24 * models2[2].grid_params.mubar
        @test models2[1].grid_params.BCR["h"] == FixedBC()
        @test all(ni2.collar_pts[:, 1] .< 1.5e5)
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
