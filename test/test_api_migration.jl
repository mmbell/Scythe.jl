using Test
using Scythe
using Springsteel
using SparseArrays
using SharedArrays

@testset "API Migration v0.3.0" begin

    # ──────────────────────────────────────────────
    # 1. GridParameters backward compatibility
    # ──────────────────────────────────────────────
    @testset "GridParameters backward compat" begin
        bc_dict = Dict("u" => Springsteel.CubicBSpline.R0)
        gp = GridParameters(
            geometry = "R",
            num_cells = 10,
            xmin = -50.0,
            xmax = 50.0,
            BCL = bc_dict,
            BCR = bc_dict,
            vars = Dict("u" => 1),
        )
        grid = createGrid(gp)
        @test grid !== nothing
        @test grid isa AbstractGrid
        @test size(grid.physical, 2) == 1  # 1 variable
    end

    # ──────────────────────────────────────────────
    # 2. SpringsteelGridParameters support
    # ──────────────────────────────────────────────
    @testset "SpringsteelGridParameters support" begin
        sgp = SpringsteelGridParameters(
            geometry = "R",
            num_cells = 10,
            iMin = -50.0,
            iMax = 50.0,
            BCL = Dict("default" => Springsteel.CubicBSpline.R0),
            BCR = Dict("default" => Springsteel.CubicBSpline.R0),
            vars = Dict("u" => 1),
        )
        grid = createGrid(sgp)
        @test grid !== nothing
        @test grid isa AbstractGrid
        @test size(grid.physical, 2) == 1
    end

    # ──────────────────────────────────────────────
    # 3. Grid field access — ibasis.data for R grid
    # ──────────────────────────────────────────────
    @testset "Grid field access — R grid ibasis" begin
        bc_dict = Dict("u" => Springsteel.CubicBSpline.R0)
        gp = GridParameters(
            geometry = "R",
            num_cells = 10,
            xmin = -50.0,
            xmax = 50.0,
            BCL = bc_dict,
            BCR = bc_dict,
            vars = Dict("u" => 1),
        )
        grid = createGrid(gp)
        @test hasproperty(grid, :ibasis)
        @test hasproperty(grid.ibasis, :data)
        @test length(grid.ibasis.data) >= 1
    end

    # ──────────────────────────────────────────────
    # 3b. Grid field access — kbasis.data for RZ grid
    # ──────────────────────────────────────────────
    @testset "Grid field access — RZ grid kbasis" begin
        bc_dict = Dict("u" => Springsteel.CubicBSpline.R0)
        bc_z = Dict("u" => Springsteel.Chebyshev.R0)
        gp = GridParameters(
            geometry = "RZ",
            num_cells = 4,
            xmin = 0.0,
            xmax = 100.0,
            zmin = 0.0,
            zmax = 1000.0,
            zDim = 10,
            BCL = bc_dict,
            BCR = bc_dict,
            BCB = bc_z,
            BCT = bc_z,
            vars = Dict("u" => 1),
        )
        grid = createGrid(gp)
        @test hasproperty(grid, :kbasis)
        @test hasproperty(grid.kbasis, :data)
        @test length(grid.kbasis.data) >= 1
    end

    # ──────────────────────────────────────────────
    # 4. calcTileSizes returns Vector{SpringsteelGrid}
    # ──────────────────────────────────────────────
    @testset "calcTileSizes returns Vector" begin
        bc_dict = Dict("u" => Springsteel.CubicBSpline.R0)
        gp = GridParameters(
            geometry = "R",
            num_cells = 10,
            xmin = -50.0,
            xmax = 50.0,
            BCL = bc_dict,
            BCR = bc_dict,
            vars = Dict("u" => 1),
        )
        patch = createGrid(gp)
        tiles = calcTileSizes(patch, 2)
        @test tiles isa Vector
        @test length(tiles) == 2
        @test tiles[1] isa AbstractGrid
        @test hasproperty(tiles[1].params, :iDim)
        @test hasproperty(tiles[1].params, :iMin)
        @test hasproperty(tiles[1].params, :iMax)
        @test hasproperty(tiles[1].params, :num_cells)
        @test hasproperty(tiles[1].params, :spectralIndexL)
    end

    # ──────────────────────────────────────────────
    # 5. calcPatchMap returns SparseMatrixCSC
    # ──────────────────────────────────────────────
    @testset "calcPatchMap returns SparseMatrixCSC" begin
        bc_dict = Dict("u" => Springsteel.CubicBSpline.R0)
        gp = GridParameters(
            geometry = "R",
            num_cells = 10,
            xmin = -50.0,
            xmax = 50.0,
            BCL = bc_dict,
            BCR = bc_dict,
            vars = Dict("u" => 1),
        )
        patch = createGrid(gp)
        tiles = calcTileSizes(patch, 2)
        pmap = calcPatchMap(patch, tiles[1])
        @test pmap isa SparseMatrixCSC{Float64, Int64}
    end

    # ──────────────────────────────────────────────
    # 6. calcHaloMap returns SparseMatrixCSC
    # ──────────────────────────────────────────────
    @testset "calcHaloMap returns SparseMatrixCSC" begin
        bc_dict = Dict("u" => Springsteel.CubicBSpline.R0)
        gp = GridParameters(
            geometry = "R",
            num_cells = 10,
            xmin = -50.0,
            xmax = 50.0,
            BCL = bc_dict,
            BCR = bc_dict,
            vars = Dict("u" => 1),
        )
        patch = createGrid(gp)
        tiles = calcTileSizes(patch, 2)
        hmap = calcHaloMap(patch, tiles[1])
        @test hmap isa SparseMatrixCSC{Float64, Int64}
    end

    # ──────────────────────────────────────────────
    # 7. Transform 2-arg signatures
    # ──────────────────────────────────────────────
    @testset "splineTransform! 2-arg signature" begin
        bc_dict = Dict("u" => Springsteel.CubicBSpline.R0)
        gp = GridParameters(
            geometry = "R",
            num_cells = 10,
            xmin = -50.0,
            xmax = 50.0,
            BCL = bc_dict,
            BCR = bc_dict,
            vars = Dict("u" => 1),
        )
        patch = createGrid(gp)
        tiles = calcTileSizes(patch, 1)
        tile = tiles[1]

        # Set up a SharedArray for the spectral data
        shared = SharedArray{Float64}(size(patch.spectral))
        shared .= patch.spectral

        # 2-arg splineTransform! should work without error
        splineTransform!(shared, tile)
        @test true  # reached here without error
    end

    # ──────────────────────────────────────────────
    # 8. ModelTile construction with new field types
    # ──────────────────────────────────────────────
    @testset "ModelTile construction" begin
        bc_dict = Dict("u" => Springsteel.CubicBSpline.R0)
        gp = GridParameters(
            geometry = "R",
            num_cells = 10,
            xmin = -50.0,
            xmax = 50.0,
            BCL = bc_dict,
            BCR = bc_dict,
            vars = Dict("u" => 1),
        )
        model = ModelParameters(
            grid_params = gp,
            equation_set = "LinearAdvection1D",
            physical_params = Dict(:c_0 => 1.0, :K => 0.0),
        )
        patch = createGrid(model.grid_params)
        tiles = calcTileSizes(patch, 1)
        tile = tiles[1]

        # Create trivial halo receive map
        haloReceiveMap = sparse([1], [1], [1.0], size(patch.spectral, 1), size(patch.spectral, 2))

        mtile = createModelTile(patch, tile, model, haloReceiveMap)
        @test mtile isa Scythe.ModelTile
        @test mtile.patchMap isa SparseMatrixCSC{Float64, Int64}
        @test mtile.haloSendMap isa SparseMatrixCSC{Float64, Int64}
        @test mtile.haloReceiveMap isa SparseMatrixCSC{Float64, Int64}
    end

end
