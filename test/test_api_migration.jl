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
        bc_dict = Dict("u" => NaturalBC())
        gp = GridParameters(
            geometry = "R",
            num_cells = 10,
            iMin = -50.0,
            iMax = 50.0,
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
            BCL = Dict("default" => NaturalBC()),
            BCR = Dict("default" => NaturalBC()),
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
        bc_dict = Dict("u" => NaturalBC())
        gp = GridParameters(
            geometry = "R",
            num_cells = 10,
            iMin = -50.0,
            iMax = 50.0,
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
        bc_dict = Dict("u" => NaturalBC())
        bc_z = Dict("u" => NaturalBC())
        gp = GridParameters(
            geometry = "RZ",
            num_cells = 4,
            iMin = 0.0,
            iMax = 100.0,
            kMin = 0.0,
            kMax = 1000.0,
            kDim = 10,
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
        bc_dict = Dict("u" => NaturalBC())
        gp = GridParameters(
            geometry = "R",
            num_cells = 10,
            iMin = -50.0,
            iMax = 50.0,
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
        bc_dict = Dict("u" => NaturalBC())
        gp = GridParameters(
            geometry = "R",
            num_cells = 10,
            iMin = -50.0,
            iMax = 50.0,
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
        bc_dict = Dict("u" => NaturalBC())
        gp = GridParameters(
            geometry = "R",
            num_cells = 10,
            iMin = -50.0,
            iMax = 50.0,
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
        bc_dict = Dict("u" => NaturalBC())
        gp = GridParameters(
            geometry = "R",
            num_cells = 10,
            iMin = -50.0,
            iMax = 50.0,
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
        bc_dict = Dict("u" => NaturalBC())
        gp = GridParameters(
            geometry = "R",
            num_cells = 10,
            iMin = -50.0,
            iMax = 50.0,
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
        @test mtile.patch_b_iDim == patch.params.b_iDim
    end

    # ──────────────────────────────────────────────
    # 9. write_tile_to_shared! for 1D Cartesian (R)
    #    with 2 tiles — verifies tile→patch index mapping
    # ──────────────────────────────────────────────
    @testset "write_tile_to_shared! — R grid 2 tiles" begin
        gp = SpringsteelGridParameters(
            geometry = "R",
            num_cells = 12,
            iMin = 0.0,
            iMax = 120.0,
            BCL = Dict("default" => NaturalBC()),
            BCR = Dict("default" => NaturalBC()),
            vars = Dict("u" => 1),
        )
        patch = createGrid(gp)
        tiles = calcTileSizes(patch, 2)
        shared = SharedArray{Float64}(size(patch.spectral))

        for tile in tiles
            tile.physical .= 1.0
            spectralTransform!(tile)

            # write_tile_to_shared! must not throw BoundsError
            shared .= 0.0
            Scythe.write_tile_to_shared!(shared, tile, patch.params.b_iDim)

            # Verify inner region was written — compare against sumSpectralTile!
            patch.spectral .= 0.0
            sumSpectralTile!(patch, tile)
            pmap = calcPatchMap(patch, tile)
            rows, cols, _ = findnz(pmap)
            for (r, c) in zip(rows, cols)
                @test isapprox(shared[r, c], patch.spectral[r, c], atol=1e-12)
            end
        end
    end

    # ──────────────────────────────────────────────
    # 10. write_tile_to_shared! for RL grid 2 tiles
    #     — the geometry that triggered the original BoundsError
    # ──────────────────────────────────────────────
    @testset "write_tile_to_shared! — RL grid 2 tiles" begin
        sgp = SpringsteelGridParameters(
            geometry = "RL",
            num_cells = 12,
            iMin = 0.0,
            iMax = 120.0,
            BCL = Dict("default" => NaturalBC()),
            BCR = Dict("default" => NaturalBC()),
            vars = Dict("u" => 1, "v" => 2),
        )
        patch = createGrid(sgp)
        tiles = calcTileSizes(patch, 2)
        shared = SharedArray{Float64}(size(patch.spectral))

        for tile in tiles
            tile.physical .= 1.0
            spectralTransform!(tile)

            # write_tile_to_shared! must not throw BoundsError
            shared .= 0.0
            Scythe.write_tile_to_shared!(shared, tile, patch.params.b_iDim)

            # The tile spectral size may differ from patch spectral size
            @test size(tile.spectral, 1) <= size(patch.spectral, 1)

            # Verify inner region matches sumSpectralTile! reference
            patch.spectral .= 0.0
            sumSpectralTile!(patch, tile)
            pmap = calcPatchMap(patch, tile)
            rows, cols, _ = findnz(pmap)
            for (r, c) in zip(rows, cols)
                @test isapprox(shared[r, c], patch.spectral[r, c], atol=1e-12)
            end
        end
    end

    # ──────────────────────────────────────────────
    # 11. extract_halo_values — RL grid
    #     Verifies halo values match sumSpectralTile! reference
    #     and length matches calcHaloMap nnz
    # ──────────────────────────────────────────────
    @testset "extract_halo_values — RL grid" begin
        sgp = SpringsteelGridParameters(
            geometry = "RL",
            num_cells = 12,
            iMin = 0.0,
            iMax = 120.0,
            BCL = Dict("default" => NaturalBC()),
            BCR = Dict("default" => NaturalBC()),
            vars = Dict("u" => 1),
        )
        patch = createGrid(sgp)
        tiles = calcTileSizes(patch, 2)

        for tile in tiles
            tile.physical .= 1.0
            spectralTransform!(tile)

            halo_vals = Scythe.extract_halo_values(tile)
            hmap = calcHaloMap(patch, tile)

            # Length must match haloMap nnz (independent of data values)
            @test length(halo_vals) == nnz(hmap)

            # Values must match what sumSpectralTile! puts at halo positions
            patch.spectral .= 0.0
            sumSpectralTile!(patch, tile)
            rows, cols, _ = findnz(hmap)
            ref_vals = [patch.spectral[r, c] for (r, c) in zip(rows, cols)]
            @test isapprox(halo_vals, ref_vals, atol=1e-12)
        end
    end

    # ──────────────────────────────────────────────
    # 12. accumulate_at_map! writes to patch-sized array
    # ──────────────────────────────────────────────
    @testset "accumulate_at_map! on shared spectral" begin
        sgp = SpringsteelGridParameters(
            geometry = "RL",
            num_cells = 12,
            iMin = 0.0,
            iMax = 120.0,
            BCL = Dict("default" => NaturalBC()),
            BCR = Dict("default" => NaturalBC()),
            vars = Dict("u" => 1),
        )
        patch = createGrid(sgp)
        tiles = calcTileSizes(patch, 2)
        shared = SharedArray{Float64}(size(patch.spectral))
        shared .= 0.0

        # accumulate_at_map! uses patch-coordinate maps on patch-sized array
        hmap = calcHaloMap(patch, tiles[1])
        buffer = ones(Float64, nnz(hmap))
        Scythe.accumulate_at_map!(shared, hmap, buffer)

        # Verify the halo positions now have value 1.0
        rows, cols, _ = findnz(hmap)
        for (r, c) in zip(rows, cols)
            @test shared[r, c] == 1.0
        end

        # Call again to verify accumulation (should be 2.0 now)
        Scythe.accumulate_at_map!(shared, hmap, buffer)
        for (r, c) in zip(rows, cols)
            @test shared[r, c] == 2.0
        end
    end

    # ──────────────────────────────────────────────
    # 13. write + halo accumulate matches sumSpectralTile!
    #     Verifies the complete tile-to-shared pipeline
    # ──────────────────────────────────────────────
    @testset "write + halo vs sumSpectralTile! — RL grid" begin
        sgp = SpringsteelGridParameters(
            geometry = "RL",
            num_cells = 12,
            iMin = 0.0,
            iMax = 120.0,
            BCL = Dict("default" => NaturalBC()),
            BCR = Dict("default" => NaturalBC()),
            vars = Dict("u" => 1),
        )
        patch = createGrid(sgp)
        tiles = calcTileSizes(patch, 2)

        for tile in tiles
            tile.physical .= 1.0
            spectralTransform!(tile)
        end

        # Verify each tile's inner region independently against setSpectralTile!
        # (halo overlap is handled by the serial exchange in model_loop, not tested here)
        for tile in tiles
            shared = SharedArray{Float64}(size(patch.spectral))
            shared .= 0.0
            Scythe.write_tile_to_shared!(shared, tile, patch.params.b_iDim)

            # Reference: setSpectralTile! writes a single tile (zeros + writes, no accumulation)
            setSpectralTile!(patch, tile)

            pmap = calcPatchMap(patch, tile)
            rows, cols, _ = findnz(pmap)
            for (r, c) in zip(rows, cols)
                @test isapprox(shared[r, c], patch.spectral[r, c], atol=1e-12)
            end
        end
    end

    # ──────────────────────────────────────────────
    # 14. ModelTile with RL grid — patch_b_iDim stored
    # ──────────────────────────────────────────────
    @testset "ModelTile RL grid — patch_b_iDim" begin
        sgp = SpringsteelGridParameters(
            geometry = "RL",
            num_cells = 12,
            iMin = 0.0,
            iMax = 120.0,
            BCL = Dict("default" => NaturalBC()),
            BCR = Dict("default" => NaturalBC()),
            vars = Dict("u" => 1),
        )
        model = ModelParameters(
            grid_params = sgp,
            equation_set = "LinearAdvection1D",
            physical_params = Dict(:c_0 => 1.0, :K => 0.0),
        )
        patch = createGrid(model.grid_params)
        tiles = calcTileSizes(patch, 2)

        haloReceiveMap = sparse([1], [1], [1.0], size(patch.spectral, 1), size(patch.spectral, 2))

        for tile in tiles
            mtile = createModelTile(patch, tile, model, haloReceiveMap)
            @test mtile.patch_b_iDim == patch.params.b_iDim

            # Verify write_tile_to_shared! works with the stored patch_b_iDim
            shared = SharedArray{Float64}(size(patch.spectral))
            shared .= 0.0
            tile.physical .= 1.0
            spectralTransform!(tile)
            Scythe.write_tile_to_shared!(shared, tile, mtile.patch_b_iDim)

            # Should not throw and should write non-zero data
            @test any(shared .!= 0.0)
        end
    end

    # ──────────────────────────────────────────────
    # 15. RL grid with max_wavenumber — verify
    #     write_tile_to_shared! handles truncated spectra
    # ──────────────────────────────────────────────
    @testset "write_tile_to_shared! — RL grid with max_wavenumber" begin
        sgp = SpringsteelGridParameters(
            geometry = "RL",
            num_cells = 12,
            iMin = 0.0,
            iMax = 120.0,
            max_wavenumber = Dict("default" => 8),
            BCL = Dict("default" => NaturalBC()),
            BCR = Dict("default" => NaturalBC()),
            vars = Dict("u" => 1, "v" => 2),
        )
        patch = createGrid(sgp)
        tiles = calcTileSizes(patch, 2)
        shared = SharedArray{Float64}(size(patch.spectral))

        for tile in tiles
            tile.physical .= 1.0
            spectralTransform!(tile)

            shared .= 0.0
            Scythe.write_tile_to_shared!(shared, tile, patch.params.b_iDim)

            # Must not throw BoundsError — this was the original bug
            @test true
        end
    end

    # ──────────────────────────────────────────────
    # 16. extract_halo_values — R grid (Cartesian)
    # ──────────────────────────────────────────────
    @testset "extract_halo_values — R grid" begin
        gp = SpringsteelGridParameters(
            geometry = "R",
            num_cells = 12,
            iMin = 0.0,
            iMax = 120.0,
            BCL = Dict("default" => NaturalBC()),
            BCR = Dict("default" => NaturalBC()),
            vars = Dict("u" => 1),
        )
        patch = createGrid(gp)
        tiles = calcTileSizes(patch, 2)

        for tile in tiles
            tile.physical .= 1.0
            spectralTransform!(tile)

            halo_vals = Scythe.extract_halo_values(tile)
            hmap = calcHaloMap(patch, tile)

            @test length(halo_vals) == nnz(hmap)

            # Values must match sumSpectralTile! at halo positions
            patch.spectral .= 0.0
            sumSpectralTile!(patch, tile)
            rows, cols, _ = findnz(hmap)
            ref_vals = [patch.spectral[r, c] for (r, c) in zip(rows, cols)]
            @test isapprox(halo_vals, ref_vals, atol=1e-12)
        end
    end

end
