using Test
using Scythe
using Springsteel

@testset "SpectralGrid" begin

    # ──────────────────────────────────────────────
    # 1. Default GridParameters has correct defaults
    # ──────────────────────────────────────────────
    @testset "Default GridParameters" begin
        gp = GridParameters()
        @test gp.geometry == "R"
        @test gp.xmin == 0.0
        @test gp.xmax == 0.0
    end

    # ──────────────────────────────────────────────
    # 2. Derived fields computed correctly
    # ──────────────────────────────────────────────
    @testset "Derived fields" begin
        gp = GridParameters(num_cells=4)
        # rDim = num_cells * CubicBSpline.mubar
        @test gp.rDim == 4 * Springsteel.CubicBSpline.mubar
        # b_rDim = num_cells + 3
        @test gp.b_rDim == 4 + 3
        # spectralIndexR = spectralIndexL + b_rDim - 1
        @test gp.spectralIndexR == gp.spectralIndexL + gp.b_rDim - 1
        # patchOffsetL = (spectralIndexL - 1) * 3
        @test gp.patchOffsetL == (gp.spectralIndexL - 1) * 3
        # patchOffsetR = patchOffsetL + rDim
        @test gp.patchOffsetR == gp.patchOffsetL + gp.rDim
    end

    # ──────────────────────────────────────────────
    # 3. createGrid with "R" geometry
    # ──────────────────────────────────────────────
    @testset "createGrid R geometry" begin
        bc_dict = Dict("u" => Springsteel.CubicBSpline.R0)
        gp = GridParameters(
            geometry = "R",
            num_cells = 4,
            xmin = 0.0,
            xmax = 100.0,
            BCL = bc_dict,
            BCR = bc_dict,
            vars = Dict("u" => 1),
        )
        grid = createGrid(gp)
        @test grid !== nothing
    end

    # ──────────────────────────────────────────────
    # 4. createGrid "Z" throws error (not implemented)
    # ──────────────────────────────────────────────
    @testset "createGrid Z throws error" begin
        gp = GridParameters(geometry = "Z")
        @test_throws Exception createGrid(gp)
    end

    # ──────────────────────────────────────────────
    # 5. Unknown geometry throws error
    # ──────────────────────────────────────────────
    @testset "Unknown geometry throws error" begin
        gp = GridParameters(geometry = "XYZ")
        @test_throws Exception createGrid(gp)
    end

    # ──────────────────────────────────────────────
    # 6. Type aliases are correct
    # ──────────────────────────────────────────────
    @testset "Type aliases" begin
        @test Springsteel.real === Float64
        @test Springsteel.int === Int64
        @test Springsteel.uint === UInt64
    end

    # ──────────────────────────────────────────────
    # 7. b_zDim computed correctly when zDim > 0
    # ──────────────────────────────────────────────
    @testset "b_zDim with positive zDim" begin
        gp = GridParameters(zDim = 30)
        expected = min(30, floor(((2 * 30) - 1) / 3) + 1)
        @test gp.b_zDim == Int64(expected)

        gp2 = GridParameters(zDim = 10)
        expected2 = min(10, floor(((2 * 10) - 1) / 3) + 1)
        @test gp2.b_zDim == Int64(expected2)
    end

    # ──────────────────────────────────────────────
    # 8. vars dict defaults to Dict("u" => 1)
    # ──────────────────────────────────────────────
    @testset "Default vars dict" begin
        gp = GridParameters()
        @test gp.vars == Dict("u" => 1)
        @test haskey(gp.vars, "u")
        @test gp.vars["u"] == 1
    end

end
