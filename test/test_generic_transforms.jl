using Test
using Scythe
using Springsteel

@testset "Generic Vertical Transforms" begin

    # ──────────────────────────────────────────────
    # Helper: create a Chebyshev column for testing
    # ──────────────────────────────────────────────
    function make_test_column(; nz=32)
        cp = ChebyshevParameters(
            zmin = 0.0,
            zmax = 15000.0,
            zDim = nz,
            bDim = min(nz, Int64(floor(((2 * nz) - 1) / 3) + 1)),
            BCB = Chebyshev.R0,
            BCT = Chebyshev.R0)
        return Chebyshev1D(cp)
    end

    # A smooth test function: sin profile over the domain
    function smooth_profile(z, zmin, zmax)
        return sin.(π .* (z .- zmin) ./ (zmax - zmin))
    end

    # ──────────────────────────────────────────────
    # 1. Generic Btransform!/Atransform! match C-prefixed
    # ──────────────────────────────────────────────
    @testset "Btransform!/Atransform! match CBtransform!/CAtransform!" begin
        col1 = make_test_column()
        col2 = deepcopy(col1)

        # Set up the same input data
        z = col1.mishPoints
        data = smooth_profile(z, 0.0, 15000.0)
        col1.uMish .= data
        col2.uMish .= data

        # C-prefixed path
        CBtransform!(col1)
        CAtransform!(col1)

        # Generic path
        Btransform!(col2)
        Atransform!(col2)

        @test col1.b ≈ col2.b atol=1e-14
        @test col1.a ≈ col2.a atol=1e-14
    end

    # ──────────────────────────────────────────────
    # 2. Generic Itransform! matches CItransform!
    # ──────────────────────────────────────────────
    @testset "Itransform! matches CItransform!" begin
        col1 = make_test_column()
        col2 = deepcopy(col1)

        data = smooth_profile(col1.mishPoints, 0.0, 15000.0)
        col1.uMish .= data
        col2.uMish .= data

        CBtransform!(col1); CAtransform!(col1)
        Btransform!(col2); Atransform!(col2)

        result1 = CItransform!(col1)
        result2 = Itransform!(col2)

        @test result1 ≈ result2 atol=1e-14
    end

    # ──────────────────────────────────────────────
    # 3. Generic Ixtransform matches CIxtransform
    # ──────────────────────────────────────────────
    @testset "Ixtransform matches CIxtransform" begin
        col1 = make_test_column()
        col2 = deepcopy(col1)

        data = smooth_profile(col1.mishPoints, 0.0, 15000.0)
        col1.uMish .= data
        col2.uMish .= data

        CBtransform!(col1); CAtransform!(col1)
        Btransform!(col2); Atransform!(col2)

        deriv1 = CIxtransform(col1)
        deriv2 = Ixtransform(col2)

        @test deriv1 ≈ deriv2 atol=1e-14
    end

    # ──────────────────────────────────────────────
    # 4. Generic Ixxtransform matches CIxxtransform
    # ──────────────────────────────────────────────
    @testset "Ixxtransform matches CIxxtransform" begin
        col1 = make_test_column()
        col2 = deepcopy(col1)

        data = smooth_profile(col1.mishPoints, 0.0, 15000.0)
        col1.uMish .= data
        col2.uMish .= data

        CBtransform!(col1); CAtransform!(col1)
        Btransform!(col2); Atransform!(col2)

        deriv2_1 = CIxxtransform(col1)
        deriv2_2 = Ixxtransform(col2)

        @test deriv2_1 ≈ deriv2_2 atol=1e-14
    end

    # ──────────────────────────────────────────────
    # 5. Generic IInttransform matches CIInttransform
    # ──────────────────────────────────────────────
    @testset "IInttransform matches CIInttransform" begin
        col1 = make_test_column()
        col2 = deepcopy(col1)

        data = smooth_profile(col1.mishPoints, 0.0, 15000.0)
        col1.uMish .= data
        col2.uMish .= data

        CBtransform!(col1); CAtransform!(col1)
        Btransform!(col2); Atransform!(col2)

        C0 = 1.0
        int1 = CIInttransform(col1, C0)
        int2 = IInttransform(col2, C0)

        @test int1 ≈ int2 atol=1e-14
    end

    # ──────────────────────────────────────────────
    # 6. operator_matrix matches Chebyshev-specific matrices
    # ──────────────────────────────────────────────
    @testset "operator_matrix matches Chebyshev dct matrices" begin
        nz = 32
        b_zDim = min(nz, Int64(floor(((2 * nz) - 1) / 3) + 1))

        # Create an RZ grid with Chebyshev vertical
        gp = GridParameters(
            geometry = "RZ",
            num_cells = 4,
            xmin = 0.0,
            xmax = 100000.0,
            zmin = 0.0,
            zmax = 15000.0,
            zDim = nz,
            BCL = Dict("u" => NaturalBC()),
            BCR = Dict("u" => NaturalBC()),
            BCB = Dict("u" => NaturalBC()),
            BCT = Dict("u" => NaturalBC()),
            vars = Dict("u" => 1),
        )
        grid = createGrid(gp)

        # Compare operator_matrix with direct Chebyshev matrix functions
        M0 = operator_matrix(grid, :k, 0)
        M1 = operator_matrix(grid, :k, 1)
        M2 = operator_matrix(grid, :k, 2)

        column_length = 15000.0
        dct0 = Chebyshev.dct_matrix(nz)
        dct1 = Chebyshev.dct_1st_derivative(nz, column_length)
        dct2 = Chebyshev.dct_2nd_derivative(nz, column_length)

        @test M0 ≈ dct0 atol=1e-12
        @test M1 ≈ dct1 atol=1e-12
        @test M2 ≈ dct2 atol=1e-12
    end

    # ──────────────────────────────────────────────
    # 7. Helmholtz matrix via operator_matrix matches direct build
    # ──────────────────────────────────────────────
    @testset "Helmholtz semiimplicit matrix via operator_matrix" begin
        nz = 32

        gp = GridParameters(
            geometry = "RZ",
            num_cells = 4,
            xmin = 0.0,
            xmax = 100000.0,
            zmin = 0.0,
            zmax = 15000.0,
            zDim = nz,
            BCL = Dict("w" => NaturalBC(), "xi" => NaturalBC()),
            BCR = Dict("w" => NaturalBC(), "xi" => NaturalBC()),
            BCB = Dict("w" => NaturalBC(), "xi" => NaturalBC()),
            BCT = Dict("w" => NaturalBC(), "xi" => NaturalBC()),
            vars = Dict("w" => 1, "xi" => 2),
        )
        grid = createGrid(gp)

        model = Scythe.ModelParameters(
            grid_params = gp,
            options = Dict(:semiimplicit => true, :exact_reference_state => false),
        )

        Pxi_bar = 100000.0
        ts_term = 1.25 * 1.0

        # Build via generic operator_matrix path
        M0 = operator_matrix(grid, :k, 0)
        M2 = operator_matrix(grid, :k, 2)
        h = (ts_term * ts_term * Pxi_bar) .* M2 .- M0
        bc1 = M0[1,:]
        bc2 = M0[nz,:]
        h_generic = [bc1[:]'; bc2[:]'; h[2:nz-1,:]]

        # Build via direct Chebyshev path
        column_length = 15000.0
        dct = Chebyshev.dct_matrix(nz)
        dct2 = Chebyshev.dct_2nd_derivative(nz, column_length)
        h_cheb = (ts_term * ts_term * Pxi_bar) .* dct2 .- dct
        bc1_cheb = dct[1,:]
        bc2_cheb = dct[nz,:]
        h_direct = [bc1_cheb[:]'; bc2_cheb[:]'; h_cheb[2:nz-1,:]]

        @test h_generic ≈ h_direct atol=1e-10
    end

    # ──────────────────────────────────────────────
    # 8. transform_reference_state! produces correct derivatives
    # ──────────────────────────────────────────────
    @testset "transform_reference_state! derivatives" begin
        col = make_test_column(nz=64)
        z = col.mishPoints
        zmin = 0.0
        zmax = 15000.0

        # Use sin(π z/L) which has known derivatives
        ref = zeros(Float64, length(z), 3)
        ref[:,1] .= sin.(π .* z ./ zmax)

        Scythe.transform_reference_state!(col, ref)

        # Analytical first derivative: (π/L) cos(π z/L)
        expected_dz = (π / zmax) .* cos.(π .* z ./ zmax)
        # Analytical second derivative: -(π/L)^2 sin(π z/L)
        expected_dzz = -(π / zmax)^2 .* sin.(π .* z ./ zmax)

        # Spectral derivatives should be accurate for smooth functions
        @test ref[:,2] ≈ expected_dz rtol=1e-4
        @test ref[:,3] ≈ expected_dzz rtol=1e-3
    end

end
