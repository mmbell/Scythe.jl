using Test
using Scythe

@testset "ReferenceState" begin

    # ──────────────────────────────────────────────
    # 1. empty_reference_state returns a valid ReferenceState
    # ──────────────────────────────────────────────
    @testset "empty_reference_state returns ReferenceState" begin
        rs = Scythe.empty_reference_state()
        @test isa(rs, Scythe.ReferenceState)
    end

    # ──────────────────────────────────────────────
    # 2. empty_reference_state has Pxi_bar == 0.0
    # ──────────────────────────────────────────────
    @testset "empty_reference_state Pxi_bar is zero" begin
        rs = Scythe.empty_reference_state()
        @test rs.Pxi_bar == 0.0
    end

    # ──────────────────────────────────────────────
    # 3. Fields sbar, xibar, mubar, satbar are arrays
    # ──────────────────────────────────────────────
    @testset "Array fields are arrays" begin
        rs = Scythe.empty_reference_state()
        @test isa(rs.sbar, Array{Float64})
        @test isa(rs.xibar, Array{Float64})
        @test isa(rs.mubar, Array{Float64})
        @test isa(rs.satbar, Array{Float64})
    end

    # ──────────────────────────────────────────────
    # 4. ReferenceState can be constructed with explicit arrays
    # ──────────────────────────────────────────────
    @testset "Explicit construction" begin
        s = zeros(Float64, 10, 3)
        xi = ones(Float64, 10, 3)
        mu = fill(0.5, 10, 3)
        sat = fill(0.8, 10, 3)
        pxi = 42.0

        rs = Scythe.ReferenceState(s, xi, mu, sat, pxi)
        @test rs.Pxi_bar == 42.0
        @test rs.sbar === s
        @test rs.xibar === xi
        @test rs.mubar === mu
        @test rs.satbar === sat
        @test size(rs.sbar) == (10, 3)
    end

    # ──────────────────────────────────────────────
    # 5. ReferenceState struct has 5 fields
    # ──────────────────────────────────────────────
    @testset "ReferenceState has 5 fields" begin
        @test fieldcount(Scythe.ReferenceState) == 5
        rs = Scythe.empty_reference_state()
        @test hasproperty(rs, :satbar)
        @test hasproperty(rs, :Pxi_bar)
    end

end
