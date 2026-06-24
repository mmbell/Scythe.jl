using Test
using Scythe
using Springsteel
using SparseArrays
using SharedArrays
using LinearAlgebra

include("distributed_test_helpers.jl")

# Tests for the parameterized diffusivity floor (:K_min_free, :K_min_bl) in Twoway_PV_mixing.
# With the Smagorinsky length scale set to 0, K = max(Ls²|S|, K_min) = K_min, so the floor IS
# the diffusion coefficient — making it directly testable against the analytic vector-Laplacian.
@testset "Twoway_PV_mixing diffusivity floor" begin

    """Axisymmetric (wavenumber-0) Twoway_PV_mixing model with configurable floor/length scales."""
    function make_pv_mixing_model(; num_cells=20, iMax=3.0e5, ts=1.0,
                                    Ls_free=0.0, Ls_bl=0.0,
                                    K_min_free=0.0, K_min_bl=0.0,
                                    include_floor_keys=true)
        gp = SpringsteelGridParameters(
            geometry = "RL", iMin = 0.0, iMax = iMax, num_cells = num_cells,
            max_wavenumber = Dict("h"=>0,"u"=>0,"v"=>0,"ub"=>0,"vb"=>0,"wb"=>0),
            BCL = Dict("h"=>NeumannBC(),"u"=>DirichletBC(),"v"=>DirichletBC(),
                       "ub"=>DirichletBC(),"vb"=>DirichletBC(),"wb"=>NeumannBC()),
            BCR = Dict("h"=>NaturalBC(),"u"=>NeumannBC(),"v"=>NaturalBC(),
                       "ub"=>NeumannBC(),"vb"=>NaturalBC(),"wb"=>NaturalBC()),
            vars = Dict("h"=>1,"u"=>2,"v"=>3,"ub"=>4,"vb"=>5,"wb"=>6),
        )
        pp = Dict(:g=>9.81, :Ls_free=>Ls_free, :Ls_bl=>Ls_bl, :Cd=>2.4e-3,
                  :Hfree=>2000.0, :Hb=>1000.0, :f=>5.0e-5, :S1=>1e-5)
        if include_floor_keys
            pp[:K_min_free] = K_min_free
            pp[:K_min_bl]   = K_min_bl
        end
        model = ModelParameters(
            ts = ts, integration_time = ts, output_interval = ts,
            equation_set = "Twoway_PV_mixing", initial_conditions = "",
            grid_params = gp, physical_params = pp)
        return model
    end

    function make_single_process_mtile(model, patch)
        haloReceiveMap = sparse(Int64[], Int64[], Float64[],
                                size(patch.spectral, 1), size(patch.spectral, 2))
        return createModelTile(patch, patch, model, haloReceiveMap)
    end

    """Axisymmetric Gaussian ring in vg only (ug=h=ub=vb=0) so the vg tendency is pure diffusion."""
    function set_vg_ring!(patch; Rmax=50000.0, sigma=30000.0, amp=20.0)
        gp = getGridpoints(patch)
        for i in 1:size(patch.physical, 1)
            r_m = gp[i, 1]
            patch.physical[i, :, 1] .= 0.0
            patch.physical[i, 3, 1] = amp * exp(-(r_m - Rmax)^2 / (2 * sigma^2))
        end
        spectralTransform!(patch)
        gridTransform!(patch)
    end

    function vg_tendency(model)
        patch = createGrid(model.grid_params)
        set_vg_ring!(patch)
        mtile = make_single_process_mtile(model, patch)
        Scythe.advance_column(mtile, -1, 1)
        return copy(mtile.expdot_n[:, 3]), patch
    end

    # ── 1. K_min_free = 0 with Ls_free = 0 ⇒ free atmosphere is inviscid; vg tendency ≈ 0
    @testset "Floor off ⇒ inviscid free layer" begin
        T0, _ = vg_tendency(make_pv_mixing_model(K_min_free=0.0, K_min_bl=0.0))
        @test maximum(abs.(T0)) < 1e-9
        println("  max |vg tendency| (K_min_free=0) = $(round(maximum(abs.(T0)), sigdigits=3))")
    end

    # ── 2. With Ls_free=0, the floor IS the coefficient: vg tendency = K_min_free · ∇²vg
    @testset "Floor is the diffusion coefficient" begin
        Kf = 5000.0
        model = make_pv_mixing_model(Ls_free=0.0, K_min_free=Kf, K_min_bl=0.0)
        T, patch = vg_tendency(model)
        r   = getGridpoints(patch)[:, 1]
        vg  = patch.physical[:, 3, 1]
        vgr = patch.physical[:, 3, 2]
        vgrr= patch.physical[:, 3, 3]
        # axisymmetric ⇒ λ-derivative terms vanish; vector-Laplacian radial component:
        expected = @. Kf * (vgr / r + vgrr - vg / (r * r))
        interior = r .> 1.0e4          # avoid the 1/r-sensitive innermost ring
        @test isapprox(T[interior], expected[interior]; rtol=1e-5, atol=1e-10)
        println("  max rel err vs Kf·∇²vg = ",
                round(maximum(abs.((T .- expected)[interior]) ./ maximum(abs.(expected[interior]))), sigdigits=3))
    end

    # ── 3. Tendency scales linearly with the floor parameter (proves it's used, not hardcoded)
    @testset "Tendency scales with the floor parameter" begin
        T1, _ = vg_tendency(make_pv_mixing_model(Ls_free=0.0, K_min_free=2000.0))
        T2, _ = vg_tendency(make_pv_mixing_model(Ls_free=0.0, K_min_free=4000.0))
        interior = getGridpoints(createGrid(make_pv_mixing_model().grid_params))[:,1] .> 1.0e4
        @test isapprox(T2[interior], 2.0 .* T1[interior]; rtol=1e-6, atol=1e-12)
    end

    # ── 4. Missing :K_min_free raises a clear KeyError (required param, matches convention)
    @testset "Missing floor key errors" begin
        model = make_pv_mixing_model(include_floor_keys=false)
        patch = createGrid(model.grid_params)
        set_vg_ring!(patch)
        mtile = make_single_process_mtile(model, patch)
        @test_throws KeyError Scythe.advance_column(mtile, -1, 1)
    end
end
