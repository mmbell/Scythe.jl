using Test
using Scythe
using Springsteel
using SparseArrays
using LinearAlgebra

# Tests for the partial-density moisture equation set primitive_equation_XZ_rhod_pd.
#
# The prognostic moisture variables are the *partial densities* rho_v = rho_d*q_v,
# rho_c = rho_d*q_c, rho_r = rho_d*q_r (extensive, linear in mass) so the per-step
# cubic-spline smoothing conserves the physical water mass ∫rho_x rather than the
# intensive ∫q_x. See reference/phase3_moisture_partial_density.md.
#
# This set consumes a *physical* Springsteel reference state directly (MoistReferenceState
# or CondensateReferenceState), so the reference supplies rho_vbar / rho_cbar profiles
# with spectrally consistent derivatives.

@testset "Partial-density moisture (rhod_pd)" begin

    """Write a constant-theta, constant-q_v sounding for reference construction."""
    function write_test_sounding(path; theta=300.0, q_v_gkg=5.0, zmax=4000.0, dz=100.0)
        open(path, "w") do f
            println(f, "1000.0\t$(theta)\t$(q_v_gkg)")
            for z in dz:dz:zmax
                println(f, "$(z)\t$(theta)\t$(q_v_gkg)")
            end
        end
        return path
    end

    """Build an XZ partial-density model, patch, and single-process ModelTile."""
    function make_pd_mtile(tmpdir; num_cells=8, kDim=16, q_v_gkg=5.0)
        sounding = write_test_sounding(joinpath(tmpdir, "pd_test.ref"); q_v_gkg=q_v_gkg)
        varlist = ["s", "rho_d", "rho_v", "u", "w", "rho_c", "rho_r", "mu_sat"]
        vars = Dict(v => i for (i, v) in enumerate(varlist))
        scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
        wall_bc = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))
        gp = GridParameters(
            geometry = "RZ",
            num_cells = num_cells,
            iMin = 0.0,
            iMax = 2000.0,
            kMin = 0.0,
            kMax = 2000.0,
            kDim = kDim,
            BCL = wall_bc,
            BCR = wall_bc,
            BCB = wall_bc,
            BCT = wall_bc,
            vars = vars,
        )
        model = ModelParameters(
            ts = 0.1,
            integration_time = 1.0,
            output_interval = 1.0,
            equation_set = "primitive_equation_XZ_rhod_pd",
            ref_state_file = sounding,
            grid_params = gp,
            physical_params = Dict(:Khdiff => 0.0, :Kvdiff => 0.0, :Kv_mudiff => 0.0,
                                   :alpha => 0.0, :z_damp => 20.0e3),
            options = Dict(:semiimplicit => false, :exact_reference_state => false,
                           :precipitation => false, :vertical_mixing => false),
        )
        patch = createGrid(gp)
        patch.physical .= 0.0
        spectralTransform!(patch)
        gridTransform!(patch)
        haloReceiveMap = sparse(Int64[], Int64[], Float64[],
                                size(patch.spectral, 1), size(patch.spectral, 2))
        mtile = createModelTile(patch, patch, model, haloReceiveMap)
        return mtile, patch, model
    end

    # ──────────────────────────────────────────────
    # 1. The model tile carries a physical reference state
    # ──────────────────────────────────────────────
    @testset "physical reference state" begin
        mktempdir() do tmpdir
            mtile, patch, model = make_pd_mtile(tmpdir)
            @test mtile.ref_state isa Springsteel.AbstractReferenceState
            # Vapor partial-density profile is a real (nlevels, 3) array (value + derivs)
            rv = Springsteel.ref_rho_v(mtile.ref_state)
            @test rv isa AbstractMatrix
            @test size(rv, 2) == 3
            @test all(rv[:, 1] .> 0.0)
        end
    end

    # ──────────────────────────────────────────────
    # 2. Resting state on a hydrostatic reference: tendencies vanish
    # ──────────────────────────────────────────────
    @testset "resting state preserved" begin
        mktempdir() do tmpdir
            mtile, patch, model = make_pd_mtile(tmpdir)
            kDim = model.grid_params.kDim
            ncols = div(size(patch.physical, 1), kDim)

            for c in 1:ncols
                Scythe.advance_column(mtile, c, 1)
            end

            # Pure perturbation form ⇒ all dynamical tendencies are identically zero
            # at rest on any reference (s, rho_d, rho_v, u, w, rho_c, rho_r)
            for v in 1:7
                @test maximum(abs.(mtile.expdot_n[:, v])) < 1.0e-9
            end
            # The advected saturation-ratio tendency is zero to smoothing accuracy
            @test maximum(abs.(mtile.expdot_n[:, 8])) < 1.0e-7

            # State after one step is still at rest
            @test maximum(abs.(mtile.var_np1[:, 1:7])) < 1.0e-9
            @test all(isfinite.(mtile.var_np1))
        end
    end

    # ──────────────────────────────────────────────
    # 3. condensation_adjustment_pd conserves rho_v + rho_c exactly
    # ──────────────────────────────────────────────
    @testset "condensation conserves water mass" begin
        mktempdir() do tmpdir
            mtile, patch, model = make_pd_mtile(tmpdir)
            kDim = model.grid_params.kDim
            vars = model.grid_params.vars
            rho_v_i = vars["rho_v"]; rho_c_i = vars["rho_c"]
            s_i = vars["s"]; mu_sat_i = vars["mu_sat"]

            # Subsaturated resting state: adjustment changes nothing
            mtile.var_np1 .= 0.0
            before = copy(mtile.var_np1)
            Scythe.condensation_adjustment_pd(mtile, 1, kDim, 1)
            @test maximum(abs.(mtile.var_np1 .- before)) < 1.0e-12

            # Force supersaturation in the lower half of the first column: add vapor
            # and set the advected saturation ratio above 1 there.
            mtile.var_np1 .= 0.0
            rho_dbar = Springsteel.ref_rho_d(mtile.ref_state)[:, 1]
            satbar_t = Scythe.mu_transform.(Springsteel.ref_sat(mtile.ref_state)[:, 1])
            pert = 1:div(kDim, 2)
            dq = 20.0e-3                                 # +20 g/kg vapor (q_sat ≈ 22 g/kg)
            mtile.var_np1[pert, rho_v_i] .= rho_dbar[pert] .* dq
            # Advect a just-saturated ratio (qss = 0) so the actual supersaturation
            # (q_v - q_sat > 0) drives condensation in the perturbed levels.
            mtile.var_np1[pert, mu_sat_i] .= Scythe.mu_transform(1.0) .- satbar_t[pert]

            rho_v_before = copy(mtile.var_np1[1:kDim, rho_v_i])
            rho_c_before = copy(mtile.var_np1[1:kDim, rho_c_i])
            s_before = copy(mtile.var_np1[1:kDim, s_i])
            total_before = rho_v_before .+ rho_c_before

            Scythe.condensation_adjustment_pd(mtile, 1, kDim, 1)

            rho_v_after = mtile.var_np1[1:kDim, rho_v_i]
            rho_c_after = mtile.var_np1[1:kDim, rho_c_i]
            s_after = mtile.var_np1[1:kDim, s_i]
            total_after = rho_v_after .+ rho_c_after

            # Water mass (vapor + cloud partial density) is exactly conserved
            @test maximum(abs.(total_after .- total_before)) < 1.0e-12

            # Where supersaturated: vapor condensed to cloud, entropy adjusted
            @test all(rho_v_after[pert] .< rho_v_before[pert])
            @test all(rho_c_after[pert] .> rho_c_before[pert])
            @test all(s_after[pert] .!= s_before[pert])

            # Subsaturated upper half untouched
            unsat = (div(kDim, 2)+1):kDim
            @test all(rho_v_after[unsat] .== rho_v_before[unsat])
            @test all(rho_c_after[unsat] .== rho_c_before[unsat])

            @test all(isfinite.(mtile.var_np1))
        end
    end

    # ──────────────────────────────────────────────
    # 4. Condensate reference (saturated cloudy neutral base)
    # ──────────────────────────────────────────────

    """Fabricate a saturated, cloudy column (q_v = q_sat, q_l = 1 g/kg) on levels `z`."""
    function saturated_cloudy_column(z; q_l=1.0e-3)
        n = length(z)
        s = zeros(n); rho_d = zeros(n); rho_v = zeros(n); rho_c = zeros(n)
        Tk = zeros(n); p = zeros(n); q_v = zeros(n)
        for k in 1:n
            Tk[k] = 290.0 - 0.005 * z[k]
            p[k] = 1000.0 * exp(-z[k] / 8000.0)
            q_v[k] = Scythe.q_sat_liquid(Tk[k], p[k])
            e = Scythe.vapor_pressure(p[k], q_v[k])
            rho_d[k] = 100.0 * (p[k] - e) / (Scythe.Rd * Tk[k])
            rho_v[k] = rho_d[k] * q_v[k]
            rho_c[k] = rho_d[k] * q_l
            s[k] = Scythe.entropy(Tk[k], rho_d[k], q_v[k])
        end
        theta = Tk .* (1000.0 ./ p).^(Scythe.Rd / Scythe.Cpd)
        q_t = q_v .+ q_l
        theta_rho = theta .* (1.0 .+ (q_v ./ Scythe.Eps)) ./ (1.0 .+ q_t)
        base = (; s, rho_d, q_v, q_l = fill(q_l, n), p, theta_rho)
        return (; z, s, rho_d, rho_v, rho_c, base)
    end

    """Build a pd ModelTile whose reference is a saturated CondensateReferenceState."""
    function make_pd_condensate_mtile(tmpdir; num_cells=8, kDim=16)
        varlist = ["s", "rho_d", "rho_v", "u", "w", "rho_c", "rho_r", "mu_sat"]
        vars = Dict(v => i for (i, v) in enumerate(varlist))
        scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
        wall_bc = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))
        gp = GridParameters(
            geometry = "RZ", num_cells = num_cells,
            iMin = 0.0, iMax = 2000.0, kMin = 0.0, kMax = 2000.0, kDim = kDim,
            BCL = wall_bc, BCR = wall_bc, BCB = wall_bc, BCT = wall_bc, vars = vars,
        )
        patch = createGrid(gp)
        gridpoints = Scythe.getGridpoints(patch)
        z = gridpoints[1:kDim, 2]
        col = saturated_cloudy_column(z)
        ref_file = joinpath(tmpdir, "pd_condensate.ref")
        Scythe.write_exact_ref_pd(ref_file, z, col.s, col.rho_d, col.rho_v, col.rho_c)
        model = ModelParameters(
            ts = 0.1, integration_time = 1.0, output_interval = 1.0,
            equation_set = "primitive_equation_XZ_rhod_pd",
            ref_state_file = ref_file, grid_params = gp,
            physical_params = Dict(:Khdiff => 0.0, :Kvdiff => 0.0, :Kv_mudiff => 0.0,
                                   :alpha => 0.0, :z_damp => 20.0e3),
            options = Dict(:semiimplicit => false, :exact_reference_state => true,
                           :precipitation => false, :vertical_mixing => false),
        )
        patch.physical .= 0.0
        spectralTransform!(patch)
        gridTransform!(patch)
        haloReceiveMap = sparse(Int64[], Int64[], Float64[],
                                size(patch.spectral, 1), size(patch.spectral, 2))
        mtile = createModelTile(patch, patch, model, haloReceiveMap)
        return mtile, patch, model, col
    end

    @testset "write_exact_ref_pd round-trip" begin
        mktempdir() do tmpdir
            mtile, patch, model, col = make_pd_condensate_mtile(tmpdir)
            rs = mtile.ref_state
            @test rs isa Springsteel.CondensateReferenceState
            # Condensate profile is a real (nlevels, 3) array with the written values
            rc = Springsteel.ref_rho_c(rs)
            @test rc isa AbstractMatrix
            @test all(rc[:, 1] .> 0.0)
            @test rc[:, 1] ≈ col.rho_c rtol=1e-6
            @test Springsteel.ref_rho_d(rs)[:, 1] ≈ col.rho_d rtol=1e-6
            @test Springsteel.ref_rho_v(rs)[:, 1] ≈ col.rho_v rtol=1e-6
        end
    end

    @testset "saturated cloudy base is neutrally buoyant" begin
        mktempdir() do tmpdir
            mtile, patch, model, col = make_pd_condensate_mtile(tmpdir)
            kDim = model.grid_params.kDim
            ncols = div(size(patch.physical, 1), kDim)

            for c in 1:ncols
                Scythe.advance_column(mtile, c, 1)
            end

            # The condensate-inclusive rhobar makes the cloudy base neutral: all
            # tendencies (including w-buoyancy) vanish at rest. (If rhobar omitted
            # rho_cbar, the w-equation would feel a spurious -g·rho_cbar buoyancy.)
            for v in 1:7
                @test maximum(abs.(mtile.expdot_n[:, v])) < 1.0e-9
            end
            @test maximum(abs.(mtile.var_np1[:, 1:7])) < 1.0e-9
            @test all(isfinite.(mtile.var_np1))
        end
    end

    @testset "moist_buoyancy_bubble_pd! perturbation field" begin
        mktempdir() do tmpdir
            mtile, patch, model, col = make_pd_condensate_mtile(tmpdir)
            gridpoints = Scythe.getGridpoints(patch)
            vars = model.grid_params.vars

            patch.physical .= 0.0
            # Bubble centered in the (small test) domain
            Scythe.moist_buoyancy_bubble_pd!(patch, gridpoints, col.base, mtile.ref_state;
                                             q_t=0.021, xc=1000.0, xr=500.0,
                                             zc=1000.0, zr=500.0, amp=2.0/300.0)

            # Rain partial density is never seeded
            @test all(patch.physical[:, vars["rho_r"], 1] .== 0.0)

            # Identify in/out of the bubble
            inb = falses(size(gridpoints, 1))
            for i in 1:size(gridpoints, 1)
                L = sqrt(((gridpoints[i,1]-1000.0)/500.0)^2 + ((gridpoints[i,2]-1000.0)/500.0)^2)
                inb[i] = L <= 1.0
            end
            @test count(inb) > 0

            s_p = patch.physical[:, vars["s"], 1]
            rho_v_p = patch.physical[:, vars["rho_v"], 1]
            # Outside the bubble the perturbations are at the spectral-smoothing level
            @test maximum(abs.(s_p[.!inb])) < 1.0e-6
            @test maximum(abs.(rho_v_p[.!inb])) < 1.0e-6
            # Inside the bubble the warm, more-saturated air departs from the base
            @test maximum(abs.(s_p[inb])) > 1.0e-3
            @test all(isfinite.(patch.physical))
        end
    end
end
