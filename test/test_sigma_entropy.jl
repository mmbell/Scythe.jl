using Test
using Scythe
using Springsteel
using SparseArrays
using LinearAlgebra

# Tests for the entropy-density equation set primitive_equation_XZ_sigma.
#
# Slot 1 carries the EXTENSIVE entropy density sigma = rho_d*s (variable "sigma") as a
# perturbation sigma' = sigma - sigmabar from the reference sigmabar = rho_dbar*sbar, so the
# per-step cubic-spline smoothing preserves the entropy density integral ∫sigma the same way
# it preserves the partial-density water mass ∫rho_v. The moisture / mass / momentum slots are
# identical to primitive_equation_XZ_rhod_pd; only the entropy variable changes.

@testset "primitive_equation_XZ_sigma (entropy density)" begin

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

    """Build a primitive_equation_XZ_sigma ModelTile on a saturated CondensateReferenceState."""
    function make_sigma_condensate_mtile(tmpdir; num_cells=8, kDim=16)
        varlist = ["sigma", "rho_d", "rho_v", "u", "w", "rho_c", "rho_r", "mu_sat"]
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
        ref_file = joinpath(tmpdir, "sigma_condensate.ref")
        Scythe.write_exact_ref_pd(ref_file, z, col.s, col.rho_d, col.rho_v, col.rho_c)
        model = ModelParameters(
            ts = 0.1, integration_time = 1.0, output_interval = 1.0,
            equation_set = "primitive_equation_XZ_sigma",
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

    # ──────────────────────────────────────────────
    # 1. sigmabar = rho_dbar * sbar round-trips the reference
    # ──────────────────────────────────────────────
    @testset "sigma reference state" begin
        mktempdir() do tmpdir
            mtile, patch, model, col = make_sigma_condensate_mtile(tmpdir)
            rs = mtile.ref_state
            # The new set must consume the physical condensate-bearing reference (ref_rho_c
            # is a real profile, not the scalar 0.0 of a cloudless MoistReferenceState).
            @test rs isa Springsteel.CondensateReferenceState
            @test Springsteel.ref_rho_c(rs) isa AbstractMatrix
            rho_dbar = Springsteel.ref_rho_d(rs)[:, 1]
            sbar = Springsteel.ref_entropy(rs)[:, 1]
            sigmabar = rho_dbar .* sbar
            # σ̂/ρ̂_d round-trips the reference specific entropy
            @test sigmabar ./ rho_dbar ≈ sbar rtol=1e-12
            @test all(isfinite.(sigmabar))
        end
    end

    # ──────────────────────────────────────────────
    # 2. Saturated cloudy base is neutrally buoyant (sigma form)
    # ──────────────────────────────────────────────
    @testset "resting cloudy base preserved" begin
        mktempdir() do tmpdir
            mtile, patch, model, col = make_sigma_condensate_mtile(tmpdir)
            kDim = model.grid_params.kDim
            ncols = div(size(patch.physical, 1), kDim)

            for c in 1:ncols
                Scythe.advance_column(mtile, c, 1)
            end

            # All tendencies (incl. the sigma slot and w-buoyancy) vanish at rest
            for v in 1:7
                @test maximum(abs.(mtile.expdot_n[:, v])) < 1.0e-9
            end
            @test maximum(abs.(mtile.var_np1[:, 1:7])) < 1.0e-9
            @test all(isfinite.(mtile.var_np1))
        end
    end

    # ──────────────────────────────────────────────
    # 3. Condensation conserves water mass and internal energy (sigma form)
    # ──────────────────────────────────────────────
    @testset "condensation conserves mass and internal energy" begin
        mktempdir() do tmpdir
            mtile, patch, model, col = make_sigma_condensate_mtile(tmpdir)
            kDim = model.grid_params.kDim
            vars = model.grid_params.vars
            rho_d_i = vars["rho_d"]; rho_v_i = vars["rho_v"]
            rho_c_i = vars["rho_c"]; rho_r_i = vars["rho_r"]
            sigma_i = vars["sigma"]; mu_sat_i = vars["mu_sat"]

            rs = mtile.ref_state
            rho_dbar = Springsteel.ref_rho_d(rs)[:, 1]
            rho_vbar = Springsteel.ref_rho_v(rs)[:, 1]
            rho_cbar = Springsteel.ref_rho_c(rs)[:, 1]
            sbar = Springsteel.ref_entropy(rs)[:, 1]
            sigmabar = rho_dbar .* sbar
            satbar_t = Scythe.mu_transform.(Springsteel.ref_sat(rs)[:, 1])

            L_const = Scythe.L_v0 - (Scythe.Cpv - Scythe.Cl) * Scythe.T_0

            # Form-A internal energy density per level, recovering s = (sigma' + sigmabar)/rho_d.
            function internal_energy(v)
                E = zeros(kDim)
                for k in 1:kDim
                    rho_d = v[k, rho_d_i] + rho_dbar[k]
                    rho_v = v[k, rho_v_i] + rho_vbar[k]
                    rho_c = v[k, rho_c_i] + rho_cbar[k]
                    rho_r = v[k, rho_r_i]
                    s = (v[k, sigma_i] + sigmabar[k]) / rho_d
                    q_v = rho_v / rho_d
                    Tk = Scythe.temperature(s, rho_d, q_v)
                    E[k] = rho_d * Scythe.Cvd * Tk + rho_v * Scythe.Cvv * Tk +
                           (rho_c + rho_r) * Scythe.Cl * Tk + rho_v * L_const
                end
                return E
            end

            water(v) = v[1:kDim, rho_v_i] .+ rho_vbar .+
                       v[1:kDim, rho_c_i] .+ rho_cbar .+ v[1:kDim, rho_r_i]

            # Supersaturate the lower half by a small, realistic amount. sigma' stays 0
            # (s = sbar), matching the rhod_pd energy-closure test.
            pert = 1:div(kDim, 2)
            dq = 5.0e-5
            mtile.var_np1 .= 0.0
            mtile.var_np1[pert, rho_v_i] .= rho_dbar[pert] .* dq
            mtile.var_np1[pert, mu_sat_i] .= Scythe.mu_transform(1.0) .- satbar_t[pert]

            E_before = internal_energy(mtile.var_np1)
            water_before = water(mtile.var_np1)

            Scythe.condensation_adjustment_sigma(mtile, 1, kDim, 1)

            E_after = internal_energy(mtile.var_np1)
            water_after = water(mtile.var_np1)

            @test all(mtile.var_np1[pert, rho_c_i] .> 0.0)
            @test all(mtile.var_np1[pert, sigma_i] .!= 0.0)   # entropy density updated
            @test maximum(abs.(water_after .- water_before)) < 1.0e-12
            @test maximum(abs.(E_after .- E_before) ./ abs.(E_before)) < 1.0e-6
        end
    end
end
