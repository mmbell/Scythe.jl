using Test
using Scythe
using Springsteel
using SparseArrays
using LinearAlgebra

# Tests for the restored BF02 qss-relaxation condensation scheme.
# The scheme was working at commit a4bf2a0 ("Fixes for passing BF02 moist test")
# but broke when ReferenceState dropped mu_lbar and the condensation functions
# were repurposed for the primitive equation variable set (mu_c/mu_r/sat_ratio).
# The restored functions use the current linear mu convention (mu_transform).

@testset "BF02 qss-relaxation restoration" begin

    """Write a constant-theta sounding file for reference state construction."""
    function write_test_sounding(path; theta=300.0, q_v_gkg=5.0, zmax=15000.0, dz=100.0)
        open(path, "w") do f
            println(f, "1000.0\t$(theta)\t$(q_v_gkg)")
            for z in dz:dz:zmax
                println(f, "$(z)\t$(theta)\t$(q_v_gkg)")
            end
        end
        return path
    end

    """Build a small 7-var RZ BF02 model, patch, and single-process ModelTile."""
    function make_bf02_mtile(tmpdir; num_cells=8, kDim=16, semiimplicit=false, q_v_gkg=5.0)
        sounding = write_test_sounding(joinpath(tmpdir, "bf02_test.ref"); q_v_gkg=q_v_gkg)
        vars = Dict("s" => 1, "xi" => 2, "mu" => 3, "u" => 4, "w" => 5,
                    "mu_l" => 6, "qss" => 7)
        scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
        bcl = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))
        gp = GridParameters(
            geometry = "RZ",
            num_cells = num_cells,
            iMin = 0.0,
            iMax = 2000.0,
            kMin = 0.0,
            kMax = 2000.0,
            kDim = kDim,
            BCL = bcl,
            BCR = bcl,
            BCB = bcl,
            BCT = bcl,
            vars = vars,
        )
        model = ModelParameters(
            ts = 0.1,
            integration_time = 1.0,
            output_interval = 1.0,
            equation_set = "BF02_test",
            ref_state_file = sounding,
            grid_params = gp,
            physical_params = Dict(:K => 0.0, :Kvdiff => 0.0),
            options = Dict(:semiimplicit => semiimplicit, :exact_reference_state => false),
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
    # 1. q_condensation_relaxation: a4bf2a0 formula
    # ──────────────────────────────────────────────
    @testset "q_condensation_relaxation formula" begin
        Tk = 290.0; p = 900.0; q_v = 0.012; q_l = 1.0e-4
        N_c = 500.0; r_c = 10.0
        Q_s = Scythe.Q_s_factor(Tk, p, q_v, q_l)
        invtau = Scythe.invtau_condensation(Tk, p, N_c, r_c)

        # Unlimited regime: q_cond = qss/(1+Q_s) * invtau
        qss = 1.0e-4
        expected = (qss / (1.0 + Q_s)) * invtau
        @test Scythe.q_condensation_relaxation(qss, Tk, p, q_v, q_l, N_c, r_c) ≈ expected

        # Condensation limited by available vapor
        @test Scythe.q_condensation_relaxation(1.0, Tk, p, q_v, q_l, N_c, r_c) ≈ q_v * invtau

        # Evaporation limited by available liquid
        @test Scythe.q_condensation_relaxation(-1.0, Tk, p, q_v, q_l, N_c, r_c) ≈ -q_l * invtau

        # Zero supersaturation: no condensation
        @test Scythe.q_condensation_relaxation(0.0, Tk, p, q_v, q_l, N_c, r_c) == 0.0
    end

    # ──────────────────────────────────────────────
    # 2. s_condensation_relaxation: a4bf2a0 entropy formula
    # ──────────────────────────────────────────────
    @testset "s_condensation_relaxation formula" begin
        Tk = 290.0; p = 900.0; rho_d = 1.05; q_v = 0.012; q_l = 1.0e-4
        q_cond = 1.0e-7
        Cm = (q_l * Scythe.Cl) / (Scythe.Cvd + (q_v * Scythe.Cvv) + (q_l * Scythe.Cl))
        # The Rv*log(e/sat_e) (R_v ln H) term was removed from the formula in the
        # sigma-era revision; the expected value matches the code as benchmarked.
        expected = q_cond * (((-Scythe.L_v(Tk) * Cm) / Tk) -
                             (Scythe.Cl * log(Tk / Scythe.T_0)))
        @test Scythe.s_condensation_relaxation(q_cond, Tk, rho_d, q_v, q_l, p) ≈ expected

        # Zero condensation produces zero entropy change
        @test Scythe.s_condensation_relaxation(0.0, Tk, rho_d, q_v, q_l, p) == 0.0
    end

    # ──────────────────────────────────────────────
    # 3. condensation_adjustment_qss on a ModelTile
    # ──────────────────────────────────────────────
    @testset "condensation_adjustment_qss" begin
        mktempdir() do tmpdir
            mtile, patch, model = make_bf02_mtile(tmpdir)
            kDim = model.grid_params.kDim
            vars = model.grid_params.vars
            mu_idx = vars["mu"]; mu_l_idx = vars["mu_l"]; s_idx = vars["s"]

            # Subsaturated resting state: adjustment must not change anything
            mtile.var_np1 .= 0.0
            Scythe.condensation_adjustment_qss(mtile, 1, kDim, 1)
            @test all(mtile.var_np1 .== 0.0)

            # Supersaturate the lower half of the first column by adding vapor
            dq = 5.0e-3   # 5 g/kg perturbation
            mu_pert = Scythe.mu_transform(dq)
            mtile.var_np1 .= 0.0
            pert_levels = 1:div(kDim, 2)
            mtile.var_np1[pert_levels, mu_idx] .= mu_pert

            mu_before = copy(mtile.var_np1[1:kDim, mu_idx])
            mu_l_before = copy(mtile.var_np1[1:kDim, mu_l_idx])
            s_before = copy(mtile.var_np1[1:kDim, s_idx])

            Scythe.condensation_adjustment_qss(mtile, 1, kDim, 1)

            mu_after = mtile.var_np1[1:kDim, mu_idx]
            mu_l_after = mtile.var_np1[1:kDim, mu_l_idx]
            s_after = mtile.var_np1[1:kDim, s_idx]

            # Identify which levels actually became supersaturated
            sbar = mtile.ref_state.sbar[:, 1]
            xibar = mtile.ref_state.xibar[:, 1]
            mubar = mtile.ref_state.mubar[:, 1]
            thermo = Scythe.thermodynamic_tuple.(sbar .+ s_before, xibar, mubar .+ mu_before)
            q_v0 = [x[1] for x in thermo]
            Tk0 = [x[3] for x in thermo]
            p0 = [x[4] for x in thermo]
            q_sat0 = Scythe.q_sat_liquid.(Tk0, p0)
            supersat = q_v0 .> q_sat0
            @test count(supersat) > 0

            # Where supersaturated: vapor condensed to liquid, entropy adjusted
            @test all(mu_after[supersat] .< mu_before[supersat])
            @test all(mu_l_after[supersat] .> mu_l_before[supersat])
            @test all(s_after[supersat] .!= s_before[supersat])

            # Linear mu transform: total water exactly conserved
            @test mu_after .+ mu_l_after ≈ mu_before .+ mu_l_before atol=1e-12

            # Subsaturated levels with no liquid water are untouched
            unsat = .!supersat
            @test all(mu_after[unsat] .== mu_before[unsat])
            @test all(mu_l_after[unsat] .== mu_l_before[unsat])
            @test all(s_after[unsat] .== s_before[unsat])

            # Everything stays finite
            @test all(isfinite.(mtile.var_np1))
        end
    end

    # ──────────────────────────────────────────────
    # 4. BF02_test equation set: resting state is preserved
    # ──────────────────────────────────────────────
    @testset "BF02_test resting state" begin
        mktempdir() do tmpdir
            mtile, patch, model = make_bf02_mtile(tmpdir)
            kDim = model.grid_params.kDim
            ncols = div(size(patch.physical, 1), kDim)

            for c in 1:ncols
                Scythe.advance_column(mtile, c, 1)
            end

            # Zero perturbations on a hydrostatic reference: tendencies vanish
            # and the state after one step is still at rest
            for v in [1, 2, 4, 5, 6, 7]   # s, xi, u, w, mu_l, qss
                @test maximum(abs.(mtile.expdot_n[:, v])) < 1.0e-10
            end
            @test maximum(abs.(mtile.var_np1)) < 1.0e-10
            @test all(isfinite.(mtile.var_np1))
        end
    end

    # ──────────────────────────────────────────────
    # 5. BF02_test with semi-implicit: Kvdiff=0 Helmholtz works
    # ──────────────────────────────────────────────
    @testset "Semi-implicit with Kvdiff=0" begin
        mktempdir() do tmpdir
            mtile, patch, model = make_bf02_mtile(tmpdir; semiimplicit=true)
            @test mtile.diffusion_matrix isa LinearAlgebra.Factorization
            @test mtile.h_matrix isa LinearAlgebra.Factorization

            # The factorized matrices must produce finite solutions
            n = size(mtile.diffusion_matrix, 1)
            x = mtile.diffusion_matrix \ ones(n)
            @test all(isfinite.(x))
            m = size(mtile.h_matrix, 1)
            y = mtile.h_matrix \ ones(m)
            @test all(isfinite.(y))

            # And a full column advance runs without error
            kDim = model.grid_params.kDim
            ncols = div(size(patch.physical, 1), kDim)
            for c in 1:ncols
                Scythe.advance_column(mtile, c, 1)
            end
            @test all(isfinite.(mtile.var_np1))
        end
    end
end
