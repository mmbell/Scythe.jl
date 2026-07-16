using Test
using Scythe
using Springsteel
using SparseArrays

# Tests for the bulk surface enthalpy/moisture fluxes (options[:surface_fluxes])
# of the total-energy set: F_sh = rho_d1*Cpd*Ck*U1*(SST - T1) and
# F_q = Ck*U1*(rho_vs(SST, p1) - rho_v1) enter as the bottom nodes of the
# Louis-BL heat and water flux columns (src/mc_boundary_layer.jl).
#
# The tests difference surface_fluxes on/off with louis_bl ON in both runs, so
# the drag and interior mixing cancel identically and D is exactly the surface
# flux contribution. A UNIFORM swirl carries no shear (Kv = 0), so the interior
# fluxes vanish and only the surface nodes act.

@testset "Surface enthalpy + moisture fluxes (mc)" begin

    import Springsteel.Thermodynamics: rho_v_sat, Rd, Rv, Cpd, Cvv, gravity

    """Dry, neutrally stable (theta = 300 K) analytic adiabat: subsaturated air
    over a warm sea, so both flux channels have a well-defined sign."""
    function dry_adiabatic_column(z; theta0=300.0)
        n = length(z)
        exner = @. 1.0 - (gravity * z) / (Cpd * theta0)
        Tk = theta0 .* exner
        p_Pa = @. 100000.0 * exner^(Cpd / Rd)
        rho_d = p_Pa ./ (Rd .* Tk)
        return (; z, Tk, p_Pa, rho_d, rho_v = zeros(n), rho_c = zeros(n))
    end

    """Axisym RiRk tile (Natural rho_r bottom) over a fixed-SST ocean."""
    function make_sfc_mtile(tmpdir; surface_fluxes=true, SST=nothing, Ck=1.0e-3,
                            U_min=0.0, Cd=-1.0)
        vars = Dict(v => i for (i, v) in enumerate(Scythe.MC_VARS_CYL))
        scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
        side_bc = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))
        bot_bc = merge(scalar_bc, Dict("w" => DirichletBC(), "rho_r" => NaturalBC()))
        top_bc = merge(scalar_bc, Dict("w" => DirichletBC()))
        gp = GridParameters(geometry = "RiRk",
            iMin = 100.0e3, iMax = 125.6e3, num_cells_i = 8,
            kMin = 0.0, kMax = 2000.0, num_cells_k = 8,
            BCL = side_bc, BCR = side_bc, BCB = bot_bc, BCT = top_bc, vars = vars)
        ref_file = joinpath(tmpdir, "sfc_pressure.ref")
        pp = Dict(:Khdiff => 0.0, :Kvdiff => 0.0, :Kvdiff_heat => 0.0,
                  :Kvdiff_water => 0.0, :tau_qss => 10.0, :alpha => 0.0,
                  :z_damp => 20.0e3, :f => 0.0, :Cd => Cd, :Ls => 0.0,
                  :K_min => 0.0, :l_inf => 80.0, :sfc_wind_factor => 1.0,
                  :Ck => Ck, :U_min => U_min)
        SST === nothing || (pp[:SST] = SST)
        model = ModelParameters(
            ts = 0.1, integration_time = 1.0, output_interval = 1.0,
            equation_set = "moist_compressible_axisym",
            ref_state_file = ref_file, grid_params = gp,
            physical_params = pp,
            options = Dict(:semiimplicit => true, :exact_reference_state => true,
                           :precipitation => false, :louis_bl => true,
                           :surface_fluxes => surface_fluxes))
        gp = model.grid_params
        patch = createGrid(gp)
        gridpoints = Scythe.getGridpoints(patch)
        kDim = gp.kDim
        z = gridpoints[1:kDim, end]
        col = dry_adiabatic_column(z)
        Scythe.write_exact_ref_mc(ref_file, z, col.p_Pa, col.rho_d, col.rho_v, col.rho_c)
        patch.physical .= 0.0
        spectralTransform!(patch)
        gridTransform!(patch)
        haloReceiveMap = sparse(Int64[], Int64[], Float64[],
                                size(patch.spectral, 1), size(patch.spectral, 2))
        mtile = createModelTile(patch, patch, model, haloReceiveMap)
        return mtile, patch, model, gp, z
    end

    "Set a uniform swirl v = V0 and compensate E_t with the FITTED kinetic energy."
    function set_swirl!(patch, vars, kDim, z, V0)
        patch.physical[:, vars["v"], 1] .= V0
        spectralTransform!(patch)
        gridTransform!(patch)
        col = dry_adiabatic_column(z)
        for i in 1:size(patch.physical, 1)
            k = mod1(i, kDim)
            ke = 0.5 * (patch.physical[i, vars["u"], 1]^2 +
                        patch.physical[i, vars["v"], 1]^2 +
                        patch.physical[i, vars["w"], 1]^2)
            patch.physical[i, vars["E_t"], 1] += col.rho_d[k] * ke
        end
        return nothing
    end

    """Tendency difference D = (surface_fluxes on) - (surface_fluxes off), with
    louis_bl on in both; also returns the on-state handles."""
    function sfc_increment(V0; kwargs...)
        local D, patch_on, gp_on, z_on, mtile_on
        mktempdir() do tmp1
            mktempdir() do tmp2
                out = []
                for (tmp, flag) in ((tmp1, true), (tmp2, false))
                    mtile, patch, model, gp, z = make_sfc_mtile(tmp;
                        surface_fluxes=flag, kwargs...)
                    vars = model.grid_params.vars
                    set_swirl!(patch, vars, gp.kDim, z, V0)
                    spectralTransform!(patch)
                    gridTransform!(patch)
                    for c in 1:div(size(patch.physical, 1), gp.kDim)
                        Scythe.advance_column(mtile, c, 1)
                    end
                    push!(out, (mtile, patch, gp, z))
                end
                (mtile_on, patch_on, gp_on, z_on) = out[1]
                (mtile_off, _, _, _) = out[2]
                D = mtile_on.expdot_n .- mtile_off.expdot_n
            end
        end
        return D, patch_on, gp_on, z_on, mtile_on
    end

    """Exact column integral on the RiRk mish: the points are 3-node
    Gauss-Legendre per 250-m cell, so the GL weights integrate the fitted
    profiles properly (trapezoid underestimates the surface-localized flux
    divergence by ~11%)."""
    function trapz(z, y)
        kDim = length(y)
        dz = 2000.0 / (kDim / 3)
        w = (dz / 2.0) .* (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)
        s = 0.0
        for c in 0:div(kDim, 3)-1, j in 1:3
            s += w[j] * y[3c + j]
        end
        return s
    end

    # ──────────────────────────────────────────────
    # 1. Closed energy and water books over a warm sea
    # ──────────────────────────────────────────────
    @testset "closed column books: energy and water gains = surface fluxes" begin
        V0 = 15.0
        Ck = 1.0e-3
        D, patch, gp, z, mtile = sfc_increment(V0; Ck=Ck)   # default SST = 301.15 K
        kDim = gp.kDim
        vars = Dict(v => i for (i, v) in enumerate(Scythe.MC_VARS_CYL))
        npts = size(patch.physical, 1)
        ncols = div(npts, kDim)
        c = div(ncols, 2)
        rng = ((c - 1) * kDim + 1):(c * kDim)

        # Expected fluxes from the driver's own state: reference thermo at the
        # lowest mish point and the FITTED surface swirl
        rho_dbar = Springsteel.ref_rho_d(mtile.ref_state)[:, 1]
        pbar = Springsteel.ref_pressure(mtile.ref_state)[:, 1]
        Tbar = mtile.ref_state.Tbar[:, 1]
        v1 = patch.physical[rng, vars["v"], 1][1]
        U1 = abs(v1)
        SST = 301.15
        F_sh = rho_dbar[1] * Cpd * Ck * U1 * (SST - Tbar[1])
        F_q = Ck * U1 * rho_v_sat(SST, pbar[1] / 100.0)     # rho_v1 = 0 (dry base)
        @test F_sh > 0.0
        @test F_q > 0.0

        D1 = D[rng, 1]; D3 = D[rng, 3]; D6 = D[rng, 6]; D7 = D[rng, 7]

        # Total water gain telescopes to F_q exactly (up to the spline fit)
        @test isapprox(trapz(z, D3), F_q; rtol=0.1)

        # Energy gain = sensible flux + the internal+potential energy the vapor
        # carries at ambient conditions: F_sh + (Cvv*T1 + g*z1 + ke1)*F_q. The
        # energy coefficients vary over the (surface-localized) flux divergence,
        # so the tolerance is a few percent.
        ke1 = 0.5 * v1^2
        want_E = F_sh + ((Cvv * Tbar[1]) + (gravity * z[1]) + ke1) * F_q
        @test isapprox(trapz(z, D6), want_E; rtol=0.1)

        # Signs: the lower levels warm/moisten (pressure and Q_ss rise)
        @test D1[1] > 0.0
        @test D7[1] > 0.0
        # and the response is surface-localized (upper half ~unaffected)
        @test maximum(abs.(D3[div(kDim, 2):end])) < 0.05 * maximum(abs.(D3))
    end

    # ──────────────────────────────────────────────
    # 2. Dry/moist split and the calm limit
    # ──────────────────────────────────────────────
    @testset "SST = T1: only the moisture chain fires" begin
        V0 = 15.0
        # Choose SST equal to the reference surface temperature at the lowest
        # mish point; the sensible flux vanishes, the moisture flux does not.
        D0, patch0, gp0, z0, mtile0 = sfc_increment(V0)      # to read T1
        Tbar = mtile0.ref_state.Tbar[:, 1]
        D, patch, gp, z, mtile = sfc_increment(V0; SST=Tbar[1])
        kDim = gp.kDim
        rng = 1:kDim
        pbar = Springsteel.ref_pressure(mtile.ref_state)[:, 1]
        vars = Dict(v => i for (i, v) in enumerate(Scythe.MC_VARS_CYL))
        v1 = patch.physical[rng, vars["v"], 1][1]
        F_q = 1.0e-3 * abs(v1) * rho_v_sat(Tbar[1], pbar[1] / 100.0)
        @test isapprox(trapz(z, D[rng, 3]), F_q; rtol=0.1)
        # Energy gain reduces to the vapor internal+potential energy term
        ke1 = 0.5 * v1^2
        want_E = ((Cvv * Tbar[1]) + (gravity * z[1]) + ke1) * F_q
        @test isapprox(trapz(z, D[rng, 6]), want_E; rtol=0.1)
    end

    @testset "calm limit: no wind, no fluxes (bitwise)" begin
        D, patch, gp, z, mtile = sfc_increment(0.0)
        @test maximum(abs.(D)) == 0.0
    end

    @testset "gustiness floor U_min revives the calm fluxes" begin
        U_min = 5.0
        D, patch, gp, z, mtile = sfc_increment(0.0; U_min=U_min)
        kDim = gp.kDim
        pbar = Springsteel.ref_pressure(mtile.ref_state)[:, 1]
        F_q = 1.0e-3 * U_min * rho_v_sat(301.15, pbar[1] / 100.0)
        @test isapprox(trapz(z, D[1:kDim, 3]), F_q; rtol=0.1)
        # No momentum drag from the floor (u1 = v1 = 0: the stress is still 0)
        @test maximum(abs.(D[:, 4])) == 0.0
        @test maximum(abs.(D[:, 9])) == 0.0
    end

    # ──────────────────────────────────────────────
    # 3. Linearity in the surface wind
    # ──────────────────────────────────────────────
    @testset "fluxes scale linearly with U1" begin
        vars = Dict(v => i for (i, v) in enumerate(Scythe.MC_VARS_CYL))
        got = Float64[]
        v1s = Float64[]
        for V0 in (10.0, 20.0)
            D, patch, gp, z, mtile = sfc_increment(V0)
            kDim = gp.kDim
            push!(got, trapz(z, D[1:kDim, 3]))
            push!(v1s, patch.physical[1:kDim, vars["v"], 1][1])
        end
        @test isapprox(got[2] / got[1], v1s[2] / v1s[1]; rtol=0.02)
    end

    # ──────────────────────────────────────────────
    # 4. Config guard rails
    # ──────────────────────────────────────────────
    @testset "SST in Celsius (or otherwise absurd) errors" begin
        mktempdir() do tmpdir
            mtile, patch, model, gp, z = make_sfc_mtile(tmpdir; SST=28.0)
            @test_throws Exception Scythe.advance_column(mtile, 1, 1)
        end
    end
end
