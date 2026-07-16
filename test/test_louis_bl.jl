using Test
using Scythe
using Springsteel
using SparseArrays

# Tests for the Louis boundary layer (options[:louis_bl]) and the Smagorinsky
# horizontal closure (physical_params[:Ls] > 0) of the total-energy set
# (src/mc_boundary_layer.jl, mc_smag_k! in src/mc_geometry.jl).
#
# Driver-level tests isolate the BL contribution exactly by running the SAME
# state through advance_column with louis_bl on and off and differencing the
# explicit tendencies — every assertion below is on that difference, so
# advection/PGF/condensation cancel identically.

@testset "Louis boundary layer + Smagorinsky (mc)" begin

    import Springsteel.Thermodynamics: rho_v_sat, Rd, Rv, Cpd, gravity

    # ──────────────────────────────────────────────
    # 1. Pure pieces: mixing length, drag coefficient
    # ──────────────────────────────────────────────
    @testset "Louis mixing length" begin
        @test Scythe.louis_length(0.0, 80.0) == 0.0            # kappa*z -> 0 at the wall
        z = 10.0
        @test Scythe.louis_length(z, 80.0) ≈ 1.0 / (1.0 / (0.4 * z) + 1.0 / 80.0)
        @test Scythe.louis_length(1.0e6, 80.0) ≈ 80.0 rtol=1e-3  # asymptotic l_inf
        @test Scythe.louis_length(50.0, 80.0) > Scythe.louis_length(10.0, 80.0)
        @test Scythe.louis_length(50.0, 160.0) > Scythe.louis_length(50.0, 80.0)
    end

    @testset "Komori et al. (2018) drag coefficient" begin
        @test Scythe.komori_cd(3.0) == 1.0e-3
        @test Scythe.komori_cd(20.0) ≈ 4.4e-4 * sqrt(20.0)
        @test Scythe.komori_cd(33.6) == 2.55e-3
        @test Scythe.komori_cd(40.0) == 2.55e-3
    end

    # ──────────────────────────────────────────────
    # 2. Smagorinsky strain closure (analytic flows)
    # ──────────────────────────────────────────────
    @testset "Smagorinsky strain closure (axisym)" begin
        n = 8
        r = collect(range(1.0e4, 5.0e4; length=n))
        mkviews(f, f_x) = (f=f, f_x=f_x, f_xx=zeros(n), f_z=zeros(n), f_zz=zeros(n),
                           f_l=nothing, f_ll=nothing)
        K = zeros(n)
        Ls = 200.0

        # Solid-body rotation: every strain component vanishes -> the K_min floor
        Omg = 1.0e-4
        uv0 = mkviews(zeros(n), zeros(n))
        vvsb = mkviews(Omg .* r, fill(Omg, n))
        Scythe.mc_smag_k!(K, Scythe.MCAxisymRZ(), uv0, vvsb, r, Ls, 5.0)
        @test all(K .== 5.0)

        # v = gam*r^2: only S_rl = gam*r/2 -> K = Ls^2 * gam * r
        gam = 1.0e-8
        vvq = mkviews(gam .* r .^ 2, 2.0 .* gam .* r)
        Scythe.mc_smag_k!(K, Scythe.MCAxisymRZ(), uv0, vvq, r, Ls, 0.0)
        @test K ≈ Ls^2 .* gam .* r

        # Pure radial convergence u = -a*r: S_rr = S_ll = -a -> K = 2 Ls^2 a
        a = 1.0e-5
        uvc = mkviews(-a .* r, fill(-a, n))
        vv0 = mkviews(zeros(n), zeros(n))
        Scythe.mc_smag_k!(K, Scythe.MCAxisymRZ(), uvc, vv0, r, Ls, 0.0)
        @test K ≈ fill(2.0 * Ls^2 * a, n)
    end

    # ──────────────────────────────────────────────
    # 3. Driver-level: axisym RiRk tile machinery
    # ──────────────────────────────────────────────

    """Saturated cloudy column exactly on the Q_ss = 0 manifold (density form)."""
    function saturated_cloudy_column(z; q_l=1.0e-3)
        Tk = @. 290.0 - 0.005 * z
        p_Pa = @. 90000.0 * exp(-z / 8000.0)
        rho_v = rho_v_sat.(Tk, p_Pa ./ 100.0)
        rho_d = (p_Pa .- (Rv .* Tk .* rho_v)) ./ (Rd .* Tk)
        rho_c = q_l .* rho_d
        return (; z, Tk, p_Pa, rho_d, rho_v, rho_c)
    end

    """Dry, neutrally stable (theta = 300 K) analytic adiabat. The dry base pins
    the diagnostic vapor to exactly zero, so retrieval-level temperature wiggles
    (from the spline filter on an imposed KE profile) cannot leak into the vapor
    channel, whose (L_v - R_v T) energy factor amplifies even 1e-5 K wiggles to
    visible E_t tendencies."""
    function dry_adiabatic_column(z; theta0=300.0)
        n = length(z)
        exner = @. 1.0 - (gravity * z) / (Cpd * theta0)
        Tk = theta0 .* exner
        p_Pa = @. 100000.0 * exner^(Cpd / Rd)
        rho_d = p_Pa ./ (Rd .* Tk)
        return (; z, Tk, p_Pa, rho_d, rho_v = zeros(n), rho_c = zeros(n))
    end

    """Axisym RiRk tile with a Natural rho_r bottom (the surface-flux basis)."""
    function make_bl_mtile(tmpdir; louis_bl=true, Cd=-1.0, Ls=0.0, K_min=0.0,
                           sfc_wind_factor=1.0, l_inf=80.0, dry=false,
                           equation_set="moist_compressible_axisym")
        varlist = equation_set == "moist_compressible_XZ" ? Scythe.MC_VARS :
                                                            Scythe.MC_VARS_CYL
        vars = Dict(v => i for (i, v) in enumerate(varlist))
        scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
        side_bc = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))
        bot_bc = merge(scalar_bc, Dict("w" => DirichletBC(), "rho_r" => NaturalBC()))
        top_bc = merge(scalar_bc, Dict("w" => DirichletBC()))
        gp = GridParameters(geometry = "RiRk",
            iMin = 100.0e3, iMax = 125.6e3, num_cells_i = 8,
            kMin = 0.0, kMax = 2000.0, num_cells_k = 8,
            BCL = side_bc, BCR = side_bc, BCB = bot_bc, BCT = top_bc, vars = vars)
        ref_file = joinpath(tmpdir, "bl_pressure.ref")
        model = ModelParameters(
            ts = 0.1, integration_time = 1.0, output_interval = 1.0,
            equation_set = equation_set,
            ref_state_file = ref_file, grid_params = gp,
            physical_params = Dict(:Khdiff => 0.0, :Kvdiff => 0.0,
                                   :Kvdiff_heat => 0.0, :Kvdiff_water => 0.0,
                                   :tau_qss => 10.0, :alpha => 0.0, :z_damp => 20.0e3,
                                   :f => 0.0, :Cd => Cd, :Ls => Ls, :K_min => K_min,
                                   :l_inf => l_inf,
                                   :sfc_wind_factor => sfc_wind_factor),
            options = Dict(:semiimplicit => false, :exact_reference_state => true,
                           :precipitation => false, :louis_bl => louis_bl))
        # gp.kDim is derived inside ModelParameters (compute_derived_params) — the
        # grid must come from model.grid_params, not the raw gp.
        gp = model.grid_params
        patch = createGrid(gp)
        gridpoints = Scythe.getGridpoints(patch)
        kDim = gp.kDim
        z = gridpoints[1:kDim, end]
        col = dry ? dry_adiabatic_column(z) : saturated_cloudy_column(z)
        Scythe.write_exact_ref_mc(ref_file, z, col.p_Pa, col.rho_d, col.rho_v, col.rho_c)
        patch.physical .= 0.0
        spectralTransform!(patch)
        gridTransform!(patch)
        haloReceiveMap = sparse(Int64[], Int64[], Float64[],
                                size(patch.spectral, 1), size(patch.spectral, 2))
        mtile = createModelTile(patch, patch, model, haloReceiveMap)
        return mtile, patch, model, gp, z
    end

    """Advance every column once with louis_bl on and off on the SAME initial
    state (set by setup!(patch, vars, kDim, z)); return the per-point tendency
    difference D[:, slot] = BL contribution, plus the on-state grid handles."""
    function bl_increment(setup!; kwargs...)
        local D, patch_on, gp_on, z_on, mtile_on
        mktempdir() do tmp1
            mktempdir() do tmp2
                out = []
                for (tmp, flag) in ((tmp1, true), (tmp2, false))
                    mtile, patch, model, gp, z = make_bl_mtile(tmp; louis_bl=flag,
                                                               kwargs...)
                    setup!(patch, model.grid_params.vars, gp.kDim, z)
                    spectralTransform!(patch)
                    gridTransform!(patch)
                    ncols = div(size(patch.physical, 1), gp.kDim)
                    for c in 1:ncols
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
    Gauss-Legendre per 250-m cell (kMax = 2000), so the GL weights integrate
    the fitted profiles properly (trapezoid underestimates surface-localized
    divergences by ~11%)."""
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
    # 4. Resting state: the BL adds exactly nothing
    # ──────────────────────────────────────────────
    @testset "resting column: BL contribution is zero" begin
        mktempdir() do tmpdir
            mtile, patch, model, gp, z = make_bl_mtile(tmpdir; louis_bl=true)
            ncols = div(size(patch.physical, 1), gp.kDim)
            for c in 1:ncols
                Scythe.advance_column(mtile, c, 1)
            end
            scales = Dict(1 => 1.0e5, 2 => 1.0, 3 => 1.0, 4 => 1.0, 5 => 1.0,
                          6 => 2.0e8, 7 => 1.0e-2, 8 => 1.0, 9 => 1.0)
            for v in 1:9
                @test maximum(abs.(mtile.expdot_n[:, v])) / scales[v] < 1.0e-9
            end
        end
    end

    # NOTE on the wind tests below: E_t includes the kinetic energy, so any
    # imposed wind must come with E_t' = rho_t*ke or the temperature retrieval
    # reinterprets the same E_t as less internal energy — a real thermal
    # perturbation that (correctly) excites the heat/water mixing channels and
    # muddies the momentum-only assertions. u also carries Neumann vertical BCs
    # here (the physical mc configuration), so imposed shear profiles must have
    # zero end-slope to be representable on the basis.

    "E_t' = rho_t*ke compensation so an imposed wind leaves the retrieval resting."
    function compensate_ke!(patch, vars, kDim, z; dry=false)
        # Fit the imposed winds first and compensate with the FITTED kinetic
        # energy: a raw-value compensation leaves a ~1% thermal residual from
        # the spline fit/filter of the wind profile itself.
        spectralTransform!(patch)
        gridTransform!(patch)
        col = dry ? dry_adiabatic_column(z) : saturated_cloudy_column(z)
        rho_t = col.rho_d .+ col.rho_v .+ col.rho_c
        ui = vars["u"]; vi = vars["v"]; wi = vars["w"]; ei = vars["E_t"]
        for i in 1:size(patch.physical, 1)
            k = mod1(i, kDim)
            ke = 0.5 * (patch.physical[i, ui, 1]^2 + patch.physical[i, vi, 1]^2 +
                        patch.physical[i, wi, 1]^2)
            patch.physical[i, ei, 1] += rho_t[k] * ke
        end
        return nothing
    end

    # ──────────────────────────────────────────────
    # 5. Surface drag on a uniform swirl (no shear: Kv = 0, drag only)
    # ──────────────────────────────────────────────
    @testset "surface drag: sign, magnitude, Komori branches, KE invariant" begin
        for V0 in (20.0, 40.0)
            set_swirl! = (patch, vars, kDim, z) -> begin
                patch.physical[:, vars["v"], 1] .= V0
                compensate_ke!(patch, vars, kDim, z)
            end
            D, patch, gp, z, mtile = bl_increment(set_swirl!)
            kDim = gp.kDim
            vars = Dict(v => i for (i, v) in enumerate(Scythe.MC_VARS_CYL))

            npts = size(patch.physical, 1)
            ncols = div(npts, kDim)
            c = div(ncols, 2)                       # interior column
            rng = ((c - 1) * kDim + 1):(c * kDim)
            v_fit = patch.physical[rng, vars["v"], 1]
            rho_t = Springsteel.ref_rho_t(mtile.ref_state)[:, 1]

            D4 = D[rng, 4]; D5 = D[rng, 5]; D9 = D[rng, 9]; D6 = D[rng, 6]

            # u and w carry no drag (u1 = 0) and no mixing (no shear)
            @test maximum(abs.(D4)) < 1.0e-10
            @test maximum(abs.(D5)) < 1.0e-10

            # Column-integrated v momentum loss = the surface stress
            # rho_t(1) Cd U1 v1, with U1 = v1 = the FITTED surface swirl. The
            # analytic g(z) delivery makes this exact up to the quadrature.
            v1 = v_fit[1]
            Cd = Scythe.komori_cd(abs(v1))
            got = trapz(z, rho_t .* D9)
            want = -rho_t[1] * Cd * v1 * v1
            @test got < 0.0
            @test isapprox(got, want; rtol=0.02)

            # E_t follows the resolved KE down pointwise: dE = rho_t * v * dv/dt
            # (thermo channels are inert with the KE-compensated E_t)
            @test isapprox(D6, rho_t .* v_fit .* D9;
                           atol=1.0e-4 * maximum(abs.(D6)))
        end
        # V0 = 40 hits the Komori cap: check the branch actually engaged
        @test Scythe.komori_cd(40.0) == 2.55e-3
    end

    # ──────────────────────────────────────────────
    # 6. Interior mixing of sheared v (drag inert: v(0) = 0)
    # ──────────────────────────────────────────────
    @testset "interior momentum mixing: conservation, magnitude, KE invariant" begin
        V0 = 2.0
        H = 2000.0
        # v = V0 sin^2(pi z / 2H): zero value AND zero slope at both ends, so the
        # profile is representable on the Neumann basis and the surface drag and
        # boundary fluxes all vanish — interior mixing must conserve momentum.
        set_shear! = (patch, vars, kDim, z) -> begin
            for i in 1:size(patch.physical, 1)
                k = mod1(i, kDim)
                patch.physical[i, vars["v"], 1] = V0 * sin(0.5 * pi * z[k] / H)^2
            end
            compensate_ke!(patch, vars, kDim, z; dry=true)
        end
        D, patch, gp, z, mtile = bl_increment(set_shear!; Cd=0.0, dry=true)
        kDim = gp.kDim
        vars = Dict(v => i for (i, v) in enumerate(Scythe.MC_VARS_CYL))
        npts = size(patch.physical, 1)
        ncols = div(npts, kDim)
        c = div(ncols, 2)
        rng = ((c - 1) * kDim + 1):(c * kDim)
        v_fit = patch.physical[rng, vars["v"], 1]
        vz_fit = patch.physical[rng, vars["v"], 4]   # dz slot (2D grids: slot 4)
        rho_t = Springsteel.ref_rho_t(mtile.ref_state)[:, 1]

        D9 = D[rng, 9]; D6 = D[rng, 6]
        @test maximum(abs.(D9)) > 0.0

        # Interior mixing conserves column momentum (zero boundary fluxes)
        @test abs(trapz(z, rho_t .* D9)) < 0.05 * trapz(z, abs.(rho_t .* D9))

        # Quantitative: rho_t*dv/dt matches a finite-difference d/dz of the
        # analytic flux rho_t*Kv*dv/dz built from the FITTED shear, away from the
        # boundary nodes where the fitted bottom-node correction is localized
        flux = rho_t .* Scythe.louis_length.(z, 80.0) .^ 2 .* abs.(vz_fit) .* vz_fit
        fd = [(flux[i+1] - flux[i-1]) / (z[i+1] - z[i-1]) for i in 4:kDim-3]
        got = rho_t[4:kDim-3] .* D9[4:kDim-3]
        @test maximum(abs.(got .- fd)) < 0.25 * maximum(abs.(got))

        # E_t follows the resolved KE pointwise. Tolerance 2%: the fitted-KE
        # compensation is filtered through the spline basis (sin^4 harmonics at
        # ~4 cells/wavelength), leaving a ~1% s_t' residual that the ACTIVE heat
        # channel genuinely mixes; the tight (1e-4) version of this identity is
        # verified in the drag test above, where Kv = 0 keeps the channel inert.
        @test isapprox(D6, rho_t .* v_fit .* D9; atol=2.0e-2 * maximum(abs.(D6)))
    end

    # ──────────────────────────────────────────────
    # 7. Scalar mixing conserves column energy and total water
    # ──────────────────────────────────────────────
    @testset "interior scalar mixing: closed column books" begin
        V0 = 2.0
        H = 2000.0
        set_moist! = (patch, vars, kDim, z) -> begin
            for i in 1:size(patch.physical, 1)
                k = mod1(i, kDim)
                # Background shear so Kv > 0 (v(0) = 0: no drag), plus a vapor
                # bump confined to the lower half with zero-gradient ends
                patch.physical[i, vars["v"], 1] = V0 * sin(0.5 * pi * z[k] / H)^2
                patch.physical[i, vars["Q_ss"], 1] =
                    z[k] < 1000.0 ? 1.0e-4 * (1.0 + cos(pi * z[k] / 1000.0)) : 0.0
            end
            compensate_ke!(patch, vars, kDim, z)
        end
        D, patch, gp, z, mtile = bl_increment(set_moist!; Cd=0.0)
        kDim = gp.kDim
        vars = Dict(v => i for (i, v) in enumerate(Scythe.MC_VARS_CYL))
        npts = size(patch.physical, 1)
        ncols = div(npts, kDim)
        c = div(ncols, 2)
        rng = ((c - 1) * kDim + 1):(c * kDim)
        rho_t = Springsteel.ref_rho_t(mtile.ref_state)[:, 1]
        u_fit = patch.physical[rng, vars["u"], 1]
        v_fit = patch.physical[rng, vars["v"], 1]
        w_fit = patch.physical[rng, vars["w"], 1]

        D1 = D[rng, 1]; D3 = D[rng, 3]; D4 = D[rng, 4]; D5 = D[rng, 5]
        D6 = D[rng, 6]; D7 = D[rng, 7]; D9 = D[rng, 9]

        # The moisture perturbation excites the vapor and heat channels
        @test maximum(abs.(D7)) > 0.0
        @test maximum(abs.(D1)) > 0.0
        @test all(isfinite.(D))

        # Interior mixing with zero surface nodes conserves the column energy of
        # the SCALAR channels (subtract the resolved-KE part, which tracks the
        # momentum mixing and is not separately conservative) and total water
        D6s = D6 .- rho_t .* ((u_fit .* D4) .+ (w_fit .* D5) .+ (v_fit .* D9))
        @test abs(trapz(z, D6s)) < 0.05 * trapz(z, abs.(D6s)) + 1.0e-30
        @test abs(trapz(z, D3)) < 0.05 * trapz(z, abs.(D3)) + 1.0e-30
    end

    # ──────────────────────────────────────────────
    # 8. Guard rails
    # ──────────────────────────────────────────────
    @testset "Smagorinsky on the Cartesian slice errors" begin
        mktempdir() do tmpdir
            mtile, patch, model, gp, z = make_bl_mtile(tmpdir; louis_bl=false,
                                                       Ls=200.0, K_min=5.0,
                                                       equation_set="moist_compressible_XZ")
            @test_throws Exception Scythe.advance_column(mtile, 1, 1)
        end
    end
end
