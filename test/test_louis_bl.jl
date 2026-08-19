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

    import Springsteel.Thermodynamics: rho_v_sat, Rd, Rv, Cpd, Cvv, gravity

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

    """Axisym RiRk tile with a Natural rho_r bottom (the surface-flux basis).

    `ctrans` / `rtrans` select the water control-variable transforms. With both `:none`
    (the default) every dict below is built from names identical to the old literals, so
    the fixture is bitwise what it was."""
    function make_bl_mtile(tmpdir; louis_bl=true, Cd=-1.0, Ls=0.0, K_min=0.0,
                           sfc_wind_factor=1.0, l_inf=80.0, dry=false,
                           ctrans=:none, rtrans=:none, cmu=1.0e-7, rmu=1.0e-7,
                           equation_set="moist_compressible_axisym")
        cyl = equation_set != "moist_compressible_XZ"
        # Every name-keyed dict must come from mc_var_names, not a literal: a stale key is
        # ignored silently rather than raising (Scythe.check_mc_var_names is the backstop).
        opts_names = Dict{Symbol,Any}(:condensate_transform => ctrans,
                                      :rain_transform => rtrans)
        varlist = Scythe.mc_var_names(opts_names; cyl = cyl)
        rain_name = Scythe.rain_var_name(opts_names)
        vars = Dict(v => i for (i, v) in enumerate(varlist))
        scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
        side_bc = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))
        bot_bc = merge(scalar_bc, Dict("w" => DirichletBC(), rain_name => NaturalBC()))
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
                                   :condensate_mu => cmu, :rain_mu => rmu,
                                   :sfc_wind_factor => sfc_wind_factor),
            options = Dict{Symbol,Any}(:semiimplicit => true,
                           :exact_reference_state => true,
                           :precipitation => false, :louis_bl => louis_bl,
                           :condensate_transform => ctrans,
                           :rain_transform => rtrans))
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
        # Differenced on/off, per this file's design (see the header): the assertion is
        # that the BOUNDARY LAYER adds nothing at rest, which is not the same as the
        # equation set having no resting tendency at all.
        #
        # It matters here. This tile's base is a saturated cloudy column, and the
        # reference's pointwise water partition is not bit-identical to the one the
        # model diagnoses from the FITTED densities -- they differ by the spline fit
        # error, ~9e-8 kg/m^3 of vapor. qss_relaxation reconciles that at 1/tau, so
        # slot 7 carries a resting tendency of ~9e-9 kg/m^3/s that belongs to the
        # equation set, not to the BL. It is thermodynamically inert (the retrieval
        # does not read Q_ss), it is bounded by the fit error rather than accumulating,
        # and options[:consistent_qss_reference] removes it at the source. Crucially
        # the CONDENSATE tendency is ~4e-18: no cloud is being manufactured at rest,
        # which is the property this whole formulation exists to guarantee.
        mktempdir() do tmpdir
            m_on, p_on, _, gp, _ = make_bl_mtile(tmpdir; louis_bl=true)
            m_off, p_off, _, _, _ = make_bl_mtile(tmpdir; louis_bl=false)
            ncols = div(size(p_on.physical, 1), gp.kDim)
            for c in 1:ncols
                Scythe.advance_column(m_on, c, 1)
                Scythe.advance_column(m_off, c, 1)
            end
            scales = Dict(1 => 1.0e5, 2 => 1.0, 3 => 1.0, 4 => 1.0, 5 => 1.0,
                          6 => 2.0e8, 7 => 1.0e-2, 8 => 1.0, 9 => 1.0, 10 => 1.0,
                          11 => 1.0)                       # 11 = the prognostic vapor
            for v in 1:11
                d = maximum(abs.(m_on.expdot_n[:, v] .- m_off.expdot_n[:, v]))
                @test d / scales[v] < 1.0e-9
            end
            # The BL adds EXACTLY nothing to the cloud at rest, and this is the sharper
            # statement than an absolute bound: at rest the Louis diffusivity is
            # `l^2 |dV/dz| = 0`, so every eddy flux is an exact zero and the on/off runs
            # agree bit for bit.
            @test m_on.expdot_n[:, 9] == m_off.expdot_n[:, 9]
            # The ABSOLUTE residual is not zero on this SATURATED base, and it is not the
            # BL's. The fitted reference vapor and rho_vs(T_retrieved) disagree at the fit
            # level, so the condensation closure runs at ~1e-8 kg/m^3/s with no
            # perturbation anywhere — the reference-state crumb `consistent_qss_reference`
            # exists to remove (see its testsets in test_moist_compressible.jl). It used to
            # be masked here: the retired regime-blended retrieval made rho_v equal
            # Q_ss + rho_vs identically in cloud, so the drive clip returned Q_ssbar = 0
            # exactly and the crumb was invisible rather than absent.
            @test maximum(abs.(m_on.expdot_n[:, 9])) < 1.0e-7
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

            D4 = D[rng, 4]; D5 = D[rng, 5]; D9 = D[rng, 10]; D6 = D[rng, 6]

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

        D9 = D[rng, 10]; D6 = D[rng, 6]
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
        # The vapor bump must be seeded as WATER MASS, not as Q_ss. Q_ss no longer
        # carries the vapor: rho_v is the residual rho_t - rho_d - rho_c - rho_r, and
        # the retrieval does not read Q_ss at all, so a Q_ss-only bump (what this test
        # used to impose) moves no water, no energy and no temperature — it perturbs
        # the supersaturation TRACKER and nothing else. Verified: with the old seed
        # the total-water and cloud increments are identically 0.0.
        #
        # This is the T-invariant vapor seed of the water-diffusion testsets:
        # drho_t = drho_v = seed at fixed rho_d, with the vapor partial pressure
        # (dp = Rv*T*seed) and the vapor internal + potential energy
        # (dE_t = (Cvv*T + g*z)*seed) added so the retrieval is untouched. Q_ss moves
        # with it so the tracker stays consistent and the reconciliation stays quiet.
        set_moist! = (patch, vars, kDim, z) -> begin
            col_bl = saturated_cloudy_column(z)
            for i in 1:size(patch.physical, 1)
                k = mod1(i, kDim)
                # Background shear so Kv > 0 (v(0) = 0: no drag), plus a vapor
                # bump confined to the lower half with zero-gradient ends
                patch.physical[i, vars["v"], 1] = V0 * sin(0.5 * pi * z[k] / H)^2
                seed = z[k] < 1000.0 ? 1.0e-4 * (1.0 + cos(pi * z[k] / 1000.0)) : 0.0
                Tref = col_bl.Tk[k]
                patch.physical[i, vars["rho_t"], 1] += seed
                # ...and the VAPOR SLOT, which is prognostic: `drho_t = drho_v = seed` is
                # the whole point of the T-invariant seed, and leaving rho_v behind would
                # make the bump a reconciliation gap rather than a moisture perturbation.
                patch.physical[i, vars["rho_v"], 1] += seed
                patch.physical[i, vars["Q_ss"], 1] += seed
                patch.physical[i, vars["p"], 1] += Rv * Tref * seed
                patch.physical[i, vars["E_t"], 1] +=
                    ((Cvv * Tref) + (gravity * z[k])) * seed
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
        D6 = D[rng, 6]; D7 = D[rng, 7]; D9 = D[rng, 10]

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

        # THE VAPOR FLUX LANDS ON THE VAPOR SLOT. The closure mixes TOTAL water and CLOUD;
        # the vapor's share of that flux is `vdot_v = vdot_w - vdot_c`, which used to be
        # inferred after the fact from the other two and is now applied to slot rho_v
        # directly. With no condensate transform the cloud Jacobian is an exact 1.0, so the
        # three legs agree BITWISE and the closure opens no reconciliation gap at all.
        rv_i = vars["rho_v"]
        Drv = D[rng, rv_i]; Dc = D[rng, vars["rho_c"]]
        @test maximum(abs.(Drv)) > 0.0                   # the leg is live
        # Not `==`: D3 and Dc are each an on/off DIFFERENCE, so `D3 .- Dc` reassociates
        # what the kernel computed as `vdot_w - vdot_c` in one expression. The agreement is
        # eleven decades under the signal, which is that reassociation and nothing else.
        sc = maximum(abs.(Drv))
        @test maximum(abs.(Drv .- (D3 .- Dc))) < 1.0e-9 * sc
        @test all(isfinite.(Drv))
        # ...and it conserves the column vapor for the same reason slot 3 conserves total
        # water: interior mixing with zero surface nodes moves it around, not in or out.
        @test abs(trapz(z, Drv)) < 0.05 * trapz(z, abs.(Drv)) + 1.0e-30
    end

    # ──────────────────────────────────────────────
    # 7b. Cloud mixing under the condensate transform
    # ──────────────────────────────────────────────
    @testset "Louis BL cloud mixing under the condensate transform" begin
        import Springsteel.Thermodynamics: Cl
        V0 = 2.0
        H = 2000.0
        mu = 1.0e-7

        # A cloud BOUNDED AWAY FROM ZERO (5e-4 .. 1.5e-3 kg/m^3, i.e. rho_c >> mu
        # everywhere) with zero end-slope so it is representable on the Neumann basis.
        # That is deliberate and it is what makes this test sharp: where rho_c >> mu,
        # bhyp is EXACTLY affine -- bhyp(rho) = (rho+mu)/2 - mu^2/(2(rho+mu)) -- and a
        # spline fit is linear, so the two arms' fitted cloud fields agree to ~1e-8
        # relative (measured ~5e-9 here) and every difference below is the SCHEME, not
        # the fit. It is the same regime in which bf02_moist reproduces the untransformed
        # run to twelve significant figures.
        cloud(zk) = 1.0e-3 * (1.0 + (0.5 * cos(pi * zk / H)))

        # Seed cloud as MASS, T-invariant: rho_t carries the same increment (so the vapor
        # residual stays ~0 and the vapor rate is exactly zero in the CORRECT scheme --
        # which is precisely the quantity the bug corrupts), liquid adds no partial
        # pressure, and E_t takes the liquid internal + potential energy.
        function set_cloud!(tf)
            return (patch, vars, kDim, z) -> begin
                col_bl = dry_adiabatic_column(z)
                ci = Scythe.mc_slot(vars, "rho_c")
                for i in 1:size(patch.physical, 1)
                    k = mod1(i, kDim)
                    patch.physical[i, vars["v"], 1] = V0 * sin(0.5 * pi * z[k] / H)^2
                    rc = cloud(z[k])
                    # rho_cbar == 0 on the dry base, so nu' = bhyp(rho_c) with no
                    # background to subtract.
                    patch.physical[i, ci, 1] = Scythe.condensate_slot(rc, 0.0, tf, mu)
                    patch.physical[i, vars["rho_t"], 1] += rc
                    patch.physical[i, vars["E_t"], 1] +=
                        ((Cl * col_bl.Tk[k]) + (gravity * z[k])) * rc
                end
                compensate_ke!(patch, vars, kDim, z; dry=true)
            end
        end

        Dn, patch, gp, z, mtile = bl_increment(set_cloud!(:none); Cd=0.0, dry=true,
                                               ctrans=:none, cmu=mu)
        Db, _, _, _, _ = bl_increment(set_cloud!(:bhyp); Cd=0.0, dry=true,
                                      ctrans=:bhyp, cmu=mu)
        kDim = gp.kDim
        ncols = div(size(patch.physical, 1), kDim)
        c = div(ncols, 2)
        rng = ((c - 1) * kDim + 1):(c * kDim)

        @test all(isfinite.(Db))
        # The BL is genuinely mixing cloud, or nothing below means anything
        @test maximum(abs.(Dn[rng, 9])) > 0.0

        # THE BUG WITNESS. Slots 1 (p), 6 (E_t) and 7 (Q_ss) are fed through
        # `vdot_v = vdot_w - vdot_c`. Before the fix `vdot_c` was a nu-space rate --
        # here exactly half the density rate, since J == 0.5 for rho_c >> mu -- being
        # subtracted from a density rate, so the transformed arm carried a spurious
        # vapor tendency of 0.5*rhodot_c into all three. With the flux built in density
        # space they agree. Slot 3 (total water) never saw the bug and is the control.
        for s in (1, 3, 6, 7)
            scale = maximum(abs.(Dn[rng, s]))
            @test isapprox(Db[rng, s], Dn[rng, s]; atol = 1.0e-6 * scale + 1.0e-300)
        end

        # The Jacobian, at its exact linear-regime value: slot 9 holds nu, so it receives
        # J*rhodot_c, and J == 0.5 + O((mu/rho)^2) here.
        #
        # NOTE this line does NOT discriminate the bug, and cannot: for rho_c >> mu the old
        # code's Kv*dz(nu) IS half the density flux, so the two formulations agree on slot 9
        # algebraically. Verified by reverting the fix -- slot 9 passes, slots 1/6/7 fail
        # (slot 6 by more than a factor of two). The discrimination lives above; this is the
        # consistency check that the Jacobian is applied once and in the right direction.
        sc9 = maximum(abs.(Dn[rng, 9]))
        @test isapprox(Db[rng, 9], 0.5 .* Dn[rng, 9]; atol = 1.0e-6 * sc9)

        # Interior mixing with no surface cloud source conserves column cloud mass, in
        # DENSITY terms -- i.e. after dividing the transformed slot's rate back by J.
        d9 = Db[rng, 9] ./ 0.5
        @test abs(trapz(z, d9)) < 0.05 * trapz(z, abs.(d9))

        # Cloud-free reference AND no cloud perturbation => nu' == 0 => nu_z == 0 => the
        # transformed BL adds EXACTLY zero to slot 9. Asserted as == 0.0, not a tolerance.
        set_bare! = (patch, vars, kDim, z) -> begin
            for i in 1:size(patch.physical, 1)
                k = mod1(i, kDim)
                patch.physical[i, vars["v"], 1] = V0 * sin(0.5 * pi * z[k] / H)^2
            end
            compensate_ke!(patch, vars, kDim, z; dry=true)
        end
        D0, _, _, _, _ = bl_increment(set_bare!; Cd=0.0, dry=true, ctrans=:bhyp, cmu=mu)
        @test all(D0[:, 9] .== 0.0)

        # Rain is deliberately out of scope: the BL borrows slot 8's spline column as a
        # basis but never its values, so `rain_transform` needs nothing here.
        Dr, _, _, _, _ = bl_increment(set_cloud!(:none); Cd=0.0, dry=true,
                                      rtrans=:bhyp, rmu=mu)
        @test all(Dr[:, 8] .== 0.0)
    end

    # ──────────────────────────────────────────────
    # 7c. A negative vapor must not kill the column
    # ──────────────────────────────────────────────
    @testset "BL runs where the vapor is negative" begin
        # The BL stages the moist entropy s_t as its heat control variable, and
        # `entropy` takes log(q_v*rho_d/rho_v0). Negative vapor is a RESOLUTION
        # DIAGNOSTIC and is never clamped, so q_v does go negative -- at the tropopause,
        # where the vapor present is smaller than the undershoot of an unresolved spike --
        # and this used to throw a DomainError out of the first call of the first timestep.
        # It is what stopped the 3-nest TC dead, transformed or not.
        #
        # The seed is now on the VAPOR SLOT itself. It used to remove total water and let
        # the residual go negative; with rho_v prognostic that would move rho_t and leave
        # the vapor alone, i.e. it would test the reconciliation gap instead of the
        # negative vapor. The dry base has rho_vbar == 0 exactly, so a negative
        # perturbation IS a negative vapor.
        V0 = 2.0
        H = 2000.0
        set_negvapor! = (patch, vars, kDim, z) -> begin
            for i in 1:size(patch.physical, 1)
                k = mod1(i, kDim)
                patch.physical[i, vars["v"], 1] = V0 * sin(0.5 * pi * z[k] / H)^2
                patch.physical[i, vars["rho_t"], 1] -= 5.0e-6
                patch.physical[i, vars["rho_v"], 1] -= 5.0e-6
            end
            compensate_ke!(patch, vars, kDim, z; dry=true)
        end
        D, patch, gp, z, mtile = bl_increment(set_negvapor!; Cd=0.0, dry=true)
        @test all(isfinite.(D))
        @test maximum(abs.(D)) > 0.0        # it really did run the BL
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
