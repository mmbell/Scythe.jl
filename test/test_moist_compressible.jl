using Test
using Scythe
using Springsteel

# Tests for the total-energy moist compressible equation set moist_compressible_XZ
# (src/moist_compressible.jl): prognostic p, rho_d, rho_t, u, w, E_t, Q_ss, rho_r
# with T diagnosed from a univariate Newton retrieval and rho_v = Q_ss + rho_vs(T,p),
# rho_c = rho_t - rho_d - rho_v - rho_r recovered diagnostically.

@testset "moist_compressible (total energy)" begin

    import Springsteel.Thermodynamics: rho_v_sat, internal_energy_bf02, L_v,
        Rd, Rv, Cvd, Cvv, Cpd, Cpv, Cl, gravity

    # ──────────────────────────────────────────────
    # 1. Density-form saturation partials vs finite differences
    # ──────────────────────────────────────────────
    @testset "rho_v_sat partial derivatives" begin
        for (Tk, p_hPa) in ((300.0, 1000.0), (270.0, 700.0), (240.0, 400.0))
            hT = 1.0e-3
            fd_T = (rho_v_sat(Tk + hT, p_hPa) - rho_v_sat(Tk - hT, p_hPa)) / (2.0 * hT)
            @test Scythe.drho_vsat_dT(Tk, p_hPa) ≈ fd_T rtol = 1e-6

            hp_Pa = 10.0
            fd_p = (rho_v_sat(Tk, (100.0 * p_hPa + hp_Pa) / 100.0) -
                    rho_v_sat(Tk, (100.0 * p_hPa - hp_Pa) / 100.0)) / (2.0 * hp_Pa)
            @test Scythe.drho_vsat_dp(Tk, p_hPa) ≈ fd_p rtol = 1e-6
            @test Scythe.drho_vsat_dT(Tk, p_hPa) > 0.0
            @test Scythe.drho_vsat_dp(Tk, p_hPa) > 0.0    # Buck enhancement factor only
        end
    end

    # ──────────────────────────────────────────────
    # 2. Temperature retrieval round-trips forward-built states
    # ──────────────────────────────────────────────
    # Build a state from (T, p, saturation fraction, q_l, winds, z), compute the
    # prognostic set (p, rho_d, rho_t, E_t, Q_ss), and invert for T.
    function forward_state(Tk, p_Pa, sat_frac, q_l, u, w, z)
        rho_v = sat_frac * rho_v_sat(Tk, p_Pa / 100.0)
        rho_d = (p_Pa - Rv * Tk * rho_v) / (Rd * Tk)     # EOS with the chosen vapor
        rho_c = q_l * rho_d
        rho_t = rho_d + rho_v + rho_c
        q_v = rho_v / rho_d
        ke = 0.5 * (u^2 + w^2)
        E_t = rho_d * internal_energy_bf02(Tk, q_v, rho_c / rho_d) +
              rho_t * (ke + gravity * z)
        Q_ss = rho_v - rho_v_sat(Tk, p_Pa / 100.0)
        M = p_Pa + E_t - rho_t * (ke + gravity * z)
        return (; M, rho_d, rho_t, Q_ss, p_Pa, rho_v, rho_c)
    end

    @testset "retrieve_temperature round-trip" begin
        cases = (
            (Tk=300.0, p=100000.0, sat=0.0, q_l=0.0,    u=0.0,  w=0.0,  z=0.0),     # dry
            (Tk=285.0, p=90000.0,  sat=1.0, q_l=1.0e-3, u=0.0,  w=0.0,  z=1000.0),  # saturated cloudy
            (Tk=230.0, p=40000.0,  sat=0.5, q_l=0.0,    u=0.0,  w=0.0,  z=8000.0),  # cold aloft
            (Tk=295.0, p=95000.0,  sat=0.9, q_l=5.0e-4, u=15.0, w=5.0,  z=500.0),   # windy
        )
        for c in cases
            st = forward_state(c.Tk, c.p, c.sat, c.q_l, c.u, c.w, c.z)
            for guess in (c.Tk - 20.0, c.Tk + 20.0, 273.0)
                Tret = Scythe.retrieve_temperature(st.M, st.rho_d, st.rho_t, st.Q_ss,
                                                   st.p_Pa, guess)
                @test Tret ≈ c.Tk rtol = 1e-8
            end
            # Diagnostic recovery of the water partition
            Tret = Scythe.retrieve_temperature(st.M, st.rho_d, st.rho_t, st.Q_ss,
                                               st.p_Pa, c.Tk)
            rho_v = st.Q_ss + rho_v_sat(Tret, st.p_Pa / 100.0)
            rho_c = st.rho_t - st.rho_d - rho_v
            @test rho_v ≈ st.rho_v rtol = 1e-8 atol = 1e-14
            @test rho_c ≈ st.rho_c rtol = 1e-6 atol = 1e-10
        end

        # F(T) is monotone increasing over the physical range for a saturated state
        st = forward_state(285.0, 90000.0, 1.0, 1.0e-3, 0.0, 0.0, 1000.0)
        F(T) = (st.rho_d * Cpd + (st.rho_t - st.rho_d) * Cpv) * T +
               (st.Q_ss + st.rho_d + rho_v_sat(T, st.p_Pa / 100.0) - st.rho_t) * L_v(T) -
               st.M
        Ts = 200.0:5.0:330.0
        @test all(diff(F.(Ts)) .> 0.0)
    end

    # ──────────────────────────────────────────────
    # 3. Q_s identity: condensation-induced (dT, dp) reproduce the rho_vs change
    # ──────────────────────────────────────────────
    # This test locks the SIGN of the pressure-equation condensation coefficient:
    # dp|cond = (R_m/C_vt)*(L_v − R_v*C_pt*T/R_m)*delta (minus, from the corrected TeX).
    @testset "Q_s energy-consistent psychrometric identity" begin
        for (Tk, p_Pa, q_v, q_l) in ((285.0, 90000.0, 0.010, 1.0e-3),
                                     (300.0, 100000.0, 0.020, 0.0),
                                     (250.0, 50000.0, 0.001, 5.0e-4))
            rho_d = p_Pa / ((Rd + q_v * Rv) * Tk)
            C_vt = Cvd + q_v * Cvv + q_l * Cl
            R_m = Rd + q_v * Rv
            C_pt = C_vt + R_m
            delta = 1.0e-7                               # condensed mass [kg/m^3]
            dT = (L_v(Tk) - Rv * Tk) * delta / (rho_d * C_vt)
            dp = (R_m / C_vt) * (L_v(Tk) - Rv * C_pt * Tk / R_m) * delta
            drho_vs = rho_v_sat(Tk + dT, (p_Pa + dp) / 100.0) - rho_v_sat(Tk, p_Pa / 100.0)
            Q_s = Scythe.Q_s_energy(Tk, p_Pa, rho_d, q_v, q_l)
            @test drho_vs ≈ Q_s * delta rtol = 1e-4
            @test Q_s > 0.0
        end
    end

    # ──────────────────────────────────────────────
    # 4. Condensation rate limiter
    # ──────────────────────────────────────────────
    @testset "qss_condensation_rate limits" begin
        Tk = 285.0; p_hPa = 900.0; ts = 0.1
        rho_vs = rho_v_sat(Tk, p_hPa)
        rho_d = (100.0 * p_hPa - Rv * Tk * rho_vs) / (Rd * Tk)
        Q_s = Scythe.Q_s_energy(Tk, 100.0 * p_hPa, rho_d, rho_vs / rho_d, 1.0e-3)

        # Subsaturated, cloud-free: no droplets to evaporate -> zero
        @test Scythe.qss_condensation_rate(-0.5 * rho_vs, 0.5 * rho_vs, 0.0, rho_d, Tk,
                                           p_hPa, Q_s, ts) == 0.0
        # Strongly subsaturated with a little cloud: evaporation clamped by rho_c/ts
        rho_c = 1.0e-6 * rho_d
        rate = Scythe.qss_condensation_rate(-0.5 * rho_vs, 0.5 * rho_vs, rho_c, rho_d, Tk,
                                            p_hPa, Q_s, ts)
        @test rate ≈ -rho_c / ts
        # Mildly subsaturated with plenty of cloud: physical evaporation, not clamped
        rho_c = 2.0e-3 * rho_d
        rate = Scythe.qss_condensation_rate(-1.0e-4 * rho_vs, (1.0 - 1.0e-4) * rho_vs, rho_c,
                                            rho_d, Tk, p_hPa, Q_s, ts)
        @test -rho_c / ts < rate < 0.0
        # Supersaturated with cloud: condensation
        @test Scythe.qss_condensation_rate(1.0e-3 * rho_vs, (1.0 + 1.0e-3) * rho_vs, rho_c,
                                           rho_d, Tk, p_hPa, Q_s, ts) > 0.0
        # Supersaturated, cloud-free: Twomey nucleation kicks in
        @test Scythe.qss_condensation_rate(1.0e-3 * rho_vs, (1.0 + 1.0e-3) * rho_vs, 0.0,
                                           rho_d, Tk, p_hPa, Q_s, ts) > 0.0
        # Nearly saturated, cloud-free (below nucleation threshold): zero
        @test Scythe.qss_condensation_rate(1.0e-5 * rho_vs, (1.0 + 1.0e-5) * rho_vs, 0.0,
                                           rho_d, Tk, p_hPa, Q_s, ts) == 0.0
        # Dry air with spurious positive Q_ss drift (no actual vapor): the clamped
        # rho_v = 0 kills phantom condensation entirely
        @test Scythe.qss_condensation_rate(0.5 * rho_vs, 0.0, 0.0, rho_d, Tk, p_hPa,
                                           Q_s, ts) == 0.0
    end

    @testset "qss_condensation_rates cloud/rain split" begin
        Tk = 285.0; p_hPa = 900.0; ts = 0.1
        N_r = 1.0e-3   # #/cm^3
        rho_vs = rho_v_sat(Tk, p_hPa)
        rho_d = (100.0 * p_hPa - Rv * Tk * rho_vs) / (Rd * Tk)
        Q_s = Scythe.Q_s_energy(Tk, 100.0 * p_hPa, rho_d, rho_vs / rho_d, 1.0e-3)

        # With no rain (or no rain channel), the cloud rate is BIT-identical to the
        # single-category closure across the branch space.
        for (Q_ss, rho_v, rho_c) in (
                (-0.5 * rho_vs, 0.5 * rho_vs, 0.0),
                (-0.5 * rho_vs, 0.5 * rho_vs, 1.0e-6 * rho_d),
                (-1.0e-4 * rho_vs, (1.0 - 1.0e-4) * rho_vs, 2.0e-3 * rho_d),
                (1.0e-3 * rho_vs, (1.0 + 1.0e-3) * rho_vs, 2.0e-3 * rho_d),
                (1.0e-3 * rho_vs, (1.0 + 1.0e-3) * rho_vs, 0.0),
                (1.0e-5 * rho_vs, (1.0 + 1.0e-5) * rho_vs, 0.0),
                (0.5 * rho_vs, 0.0, 0.0))
            old = Scythe.qss_condensation_rate(Q_ss, rho_v, rho_c, rho_d, Tk, p_hPa, Q_s, ts)
            c0, r0 = Scythe.qss_condensation_rates(Q_ss, rho_v, rho_c, 0.0, rho_d, Tk,
                                                   p_hPa, Q_s, ts, N_r)
            @test c0 === old
            @test r0 === 0.0
            cN, rN = Scythe.qss_condensation_rates(Q_ss, rho_v, rho_c, 1.0e-3, rho_d, Tk,
                                                   p_hPa, Q_s, ts, 0.0)
            @test cN === old
            @test rN === 0.0
        end

        # Seamless split: with cloud AND rain, the unlimited rates divide in proportion
        # to the channel timescales, both directions.
        rho_c = 2.0e-3 * rho_d
        rho_r = 1.0e-3
        q_c = rho_c / rho_d
        invtau_c = Scythe.invtau_condensation(Tk, p_hPa, 100.0,
                                              Scythe.cloud_droplet_radius(100.0, q_c, rho_d))
        invtau_r = Scythe.invtau_rain(Tk, p_hPa, N_r, rho_r)
        for Q_ss in (1.0e-3 * rho_vs, -1.0e-4 * rho_vs)
            rho_v = rho_vs + Q_ss
            Qc, Qr = Scythe.qss_condensation_rates(Q_ss, rho_v, rho_c, rho_r, rho_d, Tk,
                                                   p_hPa, Q_s, ts, N_r)
            @test sign(Qc) == sign(Q_ss)
            @test sign(Qr) == sign(Q_ss)
            @test Qc / Qr ≈ invtau_c / invtau_r
            @test Qc + Qr ≈ Q_ss * (invtau_c + invtau_r) / (1.0 + Q_s)
        end

        # Rain evaporates in subsaturated cloud-free air (the O01 Qevap analogue). The
        # relaxation timescale is ~10 min, so a model step never clamps; the physical
        # rate is negative and bounded by the available rain.
        Qc, Qr = Scythe.qss_condensation_rates(-0.5 * rho_vs, 0.5 * rho_vs, 0.0, 1.0e-3,
                                               rho_d, Tk, p_hPa, Q_s, ts, N_r)
        @test Qc == 0.0
        @test -1.0e-3 / ts < Qr < 0.0
        @test Qr ≈ -0.5 * rho_vs * Scythe.invtau_rain(Tk, p_hPa, N_r, 1.0e-3) / (1.0 + Q_s)
        # With a long enough step the rho_r/ts clamp engages: no negative rain
        ts_long = 1.0e6
        Qc, Qr = Scythe.qss_condensation_rates(-0.5 * rho_vs, 0.5 * rho_vs, 0.0, 1.0e-3,
                                               rho_d, Tk, p_hPa, Q_s, ts_long, N_r)
        @test Qc == 0.0
        @test Qr ≈ -1.0e-3 / ts_long

        # No negative rain: evaporation never exceeds the available rho_r even when
        # cloud has more to give.
        Qc, Qr = Scythe.qss_condensation_rates(-0.5 * rho_vs, 0.5 * rho_vs, rho_c, 1.0e-9,
                                               rho_d, Tk, p_hPa, Q_s, ts, N_r)
        @test Qr == 0.0                # below RHO_R_MIN: channel inactive
        Qc, Qr = Scythe.qss_condensation_rates(-0.5 * rho_vs, 0.5 * rho_vs, rho_c, 2.0e-8,
                                               rho_d, Tk, p_hPa, Q_s, ts, N_r)
        @test Qr >= -2.0e-8 / ts

        # Vapor cap rescales both channels proportionally: tiny vapor, strong drive
        rho_v_tiny = 1.0e-9
        Qc, Qr = Scythe.qss_condensation_rates(0.5 * rho_vs, rho_v_tiny, rho_c, rho_r,
                                               rho_d, Tk, p_hPa, Q_s, ts, N_r)
        @test Qc + Qr ≈ rho_v_tiny / ts
        @test Qc / Qr ≈ invtau_c / invtau_r
    end

    @testset "rain condensation is gated on cloud presence" begin
        # The physical pathway to rain is condensation -> cloud -> autoconversion:
        # direct vapor deposition onto rain in CLOUD-FREE air is unphysically fast
        # under the monodisperse fixed-N_r closure (rate ∝ rho_r^{1/3}, non-Lipschitz
        # at zero), and is exactly the O01 spurious-blob pathway (ringing-seeded rain
        # growing in wave-driven supersaturation at the lid). The rain channel must
        # therefore be inert for CONDENSATION unless cloud coexists; EVAPORATION in
        # subsaturated air stays unconditional.
        Tk = 285.0; p_hPa = 900.0; ts = 0.1
        N_r = 1.0e-3   # #/cm^3
        rho_vs = rho_v_sat(Tk, p_hPa)
        rho_d = (100.0 * p_hPa - Rv * Tk * rho_vs) / (Rd * Tk)
        Q_s = Scythe.Q_s_energy(Tk, 100.0 * p_hPa, rho_d, rho_vs / rho_d, 1.0e-3)
        rho_r = 1.0e-6   # well above RHO_R_MIN

        # Supersaturated cloud-free air with rain present: NO condensation onto rain;
        # nucleation routes the full rate to the cloud channel.
        Q_ss = 1.0e-3 * rho_vs
        Qc, Qr = Scythe.qss_condensation_rates(Q_ss, rho_vs + Q_ss, 0.0, rho_r,
                                               rho_d, Tk, p_hPa, Q_s, ts, N_r)
        @test Qr === 0.0
        @test Qc > 0.0

        # Below the nucleation threshold, cloud-free + rain: nothing condenses at all
        Q_ss = 1.0e-5 * rho_vs
        Qc, Qr = Scythe.qss_condensation_rates(Q_ss, rho_vs + Q_ss, 0.0, rho_r,
                                               rho_d, Tk, p_hPa, Q_s, ts, N_r)
        @test Qc === 0.0
        @test Qr === 0.0

        # Trace cloud below the q_c = 1e-8 existence threshold counts as cloud-free
        Q_ss = 1.0e-3 * rho_vs
        Qc, Qr = Scythe.qss_condensation_rates(Q_ss, rho_vs + Q_ss, 0.5e-8 * rho_d, rho_r,
                                               rho_d, Tk, p_hPa, Q_s, ts, N_r)
        @test Qr === 0.0
        @test Qc > 0.0

        # Cloudy air: the split is un-gated (proportional rates, both channels)
        rho_c = 2.0e-3 * rho_d
        invtau_c = Scythe.invtau_condensation(Tk, p_hPa, 100.0,
                                              Scythe.cloud_droplet_radius(100.0, rho_c / rho_d, rho_d))
        invtau_r = Scythe.invtau_rain(Tk, p_hPa, N_r, rho_r)
        Qc, Qr = Scythe.qss_condensation_rates(Q_ss, rho_vs + Q_ss, rho_c, rho_r,
                                               rho_d, Tk, p_hPa, Q_s, ts, N_r)
        @test Qr > 0.0
        @test Qc / Qr ≈ invtau_c / invtau_r

        # Subsaturated cloud-free rain evaporation is NOT gated
        Qc, Qr = Scythe.qss_condensation_rates(-0.5 * rho_vs, 0.5 * rho_vs, 0.0, rho_r,
                                               rho_d, Tk, p_hPa, Q_s, ts, N_r)
        @test Qc == 0.0
        @test Qr < 0.0
    end

    # ──────────────────────────────────────────────
    # 5. Integration: equation set on a ModelTile
    # ──────────────────────────────────────────────

    using SparseArrays

    """Saturated cloudy column exactly on the Q_ss = 0 manifold (density form)."""
    function saturated_cloudy_column_mc(z; q_l=1.0e-3)
        n = length(z)
        Tk = @. 290.0 - 0.005 * z
        p_Pa = @. 90000.0 * exp(-z / 8000.0)
        rho_v = rho_v_sat.(Tk, p_Pa ./ 100.0)
        rho_d = (p_Pa .- (Rv .* Tk .* rho_v)) ./ (Rd .* Tk)
        rho_c = q_l .* rho_d
        return (; z, Tk, p_Pa, rho_d, rho_v, rho_c)
    end

    """Dry, neutrally stable (theta = 300 K) analytic adiabat -- the Straka/BF02 base."""
    function dry_adiabatic_column_mc(z; theta0=300.0)
        n = length(z)
        exner = @. 1.0 - (Scythe.gravity * z) / (Cpd * theta0)
        Tk = theta0 .* exner
        p_Pa = @. 100000.0 * exner^(Cpd / Rd)
        rho_d = p_Pa ./ (Rd .* Tk)
        return (; z, Tk, p_Pa, rho_d, rho_v = zeros(n), rho_c = zeros(n))
    end

    function make_mc_mtile(tmpdir; num_cells=8, kDim=16, semiimplicit=false, ts=0.1,
                           dry=false, Khdiff=0.0, Kvdiff=0.0, Kvdiff_heat=nothing,
                           Kvdiff_water=0.0, tau_qss=10.0,
                           u_side_bc=DirichletBC(), precipitation=false, N_r=1.0e-3,
                           q_l=1.0e-3, alpha=0.0, z_damp=20.0e3)
        varlist = Scythe.MC_VARS
        vars = Dict(v => i for (i, v) in enumerate(varlist))
        scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
        wall_bc = merge(scalar_bc, Dict("u" => DirichletBC(), "w" => DirichletBC()))
        # A z-only u profile is only representable when the side walls let u be nonzero;
        # with a Dirichlet u the spline fit forces u -> 0 at x = 0, L and u_x (hence the
        # divergence) swamps any diffusive tendency.
        side_bc = merge(scalar_bc, Dict("u" => u_side_bc, "w" => DirichletBC()))
        gp = GridParameters(
            geometry = "RZ", num_cells = num_cells,
            iMin = 0.0, iMax = 2000.0, kMin = 0.0, kMax = 2000.0, kDim = kDim,
            BCL = side_bc, BCR = side_bc, BCB = wall_bc, BCT = wall_bc, vars = vars,
        )
        patch = createGrid(gp)
        gridpoints = Scythe.getGridpoints(patch)
        z = gridpoints[1:kDim, 2]
        col = dry ? dry_adiabatic_column_mc(z) : saturated_cloudy_column_mc(z; q_l)
        ref_file = joinpath(tmpdir, "mc_pressure.ref")
        Scythe.write_exact_ref_mc(ref_file, z, col.p_Pa, col.rho_d, col.rho_v, col.rho_c)
        model = ModelParameters(
            ts = ts, integration_time = 1.0, output_interval = 1.0,
            equation_set = "moist_compressible_XZ",
            ref_state_file = ref_file, grid_params = gp,
            physical_params = Dict(:Khdiff => Khdiff, :Kvdiff => Kvdiff,
                                   :Kvdiff_heat => (Kvdiff_heat === nothing ? Kvdiff : Kvdiff_heat),
                                   :Kvdiff_water => Kvdiff_water,
                                   :Kv_mudiff => 0.0, :tau_qss => tau_qss, :N_r => N_r,
                                   :alpha => alpha, :z_damp => z_damp),
            options = Dict(:semiimplicit => semiimplicit, :exact_reference_state => true,
                           :precipitation => precipitation, :vertical_mixing => false),
        )
        patch.physical .= 0.0
        spectralTransform!(patch)
        gridTransform!(patch)
        haloReceiveMap = sparse(Int64[], Int64[], Float64[],
                                size(patch.spectral, 1), size(patch.spectral, 2))
        mtile = createModelTile(patch, patch, model, haloReceiveMap)
        return mtile, patch, model, col
    end

    """Advance every column of `mtile` for `nsteps` steps, refreshing the transforms."""
    function step_mc!(mtile, patch, model, nsteps)
        kDim = model.grid_params.kDim
        ncols = div(size(patch.physical, 1), kDim)
        for t in 1:nsteps
            for c in 1:ncols
                Scythe.advance_column(mtile, c, t)
            end
            Scythe.calcTendency(mtile)
            gridTransform!(patch)
        end
        return patch
    end

    @testset "reference routing" begin
        mktempdir() do tmpdir
            mtile, patch, model, col = make_mc_mtile(tmpdir)
            rs = mtile.ref_state
            @test rs isa Springsteel.PressureReferenceState
            @test 250.0 < sqrt(Springsteel.sound_speed_sq(rs)) < 400.0
            # Saturated base: Q_ssbar = 0 identically before smoothing
            @test maximum(abs.(Springsteel.ref_qss(rs)[:, 1])) < 1e-10
        end
    end

    @testset "resting cloudy base preserved" begin
        mktempdir() do tmpdir
            mtile, patch, model, col = make_mc_mtile(tmpdir)
            kDim = model.grid_params.kDim
            ncols = div(size(patch.physical, 1), kDim)
            for c in 1:ncols
                Scythe.advance_column(mtile, c, 1)
            end
            # Per-slot tendency tolerances scaled to the slot magnitudes
            # (p ~ 1e5 Pa, E_t ~ 2e8 J/m^3, densities ~ 1)
            scales = Dict(1 => 1.0e5, 2 => 1.0, 3 => 1.0, 4 => 1.0, 5 => 1.0,
                          6 => 2.0e8, 7 => 1.0e-2, 8 => 1.0)
            for v in 1:8
                @test maximum(abs.(mtile.expdot_n[:, v])) / scales[v] < 1.0e-9
                @test maximum(abs.(mtile.var_np1[:, v])) / scales[v] < 1.0e-9
            end
            @test all(isfinite.(mtile.var_np1))
        end
    end

    @testset "condensation closure at rest" begin
        mktempdir() do tmpdir
            mtile, patch, model, col = make_mc_mtile(tmpdir)
            kDim = model.grid_params.kDim
            vars = model.grid_params.vars
            qss_i = vars["Q_ss"]

            # Supersaturate the lower half of every column by a small, realistic amount
            rho_dbar = Springsteel.ref_rho_d(mtile.ref_state)[:, 1]
            dq = 5.0e-5
            npts = size(patch.physical, 1)
            for i in 1:npts
                k = mod1(i, kDim)
                if k <= div(kDim, 2)
                    patch.physical[i, qss_i, 1] = rho_dbar[k] * dq
                end
            end
            spectralTransform!(patch)
            gridTransform!(patch)

            ncols = div(npts, kDim)
            for c in 1:ncols
                Scythe.advance_column(mtile, c, 1)
            end

            perturbed = [i for i in 1:npts if mod1(i, kDim) <= div(kDim, 2)]
            # Exact first law: no condensation source in E_t or rho_t at rest
            @test maximum(abs.(mtile.expdot_n[:, vars["E_t"]])) == 0.0
            @test maximum(abs.(mtile.expdot_n[:, vars["rho_t"]])) == 0.0
            @test maximum(abs.(mtile.expdot_n[:, vars["rho_d"]])) == 0.0
            # Latent heating raises pressure; supersaturation relaxes
            @test all(mtile.expdot_n[perturbed, vars["p"]] .> 0.0)
            @test all(mtile.expdot_n[perturbed, qss_i] .< 0.0)
            @test all(isfinite.(mtile.var_np1))
        end
    end

    @testset "post-step retrieval physical" begin
        mktempdir() do tmpdir
            mtile, patch, model, col = make_mc_mtile(tmpdir)
            kDim = model.grid_params.kDim
            vars = model.grid_params.vars
            # Small warm perturbation via E_t' in the lower half
            E_tbar = Springsteel.ref_total_energy(mtile.ref_state)[:, 1]
            npts = size(patch.physical, 1)
            for i in 1:npts
                k = mod1(i, kDim)
                if k <= div(kDim, 2)
                    patch.physical[i, vars["E_t"], 1] = 1.0e-4 * E_tbar[k]
                end
            end
            spectralTransform!(patch)
            gridTransform!(patch)
            ncols = div(npts, kDim)
            for c in 1:ncols
                Scythe.advance_column(mtile, c, 1)
            end
            # Re-retrieve T from the advanced state
            rs = mtile.ref_state
            pbar = Springsteel.ref_pressure(rs)[:, 1]
            rho_dbar = Springsteel.ref_rho_d(rs)[:, 1]
            rho_tbar = Springsteel.ref_rho_t(rs)[:, 1]
            Q_ssbar = Springsteel.ref_qss(rs)[:, 1]
            Tbar = Springsteel.reference_temperature(rs)
            z = Scythe.getGridpoints(patch)[1:kDim, 2]
            for i in 1:npts
                k = mod1(i, kDim)
                v = mtile.var_np1
                p = v[i, vars["p"]] + pbar[k]
                rho_d = v[i, vars["rho_d"]] + rho_dbar[k]
                rho_t = v[i, vars["rho_t"]] + rho_tbar[k]
                E_t = v[i, vars["E_t"]] + E_tbar[k]
                Q_ss = v[i, vars["Q_ss"]] + Q_ssbar[k]
                ke = 0.5 * (v[i, vars["u"]]^2 + v[i, vars["w"]]^2)
                M = p + E_t - rho_t * (ke + Scythe.gravity * z[k])
                Tk = Scythe.retrieve_temperature(M, rho_d, rho_t, Q_ss, p, Tbar[k])
                @test isfinite(Tk) && 200.0 < Tk < 320.0
            end
        end
    end

    @testset "semi-implicit acoustic stability" begin
        mktempdir() do tmpdir
            # ts = 0.1 s is ~7x the explicit VERTICAL acoustic limit for the
            # boundary-clustered Chebyshev levels (kDim=32: dz_min ~ 5 m, c ~ 340 m/s
            # => ~0.015 s; the fully explicit scheme NaNs by step ~19 at this ts) while
            # staying below the HORIZONTAL limit (dx ~ 80 m), since the semi-implicit
            # adjustment is vertical-only.
            mtile, patch, model, col = make_mc_mtile(tmpdir; semiimplicit=true,
                                                     ts=0.1, kDim=32)
            kDim = model.grid_params.kDim
            vars = model.grid_params.vars
            p_i = vars["p"]; rhot_i = vars["rho_t"]
            npts = size(patch.physical, 1)
            gridpoints = Scythe.getGridpoints(patch)

            # Small pressure pulse in the domain center
            for i in 1:npts
                x = gridpoints[i, 1]; z = gridpoints[i, 2]
                L = sqrt(((x - 1000.0) / 500.0)^2 + ((z - 1000.0) / 500.0)^2)
                patch.physical[i, p_i, 1] = L <= 1.0 ? 10.0 * (cos(pi * L / 2.0))^2 : 0.0
            end
            spectralTransform!(patch)
            gridTransform!(patch)

            p0_max = maximum(abs.(patch.physical[:, p_i, 1]))
            ncols = div(npts, kDim)
            for t in 1:200
                for c in 1:ncols
                    Scythe.advance_column(mtile, c, t)
                end
                Scythe.calcTendency(mtile)
                gridTransform!(patch)
            end
            @test all(isfinite.(patch.physical[:, :, 1]))
            # Acoustic energy must not grow: the dispersing pulse DECAYS below its
            # initial amplitude after 200 vertically-stiff steps
            @test maximum(abs.(patch.physical[:, p_i, 1])) < p0_max
            @test maximum(abs.(patch.physical[:, vars["w"], 1])) < 1.0
        end
    end

    # ──────────────────────────────────────────────
    # 6. Diffusion: theta_d, heating consistency, dissipation
    # ──────────────────────────────────────────────

    @testset "potential_temperature(p_Pa, rho_d)" begin
        # Dry air: exactly Straka's theta = T (p_0/p)^kappa
        for (Tk, p_Pa) in ((300.0, 100000.0), (280.0, 85000.0), (250.0, 50000.0))
            rho_d = p_Pa / (Rd * Tk)
            theta = Tk * ((100.0 * Scythe.p_0) / p_Pa)^(Rd / Cpd)
            @test Scythe.potential_temperature(p_Pa, rho_d) ≈ theta rtol=1e-14
        end
        # Moist air: (R_m/R_d) T (p_0/p)^kappa
        Tk, p_Pa, q_v = 295.0, 95000.0, 0.015
        R_m = Rd + (q_v * Rv)
        rho_d = p_Pa / (R_m * Tk)
        expected = (R_m / Rd) * Tk * ((100.0 * Scythe.p_0) / p_Pa)^(Rd / Cpd)
        @test Scythe.potential_temperature(p_Pa, rho_d) ≈ expected rtol=1e-14

        # The mc heating coefficient rho_d*C_vt*(Cpd/Cvd)*(T/theta_d) reduces to the
        # classical rho*C_p*pi in dry air, so Q_therm = rho*Cp*pi*K*Lap(theta) exactly.
        Tk, p_Pa = 290.0, 90000.0
        rho_d = p_Pa / (Rd * Tk)
        theta_d = Scythe.potential_temperature(p_Pa, rho_d)
        exner = (p_Pa / (100.0 * Scythe.p_0))^(Rd / Cpd)
        @test rho_d * Cvd * (Cpd / Cvd) * (Tk / theta_d) ≈ rho_d * Cpd * exner rtol=1e-12
    end

    @testset "qss admissible bounds and relaxation" begin
        Tk, p_hPa = 290.0, 900.0
        rho_vs = rho_v_sat(Tk, p_hPa)
        rho_d = 1.1
        tau = 10.0

        # Dry air: rho_t == rho_d, so the interval collapses to the point -rho_vs
        lo, hi = Scythe.qss_admissible_bounds(rho_d, rho_d, 0.0, rho_vs)
        @test lo == -rho_vs && hi == -rho_vs
        # ... and the relaxation drives Q_ss there from either side, at rate 1/tau
        @test Scythe.qss_relaxation(0.0, rho_d, rho_d, 0.0, rho_vs, tau) ≈ -rho_vs / tau
        @test Scythe.qss_relaxation(-2rho_vs, rho_d, rho_d, 0.0, rho_vs, tau) ≈ rho_vs / tau

        # Cloudy air: Q_ss strictly interior => exactly zero, no nudge at all
        rho_t = rho_d + rho_vs + 1.0e-3          # 1 g/m^3 of cloud
        @test Scythe.qss_relaxation(0.0, rho_d, rho_t, 0.0, rho_vs, tau) == 0.0
        @test Scythe.qss_relaxation(1.0e-4, rho_d, rho_t, 0.0, rho_vs, tau) == 0.0

        # Supersaturated cloud-free air sits exactly at the ceiling Q_hi > 0: the
        # relaxation is zero there, so nucleation is not suppressed.
        rho_w = 1.05 * rho_vs
        rho_t = rho_d + rho_w
        _, Q_hi = Scythe.qss_admissible_bounds(rho_d, rho_t, 0.0, rho_vs)
        @test Q_hi > 0.0
        @test Scythe.qss_relaxation(Q_hi, rho_d, rho_t, 0.0, rho_vs, tau) == 0.0
        Q_s = Scythe.Q_s_energy(Tk, 100.0 * p_hPa, rho_d, rho_w / rho_d, 0.0)
        @test Scythe.qss_condensation_rate(Q_hi, rho_w, 0.0, rho_d, Tk, p_hPa, Q_s, 0.1) > 0.0
        # Drift above the ceiling is pulled back
        @test Scythe.qss_relaxation(Q_hi + 0.02, rho_d, rho_t, 0.0, rho_vs, tau) ≈ -0.02 / tau

        # Rain is liquid: the ceiling excludes it
        rho_r = 5.0e-4
        rho_t = rho_d + rho_vs + rho_r
        _, Q_hi_rain = Scythe.qss_admissible_bounds(rho_d, rho_t, rho_r, rho_vs)
        @test Q_hi_rain ≈ 0.0 atol=1e-15
    end

    @testset "retrieval clamp keeps rho_c nonnegative with rain" begin
        # Total water is all rain: vapor must clamp to zero, not to rho_w, or the
        # residual cloud rho_c = rho_t - rho_d - rho_v - rho_r goes negative.
        Tk, p_Pa = 290.0, 90000.0
        rho_d = p_Pa / (Rd * Tk)
        rho_r = 1.0e-3
        rho_t = rho_d + rho_r
        rho_vs = rho_v_sat(Tk, p_Pa / 100.0)
        Q_ss = -rho_vs                                   # rho_v = 0
        M = (rho_d * Cpd * Tk) + (-rho_r * Scythe.L_v(Tk))
        T_ret = Scythe.retrieve_temperature(M + p_Pa - p_Pa, rho_d, rho_t, Q_ss, p_Pa,
                                            Tk, rho_r)
        rho_v = clamp(Q_ss + rho_v_sat(T_ret, p_Pa / 100.0), 0.0,
                      max(rho_t - rho_d - rho_r, 0.0))
        @test rho_v ≈ 0.0 atol=1e-14
        @test rho_t - rho_d - rho_v - rho_r >= -1e-14    # rho_c >= 0
    end

    @testset "resting dry base is untouched by diffusion" begin
        # s_d' = 0 identically on the reference (s_d = C_vd ln p - C_pd ln rho_d is an
        # explicit function of p, rho_d, so at rest it equals s_dbar bit-for-bit), so every
        # diffusive tendency must vanish. A K = 75 run has to be BIT-IDENTICAL to a K = 0
        # run at rest: this catches a bad s_dbar subtraction that would cook the base.
        mktempdir() do tmpdir
            m0, p0, mod0, _ = make_mc_mtile(tmpdir; dry=true, Kvdiff=0.0, Khdiff=0.0)
            step_mc!(m0, p0, mod0, 3)
        end
        mktempdir() do tmpdir
            mK, pK, modK, _ = make_mc_mtile(tmpdir; dry=true, Kvdiff=75.0, Khdiff=75.0)
            step_mc!(mK, pK, modK, 3)
            # Nothing was seeded, so the base must stay exactly at zero perturbation
            @test maximum(abs.(pK.physical[:, 4, 1])) == 0.0   # u
            @test maximum(abs.(pK.physical[:, 5, 1])) == 0.0   # w
            @test maximum(abs.(pK.physical[:, 1, 1])) == 0.0   # p'
            @test maximum(abs.(pK.physical[:, 6, 1])) == 0.0   # E_t'
            @test all(isfinite.(pK.physical[:, :, 1]))
        end
    end

    @testset "diffusive heating sources p and E_t consistently" begin
        # In dry air the retrieval Jacobian is F_T = rho_d*C_pt, so a heating Qdot that
        # sources dE_t = Qdot and dp = (R_m/C_vt)*Qdot yields dT = dp/(rho_d*R_m) exactly.
        # Diffing a Kvdiff = 75 step against a Kvdiff = 0 step isolates the split from
        # the O(ts^2) truncation error of the rest of the scheme. Khdiff = 0 so the two
        # runs share an identical expdot.
        mktempdir() do tmpdir
            args = (; dry=true, Khdiff=0.0, kDim=16, num_cells=8)
            m0, patch0, mod0, _ = make_mc_mtile(tmpdir; args..., Kvdiff=0.0)
            gp0 = Scythe.getGridpoints(patch0)
            ref0 = m0.ref_state
            Scythe.theta_bubble_mc!(patch0, gp0, ref0;
                                    xc=1000.0, xr=400.0, zc=1000.0, zr=400.0, dtheta_max=2.0)
            spectralTransform!(patch0); gridTransform!(patch0)
            ncols = div(size(patch0.physical, 1), mod0.grid_params.kDim)
            for c in 1:ncols; Scythe.advance_column(m0, c, 1); end

            mK, patchK, modK, _ = make_mc_mtile(tmpdir; args..., Kvdiff=75.0)
            gpK = Scythe.getGridpoints(patchK)
            Scythe.theta_bubble_mc!(patchK, gpK, mK.ref_state;
                                    xc=1000.0, xr=400.0, zc=1000.0, zr=400.0, dtheta_max=2.0)
            spectralTransform!(patchK); gridTransform!(patchK)
            for c in 1:ncols; Scythe.advance_column(mK, c, 1); end

            kDim = mod0.grid_params.kDim
            pbar = Springsteel.ref_pressure(ref0)[:, 1]
            rho_dbar = Springsteel.ref_rho_d(ref0)[:, 1]
            rho_tbar = Springsteel.ref_rho_t(ref0)[:, 1]
            E_tbar = Springsteel.ref_total_energy(ref0)[:, 1]
            Q_ssbar = Springsteel.ref_qss(ref0)[:, 1]
            Tbar = Springsteel.reference_temperature(ref0)
            zs = gp0[:, 2]

            function retrieved(mt, i, k)
                p = mt.var_np1[i, 1] + pbar[k]
                rho_d = mt.var_np1[i, 2] + rho_dbar[k]
                rho_t = mt.var_np1[i, 3] + rho_tbar[k]
                E_t = mt.var_np1[i, 6] + E_tbar[k]
                Q_ss = mt.var_np1[i, 7] + Q_ssbar[k]
                ke = 0.5 * (mt.var_np1[i, 4]^2 + mt.var_np1[i, 5]^2)
                M = p + E_t - rho_t * (ke + Scythe.gravity * zs[i])
                return Scythe.retrieve_temperature(M, rho_d, rho_t, Q_ss, p, Tbar[k],
                                                   mt.var_np1[i, 8]), p, rho_d
            end

            dp_max = 0.0
            worst = 0.0
            for i in 1:size(patch0.physical, 1)
                k = mod1(i, kDim)
                T0, p0v, rho_d0 = retrieved(m0, i, k)
                TK, pKv, rho_dK = retrieved(mK, i, k)
                dp = pKv - p0v
                dp_max = max(dp_max, abs(dp))
                # Dry air: R_m = Rd, and rho_d is untouched by the diffusion split
                @test rho_dK ≈ rho_d0 rtol=1e-14
                dT_expected = dp / (rho_dK * Rd)
                worst = max(worst, abs((TK - T0) - dT_expected))
            end
            # The diffusion split actually did something...
            @test dp_max > 1.0e-6
            # ...and the retrieved temperature increment matches the EOS-slaved pressure
            # increment to machine precision. Sourcing p or E_t alone breaks this.
            @test worst < 1.0e-9
        end
    end

    @testset "momentum diffusion is a resolved-KE sink" begin
        # Eddy friction removes resolved KE to the subgrid (the future TKE shear
        # production), so E_t follows the KE DOWN and the internal energy is HELD: T and p
        # are unchanged (no dissipative heating — the review fix). The E_t decrease equals
        # the resolved KE removed, dE_t = rho_t*dke.
        mktempdir() do tmpdir
            mtile, patch, model, _ = make_mc_mtile(tmpdir; dry=true, Khdiff=0.0,
                                                   Kvdiff=75.0, kDim=16,
                                                   u_side_bc=NeumannBC())
            kDim = model.grid_params.kDim
            gridpoints = Scythe.getGridpoints(patch)
            rho_tbar = Springsteel.ref_rho_t(mtile.ref_state)[:, 1]
            # u = U0 sin(pi z/H): vanishes at the no-slip lids, u_zz != 0 in the interior,
            # and is x-independent so the divergence (hence every other tendency) is zero.
            # E_t carries the kinetic energy in this set, so a consistent shear IC must
            # seed E_t' = rho_t*ke too — without it the retrieval sees a phantom cold
            # anomaly and the (now moist-entropy) heat path fires on it.
            for i in 1:size(patch.physical, 1)
                k = mod1(i, kDim)
                u0 = 10.0 * sin(pi * gridpoints[i, 2] / 2000.0)
                patch.physical[i, 4, 1] = u0
                patch.physical[i, 6, 1] = rho_tbar[k] * 0.5 * u0^2
            end
            spectralTransform!(patch); gridTransform!(patch)

            E_t_before = copy(patch.physical[:, 6, 1])
            p_before = copy(patch.physical[:, 1, 1])
            u_before = copy(patch.physical[:, 4, 1])
            ncols = div(size(patch.physical, 1), kDim)
            for c in 1:ncols; Scythe.advance_column(mtile, c, 1); end

            @test all(isfinite.(mtile.var_np1))
            dke = 0.5 .* (mtile.var_np1[:, 4] .^ 2 .+ mtile.var_np1[:, 5] .^ 2 .-
                          u_before .^ 2)
            # Net kinetic energy falls
            @test sum(dke) < 0.0

            sheared = findall(x -> x < -1.0e-9, dke)
            @test !isempty(sheared)
            for i in sheared
                k = mod1(i, kDim)
                # E_t follows the KE down (the sink), and p is untouched (internal energy
                # held — no frictional heating; the fit-residual s_t' leaves only a
                # negligible heat increment).
                @test isapprox(mtile.var_np1[i, 6] - E_t_before[i], rho_tbar[k] * dke[i];
                               rtol=1e-4, atol=1e-7)
                @test mtile.var_np1[i, 1] ≈ p_before[i] atol=1e-6
            end
        end
    end

    @testset "Prandtl/Schmidt parameters are rejected" begin
        # Eddy mixing coefficients are not molecular ratios: heat and water diffusivities
        # are specified directly (:Khdiff_heat/:Kvdiff_heat, :Khdiff_water/:Kvdiff_water).
        vars = Dict(v => i for (i, v) in enumerate(Scythe.MC_VARS))
        scalar_bc = Dict(v => NeumannBC() for v in keys(vars))
        gp = GridParameters(
            geometry = "RZ", num_cells = 4,
            iMin = 0.0, iMax = 1000.0, kMin = 0.0, kMax = 1000.0, kDim = 8,
            BCL = scalar_bc, BCR = scalar_bc, BCB = scalar_bc, BCT = scalar_bc, vars = vars,
        )
        for bad in (:Prandtl, :Schmidt)
            @test_throws ErrorException ModelParameters(
                ts = 0.1, equation_set = "moist_compressible_XZ", grid_params = gp,
                physical_params = Dict(:Khdiff => 0.0, :Kvdiff => 0.0, bad => 1.0))
        end
    end

    @testset "heat-only vertical diffusion leaves momentum untouched" begin
        # Kvdiff = 0, Kvdiff_heat > 0: the momentum solve must be skipped entirely (a K = 0
        # solve is not the identity — it refits and refilters the column), so u and w must
        # be BIT-identical to a no-diffusion run, while the heat path sources p/E_t/Q_ss.
        mktempdir() do tmpdir
            args = (; dry=true, Khdiff=0.0, kDim=16, num_cells=8)
            function run_bubble(; kwargs...)
                mtile, patch, model, _ = make_mc_mtile(tmpdir; args..., kwargs...)
                gp = Scythe.getGridpoints(patch)
                Scythe.theta_bubble_mc!(patch, gp, mtile.ref_state;
                                        xc=1000.0, xr=400.0, zc=1000.0, zr=400.0,
                                        dtheta_max=2.0)
                spectralTransform!(patch); gridTransform!(patch)
                ncols = div(size(patch.physical, 1), model.grid_params.kDim)
                for c in 1:ncols; Scythe.advance_column(mtile, c, 1); end
                return mtile
            end
            m0 = run_bubble(; Kvdiff=0.0)
            mH = run_bubble(; Kvdiff=0.0, Kvdiff_heat=75.0)
            @test mH.var_np1[:, 4] == m0.var_np1[:, 4]   # u bit-identical
            @test mH.var_np1[:, 5] == m0.var_np1[:, 5]   # w bit-identical
            # ...but the heat path actually did something to p, E_t and Q_ss
            @test maximum(abs.(mH.var_np1[:, 1] .- m0.var_np1[:, 1])) > 1.0e-6
            @test maximum(abs.(mH.var_np1[:, 6] .- m0.var_np1[:, 6])) > 1.0e-6
        end
    end

    @testset "momentum-only vertical diffusion leaves thermodynamics untouched" begin
        # Kvdiff > 0, Kvdiff_heat = 0: the heat solve must be skipped, so p and Q_ss are
        # BIT-identical to a no-diffusion run and E_t changes only by the resolved-KE sink
        # dE_visc = rho_t*dke. The x-independent shear IC keeps the divergence (and hence
        # every explicit tendency) identical between the runs.
        mktempdir() do tmpdir
            args = (; dry=true, Khdiff=0.0, kDim=16, num_cells=8, u_side_bc=NeumannBC())
            function run_shear(; kwargs...)
                mtile, patch, model, _ = make_mc_mtile(tmpdir; args..., kwargs...)
                gridpoints = Scythe.getGridpoints(patch)
                for i in 1:size(patch.physical, 1)
                    patch.physical[i, 4, 1] = 10.0 * sin(pi * gridpoints[i, 2] / 2000.0)
                end
                spectralTransform!(patch); gridTransform!(patch)
                ncols = div(size(patch.physical, 1), model.grid_params.kDim)
                for c in 1:ncols; Scythe.advance_column(mtile, c, 1); end
                return mtile
            end
            m0 = run_shear(; Kvdiff=0.0)
            mM = run_shear(; Kvdiff=75.0, Kvdiff_heat=0.0)
            @test mM.var_np1[:, 1] == m0.var_np1[:, 1]   # p bit-identical
            @test mM.var_np1[:, 7] == m0.var_np1[:, 7]   # Q_ss bit-identical
            # Momentum diffusion acted, and E_t moved by exactly the resolved-KE sink
            @test maximum(abs.(mM.var_np1[:, 4] .- m0.var_np1[:, 4])) > 1.0e-6
            kDim = 16
            rho_tbar = Springsteel.ref_rho_t(mM.ref_state)[:, 1]
            dke = 0.5 .* (mM.var_np1[:, 4] .^ 2 .+ mM.var_np1[:, 5] .^ 2 .-
                          m0.var_np1[:, 4] .^ 2 .- m0.var_np1[:, 5] .^ 2)
            for i in findall(x -> abs(x) > 1.0e-9, dke)
                k = mod1(i, kDim)
                @test (mM.var_np1[i, 6] - m0.var_np1[i, 6]) ≈ rho_tbar[k] * dke[i] rtol=1e-6
            end
        end
    end

    @testset "Q_ss relaxation is thermodynamically inert in dry air" begin
        # rho_w = 0 => rho_v == 0 regardless of Q_ss, so the relaxation cannot move T, p,
        # E_t or the densities. It must, however, pull the drifting Q_ss back to -rho_vs.
        mktempdir() do tmpdir
            function run_dry(qss_offset)
                mtile, patch, model, _ = make_mc_mtile(tmpdir; dry=true, tau_qss=10.0)
                gridpoints = Scythe.getGridpoints(patch)
                Scythe.theta_bubble_mc!(patch, gridpoints, mtile.ref_state;
                                        xc=1000.0, xr=400.0, zc=1000.0, zr=400.0,
                                        dtheta_max=2.0)
                patch.physical[:, 7, 1] .+= qss_offset
                spectralTransform!(patch); gridTransform!(patch)
                step_mc!(mtile, patch, model, 5)
                return copy(patch.physical[:, :, 1])
            end
            base = run_dry(0.0)
            drifted = run_dry(0.02)

            # Every slot except Q_ss is bit-identical
            for v in (1, 2, 3, 4, 5, 6, 8)
                @test base[:, v] == drifted[:, v]
            end
            # ... and the drift decays toward the base at the relaxation rate
            @test maximum(abs.(drifted[:, 7] .- base[:, 7])) < 0.02
            @test maximum(abs.(drifted[:, 7] .- base[:, 7])) > 0.0
        end
    end

    # ──────────────────────────────────────────────
    # 7. Warm-rain microphysics (density form)
    # ──────────────────────────────────────────────

    """Retrieved temperature at every point of a stepped mtile (var_np1 + reference)."""
    function retrieved_T_mc(mtile, kDim, zs)
        ref = mtile.ref_state
        pbar = Springsteel.ref_pressure(ref)[:, 1]
        rho_dbar = Springsteel.ref_rho_d(ref)[:, 1]
        rho_tbar = Springsteel.ref_rho_t(ref)[:, 1]
        E_tbar = Springsteel.ref_total_energy(ref)[:, 1]
        Q_ssbar = Springsteel.ref_qss(ref)[:, 1]
        Tbar = Springsteel.reference_temperature(ref)
        n = size(mtile.var_np1, 1)
        T = zeros(n)
        for i in 1:n
            k = mod1(i, kDim)
            p = mtile.var_np1[i, 1] + pbar[k]
            rho_d = mtile.var_np1[i, 2] + rho_dbar[k]
            rho_t = mtile.var_np1[i, 3] + rho_tbar[k]
            E_t = mtile.var_np1[i, 6] + E_tbar[k]
            Q_ss = mtile.var_np1[i, 7] + Q_ssbar[k]
            ke = 0.5 * (mtile.var_np1[i, 4]^2 + mtile.var_np1[i, 5]^2)
            M = p + E_t - rho_t * (ke + Scythe.gravity * zs[i])
            T[i] = Scythe.retrieve_temperature(M, rho_d, rho_t, Q_ss, p, Tbar[k],
                                               mtile.var_np1[i, 8])
        end
        return T
    end

    """Seed a T-invariant rain bump: rho_r, rho_t and E_t move together so the
    retrieval is unchanged (delta_E_t = (C_pv*T - L_v + g*z)*delta_rho at fixed p, Q_ss)."""
    function seed_rain_bump!(patch, gridpoints, col, kDim; rho_r0=1.0e-3, zc=1200.0, zr=300.0)
        for i in 1:size(patch.physical, 1)
            k = mod1(i, kDim)
            z = gridpoints[i, 2]
            seed = rho_r0 * exp(-((z - zc) / zr)^2)
            Tref = col.Tk[k]
            patch.physical[i, 8, 1] += seed
            patch.physical[i, 3, 1] += seed
            patch.physical[i, 6, 1] += seed * ((Cpv * Tref) - L_v(Tref) +
                                               (Scythe.gravity * z))
        end
        spectralTransform!(patch)
        gridTransform!(patch)
    end

    @testset "autoconversion is thermodynamically inert" begin
        # Cloud above the 1 g/kg threshold, no rain yet: autoconversion is the only
        # active process (collection needs rain, sedimentation and the rain relaxation
        # channel need rho_r >= RHO_R_MIN). Liquid -> liquid conversion must leave every
        # thermodynamic slot BIT-identical to a precipitation-off run; only rho_r grows
        # (cloud is the diagnostic residual, so it shrinks automatically).
        mktempdir() do tmpdir
            args = (; q_l=3.0e-3, kDim=16, num_cells=8)
            function run_once(precip)
                mtile, patch, model, _ = make_mc_mtile(tmpdir; args..., precipitation=precip)
                ncols = div(size(patch.physical, 1), model.grid_params.kDim)
                for c in 1:ncols; Scythe.advance_column(mtile, c, 1); end
                return mtile
            end
            m_off = run_once(false)
            m_on = run_once(true)
            for slot in (1, 2, 3, 4, 5, 6, 7)
                @test m_on.var_np1[:, slot] == m_off.var_np1[:, slot]
            end
            @test all(m_on.var_np1[:, 8] .> 0.0)
            @test maximum(m_on.var_np1[:, 8]) < 3.0e-3   # bounded by the available cloud
            @test all(m_off.var_np1[:, 8] .== 0.0)
        end
    end

    @testset "sedimentation moves rain and total mass together" begin
        # Cloud-free saturated column with a T-invariant rain bump aloft: the rain
        # channel is quiet (Q_ss = 0) and there is no cloud to convert, so the on/off
        # difference isolates the sedimentation flux. rho_r and rho_t must receive the
        # SAME fitted divergence, the bump must fall, and the energy coupling must keep
        # the retrieved temperature unchanged to leading order.
        mktempdir() do tmpdir
            args = (; q_l=0.0, kDim=32, num_cells=8, ts=0.05)
            function run_once(precip)
                mtile, patch, model, col = make_mc_mtile(tmpdir; args..., precipitation=precip)
                gp = Scythe.getGridpoints(patch)
                seed_rain_bump!(patch, gp, col, model.grid_params.kDim)
                rho_r0 = copy(patch.physical[:, 8, 1])
                ncols = div(size(patch.physical, 1), model.grid_params.kDim)
                for c in 1:ncols; Scythe.advance_column(mtile, c, 1); end
                return mtile, gp, rho_r0
            end
            m_off, gp, rho_r0 = run_once(false)
            m_on, _, _ = run_once(true)
            @test all(isfinite.(m_on.var_np1))

            d3 = m_on.var_np1[:, 3] .- m_off.var_np1[:, 3]
            d8 = m_on.var_np1[:, 8] .- m_off.var_np1[:, 8]
            # Sedimentation actually moved mass...
            @test maximum(abs.(d8)) > 1.0e-8
            # ...and rho_t tracks rho_r exactly (identical -dF/dz in both slots)
            @test all(isapprox.(d3, d8; atol=1.0e-14))
            # The bump falls: rain-weighted mean height decreases
            kDim = 32
            zs = gp[:, 2]
            com(r) = sum(max.(r, 0.0) .* zs) / sum(max.(r, 0.0))
            @test com(m_on.var_np1[:, 8]) < com(rho_r0)
            # Full energy coupling: the local exchange leaves the retrieval unchanged;
            # only the genuine transport term (~1e-5 K per step) remains. A missing or
            # wrong e_l/gz coupling shows up at the 0.1 K level.
            dT = retrieved_T_mc(m_on, kDim, zs) .- retrieved_T_mc(m_off, kDim, zs)
            @test maximum(abs.(dT)) < 1.0e-3
        end
    end

    @testset "rain evaporation in subsaturated air" begin
        # Rain falling through dry air: the rain channel of the supersaturation
        # relaxation evaporates it (no separate Qevap parameterization). Vapor is
        # added (Q_ss rises toward saturation), rain is lost from rho_r but NOT from
        # rho_t (the vapor stays in the column), and the retrieval cools.
        mktempdir() do tmpdir
            args = (; dry=true, kDim=32, num_cells=8, ts=0.05)
            function run_once(precip)
                mtile, patch, model, col = make_mc_mtile(tmpdir; args..., precipitation=precip)
                gp = Scythe.getGridpoints(patch)
                seed_rain_bump!(patch, gp, col, model.grid_params.kDim)
                ncols = div(size(patch.physical, 1), model.grid_params.kDim)
                for c in 1:ncols; Scythe.advance_column(mtile, c, 1); end
                return mtile, gp
            end
            m_off, gp = run_once(false)
            m_on, _ = run_once(true)
            @test all(isfinite.(m_on.var_np1))

            d3 = m_on.var_np1[:, 3] .- m_off.var_np1[:, 3]
            d7 = m_on.var_np1[:, 7] .- m_off.var_np1[:, 7]
            d8 = m_on.var_np1[:, 8] .- m_off.var_np1[:, 8]
            # Evaporation added vapor somewhere in the rain shaft
            @test maximum(d7) > 0.0
            # Rain lost beyond what sedimentation moved: d8 - d3 = ts * Qdot_r < 0
            @test minimum(d8 .- d3) < 0.0
            @test all(d8 .- d3 .<= 1.0e-14)
            # Evaporative cooling with E_t held: the retrieval must cool, never warm
            kDim = 32
            zs = gp[:, 2]
            dT = retrieved_T_mc(m_on, kDim, zs) .- retrieved_T_mc(m_off, kDim, zs)
            @test minimum(dT) < 0.0
            @test maximum(dT) < 1.0e-3
        end
    end

    @testset "no condensational rain growth in cloud-free air" begin
        # The O01 spurious-blob mechanism in miniature: rain seeds in supersaturated
        # CLOUD-FREE air (spectral ringing + gravity-wave cooling at the lid) must not
        # grow by direct vapor deposition — the physical pathway is condensation ->
        # cloud -> autoconversion. With no cloud in the column, rain may only move
        # (sedimentation, where d_rho_t tracks d_rho_r exactly) or evaporate, so the
        # precipitation on/off difference must satisfy d8 - d3 <= 0 EVERYWHERE.
        mktempdir() do tmpdir
            args = (; q_l=0.0, kDim=32, num_cells=8, ts=0.05)
            function run_once(precip)
                mtile, patch, model, col = make_mc_mtile(tmpdir; args..., precipitation=precip)
                gp = Scythe.getGridpoints(patch)
                kDim = model.grid_params.kDim
                seed_rain_bump!(patch, gp, col, kDim)
                # Co-located supersaturation bump (S ~ 1e-3 at the center, well above
                # the nucleation threshold). Prognostic Q_ss only: both runs carry the
                # identical seed, so the on/off difference isolates the microphysics.
                for i in 1:size(patch.physical, 1)
                    k = mod1(i, kDim)
                    z = gp[i, 2]
                    rho_vs_k = rho_v_sat(col.Tk[k], col.p_Pa[k] / 100.0)
                    patch.physical[i, 7, 1] += 1.0e-3 * rho_vs_k *
                                               exp(-((z - 1200.0) / 300.0)^2)
                end
                spectralTransform!(patch)
                gridTransform!(patch)
                ncols = div(size(patch.physical, 1), kDim)
                for c in 1:ncols; Scythe.advance_column(mtile, c, 1); end
                return mtile
            end
            m_off = run_once(false)
            m_on = run_once(true)
            @test all(isfinite.(m_on.var_np1))
            d3 = m_on.var_np1[:, 3] .- m_off.var_np1[:, 3]
            d8 = m_on.var_np1[:, 8] .- m_off.var_np1[:, 8]
            # Sedimentation is active (the rain actually moves)...
            @test maximum(abs.(d8)) > 1.0e-8
            # ...but no rain is created by condensation anywhere in the column
            @test all(d8 .- d3 .<= 1.0e-14)
        end
    end

    # ──────────────────────────────────────────────
    # 7b. Rayleigh sponge (upper-boundary absorbing layer)
    # ──────────────────────────────────────────────

    @testset "Rayleigh sponge damps momentum and routes KE to E_t" begin
        # Momentum-only Durran-Klemp sponge: u and w are damped toward the resting
        # base state above z_damp, the destroyed resolved KE follows into E_t (the
        # FRIC_KE invariant: dE = 2*rho_t*tau*ke, T/p/Q_ss held), and everything
        # below the onset height — and every other slot — is bit-identical to an
        # alpha = 0 run.
        mktempdir() do tmpdir
            alpha = 0.2
            z_damp = 1000.0
            args = (; dry=true, kDim=32, num_cells=8, ts=0.1, u_side_bc=NeumannBC())
            function run_once(a)
                mtile, patch, model, _ = make_mc_mtile(tmpdir; args..., alpha=a,
                                                       z_damp=z_damp)
                gp = Scythe.getGridpoints(patch)
                kDim = model.grid_params.kDim
                for i in 1:size(patch.physical, 1)
                    x, z = gp[i, 1], gp[i, 2]
                    patch.physical[i, 4, 1] += 20.0 * sin(0.5 * pi * z / 2000.0)
                    patch.physical[i, 5, 1] += 5.0 * exp(-((x - 1000.0) / 300.0)^2 -
                                                         ((z - 1500.0) / 200.0)^2)
                end
                spectralTransform!(patch)
                gridTransform!(patch)
                # The filtered pre-step state the tendency actually sees
                u0 = copy(patch.physical[:, 4, 1])
                w0 = copy(patch.physical[:, 5, 1])
                rho_tp0 = copy(patch.physical[:, 3, 1])
                ncols = div(size(patch.physical, 1), kDim)
                for c in 1:ncols; Scythe.advance_column(mtile, c, 1); end
                return mtile, gp, u0, w0, rho_tp0
            end
            m_off, gp, u0, w0, rho_tp0 = run_once(0.0)
            m_on, _, _, _, _ = run_once(alpha)

            kDim = 32
            zs = gp[:, 2]
            ztop = gp[kDim, 2]
            RAY = Scythe.Rayleigh_damping.(alpha, zs, z_damp, ztop)
            rho_tbar = Springsteel.ref_rho_t(m_on.ref_state)[:, 1]
            rho_t0 = rho_tp0 .+ [rho_tbar[mod1(i, kDim)] for i in eachindex(zs)]
            ke0 = 0.5 .* ((u0 .^ 2) .+ (w0 .^ 2))

            d4 = m_on.expdot_n[:, 4] .- m_off.expdot_n[:, 4]
            d5 = m_on.expdot_n[:, 5] .- m_off.expdot_n[:, 5]
            d6 = m_on.expdot_n[:, 6] .- m_off.expdot_n[:, 6]
            below = zs .<= z_damp
            @test all(d4[below] .== 0.0)
            @test all(d5[below] .== 0.0)
            @test all(d6[below] .== 0.0)
            @test any(.!below)
            @test maximum(abs.(d4 .- (RAY .* u0))) < 1.0e-10
            @test maximum(abs.(d5 .- (RAY .* w0))) < 1.0e-10
            expected6 = 2.0 .* rho_t0 .* RAY .* ke0
            @test maximum(abs.(d6 .- expected6)) <
                  1.0e-10 * maximum(abs.(expected6)) + 1.0e-10
            # T, p, Q_ss, masses: no sponge term at all
            for slot in (1, 2, 3, 7, 8)
                @test m_on.expdot_n[:, slot] == m_off.expdot_n[:, slot]
            end
            # After the step the retrieved temperature is unchanged to O(ts^2): a
            # missing E_t coupling shows up at the ~1e-2 K level with this seed.
            dT = retrieved_T_mc(m_on, kDim, zs) .- retrieved_T_mc(m_off, kDim, zs)
            @test maximum(abs.(dT)) < 1.0e-3
        end

        # Configs without :alpha/:z_damp keys must run (the sponge defaults to off)
        mktempdir() do tmpdir
            mtile, patch, model, _ = make_mc_mtile(tmpdir; dry=true)
            delete!(model.physical_params, :alpha)
            delete!(model.physical_params, :z_damp)
            ncols = div(size(patch.physical, 1), model.grid_params.kDim)
            for c in 1:ncols; Scythe.advance_column(mtile, c, 1); end
            @test all(isfinite.(mtile.var_np1))
        end
    end

    # ──────────────────────────────────────────────
    # 8. Vertical moist diffusion (s_t heat + water species)
    # ──────────────────────────────────────────────

    @testset "resting base is untouched by full moist diffusion" begin
        # s_tbar and rho_vbar come from mc_reference_diagnostics — the SAME retrieval
        # pipeline the equation set runs — so at rest s_t' = 0 and rho_v' = 0 BIT-exactly.
        # On the DRY base every vertical diffusive tendency (heat AND water) must then
        # vanish bit-for-bit; this catches a reference profile built from the (not
        # bit-identical) file Tbar instead.
        mktempdir() do tmpdir
            mK, pK, modK, _ = make_mc_mtile(tmpdir; dry=true,
                                            Kvdiff=75.0, Kvdiff_water=75.0, Khdiff=0.0)
            step_mc!(mK, pK, modK, 3)
            for slot in (1, 2, 3, 4, 5, 6, 8)
                @test maximum(abs.(pK.physical[:, slot, 1])) == 0.0
            end
            # Q_ss carries a pre-existing ~1e-18 tracking crumb (file Tbar vs retrieved-T
            # saturation in the relaxation limiters) even with all diffusion off
            @test maximum(abs.(pK.physical[:, 7, 1])) < 1.0e-16
        end
        # The CLOUDY base carries a pre-existing machine-precision condensation crumb
        # (the file-derived Q_ssbar vs the retrieved-T saturation differ at ~1e-17, so
        # Qdot != 0 at rest even with diffusion off — verified on the pre-diffusion
        # code). The diffusion solves see that crumb through the star state, so exact
        # zero is unattainable; the base must still be preserved to noise level.
        mktempdir() do tmpdir
            mK, pK, modK, _ = make_mc_mtile(tmpdir; dry=false, q_l=1.0e-3,
                                            Kvdiff=75.0, Kvdiff_water=75.0, Khdiff=0.0)
            step_mc!(mK, pK, modK, 3)
            @test maximum(abs.(pK.physical[:, 1, 1])) < 1.0e-9    # p [Pa]
            @test maximum(abs.(pK.physical[:, 6, 1])) < 1.0e-8    # E_t [J/m^3]
            for slot in (2, 3, 4, 5, 7, 8)
                @test maximum(abs.(pK.physical[:, slot, 1])) < 1.0e-10
            end
        end
    end

    @testset "water diffusion: rain bump conserves mass and holds T" begin
        # Rain bump aloft in a cloud-free saturated column, ONLY Kvdiff_water active.
        # The on/off difference isolates the water solves: rho_d untouched bit-exactly,
        # rho_t tracks rho_r (rain is the only water moving), the column integral of the
        # Neumann solve is conserved, and the fixed-T increment map leaves the retrieval
        # unchanged to splitting-error tolerance.
        mktempdir() do tmpdir
            args = (; q_l=0.0, kDim=32, num_cells=8, ts=0.05)
            function run_once(Kw)
                mtile, patch, model, col = make_mc_mtile(tmpdir; args..., Kvdiff_water=Kw)
                gp = Scythe.getGridpoints(patch)
                seed_rain_bump!(patch, gp, col, model.grid_params.kDim)
                ncols = div(size(patch.physical, 1), model.grid_params.kDim)
                for c in 1:ncols; Scythe.advance_column(mtile, c, 1); end
                return mtile, gp
            end
            m_off, gp = run_once(0.0)
            m_on, _ = run_once(75.0)
            @test all(isfinite.(m_on.var_np1))

            # rho_d carries no water diffusion: bit-identical
            @test m_on.var_np1[:, 2] == m_off.var_np1[:, 2]
            d3 = m_on.var_np1[:, 3] .- m_off.var_np1[:, 3]
            d8 = m_on.var_np1[:, 8] .- m_off.var_np1[:, 8]
            @test maximum(abs.(d8)) > 1.0e-8            # diffusion acted on the bump
            # Rain is the only water species moving: rho_w' and rho_r' get the same
            # increment (same Neumann operator, same data)
            @test all(isapprox.(d3, d8; atol=1.0e-12))
            # Neumann solve conserves the column integral (trapezoid per column)
            kDim = 32
            zs = gp[1:kDim, 2]
            ncols = div(length(d8), kDim)
            for c in 1:ncols
                seg = d8[(c-1)*kDim+1:c*kDim]
                integral = sum(0.5 .* (seg[1:end-1] .+ seg[2:end]) .* diff(zs))
                mass = sum(0.5 .* (m_on.var_np1[(c-1)*kDim+1:c*kDim, 8][1:end-1] .+
                                   m_on.var_np1[(c-1)*kDim+1:c*kDim, 8][2:end]) .* diff(zs))
                @test abs(integral) < 1.0e-6 * max(abs(mass), 1.0e-3)
            end
            # Fixed-T map: the retrieval is invariant under the water increments
            dT = retrieved_T_mc(m_on, kDim, gp[:, 2]) .- retrieved_T_mc(m_off, kDim, gp[:, 2])
            @test maximum(abs.(dT)) < 1.0e-4
        end
    end

    @testset "water diffusion: vapor bump keeps the cloud residual zero" begin
        # Subsaturated vapor bump in dry air, ONLY Kvdiff_water active: rho_w' and
        # rho_v' diffuse through the SAME operator, so the implied cloud increment
        # delta_rho_c = delta_rho_w - delta_rho_v must stay ~0 — vapor diffusion cannot
        # manufacture cloud. Rain stays exactly zero, and the fixed-T map holds T.
        mktempdir() do tmpdir
            args = (; dry=true, kDim=32, num_cells=8, ts=0.05)
            function run_once(Kw)
                mtile, patch, model, col = make_mc_mtile(tmpdir; args..., Kvdiff_water=Kw)
                gp = Scythe.getGridpoints(patch)
                kDim = model.grid_params.kDim
                # T-invariant vapor seed: dQ_ss = drho_t = seed, dp = Rv*T*seed,
                # dE_t = (Cvv*T + g*z)*seed (the water map's drho_w = drho_v case)
                for i in 1:size(patch.physical, 1)
                    k = mod1(i, kDim)
                    zi = gp[i, 2]
                    seed = 2.0e-3 * exp(-((zi - 1000.0) / 300.0)^2)
                    Tref = col.Tk[k]
                    patch.physical[i, 7, 1] += seed
                    patch.physical[i, 3, 1] += seed
                    patch.physical[i, 1, 1] += Rv * Tref * seed
                    patch.physical[i, 6, 1] += ((Cvv * Tref) + (Scythe.gravity * zi)) * seed
                end
                spectralTransform!(patch)
                gridTransform!(patch)
                ncols = div(size(patch.physical, 1), kDim)
                for c in 1:ncols; Scythe.advance_column(mtile, c, 1); end
                return mtile, gp
            end
            m_off, gp = run_once(0.0)
            m_on, _ = run_once(75.0)
            @test all(isfinite.(m_on.var_np1))

            d3 = m_on.var_np1[:, 3] .- m_off.var_np1[:, 3]
            d7 = m_on.var_np1[:, 7] .- m_off.var_np1[:, 7]
            @test maximum(abs.(d3)) > 1.0e-8            # diffusion acted
            # Vapor-only water: the rho_w and rho_v increments must agree (no cloud
            # manufactured); the two fields ride the same operator but different
            # staging paths (prognostic zz slots vs a column refit), hence the tolerance
            @test all(isapprox.(d3, d7; atol=1.0e-9))
            # No rain appears from water diffusion of a rain-free column
            @test m_on.var_np1[:, 8] == m_off.var_np1[:, 8]
            # Fixed-T map holds the retrieval
            kDim = 32
            dT = retrieved_T_mc(m_on, kDim, gp[:, 2]) .- retrieved_T_mc(m_off, kDim, gp[:, 2])
            @test maximum(abs.(dT)) < 1.0e-4
        end
    end
end
