# test/test_cloud_evap_realization.jl -- the CLOUD donor's realization of its evaporation
# (`options[:cloud_evap_realization]`, default true; `cloud_evap_realized`,
# `cloud_donor_conductance`).
#
# The step-mean cloud phase change `qc_bar` of the ETD pre-compute is set by the vapor
# DEFICIT (the propagator bounds `Q_ss`, not `ρ_c`), so in dry air a thin cloud could be asked
# to evaporate more than it holds: the cloud slot fell to ν ≈ 0 while the pressure, the vapor
# and `Q_ss` took the FULL draw -- energy and vapor moved for cloud that did not exist
# (measured 2026-10-14 at an RCE burst peak: ~1.1 K of spurious cooling per 10 min).
#
# The cure is the file's one rule for a donor (`relaxation_realization`): every sink on the
# cloud -- evaporation, autoconversion + collection, and with ice the homogeneous freezing and
# the cloud half of the riming -- is realized as `rate_j · J₀(κ_tot Δt)` with
# `κ_tot = Σ_j rate_j/ρ_c` the reservoir's TOTAL conductance. No cap, no `Δt` in a rate, no
# branch: the draw converges to the rate as `Δt → 0` and to the reservoir as `Δt → ∞`, and
# every consumer -- the vapor `dv`, the pressure `d1`, the cloud slot `d9 = Δt J_c qc_bar`, the
# propagated `Q_ss^{n+1}` and the rain's `AUTO_COLL` source -- reads the one realized number.
#
# Every driver assertion compares two tiles advanced from the SAME state with the realization
# on and off, on the SAME step, through the per-thread ETD scratch the step leaves behind (the
# increments themselves, before the acoustic solve spreads a pressure difference up the
# column). The tile is lobe-relaxation's: a periodic XZ slice with 500 m vertical cells,
# horizontally uniform.

using Test
using Scythe
using Springsteel
using SparseArrays
using Springsteel: PeriodicBC, NeumannBC, DirichletBC

@testset "cloud evaporation realization (J₀ on the cloud donor's total)" begin

    SPAN = 8.0e3; NCELL = 4
    ZTOP = 10.0e3; NCELL_K = 20
    MU = 1.0e-7
    J0(x) = -expm1(-x) / x
    UP = 1.0 + 4.0 * eps()     # one rounding of a saturated J₀ (κΔt ≫ 1): the draw IS ρ_c

    function lr_column(z, rh)
        Tk = @. 300.0 - 0.0065 * z
        p_Pa = @. 100000.0 * exp(-z / 8000.0)
        rho_v = rh .* Scythe.rho_v_sat.(Tk, p_Pa ./ 100.0)
        rho_d = (p_Pa .- (Scythe.Rv .* Tk .* rho_v)) ./ (Scythe.Rd .* Tk)
        return (; Tk, p_Pa, rho_d, rho_v, rho_c = zeros(length(z)))
    end

    """A periodic XZ RiRk tile under `:bhyp` (or `tr`), RH `rh` everywhere, with a smooth
    cloud layer of peak `amp` [kg/m^3] centred at 2 km (`uniform`: `amp` at every level, no
    thin edge anywhere) and, if `rain > 0`, a rain layer of that peak on the same profile.
    `real` is the `options[:cloud_evap_realization]` value, or `nothing` to leave the key
    absent."""
    function ce_tile(tmpdir, tag; amp = 2.0e-6, real = true, rh = 0.3, tr = :bhyp, ts = 0.5,
                     precip = false, uniform = false, rain = 0.0)
        opts = Dict{Symbol,Any}(:semiimplicit => true, :exact_reference_state => true,
                                :precipitation => precip, :vertical_mixing => false,
                                :condensate_transform => tr, :rain_transform => tr)
        real === nothing || (opts[:cloud_evap_realization] = real)
        varnames = Scythe.mc_var_names(opts)
        vars = Dict(v => i for (i, v) in enumerate(varnames))
        side_bc = Dict{String,Any}(v => PeriodicBC() for v in varnames)
        wall_bc = merge(Dict{String,Any}(v => NeumannBC() for v in varnames),
                        Dict{String,Any}("w" => DirichletBC()))
        gp = GridParameters(geometry = "RiRk",
            iMin = 0.0, iMax = SPAN, num_cells_i = NCELL,
            kMin = 0.0, kMax = ZTOP, num_cells_k = NCELL_K,
            BCL = side_bc, BCR = side_bc, BCB = wall_bc, BCT = wall_bc, vars = vars)
        ref_file = joinpath(tmpdir, "ce_$(tag).ref")
        pp = Dict{Symbol,Float64}(:Khdiff => 0.0, :Kvdiff => 0.0, :Kvdiff_heat => 0.0,
                                  :Kvdiff_water => 0.0, :Kv_mudiff => 0.0,
                                  :tau_qss => 10.0, :alpha => 0.0, :z_damp => 20.0e3,
                                  :f => 0.0, :N_r => 1.0e-3)
        model = ModelParameters(ts = ts, integration_time = 10.0, output_interval = 10.0,
            equation_set = "moist_compressible_XZ", ref_state_file = ref_file,
            grid_params = gp, physical_params = pp, options = opts)
        gp = model.grid_params
        patch = createGrid(gp)
        pts = Scythe.getGridpoints(patch)
        z = pts[1:gp.kDim, end]
        col = lr_column(z, rh)
        Scythe.write_exact_ref_mc(ref_file, z, col.p_Pa, col.rho_d, col.rho_v, col.rho_c)
        patch.physical .= 0.0
        rc_i = Scythe.mc_slot(gp.vars, "rho_c")
        rr_i = Scythe.mc_slot(gp.vars, "rho_r")
        for i in 1:size(patch.physical, 1)
            shape = exp(-((pts[i, end] - 2000.0) / 800.0)^2)
            rc = uniform ? amp : amp * shape
            rr = rain * shape
            patch.physical[i, rc_i, 1] = Scythe.condensate_slot(rc, 0.0, tr, MU)
            patch.physical[i, rr_i, 1] = Scythe.condensate_slot(rr, 0.0, tr, MU)
            patch.physical[i, gp.vars["rho_t"], 1] = rc + rr  # the partition starts closed
        end
        spectralTransform!(patch)
        gridTransform!(patch)
        hrm = sparse(Int64[], Int64[], Float64[], size(patch.spectral, 1),
                     size(patch.spectral, 2))
        mtile = createModelTile(patch, patch, model, hrm)
        return mtile, patch, model
    end

    ncols(patch, model) = div(size(patch.physical, 1), model.grid_params.kDim)
    advance_all!(m, p, mo; t = 1) =
        (for c in 1:ncols(p, mo); Scythe.advance_column(m, c, t); end)
    scratch(m) = m.mc_scratch[Threads.threadid()]
    cevap_pts(m) = sum(view(m.mc_water_stats, Scythe.MC_CEVAP_PTS, :))
    cevap_held(m) = sum(view(m.mc_water_stats, Scythe.MC_CEVAP_HELD, :))

    """One step of two identically seeded tiles, realization `on` vs `off`."""
    function one_step(tmp, tag; kw...)
        m1, p1, mo1 = ce_tile(tmp, "$(tag)_on"; real = true, kw...)
        m0, p0, mo0 = ce_tile(tmp, "$(tag)_off"; real = false, kw...)
        @test p1.physical == p0.physical
        m1.mc_water_stats .= 0.0; m0.mc_water_stats .= 0.0
        advance_all!(m1, p1, mo1)
        advance_all!(m0, p0, mo0)
        return m1, m0, p1, mo1
    end

    # ── 1. the functions ────────────────────────────────────────────────────────────
    @testset "cloud_donor_conductance: the warm sinks as a conductance on the cloud" begin
        # evaporation (Qdot < 0) and conversion to rain, over the cloud present
        @test Scythe.cloud_donor_conductance(-3.0e-6, 1.0e-6, 2.0e-5) === (3.0e-6 + 1.0e-6) / 2.0e-5
        # CONDENSATION is a source and contributes nothing: the max is a sign test
        @test Scythe.cloud_donor_conductance(3.0e-6, 1.0e-6, 2.0e-5) === 1.0e-6 / 2.0e-5
        @test Scythe.cloud_donor_conductance(3.0e-6, 0.0, 2.0e-5) === 0.0
        # no cloud (or a negative untransformed one): no reservoir, an exact zero conductance
        @test Scythe.cloud_donor_conductance(-3.0e-6, 1.0e-6, 0.0) === 0.0
        @test Scythe.cloud_donor_conductance(-3.0e-6, 1.0e-6, -4.0e-7) === 0.0
    end

    @testset "cloud_evap_realized: J₀ on the evaporation, condensation untouched" begin
        f = 0.7
        for q in (3.0e-6, 0.0, 2.0e-7)
            @test Scythe.cloud_evap_realized(q, f) === q
        end
        @test Scythe.cloud_evap_realized(-1.0e-6, f) === -1.0e-6 * f
        # the factor of a reservoir with no sink is an exact 1.0, and 1.0 * q === q
        @test Scythe.cloud_evap_realized(-1.0e-6, 1.0) === -1.0e-6
        @test Scythe.relaxation_realization(0.0, 1.0e-6, 0.5) === 1.0
        @test Scythe.relaxation_realization(1.0e-6, 0.0, 0.5) === 1.0
    end

    @testset "J₀ asymptotics on the cloud draw" begin
        r = 1.0e-6; rho = 2.0e-6; kap = r / rho
        draw(dt) = Scythe.cloud_evap_realized(-r, Scythe.relaxation_realization(r, rho, dt))
        # Δt → 0 recovers the RATE, to O(κΔt): J₀ = 1 − κΔt/2 + O((κΔt)²)
        for dt in (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1)
            err = abs(draw(dt) + r) / r
            @test err <= 0.5 * kap * dt
            @test err >= 0.5 * kap * dt * (1.0 - kap * dt / 3.0)
        end
        # Δt → ∞: the converted MASS tends to the reservoir and never exceeds it, monotonely,
        # and with no kink where the old cap would have engaged (κΔt = 1)
        # (`UP` is round-off: once `e^{−κΔt}` underflows the draw IS the reservoir, to an ulp.)
        conv(dt) = -draw(dt) * dt
        dts = 10.0 .^ (-2:0.25:8)
        cs = conv.(dts)
        @test all(cs[kap .* dts .< 30.0] .< rho)
        @test all(cs .<= rho * UP)
        @test all(diff(cs) .>= -rho * eps())
        @test all(diff(cs)[kap .* dts[1:end-1] .< 30.0] .> 0.0)
        @test conv(1.0e8) ≈ rho rtol = 1.0e-12
        @test conv(1.0e4) ≈ rho * -expm1(-kap * 1.0e4) rtol = 1.0e-14
        # an absurdly stiff conversion is still a finite fraction of the reservoir
        @test 0.0 < -Scythe.cloud_evap_realized(-1.0, Scythe.relaxation_realization(1.0, 1.0e-12, 1.0)) * 1.0 <= 1.0e-12
    end

    @testset "the SHARE: one conductance, proportional shares, the SUM bounded" begin
        rho = 1.0e-6; e = 1.0e-6; a = 4.0e-7
        for dt in (0.1, 1.0, 10.0, 1.0e3)
            kc = Scythe.cloud_donor_conductance(-e, a, rho)
            f = Scythe.relaxation_realization(kc * rho, rho, dt)
            ev = -Scythe.cloud_evap_realized(-e, f)
            au = a * f
            @test (ev + au) * dt <= rho * UP              # the TOTAL is bounded
            @test (ev + au) * dt ≈ rho * -expm1(-kc * dt) rtol = 1.0e-14
            @test ev / au ≈ e / a rtol = 1.0e-14          # shares in proportion to the rates
            # a sink realized ALONE would take more: the share is what bounds the sum
            @test ev < e * Scythe.relaxation_realization(e, rho, dt)
        end
    end

    @testset "step-size independence: one step or two over the same interval" begin
        # The realization is an integrator coefficient on a rate, so over a fixed interval the
        # draw is the same to O(Δt) whether it is taken in one step or two. For a true
        # relaxation (rate ∝ reservoir) it is the same EXACTLY: J₀ is the exact step-mean.
        rho = 2.0e-6; kap = 0.4
        one_r(T) = kap * rho * Scythe.relaxation_realization(kap * rho, rho, T) * T
        function two_r(T)
            d1 = kap * rho * Scythe.relaxation_realization(kap * rho, rho, T / 2) * T / 2
            r2 = rho - d1
            return d1 + kap * r2 * Scythe.relaxation_realization(kap * r2, r2, T / 2) * T / 2
        end
        for T in (0.5, 5.0, 50.0)
            @test one_r(T) ≈ two_r(T) rtol = 1.0e-12
        end
        # A FROZEN rate (the deficit-driven evaporation is not ∝ ρ_c): the two differ, by an
        # amount that vanishes at least linearly with the step.
        r = 1.0e-6
        one_f(T) = r * Scythe.relaxation_realization(r, rho, T) * T
        function two_f(T)
            d1 = r * Scythe.relaxation_realization(r, rho, T / 2) * T / 2
            return d1 + r * Scythe.relaxation_realization(r, rho - d1, T / 2) * T / 2
        end
        rel(T) = abs(one_f(T) - two_f(T)) / one_f(T)
        errs = rel.((0.4, 0.2, 0.1, 0.05))
        @test all(errs[2:end] ./ errs[1:end-1] .< 0.55)
        @test errs[end] < (r / rho) * 0.05                # O(κΔt) and below it
        @test all(two_f.((0.4, 4.0, 40.0)) .<= rho)       # and never past the reservoir
    end

    @testset "_ice_donor_factors: the cloud's warm conductance joins the ice legs" begin
        hf = (mim = 1.0e-4, mimr = 2.0e-4, nimr = 3.0)
        bg = (mbiggr = 5.0e-4, nbiggr = 7.0)
        pz = Scythe._ice_empty_pre()
        qc = 1.0e-3; qr = 5.0e-4; nr = 100.0; dt = 1.0
        f0 = Scythe._ice_donor_factors(hf, bg, pz, pz, pz, qc, qr, nr, 0.0, dt)
        @test Scythe._ice_donor_factors(hf, bg, pz, pz, pz, qc, qr, nr, 0.0, dt;
                                        kc = 0.0) === f0     # kc = 0: bitwise the old factors
        kc = 0.2
        f1 = Scythe._ice_donor_factors(hf, bg, pz, pz, pz, qc, qr, nr, 0.0, dt; kc = kc)
        @test f1[1] === Scythe.relaxation_realization(hf.mim + kc * qc, qc, dt)
        @test f1[1] < f0[1]
        @test f1[2] === f0[2] && f1[3] === f0[3]          # the rain donors are not touched
        @test (hf.mim + kc * qc) * f1[1] * dt / qc <= 1.0
        # with no ice legs the factor is the warm one the driver forms, bitwise
        hz = (mim = 0.0, mimr = 0.0, nimr = 0.0)
        bz = (mbiggr = 0.0, nbiggr = 0.0)
        @test Scythe._ice_donor_factors(hz, bz, pz, pz, pz, qc, qr, nr, 0.0, dt; kc = kc)[1] ===
              Scythe.relaxation_realization(kc * qc, qc, dt)
    end

    # ── 2. the closed exchange in the driver ─────────────────────────────────────────
    @testset "thin cloud in dry air: J₀ on the draw, in every consumer" begin
        mktempdir() do tmp
            ts = 0.5
            m1, m0, p1, mo = one_step(tmp, "thin"; ts = ts)
            S1 = scratch(m1); S0 = scratch(m0)
            rc = S0.rho_c
            @test S1.rho_c == rc                          # the same state n
            qc = max.(rc, 0.0) ./ S1.rho_d
            # The unrealized draw really does exceed the cloud somewhere, or this is vacuous.
            over = (S0.Qdot_bar .* ts) .< .-max.(rc, 0.0)
            @test count(over) > 0
            # The conductance and the factor, formed the same way in both modes
            @test S1.kappa_c == S0.kappa_c
            @test S1.kappa_c == Scythe.cloud_donor_conductance.(S0.Qdot, 0.0, rc)
            @test S1.f_cloud == Scythe.relaxation_realization.(S1.kappa_c .* qc, qc, ts)
            ev = (S0.Qdot_bar .< 0.0) .& (rc .> 0.0)
            @test count(ev) >= count(over)
            # (a) J₀ < 1 at EVERY evaporating point with cloud -- not only where a cap bit
            @test all(S1.f_cloud[ev] .< 1.0)
            @test all(S1.Qdot_bar[ev] .=== S0.Qdot_bar[ev] .* S1.f_cloud[ev])
            # ...and the draw never exceeds the cloud present, anywhere
            @test all(.-(S1.Qdot_bar .* ts) .<= max.(rc, 0.0) .* UP)
            # (b) the VAPOR gains exactly the realized draw (no rain, no ice: vsrc = -qc_bar)
            @test all(S1.etd_dv[ev] .=== ts .* (.-S1.Qdot_bar[ev]))
            # (c) the CLOUD SLOT loses it through its own Jacobian
            @test all(S1.etd_d9[ev] .=== ts .* S1.Jc[ev] .* S1.Qdot_bar[ev])
            # (d) the PRESSURE's latent cooling is the same bracket on the realized rate
            @test all(S1.etd_d1[ev] ./ S0.etd_d1[ev] .≈ S1.f_cloud[ev])
            # (e) Q_ss gives back the evaporation that was not realized: (1 + Q_s) per kg
            dq = S1.Qdot_bar[ev] .- S0.Qdot_bar[ev]          # > 0: less evaporation
            @test all(dq .> 0.0)
            @test all(isapprox.(S1.etd_qnp1[ev] .- S0.etd_qnp1[ev],
                                .-(ts .* (1.0 .+ S1.Q_s[ev]) .* dq);
                                rtol = 1.0e-9, atol = 1.0e-18))
            # (f) at every point that is not an evaporating cloud, every consumer is bitwise
            for f in (:Qdot_bar, :etd_dv, :etd_d9, :etd_d1, :etd_qnp1, :etd_qbar, :etd_lam)
                @test getfield(S1, f)[.!ev] == getfield(S0, f)[.!ev]
            end
            @test all(S1.f_cloud[rc .<= 0.0] .=== 1.0)
            # (g) E_t carries no phase-change increment at all, realized or not
            E = Scythe.mc_slot(mo.grid_params.vars, "E_t")
            @test m1.expdot_n[:, E] == m0.expdot_n[:, E]
            # (h) the census counts it -- in BOTH modes (off: what it WOULD have withheld)
            @test cevap_pts(m1) == ncols(p1, mo) * count(ev)
            @test cevap_pts(m0) == cevap_pts(m1)
            @test cevap_held(m1) ≈ ncols(p1, mo) * sum(ts .* dq) rtol = 1.0e-12
            @test cevap_held(m0) == cevap_held(m1)
            @test all(isfinite, m1.var_np1)
        end
    end

    @testset "the share in the driver: evaporation and AUTO_COLL on one J₀" begin
        # Rain in the layer makes collection draw on the cloud beside the evaporation.
        mktempdir() do tmp
            ts = 0.5
            m1, m0, p1, mo = one_step(tmp, "share"; ts = ts, precip = true, rain = 5.0e-4)
            S1 = scratch(m1); S0 = scratch(m0)
            rc = S0.rho_c
            qc = max.(rc, 0.0) ./ S1.rho_d
            both = (S0.Qdot_bar .< 0.0) .& (S0.AUTO_COLL .> 0.0) .& (rc .> 0.0)
            @test count(both) > 0
            # ONE conductance over both warm sinks, formed on the UNREALIZED rates
            @test S1.kappa_c == Scythe.cloud_donor_conductance.(S0.Qdot, S0.AUTO_COLL, rc)
            @test S1.f_cloud == Scythe.relaxation_realization.(S1.kappa_c .* qc, qc, ts)
            # each sink realized at that one factor...
            @test all(S1.AUTO_COLL[both] .=== S0.AUTO_COLL[both] .* S1.f_cloud[both])
            @test all(S1.Qdot_bar[both] .=== S0.Qdot_bar[both] .* S1.f_cloud[both])
            # ...so their shares stand in the ratio of their rates
            @test all(S1.AUTO_COLL[both] ./ (.-S1.Qdot_bar[both]) .≈
                      S0.AUTO_COLL[both] ./ (.-S0.Qdot_bar[both]))
            # and the SUM over the step is bounded by the cloud present
            tot = (max.(.-S1.Qdot_bar, 0.0) .+ S1.AUTO_COLL) .* ts
            @test all(tot .<= max.(rc, 0.0) .* UP)
            # the shared factor is smaller than the evaporation's alone would have been
            fe = Scythe.relaxation_realization.(max.(.-S0.Qdot, 0.0) ./ S1.rho_d, qc, ts)
            @test all(S1.f_cloud[both] .< fe[both])
            # off: AUTO_COLL is the unrealized rate, bitwise
            @test all(S0.AUTO_COLL .=== Scythe.autoconversion_density.(max.(rc, 0.0), S0.rho_d) .+
                      Scythe.collection_density.(max.(rc, 0.0), S0.rho_r, S0.rho_d, S0.Tk))
            # the rain slot's source is the SAME realized AUTO_COLL (equal and opposite pair)
            @test all(isfinite, m1.var_np1)
        end
    end

    @testset "clipped drive: realized at state n, where the flux enters N" begin
        # Q_ss raised above the true deficit (a detached Q_ss) clips the liquid drive to
        # max(rho_v, 0) - rho_vs: the step-mean IS the frozen rate and its flux lives in N
        # (`etd_nl`), so the factor must act on Qdot_c and 1/tau_c at the fold.
        mktempdir() do tmp
            ts = 0.5
            m1, p1, mo = ce_tile(tmp, "clip_on"; real = true)
            m0, p0, _ = ce_tile(tmp, "clip_off"; real = false)
            for p in (p1, p0)
                p.physical[:, mo.grid_params.vars["Q_ss"], 1] .+= 2.0e-3
            end
            m1.mc_water_stats .= 0.0; m0.mc_water_stats .= 0.0
            advance_all!(m1, p1, mo); advance_all!(m0, p0, mo)
            S1 = scratch(m1); S0 = scratch(m0)
            clip = S0.etd_nl .!= 0.0
            ev = clip .& (S0.Qdot_bar .< 0.0) .& (S0.rho_c .> 0.0)
            @test count(ev) > 0
            @test all(S1.f_cloud[ev] .< 1.0)
            @test all(S1.Qdot_bar[ev] .=== S0.Qdot_bar[ev] .* S1.f_cloud[ev])
            @test all(.-(S1.Qdot_bar[ev] .* ts) .<= max.(S0.rho_c[ev], 0.0) .* UP)
            # N carries the realized flux, in the same ratio (no rain: the cloud is all of it)
            @test all(S1.etd_nl[ev] ./ S0.etd_nl[ev] .≈ S1.f_cloud[ev])
            @test all(S1.etd_dv[ev] .=== ts .* (.-S1.Qdot_bar[ev]))
            @test S1.etd_nl[.!ev] == S0.etd_nl[.!ev]
            @test cevap_pts(m1) == cevap_pts(m0) > 0.0
        end
    end

    # ── 3. where it is inert, and where it is not ───────────────────────────────────────
    @testset "ample cloud: J₀ just below one, never exactly one" begin
        # A uniform cloud: no thin edge anywhere. The cap of 9acd5b8 was the identity here;
        # J₀ is not, and should not be -- it is the step-mean of the depletion the rate
        # implies, and that is below the rate wherever the cloud is evaporating at all.
        mktempdir() do tmp
            ts = 0.5
            m1, m0, p1, mo = one_step(tmp, "ample"; amp = 1.0e-3, rh = 0.9, uniform = true,
                                      ts = ts)
            S1 = scratch(m1); S0 = scratch(m0)
            ev = (S0.Qdot_bar .< 0.0) .& (S0.rho_c .> 0.0)
            @test count(ev) > 0                           # it IS evaporating...
            @test all(S1.f_cloud[ev] .< 1.0)              # ...so J₀ < 1...
            # ...by O(κΔt): 1 − J₀(x) ≤ x/2, here up to ~0.05 -- a 1 g/m³ cloud at RH 0.9
            # e-folds in seconds, which is what the step-mean has to see
            x = S1.kappa_c .* 0.5                         # κ_c Δt, Δt = 0.5 s
            @test all(1.0 .- S1.f_cloud[ev] .<= x[ev] ./ 2)
            @test all(1.0 .- S1.f_cloud[ev] .>= (x[ev] ./ 2) .* (1.0 .- x[ev] ./ 3))
            @test all(S1.Qdot_bar[ev] .=== S0.Qdot_bar[ev] .* S1.f_cloud[ev])
            @test cevap_pts(m1) == ncols(p1, mo) * count(ev)
            # The UNCLIPPED path (the thin fixture's evaporating points are mostly clipped):
            # the realization acts on the step-mean, and Q_ss^{n+1} gives back the rest.
            un = ev .& (S0.etd_nl .== 0.0)
            @test count(un) > 0
            dq = S1.Qdot_bar[un] .- S0.Qdot_bar[un]
            @test all(dq .> 0.0)
            @test all(isapprox.(S1.etd_qnp1[un] .- S0.etd_qnp1[un],
                                .-(ts .* (1.0 .+ S1.Q_s[un]) .* dq);
                                rtol = 1.0e-9, atol = 1.0e-18))
            @test all(S1.etd_dv[un] .=== ts .* (.-S1.Qdot_bar[un]))
            @test S1.etd_lam == S0.etd_lam                # λ is not touched: J₀ is on q̄_c
        end
    end

    @testset "the key absent is the shipped default (true)" begin
        mktempdir() do tmp
            mA, pA, moA = ce_tile(tmp, "abs"; real = nothing)
            mB, pB, moB = ce_tile(tmp, "set"; real = true)
            advance_all!(mA, pA, moA); advance_all!(mB, pB, moB)
            @test mA.var_np1 == mB.var_np1
            @test Scythe.CLOUD_EVAP_REALIZATION_DEFAULT === true
        end
    end

    @testset "dry column: bitwise, with the realization on or off" begin
        mktempdir() do tmp
            m1, m0, _, _ = one_step(tmp, "dry"; amp = 0.0)
            @test m1.var_np1 == m0.var_np1
            @test m1.expdot_n == m0.expdot_n
            @test all(scratch(m1).f_cloud .=== 1.0)
            @test cevap_pts(m1) == 0.0
        end
    end

    @testset "untransformed cloud (:none) is realized the same way" begin
        mktempdir() do tmp
            ts = 0.5
            m1, m0, _, _ = one_step(tmp, "none"; tr = :none, ts = ts)
            S1 = scratch(m1); S0 = scratch(m0)
            ev = (S0.Qdot_bar .< 0.0) .& (S0.rho_c .> 0.0)
            @test count((S0.Qdot_bar .* ts) .< .-max.(S0.rho_c, 0.0)) > 0
            @test all(S1.Qdot_bar[ev] .=== S0.Qdot_bar[ev] .* S1.f_cloud[ev])
            @test all(.-(S1.Qdot_bar .* ts) .<= max.(S0.rho_c, 0.0) .* UP)
            @test all(S1.etd_d9[ev] .=== ts .* S1.Qdot_bar[ev])   # J_c = 1 under :none
        end
    end

    # ── 4. config, allocations ──────────────────────────────────────────────────────
    @testset "config round trip" begin
        mktempdir() do tmp
            owb = joinpath(pkgdir(Scythe), "simulations", "ocean_warm_bubble",
                           "scythe_config.toml")
            cfg = load_config(owb; set = ["output.dir=$(tmp)",
                                          "extra.options.cloud_evap_realization=false"])
            @test cfg.extra.options[:cloud_evap_realization] === false
            m = build_model(cfg)
            @test m.options[:cloud_evap_realization] === false
            back = load_config(write_config(cfg, joinpath(tmp, "rt.toml")))
            @test back.extra.options[:cloud_evap_realization] === false
            @test haskey(Scythe.KNOWN_OPTION_KEYS, :cloud_evap_realization)
        end
    end

    @testset "allocation-free with the realization acting" begin
        mktempdir() do tmp
            for (tag, kw) in (("alloc", ()), ("alloc_p", (precip = true, rain = 5.0e-4)))
                m, p, mo = ce_tile(tmp, tag; kw...)
                kDim = mo.grid_params.kDim
                Scythe.moist_compressible_XZ(m, 1, kDim, 2)          # compile
                m.mc_water_stats .= 0.0
                @test (@allocations Scythe.moist_compressible_XZ(m, 1, kDim, 2)) == 0
                @test cevap_pts(m) > 0.0
            end
        end
    end
end
