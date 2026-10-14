# test/test_cloud_evap_realization.jl -- the CLOUD donor's realization bound on evaporation
# (`options[:cloud_evap_realization]`, default true; `cloud_evap_bound`).
#
# The step-mean cloud phase change `qc_bar` of the ETD pre-compute is set by the vapor
# DEFICIT (the propagator bounds `Q_ss`, not `ρ_c`), so in dry air a thin cloud could be asked
# to evaporate more than it holds: the cloud slot fell to ν ≈ 0 while the pressure, the vapor
# and `Q_ss` took the FULL draw -- energy and vapor moved for cloud that did not exist
# (measured 2026-10-14 at an RCE burst peak: ~1.1 K of spurious cooling per 10 min). The bound
# caps the draw at the cloud present, `−qc_bar·Δt ≤ max(ρ_c, 0)`, and every consumer -- the
# vapor `dv`, the pressure `d1`, the cloud slot `d9 = Δt J_c qc_bar` and the propagated
# `Q_ss^{n+1}` -- reads the one bounded number.
#
# Every assertion compares two tiles advanced from the SAME state with the bound on and off,
# on the SAME step, through the per-thread ETD scratch the step leaves behind (the increments
# themselves, before the acoustic solve spreads a pressure difference up the column). The tile
# is lobe-relaxation's: a periodic XZ slice with 500 m vertical cells, horizontally uniform.

using Test
using Scythe
using Springsteel
using SparseArrays
using Springsteel: PeriodicBC, NeumannBC, DirichletBC

@testset "cloud evaporation realization bound" begin

    SPAN = 8.0e3; NCELL = 4
    ZTOP = 10.0e3; NCELL_K = 20
    MU = 1.0e-7

    function lr_column(z, rh)
        Tk = @. 300.0 - 0.0065 * z
        p_Pa = @. 100000.0 * exp(-z / 8000.0)
        rho_v = rh .* Scythe.rho_v_sat.(Tk, p_Pa ./ 100.0)
        rho_d = (p_Pa .- (Scythe.Rv .* Tk .* rho_v)) ./ (Scythe.Rd .* Tk)
        return (; Tk, p_Pa, rho_d, rho_v, rho_c = zeros(length(z)))
    end

    """A periodic XZ RiRk tile under `:bhyp` (or `tr`), RH `rh` everywhere, with a smooth
    cloud layer of peak `amp` [kg/m^3] centred at 2 km (`uniform`: `amp` at every level, no
    thin edge anywhere). `real` is the `options[:cloud_evap_realization]` value, or `nothing`
    to leave the key absent."""
    function ce_tile(tmpdir, tag; amp = 2.0e-6, real = true, rh = 0.3, tr = :bhyp, ts = 0.5,
                     precip = false, uniform = false)
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
        for i in 1:size(patch.physical, 1)
            rc = uniform ? amp : amp * exp(-((pts[i, end] - 2000.0) / 800.0)^2)
            patch.physical[i, rc_i, 1] = Scythe.condensate_slot(rc, 0.0, tr, MU)
            patch.physical[i, gp.vars["rho_t"], 1] = rc      # the partition starts closed
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

    """One step of two identically seeded tiles, bound `on` vs `off`."""
    function one_step(tmp, tag; kw...)
        m1, p1, mo1 = ce_tile(tmp, "$(tag)_on"; real = true, kw...)
        m0, p0, mo0 = ce_tile(tmp, "$(tag)_off"; real = false, kw...)
        @test p1.physical == p0.physical
        m1.mc_water_stats .= 0.0; m0.mc_water_stats .= 0.0
        advance_all!(m1, p1, mo1)
        advance_all!(m0, p0, mo0)
        return m1, m0, p1, mo1
    end

    # ── 1. the function ─────────────────────────────────────────────────────────────
    @testset "cloud_evap_bound: a cap on the draw, the identity everywhere else" begin
        ts = 0.5
        # condensation, and evaporation within the cloud present: returned unchanged, bitwise
        for (q, rc) in ((3.0e-6, 1.0e-5), (0.0, 1.0e-5), (-1.0e-6, 1.0e-5), (2.0e-7, 0.0),
                        (-2.0e-5, 1.0e-5))           # -q ts == rc exactly: not past it
            @test Scythe.cloud_evap_bound(q, rc, ts) === q
        end
        # past the cloud present: the whole cloud and no more
        @test Scythe.cloud_evap_bound(-3.0e-5, 1.0e-5, ts) === -(1.0e-5 / ts)
        @test -(Scythe.cloud_evap_bound(-3.0e-5, 1.0e-5, ts) * ts) ≈ 1.0e-5 rtol = 1.0e-15
        # no cloud (or a negative untransformed one): no evaporation, an exact +0.0
        @test Scythe.cloud_evap_bound(-1.0e-9, 0.0, ts) === 0.0
        @test Scythe.cloud_evap_bound(-1.0e-9, -4.0e-7, ts) === 0.0
    end

    # ── 2. the closed exchange ──────────────────────────────────────────────────────
    @testset "thin cloud in dry air: the draw is the cloud present, in every consumer" begin
        mktempdir() do tmp
            ts = 0.5
            m1, m0, p1, mo = one_step(tmp, "thin"; ts = ts)
            S1 = scratch(m1); S0 = scratch(m0)
            rc = S0.rho_c
            # The unbounded draw really does exceed the cloud somewhere, or this is vacuous.
            over = (S0.Qdot_bar .* ts) .< .-max.(rc, 0.0)
            @test count(over) > 0
            @test S1.rho_c == rc                          # the same state n
            # (a) the bounded draw is the whole cloud present, and no more
            @test all(S1.Qdot_bar[over] .=== .-(max.(rc[over], 0.0) ./ ts))
            # (b) the VAPOR gains exactly that cloud (no rain, no ice: vsrc = -qc_bar)
            @test all(S1.etd_dv[over] .≈ max.(rc[over], 0.0))
            @test all(S1.etd_dv[over] .=== ts .* (.-S1.Qdot_bar[over]))
            # (c) the CLOUD SLOT loses it through its own Jacobian
            @test all(S1.etd_d9[over] .=== ts .* S1.Jc[over] .* S1.Qdot_bar[over])
            # (d) the PRESSURE's latent cooling is the same bracket on the bounded rate
            @test all(S1.etd_d1[over] ./ S0.etd_d1[over] .≈
                      S1.Qdot_bar[over] ./ S0.Qdot_bar[over])
            @test all(abs.(S1.etd_d1[over]) .< abs.(S0.etd_d1[over]))
            # (e) Q_ss gives back the evaporation that was not realized: (1 + Q_s) per kg
            dq = S1.Qdot_bar[over] .- S0.Qdot_bar[over]          # > 0: less evaporation
            @test all(dq .> 0.0)
            @test all(isapprox.(S1.etd_qnp1[over] .- S0.etd_qnp1[over],
                                .-(ts .* (1.0 .+ S1.Q_s[over]) .* dq);
                                rtol = 1.0e-9, atol = 1.0e-18))
            # (f) everywhere the bound did not act, every consumer is bitwise the old one
            for f in (:Qdot_bar, :etd_dv, :etd_d9, :etd_d1, :etd_qnp1, :etd_qbar, :etd_lam)
                @test getfield(S1, f)[.!over] == getfield(S0, f)[.!over]
            end
            # (g) E_t carries no phase-change increment at all, bounded or not
            E = Scythe.mc_slot(mo.grid_params.vars, "E_t")
            @test m1.expdot_n[:, E] == m0.expdot_n[:, E]
            # (h) the census counts it -- in BOTH modes (off: where it WOULD have acted)
            @test cevap_pts(m1) == ncols(p1, mo) * count(over)
            @test cevap_pts(m0) == cevap_pts(m1)
            @test sum(view(m1.mc_water_stats, Scythe.MC_CEVAP_HELD, :)) ≈
                  ncols(p1, mo) * sum(ts .* dq) rtol = 1.0e-12
            @test all(isfinite, m1.var_np1)
        end
    end

    @testset "clipped drive: bounded at state n, where the flux enters N" begin
        # Q_ss raised above the true deficit (a detached Q_ss) clips the liquid drive to
        # max(rho_v, 0) - rho_vs: the step-mean IS the frozen rate and its flux lives in N
        # (`etd_nl`), so the bound must act on Qdot_c and 1/tau_c at the fold.
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
            over = clip .& ((S0.Qdot_bar .* ts) .< .-max.(S0.rho_c, 0.0))
            @test count(over) > 0
            @test all(S1.Qdot_bar[over] .=== Scythe.cloud_evap_bound.(S0.Qdot_bar[over],
                                                                     S0.rho_c[over], ts))
            # N carries the bounded flux, in the same ratio (no rain: the cloud is all of it)
            @test all(S1.etd_nl[over] ./ S0.etd_nl[over] .≈
                      S1.Qdot_bar[over] ./ S0.Qdot_bar[over])
            @test all(S1.etd_dv[over] .=== ts .* (.-S1.Qdot_bar[over]))
            @test S1.etd_nl[.!over] == S0.etd_nl[.!over]
            @test cevap_pts(m1) == cevap_pts(m0) > 0.0
        end
    end

    # ── 3. bitwise inert ────────────────────────────────────────────────────────────
    @testset "ample cloud: bitwise unchanged, census zero" begin
        # A uniform cloud: no thin edge anywhere (a Gaussian's tails ARE thin cloud in
        # subsaturated air, and the bound rightly acts there).
        mktempdir() do tmp
            m1, m0, p1, mo = one_step(tmp, "ample"; amp = 1.0e-3, rh = 0.9, uniform = true)
            S0 = scratch(m0)
            @test minimum(S0.Qdot_bar) < 0.0              # it IS evaporating...
            @test cevap_pts(m0) == 0.0                    # ...within the cloud present
            @test m1.var_np1 == m0.var_np1
            @test m1.expdot_n == m0.expdot_n
            @test cevap_pts(m1) == 0.0
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

    @testset "dry column: bitwise, with the bound on or off" begin
        mktempdir() do tmp
            m1, m0, _, _ = one_step(tmp, "dry"; amp = 0.0)
            @test m1.var_np1 == m0.var_np1
            @test cevap_pts(m1) == 0.0
        end
    end

    @testset "untransformed cloud (:none) is bounded the same way" begin
        mktempdir() do tmp
            ts = 0.5
            m1, m0, _, _ = one_step(tmp, "none"; tr = :none, ts = ts)
            S1 = scratch(m1); S0 = scratch(m0)
            over = (S0.Qdot_bar .* ts) .< .-max.(S0.rho_c, 0.0)
            @test count(over) > 0
            @test all(S1.Qdot_bar[over] .=== Scythe.cloud_evap_bound.(S0.Qdot_bar[over],
                                                                     S0.rho_c[over], ts))
            @test all(-(S1.Qdot_bar[over] .* ts) .<= max.(S0.rho_c[over], 0.0) .* (1 + 1e-15))
            @test all(S1.etd_d9[over] .=== ts .* S1.Qdot_bar[over])   # J_c = 1 under :none
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

    @testset "allocation-free with the bound acting" begin
        mktempdir() do tmp
            m, p, mo = ce_tile(tmp, "alloc")
            kDim = mo.grid_params.kDim
            Scythe.moist_compressible_XZ(m, 1, kDim, 2)          # compile
            m.mc_water_stats .= 0.0
            @test (@allocations Scythe.moist_compressible_XZ(m, 1, kDim, 2)) == 0
            @test cevap_pts(m) > 0.0
        end
    end
end
