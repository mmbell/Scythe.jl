# test/test_water_anchor.jl -- THE DERIVED WATER ANCHOR (`options[:water_anchor]`; Stage 3 of
# the rho_t retirement, reference/RHO_T_RETIREMENT_DESIGN.md §3.2-3.3 and §6 Stage 3, the
# plan's §4 Stage 3 and decision E9).
#
# Under `:derived` every reader of the total density takes
#
#     rho_t  = rho_d  + sum_k alpha_k nu_k        rho_t' = rho_d' + sum_k alpha_k nu_k'
#     rho_tbar_w = rho_dbar + sum_k alpha_k nu_kbar
#
# over the water-mass slots (vapor, cloud, rain, the three ice masses; alpha = 2 transformed,
# 1 density), in one fixed summation order; the vapor reconciliation nudge is deleted; slot 3
# keeps its own transport and SI increment and FEEDS NOTHING (the shadow, MC_SHADOW_*).
# `:slot` (the library default) is the code before this stage, bitwise.
#
# 3a: the option and config; `:slot` bitwise against goldens captured at
# 44da59c; the resting fixed point under `:derived` (every consumer on rho_tbar_w, the EOS and
# the `:energy` temperature, both SI paths, warm and ice); dry bitwise; slot 3 feeds nothing;
# the shadow starts at exactly 0.0; the setup refusals; the census rows; the R1T1X wall
# estimate; the netCDF derived fields.

using Test
using Scythe
using Springsteel
using SparseArrays

isdefined(Main, :VaporTransformFixtures) || include("vapor_transform_fixtures.jl")
isdefined(Main, :MYNNTestFixtures) || include("mynn_test_fixtures.jl")

@testset "the derived water anchor (options[:water_anchor])" begin
    F = Main.VaporTransformFixtures
    M = Main.MYNNTestFixtures

    # (Stage L is pinned off here unless a test asks for it: its paired form defaults ON under
    # the derived anchor since 3d, and most tests isolate one operator)
    derived(; sd = false, siw = :mass, extra...) =
        merge(Dict{Symbol,Any}(:water_anchor => :derived, :si_water_slots => siw,
                               :state_dependent_si => sd, :si_water_mixed_walls => true,
                               :lobe_relaxation => false),
              Dict{Symbol,Any}(extra))

    # ── 0. the option and the config ────────────────────────────────────────────────
    @testset "option, config default, legacy pins" begin
        @test Scythe.water_anchor_mode(Dict{Symbol,Any}()) === :slot
        @test Scythe.water_anchor_mode(Dict{Symbol,Any}(:water_anchor => :derived)) === :derived
        @test_throws ErrorException Scythe.water_anchor_mode(
            Dict{Symbol,Any}(:water_anchor => :sum))
        @test Scythe.WATER_ANCHOR_MODES == (:slot, :derived)
        @test Scythe.Numerics().water_anchor === :derived
        # 3d: the paired Stage L defaults ON under the derived anchor, off under :slot
        @test Scythe.lobe_relaxation_on(Dict{Symbol,Any}(:water_anchor => :derived))
        @test !Scythe.lobe_relaxation_on(Dict{Symbol,Any}())
        @test !Scythe.lobe_relaxation_on(Dict{Symbol,Any}(:water_anchor => :slot))
        @test !Scythe.lobe_relaxation_on(Dict{Symbol,Any}(:water_anchor => :derived,
                                                          :lobe_relaxation => false))
        @test haskey(Scythe.KNOWN_OPTION_KEYS, :water_anchor)
        for f in ("owb_default", "rce_default", "tc_default")
            s = read(joinpath(@__DIR__, "reference_data", "legacy_models",
                              "$(f).overlay.toml"), String)
            @test occursin("water_anchor = \"slot\"", s)
        end
    end

    # ── 1. :slot is bitwise 44da59c ─────────────────────────────────────────────────
    @testset ":slot and the absent key are bitwise 44da59c (golden capture)" begin
        # Three full steps of the seeded sheet + vapor bump + w under the vapor control
        # variable, captured at 44da59c (Stage 2 tip) before `:water_anchor` existed, on both
        # SI paths, warm/ice/Louis, slaved and explicit water acoustics; the MYNN fixture's
        # live column (density slots).
        golden = Dict(
            "ice_ref_none" => (UInt128(499824252298858703554839616), UInt128(18847435755383941090396674041)),
            "ice_sd_mass" => (UInt128(499824456177364866899577122), UInt128(18843588167511347091832914358)),
            "louis_sd_none" => (UInt128(179638334862226154838288795), UInt128(5247592769137669402363025278)),
            "mynn" => (UInt128(311926319919379716218570141), UInt128(10944031763325494353189912505)),
            "warm_ref_mass" => (UInt128(179638421285913927342441771), UInt128(5243771931477366918674490696)),
            "warm_sd_mass" => (UInt128(179638412199414936570312397), UInt128(5244273292793073352352678301)))
        mktempdir() do tmp
            for wa in (nothing, :slot)
                for (tag, ice, louis, sd, siw) in (("warm_ref_mass", false, false, false, :mass),
                                                   ("warm_sd_mass", false, false, true, :mass),
                                                   ("ice_sd_mass", true, false, true, :mass),
                                                   ("ice_ref_none", true, false, false, :none),
                                                   ("louis_sd_none", false, true, true, :none))
                    extra = Dict{Symbol,Any}(:si_water_slots => siw, :state_dependent_si => sd,
                                             :si_water_mixed_walls => true)
                    wa === nothing || (extra[:water_anchor] = wa)
                    m, p, mo = F.vt_tile(tmp, "g_$(tag)_$(wa)"; vtr = :bhyp, ice, louis, extra,
                        seed! = (p, x, V, c) -> F.vt_seed!(p, x, V, c))
                    F.vt_run!(m, p, mo, 3)
                    @test F.vt_checksum(m.var_np1) == golden[tag][1]
                    @test F.vt_checksum(p.physical) == golden[tag][2]
                end
                m, p, mo, gp, z, col = M.make_mynn_tile(tmp; num_cells_k = 25, kMax = 5.0e3,
                                                        ctrans = :none, rtrans = :none)
                wa === nothing || (mo.options[:water_anchor] = wa)
                M.set_state!(p, mo, gp.kDim, z, col; W0 = 0.3)
                for t in 1:3
                    M.step_all!(m, p, gp.kDim, t); Scythe.calcTendency(m); gridTransform!(p)
                end
                @test F.vt_checksum(m.var_np1) == golden["mynn"][1]
                @test F.vt_checksum(p.physical) == golden["mynn"][2]
            end
        end
    end

    # ── 2. the resting fixed point ──────────────────────────────────────────────────
    @testset "rest is a discrete fixed point under :derived (every consumer on rho_tbar_w)" begin
        mktempdir() do tmp
            for ice in (false, true), sd in (false, true), siw in (:mass, :none),
                temp in (:eos, :energy)
                extra = derived(; sd, siw, temperature = temp)
                m, p, mo = F.vt_tile(tmp, "rest_$(ice)_$(sd)_$(siw)_$(temp)"; vtr = :bhyp,
                                     ice, extra)
                Scythe.init_water_anchor_shadow!(p, mo)
                eb = m.ebudget
                # the reference every reader takes IS the derived one
                @test eb.wa
                @test Scythe.mc_rho_tbar_ref(m)[:, 1] == eb.rho_tbar_w
                @test Scythe.mc_rho_tbar_ref(m)[:, 2] == eb.rho_tbar_w_z
                # ...and the acoustic coefficient profile is built on it
                @test all(isfinite, m.mc_ref_diag.Pxi_prof)
                Scythe.advance_column(m, 1, 1)
                S = m.mc_scratch[Threads.threadid()]
                # the staged total at rest IS rho_tbar_w, bit for bit
                @test S.rho_t == eb.rho_tbar_w
                @test S.rho_t_z == eb.rho_tbar_w_z
                F.vt_run!(m, p, mo, 4; t0 = 2)
                @test all(==(0.0), m.var_np1)
                @test all(==(0.0), view(p.physical, :, :, 1))
            end
        end
    end

    # ── 3. dry ──────────────────────────────────────────────────────────────────────
    @testset "dry: rho_t === rho_d and rho_t' === rho_d' bitwise under :derived" begin
        # A dry base (RH 0: every water slot an exact zero, so W' = 0.0 and the `-0.0` select
        # returns rho_d' bitwise) with a w seed. Under :derived the staged total IS rho_d and
        # the staged perturbation IS the rho_d slot, at every point and every step, so nothing
        # the dynamics reads depends on a water slot. (The :slot arm is NOT bitwise the
        # :derived one: its own slot 3 differs from rho_d' by rounding, 1.6e-19 after four
        # steps, because the two continuity tendencies are different expressions; the arms
        # agree to that rounding.)
        mktempdir() do tmp
            seedw! = (p, x, V, c) -> begin
                for i in 1:size(p.physical, 1)
                    zz = x[i, end]
                    p.physical[i, V["w"], 1] = 0.5 * sin(2pi * x[i, 1] / F.SPAN) *
                                                sin(pi * zz / F.ZTOP)
                end
            end
            for sd in (false, true), siw in (:mass, :none)
                md = F.vt_tile(tmp, "dry_d_$(sd)_$(siw)"; vtr = :none, rh = z -> 0.0,
                               extra = derived(; sd, siw), seed! = seedw!, cqss = false)
                ms = F.vt_tile(tmp, "dry_s_$(sd)_$(siw)"; vtr = :none, rh = z -> 0.0,
                               extra = Dict{Symbol,Any}(:state_dependent_si => sd,
                                                        :si_water_slots => siw,
                                                        :si_water_mixed_walls => true),
                               seed! = seedw!, cqss = false)
                Scythe.init_water_anchor_shadow!(md[2], md[3])
                m = md[1]; S = m.mc_scratch[Threads.threadid()]; kD = md[3].grid_params.kDim
                ok = true
                for t in 1:5
                    for c in 1:F.vt_ncols(md[2], md[3])
                        Scythe.advance_column(m, c, t)
                        rows = ((c - 1) * kD + 1):(c * kD)
                        ok &= S.rho_t == S.rho_d
                        ok &= S.res_rho_t == md[2].physical[rows, 2, 1]
                        ok &= S.rho_t_z == S.rho_d_z
                    end
                    Scythe.calcTendency(m); gridTransform!(md[2])
                end
                @test ok
                F.vt_run!(ms..., 5)
                oth = [j for j in 1:size(ms[1].var_np1, 2) if j != 3]
                for j in oth
                    a = ms[1].var_np1[:, j]; b = m.var_np1[:, j]
                    @test maximum(abs, a .- b) <= 1.0e-12 * max(maximum(abs, a), 1.0e-300)
                end
            end
        end
    end

    # ── 4. slot 3 feeds nothing ─────────────────────────────────────────────────────
    @testset "slot 3 is a passive shadow: perturbing it moves nothing else" begin
        # The same seeded state twice, once with a large extra perturbation in slot 3 only (a
        # 'partition gap' the nudge would have acted on): under :derived every OTHER slot is
        # bitwise the unperturbed run's — no reader, the nudge gone; under :slot it is not.
        mktempdir() do tmp
            bump!(p, x, V, c) = begin
                F.vt_seed!(p, x, V, c)
                for i in 1:size(p.physical, 1)
                    p.physical[i, V["rho_t"], 1] += 2.0e-4 * exp(-((x[i, end] - 4000.0) / 800.0)^2)
                end
            end
            for ice in (false, true), sd in (false, true)
                ex = derived(; sd)
                a = F.vt_tile(tmp, "pa_$(ice)_$(sd)"; vtr = :bhyp, ice, extra = ex,
                              seed! = (p, x, V, c) -> F.vt_seed!(p, x, V, c))
                b = F.vt_tile(tmp, "pb_$(ice)_$(sd)"; vtr = :bhyp, ice, extra = ex, seed! = bump!)
                F.vt_run!(a..., 4); F.vt_run!(b..., 4)
                others = [j for j in 1:size(a[1].var_np1, 2) if j != 3]
                @test a[1].var_np1[:, others] == b[1].var_np1[:, others]
                @test a[1].var_np1[:, 3] != b[1].var_np1[:, 3]
            end
            # the control: under :slot the same bump moves the dynamics
            ex = Dict{Symbol,Any}(:si_water_slots => :mass, :si_water_mixed_walls => true)
            a = F.vt_tile(tmp, "ps_a"; vtr = :bhyp, extra = ex,
                          seed! = (p, x, V, c) -> F.vt_seed!(p, x, V, c))
            b = F.vt_tile(tmp, "ps_b"; vtr = :bhyp, extra = ex, seed! = bump!)
            F.vt_run!(a..., 4); F.vt_run!(b..., 4)
            @test a[1].var_np1[:, 5] != b[1].var_np1[:, 5]
        end
    end

    # ── 5. the shadow starts at exactly zero ────────────────────────────────────────
    @testset "MC_SHADOW_GAP starts at 0.0 (spectrally exact, one transform rounding pointwise)" begin
        mktempdir() do tmp
            for ice in (false, true)
                m, p, mo = F.vt_tile(tmp, "sh_$(ice)"; vtr = :bhyp, ice, extra = derived(),
                                     seed! = (p, x, V, c) -> F.vt_seed_vapor!(p, x, V, c;
                                         vtr = :none))
                # the IC's slot 3 is NOT the derived total (the seed writes rho_t' = rc + dv)
                eb = m.ebudget
                kD = eb.kDim
                nc = F.vt_ncols(p, mo)
                gap(P) = maximum(Scythe._eb_water_column(eb, P, (c - 1) * kD, 1)[5] for c in 1:nc)
                @test gap(p.physical) > 1.0e-6
                Scythe.init_water_anchor_shadow!(p, mo)
                # the shadow's spectral coefficients ARE the members' combination, exactly ...
                ws = Scythe.water_anchor_set(mo.options, mo.physical_params, Scythe.mc_slots(mo))
                comb = zeros(size(p.spectral, 1)); first = true
                for j in 1:6
                    sl = ws.slot[j]; a = ws.alpha[j]
                    (sl == 0 || a == 0.0) && continue
                    v = a .* p.spectral[:, sl]
                    comb = first ? v : comb .+ v; first = false
                end
                @test p.spectral[:, 3] == p.spectral[:, 2] .+ comb
                # ... and at the gridpoints the gap is one transform's rounding (exactly 0.0
                # where a single member carries the column; 6.5e-19 kg/m3 with three)
                @test gap(p.physical) <= 1.0e-15 * maximum(abs, p.physical[:, 3, 1])
                # and it then measures something: the shadow is not the derived total once
                # the operators have acted (nonlinear transport, the knee, the closures)
                m.tile.physical .= p.physical
                F.vt_run!(m, p, mo, 5)
                @test 0.0 < gap(p.physical) < 1.0e-4
            end
        end
    end

    # ── 6. refusals ─────────────────────────────────────────────────────────────────
    @testset "setup refusals under :derived" begin
        mktempdir() do tmp
            for (k, v, pp) in ((:water_lobe_attribution, true, false),
                               (:vapor_nudge_partner, true, false),
                               (:water_budget_trace, 1, false),
                               (:clamp_water, true, false),
                               # subsumed devices: an explicit true is refused, not bypassed
                               (:ice_anchor_source, true, false),
                               (:ice_anchor_floor, true, false),
                               (:ice_anchor_flux, true, false),
                               (:ice_anchor_rates, true, false),
                               (:mynn_water_closure, true, false),
                               (:mynn_exchange_realization, true, false))
                ex = derived(); ex[k] = v
                @test_throws ErrorException F.vt_tile(tmp, "ref_$(k)"; vtr = :bhyp, extra = ex)
            end
            @test_throws ErrorException F.vt_tile(tmp, "ref_kvw"; vtr = :bhyp, extra = derived(),
                pp_extra = Dict{Symbol,Float64}(:Kvdiff_water => 1.0))
            @test_throws ErrorException F.vt_tile(tmp, "ref_tau"; vtr = :bhyp, extra = derived(),
                pp_extra = Dict{Symbol,Float64}(:tau_rho_v_rec => 10.0))
            # ...an explicit false is harmless
            for k in (:ice_anchor_source, :ice_anchor_floor, :ice_anchor_flux,
                      :ice_anchor_rates, :mynn_water_closure, :mynn_exchange_realization)
                ex = derived(); ex[k] = false
                @test F.vt_tile(tmp, "ok_$(k)"; vtr = :bhyp, extra = ex) isa Tuple
            end
            # the same keys are fine under :slot
            ex = Dict{Symbol,Any}(:water_lobe_attribution => false, :ice_anchor_source => true,
                                  :mynn_water_closure => true)
            @test F.vt_tile(tmp, "ok_slot"; vtr = :bhyp, extra = ex) isa Tuple
        end
    end

    # ── 7. the census rows ──────────────────────────────────────────────────────────
    @testset "census: MC_W_*, MC_SHADOW_* layout and values" begin
        @test Scythe.MC_W_PART == Scythe.MC_VNEG_MASS + 1
        @test Scythe.MC_SHADOW_INT == Scythe.MC_W_PART + 8
        @test Scythe.MC_STATS_LAST == Scythe.MC_SHADOW_INT
        @test length(Scythe.MC_WATER_STATS) == Scythe.MC_STATS_LAST
        @test Scythe.MC_WATER_STATS[Scythe.MC_SHADOW_MAX] == :shadow_max
        mktempdir() do tmp
            for wa in (:slot, :derived)
                ex = wa === :derived ? derived() :
                    Dict{Symbol,Any}(:si_water_slots => :mass, :si_water_mixed_walls => true)
                m, p, mo = F.vt_tile(tmp, "cen_$(wa)"; vtr = :bhyp, extra = ex,
                                     seed! = (p, x, V, c) -> F.vt_seed!(p, x, V, c))
                Scythe.init_water_anchor_shadow!(p, mo)
                m.tile.physical .= p.physical
                eb = m.ebudget; st = m.mc_water_stats
                # an output step: t − 1 a multiple of the output interval (10 s / 0.5 s)
                for t in 2:21
                    Scythe.energy_census_trace(m, t)
                end
                # independent quadrature of the anchor and the partition gap
                kD = eb.kDim; nc = F.vt_ncols(p, mo)
                V = mo.grid_params.vars
                wint = 0.0; xint = 0.0
                for c in 1:nc
                    a = 0.0; x = 0.0
                    for k in 1:kD
                        g = (c - 1) * kD + k
                        W = wa === :derived ? Scythe._eb_W(eb, p.physical, g, 1, k) :
                            (p.physical[g, V["rho_t"], 1] + eb.rho_tbar[k]) -
                            (p.physical[g, V["rho_d"], 1] + eb.rho_dbar[k])
                        a += eb.wz[k] * W
                        x += eb.wz[k] * Scythe._eb_point_RX(eb, p.physical, g, 1, k)[2]
                    end
                    wint += eb.wxn[c] * a; xint += eb.wxn[c] * x
                end
                @test st[Scythe.MC_W_INT, 1] ≈ wint rtol = 1.0e-14
                @test st[Scythe.MC_W_PART, 1] ≈ xint rtol = 1.0e-12
                @test st[Scythe.MC_W_PART, 1] == st[Scythe.MC_XGAP_INT, 1]
                @test st[Scythe.MC_W_SEEN, 1] == 1.0
                if wa === :derived
                    @test st[Scythe.MC_SHADOW_INT, 1] === 0.0
                    @test st[Scythe.MC_SHADOW_MAX, 1] === 0.0
                end
            end
        end
    end

    # ── 8. the R1T1X wall estimate reads the derived perturbation ───────────────────
    @testset "update_mc_wall_bc! reads rho_d' + W' under :derived" begin
        mktempdir() do tmp
            m, p, mo = F.vt_tile(tmp, "wall"; vtr = :bhyp, extra = derived(),
                                 seed! = (p, x, V, c) -> F.vt_seed!(p, x, V, c))
            ws = Scythe.water_anchor_set(mo.options, mo.physical_params, Scythe.mc_slots(mo))
            @test ws.on
            V = mo.grid_params.vars
            # the derived perturbation from the slots, the expression the estimate averages
            g = 1
            W = Scythe.anchor_Wp(p.physical, g, 1, ws)
            rt = Scythe.anchor_rho_tp(p.physical[g, V["rho_d"], 1], W)
            ref = p.physical[g, V["rho_d"], 1] + 2.0 * p.physical[g, V["nu_v"], 1] +
                  2.0 * p.physical[g, V["nu_c"], 1] + 2.0 * p.physical[g, V["nu_r"], 1]
            @test rt ≈ ref rtol = 1.0e-14
            @test Scythe.anchor_rho_tp(-0.0, 0.0) === -0.0
            @test Scythe.anchor_rho_tp(1.0, -0.0) === 1.0
        end
    end

    # ── 9. netCDF derived fields ────────────────────────────────────────────────────
    @testset "netCDF: rho_t is rho_d + rho_w, the shadow is written beside it" begin
        z = collect(0.0:250.0:1000.0)
        nz = length(z)
        zp = zeros(nz)
        ref = Scythe.RegularReferenceProfiles(z, fill(1.0e5, nz), fill(1.1, nz), zp,
                                              fill(1.0e-3, nz), fill(1.11, nz), fill(2.6e5, nz),
                                              zp, fill(290.0, nz))
        mk(f) = [f(i, k) for i in 1:2, k in 1:nz]
        slots = (; p = mk((i, k) -> 10.0 * i), rho_d = mk((i, k) -> 1.0e-3 * k),
                   rho_t = mk((i, k) -> 0.5), E_t = mk((i, k) -> 0.0), Q_ss = mk((i, k) -> 0.0),
                   rho_v = mk((i, k) -> 1.0e-5 * i), u = mk((i, k) -> 0.0),
                   w = mk((i, k) -> 0.0), rain = mk((i, k) -> 2.0e-6),
                   cloud = mk((i, k) -> 3.0e-6 * k))
        base = Scythe.DerivedConfig(:bhyp, 1.0e-7, :bhyp, 1.0e-7, :none, 1.0, :none,
                                    (1.0e-12, 1.0e-2, 1.0e-16, 1.0e-16), 1, false, false,
                                    8.0e6, 100.0, "t", false, :eos, :bhyp, 1.0e-9)
        @test base.water_anchor === :slot
        cfg = Scythe.DerivedConfig(:bhyp, 1.0e-7, :bhyp, 1.0e-7, :none, 1.0, :none,
                                   (1.0e-12, 1.0e-2, 1.0e-16, 1.0e-16), 1, false, false,
                                   8.0e6, 100.0, "t", false, :eos, :bhyp, 1.0e-9, :derived)
        d0 = Scythe.mc_derived_fields(z, slots, ref, base)
        d = Scythe.mc_derived_fields(z, slots, ref, cfg)
        @test !haskey(d0, :rho_t_slot)
        rvb = ref.rho_tbar .- ref.rho_dbar .- ref.rho_cbar
        nvb = Scythe.bhyp.(rvb, 1.0e-9); ncb = Scythe.bhyp.(ref.rho_cbar, 1.0e-7)
        for i in 1:2, k in 1:nz
            rw = 2.0 * (slots.rho_v[i, k] + nvb[k]) + 2.0 * (slots.cloud[i, k] + ncb[k]) +
                 2.0 * slots.rain[i, k]
            @test d.rho_t[i, k] ≈ (slots.rho_d[i, k] + ref.rho_dbar[k]) + rw rtol = 1.0e-14
            wp = 2.0 * slots.rho_v[i, k] + 2.0 * slots.cloud[i, k] + 2.0 * slots.rain[i, k]
            @test d.water_shadow_gap[i, k] ≈ slots.rho_t[i, k] - (slots.rho_d[i, k] + wp) rtol = 1.0e-14
            @test d.rho_t_slot[i, k] ≈ slots.rho_t[i, k] + d.rho_tbar_w[k] rtol = 1.0e-14
        end
        @test d.rho_tbar_w ≈ ref.rho_dbar .+ 2.0 .* nvb .+ 2.0 .* ncb rtol = 1.0e-14
        @test d.rho_w_deficit == min.(d.rho_t .- (slots.rho_d .+ reshape(ref.rho_dbar, 1, nz)), 0.0) ||
              all(d.rho_w_deficit .== 0.0)
    end

    # ══ 3b: the exchange closure (water_exchange_dv and its sites) ══════════════════
    @testset "3b: every exchange block is rho_w-neutral (sum_k alpha_k dnu_k == 0)" begin
        # one scalar identity per block, on the closure every site uses, for the production
        # weights (every slot transformed: alpha = 2) and a mixed set (density vapor)
        for (av, ac, ar, ai) in ((2.0, 2.0, 2.0, 2.0), (1.0, 2.0, 2.0, 2.0), (1.0, 1.0, 1.0, 1.0))
            sumw(dv, dc, dr, d1, d2, d3) = ((((((ac * dc) + (ar * dr)) + (ai * d1)) + (ai * d2)) +
                                            (ai * d3)) + (av * dv))
            blocks = (
                "condensation"  => (3.1e-7, 0.0, 0.0, 0.0, 0.0),
                "rain evap"     => (0.0, -2.7e-8, 0.0, 0.0, 0.0),
                "deposition"    => (0.0, 0.0, 4.4e-9, 1.3e-10, 7.0e-9),
                "cond+evap+dep" => (1.9e-7, -3.3e-8, 2.2e-9, 0.0, -1.1e-9))
            for (name, (dc, dr, d1, d2, d3)) in blocks
                dv = Scythe.water_exchange_dv(av, ac, ar, ai, dc, dr, d1, d2, d3)
                @test sumw(dv, dc, dr, d1, d2, d3) == 0.0
            end
            # the multistep sources: riming/freezing (liquid -> ice), melting (ice -> rain),
            # autoconversion/collection (cloud -> rain), the population warm branch (ice ->
            # rain), the anchor source (ice removed), DeMott nucleation (vapor -> ice)
            Jc, Jr, J1, J2, J3 = 0.5000003, 0.71, 0.5, 0.62, 0.5
            for (name, (acr, ic, ir, s1, s2, s3)) in (
                    "riming"       => (0.0, -2.0e-7, 0.0, 2.0e-7, 0.0, 0.0),
                    "freezing"     => (0.0, 0.0, -5.0e-8, 0.0, 0.0, 5.0e-8),
                    "melting"      => (0.0, 0.0, 3.0e-8, -1.0e-8, -2.0e-8, 0.0),
                    "autoconv"     => (4.0e-8, 0.0, 0.0, 0.0, 0.0, 0.0),
                    "population"   => (0.0, 0.0, 7.0e-9, 0.0, -7.0e-9, 0.0),
                    "anchor"       => (0.0, 0.0, 0.0, -3.0e-9, -1.0e-9, -2.0e-9),
                    "nucleation"   => (0.0, 0.0, 0.0, 1.0e-12, 0.0, 0.0))
                x = Scythe.water_exchange_multistep_rate(av, ac, ar, ai, Jc, Jr, J1, J2, J3,
                                                         acr, ic, ir, s1, s2, s3)
                @test sumw(x, Jc * (ic - acr), Jr * (ir + acr), J1 * s1, J2 * s2, J3 * s3) == 0.0
            end
            # autoconversion above both knees moves no vapor
            @test Scythe.water_exchange_multistep_rate(av, ac, ar, ai, 0.5, 0.5, 0.5, 0.5, 0.5,
                                                       4.0e-8, 0.0, 0.0, 0.0, 0.0, 0.0) == 0.0
            # sedimentation: rho_w loses EXACTLY the fitted flux divergence (plan E8)
            for (Jr_, J1_, Frz, F1z) in ((0.5, 0.5, 2.0e-6, 3.0e-7), (0.83, 0.5000002, -1.0e-6, 0.0),
                                         (1.0, 0.9, 5.0e-7, -2.0e-7))
                xs = Scythe.sedimentation_knee_closure(av, ar, ai, Jr_, J1_, 0.5, 0.5, Frz, F1z,
                                                       0.0, 0.0)
                drw = (ar * (-Jr_ * Frz)) + (ai * (-J1_ * F1z)) + (av * xs)
                @test drw ≈ -(Frz + F1z) rtol = 1.0e-14 atol = 1.0e-22
            end
            @test Scythe.sedimentation_knee_closure(2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5,
                                                    1.0e-6, 2.0e-7, 0.0, 0.0) == 0.0
        end
        @test Scythe.water_exchange_dv(2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0) === -0.0
        # with every alpha = 2 the closure is bitwise the plain nu-sum (doubling/halving exact)
        for dc in (3.0e-7, -1.7e-9), dr in (0.0, 2.2e-8)
            @test Scythe.water_exchange_dv(2.0, 2.0, 2.0, 2.0, dc, dr, 1.0e-9, 0.0, 3.0e-10) ===
                  -((((dc + dr) + 1.0e-9) + 0.0) + 3.0e-10)
        end
    end

    @testset "3b: the ETD stage and the paired Stage L are rho_w-neutral in the model" begin
        # A supersaturated sheet column (nucleation over the lobes, condensation in the sheet,
        # rain falling and evaporating below), one step, then the ETD stage re-applied on the
        # state it started from: sum_k alpha_k dnu_k over the water slots is zero to rounding
        # at every point, the vapor's increment IS the closure (no other write), and with the
        # paired Stage L on the recovered cloud is untouched by the relaxation.
        mktempdir() do tmp
            super! = (p, x, V, c) -> begin
                F.vt_seed!(p, x, V, c; amp = 1.0e-3)
                kD = length(c.Tk)
                for i in 1:size(p.physical, 1)
                    zz = x[i, end]; k = mod1(i, kD)
                    zz > 1500.0 || continue
                    drv = 0.33 * Scythe.rho_v_sat(c.Tk[k], c.p_Pa[k] / 100.0)
                    p.physical[i, V["nu_v"], 1] += Scythe.bhyp(c.rho_v[k] + drv, 1.0e-9) -
                                                    Scythe.bhyp(c.rho_v[k], 1.0e-9)
                    p.physical[i, V["Q_ss"], 1] += drv
                    p.physical[i, V["nu_r"], 1] = Scythe.bhyp(2.0e-4 * exp(-((zz - 800.0) / 500.0)^2), 1.0e-7)
                end
            end
            for ice in (false, true), lobe in (false, true)
                ex = derived(; lobe_relaxation = lobe)
                m, p, mo = F.vt_tile(tmp, "etd_$(ice)_$(lobe)"; vtr = :bhyp, ice, extra = ex,
                                     seed! = super!)
                F.vt_run!(m, p, mo, 3)
                kD = mo.grid_params.kDim; nc = F.vt_ncols(p, mo)
                V = mo.grid_params.vars
                ws = Scythe.water_anchor_set(mo.options, mo.physical_params, Scythe.mc_slots(mo))
                wsl = [s for s in ws.slot[1:6] if s > 0]
                wal = [ws.alpha[j] for j in 1:6 if ws.slot[j] > 0]
                worst = 0.0; scale = 0.0; nz = 0
                rcbits = true
                for c in 1:nc
                    cs = (c - 1) * kD + 1; rows = cs:(cs + kD - 1)
                    Scythe.advance_column(m, c, 4)
                    X0 = copy(m.var_np1)
                    S = m.mc_scratch[Threads.threadid()]
                    m.var_np1 .= X0
                    Scythe.relaxation_adjustment_qss!(m, cs, cs + kD - 1, 4)
                    for g in rows
                        d = [m.var_np1[g, s] - X0[g, s] for s in wsl]
                        sw = sum(wal .* d)
                        worst = max(worst, abs(sw)); scale = max(scale, maximum(abs, wal .* d))
                        any(!=(0.0), d) && (nz += 1)
                        if lobe
                            n0 = X0[g, V["nu_c"]]; n1 = m.var_np1[g, V["nu_c"]]
                            # where the device acted (nu <= 0 before and after) no recovered
                            # cloud moved
                            (n0 < 0.0 && n1 <= 0.0) &&
                                (rcbits &= Scythe.ahyp(n0, 1.0e-7) === Scythe.ahyp(n1, 1.0e-7))
                        end
                    end
                end
                @test nz > 0
                @test worst <= 1.0e-10 * scale
                lobe && @test rcbits
            end
        end
    end

    # ══ 3c: the partners re-derived (plan E2, E4, E7) ══════════════════════════════════
    # R and X per point of a slot matrix (var_np1's layout) through the tile's own budget
    # reading: under the derived anchor the mechanical term reads rho_d + rho_w (E9) and
    # X = rho_w - sum_k rho_k.
    RX(m, X) = begin
        eb = m.ebudget; kD = eb.kDim; n = size(X, 1)
        R = zeros(n); Xg = zeros(n)
        for g in 1:n
            r, x, _ = Scythe._eb_point_RX(eb, X, g, 1, mod1(g, kD))
            R[g] = r; Xg[g] = x
        end
        R, Xg
    end

    @testset "3c E2: the refit partner books the affine defect, energy only" begin
        mktempdir() do tmp
            arms = Dict{Any,Any}()
            for seed in (:sheet, :uniform)
                sd! = seed === :sheet ? (p, x, V, c) -> F.vt_seed!(p, x, V, c; amp = 1.0e-3) :
                    (p, x, V, c) -> begin
                        for i in 1:size(p.physical, 1)
                            p.physical[i, V["nu_c"], 1] = Scythe.condensate_slot(1.0e-3, 0.0,
                                                                                 :bhyp, F.MU)
                        end
                    end
                ex = derived(; refit_partner = true, exchange_realization = false)
                m0, p0, mo0 = F.vt_tile(tmp, "rp_$(seed)"; vtr = :bhyp, extra = ex, seed! = sd!)
                F.vt_run!(m0, p0, mo0, 2)
                # the SAME state into both arms: the partner on and off from here
                for on in (true, false)
                    m = deepcopy(m0); mo = m.model
                    mo.options[:refit_partner] = on
                    pre = copy(m.var_np1)            # the pre-refit state the step reads
                    phys = copy(m.tile.physical)     # the post-refit state n
                    kD = mo.grid_params.kDim
                    Scythe.advance_column(m, 1, 3)
                    S = m.mc_scratch[Threads.threadid()]
                    arms[(on, seed)] = (; m, mo, pre, phys, d1 = copy(S.etd_d1),
                                          dv = copy(S.etd_dv), d8 = copy(S.etd_d8),
                                          d9 = copy(S.etd_d9), q = copy(S.etd_qnp1),
                                          Tk = copy(S.Tk), R_m = copy(S.R_m), C_vt = copy(S.C_vt),
                                          Lv = copy(S.Lv), nu_c = copy(S.nu_c), rows = 1:kD)
                end
            end
            for seed in (:sheet, :uniform)
                a = arms[(true, seed)]; b = arms[(false, seed)]
                # the water increments of the ETD are bitwise those of the unpartnered arm
                @test a.dv == b.dv
                @test a.d8 == b.d8
                @test a.d9 == b.d9
                dp = a.d1 .- b.d1
                V = a.mo.grid_params.vars
                # the defect booked, independently: Delta_c = [ahyp(nu+) - ahyp(nu-)] - 2 dnu
                lam = (a.R_m ./ a.C_vt) .* (a.Lv .- (Scythe.Cpv .* a.Tk))
                npre = a.pre[a.rows, V["nu_c"]]; npost = a.phys[a.rows, V["nu_c"], 1]
                dfc = [Scythe.affine_defect_change(npost[k], npre[k], :bhyp, F.MU)
                       for k in eachindex(npre)]
                full = [Scythe.refit_recovered_change(npost[k], npre[k], :bhyp, F.MU)
                        for k in eachindex(npre)]
                if seed === :sheet
                    # deep lobe points (nu < 0 on both sides, nu moved): the booking is the
                    # defect, -2 dnu, the pool's growth
                    lobe = [k for k in eachindex(npre) if npre[k] < -10 * F.MU &&
                            npost[k] < -10 * F.MU && npre[k] != npost[k]]
                    @test !isempty(lobe)
                    for k in lobe
                        @test dfc[k] ≈ -2.0 * (npost[k] - npre[k]) rtol = 1.0e-10
                    end
                    # above-knee points: the defect is O(mu), the full recovered change is 2 dnu,
                    # so the OLD form would book lambda_l * 2 dnu there (the negative control)
                    hi = [k for k in eachindex(npre) if npre[k] > 1.0e3 * F.MU &&
                          npost[k] > 1.0e3 * F.MU && abs(npost[k] - npre[k]) > 1.0e-9]
                    @test !isempty(hi)
                    for k in hi
                        @test abs(dfc[k]) <= 1.0e-3 * abs(full[k])
                        @test abs(full[k] - 2.0 * (npost[k] - npre[k])) <= 1.0e-3 * abs(full[k])
                    end
                    # the p partner IS the primitive of the defect (cloud is the only
                    # transformed condensate that moved; the vapor rides at kappa)
                    @test maximum(abs, dp) > 0.0
                    for k in lobe
                        @test dp[k] ≈ lam[k] * dfc[k] rtol = 1.0e-3
                    end
                else
                    # a smooth resolved cloud: the partner books <= 1e-3 of lambda_l * drho_l
                    d = [abs(dp[k]) / max(abs(lam[k] * full[k]), 1.0e-300) for k in eachindex(full)
                         if abs(full[k]) > 1.0e-12]
                    @test isempty(d) || maximum(d) <= 1.0e-3
                end
            end
        end
    end

    @testset "energy bisection forms: refit_partner_form = :recovered, refit_partner_vapor = false" begin
        mktempdir() do tmp
            ex = derived(; refit_partner = true, exchange_realization = false)
            m0, p0, mo0 = F.vt_tile(tmp, "bis"; vtr = :bhyp, extra = ex,
                                    seed! = (p, x, V, c) -> F.vt_seed!(p, x, V, c; amp = 1.0e-3))
            F.vt_run!(m0, p0, mo0, 2)
            arm(kv...) = begin
                m = deepcopy(m0); mo = m.model
                for (k, v) in kv
                    mo.options[k] = v
                end
                pre = copy(m.var_np1); phys = copy(m.tile.physical)
                Scythe.advance_column(m, 1, 3)
                S = m.mc_scratch[Threads.threadid()]
                (; mo, pre, phys, d1 = copy(S.etd_d1), dv = copy(S.etd_dv), d9 = copy(S.etd_d9),
                   Tk = copy(S.Tk), R_m = copy(S.R_m), C_vt = copy(S.C_vt), Lv = copy(S.Lv))
            end
            off = arm(:refit_partner => false)
            dfl = arm()
            rec = arm(:refit_partner_form => :recovered)
            nov = arm(:refit_partner_vapor => false)
            V = off.mo.grid_params.vars
            kD = off.mo.grid_params.kDim
            npre = off.pre[1:kD, V["nu_c"]]; npost = off.phys[1:kD, V["nu_c"], 1]
            full = [Scythe.refit_recovered_change(npost[k], npre[k], :bhyp, F.MU) for k in 1:kD]
            dfc = [Scythe.affine_defect_change(npost[k], npre[k], :bhyp, F.MU) for k in 1:kD]
            # every form is energy only: the water increments are the unpartnered arm's
            for a in (dfl, rec, nov)
                @test a.dv == off.dv
                @test a.d9 == off.d9
            end
            # :recovered books the full recovered change at the :slot coefficients
            # (phase_change_partner's dp, the vapor -(dl) in the bracket)
            prc = [Scythe.phase_change_partner(full[k], 0.0, off.Tk[k], off.R_m[k], off.C_vt[k],
                                               off.C_vt[k] + off.R_m[k], off.Lv[k], 0.0, 0.0,
                                               0.0)[1] for k in 1:kD]
            @test maximum(abs, prc) > 0.0
            @test (rec.d1 .- off.d1) ≈ prc rtol = 1.0e-9 atol = 1.0e-12 * maximum(abs, prc)
            # refit_partner_vapor = false: the cloud's defect alone, no kappa Delta_v
            pdc = [Scythe.recovered_change_partner(0.0, dfc[k], 0.0, off.Tk[k], 1.0, off.R_m[k],
                                                   off.C_vt[k], off.Lv[k], 0.0, 0.0, 0.0)[1]
                   for k in 1:kD]
            @test (nov.d1 .- off.d1) ≈ pdc rtol = 1.0e-9 atol = 1.0e-12 * maximum(abs, prc)
            # ...and the default differs from it by the vapor term exactly where nu_v moved
            nvb = Scythe.mc_nu_vbar(m0)
            dvv = [Scythe.affine_defect_change(off.phys[k, V["nu_v"], 1] + nvb[k],
                                               off.pre[k, V["nu_v"]] + nvb[k], :bhyp,
                                               Scythe.vapor_mu(off.mo.physical_params))
                   for k in 1:kD]
            @test any(!=(0.0), dvv)
            @test all(k -> (dvv[k] == 0.0) == (dfl.d1[k] == nov.d1[k]), 1:kD)
            # the form is validated
            bad = derived(; refit_partner_form = :full)
            @test_throws ErrorException F.vt_tile(tmp, "bisbad"; vtr = :bhyp, extra = bad)
        end
        # E-c: the derived R functional on a :slot tile (the same state into both)
        mktempdir() do tmp
            got = Dict{Symbol,Any}()
            for fn in (:model, :derived)
                m, p, mo = F.vt_tile(tmp, "ebf_$(fn)"; vtr = :bhyp,
                                     extra = Dict{Symbol,Any}(:energy_budget => true,
                                                              :energy_budget_functional => fn),
                                     seed! = (p, x, V, c) -> F.vt_seed!(p, x, V, c))
                eb = m.ebudget
                @test !eb.wa
                @test eb.wR == (fn === :derived)
                F.vt_run!(m, p, mo, 4)
                A = m.tile.physical
                g = 7; k = 7
                R, X, _ = Scythe._eb_point_RX(eb, A, g, 1, k)
                rt = A[g, eb.irt, 1] + eb.rho_tbar[k]; rd = A[g, eb.ird, 1] + eb.rho_dbar[k]
                W = Scythe._eb_W(eb, A, g, 1, k)
                geo = 0.5 * (A[g, eb.iu, 1]^2 + A[g, eb.iw, 1]^2) + Scythe.gravity * eb.z[k]
                got[fn] = (; R, X, shift = (rt - (rd + W)) * geo)
                snap = Scythe.energy_budget_snapshot!(m)
                tR = vec(sum(snap.dR[collect(Scythe.EB_TOP), :]; dims = 1))
                @test maximum(abs, snap.actualR .- tR) <= 1.0e-10 * maximum(abs, snap.actualR) +
                                                         1.0e-12 * maximum(abs, snap.dR)
            end
            # the trajectory is the same (the functional is diagnostic only); R differs by the
            # mechanical term alone, X not at all
            @test got[:derived].X == got[:model].X
            @test got[:derived].shift == got[:model].shift
            @test got[:derived].R ≈ got[:model].R + got[:model].shift rtol = 1.0e-12
        end
    end

    @testset "nesting: an injected vapor jump books no refit energy (the vapor re-taken)" begin
        # A parent's collar rows take the child's state between the refit and the step: here
        # ONLY the vapor differs (nu_v raised by 2e-4 on the upper half of the last column).
        # The derived partner books the vapor's own defect from var_np1, so without the vapor
        # re-take the jump is booked at kappa; with it, nothing on those rows, to the bit. With
        # every condensate untransformed the partner still runs (its gate has the vapor), so
        # the early return must not fire.
        mktempdir() do tmp
            for tr in (:bhyp, :none)
                arms = Dict{String,Any}()
                for (tag, on, fix) in (("on", true, true), ("off", false, true),
                                       ("nofix", true, false))
                    m, p, mo = F.vt_tile(tmp, "nvj_$(tr)_$(tag)"; vtr = :bhyp, tr,
                                         extra = derived(; refit_partner = on,
                                                         exchange_realization = false),
                                         seed! = (p, x, V, c) -> F.vt_seed!(p, x, V, c))
                    kD = mo.grid_params.kDim; nc = F.vt_ncols(p, mo)
                    V = mo.grid_params.vars
                    F.vt_run!(m, p, mo, 1)
                    col = ((nc - 1) * kD + 1):(nc * kD)
                    rows = collect(col[(div(kD, 2) + 1):end])
                    vals = copy(p.physical[rows, :, :])
                    vals[:, V["nu_v"], 1] .+= 2.0e-4
                    p.physical[rows, :, :] .= vals
                    inj = [(rows, vals)]
                    fix && Scythe.refit_reference_after_exchange!(m, inj, 2)
                    fix && on && @test m.var_np1[rows, V["nu_v"]] == vals[:, V["nu_v"], 1]
                    for c in 1:nc
                        Scythe.advance_column(m, c, 2)
                    end
                    arms[tag] = copy(m.mc_scratch[Threads.threadid()].etd_d1[1:kD])
                end
                kD = length(arms["on"]); cz = (div(kD, 2) + 1):kD
                on = arms["on"] .- arms["off"]; nofix = arms["nofix"] .- arms["off"]
                @test all(on[cz] .== 0.0)
                @test maximum(abs, nofix[cz]) > 0.0
                @test on[1:div(kD, 2)] == nofix[1:div(kD, 2)]
            end
        end
    end

    @testset "B2: the multistep closure books the REALIZED recovered change (knees)" begin
        # One Euler step of exactly the multistep exchange at a knee, assembled by hand from
        # the closure's own tendency (vapor nu, p): autoconversion from an above-knee cloud
        # into empty rain (J_r at nu_r = 0), and -- on an ice tile -- a deposition source into
        # empty ice (J_i at nu_i = 0). R is held to second order; the requested-mass form
        # (no partner for AC, phase_change_partner(0, s) for the ice) opens R by
        # kappa * (recovered vapor - booked vapor).
        mktempdir() do tmp
            for ice in (false, true)
                m, p, mo = F.vt_tile(tmp, "b2_$(ice)"; vtr = :bhyp, ice,
                                     extra = derived(; refit_partner = false),
                                     seed! = (p, x, V, c) -> F.vt_seed!(p, x, V, c; amp = 2.0e-3))
                F.vt_run!(m, p, mo, 2)
                kD = mo.grid_params.kDim; V = mo.grid_params.vars
                Scythe.advance_column(m, 1, 3)
                S = m.mc_scratch[Threads.threadid()]
                IS = m.mc_slots
                phys = m.tile.physical
                X0 = copy(phys[:, :, 1])
                nvb = Scythe.mc_nu_vbar(m)
                ts = mo.ts
                for f in (:AUTO_COLL, :ICE_C, :ICE_R, :Fr_z)
                    fill!(getfield(S, f), 0.0)
                end
                if ice
                    for f in (:SRC_i1q, :SRC_i2q, :SRC_i3q, :F_i1q_z, :F_i2q_z, :F_i3q_z)
                        fill!(getfield(S, f), 0.0)
                    end
                end
                rate = 1.0e-6
                cases = ice ? (:ac, :dep) : (:ac,)
                for case in cases
                    # AC: an above-knee cloud point with (nearly) empty rain; dep: empty ice
                    ks = case === :ac ?
                        [k for k in 1:kD if S.nu_c[k] > 1.0e3 * F.MU &&
                                            abs(X0[k, V["nu_r"]]) <= 10 * F.MU] :
                        [k for k in 1:kD if X0[k, IS.rho_v] + nvb[k] > 1.0e-4]
                    @test !isempty(ks)
                    isempty(ks) && continue
                    k = ks[1]
                    ex = zeros(size(m.expdot_n))
                    fill!(S.AUTO_COLL, 0.0)
                    ice && fill!(S.SRC_i1q, 0.0)
                    if case === :ac
                        S.AUTO_COLL[k] = rate
                    else
                        # empty the species at the point: slot, scratch total, its Jacobian
                        imu = Scythe.ice_mu(mo.physical_params, 1)
                        X0[k, IS.i1_q] = 0.0; S.nu_i1q[k] = 0.0
                        S.J_i1q[k] = Scythe.dbhyp(0.0, imu)
                        S.SRC_i1q[k] = rate
                    end
                    Scythe._water_exchange_multistep!(ex, phys, S, 1, IS.rho_v, mo.options,
                        mo.physical_params, IS, ice, true, nvb, :bhyp,
                        Scythe.vapor_mu(mo.physical_params), ts)
                    X1 = copy(X0)
                    X1[k, IS.rho_v] += ts * ex[k, IS.rho_v]
                    if case === :ac
                        X1[k, 9] += ts * (-(S.Jc[k] * rate)); X1[k, 8] += ts * (S.Jr[k] * rate)
                    else
                        X1[k, IS.i1_q] += ts * (S.J_i1q[k] * rate)
                    end
                    Xn = copy(X1); Xo = copy(X1)
                    Xn[k, 1] += ts * ex[k, 1]
                    # the requested-mass booking
                    bv = 0.0
                    if case === :dep
                        dpo, _, _ = Scythe.phase_change_partner(0.0, ts * rate, S.Tk[k], S.R_m[k],
                            S.C_vt[k], S.C_pt[k], S.Lv[k], Scythe.L_s(S.Tk[k]), S.Q_s[k],
                            S.Q_s_i[k])
                        Xo[k, 1] += dpo
                        bv = -ts * rate
                    end
                    # the closure's own anchor identity: sum alpha * (the exchange's nu rates) = 0
                    nur = case === :ac ? (2.0 * (-(S.Jc[k] * rate)) + 2.0 * (S.Jr[k] * rate)) :
                                         2.0 * (S.J_i1q[k] * rate)
                    @test abs(2.0 * ex[k, IS.rho_v] + nur) <= 1.0e-15 * abs(nur)
                    rv0 = Scythe.ahyp(X0[k, IS.rho_v] + nvb[k], Scythe.vapor_mu(mo.physical_params))
                    rv1 = Scythe.ahyp(X1[k, IS.rho_v] + nvb[k], Scythe.vapor_mu(mo.physical_params))
                    rec = rv1 - rv0
                    # the knee: AC moves the recovered vapor by ~-AC dt, the deposition by ~-2s dt
                    @test rec ≈ (case === :ac ? -1.0 : -2.0) * ts * rate rtol = 0.2
                    R0, _ = RX(m, X0); Rn, _ = RX(m, Xn); Ro, _ = RX(m, Xo)
                    kap = S.Tk[k] * (Scythe.Rv - (S.R_m[k] / S.C_vt[k]) * Scythe.Cvv)
                    gap = abs(kap * (rec - bv))
                    @test gap > 0.0
                    @test abs(Rn[k] - R0[k]) <= 1.0e-3 * gap
                    @test abs(Ro[k] - R0[k]) > 0.5 * gap
                end
            end
        end
    end

    @testset "B3: the sfc row mirrors the derived closures (Louis, MYNN; bhyp vapor)" begin
        # Under the derived anchor both closures put F_q g/alpha_v on the vapor's nu (the
        # shadow slot 3 F_q g); the budget hook moves exactly that out of the bl row. The
        # sfc row's R is second order (energy-consistent surface exchange), the budget closes.
        mktempdir() do tmp
            for (tag, bl, ice) in (("louis", :louis, false), ("mynn", :mynn, false),
                                   ("mynn_ice", :mynn, true))
                ex = derived(; energy_budget = true, surface_fluxes = true)
                bl === :mynn && (ex[:mynn] = true)
                m, p, mo = F.vt_tile(tmp, "b3_$(tag)"; vtr = :bhyp, ice, louis = bl === :louis,
                                     extra = ex,
                                     seed! = (p, x, V, c) -> F.vt_seed!(p, x, V, c))
                eb = m.ebudget
                @test eb.wa && eb.mynn == (bl === :mynn)
                F.vt_run!(m, p, mo, 3)
                kD = mo.grid_params.kDim
                tn = eb.tn
                g = 1   # the ground point of the first column: inside the delivery layer
                sq = tn[g, eb.jrt, Scythe.EB_OP_SFC]
                @test sq != 0.0
                @test tn[g, eb.jrv, Scythe.EB_OP_SFC] == sq / 2.0
                snap = Scythe.energy_budget_snapshot!(m)
                tR = vec(sum(snap.dR[collect(Scythe.EB_TOP), :]; dims = 1))
                @test maximum(abs, snap.actualR .- tR) <= 1.0e-10 * maximum(abs, snap.actualR) +
                                                         1.0e-12 * maximum(abs, snap.dR)
                rows = Scythe.energy_budget_rows([snap])
                sf = rows[findfirst(r -> r.stage == "explicit:sfc", rows)].dR_dt_Wm2
                sd = m.surface
                fq = maximum(abs, sd.F_q); fsh = maximum(abs, sd.F_sh)
                @test fq > 0.0
                @test abs(sf) < 1.0e-3 * (fsh + 2.5e6 * fq)
            end
        end
    end

    @testset "D1: the four ice-anchor legs are inert under :derived (census = shadow gap)" begin
        # An ice tile whose vapor rings into a lobe beside real ice: read through
        # rho_t - rho_d = W the headroom test fires there (rho_v + X < 0). Under :derived
        # nothing binds: the reader cap, the sedimentation and rate shares are the raw ice,
        # nothing is removed, and MC_ANCHOR_GAP reads the shadow gap; the :slot arm on the
        # same seed still exercises the legs.
        mktempdir() do tmp
            seedice! = (p, x, V, c) -> begin
                F.vt_seed!(p, x, V, c; amp = 1.0e-3, vamp = 3.0e-3)
                kD = length(c.Tk)
                for i in 1:size(p.physical, 1)
                    zz = x[i, end]
                    # ice in a layer where the vapor bump's flank is sharp; a vapor lobe there
                    if 2500.0 < zz < 4500.0
                        p.physical[i, V["nu_i1"], 1] = Scythe.bhyp(2.0e-4, 1.0e-9)
                        p.physical[i, V["nu_v"], 1] -= 1.5 *
                            Scythe.bhyp(c.rho_v[mod1(i, kD)], 1.0e-9)
                    end
                end
            end
            for wa in (:derived, :slot)
                ex = wa === :derived ? derived() : Dict{Symbol,Any}()
                m, p, mo = F.vt_tile(tmp, "d1_$(wa)"; vtr = :bhyp, ice = true, extra = ex,
                                     seed! = seedice!)
                F.vt_run!(m, p, mo, 3)
                st = m.mc_water_stats
                kD = mo.grid_params.kDim
                Scythe.advance_column(m, 1, 4)
                S = m.mc_scratch[Threads.threadid()]
                if wa === :derived
                    @test all(==(1.0), S.anchor_f)
                    @test S.rho_ice_t == S.rho_ice
                    @test sum(view(st, Scythe.MC_ANCHOR_PTS, :)) == 0.0
                    @test sum(view(st, Scythe.MC_ANCHOR_REMOVED, :)) == 0.0
                    G = maximum(abs, m.tile.physical[1:kD, 3, 1] .- S.res_rho_t)
                    @test maximum(view(st, Scythe.MC_ANCHOR_GAP, :)) >= G
                    # ...and the headroom test the legs WOULD read fires here (the measure)
                    hit = count(i -> S.rho_ice[i] > Scythe.anchor_headroom(S.rho_t[i] - S.rho_d[i],
                                                                           S.rho_liq[i], 0.0),
                                1:kD)
                    @test hit > 0
                else
                    @test sum(view(st, Scythe.MC_ANCHOR_PTS, :)) >= 0.0
                end
            end
        end
    end

    @testset "D5: Louis under :derived mixes the rain in nu space (MYNN's leg set)" begin
        # A sheared column with cloud, a rain bump and the vapor bump: the Louis arm against
        # the BL-off arm on the same state. The legs sum to the anchor (sum alpha D_k equals the
        # shadow's D_3), the rain leg is nonzero, and R is held to 1e-4 of the energy the legs
        # move; with K = 0 the rain slot is untouched bitwise.
        mktempdir() do tmp
            for shear in (true, false)
                seedr! = (p, x, V, c) -> begin
                    F.vt_seed!(p, x, V, c)
                    for i in 1:size(p.physical, 1)
                        zz = x[i, end]; k = mod1(i, length(c.Tk)); T = c.Tk[k]
                        rr = 3.0e-4 * exp(-((zz - 1200.0) / 500.0)^2)
                        p.physical[i, V["nu_r"], 1] = Scythe.rain_slot(rr, :bhyp, F.MU)
                        p.physical[i, V["rho_t"], 1] += rr
                        p.physical[i, V["E_t"], 1] += rr * ((Scythe.Cpv * T) - Scythe.L_v(T) +
                                                            (Scythe.gravity * zz))
                        shear && (p.physical[i, V["u"], 1] = 10.0 * tanh(zz / 800.0))
                    end
                end
                arms = Dict{Bool,Any}()
                # (K = 0 exactly through l_inf = 0: the mixing length is 0 at every level)
                m0, p0, mo0 = F.vt_tile(tmp, "d5_$(shear)"; vtr = :bhyp, louis = true,
                                        sfc = false, extra = derived(), seed! = seedr!,
                                        pp_extra = Dict{Symbol,Float64}(:l_inf => shear ? 80.0 : 0.0))
                F.vt_run!(m0, p0, mo0, 2)
                for louis in (true, false)
                    m = deepcopy(m0)
                    m.model.options[:louis_bl] = louis
                    for c in 1:F.vt_ncols(p0, mo0)
                        Scythe.advance_column(m, c, 3)
                    end
                    arms[louis] = (; m, mo = m.model, x = copy(m.tile.physical[:, :, 1]),
                                   e = copy(m.expdot_n))
                end
                mo = arms[true].mo; V = mo.grid_params.vars; ts = mo.ts
                D = arms[true].e .- arms[false].e
                if !shear
                    @test all(==(0.0), D[:, V["nu_r"]])
                    @test reinterpret(UInt64, arms[true].e[:, V["nu_r"]]) ==
                          reinterpret(UInt64, arms[false].e[:, V["nu_r"]])
                    continue
                end
                @test maximum(abs, D[:, V["nu_r"]]) > 0.0
                sumnu = (2.0 .* D[:, V["nu_v"]] .+ 2.0 .* D[:, V["nu_c"]]) .+ 2.0 .* D[:, V["nu_r"]]
                @test maximum(abs, sumnu .- D[:, 3]) <= 1.0e-12 * maximum(abs, D[:, 3])
                R0, _ = RX(arms[true].m, arms[true].x)
                x1 = copy(arms[true].x); x1 .+= ts .* D[:, 1:size(x1, 2)]
                R1, _ = RX(arms[true].m, x1)
                Es = ts .* abs.(D[:, V["E_t"]])
                @test maximum(abs, R1 .- R0) <= 1.0e-4 * maximum(Es)
            end
        end
    end

    @testset "D6: a load re-zeroes the shadow and prints the gap it discards" begin
        mktempdir() do tmp
            m, p, mo = F.vt_tile(tmp, "d6"; vtr = :bhyp, extra = derived(),
                                 seed! = (p, x, V, c) -> F.vt_seed!(p, x, V, c))
            F.vt_run!(m, p, mo, 2)
            # a converted checkpoint's slot 3: the derived total plus a known gap
            Scythe.init_water_anchor_shadow!(p, mo; report = false)
            p.physical[:, 3, 1] .+= 2.5e-6
            p.physical[5, 3, 1] += 1.0e-5
            out = mktemp() do path, io
                redirect_stdout(io) do
                    Scythe.init_water_anchor_shadow!(p, mo)
                end
                flush(io); read(path, String)
            end
            @test occursin("pre-zero shadow gap", out)
            mx = parse(Float64, match(r"max\|G\| = ([-0-9.eE+]+)", out).captures[1])
            mn = parse(Float64, match(r"mean G = ([-0-9.eE+]+)", out).captures[1])
            @test mx ≈ 1.25e-5 rtol = 1.0e-3
            @test mn ≈ 2.5e-6 + 1.0e-5 / size(p.physical, 1) rtol = 1.0e-3
            # ...and after it the gap is the rounding of one transform
            ws = Scythe.water_anchor_set(mo.options, mo.physical_params, Scythe.mc_slots(mo))
            G = [p.physical[g, 3, 1] - Scythe.anchor_rho_tp(p.physical[g, 2, 1],
                                                            Scythe.anchor_Wp(p.physical, g, 1, ws))
                 for g in axes(p.physical, 1)]
            @test maximum(abs, G) <= 1.0e-12
            # :slot: a no-op, nothing printed
            m2, p2, mo2 = F.vt_tile(tmp, "d6s"; vtr = :bhyp)
            x0 = copy(p2.physical)
            out2 = mktemp() do path, io
                redirect_stdout(io) do
                    Scythe.init_water_anchor_shadow!(p2, mo2)
                end
                flush(io); read(path, String)
            end
            @test isempty(out2) && p2.physical == x0
        end
    end

    @testset "review: E2 in the model -- the full-recovered form opens R at above-knee edges" begin
        # One step from the same post-refit state with the partner in its default (defect)
        # form and in the full-recovered form (`refit_partner_form = :recovered`, the
        # bisection's arm E-a), each against the partner OFF. R is evaluated across the
        # refit and the partner together, from the pre-refit var_np1 to the post-refit state
        # with the partner's p applied: at above-knee cloud points the defect form leaves R
        # where the refit put it to 1e-3 of lambda_l 2 dnu, while the recovered form opens it
        # by lambda_l 2 dnu.
        mktempdir() do tmp
            ex = derived(; refit_partner = true, exchange_realization = false)
            m0, p0, mo0 = F.vt_tile(tmp, "e2r"; vtr = :bhyp, extra = ex,
                                    seed! = (p, x, V, c) -> F.vt_seed!(p, x, V, c; amp = 1.0e-3))
            F.vt_run!(m0, p0, mo0, 2)
            arm(kv...) = begin
                m = deepcopy(m0)
                for (k, v) in kv
                    m.model.options[k] = v
                end
                pre = copy(m.var_np1); phys = copy(m.tile.physical[:, :, 1])
                Scythe.advance_column(m, 1, 3)
                S = m.mc_scratch[Threads.threadid()]
                (; m, pre, phys, d1 = copy(S.etd_d1), Tk = copy(S.Tk), R_m = copy(S.R_m),
                   C_vt = copy(S.C_vt), Lv = copy(S.Lv))
            end
            # (the vapor's own kappa term off in both, so the two forms differ by the condensate)
            off = arm(:refit_partner => false); dfl = arm(:refit_partner_vapor => false)
            rec = arm(:refit_partner_form => :recovered)
            V = mo0.grid_params.vars; kD = mo0.grid_params.kDim
            R0, _ = RX(off.m, off.pre)
            Rafter(a) = begin
                X = copy(a.phys); X[1:kD, 1] .+= (a.d1 .- off.d1); first(RX(off.m, X))
            end
            Rd = Rafter(dfl); Rr = Rafter(rec); Rn = first(RX(off.m, off.phys))
            npre = off.pre[1:kD, V["nu_c"]]; npost = off.phys[1:kD, V["nu_c"]]
            # lambda_l in R units: the energy of condensate appearing alone, L_v - C_pv T
            lam = off.Lv .- (Scythe.Cpv .* off.Tk)
            hi = [k for k in 1:kD if npre[k] > 1.0e3 * F.MU && npost[k] > 1.0e3 * F.MU &&
                  abs(npost[k] - npre[k]) > 1.0e-9]
            @test !isempty(hi)
            for k in hi
                full = Scythe.refit_recovered_change(npost[k], npre[k], :bhyp, F.MU)
                dfc = Scythe.affine_defect_change(npost[k], npre[k], :bhyp, F.MU)
                scale = abs(lam[k] * 2.0 * (npost[k] - npre[k]))
                # the recovered form opens R by ~lambda_l 2 dnu relative to the defect form (its
                # coefficient carries the vapor's -(dl) as well: the :slot bracket, R units)
                lamf = lam[k] - off.Tk[k] * ((off.C_vt[k] * Scythe.Rv / off.R_m[k]) - Scythe.Cvv)
                @test abs((Rr[k] - Rd[k]) + ((lamf * full) - (lam[k] * dfc))) <= 2.0e-3 * scale
                @test abs(Rr[k] - Rd[k]) > 0.5 * scale
                # ...and the defect form books only O(mu) there, R where the refit left it
                @test abs(Rd[k] - Rn[k]) <= 1.0e-3 * scale
            end
            @test abs(Rn[1] - R0[1]) >= 0.0
        end
    end

    @testset "review: the paired Stage L's negative control books the latent coefficient" begin
        # The paired lobe fill moves nu_c up by dc and nu_v down by dc; the recovered cloud
        # stays 0 and the recovered vapor falls by rec ~ 2dc, partnered at kappa. Booking the
        # latent coefficient instead (as if the recovered vapor had condensed) opens R by
        # > 0.5 lambda_l |rec| at every moved point; the kappa form holds it to 1e-4 of that.
        mktempdir() do tmp
            m, p, mo = F.vt_tile(tmp, "slneg"; vtr = :bhyp, extra = derived(; lobe_relaxation = true),
                                 seed! = (p, x, V, c) -> F.vt_seed!(p, x, V, c))
            F.vt_run!(m, p, mo, 3)
            Scythe.advance_column(m, 1, 4)
            S = m.mc_scratch[Threads.threadid()]
            IS = m.mc_slots; kD = mo.grid_params.kDim
            X0 = copy(m.var_np1)
            Xb = copy(X0)
            ws = Scythe.water_anchor_set(mo.options, mo.physical_params, IS)
            nvb = Scythe.mc_nu_vbar(m); vmu = Scythe.vapor_mu(mo.physical_params)
            Scythe._lobe_relax_paired!(Xb, m.tile.physical, S, 1, IS.rho_v, 0, 0, 0, false, ws,
                                       nvb, :bhyp, vmu, Float64[], false)
            moved = [k for k in 1:kD if Xb[k, 9] != X0[k, 9]]
            @test !isempty(moved)
            R0, _ = RX(m, X0); Rb, _ = RX(m, Xb)
            for k in moved
                rec = Scythe.ahyp(Xb[k, IS.rho_v] + nvb[k], vmu) - Scythe.ahyp(X0[k, IS.rho_v] + nvb[k], vmu)
                dpk, _ = Scythe.recovered_change_partner(rec, 0.0, 0.0, S.Tk[k], S.rho_d[k], S.R_m[k],
                    S.C_vt[k], S.Lv[k], 0.0, S.drvs_dT[k], S.drvs_dp[k])
                dpl, _, _ = Scythe.phase_change_partner(-rec, 0.0, S.Tk[k], S.R_m[k], S.C_vt[k],
                    S.C_pt[k], S.Lv[k], 0.0, S.Q_s[k], 0.0)
                Xn = copy(Xb); Xn[k, 1] += dpl - dpk
                Rn, _ = RX(m, Xn)
                lam = (S.R_m[k] / S.C_vt[k]) * (S.Lv[k] - Scythe.Cpv * S.Tk[k])
                @test abs(rec) > 0.0
                @test abs(Rb[k] - R0[k]) <= 1.0e-4 * abs(lam * rec)
                @test abs(Rn[k] - R0[k]) > 0.5 * abs(lam * rec)
            end
        end
    end

    @testset "review: _qss_share! (lambda != 0 rides etd_qnp1, lambda = 0 the tendency)" begin
        S = (; etd_lam = [0.3, 0.0], etd_qnp1 = [1.0, 2.0])
        ex = zeros(2, 8)
        Scythe._qss_share!(ex, S, 1, 1, 5.0, 0.5)
        @test S.etd_qnp1[1] == 1.0 + 2.5 && ex[1, 7] == 0.0
        Scythe._qss_share!(ex, S, 2, 2, 5.0, 0.5)
        @test S.etd_qnp1[2] == 2.0 && ex[2, 7] == 5.0
    end

    @testset "review: P3 closes against an independent R, the closures on their own row" begin
        # Warm rain (autoconversion, sedimentation) and ice: sum of the top-level rows per
        # column against R(end) - R(start) computed here from the states themselves (state n
        # before the first step, var_np1 after the last), not from the budget's own marks;
        # the multistep closures are the explicit:wexch row, nonzero under :derived.
        mktempdir() do tmp
            for ice in (false, true)
                m, p, mo = F.vt_tile(tmp, "p3i_$(ice)"; vtr = :bhyp, ice,
                                     extra = derived(; energy_budget = true, precipitation = true),
                                     seed! = (p, x, V, c) -> F.vt_seed!(p, x, V, c; amp = 2.0e-3))
                eb = m.ebudget; kD = eb.kDim; nc = F.vt_ncols(p, mo)
                colR(A) = [first(Scythe._eb_column_RX(eb, A, (c - 1) * kD, 1)) for c in 1:nc]
                Rs = colR(p.physical)
                for t in 1:6
                    for c in 1:nc
                        Scythe.advance_column(m, c, t)
                    end
                    t < 6 && (Scythe.calcTendency(m); gridTransform!(p))
                end
                Re = colR(m.var_np1)
                snap = Scythe.energy_budget_snapshot!(m)
                tR = vec(sum(snap.dR[collect(Scythe.EB_TOP), :]; dims = 1))
                @test maximum(abs, tR .- (Re .- Rs)) <= 1.0e-10 * maximum(abs, Re .- Rs) +
                                                      1.0e-12 * maximum(abs, snap.dR)
                wx = snap.dR[Scythe.EB_OP_ROW[Scythe.EB_OP_WEXCH], :]
                @test any(!=(0.0), view(eb.tnm1, :, :, Scythe.EB_OP_WEXCH))
                @test all(isfinite, wx)
            end
        end
    end

    @testset "review: E7 MYNN with bhyp vapor, surface fluxes and ice -- R held, legs = anchor" begin
        mktempdir() do tmp
            arms = Dict{Bool,Any}()
            for mynn in (true, false)
                ex = derived()
                mynn && (ex[:mynn] = true; ex[:surface_fluxes] = true)
                m, p, mo = F.vt_tile(tmp, "e7i_$(mynn)"; vtr = :bhyp, ice = true, extra = ex,
                                     seed! = (p, x, V, c) -> begin
                                         F.vt_seed!(p, x, V, c)
                                         for i in 1:size(p.physical, 1)
                                             zz = x[i, end]
                                             p.physical[i, V["u"], 1] = 10.0 * tanh(zz / 800.0)
                                             (3000.0 < zz < 6000.0) &&
                                                 (p.physical[i, V["nu_i1"], 1] = Scythe.bhyp(1.0e-4, 1.0e-9))
                                         end
                                     end)
                Scythe.init_water_anchor_shadow!(p, mo; report = false)
                for c in 1:F.vt_ncols(p, mo)
                    Scythe.advance_column(m, c, 1)
                end
                arms[mynn] = (; m, mo, x = copy(p.physical[:, :, 1]), e = copy(m.expdot_n))
            end
            mo = arms[true].mo; V = mo.grid_params.vars; IS = arms[true].m.mc_slots
            nf = size(arms[false].e, 2)
            D = arms[true].e[:, 1:nf] .- arms[false].e
            # the legs sum to the anchor's tendency (the shadow), ice mass legs included
            sumnu = 2.0 .* (((D[:, IS.rho_v] .+ D[:, 9]) .+ D[:, 8]) .+
                            ((D[:, IS.i1_q] .+ D[:, IS.i2_q]) .+ D[:, IS.i3_q]))
            @test maximum(abs, D[:, IS.i1_q]) > 0.0
            @test maximum(abs, sumnu .- D[:, 3]) <= 1.0e-12 * maximum(abs, D[:, 3])
            x = arms[true].x[:, 1:nf]
            R0, _ = RX(arms[true].m, x)
            x1 = copy(x); x1 .+= mo.ts .* D
            R1, _ = RX(arms[true].m, x1)
            Es = mo.ts .* abs.(D[:, V["E_t"]])
            @test maximum(abs, R1 .- R0) <= 1.0e-4 * maximum(Es)
        end
    end

    @testset "review: MC_W_SRC / MC_W_BUDGET after real steps (Louis surface flux, rain)" begin
        mktempdir() do tmp
            m, p, mo = F.vt_tile(tmp, "wsrc"; vtr = :bhyp, louis = true, sfc = true,
                                 extra = derived(; precipitation = true),
                                 seed! = (p, x, V, c) -> F.vt_seed!(p, x, V, c; amp = 2.0e-3))
            st = m.mc_water_stats
            out = mktemp() do path, io
                redirect_stdout(io) do
                    for t in 1:41
                        Scythe.energy_census_trace(m, t)
                        for c in 1:F.vt_ncols(p, mo)
                            Scythe.advance_column(m, c, t)
                        end
                        Scythe.calcTendency(m); gridTransform!(p)
                    end
                    Scythe.energy_census_trace(m, 42)
                end
                flush(io); read(path, String)
            end
            lines = filter(l -> occursin("energy census", l), split(out, '\n'))
            @test length(lines) >= 2
            getv(l, k) = parse(Float64, match(Regex(k * "=([-0-9.eE+]+)"), l).captures[1])
            src = getv(lines[end], "src"); bud = getv(lines[end], "budget")
            # the surface flux is real; the interval's water budget closes to a few percent of
            # it (the transport's O(mu V) and the refit of rho_w), never by the source's size
            @test src > 0.0
            @test abs(bud) <= 0.05 * src
        end
    end

    @testset "review: the three production wall families pass validate_water_anchor" begin
        for case in ("tc_axisym", "ocean_2D_rce", "ocean_warm_bubble")
            path = joinpath(@__DIR__, "..", "simulations", case, "scythe_config.toml")
            mktempdir() do tmp
                cfg = Scythe.load_config(path; set = ["output.dir=" * tmp])
                @test cfg.numerics.water_anchor === :derived
                mp = Scythe.build_model(cfg)
                models = mp isa Scythe.NestedModelParameters ? first(Scythe.build_nest(mp)) : [mp]
                @test !isempty(models)
                for md in models
                    @test Scythe.validate_si_water_slots(md) === nothing
                    @test Scythe.validate_water_anchor(md) === nothing
                end
            end
        end
    end

    @testset "3c E4: the exchange realization closes on the recovered change (vapor as closure)" begin
        # Supersaturated air over a ringing sheet: condensation is requested at lobe points
        # (recovered cloud 0). One ETD stage re-applied on the post-explicit state: at a deep
        # lobe the cloud stays at recovered 0, the vapor's recovered change is ~ -2d (d the
        # ETD's cloud increment), the net p move is kappa * rec_v (no latent heat) to 1e-8,
        # sum_k alpha_k dnu_k is unchanged by the realization, and R is held.
        mktempdir() do tmp
            super! = (p, x, V, c) -> begin
                F.vt_seed!(p, x, V, c; amp = 1.0e-3)
                kD = length(c.Tk)
                for i in 1:size(p.physical, 1)
                    zz = x[i, end]; k = mod1(i, kD)
                    zz > 1500.0 || continue
                    drv = 0.33 * Scythe.rho_v_sat(c.Tk[k], c.p_Pa[k] / 100.0)
                    p.physical[i, V["nu_v"], 1] += Scythe.bhyp(c.rho_v[k] + drv, 1.0e-9) -
                                                    Scythe.bhyp(c.rho_v[k], 1.0e-9)
                    p.physical[i, V["Q_ss"], 1] += drv
                end
            end
            for lt in (true, false)
                ex = derived(; water_lobe_transport = lt, refit_partner = false)
                m, p, mo = F.vt_tile(tmp, "xr_$(lt)"; vtr = :bhyp, extra = ex, seed! = super!)
                F.vt_run!(m, p, mo, 3)
                kD = mo.grid_params.kDim; V = mo.grid_params.vars
                found = 0
                for c in 1:F.vt_ncols(p, mo)
                    cs = (c - 1) * kD + 1; rows = cs:(cs + kD - 1)
                    Scythe.advance_column(m, c, 4)
                    S = m.mc_scratch[Threads.threadid()]
                    # the post-explicit state: undo this column's ETD stage is not possible,
                    # so re-apply it on the stage's input reconstructed from the increments
                    X1 = copy(m.var_np1)
                    X0 = copy(X1)
                    m.var_np1 .= X0
                    mo.options[:exchange_realization] = false
                    Scythe.relaxation_adjustment_qss!(m, cs, cs + kD - 1, 4)
                    Xa = copy(m.var_np1)
                    m.var_np1 .= X0
                    mo.options[:exchange_realization] = true
                    Scythe.relaxation_adjustment_qss!(m, cs, cs + kD - 1, 4)
                    Xb = copy(m.var_np1)
                    m.var_np1 .= X1
                    Ra, _ = RX(m, Xa); Rb, _ = RX(m, Xb); R0, _ = RX(m, X0)
                    nvb = Scythe.mc_nu_vbar(m)
                    for (k, g) in enumerate(rows)
                        d = S.etd_d9[k]
                        nstar = X0[g, V["nu_c"]]
                        (d > 0.0 && nstar + d < -10 * F.MU) || continue
                        found += 1
                        # the Jacobian the closure uses at the clipped point is the ETD's own
                        J = S.Jc[k]
                        Jexp = lt ? Scythe.dbhyp(Scythe.ahyp_smooth(-(nstar - 0.0), F.MU), F.MU) :
                                    1.0
                        @test d ≈ mo.ts * J * S.Qdot_bar[k] rtol = 1.0e-14
                        lt || @test J == 1.0
                        # the closure: the vapor's nu moved by -d (alpha_c = alpha_v = 2)
                        @test S.etd_dv[k] == -d
                        # the realization moved no water slot
                        for s in (V["nu_v"], V["nu_c"], V["nu_r"])
                            @test Xb[g, s] == Xa[g, s]
                        end
                        # recovered cloud stays 0; recovered vapor moved by ~ -2d
                        @test Scythe.ahyp(Xb[g, V["nu_c"]], F.MU) == 0.0
                        rv0 = Scythe.ahyp(X0[g, V["nu_v"]] + nvb[k], 1.0e-9)
                        rv1 = Scythe.ahyp(Xb[g, V["nu_v"]] + nvb[k], 1.0e-9)
                        @test (rv1 - rv0) ≈ -2.0 * d rtol = 1.0e-6
                        # net p (ETD + realization): kappa * rec_v, no latent heat
                        kap = S.Tk[k] * (Scythe.Rv - (S.R_m[k] / S.C_vt[k]) * Scythe.Cvv)
                        @test (Xb[g, 1] - X0[g, 1]) ≈ kap * (rv1 - rv0) rtol = 1.0e-8
                        # R held: |dR| <= 1e-3 L_v |req| (the unrealized arm opens ~L_v req)
                        req = mo.ts * S.Qdot_bar[k]
                        @test abs(Rb[g] - R0[g]) <= 1.0e-3 * S.Lv[k] * abs(req)
                        @test abs(Ra[g] - R0[g]) > 0.3 * S.Lv[k] * abs(req)
                    end
                end
                @test found > 0
            end
        end
    end

    @testset "3c E7: MYNN under the derived anchor — nu legs, div_w = sum, R held" begin
        MF = Main.MYNNTestFixtures
        mktempdir() do tmp
            function seed!(patch, mo, kDim, z, col; e0 = 0.4)
                V = mo.grid_params.vars
                c = Scythe.mc_slot(V, "rho_c"); r = Scythe.mc_slot(V, "rho_r")
                u = MF.bl_jet(z, 12.0, 500.0)
                rho_t = col.rho_d .+ col.rho_v
                for j in 1:size(patch.physical, 1)
                    k = mod1(j, kDim); zz = z[k]
                    patch.physical[j, V["u"], 1] = u[k]
                    haskey(V, "rho_e") &&
                        (patch.physical[j, V["rho_e"], 1] = rho_t[k] * e0 * exp(-zz / 1500.0))
                    bump = 2.0e-3 * exp(-((zz - 800.0) / 600.0)^2)
                    # a cloud whose edge rings negative (sharp top) and a healthy rain
                    rc = max(0.0, 4.0e-4 * (1.0 - ((zz - 700.0) / 300.0)^2))
                    rr = 1.0e-4 * exp(-((zz - 400.0) / 400.0)^2) + 1.0e-6
                    patch.physical[j, V["rho_v"], 1] = rho_t[k] * bump
                    patch.physical[j, V["rho_t"], 1] = rho_t[k] * bump + rc + rr
                    patch.physical[j, c, 1] = Scythe.condensate_slot(rc, 0.0, :bhyp, F.MU)
                    patch.physical[j, r, 1] = Scythe.rain_slot(rr, :bhyp, F.MU)
                end
                spectralTransform!(patch)
                gridTransform!(patch)
            end
            arms = Dict{Any,Any}()
            for (tag, mynn, wa, wc, e0) in (("d", true, :derived, true, 0.4),
                                            ("dctl", false, :derived, true, 0.4),
                                            ("s_off", true, :slot, false, 0.4),
                                            ("sctl", false, :slot, true, 0.4),
                                            ("d_k0", true, :derived, true, 0.0),
                                            ("dctl_k0", false, :derived, true, 0.0))
                m, p, mo, gp, z, col = MF.make_mynn_tile(tmp; mynn = mynn, init = :zero,
                                                         num_cells_k = 25, kMax = 5.0e3,
                                                         fluxes = false, ctrans = :bhyp,
                                                         rtrans = :bhyp,
                                                         extra_options = merge(
                                                             Dict{Symbol,Any}(:water_anchor => wa),
                                                             wa === :derived ? Dict{Symbol,Any}() :
                                                             Dict{Symbol,Any}(:mynn_water_closure => wc)))
                seed!(p, mo, gp.kDim, z, col; e0)
                Scythe.init_water_anchor_shadow!(p, mo)
                MF.step_all!(m, p, gp.kDim)
                arms[tag] = (; m, p, mo, gp, x = copy(p.physical[:, :, 1]))
            end
            mo = arms["d"].mo; V = mo.grid_params.vars; ts = mo.ts
            nf = size(arms["dctl"].m.expdot_n, 2)
            x = arms["d"].x
            D = arms["d"].m.expdot_n[:, 1:nf] .- arms["dctl"].m.expdot_n
            MN = arms["d"].m.mynn.colscratch[Threads.threadid()]
            # last column's legs: the sum of alpha * the slots' MYNN tendencies is div_w
            kD = arms["d"].gp.kDim; nc = div(size(x, 1), kD)
            rows = ((nc - 1) * kD + 1):(nc * kD)
            # (the fixture's vapor is a density slot: alpha_v = 1; cloud and rain bhyp)
            sumnu = (D[rows, V["rho_v"]] .+ 2.0 .* D[rows, Scythe.mc_slot(V, "rho_c")]) .+
                    2.0 .* D[rows, Scythe.mc_slot(V, "rho_r")]
            @test maximum(abs, sumnu .- MN.div_w) <= 4 * eps() * maximum(abs, MN.div_w) + 1.0e-30
            # the shadow receives the same sum
            @test D[rows, 3] ≈ MN.div_w rtol = 1.0e-12
            # R held by the closed legs, against the :slot arm without the water closure
            R0, X0 = RX(arms["d"].m, x)
            x1 = copy(x); x1[:, 1:nf] .+= ts .* D
            R1, X1 = RX(arms["d"].m, x1)
            Ds = arms["s_off"].m.expdot_n[:, 1:nf] .- arms["sctl"].m.expdot_n
            Rs0, _ = RX(arms["s_off"].m, x)
            xs = copy(x); xs[:, 1:nf] .+= ts .* Ds
            Rs1, _ = RX(arms["s_off"].m, xs)
            # R held to 1e-4 of the energy the legs moved (the :slot arm without the water
            # closure opens it at first order; here it is the second order of one step)
            Es = ts .* abs.(D[:, V["E_t"]])
            @test maximum(abs, R1 .- R0) <= 1.0e-4 * maximum(Es)
            # ...and against that unclosed :slot arm (measured 2.1e-4 of it: the second order
            # of one 0.5 s step, 9.6e-7 of the energy moved)
            @test maximum(abs, R1 .- R0) <= 1.0e-3 * maximum(abs, Rs1 .- Rs0)
            @info "MYNN derived dR" maximum(abs, R1 .- R0) maximum(Es) maximum(abs, Rs1 .- Rs0)
            # the D3 column identity: sum_k w_k (dE_t + drho_e) = bdry_E, to round-off
            re_i = V["rho_e"]
            wq = MF.gauss_weights(kD, 25, 5.0e3)
            lhs = sum(wq .* (D[rows, V["E_t"]] .+ arms["d"].m.expdot_n[rows, re_i]))
            rhs = arms["d"].m.mynn.bdry_E[nc]
            scale = sum(wq .* abs.(D[rows, V["E_t"]])) + sum(wq .* abs.(arms["d"].m.expdot_n[rows, re_i]))
            @test abs(lhs - rhs) <= 1.0e-10 * scale
            # rho_w moves by the legs exactly, the pool absorbs the water the clip holds
            @test maximum(abs, ts .* D[:, 3]) > 1.0e-8
            # K_h = 0: every water leg is an exact zero (the drag still acts on u)
            Dk = arms["d_k0"].m.expdot_n[:, 1:nf] .- arms["dctl_k0"].m.expdot_n
            for s in (V["rho_v"], Scythe.mc_slot(V, "rho_c"), Scythe.mc_slot(V, "rho_r"), 3, 7)
                @test all(==(0.0), Dk[:, s])
            end
        end
    end

    @testset "3c: horizontal water mixing under :derived moves rho_w by K lap(rho_w)" begin
        # the slot-3 leg is gone (the shadow is not mixed); rho_w = sum alpha nu moves by
        # sum alpha K lap(nu) = K lap(rho_w), by linearity, to rounding
        mktempdir() do tmp
            ex(k) = merge(derived(), Dict{Symbol,Any}(:si_water_slots => :mass))
            seed! = (p, x, V, c) -> begin
                F.vt_seed!(p, x, V, c)
                for i in 1:size(p.physical, 1)
                    p.physical[i, V["nu_c"], 1] *= (1.0 + 0.5 * sin(2pi * x[i, 1] / F.SPAN))
                    p.physical[i, V["nu_v"], 1] += 1.0e-4 * sin(2pi * x[i, 1] / F.SPAN) *
                                                    exp(-((x[i, end] - 2000.0) / 800.0)^2)
                end
            end
            m1, p1, mo1 = F.vt_tile(tmp, "hm_on"; vtr = :bhyp, extra = derived(), seed!,
                                    pp_extra = Dict{Symbol,Float64}(:Khdiff_water => 2000.0))
            m0, p0, mo0 = F.vt_tile(tmp, "hm_off"; vtr = :bhyp, extra = derived(), seed!)
            for c in 1:F.vt_ncols(p1, mo1)
                Scythe.advance_column(m1, c, 1); Scythe.advance_column(m0, c, 1)
            end
            D = m1.expdot_n .- m0.expdot_n
            V = mo1.grid_params.vars
            @test all(==(0.0), D[:, 3])                          # the shadow is not mixed
            ws = Scythe.water_anchor_set(mo1.options, mo1.physical_params, Scythe.mc_slots(mo1))
            drw = zeros(size(D, 1)); lap = zeros(size(D, 1))
            for j in 1:6
                s = ws.slot[j]; a = ws.alpha[j]
                (s == 0 || a == 0.0) && continue
                drw .+= a .* D[:, s]
                lap .+= a .* (2000.0 .* p1.physical[:, s, 3])      # K * d2/dx2 (XZ)
            end
            @test maximum(abs, drw .- lap) <= 1.0e-12 * maximum(abs, lap) +
                                               1.0e-13 * maximum(abs, m1.expdot_n)
        end
    end

    @testset "3c: closed periodic transport-only column: rho_w integral at the O(mu V) bound" begin
        # condensation off, no precipitation, no BL: transport and the acoustics alone. The
        # nu continuity is -div(nu u) + D_k div(u) (D_k <= mu_k/2), so the rho_w integral
        # moves at most by the knee terms: sum_k mu_k V per unit time.
        mktempdir() do tmp
            ex = derived(; condensation = false)
            m, p, mo = F.vt_tile(tmp, "tr"; vtr = :bhyp, extra = ex,
                                 seed! = (p, x, V, c) -> F.vt_seed!(p, x, V, c; wamp = 2.0))
            Scythe.init_water_anchor_shadow!(p, mo)
            eb = m.ebudget; kD = eb.kDim; nc = F.vt_ncols(p, mo)
            W(P) = sum(eb.wxn[c] * Scythe._eb_water_column(eb, P, (c - 1) * kD, 1)[1] for c in 1:nc)
            w0 = W(p.physical)
            Gs(P) = sum(eb.wxn[c] * sum(eb.wz[k] * (P[(c - 1) * kD + k, 3, 1] + eb.rho_tbar_ref[k, 1] -
                                                 (P[(c - 1) * kD + k, 2, 1] + eb.rho_dbar[k]))
                                       for k in 1:kD) for c in 1:nc)
            g0 = Gs(p.physical)
            nstep = 40
            F.vt_run!(m, p, mo, nstep)
            w1 = W(p.physical)
            # the knee bound sum_k (mu_k/2) |div v| over the column and the run (div v of
            # the seeded w: 2 pi / ztop * wamp at most)
            divmax = 2.0 * pi / F.ZTOP * 2.0
            bound = 0.5 * (1.0e-9 + 2.0 * F.MU) * divmax * F.ZTOP * (nstep * mo.ts)
            # and the linear shadow's own change, the conservation class of the old rho_t
            # (product-rule advection and the refit conserve no integral exactly)
            @info "transport-only rho_w integral change" dW = w1 - w0 bound w0 dshadow = Gs(p.physical) - g0
            @test abs(w1 - w0) <= bound
        end
    end

    @testset "3c: the P3 budget closes under :derived with the new terms" begin
        mktempdir() do tmp
            for ice in (false, true), lobe in (false, true)
                ex = derived(; energy_budget = true, lobe_relaxation = lobe)
                m, p, mo = F.vt_tile(tmp, "ebd_$(ice)_$(lobe)"; vtr = :bhyp, ice, extra = ex,
                                     seed! = (p, x, V, c) -> F.vt_seed!(p, x, V, c))
                F.vt_run!(m, p, mo, 6)
                snap = Scythe.energy_budget_snapshot!(m)
                tR = vec(sum(snap.dR[collect(Scythe.EB_TOP), :]; dims = 1))
                tX = vec(sum(snap.dX[collect(Scythe.EB_TOP), :]; dims = 1))
                @test maximum(abs, snap.actualR .- tR) <= 1.0e-10 * maximum(abs, snap.actualR) +
                                                         1.0e-12 * maximum(abs, snap.dR)
                @test maximum(abs, snap.actualX .- tX) <= 1.0e-10 * maximum(abs, snap.actualX) +
                                                         1.0e-12 * maximum(abs, snap.dX)
                # the nudge row is an exact zero: there is no nudge
                @test all(==(0.0), snap.dR[Scythe.EB_OP_ROW[Scythe.EB_OP_VREC], :])
            end
        end
    end

    @testset "3c: the paired Stage L holds R at the kappa coefficient (no latent heat)" begin
        # On the ringing sheet with the device on: across the lobe stage, R moves by
        # <= 1e-4 of the kappa-energy moved; the negative control books the latent (beta)
        # coefficient instead and opens R by > 0.5 lambda_l 2 a |nu_c|.
        mktempdir() do tmp
            m, p, mo = F.vt_tile(tmp, "sl"; vtr = :bhyp,
                                 extra = derived(; lobe_relaxation = true, energy_budget = true),
                                 seed! = (p, x, V, c) -> F.vt_seed!(p, x, V, c))
            F.vt_run!(m, p, mo, 4)
            snap = Scythe.energy_budget_snapshot!(m)
            F.vt_run!(m, p, mo, 2; t0 = 5)
            snap = Scythe.energy_budget_snapshot!(m)
            lr = snap.dR[Scythe.EB_LOBE, :]
            lx = snap.dX[Scythe.EB_LOBE, :]
            st = m.mc_water_stats
            moved = abs(sum(view(st, Scythe.MC_LOBE_C, :)))
            @test moved > 0.0
            Tk = 285.0
            kap = Tk * (Scythe.Rv - 0.55 * Scythe.Cvv)
            # the energy the kappa partner moved, column-integrated over 2 steps: ~kappa * 2 dnu
            @test maximum(abs, lr) <= 1.0e-4 * abs(kap) * 2.0 * moved * 2 + 1.0e-6
            # X (the pool) DOES change: the lobe is filled from the vapor
            @test maximum(abs, lx) > 0.0
        end
    end
end
