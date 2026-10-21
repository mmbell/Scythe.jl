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

    derived(; sd = false, siw = :mass, extra...) =
        merge(Dict{Symbol,Any}(:water_anchor => :derived, :si_water_slots => siw,
                               :state_dependent_si => sd, :si_water_mixed_walls => true),
              Dict{Symbol,Any}(extra))

    # ── 0. the option and the config ────────────────────────────────────────────────
    @testset "option, config default, legacy pins" begin
        @test Scythe.water_anchor_mode(Dict{Symbol,Any}()) === :slot
        @test Scythe.water_anchor_mode(Dict{Symbol,Any}(:water_anchor => :derived)) === :derived
        @test_throws ErrorException Scythe.water_anchor_mode(
            Dict{Symbol,Any}(:water_anchor => :sum))
        @test Scythe.WATER_ANCHOR_MODES == (:slot, :derived)
        @test Scythe.Numerics().water_anchor === :derived
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
                               (:clamp_water, true, false))
                ex = derived(); ex[k] = v
                @test_throws ErrorException F.vt_tile(tmp, "ref_$(k)"; vtr = :bhyp, extra = ex)
            end
            @test_throws ErrorException F.vt_tile(tmp, "ref_kvw"; vtr = :bhyp, extra = derived(),
                pp_extra = Dict{Symbol,Float64}(:Kvdiff_water => 1.0))
            @test_throws ErrorException F.vt_tile(tmp, "ref_tau"; vtr = :bhyp, extra = derived(),
                pp_extra = Dict{Symbol,Float64}(:tau_rho_v_rec => 10.0))
            # the same keys are fine under :slot
            ex = Dict{Symbol,Any}(:water_lobe_attribution => false)
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
end
