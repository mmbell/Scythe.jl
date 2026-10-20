# test/test_vapor_transform.jl -- the VAPOR CONTROL VARIABLE (`options[:vapor_transform]`,
# `physical_params[:vapor_mu]`; Stage 1 of the rho_t retirement plan, reference/
# RHO_T_RETIREMENT_DESIGN.md §6 and the plan's §4 Stage 1 / decision E6).
#
# Under `:bhyp` the prognostic vapor slot carries Ooyama's control variable
# `nu_v' = bhyp(rho_v) - nu_vbar`, `nu_vbar = bhyp(rho_vbar)` against the DERIVED reference vapor
# `rho_vbar = rho_tbar - rho_dbar - rho_cbar`; every reader sees the recovered `ahyp(nu_v) >= 0`;
# every density-space tendency reaches the slot through `J_v = dnu/drho` (with the lobe-
# transport continuation below nu = 0); and the reconciliation nudge is the E6 realized,
# bounded rate law toward `bhyp(max(res_rho_t, 0))`, partnered on the recovered change. rho_t
# stays prognostic. `:none` (the library default) is bitwise the code before the option.
#
# What has to hold (plan §4 Stage 1, tests first):
#   1. `:none` and the absent key are bitwise e81e00e (golden checksums, warm/ice/Louis/MYNN);
#   2. the three identities of design §3.2, and a dry column's rho_w is exactly 0.0;
#   3. rest under `:bhyp` is a discrete fixed point: every tendency is `=== 0.0`;
#   4. transport: the recovered vapor under `:bhyp`/1e-9 tracks `:none` to 1e-9 relative over
#      1000 steps wherever rho_v > 0;
#   5. the `MC_VNEG` census row counts the nu_v < 0 points;
#   6. the nudge guard: with res_rho_t < 0 aloft nu_v never crosses below min(nu_v^n, 0); the
#      old J_v*VREC form runs away to N dt res/tau; rest bitwise; the partner is an exact 0
#      where ahyp(nu_v) does not move;
#   7. (test_energy_partners.jl) R held across ETD / refit / nudge on a seeded lobe column;
#   8. (test_energy_partners.jl "1b") the primitive's special cases.
# Plus the setup refusals and the config plumbing.

using Test
using Scythe
using Springsteel
using SparseArrays
using JLD2

isdefined(Main, :VaporTransformFixtures) || include("vapor_transform_fixtures.jl")
isdefined(Main, :MYNNTestFixtures) || include("mynn_test_fixtures.jl")

@testset "vapor control variable (options[:vapor_transform])" begin
    F = Main.VaporTransformFixtures
    M = Main.MYNNTestFixtures
    VMU = 1.0e-9

    # ── 0. the option, the name, the alias, the width ─────────────────────────────────
    @testset "option, slot name, alias and width" begin
        @test Scythe.vapor_transform_mode(Dict{Symbol,Any}()) === :none
        @test Scythe.vapor_transform_mode(Dict{Symbol,Any}(:vapor_transform => :bhyp)) === :bhyp
        @test_throws ErrorException Scythe.vapor_transform_mode(
            Dict{Symbol,Any}(:vapor_transform => :bhyp_smooth))
        @test_throws ErrorException Scythe.vapor_transform_mode(
            Dict{Symbol,Any}(:vapor_transform => :log))
        @test Scythe.VAPOR_MU_DEFAULT == 1.0e-9
        @test Scythe.vapor_mu(Dict{Symbol,Any}()) == 1.0e-9
        @test Scythe.vapor_mu(Dict{Symbol,Any}(:vapor_mu => 3.0e-10)) == 3.0e-10
        @test Scythe.MC_NU_ALIAS["rho_v"] == "nu_v"
        for cyl in (false, true)
            n0 = Scythe.mc_var_names(Dict{Symbol,Any}(); cyl)
            n1 = Scythe.mc_var_names(Dict{Symbol,Any}(:vapor_transform => :bhyp); cyl)
            @test "rho_v" in n0 && !("nu_v" in n0)
            @test "nu_v" in n1 && !("rho_v" in n1)
            @test findfirst(==("rho_v"), n0) == findfirst(==("nu_v"), n1)
            # everything else in the list is untouched
            @test filter(!=("rho_v"), n0) == filter(!=("nu_v"), n1)
            # MCSlots / mc_slot resolve the vapor by ROLE under either name
            V = Dict(v => i for (i, v) in enumerate(n1))
            @test Scythe.mc_slot(V, "rho_v") == V["nu_v"]
        end
        @test Scythe.vapor_var_name(Dict{Symbol,Any}()) == "rho_v"
        @test Scythe.vapor_var_name(Dict{Symbol,Any}(:vapor_transform => :bhyp)) == "nu_v"
        @test haskey(Scythe.KNOWN_OPTION_KEYS, :vapor_transform)
        @test haskey(Scythe.KNOWN_PHYSICAL_KEYS, :vapor_mu)
    end

    # ── 1. :none is bitwise the code before the option ───────────────────────────────
    @testset ":none and the absent key are bitwise e81e00e (golden capture)" begin
        # Three full steps of the seeded sheet + vapor gap + prognostic vapor bump + w, every
        # partner at its default (on), captured from e81e00e before `:vapor_transform`
        # existed: every bit of the advanced state and of the refit patch. The MYNN arm is the
        # MYNN fixture's live column (water closure on). Both the absent key and an explicit
        # `:none` must reproduce it.
        golden = Dict(
            "warm"  => (UInt128(169252779686602122146478612), UInt128(5225351638953938681038074388)),
            "ice"   => (UInt128(490794160314241089222425181), UInt128(18828796087419452106590956635)),
            "louis" => (UInt128(169252780240349897207505981), UInt128(5225351639192754644858391880)),
            "mynn"  => (UInt128(311926319919379716218570141), UInt128(10944031763325494353189912505)))
        mktempdir() do tmp
            for vtr in (nothing, :none)
                for (tag, ice, louis) in (("warm", false, false), ("ice", true, false),
                                          ("louis", false, true))
                    m, p, mo = F.vt_tile(tmp, "g_$(tag)_$(vtr)"; vtr, ice, louis,
                        seed! = (p, x, V, c) -> F.vt_seed_vapor!(p, x, V, c))
                    F.vt_run!(m, p, mo, 3)
                    @test F.vt_checksum(m.var_np1) == golden[tag][1]
                    @test F.vt_checksum(p.physical) == golden[tag][2]
                end
                m, p, mo, gp, z, col = M.make_mynn_tile(tmp; num_cells_k = 25, kMax = 5.0e3,
                                                        ctrans = :none, rtrans = :none)
                vtr === nothing || (mo.options[:vapor_transform] = vtr)
                M.set_state!(p, mo, gp.kDim, z, col; W0 = 0.3)
                for t in 1:3
                    M.step_all!(m, p, gp.kDim, t); Scythe.calcTendency(m); gridTransform!(p)
                end
                @test F.vt_checksum(m.var_np1) == golden["mynn"][1]
                @test F.vt_checksum(p.physical) == golden["mynn"][2]
            end
        end
    end

    # ── 2. the identities ─────────────────────────────────────────────────────────────
    @testset "the three identities of design §3.2, and dry rho_w == 0.0" begin
        for mu in (1.0e-9, 1.0e-7)
            for rho in (0.0, 1.0e-12, 3.0e-10, 1.0e-9, 4.0e-7, 1.0e-4, 2.0e-2)
                nu = Scythe.bhyp(rho, mu)
                s = rho / (rho + mu)
                J = Scythe.dbhyp(rho, mu)
                # 2ν = ρ + μ s
                @test 2.0 * nu ≈ rho + mu * s rtol = 1.0e-14 atol = 1.0e-30
                # J ρ = ν − D,  D = (μ/2) s² ∈ [0, μ/2)
                D = 0.5 * mu * s^2
                @test J * rho ≈ nu - D rtol = 1.0e-10 atol = 1.0e-30
                @test 0.0 <= D < 0.5 * mu
                # 2J − 1 = μ²/(ρ + μ)² = dB/dρ,  B = μ s
                @test 2.0 * J - 1.0 ≈ mu^2 / (rho + mu)^2 rtol = 1.0e-10 atol = 1.0e-15
                h = max(rho, mu) * 1.0e-6
                dB = (mu * (rho + h) / (rho + h + mu) - mu * max(rho - h, 0.0) /
                      (max(rho - h, 0.0) + mu)) / (rho + h - max(rho - h, 0.0))
                # (a finite difference of B, where it is not lost to cancellation)
                rho <= 100.0 * mu && @test dB ≈ mu^2 / (rho + mu)^2 rtol = 1.0e-5
                # the round trip the reference relies on
                @test Scythe.ahyp(nu, mu) ≈ rho rtol = 4.0e-16 atol = 1.0e-30
            end
        end
        # ρ_w = Σρ_k (recovered) + B + Λ, B = Σ_{ν≥0} μ s, Λ = Σ_{ν<0} (2ν − ρ)
        mu = VMU
        nus = (Scythe.bhyp(1.0e-2, mu), -3.0e-7, Scythe.bhyp(2.0e-9, mu), 0.0)
        rhos = Scythe.ahyp.(nus, mu)
        B = sum(n >= 0.0 ? mu * r / (r + mu) : 0.0 for (n, r) in zip(nus, rhos))
        L = sum(n < 0.0 ? 2.0 * n - r : 0.0 for (n, r) in zip(nus, rhos))
        @test 2.0 * sum(nus) ≈ sum(rhos) + B + L rtol = 1.0e-14
        @test L < 0.0 && 0.0 <= B < length(nus) * mu
        # dry air: bhyp(0) = 0 exactly, so the water anchor of a dry column is exactly 0.0
        @test Scythe.bhyp(0.0, mu) === 0.0
        @test 2.0 * (Scythe.bhyp(0.0, mu) + Scythe.bhyp(0.0, 1.0e-7)) === 0.0
        # the slot helpers: :none is the old perturbation bitwise; :bhyp is the ν deviation
        rvb = 7.3e-3
        @test Scythe.vapor_slot(1.0e-2, 1.2, 1.19, 0.0) === 1.0e-2 - ((1.2 - 1.19) - 0.0)
        @test Scythe.vapor_slot(1.0e-2, 1.2, 1.19, 0.0, :none, mu) ===
              Scythe.vapor_slot(1.0e-2, 1.2, 1.19, 0.0)
        rbar = (1.2 - 1.19) - 0.0
        @test Scythe.vapor_slot(1.0e-2, 1.2, 1.19, 0.0, :bhyp, mu) ===
              Scythe.bhyp(1.0e-2, mu) - Scythe.bhyp(rbar, mu)
        @test Scythe.vapor_slot(rbar, 1.2, 1.19, 0.0, :bhyp, mu) === 0.0
        nub = Scythe.bhyp(rvb, mu)
        @test Scythe.recover_rho_v(0.0, nub, :bhyp, mu) === Scythe.ahyp(nub, mu)
        @test Scythe.recover_rho_v(1.0e-4, rvb, :none, mu) === 1.0e-4 + rvb
        @test Scythe.recover_rho_v(-2.0 * nub, nub, :bhyp, mu) === 0.0   # a vapor lobe reads 0
    end

    # ── the derived reference and the setup refusals ─────────────────────────────────
    @testset "derived reference nu_vbar, rho_tbar_w; setup refusals" begin
        mktempdir() do tmp
            m, p, mo = F.vt_tile(tmp, "ref_b"; vtr = :bhyp)
            # the tile's resting columns plus the four vapor-reference ones (on the energy
            # budget's reference part, `Scythe.mc_nu_vbar` & co.)
            vref(mm) = merge(mm.mc_ref_diag, (nu_vbar = Scythe.mc_nu_vbar(mm),
                                              nu_vbar_z = Scythe.mc_nu_vbar_z(mm),
                                              rho_tbar_w = Scythe.mc_rho_tbar_w(mm),
                                              rho_tbar_w_z = Scythe.mc_rho_tbar_w_z(mm)))
            d = vref(m)
            rs = m.ref_state
            rtb = Springsteel.ref_rho_t(rs); rdb = Springsteel.ref_rho_d(rs)
            rcb = Springsteel.ref_rho_c(rs)
            # the derived density reference is unchanged by the option
            @test d.rho_vbar == (rtb[:, 1] .- rdb[:, 1]) .- rcb[:, 1]
            @test d.nu_vbar == Scythe.bhyp.(d.rho_vbar, VMU)
            @test d.nu_vbar_z == Scythe.dbhyp.(d.rho_vbar, VMU) .* d.rho_vbar_z
            # rho_tbar_w = ρ̄_d + 2(ν̄_v + ν̄_c) (cloud transformed; ν̄_c = 0 on a clear base)
            nucb = Scythe.bhyp.(rcb[:, 1], F.MU)
            @test d.rho_tbar_w == rdb[:, 1] .+ 2.0 .* (d.nu_vbar .+ nucb)
            @test maximum(abs.(d.rho_tbar_w .- rtb[:, 1])) <= 2.0 * VMU    # the bias B̄ ≤ μ_v
            @test length(d.rho_tbar_w_z) == length(d.rho_tbar_w)
            # Tbar is the EOS temperature of the RECOVERED reference vapor (what the driver
            # forms at rest), so rest is exact
            pb = Springsteel.ref_pressure(rs)[:, 1]
            @test d.Tbar == Scythe.eos_temperature.(pb, rdb[:, 1], Scythe.ahyp.(d.nu_vbar, VMU))
            # :none: ν̄ IS ρ̄ (bitwise), the weight of an untransformed slot is 1
            m0, _, _ = F.vt_tile(tmp, "ref_n"; vtr = :none, tr = :none)
            d0 = vref(m0)
            @test d0.nu_vbar == d0.rho_vbar && d0.nu_vbar_z == d0.rho_vbar_z
            rs0 = m0.ref_state
            @test d0.rho_tbar_w == Springsteel.ref_rho_d(rs0)[:, 1] .+
                                   (d0.nu_vbar .+ Springsteel.ref_rho_c(rs0)[:, 1])
            # refusals: a width above 1e-2 of the driest reference vapor, and a dry base
            @test_throws ErrorException F.vt_tile(tmp, "ref_wide"; vtr = :bhyp,
                pp_extra = Dict{Symbol,Float64}(:vapor_mu => 1.0e-3))
            @test_throws ErrorException F.vt_tile(tmp, "ref_dry"; vtr = :bhyp, rh = z -> 0.0)
            # Kvdiff_water (the implicit rho_w solve) is refused under the vapor transform
            mk, pk, mok = F.vt_tile(tmp, "ref_kvw"; vtr = :bhyp,
                pp_extra = Dict{Symbol,Float64}(:Kvdiff_water => 10.0))
            @test_throws ErrorException Scythe.advance_column(mk, 1, 1)
        end
    end

    # ── 3. rest under :bhyp ───────────────────────────────────────────────────────────
    @testset "rest under :bhyp: a discrete fixed point, every tendency === 0.0" begin
        mktempdir() do tmp
            # (the Louis arm without surface fluxes: an SST above the air is not a rest state)
            for (tag, ice, louis) in (("warm", false, false), ("ice", true, false),
                                      ("louis", false, true))
                m, p, mo = F.vt_tile(tmp, "rest_$(tag)"; vtr = :bhyp, ice, louis, sfc = false)
                F.vt_run!(m, p, mo, 5)
                @test maximum(abs.(p.physical)) == 0.0
                @test maximum(abs.(m.var_np1)) == 0.0
                # every tendency is zero; the vapor and pressure slots' are +0.0 bit for bit
                # (the cloud slot carries a -0.0 at rest under :none too)
                rv_i = m.mc_slots.rho_v
                @test all(m.expdot_n .== 0.0)
                @test all(m.expdot_n[:, rv_i] .=== 0.0)
                @test all(m.expdot_n[:, 1] .=== 0.0)
                Scythe.advance_column(m, 1, 6)
                S = m.mc_scratch[Threads.threadid()]
                @test all(S.VREC .=== 0.0)
                @test S.Tk == m.mc_ref_diag.Tbar
                @test S.rho_v == Scythe.ahyp.(Scythe.mc_nu_vbar(m), VMU)
            end
        end
    end

    # ── 4. transport ──────────────────────────────────────────────────────────────────
    @testset "transport: recovered rho_v under :bhyp tracks :none to 1e-9 over 1000 steps" begin
        mktempdir() do tmp
            out = Dict{Symbol,Any}()
            for vtr in (:none, :bhyp)
                # a clear column (no condensate), so the comparison is transport, the nudge
                # (the rho_t gap at 3 km) and the thermodynamics, not a phase change. The
                # :bhyp seed is formed against the sounding's vapor; bhyp is affine to
                # (μ/ρ)² there, so against the model's derived reference it is the same
                # density bump to far below the tolerance.
                m, p, mo = F.vt_tile(tmp, "tr_$(vtr)"; vtr, rh = z -> 0.5,
                    seed! = (p, x, V, c) -> F.vt_seed_vapor!(p, x, V, c; amp = 0.0,
                        wamp = 0.3, vtr, vmu = VMU, rho_vbar = c.rho_v))
                F.vt_run!(m, p, mo, 1000)
                kDim = mo.grid_params.kDim
                nc = div(size(p.physical, 1), kDim)
                nu = p.physical[:, m.mc_slots.rho_v, 1] .+ repeat(Scythe.mc_nu_vbar(m), nc)
                out[vtr] = vtr === :none ? nu : Scythe.ahyp.(nu, VMU)
                out[Symbol(vtr, :_w)] = maximum(abs.(p.physical[:, mo.grid_params.vars["w"], 1]))
                @test all(isfinite, p.physical)
            end
            rn = out[:none]; rb = out[:bhyp]
            pos = rn .> 0.0
            @test count(pos) == length(rn)
            @test out[:none_w] > 1.0e-3                     # the column moved
            @test maximum(abs.(rb[pos] .- rn[pos]) ./ rn[pos]) <= 1.0e-9
        end
    end

    # ── 5. the census ─────────────────────────────────────────────────────────────────
    @testset "MC_VNEG: the census counts the nu_v < 0 points at the report" begin
        @test Scythe.MC_VNEG == Scythe.MC_ECEN_STEPS + 1
        @test Scythe.MC_STATS_LAST == Scythe.MC_VNEG
        @test length(Scythe.MC_WATER_STATS) == Scythe.MC_STATS_LAST
        mktempdir() do tmp
            for vtr in (:bhyp, :none)
                # the vapor slot driven below zero above 8 km (nu_v' = -2 nu_vbar there)
                function seed_neg!(patch, pts, V, col)
                    rv_i = Scythe.mc_slot(V, "rho_v")
                    kDim = length(col.Tk)
                    for i in 1:size(patch.physical, 1)
                        k = mod1(i, kDim)
                        pts[i, end] > 8.0e3 || continue
                        patch.physical[i, rv_i, 1] = vtr === :none ? -2.0 * col.rho_v[k] :
                            -2.0 * Scythe.bhyp(col.rho_v[k], VMU)
                    end
                end
                m, p, mo = F.vt_tile(tmp, "vneg_$(vtr)"; vtr, seed! = seed_neg!)
                nu = p.physical[:, m.mc_slots.rho_v, 1] .+
                     repeat(Scythe.mc_nu_vbar(m), div(size(p.physical, 1), mo.grid_params.kDim))
                nneg = count(<(0.0), nu)
                @test nneg > 0
                out_int = round(Int, mo.output_interval / mo.ts)
                for t in 2:(out_int + 1)
                    Scythe.energy_census_trace(m, t)
                end
                @test sum(view(m.mc_water_stats, Scythe.MC_VNEG, :)) == nneg
            end
            # rest: zero
            m, p, mo = F.vt_tile(tmp, "vneg_rest"; vtr = :bhyp)
            out_int = round(Int, mo.output_interval / mo.ts)
            for t in 2:(out_int + 1)
                Scythe.energy_census_trace(m, t)
            end
            @test sum(view(m.mc_water_stats, Scythe.MC_VNEG, :)) === 0.0
        end
    end

    # ── 6. the nudge guard (E6) ───────────────────────────────────────────────────────
    @testset "nudge guard: the E6 rate law, pointwise" begin
        tau = 10.0; dt = 0.5; mu = VMU
        # rest: target bhyp(res) == ν exactly -> the rate is +0.0
        for rv in (1.0e-6, 4.0e-7, 1.3e-2)
            nu = Scythe.bhyp(rv, mu)
            @test Scythe.vapor_nudge_rate(rv, nu, mu, tau) === 0.0
        end
        # the affine regime reproduces J_v·VREC to O(μ/ρ)
        rv = 1.0e-2; res = rv + 2.0e-5; nu = Scythe.bhyp(rv, mu)
        @test Scythe.vapor_nudge_rate(res, nu, mu, tau) ≈
              Scythe.dbhyp(rv, mu) * Scythe.rho_v_reconcile(res, rv, tau) rtol = 1.0e-6
        # res < 0: the new law decays ν to 0 without crossing (N = 1000 Euler steps); the old
        # J_v·VREC form runs away (the 2026-10-05 runaway): VREC → res/τ once ahyp(ν) = 0, so
        # ν → N Δt J res/τ, with J = dbhyp(0) = 1 on the clipped form and → ½ on the
        # lobe-transport continuation the slot's continuity terms use
        res = -1.0e-3
        N = 1000
        for nu0 in (Scythe.bhyp(2.0e-3, mu), 0.0, -2.0e-6)
            nnew = nu0; nclip = nu0; nlobe = nu0
            for _ in 1:N
                nnew += dt * Scythe.vapor_nudge_rate(res, nnew, mu, tau)
                rc = Scythe.ahyp(nclip, mu)
                nclip += dt * Scythe.dbhyp(max(rc, 0.0), mu) * Scythe.rho_v_reconcile(res, rc, tau)
                rl = Scythe.ahyp(nlobe, mu)
                nlobe += dt * Scythe.lobe_transport_jacobian(nlobe, rl, mu) *
                         Scythe.rho_v_reconcile(res, rl, tau)
            end
            @test nnew >= min(nu0, 0.0)
            @test abs(nnew) <= abs(nu0) + 1.0e-30
            @test nclip ≈ N * dt * res / tau rtol = 0.05            # the runaway, clipped J
            @test nlobe ≈ 0.5 * N * dt * res / tau rtol = 0.05      # ...and continued J
        end
        # the partner: an exact 0.0 where ahyp(ν_v) does not move (a lobe pulled toward 0)
        nu = -3.0e-6
        r = Scythe.vapor_nudge_rate(0.0, nu, mu, tau)
        @test r > 0.0
        @test Scythe.ahyp(nu + dt * r, mu) - Scythe.ahyp(nu, mu) === 0.0
    end

    @testset "nudge guard in the model: res_rho_t < 0 aloft, 1000 steps" begin
        mktempdir() do tmp
            # ρ_d raised by twice the reference vapor above 6 km (a smooth step), ρ_t and p not:
            # the partition then has NO water for the vapor there, res_rho_t ≈ −ρ̄_v < 0, and
            # rho_t' = p' = 0 leaves no buoyancy and no pressure gradient to start with. The
            # vapor nudge drives the vapor toward zero; its pressure partner then makes the
            # column move a little.
            function seed_deficit!(patch, pts, V, col)
                kDim = length(col.Tk)
                for i in 1:size(patch.physical, 1)
                    z = pts[i, end]; k = mod1(i, kDim)
                    sw = 0.5 * (1.0 + tanh((z - 6000.0) / 600.0))
                    patch.physical[i, V["rho_d"], 1] = 2.0 * col.rho_v[k] * sw
                end
            end
            # where res_rho_t < 0 (one step of a twin tile)
            mA, pA, moA = F.vt_tile(tmp, "guardA"; vtr = :bhyp, seed! = seed_deficit!)
            Scythe.advance_column(mA, 1, 1)
            aloft = copy(mA.mc_scratch[Threads.threadid()].res_rho_t) .< 0.0
            @test count(aloft) > 0
            m, p, mo = F.vt_tile(tmp, "guard"; vtr = :bhyp, seed! = seed_deficit!)
            kDim = mo.grid_params.kDim
            rv_i = m.mc_slots.rho_v
            nc = div(size(p.physical, 1), kDim)
            nubar = repeat(Scythe.mc_nu_vbar(m), nc)
            nu0 = p.physical[:, rv_i, 1] .+ nubar
            F.vt_run!(m, p, mo, 1000)
            nuN = p.physical[:, rv_i, 1] .+ nubar
            A = repeat(aloft, nc)
            @test all(isfinite, p.physical)
            # NO RUNAWAY. The old J_v·VREC form would take ν_v to ≈ N Δt res/(2τ) ≈ −1.7e-2
            # here, 35x the initial vapor, deep negative. The new law decays it to 0; what is
            # left is the per-step REFIT ringing of the kink the floored target has where
            # res_rho_t crosses 0 (the same pattern with the partner off and w ≡ 0: it is the
            # refit, not transport), which the nudge pulls back each step.
            # The pointwise bound ν^{n+N} ≥ min(ν^n, 0) is the rate law's own property
            # (testset above); in the model the refit's ring sits on top of it.
            # (measured: min −2.9 %, max |ν| above the crossing level 3 %, at the crossing 13 %)
            @test minimum(nuN[A]) >= -0.05 * maximum(nu0[A])
            hi = A .& (repeat(m.tilepoints[1:kDim, end], nc) .> 6500.0)
            @test count(hi) > 0
            @test maximum(abs.(nuN[hi])) <= 0.05 * maximum(nu0[A])
            # the vapor partner moved p (the nudge removed vapor at fixed E_t)
            @test maximum(abs.(p.physical[:, mo.grid_params.vars["p"], 1])) > 0.0
        end
    end

    # ── recovery at the readers ───────────────────────────────────────────────────────
    @testset "readers see the recovered vapor: driver, state-minima, netCDF" begin
        mktempdir() do tmp
            function seed_neg!(patch, pts, V, col)
                rv_i = Scythe.mc_slot(V, "rho_v")
                kDim = length(col.Tk)
                for i in 1:size(patch.physical, 1)
                    k = mod1(i, kDim)
                    pts[i, end] > 8.0e3 || continue
                    patch.physical[i, rv_i, 1] = -2.0 * Scythe.bhyp(col.rho_v[k], VMU)
                end
            end
            m, p, mo = F.vt_tile(tmp, "read"; vtr = :bhyp, seed! = seed_neg!)
            Scythe.advance_column(m, 1, 1)
            S = m.mc_scratch[Threads.threadid()]
            kDim = mo.grid_params.kDim
            nu = p.physical[1:kDim, m.mc_slots.rho_v, 1] .+ Scythe.mc_nu_vbar(m)
            @test S.rho_v == Scythe.ahyp.(nu, VMU)
            @test minimum(S.rho_v) == 0.0 && minimum(nu) < 0.0
            @test S.Tk == Scythe.eos_temperature.(S.p, S.rho_d, S.rho_v)
            # J_v is the lobe-transport Jacobian of the slot
            @test Scythe.vapor_jacobian_column(S) == Scythe.lobe_transport_jacobian.(nu, S.rho_v, VMU)
        end
        # the netCDF derived fields read the recovered vapor (zero references: the derived
        # ρ̄_v is 0, so the slot is the control variable itself)
        z = collect(range(0.0, 12000.0; length = 13))
        n_i = 3; n_k = length(z)
        Tf(i, zz) = 300.0 - 0.0065 * zz + 0.5 * i
        p = [1.0e5 * exp(-z[k] / 8000.0) for i in 1:n_i, k in 1:n_k]
        rv = [0.7 * Scythe.rho_v_sat(Tf(i, z[k]), p[i, k] / 100.0) for i in 1:n_i, k in 1:n_k]
        rd = [(p[i, k] - Scythe.Rv * Tf(i, z[k]) * rv[i, k]) / (Scythe.Rd * Tf(i, z[k]))
              for i in 1:n_i, k in 1:n_k]
        nu = Scythe.bhyp.(rv, VMU); nu[2, 7] = -1.0e-5
        zp = zeros(n_k)
        ref = Scythe.RegularReferenceProfiles(z, zp, zp, zp, zp, zp, zp, zp, zp)
        slots = (; p, rho_d = rd, rho_t = rd .+ rv, E_t = zeros(n_i, n_k), Q_ss = zeros(n_i, n_k),
                   rho_v = nu, u = zeros(n_i, n_k), w = zeros(n_i, n_k),
                   rain = zeros(n_i, n_k), cloud = zeros(n_i, n_k))
        cfg = Scythe.DerivedConfig(:bhyp, 1.0e-7, :bhyp, 1.0e-7, :none, 1.0, :none,
                                   (1.0e-12, 1.0e-2, 1.0e-16, 1.0e-16), 1, false, false,
                                   8.0e6, 100.0, "test", false, :eos, :bhyp, VMU)
        d = Scythe.mc_derived_fields(z, slots, ref, cfg)
        @test d.rho_v == Scythe.ahyp.(nu, VMU)
        @test d.rho_v[2, 7] === 0.0
        @test all(d.T_eos .=== Scythe.eos_temperature.(p, rd, d.rho_v))
        # the old 16-field constructor is :none
        cfg0 = Scythe.DerivedConfig(:bhyp, 1.0e-7, :bhyp, 1.0e-7, :none, 1.0, :none,
                                    (1.0e-12, 1.0e-2, 1.0e-16, 1.0e-16), 1, false, false,
                                    8.0e6, 100.0, "test", false, :eos)
        @test cfg0.vtrans === :none
    end

    # ── config plumbing ───────────────────────────────────────────────────────────────
    @testset "config: physics.microphysics.vapor_transform / vapor_mu" begin
        @test Scythe.Microphysics().vapor_transform === :bhyp
        @test Scythe.Microphysics().vapor_mu == 1.0e-9
        @test Scythe.OWNED_OPTION_KEYS[:vapor_transform] ==
              "physics.microphysics.vapor_transform"
    end

    # ── checkpoints ───────────────────────────────────────────────────────────────────
    @testset "allocation-free column step under :bhyp (warm and Louis)" begin
        mktempdir() do tmp
            for louis in (false, true)
                m, p, mo = F.vt_tile(tmp, "alloc_$(louis)"; vtr = :bhyp, louis,
                    seed! = (p, x, V, c) -> F.vt_seed!(p, x, V, c))
                kDim = mo.grid_params.kDim
                Scythe.moist_compressible_XZ(m, 1, kDim, 2)
                @test (@allocations Scythe.moist_compressible_XZ(m, 1, kDim, 2)) == 0
            end
        end
    end
end
