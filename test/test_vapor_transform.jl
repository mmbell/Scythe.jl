# test/test_vapor_transform.jl -- the VAPOR CONTROL VARIABLE (`options[:vapor_transform]`,
# `physical_params[:vapor_mu]`; Stage 1 of the rho_t retirement plan, reference/
# RHO_T_RETIREMENT_DESIGN.md §6 and the plan's §4 Stage 1 / decision E6).
#
# Under `:bhyp` the prognostic vapor slot carries Ooyama's control variable
# `nu_v' = bhyp(rho_v) - nu_vbar`, `nu_vbar = bhyp(rho_vbar)` against the DERIVED reference vapor
# `rho_vbar = rho_tbar - rho_dbar - rho_cbar`; every reader sees the recovered `ahyp(nu_v) >= 0`;
# every density-space tendency reaches the slot through `J_v = dnu/drho` (with the lobe-
# transport continuation below nu = 0). Since Stage 4 of the rho_t retirement the vapor IS slot
# 3, rho_t is derived (rho_d + sum_k alpha_k nu_k) and the reconciliation nudge of Stage 1 is
# gone with the prognostic rho_t. `:none` (the library default) is the density-vapor lineage.
#
# What has to hold (plan §4 Stage 1, tests first):
#   1. `:none` and the absent key are bitwise e81e00e (golden checksums, warm/ice/Louis/MYNN);
#   2. the three identities of design §3.2, and a dry column's rho_w is exactly 0.0;
#   3. rest under `:bhyp` is a discrete fixed point: every tendency is `=== 0.0`;
#   4. transport: the recovered vapor under `:bhyp`/1e-9 tracks `:none` to 1e-9 relative over
#      1000 steps wherever rho_v > 0;
#   5. the `MC_VNEG` census row counts the nu_v < 0 points;
#   6. (retired at Stage 4 with the nudge it guarded);
#   7. (test_energy_partners.jl) R held across ETD / refit on a seeded lobe column;
#   8. (test_energy_partners.jl "1b") the primitive's special cases.
# Plus the setup refusals, the config plumbing and the checkpoint conversion on load.

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

    # ── 1. :none under the derived anchor is bitwise bf577b2 (goldens re-seeded) ─────
    @testset ":none and the absent key are bitwise bf577b2 under the derived anchor (golden capture)" begin
        # Three full steps of the seeded sheet + vapor gap + prognostic vapor bump + w, every
        # partner at its default, under the DENSITY vapor (`:none`, the lineage Stage 4 keeps)
        # and the derived water anchor: every bit of the advanced state and of the refit patch,
        # summed in the Stage-4 layout (`s4_checksum`, slot 3 left out). RE-SEEDED at bf577b2
        # for Stage 4 (the :slot captures of e81e00e they replace are in git history). The
        # MYNN arm is the MYNN fixture's live column. Both the absent key and an explicit
        # `:none` must reproduce it.
        golden = Dict(
            "warm"  => (UInt128(142555738086957176692364298), UInt128(4151400111650453995826857756)),
            "ice"   => (UInt128(441078978215146199125823757), UInt128(16718446589831356995192893660)),
            "louis" => (UInt128(142555738721084531363846075), UInt128(4151400123595029663328582793)),
            "mynn"  => (UInt128(259941565336452810300720602), UInt128(9083517470135141505831894328)))
        got = Dict{String,Any}()
        mktempdir() do tmp
            for vtr in (nothing, :none)
                for (tag, ice, louis) in (("warm", false, false), ("ice", true, false),
                                          ("louis", false, true))
                    m, p, mo = F.vt_tile(tmp, "g_$(tag)_$(vtr)"; vtr, ice, louis,
                        seed! = (p, x, V, c) -> F.vt_seed_vapor!(p, x, V, c))
                    F.vt_run!(m, p, mo, 3)
                    V = mo.grid_params.vars
                    got[tag] = (F.s4_checksum(m.var_np1, V), F.s4_checksum(p.physical, V))
                    @test got[tag][1] == golden[tag][1]
                    @test got[tag][2] == golden[tag][2]
                end
                xo = Dict{Symbol,Any}()
                vtr === nothing || (xo[:vapor_transform] = vtr)
                m, p, mo, gp, z, col = M.make_mynn_tile(tmp; num_cells_k = 25, kMax = 5.0e3,
                                                        ctrans = :none, rtrans = :none,
                                                        extra_options = xo)
                M.set_state!(p, mo, gp.kDim, z, col; W0 = 0.3)
                for t in 1:3
                    M.step_all!(m, p, gp.kDim, t); Scythe.calcTendency(m); gridTransform!(p)
                end
                V = mo.grid_params.vars
                got["mynn"] = (F.s4_checksum(m.var_np1, V), F.s4_checksum(p.physical, V))
                @test got["mynn"][1] == golden["mynn"][1]
                @test got["mynn"][2] == golden["mynn"][2]
            end
        end
        get(ENV, "SCYTHE_PRINT_GOLDENS", "") == "1" &&
            foreach(k -> println("GOLDEN vapor_transform $(k) => $(got[k])"), sort(collect(keys(got))))
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
        # a NEGATIVE input density maps to the odd continuation, never through the
        # non-monotone / singular bhyp(ρ < 0) (bhyp(−1.5μ) = +0.75μ, a pole at −μ)
        for r in (-1.5 * mu, -mu, -0.3 * mu, -2.0e-6)
            @test Scythe.vapor_control(r, mu) === -Scythe.bhyp(-r, mu)
            @test Scythe.vapor_control(r, mu) < 0.0
            @test Scythe.vapor_slot(r, 1.2, 1.19, 0.0, :bhyp, mu) ===
                  -Scythe.bhyp(-r, mu) - Scythe.bhyp(rbar, mu)
            @test Scythe.ahyp(Scythe.vapor_control(r, mu), mu) === 0.0          # reads 0
            @test Scythe.ahyp_signed(Scythe.vapor_control(r, mu), mu) ≈ r rtol = 1.0e-12
        end
        @test Scythe.bhyp(-1.5 * mu, mu) > 0.0         # why the guard exists
        for r in (0.0, 1.0e-12, 3.0e-3)
            @test Scythe.vapor_control(r, mu) === Scythe.bhyp(r, mu)
        end
        @test Scythe.vapor_slot(-1.0e-3, 1.2, 1.19, 0.0, :none, mu) ===
              Scythe.vapor_slot(-1.0e-3, 1.2, 1.19, 0.0)
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
            # Kvdiff_water (the implicit rho_w solve, which read the deleted rho_t slot) is
            # refused at tile creation since Stage 4 of the rho_t retirement (it used to be
            # refused at the first column step under the vapor transform)
            @test_throws ErrorException F.vt_tile(tmp, "ref_kvw"; vtr = :bhyp,
                pp_extra = Dict{Symbol,Float64}(:Kvdiff_water => 10.0))
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
                # every tendency is zero; the pressure slot's is +0.0 bit for bit (the cloud
                # slot carries a -0.0 at rest under :none too). The VAPOR slot's sign of zero is
                # pinned exactly: its tendency is `ADV + (-(J_v (rho_v div)))` (div = +0.0, so the
                # second term is -0.0) plus the exchange closure `water_exchange_dv` of zero
                # exchanges, -(0.0)/alpha_v = -0.0, and a signed-zero sum is -0.0 only when every
                # term is. ADV (`mc_advect!` of the zero state) is -0.0 at every level but the top
                # four, where it is +0.0 (measured), so the warm and ice arms carry -0.0 below and
                # +0.0 there. Under the
                # retired `:slot` anchor the nudge rate's +0.0 was added after and made it +0.0
                # everywhere; the same -0.0 pattern was measured at bf577b2 with
                # `:water_anchor => :derived`. Louis adds its +0.0 legs last: +0.0 everywhere.
                rv_i = m.mc_slots.rho_v
                kD = mo.grid_params.kDim
                @test all(m.expdot_n .== 0.0)
                @test all(m.expdot_n[:, 1] .=== 0.0)
                e = m.expdot_n[:, rv_i]
                if louis
                    @test all(e .=== 0.0)
                else
                    @test all(g -> e[g] === (mod1(g, kD) <= kD - 4 ? -0.0 : 0.0), eachindex(e))
                end
                Scythe.advance_column(m, 1, 6)
                S = m.mc_scratch[Threads.threadid()]
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
                # a clear column (no condensate), so the comparison is transport and the
                # thermodynamics, not a phase change. The
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
        @test Scythe.MC_VNEG_MASS == Scythe.MC_VNEG + 1
        # the water-anchor block (Stage 3) follows MC_VNEG_MASS and is now the final block
        @test Scythe.MC_W_PART == Scythe.MC_VNEG_MASS + 1
        @test Scythe.MC_STATS_LAST == Scythe.MC_ANCHOR_INERT
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
                # MC_VNEG_MASS: the vapor lobe 2 min(ν_v, 0) integrated with the census's own
                # quadrature under :bhyp, 0.0 under :none; under :bhyp the pool includes it
                # (no condensate lobe here, so the pool IS the vapor lobe)
                eb = m.ebudget; kD = mo.grid_params.kDim
                ref_mass = vtr === :bhyp ?
                    sum(eb.wxn[div(g - 1, kD) + 1] * eb.wz[mod1(g, kD)] * 2.0 * min(nu[g], 0.0)
                        for g in eachindex(nu)) : 0.0
                vm = sum(view(m.mc_water_stats, Scythe.MC_VNEG_MASS, :))
                pool = sum(view(m.mc_water_stats, Scythe.MC_XGAP_POOL, :))
                if vtr === :bhyp
                    @test vm < 0.0
                    @test vm ≈ ref_mass rtol = 1.0e-12
                    @test pool ≈ vm rtol = 1.0e-10
                else
                    @test vm === 0.0
                    @test pool === 0.0
                end
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
    # (Both nudge-guard testsets -- the E6 rate law pointwise and the res_rho_t < 0 model
    # run -- are gone: the vapor reconciliation nudge, `vapor_nudge_rate`, `rho_v_reconcile`
    # and `res_rho_t` were retired with slot 3 at Stage 4 of the rho_t retirement.)

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
            # the state-minima trace's T_eos reads the RECOVERED vapor: its reported max T is
            # the max over the tile of eos_temperature(p, ρ_d, ahyp(ν_v))
            mo.options[:state_minima_trace] = 1
            out = mktemp() do path, io
                redirect_stdout(io) do
                    Scythe.state_minima_trace(m, 1)
                end
                flush(io)
                read(path, String)
            end
            delete!(mo.options, :state_minima_trace)
            mt = match(r"max T=([0-9.eE+-]+) K", out)
            @test mt !== nothing
            nc = div(size(p.physical, 1), kDim)
            V = mo.grid_params.vars
            pall = p.physical[:, V["p"], 1] .+ repeat(Springsteel.ref_pressure(m.ref_state)[:, 1], nc)
            rdall = p.physical[:, V["rho_d"], 1] .+ repeat(Springsteel.ref_rho_d(m.ref_state)[:, 1], nc)
            nuall = p.physical[:, m.mc_slots.rho_v, 1] .+ repeat(Scythe.mc_nu_vbar(m), nc)
            Trec = Scythe.eos_temperature.(pall, rdall, Scythe.ahyp.(nuall, VMU))
            @test parse(Float64, mt.captures[1]) == round(maximum(Trec); sigdigits = 5)
            # ...and its T_split is formed with the recovered vapor: the raw negative vapor
            # would give a different (hotter) T_eos aloft
            Traw = Scythe.eos_temperature.(pall, rdall, nuall)
            @test maximum(abs.(Traw .- Trec)) > 0.0
        end
        # the netCDF derived fields read the recovered vapor against bhyp of a NONZERO derived
        # reference ρ̄_t − ρ̄_d − ρ̄_c (so the shift by the transformed reference is exercised)
        z = collect(range(0.0, 12000.0; length = 13))
        n_i = 3; n_k = length(z)
        Tf(i, zz) = 300.0 - 0.0065 * zz + 0.5 * i
        p = [1.0e5 * exp(-z[k] / 8000.0) for i in 1:n_i, k in 1:n_k]
        rv = [0.7 * Scythe.rho_v_sat(Tf(i, z[k]), p[i, k] / 100.0) for i in 1:n_i, k in 1:n_k]
        rd = [(p[i, k] - Scythe.Rv * Tf(i, z[k]) * rv[i, k]) / (Scythe.Rd * Tf(i, z[k]))
              for i in 1:n_i, k in 1:n_k]
        rdb = vec(rd[2, :]); rvb = 0.9 .* vec(rv[2, :]); rcb = zeros(n_k)
        rtb = rdb .+ rvb
        zp = zeros(n_k)
        ref = Scythe.RegularReferenceProfiles(z, zp, rdb, rvb, rcb, rtb, zp, zp, zp)
        rvb_d = (rtb .- rdb) .- rcb                       # the writer's derived reference
        nub = Scythe.bhyp.(rvb_d, VMU)
        nu = Scythe.bhyp.(rv, VMU) .- reshape(nub, 1, n_k)  # the slot: ν_v' against ν̄_v
        nu[2, 7] = -1.0e-5 - nub[7]                      # a vapor lobe
        slots = (; p, rho_d = rd .- reshape(rdb, 1, n_k),
                   E_t = zeros(n_i, n_k), Q_ss = zeros(n_i, n_k),
                   rho_v = nu, u = zeros(n_i, n_k), w = zeros(n_i, n_k),
                   rain = zeros(n_i, n_k), cloud = zeros(n_i, n_k))
        cfg = Scythe.DerivedConfig(:bhyp, 1.0e-7, :bhyp, 1.0e-7, :none, 1.0, :none,
                                   (1.0e-12, 1.0e-2, 1.0e-16, 1.0e-16), 1, false, false,
                                   8.0e6, 100.0, "test", false, :eos, :bhyp, VMU)
        d = Scythe.mc_derived_fields(z, slots, ref, cfg)
        @test d.rho_v == Scythe.ahyp.(nu .+ reshape(nub, 1, n_k), VMU)
        mask = trues(n_i, n_k); mask[2, 7] = false
        @test maximum(abs.(d.rho_v[mask] .- rv[mask]) ./ rv[mask]) < 1.0e-12
        @test d.rho_v[2, 7] === 0.0
        @test all(d.T_eos .=== Scythe.eos_temperature.(p, d.rho_d, d.rho_v))
        @test d.vapor_lobe[2, 7] ≈ Scythe.ahyp_signed(-1.0e-5, VMU) rtol = 1.0e-12
        @test count(!iszero, d.vapor_lobe) == 1
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
    @testset "checkpoint: a density-vapor .jld2 converts on load into a :bhyp model" begin
        mktempdir() do tmp
            # a :none tile with a prognostic vapor bump, written as a checkpoint
            # (a NEGATIVE vapor region above 8 km, slot −2 ρ̄_v, so the odd-continuation branch
            # of the map is exercised)
            d0 = joinpath(tmp, "a") * "/"
            function seed_ck!(p, x, V, c)
                F.vt_seed_vapor!(p, x, V, c; amp = 0.0)
                rv_i = Scythe.mc_slot(V, "rho_v")
                kD = length(c.Tk)
                for i in 1:size(p.physical, 1)
                    x[i, end] > 8.0e3 && (p.physical[i, rv_i, 1] = -2.0 * c.rho_v[mod1(i, kD)])
                end
            end
            m0, p0, mo0 = F.vt_tile(tmp, "ck0"; vtr = :none, output_dir = d0, seed! = seed_ck!)
            Scythe.write_restart(p0, mo0, 0.0)
            path = Scythe.checkpoint_path(d0, 0.0)
            @test isfile(path)
            # the same configuration under :bhyp, initialised from that file
            m1, p1, mo1 = F.vt_tile(tmp, "ck1"; vtr = :bhyp, initial_conditions = path)
            patch = createGrid(mo1.grid_params)
            Scythe.load_initial_conditions!(patch, mo1)
            V0 = mo0.grid_params.vars; V1 = mo1.grid_params.vars
            rv0 = V0["rho_v"]; rv1 = V1["nu_v"]
            @test rv0 == rv1
            kDim = mo1.grid_params.kDim
            nc = div(size(patch.physical, 1), kDim)
            rvbar = repeat(m1.mc_ref_diag.rho_vbar, nc)
            nubar = repeat(Scythe.mc_nu_vbar(m1), nc)
            # every other slot is the checkpoint's own fit, bitwise
            for v in 1:length(V1)
                v == rv1 && continue
                @test patch.spectral[:, v] == p0.spectral[:, v]
            end
            # THE POINTWISE MAP, exactly: bhyp(ρ) − ν̄_v where ρ ≥ 0, −bhyp(−ρ) − ν̄_v below
            col0 = copy(p0.physical[:, rv0, 1])
            rho0 = col0 .+ rvbar
            @test count(<(0.0), rho0) > 0 && count(>(0.0), rho0) > 0
            nn = Scythe.vapor_slot_from_density!(col0, m1.mc_ref_diag.rho_vbar,
                                                 Scythe.mc_nu_vbar(m1), kDim, VMU)
            @test nn == count(<(0.0), rho0)
            for g in eachindex(rho0)
                want = rho0[g] >= 0.0 ? Scythe.bhyp(rho0[g], VMU) - nubar[g] :
                                        -Scythe.bhyp(-rho0[g], VMU) - nubar[g]
                @test isapprox(col0[g], want; rtol = 1.0e-12, atol = 1.0e-12 * abs(nubar[g]))
            end
            # ...and through the load (one refit of the converted column): the lobe region
            # stays a ν lobe (recovered 0), the positive region recovers its density
            neg = rho0 .< 0.0
            nu1 = patch.physical[:, rv1, 1] .+ nubar
            @test all(nu1[neg] .< 0.0)
            @test all(Scythe.ahyp.(nu1[neg], VMU) .== 0.0)
            # the vapor: nu' = bhyp(rho_vbar + rho_v') - nu_vbar, refit once
            # the loaded vapor IS the one refit of the mapped column (built independently here:
            # the archive's state, the mapped vapor written in, one spectral + grid transform)
            q = createGrid(mo1.grid_params)
            q.spectral .= p0.spectral
            gridTransform!(q)
            q.physical[:, rv1, 1] .= col0
            spectralTransform!(q)
            gridTransform!(q)
            @test isapprox(patch.physical[:, rv1, 1], q.physical[:, rv1, 1]; rtol = 1.0e-12)
            # and away from the step at 8 km (whose refit rings) the recovered vapor is the
            # archived density to the refit of a smooth field
            zz = repeat(m1.tilepoints[1:kDim, end], nc)
            far = zz .< 5.0e3
            rec = Scythe.ahyp.(patch.physical[:, rv1, 1] .+ nubar, VMU)
            @test maximum(abs.(rec[far .& .!neg] .- rho0[far .& .!neg]) ./ rho0[far .& .!neg]) < 1.0e-2
            # the reverse direction is refused, by name
            d2 = joinpath(tmp, "b") * "/"
            m2, p2, mo2 = F.vt_tile(tmp, "ck2"; vtr = :bhyp, output_dir = d2)
            Scythe.write_restart(p2, mo2, 0.0)
            m3, p3, mo3 = F.vt_tile(tmp, "ck3"; vtr = :none,
                                    initial_conditions = Scythe.checkpoint_path(d2, 0.0))
            err = try
                Scythe.load_initial_conditions!(createGrid(mo3.grid_params), mo3); nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("nu_v", sprint(showerror, err))
            # a CSV initial condition with a density vapor column is refused by name
            csv = joinpath(tmp, "ic_rho_v.csv")
            Scythe.write_ics_csv(csv, p0, Scythe.getGridpoints(p0))
            m4, p4, mo4 = F.vt_tile(tmp, "ck4"; vtr = :bhyp, initial_conditions = csv)
            err = try
                Scythe.load_initial_conditions!(createGrid(mo4.grid_params), mo4); nothing
            catch e
                e
            end
            @test err isa ErrorException
            msg = sprint(showerror, err)
            @test occursin("rho_v", msg) && occursin("vapor_transform = :none", msg)
        end
    end

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
