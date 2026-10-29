# test/test_driver_stages.jl -- the BITWISE GATE of the moist-compressible driver refactor
# (Stream B, reference/HANDOFF_DRIVER_COMPILE_CLIFF.md direction A): `mc_driver!` is split by
# physical stage into `@noinline` functions on plain arrays (src/mc_driver_stages.jl), and
# every split must leave the answer bit for bit what the inline driver produced.
#
# The captures were seeded at bdb0b48 (the inline driver, before any stage was moved) and are
# NEVER re-seeded by the refactor (the three step-2 arms were added at 85e3743, bitwise
# bdb0b48, before stages 5-6 moved): a mismatch is a broken move, not a new reference. Each arm
# runs three full steps (every column, then the refit) of the vapor-transform fixture tile
# (or the MYNN fixture's live column) and sums every bit of the advanced state `var_np1`, of
# the refit patch `physical` and of the water census matrix `mc_water_stats` (which the census
# hooks inside the stages write, so a hook that moved or changed order shows here too).
#
# Arms: the four of test_vapor_transform.jl's `:none` capture (warm, ice, Louis, MYNN), plus
# branch-coverage arms for the ETD pre-compute (stage 16): the vapor control variable `:bhyp`
# with the lobe relaxation and the ice attribution census on (the `lr_*`, `etd_dl*`,
# `MC_ATTR_QR_*` and `J_v` paths), warm and ice; and the warm arm with the cloud-evaporation
# realization off (the `cloud_evap_real = false` branch).

using Test
using Scythe
using Springsteel
using SparseArrays

isdefined(Main, :VaporTransformFixtures) || include("vapor_transform_fixtures.jl")
isdefined(Main, :MYNNTestFixtures) || include("mynn_test_fixtures.jl")

@testset "driver stages: bitwise against the inline driver (bdb0b48 goldens)" begin
    F = Main.VaporTransformFixtures
    M = Main.MYNNTestFixtures
    ck(m, p) = (F.vt_checksum(m.var_np1), F.vt_checksum(p.physical),
                F.vt_checksum(m.mc_water_stats))
    golden = Dict(
        "warm"           => (0x000000000075eb5a0bf508c8ec93540a, 0x000000000d69f52afed27ea5895a471c, 0x00000000000000c2a9f317525cb1b632),
        "ice"            => (0x00000000016cda1b52284c4daf73910d, 0x0000000036052cc5a9f5732255d630dc, 0x00000000000000d540964f0210a02c8d),
        "louis"          => (0x000000000075eb5a14c1e841122097bb, 0x000000000d69f52ba49626ee5f913c89, 0x00000000000000ea41d7864e72bade39),
        "mynn"           => (0x0000000000d704c47e2ba14a067c05da, 0x000000001d59b5a4bf1c701b0f3d7d38, 0x00000000000000f76355a62e89f6911b),
        "warm_bhyp_lobe" => (0x00000000007ebed336c6f996cb8b033a, 0x000000000d723872ac764e9f2ae29079, 0x00000000000000e6d4e12edc6da8526d),
        "ice_bhyp_lobe"  => (0x000000000174a39df18e267cb42a04df, 0x0000000035fcad8e5358f55aa78b1d02, 0x0000000000000182081d2aabcc8be280),
        "warm_noevapreal" => (0x00000000007e2fcc8ccf131803c276ba, 0x000000000d77f43a97962c0e8111c61a, 0x00000000000000c291525a3e2f3fdd8a),
        "warm_precip"    => (0x00000000009224188fbae1716ed8c176, 0x000000000f51da87086431320e067c56, 0x000000000000014a75177c653bbfb71e),
        "ice_nocond"     => (0x0000000000b3ffc21c93eaab97b09ccf, 0x0000000024f39e0ceba69a01916419f4, 0x00000000000000491bdaf6aee82b4624),
        # (ice_natlid equals ice: nothing reaches the lid in three steps; the arm pins the
        # scratch-column branch of the flux-column selection.)
        "ice_natlid"     => (0x00000000016cda1b52284c4daf73910d, 0x0000000036052cc5a9f5732255d630dc, 0x00000000000000d540964f0210a02c8d))
    lobe = Dict{Symbol,Any}(:lobe_relaxation => true, :ice_attr_census => true)
    arms = (
        ("warm",            false, false, nothing, Dict{Symbol,Any}(), true),
        ("ice",             true,  false, nothing, Dict{Symbol,Any}(), true),
        ("louis",           false, true,  nothing, Dict{Symbol,Any}(), true),
        ("warm_bhyp_lobe",  false, false, :bhyp,   lobe, false),
        ("ice_bhyp_lobe",   true,  false, :bhyp,   lobe, false),
        ("warm_noevapreal", false, false, nothing,
            Dict{Symbol,Any}(:cloud_evap_realization => false), true),
        # Step 2 (stages 5-6) branch coverage, seeded at 85e3743 before those stages moved:
        # single-moment precipitating rain (the `cloud_evap_real && precipitation` loop
        # without its `rain_2m` leg), the phase change off (`fill!` branch of the fold and
        # the gated ice sources/population), and the forensics natural-lid flux fits.
        ("warm_precip",     false, false, nothing,
            Dict{Symbol,Any}(:precipitation => true), (; amp = 5.0e-3)),
        ("ice_nocond",      true,  false, nothing,
            Dict{Symbol,Any}(:condensation => false), true),
        ("ice_natlid",      true,  false, nothing,
            Dict{Symbol,Any}(:sedimentation_flux_lid => :natural), true))
    got = Dict{String,Any}()
    mktempdir() do tmp
        for (tag, ice, louis, vtr, extra, vbump) in arms
            # The prognostic vapor bump is written as a density perturbation, which is the
            # slot only under `:none`; the `:bhyp` arms keep the cloud sheet, E_t bump and w.
            # (`vbump` may instead be a NamedTuple of seed keywords, e.g. a thicker cloud.)
            skw = vbump isa NamedTuple ? vbump : (;)
            seed! = vbump !== false ? ((p, x, V, c) -> F.vt_seed_vapor!(p, x, V, c; skw...)) :
                                      ((p, x, V, c) -> F.vt_seed!(p, x, V, c))
            m, p, mo = F.vt_tile(tmp, "ds_$(tag)"; vtr, ice, louis, extra, seed!)
            F.vt_run!(m, p, mo, 3)
            got[tag] = ck(m, p)
            @testset "$(tag)" begin
                @test got[tag][1] == golden[tag][1]
                @test got[tag][2] == golden[tag][2]
                @test got[tag][3] == golden[tag][3]
            end
        end
        m, p, mo, gp, z, col = M.make_mynn_tile(tmp; num_cells_k = 25, kMax = 5.0e3,
                                                ctrans = :none, rtrans = :none)
        M.set_state!(p, mo, gp.kDim, z, col; W0 = 0.3)
        for t in 1:3
            M.step_all!(m, p, gp.kDim, t); Scythe.calcTendency(m); gridTransform!(p)
        end
        got["mynn"] = ck(m, p)
        @testset "mynn" begin
            @test got["mynn"][1] == golden["mynn"][1]
            @test got["mynn"][2] == golden["mynn"][2]
            @test got["mynn"][3] == golden["mynn"][3]
        end
    end
    get(ENV, "SCYTHE_PRINT_GOLDENS", "") == "1" &&
        foreach(k -> println("GOLDEN driver_stages \"$(k)\" => $(got[k])"), sort(collect(keys(got))))
end
