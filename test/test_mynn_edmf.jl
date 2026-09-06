# ── Parity tests: pure-Julia DMP_mf vs. the ccpp-physics Fortran ──────────────
#
# Part 3 of the port (src/mynn_edmf.jl) — the mass-flux plumes — checked against
# tools/mynn_fortran_driver/ref_driver_output_r8.txt, the dump of a VERBATIM
# ccpp-physics `module_bl_mynn.F90` run on five columns at double precision with FMA
# contraction off. No Fortran runs here; the reference is a checked-in text file
# parsed by test/reference/mynn_fortran_refs.jl. The same conventions as
# test/test_mynn_closure.jl apply (mode A / mode B, the GATE lines, the comparison
# metric `|a-b| / max(|a|,|b|)`).
#
# Run standalone:
#
#     julia --project=. test/test_mynn_edmf.jl
#
# or `include` from test/runtests.jl. Everything defined here is prefixed `edmf_` /
# `EDMF_` so that including it alongside test/test_mynn_closure.jl redefines nothing.
#
# ── WHAT IS AND IS NOT TESTED, precisely ─────────────────────────────────────
#
# STEP 1, case4_convective — FULLY tested, standalone. Every input of `DMP_mf` at
# step 1 is either the frozen column, a printed mode-B scalar, or reproducible here:
# `qke` is the printed `qke_pre_predict`, and `vt/vq/sgm/qc_bl/cldfra_bl` are
# recomputed by calling the ported `mym_condensation!` on the printed init state
# (`init_tsq/init_qsq/init_cov/init_sh/init_el/init_pblh`), which is exactly what the
# driver hands it at step 1. All 26 output blocks are compared.
#
# STEP 30, case4_convective — tested THROUGH THE PORTED CHAIN, not standalone. The
# step-30 inputs `vt/vq/sgm/qc_bl/cldfra_bl` are `mym_condensation!`'s step-30 output,
# which the reference does NOT print (for a case with plumes the printed blocks are
# post-DMP_mf; here they happen to be identical, see the next paragraph, but the
# reference cannot be used to prove that in advance), and `mym_condensation!`'s own
# step-30 inputs include the step-29 `tsq/qsq/cov`, which no block carries either. So
# step 30 is exercised by running `mynn_init_column!` + 30 x `mynn_column_step_edmf!`
# and comparing the plume outputs AT step 30 with the mode-B blocks. That is a
# STRONGER statement than a standalone call (it also pins the 29 steps that produced
# the inputs) but a WEAKER isolation: a compensating error in `mym_condensation!` and
# `dmp_mf!` could in principle cancel. The step-1 testset is the isolated one.
#
# THE CHABOUREAU-BECHTOLD BLOCK gets NO coverage from the main reference: `edmf_qc`
# is identically zero at every level of every case at both steps (case 4's plumes top
# out at 653 m and `condensation_edmf` forces `qc = 0` below 100 m anyway), so
# `0.5*(edmf_qc(k)+edmf_qc(k-1)) > 0` is never true and :6631-6766 never runs. Two
# things stand in for it:
#   * the CONSEQUENCE is asserted — `dmp_mf!` must leave `vt, vq, cldfra_bl, qc_bl`
#     exactly as `mym_condensation!` left them at every level, the five plume levels
#     included (rtol 1e-9 against the printed post-DMP_mf blocks); and
#   * a SPOT-CHECK reference, tools/mynn_fortran_driver/edmf_ref_driver.f90, calls
#     `DMP_mf` directly on a synthetic saturating column, over water AND over land,
#     and its checked-in output covers :6631-6766, the `maxqc >= 1e-8` (moist plume)
#     `maxmf` branch and every `landsea < 1.5` branch — none of which the five
#     reference columns reach. See the "spot check" testset below.
# STILL UNTESTED after all that: the subsidence/detrainment block (`env_subs =
# .false.` at compile time, so it cannot run at all; the reference prints its nine
# outputs as identically zero, which IS asserted), and with it `dzp`, the `envm_*`
# arrays and the Fortran out-of-bounds `rhoz(kte+1)` read that note (c) of
# src/mynn_edmf.jl describes.

using Test
using LinearAlgebra
using Springsteel

# ── Wiring: use Scythe's copy if it has one, else include the sources ─────────
const _EDMF_EXPORTS = (:MYNNConstants, :MYNNWork, :MYNNColumn, :MYNNColumnState,
                       :MYNNOptions, :mynn_wall_heights, :mym_condensation!,
                       :mynn_init_column!, :mynn_column_step!,
                       # part 3
                       :MYNN_NUP, :EDMFWork, :EDMFGate, :condensation_edmf,
                       :dmp_mf!, :mynn_column_step_edmf!,
                       # closure helpers the tests call unqualified
                       :qsat_blend, :esat_blend, :xl_blend,
                       # constants the tests reference unqualified (MYNN_CASES and
                       # MYNN_DRIVER_DIR are test-local, from the refs parser)
                       :MYNN_P1000MB)

if isdefined(Main, :Scythe) && all(nm -> isdefined(Main.Scythe, nm), _EDMF_EXPORTS)
    for _nm in _EDMF_EXPORTS
        @eval const $_nm = getfield(Main.Scythe, $(QuoteNode(_nm)))
    end
else
    isdefined(Main, :MYNNConstants) ||
        include(joinpath(@__DIR__, "..", "src", "mynn_constants.jl"))
    isdefined(Main, :MYNNWork) ||
        include(joinpath(@__DIR__, "..", "src", "mynn_closure.jl"))
    isdefined(Main, :dmp_mf!) ||
        include(joinpath(@__DIR__, "..", "src", "mynn_edmf.jl"))
end

if !(@isdefined mynn_reference)
    include(joinpath(@__DIR__, "reference", "mynn_fortran_refs.jl"))
end

# ── Comparison helpers (own names: test_mynn_closure.jl has its own copies) ────

"""
    edmf_maxrel(got, ref) -> (maxerr, k)

Largest relative difference `|a-b| / max(|a|,|b|)` and where it occurs. NaNs must
match position for position; a NaN mismatch or a `0` against a nonzero returns `Inf`.
"""
function edmf_maxrel(got::AbstractVector, ref::AbstractVector)
    length(got) == length(ref) || return (Inf, 0)
    m = 0.0; km = 0
    for k in eachindex(ref)
        g = got[k]; r = ref[k]
        if isnan(g) || isnan(r)
            isequal(g, r) || return (Inf, k)
            continue
        end
        d = abs(g - r)
        d == 0.0 && continue
        s = max(abs(g), abs(r))
        e = s == 0.0 ? Inf : d/s
        if e > m; m = e; km = k; end
    end
    return (m, km)
end
edmf_relscalar(g::Real, r::Real) = edmf_maxrel([Float64(g)], [Float64(r)])[1]

const EDMF_OBSERVED = Dict{String,Float64}()
function edmf_note!(name::AbstractString, err::Real)
    EDMF_OBSERVED[name] = max(get(EDMF_OBSERVED, name, 0.0), Float64(err))
    return err
end

# The 26 blocks the reference prints for DMP_mf, in the driver's print order.
const EDMF_LAYER_BLOCKS = ("edmf_a", "edmf_w", "edmf_qt", "edmf_thl", "edmf_ent",
                           "edmf_qc")
const EDMF_FACE_BLOCKS  = ("s_aw", "s_awthl", "s_awqt", "s_awqv", "s_awqc",
                           "s_awu", "s_awv", "s_awqke")
const EDMF_ZERO_BLOCKS  = ("sub_thl", "sub_sqv", "sub_u", "sub_v",
                           "det_thl", "det_sqv", "det_sqc", "det_u", "det_v")
const EDMF_ALL_BLOCKS   = (EDMF_LAYER_BLOCKS..., EDMF_FACE_BLOCKS...,
                           EDMF_ZERO_BLOCKS...)

edmf_out(ew::EDMFWork, name::AbstractString) = getfield(ew, Symbol(name))

# ── The tests ─────────────────────────────────────────────────────────────────

@testset "MYNN-EDMF mass flux (DMP_mf) vs. ccpp-physics reference" begin
    ref = mynn_reference()
    c   = MYNNConstants()

    edmf_colstruct(cas) = let cm = mynn_column(cas)
        MYNNColumn(; ps = cm.ps, ts = cm.ts, ust = cm.ust, hfx = cm.hfx, qfx = cm.qfx,
                   wspd = cm.wspd, xland = cm.xland, dx = cm.dx, rmol0 = cm.rmol,
                   dz = cm.dz, u = cm.u, v = cm.v, w = cm.w, T = cm.T, th = cm.th,
                   exner = cm.exner, p = cm.p, rho = cm.rho, sqv = cm.sqv,
                   sqc = cm.sqc, sqi = cm.sqi, z = cm.z, qsfc = cm.qsfc, znt = cm.znt)
    end

    ktop_at(cas, step) = Int(mynn_scalar(ref, cas, 2.5, "B", step, "ktop_plume"))
    plume_free = [cas for cas in MYNN_CASES
                  if ktop_at(cas, 1) == 0 && ktop_at(cas, 30) == 0]

    # One `dmp_mf!` call with the reference harness's argument set. `qke`, `vt`, `vq`,
    # `sgm`, `qc_bl`, `cldfra_bl` are the caller's (they are inout); everything else
    # comes from the column and the printed mode-B scalars of that step.
    function edmf_call!(ew, work, cas, step, qke, vt, vq, sgm, qc_bl, cld)
        cm = mynn_column(cas)
        n  = cm.n
        zw = mynn_face_heights(cm.dz)
        thl, sqw, thetav = mynn_conserved(cm, c)
        zed = zeros(n)
        return dmp_mf!(1, n, cm.delt, zw, cm.dz, cm.p, cm.rho, 1, 0, 1,
            cm.u, cm.v, cm.w, cm.th, thl, thetav, cm.T, sqw, cm.sqv, cm.sqc, qke,
            cm.exner, vt, vq, sgm,
            cm.ust,
            mynn_scalar(ref, cas, 2.5, "B", step, "flt"),
            mynn_scalar(ref, cas, 2.5, "B", step, "fltv"),
            mynn_scalar(ref, cas, 2.5, "B", step, "flq"),
            cm.qfx/cm.rho[1],                                   # flqv, mode_b :357
            mynn_scalar(ref, cas, 2.5, "B", step, "pblh"),
            Int(mynn_scalar(ref, cas, 2.5, "B", step, "kpbl")),
            cm.dx, cm.xland, cm.ts/cm.exner[1],                 # th_sfc, README item 2
            ew.edmf_a, ew.edmf_w, ew.edmf_qt, ew.edmf_thl, ew.edmf_ent, ew.edmf_qc,
            ew.s_aw, ew.s_awthl, ew.s_awqt, ew.s_awqv, ew.s_awqc, ew.s_awu, ew.s_awv,
            ew.s_awqke,
            ew.sub_thl, ew.sub_sqv, ew.sub_u, ew.sub_v,
            ew.det_thl, ew.det_sqv, ew.det_sqc, ew.det_u, ew.det_v,
            qc_bl, cld, zed, zed, true, true,
            mynn_scalar(ref, cas, 2.5, "B", step, "psig_shcu"), 0, zed, c, ew)
    end

    # `mym_condensation!` on the printed INIT state — what the driver hands DMP_mf at
    # step 1 — returning (vt, vq, sgm, qc_bl, qi_bl, cldfra_bl).
    function edmf_condensation_step1(cas, work)
        cm = mynn_column(cas); n = cm.n
        zw = mynn_face_heights(cm.dz)
        thl, sqw, _ = mynn_conserved(cm, c)
        zed = zeros(n)
        vt = zeros(n); vq = zeros(n); sgm = zeros(n)
        qc_bl = zeros(n); qi_bl = zeros(n); cld = zeros(n)
        mym_condensation!(1, n, cm.dx, cm.dz, zw, cm.xland, thl, sqw, cm.sqv, cm.sqc,
            cm.sqi, zed, cm.p, cm.exner,
            mynn_block(ref, cas, 2.5, "B", 0, "init_tsq"),
            mynn_block(ref, cas, 2.5, "B", 0, "init_qsq"),
            mynn_block(ref, cas, 2.5, "B", 0, "init_cov"),
            mynn_block(ref, cas, 2.5, "B", 0, "init_sh"),
            mynn_block(ref, cas, 2.5, "B", 0, "init_el"), 2,
            qc_bl, qi_bl, cld,
            mynn_scalar(ref, cas, 2.5, "B", 0, "init_pblh"), cm.hfx,
            vt, vq, cm.th, sgm, cm.rmol, 0, zed, c, work)
        return (vt, vq, sgm, qc_bl, qi_bl, cld)
    end

    @testset "reference harness self-consistency" begin
        for cas in MYNN_CASES
            @test haskey(ref.gate, cas) && ref.gate[cas]
        end
        # case4_convective is the ONLY case with plumes, at both steps.
        @test ktop_at("case4_convective", 1)  == 5
        @test ktop_at("case4_convective", 30) == 5
        @test plume_free == ["case1_rest", "case2_o01_sea", "case3_tc_rmw",
                             "case5_highwind"]
        # env_subs = .false.: the nine subsidence/detrainment outputs are zero
        # everywhere in the reference, which is what this port must reproduce.
        for cas in MYNN_CASES, step in (1, 30), nm in EDMF_ZERO_BLOCKS
            @test all(iszero, mynn_block(ref, cas, 2.5, "B", step, nm))
        end
        # ... and so is edmf_qc, which is why the Chaboureau-Bechtold block never
        # runs. If this ever changes, the untested code above becomes testable.
        for cas in MYNN_CASES, step in (1, 30)
            @test all(iszero, mynn_block(ref, cas, 2.5, "B", step, "edmf_qc"))
        end
    end

    @testset "the activation gate, case by case" begin
        # The three parts of :6039 (`fltv2 > 0.002 .and. maxwidth > minwidth .and.
        # superadiabatic`) for every case at step 1, so a reader can see WHY the four
        # plume-free cases produce nothing. Printed as well as asserted.
        println("\nDMP_mf activation gate at step 1 " *
                "(fltv2 > 0.002 && maxwidth > minwidth && superadiabatic):")
        println("  ", rpad("case", 18), rpad("fltv2", 13), rpad("maxwidth", 11),
                rpad("minwidth", 10), rpad("superadb", 10), rpad("nup2", 6),
                rpad("ktop", 6), "gate")
        expected_active = Dict("case1_rest" => false, "case2_o01_sea" => true,
                               "case3_tc_rmw" => false, "case4_convective" => true,
                               "case5_highwind" => true)
        for cas in MYNN_CASES
            cm = mynn_column(cas); n = cm.n
            work = MYNNWork(n); ew = EDMFWork(n)
            vt, vq, sgm, qc_bl, _, cld = edmf_condensation_step1(cas, work)
            qke = copy(mynn_block(ref, cas, 2.5, "B", 1, "qke_pre_predict"))
            g = edmf_call!(ew, work, cas, 1, qke, vt, vq, sgm, qc_bl, cld)
            println("  ", rpad(cas, 18),
                    rpad(round(g.fltv2; sigdigits = 6), 13),
                    rpad(round(g.maxwidth; digits = 2), 11),
                    rpad(round(g.minwidth; digits = 2), 10),
                    rpad(g.superadiabatic, 10), rpad(g.nup2, 6),
                    rpad(g.ktop, 6), g.active)
            @test g.active == expected_active[cas]
            @test g.ktop == ktop_at(cas, 1)
            # The gate is NOT the whole story: cases 2 and 5 PASS it and still make no
            # plume — every one of the 8 fails to leave the surface interface, which
            # sets nup2 = 0 at :6288 and suppresses the flux entirely (note (e) of
            # src/mynn_edmf.jl). Pin that distinction.
            if cas in ("case2_o01_sea", "case5_highwind")
                @test g.active && g.nup2 == 0 && g.ktop == 0 && g.maxwidth > 0.0
            elseif cas in ("case1_rest", "case3_tc_rmw")
                @test !g.active && g.nup2 == 0 && g.maxwidth == 0.0
            else
                @test g.active && g.nup2 == MYNN_NUP && g.ktop == 5
            end
        end
        println()
    end

    @testset "case4_convective, step 1 (standalone)" begin
        cas = "case4_convective"
        cm  = mynn_column(cas); n = cm.n
        work = MYNNWork(n); ew = EDMFWork(n)
        vt, vq, sgm, qc_bl, qi_bl, cld = edmf_condensation_step1(cas, work)
        qke = copy(mynn_block(ref, cas, 2.5, "B", 1, "qke_pre_predict"))
        g = edmf_call!(ew, work, cas, 1, qke, vt, vq, sgm, qc_bl, cld)

        for nm in EDMF_ALL_BLOCKS
            e, k = edmf_maxrel(edmf_out(ew, nm), mynn_block(ref, cas, 2.5, "B", 1, nm))
            @test edmf_note!("dmp_mf! step 1 ($nm)", e) <= 1e-7
        end
        @test g.ktop == Int(mynn_scalar(ref, cas, 2.5, "B", 1, "ktop_plume"))
        @test edmf_note!("dmp_mf! step 1 (maxwidth)",
            edmf_relscalar(g.maxwidth, mynn_scalar(ref, cas, 2.5, "B", 1, "maxwidth"))) <= 1e-9
        @test edmf_note!("dmp_mf! step 1 (maxmf)",
            edmf_relscalar(g.maxmf, mynn_scalar(ref, cas, 2.5, "B", 1, "maxmf"))) <= 1e-9
        @test edmf_note!("dmp_mf! step 1 (ztop_plume)",
            edmf_relscalar(g.ztop, mynn_scalar(ref, cas, 2.5, "B", 1, "ztop_plume"))) <= 1e-9
        # a DRY plume: maxmf comes back negative (:6771-6775)
        @test g.maxmf < 0.0 && all(iszero, ew.edmf_qc)

        # DMP_mf must have left the subgrid cloud alone: edmf_qc == 0 everywhere means
        # the Chaboureau-Bechtold block never fired, so the printed post-DMP_mf
        # vt/vq/cldfra_bl/qc_bl are still mym_condensation's output.
        for (nm, got) in (("vt", vt), ("vq", vq), ("cldfra_bl", cld), ("qc_bl", qc_bl),
                          ("qi_bl", qi_bl), ("sgm", sgm))
            e, k = edmf_maxrel(got, mynn_block(ref, cas, 2.5, "B", 1, nm))
            @test edmf_note!("untouched by dmp_mf! ($nm)", e) <= 1e-9
        end
        # `qke` is intent(in): DMP_mf may not write it.
        @test qke == mynn_block(ref, cas, 2.5, "B", 1, "qke_pre_predict")

        # the plume is exactly 5 interfaces deep and nothing lives above it
        @test all(iszero, @view ew.edmf_a[6:end])
        @test all(iszero, @view ew.s_aw[7:end])
        @test count(!iszero, ew.edmf_a) == 5
    end

    @testset "plume-free cases (1, 2, 3, 5): no flux, nothing touched" begin
        # For these the printed post-DMP_mf vt/vq/cldfra_bl/qc_bl ARE
        # mym_condensation's output (ktop = 0, so :6631-6766 cannot have run), which
        # makes them usable as inputs at BOTH steps — including step 30, whose
        # mym_condensation output is otherwise unavailable.
        for cas in plume_free, step in (1, 30)
            cm = mynn_column(cas); n = cm.n
            work = MYNNWork(n); ew = EDMFWork(n)
            vt  = copy(mynn_block(ref, cas, 2.5, "B", step, "vt"))
            vq  = copy(mynn_block(ref, cas, 2.5, "B", step, "vq"))
            sgm = copy(mynn_block(ref, cas, 2.5, "B", step, "sgm"))
            qcb = copy(mynn_block(ref, cas, 2.5, "B", step, "qc_bl"))
            cld = copy(mynn_block(ref, cas, 2.5, "B", step, "cldfra_bl"))
            qke = copy(mynn_block(ref, cas, 2.5, "B", step, "qke_pre_predict"))
            vt0 = copy(vt); vq0 = copy(vq); sgm0 = copy(sgm)
            qcb0 = copy(qcb); cld0 = copy(cld)

            g = edmf_call!(ew, work, cas, step, qke, vt, vq, sgm, qcb, cld)

            @test g.ktop == 0
            @test g.ztop == 0.0
            @test g.maxmf == 0.0
            @test g.nup2 == 0
            for nm in EDMF_ALL_BLOCKS
                # EXACTLY zero, not "small": nothing may leak out of an inactive call
                @test all(iszero, edmf_out(ew, nm))
                @test all(iszero, mynn_block(ref, cas, 2.5, "B", step, nm))
            end
            # maxwidth is the one output an inactive call can still carry: it is
            # zeroed only when the WIDTH criterion is what failed (:6035-6037).
            @test edmf_note!("dmp_mf! inactive (maxwidth)",
                edmf_relscalar(g.maxwidth,
                    mynn_scalar(ref, cas, 2.5, "B", step, "maxwidth"))) <= 1e-9
            # the four inout cloud fields come back untouched, bit for bit
            @test isequal(vt, vt0) && isequal(vq, vq0) && isequal(sgm, sgm0)
            @test isequal(qcb, qcb0) && isequal(cld, cld0)
        end
    end

    @testset "case4_convective, step 30 (through the ported chain)" begin
        # See the header: step 30's DMP_mf inputs are not printed, so they are
        # produced by running the chain. This pins the step-30 blocks but does not
        # isolate DMP_mf the way the step-1 testset does.
        cas = "case4_convective"
        cm = mynn_column(cas); n = cm.n
        col = edmf_colstruct(cas)
        work = MYNNWork(n); ew = EDMFWork(n); st = MYNNColumnState(n)
        opts = MYNNOptions(; closure = 2.5, delt = cm.delt)
        mynn_init_column!(work, c, col, st, opts)
        local g
        for _ in 1:30
            _, _, g = mynn_column_step_edmf!(work, ew, c, col, st, opts)
        end
        for nm in EDMF_ALL_BLOCKS
            e, k = edmf_maxrel(edmf_out(ew, nm), mynn_block(ref, cas, 2.5, "B", 30, nm))
            @test edmf_note!("dmp_mf! step 30 ($nm)", e) <= 1e-7
        end
        @test g.ktop == Int(mynn_scalar(ref, cas, 2.5, "B", 30, "ktop_plume"))
        @test edmf_note!("dmp_mf! step 30 (maxwidth)",
            edmf_relscalar(g.maxwidth, mynn_scalar(ref, cas, 2.5, "B", 30, "maxwidth"))) <= 1e-9
        @test edmf_note!("dmp_mf! step 30 (maxmf)",
            edmf_relscalar(g.maxmf, mynn_scalar(ref, cas, 2.5, "B", 30, "maxmf"))) <= 1e-9
        @test edmf_note!("dmp_mf! step 30 (ztop_plume)",
            edmf_relscalar(g.ztop, mynn_scalar(ref, cas, 2.5, "B", 30, "ztop_plume"))) <= 1e-9
        # the modified-in-place cloud fields at step 30 (unmodified, as at step 1)
        for (nm, got) in (("vt", st.vt), ("vq", st.vq), ("sgm", st.sgm),
                          ("cldfra_bl", st.cldfra_bl), ("qc_bl", st.qc_bl),
                          ("qi_bl", st.qi_bl))
            e, k = edmf_maxrel(got, mynn_block(ref, cas, 2.5, "B", 30, nm))
            @test edmf_note!("step 30 cloud ($nm)", e) <= 1e-9
        end
    end

    @testset "mynn_column_step_edmf! (30 steps) vs. mode A" begin
        # case1_rest is excluded for the ust = 0 NaN divergence that
        # test/test_mynn_closure.jl pins; that exclusion is asserted, not assumed.
        @test mynn_column("case1_rest").ust == 0.0
        e2e = [cas for cas in MYNN_CASES if mynn_column(cas).ust > 0.0]
        @test e2e == ["case2_o01_sea", "case3_tc_rmw", "case4_convective",
                      "case5_highwind"]

        for cas in e2e
            cm = mynn_column(cas); n = cm.n
            col = edmf_colstruct(cas)
            work = MYNNWork(n); ew = EDMFWork(n); st = MYNNColumnState(n)
            opts = MYNNOptions(; closure = 2.5, delt = cm.delt)
            mynn_init_column!(work, c, col, st, opts)
            for step in 1:30
                mynn_column_step_edmf!(work, ew, c, col, st, opts)
                (step == 1 || step == 30) || continue
                pairs = (("qke", st.qke), ("el_pbl", st.el), ("sh3d", st.sh),
                         ("sm3d", st.sm), ("tsq", st.tsq), ("qsq", st.qsq),
                         ("cov", st.cov), ("cldfra_bl", st.cldfra_bl),
                         ("exch_h", work.out_kh), ("rublten", work.out_du),
                         ("rthblten", work.out_dth), ("rqvblten", work.out_dqv))
                for (nm, got) in pairs
                    e, k = edmf_maxrel(got, mynn_block(ref, cas, 2.5, "A", step, nm))
                    @test edmf_note!("mynn_column_step_edmf! ($nm)", e) <= 1e-9
                end
                @test edmf_relscalar(st.pblh,
                        mynn_scalar(ref, cas, 2.5, "A", step, "pblh")) <= 1e-12
                @test st.kpbl == Int(mynn_scalar(ref, cas, 2.5, "A", step, "kpbl"))
            end
        end
    end

    @testset "reproduces the edmf = false path bitwise where no plume fires" begin
        # `mynn_column_step_edmf!` is a COPY of `mynn_column_step!` with dmp_mf!
        # spliced in. On a column whose plumes never fire, every array it adds is
        # identically zero, so the two must agree BIT FOR BIT — which is what keeps
        # the copy honest while src/mynn_closure.jl is off-limits to this stage.
        for cas in plume_free
            cm = mynn_column(cas); n = cm.n
            opts = MYNNOptions(; closure = 2.5, delt = cm.delt)

            colA = edmf_colstruct(cas); wA = MYNNWork(n); stA = MYNNColumnState(n)
            mynn_init_column!(wA, c, colA, stA, opts)
            for _ in 1:30
                mynn_column_step!(wA, c, colA, stA, opts; edmf = false)
            end

            colB = edmf_colstruct(cas); wB = MYNNWork(n); stB = MYNNColumnState(n)
            eB = EDMFWork(n)
            mynn_init_column!(wB, c, colB, stB, opts)
            for _ in 1:30
                mynn_column_step_edmf!(wB, eB, c, colB, stB, opts)
            end

            for f in (:qke, :el, :sh, :sm, :tsq, :qsq, :cov, :cldfra_bl, :qc_bl,
                      :qi_bl, :vt, :vq, :sgm)
                @test isequal(getfield(stA, f), getfield(stB, f))
            end
            @test stA.pblh === stB.pblh && stA.kpbl == stB.kpbl &&
                  stA.rmol === stB.rmol
            for f in (:out_du, :out_dv, :out_dth, :out_dqv, :out_dqc, :out_dqi,
                      :out_km, :out_kh, :out_dfm, :out_dfh, :out_dfq,
                      :out_diss_heat)
                @test isequal(getfield(wA, f), getfield(wB, f))
            end
        end
    end

    @testset "spot check: the Chaboureau-Bechtold block and the land branches" begin
        # tools/mynn_fortran_driver/edmf_ref_driver.f90 calls DMP_mf directly on a
        # synthetic column (case4_convective moistened 50 %, a deep strongly heated
        # PBL) whose plumes DO saturate, twice: once over water and once over land.
        # That is the only coverage of :6631-6766, of the `maxqc >= 1e-8` (MOIST
        # plume) maxmf branch, and of every `landsea < 1.5` branch. Its output is
        # checked in and parsed by the same reader as the main reference.
        spot_path = joinpath(MYNN_DRIVER_DIR, "edmf_ref_driver_output.txt")
        @test isfile(spot_path)
        spot = mynn_reference(spot_path)

        cm = mynn_column("case4_convective"); n = cm.n
        zw = mynn_face_heights(cm.dz)
        # Rebuild edmf_ref_driver.f90's synthetic state. Every line here mirrors one
        # line of `run_spot`; the column file round-trips exactly at %.17g and each
        # transformation is a single IEEE double operation, so the two agree bitwise.
        qv  = cm.sqv .* 1.5
        qt  = copy(qv)
        qc  = zeros(n)
        thl = copy(cm.th)                              # qc = qi = 0, so thl == th
        thv = [cm.th[k]*(1.0 + c.p608*qv[k]) for k in 1:n]
        qke = fill(1.0, n)
        th_sfc = cm.ts/cm.exner[1]
        dt = 20.0; ust = 0.4; pblh = 1500.0; kpbl = 8
        flt = 0.3; flq = 3.0e-4; flqv = flq
        fltv = flt + flq*c.p608*th_sfc
        psig_shcu = 1.0

        for (cas, landsea) in (("spot_moist_water", 2.0), ("spot_moist_land", 1.0))
            ew = EDMFWork(n)
            vt = zeros(n); vq = zeros(n); sgm = zeros(n)
            qcb = zeros(n); cld = zeros(n); zed = zeros(n)
            g = dmp_mf!(1, n, dt, zw, cm.dz, cm.p, cm.rho, 1, 0, 1,
                cm.u, cm.v, cm.w, cm.th, thl, thv, cm.T, qt, qv, qc, qke,
                cm.exner, vt, vq, sgm, ust, flt, fltv, flq, flqv, pblh, kpbl,
                cm.dx, landsea, th_sfc,
                ew.edmf_a, ew.edmf_w, ew.edmf_qt, ew.edmf_thl, ew.edmf_ent, ew.edmf_qc,
                ew.s_aw, ew.s_awthl, ew.s_awqt, ew.s_awqv, ew.s_awqc, ew.s_awu,
                ew.s_awv, ew.s_awqke,
                ew.sub_thl, ew.sub_sqv, ew.sub_u, ew.sub_v,
                ew.det_thl, ew.det_sqv, ew.det_sqc, ew.det_u, ew.det_v,
                qcb, cld, zed, zed, true, true, psig_shcu, 0, zed, c, ew)

            # the branch this exists for actually ran
            @test g.ktop == 22
            @test count(!iszero, ew.edmf_qc) == 19    # the plumes condensed
            @test count(!iszero, cld) == 19           # ... and made cloud
            @test g.maxmf > 0.0                       # MOIST plume: maxmf not flipped

            for nm in EDMF_ALL_BLOCKS
                e, k = edmf_maxrel(edmf_out(ew, nm),
                                   mynn_block(spot, cas, 2.5, "B", 1, nm))
                @test edmf_note!("spot $cas ($nm)", e) <= 1e-9
            end
            # the four fields the Chaboureau-Bechtold block OVERWRITES
            for (nm, got) in (("vt", vt), ("vq", vq), ("cldfra_bl", cld),
                              ("qc_bl", qcb))
                e, k = edmf_maxrel(got, mynn_block(spot, cas, 2.5, "B", 1, nm))
                @test edmf_note!("spot $cas (CB $nm)", e) <= 1e-9
            end
            @test g.ktop == Int(mynn_scalar(spot, cas, 2.5, "B", 1, "ktop_plume"))
            for (nm, got) in (("maxwidth", g.maxwidth), ("maxmf", g.maxmf),
                              ("ztop_plume", g.ztop))
                @test edmf_note!("spot $cas ($nm)",
                    edmf_relscalar(got, mynn_scalar(spot, cas, 2.5, "B", 1, nm))) <= 1e-9
            end
        end

        # water and land really do take different branches (mf_cf's lower bound is
        # 1.2*Aup over water and 1.8*Aup over land, :6689-6700; acfac, exc_fac,
        # width_flx and hux differ too), so the two blocks must NOT be identical.
        @test mynn_block(spot, "spot_moist_water", 2.5, "B", 1, "cldfra_bl") !=
              mynn_block(spot, "spot_moist_land",  2.5, "B", 1, "cldfra_bl")
    end

    @testset "argument guards" begin
        cas = "case4_convective"
        cm = mynn_column(cas); n = cm.n
        work = MYNNWork(n); ew = EDMFWork(n)
        vt, vq, sgm, qcb, _, cld = edmf_condensation_step1(cas, work)
        qke = copy(mynn_block(ref, cas, 2.5, "B", 1, "qke_pre_predict"))
        zw = mynn_face_heights(cm.dz)
        thl, sqw, thetav = mynn_conserved(cm, c)
        zed = zeros(n)
        bad(nn, ework, fqc, fqi, kts) = dmp_mf!(kts, nn, cm.delt, zw, cm.dz, cm.p,
            cm.rho, 1, 0, 1, cm.u, cm.v, cm.w, cm.th, thl, thetav, cm.T, sqw, cm.sqv,
            cm.sqc, qke, cm.exner, vt, vq, sgm, cm.ust, 0.1, 0.1, 1e-4, 1e-4, 900.0,
            6, cm.dx, cm.xland, 303.0,
            ew.edmf_a, ew.edmf_w, ew.edmf_qt, ew.edmf_thl, ew.edmf_ent, ew.edmf_qc,
            ew.s_aw, ew.s_awthl, ew.s_awqt, ew.s_awqv, ew.s_awqc, ew.s_awu, ew.s_awv,
            ew.s_awqke, ew.sub_thl, ew.sub_sqv, ew.sub_u, ew.sub_v, ew.det_thl,
            ew.det_sqv, ew.det_sqc, ew.det_u, ew.det_v, qcb, cld, zed, zed,
            fqc, fqi, 1.0, 0, zed, c, ework)
        @test_throws ArgumentError bad(n, ew, false, true,  1)   # F_QC
        @test_throws ArgumentError bad(n, ew, true,  false, 1)   # F_QI
        @test_throws ArgumentError bad(n, EDMFWork(n - 1), true, true, 1)  # wrong size
        @test_throws ArgumentError bad(n, ew, true, true, 2)     # kts /= 1
        @test_throws ArgumentError EDMFWork(3)
        # `mynn_column_step_edmf!` checks the same shapes
        col = edmf_colstruct(cas); st = MYNNColumnState(n)
        @test_throws ArgumentError mynn_column_step_edmf!(work, EDMFWork(n - 1), c,
                                                          col, st, MYNNOptions())
    end

    @testset "condensation_edmf" begin
        # No standalone reference block (the driver never calls it directly, and
        # DMP_mf's only visible trace of it — edmf_qc — is identically zero in the
        # reference). What is pinned here is the contract the plume integration and
        # the untested Chaboureau-Bechtold block rely on.
        exn(p) = (p/MYNN_P1000MB)^c.rcp

        # A saturated parcel above 100 m condenses.
        thv, qc = condensation_edmf(0.016, 293.0, 90000.0, 800.0, 0.0, c)
        @test qc > 0.0
        # thv is the plume form of :6870, NOT th*(1+p608*qt) — exactly, same order.
        @test thv === (293.0 + c.xlvcp*qc)*(1.0 + 0.016*(c.rvovrd - 1.0) -
                                            c.rvovrd*qc)

        # The SAME parcel below 100 m condenses nothing (:6867), and thv then
        # collapses to the unsaturated form.
        thv2, qc2 = condensation_edmf(0.016, 293.0, 90000.0, 99.999, 0.0, c)
        @test qc2 == 0.0
        @test thv2 === 293.0*(1.0 + 0.016*(c.rvovrd - 1.0))

        # A dry parcel condenses nothing at any height.
        thv3, qc3 = condensation_edmf(0.002, 300.0, 95000.0, 2000.0, 0.0, c)
        @test qc3 == 0.0
        @test thv3 === 300.0*(1.0 + 0.002*(c.rvovrd - 1.0))

        # The returned qc is the SINGLE-SHOT `max(qt - qsat(T), 0)` of :6862-6864 at
        # the temperature the (damped, half-step) iteration converged to — NOT the
        # iterate itself. Those differ: with qt so large that the latent heating
        # overshoots, the loop converges to a positive qc while the final line
        # returns 0. Pinned so nobody "fixes" the loop into returning its own answer.
        _, qc4 = condensation_edmf(0.030, 300.0, 95000.0, 800.0, 0.0, c)
        @test qc4 == 0.0
        @test 0.030 - qsat_blend(exn(95000.0)*300.0, 95000.0, c) > 0.0   # yet unsaturated it is not

        # `qc` is intent(inout): the incoming value is the FIRST GUESS (:6854), and
        # DMP_mf feeds it the plume's condensate from the level below (:6172, :6247).
        # It genuinely matters: the loop stops at |dqc| < 1e-6 (:6851), which for a
        # qc of order 2e-3 is a RELATIVE tolerance of only ~5e-4, so a different
        # starting point converges to a slightly different answer. Pinned, because it
        # means `condensation_edmf` is not a pure function of (qt, thl, p, zagl) and
        # a plume's condensate carries a small path dependence up the column.
        qc_g = condensation_edmf(0.016, 293.0, 90000.0, 800.0, 0.005, c)[2]
        @test qc_g != qc
        @test isapprox(qc_g, qc; rtol = 1e-2)
    end

    @testset "no allocation on a warm call" begin
        cas = "case4_convective"
        cm = mynn_column(cas); n = cm.n
        work = MYNNWork(n); ew = EDMFWork(n)
        zw = mynn_face_heights(cm.dz)
        thl, sqw, thetav = mynn_conserved(cm, c)
        zed = zeros(n)
        vt, vq, sgm, qcb, _, cld = edmf_condensation_step1(cas, work)
        # Every reference lookup is hoisted: a `mynn_block` call inside the closure
        # would allocate in the DICT and mask what is being measured.
        qke  = copy(mynn_block(ref, cas, 2.5, "B", 1, "qke_pre_predict"))
        flt  = mynn_scalar(ref, cas, 2.5, "B", 1, "flt")
        fltv = mynn_scalar(ref, cas, 2.5, "B", 1, "fltv")
        flq  = mynn_scalar(ref, cas, 2.5, "B", 1, "flq")
        flqv = cm.qfx/cm.rho[1]
        pblh = mynn_scalar(ref, cas, 2.5, "B", 1, "pblh")
        kpbl = Int(mynn_scalar(ref, cas, 2.5, "B", 1, "kpbl"))
        psh  = mynn_scalar(ref, cas, 2.5, "B", 1, "psig_shcu")
        ths  = cm.ts/cm.exner[1]

        f_dmp() = dmp_mf!(1, n, cm.delt, zw, cm.dz, cm.p, cm.rho, 1, 0, 1,
            cm.u, cm.v, cm.w, cm.th, thl, thetav, cm.T, sqw, cm.sqv, cm.sqc, qke,
            cm.exner, vt, vq, sgm, cm.ust, flt, fltv, flq, flqv, pblh, kpbl,
            cm.dx, cm.xland, ths,
            ew.edmf_a, ew.edmf_w, ew.edmf_qt, ew.edmf_thl, ew.edmf_ent, ew.edmf_qc,
            ew.s_aw, ew.s_awthl, ew.s_awqt, ew.s_awqv, ew.s_awqc, ew.s_awu, ew.s_awv,
            ew.s_awqke, ew.sub_thl, ew.sub_sqv, ew.sub_u, ew.sub_v,
            ew.det_thl, ew.det_sqv, ew.det_sqc, ew.det_u, ew.det_v,
            qcb, cld, zed, zed, true, true, psh, 0, zed, c, ew)
        f_cond() = condensation_edmf(0.030, 300.0, 95000.0, 800.0, 0.0, c)

        col2 = edmf_colstruct(cas); work2 = MYNNWork(n); ew2 = EDMFWork(n)
        st2 = MYNNColumnState(n); opts2 = MYNNOptions(; closure = 2.5, delt = cm.delt)
        mynn_init_column!(work2, c, col2, st2, opts2)
        f_step() = mynn_column_step_edmf!(work2, ew2, c, col2, st2, opts2)

        # A plume-free column takes the other side of every branch; measure both.
        cas5 = "case5_highwind"
        cm5 = mynn_column(cas5); n5 = cm5.n
        col5 = edmf_colstruct(cas5); work5 = MYNNWork(n5); ew5 = EDMFWork(n5)
        st5 = MYNNColumnState(n5)
        opts5 = MYNNOptions(; closure = 2.5, delt = cm5.delt)
        mynn_init_column!(work5, c, col5, st5, opts5)
        f_step5() = mynn_column_step_edmf!(work5, ew5, c, col5, st5, opts5)

        for (nm, f) in (("dmp_mf!", f_dmp), ("condensation_edmf", f_cond),
                        ("mynn_column_step_edmf! (plumes)", f_step),
                        ("mynn_column_step_edmf! (no plumes)", f_step5))
            f()                     # warm up: compile and touch every branch once
            @test (@allocations f()) == 0
        end
    end
end

# ── Observed headroom, for the record ─────────────────────────────────────────
let names = sort!(collect(keys(EDMF_OBSERVED)))
    println("\nMYNN-EDMF part 3 (DMP_mf) — max relative error vs. the Fortran reference:")
    for nm in names
        e = EDMF_OBSERVED[nm]
        println("  ", rpad(nm, 38), e == 0.0 ? "0 (bitwise)" : string(e))
    end
    println()
end
