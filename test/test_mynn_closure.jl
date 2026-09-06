# ── Parity tests: pure-Julia MYNN-EDMF closure vs. the ccpp-physics Fortran ───
#
# Parts 1 AND 2 of the port (src/mynn_constants.jl, src/mynn_closure.jl) checked
# routine by routine against tools/mynn_fortran_driver/ref_driver_output_r8.txt — the
# dump of a VERBATIM ccpp-physics `module_bl_mynn.F90` run on five columns at double
# precision with FMA contraction off. No Fortran runs here; the reference is a
# checked-in text file parsed by test/reference/mynn_fortran_refs.jl.
#
# Part 1: esat/qsat/xl_blend, phim/phih, tridiag2!, mym_level2!, boulac_length0!,
# mym_length!, get_pblh!, scale_aware, mym_initialize!.
# Part 2: mym_turbulence!, mym_predict!, mym_condensation!, mynn_tendencies!,
# moisture_check!, retrieve_exchange_coeffs!, and the 30-step end-to-end
# mynn_init_column! / mynn_column_step! against mode A.
#
# Run standalone:
#
#     julia --project=. test/test_mynn_closure.jl
#
# or `include` from test/runtests.jl once the two src files are part of Scythe — the
# guard below picks up `Scythe.MYNNConstants` when it exists and falls back to
# `include`ing the sources directly otherwise.
#
# WHY THE MODE-B BLOCKS ARE USABLE AS PER-ROUTINE INPUTS. The driver runs each case
# twice: mode A calls `mynn_bl_driver` as a host would, mode B replays the driver's
# per-column call sequence by hand so every intermediate can be printed. The driver
# asserts that mode B reproduces mode A BITWISE at step 30 and prints a `GATE PASS`
# line per case; this file asserts all five gates before using any mode-B block, so a
# future reference regeneration that broke the replay would fail here loudly instead of
# quietly moving the goalposts.
#
# TOLERANCES. Everything here is a straight transcription, so most quantities come out
# bitwise identical; the tolerances (1e-12 for the pure functions, 1e-9 for `el` and
# the `mym_initialize` outputs) are the task's, and the observed errors are ~1e-16 or
# exactly zero. The comparison metric is `|a-b| / max(|a|,|b|)`, i.e. `isapprox`'s
# `rtol`, and NaN is compared with `isequal` semantics (case 1 has genuine NaNs). The
# two largest observed errors both have a named cause and neither is a transcription
# defect: ~1.6e-14 in `mym_condensation!`'s cldfra_bl/vt/vq at exactly ONE level of
# case 3 (the cloud-base level, where `0.5 + 0.36*atan(...)` cancels to 7e-3 and a
# 1-ulp libm `atan` difference is amplified), and ~4e-16 in `mym_turbulence!`'s
# el/dfm/dfh at one level of case 4, inherited from part 1's `mym_length!`.
#
# ONE KNOWN DIVERGENCE, pinned by its own testset at the end: gfortran's `MAX`
# quenches a NaN where Julia's `max` propagates it, which matters only for
# case1_rest (ust = 0). See the note at the top of part 2 of src/mynn_closure.jl.

using Test
using LinearAlgebra
using Springsteel

# ── Wiring: use Scythe's copy if it has one, else include the sources ─────────
const _MYNN_EXPORTS = (:MYNNConstants, :MYNNWork, :MYNN_CONSTANT_ORDER, :MYNN_KARMAN,
                       :MYNN_TICE, :MYNN_TREF, :MYNN_QKEMIN, :MYNN_B1, :MYNN_B2,
                       :MYNN_A1, :MYNN_A2, :MYNN_C1, :MYNN_G1, :MYNN_G2, :MYNN_PR,
                       :esat_blend, :qsat_blend, :xl_blend, :phim, :phih, :tridiag2!,
                       :mym_level2!, :boulac_length0!, :mym_length!, :get_pblh!,
                       :scale_aware, :mym_initialize!,
                       # part 2
                       :MYNN_SQFAC, :MYNN_TLIQ, :MYNN_CKMOD, :MYNN_DHEAT_OPT,
                       :MYNNColumn, :MYNNColumnState, :MYNNOptions,
                       :mynn_wall_heights, :mym_turbulence!, :mym_predict!,
                       :mym_condensation!, :moisture_check!,
                       :retrieve_exchange_coeffs!, :mynn_tendencies!,
                       :mynn_init_column!, :mynn_column_step!)

# Take Scythe's copies only when it has ALL of them: while the port is landing in
# stages, `Scythe` may carry part 1 and not part 2, and a partial pull would fail with
# an opaque `getfield` error instead of falling back to the sources.
if isdefined(Main, :Scythe) &&
   all(nm -> isdefined(Main.Scythe, nm), _MYNN_EXPORTS)
    for _nm in _MYNN_EXPORTS
        @eval const $_nm = getfield(Main.Scythe, $(QuoteNode(_nm)))
    end
else
    include(joinpath(@__DIR__, "..", "src", "mynn_constants.jl"))
    include(joinpath(@__DIR__, "..", "src", "mynn_closure.jl"))
end

if !(@isdefined mynn_reference)
    include(joinpath(@__DIR__, "reference", "mynn_fortran_refs.jl"))
end

# ── Comparison helpers ────────────────────────────────────────────────────────

"""
    mynn_maxrel(got, ref; first_index = 1) -> (maxerr, k)

Largest relative difference `|a-b| / max(|a|,|b|)` between `got` and `ref` over
`first_index:length(ref)`, and the index where it occurs. NaNs must match position for
position (`isequal`); a NaN mismatch returns `(Inf, k)`. A `0` against a nonzero also
returns `Inf`, so a silently-zeroed output cannot pass.
"""
function mynn_maxrel(got::AbstractVector, ref::AbstractVector; first_index::Int = 1)
    length(got) == length(ref) ||
        return (Inf, 0)
    m = 0.0
    km = 0
    for k in first_index:length(ref)
        g = got[k]; rr = ref[k]
        if isnan(g) || isnan(rr)
            isequal(g, rr) || return (Inf, k)
            continue
        end
        d = abs(g - rr)
        if d == 0.0
            continue
        end
        s = max(abs(g), abs(rr))
        e = s == 0.0 ? Inf : d/s
        if e > m
            m = e
            km = k
        end
    end
    return (m, km)
end

mynn_relscalar(g::Real, rr::Real) = mynn_maxrel([Float64(g)], [Float64(rr)])[1]

# Running tally, printed at the end so a reader sees how much headroom there is.
const MYNN_OBSERVED = Dict{String,Float64}()
function mynn_note!(name::AbstractString, err::Real)
    MYNN_OBSERVED[name] = max(get(MYNN_OBSERVED, name, 0.0), Float64(err))
    return err
end

# ── Fortran spot-check values (tools/mynn_fortran_driver/blend_ref_driver.f90) ─
#
# `mynn_bl_driver` only reaches esat_blend/qsat_blend/xl_blend from inside
# mym_condensation (part 2 of the port), and every reference column has rmol <= 0, so
# neither the three blend functions nor the STABLE branch of phim/phih has a block in
# ref_driver_output_r8.txt. `boulac_length0` is likewise never called by the CASE-2
# mixing length. blend_ref_driver.f90 calls all six directly, built with the reference
# build's flags (`-O0 -ffp-contract=off -fdefault-real-8 -fdefault-double-8`); its
# output is transcribed here verbatim. It is NOT run at test time.

const BLEND_T = [
    3.10000000000000000e02, 3.00000000000000000e02, 2.88000000000000000e02,
    2.73160000000000025e02, 2.70000000000000000e02, 2.67160000000000025e02,
    2.65000000000000000e02, 2.60000000000000000e02, 2.50000000000000000e02,
    2.40000000000000000e02, 2.30000000000000000e02, 1.80000000000000000e02,
]
const BLEND_ESAT = [
    6.22072348253841938e03, 3.53348965140787368e03, 1.68909864683137766e03,
    6.11583699000000024e02, 4.84580481363666536e02, 3.91028503256429872e02,
    3.29016070745575405e02, 2.15426840208737843e02, 8.30814050990492206e01,
    2.72309977507881058e01, 8.94001873318245543e00, 5.48400267905435612e-02,
]
const BLEND_XL = [
    2.41573381999999983e06, 2.43887881999999983e06, 2.46665281999999983e06,
    2.50100000000000000e06, 2.53951617078407714e06, 2.57306311580217117e06,
    2.59790025100120623e06, 2.65314802241254551e06, 2.75423463881785283e06,
    2.84277602000000002e06, 2.84512102000000002e06, 2.85684602000000002e06,
]
# 20000 Pa at the warmest temperatures activates the min(ES, P*0.15) cap.
const BLEND_P = [1.00000000000000000e05, 2.00000000000000000e04, 3.00000000000000000e04]
# temperature-major: index (i-1)*3 + j is qsat_blend(BLEND_T[i], BLEND_P[j])
const BLEND_QSAT = [
    4.12595420846357253e-02, 1.09764705882352945e-01, 1.09764705882352945e-01,
    2.27833530541697825e-02, 1.09764705882352945e-01, 8.30419475113238770e-02,
    1.06867025311354735e-02, 5.73767144536176457e-02, 3.71100639016401085e-02,
    3.82745872140607306e-03, 1.96202234815011575e-02, 1.29440476438689878e-02,
    3.02876740977568249e-03, 1.54446620591665324e-02, 1.02119185267852235e-02,
    2.44174521000300408e-03, 1.24034924047847443e-02, 8.21439302787835800e-03,
    2.05323835082800530e-03, 1.04036223062194070e-02, 6.89727647658418003e-03,
    1.34285638662028041e-03, 6.77294627414505752e-03, 4.49891888133437372e-03,
    5.17201421631448190e-04, 2.59474597584660824e-03, 1.72739834736421226e-03,
    1.69422941567309518e-04, 8.48038677014828599e-04, 5.65102296678667471e-04,
    5.56118882336208373e-05, 2.78158919899709604e-04, 1.85411641186168061e-04,
    3.41105153699338574e-07, 1.70552950975010487e-06, 1.13701863392834792e-06,
]
const STAB_ZET = [
    0.00000000000000000e00, 1.00000000000000006e-01, 5.00000000000000000e-01,
    1.00000000000000000e00, 2.00000000000000000e00, 5.00000000000000000e00,
    1.00000000000000000e01, 2.00000000000000000e01, -1.00000000000000002e-02,
    -1.00000000000000006e-01, -5.00000000000000000e-01, -1.00000000000000000e00,
    -2.00000000000000000e01,
]
const STAB_PHIM = [
    1.00000000000000000e00, 1.57139210035662824e00, 3.57006005341670729e00,
    5.36493407969610914e00, 6.62691465678732872e00, 7.04620871528959114e00,
    7.09037938581526994e00, 7.09829576029437703e00, 9.63576645623258954e-01,
    7.88069132976806253e-01, 5.75658293185110748e-01, 4.64688700814237077e-01,
    1.70006896573980049e-01,
]
const STAB_PHIH = [
    1.00000000000000000e00, 1.80896978336951531e00, 3.62893468028997690e00,
    4.57082252885281726e00, 5.31175094552480331e00, 5.88692956738170103e00,
    6.09822047107601684e00, 6.20374264381991836e00, 9.28469198404928919e-01,
    6.18398054556886567e-01, 3.25963860684671869e-01, 2.67116900306724858e-01,
    1.14532194461428816e-01,
]
# 20-level synthetic column: dz = 100 m everywhere, theta neutral-ish to 800 m, a 6 K
# inversion, then 4 K/km; qtke ramping 0.01 -> 40 m^2/s^2 so that BOTH boulac branches
# (parcel stops inside a layer / parcel runs off the end of the column) are taken.
const BOULAC_THETA = [
    3.00000000000000000e02, 3.00050000000000011e02, 3.00100000000000023e02,
    3.00149999999999977e02, 3.00199999999999989e02, 3.00250000000000000e02,
    3.00300000000000011e02, 3.00350000000000023e02, 3.06399999999999977e02,
    3.06449999999999989e02, 3.06500000000000000e02, 3.06550000000000011e02,
    3.06600000000000023e02, 3.07049999999999955e02, 3.07500000000000000e02,
    3.07949999999999989e02, 3.08400000000000034e02, 3.08850000000000023e02,
    3.09299999999999955e02, 3.09750000000000000e02,
]
const BOULAC_QTKE = [
    1.00000000000000002e-02, 2.11526315789473651e00, 4.22052631578947324e00,
    6.32578947368421041e00, 8.43105263157894669e00, 1.05363157894736847e01,
    1.26415789473684210e01, 1.47468421052631573e01, 1.68521052631578954e01,
    1.89573684210526316e01, 2.10626315789473715e01, 2.31678947368421078e01,
    2.52731578947368440e01, 2.73784210526315803e01, 2.94836842105263166e01,
    3.15889473684210529e01, 3.36942105263157856e01, 3.57994736842105254e01,
    3.79047368421052582e01, 4.00099999999999980e01,
]
const BOULAC_LB1 = [
    0.00000000000000000e00, 1.00000000000000000e02, 2.00000000000000000e02,
    3.00000000000000000e02, 3.85757065277989398e02, 2.99953077305726652e02,
    2.12549713857443805e02, 1.24516281433573468e02, 1.35131626976941476e02,
    2.44139175514307908e02, 3.52184774147137830e02, 4.59293448666364441e02,
    5.65489158217790077e02, 6.33876667983715606e02, 5.50000000000000000e02,
    4.50000000000000000e02, 3.50000000000000000e02, 2.50000000000000000e02,
    1.50000000000000000e02, 0.00000000000000000e00,
]
const BOULAC_LB2 = [
    1.87015696772034312e00, 2.25537707128917077e02, 3.29532986353168212e02,
    3.74697394662624674e02, 3.92813984108503632e02, 3.87268044967388619e02,
    3.57113186979235309e02, 2.95231090848341069e02, 3.32337747940488669e02,
    4.32288203875289469e02, 4.99740909933822252e02, 5.46034439722886759e02,
    5.75697643024638182e02, 6.41887711511456814e02, 6.20430451234504858e02,
    5.83411525990485075e02, 5.14034922019125133e02, 4.27393152600857718e02,
    3.30410731032349304e02, 0.00000000000000000e00,
]

# ── moisture_check spot check (tools/mynn_fortran_driver/moisture_ref_driver.f90) ──
#
# moisture_check has no reference block: ref_driver.f90 prints none of its arguments,
# and all five reference columns are non-negative, so in the main reference it only
# ever runs on its NO-OP path. Its CORRECTION path is exactly what a Scythe column
# with negative water will take, so moisture_ref_driver.f90 calls it directly, built
# with the reference build's flags, on two synthetic 6-level columns:
#   A -- scattered negative qc/qi and one negative qv: the condensation correction and
#        a single-layer borrow fire, the column-wide redistribution does not.
#   B -- qv(1) driven far below qvmin: the redistribution loop (:5198-5216) DOES fire.
# Both use dp = 2000 Pa, exner = 0.9, delt = 20 s. Transcribed verbatim; not run at
# test time.
const MC_DELT  = 20.0
const MC_DP    = fill(2000.0, 6)
const MC_EXNER = fill(0.9, 6)
const MC_A_IN_QV = [ 1.0e-3, -5.0e-4, 2.0e-3, 3.0e-3, 1.0e-3, 5.0e-4]
const MC_A_IN_QC = [-1.0e-5,  2.0e-5, 0.0,   -3.0e-6, 0.0,    0.0]
const MC_A_IN_QI = [ 0.0,    -2.0e-6, 0.0,    0.0,    1.0e-6, 0.0]
const MC_A_IN_QS = zeros(6)
const MC_A_IN_TH = fill(300.0, 6)
const MC_B_IN_QV = [-2.0e-3, 1.0e-3,  2.0e-3, 3.0e-3, 4.0e-3, 5.0e-3]
const MC_B_IN_QC = [ 1.0e-5, -4.0e-5, 0.0,    0.0,    0.0,    0.0]
const MC_B_IN_QI = [ 0.0,     0.0,   -1.0e-5, 0.0,    0.0,    0.0]
const MC_B_IN_QS = [ 0.0,     0.0,    0.0,   -2.0e-6, 0.0,    0.0]
const MC_B_IN_TH = [295.0, 296.0, 297.0, 298.0, 299.0, 300.0]
const MC_A_qv = [4.87999999999999936e-04, 9.99999999999999945e-21, 2.00000000000000004e-03, 2.99700000000000010e-03, 1.00000000000000002e-03, 5.00000000000000010e-04]
const MC_A_qc = [0.00000000000000000e00, 2.00000000000000016e-05, 0.00000000000000000e00, 0.00000000000000000e00, 0.00000000000000000e00, 0.00000000000000000e00]
const MC_A_qi = [0.00000000000000000e00, 0.00000000000000000e00, 0.00000000000000000e00, 0.00000000000000000e00, 9.99999999999999955e-07, 0.00000000000000000e00]
const MC_A_qs = [0.00000000000000000e00, 0.00000000000000000e00, 0.00000000000000000e00, 0.00000000000000000e00, 0.00000000000000000e00, 0.00000000000000000e00]
const MC_A_th = [3.00027678176184168e02, 3.00006274900398409e02, 3.00000000000000000e02, 3.00008303452855273e02, 3.00000000000000000e02, 3.00000000000000000e02]
const MC_A_dqv = [-2.56000000000000022e-05, 2.50000000000000046e-05, 0.00000000000000000e00, -1.49999999999999993e-07, 0.00000000000000000e00, 0.00000000000000000e00]
const MC_A_dqc = [5.00000000000000083e-07, 0.00000000000000000e00, 0.00000000000000000e00, 1.49999999999999993e-07, 0.00000000000000000e00, 0.00000000000000000e00]
const MC_A_dqi = [0.00000000000000000e00, 9.99999999999999955e-08, 0.00000000000000000e00, 0.00000000000000000e00, 0.00000000000000000e00, 0.00000000000000000e00]
const MC_A_dqs = [0.00000000000000000e00, 0.00000000000000000e00, 0.00000000000000000e00, 0.00000000000000000e00, 0.00000000000000000e00, 0.00000000000000000e00]
const MC_A_dth = [1.38390880920761421e-03, 3.13745019920318689e-04, 0.00000000000000000e00, 4.15172642762284209e-04, 0.00000000000000000e00, 0.00000000000000000e00]
const MC_B_qv = [9.99999999999999945e-21, 8.31554723039871547e-04, 1.72374364463473385e-03, 2.59687610382659893e-03, 3.46481134599946512e-03, 4.33101418249933097e-03]
const MC_B_qc = [1.00000000000000008e-05, 0.00000000000000000e00, 0.00000000000000000e00, 0.00000000000000000e00, 0.00000000000000000e00, 0.00000000000000000e00]
const MC_B_qi = [0.00000000000000000e00, 0.00000000000000000e00, 0.00000000000000000e00, 0.00000000000000000e00, 0.00000000000000000e00, 0.00000000000000000e00]
const MC_B_qs = [0.00000000000000000e00, 0.00000000000000000e00, 0.00000000000000000e00, 0.00000000000000000e00, 0.00000000000000000e00, 0.00000000000000000e00]
const MC_B_th = [2.95000000000000000e02, 2.96110712704736613e02, 2.97031374501992047e02, 2.98006274900398409e02, 2.99000000000000000e02, 3.00000000000000000e02]
const MC_B_dqv = [1.00000000000000005e-04, -8.42226384800642301e-06, -1.38128177682633139e-05, -2.01561948086700548e-05, -2.67594327000267582e-05, -3.34492908750334486e-05]
const MC_B_dqc = [0.00000000000000000e00, 2.00000000000000033e-06, 0.00000000000000000e00, 0.00000000000000000e00, 0.00000000000000000e00, 0.00000000000000000e00]
const MC_B_dqi = [0.00000000000000000e00, 0.00000000000000000e00, 5.00000000000000083e-07, 0.00000000000000000e00, 0.00000000000000000e00, 0.00000000000000000e00]
const MC_B_dqs = [0.00000000000000000e00, 0.00000000000000000e00, 0.00000000000000000e00, 9.99999999999999955e-08, 0.00000000000000000e00, 0.00000000000000000e00]
const MC_B_dth = [0.00000000000000000e00, 5.53563523683045684e-03, 1.56872509960159366e-03, 3.13745019920318689e-04, 0.00000000000000000e00, 0.00000000000000000e00]

# ── The tests ─────────────────────────────────────────────────────────────────

@testset "MYNN-EDMF closure (parts 1 and 2) vs. ccpp-physics reference" begin
    ref = mynn_reference()
    c   = MYNNConstants()

    @testset "reference harness self-consistency" begin
        # Every per-routine test below feeds on mode-B intermediates, which are only
        # meaningful because mode B reproduced mode A bitwise at step 30.
        for cas in MYNN_CASES
            @test haskey(ref.gate, cas)
            @test ref.gate[cas]
            @test ref.nlev[cas] == mynn_column(cas).n
        end
        @test length(ref.constants) == 24
    end

    @testset "host constants (bitwise)" begin
        # The 14 values columns/constants.txt carries, and the 24 the driver derived
        # and printed, must both come out of MYNNConstants() with `===`.
        host = mynn_host_constants()
        for (i, f) in enumerate((:cp, :cpv, :cliq, :cice, :p608, :ep_2, :grav,
                                 :karman, :t0c, :rcp, :r_d, :r_v, :xlf, :xlv))
            @test getfield(c, f) === host[i]
        end
        for (i, f) in enumerate(MYNN_CONSTANT_ORDER)
            @test getfield(c, f) === ref.constants[i]
        end
        mynn_note!("constants", 0.0)
    end

    @testset "closure constants" begin
        # Sanity, not parity: these have no reference block of their own, but a typo in
        # the association of a1/c1/a2/g2 would move every stability function.
        @test MYNN_A1 === 24.0*(1.0 - 3.0*0.235)/6.0
        @test MYNN_C1 === 0.235 - 1.0/(3.0*MYNN_A1*2.88449914061481660)
        @test MYNN_A2 === MYNN_A1*(MYNN_G1 - MYNN_C1)/(MYNN_G1*MYNN_PR)
        # b1**(1/3) frozen in the Fortran source: it is cbrt(24) to ~16 digits, but the
        # port must use the literal, not recompute it.
        @test abs(2.88449914061481660 - 24.0^(1.0/3.0)) < 1e-15
    end

    @testset "esat_blend / qsat_blend / xl_blend" begin
        me = 0.0; mq = 0.0; mx = 0.0
        for (i, t) in enumerate(BLEND_T)
            me = max(me, mynn_relscalar(esat_blend(t, c), BLEND_ESAT[i]))
            mx = max(mx, mynn_relscalar(xl_blend(t, c),  BLEND_XL[i]))
            for (j, P) in enumerate(BLEND_P)
                mq = max(mq, mynn_relscalar(qsat_blend(t, P, c),
                                            BLEND_QSAT[(i-1)*length(BLEND_P) + j]))
            end
        end
        @test mynn_note!("esat_blend", me) <= 1e-12
        @test mynn_note!("qsat_blend", mq) <= 1e-12
        @test mynn_note!("xl_blend",   mx) <= 1e-12

        # Branch coverage, so a future edit that collapsed the blend would be caught.
        @test esat_blend(310.0, c) > esat_blend(273.16, c) > esat_blend(230.0, c)
        @test xl_blend(230.0, c) === c.xls + (c.cpv - c.cice)*(230.0 - c.t0c)
        @test xl_blend(310.0, c) === c.xlv + (c.cpv - c.cliq)*(310.0 - c.t0c)
        # the P*0.15 cap
        @test qsat_blend(310.0, 20000.0, c) === 0.622*3000.0/max(20000.0 - 3000.0, 1e-5)
    end

    @testset "phim / phih" begin
        # (a) the direct spot check, which is the ONLY coverage of the stable (zet > 0)
        #     Cheng-Brutsaert branch — every reference column has rmol <= 0.
        mm = 0.0; mh = 0.0
        for (i, z) in enumerate(STAB_ZET)
            mm = max(mm, mynn_relscalar(phim(z), STAB_PHIM[i]))
            mh = max(mh, mynn_relscalar(phih(z), STAB_PHIH[i]))
        end
        # (b) the reference itself: the driver prints zet, pmz = phim(zet) - zet and
        #     phh = phih(zet) at steps 1 and 30 of every case (ref_driver.f90 mode_b,
        #     mirroring mynn_bl_driver :1083-1097: zet is clipped to [-20, 20], then
        #     bl_mynn_stfunc = 1 takes pmz = phim(zet) - zet and phh = phih(zet)).
        for cas in MYNN_CASES, step in (1, 30)
            zet = mynn_scalar(ref, cas, 2.5, "B", step, "zet")
            pmz = mynn_scalar(ref, cas, 2.5, "B", step, "pmz")
            phh = mynn_scalar(ref, cas, 2.5, "B", step, "phh")
            mm = max(mm, mynn_relscalar(phim(zet) - zet, pmz))
            mh = max(mh, mynn_relscalar(phih(zet), phh))
        end
        @test mynn_note!("phim", mm) <= 1e-12
        @test mynn_note!("phih", mh) <= 1e-12
        @test phim(0.0) === 1.0
        @test phih(0.0) === 1.0
    end

    @testset "scale_aware" begin
        m = 0.0
        for cas in MYNN_CASES
            colm = mynn_column(cas)
            for (step, pblh_name) in ((0, "init_pblh"), (1, "pblh"), (30, "pblh"))
                pblh = mynn_scalar(ref, cas, 2.5, "B", step, pblh_name)
                psig_bl, psig_shcu = scale_aware(colm.dx, pblh)
                m = max(m, mynn_relscalar(psig_bl,
                            mynn_scalar(ref, cas, 2.5, "B", step, "psig_bl")))
                m = max(m, mynn_relscalar(psig_shcu,
                            mynn_scalar(ref, cas, 2.5, "B", step, "psig_shcu")))
            end
        end
        @test mynn_note!("scale_aware", m) <= 1e-12
        # Both factors are clipped to [0, 1] and both are monotone in dx/PBLH: a
        # mesoscale grid gets the full parameterized mixing (Psig -> 1) and an
        # LES-scale grid gets it tapered away (the resolved motions do the work).
        pb_coarse, ps_coarse = scale_aware(1.0e6, 500.0)
        pb_les,    ps_les    = scale_aware(100.0, 3000.0)
        for p in (pb_coarse, ps_coarse, pb_les, ps_les)
            @test 0.0 <= p <= 1.0
        end
        @test pb_coarse > 0.999 && ps_coarse > 0.999
        @test pb_les < 0.5 && ps_les < 0.7
    end

    @testset "tridiag2!" begin
        # No reference block: tridiag2 is called from mynn_tendencies and mym_predict,
        # which are later stages. Checked against a dense solve of the same system and
        # against the residual, which is what "solves the system" means.
        n = 64
        a = zeros(n); b = zeros(n); cc = zeros(n); d = zeros(n); x = zeros(n)
        for k in 1:n
            a[k]  = -1.0 - 0.01k
            b[k]  =  5.0 + 0.03k
            cc[k] = -1.5 + 0.005k
            d[k]  = sinpi(k/7) + 0.25k
        end
        a[1] = 0.0; cc[n] = 0.0     # the Fortran never reads these two
        work = MYNNWork(n)
        tridiag2!(n, a, b, cc, d, x, work)
        xd = Array(Tridiagonal(a[2:n], b, cc[1:n-1]) \ d)
        @test mynn_note!("tridiag2!", mynn_maxrel(x, xd)[1]) <= 1e-12
        resid = 0.0
        for k in 1:n
            r = b[k]*x[k] + (k > 1 ? a[k]*x[k-1] : 0.0) + (k < n ? cc[k]*x[k+1] : 0.0) - d[k]
            resid = max(resid, abs(r)/max(abs(d[k]), 1e-300))
        end
        @test resid <= 1e-12
    end

    @testset "mym_level2!" begin
        # Inputs are exactly what the driver's standalone mym_level2 call gets
        # (ref_driver.f90 mode_b): the frozen column's thl/sqw/thetav, ql = sqc, and
        # the post-mym_condensation vt/vq of that step. sm2/sh2 are zeroed before the
        # call in the driver, so index 1 (never written; interface arrays start at 2)
        # is compared too; dtl..gh are Fortran automatic arrays whose index 1 is never
        # written, so those are compared from index 2.
        for cas in MYNN_CASES, step in (1, 30)
            colm = mynn_column(cas)
            n = colm.n
            thl, sqw, thetav = mynn_conserved(colm, c)
            vt = mynn_block(ref, cas, 2.5, "B", step, "vt")
            vq = mynn_block(ref, cas, 2.5, "B", step, "vq")
            dtl = zeros(n); dqw = zeros(n); dtv = zeros(n)
            gm  = zeros(n); gh  = zeros(n); sm  = zeros(n); sh = zeros(n)
            mym_level2!(1, n, colm.dz, colm.u, colm.v, thl, thetav, sqw, colm.sqc,
                        vt, vq, dtl, dqw, dtv, gm, gh, sm, sh, c)
            for (name, got, from) in (("dtl", dtl, 2), ("dqw", dqw, 2), ("dtv", dtv, 2),
                                      ("gm", gm, 2), ("gh", gh, 2),
                                      ("sm2", sm, 1), ("sh2", sh, 1))
                e, _ = mynn_maxrel(got, mynn_block(ref, cas, 2.5, "B", step, name);
                                   first_index = from)
                @test mynn_note!("mym_level2! ($name)", e) <= 1e-12
            end
        end
    end

    @testset "get_pblh!" begin
        mz = 0.0
        for cas in MYNN_CASES
            colm = mynn_column(cas)
            n = colm.n
            zw = mynn_face_heights(colm.dz)
            _, _, thetav = mynn_conserved(colm, c)

            # (a) the init call: qke is the driver's ust taper, rebuilt here exactly
            #     (mynn_bl_driver's INITIALIZE_QKE pre-pass, 5*ust*taper).
            zi, kzi = get_pblh!(1, n, thetav, mynn_taper_qke(colm.ust, zw, n), zw,
                                colm.dz, colm.xland)
            mz = max(mz, mynn_relscalar(zi, mynn_scalar(ref, cas, 2.5, "B", 0, "init_pblh")))
            @test kzi == Int(mynn_scalar(ref, cas, 2.5, "B", 0, "init_kpbl"))

            # (b) step 1: the driver calls GET_PBLH first thing in the step, so its qke
            #     is exactly mym_initialize's output, the init_qke block.
            zi, kzi = get_pblh!(1, n, thetav,
                                mynn_block(ref, cas, 2.5, "B", 0, "init_qke"), zw,
                                colm.dz, colm.xland)
            mz = max(mz, mynn_relscalar(zi, mynn_scalar(ref, cas, 2.5, "B", 1, "pblh")))
            @test kzi == Int(mynn_scalar(ref, cas, 2.5, "B", 1, "kpbl"))
        end
        @test mynn_note!("get_pblh!", mz) <= 1e-12
    end

    @testset "mym_length! (CASE 2)" begin
        # The reference `el` block is printed after mym_turbulence, and mym_turbulence
        # never touches `el` after its own mym_length call (:2703-2722 sets it, and no
        # line below that assigns el(k)) — so that block IS mym_length's output. Its inputs
        # are all printed at the same step: rmol/flt/fltv/flq are the surface scalars
        # the driver forms just before, vt/vq/cldfra_bl come from mym_condensation,
        # edmf_w/edmf_a from DMP_mf, qke_pre_predict is `qke` as handed to
        # mym_turbulence, and `dtv` is mym_turbulence's internal mym_level2 output —
        # identical to the standalone `dtv` block, which the mym_level2! testset above
        # has just proved bitwise.
        for cas in MYNN_CASES, step in (1, 30)
            colm = mynn_column(cas)
            n = colm.n
            zw = mynn_face_heights(colm.dz)
            work = MYNNWork(n)
            el = zeros(n); qkw = zeros(n)
            mym_length!(1, n, colm.xland, colm.dz, colm.dx, zw,
                        mynn_scalar(ref, cas, 2.5, "B", step, "rmol"),
                        mynn_scalar(ref, cas, 2.5, "B", step, "flt"),
                        mynn_scalar(ref, cas, 2.5, "B", step, "fltv"),
                        mynn_scalar(ref, cas, 2.5, "B", step, "flq"),
                        mynn_block(ref, cas, 2.5, "B", step, "vt"),
                        mynn_block(ref, cas, 2.5, "B", step, "vq"),
                        colm.u, colm.v,
                        mynn_block(ref, cas, 2.5, "B", step, "qke_pre_predict"),
                        mynn_block(ref, cas, 2.5, "B", step, "dtv"),
                        el,
                        mynn_scalar(ref, cas, 2.5, "B", step, "pblh"),
                        colm.th, qkw,
                        mynn_scalar(ref, cas, 2.5, "B", step, "psig_bl"),
                        mynn_block(ref, cas, 2.5, "B", step, "cldfra_bl"),
                        2,
                        mynn_block(ref, cas, 2.5, "B", step, "edmf_w"),
                        mynn_block(ref, cas, 2.5, "B", step, "edmf_a"),
                        c, work)
            e, _ = mynn_maxrel(el, mynn_block(ref, cas, 2.5, "B", step, "el"))
            @test mynn_note!("mym_length! (el)", e) <= 1e-9
            @test el[1] === 0.0                    # :2155, el(kts) = 0
            @test all(isfinite, qkw) && all(>=(0.0), qkw)
        end

        # CASE 0 and CASE 1 are out of scope and must say so rather than silently
        # producing CASE-2 numbers.
        let colm = mynn_column("case1_rest"), n = colm.n
            zw = mynn_face_heights(colm.dz)
            work = MYNNWork(n); el = zeros(n); qkw = zeros(n); zed = zeros(n)
            for badcase in (0, 1, 3)
                @test_throws ArgumentError mym_length!(1, n, colm.xland, colm.dz,
                    colm.dx, zw, 0.0, 0.0, 0.0, 0.0, zed, zed, colm.u, colm.v, zed,
                    zed, el, 500.0, colm.th, qkw, 1.0, zed, badcase, zed, zed, c, work)
            end
        end
    end

    @testset "boulac_length0!" begin
        n = length(BOULAC_THETA)
        dz = fill(100.0, n)
        zw = mynn_face_heights(dz)
        lb1 = zeros(n); lb2 = zeros(n)
        for k in 1:n
            lb1[k], lb2[k] = boulac_length0!(k, 1, n, zw, dz, BOULAC_QTKE, BOULAC_THETA, c)
        end
        @test mynn_note!("boulac_length0! (lb1)",
                         mynn_maxrel(lb1, BOULAC_LB1)[1]) <= 1e-12
        @test mynn_note!("boulac_length0! (lb2)",
                         mynn_maxrel(lb2, BOULAC_LB2)[1]) <= 1e-12
        @test lb1[n] === 0.0 && lb2[n] === 0.0   # the k == kte short-circuit
    end

    @testset "mym_initialize!" begin
        # The driver's init block (ref_driver.f90 mode_b, mirroring mynn_bl_driver
        # :814-825). NOTE the Fortran passes `sqv`, NOT `sqw`, as the total-water
        # argument `qw` at initialization (:817) — a genuine MYNN quirk: cloud and ice
        # are excluded from qw for the cold start only. Passing sqw here reproduces
        # el and qke but moves sh/sm/tsq/qsq/cov by 2-7 %.
        for cas in MYNN_CASES
            colm = mynn_column(cas)
            n = colm.n
            zw = mynn_face_heights(colm.dz)
            thl, _, thetav = mynn_conserved(colm, c)
            work = MYNNWork(n)
            el  = zeros(n); sh  = zeros(n); sm  = zeros(n)
            tsq = zeros(n); qsq = zeros(n); cov = zeros(n)
            zeroc = zeros(n)
            qke = mynn_taper_qke(colm.ust, zw, n)   # overwritten inside; see the docstring
            mym_initialize!(1, n, colm.xland, colm.dz, colm.dx, zw, colm.u, colm.v,
                            thl, colm.sqv,
                            mynn_scalar(ref, cas, 2.5, "B", 0, "init_pblh"),
                            colm.th, thetav, sh, sm, colm.ust, colm.rmol,
                            el, qke, tsq, qsq, cov,
                            mynn_scalar(ref, cas, 2.5, "B", 0, "psig_bl"),
                            zeroc, 2, zeroc, zeroc, true, c, work)
            for (name, got) in (("init_el", el), ("init_qke", qke), ("init_tsq", tsq),
                                ("init_qsq", qsq), ("init_cov", cov),
                                ("init_sh", sh), ("init_sm", sm))
                e, kbad = mynn_maxrel(got, mynn_block(ref, cas, 2.5, "B", 0, name))
                @test mynn_note!("mym_initialize! ($name)", e) <= 1e-9
            end
        end

        # Case 1 is the resting column (ust = 0): phm*(flt/ust)^2 is 0/0, so tsq, qsq
        # and cov are genuinely NaN at k = 1 in the reference (harness README item 4).
        # The port must PRODUCE that NaN, not guard it away — mym_predict overwrites all
        # three at step 1, so it never reaches the state.
        @test mynn_column("case1_rest").ust == 0.0
        for name in ("init_tsq", "init_qsq", "init_cov")
            @test isnan(mynn_block(ref, "case1_rest", 2.5, "B", 0, name)[1])
        end
        let colm = mynn_column("case1_rest"), n = colm.n
            zw = mynn_face_heights(colm.dz)
            thl, _, thetav = mynn_conserved(colm, c)
            work = MYNNWork(n)
            el = zeros(n); sh = zeros(n); sm = zeros(n)
            tsq = zeros(n); qsq = zeros(n); cov = zeros(n); zeroc = zeros(n)
            qke = mynn_taper_qke(colm.ust, zw, n)
            mym_initialize!(1, n, colm.xland, colm.dz, colm.dx, zw, colm.u, colm.v,
                            thl, colm.sqv,
                            mynn_scalar(ref, "case1_rest", 2.5, "B", 0, "init_pblh"),
                            colm.th, thetav, sh, sm, colm.ust, colm.rmol,
                            el, qke, tsq, qsq, cov,
                            mynn_scalar(ref, "case1_rest", 2.5, "B", 0, "psig_bl"),
                            zeroc, 2, zeroc, zeroc, true, c, work)
            @test isnan(tsq[1]) && isnan(qsq[1]) && isnan(cov[1])
            @test all(isfinite, @view tsq[2:n])
            @test all(isfinite, el) && all(isfinite, qke)
        end
    end

    # ── Part 2 ───────────────────────────────────────────────────────────────
    #
    # WHICH REFERENCE BLOCK IS WHOSE. In mode B the driver prints its block at each
    # of steps 1 and 30 AFTER mym_turbulence and BEFORE mym_predict (ref_driver.f90
    # mode_b :396-461), then a second, smaller block after mynn_tendencies (:483-497).
    # So `el, sh, sm, dfm, dfh, dfq, tcd, qcd, pdk..pdc, vt, vq, sgm, cldfra_bl,
    # qc_bl, qi_bl, s_aw*, edmf_*, qke_pre_predict` are mym_turbulence's (and
    # mym_condensation's and DMP_mf's) OUTPUTS and mym_predict's INPUTS, while
    # `qke, tsq, qsq, cov, diss_heat, du..dqi, k_m, k_h` are the post-predict,
    # post-tendency ones.
    #
    # WHAT IS AND IS NOT COMPUTABLE AT STEP 30. The state that a routine reads but
    # the driver does not print at the start of a step is the previous step's, and
    # mode A step 29 is not dumped either. That bites exactly once:
    #   * mym_condensation reads `qsq` BEFORE mym_predict overwrites it, so it is
    #     testable at step 1 (where that is the printed `init_qsq`) but NOT at 30.
    #   * mym_turbulence at closure <= 2.6 never reads tsq/qsq/cov at all (they are
    #     Level-3-only, :2899-2904), which the "unread" check below proves, so BOTH
    #     steps are fully testable.
    #   * mym_predict at closure 2.5 never reads its own tsq/qsq/cov either (all
    #     three are diagnosed from pdt/pdq/pdc, :3429-3440 and :3546-3559), so both
    #     steps are fully testable. Only the closure-2.6 prognostic qsq reads it,
    #     which is why that arm is checked at step 1 only.

    # The reference column `cas` as a `MYNNColumn`.
    function mynn_column_struct(cas)
        cm = mynn_column(cas)
        return MYNNColumn(; ps = cm.ps, ts = cm.ts, ust = cm.ust, hfx = cm.hfx,
                          qfx = cm.qfx, wspd = cm.wspd, xland = cm.xland, dx = cm.dx,
                          rmol0 = cm.rmol, dz = cm.dz, u = cm.u, v = cm.v, w = cm.w,
                          T = cm.T, th = cm.th, exner = cm.exner, p = cm.p,
                          rho = cm.rho, sqv = cm.sqv, sqc = cm.sqc, sqi = cm.sqi,
                          z = cm.z, qsfc = cm.qsfc, znt = cm.znt)
    end

    # `ktop_plume` at steps 1 and 30 of every case: DMP_mf only rewrites vt, vq,
    # cldfra_bl and qc_bl inside `IF (nup2 > 0 ...)` (:6580-6768), so a case whose
    # plume never fires leaves mym_condensation's output untouched in the dump.
    ktop_at(cas, step) = Int(mynn_scalar(ref, cas, 2.5, "B", step, "ktop_plume"))
    plume_free = [cas for cas in MYNN_CASES
                  if ktop_at(cas, 1) == 0 && ktop_at(cas, 30) == 0]

    @testset "which cases have active plumes" begin
        # case4_convective is the one built to fire DMP_mf; the other four never do,
        # which is what makes their mym_condensation output and their 30-step
        # evolution reproducible without the (unported) plumes.
        @test ktop_at("case4_convective", 1)  == 5
        @test ktop_at("case4_convective", 30) == 5
        @test plume_free == ["case1_rest", "case2_o01_sea", "case3_tc_rmw",
                             "case5_highwind"]
        for cas in plume_free, step in (1, 30)
            @test all(iszero, mynn_block(ref, cas, 2.5, "B", step, "edmf_a"))
            @test all(iszero, mynn_block(ref, cas, 2.5, "B", step, "s_aw"))
        end
    end

    @testset "mym_turbulence! (closure 2.5)" begin
        # Inputs: the frozen column, the post-DMP_mf vt/vq/cldfra_bl/edmf_* of the
        # step, the surface scalars, and `qke_pre_predict` (qke as handed to
        # mym_turbulence). Outputs: sh, sm, el, dfm, dfh, dfq, tcd, qcd from index 1
        # and pdk..pdc from index 2 — mym_turbulence writes pdk(kts+1:kte) only, so
        # the printed pdk(1) is whatever the PREVIOUS step's mym_predict left there.
        for cas in MYNN_CASES, step in (1, 30)
            colm = mynn_column(cas)
            n = colm.n
            zw = mynn_face_heights(colm.dz)
            thl, sqw, thetav = mynn_conserved(colm, c)
            work = MYNNWork(n)
            sh = zeros(n); sm = zeros(n); el = zeros(n)
            dfm = zeros(n); dfh = zeros(n); dfq = zeros(n)
            tcd = zeros(n); qcd = zeros(n)
            pdk = zeros(n); pdt = zeros(n); pdq = zeros(n); pdc = zeros(n)
            zed = zeros(n)
            # tsq/qsq/cov are Level-3-only; at step 1 they are genuinely the init
            # state, at step 30 the printed post-predict ones stand in (see the
            # "unread" check below, which proves the choice cannot matter).
            tsq = step == 1 ? mynn_block(ref, cas, 2.5, "B", 0, "init_tsq") :
                              mynn_block(ref, cas, 2.5, "B", step, "tsq")
            qsq = step == 1 ? mynn_block(ref, cas, 2.5, "B", 0, "init_qsq") :
                              mynn_block(ref, cas, 2.5, "B", step, "qsq")
            cov = step == 1 ? mynn_block(ref, cas, 2.5, "B", 0, "init_cov") :
                              mynn_block(ref, cas, 2.5, "B", step, "cov")
            mym_turbulence!(1, n, colm.xland, 2.5, colm.dz, colm.dx, zw,
                colm.u, colm.v, thl, thetav, colm.sqc, sqw,
                mynn_block(ref, cas, 2.5, "B", step, "qke_pre_predict"),
                tsq, qsq, cov,
                mynn_block(ref, cas, 2.5, "B", step, "vt"),
                mynn_block(ref, cas, 2.5, "B", step, "vq"),
                mynn_scalar(ref, cas, 2.5, "B", step, "rmol"),
                mynn_scalar(ref, cas, 2.5, "B", step, "flt"),
                mynn_scalar(ref, cas, 2.5, "B", step, "fltv"),
                mynn_scalar(ref, cas, 2.5, "B", step, "flq"),
                mynn_scalar(ref, cas, 2.5, "B", step, "pblh"), colm.th,
                sh, sm, el, dfm, dfh, dfq, tcd, qcd, pdk, pdt, pdq, pdc,
                zed, zed, zed, zed, 0,
                mynn_scalar(ref, cas, 2.5, "B", step, "psig_bl"),
                mynn_scalar(ref, cas, 2.5, "B", step, "psig_shcu"),
                mynn_block(ref, cas, 2.5, "B", step, "cldfra_bl"), 2,
                mynn_block(ref, cas, 2.5, "B", step, "edmf_w"),
                mynn_block(ref, cas, 2.5, "B", step, "edmf_a"),
                zed, 0, zed, c, work)
            for (name, got, from) in (("sh", sh, 1), ("sm", sm, 1), ("el", el, 1),
                                      ("dfm", dfm, 1), ("dfh", dfh, 1), ("dfq", dfq, 1),
                                      ("tcd", tcd, 1), ("qcd", qcd, 1),
                                      ("pdk", pdk, 2), ("pdt", pdt, 2),
                                      ("pdq", pdq, 2), ("pdc", pdc, 2))
                e, _ = mynn_maxrel(got, mynn_block(ref, cas, 2.5, "B", step, name);
                                   first_index = from)
                @test mynn_note!("mym_turbulence! ($name)", e) <= 1e-12
            end
            # dfq is a straight copy of dfm above kts (:3082), and the kts element of
            # all five surface-zeroed outputs is exactly 0 (:3120-3124).
            @test dfm[1] === 0.0 && dfh[1] === 0.0 && dfq[1] === 0.0
            @test all(k -> dfq[k] === dfm[k], 2:n)
        end

        # tsq/qsq/cov are read ONLY by the Level-3 block, so at closure 2.5 the
        # answer cannot depend on them. Prove it rather than assert it: feed absurd
        # values and demand bitwise-identical output.
        let cas = "case3_tc_rmw", colm = mynn_column(cas), n = colm.n
            zw = mynn_face_heights(colm.dz)
            thl, sqw, thetav = mynn_conserved(colm, c)
            work = MYNNWork(n)
            zed = zeros(n)
            junk = fill(9.0e9, n)
            outs = map(1:2) do trial
                sh = zeros(n); sm = zeros(n); el = zeros(n)
                dfm = zeros(n); dfh = zeros(n); dfq = zeros(n)
                tcd = zeros(n); qcd = zeros(n)
                pdk = zeros(n); pdt = zeros(n); pdq = zeros(n); pdc = zeros(n)
                v = trial == 1 ? mynn_block(ref, cas, 2.5, "B", 0, "init_tsq") : junk
                mym_turbulence!(1, n, colm.xland, 2.5, colm.dz, colm.dx, zw,
                    colm.u, colm.v, thl, thetav, colm.sqc, sqw,
                    mynn_block(ref, cas, 2.5, "B", 1, "qke_pre_predict"), v, v, v,
                    mynn_block(ref, cas, 2.5, "B", 1, "vt"),
                    mynn_block(ref, cas, 2.5, "B", 1, "vq"),
                    mynn_scalar(ref, cas, 2.5, "B", 1, "rmol"),
                    mynn_scalar(ref, cas, 2.5, "B", 1, "flt"),
                    mynn_scalar(ref, cas, 2.5, "B", 1, "fltv"),
                    mynn_scalar(ref, cas, 2.5, "B", 1, "flq"),
                    mynn_scalar(ref, cas, 2.5, "B", 1, "pblh"), colm.th,
                    sh, sm, el, dfm, dfh, dfq, tcd, qcd, pdk, pdt, pdq, pdc,
                    zed, zed, zed, zed, 0,
                    mynn_scalar(ref, cas, 2.5, "B", 1, "psig_bl"),
                    mynn_scalar(ref, cas, 2.5, "B", 1, "psig_shcu"),
                    mynn_block(ref, cas, 2.5, "B", 1, "cldfra_bl"), 2,
                    mynn_block(ref, cas, 2.5, "B", 1, "edmf_w"),
                    mynn_block(ref, cas, 2.5, "B", 1, "edmf_a"),
                    zed, 0, zed, c, work)
                (copy(sh), copy(sm), copy(el), copy(dfm), copy(dfh), copy(pdk))
            end
            @test all(map(isequal, outs[1], outs[2]))
        end

        # Level 3 must refuse rather than quietly run the 2.5 path.
        let colm = mynn_column("case1_rest"), n = colm.n
            zw = mynn_face_heights(colm.dz)
            thl, sqw, thetav = mynn_conserved(colm, c)
            work = MYNNWork(n); zed = zeros(n)
            sh = zeros(n); sm = zeros(n); el = zeros(n)
            for bad in (3.0, 3.5)
                @test_throws ArgumentError mym_turbulence!(1, n, colm.xland, bad,
                    colm.dz, colm.dx, zw, colm.u, colm.v, thl, thetav, colm.sqc, sqw,
                    zed, zed, zed, zed, zed, zed, 0.0, 0.0, 0.0, 0.0, 500.0, colm.th,
                    sh, sm, el, zed, zed, zed, zed, zed, zed, zed, zed, zed,
                    zed, zed, zed, zed, 0, 1.0, 1.0, zed, 2, zed, zed, zed, 0, zed,
                    c, work)
            end
        end
    end

    @testset "mym_predict!" begin
        # closure 2.5: qke from the tridiagonal solve, tsq/qsq/cov from the Level-2
        # diagnostic. None of the three variances reads its own input at 2.5, so both
        # steps are testable; the pre-predict tsq/qsq/cov of step 1 (the init blocks)
        # are passed anyway so the call is the driver's.
        for cas in MYNN_CASES, step in (1, 30)
            colm = mynn_column(cas)
            n = colm.n
            work = MYNNWork(n)
            qke = copy(mynn_block(ref, cas, 2.5, "B", step, "qke_pre_predict"))
            tsq = step == 1 ? copy(mynn_block(ref, cas, 2.5, "B", 0, "init_tsq")) : zeros(n)
            qsq = step == 1 ? copy(mynn_block(ref, cas, 2.5, "B", 0, "init_qsq")) : zeros(n)
            cov = step == 1 ? copy(mynn_block(ref, cas, 2.5, "B", 0, "init_cov")) : zeros(n)
            pdk = copy(mynn_block(ref, cas, 2.5, "B", step, "pdk"))
            pdt = copy(mynn_block(ref, cas, 2.5, "B", step, "pdt"))
            pdq = copy(mynn_block(ref, cas, 2.5, "B", step, "pdq"))
            pdc = copy(mynn_block(ref, cas, 2.5, "B", step, "pdc"))
            pdk_in2 = pdk[2]
            zed = zeros(n)
            mym_predict!(1, n, 2.5, colm.delt, colm.dz, colm.ust,
                mynn_scalar(ref, cas, 2.5, "B", step, "flt"),
                mynn_scalar(ref, cas, 2.5, "B", step, "flq"),
                mynn_scalar(ref, cas, 2.5, "B", step, "pmz"),
                mynn_scalar(ref, cas, 2.5, "B", step, "phh"),
                mynn_block(ref, cas, 2.5, "B", step, "el"),
                mynn_block(ref, cas, 2.5, "B", step, "dfq"),
                colm.rho, pdk, pdt, pdq, pdc, qke, tsq, qsq, cov,
                mynn_block(ref, cas, 2.5, "B", step, "s_aw"),
                mynn_block(ref, cas, 2.5, "B", step, "s_awqke"),
                0, zed, zed, 0, c, work)
            for (name, got) in (("qke", qke), ("tsq", tsq), ("qsq", qsq), ("cov", cov))
                e, _ = mynn_maxrel(got, mynn_block(ref, cas, 2.5, "B", step, name))
                @test mynn_note!("mym_predict! ($name)", e) <= 1e-12
            end
            # pdk(kts) is OVERWRITTEN in place with the surface production (:3293).
            pdk1 = 2.0*colm.ust^3*mynn_scalar(ref, cas, 2.5, "B", step, "pmz") /
                   (MYNN_KARMAN*0.5*colm.dz[1])
            @test pdk[1] === pdk1 - pdk_in2
        end

        # closure 2.6: the prognostic-qsq branch. Mode A at step 1 starts from the
        # same init state as mode B, and mym_turbulence is closure-independent below
        # 3.0, so the mode-B closure-2.5 step-1 blocks are its inputs verbatim.
        # case1_rest is excluded: its init_qsq(1) is NaN (ust = 0) and gfortran's MAX
        # quenches it where Julia's propagates — see the known-divergence testset.
        for cas in MYNN_CASES
            cas == "case1_rest" && continue
            colm = mynn_column(cas)
            n = colm.n
            work = MYNNWork(n)
            qke = copy(mynn_block(ref, cas, 2.5, "B", 1, "qke_pre_predict"))
            tsq = copy(mynn_block(ref, cas, 2.5, "B", 0, "init_tsq"))
            qsq = copy(mynn_block(ref, cas, 2.5, "B", 0, "init_qsq"))
            cov = copy(mynn_block(ref, cas, 2.5, "B", 0, "init_cov"))
            pdk = copy(mynn_block(ref, cas, 2.5, "B", 1, "pdk"))
            pdt = copy(mynn_block(ref, cas, 2.5, "B", 1, "pdt"))
            pdq = copy(mynn_block(ref, cas, 2.5, "B", 1, "pdq"))
            pdc = copy(mynn_block(ref, cas, 2.5, "B", 1, "pdc"))
            zed = zeros(n)
            mym_predict!(1, n, 2.6, colm.delt, colm.dz, colm.ust,
                mynn_scalar(ref, cas, 2.5, "B", 1, "flt"),
                mynn_scalar(ref, cas, 2.5, "B", 1, "flq"),
                mynn_scalar(ref, cas, 2.5, "B", 1, "pmz"),
                mynn_scalar(ref, cas, 2.5, "B", 1, "phh"),
                mynn_block(ref, cas, 2.5, "B", 1, "el"),
                mynn_block(ref, cas, 2.5, "B", 1, "dfq"),
                colm.rho, pdk, pdt, pdq, pdc, qke, tsq, qsq, cov,
                mynn_block(ref, cas, 2.5, "B", 1, "s_aw"),
                mynn_block(ref, cas, 2.5, "B", 1, "s_awqke"),
                0, zed, zed, 0, c, work)
            for (name, got) in (("qsq", qsq), ("qke", qke), ("tsq", tsq), ("cov", cov))
                e, _ = mynn_maxrel(got, mynn_block(ref, cas, 2.6, "A", 1, name))
                @test mynn_note!("mym_predict! (2.6 $name)", e) <= 1e-12
            end
            # only qsq is prognostic at 2.6; qke/tsq/cov must be the 2.5 answers
            @test mynn_block(ref, cas, 2.6, "A", 1, "qke") ==
                  mynn_block(ref, cas, 2.5, "A", 1, "qke")
        end

        let colm = mynn_column("case1_rest"), n = colm.n
            work = MYNNWork(n); zed = zeros(n); one_ = ones(n)
            for bad in (3.0, 3.5)
                @test_throws ArgumentError mym_predict!(1, n, bad, 20.0, colm.dz,
                    colm.ust, 0.0, 0.0, 1.0, 1.0, one_, zed, colm.rho,
                    copy(zed), copy(zed), copy(zed), copy(zed),
                    copy(zed), copy(zed), copy(zed), copy(zed),
                    zeros(n+1), zeros(n+1), 0, zed, zed, 0, c, work)
            end
        end
    end

    @testset "mym_condensation! (bl_mynn_cloudpdf = 2)" begin
        # Only step 1 is testable: CASE 2 reads `qsq` BEFORE mym_predict overwrites
        # it, and at step 1 that is exactly the printed `init_qsq`. At step 30 the
        # pre-predict qsq is step 29's, which no block carries.
        #
        # Only the plume-free cases are compared: where DMP_mf fires it OVERWRITES
        # vt, vq, cldfra_bl and qc_bl at the plume levels (:6731-6766) before the
        # dump, so case4_convective's printed block is not mym_condensation's output.
        for cas in plume_free
            colm = mynn_column(cas)
            n = colm.n
            zw = mynn_face_heights(colm.dz)
            thl, sqw, _ = mynn_conserved(colm, c)
            work = MYNNWork(n)
            zed = zeros(n)
            qc_bl = zeros(n); qi_bl = zeros(n); cld = zeros(n)
            vt = zeros(n); vq = zeros(n); sgm = zeros(n)
            mym_condensation!(1, n, colm.dx, colm.dz, zw, colm.xland,
                thl, sqw, colm.sqv, colm.sqc, colm.sqi, zed, colm.p, colm.exner,
                mynn_block(ref, cas, 2.5, "B", 0, "init_tsq"),
                mynn_block(ref, cas, 2.5, "B", 0, "init_qsq"),
                mynn_block(ref, cas, 2.5, "B", 0, "init_cov"),
                mynn_block(ref, cas, 2.5, "B", 0, "init_sh"),
                mynn_block(ref, cas, 2.5, "B", 0, "init_el"), 2,
                qc_bl, qi_bl, cld,
                mynn_scalar(ref, cas, 2.5, "B", 0, "init_pblh"), colm.hfx,
                vt, vq, colm.th, sgm, colm.rmol, 0, zed, c, work)
            # case1_rest's k = 1 is the NaN divergence (its own testset below).
            from = cas == "case1_rest" ? 2 : 1
            for (name, got) in (("vt", vt), ("vq", vq), ("sgm", sgm),
                                ("qc_bl", qc_bl), ("qi_bl", qi_bl),
                                ("cldfra_bl", cld))
                e, _ = mynn_maxrel(got, mynn_block(ref, cas, 2.5, "B", 1, name);
                                   first_index = from)
                @test mynn_note!("mym_condensation! ($name)", e) <= 1e-12
            end
            # the top level is copied down / zeroed (:4008-4013)
            @test vt[n] === vt[n-1] && vq[n] === vq[n-1]
            @test cld[n] === 0.0 && qc_bl[n] === 0.0 && qi_bl[n] === 0.0
        end

        # Every other cloud PDF must refuse.
        let colm = mynn_column("case2_o01_sea"), n = colm.n
            zw = mynn_face_heights(colm.dz)
            thl, sqw, _ = mynn_conserved(colm, c)
            work = MYNNWork(n); zed = zeros(n)
            for bad in (0, 1, -1, 3)
                @test_throws ArgumentError mym_condensation!(1, n, colm.dx, colm.dz,
                    zw, colm.xland, thl, sqw, colm.sqv, colm.sqc, colm.sqi, zed,
                    colm.p, colm.exner, zed, zed, zed, zed, zed, bad,
                    zeros(n), zeros(n), zeros(n), 500.0, colm.hfx,
                    zeros(n), zeros(n), colm.th, zeros(n), 0.0, 0, zed, c, work)
            end
        end

        # cloudpdf = -2 runs CASE 2 and then blanks the cloud (:4000-4006), leaving
        # vt/vq (which were built from the unblanked cloud) alone.
        let cas = "case2_o01_sea", colm = mynn_column(cas), n = colm.n
            zw = mynn_face_heights(colm.dz)
            thl, sqw, _ = mynn_conserved(colm, c)
            work = MYNNWork(n); zed = zeros(n)
            qc_bl = zeros(n); qi_bl = zeros(n); cld = zeros(n)
            vt = zeros(n); vq = zeros(n); sgm = zeros(n)
            mym_condensation!(1, n, colm.dx, colm.dz, zw, colm.xland,
                thl, sqw, colm.sqv, colm.sqc, colm.sqi, zed, colm.p, colm.exner,
                mynn_block(ref, cas, 2.5, "B", 0, "init_tsq"),
                mynn_block(ref, cas, 2.5, "B", 0, "init_qsq"),
                mynn_block(ref, cas, 2.5, "B", 0, "init_cov"),
                mynn_block(ref, cas, 2.5, "B", 0, "init_sh"),
                mynn_block(ref, cas, 2.5, "B", 0, "init_el"), -2,
                qc_bl, qi_bl, cld,
                mynn_scalar(ref, cas, 2.5, "B", 0, "init_pblh"), colm.hfx,
                vt, vq, colm.th, sgm, colm.rmol, 0, zed, c, work)
            @test all(iszero, cld) && all(iszero, qc_bl) && all(iszero, qi_bl)
            e, _ = mynn_maxrel(vt, mynn_block(ref, cas, 2.5, "B", 1, "vt"))
            @test e <= 1e-12
        end
    end

    @testset "diss_heat, mynn_tendencies! and retrieve_exchange_coeffs!" begin
        # mynn_tendencies takes the PRINTED post-DMP_mf inputs, so it is testable on
        # every case at both steps. `flqv = qfx/rho(1)` and `flqc = 0` are the
        # driver's own surface quantities (ref_driver.f90 mode_b :356-359); with
        # flqc = 0 they satisfy flq = flqv, which is asserted.
        for cas in MYNN_CASES, step in (1, 30)
            colm = mynn_column(cas)
            n = colm.n
            work = MYNNWork(n)
            thl0, sqw0, _ = mynn_conserved(colm, c)

            # (a) the driver's diss_heat block (mynn_bl_driver :1224-1234)
            el  = mynn_block(ref, cas, 2.5, "B", step, "el")
            qke = mynn_block(ref, cas, 2.5, "B", step, "qke")
            dh = zeros(n)
            for k in 1:(n-1)
                dh[k] = min(max(1.0*(qke[k]^1.5)/
                                (MYNN_B1*max(0.5*(el[k] + el[k+1]), 1.0))/c.cp,
                                0.0), 0.002)
                dh[k] = dh[k]*exp(-10000.0/max(colm.p[k], 1.0))
            end
            dh[n] = 0.0
            e, _ = mynn_maxrel(dh, mynn_block(ref, cas, 2.5, "B", step, "diss_heat"))
            @test mynn_note!("diss_heat", e) <= 1e-12

            # (b) the tendency solve
            flqv = colm.qfx/colm.rho[1]
            @test flqv === mynn_scalar(ref, cas, 2.5, "B", step, "flq")   # flqc = 0
            thl = copy(thl0); sqw = copy(sqw0)
            sqv = copy(colm.sqv); sqc = copy(colm.sqc); sqi = copy(colm.sqi)
            qv = [colm.sqv[k]/(1.0 - colm.sqv[k]) for k in 1:n]
            qc = [colm.sqc[k]/(1.0 - colm.sqv[k]) for k in 1:n]
            qi = [colm.sqi[k]/(1.0 - colm.sqv[k]) for k in 1:n]
            zed = zeros(n); zed1 = zeros(n+1)
            du = zeros(n); dv = zeros(n); dth = zeros(n); dqv = zeros(n)
            dqc = zeros(n); dqi = zeros(n); dqs = zeros(n)
            dqnc = zeros(n); dqni = zeros(n)
            dqnwfa = zeros(n); dqnifa = zeros(n); dqnbca = zeros(n); doz = zeros(n)
            mynn_tendencies!(1, n, 1, colm.delt, colm.dz, colm.rho,
                colm.u, colm.v, colm.th, colm.T, qv, qc, qi, zed, zed, zed,
                colm.ps, colm.p, colm.exner, thl, sqv, sqc, sqi, zed, sqw,
                zed, zed, zed, zed, colm.ust,
                mynn_scalar(ref, cas, 2.5, "B", step, "flt"),
                mynn_scalar(ref, cas, 2.5, "B", step, "flq"), flqv, 0.0,
                colm.wspd, 0.0, 0.0,
                mynn_block(ref, cas, 2.5, "B", step, "tsq"),
                mynn_block(ref, cas, 2.5, "B", step, "qsq"),
                mynn_block(ref, cas, 2.5, "B", step, "cov"),
                mynn_block(ref, cas, 2.5, "B", step, "tcd"),
                mynn_block(ref, cas, 2.5, "B", step, "qcd"),
                mynn_block(ref, cas, 2.5, "B", step, "dfm"),
                mynn_block(ref, cas, 2.5, "B", step, "dfh"),
                mynn_block(ref, cas, 2.5, "B", step, "dfq"),
                du, dv, dth, dqv, dqc, dqi, dqs, dqnc, dqni,
                dqnwfa, dqnifa, dqnbca, doz,
                mynn_block(ref, cas, 2.5, "B", step, "diss_heat"),
                mynn_block(ref, cas, 2.5, "B", step, "s_aw"),
                mynn_block(ref, cas, 2.5, "B", step, "s_awthl"),
                mynn_block(ref, cas, 2.5, "B", step, "s_awqt"),
                mynn_block(ref, cas, 2.5, "B", step, "s_awqv"),
                mynn_block(ref, cas, 2.5, "B", step, "s_awqc"),
                mynn_block(ref, cas, 2.5, "B", step, "s_awu"),
                mynn_block(ref, cas, 2.5, "B", step, "s_awv"),
                zed1, zed1, zed1, zed1, zed1,
                zed1, zed1, zed1, zed1, zed1, zed1, zed1,
                mynn_block(ref, cas, 2.5, "B", step, "sub_thl"),
                mynn_block(ref, cas, 2.5, "B", step, "sub_sqv"),
                mynn_block(ref, cas, 2.5, "B", step, "sub_u"),
                mynn_block(ref, cas, 2.5, "B", step, "sub_v"),
                mynn_block(ref, cas, 2.5, "B", step, "det_thl"),
                mynn_block(ref, cas, 2.5, "B", step, "det_sqv"),
                mynn_block(ref, cas, 2.5, "B", step, "det_sqc"),
                mynn_block(ref, cas, 2.5, "B", step, "det_u"),
                mynn_block(ref, cas, 2.5, "B", step, "det_v"),
                true, true, false, false, false, false, false, false,
                mynn_block(ref, cas, 2.5, "B", step, "cldfra_bl"),
                1, 0, 1, 1, 1, c, work)

            # (c) the exchange coefficients
            k_m = zeros(n); k_h = zeros(n)
            retrieve_exchange_coeffs!(1, n,
                mynn_block(ref, cas, 2.5, "B", step, "dfm"),
                mynn_block(ref, cas, 2.5, "B", step, "dfh"), colm.dz, k_m, k_h)

            for (name, got) in (("du", du), ("dv", dv), ("dth", dth), ("dqv", dqv),
                                ("dqc", dqc), ("dqi", dqi),
                                ("k_m", k_m), ("k_h", k_h))
                e, _ = mynn_maxrel(got, mynn_block(ref, cas, 2.5, "B", step, name))
                @test mynn_note!("mynn_tendencies! ($name)", e) <= 1e-12
            end
            @test k_m[1] === 0.0 && k_h[1] === 0.0
            # the unmixed species stay at zero tendency
            @test all(iszero, dqs) && all(iszero, dqnc) && all(iszero, dqni)
            @test all(iszero, dqnwfa) && all(iszero, dqnifa) && all(iszero, dqnbca)
            # thl was written in place (README item 1) — the caller's copy evolved
            @test thl != thl0
        end

        # The branches the reference never takes must refuse.
        let cas = "case2_o01_sea", colm = mynn_column("case2_o01_sea"), n = colm.n
            work = MYNNWork(n)
            thl, sqw, _ = mynn_conserved(colm, c)
            zed = zeros(n); zed1 = zeros(n+1)
            qv = [colm.sqv[k]/(1.0 - colm.sqv[k]) for k in 1:n]
            function bad_call(; mixqt = 0, qni = false)
                mynn_tendencies!(1, n, 1, colm.delt, colm.dz, colm.rho,
                    colm.u, colm.v, colm.th, colm.T, qv, zed, zed, zed, zed, zed,
                    colm.ps, colm.p, colm.exner, copy(thl), copy(colm.sqv),
                    copy(colm.sqc), copy(colm.sqi), copy(zed), copy(sqw),
                    zed, zed, zed, zed, colm.ust, 0.0, 0.0, 0.0, 0.0,
                    colm.wspd, 0.0, 0.0, zed, zed, zed, zed, zed, zed, zed, zed,
                    zeros(n), zeros(n), zeros(n), zeros(n), zeros(n), zeros(n),
                    zeros(n), zeros(n), zeros(n), zeros(n), zeros(n), zeros(n),
                    zeros(n), zed,
                    zed1, zed1, zed1, zed1, zed1, zed1, zed1, zed1, zed1, zed1,
                    zed1, zed1, zed1, zed1, zed1, zed1, zed1, zed1, zed1,
                    zed, zed, zed, zed, zed, zed, zed, zed, zed,
                    true, true, false, qni, false, false, false, false, zed,
                    1, mixqt, 1, 1, 1, c, work)
            end
            @test_throws ArgumentError bad_call(mixqt = 1)
            @test_throws ArgumentError bad_call(qni = true)
        end
    end

    @testset "moisture_check!" begin
        # (a) parity against moisture_ref_driver.f90 on the two synthetic columns —
        #     the only coverage of the CORRECTION path, which the five reference
        #     columns never take (they are non-negative everywhere).
        for (tag, qv0, qc0, qi0, qs0, th0, R) in
                (("A", MC_A_IN_QV, MC_A_IN_QC, MC_A_IN_QI, MC_A_IN_QS, MC_A_IN_TH,
                  (MC_A_qv, MC_A_qc, MC_A_qi, MC_A_qs, MC_A_th,
                   MC_A_dqv, MC_A_dqc, MC_A_dqi, MC_A_dqs, MC_A_dth)),
                 ("B", MC_B_IN_QV, MC_B_IN_QC, MC_B_IN_QI, MC_B_IN_QS, MC_B_IN_TH,
                  (MC_B_qv, MC_B_qc, MC_B_qi, MC_B_qs, MC_B_th,
                   MC_B_dqv, MC_B_dqc, MC_B_dqi, MC_B_dqs, MC_B_dth)))
            nk = 6
            qv = copy(qv0); qc = copy(qc0); qi = copy(qi0); qs = copy(qs0)
            th = copy(th0)
            dqv = zeros(nk); dqc = zeros(nk); dqi = zeros(nk)
            dqs = zeros(nk); dth = zeros(nk)
            moisture_check!(nk, MC_DELT, MC_DP, MC_EXNER, qv, qc, qi, qs, th,
                            dqv, dqc, dqi, dqs, dth, c)
            got = (qv, qc, qi, qs, th, dqv, dqc, dqi, dqs, dth)
            names = ("qv", "qc", "qi", "qs", "th", "dqv", "dqc", "dqi", "dqs", "dth")
            for (nm, g, r) in zip(names, got, R)
                e, _ = mynn_maxrel(g, r)
                @test mynn_note!("moisture_check! ($nm)", e) <= 1e-12
            end
            # every species is non-negative afterwards and vapour is at least qvmin
            @test all(>=(0.0), qc) && all(>=(0.0), qi) && all(>=(0.0), qs)
            @test all(>=(1e-20), qv)
            # state and tendency stay consistent — that is the routine's contract
            for k in 1:nk
                @test isapprox(dqc[k], (qc[k] - qc0[k])/MC_DELT; rtol = 1e-13, atol = 0)
                @test isapprox(dqi[k], (qi[k] - qi0[k])/MC_DELT; rtol = 1e-13, atol = 0)
                @test isapprox(dqs[k], (qs[k] - qs0[k])/MC_DELT; rtol = 1e-13, atol = 0)
                @test isapprox(dqv[k], (qv[k] - qv0[k])/MC_DELT; rtol = 1e-11, atol = 1e-30)
            end
            # column water is conserved to round-off (the borrow is dp-weighted and
            # the condensation is 1:1)
            tot0 = sum((qv0[k] + qc0[k] + qi0[k] + qs0[k])*MC_DP[k] for k in 1:nk)
            tot1 = sum((qv[k]  + qc[k]  + qi[k]  + qs[k] )*MC_DP[k] for k in 1:nk)
            @test isapprox(tot0, tot1; rtol = 1e-12)
        end

        # (b) column B is the one that reaches the column-wide redistribution
        #     (:5198-5216): every layer with qv > 2*qvmin loses the same FRACTION.
        @test MC_B_dqv[1] > 0.0            # the bottom layer was topped up
        @test all(<(0.0), MC_B_dqv[2:6])   # ... at everyone else's expense
        # layers 5 and 6 are otherwise untouched, so their loss is purely the
        # redistribution and must be the same fraction of qv (layer 4 also gave up
        # 2e-6 to its own negative qs, so it is excluded from the comparison).
        let frac = [(MC_B_IN_QV[k] - MC_B_qv[k])/MC_B_IN_QV[k] for k in 5:6]
            @test abs(frac[1] - frac[2]) < 1e-12
            @test 0.0 < frac[1] < 0.5      # the aa < 0.5 guard held
        end
        # column A stops short of the redistribution: its k = 1 vapour deficit is
        # zero, so the only layers with a vapour tendency are the one that condensed
        # (k = 4), the one that borrowed (k = 2) and the one it borrowed from (k = 1).
        @test MC_A_dqv[2] > 0.0 && MC_A_dqv[1] < 0.0
        @test MC_A_dqv[3] == 0.0 && MC_A_dqv[5] == 0.0 && MC_A_dqv[6] == 0.0
    end

    @testset "mynn_column_step! (30 steps) vs. mode A" begin
        # The end-to-end check: mynn_init_column! then 30 x mynn_column_step! with
        # edmf = false, compared with the driver run as a host would call it.
        #
        # RUN ON: the plume-free cases with ust > 0. case4_convective is excluded
        # because DMP_mf fires (ktop_plume = 5) and the plumes are not ported;
        # case1_rest is excluded for the NaN divergence pinned in the next testset.
        # Both exclusions are ASSERTED below rather than assumed, so a reference
        # regeneration that changed either would fail here.
        e2e_cases = [cas for cas in plume_free if mynn_column(cas).ust > 0.0]
        @test e2e_cases == ["case2_o01_sea", "case3_tc_rmw", "case5_highwind"]
        @test mynn_column("case1_rest").ust == 0.0

        for cas in e2e_cases
            colm = mynn_column(cas)
            col  = mynn_column_struct(cas)
            n = col.n
            work = MYNNWork(n)
            st   = MYNNColumnState(n)
            opts = MYNNOptions(; closure = 2.5, delt = colm.delt)

            mynn_init_column!(work, c, col, st, opts)
            for (name, got) in (("init_el", st.el), ("init_qke", st.qke),
                                ("init_tsq", st.tsq), ("init_qsq", st.qsq),
                                ("init_cov", st.cov), ("init_sh", st.sh),
                                ("init_sm", st.sm))
                e, _ = mynn_maxrel(got, mynn_block(ref, cas, 2.5, "B", 0, name))
                @test mynn_note!("mynn_init_column! ($name)", e) <= 1e-12
            end
                @test mynn_relscalar(st.pblh,
                    mynn_scalar(ref, cas, 2.5, "B", 0, "init_pblh")) <= 1e-12
            @test st.kpbl == Int(mynn_scalar(ref, cas, 2.5, "B", 0, "init_kpbl"))

            for step in 1:30
                mynn_column_step!(work, c, col, st, opts; edmf = false)
                (step == 1 || step == 30) || continue
                # mode A's names for the twelve gate-verified quantities
                pairs = (("qke", st.qke), ("el_pbl", st.el), ("sh3d", st.sh),
                         ("sm3d", st.sm), ("tsq", st.tsq), ("qsq", st.qsq),
                         ("cov", st.cov), ("cldfra_bl", st.cldfra_bl),
                         ("exch_h", work.out_kh), ("rublten", work.out_du),
                         ("rthblten", work.out_dth), ("rqvblten", work.out_dqv))
                tol = step == 1 ? 1e-12 : 1e-9
                for (name, got) in pairs
                    e, _ = mynn_maxrel(got, mynn_block(ref, cas, 2.5, "A", step, name))
                    @test mynn_note!("mynn_column_step! ($name)", e) <= tol
                end
                @test mynn_relscalar(st.pblh,
                        mynn_scalar(ref, cas, 2.5, "A", step, "pblh")) <= 1e-12
                @test st.kpbl == Int(mynn_scalar(ref, cas, 2.5, "A", step, "kpbl"))
                @test mynn_relscalar(st.rmol,
                        mynn_scalar(ref, cas, 2.5, "A", step, "rmol")) <= 1e-12
            end
        end

        # DMP_mf is a later stage and must say so, not silently run without plumes.
        let col = mynn_column_struct("case4_convective"), n = col.n
            work = MYNNWork(n); st = MYNNColumnState(n); opts = MYNNOptions()
            mynn_init_column!(work, c, col, st, opts)
            @test_throws ArgumentError mynn_column_step!(work, c, col, st, opts;
                                                         edmf = true)
        end

        # ... and with edmf = false on a case whose plumes DO fire, the answer is
        # wrong, by a lot. Pin that so nobody mistakes case 4 for a passing case.
        let cas = "case4_convective", col = mynn_column_struct(cas), n = col.n
            colm = mynn_column(cas)
            work = MYNNWork(n); st = MYNNColumnState(n)
            opts = MYNNOptions(; closure = 2.5, delt = colm.delt)
            mynn_init_column!(work, c, col, st, opts)
            for _ in 1:30
                mynn_column_step!(work, c, col, st, opts; edmf = false)
            end
            e, _ = mynn_maxrel(work.out_du,
                               mynn_block(ref, cas, 2.5, "A", 30, "rublten"))
            @test e > 1e-3      # O(1) wrong: the missing mass flux, not round-off
        end
    end

    @testset "known divergence: max() with a NaN (case1_rest, ust = 0)" begin
        # `mym_initialize` produces NaN in tsq/qsq/cov at k = 1 when ust = 0 (harness
        # README item 4); part 1 reproduces that on purpose. gfortran's MAX then
        # QUENCHES the NaN at two places downstream and Julia's `max` propagates it.
        # This testset pins exactly where, so the divergence is a documented fact
        # rather than a loosened tolerance. See the note at the top of part 2 of
        # src/mynn_closure.jl.
        cas = "case1_rest"
        colm = mynn_column(cas)
        n = colm.n
        @test colm.ust == 0.0
        @test isnan(mynn_block(ref, cas, 2.5, "B", 0, "init_qsq")[1])

        # (1) mym_condensation, :3849  r3sq = max(qsq(k), 0.0)
        let zw = mynn_face_heights(colm.dz), work = MYNNWork(n)
            thl, sqw, _ = mynn_conserved(colm, c)
            zed = zeros(n)
            qc_bl = zeros(n); qi_bl = zeros(n); cld = zeros(n)
            vt = zeros(n); vq = zeros(n); sgm = zeros(n)
            mym_condensation!(1, n, colm.dx, colm.dz, zw, colm.xland,
                thl, sqw, colm.sqv, colm.sqc, colm.sqi, zed, colm.p, colm.exner,
                mynn_block(ref, cas, 2.5, "B", 0, "init_tsq"),
                mynn_block(ref, cas, 2.5, "B", 0, "init_qsq"),
                mynn_block(ref, cas, 2.5, "B", 0, "init_cov"),
                mynn_block(ref, cas, 2.5, "B", 0, "init_sh"),
                mynn_block(ref, cas, 2.5, "B", 0, "init_el"), 2,
                qc_bl, qi_bl, cld,
                mynn_scalar(ref, cas, 2.5, "B", 0, "init_pblh"), colm.hfx,
                vt, vq, colm.th, sgm, colm.rmol, 0, zed, c, work)
            # exactly one level differs, and it differs by being NaN
            for v in (vt, vq, sgm, qc_bl, qi_bl, cld)
                @test isnan(v[1])
                @test all(isfinite, @view v[2:n])
            end
            @test isfinite(mynn_block(ref, cas, 2.5, "B", 1, "sgm")[1])
            for name in ("vt", "vq", "sgm", "qc_bl", "qi_bl", "cldfra_bl")
                got = name == "vt" ? vt : name == "vq" ? vq : name == "sgm" ? sgm :
                      name == "qc_bl" ? qc_bl : name == "qi_bl" ? qi_bl : cld
                e, _ = mynn_maxrel(got, mynn_block(ref, cas, 2.5, "B", 1, name);
                                   first_index = 2)
                @test e == 0.0      # every OTHER level is bitwise identical
            end
        end

        # (2) mym_predict at closure 2.6, :3427  qsq(k) = MAX(x(k), 1e-17).
        # The NaN in qsq(1) poisons the whole tridiagonal solve; gfortran clamps the
        # lot to 1e-17, Julia returns NaN everywhere.
        let work = MYNNWork(n), zed = zeros(n)
            qke = copy(mynn_block(ref, cas, 2.5, "B", 1, "qke_pre_predict"))
            tsq = copy(mynn_block(ref, cas, 2.5, "B", 0, "init_tsq"))
            qsq = copy(mynn_block(ref, cas, 2.5, "B", 0, "init_qsq"))
            cov = copy(mynn_block(ref, cas, 2.5, "B", 0, "init_cov"))
            mym_predict!(1, n, 2.6, colm.delt, colm.dz, colm.ust,
                mynn_scalar(ref, cas, 2.5, "B", 1, "flt"),
                mynn_scalar(ref, cas, 2.5, "B", 1, "flq"),
                mynn_scalar(ref, cas, 2.5, "B", 1, "pmz"),
                mynn_scalar(ref, cas, 2.5, "B", 1, "phh"),
                mynn_block(ref, cas, 2.5, "B", 1, "el"),
                mynn_block(ref, cas, 2.5, "B", 1, "dfq"),
                colm.rho,
                copy(mynn_block(ref, cas, 2.5, "B", 1, "pdk")),
                copy(mynn_block(ref, cas, 2.5, "B", 1, "pdt")),
                copy(mynn_block(ref, cas, 2.5, "B", 1, "pdq")),
                copy(mynn_block(ref, cas, 2.5, "B", 1, "pdc")),
                qke, tsq, qsq, cov,
                mynn_block(ref, cas, 2.5, "B", 1, "s_aw"),
                mynn_block(ref, cas, 2.5, "B", 1, "s_awqke"),
                0, zed, zed, 0, c, work)
            @test all(isnan, qsq)
            @test all(==(1e-17), mynn_block(ref, cas, 2.6, "A", 1, "qsq"))
            # qke, tsq and cov are unaffected: they never touch qsq at any closure.
            for (name, got) in (("qke", qke), ("tsq", tsq), ("cov", cov))
                e, _ = mynn_maxrel(got, mynn_block(ref, cas, 2.6, "A", 1, name))
                @test e == 0.0
            end
        end

        # (3) at closure 2.5 the qsq NaN never reaches mym_predict (qsq is diagnosed
        # from pdq), so the ONLY route into the column state is (1). Confirm that the
        # end-to-end run on case 1 does diverge, and that it diverges by NaN — i.e.
        # that this really is the whole story.
        let col = mynn_column_struct(cas)
            work = MYNNWork(n); st = MYNNColumnState(n)
            opts = MYNNOptions(; closure = 2.5, delt = colm.delt)
            mynn_init_column!(work, c, col, st, opts)
            # the init state itself is bitwise right, NaN and all
            for (name, got) in (("init_el", st.el), ("init_qke", st.qke),
                                ("init_tsq", st.tsq), ("init_qsq", st.qsq),
                                ("init_cov", st.cov))
                e, _ = mynn_maxrel(got, mynn_block(ref, cas, 2.5, "B", 0, name))
                @test e == 0.0
            end
            mynn_column_step!(work, c, col, st, opts; edmf = false)
            @test isnan(st.qke[1])
            @test all(isfinite, mynn_block(ref, cas, 2.5, "A", 1, "qke"))
        end
    end

    @testset "no allocation on a warm call" begin
        cas = "case3_tc_rmw"
        colm = mynn_column(cas)
        n = colm.n
        zw = mynn_face_heights(colm.dz)
        thl, sqw, thetav = mynn_conserved(colm, c)
        work = MYNNWork(n)
        vt  = mynn_block(ref, cas, 2.5, "B", 1, "vt")
        vq  = mynn_block(ref, cas, 2.5, "B", 1, "vq")
        qke = mynn_block(ref, cas, 2.5, "B", 1, "qke_pre_predict")
        dtvr = mynn_block(ref, cas, 2.5, "B", 1, "dtv")
        cld = mynn_block(ref, cas, 2.5, "B", 1, "cldfra_bl")
        ew  = mynn_block(ref, cas, 2.5, "B", 1, "edmf_w")
        ea  = mynn_block(ref, cas, 2.5, "B", 1, "edmf_a")
        rmol = mynn_scalar(ref, cas, 2.5, "B", 1, "rmol")
        flt  = mynn_scalar(ref, cas, 2.5, "B", 1, "flt")
        fltv = mynn_scalar(ref, cas, 2.5, "B", 1, "fltv")
        flq  = mynn_scalar(ref, cas, 2.5, "B", 1, "flq")
        pblh = mynn_scalar(ref, cas, 2.5, "B", 1, "pblh")
        psig = mynn_scalar(ref, cas, 2.5, "B", 1, "psig_bl")
        dtl = zeros(n); dqw = zeros(n); dtv = zeros(n); gm = zeros(n); gh = zeros(n)
        sm  = zeros(n); sh  = zeros(n); el  = zeros(n); qkw = zeros(n)
        tsq = zeros(n); qsq = zeros(n); cov = zeros(n); qke2 = copy(qke)
        qtke = [0.5*max(qke[k], 1e-4) for k in 1:n]
        a = fill(-1.0, n); b = fill(4.0, n); ccc = fill(-1.5, n)
        d = [sinpi(k/9) for k in 1:n]; x = zeros(n)

        f_level2()  = mym_level2!(1, n, colm.dz, colm.u, colm.v, thl, thetav, sqw,
                                  colm.sqc, vt, vq, dtl, dqw, dtv, gm, gh, sm, sh, c)
        f_length()  = mym_length!(1, n, colm.xland, colm.dz, colm.dx, zw, rmol, flt,
                                  fltv, flq, vt, vq, colm.u, colm.v, qke, dtvr, el,
                                  pblh, colm.th, qkw, psig, cld, 2, ew, ea, c, work)
        f_pblh()    = get_pblh!(1, n, thetav, qke, zw, colm.dz, colm.xland)
        f_scale()   = scale_aware(colm.dx, pblh)
        f_boulac()  = boulac_length0!(50, 1, n, zw, colm.dz, qtke, colm.th, c)
        f_tri()     = tridiag2!(n, a, b, ccc, d, x, work)
        f_init()    = mym_initialize!(1, n, colm.xland, colm.dz, colm.dx, zw, colm.u,
                                      colm.v, thl, colm.sqv, pblh, colm.th, thetav, sh,
                                      sm, colm.ust, colm.rmol, el, qke2, tsq, qsq, cov,
                                      psig, cld, 2, ew, ea, true, c, work)
        f_blends()  = esat_blend(283.0, c) + qsat_blend(283.0, 9.0e4, c) +
                      xl_blend(283.0, c) + phim(-0.5) + phih(-0.5) + phim(0.5)

        # -- part 2. Every reference block is hoisted into a local first: a
        #    `mynn_block` call inside the closure would allocate in the DICT, not in
        #    the routine, and would mask the thing being measured.
        vt1 = vt; vq1 = vq
        tsq0 = mynn_block(ref, cas, 2.5, "B", 0, "init_tsq")
        qsq0 = mynn_block(ref, cas, 2.5, "B", 0, "init_qsq")
        cov0 = mynn_block(ref, cas, 2.5, "B", 0, "init_cov")
        sh0  = mynn_block(ref, cas, 2.5, "B", 0, "init_sh")
        el0  = mynn_block(ref, cas, 2.5, "B", 0, "init_el")
        pblh0 = mynn_scalar(ref, cas, 2.5, "B", 0, "init_pblh")
        el1  = mynn_block(ref, cas, 2.5, "B", 1, "el")
        dfq1 = mynn_block(ref, cas, 2.5, "B", 1, "dfq")
        dfm1 = mynn_block(ref, cas, 2.5, "B", 1, "dfm")
        dfh1 = mynn_block(ref, cas, 2.5, "B", 1, "dfh")
        tcd1 = mynn_block(ref, cas, 2.5, "B", 1, "tcd")
        qcd1 = mynn_block(ref, cas, 2.5, "B", 1, "qcd")
        dh1  = mynn_block(ref, cas, 2.5, "B", 1, "diss_heat")
        saw1 = mynn_block(ref, cas, 2.5, "B", 1, "s_aw")
        sawq = mynn_block(ref, cas, 2.5, "B", 1, "s_awqke")
        pmz1 = mynn_scalar(ref, cas, 2.5, "B", 1, "pmz")
        phh1 = mynn_scalar(ref, cas, 2.5, "B", 1, "phh")
        shcu = mynn_scalar(ref, cas, 2.5, "B", 1, "psig_shcu")
        zed  = zeros(n); zed1 = zeros(n + 1)
        shw = zeros(n); smw = zeros(n); elw = zeros(n)
        dfmw = zeros(n); dfhw = zeros(n); dfqw = zeros(n)
        tcdw = zeros(n); qcdw = zeros(n)
        pdkw = zeros(n); pdtw = zeros(n); pdqw = zeros(n); pdcw = zeros(n)
        qkew = copy(qke); tsqw = copy(tsq0); qsqw = copy(qsq0); covw = copy(cov0)
        pdk1 = copy(mynn_block(ref, cas, 2.5, "B", 1, "pdk"))
        pdt1 = copy(mynn_block(ref, cas, 2.5, "B", 1, "pdt"))
        pdq1 = copy(mynn_block(ref, cas, 2.5, "B", 1, "pdq"))
        pdc1 = copy(mynn_block(ref, cas, 2.5, "B", 1, "pdc"))
        qcbw = zeros(n); qibw = zeros(n); cldw = zeros(n)
        vtw = zeros(n); vqw = zeros(n); sgmw = zeros(n)
        thlw = copy(thl); sqww = copy(sqw)
        sqvw = copy(colm.sqv); sqcw = copy(colm.sqc); sqiw = copy(colm.sqi)
        qvw = [colm.sqv[k]/(1.0 - colm.sqv[k]) for k in 1:n]
        qcw = [colm.sqc[k]/(1.0 - colm.sqv[k]) for k in 1:n]
        qiw = [colm.sqi[k]/(1.0 - colm.sqv[k]) for k in 1:n]
        duw = zeros(n); dvw = zeros(n); dthw = zeros(n); dqvw = zeros(n)
        dqcw = zeros(n); dqiw = zeros(n); dqsw = zeros(n)
        d1 = zeros(n); d2 = zeros(n); d3 = zeros(n); d4 = zeros(n)
        d5 = zeros(n); d6 = zeros(n)
        kmw = zeros(n); khw = zeros(n)
        col2 = mynn_column_struct(cas)
        work2 = MYNNWork(n); st2 = MYNNColumnState(n)
        opts2 = MYNNOptions(; closure = 2.5, delt = colm.delt)

        f_turb() = mym_turbulence!(1, n, colm.xland, 2.5, colm.dz, colm.dx, zw,
            colm.u, colm.v, thl, thetav, colm.sqc, sqw, qke, tsq0, qsq0, cov0,
            vt1, vq1, rmol, flt, fltv, flq, pblh, colm.th, shw, smw, elw,
            dfmw, dfhw, dfqw, tcdw, qcdw, pdkw, pdtw, pdqw, pdcw,
            zed, zed, zed, zed, 0, psig, shcu, cld, 2, ew, ea, zed, 0, zed, c, work)
        f_pred() = mym_predict!(1, n, 2.5, colm.delt, colm.dz, colm.ust, flt, flq,
            pmz1, phh1, el1, dfq1, colm.rho, pdk1, pdt1, pdq1, pdc1,
            qkew, tsqw, qsqw, covw, saw1, sawq, 0, zed, zed, 0, c, work)
        f_cond() = mym_condensation!(1, n, colm.dx, colm.dz, zw, colm.xland,
            thl, sqw, colm.sqv, colm.sqc, colm.sqi, zed, colm.p, colm.exner,
            tsq0, qsq0, cov0, sh0, el0, 2, qcbw, qibw, cldw, pblh0, colm.hfx,
            vtw, vqw, colm.th, sgmw, colm.rmol, 0, zed, c, work)
        f_tend() = mynn_tendencies!(1, n, 1, colm.delt, colm.dz, colm.rho,
            colm.u, colm.v, colm.th, colm.T, qvw, qcw, qiw, zed, zed, zed,
            colm.ps, colm.p, colm.exner, thlw, sqvw, sqcw, sqiw, zed, sqww,
            zed, zed, zed, zed, colm.ust, flt, flq, flq, 0.0, colm.wspd, 0.0, 0.0,
            tsq0, qsq0, cov0, tcd1, qcd1, dfm1, dfh1, dfq1,
            duw, dvw, dthw, dqvw, dqcw, dqiw, dqsw, d1, d2, d3, d4, d5, d6, dh1,
            saw1, zed1, zed1, zed1, zed1, zed1, zed1, zed1, zed1, zed1, zed1, zed1,
            zed1, zed1, zed1, zed1, zed1, zed1, zed1,
            zed, zed, zed, zed, zed, zed, zed, zed, zed,
            true, true, false, false, false, false, false, false, cld,
            1, 0, 1, 1, 1, c, work)
        f_mcheck() = moisture_check!(n, colm.delt, colm.p, colm.exner,
            sqvw, sqcw, sqiw, dqsw, thlw, dqvw, dqcw, dqiw, d1, dthw, c)
        f_retr() = retrieve_exchange_coeffs!(1, n, dfm1, dfh1, colm.dz, kmw, khw)
        f_colinit() = mynn_init_column!(work2, c, col2, st2, opts2)
        f_colstep() = mynn_column_step!(work2, c, col2, st2, opts2; edmf = false)

        for (name, f) in (("mym_level2!", f_level2), ("mym_length!", f_length),
                          ("get_pblh!", f_pblh), ("scale_aware", f_scale),
                          ("boulac_length0!", f_boulac), ("tridiag2!", f_tri),
                          ("mym_initialize!", f_init), ("blends+phim/phih", f_blends),
                          ("mym_turbulence!", f_turb), ("mym_predict!", f_pred),
                          ("mym_condensation!", f_cond), ("mynn_tendencies!", f_tend),
                          ("moisture_check!", f_mcheck),
                          ("retrieve_exchange_coeffs!", f_retr),
                          ("mynn_init_column!", f_colinit),
                          ("mynn_column_step!", f_colstep))
            f()                       # warm up: compile and touch every branch once
            @test (@allocations f()) == 0
        end
    end
end

# ── Observed headroom, for the record ─────────────────────────────────────────
let names = sort!(collect(keys(MYNN_OBSERVED)))
    println("\nMYNN closure parts 1+2 — max relative error vs. the Fortran reference:")
    for nm in names
        e = MYNN_OBSERVED[nm]
        println("  ", rpad(nm, 30), e == 0.0 ? "0 (bitwise)" : string(e))
    end
end
