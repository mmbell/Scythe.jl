using Test
using Scythe

# ──────────────────────────────────────────────
# Fortran reference values (test/reference/ishmael_fortran_refs.jl), built
# from tools/ishmael_fortran_driver/ -- see that directory's README.md.
# No runtime Fortran dependency: these are hand-transcribed Float64
# literals from ref_driver_output.txt.
# ──────────────────────────────────────────────
include(joinpath(@__DIR__, "reference", "ishmael_fortran_refs.jl"))

const ISHMAEL_JLD2_PATH2 = joinpath(@__DIR__, "..", "data", "ishmael_tables.jld2")

const NU_ = 4.0
const AO_ = 0.1e-6
const GAMMNU_ = 6.0                 # gamma(4.0)
const I_GAMMNU_ = 1.0 / 6.0
const FOURTHIRDSPI_ = 4.0 / 3.0 * 3.14159265
const RV_ = 461.5                   # moist air gas constant, J kg^-1 K^-1 (Fortran module const)

@testset "ISHMAEL process rates (Stage S6a)" begin

    # ──────────────────────────────────────────────
    # 0. Utility-layer cross-check against the Fortran reference (Part A
    #    "first use"). These functions live in ishmael_tables.jl, not this
    #    file, but this is where the Fortran reference driver's output
    #    gets exercised, so the cross-check is recorded here.
    # ──────────────────────────────────────────────
    @testset "Utility layer cross-check (get_igr, capacitance_gamma, var_check)" begin
        for (t, ref) in zip(ISHMAEL_REF_IGR.temps, ISHMAEL_REF_IGR.igr)
            @test Scythe.get_igr(Scythe.ISHMAEL_IGRDATA, t) ≈ ref rtol=1.0e-5
        end

        for pt in ISHMAEL_REF_POINTS
            vc = pt.var_check
            j = Scythe.capacitance_gamma(vc.ani, vc.deltastr, NU_, vc.alphstr, I_GAMMNU_)
            @test j ≈ pt.capgam rtol=1.0e-5

            aidum0 = pt.input.ani0^2 * pt.input.cni0 * pt.input.nidum
            cidum0 = pt.input.cni0^2 * pt.input.ani0 * pt.input.nidum
            r = Scythe.ishmael_var_check(NU_, AO_, FOURTHIRDSPI_, GAMMNU_, pt.input.qidum,
                pt.input.dsdum0, pt.input.ani0, pt.input.cni0, pt.input.rbdum0, pt.input.nidum,
                aidum0, cidum0)
            @test r.deltastr ≈ vc.deltastr rtol=1.0e-5
            @test r.ani ≈ vc.ani rtol=1.0e-5
            @test r.cni ≈ vc.cni rtol=1.0e-5
            @test r.rni ≈ vc.rni rtol=1.0e-5
            @test r.rhobar ≈ vc.rhobar rtol=1.0e-5
            @test r.ni ≈ vc.ni rtol=1.0e-5
            @test r.alphstr ≈ vc.alphstr rtol=1.0e-5
            @test r.alphv ≈ vc.alphv rtol=1.0e-5
            @test r.betam ≈ vc.betam rtol=1.0e-5
        end
    end

    if !isfile(ISHMAEL_JLD2_PATH2)
        @warn "ISHMAEL tables data file missing; skipping access_lookup_table cross-check" path = ISHMAEL_JLD2_PATH2
        @testset "access_lookup_table cross-check (skipped: data file missing)" begin
            @test_skip isfile(ISHMAEL_JLD2_PATH2)
        end
    else
        tables_ = Scythe.load_ishmael_tables(ISHMAEL_JLD2_PATH2)
        @testset "access_lookup_table cross-check" begin
            L = ISHMAEL_REF_LUT_ITAB
            for i in eachindex(L.dumjj)
                j = Scythe.access_lookup_table(tables_.itab, L.dumjj[i], L.dumii[i], L.dumi[i],
                    L.dumk[i], L.index[i], L.dum1[i], L.dum2[i], L.dum4[i], L.dum5[i])
                if L.proc[i] == 0.0
                    @test j == 0.0
                else
                    @test j ≈ L.proc[i] rtol=1.0e-5
                end
            end
            R = ISHMAEL_REF_LUT_ITABR
            for i in eachindex(R.dumjj)
                j = Scythe.access_lookup_table(tables_.itabr, R.dumjj[i], R.dumii[i], R.dumi[i],
                    R.dumk[i], R.index[i], R.dum1[i], R.dum2[i], R.dum4[i], R.dum5[i])
                if R.proc[i] == 0.0
                    @test j == 0.0
                else
                    @test j ≈ R.proc[i] rtol=1.0e-5
                end
            end
        end
    end

    # ──────────────────────────────────────────────
    # 1. ishmael_fall_speeds / ishmael_ventilation / ishmael_vapor_coefficients
    # ──────────────────────────────────────────────
    @testset "ishmael_fall_speeds Fortran cross-check" begin
        for pt in ISHMAEL_REF_POINTS
            vc = pt.var_check
            d = pt.derived
            vg = pt.vaporgrow
            fs = Scythe.ishmael_fall_speeds(vc.ani, vc.cni, vc.deltastr, NU_, I_GAMMNU_,
                vc.alphstr, vc.rhobar, d.rhoair, d.mu)
            @test fs.vtrni1 ≈ vg.vtbarb rtol=1.0e-4
            @test fs.vtrmi1 ≈ vg.vtbarbm rtol=1.0e-4
            @test fs.vtrzi1 ≈ vg.vtbarbz rtol=1.0e-4

            vent = Scythe.ishmael_ventilation(d.nsch, d.npr, fs.Nre)
            @test vent.fv ≈ vg.fvdum rtol=1.0e-5
            @test vent.fh ≈ vg.fhdum rtol=1.0e-5

            vco = Scythe.ishmael_vapor_coefficients(vc.ani, vc.cni, vc.deltastr, NU_, I_GAMMNU_,
                vc.alphstr, vc.rhobar, d.rhoair, d.mu, d.nsch, d.npr)
            @test vco.Cbar ≈ pt.capgam rtol=1.0e-5
            @test vco.fv ≈ vg.fvdum rtol=1.0e-5
            @test vco.fh ≈ vg.fhdum rtol=1.0e-5
            @test vco.vtrni1 ≈ vg.vtbarb rtol=1.0e-4
            @test vco.vtrmi1 ≈ vg.vtbarbm rtol=1.0e-4
        end
    end

    @testset "ishmael_fall_speeds: xm > 1e8 Mitchell-Heymsfield fallback" begin
        p = ISHMAEL_REF_XM_FALLBACK
        alphstr = AO_^(1.0 - p.dsdum)
        cni = alphstr * p.ani^p.dsdum
        rhoair = 101325.0 / (287.15 * p.temp)
        mu = 1.496e-6 * p.temp^1.5 / (p.temp + 120.0)
        fs = Scythe.ishmael_fall_speeds(p.ani, cni, p.dsdum, NU_, I_GAMMNU_, alphstr, p.rbdum,
            rhoair, mu)
        @test fs.vtrni1 ≈ p.vtbarb rtol=1.0e-4
        @test fs.vtrmi1 ≈ p.vtbarbm rtol=1.0e-4
        @test fs.vtrzi1 ≈ p.vtbarbz rtol=1.0e-4
    end

    @testset "ishmael_fall_speeds: physical checks" begin
        # Fall speeds positive, <= 25 m/s cap, mass-weighted >= number-weighted
        # (broad NU=4 distribution: larger particles dominate the mass average).
        for pt in ISHMAEL_REF_POINTS
            vc = pt.var_check
            d = pt.derived
            fs = Scythe.ishmael_fall_speeds(vc.ani, vc.cni, vc.deltastr, NU_, I_GAMMNU_,
                vc.alphstr, vc.rhobar, d.rhoair, d.mu)
            @test fs.vtrni1 > 0.0
            @test fs.vtrmi1 > 0.0
            @test fs.vtrni1 <= 25.0
            @test fs.vtrmi1 <= 25.0
            @test fs.vtrmi1 >= fs.vtrni1
        end

        # xm > 1e8 fallback point: still capped and positive
        p = ISHMAEL_REF_XM_FALLBACK
        alphstr = AO_^(1.0 - p.dsdum)
        cni = alphstr * p.ani^p.dsdum
        rhoair = 101325.0 / (287.15 * p.temp)
        mu = 1.496e-6 * p.temp^1.5 / (p.temp + 120.0)
        fs = Scythe.ishmael_fall_speeds(p.ani, cni, p.dsdum, NU_, I_GAMMNU_, alphstr, p.rbdum,
            rhoair, mu)
        @test 0.0 < fs.vtrni1 <= 25.0
        @test 0.0 < fs.vtrmi1 <= 25.0
    end

    @testset "ishmael_fall_speeds: melting size-sorting override" begin
        vc = ISHMAEL_REF_POINTS[1].var_check
        d = ISHMAEL_REF_POINTS[1].derived
        fs_normal = Scythe.ishmael_fall_speeds(vc.ani, vc.cni, vc.deltastr, NU_, I_GAMMNU_,
            vc.alphstr, vc.rhobar, d.rhoair, d.mu; in_melting=false)
        fs_melt = Scythe.ishmael_fall_speeds(vc.ani, vc.cni, vc.deltastr, NU_, I_GAMMNU_,
            vc.alphstr, vc.rhobar, d.rhoair, d.mu; in_melting=true)
        @test fs_melt.vtrni1 == fs_melt.vtrmi1
        @test fs_normal.vtrmi1 == fs_melt.vtrmi1   # vtrmi1 itself unaffected by the override
        @test fs_normal.vtrni1 != fs_melt.vtrni1   # but vtrni1 changes (unless coincidentally equal)
    end

    @testset "ishmael_ventilation: physical checks" begin
        for Nre in (0.01, 1.0, 10.0, 1000.0), nsch in (0.5, 0.6, 1.5), npr in (0.5, 0.7, 2.0)
            v = Scythe.ishmael_ventilation(nsch, npr, Nre)
            @test v.fv >= 1.0
            @test v.fh >= 1.0
            @test isfinite(v.fv)
            @test isfinite(v.fh)
        end
    end

    # ──────────────────────────────────────────────
    # 2. ishmael_deposition_partition -- end-to-end reconstruction against
    #    the vaporgrow Fortran reference. `afn`/`maxsui`/`sui_negative` are
    #    NOT ported (see the function's docstring "SEAM" section) -- to
    #    validate the port end-to-end we recompute them here using the
    #    EXACT excluded Fortran formulas (lines 3327-3368) purely as test
    #    scaffolding, from the reference's own recorded sui/sup/qvi/qvs/
    #    fvdum/fhdum/capgam/dv. This is the strongest available check that
    #    the seam is placed correctly: given what vaporgrow itself would
    #    have computed for afn/maxsui, the port reproduces vaporgrow's
    #    anf/cnf/rnf/iwcf/dsdumout/rbdum outputs.
    # ──────────────────────────────────────────────
    @testset "ishmael_deposition_partition: end-to-end vaporgrow reconstruction" begin
        for pt in ISHMAEL_REF_POINTS
            d = pt.derived
            vc = pt.var_check
            vg = pt.vaporgrow
            temp = pt.input.temp

            if temp > ISHMAEL_REF_T0
                # T > T0 passthrough: no afn/maxsui reconstruction needed.
                r = Scythe.ishmael_deposition_partition(2.0, vc.ani, vc.cni, vc.rni, vc.deltastr,
                    vc.rhobar, vc.ni, d.igr, 0.0, 0.0, vg.vtbarbm, false, pt.capgam, d.dv, temp,
                    AO_, NU_, GAMMNU_, I_GAMMNU_, FOURTHIRDSPI_)
                @test r.anf == vc.ani
                @test r.cnf == vc.cni
                @test r.rnf == vc.rni
                @test r.ard == 0.0
                @test r.crd == 0.0
                continue
            end

            svpi = d.qvi * pt.input.pres / (0.622 + d.qvi)   # invert qvi = 0.622*svpi/(pres-svpi)
            xxls = 3.15e6 - 2370.0 * temp + 0.3337e6
            xxlv = 3.1484e6 - 2370.0 * temp
            xxlf = xxls - xxlv
            alpha = d.dv * vg.fvdum * svpi * xxls / (RV_ * d.kt * vg.fhdum * temp)
            del2 = d.sui / (temp / alpha + (xxls / (RV_ * temp) - 1.0))       # rimesum=0 => del1=0
            afn = (d.dv * vg.fvdum * svpi) / (RV_ * temp) * (d.sui - del2 * (xxls / (RV_ * temp) - 1.0))

            maxsui = if d.sup >= 0.0
                1.0
            elseif d.sui >= 0.0 && d.qvi < d.qvs
                clamp(((d.sui + 1.0) * d.qvi) / (d.qvs - d.qvi) - d.qvi / (d.qvs - d.qvi), 0.0, 1.0)
            else
                0.0
            end
            sui_negative = d.sui < 0.0

            r = Scythe.ishmael_deposition_partition(2.0, vc.ani, vc.cni, vc.rni, vc.deltastr,
                vc.rhobar, vc.ni, d.igr, afn, maxsui, vg.vtbarbm, sui_negative, pt.capgam, d.dv,
                temp, AO_, NU_, GAMMNU_, I_GAMMNU_, FOURTHIRDSPI_)

            @test r.anf ≈ vg.anf rtol=1.0e-4
            @test r.cnf ≈ vg.cnf rtol=1.0e-4
            @test r.rnf ≈ vg.rnf rtol=1.0e-4
            @test r.iwcf ≈ vg.iwcf rtol=1.0e-4
            @test r.rdout ≈ vg.rdout rtol=1.0e-4
            @test r.dsdumout ≈ vg.dsdumout rtol=1.0e-4
            @test r.rbdum ≈ vg.rbdum rtol=1.0e-4
            @test isfinite(r.ard)
            @test isfinite(r.crd)
        end
    end

    @testset "ishmael_deposition_partition: safety through var_check effective values" begin
        # Feed every var_check-clamped state point straight through with a
        # small positive/negative afn; nothing should be NaN/Inf.
        for pt in ISHMAEL_REF_POINTS
            temp = pt.input.temp
            temp > ISHMAEL_REF_T0 && continue
            vc = pt.var_check
            d = pt.derived
            for afn in (1.0e-9, -1.0e-9)
                r = Scythe.ishmael_deposition_partition(2.0, vc.ani, vc.cni, vc.rni, vc.deltastr,
                    vc.rhobar, vc.ni, d.igr, afn, 0.5, 0.05, afn < 0.0, pt.capgam, d.dv, temp,
                    AO_, NU_, GAMMNU_, I_GAMMNU_, FOURTHIRDSPI_)
                for fn in (:anf, :cnf, :rnf, :iwcf, :rdout, :dsdumout, :rbdum, :ard, :crd)
                    @test isfinite(getfield(r, fn))
                end
                @test r.anf > 0.0
                @test r.cnf > 0.0
                @test r.rnf > 0.0
            end
        end
    end

    # ──────────────────────────────────────────────
    # 3. Nucleation set
    # ──────────────────────────────────────────────
    @testset "ishmael_nucleation_demott" begin
        # Inside the window: T < T0, sup >= 0
        r = Scythe.ishmael_nucleation_demott(250.0, 0.02, 2.0, 1.0, 1.0e5, 1.0e5)
        @test r.mnuccd > 0.0
        @test r.nnuccd > 0.0
        @test isfinite(r.mnuccd) && isfinite(r.nnuccd)

        # Outside: T >= T0
        r_warm = Scythe.ishmael_nucleation_demott(280.0, 0.02, 2.0, 1.0, 1.0e5, 1.0e5)
        @test r_warm == (mnuccd=0.0, nnuccd=0.0)

        # Outside: sup < 0
        r_dry = Scythe.ishmael_nucleation_demott(250.0, -0.01, 2.0, 1.0, 1.0e5, 1.0e5)
        @test r_dry == (mnuccd=0.0, nnuccd=0.0)

        # The 10 L^-1 existing-ice cap throttles nucleation once ni1+ni2 is
        # already saturating: with an enormous existing-ice number, ratel
        # -> 0 and mnuccd/nnuccd -> 0 (but not negative).
        r_capped = Scythe.ishmael_nucleation_demott(250.0, 0.02, 2.0, 1.0, 1.0e10, 1.0e10)
        @test r_capped.mnuccd >= 0.0
        @test r_capped.nnuccd >= 0.0
        @test r_capped.nnuccd < r.nnuccd
    end

    @testset "ishmael_homogeneous_freezing" begin
        # Mass- and number-conservation: entire qc/qr reservoir converts in dt.
        dt = 2.0
        r = Scythe.ishmael_homogeneous_freezing(230.0, 1.0e-3, 1.0e8, 2.0e-3, 5.0e5, dt)
        @test r.mim * dt ≈ 1.0e-3
        @test r.nim * dt ≈ 1.0e8
        @test r.mimr * dt ≈ 2.0e-3
        @test r.nimr * dt ≈ 5.0e5

        # Zero above -35C
        r_warm = Scythe.ishmael_homogeneous_freezing(240.0, 1.0e-3, 1.0e8, 2.0e-3, 5.0e5, dt)
        @test r_warm == (mim=0.0, nim=0.0, mimr=0.0, nimr=0.0)

        # Zero with no condensate (even below -35C)
        r_dry = Scythe.ishmael_homogeneous_freezing(230.0, 0.0, 0.0, 0.0, 0.0, dt)
        @test r_dry == (mim=0.0, nim=0.0, mimr=0.0, nimr=0.0)

        # Exactly at -35C: strict "<", so NOT frozen (matches Fortran's temp.lt.(T0-35.))
        r_boundary = Scythe.ishmael_homogeneous_freezing(ISHMAEL_REF_T0 - 35.0, 1.0e-3, 1.0e8,
            0.0, 0.0, dt)
        @test r_boundary.mim == 0.0
    end

    @testset "ishmael_bigg_freezing" begin
        r = Scythe.ishmael_bigg_freezing(260.0, 1.0e-3, 1.0e6, 2.0)
        @test r.mbiggr > 0.0
        @test r.nbiggr > 0.0
        @test r.mbiggr <= 1.0e-3 / 2.0   # clamp: mbiggr <= qr*i_dt
        @test r.nbiggr <= 1.0e6 / 2.0    # clamp: nbiggr <= nr*i_dt (post rain-lambda re-diagnosis)

        # Zero above -4C
        r_warm = Scythe.ishmael_bigg_freezing(272.0, 1.0e-3, 1.0e6, 2.0)
        @test r_warm == (mbiggr=0.0, nbiggr=0.0)

        # Zero with no rain
        r_dry = Scythe.ishmael_bigg_freezing(260.0, 0.0, 0.0, 2.0)
        @test r_dry == (mbiggr=0.0, nbiggr=0.0)
    end

    @testset "ishmael_rime_splintering" begin
        # fmult == 0 exactly outside [265.16, 270.16] (the "explicit fix")
        for temp in (270.16, 265.16, 260.0, 275.0, 271.0, 264.0)
            r = Scythe.ishmael_rime_splintering(temp, 1.0e-6)
            @test r.fmult == 0.0
            @test r.nmult == 0.0
            @test r.qmult == 0.0
            @test r.prdr == 1.0e-6   # riming rate passes through unchanged
        end

        # Peak of the triangular window at 268.16 K (from both sub-branches):
        # branch (>=265.16,<=268.16): fmult=(268.16-265.16)/3=1.0
        r_peak = Scythe.ishmael_rime_splintering(268.16, 1.0e-6)
        @test r_peak.fmult ≈ 1.0

        # Interior points are strictly between 0 and 1, and nmult/qmult are
        # positive with prdr reduced by exactly qmult.
        r_mid = Scythe.ishmael_rime_splintering(269.0, 1.0e-6)
        @test 0.0 < r_mid.fmult < 1.0
        @test r_mid.nmult > 0.0
        @test r_mid.qmult > 0.0
        @test r_mid.prdr ≈ 1.0e-6 - r_mid.qmult

        # No riming: no splinters regardless of temperature
        r_norime = Scythe.ishmael_rime_splintering(268.16, 0.0)
        @test r_norime.nmult == 0.0
        @test r_norime.qmult == 0.0
        @test r_norime.prdr == 0.0

        # qmult never exceeds the available riming rate (the min(qmult,prdr) clamp)
        r_tiny_rime = Scythe.ishmael_rime_splintering(268.16, 1.0e-15)
        @test r_tiny_rime.qmult <= 1.0e-15
        @test r_tiny_rime.prdr >= 0.0
    end

    # ──────────────────────────────────────────────
    # 4. ishmael_rain_lambda
    # ──────────────────────────────────────────────
    @testset "ishmael_rain_lambda" begin
        d = Scythe.ishmael_rain_lambda(1.0e-3, 1.0e6)
        @test d.lamr > 0.0
        @test d.n0rr > 0.0
        @test d.nr > 0.0
        @test Scythe.ISHMAEL_LAMMINR <= d.lamr <= Scythe.ISHMAEL_LAMMAXR

        # Clamp branches: extremely large nr/qr ratio drives lamr above LAMMAXR
        d_clamp_hi = Scythe.ishmael_rain_lambda(1.0e-9, 1.0e10)
        @test d_clamp_hi.lamr ≈ Scythe.ISHMAEL_LAMMAXR
        @test d_clamp_hi.nr != 1.0e10   # nr re-diagnosed after clamp

        # Extremely small nr/qr ratio drives lamr below LAMMINR
        d_clamp_lo = Scythe.ishmael_rain_lambda(1.0, 1.0)
        @test d_clamp_lo.lamr ≈ Scythe.ISHMAEL_LAMMINR
        @test d_clamp_lo.nr != 1.0

        # nr floored at QNSMALL before the lambda calc
        d_floor = Scythe.ishmael_rain_lambda(1.0e-3, 0.0)
        @test isfinite(d_floor.lamr)
    end

    # ──────────────────────────────────────────────
    # 5. ishmael_melting, including the rhoair(cc)->rhoair(k) bug fix
    # ──────────────────────────────────────────────
    @testset "ishmael_melting" begin
        pt = ISHMAEL_REF_POINTS[7]   # "above 0C, T=+2C" reference point
        @test pt.label == "above 0C, T=+2C (melting/passthrough)"
        d = pt.derived
        vc = pt.var_check
        temp = pt.input.temp

        # qs0 = 0.622*polysvp(T0,0)/(pres-polysvp(T0,0)) in the Fortran, i.e.
        # qvs evaluated AT T0 rather than at `temp` -- not directly available
        # from the reference. Since temp is only 2C above T0 here, d.qvs (qvs
        # at temp) is a close physical stand-in, and this test only needs
        # SOME finite, physically-scaled qs0 (it is not itself under test).
        qs0 = d.qvs

        r = Scythe.ishmael_melting(temp, vc.ni, vc.ani, vc.cni, vc.deltastr, vc.rhobar,
            pt.input.qidum, 1.0e-10, 1.0e-10, d.kt, 1.1, d.rhoair, 2.5e6, d.dv, 1.1, qs0, d.qv,
            2.7e5, 0.0, 0.0, 0.0, 2.0, vc.alphstr, GAMMNU_, I_GAMMNU_, FOURTHIRDSPI_)

        @test r.qmlt <= 0.0
        @test isfinite(r.qmlt) && isfinite(r.nmlt) && isfinite(r.amlt) && isfinite(r.cmlt)

        # Zero below freezing
        r_cold = Scythe.ishmael_melting(260.0, vc.ni, vc.ani, vc.cni, vc.deltastr, vc.rhobar,
            pt.input.qidum, 1.0e-10, 1.0e-10, d.kt, 1.1, d.rhoair, 2.5e6, d.dv, 1.1, qs0, d.qv,
            2.7e5, 0.0, 0.0, 0.0, 2.0, vc.alphstr, GAMMNU_, I_GAMMNU_, FOURTHIRDSPI_)
        @test r_cold == (qmlt=0.0, nmlt=0.0, amlt=0.0, cmlt=0.0)

        # ── The rhoair(cc)/rhoair(k) bug (line 1512) ──
        #
        # The Fortran indexes the AIR DENSITY in qmlt's numerator by the ICE-
        # SPECIES loop variable `cc` (1..3) instead of the vertical-LEVEL
        # index `k`: `rhoair(cc)*xxlv*dv*fv(cc)*(qs0-qv(k))` should read
        # `rhoair(k)*...`. `rhoair` is declared `dimension(kts:kte)` (a
        # per-level array); reading `rhoair(cc)` silently substitutes the air
        # density from vertical level `cc` for the current level's, wrong
        # whenever `cc != k`.
        #
        # This scalar pure-function API takes a single `rhoair` argument, so
        # there is no per-species array through which the aliasing bug could
        # recur -- but to PROVE the port's result differs from what the
        # Fortran's buggy indexing would have produced (not merely that the
        # bug "can't happen" by construction), we evaluate the qmlt formula
        # by hand BOTH ways: once with the correct level rhoair (matching our
        # port), and once substituting an unrelated rhoair value (standing in
        # for "level cc's" air density) as the Fortran bug actually does.
        rhoair_true = d.rhoair          # the correct level-k air density
        rhoair_bug  = 0.55              # a different level's air density (e.g. much higher up)
        @test rhoair_bug != rhoair_true

        function qmlt_hand(rhoair_numerator)
            raw = 2.0 * 3.14159265 * (d.kt * 1.1 * (ISHMAEL_REF_T0 - temp) +
                  rhoair_numerator * 2.5e6 * d.dv * 1.1 * (qs0 - d.qv)) / 2.7e5 *
                  (vc.ni * NU_ * max(vc.ani, vc.cni)) -
                  (Scythe.ISHMAEL_CPW / 2.7e5 * (temp - ISHMAEL_REF_T0) * (0.0 / d.rhoair + 0.0))
            raw = min(raw, 0.0)
            raw = max(raw, -pt.input.qidum / 2.0)
            return raw
        end

        qmlt_correct = qmlt_hand(rhoair_true)
        qmlt_buggy   = qmlt_hand(rhoair_bug)

        @test qmlt_correct != qmlt_buggy   # the bug is a REAL, observable divergence
        @test r.qmlt ≈ qmlt_correct rtol=1.0e-8   # our port matches the CORRECTED physics
    end

    # ──────────────────────────────────────────────
    # 6. Zero-allocation hot-path checks
    # ──────────────────────────────────────────────
    @testset "Zero allocations" begin
        vc = ISHMAEL_REF_POINTS[1].var_check
        d = ISHMAEL_REF_POINTS[1].derived

        Scythe.ishmael_fall_speeds(vc.ani, vc.cni, vc.deltastr, NU_, I_GAMMNU_, vc.alphstr,
            vc.rhobar, d.rhoair, d.mu)   # warm up (compile)
        @test @allocated(Scythe.ishmael_fall_speeds(vc.ani, vc.cni, vc.deltastr, NU_, I_GAMMNU_,
            vc.alphstr, vc.rhobar, d.rhoair, d.mu)) == 0

        Scythe.ishmael_ventilation(d.nsch, d.npr, 10.0)
        @test @allocated(Scythe.ishmael_ventilation(d.nsch, d.npr, 10.0)) == 0

        Scythe.ishmael_rain_lambda(1.0e-3, 1.0e6)
        @test @allocated(Scythe.ishmael_rain_lambda(1.0e-3, 1.0e6)) == 0
    end
end

# ══════════════════════════════════════════════════════════════════════════
# Stage S6b: ice-cloud/ice-rain collection (riming), rime density,
# wet-growth check, aggregation.
# ══════════════════════════════════════════════════════════════════════════
@testset "ISHMAEL process rates (Stage S6b)" begin
    if !isfile(ISHMAEL_JLD2_PATH2)
        @warn "ISHMAEL tables data file missing; skipping Stage S6b tests entirely" path = ISHMAEL_JLD2_PATH2
        @testset "Stage S6b (skipped: data file missing)" begin
            @test_skip isfile(ISHMAEL_JLD2_PATH2)
        end
    else
        tables_ = Scythe.load_ishmael_tables(ISHMAEL_JLD2_PATH2)

        # ──────────────────────────────────────────────
        # 1. ishmael_ice_cloud_riming
        # ──────────────────────────────────────────────
        @testset "ishmael_ice_cloud_riming: Fortran cross-check" begin
            for pt in ISHMAEL_REF_POINTS
                vc = pt.var_check
                ri = pt.riming_input
                ref = pt.itab_riming
                r = Scythe.ishmael_ice_cloud_riming(tables_.itab, vc.rni, ri.qc, vc.deltastr,
                    vc.rhobar, vc.ni, ri.nc, pt.derived.rhoair)
                @test r.rimesum ≈ ref.rimesum rtol=1.0e-4 atol=1.0e-30
                @test r.qi_qc_nrm ≈ ref.qi_qc_nrm rtol=1.0e-4 atol=1.0e-30
                @test r.qi_qc_nrd ≈ ref.qi_qc_nrd rtol=1.0e-4 atol=1.0e-30
            end
        end

        @testset "ishmael_ice_cloud_riming: physical checks" begin
            pt = ISHMAEL_REF_POINTS[1]
            vc = pt.var_check
            ri = pt.riming_input
            rhoair = pt.derived.rhoair

            # Zero when qc <= 1e-7 (the itab riming gate)
            r_dry = Scythe.ishmael_ice_cloud_riming(tables_.itab, vc.rni, 1.0e-8, vc.deltastr,
                vc.rhobar, vc.ni, ri.nc, rhoair)
            @test r_dry == (rimesum=0.0, qi_qc_nrm=0.0, qi_qc_nrd=0.0)

            # Zero when ni (the "ni" moment, standing in for qi=0 upstream) is zero
            r_noice = Scythe.ishmael_ice_cloud_riming(tables_.itab, vc.rni, ri.qc, vc.deltastr,
                vc.rhobar, 0.0, ri.nc, rhoair)
            @test r_noice.rimesum == 0.0

            # rimesum never negative
            for pt2 in ISHMAEL_REF_POINTS
                vc2 = pt2.var_check
                ri2 = pt2.riming_input
                r2 = Scythe.ishmael_ice_cloud_riming(tables_.itab, vc2.rni, ri2.qc, vc2.deltastr,
                    vc2.rhobar, vc2.ni, ri2.nc, pt2.derived.rhoair)
                @test r2.rimesum >= 0.0
            end
        end

        # ──────────────────────────────────────────────
        # 2. ishmael_ice_rain_riming
        # ──────────────────────────────────────────────
        @testset "ishmael_ice_rain_riming: Fortran cross-check" begin
            for pt in ISHMAEL_REF_POINTS
                vc = pt.var_check
                ri = pt.riming_input
                ref = pt.itabr_riming
                temp = pt.input.temp
                r = Scythe.ishmael_ice_rain_riming(tables_.itabr, vc.rni, ri.qr, ri.nr, vc.deltastr,
                    vc.rhobar, vc.ni, pt.derived.rhoair, temp, pt.input.qidum)
                @test r.rimesumr ≈ ref.rimesumr rtol=1.0e-4 atol=1.0e-30
                @test r.qi_qr_nrm ≈ ref.qi_qr_nrm rtol=1.0e-4 atol=1.0e-30
                @test r.qi_qr_nrd ≈ ref.qi_qr_nrd rtol=1.0e-4 atol=1.0e-30
                @test r.qi_qr_nrn ≈ ref.qi_qr_nrn rtol=1.0e-4 atol=1.0e-30
                @test r.numrateri ≈ ref.numrateri rtol=1.0e-4 atol=1.0e-30
                @test r.rainrateri ≈ ref.rainrateri rtol=1.0e-4 atol=1.0e-30
                @test r.icerateri ≈ ref.icerateri rtol=1.0e-4 atol=1.0e-30
                @test r.dQRfzri ≈ ref.dQRfzri rtol=1.0e-4 atol=1.0e-30
                @test r.dQIfzri ≈ ref.dQIfzri rtol=1.0e-4 atol=1.0e-30
                @test r.dNfzri ≈ ref.dNfzri rtol=1.0e-4 atol=1.0e-30
                @test r.dQImltri ≈ ref.dQImltri rtol=1.0e-4 atol=1.0e-30
                @test r.dNmltri ≈ ref.dNmltri rtol=1.0e-4 atol=1.0e-30
            end
        end

        @testset "ishmael_ice_rain_riming: physical checks" begin
            pt9 = ISHMAEL_REF_POINTS[9]   # freeze branch exercised here (qi, qr both > 1e-4, T<=T0)
            vc = pt9.var_check
            ri = pt9.riming_input
            rhoair = pt9.derived.rhoair
            temp = pt9.input.temp
            @test temp <= ISHMAEL_REF_T0

            # Freezing active when qr>0.1e-3 and qi>0.1e-3 (matches the reference point)
            r_freeze = Scythe.ishmael_ice_rain_riming(tables_.itabr, vc.rni, ri.qr, ri.nr, vc.deltastr,
                vc.rhobar, vc.ni, rhoair, temp, pt9.input.qidum)
            @test r_freeze.dQIfzri > 0.0
            @test r_freeze.dQImltri == 0.0   # melt branch inactive at T<=T0

            # Freezing INACTIVE when qi drops below the 0.1 g/kg threshold, even
            # though rainrateri/icerateri themselves are still computed/positive.
            r_lowqi = Scythe.ishmael_ice_rain_riming(tables_.itabr, vc.rni, ri.qr, ri.nr, vc.deltastr,
                vc.rhobar, vc.ni, rhoair, temp, 1.0e-5)
            @test r_lowqi.dQIfzri == 0.0
            @test r_lowqi.dQRfzri == 0.0
            @test r_lowqi.dNfzri == 0.0

            # Freezing INACTIVE when qr drops below the 0.1 g/kg threshold
            r_lowqr = Scythe.ishmael_ice_rain_riming(tables_.itabr, vc.rni, 1.0e-5, ri.nr, vc.deltastr,
                vc.rhobar, vc.ni, rhoair, temp, pt9.input.qidum)
            @test r_lowqr.dQIfzri == 0.0

            # Melting active (freezing inactive) above T0
            pt7 = ISHMAEL_REF_POINTS[7]
            @test pt7.input.temp > ISHMAEL_REF_T0
            vc7 = pt7.var_check
            ri7 = pt7.riming_input
            r_warm = Scythe.ishmael_ice_rain_riming(tables_.itabr, vc7.rni, ri7.qr, ri7.nr, vc7.deltastr,
                vc7.rhobar, vc7.ni, pt7.derived.rhoair, pt7.input.temp, pt7.input.qidum)
            @test r_warm.dQImltri > 0.0
            @test r_warm.dQRfzri == 0.0
            @test r_warm.dQIfzri == 0.0
            @test r_warm.dNfzri == 0.0

            # Zero when qr <= QSMALL (outer gate)
            r_dry = Scythe.ishmael_ice_rain_riming(tables_.itabr, vc.rni, 0.0, 0.0, vc.deltastr,
                vc.rhobar, vc.ni, rhoair, temp, pt9.input.qidum)
            @test r_dry == (rimesumr=0.0, qi_qr_nrm=0.0, qi_qr_nrd=0.0, qi_qr_nrn=0.0,
                numrateri=0.0, rainrateri=0.0, icerateri=0.0, dQRfzri=0.0, dQIfzri=0.0, dNfzri=0.0,
                dQImltri=0.0, dNmltri=0.0)
        end

        # ──────────────────────────────────────────────
        # 3. ishmael_wet_growth_check
        # ──────────────────────────────────────────────
        @testset "ishmael_wet_growth_check: Fortran cross-check" begin
            for pt in ISHMAEL_REF_POINTS
                vc = pt.var_check
                d = pt.derived
                vg = pt.vaporgrow
                wg = pt.wet_growth
                temp = pt.input.temp
                xxls = 3.15e6 - 2370.0 * temp + 0.3337e6
                xxlv = 3.1484e6 - 2370.0 * temp
                xxlf = xxls - xxlv
                dry = Scythe.ishmael_wet_growth_check(NU_, temp, d.rhoair, xxlv, xxlf, d.qv, d.dv,
                    d.kt, d.qvs, vg.fvdum, vg.fhdum, wg.rimetotal, vc.rni, vc.ni)
                @test dry == wg.dry_growth
            end
        end

        @testset "ishmael_wet_growth_check: physical checks" begin
            pt = ISHMAEL_REF_POINTS[6]
            vc = pt.var_check
            d = pt.derived
            vg = pt.vaporgrow
            temp = pt.input.temp
            xxls = 3.15e6 - 2370.0 * temp + 0.3337e6
            xxlv = 3.1484e6 - 2370.0 * temp
            xxlf = xxls - xxlv

            # Very high rime rate -> wet growth (dry_growth = false)
            dry_heavy = Scythe.ishmael_wet_growth_check(NU_, temp, d.rhoair, xxlv, xxlf, d.qv, d.dv,
                d.kt, d.qvs, vg.fvdum, vg.fhdum, 1.0, vc.rni, vc.ni)
            @test dry_heavy == false

            # Negligible rime rate -> dry growth (dry_growth = true)
            dry_light = Scythe.ishmael_wet_growth_check(NU_, temp, d.rhoair, xxlv, xxlf, d.qv, d.dv,
                d.kt, d.qvs, vg.fvdum, vg.fhdum, 1.0e-30, vc.rni, vc.ni)
            @test dry_light == true
        end

        # ──────────────────────────────────────────────
        # 4. ishmael_macklin_rimec1 / ishmael_macklin_density / ishmael_riming_growth
        # ──────────────────────────────────────────────
        @testset "ishmael_macklin_density: range check" begin
            for temp in (200.0, 230.0, 240.0, 250.0, 258.0, 263.0, 268.0, 270.0, 273.15, 275.0, 280.0)
                rimec1 = Scythe.ishmael_macklin_rimec1(temp)
                @test 0.0 < rimec1 <= 0.012
                for ratio in (0.0, 1.0, 10.0, 100.0, 1000.0), dry in (true, false)
                    g = Scythe.ishmael_macklin_density(rimec1, ratio, 1.0, temp, dry)
                    @test 50.0 <= g <= 900.0
                end
            end
        end

        @testset "ishmael_riming_growth: Fortran cross-check" begin
            for pt in ISHMAEL_REF_POINTS
                vc = pt.var_check
                d = pt.derived
                ri = pt.riming_input
                itab = pt.itab_riming
                itabr = pt.itabr_riming
                wg = pt.wet_growth
                ref = pt.riming_growth
                temp = pt.input.temp

                r = Scythe.ishmael_riming_growth(2.0, vc.rni, vc.deltastr, vc.rhobar, vc.ni,
                    vc.ani, vc.cni, temp, ri.qc, ri.nc, itab.qi_qc_nrm, itab.qi_qc_nrd, itab.rimesum,
                    ri.qr, ri.nr, itabr.qi_qr_nrm, itabr.qi_qr_nrd, itabr.rimesumr, d.rhoair,
                    wg.dry_growth, NU_, AO_, GAMMNU_, I_GAMMNU_, FOURTHIRDSPI_)

                # prdr/ardr/crdr are differences of two nearly-equal O(nidum*rbdum*vi)
                # quantities (iwcfr-iwci) -- single-precision Fortran cancellation
                # noise is amplified well past the table-lookup values' own 1e-4
                # rtol (worst observed: ~0.6%, see the Stage S6b report). gdenavg/
                # gdenavgr/rhorimeout are NOT difference-based and match tightly.
                @test r.prdr ≈ ref.prdr rtol=1.0e-2 atol=1.0e-25
                @test r.ardr ≈ ref.ardr rtol=1.0e-2 atol=1.0e-25
                @test r.crdr ≈ ref.crdr rtol=1.0e-2 atol=1.0e-25
                @test r.rhorimeout ≈ ref.rhorimeout rtol=1.0e-4
                @test r.gdenavg ≈ ref.gdenavg rtol=1.0e-4
                @test r.gdenavgr ≈ ref.gdenavgr rtol=1.0e-4
                @test r.dry_growth == wg.dry_growth
            end
        end

        @testset "ishmael_riming_growth: physical checks" begin
            pt = ISHMAEL_REF_POINTS[1]
            vc = pt.var_check
            d = pt.derived
            temp = pt.input.temp

            # No riming input (qc=qr=0, rimesum=rimesumr=0) -> prdr=ardr=crdr=0
            r_none = Scythe.ishmael_riming_growth(2.0, vc.rni, vc.deltastr, vc.rhobar, vc.ni,
                vc.ani, vc.cni, temp, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, d.rhoair,
                true, NU_, AO_, GAMMNU_, I_GAMMNU_, FOURTHIRDSPI_)
            @test r_none.prdr == 0.0
            @test r_none.ardr == 0.0
            @test r_none.crdr == 0.0

            # T > T0 forces dry_growth = false regardless of the wet_growth_check input
            pt7 = ISHMAEL_REF_POINTS[7]
            vc7 = pt7.var_check
            d7 = pt7.derived
            ri7 = pt7.riming_input
            itab7 = pt7.itab_riming
            itabr7 = pt7.itabr_riming
            r_warm = Scythe.ishmael_riming_growth(2.0, vc7.rni, vc7.deltastr, vc7.rhobar, vc7.ni,
                vc7.ani, vc7.cni, pt7.input.temp, ri7.qc, ri7.nc, itab7.qi_qc_nrm, itab7.qi_qc_nrd,
                itab7.rimesum, ri7.qr, ri7.nr, itabr7.qi_qr_nrm, itabr7.qi_qr_nrd, itabr7.rimesumr,
                d7.rhoair, true, NU_, AO_, GAMMNU_, I_GAMMNU_, FOURTHIRDSPI_)   # dry_growth_pre=true
            @test r_warm.dry_growth == false
            @test r_warm.ardr == 0.0   # wet-growth branch: no axis growth
            @test r_warm.crdr == 0.0

            # Rates never negative (clamped in the Fortran)
            for pt2 in ISHMAEL_REF_POINTS
                vc2 = pt2.var_check
                d2 = pt2.derived
                ri2 = pt2.riming_input
                itab2 = pt2.itab_riming
                itabr2 = pt2.itabr_riming
                wg2 = pt2.wet_growth
                r2 = Scythe.ishmael_riming_growth(2.0, vc2.rni, vc2.deltastr, vc2.rhobar, vc2.ni,
                    vc2.ani, vc2.cni, pt2.input.temp, ri2.qc, ri2.nc, itab2.qi_qc_nrm, itab2.qi_qc_nrd,
                    itab2.rimesum, ri2.qr, ri2.nr, itabr2.qi_qr_nrm, itabr2.qi_qr_nrd, itabr2.rimesumr,
                    d2.rhoair, wg2.dry_growth, NU_, AO_, GAMMNU_, I_GAMMNU_, FOURTHIRDSPI_)
                @test r2.prdr >= 0.0
                @test r2.ardr >= 0.0
                @test r2.crdr >= 0.0
            end
        end

        # ──────────────────────────────────────────────
        # 5. Aggregation: ishmael_agg_table_index / ishmael_agg_efffact /
        #    ishmael_col1 / ishmael_aggregation
        # ──────────────────────────────────────────────
        @testset "ishmael_agg_efffact: range + col1 base efficiency formula" begin
            for rhoeffmax in (100.0, 400.0, 500.0, 900.0), phieffmax in (0.01, 0.03, 0.1, 0.5, 2.0)
                e = Scythe.ishmael_agg_efffact(rhoeffmax, phieffmax)
                @test 0.0 <= e <= 1.0
            end

            # col1's own base efficiency formula (before the DGZ 1.4x special
            # case): eff = min(0.2, 10^(0.035*Tc-0.7)) is in [0, 0.2] for any Tc.
            for tempC in (-60.0, -40.0, -20.0, -14.0, -5.0, 0.0, 10.0)
                eff_base = min(0.2, 10.0^(0.035 * tempC - 0.7))
                @test 0.0 <= eff_base <= 0.2
            end
        end

        @testset "ishmael_aggregation: Fortran cross-check" begin
            for pt in ISHMAEL_REF_AGGREGATION
                r = Scythe.ishmael_aggregation(pt.dt, pt.rhoair, pt.temp, pt.q1, pt.n1, pt.d1,
                    pt.q2, pt.n2, pt.d2, pt.q3, pt.n3, pt.d3, pt.rho1, pt.rho2, pt.phi1, pt.phi2,
                    tables_.coltab, tables_.coltabn)
                @test r.qagg1 ≈ pt.qagg1 rtol=1.0e-3 atol=1.0e-30
                @test r.qagg2 ≈ pt.qagg2 rtol=1.0e-3 atol=1.0e-30
                @test r.qagg3 ≈ pt.qagg3 rtol=1.0e-3 atol=1.0e-30
                @test r.nagg1 ≈ pt.nagg1 rtol=1.0e-3 atol=1.0e-30
                @test r.nagg2 ≈ pt.nagg2 rtol=1.0e-3 atol=1.0e-30
                @test r.nagg3 ≈ pt.nagg3 rtol=1.0e-3 atol=1.0e-30
                @test r.dnew3 ≈ pt.ddum3 rtol=1.0e-3 atol=1.0e-30
            end
        end

        @testset "ishmael_aggregation: physical checks" begin
            for pt in ISHMAEL_REF_AGGREGATION
                r = Scythe.ishmael_aggregation(pt.dt, pt.rhoair, pt.temp, pt.q1, pt.n1, pt.d1,
                    pt.q2, pt.n2, pt.d2, pt.q3, pt.n3, pt.d3, pt.rho1, pt.rho2, pt.phi1, pt.phi2,
                    tables_.coltab, tables_.coltabn)

                # Mass conservation: qagg sums to zero across the 3 live species
                # (qagg1 and qagg2 are losses, qagg3 is the matching gain).
                @test r.qagg1 + r.qagg2 + r.qagg3 ≈ 0.0 atol=1.0e-16

                # Number strictly moves OUT of planar/columnar (never a gain from
                # aggregation): nagg1, nagg2 <= 0 always.
                @test r.nagg1 <= 0.0
                @test r.nagg2 <= 0.0

                @test r.qagg1 <= 0.0
                @test r.qagg2 <= 0.0
                @test r.qagg3 >= 0.0
                @test r.dnew3 > 0.0
            end

            # A "normal" point (not dominated by pre-existing aggregate self-
            # collection) should show nagg3 > 0: number actually accumulates in
            # the aggregate category.
            pt1 = ISHMAEL_REF_AGGREGATION[1]
            r1 = Scythe.ishmael_aggregation(pt1.dt, pt1.rhoair, pt1.temp, pt1.q1, pt1.n1, pt1.d1,
                pt1.q2, pt1.n2, pt1.d2, pt1.q3, pt1.n3, pt1.d3, pt1.rho1, pt1.rho2, pt1.phi1,
                pt1.phi2, tables_.coltab, tables_.coltabn)
            @test r1.nagg3 > 0.0

            # No collection when both source categories are empty
            r_empty = Scythe.ishmael_aggregation(2.0, 0.7, 258.0, 0.0, 0.0, 1.0e-6, 0.0, 0.0, 1.0e-6,
                0.0, 0.0, 1.0e-6, 300.0, 300.0, 0.5, 2.0, tables_.coltab, tables_.coltabn)
            @test r_empty.qagg1 == 0.0
            @test r_empty.qagg2 == 0.0
            @test r_empty.qagg3 == 0.0
        end

        @testset "ishmael_aggregation: the caller's realization factors" begin
            # Stage 1. The Fortran's `ratioagg` over-depletion reconciliation is not ported,
            # but its HOOK is: one factor per DONOR SPECIES, applied to that species' three
            # pairs right after the seven `col1` calls. The host forms those factors from
            # each species' total sink conductance, so the three-pair sum ends up bounded
            # against `q1`/`q2` by an exponential realization instead of a dt-shaped ratio.
            aggr(pt; kw...) = Scythe.ishmael_aggregation(pt.dt, pt.rhoair, pt.temp,
                pt.q1, pt.n1, pt.d1, pt.q2, pt.n2, pt.d2, pt.q3, pt.n3, pt.d3,
                pt.rho1, pt.rho2, pt.phi1, pt.phi2, tables_.coltab, tables_.coltabn; kw...)

            # The DEFAULT is the Fortran unchanged -- to the BIT, not to a tolerance. That
            # is what lets the cross-check above stand untouched: `1.0*x === x`.
            for pt in ISHMAEL_REF_AGGREGATION
                r0 = aggr(pt)
                r1 = aggr(pt; f_agg1 = 1.0, f_agg2 = 1.0, reservoir_caps = true)
                for nm in (:qagg1, :qagg2, :qagg3, :nagg1, :nagg2, :nagg3, :dnew3)
                    @test getproperty(r1, nm) === getproperty(r0, nm)
                end
            end

            # Each factor is LINEAR in its own donor's transfer, mass AND number together
            # (a pair's `(colamt, deltan)` is one collision count, so scaling both is what
            # keeps the surviving per-particle mass invariant), and the exchange still
            # closes: species 3 gains exactly what 1 and 2 lose.
            for pt in ISHMAEL_REF_AGGREGATION
                r0 = aggr(pt)
                rf = aggr(pt; f_agg1 = 0.3, f_agg2 = 0.7)
                @test rf.qagg1 ≈ 0.3 * r0.qagg1 rtol=1.0e-14 atol=1.0e-30
                @test rf.qagg2 ≈ 0.7 * r0.qagg2 rtol=1.0e-14 atol=1.0e-30
                @test rf.nagg1 ≈ 0.3 * r0.nagg1 rtol=1.0e-14 atol=1.0e-30
                @test rf.nagg2 ≈ 0.7 * r0.nagg2 rtol=1.0e-14 atol=1.0e-30
                @test rf.qagg1 + rf.qagg2 + rf.qagg3 ≈ 0.0 atol=1.0e-16
                # The recipient's gain rides the factors with the donors, and the third
                # species is never a donor to itself here.
                @test rf.qagg3 <= r0.qagg3 + 1.0e-30
            end

            # `reservoir_caps = false` removes the per-pair `min(colamt, rx)` bound, so the
            # draw can only grow (the losses are negative, hence `<=`). At the Fortran
            # reference points the caps do not bind, so this is an equality there; the
            # binding case is below.
            for pt in ISHMAEL_REF_AGGREGATION
                r0 = aggr(pt)
                rn = aggr(pt; reservoir_caps = false)
                @test rn.qagg1 <= r0.qagg1
                @test rn.qagg2 <= r0.qagg2
            end

            # `c_55` -- aggregate SELF-collection -- keeps its caps even under
            # `reservoir_caps = false`: it takes from species 3 and gives to species 3, so
            # no donor reservoir realizes it and the per-pair bound is all it has. Build a
            # point where that bound bites hard: 1e9 aggregates per kg on 1e-2 kg/kg.
            let pt = ISHMAEL_REF_AGGREGATION[1], n3b = 1.0e9, q3b = 1.0e-2
                ip55 = Scythe.ISHMAEL_IPAIR[5, 5]
                en3 = n3b * pt.rhoair
                tC = pt.temp - Scythe.ISHMAEL_T0
                ccap = Scythe.ishmael_col1(pt.dt, 1.0, tC, pt.d3, en3, q3b, pt.d3, en3,
                    pt.rhoair, tables_.coltab, tables_.coltabn, ip55, true)
                craw = Scythe.ishmael_col1(pt.dt, 1.0, tC, pt.d3, en3, q3b, pt.d3, en3,
                    pt.rhoair, tables_.coltab, tables_.coltabn, ip55, false)
                @test craw.colamt > ccap.colamt          # uncapped, this pair runs away
                @test ccap.colamt == q3b                 # capped, it takes the reservoir
                @test ccap.deltan <= en3 * (1.0 + 1.0e-12)
                # ...and through the public entry point with the caps switched OFF, the
                # aggregate number sink is STILL the capped one: `nagg3 >= -n3/2`, the
                # `-0.5*c_55.deltan` term at its bound. Uncapped it would be ~2e7 times that.
                r = Scythe.ishmael_aggregation(pt.dt, pt.rhoair, pt.temp,
                    pt.q1, pt.n1, pt.d1, pt.q2, pt.n2, pt.d2, q3b, n3b, pt.d3,
                    pt.rho1, pt.rho2, pt.phi1, pt.phi2, tables_.coltab, tables_.coltabn;
                    reservoir_caps = false)
                @test r.nagg3 >= -0.5 * n3b * (1.0 + 1.0e-12)
                @test r.nagg3 < 0.0                      # the cap is what it is sitting on
            end
        end

        # ──────────────────────────────────────────────
        # 6. Pure diagnostic caps: ishmael_ni_cap / ishmael_agg_size_cap
        # ──────────────────────────────────────────────
        @testset "ishmael_ni_cap" begin
            @test Scythe.ishmael_ni_cap(2.0e6, 1.0) == 1.0e6
            @test Scythe.ishmael_ni_cap(5.0e5, 1.0) == 5.0e5   # below the cap: unchanged
            @test Scythe.ishmael_ni_cap(3.0e6, 2.0) == 5.0e5   # cap = 1e6/rhoair
        end

        @testset "ishmael_agg_size_cap" begin
            # Cap binds: ani=1mm > 0.5mm -> clamped, cni/ni re-derived
            r = Scythe.ishmael_agg_size_cap(1.0e-3, 5.0e-4, 100.0, 1.0, 0.1e-6, 1.0e-3, 500.0,
                4.0, 6.0)
            @test r.ani == 0.5e-3
            @test r.cni > 0.0
            @test r.ni > 0.0
            @test isfinite(r.ni)

            # Cap does not bind: pass-through unchanged
            r2 = Scythe.ishmael_agg_size_cap(1.0e-4, 5.0e-5, 100.0, 1.0, 0.1e-6, 1.0e-3, 500.0,
                4.0, 6.0)
            @test r2 == (ani=1.0e-4, cni=5.0e-5, ni=100.0)

            # Boundary: exactly 0.5mm does NOT trigger the cap (strict >)
            r3 = Scythe.ishmael_agg_size_cap(0.5e-3, 3.0e-4, 100.0, 1.0, 0.1e-6, 1.0e-3, 500.0,
                4.0, 6.0)
            @test r3.ani == 0.5e-3
            @test r3 == (ani=0.5e-3, cni=3.0e-4, ni=100.0)   # unchanged: pass-through branch
        end

        # ──────────────────────────────────────────────
        # 7. Zero-allocation hot-path checks
        # ──────────────────────────────────────────────
        @testset "Zero allocations (Stage S6b)" begin
            vc = ISHMAEL_REF_POINTS[1].var_check
            d = ISHMAEL_REF_POINTS[1].derived
            ri = ISHMAEL_REF_POINTS[1].riming_input
            temp = ISHMAEL_REF_POINTS[1].input.temp

            Scythe.ishmael_ice_cloud_riming(tables_.itab, vc.rni, ri.qc, vc.deltastr, vc.rhobar,
                vc.ni, ri.nc, d.rhoair)
            @test @allocated(Scythe.ishmael_ice_cloud_riming(tables_.itab, vc.rni, ri.qc,
                vc.deltastr, vc.rhobar, vc.ni, ri.nc, d.rhoair)) == 0

            Scythe.ishmael_ice_rain_riming(tables_.itabr, vc.rni, ri.qr, ri.nr, vc.deltastr,
                vc.rhobar, vc.ni, d.rhoair, temp, ISHMAEL_REF_POINTS[1].input.qidum)
            @test @allocated(Scythe.ishmael_ice_rain_riming(tables_.itabr, vc.rni, ri.qr, ri.nr,
                vc.deltastr, vc.rhobar, vc.ni, d.rhoair, temp,
                ISHMAEL_REF_POINTS[1].input.qidum)) == 0

            Scythe.ishmael_wet_growth_check(NU_, temp, d.rhoair, 2.5e6, 3.0e5, d.qv, d.dv, d.kt,
                d.qvs, 1.1, 1.1, 1.0e-8, vc.rni, vc.ni)
            @test @allocated(Scythe.ishmael_wet_growth_check(NU_, temp, d.rhoair, 2.5e6, 3.0e5,
                d.qv, d.dv, d.kt, d.qvs, 1.1, 1.1, 1.0e-8, vc.rni, vc.ni)) == 0

            itab1 = ISHMAEL_REF_POINTS[1].itab_riming
            itabr1 = ISHMAEL_REF_POINTS[1].itabr_riming
            Scythe.ishmael_riming_growth(2.0, vc.rni, vc.deltastr, vc.rhobar, vc.ni, vc.ani, vc.cni,
                temp, ri.qc, ri.nc, itab1.qi_qc_nrm, itab1.qi_qc_nrd, itab1.rimesum, ri.qr, ri.nr,
                itabr1.qi_qr_nrm, itabr1.qi_qr_nrd, itabr1.rimesumr, d.rhoair, true, NU_, AO_,
                GAMMNU_, I_GAMMNU_, FOURTHIRDSPI_)
            @test @allocated(Scythe.ishmael_riming_growth(2.0, vc.rni, vc.deltastr, vc.rhobar, vc.ni,
                vc.ani, vc.cni, temp, ri.qc, ri.nc, itab1.qi_qc_nrm, itab1.qi_qc_nrd, itab1.rimesum,
                ri.qr, ri.nr, itabr1.qi_qr_nrm, itabr1.qi_qr_nrd, itabr1.rimesumr, d.rhoair, true,
                NU_, AO_, GAMMNU_, I_GAMMNU_, FOURTHIRDSPI_)) == 0

            pt1 = ISHMAEL_REF_AGGREGATION[1]
            Scythe.ishmael_aggregation(pt1.dt, pt1.rhoair, pt1.temp, pt1.q1, pt1.n1, pt1.d1, pt1.q2,
                pt1.n2, pt1.d2, pt1.q3, pt1.n3, pt1.d3, pt1.rho1, pt1.rho2, pt1.phi1, pt1.phi2,
                tables_.coltab, tables_.coltabn)
            @test @allocated(Scythe.ishmael_aggregation(pt1.dt, pt1.rhoair, pt1.temp, pt1.q1, pt1.n1,
                pt1.d1, pt1.q2, pt1.n2, pt1.d2, pt1.q3, pt1.n3, pt1.d3, pt1.rho1, pt1.rho2, pt1.phi1,
                pt1.phi2, tables_.coltab, tables_.coltabn)) == 0

            Scythe.ishmael_aggregation(pt1.dt, pt1.rhoair, pt1.temp, pt1.q1, pt1.n1, pt1.d1, pt1.q2,
                pt1.n2, pt1.d2, pt1.q3, pt1.n3, pt1.d3, pt1.rho1, pt1.rho2, pt1.phi1, pt1.phi2,
                tables_.coltab, tables_.coltabn; f_agg1 = 0.3, f_agg2 = 0.7,
                reservoir_caps = false)
            @test @allocated(Scythe.ishmael_aggregation(pt1.dt, pt1.rhoair, pt1.temp, pt1.q1, pt1.n1,
                pt1.d1, pt1.q2, pt1.n2, pt1.d2, pt1.q3, pt1.n3, pt1.d3, pt1.rho1, pt1.rho2, pt1.phi1,
                pt1.phi2, tables_.coltab, tables_.coltabn; f_agg1 = 0.3, f_agg2 = 0.7,
                reservoir_caps = false)) == 0

            Scythe.ishmael_ni_cap(1.0e6, 1.0)
            @test @allocated(Scythe.ishmael_ni_cap(1.0e6, 1.0)) == 0

            Scythe.ishmael_agg_size_cap(1.0e-3, 5.0e-4, 100.0, 1.0, 0.1e-6, 1.0e-3, 500.0, 4.0, 6.0)
            @test @allocated(Scythe.ishmael_agg_size_cap(1.0e-3, 5.0e-4, 100.0, 1.0, 0.1e-6, 1.0e-3,
                500.0, 4.0, 6.0)) == 0
        end
    end
end
