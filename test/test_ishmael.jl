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
