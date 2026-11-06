# test/reference/ishmael_fortran_refs.jl
#
# Literal transcription of tools/ishmael_fortran_driver/ref_driver_output.txt
# (built and run 2026-08-12, gfortran 15 at /opt/homebrew/bin/gfortran,
# native single precision, -fconvert=big-endian, against a public-list-
# patched COPY of CM1's module_mp_jensen_ishmael.F -- see
# tools/ishmael_fortran_driver/README.md for the full build/run story).
#
# NO runtime Fortran dependency: these are plain Float64 literals copied by
# hand from ref_driver_output.txt (which is itself checked in for the
# record). Every value here is REAL*4 precision as printed by gfortran
# (~7 significant digits) -- Julia/Fortran comparisons in test/test_ishmael.jl
# use rtol=1.0e-5 to respect that, per the same convention already
# established in test/test_ishmael_tables.jl.
#
# 10 state points (ISHMAEL_REF_POINTS) run through the var_check ->
# capacitance_gamma -> vaporgrow chain, exactly as mp_jensen_ishmael calls
# them (lines 1044-1266 of module_mp_jensen_ishmael.F). Each point's
# `input` NamedTuple is what was fed to var_check (deltastr/ani/cni/rbdum
# BEFORE var_check's clamps); `derived` are the thermodynamic quantities
# computed the same way mp_jensen_ishmael does (lines 975-993, 825-826);
# `var_check`, `capgam`, `vaporgrow` are the routines' outputs at that
# point, in Fortran call order.
#
# Plus: ISHMAEL_REF_XM_FALLBACK (standalone vaporgrow call with ani=2mm,
# bypassing var_check, to isolate the xm>1e8 Mitchell-Heymsfield fallback
# branch -- confirmed via a diagnostic-only xm print in ref_driver.f90:
# xm = 6.13366784e8), ISHMAEL_REF_IGR (11-point get_igr boundary sweep),
# and ISHMAEL_REF_LUT_ITAB / ISHMAEL_REF_LUT_ITABR (access_lookup_table).
#
# Stage S6b addition (built and run 2026-08-12, same driver/toolchain):
# each of the 10 ISHMAEL_REF_POINTS tuples above gained four more fields --
# `riming_input` (qc/nc/qr/nr fed to the itab/itabr blocks), `itab_riming`
# (ice-cloud collection, lines 1053-1106), `itabr_riming` (ice-rain
# collection + T<=T0 freeze-ri / T>T0 melt-ri branches, lines 1111-1214),
# `wet_growth` (wet_growth_check, lines 3537-3556), and `riming_growth`
# (Macklin rime-density + axis growth, lines 1291-1507) -- all computed
# inline in ref_driver.f90's per-point loop from that SAME point's
# var_check-effective (dsdum, ani, cni, rni, rbdum, nidum) state, using
# `gamma()` directly rather than the Fortran's own gamma_tab lookup (same
# substitution the Stage S6a driver already uses for iwci/vaporgrow -- see
# ref_driver.f90's own note at its `iwci = nidum*rbdum*fourthirdspi*rni**3*
# (gamma(NU+2.+dsdum)/gammnu)` line). Plus a new ISHMAEL_REF_AGGREGATION
# (6 state points into `aggregation`+`col1`, lines 3988-4541).

const ISHMAEL_REF_T0 = 273.15

const ISHMAEL_REF_POINTS = (
    (
        label = "columnar T=-6C moderate",
        input = (temp=267.149994, pres=85000.0, qidum=9.99999975e-6, dsdum0=1.20000005,
                 ani0=4.99999987e-5, cni0=1.50000007e-4, rbdum0=400.0, nidum=2.0e4),
        derived = (qv=2.84318137e-3, qvi=2.70779198e-3, qvs=2.87205772e-3, sui=4.99999523e-2,
                   sup=-1.00542307e-2, igr=2.32422996, mu=1.68727183e-5, dv=2.55384330e-5,
                   kt=2.33959388e-2, nsch=5.96260250e-1, npr=6.50863047e-4, rhoair=1.10803878),
        var_check = (deltastr=1.20000005, ani=1.69670548e-5, cni=4.73726686e-5, rni=2.38916891e-5,
                     rhobar=50.0, ni=2.0e4, ai=2.72753764e-10, ci=7.61538999e-10,
                     alphstr=25.1188831, alphv=105.217735, betam=3.20000005),
        capgam = 1.25456994e-4,
        vaporgrow = (vtbarb=3.35027501e-2, vtbarbm=8.66201594e-2, vtbarbz=1.58969983e-1,
                     fvdum=1.04550076, fhdum=1.00037217, anf=1.69726572e-5, cnf=4.74046974e-5,
                     rnf=2.39030614e-5, iwcf=1.01394162e-5, rdout=4.87974243e2,
                     dsdumout=1.20007205, rbdum=50.0),
        riming_input = (qc=2.00000009e-3, nc=2.0e8, qr=1.00000005e-3, nr=1.0e6),
        itab_riming = (rimesum=4.48879511e-8, qi_qc_nrm=9.14028761e-21, qi_qc_nrd=1.42700304e-20),
        itabr_riming = (rimesumr=1.14305607e-10, qi_qr_nrm=4.65508491e-21, qi_qr_nrd=6.40468904e-21,
                        qi_qr_nrn=1.52258706e-10, nr_final=1.0e6,
                        numrateri=3.56017578e2, rainrateri=3.46009324e-6, icerateri=2.07203954e-7,
                        dQRfzri=0.0, dQIfzri=0.0, dNfzri=0.0, dQImltri=0.0, dNmltri=0.0),
        wet_growth = (rimetotal=4.50022561e-8, dry_growth=true),
        riming_growth = (prdr=3.66969921e-8, ardr=4.87596183e-13, crdr=6.80693674e-13,
                         rhorimeout=1.13633415e2, gdenavg=1.13637527e2, gdenavgr=1.12018501e2),
    ),
    (
        label = "planar T=-12C moderate",
        input = (temp=261.149994, pres=80000.0, qidum=9.99999975e-6, dsdum0=0.75,
                 ani0=1.50000007e-4, cni0=4.99999987e-5, rbdum0=300.0, nidum=2.0e4),
        derived = (qv=1.74367952e-3, qvi=1.69289275e-3, qvs=1.90431077e-3, sui=2.99999714e-2,
                   sup=-8.43513608e-2, igr=0.453916997, mu=1.65642086e-5, dv=2.60415763e-5,
                   kt=2.29688771e-2, nsch=5.96227884e-1, npr=6.75989198e-4, rhoair=1.06682003),
        var_check = (deltastr=0.75, ani=5.33649181e-5, cni=1.11030395e-5, rni=3.16216465e-5,
                     rhobar=50.0, ni=2.0e4, ai=6.32387864e-10, ci=1.31573849e-10,
                     alphstr=1.77827943e-2, alphv=7.44883940e-2, betam=2.75),
        capgam = 1.48235355e-4,
        vaporgrow = (vtbarb=5.48652560e-2, vtbarbm=1.05660081e-1, vtbarbz=1.62697822e-1,
                     fvdum=1.21043015, fhdum=1.00183940, anf=5.33771199e-5, cnf=1.11043182e-5,
                     rnf=3.16275582e-5, iwcf=1.00785310e-5, rdout=7.00000000e2,
                     dsdumout=0.749988854, rbdum=50.0),
        riming_input = (qc=1.00000005e-3, nc=1.0e8, qr=5.00000024e-4, nr=5.0e5),
        itab_riming = (rimesum=3.38150508e-10, qi_qc_nrm=1.48558589e-22, qi_qc_nrd=3.24513143e-22),
        itabr_riming = (rimesumr=1.48860681e-11, qi_qr_nrm=1.30796965e-21, qi_qr_nrd=1.68579823e-21,
                        qi_qr_nrn=2.63609724e-11, nr_final=5.0e5,
                        numrateri=2.98149048e2, rainrateri=2.60145725e-6, icerateri=1.65812438e-7,
                        dQRfzri=0.0, dQIfzri=0.0, dNfzri=0.0, dQImltri=0.0, dNmltri=0.0),
        wet_growth = (rimetotal=3.53036572e-10, dry_growth=true),
        riming_growth = (prdr=3.12025517e-10, ardr=9.47966325e-15, crdr=3.94465446e-15,
                         rhorimeout=1.10234665e2, gdenavg=1.10414688e2, gdenavgr=1.06145195e2),
    ),
    (
        label = "density-clamp-high (>RHOI)",
        input = (temp=267.149994, pres=85000.0, qidum=4.99999987e-5, dsdum0=1.0,
                 ani0=2.99999992e-5, cni0=2.99999992e-5, rbdum0=950.0, nidum=500.0),
        derived = (qv=2.76194769e-3, qvi=2.70779198e-3, qvs=2.87205772e-3, sui=1.99999809e-2,
                   sup=-3.83383632e-2, igr=2.32422996, mu=1.68727183e-5, dv=2.55384330e-5,
                   kt=2.33959388e-2, nsch=5.96260250e-1, npr=6.50863047e-4, rhoair=1.10803878),
        var_check = (deltastr=1.0, ani=6.00224848e-5, cni=6.00224848e-5, rni=6.00224848e-5,
                     rhobar=920.0, ni=500.0, ai=1.08121456e-10, ci=1.08121456e-10,
                     alphstr=1.0, alphv=4.18879032, betam=3.0),
        capgam = 2.40089954e-4,
        vaporgrow = (vtbarb=1.60955811, vtbarbm=2.63640046, vtbarbz=3.60606527,
                     fvdum=2.91656542, fhdum=1.06176782, anf=6.00240564e-5, cnf=6.00253334e-5,
                     rnf=6.00247513e-5, iwcf=5.00042661e-5, rdout=7.00000000e2,
                     dsdumout=1.00000525, rbdum=920.0),
        # qc <= 1.e-7: itab riming gate is OFF (a coarser gate than QSMALL).
        riming_input = (qc=9.99999994e-9, nc=1.0e8, qr=0.0, nr=0.0),
        itab_riming = (rimesum=0.0, qi_qc_nrm=0.0, qi_qc_nrd=0.0),
        itabr_riming = (rimesumr=0.0, qi_qr_nrm=0.0, qi_qr_nrd=0.0,
                        qi_qr_nrn=0.0, nr_final=0.0,
                        numrateri=0.0, rainrateri=0.0, icerateri=0.0,
                        dQRfzri=0.0, dQIfzri=0.0, dNfzri=0.0, dQImltri=0.0, dNmltri=0.0),
        wet_growth = (rimetotal=0.0, dry_growth=true),
        riming_growth = (prdr=0.0, ardr=0.0, crdr=0.0,
                         rhorimeout=9.20000000e2, gdenavg=9.0e2, gdenavgr=9.0e2),
    ),
    (
        label = "density-clamp-low (<50)",
        input = (temp=253.149994, pres=70000.0, qidum=1.00000001e-7, dsdum0=0.899999976,
                 ani0=3.00000014e-4, cni0=2.69999990e-4, rbdum0=10.0, nidum=1000.0),
        derived = (qv=1.00997498e-3, qvi=9.18159087e-4, qvs=1.11715763e-3, sui=1.00000024e-1,
                   sup=-9.59422588e-2, igr=0.796458006, mu=1.61478620e-5, dv=2.81321118e-5,
                   kt=2.23994609e-2, nsch=5.96075594e-1, npr=7.48628052e-4, rhoair=0.962966859),
        var_check = (deltastr=0.899999976, ani=2.01238199e-5, cni=1.18396629e-5, rni=1.68624065e-5,
                     rhobar=50.0, ni=1000.0, ai=4.79468592e-12, ci=2.82090909e-12,
                     alphstr=0.199526161, alphv=0.835773230, betam=2.90000010),
        capgam = 6.65884727e-5,
        vaporgrow = (vtbarb=2.89112665e-2, vtbarbm=6.72734231e-2, vtbarbz=1.17027603e-1,
                     fvdum=1.04142416, fhdum=1.00037193, anf=2.01407838e-5, cnf=1.18479029e-5,
                     rnf=1.68756560e-5, iwcf=1.03302654e-7, rdout=7.00000000e2,
                     dsdumout=0.899983346, rbdum=50.0),
        riming_input = (qc=2.00000009e-3, nc=1.0e8, qr=9.99999975e-5, nr=1.0e5),
        itab_riming = (rimesum=1.89190243e-11, qi_qc_nrm=2.04021556e-22, qi_qc_nrd=3.10729660e-22),
        itabr_riming = (rimesumr=0.0, qi_qr_nrm=0.0, qi_qr_nrd=0.0,
                        qi_qr_nrn=0.0, nr_final=1.0e5,
                        numrateri=9.19864833e-1, rainrateri=1.02177111e-8, icerateri=9.17448825e-11,
                        dQRfzri=0.0, dQIfzri=0.0, dNfzri=0.0, dQImltri=0.0, dNmltri=0.0),
        wet_growth = (rimetotal=1.89190243e-11, dry_growth=true),
        riming_growth = (prdr=2.04057483e-11, ardr=4.48977649e-16, crdr=5.28303661e-16,
                         rhorimeout=1.04873619e2, gdenavg=1.04873619e2, gdenavgr=9.0e2),
    ),
    (
        label = "homogeneous-freezing boundary T=-35C",
        input = (temp=238.149994, pres=60000.0, qidum=1.99999999e-6, dsdum0=0.699999988,
                 ani0=1.99999995e-5, cni0=9.99999975e-6, rbdum0=500.0, nidum=5.0e5),
        derived = (qv=2.66321847e-4, qvi=2.31584214e-4, qvs=3.26752255e-4, sui=1.49999976e-1,
                   sup=-1.84942603e-1, igr=1.12233996, mu=1.53512065e-5, dv=2.93856046e-5,
                   kt=2.13318057e-2, nsch=5.95409811e-1, npr=8.20206071e-4, rhoair=0.877388418),
        var_check = (deltastr=0.699999988, ani=1.03715265e-5, cni=2.57685429e-6, rni=6.52015797e-6,
                     rhobar=50.0, ni=5.0e5, ai=1.38594261e-10, ci=3.44343894e-11,
                     alphstr=7.94328097e-3, alphv=3.32727395e-2, betam=2.70000005),
        capgam = 2.90601965e-5,
        vaporgrow = (vtbarb=3.65718897e-3, vtbarbm=8.10619909e-3, vtbarbz=1.37981111e-2,
                     fvdum=1.00246179, fhdum=1.00002348, anf=1.04072114e-5, cnf=2.58587102e-6,
                     rnf=6.54350833e-6, iwcf=2.30190017e-6, rdout=7.00000000e2,
                     dsdumout=0.700312018, rbdum=50.0),
        riming_input = (qc=5.00000024e-4, nc=1.0e8, qr=1.99999995e-4, nr=2.0e5),
        itab_riming = (rimesum=0.0, qi_qc_nrm=0.0, qi_qc_nrd=0.0),
        itabr_riming = (rimesumr=0.0, qi_qr_nrm=0.0, qi_qr_nrd=0.0,
                        qi_qr_nrn=0.0, nr_final=2.0e5,
                        numrateri=4.83209564e2, rainrateri=6.14143664e-6, icerateri=1.73495818e-9,
                        dQRfzri=0.0, dQIfzri=0.0, dNfzri=0.0, dQImltri=0.0, dNmltri=0.0),
        wet_growth = (rimetotal=0.0, dry_growth=true),
        riming_growth = (prdr=0.0, ardr=0.0, crdr=0.0,
                         rhorimeout=5.0e1, gdenavg=9.0e2, gdenavgr=9.0e2),
    ),
    (
        label = "near 0C from below, T=-0.5C",
        input = (temp=272.649994, pres=95000.0, qidum=4.99999987e-6, dsdum0=1.0,
                 ani0=9.99999975e-5, cni0=9.99999975e-5, rbdum0=200.0, nidum=1.0e4),
        derived = (qv=3.87967541e-3, qvi=3.86037352e-3, qvs=3.88022349e-3, sui=4.99999523e-3,
                   sup=-1.41263008e-4, igr=0.955273509, mu=1.71527681e-5, dv=2.37087534e-5,
                   kt=2.37874128e-2, nsch=5.96233308e-1, npr=5.94261684e-4, rhoair=1.21341479),
        var_check = (deltastr=1.0, ani=2.70962937e-5, cni=2.70962937e-5, rni=2.70962937e-5,
                     rhobar=50.0, ni=1.0e4, ai=1.98943459e-10, ci=1.98943459e-10,
                     alphstr=1.0, alphv=4.18879032, betam=3.0),
        capgam = 1.08385175e-4,
        vaporgrow = (vtbarb=6.96419179e-2, vtbarbm=1.58543691e-1, vtbarbz=2.70125717e-1,
                     fvdum=1.16293216, fhdum=1.00126135, anf=2.70973287e-5, cnf=2.70972814e-5,
                     rnf=2.70973142e-5, iwcf=5.00790611e-6, rdout=7.00000000e2,
                     dsdumout=0.999999344, rbdum=50.0),
        # Heavy riming near 0C: probes the wet_growth_check boundary.
        riming_input = (qc=3.00000003e-3, nc=5.0e7, qr=3.00000003e-3, nr=5.0e5),
        itab_riming = (rimesum=1.38657761e-8, qi_qc_nrm=1.88345665e-20, qi_qc_nrd=5.30403072e-20),
        itabr_riming = (rimesumr=1.08183330e-10, qi_qr_nrm=1.46950736e-20, qi_qr_nrd=3.87537639e-20,
                        qi_qr_nrn=2.16149487e-10, nr_final=5.0e5,
                        numrateri=3.12586853e2, rainrateri=1.93342203e-5, icerateri=1.41656713e-7,
                        dQRfzri=0.0, dQIfzri=0.0, dNfzri=0.0, dQImltri=0.0, dNmltri=0.0),
        wet_growth = (rimetotal=1.39739598e-8, dry_growth=true),
        riming_growth = (prdr=9.49152668e-9, ardr=2.78455627e-14, crdr=2.78455627e-14,
                         rhorimeout=8.22701111e2, gdenavg=8.22702454e2, gdenavgr=8.22530884e2),
    ),
    (
        label = "above 0C, T=+2C (melting/passthrough)",
        input = (temp=275.149994, pres=95000.0, qidum=4.99999987e-6, dsdum0=1.0,
                 ani0=9.99999975e-5, cni0=9.99999975e-5, rbdum0=200.0, nidum=1.0e4),
        derived = (qv=4.74398769e-3, qvi=4.65352135e-3, qvs=4.65352135e-3, sui=1.94404125e-2,
                   sup=1.94404125e-2, igr=1.0, mu=1.72792097e-5, dv=2.41036887e-5,
                   kt=2.39653550e-2, nsch=5.96204281e-1, npr=5.99645718e-4, rhoair=1.20238984),
        var_check = (deltastr=1.0, ani=2.70962937e-5, cni=2.70962937e-5, rni=2.70962937e-5,
                     rhobar=50.0, ni=1.0e4, ai=1.98943459e-10, ci=1.98943459e-10,
                     alphstr=1.0, alphv=4.18879032, betam=3.0),
        capgam = 1.08385175e-4,
        # T > T0: vaporgrow's own T>T0 passthrough branch (lines 3473-3479)
        # returns anf=ani, cnf=cni, rnf=rni, iwcf=iwci UNCHANGED -- but note
        # vtbarb/vtbarbm/vtbarbz are still computed from the fall-speed
        # block above the T-branch (lines 3236-3296 run unconditionally).
        vaporgrow = (vtbarb=6.94024488e-2, vtbarbm=1.58233017e-1, vtbarbz=2.69868404e-1,
                     fvdum=1.15991449, fhdum=1.00124383, anf=2.70962937e-5, cnf=2.70962937e-5,
                     rnf=2.70962937e-5, iwcf=4.99999442e-6, rdout=7.00000000e2,
                     dsdumout=1.0, rbdum=50.0),
        # T > T0: exercises the ice-rain MELT transfer branch (dQImltri/
        # dNmltri) and wet_growth_check's own pre-T0-override "wet" flag
        # (dry_growth=false BEFORE the riming-growth block's separate
        # `if temp>T0` override forces it false again).
        riming_input = (qc=1.00000005e-3, nc=1.0e8, qr=1.00000005e-3, nr=1.0e6),
        itab_riming = (rimesum=4.27053548e-9, qi_qc_nrm=2.95387247e-21, qi_qc_nrd=6.70704132e-21),
        itabr_riming = (rimesumr=2.45271553e-10, qi_qr_nrm=1.69651060e-20, qi_qr_nrd=4.37646560e-20,
                        qi_qr_nrn=3.01456471e-10, nr_final=1.0e6,
                        numrateri=1.41032120e2, rainrateri=1.53518306e-6, icerateri=6.56073098e-8,
                        dQRfzri=0.0, dQIfzri=0.0, dNfzri=0.0,
                        dQImltri=6.56073098e-8, dNmltri=1.41032120e2),
        wet_growth = (rimetotal=4.51580684e-9, dry_growth=false),
        riming_growth = (prdr=3.12981818e-9, ardr=0.0, crdr=0.0,
                         rhorimeout=9.0e2, gdenavg=9.0e2, gdenavgr=9.0e2),
    ),
    (
        label = "tiny ice (small-ice-limit branch)",
        input = (temp=263.149994, pres=85000.0, qidum=9.99999972e-10, dsdum0=1.0,
                 ani0=1.99999999e-6, cni0=1.99999999e-6, rbdum0=920.0, nidum=5.0e7),
        derived = (qv=1.94427010e-3, qvi=1.90614723e-3, qvs=2.10253987e-3, sui=1.99999809e-2,
                   sup=-7.52755404e-2, igr=0.863071024, mu=1.66673981e-5, dv=2.48505203e-5,
                   kt=2.31112298e-2, nsch=5.96246123e-1, npr=6.41118037e-4, rhoair=1.12488151),
        var_check = (deltastr=1.0, ani=1.99999909e-6, cni=1.99999909e-6, rni=1.99999999e-6,
                     rhobar=920.0, ni=270.303894, ai=2.16242832e-15, ci=2.16242832e-15,
                     alphstr=1.0, alphv=4.18879032, betam=3.0),
        capgam = 7.99999725e-6,
        vaporgrow = (vtbarb=1.07312603e-2, vtbarbm=2.91969702e-2, vtbarbz=5.62195778e-2,
                     fvdum=1.00173581, fhdum=1.00001407, anf=2.03369541e-6, cnf=2.03098784e-6,
                     rnf=2.03214609e-6, iwcf=1.03728126e-9, rdout=7.00000000e2,
                     dsdumout=0.999240220, rbdum=920.0),
        riming_input = (qc=1.00000005e-3, nc=1.0e8, qr=9.99999975e-5, nr=1.0e5),
        itab_riming = (rimesum=0.0, qi_qc_nrm=0.0, qi_qc_nrd=0.0),
        itabr_riming = (rimesumr=0.0, qi_qr_nrm=0.0, qi_qr_nrd=0.0,
                        qi_qr_nrn=0.0, nr_final=1.0e5,
                        numrateri=0.0, rainrateri=0.0, icerateri=0.0,
                        dQRfzri=0.0, dQIfzri=0.0, dNfzri=0.0, dQImltri=0.0, dNmltri=0.0),
        wet_growth = (rimetotal=0.0, dry_growth=true),
        riming_growth = (prdr=0.0, ardr=0.0, crdr=0.0,
                         rhorimeout=9.20000000e2, gdenavg=9.0e2, gdenavgr=9.0e2),
    ),
    (
        label = "large ice ani>=cni (large-ice-limit branch A)",
        input = (temp=265.149994, pres=85000.0, qidum=4.99999989e-3, dsdum0=0.699999988,
                 ani0=2.00000009e-3, cni0=1.00000005e-3, rbdum0=500.0, nidum=500.0),
        derived = (qv=2.38852645e-3, qvi=2.27478729e-3, qvs=2.46048626e-3, sui=4.99999523e-2,
                   sup=-2.92461514e-2, igr=1.61921000, mu=1.67702347e-5, dv=2.51934271e-5,
                   kt=2.32535843e-2, nsch=5.96256912e-1, npr=6.45997410e-4, rhoair=1.11639655),
        var_check = (deltastr=0.699999988, ani=1.00000005e-3, cni=6.30957293e-5, rni=3.98107048e-4,
                     rhobar=84.5091858, ni=3249.00952, ai=2.04998656e-7, ci=1.29345379e-8,
                     alphstr=7.94328097e-3, alphv=3.32727395e-2, betam=2.70000005),
        capgam = 2.61689140e-3,
        vaporgrow = (vtbarb=0.709150493, vtbarbm=0.924969494, vtbarbz=1.09805107,
                     fvdum=6.18523026, fhdum=1.38163090, anf=1.00000715e-3, cnf=6.30964569e-5,
                     rnf=3.98110453e-4, iwcf=5.00105601e-3, rdout=7.00000000e2,
                     dsdumout=0.700000584, rbdum=84.5091858),
        # qi=5e-3 > 1e-4 and qr=5e-4 > 1e-4: exercises the T<=T0 ice-rain
        # FREEZE branch (dQRfzri/dQIfzri/dNfzri).
        riming_input = (qc=1.00000005e-3, nc=1.0e8, qr=5.00000024e-4, nr=5.0e5),
        itab_riming = (rimesum=1.07438996e-6, qi_qc_nrm=2.65322462e-18, qi_qc_nrd=2.60003033e-17),
        itabr_riming = (rimesumr=6.75755587e-7, qi_qr_nrm=3.33757980e-16, qi_qr_nrd=6.32207774e-15,
                        qi_qr_nrn=9.61941510e-7, nr_final=5.0e5,
                        numrateri=1.28831653e3, rainrateri=2.21828559e-5, icerateri=1.21114519e-3,
                        dQRfzri=2.21828559e-5, dQIfzri=1.21114519e-3, dNfzri=1.28831653e3,
                        dQImltri=0.0, dNmltri=0.0),
        wet_growth = (rimetotal=1.75014554e-6, dry_growth=true),
        riming_growth = (prdr=1.43819875e-6, ardr=2.88522053e-11, crdr=3.64090138e-12,
                         rhorimeout=1.92845245e2, gdenavg=1.68506775e2, gdenavgr=2.31541183e2),
    ),
    (
        label = "large ice cni>ani (large-ice-limit branch B)",
        input = (temp=265.149994, pres=85000.0, qidum=4.99999989e-3, dsdum0=1.25,
                 ani0=1.00000005e-3, cni0=2.00000009e-3, rbdum0=500.0, nidum=500.0),
        derived = (qv=2.38852645e-3, qvi=2.27478729e-3, qvs=2.46048626e-3, sui=4.99999523e-2,
                   sup=-2.92461514e-2, igr=1.61921000, mu=1.67702347e-5, dv=2.51934271e-5,
                   kt=2.32535843e-2, nsch=5.96256912e-1, npr=6.45997410e-4, rhoair=1.11639655),
        var_check = (deltastr=1.25, ani=1.58489303e-4, cni=1.00000005e-3, rni=2.92864366e-4,
                     rhobar=50.0, ni=4935.57666, ai=1.23976065e-7, ci=7.82236157e-7,
                     alphstr=56.2341309, alphv=235.552979, betam=3.25),
        capgam = 2.04556715e-3,
        vaporgrow = (vtbarb=0.600898385, vtbarbm=1.01514518, vtbarbz=1.40523350,
                     fvdum=2.91584063, fhdum=1.06141651, anf=1.58489813e-4, cnf=1.00000529e-3,
                     rnf=2.92865530e-4, iwcf=5.00083156e-3, rdout=7.00000000e2,
                     dsdumout=1.25000036, rbdum=50.0),
        riming_input = (qc=1.00000005e-3, nc=1.0e8, qr=5.00000024e-4, nr=5.0e5),
        itab_riming = (rimesum=1.53687379e-5, qi_qc_nrm=2.49840772e-17, qi_qc_nrd=2.66297815e-16),
        itabr_riming = (rimesumr=4.49404797e-6, qi_qr_nrm=1.46114337e-15, qi_qr_nrd=4.95770112e-14,
                        qi_qr_nrn=4.16459488e-6, nr_final=5.0e5,
                        numrateri=3.65988129e2, rainrateri=6.70543795e-6, icerateri=1.78460396e-4,
                        dQRfzri=6.70543795e-6, dQIfzri=1.78460396e-4, dNfzri=3.65988129e2,
                        dQImltri=0.0, dNmltri=0.0),
        wet_growth = (rimetotal=1.98627858e-5, dry_growth=true),
        riming_growth = (prdr=1.71317915e-5, ardr=1.12919694e-10, crdr=3.56237595e-10,
                         rhorimeout=2.09897797e2, gdenavg=1.74479874e2, gdenavgr=3.31019989e2),
    ),
)

# xm > 1e8 Mitchell-Heymsfield fall-speed fallback branch (am=1.0865,
# bm=0.499, lines 3278-3281), standalone (bypasses var_check -- ani=2mm
# exceeds the 1mm var_check cap by construction, see ref_driver.f90).
const ISHMAEL_REF_XM_FALLBACK = (
    temp = 233.149994, ani = 2.00000009e-3, dsdum = 1.0, rbdum = 920.0, igr = 1.22572994,
    xm_diagnostic = 6.13366784e8,
    vtbarb = 11.6323605, vtbarbm = 15.5685453, vtbarbz = 18.6877460,
)

# get_igr boundary sweep (dT in (-1,0], dT=-1 exactly, interior points,
# dT in [-60,-59), dT=-60 exactly, dT<-60 clamp).
const ISHMAEL_REF_IGR = (
    temps = [272.850006, 272.149994, 271.649994, 267.149994, 261.149994,
             253.149994, 238.149994, 214.149994, 213.649994, 213.149994, 208.149994],
    igr   = [0.973165154, 0.910547018, 0.864308476, 2.32422996, 0.453916997,
             0.796458006, 1.12233996, 1.50086999, 1.50592494, 1.51098001, 1.51098001],
)

# access_lookup_table: itab (index alternating 1,2) and itabr (index
# cycling 1..6). See tools/ishmael_fortran_driver/ref_driver.f90 for the
# exact (dumjj,dumii,dumi,dumk,dum1,dum2,dum4,dum5) inputs.
const ISHMAEL_REF_LUT_ITAB = (
    dumjj = [5, 10, 20, 30, 45], dumii = [3, 5, 7, 9, 10], dumi = [5, 15, 25, 35, 45],
    dumk  = [2, 4, 6, 8, 10], index = [1, 2, 1, 2, 1],
    dum1  = [5.3, 15.7, 25.2, 35.9, 45.1], dum2 = [2.4, 4.6, 6.1, 8.8, 10.2],
    dum4  = [3.2, 5.9, 7.3, 9.1, 10.4], dum5 = [5.1, 10.6, 20.3, 30.8, 45.4],
    proc  = [0.0, 0.0, 9.54035458e-29, 3.20868671e-21, 1.95277762e-19],
)

const ISHMAEL_REF_LUT_ITABR = (
    dumjj = [5, 10, 20, 30, 45], dumii = [2, 4, 6, 8, 9], dumi = [5, 15, 25, 35, 45],
    dumk  = [2, 4, 6, 8, 10], index = [1, 2, 3, 4, 5],
    dum1  = [5.5, 15.1, 25.8, 35.3, 45.6], dum2 = [2.9, 4.2, 6.7, 8.1, 10.9],
    dum4  = [2.1, 4.8, 6.3, 8.6, 9.4], dum5 = [5.9, 10.2, 20.7, 30.1, 45.8],
    proc  = [0.0, 2.96468001e-23, 1.27402430e-7, 8.65409480e-11, 1.83725740e-21],
)

# ──────────────────────────────────────────────
# Stage S6b: aggregation (+ col1), lines 3988-4541. 6 state points, 3 live
# categories per point (1=planar[ICE1], 2=columnar[ICE2], 3=aggregates
# [ICE3], matching aggregation()'s own 3/4/5 category numbering). Inputs
# are what the driver fed to `aggregation` directly (dt=2.0 in every
# case); outputs are `aggregation`'s own (qagg1,qagg2,qagg3,nagg1,nagg2,
# nagg3,ddum3) -- timestep-INTEGRATED amounts (colamt already has dtlt
# baked in), NOT rates -- matching how mp_jensen_ishmael itself adds them
# straight into qi/ni with no further *dt (line 2398).
# ──────────────────────────────────────────────
const ISHMAEL_REF_AGGREGATION = (
    (
        label = "moderate planar+columnar, T=-15C",
        dt = 2.0, temp = 258.149994, rhoair = 0.699999988,
        q1=1.00000005e-3, n1=1.0e5, d1=1.99999995e-4,
        q2=1.00000005e-3, n2=1.0e5, d2=1.99999995e-4,
        q3=9.99999975e-5, n3=1.0e3, d3=5.00000024e-4,
        rho1=300.0, rho2=300.0, phi1=3.00000012e-1, phi2=3.0,
        qagg1=-6.30378906e-7, qagg2=-6.30378906e-7, qagg3=1.26075781e-6,
        nagg1=-2.98987141e1, nagg2=-2.98987141e1, nagg3=2.88421726e1,
        ddum3=5.39060507e-4,
    ),
    (
        label = "empty aggregates, dendritic growth zone T=-14C",
        dt = 2.0, temp = 259.149994, rhoair = 0.699999988,
        q1=5.00000024e-4, n1=5.0e4, d1=1.50000007e-4,
        q2=5.00000024e-4, n2=5.0e4, d2=1.50000007e-4,
        q3=0.0, n3=0.0, d3=9.99999997e-7,
        rho1=250.0, rho2=250.0, phi1=4.00000006e-1, phi2=2.5,
        qagg1=-5.03308719e-8, qagg2=-5.03308719e-8, qagg3=1.00661744e-7,
        nagg1=-3.57433772, nagg2=-3.57433772, nagg3=3.57433772,
        ddum3=3.55219119e-4,
    ),
    (
        label = "warm near 0C, quasi-spherical, T=-1C",
        dt = 2.0, temp = 272.149994, rhoair = 1.0,
        q1=1.00000005e-3, n1=2.0e4, d1=3.00000014e-4,
        q2=1.00000005e-3, n2=2.0e4, d2=3.00000014e-4,
        q3=5.00000024e-4, n3=5.0e2, d3=7.99999980e-4,
        rho1=850.0, rho2=850.0, phi1=8.99999976e-1, phi2=1.10000002,
        qagg1=-4.34819381e-9, qagg2=-4.34819381e-9, qagg3=8.69638761e-9,
        nagg1=-4.13798690e-2, nagg2=-4.13798690e-2, nagg3=1.03997700e-2,
        ddum3=1.16754312e-3,
    ),
    (
        label = "extreme oblate/prolate, T=-25C",
        dt = 2.0, temp = 248.149994, rhoair = 0.550000012,
        q1=2.00000009e-3, n1=1.0e5, d1=2.50000012e-4,
        q2=2.00000009e-3, n2=1.0e5, d2=2.50000012e-4,
        q3=1.99999995e-4, n3=2.0e3, d3=6.00000028e-4,
        rho1=100.0, rho2=100.0, phi1=5.00000007e-2, phi2=20.0,
        qagg1=-1.54249096e-6, qagg2=-1.54249096e-6, qagg3=3.08498193e-6,
        nagg1=-4.87630920e1, nagg2=-4.87630920e1, nagg3=4.72504539e1,
        ddum3=5.40475070e-4,
    ),
    (
        label = "large existing aggregate population, T=-18C",
        dt = 2.0, temp = 255.149994, rhoair = 0.75,
        q1=9.99999975e-5, n1=1.0e3, d1=9.99999975e-5,
        q2=9.99999975e-5, n2=1.0e3, d2=9.99999975e-5,
        q3=2.00000009e-3, n3=5.0e3, d3=1.00000005e-3,
        rho1=200.0, rho2=200.0, phi1=5.0e-1, phi2=2.0,
        qagg1=-1.34388247e-11, qagg2=-1.34388247e-11, qagg3=2.68776494e-11,
        # nagg3 is NEGATIVE here: aggregate self-collection (a large
        # pre-existing n3=5e3 population) dominates the small planar/
        # columnar-driven gains, and self-collection's number LOSS term
        # (-0.5*ppenxfer(5,5)) is never offset by a mass gain (aggregation
        # is mass-conserving; agg+agg->agg changes count, not mass -- see
        # the mass-balance note above `ISHMAEL_REF_AGGREGATION`).
        nagg1=-6.04582764e-3, nagg2=-6.04582764e-3, nagg3=-2.71197200,
        ddum3=8.60409637e-4,
    ),
    (
        label = "very cold, small ice, T=-45C",
        dt = 2.0, temp = 228.149994, rhoair = 0.400000006,
        q1=4.99999987e-5, n1=1.0e4, d1=4.99999987e-5,
        q2=4.99999987e-5, n2=1.0e4, d2=4.99999987e-5,
        q3=9.99999975e-6, n3=1.0e2, d3=7.99999980e-5,
        rho1=400.0, rho2=400.0, phi1=4.00000006e-1, phi2=2.5,
        qagg1=-3.00627849e-14, qagg2=-3.00627849e-14, qagg3=6.01255631e-14,
        nagg1=-5.81517161e-5, nagg2=-5.81517161e-5, nagg3=5.72866120e-5,
        ddum3=5.41925954e-4,
    ),
)
