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
