# ── MYNN-EDMF constants: the host set and the closure set ─────────────────────
#
# Part 1 of the pure-Julia port of the MYNN-EDMF boundary-layer scheme
# (ccpp-physics `module_bl_mynn.F90` + `bl_mynn_common.f90`, verbatim copies under
# tools/mynn_fortran_driver/). This file holds the numbers; src/mynn_closure.jl holds
# the routines that use them.
#
# There are two kinds of constant, and they are kept apart on purpose:
#
#   1. HOST constants — `bl_mynn_common`'s module variables, which a dycore fills in at
#      init time (`mynnedmf_wrapper_init`) plus the ten quantities that routine derives
#      from them. Scythe's dycore is Springsteel, so `MYNNConstants()` reads
#      `Springsteel.Thermodynamics` and forms the derived ten with the wrapper's exact
#      expressions and association. `tools/mynn_fortran_driver/columns/constants.txt`
#      was written from the SAME Springsteel values by tools/mynn_dump_columns.jl, so
#      the `## constants` block of the reference output is reproduced BITWISE by this
#      struct (test/test_mynn_closure.jl checks all 24 with `===`).
#
#   2. CLOSURE constants — `module_bl_mynn.F90` lines 272-341 `parameter`s, which are
#      the scheme itself and never come from the host. They are plain `const`s here,
#      transcribed with the Fortran's literals and association order preserved, each
#      carrying its Fortran line number.
#
# NAMING. Every closure constant carries a `MYNN_` prefix and the Fortran name in
# upper case (`pr` -> `MYNN_PR`, `e1c` -> `MYNN_E1C`). The Fortran names are one and
# two characters long (`a1`, `b1`, `c1`, `g1`, `pr`) and this file is `include`d
# straight into the `Scythe` module namespace, where such names would be a collision
# waiting to happen. The mapping is mechanical: strip `MYNN_`, lower-case.
#
# PRECISION. The reference (`ref_driver_output_r8.txt`) was built with
# `-fdefault-real-8 -fdefault-double-8`, i.e. every bare `real` local and every
# default-real literal in the Fortran promoted to double. Everything here is therefore
# Float64, including the constants the production (native-precision) build would round
# to single. See tools/mynn_fortran_driver/README.md, "Build flags".

# ── 1. Host constants ─────────────────────────────────────────────────────────

"""
    MYNNConstants

The `bl_mynn_common` host constants and the ten quantities `mynnedmf_wrapper_init`
derives from them, in the order the reference driver's `## constants` block prints them.

Fields 1-14 are what a dycore supplies (`bl_mynn_common.f90` :29-44); fields 15-24 are
derived exactly as `mynnedmf_wrapper_init` derives them (see `read_constants` in
tools/mynn_fortran_driver/ref_driver.f90, which mirrors that routine):

    xls    = xlv + xlf          rvovrd = r_v / r_d        ep_3  = 1. - ep_2
    gtr    = grav / tref        rk     = cp / r_d         tv0   = p608 * tref
    tv1    = (1. + p608)*tref   xlscp  = (xlv+xlf)/cp     xlvcp = xlv / cp
    g_inv  = 1. / grav

Note that `xls` and `xlscp` are formed independently — `xlscp` is `(xlv+xlf)/cp`, not
`xls/cp` — which is what the Fortran does and is bitwise identical here anyway.

`tref` (300 K), `tice` (240 K), `p1000mb`, `onethird`, `twothirds` and friends are NOT
host quantities: they are `parameter`s of `bl_mynn_common` and live below as
`MYNN_TREF`, `MYNN_TICE`, ...

Build one with the zero-argument constructor, which pulls `Springsteel.Thermodynamics`:

    c = MYNNConstants()
"""
struct MYNNConstants
    # supplied by the host (bl_mynn_common.f90 :29-44)
    cp::Float64        # Cpd
    cpv::Float64       # Cpv
    cliq::Float64      # Cl
    cice::Float64      # Ci
    p608::Float64      # Rv/Rd - 1
    ep_2::Float64      # Rd/Rv
    grav::Float64
    karman::Float64
    t0c::Float64       # T_0  (Springsteel's 273.16, NOT 273.15 — see note below)
    rcp::Float64       # Rd/Cpd
    r_d::Float64
    r_v::Float64
    xlf::Float64       # L_f0
    xlv::Float64       # L_v0
    # derived in mynnedmf_wrapper_init
    xls::Float64
    rvovrd::Float64
    ep_3::Float64
    gtr::Float64
    rk::Float64
    tv0::Float64
    tv1::Float64
    xlscp::Float64
    xlvcp::Float64
    g_inv::Float64
end

"""
    MYNN_CONSTANT_ORDER

Field names of `MYNNConstants` in the order the Fortran driver prints the `## constants`
block (`ref_driver.f90` `read_constants`). Used by the parity test.
"""
const MYNN_CONSTANT_ORDER = (:cp, :cpv, :cliq, :cice, :p608, :ep_2, :grav, :karman,
                             :t0c, :rcp, :r_d, :r_v, :xlf, :xlv, :xls, :rvovrd,
                             :ep_3, :gtr, :rk, :tv0, :tv1, :xlscp, :xlvcp, :g_inv)

"""
    MYNN_KARMAN

von Karman constant. `bl_mynn_common` takes it from the host; Springsteel has no such
constant, so the value the column dump used (tools/mynn_dump_columns.jl `KARMAN`) is
repeated here. Change it in both places or the reference stops applying.
"""
const MYNN_KARMAN = 0.4

"""
    MYNNConstants(; karman = MYNN_KARMAN)

Build the host constant set from `Springsteel.Thermodynamics`. The 14 supplied values
are formed with exactly the expressions tools/mynn_dump_columns.jl used to write
`columns/constants.txt`, and the 10 derived ones exactly as `mynnedmf_wrapper_init`
forms them, so the whole struct matches the Fortran reference bitwise.

`t0c` is Springsteel's `T_0` = 273.16 K, not the 273.15 K of the Fortran comment. The
Fortran reads whatever the host gives it, and so does the reference run; `esat_blend`'s
`t0c - 6` blend edge and `xl_blend`'s reference temperature therefore both sit 0.01 K
above the CCPP defaults. That is a deliberate consequence of using Scythe's own
thermodynamics and is what the parity tests pin.
"""
function MYNNConstants(; karman::Float64 = MYNN_KARMAN)
    Th = Springsteel.Thermodynamics
    cp   = Float64(Th.Cpd)
    cpv  = Float64(Th.Cpv)
    cliq = Float64(Th.Cl)
    cice = Float64(Th.Ci)
    r_d  = Float64(Th.Rd)
    r_v  = Float64(Th.Rv)
    p608 = r_v / r_d - 1.0
    ep_2 = r_d / r_v
    grav = Float64(Th.gravity)
    t0c  = Float64(Th.T_0)
    rcp  = r_d / cp
    xlf  = Float64(Th.L_f0)
    xlv  = Float64(Th.L_v0)
    # derived exactly as mynnedmf_wrapper_init (ref_driver.f90 read_constants)
    xls    = xlv + xlf
    rvovrd = r_v / r_d
    ep_3   = 1.0 - ep_2
    gtr    = grav / MYNN_TREF
    rk     = cp / r_d
    tv0    = p608 * MYNN_TREF
    tv1    = (1.0 + p608) * MYNN_TREF
    xlscp  = (xlv + xlf) / cp
    xlvcp  = xlv / cp
    g_inv  = 1.0 / grav
    return MYNNConstants(cp, cpv, cliq, cice, p608, ep_2, grav, karman, t0c, rcp,
                         r_d, r_v, xlf, xlv, xls, rvovrd, ep_3, gtr, rk, tv0, tv1,
                         xlscp, xlvcp, g_inv)
end

# ── bl_mynn_common `parameter`s (bl_mynn_common.f90 :46-58) ───────────────────
const MYNN_ZERO      = 0.0        # :46
const MYNN_HALF      = 0.5        # :47
const MYNN_ONE       = 1.0        # :48
const MYNN_TWO       = 2.0        # :49
const MYNN_ONETHIRD  = 1.0 / 3.0  # :50   onethird  = 1./3.
const MYNN_TWOTHIRDS = 2.0 / 3.0  # :51   twothirds = 2./3.
const MYNN_TREF      = 300.0      # :52   reference temperature (K)
const MYNN_TKMIN     = 253.0      # :53   Tripoli and Cotton (1981)
const MYNN_P1000MB   = 100000.0   # :54
const MYNN_SVP1      = 0.6112     # :55   (kPa)
const MYNN_SVP2      = 17.67      # :56
const MYNN_SVP3      = 29.65      # :57   (K)
const MYNN_TICE      = 240.0      # :58   -33 C, saturation w.r.t. ice

# ── 2. Closure constants (module_bl_mynn.F90 :272-341) ────────────────────────
#
# Stability-function parameters of module_sf_mynn (:272-273).
const MYNN_CPHM_ST   = 5.0        # :272
const MYNN_CPHM_UNST = 16.0       # :272
const MYNN_CPHH_ST   = 5.0        # :273
const MYNN_CPHH_UNST = 16.0       # :273

# Closure constants (:276-289). Association is the Fortran's, left to right.
const MYNN_PR = 0.74                                   # :277
const MYNN_G1 = 0.235                                  # :278  NN2009 = 0.235
const MYNN_B1 = 24.0                                   # :279
const MYNN_B2 = 15.0                                   # :280  CKmod / NN2009
const MYNN_C2 = 0.729                                  # :281
const MYNN_C3 = 0.340                                  # :282
const MYNN_C4 = 0.0                                    # :283
const MYNN_C5 = 0.2                                    # :284
const MYNN_A1 = MYNN_B1 * (1.0 - 3.0 * MYNN_G1) / 6.0  # :285
# :286 has the analytic form commented out; :287 is what is compiled. The magic number
# is b1**(1/3) = 24**(1/3) evaluated once and frozen in the source, so the port must
# use the SAME 18 digits rather than recomputing cbrt(24).
const MYNN_C1 = MYNN_G1 - 1.0 / (3.0 * MYNN_A1 * 2.88449914061481660)          # :287
const MYNN_A2 = MYNN_A1 * (MYNN_G1 - MYNN_C1) / (MYNN_G1 * MYNN_PR)            # :288
const MYNN_G2 = MYNN_B2 / MYNN_B1 * (1.0 - MYNN_C3) +
                2.0 * MYNN_A1 / MYNN_B1 * (3.0 - 2.0 * MYNN_C2)                # :289

const MYNN_CC2 = 1.0 - MYNN_C2                                                 # :292
const MYNN_CC3 = 1.0 - MYNN_C3                                                 # :293
const MYNN_E1C =  3.0 * MYNN_A2 * MYNN_B2 * MYNN_CC3                           # :294
const MYNN_E2C =  9.0 * MYNN_A1 * MYNN_A2 * MYNN_CC2                           # :295
const MYNN_E3C =  9.0 * MYNN_A2 * MYNN_A2 * MYNN_CC2 * (1.0 - MYNN_C5)         # :296
const MYNN_E4C = 12.0 * MYNN_A1 * MYNN_A2 * MYNN_CC2                           # :297
const MYNN_E5C =  6.0 * MYNN_A1 * MYNN_A1                                      # :298

# Min TKE in the elt integration, max z/L in els, Kq = Sqfac*Km (:302).
const MYNN_QMIN  = 0.0    # :302
const MYNN_ZMAX  = 1.0    # :302
const MYNN_SQFAC = 3.0    # :302

const MYNN_QKEMIN = 1.0e-4  # :306
const MYNN_TLIQ   = 269.0   # :307  all hydrometeors liquid above this

# Cloud-PDF constants of mym_condensation (:310). Ported here for part 2.
const MYNN_RR2 = 0.7071068  # :310  1/sqrt(2), as the Fortran spells it
const MYNN_RRP = 0.3989423  # :310  1/sqrt(2 pi)

# Canuto/Kitamura modification: 1 = on. mym_level2 and mym_turbulence branch on it
# (they compare `CKmod .eq. 1`, a real compared with an integer literal).
const MYNN_CKMOD = 1.0      # :320

# Ito et al. (2015) scale-awareness: 1 = on.
const MYNN_SCALEAWARE = 1.0 # :325

const MYNN_BL_MYNN_TOPDOWN = 0     # :329  top-down radiative-cooling diffusion off
const MYNN_BL_MYNN_EDMF_DD = 0     # :331  downdrafts off
const MYNN_DHEAT_OPT       = 1     # :334  TKE dissipative heating on
const MYNN_ENV_SUBS        = false # :337  environmental subsidence in the MF scheme
# 0: original Dyer-Hicks; 1: Cheng-Brutsaert + blended COARE, which is what `phim`
# and `phih` below implement. The switch is read by the surface-layer caller, not by
# the functions themselves.
const MYNN_BL_MYNN_STFUNC  = 1     # :341

# ── 3. mym_length CASE 2 constants (module_bl_mynn.F90 :2101-2116) ────────────
#
# CASE 2 ("local (mostly) mixing length formulation") is the only branch this port
# implements — bl_mynn_mixlength = 2 is what the reference driver and Scythe use. The
# Fortran declares cns/alp1..alp6 as plain locals and assigns them per CASE, so their
# values are branch-specific; these are the CASE-2 assignments only.
const MYNN_CNS  = 3.5        # :2105  surface layer (els) in stable conditions
const MYNN_ALP1 = 0.22       # :2106  turbulent length scale (elt)
const MYNN_ALP2 = 0.30       # :2107  buoyancy length scale (elb)
const MYNN_ALP3 = 2.0        # :2108  buoyancy enhancement factor of elb
const MYNN_ALP4 = 5.0        # :2109  surface layer (els) in unstable conditions
const MYNN_ALP5 = MYNN_ALP2  # :2110  like alp2, but for the free atmosphere
const MYNN_ALP6 = 50.0       # :2111  mass-flux mixing length

# Limits on the mixing-length calculation only — they do not move the diagnosed PBLH.
# NOTE: CASE 2 does NOT actually use minzi/maxdz/mindz. It spells 300. and 600. as
# literals (:2115, :2118-2119) with the parameter forms commented out immediately above
# (:2114, :2116-2117). They are transcribed here because they are `parameter`s of
# mym_length (:1900-1915) and CASE 0 does use mindz/maxdz; the CASE-2 code writes the
# literals, not these names.
const MYNN_MINZI = 300.0     # :1900  min mixed-layer height
const MYNN_MAXDZ = 750.0     # :1901  max (half) transition-layer depth (0.3 * 2500 m)
const MYNN_MINDZ = 300.0     # :1904  min (half) transition-layer depth

const MYNN_ZSLH = 100.0      # :1907  max height correlated to surface conditions (m)
const MYNN_CSL  = 2.0        # :1908  constant of proportionality to L, O(1)
const MYNN_CTAU = 1000.0     # :1915  constant for tau_cloud

# ── 4. The mynnedmf_wrapper surface-flux clips ────────────────────────────────
#
# NOT constants of the closure: these are the limits `mynnedmf_wrapper` puts on the
# surface fluxes it hands `mynn_bl_driver` (tools/mynn_fortran_driver/README.md item 7),
# and they are the `:flux_clip` fidelity deviation (`MYNN_DEVIATIONS`). Named here so the
# wired coupling, the offline column step and the counters all clip at the SAME numbers.
const MYNN_HFX_MIN = -500.0    # W m-2
const MYNN_HFX_MAX = 1200.0    # W m-2
const MYNN_QFX_MIN = -2.0e-4   # kg m-2 s-1
const MYNN_QFX_MAX = 5.0e-4    # kg m-2 s-1
