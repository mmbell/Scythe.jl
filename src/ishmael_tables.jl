# src/ishmael_tables.jl
#
# ISHMAEL ice-microphysics lookup-table loader and pure-utility layer.
#
# Fortran source of truth:
#   /Users/mmbell/Development/cm1r21.1/src/module_mp_jensen_ishmael.F
#
# This file ports the table I/O and the small set of PURE helper routines
# the ISHMAEL scheme leans on: quadrilinear lookup-table interpolation,
# distribution-averaged spheroid capacitance, the inherent-growth-ratio
# data, the ice-moment consistency check (var_check), and the ice-ice
# aggregation collection-table builder (mkcoltb/xjnum). It does NOT port
# any of the tendency/process routines (vaporgrow, aggregation, riming,
# etc.) -- those are a later stage.
#
# `gamma_tab` (the Fortran's 505001-point tabulation of the Cody GAMMA
# function, used purely for speed) is dropped entirely: every use is
# replaced by a direct call to SpecialFunctions.gamma / loggamma /
# gamma_inc. `ishmael-gamma-tab.bin` is therefore never read by the
# converter (tools/convert_ishmael_tables.jl) or this loader.

using JLD2
using SpecialFunctions: gamma, loggamma, gamma_inc

# ────────────────────────────────────────────────────────────────────────────
# Fixed ice-ice collection category tables (jensen_ishmael_init, lines 90-120)
# ────────────────────────────────────────────────────────────────────────────
#
# Categories (dstprms column header, lines 108-109):
#   1 cloud   2 rain   3 planar   4 columnar   5 aggreg   6 graup   7 hail
#
# Only categories 3 (planar) and 5 (aggreg) are exercised by the 3-species
# (vapor/planar/aggregate... ) subset the Julia port targets, but the full
# 7-category tables are transcribed here exactly as the Fortran DATA/reshape
# statements define them (lines 97-120) since they are small and mkcoltb
# builds all 35 pairs regardless of which categories a given equation set
# uses.

"""
Ice-ice collection pair index table `ipair(ix,iy)` (lines 97-102),
transcribed from the Fortran `reshape(..., (/ncat,ncat/))` column-major
literal. `ipair[ix,iy] > 0` gives the `coltab`/`coltabn` third-dimension
slot for the (ix,iy) category pair; `0` means "no pair defined".
"""
const ISHMAEL_IPAIR = [
    0   1   2   6  10  14  18;
    0   0   4   8  12  16  20;
    3   5  22  23  25  26  27;
    7   9  24  28  29  30  31;
   11  13   0   0  35  32  33;
   15  17   0   0   0   0  34;
   19  21   0   0   0   0   0;
]

# dstprms(icat, 1:9) (lines 105-120), one column per property, transcribed
# from the Fortran DATA groups (each group of 7 fills one dstprms column).
const ISHMAEL_SHAPE = [0.5, 0.5, 0.318, 0.318, 0.5, 0.5, 0.5]        # dstprms(:,1): capacitance shape
const ISHMAEL_CFMAS = [524.0, 524.0, 0.333, 0.333, 0.496, 157.0, 471.0]  # dstprms(:,2): m = cfmas*D^pwmas
const ISHMAEL_PWMAS = [3.0, 3.0, 2.4, 2.4, 2.4, 3.0, 3.0]            # dstprms(:,3)
const ISHMAEL_CFVT  = [3173.0, 149.0, 4.836, 4.836, 3.084, 93.3, 161.0]  # dstprms(:,4): v = cfvt*D^pwvt
const ISHMAEL_PWVT  = [2.0, 0.5, 0.25, 0.25, 0.2, 0.5, 0.5]          # dstprms(:,5)
const ISHMAEL_TABLO = [1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6]  # dstprms(:,6): table size lo
const ISHMAEL_TABHI = [1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2]  # dstprms(:,7): table size hi
const ISHMAEL_DNMIN = [0.1e-5, 0.1e-4, 0.1e-5, 0.1e-5, 0.1e-4, 0.1e-4, 0.1e-4]  # dstprms(:,8)
const ISHMAEL_DNMAX = [0.1e-2, 0.1e-1, 0.1e-1, 0.1e-1, 0.1e-1, 0.1e-1, 0.1e-1]  # dstprms(:,9)
const ISHMAEL_GNU    = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]           # gnu(icat) = 4.0 (line 129)

# Fortran module-level parameter PI = 3.14159265 (line 36) -- a single-
# precision literal, NOT Base.pi. Transcribed exactly so ported formulas
# that reference PI reproduce the Fortran's numerics rather than switching
# to full double-precision pi mid-formula.
const ISHMAEL_PI = 3.14159265

"""
    igrdata (60 values, -1 C to -60 C)

Inherent growth ratio (IGR) data from Chen and Lamb (1994) / Lamb and Scott
(1972), transcribed verbatim from the Fortran `DATA` statement (lines
195-204).
"""
const ISHMAEL_IGRDATA = Float64[
    0.910547, 0.81807, 0.6874, 0.60127, 1.59767, 2.32423, 2.08818,
    1.61921, 1.15865, 0.863071, 0.617586, 0.453917, 0.351975, 0.28794,
    0.269298, 0.28794, 0.333623, 0.418883, 0.56992, 0.796458, 1.14325,
    1.64103, 1.90138, 1.82653, 1.61921, 1.47436, 1.32463, 1.25556,
    1.22239, 1.206, 1.11522, 1.10751, 1.10738, 1.11484, 1.12234,
    1.12221, 1.14529, 1.16884, 1.20104, 1.22573, 1.25094, 1.27666,
    1.31183, 1.3388, 1.35704, 1.37553, 1.38479, 1.39411, 1.40349,
    1.41294, 1.42245, 1.43202, 1.44166, 1.45137, 1.46114, 1.47097,
    1.48087, 1.50105, 1.50087, 1.51098,
]
@assert length(ISHMAEL_IGRDATA) == 60

# ────────────────────────────────────────────────────────────────────────────
# 1. Loader
# ────────────────────────────────────────────────────────────────────────────

"""
    IshmaelTables

Immutable container for the ISHMAEL microphysics lookup tables.

- `itab::Array{Float64,5}`  -- ice-cloud collection table, `(51,51,51,11,2)`.
- `itabr::Array{Float64,5}` -- ice-rain collection table, `(51,51,51,11,6)`.
- `coltab::Array{Float64,3}`  -- aggregation mass-collection table, `(60,60,35)`.
- `coltabn::Array{Float64,3}` -- aggregation number-collection table, `(60,60,35)`.
- `igrdata::Vector{Float64}`  -- inherent-growth-ratio data, length 60.

`itab`/`itabr` are read from a JLD2 file produced by
`tools/convert_ishmael_tables.jl`. `coltab`/`coltabn` are NOT stored in that
file -- they are rebuilt at load time by [`mkcoltb`](@ref) (a few tenths of
a second), since they are cheap to (re)compute and storing them would just
duplicate `mkcoltb`'s own output.
"""
struct IshmaelTables
    itab::Array{Float64,5}
    itabr::Array{Float64,5}
    coltab::Array{Float64,3}
    coltabn::Array{Float64,3}
    igrdata::Vector{Float64}
end

"""
    load_ishmael_tables(path::String) -> IshmaelTables

Load `itab`/`itabr` from the JLD2 file at `path` (written by
`tools/convert_ishmael_tables.jl`), build `coltab`/`coltabn` via
[`mkcoltb`](@ref), and attach the fixed `igrdata` table. Errors if `itab`/
`itabr` are not shaped `(51,51,51,11,2)` / `(51,51,51,11,6)` or contain any
non-finite values.
"""
function load_ishmael_tables(path::String)
    itab_raw, itabr_raw = jldopen(path, "r") do f
        (f["itab"], f["itabr"])
    end

    itab  = convert(Array{Float64,5}, itab_raw)
    itabr = convert(Array{Float64,5}, itabr_raw)

    size(itab)  == (51, 51, 51, 11, 2) || error(
        "load_ishmael_tables: itab has shape $(size(itab)), expected (51,51,51,11,2)")
    size(itabr) == (51, 51, 51, 11, 6) || error(
        "load_ishmael_tables: itabr has shape $(size(itabr)), expected (51,51,51,11,6)")
    all(isfinite, itab)  || error("load_ishmael_tables: itab contains non-finite values")
    all(isfinite, itabr) || error("load_ishmael_tables: itabr contains non-finite values")

    coltab, coltabn = mkcoltb()

    return IshmaelTables(itab, itabr, coltab, coltabn, copy(ISHMAEL_IGRDATA))
end

# ────────────────────────────────────────────────────────────────────────────
# 2. access_lookup_table -- quadrilinear interpolation (lines 3619-3680)
# ────────────────────────────────────────────────────────────────────────────

"""
    access_lookup_table(itabdum, dumjj, dumii, dumi, dumk, index,
                         dum1, dum2, dum4, dum5) -> Float64

Quadrilinear interpolation into a 5-D ISHMAEL lookup table (`itab` or
`itabr`), fixing the 5th dimension at `index` and interpolating over the
other four: `dum1`/`dum2` interpolate between grid nodes `(dumi,dumi+1)`
and `(dumk,dumk+1)` in dims 3 and 4, then `dum4` interpolates between
`(dumii,dumii+1)` in dim 2, then `dum5` interpolates between
`(dumjj,dumjj+1)` in dim 1. Ports `access_lookup_table` (from Hugh
Morrison), lines 3619-3680, verbatim.

Caller is responsible for `1 <= dumjj,dumii,dumi < size(itabdum,·)` (the
`+1` neighbor must be in bounds) -- this mirrors the Fortran, which does no
bounds checking either. `@inbounds`/concrete `Array{Float64,5}` input keeps
this allocation-free.
"""
@inline function access_lookup_table(itabdum::Array{Float64,5},
                                      dumjj::Int, dumii::Int, dumi::Int, dumk::Int, index::Int,
                                      dum1::Float64, dum2::Float64, dum4::Float64, dum5::Float64)
    @inbounds begin
        # value at current density index (dumjj): interpolate rimed-fraction index dumii
        dproc1 = itabdum[dumjj, dumii, dumi, dumk, index] + (dum1 - dumi) *
            (itabdum[dumjj, dumii, dumi+1, dumk, index] - itabdum[dumjj, dumii, dumi, dumk, index])
        dproc2 = itabdum[dumjj, dumii, dumi, dumk+1, index] + (dum1 - dumi) *
            (itabdum[dumjj, dumii, dumi+1, dumk+1, index] - itabdum[dumjj, dumii, dumi, dumk+1, index])
        iproc1 = dproc1 + (dum2 - dumk) * (dproc2 - dproc1)

        # rimed-fraction index dumii+1
        dproc1 = itabdum[dumjj, dumii+1, dumi, dumk, index] + (dum1 - dumi) *
            (itabdum[dumjj, dumii+1, dumi+1, dumk, index] - itabdum[dumjj, dumii+1, dumi, dumk, index])
        dproc2 = itabdum[dumjj, dumii+1, dumi, dumk+1, index] + (dum1 - dumi) *
            (itabdum[dumjj, dumii+1, dumi+1, dumk+1, index] - itabdum[dumjj, dumii+1, dumi, dumk+1, index])
        gproc1 = dproc1 + (dum2 - dumk) * (dproc2 - dproc1)

        tmp1 = iproc1 + (dum4 - dumii) * (gproc1 - iproc1)

        # value at density index dumjj+1: interpolate rimed-fraction index dumii
        dproc1 = itabdum[dumjj+1, dumii, dumi, dumk, index] + (dum1 - dumi) *
            (itabdum[dumjj+1, dumii, dumi+1, dumk, index] - itabdum[dumjj+1, dumii, dumi, dumk, index])
        dproc2 = itabdum[dumjj+1, dumii, dumi, dumk+1, index] + (dum1 - dumi) *
            (itabdum[dumjj+1, dumii, dumi+1, dumk+1, index] - itabdum[dumjj+1, dumii, dumi, dumk+1, index])
        iproc1 = dproc1 + (dum2 - dumk) * (dproc2 - dproc1)

        # rimed-fraction index dumii+1
        dproc1 = itabdum[dumjj+1, dumii+1, dumi, dumk, index] + (dum1 - dumi) *
            (itabdum[dumjj+1, dumii+1, dumi+1, dumk, index] - itabdum[dumjj+1, dumii+1, dumi, dumk, index])
        dproc2 = itabdum[dumjj+1, dumii+1, dumi, dumk+1, index] + (dum1 - dumi) *
            (itabdum[dumjj+1, dumii+1, dumi+1, dumk+1, index] - itabdum[dumjj+1, dumii+1, dumi, dumk+1, index])
        gproc1 = dproc1 + (dum2 - dumk) * (dproc2 - dproc1)

        tmp2 = iproc1 + (dum4 - dumii) * (gproc1 - iproc1)

        proc = tmp1 + (dum5 - dumjj) * (tmp2 - tmp1)
    end
    return proc
end

# ────────────────────────────────────────────────────────────────────────────
# 3. capacitance_gamma -- distribution-averaged spheroid capacitance
#    (lines 3488-3533, Harrington et al. 2013)
# ────────────────────────────────────────────────────────────────────────────

"""
    capacitance_gamma(ani, dsdum, NU, alphstr, i_gammnu) -> Float64

Distribution-averaged spheroid electrostatic capacitance (Harrington et al.
2013), used to scale vapor deposition/sublimation growth. `dsdum` is the
shape parameter deltastr (`<=1` oblate, `>1` prolate); `ani` the
distribution's a-axis scale; `NU` the gamma-distribution shape parameter;
`alphstr = ao^(1-dsdum)`; `i_gammnu = 1/gamma(NU)`. Ports
`capacitance_gamma`, lines 3488-3533, replacing the Fortran's Cody `gamma`
intrinsic (and its `gamma_tab` fast-path in other routines) with
`SpecialFunctions.gamma`. The Fortran's two structurally-identical
"`dsdum.le.1`" and "`dsdum.gt.1`" *evaluation* branches (they only differ in
how `c1,c2,d1,d2` were set up above) are merged into one, but the two
oblate/prolate *parameter* branches above are kept distinct.
"""
@inline function capacitance_gamma(ani::Float64, dsdum::Float64, NU::Float64,
                                    alphstr::Float64, i_gammnu::Float64)
    local a1, a2, b1, b2, c1, c2, d1, d2
    if dsdum <= 1.0
        # Oblate spheroid (lines 3496-3504)
        a1 = 0.6369427
        a2 = 0.57 * a1
        b1 = 0.0
        b2 = 0.95
        c1 = a1 * alphstr^b1
        c2 = a2 * alphstr^b2
        d1 = b1 * (dsdum - 1.0) + 1.0
        d2 = b2 * (dsdum - 1.0) + 1.0
    else
        # Prolate spheroid (lines 3505-3514)
        a1 = 0.5714285
        a2 = 0.75 * a1
        b1 = -1.0
        b2 = -0.18
        c1 = a1 * alphstr^(b1 + 1.0)
        c2 = a2 * alphstr^(b2 + 1.0)
        d1 = b1 * (dsdum - 1.0) + dsdum
        d2 = b2 * (dsdum - 1.0) + dsdum
    end

    if dsdum == 1.0 && NU == 1.0
        # Sphere special case, line 3517-3519: capacitance = ani (radius)
        gammad1 = gamma(NU + 1.0)
        return ani * gammad1 * i_gammnu
    else
        # lines 3520-3530 (both dsdum<=1 and dsdum>1 branches, identical formula)
        gammad1 = gamma(NU + d1)
        gammad2 = gamma(NU + d2)
        return c1 * ani^d1 * gammad1 * i_gammnu + c2 * ani^d2 * gammad2 * i_gammnu
    end
end

# ────────────────────────────────────────────────────────────────────────────
# 4. get_igr -- inherent growth ratio vs temperature (lines 3953-3982)
# ────────────────────────────────────────────────────────────────────────────

"""
    get_igr(igrdata::AbstractVector{Float64}, temp::Float64) -> Float64

Linear interpolation of the inherent growth ratio (IGR) vs temperature `temp`
[K] (Lamb and Scott 1972; Chen and Lamb 1994), using the 60-point
[`ISHMAEL_IGRDATA`](@ref) table (`temp - T0` from -1 C down to -60 C, one
point per degree). Ports `get_igr`, lines 3953-3982, verbatim, including its
branch for `-1 C < temp-T0 <= 0 C` (igr1 = 1.0, i.e. no habit growth yet)
and its clamp to `igrdata[60]` beyond -60 C.
"""
@inline function get_igr(igrdata::AbstractVector{Float64}, temp::Float64)
    T0 = 273.15
    dT = temp - T0
    if dT >= -59.0 && dT <= -1.0
        n = trunc(Int, dT)                        # Fortran INT() truncates toward zero
        dum  = (abs(Float64(n)) + 1.0) - abs(dT)
        igr1 = igrdata[max(n * (-1), 1)]
        igr2 = igrdata[min(n * (-1) + 1, 60)]
        return dum * igr1 + (1.0 - dum) * igr2
    elseif dT > -1.0 && dT <= 0.0
        dum = 1.0 - abs(dT)
        igr1 = 1.0
        igr2 = igrdata[1]
        return dum * igr1 + (1.0 - dum) * igr2
    elseif dT >= -60.0 && dT < -59.0
        dum = 60.0 - abs(dT)
        igr1 = igrdata[59]
        igr2 = igrdata[60]
        return dum * igr1 + (1.0 - dum) * igr2
    elseif dT < -60.0
        return igrdata[60]
    else
        return 1.0
    end
end

# ────────────────────────────────────────────────────────────────────────────
# 5. ishmael_var_check -- ice-moment consistency check (lines 3101-3196)
# ────────────────────────────────────────────────────────────────────────────

"""
    ishmael_var_check(NU, ao, fourthirdspi, gammnu, qidum, dsdum, ani, cni,
                       rbdum, nidum, aidum, cidum; RHOI=920.0) -> NamedTuple

Pure port of `var_check` (lines 3101-3196): given one ice species' raw
moments (mass `qidum`, number `nidum`, area-axis moment `aidum`,
length-axis moment `cidum`) and derived quantities (shape `dsdum`, axis
scales `ani`/`cni`, bulk density `rbdum`), applies the Fortran's sequence
of physical-consistency clamps and re-derivations and returns the
EFFECTIVE values as a `NamedTuple` -- nothing is mutated in place (unlike
the Fortran's `INTENT(INOUT)` arguments).

Clamp sequence, each branch cited by its Fortran line range:
- deltastr (`dsdum`) clamped to `[0.55, 1.3]` (lines 3113-3123), re-deriving
  `ani`/`cni`/`aidum`/`cidum` from whichever of `ani`,`cni` was NOT clamped.
- bulk density `rbdum` clamped to `[50, RHOI]` kg/m^3 (lines 3131-3154).
- mean radius `rni` floored at 2 micron (lines 3156-3169).
- max axis length capped at 1 mm (lines 3171-3193), re-deriving `rni`
  from the (number-weighted) 8 mm / (mass-weighted, spherical) 14 mm
  large-ice limit.

The Fortran's `gamma_tab(gi)` fast-path lookup (built from a fixed
`gamma_arg = NU+2+dsdum` via a table index `gi`, lines 3128-3129, 3134-3135)
is replaced by a single direct `SpecialFunctions.gamma(gamma_arg)` call --
the table was purely a speed hack; the port drops `gamma_tab` entirely (see
module docstring).
"""
function ishmael_var_check(NU::Float64, ao::Float64, fourthirdspi::Float64, gammnu::Float64,
                            qidum::Float64, dsdum::Float64, ani::Float64, cni::Float64,
                            rbdum::Float64, nidum::Float64, aidum::Float64, cidum::Float64;
                            RHOI::Float64 = 920.0)
    # Deltastr check (lines 3113-3123)
    if dsdum < 0.55
        dsdum = 0.55
        ani = (cni / (ao^(1.0 - dsdum)))^(1.0 / dsdum)
        aidum = ani^2 * cni * nidum
        cidum = cni^2 * ani * nidum
    elseif dsdum > 1.3
        dsdum = 1.3
        cni = ao^(1.0 - dsdum) * ani^dsdum
        aidum = ani^2 * cni * nidum
        cidum = cni^2 * ani * nidum
    end

    alphstr = ao^(1.0 - dsdum)                 # line 3125
    alphv = fourthirdspi * alphstr              # line 3126
    betam = 2.0 + dsdum                         # line 3127
    gamma_arg = NU + 2.0 + dsdum                # line 3128
    gam = gamma(gamma_arg)                      # replaces gamma_tab(gi), lines 3129/3134-3135

    # Ice density check: keep rbdum in [50, RHOI] (lines 3131-3154)
    if ani > 2.0e-6
        rbdum = qidum * gammnu / (nidum * alphv * ani^betam * gam)
    else
        rbdum = RHOI
    end

    if rbdum > RHOI
        rbdum = RHOI
        ani = ((qidum * gammnu) / (rbdum * nidum * alphv * gam))^(1.0 / betam)
        cni = ao^(1.0 - dsdum) * ani^dsdum
        aidum = ani^2 * cni * nidum
        cidum = cni^2 * ani * nidum
    elseif rbdum < 50.0
        rbdum = 50.0
        ani = ((qidum * gammnu) / (rbdum * nidum * alphv * gam))^(1.0 / betam)
        cni = ao^(1.0 - dsdum) * ani^dsdum
        aidum = ani^2 * cni * nidum
        cidum = cni^2 * ani * nidum
    end

    # Small ice limit: rni >= 2 micron (lines 3156-3169)
    rni = (qidum * 3.0 / (nidum * rbdum * 4.0 * ISHMAEL_PI * (gam / gammnu)))^0.333333333333
    if rni < 2.0e-6
        rni = 2.0e-6
        nidum = 3.0 * qidum * gammnu / (4.0 * ISHMAEL_PI * rbdum * rni^3 * gam)
        ani = ((qidum * gammnu) / (rbdum * nidum * alphv * gam))^(1.0 / betam)
        cni = ao^(1.0 - dsdum) * ani^dsdum
        aidum = ani^2 * cni * nidum
        cidum = cni^2 * ani * nidum
    end

    # Large ice limit: max axis <= 1 mm (lines 3171-3193)
    maxsize = max(ani, cni)
    if maxsize > 1.0e-3
        if ani >= cni
            ani = 1.0e-3
            nidum = qidum * gammnu / (fourthirdspi * rbdum * ao^(1.0 - dsdum) * ani^(2.0 + dsdum) * gam)
            cni = ao^(1.0 - dsdum) * ani^dsdum
            aidum = ani^2 * cni * nidum
            cidum = cni^2 * ani * nidum
        else
            cni = 1.0e-3
            ani = (cni / (ao^(1.0 - dsdum)))^(1.0 / dsdum)
            nidum = qidum * gammnu / (fourthirdspi * rbdum * ao^(1.0 - dsdum) * ani^(2.0 + dsdum) * gam)
            aidum = ani^2 * cni * nidum
            cidum = cni^2 * ani * nidum
        end
        rni = (qidum / (nidum * rbdum * fourthirdspi * (gam / gammnu)))^0.333333333333
    end

    return (deltastr = dsdum, ani = ani, cni = cni, rni = rni, rhobar = rbdum,
            ni = nidum, ai = aidum, ci = cidum,
            alphstr = alphstr, alphv = alphv, betam = betam)
end

# ────────────────────────────────────────────────────────────────────────────
# 6. mkcoltb / xjnum -- aggregation collection-table builder
#    (lines 4546-4630), incomplete-gamma family (4635-4740, replaced),
#    AVINT quadrature (4745-4837, replaced)
# ────────────────────────────────────────────────────────────────────────────

"""
    ishmael_gammap(a, x) -> Float64

`P(a,x)`, the lower regularized incomplete gamma function. Replaces the
Fortran's `gammap`/`lowgseries`/`highgcontfrac` (lines 4650-4740) with
`SpecialFunctions.gamma_inc`, which returns `(p, q) = (P(a,x), Q(a,x))` in
the same lower/upper regularized convention (`p + q = 1`) as the Fortran
routines it replaces.
"""
@inline ishmael_gammap(a::Float64, x::Float64) = gamma_inc(a, x)[1]

"""
    ishmael_gammaq(a, x) -> Float64

`Q(a,x) = 1 - P(a,x)`, the upper regularized incomplete gamma function.
Replaces the Fortran `GAMMQ` (lines 4635-4646); see [`ishmael_gammap`](@ref).
"""
@inline ishmael_gammaq(a::Float64, x::Float64) = gamma_inc(a, x)[2]

"""
    xjnum(dx, cvx, pvx, cvy, pvy, vny, dnx, dny, xnu, ynu,
          gyn1, gyn2, gynp, gynp1, gynp2) -> Float64

Aggregation collection kernel: the (gamma-distribution-averaged) rate at
which category-y particles of characteristic diameter `dny` sweep out
category-x particles of diameter `dx`, per unit `dx`, via their fall-speed
difference. Ports `xjnum`, lines 4602-4630, verbatim, replacing `gammln`
with `SpecialFunctions.loggamma` and `gammap`/`gammq` with
[`ishmael_gammap`](@ref)/[`ishmael_gammaq`](@ref).
"""
@inline function xjnum(dx::Float64, cvx::Float64, pvx::Float64, cvy::Float64, pvy::Float64,
                        vny::Float64, dnx::Float64, dny::Float64, xnu::Float64, ynu::Float64,
                        gyn1::Float64, gyn2::Float64, gynp::Float64, gynp1::Float64, gynp2::Float64)
    dnxi = 1.0 / dnx
    rdx = dx * dnxi
    vx = max(cvx * dx^pvx, 1.0e-6)
    dxy = (vx / cvy)^(1.0 / pvy) / dny
    dxy = clamp(dxy, 1.0e-5, 70.0)
    ynup = ynu + pvy

    rdx >= 38.0 && return 0.0

    return exp(-rdx - loggamma(xnu) - loggamma(ynu)) * rdx^(xnu - 1.0) * dnxi * (
        vx * (dx * dx * (ishmael_gammap(ynu, dxy) - ishmael_gammaq(ynu, dxy))
              + 2.0 * dx * dny * gyn1 * (ishmael_gammap(ynu + 1.0, dxy) - ishmael_gammaq(ynu + 1.0, dxy))
              + dny * dny * gyn2 * (ishmael_gammap(ynu + 2.0, dxy) - ishmael_gammaq(ynu + 2.0, dxy)))
        - vny * (dx * dx * gynp * (ishmael_gammap(ynup, dxy) - ishmael_gammaq(ynup, dxy))
                 + 2.0 * dx * dny * gynp1 * (ishmael_gammap(ynup + 1.0, dxy) - ishmael_gammaq(ynup + 1.0, dxy))
                 + dny * dny * gynp2 * (ishmael_gammap(ynup + 2.0, dxy) - ishmael_gammaq(ynup + 2.0, dxy)))
    )
end

"""
    ishmael_avint(x, y, xlo, xup) -> Float64

Overlapping-parabola quadrature, integrating tabulated `(x[i],y[i])` data
from `xlo` to `xup`. For each interior node `i` (2 <= i <= n-1) fits the
unique quadratic through `(x[i-1],y[i-1]), (x[i],y[i]), (x[i+1],y[i+1])`,
averages each pair of quadratics that share a `[x[i],x[i+1]]` interval, and
integrates the resulting piecewise-quadratic analytically.

Replaces the SLATEC `AVINT` subroutine (lines 4745-4837) that `mkcoltb`
uses to integrate the `xjnum` collection kernel over the tabulated
size-bin grid. This is a direct transliteration (same overlapping-parabola
algorithm, same node placement, same closed-form antiderivative), not a
different quadrature rule -- see `test/test_ishmael_tables.jl` for the
rtol=1e-6 cross-check against a smooth test integrand. Requires `x` strictly
increasing and at least 2 points; for `xlo == x[1]` and `xup == x[end]`
(the only case `mkcoltb` ever uses) this reduces to `istart=2`,
`istop=n-1`, matching the Fortran's full-range branch.
"""
function ishmael_avint(x::AbstractVector{Float64}, y::AbstractVector{Float64},
                        xlo::Float64, xup::Float64)
    n = length(x)
    length(y) == n || throw(ArgumentError("ishmael_avint: x and y must have the same length"))

    if xlo > xup
        error("ishmael_avint: upper limit of integration not greater than lower limit")
    elseif xlo == xup
        return 0.0
    end
    n >= 2 || error("ishmael_avint: less than 2 function values were supplied")
    @inbounds for i in 2:n
        x[i] > x[i-1] || error("ishmael_avint: abscissas not strictly increasing")
    end

    if n < 3
        slope = (y[2] - y[1]) / (x[2] - x[1])
        fl = y[1] + slope * (xlo - x[1])
        fr = y[2] + slope * (xup - x[2])
        return 0.5 * (fl + fr) * (xup - xlo)
    end

    (x[n-2] < xlo) && error("ishmael_avint: less than 3 function values between integration limits")
    (x[3] > xup) && error("ishmael_avint: less than 3 function values between integration limits")

    i = 1
    while !(x[i] >= xlo)
        i += 1
    end
    inlft = i
    i = n
    while !(x[i] <= xup)
        i -= 1
    end
    inrt = i
    (inrt - inlft) < 2 && error("ishmael_avint: less than 3 function values between integration limits")

    istart = inlft == 1 ? 2 : inlft
    istop  = inrt == n ? n - 1 : inrt

    syl = xlo
    syl2 = syl * syl
    syl3 = syl2 * syl
    ca = cb = cc = 0.0
    total = 0.0

    @inbounds for i in istart:istop
        x1, x2, x3 = x[i-1], x[i], x[i+1]
        x12 = x1 - x2
        x13 = x1 - x3
        x23 = x2 - x3
        term1 =  y[i-1] / (x12 * x13)
        term2 = -y[i]   / (x12 * x23)
        term3 =  y[i+1] / (x13 * x23)
        a = term1 + term2 + term3
        b = -(x2 + x3) * term1 - (x1 + x3) * term2 - (x1 + x2) * term3
        c = x2 * x3 * term1 + x1 * x3 * term2 + x1 * x2 * term3
        if i == istart
            ca, cb, cc = a, b, c
        else
            ca = 0.5 * (a + ca)
            cb = 0.5 * (b + cb)
            cc = 0.5 * (c + cc)
        end
        syu = x2
        syu2 = syu * syu
        syu3 = syu2 * syu
        total += ca * (syu3 - syl3) / 3.0 + cb * 0.5 * (syu2 - syl2) + cc * (syu - syl)
        ca, cb, cc = a, b, c
        syl, syl2, syl3 = syu, syu2, syu3
    end

    syu = xup
    return total + ca * (syu^3 - syl3) / 3.0 + cb * 0.5 * (syu^2 - syl2) + cc * (syu - syl)
end

"""
    mkcoltb() -> (coltab::Array{Float64,3}, coltabn::Array{Float64,3})

Builds the `(ndn,ndn,npair) = (60,60,35)` ice-ice aggregation collection
lookup tables: `coltab` is mass mixing-ratio collection, `coltabn` is number
mixing-ratio collection, for every category pair `(ix,iy)` with
`ISHMAEL_IPAIR[ix,iy] > 0`. Ports `mkcoltb`, lines 4546-4597, over the fixed
7-category property tables (lines 90-120; `ISHMAEL_IPAIR`, `ISHMAEL_CFMAS`,
etc. above), using kernel [`xjnum`](@ref) and quadrature
[`ishmael_avint`](@ref) in place of the Fortran's SLATEC `AVINT`.
"""
function mkcoltb()
    ncat, npair, ndn, ndx = 7, 35, 60, 20

    coltab  = zeros(Float64, ndn, ndn, npair)
    coltabn = zeros(Float64, ndn, ndn, npair)

    dx = Vector{Float64}(undef, ndx)
    gx = Vector{Float64}(undef, ndx)
    fx = Vector{Float64}(undef, ndx)

    for ix in 1:ncat, iy in 1:ncat
        pidx = ISHMAEL_IPAIR[ix, iy]
        pidx > 0 || continue

        gnux, gnuy   = ISHMAEL_GNU[ix], ISHMAEL_GNU[iy]
        pwvtx, pwvty = ISHMAEL_PWVT[ix], ISHMAEL_PWVT[iy]
        cfvtx, cfvty = ISHMAEL_CFVT[ix], ISHMAEL_CFVT[iy]

        gyn   = gamma(gnuy)
        gyn1  = gamma(gnuy + 1.0) / gyn
        gyn2  = gamma(gnuy + 2.0) / gyn
        gynp  = gamma(gnuy + pwvty) / gyn
        gynp1 = gamma(gnuy + pwvty + 1.0) / gyn
        gynp2 = gamma(gnuy + pwvty + 2.0) / gyn

        dxlo = ISHMAEL_TABLO[ix] * 0.01
        dxhi = ISHMAEL_TABHI[ix] * 10.0
        @inbounds for idx in 1:ndx
            dx[idx] = dxlo * (dxhi / dxlo)^(Float64(idx - 1) / Float64(ndx - 1))
        end

        for idny in 1:ndn
            dny = ISHMAEL_TABLO[iy] * (ISHMAEL_TABHI[iy] / ISHMAEL_TABLO[iy])^(Float64(idny - 1) / Float64(ndn - 1))
            vny = cfvty * dny^pwvty
            for idnx in 1:ndn
                dnx = ISHMAEL_TABLO[ix] * (ISHMAEL_TABHI[ix] / ISHMAEL_TABLO[ix])^(Float64(idnx - 1) / Float64(ndn - 1))
                @inbounds for idx in 1:ndx
                    fval = xjnum(dx[idx], cfvtx, pwvtx, cfvty, pwvty, vny, dnx, dny,
                                 gnux, gnuy, gyn1, gyn2, gynp, gynp1, gynp2)
                    fval = fval < 1.0e-15 ? 0.0 : fval          # low limit added by AAJ
                    fx[idx] = fval
                    gx[idx] = fval * ISHMAEL_CFMAS[ix] * dx[idx]^ISHMAEL_PWMAS[ix]
                end
                coltab[idnx, idny, pidx]  = max(0.0, ishmael_avint(dx, gx, dxlo, dxhi))
                coltabn[idnx, idny, pidx] = max(0.0, ishmael_avint(dx, fx, dxlo, dxhi))
            end
        end
    end

    return coltab, coltabn
end
