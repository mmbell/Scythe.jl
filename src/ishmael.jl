# src/ishmael.jl
#
# ISHMAEL ice-microphysics process rates, part 1 (Stage S6a).
#
# Fortran source of truth:
#   /Users/mmbell/Development/cm1r21.1/src/module_mp_jensen_ishmael.F
#
# This file ports the vapor-growth/fall-speed/ventilation block
# (`vaporgrow`, lines 3203-3481, split into pieces), the nucleation set
# (DeMott 2010 deposition/condensation freezing, homogeneous freezing,
# Bigg 1953 rain freezing, Hallett-Mossop rime splintering -- all inside
# `me_ishmael`/`mp_jensen_ishmael`, lines 1509-1625), and the 5x-duplicated
# rain-DSD helper (lines 1113-1125 etc). It REUSES the utility layer
# already shipped in src/ishmael_tables.jl (`capacitance_gamma`,
# `get_igr`, `ishmael_var_check`, `access_lookup_table`,
# `ISHMAEL_IGRDATA`) rather than duplicating any of it. It does NOT port
# riming, aggregation, melting-driven precipitation accounting, or the
# density<->mixing-ratio conversion at the host-model interface -- those
# (plus the deposition-partition seam noted below) are later stages.
#
# All functions here are PURE (no mutation, no globals) and take plain
# Float64 scalars, matching the Fortran subroutines' own scalar-per-call
# structure. `@inline` is used on the small hot-path pieces
# (`ishmael_fall_speeds`, `ishmael_ventilation`, `ishmael_rain_lambda`)
# since they involve no allocation and are cheap enough to always inline;
# `ishmael_deposition_partition` and `ishmael_vapor_coefficients` are left
# to the compiler's own inlining heuristics (still allocation-free) since
# they are large enough that forcing inlining everywhere could bloat code
# size without benefit.
#
# Every ported formula is transcribed EXACTLY from the Fortran, including
# its own single-precision-flavored literals (e.g. `PI = 3.14159265`,
# reused from `ISHMAEL_PI` in ishmael_tables.jl) -- these are not
# independent physics choices, just what the Fortran wrote.

using SpecialFunctions: gamma, loggamma

# ────────────────────────────────────────────────────────────────────────────
# Shared local constants (Fortran module-level parameters not already
# defined in ishmael_tables.jl; see module_mp_jensen_ishmael.F lines 35-47,
# 628-640).
# ────────────────────────────────────────────────────────────────────────────

const ISHMAEL_G_HOME = 9.8          # gravity, m s^-2 (line 37)
const ISHMAEL_RHOI    = 920.0       # bulk ice density, kg m^-3 (line 46)
const ISHMAEL_RHOW    = 1000.0      # water density, kg m^-3 (line 43)
const ISHMAEL_T0       = 273.15     # STP temperature, K (line 44)
const ISHMAEL_CPW      = 4218.0     # heat capacity of water, J kg^-1 K^-1 (line 41)
const ISHMAEL_NU        = 4.0       # ice-distribution shape parameter (line 628); kept
                                     # as a named default, NOT hardcoded into formulas below,
                                     # so callers using a different NU (there should never be
                                     # one -- "If changed, build new lookup tables") still pass
                                     # it explicitly where the Fortran signature has it as an
                                     # argument.
const ISHMAEL_QSMALL   = 1.0e-12    # smallest ice mass, kg kg^-1 (line 631)
const ISHMAEL_QNSMALL  = 1.25e-7    # smallest ice number, # kg^-1 (line 632)
const ISHMAEL_LAMMINR  = 1.0 / 2800.0e-6   # min rain slope parameter, m^-1 (line 640)
const ISHMAEL_LAMMAXR  = 1.0 / 20.0e-6     # max rain slope parameter, m^-1 (line 639)

# ────────────────────────────────────────────────────────────────────────────
# 1a. ishmael_fall_speeds -- Best-number/Mitchell-Heymsfield(2005) fall
#     speeds (lines 3236-3296 inside vaporgrow, duplicated at 2623-2694 as
#     the "official" post-aggregation recomputation that mp_jensen_ishmael
#     actually keeps -- ported ONCE here from that shared formula).
# ────────────────────────────────────────────────────────────────────────────

"""
    ishmael_fall_speeds(ani, cni, dsdum, NU, i_gammnu, alphstr, rbdum, rhoair, mu;
                         in_melting::Bool=false) -> NamedTuple

Number-averaged (`vtrni1`), mass-averaged (`vtrmi1`), and
reflectivity-weighted (`vtrzi1`) ice fall speeds, plus the Reynolds number
`Nre` (needed by [`ishmael_ventilation`](@ref)). Ports the
Best-number/Mitchell-Heymsfield(2005) fall-speed block that appears twice
in the Fortran with identical formulas -- inside `vaporgrow` (lines
3236-3296) and again, as the value `mp_jensen_ishmael` actually keeps
after aggregation updates `ani`/`deltastr` (lines 2623-2694) -- ported
once here.

The two Fortran copies differ in one respect the port preserves exactly:
only the SECOND copy (2623-2694, the one whose output the host model
actually uses) applies the `min(...,25.)` fall-speed cap and the
melting/size-sorting override (`vtrni1 := vtrmi1` when
`temp>T0 .and. qmlt<0`, lines 2691-2694); `vaporgrow`'s own internal copy
(3236-3296) is uncapped and only used inside its own growth-solution math.
`vtrzi1` is NEVER recomputed by the capped/overridden copy (it only
touches `vtrni1(cc,k)`/`vtrmi1(cc,k)`) -- the value the host model
actually keeps for `vtrzi1` is therefore whatever `vaporgrow` produced,
UNCAPPED. This function reproduces that: `vtrni1`/`vtrmi1` are capped at
25 m/s (and `vtrni1` is swapped for `vtrmi1` under `in_melting`), while
`vtrzi1` is returned uncapped, matching what actually reaches the host
model's state in each case.

`alphstr = ao^(1-dsdum)` (the caller already has this from
[`ishmael_var_check`](@ref)/[`capacitance_gamma`](@ref)'s own internal
computation).
"""
@inline function ishmael_fall_speeds(ani::Float64, cni::Float64, dsdum::Float64, NU::Float64,
                                      i_gammnu::Float64, alphstr::Float64, rbdum::Float64,
                                      rhoair::Float64, mu::Float64; in_melting::Bool=false)
    # phii: axis-ratio-like shape diagnostic (gamma_arg = NU-1+dsdum, lines
    # 2628-2630 / 3230-3232).
    phii = cni / ani * gamma(NU - 1.0 + dsdum) * i_gammnu

    local bl, al, aa, ba, qe
    if phii < 1.0
        bl, al, aa, ba = 1.0, 2.0, ISHMAEL_PI, 2.0
        qe = (1.0 - phii) * (rbdum / ISHMAEL_RHOI) + phii
    elseif phii > 1.0
        al, bl, aa, ba = 2.0, 1.0, ISHMAEL_PI * alphstr, dsdum + 1.0
        qe = 1.0
    else
        bl, al, aa, ba = 1.0, 2.0, ISHMAEL_PI, 2.0
        qe = 1.0
    end
    qe = min(qe, 1.0)

    fourthirdspi = 4.0 / 3.0 * ISHMAEL_PI
    xn = 2.0 / rbdum * (rbdum - rhoair) * ISHMAEL_G_HOME * rhoair / mu^2 *
         (fourthirdspi * rbdum) * alphstr * al^2 / (aa * qe) * qe^(3.0 / 4.0)
    bx = dsdum + 2.0 + 2.0 * bl - ba

    # Number-averaged Best number
    xm = xn * ani^bx * gamma(NU + bx) * i_gammnu

    # Mitchell-Heymsfield (2005) fall-speed coefficients
    f_c1 = 4.0 / (5.83 * 5.83 * sqrt(0.6))
    f_c2 = (5.83 * 5.83) / 4.0
    s = sqrt(1.0 + f_c1 * sqrt(xm))
    bm = (f_c1 * sqrt(xm)) / (2.0 * (s - 1.0) * s) - (1.0e-5 * xm) / (f_c2 * (s - 1.0)^2)
    am = ((f_c2 * (s - 1.0)^2) - 1.0e-5 * xm) / xm^bm

    if xm > 1.0e8
        am = 1.0865
        bm = 0.499
    end

    Nre = am * xm^bm

    common = mu / rhoair * 0.5 * am * xn^bm * ani^(bx * bm - 1.0)

    vtrni1_raw = min(common * gamma(NU + bx * bm - 1.0) * i_gammnu, 25.0)
    vtrmi1_raw = min(common * gamma(NU + bx * bm - 1.0 + 2.0 + dsdum) /
                      gamma(NU + 2.0 + dsdum), 25.0)
    vtrzi1 = common * exp(loggamma(NU + bx * bm - 1.0 + 4.0 + 2.0 * dsdum) -
                           loggamma(NU + 4.0 + 2.0 * dsdum))

    vtrni1 = in_melting ? vtrmi1_raw : vtrni1_raw

    return (vtrni1=vtrni1, vtrmi1=vtrmi1_raw, vtrzi1=vtrzi1, Nre=Nre)
end

# ────────────────────────────────────────────────────────────────────────────
# 1b. ishmael_ventilation -- ventilation coefficients (lines 3298-3320)
# ────────────────────────────────────────────────────────────────────────────

"""
    ishmael_ventilation(nsch, npr, Nre) -> NamedTuple{(:fv, :fh)}

Vapor (`fv`) and heat (`fh`) ventilation coefficients from the Reynolds
number `Nre` (see [`ishmael_fall_speeds`](@ref)), Schmidt number `nsch`,
and Prandtl number `npr`. Ports the ventilation block of `vaporgrow`,
lines 3298-3320, verbatim (Hall and Pruppacher-style two-regime
piecewise fit).
"""
@inline function ishmael_ventilation(nsch::Float64, npr::Float64, Nre::Float64)
    xvent = nsch^(1.0 / 3.0) * Nre^0.5
    ntherm = Nre^0.5 * npr^(1.0 / 3.0)

    local bv1, bv2, gv
    if xvent <= 1.0
        bv1, bv2, gv = 1.0, 0.14, 2.0
    else
        bv1, bv2, gv = 0.86, 0.28, 1.0
    end

    local bt1, bt2, gt
    if ntherm < 1.4
        bt1, bt2, gt = 1.0, 0.108, 2.0
    else
        bt1, bt2, gt = 0.78, 0.308, 1.0
    end

    fvdum = bv1 + bv2 * xvent^gv
    fhdum = bt1 + bt2 * ntherm^gt

    return (fv=fvdum, fh=fhdum)
end

# ────────────────────────────────────────────────────────────────────────────
# 1c. ishmael_vapor_coefficients -- composes capacitance_gamma + fall
#     speeds + ventilation into what the host needs to form the
#     deposition timescale.
# ────────────────────────────────────────────────────────────────────────────

"""
    ishmael_vapor_coefficients(ani, cni, dsdum, NU, i_gammnu, alphstr,
                                rbdum, rhoair, mu, nsch, npr; in_melting=false)
        -> NamedTuple{(:Cbar, :fv, :fh, :vtrni1, :vtrmi1, :vtrzi1)}

Composes [`capacitance_gamma`](@ref) (ishmael_tables.jl) with
[`ishmael_fall_speeds`](@ref) and [`ishmael_ventilation`](@ref) into the
set of quantities the host model needs to form the deposition timescale
(`Cbar/rni` is `vaporgrow`'s `fs`, line 3233) without needing the excluded
sui/maxsui growth-solution math -- see [`ishmael_deposition_partition`](@ref)
for where that boundary is drawn. Mirrors how `mp_jensen_ishmael` computes
`capgam` (line 1258) immediately before calling `vaporgrow`, and how
`vaporgrow` immediately computes the fall speeds/ventilation before its
growth solution (lines 3236-3320).
"""
function ishmael_vapor_coefficients(ani::Float64, cni::Float64, dsdum::Float64, NU::Float64,
                                     i_gammnu::Float64, alphstr::Float64, rbdum::Float64,
                                     rhoair::Float64, mu::Float64, nsch::Float64, npr::Float64;
                                     in_melting::Bool=false)
    fs = ishmael_fall_speeds(ani, cni, dsdum, NU, i_gammnu, alphstr, rbdum, rhoair, mu;
                              in_melting=in_melting)
    vent = ishmael_ventilation(nsch, npr, fs.Nre)
    capgam = capacitance_gamma(ani, dsdum, NU, alphstr, i_gammnu)

    return (Cbar=capgam, fv=vent.fv, fh=vent.fh,
            vtrni1=fs.vtrni1, vtrmi1=fs.vtrmi1, vtrzi1=fs.vtrzi1)
end

# ────────────────────────────────────────────────────────────────────────────
# 1d. ishmael_deposition_partition -- shape/density evolution given a
#     REALIZED mass increment (lines 3337-3471 of vaporgrow).
# ────────────────────────────────────────────────────────────────────────────

"""
    ishmael_deposition_partition(dt, ani, cni, rni, dsdum, rbdum, nidum, igr,
                                  afn, maxsui, vtbranch, sui_negative, capgam, dv,
                                  temp, ao, NU, gammnu, i_gammnu, fourthirdspi;
                                  T0=273.15, RHOI=920.0) -> NamedTuple

Given a REALIZED vapor growth/sublimation rate `afn` and the deposition-
density selector `maxsui`, applies ISHMAEL's habit-dependent shape/density
evolution -- deposition density `rhodep`, new characteristic radius `rnf`
and a-axis `anf`, the sublimation shape-clamp safety logic, small-particle
floors, the updated shape parameter `dsdumout`, and the resulting c-axis
`cnf` -- exactly as `vaporgrow` does from the point it has `afn` in hand
through the end of the T<=T0 branch (lines 3337-3471), plus the axis-rate
diagnostics `ard`/`crd` that the host loop derives immediately afterward
(lines 1271-1272). Returns `(anf, cnf, rnf, iwcf, rdout, dsdumout, rbdum,
ard, crd)`.

# THE SEAM (why `afn`/`maxsui`/`sui_negative` are inputs, not computed here)

Per the Stage S6a task, `vaporgrow`'s own growth-EQUATION solution (lines
3327-3368: diagnosing `maxsui` from the Fortran's `sui`/`sup`/`qvi`/`qvs`,
then solving for `afn` via `del1`/`del2`/`alpha`, which additionally folds
in latent heating from riming) is NOT ported -- Scythe's host model
carries a PROGNOSTIC supersaturation variable (Q_ss) instead of
diagnosing `sui`/`sup` from `qv`/`qvs`/`qvi`, so it must compute its own
realized growth rate consistent with its own Q_ss formulation and hand
the result in here. This function begins exactly where that excluded math
ends: it takes the realized `afn` (units as in the Fortran's local `afn`)
and the three small pieces of information the excluded math would
otherwise have supplied --
  - `maxsui` selects which of `{RHOI*igr, RHOI/igr, RHOI}` the vapor-growth
    deposition density blends toward (lines 3338-3350); it is a pure
    number in [0,1] and does not itself require reproducing the
    diagnosed-supersaturation pathway to compute from Q_ss-consistent
    quantities.
  - `vtbranch` (`= vtbarbm`, the mass-averaged fall speed from
    [`ishmael_fall_speeds`](@ref)) gates the planar branching-size test at
    line 3339-3340.
  - `sui_negative` (`= sui<0`) is the ONE boolean the shape-clamp safety
    logic (lines 3396-3422) needs from the excluded diagnosed-`sui`
    pathway; it is passed through as a flag rather than recomputing `sui`
    itself.

This IS a faithful factorization of the Fortran, not an approximation of
it: given the same `afn`/`maxsui`/`vtbranch`/`sui_negative` a Fortran
`vaporgrow` call would have produced internally, this function reproduces
`vaporgrow`'s axis/density/shape outputs bit-for-bit (up to the
`gamma_tab`-vs-`SpecialFunctions.gamma` substitution already used
throughout ishmael_tables.jl). The orchestrator should review that this
seam placement is where Scythe's Q_ss-driven integration stage will want
to plug in.

`gammnu`, `i_gammnu`, `fourthirdspi` are the caller's already-computed
`gamma(NU)`, `1/gamma(NU)`, `4/3*PI` (as in `ishmael_var_check`).
"""
function ishmael_deposition_partition(dt::Float64, ani::Float64, cni::Float64, rni::Float64,
                                       dsdum::Float64, rbdum::Float64, nidum::Float64, igr::Float64,
                                       afn::Float64, maxsui::Float64, vtbranch::Float64,
                                       sui_negative::Bool, capgam::Float64, dv::Float64, temp::Float64,
                                       ao::Float64, NU::Float64, gammnu::Float64, i_gammnu::Float64,
                                       fourthirdspi::Float64; T0::Float64=ISHMAEL_T0,
                                       RHOI::Float64=ISHMAEL_RHOI)
    QASMALL_LOCAL = 1.0e-19   # vaporgrow's own local QASMALL (line 3212), distinct from
                              # var_check's module-scope QASMALL=1e-24.

    # iwci: initial ice water content (density-weighted), reproduced from
    # (nidum, rbdum, rni, dsdum) exactly as the host loop constructs it
    # before calling vaporgrow (lines 1247-1251): vi = fourthirdspi*rni^3*
    # gamma_tab(gi)/gammnu with gi from gamma_arg=NU+2+deltastr, then
    # iwci = nidum*rbdum*vi (nidum there is already ni(cc,k)*rhoair(k), the
    # same convention as vaporgrow's own `nidum` argument).
    gam_iwc = gamma(NU + 2.0 + dsdum)
    iwci = nidum * rbdum * fourthirdspi * rni^3 * (gam_iwc / gammnu)

    if temp > T0
        # Lines 3473-3479: T > T0 passthrough -- vaporgrow does not grow/
        # sublimate ice above freezing (the host's melting routine handles
        # it instead). ard/crd are correspondingly zero (no axis change).
        return (anf=ani, cnf=cni, rnf=rni, iwcf=iwci, rdout=rbdum, dsdumout=dsdum,
                rbdum=rbdum, ard=0.0, crd=0.0)
    end

    fs = capgam / rni                              # line 3233
    alphanr = ani / rni^(3.0 / (2.0 + igr))         # line 3234
    alphstr = ao^(1.0 - dsdum)
    phii = cni / ani * gamma(NU - 1.0 + dsdum) * i_gammnu   # line 3232 (recomputed; geometric only)

    # Deposition density selection (lines 3337-3353)
    local rhodep
    if igr <= 1.0
        if vtbranch > 0.0
            if ani > sqrt((dv * ISHMAEL_PI * 2.0 * cni) / (vtbranch * NU))
                rhodep = (RHOI * igr) * maxsui + RHOI * (1.0 - maxsui)
            else
                rhodep = RHOI
            end
        else
            rhodep = RHOI
        end
    else
        rhodep = (RHOI / igr) * maxsui + RHOI * (1.0 - maxsui)
    end
    rhodep = min(rhodep, 700.0)   # Cotton et al. 2012 high limit, line 3353

    # Sublimation branch overrides rhodep with a polynomial density decay
    # (lines 3371-3381) -- driven purely by the SIGN of afn, not by the
    # excluded diagnosed-sui pathway.
    if afn < 0.0
        videp = rni^3
        vmin = (10.0e-6)^3
        if vmin < videp
            betavol = log(RHOI / rbdum) / log(vmin / videp)
            rhodep = rbdum * (1.0 + betavol)
        else
            rhodep = rbdum
        end
    end
    rhodep = max(rhodep, 50.0)
    rhodep = min(rhodep, RHOI)

    gammnubet = gam_iwc   # gamma(NU+2+dsdum), same as used for iwci above (line 3387)

    # Characteristic r-axis and a-axis after growth timestep (lines 3389-3391)
    rnf = sqrt(max(rni^2 + 2.0 * afn * fs / rhodep * gammnu / gammnubet * dt, QASMALL_LOCAL))
    anf = alphanr * rnf^(3.0 / (2.0 + igr))

    # Do not let sublimation change prolate<->oblate or create extreme
    # shapes (lines 3393-3422). `sui_negative` stands in for the excluded
    # `sui<0.` test; `afn<0.` is reproduced directly.
    phif = phii * (rnf^3 / rni^3)^((igr - 1.0) / (igr + 2.0))
    if sui_negative || afn < 0.0
        if (phii > 1.0 && phif < 1.0) || (phii < 1.0 && phif > 1.0)
            phif = phii
            alphanr = ani / rni
            anf = alphanr * rnf
        end
        if phii > 1.0
            if phif > phii
                phif = phii
                alphanr = ani / rni
                anf = alphanr * rnf
            end
        else
            if phif < phii
                phif = phii
                alphanr = ani / rni
                anf = alphanr * rnf
            end
        end
    end

    vi = fourthirdspi * rni^3 * gam_iwc * i_gammnu
    vf = fourthirdspi * rnf^3 * gam_iwc * i_gammnu
    rdout = rhodep
    rbdumtmp = min(rbdum * (vi / vf) + rhodep * (1.0 - vi / vf), RHOI)
    iwcf = nidum * rbdumtmp * vf

    # deltastr update (lines 3431-3440); dsdumout defaults to dsdum (line
    # 3229) and is only overwritten here when igr != 1.
    dsdumout = dsdum
    if igr != 1.0
        if anf > 1.1 * ao
            dsdumout = (3.0 * log(rnf) - 2.0 * log(anf) - log(ao)) / (log(anf) - log(ao))
        else
            dsdumout = 1.0
        end
    end

    rbdum_out = rbdum   # unchanged unless one of the small-particle floors below fires

    # Small-particle floors during sublimation (lines 3442-3459)
    if afn < 0.0 && rnf < 1.0e-6
        rbdum_out = RHOI
        rdout = RHOI
        dsdumout = 1.0
        phif = 1.0
        alphanr = ani / rni
        anf = alphanr * rnf
    end
    if afn < 0.0 && anf < 1.0e-6
        rbdum_out = RHOI
        rdout = RHOI
        dsdumout = 1.0
        phif = 1.0
        alphanr = ani / rni
        anf = alphanr * rnf
    end

    # Sublimation check (lines 3462-3466)
    if afn < 0.0 && dsdumout <= 0.0
        dsdumout = 1.0
        anf = rnf
    end

    # C-axis after vapor growth (lines 3468-3471)
    cnf = phif * anf * gammnu / gamma(NU - 1.0 + dsdumout)

    # Axis-rate diagnostics (lines 1271-1272 of the host loop, folded in
    # here per the task spec). `nidum` here is whatever number-density
    # convention the caller passed in (the Fortran itself uses ni(cc,k)
    # [# kg^-1] for THIS formula but nim3dum=ni(cc,k)*rhoair(k) [# m^-3]
    # for vaporgrow's own `nidum` argument -- that density<->mixing-ratio
    # split is explicitly a LATER integration-stage concern per the task
    # spec, not this function's; ard/crd simply inherit whatever
    # convention `nidum` carries here).
    ard = (2.0 * (anf - ani) * cni + (cnf - cni) * ani) * ani * nidum / dt
    crd = (2.0 * (cnf - cni) * ani + (anf - ani) * cni) * cni * nidum / dt

    return (anf=anf, cnf=cnf, rnf=rnf, iwcf=iwcf, rdout=rdout, dsdumout=dsdumout,
            rbdum=rbdum_out, ard=ard, crd=crd)
end

# ────────────────────────────────────────────────────────────────────────────
# 2a. ishmael_nucleation_demott -- DeMott et al. (2010) deposition/
#     condensation-freezing nucleation (lines 1577-1601)
# ────────────────────────────────────────────────────────────────────────────

"""
    ishmael_nucleation_demott(temp, sup, dt, rhoair, ni_ice1, ni_ice2;
                               T0=273.15, RHOI=920.0) -> NamedTuple{(:mnuccd, :nnuccd)}

DeMott et al. (2010, PNAS) heterogeneous ice nucleation (deposition/
condensation freezing). Ports lines 1577-1601 verbatim, including the
"0.03 large-aerosol number" fit constant (Chagnon and Junge 1962) and the
10 L^-1 (`10000` after the `#/m^3 -> #/L` `/1000` conversion) existing-ice
number cap that throttles new nucleation once ICE1+ICE2 number is already
near that limit. New ice particles are 2 micron radius, density `RHOI`
(module `RHOI=920`).

Zero outside the nucleation window (`temp<T0 .and. sup>=0`), reproducing
the top-of-k-loop reset at line 967 (`mnuccd=0.; nnuccd=0.`) that the
Fortran relies on to zero these fields when the `if` at line 1577 doesn't
fire.
"""
function ishmael_nucleation_demott(temp::Float64, sup::Float64, dt::Float64, rhoair::Float64,
                                    ni_ice1::Float64, ni_ice2::Float64;
                                    T0::Float64=ISHMAEL_T0, RHOI::Float64=ISHMAEL_RHOI)
    if !(temp < T0 && sup >= 0.0)
        return (mnuccd=0.0, nnuccd=0.0)
    end

    a_demott = 0.0000594
    b_demott = 3.33
    c_demott = 0.0264
    d_demott = 0.0033

    inrate = a_demott * (273.16 - temp)^b_demott *
             0.03^((c_demott * (273.16 - temp)) + d_demott)
    inrate = inrate * 1000.0        # #/L -> #/m^3
    inrate = inrate / rhoair        # -> # kg^-1 (i_rhoair)

    fourthirdspi = 4.0 / 3.0 * ISHMAEL_PI
    i_dt = 1.0 / dt

    local mnuccd, nnuccd
    if (((ni_ice1 + ni_ice2 + inrate) * rhoair) / 1000.0) <= 10000.0
        mnuccd = inrate * (fourthirdspi * RHOI * (2.0e-6)^3) * i_dt
        nnuccd = inrate * i_dt
    else
        curnum = (ni_ice1 + ni_ice2) * rhoair / 1000.0
        ratel = max(0.0, 10000.0 - curnum)
        ratekg = ratel * 1000.0 / rhoair
        mnuccd = ratekg * (fourthirdspi * RHOI * (2.0e-6)^3) * i_dt
        nnuccd = ratekg * i_dt
    end

    return (mnuccd=mnuccd, nnuccd=nnuccd)
end

# ────────────────────────────────────────────────────────────────────────────
# 2b. ishmael_homogeneous_freezing -- T < -35C, freeze all qc/qr
#     (lines 1545-1553)
# ────────────────────────────────────────────────────────────────────────────

"""
    ishmael_homogeneous_freezing(temp, qc, nc, qr, nr, dt;
                                  QSMALL=1.0e-12, T0=273.15) -> NamedTuple{(:mim,:nim,:mimr,:nimr)}

Homogeneous freezing below -35C: ALL cloud water (`mim`,`nim`) and ALL
rainwater (`mimr`,`nimr`) freeze over one timestep. Ports lines 1545-1553
verbatim (the Fortran's `FREEZE_QC` namelist switch is assumed true, as it
gates a scheme on/off choice rather than physics; see the module
docstring). Mass- and number-conserving by construction: `mim=qc*i_dt`
converts the ENTIRE `qc` reservoir to ice number rate in one step, i.e.
`mim*dt == qc` and `nim*dt == nc` exactly.
"""
function ishmael_homogeneous_freezing(temp::Float64, qc::Float64, nc::Float64,
                                       qr::Float64, nr::Float64, dt::Float64;
                                       QSMALL::Float64=ISHMAEL_QSMALL, T0::Float64=ISHMAEL_T0)
    i_dt = 1.0 / dt
    below_35 = temp < (T0 - 35.0)

    mim = (qc > QSMALL && below_35) ? qc * i_dt : 0.0
    nim = (qc > QSMALL && below_35) ? nc * i_dt : 0.0
    mimr = (qr > QSMALL && below_35) ? qr * i_dt : 0.0
    nimr = (qr > QSMALL && below_35) ? nr * i_dt : 0.0

    return (mim=mim, nim=nim, mimr=mimr, nimr=nimr)
end

# ────────────────────────────────────────────────────────────────────────────
# 2d (rain DSD helper used by Bigg and referenced by riming, lines
#     1113-1125 etc). Placed before ishmael_bigg_freezing since Bigg needs it.
# ────────────────────────────────────────────────────────────────────────────

"""
    ishmael_rain_lambda(qr, nr; QNSMALL=1.25e-7) -> NamedTuple{(:lamr,:n0rr,:nr)}

Exponential (Marshall-Palmer) rain size-distribution slope `lamr` and
intercept `n0rr`, with `lamr` clamped to `[1/2800 micron, 1/20 micron]`
and `nr` re-diagnosed to stay consistent with the clamped `lamr` and the
(unchanged) mass `qr`. Ports the 5x-duplicated block at lines 1113-1125
(and again at 1556-1568, 1685ff, 2208ff, 3077ff -- all identical) verbatim.

DEVIATION from the task spec's suggested signature
`ishmael_rain_lambda(qr, nr, rhoair)`: the Fortran formula
`lamr = (PI*RHOW*nr/qr)**(1/3)` has NO `rhoair` dependence at any of its
five call sites (checked by grep across all five). `rhoair` is dropped
from the signature rather than carried as an unused, undocumented
parameter; see the Stage S6a report for this deviation.
"""
@inline function ishmael_rain_lambda(qr::Float64, nr::Float64; QNSMALL::Float64=ISHMAEL_QNSMALL)
    nr = max(nr, QNSMALL)
    lamr = (ISHMAEL_PI * ISHMAEL_RHOW * nr / qr)^(1.0 / 3.0)
    n0rr = nr * lamr

    if lamr < ISHMAEL_LAMMINR
        lamr = ISHMAEL_LAMMINR
        n0rr = lamr^4 * qr / (ISHMAEL_PI * ISHMAEL_RHOW)
        nr = n0rr / lamr
    elseif lamr > ISHMAEL_LAMMAXR
        lamr = ISHMAEL_LAMMAXR
        n0rr = lamr^4 * qr / (ISHMAEL_PI * ISHMAEL_RHOW)
        nr = n0rr / lamr
    end

    return (lamr=lamr, n0rr=n0rr, nr=nr)
end

# ────────────────────────────────────────────────────────────────────────────
# 2c. ishmael_bigg_freezing -- Bigg (1953) rain freezing below -4C
#     (lines 1556-1574)
# ────────────────────────────────────────────────────────────────────────────

"""
    ishmael_bigg_freezing(temp, qr, nr, dt; QSMALL=1.0e-12, T0=273.15)
        -> NamedTuple{(:mbiggr, :nbiggr)}

Bigg (1953) immersion freezing of rain drops below -4C. Ports lines
1556-1574 verbatim, using [`ishmael_rain_lambda`](@ref) for the rain DSD.
Both rates are additionally clamped to not exceed the available `qr`/`nr`
reservoir over one timestep (`min(...,qr*i_dt)`/`min(...,nr*i_dt)`, lines
1572-1573).
"""
function ishmael_bigg_freezing(temp::Float64, qr::Float64, nr::Float64, dt::Float64;
                                QSMALL::Float64=ISHMAEL_QSMALL, T0::Float64=ISHMAEL_T0)
    if !(qr > QSMALL && temp < (T0 - 4.0))
        return (mbiggr=0.0, nbiggr=0.0)
    end

    dsd = ishmael_rain_lambda(qr, nr)
    lamr = dsd.lamr
    nr_adj = dsd.nr
    i_dt = 1.0 / dt

    mbiggr = 20.0 * ISHMAEL_PI^2 * ISHMAEL_RHOW * 100.0 * nr_adj *
             (exp(0.66 * (T0 - temp)) - 1.0) / lamr^3 / lamr^3
    nbiggr = ISHMAEL_PI * 100.0 * nr_adj * (exp(0.66 * (T0 - temp)) - 1.0) / lamr^3

    mbiggr = min(mbiggr, qr * i_dt)
    nbiggr = min(nbiggr, nr_adj * i_dt)

    return (mbiggr=mbiggr, nbiggr=nbiggr)
end

# ────────────────────────────────────────────────────────────────────────────
# 2e. ishmael_rime_splintering -- Hallett-Mossop (1974) rime splintering
#     (lines 1603-1625)
# ────────────────────────────────────────────────────────────────────────────

"""
    ishmael_rime_splintering(temp, prdr) -> NamedTuple{(:fmult, :nmult, :qmult, :prdr)}

Hallett-Mossop (1974, Nature) secondary ice production: 350 splinters
(5 micron radius, density `RHOI`) per mg of rime accreted in the
265.16-270.16 K window, linearly weighted (`fmult`) to zero at both
window edges. `prdr` is one ice species' riming rate; returns the
splinter number/mass rates and the RESIDUAL riming rate after splinter
mass is subtracted (`prdr_new = prdr - qmult`, mirroring the per-species
Fortran loop at lines 1617-1624 called once per species by the caller).

FIX (documented per the task spec): the Fortran computes `fmult` INSIDE
an `if(temp.lt.270.16.and.temp.gt.265.16)` guard (lines 1605-1615) and
otherwise relies on `fmult` having been reset to `0.` at the top of the
k-loop (line 967) -- i.e. outside the window, `fmult` is only zero
because nothing overwrites that reset. This port makes the "zero outside
the window" an explicit `else` branch instead of an implicit fall-through
from a separate reset far away in the file, so it survives independent of
any future restructuring. The VALUE of `fmult` inside the window is
identical to the Fortran; only the "make the fallback explicit" part is a
robustness change, not a physics change.
"""
function ishmael_rime_splintering(temp::Float64, prdr::Float64)
    local fmult
    if temp < 270.16 && temp > 265.16
        if temp > 270.16          # unreachable given the outer guard; transcribed as-is
            fmult = 0.0
        elseif temp <= 270.16 && temp > 268.16
            fmult = (270.16 - temp) / 2.0
        elseif temp >= 265.16 && temp <= 268.16
            fmult = (temp - 265.16) / 3.0
        else                       # temp < 265.16; unreachable given the outer guard
            fmult = 0.0
        end
    else
        # Explicit fix: outside [265.16, 270.16], fmult is zero. The
        # Fortran achieves this implicitly via the line-967 top-of-loop
        # reset; see docstring.
        fmult = 0.0
    end

    if prdr > 0.0
        nmult = 35.0e4 * prdr * fmult * 1000.0
        qmult = nmult * (4.0 / 3.0 * ISHMAEL_PI) * ISHMAEL_RHOI * (5.0e-6)^3
        qmult = min(qmult, prdr)
        prdr_new = prdr - qmult
        return (fmult=fmult, nmult=nmult, qmult=qmult, prdr=prdr_new)
    else
        return (fmult=fmult, nmult=0.0, qmult=0.0, prdr=prdr)
    end
end

# ────────────────────────────────────────────────────────────────────────────
# 3. ishmael_melting -- lines 1509-1535, with the rhoair(cc)->rhoair(k)
#    bug fixed by construction (see docstring).
# ────────────────────────────────────────────────────────────────────────────

"""
    ishmael_melting(temp, ni, ani, cni, deltastr, rhobar, qi, ai, ci,
                     kt, fh, rhoair, xxlv, dv, fv, qs0, qv, xxlf, cpw,
                     rimetotal, dQImltri, dNmltri, dt,
                     alphstr, gammnu, i_gammnu, fourthirdspi;
                     T0=273.15) -> NamedTuple{(:qmlt,:nmlt,:amlt,:cmlt)}

Ice melting above 0C: mass loss `qmlt` (<=0), number loss `nmlt`, and
a-/c-axis melt rates `amlt`/`cmlt` toward a more spherical shape. Ports
lines 1509-1535, computed only when `temp>T0` (returns all zeros
otherwise, matching the module's reset defaults at lines 913-914,
939, i.e. `dQImltri`/`dNmltri`/`qmlt`/`nmlt` start at 0 and this branch is
skipped entirely for `temp<=T0`).

# THE KNOWN FORTRAN BUG (fixed here BY CONSTRUCTION, not by a branch)

Line 1512 of the Fortran reads `rhoair(cc)*xxlv*dv*fv(cc)*(qs0-qv(k))` --
`rhoair` is indexed by `cc`, the ICE-SPECIES loop variable (1..3), when it
should be indexed by `k`, the VERTICAL-LEVEL loop variable (`rhoair` is a
per-level air density, declared `rhoair(kts:kte)`; there is no
`rhoair(cat)` species-indexed array in the module at all -- `cc` just
happens to be a valid array index into the wrong dimension whenever
`cc <= kte-kts+1`, silently reading the air density from the WRONG
vertical level).

This scalar, per-species, per-level pure function takes a single
`rhoair::Float64` argument -- there is no per-species array for a `cc`
index to alias into, so the bug cannot recur through this API: the value
passed in is unambiguously "the level's air density" by construction.
Compare `test/test_ishmael.jl`'s dedicated bug-documentation test, which
evaluates the qmlt formula BOTH ways (once with the correct level rhoair,
once by deliberately substituting a different rhoair as the Fortran's
buggy indexing would have used) and asserts this function matches the
CORRECT one.
"""
function ishmael_melting(temp::Float64, ni::Float64, ani::Float64, cni::Float64,
                          deltastr::Float64, rhobar::Float64, qi::Float64, ai::Float64,
                          ci::Float64, kt::Float64, fh::Float64, rhoair::Float64,
                          xxlv::Float64, dv::Float64, fv::Float64, qs0::Float64, qv::Float64,
                          xxlf::Float64, rimetotal::Float64, dQImltri::Float64,
                          dNmltri::Float64, dt::Float64, alphstr::Float64, gammnu::Float64,
                          i_gammnu::Float64, fourthirdspi::Float64;
                          T0::Float64=ISHMAEL_T0, CPW::Float64=ISHMAEL_CPW)
    if !(temp > T0)
        return (qmlt=0.0, nmlt=0.0, amlt=0.0, cmlt=0.0)
    end

    i_dt = 1.0 / dt

    qmlt = 2.0 * ISHMAEL_PI * (kt * fh * (T0 - temp) + rhoair * xxlv * dv * fv * (qs0 - qv)) /
           xxlf * (ni * ISHMAEL_NU * max(ani, cni)) -
           (CPW / xxlf * (temp - T0) * (rimetotal / rhoair + dQImltri))

    qmlt = min(qmlt, 0.0)
    qmlt = max(qmlt, -qi * i_dt)

    # Don't let small number mixing ratios (i.e. very large particles) cause
    # ice to linger instead of precipitating (lines 1519-1524).
    if qmlt < 0.0 && (ai < 1.0e-12 || ci < 1.0e-12)
        qmlt = -qi * i_dt
    end

    nmlt = max(-ni * i_dt, ni * qmlt / qi - dNmltri)

    gam = gamma(ISHMAEL_NU + 2.0 + deltastr)
    tmpmelt = fourthirdspi * alphstr * gam * i_gammnu

    amlt = (1.0 / tmpmelt) * (1.0 / rhobar) * qmlt + ai * nmlt / ni
    cmlt = amlt * cni / ani * (1.0 + 2.0 * deltastr) / (2.0 + deltastr)

    return (qmlt=qmlt, nmlt=nmlt, amlt=amlt, cmlt=cmlt)
end

# ══════════════════════════════════════════════════════════════════════════
# Stage S6b: ice-cloud/ice-rain collection (riming), rime density, wet-
# growth check, and aggregation.
#
# EXCLUSION NOTE (applies to every function below): the Fortran's
# over-depletion / rate-limiter blocks are NOT ported, by design decision
# (the host model removes rate limiters -- ports the pure rates only):
#   - the "do not over-deplete cloud water / rainwater from ice-cloud and
#     ice-rain" block, module_mp_jensen_ishmael.F lines 1216-1242 (includes
#     the known `tmpsum`-for-`tmpsumr` bug at line 1233, which is therefore
#     also excluded, not fixed);
#   - the broader per-species overdepletion/consistency blocks, lines
#     1771-1976;
#   - the "do not over-deplete from aggregation" `ratioagg` reconciliation
#     in `aggregation`, lines 4400-4426.
# Everything up to but NOT INCLUDING those blocks is ported below.
# ══════════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────────────
# 4a. ishmael_ice_cloud_riming -- itab lookup block (lines 1053-1106)
# ────────────────────────────────────────────────────────────────────────────

"""
    ishmael_ice_cloud_riming(itab, rni, qc, deltastr, rhobar, ni, nc, rhoair;
                              QSMALL=1.0e-12) -> NamedTuple{(:rimesum,:qi_qc_nrm,:qi_qc_nrd)}

Ice-cloud collection (riming) rate from the `itab` lookup table. Ports the
index-computation-plus-lookup block, lines 1053-1106: builds the four
log-space table coordinates `(rrni,rqci,rdsi,rrho)` (Fortran's own comment,
lines 1055-1056: do not change this formula without rebuilding the offline
table), the matching integer grid indices (Fortran `INT()` truncates
toward zero, matched here by `trunc(Int, ...)`), clamps BOTH the float
coordinates and the integer indices independently to the table's valid
range `[1, size-1]` in each of the 4 dimensions (matching the Fortran's own
two-stage clamp -- `int()` is taken of the RAW value, then float and int
are each separately floored/capped), and interpolates via
[`access_lookup_table`](@ref) at `index=1` (normalized riming rate,
`qi_qc_nrm`) and `index=2` (normalized rime-density index, `qi_qc_nrd`).
`rimesum` is the dimensional riming rate (kg m^-3 s^-1).

Gated on `qc > 1e-7` -- a SEPARATE, coarser literal than the module's
`QSMALL=1e-12` used below for the small-riming-rate cutoff; transcribed
exactly as the Fortran writes it, not unified into one threshold. Returns
all-zero when `qc <= 1e-7`, or when the raw riming rate maps to a mass
mixing-ratio rate below `QSMALL` (lines 1101-1105).

`qc` passes through the Fortran's own `exp(qc)-1` transform inside `rqci`
(line 1058) verbatim -- transcribed per the Fortran's "do not change"
comment, not independently re-derived.
"""
function ishmael_ice_cloud_riming(itab::Array{Float64,5}, rni::Float64, qc::Float64,
                                   deltastr::Float64, rhobar::Float64, ni::Float64,
                                   nc::Float64, rhoair::Float64; QSMALL::Float64=ISHMAEL_QSMALL)
    if !(qc > 1.0e-7)
        return (rimesum=0.0, qi_qc_nrm=0.0, qi_qc_nrd=0.0)
    end

    rrni_raw = 13.498 * log10(0.5e6 * rni)
    rqci_raw = 8.776 * log10(1.0e7 * (exp(qc) - 1.0))
    rdsi_raw = 50.0 * (deltastr - 0.5)
    rrho_raw = 7.888 * log10(0.02 * rhobar)

    irni = trunc(Int, rrni_raw)
    iqci = trunc(Int, rqci_raw)
    idsi = trunc(Int, rdsi_raw)
    irho = trunc(Int, rrho_raw)

    rrni = clamp(rrni_raw, 1.0, Float64(size(itab, 1) - 1))
    rqci = clamp(rqci_raw, 1.0, Float64(size(itab, 2) - 1))
    rdsi = clamp(rdsi_raw, 1.0, Float64(size(itab, 3) - 1))
    rrho = clamp(rrho_raw, 1.0, Float64(size(itab, 4) - 1))

    irni = clamp(irni, 1, size(itab, 1) - 1)
    iqci = clamp(iqci, 1, size(itab, 2) - 1)
    idsi = clamp(idsi, 1, size(itab, 3) - 1)
    irho = clamp(irho, 1, size(itab, 4) - 1)

    proc1 = access_lookup_table(itab, irni, iqci, idsi, irho, 1, rdsi, rrho, rqci, rrni)
    proc2 = access_lookup_table(itab, irni, iqci, idsi, irho, 2, rdsi, rrho, rqci, rrni)

    rimesum = max(proc1 * ni * nc * rhoair^2, 0.0)

    if (rimesum / rhoair) < QSMALL
        return (rimesum=0.0, qi_qc_nrm=0.0, qi_qc_nrd=0.0)
    end

    return (rimesum=rimesum, qi_qc_nrm=proc1, qi_qc_nrd=proc2)
end

# ────────────────────────────────────────────────────────────────────────────
# 4b. ishmael_ice_rain_riming -- itabr lookup block (lines 1111-1214)
# ────────────────────────────────────────────────────────────────────────────

"""
    ishmael_ice_rain_riming(itabr, rni, qr, nr, deltastr, rhobar, ni, rhoair,
                             temp, qi; QSMALL=1.0e-12, T0=273.15) -> NamedTuple

Ice-rain collection rate from the `itabr` lookup table, PLUS the T<=T0
freeze-ri / T>T0 melt-ri transfer branches that key off the same lookup.
Ports lines 1111-1214: the rain DSD via [`ishmael_rain_lambda`](@ref)
(`rrr` is the resulting mean radius, `0.5/lamr`), the four log-space table
coordinates `(rrni,rrri,rdsi,rrho)` (same "do not change" comment as
`itab`, and the same two-stage float/int clamp as
[`ishmael_ice_cloud_riming`](@ref)), the 6-index lookup (`itabr`'s 5th
dimension: 1=riming rate, 2=rime density, 3=rain-number loss, 4=ice-number
tendency, 5=rain-mass tendency, 6=ice-mass tendency), the small-rate gate
(lines 1174-1179), the `T<=T0` freeze branch (`dQRfzri`/`dQIfzri`/`dNfzri`,
active only when BOTH `qr>0.1e-3` and `qi>0.1e-3`) vs. the `T>T0` melt
branch (`dQImltri`/`dNmltri`), and the ice-rain rime-rate small-rate gate
(lines 1206-1212).

Returns zero for everything when `qr <= QSMALL` (the `RAIN_ICE.and.
qr(k).gt.QSMALL` outer gate, line 1111 -- `RAIN_ICE` assumed true, same
convention as `FREEZE_QC` in [`ishmael_homogeneous_freezing`](@ref)).
"""
function ishmael_ice_rain_riming(itabr::Array{Float64,5}, rni::Float64, qr::Float64, nr::Float64,
                                  deltastr::Float64, rhobar::Float64, ni::Float64, rhoair::Float64,
                                  temp::Float64, qi::Float64; QSMALL::Float64=ISHMAEL_QSMALL,
                                  T0::Float64=ISHMAEL_T0)
    if !(qr > QSMALL)
        return (rimesumr=0.0, qi_qr_nrm=0.0, qi_qr_nrd=0.0, qi_qr_nrn=0.0,
                numrateri=0.0, rainrateri=0.0, icerateri=0.0,
                dQRfzri=0.0, dQIfzri=0.0, dNfzri=0.0, dQImltri=0.0, dNmltri=0.0)
    end

    dsd = ishmael_rain_lambda(qr, nr)
    lamr = dsd.lamr
    rrr = 0.5 * (1.0 / lamr)

    rrni_raw = 13.498 * log10(0.5e6 * rni)
    rrri_raw = 23.273 * log10(1.0e5 * rrr)
    rdsi_raw = 50.0 * (deltastr - 0.5)
    rrho_raw = 7.888 * log10(0.02 * rhobar)

    irni = trunc(Int, rrni_raw)
    irri = trunc(Int, rrri_raw)
    idsi = trunc(Int, rdsi_raw)
    irho = trunc(Int, rrho_raw)

    rrni = clamp(rrni_raw, 1.0, Float64(size(itabr, 1) - 1))
    rrri = clamp(rrri_raw, 1.0, Float64(size(itabr, 2) - 1))
    rdsi = clamp(rdsi_raw, 1.0, Float64(size(itabr, 3) - 1))
    rrho = clamp(rrho_raw, 1.0, Float64(size(itabr, 4) - 1))

    irni = clamp(irni, 1, size(itabr, 1) - 1)
    irri = clamp(irri, 1, size(itabr, 2) - 1)
    idsi = clamp(idsi, 1, size(itabr, 3) - 1)
    irho = clamp(irho, 1, size(itabr, 4) - 1)

    procr1 = access_lookup_table(itabr, irni, irri, idsi, irho, 1, rdsi, rrho, rrri, rrni)
    procr2 = access_lookup_table(itabr, irni, irri, idsi, irho, 2, rdsi, rrho, rrri, rrni)
    procr3 = access_lookup_table(itabr, irni, irri, idsi, irho, 3, rdsi, rrho, rrri, rrni)
    procr4 = access_lookup_table(itabr, irni, irri, idsi, irho, 4, rdsi, rrho, rrri, rrni)
    procr5 = access_lookup_table(itabr, irni, irri, idsi, irho, 5, rdsi, rrho, rrri, rrni)
    procr6 = access_lookup_table(itabr, irni, irri, idsi, irho, 6, rdsi, rrho, rrri, rrni)
    procr = (procr1, procr2, procr3, procr4, procr5, procr6)

    numrateri  = max(procr[4] * ni * nr * rhoair, 0.0)
    rainrateri = max(procr[5] * ni * nr * rhoair, 0.0)
    icerateri  = max(procr[6] * ni * nr * rhoair, 0.0)

    if rainrateri < QSMALL || icerateri < QSMALL
        numrateri = 0.0
        rainrateri = 0.0
        icerateri = 0.0
    end

    local dQRfzri, dQIfzri, dNfzri, dQImltri, dNmltri
    if temp <= T0
        if qr > 0.1e-3 && qi > 0.1e-3
            dQRfzri, dQIfzri, dNfzri = rainrateri, icerateri, numrateri
        else
            dQRfzri, dQIfzri, dNfzri = 0.0, 0.0, 0.0
        end
        dQImltri, dNmltri = 0.0, 0.0
    else
        dQImltri, dNmltri = icerateri, numrateri
        dQRfzri, dQIfzri, dNfzri = 0.0, 0.0, 0.0
    end

    rimesumr = max(procr[1] * ni * nr * rhoair^2, 0.0)
    qi_qr_nrm, qi_qr_nrd, qi_qr_nrn = procr[1], procr[2], procr[3]

    if (rimesumr / rhoair) < QSMALL
        rimesumr, qi_qr_nrm, qi_qr_nrd, qi_qr_nrn = 0.0, 0.0, 0.0, 0.0
    end

    return (rimesumr=rimesumr, qi_qr_nrm=qi_qr_nrm, qi_qr_nrd=qi_qr_nrd, qi_qr_nrn=qi_qr_nrn,
            numrateri=numrateri, rainrateri=rainrateri, icerateri=icerateri,
            dQRfzri=dQRfzri, dQIfzri=dQIfzri, dNfzri=dNfzri, dQImltri=dQImltri, dNmltri=dNmltri)
end

# ────────────────────────────────────────────────────────────────────────────
# 4c. ishmael_wet_growth_check -- Lamb & Verlinde (2011) wet-growth limit
#     (lines 3537-3556)
# ────────────────────────────────────────────────────────────────────────────

"""
    ishmael_wet_growth_check(NU, temp, rhodum, xxlv, xxlf, qvdum, dv, kt, qs0,
                              fvdum, fhdum, rimedum, rni, nidum;
                              T0=273.15, CPW=4218.0) -> Bool

Lamb and Verlinde (2011) wet-growth check: `true` for dry growth, `false`
for wet growth. Ports `wet_growth_check`, lines 3537-3556, verbatim. The
Fortran's `dgflag` is `INTENT(INOUT)` and is always reset to `.true.` by
the caller immediately before the call (line 875, top of the vertical-
level loop) -- this function reproduces that as a pure boolean return
(there is no external state to mutate): dry growth (`true`) UNLESS
`rimedum/rhodum > wetg`.

Does NOT itself apply the `temp>T0 -> dry_growth=.false.` override (lines
1300-1302) -- that line sits just after the `wet_growth_check` call in the
host loop and is folded into [`ishmael_riming_growth`](@ref) instead (see
that function's docstring), matching how this function mirrors ONLY the
subroutine boundary.
"""
@inline function ishmael_wet_growth_check(NU::Float64, temp::Float64, rhodum::Float64,
                                           xxlv::Float64, xxlf::Float64, qvdum::Float64, dv::Float64,
                                           kt::Float64, qs0::Float64, fvdum::Float64, fhdum::Float64,
                                           rimedum::Float64, rni::Float64, nidum::Float64;
                                           T0::Float64=ISHMAEL_T0, CPW::Float64=ISHMAEL_CPW)
    dum = nidum * NU * 2.0 * rni
    wetg = 2.0 * ISHMAEL_PI * dum * (kt * fhdum * (T0 - temp) + rhodum * xxlv * dv * fvdum * (qs0 - qvdum)) /
           (xxlf + CPW * (temp - T0))
    return !(rimedum / rhodum > wetg)
end

# ────────────────────────────────────────────────────────────────────────────
# 4d. ishmael_macklin_rimec1 / ishmael_macklin_density -- shared Macklin
#     (1962) rime-density pieces used twice (ice-cloud lines 1312-1329 /
#     1330-1342, ice-rain lines 1357-1374 / 1375-1387)
# ────────────────────────────────────────────────────────────────────────────

"""
    ishmael_macklin_rimec1(temp; T0=273.15) -> Float64

Piecewise-linear-in-temperature Macklin (1962) rime-density coefficient
`rimec1`, used identically for both ice-cloud (lines 1312-1329) and
ice-rain (lines 1357-1374) riming -- ported once, called twice, since the
Fortran itself duplicates this formula verbatim in both places.
"""
@inline function ishmael_macklin_rimec1(temp::Float64; T0::Float64=ISHMAEL_T0)
    dTc = temp - T0
    if dTc < -30.0
        return 0.0036
    elseif dTc < -20.0
        dum = (abs(dTc) - 20.0) / 10.0
        return dum * 0.0036 + (1.0 - dum) * 0.004
    elseif dTc < -15.0
        dum = (abs(dTc) - 15.0) / 5.0
        return dum * 0.004 + (1.0 - dum) * 0.005
    elseif dTc < -10.0
        dum = (abs(dTc) - 10.0) / 5.0
        return dum * 0.005 + (1.0 - dum) * 0.0066
    elseif dTc < -5.0
        dum = (abs(dTc) - 5.0) / 5.0
        return dum * 0.0066 + (1.0 - dum) * 0.012
    else
        return 0.012
    end
end

"""
    ishmael_macklin_density(rimec1, nrd, nrm, temp, dry_growth; T0=273.15) -> Float64

Macklin (1962) rime density `gdenavg`/`gdenavgr` (kg m^-3), clamped to
`[50,900]`: `tanh`-based blend from `rimec1` and the `qi_x_nrd/qi_x_nrm`
ratio, a near-0C warm-side blend toward 900, and a hard override to 900
above 0C or under wet growth. Ports the shared averaging block (lines
1330-1342 / 1375-1387) once; called twice (ice-cloud and ice-rain) by
[`ishmael_riming_growth`](@ref).
"""
@inline function ishmael_macklin_density(rimec1::Float64, nrd::Float64, nrm::Float64,
                                          temp::Float64, dry_growth::Bool; T0::Float64=ISHMAEL_T0)
    dTc = temp - T0
    gden = 1000.0 * (0.8 * tanh(rimec1 * nrd / nrm) + 0.1)
    if dTc > -5.0 && dTc <= 0.0
        dum = abs(dTc) / 5.0
        gden = dum * gden + (1.0 - dum) * 900.0
    end
    if dTc > 0.0 || !dry_growth
        gden = 900.0
    end
    return clamp(gden, 50.0, 900.0)
end

# ────────────────────────────────────────────────────────────────────────────
# 4e. ishmael_riming_growth -- riming mass/axis growth rates
#     (lines 1291-1507)
# ────────────────────────────────────────────────────────────────────────────

"""
    ishmael_riming_growth(dt, rni, deltastr, rbdum, nidum, ani, cni, temp,
                           qc, nc, qi_qc_nrm, qi_qc_nrd, rimesum,
                           qr, nr, qi_qr_nrm, qi_qr_nrd, rimesumr,
                           rhoair, dry_growth_pre, NU, ao, gammnu, i_gammnu,
                           fourthirdspi; T0=273.15, RHOI=920.0,
                           QSMALL=1.0e-12) -> NamedTuple

Ice riming mass and axis growth rates. Ports lines 1291-1507: the
Macklin rime-density calls ([`ishmael_macklin_rimec1`](@ref) /
[`ishmael_macklin_density`](@ref)) for ice-cloud and ice-rain separately,
the riming r-axis growth `rnfr`, the post-riming volume `vfr`, the
riming-fraction-weighted total rime density blend (`qcrimefrac`/
`gdentotal`), the dry-growth axis-shape response (`phibr`/`phifr`
branches, including the "don't flip prolate<->oblate" safety clamps), and
the wet-growth branch (mass added, no axis growth). Returns
`(prdr, ardr, crdr, rhorimeout, gdenavg, gdenavgr, dry_growth)`.

# `dry_growth_pre` / the T>T0 override

`dry_growth_pre` is the output of [`ishmael_wet_growth_check`](@ref)
(computed by the caller, since that IS a true Fortran subroutine and is
ported separately). This function applies the `temp>T0 -> dry_growth =
false` override itself (lines 1300-1302), since that line is textually
inside the 1291-1507 range this function covers: `dry_growth =
dry_growth_pre && !(temp>T0)`.

# `nidum` convention (mirrors the SEAM already documented in
[`ishmael_deposition_partition`](@ref))

`nidum` is `nim3dum = ni(cc,k)*rhoair(k)` (# m^-3), matching `iwci`'s own
convention -- used for `prdr` (`(iwcfr-iwci)/rhoair/dt`) exactly as the
Fortran does. The Fortran's `ardr`/`crdr` (lines 1485-1486) instead use
`ni(cc,k)` (# kg^-1), NOT `nim3dum` -- but per the SAME seam decision
`ishmael_deposition_partition` already made for its own `ard`/`crd`
(documented there: "the density<->mixing-ratio conversion... is
explicitly a LATER integration-stage concern"), this function reuses the
single `nidum` (# m^-3) parameter for `ardr`/`crdr` too, for convention
consistency across the module. `ardr`/`crdr` are therefore NOT expected to
bit-match the true Fortran output; `prdr` IS.

# An observed Fortran quirk, ported verbatim (not a documented task
# exclusion, just an oddity worth flagging)

When riming occurs (`rimetotal>0 && vfr>vi`), the Fortran computes a
volume-blended `rhorimeout = rbdum*(vi/vfr) + gdentotal*(1-vi/vfr)`
(clamped to `RHOI`) and uses THAT to form `iwcfr` -- but then
IMMEDIATELY overwrites its own `rhorimeout` output with
`min(gdentotal,RHOI)`, discarding the blended value entirely (lines
1414-1418). This function reproduces that exactly: `rhorimeout` in the
return value is `min(gdentotal,RHOI)`, NOT the blended density that
actually fed `iwcfr`.
"""
function ishmael_riming_growth(dt::Float64, rni::Float64, deltastr::Float64, rbdum::Float64,
                                nidum::Float64, ani::Float64, cni::Float64, temp::Float64,
                                qc::Float64, nc::Float64, qi_qc_nrm::Float64, qi_qc_nrd::Float64,
                                rimesum::Float64, qr::Float64, nr::Float64, qi_qr_nrm::Float64,
                                qi_qr_nrd::Float64, rimesumr::Float64, rhoair::Float64,
                                dry_growth_pre::Bool, NU::Float64, ao::Float64, gammnu::Float64,
                                i_gammnu::Float64, fourthirdspi::Float64; T0::Float64=ISHMAEL_T0,
                                RHOI::Float64=ISHMAEL_RHOI, QSMALL::Float64=ISHMAEL_QSMALL)
    dry_growth = dry_growth_pre && !(temp > T0)

    gam = gamma(NU + 2.0 + deltastr)
    vi = fourthirdspi * rni^3 * gam * i_gammnu
    iwci = nidum * rbdum * vi

    rnfr = rni

    gdenavg = 900.0
    if qc > QSMALL && qi_qc_nrm > 0.0
        rimec1 = ishmael_macklin_rimec1(temp; T0=T0)
        gdenavg = ishmael_macklin_density(rimec1, qi_qc_nrd, qi_qc_nrm, temp, dry_growth; T0=T0)
        rimedr = max((qi_qc_nrm / gdenavg) / (gam * i_gammnu * 4.0 * ISHMAEL_PI * rni^2), 0.0)
        rnfr += max(rimedr * nc * rhoair, 0.0) * dt
    end

    gdenavgr = 900.0
    if qr > QSMALL && qi_qr_nrm > 0.0
        rimec1r = ishmael_macklin_rimec1(temp; T0=T0)
        gdenavgr = ishmael_macklin_density(rimec1r, qi_qr_nrd, qi_qr_nrm, temp, dry_growth; T0=T0)
        rimedrr = max((qi_qr_nrm / gdenavgr) / (gam * i_gammnu * 4.0 * ISHMAEL_PI * rni^2), 0.0)
        rnfr += max(rimedrr * nr * rhoair, 0.0) * dt
    end

    vfr = fourthirdspi * rnfr^3 * gam * i_gammnu
    vfr = max(vfr, vi)
    rnfr = max(rnfr, rni)

    rimetotal = rimesum + rimesumr

    local iwcfr, rhorimeout
    if rimetotal > 0.0 && vfr > vi
        qcrimefrac = clamp(rimesum / rimetotal, 0.0, 1.0)
        gdentotal = qcrimefrac * gdenavg + (1.0 - qcrimefrac) * gdenavgr
        rhorimeout_blend = min(rbdum * (vi / vfr) + gdentotal * (1.0 - vi / vfr), RHOI)
        iwcfr = rhorimeout_blend * vfr * nidum
        rhorimeout = min(gdentotal, RHOI)   # see docstring: the blend is discarded here, per the Fortran
    else
        iwcfr = iwci
        rnfr = rni
        rhorimeout = rbdum
    end

    local prdr, ardr, crdr
    if dry_growth
        cnfr = cni
        anfr = ani
        phibr = 1.0
        if rimetotal > 0.0 && vfr > vi
            gam_m1 = gamma(NU - 1.0 + deltastr)
            phibr = cni / ani * gam_m1 * i_gammnu
            local phifr
            if phibr == 1.0
                phifr, anfr, cnfr = 1.0, rnfr, rnfr
            elseif phibr > 1.25
                phifr = phibr * ((rnfr / rni)^3)^(-0.5)
                anfr = (ani / rni^1.5) * rnfr^1.5
            elseif phibr < 0.8
                phifr = phibr * (rnfr / rni)^3
                cnfr = phifr * anfr * gammnu / gam_m1
            else
                phifr = phibr
                anfr = ((vfr * gam_m1) / (fourthirdspi * phifr * gam))^(1.0 / 3.0)
                cnfr = phifr * anfr * gammnu / gam_m1
            end

            if phibr <= 1.0 && phifr > 1.0
                phifr = 0.99
                anfr = ((vfr * gam_m1) / (fourthirdspi * phifr * gam))^(1.0 / 3.0)
                cnfr = phifr * anfr * gammnu / gam_m1
            end
            if phibr >= 1.0 && phifr < 1.0
                phifr = 1.01
                anfr = ((vfr * gam_m1) / (fourthirdspi * phifr * gam))^(1.0 / 3.0)
                cnfr = phifr * anfr * gammnu / gam_m1
            end
        end

        cnfr = max(cnfr, cni)
        anfr = max(anfr, ani)

        prdr = (iwcfr - iwci) / rhoair / dt
        ardr = (2.0 * (anfr - ani) * cni + (cnfr - cni) * ani) * ani * nidum / dt
        crdr = (2.0 * (cnfr - cni) * ani + (anfr - ani) * cni) * cni * nidum / dt

        prdr = max(prdr, 0.0)
        ardr = max(ardr, 0.0)
        crdr = max(crdr, 0.0)

        if prdr == 0.0
            ardr = 0.0
            crdr = 0.0
        end
    else
        prdr = (iwcfr - iwci) / rhoair / dt
        ardr = 0.0
        crdr = 0.0
    end

    return (prdr=prdr, ardr=ardr, crdr=crdr, rhorimeout=rhorimeout,
            gdenavg=gdenavg, gdenavgr=gdenavgr, dry_growth=dry_growth)
end

# ────────────────────────────────────────────────────────────────────────────
# 4f. Aggregation: table-index helper, col1, aggregation
#     (lines 3988-4541)
# ────────────────────────────────────────────────────────────────────────────

const ISHMAEL_AGG_NDN    = 60
const ISHMAEL_AGG_TABLO  = 1.0e-6
const ISHMAEL_AGG_TABHI  = 1.0e-2
# All 7 dstprms table-size columns (ISHMAEL_TABLO/ISHMAEL_TABHI in
# ishmael_tables.jl) are the SAME [1e-6,1e-2] for every category, so
# `dict(icat)` (mkcoltb/aggregation's per-category table-index scale
# factor) reduces to one shared constant -- no need to carry a per-
# category array through the 3 live categories used here.
const ISHMAEL_AGG_DICT   = (ISHMAEL_AGG_NDN - 1) / (log(ISHMAEL_AGG_TABHI) - log(ISHMAEL_AGG_TABLO))
const ISHMAEL_AGG_RICTMIN = 1.0001
const ISHMAEL_AGG_RICTMAX = 0.9999 * ISHMAEL_AGG_NDN

"""
    ishmael_agg_table_index(dn) -> NamedTuple{(:ict,:wct1,:wct2)}

Table-node index and interpolation weights for one aggregation category's
characteristic diameter `dn`, ports lines 4188-4207 (the `rict`/`ict`/
`wct1`/`wct2` computation, replicated 3x per category in the Fortran --
ported once here since `ISHMAEL_TABLO`/`ISHMAEL_TABHI` are identical
across categories, see [`ISHMAEL_AGG_DICT`](@ref)).
"""
@inline function ishmael_agg_table_index(dn::Float64)
    rict = ISHMAEL_AGG_DICT * (log(max(1.0e-10, dn)) - log(ISHMAEL_AGG_TABLO)) + 1.0
    rictmm = clamp(rict, ISHMAEL_AGG_RICTMIN, ISHMAEL_AGG_RICTMAX)
    ict = trunc(Int, rictmm)
    wct2 = rictmm - Float64(ict)
    wct1 = 1.0 - wct2
    return (ict=ict, wct1=wct1, wct2=wct2)
end

"""
    ishmael_agg_efffact(rhoeffmax, phieffmax) -> Float64

Shared density/shape collection-efficiency factor (`efffact = phieff *
rhoeff`), ports the identical rhoeff/phieff block computed 5x in
`aggregation` (lines 4214-4242, 4254-4281, 4289-4316, 4324-4351,
4359-4386) -- once here. `rhoeff` reduces the efficiency for high-density
(quasi-spherical / graupel-like) particles; `phieff` (Connolly et al.
2012) reduces it for near-spherical shapes, boosted to 1 for very
extreme (highly aspherical) shapes.
"""
@inline function ishmael_agg_efffact(rhoeffmax::Float64, phieffmax::Float64)
    rhoeff = rhoeffmax <= 400.0 ? 1.0 : (-0.001923 * rhoeffmax + 1.76916)
    rhoeff = clamp(rhoeff, 0.0, 1.0)

    local phieff
    if phieffmax <= 0.03
        phieff = 1.0
    elseif phieffmax < 0.5
        phieff = 0.0001 * (phieffmax + 0.07)^(-4.0)
    else
        phieff = 0.0
    end
    if phieff <= 0.001
        phieff = 0.0
    end
    phieff = clamp(phieff, 0.01, 1.0)

    return phieff * rhoeff
end

"""
    ishmael_col1(dtlt, efdum, tempC, dnx, enx, rx, dny, eny, rhoair,
                 coltab, coltabn, pidx) -> NamedTuple{(:colamt,:deltan)}

One collector(x)-collectee(y) pair's mass (`colamt`) and number
(`deltan`) transfer over timestep `dtlt`, from the tabulated collection
kernel `coltab`/`coltabn` (built by `mkcoltb`, ishmael_tables.jl). Ports
`col1`, lines 4479-4541, SPECIALIZED to the 3 live categories (planar,
columnar, aggregates) that `aggregation` ever calls it with: `cr=1.0`
always (every live-category call in `aggregation` passes `cr=1.`),
`mz=5` (aggregates) always (so the `mz.eq.5` dendritic-growth-zone
efficiency-boost branch is unconditional here, using `tempC` -- all 3 live
categories share the same air temperature, `t(k,mx)`, per
`aggregation`'s own setup, lines 4137-4144), and `icf=5`/`ipris=5` always
(so the number-transfer `deltan` blend, lines 4530-4536, always applies).
The `mx.eq.1` (cloud) and `my.eq.6/7` (graupel/hail) branches never fire
for planar/columnar/aggregates and are dropped. `qrxfer` (internal-energy
transfer) is NOT ported: `aggregation`'s own `qagg1..3`/`nagg1..3`
outputs never reference it, and the Fortran itself feeds it an
effectively-unset local (`qq(k,3)` is assigned from `t(k,3)` ONE LINE
BEFORE `t(k,3)` is itself set, lines 4137-4138).

`colamt=min(colamt,rx)`/`colamtn=min(colamtn,enx)` (lines 4517, 4528) ARE
kept -- this is a per-PAIR physical bound (one collection process cannot
remove more mass/number than its own collector species has), distinct
from the excluded CROSS-pair "over-depletion" reconciliation in
`aggregation` itself (see the module-level exclusion note above).
"""
@inline function ishmael_col1(dtlt::Float64, efdum::Float64, tempC::Float64,
                               dnx::Float64, enx::Float64, rx::Float64,
                               dny::Float64, eny::Float64, rhoair::Float64,
                               coltab::Array{Float64,3}, coltabn::Array{Float64,3}, pidx::Int)
    eff = min(0.2, 10.0^(0.035 * tempC - 0.7)) * efdum
    if abs(tempC + 14.0) <= 2.0
        eff = 1.4 * efdum
    end

    ix = ishmael_agg_table_index(dnx)
    iy = ishmael_agg_table_index(dny)

    denfac = sqrt(1.0 / rhoair)
    pref = dtlt * 0.785 * eff * denfac / rhoair   # cr=1.0 always in this specialization

    local ctab, ctabn
    @inbounds begin
        ctab  = ix.wct1 * iy.wct1 * coltab[ix.ict,     iy.ict,     pidx] +
                ix.wct2 * iy.wct1 * coltab[ix.ict + 1, iy.ict,     pidx] +
                ix.wct1 * iy.wct2 * coltab[ix.ict,     iy.ict + 1, pidx] +
                ix.wct2 * iy.wct2 * coltab[ix.ict + 1, iy.ict + 1, pidx]
        ctabn = ix.wct1 * iy.wct1 * coltabn[ix.ict,     iy.ict,     pidx] +
                ix.wct2 * iy.wct1 * coltabn[ix.ict + 1, iy.ict,     pidx] +
                ix.wct1 * iy.wct2 * coltabn[ix.ict,     iy.ict + 1, pidx] +
                ix.wct2 * iy.wct2 * coltabn[ix.ict + 1, iy.ict + 1, pidx]
    end

    colamt = min(pref * enx * eny * ctab, rx)
    colamtn = min(pref * enx * eny * ctabn, enx)

    wght = clamp(100.0 * abs(colamt) / max(1.0e-20, rx), 0.0, 1.0)
    deltan = colamtn * (1.0 - wght) + wght * enx * colamt / max(1.0e-20, rx)

    return (colamt=colamt, deltan=deltan)
end

"""
    ishmael_aggregation(dt, rhoair, temp, q1, n1, d1, q2, n2, d2, q3, n3, d3,
                         rho1, rho2, phi1, phi2, coltab, coltabn;
                         T0=273.15) -> NamedTuple

Ice-ice aggregation among the 3 live categories (1=planar, 2=columnar,
3=aggregates -- matching `aggregation`'s own 3/4/5 category numbering).
Ports `aggregation`, lines 3988-4474 (which itself calls `col1`, lines
4479-4541, [`ishmael_col1`](@ref) here), SIMPLIFIED to the 6 live
collection pairs (planar+columnar, planar+agg, columnar+agg,
planar-self, columnar-self, agg-self) via 7 [`ishmael_col1`](@ref)
sub-calls (planar+columnar needs 2: planar-collects-columnar AND
columnar-collects-planar, both feeding the SAME "planar+columnar" named
pair per the task framing). Uses [`ishmael_agg_efffact`](@ref) for each
pair-group's collection efficiency (`rhoeffmax`/`phieffmax` combining the
colliding species' `rho`/`phi`, per lines 4214-4242 etc -- self-collection
of a category reuses that category's own cross-with-aggregates
`rhoeffmax`/`phieffmax`, since the Fortran's formulas at lines 4324-4325/
4359-4360 are textually identical to 4254-4255/4289-4290). Aggregate
self-collection uses `efdum=1.0` (line 4393, no efficiency reduction).

Returns `(qagg1, qagg2, qagg3, nagg1, nagg2, nagg3, dnew3)` -- these are
**timestep-INTEGRATED amounts, not rates** (the Fortran's `colamt` already
has `dtlt` baked in, and `mp_jensen_ishmael` adds `qagg`/`nagg` straight
into `qi`/`ni` with no further `*dt`, line 2398) -- matching the Fortran's
own return convention exactly, not converted to a rate here.

`dnew3` (`ddum3`, the updated aggregate characteristic diameter) uses a
LOCALLY-OVERRIDDEN mass-diameter relation (`cfmas=pi/6*50*0.2`,
`pwmas=3.0`, lines 4092-4093 -- note the Fortran's own `3.14159` literal
here, NOT the module's more-precise `PI=3.14159265`, transcribed as
written), DIFFERENT from the `cfmas`/`pwmas` baked into `coltab`/`coltabn`
at table-BUILD time (`ISHMAEL_CFMAS`/`ISHMAEL_PWMAS` in
ishmael_tables.jl, used by `mkcoltb`) -- these are two independent,
intentionally-different uses of "cfmas(5)" in the Fortran, both
transcribed faithfully to their own call sites.

EXCLUDED: the "do not over-deplete from aggregation" `ratioagg`
reconciliation across the 3 pairs feeding category 3/4's sink (lines
4400-4426) -- see the module-level exclusion note above. `qagg1`/`qagg2`
are therefore the UNLIMITED sum of their 3 contributing pairs' `colamt`
(each pair already individually bounded by [`ishmael_col1`](@ref)'s own
`min(colamt,rx)`, but the 3-pair SUM is not re-bounded against `q1`/`q2`).
"""
function ishmael_aggregation(dt::Float64, rhoair::Float64, temp::Float64,
                              q1::Float64, n1::Float64, d1::Float64,
                              q2::Float64, n2::Float64, d2::Float64,
                              q3::Float64, n3::Float64, d3::Float64,
                              rho1::Float64, rho2::Float64, phi1::Float64, phi2::Float64,
                              coltab::Array{Float64,3}, coltabn::Array{Float64,3};
                              T0::Float64=ISHMAEL_T0)
    tempC = temp - T0
    en1 = n1 * rhoair
    en2 = n2 * rhoair
    en3 = n3 * rhoair

    ip_34 = ISHMAEL_IPAIR[3, 4]; ip_43 = ISHMAEL_IPAIR[4, 3]
    ip_35 = ISHMAEL_IPAIR[3, 5]; ip_45 = ISHMAEL_IPAIR[4, 5]
    ip_33 = ISHMAEL_IPAIR[3, 3]; ip_44 = ISHMAEL_IPAIR[4, 4]
    ip_55 = ISHMAEL_IPAIR[5, 5]

    # planar+columnar
    ef_pc = ishmael_agg_efffact(max(rho1, rho2), max(min(phi1, 1.0 / phi1), min(phi2, 1.0 / phi2)))
    c_34 = ishmael_col1(dt, ef_pc, tempC, d1, en1, q1, d2, en2, rhoair, coltab, coltabn, ip_34)
    c_43 = ishmael_col1(dt, ef_pc, tempC, d2, en2, q2, d1, en1, rhoair, coltab, coltabn, ip_43)

    # planar+aggregates (also reused for planar self-collection, see docstring)
    ef_pa = ishmael_agg_efffact(rho1, min(phi1, 1.0 / phi1))
    c_35 = ishmael_col1(dt, ef_pa, tempC, d1, en1, q1, d3, en3, rhoair, coltab, coltabn, ip_35)
    c_33 = ishmael_col1(dt, ef_pa, tempC, d1, en1, q1, d1, en1, rhoair, coltab, coltabn, ip_33)

    # columnar+aggregates (also reused for columnar self-collection)
    ef_ca = ishmael_agg_efffact(rho2, min(phi2, 1.0 / phi2))
    c_45 = ishmael_col1(dt, ef_ca, tempC, d2, en2, q2, d3, en3, rhoair, coltab, coltabn, ip_45)
    c_44 = ishmael_col1(dt, ef_ca, tempC, d2, en2, q2, d2, en2, rhoair, coltab, coltabn, ip_44)

    # aggregate self-collection: efdum=1 (no efficiency reduction)
    c_55 = ishmael_col1(dt, 1.0, tempC, d3, en3, q3, d3, en3, rhoair, coltab, coltabn, ip_55)

    sink3 = c_34.colamt + c_35.colamt + c_33.colamt
    sink4 = c_43.colamt + c_45.colamt + c_44.colamt

    qagg1 = min(-sink3, 0.0)
    qagg2 = min(-sink4, 0.0)
    qagg3 = max(c_34.colamt + c_43.colamt + c_35.colamt + c_45.colamt + c_33.colamt + c_44.colamt, 0.0)

    nagg1 = -(c_34.deltan + c_35.deltan + c_33.deltan)
    nagg2 = -(c_43.deltan + c_45.deltan + c_44.deltan)
    # NOTE: nagg3 deliberately does NOT include c_35.deltan/c_45.deltan
    # (the planar+agg / columnar+agg number transfers), even though qagg3
    # DOES include their colamt (mass transfer) above -- transcribed
    # exactly from the Fortran (lines 4442-4443: `nagg3 = ncrossgain*
    # enxfer(k,3,5) + ncrossgain*enxfer(k,4,5) + nselfgain*ppenxfer(k,3,5)
    # + nselfgain*ppenxfer(k,4,5) - 0.5*ppenxfer(k,5,5)` -- no `paenxfer`
    # term at all). Physically sensible despite looking asymmetric: a
    # planar/columnar particle absorbed into an EXISTING aggregate adds
    # its MASS to that aggregate but does not change the aggregate COUNT
    # (it was already one aggregate); only planar+columnar collisions
    # (genuinely forming a brand-new aggregate) and self-collection
    # change the aggregate number.
    nagg3_raw = 0.5 * (c_34.deltan + c_43.deltan) +
                0.5 * (c_33.deltan + c_44.deltan) - 0.5 * c_55.deltan

    # dnew3 (ddum3): computed from the RAW (still #/m^3-convention) nagg3,
    # BEFORE the #/kg conversion below -- matches the Fortran's ordering
    # (line 4452 precedes the nagg/rhoa division at line 4459).
    pi_local = 3.14159        # the Fortran's own literal at this call site, see docstring
    cfmas5_local = pi_local / 6.0 * 50.0 * 0.2
    pwmas5_local = 3.0
    gnu5 = 4.0
    dnew3 = ((q3 + qagg3) * rhoair /
             (max(en3 + nagg3_raw, ISHMAEL_QNSMALL) * cfmas5_local * (gamma(gnu5 + pwmas5_local) / gamma(gnu5))))^(1.0 / pwmas5_local)

    nagg1 /= rhoair
    nagg2 /= rhoair
    nagg3 = nagg3_raw / rhoair

    return (qagg1=qagg1, qagg2=qagg2, qagg3=qagg3, nagg1=nagg1, nagg2=nagg2, nagg3=nagg3, dnew3=dnew3)
end

# ────────────────────────────────────────────────────────────────────────────
# 4g. Pure diagnostic caps: ice number (line 2200), aggregate size
#     (lines 2604-2612)
# ────────────────────────────────────────────────────────────────────────────

"""
    ishmael_ni_cap(ni, rhoair) -> Float64

Pure diagnostic cap on ice number mixing ratio `ni` (# kg^-1) to 1000
per liter of air. Ports line 2200,
`ni(cc,k)=min(ni(cc,k),(1000.*1000.*i_rhoair(k)))`, verbatim: `1000 L^-1 =
1e6 m^-3`, divided by `rhoair` to convert to # kg^-1. The host decides
when/whether to apply this cap -- this is the pure `min`, nothing more.
"""
@inline ishmael_ni_cap(ni::Float64, rhoair::Float64) = min(ni, 1.0e6 / rhoair)

"""
    ishmael_agg_size_cap(ani, cni, ni, dsdum, ao, qi, rbdum, NU, gammnu;
                          QNSMALL=1.25e-7, fourthirdspi=4/3*PI) -> NamedTuple{(:ani,:cni,:ni)}

Pure diagnostic cap on aggregate a-axis size to 0.5 mm ("implicit
breakup"), with number re-diagnosis to keep mass `qi` consistent (`cni`
re-derived from the capped `ani` via the fixed `deltastr` shape). Ports
lines 2604-2612 verbatim. Does NOT include the PRECEDING re-derivation of
`ani` from `qi`/`ni`/`rhobar`/`alphstr` (lines 2590-2601) -- pass in the
caller's ALREADY-DIAGNOSED `ani`/`cni`/`ni` (the host decides how those
were obtained; this is a pure diagnostic helper per the task framing).
When `ani <= 0.5mm` the cap never binds and `(ani,cni,ni)` pass through
UNCHANGED (the Fortran's own `if` guard, lines 2604-2612 -- nothing
upstream of this cap is re-derived by this function either way).
"""
function ishmael_agg_size_cap(ani::Float64, cni::Float64, ni::Float64, dsdum::Float64,
                               ao::Float64, qi::Float64, rbdum::Float64, NU::Float64,
                               gammnu::Float64; QNSMALL::Float64=ISHMAEL_QNSMALL,
                               fourthirdspi::Float64=4.0 / 3.0 * ISHMAEL_PI)
    if ani > 0.5e-3
        ani_out = 0.5e-3
        cni_out = ao^(1.0 - dsdum) * ani_out^dsdum
        gam = gamma(NU + 2.0 + dsdum)
        ni_out = qi / (rbdum * fourthirdspi * ao^(1.0 - dsdum) * ani_out^(2.0 + dsdum) * gam / gammnu)
        ni_out = max(ni_out, QNSMALL)
        return (ani=ani_out, cni=cni_out, ni=ni_out)
    else
        return (ani=ani, cni=cni, ni=ni)
    end
end
