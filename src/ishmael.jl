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
