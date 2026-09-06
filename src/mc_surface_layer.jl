# ── Shared ocean surface-exchange layer ────────────────────────────────────────
#
# ONE pure function, `surface_exchange`, that turns the lowest mish-level state
# (u, v, T, rho_d, rho_t, rho_v, p, z) and a fixed SST into the surface stress,
# sensible-heat and moisture fluxes. It is the single place the model computes a
# bulk air-sea exchange: the Louis boundary layer (`mc_louis_bl!`,
# src/mc_boundary_layer.jl) calls it today, the MYNN-EDMF closure will call the
# same function, and `benchmarks/ocean_warm_bubble.jl` re-evaluates its budget
# rows through it so a diagnostic can never drift from what the model applied.
#
# Two INDEPENDENT, option-selected layers sit on top of the same bulk formulas:
#
#   1. `options[:sfc_z0]` — where the NEUTRAL wind-speed dependence comes from:
#        :komori   (default) the Komori et al. (2018) Cd(U) fit and a constant Ck,
#                  i.e. exactly the closure this model has always run;
#        :gfdl_v7  the HWRF/HAFS z0m(U10), z0t(U10) polynomial fits (Bin Liu 2018),
#                  with Cd_N, Ch_N from the neutral log law at z1;
#        :charnock Charnock + Zeng et al. (1998) thermal roughness, as UFS sfc_diff.
#   2. `options[:sfc_stability]::Bool` — whether Monin-Obukhov stability functions
#      (and a Beljaars convective gustiness) modify those neutral coefficients.
#      Default false.
#
# The DEFAULT (`:komori`, no stability) path is BITWISE the pre-S1b inline code of
# `mc_louis_bl!`: same expressions, same association, same order. It is written
# out as its own branch for exactly that reason — do not "simplify" it into the
# general path.
#
# Zero allocations per call: `SurfaceLayerParams` is an immutable isbits struct
# built ONCE per driver call from `physical_params`/`options` (no Dict lookup in
# the hot path), the return value is an isbits NamedTuple, and nothing here
# captures an array.

"""
    SurfaceLayerParams

Immutable, isbits configuration of [`surface_exchange`](@ref). Build it once per
driver call with [`surface_layer_params`](@ref) — never inside a column loop.

# Fields
- `Cd_param::Float64` — `physical_params[:Cd]`; NEGATIVE selects the Komori fit
  (the tcbl sentinel convention), positive is a constant, `0.0` disables the drag.
- `Ck::Float64` — `physical_params[:Ck]`, the constant enthalpy/moisture exchange
  coefficient of the `:komori` path (ignored by `:gfdl_v7`/`:charnock`, which
  build `Ch` from a thermal roughness length).
- `U_min::Float64` — gustiness floor on the exchange wind [m/s].
- `sfc_fac::Float64` — `physical_params[:sfc_wind_factor]`, the multiplier taking
  the lowest mish-level wind to the exchange wind.
- `SST::Float64` — sea surface temperature [K].
- `z0_mode::Symbol` — `:komori`, `:gfdl_v7` or `:charnock` (see the file header).
- `stability::Bool` — Monin-Obukhov stability functions on/off.
- `surface_fluxes::Bool` — whether the enthalpy/moisture fluxes are computed at
  all (`options[:surface_fluxes]`); with it off both are an exact `0.0` and the
  stability iteration sees a neutral surface layer.
- `n_stab_iter::Int` — fixed-point iterations of the `z/L` solve (default 3).
  A knob for convergence studies only; production runs never change it.
"""
struct SurfaceLayerParams
    Cd_param::Float64
    Ck::Float64
    U_min::Float64
    sfc_fac::Float64
    SST::Float64
    z0_mode::Symbol
    stability::Bool
    surface_fluxes::Bool
    n_stab_iter::Int
end

"The `options[:sfc_z0]` values `surface_layer_params` accepts."
const SFC_Z0_MODES = (:komori, :gfdl_v7, :charnock)

"""
    surface_layer_params(physical_params, options; surface_fluxes) -> SurfaceLayerParams

Resolve the surface-layer configuration ONCE, at driver-preamble time, and
validate it LOUDLY: an unrecognized `options[:sfc_z0]` dies at setup with the
valid values listed, rather than silently falling back to a default the run did
not ask for (the `validate_radiation_options` convention, src/radiation_state.jl).
"""
function surface_layer_params(physical_params, options; surface_fluxes::Bool = false)
    z0_mode = get(options, :sfc_z0, :komori)
    z0_mode isa Symbol && z0_mode in SFC_Z0_MODES || error(
        "options[:sfc_z0] = $(repr(z0_mode)) is not a surface roughness closure. The " *
        "valid values are " * join(string.(":", SFC_Z0_MODES), ", ") * " (default :komori)")
    stability = get(options, :sfc_stability, false)::Bool
    n_iter = Int(get(options, :sfc_stability_iterations, 3))
    n_iter >= 1 || error("options[:sfc_stability_iterations] must be >= 1, got $n_iter")
    return SurfaceLayerParams(get(physical_params, :Cd, -1.0),
                              get(physical_params, :Ck, 1.0e-3),
                              get(physical_params, :U_min, 0.0),
                              get(physical_params, :sfc_wind_factor, 1.0),
                              get(physical_params, :SST, 301.15),
                              z0_mode, stability, surface_fluxes, n_iter)
end

"""
    komori_cd(U)

Wind-speed-dependent surface drag coefficient, Komori et al. (2018): 1.0e-3 below
5.2 m/s, 4.4e-4 U^0.5 to 33.6 m/s, capped at 2.55e-3 in the high-wind regime.
Selected by a NEGATIVE configured `:Cd` (the tcbl sentinel convention); a positive
`:Cd` is used as a constant, and `:Cd = 0` disables the drag.
"""
@inline function komori_cd(U)
    if U < 5.2
        return 1.0e-3
    elseif U < 33.6
        return 4.4e-4 * sqrt(U)
    end
    return 2.55e-3
end

# ── Constants ─────────────────────────────────────────────────────────────────

"von Karman constant of the surface layer (the same 0.4 as `louis_length`)."
const SFC_KARMAN = 0.4
"Kinematic viscosity of air [m^2/s] used by the Charnock thermal roughness."
const SFC_NU_AIR = 1.5e-5
"UFS `sfc_diff.f` `z0s_max`: the high-wind cap on the Charnock roughness [m]."
const SFC_Z0S_MAX = 3.17e-3
"Beljaars (1995) convective boundary-layer depth used by the gustiness velocity [m]."
const SFC_Z_I = 1000.0
"Beljaars (1995) gustiness coefficient: `U_eff^2 = U^2 + (beta w*)^2`."
const SFC_BETA_GUST = 1.2

# ── GFDL/HWRF roughness fits (Bin Liu, NOAA/NCEP/EMC 2018) ────────────────────

"""
    znot_m_v7(uref)

Aerodynamic roughness length over water [m] from the 10-m wind `uref` [m/s].
Port of `SUBROUTINE znot_m_v7` in
`ccpp-physics/physics/SFC_Layer/GFDL/module_sf_exchcoef.f90` (Apache-2.0; Bin Liu,
NOAA/NCEP/EMC 2018), which matches the COARE v3.5 Cd-U10 relationship (Edson et
al. 2013) at low-to-moderate winds and observational fits at high winds.

Piecewise in `uref`: exponential-of-cubic below 6.5 m/s, quintic to 15.7 m/s,
exponential-of-quintic to 53 m/s, constant `3.371427455376717e-04` above. The
integer powers are expanded the way gfortran expands `**n` (binary), so the
transcription stays within a few ulp of the Fortran (see
`test/reference/gfdl_sfc_refs.jl`).
"""
@inline function znot_m_v7(uref::Float64)
    u2 = uref * uref
    u3 = u2 * uref
    u4 = u2 * u2
    u5 = u4 * uref
    if uref >= 0.0 && uref <= 6.5
        return exp(-8.396975715683501e+00 + (-1.597898515251717e+00 * uref) +
                   (2.855780863283819e-01 * u2) + (-1.296521881682694e-02 * u3))
    elseif uref > 6.5 && uref <= 15.7
        return (3.790846746036765e-10 * u5) + (3.281964357650687e-09 * u4) +
               (1.962282433562894e-07 * u3) + (-1.240239171056262e-06 * u2) +
               (1.739759082358234e-07 * uref) + 2.147264020369413e-05
    elseif uref > 15.7 && uref <= 53.0
        return exp((1.897534489606422e-07 * u5) + (-3.019495980684978e-05 * u4) +
                   (1.931392924987349e-03 * u3) + (-6.797293095862357e-02 * u2) +
                   (1.346757797103756e+00 * uref) + (-1.707846930193362e+01))
    end
    return 3.371427455376717e-04
end

"""
    znot_t_v7(uref)

Scalar (thermal) roughness length over water [m] from the 10-m wind `uref` [m/s].
Port of `SUBROUTINE znot_t_v7` in
`ccpp-physics/physics/SFC_Layer/GFDL/module_sf_exchcoef.f90` (Apache-2.0; Bin Liu,
NOAA/NCEP/EMC 2018): the COARE Ck-U10 relationship at low-to-moderate winds,
retaining the FY2015 HWRF Ck-U10 relationship at high winds so it is consistent
with the slightly reduced `znot_m_v7` drag there.

Six pieces plus two constants: `1.1e-4` below 5.9 m/s and
`6.840803042788488e-05` above 80 m/s.
"""
@inline function znot_t_v7(uref::Float64)
    u2 = uref * uref
    u3 = u2 * uref
    u4 = u2 * u2
    u5 = u4 * uref
    u6 = u4 * u2
    if uref >= 0.0 && uref < 5.9
        return 1.100000000000000e-04
    elseif uref >= 5.9 && uref <= 15.4
        return (-9.193764479895316e-10 * u5) + (7.052217518653943e-08 * u4) +
               (-2.163419217747114e-06 * u3) + (3.342963077911962e-05 * u2) +
               (-2.633566691328004e-04 * uref) + 8.644979973037803e-04
    elseif uref > 15.4 && uref <= 21.6
        return (-9.402722450219142e-12 * u5) + (1.325396583616614e-09 * u4) +
               (-7.299148051141852e-08 * u3) + (1.982901461144764e-06 * u2) +
               (-2.680293455916390e-05 * uref) + 1.484341646128200e-04
    elseif uref > 21.6 && uref <= 42.6
        return (7.921446674311864e-12 * u5) + (-1.019028029546602e-09 * u4) +
               (5.251986927351103e-08 * u3) + (-1.337841892062716e-06 * u2) +
               (1.659454106237737e-05 * uref) + (-7.558911792344770e-05)
    elseif uref > 42.6 && uref <= 53.0
        return (-2.694370426850801e-10 * u5) + (5.817362913967911e-08 * u4) +
               (-5.000813324746342e-06 * u3) + (2.143803523428029e-04 * u2) +
               (-4.588070983722060e-03 * uref) + 3.924356617245624e-02
    elseif uref > 53.0 && uref <= 80.0
        return (-1.663918773476178e-13 * u6) + (6.724854483077447e-11 * u5) +
               (-1.127030176632823e-08 * u4) + (1.003683177025925e-06 * u3) +
               (-5.012618091180904e-05 * u2) + (1.329762020689302e-03 * uref) +
               (-1.450062148367566e-02)
    end
    return 6.840803042788488e-05
end

# ── Monin-Obukhov stability functions ─────────────────────────────────────────
#
# `psi_m_sfc`/`psi_h_sfc` are the INTEGRATED stability corrections that go with
# the `phim`/`phih` MYNN-EDMF uses (`module_bl_mynn.F90` :7528-7626, "New
# stability function parameters ... (Puhales, 2020, WRF 4.2.1)"). The relation
# between the two is phi(zeta) = 1 - zeta * dpsi/dzeta, and the Fortran is
# written in exactly that form (`phi_m = 1-zet*dummy_2`).
#
# STABLE (zeta >= 0) — Cheng and Brutsaert (2005), valid to zeta ~ O(10):
#     psi_m = -a ln[zeta + (1 + zeta^b )^(1/b)],  a = 6.1, b = 2.5
#     psi_h = -c ln[zeta + (1 + zeta^d )^(1/d)],  c = 5.3, d = 1.1
#   Differentiating gives precisely the Fortran's `dummy_2` with am_st/bm_st and
#   ah_st/bh_st, so these psi integrate MYNN's stable phi EXACTLY.
#
# UNSTABLE (zeta < 0) — Businger-Dyer / Kansas (Paulson 1970, Dyer & Hicks 1970),
#   with x = (1 - 16 zeta)^(1/4) (MYNN's `cphm_unst = cphh_unst = 16.0`):
#     psi_m = 2 ln((1+x)/2) + ln((1+x^2)/2) - 2 atan(x) + pi/2
#     psi_h = 2 ln((1+x^2)/2)
#   These are `dummy_psi` in the Fortran, verbatim (its `1.570796` is pi/2 and its
#   `dummy_0` is x, resp. x^2 for heat). DEVIATION, stated plainly: MYNN's unstable
#   phi integrates the GRACHEV et al. (2000) BLEND of this Kansas psi with a free-
#   convection psi_c (the Fortran's `dummy_4`), weighted 1/(1+zeta^2) and
#   zeta^2/(1+zeta^2). This module implements the Kansas member alone, as specified
#   for stage S1b — i.e. the weakly-unstable limit (-1 < zeta < 0), where the blend
#   weight on psi_c is < 1/2. Adding the convective member is a one-function change
#   here if the deep-unstable limit ever matters.

"""
    psi_m_sfc(zeta)

Integrated Monin-Obukhov stability correction for MOMENTUM. Cheng & Brutsaert
(2005) for `zeta >= 0`, Businger-Dyer/Kansas for `zeta < 0`; see the comment block
above for which `phim` (MYNN-EDMF `module_bl_mynn.F90`) each branch integrates.
`psi_m_sfc(0) == 0`, `psi_m_sfc < 0` when stable (drag reduced) and `> 0` when
unstable (drag enhanced).
"""
@inline function psi_m_sfc(zeta::Float64)
    if zeta >= 0.0
        a = 6.1; b = 2.5
        return -a * log(zeta + (1.0 + zeta^b)^(1.0 / b))
    end
    x = (1.0 - (16.0 * zeta))^0.25
    return (2.0 * log(0.5 * (1.0 + x))) + log(0.5 * (1.0 + (x * x))) -
           (2.0 * atan(x)) + 1.5707963267948966
end

"""
    psi_h_sfc(zeta)

Integrated Monin-Obukhov stability correction for HEAT (and, here, moisture).
Cheng & Brutsaert (2005) `-5.3 ln[zeta + (1+zeta^1.1)^(1/1.1)]` for `zeta >= 0`;
Businger-Dyer `2 ln((1+x^2)/2)`, `x = (1-16 zeta)^(1/4)`, for `zeta < 0`.
"""
@inline function psi_h_sfc(zeta::Float64)
    if zeta >= 0.0
        c = 5.3; d = 1.1
        return -c * log(zeta + (1.0 + zeta^d)^(1.0 / d))
    end
    x2 = sqrt(1.0 - (16.0 * zeta))
    return 2.0 * log(0.5 * (1.0 + x2))
end

# The z/L the iteration is allowed to reach. Cheng-Brutsaert is documented to
# zeta ~ O(10) and the Kansas forms are a weakly-unstable fit, so clamping is
# honest about where the functions stop meaning anything — and it keeps a single
# pathological column from producing an Inf coefficient.
const SFC_ZETA_MAX = 10.0

# ── Neutral roughness lengths ─────────────────────────────────────────────────

"""
    sfc_roughness(mode, U1, z1, Cd_param, Ck) -> (z0m, z0t, U10)

Neutral momentum and thermal roughness lengths [m] at the lowest model level `z1`
for the exchange wind `U1`, and the neutral-log-profile 10-m wind `U10` [m/s] that
goes with the roughness returned (the value `:gfdl_v7` evaluated its fits at, so a
caller reporting `U10` reports the one the closure actually used).

- `:komori` — INVERT the closure the model already has: `z0m = z1 exp(-k/sqrt(Cd_N))`
  from the Komori (or constant) `Cd_N`, and `z0t = z1 exp(-k^2/(Ck ln(z1/z0m)))`
  from the constant `Ck`. This is only ever reached with stability ON; without it
  the `:komori` branch of `surface_exchange` never forms a roughness length at all.
- `:gfdl_v7` — [`znot_m_v7`](@ref)/[`znot_t_v7`](@ref) of the 10-m wind, with U10
  obtained from `U1` through the neutral log profile and iterated three times
  (z0m depends on U10, U10 depends on z0m). U10 is clamped to [1, 85] m/s, the
  range over which the Fortran fits are defined.
- `:charnock` — `z0m = 0.018 ust^2/g + 0.11 nu/ust` capped at `z0s_max = 3.17e-3 m`
  (UFS `sfc_diff.f`), with `ust` from the neutral log law and the pair iterated
  three times from a `z0m = 1e-4 m` first guess; then the Zeng et al. (1998)
  thermal roughness `z0t = z0m exp(-min(7, 2.67 Re*^(1/4) - 2.57))`,
  `Re* = ust z0m/nu`. The `min(7, .)` and the `Re* >= 1e-6` floor are UFS's
  (sfc_diff.f:369-386) and keep a calm column from producing `z0t = 0`.
"""
@inline function sfc_roughness(mode::Symbol, U1::Float64, z1::Float64,
                               Cd_param::Float64, Ck::Float64)
    if mode === :gfdl_v7
        U10 = min(max(U1, 1.0), 85.0)
        z0m = znot_m_v7(U10)
        for _ in 1:3
            U10 = min(max(U1 * log(10.0 / z0m) / log(z1 / z0m), 1.0), 85.0)
            z0m = znot_m_v7(U10)
        end
        return (z0m, znot_t_v7(U10), U10)
    elseif mode === :charnock
        z0m = 1.0e-4
        ust = 0.0
        for _ in 1:3
            Cd_N = (SFC_KARMAN * SFC_KARMAN) / (log(z1 / z0m)^2)
            ust = sqrt(Cd_N) * U1
            ust = max(ust, 1.0e-6)
            z0m = min((0.018 * ust * ust / gravity) + (0.11 * SFC_NU_AIR / ust),
                      SFC_Z0S_MAX)
        end
        re_star = max(ust * z0m / SFC_NU_AIR, 1.0e-6)
        rat = min(7.0, (2.67 * sqrt(sqrt(re_star))) - 2.57)
        return (z0m, z0m * exp(-rat), U1 * log(10.0 / z0m) / log(z1 / z0m))
    end
    # :komori — invert the model's own Cd(U)/Ck onto roughness lengths at z1
    Cd_N = Cd_param < 0.0 ? komori_cd(U1) : Cd_param
    Cd_N = max(Cd_N, 1.0e-6)
    z0m = z1 * exp(-SFC_KARMAN / sqrt(Cd_N))
    ln_zm = log(z1 / z0m)
    z0t = z1 * exp(-(SFC_KARMAN * SFC_KARMAN) / (max(Ck, 1.0e-8) * ln_zm))
    return (z0m, z0t, U1 * log(10.0 / z0m) / ln_zm)
end

# ── The one exchange function ─────────────────────────────────────────────────

"""
    surface_exchange(u1, v1, T1, rho_d1, rho_t1, rho_v1, p1_Pa, z1, SST, params)

Bulk air-sea exchange at the lowest model level. PURE and allocation-free: every
argument is a scalar, the result is an isbits `NamedTuple`

    (ust, tau_u, tau_v, F_sh, F_q, inv_L, z0m, z0t, Cd, Ch, U1, U10, w_star)

with `tau_u`, `tau_v` the momentum stress [Pa] (positive = removed from the flow,
the sign `mc_louis_bl!` subtracts), `F_sh` the sensible heat flux [W/m^2] into the
column, `F_q` the moisture flux [kg/m^2/s], `ust` the friction velocity [m/s],
`inv_L` the inverse Obukhov length [1/m], and `Cd`/`Ch` the coefficients actually
applied. `u1`/`v1` are the RAW lowest-level wind components — `params.sfc_fac` is
applied here, not by the caller.

`SST` is passed explicitly (rather than read from `params`) because the caller may
carry a per-column sea temperature later; today it is always `params.SST`.

# The default path
With `params.z0_mode === :komori` and `params.stability == false` this is
BITWISE the pre-S1b inline code of `mc_louis_bl!`:

    U1    = max(sqrt(u1^2 + v1^2), U_min)          (u1, v1 already * sfc_fac)
    Cd    = Cd_param < 0 ? komori_cd(U1) : Cd_param
    tau_u = rho_t1 * (Cd * U1) * u1
    F_sh  = rho_d1 * Cpd * Ck * U1 * (SST - T1)
    F_q   = Ck * U1 * (rho_v_sat(SST, p1_Pa/100) - rho_v1)

`z0m`, `z0t`, `inv_L` and `w_star` come back as `0.0` there, and `U10` as `U1`:
that closure never forms a roughness length or an Obukhov length, so it has no
10-m profile to evaluate either, and reporting a fabricated one would be worse
than reporting none.

# The general path
`z0m`, `z0t` come from [`sfc_roughness`](@ref) and the coefficients from the
neutral log law, `Cd_N = k^2/ln^2(z1/z0m)`, `Ch_N = k^2/(ln(z1/z0m) ln(z1/z0t))`
(one `Ch` for heat AND moisture). With `params.stability` on they become

    Cd = k^2 / [ln(z1/z0m) - psi_m(z1/L)]^2
    Ch = k^2 / {[ln(z1/z0m) - psi_m(z1/L)] [ln(z1/z0t) - psi_h(z1/L)]}

with `1/L = -k (g/theta_v1) <w'theta_v'>_s / ust^3` and
`<w'theta_v'>_s = F_sh/(rho_t1 Cpd) + 0.61 theta1 F_q/rho_t1`. The potential
temperatures are referenced to the HYDROSTATIC surface pressure
`p_s = p1 + rho_t1 g z1`, so `theta_s = SST` exactly and the ~0.55 K dry-adiabatic
offset over `z1` is not charged to the air-sea temperature difference. (The
`<w'theta_v'>` expression above is the specified one; its first term is `w'T'`
rather than `w'theta'`, an `exner(z1) ~ 0.995` inconsistency of about half a
percent at a 56 m first level.)

The iteration starts from the bulk Richardson number
`Ri_b = g z1 (theta_v1 - theta_vs)/(theta_v1 U^2)` mapped to `z/L` by the neutral
relation `zeta = Ri_b ln^2(z1/z0m)/ln(z1/z0t)`, then takes `params.n_stab_iter`
(default 3) fixed-point steps. `zeta` is clamped to `[-10, 10]` — past that the
Cheng-Brutsaert and Kansas fits are extrapolation.

Convective GUSTINESS (Beljaars 1995) rides along with stability: when the buoyancy
flux is upward, `w* = (g z_i <w'theta_v'>_s / theta_v1)^(1/3)` with `z_i = 1000 m`
enters the exchange wind as `U_eff = sqrt(U1^2 + (1.2 w*)^2)`. `U_min` still floors
`U1` first, so the two gust sources compose rather than replace one another.
"""
@inline function surface_exchange(u1::Float64, v1::Float64, T1::Float64,
                                  rho_d1::Float64, rho_t1::Float64, rho_v1::Float64,
                                  p1_Pa::Float64, z1::Float64, SST::Float64,
                                  params::SurfaceLayerParams)
    u1 = u1 * params.sfc_fac
    v1 = v1 * params.sfc_fac
    U1 = max(sqrt((u1 * u1) + (v1 * v1)), params.U_min)

    if params.z0_mode === :komori && !params.stability
        # ── THE FROZEN DEFAULT ──────────────────────────────────────────────
        # Verbatim from mc_boundary_layer.jl before S1b. Every expression's
        # association is the original's; this branch exists to be bitwise, not
        # to be tidy.
        Cd = params.Cd_param < 0.0 ? komori_cd(U1) : params.Cd_param
        drag_coeff = Cd * U1
        F_sh = 0.0
        F_q = 0.0
        if params.surface_fluxes
            p1_hPa = p1_Pa / 100.0
            F_sh = rho_d1 * Cpd * params.Ck * U1 * (SST - T1)
            F_q = params.Ck * U1 * (rho_v_sat(SST, p1_hPa) - rho_v1)
        end
        return (ust = sqrt(Cd) * U1,
                tau_u = rho_t1 * drag_coeff * u1,
                tau_v = rho_t1 * drag_coeff * v1,
                F_sh = F_sh, F_q = F_q, inv_L = 0.0, z0m = 0.0, z0t = 0.0,
                Cd = Cd, Ch = params.Ck, U1 = U1, U10 = U1, w_star = 0.0)
    end

    # ── Neutral roughness lengths and log-law coefficients ──────────────────
    z0m, z0t, U10 = sfc_roughness(params.z0_mode, U1, z1, params.Cd_param, params.Ck)
    ln_zm = log(z1 / z0m)
    ln_zt = log(z1 / z0t)
    k2 = SFC_KARMAN * SFC_KARMAN
    Cd = k2 / (ln_zm * ln_zm)
    Ch = k2 / (ln_zm * ln_zt)

    p1_hPa = p1_Pa / 100.0
    rho_vs_sfc = params.surface_fluxes ? rho_v_sat(SST, p1_hPa) : 0.0
    U_eff = U1
    inv_L = 0.0
    w_star = 0.0

    if params.stability
        # Potential temperatures against the hydrostatic surface pressure, so
        # theta_s == SST exactly (see the docstring).
        p_s = p1_Pa + (rho_t1 * gravity * z1)
        exner1 = (p1_Pa / p_s)^(Rd / Cpd)
        theta1 = T1 / exner1
        q_v1 = rho_v1 / rho_t1
        theta_v1 = theta1 * (1.0 + (0.61 * q_v1))
        q_vs = params.surface_fluxes ? rho_vs_sfc / (rho_d1 + rho_vs_sfc) : 0.0
        theta_vs = SST * (1.0 + (0.61 * q_vs))

        # Bulk-Richardson first guess mapped to z/L by the NEUTRAL relation
        # Ri_b = zeta ln(z1/z0t)/ln^2(z1/z0m).
        Ri_b = gravity * z1 * (theta_v1 - theta_vs) / (theta_v1 * U1 * U1)
        zeta = clamp(Ri_b * ln_zm * ln_zm / ln_zt, -SFC_ZETA_MAX, SFC_ZETA_MAX)

        for _ in 1:params.n_stab_iter
            fm = max(ln_zm - psi_m_sfc(zeta), 0.1)
            fh = max(ln_zt - psi_h_sfc(zeta), 0.1)
            Cd = k2 / (fm * fm)
            Ch = k2 / (fm * fh)
            U_eff = w_star > 0.0 ?
                    sqrt((U1 * U1) + ((SFC_BETA_GUST * w_star) * (SFC_BETA_GUST * w_star))) :
                    U1
            ust = max(sqrt(Cd) * U_eff, 1.0e-6)
            F_sh_it = params.surface_fluxes ?
                      rho_d1 * Cpd * Ch * U_eff * (SST - T1) : 0.0
            F_q_it = params.surface_fluxes ? Ch * U_eff * (rho_vs_sfc - rho_v1) : 0.0
            wtv = (F_sh_it / (rho_t1 * Cpd)) + (0.61 * theta1 * F_q_it / rho_t1)
            inv_L = -SFC_KARMAN * (gravity / theta_v1) * wtv / (ust * ust * ust)
            zeta = clamp(z1 * inv_L, -SFC_ZETA_MAX, SFC_ZETA_MAX)
            w_star = wtv > 0.0 ? cbrt(gravity * SFC_Z_I * wtv / theta_v1) : 0.0
        end
        U_eff = w_star > 0.0 ?
                sqrt((U1 * U1) + ((SFC_BETA_GUST * w_star) * (SFC_BETA_GUST * w_star))) :
                U1
    end

    F_sh = params.surface_fluxes ? rho_d1 * Cpd * Ch * U_eff * (SST - T1) : 0.0
    F_q = params.surface_fluxes ? Ch * U_eff * (rho_vs_sfc - rho_v1) : 0.0
    drag_coeff = Cd * U_eff
    return (ust = sqrt(Cd) * U_eff,
            tau_u = rho_t1 * drag_coeff * u1,
            tau_v = rho_t1 * drag_coeff * v1,
            F_sh = F_sh, F_q = F_q, inv_L = inv_L, z0m = z0m, z0t = z0t,
            Cd = Cd, Ch = Ch, U1 = U1, U10 = U10, w_star = w_star)
end
