# ── Microphysics → cloud-optics conversion (Stage S3a) ─────────────────────
#
# Pure column physics: turns the model's PROGNOSTIC densities (cloud liquid,
# rain, and the twelve ISHMAEL ice moments) into the five column arrays an
# RRTMGP-style cloud-optics lookup wants -- liquid/ice water path, liquid/ice
# effective radius, and a 0/1 cloud fraction (plan section D5). This file
# names no radiative-transfer library, allocates nothing in the per-column
# loop, and takes plain `AbstractVector`/`AbstractMatrix` column arrays, so
# it is testable with no ModelTile, no solver and no artifacts -- the same
# split `radiation_state.jl` documents for the rest of the driver.
#
# It is included AFTER moist_compressible.jl (for `_ice_effective`, whose
# gamma-PSD moments are the ice half of the mapping, and `ISHMAEL_NU` from
# ishmael.jl, included earlier still) and BEFORE radiation.jl (whose driver
# calls `cloud_optics_column!` once per column, per radiation update).

using SpecialFunctions: gamma

# RRTMGP's cloud-optics lookup tables are populated only over these ranges;
# outside them the library silently clamps (plan D2, caveat 2), so every
# excursion here is counted instead of silently accepted.
const RAD_RE_LIQ_MIN = 2.5      # micron
const RAD_RE_LIQ_MAX = 21.5     # micron
const RAD_RE_ICE_MIN = 5.0      # micron
const RAD_RE_ICE_MAX = 90.0     # micron
const RAD_RE_LIQ_BENIGN = 10.0  # micron, written where lwp == 0 (RRTMGP ignores it there)
const RAD_RE_ICE_BENIGN = 30.0  # micron, written where the ice weighting is empty

# Column offset of each ISHMAEL species' `rho_ik` in the `(nlay, 12)` ice matrix.
# `MC_ICE_VARS` (src/moist_compressible.jl) is species-major, four moments per
# species in the order `(rho, n, a, c)`, so species `s`'s block starts at
# column `4*(s-1) + 1`.
const RAD_ICE_SPECIES_COL0 = (1, 5, 9)

"""
    CloudOpticsParams(options, physical_params)

Physical constants for [`cloud_optics_column!`](@ref), resolved ONCE by the radiation
driver from the same option/physical-parameter tables the rest of the model reads
(`get(..., default)`, the style `validate_radiation_options` in `radiation_state.jl`
uses).

- `N_c` [#/cm³]: `physical_params[:max_N_c]`, default 100.0 -- the SAME fixed droplet
  number the warm-rain closure uses (`mc_driver!`, `src/moist_compressible.jl:6911`),
  so the radiation liquid effective radius and the microphysics droplet radius never
  disagree about `N_c`.
- `k_factor`: `physical_params[:cloud_k_factor]`, default 0.8 -- the Martin et al.
  (1994) maritime spectral-dispersion factor relating the volume-mean radius
  `cloud_droplet_radius` returns to the effective radius RRTMGP wants.
- `q_min` [kg/kg]: `physical_params[:radiation_q_min]`, default 1e-6 -- the
  cloud-fraction threshold (plan D5/D7).
- `rain_in_cloud`: `options[:radiation_rain_in_cloud]`, default `false` -- rain's
  per-mass extinction is far below cloud's and its size is far above the liquid
  table's 21.5 µm edge, so it is excluded from `lwp` unless explicitly requested.
- `ice_on`: `options[:ice_microphysics] !== :none`. Informational only -- the driver
  (not this file) decides whether to pass an ice matrix or `nothing` to
  [`cloud_optics_column!`](@ref) at every call; this field just records the
  configuration for diagnostics/output and is never read inside the per-column loop.
"""
struct CloudOpticsParams
    N_c::Float64
    k_factor::Float64
    q_min::Float64
    rain_in_cloud::Bool
    ice_on::Bool
end

function CloudOpticsParams(options, physical_params)
    N_c = Float64(get(physical_params, :max_N_c, 100.0))
    k_factor = Float64(get(physical_params, :cloud_k_factor, 0.8))
    q_min = Float64(get(physical_params, :radiation_q_min, 1.0e-6))
    rain_in_cloud = get(options, :radiation_rain_in_cloud, false)::Bool
    ice_on = get(options, :ice_microphysics, :none) !== :none
    return CloudOpticsParams(N_c, k_factor, q_min, rain_in_cloud, ice_on)
end

"""
    cloud_optics_column!(cld, P, rho_d, rho_c, rho_r, ice, dz) -> (n_clamp_liq, n_clamp_ice)

Fill one column's [`CloudOpticsColumn`](@ref) (`lwp`, `iwp`, `re_liq`, `re_ice`, `cf`)
from the model's prognostic condensate. Pure, allocation-free, serial -- the radiation
driver calls this once per column, per radiation update, outside the threaded column
loop (same calling convention as `radiation_column_state!`/`radiation_levels!`).

# Arguments
- `rho_d`, `rho_c`, `rho_r`: dry-air, cloud-liquid and rain partial densities [kg/m³],
  length `nlay` (layer midpoints, index 1 = surface, matching `RadiationWork`).
- `ice`: `(nlay, 12)` matrix of the twelve ISHMAEL moments in [`MC_ICE_VARS`](@ref)
  order (species-major: `rho_i1,n_i1,a_i1,c_i1, rho_i2,n_i2,a_i2,c_i2, rho_i3,n_i3,
  a_i3,c_i3`), in the model's own DENSITY units [kg/m³, #/m³, m³/m³, m³/m³] -- NOT the
  mixing ratios [`_ice_effective`](@ref) takes; this function does that division
  itself, exactly as `mc_ice_sources!` does at `src/moist_compressible.jl:5828-5837`
  (`q = max(ρ, 0)/ρ_d`, etc., with `ρ_d` the DRY-air density -- `rhoair = rho_d[i]`
  there). Pass `nothing` when `:ice_microphysics` is off; every ice output is then the
  clear-sky default (`iwp = 0`, `re_ice` benign) and `n_clamp_ice` is always 0.
- `dz`: layer thickness [m], length `nlay`.

Returns the number of CLOUDY points where `re_liq`/`re_ice` had to be clamped into
RRTMGP's valid table range. The driver accumulates these into `RadiationState`'s
`n_clamp_re_liq`/`n_clamp_re_ice` counters (never silently reset) -- a plain
`NTuple{2,Int}` was chosen over a dedicated counter struct because the driver already
owns the long-lived counters and this function's whole contract is "how many did I
just clamp", not "how many have there ever been".

# Clamp accounting (S4)
Only layers with `cf == 1` are counted. RRTMGP MASKS the cloud optics of every layer
whose cloud fraction is zero, so an effective radius written there is never read and a
clamp there is not a saturated table -- it is a clear layer. Counting them made the
counter useless: on the S3b warm arm roughly a third of all gridpoints carried a trace
positive `rho_c` below `q_min`, and every one of them clamped low. `cf` is therefore
computed FIRST, before either radius, and the two counters are gated on it. The clamped
VALUES are still written in both cases -- RRTMGP wants a finite in-table number in every
cell whether or not it reads it.

# Water paths
`lwp = 1000 * max(rho_c [+ rho_r if rain_in_cloud], 0) * dz` [g/m² per layer]. Rain is
added to the SUM before the floor, not floored on its own, so a resolution-diagnostic
negative `rho_c` can be offset by a positive `rho_r` (and vice versa) -- the model
never floors one partial density in isolation, and this mirrors that.

`iwp = 1000 * max(rho_i1 + rho_i2 + rho_i3, 0) * dz`, summed BEFORE the floor for the
same reason, over all three species UNCONDITIONALLY (species 3 = aggregates folds into
IWP like WRF-RRTMG's snow category: Iacono et al. 2008; Thompson et al. 2008 make the
same choice for their own aggregate species).

# Liquid effective radius
Monodisperse volume radius (`cloud_droplet_radius`, `src/microphysics.jl:502-527`,
which already returns micron) rescaled to RRTMGP's effective radius by the Martin et
al. (1994) maritime spectral-dispersion factor `k = 0.8`: `r_eff = r_vol / cbrt(k)`.
Clamped to [2.5, 21.5] µm, counting IN CLOUDY LAYERS ONLY (see "Clamp accounting").
Set to a benign in-range value (10.0 µm, ignored by RRTMGP wherever `lwp == 0`) so a
clear layer never carries a zero or NaN effective radius into the solver.

THE `q_min` WINDOW (documented, intended, S4). At the fixed `N_c = 100` cm⁻³ the
monodisperse radius reaches the table floor `2.5 µm` at

    r_vol = 2.5 · k^{1/3} = 2.321 µm  ⟹  ρ_c = (4/3)π r_vol³ ρ_w N_c = 5.2e-6 kg/m³,

whereas the cloud-fraction threshold `q_min = 1e-6 kg/kg` calls a layer cloudy from
`ρ_c ≈ 1.2e-6 kg/m³` upward. Layers in the window `1.2e-6 < ρ_c < 5.2e-6 kg/m³` are
therefore CLOUD whose true effective radius is below the table, and they are clamped to
2.5 µm and counted. That is intended and is kept: at O01's 250 m spacing such a layer
carries under 1 g/m² of liquid water path (optical depth ~1e-3), so the clamped size is
radiatively irrelevant, while RAISING `q_min` to close the window would start discarding
real thin cloud. The counter is the honest record of it, not an error signal.

# Ice effective radius -- the gamma-PSD moment factor
[`_ice_effective`](@ref) (via [`ishmael_var_check`](@ref)) returns `rni`, the gamma
size distribution's SCALE length -- NOT a mean -- and `deltastr` (`δ`), the
axis-ratio shape exponent. The population is `n(a) ∝ a^{ν-1} e^{-a/a_n}` in the
crystal's a-axis `a`, with `a_n ≡ rni` and `ν = ISHMAEL_NU = 4`; the equivalent-volume
sphere radius at a-axis length `a` is `r(a) = rni · (a/a_n)^{(2+δ)/3}` (`ishmael_tables.jl`'s
own `betam = 2+δ` mass-radius exponent). RRTMGP's `r_eff` is the standard `⟨r³⟩/⟨r²⟩`
(three-quarters volume-to-area ratio for spheres), and both moments of a gamma
distribution have the closed form

    ⟨r^p⟩ = ∫ r(a)^p n(a) da / ∫ n(a) da = rni^p · Γ(ν + p(2+δ)/3) / Γ(ν)

(the `a_n` powers cancel between the numerator and denominator gamma integrals, and so
does `Γ(ν)`), giving

    r_eff,k = rni · Γ(ν + 2 + δ) / Γ(ν + (4 + 2δ)/3)

-- exactly the closed form the plan specifies. SPHERICAL SANITY CHECK: at δ = 1
(aspect ratio 1, `cni = ani`) and `ν = 4`, `Γ(7)/Γ(6) = 720/120 = 6`, so
`r_eff = 6·rni` at solid-ice density, the standard `(ν+2)/ν` gamma-distribution
effective-radius result (`test_radiation_cloud_optics.jl` asserts exactly this as its
round-trip check).

# Ice effective radius -- the BULK-DENSITY correction (S4)

The gamma moment above is `(3/4)V/A` for a population of spheres of the PARTICLE's own
bulk density `rhobar`, because `rni` is defined by `ishmael_var_check` (`src/ishmael_tables.jl:454`,
and again at :480 under the large-ice cap) as

    rni = [ 3 q Γ(ν) / (4π n rhobar Γ(ν+2+δ)) ]^{1/3},

i.e. the radius of a sphere of density `rhobar` carrying the particle's mass. That is
NOT what an ice-optics table wants. Fu (1996)'s generalized effective size -- the
quantity RRTMGP's `re_ice` parameterizes, and the one that makes the extinction per unit
IWC come out right -- is built on the SOLID-ICE volume `m/ρ_ice` divided by the
particle's PROJECTED AREA:

    D_ge = (2√3 / 3) · IWC / (ρ_ice A),      r_eff = (3/4) V_solid / A.

A low-density particle of a given `rni` has the same projected area as a solid-ice
particle of that `rni` but only `rhobar/ρ_ice` of its solid volume, so

    r_eff,k = rni_k · Γ(ν+2+δ_k)/Γ(ν+(4+2δ_k)/3) · rhobar_k / RHOI,   RHOI = 920 kg/m³.

The factor is LINEAR in `rhobar`, not its cube root: the volume changes, the area does
not. `rhobar` is `_ice_effective`'s own `rhobar`, already clamped by `ishmael_var_check`
to `[50, RHOI]`, so the factor lies in `[0.0543, 1]`.

This is the fix for the S3b finding that the ISHMAEL AGGREGATES (species 3, whose bulk
density floors at 50 kg/m³) pinned at the 90 µm table ceiling with ~15k clamps per call
and an IWP-weighted `re_ice` of 78 µm: at `rhobar = 50` their effective size is 5.4 % of
the uncorrected value, which is where a low-density aggregate's radiative size actually
is. It also moves the low end of the reachable range down by the same factor (2 µm ×
4.342 × 0.0543 ≈ 0.47 µm), so the `RAD_RE_ICE_MIN = 5 µm` floor -- previously
unreachable -- is now a live clamp for near-massless low-density crystals.

TODO (documented refinement, still not shipped): the moment above uses the EQUAL-VOLUME
SPHERE's projected area. For the non-spherical ISHMAEL habits the true area is the
spheroid's Cauchy mean projected area (`S/4`, a function of `ani/cni`), which is LARGER
than the equal-volume sphere's for both oblate and prolate shapes, so `r_eff` is still
overestimated by that area ratio -- see plan D5. The bulk-density correction had to come
first because it is the larger factor (up to 18×) and because the Cauchy ratio multiplies
it rather than replacing it.

# Combination across species
Harmonic mass weighting, `re_ice = IWC_tot / Σ_k (IWC_k / r_eff,k)`, which preserves
`Σ β_ext ∝ Σ IWC_k / r_k` (the quantity that actually sets the optical depth) rather
than averaging radii directly. `IWC_tot` and the sum run over the SAME set of
species -- those with a carried number density `n_k > 0` (a species with mass but no
number has no gamma PSD to take a moment of; `_ice_species_pre`'s "population gate",
`src/moist_compressible.jl:5860-5900`, makes the identical call for the process rates)
-- so the combined `r_eff` is guaranteed to lie within `[min_k r_eff,k, max_k
r_eff,k]`. Gated on `IWC_tot > 0`; clamped to [5, 90] µm, counting; benign 30.0 µm
where the weighting is empty (whether because `iwp == 0` or because every species with
positive mass there lacks a valid number density).

# Cloud fraction
`cf = (max(rho_c,0) + Σ_k max(rho_ik,0)) / rho_d > q_min ? 1.0 : 0.0`, STRICTLY 0 or 1
-- Scythe resolves its clouds and has no subgrid scheme, so McICA overlap is
deterministic. The ice sum here is unconditional (not gated by number), unlike the
`re_ice` weighting: cloud fraction only asks "is there condensate", not "do I know its
size".

# Vapor
Not handled here -- `vmr_h2o` (plan D5's last bullet) is a whole-column quantity the
main radiation driver fills directly from `rho_v`/`rho_d`, not a `CloudOpticsColumn`
field.
"""
function cloud_optics_column!(cld::CloudOpticsColumn, P::CloudOpticsParams,
                              rho_d::AbstractVector, rho_c::AbstractVector,
                              rho_r::AbstractVector,
                              ice::Union{Nothing,AbstractMatrix},
                              dz::AbstractVector)
    nlay = length(rho_d)
    n_clamp_liq = 0
    n_clamp_ice = 0

    @inbounds for k in 1:nlay
        rd = rho_d[k]

        # ── Condensate sums, and the cloud fraction they decide (FIRST, S4) ──
        # The mass sums have to be taken before either effective radius, because the
        # clamp counters are gated on `cf`: a clamp in a layer RRTMGP masks out is not a
        # saturated table. `rho_i_sum` is the RAW (unfloored) species sum for `iwp`;
        # `cf_ice_sum` is Σ max(rho_ik, 0), unconditional (cloud fraction asks "is there
        # condensate", not "do I know its size"), and is used for `cf` alone.
        rho_i_sum = 0.0
        cf_ice_sum = 0.0
        if ice !== nothing
            for s in 1:3
                rho_ik = ice[k, RAD_ICE_SPECIES_COL0[s]]
                rho_i_sum += rho_ik
                cf_ice_sum += max(rho_ik, 0.0)
            end
        end
        # Strict threshold, 0/1 only (McICA determinism).
        cloudy = (max(rho_c[k], 0.0) + cf_ice_sum) / rd > P.q_min
        cld.cf[k] = cloudy ? 1.0 : 0.0

        # ── Liquid water path and effective radius ──
        liq_sum = rho_c[k] + (P.rain_in_cloud ? rho_r[k] : 0.0)
        lwp_k = 1000.0 * max(liq_sum, 0.0) * dz[k]
        cld.lwp[k] = lwp_k

        if lwp_k > 0.0
            q_c = max(rho_c[k], 0.0) / rd
            r_vol = cloud_droplet_radius(P.N_c, q_c, rd)   # already micron
            r_eff = r_vol / cbrt(P.k_factor)
            if r_eff < RAD_RE_LIQ_MIN
                r_eff = RAD_RE_LIQ_MIN
                cloudy && (n_clamp_liq += 1)
            elseif r_eff > RAD_RE_LIQ_MAX
                r_eff = RAD_RE_LIQ_MAX
                cloudy && (n_clamp_liq += 1)
            end
            cld.re_liq[k] = r_eff
        else
            cld.re_liq[k] = RAD_RE_LIQ_BENIGN
        end

        # ── Per-species ice effective radius, harmonic combination ──
        iwc_weight_sum = 0.0   # Σ_k IWC_k over species with a valid (n>0) gamma PSD
        inv_re_sum = 0.0       # Σ_k IWC_k / r_eff,k, same species set as iwc_weight_sum

        if ice !== nothing
            for s in 1:3
                c0 = RAD_ICE_SPECIES_COL0[s]
                rho_ik = ice[k, c0]
                q = max(rho_ik, 0.0) / rd
                n = max(ice[k, c0 + 1], 0.0) / rd
                if q > 0.0 && n > 0.0
                    a = max(ice[k, c0 + 2], 0.0) / rd
                    c = max(ice[k, c0 + 3], 0.0) / rd
                    eff = _ice_effective(q, n, a, c, s)
                    delta = eff.deltastr
                    # r_eff,k = rni * Gamma(nu+2+delta)/Gamma(nu+(4+2*delta)/3)
                    #                * rhobar / RHOI
                    # The gamma factor is the population moment of the EQUAL-VOLUME
                    # SPHERE radius (delta=1, nu=4 -> 6*rni); `rhobar/RHOI` converts the
                    # particle's own bulk volume to the SOLID-ICE volume Fu (1996)'s
                    # generalized effective size is built on, at unchanged projected
                    # area. Both steps are derived in the docstring. `rhobar` comes
                    # clamped to [50, RHOI] by `ishmael_var_check`, so the factor is in
                    # [0.0543, 1] and can never invert the sign or blow up.
                    r_eff_k = eff.rni * 1.0e6 *
                              (gamma(ISHMAEL_NU + 2.0 + delta) /
                               gamma(ISHMAEL_NU + (4.0 + 2.0 * delta) / 3.0)) *
                              (eff.rhobar / ISHMAEL_RHOI)
                    iwc_k = max(rho_ik, 0.0)
                    iwc_weight_sum += iwc_k
                    inv_re_sum += iwc_k / r_eff_k
                end
            end
        end

        cld.iwp[k] = 1000.0 * max(rho_i_sum, 0.0) * dz[k]

        if iwc_weight_sum > 0.0
            r_eff = iwc_weight_sum / inv_re_sum
            if r_eff < RAD_RE_ICE_MIN
                r_eff = RAD_RE_ICE_MIN
                cloudy && (n_clamp_ice += 1)
            elseif r_eff > RAD_RE_ICE_MAX
                r_eff = RAD_RE_ICE_MAX
                cloudy && (n_clamp_ice += 1)
            end
            cld.re_ice[k] = r_eff
        else
            cld.re_ice[k] = RAD_RE_ICE_BENIGN
        end
    end

    return (n_clamp_liq, n_clamp_ice)
end
