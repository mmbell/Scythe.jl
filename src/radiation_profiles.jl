# ── The library-free half of the RRTMGP radiation path ───────────────────────
#
# This file names NO radiative-transfer library. RRTMGP and ClimaComms are WEAK
# dependencies of Scythe: everything that names them lives in the package extension
# ext/ScytheRRTMGPExt.jl, which Julia loads automatically once the user has done
# `using RRTMGP, ClimaComms` in the same session (and, under Distributed, on every
# worker). Without them, Scythe loads and runs every other configuration; a model that
# asks for `options[:radiation] = :rrtmgp` is refused at tile construction with a
# message naming the missing `using` (see `require_rrtmgp_extension`, src/radiation.jl).
# The split is what keeps RRTMGP's own compat bounds (notably its weak-dep
# `CUDA = "5.9, 6"`) out of any project that `Pkg.develop`s Scythe without using
# radiation.
#
# What lives here (all plain Julia on plain arrays):
#   * the fixed choices and the per-process cache containers (the extension fills them),
#   * the standard-atmosphere interpolation that builds the stratospheric extension
#     and the ozone profile Scythe does not carry (`standard_extension`,
#     `model_ozone`; the sampled profile itself, `_standard_profile`, calls
#     `RRTMGP.standard_atmosphere` and so is defined in the extension),
#   * the well-mixed gas table (`default_gases`),
#   * the boundary-row seed and the typed column-write loops that
#     `rrtmgp_solve_columns!` hands the solver's getter views to,
#   * `radiation_divergence!`, the flux-difference-to-W/m^3 store.
#
# What lives in ext/ScytheRRTMGPExt.jl: `rrtmgp_context`, `rrtmgp_method`,
# `rrtmgp_lookups`, `_standard_profile`, `rrtmgp_solver`, `rrtmgp_solve_columns!`,
# `rrtmgp_fluxes` and `radiation_offline_column`, each a METHOD of a stub declared in
# src/radiation.jl.
#
# Conventions inherited from src/radiation_state.jl and repeated because they
# are easy to get backwards:
#   * layer index 1 is the SURFACE and z ascends with the index. RRTMGP uses
#     the same convention (verified in S0), so nothing is ever flipped.
#   * `flux_net = up - down`, so a longwave-cooling layer has
#     `flux_net[k+1] > flux_net[k]` and `q = (F[k] - F[k+1])/dz < 0`.
#   * heating is W/m^3, never K/s.

# ── Fixed choices, named so they are greppable ────────────────────────────────

"""
Top of the stratospheric extension [m]. Above this, RRTMGP's own isothermal
boundary layer carries the column to the lookup tables' minimum pressure.

CAVEAT, measured and deliberately accepted: RRTMGP's idealized standard
atmosphere is a two-segment profile that warms at a constant `Γ_strat` FOREVER
above the tropopause, so at 70 km it reports ~302 K where the real mesosphere is
near 220 K (it has no stratopause). The layers concerned hold under 0.006% of
the column mass (p < 40 Pa above 50 km), and the measured effect is small: a
25 km model with a 15-layer extension to 70 km gives a clear-sky tropical OLR of
282.37 W/m^2 against S0's independent 45 km column at 281.86, i.e. 0.5 W/m^2.
For comparison, dropping the extension entirely costs 2.3 W/m^2 and dropping
the prescribed ozone with it costs 12 W/m^2 -- the extension earns its keep
mostly by CARRYING THE OZONE, not by its own thin mass. If a configuration ever
needs the upper mesosphere right, the fix is a real climatology here, not a
lower ceiling.
"""
const RRTMGP_EXTENSION_TOP = 70.0e3

"Number of layers used to SAMPLE the analytic standard atmosphere (50 m over
0-70 km). This is a sampling resolution for interpolation, not a solver grid:
the extension itself has `n_ext` layers. Fine enough that linear interpolation
of `ln p`, `T` and the mixing ratios is exact to well under a part in 10^5."
const RRTMGP_PROFILE_SAMPLES = 1400

"e-folding depth [m] of the seam temperature correction. The extension's own
temperature is an idealized climatology; the model's top-face temperature is
the truth at the seam, so the difference is blended out over ~5 km rather than
left as a discontinuity (which would radiate as a spurious thin emitting layer)."
const RRTMGP_SEAM_DEPTH = 5.0e3

"Floor on the extension's water-vapor VMR after the seam rescaling. 2 ppmv is
the stratospheric background; a column whose top layer is anomalously dry must
not scale the whole stratosphere to zero."
const RRTMGP_VMR_H2O_FLOOR = 2.0e-6

"Boundary-row seed values (Pa, K). See `_seed_boundary_row!`."
const RRTMGP_BOUNDARY_P = 1.0
const RRTMGP_BOUNDARY_T = 250.0

# ── Per-process caches ────────────────────────────────────────────────────────
#
# One lock guards all three caches. They are written from tile construction,
# which can run concurrently across threads in a nested/multi-patch setup, and
# read once per radiation call thereafter.

const _RRTMGP_CACHE_LOCK = ReentrantLock()
const _RRTMGP_CONTEXT = Ref{Any}(nothing)
const _RRTMGP_LOOKUPS = Dict{Symbol,Any}()
const _RRTMGP_PROFILES = Dict{Symbol,Any}()
#
# The containers live here, in Scythe, so that the extension module owns no
# global state of its own; ext/ScytheRRTMGPExt.jl fills them.

# ── Standard-atmosphere sampling ──────────────────────────────────────────────

# `_standard_profile(kind)`, the cached 50 m sampling of RRTMGP's analytic standard
# atmosphere, is defined in ext/ScytheRRTMGPExt.jl (it calls
# `RRTMGP.standard_atmosphere`); the stub is declared in src/radiation.jl.

"""
    _interp_linear(xs, ys, x)

Linear interpolation of `ys` at `x` over strictly increasing `xs`, with CONSTANT
extrapolation outside the range. Constant rather than linear extrapolation
because every use here is a sample lookup inside the sampled range (0-70 km);
clamping turns a floating-point excursion at the exact endpoint into an
identity instead of an extrapolated surprise.
"""
@inline function _interp_linear(xs::AbstractVector, ys::AbstractVector, x::Real)
    n = length(xs)
    @inbounds begin
        x <= xs[1] && return ys[1]
        x >= xs[n] && return ys[n]
        i = searchsortedlast(xs, x)
        i >= n && return ys[n]
        w = (x - xs[i]) / (xs[i + 1] - xs[i])
        return (1.0 - w) * ys[i] + w * ys[i + 1]
    end
end

"""
    standard_extension(kind, n_ext, z_top) -> StratosphericExtension

Build the `n_ext` radiative-only layers that sit above the model top `z_top` [m]
and carry the column to `RRTMGP_EXTENSION_TOP` (70 km).

Scythe's top is 25 km, about 25 hPa. RRTMGP integrates exactly the layers it is
handed, so without an extension the outgoing longwave is missing the whole
stratosphere and the shortwave never meets the ozone layer (which peaks near
30 km, ABOVE the model top -- see [`model_ozone`](@ref)).

# Spacing
The faces are uniform in `ln p` between the model top and 70 km, not uniform in
z. Radiative transfer is an integral in optical depth, and optical depth is
proportional to absorber amount, which is proportional to `Δp`; uniform-`ln p`
layers therefore carry comparable optical depth each, whereas uniform-z layers
would put nearly all of the mass in the bottom one or two. `z_face[1] == z_top`
and `z_face[end] == 70 km` are exact; the interior faces come from inverting the
sampled `p(z)`.

# Values
`T`, `p`, `vmr_h2o` and `vmr_o3` are the standard atmosphere's OWN values, at
the layer midpoints and the faces. They are the same for every column of a tile.
The per-column corrections at the seam (pressure rescaling to the column's own
top-face pressure, the temperature offset, the humidity ratio) are applied
inside [`rrtmgp_solve_columns!`](@ref), where the column state is available;
keeping them out of here is what lets one `StratosphericExtension` serve a whole
tile with no per-column storage.

`kind === :none` or `n_ext == 0` returns [`EMPTY_EXTENSION`](@ref).
"""
function standard_extension(kind::Symbol, n_ext::Int, z_top::Float64)
    (kind === :none || n_ext == 0) && return EMPTY_EXTENSION
    n_ext > 0 || error("standard_extension: n_ext must be >= 0, got $n_ext")
    z_top < RRTMGP_EXTENSION_TOP || error(
        "standard_extension: the model top z_top = $z_top m is not below the " *
        "extension top $(RRTMGP_EXTENSION_TOP) m; there is nothing to extend into")
    z_top >= 0.0 || error("standard_extension: z_top = $z_top m must be non-negative")

    prof = _standard_profile(kind)

    lnp_bot = _interp_linear(prof.z_lev, prof.lnp_lev, z_top)
    lnp_top = prof.lnp_lev[end]

    z_face = Vector{Float64}(undef, n_ext + 1)
    z_face[1] = z_top
    z_face[n_ext + 1] = RRTMGP_EXTENSION_TOP
    @inbounds for i in 2:n_ext
        lnp = lnp_bot + ((i - 1) / n_ext) * (lnp_top - lnp_bot)
        z_face[i] = _interp_linear(prof.lnp_rev, prof.z_rev, lnp)
    end
    @inbounds for i in 1:n_ext
        z_face[i] < z_face[i + 1] || error(
            "standard_extension: the log-p face spacing did not come out " *
            "monotone (z_face[$i] = $(z_face[i]) >= z_face[$(i+1)] = " *
            "$(z_face[i+1])); check n_ext = $n_ext and z_top = $z_top")
    end

    z = Vector{Float64}(undef, n_ext)
    @inbounds for k in 1:n_ext
        z[k] = 0.5 * (z_face[k] + z_face[k + 1])
    end
    dz = diff(z_face)

    T = [_interp_linear(prof.z_lev, prof.t_lev, zk) for zk in z]
    T_face = [_interp_linear(prof.z_lev, prof.t_lev, zk) for zk in z_face]
    p = [exp(_interp_linear(prof.z_lev, prof.lnp_lev, zk)) for zk in z]
    p_face = [exp(_interp_linear(prof.z_lev, prof.lnp_lev, zk)) for zk in z_face]
    vmr_h2o = [_interp_linear(prof.z_lay, prof.vmr_h2o, zk) for zk in z]
    vmr_o3 = [_interp_linear(prof.z_lay, prof.vmr_o3, zk) for zk in z]

    return StratosphericExtension(n_ext, z, z_face, dz, T, T_face, p, p_face,
                                  vmr_h2o, vmr_o3)
end

"""
    model_ozone(kind, z) -> Vector{Float64}

Ozone volume mixing ratio on the MODEL layer heights `z` [m], from the same
standard atmosphere the extension uses. Zeros when `kind === :none`.

Scythe carries no ozone: it is a stratospheric absorber with no tropospheric
source and no role in the moist thermodynamics, so it has never been a
prognostic variable. It still matters radiatively inside the model domain --
the 9.6 micron band cools the upper troposphere, and the ultraviolet heating
begins below 25 km -- so a prescribed profile is supplied.

# Interpolation variable
Interpolated LINEARLY IN z, not in `ln p`. The analytic profile is a Gaussian in
`ln p` (peak ~7.5 ppmv near 1200 Pa, about 30 km, over a 30 ppbv tropospheric
background), so `ln p` is the natural variable; but this function's signature
gives it only heights, and using `ln p` would require the column's own pressure
and therefore a per-column ozone profile. At the 50 m sampling of
[`_standard_profile`](@ref) the difference between the two is far below the
uncertainty in the climatology itself (the profile does not know about the
column's actual ozone at all). If a per-column, log-p ozone ever becomes worth
having, it belongs beside the seam corrections in
[`rrtmgp_solve_columns!`](@ref), not here.

Because the peak sits ABOVE a 25 km model top, `model_ozone` on the model layers
is monotonically increasing with height, and the bulk of the ozone column lives
in the extension.
"""
function model_ozone(kind::Symbol, z::AbstractVector)
    kind === :none && return zeros(Float64, length(z))
    prof = _standard_profile(kind)
    return [_interp_linear(prof.z_lay, prof.vmr_o3, zk) for zk in z]
end

# ── Well-mixed gases ──────────────────────────────────────────────────────────

"""
    default_gases(physical_params = Dict{Symbol,Float64}()) -> Dict{String,Float64}

The well-mixed greenhouse-gas volume mixing ratios, keyed by RRTMGP's OWN gas
names (lower case, `"co2"` not `:co2_ppm`), read from `physical_params` with
present-day defaults.

    :co2_ppm   420    -> "co2"    420e-6
    :ch4_ppb   1920   -> "ch4"    1.92e-6
    :n2o_ppb   336    -> "n2o"    336e-9
    :o2_vmr    0.209  -> "o2"     0.209
    :n2_vmr    0.781  -> "n2"     0.781
    :cfc11_ppt 233    -> "cfc11"  233e-12
    :cfc12_ppt 520    -> "cfc12"  520e-12

The unit in the `physical_params` key name is deliberate: mixing ratios are
quoted in ppm/ppb/ppt in every source a user will consult, and a bare `:co2`
would invite a run configured three orders of magnitude off with no way to tell
from the configuration.

Gases the loaded lookup tables do not carry are skipped (with a one-time notice)
by [`rrtmgp_solver`](@ref), not here -- the set of available gases is a property
of the tables, and this function must not need them.
"""
function default_gases(physical_params = Dict{Symbol,Float64}())
    pp = physical_params
    return Dict{String,Float64}(
        "co2" => 1.0e-6 * get(pp, :co2_ppm, 420.0),
        "ch4" => 1.0e-9 * get(pp, :ch4_ppb, 1920.0),
        "n2o" => 1.0e-9 * get(pp, :n2o_ppb, 336.0),
        "o2" => get(pp, :o2_vmr, 0.209),
        "n2" => get(pp, :n2_vmr, 0.781),
        "cfc11" => 1.0e-12 * get(pp, :cfc11_ppt, 233.0),
        "cfc12" => 1.0e-12 * get(pp, :cfc12_ppt, 520.0),
    )
end

# ── Solver construction ───────────────────────────────────────────────────────

# `rrtmgp_solver` itself is in ext/ScytheRRTMGPExt.jl; the boundary-row seed it
# applies names no RRTMGP type and stays here.
#
# The isothermal boundary layer is RRTMGP's internal pad above the domain: one
# extra layer whose values `prepare_atmosphere!` derives from the top domain
# row, and which every getter masks off. Its row is therefore NEVER written by
# the host -- but `validate_inputs` (the `check_values[] = true` path) runs
# BEFORE `prepare_atmosphere!` fills it, and reduces over the FULL arrays. The
# S0 smoke test hit exactly this: RRTMGP's own `solve()` fails its own
# validation on the first call because the pad row is still zero.
#
# So the pad row gets a finite, positive, physically-innocuous seed at
# construction, which `prepare_atmosphere!` then overwrites on every call
# (grid_adaptation.jl: p_lay[end] = (p_lev[end-1] + p_min)/2, t_lay[end] =
# t_lev[end-1], and the same for the vmr and cloud fields). The values below are
# never seen by the radiative transfer.
function _seed_boundary_row!(layerdata, p_lev, t_lev, vmr_h2o, vmr_o3, nlay)
    if layerdata !== nothing
        @views layerdata[2, nlay, :] .= RRTMGP_BOUNDARY_P
        @views layerdata[3, nlay, :] .= RRTMGP_BOUNDARY_T
    end
    @views p_lev[nlay + 1, :] .= RRTMGP_BOUNDARY_P
    @views t_lev[nlay + 1, :] .= RRTMGP_BOUNDARY_T
    vmr_h2o === nothing || (@views vmr_h2o[nlay, :] .= 0.0)
    vmr_o3 === nothing || (@views vmr_o3[nlay, :] .= 0.0)
    return nothing
end

# ── Column writes (called by `rrtmgp_solve_columns!` in the extension) ─────────

# Typed inner loops. `solver` is `Any` at the call site (RadiationState holds it
# untyped so that ModelTile carries no RRTMGP type), so the getter results come
# back dynamically; these barriers make everything after that concrete, which is
# what keeps the write allocation-free.
function _write_thermo!(sol_t_lay, sol_p_lay, sol_t_lev, sol_p_lev,
                        T_lay, p_lay, T_lev, p_lev, ext, nlay_model, ncol)
    n_ext = ext.nlay
    z_top = n_ext > 0 ? ext.z_face[1] : 0.0
    @inbounds for c in 1:ncol
        for k in 1:nlay_model
            sol_t_lay[k, c] = T_lay[k, c]
            sol_p_lay[k, c] = p_lay[k, c]
        end
        for k in 1:(nlay_model + 1)
            sol_t_lev[k, c] = T_lev[k, c]
            sol_p_lev[k, c] = p_lev[k, c]
        end
        n_ext == 0 && continue
        p_seam = p_lev[nlay_model + 1, c]
        p_scale = p_seam / ext.p_face[1]
        dT = T_lev[nlay_model + 1, c] - ext.T_face[1]
        for j in 1:n_ext
            sol_p_lay[nlay_model + j, c] = ext.p[j] * p_scale
            sol_t_lay[nlay_model + j, c] =
                ext.T[j] + dT * exp(-(ext.z[j] - z_top) / RRTMGP_SEAM_DEPTH)
            sol_p_lev[nlay_model + 1 + j, c] = ext.p_face[j + 1] * p_scale
            sol_t_lev[nlay_model + 1 + j, c] =
                ext.T_face[j + 1] +
                dT * exp(-(ext.z_face[j + 1] - z_top) / RRTMGP_SEAM_DEPTH)
        end
    end
    return nothing
end

# `o3` may be shared across columns (a vector) or per column (a matrix); both
# resolve at compile time inside `_write_gases!`, so neither costs a branch.
@inline _o3_at(o3::AbstractVector, k, c) = @inbounds o3[k]
@inline _o3_at(o3::AbstractMatrix, k, c) = @inbounds o3[k, c]

_check_o3_shape(o3::AbstractVector, nlay_model, ncol) =
    length(o3) == nlay_model || error(
        "rrtmgp_solve_columns!: length(o3) = $(length(o3)) should be " *
        "nlay_model = $nlay_model")
_check_o3_shape(o3::AbstractMatrix, nlay_model, ncol) =
    size(o3) == (nlay_model, ncol) || error(
        "rrtmgp_solve_columns!: size(o3) = $(size(o3)) should be " *
        "($nlay_model, $ncol)")

function _write_gases!(sol_h2o, sol_o3, vmr_h2o, o3, p_lev, ext, nlay_model, ncol)
    n_ext = ext.nlay
    @inbounds for c in 1:ncol
        for k in 1:nlay_model
            sol_h2o[k, c] = vmr_h2o[k, c]
            sol_o3[k, c] = _o3_at(o3, k, c)
        end
        n_ext == 0 && continue
        q_ext0 = ext.vmr_h2o[1]
        q_mod = vmr_h2o[nlay_model, c]
        ratio = (q_ext0 > 0.0 && isfinite(q_mod) && q_mod > 0.0) ? q_mod / q_ext0 : 1.0
        for j in 1:n_ext
            sol_h2o[nlay_model + j, c] =
                max(ratio * ext.vmr_h2o[j], RRTMGP_VMR_H2O_FLOOR)
            sol_o3[nlay_model + j, c] = ext.vmr_o3[j]
        end
    end
    return nothing
end

function _write_clouds!(sol_lwp, sol_iwp, sol_re_liq, sol_re_ice, sol_cf,
                        cld, nlay_model, ncol)
    lwp, iwp, re_liq, re_ice, cf = cld
    nlay_tot = size(sol_lwp, 1)
    @inbounds for c in 1:ncol
        for k in 1:nlay_model
            sol_lwp[k, c] = lwp[k, c]
            sol_iwp[k, c] = iwp[k, c]
            sol_re_liq[k, c] = re_liq[k, c]
            sol_re_ice[k, c] = re_ice[k, c]
            sol_cf[k, c] = cf[k, c]
        end
        for k in (nlay_model + 1):nlay_tot
            sol_lwp[k, c] = 0.0
            sol_iwp[k, c] = 0.0
            sol_re_liq[k, c] = 0.0
            sol_re_ice[k, c] = 0.0
            sol_cf[k, c] = 0.0
        end
    end
    return nothing
end

# ── Flux divergence -> gridpoint heating ──────────────────────────────────────

"""
    radiation_divergence!(q, flux_net, dz, nlay_model, ncol, kDim, stride)

Store the radiative heating `q` [W/m^3] on the model gridpoints from the net
flux at the faces:

    q[(c-1)*kDim + k] = (flux_net[k, c] - flux_net[k+1, c]) / dz[k]

for `k` in `1:nlay_model` and `c` in `1:ncol`. The extension layers (rows
`nlay_model+2 : end` of `flux_net`) are discarded: they are radiative
bookkeeping above the model top, not model state.

`q` is GRIDPOINT-indexed with the same row layout as `expdot`
(`j = (c-1)*kDim + k`), so the driver's `QDOT_TH` fold is a straight indexed add
with no reshape.

# Sign
`flux_net = up - down`, so a layer that loses energy to space has
`flux_net[k+1] > flux_net[k]` and `q < 0`. Cooling is negative, as everywhere
else in the radiation code.

# Stride
With `options[:radiation_layer_stride] = stride > 1`, one radiation layer covers
`stride` model gridpoints and `kDim == nlay_model * stride`. The same value is
written to all `stride` gridpoints of layer `k`. That is exactly conservative
rather than approximately so: `dz[k]` is the SUM of the `stride` sub-thicknesses
(it is built from the strided faces), so `q_k * dz_k` is the layer's whole flux
difference however it is distributed within the layer, and the telescoping
identity below holds for any stride.

# Exactness
Summing over the model layers telescopes:

    sum_k q[k] * dz[k] == flux_net[1, c] - flux_net[nlay_model+1, c]

to round-off, for every column. That identity is the reason the store is written
as a flux DIFFERENCE divided by `dz` rather than as a fitted derivative: it makes
the discrete heating conserve energy by construction, not by accuracy.

This function names no RRTMGP type and touches no solver, which is why it stays in
Scythe proper rather than moving into the RRTMGP extension with the solve it consumes.
"""
function radiation_divergence!(q::AbstractVector, flux_net::AbstractMatrix,
                               dz::AbstractVector, nlay_model::Int, ncol::Int,
                               kDim::Int, stride::Int)
    stride >= 1 || error("radiation_divergence!: stride = $stride must be >= 1")
    nlay_model * stride == kDim || error(
        "radiation_divergence!: nlay_model * stride = $(nlay_model * stride) " *
        "should equal kDim = $kDim")
    length(dz) >= nlay_model || error(
        "radiation_divergence!: length(dz) = $(length(dz)) is short of " *
        "nlay_model = $nlay_model")
    size(flux_net, 1) >= nlay_model + 1 || error(
        "radiation_divergence!: flux_net has $(size(flux_net, 1)) levels, " *
        "need at least $(nlay_model + 1)")
    length(q) >= ncol * kDim || error(
        "radiation_divergence!: length(q) = $(length(q)) is short of " *
        "ncol * kDim = $(ncol * kDim)")

    @inbounds for c in 1:ncol
        base = (c - 1) * kDim
        for k in 1:nlay_model
            val = (flux_net[k, c] - flux_net[k + 1, c]) / dz[k]
            off = base + (k - 1) * stride
            for s in 1:stride
                q[off + s] = val
            end
        end
    end
    return nothing
end
