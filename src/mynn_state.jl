# ── MYNN-EDMF state container and option validation ───────────────────────────
#
# This file is the PLUMBING half of the MYNN-EDMF boundary layer: the struct `ModelTile`
# carries, the option table that resolves a configuration, and nothing else. It names no
# closure routine at all -- every call into `mym_turbulence!`, `mym_predict!` and friends
# will live in src/mc_mynn_bl.jl (S5), which is included after mc_boundary_layer.jl. The
# split is the same one radiation_state.jl / radiation.jl makes, for the same two reasons:
#
#   1. `ModelTile` holds its `MYNNState` CONCRETELY (the `EMPTY_RADIATION` argument, and
#      before it `EMPTY_ISHMAEL_TABLES`): a `Union{...,Nothing}` field makes
#      `mtile.mynn.K_m` a type-unstable load in the per-column driver's hot path, so the
#      struct definition has to exist before semiimplicit.jl, which is where `ModelTile`
#      is defined.
#   2. Everything here is testable with no closure call and no boundary-layer physics at
#      all, which is what test/test_mynn_driver.jl exercises at this stage.
#
# STAGE S4 SCOPE. The closure is OFF. What exists here is the container, the validated
# option table and the setup line; the arrays are allocated and zeroed and NOTHING writes
# them. The prognostic TKE-density slot `rho_e` (moist_compressible.jl, `mc_var_names`)
# is registered and TRANSPORTED, with zero sources. A configuration without
# `options[:mynn]` must be BYTE-identical to the code that had no MYNN at all, which is
# why every fold added by this stage is inside an `if mynn_on` and never an `x + 0.0`
# (which is not the identity for `x = -0.0`).
#
# Conventions used throughout the MYNN coupling:
#   * The prognostic slot is the TKE DENSITY `rho_e = rho_t * e` [J/m^3], a TOTAL with
#     `rho_e_bar == 0`, carried so the transport is the same continuity equation every
#     other total in the set takes. The closure's own variable is `qke = 2e` [m^2/s^2];
#     the conversion is S5's business, not this file's.
#   * Gridpoint-indexed arrays are laid out exactly like `expdot` rows
#     (row `j = (c-1)*kDim + k`, layer 1 at the ground), so a driver fold is a straight
#     indexed read with no reshape.
#   * `K_m`, `K_h` are the exchange coefficients [m^2/s] at the LAYER points, which is
#     where `retrieve_exchange_coeffs!` leaves them.

"""
    MYNNColumnScratch

Thread `t`'s per-column working columns for the MYNN apply (src/mc_mynn_bl.jl): the
closure inputs, the per-column staging of the held closure state, the diffusivities, the
fitted fluxes and their divergences.

WHY THESE ARE NOT IN `MC_SCRATCH_SLOTS`. Every other per-column temporary in the
moist-compressible set lives in that `NamedTuple`, and these started there too. Appending
twenty-nine names to it widens the NamedTuple TYPE that `mc_driver!` and everything it
calls specialise on, and `mc_driver!` is large enough that the extra fields pushed LLVM
over a register-pressure cliff: the `@turbo` broadcast in the vertical-diffusion block
stopped compiling in seconds and started spending unbounded time in the MachineScheduler's
register-pressure tracking, wedging the test suite. Nothing about these columns needs to be
in the shared pool -- they are read and written only inside `mc_mynn_bl!` and its
`@noinline` helpers -- so they are a concrete struct of their own, held per thread in
[`MYNNState`](@ref) beside [`MYNNWork`](@ref) and passed down as ONE argument.

The naming drops the `mn_` prefix the slots carried (`S.mn_exner` is `MN.exner`): the
prefix existed to keep them apart from the other ~400 names in the shared pool, and there
is nothing here to collide with.

`exner` and `q`/`qke` are the closure INPUTS built from the retrieved state each step (the
Exner function, `q = sqrt(2e)` from the `rho_e` slot and the `[qkemin, 150]`-clamped copy
the closure integrates against); `el`..`qsq` are the per-column staging of the HELD closure
state, loaded out of `MYNNState` on an update step and written back (a per-thread scratch
column is shared between columns, so it cannot be assumed to still hold this column's
values); `Km`..`Ke` the diffusivities; `Su`..`Sv` the FITTED momentum fluxes, which the
discrete shear production multiplies (the value, not the raw product -- D3); `div_*` the
fitted flux divergences of the species/energy/TKE legs (S8 appends the rain-number leg
`div_nr` and the twelve ice-moment legs `div_i1q`..`div_i3c`); `e`/`e_z` the mass-specific
TKE and its gradient; `wk` one general work column.

Every field is a `Vector{Float64}` of length `kDim`, so the struct is concrete and a field
load inside a `@noinline` helper is a plain pointer load.
"""
struct MYNNColumnScratch
    n::Int
    # -- closure inputs --------------------------------------------------------
    exner::Vector{Float64}
    q::Vector{Float64}
    qke::Vector{Float64}
    # -- per-column staging of the held closure state --------------------------
    el::Vector{Float64}
    sm::Vector{Float64}
    sh::Vector{Float64}
    vt::Vector{Float64}
    vq::Vector{Float64}
    sgm::Vector{Float64}
    cfb::Vector{Float64}
    qcb::Vector{Float64}
    qib::Vector{Float64}
    qsq::Vector{Float64}
    # -- diffusivities ---------------------------------------------------------
    Km::Vector{Float64}
    Kh::Vector{Float64}
    Ke::Vector{Float64}
    # -- fitted momentum fluxes ------------------------------------------------
    Su::Vector{Float64}
    Sw::Vector{Float64}
    Sv::Vector{Float64}
    # -- fitted flux divergences -----------------------------------------------
    div_w::Vector{Float64}
    div_v::Vector{Float64}
    div_c::Vector{Float64}
    div_r::Vector{Float64}
    div_e::Vector{Float64}
    div_Ew::Vector{Float64}
    div_Pm::Vector{Float64}
    # -- the two-moment rain NUMBER leg (S8; options[:rain_moments] = 2) --------
    div_nr::Vector{Float64}
    # -- the twelve ISHMAEL ICE moment legs (S8) --------------------------------
    # `<species><moment>` exactly as `MC_SCRATCH_SLOTS` names them: `i1q` the mass of
    # species 1, `i1n` its number, `i1a`/`i1c` its two spheroid volume moments. All
    # thirteen are unread and untouched with ice / two-moment rain off.
    div_i1q::Vector{Float64}
    div_i1n::Vector{Float64}
    div_i1a::Vector{Float64}
    div_i1c::Vector{Float64}
    div_i2q::Vector{Float64}
    div_i2n::Vector{Float64}
    div_i2a::Vector{Float64}
    div_i2c::Vector{Float64}
    div_i3q::Vector{Float64}
    div_i3n::Vector{Float64}
    div_i3a::Vector{Float64}
    div_i3c::Vector{Float64}
    # -- TKE and one general work column ---------------------------------------
    e::Vector{Float64}
    e_z::Vector{Float64}
    wk::Vector{Float64}
end

"""
    MYNNColumnScratch(n)

`n` zeroed columns of length `n`, one field at a time, the `MYNNWork(n)` pattern.
"""
function MYNNColumnScratch(n::Integer)
    n = Int(n)
    args = Any[n]
    for _ in 2:fieldcount(MYNNColumnScratch)
        push!(args, zeros(Float64, n))
    end
    return MYNNColumnScratch(args...)
end

# ── Fidelity: the named deviations from the verbatim Fortran ──────────────────

"""
    MYNN_DEVIATIONS

The seven NAMED deviations from the verbatim-Fortran MYNN closure and coupling. Each is
one branch in one place, each is off under `:mynn_fidelity = :fortran` (the default), and
each exists to be MEASURED against that default rather than argued about:

  * `:gtr_local` -- the buoyancy parameter `g/theta_v` per LEVEL instead of the Fortran's
    single `g/MYNN_TREF` with `MYNN_TREF = 300 K`. Used by `G_H` (`mym_level2!`), the
    mixing length's `vsc`/`bv`/`wstar` (`mym_length!`), the plume surface scaling and
    Brunt-Vaisala frequency (`dmp_mf!`) and the surface `rmol`. In a tropical boundary
    layer `theta_v` runs ~305 K, so the Fortran's `gtr` is ~2 % high everywhere and more
    than that in the warm core of a TC.
  * `:K_interface` -- `K_m`/`K_h` formed by AVERAGING the closure's wall values of
    `el*S` onto the mish point, the way `mym_predict!`'s `0.5*(el(k+1)+el(k))` and the
    Fortran's `dfm(k) = elq*sm(k)/dzk` treat them, instead of the colocated
    `K = el(k) q(k) S(k)` of D1.
  * `:sqfac1` -- `K_e = 1 * K_m` instead of `MYNN_SQFAC * K_m = 3 K_m`. The Fortran's
    `Sqfac = 3` is a TKE-transport enhancement inside its implicit tridiagonal solve;
    Scythe integrates `rho_e` explicitly, where the same factor triples the diffusion
    number the timestep has to survive (`D_gal`).
  * `:pdk1` -- the surface TKE source is the Fortran's log-layer production
    `pdk1 = 2 u*^3 pmz / (karman 0.5 dz(1))` instead of the drag work
    `(tau_u u + tau_v v) g(z)` the coupling delivers today. See `_mynn_apply_column!`.
  * `:exner_single` -- `th_sfc = SST/exner(1)`, dividing by the surface Exner function
    ONCE. The Fortran divides TWICE (`mynn_bl_driver` :1063 with a `ts` that is already
    `T_sfc/exner(1)`; harness README item 2), which inflates the surface potential
    temperature by ~3 % and through `fltv` inflates `rmol` and the mixing length.
  * `:rmol_sfc` -- `1/L` taken from `surface_exchange`'s Monin-Obukhov solve
    (`sx.inv_L`) instead of recomputed inside the closure from `fltv` and `u*`. Needs
    `options[:sfc_stability] = true`; without it `inv_L` is identically zero and the
    switch would silently neutralize the surface-layer stability functions.
  * `:flux_clip` -- the `mynnedmf_wrapper` clips of harness README item 7,
    `hfx in [-500, 1200] W/m^2` and `qfx in [-2e-4, 5e-4] kg/(m^2 s)`, applied to what
    the CLOSURE sees. The model's own surface delivery stays UNCLIPPED, so the energy
    budget against the surface fluxes keeps its meaning. The COUNTERS
    (`n_hfx_clip`/`n_qfx_clip`) are accumulated whether or not the switch is on, so a run
    always reports how often the Fortran would have clipped.

There is deliberately no `:scythe` BUNDLE at this stage: a bundle is a claim that a set
of deviations is the better model, and no deviation has been measured yet. One gets
added, with its members named, when the evidence is in.
"""
const MYNN_DEVIATIONS = (:gtr_local, :K_interface, :sqfac1, :pdk1, :exner_single,
                         :rmol_sfc, :flux_clip)

"""
    MYNNFidelity

The RESOLVED form of `options[:mynn_fidelity]`: one field per deviation of
[`MYNN_DEVIATIONS`](@ref), so every use site is a `fid.<name>` load and never a `Symbol`
comparison or a `in` on a vector in a per-column loop. `names` keeps the requested list
for the setup line, the NetCDF attribute and the tests.

`sqfac` is a Float64 rather than a Bool because it is a FACTOR: `:sqfac1` sets it to 1.0
and its absence leaves it at [`MYNN_SQFAC`](@ref) `= 3.0`, so `K_e = fid.sqfac*K_m` is
the same IEEE product as today's `MYNN_SQFAC*K_m` on the default path.
"""
struct MYNNFidelity
    gtr_local::Bool
    K_interface::Bool
    sqfac::Float64
    pdk1::Bool
    exner_single::Bool
    rmol_sfc::Bool
    flux_clip::Bool
    names::Vector{Symbol}
end

"The no-deviation fidelity: verbatim Fortran, and the default of every configuration."
const MYNN_FORTRAN_FIDELITY = MYNNFidelity(false, false, MYNN_SQFAC, false, false, false,
                                           false, Symbol[])

"""
    MYNNFidelity(names::AbstractVector{Symbol}) -> MYNNFidelity

Resolve a list of deviation names. An empty list IS `:fortran`. An unrecognized name or a
repeat raises with the valid list, in the `_mynn_check` style: a misspelled deviation that
silently did nothing would be a measurement of the control reported as a measurement of
the treatment.
"""
function MYNNFidelity(names::AbstractVector{Symbol})
    seen = Symbol[]
    for nm in names
        nm in MYNN_DEVIATIONS || error(
            "options[:mynn_fidelity] does not have a deviation :$(nm); the deviations " *
            "are " * join(string.(":", MYNN_DEVIATIONS), ", ", " and ") *
            ", and :fortran (or an empty vector) is no deviation at all")
        nm in seen && error(
            "options[:mynn_fidelity] names :$(nm) twice; each deviation is one switch")
        push!(seen, nm)
    end
    return MYNNFidelity(:gtr_local in seen, :K_interface in seen,
                        (:sqfac1 in seen) ? 1.0 : MYNN_SQFAC,
                        :pdk1 in seen, :exner_single in seen, :rmol_sfc in seen,
                        :flux_clip in seen, seen)
end

Base.:(==)(a::MYNNFidelity, b::MYNNFidelity) = a.names == b.names

"""
    mynn_fidelity_string(f::MYNNFidelity) -> String

`"fortran"` for the no-deviation default, otherwise the comma-joined deviation names --
the form the setup line prints, the NetCDF `fidelity` attribute carries and
[`parse_mynn_fidelity`](@ref) reads back.
"""
mynn_fidelity_string(f::MYNNFidelity) =
    isempty(f.names) ? "fortran" : join(string.(f.names), ",")

"""
    parse_mynn_fidelity(s::AbstractString) -> Union{Symbol,Vector{Symbol}}

`options[:mynn_fidelity]` from a STRING, for the environment knobs
(`SCYTHE_OWB_MYNN_FIDELITY`, `SCYTHE_TC_MYNN_FIDELITY`) and for reading a run's own
`fidelity` attribute back. `"fortran"` (and the empty string) give `:fortran`; anything
else is split on commas into a `Vector{Symbol}`, which
[`validate_mynn_options`](@ref) then checks -- a typo dies at setup, not silently.
"""
function parse_mynn_fidelity(s::AbstractString)
    t = strip(s)
    (isempty(t) || t == "fortran") && return :fortran
    return Symbol[Symbol(strip(p)) for p in split(t, ',') if !isempty(strip(p))]
end
"""
    MYNNState

Everything one tile needs to run and hold the MYNN-EDMF boundary layer.

The struct is `mutable` but nearly every field is `const`: the arrays are written in place
and only the three clamp/cap counters change identity between calls. The `const`
annotations are not decoration -- they let the compiler hoist the array loads in the
per-column fold.

# Resolved configuration
`closure`, `edmf`, `scale_aware`, `init_mode`, `fidelity`, `water_carry`,
`interval_steps`, `K_max`, `output`, `trace`, `check_values` are what
[`validate_mynn_options`](@ref) returned; `active` is `options[:mynn] == true`.

# Held column state (GRIDPOINT-indexed, `expdot` layout)
`el` (mixing length [m]), `sm`/`sh` (the stability functions), `vt`/`vq` (the buoyancy
coefficients of `mym_condensation!`), `sgm` (the cloud-PDF width), `cldfra_bl`/`qc_bl`/
`qi_bl` (the subgrid cloud), and `K_m`/`K_h` [m^2/s]. These are the fields
`MYNNColumnState` carries per column in the pure port (src/mynn_closure.jl), flattened
onto the tile. `qke` is NOT among them: the TKE is the PROGNOSTIC slot `rho_e`, which is
the whole point of the coupling -- a second copy carried here could disagree with it.

# Plume sums (S7), GRIDPOINT-indexed and held on the cadence
`s_aw` is `Sigma_aw = sum_i a_i w_i` [m/s] and `s_aw_st`, `s_aw_qw`, `s_aw_qv`, `s_aw_u`,
`s_aw_v`, `s_aw_e` the plume-weighted sums `sum_i a_i w_i phi_i` of the plume's entropy
[J/(kg K)], its total and vapour SPECIFIC water [kg/kg], its `u`, `v` [m/s] and its
mass-specific TKE `e = qke/2` [J/kg]. They are Scythe's variables, not the Fortran's:
`DMP_mf` returns `s_aw*` on the INTERFACES and multiplied by the interface density, and
`_mynn_plume_sums!` (src/mc_mynn_bl.jl) maps them onto the mish and divides the density
back out, so the every-step assembly forms `M_phi = rho_t Sigma_aw (phi_up - bar phi)`
with the CURRENT environment. All seven are exact zeros with `:mynn_edmf = 0` and on any
column whose plumes did not fire, which is what makes the off path bitwise.

# Column geometry (built once; the mish is the same for every column of a tile)
`z_lay` (mish heights), `dz` (MYNN layer thickness from midpoint faces), `zw` (wall
heights, `kDim+1` long), `w_mish` (the Gauss quadrature weight of each mish point, which
is what the D3 column identity is summed with), `dx` (the patch's own horizontal cell
width, for `SCALE_AWARE`), `z_top`, `dz_cell`, `dz_min` and this patch's `ts`. Nothing
here is a hardcoded number: the census limits and the gray-zone taper are computed for
the grid and timestep the run actually has (D12).

# Per-column diagnostics and census
`pblh` [m], `kpbl` (the layer index of the PBL top), `ust` [m/s], `rmol` (1/L),
`last_update_step`, the D3 energy identity's right-hand side `bdry_E` [W/m^2], the two
explicit-diffusion numbers `D_gal`/`D_mish`, the TKE stiffness `ts_tau`, the column
maxima `K_m_max`/`K_h_max`, the two shear productions `Ps_disc`/`Ps_mynn` [W/m^2] and the
three per-column counters `n_clamp_col`/`n_capK_col`/`n_diffnum_col`. All of length
`ncol`. The counters are per COLUMN rather than per tile so the reduction is exact under
`@threads :static`: each column is written by the one thread that owns it, so nothing
races and no increment is lost; `mynn_write_final!` sums them into the tile scalars. `last_update_step` starts at `typemin(Int)` for
the same reason `RadiationState.last_call_step` does: "no call has happened yet" has to be
distinguishable from "called at step 0", so the first step forces an update whatever the
cadence is (and it is what the `:mynn_init = :taper` cold start keys off).

# Work and constants
`work[t]` is thread `t`'s [`MYNNWork`](@ref) -- the preallocated scratch that keeps the
closure routines allocation-free -- and `colscratch[t]` its [`MYNNColumnScratch`](@ref),
the apply half's own per-column columns; both are indexed by `threadid()` under the same
`@threads :static` ownership rule `scratch_columns` uses. `constants` is the host constant
set, built once from Springsteel.

# Counters
`n_clamp_e`, `n_cap_K`, `n_diffnum` accumulate how often the TKE had to be floored, an
exchange coefficient hit `physical_params[:mynn_K_max]`, and the vertical diffusion number
exceeded its stability bound. `n_gate`, `n_stall`, `n_plume` are the S7 plume counts
(gate passed / passed but every plume stalled at the first interface / a mass flux was
actually produced), in COLUMN-UPDATES. All six are NOT `const`: they are the reason a
clamp, and a plume that never fires, is never silent. `n_hfx_clip`, `n_qfx_clip` count
how often the Fortran wrapper's surface-flux clips (`:flux_clip`) would have bitten,
whether or not that deviation is on.
"""
mutable struct MYNNState
    # ── resolved configuration (see validate_mynn_options) ──
    const active::Bool
    const closure::Float64          # 2.5 (2.6 arrives later)
    const edmf::Int                 # 0 (eddy diffusivity only) | 1 (mass-flux plumes)
    const edmf_mom::Bool            # do the plumes carry momentum? (:mynn_edmf_mom)
    const scale_aware::Bool
    const init_mode::Symbol         # :taper | :zero
    const fidelity::MYNNFidelity    # the resolved deviation switches (MYNN_DEVIATIONS)
    const water_carry::Symbol       # :flux | :fixed_T
    const mix_numbers::Bool         # mix the two-moment rain NUMBER? (:mynn_mix_numbers)
    const interval_steps::Int       # BL cadence, in model steps
    const ncol::Int
    const kDim::Int
    const K_max::Float64
    const output::Bool
    const trace::Bool
    const check_values::Bool
    # ── held column state, GRIDPOINT-indexed ──
    const el::Vector{Float64}
    const sm::Vector{Float64}
    const sh::Vector{Float64}
    const vt::Vector{Float64}
    const vq::Vector{Float64}
    const sgm::Vector{Float64}
    const cldfra_bl::Vector{Float64}
    const qc_bl::Vector{Float64}
    const qi_bl::Vector{Float64}
    const K_m::Vector{Float64}
    const K_h::Vector{Float64}
    # `gh` is `mym_level2!`'s buoyancy-gradient function G_H, held so the EVERY-STEP
    # buoyancy exchange rho*P_b = rho_t K_h G_H can be formed between closure updates;
    # `qsq` is the Level-2 water-variance diagnostic the NEXT `mym_condensation!` reads
    # (it is the only one of tsq/qsq/cov that CASE 2 actually uses -- see that routine's
    # docstring). Both are gridpoint-indexed like the fields above.
    const gh::Vector{Float64}
    const gm::Vector{Float64}
    const qsq::Vector{Float64}
    # ── PER-GRIDPOINT budget columns (S9), held so the sidecar (src/mynn_io.jl) can write
    # them without recomputing the apply loop. Filled in `_mynn_apply_column!`
    # (src/mc_mynn_bl.jl) from exactly the values that already feed the D3 identity:
    # `g_Ps` the discrete shear production, `g_Ps_mynn` the closure's own
    # `rho_t K_m G_M`, `g_Pb` the buoyancy exchange `rho_t K_h G_H`, `g_eps` the
    # dissipation `rho_t q^3/(B_1 l)`, `g_tke_transport` the fitted TKE turbulent-transport
    # divergence `dz(S_e)`. All [W/m^3] except `g_tke_transport` (also W/m^3, a divergence
    # of an energy flux). Exact zeros with `:mynn` off (empty) and, on an active tile,
    # exactly the per-point terms `_mynn_apply_column!` already computed -- nothing here
    # changes what any tendency receives.
    const g_Ps::Vector{Float64}
    const g_Ps_mynn::Vector{Float64}
    const g_Pb::Vector{Float64}
    const g_eps::Vector{Float64}
    const g_tke_transport::Vector{Float64}
    # ── mass-flux plume sums (S7); zero at this stage ──
    const s_aw::Vector{Float64}
    const s_aw_st::Vector{Float64}
    const s_aw_qw::Vector{Float64}
    const s_aw_qv::Vector{Float64}
    const s_aw_u::Vector{Float64}
    const s_aw_v::Vector{Float64}
    const s_aw_e::Vector{Float64}
    # ── per-column plume census (S7); zero with :mynn_edmf = 0 ──
    # `plume_ktop`/`plume_ztop`/`aw_max` are RUNNING MAXIMA over the run of that column's
    # plume state (the highest interface a plume reached, its height and the largest
    # `Sigma_aw`), advanced only on an update that actually produced a flux; the three
    # counters accumulate COLUMN-UPDATES, which is what makes the
    # gate/stall distinction of tools/mynn_fortran_driver/README.md item 12 readable:
    # `n_gate` passed `fltv2 > 0.002 && maxwidth > minwidth && superadiabatic`, `n_stall`
    # passed it and still made no flux because every plume failed to leave the first
    # interface (`nup2 = 0` at :6288), `n_plume` actually produced a mass flux.
    const plume_ktop::Vector{Int}
    const plume_ztop::Vector{Float64}
    const aw_max::Vector{Float64}
    const n_gate_col::Vector{Int}
    const n_stall_col::Vector{Int}
    const n_plume_col::Vector{Int}
    # ── per-column ──
    const pblh::Vector{Float64}
    const kpbl::Vector{Int}
    const ust::Vector{Float64}
    const rmol::Vector{Float64}
    # `pmz = phim(zeta) - zeta`, the surface-layer momentum stability function, per
    # column. Written whenever `check_values` or the `:pdk1` deviation asks for it and
    # read ONLY by `:pdk1` (`_mynn_apply_column!` builds its log-layer surface
    # production from it); under `:fortran` it is written by nothing and read by nothing.
    const pmz::Vector{Float64}
    const last_update_step::Vector{Int}
    # ── column geometry, built ONCE from the tile's mish (identical for every column) ──
    const z_lay::Vector{Float64}     # mish heights [m]
    const dz::Vector{Float64}        # MYNN layer thickness from midpoint faces [m]
    const zw::Vector{Float64}        # wall heights, length kDim+1 [m]
    const w_mish::Vector{Float64}    # Gauss quadrature weight of each mish point [m]
    const dx::Float64                # horizontal cell width for SCALE_AWARE [m]
    const z_top::Float64             # domain lid [m]
    const dz_cell::Float64           # vertical B-spline cell width [m]
    const dz_min::Float64            # smallest actual mish spacing [m]
    const ts::Float64                # this patch's model timestep [s]
    # ── PER-COLUMN census (D12) and the column energy identity (D3) ──
    # Written by the ONE thread that owns the column, so no atomics and no lost counts;
    # the tile totals below are reductions over these, taken when they are printed.
    const bdry_E::Vector{Float64}    # boundary + surface energy input [W/m^2]
    const D_gal::Vector{Float64}     # K_e ts 10 / dz_cell^2
    const D_mish::Vector{Float64}    # K_e ts / dz_min^2
    const ts_tau::Vector{Float64}    # ts / tau_eps, tau_eps = B1 l /(2q)
    const K_m_max::Vector{Float64}
    const K_h_max::Vector{Float64}
    # The two shear productions, column-integrated [W/m^2]: `Ps_disc` is what the TKE
    # slot actually received (`u_z S_u + w_z S_w + v_z S_v` on the FITTED fluxes, the
    # form the D3 identity needs) and `Ps_mynn` the closure's own `rho_t K_m gm`. Their
    # ratio is the "design A leaves <pdk - rho P_s> unclosed" residual, measured rather
    # than argued.
    const Ps_disc::Vector{Float64}
    const Ps_mynn::Vector{Float64}
    const n_clamp_col::Vector{Int}
    const n_capK_col::Vector{Int}
    const n_diffnum_col::Vector{Int}
    # How often the Fortran wrapper's flux clips (harness README item 7) would have
    # bitten, in COLUMN-UPDATES. Counted on EVERY run, `:flux_clip` or not, so a run
    # always reports whether the deviation would have changed anything; the clip itself
    # happens only under the switch.
    const n_hfx_clip_col::Vector{Int}
    const n_qfx_clip_col::Vector{Int}
    # ── per-thread scratch and the host constants ──
    const work::Vector{MYNNWork}
    # Thread `t`'s per-column apply scratch. Held HERE rather than appended to
    # `MC_SCRATCH_SLOTS` — see [`MYNNColumnScratch`](@ref) for the register-pressure
    # reason. Empty on the MYNN-off value, like `work`.
    const colscratch::Vector{MYNNColumnScratch}
    # Thread `t`'s `DMP_mf` scratch. EMPTY unless `:mynn_edmf = 1`: one `EDMFWork` is nine
    # (kDim+1, 8) plume matrices plus ~25 columns, and a run without the plumes must not
    # pay for it. Indexed by `threadid()` under the same ownership rule as `work`.
    const ework::Vector{EDMFWork}
    const constants::MYNNConstants
    # ── clamp/cap counters (never silent) ──
    n_clamp_e::Int
    n_cap_K::Int
    n_diffnum::Int
    n_gate::Int
    n_stall::Int
    n_plume::Int
    n_hfx_clip::Int
    n_qfx_clip::Int
end

"""
    MYNNState(; kwargs...)

Keyword constructor with the INACTIVE value as its default for every field, so
[`mc_mynn_state`](@ref) names only what a live configuration actually changes. The
positional list above is long enough that a positional call is a bug waiting to happen.
"""
function MYNNState(;
        active::Bool = false,
        closure::Float64 = 2.5,
        edmf::Int = 0,
        edmf_mom::Bool = true,
        scale_aware::Bool = true,
        init_mode::Symbol = :taper,
        fidelity::MYNNFidelity = MYNN_FORTRAN_FIDELITY,
        water_carry::Symbol = :flux,
        mix_numbers::Bool = true,
        interval_steps::Int = 1,
        ncol::Int = 0,
        kDim::Int = 0,
        K_max::Float64 = Inf,
        output::Bool = false,
        trace::Bool = false,
        check_values::Bool = false,
        el::Vector{Float64} = Float64[],
        sm::Vector{Float64} = Float64[],
        sh::Vector{Float64} = Float64[],
        vt::Vector{Float64} = Float64[],
        vq::Vector{Float64} = Float64[],
        sgm::Vector{Float64} = Float64[],
        cldfra_bl::Vector{Float64} = Float64[],
        qc_bl::Vector{Float64} = Float64[],
        qi_bl::Vector{Float64} = Float64[],
        K_m::Vector{Float64} = Float64[],
        K_h::Vector{Float64} = Float64[],
        gh::Vector{Float64} = Float64[],
        gm::Vector{Float64} = Float64[],
        qsq::Vector{Float64} = Float64[],
        g_Ps::Vector{Float64} = Float64[],
        g_Ps_mynn::Vector{Float64} = Float64[],
        g_Pb::Vector{Float64} = Float64[],
        g_eps::Vector{Float64} = Float64[],
        g_tke_transport::Vector{Float64} = Float64[],
        s_aw::Vector{Float64} = Float64[],
        s_aw_st::Vector{Float64} = Float64[],
        s_aw_qw::Vector{Float64} = Float64[],
        s_aw_qv::Vector{Float64} = Float64[],
        s_aw_u::Vector{Float64} = Float64[],
        s_aw_v::Vector{Float64} = Float64[],
        s_aw_e::Vector{Float64} = Float64[],
        plume_ktop::Vector{Int} = Int[],
        plume_ztop::Vector{Float64} = Float64[],
        aw_max::Vector{Float64} = Float64[],
        n_gate_col::Vector{Int} = Int[],
        n_stall_col::Vector{Int} = Int[],
        n_plume_col::Vector{Int} = Int[],
        pblh::Vector{Float64} = Float64[],
        kpbl::Vector{Int} = Int[],
        ust::Vector{Float64} = Float64[],
        rmol::Vector{Float64} = Float64[],
        pmz::Vector{Float64} = Float64[],
        last_update_step::Vector{Int} = Int[],
        z_lay::Vector{Float64} = Float64[],
        dz::Vector{Float64} = Float64[],
        zw::Vector{Float64} = Float64[],
        w_mish::Vector{Float64} = Float64[],
        dx::Float64 = 0.0,
        z_top::Float64 = 0.0,
        dz_cell::Float64 = 0.0,
        dz_min::Float64 = 0.0,
        ts::Float64 = 0.0,
        bdry_E::Vector{Float64} = Float64[],
        D_gal::Vector{Float64} = Float64[],
        D_mish::Vector{Float64} = Float64[],
        ts_tau::Vector{Float64} = Float64[],
        K_m_max::Vector{Float64} = Float64[],
        Ps_disc::Vector{Float64} = Float64[],
        Ps_mynn::Vector{Float64} = Float64[],
        K_h_max::Vector{Float64} = Float64[],
        n_clamp_col::Vector{Int} = Int[],
        n_capK_col::Vector{Int} = Int[],
        n_diffnum_col::Vector{Int} = Int[],
        n_hfx_clip_col::Vector{Int} = Int[],
        n_qfx_clip_col::Vector{Int} = Int[],
        work::Vector{MYNNWork} = MYNNWork[],
        colscratch::Vector{MYNNColumnScratch} = MYNNColumnScratch[],
        ework::Vector{EDMFWork} = EDMFWork[],
        constants::MYNNConstants = MYNNConstants(),
        n_clamp_e::Int = 0,
        n_cap_K::Int = 0,
        n_diffnum::Int = 0,
        n_gate::Int = 0,
        n_stall::Int = 0,
        n_plume::Int = 0,
        n_hfx_clip::Int = 0,
        n_qfx_clip::Int = 0)
    return MYNNState(active, closure, edmf, edmf_mom, scale_aware, init_mode, fidelity,
                     water_carry, mix_numbers,
                     interval_steps, ncol, kDim, K_max, output, trace, check_values,
                     el, sm, sh, vt, vq, sgm, cldfra_bl, qc_bl, qi_bl, K_m, K_h,
                     gh, gm, qsq, g_Ps, g_Ps_mynn, g_Pb, g_eps, g_tke_transport,
                     s_aw, s_aw_st, s_aw_qw, s_aw_qv, s_aw_u, s_aw_v, s_aw_e,
                     plume_ktop, plume_ztop, aw_max, n_gate_col, n_stall_col, n_plume_col,
                     pblh, kpbl, ust, rmol, pmz, last_update_step,
                     z_lay, dz, zw, w_mish, dx, z_top, dz_cell, dz_min, ts,
                     bdry_E, D_gal, D_mish, ts_tau, K_m_max, K_h_max, Ps_disc, Ps_mynn,
                     n_clamp_col, n_capK_col, n_diffnum_col,
                     n_hfx_clip_col, n_qfx_clip_col,
                     work, colscratch, ework, constants,
                     n_clamp_e, n_cap_K, n_diffnum, n_gate, n_stall, n_plume,
                     n_hfx_clip, n_qfx_clip)
end

"""
The MYNN-OFF value: inactive, every array empty, no per-thread work.

`ModelTile` carries its `MYNNState` concretely for the same reason it carries
`RadiationState` concretely (see [`EMPTY_RADIATION`](@ref)) -- a `Union` field would make
`mtile.mynn.active` a type-unstable load in `mc_driver!`'s preamble. A run with the
closure off never reads past `.active`, so the empty arrays are unreachable rather than
merely harmless.

Shared `const` across tiles, like `EMPTY_RADIATION` and for the same reason: it is a
`mutable struct`, so sharing is safe only because nothing ever writes to the off value
(the driver gate is `.active` / the slot index), and it keeps tile construction
allocation-free on the common path.
"""
const EMPTY_MYNN = MYNNState()

# ── Option validation ─────────────────────────────────────────────────────────

# Every options key this module reads. Anything else starting with "mynn" is a typo and is
# refused: a silently ignored `:mynn_intervals` would look like a working run at the
# default cadence, which is exactly the kind of failure that costs a campaign (cf. the
# Louis-BL and Khdiff_water blockers, and `RADIATION_OPTION_KEYS`).
const MYNN_OPTION_KEYS = Set{Symbol}((
    :mynn, :mynn_interval, :mynn_edmf, :mynn_edmf_mom, :mynn_closure, :mynn_scale_aware,
    :mynn_init, :mynn_fidelity, :mynn_water_carry, :mynn_mix_numbers, :mynn_output,
    :mynn_trace, :mynn_check_values))

const MYNN_INIT_MODES = (:taper, :zero)
# The Symbol values `:mynn_fidelity` accepts. `:fortran` is the verbatim closure and the
# only single-Symbol setting there is; every DEVIATION is named individually, as a
# `Vector{Symbol}` drawn from `MYNN_DEVIATIONS` (an empty vector is `:fortran`). A bundle
# Symbol -- `:scythe`, say -- would be a claim that a particular set of deviations is the
# better model, and it gets added here only when the evidence says so.
const MYNN_FIDELITIES = (:fortran,)
const MYNN_WATER_CARRIES = (:flux, :fixed_T)

_mynn_check(value, allowed, key) = value in allowed || error(
    "options[:$key] = :$(value) is not recognized; use " *
    join(string.(":", allowed), ", ", " or "))

"""
    _mynn_resolve_fidelity(value, options) -> MYNNFidelity

`options[:mynn_fidelity]` -> the resolved switch set. Accepts `:fortran` (the default) and
a `Vector{Symbol}` of [`MYNN_DEVIATIONS`](@ref) names; an empty vector is `:fortran`.
Anything else -- another Symbol, a vector with a non-Symbol element, a string -- raises
with the valid names, as does an unknown or repeated deviation
([`MYNNFidelity`](@ref)).

`:rmol_sfc` additionally REQUIRES `options[:sfc_stability] = true`: without the
Monin-Obukhov solve `surface_exchange` returns `inv_L = 0.0` identically
(src/mc_surface_layer.jl), so the deviation would not replace the closure's `1/L` with a
better one -- it would replace it with a neutral surface layer, and report that as a
measurement of the deviation.
"""
function _mynn_resolve_fidelity(value, options)
    fid = if value isa Symbol
        _mynn_check(value, MYNN_FIDELITIES, "mynn_fidelity")
        MYNN_FORTRAN_FIDELITY
    elseif value isa AbstractVector
        all(x -> x isa Symbol, value) || error(
            "options[:mynn_fidelity] as a list must hold Symbols (got " *
            "$(repr(value))); the deviations are " *
            join(string.(":", MYNN_DEVIATIONS), ", ", " and "))
        MYNNFidelity(collect(Symbol, value))
    else
        error("options[:mynn_fidelity] must be :fortran or a Vector{Symbol} of " *
              join(string.(":", MYNN_DEVIATIONS), ", ", " and ") *
              " (got $(repr(value)))")
    end
    (!fid.rmol_sfc || get(options, :sfc_stability, false) === true) || error(
        "options[:mynn_fidelity] names :rmol_sfc, which takes 1/L from " *
        "surface_exchange's Monin-Obukhov solve -- but options[:sfc_stability] is not " *
        "true, so that inv_L is identically 0.0 and the deviation would measure a " *
        "neutral surface layer instead of a better one. Set " *
        "options[:sfc_stability] = true, or drop :rmol_sfc")
    return fid
end

"""
    validate_mynn_options(options, physical_params, equation_set, ts, kDim)

Check the MYNN-EDMF configuration LOUDLY and return the resolved settings as a NamedTuple
`(active, closure, edmf, edmf_mom, scale_aware, init_mode, fidelity, water_carry,
mix_numbers, interval_steps, K_max, output, trace, check_values)`.

Called once per tile from [`mc_mynn_state`](@ref), before any array is allocated, so a
misconfigured run dies at setup rather than several minutes in -- or, worse, runs to
completion with the wrong closure.

`options[:mynn]` absent or `false` short-circuits after the type check: a run with the
closure off must never be able to fail on a MYNN rule, so a configuration carrying a stale
`:mynn_interval` still starts.

Errors:
- an unknown `:mynn_*` key, or a non-`Bool` `:mynn`;
- `options[:mynn]` on an equation set that is not a pressure-reference
  (`moist_compressible_*`) set -- nothing else carries the `rho_e` slot;
- `:mynn_closure` other than 2.5 (2.6 is a later stage);
- `:mynn_edmf` other than 0/1, or `= 1` on a column shorter than four layers;
- an unrecognized `:mynn_init` or `:mynn_water_carry`, or a non-`Bool`
  `:mynn_mix_numbers`;
- a `:mynn_fidelity` that is neither `:fortran` nor a `Vector{Symbol}` of
  [`MYNN_DEVIATIONS`](@ref) names, one that repeats a deviation, or one that names
  `:rmol_sfc` without `options[:sfc_stability] = true`
  ([`_mynn_resolve_fidelity`](@ref));
- a non-positive `:mynn_interval`, or one SHORTER than the model timestep (the cadence is
  in seconds so it is nest-invariant; a cadence below one step is not a cadence);
- a non-positive `physical_params[:mynn_K_max]`.
"""
function validate_mynn_options(options, physical_params, equation_set, ts, kDim)
    on = get(options, :mynn, false)
    isa(on, Bool) || error(
        "options[:mynn] must be a Bool (got $(repr(on))); the closure variant is chosen " *
        "with options[:mynn_closure], not here")

    # Resolve the whole table first so the off path returns the same NamedTuple shape.
    closure = Float64(get(options, :mynn_closure, 2.5))
    edmf = Int(get(options, :mynn_edmf, 0))
    edmf_mom = get(options, :mynn_edmf_mom, true)::Bool
    scale_aware = get(options, :mynn_scale_aware, true)::Bool
    init_mode = get(options, :mynn_init, :taper)
    water_carry = get(options, :mynn_water_carry, :flux)
    # Mix the two-moment rain NUMBER alongside the rain MASS (S8). Default ON, because
    # mixing a mass without its number is a statement about the drop size: `K_h ∂z ρ_r`
    # with `n_r` left behind rescales the mean diameter of every column the operator
    # touches, and the fall speeds, the evaporation timescale and the self-collection all
    # read that size. The knob exists as the FORENSIC comparison (WRF's own
    # `bl_mynn_mixscalars = 0` is the same lever) and never as a production setting; the
    # twelve ICE moments are NOT gated by it, because there the mass and its number/axis
    # moments are what the habit prediction is made of.
    mix_numbers = get(options, :mynn_mix_numbers, true)::Bool
    interval_sec = Float64(get(options, :mynn_interval, 20.0))
    # Default OFF (N2): the MYNN fields now ride in the model's own comprehensive
    # `<t>.nc` (src/netcdf_output.jl writes them from `mynn_diagnostics`, regridded onto
    # the regular output grid), so the mish-native sidecar is no longer what a reader
    # reaches for -- it is the FORENSIC file, on the closure's own mish with no
    # interpolation, and it is opted into explicitly. The benchmarks whose diagnostics
    # parse it (`benchmarks/ocean_warm_bubble.jl`, `benchmarks/o01_rainfall.jl`) set
    # `:mynn_output => true` themselves. `:radiation_output` moved the same way and for
    # the same reason (src/radiation_state.jl).
    output = get(options, :mynn_output, false)::Bool
    trace = get(options, :mynn_trace, true)::Bool
    check_values = get(options, :mynn_check_values, false)::Bool
    K_max = Float64(get(physical_params, :mynn_K_max, Inf))

    # The OFF path is resolved BEFORE anything can raise, and takes the no-deviation
    # fidelity unexamined: a run with the closure off must never be able to fail on a
    # MYNN rule, and nothing reads the field on that path.
    resolved = (; active = on, closure, edmf, edmf_mom, scale_aware, init_mode,
                fidelity = MYNN_FORTRAN_FIDELITY,
                water_carry, mix_numbers, interval_steps = 1, K_max, output, trace,
                check_values)
    on || return resolved

    for key in keys(options)
        (startswith(String(key), "mynn") && !(key in MYNN_OPTION_KEYS)) && error(
            "options[:$key] is not a recognized MYNN option. The MYNN keys are " *
            join(string.(":", sort!(collect(MYNN_OPTION_KEYS); by = String)), ", "))
    end

    uses_pressure_reference(equation_set) || error(
        "options[:mynn] needs a pressure-reference (moist_compressible) equation set — " *
        "the TKE density is an APPENDED prognostic slot of that set and the tendencies " *
        "enter through its momentum/heat/water channels (got equation_set = " *
        "\"$(equation_set)\")")

    closure == 2.5 || error(
        "options[:mynn_closure] = $closure is not available; the port is Level 2.5. " *
        "Level 2.6 (the prognostic tsq/qsq/cov set) arrives later — it needs three more " *
        "carried fields and a second predictor")
    (edmf == 0 || edmf == 1) || error(
        "options[:mynn_edmf] must be 0 (no mass flux) or 1, got $edmf")
    # `DMP_mf` reaches k+1 and k-1 from the interior interfaces and `EDMFWork` refuses a
    # column shorter than four layers; say so here rather than in the constructor.
    (edmf == 0 || kDim >= 4) || error(
        "options[:mynn_edmf] = 1 needs at least 4 vertical layers (grid_params.kDim = " *
        "$kDim); DMP_mf integrates the plumes on kts+1 : kte-1")

    _mynn_check(init_mode, MYNN_INIT_MODES, "mynn_init")
    fidelity = _mynn_resolve_fidelity(get(options, :mynn_fidelity, :fortran), options)
    _mynn_check(water_carry, MYNN_WATER_CARRIES, "mynn_water_carry")

    ts > 0.0 || error("validate_mynn_options: the model timestep must be positive, " *
                      "got ts = $ts")
    interval_sec > 0.0 || error(
        "options[:mynn_interval] must be positive seconds, got $interval_sec")
    interval_sec >= ts || error(
        "options[:mynn_interval] = $interval_sec s is shorter than the model timestep " *
        "ts = $ts s. The cadence is in SECONDS so it is nest-invariant (like " *
        ":output_interval and :radiation_interval); a boundary layer cannot be called " *
        "more often than the model steps")
    # Seconds -> steps, exactly once, here.
    interval_steps = max(1, round(Int, interval_sec / ts))

    (isinf(K_max) || K_max > 0.0) || error(
        "physical_params[:mynn_K_max] must be positive (or Inf for no cap), got $K_max")
    kDim > 0 || error("options[:mynn] needs a vertical dimension (grid_params.kDim = 0); " *
                      "the boundary layer is a column process")

    return (; active = on, closure, edmf, edmf_mom, scale_aware, init_mode, fidelity,
            water_carry, mix_numbers, interval_steps, K_max, output, trace, check_values)
end

# ── Setup ─────────────────────────────────────────────────────────────────────

"""
    mynn_setup_line(model, cfg, ncol, kDim)

One line, `mynn: ...`, stating the resolved configuration of this tile's boundary layer:
closure level, the EDMF switch, scale awareness, the initialization mode, the water-carry
convention, the cadence in seconds and in this patch's steps (nests differ in `ts`), the
`K` cap and the column geometry. Printed to stdout so it lands in the per-nest
`scythe_out.log`, beside the `radiation:` line (see [`radiation_setup_line`](@ref) for why
the tile constructor and not the run script prints it).
"""
function mynn_setup_line(model::ModelParameters, cfg, ncol::Int, kDim::Int)
    cfg.trace || return nothing
    interval_s = Float64(get(model.options, :mynn_interval, 20.0))
    println("mynn: closure=$(cfg.closure) edmf=$(cfg.edmf) " *
            "edmf_mom=$(cfg.edmf_mom) " *
            "scale_aware=$(cfg.scale_aware) init=:$(cfg.init_mode) " *
            "water_carry=:$(cfg.water_carry) mix_numbers=$(cfg.mix_numbers) " *
            "fidelity=$(mynn_fidelity_string(cfg.fidelity)) " *
            "interval=$(interval_s) s (= $(cfg.interval_steps) steps at ts=$(model.ts) s) " *
            "K_max=$(cfg.K_max) m^2/s columns=$(ncol) layers=$(kDim)")
    return nothing
end

"""
    mc_mynn_state(model, tile, tilepoints) -> MYNNState

The tile's MYNN-EDMF state, or [`EMPTY_MYNN`](@ref) when `options[:mynn]` is absent or
`false`.

Setup path only — once per tile, never per column, and the exact analogue of
[`mc_radiation_state`](@ref) beside it in `createModelTile`. Everything that can be wrong
about a MYNN configuration is diagnosed HERE, by [`validate_mynn_options`](@ref), rather
than at the first boundary-layer call.

The vertical basis is checked here rather than in the validator because it is a property
of the GRID, not of the options: the closure's mixing length, its wall treatment and the
`rho_e` fit all assume the cubic B-spline (RiRk) column, and a Chebyshev vertical would
run it silently on a basis it was never derived for.
"""
function mc_mynn_state(model::ModelParameters, tile, tilepoints)

    get(model.options, :mynn, false) === true || return EMPTY_MYNN

    kDim = model.grid_params.kDim
    cfg = validate_mynn_options(model.options, model.physical_params,
                                model.equation_set, model.ts, kDim)

    ncol = num_columns(tile)
    ncol > 0 || error(
        "options[:mynn] needs a grid with vertical columns (num_columns(tile) = 0); the " *
        "boundary layer is a column process")
    tile.kbasis isa Springsteel.SplineBasisArray || error(
        "options[:mynn] requires the cubic B-spline (RiRk) vertical — the closure's " *
        "mixing length and wall treatment are derived on that column, and the TKE " *
        "density slot is fitted on it")

    npoints = size(tile.physical, 1)
    zpt() = zeros(Float64, npoints)
    zcol() = zeros(Float64, ncol)

    # ── Column geometry, built ONCE ───────────────────────────────────────────────
    # One column of the vertical mish (`tilepoints[:, end]` is z on every
    # moist-compressible geometry -- the expression `mc_radiation_state` uses), and the
    # MYNN layer thickness from faces at the MIDPOINTS between mish points, plus the
    # ground and the lid. That is the radiation convention (`radiation_faces`) and the
    # one `tools/mynn_dump_columns.jl` dumped the Fortran reference columns with, so the
    # closure sees exactly the `dz`/`zw` the parity harness validated. `sum(dz) == z_top`
    # by construction, and `zw` is bitwise `mynn_wall_heights(dz)`.
    z_lay = collect(Float64, tilepoints[1:kDim, end])
    z_bottom = Float64(model.grid_params.kMin)
    z_top = Float64(model.grid_params.kMax)
    # The closure's heights are ABOVE GROUND (`zw[1] == 0` is what `mynn_wall_heights`,
    # the mixing-length wall term and the TKE taper all assume) while the mish carries
    # absolute z. Every moist-compressible configuration puts the ground at z = 0, so
    # rather than carry two height systems this refuses the one case where they differ.
    z_bottom == 0.0 || error(
        "options[:mynn] needs the ground at z = 0 (grid_params.kMin = $(z_bottom)); the " *
        "closure's wall heights, mixing length and TKE taper are all above-ground-level")
    zw, _ = radiation_faces(z_lay, z_bottom, z_top)
    dz = diff(zw)
    # The mish points are Gauss quadrature nodes of the vertical B-spline cells, so the
    # column integral of any fitted field is the weighted sum with these weights -- what
    # the D3 energy identity is checked against, and what `benchmarks/common/diagnostics.jl`
    # `gauss_cell_weights` builds for the same purpose.
    dz_cell = (z_top - z_bottom) / model.grid_params.num_cells_k
    qw_cell = Springsteel.CubicBSpline._quadrature_rule(model.grid_params.mubar,
                                                        model.grid_params.quadrature)[2]
    w_mish = repeat(qw_cell .* dz_cell, outer = model.grid_params.num_cells_k)
    length(w_mish) == kDim || error(
        "mc_mynn_state: the vertical mish has $(kDim) points but " *
        "$(model.grid_params.num_cells_k) cells x mubar = $(length(w_mish)); the MYNN " *
        "column quadrature assumes one Gauss rule per B-spline cell")
    dz_min = minimum(diff(z_lay))
    # SCALE_AWARE's horizontal scale: this patch's own i-direction cell width (dr on the
    # cylinders), so `Psig_bl` tapers the closure in the gray zone from the grid the run
    # actually has -- never a hardcoded number (D12).
    dx = (Float64(model.grid_params.iMax) - Float64(model.grid_params.iMin)) /
         model.grid_params.num_cells_i
    dx > 0.0 || error("mc_mynn_state: the horizontal cell width came out $(dx) m; " *
                      "SCALE_AWARE needs a positive dx")

    st = MYNNState(; active = true, closure = cfg.closure, edmf = cfg.edmf,
        edmf_mom = cfg.edmf_mom,
        scale_aware = cfg.scale_aware, init_mode = cfg.init_mode,
        fidelity = cfg.fidelity, water_carry = cfg.water_carry,
        mix_numbers = cfg.mix_numbers,
        interval_steps = cfg.interval_steps, ncol, kDim, K_max = cfg.K_max,
        output = cfg.output, trace = cfg.trace, check_values = cfg.check_values,
        el = zpt(), sm = zpt(), sh = zpt(), vt = zpt(), vq = zpt(), sgm = zpt(),
        cldfra_bl = zpt(), qc_bl = zpt(), qi_bl = zpt(), K_m = zpt(), K_h = zpt(),
        gh = zpt(), gm = zpt(), qsq = zpt(),
        g_Ps = zpt(), g_Ps_mynn = zpt(), g_Pb = zpt(), g_eps = zpt(),
        g_tke_transport = zpt(),
        s_aw = zpt(), s_aw_st = zpt(), s_aw_qw = zpt(), s_aw_qv = zpt(),
        s_aw_u = zpt(), s_aw_v = zpt(), s_aw_e = zpt(),
        plume_ktop = zeros(Int, ncol), plume_ztop = zcol(), aw_max = zcol(),
        n_gate_col = zeros(Int, ncol), n_stall_col = zeros(Int, ncol),
        n_plume_col = zeros(Int, ncol),
        pblh = zcol(), kpbl = zeros(Int, ncol), ust = zcol(), rmol = zcol(),
        pmz = zcol(),
        z_lay = z_lay, dz = dz, zw = zw, w_mish = w_mish, dx = dx, z_top = z_top,
        dz_cell = dz_cell, dz_min = dz_min, ts = Float64(model.ts),
        bdry_E = zcol(), D_gal = zcol(), D_mish = zcol(), ts_tau = zcol(),
        K_m_max = zcol(), K_h_max = zcol(), Ps_disc = zcol(), Ps_mynn = zcol(),
        n_clamp_col = zeros(Int, ncol), n_capK_col = zeros(Int, ncol),
        n_diffnum_col = zeros(Int, ncol),
        n_hfx_clip_col = zeros(Int, ncol), n_qfx_clip_col = zeros(Int, ncol),
        # `typemin(Int)` rather than 0: "no call yet" must be distinguishable from
        # "called at step 0" so the first step forces an update whatever the cadence is
        # (the `RadiationState.last_call_step` argument).
        last_update_step = fill(typemin(Int), ncol),
        # Per-thread, indexed by `threadid()` under the same `@threads :static` ownership
        # rule `scratch_columns` uses. One `MYNNWork` is ~40 kDim-length columns.
        work = [MYNNWork(kDim) for _ in 1:Threads.maxthreadid()],
        colscratch = [MYNNColumnScratch(kDim) for _ in 1:Threads.maxthreadid()],
        # Allocated ONLY with the plumes on: nine (kDim+1, 8) matrices per thread.
        ework = cfg.edmf == 1 ? [EDMFWork(kDim) for _ in 1:Threads.maxthreadid()] :
                                EDMFWork[],
        constants = MYNNConstants())

    mynn_setup_line(model, cfg, ncol, kDim)
    return st
end
