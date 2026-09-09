# ── MYNN-EDMF closure: level-2 gradients, mixing length, PBLH, initialization ──
#
# Part 1 of the pure-Julia port of the MYNN-EDMF boundary-layer scheme. Source of
# truth: `tools/mynn_fortran_driver/module_bl_mynn.F90`, a VERBATIM copy of
# ccpp-physics at 72570a3f (see tools/mynn_fortran_driver/README.md). Every routine
# below carries the Fortran line range it was transcribed from; line numbers refer to
# that file unless stated otherwise. The numbers this port must reproduce live in
# `tools/mynn_fortran_driver/ref_driver_output_r8.txt` and are checked by
# test/test_mynn_closure.jl — no Fortran runs at test time.
#
# WHAT IS HERE (part 1): esat_blend, qsat_blend, xl_blend, phim, phih, tridiag2!,
# mym_level2!, boulac_length0!, mym_length! (CASE 2 only), get_pblh!, scale_aware and
# mym_initialize!.
#
# WHAT IS HERE (part 2, the second half of this file): mym_turbulence!, mym_predict!,
# mym_condensation!, mynn_tendencies!, moisture_check!, retrieve_exchange_coeffs! and
# the per-column drivers mynn_init_column! / mynn_column_step!. The mass-flux plumes
# (DMP_mf) and the 3-D wrapper are later stages still.
#
# ── Conventions ───────────────────────────────────────────────────────────────
#
# INDEXING. The Fortran runs `kts:kte`; this port fixes `kts = 1`, `kte = n` and takes
# plain `Vector{Float64}` of length `n` (or `n+1` for `zw`). Both bounds are still
# passed explicitly so the transcription reads like the original and so a future
# sub-column call is a one-line change. Layer 1 is the lowest model layer.
#
# STAGGERING. This is the single easiest thing to get wrong. Layer-centred fields —
# `u, v, thl, qw, theta, thetav, qke, dz, cldfra_bl1D` — are cell means at the mish
# points. INTERFACE fields — `el, sh, sm, sm2, sh2, qkw, dtl, dqw, dtv, gm, gh` — live
# on the box WALLS: index `k` means the wall between layer `k-1` and layer `k`, whose
# height is `zw(k)`. Consequences that the code below relies on:
#   * `zw` has length `n+1`, `zw(1) = 0` is the ground and `zw(k) = sum(dz(1:k-1))`.
#   * every interface loop runs `kts+1:kte`, so element `kts` of an interface array is
#     NEVER written by these routines. `mym_level2!` leaves `sm(1)`, `sh(1)` exactly as
#     the caller left them (the reference driver zeroes them, and `init_sh(1) = 0` in
#     the reference is that zero, not a computed value). `mym_length!` writes
#     `el(kts) = 0` explicitly.
#   * `dz(k)` and `dz(k-1)` straddle wall `k`, hence the recurring
#     `dzk = 0.5*(dz(k)+dz(k-1))`, `afk = dz(k)/(dz(k)+dz(k-1))`, `abk = 1-afk`.
# See module_bl_mynn.F90 :72-83 (the grid-arrangement box in the module header) and
# :1844-1846 (mym_length's own note).
#
# FLOATING POINT. The reference build is `-O0 -ffp-contract=off` with every bare `real`
# promoted to double, so it is unfused IEEE double throughout — which is exactly what
# Julia gives by default. Therefore: the operation ORDER and the ASSOCIATION inside
# every expression below are the Fortran's, verbatim; do NOT "simplify" an expression,
# hoist a subexpression, or reuse a value the Fortran recomputes. Do NOT introduce
# `muladd`, `@fastmath`, `@simd` or `@turbo` — a fused multiply-add changes the answer
# and the reference has none. `x**2` with an INTEGER exponent is a multiplication in
# Fortran and is written `x^2` here (Julia's `literal_pow` lowers it to `x*x`);
# `x**2.` and `x**onethird` have REAL exponents and are `libm` `pow` calls in both
# languages, so they are written `x^2.0` and `x^(1.0/3.0)`.
#
# ALLOCATION. Every `!` routine is allocation-free on a warm call: all locals that the
# Fortran declares as automatic arrays live in a preallocated `MYNNWork`, and the
# routines with several scalar `intent(out)`s return a `Tuple` (isbits, stack-only)
# rather than writing into `Ref`s. test/test_mynn_closure.jl asserts `@allocations == 0`.

# ── Work space ────────────────────────────────────────────────────────────────

"""
    MYNNWork(n)

Preallocated scratch for the MYNN closure on a column of `n` layers. One instance is
reused for every call, which is what keeps the routines allocation-free.

The fields are the automatic arrays the Fortran declares inside its subroutines:

| field                         | Fortran home                          |
|:------------------------------|:--------------------------------------|
| `ql,vt,vq`                    | `mym_initialize` locals (:1542-1544)  |
| `pdk,pdt,pdq,pdc`             | `mym_initialize` locals (:1542-1544)  |
| `dtl,dqw,dtv,gm,gh`           | `mym_initialize` locals (:1542-1544)  |
| `qkw`                         | `mym_length`'s `intent(out)` (:1879)  |
| `qtke,elBLmin,elBLavg,thetaw` | `mym_length` locals (:1883)           |
| `tri_cp,tri_dp`               | `tridiag2`'s `cp`/`dp` (:5436)        |

`elBLmin`, `elBLavg` and `thetaw` are unused by the CASE-2 mixing length (they feed
`boulac_length`, which only CASE 1 calls) but are carried so `boulac_length0!` and a
future CASE-1 port need no new allocation.

PART 2 added the rest of the closure's automatic arrays, prefixed by owner so that no
two live ranges can overlap (no two of these routines is ever active at once):

| prefix   | owner                                                  |
|:---------|:-------------------------------------------------------|
| `p_*`    | `mym_predict!` locals (:3228-3238)                     |
| `c_*`    | `mym_condensation!` locals (:3633)                     |
| `e_*`    | `mynn_tendencies!` locals (:4107-4113)                 |
| `s_*`    | `mynn_column_step!`'s per-call working copies of the frozen column |
| `out_*`  | `mynn_column_step!`'s per-step OUTPUTS — the arrays a host would own |

`z_n` (length `n`) and `z_np1` (length `n+1`) are shared all-zero buffers standing in
for the arguments the reference run passes as zero columns: the mass-flux plume sums
(`s_aw*`, `sd_aw*`, `sub_*`, `det_*`), `TKEprodTD`, `rstoch_col`, the unmixed species
(`qs`, `qnc`, `qni`, `qnwfa`, `qnifa`, `qnbca`, `ozone`) and the zeroed `edmf_w1`,
`edmf_a1`. NOTHING on the ported code paths writes them; if a later stage (DMP_mf)
starts filling the plume sums it must give them arrays of their own.

The length-`n+1` (interface) fields are listed in `MYNN_WORK_FACE_FIELDS`.
"""
struct MYNNWork
    n::Int
    # -- part 1 ---------------------------------------------------------------
    ql::Vector{Float64}
    vt::Vector{Float64}
    vq::Vector{Float64}
    pdk::Vector{Float64}
    pdt::Vector{Float64}
    pdq::Vector{Float64}
    pdc::Vector{Float64}
    dtl::Vector{Float64}
    dqw::Vector{Float64}
    dtv::Vector{Float64}
    gm::Vector{Float64}
    gh::Vector{Float64}
    qkw::Vector{Float64}
    qtke::Vector{Float64}
    elBLmin::Vector{Float64}
    elBLavg::Vector{Float64}
    thetaw::Vector{Float64}
    tri_cp::Vector{Float64}
    tri_dp::Vector{Float64}
    # -- shared all-zero stand-ins (never written) -----------------------------
    z_n::Vector{Float64}
    z_np1::Vector{Float64}
    # -- mym_predict! ----------------------------------------------------------
    p_qkw::Vector{Float64}
    p_bp::Vector{Float64}
    p_rp::Vector{Float64}
    p_df3q::Vector{Float64}
    p_dtz::Vector{Float64}
    p_a::Vector{Float64}
    p_b::Vector{Float64}
    p_c::Vector{Float64}
    p_d::Vector{Float64}
    p_x::Vector{Float64}
    p_rhoinv::Vector{Float64}
    p_tke_up::Vector{Float64}
    p_dzinv::Vector{Float64}
    p_rhoz::Vector{Float64}
    p_kqdz::Vector{Float64}
    p_kmdz::Vector{Float64}
    # -- mym_condensation! -----------------------------------------------------
    c_alp::Vector{Float64}
    c_a::Vector{Float64}
    c_bet::Vector{Float64}
    c_b::Vector{Float64}
    c_ql::Vector{Float64}
    c_q1::Vector{Float64}
    c_rh::Vector{Float64}
    # -- mynn_tendencies! ------------------------------------------------------
    e_dtz::Vector{Float64}
    e_delp::Vector{Float64}
    e_sqv2::Vector{Float64}
    e_sqc2::Vector{Float64}
    e_sqi2::Vector{Float64}
    e_sqs2::Vector{Float64}
    e_sqw2::Vector{Float64}
    e_qni2::Vector{Float64}
    e_qnc2::Vector{Float64}
    e_qnwfa2::Vector{Float64}
    e_qnifa2::Vector{Float64}
    e_qnbca2::Vector{Float64}
    e_rhoinv::Vector{Float64}
    e_a::Vector{Float64}
    e_b::Vector{Float64}
    e_c::Vector{Float64}
    e_d::Vector{Float64}
    e_x::Vector{Float64}
    e_rhoz::Vector{Float64}
    e_khdz::Vector{Float64}
    e_kmdz::Vector{Float64}
    # -- mynn_column_step! working copies of the frozen column -----------------
    s_u::Vector{Float64}
    s_v::Vector{Float64}
    s_w::Vector{Float64}
    s_th::Vector{Float64}
    s_tk::Vector{Float64}
    s_sqv::Vector{Float64}
    s_sqc::Vector{Float64}
    s_sqi::Vector{Float64}
    s_sqw::Vector{Float64}
    s_thl::Vector{Float64}
    s_thetav::Vector{Float64}
    s_qv::Vector{Float64}
    s_qc::Vector{Float64}
    s_qi::Vector{Float64}
    # -- mynn_column_step! per-step outputs ------------------------------------
    out_dfm::Vector{Float64}
    out_dfh::Vector{Float64}
    out_dfq::Vector{Float64}
    out_tcd::Vector{Float64}
    out_qcd::Vector{Float64}
    out_pdk::Vector{Float64}
    out_pdt::Vector{Float64}
    out_pdq::Vector{Float64}
    out_pdc::Vector{Float64}
    out_qwt::Vector{Float64}
    out_qshear::Vector{Float64}
    out_qbuoy::Vector{Float64}
    out_qdiss::Vector{Float64}
    out_diss_heat::Vector{Float64}
    out_du::Vector{Float64}
    out_dv::Vector{Float64}
    out_dth::Vector{Float64}
    out_dqv::Vector{Float64}
    out_dqc::Vector{Float64}
    out_dqi::Vector{Float64}
    out_dqs::Vector{Float64}
    out_dqnc::Vector{Float64}
    out_dqni::Vector{Float64}
    out_dqnwfa::Vector{Float64}
    out_dqnifa::Vector{Float64}
    out_dqnbca::Vector{Float64}
    out_dozone::Vector{Float64}
    out_km::Vector{Float64}
    out_kh::Vector{Float64}
    # -- the `:gtr_local` fidelity deviation (MYNN_DEVIATIONS) -----------------
    # `g/theta_v` per LEVEL, filled from the staged `theta_v` by whichever caller asked
    # for the deviation and passed as the `gtr_k` keyword to `mym_level2!`,
    # `mym_length!`, `mym_turbulence!` and `dmp_mf!`. Preallocated here so the switch
    # costs no allocation; left all-zero, and never passed, under `:fortran`.
    gtr_k::Vector{Float64}
end

"""
    MYNN_WORK_FACE_FIELDS

The `MYNNWork` fields that live on the box walls and are therefore `n+1` long; every
other field is `n` long.
"""
const MYNN_WORK_FACE_FIELDS = (:z_np1, :p_rhoz, :p_kqdz, :p_kmdz,
                               :e_rhoz, :e_khdz, :e_kmdz)

function MYNNWork(n::Integer)
    n = Int(n)
    names = fieldnames(MYNNWork)
    args = Any[n]
    for f in names[2:end]
        push!(args, zeros(Float64, (f in MYNN_WORK_FACE_FIELDS) ? n + 1 : n))
    end
    return MYNNWork(args...)
end

# ── Saturation and latent-heat blends (module_bl_mynn.F90 :7395-7524) ─────────
#
# Liquid (J*) and ice (K*) saturation-vapour-pressure polynomials, identical to
# module_mp_thompson.F v3.6, blended over `tice < t < t0c-6`. The Fortran declares the
# coefficients as `parameter`s inside each function; they are repeated inside each
# function here for the same reason (they are the function's own definition), and the
# constant folder makes that free.
#
# The blend edges depend on the HOST `t0c`, so these take a `MYNNConstants`. With
# Springsteel's `T_0 = 273.16` the warm edge is 267.16 K, not the 268.16 K of the
# Fortran's own comment — see the `MYNNConstants` docstring.

"""
    esat_blend(t, c::MYNNConstants) -> Float64

Saturation vapour pressure (Pa) at temperature `t` (K), phase-blended between liquid
and ice. Fortran: `FUNCTION esat_blend` (:7395-7438).
"""
function esat_blend(t::Float64, c::MYNNConstants)
    # liquid
    J0 =  0.611583699e03; J1 =  0.444606896e02; J2 =  0.143177157e01
    J3 =  0.264224321e-1; J4 =  0.299291081e-3; J5 =  0.203154182e-5
    J6 =  0.702620698e-8; J7 =  0.379534310e-11; J8 = -0.321582393e-13
    # ice
    K0 =  0.609868993e03; K1 =  0.499320233e02; K2 =  0.184672631e01
    K3 =  0.402737184e-1; K4 =  0.565392987e-3; K5 =  0.521693933e-5
    K6 =  0.307839583e-7; K7 =  0.105785160e-9; K8 =  0.161444444e-12

    XC = max(-80.0, t - c.t0c)

    if t >= (c.t0c - 6.0)
        return J0+XC*(J1+XC*(J2+XC*(J3+XC*(J4+XC*(J5+XC*(J6+XC*(J7+XC*J8)))))))
    elseif t <= MYNN_TICE
        return K0+XC*(K1+XC*(K2+XC*(K3+XC*(K4+XC*(K5+XC*(K6+XC*(K7+XC*K8)))))))
    else
        ESL = J0+XC*(J1+XC*(J2+XC*(J3+XC*(J4+XC*(J5+XC*(J6+XC*(J7+XC*J8)))))))
        ESI = K0+XC*(K1+XC*(K2+XC*(K3+XC*(K4+XC*(K5+XC*(K6+XC*(K7+XC*K8)))))))
        chi = ((c.t0c - 6.0) - t) / ((c.t0c - 6.0) - MYNN_TICE)
        return (1.0 - chi)*ESL + chi*ESI
    end
end

"""
    qsat_blend(t, P, c::MYNNConstants) -> Float64

Phase-blended saturation mixing ratio (kg/kg) at temperature `t` (K) and pressure `P`
(Pa). Fortran: `FUNCTION qsat_blend` (:7446-7495).

Note the `min(ESL, P*0.15)` cap the Fortran applies BEFORE the blend, and that the
0.622 is a bare literal in the Fortran — not `ep_2`, which for Springsteel's constants
is 0.62197..., a 0.005 % difference. Reproduced as written.
"""
function qsat_blend(t::Float64, P::Float64, c::MYNNConstants)
    # liquid
    J0 =  0.611583699e03; J1 =  0.444606896e02; J2 =  0.143177157e01
    J3 =  0.264224321e-1; J4 =  0.299291081e-3; J5 =  0.203154182e-5
    J6 =  0.702620698e-8; J7 =  0.379534310e-11; J8 = -0.321582393e-13
    # ice
    K0 =  0.609868993e03; K1 =  0.499320233e02; K2 =  0.184672631e01
    K3 =  0.402737184e-1; K4 =  0.565392987e-3; K5 =  0.521693933e-5
    K6 =  0.307839583e-7; K7 =  0.105785160e-9; K8 =  0.161444444e-12

    XC = max(-80.0, t - c.t0c)

    if t >= (c.t0c - 6.0)
        ESL = J0+XC*(J1+XC*(J2+XC*(J3+XC*(J4+XC*(J5+XC*(J6+XC*(J7+XC*J8)))))))
        ESL = min(ESL, P*0.15)
        return 0.622*ESL/max(P - ESL, 1e-5)
    elseif t <= MYNN_TICE
        ESI = K0+XC*(K1+XC*(K2+XC*(K3+XC*(K4+XC*(K5+XC*(K6+XC*(K7+XC*K8)))))))
        ESI = min(ESI, P*0.15)
        return 0.622*ESI/max(P - ESI, 1e-5)
    else
        ESL = J0+XC*(J1+XC*(J2+XC*(J3+XC*(J4+XC*(J5+XC*(J6+XC*(J7+XC*J8)))))))
        ESL = min(ESL, P*0.15)
        ESI = K0+XC*(K1+XC*(K2+XC*(K3+XC*(K4+XC*(K5+XC*(K6+XC*(K7+XC*K8)))))))
        ESI = min(ESI, P*0.15)
        RSLF = 0.622*ESL/max(P - ESL, 1e-5)
        RSIF = 0.622*ESI/max(P - ESI, 1e-5)
        chi = ((c.t0c - 6.0) - t) / ((c.t0c - 6.0) - MYNN_TICE)
        return (1.0 - chi)*RSLF + chi*RSIF
    end
end

"""
    xl_blend(t, c::MYNNConstants) -> Float64

Temperature-dependent blended latent heat (J/kg) — vaporization below the freezing
point, sublimation below `tice`, a linear blend between. Chaboureau and Bechtold
(2002), Appendix. Fortran: `FUNCTION xl_blend` (:7504-7524).

The blend edge here is `t0c` itself (not `t0c-6` as in `esat_blend`/`qsat_blend`),
which is the Fortran's own asymmetry.
"""
function xl_blend(t::Float64, c::MYNNConstants)
    if t >= c.t0c
        return c.xlv + (c.cpv - c.cliq)*(t - c.t0c)   # vaporization/condensation
    elseif t <= MYNN_TICE
        return c.xls + (c.cpv - c.cice)*(t - c.t0c)   # sublimation/deposition
    else
        xlvt = c.xlv + (c.cpv - c.cliq)*(t - c.t0c)
        xlst = c.xls + (c.cpv - c.cice)*(t - c.t0c)
        chi  = (c.t0c - t) / (c.t0c - MYNN_TICE)
        return (1.0 - chi)*xlvt + chi*xlst
    end
end

# ── Surface-layer stability functions (module_bl_mynn.F90 :7528-7626) ─────────
#
# Puhales (2020), WRF 4.2.1 forms (`bl_mynn_stfunc = 1`): Grachev et al. (2000) in
# unstable conditions, Cheng and Brutsaert (2005) in stable ones. The `dummy_*` names
# are the Fortran's and are kept so the two files diff line for line.
#
# Note the last two lines of `phim`: the Fortran computes `phi_m` and then returns it
# UNCHANGED, with `phim = phi_m - zet` commented out (:7573-7574). The caller subtracts
# `zet` itself (`pmz = phim(zet) - zet`, ref_driver.f90; mynn_bl_driver :1085). Do not
# fold that subtraction in here.

"""
    phim(zet) -> Float64

Non-dimensional momentum gradient phi_m at stability parameter `zet` = z/L.
Fortran: `FUNCTION phim` (:7528-7577).
"""
function phim(zet::Float64)
    am_st = 6.1; bm_st = 2.5; rbm_st = 1.0/bm_st
    am_unst = 10.0

    if zet >= 0.0
        dummy_0  = 1 + zet^bm_st
        dummy_1  = zet + dummy_0^(rbm_st)
        dummy_11 = 1 + dummy_0^(rbm_st - 1)*zet^(bm_st - 1)
        dummy_2  = (-am_st/dummy_1)*dummy_11
        phi_m = 1 - zet*dummy_2
    else
        dummy_0 = (1.0 - MYNN_CPHM_UNST*zet)^0.25
        phi_m = 1.0/dummy_0
        dummy_psi = 2.0*log(0.5*(1.0 + dummy_0)) + log(0.5*(1.0 + dummy_0^2)) -
                    2.0*atan(dummy_0) + 1.570796

        dummy_0  = (1.0 - am_unst*zet)                     # parenthesis arg
        dummy_1  = dummy_0^0.333333                        # y
        dummy_11 = -0.33333*am_unst*dummy_0^(-0.6666667)   # dy/dzet
        dummy_2  = 0.33333*(dummy_1^2.0 + dummy_1 + 1.0)   # f
        dummy_22 = 0.3333*dummy_11*(2.0*dummy_1 + 1.0)     # df/dzet
        dummy_3  = 0.57735*(2.0*dummy_1 + 1.0)             # g
        dummy_33 = 1.1547*dummy_11                         # dg/dzet
        dummy_4  = 1.5*log(dummy_2) - 1.73205*atan(dummy_3) + 1.813799364      # psic
        dummy_44 = (1.5/dummy_2)*dummy_22 - 1.73205*dummy_33/(1.0 + dummy_3^2) # dpsic/dzet

        dummy_0  = zet^2
        dummy_1  = 1.0/(1.0 + dummy_0)   # denominator
        dummy_11 = 2.0*zet               # d(denominator)/dzet
        dummy_2  = ((1 - phi_m)/zet + dummy_11*dummy_4 + dummy_0*dummy_44)*dummy_1
        dummy_22 = -dummy_11*(dummy_psi + dummy_0*dummy_4)*dummy_1^2

        phi_m = 1.0 - zet*(dummy_2 + dummy_22)
    end

    # :7573 `phim = phi_m - zet` is commented out in the Fortran; the caller does it.
    return phi_m
end

"""
    phih(zet) -> Float64

Non-dimensional heat gradient phi_h at stability parameter `zet` = z/L.
Fortran: `FUNCTION phih` (:7580-7626).
"""
function phih(zet::Float64)
    ah_st = 5.3; bh_st = 1.1; rbh_st = 1.0/bh_st
    ah_unst = 34.0

    if zet >= 0.0
        dummy_0  = 1 + zet^bh_st
        dummy_1  = zet + dummy_0^(rbh_st)
        dummy_11 = 1 + dummy_0^(rbh_st - 1)*zet^(bh_st - 1)
        dummy_2  = (-ah_st/dummy_1)*dummy_11
        return 1 - zet*dummy_2
    else
        dummy_0 = (1.0 - MYNN_CPHH_UNST*zet)^0.5
        phh = 1.0/dummy_0
        dummy_psi = 2.0*log(0.5*(1.0 + dummy_0))

        dummy_0  = (1.0 - ah_unst*zet)
        dummy_1  = dummy_0^0.333333
        dummy_11 = -0.33333*ah_unst*dummy_0^(-0.6666667)
        dummy_2  = 0.33333*(dummy_1^2.0 + dummy_1 + 1.0)
        dummy_22 = 0.3333*dummy_11*(2.0*dummy_1 + 1.0)
        dummy_3  = 0.57735*(2.0*dummy_1 + 1.0)
        dummy_33 = 1.1547*dummy_11
        dummy_4  = 1.5*log(dummy_2) - 1.73205*atan(dummy_3) + 1.813799364
        dummy_44 = (1.5/dummy_2)*dummy_22 - 1.73205*dummy_33/(1.0 + dummy_3^2)

        dummy_0  = zet^2
        dummy_1  = 1.0/(1.0 + dummy_0)
        dummy_11 = 2.0*zet
        dummy_2  = ((1 - phh)/zet + dummy_11*dummy_4 + dummy_0*dummy_44)*dummy_1
        dummy_22 = -dummy_11*(dummy_psi + dummy_0*dummy_4)*dummy_1^2

        return 1.0 - zet*(dummy_2 + dummy_22)
    end
end

# ── Tridiagonal solve (module_bl_mynn.F90 :5423-5455) ─────────────────────────

"""
    tridiag2!(n, a, b, c, d, x, work::MYNNWork)

Thomas algorithm: solve the tridiagonal system with sub-diagonal `a`, main diagonal
`b`, super-diagonal `c` and right-hand side `d` into `x`, all of length ≥ `n`.
Fortran: `subroutine tridiag2` (:5423-5455).

`a(1)` and `c(n)` are never read, as in the Fortran. The `cp`/`dp` scratch comes from
`work` (`tri_cp`, `tri_dp`) so the call allocates nothing; they must be at least `n`
long, which `MYNNWork(n)` guarantees for a full column.
"""
function tridiag2!(n::Int, a::Vector{Float64}, b::Vector{Float64}, c::Vector{Float64},
                   d::Vector{Float64}, x::Vector{Float64}, work::MYNNWork)
    cp = work.tri_cp
    dp = work.tri_dp

    # initialize c-prime and d-prime
    cp[1] = c[1]/b[1]
    dp[1] = d[1]/b[1]
    # solve for vectors c-prime and d-prime
    for i in 2:n
        m = b[i] - cp[i-1]*a[i]
        cp[i] = c[i]/m
        dp[i] = (d[i] - dp[i-1]*a[i])/m
    end
    # initialize x
    x[n] = dp[n]
    # solve for x from the vectors c-prime and d-prime
    for i in (n-1):-1:1
        x[i] = dp[i] - cp[i]*x[i+1]
    end
    return x
end

# ── Level-2 gradients and stability functions (:1719-1830) ───────────────────

"""
    _gtr_face(gtr_k, c::MYNNConstants, dz, k) -> Float64

The buoyancy parameter `g/theta_v` at the wall between levels `k-1` and `k`.

`gtr_k === nothing` is the Fortran (`:mynn_fidelity = :fortran`): a single
`c.gtr = g/MYNN_TREF` with `MYNN_TREF = 300 K`, returned with no arithmetic at all, so
the default path is the same instruction it always was. With the `:gtr_local` deviation
`gtr_k` is the per-LEVEL `g/theta_v` and this is the interface interpolation the
neighbouring `vt`/`vq` lines of `mym_level2!` already use (`x[k]*abk + x[k-1]*afk`).

Julia specializes on the argument type, so the `Nothing` method is a compile-time
constant fold and neither branch is ever tested at run time.
"""
@inline _gtr_face(::Nothing, c::MYNNConstants, dz::Vector{Float64}, k::Int) = c.gtr
@inline function _gtr_face(gtr_k::Vector{Float64}, ::MYNNConstants,
                           dz::Vector{Float64}, k::Int)
    @inbounds begin
        afk = dz[k]/(dz[k] + dz[k-1])
        return gtr_k[k]*(1.0 - afk) + gtr_k[k-1]*afk
    end
end

"""
    _gtr_lev(gtr_k, c::MYNNConstants, k) -> Float64

`g/theta_v` at LEVEL `k`, the companion of [`_gtr_face`](@ref) for the terms the Fortran
builds at a level rather than across an interface: the surface ones (`vsc`, `wstar`, the
plume scaling, `rmol`, all at `kts`) and `dmp_mf!`'s overshoot Brunt-Vaisala frequency.
"""
@inline _gtr_lev(::Nothing, c::MYNNConstants, k::Int) = c.gtr
@inline _gtr_lev(gtr_k::Vector{Float64}, ::MYNNConstants, k::Int) = @inbounds gtr_k[k]


"""
    mym_level2!(kts, kte, dz, u, v, thl, thetav, qw, ql, vt, vq,
                dtl, dqw, dtv, gm, gh, sm, sh, c::MYNNConstants)

Vertical gradients of theta_l, q_w and theta_v, the non-dimensional shear G_M and
stratification G_H, and the LEVEL-2 stability functions S_m, S_h. Fortran: `SUBROUTINE mym_level2`
(:1719-1830).

All seven outputs are INTERFACE fields written for `kts+1:kte` only; element `kts` is
left untouched (see the staggering note at the top of this file). `thetav` is accepted
and ignored — the Fortran offers `dtq` from `thetav` as a commented-out alternative
(:1780) and keeps the argument.

The keyword `gtr_k` is the `:gtr_local` fidelity deviation (see `MYNN_DEVIATIONS`):
`nothing` (the default, and `:fortran`) uses the Fortran's single `c.gtr = g/300 K` for
`G_H`; a per-level `g/theta_v` column uses the same interface interpolation the `vt`/`vq`
lines above it use. See [`_gtr_face`](@ref).

The Canuto/Kitamura modification (`CKmod = 1`) recomputes `f1, smc, shc, ri1..ri4`
INSIDE the k-loop with `a2*a2fac` in place of `a2`, which makes the pre-loop block
(:1753-1765) dead. That block is transcribed anyway, marked, so the file stays a
line-for-line image of the Fortran.
"""
function mym_level2!(kts::Int, kte::Int,
                     dz::Vector{Float64},
                     u::Vector{Float64}, v::Vector{Float64}, thl::Vector{Float64},
                     thetav::Vector{Float64}, qw::Vector{Float64},
                     ql::Vector{Float64}, vt::Vector{Float64}, vq::Vector{Float64},
                     dtl::Vector{Float64}, dqw::Vector{Float64}, dtv::Vector{Float64},
                     gm::Vector{Float64}, gh::Vector{Float64},
                     sm::Vector{Float64}, sh::Vector{Float64},
                     c::MYNNConstants;
                     gtr_k::Union{Nothing,Vector{Float64}} = nothing)
    # -- :1753-1765: recomputed inside the loop under CKmod, hence dead. Kept verbatim.
    rfc = MYNN_G1/(MYNN_G1 + MYNN_G2)
    f1  = MYNN_B1*(MYNN_G1 - MYNN_C1) + 3.0*MYNN_A2*(1.0 - MYNN_C2)*(1.0 - MYNN_C5) +
          2.0*MYNN_A1*(3.0 - 2.0*MYNN_C2)
    f2  = MYNN_B1*(MYNN_G1 + MYNN_G2) - 3.0*MYNN_A1*(1.0 - MYNN_C2)
    rf1 = MYNN_B1*(MYNN_G1 - MYNN_C1)/f1
    rf2 = MYNN_B1*MYNN_G1/f2
    smc = MYNN_A1/MYNN_A2*f1/f2
    shc = 3.0*MYNN_A2*(MYNN_G1 + MYNN_G2)

    ri1 = 0.5/smc
    ri2 = rf1*smc
    ri3 = 4.0*rf2*smc - 2.0*ri2
    ri4 = ri2^2
    # -- end dead block

    @inbounds for k in (kts+1):kte
        dzk = 0.5  *(dz[k] + dz[k-1])
        afk = dz[k]/(dz[k] + dz[k-1])
        abk = 1.0 - afk
        duz = (u[k] - u[k-1])^2 + (v[k] - v[k-1])^2
        duz = duz                     /dzk^2
        dtz = (thl[k] - thl[k-1])/(dzk)
        dqz = (qw[k]  - qw[k-1] )/(dzk)

        vtt = 1.0    + vt[k]*abk + vt[k-1]*afk   # Beta-theta in NN09, Eq. 39
        vqq = c.tv0  + vq[k]*abk + vq[k-1]*afk   # Beta-q
        dtq = vtt*dtz + vqq*dqz

        dtl[k] = dtz
        dqw[k] = dqz
        dtv[k] = dtq

        gm[k] =  duz
        gh[k] = -dtq*_gtr_face(gtr_k, c, dz, k)

        #   **  Gradient Richardson number  **
        ri = -gh[k]/max(duz, 1.0e-10)

        # a2fac is needed for the Canuto/Kitamura mod
        a2fac = if MYNN_CKMOD == 1
            1.0/(1.0 + max(ri, 0.0))
        else
            1.0
        end

        rfc = MYNN_G1/(MYNN_G1 + MYNN_G2)
        f1  = MYNN_B1*(MYNN_G1 - MYNN_C1) +
              3.0*MYNN_A2*a2fac*(1.0 - MYNN_C2)*(1.0 - MYNN_C5) +
              2.0*MYNN_A1*(3.0 - 2.0*MYNN_C2)
        f2  = MYNN_B1*(MYNN_G1 + MYNN_G2) - 3.0*MYNN_A1*(1.0 - MYNN_C2)
        rf1 = MYNN_B1*(MYNN_G1 - MYNN_C1)/f1
        rf2 = MYNN_B1*MYNN_G1/f2
        smc = MYNN_A1/(MYNN_A2*a2fac)*f1/f2
        shc = 3.0*(MYNN_A2*a2fac)*(MYNN_G1 + MYNN_G2)

        ri1 = 0.5/smc
        ri2 = rf1*smc
        ri3 = 4.0*rf2*smc - 2.0*ri2
        ri4 = ri2^2

        #   **  Flux Richardson number  **
        rf = min(ri1*(ri + ri2 - sqrt(ri^2 - ri3*ri + ri4)), rfc)

        sh[k] = shc*(rfc - rf)/(1.0 - rf)
        sm[k] = smc*(rf1 - rf)/(rf2 - rf) * sh[k]
    end
    return nothing
end

# ── BouLac length at one level (:2259-2407) ──────────────────────────────────

"""
    boulac_length0!(k, kts, kte, zw, dz, qtke, theta, c::MYNNConstants) -> (lb1, lb2)

Bougeault-Lacarrere length scales at level `k`: `lb1` is `min(dlu, dld)` and `lb2` is
`sqrt(dlu*dld)`, where `dlu`/`dld` are the distances a parcel with TKE `qtke(k)` can
travel up/down against the stratification of `theta`. Fortran: `SUBROUTINE
boulac_length0` (:2259-2407).

Not called by the CASE-2 mixing length — CASE 1 calls the whole-column `boulac_length`
(:2411-…), and CASE 2's only use of a BouLac length is commented out (:2185-2190). It
is ported here because part 2 and a CASE-1 port both need it and because the
`found`-flag `DO WHILE` structure is the easiest thing in the file to get subtly wrong.

Two Fortran details preserved: the exact-equality guards `bbb .ne. 0.` and
`theta(izz) .ne. theta(k)` (floating-point equality on purpose — a uniform layer must
take the linear branch), and the `k == kte` short-circuit that zeroes BOTH outputs
AFTER they have been computed.
"""
function boulac_length0!(k::Int, kts::Int, kte::Int,
                         zw::Vector{Float64}, dz::Vector{Float64},
                         qtke::Vector{Float64}, theta::Vector{Float64},
                         c::MYNNConstants)
    #----------------------------------
    # FIND DISTANCE UPWARD
    #----------------------------------
    zup = 0.0
    dlu = zw[kte+1] - zw[k] - dz[k]*0.5
    zzz = 0.0
    zup_inf = 0.0
    beta = c.gtr           # Buoyancy coefficient (g/tref)

    if k < kte             # can't integrate upwards from the highest level
        found = 0
        izz = k
        while found == 0
            if izz < kte
                dzt = dz[izz]                            # layer depth above
                zup = zup - beta*theta[k]*dzt            # initial PE the parcel has at k
                zup = zup + beta*(theta[izz+1] + theta[izz])*dzt*0.5
                zzz = zzz + dzt                          # depth of layer k to izz+1
                if qtke[k] < zup && qtke[k] >= zup_inf
                    bbb = (theta[izz+1] - theta[izz])/dzt
                    if bbb != 0.0
                        # fractional distance up into the layer where TKE becomes < PE
                        tl = (-beta*(theta[izz] - theta[k]) +
                              sqrt(max(0.0, (beta*(theta[izz] - theta[k]))^2 +
                                            2.0*bbb*beta*(qtke[k] - zup_inf))))/bbb/beta
                    else
                        if theta[izz] != theta[k]
                            tl = (qtke[k] - zup_inf)/(beta*(theta[izz] - theta[k]))
                        else
                            tl = 0.0
                        end
                    end
                    dlu = zzz - dzt + tl
                    found = 1
                end
                zup_inf = zup
                izz = izz + 1
            else
                found = 1
            end
        end
    end

    #----------------------------------
    # FIND DISTANCE DOWN
    #----------------------------------
    zdo = 0.0
    zdo_sup = 0.0
    dld = zw[k]
    zzz = 0.0

    if k > kts             # can't integrate downwards from the lowest level
        found = 0
        izz = k
        while found == 0
            if izz > kts
                dzt = dz[izz-1]
                zdo = zdo + beta*theta[k]*dzt
                zdo = zdo - beta*(theta[izz-1] + theta[izz])*dzt*0.5
                zzz = zzz + dzt
                if qtke[k] < zdo && qtke[k] >= zdo_sup
                    bbb = (theta[izz] - theta[izz-1])/dzt
                    if bbb != 0.0
                        tl = (beta*(theta[izz] - theta[k]) +
                              sqrt(max(0.0, (beta*(theta[izz] - theta[k]))^2 +
                                            2.0*bbb*beta*(qtke[k] - zdo_sup))))/bbb/beta
                    else
                        if theta[izz] != theta[k]
                            tl = (qtke[k] - zdo_sup)/(beta*(theta[izz] - theta[k]))
                        else
                            tl = 0.0
                        end
                    end
                    dld = zzz - dzt + tl
                    found = 1
                end
                zdo_sup = zdo
                izz = izz - 1
            else
                found = 1
            end
        end
    end

    #----------------------------------
    # GET MINIMUM (OR AVERAGE)
    #----------------------------------
    # The surface-layer length scale can exceed z for large z/L, so keep the maximum
    # distance down > z.
    dld = min(dld, zw[k+1])   # not used in the PBL anyway, only the free atmosphere
    lb1 = min(dlu, dld)       # minimum
    # JOE-fight floating point errors
    dlu = max(0.1, min(dlu, 1000.0))
    dld = max(0.1, min(dld, 1000.0))
    lb2 = sqrt(dlu*dld)       # average - biased towards the smallest

    if k == kte
        lb1 = 0.0
        lb2 = 0.0
    end

    return (lb1, lb2)
end

# ── Master mixing length, CASE 2 (:2101-2244) ────────────────────────────────

"""
    mym_length!(kts, kte, xland, dz, dx, zw, rmo, flt, fltv, flq, vt, vq,
                u1, v1, qke, dtv, el, zi, theta, qkw, Psig_bl, cldfra_bl1D,
                bl_mynn_mixlength, edmf_w1, edmf_a1, c::MYNNConstants, work::MYNNWork)

Master length scale `el` on the box walls, CASE 2 of `SUBROUTINE mym_length`
(:2101-2244) — the "local (mostly)" formulation, `bl_mynn_mixlength = 2`, which is
what mynn_bl_driver runs and what the reference driver dumps. CASE 0 and CASE 1 are
OUT OF SCOPE and requesting them raises.

`qkw` is an `intent(out)` of the Fortran, and `mym_initialize` reads it after the call
(:1615-1620) — so it is a genuine second output, not scratch, and is written here for
`kts:kte` on every call. `el` is `intent(out)` as well; `el(kts)` is set to 0 (:2155).

Inputs `vt`, `vq` and `xland` are accepted and unused in CASE 2: `vflx` is taken
straight from `fltv` (:2151, the `(vt+1)*flt + (vq+tv0)*flq` form is commented out on
:2150) and `xland` is a CASE-0/1-era argument. `theta` is likewise unused in CASE 2
(only CASE 1 builds `thetaw` from it). They stay in the signature so the call site
matches the Fortran.

The keyword `gtr_k` is the `:gtr_local` fidelity deviation: `nothing` (the default) is
the Fortran's `c.gtr`; a per-level `g/theta_v` column makes `vsc` and `wstar` use the
SURFACE value and `bv` the interface one ([`_gtr_lev`](@ref), [`_gtr_face`](@ref)).

Dead-but-transcribed, marked in place: `Uonset`/`Ugrid` (:2103-2104 — the `cns` taper
that would use them is commented out on :2105) and `cldavg` (:2161, computed per level
and never read in this branch).
"""
function mym_length!(kts::Int, kte::Int, xland::Float64,
                     dz::Vector{Float64}, dx::Float64, zw::Vector{Float64},
                     rmo::Float64, flt::Float64, fltv::Float64, flq::Float64,
                     vt::Vector{Float64}, vq::Vector{Float64},
                     u1::Vector{Float64}, v1::Vector{Float64}, qke::Vector{Float64},
                     dtv::Vector{Float64}, el::Vector{Float64},
                     zi::Float64, theta::Vector{Float64}, qkw::Vector{Float64},
                     Psig_bl::Float64, cldfra_bl1D::Vector{Float64},
                     bl_mynn_mixlength::Int,
                     edmf_w1::Vector{Float64}, edmf_a1::Vector{Float64},
                     c::MYNNConstants, work::MYNNWork;
                     gtr_k::Union{Nothing,Vector{Float64}} = nothing)
    bl_mynn_mixlength == 2 ||
        throw(ArgumentError("mym_length!: only CASE 2 (bl_mynn_mixlength = 2) is " *
                            "ported; got $(bl_mynn_mixlength). CASE 0 and CASE 1 of " *
                            "module_bl_mynn.F90 :1922-2099 are out of scope."))

    qtke = work.qtke

    # CASE (2) !Local (mostly) mixing length formulation                    :2101
    Uonset = 3.5 + dz[kts]*0.1                                            # :2103 (dead)
    Ugrid  = sqrt(u1[kts]^2 + v1[kts]^2)                                  # :2104 (dead)
    cns  = MYNN_CNS   # :2105  the "JOE-test" Ugrid/Uonset taper is commented out
    alp1 = MYNN_ALP1
    alp2 = MYNN_ALP2
    alp3 = MYNN_ALP3
    alp4 = MYNN_ALP4
    alp5 = MYNN_ALP5  # = alp2, "like alp2, but for free atmosphere"
    alp6 = MYNN_ALP6  # used for the MF mixing length

    # Limits on the height integration for elt and the transition-layer depth. The
    # Fortran spells 300./600. as literals here (the minzi/mindz/maxdz parameter forms
    # are commented out on :2114, :2116-2117), so the literals are what is written.
    zi2 = max(zi,      300.0)                                             # :2115
    h1  = max(0.3*zi2, 300.0)                                             # :2118
    h1  = min(h1, 600.0)                                                  # :2119
    h2  = h1*0.5                                                          # :2120  1/4 depth

    @inbounds begin
        qtke[kts] = max(0.5*qke[kts], 0.5*MYNN_QKEMIN)  # tke at full sigma levels
        qkw[kts]  = sqrt(max(qke[kts], MYNN_QKEMIN))

        for k in (kts+1):kte
            afk = dz[k]/(dz[k] + dz[k-1])
            abk = 1.0 - afk
            qkw[k]  = sqrt(max(qke[k]*abk + qke[k-1]*afk, MYNN_QKEMIN))
            qtke[k] = 0.5*qkw[k]^2   # qkw -> TKE
        end

        elt = 1.0e-5
        vsc = 1.0e-5

        #   **  Strictly, zwk*h(i,j) -> ( zwk*h(i,j)+z0 )  **
        PBLH_PLUS_ENT = max(zi + h1, 100.0)
        k = kts + 1
        zwk = zw[k]
        while zwk <= PBLH_PLUS_ENT
            dzk = 0.5*(dz[k] + dz[k-1])
            qdz = min(max(qkw[k] - MYNN_QMIN, 0.03), 30.0)*dzk
            elt = elt + qdz*zwk
            vsc = vsc + qdz
            k   = k + 1
            zwk = zw[k]
        end

        elt = min(max(alp1*elt/vsc, 10.0), 400.0)
        # avoid the buoyancy-flux functions, ill-defined at the surface (:2149-2151)
        vflx = fltv
        vsc  = (_gtr_lev(gtr_k, c, kts)*elt*max(vflx, 0.0))^MYNN_ONETHIRD

        #   **  Strictly, el(i,j,1) is not zero.  **
        el[kts] = 0.0                                                     # :2155
        zwk1    = zw[kts+1]                                               # :2156 (dead)

        for k in (kts+1):kte
            zwk = zw[k]                      # full-sigma levels
            dzk = 0.5*(dz[k] + dz[k-1])
            cldavg = 0.5*(cldfra_bl1D[k-1] + cldfra_bl1D[k])              # :2161 (dead)

            #   **  Length scale limited by the buoyancy effect  **
            local elb::Float64, elf::Float64, elb_mf::Float64
            if dtv[k] > 0.0
                # impose a min value on bv
                bv = max(sqrt(_gtr_face(gtr_k, c, dz, k)*dtv[k]), 0.001)
                elb_mf = max(alp2*qkw[k],
                             alp6*edmf_a1[k-1]*edmf_w1[k-1]) / bv *
                         (1.0 + alp3*sqrt(vsc/(bv*elt)))
                elb = min(max(alp5*qkw[k], alp6*edmf_a1[k]*edmf_w1[k])/bv, zwk)

                wstar = 1.25*(_gtr_lev(gtr_k, c, kts)*zi*max(vflx, 1.0e-4))^MYNN_ONETHIRD
                tau_cloud = min(max(MYNN_CTAU * wstar/c.grav, 30.0), 150.0)
                # minimize the influence of the surface heat flux far from the PBLH
                wt = 0.5*tanh((zwk - (zi2 + h1))/h2) + 0.5
                tau_cloud = tau_cloud*(1.0 - wt) + 50.0*wt
                elf = min(max(tau_cloud*sqrt(min(qtke[k], 40.0)),
                              alp6*edmf_a1[k]*edmf_w1[k]/bv), zwk)
            else
                # tau_cloud is an eddy turnover timescale; Teixeira and Cheinet (2004)
                # Eq. 1, Cheinet and Teixeira (2003) Eq. 7.
                wstar = 1.25*(_gtr_lev(gtr_k, c, kts)*zi*max(vflx, 1.0e-4))^MYNN_ONETHIRD
                tau_cloud = min(max(MYNN_CTAU * wstar/c.grav, 50.0), 200.0)
                wt = 0.5*tanh((zwk - (zi2 + h1))/h2) + 0.5
                tau_cloud = tau_cloud*(1.0 - wt) + max(100.0, dzk*0.25)*wt

                elb = min(tau_cloud*sqrt(min(qtke[k], 40.0)), zwk)
                elf = elb
                elb_mf = elb
            end
            elf    = elf/(1.0 + (elf/800.0))  # bound the free-atmos length to < 800 m
            elb_mf = max(elb_mf, 0.01)        # to avoid divide-by-zero below

            #   **  Length scale in the surface layer  **
            if rmo > 0.0
                els = c.karman*zwk/(1.0 + cns*min(zwk*rmo, MYNN_ZMAX))
            else
                els = c.karman*zwk*(1.0 - alp4*zwk*rmo)^0.2
            end

            #   ** NOW BLEND THE MIXING LENGTH SCALES:
            wt = 0.5*tanh((zwk - (zi2 + h1))/h2) + 0.5

            # try squared-blending
            el[k] = sqrt(els^2/(1.0 + (els^2/elt^2) + (els^2/elb_mf^2)))
            el[k] = el[k]*(1.0 - wt) + elf*wt

            # scale-awareness: simple asymptotic kz -> 12 m (should be ~dz)
            el_les = min(els/(1.0 + (els/12.0)), elb_mf)
            el[k]  = el[k]*Psig_bl + (1.0 - Psig_bl)*el_les
        end
    end
    return nothing
end

# ── Diagnostic PBL height (:5518-5658) ───────────────────────────────────────

"""
    get_pblh!(kts, kte, thetav1D, qke1D, zw1D, dz1D, landsea) -> (zi, kzi)

Hybrid diagnostic PBL height. The 1.5-theta-increase method (Nielsen-Gammon et al.
2008) blended by `tanh` with a TKE-threshold method (Banta and Pichugina 2008), the
latter weighted up in stable, shallow layers. Fortran: `SUBROUTINE GET_PBLH`
(:5518-5658).

Returns the height `zi` (m) and the level index `kzi` (`kpbl`), the two `intent(out)`
scalars of the Fortran, as a tuple. Nothing is written, so the trailing `!` is a slight
misnomer kept for symmetry with the rest of the file; the routine allocates nothing.

`landsea` is the wrapper's `xland`: `>= 1.5` is water (`delt_thv = 1.0`), below is land
(1.25). `zw1D` must be length `kte+1`.

Two Fortran shapes that a "cleanup" would break: the first `DO WHILE (zw1D(k) <= 200.)`
has NO upper guard on `k` (it relies on `zw` crossing 200 m within the column — true for
every Scythe mish), and both search loops carry an `IF (k .EQ. kte-1)` safeguard that
FORCES the answer to `zw1D(kts+1)` at the last level whether or not the threshold was
ever met.
"""
function get_pblh!(kts::Int, kte::Int,
                   thetav1D::Vector{Float64}, qke1D::Vector{Float64},
                   zw1D::Vector{Float64}, dz1D::Vector{Float64},
                   landsea::Float64)
    sbl_lim  = 200.0   # upper limit of stable BL height (m)
    sbl_damp = 400.0   # transition length for blending (m)

    # Initialize KPBL (kzi)
    kzi = 2

    @inbounds begin
        #> - FIND MIN THETAV IN THE LOWEST 200 M AGL
        k = kts + 1
        kthv = 1
        minthv = 9.0e9
        while zw1D[k] <= 200.0
            if minthv > thetav1D[k]
                minthv = thetav1D[k]
                kthv = k
            end
            k = k + 1
        end

        #> - FIND THETAV-BASED PBLH (BEST FOR DAYTIME).
        zi = 0.0
        delt_thv = if (landsea - 1.5) >= 0
            1.0    # WATER
        else
            1.25   # LAND
        end

        zi = 0.0
        for k in (kts+1):(kte-1)
            if thetav1D[k] >= (minthv + delt_thv)
                zi = zw1D[k] - dz1D[k-1]*
                     min((thetav1D[k] - (minthv + delt_thv))/
                         max(thetav1D[k] - thetav1D[k-1], 1e-6), 1.0)
            end
            if k == kte - 1
                zi = zw1D[kts+1]   # EXIT SAFEGUARD
            end
            if zi != 0.0
                break
            end
        end

        #> - FOR STABLE BOUNDARY LAYERS, USE THE TKE METHOD TO COMPLEMENT THE
        #!   THETAV-BASED DEFINITION.
        ktke = 1
        maxqke = max(qke1D[kts], 0.0)
        # Use 5% of the tke max (Kosovic and Curry, 2000; JAS): TKEeps = maxqke/40.
        TKEeps = maxqke/40.0
        TKEeps = max(TKEeps, 0.02)
        PBLH_TKE = 0.0

        for k in (kts+1):(kte-1)
            # QKE CAN BE NEGATIVE (IF CKmod == 0)... MAKE TKE NON-NEGATIVE.
            qtke   = max(qke1D[k]/2.0,   0.0)
            qtkem1 = max(qke1D[k-1]/2.0, 0.0)
            if qtke <= TKEeps
                PBLH_TKE = zw1D[k] - dz1D[k-1]*
                           min((TKEeps - qtke)/max(qtkem1 - qtke, 1e-6), 1.0)
                # IN CASE OF NEAR ZERO TKE, SET PBLH = LOWEST LEVEL.
                PBLH_TKE = max(PBLH_TKE, zw1D[kts+1])
            end
            if k == kte - 1
                PBLH_TKE = zw1D[kts+1]   # EXIT SAFEGUARD
            end
            if PBLH_TKE != 0.0
                break
            end
        end

        #> - Cap the TKE-based PBLH to the theta_v one +/- 350 m.
        PBLH_TKE = min(PBLH_TKE, zi + 350.0)
        PBLH_TKE = max(PBLH_TKE, max(zi - 350.0, 10.0))

        wt = 0.5*tanh((zi - sbl_lim)/sbl_damp) + 0.5
        if maxqke <= 0.05
            # Cold pool situation - default to the theta_v-based definition
        else
            # BLEND THE TWO PBLH TYPES HERE:
            zi = PBLH_TKE*(1.0 - wt) + zi*wt
        end

        # Compute KPBL (kzi)
        for k in (kts+1):(kte-1)
            if zw1D[k] >= zi
                kzi = k - 1
                break
            end
        end

        return (zi, kzi)
    end
end

# ── Scale-awareness (:7314-7385) ─────────────────────────────────────────────

"""
    scale_aware(dx, PBL1) -> (Psig_bl, Psig_shcu)

Honnert et al. (2011) / Shin and Hong (2013) similarity factors that taper the local
(`Psig_bl`) and non-local (`Psig_shcu`) mixing as the grid spacing `dx` approaches the
PBL depth `PBL1`. Fortran: `SUBROUTINE SCALE_AWARE` (:7314-7385).

`Psig_shcu` uses `min(PBL1+500, 3500)` — a 500 m assumed shallow-cumulus depth — where
`Psig_bl` uses `min(PBL1, 3000)`. Both are clipped to [0, 1] at the end.
"""
function scale_aware(dx::Float64, PBL1::Float64)
    Psig_bl   = 1.0
    Psig_shcu = 1.0

    dxdh = max(2.5*dx, 10.0)/min(PBL1, 3000.0)
    # New form to preserve parameterized mixing - only down 5% at dx = 750 m
    Psig_bl = ((dxdh^2) + 0.106*(dxdh^0.667))/((dxdh^2) + 0.066*(dxdh^0.667) + 0.071)

    # assume a 500 m cloud depth for shallow-cu clouds
    dxdh = max(2.5*dx, 10.0)/min(PBL1 + 500.0, 3500.0)
    # Shin and Hong (2013), TKE in the entrainment zone
    Psig_shcu = ((dxdh^2) + 0.145*(dxdh^0.667))/((dxdh^2) + 0.172*(dxdh^0.667) + 0.170)

    if Psig_bl > 1.0;   Psig_bl = 1.0;   end
    if Psig_bl < 0.0;   Psig_bl = 0.0;   end

    if Psig_shcu > 1.0; Psig_shcu = 1.0; end
    if Psig_shcu < 0.0; Psig_shcu = 0.0; end

    return (Psig_bl, Psig_shcu)
end

# ── Initialization (:1515-1675) ──────────────────────────────────────────────

"""
    mym_initialize!(kts, kte, xland, dz, dx, zw, u, v, thl, qw, zi, theta, thetav,
                    sh, sm, ust, rmo, el, qke, tsq, qsq, cov, Psig_bl, cldfra_bl1D,
                    bl_mynn_mixlength, edmf_w1, edmf_a1, INITIALIZE_QKE,
                    c::MYNNConstants, work::MYNNWork)

Cold-start the mixing length `el`, the TKE `qke`, and the variances `tsq`, `qsq`,
`cov` by iterating `mym_length!` five times against the level-2 fluxes. Fortran:
`SUBROUTINE mym_initialize` (:1515-1675).

`sh`, `sm`, `el`, `qke`, `tsq`, `qsq` and `cov` are written in place. `sh`/`sm` come
from `mym_level2!`, so their `kts` element is NOT touched here and keeps whatever the
caller had (the reference driver zeroes them before the call — that is why `init_sh(1)`
and `init_sm(1)` are exactly 0 in the reference output).

FOUR Fortran facts a reader will trip over:

 1. `pmz = 1.`, `phh = 1.`, `flt = flq = fltv = 0.` are LOCALS with initializers
    (:1546-1547), not arguments. The commented-out signature line :1519 and the
    commented-out declaration :1533 show they used to be passed in. So every `flt/ust` below is `0/ust`, and `(b1*pmz)` is just `b1`.
 2. When `ust = 0` (a truly resting column — case 1 of the reference) `flt/ust` is
    `0/0 = NaN`, and `tsq(kts)`, `qsq(kts)`, `cov(kts)` come out NaN (:1580-1582 and again
    :1632-1634). That is REPRODUCED, not guarded: `mym_predict` overwrites all three at
    step 1 so they never reach the state, and the reference blocks carry the NaN. Guard
    it and the port stops matching.
 3. `qke` is `intent(inout)` but with `INITIALIZE_QKE = .true.` its input value is
    irrelevant — :1571 overwrites `qke(kts)` and :1573-1576 re-taper the column from
    it, so the caller's taper is thrown away and rebuilt.
 4. `qkw` is `mym_length`'s second output and is read straight back at :1615-1620; it is not
    scratch. It comes from `work.qkw` here.

`spp_pbl` and `rstoch_col` are arguments of the Fortran that its body never reads, so
they are dropped from this signature.
"""
function mym_initialize!(kts::Int, kte::Int, xland::Float64,
                         dz::Vector{Float64}, dx::Float64, zw::Vector{Float64},
                         u::Vector{Float64}, v::Vector{Float64},
                         thl::Vector{Float64}, qw::Vector{Float64},
                         zi::Float64, theta::Vector{Float64}, thetav::Vector{Float64},
                         sh::Vector{Float64}, sm::Vector{Float64},
                         ust::Float64, rmo::Float64,
                         el::Vector{Float64}, qke::Vector{Float64},
                         tsq::Vector{Float64}, qsq::Vector{Float64},
                         cov::Vector{Float64},
                         Psig_bl::Float64, cldfra_bl1D::Vector{Float64},
                         bl_mynn_mixlength::Int,
                         edmf_w1::Vector{Float64}, edmf_a1::Vector{Float64},
                         INITIALIZE_QKE::Bool,
                         c::MYNNConstants, work::MYNNWork)
    ql  = work.ql;  vt  = work.vt;  vq  = work.vq
    pdk = work.pdk; pdt = work.pdt; pdq = work.pdq; pdc = work.pdc
    dtl = work.dtl; dqw = work.dqw; dtv = work.dtv
    gm  = work.gm;  gh  = work.gh;  qkw = work.qkw

    # locals with initializers, :1537-1538
    pmz = 1.0
    phh = 1.0
    flt = 0.0
    fltv = 0.0
    flq = 0.0

    @inbounds begin
        #> - At first ql, vt and vq are set to zero.
        for k in kts:kte
            ql[k] = 0.0
            vt[k] = 0.0
            vq[k] = 0.0
        end

        #> - Call mym_level2() to calculate the stability functions at level 2.
        mym_level2!(kts, kte, dz, u, v, thl, thetav, qw, ql, vt, vq,
                    dtl, dqw, dtv, gm, gh, sm, sh, c)

        #   **  Preliminary setting  **
        el[kts] = 0.0
        if INITIALIZE_QKE
            qke[kts] = 1.5 * ust^2 * (MYNN_B1*pmz)^(2.0/3.0)
            for k in (kts+1):kte
                # linearly taper off towards the top of the pbl
                qke[k] = qke[kts]*max((ust*700.0 - zw[k])/(max(ust, 0.01)*700.0), 0.01)
            end
        end

        phm      = phh*MYNN_B2 / (MYNN_B1*pmz)^(1.0/3.0)
        tsq[kts] = phm*(flt/ust)^2
        qsq[kts] = phm*(flq/ust)^2
        cov[kts] = phm*(flt/ust)*(flq/ust)

        for k in (kts+1):kte
            vkz = c.karman*zw[k]
            el[k] = vkz/(1.0 + vkz/100.0)

            tsq[k] = 0.0
            qsq[k] = 0.0
            cov[k] = 0.0
        end

        #   **  Initialization with an iterative manner          **
        #   **  lmax is the iteration count. This is arbitrary.  **
        lmax = 5

        for l in 1:lmax
            #> - call mym_length() to calculate the master length scale.
            mym_length!(kts, kte, xland, dz, dx, zw, rmo, flt, fltv, flq, vt, vq,
                        u, v, qke, dtv, el, zi, theta, qkw, Psig_bl, cldfra_bl1D,
                        bl_mynn_mixlength, edmf_w1, edmf_a1, c, work)

            for k in (kts+1):kte
                elq = el[k]*qkw[k]
                pdk[k] = elq*(sm[k]*gm[k] + sh[k]*gh[k])
                pdt[k] = elq* sh[k]*dtl[k]^2
                pdq[k] = elq* sh[k]*dqw[k]^2
                pdc[k] = elq* sh[k]*dtl[k]*dqw[k]
            end

            #   **  Strictly, vkz*h(i,j) -> karman*( 0.5*dz(1)*h(i,j)+z0 )  **
            vkz = c.karman*0.5*dz[kts]
            elv = 0.5*(el[kts+1] + el[kts]) / vkz
            if INITIALIZE_QKE
                qke[kts] = 1.0 * max(ust, 0.02)^2 * (MYNN_B1*pmz*elv)^(2.0/3.0)
            end

            phm      = phh*MYNN_B2 / (MYNN_B1*pmz/elv^2)^(1.0/3.0)
            tsq[kts] = phm*(flt/ust)^2
            qsq[kts] = phm*(flq/ust)^2
            cov[kts] = phm*(flt/ust)*(flq/ust)

            for k in (kts+1):(kte-1)
                b1l = MYNN_B1*0.25*(el[k+1] + el[k])
                # add MIN to limit unreasonable QKE
                tmpq = min(max(b1l*(pdk[k+1] + pdk[k]), MYNN_QKEMIN), 125.0)
                if INITIALIZE_QKE
                    qke[k] = tmpq^MYNN_TWOTHIRDS
                end

                b2l = if qke[k] <= 0.0
                    0.0
                else
                    MYNN_B2*(b1l/MYNN_B1) / sqrt(qke[k])
                end

                tsq[k] = b2l*(pdt[k+1] + pdt[k])
                qsq[k] = b2l*(pdq[k+1] + pdq[k])
                cov[k] = b2l*(pdc[k+1] + pdc[k])
            end
        end

        if INITIALIZE_QKE
            qke[kts] = 0.5*(qke[kts] + qke[kts+1])
            qke[kte] = qke[kte-1]
        end
        tsq[kte] = tsq[kte-1]
        qsq[kte] = qsq[kte-1]
        cov[kte] = cov[kte-1]
    end
    return nothing
end

# ══════════════════════════════════════════════════════════════════════════════
#  PART 2 — turbulence, prediction, condensation, tendencies and the column step
# ══════════════════════════════════════════════════════════════════════════════
#
# Same rules as part 1: the operation order and the association inside every
# expression are the Fortran's, `x^2` is `x*x` (integer exponent), `x^0.19` is a `pow`,
# no `muladd`/`@fastmath`, every `!` routine is allocation-free on a warm call with its
# automatic arrays taken from `MYNNWork`.
#
# WHAT IS HERE: mym_turbulence!, mym_predict!, mym_condensation!, mynn_tendencies!,
# moisture_check!, retrieve_exchange_coeffs!, and the two column drivers
# mynn_init_column! / mynn_column_step! that replay the reference driver's mode-B
# sequence. The mass-flux plumes (DMP_mf) are still NOT ported: `mynn_column_step!`
# takes `edmf = false` and leaves every plume sum zero.
#
# SCOPE GATES. Level 3 (`closure >= 3.0`) raises in both mym_turbulence! and
# mym_predict!; only `bl_mynn_cloudpdf = ±2` is ported in mym_condensation!; and
# mynn_tendencies! implements the argument set the reference exercises (FLAG_QC =
# FLAG_QI = true, every other species flag false, `bl_mynn_mixqt = 0`) and raises on
# the rest. Each gate names the Fortran lines it is refusing to run.
#
# ── THE ONE KNOWN DIVERGENCE FROM THE REFERENCE: max() with a NaN ─────────────
#
# `mym_initialize` produces a genuine NaN in `tsq(kts)`, `qsq(kts)`, `cov(kts)` when
# `ust = 0` (`phm*(flt/ust)**2` is 0/0 — harness README item 4, reproduced on purpose
# by part 1). Two `MAX` calls downstream then see that NaN:
#
#     mym_condensation (:3849)  r3sq = max( qsq(k), 0.0 )
#     mym_predict      (:3427)  qsq(k) = MAX( x(k), 1e-17 )   [closure > 2.5 only]
#
# The Fortran standard leaves `MAX` undefined for a NaN argument. The reference build
# (gfortran 15.2, arm64, -O0) returns the OTHER argument — 0.0 and 1e-17 — so the NaN
# is silently quenched and the Fortran column stays finite. Julia's `max` propagates
# the NaN, as IEEE-754 `maximum` does, and this port keeps Julia's semantics: a 0/0 in
# the variances is a real defect of a resting column and should be visible, not
# swallowed inside a clip whose only intent is to remove negative variances.
#
# CONSEQUENCE, and it is confined to `case1_rest` (the only reference column with
# `ust = 0`): at closure 2.5 `mym_condensation!` returns NaN in `vt/vq/sgm/qc_bl/
# qi_bl/cldfra_bl` at k = 1 where the Fortran returns finite values (every other level
# is bitwise identical), and at closure 2.6 `mym_predict!` returns an all-NaN `qsq`
# where the Fortran returns an all-1e-17 one. Because `mym_level2!` averages `vt`/`vq`
# across walls, that single NaN then spreads up the column on the next step, so the
# 30-step end-to-end comparison is not run on case 1. test/test_mynn_closure.jl pins
# all of this in an explicit "known divergence" testset rather than hiding it in a
# tolerance. Any physical column has `ust > 0` and is unaffected.

# ── Level 2.5 turbulence closure (:2619-3149) ────────────────────────────────

"""
    mym_turbulence!(kts, kte, xland, closure, dz, dx, zw, u, v, thl, thetav, ql, qw,
                    qke, tsq, qsq, cov, vt, vq, rmo, flt, fltv, flq, zi, theta,
                    sh, sm, el, dfm, dfh, dfq, tcd, qcd, pdk, pdt, pdq, pdc,
                    qWT1D, qSHEAR1D, qBUOY1D, qDISS1D, tke_budget,
                    Psig_bl, Psig_shcu, cldfra_bl1D, bl_mynn_mixlength,
                    edmf_w1, edmf_a1, TKEprodTD, spp_pbl, rstoch_col,
                    c::MYNNConstants, work::MYNNWork)

The Level-2.5 stability functions, the master length scale, the production terms and
the eddy diffusivities. Fortran: `SUBROUTINE mym_turbulence` (:2619-3149).

Calls `mym_level2!` and then `mym_length!`, so `sh`, `sm` and `el` are OUTPUTS as well
as (for `sh`/`sm` at `kts`) untouched inputs — the interface loops all start at
`kts+1`, exactly as in part 1. `dfm`, `dfh`, `dfq`, `tcd`, `qcd`, `pdk`, `pdt`, `pdq`
and `pdc` are `intent(out)` of the Fortran and are fully written here.

`closure >= 3.0` RAISES: the Level-3 block (:2898-3037) is out of scope. Nothing on
the ≤ 2.6 path depends on it — `gamt`, `gamq`, `gamv` are set to zero in the `ELSE`
(:3041-3043) and `qdiv` keeps its Helfand–Labraga value because the `qdiv = 1.0` reset
at :3023 is inside the Level-3 branch. At closure 2.6 the Level-3 branch is not taken
either, so 2.5 and 2.6 give bitwise identical turbulence.

Note also that at closure ≤ 2.6 `tsq`, `qsq` and `cov` are read ONLY inside the
Level-3 block, i.e. never: they are accepted so the call site matches the Fortran.

The keyword `gtr_k` (the `:gtr_local` fidelity deviation) is forwarded unchanged to both
`mym_level2!` and `mym_length!` and is used nowhere else in this routine.

Fortran shapes preserved that a cleanup would break:

  * `q2sq` (:2731) is formed from `sh(k)`/`sm(k)` BEFORE the `max(·, 1e-5)` floor on
    :2733-2735; `sm(k)` itself is never floored, only `sh(k)`.
  * `sh20`/`sm20` (:2733-2734), `Prnum` (:2754) and `sm25max`..`sh25min` (:2857-2860)
    are computed and then unused — every line that would read them is commented out.
    Transcribed, marked, because they document the intended limiters.
  * `tcd(kte)`/`qcd(kte)` are zeroed (:3126-3127) BEFORE the difference loop
    (:3130-3134), which then overwrites `tcd(k)` in ASCENDING `k` while reading the
    still-original `tcd(k+1)`.
  * the `tke_budget` arrays are written only when `tke_budget == 1`; with the
    reference's `tke_budget = 0` they are accepted and left untouched.
"""
function mym_turbulence!(kts::Int, kte::Int, xland::Float64, closure::Float64,
                         dz::Vector{Float64}, dx::Float64, zw::Vector{Float64},
                         u::Vector{Float64}, v::Vector{Float64},
                         thl::Vector{Float64}, thetav::Vector{Float64},
                         ql::Vector{Float64}, qw::Vector{Float64},
                         qke::Vector{Float64}, tsq::Vector{Float64},
                         qsq::Vector{Float64}, cov::Vector{Float64},
                         vt::Vector{Float64}, vq::Vector{Float64},
                         rmo::Float64, flt::Float64, fltv::Float64, flq::Float64,
                         zi::Float64, theta::Vector{Float64},
                         sh::Vector{Float64}, sm::Vector{Float64}, el::Vector{Float64},
                         dfm::Vector{Float64}, dfh::Vector{Float64}, dfq::Vector{Float64},
                         tcd::Vector{Float64}, qcd::Vector{Float64},
                         pdk::Vector{Float64}, pdt::Vector{Float64},
                         pdq::Vector{Float64}, pdc::Vector{Float64},
                         qWT1D::Vector{Float64}, qSHEAR1D::Vector{Float64},
                         qBUOY1D::Vector{Float64}, qDISS1D::Vector{Float64},
                         tke_budget::Int,
                         Psig_bl::Float64, Psig_shcu::Float64,
                         cldfra_bl1D::Vector{Float64}, bl_mynn_mixlength::Int,
                         edmf_w1::Vector{Float64}, edmf_a1::Vector{Float64},
                         TKEprodTD::Vector{Float64},
                         spp_pbl::Int, rstoch_col::Vector{Float64},
                         c::MYNNConstants, work::MYNNWork;
                         gtr_k::Union{Nothing,Vector{Float64}} = nothing)
    closure < 3.0 ||
        throw(ArgumentError("mym_turbulence!: closure = $(closure); the Level-3 " *
                            "branches (module_bl_mynn.F90 :2898-3037) are out of " *
                            "scope. Only closure <= 2.6 is ported."))

    Prlimit = 5.0                                                        # :2690

    dtl = work.dtl; dqw = work.dqw; dtv = work.dtv
    gm  = work.gm;  gh  = work.gh;  qkw = work.qkw

    mym_level2!(kts, kte, dz, u, v, thl, thetav, qw, ql, vt, vq,
                dtl, dqw, dtv, gm, gh, sm, sh, c; gtr_k = gtr_k)         # :2705

    mym_length!(kts, kte, xland, dz, dx, zw, rmo, flt, fltv, flq, vt, vq,
                u, v, qke, dtv, el, zi, theta, qkw, Psig_bl, cldfra_bl1D,
                bl_mynn_mixlength, edmf_w1, edmf_a1, c, work;
                gtr_k = gtr_k)                                           # :2711

    @inbounds for k in (kts+1):kte
        dzk  = 0.5  *(dz[k] + dz[k-1])
        afk  = dz[k]/(dz[k] + dz[k-1])
        abk  = 1.0 - afk
        elsq = el[k]^2
        q3sq = qkw[k]^2
        q2sq = MYNN_B1*elsq*(sm[k]*gm[k] + sh[k]*gh[k])

        sh20  = max(sh[k], 1e-5)     # :2733 (dead: sh25max/min are literals below)
        sm20  = max(sm[k], 1e-5)     # :2734 (dead)
        sh[k] = max(sh[k], 1e-5)

        # Canuto/Kitamura mod
        duz = (u[k] - u[k-1])^2 + (v[k] - v[k-1])^2
        duz = duz                    /dzk^2
        #   **  Gradient Richardson number  **
        ri = -gh[k]/max(duz, 1.0e-10)
        a2fac = if MYNN_CKMOD == 1
            1.0/(1.0 + max(ri, 0.0))
        else
            1.0
        end

        # level 2.0 Prandtl number, Zilitinkevich et al. (2006) modified towards
        # Esau and Grachev (2007). :2754 — computed, then never read (every use is
        # commented out).
        Prnum = min(0.76 + 4.0*max(ri, 0.0), Prlimit)

        # Modified: Dec/22/2005 (dlsq -> elsq)
        gmel = gm[k]*elsq
        ghel = gh[k]*elsq

        #   **  Since qkw is set to more than 0.0, q3sq > 0.0.  **
        #   **  Limitation on q, instead of L/q  **
        dlsq = elsq
        if q3sq/dlsq < -gh[k]
            q3sq = -dlsq*gh[k]
        end

        local e1::Float64, e2::Float64, e3::Float64, e4::Float64
        local eden::Float64, qdiv::Float64
        if q3sq < q2sq
            # Apply the Helfand & Labraga mod
            qdiv = sqrt(q3sq/q2sq)        # HL89: (1-alfa)

            # Use the level 2.0 functions as in the original MYNN
            sh[k] = sh[k] * qdiv
            sm[k] = sm[k] * qdiv

            # Recalculate the terms for later use
            e1   = q3sq - MYNN_E1C*ghel*a2fac      * qdiv^2
            e2   = q3sq - MYNN_E2C*ghel*a2fac      * qdiv^2
            e3   = e1   + MYNN_E3C*ghel*a2fac^2    * qdiv^2
            e4   = e1   - MYNN_E4C*ghel*a2fac      * qdiv^2
            eden = e2*e4 + e3*MYNN_E5C*gmel        * qdiv^2
            eden = max(eden, 1.0e-20)
        else
            e1   = q3sq - MYNN_E1C*ghel*a2fac
            e2   = q3sq - MYNN_E2C*ghel*a2fac
            e3   = e1   + MYNN_E3C*ghel*a2fac^2
            e4   = e1   - MYNN_E4C*ghel*a2fac
            eden = e2*e4 + e3*MYNN_E5C*gmel
            eden = max(eden, 1.0e-20)

            qdiv = 1.0
            # Use the level 2.5 stability functions
            sm[k] = q3sq*MYNN_A1*(e3 - 3.0*MYNN_C1*e4)/eden
            sh[k] = q3sq*(MYNN_A2*a2fac)*(e2 + 3.0*MYNN_C1*MYNN_E5C*gmel)/eden
        end  # end Helfand & Labraga check

        # Impose broad limits on Sh and Sm (:2856-2860). gmelq, sm25min and sh25min
        # are dead: gmelq is only read by the commented-out sm25max, and the two
        # minima are the literal 0.0 that the IFs below compare against.
        gmelq   = max(gmel/q3sq, 1e-8)
        sm25max = 4.0
        sh25max = 4.0
        sm25min = 0.0
        sh25min = 0.0

        # Enforce the constraints for the level 2.5 functions. Only the two `sh`
        # lines are live; the `sm` pair is commented out in the Fortran (:2883-2884).
        if sh[k] > sh25max; sh[k] = sh25max; end
        if sh[k] < sh25min; sh[k] = sh25min; end

        # surface-layer Pr: keep the same Pr limit in the surface layer
        shb   = max(sh[k], 0.002)
        sm[k] = min(sm[k], Prlimit*shb)

        #   **  At Level 2.5, qdiv is not reset.  **                     :3039-3044
        gamt = 0.0
        gamq = 0.0
        gamv = 0.0

        # Add a min background stability function (diffusivity) within model levels
        # with active plumes and clouds.
        cldavg = 0.5*(cldfra_bl1D[k-1] + cldfra_bl1D[k])
        if edmf_a1[k] > 0.001 || cldavg > 0.02
            # for mass-flux columns
            sm[k] = max(sm[k], 0.03*min(10.0*edmf_a1[k]*edmf_w1[k], 1.0))
            sh[k] = max(sh[k], 0.03*min(10.0*edmf_a1[k]*edmf_w1[k], 1.0))
            # for clouds
            sm[k] = max(sm[k], 0.05*min(cldavg, 1.0))
            sh[k] = max(sh[k], 0.05*min(cldavg, 1.0))
        end

        elq = el[k]*qkw[k]
        elh = elq*qdiv

        # Production of TKE (pdk), T-variance (pdt), q-variance (pdq), covariance (pdc)
        pdk[k] = elq*(sm[k]*gm[k] + sh[k]*gh[k] + gamv) + 0.5*TKEprodTD[k]
        pdt[k] = elh*(sh[k]*dtl[k] + gamt)*dtl[k]
        pdq[k] = elh*(sh[k]*dqw[k] + gamq)*dqw[k]
        pdc[k] = elh*(sh[k]*dtl[k] + gamt)*dqw[k]*0.5 +
                 elh*(sh[k]*dqw[k] + gamq)*dtl[k]*0.5

        # Countergradient terms
        tcd[k] = elq*gamt
        qcd[k] = elq*gamq

        # Eddy diffusivity/viscosity divided by dz. In mym_predict, dfq for the TKE
        # and the scalar variances are 3.0*dfm and 1.0*dfm respectively (Sqfac).
        dfm[k] = elq*sm[k] / dzk
        dfh[k] = elq*sh[k] / dzk
        dfq[k] =     dfm[k]

        if tke_budget == 1                                              # :3085-3115
            #!!  TKE budget  (Puhales, 2020, WRF 4.2.1)
            qSHEAR1D[k] = elq*sm[k]*gm[k]                     # staggered
            qBUOY1D[k]  = elq*(sh[k]*gh[k] + gamv) + 0.5*TKEprodTD[k]
        end
    end

    @inbounds begin
        dfm[kts] = 0.0
        dfh[kts] = 0.0
        dfq[kts] = 0.0
        tcd[kts] = 0.0
        qcd[kts] = 0.0

        tcd[kte] = 0.0
        qcd[kte] = 0.0

        for k in kts:(kte-1)
            dzk = dz[k]
            tcd[k] = (tcd[k+1] - tcd[k])/(dzk)
            qcd[k] = (qcd[k+1] - qcd[k])/(dzk)
        end

        if spp_pbl == 1                                                 # :3136-3141
            for k in kts:kte
                dfm[k] = dfm[k] + dfm[k]*rstoch_col[k]*1.5*
                         max(exp(-max(zw[k] - 8000.0, 0.0)/2000.0), 0.001)
                dfh[k] = dfh[k] + dfh[k]*rstoch_col[k]*1.5*
                         max(exp(-max(zw[k] - 8000.0, 0.0)/2000.0), 0.001)
            end
        end
    end
    return nothing
end

# ── Prediction of qke and the variances (:3197-3567) ─────────────────────────

"""
    mym_predict!(kts, kte, closure, delt, dz, ust, flt, flq, pmz, phh,
                 el, dfq, rho, pdk, pdt, pdq, pdc, qke, tsq, qsq, cov,
                 s_aw, s_awqke, bl_mynn_edmf_tke, qWT1D, qDISS1D, tke_budget,
                 c::MYNNConstants, work::MYNNWork)

Step `qke` forward with a Crank-Nicolson tridiagonal solve and set the variances
`tsq`, `qsq`, `cov`. Fortran: `SUBROUTINE mym_predict` (:3197-3567).

Two closure paths are ported:

  * `closure <= 2.5` — `qsq`, `tsq` and `cov` all come from the Level-2 diagnostic
    `b2l*(pd·(k+1) + pd·(k))` (:3429-3440 and :3543-3559). None of them reads its own
    input value, so the routine is testable at any step from the printed `pdt/pdq/pdc`.
  * `2.5 < closure < 3.0` (the reference's 2.6) — `qsq` becomes prognostic with its
    own tridiagonal solve (:3394-3428) and IS a function of the incoming `qsq`;
    `tsq`/`cov` stay diagnostic.

`closure >= 3.0` RAISES (:3443-3541 out of scope).

`pdk`, `pdt`, `pdq` and `pdc` are `intent(inout)` and their `kts` element is
OVERWRITTEN here (:3293-3300) — `pdk(kts) = pdk1 - pdk(kts+1)` with the surface
production `pdk1`, and the other three simply copy level `kts+1`.

`qWT1D`/`qDISS1D` are written only when `tke_budget == 1`; with the reference's
`tke_budget = 0` they are accepted and left untouched (the Fortran declares them
`intent(out)`, so they are undefined there — do not rely on their contents).

The keyword `sqfac` is the `:sqfac1` fidelity deviation (see `MYNN_DEVIATIONS`): the
TKE's own diffusivity is `sqfac*dfq`, and the default [`MYNN_SQFAC`](@ref) `= 3.0` is
the Fortran's `Sqfac`.
"""
function mym_predict!(kts::Int, kte::Int, closure::Float64, delt::Float64,
                      dz::Vector{Float64}, ust::Float64,
                      flt::Float64, flq::Float64, pmz::Float64, phh::Float64,
                      el::Vector{Float64}, dfq::Vector{Float64}, rho::Vector{Float64},
                      pdk::Vector{Float64}, pdt::Vector{Float64},
                      pdq::Vector{Float64}, pdc::Vector{Float64},
                      qke::Vector{Float64}, tsq::Vector{Float64},
                      qsq::Vector{Float64}, cov::Vector{Float64},
                      s_aw::Vector{Float64}, s_awqke::Vector{Float64},
                      bl_mynn_edmf_tke::Int,
                      qWT1D::Vector{Float64}, qDISS1D::Vector{Float64},
                      tke_budget::Int,
                      c::MYNNConstants, work::MYNNWork;
                      sqfac::Float64 = MYNN_SQFAC)
    closure < 3.0 ||
        throw(ArgumentError("mym_predict!: closure = $(closure); the Level-3 " *
                            "prognostic tsq/cov branch (module_bl_mynn.F90 " *
                            ":3443-3541) is out of scope. Only closure < 3.0 is ported."))

    qkw = work.p_qkw; bp = work.p_bp; rp = work.p_rp; df3q = work.p_df3q
    dtz = work.p_dtz
    a = work.p_a; b = work.p_b; cc = work.p_c; d = work.p_d; x = work.p_x
    rhoinv = work.p_rhoinv
    rhoz = work.p_rhoz; kqdz = work.p_kqdz; kmdz = work.p_kmdz

    # REGULATE THE MOMENTUM MIXING FROM THE MASS-FLUX SCHEME (on or off)
    onoff = bl_mynn_edmf_tke == 0 ? 0.0 : 1.0

    @inbounds begin
        #   **  Strictly, vkz*h(i,j) -> karman*( 0.5*dz(1)*h(i,j)+z0 )  **
        vkz = c.karman*0.5*dz[kts]

        #   **  dfq for the TKE is 3.0*dfm.  **
        for k in kts:kte
            qkw[k]  = sqrt(max(qke[k], 0.0))
            df3q[k] = sqfac*dfq[k]
            dtz[k]  = delt/dz[k]
        end

        # Prepare "constants" for the diffusion equation: khdz = rho*Kh/dz = rho*dfh
        rhoz[kts]   = rho[kts]
        rhoinv[kts] = 1.0/rho[kts]
        kqdz[kts]   = rhoz[kts]*df3q[kts]
        kmdz[kts]   = rhoz[kts]*dfq[kts]
        for k in (kts+1):kte
            rhoz[k]   = (rho[k]*dz[k-1] + rho[k-1]*dz[k])/(dz[k-1] + dz[k])
            rhoz[k]   = max(rhoz[k], 1e-4)
            rhoinv[k] = 1.0/max(rho[k], 1e-4)
            kqdz[k]   = rhoz[k]*df3q[k]   # for TKE
            kmdz[k]   = rhoz[k]*dfq[k]    # for T'2, q'2 and T'q'
        end
        rhoz[kte+1] = rhoz[kte]
        kqdz[kte+1] = rhoz[kte+1]*df3q[kte]
        kmdz[kte+1] = rhoz[kte+1]*dfq[kte]

        # stability criteria for mf
        for k in (kts+1):(kte-1)
            kqdz[k] = max(kqdz[k],  0.5* s_aw[k])
            kqdz[k] = max(kqdz[k], -0.5*(s_aw[k] - s_aw[k+1]))
            kmdz[k] = max(kmdz[k],  0.5* s_aw[k])
            kmdz[k] = max(kmdz[k], -0.5*(s_aw[k] - s_aw[k+1]))
        end

        pdk1 = 2.0*ust^3*pmz/(vkz)
        phm  = 2.0/ust   *phh/(vkz)
        pdt1 = phm*flt^2
        pdq1 = phm*flq^2
        pdc1 = phm*flt*flq

        #   **  pdk(1)+pdk(2) corresponds to pdk1.  **
        pdk[kts] = pdk1 - pdk[kts+1]
        # (the analogous pdt1/pdq1/pdc1 lines are commented out at :3295-3297)
        pdt[kts] = pdt[kts+1]
        pdq[kts] = pdq[kts+1]
        pdc[kts] = pdc[kts+1]

        #   **  Prediction of twice the turbulent kinetic energy  **
        for k in kts:(kte-1)
            b1l   = MYNN_B1*0.5*(el[k+1] + el[k])
            bp[k] = 2.0*qkw[k] / b1l
            rp[k] = pdk[k+1] + pdk[k]
        end

        for k in kts:(kte-1)
            a[k]  =   - dtz[k]*kqdz[k]*rhoinv[k] +
                        0.5*dtz[k]*rhoinv[k]*s_aw[k]*onoff
            b[k]  = 1.0 + dtz[k]*(kqdz[k] + kqdz[k+1])*rhoinv[k] +
                        0.5*dtz[k]*rhoinv[k]*(s_aw[k] - s_aw[k+1])*onoff +
                        bp[k]*delt
            cc[k] =   - dtz[k]*kqdz[k+1]*rhoinv[k] -
                        0.5*dtz[k]*rhoinv[k]*s_aw[k+1]*onoff
            d[k]  = rp[k]*delt + qke[k] +
                        dtz[k]*rhoinv[k]*(s_awqke[k] - s_awqke[k+1])*onoff
        end

        # "prescribed value" at the top
        a[kte]  = 0.0
        b[kte]  = 1.0
        cc[kte] = 0.0
        d[kte]  = qke[kte]

        tridiag2!(kte, a, b, cc, d, x, work)

        for k in kts:kte
            qke[k] = max(x[k], MYNN_QKEMIN)
            qke[k] = min(qke[k], 150.0)
        end

        if tke_budget == 1                                              # :3368-3391
            tke_up = work.p_tke_up
            dzinv  = work.p_dzinv
            for k in kts:kte
                tke_up[k] = 0.5*qke[k]
                dzinv[k]  = 1.0/dz[k]
            end
            k = kts
            qWT1D[k] = dzinv[k]*(
                       (kqdz[k+1]*(tke_up[k+1] - tke_up[k]) - kqdz[k]*tke_up[k]) +
                       0.5*rhoinv[k]*(s_aw[k+1]*tke_up[k+1] +
                                      (s_aw[k+1] - s_aw[k])*tke_up[k] +
                                      (s_awqke[k] - s_awqke[k+1]))*onoff)
            for k in (kts+1):(kte-1)
                qWT1D[k] = dzinv[k]*(
                       (kqdz[k+1]*(tke_up[k+1] - tke_up[k]) -
                        kqdz[k]*(tke_up[k] - tke_up[k-1])) +
                       0.5*rhoinv[k]*(s_aw[k+1]*tke_up[k+1] +
                                      (s_aw[k+1] - s_aw[k])*tke_up[k] -
                                                  s_aw[k]*tke_up[k-1] +
                                      (s_awqke[k] - s_awqke[k+1]))*onoff)
            end
            k = kte
            qWT1D[k] = dzinv[k]*(-kqdz[k]*(tke_up[k] - tke_up[k-1]) +
                       0.5*rhoinv[k]*(-s_aw[k]*tke_up[k] - s_aw[k]*tke_up[k-1] +
                                       s_awqke[k])*onoff)
            for k in kts:kte
                qDISS1D[k] = bp[k]*tke_up[k]
            end
        end

        if closure > 2.5
            #   **  Prediction of the moisture variance  **
            for k in kts:(kte-1)
                b2l   = MYNN_B2*0.5*(el[k+1] + el[k])
                bp[k] = 2.0*qkw[k] / b2l
                rp[k] = pdq[k+1] + pdq[k]
            end

            for k in kts:(kte-1)
                a[k]  =   - dtz[k]*kmdz[k]*rhoinv[k]
                b[k]  = 1.0 + dtz[k]*(kmdz[k] + kmdz[k+1])*rhoinv[k] + bp[k]*delt
                cc[k] =   - dtz[k]*kmdz[k+1]*rhoinv[k]
                d[k]  = rp[k]*delt + qsq[k]
            end

            a[kte]  = -1.0
            b[kte]  =  1.0
            cc[kte] =  0.0
            d[kte]  =  0.0

            tridiag2!(kte, a, b, cc, d, x, work)

            for k in kts:kte
                # NOTE :3427 — with a NaN `x(k)` (ust = 0, see the divergence note at
                # the top of part 2) gfortran's MAX returns 1e-17; Julia's propagates.
                qsq[k] = max(x[k], 1e-17)
            end
        else
            # level 2.5 - use the level 2 diagnostic
            for k in kts:(kte-1)
                b2l = if qkw[k] <= 0.0
                    0.0
                else
                    MYNN_B2*0.25*(el[k+1] + el[k])/qkw[k]
                end
                qsq[k] = b2l*(pdq[k+1] + pdq[k])
            end
            qsq[kte] = qsq[kte-1]
        end
        # !!!!!!!!!!!!!!!!!!!!!! end level 2.6

        # closure >= 3.0 has been refused above; this is the Fortran's ELSE (:3543).
        # Not level 3 - default to the level 2 diagnostic
        for k in kts:(kte-1)
            b2l = if qkw[k] <= 0.0
                0.0
            else
                MYNN_B2*0.25*(el[k+1] + el[k])/qkw[k]
            end
            tsq[k] = b2l*(pdt[k+1] + pdt[k])
            cov[k] = b2l*(pdc[k+1] + pdc[k])
        end
        tsq[kte] = tsq[kte-1]
        cov[kte] = cov[kte-1]
    end
    return nothing
end

# ── Subgrid cloud PDF, CASE 2 only (:3603-4021) ──────────────────────────────

"""
    mym_condensation!(kts, kte, dx, dz, zw, xland, thl, qw, qv, qc, qi, qs,
                      p, exner, tsq, qsq, cov, Sh, el, bl_mynn_cloudpdf,
                      qc_bl1D, qi_bl1D, cldfra_bl1D, PBLH1, HFX1,
                      vt, vq, th, sgm, rmo, spp_pbl, rstoch_col,
                      c::MYNNConstants, work::MYNNWork)

Subgrid cloud fraction, subgrid condensate and the buoyancy-flux coefficients
`vt` (beta-theta) and `vq` (beta-q). Fortran: `SUBROUTINE mym_condensation`
(:3603-4021), `CASE (2, -2)` — the Chaboureau and Bechtold (2002) diagnostic
statistical scheme with `sigma` from the prognostic `qsq`.

`bl_mynn_cloudpdf` must be `2` or `-2`; CASE 0 (:3689) and CASE (1,-1) (:3757) raise.
A NEGATIVE value additionally zeroes `cldfra_bl1D`, `qc_bl1D` and `qi_bl1D` afterwards
(:4000-4006), which is the "isolate the mass-flux clouds" test switch.

`vt`, `vq` and `sgm` are `intent(inout)` of the Fortran and are carried across steps by
the caller (harness README item 6). CASE 2 writes every level of all three, so their
entry values never matter — but they are still carried so the port mirrors the driver.

Fortran shapes preserved:

  * the tropopause search (:3673-3683) is a `DO k = kte-3, kts, -1` with an `EXIT`;
    `k_tropo = MAX(kts+2, k+2)` reads the loop variable AFTER the loop, which is
    `kts-1` when the loop ran to completion. That is reproduced exactly.
  * `zagl` is accumulated as `zagl + 0.5*(dz(k) + dzm1)` with `dzm1` starting at 0
    (:3819-3823), i.e. the layer mid-heights.
  * `cfmax` is taken from `cldfra_bl1D(k)` BEFORE the `cld_factor` amplification of
    :3994, so `vt`/`vq` see the unamplified cloud fraction.
  * `ql(kte) = ql(kte-1)` (:4008) reads a local that CASE 2 never writes. Transcribed
    against `work.c_ql`, which is zero, and the value is not an output.
  * `Sh`, `el`, `HFX1`, `rmo`, `tsq` and `cov` are unread by CASE 2 (only CASE 1 uses
    `Sh`/`el`, and only CASE 0 uses `tsq`/`cov`); they stay in the signature so the
    call site matches the Fortran.
"""
function mym_condensation!(kts::Int, kte::Int, dx::Float64,
                           dz::Vector{Float64}, zw::Vector{Float64}, xland::Float64,
                           thl::Vector{Float64}, qw::Vector{Float64},
                           qv::Vector{Float64}, qc::Vector{Float64},
                           qi::Vector{Float64}, qs::Vector{Float64},
                           p::Vector{Float64}, exner::Vector{Float64},
                           tsq::Vector{Float64}, qsq::Vector{Float64},
                           cov::Vector{Float64},
                           Sh::Vector{Float64}, el::Vector{Float64},
                           bl_mynn_cloudpdf::Int,
                           qc_bl1D::Vector{Float64}, qi_bl1D::Vector{Float64},
                           cldfra_bl1D::Vector{Float64},
                           PBLH1::Float64, HFX1::Float64,
                           vt::Vector{Float64}, vq::Vector{Float64},
                           th::Vector{Float64}, sgm::Vector{Float64}, rmo::Float64,
                           spp_pbl::Int, rstoch_col::Vector{Float64},
                           c::MYNNConstants, work::MYNNWork)
    (bl_mynn_cloudpdf == 2 || bl_mynn_cloudpdf == -2) ||
        throw(ArgumentError("mym_condensation!: bl_mynn_cloudpdf = " *
                            "$(bl_mynn_cloudpdf); only CASE (2, -2) " *
                            "(module_bl_mynn.F90 :3814-3995) is ported. CASE 0 " *
                            "(:3689) and CASE (1, -1) (:3757) are out of scope."))

    qpct_sfc = 0.025                                                    # :3642
    qpct_pbl = 0.030                                                    # :3643
    qpct_trp = 0.040                                                    # :3644
    rhcrit   = 0.83                                                     # :3645
    rhmax    = 1.02                                                     # :3646

    alp = work.c_alp; a = work.c_a; bet = work.c_bet; b = work.c_b
    ql  = work.c_ql;  q1 = work.c_q1; rh = work.c_rh

    @inbounds begin
        # First, an estimate of the tropopause height, as in the Thompson subgrid-cloud
        # scheme. `ktrop` is the Fortran DO variable after the loop: the level the EXIT
        # fired at, kts-1 if the loop ran to completion, and the untouched initial
        # value kte-3 if the loop has zero trips (kte < kts+3).
        ktrop = kte - 3
        for k in (kte-3):-1:kts
            theta1 = th[k]
            theta2 = th[k+2]
            ht1 = 44307.692 * (1.0 - (p[k]  /101325.0)^0.190)
            ht2 = 44307.692 * (1.0 - (p[k+2]/101325.0)^0.190)
            if ((theta2 - theta1)/(ht2 - ht1) < 10.0/1500.0) &&
               (ht1 < 19000.0) && (ht1 > 4000.0)
                ktrop = k
                break
            end
            ktrop = k - 1
        end
        k_tropo = max(kts + 2, ktrop + 2)

        # CASE (2, -2): diagnostic statistical scheme of Chaboureau and Bechtold
        # (2002), JAS, but with higher-order moments used to estimate sigma.
        pblh2 = max(10.0, PBLH1)
        zagl = 0.0
        dzm1 = 0.0
        for k in kts:(kte-1)
            zagl = zagl + 0.5*(dz[k] + dzm1)
            dzm1 = dz[k]

            t       = th[k]*exner[k]
            xl      = xl_blend(t, c)                # obtain latent heat
            qsat_tk = qsat_blend(t, p[k], c)        # q_sat at tk and p
            rh[k]   = max(min(rhmax, qw[k]/max(1.0e-10, qsat_tk)), 0.001)

            # dqw/dT: Clausius-Clapeyron
            dqsl   = qsat_tk*c.ep_2*c.xlv/(c.r_d*t^2)
            alp[k] = 1.0/(1.0 + dqsl*c.xlvcp)
            bet[k] = dqsl*exner[k]

            rsl  = xl*qsat_tk / (c.r_v*t^2)   # slope of the C-C curve at t; CB02 Eq. 4
            cpm  = c.cp + qw[k]*c.cpv         # CB02, sec. 2, para. 1
            a[k] = 1.0/(1.0 + xl*rsl/cpm)     # CB02 variable "a"
            b[k] = a[k]*rsl                   # CB02 variable "b"

            # SPP
            qw_pert = qw[k] + qw[k]*0.5*rstoch_col[k]*Float64(spp_pbl)

            # This form of qmq (the numerator of Q1) no longer uses the a(k) factor
            qmq = qw_pert - qsat_tk           # saturation deficit/excess

            # Eq. (6) of Chaboureau and Bechtold (2002), all but the first term of
            # sig_r neglected.
            # NOTE :3849 — with a NaN `qsq(k)` (ust = 0, see the divergence note
            # at the top of part 2) gfortran's MAX returns 0.0; Julia's propagates.
            r3sq = max(qsq[k], 0.0)
            # Calculate sigma using higher-order moments
            sgm[k] = sqrt(r3sq)
            # Constrain sigma relative to the saturation water vapour
            sgm[k] = min(sgm[k], qsat_tk*0.666)

            # vertical-grid-spacing dependence of the minimum sgm
            # (= 1 for dz < 100 m, = 0 for dz > 600 m)
            wt     = max(500.0 - max(dz[k] - 100.0, 0.0), 0.0)/500.0
            sgm[k] = sgm[k] + sgm[k]*0.2*(1.0 - wt)   # inflate sgm for coarse dz

            # allow the min sgm to vary with dz and z
            qpct   = qpct_pbl*wt + qpct_trp*(1.0 - wt)
            qpct   = min(qpct, max(qpct_sfc, qpct_pbl*zagl/500.0))
            sgm[k] = max(sgm[k], qsat_tk*qpct)

            q1[k] = qmq / sgm[k]   # Q1, the normalized saturation

            # Condition for falling/settling into low-RH layers, so at least some
            # cloud fraction is applied for all qc, qs and qi.
            rh_hack = rh[k]
            wt2     = min(max(zagl - pblh2, 0.0)/300.0, 1.0)
            # ensure adequate RH & q1 when qi is at least 1e-9 (above the PBLH)
            if (qi[k] + qs[k]) > 1.0e-9 && (zagl > pblh2)
                rh_hack = min(rhmax, rhcrit + wt2*0.045*(9.0 + log10(qi[k] + qs[k])))
                rh[k]   = max(rh[k], rh_hack)
                q1_rh   = -3.0 + 3.0*(rh[k] - rhcrit)/(1.0 - rhcrit)
                q1[k]   = max(q1_rh, q1[k])
            end
            # ensure adequate rh & q1 when qc is at least 1e-6 (above the PBLH)
            if qc[k] > 1.0e-6 && (zagl > pblh2)
                rh_hack = min(rhmax, rhcrit + wt2*0.08*(6.0 + log10(qc[k])))
                rh[k]   = max(rh[k], rh_hack)
                q1_rh   = -3.0 + 3.0*(rh[k] - rhcrit)/(1.0 - rhcrit)
                q1[k]   = max(q1_rh, q1[k])
            end

            q1k = q1[k]            # backup Q1 for later modification

            # Specify the cloud fraction. "Best compromise": improves marine stratus
            # without adding much cold bias.
            cldfra_bl1D[k] = max(0.0, min(1.0, 0.5 + 0.36*atan(1.8*(q1[k] + 0.2))))

            # Specify the hydrometeors. The cloud-water formulations are CB02 Eq. 8.
            maxqc = max(qw[k] - qsat_tk, 0.0)
            local ql_water::Float64, ql_ice::Float64
            if q1k < 0.0            # unsaturated
                ql_water = sgm[k]*exp(1.2*q1k - 1.0)
                ql_ice   = sgm[k]*exp(1.2*q1k - 1.0)
            elseif q1k > 2.0        # supersaturated
                ql_water = min(sgm[k]*q1k, maxqc)
                ql_ice   =     sgm[k]*q1k
            else                    # slightly saturated (0 > q1 < 2)
                ql_water = min(sgm[k]*(exp(-1.0) + 0.66*q1k + 0.086*q1k^2), maxqc)
                ql_ice   =     sgm[k]*(exp(-1.0) + 0.66*q1k + 0.086*q1k^2)
            end

            if cldfra_bl1D[k] < 0.001
                ql_ice   = 0.0
                ql_water = 0.0
                cldfra_bl1D[k] = 0.0
            end

            liq_frac   = min(1.0, max(0.0, (t - MYNN_TICE)/(MYNN_TLIQ - MYNN_TICE)))
            qc_bl1D[k] = liq_frac*ql_water
            qi_bl1D[k] = (1.0 - liq_frac)*ql_ice

            # Above the tropopause: eliminate the subgrid clouds from the CB scheme.
            if k >= k_tropo
                cldfra_bl1D[k] = 0.0
                qc_bl1D[k]     = 0.0
                qi_bl1D[k]     = 0.0
            end

            # Buoyancy-flux-related calculations follow.
            # limiting Q1 to avoid too much diffusion in cloud layers
            if (xland - 1.5) >= 0     # water
                q1k = max(q1[k], -2.5)
            else                      # land
                q1k = max(q1[k], -2.0)
            end
            # "Fng" is the non-Gaussian transport factor of Bechtold and Siebesma
            # (1998, JAS); Bechtold et al. (1995) sec. 3(c) Eq. 20 is commented out.
            local Fng::Float64
            if q1k >= 1.0
                Fng = 1.0
            elseif q1k >= -1.7 && q1k < 1.0
                Fng = exp(-0.4*(q1k - 1.0))
            elseif q1k >= -2.5 && q1k < -1.7
                Fng = 3.0 + exp(-3.8*(q1k + 1.7))
            else
                Fng = min(23.9 + exp(-1.6*(q1k + 2.5)), 60.0)
            end

            cfmax = min(cldfra_bl1D[k], 0.6)
            # Further limit the cf going into vt & vq near the surface
            zsl   = min(max(25.0, 0.1*pblh2), 100.0)
            wt    = min(zagl/zsl, 1.0)   # = 0 at z = 0 m, = 1 above the Ekman layer
            cfmax = cfmax*wt

            # bb is "b" in BCMT95; their "b" differs from CB02's b(k) by a factor
            # T/theta. The sat-mixing-ratio to sat-specific-humidity conversion is
            # neglected here.
            bb    = b[k]*t/th[k]
            qww   = 1.0 + 0.61*qw[k]
            alpha = 0.61*th[k]
            beta  = (th[k]/t)*(xl/c.cp) - 1.61*th[k]
            vt[k] = qww   - cfmax*beta*bb*Fng   - 1.0
            vq[k] = alpha + cfmax*beta*a[k]*Fng - c.tv0

            # dampen the amplification factor where need be
            fac_damp   = min(zagl * 0.0025, 1.0)
            cld_factor = 1.0 + fac_damp*min((max(0.0, (rh[k] - 0.92)) / 0.145)^2, 0.37)
            cldfra_bl1D[k] = min(1.0, cld_factor*cldfra_bl1D[k])
        end
        # end cloudPDF option

        # For testing purposes only, option for isolating on the mass-flux clouds.
        if bl_mynn_cloudpdf < 0
            for k in kts:(kte-1)
                cldfra_bl1D[k] = 0.0
                qc_bl1D[k] = 0.0
                qi_bl1D[k] = 0.0
            end
        end

        ql[kte] = ql[kte-1]     # :4008 — a local CASE 2 never writes; not an output
        vt[kte] = vt[kte-1]
        vq[kte] = vq[kte-1]
        qc_bl1D[kte]     = 0.0
        qi_bl1D[kte]     = 0.0
        cldfra_bl1D[kte] = 0.0
    end
    return nothing
end

# ── Non-negative moisture correction (:5133-5220) ────────────────────────────

"""
    moisture_check!(kte, delt, dp, exner, qv, qc, qi, qs, th, dqv, dqc, dqi, dqs, dth,
                    c::MYNNConstants)

Force `qc`, `qi`, `qs` >= 0 and `qv` >= 1e-20 by condensing vapour and by borrowing
from the layer below, updating both the state and the tendencies. Adopted from the
CAM-UW shallow-cumulus scheme. Fortran: `SUBROUTINE moisture_check` (:5133-5220).

Everything is `intent(inout)`: `qv`, `qc`, `qi`, `qs` and `th` (which `mynn_tendencies`
calls with its WORKING `thl`, not theta — harness README item 1) as well as all five
tendencies. `dp` is the layer pressure thickness `delp`.

The Fortran reads `dqv2` AFTER the `do k = kte, 1, -1` loop (:5198), i.e. the vapour
deficit of the LOWEST layer, to decide whether to redistribute. That is not a typo and
is reproduced: `dqv2` is declared outside the loop here for the same reason.
"""
function moisture_check!(kte::Int, delt::Float64,
                         dp::Vector{Float64}, exner::Vector{Float64},
                         qv::Vector{Float64}, qc::Vector{Float64},
                         qi::Vector{Float64}, qs::Vector{Float64},
                         th::Vector{Float64},
                         dqv::Vector{Float64}, dqc::Vector{Float64},
                         dqi::Vector{Float64}, dqs::Vector{Float64},
                         dth::Vector{Float64}, c::MYNNConstants)
    qvmin = 1e-20
    qcmin = 0.0
    qimin = 0.0

    dqv2 = 0.0
    @inbounds begin
        for k in kte:-1:1     # From the top to the surface
            dqc2 = max(0.0, qcmin - qc[k])   # qc deficit (>= 0)
            dqi2 = max(0.0, qimin - qi[k])   # qi deficit (>= 0)
            dqs2 = max(0.0, qimin - qs[k])   # qs deficit (>= 0)

            # fix the tendencies
            dqc[k] = dqc[k] +  dqc2/delt
            dqi[k] = dqi[k] +  dqi2/delt
            dqs[k] = dqs[k] +  dqs2/delt
            dqv[k] = dqv[k] - (dqc2 + dqi2 + dqs2)/delt
            dth[k] = dth[k] + c.xlvcp/exner[k]*(dqc2/delt) +
                              c.xlscp/exner[k]*((dqi2 + dqs2)/delt)
            # update the species
            qc[k] = qc[k] +  dqc2
            qi[k] = qi[k] +  dqi2
            qs[k] = qs[k] +  dqs2
            qv[k] = qv[k] -  dqc2 - dqi2 - dqs2
            th[k] = th[k] +  c.xlvcp/exner[k]*dqc2 +
                             c.xlscp/exner[k]*(dqi2 + dqs2)

            # then fix qv
            dqv2   = max(0.0, qvmin - qv[k])   # qv deficit (>= 0)
            dqv[k] = dqv[k] + dqv2/delt
            qv[k]  = qv[k]  + dqv2
            if k != 1
                qv[k-1]  = qv[k-1]  - dqv2*dp[k]/dp[k-1]
                dqv[k-1] = dqv[k-1] - dqv2*dp[k]/dp[k-1]/delt
            end
            qv[k] = max(qv[k], qvmin)
            qc[k] = max(qc[k], qcmin)
            qi[k] = max(qi[k], qimin)
            qs[k] = max(qs[k], qimin)
        end

        # Extra moisture used to satisfy 'qv(1) >= qvmin' is proportionally extracted
        # from all the layers that have 'qv > 2*qvmin'. This fully preserves the
        # column moisture.
        if dqv2 > 1.0e-20
            ssum = 0.0
            for k in 1:kte
                if qv[k] > 2.0*qvmin
                    ssum = ssum + qv[k]*dp[k]
                end
            end
            aa = dqv2*dp[1]/max(1.0e-20, ssum)
            if aa < 0.5
                for k in 1:kte
                    if qv[k] > 2.0*qvmin
                        dum    = aa*qv[k]
                        qv[k]  = qv[k] - dum
                        dqv[k] = dqv[k] - dum/delt
                    end
                end
            else
                # "Full moisture conservation is impossible" — the Fortran only
                # writes a (commented-out) message here.
            end
        end
    end
    return nothing
end

# ── Exchange coefficients from the diffusivities (:5359-5383) ────────────────

"""
    retrieve_exchange_coeffs!(kts, kte, dfm, dfh, dz, K_m, K_h)

Undo `mym_turbulence`'s division by `dzk`: `K = df*0.5*(dz(k)+dz(k-1))`, with
`K(kts) = 0`. Fortran: `SUBROUTINE retrieve_exchange_coeffs` (:5359-5383).
"""
function retrieve_exchange_coeffs!(kts::Int, kte::Int,
                                   dfm::Vector{Float64}, dfh::Vector{Float64},
                                   dz::Vector{Float64},
                                   K_m::Vector{Float64}, K_h::Vector{Float64})
    @inbounds begin
        K_m[kts] = 0.0
        K_h[kts] = 0.0
        for k in (kts+1):kte
            dzk = 0.5  *(dz[k] + dz[k-1])
            K_m[k] = dfm[k]*dzk
            K_h[k] = dfh[k]*dzk
        end
    end
    return nothing
end

# ── Implicit tendency solve (:4027-5130) ─────────────────────────────────────

"""
    mynn_tendencies!(kts, kte, i, delt, dz, rho, u, v, th, tk, qv, qc, qi, qs, qnc, qni,
                     psfc, p, exner, thl, sqv, sqc, sqi, sqs, sqw,
                     qnwfa, qnifa, qnbca, ozone,
                     ust, flt, flq, flqv, flqc, wspd, uoce, voce, tsq, qsq, cov,
                     tcd, qcd, dfm, dfh, dfq,
                     Du, Dv, Dth, Dqv, Dqc, Dqi, Dqs, Dqnc, Dqni,
                     Dqnwfa, Dqnifa, Dqnbca, Dozone, diss_heat,
                     s_aw, s_awthl, s_awqt, s_awqv, s_awqc, s_awu, s_awv,
                     s_awqnc, s_awqni, s_awqnwfa, s_awqnifa, s_awqnbca,
                     sd_aw, sd_awthl, sd_awqt, sd_awqv, sd_awqc, sd_awu, sd_awv,
                     sub_thl, sub_sqv, sub_u, sub_v, det_thl, det_sqv, det_sqc,
                     det_u, det_v,
                     FLAG_QC, FLAG_QI, FLAG_QNC, FLAG_QNI, FLAG_QS,
                     FLAG_QNWFA, FLAG_QNIFA, FLAG_QNBCA, cldfra_bl1d,
                     bl_mynn_cloudmix, bl_mynn_mixqt, bl_mynn_edmf, bl_mynn_edmf_mom,
                     bl_mynn_mixscalars, c::MYNNConstants, work::MYNNWork)

The implicit vertical-diffusion solve for `u`, `v`, `thl`, `sqc`, `sqv`, `sqi` and
ozone, and the tendencies that follow from it. Fortran: `SUBROUTINE mynn_tendencies`
(:4027-5130).

SCOPE. Ported for the argument set the reference driver exercises: `FLAG_QC =
FLAG_QI = true`, every other species flag `false`, `bl_mynn_mixqt = 0`. The branches
that set are refused with an `ArgumentError` naming the Fortran lines:

  * `bl_mynn_mixqt > 0` — the total-water solve (:4364-4427) plus the saturation
    back-out (:4896-4923).
  * any of `FLAG_QNI`, `FLAG_QNC`, `FLAG_QNWFA`, `FLAG_QNIFA`, `FLAG_QNBCA` true —
    the five scalar solves (:4650-4855). `FLAG_QS` too, although the Fortran's snow
    solve is already dead (`.AND. .false.`, :4617).

`bl_mynn_cloudmix` and the two `FLAG_QC`/`FLAG_QI` switches ARE branched on both ways;
their `ELSE` legs are one-line copies.

`thl`, `sqv`, `sqc`, `sqi`, `sqs` and `sqw` are `intent(inout)` (:4092): `thl` is
overwritten by the solve AND then again by `moisture_check!`. A caller that hands in
its own column arrays therefore lets that column EVOLVE — the reference driver
re-gathers working copies from the frozen host arrays on every call, and so does
`mynn_column_step!` (harness README item 1).

Dead-but-transcribed: `dztop` (:4124, only the commented-out gradient BCs read it),
`ustdrag`/`ustdiff` (:4170-4171) and the `dzk` of the interface loop (:4149). `dfq`,
`tsq`, `qsq`, `cov`, `qc`, `qi`, `qs` and `bl_mynn_edmf` are arguments the body never
reads; `qv` is read only by `rhosfc`. `i` is the Fortran's column index, used only in
debug prints.
"""
function mynn_tendencies!(kts::Int, kte::Int, i::Int, delt::Float64,
                          dz::Vector{Float64}, rho::Vector{Float64},
                          u::Vector{Float64}, v::Vector{Float64},
                          th::Vector{Float64}, tk::Vector{Float64},
                          qv::Vector{Float64}, qc::Vector{Float64},
                          qi::Vector{Float64}, qs::Vector{Float64},
                          qnc::Vector{Float64}, qni::Vector{Float64},
                          psfc::Float64, p::Vector{Float64}, exner::Vector{Float64},
                          thl::Vector{Float64}, sqv::Vector{Float64},
                          sqc::Vector{Float64}, sqi::Vector{Float64},
                          sqs::Vector{Float64}, sqw::Vector{Float64},
                          qnwfa::Vector{Float64}, qnifa::Vector{Float64},
                          qnbca::Vector{Float64}, ozone::Vector{Float64},
                          ust::Float64, flt::Float64, flq::Float64,
                          flqv::Float64, flqc::Float64, wspd::Float64,
                          uoce::Float64, voce::Float64,
                          tsq::Vector{Float64}, qsq::Vector{Float64},
                          cov::Vector{Float64},
                          tcd::Vector{Float64}, qcd::Vector{Float64},
                          dfm::Vector{Float64}, dfh::Vector{Float64},
                          dfq::Vector{Float64},
                          Du::Vector{Float64}, Dv::Vector{Float64},
                          Dth::Vector{Float64}, Dqv::Vector{Float64},
                          Dqc::Vector{Float64}, Dqi::Vector{Float64},
                          Dqs::Vector{Float64}, Dqnc::Vector{Float64},
                          Dqni::Vector{Float64}, Dqnwfa::Vector{Float64},
                          Dqnifa::Vector{Float64}, Dqnbca::Vector{Float64},
                          Dozone::Vector{Float64}, diss_heat::Vector{Float64},
                          s_aw::Vector{Float64}, s_awthl::Vector{Float64},
                          s_awqt::Vector{Float64}, s_awqv::Vector{Float64},
                          s_awqc::Vector{Float64}, s_awu::Vector{Float64},
                          s_awv::Vector{Float64}, s_awqnc::Vector{Float64},
                          s_awqni::Vector{Float64}, s_awqnwfa::Vector{Float64},
                          s_awqnifa::Vector{Float64}, s_awqnbca::Vector{Float64},
                          sd_aw::Vector{Float64}, sd_awthl::Vector{Float64},
                          sd_awqt::Vector{Float64}, sd_awqv::Vector{Float64},
                          sd_awqc::Vector{Float64}, sd_awu::Vector{Float64},
                          sd_awv::Vector{Float64},
                          sub_thl::Vector{Float64}, sub_sqv::Vector{Float64},
                          sub_u::Vector{Float64}, sub_v::Vector{Float64},
                          det_thl::Vector{Float64}, det_sqv::Vector{Float64},
                          det_sqc::Vector{Float64}, det_u::Vector{Float64},
                          det_v::Vector{Float64},
                          FLAG_QC::Bool, FLAG_QI::Bool, FLAG_QNC::Bool,
                          FLAG_QNI::Bool, FLAG_QS::Bool, FLAG_QNWFA::Bool,
                          FLAG_QNIFA::Bool, FLAG_QNBCA::Bool,
                          cldfra_bl1d::Vector{Float64},
                          bl_mynn_cloudmix::Int, bl_mynn_mixqt::Int,
                          bl_mynn_edmf::Int, bl_mynn_edmf_mom::Int,
                          bl_mynn_mixscalars::Int,
                          c::MYNNConstants, work::MYNNWork)
    bl_mynn_mixqt == 0 ||
        throw(ArgumentError("mynn_tendencies!: bl_mynn_mixqt = $(bl_mynn_mixqt); the " *
                            "total-water solve (module_bl_mynn.F90 :4364-4427) and " *
                            "its saturation back-out (:4896-4923) are out of scope."))
    (FLAG_QNC || FLAG_QNI || FLAG_QS || FLAG_QNWFA || FLAG_QNIFA || FLAG_QNBCA) &&
        throw(ArgumentError("mynn_tendencies!: the number-concentration, snow and " *
                            "aerosol solves (module_bl_mynn.F90 :4613-4855) are out " *
                            "of scope; FLAG_QNC/QNI/QS/QNWFA/QNIFA/QNBCA must all be " *
                            "false."))

    nonloc = 1.0     # :4122 (dead here: every scalar solve that reads it is refused)

    dtz    = work.e_dtz;    delp   = work.e_delp
    sqv2   = work.e_sqv2;   sqc2   = work.e_sqc2;   sqi2 = work.e_sqi2
    sqs2   = work.e_sqs2;   sqw2   = work.e_sqw2
    qni2   = work.e_qni2;   qnc2   = work.e_qnc2
    qnwfa2 = work.e_qnwfa2; qnifa2 = work.e_qnifa2; qnbca2 = work.e_qnbca2
    rhoinv = work.e_rhoinv
    a = work.e_a; b = work.e_b; cc = work.e_c; d = work.e_d; x = work.e_x
    rhoz = work.e_rhoz; khdz = work.e_khdz; kmdz = work.e_kmdz

    @inbounds begin
        dztop = 0.5*(dz[kte] + dz[kte-1])        # :4124 (dead)

        # REGULATE THE MOMENTUM MIXING FROM THE MASS-FLUX SCHEME (on or off). s_awu
        # and s_awv already arrive as 0 when bl_mynn_edmf_mom == 0, so only the MF
        # term needs zeroing.
        onoff = bl_mynn_edmf_mom == 0 ? 0.0 : 1.0

        # Prepare "constants" for the diffusion equation: khdz = rho*Kh/dz = rho*dfh
        rhosfc      = psfc/(c.r_d*(tk[kts] + c.p608*qv[kts]))
        dtz[kts]    = delt/dz[kts]
        rhoz[kts]   = rho[kts]
        rhoinv[kts] = 1.0/rho[kts]
        khdz[kts]   = rhoz[kts]*dfh[kts]
        kmdz[kts]   = rhoz[kts]*dfm[kts]
        delp[kts]   = psfc - (p[kts+1]*dz[kts] + p[kts]*dz[kts+1])/(dz[kts] + dz[kts+1])
        for k in (kts+1):kte
            dtz[k]    = delt/dz[k]
            rhoz[k]   = (rho[k]*dz[k-1] + rho[k-1]*dz[k])/(dz[k-1] + dz[k])
            rhoz[k]   = max(rhoz[k], 1e-4)
            rhoinv[k] = 1.0/max(rho[k], 1e-4)
            dzk       = 0.5  *(dz[k] + dz[k-1])    # :4149 (dead)
            khdz[k]   = rhoz[k]*dfh[k]
            kmdz[k]   = rhoz[k]*dfm[k]
        end
        for k in (kts+1):(kte-1)
            delp[k] = (p[k]*dz[k-1] + p[k-1]*dz[k])/(dz[k] + dz[k-1]) -
                      (p[k+1]*dz[k] + p[k]*dz[k+1])/(dz[k] + dz[k+1])
        end
        delp[kte]   = delp[kte-1]
        rhoz[kte+1] = rhoz[kte]
        khdz[kte+1] = rhoz[kte+1]*dfh[kte]
        kmdz[kte+1] = rhoz[kte+1]*dfm[kte]

        # stability criteria for mf
        for k in (kts+1):(kte-1)
            khdz[k] = max(khdz[k],  0.5*s_aw[k])
            khdz[k] = max(khdz[k], -0.5*(s_aw[k] - s_aw[k+1]))
            kmdz[k] = max(kmdz[k],  0.5*s_aw[k])
            kmdz[k] = max(kmdz[k], -0.5*(s_aw[k] - s_aw[k+1]))
        end

        ustdrag = min(ust*ust, 0.99)/wspd   # :4170 (dead) limit at ~ 20 m/s
        ustdiff = min(ust*ust, 0.01)/wspd   # :4171 (dead) limit at ~  2 m/s
        for k in kts:kte
            Dth[k] = 0.0     # must initialize for the moisture_check routine
        end

        # ==================================== u
        k = kts
        # rho-weighted (drag in the b-vector)
        a[k]  =  -dtz[k]*kmdz[k]*rhoinv[k]
        b[k]  = 1.0 + dtz[k]*(kmdz[k+1] + rhosfc*ust^2/wspd)*rhoinv[k] -
                0.5*dtz[k]*rhoinv[k]*s_aw[k+1]*onoff -
                0.5*dtz[k]*rhoinv[k]*sd_aw[k+1]*onoff
        cc[k] =  -dtz[k]*kmdz[k+1]*rhoinv[k] -
                0.5*dtz[k]*rhoinv[k]*s_aw[k+1]*onoff -
                0.5*dtz[k]*rhoinv[k]*sd_aw[k+1]*onoff
        d[k]  = u[k] + dtz[k]*uoce*ust^2/wspd -
                dtz[k]*rhoinv[k]*s_awu[k+1]*onoff +
                dtz[k]*rhoinv[k]*sd_awu[k+1]*onoff +
                sub_u[k]*delt + det_u[k]*delt

        for k in (kts+1):(kte-1)
            a[k]  =  -dtz[k]*kmdz[k]*rhoinv[k] +
                     0.5*dtz[k]*rhoinv[k]*s_aw[k]*onoff +
                     0.5*dtz[k]*rhoinv[k]*sd_aw[k]*onoff
            b[k]  = 1.0 + dtz[k]*(kmdz[k] + kmdz[k+1])*rhoinv[k] +
                     0.5*dtz[k]*rhoinv[k]*(s_aw[k] - s_aw[k+1])*onoff +
                     0.5*dtz[k]*rhoinv[k]*(sd_aw[k] - sd_aw[k+1])*onoff
            cc[k] =  -dtz[k]*kmdz[k+1]*rhoinv[k] -
                     0.5*dtz[k]*rhoinv[k]*s_aw[k+1]*onoff -
                     0.5*dtz[k]*rhoinv[k]*sd_aw[k+1]*onoff
            d[k]  = u[k] + dtz[k]*rhoinv[k]*(s_awu[k] - s_awu[k+1])*onoff -
                     dtz[k]*rhoinv[k]*(sd_awu[k] - sd_awu[k+1])*onoff +
                     sub_u[k]*delt + det_u[k]*delt
        end

        # prescribed value at the top
        a[kte]  = 0.0
        b[kte]  = 1.0
        cc[kte] = 0.0
        d[kte]  = u[kte]

        tridiag2!(kte, a, b, cc, d, x, work)

        for k in kts:kte
            Du[k] = (x[k] - u[k])/delt
        end

        # ==================================== v
        k = kts
        a[k]  =  -dtz[k]*kmdz[k]*rhoinv[k]
        b[k]  = 1.0 + dtz[k]*(kmdz[k+1] + rhosfc*ust^2/wspd)*rhoinv[k] -
                0.5*dtz[k]*rhoinv[k]*s_aw[k+1]*onoff -
                0.5*dtz[k]*rhoinv[k]*sd_aw[k+1]*onoff
        cc[k] =  -dtz[k]*kmdz[k+1]*rhoinv[k] -
                0.5*dtz[k]*rhoinv[k]*s_aw[k+1]*onoff -
                0.5*dtz[k]*rhoinv[k]*sd_aw[k+1]*onoff
        d[k]  = v[k] + dtz[k]*voce*ust^2/wspd -
                dtz[k]*rhoinv[k]*s_awv[k+1]*onoff +
                dtz[k]*rhoinv[k]*sd_awv[k+1]*onoff +
                sub_v[k]*delt + det_v[k]*delt

        for k in (kts+1):(kte-1)
            a[k]  =  -dtz[k]*kmdz[k]*rhoinv[k] +
                     0.5*dtz[k]*rhoinv[k]*s_aw[k]*onoff +
                     0.5*dtz[k]*rhoinv[k]*sd_aw[k]*onoff
            b[k]  = 1.0 + dtz[k]*(kmdz[k] + kmdz[k+1])*rhoinv[k] +
                     0.5*dtz[k]*rhoinv[k]*(s_aw[k] - s_aw[k+1])*onoff +
                     0.5*dtz[k]*rhoinv[k]*(sd_aw[k] - sd_aw[k+1])*onoff
            cc[k] =  -dtz[k]*kmdz[k+1]*rhoinv[k] -
                     0.5*dtz[k]*rhoinv[k]*s_aw[k+1]*onoff -
                     0.5*dtz[k]*rhoinv[k]*sd_aw[k+1]*onoff
            d[k]  = v[k] + dtz[k]*rhoinv[k]*(s_awv[k] - s_awv[k+1])*onoff -
                     dtz[k]*rhoinv[k]*(sd_awv[k] - sd_awv[k+1])*onoff +
                     sub_v[k]*delt + det_v[k]*delt
        end

        a[kte]  = 0.0
        b[kte]  = 1.0
        cc[kte] = 0.0
        d[kte]  = v[kte]

        tridiag2!(kte, a, b, cc, d, x, work)

        for k in kts:kte
            Dv[k] = (x[k] - v[k])/delt
        end

        # ==================================== thl tendency
        k = kts
        # rho-weighted: rhosfc*X*rhoinv(k)
        a[k]  =  -dtz[k]*khdz[k]*rhoinv[k]
        b[k]  = 1.0 + dtz[k]*(khdz[k+1] + khdz[k])*rhoinv[k] - 0.5*dtz[k]*rhoinv[k]*s_aw[k+1] - 0.5*dtz[k]*rhoinv[k]*sd_aw[k+1]
        cc[k] =  -dtz[k]*khdz[k+1]*rhoinv[k]                 - 0.5*dtz[k]*rhoinv[k]*s_aw[k+1] - 0.5*dtz[k]*rhoinv[k]*sd_aw[k+1]
        d[k]  = thl[k] + dtz[k]*rhosfc*flt*rhoinv[k] + tcd[k]*delt -
                dtz[k]*rhoinv[k]*s_awthl[k+1] - dtz[k]*rhoinv[k]*sd_awthl[k+1] +
                diss_heat[k]*delt + sub_thl[k]*delt + det_thl[k]*delt

        for k in (kts+1):(kte-1)
            a[k]  =  -dtz[k]*khdz[k]*rhoinv[k] + 0.5*dtz[k]*rhoinv[k]*s_aw[k] + 0.5*dtz[k]*rhoinv[k]*sd_aw[k]
            b[k]  = 1.0 + dtz[k]*(khdz[k] + khdz[k+1])*rhoinv[k] +
                    0.5*dtz[k]*rhoinv[k]*(s_aw[k] - s_aw[k+1]) + 0.5*dtz[k]*rhoinv[k]*(sd_aw[k] - sd_aw[k+1])
            cc[k] =  -dtz[k]*khdz[k+1]*rhoinv[k] - 0.5*dtz[k]*rhoinv[k]*s_aw[k+1] - 0.5*dtz[k]*rhoinv[k]*sd_aw[k+1]
            d[k]  = thl[k] + tcd[k]*delt +
                    dtz[k]*rhoinv[k]*(s_awthl[k] - s_awthl[k+1]) + dtz[k]*rhoinv[k]*(sd_awthl[k] - sd_awthl[k+1]) +
                    diss_heat[k]*delt +
                    sub_thl[k]*delt + det_thl[k]*delt
        end

        a[kte]  = 0.0
        b[kte]  = 1.0
        cc[kte] = 0.0
        d[kte]  = thl[kte]

        tridiag2!(kte, a, b, cc, d, x, work)

        for k in kts:kte
            thl[k] = x[k]
        end

        # bl_mynn_mixqt == 0 (refused otherwise): the total-water solve is skipped
        for k in kts:kte
            sqw2[k] = sqw[k]
        end

        # ==================================== cloud water (sqc)
        if bl_mynn_cloudmix > 0 && FLAG_QC
            k = kts
            a[k]  =  -dtz[k]*khdz[k]*rhoinv[k]
            b[k]  = 1.0 + dtz[k]*(khdz[k+1] + khdz[k])*rhoinv[k] - 0.5*dtz[k]*rhoinv[k]*s_aw[k+1] - 0.5*dtz[k]*rhoinv[k]*sd_aw[k+1]
            cc[k] =  -dtz[k]*khdz[k+1]*rhoinv[k]                 - 0.5*dtz[k]*rhoinv[k]*s_aw[k+1] - 0.5*dtz[k]*rhoinv[k]*sd_aw[k+1]
            d[k]  = sqc[k] + dtz[k]*rhosfc*flqc*rhoinv[k] + qcd[k]*delt -
                    dtz[k]*rhoinv[k]*s_awqc[k+1] - dtz[k]*rhoinv[k]*sd_awqc[k+1] +
                    det_sqc[k]*delt

            for k in (kts+1):(kte-1)
                a[k]  =  -dtz[k]*khdz[k]*rhoinv[k] + 0.5*dtz[k]*rhoinv[k]*s_aw[k] + 0.5*dtz[k]*rhoinv[k]*sd_aw[k]
                b[k]  = 1.0 + dtz[k]*(khdz[k] + khdz[k+1])*rhoinv[k] +
                        0.5*dtz[k]*rhoinv[k]*(s_aw[k] - s_aw[k+1]) + 0.5*dtz[k]*rhoinv[k]*(sd_aw[k] - sd_aw[k+1])
                cc[k] =  -dtz[k]*khdz[k+1]*rhoinv[k] - 0.5*dtz[k]*rhoinv[k]*s_aw[k+1] - 0.5*dtz[k]*rhoinv[k]*sd_aw[k+1]
                d[k]  = sqc[k] + qcd[k]*delt + dtz[k]*rhoinv[k]*(s_awqc[k] - s_awqc[k+1]) + dtz[k]*rhoinv[k]*(sd_awqc[k] - sd_awqc[k+1]) +
                        det_sqc[k]*delt
            end

            a[kte]  = 0.0
            b[kte]  = 1.0
            cc[kte] = 0.0
            d[kte]  = sqc[kte]

            tridiag2!(kte, a, b, cc, d, sqc2, work)
        else
            # If not mixing clouds, set the "updated" array equal to the original
            for k in kts:kte
                sqc2[k] = sqc[k]
            end
        end

        # ==================================== water vapour only (sqv)
        k = kts
        # limit unreasonably large negative fluxes: do not allow a specified surface
        # flux to reduce qv below 1e-8 kg/kg
        qvflux = flqv
        if qvflux < 0.0
            qvflux = max(qvflux, (min(0.9*sqv[kts] - 1e-8, 0.0)/dtz[kts]))
        end

        a[k]  =  -dtz[k]*khdz[k]*rhoinv[k]
        b[k]  = 1.0 + dtz[k]*(khdz[k+1] + khdz[k])*rhoinv[k] - 0.5*dtz[k]*rhoinv[k]*s_aw[k+1] - 0.5*dtz[k]*rhoinv[k]*sd_aw[k+1]
        cc[k] =  -dtz[k]*khdz[k+1]*rhoinv[k]                 - 0.5*dtz[k]*rhoinv[k]*s_aw[k+1] - 0.5*dtz[k]*rhoinv[k]*sd_aw[k+1]
        d[k]  = sqv[k] + dtz[k]*rhosfc*qvflux*rhoinv[k] + qcd[k]*delt -
                dtz[k]*rhoinv[k]*s_awqv[k+1] - dtz[k]*rhoinv[k]*sd_awqv[k+1] +
                sub_sqv[k]*delt + det_sqv[k]*delt

        for k in (kts+1):(kte-1)
            a[k]  =  -dtz[k]*khdz[k]*rhoinv[k] + 0.5*dtz[k]*rhoinv[k]*s_aw[k] + 0.5*dtz[k]*rhoinv[k]*sd_aw[k]
            b[k]  = 1.0 + dtz[k]*(khdz[k] + khdz[k+1])*rhoinv[k] +
                    0.5*dtz[k]*rhoinv[k]*(s_aw[k] - s_aw[k+1]) + 0.5*dtz[k]*rhoinv[k]*(sd_aw[k] - sd_aw[k+1])
            cc[k] =  -dtz[k]*khdz[k+1]*rhoinv[k] - 0.5*dtz[k]*rhoinv[k]*s_aw[k+1] - 0.5*dtz[k]*rhoinv[k]*sd_aw[k+1]
            d[k]  = sqv[k] + qcd[k]*delt + dtz[k]*rhoinv[k]*(s_awqv[k] - s_awqv[k+1]) + dtz[k]*rhoinv[k]*(sd_awqv[k] - sd_awqv[k+1]) +
                    sub_sqv[k]*delt + det_sqv[k]*delt
        end

        a[kte]  = 0.0
        b[kte]  = 1.0
        cc[kte] = 0.0
        d[kte]  = sqv[kte]

        tridiag2!(kte, a, b, cc, d, sqv2, work)

        # ==================================== cloud ice (sqi)
        if bl_mynn_cloudmix > 0 && FLAG_QI
            k = kts
            a[k]  =  -dtz[k]*khdz[k]*rhoinv[k]
            b[k]  = 1.0 + dtz[k]*(khdz[k+1] + khdz[k])*rhoinv[k]
            cc[k] =  -dtz[k]*khdz[k+1]*rhoinv[k]
            d[k]  = sqi[k]

            for k in (kts+1):(kte-1)
                a[k]  =  -dtz[k]*khdz[k]*rhoinv[k]
                b[k]  = 1.0 + dtz[k]*(khdz[k] + khdz[k+1])*rhoinv[k]
                cc[k] =  -dtz[k]*khdz[k+1]*rhoinv[k]
                d[k]  = sqi[k]
            end

            a[kte]  = 0.0
            b[kte]  = 1.0
            cc[kte] = 0.0
            d[kte]  = sqi[kte]

            tridiag2!(kte, a, b, cc, d, sqi2, work)
        else
            for k in kts:kte
                sqi2[k] = sqi[k]
            end
        end

        # ==================================== snow (sqs): hard-coded off (:4617)
        for k in kts:kte
            sqs2[k] = sqs[k]
        end

        # the five scalar solves (qni, qnc, qnwfa, qnifa, qnbca) are refused above;
        # their ELSE legs are the copies below.
        for k in kts:kte
            qni2[k]   = qni[k]
            qnc2[k]   = qnc[k]
            qnwfa2[k] = qnwfa[k]
            qnifa2[k] = qnifa[k]
            qnbca2[k] = qnbca[k]
        end

        # ==================================== ozone — local mixing only
        k = kts
        a[k]  =  -dtz[k]*khdz[k]*rhoinv[k]
        b[k]  = 1.0 + dtz[k]*(khdz[k+1] + khdz[k])*rhoinv[k]
        cc[k] =  -dtz[k]*khdz[k+1]*rhoinv[k]
        d[k]  = ozone[k]

        for k in (kts+1):(kte-1)
            a[k]  =  -dtz[k]*khdz[k]*rhoinv[k]
            b[k]  = 1.0 + dtz[k]*(khdz[k] + khdz[k+1])*rhoinv[k]
            cc[k] =  -dtz[k]*khdz[k+1]*rhoinv[k]
            d[k]  = ozone[k]
        end

        a[kte]  = 0.0
        b[kte]  = 1.0
        cc[kte] = 0.0
        d[kte]  = ozone[kte]

        tridiag2!(kte, a, b, cc, d, x, work)

        for k in kts:kte
            Dozone[k] = (x[k] - ozone[k])/delt
        end

        # ==================================== tendencies
        # (the bl_mynn_mixqt > 0 saturation back-out at :4896-4923 is refused above)

        # WATER VAPOR TENDENCY
        for k in kts:kte
            Dqv[k] = (sqv2[k] - sqv[k])/delt
        end

        if bl_mynn_cloudmix > 0
            # CLOUD WATER TENDENCY
            if FLAG_QC
                for k in kts:kte
                    Dqc[k] = (sqc2[k] - sqc[k])/delt
                end
            else
                for k in kts:kte
                    Dqc[k] = 0.0
                end
            end
            # CLOUD WATER NUM CONC TENDENCY (FLAG_QNC is false)
            for k in kts:kte
                Dqnc[k] = 0.0
            end
            # CLOUD ICE TENDENCY
            if FLAG_QI
                for k in kts:kte
                    Dqi[k] = (sqi2[k] - sqi[k])/delt
                end
            else
                for k in kts:kte
                    Dqi[k] = 0.0
                end
            end
            # CLOUD SNOW TENDENCY (disabled, :4981)
            for k in kts:kte
                Dqs[k] = 0.0
            end
            # CLOUD ICE NUM CONC TENDENCY (FLAG_QNI is false)
            for k in kts:kte
                Dqni[k] = 0.0
            end
        else
            # CLOUDS ARE NOT MIXED (bl_mynn_cloudmix == 0)
            for k in kts:kte
                Dqc[k]  = 0.0
                Dqnc[k] = 0.0
                Dqi[k]  = 0.0
                Dqni[k] = 0.0
                Dqs[k]  = 0.0
            end
        end

        # ensure non-negative moist species
        moisture_check!(kte, delt, delp, exner, sqv2, sqc2, sqi2, sqs2, thl,
                        Dqv, Dqc, Dqi, Dqs, Dth, c)

        # OZONE TENDENCY CHECK
        for k in kts:kte
            if Dozone[k]*delt + ozone[k] < 0.0
                Dozone[k] = -ozone[k]*0.99/delt
            end
        end

        # THETA TENDENCY. NOTE this OVERWRITES the dth that moisture_check! just
        # added to (:5032-5051); the correction survives only through `thl`, which
        # moisture_check! also wrote.
        if FLAG_QI
            for k in kts:kte
                Dth[k] = (thl[k] + c.xlvcp/exner[k]*sqc2[k] +
                                   c.xlscp/exner[k]*(sqi2[k]) - th[k])/delt
            end
        else
            for k in kts:kte
                Dth[k] = (thl[k] + c.xlvcp/exner[k]*sqc2[k] - th[k])/delt
            end
        end

        # AEROSOL TENDENCIES (FLAG_QNWFA/QNIFA false)
        for k in kts:kte
            Dqnwfa[k] = 0.0
            Dqnifa[k] = 0.0
        end
        # BLACK-CARBON TENDENCIES (FLAG_QNBCA false)
        for k in kts:kte
            Dqnbca[k] = 0.0
        end
    end
    return nothing
end

# ── The per-column driver: mode B of the reference harness ───────────────────
#
# `mynn_bl_driver` (:600-1400) is a 3-D routine that loops over columns; what the
# reference harness calls "mode B" is its per-column body, replayed call by call so
# that every intermediate is visible (tools/mynn_fortran_driver/ref_driver.f90). The
# three types and the two functions below are that body, in Julia, for ONE column.
#
# The mass-flux plumes (`DMP_mf`, :5700-6820) are a later stage: `mynn_column_step!`
# takes `edmf` and raises when it is `true`, and with `edmf = false` every plume sum
# (`s_aw*`, `sd_aw*`, `sub_*`, `det_*`, `edmf_a1`, `edmf_w1`) stays identically zero,
# which is exactly what the Fortran produces on a column with no active plumes.

"""
    MYNNColumn

One frozen input column, in the units and conventions of `mynnedmf_wrapper`:
`sqv`/`sqc`/`sqi` are SPECIFIC contents (`rho_x/rho_t`), `rho` is the moist density,
`ts` is `T_sfc/exner(1)` (which the driver divides by `exner(1)` AGAIN — harness
README item 2), and `zw` is the wall-height array of length `n+1`.

Build one with the keyword constructor; `zw` defaults to the driver's
`zw(1) = 0, zw(k) = zw(k-1) + dz(k-1)` (:1006-1010) and `uoce`/`voce` to zero.

`z`, `qsfc` and `znt` are carried for provenance and are not read by anything here.
"""
struct MYNNColumn
    n::Int
    ps::Float64
    ts::Float64
    qsfc::Float64
    ust::Float64
    hfx::Float64
    qfx::Float64
    wspd::Float64
    znt::Float64
    xland::Float64
    dx::Float64
    rmol0::Float64
    uoce::Float64
    voce::Float64
    z::Vector{Float64}
    dz::Vector{Float64}
    zw::Vector{Float64}
    u::Vector{Float64}
    v::Vector{Float64}
    w::Vector{Float64}
    T::Vector{Float64}
    th::Vector{Float64}
    exner::Vector{Float64}
    p::Vector{Float64}
    rho::Vector{Float64}
    sqv::Vector{Float64}
    sqc::Vector{Float64}
    sqi::Vector{Float64}
end

"""
    mynn_wall_heights(dz) -> Vector{Float64}

The wall heights `zw` of a column with layer thicknesses `dz`: `zw[1] = 0`,
`zw[k] = zw[k-1] + dz[k-1]`, length `n+1` (`mynn_bl_driver` :1006-1010).
"""
function mynn_wall_heights(dz::Vector{Float64})
    n = length(dz)
    zw = zeros(Float64, n + 1)
    for k in 2:n
        zw[k] = zw[k-1] + dz[k-1]
    end
    zw[n+1] = zw[n] + dz[n]
    return zw
end

function MYNNColumn(; ps::Real, ts::Real, ust::Real, hfx::Real, qfx::Real,
                      wspd::Real, xland::Real, dx::Real, rmol0::Real,
                      dz::Vector{Float64}, u::Vector{Float64}, v::Vector{Float64},
                      w::Vector{Float64}, T::Vector{Float64}, th::Vector{Float64},
                      exner::Vector{Float64}, p::Vector{Float64},
                      rho::Vector{Float64}, sqv::Vector{Float64},
                      sqc::Vector{Float64}, sqi::Vector{Float64},
                      z::Vector{Float64} = zeros(Float64, length(dz)),
                      zw::Union{Nothing,Vector{Float64}} = nothing,
                      qsfc::Real = 0.0, znt::Real = 0.0,
                      uoce::Real = 0.0, voce::Real = 0.0)
    n = length(dz)
    zwv = zw === nothing ? mynn_wall_heights(dz) : zw
    length(zwv) == n + 1 || throw(ArgumentError("MYNNColumn: zw must be n+1 long"))
    for (nm, a) in (("u", u), ("v", v), ("w", w), ("T", T), ("th", th),
                    ("exner", exner), ("p", p), ("rho", rho), ("sqv", sqv),
                    ("sqc", sqc), ("sqi", sqi), ("z", z))
        length(a) == n || throw(ArgumentError("MYNNColumn: $nm must be n = $n long"))
    end
    return MYNNColumn(n, Float64(ps), Float64(ts), Float64(qsfc), Float64(ust),
                      Float64(hfx), Float64(qfx), Float64(wspd), Float64(znt),
                      Float64(xland), Float64(dx), Float64(rmol0), Float64(uoce),
                      Float64(voce), z, dz, zwv, u, v, w, T, th, exner, p, rho,
                      sqv, sqc, sqi)
end

"""
    MYNNColumnState(n)

The column state `mynn_bl_driver` carries between calls: the TKE `qke`, the mixing
length `el`, the stability functions `sh`/`sm`, the variances `tsq`/`qsq`/`cov`, the
subgrid cloud `cldfra_bl`/`qc_bl`/`qi_bl`, the buoyancy coefficients `vt`/`vq`, the
cloud-PDF width `sgm`, and the diagnosed `pblh`/`kpbl`/`rmol`.

`vt`, `vq` and `sgm` are the automatic arrays the Fortran zeroes only in its init
block (harness README item 6); they are carried here for the same reason.
"""
mutable struct MYNNColumnState
    n::Int
    qke::Vector{Float64}
    el::Vector{Float64}
    sh::Vector{Float64}
    sm::Vector{Float64}
    tsq::Vector{Float64}
    qsq::Vector{Float64}
    cov::Vector{Float64}
    cldfra_bl::Vector{Float64}
    qc_bl::Vector{Float64}
    qi_bl::Vector{Float64}
    vt::Vector{Float64}
    vq::Vector{Float64}
    sgm::Vector{Float64}
    pblh::Float64
    kpbl::Int
    rmol::Float64
end

function MYNNColumnState(n::Integer)
    n = Int(n)
    z() = zeros(Float64, n)
    return MYNNColumnState(n, z(), z(), z(), z(), z(), z(), z(), z(), z(), z(),
                           z(), z(), z(), 0.0, 0, 0.0)
end

"""
    MYNNOptions(; kwargs...)

The namelist switches `mynn_bl_driver` takes. The defaults are the reference driver's
(`ref_driver.f90` mode_b / mode_a): closure 2.5, `delt = 20 s`, `bl_mynn_mixlength = 2`,
`bl_mynn_cloudpdf = 2`, `tke_budget = 0`, `bl_mynn_edmf_tke = 0`, `cloudmix = 1`,
`mixqt = 0`, `edmf = 1`, `edmf_mom = 1`, `mixscalars = 1`, `spp_pbl = 0`,
`dheat_opt = 1`, `flag_qc = flag_qi = true`.
"""
Base.@kwdef struct MYNNOptions
    closure::Float64            = 2.5
    delt::Float64               = 20.0
    bl_mynn_mixlength::Int      = 2
    bl_mynn_cloudpdf::Int       = 2
    tke_budget::Int             = 0
    bl_mynn_edmf_tke::Int       = 0
    bl_mynn_cloudmix::Int       = 1
    bl_mynn_mixqt::Int          = 0
    bl_mynn_edmf::Int           = 1
    bl_mynn_edmf_mom::Int       = 1
    bl_mynn_mixscalars::Int     = 1
    spp_pbl::Int                = 0
    dheat_opt::Int              = MYNN_DHEAT_OPT
    flag_qc::Bool               = true
    flag_qi::Bool               = true
    # ── the fidelity deviations the OFFLINE path can carry (MYNN_DEVIATIONS) ──
    # The wired model reads them off `MYNNState.fidelity`; the replay harness has no
    # `MYNNState`, so the four deviations that live inside `mynn_column_step!` and the
    # kernels it calls are namelist switches here. Every default is the Fortran, so a
    # parity call that names none of them is bitwise what it always was.
    # (`:K_interface`, `:pdk1` and `:rmol_sfc` are COUPLING deviations -- they live in
    # src/mc_mynn_bl.jl, which the offline path does not run -- so they are absent here.)
    gtr_local::Bool             = false
    sqfac::Float64              = MYNN_SQFAC
    exner_single::Bool          = false
    flux_clip::Bool             = false
end

# Re-gather the frozen column into the working copies (mynn_bl_driver :1004-1013 and
# ref_driver.f90 mode_b). `mynn_tendencies!` and `moisture_check!` WRITE thl (and
# would write sqv/sqc/sqi/sqw if mixqt were on), so a replay that handed them the
# host column directly would let the column evolve — harness README item 1.
@inline function _mynn_gather!(work::MYNNWork, col::MYNNColumn, c::MYNNConstants)
    n = col.n
    @inbounds for k in 1:n
        work.s_u[k]  = col.u[k];   work.s_v[k]   = col.v[k]
        work.s_w[k]  = col.w[k];   work.s_th[k]  = col.th[k]
        work.s_tk[k] = col.T[k]
        work.s_sqv[k] = col.sqv[k]
        work.s_sqc[k] = col.sqc[k]
        work.s_sqi[k] = col.sqi[k]
    end
    @inbounds for k in 1:n
        work.s_sqw[k]    = work.s_sqv[k] + work.s_sqc[k] + work.s_sqi[k]
        work.s_thl[k]    = work.s_th[k] - c.xlvcp/col.exner[k]*work.s_sqc[k] -
                                          c.xlscp/col.exner[k]*work.s_sqi[k]
        work.s_thetav[k] = work.s_th[k]*(1.0 + c.p608*work.s_sqv[k])
        work.s_qv[k]     = work.s_sqv[k]/(1.0 - work.s_sqv[k])
        work.s_qc[k]     = work.s_sqc[k]/(1.0 - work.s_sqv[k])
        work.s_qi[k]     = work.s_sqi[k]/(1.0 - work.s_sqv[k])
        # `g/theta_v` per level, for the `:gtr_local` deviation. Filled unconditionally
        # on the OFFLINE path (a replay is not a hot loop) and read only when the caller
        # passes `work.gtr_k` on as the `gtr_k` keyword; under `:fortran` nothing reads
        # it, so it changes no result.
        work.gtr_k[k]    = c.grav/work.s_thetav[k]
    end
    return nothing
end

"""
    mynn_init_column!(work, c::MYNNConstants, col::MYNNColumn, st::MYNNColumnState,
                      opts::MYNNOptions) -> (Psig_bl, Psig_shcu)

The cold start: zero the carried state, lay down the `5*ust` TKE taper, diagnose the
PBL height, get the scale-awareness factors and run `mym_initialize!`. This is
`mynn_bl_driver`'s `initflag = 1` block (:660-830), i.e. the init block of the
reference harness's mode B.

Two things a reader will trip over, both deliberate:

  * the first-guess `qke(k) = 5*ust*max((ust*700 - zw(k))/(max(ust,0.01)*700), 0.01)`
    is NOT the taper `mym_initialize!` then rebuilds (`INITIALIZE_QKE = true`
    overwrites the whole column); it exists only so `GET_PBLH` has a TKE profile.
  * `mym_initialize!` is handed `sqv`, not `sqw`, as its total-water argument (:817)
    — cloud and ice are excluded from `q_w` for the cold start only (harness README
    item 9).

Returns the two scale-awareness factors, which the caller may keep for reference;
`mynn_column_step!` recomputes them itself every step, as the driver does.
"""
function mynn_init_column!(work::MYNNWork, c::MYNNConstants, col::MYNNColumn,
                           st::MYNNColumnState, opts::MYNNOptions)
    n = col.n
    (work.n == n && st.n == n) ||
        throw(ArgumentError("mynn_init_column!: work/state sized for " *
                            "$(work.n)/$(st.n), column has n = $n"))
    kts = 1; kte = n

    fill!(st.el, 0.0);  fill!(st.sh, 0.0);  fill!(st.sm, 0.0)
    fill!(st.tsq, 0.0); fill!(st.qsq, 0.0); fill!(st.cov, 0.0)
    fill!(st.vt, 0.0);  fill!(st.vq, 0.0);  fill!(st.sgm, 0.0)
    fill!(st.cldfra_bl, 0.0); fill!(st.qc_bl, 0.0); fill!(st.qi_bl, 0.0)
    st.pblh = 0.0
    st.kpbl = 0
    st.rmol = col.rmol0

    _mynn_gather!(work, col, c)

    @inbounds for k in kts:kte
        st.qke[k] = 5.0*col.ust*max((col.ust*700.0 - col.zw[k])/
                                    (max(col.ust, 0.01)*700.0), 0.01)
    end

    zi, kzi = get_pblh!(kts, kte, work.s_thetav, st.qke, col.zw, col.dz, col.xland)
    st.pblh = zi
    st.kpbl = kzi
    Psig_bl, Psig_shcu = scale_aware(col.dx, zi)

    mym_initialize!(kts, kte, col.xland, col.dz, col.dx, col.zw,
                    work.s_u, work.s_v, work.s_thl, work.s_sqv,
                    st.pblh, work.s_th, work.s_thetav, st.sh, st.sm,
                    col.ust, st.rmol, st.el, st.qke, st.tsq, st.qsq, st.cov,
                    Psig_bl, st.cldfra_bl, opts.bl_mynn_mixlength,
                    work.z_n, work.z_n, true, c, work)

    return (Psig_bl, Psig_shcu)
end

"""
    mynn_column_step!(work, c::MYNNConstants, col::MYNNColumn, st::MYNNColumnState,
                      opts::MYNNOptions; edmf::Bool) -> (Psig_bl, Psig_shcu)

One boundary-layer time step on one frozen column, reproducing the reference
harness's mode-B call sequence exactly (`ref_driver.f90` mode_b, mirroring
`mynn_bl_driver` :900-1330):

    re-gather the frozen column
      -> GET_PBLH -> SCALE_AWARE
      -> surface fluxes / rmol / zet / pmz / phh
      -> mym_condensation!
      -> [DMP_mf — NOT ported; with edmf = false every plume sum stays zero]
      -> mym_turbulence!
      -> mym_predict!
      -> diss_heat (mynn_bl_driver :1224-1234)
      -> mynn_tendencies!
      -> retrieve_exchange_coeffs!

`edmf = true` RAISES: `DMP_mf` (:5700-6820) is a later stage. Every quantity the
plumes would supply — `s_aw`, `s_awthl`, `s_awqt`, `s_awqv`, `s_awqc`, `s_awu`,
`s_awv`, `s_awqke`, the `sd_aw*` downdraft set, `sub_*`, `det_*`, `edmf_a1`,
`edmf_w1` — is the shared zero buffer `work.z_n`/`work.z_np1`, which is what the
Fortran produces for a column whose `ktop_plume` is 0.

The carried state `st` is updated in place. The per-step OUTPUTS live in `work`,
because the Fortran's caller owns them rather than the scheme:

| output                                          | field                           |
|:------------------------------------------------|:--------------------------------|
| `du, dv, dth, dqv, dqc, dqi`                     | `work.out_du` … `work.out_dqi`  |
| `K_m, K_h` (the driver's `exch_m`, `exch_h`)     | `work.out_km`, `work.out_kh`    |
| `dfm, dfh, dfq, tcd, qcd, pdk, pdt, pdq, pdc`    | `work.out_dfm` …                |
| `diss_heat`                                      | `work.out_diss_heat`            |
| the `tke_budget` arrays (untouched at budget 0)  | `work.out_qwt` …                |

The surface block reproduces two harness quirks verbatim: `th_sfc = ts/exner(1)` even
though `ts` is ALREADY `T_sfc/exner(1)` (README item 2), and the drag denominator
`ust^2/wspd`, which needs `wspd > 0` even in a resting column (README item 3).

Returns the two scale-awareness factors of this step.
"""
function mynn_column_step!(work::MYNNWork, c::MYNNConstants, col::MYNNColumn,
                           st::MYNNColumnState, opts::MYNNOptions; edmf::Bool, ework = nothing)
    # S6 hook: `ework` is an EDMFWork (src/mynn_edmf.jl, included after this file, hence
    # unannotated). The bodies stay duplicated on purpose; the test "reproduces the
    # edmf = false path bitwise where no plume fires" keeps them honest.
    if edmf
        ework === nothing && throw(ArgumentError("mynn_column_step!: edmf = true needs " *
                                                 "an EDMFWork (src/mynn_edmf.jl)"))
        return mynn_column_step_edmf!(work, ework, c, col, st, opts)[1:2]
    end
    n = col.n
    (work.n == n && st.n == n) ||
        throw(ArgumentError("mynn_column_step!: work/state sized for " *
                            "$(work.n)/$(st.n), column has n = $n"))
    kts = 1; kte = n
    zn  = work.z_n
    zn1 = work.z_np1

    _mynn_gather!(work, col, c)

    zi, kzi = get_pblh!(kts, kte, work.s_thetav, st.qke, col.zw, col.dz, col.xland)
    st.pblh = zi
    st.kpbl = kzi
    Psig_bl, Psig_shcu = scale_aware(col.dx, zi)

    # -- surface fluxes and stability functions (mynn_bl_driver :1060-1097) --------
    # The four OFFLINE fidelity deviations (MYNN_DEVIATIONS) enter here and in the
    # keywords below. `hfx`/`qfx` are the wrapper-clipped fluxes under `:flux_clip` and
    # the column's own otherwise; `th_sfc` divides by the surface Exner function ONCE
    # under `:exner_single` -- and `col.ts` IS already `T_sfc/exner(1)` (harness README
    # item 2), so the single-division form is `col.ts` itself.
    hfx    = opts.flux_clip ? clamp(col.hfx, MYNN_HFX_MIN, MYNN_HFX_MAX) : col.hfx
    qfx    = opts.flux_clip ? clamp(col.qfx, MYNN_QFX_MIN, MYNN_QFX_MAX) : col.qfx
    gtr_k  = opts.gtr_local ? work.gtr_k : nothing
    cpm    = c.cp*(1.0 + 0.84*work.s_qv[kts])
    flqv   = qfx/col.rho[kts]
    flqc   = 0.0
    th_sfc = opts.exner_single ? col.ts : col.ts/col.exner[kts]
    flq    = flqv + flqc
    flt    = hfx/(col.rho[kts]*cpm) - c.xlvcp*flqc/col.exner[kts]
    fltv   = flt + flqv*c.p608*th_sfc
    rmol   = -c.karman*_gtr_lev(gtr_k, c, kts)*fltv/max(col.ust^3, 1.0e-6)
    zet    = 0.5*col.dz[kts]*rmol
    zet    = max(zet, -20.0)
    zet    = min(zet,  20.0)
    phi_m  = phim(zet)
    pmz    = phi_m - zet
    phh    = phih(zet)
    st.rmol = rmol

    mym_condensation!(kts, kte, col.dx, col.dz, col.zw, col.xland,
                      work.s_thl, work.s_sqw, work.s_sqv, work.s_sqc, work.s_sqi, zn,
                      col.p, col.exner, st.tsq, st.qsq, st.cov, st.sh, st.el,
                      opts.bl_mynn_cloudpdf, st.qc_bl, st.qi_bl, st.cldfra_bl,
                      st.pblh, hfx, st.vt, st.vq, work.s_th, st.sgm, st.rmol,
                      opts.spp_pbl, zn, c, work)

    # DMP_mf would run here; with edmf = false every plume sum stays zero.

    mym_turbulence!(kts, kte, col.xland, opts.closure, col.dz, col.dx, col.zw,
                    work.s_u, work.s_v, work.s_thl, work.s_thetav, work.s_sqc,
                    work.s_sqw, st.qke, st.tsq, st.qsq, st.cov, st.vt, st.vq,
                    st.rmol, flt, fltv, flq, st.pblh, work.s_th,
                    st.sh, st.sm, st.el,
                    work.out_dfm, work.out_dfh, work.out_dfq,
                    work.out_tcd, work.out_qcd,
                    work.out_pdk, work.out_pdt, work.out_pdq, work.out_pdc,
                    work.out_qwt, work.out_qshear, work.out_qbuoy, work.out_qdiss,
                    opts.tke_budget, Psig_bl, Psig_shcu, st.cldfra_bl,
                    opts.bl_mynn_mixlength, zn, zn, zn, opts.spp_pbl, zn, c, work;
                    gtr_k = gtr_k)

    mym_predict!(kts, kte, opts.closure, opts.delt, col.dz, col.ust, flt, flq,
                 pmz, phh, st.el, work.out_dfq, col.rho,
                 work.out_pdk, work.out_pdt, work.out_pdq, work.out_pdc,
                 st.qke, st.tsq, st.qsq, st.cov, zn1, zn1, opts.bl_mynn_edmf_tke,
                 work.out_qwt, work.out_qdiss, opts.tke_budget, c, work;
                 sqfac = opts.sqfac)

    # -- dissipative heating (mynn_bl_driver :1224-1234) --------------------------
    dh = work.out_diss_heat
    if opts.dheat_opt > 0
        @inbounds for k in kts:(kte-1)
            # Set the max dissipative heating rate to 7.2 K per hour
            dh[k] = min(max(1.0*(st.qke[k]^1.5)/
                            (MYNN_B1*max(0.5*(st.el[k] + st.el[k+1]), 1.0))/c.cp,
                            0.0), 0.002)
            # Limit the heating above 100 mb
            dh[k] = dh[k] * exp(-10000.0/max(col.p[k], 1.0))
        end
        @inbounds dh[kte] = 0.0
    else
        fill!(dh, 0.0)
    end

    mynn_tendencies!(kts, kte, 1, opts.delt, col.dz, col.rho,
                     work.s_u, work.s_v, work.s_th, work.s_tk,
                     work.s_qv, work.s_qc, work.s_qi, zn, zn, zn,
                     col.ps, col.p, col.exner,
                     work.s_thl, work.s_sqv, work.s_sqc, work.s_sqi, zn, work.s_sqw,
                     zn, zn, zn, zn,
                     col.ust, flt, flq, flqv, flqc, col.wspd, col.uoce, col.voce,
                     st.tsq, st.qsq, st.cov, work.out_tcd, work.out_qcd,
                     work.out_dfm, work.out_dfh, work.out_dfq,
                     work.out_du, work.out_dv, work.out_dth, work.out_dqv,
                     work.out_dqc, work.out_dqi, work.out_dqs,
                     work.out_dqnc, work.out_dqni,
                     work.out_dqnwfa, work.out_dqnifa, work.out_dqnbca,
                     work.out_dozone, dh,
                     zn1, zn1, zn1, zn1, zn1, zn1, zn1, zn1, zn1, zn1, zn1, zn1,
                     zn1, zn1, zn1, zn1, zn1, zn1, zn1,
                     zn, zn, zn, zn, zn, zn, zn, zn, zn,
                     opts.flag_qc, opts.flag_qi, false, false, false, false, false,
                     false, st.cldfra_bl,
                     opts.bl_mynn_cloudmix, opts.bl_mynn_mixqt, opts.bl_mynn_edmf,
                     opts.bl_mynn_edmf_mom, opts.bl_mynn_mixscalars, c, work)

    retrieve_exchange_coeffs!(kts, kte, work.out_dfm, work.out_dfh, col.dz,
                              work.out_km, work.out_kh)

    return (Psig_bl, Psig_shcu)
end
