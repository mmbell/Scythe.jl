# ── MYNN-EDMF mass flux: DMP_mf, condensation_edmf and the EDMF column step ────
#
# Part 3 of the pure-Julia port of the MYNN-EDMF boundary-layer scheme
# (ccpp-physics `module_bl_mynn.F90`, verbatim copy under tools/mynn_fortran_driver/).
# Parts 1 and 2 are src/mynn_constants.jl and src/mynn_closure.jl; every convention
# stated in the header of src/mynn_closure.jl ("Conventions": indexing, staggering,
# floating point, allocation) applies here unchanged and is not repeated.
#
# WHAT IS PORTED. `DMP_mf` (:5680-6826) and `condensation_edmf` (:6830-6887) for the
# argument set the reference harness uses:
#
#     momentum_opt = 1, tke_opt = 0, scalar_opt = 1 (with all-zero scalar columns),
#     F_QC = F_QI = .true., F_QNC = F_QNI = F_QNWFA = F_QNIFA = F_QNBCA = .false.,
#     mix_chem = .false., spp_pbl = 0, nchem = ndvel = 0.
#
# WHAT IS LEFT OUT, deliberately, each with a one-line reason:
#
#   * CHEMISTRY / SMOKE (`mix_chem`, `UPCHEM`, `s_awchem`, `edmf_chem`, `chemn`,
#     :5807-5813, :5904, :5922, :6151-6167, :6235-6242, :6377, :6428-6436, :6541-6546).
#     `mix_chem = .false.` in the reference and in Scythe; the branches are entirely
#     guarded and touch nothing else. `dmp_mf!` has no `nchem` argument at all.
#   * The AEROSOL / NUMBER-CONCENTRATION plumes `UPQNC, UPQNI, UPQNWFA, UPQNIFA,
#     UPQNBCA` and their sums `s_awqnc … s_awqnbca` (:5787-5789, :6125-6130,
#     :6215-6220, :6362-6371, :6449-6459). The Fortran computes them unconditionally,
#     but their sole input is `qnc/qni/qnwfa/qnifa/qnbca`, which the reference driver
#     passes as ZERO columns and which Scythe has no source for; every one of them is
#     then identically zero at every level and NOTHING reads them back — they leave
#     `DMP_mf` only through `s_awqn*`, which `mynn_tendencies` multiplies by the
#     (false) `FLAG_QNC…` switches. Carrying five all-zero (n+1, 8) matrices would be
#     pure cost. The flux limiter's `s_awqn* = s_awqn**adjustment` lines are dropped
#     for the same reason.
#   * `ENTf` (real) and `ENTi` (integer), :5793-5794. DECLARED AND NEVER USED — a
#     leftover of the stochastic-entrainment formulation that :6182-6199 replaced.
#     No array is allocated for them here.
#   * `s_aw2` (:5779), `ERF` (:5824), `wlv` (:6122), `qsl` (:6636), `UPQV` (:5785,
#     zeroed at :5877 and never written or read again), `edmf_a1`'s `sgm` argument
#     (`intent(inout)` at :5832 but referenced only inside a commented-out `print`):
#     all dead. `wlv` and `qsl` are transcribed as comments, not as code; `sgm` is
#     still an argument of `dmp_mf!` so the call site matches the Fortran, and it is
#     never touched.
#   * `debug_mf` (= 0) and the unconditional `if (edmf_w(1) > 4.0)` debug print
#     (:6779-6822).
#
# ── Fortran constructs this port had to INTERPRET (all line numbers in
#    tools/mynn_fortran_driver/module_bl_mynn.F90) ───────────────────────────────
#
#  (a) `k50` (:5945) is READ at :5977 but is only ASSIGNED inside the taper loop,
#      under `if (ZW(k)<=50.)`. It is never initialised. In practice `zw(kts) = 0`
#      and the loop's `exit` test `zw(k) > pblh + 500.` cannot fire at k = kts
#      (pblh >= 0), so `k50 >= 1` always and the read is safe; this port initialises
#      it to `kts` and says so, rather than reproducing an uninitialised read.
#  (b) `dzp` (:6316-6318) is assigned in the `Wn <= 0 .and. overshoot == 0` branch
#      ONLY when `THVk - THVkm1 > 0`; otherwise it keeps whatever it held from the
#      previous k (or, on the first k of the first plume, is UNDEFINED). It is read
#      three lines later by the Asai-Kasahara environment update. That update writes
#      only `envm_*`, which nothing reads unless `env_subs` is true — so on this
#      configuration `dzp` is dead. Initialised to 0 here; see (c).
#  (c) `env_subs` is a compile-time `.false.` parameter (:337). The ENTIRE
#      subsidence/detrainment block (:6549-6620) is therefore unreachable, and with
#      it every `envm_*` and `envi_*` array and `sub_thl … det_v` output, which the
#      reference prints as all-zero for every case and every step. It is transcribed
#      below anyway (the task asks for it) but it is UNTESTED and carries a Fortran
#      out-of-bounds read: `sub_thl(kts)` (:6586) divides by `rhoz(k)` where `k` is
#      the loop variable LEFT OVER from the `DO k=kts,kte` transform loop above, i.e.
#      `k = kte+1`, one past the end of `rhoz(kts:kte)`. The same stale `k` appears in
#      `sub_sqv(kts)`, `sub_u(kts)` and `sub_v(kts)`. This port CLAMPS that index to
#      `kte` (marked `# FORTRAN OOB` below) — it cannot reproduce reading past the end
#      of an array, and no reference number depends on it.
#  (d) `wmin` is REUSED (:6182) as the per-plume entrainment floor `0.3 + l*0.0005`
#      after having been the surface-updraft `MIN(sigmaW*pwmin, 0.1)` (:6115). The
#      surface value is fully consumed at :6125 before the reuse, so the shadowing is
#      harmless; it is spelled with two names here (`wmin_sfc`, `wmin`) for the reader.
#  (e) `NUP2` is a REAL holding an INTEGER count (:5773, `nup2 = nup`). Its only live
#      uses are the `nup2 > 0` gate at :6389 and the "plume failed to leave the
#      surface" reset at :6288. That reset (`k == kts+1 .and. Wn == 0` -> `NUP2 = 0`,
#      `exit`) leaves the *outer* plume loop running: plumes i+1 … NUP are still
#      integrated and can still raise `ktop`, but `nup2` stays 0, so NO flux is
#      computed and `edmf_*` stay zero while `ktop`/`ztop` are nonzero. Reproduced
#      verbatim; `nup2` is an `Int` here.
#  (f) `EntExp = ENT*(ZW(k+1)-ZW(k))` (:6208) uses the layer ABOVE the interface,
#      while the w-equation's step `MIN(ZW(k)-ZW(k-1), 250.)` (:6270-6274) uses the
#      layer BELOW it. Both are transcribed as written.
#  (g) `ENT` is preset to 0.001 (:5875) and written only at the levels a plume
#      actually reaches; above `ktop` it keeps 0.001, which `edmf_ent` then multiplies
#      by `UPA = 0`. Harmless, but it means `ENT` must be RE-PRESET on every call, not
#      just allocated once — `dmp_mf!` fills it, see `_edmf_reset!`.
#  (h) `IF (s_aw(kts+1) /= 0.)` (:6474) sets `dzi(kts)`; the `ELSE` leaves it
#      undefined but also sets `flx1 = 0`, so it is not read. `dzi(kts)` is written
#      again unconditionally at :6626 with the same expression.
#
# ── COVERAGE: what the MAIN reference does not reach, and what does ───────────
# The recomputation of `vt, vq, cldfra_bl1d, qc_bl1d` at :6631-6766 runs only where
# `0.5*(edmf_qc(k)+edmf_qc(k-1)) > 0`. In `ref_driver_output_r8.txt` `edmf_qc` is
# IDENTICALLY ZERO at every level of every case and both steps (case 4's plumes top
# out at 653 m, well below any condensation, and `condensation_edmf` additionally
# forces `QC = 0` below 100 m). So :6631-6766 — roughly 140 lines: `Aup/THp/QTp/QCp`,
# the CB02 `a/b9`, `sigq`, `Q1`, `mf_cf`, the `qc_bl1d` fits and the `Fng` piecewise
# buoyancy update — gets nothing from the main reference, and neither does the
# `maxqc >= 1e-8` (moist plume) branch of the `maxmf` sign flip (:6771-6775) nor any
# `landsea < 1.5` (LAND) branch, all five reference columns being water (`xland = 2`).
# tools/mynn_fortran_driver/edmf_ref_driver.f90 exists for exactly that gap: it calls
# `DMP_mf` directly on a synthetic saturating column, once over water and once over
# land, and test/test_mynn_edmf.jl checks this port against its output (bitwise, at
# 22 plume levels with cloud at 19 of them).
# WHAT REMAINS UNTESTED: the environmental subsidence/detrainment block (:6549-6620),
# which `env_subs = .false.` makes unreachable — with it `dzp`, every `envm_*`/`envi_*`
# array and the nine `sub_*`/`det_*` outputs (which the reference does pin, as
# identically zero). See note (c).

# ── Work space ────────────────────────────────────────────────────────────────

"""
    MYNN_NUP

Number of plumes in the Neggers size distribution, `integer, parameter :: nup = 8`
(module_bl_mynn.F90 :5771). Fixed at compile time in the Fortran and fixed here.
"""
const MYNN_NUP = 8

# DMP_mf's own `parameter`s (:5799-5817, :5843, :5859-5872, :6083-6085), each with its
# Fortran line. Kept as file-local `const`s rather than exported names: they are
# private to the mass-flux scheme and several of them (`Wa`, `Wb`, `Wc`, `L0`, `ENT0`)
# collide with common short names.
const MYNN_MF_WA        = 2.0/3.0   # :5801  (unused: the Simpson-Wiggert form below
const MYNN_MF_WB        = 0.002     # :5802   replaced the StEM w-equation, :6262-6268)
const MYNN_MF_WC        = 1.5       # :5803
const MYNN_MF_L0        = 100.0     # :5808  (unused: Suselj entrainment, :6185-6188)
const MYNN_MF_ENT0      = 0.1       # :5809
const MYNN_MF_ATOT      = 0.10      # :5812  max total fractional area of all updrafts
const MYNN_MF_LMAX      = 1000.0    # :5813  diameter of the largest plume (m)
const MYNN_MF_LMIN      = 300.0     # :5814  diameter of the smallest plume (m)
const MYNN_MF_DLMIN     = 0.0       # :5815  increase of the smallest plume at large flux
const MYNN_MF_DCUT      = 1.2       # :5818  max plume diameter relative to dx
const MYNN_MF_CF_THRESH = 0.5       # :5843  only overwrite stratus CF below this
const MYNN_MF_CDET      = 1.0/45.0  # :5859
const MYNN_MF_DZPMAX    = 300.0     # :5860  cap on dz in the detrainment
const MYNN_MF_CSUB      = 0.25      # :5865  portion of plume w that subsides
const MYNN_MF_PGFAC     = 0.00      # :5868  pressure-gradient factor on momentum
const MYNN_MF_FLUXPORTION = 0.75    # :5854  heat-flux limiter portion
const MYNN_MF_DEBUG     = 0         # :5771  debug_mf

"""
    EDMFWork(n; nup = MYNN_NUP)

Preallocated scratch for `dmp_mf!` on a column of `n` layers, plus the arrays the
Fortran's CALLER owns (the `edmf_*`, `s_aw*`, `sub_*` and `det_*` outputs). Two
sections, mirroring `MYNNWork`:

LOCALS — the automatic arrays `DMP_mf` declares (module_bl_mynn.F90 :5782-5871):

| field                                             | Fortran                       |
|:--------------------------------------------------|:------------------------------|
| `upw, upthl, upqt, upqc, upa, upu, upv, upthv, upqke` | `(kts:kte+1, 1:NUP)` (:5783-5786) |
| `ent`                                              | `(kts:kte, 1:NUP)` (:5793)    |
| `rhoz`                                             | `(kts:kte)` (:5847)           |
| `exneri, dzi`                                      | `(kts:kte)` (:5847)           |
| `edmf_th`                                          | `(kts:kte)` (:5747)           |
| `envm_thl, envm_sqv, envm_sqc, envm_u, envm_v`     | `(kts:kte)` (:5872-5874)      |
| `envi_a, envi_w`                                   | `(kts:kte+1)` (:5875)         |

`UPQV` is not carried (dead, see the file header), and neither are the five
number-concentration plume matrices nor `ENTf`/`ENTi`.

OUTPUTS — what a host allocates and `DMP_mf` fills. They live here, not in
`MYNNWork`, so that a column stepped with `edmf = false` pays nothing for them:

| field                                                          | length |
|:---------------------------------------------------------------|:-------|
| `edmf_a, edmf_w, edmf_qt, edmf_thl, edmf_ent, edmf_qc`          | `n`    |
| `s_aw, s_awthl, s_awqt, s_awqv, s_awqc, s_awu, s_awv, s_awqke`  | `n+1`  |
| `sub_thl, sub_sqv, sub_u, sub_v`                                | `n`    |
| `det_thl, det_sqv, det_sqc, det_u, det_v`                       | `n`    |

The `sub_*`/`det_*` nine are always zero on this configuration (`env_subs = .false.`),
which is exactly what the reference prints; they exist so the `mynn_tendencies!` call
site matches the Fortran's.
"""
struct EDMFWork
    n::Int
    nup::Int
    # -- plume properties on interfaces, (n+1, nup) ----------------------------
    upw::Matrix{Float64}
    upthl::Matrix{Float64}
    upqt::Matrix{Float64}
    upqc::Matrix{Float64}
    upa::Matrix{Float64}
    upu::Matrix{Float64}
    upv::Matrix{Float64}
    upthv::Matrix{Float64}
    upqke::Matrix{Float64}
    # -- entrainment, (n, nup) -------------------------------------------------
    ent::Matrix{Float64}
    # -- layer/interface locals -------------------------------------------------
    rhoz::Vector{Float64}
    exneri::Vector{Float64}
    dzi::Vector{Float64}
    edmf_th::Vector{Float64}
    envm_thl::Vector{Float64}
    envm_sqv::Vector{Float64}
    envm_sqc::Vector{Float64}
    envm_u::Vector{Float64}
    envm_v::Vector{Float64}
    envi_a::Vector{Float64}
    envi_w::Vector{Float64}
    # -- outputs the caller owns -------------------------------------------------
    edmf_a::Vector{Float64}
    edmf_w::Vector{Float64}
    edmf_qt::Vector{Float64}
    edmf_thl::Vector{Float64}
    edmf_ent::Vector{Float64}
    edmf_qc::Vector{Float64}
    s_aw::Vector{Float64}
    s_awthl::Vector{Float64}
    s_awqt::Vector{Float64}
    s_awqv::Vector{Float64}
    s_awqc::Vector{Float64}
    s_awu::Vector{Float64}
    s_awv::Vector{Float64}
    s_awqke::Vector{Float64}
    sub_thl::Vector{Float64}
    sub_sqv::Vector{Float64}
    sub_u::Vector{Float64}
    sub_v::Vector{Float64}
    det_thl::Vector{Float64}
    det_sqv::Vector{Float64}
    det_sqc::Vector{Float64}
    det_u::Vector{Float64}
    det_v::Vector{Float64}
end

"""
    EDMF_WORK_FACE_FIELDS

The `EDMFWork` vector fields that live on the box walls and are therefore `n+1` long.
"""
const EDMF_WORK_FACE_FIELDS = (:envi_a, :envi_w, :s_aw, :s_awthl, :s_awqt, :s_awqv,
                               :s_awqc, :s_awu, :s_awv, :s_awqke)

function EDMFWork(n::Integer; nup::Integer = MYNN_NUP)
    n = Int(n); nup = Int(nup)
    n >= 4 || throw(ArgumentError("EDMFWork: DMP_mf needs n >= 4, got $n"))
    mats = (:upw, :upthl, :upqt, :upqc, :upa, :upu, :upv, :upthv, :upqke)
    args = Any[n, nup]
    for f in fieldnames(EDMFWork)[3:end]
        push!(args, f === :ent          ? zeros(Float64, n, nup) :
                    f in mats           ? zeros(Float64, n + 1, nup) :
                    f in EDMF_WORK_FACE_FIELDS ? zeros(Float64, n + 1) :
                                          zeros(Float64, n))
    end
    return EDMFWork(args...)
end

# `DMP_mf` :5859-5926: zero every plume matrix, preset ENT to 0.001, and zero every
# output. Note (g) in the file header: ENT must be RE-preset each call.
@inline function _edmf_reset!(w::EDMFWork)
    for M in (w.upw, w.upthl, w.upthv, w.upqt, w.upa, w.upu, w.upv, w.upqc, w.upqke)
        fill!(M, 0.0)
    end
    fill!(w.ent, 0.001)                                                   # :5875
    for v in (w.edmf_a, w.edmf_w, w.edmf_qt, w.edmf_thl, w.edmf_ent, w.edmf_qc,
              w.s_aw, w.s_awthl, w.s_awqt, w.s_awqv, w.s_awqc, w.s_awu, w.s_awv,
              w.s_awqke, w.sub_thl, w.sub_sqv, w.sub_u, w.sub_v,
              w.det_thl, w.det_sqv, w.det_sqc, w.det_u, w.det_v)
        fill!(v, 0.0)
    end
    return nothing
end

# ── condensation_edmf (module_bl_mynn.F90 :6830-6887) ─────────────────────────

"""
    condensation_edmf(qt, thl, p, zagl, qc_in, c::MYNNConstants) -> (thv, qc)

Zero-or-one condensation for a mass-flux plume parcel: given total water `qt`, liquid
water potential temperature `thl`, pressure `p` and height above ground `zagl`, return
the virtual potential temperature `thv` and the condensate `qc`.

`condensation_edmf` (:6830-6887) is a fixed-point iteration on
`T = exner*thl + xlvcp*qc`, `qc = 0.5*qc + 0.5*max(qt - qsat_blend(T,p), 0)`, at most
50 sweeps, stopping when `|qc - qc_old| < 1e-6`. `QC` is `intent(inout)`: the Fortran
comment at :6854 is explicit that the INCOMING value is the first guess, and `DMP_mf`
supplies the plume's condensate at the level below (:6247), resetting it to 0 only at
the bottom of each plume (:6172). `qc_in` is that first guess here; the returned `qc`
replaces it.

Two things the loop does that a reader will want to check twice:

  * the iteration's result is DISCARDED as a value and used only as the last `T`: the
    three lines after the loop recompute `T`, `qs` and then set `qc = max(qt-qs, 0)`
    outright (:6862-6864), so the 0.5-damping only ever moves the temperature.
  * `if (zagl < 100.) qc = 0` (:6867) — no plume condensate below 100 m, whatever the
    thermodynamics say. This is why the reference run's `edmf_qc` is identically zero:
    case 4's plumes top out at 653 m but never saturate.

`thv = (thl + xlvcp*qc)*(1 + qt*(rvovrd-1) - rvovrd*qc)` (:6870), NOT the
`th*(1+p608*qt)` of the rest of the scheme; the Fortran notes at :6880-6881 that the
two agree.
"""
@inline function condensation_edmf(qt::Float64, thl::Float64, p::Float64,
                                   zagl::Float64, qc_in::Float64, c::MYNNConstants)
    niter = 50                                                            # :6849
    diff  = 1.0e-6                                                        # :6851

    qc  = qc_in
    exn = (p/MYNN_P1000MB)^c.rcp                                          # :6853
    t   = 0.0
    for _ in 1:niter                                                      # :6855
        t = exn*thl + c.xlvcp*qc
        qs = qsat_blend(t, p, c)
        qcold = qc
        qc = 0.5*qc + 0.5*max((qt - qs), 0.0)
        if abs(qc - qcold) < diff
            break
        end
    end

    t = exn*thl + c.xlvcp*qc                                              # :6862
    qs = qsat_blend(t, p, c)
    qc = max(qt - qs, 0.0)

    if zagl < 100.0                                                       # :6867
        qc = 0.0
    end

    thv = (thl + c.xlvcp*qc)*(1.0 + qt*(c.rvovrd - 1.0) - c.rvovrd*qc)    # :6870
    return (thv, qc)
end

# ── DMP_mf (module_bl_mynn.F90 :5680-6826) ────────────────────────────────────

"""
    EDMFGate

What `dmp_mf!` returns: the four `intent(out)` scalars the Fortran declares plus the
activation diagnostics, so a caller can print WHY the scheme did or did not fire.

| field          | Fortran                     | meaning                                    |
|:---------------|:----------------------------|:-------------------------------------------|
| `maxwidth`     | `intent(out)` (:5751)       | diameter of the largest plume (m), 0 if off |
| `ktop`         | `intent(out)` (:5750)       | highest interface index a plume reached     |
| `maxmf`        | `intent(out)` (:5751)       | max `edmf_a*edmf_w`; NEGATIVE for a dry plume |
| `ztop`         | `intent(out)` (:5751)       | `zw(ktop)`, 0 when `ktop == 0`              |
| `active`       | :6039                       | the gate `fltv2 > 0.002 && maxwidth > minwidth && superadiabatic` |
| `fltv2`        | :5964                       | `fltv`, sign-flipped when `Psig_w == 0`     |
| `minwidth`     | :6031-6034                  | the smallest plume diameter actually used   |
| `superadiabatic` | :6011-6027                | the `dT/dz < hux` test through 50 m         |
| `psig_w`       | :5959-5960                  | resolved-w taper, `min(1-max(0,maxw-1), Psig_shcu)` |
| `cloud_base`   | :5951-5954                  | first level with `qc_sgs > 1e-5` and CF >= 0.5 (9000 m = none) |
| `k50`          | :5948                       | highest level with `zw <= 50 m`             |
| `nup2`         | :5773, :6288, :6386         | plume count actually used (0 = no flux)     |
| `adjustment`   | :6483-6501                  | the heat-flux limiter's column rescale      |

`active == false` is the only way `maxwidth` comes back as 0: :6035 zeroes it when
`maxwidth <= minwidth`.
"""
struct EDMFGate
    maxwidth::Float64
    ktop::Int
    maxmf::Float64
    ztop::Float64
    active::Bool
    fltv2::Float64
    minwidth::Float64
    superadiabatic::Bool
    psig_w::Float64
    cloud_base::Float64
    k50::Int
    nup2::Int
    adjustment::Float64
end

"""
    dmp_mf!(kts, kte, dt, zw, dz, p, rho,
            momentum_opt, tke_opt, scalar_opt,
            u, v, w, th, thl, thv, tk, qt, qv, qc, qke,
            exner, vt, vq, sgm,
            ust, flt, fltv, flq, flqv, pblh, kpbl, dx, landsea, ts,
            edmf_a, edmf_w, edmf_qt, edmf_thl, edmf_ent, edmf_qc,
            s_aw, s_awthl, s_awqt, s_awqv, s_awqc, s_awu, s_awv, s_awqke,
            sub_thl, sub_sqv, sub_u, sub_v,
            det_thl, det_sqv, det_sqc, det_u, det_v,
            qc_bl1d, cldfra_bl1d, qc_bl1d_old, cldfra_bl1d_old,
            F_QC, F_QI, Psig_shcu, spp_pbl, rstoch_col,
            c::MYNNConstants, work::EDMFWork) -> EDMFGate

The Neggers/Siebesma multi-plume mass-flux component of MYNN-EDMF, `DMP_mf`
(module_bl_mynn.F90 :5680-6826), for the reference harness's argument set
(`momentum_opt = 1, tke_opt = 0, scalar_opt = 1`, no chemistry, no aerosol,
`spp_pbl = 0`). See the file header for what is left out and why, and for the eight
Fortran constructs (a)-(h) that had to be interpreted.

THE SEQUENCE, with its Fortran line ranges:

 1. zero the plume matrices and the outputs, preset `ENT = 0.001`         :5859-5926
 2. resolved-w taper and cloud-base search over `zw <= pblh + 500 m`      :5938-5960
 3. the SUPERADIABATIC test through 50 m (verbatim, note (a))             :5966-5990
 4. the five plume-width criteria -> `maxwidth`, `minwidth`               :5992-6037
 5. THE GATE: `fltv2 > 0.002 .and. maxwidth > minwidth .and. superadiabatic`  :6039
 6. the Neggers `N = C l^d` size distribution and the area fractions      :6041-6078
 7. surface-layer initialisation of the 8 plumes from `sigmaW/QT/TH`      :6080-6141
 8. per-plume integration: entrainment, `condensation_edmf`, the
    Simpson-Wiggert w-equation with its +-1.25 m/s per 200 m limiter,
    the overshoot/Froude termination and the Asai-Kasahara environment
    update (note (b)/(c))                                                :6169-6366
 9. the interface sums `s_aw*` on `kts:kte+1`, then momentum and TKE      :6389-6459
10. the HEAT-FLUX LIMITER, which rescales `s_aw*` AND `UPA` column-wide   :6463-6501
11. the area-weighted means `edmf_*`, `maxmf`                             :6505-6528
12. the environmental subsidence/detrainment (unreachable, note (c))      :6549-6620
13. the Chaboureau-Bechtold recomputation of `vt, vq, cldfra_bl1d,
    qc_bl1d` at plume levels (NO REFERENCE COVERAGE, see the header)      :6622-6768
14. the `maxmf` sign flip that marks a DRY plume                          :6771-6775

INPUTS are the frozen column (`u, v, w, th, thl, thv, tk, qt, qv, qc`, `p`, `rho`,
`exner`, `dz`, `zw`), the current `qke`, the surface block (`ust, flt, fltv, flq,
flqv, pblh, kpbl, dx, landsea, ts`) and the subgrid cloud `qc_bl1d`/`cldfra_bl1d` as
`mym_condensation!` left them. `qt` is `sqw`, `qv` is `sqv` and `qc` is `sqc`, i.e.
SPECIFIC contents — the `mynnedmf_wrapper` convention. `ts` is the driver's `th_sfc`
(`= ts/exner(1)`, harness README item 2), not a temperature.

`qc_bl1d_old` and `cldfra_bl1d_old` are `intent(inout)` arguments of the Fortran
(:5781) that its body NEVER reads or writes; they are accepted here so the call site
matches, and ignored. `sgm` is the same (see the header). `F_QC`/`F_QI` are
`logical, optional` at :5713 and likewise never referenced in the body — they are
required to be `true` here so a caller cannot believe they do something.

OUTPUTS are written into the passed arrays (which the caller usually takes from
`work`); the four `intent(out)` scalars and the activation diagnostics come back in an
`EDMFGate`.

Allocation-free on a warm call.
"""
function dmp_mf!(kts::Int, kte::Int, dt::Float64,
                 zw::Vector{Float64}, dz::Vector{Float64},
                 p::Vector{Float64}, rho::Vector{Float64},
                 momentum_opt::Int, tke_opt::Int, scalar_opt::Int,
                 u::Vector{Float64}, v::Vector{Float64}, w::Vector{Float64},
                 th::Vector{Float64}, thl::Vector{Float64}, thv::Vector{Float64},
                 tk::Vector{Float64}, qt::Vector{Float64}, qv::Vector{Float64},
                 qc::Vector{Float64}, qke::Vector{Float64},
                 exner::Vector{Float64}, vt::Vector{Float64}, vq::Vector{Float64},
                 sgm::Vector{Float64},
                 ust::Float64, flt::Float64, fltv::Float64, flq::Float64,
                 flqv::Float64, pblh::Float64, kpbl::Int, dx::Float64,
                 landsea::Float64, ts::Float64,
                 edmf_a::Vector{Float64}, edmf_w::Vector{Float64},
                 edmf_qt::Vector{Float64}, edmf_thl::Vector{Float64},
                 edmf_ent::Vector{Float64}, edmf_qc::Vector{Float64},
                 s_aw::Vector{Float64}, s_awthl::Vector{Float64},
                 s_awqt::Vector{Float64}, s_awqv::Vector{Float64},
                 s_awqc::Vector{Float64}, s_awu::Vector{Float64},
                 s_awv::Vector{Float64}, s_awqke::Vector{Float64},
                 sub_thl::Vector{Float64}, sub_sqv::Vector{Float64},
                 sub_u::Vector{Float64}, sub_v::Vector{Float64},
                 det_thl::Vector{Float64}, det_sqv::Vector{Float64},
                 det_sqc::Vector{Float64}, det_u::Vector{Float64},
                 det_v::Vector{Float64},
                 qc_bl1d::Vector{Float64}, cldfra_bl1d::Vector{Float64},
                 qc_bl1d_old::Vector{Float64}, cldfra_bl1d_old::Vector{Float64},
                 F_QC::Bool, F_QI::Bool, Psig_shcu::Float64,
                 spp_pbl::Int, rstoch_col::Vector{Float64},
                 c::MYNNConstants, work::EDMFWork)
    (F_QC && F_QI) ||
        throw(ArgumentError("dmp_mf!: F_QC = $F_QC, F_QI = $F_QI. Both are " *
                            "`logical, optional` arguments of DMP_mf " *
                            "(module_bl_mynn.F90 :5713) that its body never " *
                            "references; they must be true here so the call site " *
                            "cannot imply otherwise."))
    (momentum_opt >= 0 && tke_opt >= 0 && scalar_opt >= 0) ||
        throw(ArgumentError("dmp_mf!: momentum_opt/tke_opt/scalar_opt must be >= 0"))
    work.n == kte - kts + 1 ||
        throw(ArgumentError("dmp_mf!: EDMFWork sized for $(work.n) layers, " *
                            "kts:kte = $kts:$kte"))
    kts == 1 ||
        throw(ArgumentError("dmp_mf!: this port fixes kts = 1 (got $kts)"))

    nup = work.nup
    upw = work.upw; upthl = work.upthl; upqt = work.upqt; upqc = work.upqc
    upa = work.upa; upu = work.upu; upv = work.upv; upthv = work.upthv
    upqke = work.upqke; ent = work.ent
    rhoz = work.rhoz; exneri = work.exneri; dzi = work.dzi
    edmf_th = work.edmf_th
    envm_thl = work.envm_thl; envm_sqv = work.envm_sqv; envm_sqc = work.envm_sqc
    envm_u = work.envm_u; envm_v = work.envm_v
    envi_a = work.envi_a; envi_w = work.envi_w

    _edmf_reset!(work)
    nup2 = nup                                                            # :5773

    @inbounds begin
    # ── 2. resolved-w taper and cloud-base search (:5938-5960) ────────────────
    maxw = 0.0
    cloud_base = 9000.0
    k50 = kts                     # note (a): the Fortran leaves this uninitialised
    for k in 1:(kte-1)
        if zw[k] > pblh + 500.0
            break
        end
        wpbl = w[k]
        if w[k] < 0.0
            wpbl = 2.0*w[k]
        end
        maxw = max(maxw, abs(wpbl))
        # Find highest k-level below 50m AGL
        if zw[k] <= 50.0
            k50 = k
        end
        # Search for cloud base
        qc_sgs = max(qc[k], qc_bl1d[k])
        if qc_sgs > 1.0e-5 && cldfra_bl1d[k] >= 0.5 && cloud_base == 9000.0
            cloud_base = 0.5*(zw[k] + zw[k+1])
        end
    end

    # do nothing for small w (< 1 m/s), but linearly taper off for w > 1.0 m/s
    maxw = max(0.0, maxw - 1.0)
    Psig_w = max(0.0, 1.0 - maxw)
    Psig_w = min(Psig_w, Psig_shcu)

    # Completely shut off MF for strong resolved-scale vertical velocities (:5963-5964)
    fltv2 = fltv
    if Psig_w == 0.0 && fltv > 0.0
        fltv2 = -1.0*fltv
    end

    # ── 3. the superadiabatic test (:5966-5990) ──────────────────────────────
    superadiabatic = false
    hux = (landsea - 1.5) >= 0 ? -0.001 :   # WATER: dT/dz < -0.1 K per 100 m
                                 -0.005     # LAND : dT/dz < -0.5 K per 100 m
    tvs = ts*(1.0 + c.p608*qv[kts])
    for k in 1:max(1, k50 - 1)   # "-1" because k50 used interface heights (zw)
        if k == 1
            if (thv[k] - tvs)/(0.5*dz[k]) < hux
                superadiabatic = true
            else
                superadiabatic = false
                break
            end
        else
            if (thv[k] - thv[k-1])/(0.5*(dz[k] + dz[k-1])) < hux
                superadiabatic = true
            else
                superadiabatic = false
                break
            end
        end
    end

    # ── 4. the five plume-width criteria (:5992-6037) ────────────────────────
    maxwidth = min(dx*MYNN_MF_DCUT, MYNN_MF_LMAX)                # (1) largest = 1.2 dx
    maxwidth = min(maxwidth, 1.1*pblh)                           # (2) scale break
    if (landsea - 1.5) < 0                                       # (3) cloud deck
        maxwidth = min(maxwidth, 0.5*cloud_base)                 #     land
    else
        maxwidth = min(maxwidth, 0.9*cloud_base)                 #     water
    end
    wspd_pbl = sqrt(max(u[kts]^2 + v[kts]^2, 0.01))              # (4) wind-speed limit
    width_flx = if (landsea - 1.5) < 0                           # (5) weak forcing
        max(min(1000.0*(0.6*tanh((fltv - 0.040)/0.04) + 0.5), 1000.0), 0.0)   # land
    else
        max(min(1000.0*(0.6*tanh((fltv - 0.007)/0.02) + 0.5), 1000.0), 0.0)   # water
    end
    maxwidth = min(maxwidth, width_flx)
    minwidth = MYNN_MF_LMIN
    # allow the min plume size to grow in large-flux conditions
    if maxwidth >= (MYNN_MF_LMAX - 1.0) && fltv > 0.2
        minwidth = MYNN_MF_LMIN + MYNN_MF_DLMIN*min((fltv - 0.2)/0.3, 1.0)
    end
    if maxwidth <= minwidth        # deactivate the MF component
        nup2 = 0
        maxwidth = 0.0
    end

    ktop = 0
    ztop = 0.0
    maxmf = 0.0

    # scalars the Fortran leaves live across the gate for the debug print
    wstar = 0.0; sigmaW = 0.0; sigmaQT = 0.0; sigmaTH = 0.0
    adjustment = 1.0

    # ── 5. THE GATE (:6039) ──────────────────────────────────────────────────
    active = (fltv2 > 0.002) && (maxwidth > minwidth) && superadiabatic
    if active

        # ── 6. the Neggers size distribution (:6041-6078) ────────────────────
        cn = 0.0
        d  = -1.9                       # Neggers 2015 (JAMES); N = C l^d
        dl = (maxwidth - minwidth)/Float64(nup - 1)
        for i in 1:nup
            l = minwidth + dl*Float64(i - 1)          # diameter of plume i
            cn = cn + l^d * (l*l)/(dx*dx) * dl        # fractional area of plume i
        end
        C = MYNN_MF_ATOT/cn                           # normalise to Atot

        # updraft area as a function of the buoyancy flux
        acfac = if (landsea - 1.5) < 0
            0.5*tanh((fltv2 - 0.02)/0.05) + 0.5       # land
        else
            0.5*tanh((fltv2 - 0.015)/0.04) + 0.5      # water
        end
        # taper the scheme off linearly above a 10 m/s surface wind
        ac_wsp = wspd_pbl <= 10.0 ? 1.0 : 1.0 - min((wspd_pbl - 10.0)/15.0, 1.0)
        acfac = acfac * ac_wsp

        An2 = 0.0
        for i in 1:nup
            l = minwidth + dl*Float64(i - 1)
            N = C*l^d                                 # number density of plume i
            upa[1,i] = N*l*l/(dx*dx) * dl             # fractional area of plume i
            upa[1,i] = upa[1,i]*acfac
            An2 = An2 + upa[1,i]                      # (diagnostic only)
        end

        # ── 7. surface conditions for the updrafts (:6080-6141) ──────────────
        z0    = 50.0
        pwmin = 0.1
        pwmax = 0.4

        wstar  = max(1.0e-2, (c.gtr*fltv2*pblh)^MYNN_ONETHIRD)
        qstar  = max(flq, 1.0e-5)/wstar
        thstar = flt/wstar

        csigma = 1.34                    # the same over water and land (:6091-6095)

        exc_fac = if MYNN_ENV_SUBS
            0.0
        elseif (landsea - 1.5) >= 0
            0.58*4.0     # water: compensate for the decreased pwmin/pwmax
        else
            0.58         # land: the superadiabatic layers are already large enough
        end
        exc_fac = exc_fac * ac_wsp       # decrease the excess for large wind speeds

        # sigmaW is typically about 0.5*wstar
        sigmaW  = csigma*wstar*(z0/pblh)^MYNN_ONETHIRD*(1 - 0.8*z0/pblh)
        sigmaQT = csigma*qstar*(z0/pblh)^MYNN_ONETHIRD
        sigmaTH = csigma*thstar*(z0/pblh)^MYNN_ONETHIRD

        wmin_sfc = min(sigmaW*pwmin, 0.1)    # note (d): the Fortran calls this `wmin`
        wmax     = min(sigmaW*pwmax, 0.5)

        # surface updraft properties at the interface between k = 1 and 2
        for i in 1:nup
            # wlv = wmin_sfc + (wmax-wmin_sfc)/nup2*(i-1)   -- :6122, dead
            upw[1,i]  = wmin_sfc + Float64(i)/Float64(nup)*(wmax - wmin_sfc)
            upu[1,i]  = (u[kts]*dz[kts+1] + u[kts+1]*dz[kts])/(dz[kts] + dz[kts+1])
            upv[1,i]  = (v[kts]*dz[kts+1] + v[kts+1]*dz[kts])/(dz[kts] + dz[kts+1])
            upqc[1,i] = 0.0

            exc_heat   = exc_fac*upw[1,i]*sigmaTH/sigmaW
            upthv[1,i] = (thv[kts]*dz[kts+1] + thv[kts+1]*dz[kts])/(dz[kts] + dz[kts+1]) +
                         exc_heat
            upthl[1,i] = (thl[kts]*dz[kts+1] + thl[kts+1]*dz[kts])/(dz[kts] + dz[kts+1]) +
                         exc_heat

            exc_moist = exc_fac*upw[1,i]*sigmaQT/sigmaW
            upqt[1,i] = (qt[kts]*dz[kts+1] + qt[kts+1]*dz[kts])/(dz[kts] + dz[kts+1]) +
                        exc_moist

            upqke[1,i] = (qke[kts]*dz[kts+1] + qke[kts+1]*dz[kts])/(dz[kts] + dz[kts+1])
        end

        # environmental variables which detrainment can modify (:6143-6149)
        for k in kts:kte
            envm_thl[k] = thl[k]
            envm_sqv[k] = qv[k]
            envm_sqc[k] = qc[k]
            envm_u[k]   = u[k]
            envm_v[k]   = v[k]
        end
        for k in kts:(kte-1)
            rhoz[k] = (rho[k]*dz[k+1] + rho[k+1]*dz[k])/(dz[k+1] + dz[k])
        end
        rhoz[kte] = rho[kte]

        # scale-adaptive factor on the pressure-gradient term (:6153)
        dxsa = 1.0 - min(max((12000.0 - dx)/(12000.0 - 3000.0), 0.0), 1.0)

        # ── 8. integrate each updraft (:6155-6366) ───────────────────────────
        for i in 1:nup
            QCn = 0.0
            overshoot = 0
            dzp = 0.0                # note (b): the Fortran leaves this undefined
            l = minwidth + dl*Float64(i - 1)          # diameter of plume i
            for k in (kts+1):(kte-1)
                # -- entrainment, Tian and Kuang (2016) (:6182-6200) ----------
                wmin = 0.3 + l*0.0005                          # note (d)
                ent[k,i] = 0.33/(min(max(upw[k-1,i], wmin), 0.9)*l)
                ent[k,i] = max(ent[k,i], 0.0003)               # background minimum
                # increase entrainment for plumes extending very high
                if zw[k] >= min(pblh + 1500.0, 4000.0)
                    ent[k,i] = ent[k,i] + (zw[k] - min(pblh + 1500.0, 4000.0))*5.0e-6
                end
                ent[k,i] = ent[k,i] * (1.0 - rstoch_col[k])    # SPP (spp_pbl = 0)
                ent[k,i] = min(ent[k,i], 0.9/(zw[k+1] - zw[k]))

                # -- environment u & v at the interface levels (:6202-6206) ---
                Uk   = (u[k]*dz[k+1] + u[k+1]*dz[k])/(dz[k+1] + dz[k])
                Ukm1 = (u[k-1]*dz[k] + u[k]*dz[k-1])/(dz[k-1] + dz[k])
                Vk   = (v[k]*dz[k+1] + v[k+1]*dz[k])/(dz[k+1] + dz[k])
                Vkm1 = (v[k-1]*dz[k] + v[k]*dz[k-1])/(dz[k-1] + dz[k])

                # -- linear entrainment (:6208-6220), note (f) ----------------
                EntExp = ent[k,i]*(zw[k+1] - zw[k])
                EntExm = EntExp*0.3333       # reduced entrainment for momentum
                QTn  = upqt[k-1,i] *(1.0 - EntExp) + qt[k]*EntExp
                THLn = upthl[k-1,i]*(1.0 - EntExp) + thl[k]*EntExp
                Un   = upu[k-1,i]  *(1.0 - EntExm) + u[k]*EntExm +
                       dxsa*MYNN_MF_PGFAC*(Uk - Ukm1)
                Vn   = upv[k-1,i]  *(1.0 - EntExm) + v[k]*EntExm +
                       dxsa*MYNN_MF_PGFAC*(Vk - Vkm1)
                QKEn = upqke[k-1,i]*(1.0 - EntExp) + qke[k]*EntExp

                # qc, qt & thl as entrainment alone left them (:6224-6226)
                qc_ent  = QCn
                qt_ent  = QTn
                thl_ent = THLn

                # -- plume thermodynamics (:6245-6247) ------------------------
                Pk = (p[k]*dz[k+1] + p[k+1]*dz[k])/(dz[k+1] + dz[k])
                THVn, QCn = condensation_edmf(QTn, THLn, Pk, zw[k+1], QCn, c)

                # -- buoyancy and the w-equation (:6249-6285) -----------------
                THVk   = (thv[k]*dz[k+1] + thv[k+1]*dz[k])/(dz[k+1] + dz[k])
                THVkm1 = (thv[k-1]*dz[k] + thv[k]*dz[k-1])/(dz[k-1] + dz[k])

                B = c.grav*(THVn/THVk - 1.0)
                BCOEFF = B > 0.0 ? 0.15 : 0.2

                # TEMF form; note (f): the step is the layer BELOW the interface
                Wn = if upw[k-1,i] < 0.2
                    upw[k-1,i] + (-2.0 * ent[k,i] * upw[k-1,i] +
                                  BCOEFF*B / max(upw[k-1,i], 0.2)) *
                                 min(zw[k] - zw[k-1], 250.0)
                else
                    upw[k-1,i] + (-2.0 * ent[k,i] * upw[k-1,i] +
                                  BCOEFF*B / upw[k-1,i]) *
                                 min(zw[k] - zw[k-1], 250.0)
                end
                # do not accelerate more than 1.25 m/s over 200 m (max 2 m/s)
                if Wn > upw[k-1,i] + min(1.25*(zw[k] - zw[k-1])/200.0, 2.0)
                    Wn = upw[k-1,i] + min(1.25*(zw[k] - zw[k-1])/200.0, 2.0)
                end
                # symmetrical max decrease
                if Wn < upw[k-1,i] - min(1.25*(zw[k] - zw[k-1])/200.0, 2.0)
                    Wn = upw[k-1,i] - min(1.25*(zw[k] - zw[k-1])/200.0, 2.0)
                end
                Wn = min(max(Wn, 0.0), 3.0)

                # the plume must make it up at least one level (:6286-6291), note (e)
                if k == kts + 1 && Wn == 0.0
                    nup2 = 0
                    break
                end

                # -- overshoot / termination (:6304-6318) ---------------------
                if Wn <= 0.0 && overshoot == 0
                    overshoot = 1
                    if THVk - THVkm1 > 0.0
                        bvf = sqrt(c.gtr*(THVk - THVkm1)/dz[k])
                        Frz = upw[k-1,i]/(bvf*dz[k])   # vertical Froude number
                        dzp = dz[k]*max(min(Frz, 1.0), 0.0)
                    end
                    # note (b): dzp keeps its previous value when THVk <= THVkm1
                else
                    dzp = dz[k]
                end

                # -- Asai and Kasahara (1967) environment update (:6325-6348) --
                #    writes envm_* only, which nothing reads (env_subs = false)
                aratio  = min(upa[k-1,i]/(1.0 - upa[k-1,i]), 0.5)
                detturb = 0.00008
                oow     = -0.060/max(1.0, (0.5*(Wn + upw[k-1,i])))
                detrate   = min(max(oow*(Wn - upw[k-1,i])/dz[k], detturb), 0.0002)
                detrateUV = min(max(oow*(Wn - upw[k-1,i])/dz[k], detturb), 0.0001)
                envm_thl[k] = envm_thl[k] +
                    (0.5*(thl_ent + upthl[k-1,i]) - thl[k])*detrate*aratio*min(dzp, MYNN_MF_DZPMAX)
                qv_ent = 0.5*(max(qt_ent - qc_ent, 0.0) +
                              max(upqt[k-1,i] - upqc[k-1,i], 0.0))
                envm_sqv[k] = envm_sqv[k] +
                    (qv_ent - qv[k])*detrate*aratio*min(dzp, MYNN_MF_DZPMAX)
                if upqc[k-1,i] > 1.0e-8
                    qc_grid = qc[k] > 1.0e-6 ? qc[k] : cldfra_bl1d[k]*qc_bl1d[k]
                    envm_sqc[k] = envm_sqc[k] +
                        max(upa[k-1,i]*0.5*(QCn + upqc[k-1,i]) - qc_grid, 0.0)*
                        detrate*aratio*min(dzp, MYNN_MF_DZPMAX)
                end
                envm_u[k] = envm_u[k] +
                    (0.5*(Un + upu[k-1,i]) - u[k])*detrateUV*aratio*min(dzp, MYNN_MF_DZPMAX)
                envm_v[k] = envm_v[k] +
                    (0.5*(Vn + upv[k-1,i]) - v[k])*detrateUV*aratio*min(dzp, MYNN_MF_DZPMAX)

                # -- commit or stop (:6350-6366) ------------------------------
                if Wn > 0.0
                    upw[k,i]   = Wn
                    upthv[k,i] = THVn
                    upthl[k,i] = THLn
                    upqt[k,i]  = QTn
                    upqc[k,i]  = QCn
                    upu[k,i]   = Un
                    upv[k,i]   = Vn
                    upqke[k,i] = QKEn
                    upa[k,i]   = upa[k-1,i]
                    ktop = max(ktop, k)
                else
                    break
                end
            end
        end
    else
        # at least one activation condition failed (:6382-6386)
        nup2 = 0
    end

    ktop = min(ktop, kte - 1)                                             # :6388
    ztop = ktop == 0 ? 0.0 : zw[ktop]

    if nup2 > 0
        # ── 9. the interface flux sums (:6392-6459) ──────────────────────────
        # all s_aw* are == 0 at k = 1
        for i in 1:nup, k in kts:(kte-1)
            s_aw[k+1]    = s_aw[k+1]    + rhoz[k]*upa[k,i]*upw[k,i]*Psig_w
            s_awthl[k+1] = s_awthl[k+1] + rhoz[k]*upa[k,i]*upw[k,i]*upthl[k,i]*Psig_w
            s_awqt[k+1]  = s_awqt[k+1]  + rhoz[k]*upa[k,i]*upw[k,i]*upqt[k,i]*Psig_w
            # to conform to grid-mean properties, move qc to qv in saturated layers
            qc_plume = upqc[k,i]
            s_awqc[k+1]  = s_awqc[k+1]  + rhoz[k]*upa[k,i]*upw[k,i]*qc_plume*Psig_w
            s_awqv[k+1]  = s_awqt[k+1]  - s_awqc[k+1]
        end
        if momentum_opt > 0
            for i in 1:nup, k in kts:(kte-1)
                s_awu[k+1] = s_awu[k+1] + rhoz[k]*upa[k,i]*upw[k,i]*upu[k,i]*Psig_w
                s_awv[k+1] = s_awv[k+1] + rhoz[k]*upa[k,i]*upw[k,i]*upv[k,i]*Psig_w
            end
        end
        if tke_opt > 0
            for i in 1:nup, k in kts:(kte-1)
                s_awqke[k+1] = s_awqke[k+1] +
                               rhoz[k]*upa[k,i]*upw[k,i]*upqke[k,i]*Psig_w
            end
        end
        # scalar_opt > 0 would fill s_awqnc … s_awqnbca here (:6449-6459); see the
        # file header — every one of them is identically zero on this configuration.

        # ── 10. the heat-flux limiter (:6463-6501) ───────────────────────────
        # The flux of heat out of the top of layer 1 must be < fluxportion * the
        # surface heat flux; the correction rescales the WHOLE column, UPA included.
        flx1 = 0.0
        if s_aw[kts+1] != 0.0
            dzi[kts] = 0.5*(dz[kts] + dz[kts+1])   # dz centred at the interface
            flx1 = max(s_aw[kts+1]*(th[kts] - th[kts+1])/dzi[kts], 1.0e-5)
        else
            flx1 = 0.0
        end
        adjustment = 1.0
        if flx1 > MYNN_MF_FLUXPORTION*flt/dz[kts] && flx1 > 0.0
            adjustment = MYNN_MF_FLUXPORTION*flt/dz[kts]/flx1
            for k in eachindex(s_aw)
                s_aw[k]    *= adjustment
                s_awthl[k] *= adjustment
                s_awqt[k]  *= adjustment
                s_awqc[k]  *= adjustment
                s_awqv[k]  *= adjustment
            end
            if momentum_opt > 0
                for k in eachindex(s_awu)
                    s_awu[k] *= adjustment
                    s_awv[k] *= adjustment
                end
            end
            if tke_opt > 0
                for k in eachindex(s_awqke)
                    s_awqke[k] *= adjustment
                end
            end
            for idx in eachindex(upa)
                upa[idx] *= adjustment
            end
        end

        # ── 11. area-weighted mean updraft properties (:6505-6528) ───────────
        # all edmf_* at k = 1 are the interface at the top of the first model layer
        for k in kts:(kte-1), i in 1:nup
            edmf_a[k]   = edmf_a[k]   + upa[k,i]
            edmf_w[k]   = edmf_w[k]   + upa[k,i]*upw[k,i]
            edmf_qt[k]  = edmf_qt[k]  + upa[k,i]*upqt[k,i]
            edmf_thl[k] = edmf_thl[k] + upa[k,i]*upthl[k,i]
            edmf_ent[k] = edmf_ent[k] + upa[k,i]*ent[k,i]
            edmf_qc[k]  = edmf_qc[k]  + upa[k,i]*upqc[k,i]
        end
        for k in kts:(kte-1)
            # only edmf_a is multiplied by Psig_w: that carries the scale-awareness
            # of the subsidence below
            if edmf_a[k] > 0.0
                edmf_w[k]   = edmf_w[k]/edmf_a[k]
                edmf_qt[k]  = edmf_qt[k]/edmf_a[k]
                edmf_thl[k] = edmf_thl[k]/edmf_a[k]
                edmf_ent[k] = edmf_ent[k]/edmf_a[k]
                edmf_qc[k]  = edmf_qc[k]/edmf_a[k]
                edmf_a[k]   = edmf_a[k]*Psig_w
                # find the maximum mass flux in the column
                if edmf_a[k]*edmf_w[k] > maxmf
                    maxmf = edmf_a[k]*edmf_w[k]
                end
            end
        end

        # ── 12. environmental subsidence and detrainment (:6549-6620) ────────
        # UNREACHABLE: env_subs is a compile-time .false. parameter (:337). See note
        # (c) in the file header, including the Fortran out-of-bounds read this
        # clamps. Nothing below has reference coverage.
        if MYNN_ENV_SUBS
            for k in (kts+1):(kte-1)
                # smooth w & a: sharp gradients in plume variables are unlikely to
                # extend to the environment. w is treated as negative further below.
                envi_w[k] = MYNN_ONETHIRD*(edmf_w[k-1] + edmf_w[k] + edmf_w[k+1])
                envi_a[k] = MYNN_ONETHIRD*(edmf_a[k-1] + edmf_a[k] + edmf_a[k+1])*adjustment
            end
            envi_w[kts]   = edmf_w[kts]
            envi_a[kts]   = edmf_a[kts]
            envi_w[kte]   = 0.0
            envi_a[kte]   = edmf_a[kte]
            envi_w[kte+1] = 0.0
            envi_a[kte+1] = edmf_a[kte]
            # limiter for very long time steps (dt > 300 s), first model level only
            sublim = envi_w[kts] > 0.9*dz[kts]/dt ? 0.9*dz[kts]/dt/envi_w[kts] : 1.0
            for k in kts:kte
                temp = envi_a[k]
                envi_a[k] = 1.0 - temp
                envi_w[k] = MYNN_MF_CSUB*sublim*envi_w[k]*temp/(1.0 - temp)
            end
            krh = kte     # FORTRAN OOB: the Fortran reads rhoz(k) with the stale
                          # k = kte+1 left by the loop above; clamped, see note (c)
            dzi[kts] = 0.5*(dz[kts] + dz[kts+1])
            sub_thl[kts] = 0.5*envi_w[kts]*envi_a[kts]*
                (rho[kts+1]*thl[kts+1] - rho[kts]*thl[kts])/dzi[kts]/rhoz[krh]
            sub_sqv[kts] = 0.5*envi_w[kts]*envi_a[kts]*
                (rho[kts+1]*qv[kts+1] - rho[kts]*qv[kts])/dzi[kts]/rhoz[krh]
            for k in (kts+1):(kte-1)
                dzi[k] = 0.5*(dz[k] + dz[k+1])
                sub_thl[k] = 0.5*(envi_w[k] + envi_w[k-1])*0.5*(envi_a[k] + envi_a[k-1])*
                    (rho[k+1]*thl[k+1] - rho[k]*thl[k])/dzi[k]/rhoz[k]
                sub_sqv[k] = 0.5*(envi_w[k] + envi_w[k-1])*0.5*(envi_a[k] + envi_a[k-1])*
                    (rho[k+1]*qv[k+1] - rho[k]*qv[k])/dzi[k]/rhoz[k]
            end
            for k in kts:(kte-1)
                det_thl[k] = MYNN_MF_CDET*(envm_thl[k] - thl[k])*envi_a[k]*Psig_w
                det_sqv[k] = MYNN_MF_CDET*(envm_sqv[k] - qv[k])*envi_a[k]*Psig_w
                det_sqc[k] = MYNN_MF_CDET*(envm_sqc[k] - qc[k])*envi_a[k]*Psig_w
            end
            if momentum_opt > 0
                sub_u[kts] = 0.5*envi_w[kts]*envi_a[kts]*
                    (rho[kts+1]*u[kts+1] - rho[kts]*u[kts])/dzi[kts]/rhoz[krh]
                sub_v[kts] = 0.5*envi_w[kts]*envi_a[kts]*
                    (rho[kts+1]*v[kts+1] - rho[kts]*v[kts])/dzi[kts]/rhoz[krh]
                for k in (kts+1):(kte-1)
                    sub_u[k] = 0.5*(envi_w[k] + envi_w[k-1])*0.5*(envi_a[k] + envi_a[k-1])*
                        (rho[k+1]*u[k+1] - rho[k]*u[k])/dzi[k]/rhoz[k]
                    sub_v[k] = 0.5*(envi_w[k] + envi_w[k-1])*0.5*(envi_a[k] + envi_a[k-1])*
                        (rho[k+1]*v[k+1] - rho[k]*v[k])/dzi[k]/rhoz[k]
                end
                for k in kts:(kte-1)
                    det_u[k] = MYNN_MF_CDET*(envm_u[k] - u[k])*envi_a[k]*Psig_w
                    det_v[k] = MYNN_MF_CDET*(envm_v[k] - v[k])*envi_a[k]*Psig_w
                end
            end
        end

        # ── 13. plume exner, theta and interface dz (:6622-6628) ─────────────
        for k in kts:(kte-1)
            exneri[k]  = (exner[k]*dz[k+1] + exner[k+1]*dz[k])/(dz[k+1] + dz[k])
            edmf_th[k] = edmf_thl[k] + c.xlvcp/exneri[k]*edmf_qc[k]
            dzi[k]     = 0.5*(dz[k] + dz[k+1])
        end

        # ── the Chaboureau-Bechtold shallow-cu cloud (:6630-6768) ────────────
        # cldfra_bl1d and qc_bl1d were already set by mym_condensation; here a
        # shallow-cu component is ADDED, but never at k = 1 (the loop starts at 2).
        # NO REFERENCE COVERAGE — see the warning in the file header.
        for k in (kts+1):(kte-2)
            if k > ktop
                break
            end
            if 0.5*(edmf_qc[k] + edmf_qc[k-1]) > 0.0 &&
               cldfra_bl1d[k] < MYNN_MF_CF_THRESH
                # interpolate plume quantities to mass levels
                Aup = (edmf_a[k]*dzi[k-1]  + edmf_a[k-1]*dzi[k])/(dzi[k-1] + dzi[k])
                THp = (edmf_th[k]*dzi[k-1] + edmf_th[k-1]*dzi[k])/(dzi[k-1] + dzi[k])
                QTp = (edmf_qt[k]*dzi[k-1] + edmf_qt[k-1]*dzi[k])/(dzi[k-1] + dzi[k])
                esat = esat_blend(tk[k], c)
                # qsl = ep_2*esat/max(1e-7, (p(k) - ep_3*esat))   -- :6636, dead

                # condensed liquid in the plume, on mass levels
                QCp = if edmf_qc[k] > 0.0 && edmf_qc[k-1] > 0.0
                    (edmf_qc[k]*dzi[k-1] + edmf_qc[k-1]*dzi[k])/(dzi[k-1] + dzi[k])
                else
                    max(edmf_qc[k], edmf_qc[k-1])
                end

                xl      = xl_blend(tk[k], c)         # blended latent heat
                qsat_tk = qsat_blend(tk[k], p[k], c) # saturation mixing ratio at T, p
                rsl = xl*qsat_tk / (c.r_v*tk[k]^2)   # C-C slope, CB02 Eqn. 4
                cpm = c.cp + qt[k]*c.cpv             # CB02 sec. 2, para. 1
                a   = 1.0/(1.0 + xl*rsl/cpm)         # CB02 "a"
                b9  = a*rsl                          # CB02 "b"

                q2p = c.xlvcp/exner[k]
                pt  = thl[k] + q2p*QCp*Aup           # potential temp (env + plume)
                bb  = b9*tk[k]/pt                    # "b9" of BCMT95 (a factor T/theta
                                                     # from CB02's; the sat. mixing-ratio
                                                     # to specific-humidity conversion
                                                     # is neglected, as in the Fortran)
                qww   = 1.0 + 0.61*qt[k]
                alpha = 0.61*pt
                beta  = pt*xl/(tk[k]*c.cp) - 1.61*pt

                # convective component of the cloud fraction
                f = a > 0.0 ? min(1.0/a, 4.0) : 1.0  # vertical profile scaling (CB2005)

                sigq = 10.0 * Aup * (QTp - qt[k])    # per S. de Roode
                sigq = max(sigq, qsat_tk*0.02)       # constrain wrt saturation
                sigq = min(sigq, qsat_tk*0.25)

                qmq = a * (qt[k] - qsat_tk)          # saturation deficit/excess
                Q1  = qmq/sigq

                # original CB, the same expression over water and land; only the
                # lower bound on mf_cf differs (:6688-6702)
                mf_cf = min(max(0.5 + 0.36 * atan(1.55*Q1), 0.01), 0.6)
                if (landsea - 1.5) >= 0
                    mf_cf = max(mf_cf, 1.2 * Aup)    # water
                else
                    mf_cf = max(mf_cf, 1.8 * Aup)    # land
                end
                mf_cf = min(mf_cf, 5.0 * Aup)

                # update the cloud fraction and the (grid-mean, not in-cloud) water
                # where the mass-flux scheme is active. Water and land are identical
                # in the Fortran (:6716-6733); written once here.
                if QCp * Aup > 5.0e-5
                    qc_bl1d[k] = 1.86 * (QCp * Aup) - 2.2e-5
                else
                    qc_bl1d[k] = 1.18 * (QCp * Aup)
                end
                cldfra_bl1d[k] = mf_cf
                # Ac_mf = mf_cf  -- :6722, assigned and never read

                # recompute the buoyancy-flux terms for mass-flux clouds, with the
                # Bechtold and Siebesma (1998) piecewise Fng (:6740-6764)
                Q1 = max(Q1, -2.25)
                Fng = if Q1 >= 1.0
                    1.0
                elseif Q1 >= -1.7 && Q1 < 1.0
                    exp(-0.4*(Q1 - 1.0))
                elseif Q1 >= -2.5 && Q1 < -1.7
                    3.0 + exp(-3.8*(Q1 + 1.7))
                else
                    min(23.9 + exp(-1.6*(Q1 + 2.5)), 60.0)
                end
                # link the buoyancy-flux function to active clouds only (c*Aup)
                vt[k] = qww   - (1.5*Aup)*beta*bb*Fng - 1.0
                vq[k] = alpha + (1.5*Aup)*beta*a*Fng  - c.tv0
            end
        end
    end  # nup2 > 0

    # ── 14. mark a dry plume with a negative maxmf (:6771-6775) ──────────────
    if ktop > 0
        maxqc = -Inf
        for k in 1:ktop
            maxqc = max(maxqc, edmf_qc[k])
        end
        if maxqc < 1.0e-8
            maxmf = -1.0*maxmf
        end
    end

    end # @inbounds

    return EDMFGate(maxwidth, ktop, maxmf, ztop, active, fltv2, minwidth,
                    superadiabatic, Psig_w, cloud_base, k50, nup2, adjustment)
end

# ── The per-column driver WITH the mass flux ─────────────────────────────────

"""
    mynn_column_step_edmf!(work, ework, c, col, st, opts) -> (Psig_bl, Psig_shcu, gate)

One boundary-layer time step on one frozen column WITH the mass-flux plumes: exactly
`mynn_column_step!` (src/mynn_closure.jl) with `dmp_mf!` called between
`mym_condensation!` and `mym_turbulence!`, which is where `mynn_bl_driver` calls it
(:1150-1180, and `ref_driver.f90` mode_b).

    re-gather the frozen column
      -> GET_PBLH -> SCALE_AWARE
      -> surface fluxes / rmol / zet / pmz / phh
      -> mym_condensation!
      -> dmp_mf!                      <-- the only difference
      -> mym_turbulence!    (now fed the real edmf_w / edmf_a)
      -> mym_predict!       (now fed the real s_aw / s_awqke)
      -> diss_heat
      -> mynn_tendencies!   (now fed the real s_aw*, sub_*, det_*)
      -> retrieve_exchange_coeffs!

WHY THIS IS A COPY AND NOT A FLAG. `mynn_column_step!` (src/mynn_closure.jl) is owned
by another stage and this one may not edit it. The two functions must stay in step:
with `dmp_mf!` producing all-zero plume sums (any of cases 1, 2, 3, 5) this one
reproduces `mynn_column_step!(…; edmf = false)` BITWISE, and `test/test_mynn_edmf.jl`
asserts exactly that on all four.

TO FOLD IT BACK IN, `mynn_column_step!` needs exactly this — one signature line and a
four-line body replacing its `edmf && throw(ArgumentError(...))` guard (note the
UNANNOTATED `ework`: `EDMFWork` is defined in this file, which is `include`d AFTER
mynn_closure.jl, so a type annotation there would be a forward reference):

    function mynn_column_step!(work::MYNNWork, c::MYNNConstants, col::MYNNColumn,
                               st::MYNNColumnState, opts::MYNNOptions;
                               edmf::Bool, ework = nothing)          # <- ework added
        if edmf
            ework === nothing && throw(ArgumentError("mynn_column_step!: edmf = " *
                "true needs an EDMFWork (src/mynn_edmf.jl)"))
            return mynn_column_step_edmf!(work, ework, c, col, st, opts)[1:2]
        end

plus `include("mynn_edmf.jl")` after `include("mynn_closure.jl")` in src/Scythe.jl.
That keeps one entry point without merging the two bodies; the bitwise testset is
what keeps the duplication honest.

`ework` carries both the `DMP_mf` locals and the plume outputs, which stay readable
after the call: `ework.edmf_a`, `ework.edmf_w`, `ework.edmf_qt`, `ework.edmf_thl`,
`ework.edmf_ent`, `ework.edmf_qc`, `ework.s_aw…`, `ework.sub_*`, `ework.det_*`. The
returned `EDMFGate` carries `maxwidth`, `ktop`, `maxmf`, `ztop` and the activation
diagnostics.

The downdraft set (`sd_aw*`, `bl_mynn_edmf_dd = 0`, :331) stays the zero buffer:
`DMP_mf` never fills it, `DDMF_JPL` does, and that routine is not ported.
"""
function mynn_column_step_edmf!(work::MYNNWork, ework::EDMFWork, c::MYNNConstants,
                                col::MYNNColumn, st::MYNNColumnState,
                                opts::MYNNOptions)
    n = col.n
    (work.n == n && st.n == n && ework.n == n) ||
        throw(ArgumentError("mynn_column_step_edmf!: work/state/edmf sized for " *
                            "$(work.n)/$(st.n)/$(ework.n), column has n = $n"))
    kts = 1; kte = n
    zn  = work.z_n
    zn1 = work.z_np1

    _mynn_gather!(work, col, c)

    zi, kzi = get_pblh!(kts, kte, work.s_thetav, st.qke, col.zw, col.dz, col.xland)
    st.pblh = zi
    st.kpbl = kzi
    Psig_bl, Psig_shcu = scale_aware(col.dx, zi)

    # -- surface fluxes and stability functions (mynn_bl_driver :1060-1097) --------
    cpm    = c.cp*(1.0 + 0.84*work.s_qv[kts])
    flqv   = col.qfx/col.rho[kts]
    flqc   = 0.0
    th_sfc = col.ts/col.exner[kts]
    flq    = flqv + flqc
    flt    = col.hfx/(col.rho[kts]*cpm) - c.xlvcp*flqc/col.exner[kts]
    fltv   = flt + flqv*c.p608*th_sfc
    rmol   = -c.karman*c.gtr*fltv/max(col.ust^3, 1.0e-6)
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
                      st.pblh, col.hfx, st.vt, st.vq, work.s_th, st.sgm, st.rmol,
                      opts.spp_pbl, zn, c, work)

    # -- THE MASS FLUX (module_bl_mynn.F90 :5680-6826) ----------------------------
    # `qc_bl1d_old`/`cldfra_bl1d_old` are the previous step's fields in the driver;
    # DMP_mf reads neither (see dmp_mf!'s docstring), so the zero buffer stands in.
    gate = dmp_mf!(kts, kte, opts.delt, col.zw, col.dz, col.p, col.rho,
                   opts.bl_mynn_edmf_mom, opts.bl_mynn_edmf_tke,
                   opts.bl_mynn_mixscalars,
                   work.s_u, work.s_v, work.s_w, work.s_th, work.s_thl,
                   work.s_thetav, work.s_tk, work.s_sqw, work.s_sqv, work.s_sqc,
                   st.qke, col.exner, st.vt, st.vq, st.sgm,
                   col.ust, flt, fltv, flq, flqv, st.pblh, st.kpbl, col.dx,
                   col.xland, th_sfc,
                   ework.edmf_a, ework.edmf_w, ework.edmf_qt, ework.edmf_thl,
                   ework.edmf_ent, ework.edmf_qc,
                   ework.s_aw, ework.s_awthl, ework.s_awqt, ework.s_awqv,
                   ework.s_awqc, ework.s_awu, ework.s_awv, ework.s_awqke,
                   ework.sub_thl, ework.sub_sqv, ework.sub_u, ework.sub_v,
                   ework.det_thl, ework.det_sqv, ework.det_sqc, ework.det_u,
                   ework.det_v,
                   st.qc_bl, st.cldfra_bl, zn, zn,
                   opts.flag_qc, opts.flag_qi, Psig_shcu, opts.spp_pbl, zn, c, ework)

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
                    opts.bl_mynn_mixlength, ework.edmf_w, ework.edmf_a, zn,
                    opts.spp_pbl, zn, c, work)

    mym_predict!(kts, kte, opts.closure, opts.delt, col.dz, col.ust, flt, flq,
                 pmz, phh, st.el, work.out_dfq, col.rho,
                 work.out_pdk, work.out_pdt, work.out_pdq, work.out_pdc,
                 st.qke, st.tsq, st.qsq, st.cov, ework.s_aw, ework.s_awqke,
                 opts.bl_mynn_edmf_tke,
                 work.out_qwt, work.out_qdiss, opts.tke_budget, c, work)

    # -- dissipative heating (mynn_bl_driver :1224-1234) --------------------------
    dh = work.out_diss_heat
    if opts.dheat_opt > 0
        @inbounds for k in kts:(kte-1)
            dh[k] = min(max(1.0*(st.qke[k]^1.5)/
                            (MYNN_B1*max(0.5*(st.el[k] + st.el[k+1]), 1.0))/c.cp,
                            0.0), 0.002)
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
                     ework.s_aw, ework.s_awthl, ework.s_awqt, ework.s_awqv,
                     ework.s_awqc, ework.s_awu, ework.s_awv,
                     zn1, zn1, zn1, zn1, zn1,
                     zn1, zn1, zn1, zn1, zn1, zn1, zn1,
                     ework.sub_thl, ework.sub_sqv, ework.sub_u, ework.sub_v,
                     ework.det_thl, ework.det_sqv, ework.det_sqc, ework.det_u,
                     ework.det_v,
                     opts.flag_qc, opts.flag_qi, false, false, false, false, false,
                     false, st.cldfra_bl,
                     opts.bl_mynn_cloudmix, opts.bl_mynn_mixqt, opts.bl_mynn_edmf,
                     opts.bl_mynn_edmf_mom, opts.bl_mynn_mixscalars, c, work)

    retrieve_exchange_coeffs!(kts, kte, work.out_dfm, work.out_dfh, col.dz,
                              work.out_km, work.out_kh)

    return (Psig_bl, Psig_shcu, gate)
end
