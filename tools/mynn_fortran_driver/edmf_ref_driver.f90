! Spot-check driver for the Chaboureau-Bechtold branch of DMP_mf (Scythe.jl, stage S3).
!
! WHY THIS PROGRAM EXISTS. In ref_driver_output_r8.txt the mass-flux plumes NEVER
! condense: `edmf_qc` is identically zero at every level of every case at both steps
! (case4_convective's plumes top out at 653 m, and condensation_edmf forces qc = 0
! below 100 m anyway). The whole shallow-cumulus cloud section of DMP_mf --
! module_bl_mynn.F90 :6631-6766, the Aup/THp/QTp/QCp interpolation, the CB02 a/b9,
! sigq, Q1, mf_cf, the qc_bl1d fits and the Bechtold-Siebesma Fng that overwrites
! vt/vq/cldfra_bl1d/qc_bl1d -- plus the `maxqc >= 1e-8` (MOIST plume) branch of the
! maxmf sign at :6772-6775 therefore has NO coverage in the main reference, and
! neither does any `landsea < 1.5` (LAND) branch, all five columns being water.
!
! This program calls DMP_mf DIRECTLY, twice, on a synthetic column built from
! columns/case4_convective.txt by moistening it 50 % and imposing a deep, strongly
! heated PBL. Both calls produce plumes ~22 interfaces deep that saturate, so the CB
! block runs at ~19 levels; the second call flips landsea to 1 (LAND) to take the
! other side of every land/water branch.
!
! The synthetic state is built with operations that a Julia port reproduces bitwise
! (the column file round-trips exactly at %.17g, and every transformation below is a
! single IEEE double operation), which is what lets test/test_mynn_edmf.jl rebuild the
! identical arguments without a second input file.
!
! Output: the same block format as ref_driver.f90, so test/reference/mynn_fortran_refs.jl
! parses it unchanged -- `mynn_reference(<this file's output>)`, cases
! "spot_moist_water" and "spot_moist_land", closure 2.50, mode B, step 1. It is
! checked in as edmf_ref_driver_output.txt; run.sh does NOT build it.
!
! To regenerate, after run.sh has built build/r8:
!   cd build/r8 && /opt/homebrew/bin/gfortran -ffree-line-length-none -O0 -ffp-contract=off \
!       -fdefault-real-8 -fdefault-double-8 stub_machine.o bl_mynn_common.o \
!       module_bl_mynn.o ../../edmf_ref_driver.f90 -o edmf_ref_driver && \
!       ./edmf_ref_driver > ../../edmf_ref_driver_output.txt
program edmf_ref_driver
  use machine, only: kind_phys
  use bl_mynn_common
  use module_bl_mynn
  implicit none

  integer :: n, k, u
  real(kind_phys) :: ps, ts, qsfc, ust0, hfx, qfx, wspd, znt, xland0, dx, rmol0, delt
  real(kind_phys), allocatable :: z(:), dz(:), uu(:), vv(:), ww(:), tk(:), th(:), &
       ex(:), p(:), rho(:), sqv(:), sqc(:), sqi(:)

  call read_constants('columns/constants.txt')
  call read_column('columns/case4_convective.txt')
  call run_spot('spot_moist_water', 2.0_kind_phys)
  call run_spot('spot_moist_land',  1.0_kind_phys)

contains

  subroutine read_constants(fn)
    character(len=*), intent(in) :: fn
    integer :: uu2
    open(newunit = uu2, file = fn, status = 'old', action = 'read')
    read(uu2, *) cp, cpv, cliq, cice, p608, ep_2, grav, karman, t0c, rcp, r_d, r_v, xlf, xlv
    close(uu2)
    xls    = xlv + xlf
    rvovrd = r_v / r_d
    ep_3   = 1. - ep_2
    gtr    = grav / tref
    rk     = cp / r_d
    tv0    = p608 * tref
    tv1    = (1. + p608) * tref
    xlscp  = (xlv + xlf) / cp
    xlvcp  = xlv / cp
    g_inv  = 1. / grav
  end subroutine read_constants

  subroutine read_column(fn)
    character(len=*), intent(in) :: fn
    open(newunit = u, file = fn, status = 'old', action = 'read')
    read(u, *) n
    allocate(z(n), dz(n), uu(n), vv(n), ww(n), tk(n), th(n), ex(n), p(n), rho(n), &
         sqv(n), sqc(n), sqi(n))
    read(u, *) ps, ts, qsfc, ust0, hfx, qfx, wspd, znt, xland0, dx, rmol0, delt
    do k = 1, n
       read(u, *) z(k), dz(k), uu(k), vv(k), ww(k), tk(k), th(k), ex(k), p(k), &
            rho(k), sqv(k), sqc(k), sqi(k)
    end do
    close(u)
  end subroutine read_column

  subroutine pv(k2, x)
    integer, intent(in) :: k2
    real(kind_phys), intent(in) :: x
    print '(i5,1x,es25.17e3)', k2, x
  end subroutine pv

  subroutine block(name, cas, x)
    character(len=*), intent(in) :: name, cas
    real(kind_phys), intent(in) :: x(:)
    integer :: k2
    print '(a,a,a,a,a,i0)', '## ', cas, ' closure=2.50 mode=B step=1 ', name, ' n=', size(x)
    do k2 = 1, size(x)
       call pv(k2, x(k2))
    end do
  end subroutine block

  subroutine scalar(name, cas, x)
    character(len=*), intent(in) :: name, cas
    real(kind_phys), intent(in) :: x
    call block(name, cas, [x])
  end subroutine scalar

  subroutine run_spot(cas, landsea)
    character(len=*), intent(in) :: cas
    real(kind_phys), intent(in) :: landsea
    integer, parameter :: nchem = 1, ndvel = 1
    integer :: kpbl, ktop
    real(kind_phys), dimension(n) :: qt, qv, qc, thl, thv, qke, qnc, qni, qnwfa, &
         qnifa, qnbca, vt, vq, sgm, qc_bl, cldfra_bl, qc_bl_old, cldfra_bl_old, &
         rstoch, edmf_a, edmf_w, edmf_qt, edmf_thl, edmf_ent, edmf_qc, &
         sub_thl, sub_sqv, sub_u, sub_v, det_thl, det_sqv, det_sqc, det_u, det_v
    real(kind_phys), dimension(n + 1) :: zw, s_aw, s_awthl, s_awqt, s_awqv, s_awqc, &
         s_awu, s_awv, s_awqke, s_awqnc, s_awqni, s_awqnwfa, s_awqnifa, s_awqnbca
    real(kind_phys) :: chem1(n, nchem), s_awchem(n + 1, nchem)
    real(kind_phys) :: pblh, flt, flq, fltv, flqv, ust, th_sfc, psig_shcu, dt, maxwidth, &
         maxmf, ztop

    ! wall heights, exactly as mynn_bl_driver builds them (:1006-1010)
    zw(1) = 0.
    do k = 2, n
       zw(k) = zw(k - 1) + dz(k - 1)
    end do
    zw(n + 1) = zw(n) + dz(n)

    ! -- the synthetic moist column: 50 % more vapour, no condensate ------------
    do k = 1, n
       qv(k)  = sqv(k)*1.5
       qt(k)  = qv(k)
       qc(k)  = 0.
       thl(k) = th(k)                       ! qc = qi = 0, so thl == th
       thv(k) = th(k)*(1. + p608*qv(k))
       qke(k) = 1.
    end do
    qnc = 0.; qni = 0.; qnwfa = 0.; qnifa = 0.; qnbca = 0.
    rstoch = 0.; chem1 = 0.; s_awchem = 0.
    vt = 0.; vq = 0.; sgm = 0.
    qc_bl = 0.; cldfra_bl = 0.; qc_bl_old = 0.; cldfra_bl_old = 0.
    edmf_a = 0.; edmf_w = 0.; edmf_qt = 0.; edmf_thl = 0.; edmf_ent = 0.; edmf_qc = 0.
    s_aw = 0.; s_awthl = 0.; s_awqt = 0.; s_awqv = 0.; s_awqc = 0.
    s_awu = 0.; s_awv = 0.; s_awqke = 0.
    s_awqnc = 0.; s_awqni = 0.; s_awqnwfa = 0.; s_awqnifa = 0.; s_awqnbca = 0.
    sub_thl = 0.; sub_sqv = 0.; sub_u = 0.; sub_v = 0.
    det_thl = 0.; det_sqv = 0.; det_sqc = 0.; det_u = 0.; det_v = 0.

    ! -- a deep, strongly heated PBL so the plumes reach their LCL ---------------
    dt        = 20.
    ust       = 0.4
    pblh      = 1500.
    kpbl      = 8
    flt       = 0.3
    flq       = 3.0e-4
    flqv      = flq
    th_sfc    = ts/ex(1)                    ! the driver's double division, README item 2
    fltv      = flt + flq*p608*th_sfc
    psig_shcu = 1.

    call dmp_mf(1, n, dt, zw, dz, p, rho, 1, 0, 1, uu, vv, ww, th, thl, thv, tk, &
         qt, qv, qc, qke, qnc, qni, qnwfa, qnifa, qnbca, ex, vt, vq, sgm, &
         ust, flt, fltv, flq, flqv, pblh, kpbl, dx, landsea, th_sfc, &
         edmf_a, edmf_w, edmf_qt, edmf_thl, edmf_ent, edmf_qc, &
         s_aw, s_awthl, s_awqt, s_awqv, s_awqc, s_awu, s_awv, s_awqke, &
         s_awqnc, s_awqni, s_awqnwfa, s_awqnifa, s_awqnbca, &
         sub_thl, sub_sqv, sub_u, sub_v, det_thl, det_sqv, det_sqc, det_u, det_v, &
         nchem, chem1, s_awchem, .false., qc_bl, cldfra_bl, qc_bl_old, cldfra_bl_old, &
         .true., .true., .false., .false., .false., .false., .false., psig_shcu, &
         maxwidth, ktop, maxmf, ztop, 0, rstoch)

    print '(a,a,a,i0)', '## ', cas, ' BEGIN n=', n
    call block('edmf_a',   cas, edmf_a)
    call block('edmf_w',   cas, edmf_w)
    call block('edmf_qt',  cas, edmf_qt)
    call block('edmf_thl', cas, edmf_thl)
    call block('edmf_ent', cas, edmf_ent)
    call block('edmf_qc',  cas, edmf_qc)
    call block('s_aw',     cas, s_aw)
    call block('s_awthl',  cas, s_awthl)
    call block('s_awqt',   cas, s_awqt)
    call block('s_awqv',   cas, s_awqv)
    call block('s_awqc',   cas, s_awqc)
    call block('s_awu',    cas, s_awu)
    call block('s_awv',    cas, s_awv)
    call block('s_awqke',  cas, s_awqke)
    call block('sub_thl',  cas, sub_thl)
    call block('sub_sqv',  cas, sub_sqv)
    call block('sub_u',    cas, sub_u)
    call block('sub_v',    cas, sub_v)
    call block('det_thl',  cas, det_thl)
    call block('det_sqv',  cas, det_sqv)
    call block('det_sqc',  cas, det_sqc)
    call block('det_u',    cas, det_u)
    call block('det_v',    cas, det_v)
    call block('vt',        cas, vt)
    call block('vq',        cas, vq)
    call block('cldfra_bl', cas, cldfra_bl)
    call block('qc_bl',     cas, qc_bl)
    call scalar('maxwidth',   cas, maxwidth)
    call scalar('maxmf',      cas, maxmf)
    call scalar('ztop_plume', cas, ztop)
    call scalar('ktop_plume', cas, real(ktop, kind_phys))
    print '(a,a,a)', '## ', cas, ' END'
  end subroutine run_spot

end program edmf_ref_driver
