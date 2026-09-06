! Reference driver for the MYNN-EDMF pure-Julia port (Scythe.jl, plan stage S0).
!
! Runs the VERBATIM ccpp-physics module_bl_mynn.F90 on single columns written by
! tools/mynn_dump_columns.jl and prints, at full double precision, everything the Julia
! port (src/mynn_closure.jl, src/mynn_edmf.jl) is checked against:
!
!   mode A  -- mynn_bl_driver called as a host model would (initflag=1 on the first
!              call, then 29 more calls at delt), the column state FROZEN (the driver
!              never updates u/th/q itself); end-of-step state and tendencies printed
!              at steps 1 and 30, for closure 2.5 and 2.6.
!   mode B  -- the per-column call sequence of mynn_bl_driver (GET_PBLH -> SCALE_AWARE ->
!              mym_initialize | mym_condensation -> DMP_mf -> mym_turbulence ->
!              mym_predict -> mynn_tendencies -> retrieve_exchange_coeffs) replicated
!              here so the output of EVERY routine is printed at steps 1 and 30
!              (closure 2.5 only). Its step-30 state must equal mode A's bitwise: that
!              GATE line proves the replication is faithful, and it is what makes the
!              per-routine parity tests in test/test_mynn_closure.jl meaningful.
!
! Output format: a block header line  "## <case> closure=<c> mode=<A|B> step=<s> <name> n=<len>"
! followed by <len> lines "k value" (ES25.17). Scalars have n=1. Parsed by
! test/reference/mynn_fortran_refs.jl (no Fortran at test time).
!
! Host constants come from columns/constants.txt (written from Springsteel's constants,
! see the dump script) and are copied into bl_mynn_common exactly as
! mynnedmf_wrapper_init does, derived quantities included.
!
! Fortran quirks reproduced on purpose (the port must match them before deviating):
!   * ts is the wrapper's T_sfc/exner(1) and the driver divides by exner(1) AGAIN
!     (th_sfc = ts/ex1(kts)), so fltv sees theta_sfc/exner; ~1% at the surface.
!   * the drag term is ust**2/wspd, so the resting case carries wspd = 0.1 with ust = 0.
!   * snow is passed as a zero column (kzero) to mynn_tendencies, as the driver does.
program ref_driver
  use machine, only: kind_phys
  use bl_mynn_common
  use module_bl_mynn
  implicit none

  character(len=*), parameter :: cases(5) = [character(len=24) :: &
       'case1_rest', 'case2_o01_sea', 'case3_tc_rmw', 'case4_convective', 'case5_highwind']
  integer :: ic
  logical :: ok
  ! column state, host-associated by every internal procedure below
  integer :: n
  real(kind_phys) :: ps, ts, qsfc, ust, hfx, qfx, wspd, znt, xland, dx, rmol0, delt
  real(kind_phys), allocatable :: z(:), dz(:), uu(:), vv(:), ww(:), tk(:), th(:), ex(:), &
       p(:), rho(:), sqv(:), sqc(:), sqi(:)
  real(kind_phys), allocatable :: stA(:,:), stB(:,:)

  call read_constants('columns/constants.txt')
  do ic = 1, size(cases)
     inquire(file = 'columns/' // trim(cases(ic)) // '.txt', exist = ok)
     if (.not. ok) then
        print '(a)', '## ' // trim(cases(ic)) // ' MISSING'
        cycle
     end if
     call run_case(trim(cases(ic)))
  end do

contains

  subroutine read_constants(fn)
    character(len=*), intent(in) :: fn
    integer :: u
    open(newunit = u, file = fn, status = 'old', action = 'read')
    read(u, *) cp, cpv, cliq, cice, p608, ep_2, grav, karman, t0c, rcp, r_d, r_v, xlf, xlv
    close(u)
    ! derived exactly as mynnedmf_wrapper_init
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
    print '(a)', '## constants n=24'
    call pv(1, cp); call pv(2, cpv); call pv(3, cliq); call pv(4, cice); call pv(5, p608)
    call pv(6, ep_2); call pv(7, grav); call pv(8, karman); call pv(9, t0c); call pv(10, rcp)
    call pv(11, r_d); call pv(12, r_v); call pv(13, xlf); call pv(14, xlv); call pv(15, xls)
    call pv(16, rvovrd); call pv(17, ep_3); call pv(18, gtr); call pv(19, rk); call pv(20, tv0)
    call pv(21, tv1); call pv(22, xlscp); call pv(23, xlvcp); call pv(24, g_inv)
  end subroutine read_constants

  subroutine pv(k, x)
    integer, intent(in) :: k
    real(kind_phys), intent(in) :: x
    print '(i5,1x,es25.17e3)', k, x
  end subroutine pv

  subroutine block(name, cas, closure, mode, step, x)
    character(len=*), intent(in) :: name, cas, mode
    real(kind_phys), intent(in) :: closure
    integer, intent(in) :: step
    real(kind_phys), intent(in) :: x(:)
    integer :: k
    print '(a,a,a,f4.2,a,a,a,i0,a,a,a,i0)', '## ', cas, ' closure=', closure, ' mode=', mode, &
         ' step=', step, ' ', name, ' n=', size(x)
    do k = 1, size(x)
       call pv(k, x(k))
    end do
  end subroutine block

  subroutine scalar(name, cas, closure, mode, step, x)
    character(len=*), intent(in) :: name, cas, mode
    real(kind_phys), intent(in) :: closure, x
    integer, intent(in) :: step
    call block(name, cas, closure, mode, step, [x])
  end subroutine scalar

  subroutine iscalar(name, cas, closure, mode, step, i)
    character(len=*), intent(in) :: name, cas, mode
    real(kind_phys), intent(in) :: closure
    integer, intent(in) :: step, i
    call block(name, cas, closure, mode, step, [real(i, kind_phys)])
  end subroutine iscalar

  ! ---------------------------------------------------------------------------
  subroutine run_case(cas)
    character(len=*), intent(in) :: cas
    integer :: u, k
    if (allocated(z)) deallocate(z, dz, uu, vv, ww, tk, th, ex, p, rho, sqv, sqc, sqi, stA, stB)

    open(newunit = u, file = 'columns/' // cas // '.txt', status = 'old', action = 'read')
    read(u, *) n
    read(u, *) ps, ts, qsfc, ust, hfx, qfx, wspd, znt, xland, dx, rmol0, delt
    allocate(z(n), dz(n), uu(n), vv(n), ww(n), tk(n), th(n), ex(n), p(n), rho(n), sqv(n), sqc(n), sqi(n))
    do k = 1, n
       read(u, *) z(k), dz(k), uu(k), vv(k), ww(k), tk(k), th(k), ex(k), p(k), rho(k), sqv(k), sqc(k), sqi(k)
    end do
    close(u)
    print '(a,a,a,i0)', '## ', cas, ' BEGIN n=', n

    allocate(stA(n, 12), stB(n, 12))
    call mode_a(cas, 2.5_kind_phys, stA)
    call mode_b(cas, 2.5_kind_phys, stB)
    call gate(cas, stA, stB)
    call mode_a(cas, 2.6_kind_phys, stA)
    print '(a,a,a)', '## ', cas, ' END'
  end subroutine run_case

    ! -- mode A: the public driver, frozen column -----------------------------
    subroutine mode_a(cas, closure, st)
      character(len=*), intent(in) :: cas
      real(kind_phys), intent(in) :: closure
      real(kind_phys), intent(out) :: st(:,:)
      integer, parameter :: nsteps = 30
      integer :: step, initflag
      real(kind_phys), dimension(1, n) :: dz2, u2, v2, w2, th2, sqv2, sqc2, sqi2, sqs2, qnc2, qni2, &
           qnwfa2, qnifa2, qnbca2, ozone2, p2, ex2, rho2, t2, qke, qke_adv, sh3d, sm3d, tsq, qsq, cov, &
           rublten, rvblten, rthblten, rqvblten, rqcblten, rqiblten, rqncblten, rqniblten, rqsblten, &
           rqnwfablten, rqnifablten, rqnbcablten, dozone, exch_h, exch_m, el_pbl, qc_bl, qi_bl, cldfra_bl, &
           edmf_a, edmf_w, edmf_qt, edmf_thl, edmf_ent, edmf_qc, sub_thl3d, sub_sqv3d, det_thl3d, det_sqv3d, &
           rthraten
      real(kind_phys), dimension(1) :: dx1, znt1, xland1, ts1, qsfc1, ps1, ust1, ch1, hfx1, qfx1, rmol1, &
           wspd1, uoce1, voce1, pblh1, maxwidth1, maxmf1, ztop1, emis1
      integer, dimension(1) :: kpbl1, ktop1
      real(kind_phys) :: dum(1)

      dz2(1, :) = dz; u2(1, :) = uu; v2(1, :) = vv; w2(1, :) = ww; th2(1, :) = th
      sqv2(1, :) = sqv; sqc2(1, :) = sqc; sqi2(1, :) = sqi; sqs2 = 0.; qnc2 = 0.; qni2 = 0.
      qnwfa2 = 0.; qnifa2 = 0.; qnbca2 = 0.; ozone2 = 0.; p2(1, :) = p; ex2(1, :) = ex
      rho2(1, :) = rho; t2(1, :) = tk; rthraten = 0.
      qke = 0.; qke_adv = 0.; sh3d = 0.; sm3d = 0.; tsq = 0.; qsq = 0.; cov = 0.
      rublten = 0.; rvblten = 0.; rthblten = 0.; rqvblten = 0.; rqcblten = 0.; rqiblten = 0.
      rqncblten = 0.; rqniblten = 0.; rqsblten = 0.; rqnwfablten = 0.; rqnifablten = 0.; rqnbcablten = 0.
      dozone = 0.; exch_h = 0.; exch_m = 0.; el_pbl = 0.; qc_bl = 0.; qi_bl = 0.; cldfra_bl = 0.
      edmf_a = 0.; edmf_w = 0.; edmf_qt = 0.; edmf_thl = 0.; edmf_ent = 0.; edmf_qc = 0.
      sub_thl3d = 0.; sub_sqv3d = 0.; det_thl3d = 0.; det_sqv3d = 0.
      dx1 = dx; znt1 = znt; xland1 = xland; ts1 = ts; qsfc1 = qsfc; ps1 = ps; ust1 = ust
      ch1 = 0.; hfx1 = hfx; qfx1 = qfx; rmol1 = rmol0; wspd1 = wspd; uoce1 = 0.; voce1 = 0.
      pblh1 = 0.; maxwidth1 = 0.; maxmf1 = 0.; ztop1 = 0.; emis1 = 0.; kpbl1 = 0; ktop1 = 0

      do step = 1, nsteps
         initflag = merge(1, 0, step == 1)
         call mynn_bl_driver(initflag = initflag, restart = .false., cycling = .false., &
              delt = delt, dz = dz2, dx = dx1, znt = znt1, &
              u = u2, v = v2, w = w2, th = th2, sqv3d = sqv2, sqc3d = sqc2, sqi3d = sqi2, &
              sqs3d = sqs2, qnc = qnc2, qni = qni2, qnwfa = qnwfa2, qnifa = qnifa2, qnbca = qnbca2, &
              ozone = ozone2, p = p2, exner = ex2, rho = rho2, t3d = t2, &
              xland = xland1, ts = ts1, qsfc = qsfc1, ps = ps1, &
              ust = ust1, ch = ch1, hfx = hfx1, qfx = qfx1, rmol = rmol1, wspd = wspd1, &
              uoce = uoce1, voce = voce1, qke = qke, qke_adv = qke_adv, sh3d = sh3d, sm3d = sm3d, &
              nchem = 1, kdvel = 1, ndvel = 1, smoke_dbg = .false., emis_ant_no = emis1, &
              mix_chem = .false., enh_mix = .false., rrfs_sd = .false., &
              tsq = tsq, qsq = qsq, cov = cov, &
              rublten = rublten, rvblten = rvblten, rthblten = rthblten, rqvblten = rqvblten, &
              rqcblten = rqcblten, rqiblten = rqiblten, rqncblten = rqncblten, rqniblten = rqniblten, &
              rqsblten = rqsblten, rqnwfablten = rqnwfablten, rqnifablten = rqnifablten, &
              rqnbcablten = rqnbcablten, dozone = dozone, exch_h = exch_h, exch_m = exch_m, &
              pblh = pblh1, kpbl = kpbl1, el_pbl = el_pbl, qc_bl = qc_bl, qi_bl = qi_bl, &
              cldfra_bl = cldfra_bl, bl_mynn_tkeadvect = .false., tke_budget = 0, &
              bl_mynn_cloudpdf = 2, bl_mynn_mixlength = 2, icloud_bl = 1, closure = closure, &
              bl_mynn_edmf = 1, bl_mynn_edmf_mom = 1, bl_mynn_edmf_tke = 0, bl_mynn_mixscalars = 1, &
              bl_mynn_output = 1, bl_mynn_cloudmix = 1, bl_mynn_mixqt = 0, &
              edmf_a = edmf_a, edmf_w = edmf_w, edmf_qt = edmf_qt, edmf_thl = edmf_thl, &
              edmf_ent = edmf_ent, edmf_qc = edmf_qc, sub_thl3d = sub_thl3d, sub_sqv3d = sub_sqv3d, &
              det_thl3d = det_thl3d, det_sqv3d = det_sqv3d, &
              maxwidth = maxwidth1, maxmf = maxmf1, ztop_plume = ztop1, ktop_plume = ktop1, &
              spp_pbl = 0, rthraten = rthraten, &
              flag_qc = .true., flag_qi = .true., flag_qnc = .false., flag_qni = .false., &
              flag_qs = .false., flag_qnwfa = .false., flag_qnifa = .false., flag_qnbca = .false., &
              flag_ozone = .false., &
              ids = 1, ide = 1, jds = 1, jde = 1, kds = 1, kde = n, &
              ims = 1, ime = 1, jms = 1, jme = 1, kms = 1, kme = n, &
              its = 1, ite = 1, jts = 1, jte = 1, kts = 1, kte = n)
         if (step == 1 .or. step == nsteps) then
            call block('el_pbl',    cas, closure, 'A', step, el_pbl(1, :))
            call block('qke',       cas, closure, 'A', step, qke(1, :))
            call block('sh3d',      cas, closure, 'A', step, sh3d(1, :))
            call block('sm3d',      cas, closure, 'A', step, sm3d(1, :))
            call block('tsq',       cas, closure, 'A', step, tsq(1, :))
            call block('qsq',       cas, closure, 'A', step, qsq(1, :))
            call block('cov',       cas, closure, 'A', step, cov(1, :))
            call block('qc_bl',     cas, closure, 'A', step, qc_bl(1, :))
            call block('qi_bl',     cas, closure, 'A', step, qi_bl(1, :))
            call block('cldfra_bl', cas, closure, 'A', step, cldfra_bl(1, :))
            call block('exch_m',    cas, closure, 'A', step, exch_m(1, :))
            call block('exch_h',    cas, closure, 'A', step, exch_h(1, :))
            call block('rublten',   cas, closure, 'A', step, rublten(1, :))
            call block('rvblten',   cas, closure, 'A', step, rvblten(1, :))
            call block('rthblten',  cas, closure, 'A', step, rthblten(1, :))
            call block('rqvblten',  cas, closure, 'A', step, rqvblten(1, :))
            call block('rqcblten',  cas, closure, 'A', step, rqcblten(1, :))
            call block('rqiblten',  cas, closure, 'A', step, rqiblten(1, :))
            call block('edmf_a',    cas, closure, 'A', step, edmf_a(1, :))
            call block('edmf_w',    cas, closure, 'A', step, edmf_w(1, :))
            call block('edmf_qt',   cas, closure, 'A', step, edmf_qt(1, :))
            call block('edmf_thl',  cas, closure, 'A', step, edmf_thl(1, :))
            call block('edmf_ent',  cas, closure, 'A', step, edmf_ent(1, :))
            call block('edmf_qc',   cas, closure, 'A', step, edmf_qc(1, :))
            call block('sub_thl',   cas, closure, 'A', step, sub_thl3d(1, :))
            call block('sub_sqv',   cas, closure, 'A', step, sub_sqv3d(1, :))
            call block('det_thl',   cas, closure, 'A', step, det_thl3d(1, :))
            call block('det_sqv',   cas, closure, 'A', step, det_sqv3d(1, :))
            call scalar('pblh',     cas, closure, 'A', step, pblh1(1))
            call iscalar('kpbl',    cas, closure, 'A', step, kpbl1(1))
            call scalar('rmol',     cas, closure, 'A', step, rmol1(1))
            call scalar('maxwidth', cas, closure, 'A', step, maxwidth1(1))
            call scalar('maxmf',    cas, closure, 'A', step, maxmf1(1))
            call scalar('ztop_plume', cas, closure, 'A', step, ztop1(1))
            call iscalar('ktop_plume', cas, closure, 'A', step, ktop1(1))
         end if
      end do
      st(:, 1) = qke(1, :); st(:, 2) = el_pbl(1, :); st(:, 3) = sh3d(1, :); st(:, 4) = sm3d(1, :)
      st(:, 5) = tsq(1, :); st(:, 6) = qsq(1, :); st(:, 7) = cov(1, :); st(:, 8) = cldfra_bl(1, :)
      st(:, 9) = exch_h(1, :); st(:, 10) = rublten(1, :); st(:, 11) = rthblten(1, :); st(:, 12) = rqvblten(1, :)
      dum(1) = pblh1(1)
      st(1, 9) = st(1, 9) + 0. * dum(1)
    end subroutine mode_a

    ! -- mode B: the per-column sequence replayed by hand ---------------------
    subroutine mode_b(cas, closure, st)
      character(len=*), intent(in) :: cas
      real(kind_phys), intent(in) :: closure
      real(kind_phys), intent(out) :: st(:,:)
      integer, parameter :: nsteps = 30
      integer, parameter :: nchem = 1, ndvel = 1
      integer :: step, k, kpbl, ktop
      real(kind_phys), dimension(n) :: u1, v1, w1, th1, tk1, sqv1, sqc1, sqi1
      real(kind_phys), dimension(n) :: sqw, thl, thetav, qv1, qc1, qi1, kzero, qnc1, qni1, qnwfa1, &
           qnifa1, qnbca1, ozone1, sqs, qke1, el, sh, sm, tsq1, qsq1, cov1, cldfra_bl1d, qc_bl1d, qi_bl1d, &
           cldfra_bl1d_old, qc_bl1d_old, qi_bl1d_old, rstoch_col, edmf_a1, edmf_w1, edmf_qt1, edmf_thl1, &
           edmf_ent1, edmf_qc1, vt, vq, sgm, khtopdown, tkeprodtd, dfm, dfh, dfq, tcd, qcd, pdk, pdt, pdq, &
           pdc, qwt1, qshear1, qbuoy1, qdiss1, diss_heat, du1, dv1, dth1, dqv1, dqc1, dqi1, dqs1, dqnc1, &
           dqni1, dqnwfa1, dqnifa1, dqnbca1, dozone1, k_m1, k_h1, sub_thl, sub_sqv, sub_u, sub_v, &
           det_thl, det_sqv, det_sqc, det_u, det_v, dtl, dqw, dtv, gm, gh, sm2, sh2
      real(kind_phys), dimension(n + 1) :: zw, s_aw1, s_awthl1, s_awqt1, s_awqv1, s_awqc1, s_awu1, &
           s_awv1, s_awqke1, s_awqnc1, s_awqni1, s_awqnwfa1, s_awqnifa1, s_awqnbca1, sd_aw1, sd_awthl1, &
           sd_awqt1, sd_awqv1, sd_awqc1, sd_awu1, sd_awv1, sd_awqke1
      real(kind_phys) :: chem1(n, nchem), s_awchem1(n + 1, nchem), vd1(ndvel)
      real(kind_phys) :: pblh, rmol, psig_bl, psig_shcu, cpm, exnerg, flqv, flqc, th_sfc, flq, flt, fltv, &
           zet, phi_m, pmz, phh, maxwidth, maxmf, ztop

      ! the frozen conserved variables and the face heights (driver :767-771, :1006-1010)
      zw(1) = 0.
      do k = 2, n
         zw(k) = zw(k - 1) + dz(k - 1)
      end do
      zw(n + 1) = zw(n) + dz(n)
      sqs = 0.; kzero = 0.; qnc1 = 0.; qni1 = 0.; qnwfa1 = 0.; qnifa1 = 0.; qnbca1 = 0.; ozone1 = 0.
      rstoch_col = 0.; chem1 = 0.; s_awchem1 = 0.; vd1 = 0.
      ! re-gather the FROZEN host column into working copies (the driver does this every
      ! call: thl/sqw/sqv/sqc are intent(inout) in mynn_tendencies and moisture_check
      ! writes thl, so without fresh copies the replayed column would evolve)
      u1 = uu; v1 = vv; w1 = ww; th1 = th; tk1 = tk; sqv1 = sqv; sqc1 = sqc; sqi1 = sqi
      do k = 1, n
         sqw(k) = sqv1(k) + sqc1(k) + sqi1(k)
         thl(k) = th1(k) - xlvcp / ex(k) * sqc1(k) - xlscp / ex(k) * sqi1(k)
         thetav(k) = th1(k) * (1. + p608 * sqv1(k))
         qv1(k) = sqv1(k) / (1. - sqv1(k))
         qc1(k) = sqc1(k) / (1. - sqv1(k))
         qi1(k) = sqi1(k) / (1. - sqv1(k))
      end do

      ! ---- init block (driver :660-830, INITIALIZE_QKE = .true.) ----
      el = 0.; sh = 0.; sm = 0.; tsq1 = 0.; qsq1 = 0.; cov1 = 0.
      vt = 0.; vq = 0.; sgm = 0.
      cldfra_bl1d = 0.; qc_bl1d = 0.; qi_bl1d = 0.; edmf_w1 = 0.; edmf_a1 = 0.
      pblh = 0.; kpbl = 0; rmol = rmol0
      do k = 1, n
         qke1(k) = 5. * ust * max((ust * 700. - zw(k)) / (max(ust, 0.01) * 700.), 0.01)
      end do
      call get_pblh(1, n, pblh, thetav, qke1, zw, dz, xland, kpbl)
      call scale_aware(dx, pblh, psig_bl, psig_shcu)
      call mym_initialize(1, n, xland, dz, dx, zw, u1, v1, thl, sqv1, pblh, th1, thetav, sh, sm, &
           ust, rmol, el, qke1, tsq1, qsq1, cov1, psig_bl, cldfra_bl1d, 2, edmf_w1, edmf_a1, &
           .true., 0, rstoch_col)
      call block('init_el',   cas, closure, 'B', 0, el)
      call block('init_qke',  cas, closure, 'B', 0, qke1)
      call block('init_tsq',  cas, closure, 'B', 0, tsq1)
      call block('init_qsq',  cas, closure, 'B', 0, qsq1)
      call block('init_cov',  cas, closure, 'B', 0, cov1)
      call block('init_sh',   cas, closure, 'B', 0, sh)
      call block('init_sm',   cas, closure, 'B', 0, sm)
      call scalar('init_pblh', cas, closure, 'B', 0, pblh)
      call iscalar('init_kpbl', cas, closure, 'B', 0, kpbl)
      call scalar('psig_bl',  cas, closure, 'B', 0, psig_bl)
      call scalar('psig_shcu', cas, closure, 'B', 0, psig_shcu)

      ! ---- main loop (driver :900-1330) ----
      do step = 1, nsteps
         ! re-gather the FROZEN host column into working copies (the driver does this every
         ! call: thl/sqw/sqv/sqc are intent(inout) in mynn_tendencies and moisture_check
         ! writes thl, so without fresh copies the replayed column would evolve)
         u1 = uu; v1 = vv; w1 = ww; th1 = th; tk1 = tk; sqv1 = sqv; sqc1 = sqc; sqi1 = sqi
         do k = 1, n
            sqw(k) = sqv1(k) + sqc1(k) + sqi1(k)
            thl(k) = th1(k) - xlvcp / ex(k) * sqc1(k) - xlscp / ex(k) * sqi1(k)
            thetav(k) = th1(k) * (1. + p608 * sqv1(k))
            qv1(k) = sqv1(k) / (1. - sqv1(k))
            qc1(k) = sqc1(k) / (1. - sqv1(k))
            qi1(k) = sqi1(k) / (1. - sqv1(k))
         end do
         cldfra_bl1d_old = cldfra_bl1d; qc_bl1d_old = qc_bl1d; qi_bl1d_old = qi_bl1d
         edmf_a1 = 0.; edmf_w1 = 0.; edmf_qc1 = 0.; edmf_qt1 = 0.; edmf_thl1 = 0.; edmf_ent1 = 0.
         s_aw1 = 0.; s_awthl1 = 0.; s_awqt1 = 0.; s_awqv1 = 0.; s_awqc1 = 0.; s_awu1 = 0.; s_awv1 = 0.
         s_awqke1 = 0.; s_awqnc1 = 0.; s_awqni1 = 0.; s_awqnwfa1 = 0.; s_awqnifa1 = 0.; s_awqnbca1 = 0.
         sd_aw1 = 0.; sd_awthl1 = 0.; sd_awqt1 = 0.; sd_awqv1 = 0.; sd_awqc1 = 0.; sd_awu1 = 0.
         sd_awv1 = 0.; sd_awqke1 = 0.
         sub_thl = 0.; sub_sqv = 0.; sub_u = 0.; sub_v = 0.
         det_thl = 0.; det_sqv = 0.; det_sqc = 0.; det_u = 0.; det_v = 0.
         ! vt, vq, sgm are carried across steps, not reset: mynn_bl_driver zeroes these
         ! automatic arrays only in its init block. mym_condensation (cloudpdf 2) writes
         ! every level, so their entry values are never read; carrying them mirrors the
         ! driver exactly should that ever change.
         dqc1 = 0.; dqi1 = 0.; dqs1 = 0.; dqni1 = 0.; dqnc1 = 0.; dqnwfa1 = 0.; dqnifa1 = 0.
         dqnbca1 = 0.; dozone1 = 0.

         call get_pblh(1, n, pblh, thetav, qke1, zw, dz, xland, kpbl)
         call scale_aware(dx, pblh, psig_bl, psig_shcu)

         cpm = cp * (1. + 0.84 * qv1(1))
         exnerg = (ps / p1000mb)**rcp
         flqv = qfx / rho(1)
         flqc = 0.0
         th_sfc = ts / ex(1)
         flq = flqv + flqc
         flt = hfx / (rho(1) * cpm) - xlvcp * flqc / ex(1)
         fltv = flt + flqv * p608 * th_sfc
         rmol = -karman * gtr * fltv / max(ust**3, 1.0e-6)
         zet = 0.5 * dz(1) * rmol
         zet = max(zet, -20.)
         zet = min(zet, 20.)
         phi_m = phim(zet)
         pmz = phi_m - zet
         phh = phih(zet)

         call mym_condensation(1, n, dx, dz, zw, xland, thl, sqw, sqv1, sqc1, sqi1, sqs, p, ex, &
              tsq1, qsq1, cov1, sh, el, 2, qc_bl1d, qi_bl1d, cldfra_bl1d, pblh, hfx, vt, vq, th1, &
              sgm, rmol, 0, rstoch_col)
         khtopdown = 0.; tkeprodtd = 0.

         call dmp_mf(1, n, delt, zw, dz, p, rho, 1, 0, 1, u1, v1, w1, th1, thl, thetav, tk1, &
              sqw, sqv1, sqc1, qke1, qnc1, qni1, qnwfa1, qnifa1, qnbca1, ex, vt, vq, sgm, &
              ust, flt, fltv, flq, flqv, pblh, kpbl, dx, xland, th_sfc, &
              edmf_a1, edmf_w1, edmf_qt1, edmf_thl1, edmf_ent1, edmf_qc1, &
              s_aw1, s_awthl1, s_awqt1, s_awqv1, s_awqc1, s_awu1, s_awv1, s_awqke1, &
              s_awqnc1, s_awqni1, s_awqnwfa1, s_awqnifa1, s_awqnbca1, &
              sub_thl, sub_sqv, sub_u, sub_v, det_thl, det_sqv, det_sqc, det_u, det_v, &
              nchem, chem1, s_awchem1, .false., qc_bl1d, cldfra_bl1d, qc_bl1d_old, cldfra_bl1d_old, &
              .true., .true., .false., .false., .false., .false., .false., psig_shcu, &
              maxwidth, ktop, maxmf, ztop, 0, rstoch_col)

         ! a standalone Level-2 call, identical to the one mym_turbulence makes first, so the
         ! gradients and the Level-2 stability functions are visible
         sh2 = 0.; sm2 = 0.
         call mym_level2(1, n, dz, u1, v1, thl, thetav, sqw, sqc1, vt, vq, dtl, dqw, dtv, gm, gh, sm2, sh2)

         call mym_turbulence(1, n, xland, closure, dz, dx, zw, u1, v1, thl, thetav, sqc1, sqw, &
              qke1, tsq1, qsq1, cov1, vt, vq, rmol, flt, fltv, flq, pblh, th1, sh, sm, el, &
              dfm, dfh, dfq, tcd, qcd, pdk, pdt, pdq, pdc, qwt1, qshear1, qbuoy1, qdiss1, 0, &
              psig_bl, psig_shcu, cldfra_bl1d, 2, edmf_w1, edmf_a1, tkeprodtd, 0, rstoch_col)

         if (step == 1 .or. step == nsteps) then
            call scalar('flt',   cas, closure, 'B', step, flt)
            call scalar('flq',   cas, closure, 'B', step, flq)
            call scalar('fltv',  cas, closure, 'B', step, fltv)
            call scalar('rmol',  cas, closure, 'B', step, rmol)
            call scalar('zet',   cas, closure, 'B', step, zet)
            call scalar('pmz',   cas, closure, 'B', step, pmz)
            call scalar('phh',   cas, closure, 'B', step, phh)
            call scalar('pblh',  cas, closure, 'B', step, pblh)
            call iscalar('kpbl', cas, closure, 'B', step, kpbl)
            call scalar('psig_bl', cas, closure, 'B', step, psig_bl)
            call scalar('psig_shcu', cas, closure, 'B', step, psig_shcu)
            call block('vt',     cas, closure, 'B', step, vt)
            call block('vq',     cas, closure, 'B', step, vq)
            call block('sgm',    cas, closure, 'B', step, sgm)
            call block('qc_bl',  cas, closure, 'B', step, qc_bl1d)
            call block('qi_bl',  cas, closure, 'B', step, qi_bl1d)
            call block('cldfra_bl', cas, closure, 'B', step, cldfra_bl1d)
            call block('dtl',    cas, closure, 'B', step, dtl)
            call block('dqw',    cas, closure, 'B', step, dqw)
            call block('dtv',    cas, closure, 'B', step, dtv)
            call block('gm',     cas, closure, 'B', step, gm)
            call block('gh',     cas, closure, 'B', step, gh)
            call block('sm2',    cas, closure, 'B', step, sm2)
            call block('sh2',    cas, closure, 'B', step, sh2)
            call block('el',     cas, closure, 'B', step, el)
            call block('sh',     cas, closure, 'B', step, sh)
            call block('sm',     cas, closure, 'B', step, sm)
            call block('dfm',    cas, closure, 'B', step, dfm)
            call block('dfh',    cas, closure, 'B', step, dfh)
            call block('dfq',    cas, closure, 'B', step, dfq)
            call block('pdk',    cas, closure, 'B', step, pdk)
            call block('pdt',    cas, closure, 'B', step, pdt)
            call block('pdq',    cas, closure, 'B', step, pdq)
            call block('pdc',    cas, closure, 'B', step, pdc)
            call block('tcd',    cas, closure, 'B', step, tcd)
            call block('qcd',    cas, closure, 'B', step, qcd)
            call block('edmf_a', cas, closure, 'B', step, edmf_a1)
            call block('edmf_w', cas, closure, 'B', step, edmf_w1)
            call block('edmf_qt', cas, closure, 'B', step, edmf_qt1)
            call block('edmf_thl', cas, closure, 'B', step, edmf_thl1)
            call block('edmf_ent', cas, closure, 'B', step, edmf_ent1)
            call block('edmf_qc', cas, closure, 'B', step, edmf_qc1)
            call block('s_aw',    cas, closure, 'B', step, s_aw1)
            call block('s_awthl', cas, closure, 'B', step, s_awthl1)
            call block('s_awqt',  cas, closure, 'B', step, s_awqt1)
            call block('s_awqv',  cas, closure, 'B', step, s_awqv1)
            call block('s_awqc',  cas, closure, 'B', step, s_awqc1)
            call block('s_awu',   cas, closure, 'B', step, s_awu1)
            call block('s_awv',   cas, closure, 'B', step, s_awv1)
            call block('s_awqke', cas, closure, 'B', step, s_awqke1)
            call block('sub_thl', cas, closure, 'B', step, sub_thl)
            call block('sub_sqv', cas, closure, 'B', step, sub_sqv)
            call block('sub_u',   cas, closure, 'B', step, sub_u)
            call block('sub_v',   cas, closure, 'B', step, sub_v)
            call block('det_thl', cas, closure, 'B', step, det_thl)
            call block('det_sqv', cas, closure, 'B', step, det_sqv)
            call block('det_sqc', cas, closure, 'B', step, det_sqc)
            call block('det_u',   cas, closure, 'B', step, det_u)
            call block('det_v',   cas, closure, 'B', step, det_v)
            call scalar('maxwidth', cas, closure, 'B', step, maxwidth)
            call scalar('maxmf',    cas, closure, 'B', step, maxmf)
            call scalar('ztop_plume', cas, closure, 'B', step, ztop)
            call iscalar('ktop_plume', cas, closure, 'B', step, ktop)
            call block('qke_pre_predict', cas, closure, 'B', step, qke1)
         end if

         call mym_predict(1, n, closure, delt, dz, ust, flt, flq, pmz, phh, el, dfq, rho, &
              pdk, pdt, pdq, pdc, qke1, tsq1, qsq1, cov1, s_aw1, s_awqke1, 0, qwt1, qdiss1, 0)

         do k = 1, n - 1
            diss_heat(k) = min(max(1.0 * (qke1(k)**1.5) / (b1 * max(0.5 * (el(k) + el(k + 1)), 1.)) / cp, 0.0), 0.002)
            diss_heat(k) = diss_heat(k) * exp(-10000. / max(p(k), 1.))
         end do
         diss_heat(n) = 0.

         call mynn_tendencies(1, n, 1, delt, dz, rho, u1, v1, th1, tk1, qv1, qc1, qi1, kzero, qnc1, qni1, &
              ps, p, ex, thl, sqv1, sqc1, sqi1, kzero, sqw, qnwfa1, qnifa1, qnbca1, ozone1, &
              ust, flt, flq, flqv, flqc, wspd, 0.0_kind_phys, 0.0_kind_phys, tsq1, qsq1, cov1, tcd, qcd, &
              dfm, dfh, dfq, du1, dv1, dth1, dqv1, dqc1, dqi1, dqs1, dqnc1, dqni1, dqnwfa1, dqnifa1, &
              dqnbca1, dozone1, diss_heat, s_aw1, s_awthl1, s_awqt1, s_awqv1, s_awqc1, s_awu1, s_awv1, &
              s_awqnc1, s_awqni1, s_awqnwfa1, s_awqnifa1, s_awqnbca1, sd_aw1, sd_awthl1, sd_awqt1, &
              sd_awqv1, sd_awqc1, sd_awu1, sd_awv1, sub_thl, sub_sqv, sub_u, sub_v, det_thl, det_sqv, &
              det_sqc, det_u, det_v, .true., .true., .false., .false., .false., .false., .false., .false., &
              cldfra_bl1d, 1, 0, 1, 1, 1)
         call retrieve_exchange_coeffs(1, n, dfm, dfh, dz, k_m1, k_h1)

         if (step == 1 .or. step == nsteps) then
            call block('qke',      cas, closure, 'B', step, qke1)
            call block('tsq',      cas, closure, 'B', step, tsq1)
            call block('qsq',      cas, closure, 'B', step, qsq1)
            call block('cov',      cas, closure, 'B', step, cov1)
            call block('diss_heat', cas, closure, 'B', step, diss_heat)
            call block('du',       cas, closure, 'B', step, du1)
            call block('dv',       cas, closure, 'B', step, dv1)
            call block('dth',      cas, closure, 'B', step, dth1)
            call block('dqv',      cas, closure, 'B', step, dqv1)
            call block('dqc',      cas, closure, 'B', step, dqc1)
            call block('dqi',      cas, closure, 'B', step, dqi1)
            call block('k_m',      cas, closure, 'B', step, k_m1)
            call block('k_h',      cas, closure, 'B', step, k_h1)
         end if
      end do
      st(:, 1) = qke1; st(:, 2) = el; st(:, 3) = sh; st(:, 4) = sm; st(:, 5) = tsq1; st(:, 6) = qsq1
      st(:, 7) = cov1; st(:, 8) = cldfra_bl1d; st(:, 9) = k_h1; st(:, 10) = du1; st(:, 11) = dth1
      st(:, 12) = dqv1
    end subroutine mode_b

    subroutine gate(cas, a, b)
      character(len=*), intent(in) :: cas
      real(kind_phys), intent(in) :: a(:,:), b(:,:)
      character(len=10), parameter :: nm(12) = [character(len=10) :: 'qke', 'el', 'sh', 'sm', 'tsq', &
           'qsq', 'cov', 'cldfra_bl', 'k_h', 'du', 'dth', 'dqv']
      integer :: j
      logical :: pass
      pass = .true.
      do j = 1, 12
         print '(a,a,a,a,a,es12.4)', '## ', cas, ' GATE ', trim(nm(j)), ' maxabsdiff=', maxval(abs(a(:, j) - b(:, j)))
         if (any(a(:, j) /= b(:, j))) pass = .false.
      end do
      if (pass) then
         print '(a,a,a)', '## ', cas, ' GATE PASS (mode B == mode A bitwise at step 30)'
      else
         print '(a,a,a)', '## ', cas, ' GATE FAIL'
      end if
    end subroutine gate

end program ref_driver
