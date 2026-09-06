! Spot-check driver for moisture_check, the one part-2 routine with no reference block.
!
! moisture_check (module_bl_mynn.F90 :5133-5220) is called from inside mynn_tendencies
! (:5016) and none of its arguments is printed by ref_driver.f90 -- and, worse, all five
! reference columns are non-negative everywhere, so in the main reference the routine
! runs only on its NO-OP path. Its correction path (condense vapour into a negative
! condensate, borrow vapour from the layer below, then redistribute the borrow over the
! whole column) is therefore untested by the main harness, and it is exactly the path a
! Scythe column with negative water will take.
!
! This program calls it on two synthetic 6-level columns:
!
!   column A -- scattered negative qc/qi plus one negative qv, so the condensation
!               correction and the single-layer borrow both fire but the final
!               column-wide redistribution (:5198-5216) does not (dqv2(k=1) = 0).
!   column B -- qv(1) driven far below qvmin so that the borrow reaches the bottom and
!               the redistribution loop DOES fire with aa < 0.5.
!
! Output is the "## <name> n=<len>" / "k value" (ES25.17E3) format of ref_driver.f90,
! for the five corrected state arrays and the five corrected tendencies of each column.
! The numbers are transcribed by hand into test/test_mynn_closure.jl; this program is
! NOT run at test time and run.sh does not build it. Build it with the SAME flags as
! the reference build:
!
!   cd tools/mynn_fortran_driver/build/r8 && \
!     /opt/homebrew/bin/gfortran -ffree-line-length-none -O0 -ffp-contract=off \
!       -fdefault-real-8 -fdefault-double-8 stub_machine.o bl_mynn_common.o \
!       module_bl_mynn.o ../../moisture_ref_driver.f90 -o moisture_ref_driver && \
!     ./moisture_ref_driver
!
program moisture_ref_driver
  use machine, only: kind_phys
  use bl_mynn_common
  use module_bl_mynn
  implicit none

  integer, parameter :: nk = 6
  real(kind_phys), parameter :: delt = 20.0d0
  real(kind_phys) :: dp(nk), ex(nk)
  real(kind_phys) :: qv(nk), qc(nk), qi(nk), qs(nk), th(nk)
  real(kind_phys) :: dqv(nk), dqc(nk), dqi(nk), dqs(nk), dth(nk)
  integer :: u

  ! the host constants, read exactly as ref_driver.f90 does (moisture_check uses
  ! xlvcp and xlscp, which mynnedmf_wrapper_init derives from xlv, xlf and cp)
  open(newunit = u, file = 'columns/constants.txt', status = 'old', action = 'read')
  read(u, *) cp, cpv, cliq, cice, p608, ep_2, grav, karman, t0c, rcp, r_d, r_v, xlf, xlv
  close(u)
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

  dp = 2000.0d0
  ex = 0.9d0

  ! ---- column A: condensation correction + a single-layer borrow ----------------
  qv = [ 1.0d-3, -5.0d-4,  2.0d-3, 3.0d-3, 1.0d-3, 5.0d-4 ]
  qc = [-1.0d-5,  2.0d-5,  0.0d0, -3.0d-6, 0.0d0,  0.0d0  ]
  qi = [ 0.0d0,  -2.0d-6,  0.0d0,  0.0d0,  1.0d-6, 0.0d0  ]
  qs = 0.0d0
  th = 300.0d0
  dqv = 0.0d0; dqc = 0.0d0; dqi = 0.0d0; dqs = 0.0d0; dth = 0.0d0
  call moisture_check(nk, delt, dp, ex, qv, qc, qi, qs, th, dqv, dqc, dqi, dqs, dth)
  call block('A_qv', qv); call block('A_qc', qc); call block('A_qi', qi)
  call block('A_qs', qs); call block('A_th', th)
  call block('A_dqv', dqv); call block('A_dqc', dqc); call block('A_dqi', dqi)
  call block('A_dqs', dqs); call block('A_dth', dth)

  ! ---- column B: the borrow reaches k = 1 and the redistribution fires ----------
  qv = [-2.0d-3,  1.0d-3,  2.0d-3, 3.0d-3, 4.0d-3, 5.0d-3 ]
  qc = [ 1.0d-5, -4.0d-5,  0.0d0,  0.0d0,  0.0d0,  0.0d0  ]
  qi = [ 0.0d0,   0.0d0,  -1.0d-5, 0.0d0,  0.0d0,  0.0d0  ]
  qs = [ 0.0d0,   0.0d0,   0.0d0, -2.0d-6, 0.0d0,  0.0d0  ]
  th = [ 295.0d0, 296.0d0, 297.0d0, 298.0d0, 299.0d0, 300.0d0 ]
  dqv = 0.0d0; dqc = 0.0d0; dqi = 0.0d0; dqs = 0.0d0; dth = 0.0d0
  call moisture_check(nk, delt, dp, ex, qv, qc, qi, qs, th, dqv, dqc, dqi, dqs, dth)
  call block('B_qv', qv); call block('B_qc', qc); call block('B_qi', qi)
  call block('B_qs', qs); call block('B_th', th)
  call block('B_dqv', dqv); call block('B_dqc', dqc); call block('B_dqi', dqi)
  call block('B_dqs', dqs); call block('B_dth', dth)

contains

  subroutine block(name, x)
    character(len=*), intent(in) :: name
    real(kind_phys), intent(in) :: x(:)
    integer :: k
    print '(a,a,a,i0)', '## ', name, ' n=', size(x)
    do k = 1, size(x)
       print '(i5,1x,es25.17e3)', k, x(k)
    end do
  end subroutine block

end program moisture_ref_driver
