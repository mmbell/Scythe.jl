! Spot-check driver for the MYNN-EDMF pure-function reference values that the main
! ref_driver.f90 never prints.
!
! mynn_bl_driver only reaches esat_blend/qsat_blend/xl_blend from inside
! mym_condensation, whose *outputs* (vt, vq, sgm, cldfra_bl) are what the main
! reference dumps -- so the three blend functions have no direct reference block.
! Likewise, every one of the five reference columns has rmol <= 0, so phim/phih are
! only ever evaluated on their UNSTABLE branch; the Cheng-Brutsaert stable branch
! (zet > 0) is never exercised.
!
! This program fills both gaps: it reads columns/constants.txt exactly as ref_driver
! does and prints
!
!   esat_blend(t), xl_blend(t)          at 12 temperatures spanning both blend edges
!                                       and the XC = max(-80, t-t0c) clamp
!   qsat_blend(t, P)                    at those temperatures for 3 pressures, one of
!                                       which activates the min(ES, P*0.15) cap
!   phim(zet), phih(zet)                at 13 zet spanning both branches
!   boulac_length0(k, ...)              lb1/lb2 at every level of a synthetic 20-level
!                                       column (the CASE-2 mixing length never calls it,
!                                       so it has no block in the main reference either)
!
! in the same "## <name> n=<len>" / "k value" (ES25.17E3) format as ref_driver.f90.
! The numbers are transcribed by hand into test/test_mynn_closure.jl -- this program is
! NOT run at test time. Build it with the SAME flags as the reference build in run.sh:
!
!   cd tools/mynn_fortran_driver/build/r8 && \
!     /opt/homebrew/bin/gfortran -ffree-line-length-none -O0 -ffp-contract=off \
!       -fdefault-real-8 -fdefault-double-8 stub_machine.o bl_mynn_common.o \
!       module_bl_mynn.o ../../blend_ref_driver.f90 -o blend_ref_driver && \
!     ./blend_ref_driver
!
program blend_ref_driver
  use machine, only: kind_phys
  use bl_mynn_common
  use module_bl_mynn
  implicit none

  integer, parameter :: nt = 12, np = 3, nz = 13, nk = 20
  real(kind_phys), parameter :: tt(nt) = [310.0d0, 300.0d0, 288.0d0, 273.16d0, &
       270.0d0, 267.16d0, 265.0d0, 260.0d0, 250.0d0, 240.0d0, 230.0d0, 180.0d0]
  real(kind_phys), parameter :: pp(np) = [100000.0d0, 20000.0d0, 30000.0d0]
  real(kind_phys), parameter :: zz(nz) = [0.0d0, 0.1d0, 0.5d0, 1.0d0, 2.0d0, 5.0d0, &
       10.0d0, 20.0d0, -0.01d0, -0.1d0, -0.5d0, -1.0d0, -20.0d0]
  integer :: i, j, k
  real(kind_phys) :: v(nt*np)
  real(kind_phys) :: bdz(nk), bzw(nk+1), bth(nk), bqtke(nk), blb1(nk), blb2(nk)

  call read_constants('columns/constants.txt')

  print '(a,i0)', '## blend_t n=', nt
  do i = 1, nt
     call pv(i, tt(i))
  end do
  print '(a,i0)', '## blend_esat n=', nt
  do i = 1, nt
     call pv(i, esat_blend(tt(i)))
  end do
  print '(a,i0)', '## blend_xl n=', nt
  do i = 1, nt
     call pv(i, xl_blend(tt(i)))
  end do
  ! qsat is printed as one flat block, temperature-major: k = (i-1)*np + j
  k = 0
  do i = 1, nt
     do j = 1, np
        k = k + 1
        v(k) = qsat_blend(tt(i), pp(j))
     end do
  end do
  print '(a,i0)', '## blend_p n=', np
  do j = 1, np
     call pv(j, pp(j))
  end do
  print '(a,i0)', '## blend_qsat n=', nt*np
  do k = 1, nt*np
     call pv(k, v(k))
  end do

  print '(a,i0)', '## stab_zet n=', nz
  do i = 1, nz
     call pv(i, zz(i))
  end do
  print '(a,i0)', '## stab_phim n=', nz
  do i = 1, nz
     call pv(i, phim(zz(i)))
  end do
  print '(a,i0)', '## stab_phih n=', nz
  do i = 1, nz
     call pv(i, phih(zz(i)))
  end do

  ! ---- boulac_length0 on a synthetic column ----
  ! 100 m layers; theta neutral to 800 m, a 6 K inversion, then 4 K/km aloft; qtke
  ! spanning 0.01 to 40 m^2/s^2 so that both the "parcel stops inside a layer" branch
  ! and the "runs off the end of the column" branch are taken.
  do k = 1, nk
     bdz(k) = 100.0d0
  end do
  bzw(1) = 0.0d0
  do k = 2, nk + 1
     bzw(k) = bzw(k-1) + bdz(k-1)
  end do
  do k = 1, nk
     bth(k) = 300.0d0 + 0.5d0*bzw(k)*0.001d0
     if (k > 8)  bth(k) = bth(k) + 6.0d0
     if (k > 12) bth(k) = bth(k) + 4.0d0*(bzw(k) - bzw(13))*0.001d0
     bqtke(k) = 0.01d0 + 40.0d0*real(k - 1, kind_phys)/real(nk - 1, kind_phys)
  end do
  do k = 1, nk
     call boulac_length0(k, 1, nk, bzw, bdz, bqtke, bth, blb1(k), blb2(k))
  end do
  print '(a,i0)', '## boulac_dz n=', nk
  do k = 1, nk
     call pv(k, bdz(k))
  end do
  print '(a,i0)', '## boulac_theta n=', nk
  do k = 1, nk
     call pv(k, bth(k))
  end do
  print '(a,i0)', '## boulac_qtke n=', nk
  do k = 1, nk
     call pv(k, bqtke(k))
  end do
  print '(a,i0)', '## boulac_lb1 n=', nk
  do k = 1, nk
     call pv(k, blb1(k))
  end do
  print '(a,i0)', '## boulac_lb2 n=', nk
  do k = 1, nk
     call pv(k, blb2(k))
  end do

contains

  subroutine read_constants(fn)
    character(len=*), intent(in) :: fn
    integer :: u
    open(newunit = u, file = fn, status = 'old', action = 'read')
    read(u, *) cp, cpv, cliq, cice, p608, ep_2, grav, karman, t0c, rcp, r_d, r_v, xlf, xlv
    close(u)
    ! derived exactly as mynnedmf_wrapper_init (same block as ref_driver.f90)
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

  subroutine pv(k, x)
    integer, intent(in) :: k
    real(kind_phys), intent(in) :: x
    print '(i5,1x,es25.17e3)', k, x
  end subroutine pv

end program blend_ref_driver
