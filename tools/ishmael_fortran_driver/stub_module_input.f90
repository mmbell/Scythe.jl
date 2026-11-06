! stub_module_input.f90
!
! module_mp_jensen_ishmael.F's mp_jensen_ishmael subroutine (the one
! CM1-column driver routine we do NOT call from ref_driver.f90) does
! `use input, only : timestats, mytime, time_microphy, time_dbz, ibr, ier,
! jbr, jer, kbr, ker` at line 325. CM1's real `input` module is its
! namelist/grid-config module, far outside ISHMAEL's own physics and not
! needed here since ref_driver.f90 never calls mp_jensen_ishmael -- this
! is a minimal stand-in with matching symbol names purely so the module
! compiles standalone.
module input
  implicit none
  integer, parameter :: ibr = 1, ier = 1, jbr = 1, jer = 1, kbr = 1, ker = 1
  real :: mytime = 0.0, time_microphy = 0.0, time_dbz = 0.0
contains
  subroutine timestats(dummy)
    real, intent(in), optional :: dummy
  end subroutine timestats
end module input
