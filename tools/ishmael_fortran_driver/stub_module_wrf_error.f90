! stub_module_wrf_error.f90
!
! module_mp_jensen_ishmael.F does `use module_wrf_error` at the top but
! (verified by grep) never calls anything from it -- the CM1 build links
! WRF's real module_wrf_error.F (registry error/message plumbing) which
! this standalone driver has no need of and does not want to drag in.
! This is an empty stand-in so the `use` statement resolves; it is not a
! modification of the ISHMAEL physics module itself.
module module_wrf_error
  implicit none
end module module_wrf_error
