! Minimal stand-in for ccpp-physics/physics/hooks/machine.F: only `kind_phys` is
! consumed (by bl_mynn_common). Double precision, as in a default (non-SINGLE_PREC)
! CCPP build.
module machine
  implicit none
  integer, parameter :: kind_phys = 8
end module machine
