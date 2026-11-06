! Reference driver for the GFDL/HWRF v7 sea-surface roughness fits.
!
! Prints znot_m_v7(U10) and znot_t_v7(U10) at the wind speeds test/test_surface_layer.jl
! checks the Julia port (src/mc_surface_layer.jl) against. The module it calls is a
! VERBATIM copy of ccpp-physics/physics/SFC_Layer/GFDL/module_sf_exchcoef.f90
! (Apache-2.0; the v7 fits are Bin Liu, NOAA/NCEP/EMC 2018) sitting next to this file.
!
! Build (see run_sfc.sh): gfortran -fdefault-real-8 -ffp-contract=off -O0
! -- that module declares its arguments as bare `real`, so without the promotion the
! output is single precision and useless as a double-precision reference.
!
! Output: one line per wind speed, `uref z0m z0t` in ES25.17E3.
PROGRAM sfc_ref_driver
  USE module_sf_exchcoef, ONLY: znot_m_v7, znot_t_v7
  IMPLICIT NONE
  INTEGER, PARAMETER :: nu = 17
  REAL :: u(nu), z0m, z0t
  INTEGER :: i
  DATA u / 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, &
           45.0, 50.0, 55.0, 60.0, 70.0, 80.0, 85.0 /
  WRITE(*,'(A)') '## gfdl_v7 znot uref z0m z0t'
  DO i = 1, nu
     CALL znot_m_v7(u(i), z0m)
     CALL znot_t_v7(u(i), z0t)
     WRITE(*,'(3ES25.17E3)') u(i), z0m, z0t
  END DO
END PROGRAM sfc_ref_driver
