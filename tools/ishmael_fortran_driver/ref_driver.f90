! ref_driver.f90
!
! Fortran reference driver for Scythe.jl's ISHMAEL ice-microphysics port
! (Stage S6a). Prints, at hand-picked state points and at full float
! precision (ES16.8), the inputs and outputs of the CM1
! module_mp_jensen_ishmael.F unit routines that Scythe.jl's
! src/ishmael_tables.jl and src/ishmael.jl port: var_check,
! capacitance_gamma, get_igr, access_lookup_table (itab and itabr
! patterns), and vaporgrow.
!
! Built against a COPY of the CM1 source (module_mp_jensen_ishmael.F in
! this directory) with exactly one substantive change: five additional
! routines (var_check, capacitance_gamma, get_igr, access_lookup_table,
! vaporgrow) and polysvp are added to the module's `public ::` list so
! this external driver program can call them directly. No other lines
! were touched -- see the "One-line change" comment at the top of that
! file's public statement.
!
! Compiled/run at NATIVE SINGLE PRECISION (gfortran default real,
! matching how CM1 itself runs jensen_ishmael) with -fconvert=big-endian
! (matching the big_endian convert= on the .bin table reads). Must be run
! from a directory containing ishmael-qi-qc.bin, ishmael-qi-qr.bin,
! ishmael-gamma-tab.bin (jensen_ishmael_init reads units 20/30/40 from the
! CWD) -- see run.sh.

program ref_driver
  use module_mp_jensen_ishmael
  implicit none

  ! Local mirrors of module-private scalar parameters (module contains
  ! `private` at the top, so named constants are not visible here even
  ! though they are true Fortran PARAMETERs -- these are transcribed
  ! literals, not independent physics choices; see the module source
  ! for each: NU line 628, ao line 634, RD/RV/T0/RHOI/G_HOME/PI lines
  ! 35-47, LAMMINR/LAMMAXR lines 639-640).
  real, parameter :: PI    = 3.14159265
  real, parameter :: NU    = 4.0
  real, parameter :: ao    = 0.1e-6
  real, parameter :: RD    = 287.15
  real, parameter :: RV    = 461.5
  real, parameter :: CP    = 1004.0
  real, parameter :: T0    = 273.15
  real, parameter :: RHOI  = 920.0
  real, parameter :: G_HOME= 9.8
  real, parameter :: LAMMINR = 1./2800.E-6
  real, parameter :: LAMMAXR = 1./20.E-6

  real :: gammnu, i_gammnu, fourthirdspi, i_cp
  integer :: i

  ! var_check / capacitance_gamma / vaporgrow chain state points
  integer, parameter :: NPT = 10
  real, dimension(NPT) :: pt_temp   ! K
  real, dimension(NPT) :: pt_pres   ! Pa
  real, dimension(NPT) :: pt_relh_ice  ! RH wrt ice (drives sui/sup via qv)
  real, dimension(NPT) :: pt_qi     ! kg/kg  (qidum)
  real, dimension(NPT) :: pt_ds0    ! deltastr guess (pre var_check)
  real, dimension(NPT) :: pt_ani0   ! ani guess (pre var_check), m
  real, dimension(NPT) :: pt_cni0   ! cni guess (pre var_check), m
  real, dimension(NPT) :: pt_rb0    ! rhobar guess (pre var_check), kg/m3
  real, dimension(NPT) :: pt_ni     ! nidum, #/m3
  character(len=48), dimension(NPT) :: pt_label

  real :: qidum, dsdum, ani, cni, rbdum, nidum, aidum, cidum
  real :: rni, alphstr, alphv, betam
  real :: capgam
  real :: temp, pres, qv, svpi, svpl, qvi, qvs, qs0, sui, sup
  real :: mu, dv, kt, nsch, npr, xxls, xxlv, xxlf
  real :: igr
  real :: dt, rimesum, iwci, rhodum
  real :: vtbarb, vtbarbm, vtbarbz, anf, cnf, rnf, iwcf
  real :: fvdum, fhdum, rdout, dsdumout

  ! access_lookup_table state points (itab: index 1,2 ; itabr: index 1..6)
  integer, parameter :: NLUT = 10
  integer, dimension(NLUT) :: lut_jj, lut_ii, lut_i, lut_k
  real,    dimension(NLUT) :: lut_d1, lut_d2, lut_d4, lut_d5
  real :: proc

  ! get_igr dedicated temperature sweep
  integer, parameter :: NIGR = 11
  real, dimension(NIGR) :: igr_temp

  gammnu       = gamma(NU)
  i_gammnu     = 1.0/gammnu
  fourthirdspi = 4./3.*PI
  i_cp         = 1./CP

  call jensen_ishmael_init(0)

  print *, '=== ISHMAEL Fortran reference driver (Scythe.jl Stage S6a) ==='
  print *, 'gammnu, i_gammnu, fourthirdspi ='
  write(*,'(3ES16.8)') gammnu, i_gammnu, fourthirdspi

  !----------------------------------------------------------------------
  ! 10 state points spanning: columnar (igr>1, T~-6C), planar (igr<1,
  ! T~-12C), density-clamp-high, density-clamp-low, T=-35C boundary,
  ! T just below 0C, T just above 0C (T0 passthrough branch), tiny ice
  ! (small-ice-limit branch), large ice ani>=cni branch, large ice
  ! cni>ani branch.
  !----------------------------------------------------------------------
  pt_label(1)  = 'columnar T=-6C moderate'
  pt_temp(1)   = T0 - 6.0;  pt_pres(1) = 85000.0; pt_relh_ice(1) = 1.05
  pt_qi(1)     = 1.0e-5; pt_ds0(1) = 1.2; pt_ani0(1) = 5.0e-5; pt_cni0(1) = 1.5e-4
  pt_rb0(1)    = 400.0;  pt_ni(1)  = 2.0e4

  pt_label(2)  = 'planar T=-12C moderate'
  pt_temp(2)   = T0 - 12.0; pt_pres(2) = 80000.0; pt_relh_ice(2) = 1.03
  pt_qi(2)     = 1.0e-5; pt_ds0(2) = 0.75; pt_ani0(2) = 1.5e-4; pt_cni0(2) = 5.0e-5
  pt_rb0(2)    = 300.0;  pt_ni(2)  = 2.0e4

  pt_label(3)  = 'density-clamp-high (>RHOI)'
  pt_temp(3)   = T0 - 6.0;  pt_pres(3) = 85000.0; pt_relh_ice(3) = 1.02
  pt_qi(3)     = 5.0e-5; pt_ds0(3) = 1.0; pt_ani0(3) = 3.0e-5; pt_cni0(3) = 3.0e-5
  pt_rb0(3)    = 950.0;  pt_ni(3)  = 5.0e2

  pt_label(4)  = 'density-clamp-low (<50)'
  pt_temp(4)   = T0 - 20.0; pt_pres(4) = 70000.0; pt_relh_ice(4) = 1.10
  pt_qi(4)     = 1.0e-7; pt_ds0(4) = 0.9; pt_ani0(4) = 3.0e-4; pt_cni0(4) = 2.7e-4
  pt_rb0(4)    = 10.0;   pt_ni(4)  = 1.0e3

  pt_label(5)  = 'homogeneous-freezing boundary T=-35C'
  pt_temp(5)   = T0 - 35.0; pt_pres(5) = 60000.0; pt_relh_ice(5) = 1.15
  pt_qi(5)     = 2.0e-6; pt_ds0(5) = 0.7; pt_ani0(5) = 2.0e-5; pt_cni0(5) = 1.0e-5
  pt_rb0(5)    = 500.0;  pt_ni(5)  = 5.0e5

  pt_label(6)  = 'near 0C from below, T=-0.5C'
  pt_temp(6)   = T0 - 0.5;  pt_pres(6) = 95000.0; pt_relh_ice(6) = 1.005
  pt_qi(6)     = 5.0e-6; pt_ds0(6) = 1.0; pt_ani0(6) = 1.0e-4; pt_cni0(6) = 1.0e-4
  pt_rb0(6)    = 200.0;  pt_ni(6)  = 1.0e4

  pt_label(7)  = 'above 0C, T=+2C (melting/passthrough)'
  pt_temp(7)   = T0 + 2.0;  pt_pres(7) = 95000.0; pt_relh_ice(7) = 1.0
  pt_qi(7)     = 5.0e-6; pt_ds0(7) = 1.0; pt_ani0(7) = 1.0e-4; pt_cni0(7) = 1.0e-4
  pt_rb0(7)    = 200.0;  pt_ni(7)  = 1.0e4

  pt_label(8)  = 'tiny ice (small-ice-limit branch)'
  pt_temp(8)   = T0 - 10.0; pt_pres(8) = 85000.0; pt_relh_ice(8) = 1.02
  pt_qi(8)     = 1.0e-9; pt_ds0(8) = 1.0; pt_ani0(8) = 2.0e-6; pt_cni0(8) = 2.0e-6
  pt_rb0(8)    = 920.0;  pt_ni(8)  = 5.0e7

  pt_label(9)  = 'large ice ani>=cni (large-ice-limit branch A)'
  pt_temp(9)   = T0 - 8.0;  pt_pres(9) = 85000.0; pt_relh_ice(9) = 1.05
  pt_qi(9)     = 5.0e-3; pt_ds0(9) = 0.7; pt_ani0(9) = 2.0e-3; pt_cni0(9) = 1.0e-3
  pt_rb0(9)    = 500.0;  pt_ni(9)  = 5.0e2

  pt_label(10) = 'large ice cni>ani (large-ice-limit branch B)'
  pt_temp(10)  = T0 - 8.0;  pt_pres(10) = 85000.0; pt_relh_ice(10) = 1.05
  pt_qi(10)    = 5.0e-3; pt_ds0(10) = 1.25; pt_ani0(10) = 1.0e-3; pt_cni0(10) = 2.0e-3
  pt_rb0(10)   = 500.0;  pt_ni(10)  = 5.0e2

  print *, ''
  print *, '=== var_check / capacitance_gamma / vaporgrow chain ==='
  do i = 1, NPT
     temp  = pt_temp(i)
     pres  = pt_pres(i)
     qidum = pt_qi(i)
     dsdum = pt_ds0(i)
     ani   = pt_ani0(i)
     cni   = pt_cni0(i)
     rbdum = pt_rb0(i)
     nidum = pt_ni(i)
     aidum = ani**2 * cni * nidum
     cidum = cni**2 * ani * nidum

     ! Thermodynamic derived quantities (mirrors mp_jensen_ishmael lines
     ! 975-993 and 825-826; sui/sup use relh_ice/qvi to set a target
     ! ice-relative humidity, consistent with the Fortran's qv(k) input).
     svpi = polysvp(temp, 1)
     qvi  = 0.622*svpi/(pres-svpi)
     svpl = polysvp(temp, 0)
     qvs  = 0.622*svpl/(pres-svpl)
     qs0  = 0.622*polysvp(T0,0)/(pres-polysvp(T0,0))
     qv   = pt_relh_ice(i) * qvi
     if (temp .gt. T0) qvi = qvs
     sui  = qv/qvi - 1.0
     sup  = qv/qvs - 1.0
     xxls = 3.15e6 - 2370.*temp + 0.3337e6
     xxlv = 3.1484e6 - 2370.*temp
     xxlf = xxls - xxlv
     mu   = 1.496e-6*temp**1.5/(temp+120.)
     dv   = 8.794e-5*temp**1.81/pres
     kt   = 2.3823e-2 + 7.1177e-5*(temp-T0)
     nsch = mu/( (pres/(RD*temp)) )/dv
     npr  = mu/( (pres/(RD*temp)) )/kt
     rhodum = pres/(RD*temp)

     igr = get_igr(igrdata, temp)

     print *, ''
     print *, '--- point ', i, ': ', trim(pt_label(i))
     print *, 'inputs: temp, pres, qidum, dsdum0, ani0, cni0, rbdum0, nidum'
     write(*,'(8ES16.8)') temp, pres, qidum, dsdum, ani, cni, rbdum, nidum
     print *, 'derived: qv, qvi, qvs, sui, sup, igr, mu, dv, kt, nsch, npr, rhoair'
     write(*,'(12ES16.8)') qv, qvi, qvs, sui, sup, igr, mu, dv, kt, nsch, npr, rhodum

     call var_check(NU, ao, fourthirdspi, gammnu, qidum, dsdum, ani, cni, &
          rni, rbdum, nidum, aidum, cidum, alphstr, alphv, betam)

     print *, 'var_check outputs: dsdum, ani, cni, rni, rbdum, nidum, aidum, cidum, alphstr, alphv, betam'
     write(*,'(11ES16.8)') dsdum, ani, cni, rni, rbdum, nidum, aidum, cidum, alphstr, alphv, betam

     capgam = capacitance_gamma(ani, dsdum, NU, alphstr, i_gammnu)
     print *, 'capacitance_gamma output:'
     write(*,'(1ES16.8)') capgam

     ! vaporgrow chain (skip the T>T0 case's thermodynamic setup subtleties;
     ! vaporgrow itself branches on temp internally)
     dt = 2.0
     rimesum = 0.0
     iwci = nidum*rbdum*fourthirdspi*rni**3*(gamma(NU+2.+dsdum)/gammnu)
     vtbarb = 0.0; vtbarbm = 0.0; vtbarbz = 0.0
     fvdum = 1.0; fhdum = 1.0; dsdumout = dsdum

     call vaporgrow(dt, ani, cni, rni, igr, nidum, temp, rimesum, pres,   &
          NU, alphstr, sui, sup, qvs, qvi, mu, iwci, rhodum, qidum,       &
          dv, kt, ao, nsch, npr, gammnu, i_gammnu, fourthirdspi, svpi,    &
          xxls, xxlv, xxlf, capgam, vtbarb, vtbarbm, vtbarbz, anf, cnf,   &
          rnf, iwcf, fvdum, fhdum, rbdum, dsdum, rdout, dsdumout)

     print *, 'vaporgrow outputs: vtbarb, vtbarbm, vtbarbz, fvdum, fhdum, anf, cnf, rnf, iwcf, rdout, dsdumout, rbdum(inout)'
     write(*,'(12ES16.8)') vtbarb, vtbarbm, vtbarbz, fvdum, fhdum, anf, cnf, rnf, iwcf, rdout, dsdumout, rbdum
  end do

  !----------------------------------------------------------------------
  ! xm > 1e8 fall-speed fallback branch: bypass var_check (this point is
  ! intentionally NOT a var_check-consistent state -- ani=2mm exceeds the
  ! 1mm var_check cap -- it isolates the am/bm large-Best-number fallback
  ! inside vaporgrow's fall-speed block).
  !----------------------------------------------------------------------
  print *, ''
  print *, '=== xm > 1e8 fall-speed fallback branch (standalone, bypasses var_check) ==='
  temp = T0 - 40.0; pres = 101325.0
  dsdum = 1.0; ani = 2.0e-3; cni = ao**(1.-dsdum)*ani**dsdum
  rni = ani; rbdum = RHOI; nidum = 1.0e2; qidum = nidum*rbdum*fourthirdspi*rni**3*(gamma(NU+2.+dsdum)/gammnu)
  alphstr = ao**(1.-dsdum)
  svpi = polysvp(temp,1); qvi = 0.622*svpi/(pres-svpi)
  svpl = polysvp(temp,0); qvs = 0.622*svpl/(pres-svpl)
  sui = 0.05; sup = -0.05
  xxls = 3.15e6 - 2370.*temp + 0.3337e6
  xxlv = 3.1484e6 - 2370.*temp
  xxlf = xxls - xxlv
  mu = 1.496e-6*temp**1.5/(temp+120.)
  dv = 8.794e-5*temp**1.81/pres
  kt = 2.3823e-2 + 7.1177e-5*(temp-T0)
  rhodum = pres/(RD*temp)
  nsch = mu/rhodum/dv
  npr  = mu/rhodum/kt
  igr = get_igr(igrdata, temp)
  capgam = capacitance_gamma(ani, dsdum, NU, alphstr, i_gammnu)
  dt = 2.0; rimesum = 0.0
  iwci = nidum*rbdum*fourthirdspi*rni**3*(gamma(NU+2.+dsdum)/gammnu)
  vtbarb=0.; vtbarbm=0.; vtbarbz=0.; fvdum=1.; fhdum=1.; dsdumout=dsdum
  print *, 'inputs: temp, ani, dsdum, rbdum, igr'
  write(*,'(5ES16.8)') temp, ani, dsdum, rbdum, igr

  ! Diagnostic-only: recompute xm exactly as vaporgrow does internally
  ! (lines 3236-3264 of module_mp_jensen_ishmael.F) purely to confirm this
  ! state point actually lands in the xm>1e8 fallback branch (am=1.0865,
  ! bm=0.499, line 3278-3281). xm is not an argument/output of vaporgrow,
  ! so this block is NOT part of the routine under test -- it only proves
  ! the state point exercises the intended branch.
  block
    real :: phii_d, bl_d, al_d, aa_d, ba_d, qe_d, xn_d, bx_d, xm_d
    phii_d = cni/ani*gamma(NU-1.+dsdum)*i_gammnu
    if (phii_d .lt. 1.0) then
       bl_d=1.; al_d=2.; aa_d=PI; ba_d=2.; qe_d=(1.-phii_d)*(rbdum/RHOI)+phii_d
    else if (phii_d .gt. 1.0) then
       al_d=2.0; bl_d=1.; aa_d=PI*alphstr; ba_d=dsdum+1.; qe_d=1.0
    else
       bl_d=1.; al_d=2.; aa_d=PI; ba_d=2.; qe_d=1.0
    end if
    qe_d = min(qe_d, 1.0)
    xn_d = 2./rbdum*(rbdum-rhodum)*G_HOME*rhodum/mu**2 * (fourthirdspi*rbdum)*alphstr*al_d**2/(aa_d*qe_d)*qe_d**0.75
    bx_d = dsdum+2.+2.*bl_d-ba_d
    xm_d = xn_d*ani**bx_d*gamma(NU+bx_d)*i_gammnu
    print *, 'diagnostic xm (expect > 1e8 to confirm fallback branch):'
    write(*,'(1ES16.8)') xm_d
  end block

  call vaporgrow(dt, ani, cni, rni, igr, nidum, temp, rimesum, pres,   &
       NU, alphstr, sui, sup, qvs, qvi, mu, iwci, rhodum, qidum,       &
       dv, kt, ao, nsch, npr, gammnu, i_gammnu, fourthirdspi, svpi,    &
       xxls, xxlv, xxlf, capgam, vtbarb, vtbarbm, vtbarbz, anf, cnf,   &
       rnf, iwcf, fvdum, fhdum, rbdum, dsdum, rdout, dsdumout)
  print *, 'vaporgrow outputs: vtbarb, vtbarbm, vtbarbz'
  write(*,'(3ES16.8)') vtbarb, vtbarbm, vtbarbz

  !----------------------------------------------------------------------
  ! get_igr dedicated sweep: boundary behaviour of the piecewise table
  ! lookup (lines 3953-3982): dT in (-1,0], dT=-1 exactly, interior
  ! points, dT in [-60,-59), dT=-60 exactly, dT<-60 clamp, dT>0 (T>T0).
  !----------------------------------------------------------------------
  igr_temp(1)  = T0 - 0.3
  igr_temp(2)  = T0 - 1.0
  igr_temp(3)  = T0 - 1.5
  igr_temp(4)  = T0 - 6.0
  igr_temp(5)  = T0 - 12.0
  igr_temp(6)  = T0 - 20.0
  igr_temp(7)  = T0 - 35.0
  igr_temp(8)  = T0 - 59.0
  igr_temp(9)  = T0 - 59.5
  igr_temp(10) = T0 - 60.0
  igr_temp(11) = T0 - 65.0

  print *, ''
  print *, '=== get_igr boundary sweep ==='
  do i = 1, NIGR
     igr = get_igr(igrdata, igr_temp(i))
     write(*,'(A,I2,A,ES16.8,A,ES16.8)') 'igr_pt ', i, ' temp=', igr_temp(i), ' igr=', igr
  end do

  !----------------------------------------------------------------------
  ! access_lookup_table: 5 points into itab (index 1 and 2 alternating),
  ! 5 points into itabr (index cycling 1..6), interior grid nodes with
  ! fractional interpolation weights (not exactly on-node, to exercise
  ! all four interpolation dimensions).
  !----------------------------------------------------------------------
  lut_jj = (/ 5, 10, 20, 30, 45,  5, 10, 20, 30, 45 /)
  lut_ii = (/ 3,  5,  7,  9, 10,  2,  4,  6,  8,  9 /)
  lut_i  = (/ 5, 15, 25, 35, 45,  5, 15, 25, 35, 45 /)
  lut_k  = (/ 2,  4,  6,  8, 10,  2,  4,  6,  8, 10 /)
  lut_d1 = (/ 5.3,15.7,25.2,35.9,45.1,  5.5,15.1,25.8,35.3,45.6 /)
  lut_d2 = (/ 2.4, 4.6, 6.1, 8.8,10.2,  2.9, 4.2, 6.7, 8.1,10.9 /)
  lut_d4 = (/ 3.2, 5.9, 7.3, 9.1,10.4,  2.1, 4.8, 6.3, 8.6, 9.4 /)
  lut_d5 = (/ 5.1,10.6,20.3,30.8,45.4,  5.9,10.2,20.7,30.1,45.8 /)

  print *, ''
  print *, '=== access_lookup_table: itab (index 1, then 2) ==='
  do i = 1, 5
     call access_lookup_table(itab, lut_jj(i), lut_ii(i), lut_i(i), lut_k(i), &
          merge(1,2,mod(i,2)==1), lut_d1(i), lut_d2(i), lut_d4(i), lut_d5(i), proc)
     write(*,'(A,I2,A,I1,A,ES16.8)') 'itab_pt ', i, ' index=', merge(1,2,mod(i,2)==1), ' proc=', proc
  end do

  print *, ''
  print *, '=== access_lookup_table: itabr (index 1..6 cycling) ==='
  do i = 1, 5
     call access_lookup_table(itabr, lut_jj(i+5), lut_ii(i+5), lut_i(i+5), lut_k(i+5), &
          mod(i-1,6)+1, lut_d1(i+5), lut_d2(i+5), lut_d4(i+5), lut_d5(i+5), proc)
     write(*,'(A,I2,A,I1,A,ES16.8)') 'itabr_pt ', i, ' index=', mod(i-1,6)+1, ' proc=', proc
  end do

  print *, ''
  print *, '=== driver complete ==='

end program ref_driver
