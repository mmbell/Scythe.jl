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
  real, parameter :: RHOW    = 1000.0
  real, parameter :: QSMALLM = 1.0e-12   ! module QSMALL (distinct from the itab
                                          ! ice-cloud gate's own 1.e-7 literal)
  real, parameter :: QNSMALLM= 1.25e-7   ! module QNSMALL

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
  ! Stage S6b: ice-cloud/ice-rain riming state (qc/nc feed itab, qr/nr feed
  ! itabr via ishmael_rain_lambda; picked to exercise both the "riming
  ! active" and "riming gated off" (qc<=1e-7) branches, and the T<=T0
  ! freeze-ri / T>T0 melt-ri branches -- see inline comments at each point).
  real, dimension(NPT) :: pt_qc, pt_nc, pt_qr, pt_nr
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

  ! Stage S6b: itab (ice-cloud riming) / itabr (ice-rain riming) index
  ! computation + lookup, replicated inline (this is NOT a subroutine in
  ! the Fortran -- it lives directly in mp_jensen_ishmael's main do-loop,
  ! lines 1053-1214).
  real    :: qc, nc, qr, nr, lamr, n0rr, rrr
  real    :: rrni, rqci, rdsi, rrho, rrri
  integer :: irni, iqci, idsi, irho, irri, iti
  real    :: proc_riming1, proc_riming2
  real    :: rimesum_pt, qi_qc_nrm_pt, qi_qc_nrd_pt
  real, dimension(6) :: procr
  real    :: numrateri_pt, rainrateri_pt, icerateri_pt
  real    :: dQRfzri_pt, dQIfzri_pt, dNfzri_pt, dQImltri_pt, dNmltri_pt
  real    :: rimesumr_pt, qi_qr_nrm_pt, qi_qr_nrd_pt, qi_qr_nrn_pt

  ! Stage S6b: wet_growth_check + riming growth (lines 1291-1507), also
  ! replicated inline (same reason -- no subroutine boundary in the
  ! Fortran except wet_growth_check itself).
  real    :: rimetotal_pt
  logical :: dry_growth_pt
  real    :: vi_pt, iwci_pt, rnfr_pt, vfr_pt
  real    :: gdenavg_pt, gdenavgr_pt, rimedr_pt, rimedrr_pt
  real    :: rimec1_pt, qcrimefrac_pt, gdentotal_pt, rhorimeout_blend_pt, rhorimeout_pt
  real    :: iwcfr_pt
  real    :: cnfr_pt, anfr_pt, phibr_pt, phifr_pt, gam_pt, gam_m1_pt
  real    :: prdr_pt, ardr_pt, crdr_pt

  ! Stage S6b: aggregation state points (categories: 1=planar[ICE1],
  ! 2=columnar[ICE2], 3=aggregates[ICE3], matching aggregation()'s own
  ! 3/4/5 category numbering).
  integer, parameter :: NAGG = 6
  real, dimension(NAGG) :: agg_temp, agg_rhoair
  real, dimension(NAGG) :: agg_q1, agg_n1, agg_d1, agg_q2, agg_n2, agg_d2, agg_q3, agg_n3, agg_d3
  real, dimension(NAGG) :: agg_rho1, agg_rho2, agg_phi1, agg_phi2
  character(len=48), dimension(NAGG) :: agg_label
  real :: agg_qagg1, agg_qagg2, agg_qagg3, agg_nagg1, agg_nagg2, agg_nagg3, agg_ddum3

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
  pt_qc(1)     = 2.0e-3; pt_nc(1)  = 2.0e8; pt_qr(1) = 1.0e-3; pt_nr(1) = 1.0e6

  pt_label(2)  = 'planar T=-12C moderate'
  pt_temp(2)   = T0 - 12.0; pt_pres(2) = 80000.0; pt_relh_ice(2) = 1.03
  pt_qi(2)     = 1.0e-5; pt_ds0(2) = 0.75; pt_ani0(2) = 1.5e-4; pt_cni0(2) = 5.0e-5
  pt_rb0(2)    = 300.0;  pt_ni(2)  = 2.0e4
  pt_qc(2)     = 1.0e-3; pt_nc(2)  = 1.0e8; pt_qr(2) = 5.0e-4; pt_nr(2) = 5.0e5

  pt_label(3)  = 'density-clamp-high (>RHOI)'
  pt_temp(3)   = T0 - 6.0;  pt_pres(3) = 85000.0; pt_relh_ice(3) = 1.02
  pt_qi(3)     = 5.0e-5; pt_ds0(3) = 1.0; pt_ani0(3) = 3.0e-5; pt_cni0(3) = 3.0e-5
  pt_rb0(3)    = 950.0;  pt_ni(3)  = 5.0e2
  ! qc <= 1.e-7: exercises the itab riming "gated off" branch (rimesum=0)
  pt_qc(3)     = 1.0e-8; pt_nc(3)  = 1.0e8; pt_qr(3) = 0.0; pt_nr(3) = 0.0

  pt_label(4)  = 'density-clamp-low (<50)'
  pt_temp(4)   = T0 - 20.0; pt_pres(4) = 70000.0; pt_relh_ice(4) = 1.10
  pt_qi(4)     = 1.0e-7; pt_ds0(4) = 0.9; pt_ani0(4) = 3.0e-4; pt_cni0(4) = 2.7e-4
  pt_rb0(4)    = 10.0;   pt_ni(4)  = 1.0e3
  pt_qc(4)     = 2.0e-3; pt_nc(4)  = 1.0e8; pt_qr(4) = 1.0e-4; pt_nr(4) = 1.0e5

  pt_label(5)  = 'homogeneous-freezing boundary T=-35C'
  pt_temp(5)   = T0 - 35.0; pt_pres(5) = 60000.0; pt_relh_ice(5) = 1.15
  pt_qi(5)     = 2.0e-6; pt_ds0(5) = 0.7; pt_ani0(5) = 2.0e-5; pt_cni0(5) = 1.0e-5
  pt_rb0(5)    = 500.0;  pt_ni(5)  = 5.0e5
  pt_qc(5)     = 5.0e-4; pt_nc(5)  = 1.0e8; pt_qr(5) = 2.0e-4; pt_nr(5) = 2.0e5

  pt_label(6)  = 'near 0C from below, T=-0.5C'
  pt_temp(6)   = T0 - 0.5;  pt_pres(6) = 95000.0; pt_relh_ice(6) = 1.005
  pt_qi(6)     = 5.0e-6; pt_ds0(6) = 1.0; pt_ani0(6) = 1.0e-4; pt_cni0(6) = 1.0e-4
  pt_rb0(6)    = 200.0;  pt_ni(6)  = 1.0e4
  ! Heavy riming near 0C: intended to probe the wet_growth_check boundary
  pt_qc(6)     = 3.0e-3; pt_nc(6)  = 5.0e7; pt_qr(6) = 3.0e-3; pt_nr(6) = 5.0e5

  pt_label(7)  = 'above 0C, T=+2C (melting/passthrough)'
  pt_temp(7)   = T0 + 2.0;  pt_pres(7) = 95000.0; pt_relh_ice(7) = 1.0
  pt_qi(7)     = 5.0e-6; pt_ds0(7) = 1.0; pt_ani0(7) = 1.0e-4; pt_cni0(7) = 1.0e-4
  pt_rb0(7)    = 200.0;  pt_ni(7)  = 1.0e4
  ! T > T0: exercises the ice-rain MELT transfer branch (dQImltri/dNmltri)
  pt_qc(7)     = 1.0e-3; pt_nc(7)  = 1.0e8; pt_qr(7) = 1.0e-3; pt_nr(7) = 1.0e6

  pt_label(8)  = 'tiny ice (small-ice-limit branch)'
  pt_temp(8)   = T0 - 10.0; pt_pres(8) = 85000.0; pt_relh_ice(8) = 1.02
  pt_qi(8)     = 1.0e-9; pt_ds0(8) = 1.0; pt_ani0(8) = 2.0e-6; pt_cni0(8) = 2.0e-6
  pt_rb0(8)    = 920.0;  pt_ni(8)  = 5.0e7
  pt_qc(8)     = 1.0e-3; pt_nc(8)  = 1.0e8; pt_qr(8) = 1.0e-4; pt_nr(8) = 1.0e5

  pt_label(9)  = 'large ice ani>=cni (large-ice-limit branch A)'
  pt_temp(9)   = T0 - 8.0;  pt_pres(9) = 85000.0; pt_relh_ice(9) = 1.05
  pt_qi(9)     = 5.0e-3; pt_ds0(9) = 0.7; pt_ani0(9) = 2.0e-3; pt_cni0(9) = 1.0e-3
  pt_rb0(9)    = 500.0;  pt_ni(9)  = 5.0e2
  ! qi > 0.1e-3 and qr > 0.1e-3: exercises the T<=T0 ice-rain FREEZE branch
  pt_qc(9)     = 1.0e-3; pt_nc(9)  = 1.0e8; pt_qr(9) = 5.0e-4; pt_nr(9) = 5.0e5

  pt_label(10) = 'large ice cni>ani (large-ice-limit branch B)'
  pt_temp(10)  = T0 - 8.0;  pt_pres(10) = 85000.0; pt_relh_ice(10) = 1.05
  pt_qi(10)    = 5.0e-3; pt_ds0(10) = 1.25; pt_ani0(10) = 1.0e-3; pt_cni0(10) = 2.0e-3
  pt_rb0(10)   = 500.0;  pt_ni(10)  = 5.0e2
  pt_qc(10)    = 1.0e-3; pt_nc(10)  = 1.0e8; pt_qr(10) = 5.0e-4; pt_nr(10) = 5.0e5

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

     !--------------------------------------------------------------------
     ! Stage S6b: ice-cloud riming (itab), lines 1053-1106. NOT a
     ! subroutine in the Fortran (lives inline in mp_jensen_ishmael's
     ! main do-loop) -- replicated here verbatim, using the SAME `nidum`
     ! variable this driver already threads through var_check/vaporgrow
     ! (see the Stage S6a driver's own `pt_ni ! nidum, #/m3` convention
     ! note above; item 1's Fortran literally multiplies by `ni(cc,k)`,
     ! so this driver reuses whatever single `nidum` value it already has
     ! in scope, exactly as it already does for the var_check/vaporgrow
     ! chain).
     !--------------------------------------------------------------------
     qc = pt_qc(i); nc = pt_nc(i)
     if (qc .gt. 1.0e-7) then
        rrni = 13.498*log10(0.5e6*rni)
        rqci = 8.776*log10(1.0e7*(exp(qc)-1.))
        rdsi = 50.*(dsdum - 0.5)
        rrho = 7.888*log10(0.02*rbdum)
        irni = int(rrni); iqci = int(rqci); idsi = int(rdsi); irho = int(rrho)
        rrni = max(rrni,1.); rqci = max(rqci,1.); rdsi = max(rdsi,1.); rrho = max(rrho,1.)
        irni = max(irni,1); iqci = max(iqci,1); idsi = max(idsi,1); irho = max(irho,1)
        rrni = min(rrni, real(size(itab,1))-1.)
        rqci = min(rqci, real(size(itab,2))-1.)
        rdsi = min(rdsi, real(size(itab,3))-1.)
        rrho = min(rrho, real(size(itab,4))-1.)
        irni = min(irni, size(itab,1)-1)
        iqci = min(iqci, size(itab,2)-1)
        idsi = min(idsi, size(itab,3)-1)
        irho = min(irho, size(itab,4)-1)
        call access_lookup_table(itab, irni, iqci, idsi, irho, 1, rdsi, rrho, rqci, rrni, proc_riming1)
        call access_lookup_table(itab, irni, iqci, idsi, irho, 2, rdsi, rrho, rqci, rrni, proc_riming2)
        rimesum_pt   = max(proc_riming1*nidum*nc*rhodum**2, 0.)
        qi_qc_nrm_pt = proc_riming1
        qi_qc_nrd_pt = proc_riming2
        if ((rimesum_pt/rhodum) .lt. QSMALLM) then
           rimesum_pt = 0.; qi_qc_nrm_pt = 0.; qi_qc_nrd_pt = 0.
        endif
     else
        rimesum_pt = 0.; qi_qc_nrm_pt = 0.; qi_qc_nrd_pt = 0.
     endif
     print *, 'itab riming: qc, nc, rimesum, qi_qc_nrm, qi_qc_nrd'
     write(*,'(5ES16.8)') qc, nc, rimesum_pt, qi_qc_nrm_pt, qi_qc_nrd_pt

     !--------------------------------------------------------------------
     ! Stage S6b: ice-rain riming/freeze/melt (itabr), lines 1111-1214.
     ! Also inline in the Fortran; also replicated here verbatim.
     !--------------------------------------------------------------------
     qr = pt_qr(i); nr = pt_nr(i)
     if (qr .gt. QSMALLM) then
        nr = max(nr, QNSMALLM)
        lamr = (PI*RHOW*nr/qr)**0.333333333
        n0rr = nr*lamr
        if (lamr .lt. LAMMINR) then
           lamr = LAMMINR; n0rr = lamr**4*qr/(PI*RHOW); nr = n0rr/lamr
        else if (lamr .gt. LAMMAXR) then
           lamr = LAMMAXR; n0rr = lamr**4*qr/(PI*RHOW); nr = n0rr/lamr
        endif
        rrr = 0.5*(1./lamr)

        rrni = 13.498*log10(0.5e6*rni)
        rrri = 23.273*log10(1.0e5*rrr)
        rdsi = 50.*(dsdum - 0.5)
        rrho = 7.888*log10(0.02*rbdum)
        irni = int(rrni); irri = int(rrri); idsi = int(rdsi); irho = int(rrho)
        rrni = max(rrni,1.); rrri = max(rrri,1.); rdsi = max(rdsi,1.); rrho = max(rrho,1.)
        irni = max(irni,1); irri = max(irri,1); idsi = max(idsi,1); irho = max(irho,1)
        rrni = min(rrni, real(size(itabr,1))-1.)
        rrri = min(rrri, real(size(itabr,2))-1.)
        rdsi = min(rdsi, real(size(itabr,3))-1.)
        rrho = min(rrho, real(size(itabr,4))-1.)
        irni = min(irni, size(itabr,1)-1)
        irri = min(irri, size(itabr,2)-1)
        idsi = min(idsi, size(itabr,3)-1)
        irho = min(irho, size(itabr,4)-1)
        do iti = 1, 6
           call access_lookup_table(itabr, irni, irri, idsi, irho, iti, rdsi, rrho, rrri, rrni, procr(iti))
        enddo

        numrateri_pt  = max(procr(4)*nidum*nr*rhodum, 0.)
        rainrateri_pt = max(procr(5)*nidum*nr*rhodum, 0.)
        icerateri_pt  = max(procr(6)*nidum*nr*rhodum, 0.)
        if (rainrateri_pt .lt. QSMALLM .or. icerateri_pt .lt. QSMALLM) then
           numrateri_pt = 0.; rainrateri_pt = 0.; icerateri_pt = 0.
        endif

        dQRfzri_pt = 0.; dQIfzri_pt = 0.; dNfzri_pt = 0.; dQImltri_pt = 0.; dNmltri_pt = 0.
        if (temp .le. T0) then
           if (qr .gt. 0.1e-3 .and. qidum .gt. 0.1e-3) then
              dQRfzri_pt = rainrateri_pt; dQIfzri_pt = icerateri_pt; dNfzri_pt = numrateri_pt
           endif
        else
           dQImltri_pt = icerateri_pt; dNmltri_pt = numrateri_pt
        endif

        rimesumr_pt  = max(procr(1)*nidum*nr*rhodum**2, 0.)
        qi_qr_nrm_pt = procr(1); qi_qr_nrd_pt = procr(2); qi_qr_nrn_pt = procr(3)
        if ((rimesumr_pt/rhodum) .lt. QSMALLM) then
           rimesumr_pt = 0.; qi_qr_nrm_pt = 0.; qi_qr_nrd_pt = 0.; qi_qr_nrn_pt = 0.
        endif
     else
        rimesumr_pt = 0.; qi_qr_nrm_pt = 0.; qi_qr_nrd_pt = 0.; qi_qr_nrn_pt = 0.
        numrateri_pt = 0.; rainrateri_pt = 0.; icerateri_pt = 0.
        dQRfzri_pt = 0.; dQIfzri_pt = 0.; dNfzri_pt = 0.; dQImltri_pt = 0.; dNmltri_pt = 0.
     endif
     print *, 'itabr riming: qr(in), nr(final), rimesumr, qi_qr_nrm, qi_qr_nrd, qi_qr_nrn'
     write(*,'(6ES16.8)') pt_qr(i), nr, rimesumr_pt, qi_qr_nrm_pt, qi_qr_nrd_pt, qi_qr_nrn_pt
     print *, 'itabr rates: numrateri, rainrateri, icerateri, dQRfzri, dQIfzri, dNfzri, dQImltri, dNmltri'
     write(*,'(8ES16.8)') numrateri_pt, rainrateri_pt, icerateri_pt, dQRfzri_pt, dQIfzri_pt, dNfzri_pt, dQImltri_pt, dNmltri_pt

     !--------------------------------------------------------------------
     ! Stage S6b: wet_growth_check (true subroutine, lines 3537-3556) then
     ! riming growth (lines 1291-1507, inline, replicated here).
     !--------------------------------------------------------------------
     rimetotal_pt = rimesum_pt + rimesumr_pt
     dry_growth_pt = .true.
     call wet_growth_check(NU, temp, rhodum, xxlv, xxlf, qv, dv, kt, qs0, fvdum, fhdum, &
          rimetotal_pt, rni, nidum, dry_growth_pt)
     print *, 'wet_growth_check: rimetotal, dry_growth(1=dry,0=wet)'
     write(*,'(ES16.8,I3)') rimetotal_pt, merge(1,0,dry_growth_pt)

     if (temp .gt. T0) dry_growth_pt = .false.

     vi_pt   = fourthirdspi*rni**3*(gamma(NU+2.+dsdum)/gammnu)
     iwci_pt = nidum*rbdum*vi_pt
     rnfr_pt = rni

     gdenavg_pt = 900.0
     if (qc .gt. QSMALLM .and. qi_qc_nrm_pt .gt. 0.) then
        rimec1_pt  = macklin_rimec1(temp)
        gdenavg_pt = macklin_density(rimec1_pt, qi_qc_nrd_pt, qi_qc_nrm_pt, temp, dry_growth_pt)
        rimedr_pt  = max(((qi_qc_nrm_pt/gdenavg_pt)/((gamma(NU+2.+dsdum)/gammnu)*4.*PI*rni**2)),0.)
        rnfr_pt    = rnfr_pt + max((rimedr_pt*nc*rhodum),0.)*dt
     endif

     gdenavgr_pt = 900.0
     if (qr .gt. QSMALLM .and. qi_qr_nrm_pt .gt. 0.) then
        rimec1_pt   = macklin_rimec1(temp)
        gdenavgr_pt = macklin_density(rimec1_pt, qi_qr_nrd_pt, qi_qr_nrm_pt, temp, dry_growth_pt)
        rimedrr_pt  = max(((qi_qr_nrm_pt/gdenavgr_pt)/((gamma(NU+2.+dsdum)/gammnu)*4.*PI*rni**2)),0.)
        rnfr_pt     = rnfr_pt + max((rimedrr_pt*nr*rhodum),0.)*dt
     endif

     vfr_pt  = fourthirdspi*rnfr_pt**3*(gamma(NU+2.+dsdum)/gammnu)
     vfr_pt  = max(vfr_pt, vi_pt)
     rnfr_pt = max(rnfr_pt, rni)

     if (rimetotal_pt .gt. 0. .and. vfr_pt .gt. vi_pt) then
        qcrimefrac_pt = max(0., min(1., rimesum_pt/rimetotal_pt))
        gdentotal_pt  = qcrimefrac_pt*gdenavg_pt + (1.-qcrimefrac_pt)*gdenavgr_pt
        rhorimeout_blend_pt = min(rbdum*(vi_pt/vfr_pt) + gdentotal_pt*(1.-vi_pt/vfr_pt), RHOI)
        iwcfr_pt   = rhorimeout_blend_pt * vfr_pt * nidum
        rhorimeout_pt = min(gdentotal_pt, RHOI)
     else
        iwcfr_pt = iwci_pt
        rnfr_pt  = rni
        rhorimeout_pt = rbdum
     endif

     if (dry_growth_pt) then
        cnfr_pt = cni; anfr_pt = ani; phibr_pt = 1.0
        if (rimetotal_pt .gt. 0. .and. vfr_pt .gt. vi_pt) then
           gam_pt   = gamma(NU+2.+dsdum)
           gam_m1_pt= gamma(NU-1.+dsdum)
           phibr_pt = cni/ani*gam_m1_pt/gammnu
           if (phibr_pt .eq. 1.) then
              phifr_pt = 1.; anfr_pt = rnfr_pt; cnfr_pt = rnfr_pt
           else if (phibr_pt .gt. 1.25) then
              phifr_pt = phibr_pt*(((rnfr_pt/rni)**3)**(-0.5))
              anfr_pt  = (ani/rni**1.5)*rnfr_pt**1.5
           else if (phibr_pt .lt. 0.8) then
              phifr_pt = phibr_pt*(rnfr_pt/rni)**3
              cnfr_pt  = phifr_pt*anfr_pt*gammnu/gam_m1_pt
           else
              phifr_pt = phibr_pt
              anfr_pt  = ((vfr_pt*gam_m1_pt)/(fourthirdspi*phifr_pt*gam_pt))**0.3333333333
              cnfr_pt  = phifr_pt*anfr_pt*gammnu/gam_m1_pt
           endif

           if (phibr_pt .le. 1. .and. phifr_pt .gt. 1.) then
              phifr_pt = 0.99
              anfr_pt  = ((vfr_pt*gam_m1_pt)/(fourthirdspi*phifr_pt*gam_pt))**0.3333333333
              cnfr_pt  = phifr_pt*anfr_pt*gammnu/gam_m1_pt
           endif
           if (phibr_pt .ge. 1. .and. phifr_pt .lt. 1.) then
              phifr_pt = 1.01
              anfr_pt  = ((vfr_pt*gam_m1_pt)/(fourthirdspi*phifr_pt*gam_pt))**0.3333333333
              cnfr_pt  = phifr_pt*anfr_pt*gammnu/gam_m1_pt
           endif
        endif

        cnfr_pt = max(cnfr_pt, cni)
        anfr_pt = max(anfr_pt, ani)

        prdr_pt = (iwcfr_pt - iwci_pt)/rhodum/dt
        ardr_pt = (2.*(anfr_pt-ani)*cni + (cnfr_pt-cni)*ani)*ani*nidum/dt
        crdr_pt = (2.*(cnfr_pt-cni)*ani + (anfr_pt-ani)*cni)*cni*nidum/dt

        prdr_pt = max(prdr_pt, 0.)
        ardr_pt = max(ardr_pt, 0.)
        crdr_pt = max(crdr_pt, 0.)
        if (prdr_pt .eq. 0.0) then
           ardr_pt = 0.; crdr_pt = 0.
        endif
     else
        prdr_pt = (iwcfr_pt - iwci_pt)/rhodum/dt
        ardr_pt = 0.; crdr_pt = 0.
     endif

     print *, 'riming growth: prdr, ardr, crdr, rhorimeout, gdenavg, gdenavgr'
     write(*,'(6ES16.8)') prdr_pt, ardr_pt, crdr_pt, rhorimeout_pt, gdenavg_pt, gdenavgr_pt
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

  !----------------------------------------------------------------------
  ! Stage S6b: aggregation (subroutine, lines 3988-4474, which itself
  ! calls col1, lines 4479-4541) -- 6 hand-picked 3-category state points
  ! (1=planar, 2=columnar, 3=aggregates, matching aggregation()'s own 3/
  ! 4/5 category numbering) covering: moderate riming-shape ice at
  ! T=-15C, an empty-aggregate-category start in the dendritic growth
  ! zone T=-14C (exercises col1's 1.4x DGZ efficiency boost, since
  ! mz=5 always for these live categories and -16<=Tc<=-12), warm
  ! near-0C quasi-spherical ice (rho>400 -> rhoeff<1 branch; phieffmax in
  ! (0.03,0.5) -> the Connolly et al. 2012 power-law phieff branch),
  ! extreme oblate/prolate shapes at T=-25C (phieffmax<=0.03 -> phieff=1
  ! branch), a large pre-existing aggregate population at T=-18C
  ! (exercises aggregate self-collection), and small ice at very cold
  ! T=-45C.
  !----------------------------------------------------------------------
  agg_label(1) = 'moderate planar+columnar, T=-15C'
  agg_temp(1) = T0-15.0; agg_rhoair(1) = 0.70
  agg_q1(1)=1.0e-3; agg_n1(1)=1.0e5; agg_d1(1)=200.0e-6
  agg_q2(1)=1.0e-3; agg_n2(1)=1.0e5; agg_d2(1)=200.0e-6
  agg_q3(1)=1.0e-4; agg_n3(1)=1.0e3; agg_d3(1)=500.0e-6
  agg_rho1(1)=300.0; agg_rho2(1)=300.0; agg_phi1(1)=0.3; agg_phi2(1)=3.0

  agg_label(2) = 'empty aggregates, dendritic growth zone T=-14C'
  agg_temp(2) = T0-14.0; agg_rhoair(2) = 0.70
  agg_q1(2)=5.0e-4; agg_n1(2)=5.0e4; agg_d1(2)=150.0e-6
  agg_q2(2)=5.0e-4; agg_n2(2)=5.0e4; agg_d2(2)=150.0e-6
  agg_q3(2)=0.0;    agg_n3(2)=0.0;   agg_d3(2)=1.0e-6
  agg_rho1(2)=250.0; agg_rho2(2)=250.0; agg_phi1(2)=0.4; agg_phi2(2)=2.5

  agg_label(3) = 'warm near 0C, quasi-spherical, T=-1C'
  agg_temp(3) = T0-1.0; agg_rhoair(3) = 1.00
  agg_q1(3)=1.0e-3; agg_n1(3)=2.0e4; agg_d1(3)=300.0e-6
  agg_q2(3)=1.0e-3; agg_n2(3)=2.0e4; agg_d2(3)=300.0e-6
  agg_q3(3)=5.0e-4; agg_n3(3)=5.0e2; agg_d3(3)=800.0e-6
  agg_rho1(3)=850.0; agg_rho2(3)=850.0; agg_phi1(3)=0.9; agg_phi2(3)=1.1

  agg_label(4) = 'extreme oblate/prolate, T=-25C'
  agg_temp(4) = T0-25.0; agg_rhoair(4) = 0.55
  agg_q1(4)=2.0e-3; agg_n1(4)=1.0e5; agg_d1(4)=250.0e-6
  agg_q2(4)=2.0e-3; agg_n2(4)=1.0e5; agg_d2(4)=250.0e-6
  agg_q3(4)=2.0e-4; agg_n3(4)=2.0e3; agg_d3(4)=600.0e-6
  agg_rho1(4)=100.0; agg_rho2(4)=100.0; agg_phi1(4)=0.05; agg_phi2(4)=20.0

  agg_label(5) = 'large existing aggregate population, T=-18C'
  agg_temp(5) = T0-18.0; agg_rhoair(5) = 0.75
  agg_q1(5)=1.0e-4; agg_n1(5)=1.0e3; agg_d1(5)=100.0e-6
  agg_q2(5)=1.0e-4; agg_n2(5)=1.0e3; agg_d2(5)=100.0e-6
  agg_q3(5)=2.0e-3; agg_n3(5)=5.0e3; agg_d3(5)=1.0e-3
  agg_rho1(5)=200.0; agg_rho2(5)=200.0; agg_phi1(5)=0.5; agg_phi2(5)=2.0

  agg_label(6) = 'very cold, small ice, T=-45C'
  agg_temp(6) = T0-45.0; agg_rhoair(6) = 0.40
  agg_q1(6)=5.0e-5; agg_n1(6)=1.0e4; agg_d1(6)=50.0e-6
  agg_q2(6)=5.0e-5; agg_n2(6)=1.0e4; agg_d2(6)=50.0e-6
  agg_q3(6)=1.0e-5; agg_n3(6)=1.0e2; agg_d3(6)=80.0e-6
  agg_rho1(6)=400.0; agg_rho2(6)=400.0; agg_phi1(6)=0.4; agg_phi2(6)=2.5

  print *, ''
  print *, '=== aggregation (+ col1) ==='
  do i = 1, NAGG
     agg_ddum3 = agg_d3(i)
     print *, ''
     print *, '--- agg point ', i, ': ', trim(agg_label(i))
     print *, 'inputs: temp, rhoair, q1,n1,d1, q2,n2,d2, q3,n3,d3, rho1,rho2,phi1,phi2'
     write(*,'(3ES16.8)') agg_temp(i), agg_rhoair(i)
     write(*,'(3ES16.8)') agg_q1(i), agg_n1(i), agg_d1(i)
     write(*,'(3ES16.8)') agg_q2(i), agg_n2(i), agg_d2(i)
     write(*,'(3ES16.8)') agg_q3(i), agg_n3(i), agg_d3(i)
     write(*,'(4ES16.8)') agg_rho1(i), agg_rho2(i), agg_phi1(i), agg_phi2(i)

     call aggregation(2.0, agg_rhoair(i), agg_temp(i), &
          agg_q1(i), agg_n1(i), agg_d1(i), agg_q2(i), agg_n2(i), agg_d2(i), &
          agg_q3(i), agg_n3(i), agg_ddum3, agg_rho1(i), agg_rho2(i), agg_phi1(i), agg_phi2(i), &
          coltab, coltabn, agg_qagg1, agg_qagg2, agg_qagg3, agg_nagg1, agg_nagg2, agg_nagg3)

     print *, 'aggregation outputs: qagg1, qagg2, qagg3, nagg1, nagg2, nagg3, ddum3(out)'
     write(*,'(7ES16.8)') agg_qagg1, agg_qagg2, agg_qagg3, agg_nagg1, agg_nagg2, agg_nagg3, agg_ddum3
  end do

  print *, ''
  print *, '=== driver complete ==='

contains

  ! Stage S6b: the Macklin (1962) rime-density temperature-dependence
  ! piece shared verbatim between the ice-cloud and ice-rain riming
  ! blocks (module_mp_jensen_ishmael.F lines 1312-1329 and 1357-1374 --
  ! identical formula, transcribed once here since the driver needs it
  ! twice per state point).
  real function macklin_rimec1(temp) result(rimec1)
    real, intent(in) :: temp
    real :: dum
    rimec1 = 0.0066
    if ((temp-T0) .lt. -30.) then
       rimec1 = 0.0036
    else if ((temp-T0) .lt. -20. .and. (temp-T0) .ge. -30.) then
       dum = (abs((temp-T0))-20.) / 10.
       rimec1 = dum*(0.0036) + (1.-dum)*(0.004)
    else if ((temp-T0) .lt. -15. .and. (temp-T0) .ge. -20.) then
       dum = (abs((temp-T0))-15.) / 5.
       rimec1 = dum*(0.004) + (1.-dum)*(0.005)
    else if ((temp-T0) .lt. -10. .and. (temp-T0) .ge. -15.) then
       dum = (abs((temp-T0))-10.) / 5.
       rimec1 = dum*(0.005) + (1.-dum)*(0.0066)
    else if ((temp-T0) .lt. -5. .and. (temp-T0) .ge. -10.) then
       dum = (abs((temp-T0))-5.) / 5.
       rimec1 = dum*(0.0066) + (1.-dum)*(0.012)
    else if ((temp-T0) .ge. -5.) then
       rimec1 = 0.012
    endif
  end function macklin_rimec1

  ! Stage S6b: the gdenavg/gdenavgr averaging + warm-branch/dry-growth
  ! override + [50,900] clamp shared verbatim between the ice-cloud and
  ! ice-rain riming blocks (lines 1330-1342 and 1375-1387).
  real function macklin_density(rimec1, nrd, nrm, temp, dry) result(gden)
    real, intent(in) :: rimec1, nrd, nrm, temp
    logical, intent(in) :: dry
    real :: dum
    gden = 1000.*(0.8*tanh(rimec1*nrd/nrm)+0.1)
    if ((temp-T0) .gt. -5. .and. (temp-T0) .le. 0.) then
       dum = (abs((temp-T0))-0.) / 5.
       gden = dum*gden + (1.-dum)*900.
    endif
    if ((temp-T0) .gt. 0. .or. .not. dry) then
       gden = 900.
    endif
    gden = max(gden, 50.)
    gden = min(gden, 900.)
  end function macklin_density

end program ref_driver
