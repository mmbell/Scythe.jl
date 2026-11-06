# ISHMAEL Fortran reference driver

Standalone Fortran program that prints, at full float precision, the
inputs and outputs of the CM1 `module_mp_jensen_ishmael.F` unit routines
that `src/ishmael_tables.jl` and `src/ishmael.jl` (Scythe.jl) port to
Julia: `var_check`, `capacitance_gamma`, `get_igr`, `access_lookup_table`
(both `itab` and `itabr` patterns), and `vaporgrow`. The transcribed
output lives at `test/reference/ishmael_fortran_refs.jl` and is what
`test/test_ishmael.jl` checks the Julia port against (rtol 1e-5) -- there
is no runtime Fortran dependency in the test suite.

## Files

- `module_mp_jensen_ishmael.F` -- a COPY of
  `/Users/mmbell/Development/cm1r21.1/src/module_mp_jensen_ishmael.F` with
  **exactly one substantive change**: the module's `public ::` statement is
  extended to also export `var_check`, `capacitance_gamma`, `get_igr`,
  `access_lookup_table`, `vaporgrow`, `polysvp`, `itab`, `itabr`,
  `igrdata` (originally only `jensen_ishmael_init` and `mp_jensen_ishmael`
  were public). Two mechanical follow-on edits were required for that
  change to compile: the `private` attribute was dropped from the
  `itab`/`itabr`/`igrdata` declarations (Fortran disallows both a
  `private` attribute AND an explicit `public ::` statement for the same
  entity), with `gamma_tab` left `private` (not needed by the driver). See
  the inline comments at each edit site for the exact diff.
- `ref_driver.f90` -- the driver program (option **(a)** from the task
  spec: edit a copy of the .F to expose the internal routines, rather than
  routing everything through `mp_jensen_ishmael` on a 1-column grid --
  cleaner for isolating unit references). 10 hand-picked state points are
  run through the `var_check -> capacitance_gamma -> vaporgrow` chain
  (mirroring how `mp_jensen_ishmael` itself calls them, lines 1044-1266),
  covering: columnar (igr>1, T=-6C) and planar (igr<1, T=-12C) habits,
  the density-clamp-high (>RHOI) and density-clamp-low (<50) `var_check`
  branches, the T=-35C homogeneous-freezing boundary, T just below and
  just above 0C (the `vaporgrow` T>T0 passthrough branch), a tiny-ice
  point that exercises `var_check`'s small-ice-limit (rni<2 micron)
  branch, and two large-ice points that exercise both large-ice-limit
  sub-branches (`ani>=cni` vs `cni>ani`). A separate standalone point
  (bypassing `var_check`, `ani`=2mm) isolates the `xm>1e8`
  Mitchell-Heymsfield fall-speed fallback branch (`am=1.0865`,
  `bm=0.499`) -- confirmed via a diagnostic-only recomputation of `xm`
  inside the driver (not part of the routine under test) that prints
  `xm=6.13366784e8`. `get_igr` additionally gets its own 11-point boundary
  sweep, and `access_lookup_table` gets 5 points into `itab` (index
  alternating 1,2) and 5 into `itabr` (index cycling 1..6).
- `stub_module_wrf_error.f90` -- an empty stand-in for `module_wrf_error`
  (which `module_mp_jensen_ishmael.F` `use`s but, per `grep`, never
  actually calls anything from).
- `stub_module_input.f90` -- a minimal stand-in for CM1's `input` module,
  providing just the handful of symbols (`timestats`, `mytime`,
  `time_microphy`, `time_dbz`, `ibr`/`ier`/`jbr`/`jer`/`kbr`/`ker`) that
  `mp_jensen_ishmael`'s declaration section references via `use input,
  only: ...`. `ref_driver.f90` never calls `mp_jensen_ishmael`, but the
  module still needs to compile as a whole.
- `run.sh` -- builds and runs everything.
- `ref_driver_output.txt` -- captured stdout from the last `run.sh`
  (checked in for the record; `test/reference/ishmael_fortran_refs.jl` is
  a hand-transcription of this file, not auto-generated from it).

## Build and run

```sh
tools/ishmael_fortran_driver/run.sh
```

This compiles (gfortran 15, `/opt/homebrew/bin/gfortran`) at **native
single precision** (gfortran's default `real`, matching how CM1 runs
`jensen_ishmael`) with `-fconvert=big-endian` (matching the
`convert='big_endian'` on the lookup-table reads in
`jensen_ishmael_init`), symlinks the three CM1 lookup-table binaries
(`ishmael-qi-qc.bin`, `ishmael-qi-qr.bin`, `ishmael-gamma-tab.bin`) from
`/Users/mmbell/Development/cm1r21.1/run/` into a scratch `build/`
directory (`jensen_ishmael_init` reads them from units 20/30/40 in the
CWD, hence the symlinks + explicit `cd`), and runs the driver from there.

Manual equivalent:

```sh
cd tools/ishmael_fortran_driver
mkdir -p build && cd build
ln -sf /Users/mmbell/Development/cm1r21.1/run/ishmael-qi-qc.bin .
ln -sf /Users/mmbell/Development/cm1r21.1/run/ishmael-qi-qr.bin .
ln -sf /Users/mmbell/Development/cm1r21.1/run/ishmael-gamma-tab.bin .
gfortran -c -ffree-form -ffree-line-length-none -fconvert=big-endian ../stub_module_wrf_error.f90
gfortran -c -ffree-form -ffree-line-length-none -fconvert=big-endian ../stub_module_input.f90
gfortran -c -ffree-form -ffree-line-length-none -fconvert=big-endian ../module_mp_jensen_ishmael.F
gfortran -ffree-line-length-none -fconvert=big-endian \
    stub_module_wrf_error.o stub_module_input.o module_mp_jensen_ishmael.o ../ref_driver.f90 -o ref_driver
./ref_driver
```

## Notes for anyone re-running this

- Re-running is only needed if the state-point selection changes; the
  transcribed values in `test/reference/ishmael_fortran_refs.jl` are
  static and don't need Fortran at test time.
- gfortran's `real` is 4-byte here (no `-fdefault-real-8`), matching CM1;
  Julia comparisons therefore use `rtol=1.0e-5`, not tighter.
