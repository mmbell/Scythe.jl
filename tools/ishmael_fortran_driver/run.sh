#!/bin/bash
# run.sh -- build and run the ISHMAEL Fortran reference driver.
#
# Compiles the (public-list-patched) copy of module_mp_jensen_ishmael.F in
# this directory plus ref_driver.f90, at native single precision with
# big-endian binary I/O (matching how CM1 runs jensen_ishmael), then runs
# the driver from a scratch directory symlinked to the three CM1 .bin
# lookup tables (jensen_ishmael_init reads them from units 20/30/40 in the
# CWD).
set -euo pipefail

GFORTRAN=/opt/homebrew/bin/gfortran
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CM1_RUN=/Users/mmbell/Development/cm1r21.1/run
BUILD="$HERE/build"

mkdir -p "$BUILD"
cd "$BUILD"

# Symlink the three lookup-table binaries jensen_ishmael_init needs.
for f in ishmael-qi-qc.bin ishmael-qi-qr.bin ishmael-gamma-tab.bin; do
    ln -sf "$CM1_RUN/$f" "$BUILD/$f"
done

echo "== compiling stub module_wrf_error and stub input module =="
"$GFORTRAN" -c -ffree-form -ffree-line-length-none -fconvert=big-endian \
    "$HERE/stub_module_wrf_error.f90" -o stub_module_wrf_error.o
"$GFORTRAN" -c -ffree-form -ffree-line-length-none -fconvert=big-endian \
    "$HERE/stub_module_input.f90" -o stub_module_input.o

echo "== compiling module_mp_jensen_ishmael (public-list patched copy) =="
"$GFORTRAN" -c -ffree-form -ffree-line-length-none -fconvert=big-endian \
    "$HERE/module_mp_jensen_ishmael.F" -o module_mp_jensen_ishmael.o

echo "== compiling ref_driver =="
"$GFORTRAN" -ffree-line-length-none -fconvert=big-endian \
    stub_module_wrf_error.o stub_module_input.o module_mp_jensen_ishmael.o "$HERE/ref_driver.f90" \
    -o ref_driver

echo "== running ref_driver (from $BUILD, tables symlinked) =="
./ref_driver | tee "$HERE/ref_driver_output.txt"

echo ""
echo "Output written to $HERE/ref_driver_output.txt"
