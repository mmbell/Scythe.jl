#!/bin/sh
# Build and run the MYNN-EDMF reference driver in both precision modes.
#   reference build (ref_driver_output_r8.txt): -fdefault-real-8 -fdefault-double-8 so the
#     module's bare `real` locals and default-real literals are double, and -ffp-contract=off
#     so no FMA is formed (Julia forms none) -- this is what the Julia port is checked against.
#   native build (ref_driver_output_native.txt): no promotion flags = UFS production semantics
#     (kind_phys = 8 but bare `real` locals/literals single); informational only.
set -e
cd "$(dirname "$0")"
FC=${FC:-/opt/homebrew/bin/gfortran}
COMMON="-ffree-line-length-none -O0 -ffp-contract=off"
build() {
    dir=$1; shift
    mkdir -p "$dir"
    ( cd "$dir" && \
      $FC -c $COMMON "$@" ../../stub_machine.f90 && \
      $FC -c $COMMON "$@" ../../bl_mynn_common.f90 && \
      $FC -c $COMMON "$@" ../../module_bl_mynn.F90 && \
      $FC $COMMON "$@" stub_machine.o bl_mynn_common.o module_bl_mynn.o ../../ref_driver.f90 -o ref_driver )
}
build build/r8 -fdefault-real-8 -fdefault-double-8
build build/native
( cd build/r8 && ln -sfn ../../columns columns && ./ref_driver > ../../ref_driver_output_r8.txt )
( cd build/native && ln -sfn ../../columns columns && ./ref_driver > ../../ref_driver_output_native.txt )
grep -h "GATE PASS\|GATE FAIL\|MISSING" ref_driver_output_r8.txt
echo "r8 lines:     $(wc -l < ref_driver_output_r8.txt)"
echo "native lines: $(wc -l < ref_driver_output_native.txt)"
