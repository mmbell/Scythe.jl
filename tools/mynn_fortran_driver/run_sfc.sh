#!/bin/sh
# Build and run the GFDL v7 sea-surface roughness reference driver.
# `module_sf_exchcoef.f90` declares bare `real` arguments, so -fdefault-real-8 is what
# makes the printed values a DOUBLE-precision reference; -ffp-contract=off keeps gfortran
# from forming FMAs that Julia does not. Output is transcribed by hand into
# test/reference/gfdl_sfc_refs.jl -- there is no Fortran at test time.
set -e
cd "$(dirname "$0")"
FC=${FC:-/opt/homebrew/bin/gfortran}
FLAGS="-ffree-line-length-none -O0 -ffp-contract=off -fdefault-real-8"
mkdir -p build/sfc
( cd build/sfc && \
  $FC -c $FLAGS ../../module_sf_exchcoef.f90 && \
  $FC $FLAGS module_sf_exchcoef.o ../../sfc_ref_driver.f90 -o sfc_ref_driver && \
  ./sfc_ref_driver > ../../sfc_ref_driver_output.txt )
cat sfc_ref_driver_output.txt
