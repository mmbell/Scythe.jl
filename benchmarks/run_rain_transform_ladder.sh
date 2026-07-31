#!/usr/bin/env bash
# Validation ladder for the RAIN control-variable transform (options[:rain_transform]).
#
# Each arm writes to its own directory via SCYTHE_BENCH_TAG, so nothing has to be moved
# afterwards and every run's recorded output_dir is the one it actually used.
#
#   bash benchmarks/run_rain_transform_ladder.sh <arm> [<arm> ...]
#   bash benchmarks/run_rain_transform_ladder.sh all-quick
#
# Arms (see reference/FINDINGS_CONDENSATE_STAGE1.md for the cloud ladder this mirrors):
#   inert        quick O01, both options absent      -> the regression must still pass
#   c            quick O01, cloud transform only     -> reproduces the recorded bhyp arm
#   cr           quick O01, cloud + rain             -> THE measurement
#   r            quick O01, rain only                -> separates rain's effect from cloud's
#   mu8 mu6      cr with rain_mu 1e-8 / 1e-6         -> Ooyama's negligibility test
#   bf02q bf02f  bf02_moist quick/full, cloud+rain   -> the null control (no rain exists)
#   full-none full-c full-cr                         -> full resolution, where ringing is worse
#   nest-c nest-cr                                   -> the payoff: R3X, where the limiter cannot go
#
set -u
cd "$(dirname "$0")/.."

O01="julia --project=. benchmarks/o01_rainfall.jl --stage mc --grid rirk"
BF02="julia --project=. benchmarks/bf02_moist.jl --stage mc --grid rirk"

run_arm() {
  echo "================================================================"
  echo "ARM $1"
  echo "================================================================"
  shift
  ( eval "$@" )
  echo "--- exit $?"
}

for arm in "$@"; do
  case "$arm" in
    all-quick) bash "$0" inert c cr r ;;
    inert)  run_arm inert "SCYTHE_BENCH_TAG=_rl_inert $O01 --mode quick" ;;
    c)      run_arm c     "SCYTHE_BENCH_TAG=_rl_c SCYTHE_O01_CTRANS=bhyp $O01 --mode quick" ;;
    cr)     run_arm cr    "SCYTHE_BENCH_TAG=_rl_cr SCYTHE_O01_CTRANS=bhyp SCYTHE_O01_RTRANS=bhyp $O01 --mode quick" ;;
    r)      run_arm r     "SCYTHE_BENCH_TAG=_rl_r SCYTHE_O01_RTRANS=bhyp $O01 --mode quick" ;;
    # THE CONTROL for `cr`. A transformed species may not also be bounded, so `cr` differs
    # from `c` in TWO ways: rain is transformed AND rain's coefficient bound is gone. `c0` is
    # `c` with the bound dropped and nothing else, so `c0` -> `cr` isolates the transform.
    # Without it, anything read off `c` vs `cr` is confounded.
    c0)     run_arm c0    "SCYTHE_BENCH_TAG=_rl_c0 SCYTHE_O01_CTRANS=bhyp SCYTHE_O01_POSITIVITY=0 $O01 --mode quick" ;;
    # ... and its untransformed partner, for the same reason on the `inert` side.
    none0)  run_arm none0 "SCYTHE_BENCH_TAG=_rl_none0 SCYTHE_O01_POSITIVITY=0 $O01 --mode quick" ;;
    mu8)    run_arm mu8   "SCYTHE_BENCH_TAG=_rl_cr_mu8 SCYTHE_O01_CTRANS=bhyp SCYTHE_O01_RTRANS=bhyp SCYTHE_O01_RMU=1e-8 $O01 --mode quick" ;;
    mu6)    run_arm mu6   "SCYTHE_BENCH_TAG=_rl_cr_mu6 SCYTHE_O01_CTRANS=bhyp SCYTHE_O01_RTRANS=bhyp SCYTHE_O01_RMU=1e-6 $O01 --mode quick" ;;
    bf02q)  run_arm bf02q "SCYTHE_BENCH_TAG=_rl_cr SCYTHE_BF02_CTRANS=bhyp SCYTHE_BF02_RTRANS=bhyp $BF02 --mode quick" ;;
    bf02f)  run_arm bf02f "SCYTHE_BENCH_TAG=_rl_cr SCYTHE_BF02_CTRANS=bhyp SCYTHE_BF02_RTRANS=bhyp $BF02 --mode full" ;;
    full-none) run_arm full-none "SCYTHE_BENCH_TAG=_rl_none $O01 --mode full" ;;
    full-c)    run_arm full-c    "SCYTHE_BENCH_TAG=_rl_c SCYTHE_O01_CTRANS=bhyp $O01 --mode full" ;;
    full-cr)   run_arm full-cr   "SCYTHE_BENCH_TAG=_rl_cr SCYTHE_O01_CTRANS=bhyp SCYTHE_O01_RTRANS=bhyp $O01 --mode full" ;;
    # The full-resolution partner of `c0` — same reason: `full-cr` drops rain's bound, so
    # without this the full-resolution reading is confounded exactly as the quick one was.
    full-c0)   run_arm full-c0   "SCYTHE_BENCH_TAG=_rl_c0 SCYTHE_O01_CTRANS=bhyp SCYTHE_O01_POSITIVITY=0 $O01 --mode full" ;;
    nest-c)    run_arm nest-c    "SCYTHE_BENCH_TAG=_rl_c SCYTHE_O01_CTRANS=bhyp $O01 --mode quick --nests 3 --workers 5" ;;
    nest-cr)   run_arm nest-cr   "SCYTHE_BENCH_TAG=_rl_cr SCYTHE_O01_CTRANS=bhyp SCYTHE_O01_RTRANS=bhyp $O01 --mode quick --nests 3 --workers 5" ;;
    *) echo "unknown arm: $arm" ; exit 2 ;;
  esac
done
