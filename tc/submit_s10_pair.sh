#!/bin/bash
# Submit (or just print) the S10 MYNN-EDMF definitive-pair cluster jobs: a 1 h fine
# calibration run first, then the three 24 h fine arms (louis, mynn, mynn+EDMF) on
# tc/scythe_tc_multinode.sbatch (node-per-patch), per tc/MYNN_RUNS.md "The definitive
# pair".
#
#   tc/submit_s10_pair.sh                # dry run (default): print the sbatch lines,
#                                         # submit nothing
#   tc/submit_s10_pair.sh --calibrate    # submit ONLY the 1 h fine calibration job
#   tc/submit_s10_pair.sh --pair         # submit ONLY the three 24 h arms
#   tc/submit_s10_pair.sh --calibrate --pair   # submit all four jobs
#
# WALL-TIME NOTE (tc/MYNN_RUNS.md "Runs made"): the COARSE 6 h pair took 40 min (louis)
# / 52 min (mynn) at JULIA_NUM_THREADS=4 on a laptop. FINE is ~11x the per-simulated-
# -hour cost of coarse (3/6/12 km nests vs the coarse grid, ts 0.25 vs coarse's larger
# step), so a naive scale-up puts a 24 h fine arm at roughly 30-38 h of wall clock at 4
# threads -- run the calibration job FIRST and measure its actual per-simulated-hour
# cost and thread scaling before trusting that number. At 18 threads per patch (this
# script's node-per-patch shape, one 20-core node per nest) expect roughly 9-15 h per
# 24 h arm. Running all three arms concurrently on separate node allocations is fine:
# the "never run more than one heavy job at once" rule is a LAPTOP shared-memory rule
# (Distributed workers + BLAS threads on the same machine), not a cluster one -- each
# arm here gets its own node allocation and does not contend with the others.
#
# `-t 72:00:00` (scythe_tc_multinode.sbatch's default) covers a single submission at
# either estimate; RESTART_INTERVAL (tc_params.jl, 21600 s = 6 h) checkpoints often
# enough that a dead or walltime-killed arm can be CHAINED with RESTART_T=<seconds> and
# the SAME SCYTHE_TC_OUTDIR (see tc/scythe_tc_multinode.sbatch's RESTART_T knob and
# tc/MYNN_RUNS.md) rather than restarted from scratch.
#
# `--export=ALL,...` is the form both tc/RADIATION_RUNS.md and tc/MYNN_RUNS.md have
# used for every cluster submission to date (ALL forwards the submitting shell's
# environment -- SCYTHE_DIR, JULIA_BIN, etc. -- in addition to the listed overrides).
# CSV=1 is scythe_tc_multinode.sbatch's own knob (added alongside its stdout/provenance
# logging) for appending --csv to the julia invocation; --csv is wanted here so the
# replay-harness/mynn_dump_columns.jl anchor has CSV to read even though the primary
# output is now the comprehensive NetCDF.

set -uo pipefail

SCYTHE=${SCYTHE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SBATCH="tc/scythe_tc_multinode.sbatch"

do_calibrate=0
do_pair=0
for a in "$@"; do
  case "$a" in
    --calibrate) do_calibrate=1 ;;
    --pair)      do_pair=1 ;;
    *) echo "Unknown argument: $a (use --calibrate and/or --pair)" >&2; exit 2 ;;
  esac
done
dry_run=1
[ "$do_calibrate" = "1" ] || [ "$do_pair" = "1" ] && dry_run=0

CAL_CMD=(sbatch --export=ALL,SCYTHE_TC_RES=fine,SCYTHE_TC_BL=mynn,CSV=1,SCYTHE_TC_OUTDIR="$SCYTHE/tc/output/tc_s10_fine_cal" "$SBATCH" 3600)

LOUIS_CMD=(sbatch --export=ALL,SCYTHE_TC_RES=fine,SCYTHE_TC_BL=louis,CSV=1,SCYTHE_TC_OUTDIR="$SCYTHE/tc/output/tc_s10_24h_louis" "$SBATCH" 86400)
MYNN_CMD=(sbatch --export=ALL,SCYTHE_TC_RES=fine,SCYTHE_TC_BL=mynn,SCYTHE_TC_MYNN_EDMF=0,CSV=1,SCYTHE_TC_OUTDIR="$SCYTHE/tc/output/tc_s10_24h_mynn" "$SBATCH" 86400)
EDMF_CMD=(sbatch --export=ALL,SCYTHE_TC_RES=fine,SCYTHE_TC_BL=mynn,SCYTHE_TC_MYNN_EDMF=1,CSV=1,SCYTHE_TC_OUTDIR="$SCYTHE/tc/output/tc_s10_24h_mynn_edmf" "$SBATCH" 86400)

print_cmd() { echo "$*"; }

echo "# 1 h fine MYNN calibration (measure per-simulated-hour wall time and thread"
echo "# scaling first)"
print_cmd "${CAL_CMD[@]}"
echo
echo "# the definitive pair (+ EDMF arm), 24 h, fine, --csv for the replay-harness anchor"
print_cmd "${LOUIS_CMD[@]}"
print_cmd "${MYNN_CMD[@]}"
print_cmd "${EDMF_CMD[@]}"

if [ "$dry_run" = "1" ]; then
  echo
  echo "# (dry run -- nothing submitted; pass --calibrate and/or --pair to submit)"
  exit 0
fi

if [ "$do_calibrate" = "1" ]; then
  echo
  echo "Submitting calibration job..."
  "${CAL_CMD[@]}"
fi

if [ "$do_pair" = "1" ]; then
  echo
  echo "Submitting the three 24 h arms..."
  "${LOUIS_CMD[@]}"
  "${MYNN_CMD[@]}"
  "${EDMF_CMD[@]}"
fi
