#!/bin/bash
# Single-node TC production run (job shape 1: lowest risk).
# All nest patches run as local Distributed workers on one 20-core node;
# threads handle the per-column loops inside each patch.
#
#   sbatch tc/scythe_tc.sbatch [integration_time_s]
#
# Defaults to the 5-day production run (432000 s). Every job writes to its OWN
# output tree (tc_<res>_<jobid>) so runs never clobber one another -- model
# output is experimental data; move old trees aside, never delete them.
#
# Knobs (pass with --export=ALL,VAR=value):
#   SCYTHE_TC_RES=coarse|fine   grid resolution (default coarse; see tc_params.jl)
#   RESTART_T=<seconds>         chain from a previous run's JLD2 checkpoints
#   TRACE=<n>                   min-rho_d trace every n steps (0 = off)
#   SCYTHE_TC_OUTDIR=<path>     override the output tree entirely
#
# Examples:
#   sbatch tc/scythe_tc.sbatch 432000                                # 5-day, coarse
#   sbatch --export=ALL,SCYTHE_TC_RES=fine tc/scythe_tc.sbatch 86400 # 1-day, fine
#   sbatch --export=ALL,RESTART_T=86400.0 tc/scythe_tc.sbatch 432000 # chain a restart
#
#SBATCH -J scythe_tc
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 20
#SBATCH -t 72:00:00
#SBATCH -o scythe_tc_%j.log

set -uo pipefail

# 3 workers (one per nest) x 6 threads + master on 20 cores
export JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-6}
export SCYTHE_TC_RES=${SCYTHE_TC_RES:-coarse}

SCYTHE=${SCYTHE_DIR:-$HOME/Development/Scythe.jl}
INTEGRATION_TIME=${1:-432000}

# Per-job output tree: never overwrite a previous run.
export SCYTHE_TC_OUTDIR=${SCYTHE_TC_OUTDIR:-$SCYTHE/tc/output/tc_${SCYTHE_TC_RES}_${SLURM_JOB_ID:-manual}}
mkdir -p "$SCYTHE_TC_OUTDIR"

ARGS=""
[ -n "${RESTART_T:-}" ] && ARGS="$ARGS --restart $RESTART_T"
[ "${TRACE:-0}" != "0" ] && ARGS="$ARGS --trace ${TRACE}"

echo "=== Scythe TC run ==============================================="
echo "  job         : ${SLURM_JOB_ID:-manual} on $(hostname)"
echo "  resolution  : $SCYTHE_TC_RES"
echo "  integration : $INTEGRATION_TIME s"
echo "  threads     : $JULIA_NUM_THREADS"
echo "  output      : $SCYTHE_TC_OUTDIR"
echo "  extra args  : ${ARGS:-none}"
echo "  started     : $(date)"
echo "================================================================="

julia --project=$SCYTHE $SCYTHE/tc/tc_run_axisym.jl $INTEGRATION_TIME $ARGS
status=$?

echo "=== finished: $(date) (exit $status) ==="
exit $status
