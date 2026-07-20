#!/bin/bash
# Node-per-patch TC production run (job shape 2: the distributed-memory
# intent of the multi-patch design). Each nest's worker lands on its own
# 20-core node via ClusterManagers.addprocs_slurm and threads the per-column
# loops across the node; the R3X/collar payloads cross nodes over
# RemoteChannels once per parent step.
#
#   sbatch tc/scythe_tc_multinode.sbatch [integration_time_s]
#
# Before committing a 5-day run, verify with a 1-h test job that (a) the
# cross-node payload exchange is amortized and (b) thread scaling inside a
# patch actually uses ~18 threads (measure, don't assume). At the "coarse"
# resolution each patch is only 15x25 cells, so the per-patch work may be too
# small to amortize the cross-node exchange -- COMPARE against the single-node
# script (scythe_tc.sbatch) before assuming this shape is faster.
#
# Every job writes to its OWN output tree (tc_<res>_<jobid>) so runs never
# clobber one another -- model output is experimental data; move old trees
# aside, never delete them.
#
# Knobs (pass with --export=ALL,VAR=value):
#   SCYTHE_TC_RES=coarse|fine   grid resolution (default coarse; see tc_params.jl)
#   RESTART_T=<seconds>         chain from a previous run's JLD2 checkpoints
#   TRACE=<n>                   min-rho_d trace every n steps (0 = off)
#   SCYTHE_TC_OUTDIR=<path>     override the output tree entirely
#
#SBATCH -J scythe_tc
#SBATCH -N 3
#SBATCH --ntasks-per-node=1
#SBATCH -c 20
#SBATCH -t 72:00:00
#SBATCH -o scythe_tc_%j.log

set -uo pipefail

export JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-18}
export SCYTHE_SLURM_WORKERS=3      # tc_run_axisym.jl switches to addprocs_slurm
export SCYTHE_TC_RES=${SCYTHE_TC_RES:-coarse}

SCYTHE=${SCYTHE_DIR:-$HOME/Development/Scythe.jl}
INTEGRATION_TIME=${1:-432000}

# Per-job output tree: never overwrite a previous run.
export SCYTHE_TC_OUTDIR=${SCYTHE_TC_OUTDIR:-$SCYTHE/tc/output/tc_${SCYTHE_TC_RES}_mn_${SLURM_JOB_ID:-manual}}
mkdir -p "$SCYTHE_TC_OUTDIR"

ARGS=""
[ -n "${RESTART_T:-}" ] && ARGS="$ARGS --restart $RESTART_T"
[ "${TRACE:-0}" != "0" ] && ARGS="$ARGS --trace ${TRACE}"

echo "=== Scythe TC run (node-per-patch) =============================="
echo "  job         : ${SLURM_JOB_ID:-manual}"
echo "  nodes       : ${SLURM_JOB_NODELIST:-unknown}"
echo "  resolution  : $SCYTHE_TC_RES"
echo "  integration : $INTEGRATION_TIME s"
echo "  threads     : $JULIA_NUM_THREADS per patch"
echo "  output      : $SCYTHE_TC_OUTDIR"
echo "  extra args  : ${ARGS:-none}"
echo "  started     : $(date)"
echo "================================================================="

julia --project=$SCYTHE $SCYTHE/tc/tc_run_axisym.jl $INTEGRATION_TIME $ARGS
status=$?

echo "=== finished: $(date) (exit $status) ==="
exit $status
