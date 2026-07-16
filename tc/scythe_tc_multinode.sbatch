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
# patch actually uses ~18 threads (measure, don't assume).
#
#SBATCH -J scythe_tc
#SBATCH -N 3
#SBATCH --ntasks-per-node=1
#SBATCH -c 20
#SBATCH -t 72:00:00
#SBATCH -o scythe_tc_%j.log

export JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-18}
export SCYTHE_SLURM_WORKERS=3      # tc_run_axisym.jl switches to addprocs_slurm

SCYTHE=${SCYTHE_DIR:-$HOME/Development/Scythe.jl}
INTEGRATION_TIME=${1:-432000}
RESTART_ARGS=""
if [ -n "$RESTART_T" ]; then
    RESTART_ARGS="--restart $RESTART_T"
fi

julia --project=$SCYTHE $SCYTHE/tc/tc_run_axisym.jl $INTEGRATION_TIME $RESTART_ARGS
