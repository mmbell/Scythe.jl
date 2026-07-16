#!/bin/bash
# Single-node TC production run (job shape 1: lowest risk).
# All nest patches run as local Distributed workers on one 20-core node;
# threads handle the per-column loops inside each patch.
#
#   sbatch tc/scythe_tc.sbatch [integration_time_s]
#
# Defaults to the 5-day production run (432000 s). Restart chaining:
#   sbatch --export=ALL,RESTART_T=86400.0 tc/scythe_tc.sbatch 432000
#
#SBATCH -J scythe_tc
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 20
#SBATCH -t 72:00:00
#SBATCH -o scythe_tc_%j.log

# 3 workers (one per nest) x 6 threads + master on 20 cores
export JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-6}

SCYTHE=${SCYTHE_DIR:-$HOME/Development/Scythe.jl}
INTEGRATION_TIME=${1:-432000}
RESTART_ARGS=""
if [ -n "$RESTART_T" ]; then
    RESTART_ARGS="--restart $RESTART_T"
fi

julia --project=$SCYTHE $SCYTHE/tc/tc_run_axisym.jl $INTEGRATION_TIME $RESTART_ARGS
