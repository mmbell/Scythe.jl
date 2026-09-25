#!/bin/bash
# Single-node ocean_warm_bubble production run (simulations/ocean_warm_bubble/run.jl):
# 2 h at the TC coarse-grid spacing (6 km x 500 m), MYNN + ISHMAEL ice, radiation off,
# state_dependent_si -- the struct-default production physics, no case-specific
# exclusions. One process, one node; the case's own config has segment_length = 0 (a 2 h
# run needs no restart chaining), so this wrapper never spawns a segment child itself.
# Modeled on rce/scythe_rce.sbatch.
#
#   sbatch simulations/ocean_warm_bubble/scythe_owb.sbatch
#   sbatch simulations/ocean_warm_bubble/scythe_owb.sbatch --set physics.boundary_layer.scheme=louis
#   sbatch simulations/ocean_warm_bubble/scythe_owb.sbatch --set physics.radiation.scheme=rrtmgp
#   sbatch simulations/ocean_warm_bubble/scythe_owb.sbatch --resume
#
# NO ENV-VAR KNOBS: every option is a `--set key.path=value` (repeatable) or an overlay
# TOML file, passed straight through via "$@" to run.jl -- see
# simulations/ocean_warm_bubble/README.md. This replaces the legacy driver's ~25
# `SCYTHE_OWB_*` variables.
#
#SBATCH -J scythe_owb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 20
#SBATCH -t 4:00:00
#SBATCH -o scythe_owb_%j.log

set -uo pipefail

# 1 worker x 18 threads + master/OS on the remaining 2 cores.
export JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-18}

SCYTHE=${SCYTHE_DIR:-$HOME/Development/Scythe.jl}
CONFIG=$SCYTHE/simulations/ocean_warm_bubble/config.toml
JULIA=${JULIA_BIN:-julia}

# job_stdout.log lives beside the run's own output: the case config's default
# output.dir = "output", relative to config.toml. NO env-var override here (every option
# is --set/overlay, passed through "$@") -- a `--set output.dir=...` moves where the RUN
# writes without moving this wrapper's own provenance log; pass `--set output.dir=...`
# consistently and look for job_stdout.log there by hand if so.
OUTDIR=$SCYTHE/simulations/ocean_warm_bubble/output
mkdir -p "$OUTDIR"

# Redirect ALL stdout/stderr into the run tree from here on.
exec > >(tee -a "$OUTDIR/job_stdout.log") 2>&1

echo "=== provenance ====================================================="
echo "  date        : $(date)"
echo "  hostname    : $(hostname)"
echo "  git HEAD    : $(cd "$SCYTHE" && git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "  git dirty   : $(cd "$SCYTHE" && git status --porcelain 2>/dev/null | wc -l | tr -d ' ') file(s)"
echo "  JULIA_NUM_THREADS : $JULIA_NUM_THREADS"
echo "  config      : $CONFIG"
echo "  extra args  : $*"
echo "====================================================================="

# Preflight (simulations/_preflight.jl): Springsteel-version + RRTMGP-artifact checks,
# reading the RESOLVED config (config + "$@") so it checks exactly what this run needs --
# radiation off by default, so no RRTMGP artifact check unless "$@" turns it on.
"$JULIA" --project="$SCYTHE/simulations" "$SCYTHE/simulations/_preflight.jl" "$CONFIG" "$@"

echo "=== Scythe ocean_warm_bubble run =================================="
echo "  job         : ${SLURM_JOB_ID:-manual} on $(hostname)"
echo "  threads     : $JULIA_NUM_THREADS"
echo "  output      : $OUTDIR"
echo "  extra args  : $*"
echo "  started     : $(date)"
echo "==================================================================="

# --project=$SCYTHE/simulations: the workspace member that can load RRTMGP/ClimaComms
# (Scythe's WEAK deps, ext/ScytheRRTMGPExt.jl) -- only exercised if "$@" turns radiation
# on, but it is the standard launch environment for every case regardless.
"$JULIA" --project="$SCYTHE/simulations" "$SCYTHE/simulations/ocean_warm_bubble/run.jl" \
    --workers 1 --worker-threads "$JULIA_NUM_THREADS" "$@"
status=$?

echo "=== finished: $(date) (exit $status) ==="
exit $status
