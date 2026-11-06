#!/bin/bash
# Single-node ocean_warm_bubble production run (simulations/ocean_warm_bubble/run.jl):
# 2 h at the TC coarse-grid spacing (6 km x 500 m), MYNN + ISHMAEL ice, radiation off,
# state_dependent_si -- the struct-default production physics, no case-specific
# exclusions. One process, one node; the case's own config has segment_length = 0 (a 2 h
# run needs no restart chaining), so this wrapper never spawns a segment child itself.
# Modeled on rce/scythe_rce.sbatch.
#
#   sbatch simulations/ocean_warm_bubble/scythe_owb.sbatch
#   sbatch simulations/ocean_warm_bubble/scythe_owb.sbatch --resume
#
# NO ENV-VAR KNOBS for physics: a run is ONE config file, scythe_config.toml -- a different
# arm (Louis BL, radiation on, ...) is an edited copy of it (see
# simulations/ocean_warm_bubble/README.md). This replaces the legacy driver's ~25
# `SCYTHE_OWB_*` variables. Extra arguments ("$@": --set for run control, --resume) are
# passed straight through to run.jl.
#
# Environment: SCYTHE (from SCYTHE_DIR, default $HOME/Development/Scythe.jl) is the Scythe
# checkout -- or any run directory laid out as simulations/<case>/ -- holding run.jl and
# scythe_config.toml; SCYTHE_PROJECT is any Julia environment where Scythe is installed
# (default: $SCYTHE itself, i.e. the checkout's own project). Radiation (RRTMGP) is part of
# Scythe and needs no special environment. Scythe's preflight checks (Springsteel version,
# RRTMGP lookup tables when radiation is on) run inside run_simulation before anything is
# written; run tools/rrtmgp_prewarm.jl on a login node once before a radiating job.
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
PROJECT=${SCYTHE_PROJECT:-$SCYTHE}
CONFIG=$SCYTHE/simulations/ocean_warm_bubble/scythe_config.toml
JULIA=${JULIA_BIN:-julia}

# job_stdout.log lives beside the run's own output: the case config's default
# output.dir = "output", relative to scythe_config.toml. NO env-var override here -- a
# `--set output.dir=...` moves where the RUN writes without moving this wrapper's own
# provenance log; look for job_stdout.log here by hand if so.
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

echo "=== Scythe ocean_warm_bubble run =================================="
echo "  job         : ${SLURM_JOB_ID:-manual} on $(hostname)"
echo "  threads     : $JULIA_NUM_THREADS"
echo "  output      : $OUTDIR"
echo "  extra args  : $*"
echo "  started     : $(date)"
echo "==================================================================="

"$JULIA" --project="$PROJECT" "$SCYTHE/simulations/ocean_warm_bubble/run.jl" \
    --config "$CONFIG" --workers 1 --worker-threads "$JULIA_NUM_THREADS" "$@"
status=$?

echo "=== finished: $(date) (exit $status) ==="
exit $status
