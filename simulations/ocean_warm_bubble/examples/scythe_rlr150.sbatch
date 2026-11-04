#!/bin/bash
# The faithful RLR control of the 3-D ocean warm bubble (examples/rlr_150km.toml): a 150 km
# cylinder, the 30 km bubble on the axis, the box's spacing, f-plane at 15 N and physics
# (ISHMAEL ice + MYNN + two-moment rain, radiation off). Close-out Stream F, 2026-10-29.
# Compare it with the production box (simulations/ocean_warm_bubble_3d/scythe_owb3d.sbatch) by
#   julia --project=. simulations/ocean_warm_bubble/examples/rrr_rlr_compare.jl <box root> <this root> --strict-until 1200
#
#   sbatch simulations/ocean_warm_bubble/examples/scythe_rlr150.sbatch [--set key.path=value ...] [--resume]
#
# Job shape (default): FIVE 20-core nodes, one worker of 20 threads per node (--workers 5
# --slurm; 50 radial cells -> 10 per tile; the r tiling is verified against 1 worker to 1e-12
# only up to 2 workers, C1). At 3 km the cylinder has ~47,000 columns x 150 levels; at the Mac
# per-column rate of the 3 km pair (9.0e-3 thread-hours per column per simulated hour) one
# simulated hour is ~420 thread-hours: ~4 h on 100 threads before scaling losses, ~21 h on one
# node, which is why one node is no longer the default. Other shapes, one worker per node:
#   RLR_WORKERS=1 sbatch -N 1 simulations/ocean_warm_bubble/examples/scythe_rlr150.sbatch
#   RLR_WORKERS=10 sbatch -N 10 simulations/ocean_warm_bubble/examples/scythe_rlr150.sbatch
#
# `--set` is run control only (run.integration_time=7200 for a second hour; segment_length stays
# 3600 so seg_01, the hour the compare script reads, is the same), `--resume` continues a tree
# named with --set output.dir. Every fresh job writes its OWN tree,
# simulations/ocean_warm_bubble/output/rlr150_<jobid>, unless a --set output.dir is given.
# Model output is experimental data: move old trees aside, never delete them.
#
# Environment: SCYTHE (from SCYTHE_DIR, default $HOME/Development/Scythe.jl) is the Scythe
# checkout; SCYTHE_PROJECT is any Julia environment where Scythe is installed (default:
# $SCYTHE itself). The RLR geometry needs no Springsteel beyond development's Manifest.
#
#SBATCH -J scythe_rlr150
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH -c 20
#SBATCH -t 24:00:00
#SBATCH -o scythe_rlr150_%j.log

set -uo pipefail

export JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-20}
WORKERS=${RLR_WORKERS:-5}

SCYTHE=${SCYTHE_DIR:-$HOME/Development/Scythe.jl}
PROJECT=${SCYTHE_PROJECT:-$SCYTHE}
CASE=$SCYTHE/simulations/ocean_warm_bubble
JULIA=${JULIA_BIN:-julia}

SLURM_FLAG=()
[ "$WORKERS" -gt 1 ] && SLURM_FLAG=(--slurm)

# Default per-job tree for a fresh start; a user --set output.dir later in "$@" wins.
DEFAULT_SET=()
case " $* " in
  *" --resume "*) ;;
  *) DEFAULT_SET=(--set "output.dir=$CASE/output/rlr150_${SLURM_JOB_ID:-manual}") ;;
esac
RUN_ARGS=(--config "$CASE/examples/rlr_150km.toml" --workers "$WORKERS" ${SLURM_FLAG[@]+"${SLURM_FLAG[@]}"}
          --worker-threads "$JULIA_NUM_THREADS" ${DEFAULT_SET[@]+"${DEFAULT_SET[@]}"} "$@")

# The run root, resolved by Scythe itself from the SAME arguments run.jl gets, so
# job_stdout.log lands in the tree; also fails fast on a config that does not load.
ROOT_OUT=$("$JULIA" --project="$PROJECT" -e \
  'using Scythe; a = Scythe.parse_launcher_args(ARGS); println(load_config(a.config_path; set = a.set).output.dir)' \
  -- "${RUN_ARGS[@]}" 2>&1)
ROOT=$(echo "$ROOT_OUT" | tail -1)
case "$ROOT" in
  /*) ;;
  *) echo "$ROOT_OUT" >&2
     echo "could not resolve the run root from: ${RUN_ARGS[*]} - not starting the run." >&2
     exit 3 ;;
esac
mkdir -p "$ROOT"

# Redirect ALL stdout/stderr into the run tree from here on.
exec > >(tee -a "$ROOT/job_stdout.log") 2>&1

echo "=== provenance ====================================================="
echo "  date        : $(date)"
echo "  job         : ${SLURM_JOB_ID:-manual} on $(hostname)"
echo "  nodes       : ${SLURM_JOB_NODELIST:-unknown}"
echo "  git HEAD    : $(cd "$SCYTHE" && git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "  git dirty   : $(cd "$SCYTHE" && git status --porcelain 2>/dev/null | wc -l | tr -d ' ') file(s)"
echo "  JULIA_NUM_THREADS : $JULIA_NUM_THREADS   workers: $WORKERS"
echo "  run root    : $ROOT"
echo "  run.jl args : ${RUN_ARGS[*]}"
echo "====================================================================="
echo "  started     : $(date)"

"$JULIA" --project="$PROJECT" "$CASE/run.jl" "${RUN_ARGS[@]}"
status=$?

echo "=== finished: $(date) (exit $status) ==="
exit $status
