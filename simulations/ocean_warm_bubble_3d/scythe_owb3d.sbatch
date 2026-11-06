#!/bin/bash
# The 3-D ocean warm bubble production run (simulations/ocean_warm_bubble_3d/run.jl): the RRR
# box, 300 km x 300 km x 25 km, 100 x 100 x 50 cells (3 km; 90,000 columns x 150 levels), two
# one-hour segments. 11 workers, one per 20-core node (ClusterManagers.addprocs_slurm through
# run.jl --slurm), each holding one x tile of the box (100 cells -> 9-10 cells per tile) and
# threading its per-column loop across the node; the tiles exchange their 3-row spectral halos
# once per step. Close-out Stream C2, stage R2e; NOT for the Mac (rrr_120km.toml is the Mac
# size). Cost estimate from the Mac 3 km pair (14,400 columns, 20 min in 15,627 s on 10
# threads = 9.0e-3 thread-hours per column per simulated hour, the same per-column rate as the
# 6 km hour): 90,000 columns x 2 h = ~1,600 thread-hours, ~7.5 h on 220 threads before scaling
# losses; the 24 h limit below leaves a margin, and the second segment can be --resume'd.
#
#   sbatch simulations/ocean_warm_bubble_3d/scythe_owb3d.sbatch [--set key.path=value ...] [--resume]
#
# A run is ONE config file, $SCYTHE/simulations/ocean_warm_bubble_3d/scythe_config.toml;
# everything after the script name goes to run.jl unchanged (--set for run control,
# --resume). Every fresh job writes its OWN tree, output/owb3d_<jobid>, unless a --set
# output.dir is given; a resumed job must name its tree with --set output.dir. Model output
# is experimental data: move old trees aside, never delete them.
#
# Environment: SCYTHE (from SCYTHE_DIR, default $HOME/Development/Scythe.jl) is the Scythe
# checkout holding simulations/ocean_warm_bubble_3d/; SCYTHE_PROJECT is any Julia
# environment where Scythe is installed (default: $SCYTHE itself). Scythe's preflight
# (Springsteel version and its RRR tiling methods) runs inside run_simulation before
# anything is written.
#
# stdout/stderr are tee'd to <run root>/job_stdout.log, preceded by a provenance header;
# the run root also gets scythe_config.toml (resolved config + provenance) and segments.csv.
#
#SBATCH -J scythe_owb3d
#SBATCH -N 11
#SBATCH --ntasks-per-node=1
#SBATCH -c 20
#SBATCH -t 24:00:00
#SBATCH -o scythe_owb3d_%j.log

set -uo pipefail

# one worker per node (one x tile each) x 20 threads; the master runs on the first node
export JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-20}

SCYTHE=${SCYTHE_DIR:-$HOME/Development/Scythe.jl}
PROJECT=${SCYTHE_PROJECT:-$SCYTHE}
CASE=$SCYTHE/simulations/ocean_warm_bubble_3d
JULIA=${JULIA_BIN:-julia}

# Default per-job tree for a fresh start; a user --set output.dir later in "$@" wins.
DEFAULT_SET=()
case " $* " in
  *" --resume "*) ;;
  *) DEFAULT_SET=(--set "output.dir=$CASE/output/owb3d_${SLURM_JOB_ID:-manual}") ;;
esac
RUN_ARGS=(--config "$CASE/scythe_config.toml" --workers 11 --slurm
          --worker-threads "$JULIA_NUM_THREADS" ${DEFAULT_SET[@]+"${DEFAULT_SET[@]}"} "$@")

# The run root, resolved by Scythe itself from the SAME arguments run.jl gets (a relative
# --set output.dir resolves against the submit directory, exactly as run.jl will see it),
# so job_stdout.log lands in the tree. Also fails fast, in seconds, on a config that does
# not load. The full preflight (Springsteel version, RRTMGP lookup tables) runs inside
# run_simulation before anything is written.
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

# Redirect ALL stdout/stderr into the run tree from here on: `-o` above lands in the
# SUBMIT cwd, which is easy to lose track of once a job is resubmitted elsewhere.
exec > >(tee -a "$ROOT/job_stdout.log") 2>&1

echo "=== provenance ====================================================="
echo "  date        : $(date)"
echo "  job         : ${SLURM_JOB_ID:-manual} on $(hostname)"
echo "  nodes       : ${SLURM_JOB_NODELIST:-unknown}"
echo "  git HEAD    : $(cd "$SCYTHE" && git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "  git dirty   : $(cd "$SCYTHE" && git status --porcelain 2>/dev/null | wc -l | tr -d ' ') file(s)"
echo "  JULIA_NUM_THREADS : $JULIA_NUM_THREADS"
echo "  run root    : $ROOT"
echo "  run.jl args : ${RUN_ARGS[*]}"
echo "====================================================================="
echo "  started     : $(date)"

"$JULIA" --project="$PROJECT" "$CASE/run.jl" "${RUN_ARGS[@]}"
status=$?

echo "=== finished: $(date) (exit $status) ==="
exit $status
