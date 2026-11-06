#!/bin/bash
# Single-node TC production run (job shape 1: lowest risk). All three nest patches run as
# local Distributed workers on one 20-core node; threads handle the per-column loops
# inside each patch.
#
#   sbatch simulations/tc_axisym/scythe_tc.sbatch [--set key.path=value ...] [--resume]
#
# A run is ONE config file, $SCYTHE/simulations/tc_axisym/scythe_config.toml: an alternate
# grid (examples/fine.toml, examples/rlr.toml) is copied into it by hand, never layered on
# the command line. Everything after the script name goes to simulations/tc_axisym/run.jl
# unchanged: `--set` for run control (output.dir, run.integration_time) and --resume. The
# env knobs of the old tc/ wrappers are gone:
#
#   RESTART_T=<s> + SCYTHE_TC_OUTDIR   ->  --resume --set output.dir=<that run's root>
#   <integration_time> positional       ->  --set run.integration_time=<s>
#   SCYTHE_TC_RES=fine                  ->  copy examples/fine.toml into scythe_config.toml
#   CSV=1                               ->  output.formats = ["netcdf", "csv"] in the config
#   TRACE=<n>                           ->  diagnostics.state_minima_trace = <n> in the config
#   SCYTHE_TC_OUTDIR=<path>             ->  --set output.dir=<path>
#
# Environment: SCYTHE (from SCYTHE_DIR, default $HOME/Development/Scythe.jl) is the Scythe
# checkout -- or any run directory laid out as simulations/<case>/ -- holding run.jl and
# scythe_config.toml; SCYTHE_PROJECT is any Julia environment where Scythe is installed
# (default: $SCYTHE itself, i.e. the checkout's own project). Radiation (RRTMGP) is part of
# Scythe and needs no special environment. Scythe's preflight checks (Springsteel version,
# RRTMGP lookup tables when radiation is on) run inside run_simulation before anything is
# written; run tools/rrtmgp_prewarm.jl on a login node once before a radiating job.
#
# Every fresh job writes its OWN tree, simulations/tc_axisym/output/tc_<jobid>, unless a
# --set output.dir is given (a later --set wins over the default injected here). Model
# output is experimental data: move old trees aside, never delete them. A resumed job
# must name its tree with --set output.dir (the default per-job tree would be empty).
#
# stdout/stderr are tee'd to <run root>/job_stdout.log (in addition to the %j.log this
# SBATCH -o writes in the submit cwd), preceded by a provenance header; the run root also
# gets scythe_config.toml (resolved config + provenance) and segments.csv from the driver.
#
# Examples (from the repository root):
#   sbatch simulations/tc_axisym/scythe_tc.sbatch                                  # 5-day coarse
#   sbatch simulations/tc_axisym/scythe_tc.sbatch --set run.integration_time=86400      # 1-day run
#   sbatch simulations/tc_axisym/scythe_tc.sbatch --resume \
#          --set output.dir=$PWD/simulations/tc_axisym/output/tc_12345             # continue
#
#SBATCH -J scythe_tc
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 20
#SBATCH -t 72:00:00
#SBATCH -o scythe_tc_%j.log
#SBATCH -p all
# Need to set this for some Slurm configurations to explicitly use PMI2 for worker communication
export SLURM_MPI_TYPE=pmi2

set -uo pipefail

# 3 workers (one per nest) x 6 threads + master on 20 cores
export JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-6}

SCYTHE=${SCYTHE_DIR:-/bell-scratch/mmbell/scythe_2.0/Scythe.jl}
PROJECT=${SCYTHE_PROJECT:-$SCYTHE}
CASE=$SCYTHE/simulations/tc_axisym
JULIA=${JULIA_BIN:-julia}

# Default per-job tree for a fresh start; a user --set output.dir later in "$@" wins.
DEFAULT_SET=()
case " $* " in
  *" --resume "*) ;;
  *) DEFAULT_SET=(--set "output.dir=$CASE/output/tc_${SLURM_JOB_ID:-manual}") ;;
esac
RUN_ARGS=(--config "$CASE/scythe_config.toml" --workers-per-patch 1,1,1
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
