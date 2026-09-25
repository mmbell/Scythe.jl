#!/bin/bash
# Single-node TC production run (job shape 1: lowest risk). All three nest patches run as
# local Distributed workers on one 20-core node; threads handle the per-column loops
# inside each patch.
#
#   sbatch simulations/tc_axisym/scythe_tc.sbatch [overlays...] [--set key.path=value ...]
#
# Everything after the script name goes to simulations/tc_axisym/run.jl unchanged, so the
# overlays (fine.toml, rlr.toml -- give their paths) and every --set work as they do
# locally; see simulations/tc_axisym/README.md for the arms. The env knobs of the old
# tc/ wrappers are gone:
#
#   RESTART_T=<s> + SCYTHE_TC_OUTDIR   ->  --resume --set output.dir=<that run's root>
#   <integration_time> positional       ->  --set run.integration_time=<s>
#   SCYTHE_TC_RES=fine                  ->  $SCYTHE/simulations/tc_axisym/fine.toml
#   CSV=1                               ->  --set 'output.formats=["netcdf","csv"]'
#   TRACE=<n>                           ->  --set diagnostics.state_minima_trace=<n>
#   SCYTHE_TC_OUTDIR=<path>             ->  --set output.dir=<path>
#
# Every fresh job writes its OWN tree, simulations/tc_axisym/output/tc_<jobid>, unless a
# --set output.dir is given (a later --set wins over the default injected here). Model
# output is experimental data: move old trees aside, never delete them. A resumed job
# must name its tree with --set output.dir (the default per-job tree would be empty).
#
# stdout/stderr are tee'd to <run root>/job_stdout.log (in addition to the %j.log this
# SBATCH -o writes in the submit cwd), preceded by a provenance header; the run root also
# gets config.toml (resolved config + provenance) and segments.csv from the driver.
#
# Examples (from the repository root):
#   sbatch simulations/tc_axisym/scythe_tc.sbatch                                  # 5-day coarse
#   sbatch simulations/tc_axisym/scythe_tc.sbatch simulations/tc_axisym/fine.toml \
#          --set run.integration_time=86400                                        # 1-day fine
#   sbatch simulations/tc_axisym/scythe_tc.sbatch --resume \
#          --set output.dir=$PWD/simulations/tc_axisym/output/tc_12345             # continue
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

SCYTHE=${SCYTHE_DIR:-$HOME/Development/Scythe.jl}
CASE=$SCYTHE/simulations/tc_axisym
JULIA=${JULIA_BIN:-julia}

# Default per-job tree for a fresh start; a user --set output.dir later in "$@" wins.
DEFAULT_SET=()
case " $* " in
  *" --resume "*) ;;
  *) DEFAULT_SET=(--set "output.dir=$CASE/output/tc_${SLURM_JOB_ID:-manual}") ;;
esac
RUN_ARGS=(--config "$CASE/config.toml" --workers-per-patch 1,1,1
          --worker-threads "$JULIA_NUM_THREADS" ${DEFAULT_SET[@]+"${DEFAULT_SET[@]}"} "$@")

# Preflight (Springsteel version, config, RRTMGP artifacts) with the SAME arguments run.jl
# gets; it also resolves the run root. Its output is replayed into job_stdout.log below.
PRE_OUT=$("$JULIA" --project="$SCYTHE/simulations" "$SCYTHE/simulations/_preflight.jl" \
          "${RUN_ARGS[@]}" 2>&1)
pre_status=$?
if [ $pre_status -ne 0 ]; then
  echo "$PRE_OUT" >&2
  echo "PREFLIGHT FAILED (exit $pre_status) - not starting the run." >&2
  exit 3
fi
ROOT=$(echo "$PRE_OUT" | sed -n 's/^run root: //p' | tail -1)
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
echo "=== preflight ======================================================"
echo "$PRE_OUT"
echo "====================================================================="
echo "  started     : $(date)"

"$JULIA" --project="$SCYTHE/simulations" "$CASE/run.jl" "${RUN_ARGS[@]}"
status=$?

echo "=== finished: $(date) (exit $status) ==="
exit $status
