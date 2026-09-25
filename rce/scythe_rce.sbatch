#!/bin/bash
# Single-node open-ocean RCE orchestrator run (benchmarks/ocean_rce.jl, no --segment):
# one 20-core node runs the ORCHESTRATOR, which spawns one child `julia ... --segment k`
# per segment, SEQUENTIALLY, each warm-restarting from the previous segment's JLD2
# checkpoint. Modeled on tc/scythe_tc.sbatch (single-node TC production job).
#
#   sbatch rce/scythe_rce.sbatch [DAYS] [SEG_DAYS]
#
# Defaults to a 20-day run (DAYS=20) split into 2-day segments (SEG_DAYS=2), i.e. 10
# children. Every job writes to its OWN output tree
# (benchmarks/output/ocean_rce_<mode>_mc_rirk<tag>) so runs never clobber one another --
# model output is experimental data; move old trees aside, never delete them.
#
# Knobs (pass with --export=ALL,VAR=value):
#   SCYTHE_RCE_*                every ocean_rce.jl physics/grid env knob (BL, MYNN_*, SFC*,
#                                RAD, ICE, SST, WIDTH, TS, NI, NK, OUTPUT, NOISE_*, WALL_BC,
#                                SIDECARS) -- see benchmarks/ocean_rce.jl's header. All are
#                                no-ops when unset, so the default job is the one described
#                                there: MYNN + :scythe fidelity + stability, ISHMAEL ice,
#                                diurnal all-sky RRTMGP.
#   SCYTHE_RCE_TAG=<s>           output-root suffix, mapped onto ocean_rce.jl's own
#                                SCYTHE_BENCH_TAG (so two jobs never write into the same
#                                tree without asking for it).
#   RESUME=1                     add --resume: skip segments segments.csv already marks
#                                "done", continuing an existing (not --resume'd-away) tree.
#                                REQUIRES SCYTHE_RCE_TAG (or the same DAYS/SEG_DAYS/--mode
#                                etc.) pointing at that same tree, since the root directory
#                                name is derived from the config, not stored anywhere else.
#   MODE=quick|full              --mode (default quick)
#   CSV=1                        append --csv to the julia invocation (CSV alongside the
#                                comprehensive NetCDF, for a replay harness)
#
# stdout/stderr are tee'd to $OUTDIR/job_stdout.log (in addition to the %j.log this SBATCH
# -o writes in the submit cwd), preceded by a provenance header (date, hostname, git
# HEAD/dirty count, every SCYTHE_RCE_* env var, thread count, the julia args) -- the run
# directory is then self-describing without cross-referencing the submit-dir log. Each
# SEGMENT's own scythe_out.log/scythe_err.log land under its own seg_NN/ (written by the
# model itself, src/Scythe.jl integrate_model).
#
# Examples:
#   sbatch rce/scythe_rce.sbatch                                   # 20-day, quick, default
#   sbatch rce/scythe_rce.sbatch 40 4                              # 40-day, 4-day segments
#   sbatch --export=ALL,SCYTHE_RCE_TAG=ctrl1 rce/scythe_rce.sbatch # tagged output tree
#   sbatch --export=ALL,RESUME=1,SCYTHE_RCE_TAG=ctrl1 rce/scythe_rce.sbatch
#                                                                   # continue that tree
#
#SBATCH -J scythe_rce
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 20
#SBATCH -t 72:00:00
#SBATCH -o scythe_rce_%j.log

set -uo pipefail

# 1 worker x 18 threads + master/OS on the remaining 2 cores.
export JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-18}

SCYTHE=${SCYTHE_DIR:-$HOME/Development/Scythe.jl}
DAYS=${1:-${DAYS:-20}}
SEG_DAYS=${2:-${SEG_DAYS:-2}}
MODE=${MODE:-quick}

# The output root is derived from --mode/--stage/--grid/SCYTHE_BENCH_TAG by ocean_rce.jl
# itself (benchmark_output_dir-style naming); mirror that here ONLY to point job_stdout.log
# and the provenance header at the same tree the julia process will use.
TAG_SUFFIX=""
[ -n "${SCYTHE_RCE_TAG:-}" ] && TAG_SUFFIX="_${SCYTHE_RCE_TAG}"
export SCYTHE_BENCH_TAG="${SCYTHE_BENCH_TAG:-}${TAG_SUFFIX}"
OUTDIR="$SCYTHE/benchmarks/output/ocean_rce_${MODE}_mc_rirk${SCYTHE_BENCH_TAG}"
mkdir -p "$OUTDIR"

# Redirect ALL stdout/stderr into the run tree from here on.
exec > >(tee -a "$OUTDIR/job_stdout.log") 2>&1

ARGS="--mode $MODE --stage mc --grid rirk --workers 1 --worker-threads $JULIA_NUM_THREADS"
ARGS="$ARGS --days $DAYS --segment-days $SEG_DAYS"
[ "${RESUME:-0}" != "0" ] && ARGS="$ARGS --resume"
[ "${CSV:-0}" != "0" ] && ARGS="$ARGS --csv"

JULIA=${JULIA_BIN:-julia}

# Shared preflight (tc/_preflight.sh): Springsteel-version + RRTMGP-artifact checks. The
# artifact check is gated on SCYTHE_TC_RAD or SCYTHE_RCE_RAD; the RCE default turns
# radiation on (ocean_rce.jl reads SCYTHE_RCE_RAD, default diurnal), so make that default
# explicit here so the gate sees the same value the driver will use.
export SCYTHE_RCE_RAD=${SCYTHE_RCE_RAD:-diurnal}
. "$SCYTHE/tc/_preflight.sh"
tc_preflight

echo "=== provenance ====================================================="
echo "  date        : $(date)"
echo "  hostname    : $(hostname)"
echo "  git HEAD    : $(cd "$SCYTHE" && git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "  git dirty   : $(cd "$SCYTHE" && git status --porcelain 2>/dev/null | wc -l | tr -d ' ') file(s)"
echo "  env SCYTHE_RCE_*:"
env | grep '^SCYTHE_RCE_' | sort | sed 's/^/    /'
echo "  SCYTHE_BENCH_TAG  : ${SCYTHE_BENCH_TAG:-<none>}"
echo "  JULIA_NUM_THREADS : $JULIA_NUM_THREADS"
echo "  julia args  : $ARGS"
echo "====================================================================="

echo "=== Scythe RCE run ==============================================="
echo "  job         : ${SLURM_JOB_ID:-manual} on $(hostname)"
echo "  days        : $DAYS  (segment length $SEG_DAYS d)"
echo "  threads     : $JULIA_NUM_THREADS"
echo "  output      : $OUTDIR"
echo "  extra args  : $ARGS"
echo "  started     : $(date)"
echo "==================================================================="

# --project=$SCYTHE/simulations: the launch environment that can load RRTMGP/ClimaComms,
# Scythe's WEAK deps (ext/ScytheRRTMGPExt.jl); the RCE runs diurnal RRTMGP by default.
$JULIA --project=$SCYTHE/simulations $SCYTHE/benchmarks/ocean_rce.jl $ARGS
status=$?

echo "=== finished: $(date) (exit $status) ==="
exit $status
