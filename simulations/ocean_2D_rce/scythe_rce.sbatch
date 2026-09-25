#!/bin/bash
# Single-node open-ocean RCE production run (simulations/ocean_2D_rce/run.jl): one 20-core
# node runs the segment ORCHESTRATOR (src/run_driver.jl run_simulation), which spawns one
# child `julia ... run.jl --segment k` per segment, SEQUENTIALLY, each warm-restarting from
# the previous segment's JLD2 checkpoint. Ported from rce/scythe_rce.sbatch onto the
# consolidated config/launcher.
#
#   sbatch simulations/ocean_2D_rce/scythe_rce.sbatch [-- EXTRA ARGS]
#
# Every physics/grid knob that used to be a SCYTHE_RCE_* environment variable is now a
# config key: override it with `--set key.path=value` (repeatable) or an overlay TOML file,
# both passed straight through to run.jl as "$@". The default job (no extra args) is
# exactly config.toml as committed: 20 days in 2-day segments, MYNN + ISHMAEL ice +
# diurnal all-sky RRTMGP.
#
# Examples (args after the script name are forwarded verbatim to run.jl):
#   sbatch simulations/ocean_2D_rce/scythe_rce.sbatch
#       # 20-day, quick-grid, default physics (config.toml as committed)
#   sbatch simulations/ocean_2D_rce/scythe_rce.sbatch \
#       --set run.integration_time=3456000 --set run.segment_length=345600
#       # 40-day run in 4-day segments (replaces the old positional DAYS SEG_DAYS)
#   sbatch simulations/ocean_2D_rce/scythe_rce.sbatch --set output.dir=output/ctrl1
#       # tagged output tree: simulations/ocean_2D_rce/output/ctrl1/ (replaces
#       # SCYTHE_RCE_TAG / SCYTHE_BENCH_TAG)
#   sbatch simulations/ocean_2D_rce/scythe_rce.sbatch --config output/ctrl1/config.toml --resume
#       # continue that same tree (replaces RESUME=1) -- point --config at the TREE'S OWN
#       # resolved config.toml (already an absolute output.dir), not at --set output.dir=...
#       # again: --resume refuses any --set besides run.integration_time, on principle that
#       # a resumed run keeps its recorded configuration.
#   sbatch simulations/ocean_2D_rce/scythe_rce.sbatch full.toml
#       # the full-resolution overlay (3 km / 300 m cells)
#
# stdout/stderr are tee'd to <root>/job_stdout.log (in addition to the %j.log this SBATCH
# -o writes in the submit cwd), preceded by a provenance header (date, hostname, git
# HEAD/dirty count, thread count, the full argument list) -- the run directory is then
# self-describing without cross-referencing the submit-dir log. Each SEGMENT's own
# provenance lives in its own seg_NN/config.toml and seg_NN/segment_stdout.log
# (src/run_driver.jl).
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
CASE_DIR="$SCYTHE/simulations/ocean_2D_rce"
JULIA=${JULIA_BIN:-julia}

# Mirror run.jl/config.toml's own output.dir resolution just well enough to know where to
# tee the log -- the AUTHORITATIVE resolution happens in Julia (src/config.jl
# load_config); this is provenance only. Two forms: a fresh/tagged start gives
# --set output.dir=...; a --resume of a tagged tree instead gives --config
# <root>/config.toml (see the RESUME example above) -- its dirname IS the root.
OUT_REL="output"
CONFIG_ARG=""
prev=""
for a in "$@"; do
    case "$a" in
        output.dir=*) OUT_REL="${a#output.dir=}" ;;
    esac
    [ "$prev" = "--config" ] && CONFIG_ARG="$a"
    prev="$a"
done
if [ -n "$CONFIG_ARG" ]; then
    case "$CONFIG_ARG" in
        /*) ROOT="$(dirname "$CONFIG_ARG")" ;;
        *) ROOT="$(dirname "$CASE_DIR/$CONFIG_ARG")" ;;
    esac
else
    case "$OUT_REL" in
        /*) ROOT="$OUT_REL" ;;
        *) ROOT="$CASE_DIR/$OUT_REL" ;;
    esac
fi
mkdir -p "$ROOT"

# Redirect ALL stdout/stderr into the run tree from here on.
exec > >(tee -a "$ROOT/job_stdout.log") 2>&1

echo "=== provenance ====================================================="
echo "  date        : $(date)"
echo "  hostname    : $(hostname)"
echo "  git HEAD    : $(cd "$SCYTHE" && git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "  git dirty   : $(cd "$SCYTHE" && git status --porcelain 2>/dev/null | wc -l | tr -d ' ') file(s)"
echo "  JULIA_NUM_THREADS : $JULIA_NUM_THREADS"
echo "  extra args  : $*"
echo "====================================================================="

echo "=== Scythe RCE run ==============================================="
echo "  job         : ${SLURM_JOB_ID:-manual} on $(hostname)"
echo "  threads     : $JULIA_NUM_THREADS"
echo "  output root : $ROOT"
echo "  started     : $(date)"
echo "==================================================================="

# --project=$SCYTHE/simulations: the launch environment that can load RRTMGP/ClimaComms,
# Scythe's WEAK deps (ext/ScytheRRTMGPExt.jl); this case runs diurnal RRTMGP by default.
# simulations/_preflight.jl: Springsteel-version + RRTMGP-artifact checks, reading this
# case's config so it checks exactly what this job will run.
"$JULIA" --project="$SCYTHE/simulations" "$SCYTHE/simulations/_preflight.jl" \
    "$CASE_DIR/config.toml" "$@"
status=$?
if [ $status -ne 0 ]; then
    echo "=== preflight FAILED: $(date) (exit $status) ==="
    exit $status
fi

"$JULIA" --project="$SCYTHE/simulations" "$SCYTHE/simulations/ocean_2D_rce/run.jl" \
    --workers 1 --worker-threads "$JULIA_NUM_THREADS" "$@"
status=$?

echo "=== finished: $(date) (exit $status) ==="
exit $status
