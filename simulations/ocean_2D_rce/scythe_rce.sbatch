#!/bin/bash
# Single-node open-ocean RCE production run (simulations/ocean_2D_rce/run.jl): one 20-core
# node runs the segment ORCHESTRATOR (src/run_driver.jl run_simulation), which spawns one
# child julia process (Scythe.segment_main) per segment, SEQUENTIALLY, each warm-restarting from
# the previous segment's JLD2 checkpoint. Ported from rce/scythe_rce.sbatch onto the
# consolidated config/launcher.
#
#   sbatch simulations/ocean_2D_rce/scythe_rce.sbatch [--set key.path=value ...] [--resume]
#
# Every physics/grid knob that used to be a SCYTHE_RCE_* environment variable is now a
# config key in scythe_config.toml: a run is ONE config file, so a different physics or grid
# arm is an edited copy of it (examples/full.toml holds the full-resolution grid sections to
# copy in). `--set key.path=value` is for run control (output.dir, run.integration_time,
# --resume) and is passed straight through to run.jl as "$@". The default job (no extra
# args) is exactly scythe_config.toml as committed: 20 days in 2-day segments, MYNN +
# ISHMAEL ice + diurnal all-sky RRTMGP.
#
# Environment: SCYTHE (from SCYTHE_DIR, default $HOME/Development/Scythe.jl) is the Scythe
# checkout -- or any run directory laid out as simulations/<case>/ -- holding run.jl and
# scythe_config.toml; SCYTHE_PROJECT is any Julia environment where Scythe is installed
# (default: $SCYTHE itself, i.e. the checkout's own project). Radiation (RRTMGP) is part of
# Scythe and needs no special environment. Scythe's preflight checks (Springsteel version,
# RRTMGP lookup tables when radiation is on) run inside run_simulation before anything is
# written; run tools/rrtmgp_prewarm.jl on a login node once before a radiating job.
#
# Examples (args after the script name are forwarded verbatim to run.jl):
#   sbatch simulations/ocean_2D_rce/scythe_rce.sbatch
#       # 20-day, quick-grid, default physics (scythe_config.toml as committed)
#   sbatch simulations/ocean_2D_rce/scythe_rce.sbatch \
#       --set run.integration_time=3456000 --set run.segment_length=345600
#       # 40-day run in 4-day segments (replaces the old positional DAYS SEG_DAYS)
#   sbatch simulations/ocean_2D_rce/scythe_rce.sbatch --set output.dir=output/ctrl1
#       # tagged output tree: simulations/ocean_2D_rce/output/ctrl1/ (replaces
#       # SCYTHE_RCE_TAG / SCYTHE_BENCH_TAG)
#   sbatch simulations/ocean_2D_rce/scythe_rce.sbatch --resume --set output.dir=output/ctrl1
#       # continue that same tree (replaces RESUME=1): output.dir only LOCATES the tree; its
#       # own <root>/scythe_config.toml is authoritative, and --resume refuses any --set
#       # besides output.dir and run.integration_time (to extend the run).
#
# stdout/stderr are tee'd to <root>/job_stdout.log (in addition to the %j.log this SBATCH
# -o writes in the submit cwd), preceded by a provenance header (date, hostname, git
# HEAD/dirty count, thread count, the full argument list) -- the run directory is then
# self-describing without cross-referencing the submit-dir log. Each SEGMENT's own
# provenance lives in its own seg_NN/scythe_config.toml and seg_NN/segment_stdout.log
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
PROJECT=${SCYTHE_PROJECT:-$SCYTHE}
CASE_DIR="$SCYTHE/simulations/ocean_2D_rce"
JULIA=${JULIA_BIN:-julia}

# Mirror run.jl/scythe_config.toml's own output.dir resolution just well enough to know
# where to tee the log -- the AUTHORITATIVE resolution happens in Julia (src/config.jl
# load_config); this is provenance only. A --set output.dir=... names the root (relative:
# against the case directory, where this script runs run.jl from); a --config
# <root>/scythe_config.toml names it by its dirname.
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

# From the case directory, so a relative --set output.dir resolves where ROOT above says.
cd "$CASE_DIR" || exit 3
"$JULIA" --project="$PROJECT" "$SCYTHE/simulations/ocean_2D_rce/run.jl" \
    --workers 1 --worker-threads "$JULIA_NUM_THREADS" "$@"
status=$?

echo "=== finished: $(date) (exit $status) ==="
exit $status
