#!/bin/bash
# Run-to-run reproducibility census: N identical, tagged o01_rainfall runs back to back,
# then a pairwise bitwise comparison of every snapshot against run 1.
#
#   bash benchmarks/repro_census.sh N TSTOP [extra o01_rainfall.jl args]
#
# Environment passes through (SCYTHE_O01_ICE=1, SCYTHE_O01_RAIN_MOMENTS=2, ...). TSTOP sets
# SCYTHE_O01_TSTOP (seconds; 120 exercises the dry/warm-phase path in ~5 min, 3600 is the
# production hour). LOAD=k starts k `yes` burners for the duration, to reproduce a loaded
# machine (the condition under which the historical non-reproductions were recorded).
# CENSUS_TAG overrides the tag stem (default _census<TSTOP>_).
#
# Exit 0 only if every compared file is identical. Run directories are never deleted here
# (project rule: move model output aside by hand, never rm).
set -u
N=${1:?N runs}; TSTOP=${2:?TSTOP seconds}; shift 2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || exit 1
STEM=${CENSUS_TAG:-_census${TSTOP}_}
LOGS=benchmarks/output/census_logs; mkdir -p "$LOGS"
MODE_ARGS=${*:---mode quick --stage mc --grid rirk}

pids=""
if [ "${LOAD:-0}" -gt 0 ]; then
  for c in $(seq 1 "$LOAD"); do yes > /dev/null & pids="$pids $!"; done
  echo "census: $LOAD burner(s):$pids"
fi
for i in $(seq 1 "$N"); do
  echo "census: run $i/$N start $(date '+%F %T')"
  SCYTHE_O01_TSTOP=$TSTOP SCYTHE_BENCH_TAG=${STEM}${i} \
    julia --project=. benchmarks/o01_rainfall.jl $MODE_ARGS > "$LOGS/${STEM#_}${i}.log" 2>&1
  echo "census: run $i/$N exit $? $(date '+%F %T')"
done
[ -n "$pids" ] && kill $pids 2>/dev/null

# The harness names the directory <case>_<mode>_<stage><grid><tag>; find run 1's by its tag.
A=$(ls -d benchmarks/output/o01_rainfall_*${STEM}1 2>/dev/null | head -1)
[ -d "$A" ] || { echo "census: cannot find run 1 output dir"; exit 2; }
status=0
echo "census: comparing against $A"
for i in $(seq 2 "$N"); do
  B=${A%1}$i
  for f in "$A"/*_physical.csv "$A"/*_spectral.csv "$A"/diagnostics.csv; do
    b="$B/$(basename "$f")"
    if cmp -s "$f" "$b"; then v=IDENTICAL; else v=DIFFERS; status=1; fi
    echo "run$i $(basename "$f") $v"
  done
done
echo "census: provenance (last $N records)"
tail -n "$N" benchmarks/results/o01_rainfall.jsonl | python3 -c '
import sys, json
for l in sys.stdin:
    r = json.loads(l)
    print(" ", r["timestamp"][:19], r["scythe_sha"], "dirty=%s" % r["scythe_dirty"],
          "untracked=%s" % r.get("scythe_untracked", "?"),
          "%sw x %st" % (r["nworkers"], r.get("worker_threads")),
          "blas=%s" % r.get("blas_threads", "?"), "cb=%s" % r.get("check_bounds", "?"),
          "tag=%s" % r.get("bench_tag", "?"), "wall=%s" % r["wallclock_s"])'
[ $status -eq 0 ] && echo "census: ALL IDENTICAL" || echo "census: DIFFERENCES FOUND"
exit $status
