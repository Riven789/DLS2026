#!/usr/bin/env bash
# Run every example in order. Each run.sh is self-resetting, so this is
# reproducible end to end. Expect a few minutes total (benchmarks dominate).
cd "$(dirname "$0")"
for d in 00-primitives 01-single-shot 02-bundled-verification \
         03-iterate-until-green 04-oracle-loop 05-profiler-loop \
         06-generator-critic 07-plan-fanout 08-reflection \
         09-tournament 10-capstone; do
  echo
  echo "================================================================"
  echo "== $d"
  echo "================================================================"
  if [ -f "$d/run.sh" ]; then bash "$d/run.sh"; else bash "$d/demo.sh"; fi
done
