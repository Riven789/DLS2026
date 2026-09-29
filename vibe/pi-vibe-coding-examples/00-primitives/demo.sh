#!/usr/bin/env bash
# Problem 0 — Primitives: every pi invocation form you'll reuse.
# Runs against the bundled mock; swap in a real `pi` by removing bin/ from PATH.
cd "$(dirname "$0")"
export PATH="$(cd .. && pwd)/bin:$PATH"   # mock pi (delete this line for a real pi)
rm -f .pi_mock_state.json

run() { echo "\$ $*"; "$@"; echo; }

run pi "one-shot: prints and exits"
run pi -p "print / headless mode — the workhorse for scripted loops"
run pi --json -p "structured output you can parse"
run pi -c
run pi -r
run pi --fork 3f2a91
run pi -c
run pi install npm:pi-loop-designer
echo '$ pi          # (interactive REPL — mock just prints a note)'
pi
