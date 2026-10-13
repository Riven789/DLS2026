# Vibe Coding with the Pi Agent Loop — runnable examples

A runnable companion to *"Vibe Coding with the Pi Agent Loop: Ten Challenge
Problems."* One directory per problem. Each `run.sh` contains the PDF's
cut/paste block **verbatim** (marked `BEGIN/END VERBATIM`), plus the fixture
files the problem presumes (provided tests, oracles, seeds, scorers) and an
`EXPECTED_OUTPUT.md` with a real captured run, annotated with what varies.

```
bin/pi                     deterministic MOCK of the pi CLI (see below)
bin/pytest                 shim: bare `pytest` -> python3 -m pytest
run_all.sh                 run everything in order (~3-5 min, benchmarks dominate)
00-primitives/             every pi invocation form, exercised once
01-single-shot/            spec -> artifact (sacct CSV -> GPU-hours table)
02-bundled-verification/   artifact ships with its own tests, run before "done"
03-iterate-until-green/    THE canonical loop: pytest exit code is the verdict
04-oracle-loop/            differential testing vs a slow-but-correct softmax
05-profiler-loop/          measurement as verification; target + plateau stops
06-generator-critic/       fresh-context hostile review gates the loop
07-plan-fanout/            plan -> parallel isolated workers -> integrate
08-reflection/             fix -> distill lesson to memory -> promote to skill
09-tournament/             best-of-8 vs one fitness function
10-capstone/               evolve an SpMV kernel: mutate/critique/bench/select/reflect
```

## Quickstart

Prerequisites: bash, `python3` with `numpy`, `scipy`, `pytest`
(`pip install numpy scipy pytest`). No network, no API key.

```
./run_all.sh                 # everything, in order
bash 03-iterate-until-green/run.sh    # or any single problem
```

Every `run.sh` is self-resetting (removes generated artifacts and mock state,
re-copies `seeds/`), so runs are reproducible and order-independent.

## The mock `pi` (and why it exists)

The real `pi` is an LLM agent — running it needs an API key and gives
different (better, worse, weirder) output every time. To make these examples
executable anywhere and the expected outputs meaningful, `bin/pi` is a
~700-line deterministic stand-in: it pattern-matches the tutorial's prompts
and writes the files a competent agent run would plausibly produce. **The
shell harnesses around it — the actual subject of the tutorial — are
untouched PDF code.**

Where the tutorial's pedagogy depends on failure, the mock scripts the
failure on purpose:

| Problem | Seeded first-attempt bug the loop must catch |
|---|---|
| 3 | stencil via `np.roll` — wraps at the boundary (fails 2 tests, fixed attempt 2) |
| 4 | float32 softmax, no max-shift — `exp` overflow -> NaN -> `MISMATCH` |
| 5 | broadcast-and-sum matmul (0.25 GFLOP/s) -> dgemm loop -> C-batched (HIT >28) |
| 6 | shared-accumulator race + dropped tail — 4 critic findings, clean on round 2 |
| 8 | Euler+float32, then RK4+float32 — two incidents, so a lesson RECURS and promotes to a skill |
| 9 | 3 of 8 candidates wrong (circular FFT, float32 tolerance, off-by-one) — score 0 |
| 10 | broken candidates every round, incl. a stale-result cache the benchmark would MISS but the fresh-eyes critic catches |

Mock state lives in `.pi_mock_state.json` per directory (attempt counters +
session ids). Problem 8's `~/.pi` writes are redirected into `./fakehome/` via
`PI_MOCK_HOME` so the demo never touches a real `~/.pi`.

## Running against the real `pi`

The real CLI is Mario Zechner's pi coding agent — https://pi.dev, repo
https://github.com/badlogic/pi-mono (earendil-works/pi):

```
npm install -g @mariozechner/pi-coding-agent
```

Then delete the `export PATH="$(cd .. && pwd)/bin:$PATH"` line from any
`run.sh` (and keep `bin/pytest` on PATH if bare `pytest` isn't installed).
Everything that makes the loops loops — the provided tests, oracles, scorers,
and stop rules — is exactly what transfers: `test_stencil.py`, `check.py`,
`bench.py`, `score.py`, `converge_check.py` are the durable artifacts, and
the agent's output now varies run to run, which is what Problems 9-10 exploit.
Two flags are conceptual in the PDF and mocked here: `--fork-fresh` (run the
critic in a forked/stripped session with the real CLI) and per-candidate
temperature/model variation in the tournaments.

## Fidelity notes (departures from the PDF, all marked in the run.sh files)

- PDF text extraction drops fi/fl ligatures ("rooine", "aky"); restored.
- Problem 4's heredoc lost its Python indentation in the PDF; restored.
- Problem 10's first code line ran the comment's trailing `---` into
  `seed=...`; split into two lines. The verbatim `$(cat gen/$round/$k ...)`
  cats a *directory* (injects nothing) and is kept as printed.
- Problem 5: `perf stat` is unavailable in containers/macOS — the command
  substitution injects its error text into the prompt, harmlessly; the loop's
  real signal is `bench.py`'s number.
- Problem 7: with no torch installed, the torch worker writes a flagged numpy
  fallback and the timing table says so.
- Problem 9's `score.py` uses n=2^18 (not 1e6) so the whole arena scores in
  seconds; the prompt text is unchanged.

## The three lines to leave them with

1. A loop without a machine-checkable verifier is just a chatbot.
2. Drive verdicts off exit codes and files, never off the model's prose.
3. Stop prompting one task at a time — design the loop, let the loop prompt
   the agent.
