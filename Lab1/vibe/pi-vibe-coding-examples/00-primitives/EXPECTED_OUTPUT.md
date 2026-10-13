# Expected output — Problem 0 — Primitives

Every `pi` invocation form from the cheat sheet, exercised once. The session
bookkeeping (`-c`, `-r`, `--fork`) is real mock state (`.pi_mock_state.json`);
the one-shot answers are canned.

## Captured run

Captured from one run of `run.sh` with the bundled mock `pi` in a 2-vCPU Linux container (Python 3.11, numpy 2.4.4, OpenBLAS). With the mock, the loop *structure* below is deterministic; wall-clock-derived numbers are not.

```text
$ pi one-shot: prints and exits
[pi] (mock) no scripted behavior matches this prompt — a real pi would act on it
[pi] prompt was: "one-shot: prints and exits"

$ pi -p print / headless mode — the workhorse for scripted loops
[pi] (mock) no scripted behavior matches this prompt — a real pi would act on it
[pi] prompt was: "print / headless mode — the workhorse for scripted loops"

$ pi --json -p structured output you can parse
{"session": "s-0003", "ok": true, "result": "mock pi: no scripted behavior for this prompt"}

$ pi -c
[pi] continuing session s-0003 (mock: nothing pending; see .pi_mock_state.json)

$ pi -r
[pi] sessions in this directory: s-0001, s-0002, s-0003

$ pi --fork 3f2a91
[pi] forked session 3f2a91 -> s-0004 (mock: forks share no state — cheap diversity)

$ pi -c
[pi] continuing session s-0004 (mock: nothing pending; see .pi_mock_state.json)

$ pi install npm:pi-loop-designer
[pi] installed npm:pi-loop-designer (mock no-op; the real pi fetches and registers the package)

$ pi          # (interactive REPL — mock just prints a note)
[pi] interactive REPL would open here — the mock is headless; use -p "..."
```

## What varies between runs/machines

- Session ids restart at `s-0001` because `demo.sh` resets state each run.

With a real `pi`: the one-shot and `-p` calls return model output, `--json`
returns the agent's structured result, and sessions live in your real `~/.pi`
store — so `-r` shows your actual history, and `--fork` gives you a branchable
copy (the substrate Problems 9-10 build on).
