# Q23 QA and review record

- **Author:** Paradise
- **Survey source:** Q23 — "What is a harness, why does it matter, and how much of an agent failure comes from model vs instructions vs harness?" (MASTER-LIST L1.7 / L2.8)
- **Status:** draft complete; technical/evidence review (Ollie) and teaching/release review (Kukla) pending.

## Author self-check

- [x] Five baseline PDFs build locally (note: 5 pages; deck: 10 slides; discussion/projects/quiz: 1 page each).
- [x] Every `\cite` in `note.tex` resolves in `bibliography.bib` (bibtex: 0 warnings, 0 undefined); all 6 bib entries are cited.
- [x] Code executes with the repo venv, exit code 0, all assertions pass.
- [x] All citations verified against real sources (see below) before inclusion.
- [x] Video element skipped per module contract (parked).

## Citation verification (done before writing, 2026-10-10)

| Key | Claim | Verification |
|---|---|---|
| `yao2023react` | ReAct, ICLR 2023 | arXiv:2210.03629, authors Yao/Zhao/Yu/Du/Shafran/Narasimhan/Cao — confirmed |
| `shinn2023reflexion` | Reflexion, NeurIPS 2023 | arXiv:2303.11366 + NeurIPS proceedings page; camera-ready authors Shinn/Cassano/Gopinath/Narasimhan/Yao (5 authors, no Berman — arXiv v1 listing with Berman is outdated) — confirmed |
| `schluntz2024building` | Anthropic "Building Effective Agents" | anthropic.com/engineering/building-effective-agents, authors Erik Schluntz & Barry Zhang, published **Dec 19, 2024** — the task brief said "2025"; verified year is 2024, cited as 2024 |
| `kim2023language` | Language Models can Solve Computer Tasks | arXiv:2303.17491, Kim/Baldi/McAleer, NeurIPS 2023 proceedings — confirmed |
| `yang2024sweagent` | SWE-agent / agent-computer interface | arXiv:2405.15793, Yang/Jiménez/Wettig/Lieret/Yao/Narasimhan/Press — confirmed; ACI claim verified against abstract |
| `schick2023toolformer` | Toolformer | arXiv:2302.04761, NeurIPS 2023 — cited once in note (tool-schema paragraph) — confirmed |

Hedging check: the note's summary section explicitly hedges the decomposition ("a model of failure causation; real failures are often joint"). No unverified claims of numbers from memory; SWE-agent is described qualitatively ("dramatically / by a wide margin", "substantially") rather than with a recalled percentage.

## Code run (real output, repo venv)

Command: `/home/stevens/ai-course-2026/.venv/bin/python code/attribution_harness.py` — exit code **0** (re-verified after final edit).

```
Q23 attribution harness -- ablation grid over model / instructions / harness
====================================================================================
A: good model + good harness     | T1-report            | PASS         | checker satisfied
A: good model + good harness     | T2-celsius           | PASS         | checker satisfied
A: good model + good harness     | T3-multi-step        | PASS         | checker satisfied
B: good model + broken harness   | T1-report            | harness      | logged diff: injected != true; harness-swap control passes
B: good model + broken harness   | T2-celsius           | harness      | logged diff: injected != true; harness-swap control passes
B: good model + broken harness   | T3-multi-step        | harness      | logged diff: injected != true; harness-swap control passes
C: degraded model + good harness | T1-report            | model        | model-swap control passes (instructions + harness unchanged)
C: degraded model + good harness | T2-celsius           | model        | model-swap control passes (instructions + harness unchanged)
C: degraded model + good harness | T3-multi-step        | model        | model-swap control passes (instructions + harness unchanged)
D: good model, ambiguous spec    | T4-ambiguous-spec    | instructions | clarified-spec control passes (model + harness unchanged)
E: good model + timeout harness  | T1-report            | harness      | timeout killed a correct run; harness-swap control passes
====================================================================================

Failure attribution table (layer = judge verdict):
config                             pass  model  instr  harness  joint
A: good model + good harness          3      0      0      0      0
B: good model + broken harness        0      0      0      3      0
C: degraded model + good harness      0      3      0      0      0
D: good model, ambiguous spec         0      0      1      0      0
E: good model + timeout harness       0      0      0      1      0

All assertions passed: 11 runs, 8 non-PASS, each non-PASS attributed to exactly one layer (no joint cells).
A vs B isolates the harness layer (same model + instructions).
A vs C isolates the model layer (same instructions + harness).
A vs D isolates the instruction layer (same model + harness).
E shows a stopping-rule harness failure a transcript alone would
likely misread as 'the model kept failing'.
```

Deviation from the brief's code sketch: the brief asked for 3 configs; the shipped demo runs 5 (A–D from the brief plus E, a timeout-harness cell that exercises the "timeout kills a correct run" failure mode named in the topic). Two brief-named cells are realized exactly as specced (broken harness = corrupted tool output; degraded model). The judge implements the module's instrumentation in order: logged tool-I/O diffs first, then single-layer swap controls.

## Module contents

- `note.tex` / `note.pdf` — 5 pages: anatomy (table of 6 components), three-way attribution, interaction effects, 4 concrete failure modes, instrumentation (logging/replay/ablation).
- `deck.tex` / `deck.pdf` — 10 slides (title + 9 content frames; overlays removed to honor the 5–10 slide budget).
- `discussion.tex` / `discussion.pdf` — exactly 5 questions.
- `projects.tex` / `projects.pdf` — 4 suggestions.
- `quiz.tex` / `quiz.pdf` — 1 question + answer.
- `code/attribution_harness.py` — stdlib-only, no imports at all; deterministic (no RNG, no network).
- `bibliography.bib` — 6 entries, all verified, all cited.

## Reviews

- Technical/evidence review — Ollie: pending.
- Teaching/release review — Kukla: pending.
