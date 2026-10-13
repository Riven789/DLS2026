# Q22 QA and review record

- **Author:** Paradise
- **Survey source:** responses #23 and #35 (per MASTER-QUESTIONS.tex: "What separates an LLM from an agent? Is Claude Code an agent or a harness?"); harness-engineering strand of student_survey_synthesis_anonymous.pdf (#10, #22, #30, #34).
- **Status:** author draft; awaiting Ollie (technical/evidence) and Kukla (teaching/release) reviews.
- **Video element:** parked repo-wide; skipped per module contract.

## Author self-check

- [x] Five baseline PDFs build locally with pdflatex + bibtex (note: 5 pages; deck: 10 pages/slides incl. title; discussion/projects/quiz: 1 page each).
- [x] Every `\cite` in `note.tex` resolves: 10 keys, 10 `bibitem`s in `note.bbl`, bibtex reports zero warnings.
- [x] Code runs with repo `.venv` python (3.14.7), stdlib only, exits 0; real output pasted below.
- [x] Discussion contains exactly 5 questions; projects has 4 suggestions; quiz has 1 question + answer.
- [x] All citations verified against primary sources before inclusion (see ledger below).
- [ ] Ollie technical/evidence review.
- [ ] Kukla teaching/release review.

## Citation verification ledger

All four required citations plus six supporting ones were checked against arXiv / publisher / vendor pages on 2026-10-10 before writing `bibliography.bib`:

| Key | Claim checked | Status |
|---|---|---|
| `yao2023react` | ReAct, ICLR 2023 (oral), arXiv:2210.03629, authors Yao/Zhao/Yu/Du/Shafran/Narasimhan/Cao | VERIFIED (arXiv + ICLR virtual site) |
| `shinn2023reflexion` | Reflexion, NeurIPS 2023, arXiv:2303.11366, authors Shinn/Cassano/Berman/Gopinath/Narasimhan/Yao (v4 author list) | VERIFIED (arXiv) |
| `wang2024survey` | "A Survey on LLM-based Autonomous Agents", Frontiers of Computer Science 18(6):186345, 2024, arXiv:2308.11432, 13 authors Wang…Wen | VERIFIED (arXiv + DOI 10.1007/s11704-024-40231-1 via secondary confirmation) |
| `anthropic2024agents` | "Building Effective Agents" exists at anthropic.com/engineering/building-effective-agents; workflows-vs-agents distinction quoted in note | VERIFIED (fetched full page). **Date note:** the post is from **December 2024**, not 2025 as the task brief stated; `bibliography.bib` records the real date. A 2026 banner notes the tooling landscape has changed and points to a newer "Claude Managed Agents" post. |
| `anthropic2025claudecode` | "Claude Code: Best Practices for Agentic Coding", Apr 18, 2025 | VERIFIED (Anthropic engineering index + multiple mirrors) |
| `russell2021aima` | AIMA 4th ed., Russell & Norvig, Pearson 2021; agent = percept→action mapping | VERIFIED (Pearson catalog + LC record) |
| `schick2023toolformer` | Toolformer, NeurIPS 2023 (oral), arXiv:2302.04761 | VERIFIED (arXiv + NeurIPS proceedings) |
| `yang2024sweagent` | SWE-agent, NeurIPS 2024, arXiv:2405.15793; 12.47% full SWE-bench / 18.00% Lite / 3.8% prior non-interactive best | VERIFIED (arXiv + NeurIPS proceedings PDF; numbers read directly) |
| `liu2024lost` | Lost in the Middle, TACL 12:157–173, 2024, arXiv:2307.03172 | VERIFIED (arXiv + ACL Anthology) |
| `kapoor2024agentsmatter` | AI Agents That Matter, arXiv:2407.01502, Princeton, 2024 | VERIFIED (arXiv) |

Unverified citations: **none**.

## Numerical claims spot-check

- 95%^40 ≈ 0.1285 → "about 13%" (checked numerically).
- SWE-agent numbers (3.8% → 12.47% full set; 11.0% vs 18.0% on Lite) read directly from the NeurIPS 2024 paper text.
- Toolformer described as tool-augmented single-call level; ReAct as loop; Reflexion as cross-episode self-critique — matches each paper's abstract.

## Code run (real output, repo .venv python 3.14.7, 2026-10-10)

Command: `/home/stevens/ai-course-2026/.venv/bin/python code/minimal_agent.py` — exit code 0.

```
task: Which file defines PORT, and what is its value? Answer as '<file> <value>'.

== condition A: the model, WITH the harness loop ==
  step 1: search({'pattern': 'PORT'}) -> 'matches:notes.txt, server.cfg'  [context ~76 chars]
  step 2: read_file({'name': 'notes.txt'}) -> 'TODO: the default PORT should match producti'  [context ~181 chars]
  step 3: search({'pattern': 'PORT'}) -> 'matches:notes.txt, server.cfg'  [context ~257 chars]
  step 4: read_file({'name': 'server.cfg'}) -> 'HOST=0.0.0.0\nPORT=8080\nWORKERS=4\n'  [context ~343 chars]
  step 5: finish -> 'server.cfg 8080'
  trajectory: 5 steps, context grew every step

== condition B: the SAME model, single call (no loop) ==
  one decision: search({'pattern': 'PORT'}) -> observation discarded; nothing decides again
  answer=None

ASSERT looped run solves the task:        PASS
ASSERT single-call run cannot solve it:   PASS
ASSERT both runs used the identical policy and first action: PASS

Same 'model', different control flow => the loop supplies the agency.
```

Design notes: the trajectory deliberately includes a **search false positive** (notes.txt mentions PORT but does not define it), forcing the policy into error recovery at step 3 — the harness surfaces clean tool errors and the transcript (memory) records what was already read. Context grows monotonically every step (context rot, made visible). Condition B runs the byte-identical `model()` function with no loop: one decision, observation discarded, task unsolvable. The third assertion proves both conditions share the same policy and first action, so the only variable is control flow.

## Balance check on the Claude Code question

The note (§4) and deck take the position requested by the brief but hedge it: Claude Code is strictly a harness; the harness+model composite is an agent; "agentic harness" is offered as the best single phrase, with two explicit caveats (the model/harness boundary migrates over time as models absorb harness responsibilities; "autonomy" is loop+memory+stopping rule, not a persistent internal optimizer).

## Files

- `note.tex` / `note.pdf` — 5-page tutorial (target 3–5, at upper bound)
- `deck.tex` / `deck.pdf` — 10 slides (target 5–10, at upper bound; title + 9 content)
- `discussion.tex` / `discussion.pdf` — exactly 5 questions
- `projects.tex` / `projects.pdf` — 4 suggestions
- `quiz.tex` / `quiz.pdf` — 1 question + answer
- `code/minimal_agent.py` — stdlib-only, ~150 lines, asserts pass
- `bibliography.bib` — 10 entries, all verified

## Technical/evidence review — Ollie

Pending.

## Teaching/release review — Kukla

Pending.

## Author revision and independent recheck

Pending.
