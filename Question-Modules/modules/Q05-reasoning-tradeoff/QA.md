# Q05 QA and review record

- **Author:** Kukla
- **Question:** When does extra reasoning actually improve an answer, and what is the trade-off?
- **Survey source:** #05
- **Status:** author complete; peer review (Ollie, Paradise) pending.

## Author self-check
- [x] note.tex builds to PDF (3 pages), with \cite citations + rendered bibliography.
- [x] deck.tex builds, 6 content slides + title (<= 10-slide contract).
- [x] discussion (5 Qs), projects (3), quiz (1 + answer) build.
- [x] code/best_of_n.py runs on stdlib; output verified.
- [x] Code claims are accurate: best_of_n.py is described as a MATH SIMULATION, not a real LLM run.
- [x] Uncertainty/claims calibrated; task-dependence stated explicitly.

## Build command + expected output
`python3 code/best_of_n.py` -> prints coverage tables; confirms n=5 for >=0.95 at p1=0.5.

## Author response to Paradise review (Kukla, 2026-10-11)
Both non-blocking suggestions addressed:
1. **Video-to-section mapping** — added a "For TAs" block to `video.md` mapping the bycloud explainer to the note's "why extra reasoning helps" and "diminishing returns / optimal budget" sections.
2. **Marginal-gain table** — `best_of_n.py` already prints an explicit `marginal gain` column (delta coverage per doubling of n) alongside exact + Monte-Carlo coverage; re-verified it runs (n=5 for >=0.95 at p1=0.5). No further change needed.
Author improvements DONE; awaiting Ollie 2nd sign-off for double-review gate.
