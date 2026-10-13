# Q21 QA and review record

- **Author:** Paradise
- **Survey source:** Q21 (Section 2 survey) — "How do different networks and learned RL policies combine in one system, and what is the design intuition?"
- **Status:** drafted; peer review (Ollie + Kukla) pending. Video element parked per Rick's call.

## Author self-check

- [x] note.tex builds (pdflatex + bibtex + 2x pdflatex): **5 pages**, 0 Overfull boxes, 0 unresolved `??`
- [x] deck.tex builds: **15 content slides + 3 reference slides = 18 PDF pages**, 0 undefined citations, 0 Overfull
- [x] discussion.tex builds: 1 page, **exactly 5 questions**
- [x] projects.tex builds: 1 page, **4 suggestions**
- [x] quiz.tex builds: 1 page, **1 question + answer**
- [x] bibliography.bib: 13 entries, all real and web-verified this session; every `\cite` key in note.tex and deck.tex resolves; no unused entries
- [x] code runs with repo `.venv` python (NumPy 2.5.4, Python 3.14.7), rc=0, all 4 assertions PASS
- [x] No git commit performed (per task instruction); nothing outside this module dir touched

## Citation verification (2026-10-10, web-checked)

| Key | Claim verified |
|---|---|
| sutton1999options | Artificial Intelligence 112(1-2):181-211, 1999 — options / semi-MDP framework (ACM DL + author PDF) |
| shazeer2017sparsely | ICLR 2017 — sparsely-gated MoE, top-k routing (arXiv 1701.06538 + Hinton absps PDF) |
| schrittwieser2020muzero | Nature 588:604-609, Dec 2020 — learned model + tree search (nature.com) |
| yao2023react | ICLR 2023 — ReAct interleaved reasoning/acting (arXiv 2210.03629 + iclr.cc poster page) |
| jacobs1991adaptive | Neural Computation 3(1):79-87, 1991 — adaptive mixtures of local experts (cs.toronto.edu PDF) |
| dietterich2000maxq | JAIR 13:227-303 (paper 10266) — MAXQ value function decomposition (jair.org + arXiv cs/9905014) |
| bacon2017optioncritic | AAAI 2017, pp. 1674-1680 — option-critic (ojs.aaai.org) |
| fedus2022switch | JMLR 23(120):1-39, 2022 — Switch Transformers (jmlr.org + official .bib) |
| ahn2022saycan | arXiv 2204.01691, 2022 — SayCan grounding LLM in affordances (arXiv + say-can.github.io) |
| schick2023toolformer | NeurIPS 2023 — Toolformer self-supervised tool use (arXiv 2302.04761 + neurips.cc) |
| wei2022chain | NeurIPS 2022 — chain-of-thought prompting (arXiv 2201.11903 + neurips.cc) |
| haarnoja2018composable | ICRA 2018, arXiv 1803.06773 — composable soft Q-learning (arXiv) |
| levine2016end | JMLR 17(39):1-40, 2016 — end-to-end visuomotor policies (jmlr.org v17 15-522) |

All 13 verified against publisher pages, arXiv, or author-hosted PDFs during this session. Ahn et al. listed with first 7 authors + "and others" (46-author paper; standard practice).

## Code: build & run commands (reviewers can rerun)

```
cd modules/Q21-composed-systems
/home/stevens/ai-course-2026/.venv/bin/python code/moe_vs_monolith.py
```
Runtime ~2.5 s, deterministic (seeded), NumPy only.

## Code: REAL output (captured this session, rc=0)

```
Q21 demo: learned-router MoE (4 experts) vs larger monolith
data: 4000 train / 1000 test, 4 clusters, d_in=8, d_out=4
parameters: mixture=180, monolith=316 (monolith is GIVEN more params)

held-out MSE  (learned router)  : 0.02927
held-out MSE  (monolith)        : 0.06587
held-out MSE  (random routing)  : 10.24029   <- same experts, dead router
routing accuracy vs latent cluster: 100.0%
expert x true-cluster assignment matrix (rows=expert, cols=true cluster):
[[  0 262   0   0]
 [  0   0   0 257]
 [255   0   0   0]
 [  0   0 226   0]]

[PASS] router learns cluster assignment (>= 90%)
[PASS] composed MoE beats monolith (lower MSE)
[PASS] dead router loses the advantage
[PASS] MoE improves on random routing baseline

ALL CHECKS PASSED: composition pays when the router can learn the
decomposition; the advantage is the interface, not the parameters.
```

**Honesty note:** this is a synthetic-cluster simulation, labeled as such in the script docstring and in the note (Section "The toy demo"). It demonstrates the *structural* claim (gating learns the decomposition; composition beats a larger monolith; a dead router destroys the advantage) — it is not an LLM or robot run.

## Build commands

```
cd modules/Q21-composed-systems
pdflatex note.tex && bibtex note && pdflatex note.tex && pdflatex note.tex
pdflatex deck.tex && bibtex deck && pdflatex deck.tex && pdflatex deck.tex
pdflatex discussion.tex
pdflatex projects.tex
pdflatex quiz.tex
```

## Peer review

- Ollie: pending
- Kukla: pending
