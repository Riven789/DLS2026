# Q08 (scaling-limits) — second review — Kukla

**Reviewer:** Kukla · **Date:** 2026-10-11 · **Author:** Ollie · **First review:** Paradise (APPROVE)
**Verdict: APPROVE.**

## Evidence checked (clean worktree off `main`, TeX Live 2021)
- `note.pdf` 3 pp, `deck.pdf` 9 pp, discussion/projects/quiz build clean; **0 undefined citations** after pdflatex→bibtex→pdflatex×2.
- `code/regime_shift.py` runs, exit 0: `PASS: stable interpolation=1.768 true=1.768; regime-shift extrapolation=1.250 actual=3.000` — the assertion correctly demonstrates that a within-regime fit mispredicts after a regime shift.
- Citations all verified real: kaplan2020scaling (2001.08361), hoffmann2022training (Chinchilla, 2203.15556), biderman2023emergent (2304.11158), toneva2019empirical (1812.05159). No fabrications.
- Contract: 5 discussion Qs, projects suggestions, 1 comprehension quiz w/ expected answer.

## Assessment
Excellent epistemic discipline. The module never overclaims: it separates loss-forecasting from downstream-capability claims, enumerates the *several* distinct meanings of "saturation" and "forgetting," and ties the synthetic code to a methodological warning rather than a frontier forecast. The worked forecast protocol (reserve a held-out scale, declare the metric in advance, report structural vs statistical uncertainty) is exactly the right teaching frame. Approved with no changes.
