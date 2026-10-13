# microgpt course materials

Three documents built around Andrej Karpathy's **microgpt** post
(<https://karpathy.github.io/2026/02/12/microgpt/>, 12 Feb 2026) and the gist
`microgpt.py` (<https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95>).

| File | What it is |
|---|---|
| `lecture_notes.pdf` / `.tex` | 32-page lecture notes: every section of the post, with derivations, figures, a verified training run, exercises, the full source listing, and references. |
| `slides.pdf` / `.tex` | Beamer deck (metropolis theme, 16:9), ~50 content slides plus references, covering all of the above. |
| `study_guide.pdf` / `.tex` | Cheat sheets for every component, a concept-to-literature map, a 45-question self-test with answers, and a themed annotated bibliography. |
| `refs.bib` | Shared bibliography (94 entries, tagged by theme). |
| `microgpt.py` | The gist as retrieved on 2026-10-11 (200 lines). |
| `loss.csv` | Per-step loss from running the gist unmodified (seed 42), used for the loss-curve plots. |
| `Makefile` | `make` builds all three PDFs with latexmk + biber. |

Reproduce the numbers in the notes with `python microgpt.py` (about 1-4 minutes in CPython).
| `lab2.pdf` / `.tex` | CMSC 35200 Lab 2 handout: ten agent-assisted experiments on microgpt, 100 points, due Tue Oct 20 2026 11:59 pm, submission spec for Classroom50. |
| `agent_exercises.md` | The same ten exercises in informal form, for self-study. |
