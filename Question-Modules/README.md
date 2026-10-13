# AI Course 2026 — Modular Course Materials

One learning module per student question from the Week-2 intake survey (41 responses, Oct 8 2026).
The survey's Section 2 contains **32 articulated questions** → **32 modules**.

## Module contract

Every module directory `modules/Q<NN>-slug/` contains:

1. **`note.tex`** — 3–5 page narrative explanatory note/tutorial (compiled to PDF)
2. **`bibliography.bib`** — verified citations only (no invented references)
3. **`deck.tex`** — Beamer lecture notes, 5–10 slides, 16:9
4. **`discussion.tex`** — five questions for further discussion
5. **`projects.tex`** — suggestions for projects that go deeper
6. **`code/`** — runnable, tested code where applicable (CPU-only defaults)
7. **`quiz.tex`** — one small comprehension-check question with answer
8. **Short companion videos** — one or more real, verified YouTube pointers, normally **10–15 minutes each**
   (checker tolerance 9.5–16.5 min). A module may use multiple focused videos when one cannot responsibly
   cover its learning objectives—for example, separate primers for architecture and systems behavior. Document
   each URL, channel/speaker, runtime, verification date, objective(s) served, transcript-alignment verdict, and
   one-line teaching use in `sources.md` or `video.md`. Videos are supplemental, never substitutes for the cited
   tutorial. **Enforcement:** run `python3 scripts/check_video.py <URL>` — it live-checks the link (oEmbed
   title/channel + duration) and exits 0 only if in the tolerance window; re-run during review. Prefer credible
   educators, research organizations, or primary technical speakers; flag an exceptional runtime outside 10–15
   minutes with a reason.

All documents LaTeX → PDF. Cross-review: each module is reviewed by the two
non-author agents before it is marked done.

## Owners

- **Paradise (nuc13)** — Q03, Q09, Q11, Q12, Q13, Q20, Q21, Q22, Q23, Q26, Q27 (11)
- **Ollie (cherryrd)** — Q01, Q02, Q04, Q06, Q08, Q16, Q17, Q24, Q28, Q29, Q31 (11)
- **Kukla (m1)** — Q05, Q05, Q07, Q10, Q14, Q15, Q18, Q19, Q25, Q30, Q32 (10) — *typo, should be Q05 once*

- **Kukla (m1)** — Q05, Q07, Q10, Q14, Q15, Q18, Q19, Q25, Q30, Q32 (10)

## Layout

```
questions/     master questions list (LaTeX + PDF)
preamble/      shared LaTeX preamble (preamble.sty) — unified course look
modules/       one directory per module
survey/        source survey PDF
```

## Build

```bash
cd modules/Q01-*/ && pdflatex note.tex && pdflatex deck.tex
```

Shared course palette (ink navy + terracotta) and author header signatures
(PARADISE / OLLIE / KUKLA) live in `preamble/preamble.sty`.
