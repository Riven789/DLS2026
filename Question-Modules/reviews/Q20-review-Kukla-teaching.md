# Q20 (shared-internals) — teaching/release review — Kukla

**Reviewer:** Kukla
**Date:** 2026-10-11
**Lane:** teaching / release (instructional clarity, rendering, contract completeness).
Technical/evidence review is Ollie's (`reviews/Q20-review-Ollie.md`, CHANGES
REQUESTED on the code) — this review deliberately does **not** re-litigate the
code issues; it covers the teaching material and build/release side.
**Verdict:** **CHANGES REQUESTED** (one blocking render defect; otherwise strong).

## Evidence checked

- Rebuilt all five PDFs from `modules/Q20-shared-internals/` with TeX Live 2021
  (`pdflatex`/`bibtex`/`pdflatex`×2 for the note; `xelatex` for the deck).
  Page/slide counts: note **3 pp**, deck **10 frames/pp**, discussion 1 p,
  projects 1 p, quiz 1 p — all within contract.
- Verified all 8 `\cite` keys in `note.tex` resolve from `bibliography.bib`
  (0 undefined after a correct bibtex pass; the References section renders).
- Read note, discussion (5 questions), projects (4 suggestions), quiz
  (1 comprehension question **with** a worked answer).

## Strengths (teaching side)

- The "shared skeleton vs. swapped mixer / tweaked objective" framing is an
  excellent mental model and is carried consistently from the learning objectives
  through to the "map you can carry" closer.
- The lineage table (Table 1) is exactly the one-diagram-per-idea density Rick
  wants: inherited-vs-new is legible at a glance.
- The two "transfer mistakes to avoid" (attention-as-explanation;
  vision-style transfer learning) are high-value and correctly scoped.
- Discussion questions are genuinely open/seminar-grade, not fill-in-the-blank;
  the quiz ships a model answer.

## Required revision (blocking)

1. **Garbled sentence in `note.tex` line 53** — currently renders as:
   *"The practical reading: an LLM is a \emph{ Transformers are a special case of
   "neural sequence model" that won the scaling race, not a new kind of
   machine.}"* This is a corrupted edit: a dangling `an LLM is a`, an unclosed
   thought, and **straight double-quotes** (`"neural sequence model"`) instead of
   LaTeX `` ``...'' ``, which render as wrong glyphs. This is the sentence a
   student will quote back, so it must be clean. Suggested fix:
   *"The practical reading: a Transformer is a special case of `neural sequence
   model' that won the scaling race, not a new kind of machine."* (and use
   LaTeX quotes). This is the one item that should block release.

## Non-blocking teaching notes

2. **Deck is at the top of the slide band (10 frames).** Fine as-is (contract is
   5–10), but there is no headroom; if Ollie's code-fix adds a results slide,
   split rather than cram (per Rick's no-compression rule).
3. The note uses its own self-contained preamble (not the shared
   `preamble.sty`), consistent with the rest of `main`'s modules — no action
   needed, just noting for the eventual style-unification pass.

## Release-gate status

Teaching material is **release-ready except for item 1**. The module still needs
Ollie's technical closure (code `make_data` entropy-floor bug + RNN BPTT
`dh_raw * 0` bug) and the element-8 companion-video record before it can go to
two-approval. Re-ping me after item 1 is fixed and I'll re-verify the note render
and move this to APPROVE on the teaching lane.

## Re-review after Paradise revision — Kukla (2026-10-11) — APPROVE (teaching lane)

**Verdict: APPROVE (teaching/release lane).** Re-verified against revised `main`
(commit at re-review time; floor-from-transition-matrix + train/held-out split
revision).

- **Item 1 (blocking garbled L53) — CLOSED.** `note.tex:53` now reads cleanly:
  *"The practical reading: an LLM is a particular composition of inherited parts,
  not a new kind of machine --- Transformers are a special case of ``neural
  sequence model'' that won the scaling race."* Proper LaTeX quotes, complete
  sentence, no dangling clause. This was my only release blocker — removed.
- **Build re-verified from a clean worktree off `main`:** note 3 pp (References
  render, 0 undefined citations after bibtex), deck 10 frames/pp, discussion 1 p,
  projects 1 p, quiz 1 p — all within contract.
- **Code re-run (`code/shared_skeleton.py`, exit 0):** output now honest —
  `train loss 1.5234 | held-out 2.6808 | Bayes CE 1.717`. The floor is computed
  from the actual transition matrix (not an independent re-draw), relabeled
  Bayes-optimal CE, and the train/held-out split makes the memorization-vs-source
  point correctly (held-out > Bayes CE, train < Bayes CE). This resolves the
  entropy-floor concern on the teaching side.

Teaching lane is **APPROVED**. Remaining to full release is Ollie's technical
re-review sign-off and the element-8 companion-video record (videos parked per
Rick).
