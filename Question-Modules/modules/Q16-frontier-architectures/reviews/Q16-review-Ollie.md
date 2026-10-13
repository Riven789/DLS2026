# Q16 technical/evidence review — Ollie

**Reviewer:** Ollie  
**Date:** 2026-10-10  
**Verdict:** **APPROVE, with release conditions still open**

## Evidence checked

- Rebuilt `note.tex`, `deck.tex`, `discussion.tex`, `projects.tex`, and
  `quiz.tex` locally from the module directory. All completed without undefined
  citations/references or overfull-box errors.
- Verified artifact dimensions: note 3 pages; deck 9 slides; discussion,
  projects, and quiz each one page.
- Checked all five cite keys in `note.tex` against `bibliography.bib`; all
  resolve locally.
- Reviewed the note, deck, discussion prompts, projects, and quiz for the
  stated objectives and for the architecture-versus-systems distinction.

## Strengths

The module makes the important distinctions explicit: FlashAttention and
speculative decoding are systems/inference improvements rather than automatic
architecture replacements; MoE total parameters, active parameters, and
serving cost are different measures; and closed-model claims are bounded by
what is disclosed. The stated comparison protocol appropriately avoids a
one-paper “replacement” conclusion.

## Required release closure (not a technical rejection)

1. Complete the separately required teaching/release review.
2. Complete actual visual inspection of each rendered PDF; a clean LaTeX build
   and text extraction are not a visual-layout inspection.
3. Add the required companion-video metadata and transcript-alignment record,
   or explicitly track it as pending while the course-wide backfill is parked.
4. After the second review, author records any revisions and an independent
   rebuild/recheck before changing module status to complete.
