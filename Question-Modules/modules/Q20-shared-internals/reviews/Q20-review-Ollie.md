# Q20 technical/evidence review — Ollie

**Reviewer:** Ollie  
**Date:** 2026-10-10  
**Verdict:** **CHANGES REQUESTED**

## Evidence checked

- Freshly rebuilt `note.tex`, `deck.tex`, `discussion.tex`, `projects.tex`, and
  `quiz.tex` from `modules/Q20-shared-internals/`. The note is 3 pages and the
  deck is 10 slides; the five PDFs were produced successfully.
- Ran `python3 code/shared_skeleton.py` under the repository Python environment.
- Checked every `\cite{...}` key in `note.tex` against `bibliography.bib`:
  all eight cited keys resolve locally.
- Ran `python3 -m py_compile code/shared_skeleton.py`.

## Strengths

The note gives a clear, appropriately qualified architectural lineage and the
shared-loop teaching idea is promising. The artifact set builds, the note meets
the 3-page floor, and the code is dependency-free and seeded.

## Required revision

1. **Repair the toy experiment before using its numerical results.** `make_data`
   creates a transition matrix from its supplied RNG, but `__main__` constructs a
   *different* fresh transition matrix to print the claimed entropy floor.
   Consequently the reported losses can fall below the displayed “floor”; the
   current output does so for all three models. Return the actual transition
   matrix from `make_data` (or pass it in) and compute the entropy from that same
   matrix. Explain that this is the Bayes-optimal next-token cross-entropy for
   the synthetic source, not a generic training-loss bound.
2. **Repair or remove the RNN comparison.** `TinyRNN.backward` computes
   `dh_raw = ... * 0` and never uses it. Thus the recurrence gradient is not a
   valid BPTT implementation; the claimed architecture comparison is not
   technically supported. Implement and test the recurrence gradient (preferably
   with a small finite-difference check), or clearly limit the demo so it does
   not compare learned RNN behavior.
3. **Add a short `README.md` and `QA.md`.** State that the code is an
   illustrative NumPy toy rather than a faithful CNN/RNN/Transformer benchmark;
   give the exact run command and expected qualitative result. The QA record
   must log the corrected code run, fresh artifact builds, and later review
   closure.
4. **Add the eighth module-contract record.** The companion-video/alignment
   requirement may remain pending while video work is parked, but it must be
   explicitly tracked rather than absent from the module.

## Recheck required

After revision, a non-author must rerun the code, verify that final loss is not
reported below the entropy of the *actual* data-generating transition matrix,
inspect the finite-difference/gradient test, rebuild all PDFs, and then update
this review to an approve/changes-requested closure.
