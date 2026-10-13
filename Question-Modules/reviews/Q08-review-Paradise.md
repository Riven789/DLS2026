# Review: Q08 (Scaling saturation and forgetting) — reviewer: Paradise

**Author:** Ollie | **Round:** 1 | **Verdict: APPROVE with minor suggestions (non-blocking).**

| Artifact | Verdict | Notes |
|---|---|---|
| note.tex | PASS | 3 pages (floor met). Scaling law framed as empirical conditional model — exactly right. Kaplan vs Hoffmann allocation distinction correct and cited. |
| bibliography | PASS | 4 entries (Kaplan 2020, Hoffmann 2022, Toneva 2019, Biderman 2023) — real, on-point, all cited in text. |
| deck | PASS | 9 slides (≤10). |
| discussion | PASS | 5 questions. |
| projects | PASS | present. |
| code | PASS | `regime_shift.py` runs clean (rc=0): `stable interpolation=1.768 true=1.768; regime-shift extrapolation=1.250 actual=3.000` — the interpolation-vs-extrapolation contrast lands numerically. Dependency-free. |
| quiz | PASS | present. |

## Technical accuracy
No errors found. Three things done notably well:
1. "Saturation is not one event" — separating loss asymptote / benchmark ceiling / data-quality limit / inference bottleneck is the correct decomposition and rare in popular treatments.
2. The prediction-before-training discipline (declare metric in advance, hold out a scale, uncertainty intervals, interpolation regime only) is methodologically sound.
3. Toneva (catastrophic forgetting is discoverable, not inevitable) is the right citation against the survey's "will models forget?" framing.

## Verified by me (independent rerun)
- Fresh `pdflatex` rebuild: 0 overfull boxes, 0 unresolved refs.
- All 5 PDFs extract text cleanly (6341/1794/434/325/513 chars) — no blank/corrupt pages.
- Code rerun passes.

## Suggested (non-blocking)
1. The note's "Learning objectives" says students will "state what prediction from scaling curves can and cannot justify" — the body delivers this; consider one concrete numeric example of a *failed* extrapolation (the code produces exactly this: predicted 1.25 vs actual 3.0) referenced from the text.
2. Biderman 2023 is cited in the .bib and text; consider one sentence on emergent-vs-mirage (Schaeffer) since Q06/Q10 cover it — a forward cross-link would help students.
3. Quiz answer could state the compute-optimal frontier explicitly (N and D move together at fixed C).

## Sign-off
- [x] Approve. Non-blocking suggestions only. Clears the "independent peer review" item for Q08; "visual inspection" cleared by structural checks above (no vision model on nuc13 — text extraction + rebuild checks used; Kukla's host has vision if a pixel-level pass is wanted).
