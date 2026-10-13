# Review: Q05 (When does extra reasoning help) — reviewer: Paradise

**Author:** Kukla | **Branch:** kukla/Q05 | **Round:** 1 | **Verdict: APPROVE with minor suggestions (non-blocking).**

| Artifact | Verdict | Notes |
|---|---|---|
| note.tex | PASS | 3 pages (floor met), citations + rendered bibliography. |
| bibliography | PASS | 6 entries, all real. |
| deck | PASS | 7 pages incl. title (≤10 contract). |
| discussion | PASS | 5 questions. |
| projects | PASS | 3 suggestions. |
| code | PASS | `best_of_n.py` rerun by reviewer on the branch: coverage tables correct; `n=5 for >=0.95 at p1=0.5` verified; diminishing-returns framing lands. Stdlib-only. Explicitly labeled a MATH SIMULATION, not a real LLM run — honest scoping (this was Ollie's #1 correction on Q32; Kukla already internalized it here). |
| quiz | PASS | 1 question + answer. |
| video.md | PASS | bycloud "OpenAI o1's New Paradigm: Test-Time Compute Explained" — **live-verified by me just now: 15:43, in window, channel real.** Kukla did his own transcript-QC on 2026-10-10. This is the first module with a fully closed video item. |

## Technical accuracy
No errors found. The best-of-n coverage math (1-(1-p)^n), the diminishing-returns argument, and the task-dependence caveat are all correct. Video choice is squarely on-topic (test-time compute scaling).

## Verified by me (independent rerun, on branch)
- Code rerun passes (output above).
- All 5 PDFs extract cleanly (5127/1436/1014/654/794 chars).
- Video URL duration+channel re-verified live.

## Suggested (non-blocking)
1. The video is about o1/test-time compute broadly; one sentence in video.md mapping which note sections it pairs with would help TAs.
2. best_of_n.py could print the marginal-gain table (delta coverage per extra sample) explicitly rather than only asserting it.

## Sign-off
- [x] Approve (my half). Ollie's review still needed for double sign-off. Branch ready to merge to main once Ollie signs.
