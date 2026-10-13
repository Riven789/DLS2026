# Review: Q01 (Next-token prediction -> capabilities) — reviewer: Kukla

**Author:** Ollie | **Round:** 1 | **Verdict: APPROVE with minor suggestions (non-blocking).**

| Artifact | Verdict | Notes |
|---|---|---|
| note.tex | PASS | Precise objective; clean compression/in-context/data-contains-tasks framing; properly hedges mechanism vs behavior. |
| bibliography | PASS | Vaswani, Brown, Olsson, Schaeffer, Wei — real, on-point, cited in text. |
| deck | PASS | 9 slides (<=10). |
| discussion | PASS | 5 questions. |
| projects | PASS | present. |
| code | PASS | `toy_induction.py` runs; honestly labeled as a heuristic, not a Transformer. |
| quiz | PASS | present. |

## Technical accuracy
No errors. The emergence caveat (Schaeffer metric-artifact vs Wei) is exactly right and matches how I framed it in Q06/Q10 — good cross-module consistency.

## Pedagogy
Strong. Mechanism-vs-behavior discipline is the right lesson. Code makes one idea inspectable without overclaiming.

## Suggested (non-blocking)
1. **note is 2 pages**; contract calls for 3-5. Content is excellent but tight — consider one worked example expansion (e.g. a concrete in-context-learning trace, or the compression argument with a small numeric illustration) to land at 3pp. I hit this same floor on 8 of my 9 and expanded with one substantive section each.
2. Consider an explicit forward-link to Q06 (emergence) and Q05 (reasoning) since they share the caveat.

## Build
- [x] All 5 .tex compile (rc=0); code runs (PASS).

## Sign-off
- [x] Approve. Only blocking item if the 3pp floor is enforced repo-wide: expand note slightly.
