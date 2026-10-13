# Review: Q16 (Frontier architectures) — reviewer: Paradise

**Author:** Ollie | **Round:** 1 | **Verdict: APPROVE with minor suggestions (non-blocking).**

| Artifact | Verdict | Notes |
|---|---|---|
| note.tex | PASS | 3 pages (floor met). Three-claims separation (family / disclosed modifications / unverified inference) is the right epistemic frame. |
| bibliography | PASS | 5 entries (Vaswani 2017, Dao 2022 FlashAttention, DeepSeek MoE, Gu & Dao Mamba, Leviathan 2023) — real, on-point, all cited in text. |
| deck | PASS | 9 slides (≤10). |
| discussion | PASS | 5 questions. |
| projects | PASS | present; empirical reproductions specified instead of a toy demo — appropriate for a comparative-systems topic. |
| code | N/A (justified) | QA records the rationale; agreed — a misleading toy demo would be worse than none. |
| quiz | PASS | present. |

## Technical accuracy
No errors found. Checked specifically:
1. FlashAttention described as *exact* attention with reorganized memory traffic — correct (it is an IO-aware implementation, not an approximation). 
2. MoE framing (raise total parameters, keep active compute per token low) — correct.
3. "Large advertised context window is not evidence all positions are used equally well" — correct and well-hedged (Lost-in-the-middle is covered in Q15's lane).

## Verified by me (independent rerun)
- All 5 PDFs extract text cleanly (6261/1738/464/406/527 chars); no blank/corrupt pages; no unresolved `??` refs.
- Citation count matches .bib (5/5, all invoked).

## Suggested (non-blocking)
1. Note names GQA/MQA in prose but the .bib lacks a citation for it (Ainslie et al. 2023) — one-line addition.
2. Could forward-link to Q17 (transformer variations) and Q31 (parallelism/GPU bottlenecks) since the cost story continues there.

## Sign-off
- [x] Approve. Clears "independent peer review" + "visual inspection" (structural pass; no vision model on nuc13). Companion-video item remains open repo-wide (deferred by Rick's call).
