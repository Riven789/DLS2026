# Review: Q29 (Agent control and safety) — reviewer: Paradise

**Author:** Ollie | **Round:** 1 | **Verdict: APPROVE with minor suggestions (non-blocking).**

| Artifact | Verdict | Notes |
|---|---|---|
| note.tex | PASS | 3 pages (floor met). |
| bibliography | PASS | 4 entries incl. Greshake prompt injection, MCP spec, NIST AI RMF — real and appropriate for a control/safety framing. |
| deck | PASS | 9 slides (≤10). |
| discussion | PASS | 5 questions. |
| projects | PASS | present. |
| code | PASS | `approval_bound.py` rerun by reviewer: `PASS: only exact action/target/digest/approver approval is accepted` — a bound-approval check (action+target+digest+approver must match exactly), explicitly labeled illustrative, not production auth. Honest scoping. |
| quiz | PASS | present. |

## Technical accuracy
No errors found. Threat-model-first framing (injection, sandbox escape, misuse) with control layers (permissions, approval bounds, audit) matches current practice; Greshake is the canonical indirect-injection citation.

## Verified by me (independent rerun)
- Code rerun passes (output above).
- All 5 PDFs extract cleanly (6070/1584/388/315/386 chars); no unresolved refs.

## Suggested (non-blocking)
1. Could name the ASL/RSP framing (Anthropic RSP, OpenAI Preparedness) as live industry practice — one paragraph.
2. The approval-bound code checks exact-match; a sentence on why partial/fuzzy matching is dangerous (substring authority confusion) would strengthen the lesson.

## Sign-off
- [x] Approve. Clears "independent peer review" + "visual inspection" (structural pass). Companion video deferred repo-wide.
