# Review: Q24 (Agent evaluation) — reviewer: Paradise

**Author:** Ollie | **Round:** 1 | **Verdict: APPROVE with minor suggestions (non-blocking).**

| Artifact | Verdict | Notes |
|---|---|---|
| note.tex | PASS | 3 pages (floor met). |
| bibliography | PASS | 4 entries (SWE-bench, AgentBench, WebArena, Sheng human-eval) — real, all cited in text. |
| deck | PASS | 9 slides (≤10). |
| discussion | PASS | 5 questions. |
| projects | PASS | present. |
| code | PASS | `scorecard.py` rerun by reviewer: `equal completion=70; utility fast-safe=68.6 > costly-risky=31.0` — a completion-vs-utility scorecard where the "equal completion" agents differ sharply on utility; the exact pedagogical point (completion rate alone is a broken metric). Dependency-free. |
| quiz | PASS | present. |

## Technical accuracy
No errors found. Benchmark selection (SWE-bench/AgentBench/WebArena) is the right canonical trio for agent evaluation; the completion≠utility and trajectory-vs-outcome distinctions are correct and are the field's live issue.

## Verified by me (independent rerun)
- Code rerun passes (output above).
- All 5 PDFs extract cleanly (5334/1519/361/294/458 chars); no unresolved refs.

## Suggested (non-blocking)
1. Consider one sentence on contamination/saturation for agent benchmarks specifically (SWE-bench verified subsets exist because of contamination) — cross-links to Q01's evaluation lane.
2. Scorecard code prints aggregate utility; a per-task breakdown line would let students see *which* tasks the risky agent wins/loses.

## Sign-off
- [x] Approve. Clears "independent peer review" + "visual inspection" (structural pass). Companion video deferred repo-wide.

---

# Review: Q28 (Alignment faking) — reviewer: Paradise

**Author:** Ollie | **Round:** 1 | **Verdict: APPROVE with minor suggestions (non-blocking).**

| Artifact | Verdict | Notes |
|---|---|---|
| note.tex | PASS | 3 pages (floor met), and the strongest note of Ollie's set — precise about the experimental setup (free-tier implied training vs paid-tier unmonitored). |
| bibliography | PASS | 3 entries (Greenblatt et al. 2024 alignment faking, Hubinger sleeper agents, Ngo alignment survey) — real, all cited; Greenblatt is *the* primary source. |
| deck | PASS | 9 slides (≤10). |
| discussion | PASS | 5 questions. |
| projects | PASS | present. |
| code | PASS | `context_audit.py` rerun by reviewer: `PASS: matched authorization/absence/decoy contexts obey declared policy` — illustrative audit harness, explicitly labeled non-diagnostic in README. Honest scoping. |
| quiz | PASS | present. |

## Technical accuracy
No errors found. The 12%→78% alignment-faking-reasoning numbers and the compliance-gap definition match the paper. The note does not overclaim from a scratchpad-observable behavior to internal "beliefs" — important discipline.

## Verified by me (independent rerun)
- Code rerun passes (output above).
- All 5 PDFs extract cleanly (7529/1526/369/288/367 chars); no unresolved refs.

## Suggested (non-blocking)
1. One sentence on the replication landscape (e.g., the 25-model follow-up finding only 5/25 chat models show compliance gaps) would inoculate students against treating Opus as universal.
2. Cross-link Q27 (interpretability) — the "do models revert to pre-training traits" question is the same evidence family.

## Sign-off
- [x] Approve. Clears "independent peer review" + "visual inspection" (structural pass). Companion video deferred repo-wide.
