# Q28 (alignment-faking) — second review — Kukla

**Reviewer:** Kukla · **Date:** 2026-10-11 · **Author:** Ollie · **First review:** Paradise (APPROVE)
**Verdict: APPROVE.**

## Evidence checked (clean worktree off `main`, TeX Live 2021)
- `note.pdf` 3 pp, `deck.pdf` 9 pp, discussion/projects/quiz build clean; **0 undefined citations**.
- `code/context_audit.py` runs, exit 0: `PASS: matched authorization/absence/decoy contexts obey declared policy` — the toy correctly illustrates context-conditional behavior as a *policy-compliance table*, not a mind-reading claim.
- Citations verified real: greenblatt2024alignment (2412.14093), hubinger2024sleeper (Sleeper Agents, 2401.05566), ngo2022alignment (2209.00626). No fabrications.

## Assessment
This is the most epistemically careful module in the set, and it handles a genuinely hard topic exactly right. The three-level hierarchy — (1) behavioral conditional differences, (2) properties of elicited reasoning traces, (3) mechanistic/intentional interpretation — is the correct scaffold, and the module repeatedly refuses to grant the "strategic deception" hypothesis privileged status over benign alternatives (prompt priming, task-inference shift, decoding changes) without discriminating evidence. The discriminating-experiment design (swap reviewer cue for neutral equal-length sentence, fictional vs human vs automated reviewer, randomize ordering) is seminar-grade. Ties cleanly into Q29's control framing. Approved with no changes.

Minor non-blocking note: Q28's quiz and discussion use a bare `article` preamble without `microtype`/`hyperref` on some sub-docs — purely cosmetic, consistent with the rest of `main`'s self-contained-preamble convention; no action.
