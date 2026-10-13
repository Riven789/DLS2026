# Module Canonicalization — Paradise lane (Q03/Q11/Q12/Q13/Q27 duplication resolution)

**Date:** 2026-10-11 · **Decided by:** Paradise (lane owner) after owner-review QA of both candidate sets
**Context:** Kukla's helper agent drafted modules for Paradise's lane (finishing-everything push) under question-accurate slugs, while Paradise's subagents had drafted the same Q-numbers under topic-drifted slugs (Q03 drafted as RL-finetuning — actually Q04 territory; Q11/Q12 as adjacent-but-different framings; Q13 near-identical). Both sets are on main.

**Rule applied:** the master list (questions/MASTER-QUESTIONS.tex) is authoritative for topic mapping. The module whose content answers the actual student question is canonical; the other becomes a labeled extension.

## Canonical assignments
| Q | Canonical module | Disposition of the duplicate |
|---|---|---|
| Q03 (reasoning real/illusion) | `modules/Q03-reasoning-real-or-illusion` (Kukla helper) — owner-review QA'd by Paradise: 5pp note, 11pp deck, real citations (Wei CoT, Wang self-consistency, Turpin unfaithfulness, McCoy artifacts, Schaeffer emergence), no code (topic-appropriate) | `modules/Q03-rl-finetuning` → **extension** `Q03-ext-rl-finetuning` (Paradise): solid module but topic belongs to Q04's RL-reasoning territory; relabeled, cross-linked from Q04 |
| Q11 (hallucination + knowing unsure) | `modules/Q11-hallucination-and-uncertainty` (Kukla helper) — QA'd: 5pp, code runs (softmax-forced-distribution insight), cites Ji/Kalai-Vempala family | `modules/Q11-hallucination` → **extension** `Q11-ext-mechanistic-hallucination` (Paradise): deeper mechanistic treatment (frequency-vs-accuracy staircase, Good-Turing) — genuinely complementary, kept as extension |
| Q12 (confident when wrong) | `modules/Q12-confident-when-wrong` (Kukla helper) — QA'd: 5pp, code runs (OOD confidence blowup) | `modules/Q12-knowledge-boundaries` → **extension** `Q12-ext-knowledge-boundaries` (Paradise): the boundary-map material (frequency, cutoff, retrieval) — complementary, kept as extension |
| Q13 (represent/store knowledge) | `modules/Q13-knowledge-representation-storage` (Kukla helper) — QA'd: 5pp, 12 cites | `modules/Q13-knowledge-representation` → **extension** `Q13-ext-superposition` (Paradise): superposition demo passes; complementary depth |
| Q27 (interpretability, weights, reversion) | `modules/Q27-interpretability` (Paradise — owner draft, 13 verified cites incl. Arditi refusal direction, Qi fine-tuning attacks, Greenblatt alignment faking; refusal-direction code 7/7 PASS) | `modules/Q27-interpretability-weights` → **extension** `Q27-ext-weights-fingerprint` (Kukla helper): the frozen-weights/distribution-shift insight is a good complement |

## QA evidence (owner review of helper drafts, this session)
- Q03: note 5pp/0 unresolved, deck 11pp, 12 unique natbib cites vs 11 bib entries (one entry carries two keys — checked, resolves), discussion has 5 framed questions (itemize format, not enumerate — cosmetic).
- Q11: note 5pp, code `hallucination_and_uncertainty.py` rc=0 (softmax forced-distribution + entropy insight).
- Q12: note 5pp, code `confident_when_wrong.py` rc=0 (OOD overconfidence demonstration).
- Q13: note 5pp, 11 unique cites = 11 bib entries.
- Q27-ext: note 6pp, code rc=0 (frozen-weights/distribution-shift account).

## What was NOT accepted blindly
The helper drafts were merged by a third party (Tipizen) before lane-owner review. Content QA'd above is sound; the process gap (lane-owner sign-off before merge) is flagged to the team: **helper/assistant drafts into another agent's lane require the lane owner's review before merge to main.** No harm done here; the rule prevents worse cases.

## Cross-links to add
- Q04 (Ollie, RL-based reasoning) ← `Q03-ext-rl-finetuning` (DPO/GRPO depth)
- Q03 canonical ← Turpin unfaithfulness (already cited); add pointer to Q28 alignment-faking
- Q11 canonical ↔ Q11-ext (mechanistic) ↔ Q12 canonical (confidence) — teaching sequence
