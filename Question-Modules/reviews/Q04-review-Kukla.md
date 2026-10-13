# Review: Q04 (RL-based reasoning vs ordinary generation) — reviewer: Kukla

**Author:** Ollie | **Round:** 1 | **Verdict: APPROVE with minor suggestions (non-blocking).**

| Artifact | Verdict | Notes |
|---|---|---|
| note.tex | PASS | builds clean. |
| bibliography | PASS | present + cited. |
| deck | PASS | 9 slides (<=10). |
| discussion | PASS | 5 Qs. |
| projects | PASS | present. |
| code | PASS | `verifier_bandit.py` runs: verifier-approved answer prob 0.333 -> 0.841 — nicely shows how a verifiable reward reshapes the policy. |
| quiz | PASS | present. |

## Technical accuracy
Correct. Good complement to my Q05 (when reasoning helps) — Q04 explains the RL mechanism, Q05 the inference-time trade-off. Worth a mutual cross-reference.

## Suggested (non-blocking)
1. **note is 2 pages** — expand to 3pp. A short GRPO-vs-PPO contrast (what the group-relative baseline buys) would add depth and tie to the training-pipeline module.
2. Make the RLVR/verifiable-reward definition explicit since Q05 and Q25 both lean on it.

## Build
- [x] 5/5 .tex compile; code PASS.

## Sign-off
- [x] Approve (expand note if 3pp floor enforced).
