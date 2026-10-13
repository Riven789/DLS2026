# Q16 (frontier-architectures) — second review — Kukla

**Reviewer:** Kukla · **Date:** 2026-10-11 · **Author:** Ollie · **First review:** Paradise (APPROVE)
**Verdict: APPROVE.**

## Evidence checked (clean worktree off `main`, TeX Live 2021)
- `note.pdf` 3 pp, `deck.pdf` 9 pp, discussion/projects/quiz build clean; **0 undefined citations**.
- No `code/` directory — appropriate for a conceptual/landscape module; the quiz supplies a concrete numeric reasoning check (MoE total-vs-active parameters) that substitutes well for runnable code here.
- Citations verified real: vaswani2017attention (1706.03762), gu2023mamba (2312.00752), deepseek2024moe (DeepSeek-V2, 2405.04434), dao2022flashattention (2205.14135), leviathan2023fast (speculative decoding, 2211.17192). No fabrications.

## Assessment
Strong. The three-claim framing (computational family / disclosed modifications / unverified inference) is the right way to teach "is it still a transformer?" without reverse-engineering undisclosed production stacks. Correctly separates architecture from serving-system optimizations (FlashAttention and speculative decoding explicitly flagged as *not* changing the learned function), and the MoE "accounting" section (total vs active parameters, communication cost) is precise. Note: Q16 and Q17 overlap in scope (both cover MoE/Mamba/attention variants) — this is by design per the master list, and each frames the material differently (Q16 = "what do frontier models use," Q17 = "how/why vary the transformer"), so no dedup needed. Approved.
