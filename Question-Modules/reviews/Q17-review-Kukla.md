# Q17 (transformer-variations) — second review — Kukla

**Reviewer:** Kukla · **Date:** 2026-10-11 · **Author:** Ollie · **First review:** Paradise (APPROVE)
**Verdict: APPROVE.**

## Evidence checked (clean worktree off `main`, TeX Live 2021)
- `note.pdf` 3 pp, `deck.pdf` 9 pp, discussion/projects/quiz build clean; **0 undefined citations**.
- `code/cache_accounting.py` runs, exit 0: `PASS: MHA cache=3072.0 MiB; GQA cache=768.0 MiB; reduction=4.0x` — the KV-cache arithmetic is correct and makes the MQA/GQA point concrete.
- Citations verified real: vaswani2017attention, shazeer2019fast (MQA, 1911.02150), su2021roformer (RoPE, 2104.09864), fedus2022switch (Switch, 2101.03961), gu2023mamba. No fabrications.

## Assessment
Very strong. The standout teaching move is cleanly separating the three things routinely conflated as "sparsity": sparse expert activation (MoE) vs sparse attention (position connectivity) vs quantization (numeric precision) — the quiz drills exactly this. The "worked design diagnosis" (100k-token document → which bottleneck → which variant) translates product requirements into measurable architectural hypotheses, which is the pedagogy Rick wants. Scope overlap with Q16 is intentional and complementary (see Q16 review). Approved with no changes.
