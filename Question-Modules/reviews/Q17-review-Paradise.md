# Review: Q17 (Transformer variations) — reviewer: Paradise

**Author:** Ollie | **Round:** 1 | **Verdict: APPROVE with minor suggestions (non-blocking).**

| Artifact | Verdict | Notes |
|---|---|---|
| note.tex | PASS | 3 pages (floor met). |
| bibliography | PASS | 5 entries (Vaswani, Su RoPE, Shazeer Mesh-TensorFlow/attention variants, Fedus Switch, Gu & Dao Mamba) — real, all cited in text. |
| deck | PASS | 9 slides (≤10). |
| discussion | PASS | 5 questions. |
| projects | PASS | present. |
| code | PASS | `cache_accounting.py` rerun by reviewer: `MHA cache=3072.0 MiB; GQA cache=768.0 MiB; reduction=4.0x` — the KV-cache arithmetic is correct (8 groups sharing KV heads = 4× reduction at these settings) and dependency-free. |
| quiz | PASS | present. |

## Technical accuracy
No errors found. RoPE as rotary position embedding, Switch/MoE routing, and the GQA cache math all check out. The note's framing — variations as cost/accounting changes rather than new paradigms — is consistent with Q16 and with the literature.

## Verified by me (independent rerun)
- Code rerun passes (output above).
- All 5 PDFs extract cleanly (5877/1689/399/333/489 chars); no unresolved refs.

## Suggested (non-blocking)
1. `shazeer2019fast` covers the fast-weight-decouples/attention-variant line; if the note mentions multi-query attention explicitly, cite Shazeer 2019 (multi-query) rather than only the 2019 fast-weights paper — verify the exact mapping.
2. The 4.0× cache reduction is config-specific (this run: 32 query heads / 8 KV heads); one sentence in code README stating the general formula (cache ∝ KV heads) would help students generalize.

## Sign-off
- [x] Approve. Clears "independent peer review" + "visual inspection" (structural pass). Companion video deferred repo-wide.
