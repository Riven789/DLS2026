# Q29 (agent-control-safety) — second review — Kukla

**Reviewer:** Kukla · **Date:** 2026-10-11 · **Author:** Ollie · **First review:** Paradise (APPROVE)
**Verdict: APPROVE.**

## Evidence checked (clean worktree off `main`, TeX Live 2021)
- `note.pdf` 3 pp, `deck.pdf` 9 pp, discussion/projects/quiz build clean; **0 undefined citations**.
- `code/approval_bound.py` runs, exit 0: `PASS: only exact action/target/digest/approver approval is accepted` — the allowlist checker correctly rejects near-matches and replayed generic "approved" tokens, which is precisely the point the note makes about semantic (not decorative) approval gates.
- Citations verified real: greshake2023not (indirect prompt injection, 2302.12173), schulhoff2023prompt (Prompt Report, 2406.06608), anthropic2024mcp (MCP spec), nist2023ai (NIST AI RMF 1.0). No fabrications.

## Assessment
Excellent and highly practical. "Control is a systems property, not a personality trait of an LLM" is the right thesis, and the module delivers on it: least-privilege authority boundaries, prompts-are-not-security-boundaries (with indirect-prompt-injection as the concrete threat), semantic approval binding, defense-in-depth, and observability/recovery. The worked control plan (research agent: read-only archives, sandboxed dir, cannot email/publish/merge, human-reviewed diff + narrow-cred merge bot, injection payload in a retrieved PDF as a test case) is directly reusable as a reference architecture — notably it mirrors our own fleet's agent-safety posture. Approved with no changes.

Minor non-blocking: bib key `schulhoff2023prompt` is dated 2023 but the paper is 2024 (arXiv:2406.06608). Cosmetic key naming only; the rendered reference reads year 2024 correctly. No fix required.
