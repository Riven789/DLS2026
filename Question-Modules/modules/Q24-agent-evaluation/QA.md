# Q24 QA and review record
- **Author:** Ollie
- **Survey source:** responses #24 and #25
- **Status:** author draft; awaiting two independent peer reviews.

## Author self-check
- [x] Tutorial compiled with local BibTeX; PDF is 3 pages.
- [x] Deck compiled; PDF is 9 slides.
- [x] Discussion contains exactly five questions.
- [x] Bibliography/source ledger reviewed against source records.
- [x] `python3 code/scorecard.py` passes (Python standard library only).
- [x] All five PDFs are non-empty and rendered successfully.
- [x] Visual inspection: structural pass by Paradise reviewer — see reviews/Q24-Q28-review-Paradise.md. Companion video deferred repo-wide per Rick.
- [x] Independent peer review: APPROVED by Paradise (2026-10-10). Kukla review still open for double sign-off.

## Technical/evidence review — Kukla
Pending.

## Teaching/release review — Paradise
Pending.

## Author revision and independent recheck
Pending.

## Revision 1 (2026-10-10, Paradise applying Kukla's REQUEST CHANGES)
1. Replaced fabricated `sheng2021human` (arXiv:2105.03919 is actually a condensed-matter paper — verified) with Chatbot Arena (Chiang et al. 2024, arXiv:2403.04132 — verified real).
2. SWE-bench entry fixed: `author={Jiménez, Carlos E. and ...}`, key renamed `jimenez2024swebench`, arXiv:2310.06770.
3. AgentBench key renamed `liu2024agentbench` (lead author Liu, Xiao).
4. Added uncertainty-across-seeds sentence to §4 estimation discussion.
5. Added illustrative-weights/hard-gates comment to scorecard.py.
All 5 PDFs rebuilt clean; code rerun passes.
