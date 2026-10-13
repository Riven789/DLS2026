# Kukla peer review — Q24 (Q24-agent-evaluation)

**Reviewer:** Kukla (m1)  
**Date:** 2026-10-10  
**Method:** fresh `latexmk` build of all 5 .tex (PASS) + `python3` run of code (PASS, numpy env) + LLM-assisted content read (argo:claude-opus-4.8) grounded in note/code/bib/quiz, with reviewer verification of flagged citations.

---

# Review of Module Q24

## Technical Accuracy

The conceptual content is largely sound and well-calibrated for a graduate audience. The trajectory-vs-response distinction, task-contract framing, and the completion/conditional-success/intervention/constraint-violation metric taxonomy are all correct and well-stated. The contamination/holdout discussion is accurate.

Minor concerns:
- "SWE-bench operationalizes part of this setting using real GitHub issues and repository test suites" — accurate. Good.
- "WebArena evaluates multi-step web interaction in a controlled environment" — accurate.
- The claim about LLM-as-judge needing validation against humans and defense against "prompt injection or style-based gaming" is correct and well-grounded (style/length bias in judges is a documented phenomenon).
- No overtly false technical claims found.

## Citation Validity — the weakest aspect

Several citation problems need fixing:

1. **`li2023swebench` author field is garbled.** It reads `author={Li, Carlos E. Jimenez and others}`. The actual first author is Carlos E. Jiménez. The BibTeX key `li...` and the `Li, Carlos` construction are both wrong — this will render as "Li, Carlos E. Jimenez" which misattributes the surname. Fix to `author={Jimenez, Carlos E. and others}` and rename the key (e.g., `jimenez2024swebench`).

2. **`li2024agentbench` key/author mismatch.** The key is `li2024agentbench` but the lead author is correctly listed as `Liu, Xiao`. The paper is real (arXiv:2308.03688) and the author is right; only the key is misleading. Low severity but should be `liu2024agentbench`.

3. **`sheng2021human` is the most serious problem.** The title "The Human-Computer Interaction Evaluation of Chatbots," the author "Sheng, Emily," and arXiv:2105.03919 do not form a verifiable, plausibly-matching record. Emily Sheng's known work is on bias/fairness in NLG, not a chatbot-HCI-evaluation paper under this title/ID. This citation appears **fabricated or misattributed**, and it is doing real work in the text (supporting "human interaction quality has multiple dimensions"). This claim could instead be supported by a genuine source — e.g., the Chatbot Arena / LMSYS work, or established dialogue-evaluation surveys (e.g., Finch & Choi dialogue evaluation survey, or the ACUTE-Eval / dialogue-system eval literature). **Replace this citation before approval.**

4. `zhou2023webarena` checks out (Zhou et al., arXiv:2307.13854).

## Pedagogy

Strong. The module directly answers the student question ("How do we judge whether an agent is good?") and reframes it productively around task contracts. The Agent A/B worked example is the clearest teaching device and is grad-appropriate. The evaluation-card checklist is practical and transferable. Scope is well-controlled. The quiz targets exactly the central misconception. This is the **strongest aspect of the module.**

One pedagogical gap: the note never defines or references statistical reporting of uncertainty (confidence intervals / variance across seeds), despite emphasizing "distributions, not just means." A sentence on reporting variability across runs would strengthen the "estimation" framing in §4.

## Code

The toy scorecard correctly illustrates the intended point: two agents with *equal* completion rates are separated once cost and violations enter a multi-objective utility. The assertions (equal success, `a > b`) are conceptually valid demonstrations.

One caveat worth noting (not blocking): the chosen weights (`0.02*cost`, `5*violations`) are arbitrary and the `utility` function is linear and additive. This is fine for a toy, but a one-line comment stating these weights are illustrative — and that in reality constraint violations are often treated as hard gates, not trade-off terms — would prevent students from inferring that safety violations are simply "priced in." Given §3 argues Agent B "fails safely," a linear penalty slightly undersells the point.

## Strongest / Weakest
- **Strongest:** pedagogy — the task-contract framing plus the Agent A/B example cleanly answer the question and expose the target misconception.
- **Weakest:** citation validity — one likely-fabricated reference (`sheng2021human`) and a garbled author/key in the SWE-bench entry.

VERDICT: REQUEST CHANGES
1. Replace `sheng2021human` with a verifiable source for the multi-dimensionality of chatbot/dialogue evaluation (e.g., Chatbot Arena / LMSYS, or an established dialogue-evaluation survey), and verify the arXiv ID matches the actual record.
2. Fix the SWE-bench entry: change `author={Li, Carlos E. Jimenez and others}` to `author={Jimenez, Carlos E. and others}` and rename the key (e.g., `jimenez2024swebench`), updating the `\cite` call in §1.
3. Rename the AgentBench key `li2024agentbench` to `liu2024agentbench` to match its correct lead author (Liu, Xiao).
4. Add one sentence in §3 or §4 on reporting uncertainty across seeds/runs (confidence intervals or variance), to match the "distributions, not just means" claim.
5. Add a one-line comment in `scorecard.py` noting the utility weights are illustrative and that safety violations are often better modeled as hard gates than linear trade-offs.

## Re-verification after Paradise revision — Kukla (2026-10-11) — APPROVE

**Verdict: APPROVE.** Re-verified all 5 requested changes against `main` from a
clean worktree (TeX Live 2021, full pdflatex→bibtex→pdflatex×2 + xelatex + code):

1. **Fabricated `sheng2021human` — RESOLVED.** Removed; replaced with the verified
   `chiang2024chatbotarena` (Chiang et al., *Chatbot Arena*, arXiv:2403.04132,
   2024 — real, authors correct). Confirmed `sheng2021human` appears 0× in both
   bib and note. ✓
2. **SWE-bench entry — RESOLVED (+1 defect I fixed on review).** Key is now
   `jimenez2024swebench` and renders "Carlos E. Jiménez" as lead author. However
   Paradise's revision left the bib entry **malformed** (duplicate
   `title/journal/year/url` fields after the entry already closed) and the author
   list garbled (`Lierler, Shunyu` — wrong surname; should be `Yao, Shunyu`; Pei
   dropped). I corrected the entry to the verified author list (Jiménez, Yang,
   Wettig, Yao, Pei, Press, Narasimhan) and removed the duplicate fields.
   Rebuilt: `note.bbl` now renders "Carlos E. Jiménez, John Yang, Alexander
   Wettig, Shunyu Yao, Kexin Pei, Ofir…". ✓
3. **AgentBench key — RESOLVED.** Now `liu2024agentbench` (Liu, Xiao;
   arXiv:2308.03688), cited consistently in note.tex. ✓
4. **Seed/uncertainty sentence — RESOLVED.** §4 now reads "Report uncertainty
   across seeds and evaluation reruns (variance or confidence intervals), not only
   point estimates…". ✓
5. **Scorecard illustrative-weights comment — RESOLVED.** `code/scorecard.py` has
   the NOTE block: weights illustrative, violations usually hard gates. Code runs:
   `PASS: equal completion=70; utility fast-safe=68.6 > costly-risky=31.0` (exit 0). ✓

Build after my bib fix: note 3 pp, deck 9 frames, discussion/projects/quiz clean,
0 undefined citations. **Q24 approved from the Kukla review lane.**
