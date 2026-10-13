# Q11 QA and review record

- **Author:** Paradise (drafting agent)
- **Topic (survey Q11):** Why do LLMs hallucinate — mechanistic and statistical accounts of fluent falsehood.
- **Status:** initial draft complete; technical/evidence review (Ollie) and teaching/release review (Kukla) pending.
- **Video element:** parked per module contract; tracked here so it is not silently dropped. To do when un-parked: companion video + alignment record.

## Author self-check

- [x] All five PDFs build clean (`pdflatex` + `bibtex`, exit 0, no undefined-citation/reference warnings): note 5 pp, deck 10 slides, discussion 1 p, projects 1 p, quiz 1 p.
- [x] Every `\cite` key in `note.tex` resolves in `bibliography.bib` (10/10 keys in `note.bbl`); all 10 entries verified against ACL Anthology / ACM DL / NeurIPS proceedings / arXiv records before citing (see below).
- [x] Code runs under repo `.venv` Python 3.14.7 / NumPy 2.5.4, seeded, exits 0, all 8 assertions pass.
- [x] Discussion = exactly 5 questions; projects = 4 suggestions; quiz = 1 question + answer.
- [x] Toy-model claims hedged in note + code docstring: the demo isolates one statistical mechanism (smoothing/shrinkage), it is not a Transformer and does not implement the Kalai–Vempala theorem.
- [x] Every printed number in the demo is computed from the same generated corpus (Ollie's Q20 lesson applied).

## Citation verification record (checked 2026-10-10, before citing)

| Key | Venue, verified via |
|---|---|
| ji2023survey | ACM Computing Surveys 55(12), Art. 248, Dec 2023 — dl.acm.org/doi/10.1145/3571730 |
| kalai2024calibrated | STOC 2024, pp. 160–171 — dl.acm.org/doi/10.1145/3618260.3649777 (arXiv:2311.14648) |
| lin2022truthfulqa | ACL 2022, pp. 3214–3252 — aclanthology.org/2022.acl-long.229 |
| gekhman2024finetuning | EMNLP 2024, pp. 7765–7784 — aclanthology.org/2024.emnlp-main.444 (arXiv:2405.05904) |
| shuster2021retrieval | Findings EMNLP 2021, pp. 3784–3803 — aclanthology.org/2021.findings-emnlp.320 |
| lewis2020rag | NeurIPS 2020 — proceedings.neurips.cc, paper 6b493230 |
| li2023halueval | EMNLP 2023, pp. 6449–6464 — aclanthology.org/2023.emnlp-main.397 (arXiv:2305.11747) |
| meng2022rome | NeurIPS 2022 — openreview.net/forum?id=-h6WAS6eE4 (verified NeurIPS, not ICLR; the ICLR 2023 paper is the follow-up MEMIT) |
| mallen2023trust | ACL 2023, pp. 9802–9822 — aclanthology.org/2023.acl-long.546 |
| kandpal2023longtail | ICML 2023 — proceedings.mlr.press/v202/kandpal23a |

Unverified citations: **none**.

## Code run (real output, repo .venv)

Command: `/home/stevens/ai-course-2026/.venv/bin/python code/hallucination_toy.py` (run 2026-10-10, exit 0)

```
========================================================================
Q11 toy: a smoothed count 'LM' on a 100% truthful factual corpus
========================================================================
subjects: 600 (Zipf a=1.3) | fact mentions: 4000 | discourse mentions: 6000
smoothing alpha=12.0 (generalization pressure) | novelty floor gamma=0.05 over 720 open-vocab objects

Frequency vs. accuracy (closed-book QA, greedy decoding):
 train count  #facts    acc  med P(truth)
           0     234   0.00         0.000
           1     160   0.00         0.073
           2      75   0.00         0.136
         3-5      68   0.44         0.191
        6-15      36   1.00         0.382
         16+      27   1.00         0.720

ASCII accuracy curve (each '#' ~ 5%):
  count     0 |                       0.00
  count     1 |                       0.00
  count     2 |                       0.00
  count   3-5 | #########             0.44
  count  6-15 | ####################  1.00
  count   16+ | ####################  1.00

argmax-correct boundary (greedy accuracy = 1.0):      count >= 6-15
reliability boundary  (median P(truth) >= 0.5):        count >= 16+
  -> the model's argmax turns correct BEFORE its own confidence
     crosses 0.5: sampling hallucinates on a strictly larger set
     of facts than greedy decoding.

most popular entity: ENT-001 (unigram mass 0.287); all greedy errors misattribute TO IT
representative subjects (rank ~ attestation):
  subject  n_train     true  greedy pred  P(true)  P(pred)  verdict
 SUBJ-001     1143  ENT-001      ENT-001    0.943    0.943  correct
 SUBJ-008       88  ENT-008      ENT-008    0.838    0.838  correct
 SUBJ-030        8  ENT-030      ENT-030    0.382    0.382  correct
 SUBJ-060        5  ENT-060      ENT-060    0.280    0.280  correct
 SUBJ-100        4  ENT-100      ENT-100    0.238    0.238  correct
 SUBJ-150        3  ENT-150      ENT-001    0.190    0.218  MISATTRIBUTED
 SUBJ-400        1  ENT-400      ENT-001    0.073    0.252  MISATTRIBUTED
 SUBJ-550        0  ENT-550      ENT-001    0.000    0.273  MISATTRIBUTED

Good-Turing / Kalai-Vempala flavor (facts = subject->answer pairs):
  monofact rate (facts stated exactly once / mentions): 0.0400
  empirical missing-fact rate (fresh mentions):          0.0312
  true missing mass (population, reference):             0.0328
  -> a calibrated model cannot know unattested facts; the
     monofact rate estimates how often that happens.

Sampling mode (3000 samples from P(o|s), stratified):
  correct answers:          797 ( 26.6%)
  misattributions:         2181 ( 72.7%)
  fabrications (GHOSTs):     22 (  0.7%)
    e.g. bucket [0] SUBJ-175 -> GHOST-073 (never attested as any answer)
    e.g. bucket [0] SUBJ-188 -> GHOST-066 (never attested as any answer)
    e.g. bucket [0] SUBJ-188 -> GHOST-107 (never attested as any answer)

PASS: unattested facts are unanswerable (bucket-0 greedy accuracy == 0)
PASS: rare facts are overridden by plausible continuations (bucket-1 greedy accuracy <= 0.1)
PASS: greedy accuracy is non-decreasing across frequency buckets
PASS: high-frequency facts are all correct (bucket 16+ accuracy == 1.0)
PASS: Good-Turing: |monofact rate - empirical missing rate| <= 0.05
PASS: greedy errors exist and are misattributions to the single most popular entity (the plausible continuation)
PASS: sampling fabricates: >=1 unattested object sampled, and fabrications are rarer than misattributions
PASS: confidence tracks attestation: median P(truth) strictly increases from bucket 1 to bucket 16+

ALL 8 ASSERTIONS PASSED
```

## Notes for reviewers

- The demo's "fabrication" channel is an explicit `gamma`-floor over unattested objects, standing in for a real model's ability to compose novel-but-well-formed entities; it is a modeling choice, not an emergent property, and is labeled as such in the code docstring.
- The Kalai–Vempala discussion is deliberately hedged in the note (arbitrary facts only; ideal data; pre-training vs.\ post-training) — reviewers should check those hedges survive any edits.
- ROME is cited as NeurIPS 2022 (commonly miscited as ICLR 2023; ICLR 2023 is the MEMIT follow-up).

## Review log

- Technical/evidence review (Ollie): pending.
- Teaching/release review (Kukla): pending.
