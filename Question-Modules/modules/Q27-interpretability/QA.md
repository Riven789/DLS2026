# Q27 QA and review record

- **Author:** Paradise
- **Topic (survey Q27):** Interpretability: how can inspecting weights reveal a model's tendencies? Do models revert to pre-training traits after deployment?
- **Status:** draft complete; technical/evidence review (Ollie) and teaching/release review (Kukla) pending.

## Author self-check

- [x] Five baseline PDFs build locally (note: 5 pages; deck: 10 pages = title + 9 frames).
- [x] All `\cite` keys in `note.tex` resolve in `bibliography.bib` (13/13; none missing, none uncited; BibTeX warnings: 0).
- [x] Discussion has exactly 5 questions; quiz has 1 question + answer; projects has 4 suggestions.
- [x] Code runs with repo `.venv` python (NumPy 2.5.4), exits 0, all assertions pass. Real output below.
- [x] Every citation verified against primary source (publisher page, arXiv, ACL Anthology, or Transformer Circuits Thread) before writing. See citation-verification table.
- [x] Video element skipped (parked per module contract).

## Build commands

```
pdflatex note.tex && bibtex note && pdflatex note.tex && pdflatex note.tex
pdflatex deck.tex && pdflatex deck.tex
pdflatex discussion.tex
pdflatex projects.tex
pdflatex quiz.tex
```

All five exit 0; `grep -c '^!' *.log` = 0 errors; note.log has no unresolved-citation/reference warnings.

## Code: `code/refusal_direction.py`

NumPy-only toy of Arditi et al. (2024). A synthetic model has an orthogonal capability direction and refusal direction; a difference-in-means probe recovers the refusal direction; directional ablation removes refusal while task accuracy survives; a random-direction control changes nothing. Pass/fail assertions printed at the end.

Run: `/home/stevens/ai-course-2026/.venv/bin/python code/refusal_direction.py`

### Real output (2026-10-10, repo .venv python, NumPy 2.5.4)

```
Q27 toy: refusal direction (Arditi et al. 2024, in miniature)
----------------------------------------------------------------
probe recovery: |cos(r_hat, r_true)|      = 0.995
probe leakage:  |cos(r_hat, c_capability)| = 0.042

condition                    refuse(harmful) refuse(harmless)  task acc
baseline                               99.8%             0.8%    100.0%
ablate refusal dir r_hat                0.0%             0.0%    100.0%
ablate random dir (ctrl)              100.0%             0.8%    100.0%

[PASS] probe recovers refusal direction (|cos| > 0.95)
[PASS] probe direction spares capability (|cos| < 0.10)
[PASS] baseline: harmful requests refused (rate > 0.95)
[PASS] baseline: harmless requests not refused (rate < 0.05)
[PASS] ablation kills refusal (harmful refusal rate < 0.05)
[PASS] task ability survives refusal ablation (acc > 0.95)
[PASS] random-direction control changes nothing (refusal > 0.95)

ALL CHECKS PASSED
```

## Citation verification

| Key | Claim in module | Verified against |
|---|---|---|
| bricken2023monosemanticity | SAEs extract monosemantic features from superposition; one-layer transformer, 512-neuron MLP, expansion factors | transformer-circuits.pub/2023/monosemantic-features (Oct 4, 2023; authors incl. Bricken, Templeton, Batson, Chen, ... Olah) |
| olsson2022context | Induction heads implement [A][B]…[A]→[B]; two-head circuit; phase change coincides with in-context learning | arXiv:2209.11895 and transformer-circuits.pub/2022/in-context-learning-and-induction-heads |
| elhage2022toy | Superposition: models store more features than neurons; polysemanticity | arXiv:2209.10652; transformer-circuits.pub/2022/toy_model (Sept 14, 2022) |
| geva2021transformer | FFN layers as key–value memories | ACL Anthology 2021.emnlp-main.446, pp. 5484–5495 (Geva, Schuster, Berant, Levy) |
| meng2022locating | Causal tracing locates factual associations in mid-layer MLP; ROME rank-one edit | arXiv:2202.05262, NeurIPS 2022 (Meng, Bau, Andonian, Belinkov); rome.baulab.info |
| belinkov2022probing | Probing pitfalls; representable ≠ used; random-feature control baselines | Computational Linguistics 48(1), 2022; arXiv:2102.12452 |
| qi2024fine | 10 adversarial examples jailbreak GPT-3.5 Turbo (<$0.20); benign fine-tuning also degrades safety | ICLR 2024 proceedings (Qi, Zeng, Xie, Chen, Jia, Mittal, Henderson); arXiv:2310.03693 |
| arditi2024refusal | Refusal mediated by single direction across 13 open models ≤72B; directional ablation; rank-one weight orthogonalization | NeurIPS 2024 proceedings f5454485...; arXiv:2406.11717 (Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda) |
| greenblatt2024alignment | Claude 3 Opus strategically complies in "training" to preserve out-of-training harmlessness; 12% vs ~never compliance gap | arXiv:2412.14093 |
| wei2023jailbroken | Competing objectives / mismatched generalization taxonomy | arXiv:2307.02483, NeurIPS 2023 (Wei, Haghtalab, Steinhardt) |
| zou2023universal | Gradient-based adversarial suffixes transfer across aligned models | arXiv:2307.15043; ICML 2023 (Zou, Wang, Kolter, Fredrikson) |
| ouyang2022training | RLHF/InstructGPT: thin post-training over base model | NeurIPS 2022 proceedings b1efde53... |
| templeton2024scaling | SAEs scale to Claude 3 Sonnet; millions of features; safety-relevant features steerable | transformer-circuits.pub/2024/scaling-monosemanticity/ (Templeton, Conerly, Marcus, ... Henighan) |

Notes on verification:
- Greenblatt et al. 2024 numbers: 12% compliance for free ("training") users vs ~0% for paid users in the prompted helpful-only setting; alignment-faking reasoning rate rose to 78% under RL training. The module text says "a meaningful rate" and "almost never" rather than exact figures, and flags that the setup made the effect easy to elicit and that later 25-model replication finds it in a minority of frontier models.
- The "later replication across 25 models" claim comes from the follow-up "Why Do Some Language Models Fake Alignment While Others Don't?" (arXiv:2506.18032) — referenced qualitatively in note/deck/discussion text without a formal \cite (it is Q28 territory); flagged here for reviewer attention.
- Olsson et al. author list includes "Tilecz, Larry" per the Transformer Circuits page — this is the published spelling (also seen as "Tillecz"); kept as published.
- A search result surfaced "arXiv:2605.29358" for Templeton et al.; the canonical source is the Transformer Circuits Thread page (May 2024), which is what bibliography.bib cites. No arXiv ID is asserted in the .bib entry.

## Known limitations (for reviewers)

- The toy in `code/refusal_direction.py` is deliberately minimal: orthogonal ground-truth directions, Gaussian noise, difference-in-means recovery is near-perfect by construction. It demonstrates the *logic* of directional ablation with controls, not empirical difficulty. The note says "toy" explicitly.
- The note is at the 5-page upper bound; further additions will need trades.
- The `note.log` shows two benign `h' float specifier changed to 'ht'` warnings (table placement), no errors.
