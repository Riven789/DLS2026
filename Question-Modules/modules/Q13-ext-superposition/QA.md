# Q13 QA and build record

- **Author:** Paradise (module drafting agent)
- **Topic:** How do models represent and store knowledge, and use it to answer? (survey Q13, week 4, cluster C)
- **Status:** drafted; technical review (Ollie) and teaching review (Kukla) pending.
- **Video element:** skipped (parked repo-wide).

## Author self-check

- [x] All five baseline PDFs build locally with `pdflatex` + `bibtex` (note) — zero LaTeX errors, zero BibTeX warnings.
- [x] Page/frame counts within contract: note 5 pages; deck 10 frames (title + 9 content); discussion exactly 5 questions; projects 4 suggestions; quiz 1 question + answer.
- [x] Every `\cite` key in `note.tex` resolves in `bibliography.bib` (10/10 entries used, no undefined-citation warnings in `note.log`).
- [x] All citations verified real via web lookup before use (see below).
- [x] `code/superposition_demo.py` runs with repo `.venv` python (NumPy 2.5.4), exit code 0, all assertions pass.
- [x] Claims hedged: localization results presented as model/scale-specific evidence, not law; editing framed as first approximation with open problems.

## Citation verification (all real, checked 2026-10-10)

| Key | Paper | Venue / verification |
|---|---|---|
| `geva2021transformer` | Transformer Feed-Forward Layers Are Key-Value Memories (Geva, Schuster, Berant, Levy) | EMNLP 2021, pp. 5484–5495, aclanthology.org/2021.emnlp-main.446 |
| `meng2022locating` | Locating and Editing Factual Associations in GPT (Meng, Bau, Andonian, Belinkov) | NeurIPS 2022, rome.baulab.info |
| `elhage2022toy` | Toy Models of Superposition (Elhage et al.) | Transformer Circuits Thread, Sept 2022, arXiv:2209.10652 |
| `meng2023memit` | Mass-Editing Memory in a Transformer (Meng, Sen Sharma, Andonian, Belinkov, Bau) | ICLR 2023, memit.baulab.info |
| `belinkov2019analysis` | Analysis Methods in Neural Language Processing: A Survey (Belinkov, Glass) | TACL 7:49–72, 2019 |
| `hupkes2018visualisation` | Visualisation and 'Diagnostic Classifiers' Reveal how Recurrent and Recursive Neural Networks Process Hierarchical Structure (Hupkes, Veldhoen, Zuidema) | JAIR 61:747–785, 2018 |
| `petroni2019language` | Language Models as Knowledge Bases? (Petroni et al.) | EMNLP 2019, arXiv:1909.01066 |
| `elhage2021mathematical` | A Mathematical Framework for Transformer Circuits (Elhage, Nanda, Olsson, et al.) | Transformer Circuits Thread, Dec 2021 |
| `sukhbaatar2015end` | End-to-End Memory Networks (Sukhbaatar, Weston, Fergus) | NeurIPS 2015 (cited by Geva et al. as the KV-memory lineage) |
| `olsson2022context` | In-Context Learning and Induction Heads (Olsson et al.) | Transformer Circuits Thread, 2022 |

Unverified citations: **none**. (Nanda 2022 "Foundations of Superposition" was considered as optional but not cited, so it is not listed in the bibliography.)

## Build commands

```
cd modules/Q13-knowledge-representation
pdflatex note.tex && bibtex note && pdflatex note.tex && pdflatex note.tex
pdflatex deck.tex && pdflatex deck.tex        # ×2 for stable frame numbering
pdflatex discussion.tex && pdflatex discussion.tex
pdflatex projects.tex && pdflatex projects.tex
pdflatex quiz.tex && pdflatex quiz.tex
.venv/bin/python code/superposition_demo.py   # from repo root
```

## Real code output

Command: `/home/stevens/ai-course-2026/.venv/bin/python modules/Q13-knowledge-representation/code/superposition_demo.py`
(NumPy 2.5.4; exit code 0)

```
Q13 superposition demo — features in superposition (Elhage et al. 2022)

Case 1: d slightly < N, sparse features
  N=200 features in d=160 dims | sparsity=0.98 (P(feature off))
  feature bank: mean |cos|=0.063, max |cos|=0.330 (orthogonal would be 0)
  readout: true-positive rate=1.000, false-positive (crosstalk) rate=0.017
  => PASS: sparse features recovered despite d<N

Case 2: d << N, sparse features
  N=200 features in d=20 dims | sparsity=0.98 (P(feature off))
  feature bank: mean |cos|=0.180, max |cos|=0.760 (orthogonal would be 0)
  readout: true-positive rate=0.930, false-positive (crosstalk) rate=0.199
  => FAIL: interference dominates

Case 3: d slightly < N, dense features
  N=200 features in d=160 dims | sparsity=0.20 (P(feature off))
  feature bank: mean |cos|=0.063, max |cos|=0.326 (orthogonal would be 0)
  readout: true-positive rate=0.760, false-positive (crosstalk) rate=0.416
  => FAIL: interference dominates

Control: orthonormal basis (d == N)
  readout: true-positive rate=1.000, false-positive rate=0.000
  => PASS: no interference when directions are orthogonal

ALL CHECKS PASSED:
  * sparse features survive d<N (superposition buys capacity),
  * too few dimensions (d<<N) or dense co-activation breaks readout
    (interference is the price),
  * orthogonal directions read out exactly (the ideal superposition
    approximates at high dimension).

Moral: sparsity is the resource that lets a layer store more
features than dimensions; the ReLU readout is the interference filter.
```

## Review

- Technical/evidence review — Ollie: pending.
- Teaching/release review — Kukla: pending.
