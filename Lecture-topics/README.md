# Lecture Topics — DLS-2026

Topical lecture kits: each topic directory gathers the **slides**, the
**lecture notes**, the **reading list**, and the **papers** for one lecture
block. Topic kits complement the chronological `Lecture-notes/Week-N/`
directories (which follow the calendar).

| Topic | Slides | Notes | Papers | Status |
|---|---|---|---|---|
| [`Scaling-Laws/`](Scaling-Laws/) | `scaling_laws_lecture.pdf` (31 pp) | `scaling_laws_notes.pdf` (43 pp companion, one section per slide) + `scaling_laws_explained.pdf` (52 pp extended) | 43 papers in 8 sections (before-LMs → Kaplan/Chinchilla → refinements → architecture knobs → loss-to-capability → why power laws → ladders → frontier case studies) | complete |
| [`Post-Training/`](Post-Training/) | `posttraining_tutorial.pdf` (32 pp, "Post-Training, Visually") | notes materializing | 77 papers in 9 sections (SFT → preference optimization/RLHF/DPO → RL with verifiable rewards → reasoning/test-time compute → long context → safety → tools & agents → open recipes → evaluation) | slides + papers done, notes pending |
| [`Sparsity-and-Precision/`](Sparsity-and-Precision/) | `sparsity_precision_tutorial.pdf` (33 pp, "Sparsity and Precision, Visually") | LaTeX source included (`sparsity_precision_tutorial.tex`) | 104 papers in 9 sections (cost model → MoE → pruning → activation/attention sparsity → number formats → low-precision training → precision scaling laws → inference quantization → practice) | complete |
| [`Pre-training-Case-Study/`](Pre-training-Case-Study/) | `pretraining_tutorial.pdf` (34 pp, "Frontier Pre-training, Visually" — MAI-Base-2) | `Frontier-Pre-training-at-Microsoft-AI.pdf` (13 pp, the MAI report it visualizes) + `deepseek2024v3.pdf` (53 pp DeepSeek-V3 technical report) | 51 papers in 9 sections (overview → the model → scaling ladders → hyperparameter laws → MoE systems → Muon → numerics/stability → data & vision → comparators) | complete |
| [`Transformer/`](Transformer/) | see `Lecture-notes/` Week-2/3 decks | — | 56 papers in the companion reading list (architecture, attention, training) | papers kit staged |

## Reading lists

Each topic's papers directory contains `README.md` (annotated reading list,
grouped by lecture section), `reading_list.csv` and `reading_list.bib` (same
list as data), and one subdirectory per lecture section holding the PDFs.
Paper sets are also available as zips in the Dropbox student download area.

## Notes

- Post-Training lecture notes are still materializing; the slides and the full
  reading list are ready and staged here.
- The Pre-training case study is a visual companion to Microsoft AI's
  *Frontier Pre-training at Microsoft AI* (2026) report on MAI-Base-2 —
  scaling ladders, hyperparameter laws, MoE systems, Muon, stability, data —
  with DeepSeek-V3 as the second frontier case.
