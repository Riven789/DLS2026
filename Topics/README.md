# Topics — DLS-2026

Deep-dive topic sections that go beyond the week-by-week lecture notes. Each topic
folder holds the **slide deck**, any **lecture notes**, a **visual tutorial**, and a
**companion reading list** of primary papers grouped by subtopic.

> **Some lecture notes are still being written** and will be added to the relevant
> folders as they are finished. The decks, tutorials, and reading lists below are final.

## Sections

| # | Topic | What's inside |
|---|-------|---------------|
| 1 | [**Scaling Laws**](01-Scaling-Laws/) | Lecture deck, lecture notes, the long-form "Scaling Laws, Explained" deck, and a 46-paper reading list (Kaplan → Chinchilla → refinements → frontier case studies). |
| 2 | [**Pretraining & the MAI Case Study**](02-Pretraining-and-the-MAI-Case-Study/) | Frontier-pretraining lecture + notes, the **Microsoft AI (MAI) frontier-pretraining case study**, the DeepSeek-V3 technical report, and a 60-paper reading list. |
| 3 | [**Post-training**](03-Post-training/) | Post-training lecture (SFT, RLHF/DPO, RLVR, reasoning & test-time compute, long context, safety, tools/agents, evaluation) and a 90-paper reading list. |
| 4 | [**Sparsity & Precision**](04-Sparsity-and-Precision/) | Sparsity & precision lecture (MoE, pruning, activation/attention sparsity, number formats, low-precision training, quantization for inference) and a 113-paper reading list. |

## How the reading lists work

Each topic has a `papers/` folder containing:

- `README.md` — the annotated reading list, grouped by subtopic and keyed to the slides.
- `reading_list.csv` / `reading_list.bib` — the same list as data (for tooling / citation).
- `fetch_papers.sh` — downloads every arXiv PDF in the list into one folder per subtopic.

To pull the PDFs locally:

```bash
cd Topics/<topic>/papers
bash fetch_papers.sh       # needs curl + zip; respects arXiv's rate limit (a few minutes)
```

The downloaded PDF collections are large (120–210 MB each) and are **not committed to
this repo**. Pre-built zips are available in the course **Dropbox → Student Downloads**;
or just run `fetch_papers.sh` to build them yourself. A handful of non-arXiv entries
(listed with a DOI/URL) must be opened in a browser.

---
*Student roster data is intentionally NOT included anywhere in this directory.*
