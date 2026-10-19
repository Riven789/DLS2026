# Pretraining & the MAI Case Study

What frontier-scale pretraining actually involves — data, tokenization, architecture,
optimization, systems, and stability — grounded in a real frontier run.

## Materials
- **[Frontier-Pretraining-Lecture.pdf](Frontier-Pretraining-Lecture.pdf)** — the class lecture deck.
- **[Frontier-Pretraining-Notes.pdf](Frontier-Pretraining-Notes.pdf)** — written lecture notes.
- **[MAI-Pretraining-Case-Study.pdf](MAI-Pretraining-Case-Study.pdf)** — the **Microsoft AI (MAI) frontier-pretraining case study**.
- **[DeepSeek-V3-Technical-Report.pdf](DeepSeek-V3-Technical-Report.pdf)** — a second, openly documented frontier run to compare against.
- **[papers/](papers/)** — 60-paper reading list, grouped by subtopic.

## What it covers
Data collection/cleaning/mixing → tokenization → architecture choices at scale →
optimizers and learning-rate schedules (incl. Muon) → distributed training and systems →
loss spikes and stability → the MAI and DeepSeek-V3 runs as end-to-end case studies.

## Reading list
See **[papers/README.md](papers/README.md)**. To download the PDFs:
```bash
cd papers && bash fetch_papers.sh
```
