# Sparsity & Precision

Making models cheaper to train and serve by using fewer active parameters and fewer bits.

## Materials
- **[Sparsity-and-Precision-Lecture.pdf](Sparsity-and-Precision-Lecture.pdf)** — the class lecture deck.
- **[papers/](papers/)** — 113-paper reading list, grouped by subtopic.

> *Written lecture notes for this topic are still being prepared and will be added here.*

## What it covers
The cost model → mixture-of-experts → weight pruning → activation and attention sparsity →
number formats and outliers → low-precision training → scaling laws for precision →
quantization for inference → models and systems in practice.

## Reading list
See **[papers/README.md](papers/README.md)**. To download the PDFs:
```bash
cd papers && bash fetch_papers.sh
```
