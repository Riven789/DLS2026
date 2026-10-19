# Post-training

Turning a pretrained base model into a useful, aligned, tool-using assistant.

## Materials
- **[Post-training-Lecture.pdf](Post-training-Lecture.pdf)** — the class lecture deck.
- **[papers/](papers/)** — 90-paper reading list, grouped by subtopic.

> *Written lecture notes for this topic are still being prepared and will be added here.*

## What it covers
SFT and instruction data → preference optimization (RLHF, reward models, PPO, DPO) →
RL with verifiable rewards and RL systems → reasoning, test-time compute, R1-style
distillation and token budgets → long context → safety and alignment → tools and agents →
open recipes and on-policy distillation → evaluation.

## Reading list
See **[papers/README.md](papers/README.md)**. To download the PDFs:
```bash
cd papers && bash fetch_papers.sh
```
