# Q26 QA and review record

- **Author:** Paradise
- **Survey source:** Section-2 question Q26 — "How does fine-tuning change a model's behavior? Are there post-training pipelines that adapt well to new domains? How do we tune systems online?"
- **Status:** drafted (this file is the author self-check); peer reviews by Ollie + Kukla pending.
- **Video element:** parked per Rick's call (content and review are the critical path; companions generated after content review).

## Author self-check

- [x] `note.pdf` builds clean (`pdflatex + bibtex + 2x pdflatex`), **4 pages** (contract: 3–5), zero unresolved `??`, 0 Overfull boxes.
- [x] `deck.pdf` builds clean, **10 slides** (contract: 5–10), 16:9, 0 Overfull boxes.
- [x] `discussion.pdf` builds clean, 1 page, **exactly 5 questions**.
- [x] `projects.pdf` builds clean, 1 page, 4 suggestions (contract: 3–4).
- [x] `quiz.pdf` builds clean, 1 page, 1 question + answer.
- [x] Every `\cite` in note.tex resolves to a `bibliography.bib` entry (9 keys cited, 9 entries; comm check = 0 missing). All 9 references verified real against arXiv/ACL/PNAS/NeurIPS/ICLR records (see citation log below).
- [x] `code/lora_vs_full.py` runs with the repo `.venv` python, exit 0, 10/10 assertions pass, output demonstrates the note's central claim (LoRA preserves task A better than full FT at some cost on B).
- [x] Demo honesty: synthetic two-task linear setup, clearly scoped in the module docstring — it is a toy demonstration of the *mechanism* (rank-constrained update ⇒ retention), not an LLM experiment. Both methods trained to convergence (decayed LR + early stopping) so the contrast is not an undertraining artifact. Data has learnable structure (deterministic linear rules), not random labels.

## Build / run commands (for reviewers)

```bash
cd modules/Q26-finetuning-adaptation
pdflatex note.tex && bibtex note && pdflatex note.tex && pdflatex note.tex
pdflatex deck.tex && pdflatex discussion.tex && pdflatex projects.tex && pdflatex quiz.tex
/home/stevens/ai-course-2026/.venv/bin/python code/lora_vs_full.py   # rc=0 expected
```

## Real code output (2026-10-10, repo .venv python, deterministic seeds)

```
== Task A: train base linear layer W (the 'pretrained' model) ==
    [A-train] converged: step 20000, train loss 0.0834
base accuracy on A (held-out): 0.883

== Task B: adapt the task-A model to a shifted, partly noisy task ==
task-B training set: 1024 samples, 107 labels flipped (10% noise)
task-A model on task B, before adapting: 0.436

-- (a) full fine-tune (all 64x64 = 4096 weights updated) --
    [full-B] converged: step 8534, train loss 0.6992
-- (b) LoRA rank-4 (W frozen; (64x4)+(4x64) = 512 factors trained) --
    [lora-B] converged: step 20000, train loss 1.8978

== Results (held-out accuracy; chance = 0.125) ==
                          task A (retention)  task B (new task)
base (task-A model):                  0.883             0.436
full fine-tune:                       0.400             0.740
LoRA rank-4:                          0.504             0.688

forgetting on A (drop):    full = 0.482   LoRA = 0.379
plasticity on B (gain):    full = +0.305   LoRA = +0.252

noise memorization (accuracy on noisy train labels minus clean test):
  full fine-tune: +0.115   (fits the flipped labels)
  LoRA rank-4:    -0.002   (rank-4 cannot memorize i.i.d. label noise)

update size ||W - W_A||_F:  full = 20.144   LoRA = 19.514
update rank (numerical):    full = 7   LoRA = 4  (LoRA bounds the RANK, not the norm)
trainable parameters:       full = 4096   LoRA = 512 (12% of full)

== Assertions ==
[PASS] base model learned task A (acc > 0.80)
[PASS] task A does not transfer to B (base < 0.60)
[PASS] full fine-tune adapted to B (gain > 0.15)
[PASS] full fine-tune catastrophically forgot A (drop > 0.25)
[PASS] LoRA retains task A far better than full fine-tune (gap > 0.08)
[PASS] LoRA forgets task A less than full fine-tune
[PASS] LoRA still adapted to B (gain > 0.10)
[PASS] ...at a cost on B (below full fine-tune's B)
[PASS] full FT memorized noise (B train-test gap > 0.05)
[PASS] LoRA memorized less noise (smaller gap)

(total wall-clock: 24.8s)

RESULT: at matched convergence, the rank-4 LoRA update preserves the base task
better than full fine-tuning at some cost on the new task -- the bounded-update
claim, demonstrated. Full FT additionally memorizes label noise that the
rank-constrained update cannot fit.
```

### Reading the numbers
- **2x2 contrast:** full FT trades A (0.883→0.400) for B (0.436→0.740); LoRA trades less A (0.883→0.504) for slightly less B (0.436→0.688). Stability–plasticity, at matched convergence.
- **Mechanism is the rank, not the step count or norm:** update norms are comparable (20.1 vs 19.5) but numerical ranks are 7 vs 4 — the constraint limits *which* directions the update may touch, exactly the "low intrinsic rank of adaptation deltas" story from Hu et al. 2022.
- **Noise memorization** (full +0.115 train-over-test gap vs LoRA −0.002) is the demo's second teaching point: a capacity-limited residual cannot fit i.i.d. label noise.

## Citation verification log (all checked live, 2026-10-10)

| Key | Claim it supports | Verified against |
|---|---|---|
| hu2022lora | LoRA $W_0 + \frac{\alpha}{r}BA$; 10,000x fewer trainable params on GPT-3; no added inference latency; low "intrinsic rank" of adaptation $\Delta W$; $r$=1–2 sufficing even at $d$=12,288 | arXiv:2106.09685 + ICLR 2022 poster page |
| dettmers2023qlora | 4-bit NF4 quantized frozen base + LoRA; 65B on one 48GB GPU; double quantization; paged optimizers | arXiv:2305.14314 + NeurIPS 2023 proceedings |
| kirkpatrick2017overcoming | EWC; catastrophic forgetting definition; Fisher-weighted consolidation; permuted-MNIST experiments | PNAS 114(13):3521–3526, doi:10.1073/pnas.1611835114 |
| french1999catastrophic | Forgetting is abrupt not gradual; classic list-learning collapse; survey of causes/mitigations | Trends Cogn. Sci. 3(4):128–135 (PubMed 10322466) |
| lewis2020retrieval | RAG = parametric + non-parametric memory; dense Wikipedia index + retriever; knowledge-intensive tasks | NeurIPS 2020 proceedings + arXiv:2005.11401 |
| ovadia2024fine | RAG consistently beats unsupervised FT for knowledge injection (seen and unseen); FT "not competitive"; lower LR needed for continual pre-training to avoid forgetting; FT effect unstable | arXiv:2312.05934 + ACL Anthology 2024.emnlp-main.15 (pp. 237–250) |
| sun2020test | Test-time training: unlabeled test input becomes self-supervised problem; model updated before prediction; extends to online streams | ICML 2020 (PMLR 119) + arXiv:1909.13231 |
| ouyang2022training | InstructGPT: 1.3B preferred over 175B GPT-3; SFT + RLHF pipeline; behavior reshaping | NeurIPS 2022 proceedings (paper b1efde53...) |
| rafailov2023direct | DPO: classification-like objective on the policy; KL-anchored reward maximization solved without RL; stability | NeurIPS 2023 proceedings (paper a85b405e...) |

Note on Ovadia et al.: the paper was posted to arXiv in Dec 2023 but the peer-reviewed venue is EMNLP 2024 main (pages 237–250); we cite the EMNLP version. The task brief called it "Ovadia et al. 2023" — both datings appear in the wild; the ACL Anthology record is canonical.

## Known scope limits (for reviewers)
- The note's LoRA preservation story is demonstrated on a linear two-task toy; real Transformers add depth, attention, and optimizer state, so magnitudes differ (the *mechanism* — rank-bounded update, noise immunity — is what transfers).
- The "why naive online SGD destabilizes RLHF models" section is reasoning from the structure of the objectives (no KL anchor ⇒ unbounded drift; entropy collapse; compounding), consistent with the cited anchors, but we know of no single clean citation that isolates this failure mode experimentally — flagged as an inference, appropriately hedged in the note.
