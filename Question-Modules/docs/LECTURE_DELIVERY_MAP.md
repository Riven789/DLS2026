# DLS 2026 — Master Talks, Papers & Presentation Production Register

**Created:** 2026-10-10  
**Evidence base:** 41 anonymous Week-2 student responses and the existing DLS-2026 draft package.  
**Purpose:** one production register for every student-requested lecture, tutorial, reading set, and slide deck. This is a planning register—not a claim that materials are ready to teach.

## Production contract

Each approved lecture package should contain:

1. **Tutorial / instructor notes** — learning objectives, prerequisite check, worked derivations or systems example, live-demo plan, and discussion prompts.
2. **Beamer deck** — source plus rendered PDF; every substantive technical claim gets a source in notes or a references slide.
3. **Reading card** — 1 foundational paper, 1–2 current/technical papers, a reason to read each, and specific questions; links rather than redistributed PDFs unless rights permit.
4. **Student activity** — notebook, coding exercise, evidence-review exercise, or structured discussion/debate.
5. **Quality gate** — factual/mathematical review, fresh link check, render/overflow check, runnable demo check, and instructor approval before distribution.

**Naming convention:** `W##_L#.#_short-name/` with `tutorial.md`, `slides.tex`, `slides.pdf`, `reading-card.md`, `activity/`, and `QA.md`.

## Existing starting assets (audit before reuse)

- `02-introduction/DLS-2026-Introduction-DRAFT-v3.pptx/.pdf` — existing introduction deck, draft only.
- `00-sources/READING_MAP.md` — already includes Transformer, Chinchilla, FlashAttention, GQA, MLA, linear attention, multimodal, inference-time scaling, Jev, and FNO anchors.
- `03-course-repository/modules/` — scaffold and the release-gated Week-2 Lab 1; most later modules remain planned.
- `04-preparation/PREPARATION_TASKS.md` — course-wide review, evidence, pilot, and release gates.
- Historical DLS material is referenced by the DLS 2026 package but must be audited before reuse.

## Priority order

**P0 (needed immediately for Week 3):** L1.1, L1.2, L3.3.  
**P1 (next two weeks):** L1.3, L2.1, L1.4, L2.2, L3.1.  
**P2:** all remaining mapped sessions, built in week order with the reading cards first.

---

## A. Build / systems track (L1)

| ID | Talk / tutorial / deck | Student demand | Proposed week | Initial paper anchors | Status |
|---|---|---:|---:|---|---|
| L1.1 | **The Modern AI Stack, End to End** — data, tokenizer, transformer, training, inference server, harness, costs/bottlenecks | 6 named responses | 3 | *Attention Is All You Need*; selected current systems overview | P0 — not started |
| L1.2 | **Deep Learning from First Principles, in Code** — MLP, losses, backprop, optimization, PyTorch lab | 8 | 3 | Rumelhart et al. backprop; PyTorch autograd documentation | P0 — not started |
| L1.3 | **Building a Transformer** — BPE, masked MHA, residual stream, LayerNorm, positions; nanoGPT-scale implementation | 6 | 4 | *Attention Is All You Need*; RoPE reference | P1 — not started |
| L1.4 | **The Training Pipeline: Pre-training to Post-training** — SFT, RLHF/PPO, DPO, RLVR/GRPO; fine-tune vs prompt | 8 | 5 | InstructGPT; DPO; GRPO/RLVR source set | P1 — not started |
| L1.5 | **GPUs and Distributed Training** — SMs, memory, tensor cores, CUDA/Triton, parallelism, ZeRO/FSDP, Megatron/DeepSpeed | 9 | 7 | FlashAttention; ZeRO; Megatron-LM | P2 — not started |
| L1.6 | **Inference, Serving and Cost** — KV cache, batching, PagedAttention, quantization, speculative decoding, monitoring | 6 | 8 | vLLM/PagedAttention; speculative decoding source | P2 — not started |
| L1.7 | **Agent and Harness Engineering** — tools/MCP, context/memory, recovery, sandboxing, multi-agent coordination, token budgets | 11 | 9 | MCP specification; agent-evaluation / tool-use sources | P2 — not started |
| L1.8 | **Evaluating Models and Agents** — contamination, saturation, LLM judges, human evaluation, trajectories, verification | 8 | 10 | HELM; agent-evaluation sources | P2 — not started |
| L1.9 | **Working AI-Native Without Losing the Plot** — specs, review/verification, codebase mental models, reading the frontier | 7 | 11 | software-engineering / AI-use evidence set | P2 — not started |

## B. Internals / research track (L2)

| ID | Talk / tutorial / deck | Student demand | Proposed week | Initial paper anchors | Status |
|---|---|---:|---:|---|---|
| L2.1 | **From Next-Token Prediction to Capabilities** — compression, ICL, generalization, emergence, cross-entropy/perplexity | 6 | 4 | *Attention Is All You Need*; ICL/emergence evidence set | P1 — not started |
| L2.2 | **Reasoning: What It Is and When It Helps** — CoT, test-time compute, RLVR/GRPO, faithfulness debate | 4 | 5 | Chain-of-Thought Prompting; *s1*; reasoning-faithfulness sources | P1 — not started |
| L2.3 | **Scaling Laws and Their Limits** — Kaplan/Chinchilla, compute-optimality, data wall, forgetting, inference compute | 7 | 6 | Kaplan scaling laws; Chinchilla | P2 — not started |
| L2.4 | **Knowledge, Memory and Context** — MLP memory, superposition/features, context windows, RAG, KV cache, agent memory | 4 | 8 | transformer knowledge / RAG source set | P2 — not started |
| L2.5 | **Hallucination, Calibration and Simple Mistakes** — incentives, calibration, jaggedness, retrieval/abstention/verifiers | 5 | 10 | calibration and hallucination evidence set | P2 — not started |
| L2.6 | **Looking Inside: Interpretability and Alignment** — probes, SAEs, circuits, attribution; sycophancy and controllability | 5 | 10 | sparse autoencoder/circuits sources; alignment evidence | P2 — not started |
| L2.7 | **Beyond the Vanilla Transformer** — MoE, long context, multimodality, Mamba/linear attention/diffusion/JEPA/world models | 10 | 6 | Mamba; DeepSeek-V2/MLA; Kimi Linear; JEPA source | P2 — not started |
| L2.8 | **Model, Agent, Harness: Getting the Vocabulary Right** — definitions, failure attribution, models improving models | 6 | 9 | MCP specification; selected agent-system papers | P2 — not started |
| L2.9 | **Inside the Framework: PyTorch, Autodiff and Compilers** — tape vs functional AD, rematerialization, Dynamo/AOTAutograd/Inductor/Triton, JAX/XLA | 3 | 7 | PyTorch 2 compiler sources; JAX/XLA sources | P2 — not started |

## C. Society, professional practice & discussion track (L3)

| ID | Talk / tutorial / deck | Student demand | Proposed week | Initial evidence / reading anchors | Status |
|---|---|---:|---:|---|---|
| L3.1 | **AI and Work: Evidence, Not Just Forecasts** — labor data, task exposure, productivity, entry-level work, careers | 13 | 5 | labor-market and productivity studies; explicit uncertainty section | P1 — not started |
| L3.2 | **The Political Economy of AI** — compute/capital concentration, open/closed models, access, safety nets, ownership | 6 | 10 | compute governance and political-economy source set | P2 — not started |
| L3.3 | **Thinking With AI, Not Instead of It** — cognitive offloading, learning, deliberate practice, course AI-use policy | 13 | 3 | cognitive-offloading / learning evidence set; course policy | P0 — not started |
| L3.4 | **AI in Science: Acceleration and Integrity** — math/protein/materials cases, verification, peer review, credit | 8 | 8 | field-specific case studies; research-integrity standards | P2 — not started |
| L3.5 | **Safety, Security and Control** — agent threat models, prompt injection, sandbox escape, evaluations, scalable oversight | 8 | 9 | NIST AI RMF; frontier safety frameworks; agent-security evidence | P2 — not started |
| L3.6 | **The Physical and Financial Footprint** — energy/water, grid, GPU lifecycle, hyperscaler/neocloud economics | 3 | 7 | energy/water measurement sources; finance/economics sources | P2 — not started |
| L3.7 | **Optimistic Futures** (panel/debate) — health, robotics, access, discovery, historical analogies | 8 | 11 | balanced cases for benefits and distributional risks | P2 — not started |

---

## Week-by-week delivery map

| Week | Required package set | Delivery sequence |
|---|---|---|
| 3 | L1.1 Modern AI Stack; L1.2 DL First Principles; L3.3 Thinking With AI | **Build and review first.** |
| 4 | L1.3 Building a Transformer; L2.1 Next-Token to Capabilities | Build after Week-3 QA. |
| 5 | L1.4 Training Pipeline; L2.2 Reasoning; L3.1 AI and Work | Build reading cards early; evidence review is essential. |
| 6 | L2.3 Scaling Laws; L2.7 Beyond the Transformer | Pair technical derivations with comparison notebook. |
| 7 | L1.5 GPUs/Distributed Training; L2.9 Framework Internals; L3.6 Footprint | Ensure demos run on available student hardware. |
| 8 | L1.6 Inference/Serving; L2.4 Knowledge/Memory/Context; L3.4 AI in Science | Connect to reproducibility/evidence activity. |
| 9 | L1.7 Agent/Harness; L2.8 Model/Agent/Harness; L3.5 Safety/Security | Coordinate vocabulary to avoid duplicate decks. |
| 10 | L1.8 Evaluation; L2.5 Hallucination; L2.6 Interpretability; L3.2 Political Economy | Keep scope realistic: one integrated evaluation/interpretability deck may be preferable. |
| 11 | L1.9 Working AI-Native; L3.7 Optimistic Futures debate | Capstone synthesis and student-facing reading guide. |

## Coordination and review rules

- **One canonical register:** this file. Update status and ownership here before creating a deck.
- **No duplicated decks:** cross-track overlap is intentional but must share a single vocabulary/reference base—especially L1.7 and L2.8.
- **Source hygiene:** distinguish peer-reviewed results, preprints, vendor claims, and instructor synthesis; use evidence dates on contested current topics.
- **Academic integrity:** activities should teach verification and independent reasoning, not merely prompt production.
- **Release discipline:** do not publish or distribute course material without Rick’s review/approval and the course package’s pilot/release gates.

## Immediate next actions

1. Create the Week-3 package outlines for L1.1, L1.2, and L3.3.
2. Audit reusable slides/notes from the existing DLS catalog and record what is safe, current, and attributable to reuse.
3. Build the Week-3 reading cards and source ledger before drafting final technical claims.
4. Produce one small, runnable Week-3 code activity and independently test it in a clean environment.
