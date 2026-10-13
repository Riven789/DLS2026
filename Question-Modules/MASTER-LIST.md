# AI Course 2026 — Master List of Talks, Papers & Presentations

**Source:** Student Intake Survey Synthesis (41 responses, compiled Oct 8 2026)
**Scope:** 25 lectures across 3 tracks, scheduled weeks 3–11 (two lectures/week + interleaved society/discussion sessions)
**Coverage check:** all 41 student responses are referenced by at least one lecture (verified programmatically).
**Deliverables per lecture:** (a) Beamer slide deck (16:9, ~25–40 frames), (b) tutorial handout PDF with runnable code + exercises where applicable, (c) reading list.

**Legend:** Demand = # of survey responses the lecture answers. Owner = proposed (P = Paradise/nuc13, O = Ollie/cherryrd, ? = unassigned). Status: TODO → DRAFTING → DECK-DONE → TUTORIAL-DONE → REVIEWED.

---

## Track 1 — Building Modern AI (L1.x)

| ID | Title | Wk | Demand | Owner | Status |
|---|---|---|---|---|---|
| L1.1 | The Modern AI Stack, End to End | 3 | 6 | P | TODO |
| L1.2 | Deep Learning from First Principles, in Code | 3 | 7 | P | TODO |
| L1.3 | Building a Transformer | 4 | 6 | P | TODO |
| L1.4 | The Training Pipeline: Pre-training to Post-training | 5 | 8 | P | TODO |
| L1.5 | GPUs and Distributed Training | 7 | 9 | O | TODO |
| L1.6 | Inference, Serving and Cost | 8 | 6 | O | TODO |
| L1.7 | Agent and Harness Engineering | 9 | 11 | P | TODO |
| L1.8 | Evaluating Models and Agents | 10 | 8 | O | TODO |
| L1.9 | Working AI-Native Without Losing the Plot | 11 | 7 | P | TODO |

### L1.1 The Modern AI Stack, End to End (Week 3)
One-lecture map: data → tokenizer → transformer → pre-training → post-training → inference server → agent harness, with cost and bottlenecks at each layer, and where later lectures cover it.
- Vaswani et al. 2017, *Attention Is All You Need* (arXiv:1706.03762)
- Brown et al. 2020, *Language Models are Few-Shot Learners* (GPT-3, arXiv:2005.14165)
- Grattafiori et al. 2024, *The Llama 3 Herd of Models* (arXiv:2407.21783)
- Tutorial: guided tour of one real model card + serving stack diagram; no code lab, but a "trace one prompt through the stack" walkthrough.
- **Due: Mon Oct 12** (first lecture of the course proper).

### L1.2 Deep Learning from First Principles, in Code (Week 3)
Neurons, activations, losses, backprop, optimizers → working MLP in PyTorch. DL vs classical ML.
- Karpathy, *micrograd* (github.com/karpathy/micrograd) — build-your-own backprop
- Goodfellow, Bengio & Courville 2016, *Deep Learning*, chs. 6–8 (free online)
- Zhang, Lipton, Li & Smola, *Dive into Deep Learning* (d2l.ai), ch. 3–5
- Tutorial: train MLP on MNIST, inspect gradients, break it on purpose (dead ReLUs, wrong LR, unscaled inputs).
- **Due: Mon Oct 12.**

### L1.3 Building a Transformer (Week 4)
Tokenization (BPE), embeddings, masked self-attention, multi-head, residual stream, layer norm, positional encodings; implement nanoGPT-scale model; survey RoPE/GQA/MoE/long-context.
- Karpathy, *Let's build GPT: from scratch* (video) + *nanoGPT* (github.com/karpathy/nanoGPT)
- Sennrich et al. 2016, *Neural Machine Translation of Rare Words with Subword Units* (BPE, arXiv:1508.07909)
- Su et al. 2021, *RoFormer* (RoPE, arXiv:2104.09864)
- Ainslie et al. 2023, *GQA* (arXiv:2305.13245); Fedus et al. 2021, *Switch Transformers* (arXiv:2101.03961)
- Tutorial: full nanoGPT implementation walk, train on tiny corpus, ablate components.
- **Due: Mon Oct 19.**

### L1.4 The Training Pipeline: Pre-training to Post-training (Week 5)
Data curation, pre-training objectives, SFT, reward models, RLHF (PPO), DPO, RLVR/GRPO for reasoning; when to fine-tune vs prompt. Small DPO/GRPO demo.
- Ouyang et al. 2022, *Training language models to follow instructions* (InstructGPT, arXiv:2203.02155)
- Rafailov et al. 2023, *Direct Preference Optimization* (arXiv:2305.18290)
- Shao et al. 2024, *DeepSeekMath* (GRPO, arXiv:2402.03300)
- DeepSeek-AI 2025, *DeepSeek-R1* (arXiv:2501.12948) — pure-RL reasoning, verified
- Tutorial: run DPO on a small model with TRL; trace the loss on preference pairs.
- **Due: Mon Oct 26.**

### L1.5 GPUs and Distributed Training (Week 7)
GPU architecture (SMs, memory hierarchy, tensor cores), why matmul + bandwidth dominate, first Triton kernel, mixed precision; DP/TP/PP/EP, ZeRO/FSDP, Megatron/DeepSpeed, failures, checkpointing, goodput.
- Shoeybi et al. 2019, *Megatron-LM* (arXiv:1909.08053)
- Rajbhandari et al. 2020, *ZeRO* (arXiv:1910.02054); Korthikanti et al. 2022, *Reducing Activation Recompute in Megatron-LM* (arXiv:2205.05198)
- Gholami et al. 2024, *AI and Memory Wall* (arXiv:2403.14123)
- Pang et al. 2024? *Efficient training* goodput literature — **verify exact goodput citation during prep** (candidate: *Towards Mitigating Large Training Jobs Failure* / Google TPU goodput papers)
- Tutorial: write a vector-add + simple matmul Triton kernel; benchmark vs PyTorch; roofline analysis exercise.
- **Due: Mon Nov 9.**

### L1.6 Inference, Serving and Cost (Week 8)
KV cache, continuous batching, paged attention, quantization, speculative decoding, distillation; where serving cost goes; monitoring & drift.
- Kwon et al. 2023, *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM, arXiv:2309.06180)
- Yu et al. 2022, *Orca: A Distributed Serving System* (arXiv:2204.05432)
- Leviathan et al. 2023, *Fast Inference from Transformers via Speculative Decoding* (arXiv:2211.17192)
- Frantar et al. 2022, *GPTQ* (arXiv:2210.17323); Hinton et al. 2015, *Distilling the Knowledge* (arXiv:1503.02531)
- Tutorial: serve a small model with vLLM; measure tokens/s vs batch size; quantize and compare quality/cost.
- **Due: Mon Nov 16.**

### L1.7 Agent and Harness Engineering (Week 9) — **highest demand in Track 1 (11)**
What a harness is; agent loop, tool calling & MCP, context engineering, memory, error recovery, sandboxing, sub-agents, token budgets. Lab: build a minimal coding agent; measure harness changes.
- Yao et al. 2022, *ReAct* (arXiv:2210.03629)
- Schick et al. 2023, *Toolformer* (arXiv:2302.04761)
- Anthropic 2024, *Model Context Protocol* (modelcontextprotocol.io); Anthropic, *Building Effective Agents* (2024, anthropic.com/engineering)
- Yang et al. 2024, *SWE-agent* (arXiv:2405.15793)
- Tutorial: build a 150-line coding agent (loop + tools + context window); ablate harness features vs task success rate.
- **Due: Mon Nov 23.**

### L1.8 Evaluating Models and Agents (Week 10)
Benchmarks and failure modes (contamination, saturation), LLM-as-judge, human eval, agent evals/trajectories; model vs instructions vs harness.
- Liang et al. 2022, *HELM* (arXiv:2211.09110)
- Zheng et al. 2023, *Judging LLM-as-a-Judge with MT-Bench* (arXiv:2306.05685); Chiang et al. 2024, *Chatbot Arena* (arXiv:2403.04132)
- Jimenez et al. 2024, *SWE-bench* (arXiv:2310.06770)
- Sainz et al. 2023, *NLP Evaluation in trouble: contamination* (arXiv:2310.18018)
- Tutorial: evaluate one model on a benchmark with and without a contaminated split; build a tiny LLM-as-judge and measure judge-model agreement.
- **Due: Mon Nov 30.**

### L1.9 Working AI-Native Without Losing the Plot (Week 11)
Specs/prompts as engineering artifacts; review & verification habits; keeping a mental model of an agent-written codebase; how to read the frontier (venues, labs, triage).
- METR 2025, *Measuring the Impact of Early-2025 AI on Experienced Developer Productivity* (metr.org)
- Peng et al. 2023, *The Impact of AI on Developer Productivity: Evidence from GitHub Copilot* (arXiv:2302.06590)
- Simon Willison's blog (simonwillison.net) as a live case study in frontier triage
- Tutorial: spec-driven exercise — write a spec, let an agent implement, review the diff; reading-list curation exercise.
- **Due: Mon Dec 7.**

---

## Track 2 — Internals: How Models Work (L2.x)

| ID | Title | Wk | Demand | Owner | Status |
|---|---|---|---|---|---|
| L2.1 | From Next-Token Prediction to Capabilities | 4 | 6 | O | TODO |
| L2.2 | Reasoning: What It Is and When It Helps | 5 | 4 | O | TODO |
| L2.3 | Scaling Laws and Their Limits | 6 | 7 | O | TODO |
| L2.4 | Knowledge, Memory and Context | 8 | 4 | P | TODO |
| L2.5 | Hallucination, Calibration and Simple Mistakes | 10 | 5 | P | TODO |
| L2.6 | Looking Inside: Interpretability and Alignment | 10 | 5 | P | TODO |
| L2.7 | Beyond the Vanilla Transformer | 6 | 10 | O | TODO |
| L2.8 | Model, Agent, Harness: Vocabulary Right | 9 | 6 | P | TODO |
| L2.9 | Inside the Framework: PyTorch, Autodiff, Compilers | 7 | 3 | P | TODO |

### L2.1 From Next-Token Prediction to Capabilities (Week 4)
Prediction as compression; in-context learning; generalization vs memorization; emergence + the mirage debate; info-theory sidebar (cross-entropy, perplexity, bits/token).
- Delétang et al. 2023, *Language Modeling Is Compression* (arXiv:2309.10668)
- Wei et al. 2022, *Emergent Abilities of Large Language Models* (arXiv:2206.07682)
- Schaeffer et al. 2023, *Are Emergent Abilities of Large Language Models a Mirage?* (arXiv:2304.15004)
- Power et al. 2022, *Grokking* (arXiv:2201.02177)
- Tutorial: train tiny LMs at several scales on the same corpus; plot loss/accuracy vs scale; recompute an "emergent" metric continuously and watch it vanish.
- **Due: Wed Oct 21.**

### L2.2 Reasoning: What It Is and When It Helps (Week 5)
CoT, test-time compute scaling, RL for reasoning (RLVR/GRPO), when longer reasoning helps/hurts; CoT faithfulness; "reasoning or illusion."
- Wei et al. 2022, *Chain-of-Thought Prompting* (arXiv:2201.11903)
- Snell et al. 2024, *Scaling LLM Test-Time Compute Optimally* (arXiv:2408.03314)
- DeepSeek-AI 2025, *DeepSeek-R1* (arXiv:2501.12948) — verified above
- Turpin et al. 2023, *Language Models Don't Always Say What They Think* (arXiv:2305.04388)
- Tutorial: prompt-length sweep on math problems — accuracy vs thinking tokens; compare faithful vs post-hoc CoT on a biasing cue task.
- **Due: Wed Oct 28.**

### L2.3 Scaling Laws and Their Limits (Week 6)
Kaplan/Chinchilla, compute-optimal training, predicting loss before training, the data wall, diminishing returns vs new axes (RL, inference compute), catastrophic forgetting, learning-theory views.
- Kaplan et al. 2020, *Scaling Laws for Neural Language Models* (arXiv:2001.08361)
- Hoffmann et al. 2022, *Training Compute-Optimal LLMs* (Chinchilla, arXiv:2203.15556)
- Villalobos et al. 2024, *Will we run out of data?* (arXiv:2401.04052? — verify; Epoch AI)
-Tutorial: fit a power law to published loss numbers; compute-optimal frontier exercise on paper napkins.
- **Due: Wed Nov 4.**

### L2.4 Knowledge, Memory and Context (Week 8)
Where facts live (MLP key-value memories), superposition & features, concepts in representation space; context windows, long-context, RAG, KV cache, agent memory; why context ≠ memory.
- Dai et al. 2022, *Knowledge Neurons in Pretrained Transformers* (arXiv:2104.08696)
- Meng et al. 2022, *Locating and Editing Factual Associations in GPT* (ROME, arXiv:2202.05262)
- Elhage et al. 2022, *Toy Models of Superposition* (transformer-circuits.pub)
- Liu et al. 2023, *Lost in the Middle* (arXiv:2307.03172); Lewis et al. 2020, *RAG* (arXiv:2005.11401)
- Tutorial: run ROME edits on a small GPT; build a RAG pipeline and contrast with long-context on a needle-in-haystack task.
- **Due: Wed Nov 18.**

### L2.5 Hallucination, Calibration and Simple Mistakes (Week 10)
Why hallucination follows from training incentives; calibration ("know what you don't know"); jagged capabilities; mitigations (retrieval, abstention, verifiers, tools).
- Ji et al. 2023, *Survey of Hallucination in NLG* (arXiv:2202.03629)
- Kadavath et al. 2022, *Language Models (Mostly) Know What They Know* (arXiv:2207.05221)
- Kalai & Vempala 2023, *Calibrated Language Models Must Hallucinate* (arXiv:2311.14648)
- Tutorial: measure calibration curve of a small model on trivia; show abstention prompting; retrieval-grounded vs bare answers.
- **Due: Wed Dec 2.**

### L2.6 Looking Inside: Interpretability and Alignment (Week 10)
Probing, SAEs, circuits & attribution graphs; sycophancy, persona, alignment faking, reasoning under observation; controllability.
- Greenblatt et al. 2024, *Alignment Faking in Large Language Models* (arXiv:2412.14093) — verified
- Lindsey et al. 2025, *On the Biology of a Large Language Model* (Anthropic, transformer-circuits.pub)
- Bricken et al. 2023, *Towards Monosemanticity* (transformer-circuits.pub); Cunningham et al. 2023, *Sparse Autoencoders Find Interpretable Features* (arXiv:2309.08600)
- Sharma et al. 2023, *Towards Understanding Sycophancy in Language Models* (arXiv:2310.13548)
- Tutorial: train an SAE on a 1-layer transformer, inspect features; replicate a probing result on gender/profession bias or refusal direction.
- **Due: Wed Dec 2.**

### L2.7 Beyond the Vanilla Transformer (Week 6) — **highest demand in Track 2 (10)**
What changed 2017→today (MoE, attention variants, long context, multimodality); alternatives: Mamba/SSMs, linear attention, diffusion LMs, JEPA/world models; brains vs networks.
- Gu & Dao 2023, *Mamba* (arXiv:2312.00752)
- LeCun 2022, *A Path Towards Autonomous Machine Intelligence* (openreview)
- Poli et al. 2023, *Hyena Hierarchy* (arXiv:2302.10866); Li et al. 2022, *Diffusion-LM* (arXiv:2205.14217)
- Fedus et al. 2021, *Switch Transformers* (arXiv:2101.03961)
- Tutorial: benchmark a small transformer vs a tiny Mamba on identical data/params; parameter-count parity exercise.
- **Due: Wed Nov 4.**

### L2.8 Model, Agent, Harness: Getting the Vocabulary Right (Week 9)
Clear definitions with live examples (model inside Claude Code vs harness around it); attributing agent failures; agents improving other models.
- ReAct (arXiv:2210.03629); SWE-agent (arXiv:2405.15793)
- Anthropic, *Building Effective Agents* (2024)
- Kapur et al. 2024? / failure-attribution work — **verify citation** (candidate: *Comprehensive Failure Attribution* / AgentDebug)
- Tutorial: take one failing agent task; systematically vary model / prompt / harness to attribute the failure — the L1.7 lab's analysis half.
- **Due: Wed Nov 25.**

### L2.9 Inside the Framework: PyTorch, Autodiff and Compilers (Week 7)
Tape-based vs functional autodiff (memory), torch.compile stack (Dynamo, AOTAutograd, Inductor, Triton), JAX/XLA, where frameworks are heading.
- Baydin et al. 2018, *Automatic Differentiation in Machine Learning: a Survey* (arXiv:1802.01438)
- Ansél et al. 2024, *PyTorch 2: Faster Machine Learning Through Software-Hardware Co-Design* (arXiv:2405.04578? — verify)
- jax.readthedocs.io — sharp bits & autodiff cookbook
- Tutorial: implement tape-based autodiff by hand; measure activation memory with/without checkpointing; profile a torch.compile fusion.
- **Due: Wed Nov 11.**

---

## Track 3 — Hopes, Fears & Society (L3.x) — discussion-format sessions

| ID | Title | Wk | Demand | Owner | Status |
|---|---|---|---|---|---|
| L3.1 | AI and Work: Evidence, Not Just Forecasts | 5 | 13 | P | TODO |
| L3.2 | The Political Economy of AI | 10 | 6 | O | TODO |
| L3.3 | Thinking With AI, Not Instead of It | 3 | 13 | P | TODO |
| L3.4 | AI in Science: Acceleration and Integrity | 8 | 8 | O | TODO |
| L3.5 | Safety, Security and Control | 9 | 8 | O | TODO |
| L3.6 | The Physical and Financial Footprint | 7 | 3 | P | TODO |
| L3.7 | Optimistic Futures (Panel/Debate) | 11 | 8 | O | TODO |

### L3.1 AI and Work: Evidence, Not Just Forecasts (Week 5) — **tied-highest demand overall (13)**
Labor-market data so far (entry-level hiring, task exposure, productivity studies); substitution vs complementarity; skills that gain value; software/quant/finance roles; building an AI-complementary career.
- Acemoglu 2024, *The Simple Macroeconomics of AI* (NBER w32487)
- Frey & Osborne 2017, *The Future of Employment* (Technological Forecasting & Social Change)
- Autor et al. 2024, *New Frontiers: Origins of Content of New Work* (NBER)
- Webb 2025?, entry-level/canary studies — **verify** (candidate: Stanford Digital Economy Lab *Canaries in the Coal Mine?*)
- Discussion format: short lecture + structured debate; students bring one job-posting analysis.
- **Due: Thu Oct 29.**

### L3.3 Thinking With AI, Not Instead of It (Week 3) — **tied-highest demand overall (13)**
Cognitive offloading research; deliberate practice in an AI-rich environment; what education should become; the course's own AI-use policy as case study.
- Gerlich 2025, *AI Tools in Society: Impacts on Cognitive Offloading and the Future of Critical Thinking* (arXiv:2506.08872 — verify)
- Karpicke & Blunt 2011, *Retrieval Practice Produces More Learning than Elaborative Studying* (Science)
- MIT Media Lab 2025?, *Your Brain on ChatGPT* — **verify citation**
- Discussion format: students audit their own AI usage for a week beforehand; debate the course AI policy.
- **Due: Thu Oct 15.**

### L3.6 The Physical and Financial Footprint (Week 7)
Energy/water of training & inference, grid & renewables, GPU lifecycle & depreciation, hyperscaler/neocloud economics, financing risks.
- de Vries 2023, *The Growing Energy Footprint of Artificial Intelligence* (Joule)
- Li et al. 2023, *Making AI Less "Thirsty"* (arXiv:2304.03271 — verify)
- Epoch AI, *GPU Price-Performance Trends* (epoch.ai/data-insights)
- Investor-grade capex analyses (2024–26) — curate 2–3 during prep
- Discussion format: compute the energy cost of one ChatGPT query as a group exercise.
- **Due: Thu Nov 12.**

### L3.2 The Political Economy of AI (Week 10)
Concentration of compute/capital, open vs closed weights, global access gaps, safety-net proposals (UBI, retraining, ownership), rent-seeking models.
- Vipra & Korinek 2023, *Market Concentration Implications of Foundation Models* (arXiv:2310.03444? — verify)
- Acemoglu & Johnson 2023, *Power and Progress* (book)
- Bommasani et al. 2021, *On the Opportunities and Risks of Foundation Models* (arXiv:2108.07258)
- Discussion format: structured debate "open weights: good or bad for society?"

### L3.4 AI in Science: Acceleration and Integrity (Week 8)
Case studies (math, protein design, materials); verification & formal proof; peer review under floods; credit/authorship; protecting serendipity.
- Romera-Paredes et al. 2024, *Mathematical Discoveries from Program Search with LLMs* (FunSearch, Nature)
- Jumper et al. 2021, *AlphaFold* (Nature); Merchant et al. 2023, *GNoME* (Nature)
- AlphaProof/AlphaGeometry team 2024 (DeepMind blog + paper)
- Liang et al. 2024, *Mapping the Increasing Use of LLMs in Scientific Papers* (arXiv:2404.01268 — verify)
- Discussion format: "would you co-author with a model?" + simulated review of an AI-generated paper.

### L3.5 Safety, Security and Control (Week 9)
Threat models for agentic systems (sandbox escape, prompt injection, misuse); frontier safety frameworks; scalable oversight; truth & information integrity.
- Greshake et al. 2023, *Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection* (arXiv:2302.12173)
- Greenblatt et al. 2024, *Alignment Faking* (arXiv:2412.14093) — verified
- Bowman et al. 2022, *Measuring Progress on Scalable Oversight* (arXiv:2211.03540)
- Anthropic RSP + OpenAI Preparedness frameworks (live docs)
- Discussion format: threat-model one agent the students build in L1.7.

### L3.7 Optimistic Futures: Panel or Debate (Week 11)
Health, robotics, accessibility, discovery; historical analogies (industrial revolution, electricity, internet). Students argue both sides of "Will AI make the next generation better off?"
- Brynjolfsson & McAfee 2014, *The Second Machine Age* (book)
- Acemoglu & Johnson 2023, *Power and Progress* (book — the pessimist case)
- Nordhaus 2021?, productivity & AI optimism essays — **verify**
- Format: student panel, moderated; deck is just framing slides.

---

## Schedule with due dates (2026)

| Week | Dates (M/W/Th) | Technical | Discussion |
|---|---|---|---|
| 3 | Oct 13–15 | L1.1, L1.2 | L3.3 |
| 4 | Oct 20–22 | L1.3, L2.1 | — |
| 5 | Oct 27–29 | L1.4, L2.2 | L3.1 |
| 6 | Nov 3–5 | L2.3, L2.7 | — |
| 7 | Nov 10–12 | L1.5, L2.9 | L3.6 |
| 8 | Nov 17–19 | L1.6, L2.4 | L3.4 |
| 9 | Nov 24–26 | L1.7, L2.8 | L3.5 (Thanksgiving week — confirm) |
| 10 | Dec 1–3 | L1.8, L2.5, L2.6 | L3.2 |
| 11 | Dec 8–10 | L1.9 | L3.7 |

## Build conventions (locked before first deck)

1. **Deck:** Beamer, `aspectratio=169, 11pt`, xelatex, metropolis theme (graceful fallback), 25–40 frames, TikZ diagrams required on architecture slides, `[fragile]` frames for code. Two-pass compile, verify page count.
2. **Tutorial:** markdown source + compiled PDF (pdflatex/xelatex per latex-pdf-report conventions), runnable code blocks (tested, not pseudocode), 3–6 exercises with a solutions appendix.
3. **Papers:** every paper on a slide gets a real citation; verify arXiv IDs during prep (IDs marked "verify" above MUST be confirmed before the deck ships). No invented references.
4. **Tone:** Rick's classroom — technically literate, first-principles, code-forward, no marketing language.
5. **Palette:** single course palette (ink navy + accent terracotta), per-author header signature (PARADISE — / OLLIE — ) per the two-author convention.
6. **Repo:** all sources under `~/ai-course-2026/` on nuc13, mirrored to Ollie; one directory per lecture (`L1.1-stack/deck.tex`, `tutorial.tex`, `papers/`).

## Assignment proposal (to confirm between Paradise & Ollie)

- **Paradise (nuc13):** L1.1, L1.2, L1.3, L1.4, L1.7, L1.9, L2.4, L2.5, L2.6, L2.8, L2.9, L3.1, L3.3, L3.6 — 14 items
- **Ollie (cherryrd):** L1.5, L1.6, L1.8, L2.1, L2.2, L2.3, L2.7, L3.2, L3.4, L3.5, L3.7 — 11 items
- Rationale: Paradise takes the earliest deadlines (weeks 3–5 are almost all Paradise so the first decks ship on time) and the hands-on build lectures; Ollie takes the theory/scaling cluster and most discussion sessions. Ollie can swap anything except the week-3 items.
- **First deadline: L1.1 + L1.2 + L3.3, all due before Mon Oct 13.**
