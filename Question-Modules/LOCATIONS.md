# Student-Facing Locations — AI Course 2026

One canonical module per student question, plus labeled extensions. All materials are in the private GitHub repo **rick-stevens-ai/ai-course-2026** (browse: github.com/rick-stevens-ai/ai-course-2026 — every module directory below is a path inside it).

## Start here
- `questions/MASTER-QUESTIONS.pdf` — the 32 student questions, organized in 7 topical clusters, with the owner/reviewer matrix
- `README.md` — the module contract (what every module contains) and organization rules
- `MASTER-LIST.md` — the original 25-lecture map the modules hang on (weeks 3–11)
- `survey/student_survey_synthesis_anonymous.pdf` — the source survey (41 responses)

## The 32 canonical modules (`modules/`)
Each contains: note.pdf (3–5 pg tutorial), deck.pdf (5–10 slides), discussion.pdf (5 questions), projects.pdf, quiz.pdf, code where applicable, QA.md.

**Cluster A — Prediction, Intelligence, Reasoning (Q01–Q05)**
- Q01 `modules/Q01-next-token-capabilities/` — How does next-token prediction lead to problem solving, coding, instruction following?
- Q02 `modules/Q02-intelligence-emergence/` — Is next-token generation intelligence?
- Q03 `modules/Q03-reasoning-real-or-illusion/` — Is reasoning real, a property of language, or an illusion?
- Q04 `modules/Q04-rl-reasoning/` — How does RL-based reasoning differ from ordinary generation?
- Q05 `modules/Q05-reasoning-tradeoff/` — When does extra reasoning actually improve an answer?

**Cluster B — Computation, Emergence, Scaling (Q06–Q10)**
- Q06 `modules/Q06-generalization-emergence/` — What are models computing? Generalization vs. memorization?
- Q07 `modules/Q07-scaling-laws/` — Why does scaling work? Is there a science of scaling?
- Q08 `modules/Q08-scaling-limits/` — Will performance saturate? Can we predict more parameters?
- Q09 `modules/Q09-data-composition/` — How do training data, architecture, compute each affect performance?
- Q10 `modules/Q10-jagged-frontier/` — Why can models solve complex problems yet make simple mistakes?

**Cluster C — Hallucination, Knowledge, Memory (Q11–Q15)**
- Q11 `modules/Q11-hallucination-and-uncertainty/` — Why do models hallucinate? Can they know they're unsure?
- Q12 `modules/Q12-confident-when-wrong/` — Why do models give confident answers when wrong?
- Q13 `modules/Q13-knowledge-representation-storage/` — How do models represent and store knowledge?
- Q14 `modules/Q14-concepts-features-memory/` — How are "concepts" maintained and generated?
- Q15 `modules/Q15-context-retrieval-memory/` — Context windows, retrieval, conversation memory?

**Cluster D — Architectures, Brains, Consciousness (Q16–Q21)**
- Q16 `modules/Q16-frontier-architectures/` — What differentiates frontier architectures?
- Q17 `modules/Q17-transformer-variations/` — Transformer variations and why they matter?
- Q18 `modules/Q18-llm-vs-brains/` — LLMs vs. brains?
- Q19 `modules/Q19-ai-consciousness/` — Could AI be conscious?
- Q20 `modules/Q20-shared-internals/` — How much of LLM internals are shared with other deep learning?
- Q21 `modules/Q21-composed-systems/` — How do networks and RL policies combine in one system?

**Cluster E — Agents, Harnesses, Evaluation (Q22–Q25)**
- Q22 `modules/Q22-llm-vs-agent/` — What separates an LLM from an agent? Is Claude Code a harness?
- Q23 `modules/Q23-harness-design/` — What is a harness? Model vs. instructions vs. harness failures?
- Q24 `modules/Q24-agent-evaluation/` — How do we judge whether an agent is good?
- Q25 `modules/Q25-ai-improving-ai/` — Can AI agents fine-tune and improve other models?

**Cluster F — Post-training, Interpretability, Safety (Q26–Q29)**
- Q26 `modules/Q26-finetuning-adaptation/` — How does fine-tuning change behavior? Adapting to new domains?
- Q27 `modules/Q27-interpretability/` — Can inspecting weights reveal tendencies? Do models revert?
- Q28 `modules/Q28-alignment-faking/` — Why does chain-of-thought differ when models believe they're unobserved?
- Q29 `modules/Q29-agent-control-safety/` — Is AI controllable? Safety protocols?

**Cluster G — Systems, Measurement (Q30–Q32)**
- Q30 `modules/Q30-pytorch-compiler-autodiff/` — PyTorch, compilers, autodiff?
- Q31 — **not yet drafted** (Ollie's lane, in his queue)
- Q32 `modules/Q32-information-entropy/` — What is information/entropy in this context?

## Extension modules (deeper or adjacent treatments, clearly labeled)
- `modules/Q03-ext-rl-finetuning/` — RLHF/DPO/GRPO depth (pairs with Q04)
- `modules/Q11-ext-mechanistic-hallucination/` — frequency-vs-accuracy mechanics (pairs with Q11)
- `modules/Q12-ext-knowledge-boundaries/` — the knowledge boundary map (pairs with Q11/Q12)
- `modules/Q13-ext-superposition/` — superposition demo (pairs with Q13)
- `modules/Q27-ext-weights-fingerprint/` — frozen-weights/distribution-shift view (pairs with Q27)

## Reviews (quality records)
- `reviews/` — every peer review, the canonicalization decision record, and the review matrix

## Companion videos
- `video-recommendations-ric.json` — Rick's three verified recommendations (3B1B 27:14, StatQuest 36:44, Karpathy 1:56:20 — all flagged as outside the 10–15 min window, pending Rick's waiver decision)
- Per-module video pointers: tracked in each module's `sources.md`/`video.md`; repo-wide video work is parked per Rick until content sign-offs complete
