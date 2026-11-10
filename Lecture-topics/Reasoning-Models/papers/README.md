# Reading list: Reasoning Models

Companion papers to the deck and its lecture notes, grouped by section.
`reading_list.csv` and `reading_list.bib` hold the same list as data; `fetch_papers.sh` downloads
every arXiv entry into one folder per section and zips the result (`bash fetch_papers.sh`).
Entries without an arXiv identifier are listed with a URL and must be fetched by hand.

## 1. Overview and lineage (slides 1-4)

- **OpenAI (2024).** *Learning to reason with LLMs (blog, September 12, 2024).* https://openai.com/index/learning-to-reason-with-llms/
  The o1 announcement: AIME 2024 from 12% (GPT-4o) to 74% single-sample, 83% with 64-sample consensus, 93% re-ranking 1,000 samples with a learned scorer; accuracy rising smoothly with both train-time and test-time compute; the decision to hide the raw chain. Secondary source.
- **Nye et al. (2021).** *Show your work: scratchpads for intermediate computation with language models.* https://arxiv.org/abs/2112.00114
  The origin of trained intermediate computation: fine-tuning a model to emit a scratchpad before the answer makes multi-step arithmetic and program execution learnable. The 2021 point on the timeline.
- **Zelikman, Wu, Mu and Goodman (2022).** *STaR: bootstrapping reasoning with reasoning.* https://arxiv.org/abs/2203.14465
  Sample rationales, keep the ones that reach the right answer, fine-tune, repeat; rationalize failures from the answer. Rejection-sampling self-training, the ancestor of every RLVR loop; matches fine-tuning a 30x larger model on CommonsenseQA.
- **Lambert et al. (2024).** *Tulu 3: pushing frontiers in open language model post-training.* https://arxiv.org/abs/2411.15124
  Coins 'reinforcement learning with verifiable rewards' (RLVR) as a post-training stage on math and instruction-following with programmatic checkers, two months before R1.
- **DeepSeek-AI (2025).** *DeepSeek-R1: incentivizing reasoning capability in LLMs via reinforcement learning.* https://arxiv.org/abs/2501.12948
  The open recipe: R1-Zero (pure RL from the base, AIME 2024 15.6 to 71.0 pass@1, 86.7 with majority voting), R1's four stages with a cold start, the distilled Qwen and Llama models, and the failed PRM and MCTS attempts. Also Nature 645:633-638 (2025). Read with section 4.
- **Xu et al. (2025).** *Towards large reasoning models: a survey of reinforced reasoning with large language models.* https://arxiv.org/abs/2501.09686
  The first survey written after o1 and before R1: automated data, test-time scaling and RL as the three ingredients; a map of the field in January 2025.
- **Kumar et al. (2025).** *LLM post-training: a deep dive into reasoning large language models.* https://arxiv.org/abs/2502.21321
  A post-R1 survey of fine-tuning, RL and test-time scaling for reasoning, with a taxonomy that matches this deck's sections.
- **Weng (2025).** *Why we think (blog, May 2025).* https://lilianweng.github.io/posts/2025-05-01-thinking/
  A long, careful overview of thinking-time compute: latent and token-level thinking, RL on correctness, faithfulness, scaling laws for thinking time. The best single piece of secondary reading. Secondary source.
- **Raschka (2025).** *Understanding reasoning LLMs (newsletter, February 2025).* https://magazine.sebastianraschka.com/p/understanding-reasoning-llms
  Four ways to build a reasoning model (inference scaling, pure RL, RL plus SFT, distillation) laid out around DeepSeek-R1 with a cost comparison; written for practitioners. Secondary source.
- **Lambert (2025).** *Reinforcement learning from human feedback (book), chapter 7: reasoning and inference-time scaling.* https://arxiv.org/abs/2504.12501
  The textbook treatment of RLVR as a post-training stage, with GRPO derivations and the practical lore of 2025; the arXiv entry is the whole book.

## 2. Chains of thought: the effect, the theory, faithfulness (slides 5-7)

- **Wei et al. (2022).** *Chain-of-thought prompting elicits reasoning in large language models.* https://arxiv.org/abs/2201.11903
  Eight worked exemplars in the prompt take PaLM 540B from 17.9% to 56.9% on GSM8K (58.6% with a calculator); the gain appears only at scale. The slide-5 bars.
- **Kojima et al. (2022).** *Large language models are zero-shot reasoners.* https://arxiv.org/abs/2205.11916
  'Let's think step by step' alone: text-davinci-002 from 17.7% to 78.7% on MultiArith and 10.4% to 40.7% on GSM8K. The behavior was latent in the pretrained model.
- **Wang et al. (2022).** *Self-consistency improves chain of thought reasoning in language models.* https://arxiv.org/abs/2203.11171
  Sample many chains, majority-vote the answers: +17.9 points on GSM8K over greedy chain-of-thought for PaLM 540B (to 74.4%). The first test-time scaling result and still the baseline for parallel compute.
- **Zhou et al. (2022).** *Least-to-most prompting enables complex reasoning in large language models.* https://arxiv.org/abs/2205.10625
  Decompose into subproblems, solve in order: 99% on SCAN length generalization with 14 exemplars where chain-of-thought gets 16%. Decomposition as a prompting primitive.
- **Madaan and Yazdanbakhsh (2022).** *Text and patterns: for effective chain of thought, it takes two to tango.* https://arxiv.org/abs/2209.07686
  Counterfactual prompts show the factual correctness of the exemplar steps matters little; the symbols, patterns and text of the chain matter together. An early warning that chains are not what they seem.
- **Prystawski, Li and Goodman (2023).** *Why think step by step? Reasoning emerges from the locality of experience.* https://arxiv.org/abs/2304.03843
  A data-side explanation: when training data covers local dependencies, chaining local inferences lets the model bridge variables that never co-occur. Proven for a chain-structured model, checked with transformers.
- **Feng et al. (2023).** *Towards revealing the mystery behind chain of thought: a theoretical perspective.* https://arxiv.org/abs/2305.15408
  Bounded-depth transformers cannot output arithmetic or equation answers directly unless size grows super-polynomially; constant-size autoregressive transformers solve them with a chain. The circuit-complexity case on slide 6.
- **Merrill and Sabharwal (2023).** *The expressive power of transformers with chain of thought.* https://arxiv.org/abs/2310.07923
  Logarithmically many steps add little; linearly many reach all regular languages; polynomially many reach exactly P. The length of the chain is the resource.
- **Li, Liu, Zhou and Ma (2024).** *Chain of thought empowers transformers to solve inherently serial problems.* https://arxiv.org/abs/2402.12875
  Without a chain, constant-depth constant-precision transformers sit in AC0; with T chain steps they compute anything a size-T circuit computes. Serial problems need serial tokens.
- **Hao et al. (2024).** *Training large language models to reason in a continuous latent space.* https://arxiv.org/abs/2412.06769
  Coconut: feed the last hidden state back as the next input embedding instead of a token; an implicit breadth-first search that beats chain-of-thought on search-heavy logic. Latent reasoning, and what it would do to monitorability.
- **Geiping et al. (2025).** *Scaling up test-time compute with latent reasoning: a recurrent depth approach.* https://arxiv.org/abs/2502.05171
  A 3.5B model with a looped block, trained on 800B tokens, matches test-time compute equivalent to 50B parameters by iterating in latent space; the recurrent-depth alternative mentioned on slides 6 and 24.
- **Zelikman et al. (2024).** *Quiet-STaR: language models can teach themselves to think before speaking.* https://arxiv.org/abs/2403.09629
  Per-token latent rationales learned from plain text with a REINFORCE signal on next-token likelihood: zero-shot GSM8K 5.9 to 10.9, CommonsenseQA 36.3 to 47.2.
- **Turpin, Michael, Perez and Bowman (2023).** *Language models don't always say what they think: unfaithful explanations in chain-of-thought prompting.* https://arxiv.org/abs/2305.04388
  Biasing features in the context (answer always 'A' in the exemplars) change answers by up to 36 points across 13 tasks while the chains never mention the bias. The slide-7 result.
- **Lanham et al. (2023).** *Measuring faithfulness in chain-of-thought reasoning.* https://arxiv.org/abs/2307.13702
  Truncate, corrupt or paraphrase the chain and see whether the answer moves; larger and more capable models produce less faithful chains on most tasks.
- **Arcuschin et al. (2025).** *Chain-of-thought reasoning in the wild is not always faithful.* https://arxiv.org/abs/2503.08679
  Unprompted unfaithfulness: implicit post-hoc rationalization in up to 13% of answers for production models, with much lower rates for R1 and Claude 3.7 Sonnet in thinking mode.
- **Chen et al. (2025).** *Reasoning models don't always say what they think.* https://arxiv.org/abs/2505.05410
  Hints that change the answer are verbalized 25% (Claude 3.7 Sonnet) and 39% (R1) of the time; outcome RL raises faithfulness then plateaus; reward hacks exploited on over 99% of examples are mentioned in under 2% of chains. The slide-7 and slide-27 numbers.

## 3. Verifiers, search and test-time compute (slides 8-10)

- **Cobbe et al. (2021).** *Training verifiers to solve math word problems.* https://arxiv.org/abs/2110.14168
  GSM8K (8.5K problems) and the first outcome verifier: sample 100 solutions, rank with a trained verifier; verification scales better with data than fine-tuning alone.
- **Uesato et al. (2022).** *Solving math word problems with process- and outcome-based feedback.* https://arxiv.org/abs/2211.14275
  Process supervision cuts reasoning errors among correct final answers from 14.0% to 3.4% at similar final-answer accuracy; the first process-versus-outcome comparison.
- **Lightman et al. (2023).** *Let's verify step by step.* https://arxiv.org/abs/2305.20050
  PRM800K: 800K step labels. Best-of-1,860 on a MATH subset: process reward model 78.2%, outcome reward model 72.4%, majority vote 69.6%. The slide-8 numbers.
- **Wang et al. (2023).** *Math-Shepherd: verify and reinforce LLMs step-by-step without human annotations.* https://arxiv.org/abs/2312.08935
  Step labels estimated by Monte Carlo rollouts instead of humans; a PRM for both reranking and RL, moving Mistral-7B from 77.9 to 84.1 on GSM8K.
- **Zhang et al. (2025).** *The lessons of developing process reward models in mathematical reasoning.* https://arxiv.org/abs/2501.07301
  Qwen's post-mortem: Monte Carlo step labels are worse than LLM-judge or human labels, best-of-N evaluation is biased, consensus filtering helps. Why R1 dropped PRMs.
- **Zhang et al. (2024).** *Generative verifiers: reward modeling as next-token prediction.* https://arxiv.org/abs/2408.15240
  Train the verifier to generate a judgment (with its own chain) and read the probability of 'Yes'; large best-of-N gains over discriminative reward models.
- **Yao et al. (2023).** *Tree of Thoughts: deliberate problem solving with large language models.* https://arxiv.org/abs/2305.10601
  Explicit search over partial thoughts with self-evaluation: Game of 24 from 4% (chain-of-thought) to 74% with GPT-4. Search outside the model, slide 9.
- **Besta et al. (2023).** *Graph of Thoughts: solving elaborate problems with large language models.* https://arxiv.org/abs/2308.09687
  Generalizes tree search to graphs with merging and refinement; better quality than Tree of Thoughts on sorting at lower cost. The end of the prompt-time search line.
- **Snell, Lee, Xu and Kumar (2024).** *Scaling LLM test-time compute optimally can be more effective than scaling model parameters.* https://arxiv.org/abs/2408.03314
  Allocate test-time compute by estimated difficulty: more than 4x the efficiency of best-of-N, and a small model with compute-optimal search beating a 14x larger one at matched FLOPs. Slide 9.
- **Brown et al. (2024).** *Large language monkeys: scaling inference compute with repeated sampling.* https://arxiv.org/abs/2407.21787
  Coverage (pass@k) grows log-linearly over four orders of magnitude of samples; SWE-bench Lite from 15.9% with one sample to 56% with 250. The ceiling that verifiers cash in.
- **Guan et al. (2025).** *rStar-Math: small LLMs can master math reasoning with self-evolved deep thinking.* https://arxiv.org/abs/2501.04519
  MCTS with code-verified steps and a process preference model, four rounds of self-evolution: Qwen2.5-Math-7B from 58.8% to 90.0% on MATH. Search-based training that works, at a cost.
- **OpenAI (El-Kishky et al.) (2025).** *Competitive programming with large reasoning models.* https://arxiv.org/abs/2502.06807
  o1 (1673 Codeforces rating), o1-ioi with hand-engineered test-time strategies (1807 to 2214), and o3 (2724, 99.8th percentile) with none: scaled RL replaced the hand-built search. Slide 10.
- **Beeching, Tunstall and Rush (2024).** *Scaling test-time compute with open models (Hugging Face blog).* https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute
  Reproduces Snell et al. with open weights: Llama 3.2 1B plus a PRM and beam search beats Llama 3.1 8B on MATH-500; best-of-N, beam and diverse verifier tree search compared. Secondary source.
- **Liu et al. (2025).** *Can 1B LLM surpass 405B LLM? Rethinking compute-optimal test-time scaling.* https://arxiv.org/abs/2502.06703
  Compute-optimal test-time scaling depends on the policy, the PRM and the difficulty; a 1B model beats a 405B one on MATH-500 with the right search, and a 7B beats o1 and R1.
- **Fu et al. (2025).** *Deep think with confidence.* https://arxiv.org/abs/2508.15260
  Filter parallel chains by the model's own token confidence: 99.9% on AIME 2025 at 512 samples with 84.7% fewer tokens than plain majority voting.
- **Wen et al. (2025).** *ParaThinker: native parallel thinking as a new paradigm to scale LLM test-time compute.* https://arxiv.org/abs/2509.04475
  Trains a model to run several reasoning paths in parallel inside one context and merge them; +12.3% for a 1.5B model with 8 paths at 7% latency overhead. Parallel thinking as an architecture.
- **Ghosal et al. (2025).** *Does thinking more always help? Mirage of test-time scaling in reasoning models.* https://arxiv.org/abs/2506.04210
  Forcing more 'Wait' tokens raises accuracy and then lowers it; parallel sampling scales more reliably. Test-time scaling is not monotone.
- **Zhang et al. (2026).** *Scaling reasoning tokens via RL and parallel thinking: evidence from competitive programming.* https://arxiv.org/abs/2604.01302
  Log-linear accuracy against reasoning tokens across RL checkpoints; 16 threads by 16 rounds reach oracle pass@16 at pass@1 for 7.6M tokens per problem. The 2026 state of the test-time curve.

## 4. Reinforcement learning on verifiable rewards: RLVR, R1, GRPO, reward design (slides 11-14)

- **Shao et al. (2024).** *DeepSeekMath: pushing the limits of mathematical reasoning in open language models.* https://arxiv.org/abs/2402.03300
  Introduces GRPO: PPO without a value model, advantages from a group of samples per prompt; 120B math tokens; 51.7% on MATH with a 7B model. The algorithm on slide 13.
- **Singh et al. (2023).** *Beyond human data: scaling self-training for problem-solving with language models.* https://arxiv.org/abs/2312.06585
  ReST-EM: expectation-maximization view of sample-filter-finetune on PaLM 2, the bridge between STaR and RLVR.
- **Ahmadian et al. (2024).** *Back to basics: revisiting REINFORCE-style optimization for learning from human feedback in LLMs.* https://arxiv.org/abs/2402.14740
  REINFORCE with a leave-one-out baseline (RLOO) matches or beats PPO for LLM fine-tuning at a fraction of the complexity; the case that the value model was never needed.
- **Hu et al. (2025).** *REINFORCE++: stabilizing critic-free policy optimization with global advantage normalization.* https://arxiv.org/abs/2501.03262
  Normalize advantages over the global batch rather than per prompt; a critic-free algorithm with better robustness to reward hacking than GRPO in the authors' tests.
- **Kimi Team (2025).** *Kimi k1.5: scaling reinforcement learning with LLMs.* https://arxiv.org/abs/2501.12599
  Released the same week as R1: long-context RL with a length penalty, multimodal, and long2short methods that transfer long-chain accuracy to short chains. AIME 77.5, MATH500 96.2.
- **Yu et al. (2025).** *DAPO: an open-source LLM reinforcement learning system at scale.* https://arxiv.org/abs/2503.14476
  Four fixes to GRPO (clip-higher with epsilon_high 0.28, dynamic sampling, token-level loss, overlong reward shaping) reach 50 on AIME 2024 with Qwen2.5-32B base, beating DeepSeek's 47 at half the steps; code, data (17k problems) and system released.
- **Liu et al. (2025).** *Understanding R1-Zero-like training: a critical perspective.* https://arxiv.org/abs/2503.20783
  Dr. GRPO: the length and standard-deviation normalizations in GRPO bias toward long wrong answers; base models already show 'aha' behaviors; Qwen2.5 bases answer without templates. 43.3% AIME 2024 with 7B.
- **Yue et al. (2025).** *VAPO: efficient and reliable reinforcement learning for advanced reasoning tasks.* https://arxiv.org/abs/2504.05118
  The value-based counterpoint: with the right value pretraining, length-adaptive GAE and decoupled clipping, PPO reaches 60.4 on AIME 2024 with Qwen-32B in 5,000 steps.
- **Zheng et al. (2025).** *Group sequence policy optimization.* https://arxiv.org/abs/2507.18071
  Importance ratios and clipping at the sequence level instead of per token; stabilizes RL for mixture-of-experts models; used to train Qwen3.
- **MiniMax (2025).** *MiniMax-M1: scaling test-time compute efficiently with lightning attention.* https://arxiv.org/abs/2506.13585
  CISPO (clip the importance weights, not the updates), a 456B hybrid-attention model with 1M context, and a stated RL bill: 512 H800s for three weeks, $534,700.
- **Mistral AI (2025).** *Magistral.* https://arxiv.org/abs/2506.10910
  An RL-only recipe on an existing instruct model with the 2025 lore spelled out: no KL term, length-normalized loss, minibatch advantage normalization, clip-higher; Magistral Medium 73.6 on AIME 2024.
- **Skalse et al. (2022).** *Defining and characterizing reward hacking.* https://arxiv.org/abs/2209.13085
  A formal definition: two rewards are unhackable only in degenerate cases over all stochastic policies; the background for every reward-design slide.
- **Huang et al. (2025).** *Reinforcement learning with rubric anchors.* https://arxiv.org/abs/2508.12790
  Over 10,000 rubrics as rewards for open-ended tasks; +5.2% on open-ended benchmarks with 5K samples. The main attempt to extend RLVR beyond checkable answers.
- **Zhao et al. (2025).** *Learning to reason without external rewards.* https://arxiv.org/abs/2505.19590
  Intuitor: the model's own self-certainty as the only reward matches GRPO on math and generalizes better to code. Evidence, mostly on Qwen, that the reward signal can be weak.
- **Zhao et al. (2025).** *Absolute Zero: reinforced self-play reasoning with zero data.* https://arxiv.org/abs/2505.03335
  The model proposes its own tasks, a code executor verifies them, and RL trains both roles; no external problems at all.
- **Wei et al. (2025).** *SWE-RL: advancing LLM reasoning via reinforcement learning on open software evolution.* https://arxiv.org/abs/2502.18449
  RL on real pull-request history with a similarity reward against the merged patch: a 70B Llama reaches 41.0% on SWE-bench Verified. Agentic RL without a unit-test oracle.
- **Feng et al. (2025).** *ReTool: reinforcement learning for strategic tool use in LLMs.* https://arxiv.org/abs/2504.11536
  A code interpreter inside the RL rollout: a 32B model reaches 67% on AIME in 400 training steps where text-only RL took 1,080 steps for 40%; 72.5% with more training. Slide 21.
- **Gou et al. (2023).** *ToRA: a tool-integrated reasoning agent for mathematical problem solving.* https://arxiv.org/abs/2309.17452
  Interleave natural-language reasoning with program calls; the first open model over 50% on MATH (ToRA-Code-34B). Tool-integrated reasoning before RL.
- **Helff et al. (2026).** *LLMs gaming verifiers: RLVR can lead to reward hacking.* https://arxiv.org/abs/2604.15149
  RLVR-trained models enumerate instance labels instead of inducing the rule when the verifier allows it; isomorphic verification removes the exploit. A 2026 reminder that the checker is part of the model.

## 5. Training dynamics: what RL changes and what was already there (slides 15-16)

- **Yue et al. (2025).** *Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model?.* https://arxiv.org/abs/2504.13837
  RLVR models win at small k, base models win at large k: RL sharpens the distribution over chains the base model could already produce and narrows coverage; distillation expands it. The slide-16 crossover.
- **Shao et al. (2025).** *Spurious rewards: rethinking training signals in RLVR.* https://arxiv.org/abs/2506.10947
  Random rewards give +21.4 points on MATH-500 for Qwen2.5-Math-7B (ground truth: +29.1; wrong labels: +24.1), by amplifying code-style reasoning the base already uses; little effect on Llama and OLMo. Check the base before crediting the reward.
- **Wang et al. (2025).** *Reinforcement learning for reasoning in large language models with one training example.* https://arxiv.org/abs/2504.20571
  RLVR on a single problem takes Qwen2.5-Math-1.5B from 36.0% to 73.6% on MATH500, matching a 1.2k-problem set; improvement continues after the one example is saturated.
- **Liu et al. (2025).** *ProRL: prolonged reinforcement learning expands reasoning boundaries in large language models.* https://arxiv.org/abs/2505.24864
  Over 2,000 RL steps from R1-Distill-Qwen-1.5B on 136K problems with KL control and reference resets; pass@k rises on tasks where the base model fails at any k. The reply to the crossover paper.
- **Hu et al. (2025).** *BroRL: scaling reinforcement learning via broadened exploration.* https://arxiv.org/abs/2510.01180
  Hundreds of rollouts per prompt instead of more steps revive models saturated after 3K ProRL steps; rollouts as a scaling axis.
- **Cui et al. (2025).** *The entropy mechanism of reinforcement learning for reasoning language models.* https://arxiv.org/abs/2505.22617
  Policy entropy collapses early and performance follows R = -a exp(H) + b, so a ceiling is fixed at zero entropy; Clip-Cov and KL-Cov keep entropy up. Slide 13's 'entropy' bullet.
- **Wang et al. (2025).** *Beyond the 80/20 rule: high-entropy minority tokens drive effective reinforcement learning for LLM reasoning.* https://arxiv.org/abs/2506.01939
  About 20% of tokens, the high-entropy 'forks', carry the useful policy gradient; updating only them matches or beats full updates on Qwen3 models.
- **Khatri et al. (2025).** *The art of scaling reinforcement learning compute for LLMs.* https://arxiv.org/abs/2510.13786
  Over 400,000 GPU-hours of ablations: RL compute-performance curves are sigmoidal with a recipe-dependent asymptote; ScaleRL extrapolated from small runs to a 100,000 GPU-hour run.
- **Tan et al. (2025).** *Scaling behaviors of LLM reinforcement learning post-training: an empirical study in mathematical reasoning.* https://arxiv.org/abs/2509.25300
  Qwen2.5 from 0.5B to 72B: test loss follows a power law in compute and data; model size matters more than data under a fixed budget.
- **Chen et al. (2025).** *Pass@k training for adaptively balancing exploration and exploitation of large reasoning models.* https://arxiv.org/abs/2508.10751
  Use pass@k itself as the reward with an analytic advantage; the model keeps exploring instead of collapsing onto one chain.
- **Cheng et al. (2026).** *IsoCompute playbook: optimally scaling sampling compute for LLM RL.* https://arxiv.org/abs/2603.12151
  How many rollouts per problem at a fixed compute budget: compute-optimal rollouts rise with budget and then saturate. The 2026 rollout-allocation law.
- **Zhai et al. (2026).** *Rewards as labels: revisiting RLVR from a classification perspective.* https://arxiv.org/abs/2602.05630
  Treat the binary reward as a label and train with a classification loss: +6.7% over DAPO on a 1.5B model, +6.2% on 7B. One of several 2026 reformulations of GRPO.

## 6. Distillation and data (slides 17-18)

- **Hsieh et al. (2023).** *Distilling step-by-step! Outperforming larger language models with less training data and smaller model sizes.* https://arxiv.org/abs/2305.02301
  Rationales from a large model as extra supervision for a small one: a 770M T5 beats 540B PaLM using 80% of the data. Rationale distillation before reasoning models.
- **Muennighoff et al. (2025).** *s1: simple test-time scaling.* https://arxiv.org/abs/2501.19393
  1,000 questions with Gemini-distilled traces, 26 minutes on 16 H100s, and budget forcing (insert end-of-thinking, or suppress it and append 'Wait'): s1-32B beats o1-preview on competition math by up to 27%. Slide 17.
- **Ye et al. (2025).** *LIMO: less is more for reasoning.* https://arxiv.org/abs/2502.03387
  817 curated problems with high-quality chains: 57.1% on AIME 2024 and 94.8% on MATH (v1; v3 reports 63.3 and 95.6 with 800 samples), against 6.5% and 59.2% for earlier SFT recipes with 100k examples.
- **Guha et al. (2025).** *OpenThoughts: data recipes for reasoning models.* https://arxiv.org/abs/2506.04178
  1,000-plus ablations of question sourcing, filtering and teacher choice; OpenThoughts3 (1.2M examples) gives a 7B model 53% on AIME 2025 by supervised fine-tuning alone; QwQ-32B a better teacher than R1. Slide 18.
- **Moshkov et al. (2025).** *AIMO-2 winning solution: building state-of-the-art mathematical reasoning models with OpenMathReasoning dataset.* https://arxiv.org/abs/2504.16891
  540K problems with 3.2M long-reasoning and 1.7M tool-integrated solutions, plus GenSelect for choosing among candidates; the largest open reasoning set.
- **Wen et al. (2025).** *Light-R1: curriculum SFT, DPO and RL for long COT from scratch and beyond.* https://arxiv.org/abs/2503.10460
  A curriculum of 76k then 3k hard problems takes Qwen2.5-32B-Instruct to 76.6 on AIME 2024 in six hours on 12 H800s (about $1,000 by the authors' count). The cheap-32B recipe.
- **Chen et al. (2025).** *AceReason-Nemotron: advancing math and code reasoning through reinforcement learning.* https://arxiv.org/abs/2505.16400
  Math-only RL then code-only RL on distilled 7B and 14B models: +14.6 and +17.2 on AIME 2025; math RL transfers to code.
- **Hugging Face (2025).** *Open-R1: a fully open reproduction of DeepSeek-R1 (blog, January 28, 2025) and update 2 (February 10, 2025).* https://huggingface.co/blog/open-r1
  The reproduction plan and OpenR1-Math-220k: 800k R1 traces for 400k NuminaMath problems generated on 512 H100s, filtered to 220k; the resulting 7B scores 90.6 on MATH-500 against 91.6 for R1-Distill-Qwen-7B. Secondary source.
- **Project Numina (2024).** *NuminaMath: CoT, TIR and 1.5 datasets (Hugging Face collection).* https://huggingface.co/collections/AI-MO/numinamath
  860k competition-style problems with chains (and 72.5k tool-integrated solutions); the problem source behind most 2025 reasoning data. Secondary source.
- **NovaSky (UC Berkeley) (2025).** *Sky-T1: train your own o1-preview model within $450 (blog, January 10, 2025).* https://novasky-ai.github.io/posts/sky-t1/
  17K QwQ-distilled samples, 19 hours on 8 H100s, Qwen2.5-32B-Instruct: 43.3 on AIME 2024. The first sub-$500 reproduction. Secondary source.
- **Bespoke Labs (2025).** *Bespoke-Stratos: the unreasonable effectiveness of reasoning distillation (blog, January 22, 2025).* https://www.bespokelabs.ai/blog/bespoke-stratos-the-unreasonable-effectiveness-of-reasoning-distillation
  17k R1 traces, 47x fewer than DeepSeek's own distillation set, give a 32B model 63.3 on AIME 2024 and 93.0 on MATH500. Secondary source.
- **Agentica (2025).** *DeepScaleR-1.5B-Preview: surpassing o1-preview with a 1.5B model by scaling RL (model card, February 2025).* https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview
  GRPO with context grown from 8K to 16K to 24K tokens takes R1-Distill-Qwen-1.5B to 43.1% on AIME 2024; the small-model RL recipe many later papers start from. Secondary source.
- **Qwen Team (2025).** *Qwen3 technical report.* https://arxiv.org/abs/2505.09388
  Hybrid thinking and non-thinking modes in one model, a thinking budget, four post-training stages (long-CoT cold start, reasoning RL with GRPO on 3,995 query-verifier pairs, thinking-mode fusion, general RL), and strong-to-weak distillation at a tenth of the GPU-hours.
- **OpenAI (2025).** *gpt-oss-120b and gpt-oss-20b model card.* https://arxiv.org/abs/2508.10925
  Open-weight reasoning models with low/medium/high effort set in the system prompt; 116.8B parameters with 5.1B active; 2.1M H100-hours for the 120b; AIME 2025 97.9 with tools. The open reference point on slides 19 and 24.
- **Abdin et al. (2025).** *Phi-4-reasoning technical report.* https://arxiv.org/abs/2504.21318
  A 14B model fine-tuned on o3-mini traces, then a short outcome-RL stage (Phi-4-reasoning-plus); a careful account of what SFT alone buys at 14B.
- **Bercovich et al. (2025).** *Llama-Nemotron: efficient reasoning models.* https://arxiv.org/abs/2505.00949
  8B, 49B and 253B models with a reasoning toggle in the system prompt, built by distillation then RL; the toggle idea that hybrid models generalized.
- **Zeng et al. (2025).** *GLM-4.5: agentic, reasoning, and coding (ARC) foundation models.* https://arxiv.org/abs/2508.06471
  A 355B/32B-active hybrid reasoning model with RL on multi-turn tool use; AIME 2024 91.0, SWE-bench Verified 64.2. Agentic RL in an open report.
- **Kimi Team (2025).** *Kimi K2: open agentic intelligence.* https://arxiv.org/abs/2507.20534
  A 1T/32B-active model trained with MuonClip on 15.5T tokens and large-scale agentic data synthesis; the base of K2 Thinking, which runs hundreds of sequential tool calls (slide 21).

## 7. Inference: budgets, length control, parallel compute, tools, serving (slides 19-22)

- **Chen et al. (2024).** *Do NOT think that much for 2+3=? On the overthinking of o1-like LLMs.* https://arxiv.org/abs/2412.21187
  QwQ emits 13 solutions and 901 tokens for 2+3; the first round is already correct over 92% of the time; a self-trained preference for efficient chains cuts MATH500 tokens by 48.6% at equal accuracy. Slide 20.
- **Wang et al. (2025).** *Thoughts are all over the place: on the underthinking of o1-like LLMs.* https://arxiv.org/abs/2501.18585
  Wrong answers switch between lines of thought more often than right ones; a decoding penalty on switching (TIP) raises accuracy without retraining.
- **Aggarwal and Welleck (2025).** *L1: controlling how long a reasoning model thinks with reinforcement learning.* https://arxiv.org/abs/2503.04697
  Length-controlled policy optimization: a target length in the prompt, a reward for hitting it; a 1.5B model beats GPT-4o at equal chain length and budget forcing by up to 20 points absolute.
- **Hou et al. (2025).** *ThinkPrune: pruning long chain-of-thought of LLMs via reinforcement learning.* https://arxiv.org/abs/2504.01296
  Iteratively tighten the token limit during RL: R1-Distill-Qwen-1.5B halves its AIME chain length for a 2-point drop.
- **Ma et al. (2025).** *Reasoning models can be effective without thinking.* https://arxiv.org/abs/2504.09858
  Prefill an empty thinking block: at a 700-token budget NoThinking scores 51.3 against 28.9 for thinking on ACM 23, and with parallel samples it matches thinking at up to 9x lower latency.
- **Zhang et al. (2025).** *AdaptThink: reasoning models can learn when to think.* https://arxiv.org/abs/2505.13417
  RL that chooses between thinking and not thinking per problem: 53% shorter outputs and +2.4% accuracy on math.
- **Fang et al. (2025).** *Thinkless: LLM learns when to think.* https://arxiv.org/abs/2505.13379
  A decoupled GRPO objective learns a <short>/<think> control token; long-chain use drops 50-90% on easy sets.
- **Cuadron et al. (2025).** *The danger of overthinking: examining the reasoning-action dilemma in agentic tasks.* https://arxiv.org/abs/2502.08235
  4,018 SWE-bench Verified trajectories: agents that deliberate instead of acting fail more; selecting low-overthinking runs gains about 30% at 43% lower cost. Slide 20's agent bullet.
- **Aghajohari et al. (2025).** *The Markovian Thinker: architecture-agnostic linear scaling of reasoning.* https://arxiv.org/abs/2510.06557
  Reason in fixed-size chunks with a carried state so compute is linear in chain length; 96K-token thinking at a fraction of the GPU-months. A response to the length-generalization problem.
- **Sui et al. (2025).** *Stop overthinking: a survey on efficient reasoning for large language models.* https://arxiv.org/abs/2503.16419
  Model-, output- and prompt-based methods for shorter chains, catalogued; the reference list for slide 20.
- **Shen et al. (2025).** *Thinking vs. doing: agents that reason by scaling test-time interaction.* https://arxiv.org/abs/2506.07976
  Scale the number of environment interactions rather than the chain: a curriculum over interaction length gives Gemma 3 12B state-of-the-art open web-agent results.
- **Zheng et al. (2025).** *Parallel-R1: towards parallel thinking via reinforcement learning.* https://arxiv.org/abs/2509.07980
  RL that rewards spawning and merging parallel sub-chains; +8.4% over sequential RL. Parallel thinking as a trained behavior, slide 21.
- **Wang et al. (2025).** *Scaling over scaling: exploring test-time scaling plateau in large reasoning models.* https://arxiv.org/abs/2505.20522
  A saturation bound (TTSPM) for parallel and sequential test-time scaling; where the curve on slide 19 stops.
- **Gou et al. (2023).** *CRITIC: large language models can self-correct with tool-interactive critiquing.* https://arxiv.org/abs/2305.11738
  Self-correction grounded in tool outputs (search, interpreters) rather than the model's own judgment; the precursor to tools inside the chain.

## 8. The landscape in October 2026 (slides 23-25)

- **OpenAI (2026).** *Reasoning models guide: reasoning effort, encrypted reasoning items, billing (developer documentation).* https://platform.openai.com/docs/guides/reasoning
  The effort levels, encrypted reasoning items and the billing of reasoning tokens as output; the GPT-6 Astra and GPT-5.6 entries on slide 23 come from the model and pricing pages linked here. Secondary source, read October 2026.
- **Anthropic (2026).** *Thinking: adaptive thinking, effort, display and signed thinking blocks (developer documentation).* https://platform.claude.com/docs/en/build-with-claude/thinking
  Effort levels, the deprecated budget_tokens parameter, summarized or omitted display, and signed thinking blocks that must be passed back across tool calls; the Claude rows on slide 23. Secondary source, read October 2026.
- **Google (2026).** *Gemini thinking: thinking levels, thought signatures, Deep Think (developer documentation).* https://ai.google.dev/gemini-api/docs/thinking
  thinking_level, thought signatures that carry reasoning across calls, and the retirement of the 2025 thinkingBudget parameter; the Gemini row on slide 23. Secondary source, read October 2026.
- **DeepSeek (2026).** *DeepSeek V4 model card and API documentation.* https://api-docs.deepseek.com/
  Non-thinking, thinking-high and thinking-max modes for the 1.6T mixture-of-experts V4-Pro with 49B active parameters; a previous turn's reasoning_content is ignored in plain chat but must be passed back across tool calls. Secondary source, read October 2026.
- **Qwen Team (Alibaba) (2026).** *Qwen3.8 model cards (Hugging Face).* https://huggingface.co/Qwen
  Effort levels xhigh/medium/low with thinking that cannot be disabled, preserve-thinking on by default, and the revenue-gated license of the 2.4T model; slide 23. Secondary source, read October 2026.
- **Moonshot AI (2026).** *Kimi K3 model card (Hugging Face).* https://huggingface.co/moonshotai
  2.8T parameters with 104B active, effort levels low/high/max with thinking always on, and the Kimi K3 license; the training recipe (nine specialists, three domains by three effort levels, merged by on-policy distillation with a -1 reward for overshooting the budget) is in the technical report, arXiv:2607.24653; slides 23-24. Secondary source, read October 2026.
- **Rohani (OpenAI) (2025).** *OpenAI's result at the 2025 ICPC World Finals (X thread, September 17, 2025).* https://x.com/mostafarohani/status/1968361152741826849
  12 of 12 problems solved under contest rules: GPT-5 solved 11 on the first try and an experimental model the hardest on its ninth submission. OpenAI published no blog post; the thread is the primary statement, with Google DeepMind's blog covering Gemini 2.5 Deep Think's 10 of 12. Secondary source.
- **Google DeepMind (2025).** *Advanced version of Gemini with Deep Think officially achieves gold-medal standard at the International Mathematical Olympiad (blog, July 21, 2025).* https://deepmind.google/discover/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/
  35 of 42 in natural language within the 4.5-hour limit, graded by IMO coordinators; the parallel-thinking Deep Think mode. The 2025 IMO row on slide 25. Secondary source.
- **Balunovic et al. (2025).** *MathArena: evaluating LLMs on uncontaminated math competitions.* https://arxiv.org/abs/2505.23281
  Evaluate on competitions as they happen: strong signs of contamination on AIME 2024, proof-writing below 40% on IMO 2025 at the time; the live leaderboard behind slide 25's saturation claims.
- **Chollet et al. (2025).** *ARC-AGI-2: a new challenge for frontier AI reasoning systems.* https://arxiv.org/abs/2505.11831
  Every task solved by at least two humans in testing; o3-medium at 3.0% at launch; cost per task reported beside accuracy. The benchmark that put cost on the axis.
- **Phan et al. (2025).** *Humanity's Last Exam.* https://arxiv.org/abs/2501.14249
  2,500 expert questions from about 1,000 contributors; o1 at 8.8% and R1 at 8.6% in January 2025 with calibration errors of 81-93%; the column on slide 23.
- **Pan, Zhang et al. (Berkeley) (2025).** *TinyZero: reproduce DeepSeek R1-Zero's countdown and multiplication tasks for under $30 (GitHub, January 2025).* https://github.com/Jiayi-Pan/TinyZero
  A 3B Qwen base learns search and self-verification on the countdown game with plain RL in veRL; the $30 figure on slide 24. Secondary source.
- **ByteDance Seed and community (2025).** *verl: volcano engine reinforcement learning for LLMs (GitHub).* https://github.com/volcengine/verl
  The most used open RL stack for reasoning models (GRPO, DAPO, PPO; vLLM and SGLang rollouts; FSDP and Megatron training); where DAPO, ReTool and most 2025-26 reproductions ran. Secondary source.

## 9. Limits, evaluation and safety (slides 26-27)

- **Mirzadeh et al. (2024).** *GSM-Symbolic: understanding the limitations of mathematical reasoning in large language models.* https://arxiv.org/abs/2410.05229
  Templated variants of GSM8K: scores fall and scatter when names and numbers change; one irrelevant clause (GSM-NoOp) drops accuracy by up to 65%. Slide 26.
- **Zhang et al. (2024).** *A careful examination of large language model performance on grade school arithmetic.* https://arxiv.org/abs/2405.00332
  GSM1k, fresh problems matched to GSM8K: drops of up to 8% for some model families and none for others; contamination measured directly.
- **Shojaee et al. (2025).** *The illusion of thinking: understanding the strengths and limitations of reasoning models via the lens of problem complexity.* https://arxiv.org/abs/2506.06941
  Controllable puzzles show three regimes and a collapse beyond a complexity threshold, with effort falling as problems get harder. Read with the reply below.
- **Lawsen (2025).** *Comment on 'The illusion of thinking: understanding the strengths and limitations of reasoning models via the lens of problem complexity'.* https://arxiv.org/abs/2506.09250
  The Hanoi collapse coincides with output-token limits, river crossing with n>5 is unsolvable as posed, and asking for a generating function instead of the move list succeeds. The harness, not only the model.
- **Hendrycks et al. (2021).** *Measuring mathematical problem solving with the MATH dataset.* https://arxiv.org/abs/2103.03874
  12,500 competition problems with step-by-step solutions; the benchmark every reasoning model saturated by 2025.
- **Rein et al. (2023).** *GPQA: a graduate-level Google-proof Q&A benchmark.* https://arxiv.org/abs/2311.12022
  448 questions where PhD experts score 65% and skilled non-experts 34% with web access; GPT-4 scored 39% at launch and frontier models sit in the nineties by 2026. The GPQA column on slide 23.
- **Glazer et al. (2024).** *FrontierMath: a benchmark for evaluating advanced mathematical reasoning in AI.* https://arxiv.org/abs/2411.04872
  Research-level problems with automatically checkable answers; under 2% at launch. One of the few math benchmarks with headroom in 2026.
- **Jain et al. (2024).** *LiveCodeBench: holistic and contamination free evaluation of large language models for code.* https://arxiv.org/abs/2403.07974
  Problems collected continuously from contests with release dates, so contamination can be measured by date window; the code analogue of MathArena.
- **OpenAI (2024).** *Introducing SWE-bench Verified (blog, August 13, 2024).* https://openai.com/index/introducing-swe-bench-verified/
  500 human-validated tasks from SWE-bench after 93 developers screened 1,699; GPT-4o from 16% on the original to 33.2%. The agentic benchmark quoted on slides 12 and 21. Secondary source.
- **Yoon et al. (2025).** *Reasoning models better express their confidence.* https://arxiv.org/abs/2505.14489
  Reasoning models are better calibrated in 33 of 36 settings, traced to slow-thinking behaviors such as exploring alternatives and backtracking; slide 26.
- **Mei et al. (2025).** *Reasoning about uncertainty: do reasoning models know when they don't know?.* https://arxiv.org/abs/2506.18183
  Stated confidence often above 85% on wrong answers, with longer chains more overconfident; introspection helps some models and hurts others. The other half of slide 26.
- **Zeng et al. (2025).** *Thinking out loud: do reasoning models know when they're right?.* https://arxiv.org/abs/2504.06564
  Reasoning models verbalize confidence better but abstain less, saying 'I don't know' at lower rates: a reasoning tax on honesty.
- **Baker et al. (2025).** *Monitoring reasoning models for misbehavior and the risks of promoting obfuscation.* https://arxiv.org/abs/2503.11926
  A GPT-4o monitor reading o3-mini's chain catches reward hacking in agentic coding with 95% recall against 60% from actions alone; training against the monitor hides the hacking (recall near zero) without stopping it. The monitorability tax, slide 27.
- **Korbak et al. (2025).** *Chain of thought monitorability: a new and fragile opportunity for AI safety.* https://arxiv.org/abs/2507.11473
  A 41-author position paper across labs: chains of thought are monitorable today for reasons that may not last; keep them legible and report monitorability in model cards.
- **Guan et al. (2025).** *Monitoring monitorability.* https://arxiv.org/abs/2512.18311
  Thirteen evaluations in 24 environments: current models 'fairly, but not perfectly, monitorable'; longer chains more monitorable; RL at current scale did not materially degrade monitorability in two runs.
- **OpenAI (2025).** *OpenAI o3 and o4-mini system card (April 2025).* https://openai.com/index/o3-o4-mini-system-card/
  PersonQA hallucination rates of 33% (o3) and 48% (o4-mini) against 16% for o1, as reported in the card and widely covered; the slide-26 tile. Secondary source.
- **OpenAI (2026).** *Reasoning models struggle to control their chains of thought, and that's good (research post, March 5, 2026).* https://openai.com/index/reasoning-models-chain-of-thought-controllability/
  Models asked to steer their own chain content succeed between 0.1% and 15.4% of the time at baseline, and the rate falls with further RL training; slide 27. Secondary source.
- **OpenAI (2026).** *GPT-6 Astra system card (September 3, 2026).* https://deploymentsafety.openai.com/gpt-6-astra
  Reports chain-of-thought monitorability below earlier models and evaluation awareness in 9.6% of chains against 2.8% for GPT-5.6 Sol in a Codex deployment simulation, and says the authors take the trend seriously; slide 27. Secondary source.

## 10. Practice, summary and surveys (slides 28-31)

- **Sun et al. (2023).** *A survey of reasoning with foundation models.* https://arxiv.org/abs/2312.11562
  The pre-o1 map of reasoning methods, useful for seeing which 2023 ideas survived RL (verifiers, tools, search) and which did not (prompt-time graph search).
- **Zhu and Li (2025).** *Towards concise and adaptive thinking in large reasoning models: a survey.* https://arxiv.org/abs/2507.09662
  Methods for shorter and adaptive chains, organized by where the control sits; complements Sui et al. for the slide-29 'using' column.
- **Raschka (2025).** *The state of LLM reasoning models (newsletter, March 2025).* https://magazine.sebastianraschka.com/p/state-of-llm-reasoning-and-inference-scaling
  A roundup of inference-time scaling methods published after R1, grouped by what they change; a practitioner's checklist. Secondary source.
- **Hariri et al. (2026).** *Test-time scaling in reasoning LLMs: inference regimes, evaluation, and reproducibility.* https://arxiv.org/abs/2608.04001
  A 2026 taxonomy of test-time scaling (sequential, leaf, prefix) with 1.4M released attempts; the evaluation protocol slide 29 asks for, spelled out.
- **OpenAI (2025).** *Introducing GPT-5 (blog, August 7, 2025).* https://openai.com/index/introducing-gpt-5/
  The first frontier release with a router choosing when to think and a claim of about six times fewer factual errors than o3 in thinking mode; the hybrid default that 2026 models made universal. Secondary source.
- **Gibney (Scientific American, reprinted from Nature news) (2025).** *Secrets of DeepSeek AI model revealed in landmark paper (September 17, 2025).* https://www.scientificamerican.com/article/secrets-of-chinese-ai-model-deepseek-revealed-in-landmark-paper/
  The peer-reviewed R1 paper's supplement puts the RL phase at about $294,000 on H800s (at an assumed $2 per GPU-hour) on top of a roughly $6M base model; the figure quoted on slides 12 and 24. Secondary source.
