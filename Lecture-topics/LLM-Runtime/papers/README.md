# Reading list: LLM Runtime Issues

Companion papers to the deck and its lecture notes, grouped by section.
`reading_list.csv` and `reading_list.bib` hold the same list as data; `fetch_papers.sh` downloads
every arXiv entry into one folder per section and zips the result (`bash fetch_papers.sh`).
Entries without an arXiv identifier are listed with a URL and must be fetched by hand.

## 1. Overview and the KV cache (slides 1-6) (slides 1-6)

- **Pope et al. (2022).** *Efficiently scaling transformer inference.* https://arxiv.org/abs/2211.05102  
  The inference cost model: memory- vs compute-bound phases, the KV cache as the budget, and partitioning strategies for large models. The best single starting point for slides 3-4 and 9.
- **Yu, Jeong, Kim, Kim and Chun (2022).** *Orca: a distributed serving system for transformer-based generative models.* https://www.usenix.org/conference/osdi22/presentation/yu  
  Iteration-level (continuous) batching and selective batching; the scheduling idea every engine now uses. OSDI 2022. Not on arXiv.
- **Kwon et al. (2023).** *Efficient memory management for large language model serving with PagedAttention.* https://arxiv.org/abs/2309.06180  
  vLLM: the KV cache in fixed blocks with a block table, waste from 60-80% to under 4%, 2-4x throughput at the same latency. SOSP 2023.
- **Zheng et al. (2023).** *SGLang: efficient execution of structured language model programs.* https://arxiv.org/abs/2312.07104  
  RadixAttention (a radix tree over cached prefixes), the compressed FSM for constrained decoding, and the frontend language; up to 6.4x throughput on prefix-sharing workloads.
- **Dao (2023).** *FlashAttention-2: faster attention with better parallelism and work partitioning.* https://arxiv.org/abs/2307.08691  
  The attention kernel behind most engines; why long contexts are memory-linear rather than quadratic in activation memory.
- **Ye et al. (2025).** *FlashInfer: efficient and customizable attention engine for LLM inference serving.* https://arxiv.org/abs/2501.01005  
  A serving-oriented attention library (paged KV layouts, decode kernels, JIT specialization) used by SGLang and vLLM.
- **Ainslie et al. (2023).** *GQA: training generalized multi-query transformer models from multi-head checkpoints.* https://arxiv.org/abs/2305.13245  
  Grouped-query attention: why Llama 2 70B's and Llama 3 70B's caches are one eighth of a multi-head 65B model's (slide 4).
- **DeepSeek-AI (2024).** *DeepSeek-V2: a strong, economical, and efficient mixture-of-experts language model.* https://arxiv.org/abs/2405.04434  
  Multi-head latent attention: one compressed latent per token instead of keys and values; the 69 KB per token of slide 4 comes from V3's use of it.
- **Gim et al. (2024).** *Prompt Cache: modular attention reuse for low-latency inference.* https://arxiv.org/abs/2311.04934  
  Reusing attention states for prompt segments across requests; an early form of the prefix caching of slide 6.

## 2. Scheduling, speculation, parallel and disaggregated serving, quantization (slides 7-10) (slides 7-10)

- **Agrawal et al. (2024).** *Taming throughput-latency tradeoff in LLM inference with Sarathi-Serve.* https://arxiv.org/abs/2403.02310  
  Chunked prefill with stall-free batching: decodes keep running while a long prompt is processed in pieces. OSDI 2024; the default in vLLM V1 and SGLang.
- **Leviathan, Kalman and Matias (2022).** *Fast inference from transformers via speculative decoding.* https://arxiv.org/abs/2211.17192  
  Draft k tokens with a small model, verify with one target pass, accept by rejection sampling so that the output distribution is exactly the target's; 2-3x in the paper.
- **Chen et al. (2023).** *Accelerating large language model decoding with speculative sampling.* https://arxiv.org/abs/2302.01318  
  The same idea from DeepMind, with the acceptance rule written out and results on Chinchilla 70B.
- **Cai et al. (2024).** *Medusa: simple LLM inference acceleration framework with multiple decoding heads.* https://arxiv.org/abs/2401.10774  
  Extra decoding heads on the target model predict several future tokens; tree attention verifies many candidates at once.
- **Li, Wei, Zhang and Zhang (2024).** *EAGLE: speculative sampling requires rethinking feature uncertainty.* https://arxiv.org/abs/2401.15077  
  A one-layer draft that autoregresses on the target's hidden features; the strongest open drafting method, with EAGLE-2 and EAGLE-3 following.
- **Li, Wei, Zhang and Zhang (2025).** *EAGLE-3: scaling up inference acceleration of large language models via training-time test.* https://arxiv.org/abs/2503.01840  
  Training the draft on features from several target layers with training-time test; supported in vLLM and SGLang.
- **Fu, Bailis, Stoica and Zhang (2024).** *Break the sequential dependency of LLM inference using lookahead decoding.* https://arxiv.org/abs/2402.02057  
  Speculation without a draft model, by solving for several tokens at once with Jacobi iteration; the ancestor of n-gram lookup tricks.
- **Zhong et al. (2024).** *DistServe: disaggregating prefill and decoding for goodput-optimized large language model serving.* https://arxiv.org/abs/2401.09670  
  Prefill and decode on separate GPU pools, each sized for its phase; up to 7.4x more requests under the same latency targets, and the definition of goodput used on slide 16. OSDI 2024.
- **Patel et al. (2023).** *Splitwise: efficient generative LLM inference using phase splitting.* https://arxiv.org/abs/2311.18677  
  The same split from Microsoft, with a cluster design that uses different hardware for each phase.
- **Qin et al. (2024).** *Mooncake: a KVCache-centric disaggregated architecture for LLM serving.* https://arxiv.org/abs/2407.00079  
  Kimi's production serving system: the KV cache as a distributed store across CPU, DRAM and SSD, with prefill and decode pools. The FAST 2025 best-paper version carries a longer title (Trading more storage for less computation).
- **Sheng et al. (2023).** *S-LoRA: serving thousands of concurrent LoRA adapters.* https://arxiv.org/abs/2311.03285  
  Batching requests for different LoRA adapters through one base model with unified paging of adapter weights and KV cache.
- **Chen et al. (2023).** *Punica: multi-tenant LoRA serving.* https://arxiv.org/abs/2310.18547  
  The batched LoRA kernel (SGMV) that lets one GPU serve many adapters; the basis of vLLM's LoRA support.
- **Frantar, Ashkboos, Hoefler and Alistarh (2022).** *GPTQ: accurate post-training quantization for generative pre-trained transformers.* https://arxiv.org/abs/2210.17323  
  Second-order weight-only quantization to 4 and 3 bits; one of the two standard formats served by every engine.
- **Lin et al. (2023).** *AWQ: activation-aware weight quantization for LLM compression and acceleration.* https://arxiv.org/abs/2306.00978  
  Protect the weights that matter by scaling with activation statistics; the other standard 4-bit format, with fast kernels.
- **Liu et al. (2024).** *KIVI: a tuning-free asymmetric 2bit quantization for KV cache.* https://arxiv.org/abs/2402.02750  
  Keys per channel, values per token, down to 2 bits: the KV-cache quantization row of slide 10.
- **Song, Mi, Xie and Chen (2023).** *PowerInfer: fast large language model serving with a consumer-grade GPU.* https://arxiv.org/abs/2312.12456  
  Exploit activation sparsity to keep hot neurons on the GPU and cold ones on the CPU; consumer-GPU serving before MoE offload.
- **Eliseev and Mazur (2023).** *Fast inference of mixture-of-experts language models with offloading.* https://arxiv.org/abs/2312.17238  
  Expert offloading with LRU caching and speculative expert loading; the idea behind llama.cpp's --n-cpu-moe and KTransformers.
- **Chen et al. (Tsinghua MADSys) (2025).** *KTransformers: unleashing the full potential of CPU/GPU hybrid inference for MoE models.* https://doi.org/10.1145/3731569.3764843  
  Serving 671B DeepSeek-V3/R1 on one GPU plus CPU memory: expert offload with CPU kernels, reported 4.6-19.7x prefill and 1.25-4.1x decode speedups over the baseline. SOSP 2025. Not on arXiv.

## 3. Engines: vLLM, SGLang, TensorRT-LLM, llama.cpp, Ollama, MLX (slides 11-15) (slides 11-15)

- **vLLM project (2025).** *vLLM documentation: V1 engine, features, OpenAI-compatible server (docs).* https://docs.vllm.ai/  
  Architecture of the V1 engine, feature matrix (speculative decoding, structured output backends, tool and reasoning parsers, quantization), and the serving CLI. Secondary source; changes monthly.
- **SGLang project (2025).** *SGLang documentation (docs).* https://docs.sglang.ai/  
  Runtime features, the frontend language, structured output, speculative decoding and the large-scale DeepSeek deployment write-ups. Secondary source.
- **SGLang team (2025).** *Deploying DeepSeek with PD disaggregation and large-scale expert parallelism on 96 H100 GPUs (blog).* https://lmsys.org/blog/2025-05-05-large-scale-ep/  
  The reference large-scale MoE serving write-up: P/D disaggregation, wide expert parallelism and the measured per-node throughput. Secondary source.
- **NVIDIA (2025).** *NVIDIA Dynamo: a datacenter-scale distributed inference serving framework (GitHub).* https://github.com/ai-dynamo/dynamo  
  Disaggregated serving, KV-aware routing and multi-engine support (TensorRT-LLM, vLLM, SGLang). Secondary source.
- **NVIDIA (2025).** *TensorRT-LLM (GitHub).* https://github.com/NVIDIA/TensorRT-LLM  
  NVIDIA's compiled-kernel engine; FP8 and FP4 first, peak throughput on NVIDIA hardware. Secondary source.
- **Gerganov and contributors (2023).** *llama.cpp and the GGUF file format (GitHub).* https://github.com/ggml-org/llama.cpp  
  The C/C++ engine, ggml, GGUF, the quantization types (k-quants, i-quants), layer and expert offload, and llama-server. Secondary source.
- **Ollama (2025).** *Ollama (GitHub and docs).* https://github.com/ollama/ollama  
  The local daemon, model library, Modelfile and REST/OpenAI APIs; its own ggml-based engine for multimodal models. Secondary source.
- **Apple (2023).** *MLX: an array framework for Apple silicon (GitHub).* https://github.com/ml-explore/mlx  
  Unified-memory arrays and lazy evaluation; mlx-lm is the local LLM runtime on Macs. Secondary source.
- **Dettmers, Lewis, Belkada and Zettlemoyer (2022).** *LLM.int8(): 8-bit matrix multiplication for transformers at scale.* https://arxiv.org/abs/2208.07339  
  Outlier-aware 8-bit inference that made large models runnable on small GPUs; context for every local-serving quantization format.
- **Frantar, Castro, Chen, Hoefler and Alistarh (2024).** *MARLIN: mixed-precision auto-regressive parallel inference on large language models.* https://arxiv.org/abs/2408.11743  
  The 4-bit weight, 16-bit activation kernel that reaches near-ideal speedups at small batch; the kernel behind vLLM's GPTQ/AWQ serving.

## 4. Measuring and operating (slides 16-17) (slides 16-17)

- **MLCommons (2024).** *MLPerf Inference: datacenter benchmark suite (LLM tasks).* https://mlcommons.org/benchmarks/inference-datacenter/  
  Cross-vendor LLM serving benchmarks (Llama 2 70B, Llama 3.1 405B and later additions) with fixed accuracy and latency constraints. Secondary source.
- **Reddi et al. (2020).** *MLPerf Inference benchmark.* https://arxiv.org/abs/1911.02549  
  The design of the benchmark: scenarios (server, offline), latency constraints and accuracy targets; the methodology behind slide 16's protocol.
- **Shan et al. (the AIBrix team) (2025).** *AIBrix: towards scalable, cost-effective large language model inference infrastructure.* https://arxiv.org/abs/2504.03648  
  A Kubernetes-native control plane for LLM serving: routing, autoscaling, LoRA management and a distributed KV cache; ByteDance's open stack.
- **Li et al. (2023).** *AlpaServe: statistical multiplexing with model parallelism for deep learning serving.* https://arxiv.org/abs/2302.11665  
  Why model parallelism can help serving even when a model fits on one GPU: bursty traffic and statistical multiplexing. OSDI 2023.
- **Gao et al. (2024).** *Cost-efficient large language model serving for multi-turn conversations with CachedAttention.* https://arxiv.org/abs/2403.19708  
  Hierarchical KV caching across turns of a conversation (GPU, CPU, disk); the tiered cache of slide 29.
- **Liu et al. (2024).** *CacheGen: KV cache compression and streaming for fast large language model serving.* https://arxiv.org/abs/2310.07240  
  Compressing and streaming KV caches across the network so that a cold cache is cheaper to load than to recompute; the basis of LMCache.

## 5. The API layer, structured output, tool calling, reasoning (slides 18-21) (slides 18-21)

- **Willard and Louf (2023).** *Efficient guided generation for large language models.* https://arxiv.org/abs/2307.09702  
  Outlines: compile a regex or JSON schema to a finite automaton over the tokenizer and precompute the allowed-token mask per state.
- **Dong et al. (2024).** *XGrammar: flexible and efficient structured generation engine for large language models.* https://arxiv.org/abs/2411.15100  
  Pushdown automata for context-free grammars with context-independent token masks cached per state; near-zero per-step overhead; default in vLLM and SGLang.
- **Geng, Josifoski, Peyrard and West (2023).** *Grammar-constrained decoding for structured NLP tasks without finetuning.* https://arxiv.org/abs/2305.13971  
  Constrained decoding as a general method for structured tasks; the case for grammars over fine-tuning.
- **Tam et al. (2024).** *Let me speak freely? A study on the impact of format restrictions on performance of large language models.* https://arxiv.org/abs/2408.02442  
  Evidence that strict format constraints can lower reasoning accuracy; the caveat on slide 19 that masks change the distribution.
- **Schick et al. (2023).** *Toolformer: language models can teach themselves to use tools.* https://arxiv.org/abs/2302.04761  
  A model learns where to insert API calls by self-supervision; the origin of tool use as a learned behavior.
- **Patil, Zhang, Wang and Gonzalez (2023).** *Gorilla: large language model connected with massive APIs.* https://arxiv.org/abs/2305.15334  
  Fine-tuning for API calls with retrieval over documentation; the group behind the Berkeley Function Calling Leaderboard.
- **Patil et al. (2025).** *The Berkeley Function Calling Leaderboard (BFCL): from tool use to agentic evaluation of large language models.* https://proceedings.mlr.press/v267/patil25a.html  
  The standard tool-calling evaluation: single, parallel, multiple and multi-turn calls scored by AST match and execution. ICML 2025 (PMLR 267). Not on arXiv.
- **Yao, Shinn, Razavi and Narasimhan (2024).** *tau-bench: a benchmark for tool-agent-user interaction in real-world domains.* https://arxiv.org/abs/2406.12045  
  Simulated users and tool APIs over a database in retail and airline domains; the pass^k reliability metric.
- **Liu et al. (2024).** *APIGen: automated pipeline for generating verifiable and diverse function-calling datasets.* https://arxiv.org/abs/2406.18518  
  Execution-verified synthetic function-calling data; how open models learn the call format the runtime then parses.
- **Muennighoff et al. (2025).** *s1: simple test-time scaling.* https://arxiv.org/abs/2501.19393  
  Budget forcing: control thinking length by suppressing or inserting the end-of-thinking token; the mechanism behind runtime thinking budgets (slide 21).
- **Qwen Team (2025).** *Qwen3 technical report.* https://arxiv.org/abs/2505.09388  
  Thinking and non-thinking modes in one model, the thinking budget, and the /think and /no think template switches that runtimes expose.

## 6. Tool interfaces and the Model Context Protocol (slides 22-25) (slides 22-25)

- **Model Context Protocol (2025).** *Model Context Protocol specification (2025-06-18 revision and later).* https://modelcontextprotocol.io/specification/2025-06-18  
  The protocol: JSON-RPC lifecycle, server primitives (tools, resources, prompts), client primitives (sampling, roots, elicitation), transports, authorization. Secondary source (specification).
- **Anthropic (2024).** *Introducing the Model Context Protocol (announcement).* https://www.anthropic.com/news/model-context-protocol  
  The November 2024 announcement and the original motivation: one open protocol between applications and data sources or tools. Secondary source.
- **Model Context Protocol (2025).** *MCP Registry (GitHub).* https://github.com/modelcontextprotocol/registry  
  The public registry of MCP servers (preview, September 2025). Secondary source.
- **Model Context Protocol (2025).** *MCP authorization and security best practices (specification pages).* https://modelcontextprotocol.io/specification/2025-06-18/basic/security_best_practices  
  Confused deputy, token pass-through and session hijacking as the specification describes them, with the mitigations. Secondary source.
- **Hou, Zhao, Wang and Wang (2025).** *Model Context Protocol (MCP): landscape, security threats, and future research directions.* https://arxiv.org/abs/2503.23278  
  A survey of the protocol and a threat taxonomy across the server lifecycle: creation, operation, update.
- **Radosevich and Halloran (2025).** *MCP Safety Audit: LLMs with the Model Context Protocol allow major security exploits.* https://arxiv.org/abs/2504.03767  
  Demonstrated attacks through MCP servers (code execution, credential theft) and the MCPSafetyScanner auditing tool.
- **Narajala and Habler (2025).** *Enterprise-grade security for the Model Context Protocol (MCP): frameworks and mitigation strategies.* https://arxiv.org/abs/2504.08623  
  A threat model and control framework for deploying MCP in organizations: tool poisoning, injection, privilege, supply chain.
- **Greshake et al. (2023).** *Not what you've signed up for: compromising real-world LLM-integrated applications with indirect prompt injection.* https://arxiv.org/abs/2302.12173  
  Indirect prompt injection: instructions arriving through retrieved or tool-returned content; the attack on slide 25, before MCP existed.
- **Hasan et al. (2025).** *Model Context Protocol (MCP) at first glance: studying the security and maintainability of MCP servers.* https://arxiv.org/abs/2506.13538  
  An empirical study of 1,899 open-source MCP servers: 7.2% with general vulnerabilities, 5.5% with MCP-specific tool poisoning, widespread code smells.
- **Wang et al. (2025).** *MCPTox: a benchmark for tool poisoning attack on real-world MCP servers.* https://arxiv.org/abs/2508.14925  
  1,348 tool-poisoning cases over 45 real servers and 353 tools; attack success above 70% for some agents and refusal rates under 3%.
- **Wang et al. (2025).** *MCP-Bench: benchmarking tool-using LLM agents with complex real-world tasks via MCP servers.* https://arxiv.org/abs/2508.20453  
  Agent evaluation over 28 live MCP servers and 250 tools; how well models plan and chain real tools.
- **Invariant Labs (2025).** *MCP security notification: tool poisoning attacks (blog).* https://invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks  
  The tool-poisoning, rug-pull and cross-server shadowing attacks on MCP, with demonstrations. Secondary source.
- **Wang et al. (2024).** *Executable code actions elicit better LLM agents.* https://arxiv.org/abs/2402.01030  
  CodeAct: the model's actions are Python code run in a sandbox; one tool with an unbounded surface, the second row of slide 22.
- **OWASP (2025).** *OWASP Top 10 for LLM applications (2025).* https://genai.owasp.org/llm-top-10/  
  Prompt injection, insecure output handling, excessive agency and supply-chain risks as a checklist for agent deployments. Secondary source.

## 7. Web search and retrieval (slides 26-27) (slides 26-27)

- **Nakano et al. (2021).** *WebGPT: browser-assisted question-answering with human feedback.* https://arxiv.org/abs/2112.09332  
  A text browser as the tool, imitation learning then RL from human preferences on cited answers; the origin of search-as-tool.
- **Vu et al. (2023).** *FreshLLMs: refreshing large language models with search engine augmentation.* https://arxiv.org/abs/2310.03214  
  FreshQA: models are confidently stale on facts that change; search-augmented prompting with dates fixes much of it.
- **Jin et al. (2025).** *Search-R1: training LLMs to reason and leverage search engines with reinforcement learning.* https://arxiv.org/abs/2503.09516  
  RL with outcome rewards where the policy issues search calls mid-reasoning; retrieved tokens are masked from the loss.
- **Chen et al. (2025).** *ReSearch: learning to reason with search for LLMs via reinforcement learning.* https://arxiv.org/abs/2503.19470  
  The same recipe with reasoning-and-search interleaved as a single trajectory; reflection and self-correction emerge.
- **Song et al. (2025).** *R1-Searcher: incentivizing the search capability in LLMs via reinforcement learning.* https://arxiv.org/abs/2503.05592  
  Two-stage outcome-based RL for search invocation without distillation data.
- **Wei et al. (2025).** *BrowseComp: a simple yet challenging benchmark for browsing agents.* https://arxiv.org/abs/2504.12516  
  1,266 hard-to-find, easy-to-verify questions; the benchmark for deep-research agents.
- **Lewis et al. (2020).** *Retrieval-augmented generation for knowledge-intensive NLP tasks.* https://arxiv.org/abs/2005.11401  
  RAG: a retriever and a generator trained together; the pattern behind every retrieval pipeline on slide 27.
- **Karpukhin et al. (2020).** *Dense passage retrieval for open-domain question answering.* https://arxiv.org/abs/2004.04906  
  Dense dual-encoder retrieval beating BM25; the embedding step of the pipeline.
- **Khattab and Zaharia (2020).** *ColBERT: efficient and effective passage search via contextualized late interaction over BERT.* https://arxiv.org/abs/2004.12832  
  Late interaction: token-level matching at retrieval cost; the design behind the strongest rerankers and retrievers.
- **Li, Li, Zhang, Mei and Bendersky (2024).** *Retrieval augmented generation or long-context LLMs? A comprehensive study and hybrid approach.* https://arxiv.org/abs/2407.16833  
  With enough window long context wins on quality, RAG wins on cost; Self-Route picks per query. The evidence on slide 27.
- **Liu et al. (2023).** *Lost in the middle: how language models use long contexts.* https://arxiv.org/abs/2307.03172  
  Information in the middle of a long context is used less reliably; why chunk order matters.
- **Anthropic (2024).** *Introducing contextual retrieval (blog).* https://www.anthropic.com/news/contextual-retrieval  
  Prepending document context to chunks before embedding and BM25 indexing; reported reductions in retrieval failure. Secondary source.
- **Asai et al. (2023).** *Self-RAG: learning to retrieve, generate, and critique through self-reflection.* https://arxiv.org/abs/2310.11511  
  A model that decides when to retrieve and critiques its own use of the passages; adaptive retrieval before agentic search.

## 8. Agent loops and serving agents (slides 28-29) (slides 28-29)

- **Yao et al. (2022).** *ReAct: synergizing reasoning and acting in language models.* https://arxiv.org/abs/2210.03629  
  Interleaved reasoning traces and actions in one loop; the template for every agent runtime.
- **Yang et al. (2024).** *SWE-agent: agent-computer interfaces enable automated software engineering.* https://arxiv.org/abs/2405.15793  
  A purpose-built command interface for the model (file viewer, search, edit) matters as much as the model; the ACI idea.
- **Wang et al. (2024).** *OpenHands: an open platform for AI software developers as generalist agents.* https://arxiv.org/abs/2407.16741  
  An open agent platform: sandboxed runtime, event stream, multi-agent delegation; what an agent runtime contains.
- **Jimenez et al. (2023).** *SWE-bench: can language models resolve real-world GitHub issues?.* https://arxiv.org/abs/2310.06770  
  Real repository issues with hidden tests; the standard environment and reward for coding agents.
- **Anthropic (2024).** *Building effective agents (blog).* https://www.anthropic.com/research/building-effective-agents  
  Workflows versus agents, the augmented LLM, and the case for simple loops with good tools. Secondary source.
- **Anthropic (2025).** *Effective context engineering for AI agents (blog).* https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents  
  Compaction, structured note-taking, sub-agents and just-in-time context; the context-management practices of slide 29. Secondary source.
- **Packer et al. (2023).** *MemGPT: towards LLMs as operating systems.* https://arxiv.org/abs/2310.08560  
  Treating the context window as main memory with paging to external storage; an early framing of context management as a runtime problem.
- **Google (2025).** *Agent2Agent (A2A) protocol (GitHub).* https://github.com/a2aproject/A2A  
  Agent discovery by capability cards and task exchange between agents; donated to the Linux Foundation in 2025. Secondary source.
