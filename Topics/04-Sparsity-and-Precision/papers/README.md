# Reading list: Sparsity and Precision, Visually

Companion papers to the 33-slide deck and its lecture notes, grouped by section.
`reading_list.csv` and `reading_list.bib` hold the same list as data; `fetch_papers.sh` downloads
every arXiv entry into one folder per section and zips the result (`bash fetch_papers.sh`).
Entries without an arXiv identifier are listed with a URL and must be fetched by hand.

## 1. Overview and the cost model (slides 1-3)

- **Williams, Waterman and Patterson (2009).** *Roofline: an insightful visual performance model for multicore architectures.* https://doi.org/10.1145/1498765.1498785  
  The model behind slide 2: attainable throughput against arithmetic intensity. Communications of the ACM 52(4).
- **Kaplan et al. (2020).** *Scaling laws for neural language models.* https://arxiv.org/abs/2001.08361  
  Source of the 6ND compute count and the loss-versus-resource power laws the precision law extends.
- **Hoffmann et al. (2022).** *Training compute-optimal large language models.* https://arxiv.org/abs/2203.15556  
  The parametric loss form L = A/N^a + B/D^b + E into which effective parameters are substituted on slide 25.
- **Pope et al. (2022).** *Efficiently scaling transformer inference.* https://arxiv.org/abs/2211.05102  
  The clearest published roofline analysis of prefill and decode, batching, and the KV cache; the quantitative background of slides 2 and 15.
- **Kwon et al. (2023).** *Efficient memory management for large language model serving with PagedAttention.* https://arxiv.org/abs/2309.06180  
  vLLM: how a serving stack manages the KV cache; the system in which most quantization formats on slide 27 are deployed.

## 2. Mixture of experts (slides 4-9)

- **Jacobs, Jordan, Nowlan and Hinton (1991).** *Adaptive mixtures of local experts.* https://doi.org/10.1162/neco.1991.3.1.79  
  The origin of the idea: a gating network that assigns inputs to expert networks. Neural Computation 3(1).
- **Shazeer et al. (2017).** *Outrageously large neural networks: the sparsely-gated mixture-of-experts layer.* https://arxiv.org/abs/1701.06538  
  Sparse top-k gating at scale (in an LSTM LM); introduces the load-balancing losses every later paper inherits.
- **Lepikhin et al. (2020).** *GShard: scaling giant models with conditional computation and automatic sharding.* https://arxiv.org/abs/2006.16668  
  Top-2 MoE in transformers at 600B parameters; expert parallelism and capacity factors as used since.
- **Fedus, Zoph and Shazeer (2021).** *Switch Transformers: scaling to trillion parameter models with simple and efficient sparsity.* https://arxiv.org/abs/2101.03961  
  Top-1 routing, the auxiliary loss with alpha = 0.01, and the 7x pretraining speedup; the reference MoE recipe of 2021.
- **Du et al. (2021).** *GLaM: efficient scaling of language models with mixture-of-experts.* https://arxiv.org/abs/2112.06905  
  1.2T parameters, 64 experts, top-2, trained with a third of GPT-3's energy; early evidence that MoE quality tracks total parameters.
- **Zoph et al. (2022).** *ST-MoE: designing stable and transferable sparse expert models.* https://arxiv.org/abs/2202.08906  
  The router z-loss and a careful study of MoE training stability and fine-tuning.
- **Zhou et al. (2022).** *Mixture-of-experts with expert choice routing.* https://arxiv.org/abs/2202.09368  
  Experts pick tokens instead of tokens picking experts: perfect balance by construction.
- **Clark et al. (2022).** *Unified scaling laws for routed language models.* https://arxiv.org/abs/2202.01169  
  The first MoE scaling law; recommends 64-128 experts at the scales studied and quantifies diminishing returns.
- **Fedus, Dean and Zoph (2022).** *A review of sparse expert models in deep learning.* https://arxiv.org/abs/2209.01667  
  A survey of the MoE literature up to 2022, useful for the history and the terminology.
- **Gale et al. (2022).** *MegaBlocks: efficient sparse training with mixture-of-experts.* https://arxiv.org/abs/2211.15841  
  Dropless MoE: block-sparse kernels that let experts take variable-size batches, no token dropping.
- **Jiang et al. (2024).** *Mixtral of experts.* https://arxiv.org/abs/2401.04088  
  The open 8-expert top-2 model (47B total, 13B active) that made MoE mainstream in open weights.
- **Dai et al. (2024).** *DeepSeekMoE: towards ultimate expert specialization in mixture-of-experts language models.* https://arxiv.org/abs/2401.06066  
  Fine-grained expert segmentation and shared experts: the design of every DeepSeek MoE and most 2025 open models.
- **Krajewski et al. (2024).** *Scaling laws for fine-grained mixture of experts.* https://arxiv.org/abs/2402.07871  
  Granularity as a scaling-law variable; MoE beats dense at every scale with granularity tuned.
- **DeepSeek-AI (2024).** *DeepSeek-V2: a strong, economical, and efficient mixture-of-experts language model.* https://arxiv.org/abs/2405.04434  
  Introduces multi-head latent attention (93.3% KV-cache reduction) with a 236B/21B MoE.
- **Wang et al. (2024).** *Auxiliary-loss-free load balancing strategy for mixture-of-experts.* https://arxiv.org/abs/2408.15664  
  The per-expert bias update used by DeepSeek-V3 to balance load without a loss term.
- **DeepSeek-AI (2024).** *DeepSeek-V3 technical report.* https://arxiv.org/abs/2412.19437  
  671B/37B MoE, MLA, aux-loss-free balancing, DualPipe, and the FP8 training recipe of slide 23; the central reference of the deck.
- **Abnar et al. (2025).** *Parameters vs FLOPs: scaling laws for optimal sparsity for mixture-of-experts language models.* https://arxiv.org/abs/2501.12370  
  Total parameters and FLOPs as separate axes; optimal sparsity rises with model size at fixed compute.
- **Ludziejewski et al. (2025).** *Joint MoE scaling laws: mixture of experts can be memory efficient.* https://arxiv.org/abs/2502.05172  
  Adds memory to the objective; MoE can be the memory-efficient choice too.
- **Qwen Team (2025).** *Qwen3 technical report.* https://arxiv.org/abs/2505.09388  
  235B/22B MoE with 128 experts and no shared expert; thinking budgets; strong-to-weak distillation.
- **Kimi Team (2025).** *Kimi K2: open agentic intelligence.* https://arxiv.org/abs/2507.20534  
  1.04T/32B MoE with 384 experts, MLA and the MuonClip optimizer.
- **OpenAI (2025).** *gpt-oss-120b and gpt-oss-20b model card.* https://arxiv.org/abs/2508.10925  
  128-expert top-4 MoE shipped with MXFP4 expert weights; 117B on one 80 GB GPU.
- **DeepSeek-AI (2026).** *DeepSeek-V4: towards highly efficient million-token context intelligence.* https://arxiv.org/abs/2606.19348  
  1.6T/49B and 284B/13B MoEs with compressed sparse attention and FP4 expert weights (2026).
- **Meta AI (2025).** *The Llama 4 herd (blog post).* https://ai.meta.com/blog/llama-4-multimodal-intelligence/  
  Llama 4 Maverick (400B/17B, 128 experts + shared, top-1) and Scout; alternating dense and MoE layers. Secondary source.
- **Qwen Team (2025).** *Qwen3-Next-80B-A3B model card.* https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct  
  512 experts, 10 routed + 1 shared, 80B/3B, hybrid Gated DeltaNet attention. Secondary source.

## 3. Weight pruning (slides 10-13)

- **LeCun, Denker and Solla (1989).** *Optimal brain damage.* https://proceedings.neurips.cc/paper_files/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf  
  Second-order (diagonal Hessian) saliency for pruning. NeurIPS 1989.
- **Hassibi and Stork (1992).** *Second order derivatives for network pruning: Optimal Brain Surgeon.* https://proceedings.neurips.cc/paper_files/paper/1992/file/303ed4c69846ab36c2904d3ba8573050-Paper.pdf  
  Full inverse-Hessian saliency with the compensating update; the ancestor of SparseGPT and GPTQ. NeurIPS 1992.
- **Han, Pool, Tran and Dally (2015).** *Learning both weights and connections for efficient neural networks.* https://arxiv.org/abs/1506.02626  
  Iterative magnitude pruning with retraining: 9x on AlexNet, 13x on VGG-16, no accuracy loss.
- **Han, Mao and Dally (2016).** *Deep compression: compressing deep neural networks with pruning, trained quantization and Huffman coding.* https://arxiv.org/abs/1510.00149  
  Pruning plus quantization plus coding, 35-49x smaller; the first paper to compose the two levers.
- **Frankle and Carbin (2019).** *The lottery ticket hypothesis: finding sparse, trainable neural networks.* https://arxiv.org/abs/1803.03635  
  Sparse subnetworks at 10-20% of the size train to full accuracy from their original initialization.
- **Hoefler et al. (2021).** *Sparsity in deep learning: pruning and growth for efficient inference and training in neural networks.* https://arxiv.org/abs/2102.00554  
  A comprehensive survey of sparsity methods and hardware; the reference map of the field before LLMs.
- **Mishra et al. (2021).** *Accelerating sparse deep neural networks.* https://arxiv.org/abs/2104.08378  
  The 2:4 pattern, its compressed format, and the train-prune-retrain recipe for Ampere sparse tensor cores.
- **Frantar and Alistarh (2023).** *SparseGPT: massive language models can be accurately pruned in one-shot.* https://arxiv.org/abs/2301.00774  
  One-shot OBS-style layer-wise pruning of 175B models to 50-60% in hours on one GPU.
- **Sun et al. (2023).** *A simple and effective pruning approach for large language models.* https://arxiv.org/abs/2306.11695  
  Wanda: prune by |weight| x activation norm with no weight updates; nearly matches SparseGPT at 50%.
- **Frantar et al. (2023).** *Scaling laws for sparsely-connected foundation models.* https://arxiv.org/abs/2309.08520  
  A joint law in sparsity, parameters and data; optimal sparsity rises with training data per parameter.
- **Xia et al. (2023).** *Sheared LLaMA: accelerating language model pre-training via structured pruning.* https://arxiv.org/abs/2310.06694  
  LLaMA-2 7B to 2.7B/1.3B with learned masks and 50B tokens of continued pretraining, 3% of from-scratch compute.
- **Ma, Fang and Wang (2023).** *LLM-Pruner: on the structural pruning of large language models.* https://arxiv.org/abs/2305.11627  
  Gradient-based structured pruning of LLMs with LoRA recovery; an early LLM structured-pruning baseline.
- **Ashkboos et al. (2024).** *SliceGPT: compress large language models by deleting rows and columns.* https://arxiv.org/abs/2401.15024  
  Structured pruning by orthogonal transformation then slicing; introduces the computational-invariance idea reused in QuaRot.
- **Muralidharan et al. (2024).** *Compact language models via pruning and knowledge distillation.* https://arxiv.org/abs/2407.14679  
  Minitron: Nemotron-4 15B to 8B/4B with up to 40x fewer tokens; width beats depth; distillation beats fine-tuning.
- **Sreenivas et al. (2024).** *LLM pruning and distillation in practice: the Minitron approach.* https://arxiv.org/abs/2408.11796  
  The recipe applied to Llama 3.1 8B and Mistral NeMo 12B, with teacher correction.

## 4. Activation and attention sparsity (slides 14-17)

- **Li et al. (2022).** *The lazy neuron phenomenon: on emergence of activation sparsity in transformers.* https://arxiv.org/abs/2210.06313  
  Measures the emergent sparsity of ReLU MLPs (3% nonzero in T5-Base) and shows it grows with scale.
- **Mirzadeh et al. (2023).** *ReLU strikes back: exploiting activation sparsity in large language models.* https://arxiv.org/abs/2310.04564  
  ReLUfication of LLMs: up to 3x less inference computation with little quality loss.
- **Liu et al. (2023).** *Deja Vu: contextual sparsity for efficient LLMs at inference time.* https://arxiv.org/abs/2310.17157  
  Predict the live neurons and heads per input; 2x over FasterTransformer on OPT-175B.
- **Song et al. (2023).** *PowerInfer: fast large language model serving with a consumer-grade GPU.* https://arxiv.org/abs/2312.12456  
  Hot/cold neuron partitioning between GPU and CPU using activation sparsity; the systems follow-up to Deja Vu.
- **Lee et al. (2024).** *CATS: contextually-aware thresholding for sparsity in large language models.* https://arxiv.org/abs/2404.08763  
  Threshold the gated MLP for 50% sparsity within 1-2% of the base model, training-free.
- **Liu et al. (2024).** *Training-free activation sparsity in large language models.* https://arxiv.org/abs/2408.14690  
  TEAL: magnitude thresholds model-wide, 40-50% sparsity, 1.5-1.8x decode speedup.
- **Child et al. (2019).** *Generating long sequences with sparse transformers.* https://arxiv.org/abs/1904.10509  
  Strided and fixed sparse attention patterns, O(n sqrt n); the first sparse-attention transformer.
- **Beltagy, Peters and Cohan (2020).** *Longformer: the long-document transformer.* https://arxiv.org/abs/2004.05150  
  Sliding window plus global tokens, linear in length.
- **Zaheer et al. (2020).** *Big Bird: transformers for longer sequences.* https://arxiv.org/abs/2007.14062  
  Window + global + random blocks; shows the sparse pattern is a universal approximator.
- **Shazeer (2019).** *Fast transformer decoding: one write-head is all you need.* https://arxiv.org/abs/1911.02150  
  Multi-query attention: share keys and values across heads to shrink the cache.
- **Ainslie et al. (2023).** *GQA: training generalized multi-query transformer models from multi-head checkpoints.* https://arxiv.org/abs/2305.13245  
  Grouped-query attention, the compromise used by Llama 3 and most 2024 models.
- **Dao et al. (2022).** *FlashAttention: fast and memory-efficient exact attention with IO-awareness.* https://arxiv.org/abs/2205.14135  
  Exact attention tiled to avoid materializing the L x L matrix; the kernel every sparse-attention speedup is measured against.
- **Jiang et al. (2023).** *Mistral 7B.* https://arxiv.org/abs/2310.06825  
  Sliding-window attention (4,096 tokens) with a rolling cache, plus GQA, in a widely used open model.
- **Zhang et al. (2023).** *H2O: heavy-hitter oracle for efficient generative inference of large language models.* https://arxiv.org/abs/2306.14048  
  KV eviction keeping the tokens with the most accumulated attention; 20% budgets retain accuracy.
- **Xiao et al. (2023).** *Efficient streaming language models with attention sinks.* https://arxiv.org/abs/2309.17453  
  Attention sinks: keep the first four tokens plus a window and stream millions of tokens.
- **Li et al. (2024).** *SnapKV: LLM knows what you are looking for before generation.* https://arxiv.org/abs/2404.14469  
  Compress the prompt's KV cache per head from the attention of the last prompt tokens.
- **Tang et al. (2024).** *Quest: query-aware sparsity for efficient long-context LLM inference.* https://arxiv.org/abs/2406.10774  
  Per-query page selection by score bounds; selection without a learned indexer.
- **Liu et al. (2024).** *KIVI: a tuning-free asymmetric 2bit quantization for KV cache.* https://arxiv.org/abs/2402.02750  
  Keys per channel, values per token: 2-bit KV cache with small loss.
- **Yuan et al. (2025).** *Native sparse attention: hardware-aligned and natively trainable sparse attention.* https://arxiv.org/abs/2502.11089  
  Compressed, selected and sliding-window branches, trained from scratch; 9x forward and 6x backward kernel speedups at 64k.
- **Lu et al. (2025).** *MoBA: mixture of block attention for long-context LLMs.* https://arxiv.org/abs/2502.13189  
  Route each query to its top-k blocks of the cache, as a router routes tokens to experts.
- **DeepSeek-AI (2025).** *DeepSeek-V3.2: pushing the frontier of open large language models.* https://arxiv.org/abs/2512.02556  
  DeepSeek Sparse Attention: a lightning indexer selecting the top 2048 tokens per query on top of MLA.
- **Hsieh et al. (2024).** *RULER: what's the real context size of your long-context language models?.* https://arxiv.org/abs/2404.06654  
  Thirteen long-context tasks in four categories; most models fall short of their claimed lengths.

## 5. Number formats and outliers (slides 18-21)

- **Micikevicius et al. (2022).** *FP8 formats for deep learning.* https://arxiv.org/abs/2209.05433  
  Defines E4M3 and E5M2, their special values and the rationale; the format reference for slide 18.
- **Rouhani et al. (2023).** *Microscaling data formats for deep learning.* https://arxiv.org/abs/2310.10537  
  The MX block formats (32-element blocks, E8M0 scale) with training and inference results; companion to the OCP spec.
- **Open Compute Project (2023).** *OCP Microscaling Formats (MX) specification v1.0.* https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf  
  The normative definition of MXFP8/6/4 and MXINT8. Not on arXiv.
- **NVIDIA (2025).** *Introducing NVFP4 for efficient and accurate low-precision inference (developer blog).* https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/  
  E2M1 elements, 16-element blocks with E4M3 scales, FP32 per-tensor scale. Secondary source.
- **Dettmers et al. (2022).** *LLM.int8(): 8-bit matrix multiplication for transformers at scale.* https://arxiv.org/abs/2208.07339  
  Discovers the outlier features that emerge around 6.7B parameters and the mixed-precision decomposition that handles them.
- **Bondarenko, Nagel and Blankevoort (2021).** *Understanding and overcoming the challenges of efficient transformer quantization.* https://arxiv.org/abs/2109.12948  
  Early analysis of why transformer activations (and their outliers) resist INT8, with per-embedding-group fixes.
- **Xiao et al. (2022).** *SmoothQuant: accurate and efficient post-training quantization for large language models.* https://arxiv.org/abs/2211.10438  
  Migrate activation difficulty into the weights with per-channel scales; W8A8 up to 1.56x faster.
- **Sun et al. (2024).** *Massive activations in large language models.* https://arxiv.org/abs/2402.17762  
  A handful of activations at specific tokens act as fixed biases and attention sinks; the mechanism behind the outliers.
- **Ashkboos et al. (2024).** *QuaRot: outlier-free 4-bit inference in rotated LLMs.* https://arxiv.org/abs/2404.00456  
  Hadamard rotations folded into the weights; 4-bit weights, activations and KV cache with at most 0.47 perplexity loss on Llama-2 70B.
- **Liu et al. (2024).** *SpinQuant: LLM quantization with learned rotations.* https://arxiv.org/abs/2405.16406  
  Learned rotations on the Stiefel manifold; used in Meta's Llama 3.2 quantized releases.
- **Dettmers et al. (2023).** *QLoRA: efficient finetuning of quantized LLMs.* https://arxiv.org/abs/2305.14314  
  NF4, double quantization and paged optimizers; fine-tune a 65B model on one 48 GB GPU.

## 6. Low-precision training (slides 22-24)

- **Micikevicius et al. (2018).** *Mixed precision training.* https://arxiv.org/abs/1710.03740  
  FP32 master weights and loss scaling: the recipe that made FP16 training work and that every later format inherits.
- **Kalamkar et al. (2019).** *A study of BFLOAT16 for deep learning training.* https://arxiv.org/abs/1905.12322  
  Why BF16's FP32 range removes the need for loss scaling at the cost of mantissa bits.
- **Rajbhandari et al. (2020).** *ZeRO: memory optimizations toward training trillion parameter models.* https://arxiv.org/abs/1910.02054  
  The 16-bytes-per-parameter accounting of mixed-precision training and how to partition it.
- **Peng et al. (2023).** *FP8-LM: training FP8 large language models.* https://arxiv.org/abs/2310.18313  
  FP8 weights, gradients, optimizer states and communication in GPT-175B training; an early full-stack FP8 recipe.
- **Fishman et al. (2024).** *Scaling FP8 training to trillion-token LLMs.* https://arxiv.org/abs/2409.12517  
  Instabilities that appear only after hundreds of billions of tokens in FP8 (SwiGLU outlier amplification) and their fix.
- **Xi et al. (2023).** *Training transformers with 4-bit integers.* https://arxiv.org/abs/2306.11987  
  INT4 forward and backward with Hadamard quantizers and leverage-score sampling; a precursor of the FP4 recipes.
- **Tseng et al. (2025).** *Training LLMs with MXFP4.* https://arxiv.org/abs/2502.20586  
  Stochastic rounding plus random Hadamard transforms for near-lossless MXFP4 training; more than half the FLOPs in MXFP4.
- **NVIDIA (2025).** *Recipes for pre-training LLMs with MXFP8.* https://arxiv.org/abs/2506.08027  
  Round the block scale up, not down; an 8B model on 15T tokens matches BF16.
- **NVIDIA (2025).** *Pretraining large language models with NVFP4.* https://arxiv.org/abs/2509.25149  
  12B on 10T tokens in NVFP4: 2D block scales, Hadamard transforms, stochastic rounding, 16% of layers kept higher.
- **Castro et al. (2025).** *Quartet: native FP4 training can be optimal for large language models.* https://arxiv.org/abs/2505.14669  
  End-to-end FP4 training with Blackwell kernels, framed with an accuracy-versus-efficiency law.
- **NVIDIA (2026).** *Nemotron 3 Ultra technical report.* https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf  
  550B/55B model pretrained in NVFP4 on 20T tokens with a loss gap under 0.4% (2026). Secondary source; not on arXiv.

## 7. Scaling laws for precision (slide 25)

- **Kumar et al. (2024).** *Scaling laws for precision.* https://arxiv.org/abs/2411.04330  
  Effective parameters N(1 - e^{-P/gamma}), the 7-8 bit compute-optimal precision, and PTQ degradation growing with data.
- **Dettmers and Zettlemoyer (2023).** *The case for 4-bit precision: k-bit inference scaling laws.* https://arxiv.org/abs/2212.09720  
  35,000 experiments from 19M to 176B parameters: 4-bit weights are almost universally optimal per total model bit.
- **Ouyang et al. (2024).** *Low-bit quantization favors undertrained LLMs: scaling laws for quantized LLMs with 100T training tokens.* https://arxiv.org/abs/2411.17691  
  Independent confirmation that quantization damage grows with tokens per parameter.
- **Sun et al. (2025).** *Scaling laws for floating-point quantization training.* https://arxiv.org/abs/2501.02423  
  Exponent and mantissa bits as separate variables; cost-optimal precision between 4 and 8 bits.

## 8. Quantization for inference (slides 26-29)

- **Gholami et al. (2021).** *A survey of quantization methods for efficient neural network inference.* https://arxiv.org/abs/2103.13630  
  The standard survey of quantization concepts (uniform/non-uniform, symmetric/asymmetric, PTQ/QAT) before the LLM era.
- **Frantar and Alistarh (2022).** *Optimal brain compression: a framework for accurate post-training quantization and pruning.* https://arxiv.org/abs/2208.11580  
  OBQ/OBC: the per-weight quantize-and-compensate method that GPTQ and SparseGPT scale up.
- **Frantar et al. (2022).** *GPTQ: accurate post-training quantization for generative pre-trained transformers.* https://arxiv.org/abs/2210.17323  
  OPT-175B to 3-4 bits in four GPU-hours; column-wise error feedback with a shared inverse Hessian.
- **Lin et al. (2023).** *AWQ: activation-aware weight quantization for LLM compression and acceleration.* https://arxiv.org/abs/2306.00978  
  Protect the 1% salient channels by scaling before rounding; 4-bit weights, no backprop, and the TinyChat kernels.
- **Kim et al. (2023).** *SqueezeLLM: dense-and-sparse quantization.* https://arxiv.org/abs/2306.07629  
  Sensitivity-weighted non-uniform quantization with outliers kept sparse; another route to 3-4 bits.
- **Chee et al. (2023).** *QuIP: 2-bit quantization of large language models with guarantees.* https://arxiv.org/abs/2307.13304  
  Incoherence processing by random orthogonal matrices with a theoretical bound; the start of the rotation line.
- **Tseng et al. (2024).** *QuIP#: even better LLM quantization with Hadamard incoherence and lattice codebooks.* https://arxiv.org/abs/2402.04396  
  Hadamard incoherence plus E8 lattice codebooks for 2-4 bit weights.
- **Egiazarian et al. (2024).** *Extreme compression of large language models via additive quantization.* https://arxiv.org/abs/2401.06118  
  AQLM: additive codebooks; the first Pareto-optimal scheme below 3 bits per weight.
- **Frantar et al. (2024).** *MARLIN: mixed-precision auto-regressive parallel inference on large language models.* https://arxiv.org/abs/2408.11743  
  The W4A16 kernel that reaches near-ideal 4x speedup up to batch 16-32.
- **Zhao et al. (2023).** *Atom: low-bit quantization for efficient and accurate LLM serving.* https://arxiv.org/abs/2310.19102  
  W4A4 serving with outlier channels in INT8; an early activation-quantized serving design.
- **Lin et al. (2024).** *QServe: W4A8KV4 quantization and system co-design for efficient LLM serving.* https://arxiv.org/abs/2405.04532  
  4-bit weights, 8-bit activations, 4-bit KV cache with register-level dequantization; 1.2-3.5x over TensorRT-LLM.
- **Liu et al. (2023).** *LLM-QAT: data-free quantization aware training for large language models.* https://arxiv.org/abs/2305.17888  
  QAT for LLMs using self-generated data; weights, activations and KV cache to 4 bits.
- **Wang et al. (2023).** *BitNet: scaling 1-bit transformers for large language models.* https://arxiv.org/abs/2310.11453  
  Binary weights from scratch with the straight-through estimator; the precursor of b1.58.
- **Ma et al. (2024).** *The era of 1-bit LLMs: all large language models are in 1.58 bits.* https://arxiv.org/abs/2402.17764  
  Ternary weights {-1,0,1} from pretraining; matches FP16 LLaMA from 3B parameters at the same tokens.
- **Ma et al. (2025).** *BitNet b1.58 2B4T technical report.* https://arxiv.org/abs/2504.12285  
  A 2B-parameter ternary model trained on 4T tokens, with CPU and GPU kernels.
- **Liu et al. (2025).** *Quantization hurts reasoning? An empirical study on quantized reasoning models.* https://arxiv.org/abs/2504.04823  
  Measures the damage of weight, activation and KV quantization on reasoning models over long generations; evidence for slide 29.
- **Meta AI (2024).** *Introducing quantized Llama models with increased speed and a reduced memory footprint (blog post).* https://ai.meta.com/blog/meta-llama-quantized-lightweight-models/  
  Llama 3.2 1B/3B with QAT+LoRA and SpinQuant: 2-4x faster, 56% smaller, 41% less memory. Secondary source.

## 9. Practice: models and systems (slides 30-31)

- **Grattafiori et al. (2024).** *The Llama 3 herd of models.* https://arxiv.org/abs/2407.21783  
  The dense 405B reference point of slides 9 and 30: GQA, 126 layers, BF16 and FP8 variants.
- **NVIDIA (2022).** *Transformer Engine documentation: FP8 primer and recipes.* https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html  
  Delayed and current scaling, the hybrid E4M3/E5M2 recipe, amax history. Secondary source.
- **NVIDIA and AMD (2020).** *GPU datasheets: A100, H100, HGX B200, MI300X, MI355X.* https://www.nvidia.com/en-us/data-center/  
  The peak-throughput and bandwidth figures of slide 21; dense values are half the 'with sparsity' headlines. Secondary source.
- **Moonshot AI (2026).** *Kimi K3 model card.* https://huggingface.co/moonshotai/Kimi-K3  
  2.8T/104B with 16 of 896 experts plus 2 shared; MXFP4/MXFP8 QAT from SFT onward (2026). Secondary source.
