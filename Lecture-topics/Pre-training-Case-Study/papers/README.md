# Reading list: Frontier Pre-training, Visually

Companion papers to the deck and its lecture notes, grouped by section.
`reading_list.csv` and `reading_list.bib` hold the same list as data; `fetch_papers.sh` downloads
every arXiv entry into one folder per section and zips the result (`bash fetch_papers.sh`).
Entries without an arXiv identifier are listed with a URL and must be fetched by hand.

## 1. Overview and results (slides 1-5b)

- **Microsoft AI (2026).** *Frontier pre-training at Microsoft AI (MAI-Base-2 report).* https://microsoft.ai/news/  
  The document the whole deck follows: MAI-Base-2's architecture, the scaling ladders, hyperparameter laws, MoE and Muon systems work, clipping, and the data mixture; Figures 1-10 are reproduced from it. Not on arXiv; the PDF was supplied for the lecture.
- **Chowdhery et al. (2022).** *PaLM: scaling language modeling with Pathways.* https://arxiv.org/abs/2204.02311  
  Defines model FLOPs utilization (MFU), the 21% figure the report quotes; Section 5.1 also describes restarting around loss spikes, the practice MAI-Base-2 avoided.
- **Deletang et al. (2023).** *Language modeling is compression.* https://arxiv.org/abs/2309.10668  
  Why a language model's log-loss is a compression rate: the arithmetic-coding argument behind reading bits per byte as a compression score.

## 2. The model (slides 6-9)

- **Shazeer et al. (2017).** *Outrageously large neural networks: the sparsely-gated mixture-of-experts layer.* https://arxiv.org/abs/1701.06538  
  Sparse top-k gating at scale: a learned router makes a layer's cost independent of its parameter count, and introduces the load-balancing losses later work inherits.
- **Lepikhin et al. (2020).** *GShard: scaling giant models with conditional computation and automatic sharding.* https://arxiv.org/abs/2006.16668  
  Top-2 MoE in transformers at 600B parameters; expert parallelism, capacity factors and token dropping as used since.
- **Fedus, Zoph and Shazeer (2021).** *Switch Transformers: scaling to trillion parameter models with simple and efficient sparsity.* https://arxiv.org/abs/2101.03961  
  Top-1 routing, the auxiliary load-balancing loss with its 0.01 weight, and the first trillion-parameter sparse model.
- **Dai et al. (2024).** *DeepSeekMoE: towards ultimate expert specialization in mixture-of-experts language models.* https://arxiv.org/abs/2401.06066  
  Fine-grained expert segmentation and shared experts, the design MAI-Base-2's 1 shared + 7 of 512 layout follows.
- **Ainslie et al. (2023).** *GQA: training generalized multi-query transformer models from multi-head checkpoints.* https://arxiv.org/abs/2305.13245  
  Grouped-query attention: query heads share key/value heads, shrinking the KV cache by the group size; MAI-Base-2 uses 48 queries per KV head in global layers and 6 in local ones.
- **Press and Wolf (2017).** *Using the output embedding to improve language models.* https://arxiv.org/abs/1608.05859  
  Tied input and output embeddings, the default MAI-Base-1 used and MAI-Base-2 abandoned.

## 3. Scaling ladders (slides 10-14)

- **Kaplan et al. (2020).** *Scaling laws for neural language models.* https://arxiv.org/abs/2001.08361  
  Power laws in parameters, data and compute over seven orders of magnitude, and the 6ND cost model behind every FLOPs number in the deck.
- **Hoffmann et al. (2022).** *Training compute-optimal large language models.* https://arxiv.org/abs/2203.15556  
  Chinchilla: the parametric fit L = E + A/N^a + B/D^b and the roughly 20 tokens-per-parameter optimum that MAI-Base-2's 625 TPP sits far beyond.
- **Porian et al. (2024).** *Resolving discrepancies in compute-optimal scaling of language models.* https://arxiv.org/abs/2406.19146  
  Why Kaplan and Chinchilla disagreed: embedding parameters, warmup and learning-rate tuning at small scale; a warning about what a ladder must control.
- **Muennighoff et al. (2023).** *Scaling data-constrained language models.* https://arxiv.org/abs/2305.16264  
  Repeating data up to about four epochs is nearly free, then returns decay; the regime that source scaling reproduces on every rung.
- **Bhagia et al. (2024).** *Establishing task scaling laws via compute-efficient model ladders.* https://arxiv.org/abs/2412.04403  
  The ladder method made explicit: a family of small models with one recipe predicts the large model's task performance within a few points.
- **Choshen et al. (2024).** *A hitchhiker's guide to scaling law estimation.* https://arxiv.org/abs/2410.11840  
  How to fit and extrapolate laws in practice: how many models, which sizes, intermediate checkpoints, and the usual errors.

## 4. Hyperparameter laws (slides 15-18)

- **Loshchilov and Hutter (2019).** *Decoupled weight decay regularization.* https://arxiv.org/abs/1711.05101  
  AdamW: weight decay applied as a multiplicative shrink (1 - lr*wd) rather than as an L2 term in the gradient, which is why the product lr*wd is the quantity that matters.
- **Wang and Aitchison (2024).** *How to set AdamW's weight decay as you scale model and dataset size.* https://arxiv.org/abs/2405.13698  
  The EMA view of AdamW with timescale 1/(lr*wd), and the argument that the timescale should be a fixed fraction of training; the report's exponent of -1 is this rule.
- **Malladi et al. (2022).** *On the SDEs and scaling rules for adaptive gradient algorithms.* https://arxiv.org/abs/2205.10287  
  Derives the square-root scaling rule for Adam-type optimizers: double the batch, multiply the learning rate by sqrt(2); the report's exponent of -0.5 in steps.
- **Yang et al. (2022).** *Tensor Programs V: tuning large neural networks via zero-shot hyperparameter transfer.* https://arxiv.org/abs/2203.03466  
  muP: parameterize so the optimal learning rate does not move with width; the alternative to fitting empirical hyperparameter laws.
- **DeepSeek-AI (2024).** *DeepSeek LLM: scaling open-source language models with longtermism.* https://arxiv.org/abs/2401.02954  
  Empirical scaling laws for learning rate and batch size against compute, fitted the same way as the report's laws for LR and weight decay.
- **Hu et al. (2024).** *MiniCPM: unveiling the potential of small language models with scalable training strategies.* https://arxiv.org/abs/2404.06395  
  Warmup-stable-decay schedules and muP in a practical small-model ladder; how hyperparameter sweeps are made affordable.
- **Srivastava et al. (2014).** *Dropout: a simple way to prevent neural networks from overfitting.* https://jmlr.org/papers/v15/srivastava14a.html  
  The regularizer whose rate the report sets by a scaling law in active parameters. JMLR 15. Not on arXiv.

## 5. MoE systems (slides 19-23)

- **Gale et al. (2022).** *MegaBlocks: efficient sparse training with mixture-of-experts.* https://arxiv.org/abs/2211.15841  
  Dropless MoE with block-sparse kernels: no token dropping, but memory that moves with the routing; the design MAI's fixed-memory multi-round scheme improves on.
- **Rajbhandari et al. (2022).** *DeepSpeed-MoE: advancing mixture-of-experts inference and training to power next-generation AI scale.* https://arxiv.org/abs/2201.05596  
  Expert parallelism at scale in a production training stack, with the communication and memory bookkeeping that per-expert overhead comes from.
- **Hwang et al. (2022).** *Tutel: adaptive mixture-of-experts at scale.* https://arxiv.org/abs/2206.03382  
  Adaptive parallelism and pipelining for MoE layers, including dynamic capacity factors; the systems background for dispatch/collect scheduling.
- **Zoph et al. (2022).** *ST-MoE: designing stable and transferable sparse expert models.* https://arxiv.org/abs/2202.08906  
  The router z-loss and a careful study of MoE training stability; how capacity factor and token dropping affect quality.
- **Wang et al. (2024).** *Auxiliary-loss-free load balancing strategy for mixture-of-experts.* https://arxiv.org/abs/2408.15664  
  Balancing expert load with a per-expert bias instead of a loss term; the alternative to the batched load-balancing loss the report describes.
- **Korthikanti et al. (2022).** *Reducing activation recomputation in large transformer models.* https://arxiv.org/abs/2205.05198  
  Activation recomputation (rematerialization) and how to minimize it; the compute that memory freed from MoE buffers buys back.

## 6. Muon (slides 24-25)

- **Jordan (2024).** *Muon: an optimizer for hidden layers in neural networks.* https://kellerjordan.github.io/posts/muon/  
  The origin of Muon: orthogonalize the momentum of each weight matrix with five bfloat16 Newton-Schulz iterations. Blog post; not on arXiv.
- **Bernstein and Newhouse (2024).** *Old optimizer, new norm: an anthology.* https://arxiv.org/abs/2409.20325  
  Muon as steepest descent under the spectral norm; why equalizing singular values of the update is the natural step for a matrix parameter.
- **Liu et al. (2025).** *Muon is scalable for LLM training.* https://arxiv.org/abs/2502.16982  
  Weight decay and update-scale adjustments that make Muon work at scale; about twice AdamW's compute efficiency on a 16B MoE.
- **Kimi Team (2025).** *Kimi K2: open agentic intelligence.* https://arxiv.org/abs/2507.20534  
  A 1T-parameter MoE trained with MuonClip over 15.5T tokens with no loss spikes; the largest public Muon run before MAI-Base-2.
- **Rajbhandari et al. (2020).** *ZeRO: memory optimizations toward training trillion parameter models.* https://arxiv.org/abs/1910.02054  
  Sharding optimizer state across data-parallel GPUs, which is why Muon's momentum has to be resharded before Newton-Schulz.
- **Wen et al. (2025).** *Fantastic pretraining optimizers and where to find them.* https://arxiv.org/abs/2509.02046  
  A controlled comparison of Muon, SOAP, Shampoo and AdamW across scales; how much of the gain survives careful tuning.

## 7. Numerics and stability (slides 26-29)

- **Micikevicius et al. (2022).** *FP8 formats for deep learning.* https://arxiv.org/abs/2209.05433  
  E4M3 and E5M2: the two 8-bit floats, their ranges, and tensor-wise scaling with a delayed scale, MAI-Base-1's scheme.
- **Rouhani et al. (2023).** *Microscaling data formats for deep learning.* https://arxiv.org/abs/2310.10537  
  MX formats: blocks of 32 values sharing an 8-bit power-of-two scale; MXFP8 is MAI-Base-2's training format.
- **NVIDIA (2025).** *Recipes for pre-training LLMs with MXFP8.* https://arxiv.org/abs/2506.08027  
  How to train with MXFP8 in practice: scale rounding, which tensors to quantize, and matching BF16 on an 8B model over 15T tokens.
- **Fishman et al. (2024).** *Scaling FP8 training to trillion-token LLMs.* https://arxiv.org/abs/2409.12517  
  FP8 instabilities that appear only after hundreds of billions of tokens, and their fix; why a stable FP8 recipe must be tested at full horizon.
- **Pascanu, Mikolov and Bengio (2013).** *On the difficulty of training recurrent neural networks.* https://arxiv.org/abs/1211.5063  
  Global-norm gradient clipping, the standard guard that MAI-Base-1 used alone and MAI-Base-2 supplements with activation gradient clipping.
- **Dehghani et al. (2023).** *Scaling vision transformers to 22 billion parameters.* https://arxiv.org/abs/2302.05442  
  Attention-logit growth as a cause of divergence and QK-normalization as the fix; the canonical example of a stability fix found by looking inside the network.
- **Wortsman et al. (2023).** *Small-scale proxies for large-scale transformer training instabilities.* https://arxiv.org/abs/2309.14322  
  Reproduces loss spikes at small scale by raising the learning rate; QK-norm, z-loss and the LR sensitivity of large models; what a ladder can and cannot reveal about stability.
- **OLMo Team (2024).** *2 OLMo 2 furious.* https://arxiv.org/abs/2501.00656  
  A fully open run that documents its stability fixes (initialization, normalization, z-loss) and its gradient-norm traces; a public counterpart to Figures 5 and 7.

## 8. Data and vision (slides 30-32)

- **Penedo et al. (2024).** *The FineWeb datasets: decanting the web for the finest text data at scale.* https://arxiv.org/abs/2406.17557  
  A documented web-data pipeline (parse, filter, dedup, classify) with ablations for each stage; the open counterpart to MAI's in-house pipeline.
- **Li et al. (2024).** *DataComp-LM: in search of the next generation of training sets for language models.* https://arxiv.org/abs/2406.11794  
  A benchmark for data curation at fixed compute; quality classifiers give the largest gains, as the report also finds.
- **Abbas et al. (2023).** *SemDeDup: data-efficient learning at web-scale through semantic deduplication.* https://arxiv.org/abs/2303.09540  
  Embedding-based removal of near-duplicates that exact matching misses; the technique behind diversity sampling.
- **Tirumala et al. (2023).** *D4: improving LLM pretraining via document de-duplication and diversification.* https://arxiv.org/abs/2308.12284  
  Picks a diverse subset in embedding space and beats random selection at equal tokens; the closest published analogue of the report's diversity sampling.
- **Ye et al. (2024).** *Data mixing laws: optimizing data mixtures by predicting language modeling performance.* https://arxiv.org/abs/2403.16952  
  Loss as a function of domain proportions, fitted on small runs to choose the mix for a large one; how a Figure 10 mixture can be chosen rather than guessed.
- **Dosovitskiy et al. (2020).** *An image is worth 16x16 words: transformers for image recognition at scale.* https://arxiv.org/abs/2010.11929  
  Images as sequences of linearly embedded patches; the patch embedding that an encoder-free VLM feeds straight into the language model.
- **Bavishi et al. (2023).** *Fuyu-8B: a multimodal architecture for AI agents.* https://www.adept.ai/blog/fuyu-8b  
  The first widely used encoder-free VLM: image patches projected linearly into a decoder-only transformer, any resolution. Blog post; not on arXiv.
- **Diao et al. (2024).** *Unveiling encoder-free vision-language models.* https://arxiv.org/abs/2406.11832  
  A systematic study of removing the vision encoder: what data and training it takes for a pure decoder to match encoder-based VLMs.
- **Beyer et al. (2024).** *PaliGemma: a versatile 3B VLM for transfer.* https://arxiv.org/abs/2407.07726  
  The encoder-plus-projector design the slide contrasts with MAI's, done carefully; useful for seeing exactly what the encoder-free route gives up and gains.

## 9. Comparators and frontier reports (slides 5b and 33)

- **DeepSeek-AI (2024).** *DeepSeek-V3 technical report.* https://arxiv.org/abs/2412.19437  
  The reference open frontier recipe: 671B/37B MoE with 1 shared + 8 of 256 experts, block-scaled FP8 training, and multi-token prediction; the design MAI-Base-2's MoE layout is closest to.
- **Qwen Team (2025).** *Qwen3 technical report.* https://arxiv.org/abs/2505.09388  
  The lineage of the Qwen3.8 comparator: fine-grained MoE, GQA, and a documented pre-training mixture of 36T tokens.
- **Yang et al. (2024).** *Gated Delta Networks: improving Mamba2 with delta rule.* https://arxiv.org/abs/2412.06464  
  The linear-attention layer (Gated DeltaNet) that Qwen3.8 interleaves 3:1 with full attention, per its model card.
- **Kimi Team (2025).** *Kimi Linear: an expressive, efficient attention architecture.* https://arxiv.org/abs/2510.26692  
  Kimi Delta Attention, the linear layer Kimi K3 mixes with gated MLA, per its model card.
- **Meta (2025).** *The Llama 4 herd: the beginning of a new era of natively multimodal AI innovation.* https://ai.meta.com/blog/llama-4-multimodal-intelligence/  
  Natively multimodal MoE models with interleaved local and global attention (iRoPE); a contemporary of the MAI design. Blog post; not on arXiv.
- **DeepSeek-AI (2026).** *DeepSeek-V4-Pro-Base model card (Hugging Face).* https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-Base  
  Source of the comparator row on slide 5b: 1.6T total / 49B active, 61 layers, compressed sparse attention, base weights trained on more than 32T tokens. Model card; secondary source.
- **Moonshot AI (2026).** *Kimi-K3 model card (Hugging Face).* https://huggingface.co/moonshotai/Kimi-K3  
  Source of the comparator row on slide 5b: 2.8T / 104B, 93 layers, 896 experts with 16 active and 2 shared, 69 Kimi Delta Attention + 24 gated MLA layers, MXFP4 weights. Model card; secondary source.
- **Qwen Team (2026).** *Qwen3.8-2.4T-A95B model card (Hugging Face).* https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B  
  Source of the comparator row on slide 5b: 2.4T / 95B, 92 layers, 512 experts with 10 active and 1 shared, 69 Gated DeltaNet + 23 attention layers. Model card; secondary source.
- **Xiaomi (2026).** *MiMo-V2.6-Pro-RL model card (Hugging Face).* https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL  
  Source of the comparator row on slide 5b: 1.02T / 42B, 70 layers, 384 experts with 8 active, sliding-window plus global attention. Model card; secondary source.
