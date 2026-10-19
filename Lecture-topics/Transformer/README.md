# Reading list: The Transformer, Visually

Companion papers to the deck and its lecture notes, grouped by section.
`reading_list.csv` and `reading_list.bib` hold the same list as data; `fetch_papers.sh` downloads
every arXiv entry into one folder per section and zips the result (`bash fetch_papers.sh`).
Entries without an arXiv identifier are listed with a URL and must be fetched by hand.

## 1. Overview and the residual stream (slides 1-4)

- **Vaswani et al. (2017).** *Attention is all you need.* https://arxiv.org/abs/1706.03762  
  The original paper: scaled dot-product attention, multi-head attention, sinusoidal positions, the post-norm block and the encoder-decoder layout. Short, and still the right place to start.
- **Radford, Narasimhan, Salimans and Sutskever (2018).** *Improving language understanding by generative pre-training (GPT).* https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf  
  The first decoder-only transformer language model trained on next-token prediction and then fine-tuned; the recipe the deck describes. Not on arXiv.
- **Radford et al. (2019).** *Language models are unsupervised multitask learners (GPT-2).* https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf  
  GPT-2: the 124M/355M/774M/1.5B family, pre-norm, byte-level BPE, tied embeddings; the configuration the deck counts by hand. The code and configs are at github.com/openai/gpt-2. Not on arXiv.
- **Elhage et al. (2021).** *A mathematical framework for transformer circuits.* https://transformer-circuits.pub/2021/framework/index.html  
  The residual-stream view of slide 4: heads as independent read-write operations on a shared stream, QK and OV circuits, and the first analysis of induction heads. Published as a web article, not on arXiv.
- **Phuong and Hutter (2022).** *Formal algorithms for transformers.* https://arxiv.org/abs/2207.09238  
  Every component of the deck written out as pseudocode with explicit shapes: tokenization, embeddings, attention, the block, training and decoding. The best companion for checking one's understanding line by line.
- **Bahdanau, Cho and Bengio (2014).** *Neural machine translation by jointly learning to align and translate.* https://arxiv.org/abs/1409.0473  
  Where attention came from: a translation decoder that computes a soft alignment over the source positions instead of compressing the sentence into one vector.
- **Alammar (2018).** *The illustrated transformer (blog).* https://jalammar.github.io/illustrated-transformer/  
  The most widely read visual walkthrough of the 2017 architecture; the deck's pictures of Q, K, V and multi-head attention follow the same conventions. Secondary source.
- **Karpathy (2022).** *nanoGPT (code repository).* https://github.com/karpathy/nanoGPT  
  A complete GPT-2 implementation and training loop in about 300 lines of PyTorch; reading model.py after the deck turns every slide into a few lines of code. Secondary source.

## 2. Tokens, embeddings and position (slides 5-9)

- **Sennrich, Haddow and Birch (2016).** *Neural machine translation of rare words with subword units.* https://arxiv.org/abs/1508.07909  
  Byte-pair encoding as a subword vocabulary for neural models; the method behind every tokenizer on slide 5.
- **Kudo and Richardson (2018).** *SentencePiece: a simple and language independent subword tokenizer and detokenizer for neural text processing.* https://arxiv.org/abs/1808.06226  
  The library used by Llama 2 and many others; treats the input as a raw stream, so whitespace is just another symbol.
- **Kudo (2018).** *Subword regularization: improving neural network translation models with multiple subword candidates.* https://arxiv.org/abs/1804.10959  
  The unigram language-model tokenizer, the main alternative to BPE, and why sampling among segmentations helps training.
- **Tao et al. (2024).** *Scaling laws with vocabulary: larger models deserve larger vocabularies.* https://arxiv.org/abs/2407.13623  
  Why vocabularies grew from 32k to 128k-256k: the compute-optimal vocabulary size rises with model size.
- **Mikolov, Chen, Corrado and Dean (2013).** *Efficient estimation of word representations in vector space.* https://arxiv.org/abs/1301.3781  
  word2vec: word vectors from context prediction, and the king - man + woman = queen regularity of slide 7.
- **Su et al. (2021).** *RoFormer: enhanced transformer with rotary position embedding.* https://arxiv.org/abs/2104.09864  
  RoPE: rotate queries and keys by position so that attention scores depend only on relative distance; the position scheme of Llama, Mistral, Qwen and Gemma (slide 9).
- **Shaw, Uszkoreit and Vaswani (2018).** *Self-attention with relative position representations.* https://arxiv.org/abs/1803.02155  
  The first relative-position scheme for transformers, adding learned offset vectors to keys; the idea RoPE makes parameter-free.
- **Press, Smith and Lewis (2021).** *Train short, test long: attention with linear biases enables input length extrapolation.* https://arxiv.org/abs/2108.12409  
  ALiBi: no position vectors at all, just a distance penalty added to the scores; used by BLOOM and MPT, and the clearest statement of the length-extrapolation problem.
- **Chen, Wong, Chen and Tian (2023).** *Extending context window of large language models via positional interpolation.* https://arxiv.org/abs/2306.15595  
  Rescale RoPE angles so that a longer context maps onto the trained range, then fine-tune briefly; the basis of most context extension.
- **Peng, Quesnelle, Fan and Shippole (2023).** *YaRN: efficient context window extension of large language models.* https://arxiv.org/abs/2309.00071  
  Frequency-aware RoPE rescaling: stretch the slow pairs, leave the fast ones; used by several long-context open models.

## 3. Attention (slides 10-16)

- **Luong, Pham and Manning (2015).** *Effective approaches to attention-based neural machine translation.* https://arxiv.org/abs/1508.04025  
  Dot-product scoring between a query state and key states, the scoring function the transformer adopted, compared with alternatives.
- **Clark, Khandelwal, Levy and Manning (2019).** *What does BERT look at? An analysis of BERT's attention.* https://arxiv.org/abs/1906.04341  
  Attention heads that track syntactic relations and coreference, and heads that park on [SEP] when idle; the source of slide 16's patterns.
- **Voita, Talbot, Moiseev, Sennrich and Titov (2019).** *Analyzing multi-head self-attention: specialized heads do the heavy lifting, the rest can be pruned.* https://arxiv.org/abs/1905.09418  
  Positional, syntactic and rare-word heads in a translation model, and the finding that most other heads can be removed.
- **Olsson et al. (2022).** *In-context learning and induction heads.* https://arxiv.org/abs/2209.11895  
  Induction heads as the mechanism of in-context learning in small models, and their appearance as a bend in the training curve.
- **Xiao, Tian, Chen, Han and Lewis (2023).** *Efficient streaming language models with attention sinks.* https://arxiv.org/abs/2309.17453  
  The attention sink: idle attention parks on the first token, and keeping it in the cache is what makes sliding-window generation work.
- **Dao, Fu, Ermon, Rudra and Re (2022).** *FlashAttention: fast and memory-efficient exact attention with IO-awareness.* https://arxiv.org/abs/2205.14135  
  Exact attention computed in tiles without materializing the T x T matrix; the implementation every model uses (slide 28).
- **Dao (2023).** *FlashAttention-2: faster attention with better parallelism and work partitioning.* https://arxiv.org/abs/2307.08691  
  The second version, with the work split that reaches a large fraction of peak on H100s.

## 4. MLP, norm and the block (slides 17-22)

- **Geva, Schuster, Berant and Levy (2021).** *Transformer feed-forward layers are key-value memories.* https://arxiv.org/abs/2012.14913  
  Rows of W1 as pattern detectors and columns of W2 as the updates they write; the reading of the MLP on slide 17.
- **Hendrycks and Gimpel (2016).** *Gaussian error linear units (GELUs).* https://arxiv.org/abs/1606.08415  
  GELU and SiLU, the smooth activations of BERT, GPT-2 and the SwiGLU gate.
- **Ramachandran, Zoph and Le (2017).** *Searching for activation functions.* https://arxiv.org/abs/1710.05941  
  Swish, x times sigmoid(x), found by search and shown to beat ReLU across tasks; the same function as SiLU.
- **Dauphin, Fan, Auli and Grangier (2017).** *Language modeling with gated convolutional networks.* https://arxiv.org/abs/1612.08083  
  The gated linear unit: one projection gates another elementwise; the structure SwiGLU inherits.
- **Shazeer (2020).** *GLU variants improve transformer.* https://arxiv.org/abs/2002.05202  
  SwiGLU and its relatives as the transformer MLP, with the 8/3 d hidden width that keeps the parameter count fixed (slide 18).
- **He, Zhang, Ren and Sun (2016).** *Deep residual learning for image recognition.* https://arxiv.org/abs/1512.03385  
  Residual connections: learn a correction and add it to the input, which lets very deep networks train (slide 19).
- **Ba, Kiros and Hinton (2016).** *Layer normalization.* https://arxiv.org/abs/1607.06450  
  Normalize over the features of one example rather than over a batch; the norm of the 2017 Transformer, BERT and GPT-2.
- **Zhang and Sennrich (2019).** *Root mean square layer normalization.* https://arxiv.org/abs/1910.07467  
  Drop the centering, keep the scaling: the cheaper norm used by Llama, Mistral, Qwen and Gemma (slide 20).
- **Xiong et al. (2020).** *On layer normalization in the transformer architecture.* https://arxiv.org/abs/2002.04745  
  Why post-norm needs warm-up and pre-norm does not: the gradient scale at initialization under the two placements (slide 19).
- **Tenney, Das and Pavlick (2019).** *BERT rediscovers the classical NLP pipeline.* https://arxiv.org/abs/1905.05950  
  Probing shows syntax resolved in earlier layers and semantics later; the division-of-labor note on slide 22.
- **Tay et al. (2022).** *Scale efficiently: insights from pre-training and fine-tuning transformers.* https://arxiv.org/abs/2109.10686  
  How depth and width trade off at fixed compute; deeper-narrower tends to win downstream.

## 5. Output and generation (slides 23-25)

- **Press and Wolf (2017).** *Using the output embedding to improve language models.* https://arxiv.org/abs/1608.05859  
  Weight tying: use the embedding matrix as the unembedding; saves V d parameters and improves perplexity (slide 23).
- **Holtzman, Buys, Du, Forbes and Choi (2020).** *The curious case of neural text degeneration.* https://arxiv.org/abs/1904.09751  
  Why greedy and beam search produce bland, repetitive text, and nucleus (top-p) sampling as the fix (slide 24).
- **Fan, Lewis and Dauphin (2018).** *Hierarchical neural story generation.* https://arxiv.org/abs/1805.04833  
  Top-k sampling, introduced here for story generation.
- **Shazeer (2019).** *Fast transformer decoding: one write-head is all you need.* https://arxiv.org/abs/1911.02150  
  Multi-query attention: share one key/value head across all query heads to shrink the KV cache and speed up decoding.
- **Ainslie et al. (2023).** *GQA: training generalized multi-query transformer models from multi-head checkpoints.* https://arxiv.org/abs/2305.13245  
  Grouped-query attention, the middle ground used by Llama 2 70B and all of Llama 3 (slides 25 and 29).
- **Pope et al. (2022).** *Efficiently scaling transformer inference.* https://arxiv.org/abs/2211.05102  
  The cost model of decoding: memory-bound at small batch, the KV cache as the limit, and the partitioning strategies that follow.
- **Kwon et al. (2023).** *Efficient memory management for large language model serving with PagedAttention.* https://arxiv.org/abs/2309.06180  
  vLLM: manage the KV cache in pages like virtual memory, which is what lets a GPU serve many sequences at once.
- **Leviathan, Kalman and Matias (2023).** *Fast inference from transformers via speculative decoding.* https://arxiv.org/abs/2211.17192  
  Draft several tokens with a small model and verify them in one pass of the large one; exact sampling at a fraction of the decode steps.

## 6. Training and scale (slides 26-28)

- **Loshchilov and Hutter (2017).** *Decoupled weight decay regularization.* https://arxiv.org/abs/1711.05101  
  AdamW, the optimizer used for essentially all LLM training.
- **Kaplan et al. (2020).** *Scaling laws for neural language models.* https://arxiv.org/abs/2001.08361  
  Loss as a power law in parameters, data and compute; the 6ND cost model; and the finding that shape (depth versus width) matters little (slides 22 and 28).
- **Hoffmann et al. (2022).** *Training compute-optimal large language models.* https://arxiv.org/abs/2203.15556  
  Chinchilla: about twenty training tokens per parameter at the compute-optimal point; why current models train on trillions of tokens.
- **Shoeybi et al. (2019).** *Megatron-LM: training multi-billion parameter language models using model parallelism.* https://arxiv.org/abs/1909.08053  
  How the block's matrix products are split across GPUs; the tensor-parallel layout follows the 4 d^2 + 8 d^2 structure of slide 21.
- **Rajbhandari, Rasley, Ruwase and He (2019).** *ZeRO: memory optimizations toward training trillion parameter models.* https://arxiv.org/abs/1910.02054  
  Where training memory goes (weights, gradients, optimizer states) and how to shard it; the accounting behind training at scale.
- **Brown et al. (2020).** *Language models are few-shot learners.* https://arxiv.org/abs/2005.14165  
  GPT-3: the 96-layer, 12,288-wide, 175B configuration of slide 22, and the demonstration that scale alone yields in-context learning.
- **Touvron et al. (2023).** *Llama 2: open foundation and fine-tuned chat models.* https://arxiv.org/abs/2307.09288  
  The 7B/13B/70B configurations cited on slides 22, 25 and 27: RMSNorm, SwiGLU, RoPE, GQA at 70B.
- **Grattafiori et al. (2024).** *The Llama 3 herd of models.* https://arxiv.org/abs/2407.21783  
  The 8B/70B/405B configurations, 15.6T training tokens, 128k vocabulary, RoPE base 500,000 and GQA with 8 KV heads; the deck's reference for a current model.

## 7. The family and further reading (slides 29-31)

- **Devlin, Chang, Lee and Toutanova (2018).** *BERT: pre-training of deep bidirectional transformers for language understanding.* https://arxiv.org/abs/1810.04805  
  The encoder-only design of slide 29: bidirectional attention trained to fill in masked tokens.
- **Raffel et al. (2019).** *Exploring the limits of transfer learning with a unified text-to-text transformer.* https://arxiv.org/abs/1910.10683  
  T5, the encoder-decoder design at scale, with a systematic study of the choices that go into one.
- **Touvron et al. (2023).** *LLaMA: open and efficient foundation language models.* https://arxiv.org/abs/2302.13971  
  The first Llama: the decoder-only recipe with pre-RMSNorm, SwiGLU and RoPE that most open models have copied since.
- **Jiang et al. (2023).** *Mistral 7B.* https://arxiv.org/abs/2310.06825  
  A compact current design: GQA, sliding-window attention and a rolling KV cache in a 7B model.
- **Gemma Team (2024).** *Gemma: open models based on Gemini research and technology.* https://arxiv.org/abs/2403.08295  
  A 256k-vocabulary decoder-only model with tied embeddings and GeGLU; a useful contrast with Llama's choices.
- **Qwen Team (2025).** *Qwen2.5 technical report.* https://arxiv.org/abs/2412.15115  
  A current open family with the same block (RMSNorm, SwiGLU, RoPE, GQA) and a 152k vocabulary; what 'typical LLM today' on slide 29 looks like in detail.
- **DeepSeek-AI (2024).** *DeepSeek-V3 technical report.* https://arxiv.org/abs/2412.19437  
  Where the family goes next: multi-head latent attention compressing the KV cache and a mixture-of-experts MLP; the bridge to the sparsity lecture.
- **Rush (Harvard NLP) (2018).** *The annotated transformer (web page).* https://nlp.seas.harvard.edu/annotated-transformer/  
  The 2017 paper interleaved with a working PyTorch implementation, section by section. Secondary source.
