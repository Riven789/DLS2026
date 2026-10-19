# Reading list: Scaling Laws, Visually

Companion papers to the deck and its lecture notes, grouped by section.
`reading_list.csv` and `reading_list.bib` hold the same list as data; `fetch_papers.sh` downloads
every arXiv entry into one folder per section and zips the result (`bash fetch_papers.sh`).
Entries without an arXiv identifier are listed with a URL and must be fetched by hand.

## 1. Before language models (slides 1-6)

- **Cortes, Jackel, Solla, Vapnik and Denker (1993).** *Learning curves: asymptotic values and rate of convergence.* https://proceedings.neurips.cc/paper_files/paper/1993/file/1aa48fc4880bb0c9b8a3bf979d3b917e-Paper.pdf  
  Power-law learning curves fitted on small training sets to predict large-set error; the 1993 origin of the method. NIPS 6. Not on arXiv.
- **Banko and Brill (2001).** *Scaling to very very large corpora for natural language disambiguation.* https://aclanthology.org/P01-1005.pdf  
  Accuracy keeps rising log-linearly with corpus size across a billion words, and simple models catch up with complex ones. ACL 2001. Not on arXiv.
- **Halevy, Norvig and Pereira (2009).** *The unreasonable effectiveness of data.* https://doi.org/10.1109/MIS.2009.36  
  The essay that made 'more data beats cleverer algorithms' a slogan. IEEE Intelligent Systems 24(2). Not on arXiv.
- **Sun et al. (2017).** *Revisiting unreasonable effectiveness of data in deep learning era.* https://arxiv.org/abs/1707.02968  
  JFT-300M: vision performance grows logarithmically with data over three orders of magnitude.
- **Hestness et al. (2017).** *Deep learning scaling is predictable, empirically.* https://arxiv.org/abs/1712.00409  
  Power-law generalization error across translation, speech, vision and language (character-LM exponent -0.094); the first modern 'scaling is predictable' paper.
- **Amodei and Hernandez (2018).** *AI and compute (OpenAI blog).* https://openai.com/index/ai-and-compute/  
  The 3.4-month doubling of training compute from 2012 to 2018. Secondary source; not on arXiv.
- **Rosenfeld et al. (2019).** *A constructive prediction of the generalization error across scales.* https://arxiv.org/abs/1909.12673  
  A joint functional form in model size and data that predicts error across scales, before Kaplan.
- **Sevilla et al. (2022).** *Compute trends across three eras of machine learning.* https://arxiv.org/abs/2202.05924  
  Epoch's reconstruction of training-compute growth; the context for every 'how much compute' number.

## 2. Core results: Kaplan and Chinchilla (slides 7-12)

- **Kaplan et al. (2020).** *Scaling laws for neural language models.* https://arxiv.org/abs/2001.08361  
  Power laws in N, D and C over seven orders of magnitude, the 6ND cost model, the critical batch size, and the first compute-optimal allocation.
- **Brown et al. (2020).** *Language models are few-shot learners.* https://arxiv.org/abs/2005.14165  
  GPT-3: the first frontier model whose size was justified by Kaplan's law; Figure 3.1 is the law extended to 175B.
- **Henighan et al. (2020).** *Scaling laws for autoregressive generative modeling.* https://arxiv.org/abs/2010.14701  
  The same laws for images, video, math and multimodal models; the loss floor and the irreducible term.
- **Hoffmann et al. (2022).** *Training compute-optimal large language models.* https://arxiv.org/abs/2203.15556  
  Chinchilla: three methods, the parametric fit L = E + A/N^a + B/D^b, and the 20-tokens-per-parameter rule that replaced Kaplan's allocation.
- **Besiroglu et al. (2024).** *Chinchilla scaling: a replication attempt.* https://arxiv.org/abs/2404.10102  
  Re-fits Chinchilla's third method from the paper's own figure and finds different constants; a lesson in fitting.
- **Porian et al. (2024).** *Resolving discrepancies in compute-optimal scaling of language models.* https://arxiv.org/abs/2406.19146  
  Why Kaplan and Chinchilla disagreed: embedding parameters, warmup, and learning-rate tuning at small scale.
- **McCandlish et al. (2018).** *An empirical model of large-batch training.* https://arxiv.org/abs/1812.06162  
  The gradient-noise scale and the critical batch size; the source of Kaplan's B_crit law.

## 3. Refinements (slides 13-18)

- **Sardana and Frankle (2023).** *Beyond Chinchilla-optimal: accounting for inference in language model scaling laws.* https://arxiv.org/abs/2401.00448  
  Add the inference cost 2N·D_inf to the objective: models that will be served a lot should be smaller and trained longer.
- **Gadre et al. (2024).** *Language models scale reliably with over-training and on downstream tasks.* https://arxiv.org/abs/2403.08540  
  Laws fitted at up to 32x over-training extrapolate reliably; also fits the loss-to-average-downstream-error curve used in Section 5.
- **Muennighoff et al. (2023).** *Scaling data-constrained language models.* https://arxiv.org/abs/2305.16264  
  Repeating data up to about four epochs is nearly free; the decay constants for repeated tokens and excess parameters.
- **Villalobos et al. (2022).** *Will we run out of data? Limits of LLM scaling based on human-generated data.* https://arxiv.org/abs/2211.04325  
  Epoch's estimate of the public text stock and when training runs exhaust it (revised in 2024).
- **DeepSeek-AI (2024).** *DeepSeek LLM: scaling open-source language models with longtermism.* https://arxiv.org/abs/2401.02954  
  Hyperparameter scaling laws (learning rate and batch size against compute), the non-embedding FLOPs-per-token M, and a careful compute-optimal fit.
- **Yang et al. (2022).** *Tensor Programs V: tuning large neural networks via zero-shot hyperparameter transfer.* https://arxiv.org/abs/2203.03466  
  muP: parameterize so that the optimal learning rate does not move with width, then tune on a small model.
- **Hu et al. (2024).** *MiniCPM: unveiling the potential of small language models with scalable training strategies.* https://arxiv.org/abs/2404.06395  
  Warmup-stable-decay schedules and muP in a practical small-model ladder; the WSD trick that makes ladders cheap.
- **Hagele et al. (2024).** *Scaling laws and compute-optimal training beyond fixed training durations.* https://arxiv.org/abs/2405.18392  
  With a constant learning rate and short cooldowns, one run yields loss at many token counts; scaling experiments at a fraction of the cost.
- **Dey et al. (2023).** *Cerebras-GPT: open compute-optimal language models trained on the Cerebras wafer-scale cluster.* https://arxiv.org/abs/2304.03208  
  A public Chinchilla-style ladder with muP, reporting the fits and the gain from muP in practice.
- **Wortsman et al. (2023).** *Small-scale proxies for large-scale transformer training instabilities.* https://arxiv.org/abs/2309.14322  
  Instabilities that appear at scale (attention-logit growth, output-logit divergence) reproduced in small models by raising the learning rate; what to watch on a ladder.

## 4. Architecture knobs (slide 19)

- **Clark et al. (2022).** *Unified scaling laws for routed language models.* https://arxiv.org/abs/2202.01169  
  The first scaling law with the number of experts as a variable.
- **Krajewski et al. (2024).** *Scaling laws for fine-grained mixture of experts.* https://arxiv.org/abs/2402.07871  
  Granularity as a law variable; MoE beats dense at every scale when tuned.
- **Tao et al. (2024).** *Scaling laws with vocabulary: larger models deserve larger vocabularies.* https://arxiv.org/abs/2407.13623  
  The compute-optimal vocabulary size grows with model size; most models' vocabularies are too small.
- **Kumar et al. (2024).** *Scaling laws for precision.* https://arxiv.org/abs/2411.04330  
  Precision as a third resource: effective parameters N(1 - e^{-P/gamma}), a 7-8 bit optimum, and the post-training quantization penalty.
- **Ye et al. (2024).** *Data mixing laws: optimizing data mixtures by predicting language modeling performance.* https://arxiv.org/abs/2403.16952  
  Loss as a function of domain mixture proportions, fitted on small runs to choose the mix for a large one.

## 5. Loss to capability (slides 19-20)

- **Wei et al. (2022).** *Emergent abilities of large language models.* https://arxiv.org/abs/2206.07682  
  Benchmarks that jump from chance to competence at a scale threshold; the case that capability is not smooth in loss.
- **Schaeffer, Miranda and Koyejo (2023).** *Are emergent abilities of large language models a mirage?.* https://arxiv.org/abs/2304.15004  
  The jumps come from discontinuous metrics (exact match) on smoothly improving per-token accuracy.
- **OpenAI (2023).** *GPT-4 technical report.* https://arxiv.org/abs/2303.08774  
  Section 3: loss and a coding pass rate predicted from runs with 1,000-10,000x less compute.
- **Grattafiori et al. (2024).** *The Llama 3 herd of models.* https://arxiv.org/abs/2407.21783  
  Section 3.2.1: a two-stage law, compute to loss and loss to benchmark accuracy, used to set the 405B model's size.
- **Bhagia et al. (2024).** *Establishing task scaling laws via compute-efficient model ladders.* https://arxiv.org/abs/2412.04403  
  A ladder of small models predicts task accuracy of 7B and 13B models within a few points; the ladder method made explicit.
- **Isik et al. (2024).** *Scaling laws for downstream task performance in machine translation.* https://arxiv.org/abs/2402.04177  
  When downstream metrics follow the pretraining loss and when they do not (distribution mismatch).

## 6. Why power laws (slide 20)

- **Bahri et al. (2021).** *Explaining neural scaling laws.* https://arxiv.org/abs/2102.06701  
  Variance-limited and resolution-limited regimes; exponents tied to the intrinsic dimension of the data manifold.
- **Hutter (2021).** *Learning curve theory.* https://arxiv.org/abs/2102.04074  
  A Zipf-distributed feature model that produces power-law learning curves from first principles.
- **Michaud et al. (2023).** *The quantization model of neural scaling.* https://arxiv.org/abs/2303.13506  
  Skills ('quanta') learned in order of frequency under a Zipf law give the observed exponents and the jumps underneath smooth loss.
- **Sharma and Kaplan (2020).** *A neural scaling law from the dimension of the data manifold.* https://arxiv.org/abs/2004.10802  
  Exponent alpha ~ 4/d for data on a d-dimensional manifold; the geometric argument behind Bahri et al.

## 7. Ladders and practice (slides 21-28)

- **Choshen et al. (2024).** *A hitchhiker's guide to scaling law estimation.* https://arxiv.org/abs/2410.11840  
  A large-scale study of how to fit and extrapolate laws: which models, how many, intermediate checkpoints, and the usual errors.
- **OLMo Team (2024).** *2 OLMo 2 furious.* https://arxiv.org/abs/2501.00656  
  A fully open training run with ladders, stability fixes and mid-training; the practice companion to Bhagia et al.
- **Wen et al. (2024).** *Understanding warmup-stable-decay learning rates: a river valley loss landscape perspective.* https://arxiv.org/abs/2410.05192  
  Why the WSD schedule works and why the decay phase gives the loss that ladders compare.
- **Hernandez et al. (2021).** *Scaling laws for transfer.* https://arxiv.org/abs/2102.01293  
  Effective data transferred from pretraining to fine-tuning follows a power law; how much pretraining is worth in fine-tuning tokens.

## 8. Frontier case studies (slides 29-30)

- **Microsoft AI (2026).** *Frontier pre-training at Microsoft AI (MAI-Base-2 report).* https://microsoft.ai/news/  
  The case study of the deck's frontier section: ladders, hyperparameter transfer and the final run. Not on arXiv; the PDF was supplied for the lecture.
- **DeepSeek-AI (2024).** *DeepSeek-V3 technical report.* https://arxiv.org/abs/2412.19437  
  A frontier run whose report shows the ladder (16B and 230B MoE pilots) before the 671B model.
