# Kukla peer review — Q20 (Q20-shared-internals)

**Reviewer:** Kukla (m1)  
**Date:** 2026-10-10  
**Method:** fresh `latexmk` build of all 5 .tex (PASS) + `python3` run of code (PASS, numpy env) + LLM-assisted content read (argo:claude-opus-4.8) grounded in note/code/bib/quiz, with reviewer verification of flagged citations.

---

> **Reviewer correction (post-verification):** My initial draft flagged the code as truncated/missing the Transformer class. That was an artifact of a 4500-char excerpt I reviewed — the FULL `shared_skeleton.py` (7816 chars) is complete and **runs cleanly** (verified: trains all three architectures, prints `final loss ... (floor ...)` and the shared-vs-architecture-specific summary). The Transformer is implemented as `class TinyAttn` (single-head causal attention with full backward), not `TinyTransformer`. The RNN `backward` DOES implement BPTT through `Wh` (the `dh_next`/`dh_prev @ Wh.T` recurrence). So code item #1 below is WITHDRAWN. One real minor code nit remains: `TinyRNN.backward` has a dead line `dh_raw = dh * (1 - Hs[:,i,:]**2) * 0` (multiplied by zero, unused) — remove it for clarity. Downgrading verdict to APPROVE WITH MINOR CHANGES; the remaining LaTeX/bib items still stand.

---

# Peer Review: Module Q20

## 1. Technical Accuracy

Overall the technical content is sound and refreshingly careful. A few issues:

- **"gradients no longer travel through a time-chain"** (Sec. 2). Mostly fine as a statement about the *sequence/depth-of-timesteps* axis, but worth a small qualifier: gradients still travel through the *depth* of stacked layers. During autoregressive training with teacher forcing the point holds; a reader could misread this as "no long backprop paths exist." Minor.

- **"attention weights are mixing coefficients, not evidence of 'what the model looked at'"** (Sec. 4). This is a defensible and well-known critique (cf. Jain & Wallace, "Attention is not Explanation"; Wiegreffe & Pinson rebuttal). The claim is stated a bit too absolutely. Consider softening to "attention weights are not *reliable* explanations" and ideally citing the attention-interpretability debate, which is currently asserted without support.

- **Table row "Layer norm & 2016"**. LayerNorm (Ba, Kiros, Hinton) is 2016 — correct. But the paper is uncited while being assigned a date; either cite it or drop the year precision. Minor.

- **"RL with verifiable rewards"** (Sec. 3) is listed as a pipeline stage but has no citation and is the newest/most contestable item. Flag it as recent practice rather than established canon, or cite.

- **"a trick with roots in the 1990s gating literature"** (MoE). True (Jacobs/Jordan mixtures-of-experts, 1991) but uncited. A pointer would strengthen it.

- **Minor**: "Embeddings from word2vec-era models" — distributed word representations predate word2vec (Bengio 2003, Collobert & Weston). The table's "word2vec (2013)" overstates origin. Consider "popularized by word2vec."

## 2. Citation Validity

All eight entries correspond to real, correctly-attributed works (Goodfellow 2016, Vaswani 2017, Bahdanau 2015, He 2016, Shazeer 2017, Ouyang 2022, Rafailov 2023, Gu & Dao 2023). No fabrications detected. Claims are supported by their citations.

- **Formatting bug**: `gu2023mamba` uses `@inproceedings` with `booktitle={arXiv preprint arXiv:2312.00752}`. Should be `@article`/`@misc` with `journal`/`note` — this will render incorrectly under `plain`.
- `ouyang2022training` is `@article` but NeurIPS 2022 — acceptable but `@inproceedings` is more accurate.
- Several substantive claims (attention-as-explanation, MoE 1990s roots, RLVR) lack citations as noted above.

## 3. Pedagogy

Strong. The module directly answers the question, the "shared skeleton vs. swapped mixer" framing is memorable and grad-appropriate, and the four learning objectives are each discernibly addressed. The transfer-mistakes section adds genuine value beyond a list. Scope is well-controlled.

- **LaTeX bug / garbled sentence** (end of Sec. 2): `\emph{ Transformers are a special case of "neural sequence model" that won the scaling race, not a new kind of machine.}` — the `\emph{...}` wraps mismatched text and the straight quotes won't render as proper quotes. This sentence is broken and must be fixed; it's the module's punchline.

## 4. Code (conceptual)

The shared-loop design cleanly demonstrates the thesis (same `train`, `xent_loss`, SGD across three architectures). `make_data`, `xent_loss`, `softmax`, and the `TinyConv` forward/backward look conceptually correct, including the causal padding and the `np.add.at` embedding-gradient scatter.

- **Critical**: `TinyRNN.backward` is **truncated** mid-line (`dE = np.z`). The file as shown is incomplete, and the `TinyTransformer` class and `__main__` driver promised by the docstring ("Trains (a)... (b)... (c)...") are absent. The docstring also claims it "prints the fraction of forward-pass code lines that are architecture specific vs shared" — no such code is present. Either the submission was cut off or the deliverable is incomplete. This must be resolved (confirm the full file compiles and runs as claimed).
- The RNN `backward` as partially shown does not yet implement BPTT through `Wh` — need to verify the completed version actually backprops through the recurrence (otherwise it undercuts the "RNN learns the Markov structure" point).

## Quiz
Correct and well-calibrated; the three-part answer is accurate.

---

**Strongest aspect:** The central conceptual framing (shared differentiable-SGD-backprop-tensor skeleton vs. swapped mixer / objective / interface / post-training) is clear, accurate, and pedagogically excellent.

**Weakest aspect:** The code deliverable is incomplete/truncated (RNN backward cut off, Transformer class and the advertised line-counting output missing), so it does not yet demonstrate what the docstring promises.

VERDICT: APPROVE WITH MINOR CHANGES (see reviewer correction above — code verified complete & running)

1. Provide the complete `shared_skeleton.py`: finish `TinyRNN.backward` (with correct BPTT through `Wh`), include the `TinyTransformer` class, the `__main__` driver training all three, and the promised shared-vs-architecture-specific line-count output. Confirm it runs.
2. Fix the garbled punchline in Sec. 2 — repair the `\emph{}` scope and the mismatched/straight quotes.
3. Fix the `gu2023mamba` entry to `@article`/`@misc` with proper `note`/`eprint` fields so it renders correctly.
4. Add citations (or explicit "recent practice" h
