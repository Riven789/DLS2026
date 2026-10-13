# Decoder-only dense transformer tutorial

A transparent PyTorch implementation accompanying a 20-slide undergraduate lecture. It includes a reversible UTF-8 byte tokenizer, learned token and position embeddings, pre-LayerNorm blocks, causal multi-head attention, a dense 4d GELU feedforward network, tied vocabulary projection, next-token cross-entropy, AdamW, and autoregressive generation.

The 120,704-parameter classroom model is a teaching implementation, not a pretrained general-purpose LLM. It learns a deliberately small synthetic grammar. The architecture scales conceptually, but the single-process harness is not a production distributed training system.

## Install

Python 3.10 or later and PyTorch 2.2 or later are required. This package was executed with Python 3.12 and PyTorch 2.14.1+cpu. A virtual environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For CUDA, use the PyTorch installation matching your GPU/driver environment: https://pytorch.org/get-started/locally/. No model or dataset downloads are needed. `--device auto` selects CUDA, then Apple MPS, then CPU. Only CPU execution was verified for the delivered package.

## Run the examples and tests

```bash
python examples.py
python -m unittest discover -v
```

`examples.py` prints exact tokenizer results, a toy BPE merge trace, embedding lookup plus position, LayerNorm, softmax, weighted values, a complete numerical FFN, GELU values, cross-entropy, and temperature comparisons. These match the slides. BPE merges are hand-specified for explanation; the training model uses bytes.

The tests check UTF-8 round trips, independence from future tokens, zero attention above the causal diagonal, row normalization, manual/SDPA agreement, the softmax-cross-entropy derivative, a small batch learning, generation length/context checks, and exact CPU checkpoint/resume with dropout.

## Reproduce the classroom run

```bash
python corpus.py
python train.py --steps 300 --context 64 --width 64 --layers 2 \
  --heads 4 --batch-size 8 --lr 0.001 --out runs/classroom --device cpu
python generate.py --checkpoint runs/classroom/best.pt \
  --prompt 'The atom' --tokens 75 --temperature 0 --device cpu
```

The archive includes `runs/classroom/best.pt`, `last.pt`, `run.json`, and `metrics.jsonl` from the measured run. Running the command in that same directory replaces those run files, so choose another `--out` to retain the original results.

| Measured setting | Value |
|---|---:|
| Synthetic documents | 288 |
| Training / held-out documents | 260 / 28 |
| Vocabulary | 258 |
| Layers / width / heads | 2 / 64 / 4 |
| Parameters | 120,704 |
| Optimizer steps | 300 |
| Token targets processed | 153,600 |
| Initial held-out loss | 5.56 nats (rounded) |
| Final held-out loss | 0.814793 nats |
| Final byte-token perplexity | 2.258709 |
| Final held-out next-byte accuracy | 0.772656 |

These are measurements on fixed sampled windows from a tiny synthetic distribution. They establish that the implementation learns; they do not establish broad language or scientific capability. Example greedy output from the included checkpoint is imperfect:

```text
The atome meatores at. Scientists compare the model with data.
```

## Train on your own text

Save UTF-8 documents separated by blank lines. At least ten distinct documents and enough tokens per split to exceed the context length are required.

```bash
python train.py --data your_documents.txt --out runs/custom \
  --context 256 --width 256 --layers 6 --heads 8 \
  --batch-size 8 --accum 4 --steps 5000 --warmup 100 --device cuda \
  --attention sdpa --precision bf16
```

This command is an example configuration, not a measured performance claim. Memory requirements depend on hardware. Use `--precision fp32` for CPU, MPS, or CUDA without bf16 support. The model defaults to explicit attention to expose the math; `--attention sdpa` uses PyTorch's equivalent optimized operation. Effective batch size is `batch_size × accum`. No fp16 scaler is needed for the supported fp32/bf16 paths.

The harness removes exact duplicates before a deterministic document-level split. For real corpora, additionally deduplicate near duplicates and split related sources/groups together. The byte vocabulary is fixed and uses no held-out fitting. EOS and BOS delimit documents, but windows can cross those boundaries. This model allows attention across the explicit separators. No padding or padding mask is needed for the fixed-length sampled windows.

## Checkpoints and resume

```bash
python train.py --steps 500 --stop-after 250 --out runs/resume_demo
python train.py --steps 500 --out runs/resume_demo \
  --resume runs/resume_demo/last.pt
```

Use the same model, data, batch, optimizer, precision, and total-step horizon. `--steps` always denotes the total target step count. `--stop-after` ends early without altering the scheduled horizon. `last.pt` contains weights, optimizer state, step, best validation score, RNG states, model config, and corpus hash. `best.pt` updates when validation improves. The integration test confirms exact interrupted/uninterrupted equality on CPU. Floating-point results can differ across hardware or PyTorch versions. Only load checkpoints you trust; the scripts use `weights_only=True`.

## File map

| File | Purpose |
|---|---|
| `tokenizer.py` | UTF-8 byte tokenizer and toy BPE merge trace |
| `dense_lm.py` | Config, attention, block, model, generator, optimizer |
| `train.py` | Data split, batching, training, evaluation, checkpoint/resume |
| `generate.py` | Command-line prompt completion |
| `corpus.py` | Original synthetic teaching dataset |
| `examples.py` | Numerical examples from the slides |
| `test_model.py`, `test_resume.py` | Mathematical and integration checks |
| `LECTURE_NOTES.md` | Expanded explanations for all 20 slides |
| `REFERENCES.md` | Primary sources and documentation |
| `equations/` | LaTeX source for the displayed mathematics |

## How the equations map to code

`nn.Linear` stores weights as `[output,input]`. Row-vector equations write `xW+b`, while the module applies `x @ weight.T + bias`. `E` has shape `[V,d]`, so `F.linear(x,E)` is the tied vocabulary projection `xEᵀ`.

The implementation uses learned absolute positions, LayerNorm epsilon 1e-5, exact GELU, biases in attention and FFN projections, and no output-head bias. Dropout defaults to zero. Weight decay applies to all matrices, including token and position embeddings, and excludes biases and LayerNorm vectors. These are stated baseline choices, not requirements for every transformer.

At inference, temperature zero selects argmax. Positive temperature samples, optionally after top-k filtering. BOS is suppressed and EOS stops output. Generation recomputes a rolling context without a KV cache. When the context fills, the most recent window is reindexed from position zero. This is a simple teaching convention, not a method for extending the trained context. Arbitrary generated bytes may contain invalid UTF-8 and decode with replacement characters.

## Exercises

1. Change scores `[2,1,0]` to `[20,10,0]`. Explain why softmax becomes sharply concentrated.
2. Verify by hand that weights `[0.665,0.245,0.090]` and the value vectors on slide 10 give `[0.935,0.579]`.
3. Replace GELU with the identity in an experiment. Show algebraically why two affine FFN layers then collapse to one.
4. Change a suffix of the input and confirm earlier logits are unchanged. Remove masking in a separate copy and observe the failure.
5. Compare greedy, temperature 0.5, and temperature 1.2 generation using the same checkpoint and prompt.
6. Increase depth while holding width fixed. Measure parameter count and validation loss, without interpreting the synthetic task as a real language benchmark.
