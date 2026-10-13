# The Transformer, from Tokens to Text

A 20-slide tutorial on the decoder-only, dense transformer LLM, with the code
that produced every number on the slides.

## Files

| file | what it is |
|---|---|
| `slides/transformer_tutorial.pdf` | the deck |
| `slides/transformer_tutorial.tex` | Beamer + TikZ/pgfplots source (all diagrams are source, no images) |
| `tokenizer.py` | character tokenizer and byte-level BPE (train / encode / decode) |
| `model.py` | the model: causal multi-head attention, GELU FFN, pre-LN blocks, tied unembedding, sampling |
| `train.py` | training harness: AdamW, warmup + cosine decay, gradient clipping, eval, checkpoints, CSV log |
| `walkthrough.py` | every stage by hand in NumPy (the worked examples on the slides), then the same stages inside the trained model |
| `test_model.py` | sanity checks: initial loss = ln V, causality, attention rows sum to 1, overfit one batch, sampling filters |
| `make_figures.py` | turns a finished run into `slides/generated/*` (loss curve, attention heat maps, sample) |
| `out/` | the run shown on the slides: `log.csv`, `train.log`, `tokenizer.json`, `ckpt.pt` |
| `data/tinyshakespeare.txt` | 1.1 MB training corpus (from Karpathy's char-rnn repository) |

## Run it

Requires Python 3.10+, PyTorch and NumPy.

```bash
python test_model.py        # under a minute
python walkthrough.py       # prints the worked examples
python train.py             # ~0.9M parameters; about half an hour on 2 CPU cores, far less on a GPU
python make_figures.py      # refresh slides/generated from the new run
cd slides && latexmk -pdf transformer_tutorial.tex
```

Every hyperparameter is a flag (`python train.py --help`). A larger run:

```bash
python train.py --out_dir out_big --n_layer 6 --n_head 6 --n_embd 384 \
                --block_size 256 --batch_size 64 --max_steps 5000 --lr 6e-4
```

`--tokenizer char` switches to character-level tokens (use a fresh `--out_dir`).

## Notes

- Slides 17 and 18 include lines of `model.py` and `train.py` by line number
  (`\lstinputlisting[linerange=...]`). If you edit those files, update the ranges.
- The deck uses a 192 x 108 mm canvas (16:9) with the metropolis theme and
  compiles with pdfLaTeX from TeX Live.
