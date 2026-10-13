# LSTM Networks from the Ground Up

A 20-slide tutorial on recurrent networks and the LSTM, with the code that produced every number
and figure on the slides.

## Files

| file | what it is |
|---|---|
| `slides/lstm_tutorial.pdf` | the deck |
| `slides/lstm_tutorial.tex` | Beamer + TikZ/pgfplots source (all diagrams are source, no images) |
| `lstm_numpy.py` | an LSTM layer in NumPy: one step, the forward pass, backpropagation through time by hand; checked against `torch.nn.LSTM` and finite differences |
| `lstm.py` | PyTorch models: `LSTMCellScratch` (one `nn.Linear`, mirrors the six equations), `ScratchLSTM`, and `SeqModel` (embedding, `nn.RNN` / `nn.LSTM` / `nn.GRU`, linear head) |
| `data.py` | sequences to tensors: the adding problem, sliding windows with a chronological split and train-only scaling, characters to ids, parallel text streams; the baselines (persistence, seasonal-naive, linear autoregression, unigram and bigram counts) |
| `train.py` | training and testing harness: one loop for all tasks (truncated BPTT with carried state, gradient clipping, best-checkpoint / early stopping), test report with baselines, text sampling with temperature |
| `experiments.py` | runs every experiment on the slides and writes `slides/generated/*` |
| `walkthrough.py` | the hand calculations on the slides |
| `test_lstm.py` | eight checks (NumPy vs PyTorch, BPTT vs finite differences, parameter count, the slide's worked example, no leakage in the split, packing, carried state, chance-level start) |
| `data/monthly-sunspots.csv` | 2,820 monthly sunspot numbers, 1749-1983 (from `jbrownlee/Datasets` on GitHub; original source SILSO) |
| `data/tinyshakespeare.txt` | 1.1 MB of Shakespeare (from Karpathy's char-rnn repository) |

## Run it

Requires Python 3.10+, NumPy and PyTorch. Everything runs on a CPU.

```bash
python test_lstm.py                                   # ~15 s
python walkthrough.py
python lstm_numpy.py
python train.py --task adding --seq_len 50            # LSTM solves it
python train.py --task adding --seq_len 50 --model rnn    # the simple RNN does not
python train.py --task sunspots                       # 1-month-ahead forecast + three baselines
python train.py --task sunspots --horizon 6
python train.py --task shakespeare                    # ~2 minutes; loss, perplexity, two samples
python experiments.py                                 # everything on the slides, 6-10 minutes on 2 CPU cores
python experiments.py grad clip                       # or only some sections
cd slides && latexmk -pdf lstm_tutorial.tex
```

Every hyperparameter is a flag (`python train.py --help`): `--model lstm|rnn|gru|scratch`, `--hidden`,
`--layers`, `--bidirectional`, `--dropout`, `--forget_bias`, `--clip`, `--optimizer`, `--lr`, ...

## Notes

- Slides 15 and 16 include lines of `lstm_numpy.py` and `train.py` by line number
  (`\lstinputlisting[linerange=...]`). If you edit those files, update the ranges in the `.tex`.
- `experiments.py` keeps its numbers in `slides/generated/macros.json` so that single sections can be
  rerun; `results.tex` is rewritten from it every time.
- Results are deterministic for a given seed on one machine (one thread); expect small differences
  across PyTorch versions. Wall-clock times on the slides are from a shared 2-core CPU.
- `torch.set_num_threads(1)` in `train.py`: the models are small enough that extra threads only add
  overhead.
