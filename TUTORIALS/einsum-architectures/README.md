# Einsum Architectures — 20-page Beamer tutorial

Course package for **DeepLearningSystems-2026**. The deck develops Einstein summation from index notation through MLPs, CNNs, GNNs, LSTMs, and Transformers, then connects the equations to contraction order, layout, fusion, sparsity, and memory traffic.

## Contents

- `deck.tex` — self-contained Beamer source (TikZ diagrams; no external assets)
- `deck.pdf` — compiled, exactly 20 pages
- `deck.log` — retained compilation log
- `Makefile` — build and verification targets
- `SHA256SUMS` — artifact hashes
- `verification.txt` — machine-readable QA receipt

## Build

```bash
make
make verify
```

The default compiler is `~/bin/tectonic`; override it with `make TECTONIC=/path/to/tectonic`.

## Verification contract

`make verify` requires:

1. `pdfinfo` reports exactly 20 pages;
2. the log contains no fatal LaTeX errors;
3. the log contains no overfull vertical boxes (a clipping proxy);
4. `pdftotext` extracts the deck title.
