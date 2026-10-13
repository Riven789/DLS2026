# Q20 QA and review record

- **Author:** Paradise
- **Survey source:** pending author traceability entry
- **Status:** revision requested after first independent technical/evidence review; teaching/release review pending.

## Author self-check

- [x] Five baseline PDFs build locally (note: 3 pages; deck: 10 slides).
- [x] Bibliography keys referenced by `note.tex` exist in `bibliography.bib`.
- [x] Code executes dependency-free.
- [ ] Correct synthetic-data entropy-floor reporting and RNN backpropagation; see `reviews/Q20-review-Ollie.md`.
- [ ] Add module README, companion-video/alignment record, and visual-inspection evidence.

## Technical/evidence review — Ollie (2026-10-10)

**CHANGES REQUESTED.** Evidence and required fixes: `reviews/Q20-review-Ollie.md`.

## Teaching/release review — Kukla

Pending.

## Author revision and independent recheck

Pending.

## Revision 1 (2026-10-10, Paradise) — responding to Ollie (CHANGES REQUESTED) + Kukla (minor)
- `make_data` now returns the actual transition matrix `P`; the printed floor is computed from that same matrix (was: an independent re-draw — both reviewers caught it).
- Floor relabeled as the Bayes-optimal next-token cross-entropy of the source; added train/held-out split (48/16 sequences). Held-out loss is the honest comparison; train loss below the floor on a finite sample is memorization, not a bound violation.
- Removed the dead `dh_raw = ... * 0` line in `TinyRNN.backward` (Kukla's nit; BPTT recurrence itself was already correct via `dh_next`).

```
  step 800 loss 1.6399
  step 900 loss 1.6217
  step 1000 loss 1.5993
  step 1100 loss 1.5791
  step 1200 loss 1.5665
  step 1300 loss 1.5550
  step 1400 loss 1.5413
  train loss 1.5234 | held-out 2.6808 | Bayes CE 1.717

Held-out loss -> Bayes CE from above = learned the source.
Train loss < Bayes CE on a finite sample = memorization.
Shared code lines (data/loss/loop/softmax): the 'skeleton'.
Architecture-specific lines: forward/backward mixer only.
=> Q20's claim, made concrete: swap the mixer, keep everything else.
```
