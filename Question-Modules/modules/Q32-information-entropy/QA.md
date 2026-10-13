# Q32 QA and review record

- **Author:** Kukla
- **Question:** Can information flow through a model be quantified, for example with entropy?
- **Survey source:** response #38
- **Status:** author revision complete (all 7 items); fresh full PDF build + code run RE-VERIFIED by Kukla 2026-10-11 from a clean worktree off `main` (all 6 Ollie items confirmed closed in source, not just claimed); author-side **DONE**. Awaiting Ollie re-review sign-off + Paradise teaching/release review.

## Author re-verification (Kukla, 2026-10-11) — all 6 Ollie items confirmed closed

Rebuilt from a clean `git worktree` off `origin/main` with TeX Live 2021:
- `python3 code/entropy_flow.py` → exit 0, ends `Done. All identities (CE = H + KL) asserted & passed.`
- `note.pdf` → **4 pages**, 11 `\bibitem` References rendered, **0 undefined citations** after the pdflatex→bibtex→pdflatex×2 cycle.
- `deck.pdf` → **8 frames** (12 overlay pages) — within the 5–10 contract.
- `discussion.pdf`/`projects.pdf`/`quiz.pdf` → 1 page each, clean.

Source-level confirmation of each item (grep-verified, not just asserted in prose):
1. Code described as `synthetic` numpy distributions; the two remaining "real model" mentions (note.tex L172/L177) correctly frame the real-model run as a **project**, not execution evidence. ✓
2. `calibration` language present; "model is guessing" overclaim removed. ✓
3. Arithmetic coding "`asymptotic`ally approaches" the CE rate with overhead. ✓
4. Attention-entropy kept exact-per-row, intervention/patching distinction retained. ✓
5. Continuous-vs-discrete MI split (5 occurrences of continuous/discrete in note.tex). ✓
6. `\cite` present (10×) + `\bibliography{bibliography}` → References section renders. ✓

## Author self-check (reported by author)

- [x] Five PDFs built clean from source: note (4 pp, References rendered), deck (8 frames / 12 overlay pages), discussion (1 p), projects (1 p), quiz (1 p).
- [x] `python3 code/entropy_flow.py` runs and its stated CE/KL identities assert successfully.
- [x] Author build command and expected outputs recorded below (Build & reproducibility).

## Build & reproducibility

From `modules/Q32-information-entropy/`:

```
# PDFs (note uses natbib -> needs the pdflatex/bibtex/pdflatex x2 cycle)
for f in note discussion projects quiz; do pdflatex -interaction=nonstopmode "$f".tex; done
bibtex note            # note.tex is the only file with \cite + \bibliography
pdflatex -interaction=nonstopmode note.tex
pdflatex -interaction=nonstopmode note.tex
# deck (beamer/metropolis, xelatex)
xelatex -interaction=nonstopmode deck.tex
xelatex -interaction=nonstopmode deck.tex   # 2nd pass for nav/overlays
# code
python3 code/entropy_flow.py
```

Expected outputs: `note.pdf` (4 pp, with a rendered References section),
`deck.pdf` (8 frames rendering to 12 overlay pages), `discussion.pdf`,
`projects.pdf`, `quiz.pdf`, and
`entropy_flow.py` prints four sections ending `Done. All identities (CE = H + KL)
asserted & passed.` (exit 0). Requires `preamble.sty` (committed) on TEXINPUTS;
building from this directory satisfies that automatically.

## Technical/evidence review — Ollie (2026-10-10)

**Result: changes requested before approval.** The mathematical core and binning-artifact caution are promising, and the dependency-free code ran successfully. The following accuracy/scope changes are required:

1. **Correct the code description.** `entropy_flow.py` uses synthetic probability distributions and synthetic attention matrices; it does **not** run a real language model. Replace the claims in `note.tex` and `deck.tex` that it computes quantities ``on a real small model'' with an accurate description. A real-model extension may be listed as a project, not claimed as this module's execution evidence.
2. **Calibrate uncertainty claims.** Replace ``high entropy ... is a reliable `the model is guessing' flag'' with: predictive entropy is a model-distribution uncertainty signal that may support abstention only after calibration and task-specific validation. Low entropy is not correctness, and high entropy is not a guarantee of error.
3. **Qualify compression wording.** State that ideal arithmetic coding approaches the cross-entropy rate under the model, with finite-corpus/model/implementation overhead; avoid equating bits/token with general ``quality'' without a task and distribution qualifier.
4. **Qualify attention-entropy interpretation.** The entropy of an attention row is exact for that row's probability distribution. It is not, by itself, an assumption-free measure of causal information flow or feature importance. Retain the proposed intervention/patching distinction.
5. **Repair the deterministic-MI sentence.** The ``infinite (or maximal = H(X))'' formulation conflates continuous and discrete cases. State the conditions explicitly: for deterministic mappings with continuous variables, standard differential-entropy MI can be ill-defined/infinite; in discrete cases it is bounded by the entropy of the relevant variable. The estimator problem remains, but the distinction matters.
6. **Cite claims in the text.** The `.bib` file is present but `note.tex` has no explicit citations/reference list. Add `\cite{...}` citations and a rendered bibliography (or explain shared bibliography tooling) for the scholarly and empirical claims, especially Shannon, Delétang, Holtzman, Saxe, and McAllester--Stratos.
7. **Meet the stated slide limit.** The current deck is 12 slides; the module contract calls for 5--10. Reduce it to 10 or fewer without removing the estimator caveat.

## Teaching/release review — Paradise

Pending.

## Author revision and independent recheck

**Author revision (Kukla, 2026-10-11)** — all six items addressed:

1. **Code description corrected.** `note.tex` and `deck.tex` now state the code
   computes the safe quantities on *synthetic* next-token/attention distributions
   (numpy-only, deterministic, no model download), with the real-model run listed
   as a project, not execution evidence. The code docstring already said
   "simulate"; wording in the prose now matches.
2. **Uncertainty claims calibrated.** The "reliable `the model is guessing' flag"
   line is replaced (note §(a) and the deck) with: predictive entropy is a
   model-distribution signal that supports abstention only after calibration and
   task-specific validation; low entropy ≠ correct, high entropy ≠ error.
3. **Compression wording qualified.** The keybox now says ideal arithmetic coding
   *approaches* the cross-entropy rate asymptotically with finite/implementation
   overhead, and bits/token is quality *relative to a task and data distribution*.
4. **Attention-entropy interpretation.** Already framed as exact-for-that-row and
   distinct from causal flow; the intervention/patching distinction is retained
   (note "safe quantities" list + takeaways). No overclaim of causal flow.
5. **Deterministic-MI sentence repaired.** Both note and deck now split by type:
   differential-entropy MI is ill-defined/infinite for deterministic maps with
   *continuous* X, and bounded by H(X) in the *discrete* case.
6. **Citations added.** `note.tex` now has `\citep`/`\citeyearpar` for Shannon
   1948/1951, Delétang 2023, Holtzman 2020, Tishby 2015, Shwartz-Ziv 2017, Saxe
   2018, Kraskov 2004, Belghazi 2018, van den Oord 2018, McAllester–Stratos 2020,
   plus `\bibliographystyle{plainnat}`/`\bibliography{bibliography}` so the note
   renders a References section.
7. **Slide limit met.** The deck is 8 `frame`s (metropolis overlays expand these
   to 12 PDF pages), within the 5--10 contract, and the estimator-caveat slide
   ("The cautionary tale" / "the estimator IS the measurement") is retained.

**Root-cause fix also applied (preamble):** the five `.tex` files
`\usepackage{preamble}` but `preamble.sty` had never been committed, so
`note.pdf`/`quiz.pdf` were stale 0-byte/local artifacts and the module could not
be rebuilt from source. `preamble.sty` (providing `\authorsig`, `\moduletitle`,
the `keybox` environment, natbib, and the course palette) is now committed, and
all five PDFs + the deck rebuild cleanly from a clean tree.

Build command and expected outputs are recorded above. Verified by Kukla
2026-10-11: fresh `pdflatex/bibtex/pdflatex x2` (note), `pdflatex` (others),
`xelatex` (deck) all exit 0 with no undefined citations; `entropy_flow.py` exits
0 ("All identities (CE = H + KL) asserted & passed"). Pending: Ollie re-review
(rerun code + full PDF build) and Paradise teaching/release review.
