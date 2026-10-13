# Q09 QA and build record

- **Author:** Paradise
- **Topic:** How the composition of training data shapes what a model knows and can do (survey Q09): corpora composition, dedup (exact + MinHash), quality filtering (FineWeb-Edu, perplexity), mixture weighting/annealing, code→math transfer, Muennighoff ~4-epoch reuse, curriculum ordering, tokenizer–mixture interactions, quality vs. scale.
- **Status:** drafted; technical review (Ollie) and teaching review (Kukla) pending.

## Author self-check

- [x] Five baseline PDFs build locally: note 5 pages, deck 10 slides, discussion 1 p, projects 1 p, quiz 1 p.
- [x] `pdflatex` + `bibtex` full cycle for `note.tex`; zero undefined citations; every `\cite` key in `note.tex` resolves in `bibliography.bib` (13 entries, all cited).
- [x] Deck uses textual author–year references (Beamer pattern from Q20); no undefined references.
- [x] No overfull boxes in note or deck.
- [x] Code runs dependency-free (NumPy + stdlib) with the repo venv; all assertions PASS.

## Citation verification (all 13 bib entries checked against primary sources, 2026-10-11)

| Key | Verified |
|---|---|
| penedo2024fineweb | arXiv:2406.17557, NeurIPS 2024 D&B track — 15T tokens / 96 CC snapshots; FineWeb-Edu 1.3T; MMLU 33→37, ARC 46→57; global dedup 20T→4T hurt vs per-snapshot |
| muennighoff2023scaling | arXiv:2305.16264, NeurIPS 2023 — 400+ runs, ≤9B params, ≤900B tokens; ≤4 epochs ≈ fresh; code aug ≈2× effective tokens; data-constrained optimum = smaller model + more epochs |
| lee2022deduplicating | arXiv:2107.06499, ACL 2022 — 61-word sentence ×60,000 in C4; ~10× less memorized text; fewer steps to same accuracy; eval overlap |
| kocetkov2022stack | arXiv:2211.15533 — The Stack, 3.1 TB, 30 languages, permissively licensed; near-dedup boosts code-LM performance |
| hoffmann2022chinchilla | arXiv:2203.15556 — 70B params / 1.4T tokens vs Gopher 280B/300B; equal parameter–token scaling |
| longpre2023pretrainer | arXiv:2305.13169 (also NAACL 2024) — 28× 1.5B models; temporal shift degrades, not repaired by finetuning; heterogeneous sources beneficial; filter effects not predictable from domain stats |
| wenzek2020ccnet | LREC 2020 — CCNet: LM-perplexity (Wikipedia-trained) document filtering, head/middle/tail splits |
| xie2023doremi | arXiv:2305.10429, NeurIPS 2023 — DoReMi group-DRO proxy; +6.5 pts few-shot, 2.6× fewer steps at 8B on The Pile |
| gunasekar2023textbooks | arXiv:2306.11644 — phi-1; <1B synthetic textbook tokens + filtered code; strong HumanEval/MBPP |
| villalobos2024will | arXiv:2211.04325, ICML 2024 — data stock fully utilized 2026–2032 (median 2028); ~4e14 effective tokens indexed web |
| mindermann2022prioritized | arXiv:2206.07137, ICML 2022 — RHO-LOSS; 18× fewer steps on Clothing-1M |
| sennrich2016neural | ACL 2016 — BPE subword units for open-vocab NMT |
| hu2024minicpm | arXiv:2404.06395 — MiniCPM two-stage; annealing stage raises math+code to ~70% |

Unverified citations: **none**.

## Code output (real, repo venv, 2026-10-11)

Command: `/home/stevens/ai-course-2026/.venv/bin/python code/minhash_dedup_demo.py` (exit 0; 4/4 PASS assertions)

```
=== Part 1: MinHash correctness ===
mean true Jaccard      : 0.5455
empirical P[sig match] : 0.5461
PASS: P[one-coordinate signature agreement] ~= Jaccard (|diff| < 0.02)

corpus: 240 docs = 120 originals + 120 planted near-dups
kept 120; removed 120
near-dups removed : 120/120
originals kept    : 120/120
PASS: >90% of planted near-dups removed, >90% of distinct docs retained

=== Part 2: dedup effect on memorization proxy ===
duplicated corpus : 600 docs, 150 unique seeds (4x repetition)
after MinHash dedup: 148 docs kept

training budget for both runs: 600 doc-visits
D: duplicated corpus, 1 epoch           : seen(seed) 4.0524 | unseen 4.4191 | gap +0.3666
U: deduped seeds + fresh docs, 1 epoch  : seen(seed) 4.1789 | unseen 4.2892 | gap +0.1103
PASS: at equal budget, duplicates memorize (low seen-loss, no unseen gain);
      replacing them with unique docs improves unseen loss and shrinks
      the seen/unseen gap (the memorization proxy).

repetition sweep (only the deduped corpus available, fixed content):
 epochs  visits     seen   unseen      gap
      1     148   4.3140   4.5247  +0.2107
      2     296   4.2023   4.4832  +0.2809
      4     592   4.0897   4.4503  +0.3606
      8    1184   4.0226   4.5117  +0.4891
     16    2368   3.9620   4.6046  +0.6425
     32    4736   3.9346   4.6805  +0.7459
PASS: repetition up to ~4 epochs is roughly free for unseen loss, while
      seen loss keeps falling and the gap grows; gains from 16 -> 32
      epochs are smaller than from 1 -> 4 -- diminishing returns, the
      miniature of the ~4-epoch rule of Muennighoff et al. (2023).

Interpretation:
* Duplicates give repeated gradient steps on the same content: lower loss on
  the duplicated docs (memorization) at no benefit to unseen docs. This is
  the efficiency + memorization argument for deduplication (Lee et al. 2022).
* Reallocating the same budget to unique documents (U) improves generalization
  and shrinks the seen/unseen loss gap.
* If unique data is scarce, repeating it for a few epochs is nearly free, but
  returns decay with more repetition (Muennighoff et al. 2023).
```

Runtime: <1 s. Deterministic (seeded throughout).

## Known limitations / honest hedging

- The memorization proxy is a bag-of-words mixture model, not a Transformer: it demonstrates the *statistics* of duplication vs. unique data at equal budget, not transformer memorization mechanics.
- Part 2's "4 epochs" tuning (EMA step α=0.15) is chosen so the miniature's unseen-loss optimum lands at moderate repetition; the exact turning point is a property of this toy, not a derivation of Muennighoff et al.'s law.
- Code→math transfer, annealing-at-scale, and tokenizer–mixture magnitudes are flagged in the note as practitioner priors with thin controlled-evidence base.

## Review trail

- Technical/evidence review — Ollie: pending.
- Teaching/release review — Kukla: pending.
