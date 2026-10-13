# Ten exercises to do with an agent

These exercises assume you have `microgpt.py` in a folder, a coding agent (Claude Code or similar) that can edit and run files, and the lecture notes open. Each one is built on the same rule: **the agent does the typing, you do the predicting.** Write down what you expect before the agent runs anything, then compare. The understanding comes from the gap between your prediction and the result, not from the code the agent produces.

Ground rules for every exercise:

- Keep `random.seed(42)` so runs are comparable, and tell the agent never to change hyperparameters you did not ask it to change.
- Ask the agent to show you the diff before running, and to print only what you asked for.
- Each baseline run takes one to four minutes in pure Python. For sweeps, ask the agent to run experiments in parallel or to reduce `num_steps` consistently across all arms.

---

## 1. Reproduce, then instrument the loss

**Goal.** Know what the printed loss actually measures and why it jumps around.

**Before running, write down:** the loss of a model that guesses uniformly over 27 tokens; whether the loss at step 1 should be above or below that; and whether the per-step loss will go down smoothly.

**Ask the agent to:** run the unmodified script, save every step's loss to a CSV, and plot the raw loss, a trailing 50-step mean, and a horizontal line at your predicted uniform baseline. Then add the document length and the name itself to each row.

**Then you:** sort the last 100 steps by loss and look at the names at both ends. Explain in two sentences why the raw curve is noisy and why the final smoothed value differs from the last printed value. Convert the final smoothed loss to perplexity and bits per character by hand, and have the agent check your arithmetic.

**Stretch:** have the agent hold out 10 percent of names and evaluate them every 100 steps without calling `backward()`. Predict first whether a gap will appear within 1,000 steps, and why not.

Notes sections 7.3 and 7.4.

---

## 2. Bug hunt with a gradient checker you design

**Goal.** Understand `backward()` well enough to catch it when it is wrong.

**Ask the agent to:** make three copies of the script, each with one silent bug in the autograd engine, and not tell you which bug is in which copy. Suggested bugs: replace `child.grad += ...` with `=`; skip the `p.grad = 0` reset; iterate `topo` forward instead of reversed; return the wrong local gradient for `__pow__`.

**You write the spec** for a finite-difference gradient checker: which parameters to sample, what step size `h` to use, how to compare analytic and numeric gradients (relative error, not absolute). The agent implements it to your spec.

**Then you:** run the checker on all three copies and on the original, and identify each bug from the checker output and the loss curve alone, before reading the diffs. Explain why a too-large `h` and a too-small `h` both produce false alarms.

Notes section 4; exercise 2 in the notes.

---

## 3. The bigram baseline and what attention adds

**Goal.** Put a number on how much the Transformer buys over counting.

**Before running, write down:** your guess for the per-token loss of a bigram model (a 27 by 27 table of next-letter counts) on the names, relative to microgpt's roughly 2.3.

**Ask the agent to:** implement `train0.py` from the progression table (counts, add-one smoothing, row-normalise), compute its average loss over the full dataset, and sample 20 names from it. Then do the same for a trigram model.

**Then you:** fill in a three-row table (bigram, trigram, microgpt) with loss and a qualitative judgement of the samples. Explain what information the trigram has that the bigram does not, and what microgpt has that neither has. Finally, ask the agent to shrink microgpt's `block_size` to 2 and retrain; predict whether it lands near the bigram or the trigram, and why.

Notes section 10; Karpathy's `build_microgpt.py` revisions.

---

## 4. The ablation ladder

**Goal.** Learn what each architectural piece contributes by removing it.

**Before running, rank these five ablations from least to most damaging:** (a) delete the position embedding `wpe`; (b) delete the two residual additions; (c) delete the first `rmsnorm` after the embeddings; (d) use one head of width 16 instead of four heads of width 4; (e) delete the MLP block entirely.

**Ask the agent to:** implement each ablation as a command-line flag in a single copy of the script, run all five plus the baseline with the same seed, and report the trailing 100-step mean loss and the parameter count for each.

**Then you:** compare the results with your ranking. For every place your ranking was wrong, write one sentence about what you had misunderstood. Pay special attention to (c), which the source comments is "not redundant", and to (d), which changes no parameter count at all.

Notes sections 6.3 to 6.8.

---

## 5. Look inside attention

**Goal.** Test the post's story that a head might ask "what vowels appeared recently?"

**Ask the agent to:** after training, add a hook that records the attention weights of every head at every position for a given input name, and render one heatmap per head (rows are query positions, columns are cached key positions). Run it for five names you choose, including one long one and one with a repeated letter.

**Then you:** describe each head's pattern in words before reading any analysis the agent offers. Common findings are a "previous token" head, a "first token" head, and a diffuse head. Then design one test of a hypothesis: for example, if you believe a head attends to the most recent vowel, have the agent compute the average weight on the most recent vowel versus other positions across 500 training names. Decide whether your hypothesis survives.

**Stretch:** have the agent train with `n_layer=2` and repeat. Ask whether second-layer heads look different in kind.

Notes section 6.4; Elhage et al. 2021 for the residual-stream view.

---

## 6. Port it to tensors and prove equivalence

**Goal.** See that the scalar graph and a vectorised implementation are the same algorithm.

**Ask the agent to:** write a PyTorch version that processes a whole document in one forward pass with a causal mask, using exactly the same initial weights (have it copy the `state_dict` values out of the scalar version after seeding). Then compare the logits at every position for the first training document and the gradients of every parameter after one backward pass.

**Then you:** decide what tolerance counts as "equal" and justify it. Then answer: where did the KV cache go in the tensor version, and why does it need a mask when microgpt does not? Time one training step in both versions and explain the ratio in terms of the node count table in the notes.

**Stretch:** make the agent batch 32 documents with padding, and have it explain to you how padded positions are excluded from the loss. Check that explanation against the code.

Notes sections 4.4, 6.5, 9.

---

## 7. Optimizer bake-off with a fair protocol

**Goal.** Understand what each term of Adam is doing.

**You write the protocol** first: the same number of documents for every arm, a learning-rate sweep of at least five values per arm, and the metric (trailing 100-step mean). The agent is not allowed to choose learning rates for you.

**Ask the agent to:** implement plain SGD, SGD with momentum, RMSProp, Adam without bias correction, and Adam (the original), each as a flag, and run the sweep.

**Then you:** plot best-of-sweep loss per optimizer and the curve of the best run in each arm. Explain what the first 20 steps look like without bias correction and why. Then ask the agent to print the ratio of the Adam update to the raw gradient for ten parameters, and explain why rarely used embedding rows (letters like `q` and `x`) get larger steps.

Notes section 7.5; Kingma and Ba 2015.

---

## 8. Decoding lab: memorisation versus novelty

**Goal.** Make "hallucination" and "creativity" measurable.

**Ask the agent to:** implement greedy, temperature (0.3, 0.5, 0.8, 1.0, 1.5), top-k (k = 3, 10) and nucleus (p = 0.7, 0.9) decoding as options in the inference loop, generate 500 names per setting, and report: fraction of samples that appear in the training set, number of unique samples, mean length, and the ten most frequent outputs.

**Before seeing results, predict:** which setting memorises most, which produces the most duplicates, and which produces unpronounceable output.

**Then you:** write a short paragraph on the trade-off between plausibility and diversity using the numbers, and explain why greedy decoding from `BOS` can only ever produce one name. Relate this to Holtzman et al. 2020.

Notes section 8; study guide inference cheat sheet.

---

## 9. A miniature scaling study

**Goal.** See the shape of scaling laws at a scale where you can afford to run everything.

**Ask the agent to:** run a grid over `n_embd` in {8, 16, 32, 64} and `n_layer` in {1, 2, 4} for 2,000 steps each, recording parameter count, final smoothed loss, and wall-clock time. Have it estimate training FLOPs for each run as 6 times parameters times tokens processed, and plot loss against parameters and loss against estimated FLOPs on log axes.

**Before running, predict:** whether doubling width or doubling depth helps more at this scale, and whether the largest model will be best after only 2,000 single-document steps.

**Then you:** identify which runs are undertrained (loss still falling steeply at the end). Write two paragraphs connecting what you see to Kaplan et al. 2020 and to the Chinchilla rule of roughly 20 tokens per parameter, and compute how many tokens the largest model in your grid would need to reach that ratio.

Notes section 11.5; study guide training cheat sheet.

---

## 10. New data, new tokenizer, and a measurement trap

**Goal.** Experience the "what if I change the dataset?" FAQ item, and learn why losses are only comparable per character, not per token.

**Ask the agent to:** train on a different list of short strings (English words, city names, chemical elements, anything you have a text file of). Then implement a minimal byte-pair encoding on the names with 20, 50 and 100 merges, retrain microgpt with each vocabulary, and report the per-token loss.

**Before seeing results, predict:** whether per-token loss will rise or fall as the number of merges grows, and whether that means the model got worse.

**Then you:** insist the agent also report loss per character, computed by dividing the total negative log-likelihood of each document by its character count. Compare the two tables and explain which one is the honest comparison and why. Finally, inspect the merges: which pairs merged first, and does that match the bigram counts from exercise 3?

Notes sections 2, 3, 11.2; Sennrich et al. 2016.

---

## How to tell the exercises worked

For each exercise you should be able to close the editor and, from memory, (1) state the prediction you made, (2) state the result, and (3) explain the mechanism behind any difference. If you can only do (2), you watched the agent work; redo the prediction step on a variant the agent did not run.
