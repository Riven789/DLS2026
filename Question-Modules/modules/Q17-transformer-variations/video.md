# Q17 — Companion videos: decoder-only transformers

Rick's curated progression for understanding decoder-only transformers.
Watch in this order: **visual intuition -> decoder architecture -> working implementation.**
All three verified live (oembed title + lengthSeconds) on 2026-10-11.

## 1. Visual intuition
- **3Blue1Brown — Transformers, the tech behind LLMs (Deep Learning Ch. 5)**
- https://www.youtube.com/watch?v=wjZofJX0v4M  (27:14)
- Best for visual intuition: tokens -> embeddings -> transformer layers -> next-token
  probabilities. Strong on embedding geometry, learned weights, the output projection,
  and softmax with temperature.
- Note: this chapter sets up the overall architecture; the detailed attention math is in
  the following chapter.

## 2. Decoder-only architecture
- **StatQuest with Josh Starmer — Decoder-Only Transformers, ChatGPT's Specific Transformer, Clearly Explained!!!**
- https://www.youtube.com/watch?v=bQ5BoolX9Ag  (36:44)
- Best focused explanation of the decoder-only architecture. Patient, step-by-step, aimed
  at the transformer used for autoregressive generation. Good for students who benefit from
  small concrete examples and a slower pace.

## 3. Working implementation
- **Andrej Karpathy — Let's build GPT: from scratch, in code, spelled out**
- https://www.youtube.com/watch?v=kCc8FmEb1nY  (1:56:20)
- Best for turning understanding into implementation: builds a working GPT-style model in
  PyTorch, connecting architecture to executable code and training. Assumes Python, tensors,
  and basic PyTorch (Karpathy recommends his makemore lessons for that foundation).

## Recommendation
For a technical audience, Karpathy is the single strongest recommendation. For an
undergraduate teaching sequence, the three together give a useful progression:
visual understanding -> decoder architecture -> working implementation.
