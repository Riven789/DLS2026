# Lecture Topics — DLS-2026

Topical lecture kits: each topic directory gathers the **slides**, the
**lecture notes**, the **reading list**, and the **papers** for one lecture
block. Topic kits complement the chronological `Lecture-notes/Week-N/`
directories (which follow the calendar).

| Topic | Slides | Notes | Papers | Status |
|---|---|---|---|---|
| [`Scaling-Laws/`](Scaling-Laws/) | `scaling_laws_lecture.pdf` (30 slides) | `scaling_laws_notes.pdf` (43 pp companion, one section per slide) + `scaling_laws_explained.pdf` (52 pp extended) | 44 papers in 8 sections (before-LMs → Kaplan/Chinchilla → refinements → architecture knobs → loss-to-capability → why power laws → ladders → frontier case studies) | complete |
| [`Post-Training/`](Post-Training/) | `posttraining_tutorial.pdf` (32 slides, "Post-Training, Visually") | `posttraining_notes.pdf` (67 pp companion, one section per slide) | 79 papers in 9 sections (SFT → preference optimization/RLHF/DPO → RL with verifiable rewards → reasoning/test-time compute → long context → safety → tools & agents → open recipes → evaluation) | complete |
| [`Sparsity-and-Precision/`](Sparsity-and-Precision/) | `sparsity_precision_tutorial.pdf` (33 slides) + `sparsity_precision_tutorial_revised.pdf` (17-slide undergraduate edition) | `sparsity_precision_notes.pdf` (66 pp companion, one section per slide) + LaTeX source | 106 papers in 9 sections (cost model → MoE → pruning → activation/attention sparsity → number formats → low-precision training → precision scaling laws → inference quantization → practice) | complete |
| [`Pre-training-Case-Study/`](Pre-training-Case-Study/) | `pretraining_tutorial.pdf` (35 slides, "Frontier Pre-training, Visually" — MAI-Base-2) | `pretraining_notes.pdf` (62 pp companion) + `Frontier-Pre-training-at-Microsoft-AI.pdf` (13 pp, the MAI report it visualizes) + `deepseek2024v3.pdf` (53 pp DeepSeek-V3 technical report) | 52 papers in 9 sections (overview → the model → scaling ladders → hyperparameter laws → MoE systems → Muon → numerics/stability → data & vision → comparators) | complete |
| [`Transformer/`](Transformer/) | `transformer_tutorial.pdf` (31 slides, "The Transformer, Visually") — also in `Lecture-notes/` Week-2/3 | `transformer_notes.pdf` (58 pp companion, one section per slide) | 56 papers in 7 sections (residual stream → tokens/position → attention → MLP/norm/block → output/generation → training & scale → family) | complete |
| [`LLM-Runtime/`](LLM-Runtime/) | `llm_runtime_tutorial.pdf` (31 slides) + `prefill_decode_disaggregation_slides.pdf` (3-slide focused deck) | `llm_runtime_notes.pdf` (65 pp companion, one section per slide) | 67 papers in 8 sections (KV cache → scheduling/speculation/disaggregated serving/quantization → engines: vLLM/SGLang/TensorRT-LLM/llama.cpp/Ollama/MLX → measuring & operating → API layer: structured output/tool calling/reasoning → tool interfaces & MCP → web search & retrieval → agent loops & serving agents) | complete |

## Reading lists

Each topic's papers directory contains `README.md` (annotated reading list,
grouped by lecture section), `reading_list.csv` and `reading_list.bib` (same
list as data), and one subdirectory per lecture section holding the PDFs.
Paper sets are also available as zips in the Dropbox student download area.

## Notes

- Post-Training lecture notes are still materializing; the slides and the full
  reading list are ready and staged here.
- The Pre-training case study is a visual companion to Microsoft AI's
  *Frontier Pre-training at Microsoft AI* (2026) report on MAI-Base-2 —
  scaling ladders, hyperparameter laws, MoE systems, Muon, stability, data —
  with DeepSeek-V3 as the second frontier case.
