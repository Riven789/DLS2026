# DLS 2026 Week 3 — Source Ledger & Reuse Audit

**Prepared:** 2026-10-10  
**Scope:** P0 Week-3 sessions: L1.1 *The Modern AI Stack, End to End*; L1.2 *Deep Learning from First Principles, in Code*; L3.3 *Thinking With AI, Not Instead of It*.

This is a source/reuse audit before deck production. It separates reusable material from material that must be rebuilt or evidence-checked.

## Release rule

No source is automatically student-ready. For every reused slide, retain attribution/provenance, recheck any time-sensitive assertion, and review the rendered deck and live exercise before release.

## L1.1 — The Modern AI Stack, End to End

### Learning aim
Give students one coherent map: data → tokenizer → transformer → pre-training/post-training → inference server → agent harness, with costs, failure modes, and evaluation boundaries at each layer.

### Authoritative / primary anchors

| Item | Role | Stable link / location | Audit note |
|---|---|---|---|
| Vaswani et al., *Attention Is All You Need* (2017) | Transformer architecture anchor | https://arxiv.org/abs/1706.03762 | Use for architecture, not as a complete description of present-day frontier systems. |
| Existing DLS 2026 introduction, slides 9–15 | Course-roadmap and architecture bridge | `02-introduction/DLS-2026-Introduction-DRAFT-v3.pptx` | Draft deck; slides 10, 12–15 have explicit provenance notes. Revalidate all current-model/vendor claims. |
| Existing DLS 2026 introduction, slides 16–25 | Harness, coding loop, backend/evaluation bridge | same deck | Slides 18, 20–22 reuse the pi-30 tutorial; command/version claims need a fresh pilot. |
| Existing source provenance map | Reuse authority and caveats | `00-sources/SLIDE_PROVENANCE.md` | This is the required citation trail for inherited slide objects. |

### Safe reuse candidates

- The course-roadmap framing and transformer-foundation material are candidates **only after** the Week-3 sequence is reconciled with the new survey map.
- The tensor-notation slide can become an appendix/bridge to L1.2 after shape and code examples are run.
- The coding-loop and independent-verifier concepts are useful for the stack/harness boundary, but the pi-30 commands and benchmarks must be tested on a clean student-like environment.

### Do not reuse without rebuilding or verification

- Any slide that implies a backend, model access, agent subscription, or local GPU requirement is available to students.
- Any performance, benchmark, software-version, or current-model statement.
- Any vendor claim about a model or system that is not labeled as such and independently corroborated.

## L1.2 — Deep Learning from First Principles, in Code

### Learning aim
Students implement and inspect a small MLP: forward pass, loss, backpropagation, optimization, gradient inspection, and a deliberate failure case. The goal is a working mental model, not a framework tour.

### Authoritative / primary anchors

| Item | Role | Stable link / location | Audit note |
|---|---|---|---|
| Rumelhart, Hinton & Williams, *Learning Representations by Back-Propagating Errors* (1986) | Historical / conceptual backprop anchor | https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf | Cite for the historical algorithmic idea; use a modern derivation for notation. |
| PyTorch autograd tutorial | Runnable framework reference | https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html | Official documentation says PyTorch rebuilds the dynamic computation graph after each backward pass; pin the exact version used in the student environment. |
| PyTorch stable autograd API | API reference | https://docs.pytorch.org/docs/stable/autograd.html | Link, do not copy API text into slides without version/date. |

### Existing course material status

- No dedicated MLP/backprop Week-3 lecture deck was identified in the DLS 2026 package.
- The existing tensor-notation material is a useful prerequisite bridge, but it is not a substitute for a checked MLP notebook.
- The Week-2 Lab 1 module is **release-gated** and targets agent environments; do not repurpose it as the L1.2 exercise without an explicit redesign and clean-environment pilot.

### Minimum evidence for the new activity

1. A CPU-only notebook/script runs from a fresh environment.
2. A numerical finite-difference gradient check agrees with autograd within a stated tolerance.
3. The activity makes one failure observable (e.g., learning rate too high, detached tensor, wrong shape, or unnormalized input).
4. No remote endpoint, paid service, or local GPU is a hidden prerequisite.

## L3.3 — Thinking With AI, Not Instead of It

### Learning aim
Treat AI use as a design and evidence problem: distinguish substitution from augmentation, require students to articulate/verify reasoning, and make the course’s own AI-use policy testable rather than moralizing.

### Evidence posture

The strongest defensible message is **conditional**: educational value depends substantially on task design, scaffolding, active interrogation, and verification. Do **not** claim that generative AI has been proven to cause lasting cognitive decline, or that any one pedagogical technique is proven to prevent it.

### Initial evidence anchors (must be read in full before slide claims)

| Item | Role | Link | Caveat |
|---|---|---|---|
| *Between cognitive offloading and critical autonomy* (systematic review, 2026) | Recent higher-education synthesis | https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2026.1878665/full | Search summary reports much evidence is cross-sectional/exploratory/self-reported; use this limitation prominently. |
| *Generative AI, Cognitive Offloading, and Learner Agency in Higher Education* (scoping review, 2026) | Learner-agency framing | https://pubmed.ncbi.nlm.nih.gov/42510271/ | Use as a map of the literature, not causal proof. |
| Course’s own assessment/evidence guidance | Local policy bridge | `03-course-repository/guides/backend-and-evidence.md`, `03-course-repository/review/REVIEW_PROTOCOL.md` | Audit policy language with Rick before student release. |

### Proposed Week-3 discussion/activity

- Give students a short technical explanation task twice: first create an independent outline/derivation, then use an AI system to critique or improve it.
- Require a claim ledger: what the student supplied, what the system supplied, what was independently checked, and what uncertainty remains.
- Grade the evidence trail and corrected understanding—not fluency of generated prose.

## Reuse inventory — verified from package metadata

| Asset | Status | Use decision |
|---|---|---|
| `02-introduction/DLS-2026-Introduction-DRAFT-v3.pptx/.pdf` | 46-slide draft; provenance exists | Mine selectively after source and sequence audit. |
| `00-sources/SLIDE_PROVENANCE.md` | Detailed per-slide provenance and caveats | Required reference during deck reuse. |
| `00-sources/READING_MAP.md` | 11 initial technical anchor readings | Reuse Transformer/scale/inference/GQA links; supplement Week-3 DL and learning evidence. |
| `03-course-repository/modules/M02-lab1-agent-environments/` | Draft; release-gated | Preserve; do not present as a ready Week-3 lab. |
| `04-preparation/PREPARATION_TASKS.md` | Course-wide pilot/review/release requirements | Apply T02/T24/T25 gates before distribution. |

## Week-3 build handoff

Before drafting slides, create a small source ledger per slide: **claim → source → source date/type → speaker-note citation → reviewer**. The decks should be built from this ledger, then rendered and independently reviewed with the activity run from a clean environment.
