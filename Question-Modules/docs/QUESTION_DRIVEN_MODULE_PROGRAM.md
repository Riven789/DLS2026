# DLS 2026 — Question-Driven Module Program

**Decision recorded:** 2026-10-10  
**Authority:** Rick Stevens  
**Supersedes as the production unit:** a lecture-only plan. The 25 lecture/session entries remain the **delivery map**, but the durable production unit is now one module for each distinct student-articulated question.

## Goal

Extract and normalize the roughly 30–40 distinct questions in the Week-2 survey. Create a topical module for each question, then assemble modules into the 25 scheduled lectures, tutorials, labs, and discussions as appropriate.

A topic may contain several modules; a module must answer one coherent question rather than try to cover an entire broad topic.

## Mandatory module artifact set

Every module is authored and maintained in LaTeX, with generated PDFs checked before release.

```text
modules/
  <track>/<module-id>-<slug>/
    tutorial.tex                 # 3–5 page explanatory narrative
    tutorial.pdf
    bibliography.bib
    slides.tex                   # Beamer source, 5–10 slides
    slides.pdf
    discussion_questions.tex     # exactly five further-discussion questions
    discussion_questions.pdf
    project_ideas.tex            # deeper project suggestions
    project_ideas.pdf
    understanding_check.tex      # one small formative question + answer key
    understanding_check.pdf
    code/                        # only when applicable; README + test/run command
    sources.md                   # claim-to-source ledger, source type/date
    QA.md                        # build, factual, code, author, and peer-review record
```

### Required contents

1. **Narrative explanatory note/tutorial:** 3–5 pages; learning objective, prerequisites, explanation, worked example or counterexample, limits/uncertainties, and citations.
2. **Bibliography:** BibTeX source; primary research/official documentation distinguished from review, preprint, vendor, and instructor synthesis.
3. **Lecture notes:** Beamer deck of 5–10 slides, designed to teach rather than merely summarize the tutorial.
4. **Further discussion:** exactly five questions that go beyond recall and make uncertainty, tradeoffs, evidence, or implications discussable.
5. **Deeper projects:** scoped suggestions with expected evidence/deliverables and resource constraints.
6. **Code, when applicable:** minimal reproducible implementation, CPU-first unless the objective specifically requires accelerator hardware, plus a test or acceptance command.
7. **Understanding check:** one small question, expected answer/rubric, and a note on the misconception it diagnoses.

## Organization

```text
DLS-2026/
  03-course-repository/
    modules/
      T1-build-systems/
      T2-model-internals/
      T3-society-practice/
      _shared/                   # common LaTeX class, bibliography style, figures, QA tooling
  04-preparation/
    QUESTION_DRIVEN_MODULE_PROGRAM.md
    QUESTION_MODULE_REGISTER.csv
    MASTER_TALKS_PAPERS_PRESENTATIONS_REGISTER.md
```

- **T1 — Build & systems:** architecture, training, GPU/distributed systems, serving, evaluation, agents/harnesses.
- **T2 — Model internals & research:** capability formation, reasoning, scaling, memory/context, hallucination, interpretability, post-transformer alternatives, compiler/autodiff internals.
- **T3 — Society & practice:** work, learning/independent thought, research integrity, safety/security/control, political economy, physical footprint, futures.

## Granularity rules for the question register

1. Preserve student wording in a `verbatim_question` field whenever possible.
2. Normalize only duplicates that ask the same substantive question; record the merged response IDs.
3. Split a compound question if it requires different evidence or a different activity. Example: “How do agents use memory and tools?” becomes distinct modules for **tool orchestration** and **memory/context** if each needs separate explanation and code.
4. Do not invent a question merely because it is fashionable. Every initial module must trace to at least one student response or a clearly labeled prerequisite needed to answer one.
5. Distinguish empirical questions (“what does evidence show?”), mechanistic questions (“how does it work?”), engineering questions (“how do we build/evaluate it?”), and normative questions (“what should we do?”).

## Register schema

The canonical `QUESTION_MODULE_REGISTER.csv` shall contain:

```text
module_id,track,topic,verbatim_question,normalized_question,response_ids,
question_type,scheduled_session,priority,prerequisites,anchor_sources,
code_applicable,author,peer_reviewer_1,peer_reviewer_2,
tutorial_status,beamer_status,code_status,qa_status,release_status,notes
```

## Three-way production and review protocol

There are three producing peers: **Ollie, Kukla, and Paradise**. Each owns approximately one-third of the modules, balanced by **estimated work**, not merely module count. The other two are mandatory reviewers of every completed module.

### Roles for each module

| Role | Responsibility |
|---|---|
| Author | Drafts every required artifact; records all sources; runs code/build checks; addresses reviews. |
| Peer reviewer 1 | Technical/evidence review: correctness, citations, derivations, claims, code/test behavior. |
| Peer reviewer 2 | Teaching/release review: clarity, scope, student activity, Beamer rendering, accessibility, and consistency with course policy. |
| Rick | Final instructional/editorial approval where needed; resolves contested scope or policy. |

### Review gates

1. **Author self-check:** LaTeX compiles without unresolved references; PDFs visually inspected; code exercised from a clean environment when applicable.
2. **Two peer reviews:** both reviewers leave actionable approval or requested changes in `QA.md`.
3. **Author revision:** author closes all requested changes and records what changed.
4. **Independent recheck:** at least one reviewer reruns the build/code check after revision.
5. **Release:** only then mark `release_status=ready_for_Rick`; no module is distributed without Rick’s approval.

No reviewer may approve their own authored module. Reviews are substantive, not a check-box: a reviewer must inspect the tutorial PDF and slides PDF; the technical reviewer must inspect citations and any code.

## Allocation procedure

1. First extract the full candidate-question list from the survey into `QUESTION_MODULE_REGISTER.csv`.
2. Deduplicate transparently and obtain a count in the target 30–40 range.
3. Assign modules round-robin across the three tracks while balancing difficulty and near-term Week-3 needs.
4. Set the other two peers as the reviewers for each row.
5. Freeze the first Week-3 batch only after the register and assignments are visible to all three peers.

**Initial Week-3 modules are urgent, but this does not waive review.** Their scope should be deliberately small enough to build, review, and revise before use.

## Relationship to the 25-session map

- The 25 L1/L2/L3 entries describe **when and how modules are taught**.
- The question modules are the **atomic reusable knowledge artifacts**.
- A scheduled lecture may draw on 2–5 modules; one module may support more than one scheduled session.
- The lecture register must link each scheduled session to its contributing module IDs once the question register is complete.

## Immediate next work

1. Extract all question candidates from the survey with original response IDs and create the canonical CSV register.
2. Propose transparent merges/splits and verify the unique-module count.
3. Establish a shared LaTeX/Beamer template and reproducible PDF build command in `_shared/`.
4. Allocate one-third of the validated module list to each producer and assign the two cross-reviews.
5. Start only the reviewed Week-3 module subset first; build the remaining modules in topical batches.
