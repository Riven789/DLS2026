# DLS 2026 Question Modules

Private/local course-development repository for the survey-derived DLS 2026 question modules.

## Canonical planning sources

- `register/QUESTION_MODULE_REGISTER.csv` — canonical master list of question modules.
- `docs/QUESTION_DRIVEN_MODULE_PROGRAM.md` — required artifact set, topical organization, ownership, and two-peer review protocol.
- `docs/LECTURE_DELIVERY_MAP.md` — the 25-session map; sessions are delivery containers, while question modules are the atomic reusable units.

The source survey remains in the DLS course package. Do not add identifiable student data to this repository.

## Current scope

The initial register contains 40 candidate modules: 12 build/systems, 21 model internals/research, and 7 society/practice. The first work is an audit of merges/splits and assignment of 13/13/14 authored modules across Ollie, Kukla, and Paradise. Every module then receives review from both non-authors.

## Module contract

Each module must include:

1. 3–5 page LaTeX narrative tutorial and compiled PDF
2. BibTeX bibliography
3. 5–10 slide Beamer source and compiled PDF
4. Exactly five discussion questions
5. Deeper-project suggestions
6. Minimal reproducible code when applicable
7. One formative understanding check with answer/rubric
8. Source ledger and QA/review record

See `docs/QUESTION_DRIVEN_MODULE_PROGRAM.md` for the full structure and release gates.

## Repository policy

- This is local/private until Rick explicitly approves a remote or publication.
- Keep citations and claims auditable; distinguish primary evidence, reviews, preprints, vendor claims, and instructor synthesis.
- No student-identifying data, secrets, access tokens, or copyrighted readings should be committed.
