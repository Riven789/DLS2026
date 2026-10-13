# Lab 1 Rubric — 100 points

| Criterion | Points | Full-credit evidence | Partial credit | No credit |
|---|---:|---|---|---|
| Pi installation and live configuration | 10 | Version recorded; redacted config; live final-content smoke test; tool/web-search capability demonstrated | Installed but configuration or evidence incomplete | No usable Pi evidence |
| Codex installation and live configuration | 10 | Version recorded; user-level custom provider/profile; live final-content smoke test; `--search` and approval policy shown | Installed but adapter/configuration incomplete | No usable Codex evidence |
| Security and authority policy | 6 | Clear bounded permissions; secrets redacted; no unrestricted mode; agent actions reviewed | Sensible policy but missing evidence or one avoidable exposure | Key exposed, unsafe permissions, or unreviewed destructive action |
| Three paired tasks: design and execution | 12 | Same task statements; separate workspaces; both agents used; prompts/transcripts/diffs preserved | Minor asymmetry or missing artifact | Tasks not paired or claimed without evidence |
| Three paired tasks: independent verification and analysis | 12 | Independent tests/builds; honest outcomes; concrete comparison of behavior and web/tool use | Some checks but weak comparison | Agent self-report substituted for verification |
| pi-30 experimental control | 10 | Exact commits; seed guard passed first; equal budgets/settings; clean per-task workspaces; no leakage | Small reproducibility/control gap | Contaminated seeds, answer leakage, or incomparable conditions |
| pi-30 measurements and raw evidence | 10 | Complete `runs.csv`; logs/transcripts; wall time, tokens where available, retry/tool data, and explicit `unknown` fields | Some telemetry absent but transparently documented | Aggregate claims without raw evidence |
| pi-30 scoring and statistical analysis | 10 | Verifier-based status; completion/score results; intervals; time/token summaries; failures classified | Correct basic summaries but missing interval/failure accounting | Scores based on prose or altered tests |
| LaTeX report and PDF | 10 | 4–6 pages; compiles cleanly; clear methods/results/limitations; figures/tables readable; bibliography present | Minor build/organization defect | No source or no rendered PDF |
| Reproducibility, disclosure, and judgment | 10 | Replay README, manifest, redacted settings, agent disclosure, limitations, and no unsupported conclusions | Good package with gaps | Incomplete or misleading provenance |

## Scoring principles

- **Evidence beats narrative.** A passing verifier, independent build, exact log, or raw record outranks any agent’s claim.
- **Negative results can earn full credit.** A documented endpoint outage, failed adapter, or poor benchmark result is valuable if the student preserved evidence and did not conceal it.
- **No credit for contamination.** Copying a solution, modifying a verifier, using a hidden answer key as context, or reusing another student’s outputs invalidates the affected portion.
- **Token accounting is descriptive, not a penalty.** Students are graded on accurate measurement and interpretation, not on spending the fewest tokens.
- **Accessibility/fallback.** A student blocked by endpoint availability must submit the local setup and failure evidence promptly; staff will provide an equivalent controlled make-up run.
