# 0 — Primitives (the five-minute cold open)

The `pi` invocations reused throughout the ten problems:

| Form | What it is |
|---|---|
| `pi` | interactive REPL |
| `pi "..."` | one-shot, prints, exits |
| `pi -p "..."` | print / headless mode — the workhorse for scripted loops |
| `pi --json "..."` | structured output you can parse (use sparingly) |
| `pi -c` / `pi -r` | continue last / browse past sessions |
| `pi --fork <id>` | branch a session — the substrate for tournaments |
| `~/.pi/agent/skills/<name>/SKILL.md` | a self-installed, reusable skill |
| `~/.pi/agent/memory.md` | persistent lessons across runs |

The loop anatomy (from `pi-loop-designer` — every problem from #3 on is just
these seven slots filled differently):

**Objective · State · Iteration prompt · Verification · Reflection · Stop
rules · Next prompt**

> The single most important point of the whole tutorial: stop prompting one
> task at a time. Design the loop; let the loop prompt the agent.
> Verification is what makes a loop a loop — without a machine-checkable
> signal you just have a chatbot.

`demo.sh` exercises each invocation form against the bundled mock (the mock
answers with canned session bookkeeping; a real `pi` would answer with the
model).
