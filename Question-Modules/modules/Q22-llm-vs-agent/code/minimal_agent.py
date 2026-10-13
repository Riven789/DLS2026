#!/usr/bin/env python3
"""Q22 demo: a minimal agent harness — the loop, not the model, supplies the agency.

A deliberately dumb rule-based "model" (a fixed policy, no learning) drives three
tools (search, read_file, finish) inside a harness loop to solve a tiny task:
find which file defines PORT and what its value is. The SAME policy is then run
single-call style — no loop, one action — and fails. The policy is byte-identical
in both conditions; only the control flow differs.

Trajectory highlights: a search false positive (notes.txt mentions PORT but does
not define it) forces error recovery, and the transcript grows monotonically
(non-stationary context / context rot).

Stdlib only. Run: python3 minimal_agent.py
"""

import re

# ---- the "environment": a tiny in-memory workspace ----
WORKSPACE = {
    "README.md": "# demo\nRun the server with `make serve`.\n",
    "notes.txt": "TODO: the default PORT should match production docs.\n",
    "server.cfg": "HOST=0.0.0.0\nPORT=8080\nWORKERS=4\n",
}
TASK = "Which file defines PORT, and what is its value? Answer as '<file> <value>'."


# ---- the "model": a fixed policy mapping transcript -> next action ----
# This stands in for an LLM. It is dumb on purpose: pure rules, and it only
# ever sees the task text and the transcript of past (action, observation) pairs.

def model(transcript):
    """Return the next action given the transcript so far."""
    if not transcript:
        return {"tool": "search", "args": {"pattern": "PORT"}}      # first move
    last_act, last_obs = transcript[-1]
    if last_act["tool"] == "search" and last_obs.startswith("matches:"):
        # read the first candidate we have not read yet (memory of past reads)
        candidates = last_obs[len("matches:"):].split(", ")
        already = {a["args"]["name"] for a, _ in transcript
                   if a["tool"] == "read_file"}
        for c in candidates:
            if c and c not in already:
                return {"tool": "read_file", "args": {"name": c}}
        return {"tool": "finish", "args": {"answer": None}}         # exhausted
    if last_act["tool"] == "read_file" and not last_obs.startswith("error"):
        m = re.search(r"^PORT=(\S+)$", last_obs, re.M)
        if m:  # this file actually defines PORT -> done
            return {"tool": "finish",
                    "args": {"answer": f"{last_act['args']['name']} {m.group(1)}"}}
        # false positive: mentions PORT but does not define it -> re-search
        return {"tool": "search", "args": {"pattern": "PORT"}}
    # error or unexpected observation: recover by searching again
    return {"tool": "search", "args": {"pattern": "PORT"}}


# ---- the tools ----

def search(pattern):
    hits = [name for name, body in WORKSPACE.items() if pattern in body]
    return "matches:" + ", ".join(hits) if hits else "matches:"


def read_file(name):
    if name not in WORKSPACE:
        return f"error: no such file: {name}"   # harness surfaces clean errors
    return WORKSPACE[name]


TOOLS = {"search": search, "read_file": read_file}


# ---- the harness: the loop that makes it an agent ----

def run_harness(task, max_steps=8):
    """model + loop + tools + memory (transcript) + stopping criterion."""
    transcript = []
    for step in range(1, max_steps + 1):
        action = model(transcript)                    # model decides locally...
        if action["tool"] == "finish":                # stopping criterion
            return action["args"]["answer"], transcript, step
        obs = TOOLS[action["tool"]](**action["args"])  # harness dispatches tool
        transcript.append((action, obs))              # memory grows every step
    return None, transcript, max_steps                # budget stop, no answer


def run_single_call(task):
    """Same model, NO loop: one decision, one tool call, run over."""
    action = model([])                                # exactly one decision
    if action["tool"] == "finish":
        return action["args"]["answer"], action
    TOOLS[action["tool"]](**action["args"])           # observation returned to nobody
    return None, action


if __name__ == "__main__":
    print(f"task: {TASK}\n")

    print("== condition A: the model, WITH the harness loop ==")
    answer, transcript, steps = run_harness(TASK)
    ctx = 0
    for i, (act, obs) in enumerate(transcript, 1):
        ctx += len(str(act)) + len(obs)
        print(f"  step {i}: {act['tool']}({act['args']}) -> {obs[:44]!r}"
              f"  [context ~{ctx} chars]")
    if answer is not None:
        print(f"  step {steps}: finish -> {answer!r}")
    else:
        print(f"  step {steps}: budget stop, no answer")
    print(f"  trajectory: {steps} steps, context grew every step\n")

    print("== condition B: the SAME model, single call (no loop) ==")
    answer_b, action_b = run_single_call(TASK)
    print(f"  one decision: {action_b['tool']}({action_b['args']})"
          " -> observation discarded; nothing decides again")
    print(f"  answer={answer_b!r}\n")

    # ---- pass/fail assertions ----
    a_pass = answer == "server.cfg 8080"
    b_fail = answer_b is None
    same_first = action_b == transcript[0][0]
    print(f"ASSERT looped run solves the task:        {'PASS' if a_pass else 'FAIL'}")
    print(f"ASSERT single-call run cannot solve it:   {'PASS' if b_fail else 'FAIL'}")
    print(f"ASSERT both runs used the identical policy and first action:"
          f" {'PASS' if same_first else 'FAIL'}")
    assert a_pass and b_fail and same_first
    print("\nSame 'model', different control flow => the loop supplies the agency.")
    raise SystemExit(0)
