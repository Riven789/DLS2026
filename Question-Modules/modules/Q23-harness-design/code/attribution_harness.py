#!/usr/bin/env python3
"""Q23 demo: a tiny stdlib-only agent harness with an injected failure taxonomy.

The question this module asks: when an agent run fails, which layer caused it
-- model, instructions, or harness? This demo makes attribution mechanical.

Design:
  * A scripted "model" (no network, no dependencies) solves a small task
    suite by emitting tool calls. The policy is the 'model' layer: a good
    policy reads the zip from the injected lookup result; a degraded policy
    ignores feedback and hallucinates a plausible zip (90210).
  * A harness around it: tool schemas, dispatch, result injection, a stopping
    rule. The harness is swappable: a good harness injects the true tool
    result; a broken harness injects the WRONG call's result, truncated
    (the classic wrong-result-injection bug); a timeout harness kills a
    slow-but-correct call (stopping-rule failure).
  * Environments hold per-zip ground truth, so a hallucinated zip returns
    genuinely different (real) data -- the harness did nothing wrong, the
    model just asked about the wrong place.
  * An attribution judge that uses the module's instruments:
      1. logged tool I/O  -- diff injected vs true result exposes the harness;
      2. control runs     -- swap one layer at a time (clarified spec = the
                             instruction swap; good model = the model swap;
                             good harness = the harness swap) and confirm.

Configs (the ablation grid; A/B/C are the module's three core cells):
  A: good model    + good harness    -> expect all pass
  B: good model    + broken harness  -> expect harness-attributed failures
  C: degraded model+ good harness    -> expect model-attributed failures
  D: good model    + good harness, ambiguous spec -> instruction failure
  E: good model    + timeout harness -> harness failure (timeout kills run)

Run: python3 attribution_harness.py   (exit code 0 iff all assertions pass)
"""

# ----------------------------------------------------------------------------
# Task suite. Each task: a goal, the spec given to the agent (the
# 'instructions' layer), a clarified spec (the control for the instruction
# swap), an environment with per-zip ground truth, and a checker.
# ----------------------------------------------------------------------------

TOOLS = {
    "lookup_user": {
        "desc": "lookup_user(email) -> {'name': str, 'zip': str}",
        "params": ["email"],
        "behavior": lambda args, env: {"name": env["name"], "zip": env["zip"]},
    },
    "weather": {
        "desc": "weather(zip) -> {'forecast': str, 'temp_c': int}",
        "params": ["zip"],
        "behavior": lambda args, env: {"forecast": env["zip_data"][args["zip"]]["forecast"],
                                       "temp_c": env["zip_data"][args["zip"]]["temp_c"]},
    },
    "timezone": {
        "desc": "timezone(zip) -> {'tz': str}",
        "params": ["zip"],
        "behavior": lambda args, env: {"tz": env["zip_data"][args["zip"]]["tz"]},
    },
}

# Per-zip ground truth. Note 90210: real data for the WRONG place -- a
# hallucinated zip gets a genuine (but incorrect-for-the-task) answer back.
ZIP_DATA = {
    "02139": {"forecast": "rain", "temp_c": 12, "tz": "America/New_York"},
    "10007": {"forecast": "clear", "temp_c": 21, "tz": "America/New_York"},
    "20546": {"forecast": "rain", "temp_c": 9, "tz": "America/New_York"},
    "94720": {"forecast": "fog", "temp_c": 14, "tz": "America/Los_Angeles"},
    "90210": {"forecast": "clear", "temp_c": 30, "tz": "America/Los_Angeles"},  # hallucination target
}

TASKS = [
    {
        "id": "T1-report",
        "goal": "Tell the user whether they need an umbrella.",
        "spec": "Find the user's zip with lookup_user, then get the forecast with "
                "weather(zip), then answer 'umbrella' if the forecast is rain.",
        "clarified_spec": "Find the user's zip with lookup_user, then get the forecast with "
                          "weather(zip), then answer 'umbrella' if the forecast is rain.",
        "env": {"name": "A. Turing", "zip": "02139", "zip_data": ZIP_DATA},
        "check": lambda ans: ans == "umbrella",
    },
    {
        "id": "T2-celsius",
        "goal": "Report the user's local temperature in Celsius.",
        "spec": "Find the user's zip with lookup_user, then get the temperature with "
                "weather(zip), then answer with the integer temp_c.",
        "clarified_spec": "Find the user's zip with lookup_user, then get the temperature with "
                          "weather(zip), then answer with the integer temp_c.",
        "env": {"name": "G. Hopper", "zip": "10007", "zip_data": ZIP_DATA},
        "check": lambda ans: ans == "21",
    },
    {
        "id": "T3-multi-step",
        "goal": "State the user's timezone and whether it is raining there, as 'tz:rain|no-rain'.",
        "spec": "Find the user's zip with lookup_user; get the forecast with weather(zip) and "
                "the timezone with timezone(zip); answer 'America/New_York:rain' style.",
        "clarified_spec": "Find the user's zip with lookup_user; get the forecast with weather(zip) and "
                          "the timezone with timezone(zip); answer 'America/New_York:rain' style.",
        "env": {"name": "K. Johnson", "zip": "20546", "zip_data": ZIP_DATA},
        "check": lambda ans: ans == "America/New_York:rain",
    },
    {
        "id": "T4-ambiguous-spec",
        "goal": "Tell the user the temperature where they live.",
        # AMBIGUOUS spec: no tool guidance, no field named. The model layer is
        # fine; the spec omits what the checker requires (the integer temp_c).
        "spec": "Tell the user the temperature where they live.",
        "clarified_spec": "Find the user's zip with lookup_user, then get the temperature with "
                          "weather(zip), then answer with the integer temp_c.",
        "env": {"name": "J. Liskov", "zip": "94720", "zip_data": ZIP_DATA},
        "check": lambda ans: ans == "14",
    },
]

# ----------------------------------------------------------------------------
# The 'model' layer: scripted policies over the visible context.
# ----------------------------------------------------------------------------

def _synthesize(task, visible_log):
    """Turn injected results into a final answer, DRIVEN BY THE SPEC: a good
    model answers with what its instructions asked for. Defensive by design:
    if the injected result is missing the expected field (corrupted
    upstream), the model does its best with what actually reached it and
    answers wrong -- which is exactly the evidence the judge needs to blame
    the harness. If the spec named no field (ambiguous), the model guesses
    the most salient quantity -- the model is fine, the spec is not."""
    got = {e["call"]["tool"]: e["injected"] for e in visible_log if e.get("ok")}

    def fld(tool, key, default):
        r = got.get(tool)
        return r.get(key, default) if isinstance(r, dict) else default

    spec = task["spec"]
    if "integer temp_c" in spec:
        return str(fld("weather", "temp_c", "??"))
    if "umbrella" in spec:
        return "umbrella" if fld("weather", "forecast", "clear") == "rain" else "no umbrella"
    if "tz:rain" in task["goal"] or ":rain' style" in spec:
        rain = "rain" if fld("weather", "forecast", "clear") == "rain" else "no-rain"
        return f"{fld('timezone', 'tz', 'unknown')}:{rain}"
    # Ambiguous spec: no field named -> a plausible-but-wrong guess.
    if "weather" in got:
        return fld("weather", "forecast", "unknown")
    return "unknown"


def good_model_policy(task, visible_log):
    """A competent model: follows the spec's tool order, reads the zip out of
    the injected lookup result, retries on error when there is nothing else
    to act on (opaque surface)."""
    if visible_log and visible_log[-1].get("ok") is False:
        # Last call errored. Our scripted good model never misnames args, so
        # the only errors it meets are timeouts -> same-call retry.
        last = visible_log[-1]
        return {"tool": last["call"]["tool"], "args": dict(last["call"]["args"])}
    spec_tools = [t for t in ("lookup_user", "weather", "timezone") if t in task["spec"]]
    if not spec_tools:
        # Ambiguous spec names no tools; a competent model still improvises
        # the obvious ones (the failure will be WHICH FIELD it answers with).
        spec_tools = ["lookup_user", "weather"]
    done = {e["call"]["tool"] for e in visible_log if e.get("ok")}
    for t in spec_tools:
        if t not in done:
            if t == "lookup_user":
                return {"tool": "lookup_user", "args": {"email": "user@example.com"}}
            zip_result = next((e["injected"] for e in visible_log
                               if e["call"]["tool"] == "lookup_user" and e.get("ok")), None)
            if zip_result is None:
                return None
            return {"tool": t, "args": {"zip": zip_result["zip"]}}  # uses injected result
    return _synthesize(task, visible_log)


def degraded_model_policy(task, visible_log):
    """A degraded model: same spec-following skeleton, but it IGNORES the
    injected lookup result and hallucinates a plausible zip (90210). The
    harness will faithfully return real data for the wrong place."""
    if visible_log and visible_log[-1].get("ok") is False:
        last = visible_log[-1]
        return {"tool": last["call"]["tool"], "args": dict(last["call"]["args"])}  # blind retry
    spec_tools = [t for t in ("lookup_user", "weather", "timezone") if t in task["spec"]]
    done = {e["call"]["tool"] for e in visible_log if e.get("ok")}
    for t in spec_tools:
        if t not in done:
            if t == "lookup_user":
                return {"tool": "lookup_user", "args": {"email": "user@example.com"}}
            # HALLUCINATION: ignores the injected zip, uses a plausible fake
            return {"tool": t, "args": {"zip": "90210"}}
    return _synthesize(task, visible_log)


# ----------------------------------------------------------------------------
# The harness layer: schema validation, dispatch, injection, stopping.
# ----------------------------------------------------------------------------

class GoodHarness:
    """Validates args against per-tool schemas, injects the TRUE tool result,
    formats errors correctively, stops at a turn budget."""
    name = "good-harness"

    def dispatch(self, call, task):
        schema = TOOLS[call["tool"]]
        args = call["args"]
        missing = [p for p in schema["params"] if p not in args]
        if missing:
            return {"ok": False, "true": None,
                    "injected": f"error: {call['tool']} requires {schema['params']}, "
                                f"got {sorted(args)}; did you mean to check the schema?"}
        try:
            true = schema["behavior"](args, task["env"])
            return {"ok": True, "true": true, "injected": true}
        except Exception as e:  # defensive: a real harness formats crashes
            return {"ok": False, "true": None,
                    "injected": f"error: {call['tool']} crashed ({type(e).__name__})"}


class BrokenHarness(GoodHarness):
    """Same schemas and stopping, but dispatch corrupts results: for every
    call after lookup_user it injects the result of the WRONG call (the
    wrong-tool-result-injected failure), truncated to meaninglessness."""
    name = "broken-harness (wrong-result injection)"

    def dispatch(self, call, task):
        result = super().dispatch(call, task)
        if not result["ok"] or call["tool"] == "lookup_user":
            return result
        # INJECTION BUG: swap in a different call's result, truncated.
        wrong = TOOLS["timezone"]["behavior"](call["args"], task["env"])
        result["injected"] = {k: str(v)[:3] for k, v in wrong.items()}
        return result


class TimeoutHarness(GoodHarness):
    """A stopping-rule failure: weather is slow but correct; this harness
    kills it at a timeout, every time, with an error the model cannot act
    on. A correct run dies mid-flight."""
    name = "timeout-harness (stopping rule kills correct run)"

    def dispatch(self, call, task):
        if call["tool"] == "weather":
            return {"ok": False, "true": None,
                    "injected": "error: weather timed out after 2s "
                                "(slow-but-correct call killed by stopping rule)"}
        return super().dispatch(call, task)


# ----------------------------------------------------------------------------
# Run loop + attribution judge.
# ----------------------------------------------------------------------------

def run_agent(model_policy, harness, task, max_turns=6):
    """Run one task; return (final_answer, full_log). The log records BOTH the
    true tool result and what the harness injected -- the diff that makes
    harness failures visible (the module's 'log tool I/O' point)."""
    log = []
    for _ in range(max_turns):
        out = model_policy(task, log)
        if out is None:
            return None, log
        if isinstance(out, str):  # final answer
            return out, log
        res = harness.dispatch(dict(tool=out["tool"], args=out["args"]), task)
        log.append({"call": {"tool": out["tool"], "args": out["args"]},
                    "ok": res["ok"], "true": res["true"], "injected": res["injected"]})
    return None, log  # stopped: turn budget exhausted


def attribute(task, answer, log, model_policy, harness):
    """Decide which layer caused the failure, with the evidence that decided
    it. Instruments, in the module's order: (1) logged tool I/O diffs point
    at the harness directly; (2) control runs swap one layer at a time --
    clarified spec = instruction swap, good model = model swap, good harness
    = harness swap. Returns (layer, evidence)."""
    if answer is not None and task["check"](answer):
        return "PASS", "checker satisfied"

    # 1) Logged evidence: the harness injected something other than the true
    #    result. The model may have reasoned perfectly from corrupted input.
    for e in log:
        if e["ok"] and e["true"] is not None and e["injected"] != e["true"]:
            ctrl, _ = run_agent(model_policy, GoodHarness(), task)
            ev = "logged diff: injected != true"
            if ctrl is not None and task["check"](ctrl):
                ev += "; harness-swap control passes"
            return "harness", ev

    # 2) Logged evidence: a stopping rule killed calls that were slow but
    #    correct (timeout), and the model had nothing actionable to change.
    for e in log:
        if not e["ok"] and "timed out" in str(e["injected"]):
            ctrl, _ = run_agent(model_policy, GoodHarness(), task)
            ev = "timeout killed a correct run"
            if ctrl is not None and task["check"](ctrl):
                ev += "; harness-swap control passes"
            return "harness", ev

    # 3) Instruction swap: same model, same harness, clarified spec.
    clarified = dict(task, spec=task["clarified_spec"])
    ans2, _ = run_agent(model_policy, harness, clarified)
    if ans2 is not None and clarified["check"](ans2):
        return "instructions", "clarified-spec control passes (model + harness unchanged)"

    # 4) Model swap: same instructions, same harness, good model.
    ans3, _ = run_agent(good_model_policy, harness, task)
    if ans3 is not None and task["check"](ans3):
        return "model", "model-swap control passes (instructions + harness unchanged)"

    return "joint/unattributed", "no single-layer swap clears it"


def main():
    configs = [
        ("A: good model + good harness    ", good_model_policy, GoodHarness(), TASKS[:3]),
        ("B: good model + broken harness  ", good_model_policy, BrokenHarness(), TASKS[:3]),
        ("C: degraded model + good harness", degraded_model_policy, GoodHarness(), TASKS[:3]),
        ("D: good model, ambiguous spec   ", good_model_policy, GoodHarness(), [TASKS[3]]),
        ("E: good model + timeout harness ", good_model_policy, TimeoutHarness(), [TASKS[0]]),
    ]
    rows, failures = [], 0
    print("Q23 attribution harness -- ablation grid over model / instructions / harness")
    print("=" * 84)
    for label, policy, harness, tasks in configs:
        for task in tasks:
            answer, log = run_agent(policy, harness, task)
            layer, ev = attribute(task, answer, log, policy, harness)
            rows.append((label, task["id"], layer))
            failures += layer != "PASS"
            print(f"{label} | {task['id']:<20} | {layer:<12} | {ev}")
    print("=" * 84)

    print("\nFailure attribution table (layer = judge verdict):")
    print(f"{'config':<34} {'pass':>4} {'model':>6} {'instr':>6} {'harness':>8} {'joint':>6}")
    for label, _, _, tasks in configs:
        cfg = [r[2] for r in rows if r[0] == label]
        print(f"{label:<34} {cfg.count('PASS'):>4} {cfg.count('model'):>6} "
              f"{cfg.count('instructions'):>6} {cfg.count('harness'):>6} "
              f"{cfg.count('joint/unattributed'):>6}")

    # Assertions: the demo's claims, checked mechanically.
    def v(label, tid):
        return next(r[2] for r in rows if r[0] == label and r[1] == tid)

    assert v("A: good model + good harness    ", "T1-report") == "PASS"
    assert v("A: good model + good harness    ", "T2-celsius") == "PASS"
    assert v("A: good model + good harness    ", "T3-multi-step") == "PASS"
    assert all(v("B: good model + broken harness  ", t) == "harness" for t in
               ("T1-report", "T2-celsius", "T3-multi-step"))
    assert all(v("C: degraded model + good harness", t) == "model" for t in
               ("T1-report", "T2-celsius", "T3-multi-step"))
    assert v("D: good model, ambiguous spec   ", "T4-ambiguous-spec") == "instructions"
    assert v("E: good model + timeout harness ", "T1-report") == "harness"

    print(f"\nAll assertions passed: {len(rows)} runs, {failures} non-PASS, "
          f"each non-PASS attributed to exactly one layer (no joint cells).")
    print("A vs B isolates the harness layer (same model + instructions).")
    print("A vs C isolates the model layer (same instructions + harness).")
    print("A vs D isolates the instruction layer (same model + harness).")
    print("E shows a stopping-rule harness failure a transcript alone would")
    print("likely misread as 'the model kept failing'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
