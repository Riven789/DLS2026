"""Tiny, dependency-free context-copy demonstration for Q01.

This is a teaching heuristic, not a Transformer implementation.
Run: python3 toy_induction.py
"""
from collections import Counter


def unigram_predict(history):
    """Predict the most frequent observed token; ties resolve alphabetically."""
    return sorted(Counter(history).items(), key=lambda pair: (-pair[1], pair[0]))[0][0]


def context_copy_predict(history, width=2):
    """Copy the continuation after the latest earlier matching width-token context.

    Returns None when no prior context/continuation is available.
    """
    if len(history) <= width:
        return None
    context = history[-width:]
    # Search left-to-right so the last match retained is the most recent earlier one.
    candidate = None
    for start in range(0, len(history) - width):
        if history[start : start + width] == context:
            candidate = history[start + width]
    return candidate


def check():
    history = list("ABABA")
    assert unigram_predict(history) == "A"
    assert context_copy_predict(history, 2) == "B"  # prior AB was followed by B
    assert context_copy_predict(list("XYZ"), 2) is None
    print("PASS: frequency baseline predicts A; context-copy predicts B for ABABA -> ?")


if __name__ == "__main__":
    check()
