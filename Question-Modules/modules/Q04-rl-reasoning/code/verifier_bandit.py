"""Q04: a dependency-free reward-driven policy update.
Run: python3 verifier_bandit.py
This illustrates an optimization signal; it is not RLHF or an LLM.
"""
import math

CANDIDATES = [3, 4, 5]
TARGET = 4

def softmax(logits):
    m = max(logits); exps = [math.exp(x-m) for x in logits]
    z = sum(exps); return [x/z for x in exps]

def check():
    logits = [0.0, 0.0, 0.0]
    before = softmax(logits)
    # Exact verifier reward for each possible completion; expected-reward gradient ascent.
    for _ in range(30):
        p = softmax(logits)
        rewards = [1.0 if y == TARGET else 0.0 for y in CANDIDATES]
        expected = sum(pi * ri for pi, ri in zip(p, rewards))
        for i, reward in enumerate(rewards):
            logits[i] += 0.25 * p[i] * (reward - expected)
    after = softmax(logits)
    assert after[1] > before[1] and after[1] > 0.8
    print('PASS: verifier-approved answer probability %.3f -> %.3f' % (before[1], after[1]))

if __name__ == '__main__': check()
