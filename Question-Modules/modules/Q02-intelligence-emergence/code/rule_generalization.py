"""Q02: tiny held-out rule-generalization demonstration (no dependencies).
Run: python3 rule_generalization.py
The task is deliberately narrow; it does not test 'intelligence'.
"""
def target(a, b):
    return (a + 2 * b) % 5

# Training examples omit every pair where a == b.
train = {(a, b): target(a, b) for a in range(5) for b in range(5) if a != b}
test = [(a, a) for a in range(5)]

def lookup(a, b):
    return train.get((a, b), None)

def rule_solver(a, b):
    return (a + 2 * b) % 5

def check():
    lookup_correct = sum(lookup(a, b) == target(a, b) for a, b in test)
    rule_correct = sum(rule_solver(a, b) == target(a, b) for a, b in test)
    assert lookup_correct == 0
    assert rule_correct == len(test)
    print(f'PASS: held-out diagonal pairs: lookup {lookup_correct}/{len(test)}, rule {rule_correct}/{len(test)}')

if __name__ == '__main__':
    check()
