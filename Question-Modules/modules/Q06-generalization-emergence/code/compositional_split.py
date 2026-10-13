"""Q06 held-out-combination demonstration. Run: python3 compositional_split.py"""
COLORS = range(4); SHAPES = range(4)
def target(color, shape): return (color + 2*shape) % 4
train = {(c,s): target(c,s) for c in COLORS for s in SHAPES if c != s}
test = [(i,i) for i in COLORS]
def lookup(c,s): return train.get((c,s))
def rule(c,s): return (c + 2*s) % 4
def check():
    lookup_score=sum(lookup(c,s)==target(c,s) for c,s in test)
    rule_score=sum(rule(c,s)==target(c,s) for c,s in test)
    assert lookup_score == 0 and rule_score == len(test)
    print(f'PASS: withheld combinations lookup={lookup_score}/{len(test)} rule={rule_score}/{len(test)}')
if __name__ == '__main__': check()
