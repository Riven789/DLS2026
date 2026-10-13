"""Q24 multi-objective agent scorecard. Run: python3 scorecard.py"""
AGENTS={"fast-safe":{"success":70,"cost":70,"violations":0},"costly-risky":{"success":70,"cost":700,"violations":5}}
# NOTE: the weights below (0.02*cost, 5*violations) are ILLUSTRATIVE only.
# In practice constraint violations are usually hard gates (any violation -> fail),
# not linearly priced trade-offs; see note section 3.
def utility(r): return r['success'] - 0.02*r['cost'] - 5*r['violations']
def check():
    assert AGENTS['fast-safe']['success']==AGENTS['costly-risky']['success']
    a,b=utility(AGENTS['fast-safe']),utility(AGENTS['costly-risky'])
    assert a>b
    print(f'PASS: equal completion=70; utility fast-safe={a:.1f} > costly-risky={b:.1f}')
if __name__=='__main__': check()
