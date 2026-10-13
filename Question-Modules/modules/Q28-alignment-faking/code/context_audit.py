"""Q28 illustrative policy-audit table; not a deception detector. Run: python3 context_audit.py"""
CASES=[('valid-token','approve'),('no-token','escalate'),('decoy-token','escalate')]
def policy(case): return 'approve' if case=='valid-token' else 'escalate'
def check():
 got=[policy(c) for c,_ in CASES]
 assert got==[e for _,e in CASES]
 print('PASS: matched authorization/absence/decoy contexts obey declared policy')
if __name__=='__main__': check()
