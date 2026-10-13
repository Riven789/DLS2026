"""Q29 bound-approval policy toy. Run: python3 approval_bound.py"""
APPROVED={('merge','repo-A','a1b2c3','reviewer-7')}
def allowed(action,target,digest,approver): return (action,target,digest,approver) in APPROVED
def check():
 assert allowed('merge','repo-A','a1b2c3','reviewer-7')
 assert not allowed('merge','repo-A','DIFFERENT','reviewer-7')
 assert not allowed('send-email','repo-A','a1b2c3','reviewer-7')
 print('PASS: only exact action/target/digest/approver approval is accepted')
if __name__=='__main__': check()
