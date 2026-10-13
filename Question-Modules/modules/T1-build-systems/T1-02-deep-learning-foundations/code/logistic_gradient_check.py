"""CPU-only analytic vs finite-difference logistic-loss gradient check."""
import numpy as np
x=np.array([2.,-1.]); y=1.; w=np.array([.1,.2]); b=0.
def loss(w,b):
 p=1/(1+np.exp(-(w@x+b))); return -(y*np.log(p)+(1-y)*np.log(1-p))
p=1/(1+np.exp(-(w@x+b))); analytic=np.r_[(p-y)*x,p-y]
eps=1e-6; numeric=[]
for i in range(3):
 d=np.zeros(3); d[i]=eps
 numeric.append((loss(w+d[:2],b+d[2])-loss(w-d[:2],b-d[2]))/(2*eps))
np.testing.assert_allclose(analytic,numeric,rtol=1e-5,atol=1e-6)
print('PASS analytic=',analytic,'numeric=',numeric)
