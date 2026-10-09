import numpy as np
NAME='CNN'

def generate(task,n=1200,seed=42):
    rng=np.random.default_rng(seed); x=rng.normal(0,.12,(n,1,12,12)); labels=rng.integers(0,3,n); amps=rng.uniform(.6,1.5,n)
    for j in range(n):
        pos=rng.integers(2,9)
        if labels[j]==0: x[j,0,pos:pos+2,1:11]+=amps[j]
        elif labels[j]==1: x[j,0,1:11,pos:pos+2]+=amps[j]
        else:
            off=rng.integers(-2,3)
            for r in range(1,11):
                c=r+off
                if 0<=c<11: x[j,0,r,c:c+2]+=amps[j]
    y=labels if task=='classification' else amps
    return dict(X=x,y=y)

def validate(d,labeled=True):
    x=d['X']
    if x.ndim!=4 or min(x.shape)<1 or min(x.shape[-2:])<4 or any(v%2 for v in x.shape[-2:]): raise ValueError('X must be [N,C,H,W], H/W even and >=4')
    if not np.isfinite(x).all(): raise ValueError('CNN inputs must be finite; impute upstream')

def fit_pre(d,ids):
    x=d['X'][ids]; return {'mean':x.mean((0,2,3)).tolist(),'scale':np.maximum(x.std((0,2,3)),1e-8).tolist()}

def transform(d,state):
    x=d['X']; mean=np.array(state['mean'])[None,:,None,None]; scale=np.array(state['scale'])[None,:,None,None]
    if x.shape[1]!=mean.shape[1]: raise ValueError('Channel count differs from training')
    return {'X':(x-mean)/scale}

def config(d,outputs): return dict(channels=d['X'].shape[1],filters1=6,filters2=8,outputs=outputs)
def class_names(): return ['horizontal','vertical','diagonal']
