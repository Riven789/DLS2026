import numpy as np
NAME='GNN'

def generate(task,n=1400,seed=42):
    rng=np.random.default_rng(seed); maxn=16
    x=np.zeros((n,maxn,2)); a=np.zeros((n,maxn,maxn)); mask=np.zeros((n,maxn)); y=[]
    for b in range(n):
        count=int(rng.integers(8,maxn+1)); flags=np.arange(count)%2; rng.shuffle(flags)
        x[b,:count,0]=flags; x[b,:count,1]=rng.normal(0,1,count); mask[b,:count]=1
        same_prob=rng.choice([.15,.85]) if task=='classification' else rng.uniform(.1,.9)
        for i in range(count):
            for j in range(i):
                p=same_prob if flags[i]==flags[j] else 1-same_prob
                if rng.random()<p: a[b,i,j]=a[b,j,i]=1
        # Extremely unlikely empty graph gets one explicit edge.
        if a[b].sum()==0: a[b,0,1]=a[b,1,0]=1
        ii,jj=np.nonzero(np.triu(a[b],1)); fraction=float(np.mean(flags[ii]==flags[jj]))
        y.append(int(fraction>.5) if task=='classification' else fraction)
    return dict(X=x,A=a,mask=mask,y=np.array(y))

def validate(d,labeled=True):
    x,a,m=d['X'],d['A'],d['mask']
    if x.ndim!=3 or m.shape!=x.shape[:2] or a.shape!=(len(x),x.shape[1],x.shape[1]) or x.shape[-1]!=2: raise ValueError('Expected X [B,N,2], A [B,N,N], mask [B,N]')
    if not all(np.isfinite(v).all() for v in [x,a,m]): raise ValueError('All inputs must be finite')
    if not np.isin(m,[0,1]).all() or np.any(m.sum(1)==0): raise ValueError('Every graph needs at least one valid node')
    if np.any(a<0) or not np.allclose(a,a.transpose(0,2,1)) or np.any(np.diagonal(a,axis1=1,axis2=2)!=0): raise ValueError('Expected undirected nonnegative adjacency with zero diagonal')
    if np.any(a*(1-m[:,:,None]*m[:,None,:])): raise ValueError('No edges may touch padding')
    if not np.isin(x[:,:,0][m.astype(bool)],[0,1]).all(): raise ValueError('First node feature is a Boolean 0/1')

def fit_pre(d,ids):
    valid=d['mask'][ids].astype(bool); v=d['X'][ids,:,1][valid]
    return dict(mean=float(v.mean()),scale=max(float(v.std()),1e-8))

def transform(d,state):
    x=np.asarray(d['X'],dtype=float).copy(); x[:,:,1]=(x[:,:,1]-state['mean'])/state['scale']; x*=d['mask'][...,None]
    return dict(X=x,A=d['A'].astype(float),mask=d['mask'].astype(float))

def config(d,outputs): return dict(features=d['X'].shape[-1],hidden=16,outputs=outputs)
def class_names(): return ['mostly_different','mostly_same']
