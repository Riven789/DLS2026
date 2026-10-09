import numpy as np
NAME='LSTM'

def generate(task,n=1600,seed=42):
    rng=np.random.default_rng(seed); tmax=12; x=np.zeros((n,tmax,2)); mask=np.zeros((n,tmax)); y=[]
    for b in range(n):
        length=int(rng.integers(6,tmax+1)); values=rng.uniform(-1,1,length); marks=np.sort(rng.choice(length,2,replace=False))
        x[b,:length,0]=values; x[b,marks,1]=1; mask[b,:length]=1
        value=values[marks[-1]]; y.append(int(value>0) if task=='classification' else value)
    return dict(X=x,mask=mask,y=np.array(y))

def validate(d,labeled=True):
    x,m=d['X'],d['mask']
    if x.ndim!=3 or x.shape[-1]!=2 or m.shape!=x.shape[:2]: raise ValueError('Expected X [B,T,2] and mask [B,T]')
    if not np.isfinite(x).all() or not np.isin(m,[0,1]).all() or np.any(m.sum(1)==0): raise ValueError('Finite inputs and nonempty 0/1 masks required')
    if np.any(np.diff(m,axis=1)>0): raise ValueError('Use right padding: all valid steps before padding')
    if not np.isin(x[:,:,1][m.astype(bool)],[0,1]).all(): raise ValueError('Marker feature must be Boolean 0/1')

def fit_pre(d,ids):
    v=d['X'][ids,:,0][d['mask'][ids].astype(bool)]
    return dict(mean=float(v.mean()),scale=max(float(v.std()),1e-8))

def transform(d,state):
    x=np.asarray(d['X'],dtype=float).copy(); x[:,:,0]=(x[:,:,0]-state['mean'])/state['scale']; x*=d['mask'][...,None]
    return dict(X=x,mask=d['mask'].astype(float))

def config(d,outputs): return dict(features=d['X'].shape[-1],hidden=16,outputs=outputs)
def class_names(): return ['nonpositive','positive']
