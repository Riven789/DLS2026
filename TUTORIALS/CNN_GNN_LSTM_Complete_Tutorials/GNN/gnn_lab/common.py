"""Shared training utilities. Each lab includes its own copy for independent use."""
import json
import hashlib
from pathlib import Path
import numpy as np


def sigmoid(x): return np.exp(-np.logaddexp(0., -x))
def softmax(x):
    e=np.exp(x-x.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)

def loss_grad(z,y,task):
    if task=='regression':
        d=z-y.reshape(-1,1); return float(np.mean(d*d)),2*d/d.size
    a=z-z.max(1,keepdims=True)
    lp=a-np.log(np.exp(a).sum(1,keepdims=True))
    g=np.exp(lp); g[np.arange(len(y)),y.astype(int)]-=1
    return float(-lp[np.arange(len(y)),y.astype(int)].mean()),g/len(y)

class AdamW:
    def __init__(self,p,lr=.003,decay=1e-4):
        if lr<=0 or decay<0: raise ValueError('Invalid learning rate or decay')
        self.lr,self.decay,self.t=lr,decay,0
        self.m={k:np.zeros_like(v) for k,v in p.items()}; self.v={k:np.zeros_like(v) for k,v in p.items()}
    def step(self,p,g,clip=5.):
        if any(not np.isfinite(v).all() for v in g.values()): raise FloatingPointError('Nonfinite gradients')
        norm=np.sqrt(sum(np.sum(v*v) for v in g.values()))
        scale=min(1.,clip/(norm+1e-12)) if clip else 1.
        self.t+=1
        for k,w in p.items():
            d=g[k]*scale
            self.m[k]=.9*self.m[k]+.1*d; self.v[k]=.999*self.v[k]+.001*d*d
            mh=self.m[k]/(1-.9**self.t); vh=self.v[k]/(1-.999**self.t)
            if k.startswith('W'): w*=1-self.lr*self.decay
            w-=self.lr*mh/(np.sqrt(vh)+1e-8)
        return float(norm)

def take(d,ids): return {k:v[ids] for k,v in d.items() if k!='y'}
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save_json(path,v): Path(path).write_text(json.dumps(v,indent=2,allow_nan=False))
def load_npz(path):
    with np.load(path,allow_pickle=False) as f: return {k:f[k].copy() for k in f.files}

def split(y,task,seed=7):
    rng=np.random.default_rng(seed); parts=[[],[],[]]
    groups=[np.arange(len(y))] if task=='regression' else [np.flatnonzero(y==c) for c in np.unique(y)]
    for g in groups:
        if len(g)<10: raise ValueError('Need at least 10 examples in each class or regression dataset')
        a=rng.permutation(g); n=int(.7*len(a)); v=int(.15*len(a))
        for dest,ids in zip(parts,[a[:n],a[n:n+v],a[n+v:]]): dest.extend(ids.tolist())
    return {k:rng.permutation(v).tolist() for k,v in zip(['train','val','test'],parts)}

def metrics(y,z,task):
    if task=='regression':
        d=z.ravel()-y.ravel(); ss=np.sum((y-y.mean())**2)
        return {'mae':float(abs(d).mean()),'rmse':float(np.sqrt(np.mean(d*d))),'r2':float(1-np.sum(d*d)/ss) if ss>0 else None}
    y=y.astype(int); pred=z.argmax(1); K=z.shape[1]
    cm=np.zeros((K,K),int); np.add.at(cm,(y,pred),1)
    tp=np.diag(cm); pr=np.divide(tp,cm.sum(0),out=np.zeros(K),where=cm.sum(0)>0)
    re=np.divide(tp,cm.sum(1),out=np.zeros(K),where=cm.sum(1)>0)
    f=np.divide(2*pr*re,pr+re,out=np.zeros(K),where=(pr+re)>0)
    return {'accuracy':float((pred==y).mean()),'macro_f1':float(f.mean()),'recall':re.tolist(),'confusion_matrix':cm.tolist()}

def fit(model,d,y,parts,task,epochs=60,batch=32,lr=.003,patience=15,seed=7,decay=1e-4,clip=5.):
    if min(epochs,batch,patience)<1: raise ValueError('Positive epochs, batch and patience required')
    opt=AdamW(model.params,lr,decay); rng=np.random.default_rng(seed)
    best={k:v.copy() for k,v in model.params.items()}; bestloss=float('inf'); bestepoch=0; stale=0; history=[]
    tr=np.array(parts['train']); va=np.array(parts['val'])
    def score(ids):
        total=0.
        for a in range(0,len(ids),batch):
            ii=ids[a:a+batch]; total+=model.loss(take(d,ii),y[ii],task)*len(ii)
        return total/len(ids)
    for epoch in range(1,epochs+1):
        order=rng.permutation(tr); norms=[]
        for start in range(0,len(order),batch):
            ids=order[start:start+batch]
            loss,g=model.loss_grad(take(d,ids),y[ids],task)
            if not np.isfinite(loss): raise FloatingPointError('Nonfinite loss')
            norms.append(opt.step(model.params,g,clip))
        tl,vl=score(tr),score(va)
        if not np.isfinite(tl+vl): raise FloatingPointError('Nonfinite epoch loss')
        history.append({'epoch':epoch,'train_loss':tl,'val_loss':vl,'gradient_norm_max':max(norms)})
        if vl<bestloss-1e-7:
            bestloss,bestepoch,stale=vl,epoch,0; best={k:v.copy() for k,v in model.params.items()}
        else: stale+=1
        if stale>=patience: break
    for k in model.params: model.params[k][...]=best[k]
    return history,bestepoch

class BaseModel:
    def loss(self,d,y,task): return loss_grad(self.forward(d),y,task)[0]
    def save(self,path): np.savez_compressed(path,**self.params)
    def load(self,path):
        with np.load(path,allow_pickle=False) as f:
            for k in self.params:
                if self.params[k].shape!=f[k].shape: raise ValueError('Checkpoint shape mismatch')
                self.params[k][...]=f[k]
        return self
