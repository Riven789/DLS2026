"""Two GraphSAGE-style mean layers. Dense padded adjacency is pedagogical,
not a scalable graph storage format. Self and neighbor maps are separate.
"""
import numpy as np
from common import BaseModel,loss_grad


def neighbor_matrix(a):
    degree=a.sum(-1,keepdims=True)
    return np.divide(a,degree,out=np.zeros_like(a,dtype=float),where=degree>0)

class Model(BaseModel):
    def __init__(self,config,seed=7):
        self.config=config; rng=np.random.default_rng(seed); f=config['features']; h=config.get('hidden',16); k=config['outputs']
        self.params={}
        for i,nin in enumerate([f,h],1):
            for role in ['s','n']: self.params[f'W{role}{i}']=rng.normal(0,1/np.sqrt(nin),(nin,h))
            self.params[f'b{i}']=np.zeros(h)
        self.params['Wo']=rng.normal(0,1/np.sqrt(h),(h,k)); self.params['bo']=np.zeros(k)
    def forward(self,d,cache=False):
        p=self.params; m=d['mask'][...,None]; a=neighbor_matrix(d['A']); x=d['X']*m; saved=[]
        for i in [1,2]:
            neigh=a@x; z=x@p[f'Ws{i}']+neigh@p[f'Wn{i}']+p[f'b{i}']
            saved.append((x,neigh,z)); x=np.maximum(z,0)*m
        n=m.sum(1); pooled=x.sum(1)/n
        out=pooled@p['Wo']+p['bo']
        return (out,(saved,a,m,n,pooled)) if cache else out
    def loss_grad(self,d,y,task):
        z,(saved,a,m,n,pooled)=self.forward(d,True); loss,dz=loss_grad(z,y,task)
        p=self.params; grads={'Wo':pooled.T@dz,'bo':dz.sum(0)}
        dh=(dz@p['Wo'].T)[:,None,:]/n[:,None,:]*m
        for i in [2,1]:
            x,neigh,z=saved[i-1]; dz=dh*(z>0)*m
            grads[f'Ws{i}']=np.einsum('bnf,bnh->fh',x,dz)
            grads[f'Wn{i}']=np.einsum('bnf,bnh->fh',neigh,dz)
            grads[f'b{i}']=dz.sum((0,1))
            dh=dz@p[f'Ws{i}'].T+a.transpose(0,2,1)@(dz@p[f'Wn{i}'].T)
        return loss,grads
