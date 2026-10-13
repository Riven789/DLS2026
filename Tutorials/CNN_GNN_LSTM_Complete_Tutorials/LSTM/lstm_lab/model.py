"""Masked many-to-one LSTM, explicit full backpropagation through time.
Gate order: input, forget, candidate, output. No peepholes or projection.
"""
import numpy as np
from common import BaseModel,loss_grad,sigmoid

class Model(BaseModel):
    def __init__(self,config,seed=7):
        self.config=config; rng=np.random.default_rng(seed)
        d=config['features']; h=config.get('hidden',16); o=config['outputs']
        self.params={'W':rng.normal(0,1/np.sqrt(d+h),(d+h,4*h)),'b':np.zeros(4*h),
                     'Wo':rng.normal(0,1/np.sqrt(h),(h,o)),'bo':np.zeros(o)}
        self.params['b'][h:2*h]=1. # Positive forget bias, teaching choice.
    def forward(self,d,cache=False):
        x=d['X']; mask=d['mask']; p=self.params; b,t,_=x.shape; H=self.config.get('hidden',16)
        h=np.zeros((b,H)); c=np.zeros_like(h); saved=[]
        for step in range(t):
            q=np.concatenate([x[:,step],h],1); v=q@p['W']+p['b']
            vi,vf,vg,vo=np.split(v,4,axis=1); i=sigmoid(vi); f=sigmoid(vf); g=np.tanh(vg); o=sigmoid(vo)
            cprev=c; raw_c=f*c+i*g; raw_h=o*np.tanh(raw_c); m=mask[:,step,None]
            c=m*raw_c+(1-m)*c; h=m*raw_h+(1-m)*h
            saved.append((q,cprev,i,f,g,o,raw_c,m))
        z=h@p['Wo']+p['bo']
        return (z,(saved,h)) if cache else z
    def loss_grad(self,d,y,task):
        z,(saved,h)=self.forward(d,True); loss,dz=loss_grad(z,y,task); p=self.params
        grads={k:np.zeros_like(v) for k,v in p.items()}
        grads['Wo']=h.T@dz; grads['bo']=dz.sum(0)
        dh=dz@p['Wo'].T; dc=np.zeros_like(dh); D=d['X'].shape[-1]
        for q,cprev,i,f,g,o,raw_c,m in reversed(saved):
            dhraw=dh*m; tc=np.tanh(raw_c)
            dcraw=dc*m+dhraw*o*(1-tc*tc)
            di=dcraw*g; df=dcraw*cprev; dg=dcraw*i; do=dhraw*tc
            dv=np.concatenate([di*i*(1-i),df*f*(1-f),dg*(1-g*g),do*o*(1-o)],1)
            grads['W']+=q.T@dv; grads['b']+=dv.sum(0)
            dq=dv@p['W'].T
            dh=dh*(1-m)+dq[:,D:]
            dc=dc*(1-m)+dcraw*f
        return loss,grads
