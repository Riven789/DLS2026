"""Two convolution blocks with explicit gradients; inputs use NCHW."""
import numpy as np
from common import BaseModel,loss_grad


def conv(x,w,b,pad=1):
    k=w.shape[-1]
    xp=np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)))
    patches=np.lib.stride_tricks.sliding_window_view(xp,(k,k),axis=(2,3))
    z=np.einsum('bchwij,ocij->bohw',patches,w,optimize=True)+b[None,:,None,None]
    return z,(x,w,patches,pad)


def conv_backward(dz,cache):
    x,w,patches,pad=cache; k=w.shape[-1]; h,t=dz.shape[-2:]
    dw=np.einsum('bohw,bchwij->ocij',dz,patches,optimize=True)
    db=dz.sum((0,2,3))
    dxp=np.zeros((len(x),x.shape[1],x.shape[2]+2*pad,x.shape[3]+2*pad))
    for i in range(k):
        for j in range(k):
            dxp[:,:,i:i+h,j:j+t]+=np.einsum('bohw,oc->bchw',dz,w[:,:,i,j],optimize=True)
    dx=dxp[:,:,pad:-pad,pad:-pad] if pad else dxp
    return dx,dw,db


def avg_pool(x):
    b,c,h,w=x.shape
    if h%2 or w%2: raise ValueError('Average pooling requires even height and width')
    return x.reshape(b,c,h//2,2,w//2,2).mean((3,5))

def pool_backward(g): return np.repeat(np.repeat(g,2,axis=2),2,axis=3)/4

class Model(BaseModel):
    def __init__(self,config,seed=7):
        self.config=config; rng=np.random.default_rng(seed)
        cin=config['channels']; f1=config.get('filters1',6); f2=config.get('filters2',8); out=config['outputs']
        self.params={'W1':rng.normal(0,np.sqrt(2/(cin*9)),(f1,cin,3,3)),'b1':np.zeros(f1),
          'W2':rng.normal(0,np.sqrt(2/(f1*9)),(f2,f1,3,3)),'b2':np.zeros(f2),
          'Wo':rng.normal(0,1/np.sqrt(f2),(f2,out)),'bo':np.zeros(out)}
    def forward(self,d,cache=False):
        p=self.params; z1,c1=conv(d['X'],p['W1'],p['b1']); a1=np.maximum(z1,0); pooled=avg_pool(a1)
        z2,c2=conv(pooled,p['W2'],p['b2']); a2=np.maximum(z2,0); g=a2.mean((2,3))
        out=g@p['Wo']+p['bo']
        return (out,(z1,c1,z2,c2,g)) if cache else out
    def loss_grad(self,d,y,task):
        z,(z1,c1,z2,c2,g)=self.forward(d,True); loss,dz=loss_grad(z,y,task)
        grads={'Wo':g.T@dz,'bo':dz.sum(0)}
        da2=np.broadcast_to((dz@self.params['Wo'].T)[:,:,None,None],z2.shape)/(z2.shape[2]*z2.shape[3])
        dpool,grads['W2'],grads['b2']=conv_backward(da2*(z2>0),c2)
        _,grads['W1'],grads['b1']=conv_backward(pool_backward(dpool)*(z1>0),c1)
        return loss,grads
