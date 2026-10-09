import tempfile,unittest
from pathlib import Path
import numpy as np
import data_api as api
from model import Model
from common import loss_grad,softmax,split,take

class CoreTests(unittest.TestCase):
    def setUp(self):
        self.d=api.generate('classification',n=6,seed=31)
        self.cfg=api.config(self.d,3 if api.NAME=='CNN' else 2)
        if api.NAME=='CNN': self.cfg.update(filters1=2,filters2=2)
        else: self.cfg['hidden']=3
    def test_parameter_gradients(self):
        d=take(self.d,[0,1]); rng=np.random.default_rng(4)
        for task in ['classification','regression']:
            cfg={**self.cfg,'outputs':1 if task=='regression' else self.cfg['outputs']}
            net=Model(cfg,seed=23); y=np.array([.2,-.4]) if task=='regression' else np.array([0,1])
            _,g=net.loss_grad(d,y,task)
            for key,v in net.params.items():
                for flat in rng.choice(v.size,min(6,v.size),replace=False):
                    idx=np.unravel_index(flat,v.shape); old=v[idx]; eps=1e-6
                    v[idx]=old+eps; plus=net.loss(d,y,task)
                    v[idx]=old-eps; minus=net.loss(d,y,task)
                    v[idx]=old
                    np.testing.assert_allclose(g[key][idx],(plus-minus)/(2*eps),atol=3e-6,rtol=3e-4,err_msg=f'{task}: {key} {idx}')
    def test_checkpoint(self):
        net=Model(self.cfg); d=take(self.d,[0,1])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'w.npz'; net.save(path)
            reloaded=Model(self.cfg,seed=99).load(path)
            np.testing.assert_array_equal(net.forward(d),reloaded.forward(d))
    def test_split(self):
        y=np.arange(120)%2; p=split(y,'classification')
        self.assertEqual(sorted(sum(p.values(),[])),list(range(120)))
        self.assertEqual(p,split(y,'classification'))
    def test_stable_loss(self):
        loss,g=loss_grad(np.array([[1000.,-1000.],[-1000.,1000.]]),np.array([1,0]),'classification')
        self.assertTrue(np.isfinite(loss) and np.isfinite(g).all())
    def test_train_only_statistics(self):
        ids=[0,1,2]; state=api.fit_pre(self.d,ids)
        changed={k:v.copy() for k,v in self.d.items()}
        if api.NAME=='CNN': changed['X'][3:]+=1000
        elif api.NAME=='GNN': changed['X'][3:,:,1]+=1000
        else: changed['X'][3:,:,0]+=1000
        self.assertEqual(state,api.fit_pre(changed,ids))
