import unittest
import numpy as np
import data_api as api
from model import Model
from common import take

class LSTMTests(unittest.TestCase):
    def test_masked_padding_preserves_state_and_gradients(self):
        d=api.generate('classification',n=4); y=d.pop('y'); net=Model(api.config(d,2))
        original,g=net.loss_grad(d,y,'classification')
        padded={'X':np.pad(d['X'],((0,0),(0,3),(0,0)),constant_values=17),'mask':np.pad(d['mask'],((0,0),(0,3)))}
        loss,g2=net.loss_grad(padded,y,'classification')
        self.assertAlmostEqual(original,loss)
        for key in g: np.testing.assert_allclose(g[key],g2[key],atol=1e-12)
    def test_one_step_and_invalid_mask(self):
        net=Model({'features':2,'hidden':1,'outputs':1})
        for v in net.params.values(): v.fill(0)
        net.params['b'][2]=np.arctanh(.4); net.params['Wo'].fill(1)
        z=net.forward({'X':np.zeros((1,1,2)),'mask':np.ones((1,1))})
        np.testing.assert_allclose(z,.5*np.tanh(.2))
        d=api.generate('classification',n=1); d['mask'][0,:3]=[1,0,1]
        with self.assertRaises(ValueError): api.validate(d)
