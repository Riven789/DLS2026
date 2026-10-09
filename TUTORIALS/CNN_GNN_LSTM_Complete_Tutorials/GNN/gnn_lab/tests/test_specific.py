import unittest
import numpy as np
import data_api as api
from model import Model,neighbor_matrix
from common import take

class GNNTests(unittest.TestCase):
    def test_permutation_and_padding_invariance(self):
        d=api.generate('classification',n=4); d=take(d,range(4)); net=Model(api.config(d,2))
        p=np.random.default_rng(5).permutation(d['X'].shape[1]); original=net.forward(d)
        perm={'X':d['X'][:,p], 'mask':d['mask'][:,p], 'A':d['A'][:,p][:,:,p]}
        np.testing.assert_allclose(original,net.forward(perm),atol=1e-12)
        pad={'X':np.pad(d['X'],((0,0),(0,3),(0,0))),'mask':np.pad(d['mask'],((0,0),(0,3))),'A':np.pad(d['A'],((0,0),(0,3),(0,3)))}
        np.testing.assert_allclose(original,net.forward(pad),atol=1e-12)
    def test_isolated_nodes_and_invalid_edges(self):
        a=np.array([[[0.,1.,0.],[1.,0.,0.],[0.,0.,0.]]]); m=neighbor_matrix(a)
        np.testing.assert_array_equal(m.sum(-1),[[1,1,0]])
        d=api.generate('classification',n=2); d['mask'][0,-1]=0; d['A'][0,0,-1]=d['A'][0,-1,0]=1
        with self.assertRaises(ValueError): api.validate(d)
