import unittest
import numpy as np
from model import conv,conv_backward,avg_pool,pool_backward

class CNNTests(unittest.TestCase):
    def test_worked_convolution(self):
        x=np.array([[[[1.,2.,0.],[0.,1.,3.],[2.,1.,0.]]]])
        w=np.array([[[[1.,0.],[0.,-1.]]]])
        z,_=conv(x,w,np.array([0.]),pad=0)
        np.testing.assert_array_equal(z,np.array([[[[0.,-1.],[-1.,1.]]]]))
    def test_pool_gradient_and_input_gradient(self):
        x=np.arange(16,dtype=float).reshape(1,1,4,4)
        np.testing.assert_allclose(avg_pool(x),[[[[2.5,4.5],[10.5,12.5]]]])
        np.testing.assert_allclose(pool_backward(np.ones((1,1,2,2))),.25)
        w=np.ones((1,1,3,3)); z,cache=conv(x,w,np.zeros(1))
        dx,_,_=conv_backward(np.ones_like(z),cache)
        np.testing.assert_allclose(dx[0,0],[[4,6,6,4],[6,9,9,6],[6,9,9,6],[4,6,6,4]])
