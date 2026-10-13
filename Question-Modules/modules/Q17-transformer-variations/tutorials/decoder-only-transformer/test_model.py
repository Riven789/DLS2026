"""Meaningful mathematical and integration invariants; CPU, no downloads."""
import copy
import unittest
import torch
from torch.nn import functional as F
from dense_lm import Config, DenseLM, make_optimizer
from tokenizer import ByteTokenizer

class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(7)
        self.model=DenseLM(Config(context=16,width=32,layers=2,heads=4))

    def test_tokenizer_roundtrip(self):
        t=ByteTokenizer()
        for text in ['Hi!','café','科学 🧪','']:
            self.assertEqual(t.decode(t.encode(text,bos=True,eos=True)),text)

    def test_future_cannot_change_past(self):
        self.model.eval(); x=torch.randint(0,258,(2,12)); y=x.clone(); y[:,7:]=0
        torch.testing.assert_close(self.model(x)[0][:,:7],self.model(y)[0][:,:7],rtol=0,atol=0)

    def test_mask_and_rows(self):
        attn=self.model.blocks[0].attn.eval()
        _,a=attn(torch.randn(2,10,32),return_attention=True)
        torch.testing.assert_close(a.sum(-1),torch.ones(2,4,10))
        self.assertEqual(torch.count_nonzero(torch.triu(a,diagonal=1)).item(),0)

    def test_manual_matches_sdpa(self):
        manual=self.model.blocks[0].attn.eval(); fused=copy.deepcopy(manual); fused.mode='sdpa'
        x=torch.randn(2,12,32)
        torch.testing.assert_close(manual(x),fused(x),rtol=1e-5,atol=1e-6)

    def test_loss_derivative(self):
        logits=torch.tensor([[2.,1.,0.]],requires_grad=True)
        F.cross_entropy(logits,torch.tensor([0])).backward()
        expected=logits.detach().softmax(-1); expected[0,0]-=1
        torch.testing.assert_close(logits.grad,expected)

    def test_small_batch_learns(self):
        x=torch.arange(12).repeat(4,1); y=(x+1)%258
        optimizer=make_optimizer(self.model,lr=0.01)
        first=self.model(x,y)[1].item()
        for _ in range(35):
            optimizer.zero_grad(); loss=self.model(x,y)[1]; loss.backward(); optimizer.step()
        self.assertLess(self.model(x,y)[1].item(),first*0.3)

    def test_generation_and_context_guard(self):
        output=self.model.generate(torch.tensor([[256,72]]),max_new_tokens=3,temperature=0,eos_id=-1)
        self.assertEqual(tuple(output.shape),(1,5))
        with self.assertRaises(ValueError): self.model(torch.zeros(1,17,dtype=torch.long))

if __name__=='__main__': unittest.main(verbosity=2)
