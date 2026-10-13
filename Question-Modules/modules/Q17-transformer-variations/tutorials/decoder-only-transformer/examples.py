"""Numerical examples used in the deck. Run directly, no training required."""
import json
import math
import torch
from torch.nn import functional as F
from tokenizer import ByteTokenizer, toy_bpe_demo

def examples():
    tok=ByteTokenizer()
    E=torch.tensor([[0.1,0.2,-0.1],[0.5,-0.3,0.8],[0.,0.4,0.2],[0.7,0.1,-0.2]])
    # Toy token ID=2 and position=0, unrelated to the byte vocabulary.
    position=torch.tensor([0.01,-0.02,0.03])
    query=torch.tensor([math.sqrt(2.),0.])
    keys=torch.tensor([[2.,0.],[1.,1.],[0.,2.]])
    scores=(query@keys.T)/math.sqrt(2.)
    a=F.softmax(scores,dim=-1)
    values=torch.tensor([[1.,0.],[0.,2.],[3.,1.]])
    x=torch.tensor([1.,2.,3.])
    gelu_x=torch.tensor([-2.,-1.,0.,1.,2.])
    ffn_in=torch.tensor([1.,-1.]); w1=torch.tensor([[1.,0.,1.,-1.],[0.,1.,1.,1.]])
    w2=torch.tensor([[1.,0.],[0.,1.],[1.,1.],[0.5,-0.5]])
    out={'byte_Hi':tok.encode('Hi!'),'byte_cafe':tok.encode('café'),'bpe_trace':toy_bpe_demo(),
         'embedding_row':E[2].tolist(),'embedding_plus_position':(E[2]+position).tolist(),
         'query':query.tolist(),'keys':keys.tolist(),'scores':scores.tolist(),'attention_weights':a.tolist(),'values':values.tolist(),'weighted_sum':(a@values).tolist(),
         'layernorm':F.layer_norm(x,(3,)).tolist(),'gelu_x':gelu_x.tolist(),'gelu_y':F.gelu(gelu_x).tolist(),
         'ffn_pre_activation':(ffn_in@w1).tolist(),'ffn_output':(F.gelu(ffn_in@w1)@w2).tolist(),
         'cross_entropy_target_0':F.cross_entropy(scores[None],torch.tensor([0])).item(),
         'temperatures':{str(t):F.softmax(scores/t,dim=-1).tolist() for t in [0.5,1.,2.]}}
    print(json.dumps(out,indent=2)); return out

if __name__=='__main__': examples()
