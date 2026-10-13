"""Generate text from a tutorial checkpoint."""
import argparse
import torch
from dense_lm import Config, DenseLM
from tokenizer import ByteTokenizer
from train import device_for

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default='runs/demo/best.pt'); p.add_argument('--prompt',default='The atom')
    p.add_argument('--tokens',type=int,default=100); p.add_argument('--temperature',type=float,default=0.8)
    p.add_argument('--top-k',type=int,default=40); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='auto',choices=['auto','cpu','cuda','mps']); p.add_argument('--threads',type=int,default=4)
    a=p.parse_args(); torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    device=device_for(a.device); model=DenseLM(Config(**ckpt['config'])).to(device)
    model.load_state_dict(ckpt['model']); tok=ByteTokenizer()
    ids=torch.tensor([tok.encode(a.prompt,bos=True)],device=device)
    result=model.generate(ids,a.tokens,a.temperature,a.top_k)
    print(tok.decode(result[0].tolist()))

if __name__=='__main__': main()
