"""Single-process CPU/CUDA/MPS training, validation, JSONL logs and resume."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time
import torch
from dense_lm import Config, DenseLM, make_optimizer
from tokenizer import ByteTokenizer


def device_for(name):
    if name != 'auto':
        return torch.device(name)
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


def lr_at(step, total, warmup, peak):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup - 1)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, progress))))


def streams(path, seed, context):
    raw = Path(path).read_bytes()
    # Exact duplicates removed BEFORE splitting. In real corpora also deduplicate
    # near duplicates and split by source/group to prevent related-document leaks.
    docs = sorted(set(d.strip() for d in raw.decode('utf-8').split('\n\n') if d.strip()))
    if len(docs) < 10:
        raise ValueError('Need >=10 documents separated by blank lines')
    random.Random(seed).shuffle(docs)
    cut = max(1, len(docs) // 10)
    tok = ByteTokenizer()
    def encode(items):
        return torch.tensor([i for d in items for i in tok.encode(d, bos=True, eos=True)], dtype=torch.long)
    train, val = encode(docs[cut:]), encode(docs[:cut])
    if min(len(train), len(val)) <= context:
        raise ValueError('Each split must have more than context tokens')
    return train, val, hashlib.sha256(raw).hexdigest()


def batch(data, size, context, generator, device):
    starts = torch.randint(len(data) - context, (size,), generator=generator)
    x = torch.stack([data[i:i+context] for i in starts])
    y = torch.stack([data[i+1:i+context+1] for i in starts])
    return x.to(device), y.to(device)


def autocast_for(device, precision):
    return torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()


@torch.no_grad()
def evaluate(model, data, args, device):
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(args.seed + 1000)
    losses, correct, count = [], 0, 0
    for _ in range(args.eval_batches):
        x, y = batch(data, args.batch_size, model.cfg.context, generator, device)
        with autocast_for(device, args.precision):
            logits, loss = model(x, y)
        losses.append(loss.item())
        correct += (logits.argmax(-1) == y).sum().item()
        count += y.numel()
    model.train(was_training)
    value = sum(losses) / len(losses)
    return {'val_loss': value, 'val_ppl': math.exp(min(80, value)), 'val_accuracy': correct / count}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/science.txt'); p.add_argument('--out', default='runs/demo')
    p.add_argument('--steps', type=int, default=500, help='Total optimizer-step horizon, including resumed steps')
    p.add_argument('--stop-after', type=int, default=0, help='Stop early without changing LR horizon')
    p.add_argument('--context', type=int, default=128); p.add_argument('--width', type=int, default=128)
    p.add_argument('--layers', type=int, default=4); p.add_argument('--heads', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=8); p.add_argument('--accum', type=int, default=1)
    p.add_argument('--lr', type=float, default=3e-4); p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--dropout', type=float, default=0.0); p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--eval-every', type=int, default=50); p.add_argument('--eval-batches', type=int, default=10)
    p.add_argument('--device', default='auto', choices=['auto','cpu','cuda','mps'])
    p.add_argument('--precision', default='fp32', choices=['fp32','bf16'])
    p.add_argument('--attention', default='manual', choices=['manual','sdpa'])
    p.add_argument('--seed', type=int, default=42); p.add_argument('--threads', type=int, default=4)
    p.add_argument('--resume', default='')
    args = p.parse_args()
    if min(args.steps,args.batch_size,args.accum,args.eval_every,args.eval_batches,args.threads) < 1 or args.warmup < 0 or args.lr <= 0:
        p.error('Positive sizes/steps/lr required, warmup >= 0')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = device_for(args.device)
    if args.precision == 'bf16' and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        p.error('bf16 requires a CUDA device with bf16 support; use fp32 otherwise')
    cfg = Config(context=args.context,width=args.width,layers=args.layers,heads=args.heads,dropout=args.dropout,attention=args.attention)
    train, val, digest = streams(args.data, args.seed, cfg.context)
    model = DenseLM(cfg).to(device)
    optimizer = make_optimizer(model,args.lr,args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed + 1)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    start, best = 0, float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=True)
        if ckpt['config'] != asdict(cfg) or ckpt['data_sha256'] != digest:
            raise ValueError('Resume requires the same model configuration and corpus')
        for key in ('steps','batch_size','accum','lr','weight_decay','warmup','seed','precision'):
            if ckpt['args'][key] != vars(args)[key]:
                raise ValueError(f'Resume must preserve {key}')
        model.load_state_dict(ckpt['model']); optimizer.load_state_dict(ckpt['optimizer'])
        generator.set_state(ckpt['batch_rng']); torch.set_rng_state(ckpt['torch_rng'])
        if device.type == 'cuda' and ckpt.get('cuda_rng'):
            torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
        if device.type == 'mps' and ckpt.get('mps_rng') is not None:
            torch.mps.set_rng_state(ckpt['mps_rng'])
        start, best = ckpt['step'], ckpt['best_val']
    metadata = {'config':asdict(cfg),'args':vars(args),'device':str(device),'torch_version':str(torch.__version__),
                'parameters':sum(p.numel() for p in model.parameters()),'train_tokens':len(train),'val_tokens':len(val),'data_sha256':digest}
    (out/'run.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata),flush=True)
    initial = {'step':start,**evaluate(model,val,args,device)}
    print(json.dumps(initial),flush=True)
    with (out/'metrics.jsonl').open('a' if args.resume else 'w') as log:
        log.write(json.dumps(initial)+'\n'); log.flush()
        for step in range(start,args.steps):
            tick = time.perf_counter()
            model.train(); optimizer.zero_grad(set_to_none=True)
            lr = lr_at(step,args.steps,args.warmup,args.lr)
            for group in optimizer.param_groups: group['lr'] = lr
            train_loss = 0.0
            for _ in range(args.accum):
                x,y = batch(train,args.batch_size,cfg.context,generator,device)
                with autocast_for(device,args.precision): _,loss = model(x,y)
                train_loss += loss.detach().item()/args.accum
                (loss/args.accum).backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            optimizer.step()
            record = {'step':step+1,'train_loss':train_loss,'lr':lr,'grad_norm':float(norm),
                      'tokens':(step+1)*args.batch_size*args.accum*cfg.context,'seconds':time.perf_counter()-tick}
            stop = args.stop_after and step+1 >= args.stop_after
            if (step+1)%args.eval_every == 0 or step+1 == args.steps or stop:
                record.update(evaluate(model,val,args,device))
                improved = record['val_loss'] < best
                best = min(best,record['val_loss'])
                checkpoint = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':asdict(cfg),
                    'args':vars(args),'data_sha256':digest,'step':step+1,'best_val':best,
                    'torch_rng':torch.get_rng_state(),'batch_rng':generator.get_state(),
                    'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                    'mps_rng':torch.mps.get_rng_state() if device.type=='mps' else None}
                torch.save(checkpoint,out/'last.pt')
                if improved: torch.save(checkpoint,out/'best.pt')
                print(json.dumps(record),flush=True)
            log.write(json.dumps(record)+'\n'); log.flush()
            if stop: break


if __name__ == '__main__': main()
