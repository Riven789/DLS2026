"""
train.py -- training harness for the tutorial GPT.

    python train.py                       # BPE tokens, ~0.9M parameters, CPU-friendly
    python train.py --tokenizer char      # character-level instead
    python train.py --n_layer 6 --n_embd 384 --n_head 6 --block_size 256 \
                    --batch_size 64 --max_steps 5000      # a GPU-sized run

What it does:
  1. trains (or loads) a tokenizer and encodes the corpus once,
  2. samples random (input, target) windows, target = input shifted by one token,
  3. optimises with AdamW, linear warmup + cosine decay, gradient clipping,
  4. periodically estimates train/val loss, logs to CSV, saves the best checkpoint,
  5. prints a sample at the end.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import asdict

import torch

from model import GPT, GPTConfig
from tokenizer import BPETokenizer, CharTokenizer, load_tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--data", default="data/tinyshakespeare.txt")
    p.add_argument("--tokenizer", choices=["bpe", "char"], default="bpe")
    p.add_argument("--vocab_size", type=int, default=512, help="BPE vocabulary size")
    p.add_argument("--out_dir", default="out")
    # model
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    # optimisation
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3, help="peak learning rate")
    p.add_argument("--min_lr", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # bookkeeping
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_batches", type=int, default=40)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def lr_at(step: int, args: argparse.Namespace) -> float:
    """Linear warmup to args.lr, then cosine decay to args.min_lr."""
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    """x = tokens[i : i+T],  y = tokens[i+1 : i+T+1]  for random offsets i."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model: GPT, splits: dict, args: argparse.Namespace) -> dict:
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = [model(*get_batch(data, args.block_size, args.batch_size, args.device))[1].item()
                  for _ in range(args.eval_batches)]
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def make_optimizer(model: GPT, args: argparse.Namespace) -> torch.optim.AdamW:
    """AdamW with decoupled weight decay applied to weight matrices and embeddings
    (parameters with >= 2 dimensions) but not to biases or LayerNorm gains."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": args.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=args.lr, betas=(args.beta1, args.beta2), eps=1e-8)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 1. tokenizer + data ------------------------------------------------
    with open(args.data, encoding="utf-8") as f:
        text = f.read()
    tok_path = os.path.join(args.out_dir, "tokenizer.json")
    if os.path.exists(tok_path):
        tok = load_tokenizer(tok_path)
        assert tok.kind == args.tokenizer, f"{tok_path} holds a {tok.kind} tokenizer; use a new --out_dir"
    else:
        t0 = time.time()
        tok = (CharTokenizer.train(text) if args.tokenizer == "char"
               else BPETokenizer.train(text, args.vocab_size))
        tok.save(tok_path)
        print(f"trained {args.tokenizer} tokenizer, V = {tok.vocab_size}, {time.time() - t0:.1f}s")
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}
    print(f"corpus: {len(text):,} characters -> {len(data):,} tokens "
          f"({len(text) / len(data):.2f} chars/token); train {n:,}, val {len(data) - n:,}")

    # ---- 2. model -----------------------------------------------------------
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    model = GPT(cfg).to(args.device)
    print(f"model: {model.num_params():,} parameters; "
          f"a uniform guess would score ln V = {math.log(cfg.vocab_size):.3f}")
    opt = make_optimizer(model, args)

    # ---- 3. training loop ---------------------------------------------------
    log_path = os.path.join(args.out_dir, "log.csv")
    log_f = open(log_path, "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["step", "train_loss", "val_loss", "lr", "elapsed_s"])
    best_val, t0 = float("inf"), time.time()

    for step in range(args.max_steps + 1):
        if step % args.eval_interval == 0 or step == args.max_steps:
            est = estimate_loss(model, splits, args)
            lr_now = lr_at(min(step, args.max_steps - 1), args)
            log.writerow([step, f"{est['train']:.4f}", f"{est['val']:.4f}",
                          f"{lr_now:.6f}", f"{time.time() - t0:.1f}"])
            log_f.flush()
            print(f"step {step:5d} | train {est['train']:.4f} | val {est['val']:.4f} "
                  f"| lr {lr_now:.2e} | {time.time() - t0:6.1f}s")
            if est["val"] < best_val:
                best_val = est["val"]
                torch.save({"model": model.state_dict(), "config": asdict(cfg), "step": step,
                            "val_loss": best_val, "args": vars(args)},
                           os.path.join(args.out_dir, "ckpt.pt"))
        if step == args.max_steps:
            break

        for g in opt.param_groups:                       # set this step's learning rate
            g["lr"] = lr_at(step, args)
        x, y = get_batch(splits["train"], args.block_size, args.batch_size, args.device)
        _, loss = model(x, y)                            # forward pass
        opt.zero_grad(set_to_none=True)
        loss.backward()                                  # backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()                                       # AdamW update

    log_f.close()
    print(f"done: best val loss {best_val:.4f}; log -> {log_path}")

    # ---- 4. a sample from the best checkpoint -------------------------------
    ckpt = torch.load(os.path.join(args.out_dir, "ckpt.pt"), map_location=args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    prompt = torch.tensor([tok.encode("\n")], device=args.device)
    out = model.generate(prompt, max_new_tokens=200, temperature=0.8, top_k=50)
    print("-" * 60)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
