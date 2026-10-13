"""Transparent pre-LayerNorm, decoder-only, dense transformer.

Notation: B=batch, T=tokens, C=model width, H=heads, D=C/H, V=vocabulary.
nn.Linear stores weights transposed relative to the row-vector equations.
"""
import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int = 258
    context: int = 128
    width: int = 128
    layers: int = 4
    heads: int = 4
    dropout: float = 0.0
    attention: str = 'manual'

    def __post_init__(self):
        if min(self.vocab_size, self.context, self.width, self.layers, self.heads) < 1:
            raise ValueError('Model dimensions must be positive')
        if self.width % self.heads:
            raise ValueError('width must be divisible by heads')
        if self.attention not in ('manual', 'sdpa') or not 0 <= self.dropout < 1:
            raise ValueError('Invalid attention or dropout')


class CausalAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.dim = cfg.heads, cfg.width // cfg.heads
        self.mode, self.p = cfg.attention, cfg.dropout
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width)
        self.proj = nn.Linear(cfg.width, cfg.width)
        self.drop = nn.Dropout(cfg.dropout)
        self.register_buffer('allowed', torch.ones(cfg.context, cfg.context, dtype=torch.bool).tril(), persistent=False)

    def forward(self, x, return_attention=False):
        B, T, C = x.shape
        # [B,T,3C] -> three [B,H,T,D] tensors
        q, k, v = self.qkv(x).reshape(B, T, 3, self.heads, self.dim).permute(2, 0, 3, 1, 4).unbind(0)
        weights = None
        if self.mode == 'sdpa' and not return_attention:
            z = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                    dropout_p=self.p if self.training else 0.0)
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dim)
            scores = scores.masked_fill(~self.allowed[:T, :T], float('-inf'))
            # FP32 softmax is useful when projections use bf16 autocast.
            weights = F.softmax(scores.float(), dim=-1).to(v.dtype)
            z = self.drop(weights) @ v
        z = z.transpose(1, 2).contiguous().reshape(B, T, C)
        output = self.drop(self.proj(z))
        return (output, weights) if return_attention else output


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.width), nn.LayerNorm(cfg.width)
        self.attn = CausalAttention(cfg)
        self.ffn = nn.Sequential(nn.Linear(cfg.width, 4 * cfg.width),
            nn.GELU(approximate='none'), nn.Linear(4 * cfg.width, cfg.width), nn.Dropout(cfg.dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class DenseLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.width)
        self.position_embedding = nn.Embedding(cfg.context, cfg.width)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.final_norm = nn.LayerNorm(cfg.width)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, ids, targets=None):
        B, T = ids.shape
        if T < 1 or T > self.cfg.context:
            raise ValueError('Sequence length must be 1..context')
        positions = torch.arange(T, device=ids.device)
        x = self.drop(self.token_embedding(ids) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        # Weight tying: logits[b,t,v] = dot(x[b,t], E[v]); no output bias.
        logits = F.linear(x, self.token_embedding.weight)
        loss = None if targets is None else F.cross_entropy(
            logits.float().reshape(B*T, -1), targets.reshape(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=80, temperature=0.8, top_k=40, eos_id=257):
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError('Teaching generator supports one prompt at a time')
        if max_new_tokens < 0 or temperature < 0 or top_k < 0:
            raise ValueError('Sampling arguments must be nonnegative')
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                # No KV cache: recompute the recent window, positions start at 0.
                logits, _ = self(ids[:, -self.cfg.context:])
                scores = logits[:, -1, :].float()
                scores[:, 256] = -torch.inf  # BOS is a prompt marker, not emitted text.
                if temperature == 0:
                    next_id = scores.argmax(-1, keepdim=True)
                else:
                    scores = scores / temperature
                    if top_k:
                        values, indices = torch.topk(scores, min(top_k, scores.size(-1)))
                        scores = torch.full_like(scores, -torch.inf).scatter(-1, indices, values)
                    next_id = torch.multinomial(F.softmax(scores, dim=-1), 1)
                ids = torch.cat((ids, next_id), dim=1)
                if next_id.item() == eos_id:
                    break
        finally:
            self.train(was_training)
        return ids


def make_optimizer(model, lr=3e-4, weight_decay=0.1):
    # All matrices, including tied E and position embeddings, receive decay.
    # Biases and LayerNorm scale/shift vectors do not. Tied E appears once.
    matrices, vectors = [], []
    for p in model.parameters():
        (matrices if p.ndim >= 2 else vectors).append(p)
    return torch.optim.AdamW([{'params': matrices, 'weight_decay': weight_decay},
                              {'params': vectors, 'weight_decay': 0.0}],
                             lr=lr, betas=(0.9, 0.95), eps=1e-8)
