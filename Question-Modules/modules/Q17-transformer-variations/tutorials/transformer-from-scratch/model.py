"""
model.py -- a decoder-only, dense transformer language model (GPT-style).

Every equation on the slides appears here as one or two lines of code.
Notation (same as the slides):

    V  vocabulary size          T  sequence length (<= T_max, the context window)
    d  model width              h  number of attention heads,  d_k = d / h
    L  number of blocks         B  batch size

Architecture: learned token + position embeddings, L pre-LayerNorm blocks
(causal multi-head self-attention, then a GELU feed-forward network), a final
LayerNorm, and an unembedding matrix tied to the token embedding.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 512    # V
    block_size: int = 128    # T_max
    n_layer: int = 4         # L
    n_head: int = 4          # h
    n_embd: int = 128        # d
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """A = softmax(Q K^T / sqrt(d_k) + M),   output = (A V) W_O."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.h, self.d_k = cfg.n_head, cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)   # W_Q, W_K, W_V stacked
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)      # W_O
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        # M: 0 on and below the diagonal, -inf above it (no looking ahead)
        mask = torch.full((cfg.block_size, cfg.block_size), float("-inf")).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        self.last_attn = None                              # kept for visualisation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)                       # each (B, T, d)
        # split d into h heads of width d_k:  (B, T, d) -> (B, h, T, d_k)
        q, k, v = (t.view(B, T, self.h, self.d_k).transpose(1, 2) for t in (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)       # (B, h, T, T)
        scores = scores + self.mask[:T, :T]                          # causal mask
        attn = F.softmax(scores, dim=-1)                             # rows sum to 1
        self.last_attn = attn.detach()
        out = self.attn_drop(attn) @ v                               # (B, h, T, d_k)
        out = out.transpose(1, 2).reshape(B, T, d)                   # concatenate heads
        return self.resid_drop(self.proj(out))


class FeedForward(nn.Module):
    """FFN(x) = W_2 GELU(W_1 x + b_1) + b_2, applied to every position independently."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)   # expand  d -> 4d
        self.fc2 = nn.Linear(4 * cfg.n_embd, cfg.n_embd)   # project 4d -> d
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    """Pre-LN residual block:  x <- x + Attn(LN(x));   x <- x + FFN(LN(x))."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)    # E  (V, d)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)    # P  (T_max, d)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight                     # weight tying: W_U = E

        self.apply(self._init_weights)
        # GPT-2 trick: shrink the layers that write into the residual stream so
        # the stream's variance stays O(1) after 2L additions.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: (B, T) token ids.  Returns logits (B, T, V) and, if targets given, the loss."""
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence longer than the context window"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))       # X^0 = E[x] + P
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))                           # Z = LN(X^L) E^T
        loss = None
        if targets is not None:
            # mean over all B*T positions of  -log p(correct next token)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, top_p: float | None = None) -> torch.Tensor:
        """Autoregressive sampling: predict one token, append it, repeat."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size:])        # crop to the window
            logits = logits[:, -1, :]                              # last position only
            idx_next = sample_next(logits, temperature, top_k, top_p)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


def sample_next(logits: torch.Tensor, temperature: float = 1.0,
                top_k: int | None = None, top_p: float | None = None) -> torch.Tensor:
    """Turn a (B, V) row of logits into one sampled token id per row, shape (B, 1)."""
    if temperature == 0.0:                                         # greedy decoding
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:                                          # keep the k best
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None:                                          # nucleus sampling
        sorted_logits, order = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        # drop a token if the mass *before* it already exceeds top_p
        drop = probs.cumsum(dim=-1) - probs > top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(1, order, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
