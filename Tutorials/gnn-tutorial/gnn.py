"""
gnn.py -- graph neural network layers in plain PyTorch (no PyTorch Geometric, no DGL).

    GCNLayer    H' = A_hat H W                 A_hat = D~^(-1/2) (A + I) D~^(-1/2)   [Kipf & Welling]
    SAGELayer   H' = H W_self + (D^-1 A H) W_neigh        mean aggregator            [Hamilton et al.]
    GATLayer    H' = sum_u alpha_vu W h_u      learned attention weights alpha       [Velickovic et al.]
    GINLayer    H' = MLP((1 + eps) H + A H)    sum aggregator                        [Xu et al.]
    GNN         a stack of such layers with ReLU and dropout; kind="mlp" ignores the graph

The graph enters only through `ops`, built once by graph_operators(edges, n).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)          # the matrices here are small; extra threads only add overhead


def sparse_matrix(rows, cols, vals, n: int) -> torch.Tensor:
    return torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (n, n), check_invariants=False).coalesce()


def graph_operators(edges: np.ndarray, n: int) -> dict:
    """Everything the layers need to know about the graph.  `edges` lists each undirected edge once."""
    e = torch.as_tensor(edges, dtype=torch.long)
    src = torch.cat([e[:, 0], e[:, 1]])                  # both directions: u -> v and v -> u
    dst = torch.cat([e[:, 1], e[:, 0]])
    loops = torch.arange(n)
    src_l, dst_l = torch.cat([src, loops]), torch.cat([dst, loops])   # A~ = A + I
    ones, ones_l = torch.ones(len(src)), torch.ones(len(src_l))
    deg = torch.zeros(n).index_add_(0, dst, ones)        # degree d_v
    deg_l = deg + 1                                      # degree with the self-loop
    return dict(
        n=n,
        A=sparse_matrix(dst, src, ones, n),                                         # plain adjacency
        A_hat=sparse_matrix(dst_l, src_l, (deg_l[dst_l] * deg_l[src_l]).rsqrt(), n),  # GCN operator
        A_mean=sparse_matrix(dst, src, 1.0 / deg.clamp(min=1)[dst], n),             # neighbour mean
        edge_index=torch.stack([src_l, dst_l]),                                     # with self-loops
    )


class Linear(nn.Linear):
    """nn.Linear that also accepts a sparse input matrix (Cora's word features are 98.7% zeros)."""

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        if not H.is_sparse:
            return super().forward(H)
        out = torch.sparse.mm(H, self.weight.T)
        return out if self.bias is None else out + self.bias


def dropout(H: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Dropout; on a sparse matrix it acts on the stored non-zeros (zeros stay zero either way)."""
    if not H.is_sparse:
        return F.dropout(H, p, training)
    return torch.sparse_coo_tensor(H.indices(), F.dropout(H.values(), p, training), H.shape,
                                   check_invariants=False)


class GCNLayer(nn.Module):
    """Graph convolution: every node averages itself and its neighbours with weights
    1/sqrt(d_v d_u), then all nodes share one linear map."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.lin = Linear(n_in, n_out, bias=False)        # W
        self.bias = nn.Parameter(torch.zeros(n_out))
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, H: torch.Tensor, ops: dict) -> torch.Tensor:
        return torch.sparse.mm(ops["A_hat"], self.lin(H)) + self.bias     # A_hat (H W) + b


class DenseLayer(Linear):
    """An ordinary linear layer with the same call signature: it ignores the graph (the MLP baseline)."""

    def forward(self, H: torch.Tensor, ops: dict) -> torch.Tensor:
        return super().forward(H)


class SAGELayer(nn.Module):
    """GraphSAGE with the mean aggregator.  Keeping W_self and W_neigh separate is the same as
    concatenating [h_v, mean of neighbours] and applying one weight matrix."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.lin_self, self.lin_neigh = Linear(n_in, n_out), Linear(n_in, n_out, bias=False)

    def forward(self, H: torch.Tensor, ops: dict) -> torch.Tensor:
        return self.lin_self(H) + torch.sparse.mm(ops["A_mean"], self.lin_neigh(H))


class GINLayer(nn.Module):
    """Graph isomorphism network layer: SUM over neighbours, then a small MLP."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(()))
        self.mlp = nn.Sequential(nn.Linear(n_in, n_out), nn.ReLU(), nn.Linear(n_out, n_out))

    def forward(self, H: torch.Tensor, ops: dict) -> torch.Tensor:
        H = H.to_dense() if H.is_sparse else H
        return self.mlp((1 + self.eps) * H + torch.sparse.mm(ops["A"], H))


class GATLayer(nn.Module):
    """Graph attention: node v weights each neighbour u (and itself) by
    alpha_vu = softmax_u( LeakyReLU( a_dst . W h_v + a_src . W h_u ) ),  `heads` times in parallel."""

    def __init__(self, n_in: int, n_out: int, heads: int = 1, concat: bool = True, attn_dropout: float = 0.0):
        super().__init__()
        self.heads, self.n_out, self.concat, self.attn_dropout = heads, n_out, concat, attn_dropout
        self.lin = Linear(n_in, heads * n_out, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, n_out))
        self.a_dst = nn.Parameter(torch.empty(heads, n_out))
        self.bias = nn.Parameter(torch.zeros(heads * n_out if concat else n_out))
        for p in (self.lin.weight, self.a_src, self.a_dst):
            nn.init.xavier_uniform_(p)

    def attention(self, Z: torch.Tensor, ops: dict) -> torch.Tensor:
        """alpha with shape (number of edges incl. self-loops, heads); sums to 1 over each node's edges."""
        src, dst = ops["edge_index"]                                   # message goes src -> dst
        score = F.leaky_relu((Z * self.a_dst).sum(-1)[dst] + (Z * self.a_src).sum(-1)[src], 0.2)
        w = torch.exp(score - score.max().detach())                    # softmax over incoming edges
        return w / torch.zeros(ops["n"], self.heads).index_add_(0, dst, w)[dst].clamp(min=1e-16)

    def forward(self, H: torch.Tensor, ops: dict) -> torch.Tensor:
        src, dst = ops["edge_index"]
        Z = self.lin(H).view(-1, self.heads, self.n_out)               # (n, heads, n_out)
        alpha = F.dropout(self.attention(Z, ops), self.attn_dropout, self.training)
        out = torch.zeros_like(Z).index_add_(0, dst, alpha.unsqueeze(-1) * Z[src])
        return (out.flatten(1) if self.concat else out.mean(1)) + self.bias


class GNN(nn.Module):
    """n_layers message-passing layers; ReLU and dropout between them; the last layer gives logits."""

    def __init__(self, n_in: int, hidden: int, n_out: int, n_layers: int = 2, kind: str = "gcn",
                 dropout: float = 0.5, residual: bool = False, heads: int = 8):
        super().__init__()
        self.dropout, self.residual, self.kind = dropout, residual, kind
        sizes = [n_in] + [hidden] * (n_layers - 1) + [n_out]
        self.layers = nn.ModuleList()
        for l, (m, k) in enumerate(zip(sizes[:-1], sizes[1:])):
            last = l == n_layers - 1
            if kind == "gat":                            # hidden layers: `heads` heads, concatenated
                m = m if l == 0 else m * heads
                self.layers.append(GATLayer(m, k, 1 if last else heads, not last, dropout))
            else:
                self.layers.append({"gcn": GCNLayer, "mlp": DenseLayer, "sage": SAGELayer,
                                    "gin": GINLayer}[kind](m, k))

    def forward(self, X: torch.Tensor, ops: dict, return_hidden: bool = False) -> torch.Tensor:
        H = hidden = X
        for l, layer in enumerate(self.layers):
            H_in = H
            H = layer(dropout(H, self.dropout, self.training), ops)
            if l < len(self.layers) - 1:
                H = F.elu(H) if self.kind == "gat" else F.relu(H)
                if self.residual and H.shape == H_in.shape:
                    H = H + H_in                         # skip connection: keep the node's own signal
                hidden = H
        return (H, hidden) if return_hidden else H

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ------------------------------------------------------------------ graph-level prediction
def batch_graphs(graphs: list[tuple[np.ndarray, int]]) -> tuple[np.ndarray, int, torch.Tensor]:
    """Merge several graphs [(edges, n), ...] into one big graph with no edges between them.
    Its adjacency matrix is block-diagonal.  Returns (edges, total n, graph id of every node)."""
    edges, batch, offset = [], [], 0
    for i, (e, n) in enumerate(graphs):
        edges.append(np.asarray(e) + offset)             # shift the node numbers
        batch += [i] * n
        offset += n
    return np.concatenate(edges), offset, torch.tensor(batch)


def readout(H: torch.Tensor, batch: torch.Tensor, how: str = "sum") -> torch.Tensor:
    """One vector per graph: sum (or mean) of the node vectors that belong to it."""
    n_graphs = int(batch.max()) + 1
    out = torch.zeros(n_graphs, H.shape[1], dtype=H.dtype).index_add_(0, batch, H)
    if how == "mean":
        out = out / torch.bincount(batch, minlength=n_graphs).unsqueeze(1)
    return out


# ------------------------------------------------------------------ a baseline that uses no features
def label_propagation(ops: dict, y: torch.Tensor, train_mask: torch.Tensor, n_classes: int,
                      alpha: float = 0.9, iters: int = 50) -> torch.Tensor:
    """Label spreading [Zhou et al. 2004]: start from the known labels and repeatedly let every node
    take the normalised average of its neighbours' scores:  F <- alpha S F + (1 - alpha) Y,
    with S = D^(-1/2) A D^(-1/2).  No features, no trainable weights."""
    A = ops["A"]
    d = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1)
    i, j = A.indices()
    S = sparse_matrix(i, j, (d[i] * d[j]).rsqrt(), ops["n"])
    Y = torch.zeros(ops["n"], n_classes)
    Y[train_mask] = F.one_hot(y[train_mask], n_classes).float()
    scores = Y.clone()
    for _ in range(iters):
        scores = alpha * torch.sparse.mm(S, scores) + (1 - alpha) * Y
    return scores
