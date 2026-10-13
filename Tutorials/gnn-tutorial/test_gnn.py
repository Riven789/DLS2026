"""
test_gnn.py -- checks worth running before trusting any result.

    python test_gnn.py
"""
import numpy as np
import torch
import torch.nn.functional as F

from data import adjacency, karate_club, load_cora, make_split, planetoid_style_split, toy_graph
from gcn_numpy import NumpyGCN, gradient_check, normalized_adjacency
from gnn import GNN, batch_graphs, graph_operators, label_propagation, readout
from train import evaluate, fit, prepare


def test_backprop_matches_finite_differences():
    g = karate_club()
    A_hat = normalized_adjacency(adjacency(g["n"], g["edges"]))
    mask = np.zeros(34, dtype=bool)
    mask[[0, 5, 20, 33]] = True
    assert gradient_check(NumpyGCN([34, 5, 2], seed=1), A_hat, g["X"], g["y"], mask) < 1e-8
    assert gradient_check(NumpyGCN([34, 4, 3, 2], seed=2), A_hat, g["X"], g["y"], mask) < 1e-8


def test_numpy_and_pytorch_agree():
    """Same weights, same graph -> same normalised adjacency, loss and gradients in both codes."""
    g = karate_club()
    A_hat = normalized_adjacency(adjacency(g["n"], g["edges"]))
    ops = graph_operators(g["edges"], g["n"])
    assert np.allclose(ops["A_hat"].to_dense().numpy(), A_hat, atol=1e-6)
    assert np.allclose(ops["A_mean"].to_dense().sum(1).numpy(), 1, atol=1e-6)      # rows average
    net = NumpyGCN([34, 6, 2], seed=3)
    model = GNN(34, 6, 2, n_layers=2, kind="gcn", dropout=0.0).double()
    ops["A_hat"] = torch.from_numpy(A_hat).to_sparse()                 # float64 for a tight comparison
    with torch.no_grad():
        for layer, W, b in zip(model.layers, net.W, net.b):
            layer.lin.weight.copy_(torch.from_numpy(W.T))          # PyTorch stores W as (out, in)
            layer.bias.copy_(torch.from_numpy(b))
    mask = np.zeros(34, dtype=bool)
    mask[[0, 33]] = True
    logits = model(torch.from_numpy(g["X"]).double(), ops)
    loss = F.cross_entropy(logits[torch.from_numpy(mask)], torch.from_numpy(g["y"])[torch.from_numpy(mask)])
    loss.backward()
    assert abs(loss.item() - net.loss(A_hat, g["X"].astype(float), g["y"], mask)) < 1e-10
    gW, gb = net.backward(g["y"], mask)
    for layer, g_w, g_b in zip(model.layers, gW, gb):
        assert np.allclose(layer.lin.weight.grad.numpy().T, g_w, atol=1e-10)
        assert np.allclose(layer.bias.grad.numpy(), g_b, atol=1e-10)


def test_permutation_equivariance_and_attention():
    """Relabelling the nodes permutes the outputs of every layer type in the same way."""
    g = toy_graph()
    perm = np.array([3, 0, 5, 2, 1, 4])                  # new node i is old node perm[i]
    inv = np.argsort(perm)
    X, X_p, edges_p = torch.from_numpy(g["X"]), torch.from_numpy(g["X"][perm]), inv[g["edges"]]
    ops, ops_p = graph_operators(g["edges"], 6), graph_operators(edges_p, 6)
    for kind in ("gcn", "sage", "gat", "gin"):
        torch.manual_seed(0)
        model = GNN(2, 8, 3, n_layers=2, kind=kind, dropout=0.0, heads=2).eval()
        out, out_p = model(X, ops), model(X_p, ops_p)
        assert torch.allclose(out_p, out[perm], atol=1e-5), kind
        assert torch.allclose(out_p.sum(0), out.sum(0), atol=1e-5), kind      # readout: invariant
    k = karate_club()                                     # GAT: a node's attention weights sum to 1
    ops = graph_operators(k["edges"], 34)
    torch.manual_seed(0)
    layer = GNN(34, 4, 2, kind="gat", heads=3, dropout=0.0).layers[0]
    alpha = layer.attention(layer.lin(torch.from_numpy(k["X"])).view(34, 3, 4), ops)
    assert torch.allclose(torch.zeros(34, 3).index_add_(0, ops["edge_index"][1], alpha), torch.ones(34, 3), atol=1e-5)


def test_batching_and_readout():
    """A graph gives the same result alone and inside a batch; WL-equivalent graphs collide."""
    tri = (np.array([[0, 1], [1, 2], [0, 2]]), 3)
    path = (np.array([[0, 1], [1, 2], [2, 3]]), 4)
    edges, n, batch = batch_graphs([tri, path])
    assert batch.tolist() == [0, 0, 0, 1, 1, 1, 1]
    A = adjacency(n, edges)
    assert A[:3, 3:].sum() == 0 and A[3:, :3].sum() == 0              # block-diagonal
    torch.manual_seed(0)
    X = torch.randn(n, 5)
    model = GNN(5, 8, 4, n_layers=2, kind="gin", dropout=0.0).eval()
    together = readout(model(X, graph_operators(edges, n)), batch)
    alone = model(X[:3], graph_operators(tri[0], 3)).sum(0)
    assert torch.allclose(together[0], alone, atol=1e-5)
    two_triangles = np.array([[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]])
    hexagon = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 5]])
    edges, n, batch = batch_graphs([(two_triangles, 6), (hexagon, 6)])
    out = readout(model(torch.ones(n, 5), graph_operators(edges, n)), batch)
    assert torch.allclose(out[0], out[1], atol=1e-5)                  # the 1-WL limit


def test_data_and_masks():
    g = load_cora()
    assert (g["n"], len(g["edges"]), g["X"].shape[1], len(g["classes"])) == (2708, 5278, 1433, 7)
    assert (g["edges"][:, 0] < g["edges"][:, 1]).all() and len(np.unique(g["edges"], axis=0)) == 5278
    assert np.allclose(g["X"].sum(1), 1, atol=1e-5)                   # row-normalised
    m = planetoid_style_split(g["y"], seed=0)
    assert (m["train"].sum(), m["val"].sum(), m["test"].sum()) == (140, 500, 1000)
    assert not (m["train"] & m["val"]).any() and not (m["train"] & m["test"]).any()
    assert not (m["val"] & m["test"]).any()
    assert (np.bincount(g["y"][m["train"]]) == 20).all()
    assert (planetoid_style_split(g["y"], 1)["test"] != m["test"]).any()   # the seed matters
    k = karate_club()
    assert len(k["edges"]) == 78 and np.bincount(k["y"]).tolist() == [17, 17]
    assert np.bincount(k["edges"].ravel())[[0, 33]].tolist() == [16, 17]
    try:                                                              # optional cross-check
        import networkx as nx
        G = nx.karate_club_graph()
        assert sorted(map(tuple, k["edges"].tolist())) == sorted(G.edges())
        assert [int(G.nodes[i]["club"] != "Mr. Hi") for i in range(34)] == k["y"].tolist()
    except ImportError:
        pass


def test_test_labels_do_not_leak():
    """Changing the labels of validation/test/unlabelled nodes must not change training at all."""
    g = karate_club()
    outs = []
    for flip in (False, True):
        d = prepare(g, seed=0)
        if flip:
            d["y"] = torch.where(d["masks"]["train"], d["y"], 1 - d["y"])
        torch.manual_seed(0)
        model = GNN(34, 8, 2, dropout=0.0)
        fit(model, d, epochs=30, patience=0)
        outs.append(model(d["X"], d["ops"]))
    assert torch.equal(outs[0], outs[1])
    scores = [label_propagation(d["ops"], y, d["masks"]["train"], 2) for y in (d["y"], 1 - d["y"])]
    assert torch.allclose(scores[0], scores[1].flip(1))               # only the 2 train labels matter


def test_can_fit_the_labelled_nodes():
    d = prepare(karate_club(), seed=0)
    d["masks"]["train"] = torch.ones(34, dtype=torch.bool)            # label everything: must reach 100%
    torch.manual_seed(0)
    model = GNN(34, 16, 2, dropout=0.0)
    fit(model, d, lr=0.05, weight_decay=0.0, epochs=300, patience=0)
    assert evaluate(model, d, "train")["accuracy"] == 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok   {name}")
