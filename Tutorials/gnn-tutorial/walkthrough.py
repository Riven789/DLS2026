"""
walkthrough.py -- the worked examples on the slides, computed step by step.

    python walkthrough.py

Slides number the nodes 1..6; the arrays below are printed in that order.
"""
import numpy as np
import torch

from data import adjacency, toy_graph
from gcn_numpy import gcn_layer, normalized_adjacency, softmax
from gnn import GNN, batch_graphs, graph_operators, readout

np.set_printoptions(precision=3, suppress=True, linewidth=140)


def header(t: str) -> None:
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


g = toy_graph()
n, edges, X = g["n"], g["edges"], g["X"].astype(float)
A = adjacency(n, edges)

# ---------------------------------------------------------------- 1. representing the graph
header("1. THE EXAMPLE GRAPH: EDGE LIST, A, D, X            (slide 3)")
print("edge list (1-based):", [(int(a) + 1, int(b) + 1) for a, b in edges])
print("A =\n", A.astype(int))
d = A.sum(1)
print("degrees d = A 1 =", d.astype(int), "   sum of degrees =", int(d.sum()), "= 2 |E| =", 2 * len(edges))
print("X (one row per node) =\n", X)
print("A symmetric:", bool((A == A.T).all()), "  trace(A) =", int(np.trace(A)), "(no self-loops)")
print("(A^2)[v,u] counts walks of length 2; diagonal of A^2 =", np.diag(A @ A).astype(int), "= degrees")

# ---------------------------------------------------------------- 2. aggregators
header("2. MESSAGE PASSING AT NODE 3: SUM, MEAN, MAX        (slide 7)")
v = 2
nbrs = np.flatnonzero(A[v])
print("neighbours of node 3:", (nbrs + 1).tolist(), " their features:", X[nbrs].tolist())
print("sum  =", X[nbrs].sum(0), "   mean =", X[nbrs].mean(0), "   max =", X[nbrs].max(0))
print("in matrix form: (A X)[3] =", (A @ X)[v], "   (D^-1 A X)[3] =", (A @ X)[v] / d[v])

# ---------------------------------------------------------------- 3. the GCN layer
header("3. ONE GCN LAYER BY HAND:  H' = ReLU(A_hat X W)     (slides 8 and 9)")
A_tilde = A + np.eye(n)
d_tilde = A_tilde.sum(1)
A_hat = normalized_adjacency(A)
print("A~ = A + I, degrees d~ =", d_tilde.astype(int))
print("A_hat = D~^-1/2 A~ D~^-1/2 =\n", A_hat)
print("entries: 1/3 =", round(1 / 3, 4), "  1/sqrt(12) =", round(1 / np.sqrt(12), 4), "  1/4 =", 0.25)
print("row sums of A_hat =", A_hat.sum(1), " (close to 1, not exactly 1)")
S = A_hat @ X
print("\nstep 1, aggregate:  S = A_hat X =\n", S)
print("node 3 in detail: x3/4 + x1/sqrt(12) + x2/sqrt(12) + x4/4 =",
      X[2] / 4 + X[0] / np.sqrt(12) + X[1] / np.sqrt(12) + X[3] / 4)
W = np.array([[1.0, -1.0], [-1.0, 1.5]])
print("\nstep 2, transform with the shared W =\n", W)
print("S W =\n", S @ W)
H1 = gcn_layer(A_hat, X, W, 0.0)
print("step 3, ReLU:  H' =\n", H1)
print("\nfor comparison, mean normalisation D~^-1 A~ X (plain average incl. self) =\n",
      normalized_adjacency(A, norm="mean") @ X)
print("and without any normalisation, A~ X (sums grow with the degree) =\n", A_tilde @ X)
lam = np.linalg.eigvalsh(A_hat)
print("eigenvalues of A_hat:", lam, " -> all in (-1, 1], largest exactly 1")
lam_raw = np.linalg.eigvalsh(A_tilde)
print("eigenvalues of A~ (no normalisation):", lam_raw, " -> largest", round(lam_raw.max(), 3), "> 1")

# ---------------------------------------------------------------- 4. permutation
header("4. RELABELLING THE NODES                            (slide 6)")
perm = np.array([3, 0, 5, 2, 1, 4])                    # new node i is old node perm[i]
P = np.eye(n)[perm]
A_p, X_p = P @ A @ P.T, P @ X
print("permutation (new order of the old nodes, 1-based):", (perm + 1).tolist())
H_p = gcn_layer(normalized_adjacency(A_p), X_p, W, 0.0)
print("GCN layer on the relabelled graph =\n", H_p)
print("max |GCN(PAP^T, PX) - P GCN(A, X)| =", np.abs(H_p - P @ H1).max(), " -> equivariant")
print("sum readout, original:", H1.sum(0), "  relabelled:", H_p.sum(0), " -> invariant")
rng = np.random.default_rng(0)
W_flat = rng.normal(size=(n * n + X.size,))            # an 'MLP' on the flattened [A, X]
f = lambda A_, X_: float(np.concatenate([A_.ravel(), X_.ravel()]) @ W_flat)
print("a linear layer on the flattened [A, X]: original %.3f, relabelled %.3f -> NOT invariant"
      % (f(A, X), f(A_p, X_p)))
print("number of orderings of 6 nodes: 6! =", 720, ";  of Cora's 2708 nodes: about 10^%d"
      % int(sum(np.log10(np.arange(1, 2709)))))

# ---------------------------------------------------------------- 5. receptive field
header("5. STACKING LAYERS: WHO CAN INFLUENCE NODE 1?       (slide 10)")
R = np.eye(n)
for k in range(0, 4):
    print(f"after {k} layers node 1 depends on the inputs of nodes", (np.flatnonzero(R[0]) + 1).tolist())
    R = ((A_tilde @ R) > 0).astype(float)
print("(A_hat^2)[1] =", np.linalg.matrix_power(A_hat, 2)[0])
print("(A_hat^3)[1] =", np.linalg.matrix_power(A_hat, 3)[0])

# ---------------------------------------------------------------- 6. over-smoothing
header("6. OVER-SMOOTHING: A_hat^k X FOR GROWING k          (slide 11)")
for k in (0, 1, 2, 4, 16, 64):
    Hk = np.linalg.matrix_power(A_hat, k) @ X
    print(f"k = {k:2d}: rows", " ".join(f"({a:.2f},{b:.2f})" for a, b in Hk))
u = np.sqrt(d_tilde) / np.linalg.norm(np.sqrt(d_tilde))
print("limit: u u^T X with u = sqrt(d~)/|sqrt(d~)| =", u)
print("u u^T X =\n", np.outer(u, u) @ X)
print("second-largest |eigenvalue| of A_hat:", round(np.sort(np.abs(lam))[-2], 4),
      " -> the differences shrink by this factor per layer")

# ---------------------------------------------------------------- 7. attention
header("7. GAT ATTENTION AT NODE 3 (W = I, a_dst = 0, a_src = (1, 0))   (slide 12)")
nb = [0, 1, 2, 3]
score = X[nb] @ np.array([1.0, 0.0])                   # e_3u = LeakyReLU(a_dst.h_3 + a_src.h_u), all >= 0 here
alpha = softmax(score[None])[0]
print("scores e_3u for u = 1, 2, 3, 4:", score, " -> alpha =", alpha, " (sum %.3f)" % alpha.sum())
print("attention output:", alpha @ X[nb], "   GCN weights for comparison:", A_hat[2, nb])

# ---------------------------------------------------------------- 8. WL test
header("8. WHAT MESSAGE PASSING CANNOT SEE: 1-WL            (slide 13)")
tri2 = np.array([[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]])      # two triangles
hexa = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 5]])      # one 6-cycle


def wl_colours(n_nodes: int, e: np.ndarray, rounds: int = 3) -> list[int]:
    """Weisfeiler-Lehman colour refinement: new colour = hash(own colour, multiset of neighbours')."""
    nbr = [[] for _ in range(n_nodes)]
    for a, b in e:
        nbr[a].append(b)
        nbr[b].append(a)
    colour = [0] * n_nodes
    for _ in range(rounds):
        sig = [(colour[v_], tuple(sorted(colour[u_] for u_ in nbr[v_]))) for v_ in range(n_nodes)]
        table = {s: i for i, s in enumerate(sorted(set(sig)))}
        colour = [table[s] for s in sig]
    return sorted(colour)


print("WL colours, two triangles:", wl_colours(6, tri2), "  hexagon:", wl_colours(6, hexa), " -> identical")
print("but triangles: trace(A^3)/6 =", int(np.trace(np.linalg.matrix_power(adjacency(6, tri2), 3)) / 6),
      "vs", int(np.trace(np.linalg.matrix_power(adjacency(6, hexa), 3)) / 6))
e_b, n_b, batch = batch_graphs([(tri2, 6), (hexa, 6)])
torch.manual_seed(0)
gin = GNN(1, 8, 4, n_layers=3, kind="gin", dropout=0.0)
with torch.no_grad():                                  # every node starts with the same feature, 1
    out = readout(gin(torch.ones(n_b, 1), graph_operators(e_b, n_b)), batch)
print("random 3-layer GIN, sum readout of each graph:\n", out.numpy())
print("max difference between the two graphs:", float((out[0] - out[1]).abs().max()))
print("the toy graph from slide 3:", wl_colours(6, edges), " (bridge nodes 3, 4 get their own colour)")

# ---------------------------------------------------------------- 9. batching
header("9. GRAPH-LEVEL PREDICTION: BATCH = BLOCK-DIAGONAL   (slide 14)")
g1 = (np.array([[0, 1], [1, 2], [0, 2]]), 3)           # triangle
g2 = (np.array([[0, 1]]), 2)                           # one edge
e_b, n_b, batch = batch_graphs([g1, g2])
print("batched edge list (0-based):", e_b.tolist(), "  graph id per node:", batch.tolist())
print("batched A (block-diagonal) =\n", adjacency(n_b, e_b).astype(int))
Xb = np.array([[1.0, 0], [0, 1], [1, 1], [2, 0], [0, 2]])
Hb = normalized_adjacency(adjacency(n_b, e_b)) @ Xb
print("A_hat X =\n", Hb)
Ht = torch.from_numpy(Hb)
print("sum readout per graph =\n", readout(Ht, batch, "sum").numpy())
print("mean readout per graph =\n", readout(Ht, batch, "mean").numpy())
alone = normalized_adjacency(adjacency(*g1[::-1])) @ Xb[:3]
print("triangle processed alone, sum readout:", alone.sum(0), " -> same as inside the batch")

# ---------------------------------------------------------------- 10. masked loss
header("10. THE LOSS USES LABELLED NODES ONLY               (slide 15)")
Z = np.array([[2.0, 0.0], [1.0, 1.0], [0.5, 1.5], [0.0, 1.0], [-1.0, 2.0], [0.0, 3.0]])
y = g["y"]
mask = np.array([True, False, False, False, False, True])
Pm = softmax(Z)
print("logits Z =\n", Z, "\nsoftmax P =\n", Pm)
print("labels y =", y, "  labelled (train) nodes: 1 and 6")
print("loss = -(ln P[1, y1] + ln P[6, y6]) / 2 = -(ln %.4f + ln %.4f) / 2 = %.4f"
      % (Pm[0, y[0]], Pm[5, y[5]], -np.log(Pm[mask, y[mask]]).mean()))
print("dL/dZ is zero in rows 2..5; they still get updated representations through the shared W")
