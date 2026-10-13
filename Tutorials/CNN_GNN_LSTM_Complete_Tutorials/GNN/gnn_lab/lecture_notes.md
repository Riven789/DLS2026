# GNN tutorial notes

## 1. Graph neural networks

A graph neural network learns from entities and their relationships. Nodes hold feature vectors and edges determine which nodes exchange information. Unlike a CNN, a GNN does not require a regular grid or a fixed number of neighbors. Unlike a dense MLP, it encodes a deliberate connectivity pattern rather than treating every input position as unrelated.

The laboratory implements two GraphSAGE-style mean aggregation layers directly in NumPy. It uses separate self and neighbor weight matrices, a nonlinearity, masked graph pooling and a prediction head. It does not implement neighbor sampling or the original GraphSAGE normalization step, so it is explicitly a simplified teaching variant. Classification and regression tasks ask about the fraction of edges connecting nodes with the same Boolean attribute. Small dense adjacency arrays make every operation inspectable. They are not intended as a scalable graph engine.

References
Hamilton, Ying and Leskovec (2017). Inductive Representation Learning on Large Graphs. https://arxiv.org/abs/1706.02216
Gilmer et al. (2017). Neural Message Passing for Quantum Chemistry. https://arxiv.org/abs/1704.01212

## 2. A graph needs structure and features

Write a graph as G=(V,E), where V is its node set and E is its edge set. An adjacency matrix A has one row and column per node. In this tutorial A[v,u] describes a contribution from node u to node v. An undirected graph has a symmetric adjacency matrix. X has one feature-vector row per node, aligned to the same node ordering.

The drawing and matrix describe a four-node example. Edge presence is not a label unless the task explicitly predicts edges. A node can have numerical measurements, logical flags, categories encoded as vectors, or learned embeddings. Node IDs are bookkeeping, not automatically meaningful numerical features. Reordering the node list requires reordering both axes of A and the rows of X together. Edge lists are usually more efficient for sparse graphs, while the small dense matrices here make multiplication and differentiation easy to follow.

References
Hamilton, Ying and Leskovec (2017). Inductive Representation Learning on Large Graphs. https://arxiv.org/abs/1706.02216
Gilmer et al. (2017). Neural Message Passing for Quantum Chemistry. https://arxiv.org/abs/1704.01212

## 3. Prediction can target nodes, edges, or graphs

A node task predicts one value or class for each selected node, such as a type or a local property. The network may compute embeddings for many nodes but the loss should include only labeled training nodes. An edge task predicts a relationship or edge property using endpoint embeddings and possibly edge features. A graph task combines all node embeddings into one representation and predicts a whole-graph label.

These distinctions affect batching and leakage. A graph-level task needs graph-level splits. A link predictor must not include held-out target edges in its message-passing adjacency if that would reveal the answer. Node classification on one known graph may legitimately use the unlabeled graph structure under a declared transductive protocol. The included harness implements graph-level classification and scalar graph regression on independent generated graphs. Node and edge tasks are explained conceptually rather than silently claimed as executable CLI modes.

References
Hamilton, Ying and Leskovec (2017). Inductive Representation Learning on Large Graphs. https://arxiv.org/abs/1706.02216
Gilmer et al. (2017). Neural Message Passing for Quantum Chemistry. https://arxiv.org/abs/1704.01212

## 4. Message passing updates each node

Let h_v be the hidden feature vector at node v. A message function can use the source node, destination node and edge attributes. An aggregation operation combines messages over the neighborhood. The update function then produces the next hidden vector. Weights are shared over nodes and graphs, which lets the same learned rule apply to previously unseen graphs with compatible features.

Aggregation must handle a varying number of neighbors and avoid depending on the arbitrary order in which neighbors are listed. Sum, mean and maximum are common choices. They preserve different information: a mean loses the number of repeated identical messages unless degree is provided separately, while a sum changes with neighborhood size. Our example uses the mean and a separate self transformation. It defines an empty-neighborhood mean as zero, so isolated nodes still have a self update. Message passing is differentiable feature computation, not exchange of natural-language messages.

References
Hamilton, Ying and Leskovec (2017). Inductive Representation Learning on Large Graphs. https://arxiv.org/abs/1706.02216
Gilmer et al. (2017). Neural Message Passing for Quantum Chemistry. https://arxiv.org/abs/1704.01212

## 5. A neighbor update calculated by hand

Consider a center node with value 2 and two neighbors with values 1 and 3. Their mean is 2. With self weight 0.5, neighbor weight 1.5 and bias −1, the preactivation is 2×0.5+2×1.5−1=3. ReLU returns 3. Each node applies this same rule using its own current value and its own neighbors.

This update depends on attributes and topology together. Moving an edge changes which features appear in the mean even when the set of node features is unchanged. The new value is not an average alone: the model learns how much to use self and neighbor information, potentially with different signs and different output channels. In a full layer, every node uses the previous layer’s values. Do not update nodes in place one after another, because that would make the result depend on node traversal order. The matrix implementation updates all nodes simultaneously.

References
Hamilton, Ying and Leskovec (2017). Inductive Representation Learning on Large Graphs. https://arxiv.org/abs/1706.02216
Gilmer et al. (2017). Neural Message Passing for Quantum Chemistry. https://arxiv.org/abs/1704.01212

## 6. Mean aggregation as matrix multiplication

For a nonnegative adjacency matrix, let d_v be the sum of the entries in row v. The row-normalized matrix M has entries A[v,u]/d_v when d_v is positive and zeros when the node is isolated. Multiplying M by H computes a weighted neighbor mean. For binary adjacency, this is the ordinary neighbor mean. The code accepts symmetric nonnegative weights while the generated data use zeros and ones.

The layer computes Z=H Ws + (M H) Wn + b, followed by ReLU and a validity mask. It excludes self loops in A because self information already has a separate learned transformation. Two matrices are equivalent to applying one matrix to concatenated self and neighbor features, with the matrix split into blocks. This resembles the mean-aggregation family of GraphSAGE. The lab does not sample neighborhoods or normalize hidden vectors to unit norm. Those differences are explicit choices for a small, exact teaching implementation.

References
Hamilton, Ying and Leskovec (2017). Inductive Representation Learning on Large Graphs. https://arxiv.org/abs/1706.02216
Gilmer et al. (2017). Neural Message Passing for Quantum Chemistry. https://arxiv.org/abs/1704.01212

## 7. GCN, GraphSAGE, and graph attention

The GCN rule associated with Kipf and Welling adds self loops and uses symmetric degree normalization: H′=σ(Dtilde^−1/2 Atilde Dtilde^−1/2 H W), with Atilde=A+I. That rule differs from a separate self matrix plus a row-normalized neighbor mean. Similar-looking matrix formulas should not be treated as interchangeable without examining the definitions.

GraphSAGE learns an aggregation-based function that can apply to unseen nodes and graphs; its original framework includes neighbor sampling and several possible aggregators. Graph attention learns coefficients over a node’s allowed neighborhood and normalizes them locally. In each case, architecture and preprocessing determine which graph information reaches the output. The table is a comparison, not a claim that the lab implements all three. We implement one small mean-aggregation variant and provide references for the others. Attention weights alone are not a validated causal explanation.

References
Kipf and Welling (2017). Semi-Supervised Classification with Graph Convolutional Networks. https://arxiv.org/abs/1609.02907
Hamilton, Ying and Leskovec (2017). Inductive Representation Learning on Large Graphs. https://arxiv.org/abs/1706.02216
Velickovic et al. (2018). Graph Attention Networks. https://arxiv.org/abs/1710.10903

## 8. Node order must not change graph meaning

A permutation matrix P represents a reordering of nodes. The corresponding inputs are X′=PX and A′=PAPᵀ. A node-level equivariant model satisfies f(X′,A′)=P f(X,A): the embeddings move to the matching new positions. A graph-level invariant model combines those embeddings with an order-independent readout so that its final prediction is unchanged.

Mean neighbor aggregation with shared weights has the required symmetry. The row normalization transforms consistently under reordering, and elementwise nonlinearities commute with the permutation. A sum or mean over valid nodes is invariant. The lab tests this by shuffling every graph’s node ordering, permuting both axes of adjacency, and comparing the logits. Reordering only X is a different graph representation and should change the result. Learned node IDs or position-specific weights can break the property unless their role is deliberately defined.

References
Hamilton, Ying and Leskovec (2017). Inductive Representation Learning on Large Graphs. https://arxiv.org/abs/1706.02216
Gilmer et al. (2017). Neural Message Passing for Quantum Chemistry. https://arxiv.org/abs/1704.01212

## 9. Depth controls the neighborhood reach

After one layer, a node’s representation depends on itself and its direct neighbors. After two layers it can depend on two-hop neighbors because its neighbors have already incorporated their own neighborhoods. In a local message-passing stack of depth L, the receptive field extends at most L edges, unless extra global operations provide longer-range information.

Increasing depth is not automatically an improvement. Repeated aggregation can make embeddings too similar, often called oversmoothing. Large neighborhoods can also force many distinct signals through a fixed-width vector, sometimes called oversquashing. Residual connections, normalization, appropriate neighborhood definitions or alternative architectures may help, depending on the task. The lab uses two layers and global graph pooling. Pooling sees all final node representations, but it does not grant each individual node unlimited relational reasoning. When debugging, distinguish the node’s local reach from the graph readout’s access to all nodes.

References
Hamilton, Ying and Leskovec (2017). Inductive Representation Learning on Large Graphs. https://arxiv.org/abs/1706.02216
Gilmer et al. (2017). Neural Message Passing for Quantum Chemistry. https://arxiv.org/abs/1704.01212

## 10. Graph pooling and validity masks

Different graphs have different numbers of nodes. A graph-level prediction requires a representation with fixed width. Mean pooling sums the valid node embeddings and divides by the number of valid nodes. Sum pooling preserves a dependence on graph size, while max pooling keeps the largest feature value. The appropriate readout depends on whether the target is intensive, extensive or otherwise size-dependent.

The lab stores several small graphs in padded arrays and supplies a zero/one mask. It multiplies hidden activations by that mask after each layer and divides by the actual valid-node count. Padded nodes must not have edges. A learned bias would otherwise create nonzero embeddings at padded positions, which is why masking is part of the computation rather than an optional cleanup step. The tests verify that adding more padding does not change predictions. A graph with no valid nodes is rejected rather than assigned an arbitrary mean.

References
Hamilton, Ying and Leskovec (2017). Inductive Representation Learning on Large Graphs. https://arxiv.org/abs/1706.02216
Gilmer et al. (2017). Neural Message Passing for Quantum Chemistry. https://arxiv.org/abs/1704.01212

## 11. Batching graphs

The lab uses X with shape B,Nmax,F, A with shape B,Nmax,Nmax, and a mask with shape B,Nmax. Matrix multiplication operates independently on each graph, so no message passes from one example to another. This representation is straightforward but stores many zeros and requires memory proportional to B times Nmax squared for adjacency.

A common scalable alternative concatenates node arrays, offsets each graph’s edge indices and keeps a graph-membership vector for pooling. The combined graph is a disjoint union: there are no edges across examples. Sparse message-passing kernels then work in terms of actual edges, though feature transformations and aggregation cost also depend on hidden width. Neighbor sampling can reduce the cost of very large neighborhoods but introduces another source of approximation and randomness. Those options are described, not implemented by this NumPy lab. Batch size here means graphs per update, while actual work also depends on nodes and edges.

References
Hamilton, Ying and Leskovec (2017). Inductive Representation Learning on Large Graphs. https://arxiv.org/abs/1706.02216
Gilmer et al. (2017). Neural Message Passing for Quantum Chemistry. https://arxiv.org/abs/1704.01212

## 12. Node features, edge features, and labels

Numerical node measurements often need scaling with statistics fitted only on training nodes from training graphs. Logical attributes use explicit zero/one values. Categories can use one-hot vectors or embeddings, with a defined treatment of missing or unseen values. A graph model cannot infer whether a number is a physical quantity, an arbitrary ID, or a category code without an appropriate representation.

The included feature schema is deliberately fixed: column zero is a Boolean flag and column one is a numerical nuisance feature. The numerical column is standardized over valid training nodes; the Boolean stays zero/one. Missing values are rejected and should be handled upstream with a documented imputation and indicator policy. Generated edges are undirected and binary. Edge types, lengths, bond orders or directions require an edge-aware message function; this specific model has no separate edge-feature tensor. For geometry-dependent tasks, rotation/translation symmetries require additional architectural choices beyond invariance to node ordering.

References
Hamilton, Ying and Leskovec (2017). Inductive Representation Learning on Large Graphs. https://arxiv.org/abs/1706.02216
Gilmer et al. (2017). Neural Message Passing for Quantum Chemistry. https://arxiv.org/abs/1704.01212

## 13. Graph evaluation and leakage

Inductive graph evaluation holds out complete graphs or newly arriving nodes under a stated information policy. The lab uses independent whole graphs and holds out entire graph examples. A transductive node task may expose the full known adjacency and unlabeled node features while withholding their labels; this is a different protocol, not automatically a mistake. The report must say which protocol applies.

Link prediction needs special care. If a held-out positive edge is still present in the adjacency used to compute embeddings, the model may receive the answer through message passing. Negative sampling must also reflect the actual candidate-pair problem. Time-dependent networks should not include future relationships or features when forecasting the past. Molecules or materials with related scaffolds or structures may need group-aware splits. For custom lab data, an optional split array allows explicit train/validation/test assignments, but the harness does not infer scientific groups or temporal policies.

References
Kipf and Welling (2017). Semi-Supervised Classification with Graph Convolutional Networks. https://arxiv.org/abs/1609.02907
Hamilton, Ying and Leskovec (2017). Inductive Representation Learning on Large Graphs. https://arxiv.org/abs/1706.02216
scikit-learn. Common pitfalls and recommended practices. https://scikit-learn.org/stable/common_pitfalls.html

## 14. Graph classification and regression

The generated graphs contain two kinds of nodes distinguished by a Boolean flag. The classification label states whether more than half of the actual edges connect matching flags. The regression target is that actual fraction. Labels are computed from each complete generated graph rather than assigned from an unobserved generation parameter. A nuisance numerical feature tests whether the model can learn to focus on useful graph information.

Classification uses two logits, softmax and cross entropy. Regression uses one linear output and mean squared error with training-only target scaling. The linear regressor is not constrained to the interval zero to one; a bounded head could be an extension. Because the labels are deliberately simple functions of the supplied graph, an exact hand-written statistic solves the task perfectly. The purpose is to verify learning and message-passing behavior, not to argue that a GNN is necessary or beats a strong task-specific baseline.

References
Hamilton, Ying and Leskovec (2017). Inductive Representation Learning on Large Graphs. https://arxiv.org/abs/1706.02216
Gilmer et al. (2017). Neural Message Passing for Quantum Chemistry. https://arxiv.org/abs/1704.01212

## 15. Gradients flow through aggregation

Let N=MH be the neighbor-feature matrix and Z=H Ws+N Wn+b. Given Δ=∂L/∂Z, the self-weight derivative is HᵀΔ, while the neighbor-weight derivative is NᵀΔ. The bias derivative sums Δ over nodes and graphs. The derivative with respect to H has both a direct self contribution and an aggregation contribution Mᵀ(Δ Wnᵀ). In the batch implementation, einsum sums over batch and node axes.

ReLU masks the incoming derivative and graph pooling distributes it to valid nodes with the appropriate denominator. The adjacency and row-normalization are fixed input data, so this model does not differentiate with respect to edges or learn graph construction. A different model could learn edge weights or attention, requiring their derivatives too. Finite-difference tests sample entries of every parameter tensor under both tasks, while permutation and padding tests check the structural behavior that gradient tests alone would not establish.

References
Hamilton, Ying and Leskovec (2017). Inductive Representation Learning on Large Graphs. https://arxiv.org/abs/1706.02216
Gilmer et al. (2017). Neural Message Passing for Quantum Chemistry. https://arxiv.org/abs/1704.01212

## 16. Message passing in executable NumPy

The neighbor_matrix function divides each adjacency row by its degree, with an explicit zero-row result for isolated nodes. The forward pass multiplies that matrix by the current node features and combines neighbor and self transformations. A mask removes padded activations after each nonlinearity. Two layers feed a masked graph mean and a linear prediction head.

This excerpt omits dictionary lookups and caching so the mathematical sequence is easy to see. The complete model.py implements the same sequence and its backward pass. There are 642 trainable parameters for two-class classification: 80 in the first two-to-sixteen layer, 528 in the second sixteen-to-sixteen layer and 34 in the output head. Scalar regression uses 625 parameters. The architecture accepts different node counts under the same feature schema. It does not support arbitrary new feature columns without rebuilding and training a compatible model.

References
Hamilton, Ying and Leskovec (2017). Inductive Representation Learning on Large Graphs. https://arxiv.org/abs/1706.02216
Gilmer et al. (2017). Neural Message Passing for Quantum Chemistry. https://arxiv.org/abs/1704.01212

## 17. A reproducible graph-learning harness

The one-command runner executes seven tests, generates data, trains classification and regression models, evaluates saved test splits and predicts unlabeled graphs. Default settings are AdamW with learning rate 0.003, weight decay 0.0001, batch size 32 graphs, at most eighty epochs and patience fifteen. Global gradient clipping has a norm cap of five. Each run uses a fresh output directory.

Training scores the validation graphs at each epoch and restores the checkpoint with the lowest validation loss. metadata.json contains the masks’ expected schema, preprocessing statistics, task, architecture, split indices, source-data hash and software versions. evaluate refuses a changed data archive because its row indices identify the original graph examples. predict accepts new graphs with compatible feature meaning. The saved numerical arrays do not use Python pickle. Exact training resumption is outside scope because optimizer state and random-generator state are not saved.

The training loop shuffles independent graphs into minibatches of 32, computes the loss and full weight gradients, then updates with AdamW at learning rate 0.003. AdamW tracks bias-corrected first and second gradient moments and applies decoupled weight decay 0.0001 to weight matrices. The learning rate scales each update. Too large a rate can destabilize loss and too small a rate slows fitting. Global gradient-norm clipping at 5 caps large updates. Validation chooses the best epoch, with patience 15 and a cap of 80 epochs. Test data never choose optimizer settings. Biases are excluded from decay. These controls live in common.py and the CLI.

References
Hamilton, Ying and Leskovec (2017). Inductive Representation Learning on Large Graphs. https://arxiv.org/abs/1706.02216
Gilmer et al. (2017). Neural Message Passing for Quantum Chemistry. https://arxiv.org/abs/1704.01212

## 18. Measured results on held-out graphs

The displayed metrics and learning curves come from the included reference runs. Data generation uses seed 42; the split and model use seed 7. Each task contains 1,400 graphs with eight to sixteen valid nodes, two node features and undirected edges. Classification predicts whether the observed matching-flag edge fraction exceeds one half. Regression predicts the fraction directly.

The majority-class and training-mean baselines provide simple sanity checks. An exact algorithm that counts matching-flag edges would achieve perfect results on this constructed task, so these results demonstrate the network and harness rather than the necessity of machine learning. The independent held-out graphs share the generator’s distribution. They do not evaluate extrapolation to larger graphs or a different scientific domain. Full predictions, confusion matrices, selected epochs and software versions accompany the lab. The test suite also checks that reordering nodes and adding padding preserve graph predictions.

References


## 19. Limits, diagnostics, and scaling

Some different graphs cannot be distinguished by a particular message-passing architecture and feature set. For example, regular graphs with identical node features can remain indistinguishable under simple mean aggregation, even when their global structures differ. Additional depth alone does not guarantee recovery of the missing distinction. Better features, positional information, more expressive aggregation or other graph models may be needed.

Oversmoothing concerns representations becoming too similar, while dense adjacency storage can become infeasible long before the mathematical model runs out of parameters. Compare against simple feature-only and structural baselines to determine whether edges actually help. Inspect node-count and degree shifts between training and deployment. The seven tests cover gradients, checkpoint loading, splits, stable loss, training-only statistics, isolated nodes, invalid padding edges, permutation invariance and padding invariance. Those implementation checks do not establish that the graph construction or labels answer the intended scientific question.

References
Hamilton, Ying and Leskovec (2017). Inductive Representation Learning on Large Graphs. https://arxiv.org/abs/1706.02216
Gilmer et al. (2017). Neural Message Passing for Quantum Chemistry. https://arxiv.org/abs/1704.01212

## 20. Laboratory and further reading

Start with the exact neighbor calculation, then run the laboratory and inspect a generated graph with its node flags and label. Verify by hand that the target fraction matches the actual edges. Trace a two-layer forward pass and confirm the classification parameter count. Use the permutation test as a template whenever modifying aggregation or readout.

Exercises include replacing mean with sum pooling, adding an edge feature to the message function, comparing against a feature-only network and an exact structural statistic, and evaluating larger graphs from a deliberately shifted generator. Add sparse disjoint batching only after the dense version and its gradients are understood. Keep validation-based model selection and separate test evaluation intact. The references describe broader message-passing frameworks, GCN, GraphSAGE and graph attention; they are distinct methods with different equations and implementation choices.

References
Gilmer et al. (2017). Neural Message Passing for Quantum Chemistry. https://arxiv.org/abs/1704.01212
Kipf and Welling (2017). Semi-Supervised Classification with Graph Convolutional Networks. https://arxiv.org/abs/1609.02907
Hamilton, Ying and Leskovec (2017). Inductive Representation Learning on Large Graphs. https://arxiv.org/abs/1706.02216
Velickovic et al. (2018). Graph Attention Networks. https://arxiv.org/abs/1710.10903