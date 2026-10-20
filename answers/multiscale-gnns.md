# Multiscale GNNs: Oversmoothing, Oversquashing and Graph U-Nets

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A message-passing GNN moves information one hop per layer. On a simulation mesh that causes three problems. **Under-reaching:** information needs as many layers as there are hops between two points, and refining the mesh multiplies the hops while the physical distance stays the same. **Oversmoothing:** stacking many averaging layers makes neighbouring node features converge, which erases sharp gradients and, in the limit, all spatial detail. **Oversquashing:** information from an exponentially growing (or bottlenecked) neighbourhood has to squeeze through fixed-size vectors, so distant influences get lost. Multiscale GNNs fix the reach problem by passing messages on a hierarchy of coarser graphs as well as the fine one: MultiScale MeshGraphNets (two mesh levels), BSMS-GNN (bi-stride pooling), Graph U-Nets (learned top-k pooling) and X-MeshGraphNet (multiscale point-cloud graphs). A long-range edge on a coarse level replaces many short hops on the fine level.

**On this page**

- [Oversmoothing and oversquashing](#oversmoothing-and-oversquashing)
- [Why mesh refinement makes it worse](#why-mesh-refinement-makes-it-worse)
- [Multiscale message passing](#multiscale-message-passing)
- [Graph U-Net](#graph-u-net)
- [Code: smoothing, hop counts and bi-stride pooling](#code-smoothing-hop-counts-and-bi-stride-pooling)
- [Code: gPool and gUnpool](#code-gpool-and-gunpool)
- [In an interview](#in-an-interview)

## Oversmoothing and oversquashing

**Oversmoothing.** Li, Han and Wu (AAAI 2018) showed that the graph convolution in a GCN is a special form of **Laplacian smoothing**: each layer replaces a node's feature by a weighted average of itself and its neighbours. That is why GCNs work (neighbouring nodes in a citation graph tend to share labels), but repeated many times it makes all node features in a connected region converge to the same value. The analogy for engineers: repeatedly applying one Jacobi smoothing sweep of the heat equation. High-frequency content disappears first, then everything else.

For physics this matters because the solution often has **sharp, local features**: a stress concentration at a fillet, a shock, a thin boundary layer. A deep stack of plain averaging layers blurs them. The demo [below](#code-smoothing-hop-counts-and-bi-stride-pooling) shows that four averaging layers already remove more than 99% of the fine-detail energy of random features on a mesh.

**What limits it in practice:** learned edge and node MLPs (messages are not plain averages), residual connections (h ← h + update), normalization, encode-process-decode designs such as MeshGraphNets, and fewer, more expressive layers on coarser graphs.

**Oversquashing.** Alon and Yahav (ICLR 2021) explained why GNNs struggle with long-range interactions: when a node aggregates messages along long paths, the number of nodes in its receptive field grows quickly with depth, and all of that information has to be compressed into one fixed-size vector. The result is a **bottleneck** that over-squashes exponentially growing information. They found that GNNs which absorb incoming edges equally, such as GCN and GIN, are more susceptible than attention-based GAT and gated GGNN, and that simply adding a fully adjacent layer (every node connected to every other) at the end improved prior long-range results without tuning. Topping et al. (ICLR 2022) made this precise with an edge-based **graph curvature**: negatively curved edges, the bridges between otherwise separate regions, are responsible for oversquashing, and they proposed **rewiring** the graph around them.

**On meshes specifically:** a 2D mesh's k-hop neighbourhood grows like k² (and k³ in 3D), not exponentially as in a tree, so classic oversquashing is milder than on social or molecular graphs. It does appear at **geometric bottlenecks**: a thin rib joining two plates, a narrow gap between a wheel and a wheel arch, a cooling channel. The more common mesh problem is under-reaching.

| Problem | What you observe | Typical fixes |
| --- | --- | --- |
| Under-reaching | A local disturbance never affects distant nodes; error grows with mesh resolution | More layers, coarse levels, global tokens or attention |
| Oversmoothing | Fields look blurred, peaks are underestimated, deeper models get worse | Residuals, normalization, learned messages, fewer layers on coarser graphs |
| Oversquashing | Long-range dependencies fail even with enough layers, especially across bottlenecks | Rewiring, extra long-range edges, attention or a global (fully connected) layer |

## Why mesh refinement makes it worse

Incompressible flow and linear elasticity are **elliptic**: a change anywhere affects the solution everywhere (pressure is the obvious example). A message-passing GNN with L layers only lets a node see L hops. Fortunato et al. (MultiScale MeshGraphNets, 2022) put the core issue this way: at higher resolution, points that are the same physical distance apart become further apart in graph space, so a fixed number of message-passing steps covers a smaller physical region.

On a structured n × n mesh of the unit square, crossing from corner to corner takes 2(n − 1) hops: 30 hops at 16 × 16, 254 at 128 × 128. Doubling the resolution doubles the number of layers needed, and the cost per layer grows with the node count too. Fifteen message-passing steps, a typical MeshGraphNets setting, cover only a small fraction of a fine mesh.

## Multiscale message passing

The common idea: build a hierarchy of graphs, pass messages on each level, and move features between levels.

1. **Two mesh levels (MultiScale MeshGraphNets, Fortunato et al. 2022).** Two contributions: learning high-resolution dynamics on a coarser mesh, and a hierarchical scheme that passes messages on a fine and a coarse mesh in alternation. The paper reports better accuracy with less computation. The coarse mesh has to be generated separately, which takes manual effort.
2. **Bi-stride pooling (BSMS-GNN, Cao et al., ICML 2023).** Pick a seed node, run a breadth-first search, and keep the nodes on **every other BFS frontier**. Connect kept nodes that were at most two hops apart. Repeating this roughly halves the nodes and the graph diameter at each level, needs no hand-drawn coarse meshes, and avoids the wrong edges that spatial-proximity coarsening can create across thin geometry (two surfaces close in space but not connected through the material). It uses one message-passing step per level and non-parametric pooling and unpooling by interpolation, like a U-Net.
3. **Clustering or voxel coarsening.** Group nodes into spatial cells or clusters (for example with farthest point sampling or a voxel grid), average features to cluster centres, and interpolate back. Simple and fast, but watch for edges that jump across thin walls.
4. **Multiscale point-cloud graphs (X-MeshGraphNet, Nabian et al., NVIDIA 2024).** Build graphs directly from points sampled on the geometry (k-nearest-neighbour graphs from STL files, so no simulation mesh is needed at inference), combine coarse and fine point clouds into a multiscale graph, and split very large graphs into partitions with **halo regions** so training is equivalent to full-graph training. See [Scaling a GNN to 10 million nodes](scaling-to-large-meshes.md#scaling-a-gnn-to-10-million-nodes).
5. **Learned pooling (Graph U-Nets).** Let the network choose which nodes to keep. See the next section.

**Global alternatives.** Attention over a small set of tokens (Transolver slices, UPT latent tokens, AB-UPT anchors) gives every point a global receptive field in one layer. See [Transformers for physics](transformers-for-physics.md#gnn-or-transformer).

## Graph U-Net

Gao and Ji (ICML 2019) built a U-Net for graphs from two new operations.

**gPool (graph pooling).** Score every node by projecting its feature vector onto a **trainable vector p**, y = Xp / ‖p‖. Keep the k nodes with the highest scores, and keep the edges among them. Multiply the kept features by sigmoid(y) of their scores: this **gate** is what lets gradients reach p, because a hard top-k selection on its own is not differentiable.

**gUnpool (graph unpooling).** The inverse of gPool. Remember the indices of the kept nodes, create a zero feature matrix for the full graph, and put the coarse features back at those positions.

**Graph connectivity augmentation.** Dropping nodes also drops their edges, so the pooled graph can fall apart into isolated nodes. Gao and Ji compute the pooled graph from the **second graph power**, connecting nodes up to two hops apart before selecting, so that two kept nodes with a dropped node between them stay connected. They also gave each node's own feature more weight in the GCN layer (adding 2I instead of I to the adjacency matrix).

**Architecture.** Encoder blocks of gPool followed by a GCN layer; decoder blocks of gUnpool followed by a GCN layer; skip connections between matching levels (addition or concatenation), exactly like an image U-Net.

**Use on simulation meshes.** Graph U-Nets appear as baselines in mesh-based physics papers (AB-UPT, for example, lists Graph-UNet among methods that process the full simulation mesh). A known weakness for physics: learned top-k selection can pick nodes that cluster in one region and leave other regions without any coarse node. Geometric coarsening (bi-stride, voxel clustering, farthest point sampling) guarantees coverage, which is why physics models usually prefer it.

## Code: smoothing, hop counts and bi-stride pooling

Three experiments on a structured 2D mesh graph: how fast repeated neighbour averaging removes detail, how the hop count to cross the domain grows with resolution, and how bi-stride pooling shrinks both the graph and the number of hops.

```python
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

def grid_graph(n):
    """Adjacency (CSR) of an n x n structured mesh with 4-neighbour edges."""
    idx = np.arange(n * n).reshape(n, n)
    e = np.concatenate([np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()]),
                        np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()])], axis=1)
    A = sp.coo_matrix((np.ones(e.shape[1]), (e[0], e[1])), shape=(n * n, n * n))
    return (A + A.T).tocsr()

# 1. Repeated neighbour averaging (the linear part of a GCN or mean-aggregation layer) smooths features.
n = 64
A = grid_graph(n)
deg = np.asarray(A.sum(1)).ravel()
P = sp.diags(1.0 / (deg + 1)) @ (A + sp.eye(n * n))          # mean over the node and its neighbours
L = sp.diags(deg) - A                                          # graph Laplacian
rng = np.random.default_rng(0)
X0 = rng.normal(size=(n * n, 8))                               # 7 noise channels (fine detail) ...
X0[:, 0] = np.sin(2 * np.pi * (np.arange(n * n) // n) / n)     # ... and one smooth, large-scale channel

def dirichlet(X):                                              # sum over edges of (x_i - x_j)^2, per channel
    return np.einsum("ij,ij->j", X, L @ X)

X, done = X0.copy(), 0
print("layers   fine-detail energy left   large-scale energy left")
for k in [1, 4, 16, 64]:
    for _ in range(k - done):
        X = P @ X
    done = k
    ratio = dirichlet(X) / dirichlet(X0)
    print(f"{k:>6}   {ratio[1:].mean():>14.4f}            {ratio[0]:>10.4f}")

# 2. The same physical distance takes more hops on a finer mesh.
print("mesh        nodes   hops corner to corner")
for m in [16, 32, 64, 128]:
    d = shortest_path(grid_graph(m), unweighted=True, indices=0)
    print(f"{m:>3} x {m:<3} {m * m:>7}   {int(d[-1]):>5}")

# 3. Bi-stride pooling (BSMS-GNN): keep every other BFS frontier, connect kept nodes at most 2 hops apart.
def bistride_pool(A, seed=0):
    depth = shortest_path(A, unweighted=True, indices=seed)
    keep = np.flatnonzero(depth % 2 == 0)
    A2 = ((A + A @ A) > 0).astype(float)
    A2.setdiag(0)
    A2.eliminate_zeros()
    return keep, A2[keep][:, keep].tocsr()

A_level, levels = grid_graph(64), []
print("level   nodes   hops to cross from the seed corner")
for level in range(4):
    depth = shortest_path(A_level, unweighted=True, indices=0)
    levels.append((A_level.shape[0], int(depth.max())))
    print(f"{level:>5} {A_level.shape[0]:>7}   {int(depth.max()):>5}")
    keep, A_level = bistride_pool(A_level)
```

Output:

```
layers   fine-detail energy left   large-scale energy left
     1           0.1203                0.9680
     4           0.0073                0.9130
    16           0.0005                0.7888
    64           0.0000                0.5345
mesh        nodes   hops corner to corner
 16 x 16      256      30
 32 x 32     1024      62
 64 x 64     4096     126
128 x 128   16384     254
level   nodes   hops to cross from the seed corner
    0    4096     126
    1    2048      63
    2    1024      31
    3     512      15
```

**Reading the results:**

- One averaging layer removes 88% of the fine-detail energy and four layers remove more than 99%. After 64 layers even the smooth, domain-scale wave has lost almost half its energy. This is oversmoothing in its purest form; learned messages and residual connections slow it down but do not remove the tendency.
- The hops to cross the domain double with each refinement: 30, 62, 126, 254.
- Each bi-stride level halves both the node count and the hops. Three levels turn a 126-hop problem into a 15-hop one, so a handful of message-passing steps per level reaches across the whole mesh.

## Code: gPool and gUnpool

A direct implementation of the Graph U-Net pooling equations, on an 8 × 8 mesh graph. It checks the shapes, that gradients reach the projection vector through the gate, and how many kept nodes become isolated with and without the second graph power.

```python
import torch

def gpool(X, A, p, k, two_hop=True):
    """Graph U-Net pooling: score nodes by projection onto the trainable vector p, keep the top k,
    gate the kept features by sigmoid(score) so that p receives gradients, keep edges among kept nodes."""
    y = X @ p / p.norm()
    idx = torch.topk(y, k).indices.sort().values                  # keep the original node order
    if two_hop:                                                    # second graph power: nodes <= 2 hops apart
        A = ((A + A @ A) > 0).float()
        A.fill_diagonal_(0)
    return X[idx] * torch.sigmoid(y[idx])[:, None], A[idx][:, idx], idx

def gunpool(X_coarse, idx, n_nodes):
    """Put pooled features back at their original positions; the other nodes start at zero."""
    out = X_coarse.new_zeros(n_nodes, X_coarse.shape[1])
    out[idx] = X_coarse
    return out

torch.manual_seed(0)
A_grid = torch.tensor(grid_graph(8).toarray(), dtype=torch.float32)   # 64 nodes
X = torch.randn(64, 4)
p = torch.randn(4, requires_grad=True)

for two_hop in [False, True]:
    Xp, Ap, idx = gpool(X, A_grid, p, k=16, two_hop=two_hop)
    isolated = int((Ap.sum(1) == 0).sum())
    print(f"second graph power {two_hop!s:<5}: pooled {tuple(Xp.shape)}, adjacency {tuple(Ap.shape)}, "
          f"isolated nodes {isolated}")

Xp.sum().backward()
print("gradient reaches p:", p.grad is not None and p.grad.abs().sum().item() > 0)
X_up = gunpool(Xp.detach(), idx, 64)
print("unpooled", tuple(X_up.shape), "non-zero rows", int((X_up.abs().sum(1) > 0).sum()))
```

Output:

```
second graph power False: pooled (16, 4), adjacency (16, 16), isolated nodes 9
second graph power True : pooled (16, 4), adjacency (16, 16), isolated nodes 2
gradient reaches p: True
unpooled (64, 4) non-zero rows 16
```

With plain edges, 9 of the 16 kept nodes have no neighbour left in the pooled graph, so a GCN layer after pooling could not pass any message to them. The second graph power reduces that to 2. The random scores here pick scattered nodes, which is the coverage problem above in miniature.

## In an interview

**Thirty-second answer to "What are oversmoothing and oversquashing, and how do multiscale GNNs help?"** "Message passing moves information one hop per layer. Oversmoothing: each layer is a kind of Laplacian smoothing, so stacking many layers averages features until local detail like a stress peak is gone. Oversquashing: information from a large or bottlenecked neighbourhood is compressed into fixed-size vectors, so long-range influence is lost. On meshes there is also simple under-reaching: refining the mesh multiplies the hops between two physical points. Multiscale GNNs add coarser graph levels, built from a coarse mesh, bi-stride pooling or clustering, so a few messages on a coarse level cover long distances while the fine level keeps local detail."

**Likely follow-ups**

- *Why not just add more layers?* Cost grows linearly, oversmoothing gets worse, and training gets harder; a hierarchy is cheaper and better behaved.
- *What is the risk of spatial-proximity coarsening?* Edges that jump across thin walls or gaps and connect regions that are not physically connected; bi-stride pooling on the mesh graph avoids this.
- *How does a Graph U-Net choose which nodes to keep?* By projecting node features on a learned vector and keeping the top k, with a sigmoid gate so the vector is trainable.

## References

- [Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning (Li, Han & Wu)](https://arxiv.org/abs/1801.07606)
- [On the Bottleneck of Graph Neural Networks and its Practical Implications (Alon & Yahav)](https://arxiv.org/abs/2006.05205)
- [Understanding over-squashing and bottlenecks on graphs via curvature (Topping et al.)](https://arxiv.org/abs/2111.14522)
- [MultiScale MeshGraphNets (Fortunato et al.)](https://arxiv.org/abs/2210.00612)
- [Efficient Learning of Mesh-Based Physical Simulation with BSMS-GNN (Cao et al.)](https://arxiv.org/abs/2210.02573)
- [X-MeshGraphNet (Nabian et al.)](https://arxiv.org/abs/2411.17164)
- [Graph U-Nets (Gao & Ji)](https://arxiv.org/abs/1905.05178)
