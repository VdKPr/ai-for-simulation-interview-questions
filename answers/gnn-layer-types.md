# GNN Layer Types: GCN, GAT, GraphSAGE and GIN

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** All four layers update a node from its neighbours; they differ in **how neighbours are weighted and combined**. **GCN** (Kipf & Welling, 2017) averages neighbours with fixed weights given by node degrees, 1/√(dᵢdⱼ), then applies one shared linear map. **GAT** (Veličković et al., 2018) learns an attention weight for every edge, so different neighbours count differently. **GraphSAGE** (Hamilton et al., 2017) concatenates a node's own feature with an aggregate (mean, max or LSTM) of a **sampled** set of neighbours, which makes it inductive and scalable to huge graphs. **GIN** (Xu et al., 2019) uses a **sum** followed by an MLP, which is provably as expressive as the Weisfeiler-Lehman graph isomorphism test, the maximum for this family of models; mean and max aggregation cannot tell some different neighbourhoods apart. For simulation meshes, none of the four is the usual choice: MeshGraphNets-style layers with **edge features** (relative positions, distances) and MLP messages work better, because the geometry of each edge carries the physics.

**On this page**

- [The common template](#the-common-template)
- [GCN](#gcn)
- [GAT](#gat)
- [GraphSAGE](#graphsage)
- [GIN and expressive power](#gin-and-expressive-power)
- [Comparison](#comparison)
- [Which layer for a simulation mesh](#which-layer-for-a-simulation-mesh)
- [Code: the four layers from scratch](#code-the-four-layers-from-scratch)
- [In an interview](#in-an-interview)

## The common template

Every layer computes, for each node i with neighbours N(i):

```math
h_i' = \phi\Big(h_i,\ \bigoplus_{j \in \mathcal{N}(i)} \psi(h_i, h_j, e_{ij})\Big)
```

- ψ produces a message from neighbour j (possibly using the edge feature e_ij),
- ⊕ is a permutation-invariant aggregation (sum, mean, max, or an attention-weighted sum),
- φ updates the node from its old state and the aggregate.

The choice of ψ, ⊕ and φ is what distinguishes the layers below. See [Message passing](gnns-and-meshgraphnets.md#message-passing) for the general framework.

## GCN

**Formula.** With self-loops added (Â = A + I) and D̂ the degree matrix of Â:

```math
H' = \sigma\big(\hat D^{-1/2}\,\hat A\,\hat D^{-1/2}\,H\,W\big)
```

Per node: h'_i = σ(Σ_{j ∈ N(i) ∪ {i}} W h_j / √(d̂_i d̂_j)).

**Where it comes from.** Kipf and Welling derived it as a localized first-order approximation of spectral graph convolutions (filters defined through the eigenvectors of the graph Laplacian). The result is cheap: cost scales linearly with the number of edges.

**Strengths.** Simple, fast, few parameters, strong on node classification in citation networks, the setting it was designed for.

**Weaknesses.**

- Fixed weights: every neighbour's contribution depends only on degrees, not on content or geometry.
- No edge features.
- It is a Laplacian smoothing step, so deep GCNs oversmooth. See [Oversmoothing and oversquashing](multiscale-gnns.md#oversmoothing-and-oversquashing).
- The original formulation is transductive (the whole graph is known at training time), although the layer itself can be applied to new graphs.

## GAT

**Formula.** For each edge, a learned score compares the transformed features of the two endpoints; a softmax over each node's neighbours turns the scores into weights:

```math
e_{ij} = \mathrm{LeakyReLU}\big(a^\top [W h_i \,\|\, W h_j]\big), \qquad
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}, \qquad
h_i' = \sigma\Big(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} W h_j\Big)
```

Several independent attention heads are computed and concatenated (or averaged in the last layer).

**Strengths.** Different neighbours get different importance, learned from the data. No expensive matrix operations and no need to know the graph structure in advance, so it works inductively on unseen graphs. Alon and Yahav found attention-based layers less prone to oversquashing than layers that weight all edges equally.

**Weaknesses.** More parameters and memory per edge; attention weights over a node's neighbours always sum to one, so like mean aggregation it cannot count neighbours. The original scoring function is limited in expressivity (GATv2 fixed this by applying the nonlinearity before the dot product with a).

## GraphSAGE

**Formula** (mean aggregator):

```math
h_i' = \sigma\Big(W \cdot \big[\,h_i \,\|\, \mathrm{mean}_{j \in S(i)}\, h_j\,\big]\Big), \qquad h_i' \leftarrow h_i' / \lVert h_i' \rVert
```

where S(i) is a fixed-size **random sample** of the neighbours. The paper also proposed max-pooling and LSTM aggregators.

**Strengths.**

- **Inductive:** it learns functions that generate embeddings from a node's neighbourhood, so it applies to nodes and graphs never seen in training.
- **Scalable:** neighbour sampling bounds the cost per node and enables mini-batch training on graphs with millions of nodes.
- Concatenating the node's own feature with the neighbour aggregate (rather than mixing them) keeps the node's identity, which helps against oversmoothing.

**Weaknesses.** Sampling adds variance. Mean and max aggregators lose information about neighbourhood size and multiplicity (see GIN below).

## GIN and expressive power

**The question Xu et al. asked:** can a GNN tell two different graphs, or two different neighbourhoods, apart? Message passing compares neighbourhoods by aggregating a **multiset** of neighbour features. If two different multisets aggregate to the same value, no later layer can separate them.

- **Mean** cannot distinguish {a} from {a, a}: both average to a.
- **Max** cannot distinguish {a, b} from {a, a, b}.
- **Sum** of suitably transformed features is injective on multisets, so it can.

They proved that a neighbourhood-aggregation GNN is at most as powerful as the **1-dimensional Weisfeiler-Lehman (WL) test** for graph isomorphism, showed that GCN and GraphSAGE fall short of that bound, and proposed GIN, which reaches it:

```math
h_i' = \mathrm{MLP}\Big((1 + \epsilon)\, h_i + \sum_{j \in \mathcal{N}(i)} h_j\Big)
```

The MLP (rather than one linear map) is needed so the update can model an injective function of the sum.

**Why it matters for physics.** Counting matters: the number of contacts on a node, the number of elements meeting at a vertex, the number of particles within a radius (density in SPH). Sum aggregation can represent such quantities; mean aggregation cannot. MeshGraphNets and GNS use **sum** aggregation for this reason. The price is sensitivity to node degree: nodes with many neighbours get larger aggregates, so normalization (or degree-aware features) matters on meshes with varying refinement.

## Comparison

| | GCN | GAT | GraphSAGE | GIN |
| --- | --- | --- | --- | --- |
| Neighbour weights | Fixed, 1/√(dᵢdⱼ) | Learned attention per edge | Equal (mean) or learned (max, LSTM) | Equal, sum |
| Own feature | Mixed in via a self-loop | Via self-loop | Concatenated separately | (1 + ε) h_i |
| Update | Linear + nonlinearity | Linear + nonlinearity, multi-head | Linear on concatenation | MLP |
| Edge features | No | Not in the original | No | No (GINE variant adds them) |
| Expressive power | Below WL | Below WL (weights sum to 1) | Below WL with mean/max | Equal to 1-WL |
| Scaling trick | Sparse matrix multiply | Sparse attention | Neighbour sampling, mini-batches | Sparse sum |
| Designed for | Semi-supervised node classification | Node classification, inductive | Large, evolving graphs | Graph classification |

## Which layer for a simulation mesh

1. **Edge features carry the geometry.** On a mesh, the relative position x_j − x_i and the edge length tell the network the local stencil. GCN, the original GAT, GraphSAGE and GIN ignore them; MeshGraphNets-style layers put an MLP on [h_i, h_j, e_ij] for every edge. See [Node and edge features](gnns-and-meshgraphnets.md#node-and-edge-features).
2. **Sum aggregation** keeps density and connectivity information (GIN's insight), as MeshGraphNets and GNS do.
3. **Residual updates and normalization** let you stack the 10-15 layers mesh problems need without oversmoothing.
4. **Equivariant layers** (EGNN, SEGNN) when the output must rotate with the input. See [When you need an equivariant GNN](equivariance-and-symmetry.md#when-you-need-an-equivariant-gnn).
5. **GCN or GraphSAGE are fine baselines** for quick experiments, for scalar regression on graph-level features, or for very large graphs where sampling is essential.

A useful interview line: "GCN, GAT, GraphSAGE and GIN were designed for citation, social and molecular graphs, where nodes carry rich features and edges are just relations. On a mesh, the edges are the geometry, so I would use a layer with edge MLPs and sum aggregation."

## Code: the four layers from scratch

Each layer is written with plain PyTorch scatter operations (`index_add_`), the same way PyTorch Geometric implements them. The checks: every layer is permutation-equivariant (relabelling the nodes relabels the outputs), and only sum aggregation distinguishes a node with one neighbour from a node with two identical neighbours.

```python
import torch, torch.nn as nn

def scatter_sum(values, index, n):
    return torch.zeros(n, values.shape[1], dtype=values.dtype).index_add_(0, index, values)

class GCNLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
    def forward(self, h, edge_index):
        n = h.shape[0]
        loops = torch.arange(n)
        src = torch.cat([edge_index[0], loops]); dst = torch.cat([edge_index[1], loops])   # add self-loops
        deg = torch.zeros(n).index_add_(0, dst, torch.ones(len(dst)))
        norm = (deg[src] * deg[dst]).rsqrt()[:, None]                                   # 1 / sqrt(d_i d_j)
        return torch.relu(scatter_sum(norm * self.lin(h)[src], dst, n))

class GATLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out, bias=False)
        self.att = nn.Linear(2 * d_out, 1, bias=False)
    def forward(self, h, edge_index):
        n = h.shape[0]
        loops = torch.arange(n)
        src = torch.cat([edge_index[0], loops]); dst = torch.cat([edge_index[1], loops])
        z = self.lin(h)
        score = nn.functional.leaky_relu(self.att(torch.cat([z[dst], z[src]], 1)), 0.2).squeeze(1)
        score = (score - score.max()).exp()                                             # softmax over each node's edges
        alpha = score / torch.zeros(n).index_add_(0, dst, score)[dst]
        return torch.relu(scatter_sum(alpha[:, None] * z[src], dst, n))

class SAGELayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(2 * d_in, d_out)
    def forward(self, h, edge_index):
        src, dst = edge_index
        n = h.shape[0]
        count = torch.zeros(n).index_add_(0, dst, torch.ones(len(dst))).clamp(min=1)[:, None]
        mean = scatter_sum(h[src], dst, n) / count                                      # mean over neighbours
        out = torch.relu(self.lin(torch.cat([h, mean], 1)))
        return out / out.norm(dim=1, keepdim=True).clamp(min=1e-12)

class GINLayer(nn.Module):
    def __init__(self, d_in, d_out, eps=0.0):
        super().__init__()
        self.eps = eps
        self.mlp = nn.Sequential(nn.Linear(d_in, d_out), nn.ReLU(), nn.Linear(d_out, d_out))
    def forward(self, h, edge_index):
        src, dst = edge_index
        return self.mlp((1 + self.eps) * h + scatter_sum(h[src], dst, h.shape[0]))       # sum, then an MLP

torch.manual_seed(0)
n, d = 12, 8
h = torch.randn(n, d)
undirected = torch.randint(0, n, (2, 30))
edge_index = torch.cat([undirected, undirected.flip(0)], 1)                            # both directions
perm = torch.randperm(n)
inverse = torch.empty_like(perm); inverse[perm] = torch.arange(n)
layers = {"GCN": GCNLayer(d, 16), "GAT": GATLayer(d, 16), "SAGE": SAGELayer(d, 16), "GIN": GINLayer(d, 16)}
for name, layer in layers.items():
    out = layer(h, edge_index)
    out_perm = layer(h[perm], inverse[edge_index])                                    # relabel the nodes
    print(f"{name:<5} output {tuple(out.shape)}  permutation-equivariant: {torch.allclose(out[perm], out_perm, atol=1e-5)}")

# Node 0 has one neighbour with feature a; node 1 has two neighbours, both with feature a.
a = torch.randn(1, d)
h2 = torch.cat([torch.zeros(2, d), a, a, a])                        # nodes 0, 1 (targets), 2, 3, 4 (feature a)
edges = torch.tensor([[2, 3, 4], [0, 1, 1]])                         # 2 -> 0;  3 -> 1 and 4 -> 1
mean_agg = scatter_sum(h2[edges[0]], edges[1], 5) / torch.tensor([1., 2., 1., 1., 1.])[:, None]
sum_agg = scatter_sum(h2[edges[0]], edges[1], 5)
print("mean aggregation tells node 0 from node 1:", not torch.allclose(mean_agg[0], mean_agg[1]))
print("sum aggregation tells node 0 from node 1: ", not torch.allclose(sum_agg[0], sum_agg[1]))
```

Output:

```
GCN   output (12, 16)  permutation-equivariant: True
GAT   output (12, 16)  permutation-equivariant: True
SAGE  output (12, 16)  permutation-equivariant: True
GIN   output (12, 16)  permutation-equivariant: True
mean aggregation tells node 0 from node 1: False
sum aggregation tells node 0 from node 1:  True
```

## In an interview

**Thirty-second answer to "GCN vs GAT vs GraphSAGE vs GIN?"** "They all aggregate neighbours, but differently. GCN averages neighbours with fixed degree-based weights, a first-order spectral convolution. GAT learns an attention weight per edge. GraphSAGE concatenates the node with an aggregate of sampled neighbours, so it is inductive and scales to huge graphs. GIN uses a sum and an MLP, which makes it as expressive as the Weisfeiler-Lehman test; mean or max cannot tell a node with one neighbour from one with two identical neighbours. For meshes I would use MeshGraphNets-style layers, because they use edge features like relative position and sum aggregation."

**Likely follow-ups**

- *Why does GCN oversmooth?* Each layer is a normalized averaging step, a form of Laplacian smoothing; many of them converge features.
- *What is the Weisfeiler-Lehman test?* Iteratively relabel each node by hashing its label with the multiset of neighbour labels; two graphs whose label histograms differ are not isomorphic. Message passing with injective aggregation mimics it.
- *How would you add edge features to GIN or GAT?* Add the transformed edge feature into each message (GINE), or into the attention score.

## References

- [Semi-Supervised Classification with Graph Convolutional Networks (Kipf & Welling)](https://arxiv.org/abs/1609.02907)
- [Graph Attention Networks (Veličković et al.)](https://arxiv.org/abs/1710.10903)
- [Inductive Representation Learning on Large Graphs (GraphSAGE; Hamilton, Ying & Leskovec)](https://arxiv.org/abs/1706.02216)
- [How Powerful are Graph Neural Networks? (GIN; Xu et al.)](https://arxiv.org/abs/1810.00826)
- [How Attentive are Graph Attention Networks? (GATv2; Brody, Alon & Yahav)](https://arxiv.org/abs/2105.14491)
- [PyTorch Geometric documentation](https://pytorch-geometric.readthedocs.io/)
