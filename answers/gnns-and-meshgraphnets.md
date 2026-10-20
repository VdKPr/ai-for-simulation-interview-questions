# GNNs and MeshGraphNets

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A simulation mesh *is* a graph: nodes are mesh vertices (or cells), edges are their connections. A graph neural network (GNN) works directly on it by **message passing**: in each layer, every edge computes a message from the features of its two endpoints, each node sums the messages it receives, and updates its own features. That matches how PDE discretizations work (a node is updated from its neighbours through a stencil or flux balance), it handles unstructured meshes of any size and topology, it does not depend on how nodes are numbered, and with relative-position edge features it is translation-invariant. **MeshGraphNets** (Pfaff et al., ICLR 2021) is the reference architecture: encode node and edge features, process them with a stack of message-passing blocks with residual connections, decode a per-node update, and roll out in time; trained with noise injection it ran one to two orders of magnitude faster than the simulators it learned from. The main limitation is reach: each layer moves information one hop, so global effects need many layers, multiscale graphs or attention.

**On this page**

- [Why GNNs fit simulation meshes](#why-gnns-fit-simulation-meshes)
- [Message passing](#message-passing)
- [The graph-network framework](#the-graph-network-framework)
- [MeshGraphNets: encoder, processor, decoder](#meshgraphnets-encoder-processor-decoder)
- [Node and edge features](#node-and-edge-features)
- [How many message-passing steps](#how-many-message-passing-steps)
- [Code: one message-passing layer](#code-one-message-passing-layer)
- [Practical issues](#practical-issues)
- [In an interview](#in-an-interview)

## Why GNNs fit simulation meshes

1. **The data already is a graph.** Mesh nodes and element edges give the connectivity; no rasterization to a grid is needed, so boundary layers, thin walls and local refinement keep their resolution.
2. **Locality matches the physics.** A finite-difference stencil or a finite-volume flux balance updates each unknown from its neighbours. A message-passing layer does the same with learned functions. MP-PDE showed that message passing can represent classical finite-difference, finite-volume and WENO schemes. See [Message Passing Neural PDE Solvers](autoregressive-rollouts.md#message-passing-neural-pde-solvers).
3. **Any size and topology.** The same learned functions apply to every node and edge, so a model trained on meshes of 10,000 nodes can run on a mesh of 100,000, and on new topologies.
4. **No dependence on node numbering.** Sum or mean aggregation makes the layer permutation-equivariant: renumbering nodes renumbers the outputs, nothing else changes. The code below checks this.
5. **Relative features give translation invariance**, and other symmetries can be built in. See [Equivariance and symmetry](equivariance-and-symmetry.md).
6. **Boundary conditions fit naturally** as node types (wall, inlet, outlet, fixed) and features.

**The trade-offs:** information travels only one hop per layer, so long-range effects are expensive; memory grows with the number of edges times the hidden size times the number of layers; and predictions can depend on how the geometry was meshed.

## Message passing

For each directed edge (i → j) with features e_ij and nodes with features h_i:

```math
\begin{aligned}
e'_{ij} &= \phi_e\big(e_{ij},\, h_i,\, h_j\big) && \text{edge update (the message)}\\
m_j &= \textstyle\sum_{i \in \mathcal{N}(j)} e'_{ij} && \text{aggregation (sum, mean or max)}\\
h'_j &= \phi_v\big(h_j,\, m_j\big) && \text{node update}
\end{aligned}
```

φ_e and φ_v are small MLPs **shared** across all edges and nodes. The aggregation must be permutation-invariant (sum, mean, max), because a node's neighbours have no natural order. **Sum** keeps information about how many neighbours there are and is common in physics models, where it plays the role of summing fluxes; **mean** normalizes for varying degree; **max** picks out extremes.

Stacking L such layers gives each node a receptive field of L hops. Residual connections (h' = h + update) and layer normalization keep deep stacks trainable.

## The graph-network framework

Battaglia et al. (2018) generalized message passing into the **graph network (GN) block**, which acts on a graph with edge features e_k, node features v_i and a **global** feature u:

1. **Edge update:** e′_k = φ_e(e_k, v_{receiver}, v_{sender}, u).
2. **Edge-to-node aggregation:** for each node, ē′_i = ρ_{e→v}(updated edges arriving at i).
3. **Node update:** v′_i = φ_v(ē′_i, v_i, u).
4. **Edge-to-global and node-to-global aggregation:** ē′ = ρ_{e→u}(all updated edges), v̄′ = ρ_{v→u}(all updated nodes).
5. **Global update:** u′ = φ_u(ē′, v̄′, u).

**Why it is useful:**

- It unifies many architectures as special cases (message-passing neural networks, interaction networks, graph convolutional networks, Deep Sets), so you can describe a model by which updates and aggregations it uses.
- The **global attribute** is a natural place for global conditions (inlet speed, load magnitude, a material parameter) and global outputs (drag, total mass flow), and it lets information jump across the graph in one step.
- The **encode-process-decode** pattern (encode raw features into latent graphs, process with repeated GN blocks, decode the outputs) is the template that GNS and MeshGraphNets follow.

## MeshGraphNets: encoder, processor, decoder

MeshGraphNets learns the dynamics of a mesh-based simulation: given the state at time t, predict the state at t + Δt.

```
mesh state at t ──▶ ENCODER ──▶ PROCESSOR (L message-passing blocks) ──▶ DECODER ──▶ per-node update ──▶ state at t+Δt
                    MLPs embed       each block: edge MLP, sum, node MLP,        MLP to e.g. acceleration    integrate
                    node & edge      residual connections                        or rate of change           (Euler)
                    features
```

- **Encoder:** separate MLPs embed node features and edge features into a latent size (for example 128).
- **Processor:** a stack of message-passing blocks with residual connections and their own weights per block. The paper used 15 blocks.
- **Decoder:** an MLP maps each node's final latent vector to the output, typically a time derivative (acceleration for cloth and structures, rate of change of velocity for fluids), which is integrated with a simple update rule to get the next state.
- **Training:** supervised on one-step targets (normalized), with **noise injected into the input states** so the model learns to correct its own errors during rollouts. See [Noise injection](autoregressive-rollouts.md#noise-injection).
- **Mesh edges and world edges:**
  - **mesh edges** follow the simulation mesh connectivity and carry information through the material or fluid;
  - **world edges** connect nodes that are close in physical space but far apart along the mesh (within a radius), which lets the model handle **contact and collisions**, for example cloth folding onto itself or a deforming plate touching an actuator.
- **Adaptive remeshing:** the model can also predict a sizing field to adapt the mesh during the rollout, which helps resolution-independent dynamics.
- **Results:** accurate dynamics for aerodynamics, structural mechanics and cloth, running one to two orders of magnitude faster than the simulators it was trained on.

## Node and edge features

**Node features:**

- **Node type** as a one-hot vector: normal, wall, inlet, outlet, fixed boundary, actuator, and so on. This tells the model where the boundary conditions are.
- **The current physical state:** velocity, displacement, temperature, pressure.
- **Global or material conditions** broadcast to every node if they vary between samples (or use the global attribute).
- Optionally geometric features: distance to the nearest wall, surface normals.

**Edge features:**

- The **relative displacement** between the endpoints in mesh (reference) coordinates, u_ij = u_i − u_j, and its length |u_ij|.
- For Lagrangian systems (cloth, deforming solids), also the **relative position in world space**, x_ij = x_i − x_j, and its length; for world edges only the world-space quantities.

**Why relative positions, not absolute coordinates?** The physics does not depend on where the object sits in space. With relative features, translating the whole system changes nothing in the inputs, so the model is translation-invariant by construction and cannot overfit to absolute positions. Absolute coordinates should only be given when they carry real meaning (gravity direction, a fixed inflow), and then as explicit inputs. See [Invariance and equivariance](equivariance-and-symmetry.md#invariance-and-equivariance).

**Normalize everything** with training statistics: node features, edge features and targets. See [Normalizing fields and coordinates](nondimensionalization-and-normalization.md#normalizing-fields-and-coordinates).

## How many message-passing steps

**Each message-passing step moves information by one edge.** After L steps, a node can only depend on nodes within L hops. The code below shows it on a chain: after 15 layers, information from node 0 has reached exactly node 15.

**A rule of thumb:** if an effect must travel a physical distance D on a mesh with edge length about h, the model needs about **D / h** steps.

- For **local, hyperbolic-like dynamics** (a wave travelling a few cells per time step, cloth), the information needed for one time step comes from a small neighbourhood, so 10 to 15 steps suffice. GNS found the number of message-passing steps to be one of the main determinants of accuracy.
- For **elliptic, global effects** (pressure in incompressible flow, stress in a statically loaded part), every point depends on the whole domain. On a mesh 1,000 edges across, a single-scale GNN would need on the order of 1,000 steps: impractical and prone to oversmoothing.
- **Refining the mesh makes it worse:** halve h and the same physical distance needs twice the steps. MultiScale MeshGraphNets were motivated by exactly this: equally distant points in space become further apart in graph space at high resolution.

**Remedies:** multiscale graphs with coarse levels that carry long-range information cheaply ([Multiscale GNNs](multiscale-gnns.md)), a global node or global attribute, attention or transformer layers ([GNN or transformer?](transformers-for-physics.md#gnn-or-transformer)), and larger time steps per prediction balanced against the reach of the processor.

## Code: one message-passing layer

```python
import torch, torch.nn as nn

class MessagePassingLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.node_mlp = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

    def forward(self, h, e, edge_index):
        """h: (N, H) node features, e: (E, H) edge features, edge_index: (2, E) sender/receiver."""
        src, dst = edge_index
        e_new = self.edge_mlp(torch.cat([e, h[src], h[dst]], dim=-1))
        agg = torch.zeros_like(h).index_add_(0, dst, e_new)       # sum messages per receiver
        h_new = self.node_mlp(torch.cat([h, agg], dim=-1))
        return h + h_new, e + e_new                                # residual connections

torch.manual_seed(0)
N, E, H = 10, 30, 16
h, e, edge_index = torch.randn(N, H), torch.randn(E, H), torch.randint(0, N, (2, E))
layer = MessagePassingLayer(H)
h1, e1 = layer(h, e, edge_index)

perm = torch.randperm(N)                                       # renumber the nodes
inv = torch.empty_like(perm); inv[perm] = torch.arange(N)
h2, e2 = layer(h[perm], e, inv[edge_index])                    # same graph, new numbering
equivariant = torch.allclose(h2, h1[perm], atol=1e-6) and torch.allclose(e2, e1, atol=1e-6)
print("permutation equivariant:", equivariant)

# Receptive field: on a chain of 100 nodes, how far does information from node 0 travel in L layers?
n = 100
chain = torch.stack([torch.arange(n - 1), torch.arange(1, n)])
chain = torch.cat([chain, chain.flip(0)], dim=1)               # edges in both directions
signal = torch.zeros(n); signal[0] = 1.0
reached = []
for L in range(1, 16):
    signal = signal + torch.zeros(n).index_add_(0, chain[1], signal[chain[0]])   # one round of sum aggregation
    reached.append(int((signal > 0).sum()) - 1)
print("hops reached after 1..15 layers:", reached)
```

Output:

```
permutation equivariant: True
hops reached after 1..15 layers: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
```

`index_add_` performs the scatter-sum that gives each receiver the sum of its incoming messages; libraries such as PyTorch Geometric and DGL provide optimized versions. The renumbering test is worth keeping in any GNN code base as a unit test.

## Practical issues

- **Remeshing robustness.** Predictions can change when the same geometry is meshed differently, because neighbourhoods and edge lengths change. Train on several mesh resolutions, use edge features that scale properly, and test by remeshing. See [Why remeshing can change a prediction](geometry-representations.md#why-remeshing-can-change-a-prediction).
- **Graph construction.** Make edges bidirectional, remove duplicates, decide whether to include mid-side nodes of quadratic elements, and for point clouds choose k-nearest-neighbour or radius graphs. See [k-nearest-neighbour and radius graphs](mesh-data-processing.md#k-nearest-neighbour-and-radius-graphs).
- **Memory.** Activations scale with (nodes + edges) × hidden size × layers, and must be stored for backpropagation. For million-node meshes, see [GPU memory for a GNN on a 1-million-node mesh](scaling-to-large-meshes.md#gpu-memory-for-a-gnn-on-a-1-million-node-mesh).
- **Oversmoothing and oversquashing** in deep or large graphs. See [Oversmoothing and oversquashing](multiscale-gnns.md#oversmoothing-and-oversquashing).
- **Layer types.** GCN, GAT, GraphSAGE and GIN differ in how they weight and combine neighbours; physics models usually use the general edge-MLP form above. See [GNN layer types](gnn-layer-types.md).

## In an interview

**Thirty-second answer to "Why are GNNs a natural fit for simulation meshes?"** "A mesh is a graph, so a GNN works on it directly without rasterizing, keeping boundary layers and local refinement. Message passing updates each node from its neighbours, which mirrors stencils and flux balances, and MP-PDE showed it can represent classical finite-volume schemes. The same weights apply to any mesh size and topology, outputs don't depend on node numbering, relative edge features give translation invariance, and boundary conditions go in as node types. The catch is that each layer moves information one hop, so global effects like pressure need multiscale graphs or attention."

**Thirty-second answer to "Explain MeshGraphNets."** "Encode node features, such as node type and current velocity, and edge features, the relative displacement and its length in mesh space, plus world space for deforming bodies, with MLPs. Process with about fifteen message-passing blocks with residual connections. Decode a per-node time derivative and integrate it to get the next state. Mesh edges carry information through the material; world edges connect nodes close in space but far along the mesh, for contact. Train on one-step targets with noise injection so rollouts stay stable."

**Likely follow-ups**

- *Why relative positions?* Translation invariance by construction and no overfitting to absolute location.
- *How many message-passing steps do you need?* About the distance information must travel divided by the edge length; elliptic problems need multiscale designs.
- *How would you test a GNN implementation?* Permutation test, translation test, and overfit-one-graph.

## References

- [Learning Mesh-Based Simulation with Graph Networks (MeshGraphNets, Pfaff et al.)](https://arxiv.org/abs/2010.03409) and [code](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)
- [Relational inductive biases, deep learning, and graph networks (Battaglia et al.)](https://arxiv.org/abs/1806.01261)
- [Neural Message Passing for Quantum Chemistry (Gilmer et al.)](https://arxiv.org/abs/1704.01212)
- [Learning to Simulate Complex Physics with Graph Networks (GNS, Sanchez-Gonzalez et al.)](https://arxiv.org/abs/2002.09405)
- [A Gentle Introduction to Graph Neural Networks (Distill)](https://distill.pub/2021/gnn-intro/)
- [MultiScale MeshGraphNets (Fortunato et al.)](https://arxiv.org/abs/2210.00612) and [X-MeshGraphNet (Nabian et al.)](https://arxiv.org/abs/2411.17164)
- [PyG](https://github.com/pyg-team/pytorch_geometric), [DGL](https://github.com/dmlc/dgl), [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo)
