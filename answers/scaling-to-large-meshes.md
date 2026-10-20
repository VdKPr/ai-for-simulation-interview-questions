# Scaling to Large Meshes

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Industrial simulations have 10⁶ to 10⁸ cells, and training a mesh model stores activations for every node, edge and layer, so **memory runs out long before compute**: a 15-layer GNN with hidden size 128 on a 1-million-node tetrahedral mesh needs on the order of hundreds of gigabytes of activations. The tools: **train on random subsets of points** and predict at full resolution with a query decoder (UPT, AB-UPT); **partition the mesh into subdomains with halo regions** (as many hops deep as there are message-passing layers) so each GPU computes exact messages for its part; **multiscale or latent-token architectures** that avoid one huge flat graph; **gradient checkpointing** (recompute activations in the backward pass instead of storing them); **mixed precision** where safe, but not for long sums or small differences of large numbers; **data, model and spatial parallelism** across GPUs; and an input pipeline built for terabytes (chunked binary formats, precomputed features, prefetching from fast storage). Distributed data tools such as **Spark and Dask** handle the conversion of thousands of raw solver outputs into training-ready arrays.

**On this page**

- [GPU memory for a GNN on a 1-million-node mesh](#gpu-memory-for-a-gnn-on-a-1-million-node-mesh)
- [Scaling a GNN to 10 million nodes](#scaling-a-gnn-to-10-million-nodes)
- [Train on subsamples, predict at full resolution](#train-on-subsamples-predict-at-full-resolution)
- [Domain decomposition and halo regions](#domain-decomposition-and-halo-regions)
- [Gradient checkpointing](#gradient-checkpointing)
- [When mixed precision hurts](#when-mixed-precision-hurts)
- [Data parallel vs model parallel](#data-parallel-vs-model-parallel)
- [Feeding terabytes to the GPUs](#feeding-terabytes-to-the-gpus)
- [Spark and Dask](#spark-and-dask)
- [Code: memory, halos, precision and checkpointing](#code-memory-halos-precision-and-checkpointing)
- [In an interview](#in-an-interview)

## GPU memory for a GNN on a 1-million-node mesh

Do this arithmetic out loud in an interview; it is a strong signal.

1. **Edges.** A tetrahedral mesh has about 6-7 edges per node, so about 1.4 × 10⁷ **directed** edges for message passing (a triangle surface mesh has about 3 edges per vertex, 6 directed).
2. **Per layer, per tensor.** One hidden-size-128 float32 tensor over all edges is 1.4 × 10⁷ × 128 × 4 B ≈ **7.2 GB**; over all nodes, 10⁶ × 128 × 4 B ≈ 0.5 GB.
3. **Tensors stored per layer.** An MLP on edges stores its input (the concatenation of sender, receiver and edge features), hidden activations and output for the backward pass: several edge-sized tensors per layer. With three, a layer stores about 22 GB.
4. **Layers.** With 15 message-passing layers (the MeshGraphNets default), roughly **350 GB** in float32, before parameters, optimizer state and framework overhead. The code below gives 346 GB.
5. **Conclusion.** This does not fit on an 80 GB GPU. Options: fewer or narrower layers, bf16 activations (half), gradient checkpointing (store only layer inputs), subsampling, partitioning across GPUs, or a multiscale or latent architecture.

Parameters are small by comparison (a few million for typical mesh GNNs): **activations, not weights, dominate memory** in mesh models, which decides which parallelism helps.

## Scaling a GNN to 10 million nodes

Ten times more nodes: about 3.5 TB of activations by the same arithmetic. A combination of techniques is needed.

1. **Partition the graph** across many GPUs with halo regions (below), using a balanced, low-cut partitioner (METIS or a geometric partitioner); this is the approach of X-MeshGraphNet, which partitions large graphs with halos and aggregates gradients so that training is equivalent to full-graph training.
2. **Multiscale message passing** so long-range information travels on coarse levels and the fine level needs few layers. See [Multiscale GNNs](multiscale-gnns.md).
3. **Gradient checkpointing** per layer and **bf16** activations.
4. **Subsample nodes for training** if the architecture allows it (radius graphs rebuilt on the subsample, or point-based models).
5. **Consider an architecture change:** latent-token transformers (Transolver, UPT, AB-UPT) or point-based neural operators (DoMINO) avoid full-mesh message passing; AB-UPT trains on 140-million-cell meshes on a single GPU by sampling anchors and queries. See [Physics transformer architectures](physics-transformer-architectures.md).
6. **Inference** needs no stored activations, so it is far cheaper; run it in chunks or partitions.

## Train on subsamples, predict at full resolution

**Idea.** Each training step uses a random subset of points (or cells) from a simulation, perhaps 1-10% of them; at inference, the model is queried at every point, in chunks.

- **Why it works:** for models whose prediction at a query point depends on a fixed-size encoded context (neural-field decoders, cross-attention to latent tokens or anchors), subsampling queries changes nothing about the function being learned; it just evaluates the loss on a random subset, an unbiased estimate of the full loss.
- **Bonus:** each simulation yields many different training examples, a strong regularizer when there are only hundreds of simulations.
- **Caveat for GNNs:** neighbourhoods change when points are subsampled, so the graph must be built consistently between training and inference (radius graphs in physical units, or train on the same density you predict on), or the model sees a different graph at inference.
- **Weighting:** sample points uniformly in space or by cell volume, not uniformly over mesh cells, or refined regions dominate.

## Domain decomposition and halo regions

**Problem.** Split the mesh into P subdomains, one per GPU. A node near a cut needs messages from neighbours that live on another GPU.

**Halo (ghost) nodes.** Add to each subdomain copies of the nodes within k hops outside its boundary. With L message-passing layers, information travels L hops, so **a halo of L hops makes the owned nodes' outputs exact**: identical to the full-graph computation. Only the owned nodes' outputs and losses count; halos are recomputed or exchanged.

**Two implementations.**

1. **Static halos:** include the full L-hop halo in each partition and compute everything locally; no communication in the forward pass, but halo overhead grows with L and with the surface-to-volume ratio of partitions (52% extra nodes in the code below for a 5-hop halo on small partitions).
2. **Halo exchange per layer:** keep a one-hop halo and exchange updated halo features between GPUs after every layer (as in classical parallel CFD codes); less memory and computation, but communication at each layer.

**Gradients:** each GPU backpropagates through its partition; gradients of the shared weights are summed across GPUs (as in data parallelism), so one step equals full-graph training.

**Choosing partitions:** balance the number of nodes and minimize cut edges (METIS); keep partitions compact so halos are small.

## Gradient checkpointing

**Idea (Chen et al., 2016, "Training Deep Nets with Sublinear Memory Cost").** Instead of storing every intermediate activation for the backward pass, store only some (for example each block's input) and **recompute** the others during backpropagation. Memory for a chain of n blocks drops from O(n) to about O(√n) (or to one block's worth of internals plus the stored inputs), at the cost of roughly one extra forward pass, about 30% more compute.

**In PyTorch:** `torch.utils.checkpoint.checkpoint(block, x)` around each message-passing layer or transformer block. The code below measures the activations saved for backward dropping from 478 MB to 50 MB for 12 MLP blocks.

**Where it helps most:** wide MLPs on edges and long chains of layers. **Same idea elsewhere:** checkpointing forward states for adjoint solvers of time-dependent problems. See [Time-dependent and nonlinear problems](adjoint-method.md#time-dependent-and-nonlinear-problems).

## When mixed precision hurts

**Background.** Mixed-precision training (Micikevicius et al., 2018) runs matrix multiplies and stores activations in 16-bit formats, keeping a float32 master copy of the weights. **fp16** has 10 mantissa bits and a maximum of 65,504 (it needs loss scaling to avoid underflowing gradients); **bf16** has the float32 exponent range but only 7 mantissa bits (about 2-3 significant decimal digits).

**Where physics breaks it.**

1. **Long accumulations.** A node aggregating thousands of messages, or a global integral (drag from millions of surface elements), accumulates in low precision: once the running total is large, small contributions are rounded away. In the code below, summing 20,000 values of 0.001 stops at 4 in fp16 and 0.5 in bf16. Accumulate in float32 (scatter-add and reductions in fp32).
2. **Small differences of large numbers.** Absolute pressure ≈ 101,325 Pa with variations of tens of pascals: bf16 cannot represent the difference (the gauge pressure becomes 0), and fp16 overflows. Work with gauge pressure or pressure coefficients, normalized fields and non-dimensional coordinates. See [Nondimensionalization and normalization](nondimensionalization-and-normalization.md).
3. **Coordinates and geometry.** Positions in millimetres on a 5-metre car have 5 significant digits; relative positions and small distances (boundary-layer cells of micrometres) lose precision; compute edge vectors in fp32 before casting.
4. **Derivatives** (PINN residuals by autograd, finite differences of predictions) amplify rounding error.
5. **Ill-conditioned linear algebra** inside the model (solves, eigendecompositions, FFTs with large dynamic range).

**Practice:** autocast matrix multiplies to bf16, keep reductions, losses, normalization and geometric preprocessing in fp32, check that bf16 training matches fp32 on a small run, and re-validate accuracy after quantizing for inference.

## Data parallel vs model parallel

| | Data parallel | Model / tensor parallel | Spatial (graph, domain) parallel | Pipeline parallel |
| --- | --- | --- | --- | --- |
| What is split | The batch: each GPU has a full model copy and different samples | Each layer's weights across GPUs | One sample's mesh across GPUs (partitions with halos) | Consecutive layers on different GPUs |
| Communication | All-reduce of gradients each step | Activations within each layer | Halo features (per layer or once) and gradient all-reduce | Activations between stages |
| Requirement | One sample must fit on one GPU | Very large layers | Large samples; good partitions | Many layers; micro-batches to fill the pipeline |
| Typical for mesh models | Default when samples fit | Rarely needed (models are small) | **The main tool for huge meshes** | Rare |

- **ZeRO and FSDP** shard parameters, gradients and optimizer states across data-parallel workers; they matter for billion-parameter foundation models, less for typical mesh GNNs whose memory is dominated by activations.
- **Sequence parallelism** for transformers splits the token dimension across GPUs; Transolver++ used it to process 2.5-million-cell meshes on four GPUs.
- **Rule:** if a sample fits, data parallel; if it does not, split the sample (spatial parallelism or subsampling) before splitting the model.

## Feeding terabytes to the GPUs

1. **Convert once, offline**, from solver formats to chunked binary arrays (Zarr, HDF5, or sharded tar files), one sample per group or shard, in float32 or float16 where precision allows. Never parse VTK text or solver output at training time. See [Simulation file formats](simulation-file-formats.md#handling-terabytes-of-simulation-files).
2. **Precompute expensive features:** signed distances, normals, graphs, normalization statistics (from the training split only).
3. **Store the data where the GPUs are:** stage hot data on local NVMe; stream the rest from object storage with large sequential reads.
4. **Parallel, asynchronous loading:** many loader workers, prefetching, pinned memory, overlapping transfer with compute; GPU-side decompression or NVIDIA DALI for high throughput.
5. **Read only what each step needs:** chunk layout aligned with access (random point subsets need either small chunks or whole-sample reads followed by in-memory sampling).
6. **Shuffle at the shard level** plus a shuffle buffer, rather than random access to individual samples on slow storage.
7. **Measure:** GPU utilization and time per step spent waiting for data; if the GPU waits, the loader is the bottleneck.

## Spark and Dask

**Where they fit.** Turning thousands of raw simulation outputs into training data is embarrassingly parallel: read each case, convert formats, compute derived fields and integrated quantities, check quality, write arrays and metadata. Distributed frameworks run these jobs across a cluster with retries and scheduling.

- **Dask:** Python-native, mirrors NumPy, pandas and scikit-learn APIs, runs arbitrary Python functions over many files (`dask.delayed`, `dask.bag`), chunked arrays for out-of-core computation, and integrates with Zarr and xarray. A natural fit for array-heavy scientific preprocessing.
- **Apache Spark:** the standard for large tabular ETL (metadata catalogues, test and sensor data, joins across many sources), SQL-like processing, and data lakes (Delta Lake, Iceberg); PySpark for Python. Less natural for custom mesh processing.
- **Ray:** general distributed Python, also used for distributed training, hyperparameter search and inference.
- PhysicsX job descriptions list Spark and Dask among the expected skills.

**A typical split:** Spark (or a database) manages the catalogue of cases and their metadata; Dask or Ray runs the per-case mesh processing; the training job reads the resulting Zarr or HDF5 store.

## Code: memory, halos, precision and checkpointing

Four small experiments: the memory arithmetic as a function, a domain decomposition of a 40 × 40 mesh graph into four parts showing that a halo of L hops reproduces L layers of message passing exactly, two mixed-precision failures, and the activation memory saved by gradient checkpointing, measured with PyTorch's saved-tensor hooks.

```python
import numpy as np
import scipy.sparse as sp
import torch, torch.nn as nn
from torch.utils.checkpoint import checkpoint

# 1. Memory arithmetic for message passing (activations stored for the backward pass).
def gnn_activation_memory_gb(n_nodes, edges_per_node, hidden, layers, bytes_per_value=4, tensors_per_edge=3, tensors_per_node=3):
    """Rough training-memory estimate: each layer stores a few hidden-size tensors per directed edge and per node."""
    edges = n_nodes * edges_per_node
    per_layer = (edges * tensors_per_edge + n_nodes * tensors_per_node) * hidden * bytes_per_value
    return layers * per_layer / 1e9
for n, epn in [(1e6, 14), (1e7, 14)]:
    print(f"{n:.0e} nodes, {epn} directed edges per node, hidden 128, 15 layers: "
          f"fp32 {gnn_activation_memory_gb(n, epn, 128, 15):,.0f} GB, bf16 {gnn_activation_memory_gb(n, epn, 128, 15, 2):,.0f} GB")

# 2. Domain decomposition with halos: k layers of message passing need a k-hop halo to be exact.
def grid(nx, ny):
    idx = np.arange(nx * ny).reshape(nx, ny)
    e = np.concatenate([np.stack([idx[:-1].ravel(), idx[1:].ravel()]), np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()])], 1)
    A = sp.coo_matrix((np.ones(e.shape[1]), (e[0], e[1])), shape=(nx * ny,) * 2)
    return (A + A.T).tocsr()
nx = ny = 40
A = grid(nx, ny)
P = sp.diags(1 / (np.asarray(A.sum(1)).ravel() + 1)) @ (A + sp.eye(nx * ny))   # one mean-aggregation layer
rng = np.random.default_rng(0)
h0 = rng.normal(size=(nx * ny, 4))
layers = 5
full = h0.copy()
for _ in range(layers):
    full = np.tanh(P @ full)                                       # nonlinear message passing on the whole graph
part = (np.arange(nx * ny) // ny) // (nx // 2) * 2 + (np.arange(nx * ny) % ny) // (ny // 2)   # 4 quadrants
def run_partition(p, halo):
    owned = np.flatnonzero(part == p)
    nodes = set(owned)
    frontier = set(owned)
    for _ in range(halo):                                          # grow the halo hop by hop
        frontier = set(A[list(frontier)].indices) - nodes
        nodes |= frontier
    nodes = np.array(sorted(nodes))
    local = {g: i for i, g in enumerate(nodes)}
    A_loc = A[nodes][:, nodes]
    P_loc = sp.diags(1 / (np.asarray(A[nodes].sum(1)).ravel() + 1)) @ (A_loc + sp.eye(len(nodes)))   # true degrees
    h = h0[nodes].copy()
    for _ in range(layers):
        h = np.tanh(P_loc @ h)
    return owned, h[[local[g] for g in owned]], len(nodes) / len(owned) - 1
for halo in [0, 2, 5]:
    err, overhead = 0.0, []
    for p in range(4):
        owned, h_loc, ov = run_partition(p, halo)
        err = max(err, np.abs(h_loc - full[owned]).max())
        overhead.append(ov)
    print(f"halo of {halo} hops for {layers} layers: max error vs full graph {err:.1e}, halo overhead {100 * np.mean(overhead):.0f}% extra nodes")

# 3. Mixed precision pitfalls: long sums and small differences of large numbers.
def running_sum(values, dtype):                                    # accumulate one value at a time in the given precision
    total = torch.zeros((), dtype=dtype)
    for v in values.to(dtype):
        total = total + v
    return total.float().item()
contrib = torch.full((20_000,), 1e-3)                               # many small contributions, exact sum 20
print(f"running sum of 20,000 values of 0.001 (exact 20): fp32 {running_sum(contrib, torch.float32):.3f}, "
      f"fp16 {running_sum(contrib, torch.float16):.3f}, bf16 {running_sum(contrib, torch.bfloat16):.3f}")
p_inf, p = torch.tensor(101325.0), torch.tensor(101325.0 + 37.5)    # absolute pressures in Pa, gauge 37.5 Pa
for dt in [torch.float32, torch.bfloat16, torch.float16]:
    print(f"gauge pressure from absolute values in {str(dt).split('.')[-1]:>8}: {(p.to(dt) - p_inf.to(dt)).float().item():8.2f} Pa (exact 37.50)")

# 4. Gradient checkpointing: memory saved for backward, measured with saved-tensor hooks.
class Block(nn.Module):
    def __init__(self, d=256):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x):
        return x + self.net(x)
blocks = nn.ModuleList([Block() for _ in range(12)])
x = torch.randn(4096, 256, requires_grad=True)
def saved_bytes(use_checkpoint):
    total = [0]
    def pack(t):
        total[0] += t.numel() * t.element_size(); return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        h = x
        for b in blocks:
            h = checkpoint(b, h, use_reentrant=False) if use_checkpoint else b(h)
        h.sum().backward()
    return total[0] / 1e6
print(f"activations saved for backward, 12 blocks: without checkpointing {saved_bytes(False):.0f} MB, "
      f"with checkpointing {saved_bytes(True):.0f} MB (at the cost of recomputing each block's forward pass)")
```

Output:

```
1e+06 nodes, 14 directed edges per node, hidden 128, 15 layers: fp32 346 GB, bf16 173 GB
1e+07 nodes, 14 directed edges per node, hidden 128, 15 layers: fp32 3,456 GB, bf16 1,728 GB
halo of 0 hops for 5 layers: max error vs full graph 2.6e-01, halo overhead 0% extra nodes
halo of 2 hops for 5 layers: max error vs full graph 6.4e-02, halo overhead 20% extra nodes
halo of 5 hops for 5 layers: max error vs full graph 0.0e+00, halo overhead 52% extra nodes
running sum of 20,000 values of 0.001 (exact 20): fp32 20.001, fp16 4.000, bf16 0.500
gauge pressure from absolute values in  float32:    37.50 Pa (exact 37.50)
gauge pressure from absolute values in bfloat16:     0.00 Pa (exact 37.50)
gauge pressure from absolute values in  float16:      nan Pa (exact 37.50)
activations saved for backward, 12 blocks: without checkpointing 478 MB, with checkpointing 50 MB (at the cost of recomputing each block's forward pass)
```

**What it shows.**

- The estimate reproduces the arithmetic above: about 350 GB of activations for 1 million nodes in fp32, about 3.5 TB for 10 million, halved by bf16. Neither fits on one GPU.
- With no halo, partitioned message passing is wrong near the cuts (error 0.26); a 2-hop halo is still wrong for 5 layers; a 5-hop halo is **exact**, at the cost of 52% extra nodes on these small partitions (less on larger ones, because halos grow with the partition surface, not its volume).
- Low-precision running sums stall once the total is large compared with each contribution, and bf16 cannot resolve a 37.5 Pa difference between two absolute pressures; fp16 overflows on them entirely.
- Checkpointing cuts the activations kept for the backward pass almost tenfold for 12 blocks, in exchange for recomputing each block once.

## In an interview

**Thirty-second answer to "How much GPU memory does a GNN need on a 1-million-node mesh, and how would you scale it to 10 million nodes?"** "A tet mesh has about 14 directed edges per node, so 14 million edges; one 128-wide float32 tensor over them is about 7 GB, a layer stores a few of them, and 15 layers come to a few hundred gigabytes of activations, so it does not fit on one GPU. For 10 million nodes I would partition the graph across GPUs with halos as deep as the number of layers, so messages are exact, use multiscale message passing to cut the layer count, checkpoint each layer, keep activations in bf16 with fp32 accumulations, and subsample nodes in training if the model allows. Or switch to a latent-token architecture like AB-UPT that samples points and avoids full-mesh message passing."

**Likely follow-ups**

- *Why are halos needed and how deep?* Messages near cuts need neighbours from other partitions; L layers need L hops for exactness.
- *When does mixed precision hurt?* Long accumulations, small differences of large values, coordinates and derivatives; accumulate in fp32 and normalize fields.
- *Data or model parallel?* Data parallel if a sample fits; otherwise split the sample spatially before splitting the model.

## References

- [X-MeshGraphNet: Scalable Multi-Scale Graph Neural Networks for Physics Simulation (Nabian et al.)](https://arxiv.org/abs/2411.17164) and [MultiScale MeshGraphNets](https://arxiv.org/abs/2210.00612)
- [AB-UPT](https://arxiv.org/abs/2502.09692), [Universal Physics Transformers](https://arxiv.org/abs/2402.12365) and [Transolver++](https://arxiv.org/abs/2502.02414)
- [Training Deep Nets with Sublinear Memory Cost (Chen et al.)](https://arxiv.org/abs/1604.06174)
- [Mixed Precision Training (Micikevicius et al.)](https://arxiv.org/abs/1710.03740) and [ZeRO (Rajbhandari et al.)](https://arxiv.org/abs/1910.02054)
- [NVIDIA PhysicsNeMo (distributed training utilities)](https://github.com/NVIDIA/physicsnemo)
- [Dask](https://www.dask.org/), [Apache Spark](https://spark.apache.org/) and [Ray](https://www.ray.io/)
- [PhysicsX ML Engineer job description](https://job-boards.greenhouse.io/physicsx/jobs/4880947101)
