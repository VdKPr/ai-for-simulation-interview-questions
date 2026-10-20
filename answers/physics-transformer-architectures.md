# Physics Transformer Architectures: Transolver, UPT, AB-UPT, DoMINO and GeoTransolver

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** These are the models behind most current results on large 3D simulation data such as car aerodynamics. All avoid attention over millions of raw points. **Transolver** (Tsinghua, ICML 2024) softly assigns every mesh point to a few dozen learned **slices** of similar physical state, turns each slice into a token, attends among the tokens and broadcasts back; its cost is linear in the number of points. **Universal Physics Transformers (UPT)** (JKU Linz and Emmi AI, NeurIPS 2024) encode any mesh or particle set into a fixed number of **latent tokens**, evolve the physics in that latent space, and decode at any query point with cross-attention. **AB-UPT** (Emmi AI, 2025) splits the model into **geometry, surface and volume branches** and uses **anchor attention**: a few thousand anchor points attend to each other, and every other point only cross-attends to them; it trains on DrivAerML-scale meshes (about 140 million cells) on one GPU and can produce a **divergence-free** vorticity field by construction. **DoMINO** (NVIDIA, 2025) learns a global geometry encoding on a grid around the car and predicts each point from a **local** encoding gathered with multi-scale stencils around it. **GeoTransolver** (NVIDIA, 2025) combines Transolver's physics-attention with cross-attention to a shared geometry and boundary-condition context built from DoMINO-style multi-scale ball queries.

**On this page**

- [Transolver and Physics-Attention](#transolver-and-physics-attention)
- [Universal Physics Transformers](#universal-physics-transformers)
- [AB-UPT: anchor tokens and divergence-free fields](#ab-upt-anchor-tokens-and-divergence-free-fields)
- [DoMINO](#domino)
- [GeoTransolver](#geotransolver)
- [Separate surface and volume branches](#separate-surface-and-volume-branches)
- [Comparison](#comparison)
- [Code: physics-attention, anchor attention and a divergence-free output](#code-physics-attention-anchor-attention-and-a-divergence-free-output)
- [In an interview](#in-an-interview)

## Transolver and Physics-Attention

**Idea.** Mesh points are just a sampling of an underlying continuous physical state. Many points share similar states: on a car, the front bumper, windshield base and license plate all see stagnation-like flow, even though they are not neighbours. Transolver's **Physics-Attention** learns to group points by physical state and attends among the groups.

**The four steps** (one head; N points with C-dimensional features x_i, M slices, typically 32 to 64):

1. **Slice.** A pointwise linear layer and a softmax over the M slices give each point a membership vector: w_i = softmax(Project(x_i)), with Σ_j w_ij = 1. The softmax makes assignments low-entropy, so points commit to a few slices.
2. **Aggregate into tokens.** Each slice becomes a token, the membership-weighted mean of point features: z_j = Σ_i w_ij x_i / Σ_i w_ij.
3. **Attend among tokens.** Standard softmax attention over the M tokens: z' = softmax(qkᵀ/√C) v. This costs M² instead of N².
4. **Deslice.** Broadcast back to points with the same weights: x'_i = Σ_j w_ij z'_j.

Total cost per layer is O(NMC + M²C): **linear in N**. Because slice weights come from features, not coordinates, slices can be **spatially disconnected**: the paper's visualizations show the windshield, license plate and headlights of a car assigned to the same slice, and shock and wake regions of an airfoil picked out. The model is otherwise a standard transformer (layer norms, MLPs) with Physics-Attention in place of self-attention.

**Results.** Transolver reported a 22% relative gain over the previous state of the art across six standard benchmarks (elasticity, plasticity, airfoil, pipe, Navier-Stokes, Darcy) and strong results on car and airfoil design tasks, with better out-of-distribution generalization.

**Transolver++** (2025) scaled this to **million-point** meshes on a single GPU with a parallelism framework and a local adaptive mechanism, reporting a 13% gain on six benchmarks and over 20% on million-scale industrial simulations (cars, 3D aircraft). AB-UPT's authors note that processing 2.5 million cells with Transolver++ used four GPUs with sequence parallelism, and estimate by linear extrapolation that about 256 GPUs would be needed for a 140-million-cell mesh.

## Universal Physics Transformers

**Problem.** Models are usually tied to one discretization: grids (CNNs, FNO), meshes (GNNs) or particles (GNS). UPT (Alkin et al., NeurIPS 2024) handles meshes and particles with one architecture and keeps the cost independent of the number of points after encoding.

**Architecture.**

1. **Encoder.** Points (mesh cells or particles) are aggregated onto a sampled subset of **supernodes** by message passing within a radius (supernode pooling), then a Perceiver-style cross-attention compresses the supernodes into a **fixed number of latent tokens**.
2. **Approximator.** A transformer advances the latent tokens in time: the dynamics are **propagated entirely in latent space**, without decoding to the mesh at every step.
3. **Decoder.** A query point (any position, any time) cross-attends to the latent tokens and outputs the field there, acting as a **conditional neural field**.
4. **Inverse encoding and decoding objectives** keep the latent space well behaved: the latent must be decodable back to the input, and encodable from the output, so latent rollouts stay consistent.

**Demonstrated on** mesh-based transient fluid simulations, steady-state Reynolds-averaged Navier-Stokes, and Lagrangian (particle) dynamics, with no fixed grid or particle structure in the latent space.

**Why it matters.** UPT showed that a fixed-size latent space plus point-query decoding gives a model whose training and inference cost does not grow with mesh size beyond the encoder, the basis on which AB-UPT was built.

## AB-UPT: anchor tokens and divergence-free fields

AB-UPT ("Anchored-Branched Universal Physics Transformers", Alkin, Bleeker et al., Emmi AI, 2025) targets industrial car aerodynamics, where volume meshes exceed 100 million cells but datasets have only hundreds of samples.

**Three design choices.**

1. **Multi-branch architecture.** Separate **geometry**, **surface** and **volume** branches. The geometry branch encodes the input geometry (which need not be the simulation mesh) into a reduced set of tokens via supernode pooling; the surface and volume branches cross-attend to those geometry tokens, self-attend within the branch, and cross-attend to each other to model surface-volume interactions. This **decouples geometry encoding from field prediction**. See [Separate surface and volume branches](#separate-surface-and-volume-branches).
2. **Anchor attention.** In each branch, a small random subset of points are **anchor tokens** that self-attend and carry the physical context; all other points are **query tokens** that only cross-attend to the anchors. Consequences:
   - cost is quadratic only in the number of anchors and linear in the number of queries;
   - each query's prediction is independent of the other queries, so inference covers 10⁸ points in chunks;
   - training on random subsets of anchors and queries turns each simulation into many training examples;
   - the number of anchors sets the model's computational capacity, which can be changed at inference.
3. **Divergence-free formulation.** Because a query's output is a smooth function of its coordinates (a conditional neural field), differential operators can be applied to it with automatic differentiation. Vorticity ω = ∇ × u is divergence-free by definition, so AB-UPT predicts a field whose **curl** is the output; ∇·(∇×A) = 0 holds exactly, by construction, with no penalty term. The paper reports that this hard constraint did not degrade accuracy. See [Hard vs soft constraints](hard-vs-soft-constraints.md).

**Results reported.** State-of-the-art surface and volume field accuracy on automotive CFD datasets with meshes from 33 thousand to 150 million cells, including DrivAerML (hybrid RANS-LES, about 140 million cells); near-perfect drag and lift coefficients computed from predicted surface fields; training on a **single GPU in under a day**; predictions of surface and volume fields within seconds; and inference **from the CAD geometry alone**, without generating a CFD mesh. No input features beyond coordinates were needed (normals or SDF can be added). A follow-up applied AB-UPT to the Luminary SHIFT-SUV and SHIFT-Wing datasets with similar conclusions.

## DoMINO

DoMINO ("Decomposable Multi-scale Iterative Neural Operator", Ranade et al., NVIDIA, 2025) is a point-cloud model built in NVIDIA Modulus (now PhysicsNeMo).

**Architecture.**

1. **Input:** the 3D surface mesh (STL) of the geometry and a bounding box defining the computational domain.
2. **Global geometry encoding:** the surface point cloud is projected onto a structured m × m × m × f grid over the domain, and a **multi-resolution, iterative** scheme propagates geometry information through the volume to capture short- and long-range dependencies.
3. **Local encoding:** the global encoding is dense and high-dimensional, but the solution at a point depends mostly on its neighbourhood. For each sampled query point (on the surface or in the volume), DoMINO builds a **dynamic computational stencil** of nearby points at several scales and extracts a local encoding from the global one.
4. **Prediction:** a network maps the local encoding and the point's position to the solution fields there (surface pressure and wall shear; volume velocity, pressure and turbulence quantities).

**Results reported** on DrivAerML (hybrid RANS-LES; designs whose drag ranges from 300 to 600 N): accurate surface and volume fields, design trends and drag, in- and out-of-distribution, evaluated with engineering metrics. Surface predictions validated on a **uniformly sampled point cloud** instead of the simulation mesh showed that the model does not depend on the mesh it was trained on.

## GeoTransolver

GeoTransolver (Adams, Ranade et al., NVIDIA, 2025) combines the two previous ideas.

- Each block's attention is replaced by **GALE**: physics-aware self-attention over learned state slices (as in Transolver) **plus cross-attention to a shared context** of geometry, global parameters and boundary conditions.
- The context is computed once from **multi-scale ball queries** around the geometry (inspired by DoMINO) and reused in every block, so geometry information is re-injected throughout the network instead of only at the input.
- It is available in PhysicsNeMo and was benchmarked on DrivAerML and Luminary's SHIFT-SUV and SHIFT-Wing against DoMINO, Transolver and published AB-UPT results, using drag and lift R² and relative L1 field errors. The authors report better accuracy, robustness to geometry and regime shifts, and data efficiency.
- A 2026 NVIDIA study applied GeoTransolver to **crash** (bumper beam and full-vehicle), and found one-shot prediction more accurate and cheaper than time-conditional or autoregressive variants. See [Crash simulation surrogates](crash-simulation-surrogates.md).

## Separate surface and volume branches

**Why separate them.**

1. **Different quantities.** Surface: pressure coefficient and wall shear stress, which give forces. Volume: velocity, pressure, vorticity and turbulence in the wake and around the body.
2. **Different point sets and resolutions.** A car surface has millions of faces; the volume mesh has 10-100 times more cells, concentrated in the boundary layer and wake.
3. **Different consumers.** Drag and lift need only the surface; wake analysis, cooling flows and comfort need the volume. A surface-only branch can be trained and served cheaply when only forces are needed.
4. **Different inputs at inference.** The surface is known from the STL; the volume points are just queries in space.

**How they interact.** A shared geometry encoding feeds both branches (AB-UPT's geometry tokens; DoMINO's global grid encoding; GeoTransolver's shared context). Cross-attention between the surface and volume branches lets the wake influence the base pressure and vice versa. Training uses a weighted sum of surface and volume losses, each normalized per field.

**Practical tips.**

- Normalize each output field separately and non-dimensionalize (C_p, u/U∞). See [Normalizing fields and coordinates](nondimensionalization-and-normalization.md#normalizing-fields-and-coordinates).
- Weight the surface loss by face area so that refined regions do not dominate.
- Compute drag and lift from the predicted surface fields with exact areas and normals and report them separately from field errors. See [Engineering metrics in recent car-aerodynamics papers](evaluation-metrics.md#engineering-metrics-in-recent-car-aerodynamics-papers).

## Comparison

| | Transolver | UPT | AB-UPT | DoMINO | GeoTransolver |
| --- | --- | --- | --- | --- | --- |
| Core mechanism | Attention among learned physical-state slices | Latent tokens + cross-attention decoder | Anchor self-attention, query cross-attention, three branches | Global grid encoding + local multi-scale stencils | Slices + cross-attention to shared geometry/BC context |
| Cost in N | Linear | Encoder linear; then fixed | Linear in queries; quadratic in anchors only | Linear in query points | Linear |
| Needs simulation mesh at inference | Uses the mesh points as input | No (any points) | No (CAD geometry suffices) | No (STL + query points) | No (point clouds) |
| Time-dependent | Yes (benchmarks) | Yes (latent rollout) | Steady (in the paper) | Steady | Steady (and crash, one-shot) |
| Hard physics constraints | No | No | Divergence-free option | No | No |
| Code | thuml/Transolver, Neural-Solver-Library | ml-jku/UPT | Emmi AI (Noether framework) | NVIDIA PhysicsNeMo | NVIDIA PhysicsNeMo |

## Code: physics-attention, anchor attention and a divergence-free output

Three building blocks, each with the property that matters checked: Transolver's Physics-Attention is permutation-equivariant and linear in N; AB-UPT-style anchor attention makes each query independent of the others, so predictions can be computed in chunks; and outputting the curl of a predicted vector potential gives an exactly divergence-free field.

```python
import torch, torch.nn as nn

class PhysicsAttention(nn.Module):
    """Transolver's Physics-Attention (one head): soft-assign N points to M slices, attend among
    the M slice tokens, and broadcast the result back to the points. Cost O(N M d + M^2 d)."""
    def __init__(self, d, n_slices=32):
        super().__init__()
        self.project = nn.Linear(d, n_slices)           # slice logits for every point
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
    def forward(self, x):                               # x: (batch, N, d)
        w = torch.softmax(self.project(x), dim=-1)      # (B, N, M): memberships, sum to 1 over slices
        z = torch.einsum("bnm,bnd->bmd", w, x) / (w.sum(1)[..., None] + 1e-6)   # slice tokens: weighted means
        q, k, v = self.qkv(z).chunk(3, dim=-1)
        z = torch.softmax(q @ k.transpose(1, 2) / q.shape[-1] ** 0.5, dim=-1) @ v   # attention among M tokens
        return self.out(torch.einsum("bnm,bmd->bnd", w, z))                       # deslice back to points

class AnchorCrossAttention(nn.Module):
    """AB-UPT-style anchor attention (simplified): anchors self-attend; queries only cross-attend to anchors."""
    def __init__(self, d, heads=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True)
    def forward(self, anchors, queries):
        a = anchors + self.self_attn(anchors, anchors, anchors)[0]
        return queries + self.cross_attn(queries, a, a)[0]

torch.manual_seed(0)
d = 64
layer = PhysicsAttention(d)
x = torch.randn(1, 5000, d)
perm = torch.randperm(5000)
y = layer(x)
print("output", tuple(y.shape), "| permutation-equivariant:", torch.allclose(layer(x[:, perm]), y[:, perm], atol=1e-5))
n, m = 1_000_000, 32
print(f"N = {n:,}, M = {m}: slice-attention scores {m * m:,} vs full attention {n * n:.0e}")

attn = AnchorCrossAttention(d).eval()
anchors, queries = torch.randn(1, 512, d), torch.randn(1, 3000, d)
with torch.no_grad():
    full = attn(anchors, queries)
    chunked = torch.cat([attn(anchors, q) for q in queries.split(1000, dim=1)], dim=1)
print("query predictions independent of other queries (chunked == full):", torch.allclose(full, chunked, atol=1e-5))

# Divergence-free by construction: predict a vector potential A(x) and output omega = curl A.
field = nn.Sequential(nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 3))
pts = torch.rand(200, 3, requires_grad=True)
A = field(pts)
J = torch.stack([torch.autograd.grad(A[:, i].sum(), pts, create_graph=True)[0] for i in range(3)], 1)  # dA_i/dx_j
omega = torch.stack([J[:, 2, 1] - J[:, 1, 2], J[:, 0, 2] - J[:, 2, 0], J[:, 1, 0] - J[:, 0, 1]], 1)
div = sum(torch.autograd.grad(omega[:, i].sum(), pts, retain_graph=True)[0][:, i] for i in range(3))
print(f"max |div(curl A)| over 200 points: {div.abs().max().item():.2e} (mean |omega| {omega.abs().mean().item():.2e})")
```

Output:

```
output (1, 5000, 64) | permutation-equivariant: True
N = 1,000,000, M = 32: slice-attention scores 1,024 vs full attention 1e+12
query predictions independent of other queries (chunked == full): True
max |div(curl A)| over 200 points: 2.24e-08 (mean |omega| 6.61e-02)
```

The divergence is zero up to single-precision round-off, for an untrained network: the constraint comes from the architecture, not from training. In a real model the vector potential would be the output of the query decoder, conditioned on the anchor tokens.

## In an interview

**Thirty-second answer to "Explain Transolver's Physics-Attention."** "Instead of attending over a million mesh points, Transolver learns to group them. A linear layer and a softmax give every point soft membership weights over, say, 32 slices; each slice token is the weighted mean of its points' features; attention runs among the 32 tokens; and the result is broadcast back to points with the same weights. Cost is linear in the number of points, and the slices end up grouping points in similar physical states, like all the stagnation regions on a car, even when they are far apart."

**Thirty-second answer to "What does AB-UPT add?"** "Three things: separate geometry, surface and volume branches linked by cross-attention; anchor attention, where a few thousand anchor points self-attend and every other point only cross-attends to them, so cost is linear in the number of query points and predictions can be chunked over 100-million-cell meshes; and, because the decoder is a neural field, a divergence-free vorticity output via a curl. It trains on DrivAerML on one GPU in under a day and can predict from CAD alone."

**Likely follow-ups**

- *How does DoMINO differ?* It is not attention-based at its core: a global grid encoding of the geometry plus local multi-scale stencils around each query point.
- *Why separate surface and volume branches?* Different fields, point sets and resolutions, and forces need only the surface.
- *Which would you start with on a new dataset?* A strong simple baseline first, then whichever of these has a maintained implementation for the data size (PhysicsNeMo provides DoMINO and Transolver-type models), compared on the same split and metrics.

## References

- [Transolver: A Fast Transformer Solver for PDEs on General Geometries (Wu et al., ICML 2024)](https://arxiv.org/abs/2402.02366) and [code](https://github.com/thuml/Transolver)
- [Transolver++ (Luo et al.)](https://arxiv.org/abs/2502.02414)
- [Universal Physics Transformers (Alkin et al., NeurIPS 2024)](https://arxiv.org/abs/2402.12365) and [code](https://github.com/ml-jku/UPT)
- [AB-UPT (Alkin, Bleeker et al.)](https://arxiv.org/abs/2502.09692) and [AB-UPT for Automotive and Aerospace Applications](https://arxiv.org/abs/2510.15808)
- [DoMINO (Ranade et al.)](https://arxiv.org/abs/2501.13350)
- [GeoTransolver (Adams et al.)](https://arxiv.org/abs/2512.20399)
- [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) and [Neural-Solver-Library](https://github.com/thuml/Neural-Solver-Library)
