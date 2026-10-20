# Transformers for Physics

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Attention computes, for every point, a weighted sum over all other points with weights that depend on the data. That is a learned, data-dependent **kernel integral**, so attention can be read as a **neural operator**: as the number of sampled points grows, it converges to an integral over the domain, and the same weights apply at any resolution. It also gives every point a **global receptive field** in one layer, which suits physics with long-range coupling such as pressure in incompressible flow. The obstacle is cost: exact attention over N points needs N² scores per head and layer, and a million-point car mesh would need about 10¹² of them, 2 TB in half precision. Physics transformers therefore attend over a **small set of tokens** rather than all points (Transolver's learned slices, UPT's latent tokens, AB-UPT's anchors), or use **linear (Galerkin-type) attention** that never forms the N × N matrix. Positions, geometry and boundary conditions enter as coordinate embeddings, geometry tokens and conditioning layers. Transformers tend to win when data is plentiful and meshes are huge; GNNs keep an edge with small data and strongly local physics.

**On this page**

- [Is attention a neural operator?](#is-attention-a-neural-operator)
- [Why vanilla attention fails on a million points](#why-vanilla-attention-fails-on-a-million-points)
- [Linear and Galerkin-type attention](#linear-and-galerkin-type-attention)
- [Encoding positions and boundary conditions](#encoding-positions-and-boundary-conditions)
- [Do scaling laws hold for neural surrogates?](#do-scaling-laws-hold-for-neural-surrogates)
- [GNN or transformer?](#gnn-or-transformer)
- [Code: attention as an integral, and linear attention](#code-attention-as-an-integral-and-linear-attention)
- [In an interview](#in-an-interview)

## Is attention a neural operator?

**The neural-operator layer.** A neural operator layer has the form

```math
v'(x) = \sigma\Big(W v(x) + \int_\Omega \kappa\big(x, y, v(x), v(y)\big)\, v(y)\, dy\Big)
```

a pointwise linear map plus an integral against a kernel κ. See [What operator learning is](neural-operators.md#what-operator-learning-is).

**Attention has the same structure.** With queries q(x) = W_Q v(x), keys k(y) = W_K v(y) and values W_V v(y), softmax attention over N sampled points y_j is

```math
v'(x) = \sum_{j=1}^{N} \frac{\exp\big(q(x)\cdot k(y_j)/\sqrt{d}\big)}{\sum_{l}\exp\big(q(x)\cdot k(y_l)/\sqrt{d}\big)}\, W_V v(y_j)
\;\xrightarrow{N\to\infty}\;
\frac{\int_\Omega \exp\big(q(x)\cdot k(y)/\sqrt{d}\big)\, W_V v(y)\, dy}{\int_\Omega \exp\big(q(x)\cdot k(y)/\sqrt{d}\big)\, dy}
```

for points sampled uniformly. The numerator and the denominator are both Monte Carlo estimates of integrals, so attention is a **normalized kernel integral operator** whose kernel depends on the input function itself (a nonlinear operator). Kovachki et al.'s neural-operator review made this observation explicit, and Cao's Galerkin Transformer (NeurIPS 2021) analyzed attention as a projection in Hilbert space.

**What follows from it.**

1. **Discretization consistency.** Because attention approximates an integral, the same trained weights can be applied to a different number or arrangement of points, and the output converges as sampling gets denser. The [code below](#code-attention-as-an-integral-and-linear-attention) shows the error against the continuous operator falling from 0.096 with 100 points to 0.0035 with 100,000.
2. **Non-uniform sampling needs care.** The Monte Carlo argument assumes uniform samples. A mesh refined near a wall over-weights the wall region unless the sum is weighted by cell volume or area (the same issue as [remeshing](geometry-representations.md#why-remeshing-can-change-a-prediction)).
3. **Cross-attention is an operator from one domain to another**: queries at arbitrary output points attend to an encoded input function. This is how OFormer, UPT and AB-UPT decode at any resolution, acting as a conditional **neural field**.
4. **It is not automatically a good operator.** Positional information must be provided (attention itself is permutation-equivariant and knows nothing about where points are), and attending to raw points is expensive, which is what the rest of this page is about.

## Why vanilla attention fails on a million points

**Cost.** For N points with feature width d, one attention head computes an N × N score matrix: O(N²d) operations and O(N²) memory for the scores.

| Points N | Scores per head and layer | fp16 memory for one score matrix |
| --- | --- | --- |
| 10⁴ (2D benchmark) | 10⁸ | 0.2 GB |
| 10⁶ (a car surface) | 10¹² | 2 TB |
| 10⁸ (a car volume mesh) | 10¹⁶ | 20 PB |

Industrial CFD meshes reach 10⁸ cells; DrivAerML's hybrid RANS-LES runs use about 140 million volume cells.

**FlashAttention** computes exact attention in tiles without storing the N × N matrix, which removes the memory problem, but the compute is still N²: about 10¹² multiply-adds per head per layer at a million points, which is too slow for training on many samples.

**Beyond cost, three modelling problems.**

1. **Most pairwise interactions are irrelevant.** Two points on opposite sides of a car rarely need a direct interaction; the information can be summarized.
2. **Attention among raw points can be hard to learn.** Transolver's authors argue that the many individual mesh points "overwhelm" attention, and that attending to aggregated **physical states** is both cheaper and more informative.
3. **Training data has few samples but many points.** A dataset might have 500 simulations of 10⁸ cells each. Processing every cell of every sample at every step is wasteful; sampling subsets of points is both cheaper and a strong regularizer.

**The standard workarounds.**

| Strategy | Idea | Examples |
| --- | --- | --- |
| Physics-attention slices | Softly assign points to M learned slices of similar physical state; attend among M slice tokens; broadcast back. Linear in N | Transolver, Transolver++, GeoTransolver |
| Latent tokens | Encode N points into M ≪ N latent tokens; process them with full attention; decode at any query points with cross-attention | UPT, OFormer-style encoders |
| Anchor tokens | A random subset of M points self-attends; every other point only cross-attends to the anchors, so query cost is linear | AB-UPT |
| Linear attention | Change the order of matrix products so no N × N matrix is formed: O(N d²) | Galerkin Transformer, GNOT, OFormer |
| Local or windowed attention | Restrict attention to neighbourhoods or windows; combine with downsampling | Point Transformer, Erwin, Swin-style models |
| Point-based neural operators | Local geometric context at each query point, multiscale | DoMINO |

The details of Transolver, UPT, AB-UPT, DoMINO and GeoTransolver are on [Physics transformer architectures](physics-transformer-architectures.md).

## Linear and Galerkin-type attention

**The trick.** Drop the softmax and normalize differently. Then attention is a product of three matrices, and the order of multiplication can be changed:

```math
\underbrace{(QK^\top)}_{N\times N}V \;=\; Q\,\underbrace{(K^\top V)}_{d\times d}
```

The left side costs O(N²d); the right side costs O(Nd²). For N = 10⁶ and d = 64 that is a factor N/d ≈ 15,000.

**Fourier-type and Galerkin-type attention (Cao, 2021).** Cao proved that the softmax is sufficient but not necessary for attention to approximate operators, and studied two softmax-free variants with layer normalization applied to the queries and keys (Fourier-type) or the keys and values (Galerkin-type):

- **Fourier-type:** z = (Q̃ K̃ᵀ) V / N, still quadratic, interpreted as a Fredholm integral with a learned kernel.
- **Galerkin-type:** z = Q (K̃ᵀ Ṽ) / N, linear in N. Each column of K̃ and Ṽ is a learned basis function sampled at the N points, and K̃ᵀṼ / N is a matrix of their inner products (a Monte Carlo approximation of ∫ k_i(y) v_j(y) dy). The output is a linear combination of the value basis with coefficients set by the queries: a **Petrov-Galerkin projection** learned from data. Cao showed its approximation capacity per layer is comparable to such a projection, with an estimate independent of sequence length.

The Galerkin Transformer improved accuracy and training cost over softmax attention on Burgers' equation, interface Darcy flow and an inverse coefficient-identification problem.

**Related linear-attention operators.**

- **OFormer** (Li, Meidani and Barati Farimani): self-attention, cross-attention and pointwise MLPs with linear attention; few assumptions on how input and query points are sampled.
- **GNOT** (Hao et al.): "heterogeneous normalized attention" for multiple input functions and irregular meshes, and a **geometric gating** mechanism that acts like a soft domain decomposition for multiscale problems.
- General linear transformers (Performer, linear attention with kernel feature maps) have been used in the same way.

**Trade-offs.** Linear attention gives up the sharp, normalized weighting of softmax; the "kernel" has rank at most d, so it cannot represent arbitrary pairwise interactions. In practice it works well for smooth, global coupling and benefits from being combined with local mixing. Transolver's authors note that linear attention applied directly to massive point sets can still struggle to learn informative relations, which motivates attending over aggregated slices instead.

## Encoding positions and boundary conditions

Attention is permutation-equivariant: without extra information it cannot tell where a point is. Everything physical has to be supplied.

**Positions.**

- **Fourier features or sinusoidal embeddings** of (x, y, z): γ(x) = [sin(2πBx), cos(2πBx)] with several frequencies. Plain coordinates fed to an MLP suffer from spectral bias and miss fine detail. See [Fourier features and positional encodings](spectral-bias-and-fourier-features.md#fourier-features-and-positional-encodings). AB-UPT embeds coordinates with the transformer's sinusoidal positional encoding.
- **Relative positions** inside local attention (Point Transformer), or **rotary embeddings** extended to 3D.
- **Normalized frames:** non-dimensionalize coordinates (divide by the car length) and fix the axis convention, so embeddings mean the same thing across designs.

**Geometry.**

- Per-point features: surface normal, signed distance to the wall, curvature, a surface/volume flag. See [Signed distance functions](geometry-representations.md#signed-distance-functions).
- A **separate geometry branch** that encodes the surface into tokens, which the surface and volume branches cross-attend to (AB-UPT, GeoTransolver's shared geometry context).

**Boundary conditions and global parameters** (inlet speed, yaw angle, Reynolds number, load magnitude, material).

- **Extra tokens** carrying the condition, which all points can attend to.
- **Conditioning layers:** FiLM (scale and shift features by functions of the condition) or adaptive layer normalization (as in diffusion transformers), applied in every block. Poseidon, for example, conditions its layer norms on time.
- **Per-point boundary features:** a one-hot boundary type (wall, inlet, outlet, symmetry) and the prescribed value on boundary points, as in MeshGraphNets node types.
- **Hard constraints** in the output layer where possible (prescribed values on Dirichlet boundaries, divergence-free fields). See [Hard vs soft constraints](hard-vs-soft-constraints.md).

**Training on subsets.** Because each query point's prediction depends only on the encoded context, models such as UPT and AB-UPT train on random subsets of points per step (many effective training examples from few simulations) and predict all points at inference in chunks.

## Do scaling laws hold for neural surrogates?

**Background.** In language and vision, test loss falls as a power law in model size, data and compute (Kaplan et al., Hoffmann et al.) until an irreducible floor. Whether the same holds for physics surrogates matters because data costs simulator time.

**What the evidence shows (2023-2026).**

1. **Data scaling is real, with power-law regions.** A study of a GNN surrogate for double-delta wings (MF-VortexNet, 2025) found three regimes as the training set grew from 40 to 1,280 snapshots: a small-data region (the model makes a best guess), a power-law region where test error fell roughly as D^−0.61, and a plateau set by the fixed training budget. From the fit they estimated about eight samples per design dimension were sufficient under that budget.
2. **The data axis has a cost dimension that language does not have.** Setinek, Galletti and Brandstetter (2025) reformulated scaling laws for CFD surrogates in terms of the **compute budget for data generation** and the **mix of fidelities**: error fell steadily with budget, and at small budgets a mixture of cheap low-fidelity and expensive high-fidelity RANS runs beat spending everything on high fidelity (except for wall shear stress, where low fidelity did not help). A follow-up with Thuerey and Koehler (2026), over 7,000 training runs across six PDE problems and four architectures, found that optimal fidelity mixtures matched high-fidelity-only accuracy with on average about 3× and up to 6.3× less data-generation compute. See [Multi-fidelity learning](multi-fidelity-learning.md).
3. **Bigger pretrained models transfer better.** Subramanian et al. (2023) found that pretrained-then-fine-tuned operators reached a target accuracy with orders of magnitude fewer downstream samples, with gains growing with model size. Poseidon (2024) and DPOT (2024, up to 0.5B parameters) reported improvements with both model and data scale. See [Foundation models for physics](foundation-models-for-physics.md).
4. **Industrial meshes add "problem scale".** The AB-UPT authors argue that scale in CFD also means mesh resolution and simulation fidelity: few samples, each with 10⁸ cells. Architectures that sample points (anchors, latent tokens) turn one huge simulation into many training examples.

**What to say in an interview.** "Yes, in the sense that error keeps falling as a power law with more simulations over a useful range, and larger pretrained models transfer better. But data is the bottleneck: every sample costs solver hours, the curves flatten at a floor set by data fidelity and training budget, and with tens of simulations a small model with strong priors, or a GP on design parameters, often wins. I would measure a learning curve on my own data before buying a bigger model."

## GNN or transformer?

| | GNN (MeshGraphNets-style) | Transformer (Transolver / UPT / AB-UPT-style) |
| --- | --- | --- |
| Receptive field | Local; grows one hop per layer | Global in one layer, through tokens |
| Inductive bias | Mesh connectivity matches PDE stencils; strong locality | Weak locality; learns structure from data |
| Hardware efficiency | Gather/scatter on irregular memory; harder to keep GPUs busy | Dense matrix multiplies; efficient on GPUs |
| Resolution | Hop count grows with refinement; needs multiscale for large meshes | Token count fixed; decoders query any resolution |
| Needs a mesh at inference? | Usually (or a k-NN graph from points) | Often only points or the CAD surface |
| Data appetite | Does well with less data | Tends to benefit from more data and larger models |
| Time-dependent rollouts | Natural (local updates; MeshGraphNets, GNS) | Possible in latent space (UPT) |
| Best at | Local, transient, contact-rich dynamics; small data | Large steady 3D fields; global coupling; many points |

**How to decide.**

1. **Locality of the physics.** Contact, cloth, local plasticity and wave fronts favour message passing. Elliptic, global coupling (pressure) favours attention or multiscale.
2. **Mesh size.** At 10⁵ nodes either works; at 10⁷-10⁸, token-based transformers are far easier to train than full-mesh GNNs (which need partitioning and halo exchange; see [Scaling a GNN to 10 million nodes](scaling-to-large-meshes.md#scaling-a-gnn-to-10-million-nodes)).
3. **Data size.** With tens to a few hundred samples, try a GNN or a small model first; transformers shine with more data or pretraining.
4. **What is available at inference.** If customers send CAD or STL, a model that needs only surface points avoids meshing.
5. **Hybrids** are common and often best: local message passing or local attention for detail plus global tokens for long-range coupling.

## Code: attention as an integral, and linear attention

First, a single attention head with fixed query and key maps is applied to a function sampled at N random points; as N grows, its output converges to the continuous integral operator, which is what makes attention discretization-consistent. Second, Galerkin-type (softmax-free) attention gives exactly the same result whichever way the matrix product is bracketed, but one way is linear in N.

```python
import numpy as np

rng = np.random.default_rng(0)
u = lambda y: np.sin(2 * np.pi * y) + 0.5 * np.cos(6 * np.pi * y)     # the input function on [0, 1]
x_out = np.linspace(0, 1, 5)                                             # where the output is evaluated

def softmax_attention(x_query, y, values, scale=8.0):
    """One head with fixed query/key maps q(x) = scale * x, k(y) = y:
    output(x) = sum_j softmax_j(q(x) k(y_j)) v(y_j), a normalized kernel integral as N grows."""
    logits = scale * np.outer(x_query, y)
    w = np.exp(logits - logits.max(1, keepdims=True))
    return (w * values).sum(1) / w.sum(1)

fine = np.linspace(0, 1, 200001)
reference = softmax_attention(x_out, fine, u(fine))                     # the continuous operator, by quadrature
for n in [100, 1000, 10000, 100000]:
    errs = [np.abs(softmax_attention(x_out, y, u(y)) - reference).max() for y in rng.random((5, n))]
    print(f"N = {n:>6} random points: max deviation from the continuous operator {np.mean(errs):.4f}")

# Galerkin-type attention: (Q K^T) V / N equals Q (K^T V) / N, but the right side never forms an N x N matrix.
N, d = 4096, 32
Q, K, V = (rng.normal(size=(N, d)) for _ in range(3))
quadratic = (Q @ K.T) @ V / N
linear = Q @ (K.T @ V) / N
print("same result:", np.allclose(quadratic, linear))
print(f"multiply-adds: quadratic {2 * N * N * d:.2e}, linear {2 * N * d * d:.2e}")
for n_points in [1e4, 1e6, 1e8]:
    print(f"N = {n_points:.0e}: one N x N score matrix in fp16 = {n_points**2 * 2 / 1e9:,.1f} GB")
```

Output:

```
N =    100 random points: max deviation from the continuous operator 0.0955
N =   1000 random points: max deviation from the continuous operator 0.0448
N =  10000 random points: max deviation from the continuous operator 0.0159
N = 100000 random points: max deviation from the continuous operator 0.0035
same result: True
multiply-adds: quadratic 1.07e+09, linear 8.39e+06
N = 1e+04: one N x N score matrix in fp16 = 0.2 GB
N = 1e+06: one N x N score matrix in fp16 = 2,000.0 GB
N = 1e+08: one N x N score matrix in fp16 = 20,000,000.0 GB
```

The deviation shrinks roughly like 1/√N, the Monte Carlo rate: the attention layer is approximating one fixed operator, not memorizing a grid. The second part shows the two bracketings agree to machine precision while the linear one needs 128 times fewer operations at N = 4,096 (the ratio N/d), and the memory table is why exact attention over a full car mesh is not an option.

## In an interview

**Thirty-second answer to "Why does vanilla attention fail on a million mesh points, and what do physics transformers do instead?"** "Attention compares every point with every other, so a million points means 10¹² scores per head per layer: terabytes of memory, and even FlashAttention leaves the N-squared compute. Physics transformers compress first. Transolver softly assigns points to a few dozen learned slices of similar physical state and attends among those tokens; UPT encodes points into a fixed set of latent tokens and decodes anywhere with cross-attention; AB-UPT lets a few thousand anchor points attend to each other while all other points only cross-attend to the anchors; Galerkin-type linear attention reorders the matrix product to cost N d². All are linear in the number of points."

**Likely follow-ups**

- *Is attention a neural operator?* Yes: softmax attention is a Monte Carlo estimate of a normalized kernel integral whose kernel depends on the input, so it converges as sampling gets denser and applies at any resolution.
- *How do you feed boundary conditions?* As per-point boundary-type features, extra condition tokens, or FiLM/adaptive layer-norm conditioning; hard constraints where possible.
- *Do scaling laws hold?* Over useful ranges, error falls as a power law with data, and pretrained models transfer better with size, but data generation cost and fidelity set the limits.

## References

- [Choose a Transformer: Fourier or Galerkin (Cao, NeurIPS 2021)](https://arxiv.org/abs/2105.14995)
- [Neural Operator: Learning Maps Between Function Spaces (Kovachki et al.)](https://arxiv.org/abs/2108.08481)
- [Transformer for Partial Differential Equations' Operator Learning (OFormer; Li, Meidani & Barati Farimani)](https://arxiv.org/abs/2205.13671)
- [GNOT: A General Neural Operator Transformer for Operator Learning (Hao et al.)](https://arxiv.org/abs/2302.14376)
- [Transolver (Wu et al., ICML 2024)](https://arxiv.org/abs/2402.02366), [Universal Physics Transformers (Alkin et al.)](https://arxiv.org/abs/2402.12365), [AB-UPT (Alkin et al.)](https://arxiv.org/abs/2502.09692)
- [FlashAttention (Dao et al.)](https://arxiv.org/abs/2205.14135)
- [Towards Multi-Fidelity Scaling Laws of Neural Surrogates in CFD (Setinek, Galletti & Brandstetter)](https://arxiv.org/abs/2511.01830)
- [Compute-optimal data scaling for neural surrogates via multi-fidelity training (Setinek et al.)](https://arxiv.org/abs/2610.17221)
- [A Multi-fidelity Double-Delta Wing Dataset and Empirical Scaling Laws for GNN-based Aerodynamic Field Surrogates](https://arxiv.org/abs/2512.20941)
- [Towards Foundation Models for Scientific Machine Learning: Characterizing Scaling and Transfer Behavior (Subramanian et al.)](https://arxiv.org/abs/2306.00258)
- [ETH Zürich AISE: Attention as a Neural Operator (video)](https://www.youtube.com/watch?v=wJSgLRiU7D4)
