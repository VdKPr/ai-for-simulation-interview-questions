# Geodesic Convolutions and Shape Optimization

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A **geodesic convolution** applies a learned filter on a curved surface by measuring neighbourhoods *along the surface* instead of through 3D space. Masci et al. (2015) built **Geodesic CNNs (GCNN)**: around each surface point they construct local geodesic polar coordinates, extract a "patch" of features in those coordinates, and apply learned filters to it, which makes a CNN-like operation on a mesh. Baqué, Remelli, Fleuret and Fua (ICML 2018) used Geodesic CNNs to **emulate a fluid-dynamics simulator** on 3D shapes, then used the network as a **differentiable objective**: gradients of the predicted drag or lift with respect to the shape parameters drive a gradient-based optimizer. Remeshing every shape with a **polycube map** gave a regular structure that let the computation run on GPUs. They reported 5 to 20% better results than state-of-the-art methods on standard problems and handled cases earlier methods could not. Baqué, Fleuret and Fua went on to found Neural Concept, an EPFL spin-off whose products build on this line of work.

**On this page**

- [Why measure along the surface](#why-measure-along-the-surface)
- [Geodesic CNNs and their successors](#geodesic-cnns-and-their-successors)
- [Geodesic Convolutional Shape Optimization](#geodesic-convolutional-shape-optimization)
- [Surrogate-driven shape optimization in practice](#surrogate-driven-shape-optimization-in-practice)
- [Code: geodesic vs Euclidean neighbourhoods](#code-geodesic-vs-euclidean-neighbourhoods)
- [Code: optimizing a shape through a surrogate](#code-optimizing-a-shape-through-a-surrogate)
- [In an interview](#in-an-interview)

## Why measure along the surface

Surface quantities (pressure, wall shear, heat flux, shell stress) live on a 2D surface embedded in 3D. Two points can be close in 3D but far apart on the surface: the inside and outside of a thin wall, the two sides of a trailing edge, adjacent fins of a heat sink, the upper and lower legs of a U-bend. Physically, they barely interact through the surface; the flow on the pressure side of a wing is not the flow on the suction side.

A filter that gathers neighbours inside a 3D ball mixes those unrelated regions. A filter defined with **geodesic distance**, the shortest path along the surface, does not. Geodesic neighbourhoods are also **intrinsic**: they do not change when the surface bends without stretching (an isometry), which makes features robust to pose and articulation, the original motivation in shape analysis.

The demo [below](#code-geodesic-vs-euclidean-neighbourhoods) builds a thin U-shaped strip: the two free ends are 0.4 apart in 3D but 2.63 apart along the surface, and a 3D ball around one end picks up nearly twice as many vertices as the geodesic disc, half of them on the wrong leg.

## Geodesic CNNs and their successors

**Geodesic CNN (Masci, Boscaini, Bronstein, Vandergheynst, 2015).**

1. Around each vertex, build a **local geodesic polar coordinate system**: radial coordinate ρ = geodesic distance, angular coordinate θ = direction of the geodesic leaving the vertex.
2. Interpolate the input features into a fixed set of (ρ, θ) bins: this is the **patch operator**, the analogue of the k × k pixel window of an image CNN.
3. Multiply the patch by learned filter weights and sum, as in an ordinary convolution.
4. **Angular ambiguity:** a surface has no canonical "up" direction, so the filter's orientation is arbitrary. GCNN applies the filter at all rotations and takes the maximum (angular max-pooling).

Cascades of such layers learned shape descriptors for retrieval and correspondence.

**Successors.**

- **Anisotropic CNNs (ACNN)** used anisotropic heat kernels aligned with principal curvature directions, removing the angular ambiguity.
- **MoNet (Monti et al., 2017)** generalized the idea: define pseudo-coordinates for each neighbour (geodesic polar coordinates, or degree-based coordinates on a general graph) and learn Gaussian kernel weights over them. GCNN, ACNN and several graph CNNs become special cases of one framework.
- **Mesh-based GNNs with edge features** (MeshGraphNets) and point-based networks largely replaced explicit geodesic patches in physics surrogates. Relative positions and normals on the edges give the network similar local-geometry information without computing geodesic coordinates. See [Node and edge features](gnns-and-meshgraphnets.md#node-and-edge-features).

## Geodesic Convolutional Shape Optimization

**The problem Baqué et al. addressed.** Aerodynamic shape optimization with a CFD solver in the loop is so expensive that engineers either try a few hand-designed shapes or restrict themselves to shapes described by a handful of parameters.

**Their approach.**

1. **Generate training data:** run a CFD solver on many shapes and record the quantities of interest (drag, lift, pressure).
2. **Remesh every shape onto a common structure with a polycube map,** which maps the surface onto a set of cube faces. Geodesic convolutions on this regular parametrization can be computed efficiently on GPUs rather than CPUs, which the paper calls the key to making the approach practical.
3. **Train a Geodesic CNN to emulate the simulator:** shape in, predicted aerodynamic quantities out.
4. **Optimize through the network.** Because the network is differentiable, the predicted objective has a gradient with respect to the shape parameters (the mesh deformation). A gradient-based optimizer follows it, with constraints such as a fixed volume or bounding box.
5. **Check the result** with the real solver.

**Results reported.** On standard problems, 5 to 20% better than state-of-the-art methods, and the approach worked on cases (complex 3D shapes, many free shape parameters) that previous methods could not handle. Neural Concept's early proof of concept, shown by EPFL in 2019, was an AI-designed aerodynamic bike.

**Why it matters for interviews.** The paper is an early, clean example of the full loop that the AI-for-simulation industry now sells: a geometric deep-learning surrogate, trained on simulations, used as a differentiable proxy for design optimization, with the real solver as the final check.

## Surrogate-driven shape optimization in practice

1. **Parametrize the shape.** CAD parameters, free-form deformation (FFD) boxes, mesh-vertex displacements with smoothing, or a latent code from a shape autoencoder. The parametrization decides which shapes are reachable and whether they are manufacturable.
2. **Use surrogate gradients**, from autograd through the network, or a gradient-free optimizer (genetic algorithm, CMA-ES, Bayesian optimization) when the parametrization is not differentiable.
3. **Constrain** volume, packaging, thickness and manufacturing rules. See [Keeping designs manufacturable](surrogate-based-optimization.md#keeping-designs-manufacturable).
4. **Stay inside the training distribution.** An optimizer is an adversary for the surrogate: it searches for inputs where the surrogate says the objective is good, and the easiest such inputs are often where the surrogate is wrong. Bound the parameters to the trained range, penalize uncertainty, or use an out-of-distribution score. See [Why optimizers exploit surrogates](surrogate-based-optimization.md#why-optimizers-exploit-surrogates).
5. **Verify the best candidates with the solver**, add them to the training set, retrain, and repeat. This is active learning in the optimization loop.

| | Adjoint solver | Surrogate gradient |
| --- | --- | --- |
| Cost of one gradient | About one extra solve | Milliseconds after training |
| Accuracy | Exact gradient of the discretized model | Only as good as the surrogate, and only in its training range |
| Up-front cost | Implementing the adjoint | Hundreds to thousands of training simulations |
| Best for | Final refinement of one design | Exploring many designs, interactive design |

See [The adjoint method](adjoint-method.md) and [Surrogate gradients vs the adjoint](surrogate-based-optimization.md#surrogate-gradients-vs-the-adjoint).

## Code: geodesic vs Euclidean neighbourhoods

A thin strip bent into a U. Geodesic distance is approximated by shortest paths along mesh edges (Dijkstra), which overestimates the true geodesic slightly; exact methods (the heat method, fast marching) exist in libraries such as `potpourri3d` and `pygeodesic`.

```python
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

def u_strip(n_along=80, n_across=6, radius=0.2, length=1.0):
    """A thin strip bent into a U: two flat legs joined by a half-cylinder. Returns vertices and triangles."""
    total = 2 * length + np.pi * radius
    pts = []
    for s in np.linspace(0, total, n_along):                # arc length along the strip
        if s < length:                                      # lower leg
            x, z = length - s, -radius
        elif s < length + np.pi * radius:                   # bend
            phi = (s - length) / radius
            x, z = -radius * np.sin(phi), -radius * np.cos(phi)
        else:                                               # upper leg
            x, z = s - length - np.pi * radius, radius
        for y in np.linspace(0, 0.2, n_across):
            pts.append((x, y, z))
    V = np.array(pts)
    idx = np.arange(n_along * n_across).reshape(n_along, n_across)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    return V, np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])

def edge_graph(V, F):
    """Sparse graph of mesh edges weighted by their length."""
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    w = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)
    G = sp.coo_matrix((w, (e[:, 0], e[:, 1])), shape=(len(V), len(V)))
    return (G + G.T).tocsr()

V, F = u_strip()
G = edge_graph(V, F)
i, j = 0, len(V) - 6                     # free end of the lower leg, and the vertex directly above it
d_geo = dijkstra(G, indices=i)
print(f"Euclidean distance {np.linalg.norm(V[i] - V[j]):.3f}   geodesic (along the surface) {d_geo[j]:.3f}")
ball = np.flatnonzero(np.linalg.norm(V - V[i], axis=1) < 0.45)
disc = np.flatnonzero(d_geo < 0.45)
print(f"vertices within 0.45 of the end: Euclidean ball {len(ball)}, geodesic disc {len(disc)}")
```

Output:

```
Euclidean distance 0.400   geodesic (along the surface) 2.628
vertices within 0.45 of the end: Euclidean ball 114, geodesic disc 58
```

The geodesic distance is 2 + π × 0.2 = 2.628, as expected. A layer that pools over the Euclidean ball would mix the lower and upper legs, which on a real part could be the two sides of a thin wall.

## Code: optimizing a shape through a surrogate

The full loop on a problem with a known answer. The "simulator" computes the perimeter and area of a closed curve r(θ) = 1 + Σ (a_k cos kθ + b_k sin kθ), k = 2, 3, 4. A small MLP surrogate is trained on 400 random designs, then gradient descent **through the surrogate** minimizes the perimeter at fixed area, with the coefficients kept inside the trained range. The isoperimetric inequality says the answer is a circle with perimeter 2π.

```python
import math
import torch, torch.nn as nn

torch.manual_seed(0)
K = torch.arange(2, 5).float()                              # shape modes k = 2, 3, 4
theta = torch.linspace(0, 2 * math.pi, 721)[:-1]

def radius(c):                                              # c: (..., 6) = cosine and sine coefficients
    a, b = c[..., :3], c[..., 3:]
    return 1 + (a[..., None] * torch.cos(K[:, None] * theta) + b[..., None] * torch.sin(K[:, None] * theta)).sum(-2)

def true_solver(c):
    """Stand-in for the expensive simulation: perimeter and area of the closed curve r(theta)."""
    r = radius(c)
    dr = torch.gradient(r, spacing=(theta[1] - theta[0]).item(), dim=-1)[0]
    dth = 2 * math.pi / theta.numel()
    return (torch.sqrt(r ** 2 + dr ** 2) * dth).sum(-1), (0.5 * r ** 2 * dth).sum(-1)

lo, hi = -0.15, 0.15                                        # the design space covered by training data
C = lo + (hi - lo) * torch.rand(400, 6)                     # 400 "simulations"
P, A = true_solver(C)
Y = torch.stack([P, A], 1)
mu, sd = Y.mean(0), Y.std(0)
net = nn.Sequential(nn.Linear(6, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2))
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
for step in range(3000):
    loss = ((net(C) - (Y - mu) / sd) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
surrogate = lambda c: net(c) * sd + mu

start = torch.tensor([[0.12, -0.1, 0.08, 0.1, 0.05, -0.12]])    # a lumpy starting shape
c = start.clone().requires_grad_(True)
opt_c = torch.optim.Adam([c], lr=0.01)
for it in range(300):
    p, a = surrogate(c)[0]
    objective = p + 50 * (a - math.pi) ** 2                  # minimise perimeter at fixed area (penalty)
    opt_c.zero_grad(); objective.backward(); opt_c.step()
    with torch.no_grad():
        c.clamp_(lo, hi)                                     # stay inside the trained design space
with torch.no_grad():
    p_s, _ = surrogate(c)[0]
    p_t, a_t = true_solver(c)
    p0, a0 = true_solver(start)
print(f"start:     true perimeter {p0.item():.4f}  area {a0.item():.4f}")
print(f"optimised: surrogate perimeter {p_s.item():.4f}  true perimeter {p_t.item():.4f}  true area {a_t.item():.4f}")
print(f"circle of the same area: perimeter {2 * math.pi:.4f}")
print(f"largest remaining shape coefficient {c.abs().max().item():.4f}")
```

Output:

```
start:     true perimeter 7.0560  area 3.2322
optimised: surrogate perimeter 6.3040  true perimeter 6.2846  true area 3.1419
circle of the same area: perimeter 6.2832
largest remaining shape coefficient 0.0121
```

The optimizer, guided only by surrogate gradients, finds an almost perfect circle: the true perimeter is within 0.02% of the optimum and the area constraint holds. Note the last two steps of the workflow: the result is **checked with the true solver**, and the surrogate's own estimate (6.3040) is 0.3% off, a reminder that the final numbers must come from the solver.

## In an interview

**Thirty-second answer to "What is geodesic convolution, and how was it used for shape optimization?"** "It is a convolution on a curved surface: around each point you build local geodesic polar coordinates, so neighbourhoods are measured along the surface, not through space, and you apply learned filters to that patch. Baqué and colleagues trained Geodesic CNNs to emulate a CFD solver on 3D shapes, remeshed with polycube maps so it ran on GPUs, and then used the network as a differentiable objective: gradients of predicted drag with respect to shape parameters drive the optimizer. They reported 5 to 20% better designs than prior methods, and the authors founded Neural Concept on this idea."

**Likely follow-ups**

- *Why not Euclidean neighbourhoods?* Points close in 3D but far along the surface (two sides of a thin wall) would be mixed.
- *What is the main risk of optimizing through a surrogate?* The optimizer exploits surrogate errors outside the training distribution; bound the design space, use uncertainty, and verify with the solver.
- *What replaced explicit geodesic patches in recent models?* Mesh GNNs with relative-position edge features, point-cloud networks, and transformers with geometry encodings.

## References

- [Geodesic Convolutional Shape Optimization (Baqué, Remelli, Fleuret & Fua, ICML 2018)](https://proceedings.mlr.press/v80/baque18a.html) ([arXiv](https://arxiv.org/abs/1802.04016))
- [Geodesic convolutional neural networks on Riemannian manifolds (Masci et al.)](https://arxiv.org/abs/1501.06297)
- [Geometric deep learning on graphs and manifolds using mixture model CNNs (MoNet; Monti et al.)](https://arxiv.org/abs/1611.08402)
- [Neural Concept Series A press release (company background)](https://www.neuralconcept.com/press-release/neural-concept-raises-series-a)
- [EPFL news: IC spin-off Neural Concept secures "seven-figure" investment](https://actu.epfl.ch/news/ic-spin-off-neural-concept-secures-seven-figure-in/)
