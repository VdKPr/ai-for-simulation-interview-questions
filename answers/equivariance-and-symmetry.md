# Equivariance and Symmetry

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A model is **invariant** to a transformation if its output does not change when the input is transformed: f(g·x) = f(x). It is **equivariant** if its output transforms in the matching way: f(g·x) = g·f(x). Physics has symmetries: the laws do not care where the origin is (translation), how the coordinate axes are oriented (rotation, reflection), how mesh nodes are numbered (permutation), or, in non-dimensional form, how big things are (similarity). Building the true symmetries into a model saves data and improves generalization. The catch is that a transformation is a symmetry of *your* problem only if it maps the **whole** problem, including boundary conditions, inflow direction, gravity and material orientation, onto another valid problem. Rotating a car while the wind stays fixed is not a symmetry; it is a different, yawed case. You need an **equivariant GNN** when the outputs are vectors or tensors in a frame-free physical setting (molecules, particles, structures under arbitrary orientation) and you cannot rely on a fixed frame. In most engineering datasets a natural frame exists (the vehicle frame, the gravity direction, the inflow), and **canonical orientation** in that frame is simpler and often better than trying to learn rotational invariance.

**On this page**

- [Invariance and equivariance](#invariance-and-equivariance)
- [What transforms how](#what-transforms-how)
- [Which symmetries matter in engineering problems](#which-symmetries-matter-in-engineering-problems)
- [When data augmentation is physically valid](#when-data-augmentation-is-physically-valid)
- [Ways to build symmetry into a model](#ways-to-build-symmetry-into-a-model)
- [When you need an equivariant GNN](#when-you-need-an-equivariant-gnn)
- [Code: checking equivariance numerically](#code-checking-equivariance-numerically)
- [Canonical orientation or not](#canonical-orientation-or-not)
- [In an interview](#in-an-interview)

## Invariance and equivariance

Let g be a transformation (a rotation, a translation, a permutation of node order), acting on inputs and outputs.

- **Invariant:** f(g·x) = f(x). Rotating a bracket does not change its maximum von Mises stress; renumbering mesh nodes does not change the drag.
- **Equivariant:** f(g·x) = g·f(x). Rotating the bracket rotates its displacement field; renumbering the nodes renumbers the predicted per-node values in the same way.

Invariance is the special case where the transformation acts trivially on the output. A model can be equivariant to some transformations and invariant to others: a graph network is typically permutation-equivariant for node outputs and permutation-invariant for global outputs.

**Why it helps:** a model that respects a symmetry does not have to learn it from data. It gets the right answer on every transformed version of every training example for free, which is a large effective increase in data, and it cannot make the physically impossible error of changing its answer when the coordinate system changes.

## What transforms how

| Quantity | Under a rotation R of the whole problem | Examples |
| --- | --- | --- |
| Scalars | Unchanged | Pressure, temperature, density, von Mises stress, drag *magnitude* in the rotated frame |
| Vectors | v → R v | Velocity, displacement, force, heat flux, surface normals |
| Rank-2 tensors | σ → R σ Rᵀ | Stress, strain, the conductivity tensor of an anisotropic material |
| Pseudovectors | Rotate like vectors, but flip sign under reflection | Vorticity, angular velocity, moments |

The last row matters for mirroring: reflecting a flow reflects velocity like an ordinary vector, but vorticity (a cross product) picks up an extra sign change.

## Which symmetries matter in engineering problems

- **Translation:** moving the whole problem does not change the physics. Use relative positions instead of absolute coordinates, or a consistent reference point.
- **Rotation and reflection:** hold only if *everything* rotates together, including loads, supports, the flow direction and gravity. Isotropic materials are rotation-invariant; composites and additively manufactured parts (with a build direction) are not.
- **Permutation:** the numbering of mesh nodes or points is arbitrary. Every model on meshes or point clouds must be permutation-equivariant or invariant; graph networks and PointNet are by construction. See [PointNet](point-cloud-learning.md#pointnet).
- **Mirror symmetry of a specific design:** a symmetric car at zero yaw allows a half model, but only for symmetric flow states.
- **Scale and similarity:** problems with the same dimensionless groups are similar. Non-dimensionalization exploits this. See [Why non-dimensionalize](nondimensionalization-and-normalization.md#why-non-dimensionalize).
- **Time translation:** for autonomous systems, dynamics do not depend on absolute time, only on the state.
- **Linearity and superposition** are not symmetries in this group sense, but they are structure worth building in for linear problems. See [Linear elasticity](stress-strain-and-linear-elasticity.md#linear-elasticity-the-full-problem).

## When data augmentation is physically valid

Augmentation creates extra training examples by transforming existing ones. It is valid only if the transformed example is a **true solution of a real problem**:

1. **Transform inputs and outputs consistently.** Rotate the geometry *and* rotate vector outputs (v → Rv) *and* transform tensor outputs (σ → RσRᵀ). Rotating the geometry while leaving the velocity vectors unchanged creates wrong labels.
2. **Transform the boundary conditions and external fields too.** If the flow comes from the −x direction and gravity points along −z, a rotated sample must have a rotated inflow and rotated gravity. Usually these are fixed in the dataset, so the rotation is *not* a symmetry of the problem.
3. **Respect material orientation.** For anisotropic materials, rotate the material axes with the part.
4. **Mirroring** is valid if you mirror everything, including flipping the sign of the normal component of vectors and handling pseudovectors correctly, and if the mirrored configuration is physically meaningful (no chirality-dependent physics, such as rotating machinery with a fixed spin direction).
5. **Do not augment across regimes.** Scaling a geometry changes the Reynolds number unless the speed or viscosity changes accordingly; in dimensional variables, a scaled car is a different flow.

**The common trap:** "rotate the car by 10 degrees to get more data". The wind still comes from the front, so the rotated car is a car at 10 degrees of yaw: a genuinely different case with different drag and side force. Using the original labels for it is wrong. Valid alternatives: small augmentations that do not change the physics (renumbering nodes, resampling surface points, translating the whole domain), or simulating real yawed cases.

## Ways to build symmetry into a model

| Approach | How | Trade-off |
| --- | --- | --- |
| **Data augmentation** | Transform inputs and targets consistently (above) | Cheap to implement; symmetry only approximate; data loading more complex |
| **Invariant features** | Use relative positions, distances, angles, local frames. MeshGraphNets encodes relative displacements on edges, which gives translation invariance | Simple; may throw away useful absolute information (gravity, inflow) unless it is given back as an input |
| **Canonicalization** | Align every shape to a reference frame (principal axes, a vehicle frame) before the model | Simple and keeps standard architectures; principal axes are ambiguous for near-symmetric shapes; see [Canonical orientation or not](#canonical-orientation-or-not) |
| **Equivariant architectures** | Layers that are exactly equivariant by construction (EGNN for E(n), tensor-field and spherical-harmonic networks for higher-order tensors) | Exact symmetry and better data efficiency where it truly applies; more complex, often slower |
| **Permutation invariance** | Sum, mean or max aggregation over sets and neighbourhoods | Built into GNNs, PointNet and transformers without positional order |

## When you need an equivariant GNN

**You probably need one when:**

- the problem has **no preferred frame**, and inputs arrive in arbitrary orientations: molecules and materials, particle systems, parts scanned or placed in random orientations, structures loaded in many directions relative to their own geometry;
- outputs are **vectors or tensors** (forces, displacements, stresses), so mere invariant features are not enough, and the orientation must be propagated correctly;
- **data is scarce** relative to the variety of orientations, so learning rotations from augmentation would cost too much data;
- **exactness matters:** for example, a learned interatomic potential must conserve angular momentum, or a learned constitutive model must be objective (frame-indifferent).

**You may not need one when:**

- the dataset has a natural frame that every sample shares: a car in the wind-tunnel frame, gravity along −z, a bracket always mounted the same way. The physics *depends* on that frame, so rotational invariance would even be wrong;
- you can canonicalize reliably;
- translation invariance (relative edge features) and permutation invariance are enough, which is the case for many mesh-based simulators.

**EGNN** (Satorras, Hoogeboom and Welling, 2021) is a simple E(n)-equivariant graph network: messages are computed only from invariant quantities (node features and squared distances), and positions are updated along relative position vectors, so rotations, reflections and translations of the input carry through exactly. The code below checks it numerically.

## Code: checking equivariance numerically

Whatever model you build, test the symmetry claim: transform the input, run the model, and compare with the transformed output.

```python
import torch

torch.manual_seed(0)

class EGNNLayer(torch.nn.Module):
    """E(n)-equivariant message passing (Satorras et al., 2021): messages see only invariant
    quantities (features and squared distances); positions are updated along relative vectors."""
    def __init__(self, dim):
        super().__init__()
        self.phi_e = torch.nn.Sequential(torch.nn.Linear(2 * dim + 1, dim), torch.nn.SiLU(), torch.nn.Linear(dim, dim))
        self.phi_x = torch.nn.Sequential(torch.nn.Linear(dim, dim), torch.nn.SiLU(), torch.nn.Linear(dim, 1))
        self.phi_h = torch.nn.Sequential(torch.nn.Linear(2 * dim, dim), torch.nn.SiLU(), torch.nn.Linear(dim, dim))

    def forward(self, h, x, edges):
        src, dst = edges
        rel = x[dst] - x[src]                                    # relative position: rotates with the input
        m = self.phi_e(torch.cat([h[dst], h[src], (rel**2).sum(-1, keepdim=True)], -1))   # invariant message
        x_new = x + torch.zeros_like(x).index_add_(0, dst, rel * self.phi_x(m)) / 10.0    # equivariant update
        h_new = h + self.phi_h(torch.cat([h, torch.zeros_like(h).index_add_(0, dst, m)], -1))  # invariant update
        return h_new, x_new

n, dim = 20, 16
h, x = torch.randn(n, dim), torch.randn(n, 3)
edges = torch.randint(0, n, (2, 80))
Q, _ = torch.linalg.qr(torch.randn(3, 3))                       # random rotation (or reflection)
t = torch.randn(3)                                               # random translation

layer = EGNNLayer(dim)
h1, x1 = layer(h, x, edges)
h2, x2 = layer(h, x @ Q.T + t, edges)                            # transform the input
invariant = torch.allclose(h1, h2, atol=1e-5)
equivariant = torch.allclose(x1 @ Q.T + t, x2, atol=1e-5)
print("features invariant:   ", invariant)
print("positions equivariant:", equivariant)

mlp = torch.nn.Sequential(torch.nn.Linear(3, 32), torch.nn.SiLU(), torch.nn.Linear(32, 3))
print("plain MLP on coordinates equivariant:", torch.allclose(mlp(x) @ Q.T + t, mlp(x @ Q.T + t), atol=1e-5))
```

Output:

```
features invariant:    True
positions equivariant: True
plain MLP on coordinates equivariant: False
```

The same test (transform, predict, compare) works for any model and any claimed symmetry, including permutation: shuffle the node order and check that per-node outputs shuffle identically.

## Canonical orientation or not

**Canonicalization** means putting every shape into a standard pose before the model sees it, so the model never has to deal with arbitrary orientations.

**When it is the right call:**

- **Engineering data usually has a natural frame.** Cars are in the vehicle frame with the wind along x; brackets are in the assembly frame with known load directions; gravity points down. Use that frame. Here the orientation *is* part of the physics, and removing it would remove information.
- **Inputs from different sources** (CAD exports in different coordinate systems, scanned parts) should be brought into the same frame, with the same units and origin conventions.

**Pitfalls of automatic canonicalization** (for example aligning principal axes from PCA of the points):

- **Sign ambiguity:** each principal axis can point either way; sign conventions (for example by the third moment) are needed.
- **Order ambiguity and instability:** when two principal moments are nearly equal (near-symmetric shapes), a tiny change in the shape can swap the axes, so similar designs end up in very different poses. That makes predictions jump discontinuously.
- **It must be applied consistently** at training and inference, and the outputs (vectors, tensors) transformed back to the original frame.

**A practical rule:** if a meaningful physical frame exists, use it, and give the model the physically relevant directions (inflow, gravity, load) as inputs. If inputs genuinely arrive in arbitrary orientations and there is no physical frame, use equivariant models or carefully disambiguated canonicalization, and test on rotated copies of test shapes.

## In an interview

**Thirty-second answer to "What are invariance and equivariance, and which symmetries matter?"** "Invariant means the output does not change when the input is transformed; equivariant means it transforms the same way, like a displacement field rotating with the part. Translation, rotation, reflection, node permutation and similarity matter in engineering, but only if the whole problem transforms together, including boundary conditions, inflow and gravity. Rotating a car under a fixed wind is a new yawed case, not a symmetry. I build in permutation and translation invariance almost always, use the natural engineering frame for orientation, and use exactly equivariant layers only when there is no preferred frame."

**Likely follow-ups**

- *When is augmentation by rotation valid?* Only if inputs, outputs (vectors rotated, tensors transformed) and all boundary conditions and external fields rotate together.
- *When do you need an equivariant GNN?* Frame-free problems with vector or tensor outputs and limited data, such as particles, molecules or parts in arbitrary orientations.
- *How do you check a symmetry?* Transform the input, run the model and compare with the transformed output, as in the code above.

## References

- [E(n) Equivariant Graph Neural Networks (Satorras, Hoogeboom & Welling, 2021)](https://arxiv.org/abs/2102.09844)
- [Learning Mesh-Based Simulation with Graph Networks (Pfaff et al.): relative edge features](https://arxiv.org/abs/2010.03409)
- [PointNet (Qi et al.): permutation invariance](https://arxiv.org/abs/1612.00593)
- [Relational inductive biases, deep learning, and graph networks (Battaglia et al.)](https://arxiv.org/abs/1806.01261)
- [Physics Informed Machine Learning playlist (Steve Brunton): symmetries and invariances](https://www.youtube.com/playlist?list=PLMrJAkhIeNNQ0BaKuBKY43k4xMo6NSbBa)
