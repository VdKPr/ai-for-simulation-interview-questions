# Geometry Representations

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** In most engineering surrogates the geometry *is* the main input, and its representation decides what the model can generalize to. **Design parameters** are compact and work well with Gaussian processes, but lock you into one parametric family. **Surface meshes and point clouds** handle free-form shapes and suit GNNs, point networks and transformers. **Voxel grids and signed distance fields** suit convolutional and Fourier models; the **signed distance function (SDF)**, the distance to the nearest surface with a sign for inside or outside, is especially popular because it is smooth, defined everywhere in the volume, and its value is literally the wall distance that boundary layers depend on. **Neural fields** represent a shape as a network you can query anywhere. **CAD (B-rep)** is what customers actually own but is the hardest to learn from directly. A model must also be **robust to remeshing**: the same shape meshed differently should give the same prediction.

**On this page**

- [Ways to represent 3D geometry](#ways-to-represent-3d-geometry)
- [Signed distance functions](#signed-distance-functions)
- [CAD models and CAD datasets](#cad-models-and-cad-datasets)
- [Neural fields](#neural-fields)
- [Why remeshing can change a prediction](#why-remeshing-can-change-a-prediction)
- [Worked example: surface pressure on a car from an STL](#worked-example-surface-pressure-on-a-car-from-an-stl)
- [Code: a signed distance field and a remeshing test](#code-a-signed-distance-field-and-a-remeshing-test)
- [In an interview](#in-an-interview)

## Ways to represent 3D geometry

| Representation | What it is | Strengths | Weaknesses | Typical models |
| --- | --- | --- | --- | --- |
| **Design parameters** | A vector of dimensions or morph parameters (length, angle, fillet radius) | Tiny; easy experiment design; ideal for GPs and small data | Only one parametric family; no transfer to new families; needs the parametric CAD | GP, MLP, POD + regression |
| **Surface mesh** (STL, OBJ, PLY) | Triangles or quads with connectivity | Exact surface; local structure; what CAD exports | Element size and quality vary wildly; often not watertight; connectivity varies between designs | Mesh GNNs, mesh transformers |
| **Volume mesh** (the simulation mesh) | Cells with the solution at nodes or cells | Where the solution lives; captures the boundary layer | 10⁶-10⁸ cells; only exists after meshing, so not available at inference unless you mesh | GNNs, point transformers |
| **Point cloud** | Points sampled on the surface (and normals), or in the volume | Simple; any shape; easy to resample and augment | No connectivity; results depend on sampling density | PointNet/++, DGCNN, transformers |
| **Voxels / regular grid** | Occupancy or values on a 3D grid | Works with CNNs, U-Nets and FNO | Memory grows as n³; staircase walls; thin parts vanish at low resolution | 3D U-Net, FNO |
| **Signed distance field** | Distance to the nearest surface, negative inside, on a grid or at query points | Smooth; defined everywhere; encodes wall distance | Needs a watertight surface for the sign; extra computation | Input to FNO, GINO, DoMINO |
| **Neural field** | A network f(x) (or f(x, z) with a shape code z) giving SDF or occupancy | Continuous, compact, arbitrary resolution | Must be fitted or encoded; fitting can be slow | DeepSDF, occupancy networks, SIREN |
| **CAD / B-rep** | Faces (NURBS surfaces), edges and vertices, plus topology | Exact; editable; carries design intent | Complex, heterogeneous data; few learning tools | B-rep graph networks; CAD-sequence models (DeepCAD) |
| **Multi-view images** | Renderings from several directions | Reuse 2D CNNs | Loses hidden surfaces and exact geometry | Quick classifiers |

**How to choose.**

1. **What is available at inference?** Customers usually have CAD or STL, not a simulation mesh. A model that needs the simulation mesh as input forces meshing at inference, which can take as long as the solve. AB-UPT, for example, was designed so that it can predict from a CAD geometry alone, without the CFD mesh.
2. **Is the design space parametric?** If every design comes from one parametric CAD model, parameters with a GP or MLP often beat any geometric deep-learning model on small data. See [Classical surrogates](classical-surrogates.md).
3. **Which outputs are needed?** Scalars (drag, peak temperature), surface fields (pressure, wall shear), or volume fields (wake velocity). Surface outputs pair naturally with surface inputs; volume outputs need a volume representation or a field decoder.
4. **How much data, and how different are the test geometries?** Free-form representations are needed to transfer across design families, but they need more data.

## Signed distance functions

**Definition.** For a closed surface ∂Ω, the signed distance at point x is

```math
\phi(\mathbf{x}) = \pm \min_{\mathbf{y} \in \partial\Omega} \lVert \mathbf{x} - \mathbf{y} \rVert
```

with the sign negative inside the body and positive outside (some codes use the opposite convention). The surface is the zero level set, φ = 0.

**Properties.**

- Away from the surface, the SDF satisfies the **Eikonal equation** |∇φ| = 1, and ∇φ is the outward normal of the nearest surface point.
- It is continuous everywhere and smooth almost everywhere (kinks occur on the medial axis, where two surface points are equally near).
- Its value tells every volume point **how far it is from the wall**, and its gradient says **in which direction**.

**Why so many surrogates take it as input.**

1. **It turns a surface into a volume field** on a regular grid, so convolutional and Fourier models (U-Net, FNO, GINO's latent grid) can consume arbitrary shapes.
2. **Wall distance is physically meaningful.** Boundary layers, wall functions and turbulence models are written in terms of distance to the wall (y+ is a scaled wall distance). An SDF hands the network that coordinate directly.
3. **It distinguishes inside from outside**, so the model knows where the flow is.
4. **It is smooth**, unlike a binary occupancy mask with staircase edges.

GINO (Li et al., NeurIPS 2023), for example, takes the SDF of the shape on a latent grid as input, and DoMINO uses SDF-type features from the local geometry of each query point.

**Computing it from an STL.** Unsigned distance: nearest-triangle queries with a bounding-volume hierarchy (libraries: `trimesh`, Open3D, `pysdf`, `igl`). Sign: a winding-number or ray-parity test, which needs a **watertight** mesh; holes and flipped normals give wrong signs, so check and repair first. The [code below](#code-a-signed-distance-field-and-a-remeshing-test) does the 2D version for an airfoil.

## CAD models and CAD datasets

**What a CAD model is.** Parametric CAD stores a **feature history** (sketch, extrude, revolve, fillet, with dimensions and constraints) and its result as a **B-rep** (boundary representation): faces carrying exact surfaces (planes, cylinders, NURBS), edges carrying exact curves, and a topology graph saying which faces meet along which edges. Exchange formats: STEP and IGES (neutral), Parasolid and ACIS (kernels), native formats per CAD system. An STL is a triangulation of the B-rep, with all of that structure thrown away.

**Learning directly from CAD.**

- **B-rep graph networks** treat faces as nodes and shared edges as edges, with sampled UV grids on each face as features (UV-Net and BRepNet are examples).
- **CAD-sequence models** generate or reconstruct the feature history itself (DeepCAD; Text2CAD for text-to-CAD). See [Text-to-CAD](text-to-cad.md).
- **Most physics surrogates still convert CAD to a surface mesh or point cloud** before learning, because the solver data lives on meshes.

**Datasets.**

| Dataset | Content | Notes |
| --- | --- | --- |
| ABC (Koch et al., CVPR 2019) | One million CAD models, each a collection of explicitly parametrized curves and surfaces | Ground truth for normals, patch segmentation and feature detection; can be sampled at any resolution |
| Fusion 360 Gallery (Willis et al., SIGGRAPH 2021) | 8,625 human design sequences in a simple sketch-and-extrude language, plus the Fusion 360 Gym environment | Programmatic CAD reconstruction |
| DeepCAD (Wu et al., ICCV 2021) | 178,238 models with their CAD construction sequences | Generative models of CAD sequences |
| ShapeNet, ModelNet | Meshes of everyday objects (cars, chairs, planes) | Not engineering CAD, but used for car shapes in early surrogate papers |
| Engineering simulation datasets | DrivAerNet++, DrivAerML, SimJEB and others ship geometry with simulation results | See [Public simulation datasets](public-simulation-datasets.md) |

## Neural fields

A **neural field** (implicit neural representation) is a network that maps a coordinate to a value: f_θ(x) → SDF, occupancy or a physical field. The shape is the network's zero level set, at any resolution, with memory independent of resolution.

- **DeepSDF (Park et al., CVPR 2019)** learns one decoder for a whole class of shapes: f_θ(x, z) with a latent code z per shape. A new shape is encoded by optimizing its z to fit observed points (auto-decoding). Interpolating between codes interpolates shapes, and the model completes shapes from partial or noisy data while being an order of magnitude smaller than earlier representations.
- **Occupancy Networks (Mescheder et al., CVPR 2019)** represent the surface as the decision boundary of a classifier that predicts inside or outside.
- **SIREN (Sitzmann et al., NeurIPS 2020)** uses sine activations with a principled initialization so that the network represents fine detail **and its derivatives**. It can fit an SDF by solving the Eikonal equation and can solve Poisson, Helmholtz and wave problems, which connects neural fields to [PINNs](pinns.md) and [spectral bias](spectral-bias-and-fourier-features.md).

**In surrogates,** neural fields appear in two places. Geometry: a shape latent code (from a DeepSDF-style encoder) can serve as the input of a field predictor. Output: **neural-field decoders** query the predicted solution at any point, which is how UPT, AB-UPT and GINO produce outputs at arbitrary resolution. See [Physics transformer architectures](physics-transformer-architectures.md).

## Why remeshing can change a prediction

The same shape can be meshed in many ways: coarser or finer, uniform or graded, triangles or quads, a different mesher. A good surrogate's prediction should depend on the **shape**, not on the **mesh**. Common reasons it does not:

1. **Node counts and sums.** A model that sums or averages over nodes (global pooling, a GIN-style sum) weights finely meshed regions more heavily. The [demo below](#code-a-signed-distance-field-and-a-remeshing-test) shows a node average swinging from 0 to −0.41 for the same field when the mesh is graded; an area-weighted average stays at 0.
2. **Hop-based receptive fields.** A GNN with L layers sees L hops, which covers a smaller physical region on a finer mesh. See [Why mesh refinement makes it worse](multiscale-gnns.md#why-mesh-refinement-makes-it-worse).
3. **Edge features that encode mesh size.** Unnormalized edge lengths or k-NN graphs with a fixed k change meaning when density changes.
4. **Training on one mesher's output.** If every training sample came from the same meshing settings, the network can learn mesher artefacts; a different element-size distribution is then out of distribution.
5. **Sampling noise.** Point-based models see a random subset; different subsets give slightly different answers.

**How to test for it.**

- Re-mesh or resample held-out test shapes at several resolutions and with a different mesher, and report the spread of predictions alongside the error. Treat a spread larger than the target accuracy as a bug.
- Check integrated quantities (drag, lift, total heat flux) as well as point errors.
- Compare predictions on matched points by interpolating between the meshes.

**How to make models robust.**

- Use **area- or volume-weighted** pooling and losses; physical integrals, not node counts.
- Use **radius graphs in physical units** rather than fixed-k graphs, or include local mesh size as a feature.
- **Train with resampling and remeshing augmentation:** random point subsets, several meshing settings.
- Prefer **discretization-invariant** architectures: neural operators and attention-based models with neural-field decoders. See [Discretization invariance](neural-operators.md#discretization-invariance).

## Worked example: surface pressure on a car from an STL

**Task.** A customer uploads an STL of a new car design; return the surface pressure (and wall shear) field and the drag coefficient in seconds.

1. **Clarify the scope.** Which flow conditions (one speed and yaw, or a range)? Which accuracy is needed for drag (counts: 0.001 in C_d)? Which design families will be tested? Is the training data RANS or scale-resolving (DrivAerML uses hybrid RANS-LES; DrivAerNet++ uses RANS)?
2. **Preprocess the STL.** Check and fix: watertightness, duplicate vertices, flipped normals, units (mm vs m), coordinate frame (x from front to back, z up), and position relative to the ground plane. Wheels and underbody details matter for drag; check that they are present. See [Mesh data processing](mesh-data-processing.md).
3. **Represent the geometry.** Area-weighted point samples with normals (for example 100,000-500,000 surface points), optionally SDF values on a coarse volume grid for global context, and the global conditions (speed, yaw) as extra inputs.
4. **Choose a model.**
   - Strong baselines: RegDGCNN or PointNet++ on points; MeshGraphNets on the surface mesh.
   - Current state of the art on DrivAerML-type data: Transolver, AB-UPT, DoMINO and GeoTransolver, which handle millions of points and predict surface and volume fields. See [Physics transformer architectures](physics-transformer-architectures.md).
5. **Normalize targets.** Predict the pressure coefficient C_p = (p − p∞) / (½ρU²) rather than raw pressure, so designs at different speeds are comparable. See [Nondimensionalization](nondimensionalization-and-normalization.md).
6. **Train.** Relative L2 loss on C_p (and wall shear); random point subsets per step; augment only with transformations that keep the physics the same (a mirror image is valid only if the flow is symmetric, which a nonzero yaw angle breaks). Split train/validation/test by **design**, not by point. See [Splitting simulation data](data-splits-and-leakage.md#splitting-simulation-data).
7. **Compute forces from the prediction.** Integrate pressure and shear over the surface: F = ∮ (−(p − p∞) n + τ_w) dA, then C_d = 2 F·e_x / (ρ U² A_ref). Use the exact areas and normals of the original STL, not the sample counts. See [Drag force from pressure and wall shear](mesh-data-processing.md#drag-force-from-pressure-and-wall-shear).
8. **Evaluate like an engineer.** Field relative L2 on surface pressure; drag and lift error in counts; R² of predicted vs true drag across test designs; ranking of design variants; robustness to remeshing; and the speed-up including data generation. See [Evaluation metrics](evaluation-metrics.md).
9. **Deploy with guards.** An out-of-distribution score for new geometries, uncertainty, and a fallback to CFD. See [Detecting an out-of-distribution geometry](out-of-distribution-and-generalization.md#detecting-an-out-of-distribution-geometry).

The full system-design version of this question is [Design: car aerodynamics surrogate](design-car-aerodynamics-surrogate.md).

## Code: a signed distance field and a remeshing test

Two demonstrations. First, the signed distance field of a NACA 0012 airfoil on a grid, with checks of its known properties. Second, the same smooth surface field on three meshes of the same circle, averaged per node and per unit length.

```python
import numpy as np
from matplotlib.path import Path

def naca4(t=0.12, n=200):
    """Closed outline of a symmetric NACA 00xx airfoil with chord 1 (upper surface, then lower)."""
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    return np.concatenate([np.stack([x[::-1], yt[::-1]], 1), np.stack([x[1:], -yt[1:]], 1)])

def signed_distance(points, polygon):
    """Signed distance from points (M, 2) to a closed polygon: negative inside, positive outside."""
    a, b = polygon, np.roll(polygon, -1, axis=0)
    ab = b - a
    t = np.clip(((points[:, None, :] - a[None]) * ab).sum(-1) / (ab * ab).sum(-1), 0, 1)
    nearest = a[None] + t[..., None] * ab[None]                       # nearest point on every segment
    dist = np.linalg.norm(points[:, None, :] - nearest, axis=-1).min(1)
    inside = Path(polygon).contains_points(points)                   # sign from a point-in-polygon test
    return np.where(inside, -dist, dist)

foil = naca4()
xs, ys = np.meshgrid(np.linspace(-0.5, 1.5, 201), np.linspace(-0.5, 0.5, 101))
sdf = signed_distance(np.stack([xs.ravel(), ys.ravel()], 1), foil).reshape(ys.shape)
print(f"SDF grid {sdf.shape}: min {sdf.min():.4f} (inside), max {sdf.max():.4f}")
print(f"SDF at (0.3, 0), inside where the airfoil is thickest: {signed_distance(np.array([[0.3, 0.0]]), foil)[0]:.4f}")
print(f"SDF at (0.5, 0.3), outside: {signed_distance(np.array([[0.5, 0.3]]), foil)[0]:.4f}")
gy, gx = np.gradient(sdf, ys[1, 0] - ys[0, 0], xs[0, 1] - xs[0, 0])
print(f"mean |grad SDF| away from the surface (Eikonal: 1): {np.hypot(gx, gy)[np.abs(sdf) > 0.05].mean():.4f}")

def circle_mesh(n, grade):
    """n points on a unit circle; grade > 0 clusters them near angle pi, as a graded mesh would."""
    s = np.linspace(0, 1, n, endpoint=False)
    return 2 * np.pi * (s + grade * np.sin(2 * np.pi * s) / (2 * np.pi))

pressure = np.cos                                                   # a smooth surface field; its true average is 0
for n, grade in [(40, 0.0), (400, 0.0), (400, 0.9)]:
    th = circle_mesh(n, grade)
    seg = np.diff(np.append(th, th[0] + 2 * np.pi))                 # arc length between neighbouring points
    length = 0.5 * (seg + np.roll(seg, 1))                          # arc length owned by each point
    node_avg = pressure(th).mean()                                  # what a model pooling over nodes sees
    weighted = (pressure(th) * length).sum() / length.sum()
    print(f"n={n:4d}, grading {grade:.1f}: node average {node_avg:+.4f}   length-weighted average {weighted:+.4f}")
```

Output:

```
SDF grid (101, 201): min -0.0600 (inside), max 0.7071
SDF at (0.3, 0), inside where the airfoil is thickest: -0.0600
SDF at (0.5, 0.3), outside: 0.2467
mean |grad SDF| away from the surface (Eikonal: 1): 0.9977
n=  40, grading 0.0: node average +0.0000   length-weighted average -0.0000
n= 400, grading 0.0: node average +0.0000   length-weighted average -0.0000
n= 400, grading 0.9: node average -0.4059   length-weighted average -0.0000
```

The SDF is −0.06 at the thickest point, exactly the airfoil's half-thickness there, and its gradient has unit length away from the surface, as the Eikonal equation requires. In the second test the shape and the field are identical in all three cases, yet the node average jumps to −0.41 on the graded mesh because most nodes sit where cos θ is negative. Weighting by the length (or area) each node represents removes the dependence on the mesh.

## In an interview

**Thirty-second answer to "What is a signed distance function, and why do so many surrogates take it as input?"** "It is the distance from a point to the nearest surface, negative inside the body and positive outside, so the surface is its zero level set and its gradient has unit length. Surrogates like it because it turns any shape into a smooth field on a grid that CNNs and FNOs can read, it tells the model inside from outside, and its value is the wall distance that boundary layers and turbulence models depend on. You compute it from a watertight STL with nearest-triangle queries and a winding-number sign test."

**Likely follow-ups**

- *Why can remeshing change a prediction, and how do you test it?* Node-count pooling, hop-based receptive fields and mesh-size-dependent edges; remesh test shapes at several resolutions and report the spread.
- *What is a neural field?* A network that maps coordinates to SDF, occupancy or a physical field, queried at any resolution; DeepSDF adds a latent code per shape.
- *Why not learn from CAD directly?* B-rep data is heterogeneous and the solver results live on meshes, so most pipelines tessellate; B-rep graph networks exist for tasks like feature recognition.

## References

- [DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation (Park et al.)](https://arxiv.org/abs/1901.05103)
- [Occupancy Networks (Mescheder et al.)](https://arxiv.org/abs/1812.03828)
- [Implicit Neural Representations with Periodic Activation Functions (SIREN; Sitzmann et al.)](https://arxiv.org/abs/2006.09661)
- [Geometry-Informed Neural Operator (GINO; Li et al.)](https://arxiv.org/abs/2309.00583)
- [DoMINO (Ranade et al.)](https://arxiv.org/abs/2501.13350) and [AB-UPT (Alkin et al.)](https://arxiv.org/abs/2502.09692)
- [ABC: A Big CAD Model Dataset (Koch et al.)](https://arxiv.org/abs/1812.06216), [Fusion 360 Gallery (Willis et al.)](https://arxiv.org/abs/2010.02392), [DeepCAD (Wu et al.)](https://arxiv.org/abs/2105.09492)
- [trimesh documentation (proximity and signed distance)](https://trimesh.org/)
