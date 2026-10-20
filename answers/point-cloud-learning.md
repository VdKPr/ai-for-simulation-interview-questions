# Point-Cloud Learning

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A point cloud is an **unordered set** of points (x, y, z, plus optional features such as normals), so a model must give the same answer whatever order the points are listed in. **PointNet** (Qi et al., CVPR 2017) achieves this by applying the **same MLP to every point** and combining the results with a **symmetric function, max pooling**, which does not care about order. Its weakness is that each point is processed alone, so local geometry is invisible until the final pooling. **PointNet++** applies PointNet recursively on nested neighbourhoods (farthest point sampling for centres, ball query for groups) to learn local features at growing scales, and adapts to uneven sampling density. **DGCNN** builds a k-nearest-neighbour graph at every layer, in feature space, and applies **EdgeConv** to edge features (xᵢ, xⱼ − xᵢ). **Point Transformer** uses self-attention inside local neighbourhoods with relative position encodings. In engineering surrogates, points sampled from an STL or CAD surface are a common geometry input: DrivAerNet's RegDGCNN, for example, predicts car drag from point clouds.

**On this page**

- [PointNet](#pointnet)
- [PointNet++](#pointnet-1)
- [DGCNN (EdgeConv) and Point Transformer](#dgcnn-edgeconv-and-point-transformer)
- [Sampling points from a surface mesh](#sampling-points-from-a-surface-mesh)
- [Batching geometries with different point counts](#batching-geometries-with-different-point-counts)
- [Code: farthest point sampling](#code-farthest-point-sampling)
- [Code: a minimal PointNet](#code-a-minimal-pointnet)
- [In an interview](#in-an-interview)

## PointNet

**Architecture.**

1. **Shared per-point MLP.** The same MLP h maps each point xᵢ (3 coordinates, or more with normals and other features) to a feature vector, for example 1,024-dimensional. "Shared" means one set of weights for all points: in code, a `Linear` layer applied to a tensor of shape (batch, N, features), or a 1 × 1 convolution.
2. **Symmetric aggregation.** Max pooling over the N points gives one global feature vector g = maxᵢ h(xᵢ), element-wise.
3. **Head.** An MLP on g predicts a shape-level output: a class, or a drag coefficient.
4. **Per-point outputs (segmentation, or a surface field such as pressure).** Concatenate each point's local feature with the global feature and apply another shared MLP, so each point's prediction depends on its own position and on the whole shape.
5. **T-Nets (optional).** Small networks predict a 3 × 3 alignment of the input and a feature-space transform, as a learned canonicalization. Many later models drop them.

**Why max pooling makes it permutation-invariant.** Any reordering of the points permutes the rows of the matrix of per-point features but does not change the maximum of each column. Sum and mean are also symmetric. The paper proves that a network of the form γ(maxᵢ h(xᵢ)) can approximate any continuous set function, and shows that max pooling works better in practice than the alternatives they tried.

**What max pooling learns.** Each of the global feature's channels is "activated" by a few points. The paper calls them **critical points**: a sparse set, often on the shape's outline, that alone determines the global feature. Points can be added or removed without changing the output as long as the critical points remain, which explains PointNet's robustness to missing or extra points.

**Limitation.** Points interact only through the single global max. There is no notion of a local neighbourhood, so fine local geometry (a fillet radius, a spoiler lip) is hard to capture, and features do not transfer well between shapes at different scales.

## PointNet++

PointNet++ (Qi et al., NeurIPS 2017) adds the locality that PointNet lacks, in the way a CNN builds features over growing receptive fields.

**Set abstraction level** (repeated several times):

1. **Sampling:** choose M centroids with farthest point sampling, which covers the shape evenly. See the [code below](#code-farthest-point-sampling).
2. **Grouping:** for each centroid, collect the points within radius r (a **ball query**, capped at K points) or its K nearest neighbours.
3. **Local PointNet:** apply a shared MLP to each point's coordinates **relative to its centroid**, then max-pool within the group. Each centroid now carries a feature describing its neighbourhood.
4. **Repeat** with fewer centroids and larger radii, so features describe larger regions.

**Non-uniform density.** Real scans and surface samplings are denser in some places (small features, curved regions) than others, and a network trained on uniform densities degrades badly. PointNet++ proposed **multi-scale grouping** (several radii per centroid, features concatenated) and **multi-resolution grouping** (combine features from the current and previous level) to adapt.

**Feature propagation** (for per-point outputs): interpolate features from the coarse centroids back to the original points by inverse-distance weighting of the k nearest centroids, concatenate skip features from the matching encoder level, and apply a shared MLP. This is the point-cloud analogue of a U-Net decoder.

## DGCNN (EdgeConv) and Point Transformer

**DGCNN** (Wang et al., ACM TOG 2019) introduced **EdgeConv**.

1. Build a k-nearest-neighbour graph over the current features.
2. For each edge (i, j), compute an edge feature h_Θ(xᵢ, xⱼ − xᵢ): the point itself (global position) and the offset to its neighbour (local shape).
3. Aggregate over neighbours with max.

The key word is **dynamic**: the k-NN graph is recomputed at **every layer in feature space**, not only from 3D coordinates. Deeper layers can therefore connect points that are far apart in space but similar in feature space (two wingtips, the four wheels), capturing semantic structure over long distances. The cost is a k-NN search per layer, O(N²) naively, which limits the point count. **RegDGCNN**, the regression variant used in DrivAerNet, predicts drag coefficients directly from car point clouds.

**Point Transformer** (Zhao et al., ICCV 2021) applies self-attention **within local neighbourhoods** (k nearest neighbours). It uses vector attention (a separate weight for each feature channel) and a learned encoding of the **relative position** xᵢ − xⱼ added to both the attention and the values. It reached 70.4% mIoU on the S3DIS scene-segmentation benchmark (Area 5), the first model above 70%. Later versions (Point Transformer V2, V3) made it faster and larger.

**Summary.**

| Model | How points interact | Locality | Typical use in simulation |
| --- | --- | --- | --- |
| PointNet | Only through a global max | None | Fast baseline; global scalars |
| PointNet++ | Local PointNets on nested balls | Hierarchical, metric space | Surface fields, scalars |
| DGCNN | EdgeConv on a k-NN graph rebuilt in feature space | Local plus feature-space | Drag regression (RegDGCNN) |
| Point Transformer | Local vector self-attention with relative positions | Local neighbourhoods | Fields on large point sets |

For full-car fields with millions of points, physics-specific transformers (Transolver, AB-UPT) and point-based neural operators (DoMINO) now dominate. See [Physics transformer architectures](physics-transformer-architectures.md).

## Sampling points from a surface mesh

STL and other surface meshes have very uneven triangle sizes: tiny triangles around a fillet, huge ones on a flat door. Taking the mesh vertices as the point cloud over-represents finely meshed regions. Sample by area instead.

1. **Choose triangles in proportion to area.** Compute each triangle's area, normalize to a probability, and draw triangle indices.
2. **Sample uniformly inside each chosen triangle** with barycentric coordinates: draw u, v ~ U(0, 1); if u + v > 1, replace them by (1 − u, 1 − v); the point is a + u(b − a) + v(c − a).
3. **Attach features:** the face normal (consistently oriented outward), and, if needed, curvature, part labels or the signed distance of nearby volume points.
4. **Optionally thin the cloud** with farthest point sampling for even coverage, or with a voxel grid (keep one point per cell) for speed.
5. **Interpolate fields** (pressure, wall shear) from the mesh to the samples with the same barycentric weights.

Area-weighted sampling is unbiased for surface integrals: the mean of a quantity over the samples, times the total area, estimates its integral. That matters if forces are computed from predicted point values.

## Batching geometries with different point counts

Different designs have different numbers of points, but GPUs want rectangular tensors. Options:

1. **Resample to a fixed count** (for example 10,000 or 100,000 points per shape) each epoch. Simple, and the random resampling is a useful augmentation. At inference, predict in chunks or average over several samplings.
2. **Pad and mask.** Pad every shape to the largest point count, keep a boolean mask, and exclude padded points from pooling (set them to −∞ before a max, or weight them by zero in a mean) and from the loss. The code below does this.
3. **Concatenate with a batch index** (the PyTorch Geometric convention): stack all points of all shapes into one long tensor plus a vector saying which shape each point belongs to, and pool with scatter operations. No padding waste.
4. **Bucket by size** to reduce padding, or use one shape per GPU for very large shapes.

**Common bug:** forgetting the mask in max pooling. Padding with zeros and taking the max lets the zeros win whenever all real features are negative, so the output depends on how much padding was added.

## Code: farthest point sampling

Greedy farthest point sampling: start from one point, then repeatedly add the point farthest from everything chosen so far. It costs O(N · m) for m samples, so for millions of points use a GPU implementation or voxel down-sampling first. The demo samples a surface mesh by area first, then compares how evenly random and FPS subsets cover it.

```python
import numpy as np
from scipy.spatial import cKDTree

def sample_surface(V, F, n, rng):
    """Area-weighted uniform sampling of a triangle mesh; returns points and their face normals."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cross = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    tri = rng.choice(len(F), size=n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    pts = a[tri] + u[:, None] * (b - a)[tri] + v[:, None] * (c - a)[tri]
    return pts, cross[tri] / np.linalg.norm(cross[tri], axis=1, keepdims=True)

def farthest_point_sampling(points, m, start=0):
    """points: (N, d) array. Returns the indices of m points that cover the cloud evenly."""
    chosen = np.empty(m, dtype=int)
    chosen[0] = start
    dist = np.full(len(points), np.inf)
    for i in range(1, m):
        dist = np.minimum(dist, np.linalg.norm(points - points[chosen[i - 1]], axis=1))
        chosen[i] = int(np.argmax(dist))
    return chosen

# A unit cube made of 12 triangles: every face has area 1, so 1/6 of the samples should land on each face.
V = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], float)
F = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
rng = np.random.default_rng(0)
pts, normals = sample_surface(V, F, 20000, rng)
on_top = np.isclose(pts[:, 2], 1).mean()
print(f"fraction of samples on the top face: {on_top:.3f} (exact 1/6 = {1/6:.3f})")

m = 256
idx_fps = farthest_point_sampling(pts, m)
idx_rand = rng.choice(len(pts), m, replace=False)
for name, idx in [("random", idx_rand), ("FPS", idx_fps)]:
    gap, _ = cKDTree(pts[idx]).query(pts)          # distance from every surface sample to the nearest chosen point
    print(f"{name:>6}: worst coverage gap {gap.max():.3f}, mean gap {gap.mean():.3f}")
```

Output:

```
fraction of samples on the top face: 0.160 (exact 1/6 = 0.167)
random: worst coverage gap 0.223, mean gap 0.075
   FPS: worst coverage gap 0.118, mean gap 0.059
```

Area-weighted sampling puts the expected share of points on each face (0.160 against 1/6 ≈ 0.167). With the same budget of 256 points, FPS roughly halves the worst coverage gap compared with random sampling: no region of the surface is left more than 0.12 from a chosen point.

## Code: a minimal PointNet

A shared per-point MLP, masked max pooling for padded batches, and two heads: one global value per shape (for example a drag coefficient) and one value per point (for example surface pressure). The checks confirm permutation invariance of the global output, permutation equivariance of the per-point output, and that padding does not change anything.

```python
import torch, torch.nn as nn

class PointNet(nn.Module):
    def __init__(self, in_dim=3, hidden=64, out_dim=1, point_out=1):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
        self.point_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, point_out))

    def encode(self, pts, mask=None):
        feats = self.shared(pts)                                   # the same MLP applied to every point
        if mask is not None:                                       # padded points must never win the max
            feats = feats.masked_fill(~mask[..., None], float("-inf"))
        return feats, feats.max(dim=1).values                      # symmetric function -> order does not matter

    def forward(self, pts, mask=None):                             # pts: (batch, N, in_dim)
        return self.head(self.encode(pts, mask)[1])               # one value per shape

    def per_point(self, pts, mask=None):
        feats, g = self.encode(pts, mask)
        g = g[:, None, :].expand(-1, pts.shape[1], -1)
        return self.point_head(torch.cat([feats.clamp(min=0), g], dim=-1))   # local feature + global context

torch.manual_seed(0)
net = PointNet()
P = torch.randn(2, 100, 3)
perm = torch.randperm(100)
print("global output permutation-invariant:", torch.allclose(net(P), net(P[:, perm]), atol=1e-6))
print("per-point output permutation-equivariant:", torch.allclose(net.per_point(P)[:, perm], net.per_point(P[:, perm]), atol=1e-6))

# Batch two shapes with 100 and 60 points by padding to 100 and masking.
shape_b = torch.randn(60, 3) - 3.0                                 # all coordinates negative
padded = torch.zeros(2, 100, 3); padded[0] = P[0]; padded[1, :60] = shape_b
mask = torch.zeros(2, 100, dtype=torch.bool); mask[0] = True; mask[1, :60] = True
alone = net(shape_b[None])
print("masked padding matches unpadded result:", torch.allclose(net(padded, mask)[1], alone[0], atol=1e-6))
print("unmasked padding changes the result:   ", not torch.allclose(net(padded)[1], alone[0], atol=1e-6))
```

Output:

```
global output permutation-invariant: True
per-point output permutation-equivariant: True
masked padding matches unpadded result: True
unmasked padding changes the result:    True
```

## In an interview

**Thirty-second answer to "Explain PointNet. Why does max pooling make it permutation-invariant?"** "A point cloud is a set, so the output must not depend on point order. PointNet applies one shared MLP to each point independently, then takes the element-wise max over points to get a global feature, then an MLP head. Reordering the points only reorders the rows being maxed over, so the max is unchanged. For per-point outputs you concatenate each point's feature with the global feature. The weakness is no local structure, which PointNet++ fixes with farthest point sampling, ball-query grouping and local PointNets in a hierarchy."

**Likely follow-ups**

- *Why not sum or mean?* Both are symmetric too; max was empirically best in the paper and makes the model robust to missing points, but sum keeps counting information (see [GIN](gnn-layer-types.md#gin-and-expressive-power)).
- *How do you get points from an STL?* Sample triangles by area, then uniformly inside each with barycentric coordinates, and keep normals.
- *What is dynamic about DGCNN?* The k-NN graph is rebuilt in feature space at each layer.

## References

- [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation (Qi et al., CVPR 2017)](https://arxiv.org/abs/1612.00593)
- [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space (Qi et al., NeurIPS 2017)](https://arxiv.org/abs/1706.02413)
- [Dynamic Graph CNN for Learning on Point Clouds (Wang et al.)](https://arxiv.org/abs/1801.07829)
- [Point Transformer (Zhao et al.)](https://arxiv.org/abs/2012.09164)
- [DrivAerNet: a parametric car dataset with RegDGCNN baselines (Elrefaie, Dai & Ahmed)](https://arxiv.org/abs/2403.08055)
- [Open3D](https://github.com/isl-org/Open3D) and [PyTorch3D](https://github.com/facebookresearch/pytorch3d)
