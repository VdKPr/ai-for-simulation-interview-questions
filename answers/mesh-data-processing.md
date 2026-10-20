# Mesh Data Processing

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Expect a coding round on geometry data. Candidates report PhysicsX rounds that included converting flat vertex data into a mesh format and live coding on a mesh dataset: identifying the element types and computing a bounding box. The core skills are the **data structures** (a vertex array, connectivity arrays per element type, point data vs cell data), turning **triangle soup into an indexed mesh** by welding duplicate vertices, and writing **vectorized NumPy** for the standard operations: face areas and normals, unique and boundary edges, vertex adjacency in sparse (CSR) form, the volume of a closed mesh, the force on a body from surface pressure and wall shear, interpolation of fields between meshes, and k-nearest-neighbour and radius graphs for GNNs. On millions of points, the rules are: no Python loops over elements, scatter operations instead of loops, spatial trees instead of distance matrices, and chunking to bound memory. Every function below runs on the small test meshes defined at the top, and its output is shown.

**On this page**

- [Data structures](#data-structures)
- [Flat arrays to a mesh](#flat-arrays-to-a-mesh)
- [Element types and the bounding box](#element-types-and-the-bounding-box)
- [Areas, face normals and vertex normals](#areas-face-normals-and-vertex-normals)
- [Unique edges and boundary edges](#unique-edges-and-boundary-edges)
- [Vertex adjacency in CSR form](#vertex-adjacency-in-csr-form)
- [Volume of a closed mesh](#volume-of-a-closed-mesh)
- [Drag force from pressure and wall shear](#drag-force-from-pressure-and-wall-shear)
- [Interpolating fields between meshes](#interpolating-fields-between-meshes)
- [k-nearest-neighbour and radius graphs](#k-nearest-neighbour-and-radius-graphs)
- [Making Python fast on millions of points](#making-python-fast-on-millions-of-points)
- [In an interview](#in-an-interview)

## Data structures

- **Vertices (points)** `V`: an (n, 3) float array of coordinates.
- **Connectivity** per element type: (m, 3) integer indices for triangles, (m, 4) for quads or tetrahedra, (m, 8) for hexahedra. Real meshes mix types, so libraries store a list of blocks, one per type (meshio's `mesh.cells`), or one flat VTK-style array.
- **Point data vs cell data:** values at vertices (interpolated within elements) or one value per element (finite-volume results). Never mix them up; converting between them is averaging or interpolation.
- **VTK cell-type IDs** you will see: vertex 1, line 3, triangle 5, quad 9, tetra 10, hexahedron 12, wedge 13, pyramid 14, quadratic triangle 22, quadratic tetra 24.
- **Orientation:** triangle vertex order defines the normal direction (right-hand rule); a consistently oriented closed surface has all normals pointing outward.

File formats and libraries are covered in [Simulation file formats](simulation-file-formats.md). Useful libraries: meshio (read and write), PyVista (VTK in Python), trimesh (surface meshes), Open3D (point clouds), SciPy (sparse matrices and k-d trees).

The test data used throughout: a unit cube as 8 vertices and 12 outward-oriented triangles, the same cube as a triangle soup, a VTK-style mixed cell array, and random points for interpolation and graphs.

```python
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

cube_V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
cube_F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                   [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])     # outward normals
flat_xyz = cube_V.reshape(-1)                      # [x0, y0, z0, x1, y1, z1, ...] as a file might store it
flat_faces = cube_F.reshape(-1)                    # [i0, j0, k0, i1, j1, k1, ...]
soup = cube_V[cube_F].reshape(-1)                  # STL-style triangle soup: 9 numbers per triangle
cells = [3, 0, 1, 2, 4, 0, 1, 2, 3]                # VTK-style: [count, ids..., count, ids...] (a triangle and a quad)
rng = np.random.default_rng(0)
src_pts = rng.random((2000, 3)); src_val = src_pts[:, 0]; dst_pts = rng.random((100, 3))
print("test data ready:", cube_V.shape, cube_F.shape, len(soup), "soup values")
```

Output:

```
test data ready: (8, 3) (12, 3) 108 soup values
```

## Flat arrays to a mesh

**Flat arrays.** Many files and APIs give coordinates as one flat list [x₀, y₀, z₀, x₁, …] and connectivity as [i₀, j₀, k₀, i₁, …]. Reshape them (`reshape(-1, 3)`), after checking that the length is divisible by 3 and that indices are within range and start at 0 (OBJ starts at 1).

**Mixed element types (VTK style).** A flat array [n, ids…, n, ids…] stores each cell's vertex count followed by its indices; parse it into per-cell index arrays (or group by count).

**Triangle soup to an indexed mesh ("welding").** STL stores every triangle with its own three vertices, so each shared vertex appears about six times and the mesh has no connectivity. Merge coincident vertices: round coordinates to a tolerance, find unique rows, and use the inverse mapping as the new connectivity. The tolerance is in model units: too coarse merges distinct vertices (small features collapse), too fine leaves cracks where vertices differ by round-off.

```python
V = flat_xyz.reshape(-1, 3)                        # flat [x0, y0, z0, x1, ...] -> (n, 3)
F = flat_faces.reshape(-1, 3)                      # if every face is a triangle
assert F.min() >= 0 and F.max() < len(V), "indices out of range (1-based OBJ indices?)"

def parse_vtk_cells(cells):
    """VTK-style flat cell array [n, i0, ..., n, j0, ...] -> list of index arrays (one per cell)."""
    out, i = [], 0
    while i < len(cells):
        n = cells[i]
        out.append(np.asarray(cells[i + 1:i + 1 + n]))
        i += n + 1
    return out

def weld(soup_xyz, decimals=9):
    """Triangle soup (e.g. STL: 9 floats per triangle) -> indexed mesh (V, F) with shared vertices."""
    P = soup_xyz.reshape(-1, 3)
    V, inv = np.unique(np.round(P, decimals), axis=0, return_inverse=True)
    return V, inv.reshape(-1, 3)

V2, F2 = weld(soup)
print(f"soup: {len(soup) // 3} vertex entries -> welded: {len(V2)} unique vertices, {len(F2)} triangles")
print("cells parsed from the VTK-style array:", [c.tolist() for c in parse_vtk_cells(cells)])
```

Output:

```
soup: 36 vertex entries -> welded: 8 unique vertices, 12 triangles
cells parsed from the VTK-style array: [[0, 1, 2], [0, 1, 2, 3]]
```

## Element types and the bounding box

**Element types.** Count cells per type (by vertex count in a flat array, or by block type in meshio); mixed meshes are normal (tetrahedra with prism layers near walls, pyramids between them). The axis-aligned **bounding box** is the per-axis minimum and maximum of the vertex coordinates; its extent gives the size and, compared with expectations, catches unit errors (a 4,500 "m" car is in millimetres) and misplaced parts. An oriented bounding box (from principal axes) is tighter but rarely needed.

```python
lo, hi = V.min(axis=0), V.max(axis=0)              # axis-aligned bounding box
extent = hi - lo
sizes, counts = np.unique([len(c) for c in parse_vtk_cells(cells)], return_counts=True)
names = {3: "triangle", 4: "quad (or tetra)", 8: "hexahedron"}
print("bounding box:", lo, hi, "extent", extent)
print("cell types:", {names[s]: int(c) for s, c in zip(sizes, counts)})
# with meshio: for block in mesh.cells: print(block.type, len(block.data))
```

Output:

```
bounding box: [0. 0. 0.] [1. 1. 1.] extent [1. 1. 1.]
cell types: {'triangle': 1, 'quad (or tetra)': 1}
```

Note that 4 vertices is ambiguous in a flat array (a quad or a tetrahedron): real formats store the cell type, so use it rather than the count when it is available.

## Areas, face normals and vertex normals

For a triangle (a, b, c), the cross product (b − a) × (c − a) is perpendicular to the face, its length is twice the area, and its direction follows the vertex order. **Vertex normals** average the normals of the faces around each vertex; summing the unnormalized cross products weights faces by area automatically. Scatter-add (`np.add.at`) does the accumulation without a Python loop over faces.

```python
def face_areas_normals(V, F):
    c = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    norm_c = np.linalg.norm(c, axis=1)
    return 0.5 * norm_c, c / norm_c[:, None], c    # areas, unit normals, raw cross products

def vertex_normals(V, F):
    _, _, c = face_areas_normals(V, F)             # |c| = 2 * area, so summing c is area-weighted
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], c)                  # scatter each face's vector to its three vertices
    return vn / np.linalg.norm(vn, axis=1, keepdims=True)

area, normal, _ = face_areas_normals(V2, F2)
print(f"total area {area.sum():.3f} (cube: 6), normal of face 0 {normal[0]}, vertex normal of vertex 0 "
      f"{np.round(vertex_normals(V2, F2)[0], 3)}")
```

Output:

```
total area 6.000 (cube: 6), normal of face 0 [ 0.  0. -1.], vertex normal of vertex 0 [-0.333 -0.667 -0.667]
```

## Unique edges and boundary edges

List each triangle's three edges, sort each pair so (i, j) and (j, i) are the same edge, and count with `np.unique`. In a closed manifold surface, every edge belongs to exactly **two** faces. Edges with **one** face are on a boundary (a hole, or the edge of an open surface); edges with **more than two** faces are non-manifold (T-junctions, internal walls). Both must be repaired before computing signed distances or volumes. Euler's formula V − E + F = 2 is a quick check for a closed surface of genus zero.

```python
def edges_and_boundary(F):
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    E.sort(axis=1)                                 # (i, j) and (j, i) are the same edge
    uniq, counts = np.unique(E, axis=0, return_counts=True)
    return uniq, uniq[counts == 1], uniq[counts > 2]   # all edges, boundary edges, non-manifold edges

E_all, E_boundary, E_nonmanifold = edges_and_boundary(F2)
print(f"edges {len(E_all)}, boundary {len(E_boundary)}, non-manifold {len(E_nonmanifold)}; "
      f"Euler characteristic V - E + F = {len(V2) - len(E_all) + len(F2)}")
_, b_open, _ = edges_and_boundary(F2[2:])          # remove two triangles: open a hole in the cube
print(f"after deleting 2 triangles: {len(b_open)} boundary edges (the rim of the hole)")
```

Output:

```
edges 18, boundary 0, non-manifold 0; Euler characteristic V - E + F = 2
after deleting 2 triangles: 4 boundary edges (the rim of the hole)
```

## Vertex adjacency in CSR form

A **compressed sparse row (CSR)** matrix stores, for each vertex v, its neighbours contiguously in `indices[indptr[v]:indptr[v + 1]]`. Building it from the unique edges (both directions) gives O(1) access to neighbours, the degree of each vertex (row lengths), graph Laplacians for smoothing, and the edge list a GNN needs. This is the same structure as a finite-element stiffness matrix's sparsity pattern. See [Assembly and sparsity](finite-element-method.md#assembly-and-sparsity).

```python
def vertex_adjacency(F, n_vertices):
    E, _, _ = edges_and_boundary(F)
    rows = np.concatenate([E[:, 0], E[:, 1]])
    cols = np.concatenate([E[:, 1], E[:, 0]])
    return sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_vertices, n_vertices)).tocsr()

A = vertex_adjacency(F2, len(V2))
print("neighbours of vertex 0:", A.indices[A.indptr[0]:A.indptr[1]].tolist(), "| degrees:", np.diff(A.indptr).tolist())
```

Output:

```
neighbours of vertex 0: [1, 2, 4, 5, 6] | degrees: [5, 5, 4, 4, 4, 4, 5, 5]
```

## Volume of a closed mesh

By the divergence theorem, the volume enclosed by a closed, consistently oriented triangle mesh is the sum over faces of the signed volumes of the tetrahedra formed by each face and the origin: V = (1/6) Σ v₀ · (v₁ × v₂). The result does not depend on where the origin is (for a closed surface), a **negative** volume means the normals point inward, and a result that changes when the mesh is translated means the surface is not closed. The same idea gives the centroid and inertia tensor.

```python
def closed_mesh_volume(V, F):
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0   # negative => inward normals

print(f"volume {closed_mesh_volume(V2, F2):.3f}; with flipped faces {closed_mesh_volume(V2, F2[:, ::-1]):.3f}; "
      f"after translating by (5, 5, 5) {closed_mesh_volume(V2 + 5.0, F2):.3f}")
```

Output:

```
volume 1.000; with flipped faces -1.000; after translating by (5, 5, 5) 1.000
```

## Drag force from pressure and wall shear

The force the fluid exerts on a body is the surface integral of pressure and wall shear stress:

```math
\mathbf F = \oint_S \Big( -(p - p_\infty)\,\mathbf n + \boldsymbol\tau_w \Big)\, dA
```

with **n** the outward normal (out of the body, into the fluid): pressure pushes **into** the body, against n. Drag is the component along the free-stream direction and lift the component normal to it; coefficients divide by the dynamic pressure and a reference area, C_D = F·e_x / (½ρU²A_ref).

**Practical points.**

- On a closed surface a uniform pressure integrates to zero, so p∞ only matters numerically; on **open** surfaces (half models, missing underbody parts) subtracting p∞ matters.
- If pressure is stored at vertices, average it to faces (or integrate with the linear shape functions).
- Use the exact areas and normals of the surface mesh, not point counts; check orientation first (volume sign).
- The split into pressure drag and friction drag is itself useful for engineers.

```python
def surface_force(V, F, p_face, tau_face, p_inf=0.0):
    """Per-face force from the fluid on a body: (-(p - p_inf) n + tau_w) * area. Normals must point out of the body."""
    area, n, _ = face_areas_normals(V, F)
    return ((-(p_face - p_inf))[:, None] * n + tau_face) * area[:, None]

uniform = surface_force(V2, F2, np.full(len(F2), 3.0), np.zeros((len(F2), 3))).sum(0)
centroids = V2[F2].mean(axis=1)
p_front = np.where(np.isclose(centroids[:, 0], 0.0), 1.0, 0.0)      # pressure 1 on the x = 0 face only
tau = np.tile([0.01, 0.0, 0.0], (len(F2), 1))                        # wall shear 0.01 in +x everywhere
_, n_face, _ = face_areas_normals(V2, F2)
tau[np.abs(n_face[:, 0]) > 0.5] = 0.0                                # no tangential shear on faces normal to x
force = surface_force(V2, F2, p_front, tau).sum(0)
print(f"uniform pressure: net force {np.round(uniform, 12)}; front pressure + shear: force {np.round(force, 4)} "
      f"(pressure part 1.0, shear part 4 faces x 0.01)")
```

Output:

```
uniform pressure: net force [0. 0. 0.]; front pressure + shear: force [1.04 0.   0.  ] (pressure part 1.0, shear part 4 faces x 0.01)
```

## Interpolating fields between meshes

Transfer a field from one mesh (or point set) to another: CFD results to a structural mesh, training meshes to a common set of query points, coarse to fine.

| Method | How | Notes |
| --- | --- | --- |
| Nearest neighbour | Value of the closest source point (k-d tree query) | Fast, piecewise constant, first-order |
| Inverse-distance weighting (IDW) | Weighted mean of the k nearest values, weights 1/d² | Smooth, simple; can blur sharp features and is not conservative |
| Barycentric (linear FE) | Locate the containing triangle or tetrahedron, use its shape functions | Exact for linear fields; needs point location |
| Radial basis functions | Fit an RBF interpolant to the source points | Smooth, accurate, more expensive |
| Conservative remapping | Integrate over intersections of source and target cells | Preserves integrals (mass, force); needed for coupled simulations |

```python
tree = cKDTree(src_pts)
_, nn_idx = tree.query(dst_pts, k=1)
nearest = src_val[nn_idx]                                        # nearest neighbour
d, knn_idx = tree.query(dst_pts, k=8)
w = 1.0 / np.maximum(d, 1e-12) ** 2
idw = (w * src_val[knn_idx]).sum(axis=1) / w.sum(axis=1)         # inverse-distance weighting
exact = dst_pts[:, 0]                                            # the field is f(x, y, z) = x
print(f"interpolating f = x from 2,000 scattered points: max error nearest {np.abs(nearest - exact).max():.3f}, "
      f"IDW (k = 8) {np.abs(idw - exact).max():.3f}")
```

Output:

```
interpolating f = x from 2,000 scattered points: max error nearest 0.101, IDW (k = 8) 0.057
```

## k-nearest-neighbour and radius graphs

When there is no mesh (point clouds from an STL, particles) or the mesh is too irregular, build a graph from geometry:

- **k-nearest-neighbour (k-NN) graph:** connect each point to its k closest points. Every node has the same in-degree, the graph adapts to local density (edges are short where points are dense), and it is always connected locally; but edges are not symmetric (j may be among i's neighbours without the converse), and on non-uniform sampling an edge can mean very different physical distances. Used by DGCNN and X-MeshGraphNet.
- **Radius graph:** connect all pairs closer than r. Symmetric, with edges of bounded physical length, so a message always means "within r", which is physically meaningful (GNS); but the degree varies with density (dense regions get many edges, sparse regions may be isolated), and memory grows quickly if r is large.
- **Choice:** radius graphs for particle simulations with roughly uniform density and physical interaction ranges; k-NN for non-uniformly sampled surfaces; often a radius graph with a cap on neighbours per node, or k-NN with a distance cutoff.
- **Both directions:** store each undirected edge twice for message passing; exclude self-loops.
- **Scale:** k-d trees build in O(N log N) and query in roughly O(log N) per point; GPU libraries (torch-cluster, PyTorch3D, FAISS) handle millions of points.

```python
def knn_graph(points, k):
    _, idx = cKDTree(points).query(points, k=k + 1)               # the first neighbour is the point itself
    src = np.repeat(np.arange(len(points)), k)
    dst = idx[:, 1:].reshape(-1)
    return np.stack([src, dst])                                   # (2, N * k), directed

def radius_graph(points, r):
    pairs = cKDTree(points).query_pairs(r, output_type="ndarray")  # unique pairs i < j within r
    return np.concatenate([pairs.T, pairs[:, ::-1].T], axis=1)    # both directions

g, rg = knn_graph(src_pts, 4), radius_graph(src_pts, 0.05)
deg = np.bincount(rg[0], minlength=len(src_pts))
print(f"k-NN graph (k = 4): {g.shape[1]} directed edges, every node degree 4 | radius graph (r = 0.05): "
      f"{rg.shape[1]} directed edges, degree min {deg.min()} max {deg.max()}, isolated nodes {np.sum(deg == 0)}")
```

Output:

```
k-NN graph (k = 4): 8000 directed edges, every node degree 4 | radius graph (r = 0.05): 1962 directed edges, degree min 0 max 6, isolated nodes 742
```

## Making Python fast on millions of points

1. **Profile first** (`cProfile`, `line_profiler`, `py-spy`): optimize the step that actually dominates.
2. **Vectorize:** no Python loops over points or elements; express operations as array expressions, fancy indexing (`V[F]`) and reductions.
3. **Scatter operations** for accumulation (`np.add.at`, `np.bincount` with weights, `torch.index_add_` or `scatter_add_`), the pattern behind vertex normals, assembly and message passing.
4. **Spatial data structures:** k-d trees or uniform grids instead of O(N²) distance matrices.
5. **Sparse matrices** for adjacency and Laplacians; never dense n × n arrays.
6. **Memory:** float32 where precision allows, chunked processing for queries over very large sets, memory-mapped or chunked storage (Zarr, HDF5).
7. **Compile what resists vectorization:** Numba for loops with irregular control flow; Cython or C++ extensions for hot kernels.
8. **GPU:** PyTorch, CuPy or JAX for large array operations; GPU neighbour search for graphs.
9. **Parallelize across files**, not within a small array: process many meshes concurrently with multiprocessing, Dask or Ray. See [Spark and Dask](scaling-to-large-meshes.md#spark-and-dask).

```python
import time
pts = rng.random((200_000, 3))
faces = rng.integers(0, len(pts), (400_000, 3))
vals = rng.random(len(faces))

t0 = time.perf_counter()
acc_loop = np.zeros(len(pts))
for f, v in zip(faces[:20_000], vals[:20_000]):                  # Python loop, on only 5% of the faces
    for i in f:
        acc_loop[i] += v
t_loop = (time.perf_counter() - t0) * 20                          # scaled to all faces
t0 = time.perf_counter()
acc = np.bincount(faces.ravel(), weights=np.repeat(vals, 3), minlength=len(pts))   # vectorized scatter-add
t_vec = time.perf_counter() - t0
check = np.bincount(faces[:20_000].ravel(), weights=np.repeat(vals[:20_000], 3), minlength=len(pts))
print(f"scatter-add over 400,000 faces: Python loop about {t_loop:.1f} s, np.bincount {1e3 * t_vec:.0f} ms, "
      f"same result on the subset: {np.allclose(check, acc_loop)}")
t0 = time.perf_counter()
_ = cKDTree(pts).query(pts[:50_000], k=8)
print(f"8 nearest neighbours of 50,000 points among 200,000 with a k-d tree: {time.perf_counter() - t0:.2f} s "
      f"(a dense distance matrix would need {50_000 * 200_000 * 8 / 1e9:.0f} GB)")
```

Output (timings from one run):

```
scatter-add over 400,000 faces: Python loop about 0.8 s, np.bincount 52 ms, same result on the subset: True
8 nearest neighbours of 50,000 points among 200,000 with a k-d tree: 0.26 s (a dense distance matrix would need 80 GB)
```

## In an interview

**Thirty-second answer to "You get flat vertex data and a list of cells: turn it into a mesh, identify the elements and give the bounding box."** "Reshape the coordinates to an n-by-3 array and check the length and index range. Parse the connectivity: if it is VTK-style, each cell's vertex count followed by its indices; group cells by type, using the stored type rather than the count because four vertices could be a quad or a tet. If it is a triangle soup like STL, weld duplicate vertices with np.unique on rounded coordinates and use the inverse indices as faces. The bounding box is the per-axis min and max, and its extent is a quick unit check. Then I would check boundary and non-manifold edges and the sign of the enclosed volume before using the mesh."

**Likely follow-ups**

- *How do you find boundary edges?* Sort each face's edges, count unique edges; count 1 is boundary, more than 2 is non-manifold.
- *How do you compute drag from a surface solution?* Sum −(p − p∞) n + τ_w over faces times area with outward normals, then project on the flow direction.
- *k-NN or radius graph?* Radius for physical interaction ranges and uniform density; k-NN for non-uniform sampling; cap degrees either way.

## References

- [meshio](https://github.com/nschloe/meshio), [PyVista](https://github.com/pyvista/pyvista) and [trimesh](https://github.com/mikedh/trimesh)
- [SciPy spatial (cKDTree)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html) and [SciPy sparse matrices](https://docs.scipy.org/doc/scipy/reference/sparse.html)
- M. Botsch, L. Kobbelt, M. Pauly, P. Alliez and B. Lévy, *Polygon Mesh Processing*, A K Peters, 2010.
- [VTK file formats and cell types](https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html)
- Interview reports: [PhysicsX on Glassdoor](https://www.glassdoor.com.mx/Entrevista/PhysicsX-Entrevista-E9306867-RVW97800229.htm) and [PhysicsX guide (Dataford)](https://dataford.io/interview-guides/physicsx/machine-learning-engineer)
