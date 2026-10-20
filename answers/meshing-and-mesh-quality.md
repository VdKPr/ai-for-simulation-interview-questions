# Meshing and Mesh Quality

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A mesh divides the geometry into cells or elements on which the equations are discretized. A **structured** mesh has a regular, implicit (i, j, k) connectivity, like a deformed grid; it is efficient and accurate but hard to fit to complex shapes. An **unstructured** mesh stores its connectivity explicitly and can fill any shape automatically. **Tetrahedra** mesh almost anything automatically; **hexahedra** are more accurate and efficient per cell but hard to generate for complex parts; **prisms** (wedges) stacked in thin layers resolve boundary layers at walls; polyhedra are popular in CFD. **Mesh quality** measures how far the cells are from ideal shapes (aspect ratio, skewness, Jacobian, orthogonality, growth rate). Poor quality degrades accuracy, slows or breaks convergence, and in explicit dynamics a single tiny or distorted element can shrink the time step of the whole model.

**On this page**

- [Structured vs unstructured meshes](#structured-vs-unstructured-meshes)
- [Element shapes: tetrahedra, hexahedra, prisms and others](#element-shapes-tetrahedra-hexahedra-prisms-and-others)
- [Mesh quality metrics](#mesh-quality-metrics)
- [How mesh quality affects a simulation](#how-mesh-quality-affects-a-simulation)
- [Mesh sizing: where the cells go](#mesh-sizing-where-the-cells-go)
- [Meshing tools](#meshing-tools)
- [Why this matters for machine learning](#why-this-matters-for-machine-learning)
- [In an interview](#in-an-interview)

## Structured vs unstructured meshes

**Structured meshes** have the topology of a regular grid. Every interior node has the same number of neighbours, and neighbours are found by index arithmetic: the node to the right of (i, j, k) is (i+1, j, k). The grid can be stretched and curved to follow a body, as in the O-grids and C-grids wrapped around airfoils.

- *Pros:* little memory for connectivity, efficient solvers, very good accuracy per cell, easy to align cells with the flow or a boundary layer.
- *Cons:* hard to fit to complex shapes; usually needs manual decomposition into blocks (**multi-block structured** meshes), which takes expert time.

**Unstructured meshes** store which nodes each element uses (the connectivity table). Elements of any shape can be mixed, and neighbours must be looked up.

- *Pros:* automatic meshing of complex geometry; local refinement anywhere.
- *Cons:* more memory and indirect addressing, and usually more cells for the same accuracy.

```
 structured (deformed grid)            unstructured (triangles)
 ┌───┬───┬───┬───┐                     ╱╲───╱╲───╱╲
 │   │   │   │   │                    ╱  ╲ ╱  ╲ ╱  ╲
 ├───┼───┼───┼───┤                   ╱────╳────╳────╲
 │   │   │   │   │                    ╲  ╱ ╲  ╱ ╲  ╱
 └───┴───┴───┴───┘                     ╲╱───╲╱───╲╱
 neighbours by index (i±1, j±1)        neighbours from a connectivity table
```

**Hybrid meshes** combine them: prism layers or structured blocks near walls, unstructured tetrahedra or polyhedra in the rest of the domain. This is the norm in industrial CFD.

## Element shapes: tetrahedra, hexahedra, prisms and others

| Shape | Strengths | Weaknesses | Typical use |
| --- | --- | --- | --- |
| **Tetrahedron** | Fills any geometry automatically; robust meshers | Many more cells for the same accuracy; linear tets are too stiff in bending; poor in thin boundary layers | Complex solid parts in FEA (with quadratic tets); volume filling in CFD |
| **Hexahedron** | Most accurate and efficient per cell; aligns well with flow and stress gradients | Hard to generate automatically for complex shapes | Swept and mapped parts in FEA; structured CFD; Cartesian background meshes |
| **Prism / wedge** | Thin, high-aspect-ratio layers on walls, extruded from the surface triangles | Only meaningful as layers | Boundary-layer ("inflation") layers in CFD; thin regions |
| **Pyramid** | Joins hex faces to tet faces | Lower quality; transitional only | Transition zones in hybrid meshes |
| **Polyhedron** | Many neighbours per cell, so good gradient estimates; fewer cells than tets for the same accuracy | Only in solvers that support it | General-purpose CFD (finite-volume codes) |
| **Shell / beam elements** | Thin structures without meshing the thickness | Idealization; needs care at junctions | Sheet metal, car bodies, frames ([Elements](elements-and-shape-functions.md#element-families)) |

**Cartesian and cut-cell (octree) meshes** start from a regular hexahedral background grid, refine it near the geometry, and cut or snap cells to the surface (OpenFOAM's snappyHexMesh works this way). They are highly automatic and robust for dirty geometry. **Immersed-boundary methods** go further and do not fit the mesh to the body at all; the WindsorML dataset, for example, was produced with a Cartesian immersed-boundary method.

## Mesh quality metrics

Each metric measures one way a cell can deviate from an ideal shape. Different solvers define and threshold them differently, so always check the solver's own documentation.

| Metric | What it measures | Ideal | Problem when |
| --- | --- | --- | --- |
| **Aspect ratio** | Ratio of the longest to the shortest dimension | 1 (but high values are fine in boundary layers, aligned with the flow) | High in regions with gradients in the long direction |
| **Skewness** | How far the cell is from an equilateral or equiangular shape | 0 | Close to 1 (degenerate, flattened cells) |
| **Jacobian (ratio)** | Distortion of the mapping from the reference element; negative means inverted | Positive and uniform across the element | Near zero or negative: element is invalid |
| **Minimum / maximum angle** | Sharpness of corners | 60° for triangles, 90° for quads | Very small or close to 180° |
| **Non-orthogonality** (finite volume) | Angle between the line joining two cell centres and the normal of their shared face | 0° | Large angles (OpenFOAM's checkMesh, for example, flags faces above 70° as severely non-orthogonal) |
| **Warpage** | How far a quad face is from being planar | 0 | Large: inaccurate face integrals |
| **Growth (expansion) ratio** | Size ratio of adjacent cells | Close to 1; about 1.2 or less is common practice in boundary layers | Sudden jumps in cell size |
| **Volume ratio / minimum volume** | Neighbour size jumps; tiny cells | Smooth variation | Tiny cells limit explicit time steps; negative volumes stop solvers |

## How mesh quality affects a simulation

- **Accuracy.** Interpolation and gradient errors grow with distortion. In finite volumes, non-orthogonality and skewness introduce errors in the face fluxes; in convection-dominated flow, cells not aligned with the flow add numerical diffusion that smears gradients.
- **Convergence.** Badly shaped elements make the system ill-conditioned, so iterative solvers need more iterations or stall. Highly non-orthogonal cells may need extra correction loops or limiters.
- **Stability and validity.** A negative Jacobian or negative cell volume stops most solvers. In large-deformation analyses, elements can invert during the run.
- **Explicit time step.** The stable time step of an explicit code scales with the smallest element dimension divided by the wave speed. One tiny sliver element can make the whole crash run many times slower. See [The CFL condition](time-integration-and-stability.md#the-cfl-condition).
- **Locking and stiffness.** Element type and quality also change structural stiffness. See [Why linear tetrahedra are too stiff](elements-and-shape-functions.md#why-linear-tetrahedra-are-too-stiff-in-bending).

## Mesh sizing: where the cells go

Mesh quality is not only about shapes; it is also about putting resolution where the solution varies:

- **Near stress concentrations** (fillets, holes, notches) in FEA: several elements across a fillet radius.
- **Near walls in CFD:** prism layers sized for the wall treatment, with the first cell height chosen for a target y+. See [Boundary layers and y+](boundary-layers-and-wall-treatment.md#the-boundary-layer-and-y).
- **In wakes, shear layers and separation zones,** with refinement boxes behind a car or around its wheels.
- **Through the thickness** of thin parts, or use shells instead.
- **Smooth transitions** between fine and coarse regions (growth ratio close to 1).

Then **check mesh independence**: refine systematically and confirm the quantities of interest stop changing. See [Mesh convergence](verification-validation-and-mesh-convergence.md#mesh-convergence-and-the-grid-convergence-index).

## Meshing tools

- **Open source:** Gmsh, Netgen, TetGen, OpenFOAM's blockMesh and snappyHexMesh, cfMesh.
- **Commercial pre-processors and meshers:** ANSA (BETA CAE), HyperMesh (Altair), Ansys Meshing and Fluent Meshing, Simcenter STAR-CCM+ meshing, Fidelity Pointwise (Cadence), and the meshers built into most FEA suites.
- **Geometry kernels for scripted CAD and meshing:** Open CASCADE (used by Gmsh and CadQuery).

## Why this matters for machine learning

- **Every sample has a different mesh.** Node counts, element types and connectivity vary across a dataset, so models must handle variable-size inputs: graphs, point clouds or set-based transformers. See [Choosing an architecture](choosing-an-architecture.md).
- **Mesh density is a hidden weighting.** Meshes are fine where the solution changes fast (near walls, notches). A loss averaged over nodes therefore over-weights those regions and under-weights coarse regions. Weight by cell area or volume when the metric should reflect the physical domain. See [Field metrics](evaluation-metrics.md#field-metrics).
- **Predictions should not depend on the mesh.** The same shape meshed twice should give nearly the same prediction. Test this explicitly. See [Why remeshing can change a prediction](geometry-representations.md#why-remeshing-can-change-a-prediction).
- **Meshing at inference time costs time.** If a surrogate needs a volume mesh as input, meshing may dominate the end-to-end time of a "seconds" prediction. Methods that build their graphs or point clouds directly from the STL (for example X-MeshGraphNet, or point-cloud models) avoid this.
- **Mesh quality is a data-quality issue.** Failed or poor meshes produce bad labels; record mesh statistics with every sample and filter or investigate outliers. See [Generating simulation training data](generating-simulation-training-data.md).

## In an interview

**Thirty-second answer.** "Structured meshes have implicit grid connectivity: efficient and accurate but hard to fit complex shapes. Unstructured meshes store connectivity explicitly and can mesh anything automatically. Tets mesh anything but need many cells and are stiff if linear; hexes are more accurate per cell but hard to generate; prisms give thin, stretched layers for boundary layers; polyhedra are common in CFD. Quality metrics such as aspect ratio, skewness, Jacobian, non-orthogonality and growth rate tell you how distorted the cells are; poor quality hurts accuracy, convergence and, in explicit codes, the time step."

**Likely follow-ups**

- *Why prisms near walls?* Boundary layers have steep gradients normal to the wall and gentle ones along it, so thin, stretched cells aligned with the wall resolve them efficiently.
- *A crash model runs ten times slower than expected. What do you check?* The smallest elements, which set the explicit time step.
- *How does mesh density bias a neural network's loss?* Fine regions have more nodes, so an unweighted node average over-weights them.

## References

- [Gmsh reference manual](https://gmsh.info/doc/texinfo/gmsh.html)
- [OpenFOAM documentation](https://www.openfoam.com/documentation/overview)
- [X-MeshGraphNet: Scalable Multi-Scale Graph Neural Networks for Physics Simulation (Nabian et al.)](https://arxiv.org/abs/2411.17164)
- [WindsorML: High-Fidelity CFD Dataset for Automotive Aerodynamics (Ashton et al.)](https://arxiv.org/abs/2407.19320)
