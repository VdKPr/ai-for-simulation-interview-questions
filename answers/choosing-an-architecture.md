# Choosing an Architecture

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Choose from the data and the job, not from fashion. If the design is described by a **few parameters**, start with a Gaussian process or a small MLP on those parameters. If the geometry can be represented on a **regular grid** without losing what matters (2D sections, voxelized shapes, image-like fields), a **U-Net** or a Fourier neural operator is fast and mature. If the data lives on **simulation meshes** and local physics matters, a **graph neural network** uses the mesh directly and generalizes well across mesh sizes, but needs many message-passing steps (or a multiscale design) for global effects and gets memory-hungry on large meshes. If the input is **raw CAD or STL** without a usable mesh, a **point-cloud network** samples the surface directly. If the problem needs **global context on millions of points** (pressure on a whole car, where the wake affects the front), **transformer-style neural operators** (Transolver, UPT, AB-UPT and relatives) are the current state of the art, at the cost of more data and compute. Whatever you choose, benchmark it against a simple baseline and test it on held-out designs and remeshed inputs.

**On this page**

- [The questions that decide the architecture](#the-questions-that-decide-the-architecture)
- [The main options](#the-main-options)
- [A decision guide](#a-decision-guide)
- [Worked examples](#worked-examples)
- [Common mistakes](#common-mistakes)
- [In an interview](#in-an-interview)

## The questions that decide the architecture

1. **What does the input look like?** A parameter vector, a structured grid, a simulation mesh, a surface mesh or STL, a point cloud, a CAD model?
2. **What must the output be?** A few scalars, a surface field, a volume field, a time history? At the input points or at arbitrary query points?
3. **How large are the inputs?** Thousands of nodes or tens of millions?
4. **How much data is there?** Tens of simulations or tens of thousands?
5. **How far does information travel?** Local physics (stress around a hole) or global effects (pressure everywhere depends on the whole flow)?
6. **How much does the geometry vary?** One family of morphs, or completely different topologies?
7. **What will be available at inference?** Will users have a volume mesh, or only a CAD file? Meshing at inference time can dominate latency.
8. **What are the constraints?** Latency, GPU memory, deployment hardware, need for uncertainty, physics constraints to enforce.

## The main options

| Architecture | Input it wants | Strengths | Weaknesses | Typical fit |
| --- | --- | --- | --- | --- |
| **GP / MLP on parameters** | Design parameter vector | Data-efficient, uncertainty (GP), simple | Needs a parametrization; no free-form geometry | Parametric studies with tens to hundreds of runs ([Gaussian processes](gaussian-processes.md)) |
| **CNN / U-Net** | Regular grid (images, voxels, SDF on a grid) | Fast, mature, multiscale by design, cheap to train | Resolution fixed by the grid; staircase boundaries; memory grows with grid size in 3D | 2D problems, voxelized shapes, near-image data |
| **Fourier neural operator** | Regular grid (Geo-FNO and GINO extend it) | Global receptive field per layer, resolution-flexible on grids | Uniform grids and periodicity assumptions; complex geometry needs extensions | Grid-based PDE benchmarks, weather-like data ([FNO](fourier-neural-operator.md)) |
| **GNN on the mesh** (MeshGraphNets and relatives) | Simulation mesh as a graph | Uses the native mesh; local, physics-like updates; relative features give translation invariance; handles varying sizes | Global effects need many layers or multiscale graphs; memory on large meshes; predictions can depend on the mesh | Mesh-based dynamics, structural FEM data, moderate mesh sizes ([GNNs](gnns-and-meshgraphnets.md)) |
| **Point-cloud networks** (PointNet++, DGCNN, Point Transformer) | Points sampled from a surface or volume | No mesh needed; works from STL; permutation invariant | Weaker built-in physics structure; neighbourhoods from k-NN rather than mesh connectivity | Drag or surface fields from CAD, classification ([Point-cloud learning](point-cloud-learning.md)) |
| **Transformer-style neural operators** (Transolver, UPT, AB-UPT, GeoTransolver, DoMINO-type models) | Point sets or meshes, queried anywhere | Global context; scale to millions of points through linear attention, slicing or latent tokens; flexible query points | More data and compute to train; many design choices | Large 3D aerodynamics and structural problems ([Transformers for physics](transformers-for-physics.md), [Architectures](physics-transformer-architectures.md)) |
| **Reduced-order models** (POD + regression) | Fields on one fixed mesh | Very data-efficient, interpretable | Needs a common mesh; linear subspace limits | One geometry with varying conditions ([ROMs](reduced-order-models.md)) |

## A decision guide

```
Is the design described by a few parameters, with tens to hundreds of runs?
 ├─ yes ─▶ GP or small MLP on parameters (baseline that must be beaten anyway).
 │         Need fields? POD + GP if the mesh is fixed.
 └─ no (free-form geometry) ─▶ What representation exists at inference?
        ├─ regular grid is acceptable ─▶ U-Net / FNO
        ├─ simulation mesh, moderate size, local physics ─▶ GNN (multiscale if information must travel far)
        ├─ only CAD / STL ─▶ point-cloud network, or a transformer operating on surface points
        └─ millions of points, global effects (car aero) ─▶ transformer-style neural operator
```

Two meta-rules:

- **Start simple, then justify complexity.** A baseline that is quick to train tells you how hard the problem is and gives a number the complex model must beat. See [Baselines to beat](evaluation-metrics.md#baselines-to-beat).
- **The data budget limits the model.** With 50 simulations, a large transformer trained from scratch is unlikely to beat a well-regularized GNN or a GP; with 5,000 it may. See [Learning from small simulation datasets](learning-from-small-simulation-datasets.md).

## Worked examples

**Drag and surface pressure of new car shapes from STL files, 500 to 8,000 training cases.** Users have CAD, not volume meshes, and the answer has global dependencies. A transformer-style neural operator on surface points (with a geometry encoder) is the current state of the art on public car datasets; a point-cloud network is a strong simpler baseline; a GP on morph parameters is a baseline for drag alone. See [Design: car aerodynamics surrogate](design-car-aerodynamics-surrogate.md).

**Stress field of brackets from FEA meshes, a few hundred designs.** The data is on tetrahedral meshes and stress is mostly local around features and loads. A GNN on the mesh (with node types for supports and loads), possibly multiscale, is natural; a point-cloud network on surface samples is an alternative if meshes are not available at inference. See [Design: stress prediction platform](design-stress-prediction-platform.md).

**Transient cloth or plate deformation over many time steps.** Mesh-based dynamics with local interactions: MeshGraphNets-style GNNs were designed for this, trained with noise injection for stable rollouts. See [Autoregressive rollouts](autoregressive-rollouts.md).

**2D airfoil flow fields (AirfRANS).** Meshes of about 250,000 cells; GNNs, point-cloud networks and transformers all work; a 2D U-Net on an interpolated grid is a quick baseline but loses boundary-layer detail.

**Temperature field of a battery module with 20 design parameters and 150 runs.** Parameters exist and data is small: GP or MLP on parameters, or POD plus GP if the mesh is shared. Free-form geometry models would be overkill.

## Common mistakes

- **Choosing the architecture before looking at the data** (or because it was in the latest paper).
- **Ignoring what is available at inference.** A model that needs a volume mesh cannot give "results in seconds from a CAD file" if meshing takes 20 minutes.
- **No baseline.** Without a simple comparison, nobody knows whether the complex model helps.
- **Testing only on random splits** of one design family, which hides poor generalization to new shapes. See [What generalizing to new geometries means](out-of-distribution-and-generalization.md#what-generalizing-to-new-geometries-means).
- **Not testing mesh dependence.** The same shape meshed differently should give the same prediction. See [Why remeshing can change a prediction](geometry-representations.md#why-remeshing-can-change-a-prediction).
- **Underestimating memory.** GNNs and full attention on million-node meshes need careful engineering. See [Scaling to large meshes](scaling-to-large-meshes.md).

## In an interview

**Thirty-second answer.** "It depends on the input representation, the output, the size of the problem, the amount of data and how global the physics is. With a few design parameters and little data, a GP or MLP. On regular grids, a U-Net or FNO. On native simulation meshes with mostly local physics, a GNN, multiscale if information must travel far. From raw CAD or STL, a point-cloud model. For millions of points with global effects, like a whole car, a transformer-style neural operator such as Transolver or AB-UPT. I would always start with a simple baseline, and test on held-out designs and remeshed inputs."

**Likely follow-ups**

- *GNN or transformer for a car?* Transformers capture the global dependence and scale with linear attention; GNNs need many layers or multiscale graphs. See [GNN or transformer?](transformers-for-physics.md#gnn-or-transformer).
- *Why not a CNN for 3D?* Voxel grids fine enough for thin boundary layers are huge, and staircase boundaries lose surface detail.
- *What if users only have STL files?* Use models that work on surface samples, not volume meshes, or build graphs from point clouds as X-MeshGraphNet does.

## References

- [Learning Mesh-Based Simulation with Graph Networks (MeshGraphNets, Pfaff et al.)](https://arxiv.org/abs/2010.03409)
- [Transolver: A Fast Transformer Solver for PDEs on General Geometries (Wu et al.)](https://arxiv.org/abs/2402.02366)
- [AB-UPT: Scaling Neural CFD Surrogates for High-Fidelity Automotive Aerodynamics (Alkin et al.)](https://arxiv.org/abs/2502.09692)
- [PointNet++ (Qi et al.)](https://arxiv.org/abs/1706.02413)
- [Fourier Neural Operator (Li et al.)](https://arxiv.org/abs/2010.08895)
- [AirfRANS: benchmarks of MLP, GraphSAGE, PointNet and Graph U-Net on airfoils (Bonnet et al.)](https://arxiv.org/abs/2212.07564)
