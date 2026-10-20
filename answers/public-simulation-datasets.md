# Public Simulation Datasets and Benchmarks

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Public datasets for AI for simulation fall into three groups. **Industrial-style geometry datasets** give 3D shapes with high-fidelity simulation results: DrivAerNet and DrivAerNet++, DrivAerML, AhmedML, WindsorML and Luminary's SHIFT datasets for car aerodynamics, AirfRANS for 2D airfoils, SimJEB and DeepJEB for structural brackets. **Benchmark suites** collect many PDE problems with standard splits and baselines: PDEBench, The Well, PDEArena, CFDBench and APEBench, plus LagrangeBench for particle fluids. **Classic neural-operator problems** (Darcy flow, Burgers, 2D Navier–Stokes vorticity) come from the FNO and DeepONet papers. Before using any of them, check three things: the **licence** (several popular ones are non-commercial), the **physics fidelity** (RANS or scale-resolving, mesh resolution, converged or not), and what is **missing** relative to the real use case (yaw angles, operating conditions, transient data, uncertainty estimates).

**On this page**

- [Car aerodynamics datasets](#car-aerodynamics-datasets)
- [Airfoil and wing datasets](#airfoil-and-wing-datasets)
- [Structural datasets](#structural-datasets)
- [Other mesh and particle datasets](#other-mesh-and-particle-datasets)
- [Licences](#licences)
- [What an external-aerodynamics dataset contains](#what-an-external-aerodynamics-dataset-contains)
- [Standard benchmark problems for neural operators](#standard-benchmark-problems-for-neural-operators)
- [PDEBench, The Well, PDEArena, CFDBench and APEBench](#pdebench-the-well-pdearena-cfdbench-and-apebench)
- [Benchmarks for time-dependent neural PDE solvers](#benchmarks-for-time-dependent-neural-pde-solvers)
- [How to choose a dataset for a project](#how-to-choose-a-dataset-for-a-project)
- [In an interview](#in-an-interview)

## Car aerodynamics datasets

| Dataset | Geometries | Simulation | Licence | Notes |
| --- | --- | --- | --- | --- |
| [DrivAerNet](https://arxiv.org/abs/2403.08055) (MIT, 2024) | 4,000 parametric variants of the DrivAer car, about 0.5 million surface faces each | High-fidelity CFD; full 3D pressure, velocity and wall shear stress | See the repository | Models wheels and underbody; introduced the RegDGCNN drag-regression baseline |
| [DrivAerNet++](https://arxiv.org/abs/2406.09624) (MIT, 2024) | 8,000 designs: fastback, notchback and estateback, varied underbodies and wheels | High-fidelity CFD; meshes, parametric models, coefficients, surface and volume fields, part segmentation, point clouds | CC BY-NC (non-commercial) | More than 39 TB; benchmark results for drag prediction |
| [DrivAerML](https://arxiv.org/abs/2408.11969) (2024) | 500 morphed variants of the DrivAer notchback | Scale-resolving hybrid RANS-LES with validated automatic workflows | CC-BY-SA | Industry partners; open formats; used by DoMINO, AB-UPT and GeoTransolver |
| [AhmedML](https://arxiv.org/abs/2407.20801) (2024) | 500 variants of the Ahmed body (a simplified car shape) | Time-accurate hybrid RANS-LES in OpenFOAM | CC-BY-SA | Boundary, volume, geometry and time-averaged forces; OpenFOAM case set-up included |
| [WindsorML](https://arxiv.org/abs/2407.19320) (2024) | 355 variants of the Windsor body | Wall-modelled LES on GPUs with a Cartesian immersed-boundary method, more than 280 million cells | CC-BY-SA | Time-averaged volume and boundary data, forces and moments |
| [SHIFT-SUV](https://huggingface.co/datasets/luminary-shift/SUV) (Luminary Cloud) | Thousands of parametrically morphed variants of the AeroSUV platform (from FKFS) | High-fidelity CFD, developed with Honda and NVIDIA | CC-BY-NC-4.0 | Terabyte scale; also used in the AB-UPT and GeoTransolver reports |

The Ashton et al. datasets (DrivAerML, AhmedML, WindsorML) were designed together with industry, use **scale-resolving** turbulence modelling and permissive licences, which makes them the closest public analogue of industrial car aerodynamics.

## Airfoil and wing datasets

- **[AirfRANS](https://arxiv.org/abs/2212.07564)** (NeurIPS 2022 Datasets and Benchmarks): 1,000 steady 2D incompressible RANS simulations (OpenFOAM simpleFoam, k-ω SST) over NACA 4- and 5-digit airfoils, at Reynolds numbers from 2 to 6 million and angles of attack from −5° to 15°, on meshes of more than 250,000 cells. It defines four tasks: full data (800 train, 200 test), scarce data (200 train), Reynolds-number extrapolation (train on 3 to 5 million) and angle-of-attack extrapolation (train on −2.5° to 12.5°). It also stresses metrics on surface forces and the **rank correlation** of drag and lift, because design optimization needs the right ranking. The repository's licence is ODbL. A very good first field-prediction project.
- **[SHIFT-Wing](https://huggingface.co/datasets/luminary-shift/WING)** (Luminary Cloud): transonic wing aerodynamics, CC-BY-NC-4.0.

## Structural datasets

- **[SimJEB](https://arxiv.org/abs/2105.03534)** (Simulated Jet Engine Bracket Dataset, 2021): 381 crowd-sourced bracket designs from the GrabCAD Jet Engine Bracket Challenge (over 700 entries from 320 designers in 56 countries), cleaned, categorized, meshed and analysed with FEA under the original competition's load cases. All designs share the same function, loads and supports, but are far more diverse and realistic than synthetic parametric sets. The data is under ODC-By, while the CAD models carry a non-commercial restriction. A strong dataset for geometry-to-stress surrogates and for practising realistic geometry processing.
- **[DeepJEB](https://arxiv.org/abs/2406.09047)** (2024): a larger synthetic jet-engine bracket dataset created with deep generative models and automated simulation pipelines. The authors report up to 23% higher R² and over 70% lower mean absolute percentage error for surrogates trained on it compared with traditional bracket datasets.
- **MeshGraphNets datasets** (DeepMind): includes *DeformingPlate*, a structural (hyper-elastic plate deformed by an actuator) mesh dataset alongside cloth and flow datasets, used widely to benchmark mesh-based GNNs.

Public structural datasets are still far scarcer than aerodynamics ones; much of the data in industry is proprietary.

## Other mesh and particle datasets

- **[EAGLE](https://arxiv.org/abs/2302.10803)** (ICLR 2023): about 1.1 million 2D meshes from unsteady flows produced by a moving flow source (like a drone) in 600 scenes of three types; introduced a mesh transformer with node clustering and global attention.
- **[LagrangeBench](https://arxiv.org/abs/2309.16342)** (NeurIPS 2023): seven smoothed particle hydrodynamics datasets (four 2D, three 3D), including Taylor-Green vortex, lid-driven cavity, reverse Poiseuille flow and dam break, with JAX implementations of GNS and SEGNN and physical metrics such as kinetic-energy error and the Sinkhorn distance between particle distributions.
- **[BubbleML](https://arxiv.org/abs/2307.14623)**: 79 boiling simulations (pool, flow and sub-cooled boiling) for multiphysics phase-change learning.
- **CAD datasets** for geometry and CAD generation: ABC, Fusion 360 Gallery, DeepCAD. See [CAD models and CAD datasets](geometry-representations.md#cad-models-and-cad-datasets).

## Licences

**Several of the most-used datasets are non-commercial.** DrivAerNet++ and the SHIFT datasets are CC BY-NC, and SimJEB's CAD models are non-commercial. That is fine for research and personal projects, but a company cannot simply train a product on them. DrivAerML, AhmedML and WindsorML are CC-BY-SA (commercial use allowed, with attribution and share-alike conditions); AirfRANS is ODbL.

Practical rules:

- Read the licence of the *data*, not only of the code repository; they can differ.
- Respect attribution requirements (for example, SHIFT-SUV asks users to attribute FKFS for the original AeroSUV model).
- In an interview, mentioning licence checks unprompted signals real-world experience.

## What an external-aerodynamics dataset contains

**Typically present:**

- **Geometry:** the surface as STL or a surface mesh, sometimes the parametric design variables and the morphing parameters.
- **Surface fields:** pressure (or pressure coefficient) and wall shear stress on the body.
- **Volume fields:** velocity, pressure and turbulence quantities, usually **time-averaged** for scale-resolving runs, sometimes subsampled to keep the size manageable.
- **Integrated quantities:** drag, lift (often front and rear), side force and moments, as coefficients.
- **Metadata:** inlet speed and Reynolds number, reference area and length, solver and mesh settings.

**Typically missing, or limited:**

- **Operating-condition variation.** Most datasets use one speed, zero yaw (no crosswind) and one ride height. Real development needs yaw sweeps, varying ride heights and wheel rotation conditions.
- **Transient data.** Instantaneous fields and spectra are rarely provided, only time averages, so unsteady loads, aeroacoustics and wake dynamics cannot be learned.
- **Cooling and underhood flow.** Grilles are often closed and engine bays simplified, although cooling drag is a real part of a car's drag.
- **Uncertainty information.** Mesh-convergence and averaging-time uncertainties per sample are rarely given, so the label noise floor is unknown. See [When the labels have discretization error](verification-validation-and-mesh-convergence.md#when-the-labels-have-discretization-error).
- **Experimental validation per geometry.** At best the base geometry has been compared with wind-tunnel data.
- **Design diversity.** Variants are morphs of one or a few base cars; truly new shapes (different vehicle classes, new topologies) are out of distribution. See [What generalizing to new geometries means](out-of-distribution-and-generalization.md#what-generalizing-to-new-geometries-means).
- **Consistent meshes.** Each design has its own mesh, so there is no node-to-node correspondence between samples.
- **Permissive licences,** for some of the largest sets.

## Standard benchmark problems for neural operators

The original FNO and DeepONet papers popularized a set of problems that remain the common reference points:

| Problem | Type | Input → output | Why it is used |
| --- | --- | --- | --- |
| **Darcy flow** (2D) | Elliptic | Permeability (often piecewise constant) → pressure | Steady, rough coefficients; tests learning a solution operator of an elliptic PDE |
| **Burgers' equation** (1D) | Nonlinear advection-diffusion | Initial condition → solution at a later time | Shocks or steep fronts at low viscosity; cheap |
| **Navier–Stokes, vorticity form** (2D, periodic) | Time-dependent, nonlinear | Vorticity history → future vorticity, at viscosities down to 10⁻⁵ | Turbulence-like dynamics; tests rollouts and resolution |
| **Geo-FNO problems**: airfoil, pipe flow, elasticity, plasticity | Irregular geometries | Geometry (or mesh) → fields | Tests non-uniform meshes |
| **Advection, diffusion-reaction, shallow water, compressible flow** | Various | Initial / boundary conditions → fields | Included in PDEBench and similar suites |

The CNO paper added a "representative PDE benchmark" suite with multi-scale solutions (for example Poisson, wave, transport, Allen-Cahn, Navier–Stokes, Darcy and compressible Euler problems). These benchmarks are useful for comparing architectures, but they are far simpler than industrial problems: periodic or simple domains, uniform grids, and tens of thousands of unknowns rather than millions.

## PDEBench, The Well, PDEArena, CFDBench and APEBench

| Suite | What it is | Use it for |
| --- | --- | --- |
| **[PDEBench](https://arxiv.org/abs/2210.07182)** (NeurIPS 2022 D&B) | Time-dependent PDE tasks across a wide range of equations (advection, Burgers, diffusion-reaction, compressible and incompressible Navier–Stokes, Darcy, shallow water and more), with large ready-to-use datasets over many initial and boundary conditions and parameters, plus baselines (FNO, U-Net, PINN) and data-generation code | Comparing new architectures with standard baselines; parameter generalization |
| **[The Well](https://arxiv.org/abs/2412.00568)** (Polymathic AI, NeurIPS 2024 D&B) | 15 TB across 16 datasets of diverse spatiotemporal physics: biological systems, fluid dynamics, acoustic scattering, magneto-hydrodynamics, supernova explosions, with a unified PyTorch interface | Testing models on harder, more varied physics; pretraining foundation models |
| **[PDEArena](https://arxiv.org/abs/2209.15616)** (Microsoft) | A PyTorch benchmark framework with fluid problems (vorticity-stream and velocity forms) comparing FNO, ResNet and modern U-Net variants, including generalization across PDE parameters and time scales | Strong, well-tuned baselines for time-dependent fluid surrogates |
| **[CFDBench](https://arxiv.org/abs/2310.05963)** | Four classic flows: lid-driven cavity, laminar flow in a circular tube, dam flow over steps, and the cylinder wake (Kármán vortex street), with 739 cases and 302,000 frames | Generalization to unseen boundary conditions, fluid properties and domain shapes |
| **[APEBench](https://arxiv.org/abs/2411.00180)** (TUM, NeurIPS 2024) | 46 PDEs in 1D, 2D and 3D generated by an integrated, differentiable JAX pseudo-spectral solver; a taxonomy of unrolled training; an identifier for PDE dynamics tied to classical stability criteria; emphasis on rollout metrics | Autoregressive emulators, unrolled and differentiable-physics training, neural-hybrid models |

## Benchmarks for time-dependent neural PDE solvers

For rollouts specifically:

- **APEBench** is built around autoregressive rollouts, with long-horizon metrics and the solver in the loop for unrolled training.
- **PDEArena** and **PDEBench** provide time-dependent fluid problems with baselines.
- **The Well** has long, complex trajectories across very different physics.
- **LagrangeBench** for particle-based (Lagrangian) dynamics.
- **MeshGraphNets datasets** (cylinder flow, airfoil, cloth, deforming plate) for mesh-based rollouts.
- **EAGLE** for unsteady flows on changing scenes with large meshes.

What to report on them: error versus rollout time, the time until correlation with the reference drops below a threshold, energy spectra for turbulent flows, and stability over long horizons. See [Evaluating a rollout](autoregressive-rollouts.md#evaluating-a-rollout).

## How to choose a dataset for a project

1. **Match the job.** For a car-aerodynamics role, DrivAerML or AhmedML (permissive licences, high fidelity) or DrivAerNet++ (scale, non-commercial). For structures, SimJEB. For a first field-prediction project on a laptop, AirfRANS.
2. **Check the licence** against how you will use the result.
3. **Check the physics:** solver, turbulence model, mesh resolution, convergence, and whether it matches the regime you care about.
4. **Use the official splits** when comparing with papers, and add your own harder split (held-out family or extrapolation) to show judgement.
5. **Start with a baseline** a reviewer would expect (GP on parameters, PointNet or a GNN), then your model.

## In an interview

**Thirty-second answer to "Which public datasets would you use?"** "For car aerodynamics, DrivAerML, AhmedML and WindsorML: scale-resolving CFD with permissive CC-BY-SA licences; DrivAerNet++ for scale, but it is non-commercial. AirfRANS for 2D airfoils with ready-made extrapolation tasks. SimJEB and DeepJEB for structural brackets. For benchmarking architectures, PDEBench, PDEArena, The Well and APEBench. I always check the licence, the fidelity and what is missing, such as yaw cases and uncertainty."

**Likely follow-ups**

- *What is usually missing from car datasets?* Operating-condition variation (yaw, ride height), transient data, cooling flows, per-sample uncertainty, and design diversity beyond morphs.
- *Can a company train its product on DrivAerNet++?* Not under its non-commercial licence without permission.
- *Why use the extrapolation splits of AirfRANS?* Random splits only test interpolation; Reynolds and angle-of-attack extrapolation shows how the model behaves out of distribution.

## References

All dataset papers are linked in the tables above. Also:

- [AirfRANS documentation](https://airfrans.readthedocs.io/en/latest/notes/dataset.html)
- [Fourier Neural Operator for Parametric Partial Differential Equations (Li et al.): Darcy, Burgers and Navier–Stokes benchmarks](https://arxiv.org/abs/2010.08895)
- [Convolutional Neural Operators (Raonić et al.): representative PDE benchmarks](https://arxiv.org/abs/2302.01178)
- [Learning Mesh-Based Simulation with Graph Networks (Pfaff et al.): MeshGraphNets datasets](https://arxiv.org/abs/2010.03409)
- [Learning resources: datasets and benchmarks](../resources.md#datasets-and-benchmarks)
