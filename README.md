# AI for Simulation Interview Questions and Answers

> Interview questions and answers for jobs that combine machine learning with engineering simulation: physics AI, scientific machine learning, neural surrogates and AI for CAE.
>
> Useful for roles such as:
>
> - Machine Learning Engineer, Physics AI
> - Scientific Machine Learning Engineer / Research Engineer
> - Applied Research Scientist, AI for Simulation
> - Forward Deployed Engineer / Application Engineer, Engineering AI
> - Simulation (CAE) Engineer moving into machine learning
> - Data Scientist working with simulation or test data

The questions come from three places: what AI-for-simulation companies ask for in their job descriptions, what candidates report from their interviews, and the core papers and courses of the field. Answers link either to a page in [`answers/`](answers/README.md), written for this guide and reused wherever the same concept comes up, or to the primary source: the paper, the documentation or a lecture.

## What These Interviews Test

Based on job descriptions and candidate reports (sources in [industry.md](industry.md)):

- **Coding on geometry data.** Candidates report PhysicsX coding rounds that include converting flat vertex data into a mesh format, and live coding on a mesh dataset: identifying the elements and finding the bounding box. See [Coding](#coding-and-practical-implementation) and [Mesh data processing](answers/mesh-data-processing.md).
- **3D deep learning on meshes and point clouds.** Graph networks, point-cloud networks and transformers appear in PhysicsX, NVIDIA and Luminary job descriptions.
- **Simulation literacy.** CFD, FEA and thermal analysis, and validating models against ground-truth simulation (Neural Concept, Luminary, NVIDIA).
- **Uncertainty and honest evaluation.** For example, how a Gaussian process estimates uncertainty (reported in a PhysicsX interview), and "rigorous experimentation and honest evaluation of results" (Luminary job description).
- **Engineering and MLOps.** Data pipelines, Docker and Kubernetes, cloud platforms, distributed computing, experiment tracking and model registries.
- **Customer-facing work.** Scoping projects with engineering customers, explaining what a model can and cannot do, and travel to customer sites for forward-deployed roles.
- **Presenting your own project.** A technical presentation of past work is a reported final-round format.

## How to Use This Guide

1. Read [Must Know](#must-know) first.
2. Then work on whichever side is weaker for you: simulation fundamentals or machine learning.
3. For each question, answer it out loud in about two minutes **before** opening the link. Then check.
4. Write the [coding questions](#coding-and-practical-implementation) by hand, without AI help.
5. Use the [system design framework](answers/system-design-framework.md) for every design question.
6. Prepare your own stories for the [behavioral questions](#behavioral-and-customer-facing-scenarios). These interviews reward people who can say when a model should *not* be trusted.

## Table of Contents

- [Must Know](#must-know)
- [Simulation Fundamentals: PDEs and Numerical Methods](#simulation-fundamentals-pdes-and-numerical-methods)
- [Structural Mechanics (FEA)](#structural-mechanics-fea)
- [Fluid Dynamics (CFD)](#fluid-dynamics-cfd)
- [Heat Transfer and Multiphysics](#heat-transfer-and-multiphysics)
- [Machine Learning Fundamentals for Physical Data](#machine-learning-fundamentals-for-physical-data)
- [Physics-Informed Neural Networks (PINNs)](#physics-informed-neural-networks-pinns)
- [Neural Operators](#neural-operators)
- [Graph Neural Networks and Mesh-Based Learning](#graph-neural-networks-and-mesh-based-learning)
- [3D Geometry and Geometric Deep Learning](#3d-geometry-and-geometric-deep-learning)
- [Transformers for Physics and Large-Scale Surrogates](#transformers-for-physics-and-large-scale-surrogates)
- [Surrogate Models, Reduced-Order Models and Design of Experiments](#surrogate-models-reduced-order-models-and-design-of-experiments)
- [Time-Dependent Problems and Autoregressive Rollouts](#time-dependent-problems-and-autoregressive-rollouts)
- [Uncertainty, Generalization and Out-of-Distribution Detection](#uncertainty-generalization-and-out-of-distribution-detection)
- [Simulation Data, Datasets and Benchmarks](#simulation-data-datasets-and-benchmarks)
- [Evaluation and Testing](#evaluation-and-testing)
- [Design Optimization, Inverse Problems and Generative Design](#design-optimization-inverse-problems-and-generative-design)
- [Generative Models and Foundation Models for Physics](#generative-models-and-foundation-models-for-physics)
- [LLM Agents for Simulation and CAD](#llm-agents-for-simulation-and-cad)
- [ML Engineering, Scaling and Production](#ml-engineering-scaling-and-production)
- [Coding and Practical Implementation](#coding-and-practical-implementation)
- [AI-for-Simulation System Design](#ai-for-simulation-system-design)
- [Behavioral and Customer-Facing Scenarios](#behavioral-and-customer-facing-scenarios)
- [Know the Industry](#know-the-industry)

Also in this repository:

- [All answer pages](answers/README.md)
- [Learning resources](resources.md): courses, books, libraries, datasets and paper lists
- [Industry and interview notes](industry.md): companies, products and reported interview formats, with sources

### Prepared and maintained by Varad Pawar

- [LinkedIn](https://www.linkedin.com/in/varadkpawar)
- [GitHub](https://github.com/VdKPr)

---

> **Note:** New questions and answers will keep being added. Corrections and contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Must Know

- Surrogate model
- Physics-informed neural network (PINN)
- Neural operator (FNO, DeepONet)
- Graph neural network on a mesh (MeshGraphNets)
- Geometry encoding: meshes, point clouds, signed distance functions
- Autoregressive rollout
- Uncertainty and out-of-distribution detection
- Verification and validation

Two overviews to start with: [Physics Informed Machine Learning: High Level Overview of AI and ML in Science and Engineering](https://www.youtube.com/watch?v=JoFW2uSd3Uo) (Steve Brunton) and [ETH Zürich AISE: Course Introduction](https://www.youtube.com/watch?v=LkKvhvsf6jY) (Ben Moseley).

- What is a surrogate model, and why do engineering companies want one?
  - Answer: [Surrogate models](answers/surrogate-models.md)
- When should you *not* replace a numerical solver with machine learning?
  - Answer: [ML vs numerical solvers](answers/ml-vs-numerical-solvers.md) and [Weak baselines and reporting biases lead to overoptimism in machine learning for fluid-related PDEs](https://arxiv.org/abs/2407.07218)
- What is the difference between a PINN and a neural operator?
  - Answer: [PINNs](answers/pinns.md) and [Neural operators](answers/neural-operators.md)
- Why are graph neural networks a natural fit for simulation meshes?
  - Answer: [GNNs and MeshGraphNets](answers/gnns-and-meshgraphnets.md)
- Walk me through a CAE workflow from CAD to report. Where can machine learning help, and where can it hurt?
  - Answer: [Surrogate models](answers/surrogate-models.md) and [System design framework](answers/system-design-framework.md)
- What does "the model generalizes to new geometries" actually mean, and how would you test the claim?
  - Answer: [Out-of-distribution detection and generalization](answers/out-of-distribution-and-generalization.md) and [Data splits and leakage](answers/data-splits-and-leakage.md)
- How do you know whether a surrogate's prediction can be trusted for one specific new design?
  - Answer: [Uncertainty quantification](answers/uncertainty-quantification.md) and [Out-of-distribution detection and generalization](answers/out-of-distribution-and-generalization.md)
- A paper claims a 1000× speed-up over CFD. What questions would you ask?
  - Answer: [Evaluation metrics](answers/evaluation-metrics.md) and [ML vs numerical solvers](answers/ml-vs-numerical-solvers.md)
- What is the difference between verification and validation?
  - Answer: [Verification, validation and mesh convergence](answers/verification-validation-and-mesh-convergence.md)
- What is the difference between interpolation and extrapolation in a design space?
  - Answer: [Out-of-distribution detection and generalization](answers/out-of-distribution-and-generalization.md)

## Simulation Fundamentals: PDEs and Numerical Methods

- What does it mean to "solve a PDE" numerically?
  - Answer: Explained in this lecture: [ETH Zürich AISE: Introduction to Operator Learning Part 1](https://www.youtube.com/watch?v=yhHhMmiNl_g) (chapter "What does it mean to solve a PDE?") and [ETH Zürich AISE: Importance of PDEs in Science](https://www.youtube.com/watch?v=UiZxDRBd0Q8)
- Compare the finite element, finite volume and finite difference methods. Where is each used?
  - Answer: [FEM vs FVM vs FDM](answers/fem-vs-fvm-vs-fdm.md)
- What is the weak form of a PDE, and why does FEM use it?
  - Answer: [The finite element method](answers/finite-element-method.md) and [Weak formulation](https://en.wikipedia.org/wiki/Weak_formulation)
- What is the Galerkin method?
  - Answer: [The finite element method](answers/finite-element-method.md) and [Galerkin method](https://en.wikipedia.org/wiki/Galerkin_method)
- What are shape functions, and what is the difference between linear and quadratic elements?
  - Answer: [The finite element method](answers/finite-element-method.md)
- How is the global stiffness matrix assembled, and why is it sparse?
  - Answer: [The finite element method](answers/finite-element-method.md)
- What are Dirichlet, Neumann and Robin boundary conditions? Give a physical example of each.
  - Answer: [Boundary conditions](answers/boundary-conditions.md), [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_boundary_condition), [Neumann](https://en.wikipedia.org/wiki/Neumann_boundary_condition) and [Robin](https://en.wikipedia.org/wiki/Robin_boundary_condition)
- Why are Dirichlet conditions called "essential" and Neumann conditions "natural" in FEM?
  - Answer: [The finite element method](answers/finite-element-method.md)
- What is the CFL condition, and what happens if you violate it?
  - Answer: [Courant–Friedrichs–Lewy condition](https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition) and [FEM vs FVM vs FDM](answers/fem-vs-fvm-vs-fdm.md)
- Explicit vs implicit time integration: what are the trade-offs?
  - Answer: [Explicit and implicit methods](https://en.wikipedia.org/wiki/Explicit_and_implicit_methods) and [FEM vs FVM vs FDM](answers/fem-vs-fvm-vs-fdm.md)
- What is a stiff equation, and why does stiffness push you towards implicit methods?
  - Answer: [Stiff equation](https://en.wikipedia.org/wiki/Stiff_equation)
- How are large sparse linear systems solved? Compare direct solvers, conjugate gradient and multigrid.
  - Answer: [Conjugate gradient method](https://en.wikipedia.org/wiki/Conjugate_gradient_method), [Multigrid method](https://en.wikipedia.org/wiki/Multigrid_method) and [FEM vs FVM vs FDM](answers/fem-vs-fvm-vs-fdm.md)
- How does Newton–Raphson solve a nonlinear problem, and what do you do when it does not converge?
  - Answer: [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method) and [The finite element method](answers/finite-element-method.md)
- Can machine learning speed up a nonlinear solver *without* changing its answer?
  - Answer: [Differentiable physics and hybrid solvers](answers/differentiable-physics-and-hybrid-solvers.md) and [Jin et al., neural hybrid Newton solvers](https://arxiv.org/abs/2407.03945)
- What is a mesh convergence study? What is the grid convergence index?
  - Answer: [Verification, validation and mesh convergence](answers/verification-validation-and-mesh-convergence.md) and [Examining Spatial (Grid) Convergence (NASA)](https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html)
- What are the sources of error in a simulation result?
  - Answer: [Verification, validation and mesh convergence](answers/verification-validation-and-mesh-convergence.md)
- Structured vs unstructured meshes; tetrahedra vs hexahedra vs prisms. What is mesh quality?
  - Answer: [Mesh data processing](answers/mesh-data-processing.md) and [Gmsh reference manual](https://gmsh.info/doc/texinfo/gmsh.html)
- What is automatic differentiation, and how is it different from finite differences and symbolic differentiation?
  - Answer: [Automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) and [PINNs](answers/pinns.md)
- Hands-on: work through a PDE tutorial yourself.
  - Answer: [The FEniCSx tutorial](https://jsdokken.com/dolfinx-tutorial/) (FEM) and [CFD Python: 12 steps to Navier–Stokes](https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/) (finite differences)

## Structural Mechanics (FEA)

- Explain stress, strain, Young's modulus and Poisson's ratio. What is linear elasticity?
  - Answer: [Linear elasticity](https://en.wikipedia.org/wiki/Linear_elasticity)
- What is von Mises stress, and why is it used for ductile metals?
  - Answer: [Von Mises yield criterion](https://en.wikipedia.org/wiki/Von_Mises_yield_criterion)
- What is a stress concentration? How is it different from a stress singularity?
  - Answer: [Stress concentrations and singularities](answers/stress-concentrations-and-singularities.md) and [Stress concentration](https://en.wikipedia.org/wiki/Stress_concentration)
- Why does the peak stress at a sharp corner keep rising as you refine the mesh?
  - Answer: [Stress concentrations and singularities](answers/stress-concentrations-and-singularities.md)
- What is Saint-Venant's principle, and why does it matter when two models apply the same load differently?
  - Answer: [Saint-Venant's principle](https://en.wikipedia.org/wiki/Saint-Venant%27s_principle) and [Stress concentrations and singularities](answers/stress-concentrations-and-singularities.md)
- What makes an analysis nonlinear (material, geometry, contact)?
  - Answer: [The finite element method](answers/finite-element-method.md) and [Plasticity](https://en.wikipedia.org/wiki/Plasticity_(physics))
- Why does plasticity make the solution depend on the load history, and why does that complicate a surrogate?
  - Answer: [Plasticity](https://en.wikipedia.org/wiki/Plasticity_(physics)) and [Surrogate models](answers/surrogate-models.md)
- What is buckling, and why is a linear buckling analysis an eigenvalue problem?
  - Answer: [Buckling](https://en.wikipedia.org/wiki/Buckling)
- What is modal analysis, and what are natural frequencies and mode shapes used for?
  - Answer: [Modal analysis](https://en.wikipedia.org/wiki/Modal_analysis)
- What is fatigue, and why is fatigue life hard to predict with a surrogate?
  - Answer: [Fatigue (material)](https://en.wikipedia.org/wiki/Fatigue_(material)) and [Stress concentrations and singularities](answers/stress-concentrations-and-singularities.md)
- Linear tetrahedral elements are "too stiff" in bending. Why, and what do you do about it?
  - Answer: [The finite element method](answers/finite-element-method.md) (section on locking)
- How do you apply a bolt load or a remote load in FEA (rigid vs distributing couplings), and how does the choice affect nearby stresses?
  - Answer: [Boundary conditions](answers/boundary-conditions.md) and [SimJEB: Simulated Jet Engine Bracket Dataset](https://arxiv.org/abs/2105.03534)
- What is a safety factor? How would you predict one with a surrogate, and what goes wrong?
  - Answer: [Stress concentrations and singularities](answers/stress-concentrations-and-singularities.md) and [Uncertainty quantification](answers/uncertainty-quantification.md)
- How do you check that an FEA result is sane before you trust it?
  - Answer: [Verification, validation and mesh convergence](answers/verification-validation-and-mesh-convergence.md)
- Why are crash-simulation surrogates especially hard?
  - Answer: [High-Fidelity Industrial Crash Dynamics Prediction via Geometry-Aware Operator Learning](https://arxiv.org/abs/2605.27758) and [Autoregressive rollouts](answers/autoregressive-rollouts.md)
- Which public structural datasets could you train on?
  - Answer: [SimJEB](https://simjeb.github.io/) and [DeepJEB](https://arxiv.org/abs/2406.09047); see also [Learning resources](resources.md#datasets-and-benchmarks)

## Fluid Dynamics (CFD)

- Write down the incompressible Navier–Stokes equations and explain each term.
  - Answer: [Navier–Stokes equations](https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations) and [CFD essentials](answers/cfd-essentials.md)
- What is the Reynolds number, and what happens physically as it increases?
  - Answer: [Reynolds number](https://en.wikipedia.org/wiki/Reynolds_number) and [CFD essentials](answers/cfd-essentials.md)
- Laminar vs turbulent flow: why is turbulence so expensive to simulate?
  - Answer: [CFD essentials](answers/cfd-essentials.md) and [Turbulence modeling](https://en.wikipedia.org/wiki/Turbulence_modeling)
- RANS vs LES vs DNS: compare cost, accuracy and typical use.
  - Answer: [CFD essentials](answers/cfd-essentials.md), [RANS](https://en.wikipedia.org/wiki/Reynolds-averaged_Navier%E2%80%93Stokes_equations), [LES](https://en.wikipedia.org/wiki/Large_eddy_simulation) and [DNS](https://en.wikipedia.org/wiki/Direct_numerical_simulation)
- What does a turbulence model such as Spalart–Allmaras or k-ω SST do?
  - Answer: [CFD essentials](answers/cfd-essentials.md) and [NASA Turbulence Modeling Resource](https://turbmodels.larc.nasa.gov/)
- What is a boundary layer? What is y+, and why does it drive mesh design near walls?
  - Answer: [CFD essentials](answers/cfd-essentials.md), [Boundary layer](https://en.wikipedia.org/wiki/Boundary_layer), [Law of the wall](https://en.wikipedia.org/wiki/Law_of_the_wall) and [Dimensionless wall distance (y plus)](https://www.cfd-online.com/Wiki/Dimensionless_wall_distance_(y_plus))
- What are wall functions, and when are they acceptable?
  - Answer: [CFD essentials](answers/cfd-essentials.md)
- Incompressible vs compressible flow: when do you need a compressible solver?
  - Answer: [CFD essentials](answers/cfd-essentials.md)
- What is flow separation, and why do surrogates struggle near it?
  - Answer: [CFD essentials](answers/cfd-essentials.md) and [Out-of-distribution detection and generalization](answers/out-of-distribution-and-generalization.md)
- What are the drag and lift coefficients, and how do you compute them from surface pressure and wall shear stress?
  - Answer: [Drag coefficient](https://en.wikipedia.org/wiki/Drag_coefficient), [CFD essentials](answers/cfd-essentials.md) and [Mesh data processing](answers/mesh-data-processing.md)
- Steady vs transient simulation: when is a steady RANS answer misleading?
  - Answer: [CFD essentials](answers/cfd-essentials.md)
- Which boundary conditions does an external-aerodynamics simulation use?
  - Answer: [Boundary conditions](answers/boundary-conditions.md)
- How does an incompressible solver couple pressure and velocity?
  - Answer: [CFD essentials](answers/cfd-essentials.md)
- What does a typical external-aerodynamics dataset contain, and what is usually missing?
  - Answer: [DrivAerML: High-Fidelity CFD Dataset for Road-Car External Aerodynamics](https://arxiv.org/abs/2408.11969) and [AirfRANS](https://arxiv.org/abs/2212.07564)
- Why does a divergence-free constraint matter, and how can a model enforce it exactly?
  - Answer: [Hard vs soft constraints](answers/hard-vs-soft-constraints.md) and [AB-UPT](https://arxiv.org/abs/2502.09692)
- Hands-on: which open-source CFD solvers would you use to generate data?
  - Answer: [OpenFOAM](https://github.com/OpenFOAM/OpenFOAM-dev), [SU2](https://github.com/su2code/SU2) and [CFD Python: 12 steps to Navier–Stokes](https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/)

## Heat Transfer and Multiphysics

- Explain conduction, convection and radiation, and the boundary conditions used for each.
  - Answer: [Thermal conduction](https://en.wikipedia.org/wiki/Thermal_conduction) and [Boundary conditions](answers/boundary-conditions.md)
- What is conjugate heat transfer?
  - Answer: [CFD essentials](answers/cfd-essentials.md)
- What is thermo-mechanical coupling? Explain one-way vs two-way coupling.
  - Answer: [The finite element method](answers/finite-element-method.md) (section on multiphysics)
- Why does a mismatch in thermal expansion cause warpage in layered parts such as electronics packages?
  - Answer: [The finite element method](answers/finite-element-method.md) (section on multiphysics)
- Which dimensionless numbers matter in heat transfer (Prandtl, Nusselt, Biot), and why would you feed them to a model?
  - Answer: [Non-dimensionalization and normalization](answers/nondimensionalization-and-normalization.md)
- What is the difference between partitioned and monolithic coupling in fluid–structure interaction?
  - Answer: [The finite element method](answers/finite-element-method.md) and [preCICE (coupling library)](https://github.com/precice/precice)
- How would you build a surrogate for the thermal design of an electronics enclosure or a data-center room?
  - Answer: [System design framework](answers/system-design-framework.md) and [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) (includes data-center airflow and thermal recipes)

## Machine Learning Fundamentals for Physical Data

- How do you normalize physical fields and coordinates? Why per channel, and why only with training-set statistics?
  - Answer: [Non-dimensionalization and normalization](answers/nondimensionalization-and-normalization.md)
- Why non-dimensionalize inputs and outputs before training?
  - Answer: [Non-dimensionalization and normalization](answers/nondimensionalization-and-normalization.md) and [Buckingham π theorem](https://en.wikipedia.org/wiki/Buckingham_%CF%80_theorem)
- Your targets span several orders of magnitude across samples. MSE or relative L2 loss?
  - Answer: [Evaluation metrics](answers/evaluation-metrics.md) and [Non-dimensionalization and normalization](answers/nondimensionalization-and-normalization.md)
- How do you handle heavy-tailed targets such as peak stress or stagnation pressure?
  - Answer: [Non-dimensionalization and normalization](answers/nondimensionalization-and-normalization.md)
- How do you control overfitting when you only have 50–500 simulations?
  - Answer: [Surrogate models](answers/surrogate-models.md) and [Gaussian processes](answers/gaussian-processes.md)
- What is spectral bias, and why does it matter for physical fields?
  - Answer: [PINN training failures](answers/pinn-training-failures.md), [On the Spectral Bias of Neural Networks](https://arxiv.org/abs/1806.08734) and [Fourier Features Let Networks Learn High Frequency Functions](https://arxiv.org/abs/2006.10739)
- What are Fourier features or positional encodings for coordinates, and when do you need them?
  - Answer: [Transformers for physics](answers/transformers-for-physics.md) and [Fourier Features Let Networks Learn High Frequency Functions](https://arxiv.org/abs/2006.10739)
- How do you choose between a U-Net on a grid, a GNN on the mesh, a point-cloud network and a transformer?
  - Answer: [Geometry representations](answers/geometry-representations.md) and [GNNs and MeshGraphNets](answers/gnns-and-meshgraphnets.md)
- What are invariance and equivariance? Which symmetries matter in engineering problems?
  - Answer: [Equivariance and symmetry](answers/equivariance-and-symmetry.md)
- When is data augmentation (rotation, mirroring, scaling) physically valid, and when is it wrong?
  - Answer: [Equivariance and symmetry](answers/equivariance-and-symmetry.md)
- How do you do model selection when you have very few simulations?
  - Answer: [Data splits and leakage](answers/data-splits-and-leakage.md)
- What does the universal approximation theorem guarantee, and what does it not?
  - Answer: [Universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem) and [DeepONet](answers/deeponet.md)
- Explain backpropagation. Implement logistic regression from scratch.
  - Answer: [Machine Learning Interviews Book (Chip Huyen)](https://huyenchip.com/ml-interviews-book/), [Machine-Learning-Interviews](https://github.com/alirezadir/Machine-Learning-Interviews) and [PhysicsX ML Engineer interview guide (InterviewQuery)](https://www.interviewquery.com/interview-guides/physicsx-ml-engineer)
- Why might you switch from Adam to L-BFGS late in training a physics model?
  - Answer: [PINN training failures](answers/pinn-training-failures.md)
- Your model will not converge on a physical dataset. How do you debug it?
  - Answer: [Non-dimensionalization and normalization](answers/nondimensionalization-and-normalization.md) (debugging checklist) and [PINN training failures](answers/pinn-training-failures.md)

## Physics-Informed Neural Networks (PINNs)

- What is a PINN? Write down its loss function.
  - Answer: [PINNs](answers/pinns.md), [So, what is a physics-informed neural network? (Ben Moseley)](https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/) and [Physics Informed Deep Learning (Part I) (Raissi et al.)](https://arxiv.org/abs/1711.10561)
- How are the derivatives in the PDE residual computed?
  - Answer: [PINNs](answers/pinns.md) and [Automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)
- How do PINNs solve inverse problems, such as recovering a material parameter from sparse measurements?
  - Answer: [PINNs](answers/pinns.md), [Physics Informed Deep Learning (Part II)](https://arxiv.org/abs/1711.10566) and [DeepXDE documentation](https://deepxde.readthedocs.io/)
- What are collocation points, and how should you choose them?
  - Answer: [PINNs](answers/pinns.md)
- Why do PINNs often fail to train?
  - Answer: [PINN training failures](answers/pinn-training-failures.md) and [Characterizing possible failure modes in PINNs](https://arxiv.org/abs/2109.01050)
- What is loss imbalance, and how do adaptive loss weights work?
  - Answer: [PINN training failures](answers/pinn-training-failures.md), [Understanding and mitigating gradient pathologies in PINNs](https://arxiv.org/abs/2001.04536) and [When and why PINNs fail to train: an NTK perspective](https://arxiv.org/abs/2007.14527)
- What is the causality problem in time-dependent PINNs?
  - Answer: [Respecting causality is all you need for training PINNs](https://arxiv.org/abs/2203.07404)
- Soft vs hard enforcement of boundary conditions: trade-offs?
  - Answer: [Hard vs soft constraints](answers/hard-vs-soft-constraints.md) and [PINNs with hard constraints for inverse design](https://arxiv.org/abs/2102.04626)
- When does a PINN beat a classical solver, and when does it lose?
  - Answer: [PINNs](answers/pinns.md), [ML vs numerical solvers](answers/ml-vs-numerical-solvers.md) and [ETH Zürich AISE: PINNs – Limitations and Extensions Part 1](https://www.youtube.com/watch?v=S11QK8baGVI)
- How do PINNs cope with high-frequency or multi-scale solutions?
  - Answer: [PINN training failures](answers/pinn-training-failures.md) and [Finite Basis PINNs (FBPINNs)](https://arxiv.org/abs/2107.07871)
- What does a good PINN training recipe look like in practice?
  - Answer: [An Expert's Guide to Training PINNs](https://arxiv.org/abs/2308.08468) and its code, [jaxpi](https://github.com/PredictiveIntelligenceLab/jaxpi)
- Why does a PINN usually have to be retrained for each new geometry or boundary condition, and why does that matter commercially?
  - Answer: [PINNs](answers/pinns.md) and [Neural operators](answers/neural-operators.md)
- What is a physics-informed neural operator (PINO)?
  - Answer: [Physics-Informed Neural Operator for Learning PDEs](https://arxiv.org/abs/2111.03794)
- Where can I see PINNs taught properly from scratch?
  - Answer: [ETH Zürich AISE: PINNs – Introduction](https://www.youtube.com/watch?v=D-F7BYRhAkQ), [Physics Informed Neural Networks (PINNs) (Steve Brunton)](https://www.youtube.com/watch?v=-zrY7P2dVC4) and the review [Scientific Machine Learning through PINNs: Where we are and What's next](https://arxiv.org/abs/2201.05624)

## Neural Operators

- What is operator learning, and how is it different from ordinary supervised learning?
  - Answer: [Neural operators](answers/neural-operators.md), [ETH Zürich AISE: Introduction to Operator Learning Part 1](https://www.youtube.com/watch?v=yhHhMmiNl_g) and [Part 2](https://www.youtube.com/watch?v=lEUgPvDi5O8)
- What does "discretization invariance" mean? Does it hold in practice?
  - Answer: [Neural operators](answers/neural-operators.md) and [Neural Operator: Learning Maps Between Function Spaces (Kovachki et al.)](https://arxiv.org/abs/2108.08481)
- Explain the Fourier Neural Operator architecture.
  - Answer: [Fourier Neural Operator](answers/fourier-neural-operator.md), [FNO paper (Li et al.)](https://arxiv.org/abs/2010.08895), [Zongyi Li's FNO blog](https://zongyi-li.github.io/blog/2020/fourier-pde/) and [ETH Zürich AISE: Fourier Neural Operators](https://www.youtube.com/watch?v=b96wRdjH1Lg)
- Why does FNO keep only a limited number of Fourier modes?
  - Answer: [Fourier Neural Operator](answers/fourier-neural-operator.md)
- What are FNO's limitations (uniform grids, periodicity, complex geometry), and how do Geo-FNO and GINO address them?
  - Answer: [Fourier Neural Operator](answers/fourier-neural-operator.md), [Geo-FNO](https://arxiv.org/abs/2207.05209) and [GINO](https://arxiv.org/abs/2309.00583)
- Explain DeepONet: what do the branch and trunk networks do?
  - Answer: [DeepONet](answers/deeponet.md), [DeepONet paper (Lu et al.)](https://arxiv.org/abs/1910.03193) and [ETH Zürich AISE: Spectral Neural Operators and Deep Operator Networks](https://www.youtube.com/watch?v=BxklDO0TMlA)
- DeepONet vs FNO: when would you choose each?
  - Answer: [DeepONet](answers/deeponet.md) and [A comprehensive and fair comparison of two neural operators](https://arxiv.org/abs/2111.05512)
- What is a graph neural operator?
  - Answer: [Neural Operator: Graph Kernel Network](https://arxiv.org/abs/2003.03485) and [Neural operators](answers/neural-operators.md)
- What is a convolutional neural operator (CNO)?
  - Answer: [Convolutional Neural Operators](https://arxiv.org/abs/2302.01178) and [ETH Zürich AISE: Convolutional Neural Operators](https://www.youtube.com/watch?v=5XaLKR08TwI)
- How do neural operators handle time-dependent problems?
  - Answer: [Autoregressive rollouts](answers/autoregressive-rollouts.md) and [ETH Zürich AISE: Time-Dependent Neural Operators](https://www.youtube.com/watch?v=u1KFcAvjyCI)
- Is attention a neural operator?
  - Answer: [ETH Zürich AISE: Attention as a Neural Operator](https://www.youtube.com/watch?v=wJSgLRiU7D4), [Choose a Transformer: Fourier or Galerkin](https://arxiv.org/abs/2105.14995) and [Transformers for physics](answers/transformers-for-physics.md)
- How much training data does a neural operator need, and where does it come from?
  - Answer: [Neural operators](answers/neural-operators.md) and [Design of experiments and active learning](answers/design-of-experiments-and-active-learning.md)
- Which benchmark problems are standard for neural operators?
  - Answer: [FNO paper](https://arxiv.org/abs/2010.08895) (Burgers, Darcy flow, Navier–Stokes), [PDEBench](https://arxiv.org/abs/2210.07182) and [Neural-Solver-Library](https://github.com/thuml/Neural-Solver-Library)
- Where can I learn neural operators hands-on?
  - Answer: [From Theory to Application: A Practical Introduction to Neural Operators](https://arxiv.org/abs/2503.05598) and the [neuraloperator library](https://neuraloperator.github.io/dev/index.html)

## Graph Neural Networks and Mesh-Based Learning

- What is message passing? Write the update equations.
  - Answer: [GNNs and MeshGraphNets](answers/gnns-and-meshgraphnets.md), [A Gentle Introduction to Graph Neural Networks (Distill)](https://distill.pub/2021/gnn-intro/) and [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212)
- What is the graph-network framework (edge, node and global updates)?
  - Answer: [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261)
- Explain MeshGraphNets: encoder, processor, decoder; mesh edges vs world edges.
  - Answer: [GNNs and MeshGraphNets](answers/gnns-and-meshgraphnets.md), [Learning Mesh-Based Simulation with Graph Networks](https://arxiv.org/abs/2010.03409) and [DeepMind's MeshGraphNets code](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)
- Which features go on the nodes and edges of a physics GNN, and why use relative positions?
  - Answer: [GNNs and MeshGraphNets](answers/gnns-and-meshgraphnets.md) and [Equivariance and symmetry](answers/equivariance-and-symmetry.md)
- How does the Graph Network-based Simulator (GNS) model particle systems?
  - Answer: [Learning to Simulate Complex Physics with Graph Networks](https://arxiv.org/abs/2002.09405) and [its code](https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate)
- How many message-passing steps do you need, and how does that relate to how far information must travel across the mesh?
  - Answer: [GNNs and MeshGraphNets](answers/gnns-and-meshgraphnets.md)
- What are oversmoothing and oversquashing? How do multiscale GNNs help?
  - Answer: [Deeper Insights into GCNs](https://arxiv.org/abs/1801.07606), [On the Bottleneck of GNNs](https://arxiv.org/abs/2006.05205), [MultiScale MeshGraphNets](https://arxiv.org/abs/2210.00612) and [X-MeshGraphNet](https://arxiv.org/abs/2411.17164)
- How do you build a graph from a point cloud: k-nearest neighbors or a radius graph?
  - Answer: [Mesh data processing](answers/mesh-data-processing.md), [k-d tree](https://en.wikipedia.org/wiki/K-d_tree) and [PyG documentation](https://pytorch-geometric.readthedocs.io/)
- GCN vs GAT vs GraphSAGE vs GIN: what are the differences?
  - Answer: [GCN](https://arxiv.org/abs/1609.02907), [GAT](https://arxiv.org/abs/1710.10903), [GraphSAGE](https://arxiv.org/abs/1706.02216) and [GIN: How Powerful are GNNs?](https://arxiv.org/abs/1810.00826)
- How would you scale a GNN to a mesh with 10 million nodes?
  - Answer: [Scaling to large meshes](answers/scaling-to-large-meshes.md) and [X-MeshGraphNet](https://arxiv.org/abs/2411.17164)
- What did "Message Passing Neural PDE Solvers" add to earlier GNN simulators?
  - Answer: [Message Passing Neural PDE Solvers](https://arxiv.org/abs/2202.03376) and [Autoregressive rollouts](answers/autoregressive-rollouts.md)
- When do you need an equivariant GNN?
  - Answer: [E(n) Equivariant Graph Neural Networks](https://arxiv.org/abs/2102.09844) and [Equivariance and symmetry](answers/equivariance-and-symmetry.md)
- What is a Graph U-Net?
  - Answer: [Graph U-Nets](https://arxiv.org/abs/1905.05178)
- Which libraries would you use?
  - Answer: [PyG](https://github.com/pyg-team/pytorch_geometric), [DGL](https://github.com/dmlc/dgl) and [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo)

## 3D Geometry and Geometric Deep Learning

- What are the ways to represent 3D geometry for machine learning (mesh, point cloud, voxels, SDF, CAD parameters, B-rep)? Pros and cons of each?
  - Answer: [Geometry representations](answers/geometry-representations.md)
- What is a signed distance function, and why do so many surrogates take it as input?
  - Answer: [Signed distance function](https://en.wikipedia.org/wiki/Signed_distance_function), [Geometry representations](answers/geometry-representations.md) and [GINO](https://arxiv.org/abs/2309.00583)
- Explain PointNet. Why does max pooling make it permutation-invariant?
  - Answer: [Point-cloud learning](answers/point-cloud-learning.md) and [PointNet](https://arxiv.org/abs/1612.00593)
- What did PointNet++ add?
  - Answer: [Point-cloud learning](answers/point-cloud-learning.md) and [PointNet++](https://arxiv.org/abs/1706.02413)
- What is EdgeConv (DGCNN), and what is a Point Transformer?
  - Answer: [Dynamic Graph CNN](https://arxiv.org/abs/1801.07829) and [Point Transformer](https://arxiv.org/abs/2012.09164)
- How do you sample points from a surface mesh? What is farthest point sampling?
  - Answer: [Point-cloud learning](answers/point-cloud-learning.md) and [Mesh data processing](answers/mesh-data-processing.md)
- How do you batch geometries that have very different numbers of points?
  - Answer: [Point-cloud learning](answers/point-cloud-learning.md)
- How do you represent a CAD model for machine learning? Which public CAD datasets exist?
  - Answer: [Geometry representations](answers/geometry-representations.md), [ABC dataset](https://arxiv.org/abs/1812.06216), [Fusion 360 Gallery](https://arxiv.org/abs/2010.02392) and [DeepCAD](https://arxiv.org/abs/2105.09492)
- What is an implicit neural representation or neural field?
  - Answer: [DeepSDF](https://arxiv.org/abs/1901.05103), [SIREN](https://arxiv.org/abs/2006.09661) and [Geometry representations](answers/geometry-representations.md)
- Why can remeshing the same geometry change a model's prediction, and how do you test for it?
  - Answer: [Geometry representations](answers/geometry-representations.md) and [Out-of-distribution detection and generalization](answers/out-of-distribution-and-generalization.md)
- How would you predict surface pressure on a car from its STL file?
  - Answer: [System design framework](answers/system-design-framework.md), [Point-cloud learning](answers/point-cloud-learning.md) and [DrivAerNet++](https://arxiv.org/abs/2406.09624)
- What is geodesic convolution, and how was it used for shape optimization?
  - Answer: [Geodesic Convolutional Shape Optimization](https://arxiv.org/abs/1802.04016) (by the founders of Neural Concept)
- Should you align shapes to a canonical orientation before training?
  - Answer: [Equivariance and symmetry](answers/equivariance-and-symmetry.md)

## Transformers for Physics and Large-Scale Surrogates

- Why is vanilla self-attention hard to use on a mesh with a million points?
  - Answer: [Transformers for physics](answers/transformers-for-physics.md) and [FlashAttention](https://arxiv.org/abs/2205.14135)
- How does Transolver's Physics-Attention work?
  - Answer: [Transolver](https://arxiv.org/abs/2402.02366), [Transolver++](https://arxiv.org/abs/2502.02414) and [Transformers for physics](answers/transformers-for-physics.md)
- What are Universal Physics Transformers, and why encode a mesh into a fixed set of latent tokens?
  - Answer: [Universal Physics Transformers](https://arxiv.org/abs/2402.12365), [UPT code](https://github.com/ml-jku/UPT) and [Transformers for physics](answers/transformers-for-physics.md)
- What are AB-UPT's anchor tokens, and how does AB-UPT enforce a divergence-free field?
  - Answer: [AB-UPT](https://arxiv.org/abs/2502.09692), [AB-UPT for Automotive and Aerospace Applications](https://arxiv.org/abs/2510.15808) and [Hard vs soft constraints](answers/hard-vs-soft-constraints.md)
- What is DoMINO, and what is GeoTransolver?
  - Answer: [DoMINO](https://arxiv.org/abs/2501.13350), [GeoTransolver](https://arxiv.org/abs/2512.20399) and [PhysicsNeMo: Transformer models for external aerodynamics](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/cfd/external_aerodynamics/transformer_models/README.html)
- What is linear (Galerkin-type) attention, and how is it used in operator learning?
  - Answer: [Choose a Transformer: Fourier or Galerkin](https://arxiv.org/abs/2105.14995), [OFormer: Transformer for PDEs' Operator Learning](https://arxiv.org/abs/2205.13671) and [GNOT](https://arxiv.org/abs/2302.14376)
- How do you encode positions and boundary conditions as transformer inputs?
  - Answer: [Transformers for physics](answers/transformers-for-physics.md)
- Why do recent car-aerodynamics models use separate surface and volume branches?
  - Answer: [AB-UPT](https://arxiv.org/abs/2502.09692) and [Transformers for physics](answers/transformers-for-physics.md)
- Do scaling laws hold for neural surrogates? Is a bigger model always better?
  - Answer: [ETH Zürich AISE: Windowed Attention and Scaling Laws](https://www.youtube.com/watch?v=YtJhReM5bHY) and [Towards Foundation Models for Scientific Machine Learning: Characterizing Scaling and Transfer Behavior](https://arxiv.org/abs/2306.00258)
- When would a GNN beat a transformer on a mesh problem, and vice versa?
  - Answer: [Transformers for physics](answers/transformers-for-physics.md) and [GNNs and MeshGraphNets](answers/gnns-and-meshgraphnets.md)
- Which open code bases implement these models?
  - Answer: [Neural-Solver-Library](https://github.com/thuml/Neural-Solver-Library), [Transolver](https://github.com/thuml/Transolver), [UPT](https://github.com/ml-jku/UPT), [Noether (Emmi AI)](https://github.com/Emmi-AI/noether) and [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo)

## Surrogate Models, Reduced-Order Models and Design of Experiments

- Scalar surrogate (one number, such as drag) vs field surrogate (the whole pressure field): when do you need each?
  - Answer: [Surrogate models](answers/surrogate-models.md)
- When are classical surrogates (polynomial response surfaces, radial basis functions, Kriging) enough?
  - Answer: [Gaussian processes](answers/gaussian-processes.md), [Kriging](https://en.wikipedia.org/wiki/Kriging) and [Surrogate Modeling Toolbox (SMT)](https://github.com/SMTorg/smt)
- Explain Gaussian process regression. How does a GP estimate its own uncertainty?
  - Answer: [Gaussian processes](answers/gaussian-processes.md), [A Visual Exploration of Gaussian Processes (Distill)](https://distill.pub/2019/visual-exploration-gaussian-processes/) and [Gaussian Processes for Machine Learning (Rasmussen & Williams)](https://gaussianprocess.org/gpml/)
- Which kernel would you choose, and how are the hyperparameters fitted?
  - Answer: [Gaussian processes](answers/gaussian-processes.md)
- Why do GPs scale badly with the number of samples, and what are the workarounds?
  - Answer: [Gaussian processes](answers/gaussian-processes.md) and [GPyTorch](https://github.com/cornellius-gp/gpytorch)
- What is proper orthogonal decomposition (POD), and how do you compute it from snapshots?
  - Answer: [Reduced-order models](answers/reduced-order-models.md), [Proper orthogonal decomposition](https://en.wikipedia.org/wiki/Proper_orthogonal_decomposition) and [Modal Analysis of Fluid Flows: An Overview](https://arxiv.org/abs/1702.01453)
- Intrusive (Galerkin-projection) vs non-intrusive reduced-order models: what is the difference?
  - Answer: [Reduced-order models](answers/reduced-order-models.md)
- What is dynamic mode decomposition (DMD)?
  - Answer: [Dynamic mode decomposition](https://en.wikipedia.org/wiki/Dynamic_mode_decomposition) and [Reduced-order models](answers/reduced-order-models.md)
- When does a linear subspace fail, and what do autoencoder-based ROMs offer instead?
  - Answer: [Reduced-order models](answers/reduced-order-models.md) and [Model reduction on nonlinear manifolds using deep convolutional autoencoders](https://arxiv.org/abs/1812.08373)
- What is SINDy?
  - Answer: [Sparse identification of nonlinear dynamics (SINDy)](https://arxiv.org/abs/1509.03580) and [PySINDy](https://github.com/dynamicslab/pysindy)
- What is the Koopman operator, and how is deep learning used to approximate it?
  - Answer: [Deep learning for universal linear embeddings of nonlinear dynamics](https://arxiv.org/abs/1712.09707)
- Latin hypercube vs Sobol vs random sampling: which would you use to plan simulations?
  - Answer: [Design of experiments and active learning](answers/design-of-experiments-and-active-learning.md), [Latin hypercube sampling](https://en.wikipedia.org/wiki/Latin_hypercube_sampling) and [Sobol sequence](https://en.wikipedia.org/wiki/Sobol_sequence)
- How many simulations do you need to train a surrogate?
  - Answer: [Design of experiments and active learning](answers/design-of-experiments-and-active-learning.md) and [Surrogate models](answers/surrogate-models.md)
- What is active learning for simulations, and which acquisition function would you use?
  - Answer: [Design of experiments and active learning](answers/design-of-experiments-and-active-learning.md) and [A Tutorial on Bayesian Optimization](https://arxiv.org/abs/1807.02811)
- What is global sensitivity analysis (Sobol indices), and how does it guide simplifying a design space?
  - Answer: [Design of experiments and active learning](answers/design-of-experiments-and-active-learning.md)
- How do you combine many cheap low-fidelity simulations with a few expensive high-fidelity ones?
  - Answer: [Multi-fidelity learning](answers/multi-fidelity-learning.md) and [A composite neural network that learns from multi-fidelity data](https://arxiv.org/abs/1903.00104)
- What are Hamiltonian and Lagrangian neural networks?
  - Answer: [Hamiltonian Neural Networks](https://arxiv.org/abs/1906.01563), [Lagrangian Neural Networks](https://arxiv.org/abs/2003.04630) and [Hard vs soft constraints](answers/hard-vs-soft-constraints.md)
- A good book covering this whole area?
  - Answer: [Data-Driven Science and Engineering (Brunton & Kutz)](http://databookuw.com/)

## Time-Dependent Problems and Autoregressive Rollouts

- Why does a model trained for one-step prediction drift when you roll it out for many steps?
  - Answer: [Autoregressive rollouts](answers/autoregressive-rollouts.md)
- Why does injecting noise into training inputs help rollouts?
  - Answer: [Autoregressive rollouts](answers/autoregressive-rollouts.md), [Learning to Simulate Complex Physics with Graph Networks](https://arxiv.org/abs/2002.09405) and [MeshGraphNets](https://arxiv.org/abs/2010.03409)
- What is the pushforward trick?
  - Answer: [Message Passing Neural PDE Solvers](https://arxiv.org/abs/2202.03376) and [Autoregressive rollouts](answers/autoregressive-rollouts.md)
- What is unrolled training, and what is "solver-in-the-loop"?
  - Answer: [Solver-in-the-Loop](https://arxiv.org/abs/2007.00016) and [Differentiable physics and hybrid solvers](answers/differentiable-physics-and-hybrid-solvers.md)
- Why do long rollouts lose small-scale (high-frequency) detail, and how does PDE-Refiner address it?
  - Answer: [PDE-Refiner](https://arxiv.org/abs/2308.05732) and [Autoregressive rollouts](answers/autoregressive-rollouts.md)
- How do you evaluate a rollout: error vs time, correlation time, energy spectrum?
  - Answer: [Autoregressive rollouts](answers/autoregressive-rollouts.md) and [Evaluation metrics](answers/evaluation-metrics.md)
- What is time-bundling, and how do you choose the model's time step?
  - Answer: [Message Passing Neural PDE Solvers](https://arxiv.org/abs/2202.03376) and [Autoregressive rollouts](answers/autoregressive-rollouts.md)
- What are neural ODEs?
  - Answer: [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366), [Diffrax](https://github.com/patrick-kidger/diffrax) and [ETH Zürich AISE: Neural Differential Equations](https://www.youtube.com/watch?v=jnjYsm4NjhE)
- How can machine learning correct a coarse-grid solver instead of replacing it?
  - Answer: [Machine learning accelerated computational fluid dynamics](https://arxiv.org/abs/2102.01010), [Learned Coarse Models for Efficient Turbulence Simulation](https://arxiv.org/abs/2112.15275) and [Differentiable physics and hybrid solvers](answers/differentiable-physics-and-hybrid-solvers.md)
- For a chaotic flow, what can a surrogate realistically predict?
  - Answer: [Autoregressive rollouts](answers/autoregressive-rollouts.md)
- How do you detect that a rollout is about to blow up?
  - Answer: [Autoregressive rollouts](answers/autoregressive-rollouts.md)
- Which benchmarks test time-dependent neural PDE solvers?
  - Answer: [PDEArena: Towards Multi-spatiotemporal-scale Generalized PDE Modeling](https://arxiv.org/abs/2209.15616), [PDEBench](https://arxiv.org/abs/2210.07182) and [The Well](https://arxiv.org/abs/2412.00568)

## Uncertainty, Generalization and Out-of-Distribution Detection

- Aleatoric vs epistemic uncertainty: which dominates in a simulation surrogate?
  - Answer: [Uncertainty quantification](answers/uncertainty-quantification.md)
- How do deep ensembles estimate uncertainty?
  - Answer: [Deep Ensembles](https://arxiv.org/abs/1612.01474) and [Uncertainty quantification](answers/uncertainty-quantification.md)
- What is Monte Carlo dropout, and why is it often overconfident?
  - Answer: [Dropout as a Bayesian Approximation](https://arxiv.org/abs/1506.02142) and [Uncertainty quantification](answers/uncertainty-quantification.md)
- What is conformal prediction? What does its coverage guarantee assume?
  - Answer: [A Gentle Introduction to Conformal Prediction](https://arxiv.org/abs/2107.07511), [Conformal prediction](https://en.wikipedia.org/wiki/Conformal_prediction) and [MAPIE](https://github.com/scikit-learn-contrib/MAPIE)
- How do you check that uncertainty estimates are calibrated?
  - Answer: [Uncertainty quantification](answers/uncertainty-quantification.md)
- Which UQ methods are used in scientific machine learning, and how are they compared?
  - Answer: [Uncertainty Quantification in Scientific Machine Learning: Methods, Metrics, and Comparisons](https://arxiv.org/abs/2201.07766) and [A Review of Uncertainty Quantification in Deep Learning](https://arxiv.org/abs/2011.06225)
- What does "out of distribution" mean for a geometry, and how would you detect it?
  - Answer: [Out-of-distribution detection and generalization](answers/out-of-distribution-and-generalization.md), [A Baseline for Detecting Misclassified and Out-of-Distribution Examples](https://arxiv.org/abs/1610.02136) and [Mahalanobis-distance OOD detection](https://arxiv.org/abs/1807.03888)
- When should the system fall back to running the full solver?
  - Answer: [Out-of-distribution detection and generalization](answers/out-of-distribution-and-generalization.md) and [Production and deployment](answers/production-and-deployment.md)
- A model was trained on Reynolds numbers from 1e5 to 1e6. A customer asks for 5e6. What do you say?
  - Answer: [Out-of-distribution detection and generalization](answers/out-of-distribution-and-generalization.md)
- How do you handle uncertainty when the training data itself comes from high-fidelity simulations with their own errors?
  - Answer: [Uncertainty quantification](answers/uncertainty-quantification.md) and [Verification, validation and mesh convergence](answers/verification-validation-and-mesh-convergence.md)
- How does a surrogate's uncertainty carry through to a design decision, such as passing a safety factor?
  - Answer: [Uncertainty quantification](answers/uncertainty-quantification.md)
- A new design family arrives in production. How do you notice, and what do you do?
  - Answer: [Production and deployment](answers/production-and-deployment.md) and [Out-of-distribution detection and generalization](answers/out-of-distribution-and-generalization.md)

## Simulation Data, Datasets and Benchmarks

- How would you generate a training set of 1,000 CFD simulations? Walk through the pipeline.
  - Answer: [Design of experiments and active learning](answers/design-of-experiments-and-active-learning.md) and [System design framework](answers/system-design-framework.md)
- What makes simulation training data "physically meaningful and sufficient" for a use case?
  - Answer: [Verification, validation and mesh convergence](answers/verification-validation-and-mesh-convergence.md) and [Design of experiments and active learning](answers/design-of-experiments-and-active-learning.md)
- Which file formats will you meet (STL, OBJ, VTK/VTU, CGNS, HDF5, Zarr, solver input decks), and how do you handle terabytes of them?
  - Answer: [Mesh data processing](answers/mesh-data-processing.md) and [Scaling to large meshes](answers/scaling-to-large-meshes.md)
- How should you split simulation data into train, validation and test sets?
  - Answer: [Data splits and leakage](answers/data-splits-and-leakage.md)
- Give three ways information can leak from the test set into training in simulation datasets.
  - Answer: [Data splits and leakage](answers/data-splits-and-leakage.md)
- Name public datasets for car aerodynamics, airfoils and structures. What are their licences?
  - Answer: [Learning resources: datasets](resources.md#datasets-and-benchmarks), [DrivAerNet++](https://arxiv.org/abs/2406.09624), [DrivAerML](https://arxiv.org/abs/2408.11969), [AhmedML](https://arxiv.org/abs/2407.20801), [WindsorML](https://arxiv.org/abs/2407.19320), [AirfRANS](https://arxiv.org/abs/2212.07564), [SHIFT-SUV](https://huggingface.co/datasets/luminary-shift/SUV), [SHIFT-Wing](https://huggingface.co/datasets/luminary-shift/WING) and [SimJEB](https://simjeb.github.io/)
- What are PDEBench, The Well, PDEArena, CFDBench and APEBench used for?
  - Answer: [PDEBench](https://arxiv.org/abs/2210.07182), [The Well](https://arxiv.org/abs/2412.00568), [PDEArena](https://arxiv.org/abs/2209.15616), [CFDBench](https://arxiv.org/abs/2310.05963) and [APEBench](https://github.com/tum-pbs/apebench)
- What did McGreivy & Hakim find about baselines in machine learning for fluid PDEs?
  - Answer: [Weak baselines and reporting biases lead to overoptimism in ML for fluid-related PDEs](https://arxiv.org/abs/2407.07218) (79% of the papers they reviewed that claimed to beat a standard numerical method compared against a weak baseline) and [ML vs numerical solvers](answers/ml-vs-numerical-solvers.md)
- The "ground truth" labels have discretization error. How should that change your evaluation?
  - Answer: [Verification, validation and mesh convergence](answers/verification-validation-and-mesh-convergence.md)
- The customer has 40 proprietary simulations. Can you still deliver a useful model?
  - Answer: [Foundation models for physics](answers/foundation-models-for-physics.md), [Multi-fidelity learning](answers/multi-fidelity-learning.md) and [Gaussian processes](answers/gaussian-processes.md)
- How do you version simulation data together with models?
  - Answer: [Production and deployment](answers/production-and-deployment.md)

## Evaluation and Testing

- Which metrics would you use for field prediction: relative L2, MSE, maximum error?
  - Answer: [Evaluation metrics](answers/evaluation-metrics.md)
- How do you evaluate integrated quantities such as drag, lift, maximum stress or mass flow?
  - Answer: [Evaluation metrics](answers/evaluation-metrics.md)
- How do you report a speed-up fairly?
  - Answer: [Evaluation metrics](answers/evaluation-metrics.md) and [ML vs numerical solvers](answers/ml-vs-numerical-solvers.md)
- Which baselines must a neural surrogate beat before anyone should care?
  - Answer: [Evaluation metrics](answers/evaluation-metrics.md) and [Weak baselines and reporting biases](https://arxiv.org/abs/2407.07218)
- What physics-consistency checks would you run on predictions (conservation, divergence, equilibrium, symmetry)?
  - Answer: [Evaluation metrics](answers/evaluation-metrics.md) and [Hard vs soft constraints](answers/hard-vs-soft-constraints.md)
- How many test cases are enough? How do you report variability across seeds and splits?
  - Answer: [Evaluation metrics](answers/evaluation-metrics.md)
- The customer will use the surrogate to rank designs, not for absolute values. How does that change evaluation?
  - Answer: [Evaluation metrics](answers/evaluation-metrics.md)
- How do you evaluate generalization to unseen geometries?
  - Answer: [Data splits and leakage](answers/data-splits-and-leakage.md) and [Out-of-distribution detection and generalization](answers/out-of-distribution-and-generalization.md)
- Your model's average error is low, but it fails on the designs the customer cares about. What do you do?
  - Answer: [Evaluation metrics](answers/evaluation-metrics.md) (slice analysis)
- How would you build a regression test suite for a surrogate?
  - Answer: [Production and deployment](answers/production-and-deployment.md)
- Which engineering-specific metrics did recent car-aerodynamics papers report?
  - Answer: [DoMINO](https://arxiv.org/abs/2501.13350) and [AB-UPT](https://arxiv.org/abs/2502.09692)
- How would you evaluate an LLM agent that sets up simulations?
  - Answer: [LLM agents for simulation](answers/llm-agents-for-simulation.md) and [FEABench](https://arxiv.org/abs/2504.06260)

## Design Optimization, Inverse Problems and Generative Design

- Walk through a surrogate-based optimization loop.
  - Answer: [Surrogate-based optimization](answers/surrogate-based-optimization.md)
- Gradients through a differentiable surrogate vs the adjoint method: compare.
  - Answer: [Surrogate-based optimization](answers/surrogate-based-optimization.md) and [The adjoint method](answers/adjoint-method.md)
- Explain the adjoint method. Why does its cost not grow with the number of design variables?
  - Answer: [The adjoint method](answers/adjoint-method.md) and [Adjoint state method](https://en.wikipedia.org/wiki/Adjoint_state_method)
- Explain Bayesian optimization and acquisition functions.
  - Answer: [A Tutorial on Bayesian Optimization](https://arxiv.org/abs/1807.02811), [Bayesian optimization](https://en.wikipedia.org/wiki/Bayesian_optimization) and [BoTorch](https://github.com/pytorch/botorch)
- Why can an optimizer "exploit" a surrogate, and how do you prevent it?
  - Answer: [Surrogate-based optimization](answers/surrogate-based-optimization.md)
- Explain topology optimization with SIMP.
  - Answer: [Topology optimization](answers/topology-optimization.md), [Topology optimization (Wikipedia)](https://en.wikipedia.org/wiki/Topology_optimization) and [Efficient topology optimization in MATLAB using 88 lines of code](https://www.topopt.mek.dtu.dk/apps-and-software/efficient-topology-optimization-in-matlab)
- How has machine learning been used for topology optimization?
  - Answer: [Neural networks for topology optimization](https://arxiv.org/abs/1709.09578) and [Neural reparameterization improves structural optimization](https://arxiv.org/abs/1909.04240)
- How do you identify unknown parameters from measurements (an inverse problem), and why does it need regularization?
  - Answer: [PINNs](answers/pinns.md) and [Physics Informed Deep Learning (Part II)](https://arxiv.org/abs/1711.10566)
- How would you build an inverse-design tool: target performance in, geometry out?
  - Answer: [Generative models for physics](answers/generative-models-for-physics.md) and [Surrogate-based optimization](answers/surrogate-based-optimization.md)
- How do you handle multiple objectives (for example drag vs downforce vs mass)?
  - Answer: [Surrogate-based optimization](answers/surrogate-based-optimization.md)
- How do you keep optimized or generated designs manufacturable?
  - Answer: [Surrogate-based optimization](answers/surrogate-based-optimization.md) and [Topology optimization](answers/topology-optimization.md)
- What are differentiable simulators, and when are they useful?
  - Answer: [Differentiable physics and hybrid solvers](answers/differentiable-physics-and-hybrid-solvers.md), [JAX-FEM](https://github.com/deepmodeling/jax-fem), [PhiFlow](https://github.com/tum-pbs/PhiFlow) and [JAX-CFD](https://github.com/google/jax-cfd)

## Generative Models and Foundation Models for Physics

- How do denoising diffusion models work, and why use them for physics?
  - Answer: [Generative models for physics](answers/generative-models-for-physics.md), [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) and [ETH Zürich AISE: Introduction to Diffusion Models](https://www.youtube.com/watch?v=Tohlijxz3XQ)
- What is flow matching?
  - Answer: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- How are diffusion models used for probabilistic forecasting?
  - Answer: [GenCast](https://arxiv.org/abs/2312.15796)
- How can a generative model solve a PDE from partial observations?
  - Answer: [DiffusionPDE](https://arxiv.org/abs/2406.17763)
- What is a foundation model for PDEs, and what has been shown to transfer?
  - Answer: [Foundation models for physics](answers/foundation-models-for-physics.md), [Multiple Physics Pretraining](https://arxiv.org/abs/2310.02994), [DPOT](https://arxiv.org/abs/2403.03542) and [Poseidon](https://arxiv.org/abs/2405.19101)
- What are the "large geometry" or "large engineering" models that companies talk about?
  - Answer: [Foundation models for physics](answers/foundation-models-for-physics.md), [PhysicsX LGM-Aero](https://www.physicsx.ai/newsroom/introducing-lgm-aero-genai-for-aero-engineering-and-airplane-showcase-application-for-aerostructures), [Luminary SHIFT models](https://luminary.ai/resources/introducing-luminary-shift-models-a-suite-of-physics-ai-foundation-models-to-transform-engineering-design/) and [Emmi AI](https://www.emmi.ai/)
- What does machine-learning weather forecasting tell us about AI for simulation?
  - Answer: [FourCastNet](https://arxiv.org/abs/2202.11214), [GraphCast](https://arxiv.org/abs/2212.12794), [Aurora](https://arxiv.org/abs/2405.13063) and [Foundation models for physics](answers/foundation-models-for-physics.md)
- How would you fine-tune a pretrained physics model on a customer's small dataset?
  - Answer: [Foundation models for physics](answers/foundation-models-for-physics.md)
- "Data is the moat in physics AI." Do you agree?
  - Answer: [Foundation models for physics](answers/foundation-models-for-physics.md)

## LLM Agents for Simulation and CAD

- What can LLM agents do in a CAE workflow today, and where do they fail?
  - Answer: [LLM agents for simulation](answers/llm-agents-for-simulation.md)
- What did FEABench measure, and what did it find?
  - Answer: [FEABench: Evaluating Language Models on Multiphysics Reasoning Ability](https://arxiv.org/abs/2504.06260)
- How do Foam-Agent, MetaOpenFOAM and OpenFOAMGPT automate CFD?
  - Answer: [Foam-Agent](https://arxiv.org/abs/2505.04997), [MetaOpenFOAM](https://arxiv.org/abs/2407.21320) and [OpenFOAMGPT](https://arxiv.org/abs/2501.06327)
- How do agents automate FEA set-up?
  - Answer: [MooseAgent](https://arxiv.org/abs/2504.08621), [FeaGPT](https://arxiv.org/abs/2510.21993), [CAX-Agent](https://arxiv.org/abs/2605.15218) and [A Constrained Natural-Language Interface for FEniCS](https://arxiv.org/abs/2606.10928)
- Why should the LLM not write solver physics directly?
  - Answer: [LLM agents for simulation](answers/llm-agents-for-simulation.md)
- How do text-to-CAD systems work?
  - Answer: [Text2CAD](https://arxiv.org/abs/2409.17106), [Text-to-CadQuery](https://arxiv.org/abs/2505.06507) and [Generating CAD Code with Vision-Language Models](https://arxiv.org/abs/2410.05340)
- What is the ReAct pattern, and why does it suit tool-driven engineering workflows?
  - Answer: [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) and [LLM agents for simulation](answers/llm-agents-for-simulation.md)
- How would you evaluate an agent that sets up and runs simulations?
  - Answer: [LLM agents for simulation](answers/llm-agents-for-simulation.md)

## ML Engineering, Scaling and Production

- Roughly how much GPU memory does a GNN on a 1-million-node mesh need? Show the arithmetic.
  - Answer: [Scaling to large meshes](answers/scaling-to-large-meshes.md)
- How can you train on subsampled points but predict at full resolution?
  - Answer: [Scaling to large meshes](answers/scaling-to-large-meshes.md) and [AB-UPT](https://arxiv.org/abs/2502.09692)
- What are domain decomposition and graph partitioning with halo regions?
  - Answer: [Scaling to large meshes](answers/scaling-to-large-meshes.md) and [X-MeshGraphNet](https://arxiv.org/abs/2411.17164)
- When does mixed precision hurt a physics model?
  - Answer: [Mixed Precision Training](https://arxiv.org/abs/1710.03740) and [Scaling to large meshes](answers/scaling-to-large-meshes.md)
- What is gradient checkpointing?
  - Answer: [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)
- Data parallel vs model parallel training: when do you need each?
  - Answer: [Scaling to large meshes](answers/scaling-to-large-meshes.md) and [ZeRO](https://arxiv.org/abs/1910.02054)
- How do you feed terabytes of simulation results to the GPUs without stalling?
  - Answer: [Scaling to large meshes](answers/scaling-to-large-meshes.md)
- When would you use Spark or Dask for simulation data?
  - Answer: [Scaling to large meshes](answers/scaling-to-large-meshes.md)
- How would you speed up a Python function that processes a 3D point cloud with millions of points?
  - Answer: [Mesh data processing](answers/mesh-data-processing.md)
- ONNX vs TorchScript (or `torch.export`) for deploying a model: trade-offs?
  - Answer: [Production and deployment](answers/production-and-deployment.md)
- How do Docker and Kubernetes fit into training and serving surrogates?
  - Answer: [Production and deployment](answers/production-and-deployment.md)
- How do you track experiments and manage model versions?
  - Answer: [Production and deployment](answers/production-and-deployment.md)
- How do you integrate a surrogate into an engineer's existing CAE workflow?
  - Answer: [Production and deployment](answers/production-and-deployment.md) and [PhysicsNeMo-CFD](https://github.com/NVIDIA/physicsnemo-cfd)
- How do you monitor a deployed surrogate and decide when to retrain?
  - Answer: [Production and deployment](answers/production-and-deployment.md)
- Customers will not upload their CAD to your cloud. What are your options?
  - Answer: [Production and deployment](answers/production-and-deployment.md)

## Coding and Practical Implementation

Write these in NumPy or PyTorch without looking anything up. The first two are reported from real interviews (see [industry.md](industry.md)).

- Convert flat vertex and face arrays into a mesh data structure. *(reported in a PhysicsX coding round)*
  - Answer: [Mesh data processing](answers/mesh-data-processing.md)
- Given a mesh dataset, identify the element types and compute the bounding box. *(reported in a PhysicsX live-coding round)*
  - Answer: [Mesh data processing](answers/mesh-data-processing.md)
- Compute triangle areas, face normals and area-weighted vertex normals, fully vectorized.
  - Answer: [Mesh data processing](answers/mesh-data-processing.md)
- Extract the unique edges and the boundary edges of a triangle mesh.
  - Answer: [Mesh data processing](answers/mesh-data-processing.md)
- Build vertex adjacency in sparse (CSR) form from a face array.
  - Answer: [Mesh data processing](answers/mesh-data-processing.md)
- Compute the volume of a closed triangle mesh.
  - Answer: [Mesh data processing](answers/mesh-data-processing.md) and [Divergence theorem](https://en.wikipedia.org/wiki/Divergence_theorem)
- Compute the drag force on a body from surface pressure and wall shear stress.
  - Answer: [Mesh data processing](answers/mesh-data-processing.md) and [CFD essentials](answers/cfd-essentials.md)
- Interpolate a field from one mesh onto another (nearest neighbour, inverse distance, barycentric).
  - Answer: [Mesh data processing](answers/mesh-data-processing.md)
- Build a k-nearest-neighbour or radius graph with a k-d tree.
  - Answer: [Mesh data processing](answers/mesh-data-processing.md) and [k-d tree](https://en.wikipedia.org/wiki/K-d_tree)
- Implement farthest point sampling.
  - Answer: [Point-cloud learning](answers/point-cloud-learning.md)
- Implement the per-sample relative L2 error, including the area-weighted version for unstructured meshes.
  - Answer: [Evaluation metrics](answers/evaluation-metrics.md)
- Implement per-channel normalization that uses training-set statistics only.
  - Answer: [Non-dimensionalization and normalization](answers/nondimensionalization-and-normalization.md)
- Implement a 1D finite element solver for the Poisson equation (or an axially loaded bar).
  - Answer: [The finite element method](answers/finite-element-method.md)
- Implement an explicit finite-difference solver for the 1D heat equation and check the stability limit.
  - Answer: [CFD Python: 12 steps to Navier–Stokes](https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/) and [FEM vs FVM vs FDM](answers/fem-vs-fvm-vs-fdm.md)
- Implement a PINN for the 1D Poisson or Burgers equation with PyTorch autograd.
  - Answer: [PINNs](answers/pinns.md) and the ETH course tutorials ([camlab-ethz/AI_Science_Engineering](https://github.com/camlab-ethz/AI_Science_Engineering))
- Implement a 1D Fourier layer: `rfft`, keep the first k modes, multiply by complex weights, `irfft`.
  - Answer: [Fourier Neural Operator](answers/fourier-neural-operator.md) and [the original FNO code](https://github.com/zongyi-li/fourier_neural_operator)
- Implement one message-passing layer: edge MLP, sum aggregation, node MLP, residual connection.
  - Answer: [GNNs and MeshGraphNets](answers/gnns-and-meshgraphnets.md)
- Implement PointNet: a shared per-point MLP followed by max pooling.
  - Answer: [Point-cloud learning](answers/point-cloud-learning.md)
- Implement an autoregressive rollout loop with training-time noise injection.
  - Answer: [Autoregressive rollouts](answers/autoregressive-rollouts.md)
- Implement Gaussian process regression from scratch (RBF kernel, Cholesky solve, predictive variance).
  - Answer: [Gaussian processes](answers/gaussian-processes.md)
- Compute POD modes from a snapshot matrix with the SVD.
  - Answer: [Reduced-order models](answers/reduced-order-models.md)
- Implement Latin hypercube sampling.
  - Answer: [Design of experiments and active learning](answers/design-of-experiments-and-active-learning.md)
- Implement split conformal prediction intervals for a regression model.
  - Answer: [Uncertainty quantification](answers/uncertainty-quantification.md)
- General algorithm questions also appear (three problems on Coderbyte or CoderPad are reported). Practise standard data-structure problems.
  - Answer: [Machine-Learning-Interviews](https://github.com/alirezadir/Machine-Learning-Interviews) and [PhysicsX interview guide (Dataford)](https://dataford.io/interview-guides/physicsx/machine-learning-engineer)
- Read a VTK or STL file, attach a field, and save it for ParaView.
  - Answer: [meshio](https://github.com/nschloe/meshio), [PyVista](https://github.com/pyvista/pyvista) and [Mesh data processing](answers/mesh-data-processing.md)

## AI-for-Simulation System Design

Answer every question with the same structure: [System design framework](answers/system-design-framework.md).

- Design a system that predicts drag and surface pressure for a new car shape in seconds, from its STL.
  - Answer: [System design framework](answers/system-design-framework.md) (worked example), [DrivAerML](https://arxiv.org/abs/2408.11969), [AB-UPT](https://arxiv.org/abs/2502.09692) and [DoMINO](https://arxiv.org/abs/2501.13350)
- Design a platform where engineers upload CAD and get stress predictions with a confidence level.
  - Answer: [System design framework](answers/system-design-framework.md), [Uncertainty quantification](answers/uncertainty-quantification.md) and [Out-of-distribution detection and generalization](answers/out-of-distribution-and-generalization.md)
- Design the pipeline that generates and stores 10,000 CFD simulations for training.
  - Answer: [Design of experiments and active learning](answers/design-of-experiments-and-active-learning.md) and [Scaling to large meshes](answers/scaling-to-large-meshes.md)
- Design an active learning loop that decides which simulation to run next.
  - Answer: [Design of experiments and active learning](answers/design-of-experiments-and-active-learning.md) and [A Tutorial on Bayesian Optimization](https://arxiv.org/abs/1807.02811)
- Design a lightweighting loop for a structural bracket that never ships an unverified design.
  - Answer: [Surrogate-based optimization](answers/surrogate-based-optimization.md) and [SimJEB](https://simjeb.github.io/)
- Design a validation strategy that would convince a sceptical customer to trust your surrogate.
  - Answer: [Verification, validation and mesh convergence](answers/verification-validation-and-mesh-convergence.md) and [Evaluation metrics](answers/evaluation-metrics.md)
- Design a real-time digital twin of a heat exchanger that combines sensor data with simulation.
  - Answer: [System design framework](answers/system-design-framework.md) and [Multi-fidelity learning](answers/multi-fidelity-learning.md)
- Design a crash-simulation surrogate for a vehicle programme.
  - Answer: [System design framework](answers/system-design-framework.md), [Autoregressive rollouts](answers/autoregressive-rollouts.md) and [High-Fidelity Industrial Crash Dynamics Prediction](https://arxiv.org/abs/2605.27758)
- Design a thermal surrogate for a battery pack or an electronics enclosure.
  - Answer: [System design framework](answers/system-design-framework.md)
- Design a surrogate that combines plenty of RANS data with a little LES data.
  - Answer: [Multi-fidelity learning](answers/multi-fidelity-learning.md)
- Design a pipeline that fine-tunes a pretrained physics model on a customer's 50 simulations.
  - Answer: [Foundation models for physics](answers/foundation-models-for-physics.md)
- Design an inverse-design tool: a target pressure distribution in, a geometry out.
  - Answer: [Generative models for physics](answers/generative-models-for-physics.md) and [Surrogate-based optimization](answers/surrogate-based-optimization.md)
- Design an LLM agent that runs a structural check from a plain-language request.
  - Answer: [LLM agents for simulation](answers/llm-agents-for-simulation.md)
- Design a serving system for 1,000 engineers using a surrogate interactively.
  - Answer: [Production and deployment](answers/production-and-deployment.md)
- Design monitoring that detects when incoming designs drift outside the training data.
  - Answer: [Out-of-distribution detection and generalization](answers/out-of-distribution-and-generalization.md) and [Production and deployment](answers/production-and-deployment.md)

## Behavioral and Customer-Facing Scenarios

All of these are covered in [Behavioral and customer scenarios](answers/behavioral-and-customer-scenarios.md).

- A customer's simulation engineer says: "I don't trust black boxes." How do you respond?
- Your surrogate disagrees with wind-tunnel data. What do you do?
- A customer wants 1% accuracy from 30 simulations. How do you handle the conversation?
- One week before delivery, you find the model fails on a design family the customer cares about. What do you do?
- Explain a neural operator to a non-ML engineering manager in two minutes.
- Tell me about a time you turned research into something people actually used.
- Tell me about a time you found out your own results were wrong.
- How do you choose between a quick proof of concept and a robust product feature?
- How do you scope a project with a new customer in the first meeting?
- You disagree with a domain expert about the physics. How do you resolve it?
- When would you tell a customer that machine learning is the wrong tool?
- Present a technical project you have worked on. *(reported as a final-round format)*
- Why this company, and why AI for simulation?
- How do you keep up with a field that moves this fast?

## Know the Industry

- What do PhysicsX, Neural Concept, Luminary, Emmi AI, Monolith, Navier AI and BeyondMath do, and how do they differ?
  - Answer: [Industry and interview notes](industry.md)
- What is Ansys SimAI?
  - Answer: [Ansys SimAI](https://www.ansys.com/products/simai), [PySimAI documentation](https://simai.docs.pyansys.com) and [Explaining SimAI](https://ansys.synopsys.com/blog/explaining-simai)
- What is NVIDIA PhysicsNeMo?
  - Answer: [NVIDIA PhysicsNeMo](https://developer.nvidia.com/physicsnemo) and [its documentation](https://docs.nvidia.com/physicsnemo/latest/)
- What changed when Synopsys acquired Ansys and Siemens acquired Altair?
  - Answer: [Synopsys Completes Acquisition of Ansys (17 July 2025)](https://news.synopsys.com/2025-07-17-Synopsys-Completes-Acquisition-of-Ansys), [Siemens Completes Acquisition of Altair (March 2025)](https://www.digitalengineering247.com/article/siemens-completes-acquisition-of-altair) and [Industry and interview notes](industry.md)
- What did PhysicsX release with LGM-Aero and Ai.rplane?
  - Answer: [Introducing LGM-Aero and Ai.rplane](https://www.physicsx.ai/newsroom/introducing-lgm-aero-genai-for-aero-engineering-and-airplane-showcase-application-for-aerostructures)
- What are Luminary's SHIFT models and datasets?
  - Answer: [Introducing Luminary SHIFT Models](https://luminary.ai/resources/introducing-luminary-shift-models-a-suite-of-physics-ai-foundation-models-to-transform-engineering-design/), [SHIFT-Wing](https://luminary.ai/resources/luminary-cloud-releases-shift-wing-a-physics-ai-model-for-transonic-wing-design-in-collaboration-with-otto-aviation-leveraging-nvidia-technologies/) and [Industry and interview notes](industry.md)
- How do physics-AI companies make money: platform licences, services or forward-deployed engineering?
  - Answer: [Industry and interview notes](industry.md)
- Which industries buy these tools, and why do some customers hesitate?
  - Answer: [Industry and interview notes](industry.md) and [Behavioral and customer scenarios](answers/behavioral-and-customer-scenarios.md)

---

## How This Guide Was Built

- **Questions** come from job descriptions at AI-for-simulation companies, candidate interview reports, and the standard papers and courses of the field. Reported interview items are marked as such, with sources in [industry.md](industry.md).
- **Answers** link to the primary source wherever one exists. The pages in [`answers/`](answers/README.md) were written for this guide; each one ends with its references.
- **Links** were checked when they were added. If one breaks, please open an issue.
- Company facts (funding, products, acquisitions) are dated. Check them again before an interview.

Format inspired by [AI Engineering Interview Questions and Answers](https://github.com/amitshekhariitbhu/ai-engineering-interview-questions).
