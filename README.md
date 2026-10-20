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

The questions come from three places: what AI-for-simulation companies ask for in their job descriptions, what candidates report from their interviews, and the core papers and courses of the field. Every answer links to a section of a page in [`answers/`](answers/README.md), written for this guide: a short answer you could give in the first thirty seconds, then the details, tested code where it helps, a model interview answer with likely follow-ups, and references to the primary sources (papers, documentation, lectures).

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
  - Answer: [Surrogate Models](answers/surrogate-models.md)
- When should you *not* replace a numerical solver with machine learning?
  - Answer: [When not to replace a solver with ML](answers/ml-vs-numerical-solvers.md#when-not-to-replace-a-solver-with-ml)
- What is the difference between a PINN and a neural operator?
  - Answer: [PINN vs neural operator](answers/neural-operators.md#pinn-vs-neural-operator)
- Why are graph neural networks a natural fit for simulation meshes?
  - Answer: [Why GNNs fit simulation meshes](answers/gnns-and-meshgraphnets.md#why-gnns-fit-simulation-meshes)
- Walk me through a CAE workflow from CAD to report. Where can machine learning help, and where can it hurt?
  - Answer: [The CAE Workflow, and Where ML Fits](answers/cae-workflow.md)
- What does "the model generalizes to new geometries" actually mean, and how would you test the claim?
  - Answer: [What generalizing to new geometries means](answers/out-of-distribution-and-generalization.md#what-generalizing-to-new-geometries-means)
- How do you know whether a surrogate's prediction can be trusted for one specific new design?
  - Answer: [Trusting one prediction for one new design](answers/out-of-distribution-and-generalization.md#trusting-one-prediction-for-one-new-design)
- A paper claims a 1000× speed-up over CFD. What questions would you ask?
  - Answer: [Questions to ask about a 1000x speed-up claim](answers/ml-vs-numerical-solvers.md#questions-to-ask-about-a-1000x-speed-up-claim)
- What is the difference between verification and validation?
  - Answer: [Verification vs validation](answers/verification-validation-and-mesh-convergence.md#verification-vs-validation)
- What is the difference between interpolation and extrapolation in a design space?
  - Answer: [Interpolation vs extrapolation](answers/out-of-distribution-and-generalization.md#interpolation-vs-extrapolation)

## Simulation Fundamentals: PDEs and Numerical Methods

- What does it mean to "solve a PDE" numerically?
  - Answer: [Solving a PDE Numerically](answers/solving-pdes-numerically.md)
- Compare the finite element, finite volume and finite difference methods. Where is each used?
  - Answer: [FEM vs FVM vs FDM](answers/fem-vs-fvm-vs-fdm.md)
- What is the weak form of a PDE, and why does FEM use it?
  - Answer: [The weak form](answers/weak-form-and-galerkin.md#the-weak-form)
- What is the Galerkin method?
  - Answer: [The Galerkin method](answers/weak-form-and-galerkin.md#the-galerkin-method)
- What are shape functions, and what is the difference between linear and quadratic elements?
  - Answer: [Shape functions](answers/elements-and-shape-functions.md#shape-functions)
- How is the global stiffness matrix assembled, and why is it sparse?
  - Answer: [Assembly and sparsity](answers/finite-element-method.md#assembly-and-sparsity)
- What are Dirichlet, Neumann and Robin boundary conditions? Give a physical example of each.
  - Answer: [Dirichlet, Neumann and Robin conditions](answers/boundary-conditions.md#dirichlet-neumann-and-robin-conditions)
- Why are Dirichlet conditions called "essential" and Neumann conditions "natural" in FEM?
  - Answer: [Essential and natural boundary conditions](answers/weak-form-and-galerkin.md#essential-and-natural-boundary-conditions)
- What is the CFL condition, and what happens if you violate it?
  - Answer: [The CFL condition](answers/time-integration-and-stability.md#the-cfl-condition)
- Explicit vs implicit time integration: what are the trade-offs?
  - Answer: [Explicit vs implicit time integration](answers/time-integration-and-stability.md#explicit-vs-implicit-time-integration)
- What is a stiff equation, and why does stiffness push you towards implicit methods?
  - Answer: [Stiff equations](answers/time-integration-and-stability.md#stiff-equations)
- How are large sparse linear systems solved? Compare direct solvers, conjugate gradient and multigrid.
  - Answer: [Sparse Linear Solvers](answers/sparse-linear-solvers.md)
- How does Newton–Raphson solve a nonlinear problem, and what do you do when it does not converge?
  - Answer: [Newton-Raphson and Nonlinear Solvers](answers/newton-raphson-and-nonlinear-solvers.md)
- Can machine learning speed up a nonlinear solver *without* changing its answer?
  - Answer: [Speeding up a solver with ML without changing its answer](answers/newton-raphson-and-nonlinear-solvers.md#speeding-up-a-solver-with-ml-without-changing-its-answer)
- What is a mesh convergence study? What is the grid convergence index?
  - Answer: [Mesh convergence and the grid convergence index](answers/verification-validation-and-mesh-convergence.md#mesh-convergence-and-the-grid-convergence-index)
- What are the sources of error in a simulation result?
  - Answer: [Sources of error in a simulation](answers/verification-validation-and-mesh-convergence.md#sources-of-error-in-a-simulation)
- Structured vs unstructured meshes; tetrahedra vs hexahedra vs prisms. What is mesh quality?
  - Answer: [Meshing and Mesh Quality](answers/meshing-and-mesh-quality.md)
- What is automatic differentiation, and how is it different from finite differences and symbolic differentiation?
  - Answer: [Automatic Differentiation](answers/automatic-differentiation.md)
- Hands-on: work through a PDE tutorial yourself.
  - Answer: [Hands-on: PDE tutorials](resources.md#hands-on-pde-tutorials)

## Structural Mechanics (FEA)

- Explain stress, strain, Young's modulus and Poisson's ratio. What is linear elasticity?
  - Answer: [Stress, Strain and Linear Elasticity](answers/stress-strain-and-linear-elasticity.md)
- What is von Mises stress, and why is it used for ductile metals?
  - Answer: [Von Mises Stress, Yield and Safety Factors](answers/von-mises-stress-and-yield.md)
- What is a stress concentration? How is it different from a stress singularity?
  - Answer: [Stress concentration vs stress singularity](answers/stress-concentrations-and-singularities.md#stress-concentration-vs-stress-singularity)
- Why does the peak stress at a sharp corner keep rising as you refine the mesh?
  - Answer: [Why the peak stress rises with mesh refinement](answers/stress-concentrations-and-singularities.md#why-the-peak-stress-rises-with-mesh-refinement)
- What is Saint-Venant's principle, and why does it matter when two models apply the same load differently?
  - Answer: [Saint-Venant's principle](answers/stress-concentrations-and-singularities.md#saint-venants-principle)
- What makes an analysis nonlinear (material, geometry, contact)?
  - Answer: [Three sources of nonlinearity](answers/nonlinear-structural-analysis.md#three-sources-of-nonlinearity)
- Why does plasticity make the solution depend on the load history, and why does that complicate a surrogate?
  - Answer: [Plasticity and load-history dependence](answers/nonlinear-structural-analysis.md#plasticity-and-load-history-dependence)
- What is buckling, and why is a linear buckling analysis an eigenvalue problem?
  - Answer: [Buckling](answers/buckling.md)
- What is modal analysis, and what are natural frequencies and mode shapes used for?
  - Answer: [Modal Analysis](answers/modal-analysis.md)
- What is fatigue, and why is fatigue life hard to predict with a surrogate?
  - Answer: [Fatigue](answers/fatigue.md)
- Linear tetrahedral elements are "too stiff" in bending. Why, and what do you do about it?
  - Answer: [Why linear tetrahedra are too stiff in bending](answers/elements-and-shape-functions.md#why-linear-tetrahedra-are-too-stiff-in-bending)
- How do you apply a bolt load or a remote load in FEA (rigid vs distributing couplings), and how does the choice affect nearby stresses?
  - Answer: [Bolt loads and remote loads](answers/boundary-conditions.md#bolt-loads-and-remote-loads)
- What is a safety factor? How would you predict one with a surrogate, and what goes wrong?
  - Answer: [Predicting a safety factor with a surrogate](answers/von-mises-stress-and-yield.md#predicting-a-safety-factor-with-a-surrogate)
- How do you check that an FEA result is sane before you trust it?
  - Answer: [Sanity checks before trusting an FEA result](answers/verification-validation-and-mesh-convergence.md#sanity-checks-before-trusting-an-fea-result)
- Why are crash-simulation surrogates especially hard?
  - Answer: [Why crash surrogates are hard](answers/crash-simulation-surrogates.md#why-crash-surrogates-are-hard)
- Which public structural datasets could you train on?
  - Answer: [Structural datasets](answers/public-simulation-datasets.md#structural-datasets)

## Fluid Dynamics (CFD)

- Write down the incompressible Navier–Stokes equations and explain each term.
  - Answer: [The incompressible Navier-Stokes equations](answers/cfd-essentials.md#the-incompressible-navier-stokes-equations)
- What is the Reynolds number, and what happens physically as it increases?
  - Answer: [The Reynolds number](answers/cfd-essentials.md#the-reynolds-number)
- Laminar vs turbulent flow: why is turbulence so expensive to simulate?
  - Answer: [Why turbulence is expensive to simulate](answers/turbulence-modelling.md#why-turbulence-is-expensive-to-simulate)
- RANS vs LES vs DNS: compare cost, accuracy and typical use.
  - Answer: [RANS vs LES vs DNS](answers/turbulence-modelling.md#rans-vs-les-vs-dns)
- What does a turbulence model such as Spalart–Allmaras or k-ω SST do?
  - Answer: [What a turbulence model does](answers/turbulence-modelling.md#what-a-turbulence-model-does)
- What is a boundary layer? What is y+, and why does it drive mesh design near walls?
  - Answer: [The boundary layer and y+](answers/boundary-layers-and-wall-treatment.md#the-boundary-layer-and-y)
- What are wall functions, and when are they acceptable?
  - Answer: [Wall functions](answers/boundary-layers-and-wall-treatment.md#wall-functions)
- Incompressible vs compressible flow: when do you need a compressible solver?
  - Answer: [Incompressible vs compressible flow](answers/cfd-essentials.md#incompressible-vs-compressible-flow)
- What is flow separation, and why do surrogates struggle near it?
  - Answer: [Flow separation](answers/boundary-layers-and-wall-treatment.md#flow-separation)
- What are the drag and lift coefficients, and how do you compute them from surface pressure and wall shear stress?
  - Answer: [Aerodynamic Forces and Coefficients](answers/aerodynamic-forces-and-coefficients.md)
- Steady vs transient simulation: when is a steady RANS answer misleading?
  - Answer: [Steady vs transient simulation](answers/cfd-essentials.md#steady-vs-transient-simulation)
- Which boundary conditions does an external-aerodynamics simulation use?
  - Answer: [External-aerodynamics boundary conditions](answers/boundary-conditions.md#external-aerodynamics-boundary-conditions)
- How does an incompressible solver couple pressure and velocity?
  - Answer: [Pressure-velocity coupling](answers/cfd-essentials.md#pressure-velocity-coupling)
- What does a typical external-aerodynamics dataset contain, and what is usually missing?
  - Answer: [What an external-aerodynamics dataset contains](answers/public-simulation-datasets.md#what-an-external-aerodynamics-dataset-contains)
- Why does a divergence-free constraint matter, and how can a model enforce it exactly?
  - Answer: [Enforcing a divergence-free field exactly](answers/hard-vs-soft-constraints.md#enforcing-a-divergence-free-field-exactly)
- Hands-on: which open-source CFD solvers would you use to generate data?
  - Answer: [Open-source solvers for generating data](answers/generating-simulation-training-data.md#open-source-solvers-for-generating-data)

## Heat Transfer and Multiphysics

- Explain conduction, convection and radiation, and the boundary conditions used for each.
  - Answer: [Conduction, convection and radiation](answers/heat-transfer-basics.md#conduction-convection-and-radiation)
- What is conjugate heat transfer?
  - Answer: [Conjugate heat transfer](answers/heat-transfer-basics.md#conjugate-heat-transfer)
- What is thermo-mechanical coupling? Explain one-way vs two-way coupling.
  - Answer: [One-way vs two-way coupling](answers/multiphysics-coupling.md#one-way-vs-two-way-coupling)
- Why does a mismatch in thermal expansion cause warpage in layered parts such as electronics packages?
  - Answer: [Thermal Stress and Warpage](answers/thermal-stress-and-warpage.md)
- Which dimensionless numbers matter in heat transfer (Prandtl, Nusselt, Biot), and why would you feed them to a model?
  - Answer: [Dimensionless numbers: Prandtl, Nusselt and Biot](answers/heat-transfer-basics.md#dimensionless-numbers-prandtl-nusselt-and-biot)
- What is the difference between partitioned and monolithic coupling in fluid–structure interaction?
  - Answer: [Partitioned vs monolithic coupling](answers/multiphysics-coupling.md#partitioned-vs-monolithic-coupling)
- How would you build a surrogate for the thermal design of an electronics enclosure or a data-center room?
  - Answer: [Design: Thermal Surrogate](answers/design-thermal-surrogate.md)

## Machine Learning Fundamentals for Physical Data

- How do you normalize physical fields and coordinates? Why per channel, and why only with training-set statistics?
  - Answer: [Normalizing fields and coordinates](answers/nondimensionalization-and-normalization.md#normalizing-fields-and-coordinates)
- Why non-dimensionalize inputs and outputs before training?
  - Answer: [Why non-dimensionalize](answers/nondimensionalization-and-normalization.md#why-non-dimensionalize)
- Your targets span several orders of magnitude across samples. MSE or relative L2 loss?
  - Answer: [MSE or relative L2 loss](answers/nondimensionalization-and-normalization.md#mse-or-relative-l2-loss)
- How do you handle heavy-tailed targets such as peak stress or stagnation pressure?
  - Answer: [Heavy-tailed targets](answers/nondimensionalization-and-normalization.md#heavy-tailed-targets)
- How do you control overfitting when you only have 50–500 simulations?
  - Answer: [Controlling overfitting with 50 to 500 simulations](answers/learning-from-small-simulation-datasets.md#controlling-overfitting-with-50-to-500-simulations)
- What is spectral bias, and why does it matter for physical fields?
  - Answer: [Spectral bias](answers/spectral-bias-and-fourier-features.md#spectral-bias)
- What are Fourier features or positional encodings for coordinates, and when do you need them?
  - Answer: [Fourier features and positional encodings](answers/spectral-bias-and-fourier-features.md#fourier-features-and-positional-encodings)
- How do you choose between a U-Net on a grid, a GNN on the mesh, a point-cloud network and a transformer?
  - Answer: [Choosing an Architecture](answers/choosing-an-architecture.md)
- What are invariance and equivariance? Which symmetries matter in engineering problems?
  - Answer: [Invariance and equivariance](answers/equivariance-and-symmetry.md#invariance-and-equivariance)
- When is data augmentation (rotation, mirroring, scaling) physically valid, and when is it wrong?
  - Answer: [When data augmentation is physically valid](answers/equivariance-and-symmetry.md#when-data-augmentation-is-physically-valid)
- How do you do model selection when you have very few simulations?
  - Answer: [Model selection with very few simulations](answers/learning-from-small-simulation-datasets.md#model-selection-with-very-few-simulations)
- What does the universal approximation theorem guarantee, and what does it not?
  - Answer: [The Universal Approximation Theorem](answers/universal-approximation.md)
- Explain backpropagation. Implement logistic regression from scratch.
  - Answer: [Backpropagation and Logistic Regression from Scratch](answers/backpropagation-and-logistic-regression.md)
- Why might you switch from Adam to L-BFGS late in training a physics model?
  - Answer: [Adam then L-BFGS](answers/optimizers-for-physics-models.md#adam-then-l-bfgs)
- Your model will not converge on a physical dataset. How do you debug it?
  - Answer: [Debugging Training on Physical Data](answers/debugging-training-on-physical-data.md)

## Physics-Informed Neural Networks (PINNs)

- What is a PINN? Write down its loss function.
  - Answer: [The PINN loss function](answers/pinns.md#the-pinn-loss-function)
- How are the derivatives in the PDE residual computed?
  - Answer: [Derivatives in the PDE residual](answers/pinns.md#derivatives-in-the-pde-residual)
- How do PINNs solve inverse problems, such as recovering a material parameter from sparse measurements?
  - Answer: [Inverse problems with PINNs](answers/pinns.md#inverse-problems-with-pinns)
- What are collocation points, and how should you choose them?
  - Answer: [Collocation points](answers/pinns.md#collocation-points)
- Why do PINNs often fail to train?
  - Answer: [Why PINNs fail to train](answers/pinn-training-failures.md#why-pinns-fail-to-train)
- What is loss imbalance, and how do adaptive loss weights work?
  - Answer: [Loss imbalance and adaptive weights](answers/pinn-training-failures.md#loss-imbalance-and-adaptive-weights)
- What is the causality problem in time-dependent PINNs?
  - Answer: [Causality in time-dependent PINNs](answers/pinn-training-failures.md#causality-in-time-dependent-pinns)
- Soft vs hard enforcement of boundary conditions: trade-offs?
  - Answer: [Soft vs hard boundary conditions in PINNs](answers/hard-vs-soft-constraints.md#soft-vs-hard-boundary-conditions-in-pinns)
- When does a PINN beat a classical solver, and when does it lose?
  - Answer: [When a PINN beats a classical solver and when it loses](answers/pinns.md#when-a-pinn-beats-a-classical-solver-and-when-it-loses)
- How do PINNs cope with high-frequency or multi-scale solutions?
  - Answer: [High-frequency and multi-scale solutions](answers/pinn-training-failures.md#high-frequency-and-multi-scale-solutions)
- What does a good PINN training recipe look like in practice?
  - Answer: [A PINN Training Recipe](answers/pinn-training-recipe.md)
- Why does a PINN usually have to be retrained for each new geometry or boundary condition, and why does that matter commercially?
  - Answer: [Why a PINN is retrained for every new problem](answers/pinns.md#why-a-pinn-is-retrained-for-every-new-problem)
- What is a physics-informed neural operator (PINO)?
  - Answer: [Physics-Informed Neural Operator (PINO)](answers/physics-informed-neural-operator.md)
- Where can I see PINNs taught properly from scratch?
  - Answer: [Hands-on: PINNs from scratch](resources.md#hands-on-pinns-from-scratch)

## Neural Operators

- What is operator learning, and how is it different from ordinary supervised learning?
  - Answer: [What operator learning is](answers/neural-operators.md#what-operator-learning-is)
- What does "discretization invariance" mean? Does it hold in practice?
  - Answer: [Discretization invariance](answers/neural-operators.md#discretization-invariance)
- Explain the Fourier Neural Operator architecture.
  - Answer: [Architecture](answers/fourier-neural-operator.md#architecture)
- Why does FNO keep only a limited number of Fourier modes?
  - Answer: [Why FNO keeps only a few Fourier modes](answers/fourier-neural-operator.md#why-fno-keeps-only-a-few-fourier-modes)
- What are FNO's limitations (uniform grids, periodicity, complex geometry), and how do Geo-FNO and GINO address them?
  - Answer: [Limitations, Geo-FNO and GINO](answers/fourier-neural-operator.md#limitations-geo-fno-and-gino)
- Explain DeepONet: what do the branch and trunk networks do?
  - Answer: [Branch and trunk networks](answers/deeponet.md#branch-and-trunk-networks)
- DeepONet vs FNO: when would you choose each?
  - Answer: [DeepONet vs FNO](answers/deeponet.md#deeponet-vs-fno)
- What is a graph neural operator?
  - Answer: [Graph neural operators](answers/neural-operators.md#graph-neural-operators)
- What is a convolutional neural operator (CNO)?
  - Answer: [Convolutional neural operators](answers/neural-operators.md#convolutional-neural-operators)
- How do neural operators handle time-dependent problems?
  - Answer: [Neural operators for time-dependent problems](answers/neural-operators.md#neural-operators-for-time-dependent-problems)
- Is attention a neural operator?
  - Answer: [Is attention a neural operator?](answers/transformers-for-physics.md#is-attention-a-neural-operator)
- How much training data does a neural operator need, and where does it come from?
  - Answer: [How much data a neural operator needs](answers/neural-operators.md#how-much-data-a-neural-operator-needs)
- Which benchmark problems are standard for neural operators?
  - Answer: [Standard benchmark problems for neural operators](answers/public-simulation-datasets.md#standard-benchmark-problems-for-neural-operators)
- Where can I learn neural operators hands-on?
  - Answer: [Hands-on: neural operators](resources.md#hands-on-neural-operators)

## Graph Neural Networks and Mesh-Based Learning

- What is message passing? Write the update equations.
  - Answer: [Message passing](answers/gnns-and-meshgraphnets.md#message-passing)
- What is the graph-network framework (edge, node and global updates)?
  - Answer: [The graph-network framework](answers/gnns-and-meshgraphnets.md#the-graph-network-framework)
- Explain MeshGraphNets: encoder, processor, decoder; mesh edges vs world edges.
  - Answer: [MeshGraphNets: encoder, processor, decoder](answers/gnns-and-meshgraphnets.md#meshgraphnets-encoder-processor-decoder)
- Which features go on the nodes and edges of a physics GNN, and why use relative positions?
  - Answer: [Node and edge features](answers/gnns-and-meshgraphnets.md#node-and-edge-features)
- How does the Graph Network-based Simulator (GNS) model particle systems?
  - Answer: [The Graph Network-based Simulator (GNS)](answers/graph-network-simulator-gns.md)
- How many message-passing steps do you need, and how does that relate to how far information must travel across the mesh?
  - Answer: [How many message-passing steps](answers/gnns-and-meshgraphnets.md#how-many-message-passing-steps)
- What are oversmoothing and oversquashing? How do multiscale GNNs help?
  - Answer: [Oversmoothing and oversquashing](answers/multiscale-gnns.md#oversmoothing-and-oversquashing)
- How do you build a graph from a point cloud: k-nearest neighbors or a radius graph?
  - Answer: [k-nearest-neighbour and radius graphs](answers/mesh-data-processing.md#k-nearest-neighbour-and-radius-graphs)
- GCN vs GAT vs GraphSAGE vs GIN: what are the differences?
  - Answer: [GNN Layer Types: GCN, GAT, GraphSAGE and GIN](answers/gnn-layer-types.md)
- How would you scale a GNN to a mesh with 10 million nodes?
  - Answer: [Scaling a GNN to 10 million nodes](answers/scaling-to-large-meshes.md#scaling-a-gnn-to-10-million-nodes)
- What did "Message Passing Neural PDE Solvers" add to earlier GNN simulators?
  - Answer: [Message Passing Neural PDE Solvers](answers/autoregressive-rollouts.md#message-passing-neural-pde-solvers)
- When do you need an equivariant GNN?
  - Answer: [When you need an equivariant GNN](answers/equivariance-and-symmetry.md#when-you-need-an-equivariant-gnn)
- What is a Graph U-Net?
  - Answer: [Graph U-Net](answers/multiscale-gnns.md#graph-u-net)
- Which libraries would you use?
  - Answer: [Libraries](resources.md#libraries)

## 3D Geometry and Geometric Deep Learning

- What are the ways to represent 3D geometry for machine learning (mesh, point cloud, voxels, SDF, CAD parameters, B-rep)? Pros and cons of each?
  - Answer: [Ways to represent 3D geometry](answers/geometry-representations.md#ways-to-represent-3d-geometry)
- What is a signed distance function, and why do so many surrogates take it as input?
  - Answer: [Signed distance functions](answers/geometry-representations.md#signed-distance-functions)
- Explain PointNet. Why does max pooling make it permutation-invariant?
  - Answer: [PointNet](answers/point-cloud-learning.md#pointnet)
- What did PointNet++ add?
  - Answer: [PointNet++](answers/point-cloud-learning.md#pointnet)
- What is EdgeConv (DGCNN), and what is a Point Transformer?
  - Answer: [DGCNN (EdgeConv) and Point Transformer](answers/point-cloud-learning.md#dgcnn-edgeconv-and-point-transformer)
- How do you sample points from a surface mesh? What is farthest point sampling?
  - Answer: [Sampling points from a surface mesh](answers/point-cloud-learning.md#sampling-points-from-a-surface-mesh)
- How do you batch geometries that have very different numbers of points?
  - Answer: [Batching geometries with different point counts](answers/point-cloud-learning.md#batching-geometries-with-different-point-counts)
- How do you represent a CAD model for machine learning? Which public CAD datasets exist?
  - Answer: [CAD models and CAD datasets](answers/geometry-representations.md#cad-models-and-cad-datasets)
- What is an implicit neural representation or neural field?
  - Answer: [Neural fields](answers/geometry-representations.md#neural-fields)
- Why can remeshing the same geometry change a model's prediction, and how do you test for it?
  - Answer: [Why remeshing can change a prediction](answers/geometry-representations.md#why-remeshing-can-change-a-prediction)
- How would you predict surface pressure on a car from its STL file?
  - Answer: [Worked example: surface pressure on a car from an STL](answers/geometry-representations.md#worked-example-surface-pressure-on-a-car-from-an-stl)
- What is geodesic convolution, and how was it used for shape optimization?
  - Answer: [Geodesic Convolutions and Shape Optimization](answers/geodesic-convolutions-and-shape-optimization.md)
- Should you align shapes to a canonical orientation before training?
  - Answer: [Canonical orientation or not](answers/equivariance-and-symmetry.md#canonical-orientation-or-not)

## Transformers for Physics and Large-Scale Surrogates

- Why is vanilla self-attention hard to use on a mesh with a million points?
  - Answer: [Why vanilla attention fails on a million points](answers/transformers-for-physics.md#why-vanilla-attention-fails-on-a-million-points)
- How does Transolver's Physics-Attention work?
  - Answer: [Transolver and Physics-Attention](answers/physics-transformer-architectures.md#transolver-and-physics-attention)
- What are Universal Physics Transformers, and why encode a mesh into a fixed set of latent tokens?
  - Answer: [Universal Physics Transformers](answers/physics-transformer-architectures.md#universal-physics-transformers)
- What are AB-UPT's anchor tokens, and how does AB-UPT enforce a divergence-free field?
  - Answer: [AB-UPT: anchor tokens and divergence-free fields](answers/physics-transformer-architectures.md#ab-upt-anchor-tokens-and-divergence-free-fields)
- What is DoMINO, and what is GeoTransolver?
  - Answer: [DoMINO](answers/physics-transformer-architectures.md#domino)
- What is linear (Galerkin-type) attention, and how is it used in operator learning?
  - Answer: [Linear and Galerkin-type attention](answers/transformers-for-physics.md#linear-and-galerkin-type-attention)
- How do you encode positions and boundary conditions as transformer inputs?
  - Answer: [Encoding positions and boundary conditions](answers/transformers-for-physics.md#encoding-positions-and-boundary-conditions)
- Why do recent car-aerodynamics models use separate surface and volume branches?
  - Answer: [Separate surface and volume branches](answers/physics-transformer-architectures.md#separate-surface-and-volume-branches)
- Do scaling laws hold for neural surrogates? Is a bigger model always better?
  - Answer: [Do scaling laws hold for neural surrogates?](answers/transformers-for-physics.md#do-scaling-laws-hold-for-neural-surrogates)
- When would a GNN beat a transformer on a mesh problem, and vice versa?
  - Answer: [GNN or transformer?](answers/transformers-for-physics.md#gnn-or-transformer)
- Which open code bases implement these models?
  - Answer: [Open code bases for physics models](resources.md#open-code-bases-for-physics-models)

## Surrogate Models, Reduced-Order Models and Design of Experiments

- Scalar surrogate (one number, such as drag) vs field surrogate (the whole pressure field): when do you need each?
  - Answer: [Scalar vs field surrogates](answers/surrogate-models.md#scalar-vs-field-surrogates)
- When are classical surrogates (polynomial response surfaces, radial basis functions, Kriging) enough?
  - Answer: [Classical Surrogates: Response Surfaces, RBF Interpolation and Kriging](answers/classical-surrogates.md)
- Explain Gaussian process regression. How does a GP estimate its own uncertainty?
  - Answer: [Gaussian process regression](answers/gaussian-processes.md#gaussian-process-regression)
- Which kernel would you choose, and how are the hyperparameters fitted?
  - Answer: [Choosing a kernel and fitting hyperparameters](answers/gaussian-processes.md#choosing-a-kernel-and-fitting-hyperparameters)
- Why do GPs scale badly with the number of samples, and what are the workarounds?
  - Answer: [Why GPs scale badly, and the workarounds](answers/gaussian-processes.md#why-gps-scale-badly-and-the-workarounds)
- What is proper orthogonal decomposition (POD), and how do you compute it from snapshots?
  - Answer: [Proper orthogonal decomposition](answers/reduced-order-models.md#proper-orthogonal-decomposition)
- Intrusive (Galerkin-projection) vs non-intrusive reduced-order models: what is the difference?
  - Answer: [Intrusive vs non-intrusive ROMs](answers/reduced-order-models.md#intrusive-vs-non-intrusive-roms)
- What is dynamic mode decomposition (DMD)?
  - Answer: [Dynamic mode decomposition](answers/reduced-order-models.md#dynamic-mode-decomposition)
- When does a linear subspace fail, and what do autoencoder-based ROMs offer instead?
  - Answer: [When a linear subspace fails](answers/reduced-order-models.md#when-a-linear-subspace-fails)
- What is SINDy?
  - Answer: [SINDy](answers/sindy-and-koopman.md#sindy)
- What is the Koopman operator, and how is deep learning used to approximate it?
  - Answer: [The Koopman operator](answers/sindy-and-koopman.md#the-koopman-operator)
- Latin hypercube vs Sobol vs random sampling: which would you use to plan simulations?
  - Answer: [Latin hypercube vs Sobol vs random sampling](answers/design-of-experiments-and-active-learning.md#latin-hypercube-vs-sobol-vs-random-sampling)
- How many simulations do you need to train a surrogate?
  - Answer: [How many simulations you need](answers/design-of-experiments-and-active-learning.md#how-many-simulations-you-need)
- What is active learning for simulations, and which acquisition function would you use?
  - Answer: [Active learning and acquisition functions](answers/design-of-experiments-and-active-learning.md#active-learning-and-acquisition-functions)
- What is global sensitivity analysis (Sobol indices), and how does it guide simplifying a design space?
  - Answer: [Global sensitivity analysis and Sobol indices](answers/design-of-experiments-and-active-learning.md#global-sensitivity-analysis-and-sobol-indices)
- How do you combine many cheap low-fidelity simulations with a few expensive high-fidelity ones?
  - Answer: [Multi-Fidelity Learning](answers/multi-fidelity-learning.md)
- What are Hamiltonian and Lagrangian neural networks?
  - Answer: [Hamiltonian and Lagrangian Neural Networks](answers/hamiltonian-and-lagrangian-neural-networks.md)
- A good book covering this whole area?
  - Answer: [Books (free online)](resources.md#books-free-online)

## Time-Dependent Problems and Autoregressive Rollouts

- Why does a model trained for one-step prediction drift when you roll it out for many steps?
  - Answer: [Why one-step models drift](answers/autoregressive-rollouts.md#why-one-step-models-drift)
- Why does injecting noise into training inputs help rollouts?
  - Answer: [Noise injection](answers/autoregressive-rollouts.md#noise-injection)
- What is the pushforward trick?
  - Answer: [The pushforward trick](answers/autoregressive-rollouts.md#the-pushforward-trick)
- What is unrolled training, and what is "solver-in-the-loop"?
  - Answer: [Unrolled training and solver-in-the-loop](answers/differentiable-physics-and-hybrid-solvers.md#unrolled-training-and-solver-in-the-loop)
- Why do long rollouts lose small-scale (high-frequency) detail, and how does PDE-Refiner address it?
  - Answer: [Losing small scales and PDE-Refiner](answers/autoregressive-rollouts.md#losing-small-scales-and-pde-refiner)
- How do you evaluate a rollout: error vs time, correlation time, energy spectrum?
  - Answer: [Evaluating a rollout](answers/autoregressive-rollouts.md#evaluating-a-rollout)
- What is time-bundling, and how do you choose the model's time step?
  - Answer: [Time-bundling and choosing the time step](answers/autoregressive-rollouts.md#time-bundling-and-choosing-the-time-step)
- What are neural ODEs?
  - Answer: [Neural ODEs](answers/neural-odes.md)
- How can machine learning correct a coarse-grid solver instead of replacing it?
  - Answer: [Correcting a coarse solver](answers/differentiable-physics-and-hybrid-solvers.md#correcting-a-coarse-solver)
- For a chaotic flow, what can a surrogate realistically predict?
  - Answer: [What a surrogate can predict for a chaotic flow](answers/autoregressive-rollouts.md#what-a-surrogate-can-predict-for-a-chaotic-flow)
- How do you detect that a rollout is about to blow up?
  - Answer: [Detecting a rollout that is about to blow up](answers/autoregressive-rollouts.md#detecting-a-rollout-that-is-about-to-blow-up)
- Which benchmarks test time-dependent neural PDE solvers?
  - Answer: [Benchmarks for time-dependent neural PDE solvers](answers/public-simulation-datasets.md#benchmarks-for-time-dependent-neural-pde-solvers)

## Uncertainty, Generalization and Out-of-Distribution Detection

- Aleatoric vs epistemic uncertainty: which dominates in a simulation surrogate?
  - Answer: [Aleatoric vs epistemic uncertainty](answers/uncertainty-quantification.md#aleatoric-vs-epistemic-uncertainty)
- How do deep ensembles estimate uncertainty?
  - Answer: [Deep ensembles](answers/uncertainty-quantification.md#deep-ensembles)
- What is Monte Carlo dropout, and why is it often overconfident?
  - Answer: [Monte Carlo dropout](answers/uncertainty-quantification.md#monte-carlo-dropout)
- What is conformal prediction? What does its coverage guarantee assume?
  - Answer: [Conformal Prediction](answers/conformal-prediction.md)
- How do you check that uncertainty estimates are calibrated?
  - Answer: [Checking calibration](answers/uncertainty-quantification.md#checking-calibration)
- Which UQ methods are used in scientific machine learning, and how are they compared?
  - Answer: [UQ methods in scientific ML and how they are compared](answers/uncertainty-quantification.md#uq-methods-in-scientific-ml-and-how-they-are-compared)
- What does "out of distribution" mean for a geometry, and how would you detect it?
  - Answer: [Detecting an out-of-distribution geometry](answers/out-of-distribution-and-generalization.md#detecting-an-out-of-distribution-geometry)
- When should the system fall back to running the full solver?
  - Answer: [When to fall back to the full solver](answers/out-of-distribution-and-generalization.md#when-to-fall-back-to-the-full-solver)
- A model was trained on Reynolds numbers from 1e5 to 1e6. A customer asks for 5e6. What do you say?
  - Answer: [Example: trained on Re 1e5 to 1e6, asked for 5e6](answers/out-of-distribution-and-generalization.md#example-trained-on-re-1e5-to-1e6-asked-for-5e6)
- How do you handle uncertainty when the training data itself comes from high-fidelity simulations with their own errors?
  - Answer: [When the training data has its own errors](answers/uncertainty-quantification.md#when-the-training-data-has-its-own-errors)
- How does a surrogate's uncertainty carry through to a design decision, such as passing a safety factor?
  - Answer: [From uncertainty to a design decision](answers/uncertainty-quantification.md#from-uncertainty-to-a-design-decision)
- A new design family arrives in production. How do you notice, and what do you do?
  - Answer: [A new design family arrives in production](answers/out-of-distribution-and-generalization.md#a-new-design-family-arrives-in-production)

## Simulation Data, Datasets and Benchmarks

- How would you generate a training set of 1,000 CFD simulations? Walk through the pipeline.
  - Answer: [Pipeline for 1,000 CFD simulations](answers/generating-simulation-training-data.md#pipeline-for-1000-cfd-simulations)
- What makes simulation training data "physically meaningful and sufficient" for a use case?
  - Answer: [What makes training data physically meaningful and sufficient](answers/generating-simulation-training-data.md#what-makes-training-data-physically-meaningful-and-sufficient)
- Which file formats will you meet (STL, OBJ, VTK/VTU, CGNS, HDF5, Zarr, solver input decks), and how do you handle terabytes of them?
  - Answer: [Simulation File Formats](answers/simulation-file-formats.md)
- How should you split simulation data into train, validation and test sets?
  - Answer: [Splitting simulation data](answers/data-splits-and-leakage.md#splitting-simulation-data)
- Give three ways information can leak from the test set into training in simulation datasets.
  - Answer: [Three ways test information leaks into training](answers/data-splits-and-leakage.md#three-ways-test-information-leaks-into-training)
- Name public datasets for car aerodynamics, airfoils and structures. What are their licences?
  - Answer: [Car aerodynamics datasets](answers/public-simulation-datasets.md#car-aerodynamics-datasets)
- What are PDEBench, The Well, PDEArena, CFDBench and APEBench used for?
  - Answer: [PDEBench, The Well, PDEArena, CFDBench and APEBench](answers/public-simulation-datasets.md#pdebench-the-well-pdearena-cfdbench-and-apebench)
- What did McGreivy & Hakim find about baselines in machine learning for fluid PDEs?
  - Answer: [Weak baselines and reporting biases](answers/ml-vs-numerical-solvers.md#weak-baselines-and-reporting-biases)
- The "ground truth" labels have discretization error. How should that change your evaluation?
  - Answer: [When the labels have discretization error](answers/verification-validation-and-mesh-convergence.md#when-the-labels-have-discretization-error)
- The customer has 40 proprietary simulations. Can you still deliver a useful model?
  - Answer: [A customer with 40 simulations](answers/learning-from-small-simulation-datasets.md#a-customer-with-40-simulations)
- How do you version simulation data together with models?
  - Answer: [Versioning data together with models](answers/production-and-deployment.md#versioning-data-together-with-models)

## Evaluation and Testing

- Which metrics would you use for field prediction: relative L2, MSE, maximum error?
  - Answer: [Field metrics](answers/evaluation-metrics.md#field-metrics)
- How do you evaluate integrated quantities such as drag, lift, maximum stress or mass flow?
  - Answer: [Integrated quantities](answers/evaluation-metrics.md#integrated-quantities)
- How do you report a speed-up fairly?
  - Answer: [Reporting a speed-up fairly](answers/evaluation-metrics.md#reporting-a-speed-up-fairly)
- Which baselines must a neural surrogate beat before anyone should care?
  - Answer: [Baselines to beat](answers/evaluation-metrics.md#baselines-to-beat)
- What physics-consistency checks would you run on predictions (conservation, divergence, equilibrium, symmetry)?
  - Answer: [Physics-consistency checks](answers/evaluation-metrics.md#physics-consistency-checks)
- How many test cases are enough? How do you report variability across seeds and splits?
  - Answer: [How many test cases, and variability across seeds](answers/evaluation-metrics.md#how-many-test-cases-and-variability-across-seeds)
- The customer will use the surrogate to rank designs, not for absolute values. How does that change evaluation?
  - Answer: [When the surrogate is used for ranking](answers/evaluation-metrics.md#when-the-surrogate-is-used-for-ranking)
- How do you evaluate generalization to unseen geometries?
  - Answer: [Evaluating generalization to unseen geometries](answers/evaluation-metrics.md#evaluating-generalization-to-unseen-geometries)
- Your model's average error is low, but it fails on the designs the customer cares about. What do you do?
  - Answer: [Low average error but failures where it matters](answers/evaluation-metrics.md#low-average-error-but-failures-where-it-matters)
- How would you build a regression test suite for a surrogate?
  - Answer: [A regression test suite for a surrogate](answers/production-and-deployment.md#a-regression-test-suite-for-a-surrogate)
- Which engineering-specific metrics did recent car-aerodynamics papers report?
  - Answer: [Engineering metrics in recent car-aerodynamics papers](answers/evaluation-metrics.md#engineering-metrics-in-recent-car-aerodynamics-papers)
- How would you evaluate an LLM agent that sets up simulations?
  - Answer: [How to evaluate a simulation agent](answers/llm-agents-for-simulation.md#how-to-evaluate-a-simulation-agent)

## Design Optimization, Inverse Problems and Generative Design

- Walk through a surrogate-based optimization loop.
  - Answer: [The optimization loop](answers/surrogate-based-optimization.md#the-optimization-loop)
- Gradients through a differentiable surrogate vs the adjoint method: compare.
  - Answer: [Surrogate gradients vs the adjoint](answers/surrogate-based-optimization.md#surrogate-gradients-vs-the-adjoint)
- Explain the adjoint method. Why does its cost not grow with the number of design variables?
  - Answer: [The Adjoint Method](answers/adjoint-method.md)
- Explain Bayesian optimization and acquisition functions.
  - Answer: [Bayesian Optimization](answers/bayesian-optimization.md)
- Why can an optimizer "exploit" a surrogate, and how do you prevent it?
  - Answer: [Why optimizers exploit surrogates](answers/surrogate-based-optimization.md#why-optimizers-exploit-surrogates)
- Explain topology optimization with SIMP.
  - Answer: [SIMP](answers/topology-optimization.md#simp)
- How has machine learning been used for topology optimization?
  - Answer: [Machine learning for topology optimization](answers/topology-optimization.md#machine-learning-for-topology-optimization)
- How do you identify unknown parameters from measurements (an inverse problem), and why does it need regularization?
  - Answer: [Inverse Problems and Regularization](answers/inverse-problems-and-regularization.md)
- How would you build an inverse-design tool: target performance in, geometry out?
  - Answer: [Building an inverse-design tool](answers/generative-models-for-physics.md#building-an-inverse-design-tool)
- How do you handle multiple objectives (for example drag vs downforce vs mass)?
  - Answer: [Multiple objectives](answers/surrogate-based-optimization.md#multiple-objectives)
- How do you keep optimized or generated designs manufacturable?
  - Answer: [Keeping designs manufacturable](answers/surrogate-based-optimization.md#keeping-designs-manufacturable)
- What are differentiable simulators, and when are they useful?
  - Answer: [Differentiable simulators](answers/differentiable-physics-and-hybrid-solvers.md#differentiable-simulators)

## Generative Models and Foundation Models for Physics

- How do denoising diffusion models work, and why use them for physics?
  - Answer: [Denoising diffusion models](answers/generative-models-for-physics.md#denoising-diffusion-models)
- What is flow matching?
  - Answer: [Flow matching](answers/generative-models-for-physics.md#flow-matching)
- How are diffusion models used for probabilistic forecasting?
  - Answer: [Probabilistic forecasting with diffusion](answers/generative-models-for-physics.md#probabilistic-forecasting-with-diffusion)
- How can a generative model solve a PDE from partial observations?
  - Answer: [Solving PDEs from partial observations](answers/generative-models-for-physics.md#solving-pdes-from-partial-observations)
- What is a foundation model for PDEs, and what has been shown to transfer?
  - Answer: [What a PDE foundation model is and what transfers](answers/foundation-models-for-physics.md#what-a-pde-foundation-model-is-and-what-transfers)
- What are the "large geometry" or "large engineering" models that companies talk about?
  - Answer: [Large geometry and large engineering models](answers/foundation-models-for-physics.md#large-geometry-and-large-engineering-models)
- What does machine-learning weather forecasting tell us about AI for simulation?
  - Answer: [Lessons from ML weather forecasting](answers/foundation-models-for-physics.md#lessons-from-ml-weather-forecasting)
- How would you fine-tune a pretrained physics model on a customer's small dataset?
  - Answer: [Fine-tuning on a small customer dataset](answers/foundation-models-for-physics.md#fine-tuning-on-a-small-customer-dataset)
- "Data is the moat in physics AI." Do you agree?
  - Answer: [Is data the moat?](answers/foundation-models-for-physics.md#is-data-the-moat)

## LLM Agents for Simulation and CAD

- What can LLM agents do in a CAE workflow today, and where do they fail?
  - Answer: [What agents can do today and where they fail](answers/llm-agents-for-simulation.md#what-agents-can-do-today-and-where-they-fail)
- What did FEABench measure, and what did it find?
  - Answer: [FEABench](answers/simulation-agents-research.md#feabench)
- How do Foam-Agent, MetaOpenFOAM and OpenFOAMGPT automate CFD?
  - Answer: [CFD agents: Foam-Agent, MetaOpenFOAM and OpenFOAMGPT](answers/simulation-agents-research.md#cfd-agents-foam-agent-metaopenfoam-and-openfoamgpt)
- How do agents automate FEA set-up?
  - Answer: [FEA agents](answers/simulation-agents-research.md#fea-agents)
- Why should the LLM not write solver physics directly?
  - Answer: [Why the LLM should not write solver physics](answers/llm-agents-for-simulation.md#why-the-llm-should-not-write-solver-physics)
- How do text-to-CAD systems work?
  - Answer: [Text-to-CAD](answers/text-to-cad.md)
- What is the ReAct pattern, and why does it suit tool-driven engineering workflows?
  - Answer: [The ReAct pattern](answers/llm-agents-for-simulation.md#the-react-pattern)
- How would you evaluate an agent that sets up and runs simulations?
  - Answer: [How to evaluate a simulation agent](answers/llm-agents-for-simulation.md#how-to-evaluate-a-simulation-agent)

## ML Engineering, Scaling and Production

- Roughly how much GPU memory does a GNN on a 1-million-node mesh need? Show the arithmetic.
  - Answer: [GPU memory for a GNN on a 1-million-node mesh](answers/scaling-to-large-meshes.md#gpu-memory-for-a-gnn-on-a-1-million-node-mesh)
- How can you train on subsampled points but predict at full resolution?
  - Answer: [Train on subsamples, predict at full resolution](answers/scaling-to-large-meshes.md#train-on-subsamples-predict-at-full-resolution)
- What are domain decomposition and graph partitioning with halo regions?
  - Answer: [Domain decomposition and halo regions](answers/scaling-to-large-meshes.md#domain-decomposition-and-halo-regions)
- When does mixed precision hurt a physics model?
  - Answer: [When mixed precision hurts](answers/scaling-to-large-meshes.md#when-mixed-precision-hurts)
- What is gradient checkpointing?
  - Answer: [Gradient checkpointing](answers/scaling-to-large-meshes.md#gradient-checkpointing)
- Data parallel vs model parallel training: when do you need each?
  - Answer: [Data parallel vs model parallel](answers/scaling-to-large-meshes.md#data-parallel-vs-model-parallel)
- How do you feed terabytes of simulation results to the GPUs without stalling?
  - Answer: [Feeding terabytes to the GPUs](answers/scaling-to-large-meshes.md#feeding-terabytes-to-the-gpus)
- When would you use Spark or Dask for simulation data?
  - Answer: [Spark and Dask](answers/scaling-to-large-meshes.md#spark-and-dask)
- How would you speed up a Python function that processes a 3D point cloud with millions of points?
  - Answer: [Making Python fast on millions of points](answers/mesh-data-processing.md#making-python-fast-on-millions-of-points)
- ONNX vs TorchScript (or `torch.export`) for deploying a model: trade-offs?
  - Answer: [ONNX vs TorchScript vs torch.export](answers/production-and-deployment.md#onnx-vs-torchscript-vs-torchexport)
- How do Docker and Kubernetes fit into training and serving surrogates?
  - Answer: [Docker and Kubernetes](answers/production-and-deployment.md#docker-and-kubernetes)
- How do you track experiments and manage model versions?
  - Answer: [Experiment tracking and model versions](answers/production-and-deployment.md#experiment-tracking-and-model-versions)
- How do you integrate a surrogate into an engineer's existing CAE workflow?
  - Answer: [Integrating into an engineer's CAE workflow](answers/production-and-deployment.md#integrating-into-an-engineers-cae-workflow)
- How do you monitor a deployed surrogate and decide when to retrain?
  - Answer: [Monitoring and retraining](answers/production-and-deployment.md#monitoring-and-retraining)
- Customers will not upload their CAD to your cloud. What are your options?
  - Answer: [When customers will not upload their CAD](answers/production-and-deployment.md#when-customers-will-not-upload-their-cad)

## Coding and Practical Implementation

Write these in NumPy or PyTorch without looking anything up. The first two are reported from real interviews (see [industry.md](industry.md)).

- Convert flat vertex and face arrays into a mesh data structure. *(reported in a PhysicsX coding round)*
  - Answer: [Flat arrays to a mesh](answers/mesh-data-processing.md#flat-arrays-to-a-mesh)
- Given a mesh dataset, identify the element types and compute the bounding box. *(reported in a PhysicsX live-coding round)*
  - Answer: [Element types and the bounding box](answers/mesh-data-processing.md#element-types-and-the-bounding-box)
- Compute triangle areas, face normals and area-weighted vertex normals, fully vectorized.
  - Answer: [Areas, face normals and vertex normals](answers/mesh-data-processing.md#areas-face-normals-and-vertex-normals)
- Extract the unique edges and the boundary edges of a triangle mesh.
  - Answer: [Unique edges and boundary edges](answers/mesh-data-processing.md#unique-edges-and-boundary-edges)
- Build vertex adjacency in sparse (CSR) form from a face array.
  - Answer: [Vertex adjacency in CSR form](answers/mesh-data-processing.md#vertex-adjacency-in-csr-form)
- Compute the volume of a closed triangle mesh.
  - Answer: [Volume of a closed mesh](answers/mesh-data-processing.md#volume-of-a-closed-mesh)
- Compute the drag force on a body from surface pressure and wall shear stress.
  - Answer: [Drag force from pressure and wall shear](answers/mesh-data-processing.md#drag-force-from-pressure-and-wall-shear)
- Interpolate a field from one mesh onto another (nearest neighbour, inverse distance, barycentric).
  - Answer: [Interpolating fields between meshes](answers/mesh-data-processing.md#interpolating-fields-between-meshes)
- Build a k-nearest-neighbour or radius graph with a k-d tree.
  - Answer: [k-nearest-neighbour and radius graphs](answers/mesh-data-processing.md#k-nearest-neighbour-and-radius-graphs)
- Implement farthest point sampling.
  - Answer: [Code: farthest point sampling](answers/point-cloud-learning.md#code-farthest-point-sampling)
- Implement the per-sample relative L2 error, including the area-weighted version for unstructured meshes.
  - Answer: [Code: per-sample relative L2 error](answers/evaluation-metrics.md#code-per-sample-relative-l2-error)
- Implement per-channel normalization that uses training-set statistics only.
  - Answer: [Code: per-channel normalizer](answers/nondimensionalization-and-normalization.md#code-per-channel-normalizer)
- Implement a 1D finite element solver for the Poisson equation (or an axially loaded bar).
  - Answer: [Code: 1D FEM for the Poisson equation](answers/finite-element-method.md#code-1d-fem-for-the-poisson-equation)
- Implement an explicit finite-difference solver for the 1D heat equation and check the stability limit.
  - Answer: [Code: explicit heat-equation solver and its stability limit](answers/time-integration-and-stability.md#code-explicit-heat-equation-solver-and-its-stability-limit)
- Implement a PINN for the 1D Poisson or Burgers equation with PyTorch autograd.
  - Answer: [Code: a PINN for the 1D Poisson equation](answers/pinns.md#code-a-pinn-for-the-1d-poisson-equation)
- Implement a 1D Fourier layer: `rfft`, keep the first k modes, multiply by complex weights, `irfft`.
  - Answer: [Code: a 1D Fourier layer](answers/fourier-neural-operator.md#code-a-1d-fourier-layer)
- Implement one message-passing layer: edge MLP, sum aggregation, node MLP, residual connection.
  - Answer: [Code: one message-passing layer](answers/gnns-and-meshgraphnets.md#code-one-message-passing-layer)
- Implement PointNet: a shared per-point MLP followed by max pooling.
  - Answer: [Code: a minimal PointNet](answers/point-cloud-learning.md#code-a-minimal-pointnet)
- Implement an autoregressive rollout loop with training-time noise injection.
  - Answer: [Code: rollout with noise injection](answers/autoregressive-rollouts.md#code-rollout-with-noise-injection)
- Implement Gaussian process regression from scratch (RBF kernel, Cholesky solve, predictive variance).
  - Answer: [Code: GP regression from scratch](answers/gaussian-processes.md#code-gp-regression-from-scratch)
- Compute POD modes from a snapshot matrix with the SVD.
  - Answer: [Code: POD with the SVD](answers/reduced-order-models.md#code-pod-with-the-svd)
- Implement Latin hypercube sampling.
  - Answer: [Code: Latin hypercube sampling](answers/design-of-experiments-and-active-learning.md#code-latin-hypercube-sampling)
- Implement split conformal prediction intervals for a regression model.
  - Answer: [Code: split conformal intervals](answers/conformal-prediction.md#code-split-conformal-intervals)
- General algorithm questions also appear (three problems on Coderbyte or CoderPad are reported). Practise standard data-structure problems.
  - Answer: [Coding Round Preparation](answers/coding-round-preparation.md)
- Read a VTK or STL file, attach a field, and save it for ParaView.
  - Answer: [Code: read a mesh, attach a field, save for ParaView](answers/simulation-file-formats.md#code-read-a-mesh-attach-a-field-save-for-paraview)

## AI-for-Simulation System Design

Answer every question with the same structure: [System design framework](answers/system-design-framework.md).

- Design a system that predicts drag and surface pressure for a new car shape in seconds, from its STL.
  - Answer: [Design: Car Aerodynamics Surrogate](answers/design-car-aerodynamics-surrogate.md)
- Design a platform where engineers upload CAD and get stress predictions with a confidence level.
  - Answer: [Design: Stress Prediction Platform](answers/design-stress-prediction-platform.md)
- Design the pipeline that generates and stores 10,000 CFD simulations for training.
  - Answer: [Design: Simulation Data Pipeline](answers/design-simulation-data-pipeline.md)
- Design an active learning loop that decides which simulation to run next.
  - Answer: [Design: Active Learning Loop](answers/design-active-learning-loop.md)
- Design a lightweighting loop for a structural bracket that never ships an unverified design.
  - Answer: [Design: Bracket Lightweighting Loop](answers/design-bracket-lightweighting-loop.md)
- Design a validation strategy that would convince a sceptical customer to trust your surrogate.
  - Answer: [Design: Validation Strategy](answers/design-validation-strategy.md)
- Design a real-time digital twin of a heat exchanger that combines sensor data with simulation.
  - Answer: [Design: Heat Exchanger Digital Twin](answers/design-heat-exchanger-digital-twin.md)
- Design a crash-simulation surrogate for a vehicle programme.
  - Answer: [System design: a crash surrogate for a vehicle programme](answers/crash-simulation-surrogates.md#system-design-a-crash-surrogate-for-a-vehicle-programme)
- Design a thermal surrogate for a battery pack or an electronics enclosure.
  - Answer: [Design: Thermal Surrogate](answers/design-thermal-surrogate.md)
- Design a surrogate that combines plenty of RANS data with a little LES data.
  - Answer: [Design: RANS–LES Multi-Fidelity Surrogate](answers/design-rans-les-multi-fidelity-surrogate.md)
- Design a pipeline that fine-tunes a pretrained physics model on a customer's 50 simulations.
  - Answer: [Design: Fine-Tuning Pipeline](answers/design-fine-tuning-pipeline.md)
- Design an inverse-design tool: a target pressure distribution in, a geometry out.
  - Answer: [Design: Inverse Design Tool](answers/design-inverse-design-tool.md)
- Design an LLM agent that runs a structural check from a plain-language request.
  - Answer: [Design: LLM Structural Check Agent](answers/design-llm-structural-check-agent.md)
- Design a serving system for 1,000 engineers using a surrogate interactively.
  - Answer: [Design: Surrogate Serving System](answers/design-surrogate-serving-system.md)
- Design monitoring that detects when incoming designs drift outside the training data.
  - Answer: [Design: Drift Monitoring](answers/design-drift-monitoring.md)

## Behavioral and Customer-Facing Scenarios

Customer scenarios are covered in [Customer scenarios](answers/customer-scenarios.md) and stories about your own work in [Behavioural questions](answers/behavioral-questions.md).

- A customer's simulation engineer says: "I don't trust black boxes." How do you respond?
  - Answer: [I don't trust black boxes](answers/customer-scenarios.md#i-dont-trust-black-boxes)
- Your surrogate disagrees with wind-tunnel data. What do you do?
  - Answer: [The surrogate disagrees with wind-tunnel data](answers/customer-scenarios.md#the-surrogate-disagrees-with-wind-tunnel-data)
- A customer wants 1% accuracy from 30 simulations. How do you handle the conversation?
  - Answer: [1% accuracy from 30 simulations](answers/customer-scenarios.md#1-accuracy-from-30-simulations)
- One week before delivery, you find the model fails on a design family the customer cares about. What do you do?
  - Answer: [A failure found one week before delivery](answers/customer-scenarios.md#a-failure-found-one-week-before-delivery)
- Explain a neural operator to a non-ML engineering manager in two minutes.
  - Answer: [Explaining a neural operator to a manager](answers/customer-scenarios.md#explaining-a-neural-operator-to-a-manager)
- Tell me about a time you turned research into something people actually used.
  - Answer: [Research that people actually used](answers/behavioral-questions.md#research-that-people-actually-used)
- Tell me about a time you found out your own results were wrong.
  - Answer: [Finding out your own results were wrong](answers/behavioral-questions.md#finding-out-your-own-results-were-wrong)
- How do you choose between a quick proof of concept and a robust product feature?
  - Answer: [Proof of concept or robust feature](answers/customer-scenarios.md#proof-of-concept-or-robust-feature)
- How do you scope a project with a new customer in the first meeting?
  - Answer: [Scoping a project in the first meeting](answers/customer-scenarios.md#scoping-a-project-in-the-first-meeting)
- You disagree with a domain expert about the physics. How do you resolve it?
  - Answer: [Disagreeing with a domain expert](answers/customer-scenarios.md#disagreeing-with-a-domain-expert)
- When would you tell a customer that machine learning is the wrong tool?
  - Answer: [When ML is the wrong tool](answers/customer-scenarios.md#when-ml-is-the-wrong-tool)
- Present a technical project you have worked on. *(reported as a final-round format)*
  - Answer: [Presenting a technical project](answers/behavioral-questions.md#presenting-a-technical-project)
- Why this company, and why AI for simulation?
  - Answer: [Why this company, and why AI for simulation](answers/behavioral-questions.md#why-this-company-and-why-ai-for-simulation)
- How do you keep up with a field that moves this fast?
  - Answer: [Keeping up with the field](answers/behavioral-questions.md#keeping-up-with-the-field)

## Know the Industry

- What do PhysicsX, Neural Concept, Luminary, Emmi AI, Monolith, Navier AI and BeyondMath do, and how do they differ?
  - Answer: [Companies](industry.md#companies)
- What is Ansys SimAI?
  - Answer: [Ansys SimAI](industry.md#ansys-simai)
- What is NVIDIA PhysicsNeMo?
  - Answer: [NVIDIA PhysicsNeMo](industry.md#nvidia-physicsnemo)
- What changed when Synopsys acquired Ansys and Siemens acquired Altair?
  - Answer: [Synopsys-Ansys and Siemens-Altair](industry.md#synopsys-ansys-and-siemens-altair)
- What did PhysicsX release with LGM-Aero and Ai.rplane?
  - Answer: [PhysicsX LGM-Aero and Ai.rplane](industry.md#physicsx-lgm-aero-and-airplane)
- What are Luminary's SHIFT models and datasets?
  - Answer: [Luminary SHIFT models and datasets](industry.md#luminary-shift-models-and-datasets)
- How do physics-AI companies make money: platform licences, services or forward-deployed engineering?
  - Answer: [How physics-AI companies make money](industry.md#how-physics-ai-companies-make-money)
- Which industries buy these tools, and why do some customers hesitate?
  - Answer: [Which industries buy, and why customers hesitate](industry.md#which-industries-buy-and-why-customers-hesitate)

---

## How This Guide Was Built

- **Questions** come from job descriptions at AI-for-simulation companies, candidate interview reports, and the standard papers and courses of the field. Reported interview items are marked as such, with sources in [industry.md](industry.md).
- **Answers** are written for this guide in [`answers/`](answers/README.md). Each question links to the section that answers it; each page starts with a short answer, goes into detail, and ends with "In an interview" (a model answer and likely follow-ups) and its references.
- **Code** on the answer pages was run before publishing, and the outputs shown are the real outputs. Timings depend on the machine.
- **Links** were checked when they were added. If one breaks, please open an issue.
- Company facts (funding, products, acquisitions) are dated. Check them again before an interview.

Format inspired by [AI Engineering Interview Questions and Answers](https://github.com/amitshekhariitbhu/ai-engineering-interview-questions).
