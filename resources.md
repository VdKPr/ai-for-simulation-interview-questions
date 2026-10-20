# Learning Resources

[← Questions](README.md) · [All answers](answers/README.md)

A short, opinionated list. Start with one course, one book and one library; add the rest when you need them.

## Courses and lectures

| Resource | Why |
| --- | --- |
| [ETH Zürich: AI in the Sciences and Engineering (2024 lecture playlist)](https://www.youtube.com/playlist?list=PLJkYEExhe7rYFkBIB2U5pf_RWzYnFLj7r), [course page](https://www.camlab.ethz.ch/teaching/ai-in-the-sciences-and-engineering-2024.html) and [tutorial notebooks](https://github.com/camlab-ethz/AI_Science_Engineering) | The most complete free course: PINNs, neural operators, attention as an operator, hybrid workflows, diffusion, JAX |
| [ETH Zürich: AI in the Sciences and Engineering (current edition)](https://camlab-ethz.github.io/ai4s-course/) | The newer course site; adds GNNs and transformers for complex geometry, generative AI and foundation models |
| [Physics Informed Machine Learning (Steve Brunton, playlist)](https://www.youtube.com/playlist?list=PLMrJAkhIeNNQ0BaKuBKY43k4xMo6NSbBa) | Short, clear videos: choosing what to model, data, architecture, loss, optimization; SINDy, FNO, DeepONet, PINNs |
| [Parallel Computing and Scientific Machine Learning (MIT 18.337 notes)](https://book.sciml.ai/) | Numerics, automatic differentiation and SciML from the ground up ([source](https://github.com/SciML/SciMLBook)) |
| [The FEniCSx tutorial (Jørgen S. Dokken)](https://jsdokken.com/dolfinx-tutorial/) | Learn FEM by writing it |
| [CFD Python: 12 steps to Navier–Stokes (Lorena Barba)](https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/) ([notebooks](https://github.com/barbagroup/CFDPython)) | Finite differences and CFD by hand, in Python |

## Books (free online)

- [Physics-based Deep Learning (Thuerey et al.)](https://physicsbaseddeeplearning.org/): physical losses, differentiable physics, solver-in-the-loop, with code ([source](https://github.com/tum-pbs/pbdl-book))
- [Data-Driven Science and Engineering (Brunton & Kutz)](http://databookuw.com/): SVD/POD, DMD, SINDy, ML for dynamics
- [Gaussian Processes for Machine Learning (Rasmussen & Williams)](https://gaussianprocess.org/gpml/)

## Interview preparation

- [Machine Learning Interviews Book (Chip Huyen)](https://huyenchip.com/ml-interviews-book/) ([source](https://github.com/chiphuyen/ml-interviews-book))
- [Machine-Learning-Interviews (alirezadir)](https://github.com/alirezadir/Machine-Learning-Interviews)
- [AI Engineering Interview Questions and Answers (Amit Shekhar)](https://github.com/amitshekhariitbhu/ai-engineering-interview-questions), for the LLM side of these roles
- Company-specific reports and aggregator guides are listed in [industry.md](industry.md#reported-interview-formats)

## Paper lists (maintained by others)

- [awesome-scientific-machine-learning](https://github.com/MartinuzziFrancesco/awesome-scientific-machine-learning)
- [PINNpapers](https://github.com/idrl-lab/PINNpapers)
- [Awesome-AI4PDE](https://github.com/lizhihao2022/Awesome-AI4PDE)
- [Awesome-Transformer-PDE](https://github.com/XiaojinStu/Awesome-Transformer-PDE)
- [awesome-machine-learning-fluid-mechanics](https://github.com/ikespand/awesome-machine-learning-fluid-mechanics)
- [Awesome-AI4CFD](https://github.com/WillDreamer/Awesome-AI4CFD)
- [awesome-neural-ode](https://github.com/Zymrael/awesome-neural-ode)

## Twenty papers to know

| Topic | Paper |
| --- | --- |
| PINNs | [Raissi et al., Part I](https://arxiv.org/abs/1711.10561) · [Krishnapriyan et al., failure modes](https://arxiv.org/abs/2109.01050) · [Expert's guide](https://arxiv.org/abs/2308.08468) |
| Neural operators | [FNO](https://arxiv.org/abs/2010.08895) · [DeepONet](https://arxiv.org/abs/1910.03193) · [Neural Operator (JMLR)](https://arxiv.org/abs/2108.08481) · [GINO](https://arxiv.org/abs/2309.00583) |
| Graph networks | [MeshGraphNets](https://arxiv.org/abs/2010.03409) · [GNS](https://arxiv.org/abs/2002.09405) · [Message Passing Neural PDE Solvers](https://arxiv.org/abs/2202.03376) |
| Transformers at scale | [Transolver](https://arxiv.org/abs/2402.02366) · [UPT](https://arxiv.org/abs/2402.12365) · [AB-UPT](https://arxiv.org/abs/2502.09692) · [DoMINO](https://arxiv.org/abs/2501.13350) |
| Geometry | [PointNet](https://arxiv.org/abs/1612.00593) · [Geodesic Convolutional Shape Optimization](https://arxiv.org/abs/1802.04016) |
| Rollouts and hybrids | [PDE-Refiner](https://arxiv.org/abs/2308.05732) · [Solver-in-the-Loop](https://arxiv.org/abs/2007.00016) · [ML-accelerated CFD](https://arxiv.org/abs/2102.01010) |
| A reality check | [Weak baselines and reporting biases (McGreivy & Hakim)](https://arxiv.org/abs/2407.07218) |

## Hands-on: PDE tutorials

Write a solver yourself before learning to replace one. In order of difficulty:

1. **Finite differences by hand:** [CFD Python: 12 steps to Navier–Stokes](https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/) ([notebooks](https://github.com/barbagroup/CFDPython)) goes from 1D linear convection to 2D Navier–Stokes in Python, one concept per step (upwinding, the CFL condition, diffusion, pressure Poisson).
2. **Finite elements by writing the weak form:** [The FEniCSx tutorial](https://jsdokken.com/dolfinx-tutorial/) covers Poisson, linear elasticity, heat and Navier–Stokes with real meshes and boundary conditions.
3. **In this guide:** a 1D FEM solver ([Code: 1D FEM for the Poisson equation](answers/finite-element-method.md#code-1d-fem-for-the-poisson-equation)) and an explicit heat solver with its stability limit ([Code: explicit heat-equation solver](answers/time-integration-and-stability.md#code-explicit-heat-equation-solver-and-its-stability-limit)), both short enough to rewrite in an interview.
4. **Real solvers:** run an [OpenFOAM](https://github.com/OpenFOAM/OpenFOAM-dev) tutorial case (lid-driven cavity, then a 2D airfoil), mesh a part with [Gmsh](https://gmsh.info/doc/texinfo/gmsh.html), and post-process in ParaView with [PyVista](https://github.com/pyvista/pyvista).

Exercise: for each solver, run a mesh-refinement study and check the observed order of convergence ([Mesh convergence](answers/verification-validation-and-mesh-convergence.md#mesh-convergence-and-the-grid-convergence-index)).

## Hands-on: PINNs from scratch

1. **In this guide:** [Code: a PINN for the 1D Poisson equation](answers/pinns.md#code-a-pinn-for-the-1d-poisson-equation), about 40 lines of PyTorch with autograd derivatives.
2. **ETH course notebooks:** the [AI in the Sciences and Engineering tutorials](https://github.com/camlab-ethz/AI_Science_Engineering) include PINNs for forward and inverse problems.
3. **A library, once you understand the basics:** [DeepXDE](https://github.com/lululxvi/deepxde) has many worked examples (Burgers, Poisson on complex domains, inverse problems).
4. **Failure modes:** reproduce the convection failure of [Krishnapriyan et al.](https://arxiv.org/abs/2109.01050) and try their curriculum fix, then read the [Expert's guide to training PINNs](https://arxiv.org/abs/2308.08468).

Exercise: solve Burgers' equation with a PINN, then with a finite-difference solver, and compare accuracy and wall-clock time honestly. See [PINN training failures](answers/pinn-training-failures.md).

## Hands-on: neural operators

1. **In this guide:** [Code: a 1D Fourier layer](answers/fourier-neural-operator.md#code-a-1d-fourier-layer).
2. **The reference implementations:** the original [Fourier Neural Operator code](https://github.com/zongyi-li/fourier_neural_operator) and [DeepONet code](https://github.com/lululxvi/deeponet); the maintained [neuraloperator library](https://github.com/neuraloperator/neuraloperator).
3. **Benchmarks to train on:** [PDEBench](https://github.com/pdebench/PDEBench) (1D Burgers and advection are small enough for a laptop GPU), then [PDEArena](https://github.com/pdearena/pdearena).
4. **Beyond regular grids:** [Transolver](https://github.com/thuml/Transolver) and the [Neural-Solver-Library](https://github.com/thuml/Neural-Solver-Library) on [AirfRANS](https://github.com/Extrality/AirfRANS).

Exercise: train an FNO on 1D Burgers at one resolution, evaluate at a finer one, and measure how much accuracy actually transfers ([Discretization invariance](answers/neural-operators.md#discretization-invariance)).

## Open code bases for physics models

| Model family | Code |
| --- | --- |
| MeshGraphNets, Graph Network Simulator | [meshgraphnets](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets) and [learning_to_simulate](https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate) (DeepMind, TensorFlow); PyTorch versions in [PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) |
| Neural operators | [neuraloperator](https://github.com/neuraloperator/neuraloperator), [DeepONet](https://github.com/lululxvi/deeponet) |
| Transformers for physics | [Transolver](https://github.com/thuml/Transolver), [UPT](https://github.com/ml-jku/UPT), [Noether (AB-UPT)](https://github.com/Emmi-AI/noether), DoMINO and GeoTransolver in [PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) |
| Autoregressive PDE surrogates | [PDEArena](https://github.com/pdearena/pdearena), [APEBench](https://github.com/tum-pbs/apebench) |
| PINNs | [DeepXDE](https://github.com/lululxvi/deepxde), [PINA](https://github.com/mathLab/PINA) |
| Differentiable solvers | [PhiFlow](https://github.com/tum-pbs/PhiFlow), [JAX-CFD](https://github.com/google/jax-cfd), [JAX-FEM](https://github.com/deepmodeling/jax-fem) |
| Point clouds and meshes | [PyG](https://github.com/pyg-team/pytorch_geometric), [PyTorch3D](https://github.com/facebookresearch/pytorch3d) |

Reading one of these end to end (data loading, normalization, model, training loop, evaluation) is the fastest way to learn the engineering that papers leave out.

## Libraries

**Physics-ML frameworks**

- [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) ([docs](https://docs.nvidia.com/physicsnemo/latest/)) and [PhysicsNeMo-CFD](https://github.com/NVIDIA/physicsnemo-cfd)
- [neuraloperator](https://github.com/neuraloperator/neuraloperator)
- [DeepXDE](https://github.com/lululxvi/deepxde)
- [PINA](https://github.com/mathLab/PINA), [PaddleScience](https://github.com/PaddlePaddle/PaddleScience), [NeuralPDE.jl](https://github.com/SciML/NeuralPDE.jl)
- [Neural-Solver-Library](https://github.com/thuml/Neural-Solver-Library), [Transolver](https://github.com/thuml/Transolver), [UPT](https://github.com/ml-jku/UPT), [Noether (Emmi AI)](https://github.com/Emmi-AI/noether)

**Graphs and 3D**

- [PyG](https://github.com/pyg-team/pytorch_geometric), [DGL](https://github.com/dmlc/dgl)
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d), [Open3D](https://github.com/isl-org/Open3D), [trimesh](https://github.com/mikedh/trimesh)

**Differentiable simulation**

- [PhiFlow](https://github.com/tum-pbs/PhiFlow), [JAX-CFD](https://github.com/google/jax-cfd), [JAX-FEM](https://github.com/deepmodeling/jax-fem), [NVIDIA Warp](https://github.com/NVIDIA/warp), [Diffrax](https://github.com/patrick-kidger/diffrax)

**Surrogates, optimization and UQ**

- [SMT](https://github.com/SMTorg/smt), [GPyTorch](https://github.com/cornellius-gp/gpytorch), [BoTorch](https://github.com/pytorch/botorch), [Optuna](https://github.com/optuna/optuna), [MAPIE](https://github.com/scikit-learn-contrib/MAPIE)

**Equation discovery**

- [PySINDy](https://github.com/dynamicslab/pysindy), [PySR](https://github.com/MilesCranmer/PySR)

## Simulation and geometry tools (for generating your own data)

- **FEM:** [FEniCSx / DOLFINx](https://github.com/FEniCS/dolfinx), [deal.II](https://github.com/dealii/dealii), [MFEM](https://github.com/mfem/mfem), [Firedrake](https://github.com/firedrakeproject/firedrake)
- **CFD:** [OpenFOAM](https://github.com/OpenFOAM/OpenFOAM-dev), [SU2](https://github.com/su2code/SU2)
- **Meshing and CAD:** [Gmsh](https://gmsh.info/doc/texinfo/gmsh.html), [CadQuery](https://github.com/CadQuery/cadquery)
- **Coupling:** [preCICE](https://github.com/precice/precice)
- **Mesh input/output and visualization:** [meshio](https://github.com/nschloe/meshio), [PyVista](https://github.com/pyvista/pyvista)

## Datasets and benchmarks

Check each dataset's licence before using it commercially. Several popular ones are **non-commercial**.

| Dataset | Domain | Notes |
| --- | --- | --- |
| [DrivAerNet](https://arxiv.org/abs/2403.08055) / [DrivAerNet++](https://arxiv.org/abs/2406.09624) ([repo](https://github.com/Mohamedelrefaie/DrivAerNet)) | Car aerodynamics (CFD) | Large parametric car dataset with deep-learning benchmarks; DrivAerNet++ is CC BY-NC |
| [DrivAerML](https://arxiv.org/abs/2408.11969) | Road-car aerodynamics | High-fidelity hybrid RANS–LES |
| [AhmedML](https://arxiv.org/abs/2407.20801), [WindsorML](https://arxiv.org/abs/2407.19320) | Bluff-body / automotive aerodynamics | High-fidelity CFD |
| [SHIFT-SUV](https://huggingface.co/datasets/luminary-shift/SUV), [SHIFT-Wing](https://huggingface.co/datasets/luminary-shift/WING) | SUV and transonic-wing aerodynamics | Luminary; CC-BY-NC-4.0; terabyte scale |
| [AirfRANS](https://arxiv.org/abs/2212.07564) ([repo](https://github.com/Extrality/AirfRANS)) | 2D airfoil RANS | A good first field-prediction project |
| [EAGLE](https://arxiv.org/abs/2302.10803) | Turbulent fluid dynamics on meshes | Mesh-transformer benchmark |
| [SimJEB](https://simjeb.github.io/) ([paper](https://arxiv.org/abs/2105.03534)) | Structural brackets (FEA) | 381 designs; data ODC-By, CAD non-commercial; predefined splits |
| [DeepJEB](https://arxiv.org/abs/2406.09047) | Synthetic jet-engine brackets | Larger structural set |
| [PDEBench](https://arxiv.org/abs/2210.07182) ([repo](https://github.com/pdebench/PDEBench)) | Many PDEs | Standard scientific-ML benchmark |
| [The Well](https://arxiv.org/abs/2412.00568) ([repo](https://github.com/PolymathicAI/the_well)) | Diverse physics simulations | Large and varied |
| [PDEArena](https://arxiv.org/abs/2209.15616) ([repo](https://github.com/pdearena/pdearena)) | Time-dependent PDE surrogates | Fluid benchmarks with strong baselines |
| [CFDBench](https://arxiv.org/abs/2310.05963) | Fluid dynamics | ML methods for CFD |
| [APEBench](https://github.com/tum-pbs/apebench) | Autoregressive PDE emulators | Benchmark suite |
| [LagrangeBench](https://arxiv.org/abs/2309.16342) ([repo](https://github.com/tumaer/lagrangebench)) | Particle-based fluids | GNS-style models |
| [BubbleML](https://arxiv.org/abs/2307.14623) | Multiphase boiling | Multiphysics |
| [ABC](https://arxiv.org/abs/1812.06216), [Fusion 360 Gallery](https://arxiv.org/abs/2010.02392), [DeepCAD](https://arxiv.org/abs/2105.09492) | CAD models | Geometry and CAD-generation work |

## Company research and documentation

- [NVIDIA PhysicsNeMo documentation](https://docs.nvidia.com/physicsnemo/latest/)
- [Ansys SimAI: Explaining SimAI](https://ansys.synopsys.com/blog/explaining-simai) and [PySimAI](https://simai.docs.pyansys.com)
- [PhysicsX newsroom: LGM-Aero](https://www.physicsx.ai/newsroom/introducing-lgm-aero-genai-for-aero-engineering-and-airplane-showcase-application-for-aerostructures)
- [Luminary: SHIFT models](https://luminary.ai/resources/introducing-luminary-shift-models-a-suite-of-physics-ai-foundation-models-to-transform-engineering-design/)
