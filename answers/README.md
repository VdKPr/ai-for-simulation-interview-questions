# Answer Pages

[← Questions](../README.md)

Every question in the guide links to a section of one of these pages. Each page starts with a **short answer** you could give in the first thirty seconds, then goes into detail with worked examples, ends with **In an interview** (a model answer and the likely follow-ups) and its **references**. Code on these pages was run before it was published, and the outputs shown are the real outputs.

See also: [Learning resources](../resources.md) and [Industry and interview notes](../industry.md).

## Must Know

| Page | Covers |
| --- | --- |
| [Surrogate Models](surrogate-models.md) | What a surrogate is; What goes in and what comes out; Types of surrogate; Why engineering companies want surrogates |
| [ML vs Numerical Solvers](ml-vs-numerical-solvers.md) | Side by side; Where ML helps; When not to replace a solver with ML; Break-even arithmetic |
| [Neural Operators](neural-operators.md) | PINN vs neural operator; What operator learning is; The neural-operator layer: a learned integral kernel; Discretization invariance |
| [GNNs and MeshGraphNets](gnns-and-meshgraphnets.md) | Why GNNs fit simulation meshes; Message passing; The graph-network framework; MeshGraphNets: encoder, processor, decoder |
| [The CAE Workflow, and Where ML Fits](cae-workflow.md) | The workflow at a glance; Each step, and where the time goes; Where machine learning helps; Where machine learning can hurt |
| [Out-of-Distribution Detection and Generalization](out-of-distribution-and-generalization.md) | What generalizing to new geometries means; Interpolation vs extrapolation; Trusting one prediction for one new design; Detecting an out-of-distribution geometry |
| [Verification, Validation and Mesh Convergence](verification-validation-and-mesh-convergence.md) | Verification vs validation; Sources of error in a simulation; Mesh convergence and the grid convergence index; Sanity checks before trusting an FEA result |

## Simulation Fundamentals: PDEs and Numerical Methods

| Page | Covers |
| --- | --- |
| [Solving a PDE Numerically](solving-pdes-numerically.md) | What a PDE problem is; The five steps of a numerical solution; A worked example: heat in a rod; Consistency, stability and convergence |
| [FEM vs FVM vs FDM](fem-vs-fvm-vs-fdm.md) | The same equation, three ways; Comparison table; Where each method is used, and why; Related methods |
| [The Weak Form and the Galerkin Method](weak-form-and-galerkin.md) | The weak form; Why FEM uses the weak form; The Galerkin method; A worked 1D example |
| [Elements and Shape Functions](elements-and-shape-functions.md) | Shape functions; Linear vs quadratic elements; Element families; Isoparametric mapping and numerical integration |
| [The Finite Element Method](finite-element-method.md) | The FEM pipeline; Element matrices; Assembly and sparsity; How sparse, and why it matters |
| [Boundary Conditions](boundary-conditions.md) | Dirichlet, Neumann and Robin conditions; Physical examples; Bolt loads and remote loads; External-aerodynamics boundary conditions |
| [Time Integration and Stability](time-integration-and-stability.md) | Explicit vs implicit time integration; The CFL condition; Stability of the explicit heat equation; Stiff equations |
| [Sparse Linear Solvers](sparse-linear-solvers.md) | Where these systems come from; Direct solvers; Iterative solvers and conjugate gradient; Condition number and preconditioning |
| [Newton-Raphson and Nonlinear Solvers](newton-raphson-and-nonlinear-solvers.md) | Where nonlinear equations come from; The Newton-Raphson iteration; Why convergence is quadratic, and why only near the solution; Load stepping, line search and arc-length methods |
| [Meshing and Mesh Quality](meshing-and-mesh-quality.md) | Structured vs unstructured meshes; Element shapes: tetrahedra, hexahedra, prisms and others; Mesh quality metrics; How mesh quality affects a simulation |
| [Automatic Differentiation](automatic-differentiation.md) | Three ways to get a derivative; Finite differences and their two errors; Forward mode; Reverse mode |

## Structural Mechanics (FEA)

| Page | Covers |
| --- | --- |
| [Stress, Strain and Linear Elasticity](stress-strain-and-linear-elasticity.md) | Stress; Strain; Young's modulus and Poisson's ratio; Hooke's law in three dimensions |
| [Von Mises Stress, Yield and Safety Factors](von-mises-stress-and-yield.md) | Why a yield criterion is needed; The von Mises stress; Why von Mises suits ductile metals; Tresca and other criteria |
| [Stress Concentrations and Singularities](stress-concentrations-and-singularities.md) | Stress concentration vs stress singularity; Stress concentration factors; Why the peak stress rises with mesh refinement; What analysts do about singularities |
| [Nonlinear Structural Analysis](nonlinear-structural-analysis.md) | Three sources of nonlinearity; How nonlinear problems are solved; Plasticity and load-history dependence; Why path dependence complicates surrogates |
| [Buckling](buckling.md) | What buckling is; Euler buckling of a column; Why linear buckling is an eigenvalue problem; How a linear buckling analysis is run |
| [Modal Analysis](modal-analysis.md) | Free vibration and the eigenvalue problem; Properties of modes; What natural frequencies and mode shapes are used for; Practical points |
| [Fatigue](fatigue.md) | What fatigue is; Stress-life: S-N curves; Mean stress; Variable loading: rainflow counting and Miner's rule |
| [Crash-Simulation Surrogates](crash-simulation-surrogates.md) | Why crash surrogates are hard; What a crash surrogate should predict; Modelling options; Temporal strategies: one-shot, time-conditional, autoregressive |
| [Public Simulation Datasets and Benchmarks](public-simulation-datasets.md) | Car aerodynamics datasets; Airfoil and wing datasets; Structural datasets; Other mesh and particle datasets |

## Fluid Dynamics (CFD)

| Page | Covers |
| --- | --- |
| [CFD Essentials](cfd-essentials.md) | The incompressible Navier-Stokes equations; The Reynolds number; Incompressible vs compressible flow; Steady vs transient simulation |
| [Turbulence Modelling](turbulence-modelling.md) | Laminar vs turbulent flow; Why turbulence is expensive to simulate; RANS vs LES vs DNS; What a turbulence model does |
| [Boundary Layers, Wall Treatment and Separation](boundary-layers-and-wall-treatment.md) | The boundary layer and y+; The law of the wall; How y+ drives near-wall mesh design; Wall functions |
| [Aerodynamic Forces and Coefficients](aerodynamic-forces-and-coefficients.md) | Where the force comes from; Drag, lift and the coefficients; Pressure drag and friction drag; Computing forces from simulation data |
| [Hard vs Soft Constraints](hard-vs-soft-constraints.md) | Soft and hard, side by side; Ways to make a constraint hard; Enforcing a divergence-free field exactly; Soft vs hard boundary conditions in PINNs |
| [Generating Simulation Training Data](generating-simulation-training-data.md) | Open-source solvers for generating data; Pipeline for 1,000 CFD simulations; What makes training data physically meaningful and sufficient |

## Heat Transfer and Multiphysics

| Page | Covers |
| --- | --- |
| [Heat Transfer Basics](heat-transfer-basics.md) | Conduction, convection and radiation; Boundary conditions for thermal problems; Conjugate heat transfer; Dimensionless numbers: Prandtl, Nusselt and Biot |
| [Multiphysics Coupling](multiphysics-coupling.md) | What coupling means; One-way vs two-way coupling; Thermo-mechanical coupling, with numbers; Partitioned vs monolithic coupling |
| [Thermal Stress and Warpage](thermal-stress-and-warpage.md) | The mechanism; The bimetallic strip: Timoshenko's formula; Where warpage matters; How designers reduce warpage |
| [Design: Thermal Surrogate](design-thermal-surrogate.md) | Requirements and clarifying questions; Physics that shapes the design; Data; Model options |

## Machine Learning Fundamentals for Physical Data

| Page | Covers |
| --- | --- |
| [Non-dimensionalization and Normalization](nondimensionalization-and-normalization.md) | Why non-dimensionalize; Normalizing fields and coordinates; MSE or relative L2 loss; Heavy-tailed targets |
| [Learning from Small Simulation Datasets](learning-from-small-simulation-datasets.md) | Why small data is the normal case; Controlling overfitting with 50 to 500 simulations; Model selection with very few simulations; A customer with 40 simulations |
| [Spectral Bias and Fourier Features](spectral-bias-and-fourier-features.md) | Spectral bias; Why it matters for physical fields; Fourier features and positional encodings; Choosing the frequency scale |
| [Choosing an Architecture](choosing-an-architecture.md) | The questions that decide the architecture; The main options; A decision guide; Worked examples |
| [Equivariance and Symmetry](equivariance-and-symmetry.md) | Invariance and equivariance; What transforms how; Which symmetries matter in engineering problems; When data augmentation is physically valid |
| [The Universal Approximation Theorem](universal-approximation.md) | What the theorem says; Versions for deep networks and for operators; What it does not guarantee; Approximation, estimation and optimization error |
| [Backpropagation and Logistic Regression from Scratch](backpropagation-and-logistic-regression.md) | Logistic regression in one page; Deriving the gradient; Backpropagation in a neural network; Practical details interviewers ask about |
| [Optimizers for Physics Models](optimizers-for-physics-models.md) | Adam in one paragraph; L-BFGS in one paragraph; Adam then L-BFGS; Learning-rate schedules and other practical choices |
| [Debugging Training on Physical Data](debugging-training-on-physical-data.md) | Step 1: can the model learn anything at all?; Step 2: is the data what you think it is?; Step 3: is the target predictable from the inputs?; Step 4: does it beat trivial baselines? |

## Physics-Informed Neural Networks (PINNs)

| Page | Covers |
| --- | --- |
| [Physics-Informed Neural Networks (PINNs)](pinns.md) | The idea in one example; The PINN loss function; Derivatives in the PDE residual; Collocation points |
| [PINN Training Failures](pinn-training-failures.md) | Why PINNs fail to train; Loss imbalance and adaptive weights; Causality in time-dependent PINNs; High-frequency and multi-scale solutions |
| [A PINN Training Recipe](pinn-training-recipe.md) | Step 1: formulate the problem well; Step 2: choose the architecture; Step 3: train; Step 4: validate |
| [Physics-Informed Neural Operator (PINO)](physics-informed-neural-operator.md) | PINN, neural operator and PINO compared; How PINO is trained; Computing the residual on an operator's output; When PINO is the right tool |

## Neural Operators

| Page | Covers |
| --- | --- |
| [Fourier Neural Operator (FNO)](fourier-neural-operator.md) | Architecture; The Fourier layer in detail; Why FNO keeps only a few Fourier modes; Limitations, Geo-FNO and GINO |
| [DeepONet](deeponet.md) | Branch and trunk networks; Why the dot product works; Variants; Limitations |
| [Transformers for Physics](transformers-for-physics.md) | Is attention a neural operator?; Why vanilla attention fails on a million points; Linear and Galerkin-type attention; Encoding positions and boundary conditions |

## Graph Neural Networks and Mesh-Based Learning

| Page | Covers |
| --- | --- |
| [The Graph Network-based Simulator (GNS)](graph-network-simulator-gns.md) | What GNS simulates; Building the graph; Encoder, processor, decoder; Integrating the prediction |
| [Multiscale GNNs: Oversmoothing, Oversquashing and Graph U-Nets](multiscale-gnns.md) | Oversmoothing and oversquashing; Why mesh refinement makes it worse; Multiscale message passing; Graph U-Net |
| [Mesh Data Processing](mesh-data-processing.md) | Data structures; Flat arrays to a mesh; Element types and the bounding box; Areas, face normals and vertex normals |
| [GNN Layer Types: GCN, GAT, GraphSAGE and GIN](gnn-layer-types.md) | The common template; GCN; GAT; GraphSAGE |
| [Scaling to Large Meshes](scaling-to-large-meshes.md) | GPU memory for a GNN on a 1-million-node mesh; Scaling a GNN to 10 million nodes; Train on subsamples, predict at full resolution; Domain decomposition and halo regions |
| [Autoregressive Rollouts](autoregressive-rollouts.md) | Message Passing Neural PDE Solvers; Why one-step models drift; Noise injection; The pushforward trick |

## 3D Geometry and Geometric Deep Learning

| Page | Covers |
| --- | --- |
| [Geometry Representations](geometry-representations.md) | Ways to represent 3D geometry; Signed distance functions; CAD models and CAD datasets; Neural fields |
| [Point-Cloud Learning](point-cloud-learning.md) | PointNet; PointNet++; DGCNN (EdgeConv) and Point Transformer; Sampling points from a surface mesh |
| [Geodesic Convolutions and Shape Optimization](geodesic-convolutions-and-shape-optimization.md) | Why measure along the surface; Geodesic CNNs and their successors; Geodesic Convolutional Shape Optimization; Surrogate-driven shape optimization in practice |

## Transformers for Physics and Large-Scale Surrogates

| Page | Covers |
| --- | --- |
| [Physics Transformer Architectures: Transolver, UPT, AB-UPT, DoMINO and GeoTransolver](physics-transformer-architectures.md) | Transolver and Physics-Attention; Universal Physics Transformers; AB-UPT: anchor tokens and divergence-free fields; DoMINO |

## Surrogate Models, Reduced-Order Models and Design of Experiments

| Page | Covers |
| --- | --- |
| [Classical Surrogates: Response Surfaces, RBF Interpolation and Kriging](classical-surrogates.md) | Polynomial response surfaces; Radial basis function interpolation; Kriging; When classical surrogates are enough |
| [Gaussian Processes](gaussian-processes.md) | Gaussian process regression; How a GP estimates its own uncertainty; Choosing a kernel and fitting hyperparameters; Why GPs scale badly, and the workarounds |
| [Reduced-Order Models](reduced-order-models.md) | Proper orthogonal decomposition; Intrusive vs non-intrusive ROMs; Dynamic mode decomposition; When a linear subspace fails |
| [SINDy and the Koopman Operator](sindy-and-koopman.md) | SINDy; The Koopman operator; SINDy, Koopman or a neural surrogate? |
| [Design of Experiments and Active Learning](design-of-experiments-and-active-learning.md) | Latin hypercube vs Sobol vs random sampling; How many simulations you need; Active learning and acquisition functions; Global sensitivity analysis and Sobol indices |
| [Multi-Fidelity Learning](multi-fidelity-learning.md) | Fidelity levels in practice; Methods; Allocating the simulation budget; When multi-fidelity fails |
| [Hamiltonian and Lagrangian Neural Networks](hamiltonian-and-lagrangian-neural-networks.md) | Why build in a conservation law; Hamiltonian neural networks; Lagrangian neural networks; Related structure-preserving models |

## Time-Dependent Problems and Autoregressive Rollouts

| Page | Covers |
| --- | --- |
| [Differentiable Physics and Hybrid Solvers](differentiable-physics-and-hybrid-solvers.md) | Hybrid patterns; Unrolled training and solver-in-the-loop; Correcting a coarse solver; Differentiable simulators |
| [Neural ODEs](neural-odes.md) | From residual networks to ODEs; Training: backpropagation through the solver and the adjoint method; Uses in scientific machine learning; Limitations |

## Uncertainty, Generalization and Out-of-Distribution Detection

| Page | Covers |
| --- | --- |
| [Uncertainty Quantification](uncertainty-quantification.md) | Aleatoric vs epistemic uncertainty; Deep ensembles; Monte Carlo dropout; Other methods |
| [Conformal Prediction](conformal-prediction.md) | The split conformal recipe; What is guaranteed and what is not; Adaptive intervals; Distribution shift and group-wise guarantees |

## Simulation Data, Datasets and Benchmarks

| Page | Covers |
| --- | --- |
| [Simulation File Formats](simulation-file-formats.md) | Geometry formats; Mesh and result formats; Solver input decks; Array formats for training data |
| [Data Splits and Leakage](data-splits-and-leakage.md) | Splitting simulation data; Three ways test information leaks into training; Small data: cross-validation done right; Checklist before reporting a number |
| [Production and Deployment](production-and-deployment.md) | The deployment pipeline; Versioning data together with models; A regression test suite for a surrogate; ONNX vs TorchScript vs torch.export |

## Evaluation and Testing

| Page | Covers |
| --- | --- |
| [Evaluation Metrics](evaluation-metrics.md) | Field metrics; Integrated quantities; Reporting a speed-up fairly; Baselines to beat |
| [LLM Agents for Simulation](llm-agents-for-simulation.md) | What agents can do today and where they fail; Why the LLM should not write solver physics; The ReAct pattern; A reference architecture |

## Design Optimization, Inverse Problems and Generative Design

| Page | Covers |
| --- | --- |
| [Surrogate-Based Optimization](surrogate-based-optimization.md) | The optimization loop; Surrogate gradients vs the adjoint; Why optimizers exploit surrogates; Multiple objectives |
| [The Adjoint Method](adjoint-method.md) | Derivation; Direct vs adjoint sensitivities; Worked example: compliance in topology optimization; Discrete vs continuous adjoint |
| [Bayesian Optimization](bayesian-optimization.md) | The algorithm; Acquisition functions; Practical choices; Extensions: noise, constraints, batches, objectives and fidelities |
| [Topology Optimization](topology-optimization.md) | SIMP; Filters, projections and crisp designs; Other methods; Making designs manufacturable |
| [Inverse Problems and Regularization](inverse-problems-and-regularization.md) | Why inverse problems are ill-posed; Regularization methods; Choosing the regularization weight; The Bayesian view |
| [Generative Models for Physics](generative-models-for-physics.md) | Why a distribution instead of an answer; Denoising diffusion models; Flow matching; Building an inverse-design tool |

## Generative Models and Foundation Models for Physics

| Page | Covers |
| --- | --- |
| [Foundation Models for Physics](foundation-models-for-physics.md) | What a PDE foundation model is and what transfers; Large geometry and large engineering models; Lessons from ML weather forecasting; Fine-tuning on a small customer dataset |

## LLM Agents for Simulation and CAD

| Page | Covers |
| --- | --- |
| [Simulation Agents: The Research](simulation-agents-research.md) | FEABench; CFD agents: Foam-Agent, MetaOpenFOAM and OpenFOAMGPT; FEA agents; Lessons across systems |
| [Text-to-CAD](text-to-cad.md) | Why CAD, not meshes; Command-sequence models: DeepCAD and Text2CAD; CAD as code: CadQuery generation; Verifying generated geometry |

## Coding and Practical Implementation

| Page | Covers |
| --- | --- |
| [Coding Round Preparation](coding-round-preparation.md) | What to practise; How to work through a problem; Graph problems on meshes; Union-find for connected parts |

## AI-for-Simulation System Design

| Page | Covers |
| --- | --- |
| [System Design Framework for AI-for-Simulation](system-design-framework.md) | The twelve steps; Building blocks that recur; Worked designs; How to run the 45 minutes |
| [Design: Car Aerodynamics Surrogate](design-car-aerodynamics-surrogate.md) | Requirements and clarifying questions; Data; Geometry processing; Model |
| [Design: Stress Prediction Platform](design-stress-prediction-platform.md) | Requirements and clarifying questions; Inputs: CAD, loads and materials; Data; Model |
| [Design: Simulation Data Pipeline](design-simulation-data-pipeline.md) | Requirements and clarifying questions; Budget arithmetic; Pipeline stages; Quality checks |
| [Design: Active Learning Loop](design-active-learning-loop.md) | Requirements and clarifying questions; The loop; Choosing the acquisition function; Batches and parallel simulations |
| [Design: Bracket Lightweighting Loop](design-bracket-lightweighting-loop.md) | Requirements and clarifying questions; Design variables and constraints; The loop; Surrogate and pessimistic constraints |
| [Design: Validation Strategy](design-validation-strategy.md) | What the sceptic needs to see; Acceptance criteria agreed in advance; Test sets that mean something; What to measure |
| [Design: Heat Exchanger Digital Twin](design-heat-exchanger-digital-twin.md) | Requirements and clarifying questions; Physics and the model hierarchy; Combining sensors and model: data assimilation; Estimating fouling |
| [Design: RANS–LES Multi-Fidelity Surrogate](design-rans-les-multi-fidelity-surrogate.md) | Requirements and clarifying questions; Why RANS and LES differ; Methods; Allocating the LES budget |
| [Design: Fine-Tuning Pipeline](design-fine-tuning-pipeline.md) | Requirements and clarifying questions; Pipeline stages; Choosing how much to fine-tune; Evaluation with 50 simulations |
| [Design: Inverse Design Tool](design-inverse-design-tool.md) | Requirements and clarifying questions; Why the inverse problem is hard; Shape parameterization; Approach 1: optimization through a forward model |
| [Design: LLM Structural Check Agent](design-llm-structural-check-agent.md) | Requirements and clarifying questions; From request to specification; Tools; The agent loop |
| [Design: Surrogate Serving System](design-surrogate-serving-system.md) | Requirements and clarifying questions; Capacity estimate; Request path; Inference servers and batching |
| [Design: Drift Monitoring](design-drift-monitoring.md) | What drift means here; Per-request OOD scores; Population drift over time; Ground-truth checks with the solver |

## Behavioral and Customer-Facing Scenarios

| Page | Covers |
| --- | --- |
| [Customer Scenarios](customer-scenarios.md) | How to answer a scenario question; I don't trust black boxes; The surrogate disagrees with wind-tunnel data; 1% accuracy from 30 simulations |
| [Behavioural Questions](behavioral-questions.md) | Preparing your stories; Research that people actually used; Finding out your own results were wrong; Presenting a technical project |
