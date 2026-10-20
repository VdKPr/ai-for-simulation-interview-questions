# The CAE Workflow, and Where ML Fits

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A computer-aided engineering (CAE) workflow turns a design question into a decision: define the question, prepare the CAD geometry, mesh it, set up the physics (materials, loads, boundary conditions, solver settings), solve, post-process the results into quantities of interest, check them (verification and validation) and report. The expensive parts are usually not the solve alone but the human time in geometry cleanup, meshing and set-up, and the waiting for long runs. Machine learning helps where the same work is repeated many times: surrogates for fast evaluation, automation of set-up, and smarter design exploration. It hurts when it is trusted outside its training data, when it hides errors that a solver would have shown, and when it breaks the traceability that sign-off needs.

**On this page**

- [The workflow at a glance](#the-workflow-at-a-glance)
- [Each step, and where the time goes](#each-step-and-where-the-time-goes)
- [Where machine learning helps](#where-machine-learning-helps)
- [Where machine learning can hurt](#where-machine-learning-can-hurt)
- [Worked example: one bracket, two workflows](#worked-example-one-bracket-two-workflows)
- [In an interview](#in-an-interview)

## The workflow at a glance

```
 design question  ──▶  CAD geometry  ──▶  geometry cleanup  ──▶  meshing
 (what decision?)       (from design)      (defeaturing,          (element type,
                                            repair, idealize)      size, quality)
                                                                       │
        ┌──────────────────────────────────────────────────────────────┘
        ▼
  physics set-up   ──▶   solve   ──▶   post-processing   ──▶   checks (V&V)   ──▶   report
  (materials, loads,     (FEM/FVM,     (fields → QoIs:          (mesh study,         and decision
   BCs, contacts,         hours to      max stress, drag,        sanity checks,
   solver settings)       days)         temperatures)            test data)
        ▲                                                                           │
        └──────────────────────── change the design and repeat ◀────────────────────┘
```

The loop at the bottom is the important part. Engineering design is iterative: a real programme runs this loop tens or hundreds of times, often with small changes between iterations. That repetition is what makes machine learning worth considering.

## Each step, and where the time goes

**1. Define the question.** What decision does the analysis support? "Does the bracket survive the 9 g load case with a safety factor of 1.5?" is a good question. "Simulate the bracket" is not. This step decides which outputs, accuracy and load cases matter.

**2. CAD geometry.** The design arrives from the CAD system: a boundary representation (B-rep) model with exact surfaces. It often contains detail that matters for manufacturing but not for the analysis: logos, small fillets, threads, tiny gaps.

**3. Geometry cleanup and idealization.** The analyst removes or simplifies features that would force a huge mesh without changing the answer (defeaturing), repairs gaps and overlaps, and may idealize parts: a thin wall becomes a shell, a long member becomes a beam, a bolt becomes a connector element. This step can take a large share of the engineer's time on complex assemblies, and it involves judgement: removing a fillet near a stress hot spot changes the result.

**4. Meshing.** The geometry is divided into elements (tetrahedra, hexahedra, prisms, shells). The analyst chooses element type and order, sizes, refinement near hot spots and boundary layers, and checks mesh quality. See [Meshing and mesh quality](meshing-and-mesh-quality.md).

**5. Physics set-up.** Materials, loads, boundary conditions, contacts between parts, turbulence model, time step, solver tolerances. Most wrong answers in CAE come from here, not from the solver: a wrong constraint, a load in the wrong units, a contact that should have been bonded. See [Boundary conditions](boundary-conditions.md).

**6. Solve.** The solver assembles and solves the discrete equations. A linear static FEA may take minutes; a scale-resolving CFD run of a car or a full-vehicle crash can take many hours to a day on a cluster.

**7. Post-processing.** Fields are turned into the numbers and pictures the decision needs: maximum von Mises stress in a region, drag and lift coefficients, peak temperatures, deformation plots.

**8. Verification and validation.** Is the number right? Check convergence, run a mesh study, compare with hand calculations and, where available, with test data. See [Verification and validation](verification-validation-and-mesh-convergence.md).

**9. Report and decide.** The result, its assumptions and its uncertainty go into a report that someone signs off. In regulated industries (aerospace, automotive safety, nuclear, medical devices) this record must be traceable.

## Where machine learning helps

| Step | How ML can help | Example |
| --- | --- | --- |
| Design question | Retrieve similar past designs and their results | Search the simulation archive by geometry similarity |
| CAD | Generative design proposals; parametric models from text | [Text-to-CAD](text-to-cad.md), [Inverse design](generative-models-for-physics.md#building-an-inverse-design-tool) |
| Cleanup | Classify features to remove; predict where detail matters | Flag small fillets that sit in high-stress regions |
| Meshing | Predict sizing fields; predict which meshes will fail quality checks | Learned refinement near expected hot spots |
| Set-up | Draft input decks, infer boundary conditions from text | [LLM agents](llm-agents-for-simulation.md), with templates and validation |
| Solve | Surrogates instead of the solve; warm starts and learned preconditioners to speed the solve | [Surrogate models](surrogate-models.md), [Newton warm starts](newton-raphson-and-nonlinear-solvers.md#speeding-up-a-solver-with-ml-without-changing-its-answer) |
| Post-processing | Turn fields into insight; detect anomalous results | Flag a result whose stress pattern looks unlike any similar design |
| V&V | Estimate uncertainty; flag out-of-distribution designs | [UQ](uncertainty-quantification.md), [OOD detection](out-of-distribution-and-generalization.md) |
| The design loop | Explore and optimize with surrogates; choose the next simulation to run | [Surrogate-based optimization](surrogate-based-optimization.md), [Active learning](design-of-experiments-and-active-learning.md#active-learning-and-acquisition-functions) |

The biggest wins usually come from the **loop**, not from one step: a surrogate that gives feedback in seconds lets designers try many more ideas while the design is still flexible, and the expensive solver is used only to confirm the best ones.

## Where machine learning can hurt

- **Silent errors.** A solver that is set up wrongly often fails loudly. A surrogate returns a smooth, plausible field for any input, including inputs it has never seen. Without out-of-distribution checks, a wrong answer looks exactly like a right one.
- **Extrapolation pressure.** Optimizers and creative designers both push towards new regions of design space, which is where the surrogate has no data. See [Why optimizers exploit surrogates](surrogate-based-optimization.md#why-optimizers-exploit-surrogates).
- **Loss of traceability.** Sign-off needs to show which model, which inputs and which assumptions produced a number. A model without versioning of data, weights and code cannot provide that. See [Production and deployment](production-and-deployment.md).
- **Automation bias.** People tend to accept a confident tool's answer. If engineers stop doing sanity checks because "the AI did it", errors propagate.
- **Wrong set-up from an LLM.** An agent that writes an input deck may choose the wrong boundary condition or unit and still produce a run that converges. That is why agents should work from validated templates, with the physics kept in the solver. See [Why the LLM should not write solver physics](llm-agents-for-simulation.md#why-the-llm-should-not-write-solver-physics).
- **Inherited bias in the training data.** If the training simulations used linear tetrahedra that are too stiff, or a turbulence model that misses separation, the surrogate learns those errors. See [Why linear tetrahedra are too stiff](elements-and-shape-functions.md#why-linear-tetrahedra-are-too-stiff-in-bending).
- **Data and IP risk.** CAD and simulation data are valuable intellectual property; sending them to a cloud service may not be allowed. See [When customers will not upload their CAD](production-and-deployment.md#when-customers-will-not-upload-their-cad).

## Worked example: one bracket, two workflows

**Traditional.** An engineer receives a new bracket design. They clean the CAD, mesh it with quadratic tetrahedra, apply the bolt preloads and the 9 g load case, run a linear static analysis (say 20 minutes), check the mesh with one refinement, and report the maximum von Mises stress against the allowable. A design change means repeating most of these steps. Ten design iterations take days.

**With ML in the loop.** The company has 400 past bracket analyses for this load case. A surrogate trained on them predicts the stress field of a new design in about a second, with an uncertainty estimate and a check of how similar the design is to the training set. The designer tries 50 variants in an afternoon. The three best go through the full traditional workflow for confirmation and sign-off. Any variant the surrogate flags as unfamiliar goes straight to the solver.

The surrogate did not replace the analysis. It replaced most of the *iterations* that never needed sign-off quality in the first place.

## In an interview

**Thirty-second answer.** "CAD comes in, gets cleaned and idealized, meshed, set up with materials, loads and boundary conditions, solved, post-processed into the numbers the decision needs, checked with mesh studies and hand calculations, and reported. Most of the time goes into cleanup, meshing, set-up and waiting for runs, and the whole loop repeats many times during a design programme. ML helps by making the loop faster: surrogates for quick evaluation, automation of repetitive set-up, smarter choice of what to simulate next. It hurts when it is trusted outside its data, when it hides errors, or when it breaks the traceability that sign-off needs, so I would keep the solver for confirmation and route uncertain cases to it."

**Likely follow-ups**

- *Which step would you automate first?* Usually the one that is repeated most and is most rule-based, for example set-up of a standard load case from a template, or a surrogate for the most common analysis in the design loop.
- *How do you make engineers trust it?* Show validation on their own past cases, show where it fails, give an uncertainty estimate and an out-of-distribution flag, and keep the solver as the final check. See [Customer scenarios](customer-scenarios.md#i-dont-trust-black-boxes).

## References

- [Learning Mesh-Based Simulation with Graph Networks (MeshGraphNets, Pfaff et al.)](https://arxiv.org/abs/2010.03409)
- [FEABench: Evaluating Language Models on Multiphysics Reasoning Ability (Mudur et al.)](https://arxiv.org/abs/2504.06260)
- [Explaining SimAI (Ansys blog)](https://ansys.synopsys.com/blog/explaining-simai)
- [DrivAerML: an open dataset generated with an automated, validated CFD workflow (Ashton et al.)](https://arxiv.org/abs/2408.11969)
