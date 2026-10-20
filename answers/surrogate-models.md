# Surrogate Models

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A surrogate model is a fast approximation of an expensive simulation. It is trained on the results of simulations that have already been run, and then it predicts the result for a new design in milliseconds or seconds instead of hours. Engineering companies want surrogates because design work needs the *same kind* of simulation to be run again and again: for design exploration, optimization, uncertainty studies and interactive tools. The cost of generating the training data is paid once and then spread over thousands of cheap predictions.

**On this page**

- [What a surrogate is](#what-a-surrogate-is)
- [What goes in and what comes out](#what-goes-in-and-what-comes-out)
- [Types of surrogate](#types-of-surrogate)
- [Why engineering companies want surrogates](#why-engineering-companies-want-surrogates)
- [Scalar vs field surrogates](#scalar-vs-field-surrogates)
- [When a surrogate pays off](#when-a-surrogate-pays-off)
- [What makes surrogates hard in practice](#what-makes-surrogates-hard-in-practice)
- [How a surrogate is built, step by step](#how-a-surrogate-is-built-step-by-step)
- [In an interview](#in-an-interview)

## What a surrogate is

A simulation is a function. You give it a design and operating conditions, it solves the governing equations (with FEM, FVM or another method) and it returns results. The problem is that this function is slow: one high-fidelity CFD run of a car can take many hours on hundreds of CPU cores, and one nonlinear crash simulation can take a day.

A surrogate learns an approximation of that function from examples:

```
        expensive simulation (hours)
design ─────────────────────────────▶ results
   │                                    ▲
   │      surrogate (milliseconds)      │
   └────────────────────────────────────┘
          trained on (design, result) pairs
          from simulations already run
```

**A small example.** A bracket is described by five numbers: two hole positions, a wall thickness, a fillet radius and a rib height. An engineer runs 200 finite element analyses with different values of these five numbers and records the maximum von Mises stress of each design. A Gaussian process or a small neural network is then fitted to these 200 pairs. It can now estimate the maximum stress of a new combination of the five numbers in a fraction of a millisecond, so an optimizer can try 100,000 combinations in a few seconds. The engineer then runs the real FEA on the few best candidates to confirm them.

That last step matters. In most serious use, a surrogate *screens* and the solver *confirms*.

Other names for the same idea: *metamodel*, *response surface*, *emulator*, *reduced model* and, more recently, *neural surrogate* or *physics AI model*.

## What goes in and what comes out

**Inputs**

| Input | Example | How it is usually encoded |
| --- | --- | --- |
| Design parameters | lengths, angles, radii | A vector of numbers |
| Full geometry | a car body, a bracket from CAD | Surface mesh, point cloud, signed distance field, graph ([Geometry representations](geometry-representations.md)) |
| Boundary conditions | inlet velocity, applied load, fixed faces | Scalars, or features on the boundary nodes ([Boundary conditions](boundary-conditions.md)) |
| Material properties | Young's modulus, viscosity, conductivity | Scalars, or a field if the material varies |
| Operating conditions | Reynolds number, temperature, angle of attack | Scalars, often non-dimensionalized ([Normalization](nondimensionalization-and-normalization.md)) |

**Outputs**

- **Scalar quantities of interest (QoIs):** drag coefficient, lift, maximum stress, pressure drop, peak temperature, first natural frequency.
- **Fields:** pressure and wall shear stress on every surface point, velocity in the volume, displacement and stress at every node.
- **Time histories:** the deformation of a crash structure over 100 milliseconds, the temperature of a battery cell over a drive cycle.

## Types of surrogate

| Family | Example | Good when |
| --- | --- | --- |
| Classical regression | Polynomial response surface, radial basis functions | Few inputs (under about 10), smooth response, tens of samples ([Classical surrogates](classical-surrogates.md)) |
| Gaussian process | Kriging from 6 parameters to drag | Few inputs, under a few thousand samples, uncertainty needed ([Gaussian processes](gaussian-processes.md)) |
| Reduced-order model | POD modes plus regression of the mode coefficients | Fields on one fixed mesh, smooth variation ([Reduced-order models](reduced-order-models.md)) |
| Neural network on a fixed grid | CNN or U-Net on voxels or images | Geometry can be rasterized without losing the important details |
| Neural network on arbitrary geometry | GNN, point-cloud network, neural operator, transformer | Geometry changes freely, for example new car shapes ([Choosing an architecture](choosing-an-architecture.md)) |
| Physics-informed | PINN, physics loss terms | Very little data, known equations, inverse problems ([PINNs](pinns.md)) |
| Hybrid | ML correction inside a solver, ML warm start for Newton | The solver's guarantees must be kept ([Hybrid solvers](differentiable-physics-and-hybrid-solvers.md)) |

A useful way to remember the split: classical methods and GPs work on a **handful of numbers**, deep learning is needed when the input is **a whole geometry**.

## Why engineering companies want surrogates

1. **Many-query work.** Design-space exploration, optimization, robustness and uncertainty studies need hundreds to hundreds of thousands of evaluations. With a full solver at several hours per run, that is impossible inside a design programme. With a surrogate it is routine.
2. **Earlier physics feedback.** If aerodynamic feedback arrives weeks after a styling change, the shape is already frozen. If it arrives in seconds, designers can use it while they still have freedom to change things.
3. **Interactive tools and digital twins.** A designer moving a slider, or a monitoring system that must update every second, cannot wait for a solver.
4. **Reuse of past simulation data.** Companies have years of simulation results on disk. A surrogate turns that archive into a tool.
5. **Fewer solver licences and less compute.** Simulation software and HPC time are expensive. Screening with a surrogate means fewer full runs.

## Scalar vs field surrogates

This is one of the first design choices, and interviewers like it because both answers can be right.

**A scalar surrogate** predicts one or a few numbers directly: drag coefficient, maximum stress, mass flow.

- Needs less data, because there is less to learn.
- Easy to fit with a Gaussian process, so you get uncertainty almost for free.
- The training target is exactly the number the decision depends on.
- But it tells the engineer nothing about *why* or *where*. A drag number does not show where the flow separates.

**A field surrogate** predicts the full field (pressure at every surface point, stress at every node), and the scalars are computed from it afterwards.

- Engineers can see hot spots, separation lines and load paths, so they can act on the result.
- One field gives many QoIs: drag, lift, moments and local loads all come from the same surface pressure and shear.
- Each simulation gives far more training signal: a million nodes of pressure instead of one drag number. This is why field models can learn from a few hundred simulations at all.
- But it needs a model that handles large, variable meshes, and small pointwise errors can add up when integrated. For example, drag on a car is the small difference between large pressure forces on the front and the rear. A 1% pressure error spread unevenly can become a much larger drag error, or errors can cancel and hide a poor field.

**What strong candidates say:**

- If the decision depends on one number and data is scarce, start with a scalar model (often a GP). It is the baseline anyway.
- If engineers need to see the flow or the stress, or many QoIs are needed, predict the field.
- In either case, **evaluate the integrated quantity separately**. A low field error does not guarantee an accurate drag, and the reverse is also true. See [Evaluation metrics](evaluation-metrics.md#integrated-quantities).
- A common compromise is a field model with an extra output head for the key scalars, trained with both losses.

## When a surrogate pays off

A surrogate is an investment. Its total cost is:

```
total cost = data generation (N runs × cost per run)
           + training and tuning
           + validation runs
           + maintenance (retraining, monitoring)
```

It only pays off if enough future queries replace real solver runs. As a rough example, with 500 training simulations and training compute equal to 50 simulations, you need more than about 550 future queries before the surrogate is cheaper than running the solver each time. One-off analyses, sign-off and certification calculations, and designs outside the training range still need the solver. The full argument is in [ML vs numerical solvers](ml-vs-numerical-solvers.md).

## What makes surrogates hard in practice

- **Small data.** Industrial projects often have tens to a few hundred high-fidelity simulations, not millions of samples. See [Learning from small simulation datasets](learning-from-small-simulation-datasets.md).
- **Geometry variability.** Different topologies, mesh sizes and mesh quality from one design to the next.
- **Extrapolation.** An optimizer pushes designs towards the edge of the training data, which is exactly where the surrogate is least reliable. See [Surrogate-based optimization](surrogate-based-optimization.md#why-optimizers-exploit-surrogates).
- **Non-smooth physics.** Flow separation, buckling, contact, plasticity and shocks give steep or discontinuous responses that smooth models blur.
- **Path dependence.** In plasticity, the result depends on the load history, not just the final load. See [Nonlinear structural analysis](nonlinear-structural-analysis.md).
- **Trust.** Engineers need an error bar and a clear signal for when *not* to trust a prediction. See [Uncertainty quantification](uncertainty-quantification.md) and [Out-of-distribution detection](out-of-distribution-and-generalization.md).
- **Label noise.** The "ground truth" is itself a simulation with discretization and modelling error. See [Verification and validation](verification-validation-and-mesh-convergence.md#when-the-labels-have-discretization-error).

## How a surrogate is built, step by step

1. **Define the use.** Which decision will the surrogate support, which outputs matter and what accuracy is needed for that decision.
2. **Define the design space.** Parameter ranges, or the family of geometries the model must cover.
3. **Plan the simulations.** A space-filling design such as a Latin hypercube. See [Design of experiments](design-of-experiments-and-active-learning.md).
4. **Run and check the simulations.** Automated meshing, solving and quality checks: convergence, mesh independence, no failed runs hidden in the data. See [Generating simulation training data](generating-simulation-training-data.md).
5. **Preprocess.** Non-dimensionalize, normalize with training statistics only, choose a geometry encoding.
6. **Split honestly.** Hold out whole designs or design families, not random points. See [Data splits and leakage](data-splits-and-leakage.md).
7. **Start simple.** A GP or a linear model on a few parameters is the baseline any deep model must beat.
8. **Train and evaluate** on the field error and on the QoIs the engineer cares about.
9. **Add uncertainty and out-of-distribution checks** so the tool knows when to send a case back to the solver.
10. **Deploy and monitor**, and retrain when new designs drift away from the training data. See [Production and deployment](production-and-deployment.md).

## In an interview

**Thirty-second answer.** "A surrogate is a fast model trained on simulation results that predicts the result for a new design in seconds. Companies want them because design work is many-query: exploration, optimization and robustness studies need thousands of evaluations, and the solver costs hours each. The catch is that a surrogate is only reliable near its training data, so in practice it screens designs, reports its uncertainty, and the solver confirms the final candidates."

**Likely follow-ups**

- *How many simulations do you need?* It depends on the number of inputs, how smooth the response is and the accuracy needed. Measure it with a learning curve: train on 25%, 50% and 100% of the data and see how the test error falls. See [How many simulations you need](design-of-experiments-and-active-learning.md#how-many-simulations-you-need).
- *Scalar or field?* See [above](#scalar-vs-field-surrogates).
- *How do you know a prediction can be trusted?* See [Trusting one prediction](out-of-distribution-and-generalization.md#trusting-one-prediction-for-one-new-design).
- *What is the first model you would try?* The simplest one that could work: a GP or gradient-boosted trees on the design parameters if they exist. It gives a baseline and a feel for how hard the problem is.

**Mistakes to avoid:** promising the surrogate "replaces simulation", quoting a speed-up without the cost of the training data, and reporting only an average error.

## References

- [Physics Informed Machine Learning: High Level Overview of AI and ML in Science and Engineering (Steve Brunton, video)](https://www.youtube.com/watch?v=JoFW2uSd3Uo)
- [Data-Driven Science and Engineering (Brunton & Kutz, free book site)](http://databookuw.com/)
- [ETH Zürich: AI in the Sciences and Engineering (course page)](https://www.camlab.ethz.ch/teaching/ai-in-the-sciences-and-engineering-2024.html)
- [Machine Learning for Fluid Mechanics (Brunton, Noack & Koumoutsakos)](https://arxiv.org/abs/1905.11075)
