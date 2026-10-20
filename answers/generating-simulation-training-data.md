# Generating Simulation Training Data

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Most AI-for-simulation projects live or die on their training data, and the data has to be *made*: by running hundreds to thousands of simulations through an automated pipeline. The pipeline defines a design space, samples it with a design of experiments, generates and checks geometry, meshes it with a fixed strategy, runs a templated solver set-up on a cluster or in the cloud, checks convergence and physics for every run, extracts fields and integrated quantities into ML-friendly formats with full metadata, and versions the result. Open-source tools cover every step: **OpenFOAM** and **SU2** for CFD; **FEniCSx**, **CalculiX**, **Code_Aster**, **MOOSE** and others for FEA and multiphysics; **Gmsh** and **CadQuery** for meshing and CAD. Data is **physically meaningful** when the physics model fits the use, the numerics are verified, the set-up is consistent across samples and the runs are quality-checked; it is **sufficient** when learning curves and held-out tests show the model meets the accuracy the decision needs across the whole design space, including its difficult regions.

**On this page**

- [Open-source solvers for generating data](#open-source-solvers-for-generating-data)
- [Pipeline for 1,000 CFD simulations](#pipeline-for-1000-cfd-simulations)
- [What makes training data physically meaningful and sufficient](#what-makes-training-data-physically-meaningful-and-sufficient)
- [Code: a learning curve and the label-noise floor](#code-a-learning-curve-and-the-label-noise-floor)
- [In an interview](#in-an-interview)

## Open-source solvers for generating data

| Purpose | Tools | Notes |
| --- | --- | --- |
| General CFD (finite volume) | [OpenFOAM](https://github.com/OpenFOAM/OpenFOAM-dev) | The most used open-source CFD toolbox; incompressible and compressible solvers, RANS, LES, multiphase, conjugate heat transfer; two main distributions (openfoam.org and openfoam.com). AhmedML was generated with it, and LLM agents such as Foam-Agent target it |
| Aerodynamics with adjoints | [SU2](https://github.com/su2code/SU2) | Compressible and incompressible flow, built-in adjoint solvers for shape optimization |
| Research FEM frameworks | [FEniCSx / DOLFINx](https://github.com/FEniCS/dolfinx), [deal.II](https://github.com/dealii/dealii), [MFEM](https://github.com/mfem/mfem), [Firedrake](https://github.com/firedrakeproject/firedrake) | Write the weak form, the framework does the rest; ideal for custom physics and for generating clean datasets |
| Structural analysis | CalculiX, Code_Aster | Abaqus-like input decks (CalculiX); a large, industrial-strength structural code from EDF (Code_Aster) |
| Multiphysics | MOOSE, Elmer | MOOSE has an LLM agent study (MooseAgent); Elmer covers thermal, structural, electromagnetic and fluid problems |
| Explicit dynamics, crash | OpenRadioss | Open-source release of the Radioss explicit solver |
| High-order and GPU CFD | NekRS, PyFR | Spectral-element and flux-reconstruction codes for scale-resolving simulations |
| Differentiable solvers | JAX-Fluids, PhiFlow, APEBench's solver | For hybrid models and gradient-based training through physics ([Differentiable simulators](differentiable-physics-and-hybrid-solvers.md#differentiable-simulators)) |
| Geometry and meshing | [Gmsh](https://gmsh.info/doc/texinfo/gmsh.html), [CadQuery](https://github.com/CadQuery/cadquery), snappyHexMesh (in OpenFOAM) | Scriptable CAD and meshing, essential for automation |
| Coupling | [preCICE](https://github.com/precice/precice) | Partitioned multiphysics coupling between solvers |
| Input/output and visualization | [meshio](https://github.com/nschloe/meshio), [PyVista](https://github.com/pyvista/pyvista), ParaView | Convert formats, extract fields, check results ([Simulation file formats](simulation-file-formats.md)) |

**Choosing.** Match the solver to what the customer uses (a surrogate trained on OpenFOAM RANS will be compared with their Fluent or STAR-CCM+ runs), automate everything from day one, and validate the set-up on a known case before generating data at scale.

## Pipeline for 1,000 CFD simulations

Suppose the task is a drag and surface-pressure surrogate for a family of car shapes. The pipeline:

```
design space ─▶ DoE sample ─▶ geometry ─▶ mesh ─▶ solver set-up ─▶ run (HPC / cloud) ─▶ QA ─▶ post-process ─▶ dataset + metadata ─▶ version
     ▲                                                                                                                         │
     └────────────────────────────── active learning: add runs where the model is uncertain ◀──────────────────────────────────┘
```

**1. Define the design space and the outputs.** Parameters and ranges (or a family of shapes and how they vary), operating conditions (speed, yaw, ride height), and exactly which outputs the model needs (surface pressure and wall shear, a subsampled volume field, force coefficients). Decide now what the held-out test sets will be: a held-out family, an extrapolation range.

**2. Sample with a design of experiments.** A Latin hypercube or Sobol sequence over the parameters, typically with a few percent of extra samples for re-runs and failures. See [Latin hypercube vs Sobol vs random sampling](design-of-experiments-and-active-learning.md#latin-hypercube-vs-sobol-vs-random-sampling).

**3. Generate and check the geometry.** Parametric CAD (CadQuery, FreeCAD) or mesh morphing (for example radial-basis-function morphing of a base shape). Check every geometry: watertight, no self-intersections, sensible volume and frontal area, manufacturable if that matters.

**4. Mesh with one fixed strategy.** The same meshing script and settings for every design: surface refinement, prism layers sized for the target y+ ([How y+ drives mesh design](boundary-layers-and-wall-treatment.md#how-y-drives-near-wall-mesh-design)), refinement boxes for the wake. Run a **mesh-independence study** on a few representative designs before the campaign, and record mesh statistics (cell count, quality metrics) for every sample.

**5. Set up the solver from templates.** Boundary conditions, turbulence model, numerical schemes and convergence criteria in a version-controlled template, with only the design-dependent values substituted. Consistency across samples matters more than perfection: if half the runs used a different turbulence model, the dataset contains two different physics.

**6. Run at scale.** Containerize the solver (Docker or Apptainer) so every run uses the identical version. Submit job arrays to an HPC scheduler (such as Slurm) or a cloud batch system (or Kubernetes). Budget explicitly: for example, 1,000 runs × 4 hours × 128 cores = 512,000 core-hours; at an illustrative 2 to 5 cents per core-hour that is roughly 10,000 to 25,000 dollars of compute, before storage.

**7. Quality assurance for every run.**
- Convergence: residuals *and* the force history have flattened (or, for transient runs, statistics are converged over enough averaging time).
- Physics: mass balance, a sensible y+ distribution, no negative cells or extreme values.
- Failures: log every failed run with its reason; retry with documented adjustments; **do not silently drop failures**, because they often cluster in physically interesting regions (strong separation) and dropping them biases the dataset. See [Data splits and leakage](data-splits-and-leakage.md).

**8. Post-process into ML formats.** Extract surface fields (pressure, wall shear) at a consistent location (node or cell), a subsampled volume field if needed, and integrated forces computed by the same script for every run. Store arrays in HDF5 or Zarr (or keep VTK/VTU for interoperability), plus the geometry as STL. See [Simulation file formats](simulation-file-formats.md).

**9. Store metadata with every sample:** design parameters, operating conditions, mesh statistics, solver and template version, convergence metrics, run time, compute cost, and a content hash. This is what makes the data trustworthy and reproducible.

**10. Version and document.** Treat the dataset like code: versioned (for example with DVC or lakeFS on object storage), with a data card describing the design space, physics, known limitations and licence. See [Versioning data together with models](production-and-deployment.md#versioning-data-together-with-models).

**11. Validate the data itself.** Compare a handful of cases with experimental data or higher-fidelity simulations, and estimate the label noise floor (mesh uncertainty, averaging noise). See [When the labels have discretization error](verification-validation-and-mesh-convergence.md#when-the-labels-have-discretization-error).

**12. Iterate with active learning.** After a first model is trained, spend part of the budget on runs where the model is most uncertain or errors matter most. See [Design: an active learning loop](design-active-learning-loop.md).

A pipeline built this way also serves the system design question of generating 10,000 simulations. See [Design: the simulation data pipeline](design-simulation-data-pipeline.md).

## What makes training data physically meaningful and sufficient

**Physically meaningful** means the labels represent the physics the model will be asked about:

1. **The right physical model for the use.** Turbulence model, steady or transient, compressibility, material model, contact, boundary conditions that match the real operating conditions. A dataset of steady RANS runs is meaningful for attached-flow drag trends, much less so for unsteady wake loads.
2. **Verified numerics.** Mesh-independent quantities of interest, converged iterations, adequate time steps and averaging windows, all checked on representative cases.
3. **Validated where possible.** At least the base configuration compared with experiments or a trusted higher-fidelity result.
4. **Consistent set-up.** Variation between samples should come from the design changes, not from different meshing, solver versions or settings. Otherwise the model learns noise from set-up differences.
5. **Quality-controlled.** No hidden failed or unconverged runs; conservation and sanity checks on every sample; failures recorded rather than discarded.
6. **Representative of deployment.** The same regime (Reynolds number, load range), the same kind of geometry and the same output definitions as the cases the customer will submit.
7. **Documented.** Metadata and a data card so that anyone can tell what the data does and does not represent.

**Sufficient** means the model trained on it meets the accuracy the decision needs, everywhere it will be used:

1. **Coverage of the design space,** including corners and the regions where the physics changes (separation onset, buckling, contact), not only the comfortable middle.
2. **A learning curve that has flattened** at an acceptable error: train on 25%, 50% and 100% of the data; if the error is still falling steeply, more data will help; if it has flattened at an unacceptable level, the model or the inputs need to change, or the label noise floor has been reached (see the code below).
3. **Held-out tests that match the claim:** held-out families or ranges if the claim is generalization. See [What generalizing to new geometries means](out-of-distribution-and-generalization.md#what-generalizing-to-new-geometries-means).
4. **Accuracy at the decision level:** for example the ranking of designs, or the false-pass rate of a safety check, not only an average field error. See [When the surrogate is used for ranking](evaluation-metrics.md#when-the-surrogate-is-used-for-ranking).
5. **Enough examples of the rare but important cases** for the error there to be measured, not guessed.

**How many simulations?** There is no universal number. A smooth response in a handful of parameters may need tens to a few hundred runs; free-form geometry with complex flow needs hundreds to thousands, and benefits from pretrained models. See [How many simulations you need](design-of-experiments-and-active-learning.md#how-many-simulations-you-need) and [A customer with 40 simulations](learning-from-small-simulation-datasets.md#a-customer-with-40-simulations).

## Code: a learning curve and the label-noise floor

A stand-in "simulator" with three design parameters and a small amount of label noise (playing the role of mesh and averaging error), fitted with kernel ridge regression (the mean of a Gaussian process with fixed hyperparameters) on more and more samples.

```python
import numpy as np

rng = np.random.default_rng(0)

def simulator(X, noise=0.01):
    """Stand-in for a solver: a smooth response of 3 design parameters plus label noise
    (think of mesh and averaging error in the 'ground truth')."""
    y = np.sin(3 * X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.3 * np.cos(4 * X[:, 2]) * X[:, 0]
    return y + noise * rng.standard_normal(len(X))

def kernel_ridge_fit_predict(X, y, Xs, length=0.4, reg=1e-4):
    """RBF kernel ridge regression (the mean of a GP with fixed hyperparameters)."""
    d2 = lambda A, B: ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    K = np.exp(-d2(X, X) / (2 * length**2))
    alpha = np.linalg.solve(K + reg * np.eye(len(X)), y - y.mean())
    return y.mean() + np.exp(-d2(Xs, X) / (2 * length**2)) @ alpha

X_test = rng.random((2000, 3))
y_test = simulator(X_test)
curve = {}
print(" N train | test RMSE")
for n in [10, 20, 40, 80, 160, 320, 640]:
    X = rng.random((n, 3))
    curve[n] = np.sqrt(np.mean((kernel_ridge_fit_predict(X, simulator(X), X_test) - y_test) ** 2))
    print(f"{n:8d} | {curve[n]:.4f}")
print("label noise floor (std of noise in the test labels): 0.0100")
```

Output:

```
 N train | test RMSE
      10 | 0.2935
      20 | 0.1352
      40 | 0.0704
      80 | 0.0453
     160 | 0.0179
     320 | 0.0159
     640 | 0.0114
label noise floor (std of noise in the test labels): 0.0100
```

The error falls quickly at first and then flattens towards the noise in the labels themselves: beyond about 160 samples, quadrupling the data improves the RMSE only from 0.018 to 0.011, and it can never go below about 0.010, because the test labels themselves are that noisy. In a real project, the floor comes from mesh and averaging uncertainty, which is why it should be estimated before deciding to buy more simulations.

## In an interview

**Thirty-second answer to "How would you generate 1,000 CFD simulations for training?"** "Define the design space, outputs and held-out sets first; sample with a Latin hypercube; generate and validity-check each geometry; mesh with one fixed, mesh-independence-checked strategy; run a version-controlled solver template in containers on a cluster or cloud batch; check convergence of the forces, mass balance and y+ for every run and record failures instead of dropping them; extract surface and volume fields and forces with the same scripts into HDF5 or Zarr with full metadata; version the dataset; validate a few cases against experiments; then use active learning to spend the remaining budget where the model is uncertain."

**Likely follow-ups**

- *Which open-source solvers would you use?* OpenFOAM or SU2 for CFD, FEniCSx or CalculiX for FEA, Gmsh and CadQuery for geometry and meshing, all scriptable.
- *Why not drop failed runs?* They often cluster in physically interesting regions, so dropping them biases the data and hides where the physics is hard.
- *How do you know you have enough data?* Learning curves that flatten at an acceptable error, held-out family tests, and decision-level metrics; and compare with the label noise floor.

## References

- [OpenFOAM (Foundation development repository)](https://github.com/OpenFOAM/OpenFOAM-dev) and [SU2](https://github.com/su2code/SU2)
- [AhmedML: an open dataset generated with OpenFOAM, with the case set-up included (Ashton et al.)](https://arxiv.org/abs/2407.20801)
- [DrivAerML: automated, validated workflows for 500 car variants (Ashton et al.)](https://arxiv.org/abs/2408.11969)
- [Foam-Agent: LLM agents that automate OpenFOAM workflows (Yue et al.)](https://arxiv.org/abs/2505.04997)
- [Learning resources: simulation and geometry tools](../resources.md#simulation-and-geometry-tools-for-generating-your-own-data)
