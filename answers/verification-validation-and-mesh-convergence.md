# Verification, Validation and Mesh Convergence

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** **Verification** asks whether the equations are being solved correctly: no bugs, small enough discretization and iteration error. **Validation** asks whether they are the right equations for the purpose: does the model agree with reality (experiments, tests, field data) for the intended use? The usual shorthand is "solving the equations right" versus "solving the right equations". A mesh convergence study is the main tool for estimating discretization error, and the grid convergence index (GCI) turns it into an error band. For machine learning this matters directly: the simulation labels a surrogate learns from have their own error, so there is a floor below which a surrogate's claimed accuracy means nothing.

**On this page**

- [Verification vs validation](#verification-vs-validation)
- [Sources of error in a simulation](#sources-of-error-in-a-simulation)
- [Mesh convergence and the grid convergence index](#mesh-convergence-and-the-grid-convergence-index)
- [Code: GCI from three meshes](#code-gci-from-three-meshes)
- [Sanity checks before trusting an FEA result](#sanity-checks-before-trusting-an-fea-result)
- [Sanity checks for CFD](#sanity-checks-for-cfd)
- [When the labels have discretization error](#when-the-labels-have-discretization-error)
- [In an interview](#in-an-interview)

## Verification vs validation

Think of a simulation as a chain: **reality → mathematical model → discretized model → computer code → numbers.** Each link can introduce error.

- **Verification** checks the links from the mathematical model to the numbers. It is a *mathematics and software* activity.
  - **Code verification:** is the software free of bugs that affect the solution? The standard tools are analytical benchmark solutions and the **method of manufactured solutions** (MMS): choose a smooth solution, substitute it into the PDE to find the source term and boundary conditions that make it exact, run the code, and check that the error falls at the theoretical rate as the mesh is refined. If a second-order method converges at first order, something is wrong.
  - **Solution verification:** for this particular run, how large are the discretization and iterative errors? The main tool is a mesh (and time-step) convergence study.
- **Validation** checks the link from reality to the mathematical model. It is a *physics* activity: compare simulation results with carefully designed experiments, for the quantities and conditions that matter, with uncertainties on both sides.

Two important points:

1. **Validation is for an intended use.** A RANS model may be validated for drag on attached flows and still be poor for a separated wake. "The model is validated" without saying for what is meaningless.
2. **Verification comes first.** If the numerical error is unknown, a good match with experiment might be luck: a discretization error cancelling a modelling error.

**For an ML surrogate**, the same two ideas apply one level up:

- *Verification-like:* does the surrogate reproduce the **solver** on held-out cases? This is what test-set errors measure.
- *Validation:* does the surrogate plus solver chain agree with **reality** for the intended use? This needs experiments or field data and is rarely shown in ML papers.

## Sources of error in a simulation

| Source | Example | How it is estimated or reduced |
| --- | --- | --- |
| Modelling error | Turbulence model, linear elasticity where there is plasticity, ignored contact or friction | Validation against experiments; comparison of models; higher-fidelity runs |
| Discretization error | Finite mesh size and time step | Mesh and time-step convergence studies, GCI |
| Iterative error | The solver stopped at a residual tolerance before full convergence | Tighter tolerance; monitor the quantities of interest, not only the residuals |
| Input uncertainty | Material properties, loads, boundary conditions, manufacturing tolerances on geometry | Sensitivity analysis and uncertainty propagation ([UQ](uncertainty-quantification.md)) |
| Round-off error | Finite-precision arithmetic, ill-conditioned systems | Usually small; check the conditioning; use double precision |
| User error | Wrong units, wrong constraint, wrong material, load on the wrong face | Checklists, sanity checks, peer review |
| Post-processing error | Averaged vs unaveraged stresses, interpolation to a probe point, a force integrated on the wrong surface | Consistent extraction scripts; check against raw results |
| Software bugs | An error in an element formulation | Code verification, benchmark suites |

In industrial practice, **user error and modelling error** usually dominate. Discretization error is the one that is easiest to quantify, which is why it gets the most attention.

## Mesh convergence and the grid convergence index

**The idea.** As the mesh is refined, the discretization error of a well-behaved quantity falls like `C · h^p`, where h is a representative element size and p is the order of accuracy of the method. Run the same case on three systematically refined meshes, measure how the quantity of interest changes, and you can estimate both p and the remaining error.

**The procedure** (Roache's GCI, as described in NASA's guidance):

1. Run the case on three meshes with a constant refinement ratio r = h_coarse / h_fine (for example r = 2, or r ≥ 1.3 for 3D meshes where doubling is too expensive). Refine systematically everywhere, not only in one region.
2. Track the **quantities of interest** (drag, peak temperature, stress at a point away from singularities), not only the residuals.
3. Let f₁ be the result on the finest mesh, f₂ the medium and f₃ the coarsest. The observed order of convergence is

```math
p = \frac{\ln\left(\dfrac{f_3 - f_2}{f_2 - f_1}\right)}{\ln r}
```

4. Richardson extrapolation estimates the value on an infinitely fine mesh:

```math
f_{\text{exact}} \approx f_1 + \frac{f_1 - f_2}{r^p - 1}
```

5. The grid convergence index of the fine mesh, with ε = (f₂ − f₁)/f₁ the relative change between the two finest meshes, is

```math
\text{GCI}_{12} = \frac{F_s\,|\varepsilon|}{r^p - 1}
```

   NASA's guidance uses a safety factor F_s = 1.25 when three or more meshes are used and 3.0 when only two are available.

6. **Check that the meshes are in the asymptotic range:** GCI₂₃ ≈ r^p · GCI₁₂. If this does not hold, the coarse mesh is too coarse for the error to behave like `C · h^p`, and the estimate is not reliable.

**Worked example.** A drag coefficient on three meshes with r = 2: coarse 0.3200, medium 0.3080, fine 0.3050.

- Differences: f₃ − f₂ = 0.0120 and f₂ − f₁ = 0.0030, so p = ln(4) / ln(2) = **2.0**, as expected for a second-order method.
- Extrapolated value: 0.3050 + (0.3050 − 0.3080)/(2² − 1) = **0.3040**.
- ε = 0.0030 / 0.3050 = 0.98%, so GCI₁₂ = 1.25 × 0.98% / 3 = **0.41%**. The fine-mesh drag is 0.3050 with a numerical uncertainty of about ±0.41%.
- Asymptotic check: GCI₂₃ = 1.25 × (0.0120 / 0.3080) / 3 = 1.62%, and r^p · GCI₁₂ = 4 × 0.41% = 1.64%. The ratio is close to 1, so the meshes are in the asymptotic range.

**Practical notes**

- If the refinement ratio is not constant, p must be found iteratively; the procedure of Celik et al. (2008, ASME *Journal of Fluids Engineering*) is the standard reference.
- **Oscillatory convergence** (the result goes up, then down) makes the ratio in the formula for p negative. Report it and use a more conservative error estimate.
- Do the same for the **time step** in transient problems.
- **Some quantities never converge.** The peak stress at a sharp re-entrant corner keeps rising with refinement because the exact solution is singular there. Leave such points out of convergence checks, or evaluate stresses a small distance away. See [Stress concentrations and singularities](stress-concentrations-and-singularities.md#why-the-peak-stress-rises-with-mesh-refinement).

## Code: GCI from three meshes

```python
import numpy as np

def gci_three_meshes(f1, f2, f3, r, Fs=1.25):
    """f1 = finest, f3 = coarsest, constant refinement ratio r > 1."""
    p = np.log(abs((f3 - f2) / (f2 - f1))) / np.log(r)       # observed order
    f_exact = f1 + (f1 - f2) / (r**p - 1)                   # Richardson extrapolation
    gci12 = Fs * abs((f2 - f1) / f1) / (r**p - 1)           # fine-mesh GCI
    gci23 = Fs * abs((f3 - f2) / f2) / (r**p - 1)
    asymptotic_ratio = gci23 / (r**p * gci12)               # close to 1 => asymptotic range
    return p, f_exact, gci12, asymptotic_ratio

p, f_ext, gci, ratio = gci_three_meshes(0.3050, 0.3080, 0.3200, r=2.0)
print(f"p = {p:.2f}, extrapolated = {f_ext:.4f}, GCI_fine = {100 * gci:.2f}%, ratio = {ratio:.3f}")
# p = 2.00, extrapolated = 0.3040, GCI_fine = 0.41%, ratio = 0.990
```

## Sanity checks before trusting an FEA result

Interviewers ask this to see whether you have actually run analyses. Go through these before believing any number:

1. **Equilibrium.** The sum of reaction forces equals the applied load in each direction, and moments balance. A mismatch means a load or constraint is not what you think.
2. **Deformed shape.** Plot the deformation, scaled up. Does the part bend the way your intuition says? Is anything flying away (an unconstrained part) or penetrating another part?
3. **Units.** mm–N–MPa versus m–N–Pa is the classic error. Steel's Young's modulus is 210,000 in MPa (right for a mm–N–MPa model) but 2.1e11 in Pa. Typing 210,000 into a model built in m–N–Pa makes the steel a million times too soft. Check every material value against the unit system, and check density too (tonnes/mm³ in the mm–N–MPa system).
4. **Hand calculation.** Away from local features, compare with beam theory, plate theory or a simple stress = force / area. If the FEA is 10 times different from the hand calculation, find out why.
5. **Mesh quality and convergence.** Check the quality report (aspect ratio, Jacobian, skewness) and refine once in the critical region. If the peak stress keeps rising, ask whether the point is a singularity. See [Meshing and mesh quality](meshing-and-mesh-quality.md).
6. **Averaged vs unaveraged stresses.** A large difference between nodal-averaged and element stresses at the hot spot means the mesh there is too coarse.
7. **Constraints and rigid-body motion.** Warnings about singular stiffness or very large displacements usually mean an unconstrained rigid-body mode.
8. **Contact status.** Is the contact open or closed where you expect? Are there unexpected penetrations?
9. **Symmetry.** A symmetric part under a symmetric load should give a symmetric answer.
10. **Energy checks in explicit dynamics.** Total energy should be conserved (allowing for work done), and artificial hourglass energy should stay small compared with internal energy. A common rule of thumb is a few percent at most.
11. **Material limits.** If the stress is far above yield in a linear analysis, the linear result is not physically meaningful in that region; a nonlinear analysis may be needed. See [Nonlinear structural analysis](nonlinear-structural-analysis.md).
12. **Compare with test data or a previous validated model**, when available.

## Sanity checks for CFD

- The quantities of interest (drag, lift, pressure drop) have stopped changing, not just the residuals.
- Mass flow in equals mass flow out; global energy balance holds in heat-transfer cases.
- The y+ distribution suits the wall treatment. See [Boundary layers and wall treatment](boundary-layers-and-wall-treatment.md).
- Drag is independent of further mesh refinement.
- The domain is large enough: the result does not change when the far-field boundaries are moved further away.
- For transient flows, statistics are averaged over enough cycles of the dominant frequency.
- Results agree with a known case or an experiment.

## When the labels have discretization error

A surrogate is trained to imitate the solver, so it learns the solver's discretization error and modelling bias along with the physics. That changes how you should evaluate it:

1. **Estimate the label error first.** Run a mesh study (GCI) on a representative subset of the training cases. If drag has about ±1% discretization uncertainty, that is your noise floor.
2. **Do not claim accuracy below the floor.** A surrogate with 0.3% drag error against labels that are uncertain by ±1% has not been shown to be "0.3% accurate". It may just be imitating the mesh error more precisely.
3. **Compare like with like.** A useful comparison is the surrogate-versus-solver error alongside the solver's own coarse-versus-fine difference. If the surrogate's error is well below the mesh-refinement difference, extra model accuracy brings no real benefit.
4. **Watch for systematic bias.** If every training case used the same meshing strategy, they share a consistent bias (for example, a slightly under-resolved wake). The surrogate inherits that bias, and test errors against the same kind of labels cannot reveal it. Only finer meshes or experiments can.
5. **Watch for inconsistent labels.** If some cases were meshed differently, or some runs were not fully converged, the labels contain noise that is not physics. Store solver logs and convergence metadata with every sample, filter out failed or unconverged runs, and investigate failures instead of dropping them silently: they often cluster in physically interesting regions (separation, buckling).
6. **Keep a high-fidelity test subset.** A few cases run on much finer meshes, or with a higher-fidelity model, show how far the whole chain is from the truth.
7. **Report two error budgets:** surrogate versus solver, and solver versus reality.

See also [When the training data has its own errors](uncertainty-quantification.md#when-the-training-data-has-its-own-errors).

## In an interview

**Thirty-second answer to "What is the difference between verification and validation?"** "Verification asks whether we solve the equations correctly: code verification with manufactured solutions, and solution verification with mesh and time-step studies. Validation asks whether they are the right equations for the purpose, by comparing with experiments for the intended use. You verify first, otherwise a good match with experiment could be a numerical error cancelling a modelling error. For a surrogate, test-set error is only the verification-like part; validation against reality is a separate claim."

**Likely follow-ups**

- *What is the GCI?* A standardized error band from a three-mesh study: estimate the observed order of convergence, extrapolate, and scale the fine-mesh change by a safety factor of 1.25. Then check the asymptotic range.
- *The peak stress keeps rising as you refine. Is the mesh bad?* Probably not: a sharp re-entrant corner is a singularity. See [Stress concentrations and singularities](stress-concentrations-and-singularities.md).
- *Which error dominates in practice?* Usually set-up and modelling error, not discretization.

## References

- [Examining Spatial (Grid) Convergence (NASA Glenn Research Center)](https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html)
- [NASA Turbulence Modeling Resource: verification cases and grid studies](https://turbmodels.larc.nasa.gov/)
- W. L. Oberkampf and C. J. Roy, *Verification and Validation in Scientific Computing*, Cambridge University Press, 2010.
- P. J. Roache, *Verification and Validation in Computational Science and Engineering*, Hermosa Publishers, 1998.
- I. B. Celik et al., "Procedure for Estimation and Reporting of Uncertainty Due to Discretization in CFD Applications", *Journal of Fluids Engineering*, 2008.
