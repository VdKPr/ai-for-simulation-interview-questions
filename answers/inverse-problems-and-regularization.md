# Inverse Problems and Regularization

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A **forward problem** computes effects from causes (temperatures from a heat source, displacements from material properties). An **inverse problem** infers causes from measured effects: the material stiffness from measured strains, a heat flux from thermocouples, an initial state from a later observation, a crack from ultrasonic signals. Most are **ill-posed** in Hadamard's sense: a solution may not exist for noisy data, may not be unique, and, above all, **does not depend continuously on the data**, because the forward physics smooths away information (diffusion damps high frequencies exponentially), so inverting it amplifies measurement noise without bound. **Regularization** restores stability by adding prior knowledge: **Tikhonov** (penalize the size or roughness of the solution), **truncated SVD** (keep only well-determined components), **total variation** (allow sharp edges, penalize oscillation), sparsity penalties, or simply stopping an iterative solver early. The regularization weight is chosen by the **discrepancy principle** (fit the data only to the noise level), the L-curve or generalized cross-validation. The **Bayesian view** treats the regularizer as a prior and gives uncertainty with the estimate. Large inverse problems are solved by gradient-based optimization with **adjoint** gradients; machine learning adds learned priors, PINN-based inversion and generative models. In the code below, naive inversion of a smoothing heat operator gives an error of 10¹⁶; regularized reconstructions reach 23-30%.

**On this page**

- [Why inverse problems are ill-posed](#why-inverse-problems-are-ill-posed)
- [Regularization methods](#regularization-methods)
- [Choosing the regularization weight](#choosing-the-regularization-weight)
- [The Bayesian view](#the-bayesian-view)
- [Solving large inverse problems](#solving-large-inverse-problems)
- [Machine learning for inverse problems](#machine-learning-for-inverse-problems)
- [Code: recovering an initial temperature from a noisy later measurement](#code-recovering-an-initial-temperature-from-a-noisy-later-measurement)
- [In an interview](#in-an-interview)

## Why inverse problems are ill-posed

Hadamard called a problem **well-posed** if a solution exists, is unique, and depends continuously on the data. Inverse problems typically fail the third condition.

**The mechanism, through the SVD.** Write the linearized forward model as d = A u + noise and decompose A = U Σ Vᵀ. The naive solution is

```math
u = \sum_i \frac{u_i^\top d}{\sigma_i}\, v_i
```

For a smoothing forward model the singular values σᵢ decay rapidly (for diffusion, exponentially: mode k decays like exp(−κk²t)). The components of the noise along small-σᵢ directions are divided by tiny numbers and explode. The condition number of the heat operator in the code below is about 10¹⁹.

**Examples in engineering.**

| Inverse problem | Unknown | Data | Why it is hard |
| --- | --- | --- | --- |
| Inverse heat conduction | Surface heat flux, initial temperature | Interior thermocouples | Diffusion destroys high frequencies |
| Material identification | Elastic moduli, plasticity parameters, damage | Displacements (digital image correlation), forces | Insensitive parameters, correlated parameters |
| Load identification | Forces on a structure | Strain gauges, accelerations | Few sensors, many possible loads |
| Non-destructive testing | Cracks, voids, delamination | Ultrasound, X-ray, eddy currents | Small features, limited views |
| Data assimilation | Initial state of a flow or weather model | Sparse, noisy observations | Chaos, huge state |
| Model calibration | Turbulence-model or contact coefficients | Experiments | Model error mixed with parameter error |

## Regularization methods

**Tikhonov regularization.** Minimize misfit plus a penalty:

```math
\min_u\; \lVert A u - d \rVert^2 + \lambda^2 \lVert L u \rVert^2
```

with L = I (prefer small solutions) or a derivative operator (prefer smooth solutions). In SVD terms (L = I), each component is multiplied by a filter factor σᵢ²/(σᵢ² + λ²): components with σᵢ ≫ λ pass, components with σᵢ ≪ λ are suppressed smoothly.

**Truncated SVD.** Keep only the r components with the largest singular values; a sharp version of the same filter.

**Total variation (TV).** Penalize Σ|uᵢ₊₁ − uᵢ| (the L1 norm of the gradient) instead of its square. Large jumps cost no more than many small steps of the same total height, so **edges are preserved** while noise is suppressed; well suited to piecewise-constant unknowns (material phases, defects).

**Sparsity (L1) penalties.** Favour solutions with few non-zero coefficients in some basis (wavelets, a dictionary of damage locations); the basis of compressed sensing.

**Iterative regularization.** Run an iterative solver (conjugate gradient, Landweber) on the unregularized problem and **stop early**: the first iterations recover the well-determined components, later ones fit noise. The number of iterations is the regularization parameter.

**Physical constraints.** Bounds (positive conductivity), known structure (a parametric shape), and fewer unknowns (parametrize a heat flux by a few coefficients) are often the strongest regularizers of all.

## Choosing the regularization weight

Too small: the solution fits the noise and oscillates. Too large: it is over-smoothed and biased. The code below shows the error rising from 0.30 to 0.46 or 0.57 when λ is off by a factor of ten either way.

- **Discrepancy principle (Morozov):** choose λ so the residual ‖Au − d‖ equals the expected noise level (about σ√n for n measurements with noise standard deviation σ): fit the data exactly as well as the noise allows, no better. Needs an estimate of the noise level, which instrument specifications or repeated measurements provide.
- **L-curve:** plot log‖Lu‖ against log‖Au − d‖ for many λ; choose the corner, where reducing the misfit further would make the solution blow up.
- **Generalized cross-validation (GCV):** choose λ that best predicts left-out data, without knowing the noise level.
- **Bayesian:** treat λ as a hyperparameter and estimate it from the data (empirical Bayes or a hierarchical prior).

## The Bayesian view

Treat the unknown as a random variable with a **prior** p(u) encoding what is known before the measurement (smoothness, bounds, typical values), and the data model as a **likelihood** p(d | u) from the noise distribution. Bayes' rule gives the **posterior** p(u | d) ∝ p(d | u) p(u).

- With Gaussian noise and a Gaussian prior, the **maximum a posteriori (MAP)** estimate is exactly Tikhonov regularization, with λ² = σ²_noise / σ²_prior: **the regularizer is a prior**.
- The posterior also gives **uncertainty**: which parts of the unknown are well determined by the data and which come only from the prior. For linear Gaussian problems the posterior covariance is explicit; otherwise use MCMC, variational inference, or a Laplace approximation around the MAP point.
- This matters for engineering decisions: "the identified stiffness is 72 ± 3 GPa" is usable; a single number is not.

## Solving large inverse problems

When the unknown is a field (a conductivity per element, an initial flow state) and the forward model is a PDE solver:

1. Formulate the objective J(u) = ½‖F(u) − d‖² + regularization, with F the solver's map from parameters to predicted measurements.
2. Compute ∇J with the **adjoint method**: one forward and one adjoint solve per gradient, whatever the number of unknowns. See [The adjoint method](adjoint-method.md).
3. Minimize with L-BFGS or Gauss-Newton (with conjugate-gradient inner solves using Hessian-vector products from second-order adjoints).
4. Use multilevel (coarse-to-fine) strategies to avoid poor local minima in nonlinear problems (full-waveform inversion is the classic example).
5. Quantify uncertainty with low-rank approximations of the posterior covariance from the Hessian at the MAP point.

Libraries: hIPPYlib and dolfin-adjoint (FEniCS-based), PyTorch or JAX for differentiable forward models.

## Machine learning for inverse problems

- **PINNs for inverse problems:** add the unknown coefficients as trainable parameters and fit data and PDE residual together. Simple to set up and well suited to sparse, scattered data, but subject to PINN training difficulties; identifiability still matters. See [Inverse problems with PINNs](pinns.md#inverse-problems-with-pinns).
- **Learned regularizers and priors:** a network trained on examples of plausible solutions replaces a hand-made penalty (learned denoisers in plug-and-play methods, diffusion-model priors). See [Solving PDEs from partial observations](generative-models-for-physics.md#solving-pdes-from-partial-observations).
- **Unrolled optimization:** a fixed number of iterations of a classical algorithm, with learned step sizes and regularizers, trained end to end.
- **Direct inversion networks:** a network maps measurements straight to the unknown (fast, but no data-consistency guarantee and weak out of distribution); best combined with a few physics-based refinement steps.
- **Surrogate-accelerated inversion:** a neural surrogate replaces the expensive forward model inside MCMC or optimization, with the solver used to verify the final estimate.

## Code: recovering an initial temperature from a noisy later measurement

The forward model is heat diffusion on a periodic rod: an initial temperature profile (a sharp step plus a smooth bump) is smoothed for a short time, and the final profile is measured at 200 points with 1% noise. The task is to recover the initial profile.

```python
import numpy as np

# Forward model: heat diffusion for time t smooths an initial temperature profile (periodic rod, n points).
n, t, kappa = 200, 0.002, 1.0
x = np.arange(n) / n
k = np.fft.fftfreq(n, d=1.0 / n) * 2 * np.pi
decay = np.exp(-kappa * k ** 2 * t)                               # each Fourier mode decays by exp(-kappa k^2 t)
A = np.real(np.fft.ifft(decay[:, None] * np.fft.fft(np.eye(n), axis=0), axis=0))   # the forward operator as a matrix
u_true = np.where((x > 0.2) & (x < 0.35), 1.0, 0.0) + 0.6 * np.exp(-((x - 0.7) / 0.05) ** 2)   # unknown initial state
rng = np.random.default_rng(0)
sigma = 0.01
d = A @ u_true + sigma * rng.normal(size=n)                        # noisy measurement of the final state
print(f"condition number of the forward operator: {np.linalg.cond(A):.1e}")
err = lambda u: np.linalg.norm(u - u_true) / np.linalg.norm(u_true)

u_naive = np.linalg.solve(A, d)
print(f"naive inversion (solve A u = d): relative error {err(u_naive):.1e}")

U, s, Vt = np.linalg.svd(A)
def tikhonov(lam):                                                 # minimize |A u - d|^2 + lam^2 |u|^2
    f = s / (s ** 2 + lam ** 2)
    return Vt.T @ (f * (U.T @ d))
def tsvd(r):
    return Vt[:r].T @ ((U[:, :r].T @ d) / s[:r])

lams = np.logspace(-4, 0, 41)
errs = [err(tikhonov(l)) for l in lams]
residuals = [np.linalg.norm(A @ tikhonov(l) - d) for l in lams]
best = int(np.argmin(errs))
target = sigma * np.sqrt(n)                                        # discrepancy principle: residual = noise level
disc = int(np.argmin(np.abs(np.array(residuals) - target)))
print(f"Tikhonov, best lambda (needs the truth) {lams[best]:.1e}: error {errs[best]:.3f}")
print(f"Tikhonov, lambda by the discrepancy principle {lams[disc]:.1e}: error {errs[disc]:.3f}")
print(f"Tikhonov, lambda 10x too small {lams[disc] / 10:.1e}: error {err(tikhonov(lams[disc] / 10)):.3f}; "
      f"10x too large {lams[disc] * 10:.1e}: error {err(tikhonov(lams[disc] * 10)):.3f}")
r_best = min(range(1, n + 1), key=lambda r: err(tsvd(r)))
print(f"truncated SVD, best rank {r_best} of {n}: error {err(tsvd(r_best)):.3f}")

# Total variation regularization keeps the sharp edges of the step (simple gradient descent on a smoothed TV).
def tv_reconstruct(lam=0.002, eps=1e-3, iters=3000, step=0.5):
    u = tikhonov(lams[disc]).copy()
    for _ in range(iters):
        du = np.roll(u, -1) - u
        w = du / np.sqrt(du ** 2 + eps ** 2)
        grad = A.T @ (A @ u - d) - lam * (w - np.roll(w, 1))
        u -= step * grad
    return u
u_tv = tv_reconstruct()
print(f"total-variation regularization: error {err(u_tv):.3f}")
```

Output:

```
condition number of the forward operator: 8.5e+18
naive inversion (solve A u = d): relative error 1.7e+16
Tikhonov, best lambda (needs the truth) 2.0e-02: error 0.284
Tikhonov, lambda by the discrepancy principle 6.3e-02: error 0.301
Tikhonov, lambda 10x too small 6.3e-03: error 0.463; 10x too large 6.3e-01: error 0.573
truncated SVD, best rank 11 of 200: error 0.302
total-variation regularization: error 0.233
```

**What it shows.**

- The forward operator's condition number is about 10¹⁹, and naive inversion turns 1% measurement noise into an error of 10¹⁶: an ill-posed problem in one line.
- Tikhonov with λ from the discrepancy principle (which needs only the noise level) gets an error of 0.30, close to the best possible Tikhonov error of 0.28 that would require knowing the answer; a λ ten times too small or too large is far worse.
- Truncated SVD with the best rank keeps only 11 of 200 components: the measurement determines only a handful of smooth modes of the initial state; the rest must come from prior knowledge.
- Total variation does best (0.23) because the unknown contains a sharp step, which quadratic penalties smear and TV preserves. Choosing the regularizer is choosing the prior.

## In an interview

**Thirty-second answer to "What is an inverse problem, and why does it need regularization?"** "It is inferring causes from measured effects, like material properties from measured displacements or a heat flux from thermocouples. Most are ill-posed: the forward physics smooths away information, so in an SVD of the forward operator the singular values decay quickly and naive inversion divides measurement noise by tiny numbers. Regularization adds prior knowledge to stabilize it: Tikhonov penalizes size or roughness, truncated SVD keeps only well-determined components, total variation keeps edges. I choose the weight with the discrepancy principle or an L-curve. In Bayesian terms the regularizer is a prior, which also gives uncertainty. For large problems, I compute gradients with the adjoint."

**Likely follow-ups**

- *How do you pick λ?* Discrepancy principle if the noise level is known, otherwise the L-curve or generalized cross-validation.
- *Tikhonov or total variation?* Tikhonov for smooth unknowns; TV for piecewise-constant ones with sharp interfaces.
- *How does it relate to Bayesian inference?* Tikhonov is the MAP estimate with a Gaussian prior and Gaussian noise.

## References

- P. C. Hansen, *Discrete Inverse Problems: Insight and Algorithms*, SIAM, 2010.
- A. Tarantola, *Inverse Problem Theory and Methods for Model Parameter Estimation*, SIAM, 2005 ([SIAM](https://doi.org/10.1137/1.9780898717921)).
- J. Kaipio and E. Somersalo, *Statistical and Computational Inverse Problems*, Springer, 2005.
- [hIPPYlib: inverse problems with FEniCS](https://hippylib.github.io/)
- [Physics-informed neural networks (Raissi, Perdikaris & Karniadakis), including inverse problems](https://doi.org/10.1016/j.jcp.2018.10.045)
