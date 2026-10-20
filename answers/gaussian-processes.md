# Gaussian Processes

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A Gaussian process (GP) is a probability distribution over functions, specified by a mean function and a **kernel** k(x, x′) that says how strongly the outputs at two inputs are correlated. Conditioning the GP on training data gives, at every new input, a **Gaussian predictive distribution**: its mean is the prediction and its variance is the model's uncertainty. The variance is small near training points and grows back to the prior variance far from them, so a GP **knows where it has no data**. Kernel hyperparameters (length scales, signal variance, noise) are fitted by **maximizing the marginal likelihood**, which balances data fit against model complexity. The price is cost: exact training needs a Cholesky factorization of an n × n matrix, O(n³) time and O(n²) memory, which limits plain GPs to roughly 10⁴ samples; inducing-point approximations, structured kernels and GPU iterative solvers push this much further. GPs are the standard surrogate for small, parametric engineering datasets and the engine of Bayesian optimization.

**On this page**

- [Gaussian process regression](#gaussian-process-regression)
- [How a GP estimates its own uncertainty](#how-a-gp-estimates-its-own-uncertainty)
- [Choosing a kernel and fitting hyperparameters](#choosing-a-kernel-and-fitting-hyperparameters)
- [Why GPs scale badly, and the workarounds](#why-gps-scale-badly-and-the-workarounds)
- [GPs for engineering outputs](#gps-for-engineering-outputs)
- [Code: GP regression from scratch](#code-gp-regression-from-scratch)
- [In an interview](#in-an-interview)

## Gaussian process regression

**The prior.** Write f ~ GP(m, k): for any finite set of inputs X = (x₁, …, x_n), the vector (f(x₁), …, f(x_n)) is jointly Gaussian with mean m(X) and covariance K = k(X, X). The kernel encodes assumptions such as smoothness and the scale over which the function varies. Usually m = 0 after centring the data (or a constant or linear trend).

**The likelihood.** Observations are y = f(X) + ε with ε ~ N(0, σ_n² I). For a deterministic simulator, σ_n² is a small "nugget" for numerical stability (or represents discretization and convergence noise).

**The posterior.** Because everything is Gaussian, conditioning is exact. At a test input x*, with k* = k(X, x*):

```math
\mu_* = k_*^\top (K + \sigma_n^2 I)^{-1} y, \qquad
\sigma_*^2 = k(x_*, x_*) - k_*^\top (K + \sigma_n^2 I)^{-1} k_*
```

**Reading the mean.** μ* = Σᵢ αᵢ k(xᵢ, x*) with α = (K + σ_n²I)⁻¹ y: a weighted sum of kernel functions centred on the training points, the same form as [RBF interpolation](classical-surrogates.md#radial-basis-function-interpolation). The GP is a kernel method; equivalently, it is Bayesian linear regression on infinitely many features.

**The weight-space view, one line.** A GP with kernel k is the limit of a Bayesian linear model with basis functions φ(x) and a Gaussian prior on the weights, with k(x, x′) = φ(x)ᵀ Σ_p φ(x′). Infinitely wide Bayesian neural networks also converge to GPs, which connects the two worlds.

## How a GP estimates its own uncertainty

This is a reported PhysicsX interview question; here is the full answer.

1. **The variance formula is "prior variance minus what the data explains".** k(x*, x*) is the uncertainty before seeing data. The subtracted term k*ᵀ(K + σ_n²I)⁻¹k* is large when x* is strongly correlated (through the kernel) with training points, so close to data the variance collapses, and far from data, where k* ≈ 0, it returns to the prior.
2. **It depends on where the data is, not on the observed values.** For fixed hyperparameters, σ*² does not involve y at all. It measures **lack of data**: epistemic uncertainty. The values y enter indirectly, through the fitted hyperparameters.
3. **Observation noise is separate.** σ_n² is aleatoric uncertainty. The predictive distribution for a *new observation* has variance σ*² + σ_n²; for the underlying function, σ*².
4. **The uncertainty is only as good as the kernel.** A stationary kernel assumes the function varies on the same length scale everywhere. Near a sudden change (onset of separation, buckling, a contact closing) it is overconfident on one side and too cautious on the other. Calibration should be checked on held-out simulations. See [Checking calibration](uncertainty-quantification.md#checking-calibration).
5. **Hyperparameter uncertainty is ignored** by the usual point estimate (maximum likelihood); with very few samples, a fully Bayesian treatment or sensible priors give more honest intervals.

The [code below](#code-gp-regression-from-scratch) shows the average predictive standard deviation rising from 0.035 inside the training range to 0.54 beyond it.

## Choosing a kernel and fitting hyperparameters

**Kernels.**

| Kernel | Formula (r = ‖x − x′‖ / ℓ) | Sample functions | Use |
| --- | --- | --- | --- |
| Squared exponential (RBF) | σ² exp(−r²/2) | Infinitely differentiable | Very smooth responses; the default in many libraries, often too smooth |
| Matérn 5/2 | σ² (1 + √5 r + 5r²/3) exp(−√5 r) | Twice differentiable | A robust default for engineering responses |
| Matérn 3/2 | σ² (1 + √3 r) exp(−√3 r) | Once differentiable | Rougher responses |
| Rational quadratic | σ² (1 + r²/2α)^−α | Mixture of length scales | Multi-scale variation |
| Periodic | σ² exp(−2 sin²(π‖x − x′‖/p)/ℓ²) | Periodic | Angles, cyclic loads |
| Linear / polynomial | σ² xᵀx′ (+ c)^p | Trends | Add to a stationary kernel for a global trend |

- **ARD (automatic relevance determination):** one length scale per input dimension, ℓ_d. A large fitted ℓ_d means the output barely varies with input d, a built-in sensitivity analysis.
- **Combining kernels:** sums model independent effects (a trend plus local variation), products model interactions (a periodic pattern whose amplitude varies).
- **Physics-informed choices:** use dimensionless inputs, log-transform quantities that span orders of magnitude, and choose differentiability to match the physics.

**Fitting hyperparameters.** Maximize the log marginal likelihood (the evidence) of the data under the GP:

```math
\log p(y \mid X, \theta) = -\tfrac12\, y^\top (K_\theta + \sigma_n^2 I)^{-1} y \;-\; \tfrac12 \log\lvert K_\theta + \sigma_n^2 I\rvert \;-\; \tfrac{n}{2}\log 2\pi
```

- The first term rewards fitting the data; the log-determinant penalizes complexity (short length scales make K closer to the identity and its determinant larger). This automatic Occam's razor is why GPs rarely overfit badly with few samples.
- Optimize log-parameters with L-BFGS, with **several restarts**: the likelihood is non-convex, and a short-length-scale, low-noise solution (interpolate everything) can compete with a smooth, noisier one.
- **Standardize** inputs to [0, 1] and outputs to zero mean and unit variance before fitting, so default initial values make sense.
- Put **bounds or priors** on hyperparameters (noise not below 1e-6 for conditioning; length scales within the data range).
- Alternatives: leave-one-out cross-validation (cheap closed form for GPs), or full Bayesian inference with MCMC over hyperparameters.

## Why GPs scale badly, and the workarounds

**The cost.** Training needs (K + σ_n²I)⁻¹y and log|K + σ_n²I|, computed through a Cholesky factorization: **O(n³) time** and **O(n²) memory**, repeated at every hyperparameter step. Prediction then costs O(n) per mean and O(n²) per variance. Doubling n multiplies the factorization time by about eight; the [code below](#code-gp-regression-from-scratch) measures this. On a workstation, exact GPs are comfortable up to a few thousand points and painful beyond about 10⁴.

**Workarounds.**

| Approach | Idea | Cost |
| --- | --- | --- |
| Inducing points (subset of regressors, FITC, variational sparse GPs) | Summarize the data through m ≪ n inducing inputs; the variational version (Titsias; Hensman et al. for stochastic training) learns their locations | O(nm²) training, O(m²) per prediction; minibatch training for millions of points |
| Structured kernel interpolation (KISS-GP) | Interpolate onto a grid where the kernel matrix has Kronecker or Toeplitz structure | Near-linear in n for low dimensions |
| Iterative solvers on GPUs (GPyTorch, BBMM) | Conjugate gradients and Lanczos for solves and log-determinants, using only matrix-vector products | O(n²) per iteration, parallel; exact GPs on about 10⁶ points reported |
| Local GPs / mixtures of experts | Partition the input space and fit a GP per region | Each local model is small |
| Random Fourier features | Approximate the kernel with D random features; Bayesian linear regression | O(nD²) |
| Kronecker structure | Grid-structured inputs (for example space × time) factorize exactly | Large speed-ups when applicable |

**The other scaling problem: dimension.** With 50+ inputs, stationary kernels need exponentially many points to cover the space, and length-scale fitting becomes unreliable. Remedies: sensitivity analysis to drop inputs ([Sobol indices](design-of-experiments-and-active-learning.md#global-sensitivity-analysis-and-sobol-indices)), active subspaces, additive kernels, or deep kernel learning (a neural network feature extractor feeding a GP).

## GPs for engineering outputs

- **Scalar outputs** (drag, peak stress): one GP per output, or a multi-output GP if outputs are correlated.
- **Field outputs:** a GP per mesh node is impractical for 10⁶ nodes. Compress fields with POD and fit one GP per mode coefficient. See [Reduced-order models](reduced-order-models.md).
- **Multi-fidelity:** co-kriging, y_high(x) = ρ y_low(x) + δ(x) with δ a GP. See [Multi-fidelity learning](multi-fidelity-learning.md).
- **Optimization:** expected improvement and other acquisition functions use the GP's mean and variance. See [Bayesian optimization](bayesian-optimization.md).
- **Active learning:** simulate where the predictive variance (or its effect on a quantity of interest) is largest. See [Active learning and acquisition functions](design-of-experiments-and-active-learning.md#active-learning-and-acquisition-functions).
- **Derivative information:** GPs can condition on gradients (from an adjoint solver), which is very data-efficient.

## Code: GP regression from scratch

Three parts: prediction and marginal-likelihood fitting with the Cholesky factorization (never invert K directly); the cubic cost of exact training; and a subset-of-regressors inducing-point approximation that fits 20,000 points in a fraction of a second. Timings depend on the machine; the trend is what matters.

```python
import time
import numpy as np
from scipy.optimize import minimize

def rbf(A, B, lengthscale=0.3, variance=1.0):
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return variance * np.exp(-0.5 * d2 / lengthscale ** 2)

def gp_predict(X, y, Xs, noise=1e-4, **kern):
    """Posterior mean and standard deviation of a zero-mean GP at test inputs Xs."""
    K = rbf(X, X, **kern) + noise * np.eye(len(X))
    L = np.linalg.cholesky(K)                                        # K = L L^T, O(n^3)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))              # (K + noise I)^-1 y
    Ks = rbf(X, Xs, **kern)
    mean = Ks.T @ alpha
    v = np.linalg.solve(L, Ks)
    var = kern.get("variance", 1.0) - (v ** 2).sum(0)               # prior variance minus what the data explains
    return mean, np.sqrt(np.maximum(var, 0.0))

def neg_log_marginal_likelihood(log_params, X, y):
    lengthscale, variance, noise = np.exp(log_params)
    K = rbf(X, X, lengthscale, variance) + (noise + 1e-8) * np.eye(len(X))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    return 0.5 * y @ alpha + np.log(np.diag(L)).sum() + 0.5 * len(X) * np.log(2 * np.pi)   # log|K| = 2 sum log diag L

rng = np.random.default_rng(0)
f = lambda x: np.sin(6 * x)
X = rng.random((15, 1))
y = f(X[:, 0]) + rng.normal(0, 0.05, 15)
starts = np.log([[0.1, 1, 0.01], [0.5, 1, 0.01], [1.0, 0.5, 0.1]])          # restarts: the likelihood is non-convex
best = min((minimize(neg_log_marginal_likelihood, x0, args=(X, y), method="L-BFGS-B") for x0 in starts),
           key=lambda r: r.fun)
ls, var, noise = np.exp(best.x)
print(f"fitted length scale {ls:.3f}, signal variance {var:.3f}, noise std {np.sqrt(noise):.3f} (true 0.05)")

Xs = np.linspace(0, 1.5, 301)[:, None]
mu, sd = gp_predict(X, y, Xs, noise=noise, lengthscale=ls, variance=var)
inside = Xs[:, 0] <= 1.0
print(f"mean predictive std: inside the data {sd[inside].mean():.3f}, beyond it (1 < x <= 1.5) {sd[~inside].mean():.3f}")
y_new = f(Xs[:, 0]) + rng.normal(0, 0.05, len(Xs))
band = 2 * np.sqrt(sd ** 2 + noise)                                          # new observations include noise
print(f"fraction of new observations inside mean +/- 2 std (x <= 1): {np.mean(np.abs(y_new - mu)[inside] <= band[inside]):.3f}")

# Cost: exact GP training is a Cholesky factorization, O(n^3) time and O(n^2) memory.
for n in [1000, 2000, 4000]:
    Z = rng.random((n, 3))
    K = rbf(Z, Z, 0.3) + 1e-3 * np.eye(n)
    t0 = time.perf_counter(); np.linalg.cholesky(K); t = time.perf_counter() - t0
    print(f"n = {n}: Cholesky {t * 1e3:7.1f} ms, kernel matrix {K.nbytes / 1e6:6.1f} MB")

# Workaround: subset-of-regressors with m << n inducing points, O(n m^2).
def sor_predict(X, y, Xs, Z, noise, **kern):
    Kmm = rbf(Z, Z, **kern) + 1e-8 * np.eye(len(Z))
    Kmn = rbf(Z, X, **kern)
    return rbf(Xs, Z, **kern) @ np.linalg.solve(noise * Kmm + Kmn @ Kmn.T, Kmn @ y)

n = 20000
Xb = rng.random((n, 2))
yb = np.sin(4 * Xb[:, 0]) * np.cos(3 * Xb[:, 1]) + rng.normal(0, 0.05, n)
Xt = rng.random((2000, 2))
yt = np.sin(4 * Xt[:, 0]) * np.cos(3 * Xt[:, 1])
for m in [10, 50, 200]:
    Z = Xb[rng.choice(n, m, replace=False)]                                  # inducing inputs: a random subset
    t0 = time.perf_counter()
    pred = sor_predict(Xb, yb, Xt, Z, 0.05 ** 2, lengthscale=0.3)
    print(f"n = {n}, m = {m:3d} inducing points: RMSE {np.sqrt(np.mean((pred - yt) ** 2)):.4f} in {time.perf_counter() - t0:.2f} s")
```

Output (timings from one run):

```
fitted length scale 0.312, signal variance 1.100, noise std 0.041 (true 0.05)
mean predictive std: inside the data 0.035, beyond it (1 < x <= 1.5) 0.538
fraction of new observations inside mean +/- 2 std (x <= 1): 0.915
n = 1000: Cholesky    14.9 ms, kernel matrix    8.0 MB
n = 2000: Cholesky    99.3 ms, kernel matrix   32.0 MB
n = 4000: Cholesky   576.0 ms, kernel matrix  128.0 MB
n = 20000, m =  10 inducing points: RMSE 0.0786 in 0.01 s
n = 20000, m =  50 inducing points: RMSE 0.0023 in 0.04 s
n = 20000, m = 200 inducing points: RMSE 0.0024 in 0.18 s
```

**What it shows.**

- Marginal-likelihood fitting recovers a sensible length scale and a noise level close to the true 0.05 from only 15 points.
- The predictive standard deviation is about 15 times larger outside the data than inside it: the GP knows where it has not seen data.
- About 92% of new observations fall inside the ±2σ band, a little below the nominal 95%, typical of maximum-likelihood hyperparameters on 15 points; check calibration on held-out data rather than assuming it.
- Doubling n multiplies the Cholesky time by roughly 6-8 and the memory by 4.
- Fifty inducing points match the smooth 20,000-point function to RMSE 0.002 in well under a second; ten are too few to represent it.

## In an interview

**Thirty-second answer to "Explain GP regression. How does a GP estimate its own uncertainty?"** "A GP puts a Gaussian prior over functions through a kernel that says how correlated outputs are at nearby inputs. Conditioning on the training data gives a Gaussian at every new point: the mean is a kernel-weighted combination of the training outputs, and the variance is the prior variance minus the part explained by correlations with training points. Near data it is small, far from data it returns to the prior, and with fixed hyperparameters it depends only on where the data is. Hyperparameters such as length scales and noise are fitted by maximizing the marginal likelihood. The caveat is that the uncertainty is only as good as the kernel, so I check calibration on held-out simulations."

**Likely follow-ups**

- *Which kernel would you pick?* Matérn 5/2 with ARD length scales on standardized inputs as the default; add structure (trends, periodicity) when physics suggests it.
- *Why do GPs scale badly?* O(n³) Cholesky and O(n²) memory at every hyperparameter step; use inducing points, structured kernels or GPU iterative solvers.
- *How do you get a field out of a GP?* POD on the snapshots, one GP per mode coefficient.

## References

- [Gaussian Processes for Machine Learning (Rasmussen & Williams, free online)](https://gaussianprocess.org/gpml/)
- [A Visual Exploration of Gaussian Processes (Distill)](https://distill.pub/2019/visual-exploration-gaussian-processes/)
- M. Titsias, "Variational Learning of Inducing Variables in Sparse Gaussian Processes", AISTATS 2009.
- [Gaussian Processes for Big Data (Hensman, Fusi & Lawrence)](https://arxiv.org/abs/1309.6835)
- [Exact Gaussian Processes on a Million Data Points (Wang et al.)](https://arxiv.org/abs/1903.08114)
- [GPyTorch](https://github.com/cornellius-gp/gpytorch), [BoTorch](https://github.com/pytorch/botorch) and [scikit-learn Gaussian processes](https://scikit-learn.org/stable/modules/gaussian_process.html)
- Interview report mentioning GP uncertainty: [PhysicsX on Glassdoor](https://www.glassdoor.com.mx/Entrevista/PhysicsX-Entrevista-E9306867-RVW97800229.htm)
