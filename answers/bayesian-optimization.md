# Bayesian Optimization

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Bayesian optimization (BO) finds the optimum of an **expensive black-box function**, such as the drag of a design that costs hours of CFD per evaluation, in as few evaluations as possible. It fits a **probabilistic surrogate** (usually a Gaussian process) to the evaluations so far, then chooses the next point by maximizing an **acquisition function** that balances **exploitation** (where the predicted value is good) against **exploration** (where the model is uncertain). **Expected improvement (EI)**, the expected amount by which a new point beats the best value so far under the GP's predictive distribution, is the classic choice; upper/lower confidence bounds, probability of improvement, knowledge gradient and entropy-based criteria are alternatives. Evaluate, refit, repeat. BO shines with tens to a few hundred evaluations and up to roughly 10-20 continuous variables; it extends to noisy evaluations, constraints, parallel batches, multiple objectives and multiple fidelities. In the code below, BO reaches within 1% of the Branin function's minimum in 30 evaluations, a level random search typically failed to reach in 1,000.

**On this page**

- [The algorithm](#the-algorithm)
- [Acquisition functions](#acquisition-functions)
- [Practical choices](#practical-choices)
- [Extensions: noise, constraints, batches, objectives and fidelities](#extensions-noise-constraints-batches-objectives-and-fidelities)
- [When not to use Bayesian optimization](#when-not-to-use-bayesian-optimization)
- [Code: expected improvement from scratch](#code-expected-improvement-from-scratch)
- [In an interview](#in-an-interview)

## The algorithm

1. **Initial design:** evaluate the objective at a small space-filling set of points (for example a Latin hypercube of 2-10 points per dimension). See [Latin hypercube vs Sobol vs random sampling](design-of-experiments-and-active-learning.md#latin-hypercube-vs-sobol-vs-random-sampling).
2. **Fit the surrogate:** a Gaussian process with a Matérn 5/2 kernel and fitted length scales gives a mean μ(x) and standard deviation σ(x) everywhere. See [Gaussian processes](gaussian-processes.md).
3. **Maximize the acquisition function** α(x) over the design space. This inner optimization is cheap (no simulations), done with multi-start gradient methods or many random candidates.
4. **Evaluate** the objective at the chosen point (run the simulation).
5. **Update** the data and repeat from step 2 until the budget is spent or improvements stall.
6. **Recommend** the best evaluated point (or the best posterior mean), verified.

## Acquisition functions

For minimization with best observed value f* = min y:

| Acquisition | Formula | Behaviour |
| --- | --- | --- |
| Probability of improvement (PI) | Φ((f* − μ − ξ)/σ) | Greedy; ignores how large the improvement is |
| **Expected improvement (EI)** | (f* − μ − ξ)Φ(z) + σφ(z), z = (f* − μ − ξ)/σ | Balances size and probability of improvement; the default since EGO (Jones, Schonlau and Welch, 1998) |
| Lower confidence bound (LCB / GP-UCB) | μ − κσ (minimize) | Explicit exploration weight κ; has regret guarantees (Srinivas et al.) |
| Knowledge gradient | Expected improvement of the best posterior mean after the evaluation | Good for noisy problems and final recommendations |
| Entropy search, max-value entropy search | Expected reduction in uncertainty about the optimum's location or value | Information-theoretic; extend naturally to multi-fidelity |
| Thompson sampling | Minimize one random sample of the posterior | Simple, naturally parallel |

**EI in words.** Under the GP, the value at x is Gaussian with mean μ and spread σ. The improvement is max(f* − f(x), 0). Its expectation is large if μ is well below f* (exploitation) or if σ is large so that a big improvement is plausible (exploration). Φ and φ are the standard normal CDF and density; ξ ≥ 0 adds a little extra exploration.

## Practical choices

- **Normalize** inputs to the unit cube and outputs to zero mean and unit variance; log-transform outputs that span orders of magnitude.
- **Kernel:** Matérn 5/2 with ARD length scales; refit hyperparameters every iteration (or every few); consider priors on length scales in higher dimensions.
- **Inner optimization:** many random or Sobol candidates followed by local gradient refinement; EI is multimodal and flat in most of the space.
- **Initial design size:** too small and the GP's hyperparameters are poorly determined; a common choice is 2d to 10d points.
- **Failed simulations:** treat crashed or unconverged runs as information (a feasibility classifier), not silently skipped points.
- **Stopping:** when EI falls below a threshold that matters in engineering units, or the budget is spent.
- **Libraries:** BoTorch (PyTorch, Monte Carlo acquisition functions, batch and multi-objective), Ax (experiment management on BoTorch), scikit-optimize, Emukit, SMT; Optuna's TPE sampler is a different, density-based BO variant popular for hyperparameter tuning.

## Extensions: noise, constraints, batches, objectives and fidelities

- **Noisy evaluations** (unsteady CFD averages, physical tests): include a noise term in the GP and use noise-aware acquisitions (noisy EI, knowledge gradient); recommend by posterior mean, not by the luckiest observation.
- **Constraints** (stress below allowable, mass below a limit): model each constraint with its own GP and multiply EI by the probability of feasibility (constrained EI).
- **Parallel batches** for a cluster that runs q simulations at once: q-EI and other Monte Carlo batch acquisitions, local penalization, or "Kriging believer" fantasies.
- **Multiple objectives:** expected hypervolume improvement (EHVI) or random scalarizations (ParEGO) to approximate the Pareto front. See [Multiple objectives](surrogate-based-optimization.md#multiple-objectives).
- **Multiple fidelities:** let the acquisition function choose both the point and the fidelity (coarse or fine mesh), trading cost against information. See [Multi-fidelity learning](multi-fidelity-learning.md).
- **High dimensions** (50+ variables): trust-region methods (TuRBO), additive or low-dimensional embeddings (REMBO), or gradient-based optimization with adjoints.
- **Gradients available** (from an adjoint): GPs can condition on derivatives, which helps a lot per evaluation.

## When not to use Bayesian optimization

- **Cheap objectives:** if an evaluation takes seconds, evolutionary algorithms or gradient methods with thousands of evaluations are simpler and often better.
- **Very high-dimensional design spaces** (shape optimization with thousands of mesh nodes): adjoint-based gradient optimization. See [The adjoint method](adjoint-method.md).
- **Large evaluation budgets** (thousands): GP cost grows as n³; switch to scalable surrogates or other optimizers.
- **Discontinuous or highly non-stationary objectives:** the GP's uncertainty is misleading near regime changes; use non-stationary models or partition the space.
- **When a deep surrogate already exists** for the design family: optimize on it with the precautions on [Surrogate-based optimization](surrogate-based-optimization.md), and use BO-style acquisitions only to choose verification simulations.

## Code: expected improvement from scratch

Bayesian optimization with a GP (Matérn 5/2, fitted length scales) and expected improvement on the two-dimensional Branin test function, against random search, five runs each from different seeds. The GP comes from scikit-learn; EI and the loop are written out.

```python
import warnings
import numpy as np
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
warnings.filterwarnings("ignore")

def branin(X):
    """The Branin test function on [0, 1]^2 (rescaled from [-5, 10] x [0, 15]); global minimum 0.397887."""
    x1, x2 = 15 * X[:, 0] - 5, 15 * X[:, 1]
    return (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10

def expected_improvement(mu, sd, best, xi=0.01):
    """EI for minimization: E[max(best - f - xi, 0)] under the GP posterior N(mu, sd^2)."""
    sd = np.maximum(sd, 1e-12)
    z = (best - mu - xi) / sd
    return (best - mu - xi) * norm.cdf(z) + sd * norm.pdf(z)

def bayes_opt(f, n_init=6, n_iter=24, seed=0):
    rng = np.random.default_rng(seed)
    X = qmc.LatinHypercube(2, seed=seed).random(n_init)
    y = f(X)
    for it in range(n_iter):
        gp = GaussianProcessRegressor(ConstantKernel() * Matern(length_scale=[0.2, 0.2], nu=2.5), alpha=1e-6,
                                      normalize_y=True, n_restarts_optimizer=3, random_state=seed).fit(X, y)
        cand = rng.random((4000, 2))                                # maximize EI over random candidates
        mu, sd = gp.predict(cand, return_std=True)
        x_next = cand[np.argmax(expected_improvement(mu, sd, y.min()))][None]
        X, y = np.vstack([X, x_next]), np.append(y, f(x_next))
    return np.minimum.accumulate(y)

def random_search(f, n=30, seed=0):
    return np.minimum.accumulate(f(np.random.default_rng(seed).random((n, 2))))

f_min = 0.397887
bo = np.array([bayes_opt(branin, seed=s) for s in range(5)])
rs = np.array([random_search(branin, seed=s) for s in range(5)])
print("best value found (mean of 5 runs); global minimum 0.398")
for n in [6, 10, 15, 20, 30]:
    print(f"after {n:2d} evaluations: Bayesian optimization {bo[:, n - 1].mean():7.3f}   random search {rs[:, n - 1].mean():7.3f}")
rs_long = np.array([random_search(branin, n=1000, seed=100 + s) for s in range(5)])
needed = [int(np.argmax(r <= bo[i, -1])) + 1 if (r <= bo[i, -1]).any() else None for i, r in enumerate(rs_long)]
print("random-search evaluations needed to match each BO run's result after 30:", [n if n else ">1000" for n in needed])
```

Output:

```
best value found (mean of 5 runs); global minimum 0.398
after  6 evaluations: Bayesian optimization   6.598   random search   5.510
after 10 evaluations: Bayesian optimization   4.786   random search   4.618
after 15 evaluations: Bayesian optimization   1.421   random search   4.618
after 20 evaluations: Bayesian optimization   0.725   random search   2.430
after 30 evaluations: Bayesian optimization   0.402   random search   2.401
random-search evaluations needed to match each BO run's result after 30: ['>1000', '>1000', '>1000', '>1000', 324]
```

**What it shows.** During the six-point initial design both methods are equally lost; then BO's model-guided choices pay off quickly: after 20 evaluations it is within 0.33 of the minimum and after 30 within 1% (0.402 against 0.398) on average, while random search is still around 2.4. Matching BO's 30-evaluation result took random search 324 evaluations in one run and more than 1,000 in the other four. With each evaluation a CFD run of several hours, that difference is the whole case for BO. The flip side: with the Branin function's 2 inputs, almost any model-based method beats random search; the advantage shrinks as dimension grows and the GP becomes harder to fit.

## In an interview

**Thirty-second answer to "How does Bayesian optimization work, and when would you use it?"** "It is for expensive black-box objectives, like a design whose drag needs a CFD run. Fit a Gaussian process to the evaluations so far, which gives a mean and an uncertainty everywhere, then pick the next point by maximizing an acquisition function such as expected improvement, which is large where the predicted value is good or the uncertainty is high. Run the simulation, refit, repeat. It is very sample-efficient for tens to hundreds of evaluations and up to about 10-20 variables, and extends to noise, constraints, parallel batches, multiple objectives and fidelities; for thousands of variables I would use adjoint gradients instead."

**Likely follow-ups**

- *EI or UCB?* EI is the robust default; UCB gives an explicit exploration knob and theory; for batches, Monte Carlo q-EI or Thompson sampling.
- *How do you handle constraints?* A GP per constraint and EI times the probability of feasibility.
- *Why not just random search?* It wastes evaluations in regions already known to be poor; BO uses the model to avoid them, as the example shows.

## References

- [A Tutorial on Bayesian Optimization (Frazier)](https://arxiv.org/abs/1807.02811)
- D. R. Jones, M. Schonlau and W. J. Welch, "Efficient Global Optimization of Expensive Black-Box Functions", *Journal of Global Optimization*, 1998.
- [Taking the Human Out of the Loop: A Review of Bayesian Optimization (Shahriari et al.)](https://ieeexplore.ieee.org/document/7352306)
- [Gaussian Process Optimization in the Bandit Setting (GP-UCB; Srinivas et al.)](https://arxiv.org/abs/0912.3995)
- [Scalable Global Optimization via Local Bayesian Optimization (TuRBO; Eriksson et al.)](https://arxiv.org/abs/1910.01739)
- [BoTorch](https://botorch.org/) and [Ax](https://ax.dev/)
