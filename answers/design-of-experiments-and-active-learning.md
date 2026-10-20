# Design of Experiments and Active Learning

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Simulations are expensive, so *which* designs you simulate matters as much as the model you train. Start with a **space-filling design** so the data covers the design space evenly: a **Latin hypercube** guarantees every input is sampled evenly on its own (one point per stratum), and a **scrambled Sobol sequence** is a low-discrepancy sequence that is very uniform in all dimensions jointly and can be extended later; plain random sampling leaves clumps and gaps. **How many simulations** you need is answered by measuring a **learning curve** (error against the number of training simulations), not by a fixed rule; ten per input dimension is a starting point for a GP. After the initial design, **active learning** chooses each next simulation with an **acquisition function**: maximum predictive variance or ensemble disagreement to reduce global error, expected improvement to find an optimum, or the probability of misclassification to locate a pass/fail boundary. **Global sensitivity analysis** with **Sobol indices** splits the output variance into contributions from each input and their interactions, so inputs that do not matter can be frozen before spending simulations on them.

**On this page**

- [Latin hypercube vs Sobol vs random sampling](#latin-hypercube-vs-sobol-vs-random-sampling)
- [How many simulations you need](#how-many-simulations-you-need)
- [Active learning and acquisition functions](#active-learning-and-acquisition-functions)
- [Global sensitivity analysis and Sobol indices](#global-sensitivity-analysis-and-sobol-indices)
- [Data pipeline realities](#data-pipeline-realities)
- [Code: Latin hypercube sampling](#code-latin-hypercube-sampling)
- [In an interview](#in-an-interview)

## Latin hypercube vs Sobol vs random sampling

**Random (Monte Carlo) sampling.** Independent uniform draws. Simple and unbiased, but with small n it leaves clusters and empty regions; the error of a mean estimate falls only as 1/√n.

**Latin hypercube sampling (LHS; McKay, Beckman and Conover, 1979).** Divide each input's range into n equal strata and place exactly one sample in each stratum of every input, pairing the strata randomly across inputs. Every one-dimensional projection is perfectly stratified, so main effects are estimated well, which is why LHS is the default in engineering tools. A plain LHS can still be poor in several dimensions jointly (in the worst case all points on a diagonal), so use an **optimized LHS**: maximin (maximize the smallest distance between points) or minimum discrepancy.

**Sobol and other low-discrepancy sequences (quasi-Monte Carlo).** Deterministic sequences (Sobol, Halton) built so that every sub-box of the unit cube receives close to its fair share of points. The integration error falls close to 1/n instead of 1/√n for smooth functions. **Scrambling** (randomized QMC) removes artefacts and allows error estimates. Sobol sequences are **extensible**: you can add points later without destroying uniformity, which matters when the budget grows. They work best with sample sizes that are powers of two.

**Others.** Full factorial and fractional factorial designs (classical DoE for few inputs and levels), central composite and Box-Behnken designs (for quadratic response surfaces), and sparse grids.

| | Random | Latin hypercube | Scrambled Sobol |
| --- | --- | --- | --- |
| 1D projections | Uneven | One point per stratum | Very even |
| Joint uniformity | Poor for small n | Medium; optimized LHS good | Very good |
| Error of a mean, smooth function | ∝ 1/√n | Better than random | ≈ ∝ 1/n |
| Extend with more points | Yes | No (a new design is needed) | Yes |
| Typical use | Baseline, very high dimensions | Initial surrogate training sets | Integration, sensitivity analysis, extensible designs |

**Recommendation.** For an initial surrogate training set, an optimized LHS or a scrambled Sobol sequence; Sobol if you expect to add simulations later or need integrals (sensitivity indices). Respect constraints: sample in a transformed space or reject infeasible designs (self-intersecting geometry), and include corner and baseline designs that engineers will ask about.

## How many simulations you need

There is no universal number; it depends on the number of inputs, how nonlinear the response is, the model, and the accuracy needed.

**Rules of thumb as starting points.**

- **About 10 samples per input dimension** for a GP on a smooth response (Loeppky, Sacks and Welch, 2009, recommend n = 10d as a reasonable initial design size).
- **At least the number of coefficients**, (d + 1)(d + 2)/2 for a quadratic response surface, ideally more.
- **Hundreds to thousands** of simulations for deep field surrogates on varied geometries; public car-aerodynamics datasets have 355-8,000 designs. See [Car aerodynamics datasets](public-simulation-datasets.md#car-aerodynamics-datasets).

**Measure instead of guessing.**

1. Run an initial design, for example 10d simulations.
2. Train on growing subsets (25%, 50%, 75%, 100%) with a fixed held-out test set, several random seeds each.
3. Plot test error against n on log-log axes. A straight line is a power law; its slope extrapolates how much error another doubling of data buys, and a flattening curve signals a floor set by noise, model capacity or label error. See [What makes training data physically meaningful and sufficient](generating-simulation-training-data.md#what-makes-training-data-physically-meaningful-and-sufficient).
4. Stop when the curve reaches the accuracy target, or when the marginal gain is not worth the simulation cost.

**What raises the requirement.** Regime changes (separation, buckling, contact, transition), many interacting inputs, sharp local features in field outputs, and free-form geometry. **What lowers it:** good input features and non-dimensional groups, symmetries, transfer from a pretrained model, multi-fidelity data. See [Learning from small simulation datasets](learning-from-small-simulation-datasets.md).

## Active learning and acquisition functions

**The loop.**

1. Initial space-filling design; run the simulations.
2. Train the surrogate with an uncertainty estimate (a GP, an ensemble of networks, or MC dropout).
3. Score many candidate designs with an **acquisition function**.
4. Simulate the best candidate (or a batch), add the results, retrain.
5. Stop when an error target, a budget, or a convergence criterion (no further improvement) is reached.

**Acquisition functions by goal.**

| Goal | Acquisition | Idea |
| --- | --- | --- |
| Reduce global error | Maximum predictive variance (uncertainty sampling) | Sample where the model knows least |
| Reduce global error | Integrated variance reduction (IMSE) | Pick the point whose addition most reduces variance over the whole domain |
| Reduce global error (networks) | Ensemble disagreement (query by committee) | Points where independently trained models disagree |
| Find an optimum | Expected improvement (EI), upper/lower confidence bound (UCB/LCB), probability of improvement | Balance predicted good value against uncertainty. See [Bayesian optimization](bayesian-optimization.md) |
| Locate a boundary (pass/fail, stall onset) | U-function or expected feasibility (AK-MCS, EGRA) | Sample where the sign of (prediction − threshold) is most uncertain |
| Estimate a rare failure probability | Same, combined with importance sampling | Concentrate on the limit state |
| Many simulations in parallel | Batch acquisition: q-EI, local penalization, k-means on top candidates, Kriging believer | Pick diverse points so the batch is not redundant |

**Important subtlety.** For a GP with fixed hyperparameters, the predictive variance does not depend on the observed values, only on where the data is. Maximum-variance sampling is therefore a clever space-filling strategy, not a way to find difficult regions; it learns where the function is hard only through the fitted length scales. The [code below](#code-latin-hypercube-sampling) shows this: max-variance sampling reduced the error by 40% compared with random sampling but put fewer points near the sharp transition. To target difficult regions, use acquisitions that depend on the observed data: ensemble disagreement, cross-validation error estimates, gradient or curvature indicators, or non-stationary models.

**Practical points.** Retrain cost can dominate for large models (retrain every batch, not every point); simulations fail in some regions, so treat failures as information (a classifier for feasibility); and keep a random or space-filling fraction in each batch to avoid getting stuck. See [Design: active learning loop](design-active-learning-loop.md).

## Global sensitivity analysis and Sobol indices

**Variance decomposition.** For independent inputs, the output variance of f(X₁, …, X_d) splits into parts due to each input alone, each pair, and so on (the ANOVA or Hoeffding decomposition). The **Sobol indices** are fractions of the variance:

```math
S_i = \frac{\operatorname{Var}\big(\mathbb E[f \mid X_i]\big)}{\operatorname{Var}(f)}, \qquad
S_{T_i} = 1 - \frac{\operatorname{Var}\big(\mathbb E[f \mid X_{\sim i}]\big)}{\operatorname{Var}(f)}
```

- **First-order index S_i:** the share of variance explained by X_i alone (its main effect).
- **Total index S_Ti:** the share involving X_i at all, including all interactions. S_Ti − S_i measures interactions.
- Σ S_i ≤ 1; the gap is the variance due to interactions.

**Estimation.** Saltelli-type estimators use two independent sample matrices A and B and, for each input i, the matrix A with column i taken from B. With N base samples this costs N(d + 2) model evaluations, usually done **on a cheap surrogate**, not the simulator. Sobol or LHS designs make the estimates converge faster. The [code below](#code-latin-hypercube-sampling) reproduces the exact indices of the standard Ishigami test function to three decimals.

**How it guides simplifying a design space.**

1. **Freeze** inputs with total index near zero at nominal values: they matter neither alone nor through interactions. This reduces the dimension of the DoE, often dramatically.
2. **Focus** sampling effort and accuracy requirements on inputs with large indices.
3. **Detect interactions** (S_Ti much larger than S_i): one-at-a-time studies would miss them.
4. **Check physics**: an input you expected to dominate with a near-zero index points to a modelling or data error.

**Cheaper alternatives.** Morris elementary-effects screening ranks inputs with about (d + 1) × 10-20 runs; GP length scales (ARD) give a quick relevance ranking; derivative-based measures use gradients. Variance-based indices assume independent inputs; with correlated inputs, use Shapley effects. Libraries: SALib, OpenTURNS, UQLab.

## Data pipeline realities

- **Invalid geometry:** some parameter combinations produce self-intersecting or non-manufacturable CAD. Check validity before meshing.
- **Failed runs** (meshing failures, solver divergence): log and investigate them; do not drop them silently. They often cluster in physically interesting regions, and silently dropping them biases the dataset.
- **Metadata:** store solver version, settings, mesh statistics and convergence history with every sample. See [Verification and validation](verification-validation-and-mesh-convergence.md).
- **Scheduling:** run in batches on the cluster and make the pipeline restartable. See [Pipeline for 1,000 CFD simulations](generating-simulation-training-data.md#pipeline-for-1000-cfd-simulations).

## Code: Latin hypercube sampling

Four experiments: how uniform random, Latin hypercube and Sobol designs are; how accurately each estimates a mean; active learning with GP variance versus random sampling; and Sobol indices of the Ishigami function, whose exact values are known.

```python
import warnings
import numpy as np
from scipy.stats import qmc
from scipy.spatial.distance import pdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
warnings.filterwarnings("ignore")

def latin_hypercube(n, d, rng=np.random.default_rng(0)):
    """n samples in [0, 1]^d; each dimension has exactly one sample per stratum."""
    cut = (np.arange(n)[:, None] + rng.random((n, d))) / n        # one point inside each of n bins
    for j in range(d):
        cut[:, j] = cut[rng.permutation(n), j]                    # shuffle bins independently per dimension
    return cut                                                    # scale to bounds: lo + cut * (hi - lo)

# 1. Space filling: discrepancy (lower = more uniform) and the smallest distance between points (higher = better).
n, d = 64, 5
rng = np.random.default_rng(1)
designs = {"random": rng.random((n, d)), "Latin hypercube": latin_hypercube(n, d, rng),
           "LHS, optimized": qmc.LatinHypercube(d, optimization="random-cd", seed=1).random(n),
           "scrambled Sobol": qmc.Sobol(d, scramble=True, seed=1).random(n)}
print(f"{'design (n=64, d=5)':<20} {'discrepancy':>12} {'min distance':>13}")
for name, S in designs.items():
    print(f"{name:<20} {qmc.discrepancy(S):>12.5f} {pdist(S).min():>13.3f}")

# 2. Estimating a mean (a stand-in for any integrated output): RMS error over 200 repeated designs.
f = lambda X: np.prod(1 + 0.5 * np.sin(2 * np.pi * X), axis=1) + X[:, 0]   # exact mean over [0, 1]^5 is 1.5
for n in [64, 256, 1024]:
    errs = {"random": [], "Latin hypercube": [], "scrambled Sobol": []}
    for rep in range(200):
        errs["random"].append(f(np.random.default_rng(rep).random((n, d))).mean() - 1.5)
        errs["Latin hypercube"].append(f(qmc.LatinHypercube(d, seed=rep).random(n)).mean() - 1.5)
        errs["scrambled Sobol"].append(f(qmc.Sobol(d, seed=rep).random(n)).mean() - 1.5)
    print(f"n = {n:4d}: RMS error  " + "  ".join(f"{k} {np.sqrt(np.mean(np.square(v))):.5f}" for k, v in errs.items()))

# 3. Active learning: add the point of largest GP standard deviation, or a random point, 30 times.
def target(X):                                                    # a response with a sharp transition at x = 0.75
    return np.tanh(20 * (X[:, 0] - 0.75)) + 0.3 * np.sin(3 * X[:, 1])
grid = np.stack(np.meshgrid(np.linspace(0, 1, 60), np.linspace(0, 1, 60)), -1).reshape(-1, 2)
def gp_fit(X, y):
    kernel = ConstantKernel() * Matern(length_scale=[0.3, 0.3], nu=2.5)
    return GaussianProcessRegressor(kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=2, random_state=0).fit(X, y)
for strategy in ["random", "max variance"]:
    r = np.random.default_rng(0)
    X = qmc.LatinHypercube(2, seed=0).random(10)                  # initial design: 10 simulations
    for step in range(30):
        gp = gp_fit(X, target(X))
        if strategy == "random":
            new = r.random((1, 2))
        else:
            new = grid[[np.argmax(gp.predict(grid, return_std=True)[1])]]
        X = np.vstack([X, new])
    rmse = np.sqrt(np.mean((gp_fit(X, target(X)).predict(grid) - target(grid)) ** 2))
    print(f"{strategy:>12}: 40 simulations, RMSE {rmse:.4f}, share of points near the transition "
          f"{np.mean(np.abs(X[:, 0] - 0.75) < 0.15):.2f}")

# 4. Sobol indices of the Ishigami function (Saltelli first-order and Jansen total-effect estimators).
def ishigami(X, a=7, b=0.1):
    return np.sin(X[:, 0]) + a * np.sin(X[:, 1]) ** 2 + b * X[:, 2] ** 4 * np.sin(X[:, 0])
N, k = 2 ** 14, 3
base = qmc.Sobol(2 * k, seed=2).random(N) * 2 * np.pi - np.pi   # inputs uniform on [-pi, pi]
A, B = base[:, :k], base[:, k:]
fA, fB = ishigami(A), ishigami(B)
var = np.var(np.concatenate([fA, fB]))
S1, ST = [], []
for i in range(k):
    ABi = A.copy(); ABi[:, i] = B[:, i]                            # A with column i taken from B
    fABi = ishigami(ABi)
    S1.append(np.mean(fB * (fABi - fA)) / var)
    ST.append(0.5 * np.mean((fA - fABi) ** 2) / var)
print("first-order S_i:", np.round(S1, 3), " exact [0.314 0.442 0.   ]")
print("total S_Ti:     ", np.round(ST, 3), " exact [0.558 0.442 0.244]")
print(f"model evaluations used: {N * (k + 2):,}")
```

Output:

```
design (n=64, d=5)    discrepancy  min distance
random                    0.02574         0.095
Latin hypercube           0.00668         0.206
LHS, optimized            0.00211         0.277
scrambled Sobol           0.00288         0.145
n =   64: RMS error  random 0.11087  Latin hypercube 0.05063  scrambled Sobol 0.02647
n =  256: RMS error  random 0.05402  Latin hypercube 0.02439  scrambled Sobol 0.00608
n = 1024: RMS error  random 0.02729  Latin hypercube 0.01238  scrambled Sobol 0.00097
      random: 40 simulations, RMSE 0.0353, share of points near the transition 0.40
max variance: 40 simulations, RMSE 0.0213, share of points near the transition 0.25
first-order S_i: [0.313 0.443 0.   ]  exact [0.314 0.442 0.   ]
total S_Ti:      [0.557 0.442 0.243]  exact [0.558 0.442 0.244]
model evaluations used: 81,920
```

**What it shows.**

- Latin hypercube and Sobol designs are far more uniform than random points (discrepancy 4-12 times lower); optimized LHS spreads points best.
- For the mean estimate, LHS halves the random-sampling error, and scrambled Sobol converges much faster: at n = 1,024 its error is about 28 times smaller than random sampling's.
- Max-variance active learning reaches a lower error than random sampling with the same 40 simulations, but because GP variance ignores the observed values, it spreads points rather than concentrating them at the sharp transition.
- The Sobol indices match the exact values to three decimals. X₃ has no effect on its own (S₃ = 0) but a total index of 0.24 through its interaction with X₁: one-at-a-time sensitivity studies would wrongly conclude it can be frozen.

## In an interview

**Thirty-second answer to "Latin hypercube vs Sobol vs random sampling?"** "Random sampling leaves clumps and gaps at small n. A Latin hypercube stratifies every input, one sample per bin, so each variable is covered evenly; an optimized LHS (maximin or minimum discrepancy) also spreads points jointly, and it is my default initial design for a surrogate. A scrambled Sobol sequence is a low-discrepancy sequence with near 1/n integration error and it is extensible, so I use it when I may add simulations later or need integrals like Sobol indices. In all cases I add the baseline and corner designs engineers care about and check that every design is geometrically valid."

**Thirty-second answer to "How many simulations do you need?"** "There is no fixed number. I start with about ten per input dimension for a smooth response, then measure a learning curve, test error against training-set size on a fixed held-out set, and extrapolate on log-log axes. Regime changes, free-form geometry and field outputs push the number up; good features, symmetries, pretrained models and multi-fidelity data push it down."

**Likely follow-ups**

- *Which acquisition function would you use?* Depends on the goal: variance or ensemble disagreement for global accuracy, expected improvement for optimization, the misclassification probability for a pass/fail boundary, with a batch-diversity term on a cluster.
- *What is the difference between first-order and total Sobol indices?* First-order is the main effect alone; total includes all interactions; an input with a small first-order but large total index acts through interactions.
- *How do you compute Sobol indices for a 10-hour simulation?* On a surrogate trained on a space-filling design, with bootstrap confidence intervals.

## References

- M. D. McKay, R. J. Beckman and W. J. Conover, "A comparison of three methods for selecting values of input variables in the analysis of output from a computer code", *Technometrics*, 1979.
- J. L. Loeppky, J. Sacks and W. J. Welch, "Choosing the sample size of a computer experiment: a practical guide", *Technometrics*, 2009.
- A. Saltelli et al., *Global Sensitivity Analysis: The Primer*, Wiley, 2008, and "Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index", *Computer Physics Communications*, 2010.
- [scipy.stats.qmc (Latin hypercube, Sobol, discrepancy)](https://docs.scipy.org/doc/scipy/reference/stats.qmc.html)
- [SALib: sensitivity analysis library](https://github.com/SALib/SALib)
- [A Tutorial on Bayesian Optimization (Frazier)](https://arxiv.org/abs/1807.02811)
- [BoTorch](https://github.com/pytorch/botorch) and [Surrogate Modeling Toolbox (SMT)](https://github.com/SMTorg/smt)
