# Design: Drift Monitoring

[← Questions](../README.md) · [All answers](README.md) · [System design framework](system-design-framework.md)

**Question.** Design monitoring that detects when incoming designs drift outside the training data.

**Short answer.** Monitor at three levels:

1. **Per request: is this design OOD?** Compute a score from the model's geometry embedding (distance to the k nearest training designs), plus range checks on global features such as size, operating conditions and loads. The threshold is set from held-out in-distribution data.
2. **Per population: is the stream of designs shifting?** Track the distribution of OOD scores and features over time. Useful tools are the fraction of requests above the threshold, a two-sample test or maximum mean discrepancy (MMD) between recent embeddings and the training embeddings, and per-feature drift statistics. This catches a gradual move that no single request flags.
3. **Ground truth: is accuracy dropping?** Run the solver on a small random sample of requests, plus all flagged ones, to measure real error over time.

Alerts go to the team and trigger actions: route designs to the solver, run targeted simulations, retrain and re-validate. The detectors themselves are validated on deliberately shifted test sets.

**On this page**

- [What drift means here](#what-drift-means-here)
- [Per-request OOD scores](#per-request-ood-scores)
- [Population drift over time](#population-drift-over-time)
- [Ground-truth checks with the solver](#ground-truth-checks-with-the-solver)
- [Code: per-request scores and a drift alarm](#code-per-request-scores-and-a-drift-alarm)
- [Alerts and actions](#alerts-and-actions)
- [Architecture](#architecture)
- [In an interview](#in-an-interview)

## What drift means here

| Kind | Example | Detected by |
| --- | --- | --- |
| Covariate shift in geometry | A new design family; a styling trend toward sharper rear edges | Embedding distance, population tests |
| Shift in conditions | Higher speeds, new load cases, new materials | Range checks on inputs |
| Data-pipeline drift | Units changed upstream (mm instead of m); a new CAD export tool | Range checks, sudden jumps in feature distributions |
| Concept drift | The solver or set-up used as ground truth changes (new turbulence model) | Ground-truth checks, version tracking |

The model's error is not observed in production; drift detection uses the **inputs** (and the model's own uncertainty) as early warnings, and occasional solver runs as the real measure.

## Per-request OOD scores

- **Embedding distance:** the mean distance from the design's geometry embedding to its k nearest training embeddings, normalized by the typical distance within the training set ([Detecting an out-of-distribution geometry](out-of-distribution-and-generalization.md#detecting-an-out-of-distribution-geometry)).
- **Ensemble disagreement:** high variance across ensemble members is a complementary signal ([Deep ensembles](uncertainty-quantification.md#deep-ensembles)).
- **Global-feature range checks:** bounding-box dimensions, surface area, volume, operating conditions vs training ranges; cheap and interpretable.
- **Threshold:** a high quantile (for example, the 99th percentile) of the score on held-out in-distribution designs, so about 1% false alarms by construction.
- **Validate the detector:** measure how often it flags designs from a held-out family (recall) and whether flagged designs really have higher error ([Validating the detector itself](out-of-distribution-and-generalization.md#validating-the-detector-itself)).

## Population drift over time

- **Flag rate:** the fraction of requests over the threshold per day or week. Under no drift it stays near the false-alarm rate (about 1%); a sustained rise is drift.
- **Score distribution:** compare recent scores with the reference (Kolmogorov–Smirnov test, or a population stability index).
- **Embedding distribution:** MMD between a window of recent embeddings and a sample of training embeddings, with a permutation test for significance.
- **Per-feature monitors:** histograms of global features; sudden jumps point to pipeline problems (units), slow trends to design evolution.
- **Windows and multiple testing:** use rolling windows (for example 7 days) and require persistence (two consecutive windows) before alerting, to avoid noisy alarms.

## Ground-truth checks with the solver

- **Random audit sample:** run the solver on, for example, 1–2% of requests chosen at random; this gives an unbiased estimate of production error and catches drift the input monitors miss.
- **All flagged requests** go to the solver anyway (fallback); their errors show how bad OOD predictions were.
- **Error tracking** by slice (design family, condition range) and over time; compare with the validation report's numbers.
- These solver runs are also the most valuable new training data ([Active learning loop](design-active-learning-loop.md)).

## Code: per-request scores and a drift alarm

A synthetic example: designs are described by 8-dimensional embeddings. Production starts in-distribution, and from day 20 a growing share of designs comes from a new family, shifted in two directions.

```python
import numpy as np
from scipy.spatial import cKDTree
rng = np.random.default_rng(0)

d = 8
train = rng.normal(size=(2000, d))                         # training-design embeddings
holdout = rng.normal(size=(500, d))                         # held-out, in-distribution
tree = cKDTree(train)
def ood_score(Z, k=10):
    dist, _ = tree.query(Z, k=k)
    return dist.mean(axis=1)
threshold = np.quantile(ood_score(holdout), 0.99)          # about 1% false alarms by construction

def mmd2(X, Y, gamma=None):
    """Squared maximum mean discrepancy with an RBF kernel (biased estimate)."""
    Z = np.vstack([X, Y])
    sq = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    gamma = gamma or 1.0 / np.median(sq[sq > 0])
    K = np.exp(-gamma * sq)
    n = len(X)
    return K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean()

def mmd_pvalue(X, Y, n_perm=200):
    observed = mmd2(X, Y)
    Z = np.vstack([X, Y])
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(len(Z))
        count += mmd2(Z[p[:len(X)]], Z[p[len(X):]]) >= observed
    return (count + 1) / (n_perm + 1)

reference = train[rng.choice(len(train), 200, replace=False)]
shift = np.zeros(d); shift[:2] = 2.5                       # the new family differs in two directions
print(" day  share of new family  flag rate  MMD p-value")
for day in [0, 10, 20, 25, 30, 40]:
    share = 0.0 if day < 20 else min(0.05 * (day - 19), 0.5)
    n = 150                                                # requests that day
    new = rng.random(n) < share
    Z = rng.normal(size=(n, d)) + new[:, None] * shift
    flags = ood_score(Z) > threshold
    print(f"{day:4d}  {share:19.2f}  {flags.mean():9.3f}  {mmd_pvalue(Z, reference):11.3f}")
newfam = rng.normal(size=(300, d)) + shift
print(f"\nrecall on a pure new-family batch: {np.mean(ood_score(newfam) > threshold):.2f}; "
      f"false-alarm rate on held-out designs: {np.mean(ood_score(rng.normal(size=(3000, d))) > threshold):.3f}")
```

Output:

```
 day  share of new family  flag rate  MMD p-value
   0                 0.00      0.013        0.975
  10                 0.00      0.020        0.512
  20                 0.05      0.053        0.308
  25                 0.30      0.133        0.005
  30                 0.50      0.267        0.005
  40                 0.50      0.340        0.005

recall on a pure new-family batch: 0.52; false-alarm rate on held-out designs: 0.013
```

How to read it:

- **Before the shift** the flag rate stays near the designed 1% false-alarm rate, and the MMD test finds nothing.
- **At a 5% share,** neither signal is conclusive on one day's 150 requests. A small, early shift needs longer windows or more requests.
- **At a 30% share,** the MMD test is decisive: p = 0.005 is the smallest value 200 permutations can give. The flag rate is about ten times the false-alarm rate.
- **Per-request recall is only 0.52.** About half of the new-family designs lie close enough to training designs to pass the threshold. That number must be measured and reported, because those designs get no warning. It is also why ground-truth audits are needed. In a real system, the embeddings come from the model's geometry encoder, and the shift is a new design family or a pipeline change.

## Alerts and actions

| Signal | Likely cause | Action |
| --- | --- | --- |
| Single request flagged | An unusual design | Route to the solver; show a low-confidence warning |
| Sudden jump in a feature (size × 1,000) | Units or pipeline change upstream | Page the team; block or correct inputs |
| Gradual rise in flag rate or MMD | Design trend or a new family | Targeted simulations of the new region; retrain; re-validate |
| Rising audit error with no input drift | Concept drift (solver or set-up changed) or a model bug | Investigate the ground-truth pipeline; roll back if needed |

Every retrained model passes the regression suite and the validation protocol before release ([A regression test suite](production-and-deployment.md#a-regression-test-suite-for-a-surrogate), [Validation strategy](design-validation-strategy.md)).

## Architecture

```
request ─► features + embedding ─► OOD score + range checks ─► flag? ─► solver queue
                 │                                                          │
                 ▼                                                          ▼
       feature/score store ─► daily jobs: flag rate, KS/PSI, MMD ─► dashboard + alerts
                                                                            │
     random 1-2% audit ─► solver ─► error by slice over time ───────────────┘
                                          │
                                          ▼
                         new data ─► retrain ─► regression suite ─► release
```

## In an interview

**Thirty-second answer.** "Three layers. Per request, an OOD score from the model's geometry embedding, the distance to the nearest training designs, plus range checks on size and operating conditions, with a threshold set at the 99th percentile on held-out data; flagged designs go to the solver. Per population, I'd track the flag rate and a two-sample test such as MMD on recent embeddings against training, because a gradual shift shows up there before individual flags pile up. And for ground truth, I'd run the solver on a small random sample of requests to measure the real error over time. Alerts trigger targeted simulations and retraining, and I'd validate the detectors on deliberately shifted data."

**Likely follow-ups**

- *Why not just monitor the model's uncertainty?* Uncertainty estimates can themselves be wrong out of distribution; combine them with input-space checks and ground truth.
- *How do you set the alert threshold for population drift?* From the distribution of the statistic over past windows without drift, with persistence rules to limit false alarms.

## References

- A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schölkopf and A. Smola, "A kernel two-sample test", *JMLR*, 2012.
- S. Rabanser, S. Günnemann and Z. C. Lipton, "Failing loudly: an empirical study of methods for detecting dataset shift", NeurIPS 2019 ([arXiv:1810.11953](https://arxiv.org/abs/1810.11953)).
- [Out-of-distribution and generalization](out-of-distribution-and-generalization.md), [Production and deployment](production-and-deployment.md#monitoring-and-retraining)
