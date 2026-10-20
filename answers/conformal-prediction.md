# Conformal Prediction

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Conformal prediction turns **any** trained model into one that outputs **prediction intervals with a guaranteed coverage**, without assumptions about the model or the data distribution. The common **split conformal** recipe: hold out a calibration set the model never trained on, compute a nonconformity score on it (usually the absolute residual |y − ŷ|), take the ⌈(n + 1)(1 − α)⌉-th smallest score as q̂, and predict [ŷ − q̂, ŷ + q̂]. If calibration and test points are **exchangeable** (roughly, drawn from the same distribution), the interval contains the truth with probability at least 1 − α, for any sample size. The guarantee is **marginal**: on average over test points, not for each individual design, and it **breaks under distribution shift**, for example when a new design family arrives. Variants make intervals adaptive (normalized scores, conformalized quantile regression), handle known shifts (weighted conformal) and give guarantees per group (Mondrian conformal). For simulation surrogates it is a cheap, honest layer on top of an ensemble or GP.

**On this page**

- [The split conformal recipe](#the-split-conformal-recipe)
- [What is guaranteed and what is not](#what-is-guaranteed-and-what-is-not)
- [Adaptive intervals](#adaptive-intervals)
- [Distribution shift and group-wise guarantees](#distribution-shift-and-group-wise-guarantees)
- [Conformal prediction for fields and integrated quantities](#conformal-prediction-for-fields-and-integrated-quantities)
- [Code: split conformal intervals](#code-split-conformal-intervals)
- [In an interview](#in-an-interview)

## The split conformal recipe

1. **Split** the labelled data into a training set and a **calibration set** of n points (a few hundred to a few thousand). Train the model on the training set only.
2. **Score** each calibration point with a nonconformity score, for regression usually sᵢ = |yᵢ − ŷ(xᵢ)|.
3. **Quantile:** sort the scores and take q̂ = the ⌈(n + 1)(1 − α)⌉-th smallest (the "+1" is the finite-sample correction).
4. **Predict** for a new input: C(x) = [ŷ(x) − q̂, ŷ(x) + q̂].

**Why it works, in one line.** If the test point is exchangeable with the calibration points, its score is equally likely to fall at any rank among the n + 1 scores, so the probability that it exceeds the ⌈(n + 1)(1 − α)⌉-th smallest calibration score is at most α.

**Cost:** one model evaluation per calibration point and a sort. Nothing is retrained. Full (transductive) conformal prediction and cross-conformal / jackknife+ use the data more efficiently at a higher computational cost.

## What is guaranteed and what is not

**Guaranteed** (under exchangeability): P(y_test ∈ C(x_test)) ≥ 1 − α, averaged over the random calibration set and test point, for any model and any sample size. With continuous scores, coverage is also at most 1 − α + 1/(n + 1), so it is not wildly conservative.

**Not guaranteed.**

1. **Conditional coverage.** The 90% holds on average; for a particular region of the design space (or one specific design), coverage can be much lower or higher. The code below shows coverage of 1.00 at low x and 0.74 at high x, averaging to 0.90.
2. **Coverage under distribution shift.** A new design family, new operating conditions, a solver change: the calibration set no longer represents the test data, and coverage drops (0.71 in the code below).
3. **Small intervals.** Conformal prediction makes intervals honest, not small; a poor model gets wide intervals.
4. **Correctness of the labels.** Coverage is of the *simulator's* output; if the simulation is wrong, the interval covers the wrong answer. See [When the training data has its own errors](uncertainty-quantification.md#when-the-training-data-has-its-own-errors).
5. **Repeated use of the same calibration set** for model selection: tuning on the calibration set breaks the guarantee; keep it untouched.

## Adaptive intervals

The plain absolute-residual score gives intervals of the **same width everywhere**, too wide where the model is accurate and too narrow where it is not.

- **Normalized scores:** s = |y − ŷ| / σ̂(x), with σ̂ from an ensemble spread, a heteroscedastic head or a GP; the interval is ŷ ± q̂ σ̂(x). Width now adapts to predicted difficulty, and coverage is still guaranteed marginally. Conditional coverage is only as good as σ̂: if σ̂ ignores a systematic model error, normalized intervals undercover there (the code below shows exactly that at low x).
- **Conformalized quantile regression (CQR; Romano, Patterson and Candès, 2019):** train a model to predict lower and upper quantiles, then conformalize them with the score max(q̂_lo(x) − y, y − q̂_hi(x)). Adaptive and often sharper.
- **Locally weighted or localized conformal:** weight calibration points by similarity to the test point.

## Distribution shift and group-wise guarantees

- **Weighted conformal prediction (Tibshirani, Barber, Candès and Ramdas, 2019):** if the input distribution changes in a known way (covariate shift with a known or estimated likelihood ratio), reweight the calibration scores to restore coverage.
- **Mondrian (group-conditional) conformal:** calibrate separately per group (design family, operating regime, mesh type) to guarantee coverage within each group, at the cost of smaller calibration sets per group.
- **Adaptive conformal inference** for time series and drift: update q̂ online from observed misses.
- **Monitoring:** in production, track the empirical coverage on cases later verified by the solver; a drop is a drift alarm. See [Monitoring and retraining](production-and-deployment.md#monitoring-and-retraining).
- **Out-of-distribution inputs:** conformal intervals say nothing reliable about inputs unlike the calibration data; combine with an input-distance check and fall back to the solver. See [Detecting an out-of-distribution geometry](out-of-distribution-and-generalization.md#detecting-an-out-of-distribution-geometry).

## Conformal prediction for fields and integrated quantities

- **Scalar quantities of interest** (drag, peak stress): apply conformal prediction directly to them; this is usually what the decision needs.
- **Whole fields:** a per-point interval at 90% does not mean the whole field is inside its band 90% of the time. For a simultaneous band, use a field-level score such as s = max over points of |y − ŷ| / σ̂ (the band covers the entire field with probability 1 − α), or a score on a norm of the error.
- **Calibration data is simulations,** so keep a calibration set of designs (split by design, not by point) aside from training and validation. See [Splitting simulation data](data-splits-and-leakage.md#splitting-simulation-data).

## Code: split conformal intervals

A synthetic surrogate setting: the true response is sin(4x) with noise that grows with x; the "trained model" is slightly biased, and a separate σ̂(x) stands in for an ensemble spread. The checks: coverage at three levels, absolute vs normalized scores in two regions, and what a covariate shift does.

```python
import numpy as np

def split_conformal(y_cal, yhat_cal, yhat_test, alpha=0.1, sigma_cal=None, sigma_test=None):
    """Split conformal intervals. With sigma_* given, scores are normalized |y - yhat| / sigma (adaptive width)."""
    s_cal = np.ones_like(y_cal) if sigma_cal is None else sigma_cal
    s_test = np.ones_like(yhat_test) if sigma_test is None else sigma_test
    scores = np.abs(y_cal - yhat_cal) / s_cal
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))                       # finite-sample correction
    q = np.inf if k > n else np.sort(scores)[k - 1]               # too few calibration points: infinite interval
    return yhat_test - q * s_test, yhat_test + q * s_test

rng = np.random.default_rng(0)
def simulate(n, shift=0.0):
    x = rng.random(n) + shift
    noise = 0.02 + 0.2 * x                                        # heteroscedastic: noisier at large x
    return x, np.sin(4 * x) + noise * rng.normal(size=n)
model = lambda x: np.sin(4 * x) + 0.05 * np.cos(9 * x)            # a slightly wrong trained model
sigma_hat = lambda x: 0.02 + 0.2 * x                              # e.g. a heteroscedastic head or ensemble spread

x_cal, y_cal = simulate(1000)                                     # calibration set, never used for training
x_te, y_te = simulate(20000)
covered = lambda y, lo, hi: np.mean((y >= lo) & (y <= hi))
for alpha in [0.2, 0.1, 0.05]:
    lo, hi = split_conformal(y_cal, model(x_cal), model(x_te), alpha)
    print(f"alpha = {alpha:.2f}: target coverage {1 - alpha:.2f}, observed {covered(y_te, lo, hi):.3f}")

for name, s_cal, s_te in [("absolute scores", None, None), ("normalized scores", sigma_hat(x_cal), sigma_hat(x_te))]:
    lo, hi = split_conformal(y_cal, model(x_cal), model(x_te), 0.1, s_cal, s_te)
    low, high = x_te < 0.2, x_te > 0.8
    print(f"{name:>17}: overall {covered(y_te, lo, hi):.3f} | x < 0.2: coverage {covered(y_te[low], lo[low], hi[low]):.3f}, "
          f"width {np.mean(hi[low] - lo[low]):.3f} | x > 0.8: coverage {covered(y_te[high], lo[high], hi[high]):.3f}, "
          f"width {np.mean(hi[high] - lo[high]):.3f}")

x_sh, y_sh = simulate(20000, shift=0.5)                           # deployment drift: x now in [0.5, 1.5]
lo, hi = split_conformal(y_cal, model(x_cal), model(x_sh), 0.1)
print(f"after a covariate shift (x in [0.5, 1.5]): coverage {covered(y_sh, lo, hi):.3f}")
lo, hi = split_conformal(y_cal, model(x_cal), model(x_sh), 0.1, sigma_hat(x_cal), sigma_hat(x_sh))
print(f"same shift, normalized scores:             coverage {covered(y_sh, lo, hi):.3f}")
```

Output:

```
alpha = 0.20: target coverage 0.80, observed 0.814
alpha = 0.10: target coverage 0.90, observed 0.895
alpha = 0.05: target coverage 0.95, observed 0.945
  absolute scores: overall 0.895 | x < 0.2: coverage 1.000, width 0.455 | x > 0.8: coverage 0.742, width 0.455
normalized scores: overall 0.901 | x < 0.2: coverage 0.777, width 0.153 | x > 0.8: coverage 0.946, width 0.765
after a covariate shift (x in [0.5, 1.5]): coverage 0.712
same shift, normalized scores:             coverage 0.940
```

**What it shows.**

- Marginal coverage matches the target at every level (0.81, 0.90, 0.95), with only 1,000 calibration points and a deliberately imperfect model.
- **Absolute scores** give one width everywhere: coverage is 1.00 where the noise is small and 0.74 where it is large. The 90% is only an average.
- **Normalized scores** fix the high-noise region (0.95) with wider intervals there and narrower ones elsewhere, but undercover at small x (0.78), where the model's systematic bias, not the noise σ̂ describes, dominates. Adaptive intervals are only as good as the scale estimate.
- **Under a covariate shift**, the absolute intervals cover 71% instead of 90%. The normalized intervals happen to hold up (0.94) because σ̂ here is the exact noise model and extrapolates correctly; with a learned σ̂ that extrapolates poorly, they would fail too. Never assume the guarantee survives a shift: monitor coverage.

## In an interview

**Thirty-second answer to "What does conformal prediction guarantee?"** "Given a held-out calibration set that is exchangeable with the test data, split conformal takes the right quantile of the calibration residuals and adds it around the prediction, and the interval then covers the truth with probability at least 1 minus alpha, for any model and any sample size. The catch is that the guarantee is marginal: on average over test points, not for each design, and it breaks under distribution shift such as a new design family. So I use normalized or quantile-based scores for adaptive width, calibrate per design family if needed, and monitor coverage in production."

**Likely follow-ups**

- *Why the (n + 1)?* The test point is one more exchangeable score; the correction makes the guarantee exact for finite n.
- *How do you get intervals that adapt to difficulty?* Normalize the score by an uncertainty estimate, or use conformalized quantile regression.
- *What about a whole pressure field?* Use a field-level score (the maximum normalized error) for a simultaneous band, or conformalize the integrated quantity that drives the decision.

## References

- [A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification (Angelopoulos & Bates)](https://arxiv.org/abs/2107.07511)
- V. Vovk, A. Gammerman and G. Shafer, *Algorithmic Learning in a Random World*, Springer, 2005.
- [Distribution-Free Predictive Inference for Regression (Lei et al.)](https://arxiv.org/abs/1604.04173)
- [Conformalized Quantile Regression (Romano, Patterson & Candès)](https://arxiv.org/abs/1905.03222)
- [Conformal Prediction Under Covariate Shift (Tibshirani et al.)](https://arxiv.org/abs/1904.06019)
- [MAPIE](https://github.com/scikit-learn-contrib/MAPIE)
