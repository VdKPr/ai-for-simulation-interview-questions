# Uncertainty Quantification

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A surrogate prediction is only usable for engineering decisions if it says **how wrong it might be**. **Aleatoric** uncertainty is variability in the data itself (measurement noise, scatter of time-averaged unsteady results, material scatter); more data does not reduce it. **Epistemic** uncertainty is the model's lack of knowledge (too few simulations near the query, extrapolation to a new design family, model misspecification); more data, in the right place, does reduce it. For deterministic simulation surrogates epistemic uncertainty usually dominates, and it grows away from the training data. **Deep ensembles** (several networks trained from different random initializations, their spread used as the uncertainty) are a strong, simple baseline; **Monte Carlo dropout** is cheaper but usually underestimates uncertainty; Gaussian processes and conformal prediction are the other workhorses. No method is trustworthy until its **calibration** has been checked on held-out and out-of-distribution data: in the code below, both an ensemble and MC dropout give 90% intervals that cover only 49% and 16% of the truth just beyond the training range. Uncertainty becomes useful when it drives a decision: accept a design only if a conservative bound passes, otherwise run the solver.

**On this page**

- [Aleatoric vs epistemic uncertainty](#aleatoric-vs-epistemic-uncertainty)
- [Deep ensembles](#deep-ensembles)
- [Monte Carlo dropout](#monte-carlo-dropout)
- [Other methods](#other-methods)
- [Checking calibration](#checking-calibration)
- [UQ methods in scientific ML and how they are compared](#uq-methods-in-scientific-ml-and-how-they-are-compared)
- [When the training data has its own errors](#when-the-training-data-has-its-own-errors)
- [From uncertainty to a design decision](#from-uncertainty-to-a-design-decision)
- [Code: ensembles, MC dropout and a calibration check](#code-ensembles-mc-dropout-and-a-calibration-check)
- [In an interview](#in-an-interview)

## Aleatoric vs epistemic uncertainty

| | Aleatoric | Epistemic |
| --- | --- | --- |
| Meaning | Inherent variability: the same input gives different outputs | Lack of knowledge about the true input-output map |
| Reduced by more data? | No | Yes, if the data is where the uncertainty is |
| In simulation surrogates | Time-averaging noise in unsteady CFD, incomplete convergence, mesh-dependent labels near singularities, scatter in test data | Few simulations near the query, new geometries or conditions, model misspecification |
| How it is modelled | A predicted variance (heteroscedastic head), noise term of a GP | Disagreement between ensemble members, GP posterior variance, Bayesian posteriors |
| Behaviour far from data | Should stay at the local noise level | Should grow |

**Why the distinction matters.** If the uncertainty is aleatoric, running more simulations of the same kind will not help, and the decision must tolerate the scatter. If it is epistemic, the next simulation should go where the uncertainty is largest (active learning; see [Active learning and acquisition functions](design-of-experiments-and-active-learning.md#active-learning-and-acquisition-functions)), and a prediction with high epistemic uncertainty should be sent to the solver.

**Total predictive uncertainty** with an ensemble of M models, each predicting a mean μ_m and a variance σ_m²:

```math
\operatorname{Var}[y] \approx \underbrace{\frac1M \sum_m \sigma_m^2}_{\text{aleatoric}} \;+\; \underbrace{\frac1M \sum_m (\mu_m - \bar\mu)^2}_{\text{epistemic}}
```

**Input uncertainty** is a third category for engineering: manufacturing tolerances, material scatter, operating conditions. It is propagated by sampling inputs and evaluating the surrogate many times (Monte Carlo), which is one of the main reasons to build a surrogate at all.

## Deep ensembles

**Method (Lakshminarayanan, Pritzel and Blundell, NeurIPS 2017).** Train M networks (typically 5) on the same data with different random initializations (and data shuffling), each with a mean and variance output and a Gaussian negative log-likelihood loss. Predict with the mixture: the average mean, and the variance decomposition above. The paper also used adversarial training as an optional smoothing.

**Why it works.** Different initializations converge to different solutions that agree where the data constrains them and disagree where it does not, so their spread tracks epistemic uncertainty. Empirically, ensembles are among the best-calibrated practical methods and degrade more gracefully under distribution shift than single-model alternatives (Ovadia et al., 2019, compared methods under dataset shift).

**Costs and tips.**

- M times the training and inference cost; for large surrogates, use fewer members, snapshot ensembles (checkpoints from one cyclic-learning-rate run), or ensembles of only the last layers.
- Diversity matters: different initializations are usually enough; bootstrapping the data adds diversity but uses less data per member.
- Ensembles still **underestimate uncertainty far from the data**: all members can extrapolate the same wrong way. They are not an out-of-distribution guarantee; see the code below.

## Monte Carlo dropout

**Method (Gal and Ghahramani, ICML 2016).** Train a network with dropout, keep dropout **active at test time**, run T stochastic forward passes, and use the mean and spread of the outputs. They showed that dropout training approximates variational inference in a Bayesian neural network (with a particular approximate posterior), so the sample spread approximates epistemic uncertainty.

**Strengths.** Nearly free if the model already uses dropout; one model; T forward passes at inference.

**Weaknesses.**

- The uncertainty depends on the dropout rate, a hyperparameter chosen for regularization, not calibration.
- It tends to **underestimate** uncertainty, especially out of distribution; the approximate posterior is crude.
- Dropout is uncommon in modern operators and transformers for physics; adding it can hurt accuracy.

In practice MC dropout is a cheap first estimate; ensembles are the better default when compute allows.

## Other methods

| Method | Idea | Notes |
| --- | --- | --- |
| Gaussian processes | Exact Bayesian posterior for a kernel model | The gold standard for small data. See [Gaussian processes](gaussian-processes.md) |
| Conformal prediction | Calibrate intervals with held-out residuals; distribution-free coverage | Wraps any model. See [Conformal prediction](conformal-prediction.md) |
| Heteroscedastic heads | Predict mean and variance with a Gaussian NLL | Aleatoric only, unless combined with an ensemble |
| Laplace approximation | Gaussian posterior around the trained weights from the Hessian | Post-hoc; last-layer versions are cheap |
| Variational Bayesian networks, HMC | Approximate or sample the weight posterior | Principled, harder to scale |
| Evidential regression | Predict parameters of a higher-order distribution | Single pass; calibration is debated |
| Quantile regression | Predict quantiles directly with the pinball loss | Simple intervals; no decomposition |
| Generative models | Sample whole fields (diffusion, flow matching) | Uncertainty for fields and chaotic dynamics. See [Generative models for physics](generative-models-for-physics.md) |

## Checking calibration

A model is **calibrated** if its stated uncertainty matches its actual errors: 90% intervals contain the truth about 90% of the time, and errors divided by predicted standard deviations look like a unit Gaussian.

1. **Coverage at several levels.** For 50%, 80%, 90% and 95% intervals, compute the fraction of held-out points inside. Plot observed against nominal coverage (a reliability diagram); the diagonal is perfect calibration, below it is overconfident.
2. **Standardized errors.** z = (y − μ)/σ should have mean ≈ 0 and variance ≈ 1; a variance above 1 means overconfidence.
3. **Proper scoring rules.** Negative log-likelihood and the continuous ranked probability score (CRPS) reward both calibration and sharpness; use them to compare UQ methods.
4. **Sharpness.** Intervals that are calibrated but wide are useless; report average width next to coverage.
5. **Calibration by subgroup and out of distribution.** Average coverage can hide failures: check it per design family, per operating regime, and on a deliberately out-of-distribution split. Marginal calibration on an in-distribution test set says little about a new design family.
6. **Error-uncertainty correlation.** Do the points with the largest predicted uncertainty have the largest errors? A rank correlation or an "error retention curve" (error of the most confident x% of predictions) answers it.
7. **Recalibration.** If miscalibrated, scale σ by a factor fitted on a calibration set (temperature scaling for regression), apply isotonic regression to the cumulative probabilities, or wrap the model with conformal prediction.

## UQ methods in scientific ML and how they are compared

Psaros, Meng, Zou, Guo and Karniadakis (Journal of Computational Physics, 2023) surveyed and compared UQ methods for scientific machine learning in one framework.

- **Sources of uncertainty they distinguish:** noise in the data (aleatoric), limited data and model choice (epistemic), neural-network hyperparameters, overparametrization, optimization and sampling errors, and model misspecification, including uncertainty in the physics model used in physics-informed training.
- **Method families compared:** Bayesian neural networks with Hamiltonian Monte Carlo, Langevin dynamics, variational inference, Laplace approximations, MC dropout, deep ensembles and functional priors, applied to function approximation, PDE solving (physics-informed networks), operator learning, problems with mixed inputs and outputs, and high-dimensional stochastic problems.
- **Evaluation:** accuracy of the mean, calibration and sharpness metrics, and post-hoc improvement (recalibration).
- **Practical takeaways:** no method dominates; HMC is accurate but expensive; ensembles are robust and simple; the evaluation must include calibration, not just accuracy; and the team released the open-source library NeuralUQ.

**How to compare UQ methods yourself:** same model backbone and data splits, then report accuracy (relative L2), NLL or CRPS, coverage at several levels, interval width, and the same on an out-of-distribution split, over several seeds. See [Evaluation metrics](evaluation-metrics.md).

## When the training data has its own errors

The labels of a simulation surrogate are not ground truth. Sources of label error:

1. **Discretization error:** a different mesh would give a different answer, most visibly near singularities and in under-resolved regions. See [When the labels have discretization error](verification-validation-and-mesh-convergence.md#when-the-labels-have-discretization-error).
2. **Iterative convergence error:** residuals not fully converged; quantities still drifting.
3. **Time-averaging error:** statistics of an unsteady (LES, DES) simulation averaged over a finite window carry sampling noise that behaves like aleatoric noise.
4. **Modelling error:** RANS turbulence models, contact or material models, boundary conditions that do not match reality. This error is systematic and the surrogate inherits it entirely.
5. **Failed or inconsistent runs:** different solver versions, settings or mesh strategies across the dataset.

**What to do.**

- **Quantify it** where possible: mesh convergence (GCI) on a subset, statistical uncertainty of time averages (batch means), repeated runs with perturbed settings. See [Mesh convergence and the grid convergence index](verification-validation-and-mesh-convergence.md#mesh-convergence-and-the-grid-convergence-index).
- **Model it as noise:** a GP noise term or a heteroscedastic head can absorb label noise so the model does not overfit it; aiming for an error below the label noise is pointless.
- **Set realistic targets:** the surrogate's error cannot meaningfully be smaller than the label error; report both.
- **Separate surrogate uncertainty from model uncertainty:** the surrogate's error bar covers "surrogate vs solver". "Solver vs reality" needs validation against experiments and must be communicated separately.
- **Clean the data:** remove or fix unconverged and inconsistent runs, and record solver versions and settings with each sample. See [Generating simulation training data](generating-simulation-training-data.md).

## From uncertainty to a design decision

1. **Propagate to the quantity of interest.** Decisions use drag, peak stress, maximum temperature, not pointwise fields. With an ensemble, compute the quantity for each member and take its spread; do not combine pointwise standard deviations, because field errors are correlated.
2. **Use a conservative bound, not the mean.** Accept a design if the pessimistic bound passes: for example, the lower 95% bound of the predicted safety factor exceeds the requirement, or the upper bound of peak temperature is below the limit.
3. **Three-way decision.** Pass (bound clears the limit), fail (the optimistic bound fails), or **uncertain: run the solver**. This is the basis of a surrogate-with-fallback system. See [When to fall back to the full solver](out-of-distribution-and-generalization.md#when-to-fall-back-to-the-full-solver) and [Predicting a safety factor with a surrogate](von-mises-stress-and-yield.md#predicting-a-safety-factor-with-a-surrogate).
4. **Account for out-of-distribution inputs separately.** Because uncertainty estimates are least reliable exactly where they are most needed, combine them with an input-distance check. See [Detecting an out-of-distribution geometry](out-of-distribution-and-generalization.md#detecting-an-out-of-distribution-geometry).
5. **Optimization:** use uncertainty-aware objectives (lower confidence bound, expected improvement, or reliability constraints) so the optimizer does not exploit regions where the surrogate is unsure. See [Why optimizers exploit surrogates](surrogate-based-optimization.md#why-optimizers-exploit-surrogates).
6. **Cost of errors:** choose confidence levels from the asymmetric cost of a false pass (a field failure) vs a false fail (an extra simulation). Safety-critical gates warrant high confidence plus solver verification.
7. **Communicate it** in the engineer's terms: "drag 0.287 ± 0.004 (95%), inside the training range" or "outside the validated range: run CFD".

## Code: ensembles, MC dropout and a calibration check

A 1D regression problem built to separate the two kinds of uncertainty: y = sin(3x) plus noise whose standard deviation grows with |x| (aleatoric), with training data on [−1, −0.2] ∪ [0.2, 1], so there is a gap and an extrapolation region (epistemic). A five-member deep ensemble and an MC-dropout model, both with Gaussian negative-log-likelihood heads, are compared in three regions, then the ensemble's calibration is checked at several confidence levels. Training takes under a minute.

```python
import torch, torch.nn as nn

torch.manual_seed(0)
f = lambda x: torch.sin(3 * x)
noise_sd = lambda x: 0.05 + 0.1 * x.abs()                          # aleatoric noise that grows with |x|
x_tr = torch.cat([torch.rand(300, 1) * 0.8 - 1.0, torch.rand(300, 1) * 0.8 + 0.2])   # a gap in (-0.2, 0.2)
y_tr = f(x_tr) + noise_sd(x_tr) * torch.randn_like(x_tr)

class GaussianMLP(nn.Module):
    """Predicts a mean and a log-variance; trained with the Gaussian negative log-likelihood."""
    def __init__(self, p_drop=0.0):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Dropout(p_drop),
                                  nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p_drop))
        self.mean, self.logvar = nn.Linear(128, 1), nn.Linear(128, 1)
    def forward(self, x):
        h = self.body(x)
        return self.mean(h), self.logvar(h).clamp(-10, 5)

def train(model, steps=3000):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        mu, logvar = model(x_tr)
        loss = (0.5 * (logvar + (y_tr - mu) ** 2 / logvar.exp())).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model

ensemble = []
for seed in range(5):                                              # deep ensemble: 5 random initializations
    torch.manual_seed(seed)
    ensemble.append(train(GaussianMLP()))
torch.manual_seed(10)
dropout_model = train(GaussianMLP(p_drop=0.1))

def decompose(outputs):                                            # mixture mean, epistemic and aleatoric variance
    mus = torch.stack([o[0] for o in outputs]); variances = torch.stack([o[1].exp() for o in outputs])
    return mus.mean(0), mus.var(0, unbiased=False), variances.mean(0)

def ensemble_predict(x):
    with torch.no_grad():
        return decompose([m(x) for m in ensemble])

def dropout_predict(x, samples=100):
    dropout_model.train()                                          # keep dropout active at test time
    with torch.no_grad():
        return decompose([dropout_model(x) for _ in range(samples)])

regions = {"training range": torch.cat([torch.rand(2000, 1) * 0.8 - 1.0, torch.rand(2000, 1) * 0.8 + 0.2]),
           "gap (-0.2, 0.2)": torch.rand(2000, 1) * 0.4 - 0.2,
           "beyond (1, 1.5)": torch.rand(2000, 1) * 0.5 + 1.0}
for name, predict in [("deep ensemble", ensemble_predict), ("MC dropout", dropout_predict)]:
    print(name)
    for region, x in regions.items():
        y = f(x) + noise_sd(x) * torch.randn_like(x)
        mu, epistemic, aleatoric = predict(x)
        sd = (epistemic + aleatoric).sqrt()
        coverage = ((y - mu).abs() <= 1.645 * sd).float().mean().item()     # nominal 90% interval
        print(f"  {region:<16} epistemic sd {epistemic.sqrt().mean():.3f}  aleatoric sd {aleatoric.sqrt().mean():.3f}  "
              f"90% coverage {coverage:.2f}  RMSE {((mu - f(x)) ** 2).mean().sqrt():.3f}")

# Calibration check of the ensemble on fresh in-distribution data: observed vs nominal coverage.
x = regions["training range"]
y = f(x) + noise_sd(x) * torch.randn_like(x)
mu, epistemic, aleatoric = ensemble_predict(x)
z = ((y - mu) / (epistemic + aleatoric).sqrt()).abs()
for nominal, quantile in [(0.5, 0.674), (0.8, 1.282), (0.95, 1.960)]:
    print(f"nominal {nominal:.0%} interval: observed coverage {(z <= quantile).float().mean():.3f}")
print(f"variance of standardized errors: {(((y - mu) ** 2) / (epistemic + aleatoric)).mean():.2f} (calibrated: 1)")
```

Output:

```
deep ensemble
  training range   epistemic sd 0.003  aleatoric sd 0.105  90% coverage 0.87  RMSE 0.023
  gap (-0.2, 0.2)  epistemic sd 0.014  aleatoric sd 0.071  90% coverage 0.94  RMSE 0.010
  beyond (1, 1.5)  epistemic sd 0.049  aleatoric sd 0.118  90% coverage 0.49  RMSE 0.227
MC dropout
  training range   epistemic sd 0.051  aleatoric sd 0.117  90% coverage 0.93  RMSE 0.027
  gap (-0.2, 0.2)  epistemic sd 0.066  aleatoric sd 0.103  90% coverage 0.98  RMSE 0.063
  beyond (1, 1.5)  epistemic sd 0.079  aleatoric sd 0.115  90% coverage 0.16  RMSE 0.596
nominal 50% interval: observed coverage 0.476
nominal 80% interval: observed coverage 0.778
nominal 95% interval: observed coverage 0.937
variance of standardized errors: 1.11 (calibrated: 1)
```

**What it shows.**

- **The decomposition behaves as theory says.** The ensemble's epistemic spread is tiny where data is dense (0.003), larger in the gap (0.014) and about 16 times larger beyond the data (0.049), while its aleatoric estimate follows the noise level, which grows with |x|.
- **In distribution, the ensemble is reasonably calibrated:** observed coverage 0.48, 0.78, 0.94 for nominal 50%, 80%, 95%, and standardized errors with variance 1.11, slightly overconfident.
- **The gap was easy:** sin(3x) is smooth across it, so both models interpolate well and cover the truth.
- **Beyond the data, both fail, and MC dropout fails badly.** The ensemble's error grows tenfold (RMSE 0.227) but its uncertainty only about doubles, so its 90% intervals cover 49% of the truth; MC dropout's cover 16%. Every member extrapolates in a similar wrong direction, so their disagreement understates the error. This is why uncertainty estimates are combined with an explicit out-of-distribution check and a fallback to the solver.

## In an interview

**Thirty-second answer to "Aleatoric vs epistemic uncertainty?"** "Aleatoric is variability in the data itself, like time-averaging noise in an LES result or material scatter; more data of the same kind does not remove it. Epistemic is the model not knowing, because there were too few simulations near this input or it is a new design family; more data in the right place does remove it. For simulation surrogates epistemic usually dominates. An ensemble separates them: the average predicted variance is aleatoric, the spread of the member means is epistemic. And I check calibration out of distribution, because that is where every method is weakest."

**Likely follow-ups**

- *Ensembles or MC dropout?* Ensembles: better calibrated and more robust, at M times the cost; MC dropout is cheap but tends to be overconfident.
- *How do you check calibration?* Coverage at several levels on held-out and out-of-distribution splits, standardized errors, NLL or CRPS, and interval width.
- *How does uncertainty become a decision?* Propagate it to the quantity of interest, gate on a conservative bound, and send uncertain or out-of-distribution cases to the solver.

## References

- [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles (Lakshminarayanan, Pritzel & Blundell)](https://arxiv.org/abs/1612.01474)
- [Dropout as a Bayesian Approximation (Gal & Ghahramani)](https://arxiv.org/abs/1506.02142)
- [Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift (Ovadia et al.)](https://arxiv.org/abs/1906.02530)
- [Uncertainty Quantification in Scientific Machine Learning: Methods, Metrics, and Comparisons (Psaros et al.)](https://arxiv.org/abs/2201.07766) and [NeuralUQ](https://github.com/Crunch-UQ4MI/neuraluq)
- [A Review of Uncertainty Quantification in Deep Learning (Abdar et al.)](https://arxiv.org/abs/2011.06225)
- [Accurate Uncertainties for Deep Learning Using Calibrated Regression (Kuleshov, Fenner & Ermon)](https://arxiv.org/abs/1807.00263)
