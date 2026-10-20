# Multi-Fidelity Learning

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Combine **many cheap, approximate simulations** (low fidelity: coarse meshes, RANS, 2D sections, linear FEA, panel methods) with **a few expensive, accurate ones** (high fidelity: fine meshes, LES, 3D, nonlinear FEA, experiments). The model learns the broad trend from the cheap data and a **correction** from the expensive data. The classic method is **co-kriging** in Kennedy and O'Hagan's autoregressive form, y_H(x) = ρ y_L(x) + δ(x), with Gaussian-process priors on the low-fidelity response and on the discrepancy δ. Neural versions use **residual learning**, the low-fidelity prediction as an **extra input** (for nonlinear correlations), or **pretraining on low fidelity and fine-tuning on high fidelity**. It works when the fidelities are strongly correlated where it matters; it fails when the cheap model misses the physics that decides the answer, such as separation, transition or a shock. Recent scaling studies of CFD surrogates found that, under a fixed data-generation budget, a mix of fidelities beats spending everything on high fidelity, with savings of about 3× on average and up to 6.3×.

**On this page**

- [Fidelity levels in practice](#fidelity-levels-in-practice)
- [Methods](#methods)
- [Allocating the simulation budget](#allocating-the-simulation-budget)
- [When multi-fidelity fails](#when-multi-fidelity-fails)
- [Digital twins: simulation plus measurements](#digital-twins-simulation-plus-measurements)
- [Code: co-kriging with 4 expensive and 11 cheap runs](#code-co-kriging-with-4-expensive-and-11-cheap-runs)
- [In an interview](#in-an-interview)

## Fidelity levels in practice

| Low fidelity | High fidelity | Typical cost ratio |
| --- | --- | --- |
| Coarse mesh, loose convergence tolerance | Fine mesh, tight tolerance | 10-100× |
| RANS | LES or hybrid RANS-LES | 10-100× |
| 2D section, axisymmetric model | Full 3D | 100× or more |
| Potential flow, panel method, vortex lattice | Viscous CFD | 1,000× |
| Linear static FEA | Nonlinear FEA (plasticity, contact, large deformation) | 10-100× |
| Beam or shell model | Solid model | 10-100× |
| Simulation | Test or experiment | Hours vs weeks, and the test may be the only "truth" |

Fidelity is also a continuum (mesh size, time step, iteration count), which recent work exploits by treating it as a tunable input rather than two levels.

## Methods

**1. Co-kriging, autoregressive (AR1) form (Kennedy and O'Hagan, 2000).**

```math
y_H(\mathbf x) = \rho\, y_L(\mathbf x) + \delta(\mathbf x), \qquad y_L \sim \mathcal{GP}, \quad \delta \sim \mathcal{GP}
```

- Fit a GP to the plentiful low-fidelity data; then fit ρ and a second GP to the discrepancy δ using the few high-fidelity points (usually at locations where the low fidelity was also run, a nested design).
- Works well when the relationship between fidelities is roughly linear and the discrepancy is smoother or simpler than y_H itself.
- Gives uncertainty for free. Extends to several levels and to recursive formulations that scale better.

**2. Nonlinear autoregressive GPs (NARGP, Perdikaris et al., 2017).** Use the low-fidelity output as an **input** to the high-fidelity GP, y_H(x) = g(x, y_L(x)), so the correlation can be nonlinear and vary across the space.

**3. Residual (delta) learning with networks.** Train a network on y_H − y_L (or on y_H − ρ y_L) using the paired designs. Often the correction is smaller and smoother than the field itself.

**4. Composite multi-fidelity networks (Meng and Karniadakis, 2020).** A low-fidelity network, then a high-fidelity network with separate **linear** and **nonlinear** sub-networks mapping (x, y_L) to y_H, so the model learns which kind of correlation applies.

**5. Transfer learning.** Pretrain a field surrogate on thousands of cheap simulations; fine-tune all or part of it on tens of expensive ones. This is the dominant pattern for deep geometric surrogates, and the same idea as fine-tuning a foundation model. See [Fine-tuning on a small customer dataset](foundation-models-for-physics.md#fine-tuning-on-a-small-customer-dataset).

**6. Fidelity as an input.** Train one model on all data with the fidelity (mesh size, model type) as a conditioning input, and query it at the highest fidelity. A 2026 study (Setinek, Paischer, Thuerey, Brandstetter and Koehler) showed that neural surrogates learn the fidelity of their training data and can be steered across the fidelity spectrum at inference.

**7. Multi-fidelity Bayesian optimization.** The acquisition function also chooses **which fidelity** to run next, weighing information gained against cost (for example, multi-fidelity knowledge gradient or max-value entropy search). See [Bayesian optimization](bayesian-optimization.md).

## Allocating the simulation budget

1. **Run a small paired set first:** both fidelities on the same 10-20 designs, to measure the correlation and the discrepancy's complexity before committing the budget.
2. **Use the cost ratio and the correlation.** High correlation and a large cost ratio favour many low-fidelity runs plus a few high-fidelity corrections; weak correlation means the cheap runs add little.
3. **Expect a budget-dependent optimum.** Setinek, Galletti and Brandstetter (2025) studied RANS airfoil surrogates trained on mixtures of low- and high-fidelity runs. With a small data-generation budget, mixtures beat high-fidelity-only training (the broader coverage from many cheap samples outweighs the accuracy of a few expensive ones); with a large budget, accuracy became limited by data fidelity and more high-fidelity data won. For wall shear stress, low-fidelity data did not help at any budget. The 2026 follow-up across six PDE problems and four architectures found that optimal mixtures matched high-fidelity-only accuracy with about 3.1× less data-generation compute on average and up to 6.3× less.
4. **Spend high-fidelity runs where the fidelities disagree,** found from the paired set or by active learning on the discrepancy model.

## When multi-fidelity fails

- **The cheap model misses the decisive physics.** RANS on massively separated flow, linear FEA past yield, a coarse mesh that cannot resolve a shock or a boundary layer. The discrepancy is then large, non-smooth and concentrated exactly where the answer matters.
- **Correlation changes across the design space.** A constant ρ cannot capture a cheap model that is good on attached flow and wrong on separated flow; use nonlinear methods (NARGP, input augmentation) and check the discrepancy against the paired set.
- **Different discretizations.** Coarse and fine meshes need interpolation onto common points, which adds its own error; for field surrogates, use mesh-independent models or project onto shared query points. See [Interpolating fields between meshes](mesh-data-processing.md#interpolating-fields-between-meshes).
- **Some quantities do not transfer.** Integrated forces may correlate well while local quantities (wall shear, peak stresses) do not, as the 2025 scaling study found for wall shear.
- **Labels are not truth either.** The "high-fidelity" simulation has its own modelling error; if experiments exist, they are the top fidelity. See [When the training data has its own errors](uncertainty-quantification.md#when-the-training-data-has-its-own-errors).

## Digital twins: simulation plus measurements

In a digital twin of a running asset (a heat exchanger, a turbine, a production line), sensor data is the highest fidelity but sparse and noisy, while simulations are dense but biased. Combine them by:

- **Data assimilation:** a surrogate predicts the state, and a Kalman filter (or ensemble Kalman filter) corrects it with each new measurement, tracking uncertainty;
- **Bias correction:** a discrepancy model, trained on measured-minus-simulated values at sensor locations, as in co-kriging;
- **Periodic fine-tuning** of the surrogate on accumulated measurements, with monitoring for drift. See [Design: heat-exchanger digital twin](design-heat-exchanger-digital-twin.md).

## Code: co-kriging with 4 expensive and 11 cheap runs

The standard test problem from Forrester, Sóbester and Keane: the expensive function y_H(x) = (6x − 2)² sin(12x − 4), and a cheap, biased model y_L(x) = 0.5 y_H(x) + 10(x − 0.5) − 5. With only four high-fidelity runs, compare a GP on high fidelity alone, AR1 co-kriging with ρ chosen by marginal likelihood, and a GP that takes the low-fidelity prediction as an extra input.

```python
import warnings
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
warnings.filterwarnings("ignore")

high = lambda x: (6 * x - 2) ** 2 * np.sin(12 * x - 4)              # the "expensive" simulation
low = lambda x: 0.5 * high(x) + 10 * (x - 0.5) - 5                  # a biased, scaled cheap model

def gp(X, y, dims=1, normalize=True):
    k = ConstantKernel(1.0, (1e-3, 1e4)) * RBF([0.2] * dims, (1e-2, 1e1))
    return GaussianProcessRegressor(k, alpha=1e-8, normalize_y=normalize, n_restarts_optimizer=5, random_state=0).fit(X, y)

x_hi = np.array([0.0, 0.4, 0.6, 1.0])                                # 4 expensive simulations
x_lo = np.linspace(0, 1, 11)                                         # 11 cheap simulations (include x_hi)
x_test = np.linspace(0, 1, 201)
rmse = lambda pred: np.sqrt(np.mean((pred - high(x_test)) ** 2))
col = lambda x: x[:, None]

gp_hi = gp(col(x_hi), high(x_hi))                                    # 1. high fidelity only
gp_lo = gp(col(x_lo), low(x_lo))                                     # GP of the cheap model
lo_at_hi, lo_at_test = gp_lo.predict(col(x_hi)), gp_lo.predict(col(x_test))

# 2. Co-kriging, y_H(x) = rho * y_L(x) + delta(x): choose rho by the marginal likelihood of the discrepancy GP.
fits = []
for r in np.linspace(0, 3, 61):
    g = gp(col(x_hi), high(x_hi) - r * lo_at_hi, normalize=False)   # same scale for every rho
    fits.append((g.log_marginal_likelihood_value_, r, g))
_, rho, gp_delta = max(fits, key=lambda t: t[0])
pred_ar1 = rho * lo_at_test + gp_delta.predict(col(x_test))

# 3. Input augmentation (allows a nonlinear relationship): a GP on (x, y_L(x)).
gp_aug = gp(np.column_stack([x_hi, lo_at_hi]), high(x_hi), dims=2)
pred_aug = gp_aug.predict(np.column_stack([x_test, lo_at_test]))

print(f"scale of the function: std of y_H = {high(x_test).std():.2f}")
print(f"low fidelity alone:              RMSE {rmse(low(x_test)):.3f}")
print(f"GP on 4 high-fidelity runs:      RMSE {rmse(gp_hi.predict(col(x_test))):.3f}")
print(f"co-kriging (rho = {rho:.2f}):         RMSE {rmse(pred_ar1):.3f}")
print(f"GP on (x, low-fidelity output):  RMSE {rmse(pred_aug):.3f}")
```

Output:

```
scale of the function: std of y_H = 4.51
low fidelity alone:              RMSE 6.116
GP on 4 high-fidelity runs:      RMSE 5.618
co-kriging (rho = 2.00):         RMSE 0.055
GP on (x, low-fidelity output):  RMSE 2.662
```

**What it shows.**

- Neither source alone is useful: the cheap model is biased (RMSE 6.1 on a function with standard deviation 4.5), and four expensive points are too few for a GP to follow the oscillation (RMSE 5.6).
- Co-kriging recovers the true scale factor ρ = 2 from the likelihood, and because the discrepancy y_H − 2y_L = −20x + 20 is linear, four high-fidelity points pin it down: RMSE 0.055, a hundred times better.
- Input augmentation halves the error but is far worse than co-kriging here: with four points, a 2D GP cannot learn the relationship that the AR1 structure assumes. More flexible methods need more high-fidelity data; the right structural assumption is worth a lot.

## In an interview

**Thirty-second answer to "How do you combine many cheap low-fidelity simulations with a few expensive high-fidelity ones?"** "Learn the trend from the cheap data and a correction from the expensive data. Classically that is co-kriging: y_high = ρ times y_low plus a discrepancy, both Gaussian processes, fitted on nested designs. With networks, I would pretrain on the low-fidelity runs and fine-tune on the high-fidelity ones, or feed the low-fidelity prediction as an input. Before committing the budget I run a small paired set to measure the correlation, because if the cheap model misses separation or yield, the correction is as hard to learn as the original problem."

**Likely follow-ups**

- *How would you split the budget?* Use the cost ratio and the correlation from a paired pilot set; studies of CFD surrogates show mixtures win at small budgets and high fidelity wins at large ones.
- *When does it fail?* Weak or varying correlation where the answer matters, quantities like wall shear that the cheap model cannot represent, and mismatched meshes.
- *What is the top fidelity?* Often experiments; simulations have their own model error.

## References

- M. C. Kennedy and A. O'Hagan, "Predicting the output from a complex computer code when fast approximations are available", *Biometrika*, 2000.
- A. I. J. Forrester, A. Sóbester and A. J. Keane, "Multi-fidelity optimization via surrogate modelling", *Proceedings of the Royal Society A*, 2007.
- [Nonlinear information fusion algorithms for data-efficient multi-fidelity modelling (Perdikaris et al.)](https://royalsocietypublishing.org/doi/10.1098/rspa.2016.0751)
- [A composite neural network that learns from multi-fidelity data (Meng & Karniadakis)](https://arxiv.org/abs/1903.00104)
- [Towards Multi-Fidelity Scaling Laws of Neural Surrogates in CFD (Setinek, Galletti & Brandstetter)](https://arxiv.org/abs/2511.01830)
- [Compute-optimal data scaling for neural surrogates via multi-fidelity training (Setinek et al.)](https://arxiv.org/abs/2610.17221)
- [Emukit (multi-fidelity GPs and Bayesian optimization)](https://github.com/EmuKit/emukit)
