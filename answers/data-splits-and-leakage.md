# Data Splits and Leakage

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Simulation datasets are full of near-copies: time steps of one transient run, parametric variants that differ by a millimetre, the same geometry at ten operating points, a mirrored car. A **random split over samples** puts near-copies on both sides, so the test set measures memorization and the error looks far better than it will be on a genuinely new design. **Split by group** (design, trajectory, base geometry, design family, time period) so that the test set matches the claim you want to make: interpolation within a family, generalization to a new family, extrapolation to higher loads, or performance on future programmes. Three common leaks: **correlated samples across the split**, **preprocessing fitted on all the data** (normalization statistics, POD bases, feature selection, augmentation before splitting), and **choosing models, epochs or seeds by test error**. The fixes are a group split, a pipeline fitted on training data only, a separate validation set (or nested cross-validation for small data), and a test set touched once.

**On this page**

- [Splitting simulation data](#splitting-simulation-data)
- [Three ways test information leaks into training](#three-ways-test-information-leaks-into-training)
- [Small data: cross-validation done right](#small-data-cross-validation-done-right)
- [Checklist before reporting a number](#checklist-before-reporting-a-number)
- [Code: a random split vs a split by design](#code-a-random-split-vs-a-split-by-design)
- [In an interview](#in-an-interview)

## Splitting simulation data

**Step 1: decide what the test should prove.**

| Claim | Split | Example |
| --- | --- | --- |
| "Interpolates within this design family" | Random split **by design**: all data of a design on one side | 500 morphed DrivAer variants, 80/10/10 by variant |
| "Generalizes to new design families" | Group k-fold or leave-one-family-out | Train on fastbacks and notchbacks, test on estate backs |
| "Extrapolates to higher loads, speeds or Reynolds numbers" | Range split: train below a threshold, test above | AirfRANS includes Reynolds-number and angle-of-attack extrapolation tasks |
| "Works on future designs" | Temporal split: older programmes train, newer ones test | Last year's products as the test set |
| "Rolls out stably" | Split by trajectory; evaluate long rollouts from unseen initial conditions | Never split frames of one run |
| "Works for a new customer" | Split by data source (customer, solver version, mesher) | Hold out one customer's designs |

**Step 2: define the groups.** A group is everything that is not independent: all time steps of a trajectory; all operating points of one geometry; all augmented copies (mirrored, rotated) of a design; all near-duplicates (find them with a geometry distance, such as a Chamfer distance or parameter distance below a threshold, and merge them into one group).

**Step 3: three sets.**

- **Training:** fit parameters and any preprocessing.
- **Validation:** choose hyperparameters, architecture, epochs and early stopping.
- **Test:** report once at the end. For small data, replace the fixed validation set with (nested) cross-validation, see below.

**Step 4: stratify when needed.** Make sure rare but important regimes (separated flow, buckled designs, high-load cases) appear in every split, or deliberately only in the test split if that is the claim.

**Step 5: use predefined splits** when a benchmark provides them (SimJEB ships three 80/20 splits; AirfRANS defines four tasks), so results are comparable with published numbers. See [Public simulation datasets](public-simulation-datasets.md).

## Three ways test information leaks into training

**1. Correlated samples on both sides of the split.**

- Consecutive time steps of one simulation are nearly identical; a model tested on frame 51 after training on frames 50 and 52 is interpolating in time, not predicting.
- Near-duplicate geometries: parametric variants with tiny differences, the same shape meshed twice, mirror images.
- The same geometry at several operating points, when the claim is generalization to new geometry: the model memorizes geometry-specific effects it could never know for a truly new design. The [code below](#code-a-random-split-vs-a-split-by-design) shows such a split underestimating the error by 40%.

**2. Preprocessing fitted with test data.**

- Normalization statistics (means, standard deviations, min-max ranges) computed over the whole dataset.
- POD bases, PCA or autoencoders fitted on all snapshots, so the test fields lie exactly in the learned subspace; for small datasets this can make reconstruction errors look several times better.
- Feature selection or sensitivity analysis done on all data.
- Augmentation before splitting: augmented copies of test designs end up in training.
- Outlier removal or filtering based on all data.

Fix: build a pipeline whose every fitted step sees only training data, and apply it unchanged to validation and test data (scikit-learn's `Pipeline` inside cross-validation enforces this).

**3. Model selection on the test set.**

- Choosing the best epoch, seed, architecture or hyperparameters by test error.
- Re-running the test after each design change ("just checking"), which slowly turns the test set into a validation set.
- With a small test set, the "best" of many equally good models looks much better than it is: the code below shows the best of 50 models with identical true error scoring 0.46 against a true 1.0.

Fix: select on validation data or by cross-validation; touch the test set once; if you must evaluate repeatedly, keep a second, untouched holdout.

**Related reporting biases.** McGreivy and Hakim (2024) found that 79% (60 of 76) of articles claiming that ML outperforms standard numerical methods for fluid-related PDEs compared against weak baselines, and that outcome and publication biases are widespread. Leakage and weak baselines produce the same symptom: claims that do not survive deployment. See [Weak baselines and reporting biases](ml-vs-numerical-solvers.md#weak-baselines-and-reporting-biases).

## Small data: cross-validation done right

With tens of simulations, a single split is too noisy to trust.

- **Group k-fold cross-validation** (or leave-one-group-out), repeated with different shuffles; report mean ± standard deviation across folds and repeats.
- **Nested cross-validation** when hyperparameters are tuned: an inner loop selects hyperparameters, an outer loop estimates the error of the whole procedure. Otherwise the tuning leaks into the estimate.
- **Bootstrap confidence intervals** on the test metric; with 20 test cases, an error of 5% might really be anywhere from 3% to 8%. Say so.
- **Prefer simpler models with stronger priors** (GPs, ROMs, physics-based features). See [Learning from small simulation datasets](learning-from-small-simulation-datasets.md).

## Checklist before reporting a number

- [ ] The split is by group and matches the claim (design, family, range, time, source).
- [ ] Near-duplicates and augmented copies are grouped together.
- [ ] Normalization, POD, feature selection and augmentation are fitted on training data only.
- [ ] Hyperparameters, epochs and seeds were chosen on validation data, never test.
- [ ] Several seeds; the spread is reported.
- [ ] In-distribution and out-of-distribution results are reported separately.
- [ ] Baselines were tuned with the same care. See [Baselines to beat](evaluation-metrics.md#baselines-to-beat).
- [ ] The test set was used once.

## Code: a random split vs a split by design

Sixty designs, each simulated at ten speeds. The model sees three design parameters and the speed; each design also has a geometric effect that its parameters do not describe (like a detail of the underbody). A random split over rows lets the model learn each design's hidden effect from its other speeds; a split by design does not. The second part shows how selecting the best of many models on a small test set flatters the winner.

```python
import warnings
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
warnings.filterwarnings("ignore")

rng = np.random.default_rng(0)
n_designs, n_speeds = 60, 10
design_params = rng.random((n_designs, 3))                         # what the model sees about each design
hidden = rng.normal(0, 0.3, n_designs)                             # geometric detail the parameters do not capture
rows = []
for d in range(n_designs):
    for v in np.linspace(20, 40, n_speeds):                        # each design simulated at 10 speeds
        drag = (1 + design_params[d] @ [0.5, -0.3, 0.2] + hidden[d]) * (v / 30) ** 2
        rows.append([*design_params[d], v, drag, d])
data = np.array(rows)
X, y, groups = data[:, :4], data[:, 4], data[:, 5].astype(int)

model = RandomForestRegressor(n_estimators=200, random_state=0)
for name, cv in [("random split over rows", KFold(5, shuffle=True, random_state=0)), ("split by design", GroupKFold(5))]:
    pred = cross_val_predict(model, X, y, cv=cv, groups=groups)
    print(f"{name:<24}: mean relative error {np.mean(np.abs(pred - y) / np.abs(y)):.3f}")

# Leak 3: choosing among many models by their test error. 50 models, all with true mean squared error 1.0.
residuals = rng.normal(0, 1, (50, 30))                             # each model's errors on the same 30 test designs
test_mse = (residuals ** 2).mean(1)
print(f"50 models with true MSE 1.00: best test MSE {test_mse.min():.2f}, median {np.median(test_mse):.2f}")
```

Output:

```
random split over rows  : mean relative error 0.165
split by design         : mean relative error 0.270
50 models with true MSE 1.00: best test MSE 0.46, median 0.98
```

The random split reports a 16.5% error; for a design the model has never seen, the honest error is 27%, about 1.6 times higher. Nothing about the model changed, only the question the split asks. In the second experiment, every model is equally good, yet the one picked by test error appears to be more than twice as accurate as it is.

## In an interview

**Thirty-second answer to "How should you split simulation data into train, validation and test?"** "By group, not by sample, and matching the claim. Time steps of one run, operating points of one geometry, mirrored copies and near-duplicate variants are not independent, so they go on the same side. If the claim is new designs within a family, split by design; new families, hold out whole families; higher Reynolds numbers, a range split; future programmes, a temporal split. Preprocessing like normalization or POD is fitted on training data only, model selection uses validation or nested cross-validation, and the test set is touched once."

**Likely follow-ups**

- *How much does a random split flatter results?* It depends on how correlated the samples are; with several operating points per design it can easily hide a large part of the true error, as the example shows.
- *What if you only have 30 simulations?* Group k-fold or leave-one-out cross-validation repeated over seeds, nested if tuning, with bootstrap intervals.
- *How do you find near-duplicates?* Distances between design parameters or geometries (Chamfer, point-cloud embeddings) below a threshold; merge them into one group.

## References

- [SimJEB: Simulated Jet Engine Bracket Dataset (predefined splits)](https://arxiv.org/abs/2105.03534)
- [AirfRANS (tasks with Reynolds-number and angle-of-attack extrapolation)](https://arxiv.org/abs/2212.07564)
- [Weak baselines and reporting biases lead to overoptimism in machine learning for fluid-related PDEs (McGreivy & Hakim)](https://arxiv.org/abs/2407.07218)
- [Leakage and the reproducibility crisis in machine-learning-based science (Kapoor & Narayanan)](https://arxiv.org/abs/2207.07048)
- [scikit-learn: cross-validation iterators for grouped data](https://scikit-learn.org/stable/modules/cross_validation.html#group-k-fold)
