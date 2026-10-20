# Evaluation Metrics

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Evaluate a surrogate at three levels. **Fields:** per-sample relative L2 error, weighted by cell area or volume on unstructured meshes, reported as a distribution (mean, median, 90th percentile), plus errors in the regions that matter (walls, hot spots, wakes). **Integrated quantities:** the numbers engineers decide on (drag and lift coefficients, peak stress, pressure drop, peak temperature), because small field errors can integrate into large force errors. **Physics consistency:** conservation, divergence, boundary conditions, equilibrium. Compare against **strong baselines** (nearest neighbour, a GP on design parameters, POD + regression, a coarse solver run) and report **speed-ups fairly** (end to end, at matched accuracy, including data generation when queries are few). Use enough test cases and **several seeds**, with confidence intervals. When the surrogate ranks designs, measure **ranking** (Spearman, top-k recall), not only error. Test **generalization to unseen geometries** with group splits and out-of-distribution sets, and **slice** results by family and regime, because a low average error often hides failures exactly where decisions are made.

**On this page**

- [Field metrics](#field-metrics)
- [Integrated quantities](#integrated-quantities)
- [Reporting a speed-up fairly](#reporting-a-speed-up-fairly)
- [Baselines to beat](#baselines-to-beat)
- [Physics-consistency checks](#physics-consistency-checks)
- [How many test cases, and variability across seeds](#how-many-test-cases-and-variability-across-seeds)
- [When the surrogate is used for ranking](#when-the-surrogate-is-used-for-ranking)
- [Evaluating generalization to unseen geometries](#evaluating-generalization-to-unseen-geometries)
- [Low average error but failures where it matters](#low-average-error-but-failures-where-it-matters)
- [Engineering metrics in recent car-aerodynamics papers](#engineering-metrics-in-recent-car-aerodynamics-papers)
- [Code: per-sample relative L2 error](#code-per-sample-relative-l2-error)
- [In an interview](#in-an-interview)

## Field metrics

| Metric | Definition | Watch out for |
| --- | --- | --- |
| **Relative L2 per sample** | ‖ŷ − y‖₂ / ‖y‖₂ for each sample, then mean, median, 90th percentile over samples | The standard in neural-operator papers. On unstructured meshes, weight by cell area or volume |
| MSE / RMSE | Mean squared error, root of it | Dominated by high-magnitude samples and fields; depends on units |
| MAE, relative L1 | Mean absolute error, normalized | More robust to outliers; GeoTransolver reports relative L1 |
| Maximum error | Worst point | Sensitive to singularities and outliers; useful as a check |
| R² per field | 1 − SS_res/SS_tot | Can look excellent while the important regions are poor |
| Regional or masked errors | Errors near walls, in the wake, at hot spots, in high-stress zones | Usually where the decision is made |
| Spectral errors | Energy spectra, error per wavenumber | Reveals lost small scales (over-smoothing) |
| Gradient errors | Errors of derived quantities (wall shear, stress from displacement, vorticity) | Derivatives amplify noise in predicted fields |

**Normalize per field and per sample.** Pressure, velocity and turbulent viscosity have very different magnitudes; report each separately. For fields passing through zero (C_p, signed stresses), the relative error's denominator can be small; report absolute errors next to relative ones.

**Weight by area or volume.** On a graded mesh, an unweighted metric is dominated by finely meshed regions; the code below shows an unweighted error of 8.6% against an area-weighted 18.2% for the same prediction.

## Integrated quantities

**What to compute.** Forces and coefficients (C_D, C_L, moments), pressure drop, mass flow, heat flux, maximum temperature, peak von Mises stress (a high percentile away from singular points; see [Stress concentrations and singularities](stress-concentrations-and-singularities.md)), first natural frequencies, displacement at key points.

**Why they need separate evaluation.** Drag is often a small difference between large forces: the front of a car is pushed back, the rear pulled back, and the drag is what remains. A field error of 2% concentrated on the rear can produce a 20% drag error, as the code below shows. Conversely, random field errors can cancel in the integral. Report both.

**How to report.**

- Absolute and relative errors, and errors in the customer's units: drag "counts" are 0.0001 in C_D in aerospace and often quoted as 0.001 in automotive.
- Correlation between predicted and true values across test designs (R², parity plot).
- Compute forces from predicted fields with the **exact** areas and normals of the surface, F = ∮ (−(p − p∞) n + τ_w) dA. See [Drag force from pressure and wall shear](mesh-data-processing.md#drag-force-from-pressure-and-wall-shear).
- If a decision depends on a quantity, consider training a dedicated head or adding an integral loss.

## Reporting a speed-up fairly

1. **End to end.** Surrogate: geometry preprocessing (cleaning, sampling, SDF) + inference + post-processing. Solver: meshing + solving + post-processing. Excluding the surrogate's preprocessing or the solver's meshing distorts the ratio.
2. **Hardware.** State it, and compare comparable budgets (a GPU against how many CPU cores).
3. **Matched accuracy.** Compare with a solver configuration that reaches the **surrogate's** error level: a coarser mesh or looser tolerance may get there much faster than the high-fidelity reference. This is the most commonly omitted comparison.
4. **Amortization.** Include data generation and training when the number of queries is small: a surrogate that needs 500 simulations to train only pays off after more than 500 queries' worth of solver time.
5. **Warm starts and batching.** State whether the solver could have been warm-started and whether the surrogate's throughput assumes batching.
6. **Accuracy at that speed.** A speed-up without an error budget is meaningless.

McGreivy and Hakim (2024) reviewed articles claiming that ML beats standard numerical methods for fluid-related PDEs and found that 79% (60 of 76) compared against weak baselines. See [Questions to ask about a 1000x speed-up claim](ml-vs-numerical-solvers.md#questions-to-ask-about-a-1000x-speed-up-claim).

## Baselines to beat

Run all of these on the same split, with the same tuning effort:

1. **Mean prediction:** the training-set mean field or quantity. Any model must beat it clearly.
2. **Nearest neighbour in design space:** return the closest training design's result. Surprisingly strong for dense parametric datasets.
3. **Regression on design parameters:** linear, GP or random forest; for fields, POD + GP. See [Classical surrogates](classical-surrogates.md).
4. **A coarse solver run** at matched cost: the honest numerical baseline.
5. **Established architectures** with public code: MeshGraphNets, a U-Net on grids, FNO, PointNet++, Transolver, or the benchmark's published models.
6. **The previous production model**, for model updates.

## Physics-consistency checks

A model can have a low MSE and still violate physics; these checks find silent failures.

| Check | How |
| --- | --- |
| Mass conservation (incompressible flow) | Divergence of predicted velocity in the volume; mass flow in = mass flow out |
| Momentum, PDE residuals | Plug the prediction into the discretized equations and measure the residual |
| Boundary conditions | No-slip at walls, prescribed inlet values, zero normal flux at symmetry planes |
| Equilibrium (structures) | Reaction forces balance applied loads; stresses satisfy traction-free surfaces |
| Energy balance (heat transfer) | Heat in = heat out + stored |
| Symmetry | Symmetric inputs give symmetric outputs |
| Positivity and bounds | Turbulent kinetic energy, density, temperature in kelvin, concentrations |
| Trends | Monotonic responses engineers know (more load gives more stress) |
| Spectra (unsteady) | Energy spectra and dominant frequencies (Strouhal number) |

NVIDIA's PhysicsNeMo-CFD benchmarking workflow, for example, evaluates models with pointwise errors, integrated quantities, spectral metrics and PDE residuals on DrivAerML, and its developers argue that metrics "should capture physical consistency and engineering relevance, not just pointwise L2 error".

## How many test cases, and variability across seeds

- **Uncertainty of the test metric.** The mean error over n test designs has a standard error of about σ/√n. With 20 test cases, the 95% interval on a 4% mean error is roughly ±0.8 percentage points (the code below gives [3.3%, 4.9%]); differences between models smaller than that are not evidence. Use bootstrap intervals, and paired comparisons (the same test designs for both models).
- **Rough sizing:** to estimate a mean error to within ±10% of its value, with errors whose standard deviation is half their mean, you need about (2 × 0.5 / 0.1)² = 100 test designs.
- **Seeds.** Train with at least 3-5 random seeds (and data shuffles) and report mean ± standard deviation; the code below shows a 13% relative spread across five seeds of the same configuration. A single-seed improvement smaller than the seed spread is noise.
- **Splits.** For small datasets, repeated group k-fold cross-validation gives more stable estimates than one split. See [Small data: cross-validation done right](data-splits-and-leakage.md#small-data-cross-validation-done-right).
- **Report distributions**, not only means: box plots over test cases and the worst cases.

## When the surrogate is used for ranking

In design exploration, the surrogate often only needs to put designs **in the right order**, not predict each value exactly.

- **Rank correlation:** Spearman's ρ or Kendall's τ between predicted and true values over the candidates.
- **Top-k recall:** the fraction of the truly best k designs that appear in the predicted top k; what matters when only the top few go on to full simulation.
- **Pairwise accuracy:** the fraction of design pairs whose order is predicted correctly, restricted to pairs whose true difference exceeds the model's error (differences smaller than the error cannot be resolved; say so).
- **Regret:** how much worse the design chosen by the surrogate is than the true best.
- **Decision-level evaluation:** simulate the surrogate's top candidates with the solver and compare with what a solver-only search would have found at the same cost.

The code below shows R² = 0.90 and Spearman 0.945, yet only 8 of the true top 10 designs in the predicted top 10, because the best designs are separated by less than the model's error. Pairwise accuracy on resolvable differences is 98.5%.

## Evaluating generalization to unseen geometries

1. **Group splits by design and by family** (leave-one-family-out), so test geometries are genuinely new. See [Splitting simulation data](data-splits-and-leakage.md#splitting-simulation-data).
2. **A deliberate out-of-distribution test set:** new families, larger or smaller parts, features absent from training (a spoiler, a new rib pattern), different operating ranges. Report in-distribution and out-of-distribution results separately. DoMINO, for example, was evaluated on both in-distribution and out-of-distribution DrivAerML samples.
3. **Error against distance from the training set:** plot test error against a geometric or latent distance to the nearest training design; error should grow smoothly, and the out-of-distribution detector should flag the far cases. See [Detecting an out-of-distribution geometry](out-of-distribution-and-generalization.md#detecting-an-out-of-distribution-geometry).
4. **Remeshing and resampling robustness:** the same geometry meshed differently should give the same answer. See [Why remeshing can change a prediction](geometry-representations.md#why-remeshing-can-change-a-prediction).
5. **Calibration out of distribution:** uncertainty intervals must widen for unseen geometries. See [Checking calibration](uncertainty-quantification.md#checking-calibration).

## Low average error but failures where it matters

1. **Slice the results** by design family, size, operating point, regime (attached vs separated flow, elastic vs yielded) and out-of-distribution score; find the clusters with high error. The code below shows a 4.2% overall error hiding 14% on vans, which are 10% of the data.
2. **Check data coverage:** are there few training samples in that slice? **Check label quality:** unconverged runs, different solver settings?
3. **Check the metric:** is the average dominated by easy regions (far field) while the hard region (the wake, the hot spot) is a small fraction of the points?
4. **Fix:** targeted simulations in the failing slice (active learning), loss weighting, a specialized model or more capacity, better input features.
5. **Agree per-slice acceptance criteria** with the customer before deployment, not only an average.
6. **Route the failing slice to the solver** until it passes.

## Engineering metrics in recent car-aerodynamics papers

| Work | Dataset(s) | Metrics reported |
| --- | --- | --- |
| DrivAerML (Ashton et al., 2024) | 500 DrivAer variants, hybrid RANS-LES | Dataset paper: time-averaged surface and volume fields and forces as targets |
| DrivAerNet / DrivAerNet++ (Elrefaie et al., 2024) | 4,000 / 8,000 car designs, RANS | Drag-coefficient regression from geometry (RegDGCNN and other models) with errors and R²; field predictions |
| DoMINO (NVIDIA, 2025) | DrivAerML | Surface and volume field errors, drag forces and design trends, in- and out-of-distribution, "a range of engineering specific metrics" |
| AB-UPT (Emmi AI, 2025) | DrivAerML and others, 33K-150M cells | Surface and volume field errors; drag and lift coefficients computed from predicted surface fields; divergence of vorticity |
| GeoTransolver (NVIDIA, 2025) | DrivAerML, SHIFT-SUV, SHIFT-Wing | Drag and lift R² and relative L1 field errors |
| PhysicsNeMo-CFD benchmark (NVIDIA, 2026) | DrivAerML | L2 errors of pressure, velocity and turbulent viscosity, drag and lift coefficient errors, spectral metrics, PDE residuals |
| AirfRANS (Bonnet et al., 2022) | 1,000 2D airfoil RANS cases | Field errors plus metrics on surface forces and boundary layers; extrapolation tasks |

The trend is clear: field L2 alone is no longer accepted; papers report the integrated forces engineers use, design trends across variants, and physics-consistency diagnostics. See [Car aerodynamics datasets](public-simulation-datasets.md#car-aerodynamics-datasets).

## Code: per-sample relative L2 error

The metric function, and five small experiments with synthetic predictions that show why each section above matters: weighting, integrated quantities, ranking, small test sets and seeds, and slices.

```python
import numpy as np
from scipy.stats import spearmanr, kendalltau

def relative_l2(pred, true, weights=None):
    """Per-sample relative L2 error. pred, true: (n_samples, n_points); weights: cell areas or volumes."""
    w = np.ones_like(true) if weights is None else weights
    num = np.sqrt((w * (pred - true) ** 2).sum(axis=1))
    den = np.sqrt((w * true ** 2).sum(axis=1))
    return num / den                                   # report the mean, the median and a high percentile

rng = np.random.default_rng(0)
# 1. Weighting: a graded surface mesh where 80% of the nodes cover 10% of the area (a refined region).
n = 5000
area = np.where(np.arange(n) < 4000, 0.1 / 4000, 0.9 / 1000)           # area represented by each node
true = rng.normal(1.0, 0.3, (50, n))
pred = true.copy()
pred[:, 4000:] += rng.normal(0, 0.2, (50, 1000))                       # errors only in the coarse, large-area region
print(f"relative L2, node-weighted {relative_l2(pred, true).mean():.3f}; area-weighted {relative_l2(pred, true, area).mean():.3f}")

# 2. Drag is a small difference of large forces: front faces push back, rear faces pull back less.
m = 2000
face_area = np.full(m, 1.0 / m)
nx = np.where(np.arange(m) < m // 2, -1.0, 1.0)                        # x-normals: front faces -1, rear faces +1
cp_true = np.where(nx < 0, 0.40, 0.35) + 0.05 * np.sin(np.arange(m))   # front and rear pressure levels
cp_pred = cp_true + np.where(nx > 0, 0.01, 0.0)                        # a 0.01 error on the rear only
drag = lambda cp: -(cp * nx * face_area).sum()                         # pressure drag (x-force)
print(f"Cp field: relative L2 {relative_l2(cp_pred[None], cp_true[None], face_area[None])[0]:.3f}; "
      f"drag: true {drag(cp_true):.4f}, predicted {drag(cp_pred):.4f}, relative error {abs(drag(cp_pred) - drag(cp_true)) / drag(cp_true):.2f}")

# 3. Ranking: 200 candidate designs; the surrogate's scatter (0.003) is a third of the design spread (0.009).
cd_true = 0.30 + 0.009 * rng.standard_normal(200)
cd_pred = cd_true + 0.003 * rng.standard_normal(200)
r2 = 1 - np.sum((cd_pred - cd_true) ** 2) / np.sum((cd_true - cd_true.mean()) ** 2)
top_true, top_pred = set(np.argsort(cd_true)[:10]), set(np.argsort(cd_pred)[:10])
diff_true, diff_pred = cd_true[:, None] - cd_true[None], cd_pred[:, None] - cd_pred[None]
resolvable = np.abs(diff_true) > 2 * 0.003
pair_acc = np.mean(np.sign(diff_true[resolvable]) == np.sign(diff_pred[resolvable]))
print(f"R2 {r2:.3f}, Spearman {spearmanr(cd_true, cd_pred)[0]:.3f}, Kendall {kendalltau(cd_true, cd_pred)[0]:.3f}, "
      f"top-10 recall {len(top_true & top_pred) / 10:.1f}, pairwise accuracy when |dCd| > 2 sigma {pair_acc:.3f}")

# 4. Few test cases: a bootstrap confidence interval of the mean error, and the spread across training seeds.
errors_20 = np.abs(rng.normal(0.04, 0.02, 20))                         # errors on 20 test designs
boot = [rng.choice(errors_20, 20).mean() for _ in range(5000)]
print(f"20 test cases: mean error {errors_20.mean():.3f}, 95% bootstrap interval "
      f"[{np.percentile(boot, 2.5):.3f}, {np.percentile(boot, 97.5):.3f}]")
seed_means = 0.04 + rng.normal(0, 0.004, 5)                            # the same model trained with 5 seeds
print(f"5 training seeds: mean error {seed_means.mean():.4f} +/- {seed_means.std(ddof=1):.4f}")

# 5. Slices: a low average error hides a failing design family.
family = rng.choice(["sedan", "SUV", "van"], 300, p=[0.6, 0.3, 0.1])
err = np.where(family == "van", np.abs(rng.normal(0.15, 0.05, 300)), np.abs(rng.normal(0.03, 0.01, 300)))
print(f"overall mean error {err.mean():.3f}; " + ", ".join(f"{f} {err[family == f].mean():.3f}" for f in ["sedan", "SUV", "van"]))
```

Output:

```
relative L2, node-weighted 0.086; area-weighted 0.182
Cp field: relative L2 0.019; drag: true 0.0250, predicted 0.0200, relative error 0.20
R2 0.901, Spearman 0.945, Kendall 0.795, top-10 recall 0.8, pairwise accuracy when |dCd| > 2 sigma 0.985
20 test cases: mean error 0.041, 95% bootstrap interval [0.033, 0.049]
5 training seeds: mean error 0.0378 +/- 0.0050
overall mean error 0.042; sedan 0.030, SUV 0.030, van 0.142
```

## In an interview

**Thirty-second answer to "Which metrics would you use to evaluate a field surrogate?"** "Three levels. Field accuracy: per-sample relative L2, area- or volume-weighted, reported as a distribution and per field, plus errors in the regions that matter like the wake or hot spots. Integrated quantities: drag, lift, peak stress, computed from the predicted fields with exact areas and normals, because a 2% field error can be a 20% drag error. Physics: divergence, conservation, boundary conditions. Then baselines like nearest neighbour and a GP, several seeds with confidence intervals, group splits for unseen geometries, slices by design family, and ranking metrics if it is used to screen designs."

**Likely follow-ups**

- *How do you report a speed-up honestly?* End to end, same hardware budget, against a solver setting at the surrogate's accuracy, including data generation when queries are few.
- *Why can R² be misleading?* It is dominated by the spread between designs; a model can have R² = 0.9 and still get the top designs wrong when they are separated by less than its error.
- *How many test cases?* Enough that the confidence interval on the metric is smaller than the differences you want to claim; with 20 cases it is wide, so say so.

## References

- [Fourier Neural Operator (relative L2 as the standard metric)](https://arxiv.org/abs/2010.08895)
- [Weak baselines and reporting biases lead to overoptimism in ML for fluid-related PDEs (McGreivy & Hakim)](https://arxiv.org/abs/2407.07218)
- [DoMINO](https://arxiv.org/abs/2501.13350), [AB-UPT](https://arxiv.org/abs/2502.09692) and [GeoTransolver](https://arxiv.org/abs/2512.20399)
- [DrivAerML](https://arxiv.org/abs/2408.11969) and [DrivAerNet++](https://arxiv.org/abs/2406.09624)
- [AirfRANS](https://arxiv.org/abs/2212.07564)
- [PhysicsNeMo-CFD](https://github.com/NVIDIA/physicsnemo-cfd) and the blog post [Don't Yet Trust the Model, Test the Physics](https://nvidia.github.io/physicsnemo/blog/2026/05/29/physicsnemo-cfd/)
