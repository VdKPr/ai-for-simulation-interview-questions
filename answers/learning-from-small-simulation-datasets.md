# Learning from Small Simulation Datasets

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Industrial simulation datasets are small: tens to a few hundred high-fidelity runs, because each costs hours. With so few samples, the main risks are overfitting and fooling yourself during model selection. Control overfitting by choosing models with strong inductive biases (Gaussian processes and kernel methods on design parameters, equivariant or physics-informed architectures for fields), by reducing the input dimension (design parameters, non-dimensional groups, compressed geometry), by regularizing (weight decay, early stopping on held-out *designs*, ensembles) and by adding information from elsewhere (pretrained models, cheaper low-fidelity simulations, physics constraints). For model selection, use cross-validation that holds out whole designs, keep the search space small, and remember that with a few dozen samples the error estimate itself is noisy: in the example below, the cross-validated error of the chosen model ranged from 0.64 to 1.6 times its true error across datasets of 40 simulations. A customer with only 40 simulations can still get something useful, but it should be scoped to what 40 samples can support: a parametric surrogate with honest error bars, sensitivity insights, and a plan for which simulations to run next.

**On this page**

- [Why small data is the normal case](#why-small-data-is-the-normal-case)
- [Controlling overfitting with 50 to 500 simulations](#controlling-overfitting-with-50-to-500-simulations)
- [Model selection with very few simulations](#model-selection-with-very-few-simulations)
- [Code: how noisy is a cross-validation estimate with 40 samples?](#code-how-noisy-is-a-cross-validation-estimate-with-40-samples)
- [A customer with 40 simulations](#a-customer-with-40-simulations)
- [In an interview](#in-an-interview)

## Why small data is the normal case

- A high-fidelity CFD or crash run costs hours to a day on a cluster, so a design programme produces hundreds, not millions, of results.
- Existing archives are heterogeneous: different solver versions, meshes and set-ups, so only a fraction is directly usable.
- Each run contains a field with millions of values, so there is a lot of supervision *per sample*. But the values within one simulation are strongly correlated, and what the model must generalize across is **designs**. For generalization to new designs, the effective sample size is closer to the number of simulations than to the number of points.

## Controlling overfitting with 50 to 500 simulations

**1. Choose the model to match the data.**

- **Design parameters available?** Start with a Gaussian process or another kernel method, or gradient-boosted trees for scalars. With tens to hundreds of samples in a handful of dimensions they are hard to beat, and GPs give calibrated uncertainty. See [Gaussian processes](gaussian-processes.md) and [Classical surrogates](classical-surrogates.md).
- **Fields on one fixed mesh?** Reduced-order models (POD plus regression of the coefficients) have very few effective parameters. See [Reduced-order models](reduced-order-models.md).
- **Free-form geometry?** Use architectures with strong inductive biases: graph networks with relative (translation-invariant) features, equivariant layers where the symmetry really holds, locality, and moderate size. See [Equivariance and symmetry](equivariance-and-symmetry.md).

**2. Reduce what has to be learned.**

- Non-dimensionalize inputs and outputs, and remove known scalings (for linear problems, predict the response to a unit load). See [Why non-dimensionalize](nondimensionalization-and-normalization.md#why-non-dimensionalize).
- Compress geometry to a few parameters or a learned low-dimensional code when possible.
- Learn the residual of a cheap physics model (a beam-theory estimate, a correlation, a coarse simulation) instead of the full answer.

**3. Regularize, and stop on the right data.**

- Weight decay, moderate model size, dropout in moderation.
- **Early stopping on held-out designs**, never on held-out points from training designs (those leak). See [Data splits and leakage](data-splits-and-leakage.md).
- **Ensembles** of several models: they reduce variance and give an uncertainty estimate. See [Deep ensembles](uncertainty-quantification.md#deep-ensembles).
- **Multi-task outputs:** predicting the field and the key scalars together often regularizes both.

**4. Add information from outside the dataset.**

- **Physics constraints:** hard constraints (divergence-free outputs, boundary conditions built in) or PDE-residual losses. See [Hard vs soft constraints](hard-vs-soft-constraints.md).
- **Valid data augmentation:** only transformations that map the whole problem to a valid problem (mirror a symmetric case together with its boundary conditions). See [When data augmentation is physically valid](equivariance-and-symmetry.md#when-data-augmentation-is-physically-valid).
- **Pretrained models:** fine-tune a model pretrained on large public or company-wide datasets. See [Fine-tuning on a small customer dataset](foundation-models-for-physics.md#fine-tuning-on-a-small-customer-dataset).
- **Multi-fidelity data:** many cheap runs (coarse meshes, RANS, 2D) plus a few expensive ones. See [Multi-fidelity learning](multi-fidelity-learning.md).

**5. Spend new simulations wisely.** Active learning picks the next runs where the model is most uncertain or the decision most sensitive, which is far more efficient than random sampling. See [Active learning and acquisition functions](design-of-experiments-and-active-learning.md#active-learning-and-acquisition-functions).

**6. Measure the effect of data.** A learning curve (error against number of training simulations) shows whether more data would help, or whether the model or the label noise is the limit. See [the learning-curve example](generating-simulation-training-data.md#code-a-learning-curve-and-the-label-noise-floor).

## Model selection with very few simulations

With a few dozen samples, a single train/validation/test split wastes data and gives noisy answers. Practical rules:

1. **Cross-validate at the design level.** k-fold cross-validation (5 or 10 folds) or leave-one-out over *simulations*, not points. If designs come in families, use leave-one-family-out so that the estimate reflects generalization to new families.
2. **Keep the search space small.** Every extra configuration compared on the same small data is another chance to pick one that looks good by luck (selection bias, the "winner's curse"). Prefer sensible defaults and a few meaningful choices over large random searches.
3. **Use nested cross-validation for an honest estimate.** Choose hyperparameters inside each outer training fold and evaluate on the outer fold; the outer score estimates the performance of the *whole procedure*, including selection. It is slightly pessimistic, because each outer model sees less data.
4. **Use the marginal likelihood for GP hyperparameters.** For Gaussian processes, maximizing the marginal likelihood fits kernel hyperparameters without a validation split. See [Choosing a kernel and fitting hyperparameters](gaussian-processes.md#choosing-a-kernel-and-fitting-hyperparameters).
5. **Report the spread, not just the mean.** Repeat cross-validation with different fold assignments and seeds; report mean and standard deviation, and compare models with paired differences fold by fold.
6. **Treat small differences as ties.** If two models differ by less than the fold-to-fold variation, pick the simpler or more robust one.
7. **Keep a final check if you can.** Even a handful of untouched designs, or newly run simulations, give an independent sanity check before delivery.

## Code: how noisy is a cross-validation estimate with 40 samples?

Twenty independent "customers", each with 40 simulations of the same smooth four-parameter response. For each, choose a kernel ridge regression model by 5-fold cross-validation over 20 hyperparameter settings, then compare its cross-validated error with its true error on 5,000 new designs.

```python
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, KFold

rng = np.random.default_rng(1)

def simulator(X):                                    # stand-in for an expensive solver, 4 design parameters
    return np.sin(2 * X[:, 0]) + X[:, 1] * X[:, 2] - 0.5 * X[:, 3] ** 2 + 0.05 * rng.standard_normal(len(X))

grid = {"alpha": np.logspace(-6, 0, 4), "gamma": np.logspace(-2, 2, 5)}   # 20 configurations
X_new = rng.random((5000, 4)); y_new = simulator(X_new)                    # 'future' designs, for checking only

ratios = []
for repeat in range(20):                             # 20 customers, each with their own 40 simulations
    X = rng.random((40, 4)); y = simulator(X)
    search = GridSearchCV(KernelRidge(kernel="rbf"), grid, cv=KFold(5, shuffle=True, random_state=repeat),
                          scoring="neg_root_mean_squared_error").fit(X, y)
    cv_estimate = -search.best_score_
    actual = np.sqrt(np.mean((search.best_estimator_.predict(X_new) - y_new) ** 2))
    ratios.append(cv_estimate / actual)
ratios = np.array(ratios)
print(f"CV estimate / actual error over 20 datasets: mean {ratios.mean():.2f}, "
      f"min {ratios.min():.2f}, max {ratios.max():.2f}")
```

Output:

```
CV estimate / actual error over 20 datasets: mean 1.06, min 0.64, max 1.60
```

On average, cross-validation is about right. But for any *one* dataset of 40 simulations, which is what a real project has, the estimate was anywhere from 36% too optimistic to 60% too pessimistic. Differences between candidate models smaller than that are not evidence. This is why the rules above emphasize small search spaces, repeated folds, reporting spread, and an independent final check.

## A customer with 40 simulations

**Can you still deliver a useful model?** Usually yes, if the deliverable fits the data. A good answer in an interview:

1. **Clarify the decision.** What will the model be used for: ranking a few candidate designs, finding which parameters matter, or giving accurate absolute values for sign-off? Forty samples can support the first two much better than the third.
2. **Look at the data.** How many design parameters? Are the 40 runs spread over the space (a design of experiments) or clustered around one baseline? Are they consistent (same solver, mesh and set-up)?
3. **Start with a parametric surrogate with uncertainty.** If the designs are described by a few parameters, a Gaussian process with leave-one-out validation is the natural first model. It gives predictions with error bars and shows where the data is thin.
4. **Use physics to reduce what must be learned.** Non-dimensional inputs, known scalings, and a simple analytical or coarse-mesh model whose error the surrogate corrects.
5. **Borrow information.** Fine-tune a model pretrained on related public or company data, or add many cheap low-fidelity runs the customer can afford (coarser mesh, simpler physics) and combine them with the 40 high-fidelity ones. See [Multi-fidelity learning](multi-fidelity-learning.md) and [Foundation models for physics](foundation-models-for-physics.md).
6. **Deliver insight, not just predictions.** Sensitivity analysis (which parameters drive the output) is often the most valuable result of a small dataset. See [Global sensitivity analysis](design-of-experiments-and-active-learning.md#global-sensitivity-analysis-and-sobol-indices).
7. **Recommend the next simulations.** Use the model's uncertainty to choose the 10 to 20 runs that would improve it most, so the customer's next investment in compute is efficient.
8. **Be explicit about limits.** State the validated range, the expected error with its uncertainty, and that designs outside the range go to the solver.

What not to do: train a large deep network on 40 samples, report a random-split error, and promise that it generalizes to new geometries.

## In an interview

**Thirty-second answer to "How do you control overfitting with 50 to 500 simulations?"** "Match the model to the data: a GP or kernel method on design parameters, POD for a fixed mesh, or a GNN or transformer with strong inductive biases for free-form geometry. Reduce what must be learned with non-dimensionalization and known scalings, regularize with weight decay, ensembles and early stopping on held-out designs, and add outside information: physics constraints, pretrained models, cheap low-fidelity data. Then measure with design-level cross-validation and a learning curve, and use active learning for the next simulations."

**Likely follow-ups**

- *Why is the effective sample size the number of simulations?* Points within one simulation are correlated; generalization is across designs.
- *How do you select hyperparameters with 30 samples?* Small search space, k-fold or leave-one-out at the design level, nested CV for the final estimate, marginal likelihood for GPs, and treat small differences as ties.
- *What would you deliver to a customer with 40 simulations?* A GP-type surrogate with error bars and a validity range, a sensitivity analysis, and a recommended set of next simulations.

## References

- C. E. Rasmussen and C. K. I. Williams, [*Gaussian Processes for Machine Learning*, MIT Press, 2006 (free online)](https://gaussianprocess.org/gpml/)
- [Towards Foundation Models for Scientific Machine Learning: Characterizing Scaling and Transfer Behavior (Subramanian et al.)](https://arxiv.org/abs/2306.00258)
- [A composite neural network that learns from multi-fidelity data (Meng & Karniadakis)](https://arxiv.org/abs/1903.00104)
- [scikit-learn: nested versus non-nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
