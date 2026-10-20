# Debugging Training on Physical Data

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** When a model will not converge on a physical dataset, debug from the bottom up instead of tuning hyperparameters at random. First make sure the **code** can learn at all: overfit a single sample, then a single small batch; if the loss will not go to nearly zero, there is a bug (shapes broadcasting silently, gradients detached, a wrong mask) or the targets are misaligned with the inputs. Then check the **data**: units, normalization with training statistics, NaN and infinite values, near-constant channels, outliers, point versus cell data, and whether each field is paired with the right geometry. Then check that the **target is predictable** from the inputs (is a boundary condition or operating point missing?) and compare with **trivial baselines** (mean field, nearest training design). Only then tune the **optimization** (learning-rate sweep, gradient norms, schedule) and the **model** (capacity, receptive field, inductive biases). Physics-specific bugs, such as sign conventions, reference pressures, coordinate frames and mesh-to-graph errors, are common and worth checking explicitly.

**On this page**

- [Step 1: can the model learn anything at all?](#step-1-can-the-model-learn-anything-at-all)
- [Step 2: is the data what you think it is?](#step-2-is-the-data-what-you-think-it-is)
- [Step 3: is the target predictable from the inputs?](#step-3-is-the-target-predictable-from-the-inputs)
- [Step 4: does it beat trivial baselines?](#step-4-does-it-beat-trivial-baselines)
- [Step 5: optimization](#step-5-optimization)
- [Step 6: model and data capacity](#step-6-model-and-data-capacity)
- [Physics-specific bugs](#physics-specific-bugs)
- [Code: a data sanity report and an overfit-one-batch test](#code-a-data-sanity-report-and-an-overfit-one-batch-test)
- [In an interview](#in-an-interview)

## Step 1: can the model learn anything at all?

**Overfit one sample, then one small batch.** A reasonably sized network should drive the training loss on a single example (or a batch of 8 to 32) to nearly zero within a few hundred to a few thousand steps. If it cannot:

- **Shape bugs:** a prediction of shape (N, 1) compared with a target of shape (N,) broadcasts to an N × N matrix, and the loss can never reach zero. The code below shows exactly this: the loss gets stuck at the variance of the target.
- **Detached gradients:** a `.detach()`, `.numpy()` or `torch.no_grad()` in the wrong place; parameters not passed to the optimizer; a frozen layer.
- **Masks and reductions:** a boundary mask that excludes everything; summing where you meant to average, so the effective learning rate is wildly off.
- **The loss is not what you think:** wrong sign, wrong term, a regularizer that dominates.
- **Misaligned inputs and targets:** a shuffling or indexing bug pairs fields with the wrong geometry.

This one test separates "my pipeline is broken" from "my model is not good enough", which saves days.

## Step 2: is the data what you think it is?

- **Statistics per channel:** min, max, mean, standard deviation, number of NaN and infinite values, for every input and target channel. Plot histograms.
- **Units and scales:** mm or m, Pa or MPa or kinematic pressure (p/ρ); degrees or radians; all samples in the same system.
- **Normalization:** statistics from the training set only, applied consistently, inverted before computing physical metrics. See [Normalizing fields and coordinates](nondimensionalization-and-normalization.md#normalizing-fields-and-coordinates).
- **Near-constant channels:** dividing by a tiny standard deviation amplifies noise into huge values.
- **Point vs cell data:** values stored at cell centres attached to node coordinates are shifted by half a cell. See [Simulation file formats](simulation-file-formats.md).
- **Mesh problems:** duplicated nodes, degenerate elements, inverted normals, disconnected pieces. See [Mesh data processing](mesh-data-processing.md).
- **Failed simulations:** unconverged runs or crashed runs with partial results in the training set. Check the solver metadata. See [What makes training data physically meaningful](generating-simulation-training-data.md#what-makes-training-data-physically-meaningful-and-sufficient).
- **Look at samples.** Plot a few input geometries next to their target fields. Many bugs are obvious to the eye and invisible in a loss curve.

## Step 3: is the target predictable from the inputs?

A model cannot learn what the inputs do not determine:

- **Missing inputs:** the operating point (inlet speed, load magnitude), boundary-condition types, material properties, or yaw angle varies across samples but is not given to the model.
- **Inconsistent labels:** samples from different solvers, turbulence models or mesh strategies; the "same" input with different outputs.
- **Path dependence:** for plasticity or hysteresis, the final state depends on the load history, which must be an input. See [Plasticity and load-history dependence](nonlinear-structural-analysis.md#plasticity-and-load-history-dependence).
- **Chaos and noise:** time-averaged fields with short averaging windows, or chaotic dynamics predicted too far ahead, have an irreducible noise floor.

A quick test: find pairs of training samples with nearly identical inputs and compare their targets. If they differ a lot, something is missing from the inputs or wrong in the labels.

## Step 4: does it beat trivial baselines?

- **The mean field** (or mean scalar) of the training set.
- **Nearest neighbour:** the field of the most similar training design.
- **A linear model or a GP** on design parameters, if they exist.

If the deep model does not beat these clearly, look for a bug or leakage before adding capacity. If it beats them by an implausible margin, look for leakage. See [Data splits and leakage](data-splits-and-leakage.md) and [Baselines to beat](evaluation-metrics.md#baselines-to-beat).

## Step 5: optimization

- **Learning-rate sweep:** a few values over orders of magnitude; or a range test that increases the rate until the loss diverges.
- **Gradient norms per layer:** zero (dead layers, saturated activations, detached graph) or exploding (bad initialization, too-high learning rate, outliers).
- **Schedule:** warm-up for transformers, decay to settle; gradient clipping for rare large gradients.
- **Loss balance:** in multi-term losses (data, physics, boundary), one term can dominate and freeze the others. See [Loss imbalance and adaptive weights](pinn-training-failures.md#loss-imbalance-and-adaptive-weights).
- **Optimizer choice:** Adam for robustness; L-BFGS to polish deterministic physics losses. See [Adam then L-BFGS](optimizers-for-physics-models.md#adam-then-l-bfgs).
- **Precision:** float16 underflow in small residuals or derivatives. See [When mixed precision hurts](scaling-to-large-meshes.md#when-mixed-precision-hurts).

## Step 6: model and data capacity

- **Training error high:** underfitting. More capacity, better features, a larger receptive field, fewer constraints.
- **Training error low, validation error high:** overfitting or distribution shift between splits. Regularize, add data, check the split. See [Controlling overfitting](learning-from-small-simulation-datasets.md#controlling-overfitting-with-50-to-500-simulations).
- **Learning curve:** if validation error is still falling steeply with more data, data is the limit; if flat, the model, the inputs or the label noise are.
- **Receptive field:** a GNN whose message-passing depth is smaller than the distance information must travel cannot predict the effect of a far-away boundary condition. See [How many message-passing steps](gnns-and-meshgraphnets.md#how-many-message-passing-steps).
- **Spectral bias:** sharp features missing while smooth ones are fine. See [Spectral bias](spectral-bias-and-fourier-features.md#spectral-bias).

## Physics-specific bugs

These are worth a dedicated check, because they produce models that train fine and are wrong:

- **Sign conventions:** wall shear stress stored as the force on the fluid rather than on the wall; normals pointing inwards; tension negative in one tool and positive in another.
- **Reference values:** pressure relative to different references in different runs; gauge versus absolute pressure; temperature in K versus °C.
- **Coordinate frames:** a few CAD exports in a different orientation or with a different origin; inflow direction not consistent.
- **Mesh-to-graph conversion:** edges only in one direction when the model expects both; missing self-loops if required; duplicate edges; node features not aligned with node ordering after reindexing.
- **Boundary information missing:** node-type features (wall, inlet, fixed, loaded) not provided, so the model cannot know where constraints are.
- **Time-step mismatches:** autoregressive training data with irregular output intervals treated as uniform.

## Code: a data sanity report and an overfit-one-batch test

```python
import numpy as np
import torch

def sanity_report(name, a):
    """Print the basic statistics of every channel; flag NaN/Inf and near-constant channels."""
    a = np.asarray(a, dtype=float).reshape(-1, a.shape[-1])
    for c in range(a.shape[1]):
        col = a[:, c]
        bad = int((~np.isfinite(col)).sum())
        finite = col[np.isfinite(col)]
        flag = " <- NaN/Inf!" if bad else (" <- nearly constant" if finite.std() < 1e-8 * max(1.0, abs(finite.mean())) else "")
        print(f"{name}[{c}]: min {finite.min():10.3g} max {finite.max():10.3g} "
              f"mean {finite.mean():10.3g} std {finite.std():10.3g} bad {bad}{flag}")

def overfit_one_batch(x, y, steps=2000, lr=1e-2, seed=0):
    """A model that cannot drive the loss on ONE small batch close to zero has a bug."""
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 64), torch.nn.Tanh(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        loss = ((model(x) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

rng = np.random.default_rng(0)
fields = np.stack([1e5 + 2e3 * rng.standard_normal(1000), 30 + rng.standard_normal(1000), np.full(1000, 1.2)], -1)
fields[17, 0] = np.nan                                     # a corrupted sample
sanity_report("targets", fields)

torch.manual_seed(0)
x = torch.randn(32, 4)
y = x[:, 0] ** 2 + x[:, 1]                                 # shape (32,): a 1-D target
loss_ok = overfit_one_batch(x, y.unsqueeze(1))             # (32, 1) vs (32, 1): correct
loss_bug = overfit_one_batch(x, y)                         # (32, 1) - (32,) broadcasts to (32, 32): silent bug
print(f"overfit one batch, shapes (32,1) vs (32,1): final loss {loss_ok:.1e}")
print(f"overfit one batch, shapes (32,1) vs (32,):  final loss {loss_bug:.1e}  (variance of y = {y.var():.2f})")
```

Output:

```
targets[0]: min   9.22e+04 max   1.06e+05 mean   9.99e+04 std   1.95e+03 bad 1 <- NaN/Inf!
targets[1]: min       26.8 max       32.9 mean         30 std       1.02 bad 0
targets[2]: min        1.2 max        1.2 mean        1.2 std   4.44e-16 bad 0 <- nearly constant
overfit one batch, shapes (32,1) vs (32,1): final loss 1.3e-05
overfit one batch, shapes (32,1) vs (32,):  final loss 2.4e+00  (variance of y = 2.50)
```

The report catches the corrupted sample and a constant channel that would blow up under standardization. The overfit test catches a broadcasting bug that raises no error: the "model" can never do better than predicting the mean, so the loss sticks at the variance of the target.

## In an interview

**Thirty-second answer to "Your model will not converge on a physical dataset. How do you debug it?"** "Bottom up. First overfit one sample and one small batch: if the loss will not go to zero, it is a bug, such as broadcasting, detached gradients or misaligned targets. Then check the data per channel: units, normalization from training statistics, NaNs, constant channels, point versus cell data, failed runs, and plot samples. Then check the target is predictable: are operating conditions or boundary conditions missing, are labels consistent? Compare with the mean and nearest-neighbour baselines. Only then sweep the learning rate, look at gradient norms and loss balance, and finally capacity and receptive field."

**Likely follow-ups**

- *The loss plateaus at exactly the variance of the target. What does that suggest?* The model is predicting the mean: a broadcasting or alignment bug, missing inputs, or dead gradients.
- *Training loss is fine, validation terrible.* Overfitting, a bad split, or distribution shift between splits; check for leakage and the split design.
- *The model gets smooth regions right but misses peaks.* Spectral bias, loss averaging, or label problems near singularities.

## References

- [A Recipe for Training Neural Networks (Andrej Karpathy, blog post)](https://karpathy.github.io/2019/04/25/recipe/)
- [An Expert's Guide to Training Physics-informed Neural Networks (Wang et al., 2023)](https://arxiv.org/abs/2308.08468)
- [Deep Learning Tuning Playbook (Google Research)](https://github.com/google-research/tuning_playbook)
