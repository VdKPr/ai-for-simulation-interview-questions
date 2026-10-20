# Non-dimensionalization and Normalization

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Physical variables come in wildly different units and scales: pressures around 10⁵ Pa, velocities around 10 m/s, turbulent viscosities around 10⁻⁴ m²/s, lengths in millimetres or metres. Neural networks train best when inputs and outputs are of order one. **Non-dimensionalization** rescales variables by characteristic physical scales so that problems with the same dimensionless groups (Reynolds number, Mach number, geometry ratios) become identical; it reduces the number of inputs, lets a model transfer across sizes and speeds, and makes extrapolation visible. **Normalization** then standardizes whatever scale differences remain, **per channel**, because otherwise the loss is dominated by whichever channel has the largest numbers, and **with training statistics only**, because anything else leaks test information and is impossible at deployment. When targets span orders of magnitude across samples, a **relative L2 loss** treats every sample equally where plain MSE would fit only the large ones; **heavy-tailed** positive targets are better modelled in log space or relative to a reference value, but the extreme values often matter most, so evaluate them separately.

**On this page**

- [Why non-dimensionalize](#why-non-dimensionalize)
- [Normalizing fields and coordinates](#normalizing-fields-and-coordinates)
- [MSE or relative L2 loss](#mse-or-relative-l2-loss)
- [Heavy-tailed targets](#heavy-tailed-targets)
- [Pitfalls](#pitfalls)
- [Code: per-channel normalizer](#code-per-channel-normalizer)
- [In an interview](#in-an-interview)

## Why non-dimensionalize

**The Buckingham π theorem.** If a physical problem involves n variables built from k independent physical dimensions (mass, length, time, temperature), it can be described by n − k independent dimensionless groups.

**Example: drag on a smooth sphere** in incompressible flow. The drag force F depends on the fluid density ρ, the speed U, the diameter D and the viscosity μ: five variables, three dimensions (mass, length, time), so two groups:

```math
F = f(\rho, U, D, \mu) \quad\Longrightarrow\quad C_D = \frac{F}{\tfrac12 \rho U^2 \tfrac{\pi}{4}D^2} = g(\text{Re}), \qquad \text{Re} = \frac{\rho U D}{\mu}
```

A dataset that varies four inputs collapses onto a single curve with one input. A model trained in the dimensionless form needs far less data and automatically covers any combination of sizes, speeds and fluids with the same Reynolds number.

**Why it helps a surrogate:**

1. **Fewer inputs.** Each dimensionless group replaces several raw inputs; data needs grow quickly with input dimension.
2. **Order-one numbers.** C_p and C_f are of order one; pressures in pascals are not.
3. **Transfer across scales.** A model trained on wind-tunnel-scale cases applies to full-scale cases at the same Reynolds number (subject to the usual caveats).
4. **Honest extrapolation checks.** A query might have speed and length each within the training ranges, but a Reynolds number outside the trained range. In dimensionless inputs that is obvious.
5. **Physical consistency.** Outputs scale correctly with inputs by construction. For linear elastic structures, for example, displacement scales with load / (E × length), so predicting the scaled displacement removes two inputs entirely. See [Stress, strain and linear elasticity](stress-strain-and-linear-elasticity.md#why-this-matters-for-machine-learning).
6. **Better conditioned physics losses.** For PINNs, the first recommendation of the Expert's Guide (Wang et al., 2023) is to non-dimensionalize so the PDE terms are of comparable size.

**Groups worth knowing:** Reynolds (inertia vs viscosity), Mach (compressibility), Prandtl, Nusselt and Biot (heat transfer; see [Dimensionless numbers](heat-transfer-basics.md#dimensionless-numbers-prandtl-nusselt-and-biot)), Strouhal (shedding frequency × length / speed), Grashof and Rayleigh (natural convection), Froude (gravity waves), and geometric ratios (thickness / length, fillet radius / diameter).

**Typical non-dimensional outputs:** pressure coefficient C_p = (p − p∞)/(½ρU²) instead of p, skin-friction coefficient instead of wall shear, Nusselt number instead of the heat-transfer coefficient, stress divided by a nominal stress (force / area) instead of stress, natural frequency times L/√(E/ρ) instead of frequency.

**Caveats.** Non-dimensionalization only collapses data if the problem really is similar: surface roughness, compressibility, gravity, or a second length scale add more groups. And it cannot create diversity that the data does not contain: if every training case has the same Reynolds number, the model knows nothing about Reynolds-number effects.

## Normalizing fields and coordinates

After non-dimensionalization, remaining scale differences are removed by standardization: subtract a mean and divide by a standard deviation.

**Why per channel.** Suppose a model predicts pressure (spread of thousands of pascals), velocity (spread of a few m/s) and turbulent viscosity (spread of 10⁻⁴ m²/s) in raw units. A mean-squared error summed over channels is then almost entirely pressure error: in the code below, a model that is equally good on all three channels has 99.999% of its raw-unit loss coming from pressure. The optimizer would ignore the other two. Standardizing each channel separately gives each channel a comparable share of the loss.

**Why training statistics only.** The normalization constants are part of the model. Computing them on the whole dataset, including validation and test samples, leaks information about the test distribution into training and gives optimistic results, especially when the test set is a held-out family or an extrapolation range. At deployment, there is no test set to compute statistics on. Fit the normalizer on the training split, save it with the model weights, and reuse it unchanged.

**Recipes:**

| Quantity | Recipe |
| --- | --- |
| Coordinates | Subtract a reference point (the domain centre, or a fixed point on the vehicle); divide **all axes by the same** characteristic length, so shapes keep their proportions and rotations stay rotations |
| Scalar fields | Per-channel z-score: mean and standard deviation over all training samples and points (area- or volume-weighted on non-uniform meshes) |
| Vector fields (velocity, displacement) | Scale all components by the **same** factor (for example the free-stream speed); per-component standardization breaks rotational structure |
| Positive, heavy-tailed fields (turbulent viscosity, peak stress) | Log transform first, or predict relative to a reference; see [Heavy-tailed targets](#heavy-tailed-targets) |
| Global conditions (inlet speed, load, temperature) | Non-dimensionalize first, then standardize |
| Positional encodings | Scale coordinates to a fixed range such as [−1, 1] before Fourier features ([Fourier features](spectral-bias-and-fourier-features.md#fourier-features-and-positional-encodings)) |

**Robust variants:** with outliers, use the median and interquartile range instead of mean and standard deviation.

## MSE or relative L2 loss

Suppose the dataset contains load cases whose stresses range from 10 MPa to 1,000 MPa, or flows whose pressures differ by orders of magnitude between operating points. With **plain MSE**, the squared errors of the large-magnitude samples are millions of times bigger than those of the small ones, so training effectively fits only the large cases. The small cases can have 50% relative error and barely affect the loss.

The **per-sample relative L2 loss** normalizes each sample by its own magnitude:

```math
\mathcal{L}_{\text{rel}} = \frac{1}{N}\sum_{i=1}^{N} \frac{\lVert \hat{y}_i - y_i \rVert_2}{\lVert y_i \rVert_2}
```

Every sample then counts equally, regardless of its scale. It is the standard loss and metric in the neural-operator literature for exactly this reason. See [Field metrics](evaluation-metrics.md#field-metrics).

**When to choose which:**

- **Relative L2** when sample magnitudes vary a lot and each sample matters equally (many operating points, many load magnitudes).
- **MSE on normalized data** when magnitudes are similar across samples, or when absolute errors are what the engineer cares about (a temperature limit in °C).
- **Better still, remove the magnitude physically:** divide stresses by the nominal stress of each load case, pressures by the dynamic pressure. Then plain MSE works and the model does not have to learn the scaling.

**Caveats of relative L2:**

- It is unstable when ‖y‖ is near zero (a nearly unloaded case, a field that is mostly zero): add a small floor to the denominator.
- For fields that cross zero (pressure relative to the free stream), relative errors at individual points are meaningless; the norm over the whole field is fine.
- Match the training loss to the metric and the decision. If the decision depends on a peak value, add a term for it.

## Heavy-tailed targets

Some targets have a long tail: most values are small and a few are very large. Examples: peak von Mises stress across designs, stagnation-region pressures, turbulent viscosity, heat flux near hot spots, fatigue lives.

**What goes wrong:** the rare large values dominate MSE when they are present and are badly fitted because there are few of them, while the network spends its capacity on the bulk.

**Options:**

- **Log transform** positive quantities: model log(y). Errors in log space are relative errors, the distribution becomes closer to normal (in the code below the skewness of a log-normal field falls from 21 to 0), and predictions can never be negative. When converting back with uncertainty, remember that exp(mean of log y) is the *median*, not the mean.
- **Power transforms** (Box-Cox, Yeo-Johnson) or quantile-based scaling for other shapes.
- **Predict relative to a reference:** a stress concentration factor K_t = σ_max/σ_nominal instead of σ_max, C_p instead of p. The reference carries the big variation, the model the remaining one.
- **Robust losses** (Huber) reduce the influence of outliers, but be careful: in engineering the extreme values are often exactly the ones that decide the design (the peak stress, the hottest spot). Down-weighting them can be wrong.
- **Explicit treatment of extremes:** a separate output head for the peak, loss weighting towards high values, or quantile losses.
- **Evaluate the tail separately:** report errors on the top 1% of values or on the per-design maximum, not only averages. See [Low average error but failures where it matters](evaluation-metrics.md#low-average-error-but-failures-where-it-matters).

Clip extreme values only if they are known to be artefacts (for example singular peaks at point constraints); otherwise clipping hides the very cases the model must get right. See [Stress concentrations and singularities](stress-concentrations-and-singularities.md#what-this-means-for-machine-learning).

## Pitfalls

- **Leakage:** fitting normalization statistics on all data, including the test set. See [Data splits and leakage](data-splits-and-leakage.md).
- **Per-sample normalization** (dividing each sample by its own maximum) throws away the amplitude; the model must then predict the scale some other way.
- **Forgetting the inverse transform** before computing physical metrics such as drag or a safety factor.
- **Near-constant channels:** dividing by a tiny standard deviation amplifies noise; add an epsilon or drop the channel.
- **Mixed unit systems** between data sources (mm and m, MPa and Pa), or pressure stored as kinematic pressure p/ρ in one solver and as pascals in another.
- **Losing the normalizer:** a model checkpoint without its normalization constants cannot be used. Version them together.

For a step-by-step procedure when training fails, see [Debugging training on physical data](debugging-training-on-physical-data.md).

## Code: per-channel normalizer

```python
import numpy as np
from scipy.stats import skew

class ChannelNormalizer:
    """Per-channel z-score. Fit on training data only; reuse for val/test/inference."""
    def fit(self, train):                       # train: (n_samples, n_points, n_channels)
        self.mean = train.mean(axis=(0, 1))
        self.std = train.std(axis=(0, 1)) + 1e-8
        return self

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse(self, z):
        return z * self.std + self.mean

rng = np.random.default_rng(0)
pressure = 1e5 + 2e3 * rng.standard_normal((100, 500))       # Pa
velocity = 30 + 5 * rng.standard_normal((100, 500))          # m/s
nu_t = np.exp(rng.normal(-8, 1.5, (100, 500)))               # turbulent viscosity: positive, heavy-tailed
data = np.stack([pressure, velocity, nu_t], -1)
train, test = data[:80], data[80:]

# A model that is equally good on every channel: error = 10% of that channel's spread.
err = 0.1 * data.std(axis=(0, 1)) * rng.standard_normal(test.shape)
share_raw = (err**2).mean(axis=(0, 1)) / (err**2).mean(axis=(0, 1)).sum()
print("share of a raw-unit MSE (p, u, nu_t):", [f"{100 * s:.4f}%" for s in share_raw])

norm = ChannelNormalizer().fit(train)                          # statistics from training data only
err_n = norm.transform(test + err) - norm.transform(test)
share_norm = (err_n**2).mean(axis=(0, 1)) / (err_n**2).mean(axis=(0, 1)).sum()
print("share after per-channel normalization:", [f"{100 * s:.1f}%" for s in share_norm])

print(f"skewness of nu_t: raw {skew(nu_t.ravel()):.1f}, after log {abs(skew(np.log(nu_t).ravel())):.2f}")
```

Output:

```
share of a raw-unit MSE (p, u, nu_t): ['99.9994%', '0.0006%', '0.0000%']
share after per-channel normalization: ['33.3%', '32.8%', '33.9%']
skewness of nu_t: raw 21.0, after log 0.00
```

## In an interview

**Thirty-second answer to "How do you normalize physical fields, and why per channel and only with training data?"** "First non-dimensionalize with physical scales, for example pressure coefficient and coordinates over a reference length, so cases with the same Reynolds number look the same. Then standardize each channel with its own mean and standard deviation, because channels differ by orders of magnitude and a raw loss would be almost entirely pressure. Coordinates get one common scale so shapes keep their proportions, and vectors one scale for all components. The statistics come from the training set only and are saved with the model, otherwise test information leaks in and the deployed model would have nothing to compute them from."

**Likely follow-ups**

- *MSE or relative L2 when magnitudes span orders of magnitude?* Relative L2 per sample, or better, remove the magnitude physically by dividing by a known scale.
- *How would you handle peak stress as a target?* Log transform or predict a concentration factor, give the peak its own head, and evaluate the tail separately rather than down-weighting it.
- *Why non-dimensionalize at all if you normalize anyway?* Normalization fixes numerical scale; non-dimensionalization removes whole input dimensions and makes the model transfer across sizes and speeds.

## References

- [An Expert's Guide to Training Physics-informed Neural Networks (Wang et al., 2023): non-dimensionalization is its first recommendation](https://arxiv.org/abs/2308.08468)
- [AI/ML+Physics, Part 2: Curating Training Data (Steve Brunton, video playlist)](https://www.youtube.com/playlist?list=PLMrJAkhIeNNQ0BaKuBKY43k4xMo6NSbBa)
- E. Buckingham, "On physically similar systems; illustrations of the use of dimensional equations", *Physical Review*, 1914.
- [Fourier Neural Operator (Li et al.): relative L2 as the standard loss and metric](https://arxiv.org/abs/2010.08895)
