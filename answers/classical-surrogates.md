# Classical Surrogates: Response Surfaces, RBF Interpolation and Kriging

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Classical surrogates map a **small vector of design parameters** to a few **scalar outputs**. A **polynomial response surface** (usually quadratic) is fitted by least squares; it is cheap, interpretable and smooth, but it cannot follow strongly nonlinear responses. **Radial basis function (RBF) interpolation** writes the response as a weighted sum of radially symmetric functions centred on the samples; it passes exactly through the data and handles nonlinearity well in low dimensions. **Kriging** is Gaussian-process regression: the same kind of interpolant, plus hyperparameters fitted to the data and a **variance** at every prediction, which is what makes it the engine of Bayesian optimization and active learning. They are **enough**, and often better than deep learning, when the design space is **parametric** (up to roughly 10-20 inputs), the outputs are **scalars** or a few POD coefficients, data is **tens to hundreds of simulations**, and the response is smooth. They stop being enough when the input is free-form geometry, the output is a full field, the response has discontinuities (separation, buckling, contact) or the data grows into the tens of thousands.

**On this page**

- [Polynomial response surfaces](#polynomial-response-surfaces)
- [Radial basis function interpolation](#radial-basis-function-interpolation)
- [Kriging](#kriging)
- [When classical surrogates are enough](#when-classical-surrogates-are-enough)
- [Code: four surrogates on a beam with 20 to 200 samples](#code-four-surrogates-on-a-beam-with-20-to-200-samples)
- [In an interview](#in-an-interview)

## Polynomial response surfaces

**Model.** For d inputs, a quadratic response surface (RSM) is

```math
\hat y(\mathbf x) = \beta_0 + \sum_{i} \beta_i x_i + \sum_{i \le j} \beta_{ij} x_i x_j
```

with (d + 1)(d + 2)/2 coefficients (15 for d = 4, 66 for d = 10), fitted by least squares.

**Strengths.**

- Needs few samples: at least as many as coefficients, ideally 1.5-2 times more.
- Coefficients are interpretable: β_i is a main effect, β_ij an interaction, β_ii curvature.
- Smooths noise (it does not interpolate), which suits noisy experiments.
- Classical **design of experiments** pairs with it: central composite and Box-Behnken designs are built to estimate quadratic models efficiently.

**Weaknesses.** A global low-order polynomial cannot represent strong nonlinearity, multiple optima or sharp transitions; higher orders need many more samples and oscillate. Use it for local models around a design point, quick screening, and as a baseline.

**Feature engineering helps more than order.** If physics says the response is a power law (deflection ∝ PL³/(Ebh³)), fitting a linear model to the **logarithms** of inputs and output is exact. Dimensional analysis and the right input variables often beat a more complex model. See [Nondimensionalization](nondimensionalization-and-normalization.md#why-non-dimensionalize).

## Radial basis function interpolation

**Model.** With n samples x_i,

```math
\hat y(\mathbf x) = \sum_{i=1}^{n} w_i\, \phi\big(\lVert \mathbf x - \mathbf x_i \rVert\big) + p(\mathbf x)
```

where φ is a radial function and p a low-degree polynomial (needed for some φ). The weights solve an n × n linear system that makes the interpolant pass through every sample.

**Common kernels.**

| φ(r) | Name | Notes |
| --- | --- | --- |
| r² log r | Thin-plate spline | No shape parameter; minimizes bending energy in 2D |
| r³ | Cubic | No shape parameter |
| √(1 + (εr)²) | Multiquadric | Shape parameter ε to tune |
| exp(−(εr)²) | Gaussian | Very smooth; ill-conditioned for small ε |

**Strengths.** Exact interpolation of deterministic simulations; no assumption of global form; works on scattered data in moderate dimensions; simple (`scipy.interpolate.RBFInterpolator`).

**Weaknesses.** No uncertainty estimate; the shape parameter matters and is usually tuned by cross-validation; the linear system becomes ill-conditioned with many or clustered points; interpolating noisy data overfits (add smoothing). RBF interpolation is also used for **mesh morphing**: displacements at control points are interpolated to all mesh nodes, which is how many parametric CFD studies deform geometries.

## Kriging

Kriging, from geostatistics (Krige, Matheron), is Gaussian-process regression: the response is modelled as a random function with a covariance kernel, and the prediction is the conditional mean given the samples.

- **Ordinary kriging** assumes an unknown constant mean; **universal kriging** a polynomial trend; simple kriging a known mean.
- The prediction is a weighted combination of kernel functions centred on the samples, like RBF interpolation, but the **kernel hyperparameters** (length scales per input, variance, noise) are **fitted by maximum likelihood**, and every prediction comes with a **variance**.
- The variance is what enables expected-improvement Bayesian optimization (the EGO algorithm of Jones, Schonlau and Welch, 1998) and active learning.

The equations, kernels and scaling are on [Gaussian processes](gaussian-processes.md).

| | Quadratic RSM | RBF interpolation | Kriging (GP) |
| --- | --- | --- | --- |
| Passes through data | No (least squares) | Yes | Yes, or smooths with a noise term |
| Uncertainty | Only regression intervals for the model form | None | Predictive variance at every point |
| Hyperparameters | None | Shape parameter | Length scales, variance, noise (fitted) |
| Training cost | Tiny | O(n³) solve | O(n³) per likelihood evaluation, many evaluations |
| Good for | Screening, local models, noisy tests | Deterministic simulations, morphing | Small data, optimization, active learning |

## When classical surrogates are enough

**Use them when:**

1. The design is described by **parameters** (dimensions, angles, material properties, load magnitudes), roughly up to 10-20 of them.
2. The outputs are **scalars** (drag, peak stress, maximum temperature, mass) or a few reduced coefficients (POD modes of a field; see [Reduced-order models](reduced-order-models.md)).
3. You have **tens to a few hundred** simulations. Deep networks are data-hungry; on this scale a well-tuned GP is usually more accurate and gives uncertainty for free.
4. The response is **smooth** over the design space.
5. You need **uncertainty** for optimization or for deciding where to simulate next.
6. You need something **today**: these fit in seconds and are easy to explain to an engineering customer.

**Move to deep learning when:**

1. The input is **free-form geometry** (a new STL each time) rather than parameters. See [Geometry representations](geometry-representations.md).
2. The output is a **field** with local features that a few POD modes cannot capture. See [Scalar vs field surrogates](surrogate-models.md#scalar-vs-field-surrogates).
3. You have **thousands of samples** or more, where GP cost (O(n³)) and RBF conditioning bite.
4. The response has **discontinuities or regime changes** that stationary kernels smear.
5. You want to **transfer across design families** or reuse a pretrained model.

**In practice:** always fit a classical surrogate as a **baseline**. If a deep model cannot beat a GP on the same 100 simulations, it is not ready. See [Baselines to beat](evaluation-metrics.md#baselines-to-beat).

## Code: four surrogates on a beam with 20 to 200 samples

The "simulation" is the tip deflection of a steel cantilever, δ = PL³/(3EI) with I = bh³/12, over four inputs (length, width, height, load). Each surrogate is trained on a Latin hypercube of n designs and tested on 2,000 random designs; the table reports the median relative error. All models fit log(δ) on inputs scaled to [0, 1], except the last row, which uses physics-informed features: the logarithms of the inputs.

```python
import warnings
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.stats import qmc
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.neural_network import MLPRegressor
warnings.filterwarnings("ignore")

lo = np.array([0.5, 0.02, 0.05, 500.0])       # length L [m], width b [m], height h [m], tip load P [N]
hi = np.array([1.5, 0.06, 0.15, 2000.0])
E = 210e9                                       # steel, Pa

def tip_deflection_mm(X):                       # the "simulation": cantilever tip deflection P L^3 / (3 E I)
    L, b, h, P = X.T
    return 1e3 * P * L**3 / (3 * E * b * h**3 / 12)

def fit_predict(name, Xtr, ytr, Xte):
    if name == "quadratic RSM":
        poly = PolynomialFeatures(2)
        return LinearRegression().fit(poly.fit_transform(Xtr), ytr).predict(poly.transform(Xte))
    if name == "RBF (thin plate)":
        return RBFInterpolator(Xtr, ytr, kernel="thin_plate_spline")(Xte)
    if name == "Kriging (GP)":
        kernel = ConstantKernel() * Matern(length_scale=[1.0] * 4, nu=2.5) + WhiteKernel(1e-6)
        gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=3, random_state=0)
        return gp.fit(Xtr, ytr).predict(Xte)
    if name == "MLP":
        mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=5000, random_state=0)
        mlp.fit(Xtr, (ytr - ytr.mean()) / ytr.std())
        return mlp.predict(Xte) * ytr.std() + ytr.mean()

rng = np.random.default_rng(0)
X_test = lo + (hi - lo) * rng.random((2000, 4))
y_test = tip_deflection_mm(X_test)
scaled = lambda X: (X - lo) / (hi - lo)
models = ["quadratic RSM", "RBF (thin plate)", "Kriging (GP)", "MLP"]
print(f"{'n':>4}  " + "  ".join(f"{m:>16}" for m in models) + "   linear in log inputs")
for n in [20, 50, 200]:
    X_train = lo + (hi - lo) * qmc.LatinHypercube(d=4, seed=1).random(n)
    y_train = tip_deflection_mm(X_train)
    errors = []
    for name in models:
        pred = fit_predict(name, scaled(X_train), np.log(y_train), scaled(X_test))
        errors.append(np.median(np.abs(np.exp(pred) - y_test) / y_test))
    loglin = LinearRegression().fit(np.log(X_train), np.log(y_train))         # physics-informed features
    errors.append(np.median(np.abs(np.exp(loglin.predict(np.log(X_test))) - y_test) / y_test))
    print(f"{n:>4}  " + "  ".join(f"{100 * e:>15.2f}%" for e in errors[:4]) + f"   {100 * errors[4]:>18.2e}%")
```

Output:

```
   n     quadratic RSM  RBF (thin plate)      Kriging (GP)               MLP   linear in log inputs
  20             4.29%             5.51%             3.92%            15.98%             2.13e-13%
  50             3.19%             3.52%             1.04%            11.23%             2.39e-13%
 200             2.58%             1.04%             0.23%             7.77%             1.17e-13%
```

**What it shows.**

- With 20 simulations, all three classical surrogates beat the MLP by a wide margin; with 200, Kriging is about 30 times more accurate than the MLP. Small data is the classical surrogates' home ground.
- Kriging improves fastest with data because its fitted length scales adapt to how strongly each input matters.
- The quadratic response surface stalls at a few percent: a fixed global polynomial cannot represent L³/h³ exactly, however much data it gets.
- **The best "model" is the right features.** A linear fit on log inputs is exact to machine precision with 20 samples, because the physics is a power law. Before reaching for a bigger model, check what dimensional analysis already tells you.

## In an interview

**Thirty-second answer to "When are classical surrogates (polynomial response surfaces, RBFs, Kriging) enough?"** "When the design is a handful of parameters, the outputs are scalars or a few POD coefficients, you have tens to hundreds of simulations, and the response is smooth. A quadratic response surface is the quick, interpretable baseline; RBF interpolation passes through deterministic simulation results and handles nonlinearity; Kriging adds fitted length scales and a variance, which is what Bayesian optimization needs. On that kind of problem a GP usually beats a neural network. I move to deep learning when the input is free-form geometry, the output is a full field, or there are thousands of samples."

**Likely follow-ups**

- *RBF vs Kriging?* Same form of interpolant; Kriging fits its kernel by maximum likelihood and gives a variance.
- *How many samples for a quadratic RSM?* At least the number of coefficients, (d + 1)(d + 2)/2, ideally 1.5-2 times more.
- *How would you get a field from a classical surrogate?* POD of the field snapshots, then one GP or RBF per mode coefficient.

## References

- A. Forrester, A. Sóbester and A. Keane, *Engineering Design via Surrogate Modelling: A Practical Guide*, Wiley, 2008.
- [Gaussian Processes for Machine Learning (Rasmussen & Williams, free online)](https://gaussianprocess.org/gpml/)
- D. R. Jones, M. Schonlau and W. J. Welch, "Efficient Global Optimization of Expensive Black-Box Functions", *Journal of Global Optimization*, 1998.
- [scipy.interpolate.RBFInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html)
- [Surrogate Modeling Toolbox (SMT)](https://github.com/SMTorg/smt)
- D. C. Montgomery, *Design and Analysis of Experiments*, Wiley (response surface methodology).
