# A PINN Training Recipe

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A good PINN recipe follows the pipeline that Wang, Sankaran, Wang and Perdikaris laid out in *An Expert's Guide to Training Physics-informed Neural Networks* (2023), plus a few standard practices. **Formulate:** non-dimensionalize the PDE so inputs, outputs and residual terms are of order one, and enforce boundary conditions exactly where a simple construction exists. **Architecture:** an MLP with smooth activations (tanh), Glorot initialization, Fourier feature embeddings of the inputs to counter spectral bias, and random weight factorization. **Training:** randomly resampled collocation points, a weighted loss whose global weights are balanced by gradient norms, causal weighting in time for time-dependent problems, curriculum or time-marching for hard cases, Adam with a decaying learning rate, and optionally L-BFGS on a fixed point set at the end. **Validation:** always against a reference solution, with every loss term logged separately. On the test problem below, this recipe reduces the error from 0.35 (a naive PINN) to 1.5 × 10⁻⁴.

**On this page**

- [Step 1: formulate the problem well](#step-1-formulate-the-problem-well)
- [Step 2: choose the architecture](#step-2-choose-the-architecture)
- [Step 3: train](#step-3-train)
- [Step 4: validate](#step-4-validate)
- [Code: the recipe on a problem where a naive PINN fails](#code-the-recipe-on-a-problem-where-a-naive-pinn-fails)
- [A checklist to copy](#a-checklist-to-copy)
- [In an interview](#in-an-interview)

## Step 1: formulate the problem well

1. **Non-dimensionalize** the PDE, the domain and the outputs. The Expert's Guide puts this first: standard initializations assume inputs and outputs of order one, and residual terms of wildly different sizes cause loss imbalance. See [Why non-dimensionalize](nondimensionalization-and-normalization.md#why-non-dimensionalize).
2. **Scale the residual** so its typical size is of order one (divide by the dominant coefficient), which helps the loss weights start sensibly.
3. **Enforce boundary conditions exactly** where you can: Dirichlet conditions with an output ansatz u = g + d·N, periodic conditions with periodic input features (sin and cos of the coordinate over the period). Each exactly satisfied condition removes a loss term and its weight. See [Soft vs hard boundary conditions in PINNs](hard-vs-soft-constraints.md#soft-vs-hard-boundary-conditions-in-pinns).
4. **Decide what is unknown** (for inverse problems) and parametrize it safely (positivity via exp or softplus).
5. **Check that a PINN is the right tool.** For a routine forward problem, a classical solver may be faster and more accurate. See [When a PINN beats a classical solver](pinns.md#when-a-pinn-beats-a-classical-solver-and-when-it-loses).

## Step 2: choose the architecture

1. **MLP with tanh activations and Glorot initialization** as the default (smooth, so second derivatives exist). The Expert's Guide recommends these defaults.
2. **Fourier feature embeddings** of the input coordinates, with a frequency scale matched to the solution (several scales for multi-scale problems). See [Fourier features](spectral-bias-and-fourier-features.md#fourier-features-and-positional-encodings).
3. **Random weight factorization:** write each weight matrix as a per-neuron scale times a direction, which the Expert's Guide found accelerates convergence and improves accuracy.
4. **Moderate size:** a few hidden layers of 64 to 256 units is typical; much larger networks are harder to train with physics losses.
5. **Alternatives for hard cases:** domain decomposition (FBPINNs) for large or multi-scale domains; modified MLP architectures with gating.

## Step 3: train

1. **Collocation points:** sample randomly (or with low-discrepancy sequences) and **resample every iteration** with Adam; use residual-based adaptive sampling for sharp features. See [Collocation points](pinns.md#collocation-points).
2. **Loss weights:** start at 1 and update the global weights every f steps (the guide suggests on the order of 1,000) by **gradient-norm balancing**, λ̂_i = (Σ_j ‖∇L_j‖) / ‖∇L_i‖, smoothed with a moving average. See [Loss imbalance and adaptive weights](pinn-training-failures.md#loss-imbalance-and-adaptive-weights).
3. **Causal weighting** for time-dependent problems: split time into segments and weight later segments by exp(−ε Σ earlier residuals). See [Causality in time-dependent PINNs](pinn-training-failures.md#causality-in-time-dependent-pinns).
4. **Curriculum or time-marching** for hard regimes: solve an easier version (smaller coefficient, shorter time window) first, then continue from it.
5. **Optimizer:** Adam with an exponentially decaying learning rate (and a short warm-up); then optionally **L-BFGS** on a fixed set of points, in double precision, to polish. See [Adam then L-BFGS](optimizers-for-physics-models.md#adam-then-l-bfgs).
6. **Log every loss term and the weights** throughout training.

## Step 4: validate

1. **Compare with a reference** (an FEM or FVM solution, or an analytical solution) on the same problem; report relative L2 and maximum errors.
2. **Check boundary and initial conditions directly**, and plot the residual over the domain.
3. **Test sensitivity** to random seeds and to the number of collocation points.
4. **For inverse problems,** test identifiability on synthetic data with known parameters and noise before using real measurements.
5. **Compare cost honestly** against a classical solver for the same accuracy. See [Reporting a speed-up fairly](evaluation-metrics.md#reporting-a-speed-up-fairly).

## Code: the recipe on a problem where a naive PINN fails

The problem −u″ = (kπ)² sin(kπx) + 1, u(0) = u(1) = 0, k = 4. A naive PINN (soft boundary conditions with equal weights, plain MLP, Adam) reaches a maximum error of about 0.35 (see [the loss-balancing example](pinn-training-failures.md#code-balancing-the-loss-terms-by-their-gradient-norms)). The recipe below uses Fourier features, exact boundary conditions, a residual scaled to order one, and Adam followed by L-BFGS.

```python
import math
import torch, torch.nn as nn

k = 4.0                                                        # -u'' = (k pi)^2 sin(k pi x) + 1, u(0) = u(1) = 0
exact = lambda x: torch.sin(k * math.pi * x) + 0.5 * x * (1 - x)
forcing = lambda x: (k * math.pi) ** 2 * torch.sin(k * math.pi * x) + 1.0

class FourierMLP(nn.Module):
    """Fourier feature embedding followed by a tanh MLP."""
    def __init__(self, n_features=32, sigma=2.0, width=64):
        super().__init__()
        self.register_buffer("B", torch.randn(1, n_features, dtype=torch.float64) * sigma)
        self.mlp = nn.Sequential(nn.Linear(2 * n_features, width), nn.Tanh(), nn.Linear(width, width),
                                 nn.Tanh(), nn.Linear(width, 1)).double()
    def forward(self, x):
        z = 2 * math.pi * x @ self.B
        return self.mlp(torch.cat([torch.sin(z), torch.cos(z)], -1))

torch.manual_seed(0)
N = FourierMLP()
u = lambda x: x * (1 - x) * N(x)                               # hard Dirichlet conditions: no boundary loss
x_fixed = torch.rand(1024, 1, dtype=torch.float64)            # fixed collocation set (needed for L-BFGS)

def residual_loss(x):
    x = x.clone().requires_grad_(True)
    ux = u(x)
    du = torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True)[0]
    d2u = torch.autograd.grad(du, x, torch.ones_like(du), create_graph=True)[0]
    return ((-d2u - forcing(x)) / (k * math.pi) ** 2).pow(2).mean()   # residual scaled to order one

opt = torch.optim.Adam(N.parameters(), lr=1e-3)
for step in range(2000):                                       # stage 1: Adam on resampled points
    loss = residual_loss(torch.rand(1024, 1, dtype=torch.float64))
    opt.zero_grad(); loss.backward(); opt.step()

lbfgs = torch.optim.LBFGS(N.parameters(), lr=1.0, max_iter=500, history_size=50,
                          tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn="strong_wolfe")
def closure():
    lbfgs.zero_grad()
    loss = residual_loss(x_fixed)
    loss.backward()
    return loss
lbfgs.step(closure)                                            # stage 2: L-BFGS on the fixed set

xt = torch.linspace(0, 1, 1001, dtype=torch.float64).unsqueeze(1)
with torch.no_grad():
    recipe_error = (u(xt) - exact(xt)).abs().max().item()
print(f"recipe (Fourier features + hard BCs + scaled residual + Adam -> L-BFGS): max error {recipe_error:.1e}")
```

Output:

```
recipe (Fourier features + hard BCs + scaled residual + Adam -> L-BFGS): max error 1.5e-04
```

Roughly three orders of magnitude better than the naive PINN on the same problem, with no loss weights to tune at all (the hard constraint removed the boundary term, and scaling the residual made the remaining term well-sized).

## A checklist to copy

```
FORMULATE   [ ] non-dimensionalized PDE, domain and outputs
            [ ] residual scaled to order one
            [ ] Dirichlet / periodic conditions enforced exactly where simple
ARCHITECT   [ ] tanh MLP, Glorot init, modest size
            [ ] Fourier features (scale matched to the solution; multi-scale if needed)
            [ ] random weight factorization (optional, recommended by the Expert's Guide)
TRAIN       [ ] collocation points resampled each step; adaptive sampling for sharp features
            [ ] global weights balanced by gradient norms every ~1,000 steps (moving average)
            [ ] causal weights for time-dependent problems
            [ ] curriculum / time-marching for hard regimes
            [ ] Adam with decaying learning rate, then L-BFGS on fixed points (double precision)
VALIDATE    [ ] compare with a reference solution; log each loss term
            [ ] check boundary/initial conditions and residual maps
            [ ] seeds and point-count sensitivity; honest cost comparison
```

## In an interview

**Thirty-second answer to "What does a good PINN training recipe look like?"** "Non-dimensionalize and scale the residual first; enforce boundary conditions exactly where simple. Use a tanh MLP with Fourier feature embeddings, and random weight factorization as the Expert's Guide recommends. Train on resampled collocation points with global weights balanced by gradient norms, causal weighting for time-dependent problems, curriculum or time-marching for hard regimes, Adam with a decaying learning rate and then L-BFGS to polish. Validate against a reference solution with every loss term logged. That combination is what turns most failing PINNs into working ones."

**Likely follow-ups**

- *Which single change helps most?* Usually non-dimensionalization and fixing loss imbalance (hard constraints or gradient-norm weights); then Fourier features for oscillatory solutions.
- *Why resample collocation points?* It prevents the network from satisfying the residual only at fixed points and acts like data augmentation; switch to a fixed set only for L-BFGS.
- *How do you know it worked?* Comparison with a reference solution, not the training loss.

## References

- [An Expert's Guide to Training Physics-informed Neural Networks (Wang, Sankaran, Wang & Perdikaris, 2023)](https://arxiv.org/abs/2308.08468) and its JAX library, [jaxpi](https://github.com/PredictiveIntelligenceLab/jaxpi)
- [Respecting causality is all you need for training PINNs (Wang, Sankaran & Perdikaris)](https://arxiv.org/abs/2203.07404)
- [On the eigenvector bias of Fourier feature networks (Wang, Wang & Perdikaris)](https://arxiv.org/abs/2012.10047)
- [Challenges in Training PINNs: A Loss Landscape Perspective (Rathore et al.)](https://arxiv.org/abs/2402.01868)
- [A comprehensive study of non-adaptive and residual-based adaptive sampling for PINNs (Wu et al.)](https://arxiv.org/abs/2207.10289)
