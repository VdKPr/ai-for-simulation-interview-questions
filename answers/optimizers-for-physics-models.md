# Optimizers for Physics Models

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** **Adam** adapts a step size for every parameter from running averages of the gradient and its square. It is robust to noisy mini-batch gradients and badly scaled parameters, which makes it the default for training surrogates and the best way to *start* training physics-informed models. But it converges slowly to very low loss, and its final accuracy is limited by the noise of its own steps. **L-BFGS** is a quasi-Newton method: it builds an approximation of the curvature (the Hessian) from the last few gradient differences and uses a line search. On a smooth, deterministic, full-batch loss it can converge much faster and much further, but it is fragile far from a minimum and with noisy gradients. So for physics models with fixed collocation points, such as PINNs, a common recipe is **Adam first** to reach a good region robustly, **then L-BFGS** to polish. In the example below, that combination reaches a loss about ten times lower than Adam alone for the same number of training steps.

**On this page**

- [Adam in one paragraph](#adam-in-one-paragraph)
- [L-BFGS in one paragraph](#l-bfgs-in-one-paragraph)
- [Adam then L-BFGS](#adam-then-l-bfgs)
- [Code: Adam, L-BFGS and the combination](#code-adam-l-bfgs-and-the-combination)
- [Learning-rate schedules and other practical choices](#learning-rate-schedules-and-other-practical-choices)
- [Beyond Adam and L-BFGS](#beyond-adam-and-l-bfgs)
- [In an interview](#in-an-interview)

## Adam in one paragraph

Adam keeps an exponential moving average of the gradient (the first moment, like momentum) and of the squared gradient (the second moment), corrects both for their initialization bias, and updates each parameter by the first moment divided by the square root of the second. Each parameter effectively gets its own step size, inversely proportional to the typical size of its gradient. That makes Adam forgiving: it copes with mini-batch noise, with parameters whose gradients differ by orders of magnitude, and with poorly tuned learning rates. The cost is precision: with a constant learning rate, Adam keeps bouncing around the minimum at a scale set by the learning rate, and on ill-conditioned problems it makes slow progress along flat directions. **AdamW** (Adam with decoupled weight decay) is the standard choice for training data-driven surrogates and transformers.

## L-BFGS in one paragraph

Newton's method uses the Hessian H to take steps −H⁻¹∇L, which converge very fast near a minimum but are unaffordable for millions of parameters. **BFGS** builds an approximation of H⁻¹ from the changes in parameters and gradients between iterations; **limited-memory BFGS** stores only the last m pairs (typically 10 to 100) and applies the approximation implicitly, so memory is linear in the number of parameters. Each iteration uses a **line search** (for example satisfying the strong Wolfe conditions) to choose a step length that actually decreases the loss. L-BFGS shines on smooth, deterministic objectives evaluated on the **full batch**: it exploits curvature that first-order methods ignore. It struggles when gradients are noisy (mini-batches, randomly resampled collocation points), because the curvature pairs become inconsistent, and when started far from a minimum in a non-convex landscape.

## Adam then L-BFGS

**Why switch late in training:**

1. **Early on, robustness matters.** The loss landscape of a PINN or a surrogate at initialization is rough and badly scaled; Adam moves steadily without being thrown off by bad curvature estimates.
2. **Late in training, precision matters.** Once near a minimum, the loss is locally smooth and close to quadratic; curvature information lets L-BFGS take much better-scaled steps along the flat, ill-conditioned directions that Adam crawls along.
3. **Physics losses are ill-conditioned.** PDE residual losses involve derivatives of the network, which makes the landscape much worse conditioned than ordinary regression. Rathore et al. (2024) analysed PINN loss landscapes and found that combining Adam with L-BFGS outperformed either alone, and proposed a further second-order refinement stage.
4. **The data is fixed.** PINNs often use a fixed set of collocation points, giving a deterministic full-batch loss: exactly what L-BFGS needs.

**Caveats:**

- L-BFGS needs **full-batch, deterministic** losses; it does not mix well with mini-batching or with resampling collocation points every step.
- It can **stop early** (line-search failures, tolerances met) or occasionally diverge; monitor it and keep the Adam checkpoint.
- Each iteration may evaluate the loss and gradient several times during the line search, so compare by wall time or function evaluations, not only by iteration count.
- Use **double precision** for the polishing stage if very low losses matter; single precision can stall the line search.
- For large data-driven surrogates trained on mini-batches, L-BFGS is rarely used; AdamW with a good schedule is the norm.

## Code: Adam, L-BFGS and the combination

A small MLP fitting a smooth 1D target on a fixed, full batch of points (the same situation as a PINN with fixed collocation points), from the same initialization.

```python
import math
import torch

x = torch.linspace(-1, 1, 200, dtype=torch.float64).unsqueeze(1)   # double precision helps L-BFGS polish
y = torch.sin(3 * math.pi * x) * torch.exp(-x**2)      # a smooth target, full batch (like fixed collocation points)

def make_model(seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Tanh(), torch.nn.Linear(32, 32),
                               torch.nn.Tanh(), torch.nn.Linear(32, 1)).double()

def mse(model):
    return ((model(x) - y) ** 2).mean()

def train_adam(model, steps, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        loss = mse(model)
        opt.zero_grad(); loss.backward(); opt.step()
    return mse(model).item()

def train_lbfgs(model, iters):
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=iters, history_size=50,
                            tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss = mse(model)
        loss.backward()
        return loss
    opt.step(closure)
    return mse(model).item()

m1 = make_model(0)
adam_only = train_adam(m1, 6000)
print(f"Adam, 6000 steps:                     loss = {adam_only:.2e}")
m2 = make_model(0)
print(f"Adam 1000 steps:                      loss = {train_adam(m2, 1000):.2e}")
adam_lbfgs = train_lbfgs(m2, 1000)
print(f"  then L-BFGS, 1000 iterations:       loss = {adam_lbfgs:.2e}")
m3 = make_model(0)
print(f"L-BFGS alone from initialization:     loss = {train_lbfgs(m3, 1000):.2e}")
```

Output (one run on CPU):

```
Adam, 6000 steps:                     loss = 3.78e-06
Adam 1000 steps:                      loss = 2.59e-04
  then L-BFGS, 1000 iterations:       loss = 3.62e-07
L-BFGS alone from initialization:     loss = 2.26e-06
```

Adam followed by L-BFGS ends about ten times lower than Adam alone; L-BFGS from a random start does better than Adam here but worse than the combination. On harder, physics-informed losses the gaps are typically larger, and L-BFGS from initialization is more likely to stall.

## Learning-rate schedules and other practical choices

- **Warm-up, then decay.** A short linear warm-up avoids early instability (especially for transformers), and a cosine or exponential decay lets Adam settle into the minimum instead of bouncing around it. Much of "Adam cannot reach low loss" disappears with a decaying schedule.
- **Learning-rate range test.** Increase the learning rate over a few hundred steps and watch where the loss starts to diverge; pick a value somewhat below it.
- **Gradient clipping** guards against occasional huge gradients from outliers or rollout instabilities.
- **Batch size.** Larger batches give smoother gradients and allow larger learning rates; for field data, batches of a few whole simulations (or subsampled points from several simulations) are typical.
- **Loss scaling.** With several loss terms (data, PDE residual, boundary conditions), their relative weights change the effective learning rate of each; see [Loss imbalance and adaptive weights](pinn-training-failures.md#loss-imbalance-and-adaptive-weights).
- **Precision.** Mixed precision speeds up surrogate training but can hurt physics losses with small residuals and derivatives. See [When mixed precision hurts](scaling-to-large-meshes.md#when-mixed-precision-hurts).

## Beyond Adam and L-BFGS

- **Second-order and natural-gradient methods** for PINNs (Gauss-Newton, natural gradient in function space, Newton-CG variants) can reach far lower errors on small and medium problems, at higher cost per step.
- **Preconditioned first-order methods** (Shampoo, SOAP and relatives) approximate curvature at scales where L-BFGS is impractical, and are increasingly used for large models.
- **Problem reformulation** often helps more than the optimizer: non-dimensionalization, hard constraints, better architectures, curriculum or causal training. See [PINN training recipe](pinn-training-recipe.md).

## In an interview

**Thirty-second answer to "Why might you switch from Adam to L-BFGS late in training a physics model?"** "Adam is robust early on: per-parameter step sizes cope with noise and bad scaling. But near the minimum it bounces at a scale set by its learning rate and crawls along ill-conditioned directions, and physics losses are badly conditioned. L-BFGS uses curvature from recent gradients with a line search and converges much faster and further on a smooth, deterministic, full-batch loss, which is what a PINN with fixed collocation points has. So: Adam to get into a good basin, then L-BFGS to polish, in double precision, keeping the Adam checkpoint in case L-BFGS stalls."

**Likely follow-ups**

- *Why not L-BFGS from the start?* It is fragile far from a minimum in a non-convex landscape and with noisy gradients.
- *Would you use L-BFGS for a large surrogate trained on mini-batches?* Usually not; AdamW with warm-up and decay is the standard.
- *What else helps a PINN reach low loss?* Non-dimensionalization, loss balancing, hard boundary conditions, causal weighting and better architectures.

## References

- [Adam: A Method for Stochastic Optimization (Kingma & Ba)](https://arxiv.org/abs/1412.6980)
- [Decoupled Weight Decay Regularization (AdamW, Loshchilov & Hutter)](https://arxiv.org/abs/1711.05101)
- [Challenges in Training PINNs: A Loss Landscape Perspective (Rathore et al., 2024)](https://arxiv.org/abs/2402.01868)
- J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd edition, Springer, 2006 (L-BFGS and line searches).
- [An Expert's Guide to Training Physics-informed Neural Networks (Wang et al., 2023)](https://arxiv.org/abs/2308.08468)
