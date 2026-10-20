# Surrogate-Based Optimization

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Run an initial set of simulations, train a surrogate with an uncertainty estimate, optimize **on the surrogate** (cheaply, with gradients, evolutionary search or Bayesian optimization), **verify** the best candidates with the real solver, add those results to the training data, retrain, and repeat until verified improvements stop. **Surrogate gradients** are nearly free but are the gradient of an approximation, reliable only where the surrogate is accurate; the **adjoint method** gives exact gradients of the discretized model for about one extra solve. The central danger is **exploitation**: an optimizer searches for inputs where the surrogate predicts a good value, and the easiest such inputs are often where the surrogate is wrong, usually outside the training data; the code below shows an optimizer "finding" a design predicted at 0.52 whose true value is 3.19. Counter it with bounds or trust regions around the data, pessimistic (uncertainty-penalized) objectives, out-of-distribution penalties, and mandatory solver verification. Real problems have **several objectives**, so find the **Pareto front** rather than one weighted optimum, and keep designs **manufacturable** through the parametrization, constraints and a final check.

**On this page**

- [The optimization loop](#the-optimization-loop)
- [Surrogate gradients vs the adjoint](#surrogate-gradients-vs-the-adjoint)
- [Why optimizers exploit surrogates](#why-optimizers-exploit-surrogates)
- [Multiple objectives](#multiple-objectives)
- [Keeping designs manufacturable](#keeping-designs-manufacturable)
- [Code: exploitation, its fixes, and a Pareto front](#code-exploitation-its-fixes-and-a-pareto-front)
- [In an interview](#in-an-interview)

## The optimization loop

```
define objectives, constraints and design variables (with engineers)
        │
initial design of experiments ─► simulate ─► train surrogate (+ uncertainty)
        ▲                                         │
        │                       optimize on the surrogate (gradients, evolutionary, Bayesian optimization)
        │                                         │
   add to data ◄── verify candidates with the solver ◄── select candidates (best predicted + most informative)
```

1. **Formulate the problem with the engineers:** the objective (drag, mass, peak temperature), constraints (stress below allowable, packaging, manufacturing), design variables and their ranges, and what "better" is worth (1% drag? 100 g?).
2. **Initial design of experiments:** space-filling samples, typically ten or more per variable. See [Design of experiments and active learning](design-of-experiments-and-active-learning.md).
3. **Surrogate with uncertainty:** a GP for few variables and samples; an ensemble of networks or a geometric model for many variables or free-form shapes.
4. **Optimize on the surrogate:** gradient-based for many variables and a differentiable pipeline; evolutionary algorithms (CMA-ES, NSGA-II) for non-differentiable or multi-modal problems; Bayesian optimization when every evaluation is precious. See [Bayesian optimization](bayesian-optimization.md).
5. **Select a batch** that balances exploitation (best predicted) and exploration (uncertain regions), and diverse enough not to waste simulations.
6. **Verify with the solver,** add the results, retrain, repeat.
7. **Stop** when verified improvement over several rounds is below what matters, or the budget is spent. **Never report an optimum that has not been verified by the solver.**

## Surrogate gradients vs the adjoint

| | Gradient through a differentiable surrogate | Adjoint solver |
| --- | --- | --- |
| Cost per gradient | Milliseconds (backpropagation), after training | About one extra linear solve per objective, whatever the number of design variables |
| Accuracy | The gradient of the approximation, not an approximation of the gradient: can be wrong even where values are accurate, and meaningless outside the training data | Exact for the discretized model (discrete adjoint) |
| Requirements | A differentiable chain: design parameters → geometry → model input → prediction | An adjoint implementation in the solver (available in some commercial and open CFD codes; automatic in differentiable solvers) |
| Up-front cost | Hundreds to thousands of training simulations | Implementing or licensing the adjoint |
| Best for | Exploring many designs quickly, interactive design, early stages | Final refinement of one design, many design variables (shape optimization with thousands of nodes) |

**Practical notes.**

- **Check surrogate gradients** against finite differences of the solver at a few designs before trusting them; the code below does this check.
- **Smoothness matters:** ReLU networks have piecewise-constant gradients; smooth activations (tanh, GELU, SiLU) give better-behaved gradients for optimization.
- **Hybrid strategies:** explore with the surrogate, then refine the best candidate with adjoint-based optimization on the solver. See [The adjoint method](adjoint-method.md) and [Geodesic convolutions and shape optimization](geodesic-convolutions-and-shape-optimization.md).

## Why optimizers exploit surrogates

**The mechanism.** A surrogate is accurate where it was trained and arbitrary elsewhere. The optimizer's job is to find the input with the best predicted value, and it does not care whether that value is right. Where the surrogate's error happens to be favourable, the optimizer goes there, the same mechanism as adversarial examples. Three common forms:

1. **Extrapolation:** the surrogate extends a trend beyond the data ("drag keeps falling as the rear angle increases") into a region where the physics changes (the flow separates and drag jumps). The [code below](#code-exploitation-its-fixes-and-a-pareto-front) shows exactly this.
2. **Holes in the data:** spurious minima in unsampled pockets inside the design space.
3. **Invalid inputs:** parameter combinations that produce self-intersecting or non-manufacturable geometry, for which the surrogate still outputs a number.

**Mitigations.**

- **Bound the search to the training distribution:** variable bounds, trust regions around the data that grow as data is added, or an out-of-distribution penalty (distance to the nearest training design in parameter or latent space). See [Detecting an out-of-distribution geometry](out-of-distribution-and-generalization.md#detecting-an-out-of-distribution-geometry).
- **Pessimistic objectives:** for the final choice, minimize μ + κσ (a pessimistic bound), so uncertain regions look worse. This is the opposite of Bayesian optimization's acquisition, which rewards uncertainty (μ − κσ) to explore on purpose. Use the optimistic version to pick where to simulate next, the pessimistic one to pick what to recommend.
- **Verify and infill:** simulate the top candidates; each verification that disagrees with the surrogate becomes training data exactly where it was wrong, so the loop corrects itself.
- **Validity checks** on geometry before evaluating the surrogate.
- **Robust optimization:** optimize the expected performance under manufacturing scatter, which also discourages sharp, fragile optima.

## Multiple objectives

**Trade-offs are the norm:** drag vs downforce, mass vs stiffness, pressure drop vs heat transfer, cost vs performance.

- **Weighted sums** f = w₁f₁ + w₂f₂ are simple but need weights chosen in advance and can only find points on convex parts of the Pareto front.
- **ε-constraint:** optimize one objective subject to bounds on the others; vary the bounds to trace the front.
- **Pareto methods:** a design is **Pareto-optimal** if no other design is at least as good in every objective and strictly better in one. Evolutionary algorithms (NSGA-II, NSGA-III) evolve a population toward the whole front; multi-objective Bayesian optimization uses expected hypervolume improvement (EHVI) or ParEGO.
- **Present the front to the engineers**, not a single answer: the choice of trade-off is a design decision. Surrogates make it cheap to explore the front interactively, then verify the chosen points with the solver.

## Keeping designs manufacturable

**Typical constraints:** minimum wall thickness and feature size; draft angles for casting and moulding; overhang angles and support for additive manufacturing; tool access for machining; standard sizes (sheet thicknesses, fasteners); packaging and keep-out zones; regulations (pedestrian-safety zones in cars, for example); stress, buckling and fatigue limits.

**How to enforce them:**

1. **In the parametrization:** choose design variables that can only produce feasible designs (CAD parameters with ranges, morphing that preserves hard points); this is the most robust option.
2. **As constraints** in the optimizer: bounds, inequality constraints with their own surrogates, or penalty terms.
3. **Feasibility classifiers:** learn which parameter combinations produce valid geometry or converged simulations, and treat the predicted probability as a constraint.
4. **Filters and projections** in topology optimization (minimum length scale, overhang filters). See [Topology optimization](topology-optimization.md).
5. **A post-check and repair step:** check every candidate against manufacturing rules before simulating it.
6. **Final verification with the solver** of the safety factor and other limits; never certify a structural design on the surrogate's prediction. See [Predicting a safety factor with a surrogate](von-mises-stress-and-yield.md#predicting-a-safety-factor-with-a-surrogate).

## Code: exploitation, its fixes, and a Pareto front

The true objective falls steadily as x₀ grows, until a regime change just beyond the sampled range (x₀ > 1.1, like flow separation) makes it rise steeply; the true optimum, 0.434, is at x₀ ≈ 1.16. An ensemble of five networks is trained on 40 simulations inside [0, 1]², then three strategies optimize on it. The code also checks a surrogate gradient against finite differences of the true function and extracts the Pareto set of a two-objective problem.

```python
import numpy as np
import torch, torch.nn as nn

torch.manual_seed(0)
def true_objective(x):
    """'Drag' of a 2-parameter design. It falls steadily as x0 grows, until a regime change just beyond the
    sampled range (x0 > 1.1, think of flow separation) makes it rise steeply."""
    return 1 + 3 * (x[..., 1] - 0.3) ** 2 - 0.5 * x[..., 0] + 4 * torch.relu(x[..., 0] - 1.1) ** 2

X = torch.rand(40, 2)                                         # 40 simulations inside the box [0, 1]^2
y = true_objective(X)

def train_member(seed):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(2000):
        loss = ((net(X).squeeze(-1) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net

ensemble = [train_member(s) for s in range(5)]
def predict(x):                                               # ensemble mean and standard deviation
    out = torch.stack([m(x).squeeze(-1) for m in ensemble])
    return out.mean(0), out.std(0)

def optimize(objective, box=None, steps=500):
    x = torch.tensor([[0.5, 0.5]], requires_grad=True)
    opt = torch.optim.Adam([x], lr=0.02)
    for _ in range(steps):
        loss = objective(x).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if box is not None:
            with torch.no_grad():
                x.clamp_(*box)                                # stay inside the trained design space
    return x.detach()

runs = {"mean, unbounded": optimize(lambda x: predict(x)[0]),
        "mean, inside the training box": optimize(lambda x: predict(x)[0], box=(0.0, 1.0)),
        "mean + 2 std (pessimistic), unbounded": optimize(lambda x: predict(x)[0] + 2 * predict(x)[1])}
print(f"{'strategy':<40} {'design':>16} {'predicted':>10} {'true':>8}")
for name, x in runs.items():
    with torch.no_grad():
        print(f"{name:<40} ({x[0, 0]:6.2f}, {x[0, 1]:5.2f}) {predict(x)[0].item():>10.3f} {true_objective(x).item():>8.3f}")
print(f"best of the 40 training simulations: {y.min().item():.3f}; "
      f"true optimum {true_objective(torch.tensor([[1.1625, 0.3]])).item():.3f} at (1.16, 0.30)")

# Check a surrogate gradient against central finite differences of the "solver" at one design.
x0 = torch.tensor([[0.3, 0.6]], requires_grad=True)
g_surrogate = torch.autograd.grad(predict(x0)[0].sum(), x0)[0][0]
eps = 1e-4
g_fd = [(true_objective(x0.detach() + eps * e) - true_objective(x0.detach() - eps * e)).item() / (2 * eps) for e in torch.eye(2)[:, None, :]]
print(f"gradient at (0.3, 0.6): surrogate {np.round(g_surrogate.numpy(), 2)}, solver finite differences {np.round(g_fd, 2)}")

# Two objectives: find the non-dominated (Pareto) designs among 300 candidates (both objectives minimized).
rng = np.random.default_rng(0)
mass = rng.uniform(1, 3, 300)
compliance = 1 / mass + rng.normal(0, 0.05, 300)             # lighter designs are less stiff
F = np.stack([mass, compliance], 1)
dominated = np.array([np.any(np.all(F <= f, 1) & np.any(F < f, 1)) for f in F])
print(f"{(~dominated).sum()} of 300 designs are Pareto-optimal (no other design is better in both objectives)")
```

Output:

```
strategy                                           design  predicted     true
mean, unbounded                          (  1.99,  0.18)      0.515    3.193
mean, inside the training box            (  1.00,  0.29)      0.554    0.500
mean + 2 std (pessimistic), unbounded    (  1.17,  0.17)      0.554    0.486
best of the 40 training simulations: 0.609; true optimum 0.434 at (1.16, 0.30)
gradient at (0.3, 0.6): surrogate [-0.5  1.8], solver finite differences [-0.5  1.8]
29 of 300 designs are Pareto-optimal (no other design is better in both objectives)
```

**What it shows.**

- **Unbounded, the optimizer exploits the surrogate:** it walks to x₀ ≈ 2, where the surrogate extrapolates the downward trend (predicted 0.52) but the truth is 3.19, more than five times worse than the best design already simulated. Nothing in the surrogate's output warns of this.
- **Bounding the search to the training box** gives a safe, verified-looking answer (true 0.500), better than all 40 training designs but short of the true optimum, which lies just outside the box.
- **The pessimistic objective** (mean plus two ensemble standard deviations) lets the optimizer leave the box only as far as the ensemble agrees, stopping at x₀ ≈ 1.17 with a true value of 0.486. It came close to the optimum here, but partly by luck: the next step in the loop must be to simulate that design and the region around it, because the cliff is exactly where the surrogate has no data.
- Inside the data, the surrogate gradient matches the solver's finite-difference gradient; that check is cheap and worth doing before gradient-based optimization.
- Of 300 random candidates, 29 are Pareto-optimal; those are the ones worth showing to engineers.

## In an interview

**Thirty-second answer to "How do you use a surrogate inside an optimization loop?"** "Start with a space-filling set of simulations, train a surrogate with uncertainty, optimize on the surrogate, verify the best few candidates with the solver, add them to the data, retrain and repeat until verified improvements level off. The main risk is that the optimizer exploits surrogate errors, typically by extrapolating a trend out of the data, so I bound the search to the training distribution or use a pessimistic objective with an ensemble, check out-of-distribution scores, and never report an optimum the solver has not confirmed."

**Likely follow-ups**

- *Surrogate gradients or the adjoint?* Surrogate gradients for cheap exploration inside the data; the adjoint for exact gradients and final refinement; check one against the other.
- *How do you handle multiple objectives?* Find the Pareto front (NSGA-II or expected hypervolume improvement) and let engineers choose the trade-off.
- *How do you keep designs manufacturable?* Through the parametrization first, then constraints and feasibility models, then a rule check and solver verification of every recommended design.

## References

- [A Tutorial on Bayesian Optimization (Frazier)](https://arxiv.org/abs/1807.02811)
- A. I. J. Forrester and A. J. Keane, "Recent advances in surrogate-based optimization", *Progress in Aerospace Sciences*, 2009.
- K. Deb et al., "A fast and elitist multiobjective genetic algorithm: NSGA-II", *IEEE Transactions on Evolutionary Computation*, 2002.
- [Geodesic Convolutional Shape Optimization (Baqué et al.)](https://arxiv.org/abs/1802.04016)
- [Conservative Objective Models for Effective Offline Model-Based Optimization (Trabucco et al.)](https://arxiv.org/abs/2107.06882), on surrogate exploitation in offline optimization
- [BoTorch](https://github.com/pytorch/botorch), [pymoo (multi-objective optimization)](https://pymoo.org/) and [Optuna](https://github.com/optuna/optuna)
