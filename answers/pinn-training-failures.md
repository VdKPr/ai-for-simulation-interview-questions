# PINN Training Failures

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** PINNs fail to train for a handful of well-understood reasons. The loss terms **compete**: gradients of the PDE residual and the boundary terms can differ by orders of magnitude, so the optimizer satisfies one and ignores the other. The loss is **ill-conditioned** because it contains derivatives of the network, which makes gradient descent slow and erratic. Networks have **spectral bias**, so high-frequency and multi-scale solutions converge extremely slowly. Time-dependent PINNs can **violate causality**, fitting late times before early times are right and converging to wrong solutions. And information from boundaries can fail to **propagate** into the domain, leaving a trivial interior solution. Each failure has known remedies: non-dimensionalization, adaptive loss weights, hard constraints, Fourier features, causal weighting, adaptive sampling, curriculum or time-marching training, domain decomposition, and Adam followed by L-BFGS. The way to find out which one is happening is to log every loss term separately and compare with a reference solution.

**On this page**

- [Why PINNs fail to train](#why-pinns-fail-to-train)
- [Loss imbalance and adaptive weights](#loss-imbalance-and-adaptive-weights)
- [Code: balancing the loss terms by their gradient norms](#code-balancing-the-loss-terms-by-their-gradient-norms)
- [Causality in time-dependent PINNs](#causality-in-time-dependent-pinns)
- [High-frequency and multi-scale solutions](#high-frequency-and-multi-scale-solutions)
- [Propagation failures and sampling](#propagation-failures-and-sampling)
- [How to diagnose a failing PINN](#how-to-diagnose-a-failing-pinn)
- [In an interview](#in-an-interview)

## Why PINNs fail to train

| Failure | What you observe | Main remedies |
| --- | --- | --- |
| **Loss imbalance (gradient pathologies)** | Residual falls but boundary conditions are violated, or the reverse | Non-dimensionalize; adaptive weights; hard boundary conditions |
| **Ill-conditioned loss** | Slow, erratic training even for simple problems; Adam plateaus | Adam then L-BFGS; second-order methods; better scaling |
| **Spectral bias** | Smooth fit with high-frequency or multi-scale features missing | Fourier features; sine activations; domain decomposition |
| **Stiff, convection- or reaction-dominated PDEs** | Training fails as the convection or reaction coefficient grows, although the network could represent the solution | Curriculum (start easy, increase the coefficient); time-window (sequence-to-sequence) training |
| **Causality violation** | Late times fitted before early times; wrong solution, especially for chaotic dynamics | Causal weighting of residuals in time; time-marching |
| **Propagation failure** | Trivial or wrong interior solution; boundary information does not spread | Adaptive (residual-based) sampling; more boundary points; hard constraints |
| **Too few collocation points** | Residual tiny at the points, large in between | More points; resampling; adaptive sampling |

**Ill-conditioning in one sentence.** Rathore et al. (2024) showed that the differential operators in the residual make the PINN loss badly conditioned, explained why combining Adam with L-BFGS works better than either alone, and proposed a further second-order method. See [Adam then L-BFGS](optimizers-for-physics-models.md#adam-then-l-bfgs).

**Stiff and convection-dominated problems.** Krishnapriyan et al. (2021) showed that on convection, reaction and reaction-diffusion problems, PINNs fail as the coefficients grow, not because the network is too small but because the optimization problem becomes very hard. Two simple remedies worked: a **curriculum** that starts with a small coefficient and increases it, and **sequence-to-sequence** training that solves short time windows one after another.

## Loss imbalance and adaptive weights

**The problem.** The total loss is a weighted sum, L = λ_r L_r + λ_b L_b (+ initial and data terms). Its gradient is λ_r ∇L_r + λ_b ∇L_b. If ∇L_r is a thousand times larger than ∇L_b, gradient descent effectively ignores the boundary term: the network finds *a* function with a small residual, but not the one with the right boundary values. Large residuals are typical whenever the PDE contains large coefficients or high derivatives: for a solution with frequency k, the second derivative scales like k², so the residual's gradient scales like k⁴.

**Two views of the same problem:**

- **Gradient pathologies** (Wang, Teng and Perdikaris, 2021): during training, the gradients of different terms differ by orders of magnitude, which is a stiffness in the gradient flow. Their remedy, *learning-rate annealing*, sets the weights from gradient statistics during training (originally the ratio of the maximum residual gradient to the mean boundary gradient, smoothed with a moving average).
- **The neural tangent kernel view** (Wang, Yu and Perdikaris, 2022): in the NTK regime, each term converges at a rate set by the eigenvalues of its block of the kernel; residual and boundary blocks can have very different eigenvalues, so their errors converge at very different speeds. Weighting by the traces of the NTK blocks equalizes the rates.

**Practical weighting schemes:**

1. **Non-dimensionalize first**, so all terms are of comparable size before any weighting. See [Why non-dimensionalize](nondimensionalization-and-normalization.md#why-non-dimensionalize).
2. **Gradient-norm balancing** (used in the Expert's Guide by Wang et al., 2023): every few hundred steps, set each weight to (sum of all gradient norms) / (gradient norm of this term), smoothed with a moving average. Each term then contributes a similar gradient magnitude. The code below shows it fixing a broken boundary condition.
3. **NTK-based weights:** equalize the convergence rates; more expensive to compute.
4. **Self-adaptive point-wise weights** (McClenny and Braga-Neto): a trainable weight per collocation point, updated by gradient *ascent*, so the hard points get more attention, like a soft attention mask.
5. **Remove the term altogether** by enforcing the boundary condition exactly in the network output. See [Soft vs hard boundary conditions in PINNs](hard-vs-soft-constraints.md#soft-vs-hard-boundary-conditions-in-pinns).

Watch for weights that grow without bound as a term goes to zero; use moving averages and check that the weights settle.

## Code: balancing the loss terms by their gradient norms

A PINN for −u″ = (kπ)² sin(kπx) + 1 with u(0) = u(1) = 0 and k = 4. With equal weights, the large residual term dominates and the boundary values drift. Balancing by gradient norms fixes it.

```python
import torch, torch.nn as nn

def grad_norm(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.sqrt(sum((g ** 2).sum() for g in grads if g is not None))

def train(adaptive, steps=3000, k=4.0, seed=0):
    """PINN for -u'' = (k pi)^2 sin(k pi x) + 1 on (0, 1), u(0) = u(1) = 0. The residual term is large
    (it scales with (k pi)^4), so with equal weights it dominates the gradients and the boundary drifts."""
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    params = list(net.parameters())
    x_b = torch.tensor([[0.0], [1.0]])
    w_r, w_b = 1.0, 1.0
    for step in range(steps):
        x = torch.rand(256, 1, requires_grad=True)
        u = net(x)
        du = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        d2u = torch.autograd.grad(du, x, torch.ones_like(du), create_graph=True)[0]
        loss_r = ((-d2u - (k * torch.pi) ** 2 * torch.sin(k * torch.pi * x) - 1.0) ** 2).mean()
        loss_b = (net(x_b) ** 2).mean()
        if adaptive and step % 100 == 0:                     # balance gradient norms (Wang et al., Expert's Guide)
            n_r, n_b = grad_norm(loss_r, params), grad_norm(loss_b, params)
            total = (n_r + n_b).item()
            w_r = 0.9 * w_r + 0.1 * total / (n_r.item() + 1e-12)
            w_b = 0.9 * w_b + 0.1 * total / (n_b.item() + 1e-12)
        loss = w_r * loss_r + w_b * loss_b
        opt.zero_grad(); loss.backward(); opt.step()
    xt = torch.linspace(0, 1, 201).unsqueeze(1)
    exact = torch.sin(k * torch.pi * xt) + 0.5 * xt * (1 - xt)
    with torch.no_grad():
        err = (net(xt) - exact).abs().max().item()
        bc = net(x_b).abs().max().item()
    return err, bc, w_b / w_r

results = {}
for adaptive in [False, True]:
    results[adaptive] = train(adaptive)
    err, bc, ratio = results[adaptive]
    print(f"{'gradient-balanced' if adaptive else 'equal weights    '}: boundary/residual weight {ratio:8.1f}, "
          f"boundary error {bc:.3f}, max error {err:.3f}")
```

Output:

```
equal weights    : boundary/residual weight      1.0, boundary error 0.347, max error 0.347
gradient-balanced: boundary/residual weight   2181.0, boundary error 0.002, max error 0.002
```

With equal weights the network satisfies the PDE in the interior but with the wrong boundary values, so the whole solution is off by about 0.35. Balancing raises the boundary weight to about 2,000 times the residual weight and the error falls by more than a hundred times.

## Causality in time-dependent PINNs

**The problem.** A classical time-stepping solver computes t = 0, then t = Δt, then 2Δt: information flows forward in time by construction. A continuous-time PINN minimizes the residual over all space-time points *at once*. Nothing stops it from reducing the residual at late times first, fitting a solution that satisfies the PDE locally but is not the one that evolves from the actual initial condition. For dissipative problems this may still work; for chaotic or turbulent dynamics, and for long time horizons, it often converges to a wrong (sometimes trivial) solution.

**The fix: causal weighting** (Wang, Sankaran and Perdikaris, 2022). Split the time interval into chunks t₁ < t₂ < … < t_M, compute the residual loss L_r(t_i) in each chunk, and weight later chunks by how well the earlier ones are already satisfied:

```math
w_i = \exp\!\left(-\varepsilon \sum_{k=1}^{i-1} \mathcal{L}_r(t_k)\right), \qquad \mathcal{L}_r = \frac{1}{M}\sum_{i=1}^{M} w_i\,\mathcal{L}_r(t_i)
```

- While the early times still have a large residual, the weights of later times are close to zero, so the network first fits the beginning.
- As early residuals shrink, later weights rise towards one, and training "marches" forward in time.
- The weights are treated as constants (no gradient flows through them).
- **ε** controls how strictly causality is enforced; the paper uses an annealing schedule of increasing ε.
- **Convergence check:** when all weights are close to one, the residual is small everywhere in time.

With this change alone, the paper reported successful PINN solutions of chaotic and turbulent benchmarks where standard PINNs failed: the Lorenz system, the Kuramoto-Sivashinsky equation in its chaotic regime, and the Navier–Stokes equations in a turbulent regime.

**Alternatives with the same idea:** **time-marching** (train on [0, T₁], use its final state as the initial condition for [T₁, T₂], and so on), and the **sequence-to-sequence** training of Krishnapriyan et al.

## High-frequency and multi-scale solutions

**The problem.** Spectral bias: gradient descent fits low frequencies first and high frequencies very slowly. In NTK terms, PINNs learn functions along the dominant eigen-directions of their kernel, and high-frequency eigenfunctions have tiny eigenvalues. Solutions with sharp layers, oscillations or several scales at once converge very slowly or not at all. See [Spectral bias](spectral-bias-and-fourier-features.md#spectral-bias).

**Remedies:**

1. **Fourier feature embeddings** of the input coordinates. Wang, Wang and Perdikaris (2021) analysed the eigenvector bias of Fourier feature networks and proposed spatio-temporal and **multi-scale Fourier feature** architectures (several frequency scales in parallel) for PINNs with multi-scale behaviour, including wave propagation and reaction-diffusion problems. See [Fourier features](spectral-bias-and-fourier-features.md#fourier-features-and-positional-encodings).
2. **Sine activations** (SIREN-style networks) with careful initialization.
3. **Domain decomposition.** Finite-basis PINNs (FBPINNs, Moseley, Markham and Nissen-Meyer) split the domain into overlapping subdomains, each with its own small network combined through smooth window functions. A high frequency over the whole domain is a low frequency within each subdomain, which counters spectral bias and scales to large domains.
4. **Adaptive sampling** in regions of steep gradients. See [Collocation points](pinns.md#collocation-points).
5. **Curriculum over frequency or coefficients:** solve a smoother version first and increase the difficulty gradually.
6. **Scaling of the output**, so that the network does not have to produce a very large or very small amplitude.

## Propagation failures and sampling

Training a PINN relies on the solution information at boundaries and initial conditions *propagating* into the interior through the residual. Daw et al. (2023) showed that with poor sampling, PINNs can get stuck at trivial solutions: the residual becomes highly imbalanced, small almost everywhere and large in a few narrow regions, which uniform sampling barely sees. Their *Retain-Resample-Release* (R3) sampling keeps high-residual points, resamples the rest and releases points once they are fitted, so that the training effort follows the propagation front. More generally, residual-based adaptive sampling (Wu et al.) addresses the same issue. See [Collocation points](pinns.md#collocation-points).

## How to diagnose a failing PINN

1. **Log every loss term separately** (residual, each boundary, initial, data), not just the total.
2. **Check boundary and initial values directly** on a fine set of points.
3. **Plot the residual over the domain:** where is it large? Near boundaries, layers, late times?
4. **Compare with a reference solution** (FEM, or an analytical solution) on a simpler version of the case. A small loss is not proof of a correct solution.
5. **Plot the gradient norms** of each term over training: orders of magnitude apart means imbalance.
6. **Make it easier and work back up:** smaller coefficient, shorter time, lower frequency, then increase.
7. **Check the formulation:** non-dimensionalized? Smooth activation? Enough points? Hard constraints possible?

A good starting recipe that combines the remedies is in [PINN training recipe](pinn-training-recipe.md).

## In an interview

**Thirty-second answer to "Why do PINNs often fail to train?"** "Mostly optimization, not capacity. The loss terms compete: residual gradients can be orders of magnitude larger than boundary gradients, so boundary conditions get ignored. The residual contains derivatives, which makes the loss ill-conditioned. Spectral bias makes high-frequency and multi-scale solutions very slow to learn. Time-dependent PINNs can fit late times before early ones and violate causality. And boundary information can fail to propagate into the domain. Remedies are non-dimensionalization, adaptive loss weights or hard constraints, Fourier features or domain decomposition, causal weighting or time-marching, adaptive sampling, and Adam followed by L-BFGS."

**Likely follow-ups**

- *How do adaptive weights work?* Set each term's weight from gradient statistics (for example inversely proportional to its gradient norm) or NTK eigenvalues, updated every few hundred steps with a moving average.
- *What is the causality problem?* A continuous-time PINN can minimize late-time residuals before the early-time solution is right; causal weights exp(−ε Σ earlier residuals) fix the order.
- *How would you solve a multi-scale problem with a PINN?* Multi-scale Fourier features, domain decomposition (FBPINNs), adaptive sampling, and a curriculum; or reconsider whether a PINN is the right tool.

## References

- [Understanding and mitigating gradient pathologies in physics-informed neural networks (Wang, Teng & Perdikaris)](https://arxiv.org/abs/2001.04536)
- [When and why PINNs fail to train: A neural tangent kernel perspective (Wang, Yu & Perdikaris)](https://arxiv.org/abs/2007.14527)
- [Characterizing possible failure modes in physics-informed neural networks (Krishnapriyan et al.)](https://arxiv.org/abs/2109.01050)
- [Respecting causality is all you need for training physics-informed neural networks (Wang, Sankaran & Perdikaris)](https://arxiv.org/abs/2203.07404)
- [On the eigenvector bias of Fourier feature networks (Wang, Wang & Perdikaris)](https://arxiv.org/abs/2012.10047)
- [Finite Basis Physics-Informed Neural Networks (FBPINNs) (Moseley, Markham & Nissen-Meyer)](https://arxiv.org/abs/2107.07871)
- [Self-Adaptive Physics-Informed Neural Networks using a Soft Attention Mechanism (McClenny & Braga-Neto)](https://arxiv.org/abs/2009.04544)
- [Mitigating Propagation Failures in PINNs using Retain-Resample-Release (R3) Sampling (Daw et al.)](https://arxiv.org/abs/2207.02338)
- [Challenges in Training PINNs: A Loss Landscape Perspective (Rathore et al.)](https://arxiv.org/abs/2402.01868)
- [An Expert's Guide to Training PINNs (Wang et al., 2023)](https://arxiv.org/abs/2308.08468) and its code, [jaxpi](https://github.com/PredictiveIntelligenceLab/jaxpi)
- [ETH Zürich AISE: PINNs, Limitations and Extensions Part 1 (video)](https://www.youtube.com/watch?v=S11QK8baGVI) and [Part 2 (video)](https://www.youtube.com/watch?v=NFtE1pyD5LA)
