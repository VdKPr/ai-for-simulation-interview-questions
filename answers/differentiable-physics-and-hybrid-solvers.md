# Differentiable Physics and Hybrid Solvers

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Instead of replacing the solver, put machine learning **inside** it, or make the solver itself **differentiable** so a network can be trained through it. **Unrolled training** backpropagates a loss through several consecutive time steps, so the model learns from the consequences of its own predictions; **solver-in-the-loop** (Um et al., NeurIPS 2020) does this with a differentiable numerical solver in the unroll, so a learned correction sees the realistic inputs it will receive when coupled to the solver. **Correcting a coarse solver** means running a cheap, under-resolved simulation and letting a network fix its error each step: Kochkov et al. (PNAS 2021) made 2D turbulence simulations as accurate as baselines with 8-10 times finer resolution in each direction, a 40-80× speed-up. **Differentiable simulators**, solvers written in frameworks with automatic differentiation (JAX-CFD, PhiFlow, JAX-FEM, Warp), give exact gradients of any output with respect to inputs, parameters and geometry: reverse-mode automatic differentiation of a solver is a discrete adjoint. Hybrids keep the solver's structure and guarantees, which makes them far easier to trust than a pure surrogate.

**On this page**

- [Hybrid patterns](#hybrid-patterns)
- [Unrolled training and solver-in-the-loop](#unrolled-training-and-solver-in-the-loop)
- [Correcting a coarse solver](#correcting-a-coarse-solver)
- [Differentiable simulators](#differentiable-simulators)
- [Code: a learned correction for a coarse Burgers solver](#code-a-learned-correction-for-a-coarse-burgers-solver)
- [In an interview](#in-an-interview)

## Hybrid patterns

| Pattern | What ML does | What the solver still guarantees |
| --- | --- | --- |
| **Warm start / initial guess** | Predicts a starting solution for Newton's method or an iterative solver | Convergence to the same tolerance; ML only saves iterations. See [Speeding up a solver with ML without changing its answer](newton-raphson-and-nonlinear-solvers.md#speeding-up-a-solver-with-ml-without-changing-its-answer) |
| **Learned preconditioner** | Approximates the inverse operator to speed up Krylov methods | The Krylov method converges to the true solution |
| **Learned closure** | Replaces an empirical model term (turbulence closure, subgrid stress, constitutive law) | Conservation and structure of the host solver |
| **Coarse-grid correction** | Corrects the error of a cheap, under-resolved solver each step | Stability and conservation of the coarse scheme, if the correction is built to respect them |
| **Learned interpolation or flux** | Replaces the interpolation of a finite-volume scheme with learned coefficients | Finite-volume conservation (Kochkov et al.; Bar-Sinai et al.) |
| **ML proposes, solver verifies** | A surrogate screens designs; the solver checks the finalists | The final answer is a solver answer |

## Unrolled training and solver-in-the-loop

**Unrolled training.** Instead of training on single steps, apply the model K times from a true state and compare each intermediate prediction with the reference:

```math
\mathcal L(\theta) = \sum_{s=1}^{K} \big\lVert \hat u_{t+s} - u_{t+s} \big\rVert^2, \qquad \hat u_{t+s} = G_\theta(\hat u_{t+s-1}),\ \ \hat u_t = u_t
```

Gradients flow back through all K steps (backpropagation through time), so the model is penalized for errors that **grow** over the unroll, not only for one-step error. This directly targets rollout stability. See [Why one-step models drift](autoregressive-rollouts.md#why-one-step-models-drift).

**Solver-in-the-loop (Um, Brand, Fei, Holl and Thuerey, NeurIPS 2020).** When the model is a **correction** to a numerical solver, the step is G_θ(u) = P(u) + C_θ(P(u)), with P the solver. Training the correction on precomputed pairs (the "non-interacting" setup) shows it only states from the reference simulation, but in use it receives states that are already shaped by its own past corrections. Putting a **differentiable solver** inside the unroll lets the model see, and receive gradients from, that interaction. They showed this produced realistic input distributions during training, stable rollouts of several hundred steps, and better accuracy than supervised variants, from advection-diffusion problems up to 3D Navier-Stokes.

**Costs and pitfalls.**

- Memory grows linearly with K; use gradient checkpointing for long unrolls. See [Gradient checkpointing](scaling-to-large-meshes.md#gradient-checkpointing).
- Gradients can explode or vanish through long unrolls, especially for chaotic dynamics; use gradient clipping, a curriculum (start with short unrolls and lengthen them), or truncated backpropagation.
- The solver must be differentiable, which rules out most commercial codes. The **pushforward trick** is a cheap approximation that needs no solver gradients. See [The pushforward trick](autoregressive-rollouts.md#the-pushforward-trick).
- APEBench (Koehler et al., NeurIPS 2024) provides a taxonomy of unrolled training strategies and a differentiable pseudo-spectral solver to study them.

## Correcting a coarse solver

**Why it works.** A coarse simulation is cheap but wrong where it cannot resolve the physics (steep fronts, small eddies). Its error is often systematic and local, depending on the resolved state, which makes it learnable. The solver supplies everything it does well (transport, conservation, the effect of boundary conditions), and the network only has to learn the missing subgrid effect.

**Kochkov et al., "Machine learning-accelerated computational fluid dynamics" (PNAS 2021).** Learned components inside a finite-volume solver for 2D turbulence (learned interpolation of fluxes, so mass conservation stays exact), trained end-to-end through the differentiable solver (JAX-CFD). Results: as accurate as baseline solvers with **8-10× finer resolution in each spatial dimension**, a **40-80× computational speed-up**, stable long simulations, and generalization to forcing functions and Reynolds numbers outside the training set.

**Other examples.** Learned coarse models for turbulence (Stachenfeld et al., 2021), learned discretizations for 1D PDEs (Bar-Sinai et al., 2019), learned subgrid-scale models for LES, and corrections of low-fidelity structural models. See also [Multi-fidelity learning](multi-fidelity-learning.md), which corrects a cheap model with a learned discrepancy without running them coupled.

**Design rules.**

1. Predict a **correction** (or a flux, or a closure term), not the full state, scaled by the time step.
2. Keep the **conservative structure**: corrections applied as fluxes between cells conserve mass by construction.
3. Train with **unrolls through the solver** if possible; otherwise with noise or pushforward.
4. Evaluate long rollouts, statistics and spectra, as for any time-stepper. See [Evaluating a rollout](autoregressive-rollouts.md#evaluating-a-rollout).
5. Test generalization to other forcing, Reynolds numbers and domain sizes; local corrections tend to transfer better than global surrogates.

## Differentiable simulators

**What they are.** Simulators written in automatic-differentiation frameworks (JAX, PyTorch, Taichi, Warp), or with hand-written adjoints, so that the derivative of any output with respect to any input (initial condition, parameters, boundary conditions, geometry, material fields, network weights) is available.

**What they enable.**

- **Gradient-based optimization and inverse problems** without deriving an adjoint by hand. Reverse-mode automatic differentiation of the discretized solver **is** the discrete adjoint method. See [The adjoint method](adjoint-method.md) and [Inverse problems and regularization](inverse-problems-and-regularization.md).
- **End-to-end training** of learned components inside the solver (solver-in-the-loop, learned closures).
- **Sensitivity analysis** and uncertainty propagation via gradients.

**Libraries.**

| Library | Domain |
| --- | --- |
| [JAX-CFD](https://github.com/google/jax-cfd) | Incompressible flow, spectral and finite-volume methods (Kochkov et al.) |
| [PhiFlow](https://github.com/tum-pbs/PhiFlow) | Fluids and PDEs on grids, multiple backends (TUM, Thuerey group) |
| [JAX-FEM](https://github.com/deepmodeling/jax-fem) | Finite elements in JAX |
| [NVIDIA Warp](https://github.com/NVIDIA/warp) | Differentiable GPU kernels for simulation |
| [APEBench / Exponax](https://github.com/Ceyron/apebench) | Pseudo-spectral solvers for benchmarking |
| [dolfin-adjoint](https://www.dolfin-adjoint.org/) | Automatic adjoints for FEniCS models |

**Limits.**

- **Commercial solvers are rarely differentiable**; adjoint solvers in commercial CFD exist but expose only specific gradients (shape sensitivities, for example).
- **Memory:** backpropagating through long time histories stores every intermediate state; checkpointing trades compute for memory.
- **Chaotic systems:** gradients over long horizons explode and become useless (the sensitivity of a chaotic trajectory to its initial condition grows exponentially); use short windows, shadowing methods or statistical objectives.
- **Non-differentiable steps:** remeshing, contact detection, adaptive time stepping and discrete decisions need smoothing or special treatment.
- **Performance:** a research differentiable solver is usually slower than an optimized production code for the forward problem.

## Code: a learned correction for a coarse Burgers solver

A reference solution of Burgers' equation (ν = 0.003) is computed on 512 points and downsampled to 32. A coarse pseudo-spectral solver on 32 points, written in PyTorch so it is differentiable, cannot resolve the steep fronts and becomes unstable. A small convolutional network adds a correction after every coarse step. It is trained twice: on single steps, and **unrolled through the coarse solver** for 4 steps (solver-in-the-loop). Training takes under a minute on a CPU.

```python
import torch, torch.nn as nn

torch.manual_seed(0)
nu = 0.003

def spectral_burgers(u, dt, n_sub):
    """Differentiable pseudo-spectral Burgers solver (periodic domain [0, 1)), RK2 in time, 2/3-rule dealiasing."""
    N = u.shape[-1]
    k = torch.fft.rfftfreq(N, d=1.0 / N) * 2 * torch.pi
    mask = (torch.arange(k.numel()) < N // 3).to(u.dtype)
    def rhs(v):
        vh = torch.fft.rfft(v) * mask
        return -torch.fft.irfft(vh, n=N) * torch.fft.irfft(1j * k * vh, n=N) + nu * torch.fft.irfft(-(k ** 2) * vh, n=N)
    for _ in range(n_sub):
        u1 = u + dt * rhs(u)
        u = 0.5 * (u + u1 + dt * rhs(u1))
    return u

def downsample(u, factor):                          # spectral truncation onto the coarse grid
    n = u.shape[-1] // factor
    return torch.fft.irfft(torch.fft.rfft(u)[..., : n // 2 + 1], n=n) / factor

fine_N, factor, DT = 512, 16, 0.02                  # fine grid, coarsening factor, one model step
def reference_trajectories(n_traj, steps, gen):
    x = torch.arange(fine_N) / fine_N
    u = sum(torch.randn(n_traj, 1, generator=gen) / m
            * torch.sin(2 * torch.pi * m * x + 2 * torch.pi * torch.rand(n_traj, 1, generator=gen)) for m in range(1, 5))
    u = u / u.abs().amax(1, keepdim=True)
    out = [downsample(u, factor)]
    for _ in range(steps):
        u = spectral_burgers(u, DT / 40, 40)
        out.append(downsample(u, factor))
    return torch.stack(out, 1)                      # (n_traj, steps + 1, 32)

gen = torch.Generator().manual_seed(0)
with torch.no_grad():
    train = reference_trajectories(64, 40, gen)
    test = reference_trajectories(16, 60, gen)

coarse = lambda u: spectral_burgers(u, DT / 4, 4)   # the cheap solver on the 32-point grid

class Corrector(nn.Module):
    def __init__(self):
        super().__init__()
        conv = lambda i, o: nn.Conv1d(i, o, 5, padding=2, padding_mode="circular")
        self.net = nn.Sequential(conv(1, 32), nn.GELU(), conv(32, 32), nn.GELU(), conv(32, 1))
    def forward(self, u):
        return self.net(u[:, None])[:, 0]

def hybrid(model, u):                               # one coarse solver step, then the learned correction
    u = coarse(u)
    return u + DT * model(u)

def fit(unroll, iters=1500):
    torch.manual_seed(1)
    model = Corrector()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for it in range(iters):
        traj = torch.randint(0, 64, (32,))
        t0 = torch.randint(0, 40 - unroll + 1, (32,))
        u, loss = train[traj, t0], 0.0
        for s in range(1, unroll + 1):              # gradients flow through the solver and the model
            u = hybrid(model, u)
            loss = loss + ((u - train[traj, t0 + s]) ** 2).mean()
        opt.zero_grad()
        (loss / unroll).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model

@torch.no_grad()
def rollout_errors(stepper, steps=60):
    u, errs = test[:, 0], []
    for t in range(1, steps + 1):
        u = stepper(u)
        errs.append(((u - test[:, t]).norm(dim=1) / test[:, t].norm(dim=1)).mean().item())
    return errs

e = rollout_errors(coarse)
print(f"coarse solver alone:        error after 10 steps {e[9]:.4f}, 30 steps {e[29]:.4f}, 60 steps {e[59]:.4f}")
for unroll in [1, 4]:
    model = fit(unroll)
    e = rollout_errors(lambda u: hybrid(model, u))
    print(f"correction, unroll {unroll} steps: error after 10 steps {e[9]:.4f}, 30 steps {e[29]:.4f}, 60 steps {e[59]:.4f}")
```

Output:

```
coarse solver alone:        error after 10 steps 0.6558, 30 steps 2.7006, 60 steps 6.1525
correction, unroll 1 steps: error after 10 steps 0.0295, 30 steps 0.0482, 60 steps 0.0614
correction, unroll 4 steps: error after 10 steps 0.0285, 30 steps 0.0359, 60 steps 0.0582
```

**What it shows.**

- On 32 points the coarse solver cannot represent the steepening fronts; its error exceeds 65% after 10 steps and it then becomes unstable.
- A small learned correction, applied after every coarse step, keeps the hybrid stable and within about 6% of the 512-point reference after 60 steps: the network fixes what the coarse grid misses while the solver does the transport.
- Training through 4 unrolled solver steps lowers the error at every horizon, most at 30 steps (about 25%), because the correction has learned from the states it produces itself. Longer unrolls are not automatically better: in preparing this example, an 8-step unroll without gradient clipping trained less well in the same number of iterations, a typical optimization difficulty of long unrolls.

## In an interview

**Thirty-second answer to "How can ML correct a coarse-grid solver instead of replacing it?"** "Run the cheap, under-resolved simulation and let a network add a correction each step, or replace one term such as the flux interpolation or a closure. The solver keeps transport, boundary conditions and conservation; the network learns the systematic subgrid error, which is local and learnable. Kochkov and colleagues did this inside a differentiable finite-volume code for 2D turbulence and matched solvers with 8-10 times finer grids, a 40-80 times speed-up. Training through the solver over several steps, solver-in-the-loop, makes the correction robust to its own feedback."

**Likely follow-ups**

- *What is unrolled training?* Backpropagating a multi-step loss through consecutive predictions, so the model is penalized for error growth; with a differentiable solver in the unroll it is solver-in-the-loop.
- *What if the solver is not differentiable?* Pushforward-style training, noise injection, or training the correction with reinforcement-style or derivative-free methods; or write a differentiable version of the coarse solver only.
- *Why are hybrids easier to trust?* The solver structure is still there: conservation, boundary conditions, and the ability to verify against the solver itself.

## References

- [Solver-in-the-Loop: Learning from Differentiable Physics to Interact with Iterative PDE-Solvers (Um et al., NeurIPS 2020)](https://arxiv.org/abs/2007.00016)
- [Machine learning-accelerated computational fluid dynamics (Kochkov et al., PNAS 2021)](https://arxiv.org/abs/2102.01010)
- [Learning data-driven discretizations for partial differential equations (Bar-Sinai et al., PNAS 2019)](https://arxiv.org/abs/1808.04930)
- [Learned Coarse Models for Efficient Turbulence Simulation (Stachenfeld et al.)](https://arxiv.org/abs/2112.15275)
- [APEBench (Koehler et al., NeurIPS 2024)](https://arxiv.org/abs/2411.00180)
- [Physics-based Deep Learning (Thuerey et al., free online book)](https://physicsbaseddeeplearning.org/)
- [ETH Zürich AISE: Introduction to Hybrid Workflows Part 1](https://www.youtube.com/watch?v=fJbt6VKYycA) and [Part 2](https://www.youtube.com/watch?v=h8BH-6tjecc)
