# Autoregressive Rollouts

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A time-stepping surrogate learns one step, u_{t+1} = F(u_t), and predicts a long trajectory by feeding its own outputs back in. It is trained on **clean** states from the solver but, during a rollout, receives its **own slightly wrong** predictions. That distribution shift makes errors compound: rollouts drift, blur or blow up. The fixes all make training look more like the rollout: **noise injection** (corrupt training inputs so the model learns to correct small errors, as in GNS and MeshGraphNets), the **pushforward trick** (train on inputs the model itself produced, from Message Passing Neural PDE Solvers), **unrolled training** through several steps, often with a differentiable solver in the loop, and **time-bundling** (predict several steps per call so there are fewer calls). MSE-trained models also **lose small scales**, because low-amplitude high-frequency components hardly affect the loss; **PDE-Refiner** fixes this with a diffusion-inspired refinement. Evaluate rollouts by **error against time**, correlation time and energy spectra, never by one-step error alone; for chaotic flows, predict **statistics**, not trajectories, beyond a few Lyapunov times; and monitor energy, conservation and ensemble spread to detect a rollout that is about to go wrong.

**On this page**

- [Message Passing Neural PDE Solvers](#message-passing-neural-pde-solvers)
- [Why one-step models drift](#why-one-step-models-drift)
- [Noise injection](#noise-injection)
- [The pushforward trick](#the-pushforward-trick)
- [Losing small scales and PDE-Refiner](#losing-small-scales-and-pde-refiner)
- [Evaluating a rollout](#evaluating-a-rollout)
- [Time-bundling and choosing the time step](#time-bundling-and-choosing-the-time-step)
- [What a surrogate can predict for a chaotic flow](#what-a-surrogate-can-predict-for-a-chaotic-flow)
- [Detecting a rollout that is about to blow up](#detecting-a-rollout-that-is-about-to-blow-up)
- [Code: rollout with noise injection](#code-rollout-with-noise-injection)
- [In an interview](#in-an-interview)

## Message Passing Neural PDE Solvers

Brandstetter, Worrall and Welling (ICLR 2022) built a time-stepping solver in which **every component is neural message passing** on the discretization, and showed that it representationally contains classical schemes: finite differences, finite volumes and WENO schemes are special cases of the messages it can compute. Three contributions became standard:

1. **Architecture.** An encoder embeds each node's recent history of the solution plus PDE parameters (coefficients, boundary information), a message-passing processor updates the nodes, and a 1D convolutional decoder outputs several future time steps at once (see time-bundling below).
2. **Stability as domain adaptation.** The test-time input distribution (the model's own outputs) differs from the training distribution (solver outputs). They framed long-rollout stability through the **zero-stability** idea from numerical analysis, small perturbations of the input should not grow, and introduced the **pushforward trick** to train on the shifted distribution. See [The pushforward trick](#the-pushforward-trick).
3. **Temporal bundling.** Predict K steps per call to reduce the number of autoregressive calls, and so the opportunities for error to compound.

They demonstrated it on families of 1D equations with varying coefficients (including shock-forming Burgers'-type equations) and on 2D problems, with generalization across PDE parameters and resolutions.

## Why one-step models drift

1. **Distribution shift (exposure bias).** Training inputs are exact solver states; rollout inputs carry accumulated model errors. The model has never seen inputs like that, so it can respond arbitrarily, and its next error is larger.
2. **Compounding.** If each step adds error ε and the dynamics amplify perturbations by a factor g per step, the error after n steps grows like ε(1 + g + g² + …). Even g slightly above 1 turns tiny one-step errors into large ones.
3. **Unphysical errors.** A solver's errors are structured (they respect conservation, smoothness and stability of the scheme); a network's errors are not, so they can inject energy, violate conservation or create noise at the grid scale that the dynamics then amplify.
4. **Spectral bias.** MSE training under-fits low-amplitude, high-frequency content; repeated application smooths the solution or lets high-frequency errors grow unchecked. See [Losing small scales and PDE-Refiner](#losing-small-scales-and-pde-refiner).
5. **Learned dynamics that are not dissipative where they should be,** or are too dissipative (spurious damping), so energy drifts over long horizons. See [Hamiltonian neural networks](hamiltonian-and-lagrangian-neural-networks.md) for an example.

The [code below](#code-rollout-with-noise-injection) shows a model whose one-step error is 0.4% reach 4.8% after 100 steps, while a model with a slightly *worse* one-step error but trained with noise reaches only 2%: one-step error is a poor predictor of rollout quality.

## Noise injection

**Idea.** During training, add noise to the input state, and train the model to predict the true next state (or the increment that reaches it) from the noisy input. The model learns to **pull slightly wrong states back** toward the data manifold instead of propagating their errors. It is almost free: one line of code.

```math
\tilde u_t = u_t + \epsilon,\ \ \epsilon \sim \mathcal N(0, \sigma^2), \qquad \mathcal L = \lVert F_\theta(\tilde u_t) - u_{t+1} \rVert^2
```

**Where it is used.**

- **GNS** adds **random-walk noise** to the input velocity history of each particle and adjusts the target acceleration accordingly; the paper found noise and the number of message-passing steps to be the main determinants of long-term accuracy. See [Training with noise](graph-network-simulator-gns.md#training-with-noise).
- **MeshGraphNets** adds noise to the dynamical input variables during training for the same reason.

**Choosing σ.** About the size of the one-step error you expect during rollouts; tune it by measuring rollout error, not one-step error. Too little has no effect; too much blurs predictions (the model learns to average). Noise can be scaled per channel by the channel's standard deviation, and its spatial structure matters (white noise at the grid scale is different from smooth noise).

**Why the target matters.** If the model predicts an increment, compute the target increment from the *noisy* input (u_{t+1} − ũ_t), so the model learns to correct the noise. If the target is computed from the clean input, the model learns to ignore noise, which is less useful.

## The pushforward trick

**Idea (Brandstetter et al., 2022).** Instead of adding artificial noise, use the **model's own errors** as the perturbation:

1. Start from a true state u_t.
2. Apply the model once **without gradients**: û_{t+1} = F_θ(u_t), detached.
3. Apply the model again **with gradients** to predict u_{t+2} from û_{t+1}, and compute the loss against the true u_{t+2}.

The gradient flows only through the last step, so the cost is one extra forward pass, not backpropagation through the whole unroll. The model learns to handle the input distribution it actually produces, the "pushed-forward" distribution, which is the domain-adaptation view. Variants push forward several steps without gradients, or mix pushforward samples with ordinary one-step samples.

**Compared with full unrolled training.** Backpropagating through K steps (backpropagation through time) is the most faithful but costs K times the memory and can have exploding or vanishing gradients; the pushforward trick is a cheap approximation that ignores how earlier steps' parameters influence later inputs. See [Unrolled training and solver-in-the-loop](differentiable-physics-and-hybrid-solvers.md#unrolled-training-and-solver-in-the-loop).

## Losing small scales and PDE-Refiner

**The problem.** Physical fields have energy spectra that fall steeply with wavenumber: most of the variance is in large scales. A model trained with MSE gets most of its loss reduction from the large scales and largely ignores the small-amplitude small scales. Lippe et al. (PDE-Refiner, NeurIPS 2023) identified this **neglect of non-dominant spatial frequencies** (often the high frequencies) as the primary pitfall limiting long rollouts: errors start in the poorly modelled small scales and propagate to the large ones through nonlinear interactions (the energy cascade runs both ways).

**PDE-Refiner.**

1. Predict the next state as usual (a one-step neural solver).
2. Refine it in a few steps inspired by diffusion models: add noise of decreasing magnitude to the prediction and let the network denoise it, conditioned on the previous state. Each refinement step focuses on progressively lower-amplitude components, so all frequencies, not just the dominant ones, are modelled accurately.
3. The denoising objective acts as a **spectral data augmentation**, which also makes training data-efficient.

**Results reported.** Stable, accurate rollouts over longer horizons than baselines on challenging benchmarks including the Kuramoto-Sivashinsky equation and 2D turbulent flows, and an **uncertainty estimate** (from the spread of refinement samples) that indicates when the surrogate becomes inaccurate.

**Other remedies.** Spectral or frequency-weighted losses, Sobolev norms, multi-scale architectures, and generative (diffusion) forecasters that sample realistic small scales. See [Probabilistic forecasting with diffusion](generative-models-for-physics.md#probabilistic-forecasting-with-diffusion).

## Evaluating a rollout

1. **Error against time.** Relative L2 error at each step, averaged over many test trajectories (with a spread), plotted against time. Report several horizons, never only one step.
2. **Correlation time.** The time until the Pearson correlation between predicted and true fields drops below a threshold (0.8 or 0.9 are common); useful for chaotic systems where the error eventually saturates for any model. PDE-Refiner, for example, compares models by how long they stay highly correlated with the reference.
3. **Energy spectra.** Compare the predicted and true kinetic-energy spectra E(k) at several times; a high-wavenumber deficit means over-smoothing, an excess means noise or instability.
4. **Conserved and physical quantities.** Mass, momentum, energy (or its expected decay), positivity of density or concentration, divergence of velocity, boundary conditions. See [Physics-consistency checks](evaluation-metrics.md#physics-consistency-checks).
5. **Statistics for chaotic flows.** Means, RMS fluctuations, spectra, probability distributions and two-point correlations, compared over long windows. See [What a surrogate can predict for a chaotic flow](#what-a-surrogate-can-predict-for-a-chaotic-flow).
6. **Stability.** Does it blow up, and when? Run much longer than the training horizon.
7. **Engineering quantities over time.** Lift and drag histories, peak temperature, Strouhal number of shedding.
8. **Cost.** Wall-clock per simulated time unit against the solver at matched accuracy. See [Reporting a speed-up fairly](evaluation-metrics.md#reporting-a-speed-up-fairly).

## Time-bundling and choosing the time step

**Time-bundling.** Predict K future steps (u_{t+1}, …, u_{t+K}) from the current window in one call. With the same horizon, the number of autoregressive calls drops by a factor of K, so there are fewer chances for errors to compound and inference is faster. MP-PDE introduced it for neural PDE solvers, decoding the bundle with a 1D convolution over time. Trade-off: the output head is bigger and errors within a bundle grow with the lead time; K is tuned on validation rollouts.

**Choosing the model's time step.** The surrogate is not bound by the solver's stability limit (CFL), so it can step much further. Considerations:

- **Larger steps** mean fewer calls (less compounding, faster) but a harder, more nonlinear map to learn per step, and less temporal detail.
- **Smaller steps** make each map close to the identity (easy to learn as an increment) but need many more calls.
- **Data:** the training frames fix the possible steps; a common choice is a multiple of the solver step at which the change per step is clearly larger than the noise floor.
- **Physics:** resolve the fastest phenomenon you need to predict (vortex shedding period, wave period) with several steps per cycle.
- **Measure:** train at 2-3 candidate steps and compare rollout error at a fixed physical time.

**Alternatives that avoid fixed steps:** time-conditioned models that predict u(t + Δt) for any Δt (Poseidon conditions on time; see [Foundation models for physics](foundation-models-for-physics.md)); neural ODEs that integrate a learned vector field with adaptive steps (see [Neural ODEs](neural-odes.md)); and **one-shot** models that map conditions directly to the state at the time of interest, which a 2026 crash study found more accurate than autoregressive rollouts. See [Crash simulation surrogates](crash-simulation-surrogates.md).

## What a surrogate can predict for a chaotic flow

**Chaos limits any predictor.** In a chaotic system (turbulence, convection, weather), nearby states separate exponentially, at a rate set by the largest Lyapunov exponent λ. An initial error δ₀ grows to the size of the signal after a time of about (1/λ) ln(size/δ₀). Every model, including the solver itself with slightly different round-off, has this **predictability horizon**. The [code below](#code-rollout-with-noise-injection) shows two Lorenz trajectories starting 10⁻⁸ apart becoming completely different after t ≈ 18, about 16 Lyapunov times, while their statistics agree.

**What a surrogate can realistically deliver.**

1. **Short-term trajectories** up to a few Lyapunov times: the correct evolution of the large scales.
2. **Statistics beyond the horizon:** means, variances, spectra, probability distributions, extreme-event frequencies, the attractor's shape. For engineering, these are usually what matter: mean drag, RMS lift, peak loads, mean heat transfer.
3. **Distributions of possible futures** from probabilistic models (ensembles, diffusion models such as GenCast in weather), rather than one deterministic trajectory that becomes wrong. See [Probabilistic forecasting with diffusion](generative-models-for-physics.md#probabilistic-forecasting-with-diffusion).
4. **Time-averaged fields directly:** many industrial surrogates predict time-averaged LES or hybrid RANS-LES fields (DrivAerML provides them), avoiding rollouts entirely.

**How to evaluate:** pointwise error is meaningless after the horizon (two correct simulations also disagree), so compare statistics and spectra, and check that the surrogate's long rollouts stay on the attractor (do not blow up or collapse to a fixed point).

## Detecting a rollout that is about to blow up

1. **Physics monitors.** Track total energy, mass and momentum each step; flag growth where physics says decay (viscous flow without forcing), violations of conservation beyond a tolerance, negative densities or temperatures below absolute zero, divergence of an incompressible field.
2. **Range checks.** Values outside the training range (velocity above twice the inlet speed, for example) or norms growing faster than physically possible.
3. **Spectral monitors.** Growth of energy at the highest wavenumbers (grid-scale noise) is an early sign of instability, often well before the large scales go wrong.
4. **Uncertainty.** Ensemble spread, MC-dropout variance or PDE-Refiner's sample spread rising quickly; disagreement between models trained with different seeds.
5. **Input out-of-distribution scores.** Distance of the current state's latent representation from the training states. See [Detecting an out-of-distribution geometry](out-of-distribution-and-generalization.md#detecting-an-out-of-distribution-geometry).
6. **Residual checks.** Plug the prediction into the discretized equations and monitor the PDE residual.

**On a trigger:** stop and hand over to the numerical solver from the last trusted state (a hybrid "surrogate with fallback"), re-initialize from the solver periodically, or flag the result as untrusted. See [When to fall back to the full solver](out-of-distribution-and-generalization.md#when-to-fall-back-to-the-full-solver).

## Code: rollout with noise injection

The test bed is viscous Burgers' equation, u_t + u u_x = ν u_xx with ν = 0.01, on a periodic grid of 128 points, solved pseudo-spectrally; the surrogate steps 20 solver steps at a time. The same small convolutional network is trained three ways on 40 trajectories of 40 steps: on clean pairs, with input noise, and with the pushforward trick. Each is then rolled out for 100 steps on 10 new trajectories, 2.5 times longer than any training trajectory. A physics monitor counts steps where the energy ∫u² rises, which viscous Burgers' never does. The second block shows the predictability horizon of the chaotic Lorenz system. Training takes about two minutes on a CPU.

```python
import numpy as np
import torch, torch.nn as nn

torch.manual_seed(0)
N, nu, dt_solver, sub = 128, 0.01, 1e-3, 20          # grid, viscosity, solver step, solver steps per model step
k = np.fft.rfftfreq(N, d=1.0 / N) * 2 * np.pi        # wavenumbers on the periodic domain [0, 1)

def burgers_step(u, n):
    """Pseudo-spectral viscous Burgers u_t + u u_x = nu u_xx with an explicit two-stage Runge-Kutta scheme."""
    def rhs(v):
        vh = np.fft.rfft(v)
        return -v * np.fft.irfft(1j * k * vh, n=N) + nu * np.fft.irfft(-(k ** 2) * vh, n=N)
    for _ in range(n):
        u1 = u + dt_solver * rhs(u)
        u = 0.5 * (u + u1 + dt_solver * rhs(u1))
    return u

def trajectory(rng, steps):
    x = np.arange(N) / N
    u = sum(rng.normal() / m * np.sin(2 * np.pi * m * x + rng.uniform(0, 2 * np.pi)) for m in range(1, 6))
    out = [u / np.abs(u).max()]                      # random smooth initial condition, max |u| = 1
    for _ in range(steps):
        out.append(burgers_step(out[-1], sub))
    return np.stack(out)

rng = np.random.default_rng(0)
train = torch.tensor(np.stack([trajectory(rng, 40) for _ in range(40)]), dtype=torch.float32)  # 40 x 41 frames
test = torch.tensor(np.stack([trajectory(rng, 100) for _ in range(10)]), dtype=torch.float32)  # 10 x 101 frames

def make_model():
    conv = lambda i, o: nn.Conv1d(i, o, 5, padding=2, padding_mode="circular")
    return nn.Sequential(conv(1, 32), nn.GELU(), conv(32, 32), nn.GELU(), conv(32, 1))
step = lambda model, u: u + model(u[:, None])[:, 0]  # the network predicts the increment

def fit(mode, iters=3000, noise_std=0.01):
    torch.manual_seed(1)
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    X, Y, Y2 = (train[:, s:train.shape[1] - 2 + s].reshape(-1, N) for s in range(3))   # (u_t, u_t+1, u_t+2)
    for it in range(iters):
        idx = torch.randint(0, len(X), (64,))
        x, y = X[idx], Y[idx]
        if mode == "noise":
            x = x + noise_std * torch.randn_like(x)      # corrupt the input; the target stays the true next state
        elif mode == "pushforward" and it % 2 == 1:      # every other step: start from the model's own prediction
            with torch.no_grad():
                x = step(model, x)                       # no gradient through the first step
            y = Y2[idx]
        loss = ((step(model, x) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model

@torch.no_grad()
def rollout(model, u0, n_steps):
    us = [u0]
    for _ in range(n_steps):
        us.append(step(model, us[-1]))
    return torch.stack(us, 1)

results = {}
for mode in ["clean", "noise", "pushforward"]:
    pred = rollout(fit(mode), test[:, 0], 100)
    err = ((pred - test).norm(dim=2) / test.norm(dim=2)).mean(0)          # relative L2 error at each step
    energy = (pred ** 2).sum(2)
    rises = (energy[:, 1:] > energy[:, :-1] * (1 + 1e-4)).float().mean().item()   # viscous Burgers: never rises
    results[mode] = err
    print(f"{mode:>12}: error after 1 step {err[1]:.4f}, 25 steps {err[25]:.4f}, 50 steps {err[50]:.4f}, "
          f"100 steps {err[100]:.4f} | steps with rising energy {100 * rises:.1f}%")
```

Output:

```
       clean: error after 1 step 0.0039, 25 steps 0.0095, 50 steps 0.0146, 100 steps 0.0478 | steps with rising energy 0.0%
       noise: error after 1 step 0.0046, 25 steps 0.0088, 50 steps 0.0096, 100 steps 0.0198 | steps with rising energy 0.0%
 pushforward: error after 1 step 0.0049, 25 steps 0.0099, 50 steps 0.0161, 100 steps 0.0409 | steps with rising energy 0.0%
```

```python
import numpy as np
from scipy.integrate import solve_ivp

def lorenz(t, s, sigma=10.0, rho=28.0, beta=8 / 3):
    x, y, z = s
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

t = np.linspace(0, 100, 20001)
a = solve_ivp(lorenz, (0, 100), [1.0, 1.0, 20.0], t_eval=t, rtol=1e-11, atol=1e-11).y
b = solve_ivp(lorenz, (0, 100), [1.0 + 1e-8, 1.0, 20.0], t_eval=t, rtol=1e-11, atol=1e-11).y   # tiny perturbation
sep = np.linalg.norm(a - b, axis=0)
print(f"initial difference 1e-8: trajectories differ by more than 1 after t = {t[np.argmax(sep > 1.0)]:.1f} "
      f"(Lyapunov time about 1.1)")
late = t > 50
print(f"pointwise error of z over 50 < t < 100: {np.abs(a[2, late] - b[2, late]).mean():.1f} (z has std {a[2, late].std():.1f})")
print(f"statistics over 50 < t < 100: mean z {a[2, late].mean():.2f} vs {b[2, late].mean():.2f}, "
      f"std z {a[2, late].std():.2f} vs {b[2, late].std():.2f}")
```

Output:

```
initial difference 1e-8: trajectories differ by more than 1 after t = 17.8 (Lyapunov time about 1.1)
pointwise error of z over 50 < t < 100: 9.7 (z has std 8.8)
statistics over 50 < t < 100: mean z 23.42 vs 23.52, std z 8.77 vs 8.67
```

**What it shows.**

- The clean-trained model has the best one-step error (0.39%) but the worst long rollout: its error grows twelvefold to 4.8% after 100 steps.
- Noise injection makes the one-step error slightly worse (0.46%) and the 100-step error 2.4 times better (2.0%), because the model has learned to correct perturbed states. Ranking models by one-step error would have picked the wrong one.
- The pushforward variant helps a little here (4.1%). Its benefit depends on how different the model's own errors are from the clean data; with longer pushforward chains or stronger dynamics the gap grows.
- The energy monitor stayed silent for all three models: none injected energy into a viscous flow within 100 steps. In an unstable rollout this is usually the first alarm, before the error against the reference (which you do not have in production) becomes large.
- For Lorenz, a perturbation of 10⁻⁸ grows to order one by t ≈ 18, after which the pointwise error is as large as the signal itself, yet the two trajectories' mean and standard deviation of z agree to about 1%. Beyond the horizon, statistics are the meaningful target.

## In an interview

**Thirty-second answer to "Why does a model trained for one-step prediction drift when you roll it out?"** "Because it is trained on clean solver states but rolled out on its own predictions, which carry small errors it has never seen; its next error is larger and they compound. Network errors are also unstructured, so they can inject energy or grid-scale noise that the dynamics amplify, and MSE training neglects small-amplitude small scales. The fixes make training look like the rollout: add noise to the inputs, train on the model's own outputs with the pushforward trick, unroll several steps with a differentiable solver, and predict several steps per call. And I always evaluate the error against time, not the one-step error."

**Likely follow-ups**

- *Why does noise injection help?* The model learns to map slightly wrong states back toward the true trajectory instead of propagating the error.
- *Noise injection or pushforward?* Noise is almost free and generic; pushforward uses the model's actual error distribution for one extra forward pass; unrolled training is the most faithful and most expensive.
- *What can you predict for turbulence?* Trajectories for a few Lyapunov times, then statistics: means, RMS, spectra, extreme-event rates.

## References

- [Message Passing Neural PDE Solvers (Brandstetter, Worrall & Welling, ICLR 2022)](https://arxiv.org/abs/2202.03376)
- [Learning to Simulate Complex Physics with Graph Networks (GNS)](https://arxiv.org/abs/2002.09405) and [MeshGraphNets](https://arxiv.org/abs/2010.03409)
- [PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers (Lippe et al., NeurIPS 2023)](https://arxiv.org/abs/2308.05732)
- [Solver-in-the-Loop (Um et al., NeurIPS 2020)](https://arxiv.org/abs/2007.00016)
- [Towards Multi-spatiotemporal-scale Generalized PDE Modeling (PDEArena; Gupta & Brandstetter)](https://arxiv.org/abs/2209.15616) and [code](https://github.com/pdearena/pdearena)
- [APEBench (Koehler et al., NeurIPS 2024)](https://arxiv.org/abs/2411.00180)
- [ETH Zürich AISE: Time-Dependent Neural Operators (video)](https://www.youtube.com/watch?v=u1KFcAvjyCI)
