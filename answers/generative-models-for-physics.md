# Generative Models for Physics

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A generative model learns a **probability distribution** and can draw new samples from it, instead of predicting one answer. The two dominant families for physics are **denoising diffusion models**, which learn to reverse a gradual noising process step by step, and **flow matching**, which learns a velocity field that carries simple noise to data along chosen (often straight) paths and samples by integrating an ODE in fewer steps. Conditioned on targets, geometry, boundary conditions or a previous state, they are used for **inverse design** (sample many shapes that meet a performance target, then verify them with a solver), **probabilistic forecasting** of chaotic systems (ensembles of plausible futures, as in GenCast for weather, which beat the leading operational ensemble on 97.4% of 1,320 targets), **sharper predictions** that keep the small scales MSE regression blurs away, and **solving PDEs from partial observations** (DiffusionPDE learns the joint distribution of coefficients and solutions and fills in what was not measured). Their costs are many network evaluations per sample, and the need to evaluate them on distributions and physical validity rather than on the error of one sample.

**On this page**

- [Why a distribution instead of an answer](#why-a-distribution-instead-of-an-answer)
- [Denoising diffusion models](#denoising-diffusion-models)
- [Flow matching](#flow-matching)
- [Building an inverse-design tool](#building-an-inverse-design-tool)
- [Probabilistic forecasting with diffusion](#probabilistic-forecasting-with-diffusion)
- [Solving PDEs from partial observations](#solving-pdes-from-partial-observations)
- [Costs and evaluation](#costs-and-evaluation)
- [Code: a diffusion model and conditional flow matching for inverse design](#code-a-diffusion-model-and-conditional-flow-matching-for-inverse-design)
- [In an interview](#in-an-interview)

## Why a distribution instead of an answer

A regression model trained with MSE predicts the **conditional mean** of the output given the input. That is the right answer when the output is determined by the input, and the wrong one when:

- **the problem has many solutions:** inverse design ("a wing with this lift and drag") has a whole family of valid shapes, and their average may not be a valid shape;
- **the outcome is uncertain:** a chaotic flow beyond its predictability horizon, or a field reconstructed from a few sensors, has many plausible states; their mean is a blurry, unphysical field;
- **small scales matter:** the mean over uncertain small-scale structure is smooth, so regression loses high-frequency content (see [Losing small scales and PDE-Refiner](autoregressive-rollouts.md#losing-small-scales-and-pde-refiner)).

A generative model draws individual, realistic samples and represents the spread between them.

## Denoising diffusion models

**Forward process.** Starting from a data sample x₀ (a field, a shape), add Gaussian noise in T small steps until only noise remains. With a variance schedule β_t and ᾱ_t = Π(1 − β_s), any step has a closed form:

```math
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon, \qquad \epsilon \sim \mathcal N(0, I)
```

**Training (DDPM, Ho, Jain and Abbeel, 2020).** A network ε_θ(x_t, t, c) learns to predict the noise ε that was added, with a simple MSE loss over random samples, time steps and noise draws. The condition c can be a target property, geometry, boundary conditions or the previous time step. Predicting the noise is equivalent to learning the **score** ∇ log p_t(x), the direction toward higher probability at each noise level (the score-based view of Song et al.).

**Sampling.** Start from pure noise and apply the learned denoising step T times (stochastic DDPM sampling or deterministic DDIM / ODE samplers with fewer steps).

**Conditioning.** Concatenate the condition to the input, inject it with cross-attention or FiLM, and strengthen it with **classifier-free guidance** (mix conditional and unconditional predictions). **Guidance with a surrogate or the physics:** add the gradient of a property predictor or a PDE residual during sampling to steer samples toward targets or physical consistency (used in TopoDiff and DiffusionPDE).

## Flow matching

**Idea (Lipman et al., ICLR 2023; related: rectified flow, stochastic interpolants).** Define a simple path between a noise sample x₀ and a data sample x₁, for example the straight line x_t = (1 − t)x₀ + t x₁, whose velocity is x₁ − x₀. Train a network v_θ(x_t, t, c) to regress that velocity:

```math
\mathcal L(\theta) = \mathbb E_{t, x_0, x_1}\, \big\lVert v_\theta(x_t, t, c) - (x_1 - x_0) \big\rVert^2
```

Sampling integrates dx/dt = v_θ(x, t, c) from noise at t = 0 to data at t = 1 with an ODE solver.

**Why it is popular.** Training is simulation-free regression like diffusion, but the paths are straighter, so sampling needs fewer steps (the code below uses 20 Euler steps); the framework includes diffusion paths as special cases; and it is the training method behind many recent generative models. For physics, the ODE view connects directly to [neural ODEs](neural-odes.md) and continuous normalizing flows.

## Building an inverse-design tool

**Goal:** given a target (a pressure distribution, a drag value, a stiffness and mass budget), propose designs that achieve it.

```
target (e.g. pressure distribution, drag) ──► conditional generator ──► many candidate shapes
          ──► validity checks ──► surrogate screening ──► solver verification ──► (feedback, retraining)
```

1. **Data:** designs paired with simulated performance, in a representation the generator can produce and a CAD tool can rebuild: design parameters, signed distance fields, point clouds or meshes, or CAD sequences. See [Geometry representations](geometry-representations.md).
2. **Generator:** a conditional diffusion or flow-matching model p(design | target), optionally with guidance from a differentiable performance surrogate and manufacturability classifiers (TopoDiff used surrogate guidance for compliance and manufacturability and reduced performance error eightfold compared with a conditional GAN). See [Machine learning for topology optimization](topology-optimization.md#machine-learning-for-topology-optimization).
3. **Sample many candidates:** generative models naturally return a **diverse set** of designs for one target, including different solution families (the code below finds both solutions of a two-solution target).
4. **Validity checks:** watertight, no self-intersections, within packaging limits, manufacturable.
5. **Screen with a forward surrogate,** then **verify the shortlist with the solver.** The generator's own belief about performance is not evidence.
6. **Close the loop:** verified designs become training data (active learning), and the target region gets denser data.
7. **Alternatives to compare against:** optimization on a forward surrogate (see [Surrogate-based optimization](surrogate-based-optimization.md)), Bayesian optimization on design parameters, and adjoint optimization. Generative models are most useful when the design space is high-dimensional and multimodal and many diverse candidates are wanted.

The system-design version of this question is [Design: an inverse-design tool](design-inverse-design-tool.md). Commercially, PhysicsX's Ai.rplane, built on its LGM-Aero model, is an example of generating aircraft designs and predicting their performance; see [Industry](../industry.md).

## Probabilistic forecasting with diffusion

**Weather: GenCast (Price et al., DeepMind, Nature 2024).** A conditional diffusion model that generates global weather forecasts as an **ensemble**: each sample is a plausible 15-day trajectory at 0.25° resolution with 12-hour steps for more than 80 variables, produced in about 8 minutes on one accelerator. Trained on decades of reanalysis data, it was more skilful than ECMWF's operational ensemble (ENS) on 97.4% of 1,320 evaluated targets, and better on extreme weather, tropical-cyclone tracks and wind-power prediction. See [Lessons from ML weather forecasting](foundation-models-for-physics.md#lessons-from-ml-weather-forecasting).

**Turbulent flows.** Kohl, Chen and Thuerey (2023) benchmarked autoregressive conditional diffusion models on 2D incompressible and transonic flows and isotropic turbulence: even simple diffusion-based solvers outperformed several established flow-prediction methods in accuracy and temporal stability, matched unrolled training, and provided posterior samples, at the cost of slower inference.

**Why diffusion helps forecasts.** Each step samples a sharp, realistic state instead of the blurry mean; the ensemble spread estimates the forecast uncertainty; and rollouts stay on the attractor longer. PDE-Refiner applies a diffusion-style refinement to deterministic forecasting for the same spectral reason.

## Solving PDEs from partial observations

**The setting.** In practice the inputs are incomplete: sparse sensors, a partially known coefficient field, a boundary condition measured at a few points. Classical solvers need the full input; regression models need a fixed input format.

**DiffusionPDE (Huang et al., NeurIPS 2024).** Learn the **joint distribution** of the coefficient (or initial condition) and the solution with a diffusion model; at sampling time, guide the samples to match the sparse observations and to satisfy the PDE (adding the gradients of an observation loss and a PDE-residual loss during denoising). The same model then solves forward problems (coefficients partially known, solution unknown), inverse problems (solution partially observed, coefficients unknown), and mixed cases, and it outperformed state-of-the-art methods in both directions in the paper's experiments.

**Related approaches:** score-based data assimilation for weather and fluids, generative priors in inverse problems (see [Machine learning for inverse problems](inverse-problems-and-regularization.md#machine-learning-for-inverse-problems)), and conditional neural fields that reconstruct fields from sparse sensors.

## Costs and evaluation

- **Inference cost:** tens to hundreds of network evaluations per sample (fewer with flow matching, DDIM, consistency models or distillation), times the number of samples in an ensemble. Budget it against the solver.
- **Training cost and data:** generative models typically need more data than deterministic regressors for the same accuracy of the mean.
- **Evaluation needs distributional metrics:** energy spectra, CRPS, ensemble spread against error (spread-skill), coverage of observed outcomes, distribution distances (Wasserstein, MMD), statistics of quantities of interest. One sample's MSE against one reference is meaningless for a stochastic model.
- **Physical validity:** generated fields should satisfy conservation and boundary conditions (check residuals); generated geometries must be valid and manufacturable. Verify with the solver.
- **Mode coverage:** check that all solution families or regimes are represented, not just the most common one.

## Code: a diffusion model and conditional flow matching for inverse design

A toy design space: valid "designs" lie on a ring (points with radius about 1), and their "performance" depends on the angle, cos θ, so some targets have two solution families. A DDPM learns to generate valid designs from noise; a conditional flow-matching model learns to generate designs with a requested performance, which are then "verified" with the true performance function. Training takes about 15 seconds.

```python
import math
import torch, torch.nn as nn

torch.manual_seed(0)
# "Designs": 2D points on a ring of radius 1 with small scatter; property(design) = angle-based performance.
def sample_designs(n):
    theta = torch.rand(n) * 2 * math.pi
    r = 1 + 0.05 * torch.randn(n)
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], 1)
def performance(x):                                               # a stand-in for the simulator: depends on the angle
    return torch.cos(torch.atan2(x[:, 1], x[:, 0]))               # 1 at angle 0, -1 at angle pi
on_manifold = lambda x: ((x.norm(dim=1) - 1).abs() < 0.15).float().mean().item()

def mlp(d_in, d_out):
    return nn.Sequential(nn.Linear(d_in, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, d_out))

# 1. Denoising diffusion (DDPM): learn to predict the noise added at each of T steps, then sample by reversing.
T = 100
betas = torch.linspace(1e-4, 0.05, T)
alphas = 1 - betas
abar = torch.cumprod(alphas, 0)
eps_net = mlp(3, 2)
opt = torch.optim.Adam(eps_net.parameters(), lr=2e-3)
for it in range(4000):
    x0 = sample_designs(256)
    t = torch.randint(0, T, (256,))
    eps = torch.randn_like(x0)
    xt = abar[t, None].sqrt() * x0 + (1 - abar[t, None]).sqrt() * eps     # forward noising in closed form
    loss = ((eps_net(torch.cat([xt, t[:, None] / T], 1)) - eps) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    x = torch.randn(2000, 2)
    for t in reversed(range(T)):                                   # reverse process: denoise step by step
        e = eps_net(torch.cat([x, torch.full((2000, 1), t / T)], 1))
        x = (x - betas[t] / (1 - abar[t]).sqrt() * e) / alphas[t].sqrt()
        if t > 0:
            x = x + betas[t].sqrt() * torch.randn_like(x)
print(f"DDPM ({T} denoising steps): {100 * on_manifold(x):.1f}% of samples on the ring; "
      f"Gaussian noise baseline {100 * on_manifold(torch.randn(2000, 2)):.1f}%")

# 2. Conditional flow matching: learn a velocity field that transports noise to designs, conditioned on a target performance.
v_net = mlp(4, 2)                                                  # inputs: x_t (2), t, target performance
opt = torch.optim.Adam(v_net.parameters(), lr=2e-3)
for it in range(4000):
    x1 = sample_designs(256)
    c = performance(x1)[:, None]
    x0 = torch.randn_like(x1)
    t = torch.rand(256, 1)
    xt = (1 - t) * x0 + t * x1                                     # straight path from noise to data
    loss = ((v_net(torch.cat([xt, t, c], 1)) - (x1 - x0)) ** 2).mean()   # regress the path's velocity
    opt.zero_grad(); loss.backward(); opt.step()
def generate(target, n=1000, steps=20):
    with torch.no_grad():
        x = torch.randn(n, 2)
        c = torch.full((n, 1), target)
        for i in range(steps):                                     # integrate dx/dt = v(x, t, c) with Euler steps
            t = torch.full((n, 1), i / steps)
            x = x + v_net(torch.cat([x, t, c], 1)) / steps
    return x
for target in [0.9, 0.0, -0.9]:
    x = generate(target)
    perf = performance(x)                                          # "verify with the simulator"
    print(f"flow matching, target performance {target:+.1f}: generated designs score {perf.mean():+.3f} +/- {perf.std():.3f}; "
          f"{100 * on_manifold(x):.1f}% valid (on the ring)")
x = generate(0.0)                                                 # performance 0 has two solutions: angles +90 and -90 degrees
print(f"target 0.0 has two solution families: {100 * (x[:, 1] > 0).float().mean():.0f}% of samples near +90 degrees, "
      f"{100 * (x[:, 1] < 0).float().mean():.0f}% near -90 degrees")
```

Output:

```
DDPM (100 denoising steps): 96.5% of samples on the ring; Gaussian noise baseline 17.7%
flow matching, target performance +0.9: generated designs score +0.927 +/- 0.035; 99.6% valid (on the ring)
flow matching, target performance +0.0: generated designs score +0.009 +/- 0.026; 95.9% valid (on the ring)
flow matching, target performance -0.9: generated designs score -0.903 +/- 0.045; 98.7% valid (on the ring)
target 0.0 has two solution families: 48% of samples near +90 degrees, 52% near -90 degrees
```

**What it shows.**

- Starting from Gaussian noise (only about 18% of whose points happen to lie near the ring), 100 learned denoising steps put 96.5% of samples on the valid-design manifold.
- The conditional flow-matching model, with only 20 integration steps, generates designs whose verified performance matches the requested target closely (for example +0.927 ± 0.035 for a target of +0.9), and over 95% of them are valid.
- For a target of 0, which two different designs achieve (angles of +90° and −90°), the samples split between both solution families. A regression model asked to map the target to one design would average them to an invalid point inside the ring: that is the core reason to use a generative model for inverse design.

## In an interview

**Thirty-second answer to "What are diffusion models and flow matching, and why use them in physics?"** "Both learn a distribution and sample from it. A diffusion model learns to predict the noise added to data at each of many noise levels, then generates by starting from noise and denoising step by step. Flow matching learns a velocity field that transports noise to data along straight paths and samples by integrating an ODE in fewer steps. In physics they matter when one answer is not enough: inverse design with many valid shapes, probabilistic forecasts of chaotic flows like GenCast's weather ensembles, sharp fields that regression would blur, and reconstructing fields from sparse measurements like DiffusionPDE. The cost is many network calls per sample, and you evaluate them with distributional metrics and solver verification."

**Likely follow-ups**

- *Why not regression for inverse design?* The mean of several valid designs is usually not a valid design.
- *How would you condition on a target and enforce physics?* Conditional inputs with classifier-free guidance, plus gradient guidance from a performance surrogate or the PDE residual during sampling.
- *How do you evaluate a probabilistic forecast?* CRPS, spread-skill, spectra and calibration of event probabilities, not the MSE of one member.

## References

- [Denoising Diffusion Probabilistic Models (Ho, Jain & Abbeel)](https://arxiv.org/abs/2006.11239)
- [Score-Based Generative Modeling through Stochastic Differential Equations (Song et al.)](https://arxiv.org/abs/2011.13456)
- [Flow Matching for Generative Modeling (Lipman et al.)](https://arxiv.org/abs/2210.02747)
- [GenCast: Probabilistic weather forecasting with machine learning (Price et al.)](https://arxiv.org/abs/2312.15796)
- [Benchmarking Autoregressive Conditional Diffusion Models for Turbulent Flow Simulation (Kohl, Chen & Thuerey)](https://arxiv.org/abs/2309.01745)
- [DiffusionPDE: Generative PDE-Solving Under Partial Observation (Huang et al.)](https://arxiv.org/abs/2406.17763)
- [Diffusion Models Beat GANs on Topology Optimization (TopoDiff)](https://arxiv.org/abs/2208.09591)
- [ETH Zürich AISE: Introduction to Diffusion Models (video)](https://www.youtube.com/watch?v=Tohlijxz3XQ)
