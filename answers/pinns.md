# Physics-Informed Neural Networks (PINNs)

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A physics-informed neural network represents the unknown solution u(x, t) of a PDE by a neural network u_θ(x, t) whose inputs are the coordinates. It is trained to minimize a loss made of the PDE residual at sampled **collocation points**, the misfit of the boundary and initial conditions, and, when measurements exist, the misfit to data. The derivatives in the residual are computed exactly with **automatic differentiation** of the network with respect to its inputs. PINNs need no mesh, blend data with physics naturally and are strongest on **inverse problems**: unknown coefficients become trainable parameters, and the PDE regularizes sparse, noisy data. For standard forward problems they are usually slower and less accurate than FEM or FVM, and a plain PINN represents **one** solution, so every new geometry, boundary condition or coefficient means training again; parametric PINNs and physics-informed neural operators address that.

**On this page**

- [The idea in one example](#the-idea-in-one-example)
- [The PINN loss function](#the-pinn-loss-function)
- [Derivatives in the PDE residual](#derivatives-in-the-pde-residual)
- [Collocation points](#collocation-points)
- [Inverse problems with PINNs](#inverse-problems-with-pinns)
- [When a PINN beats a classical solver and when it loses](#when-a-pinn-beats-a-classical-solver-and-when-it-loses)
- [Why a PINN is retrained for every new problem](#why-a-pinn-is-retrained-for-every-new-problem)
- [Code: a PINN for the 1D Poisson equation](#code-a-pinn-for-the-1d-poisson-equation)
- [Code: recovering an unknown coefficient from ten sensors](#code-recovering-an-unknown-coefficient-from-ten-sensors)
- [In an interview](#in-an-interview)

## The idea in one example

Solve −u″(x) = π² sin(πx) on (0, 1) with u(0) = u(1) = 0. The exact answer is u = sin(πx).

A classical solver would build a mesh and solve a linear system for nodal values. A PINN instead:

1. takes a small network u_θ(x) with input x and output u;
2. picks random points xᵢ in (0, 1);
3. computes u_θ″(xᵢ) by differentiating the network twice with respect to its input;
4. penalizes the residual −u_θ″(xᵢ) − π² sin(πxᵢ), plus u_θ(0)² and u_θ(1)²;
5. adjusts θ by gradient descent until all of these are small.

The trained network *is* the solution: a smooth function that can be evaluated and differentiated anywhere. The code below does exactly this and reaches a maximum error of a few times 10⁻³.

## The PINN loss function

For a PDE 𝒩[u] = 0 in a domain Ω, with boundary conditions ℬ[u] = g on ∂Ω, initial condition u(x, 0) = u₀(x), and optional measurements u_k at points x_k:

```math
\mathcal{L}(\theta) =
\underbrace{\frac{\lambda_r}{N_r}\sum_{i=1}^{N_r} \big|\mathcal{N}[u_\theta](x_i, t_i)\big|^2}_{\text{PDE residual at collocation points}}
+ \underbrace{\frac{\lambda_b}{N_b}\sum_{j=1}^{N_b} \big|\mathcal{B}[u_\theta](x_j, t_j) - g_j\big|^2}_{\text{boundary conditions}}
+ \underbrace{\frac{\lambda_0}{N_0}\sum_{l=1}^{N_0} \big|u_\theta(x_l, 0) - u_0(x_l)\big|^2}_{\text{initial condition}}
+ \underbrace{\frac{\lambda_d}{N_d}\sum_{k=1}^{N_d} \big|u_\theta(x_k, t_k) - u_k\big|^2}_{\text{data (if any)}}
```

**Example: the 1D heat equation** u_t = α u_xx on 0 < x < 1, t > 0. The residual is r = u_θ,t − α u_θ,xx, evaluated at random space-time points; the boundary term penalizes u_θ(0, t) and u_θ(1, t) against their prescribed values; the initial term compares u_θ(x, 0) with the initial temperature profile.

**Points about the loss:**

- The PDE is used in its **strong form** (pointwise residual), unlike FEM's weak form. See [The weak form](weak-form-and-galerkin.md#the-weak-form).
- Boundary and initial conditions are **soft constraints** unless built into the network. See [Soft vs hard boundary conditions in PINNs](hard-vs-soft-constraints.md#soft-vs-hard-boundary-conditions-in-pinns).
- The weights λ balance terms of very different sizes and convergence speeds; choosing them badly is the most common failure. See [Loss imbalance and adaptive weights](pinn-training-failures.md#loss-imbalance-and-adaptive-weights).
- **Non-dimensionalize** the PDE first, so that all terms are of order one. See [Why non-dimensionalize](nondimensionalization-and-normalization.md#why-non-dimensionalize).
- A small loss does not guarantee a small error: the residual can be small while the solution is wrong (for example, if a boundary condition is violated where few points were sampled). Always compare against a reference on a test case.

## Derivatives in the PDE residual

The residual needs derivatives of the network **output with respect to its inputs** (∂u/∂x, ∂²u/∂x², ∂u/∂t), not with respect to its weights. They are computed with automatic differentiation:

```py
x.requires_grad_(True)
u = model(x)
du = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]      # du/dx
d2u = torch.autograd.grad(du, x, torch.ones_like(du), create_graph=True)[0]   # d2u/dx2
```

- **create_graph=True** records the derivative computation itself, so it can be differentiated again (for the second derivative) and, later, with respect to the weights in `loss.backward()`. Training a PINN therefore differentiates through derivatives.
- **These derivatives are exact** for the network (up to floating point), not finite-difference approximations. See [Automatic differentiation](automatic-differentiation.md).
- **Cost grows with derivative order and dimension:** each derivative adds a pass through a larger graph; a Laplacian in 3D needs three second derivatives. Fourth-order problems (plates) are expensive. Forward-mode (Jacobian-vector products) and Taylor-mode methods can be more efficient for derivatives with respect to a few inputs.
- **Activations must be smooth enough:** ReLU has zero second derivative almost everywhere, so −u″ would be identically zero. Use tanh, sine (SIREN), GELU or Swish-like activations.
- **Alternatives** in some variants: finite differences on a grid (for physics-informed CNNs), or weak-form residuals with test functions that need fewer derivatives (variational PINNs, the Deep Ritz method).

## Collocation points

Collocation points are where the PDE residual is enforced. They play the role of a mesh, but they are just points.

**How to choose them:**

- **Uniform random** sampling, **Latin hypercube** or **Sobol** sequences over the domain (space and time). Low-discrepancy sequences cover the domain more evenly than pure random sampling.
- **Enough of them:** too few and the network can satisfy the residual at the points while violating it between them (the PINN analogue of overfitting). Typical problems use thousands to hundreds of thousands.
- **Resample during training** (new random points every step or every few hundred steps): this acts like data augmentation and reduces the chance of gaps. Note that L-BFGS needs a fixed set. See [Adam then L-BFGS](optimizers-for-physics-models.md#adam-then-l-bfgs).
- **Boundary and initial points** need their own sampling, and often relatively many of them, because boundary information must propagate into the domain.
- **Residual-based adaptive sampling:** periodically evaluate the residual on a large candidate set and add (or sample with higher probability) points where it is large. This concentrates effort on sharp fronts, boundary layers and shocks. Wu et al. (2023) compared non-adaptive and residual-based adaptive strategies systematically and found the adaptive ones clearly better on harder problems.
- **For time-dependent problems,** respecting causality (sampling or weighting early times before late ones) matters. See [Causality in time-dependent PINNs](pinn-training-failures.md#causality-in-time-dependent-pinns).

## Inverse problems with PINNs

In an **inverse problem**, part of the physics is unknown (a conductivity, a viscosity, a source term, a boundary condition), and there are measurements of the solution. PINNs handle this naturally:

1. Make the unknowns **trainable parameters** (a scalar, or a second network for a spatially varying field).
2. Add a **data term** comparing the network with the measurements.
3. Train everything together: the PDE residual ties the unknowns to the solution, and the data term ties the solution to reality.

The physics acts as a **regularizer** for sparse, noisy data: the network cannot just interpolate the measurements, it must also satisfy the PDE everywhere. In the code below, ten noisy sensors are enough to recover an unknown conductivity to within a fraction of a percent.

**What to watch:**

- **Identifiability:** can the data determine the unknown at all? A constant source and a constant conductivity, for example, can trade off against each other. Check with synthetic data first.
- **Positivity and bounds:** parametrize physical parameters so they stay valid (for example κ = exp(s)).
- **Noise and weighting:** the relative weight of data and physics encodes how much you trust each.
- **Field-valued unknowns** (a conductivity map) make the problem ill-posed and need regularization (smoothness priors, total variation). See [Inverse problems and regularization](inverse-problems-and-regularization.md).

Typical applications: identifying material properties from displacement measurements, estimating heat-transfer coefficients from thermocouples, reconstructing flow fields from sparse velocity or pressure measurements, data assimilation.

## When a PINN beats a classical solver and when it loses

| PINNs tend to win | PINNs tend to lose |
| --- | --- |
| Inverse problems and data assimilation, where the alternative needs an adjoint solver and regularization machinery | Standard forward problems in 2D and 3D that FEM or FVM solve well: classical solvers are typically much faster and more accurate, with error control |
| Problems mixing sparse, scattered measurements with partial physics | Many-query design work: a plain PINN is retrained for each new instance |
| High-dimensional PDEs (tens to hundreds of dimensions), where meshes are impossible | High-frequency, multi-scale, convection-dominated, shock-dominated or chaotic solutions |
| When a smooth, differentiable, mesh-free representation of the solution is itself useful | When guaranteed accuracy and convergence under refinement are required |
| Quick prototypes for unusual equations without writing a solver | Long-time integration of time-dependent problems |

**Evidence:** Grossmann et al. (2023) compared PINNs with FEM on Poisson problems in 1D to 3D, Allen-Cahn and semilinear Schrödinger equations, and found that in solution time and accuracy PINNs did not beat FEM in their study, while being fast to evaluate once trained. The broader review by McGreivy and Hakim (2024) found weak baselines in most ML-for-PDE papers that claimed to beat classical methods. See [Weak baselines and reporting biases](ml-vs-numerical-solvers.md#weak-baselines-and-reporting-biases).

**A good interview answer** is not "PINNs are bad", but "PINNs are a tool for a specific job": inverse problems, data assimilation, problems with partial physics, and high-dimensional settings.

## Why a PINN is retrained for every new problem

A PINN's weights encode **one solution** u(x) for one fixed set of coefficients, boundary conditions, sources and geometry. Change any of them and the solution changes, so the network must be trained again. Training takes minutes to hours, so a plain PINN gives no speed-up for design exploration, where thousands of variants are evaluated.

**What to do instead:**

- **Warm starts:** initialize from a PINN trained on a similar problem; faster, but still a training run per instance.
- **Parametric PINNs:** add the parameters (a Reynolds number, a geometric dimension) as extra inputs, u_θ(x, t, μ), and train over ranges of μ. Works for a few parameters; struggles with high-dimensional parameter spaces or free-form geometry.
- **Neural operators** learn the map from problem inputs (coefficient fields, boundary conditions, geometry) to solutions, from many examples, so a new instance costs one forward pass. Physics-informed versions (PINO, physics-informed DeepONet) add the PDE residual to the training loss. See [Neural operators](neural-operators.md) and [Physics-informed neural operator](physics-informed-neural-operator.md).
- **Meta-learning and transfer learning** to speed up retraining for families of problems.

## Code: a PINN for the 1D Poisson equation

```python
import torch, torch.nn as nn

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(),
                      nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
x_b = torch.tensor([[0.0], [1.0]])                        # boundary points

for step in range(3000):
    x = torch.rand(128, 1, requires_grad=True)            # collocation points, resampled every step
    u = model(x)
    du = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    d2u = torch.autograd.grad(du, x, torch.ones_like(du), create_graph=True)[0]
    residual = -d2u - torch.pi**2 * torch.sin(torch.pi * x)
    loss = (residual**2).mean() + 100.0 * (model(x_b)**2).mean()   # PDE + boundary
    opt.zero_grad(); loss.backward(); opt.step()
# exact solution: sin(pi x); test runs gave maximum errors of a few times 1e-3
```

The weight of 100 on the boundary term is a hand-tuned loss balance. Try 1 and watch the boundary values drift.

## Code: recovering an unknown coefficient from ten sensors

Solve −κ u″ = π² sin(πx), u(0) = u(1) = 0, where the conductivity κ is unknown, using ten noisy measurements of u. The true κ is 0.5.

```python
import torch, torch.nn as nn

torch.manual_seed(0)
kappa_true = 0.5                                              # unknown conductivity to recover
x_data = torch.linspace(0.05, 0.95, 10).unsqueeze(1)          # 10 sparse sensors
u_data = torch.sin(torch.pi * x_data) / kappa_true            # exact solution of -kappa u'' = pi^2 sin(pi x)
u_data = u_data + 0.01 * torch.randn_like(u_data)             # measurement noise

net = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
log_kappa = torch.nn.Parameter(torch.tensor(0.0))             # trainable unknown, kept positive via exp
opt = torch.optim.Adam(list(net.parameters()) + [log_kappa], lr=1e-3)
x_b = torch.tensor([[0.0], [1.0]])

for step in range(8000):
    x = torch.rand(128, 1, requires_grad=True)                # collocation points
    u = net(x)
    du = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    d2u = torch.autograd.grad(du, x, torch.ones_like(du), create_graph=True)[0]
    residual = -torch.exp(log_kappa) * d2u - torch.pi**2 * torch.sin(torch.pi * x)
    loss = (residual**2).mean() + 10.0 * (net(x_b)**2).mean() + 10.0 * ((net(x_data) - u_data)**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

kappa_est = torch.exp(log_kappa).item()
print(f"recovered kappa = {kappa_est:.4f} (true {kappa_true}), error {100 * abs(kappa_est - kappa_true) / kappa_true:.1f}%")
```

Output:

```
recovered kappa = 0.5016 (true 0.5), error 0.3%
```

Ten noisy point measurements would not determine κ by themselves without a model; the PDE residual turns them into a well-posed estimation problem.

## In an interview

**Thirty-second answer to "What is a PINN? Write down its loss."** "A network u_θ(x, t) that takes coordinates and outputs the solution. Its loss is the mean squared PDE residual at collocation points, plus the misfit of boundary and initial conditions, plus a data misfit if there are measurements, each with a weight. The residual's derivatives come from automatic differentiation of the network with respect to its inputs. It is mesh-free and very good for inverse problems, where unknown coefficients become trainable; for routine forward problems FEM is usually faster and more accurate, and a plain PINN has to be retrained for every new problem."

**Likely follow-ups**

- *How are the derivatives computed?* Autograd with create_graph=True, nested for higher derivatives; smooth activations required.
- *How do you choose collocation points?* Low-discrepancy sampling, resampled during training, plus residual-based adaptive refinement where the residual is large.
- *When would you use a PINN in industry?* Inverse problems and data assimilation with sparse sensors and partly known physics; not as a replacement for a validated forward solver.

## References

- [Physics Informed Deep Learning (Part I)](https://arxiv.org/abs/1711.10561) and [(Part II)](https://arxiv.org/abs/1711.10566) (Raissi, Perdikaris & Karniadakis)
- [So, what is a physics-informed neural network? (Ben Moseley)](https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/)
- [ETH Zürich AISE: PINNs, Introduction (video)](https://www.youtube.com/watch?v=D-F7BYRhAkQ) and [Physics Informed Neural Networks (Steve Brunton, video)](https://www.youtube.com/watch?v=-zrY7P2dVC4)
- [DeepXDE: a deep learning library for solving differential equations (Lu et al.)](https://arxiv.org/abs/1907.04502) and [documentation](https://deepxde.readthedocs.io/)
- [A comprehensive study of non-adaptive and residual-based adaptive sampling for PINNs (Wu et al.)](https://arxiv.org/abs/2207.10289)
- [Can Physics-Informed Neural Networks beat the Finite Element Method? (Grossmann et al., 2023)](https://arxiv.org/abs/2302.04107)
- [Scientific Machine Learning through PINNs: Where we are and What's next (Cuomo et al.)](https://arxiv.org/abs/2201.05624)
