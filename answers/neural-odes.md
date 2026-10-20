# Neural ODEs

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A neural ordinary differential equation (Chen, Rubanova, Bettencourt and Duvenaud, NeurIPS 2018) uses a neural network f_θ as the **right-hand side of an ODE**, dy/dt = f_θ(y, t), and produces outputs by integrating it with a numerical ODE solver. Instead of learning a map from one time step to the next, it learns a **continuous-time vector field**, so it can be evaluated at any time, handles **irregularly sampled** data naturally, and can use adaptive, error-controlled solvers. Gradients come either by backpropagating through the solver's operations or by the **adjoint method**, which solves a second ODE backwards in time and needs memory independent of the number of solver steps. Neural ODEs are used for learning dynamics from time series, latent-dynamics models of reduced simulations, continuous normalizing flows, and **universal differential equations** where a network supplies only the unknown term of a known physical model. Limits: the solver makes training slow (especially when the learned dynamics become stiff), trajectories of an autonomous ODE cannot cross, and long-horizon accuracy still depends on the learned field being right.

**On this page**

- [From residual networks to ODEs](#from-residual-networks-to-odes)
- [Training: backpropagation through the solver and the adjoint method](#training-backpropagation-through-the-solver-and-the-adjoint-method)
- [Uses in scientific machine learning](#uses-in-scientific-machine-learning)
- [Limitations](#limitations)
- [Code: learning a damped pendulum from irregular samples](#code-learning-a-damped-pendulum-from-irregular-samples)
- [In an interview](#in-an-interview)

## From residual networks to ODEs

A residual block computes h_{k+1} = h_k + f_θ(h_k): one explicit Euler step of size 1 of the ODE dh/dt = f_θ(h). Making the steps infinitely small gives the continuous model

```math
\frac{dy}{dt} = f_\theta\big(y(t), t\big), \qquad y(t_1) = y(t_0) + \int_{t_0}^{t_1} f_\theta\big(y(t), t\big)\,dt
```

and the "forward pass" becomes a call to an ODE solver: y(t₁) = ODESolve(f_θ, y(t₀), t₀, t₁).

**What this buys.**

1. **Continuous time.** Query the state at any time, not only at the training time step; data sampled at irregular times needs no resampling or imputation.
2. **Adaptive computation.** An adaptive solver (Dormand-Prince, for example) takes more steps where the dynamics change fast and fewer where they are smooth, trading accuracy against cost through a tolerance.
3. **Physical interpretability.** The network is a vector field, the same object as the right-hand side of a physical model, so known terms can be combined with learned ones.
4. **Memory-efficient training** with the adjoint method (below).

**For physics:** a time-stepping surrogate u_{t+1} = u_t + G_θ(u_t) is a discrete flow map with a fixed step; a neural ODE learns the generator of the flow instead. See [Autoregressive rollouts](autoregressive-rollouts.md) for the discrete version and its drift problems.

## Training: backpropagation through the solver and the adjoint method

**Discretize-then-optimize.** Treat the solver's arithmetic as part of the computational graph and backpropagate through every stage of every step. Exact gradients of the discretized model, simple, and fast for short integrations, but memory grows with the number of steps (every intermediate state is stored).

**Optimize-then-discretize: the adjoint method.** Define the adjoint state a(t) = ∂L/∂y(t). It satisfies its own ODE, solved **backwards** from t₁ to t₀ together with the state:

```math
\frac{da}{dt} = -a^\top \frac{\partial f_\theta}{\partial y}, \qquad
\frac{dL}{d\theta} = -\int_{t_1}^{t_0} a^\top \frac{\partial f_\theta}{\partial \theta}\, dt
```

The vector-Jacobian products a ᵀ∂f/∂y and a ᵀ∂f/∂θ are computed with reverse-mode automatic differentiation. Memory is **constant in the number of steps**, because intermediate states are recomputed by integrating backwards instead of being stored. This is the same adjoint method used for PDE-constrained optimization. See [The adjoint method](adjoint-method.md).

**Trade-offs.** The backward reconstruction of y(t) can be inaccurate (or unstable for dissipative or chaotic dynamics, where reversing time amplifies errors), so the adjoint gradient can differ from the true gradient of the discretized forward model. Checkpointing schemes store some states to fix this. In practice, for small problems, backpropagating through the solver is often preferred; the adjoint wins when memory is the constraint.

**Libraries:** torchdiffeq (`odeint` and `odeint_adjoint`), Diffrax (JAX), and the Julia SciML ecosystem (DifferentialEquations.jl with SciMLSensitivity), which offers many sensitivity algorithms and stiff solvers.

## Uses in scientific machine learning

- **Learning dynamics from time series:** measured or simulated trajectories at irregular times (sensor logs, experiments, clinical-style data).
- **Latent ODEs (Rubanova, Chen and Duvenaud, NeurIPS 2019):** an encoder maps an irregularly sampled sequence to an initial latent state, a neural ODE evolves it, and a decoder maps back; a continuous-time alternative to recurrent networks.
- **Reduced-order models:** compress fields with POD or an autoencoder and learn the latent dynamics with a neural ODE. See [When a linear subspace fails](reduced-order-models.md#when-a-linear-subspace-fails).
- **Universal differential equations (Rackauckas et al., 2020):** keep the known physics and let a network represent only the unknown part, du/dt = f_known(u) + NN_θ(u). The learned term can then be distilled into a symbolic expression with [SINDy](sindy-and-koopman.md#sindy).
- **Structure-preserving variants:** Hamiltonian and Lagrangian networks are neural ODEs whose vector field is derived from a learned energy. See [Hamiltonian and Lagrangian neural networks](hamiltonian-and-lagrangian-neural-networks.md).
- **Continuous normalizing flows:** generative models whose change of variables follows an ODE; flow matching trains such models without integrating during training. See [Flow matching](generative-models-for-physics.md#flow-matching).
- **Neural operators in time:** continuous-time variants evolve a field with a learned operator as the right-hand side, for example method-of-lines approaches where a GNN gives du/dt on a mesh.

## Limitations

1. **Training cost.** Every forward pass runs an ODE solver; adaptive solvers may take many steps, and the number of steps can grow during training as the learned dynamics become more complex. Regularizing the dynamics (penalizing the norm of f or its Jacobian) keeps them cheap to integrate.
2. **Stiffness.** Learned dynamics can become stiff (fast and slow time scales), which forces explicit solvers to take tiny steps; implicit solvers and rescaling help. Stiff physical systems (chemistry, plasticity) are hard targets.
3. **Expressivity of autonomous ODEs.** Solutions of dy/dt = f(y) cannot cross in state space, so a neural ODE cannot represent maps that would need trajectories to cross (in a classification setting, for example). Augmented neural ODEs (Dupont et al., 2019) add extra dimensions to the state to lift this restriction.
4. **Chaos and long horizons.** Gradients through long chaotic trajectories explode; train on short windows (multiple shooting), as in the code below.
5. **Not automatically physical.** Without structure (conservation, symmetries), long integrations drift like any learned model.
6. **For large PDE fields,** a neural ODE on 10⁶ degrees of freedom is expensive to integrate; use it on reduced or latent states, or prefer discrete time-steppers trained with rollout-aware methods.

## Code: learning a damped pendulum from irregular samples

The true system is a damped pendulum, dθ/dt = ω, dω/dt = −sin θ − 0.1ω, observed with noise at 200 **irregularly spaced** times in [0, 20]. A small MLP is the vector field; a hand-written RK4 integrator steps between whatever observation times occur (no resampling needed), and training uses short windows of 10 consecutive observations (multiple shooting), with gradients backpropagated through the solver. The trained model is then integrated from t = 0 to t = 40, twice the observed time span.

```python
import torch, torch.nn as nn

torch.manual_seed(0)
def f_true(y):                                         # damped pendulum: y = (angle, angular velocity)
    return torch.stack([y[..., 1], -torch.sin(y[..., 0]) - 0.1 * y[..., 1]], -1)

def odeint_rk4(f, y0, t, substeps=2):
    """Integrate dy/dt = f(y) from t[0] through every time in t (irregular spacing allowed) with RK4.
    t has shape (T,) or (T, batch) for a batch of windows that each have their own times."""
    ys, y = [y0], y0
    for i in range(len(t) - 1):
        h = ((t[i + 1] - t[i]) / substeps)[..., None]
        for _ in range(substeps):
            k1 = f(y); k2 = f(y + 0.5 * h * k1); k3 = f(y + 0.5 * h * k2); k4 = f(y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        ys.append(y)
    return torch.stack(ys)

y0 = torch.tensor([2.5, 0.0])
t_obs = torch.cat([torch.zeros(1), torch.sort(torch.rand(199) * 20).values])    # 200 irregular times in [0, 20]
with torch.no_grad():
    y_obs = odeint_rk4(f_true, y0, t_obs, substeps=20) + 0.02 * torch.randn(200, 2)

func = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2))   # f_theta(y)
opt = torch.optim.Adam(func.parameters(), lr=3e-3)
W = 10                                                 # windows of 10 consecutive observations
for it in range(1500):
    start = torch.randint(0, 200 - W, (32,))
    idx = start[None, :] + torch.arange(W)[:, None]    # (W, batch) observation indices
    pred = odeint_rk4(func, y_obs[start], t_obs[idx])  # integrate each window from its first observation
    loss = ((pred - y_obs[idx]) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()       # gradients flow back through the RK4 steps
print(f"final window loss {loss.item():.5f} (observation noise variance {0.02 ** 2:.5f})")

with torch.no_grad():
    t_dense = torch.linspace(0, 40, 801)               # any query times, including beyond the data (t > 20)
    truth = odeint_rk4(f_true, y0, t_dense, substeps=10)
    model = odeint_rk4(func, y0, t_dense, substeps=2)
    err = (model - truth).norm(dim=-1)
print(f"integrated from t = 0: mean state error for t <= 20 (data range) {err[t_dense <= 20].mean():.4f}, "
      f"for 20 < t <= 40 (extrapolation) {err[t_dense > 20].mean():.4f}; mean state size {truth.norm(dim=-1).mean():.3f}")
```

Output:

```
final window loss 0.00074 (observation noise variance 0.00040)
integrated from t = 0: mean state error for t <= 20 (data range) 0.1133, for 20 < t <= 40 (extrapolation) 0.0990; mean state size 0.922
```

**What it shows.** Trained only on short windows, the learned vector field reproduces the whole trajectory when integrated from the initial state, with an error around a tenth of the state size, and keeps doing so beyond the observed time span, because it learned the dynamics rather than the trajectory. The window loss sits close to the noise floor (twice the noise variance, since both the starting point and the targets are noisy). The irregular sampling needed no special handling: the solver simply steps to each observation time.

## In an interview

**Thirty-second answer to "What are neural ODEs?"** "A network that defines the right-hand side of an ODE, dy/dt = f_theta(y, t), with outputs obtained by integrating it with an ODE solver. It is the continuous limit of a residual network. It learns a vector field rather than a fixed-step map, so it can be queried at any time, handles irregularly sampled data, and can use adaptive solvers. Gradients come from backpropagating through the solver or from the adjoint method, which solves a backward ODE with constant memory. In physics it is useful for learning dynamics from time series, latent dynamics of reduced models, and universal differential equations where the network models only the unknown term."

**Likely follow-ups**

- *What is the adjoint method here?* A backward ODE for ∂L/∂y(t) whose integral gives ∂L/∂θ with memory independent of the number of steps; the same idea as PDE adjoints.
- *Why can training be slow?* Each forward pass integrates an ODE, and learned dynamics can become stiff or complicated, increasing the number of solver steps.
- *Neural ODE or a discrete time-stepper for a CFD field?* For large fields usually a discrete stepper trained with rollout-aware methods; neural ODEs shine on low-dimensional or latent states and irregular data.

## References

- [Neural Ordinary Differential Equations (Chen, Rubanova, Bettencourt & Duvenaud, NeurIPS 2018)](https://arxiv.org/abs/1806.07366)
- [Latent ODEs for Irregularly-Sampled Time Series (Rubanova, Chen & Duvenaud, NeurIPS 2019)](https://arxiv.org/abs/1907.03907)
- [Augmented Neural ODEs (Dupont, Doucet & Teh, NeurIPS 2019)](https://arxiv.org/abs/1904.01681)
- [Universal Differential Equations for Scientific Machine Learning (Rackauckas et al.)](https://arxiv.org/abs/2001.04385)
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq) and [Diffrax](https://github.com/patrick-kidger/diffrax)
- [On Neural Differential Equations (Kidger, PhD thesis)](https://arxiv.org/abs/2202.02435)
