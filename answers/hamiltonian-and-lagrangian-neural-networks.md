# Hamiltonian and Lagrangian Neural Networks

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Both build **conservation of energy** into a neural network by learning a **scalar energy function** instead of the equations of motion directly. A **Hamiltonian neural network (HNN)** (Greydanus, Dzamba and Yosinski, NeurIPS 2019) learns H(q, p), the total energy in terms of positions q and momenta p, and obtains the dynamics from Hamilton's equations, dq/dt = ∂H/∂p and dp/dt = −∂H/∂q, by automatic differentiation. Because the predicted flow is always a symplectic gradient of a learned H, that H is exactly conserved along the model's continuous-time trajectories, so long rollouts do not slowly gain or lose energy the way an unconstrained network does. A **Lagrangian neural network (LNN)** (Cranmer et al., 2020) learns the Lagrangian L(q, q̇) instead and gets accelerations from the Euler-Lagrange equations; it needs no canonical momenta, which are often unknown in practice. Both suit conservative systems (orbits, pendulums, molecular dynamics, undamped structural vibration); real engineering systems with friction, damping, forcing or contact need extensions (port-Hamiltonian and dissipative models).

**On this page**

- [Why build in a conservation law](#why-build-in-a-conservation-law)
- [Hamiltonian neural networks](#hamiltonian-neural-networks)
- [Lagrangian neural networks](#lagrangian-neural-networks)
- [Related structure-preserving models](#related-structure-preserving-models)
- [Limits for engineering problems](#limits-for-engineering-problems)
- [Code: an HNN and a plain network on a pendulum](#code-an-hnn-and-a-plain-network-on-a-pendulum)
- [In an interview](#in-an-interview)

## Why build in a conservation law

A network trained to predict time derivatives (or the next state) from the current state makes small errors at every step. For a conservative system, nothing stops those errors from adding a little energy, or removing it, each step. Over thousands of steps the orbit spirals outward or decays: the model invents damping or instability that the physics does not have. This is the long-rollout drift problem in its simplest form. See [Why one-step models drift](autoregressive-rollouts.md#why-one-step-models-drift).

Building the conservation law into the architecture removes that error mode: whatever the network learns, the quantity it treats as energy cannot change along its trajectories. The model also learns from fewer examples, because a large class of wrong answers is excluded. The same idea underlies [hard constraints](hard-vs-soft-constraints.md) and [equivariant networks](equivariance-and-symmetry.md): encode what you know exactly, learn the rest.

## Hamiltonian neural networks

**Hamiltonian mechanics in one paragraph.** For a conservative system with generalized coordinates q and conjugate momenta p, the Hamiltonian H(q, p) is the total energy (for a mechanical system, kinetic plus potential). The dynamics are

```math
\frac{dq}{dt} = \frac{\partial H}{\partial p}, \qquad \frac{dp}{dt} = -\frac{\partial H}{\partial q}
```

and dH/dt = (∂H/∂q)(dq/dt) + (∂H/∂p)(dp/dt) = 0: energy is conserved automatically. The flow also preserves phase-space volume (it is symplectic) and is reversible in time.

**The HNN.**

1. A network H_θ(q, p) outputs a single scalar.
2. Automatic differentiation gives ∂H_θ/∂q and ∂H_θ/∂p.
3. The predicted time derivative is (∂H_θ/∂p, −∂H_θ/∂q).
4. Train by matching predicted derivatives to observed ones (from finite differences of trajectories, or from a simulator): loss = ‖dq/dt_pred − dq/dt_obs‖² + ‖dp/dt_pred − dp/dt_obs‖².
5. To predict, integrate the learned vector field with an ODE solver (a symplectic integrator such as leapfrog preserves the structure best in discrete time).

**What the paper showed.** On an ideal mass-spring, an ideal pendulum, a real pendulum from measured data, the two-body problem and a pendulum observed only through pixels (with an autoencoder providing the coordinates), HNNs trained faster, generalized better and conserved energy where baseline networks drifted; as a side effect, the model is perfectly reversible in time.

**Practical requirements.** You need **canonical coordinates** (q, p) in which the system is Hamiltonian. Positions are usually observed, but momenta often are not, and in complex systems (constrained multibody systems, non-Cartesian coordinates) the canonical momenta can be hard to define. Training also needs time derivatives, which are noisy when estimated from data.

## Lagrangian neural networks

**Lagrangian mechanics.** With L(q, q̇) = kinetic − potential energy, the Euler-Lagrange equations d/dt(∂L/∂q̇) = ∂L/∂q give, after expanding the time derivative,

```math
\ddot q = \Big(\nabla_{\dot q}\nabla_{\dot q}^\top L\Big)^{-1}\Big[\nabla_q L - \big(\nabla_q \nabla_{\dot q}^\top L\big)\,\dot q\Big]
```

**The LNN (Cranmer, Greydanus, Hoyer, Battaglia, Spergel and Ho, 2020).** A network parametrizes an arbitrary L_θ(q, q̇); the formula above, computed with automatic differentiation (Hessians and a linear solve), gives accelerations; training matches observed accelerations. Advantages: it works with **positions and velocities**, which are what you usually measure, so no canonical momenta are needed; it does not restrict the functional form of the energy (some earlier Lagrangian methods assumed a quadratic kinetic energy); and it conserves energy. The paper demonstrated it on a double pendulum and a relativistic particle, and extended it to graphs for systems of many particles.

**Costs.** Each prediction needs second derivatives of the network and the inverse of a Hessian, which is more expensive and can be ill-conditioned when that Hessian is nearly singular.

## Related structure-preserving models

- **Symplectic recurrent networks and symplectic ODE-Nets:** learn H and integrate it with a symplectic scheme during training, with multi-step losses, which handles noisy data better (Chen et al., ICLR 2020).
- **Deep Lagrangian Networks (DeLaN; Lutter et al., 2019):** a Lagrangian with an explicitly positive-definite mass matrix for robot dynamics and control.
- **Graph versions:** Hamiltonian and Lagrangian graph networks for many-body systems.
- **Port-Hamiltonian and dissipative networks:** H plus learned damping and input terms, dx/dt = (J − R)∇H + Bu with R positive semidefinite, so energy can only decrease by the learned dissipation. This is the version relevant to most engineering systems.
- **GENERIC / metriplectic networks:** combine a conserved energy with non-decreasing entropy, for thermodynamically consistent models of dissipative materials.
- **Conservative PDE surrogates:** finite-volume-style architectures that update cells only through fluxes, so mass and momentum are conserved exactly. See [Physics-consistency checks](evaluation-metrics.md#physics-consistency-checks).

## Limits for engineering problems

1. **Most engineering systems are not conservative.** Structural damping, friction, turbulence, heat losses and external forcing all change energy. A pure HNN would be wrong by design; use port-Hamiltonian or dissipative formulations, or model only the conservative part.
2. **Contact and impacts** break smoothness (bouncing, crash). Special treatment is needed.
3. **High-dimensional fields.** An HNN on 10⁶ degrees of freedom is impractical directly; combine with model reduction (a Hamiltonian ROM in reduced coordinates) or graph structure.
4. **Learned energy, not true energy.** The model conserves *its own* H_θ exactly in continuous time. If H_θ is wrong, the trajectory is wrong, just stably so; and a non-symplectic integrator still introduces small energy errors in discrete time.
5. **Data requirements.** Time derivatives are needed; estimating them from noisy trajectories is the same weakness as in [SINDy](sindy-and-koopman.md#sindy).

Where they shine: orbital mechanics, molecular dynamics, undamped vibration and wave problems, robotics (with dissipation terms), and as a component in larger models whenever a known invariant should not drift.

## Code: an HNN and a plain network on a pendulum

An ideal pendulum, H = p²/2 + (1 − cos q). Both models are trained on the same 2,000 noisy samples of the time derivative (noise standard deviation 0.05); one network predicts the derivative directly, the other predicts H and differentiates it. Both are then rolled out for 300 time units (3,000 steps) with the same fourth-order Runge-Kutta integrator, and the true energy of the predicted state is tracked.

```python
import torch, torch.nn as nn

torch.manual_seed(0)
def true_field(x):                                          # ideal pendulum: dq/dt = p, dp/dt = -sin q
    q, p = x[:, :1], x[:, 1:]
    return torch.cat([p, -torch.sin(q)], 1)
energy = lambda x: 0.5 * x[:, 1] ** 2 + 1 - torch.cos(x[:, 0])
X = torch.rand(2000, 2) * 4 - 2                             # states with q, p in [-2, 2]
dX = true_field(X) + 0.05 * torch.randn(2000, 2)            # noisy derivative labels

class HNN(nn.Module):
    """Learn a scalar H(q, p); the vector field is the symplectic gradient (dH/dp, -dH/dq)."""
    def __init__(self):
        super().__init__()
        self.H = nn.Sequential(nn.Linear(2, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
    def forward(self, x):
        x = x if x.requires_grad else x.requires_grad_(True)
        dH = torch.autograd.grad(self.H(x).sum(), x, create_graph=True)[0]
        return torch.cat([dH[:, 1:], -dH[:, :1]], 1)

class Baseline(nn.Module):
    """Learn the vector field directly: (q, p) -> (dq/dt, dp/dt)."""
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(2, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 2))
    def forward(self, x):
        return self.f(x)

def train(model, steps=3000):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(steps):
        idx = torch.randint(0, 2000, (256,))
        loss = ((model(X[idx].clone()) - dX[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model

def rk4_rollout(field, x0, dt=0.1, steps=3000):
    xs, x = [x0], x0
    for _ in range(steps):
        k1 = field(x); k2 = field(x + 0.5 * dt * k1); k3 = field(x + 0.5 * dt * k2); k4 = field(x + dt * k3)
        x = (x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)).detach()
        xs.append(x)
    return torch.cat(xs)

models = {"baseline MLP": train(Baseline()), "Hamiltonian NN": train(HNN())}
x0 = torch.tensor([[1.5, 0.0]])                             # released from 1.5 rad at rest
E0 = energy(x0).item()
for name, m in models.items():
    E = energy(rk4_rollout(lambda x: m(x.clone()), x0))
    print(f"{name:>15}: energy after 300 time units {E[-1].item():.4f} (start {E0:.4f}), max drift {(E - E0).abs().max().item():.4f}")
print(f"{'true dynamics':>15}: energy after 300 time units {energy(rk4_rollout(true_field, x0))[-1].item():.4f}")
```

Output:

```
   baseline MLP: energy after 300 time units 0.2050 (start 0.9293), max drift 0.7411
 Hamiltonian NN: energy after 300 time units 0.9455 (start 0.9293), max drift 0.0340
  true dynamics: energy after 300 time units 0.9292
```

**What it shows.** The plain network fits the training derivatives about as well as the HNN, but its small errors are not structured, and over 3,000 steps they act like spurious damping: the pendulum loses most of its energy. The HNN's trajectory stays on a level set of its learned H, so the true energy stays within a few percent of the start; the remaining drift comes from H_θ not being exactly the true H, not from accumulation.

## In an interview

**Thirty-second answer to "What are Hamiltonian and Lagrangian neural networks?"** "Instead of learning the equations of motion directly, they learn a scalar energy function and derive the dynamics from it. An HNN learns H of positions and momenta and uses Hamilton's equations, dq/dt equals dH/dp and dp/dt equals minus dH/dq, through autograd, so the learned energy is exactly conserved and long rollouts do not drift. An LNN learns the Lagrangian of positions and velocities and gets accelerations from the Euler-Lagrange equations, which avoids needing canonical momenta. They are great for conservative systems; for real engineering systems with damping or forcing you need port-Hamiltonian or dissipative extensions."

**Likely follow-ups**

- *Why does the plain network drift?* Its per-step errors are unstructured and can add or remove energy every step.
- *What does HNN not guarantee?* That its H is the true energy, or exact conservation after time discretization with a non-symplectic integrator.
- *How would you handle damping?* A port-Hamiltonian model with a learned positive semidefinite dissipation matrix, so energy can only decrease through it.

## References

- [Hamiltonian Neural Networks (Greydanus, Dzamba & Yosinski, NeurIPS 2019)](https://arxiv.org/abs/1906.01563)
- [Lagrangian Neural Networks (Cranmer et al.)](https://arxiv.org/abs/2003.04630)
- [Symplectic Recurrent Neural Networks (Chen et al., ICLR 2020)](https://arxiv.org/abs/1909.13334)
- [Deep Lagrangian Networks (Lutter, Ritter & Peters, ICLR 2019)](https://arxiv.org/abs/1907.04490)
- H. Goldstein, C. Poole and J. Safko, *Classical Mechanics*, 3rd edition, Addison-Wesley (Hamiltonian and Lagrangian mechanics).
