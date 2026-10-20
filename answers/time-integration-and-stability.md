# Time Integration and Stability

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A transient simulation advances the solution in time steps. An **explicit** method computes the new state directly from the current one: each step is cheap, but the step size is limited by stability. For waves and advection, the limit is the **CFL condition**: in one step, information must not travel further than about one cell (Courant number u·Δt/Δx ≤ about 1). For diffusion with the simplest explicit scheme, αΔt/Δx² ≤ 1/2. Break the limit and errors grow exponentially until the solution blows up. An **implicit** method computes the new state by solving a system of equations that involves it: each step is expensive, but suitable implicit methods stay stable at much larger steps. A **stiff** problem contains processes on very different time scales; an explicit method must take steps small enough for the fastest one even after it has died out, so implicit methods win.

**On this page**

- [Explicit vs implicit time integration](#explicit-vs-implicit-time-integration)
- [The CFL condition](#the-cfl-condition)
- [Stability of the explicit heat equation](#stability-of-the-explicit-heat-equation)
- [Stiff equations](#stiff-equations)
- [Code: explicit heat-equation solver and its stability limit](#code-explicit-heat-equation-solver-and-its-stability-limit)
- [Code: a stiff equation, explicit vs implicit](#code-a-stiff-equation-explicit-vs-implicit)
- [Which method where](#which-method-where)
- [Why this matters for machine learning](#why-this-matters-for-machine-learning)
- [In an interview](#in-an-interview)

## Explicit vs implicit time integration

Write the semi-discrete system (after discretizing in space) as du/dt = f(u, t). The two simplest time-stepping schemes are:

```math
\text{explicit (forward) Euler:}\quad u^{n+1} = u^n + \Delta t\, f(u^n, t^n)
```

```math
\text{implicit (backward) Euler:}\quad u^{n+1} = u^n + \Delta t\, f(u^{n+1}, t^{n+1})
```

The only difference is *where* f is evaluated. In the explicit scheme everything on the right is known, so the new state is one function evaluation away. In the implicit scheme the unknown u^{n+1} appears on both sides, so every step requires solving a (linear or nonlinear) system of equations.

| | Explicit | Implicit |
| --- | --- | --- |
| Cost per step | Low: evaluate f (and, in structures, invert a diagonal mass matrix) | High: solve a linear system, or a Newton iteration for nonlinear problems |
| Stability | Conditional: Δt must be below a limit set by the mesh and the physics | Unconditional for A-stable methods such as backward Euler and the trapezoidal rule (Crank-Nicolson) |
| Step size chosen by | Stability (often far smaller than accuracy needs) | Accuracy |
| Memory | Low; easy to parallelize | Needs a matrix and a solver; harder to scale |
| Robust for | Short, violent events: impact, crash, blast, wave propagation | Long, slow processes; statics; stiff problems |

**The trade-off in one sentence:** explicit methods take many cheap steps; implicit methods take few expensive steps. Which is faster depends on how much smaller the stability limit is than the step size that accuracy requires.

**Accuracy is a separate question from stability.** Forward and backward Euler are both first-order accurate: halving Δt halves the error. Crank-Nicolson and BDF2 are second-order. A stable but inaccurate scheme gives a smooth wrong answer, which is why time-step convergence studies matter as much as mesh studies.

## The CFL condition

The Courant-Friedrichs-Lewy (CFL) condition is a *necessary* condition for an explicit scheme for waves or advection to converge: the numerical scheme must be able to "see" all the information that the real solution depends on.

**The physical picture.** In advection at speed u, information travels a distance u·Δt in one time step. An explicit scheme with a three-point stencil only uses values from the neighbouring cells, so in one step it can only move information by one cell, Δx. If the real information travels further than one cell per step, the scheme is computing the new value without the data that determines it. That cannot converge.

```
time n+1:          ●             the new value at this point...
                 ╱ │ ╲
time n:     ●───●───●───●        ...uses only these three points (one cell each side).
            ◀── u·Δt ──▶        If the true domain of dependence reaches further, CFL is violated.
```

**The Courant number:**

```math
C = \frac{|u|\,\Delta t}{\Delta x} \;\le\; C_{\max}
```

with C_max around 1 for many simple explicit schemes (the exact limit depends on the scheme and dimension). For waves, replace u with the wave speed: in compressible flow |u| + c (flow speed plus speed of sound); in structural dynamics the speed of stress waves in the material.

**A worked number.** In explicit crash simulation the stable time step is roughly the time a stress wave takes to cross the smallest element. In steel, the elastic wave speed is about √(E/ρ) = √(210 × 10⁹ / 7850) ≈ 5,200 m/s. For a 5 mm element, Δt ≈ 0.005 / 5,200 ≈ 1 microsecond. A 100-millisecond crash event therefore needs on the order of 100,000 time steps, and a single badly shaped tiny element anywhere in the model can shrink the time step for the whole model. That is why crash codes report the smallest elements, and why analysts sometimes add mass to them ("mass scaling").

**What happens if you violate it?** The highest-frequency error components are amplified at every step. They start at round-off level and grow exponentially, usually appearing as a sawtooth oscillation from cell to cell, and within a few dozen to a few hundred steps the solution overflows. Implicit methods are not limited by CFL for stability, but a Courant number far above 1 still costs accuracy in transient flows, because fast features are smeared out.

## Stability of the explicit heat equation

For diffusion, u_t = α u_xx, the forward-time centred-space (FTCS) scheme is

```math
u_i^{n+1} = u_i^n + r\left(u_{i+1}^n - 2u_i^n + u_{i-1}^n\right), \qquad r = \frac{\alpha\,\Delta t}{\Delta x^2}
```

**Von Neumann stability analysis** substitutes a single Fourier mode, u_i^n = gⁿ e^{ikiΔx}, and asks when the amplification factor g has |g| ≤ 1 for every wavenumber k:

```math
g(k) = 1 - 4r\,\sin^2\!\left(\frac{k\,\Delta x}{2}\right)
```

The worst case is the shortest wave on the grid (sin² = 1), where g = 1 − 4r. Requiring |1 − 4r| ≤ 1 gives

```math
r = \frac{\alpha\,\Delta t}{\Delta x^2} \le \frac{1}{2}
```

Notice that the limit scales with **Δx²**, not Δx. Halving the mesh size forces a time step four times smaller, so the total cost of an explicit diffusion simulation in 1D grows eight times. This is the main reason diffusion-dominated problems (heat conduction, viscous terms) are usually integrated implicitly.

## Stiff equations

**Definition.** A system is *stiff* when it contains components that evolve on very different time scales, and the fast components, although they quickly decay to near-equilibrium, still force an explicit method to take tiny steps. Formally, the eigenvalues of the Jacobian ∂f/∂u span many orders of magnitude, and the fastest ones are large and negative.

**A minimal example:**

```math
\frac{dy}{dt} = -\lambda\,\big(y - \cos t\big), \qquad \lambda = 1000, \qquad y(0) = 0 .
```

The solution relaxes to cos t within about 1/λ = 1 millisecond, then follows the slow cos t for the rest of time. After the first few milliseconds, accuracy would allow a step of 0.05 or 0.1. But explicit Euler multiplies any deviation from the slow solution by (1 − λΔt) every step, so it is stable only if |1 − λΔt| ≤ 1, that is Δt ≤ 2/λ = 0.002, for the *whole* simulation. Implicit Euler multiplies the deviation by 1/(1 + λΔt), which is below 1 for any Δt > 0.

The code below shows it: with Δt = 0.0025, only 25% above the limit, explicit Euler returns about −10⁷⁰ at t = 1; implicit Euler with a step 40 times larger (Δt = 0.1) is still within 0.00003 of the exact answer.

**Where stiffness appears:**

- chemical kinetics and combustion (reaction rates spanning many orders of magnitude);
- diffusion on fine meshes (the fastest mode decays at a rate proportional to α/Δx²);
- structural dynamics with stiff and soft parts, or stiff contact springs;
- circuit simulation, battery electrochemistry, multiphysics couplings.

**Methods for stiff problems** are implicit and have large stability regions: backward Euler, the backward differentiation formulas (BDF, used in classic stiff ODE solvers), implicit Runge-Kutta methods such as Radau, and, in structural dynamics, the Newmark and HHT-α families. Each step needs a nonlinear solve, usually Newton's method, with the Jacobian. See [Newton-Raphson](newton-raphson-and-nonlinear-solvers.md).

## Code: explicit heat-equation solver and its stability limit

```python
import numpy as np

def ftcs_heat(r, nx=101, L=1.0, alpha=0.01, steps=500):
    """Explicit FTCS scheme for u_t = alpha u_xx with u = 0 at both ends; returns the max error."""
    dx = L / (nx - 1)
    dt = r * dx**2 / alpha                      # r = alpha dt / dx^2 sets the step
    x = np.linspace(0, L, nx)
    u = np.sin(np.pi * x)                       # initial condition; exact decay is known
    for _ in range(steps):
        u[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
    exact = np.sin(np.pi * x) * np.exp(-alpha * np.pi**2 * steps * dt)
    return np.max(np.abs(u - exact))

for r in [0.25, 0.4, 0.5, 0.6]:
    print(f"r = {r:<5} max error after 500 steps = {ftcs_heat(r):.2e}")
```

Output:

```
r = 0.25  max error after 500 steps = 4.48e-06
r = 0.4   max error after 500 steps = 1.87e-05
r = 0.5   max error after 500 steps = 3.17e-05
r = 0.6   max error after 500 steps = 2.25e+56
```

Up to r = 0.5 the error is tiny; at r = 0.6 the solution has exploded. Just above the limit (say r = 0.51) the growth factor of the worst mode is only 1.04 per step, and it starts from round-off, so it can take hundreds of steps before the instability is visible. A short test run can therefore look fine and still be unstable.

## Code: a stiff equation, explicit vs implicit

```python
import numpy as np

lam = 1000.0                                   # fast time scale 1/lam; slow forcing cos(t)

def exact(t):
    return (lam**2 * np.cos(t) + lam * np.sin(t)) / (1 + lam**2) - lam**2 / (1 + lam**2) * np.exp(-lam * t)

def explicit_euler(dt, T=1.0):
    y = 0.0
    for k in range(int(round(T / dt))):
        y = y + dt * (-lam * (y - np.cos(k * dt)))          # uses the current state only
    return y

def implicit_euler(dt, T=1.0):
    y = 0.0
    for k in range(int(round(T / dt))):
        t1 = (k + 1) * dt
        y = (y + dt * lam * np.cos(t1)) / (1 + dt * lam)    # solves for the new state (linear here)
    return y

print(f"exact y(1) = {exact(1.0):.6f}")
for dt in [0.001, 0.002, 0.0025, 0.01, 0.1]:
    print(f"dt = {dt:<7} explicit: {explicit_euler(dt):12.4g}   implicit: {implicit_euler(dt):.6f}")
```

Output:

```
exact y(1) = 0.541143
dt = 0.001   explicit:       0.5411   implicit: 0.541143
dt = 0.002   explicit:      -0.4589   implicit: 0.541143
dt = 0.0025  explicit:   -2.732e+70   implicit: 0.541143
dt = 0.01    explicit:   -2.656e+95   implicit: 0.541141
dt = 0.1     explicit:   -9.044e+19   implicit: 0.541115
```

At exactly Δt = 2/λ = 0.002 the explicit amplification factor is −1: the initial transient never decays and the answer is off by 1. Above it, the solution explodes. The implicit method is accurate at every step size shown.

## Which method where

| Application | Typical choice | Why |
| --- | --- | --- |
| Crash, impact, drop tests, blast | Explicit (central difference), lumped mass | Short events, extreme nonlinearity and contact; small steps needed anyway for accuracy |
| Structural statics, slow loading | Implicit (static Newton iterations) | No inertia; equilibrium at each load step |
| Structural vibration, seismic | Implicit (Newmark, HHT-α) | Long durations; stable at steps set by the frequencies of interest |
| Steady CFD (RANS) | Implicit pseudo-time stepping | Only the converged steady state matters |
| Transient incompressible CFD | Implicit or semi-implicit (PISO, PIMPLE) | Pressure is solved implicitly; Courant number often kept near or below 1 for accuracy |
| Compressible CFD, aeroacoustics, LES | Explicit Runge-Kutta, or implicit dual time stepping | Waves must be resolved anyway |
| Chemistry, batteries | Implicit (BDF, Radau) | Stiffness |

## Why this matters for machine learning

- **Autoregressive surrogates are explicit time-steppers.** A model that predicts the next state from the current one, x_{t+1} = F(x_t), is an explicit scheme with a learned F. It can be unstable in the same way: small errors grow at each step until the rollout blows up. Training tricks such as noise injection and the pushforward trick target exactly this. See [Autoregressive rollouts](autoregressive-rollouts.md).
- **The learned time step can be much larger than the solver's,** because the network is not bound by CFL: it learns the map over a large Δt directly. That is part of where speed-ups come from, and also why high-frequency detail can be lost.
- **MP-PDE and APEBench connect learned emulators to classical stability ideas.** MP-PDE frames stability through zero-stability and the pushforward trick; APEBench characterizes its PDE dynamics with an identifier tied to the stability criteria of classical numerical methods. See [Message Passing Neural PDE Solvers](autoregressive-rollouts.md#message-passing-neural-pde-solvers).
- **Stiff physics is hard for PINNs too:** widely separated time scales make the loss landscape badly conditioned. See [PINN training failures](pinn-training-failures.md).
- **Crash datasets inherit tiny time steps.** Explicit crash simulations write out only a subset of their many time steps, and a surrogate learns on those output frames, not on the solver's internal steps. See [Crash-simulation surrogates](crash-simulation-surrogates.md).

## In an interview

**Thirty-second answer to "What is the CFL condition?"** "For an explicit scheme on waves or advection, the time step must be small enough that information does not travel further than about one cell per step: the Courant number u·Δt/Δx must stay below a limit near one. Otherwise the scheme is missing the data the true solution depends on; the shortest waves get amplified every step and the solution blows up. Implicit methods remove the stability limit but cost a linear solve per step."

**Likely follow-ups**

- *Explicit or implicit for a crash simulation?* Explicit: short event, heavy contact and nonlinearity; the stable step is about a microsecond for millimetre-scale steel elements, and the event lasts about 100 ms.
- *What makes an equation stiff?* Widely separated time scales; the fastest decaying component limits the explicit step long after it stops mattering. Implicit methods with large stability regions (BDF, Radau) are the answer.
- *How does the stability limit for diffusion scale with mesh size?* As Δx², so explicit diffusion gets expensive fast on fine meshes.

## References

- [CFD Python: 12 steps to Navier–Stokes (Lorena Barba), including the CFL condition and diffusion steps](https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/)
- R. J. LeVeque, *Finite Difference Methods for Ordinary and Partial Differential Equations*, SIAM, 2007 (stability, stiffness).
- E. Hairer and G. Wanner, *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*, Springer, 1996.
- [APEBench: A Benchmark for Autoregressive Neural Emulators of PDEs (Koehler et al.)](https://arxiv.org/abs/2411.00180)
