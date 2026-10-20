# The Adjoint Method

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** The adjoint method computes the gradient of **one** output (drag, compliance, mean temperature, a misfit to measurements) with respect to **any number** of design variables for the cost of the forward solve plus **one extra linear solve**, the adjoint equation. Finite differences, and the direct (forward) sensitivity method, need one solve per design variable instead. That is why adjoints are the standard tool for aerodynamic shape optimization, topology optimization, inverse problems and data assimilation, where there are thousands to millions of variables. For a discretized PDE R(u, p) = 0 and objective J(u, p): solve (∂R/∂u)ᵀλ = (∂J/∂u)ᵀ, then dJ/dp = ∂J/∂p − λᵀ ∂R/∂p. Reverse-mode automatic differentiation (backpropagation) applied to the solver's code computes exactly this, a **discrete adjoint**. In the code below, the adjoint gives all 400 derivatives dozens of times faster than finite differences and matches reverse-mode autodiff to 10⁻¹².

**On this page**

- [Derivation](#derivation)
- [Direct vs adjoint sensitivities](#direct-vs-adjoint-sensitivities)
- [Worked example: compliance in topology optimization](#worked-example-compliance-in-topology-optimization)
- [Discrete vs continuous adjoint](#discrete-vs-continuous-adjoint)
- [Time-dependent and nonlinear problems](#time-dependent-and-nonlinear-problems)
- [Where it is used](#where-it-is-used)
- [How it relates to machine learning](#how-it-relates-to-machine-learning)
- [Code: adjoint gradients for 400 design variables](#code-adjoint-gradients-for-400-design-variables)
- [In an interview](#in-an-interview)

## Derivation

Minimize J(u, p) over design variables p ∈ ℝᵐ, where the state u ∈ ℝⁿ satisfies the discretized governing equations R(u, p) = 0 (for a linear problem, R = K(p)u − f(p)).

1. **Chain rule:**

```math
\frac{dJ}{dp} = \frac{\partial J}{\partial p} + \frac{\partial J}{\partial u}\,\frac{du}{dp}
```

2. **Differentiate the constraint.** R(u(p), p) = 0 for all p, so

```math
\frac{\partial R}{\partial u}\,\frac{du}{dp} = -\frac{\partial R}{\partial p}
```

The matrix du/dp is n × m: computing it means solving with the Jacobian ∂R/∂u once **per design variable**.

3. **Avoid du/dp.** Substitute du/dp = −(∂R/∂u)⁻¹ ∂R/∂p into the chain rule and group the first factors:

```math
\frac{dJ}{dp} = \frac{\partial J}{\partial p} - \underbrace{\frac{\partial J}{\partial u}\Big(\frac{\partial R}{\partial u}\Big)^{-1}}_{\lambda^\top}\frac{\partial R}{\partial p},
\qquad \Big(\frac{\partial R}{\partial u}\Big)^{\!\top}\lambda = \Big(\frac{\partial J}{\partial u}\Big)^{\!\top}
```

4. **One adjoint solve** gives λ, whatever m is; then each derivative is a cheap product −λᵀ ∂R/∂p_i, usually assembled element by element.

**Lagrangian view.** The same result comes from the Lagrangian L = J + λᵀR: choosing λ so that ∂L/∂u = 0 removes the expensive du/dp term, and dJ/dp = ∂L/∂p.

**Intuition.** λ is the sensitivity of the objective to a perturbation (a source term) in each equation: it says how much a small extra load at each node would change J. For J = temperature at a point, λ is the Green's function of that point.

## Direct vs adjoint sensitivities

| | Direct (forward) sensitivity | Adjoint (reverse) sensitivity |
| --- | --- | --- |
| Extra solves | One per design variable | One per objective or constraint |
| Best when | Few inputs, many outputs (a few parameters, the whole field's sensitivity) | Many inputs, few outputs (shape or topology optimization) |
| Analogue in automatic differentiation | Forward mode | Reverse mode (backpropagation) |
| Memory | Low | Needs the forward state (and the whole history for transient problems) |

Each objective and each constraint needs its own adjoint solve; problems with thousands of local constraints (stress at every element) are expensive, which is why they are aggregated (p-norm or Kreisselmeier-Steinhauser functions).

## Worked example: compliance in topology optimization

Linear elasticity: K(ρ)u = F, compliance J = Fᵀu (the work done by the load; minimizing it maximizes stiffness).

- ∂J/∂u = Fᵀ, so the adjoint equation is Kᵀλ = F. K is symmetric, so **λ = u**: the problem is **self-adjoint** and needs no extra solve.
- With R = Ku − F and K = Σₑ Eₑ(ρₑ) k₀ (assembled), ∂R/∂ρₑ = (∂K/∂ρₑ)u, so dJ/dρₑ = −uᵀ(∂K/∂ρₑ)u = −(dEₑ/dρₑ) uₑᵀk₀uₑ.
- With SIMP, Eₑ = E_min + ρₑᵖ(E₀ − E_min), giving dJ/dρₑ = −p ρₑ^{p−1}(E₀ − E_min) uₑᵀk₀uₑ: negative everywhere (more material is always stiffer) and largest where the strain energy is largest. See [Topology optimization](topology-optimization.md).

## Discrete vs continuous adjoint

- **Discrete adjoint (discretize, then differentiate):** differentiate the discretized equations exactly as implemented. The gradient is the exact gradient of the objective the code computes, consistent to round-off, which is what optimizers need; automatic differentiation produces it automatically. Implementing it by hand for a large solver (including turbulence models, limiters and mesh deformation) is laborious.
- **Continuous adjoint (derive an adjoint PDE, then discretize):** gives physical insight (adjoint boundary conditions, adjoint fields that show where the objective is sensitive) and can be cheaper to implement for some equations, but the gradient is only consistent with the discrete objective as the mesh is refined, and inconsistencies (for example around non-differentiable limiters or turbulence models frozen in the derivation) can mislead optimizers.
- Many CFD codes offer one or both: SU2 provides continuous and discrete adjoints; OpenFOAM has adjoint solvers; commercial codes such as Fluent and STAR-CCM+ offer adjoint shape sensitivities.

## Time-dependent and nonlinear problems

- **Nonlinear steady problems:** the adjoint equation is linear, built from the Jacobian ∂R/∂u at the converged solution, so it costs roughly one Newton iteration's linear solve (often less than the nonlinear solve itself).
- **Time-dependent problems:** the adjoint runs **backwards in time** from the final time, and at each step needs the forward state at that time. Storing every state can be impossible for long runs; **checkpointing** stores selected states and recomputes the rest, trading compute for memory (the revolve algorithm is optimal for a given memory budget). This is the same idea as gradient checkpointing in deep learning. See [Gradient checkpointing](scaling-to-large-meshes.md#gradient-checkpointing).
- **Chaotic problems:** gradients of long-time averages of chaotic systems computed by standard adjoints blow up exponentially with the time horizon; shadowing methods (least-squares shadowing) or short windows are needed.

## Where it is used

- **Aerodynamic shape optimization:** sensitivity of drag or lift to every surface node; mesh morphing turns the surface sensitivities into shape updates.
- **Topology optimization:** the self-adjoint compliance gradient for every element density, every iteration.
- **Inverse problems:** identifying material properties or loads from measurements, seismic full-waveform inversion, medical imaging. See [Inverse problems and regularization](inverse-problems-and-regularization.md).
- **Data assimilation in weather forecasting:** 4D-Var adjusts the initial state of a forecast model to fit observations over a time window, using the adjoint of the model.
- **Mesh adaptation and error estimation:** adjoint-weighted residuals tell where mesh refinement most improves a quantity of interest (goal-oriented adaptivity).
- **Uncertainty propagation:** first-order sensitivity of outputs to many uncertain inputs.

## How it relates to machine learning

- **Backpropagation is the adjoint method** applied to the computational graph of a network: one backward pass gives the gradient of a scalar loss with respect to millions of weights.
- **Differentiable solvers** written in JAX or PyTorch give discrete adjoints automatically. See [Differentiable simulators](differentiable-physics-and-hybrid-solvers.md#differentiable-simulators).
- **Neural ODEs** are trained with a continuous adjoint ODE. See [Neural ODEs](neural-odes.md#training-backpropagation-through-the-solver-and-the-adjoint-method).
- **Surrogate gradients vs adjoint gradients:** a differentiable surrogate gives gradients of an approximation for almost no cost; the adjoint gives exact gradients for about one solve. Use the surrogate to explore and the adjoint (or finite differences at a few points) to check and refine. See [Surrogate gradients vs the adjoint](surrogate-based-optimization.md#surrogate-gradients-vs-the-adjoint).
- **Surrogates trained with adjoint data:** gradients from adjoint solves are extra training targets (Sobolev training, gradient-enhanced kriging) that make small datasets much more informative.

## Code: adjoint gradients for 400 design variables

A 1D steady heat-conduction problem, −(k(x)T′)′ = 1 with T = 0 at both ends, discretized with 400 linear finite elements; each element's conductivity is a design variable, and the objective is the mean temperature. The adjoint gradient (one extra solve) is compared with finite differences (400 extra solves) and with reverse-mode automatic differentiation of the same discrete model in PyTorch.

```python
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# 1D steady heat conduction -(k(x) T')' = q on (0, 1), T(0) = T(1) = 0, linear finite elements.
# Design variables: one conductivity per element (m = 400). Objective J = mean temperature (to be minimized).
m = 400
h = 1.0 / m
q = np.ones(m + 1) * h; q[0] = q[-1] = 0.5 * h                    # consistent load for a uniform source

def assemble(k):
    """Stiffness matrix on the interior nodes 1..m-1 (the Dirichlet nodes T_0 = T_m = 0 are eliminated)."""
    main = (k[:-1] + k[1:]) / h                                     # node i gets k_{i-1}/h + k_i/h
    off = -k[1:-1] / h
    return sp.diags([off, main, off], [-1, 0, 1], format="csc")

def solve(k):
    return np.concatenate([[0.0], spla.spsolve(assemble(k), q[1:m]), [0.0]])

def objective(T):
    return T.mean()

rng = np.random.default_rng(0)
k = 1.0 + 0.5 * rng.random(m)                                      # current design

# Adjoint gradient: one forward solve plus one adjoint solve, for all m derivatives at once.
t0 = time.perf_counter()
T = solve(k)
K = assemble(k)
dJdT = np.full(m - 1, 1.0 / (m + 1))                               # dJ/dT on the interior nodes
lam = np.concatenate([[0.0], spla.spsolve(K.T.tocsc(), dJdT), [0.0]])   # adjoint equation K^T lambda = dJ/dT
dT = np.diff(T)                                                    # T_{e+1} - T_e per element
dlam = np.diff(lam)
grad_adjoint = -(dlam * dT) / h                                    # dJ/dk_e = -lambda^T (dK/dk_e) T
t_adj = time.perf_counter() - t0

# Finite differences: one extra solve per design variable.
t0 = time.perf_counter()
eps = 1e-6
J0 = objective(T)
grad_fd = np.empty(m)
for e in range(m):
    kp = k.copy(); kp[e] += eps
    grad_fd[e] = (objective(solve(kp)) - J0) / eps
t_fd = time.perf_counter() - t0
print(f"max relative difference adjoint vs finite differences: {np.max(np.abs(grad_adjoint - grad_fd)) / np.max(np.abs(grad_fd)):.1e}")
print(f"time for all {m} derivatives: adjoint {1e3 * t_adj:.1f} ms, finite differences {1e3 * t_fd:.0f} ms")

# Reverse-mode automatic differentiation of the same discrete model gives the same gradient: it is a discrete adjoint.
import torch
kt = torch.tensor(k, dtype=torch.float64, requires_grad=True)
idx = torch.arange(m)
Kd = torch.zeros(m + 1, m + 1, dtype=torch.float64)
Kd = Kd.index_put((idx, idx), kt / h, accumulate=True).index_put((idx + 1, idx + 1), kt / h, accumulate=True)
Kd = Kd.index_put((idx, idx + 1), -kt / h, accumulate=True).index_put((idx + 1, idx), -kt / h, accumulate=True)
inner = slice(1, m)                                                # eliminate the Dirichlet nodes
Tin = torch.linalg.solve(Kd[inner, inner], torch.tensor(q[1:m]))
J = torch.cat([torch.zeros(1, dtype=torch.float64), Tin, torch.zeros(1, dtype=torch.float64)]).mean()
J.backward()
print(f"max relative difference adjoint vs reverse-mode autodiff: {np.max(np.abs(grad_adjoint - kt.grad.numpy())) / np.max(np.abs(grad_adjoint)):.1e}")
```

Output:

```
max relative difference adjoint vs finite differences: 3.0e-05
time for all 400 derivatives: adjoint 1.8 ms, finite differences 142 ms
max relative difference adjoint vs reverse-mode autodiff: 2.4e-12
```

The adjoint agrees with finite differences to the accuracy of the finite differences themselves (their truncation error with a 10⁻⁶ step), and with reverse-mode automatic differentiation to machine precision, which confirms that autodiff of a solver is a discrete adjoint. The cost gap, here a factor of roughly 50-100 depending on the machine, grows linearly with the number of design variables: for a million-variable shape optimization, finite differences are impossible and the adjoint still costs one solve.

## In an interview

**Thirty-second answer to "Explain the adjoint method."** "To get the gradient of one objective with respect to many design variables subject to a PDE, you differentiate the discretized equations and, instead of computing how the state changes with every variable, you solve one extra linear system with the transposed Jacobian, the adjoint equation. Then every derivative is a cheap product of the adjoint with the derivative of the residual. So the cost is one forward solve plus one adjoint solve whatever the number of variables, against one solve per variable for finite differences. Backpropagation is the same idea on a network's computational graph, and compliance in topology optimization is self-adjoint, so the adjoint is the displacement itself."

**Likely follow-ups**

- *When is the direct method better?* Few inputs and many outputs.
- *Discrete or continuous adjoint?* Discrete for gradients consistent with the code (what optimizers need); continuous for insight and sometimes easier implementation.
- *What is hard about transient adjoints?* They run backwards in time and need the forward history: checkpointing trades recomputation for memory.

## References

- [Physics-based Deep Learning (Thuerey et al., free online book): differentiable physics and gradients through solvers](https://physicsbaseddeeplearning.org/)
- M. B. Giles and N. A. Pierce, "An Introduction to the Adjoint Approach to Design", *Flow, Turbulence and Combustion*, 2000.
- [SU2 (open-source CFD with continuous and discrete adjoints)](https://github.com/su2code/SU2)
- [dolfin-adjoint (automatic adjoints for FEniCS)](https://www.dolfin-adjoint.org/)
- [Least Squares Shadowing sensitivity analysis of chaotic limit cycle oscillations (Wang, Hu & Blonigan)](https://arxiv.org/abs/1204.0159)
- A. Griewank and A. Walther, *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*, SIAM, 2008.
