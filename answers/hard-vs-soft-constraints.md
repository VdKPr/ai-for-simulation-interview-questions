# Hard vs Soft Constraints

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A **soft constraint** adds a penalty to the loss for violating a physical law or condition: a PDE residual, a boundary misfit, a divergence or conservation error. It is easy to add and flexible, but it is only approximately satisfied, and it adds loss weights that must be balanced. A **hard constraint** builds the property into the structure of the model, so it holds exactly for every prediction, whatever the weights: an output ansatz that matches the boundary values by construction, a velocity defined as the curl of a potential so its divergence is identically zero, finite-volume-style flux updates that conserve exactly. Hard constraints restrict what the model can represent and can complicate optimization, but they remove a whole class of errors and often train faster. Enforce exactly the properties that the customer's decision depends on (mass conservation, boundary conditions, equilibrium) when the structure allows it, penalize the rest, and measure violations either way.

**On this page**

- [Soft and hard, side by side](#soft-and-hard-side-by-side)
- [Ways to make a constraint hard](#ways-to-make-a-constraint-hard)
- [Enforcing a divergence-free field exactly](#enforcing-a-divergence-free-field-exactly)
- [Soft vs hard boundary conditions in PINNs](#soft-vs-hard-boundary-conditions-in-pinns)
- [Code: a divergence-free network and a hard boundary condition](#code-a-divergence-free-network-and-a-hard-boundary-condition)
- [Trade-offs](#trade-offs)
- [In an interview](#in-an-interview)

## Soft and hard, side by side

| | Soft constraint | Hard constraint |
| --- | --- | --- |
| How | Penalty term in the loss | Built into the architecture or the output transformation |
| Satisfied | Approximately; violations collect where the penalty is weak or points are sparse | Exactly (to floating-point precision), everywhere, for every input |
| Tuning | Penalty weight must be balanced against other terms | None for that constraint |
| Expressiveness | Unrestricted model | Restricted to functions that satisfy the constraint |
| Implementation | Easy | Needs a suitable construction (distance functions, potentials, conservative updates) |
| At inference | Can be violated on new inputs, especially out of distribution | Still holds on any input |

The last row is easy to overlook: a soft penalty only shapes the model on the training distribution. On a new design, nothing stops the prediction from violating mass conservation badly. A hard constraint holds even there.

## Ways to make a constraint hard

| Constraint | Construction |
| --- | --- |
| Dirichlet boundary condition | Output ansatz u(x) = g(x) + d(x)·N(x), where g matches the boundary values and d vanishes on the boundary (an approximate distance function), so u = g there for any network N |
| Neumann or periodic conditions | Special constructions (for example periodic input features sin(2πx/L), cos(2πx/L) for periodicity) |
| Divergence-free velocity, 2D | Predict a stream function ψ; set u = ∂ψ/∂y, v = −∂ψ/∂x |
| Divergence-free field, 3D | Predict a vector potential A; set u = ∇ × A (the divergence of a curl is identically zero) |
| Local conservation | Predict fluxes through cell faces and update cells finite-volume style, so what leaves one cell enters its neighbour exactly |
| Positivity (density, turbulent viscosity, concentrations) | Softplus or exponential output |
| Bounds (a volume fraction in [0, 1]) | Sigmoid output |
| Symmetry and equivariance | Equivariant architectures or symmetric output constructions ([Equivariance and symmetry](equivariance-and-symmetry.md)) |
| Energy-conserving dynamics | Learn a Hamiltonian (or Lagrangian) and derive the dynamics from it ([Hamiltonian and Lagrangian networks](hamiltonian-and-lagrangian-neural-networks.md)) |
| Linear constraints in general | Post-hoc projection of the prediction onto the constraint set (for example a Helmholtz projection that removes the divergent part of a velocity field) |

## Enforcing a divergence-free field exactly

Incompressible flow satisfies ∇·u = 0 everywhere. A network that outputs velocity directly will generally not: in the code below, an untrained direct-velocity network has divergence of order 0.1, and training with a penalty only reduces it approximately, on the training distribution.

**Exact constructions:**

1. **Stream function (2D).** Predict a scalar ψ(x, y) and define u = ∂ψ/∂y, v = −∂ψ/∂x. Then ∇·u = ∂²ψ/∂x∂y − ∂²ψ/∂y∂x = 0 identically, because mixed partial derivatives commute. Derivatives come from automatic differentiation, so the property holds to floating-point precision (about 10⁻⁸ in single precision in the code below).
2. **Vector potential (3D).** Predict a vector field A and define u = ∇ × A. The divergence of a curl is identically zero.
3. **Projection.** Predict any velocity field, then project it onto divergence-free fields by solving a Poisson equation for a correction potential φ, ∇²φ = ∇·u, and setting u ← u − ∇φ. This is exactly the pressure-projection step of incompressible solvers. See [Pressure-velocity coupling](cfd-essentials.md#pressure-velocity-coupling).
4. **Discrete conservation.** On a mesh, predict face fluxes and require that they sum to zero in each cell (or use a discrete curl on a staggered grid), which gives exact discrete conservation.

**An industrial-scale example.** AB-UPT, a transformer-based surrogate for automotive CFD on meshes from 33,000 to 150 million cells, decodes its outputs with anchored neural fields that can be differentiated with respect to position. The authors use this to build a divergence-free formulation (for example for vorticity, which is divergence-free by definition) and report enforcing the hard constraint without degrading accuracy. See [AB-UPT](physics-transformer-architectures.md#ab-upt-anchor-tokens-and-divergence-free-fields).

**Why it matters:** divergence errors mean mass is created or destroyed; integrated quantities (mass flow through an outlet, a flow split between two branches) then do not add up, and downstream tools that assume a conservative field (particle tracking, coupled heat transfer) misbehave.

## Soft vs hard boundary conditions in PINNs

**Soft:** add λ_b · mean |u_θ(x_b) − g(x_b)|² to the loss, at sampled boundary points.

- Easy for any boundary condition type (Dirichlet, Neumann, Robin) and any geometry.
- Only approximately satisfied, between and even at the sampled points.
- Introduces the weight λ_b, whose balance with the residual is a main cause of PINN failure. See [Loss imbalance and adaptive weights](pinn-training-failures.md#loss-imbalance-and-adaptive-weights).

**Hard:** write the network output so that the condition holds for any weights.

- **Dirichlet:** u(x) = g(x) + d(x)·N(x). On a simple domain, d can be a product like x(1 − x); on a complex domain, an approximate distance function (from a signed distance field, R-functions, or a separately trained network) is needed. Lu et al. (2021) used such hard constraints for inverse design with PINNs.
- **Neumann:** harder; possible with specific constructions, or handled by a mixed formulation that outputs the flux as a separate variable.
- Removes a loss term and its weight; often trains faster and more accurately.
- The construction must be smooth enough for the PDE's derivatives, and building good distance functions for complex CAD geometry is itself work.
- A poorly chosen d(x) can distort the solution near the boundary (for example making derivatives there hard to represent).

**A common compromise:** hard-enforce Dirichlet conditions on simple boundaries where a distance function is easy, and keep soft penalties for Neumann conditions or complex boundaries, with adaptive weights.

## Code: a divergence-free network and a hard boundary condition

```python
import torch, torch.nn as nn

torch.manual_seed(0)
mlp = lambda n_out: nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, n_out))

def divergence(velocity_fn, xy):
    xy = xy.clone().requires_grad_(True)
    uv = velocity_fn(xy)
    du_dx = torch.autograd.grad(uv[:, 0].sum(), xy, create_graph=True)[0][:, 0]
    dv_dy = torch.autograd.grad(uv[:, 1].sum(), xy, create_graph=True)[0][:, 1]
    return du_dx + dv_dy

psi_net = mlp(1)                                     # hard: predict a stream function psi(x, y)
def velocity_from_stream_function(xy):
    psi = psi_net(xy)
    grad = torch.autograd.grad(psi.sum(), xy, create_graph=True)[0]
    return torch.stack([grad[:, 1], -grad[:, 0]], dim=1)      # u = dpsi/dy, v = -dpsi/dx

uv_net = mlp(2)                                      # unconstrained: predict (u, v) directly

xy = torch.rand(1000, 2, dtype=torch.float32)
div_hard = divergence(velocity_from_stream_function, xy).abs().max().item()
div_soft = divergence(uv_net, xy).abs().max().item()
print(f"max |div u|, stream-function network (untrained): {div_hard:.1e}")
print(f"max |div u|, direct velocity network (untrained): {div_soft:.1e}")

# Hard Dirichlet condition: u(x) = g(x) + d(x) N(x), with d = 0 on the boundary
N = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 1))
u_hard = lambda x: 2.0 + x * (1 - x) * N(x)          # u(0) = u(1) = 2 for ANY weights of N
print("hard-constrained boundary values:", u_hard(torch.tensor([[0.0], [1.0]])).flatten().tolist())
```

Output:

```
max |div u|, stream-function network (untrained): 5.2e-08
max |div u|, direct velocity network (untrained): 1.6e-01
hard-constrained boundary values: [2.0, 2.0]
```

Both properties hold before any training, because they come from the structure, not from the weights.

## Trade-offs

- **Exactness vs flexibility.** A hard constraint can make the model unable to represent the true solution if the construction is too restrictive or the constraint is only approximately true in the data (for example, labels from a solver that conserves mass only to its tolerance).
- **Optimization.** Sometimes harder (higher derivatives in the construction, as with stream functions), sometimes easier (a smaller search space and one fewer loss term).
- **Cost.** Derivative-based constructions need extra autograd passes; projections need a Poisson solve.
- **Complex geometry.** Distance functions and potentials on real CAD are not trivial.
- **What to enforce:** prioritize properties that downstream decisions depend on or that errors would visibly break: mass conservation for flow splits, boundary conditions at walls, equilibrium for reaction forces.

## In an interview

**Thirty-second answer to "Soft vs hard enforcement of boundary conditions?"** "Soft means a penalty in the loss: easy for any boundary type, but only approximate and it adds a weight that must be balanced against the residual, which is a classic PINN failure. Hard means building the condition into the output, for example u = g + d·N with d zero on the boundary, so it holds exactly for any weights and the loss term disappears. Hard is better for Dirichlet conditions on geometry where a distance function is easy; for Neumann conditions or complex CAD, soft penalties with adaptive weights are often more practical."

**Thirty-second answer to "How can a model enforce a divergence-free field exactly?"** "Predict a potential and differentiate it: a stream function in 2D, with u = ∂ψ/∂y and v = −∂ψ/∂x, or a vector potential in 3D, with u = curl A, since the divergence of a curl is identically zero. Alternatively project the prediction onto divergence-free fields with a Poisson solve, as incompressible solvers do, or predict conservative face fluxes on the mesh. AB-UPT does this at industrial scale for automotive CFD."

**Likely follow-ups**

- *Why not just penalize the divergence?* It is only approximate, it adds a weight to tune, and it can be badly violated on out-of-distribution inputs.
- *Downside of hard constraints?* Restricted expressiveness, extra derivatives, and constructions that are hard on complex geometry.
- *How do you check constraints either way?* Compute divergence, mass balance, boundary errors and force balance on test predictions. See [Physics-consistency checks](evaluation-metrics.md#physics-consistency-checks).

## References

- [Physics-informed neural networks with hard constraints for inverse design (Lu et al., 2021)](https://arxiv.org/abs/2102.04626)
- [AB-UPT: Scaling Neural CFD Surrogates for High-Fidelity Automotive Aerodynamics Simulations via Anchored-Branched Universal Physics Transformers (Alkin et al., 2025)](https://arxiv.org/abs/2502.09692)
- [Hamiltonian Neural Networks (Greydanus et al.)](https://arxiv.org/abs/1906.01563) and [Lagrangian Neural Networks (Cranmer et al.)](https://arxiv.org/abs/2003.04630)
- [AI/ML+Physics, Part 3: Designing an Architecture (Steve Brunton, video playlist)](https://www.youtube.com/playlist?list=PLMrJAkhIeNNQ0BaKuBKY43k4xMo6NSbBa)
