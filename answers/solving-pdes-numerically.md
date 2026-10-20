# Solving a PDE Numerically

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A partial differential equation (PDE) describes how a field, such as temperature, displacement or velocity, varies in space and time. Except for simple shapes, it cannot be solved with a formula. "Solving it numerically" means replacing the continuous problem with a finite one: represent the field by a finite set of numbers on a mesh (values at nodes, cell averages or coefficients of basis functions), turn the PDE and its boundary conditions into a system of algebraic equations for those numbers, solve that system on a computer, and check that the answer converges as the mesh and time step are refined. The result is an *approximation* whose error you must estimate.

**On this page**

- [What a PDE problem is](#what-a-pde-problem-is)
- [The five steps of a numerical solution](#the-five-steps-of-a-numerical-solution)
- [A worked example: heat in a rod](#a-worked-example-heat-in-a-rod)
- [Consistency, stability and convergence](#consistency-stability-and-convergence)
- [Code: a finite-difference solver and its convergence](#code-a-finite-difference-solver-and-its-convergence)
- [What changes when machine learning solves a PDE](#what-changes-when-machine-learning-solves-a-pde)
- [In an interview](#in-an-interview)

## What a PDE problem is

A PDE relates a field to its rates of change in space and time. Three classic examples cover most of engineering:

| Equation | Form | Physics | Type |
| --- | --- | --- | --- |
| Poisson / Laplace | −∇·(k∇u) = f | Steady heat conduction, electrostatics, pressure | Elliptic: every point depends on the whole domain |
| Heat (diffusion) | ∂u/∂t = α∇²u | Transient conduction, diffusion | Parabolic: smooths out over time |
| Wave / advection | ∂²u/∂t² = c²∇²u, or ∂u/∂t + a·∇u = 0 | Vibrations, acoustics, transport | Hyperbolic: information travels at finite speed |

Navier–Stokes and the equations of elasticity are systems that combine these behaviours.

A PDE on its own is not a problem you can solve. A complete problem also needs:

- **a domain** (the geometry);
- **boundary conditions** (what happens at the edges: fixed temperature, applied heat flux, fixed support, applied load); see [Boundary conditions](boundary-conditions.md);
- **initial conditions** for time-dependent problems;
- **material properties and sources** (conductivity, viscosity, body forces).

A problem is *well-posed* (in Hadamard's sense) if a solution exists, is unique and depends continuously on the data: a small change in the inputs gives a small change in the answer. Numerical methods are designed for well-posed problems. Inverse problems, such as recovering a material property from a few measurements, are often *ill-posed*, which is why they need regularization. See [Inverse problems](inverse-problems-and-regularization.md).

## The five steps of a numerical solution

1. **Discretize the domain.** Build a mesh or grid: points, cells or elements that cover the geometry. See [Meshing and mesh quality](meshing-and-mesh-quality.md).
2. **Choose a discrete representation and discretize the equations.** The main families are:
   - *finite differences:* replace derivatives with differences of nodal values;
   - *finite volumes:* enforce conservation (flux in minus flux out) on each cell;
   - *finite elements:* write the solution as a sum of basis functions and require the weak form of the PDE to hold.

   See [FEM vs FVM vs FDM](fem-vs-fvm-vs-fdm.md).
3. **Discretize time** (for transient problems): step forward with an explicit or implicit scheme. See [Time integration and stability](time-integration-and-stability.md).
4. **Solve the algebraic system.** Linear problems give a large, sparse system **A u = b**, solved with sparse direct or iterative solvers ([Sparse linear solvers](sparse-linear-solvers.md)). Nonlinear problems are solved with Newton's method, which needs a linear solve at every iteration ([Newton-Raphson](newton-raphson-and-nonlinear-solvers.md)).
5. **Post-process and check the error.** Compute the quantities you need (heat flux, stress, drag) and check that they converge as the mesh and time step are refined. See [Verification and validation](verification-validation-and-mesh-convergence.md).

The output is always an approximation. A numerical answer without an error estimate is incomplete.

## A worked example: heat in a rod

A rod of length 1 has both ends held at temperature 0 and is heated along its length with a source f(x). In steady state the temperature u(x) satisfies

```math
-u''(x) = f(x), \qquad 0 < x < 1, \qquad u(0) = u(1) = 0 .
```

**Step 1: grid.** Choose n interior points spaced h = 1/(n + 1) apart: x₁, x₂, …, xₙ. The unknowns are the values u₁, …, uₙ.

**Step 2: discretize.** The second derivative at point i is approximated by the central difference

```math
u''(x_i) \approx \frac{u_{i-1} - 2u_i + u_{i+1}}{h^2} ,
```

which is accurate to second order: its error is proportional to h². Substituting it into the PDE at every interior point gives one equation per unknown:

```math
\frac{-u_{i-1} + 2u_i - u_{i+1}}{h^2} = f(x_i), \qquad i = 1, \dots, n,
```

with u₀ = uₙ₊₁ = 0 from the boundary conditions.

**Step 3: there is no time here.** The problem is steady.

**Step 4: solve.** The equations form a linear system **A u = f**. A is *tridiagonal*: each row has only three non-zero entries, because each equation links a point to its two neighbours. For n = 4:

```
         ┌  2 -1  0  0 ┐ ┌u1┐   ┌f1┐
  1/h² · │ -1  2 -1  0 │ │u2│ = │f2│
         │  0 -1  2 -1 │ │u3│   │f3│
         └  0  0 -1  2 ┘ └u4┘   └f4┘
```

In 2D and 3D, each row still has only a handful of non-zeros, which is why simulation matrices are **sparse** and why sparse solvers matter.

**Step 5: check.** Pick f(x) = π² sin(πx), for which the exact answer is u = sin(πx), and measure the error on finer and finer grids (code below). Each time h is halved, the error falls by a factor of about 4: the expected second-order convergence.

## Consistency, stability and convergence

Three words appear in every numerical methods course and in many interviews:

- **Consistency:** the discrete equations approach the PDE as h and Δt go to zero. The difference between them, evaluated on the exact solution, is the *truncation error*. The central difference above has truncation error of order h².
- **Stability:** errors (from round-off, or from the approximation itself) do not grow without bound as the computation proceeds. An explicit time-stepping scheme with too large a time step is the classic unstable case. See [The CFL condition](time-integration-and-stability.md#the-cfl-condition).
- **Convergence:** the numerical solution approaches the exact solution as h and Δt go to zero.

For linear problems, the **Lax equivalence theorem** links them: a consistent scheme converges if and only if it is stable. In practice, consistency is easy to arrange and **stability is what you have to work for**.

The **order of accuracy** p says how fast the error falls: error ≈ C·hᵖ. A second-order method gains a factor of 4 in accuracy each time the mesh spacing is halved; a first-order method only a factor of 2. In 3D, halving h multiplies the number of unknowns by 8, so the order of accuracy decides how expensive accuracy is.

## Code: a finite-difference solver and its convergence

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def solve_poisson_fd(n, f):
    """Solve -u'' = f on (0, 1), u(0) = u(1) = 0, with n interior points."""
    h = 1.0 / (n + 1)
    x = np.linspace(h, 1 - h, n)                       # interior grid points
    main = 2.0 * np.ones(n)
    off = -1.0 * np.ones(n - 1)
    A = sp.diags([off, main, off], [-1, 0, 1], format="csr") / h**2
    u = spla.spsolve(A, f(x))                          # the algebraic system A u = f
    return x, u

f = lambda x: np.pi**2 * np.sin(np.pi * x)            # exact solution: u = sin(pi x)
for n in [9, 19, 39, 79]:
    x, u = solve_poisson_fd(n, f)
    err = np.max(np.abs(u - np.sin(np.pi * x)))
    print(f"h = 1/{n + 1:<3d} max error = {err:.2e}")
```

Output when run:

```
h = 1/10  max error = 8.27e-03
h = 1/20  max error = 2.06e-03
h = 1/40  max error = 5.14e-04
h = 1/80  max error = 1.29e-04
```

The error falls by 4 each time h is halved, confirming second-order accuracy. This kind of check, against a known solution, is the simplest form of *code verification*.

## What changes when machine learning solves a PDE

A classical solver solves **one instance** of the problem: one geometry, one set of boundary conditions, one set of material properties. Change any of them and you solve again.

Machine learning approaches the problem differently:

- A **PINN** also solves one instance, but represents the solution with a neural network and minimizes the PDE residual at sampled points instead of building a mesh. See [PINNs](pinns.md).
- A **neural operator** or **neural surrogate** learns the *solution operator*: the map from inputs (geometry, coefficients, boundary conditions) to solutions, from many solved examples. Once trained, it predicts the solution for a new instance in one forward pass. See [Neural operators](neural-operators.md).

What is lost is the guarantee. A classical method has a convergence theory: refine the mesh and the error falls at a known rate. A trained network's error can only be measured on test data, and it can be large on inputs unlike the training set. That is why the classical ideas on this page (error estimation, convergence, stability) remain the reference that any ML method is judged against.

## In an interview

**Thirty-second answer.** "We replace the continuous field with a finite set of unknowns on a mesh, discretize the PDE with finite differences, volumes or elements so it becomes a large sparse algebraic system, step in time if the problem is transient, solve the system with sparse linear or Newton solvers, and then check that the quantities we care about converge as the mesh and time step are refined. The result is an approximation, so it needs an error estimate."

**Likely follow-ups**

- *What makes a scheme stable?* For explicit time stepping, a small enough time step relative to the mesh size (the CFL condition for waves and advection; αΔt/Δx² ≤ 1/2 for the explicit heat equation). See [Time integration and stability](time-integration-and-stability.md).
- *Why is the matrix sparse?* Each unknown interacts only with its neighbours on the mesh.
- *What does a neural operator learn that a solver does not?* The map from problem inputs to solutions across many instances, so it can answer new instances without solving again.

## References

- [ETH Zürich AISE: Introduction to Operator Learning Part 1 (video; includes "What does it mean to solve a PDE?")](https://www.youtube.com/watch?v=yhHhMmiNl_g)
- [ETH Zürich AISE: Importance of PDEs in Science (video)](https://www.youtube.com/watch?v=UiZxDRBd0Q8)
- R. J. LeVeque, *Finite Difference Methods for Ordinary and Partial Differential Equations*, SIAM, 2007.
- [The FEniCSx tutorial (Jørgen S. Dokken)](https://jsdokken.com/dolfinx-tutorial/)
