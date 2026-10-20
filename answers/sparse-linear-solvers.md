# Sparse Linear Solvers

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Discretizing a PDE gives a large sparse system **A x = b**, often with millions of unknowns, and solving it is usually the most expensive part of a simulation. **Direct solvers** factorize A (LU, or Cholesky for symmetric positive-definite matrices) and are robust and exact up to round-off, but the factors fill in, so memory and time grow quickly in 3D. **Iterative solvers** improve a guess using only matrix-vector products: **conjugate gradient (CG)** for symmetric positive-definite systems such as elasticity and conduction, **GMRES** or **BiCGStab** for non-symmetric systems such as convection. Their iteration count grows with the condition number, which grows as the mesh is refined, so they need **preconditioners**. **Multigrid** fixes the error at every scale by working on a hierarchy of coarser grids; as a solver or as a preconditioner for CG, it can solve elliptic problems in nearly O(N) work, with an iteration count that hardly grows with mesh size.

**On this page**

- [Where these systems come from](#where-these-systems-come-from)
- [Direct solvers](#direct-solvers)
- [Iterative solvers and conjugate gradient](#iterative-solvers-and-conjugate-gradient)
- [Condition number and preconditioning](#condition-number-and-preconditioning)
- [Multigrid](#multigrid)
- [Code: CG with and without multigrid](#code-cg-with-and-without-multigrid)
- [How to choose](#how-to-choose)
- [Why this matters for machine learning](#why-this-matters-for-machine-learning)
- [In an interview](#in-an-interview)

## Where these systems come from

- Linear static FEA: **K u = f**, with K symmetric positive definite (SPD) after supports are applied. See [The finite element method](finite-element-method.md#assembly-and-sparsity).
- Implicit time steps and Newton iterations: one linear system per iteration. See [Newton-Raphson](newton-raphson-and-nonlinear-solvers.md).
- CFD: the pressure equation of an incompressible solver (SPD-like, solved many times per run) and the momentum equations (non-symmetric because of convection).

Two facts drive everything: the matrices are **sparse** (a few dozen non-zeros per row), and they are **large** (10⁵ to 10⁹ unknowns).

## Direct solvers

**How they work.** Factorize A = LU (or A = LLᵀ, Cholesky, for SPD matrices), then solve two triangular systems. Production codes (MUMPS, PARDISO, SuperLU, CHOLMOD and the solvers inside commercial FEA packages) work on the sparse structure and reorder the unknowns first.

**The catch: fill-in.** The factors L and U have non-zeros where A has zeros. For a 2D mesh with N unknowns, a good ordering (nested dissection) gives factors with about N log N non-zeros and costs about N^1.5 operations. For 3D meshes it is much worse: about N^(4/3) memory and N² operations. That is why direct solvers dominate in 2D and for moderate 3D models, and run out of memory for very large 3D models.

**Strengths:**

- robust: no convergence to worry about, works for ill-conditioned and indefinite systems;
- once A is factorized, each extra right-hand side (another load case) is cheap;
- exact up to round-off.

Commercial structural codes often default to sparse direct solvers for these reasons, switching to iterative solvers for very large models.

## Iterative solvers and conjugate gradient

An iterative solver starts from a guess and improves it, using only products A·v. Memory stays close to the size of A itself.

**Conjugate gradient** is the method for SPD systems. Solving A x = b is equivalent to minimizing the energy φ(x) = ½xᵀAx − bᵀx (the same energy as in [energy minimization](weak-form-and-galerkin.md#energy-minimization-the-same-thing-seen-differently)). CG minimizes it over a growing sequence of search directions that are mutually *A-conjugate*, so no direction is ever undone. In exact arithmetic it would finish in N steps; in practice it is stopped much earlier, when the residual is small enough.

Each iteration costs one sparse matrix-vector product and a few vector operations. The algorithm fits in a few lines:

```
r = b − A x;  p = r
repeat:
    α = (rᵀr) / (pᵀ A p)
    x = x + α p
    r_new = r − α A p
    if ‖r_new‖ small enough: stop
    β = (r_newᵀ r_new) / (rᵀ r)
    p = r_new + β p;  r = r_new
```

**Non-symmetric systems** (convection in CFD) use **GMRES** (minimizes the residual over a Krylov subspace; needs restarts to bound memory) or **BiCGStab** (short recurrences, less memory, less smooth convergence).

## Condition number and preconditioning

**Why the iteration count grows.** For CG, the error falls roughly by a factor of (√κ − 1)/(√κ + 1) per iteration, where κ is the condition number of A (ratio of largest to smallest eigenvalue). The number of iterations to reach a fixed tolerance therefore grows like **√κ**.

For second-order elliptic problems discretized on a mesh of size h, κ grows like **1/h²**. So √κ grows like 1/h: **each time the mesh is halved, plain CG needs about twice as many iterations**, and each iteration works on a larger system. The code below shows exactly this: 59, 119, 239, 470, 941 iterations as N grows from about 1,000 to 262,000.

**Preconditioning** replaces A x = b with an equivalent system M⁻¹A x = M⁻¹b whose condition number is much smaller, where M approximates A but is cheap to invert. Common choices:

| Preconditioner | Idea | Notes |
| --- | --- | --- |
| Jacobi (diagonal) | M = diag(A) | Trivial and parallel; helps with badly scaled rows, little else |
| Incomplete Cholesky / ILU | Factorize but drop most of the fill-in | Strong general-purpose choice; hard to parallelize; ILU is not symmetric, so pair it with GMRES or BiCGStab rather than CG |
| Algebraic multigrid (AMG) | A multigrid hierarchy built from the matrix alone | Near mesh-independent iteration counts for elliptic problems |
| Domain decomposition | Solve local problems on subdomains, combine them | Natural for distributed-memory parallel computing |
| Physics-based / block | Exploit the structure (for example velocity and pressure blocks) | Standard for coupled and saddle-point problems |

## Multigrid

**The key observation.** Simple iterative "smoothers" such as Jacobi or Gauss-Seidel remove **high-frequency** (oscillatory, cell-to-cell) error components very quickly, but are hopelessly slow on **low-frequency** (smooth) error. But a smooth error on a fine grid looks *oscillatory* on a coarser grid, where it is cheap to remove.

**The V-cycle:**

```
fine grid      smooth ──▶ restrict residual                              prolong correction ──▶ smooth
                              ╲                                              ╱
medium grid                  smooth ──▶ restrict              prolong ──▶ smooth
                                            ╲                  ╱
coarsest grid                               solve exactly (small system)
```

1. A few smoothing sweeps on the fine grid remove the oscillatory error.
2. Transfer (restrict) the remaining residual to a coarser grid.
3. Solve for the correction there, recursively, down to a grid small enough for a direct solve.
4. Interpolate (prolong) the correction back up and add it.
5. Smooth again to remove the high-frequency error introduced by interpolation.

Because each level handles the error frequencies it is good at, the convergence rate per cycle is roughly independent of the mesh size, and one cycle costs O(N). **Geometric multigrid** builds the coarse grids from the mesh; **algebraic multigrid (AMG)** builds them from the matrix entries alone, which makes it practical for unstructured meshes (libraries: hypre's BoomerAMG, ML and MueLu in Trilinos, PETSc's GAMG, pyamg in Python).

In the code below, CG preconditioned with AMG needs **5 to 6 iterations at every mesh size**, from about 1,000 to 262,000 unknowns, while plain CG needs up to 941.

Multigrid works best for elliptic, diffusion-like problems. Strong convection, high-contrast materials, thin structures and indefinite systems need more sophisticated variants.

## Code: CG with and without multigrid

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pyamg                                        # pip install pyamg

def poisson_2d(n):
    """5-point Laplacian on an n x n interior grid of the unit square (Dirichlet walls): SPD."""
    T = sp.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1])
    I = sp.identity(n)
    return ((sp.kron(I, T) + sp.kron(T, I)) * (n + 1) ** 2).tocsr()

def cg_iterations(A, b, M=None):
    count = [0]
    x, info = spla.cg(A, b, rtol=1e-8, M=M, maxiter=20000,
                      callback=lambda xk: count.__setitem__(0, count[0] + 1))
    assert info == 0                                 # converged
    return count[0], x

results = {}
for n in [32, 64, 128, 256, 512]:
    A = poisson_2d(n)
    b = np.ones(A.shape[0])
    it_cg, x_cg = cg_iterations(A, b)                                  # plain CG
    ml = pyamg.ruge_stuben_solver(A)                                   # algebraic multigrid hierarchy
    it_amg, x_amg = cg_iterations(A, b, ml.aspreconditioner())         # CG preconditioned by AMG
    x_direct = spla.spsolve(A.tocsc(), b)                              # sparse direct, for comparison
    assert np.allclose(x_cg, x_direct, rtol=1e-5, atol=1e-9)
    results[n * n] = (it_cg, it_amg)
    print(f"N = {n*n:7d} unknowns | CG: {it_cg:4d} iterations | CG + AMG: {it_amg} iterations")
```

Output (iteration counts; one machine took about 2.5 s for plain CG, 0.5 s for CG + AMG and 2.1 s for the direct solve at the largest size):

```
N =    1024 unknowns | CG:   59 iterations | CG + AMG: 5 iterations
N =    4096 unknowns | CG:  119 iterations | CG + AMG: 5 iterations
N =   16384 unknowns | CG:  239 iterations | CG + AMG: 6 iterations
N =   65536 unknowns | CG:  470 iterations | CG + AMG: 6 iterations
N =  262144 unknowns | CG:  941 iterations | CG + AMG: 6 iterations
```

Plain CG doubles its iteration count every time the mesh spacing halves, as √κ ∝ 1/h predicts. With AMG the count is flat. In 2D at this size the sparse direct solver is still competitive; in 3D, fill-in makes it fall behind much sooner.

## How to choose

| Situation | Usual choice |
| --- | --- |
| Small to medium models; many load cases; ill-conditioned or indefinite systems | Sparse direct |
| Large 3D SPD problems (elasticity, conduction, pressure Poisson) | CG with AMG (or another strong preconditioner) |
| Non-symmetric systems (convection, CFD momentum) | GMRES or BiCGStab with ILU or AMG |
| Saddle-point systems (incompressible flow, mixed formulations, contact with Lagrange multipliers) | Block preconditioners, or direct solvers for moderate sizes |
| Massive parallel runs | Iterative methods with domain decomposition or AMG; direct solvers scale less well |
| Thin shells, high-contrast materials | Direct or specialized preconditioners; plain AMG may struggle |

## Why this matters for machine learning

- **Learned preconditioners and warm starts** can make iterative solvers faster while the solver's tolerance still guarantees the answer. A good initial guess from a surrogate means fewer iterations; a learned preconditioner changes the convergence rate, not the solution. See [Speeding up a solver with ML without changing its answer](newton-raphson-and-nonlinear-solvers.md#speeding-up-a-solver-with-ml-without-changing-its-answer).
- **Multigrid inspired network architectures.** U-Nets and multiscale GNNs pass information through coarser levels for the same reason multigrid does: local operations alone propagate long-range information too slowly. See [Multiscale GNNs](multiscale-gnns.md).
- **Message passing is like a smoother.** A GNN with k layers can only move information k hops, much as k Jacobi sweeps can. See [How many message-passing steps](gnns-and-meshgraphnets.md#how-many-message-passing-steps).
- **Differentiating through a solve** uses the same linear solver again: the adjoint (backward) pass of x = A⁻¹b needs a solve with Aᵀ. See [The adjoint method](adjoint-method.md).

## In an interview

**Thirty-second answer.** "Direct solvers factorize the sparse matrix; they are robust and cheap for extra load cases, but fill-in makes memory and time explode for large 3D models. Iterative solvers only need matrix-vector products: CG for symmetric positive-definite systems like elasticity, GMRES or BiCGStab for non-symmetric ones. Their iteration count grows with the square root of the condition number, which grows as 1/h² under mesh refinement, so they need preconditioners. Multigrid removes error at every scale using coarser grids and gives nearly mesh-independent iteration counts; AMG-preconditioned CG is the standard for big elliptic problems."

**Likely follow-ups**

- *Why does plain CG slow down on finer meshes?* κ ∝ 1/h², iterations ∝ √κ ∝ 1/h: twice the iterations per halving of h, as in the table above.
- *Why does multigrid work?* Smoothers kill high-frequency error; coarse grids turn smooth error into high-frequency error where it is cheap to kill.
- *When would you still pick a direct solver?* Moderate size, many right-hand sides, ill-conditioned or indefinite systems, or when robustness matters more than memory.

## References

- [Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd edition (free PDF from the author)](https://www-users.cse.umn.edu/~saad/IterMethBook_2ndEd.pdf)
- [J. R. Shewchuk, *An Introduction to the Conjugate Gradient Method Without the Agonizing Pain*](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)
- W. L. Briggs, V. E. Henson and S. F. McCormick, *A Multigrid Tutorial*, 2nd edition, SIAM, 2000.
- [PyAMG: algebraic multigrid solvers in Python](https://github.com/pyamg/pyamg)
