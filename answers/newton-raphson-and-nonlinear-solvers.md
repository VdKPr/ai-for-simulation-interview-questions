# Newton-Raphson and Nonlinear Solvers

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A nonlinear simulation (plasticity, large deformation, contact, nonlinear material, Navier–Stokes) gives a system of nonlinear equations R(u) = 0, where R is the out-of-balance force: internal forces minus external loads. **Newton-Raphson** linearizes R around the current guess, solves the linear system **K_t Δu = −R(u)** with the tangent stiffness matrix K_t = ∂R/∂u, updates u, and repeats until the residual is small. Close to the solution it converges *quadratically*: the number of correct digits roughly doubles each iteration. Far from the solution it can overshoot or diverge, so solvers apply the load in increments, use line search, and switch to arc-length methods at limit points. When Newton does not converge, the usual causes are too large a load step, a physical instability (buckling, snap-through, material softening), contact changing state, an unconstrained part, or an inaccurate tangent. Machine learning can make the solver faster *without changing its answer* by giving it a better starting guess or a better preconditioner, because the solver's own convergence check still decides when it is done.

**On this page**

- [Where nonlinear equations come from](#where-nonlinear-equations-come-from)
- [The Newton-Raphson iteration](#the-newton-raphson-iteration)
- [Why convergence is quadratic, and why only near the solution](#why-convergence-is-quadratic-and-why-only-near-the-solution)
- [Load stepping, line search and arc-length methods](#load-stepping-line-search-and-arc-length-methods)
- [Variants: modified Newton and quasi-Newton](#variants-modified-newton-and-quasi-newton)
- [When Newton does not converge](#when-newton-does-not-converge)
- [Code: Newton, cold start, warm start and load stepping](#code-newton-cold-start-warm-start-and-load-stepping)
- [Speeding up a solver with ML without changing its answer](#speeding-up-a-solver-with-ml-without-changing-its-answer)
- [In an interview](#in-an-interview)

## Where nonlinear equations come from

In a linear static analysis, R(u) = K u − f, and one linear solve gives the answer. Many real problems are not linear:

- **Material nonlinearity:** plasticity, hyperelastic rubber, creep, damage. The stiffness depends on the strain.
- **Geometric nonlinearity:** large displacements or rotations, so the equilibrium must be written on the deformed shape; buckling and snap-through.
- **Contact:** parts touch or separate, and the contact area changes with the load.
- **Fluids:** the convective term (u·∇)u in Navier–Stokes is quadratic in the velocity.
- **Coupled physics:** temperature-dependent material properties, for example.

See [Nonlinear structural analysis](nonlinear-structural-analysis.md) for the structural side.

## The Newton-Raphson iteration

Expand the residual around the current guess u_k and keep the linear term:

```math
R(u_k + \Delta u) \approx R(u_k) + K_t(u_k)\,\Delta u = 0, \qquad K_t = \frac{\partial R}{\partial u}
```

So each iteration is:

1. compute the residual R(u_k) (assemble internal forces, subtract external loads);
2. compute the tangent stiffness K_t(u_k);
3. solve the *linear* system K_t Δu = −R(u_k) (with a sparse solver; see [Sparse linear solvers](sparse-linear-solvers.md));
4. update u_{k+1} = u_k + Δu;
5. check convergence; if not converged, go back to step 1.

In one dimension this is the familiar picture: follow the tangent line of the curve down to zero, and repeat from there.

**Convergence criteria** used by structural codes combine:

- the **force residual**: ‖R‖ small compared with the applied load (for example below 0.5% of a reference force);
- the **displacement increment**: ‖Δu‖ small compared with the total displacement;
- sometimes the **energy**: |Δuᵀ R| small.

Using only the displacement criterion can stop too early when the solution creeps; using only the force criterion can be hard to satisfy near zero load. Good practice uses both.

## Why convergence is quadratic, and why only near the solution

If K_t is the exact derivative and the starting point is close enough to the solution, the error after the next iteration is proportional to the *square* of the current error:

```math
|e_{k+1}| \le C\,|e_k|^2
```

An error of 10⁻² becomes about 10⁻⁴, then 10⁻⁸, then 10⁻¹⁶. That is why the last few Newton iterations look like 10⁻¹, 10⁻³, 10⁻⁷, 10⁻¹⁵ in a solver log, and why a log showing slow linear decrease near the end usually means the tangent is wrong (an approximate material tangent, or a bug).

"Close enough" is the catch. Far from the solution, the linearization can be a poor model of R, and a full Newton step can overshoot badly, even making the residual larger, as in the first iterations of the code below. For problems with limit points or path dependence, a far-away start can converge to the wrong solution or not at all.

## Load stepping, line search and arc-length methods

**Load stepping (incrementation).** Instead of applying the full load at once, apply it in increments, F = λF_total with λ = 0.1, 0.2, …, 1.0, and solve each increment starting from the previous converged state. Each increment starts close to its solution, inside Newton's region of quadratic convergence. Codes cut the increment automatically when an increment fails to converge and grow it when convergence is easy. For **path-dependent** materials such as plasticity, increments are not just a convergence aid: the answer depends on the loading path, so the path must be followed. See [Plasticity and load-history dependence](nonlinear-structural-analysis.md#plasticity-and-load-history-dependence).

**Line search.** Keep the Newton direction but choose the step length: u_{k+1} = u_k + s·Δu with s in (0, 1] chosen so that the residual (or an energy) actually decreases. This prevents the large overshoots of the first iterations.

**Arc-length (Riks) methods.** At a **limit point** (the peak load of a buckling or snap-through response), the load-displacement curve turns over: beyond the peak, there is no nearby equilibrium at a higher load, and the tangent stiffness becomes singular. Load control fails there. Arc-length methods make the load factor λ an unknown and constrain the size of each step along the equilibrium path instead, so they can follow the curve through the peak and down the unstable branch. They are the standard tool for post-buckling and snap-through analysis. See [Buckling](buckling.md).

```
load F
  │          limit point
  │            ●───╮
  │          ╱      ╲            load control: fails at the peak (no nearby solution)
  │        ╱          ╲      ╱   arc-length: follows the curve over the peak
  │      ╱              ╲__╱
  │    ╱
  └──────────────────────────── displacement u
```

## Variants: modified Newton and quasi-Newton

- **Modified Newton:** keep the tangent stiffness from the start of the increment (or update it only occasionally). Each iteration then reuses the same factorization and is much cheaper, but convergence becomes linear and needs more iterations. Useful when factorization dominates the cost.
- **Quasi-Newton (BFGS and relatives):** update an approximation of the tangent (or its inverse) from the changes in residual and displacement between iterations. Faster convergence than modified Newton without forming the full tangent every time.
- **Jacobian-free Newton-Krylov:** solve the Newton linear system with an iterative Krylov method that only needs products K_t·v, approximated by finite differences of R. Common in large multiphysics codes.
- **Picard (fixed-point) iteration:** freeze the nonlinearity (for example the convecting velocity in Navier–Stokes) and solve a linear problem repeatedly. More robust far from the solution but only linearly convergent; often used for a few iterations before switching to Newton.

## When Newton does not converge

Diagnose before changing settings. Common causes, roughly in order of how often they occur:

1. **The load step is too large.** Reduce the increment; let automatic incrementation cut back.
2. **A physical instability.** Buckling, snap-through, wrinkling or material softening make the tangent singular or negative. Check whether the structure is actually failing. Use an arc-length method, or a dynamic analysis with some damping, to follow it. See [Buckling](buckling.md).
3. **Contact changing state.** Contact chattering between open and closed from one iteration to the next. Improve the initial contact (small initial overlap or gap closure), adjust contact stiffness, add a small stabilization, or refine the contact surfaces.
4. **Insufficient constraints.** A part that can move as a rigid body until contact engages has a singular tangent at the start. Add soft springs or stabilization until contact is established.
5. **An inaccurate tangent.** A material model or element with an approximate tangent gives linear instead of quadratic convergence and can stall. Check the convergence rate in the log.
6. **Distorted elements.** Large deformation can turn elements inside out (negative Jacobians). Use better element formulations, remeshing, or a finer mesh where deformation concentrates.
7. **Bad material data.** A stress-strain curve with a negative slope, or unrealistic values, can cause genuine non-uniqueness.

**A good diagnostic habit:** look at where in the model the largest residual force and the largest displacement increment occur. They usually point straight at the problem (a contact pair, an unconstrained part, a single distorted element).

## Code: Newton, cold start, warm start and load stepping

A hardening spring with internal force k·u + c·u³ under an applied load F = 20. The same structure as a nonlinear FE solve, in one unknown.

```python
import numpy as np

k, c = 1.0, 0.5                                   # hardening spring: internal force k u + c u^3

def residual(u, F):
    return k * u + c * u**3 - F                   # out-of-balance force R(u) = F_int(u) - F_ext

def tangent(u):
    return k + 3 * c * u**2                       # tangent stiffness dR/du

def newton(F, u0, tol=1e-10, max_iter=50):
    u, history = u0, []
    for it in range(max_iter):
        r = residual(u, F)
        history.append(abs(r))
        if abs(r) <= tol * max(1.0, abs(F)):
            return u, it, history
        u = u - r / tangent(u)                    # solve K_t du = -R and update
    raise RuntimeError("Newton did not converge")

u_star, iters, hist = newton(F=20.0, u0=0.0)
print(f"cold start (u0 = 0): {iters} iterations, u = {u_star:.10f}")
print("  |R| per iteration:", " ".join(f"{h:.1e}" for h in hist))

u_warm, iters_warm, hist_warm = newton(F=20.0, u0=1.02 * u_star)   # a 'surrogate' guess, 2% off
print(f"warm start (2% off): {iters_warm} iterations, |u - u_cold| = {abs(u_warm - u_star):.1e}")
print("  |R| per iteration:", " ".join(f"{h:.1e}" for h in hist_warm))

total, u = 0, 0.0                                 # load stepping: 5 increments of 4
for F in np.linspace(4, 20, 5):
    u, it, _ = newton(F, u0=u)
    total += it
print(f"load stepping (5 steps): {total} iterations in total, u = {u:.10f}")
```

Output:

```
cold start (u0 = 0): 10 iterations, u = 3.2252404628
  |R| per iteration: 2.0e+01 4.0e+03 1.2e+03 3.5e+02 9.8e+01 2.5e+01 4.5e+00 2.8e-01 1.3e-03 3.1e-08 0.0e+00
warm start (2% off): 3 iterations, |u - u_cold| = 4.9e-14
  |R| per iteration: 1.1e+00 2.0e-02 6.8e-06 8.1e-13
load stepping (5 steps): 24 iterations in total, u = 3.2252404628
```

Three lessons in one output:

1. **Far from the solution, Newton can make things worse.** The first full step from u = 0 overshoots to u = 20 and the residual jumps from 20 to 4,000. It then creeps back.
2. **Near the solution, convergence is quadratic:** 2.8e-1, 1.3e-3, 3.1e-8, then zero to machine precision.
3. **A good starting guess saves most of the work and changes nothing.** Starting 2% away from the answer, the solver needs 3 iterations instead of 10 and lands on the same solution to 5e-14. That is the warm-start idea below.

Load stepping took more iterations in total here, because this spring is well-behaved. Its value is robustness: on problems with contact, plasticity or limit points, full-load Newton often fails outright, and for path-dependent materials the increments are part of the physics.

## Speeding up a solver with ML without changing its answer

The key idea: let the ML model influence **how** the solver gets to the answer, not **what** the answer is. The solver's convergence criterion (residual below tolerance) still decides when it is done, so the result satisfies the same discrete equations to the same tolerance as without ML. If the model gives bad advice, the solver takes longer, but it does not return a wrong answer.

Ways to do this:

1. **Warm starts (better initial guesses).** A surrogate predicts the solution (displacements, flow field), and the solver starts from it instead of from zero or the previous design. Newton then begins inside its region of quadratic convergence, as in the code above. For iterative linear solvers, a good initial guess reduces the number of iterations needed to reach the tolerance. Jin et al. (2024) proposed a neural hybrid Newton solver of exactly this kind: a learned initialization for the Newton iteration inside implicit time steps of stiff nonlinear equations, trained offline, with a quantifiable rate of improvement.
2. **Learned preconditioners.** A network predicts a preconditioner (or the parameters of one) for the linear systems inside Newton. The Krylov solver still converges to the same tolerance; a better preconditioner just means fewer iterations.
3. **Learned solver settings.** Predict a good load-increment size, line-search parameters, multigrid parameters or contact stabilization from the problem features, to avoid failed increments and cutbacks.
4. **Learned coarse corrections inside multigrid or domain decomposition,** with the fine-level smoother and convergence check unchanged.
5. **Screening, not replacing:** use the surrogate to choose which designs are worth solving at all; solve those with the unchanged solver.

**What to check:** that the converged results are identical to the baseline within tolerance, that total wall time (including inference) actually drops, and what happens when the ML guess is bad (out-of-distribution designs): the solver should fall back to its default behaviour, not fail.

Contrast this with approaches that *do* change the answer, such as learned corrections to a coarse-grid solver. Those can be very effective, but they produce a different solution and need their own validation. See [Correcting a coarse solver](differentiable-physics-and-hybrid-solvers.md#correcting-a-coarse-solver).

## In an interview

**Thirty-second answer to "How does Newton-Raphson work, and what do you do when it does not converge?"** "Write the problem as a residual R(u) = internal minus external forces. Linearize it with the tangent stiffness, solve K_t Δu = −R, update, and repeat until the force residual and displacement increment are small. Near the solution it converges quadratically. If it does not converge, I reduce the load increment, use line search, and check for physical instabilities like buckling or snap-through, which need arc-length methods; then I check contact, rigid-body modes, element distortion and whether the tangent is consistent, by looking at where the largest residuals are."

**Likely follow-ups**

- *Why does convergence become linear?* The tangent is not the exact derivative of the residual (approximate material tangent, modified Newton, or a bug).
- *Why can't load control pass a limit point?* At the peak the tangent stiffness is singular and there is no equilibrium at a slightly higher load; arc-length methods treat the load as an unknown.
- *How can ML speed up the solver without changing the answer?* Warm starts, learned preconditioners and learned settings; the solver's tolerance still decides convergence.

## References

- [A fast neural hybrid Newton solver adapted to implicit methods for nonlinear dynamics (Jin et al., 2024)](https://arxiv.org/abs/2407.03945)
- N.-H. Kim, *Introduction to Nonlinear Finite Element Analysis*, Springer, 2015.
- P. Wriggers, *Nonlinear Finite Element Methods*, Springer, 2008.
- [Y. Saad, *Iterative Methods for Sparse Linear Systems* (free PDF from the author)](https://www-users.cse.umn.edu/~saad/IterMethBook_2ndEd.pdf)
