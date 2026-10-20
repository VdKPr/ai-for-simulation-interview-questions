# Design: Inverse Design Tool

[← Questions](../README.md) · [All answers](README.md) · [System design framework](system-design-framework.md)

**Question.** Design an inverse-design tool: a target pressure distribution in, a geometry out.

**Short answer.**

- **Inverse design is ill-posed.** Some targets have no exact solution, some have many, and small changes in the target can change the answer a lot. So the tool should return **a set of candidate geometries that reproduce the target within a tolerance and satisfy constraints**, each verified with the solver, rather than "the" geometry.
- **Two main approaches:**
  1. **Optimization through a differentiable forward model:** minimize ‖C_p(geometry) − C_p,target‖² plus a regularizer, using a fast surrogate (or an adjoint solver) for gradients, from several starting shapes.
  2. **A conditional generative model** (diffusion or flow matching) that samples geometries given the target, followed by refinement with the forward model.
- **Shape parameterization does much of the work:** CST or B-spline coefficients for airfoils, or a latent space of plausible shapes. It keeps candidates smooth and manufacturable.
- **Verification:** the solver checks every returned candidate. The tool reports the achieved C_p, the mismatch and the constraint margins.

**On this page**

- [Requirements and clarifying questions](#requirements-and-clarifying-questions)
- [Why the inverse problem is hard](#why-the-inverse-problem-is-hard)
- [Shape parameterization](#shape-parameterization)
- [Approach 1: optimization through a forward model](#approach-1-optimization-through-a-forward-model)
- [Approach 2: conditional generative models](#approach-2-conditional-generative-models)
- [Verification and output](#verification-and-output)
- [Architecture](#architecture)
- [In an interview](#in-an-interview)

## Requirements and clarifying questions

- **What object?** 2D airfoil sections (the classic case), 3D wings, turbomachinery blades, ducts?
- **Target:** a full C_p distribution, or features (a pressure plateau, a shock position, a pressure-gradient limit for laminar flow)?
- **Conditions:** Mach, Reynolds, angle of attack fixed or free (angle of attack is often a design variable too)?
- **Constraints:** thickness, volume, leading-edge radius, trailing-edge angle, manufacturing.
- **Interactivity:** seconds (exploration) or minutes (design refinement)?

## Why the inverse problem is hard

- **Existence:** an arbitrary C_p curve may not correspond to any physical shape at those conditions (for example, it must close the lift and satisfy the Kutta condition). Treat the target as a **soft objective**, not a constraint.
- **Non-uniqueness:** different shapes can produce nearly the same distribution, especially if only part of the curve is specified.
- **Instability:** small target changes can cause large shape changes; regularization (smoothness penalties, staying in a learned shape space) stabilizes it ([Inverse problems and regularization](inverse-problems-and-regularization.md)).
- **Surrogate exploitation:** optimizing through a surrogate finds shapes where the surrogate is wrong; constrain to its training distribution and verify with the solver ([Surrogate-based optimization](surrogate-based-optimization.md)).

## Shape parameterization

| Parameterization | Use |
| --- | --- |
| CST (class-shape transformation) or Bernstein polynomials | Airfoils; few parameters, smooth, standard |
| B-spline or free-form deformation control points | General shapes and 3D; more freedom |
| Learned latent space (autoencoder, VAE) | Restricts search to realistic shapes; enables generative models ([Geometry representations](geometry-representations.md)) |
| Signed distance fields | Free topology; harder to constrain |

Constraints such as minimum thickness should be enforced in the parameterization or as explicit penalties, never left to chance.

## Approach 1: optimization through a forward model

```math
\min_{z,\;\alpha} \;\; \sum_i w_i \big(C_p(s_i; z, \alpha) - C_p^\text{target}(s_i)\big)^2 \;+\; \lambda\, R(z) \quad \text{s.t. } g(z) \le 0
```

- z: shape parameters; α: angle of attack (if free); s_i: positions along the surface; R: regularizer; g: geometric constraints.
- **Forward model:** a differentiable surrogate (fast gradients by autodiff) or the solver's **adjoint** (accurate gradients, slower) ([Adjoint method](adjoint-method.md), [Automatic differentiation](automatic-differentiation.md)).
- **Multi-start** from different initial shapes to find several solutions; keep distinct ones.
- **Trust region:** penalize distance from the surrogate's training data (embedding distance or ensemble disagreement).
- **Classical alternatives** (useful baselines): residual-correction methods for airfoil inverse design, and adjoint-based inverse design with the full solver.

## Approach 2: conditional generative models

- Train a **conditional diffusion or flow-matching model** p(shape | C_p target, conditions) on pairs from the simulation database. Sampling gives many diverse candidates in seconds, which handles non-uniqueness naturally ([Generative models for physics](generative-models-for-physics.md)).
- The demo on that page shows conditional flow matching finding **both** solution families for a target, where a single optimization run finds one.
- **Refine** each sample with a few gradient steps through the forward surrogate, then verify.
- **Out-of-distribution targets** (unlike anything in training) produce poor samples; detect them with the C_p mismatch of the forward model and a target-embedding OOD score.

## Verification and output

- Every candidate is run through the **solver** at the design conditions.
- Output per candidate: geometry (CAD-ready coordinates or spline), achieved C_p overlaid on the target, mismatch metrics (overall and in regions the user marked as important), constraint margins, aerodynamic coefficients.
- Candidates ranked by mismatch and by secondary objectives (drag, robustness to off-design conditions).
- If no candidate reaches the tolerance, say so: the target may be infeasible; show the closest achievable distributions.

## Architecture

```
target Cp + conditions + constraints
        │
        ├─► target OOD check
        ▼
 conditional generator ─► N candidate shapes ─┐
 multi-start optimizer (surrogate grads) ─────┼─► refine (gradient steps) ─► dedupe ─► solver verification
                                              │                                          │
                        shape parameterization (CST / B-spline / latent) ◄──────────────┘
                                                                                         ▼
                                       ranked candidates: geometry, achieved Cp vs target, margins
```

## In an interview

**Thirty-second answer.** "Inverse design is ill-posed. A target might have no solution or several, so I'd return a set of verified candidates rather than one answer. I'd parameterize shapes compactly, with CST for airfoils or a learned latent space, so candidates stay smooth and manufacturable. Then either optimize the Cp mismatch through a differentiable surrogate or adjoint from several starting shapes, or sample candidates from a conditional diffusion or flow-matching model and refine them. Constraints are explicit, the search stays where the surrogate is trustworthy, and every candidate is checked with the solver, with the achieved distribution shown against the target."

**Likely follow-ups**

- *How do you know if the target is achievable?* If no verified candidate approaches it, report the closest; check physical consistency (lift and Kutta condition).
- *Surrogate or adjoint gradients?* Surrogate for speed and exploration; adjoint for final refinement where the surrogate is uncertain.

## References

- B. M. Kulfan, "Universal parametric geometry representation method", *Journal of Aircraft*, 2008 (CST).
- A. Jameson, "Aerodynamic design via control theory", *Journal of Scientific Computing*, 1988.
- [Inverse problems and regularization](inverse-problems-and-regularization.md), [Generative models for physics](generative-models-for-physics.md), [Adjoint method](adjoint-method.md)
