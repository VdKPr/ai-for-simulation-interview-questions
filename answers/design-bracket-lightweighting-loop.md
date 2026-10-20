# Design: Bracket Lightweighting Loop

[← Questions](../README.md) · [All answers](README.md) · [System design framework](system-design-framework.md)

**Question.** Design a lightweighting loop for a structural bracket that never ships an unverified design.

**Short answer.**

- **The surrogate proposes, the solver disposes.** The loop minimizes mass subject to stress, displacement and manufacturability constraints. Design variables are parameters such as rib thicknesses and hole sizes, or a topology-optimization density field.
- **A fast surrogate (GP or GNN) explores** thousands of candidates, with **pessimistic constraints**: predicted stress plus k standard deviations must stay below the allowable.
- **Every design that leaves the loop is verified** by full FEA with a converged mesh. A design that fails verification is added to the training data and the loop continues.
- **"Never ships unverified" is enforced by the system, not by good intentions.** The release step accepts only designs carrying a verification record (FEA result, mesh-convergence check, model and solver versions) and an engineer's sign-off.

**On this page**

- [Requirements and clarifying questions](#requirements-and-clarifying-questions)
- [Design variables and constraints](#design-variables-and-constraints)
- [The loop](#the-loop)
- [Surrogate and pessimistic constraints](#surrogate-and-pessimistic-constraints)
- [Verification gate](#verification-gate)
- [Architecture](#architecture)
- [Risks and roadmap](#risks-and-roadmap)
- [In an interview](#in-an-interview)

## Requirements and clarifying questions

- **Load cases:** which ones (static, fatigue, modal), and are they fixed? A bracket usually has several load cases; the design must pass all of them.
- **Constraints:** allowable stress (with safety factor), maximum displacement, minimum natural frequency, fatigue life ([Fatigue](fatigue.md), [Modal analysis](modal-analysis.md)).
- **Manufacturing process:** machining, casting or additive manufacturing; minimum wall thickness, draft angles, overhangs.
- **Fixed geometry:** bolt holes, interfaces, keep-out zones.
- **Material:** fixed or a choice?
- **What "verified" means here:** which solver, mesh standard, convergence check and sign-off process.

## Design variables and constraints

| Approach | Variables | Pros | Cons |
| --- | --- | --- | --- |
| Parametric sizing | Rib thicknesses, fillet radii, pocket depths (5–30 parameters) | Manufacturable by construction; GPs work well | Limited to the topology chosen |
| Topology optimization | Element densities (thousands to millions) | Finds new load paths | Needs interpretation into CAD; adjoint gradients do the work, surrogates help less ([Topology optimization](topology-optimization.md)) |
| Generative + surrogate | Latent shape codes | Broad design space | Harder to guarantee manufacturability |

A practical combination: topology optimization suggests the layout; a parametric CAD model of that layout is then sized with the surrogate loop.

The optimization problem:

```math
\min_{x} \; m(x) \quad \text{s.t.} \quad \sigma_\text{max}(x) \le \frac{\sigma_\text{allow}}{SF},\quad \delta_\text{max}(x) \le \delta_\text{allow},\quad f_1(x) \ge f_\text{min},\quad g_\text{mfg}(x) \le 0
```

Mass and manufacturability constraints are computed exactly from the geometry; only the physics constraints need the surrogate.

## The loop

1. **Initial data:** a space-filling set of FEA runs on the parametric model (about 10 × the number of parameters).
2. **Train** surrogates for each constrained quantity (peak stress per load case, displacement, frequency).
3. **Optimize** on the surrogate with pessimistic constraints (below), using a gradient-based or evolutionary optimizer ([Surrogate-based optimization](surrogate-based-optimization.md)).
4. **Verify** the best few candidates with full FEA.
5. If a candidate fails, or the surrogate's error at it is large, **add it to the training data** and repeat from step 2. This is constrained Bayesian optimization in effect ([Bayesian optimization](bayesian-optimization.md)).
6. **Stop** when a verified design meets all constraints and the mass improvement per iteration is small.

## Surrogate and pessimistic constraints

- **Optimizers exploit surrogate errors:** they find places where the model under-predicts stress. The demo in [Surrogate-based optimization](surrogate-based-optimization.md) shows this effect.
- **Counter it with pessimism:** require μ_σ(x) + k·s_σ(x) ≤ σ_allow/SF, with k around 2, using an ensemble or GP standard deviation. Or use the probability of feasibility as a constraint.
- **Trust regions:** limit each iteration's step to the region where the surrogate has data nearby.
- **Peak stress is local:** use a model that predicts the stress field and take its maximum, or a dedicated peak head. Check that the predicted hot-spot location matches FEA.

## Verification gate

The guarantee comes from the process:

- **FEA on the final geometry** (the exact CAD to be released, not the parameters), with a mesh-convergence check on peak stress at the hot spots ([Mesh convergence](verification-validation-and-mesh-convergence.md#mesh-convergence-and-the-grid-convergence-index)).
- **All load cases**, including those not used in the optimization (a design optimized for static load may fail in modal or fatigue checks).
- **Verification record** stored with the design: geometry hash, solver and version, mesh statistics, results, check status.
- **Release system rule:** a design without a passing verification record **cannot** be exported to the PLM or manufacturing system. An engineer reviews and signs.
- **Audit:** surrogate predictions vs verified results are logged, which also measures surrogate accuracy over time.

## Architecture

```
parametric CAD ─► initial FEA set ─► surrogates (σ, δ, f per load case) ─► optimizer with μ + kσ constraints
       ▲                                                                              │
       │                                                                              ▼
       └──── add to data ◄── fail ── full FEA + convergence check on exact CAD ◄── top candidates
                                             │ pass
                                             ▼
                    verification record ─► engineer sign-off ─► release (blocked without record)
```

## Risks and roadmap

- **Surrogate exploitation:** pessimistic constraints, trust regions and verification.
- **Singular stresses** at sharp corners make peak values mesh-dependent: enforce minimum fillet radii in the parametrization ([Singularities](stress-concentrations-and-singularities.md)).
- **Missing failure modes** (buckling of thin ribs, fatigue): include them as load cases or checks ([Buckling](buckling.md)).
- **v0:** parametric sizing with a GP and full verification.
- **v1:** a field surrogate (GNN) for peak location; multiple load cases.
- **v2:** topology optimization upstream; manufacturability checks automated.

## In an interview

**Thirty-second answer.** "The surrogate only proposes. I'd minimize mass over a parametric bracket subject to stress, displacement and frequency constraints for every load case. Mass and manufacturability are computed exactly; the surrogates predict the physics. Because optimizers exploit surrogate errors, the constraints are pessimistic, mean plus about two standard deviations, inside trust regions. The best candidates get full FEA with a mesh-convergence check on the exact CAD, failures go back into the training data, and the release system refuses any design without a passing verification record and an engineer's sign-off."

**Likely follow-ups**

- *Why not just run topology optimization with FEA?* Often you should; adjoint gradients make it efficient. The surrogate helps for many load cases, non-differentiable constraints, or design-space exploration.
- *How many FEA runs?* Tens for initial data in a 10-parameter problem, plus a few per iteration; far fewer than a pure FEA search.

## References

- [Topology optimization](topology-optimization.md), [Adjoint method](adjoint-method.md), [Surrogate-based optimization](surrogate-based-optimization.md)
- M. P. Bendsøe and O. Sigmund, *Topology Optimization: Theory, Methods and Applications*, Springer, 2003.
- M. A. Gelbart, J. Snoek and R. P. Adams, "Bayesian optimization with unknown constraints", UAI 2014 ([arXiv:1403.5607](https://arxiv.org/abs/1403.5607)).
