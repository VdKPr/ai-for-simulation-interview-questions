# The Weak Form and the Galerkin Method

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** The *strong form* of a PDE demands that the equation holds at every point. The *weak form* only demands that it holds on average against every test function: multiply the equation by a test function, integrate over the domain and integrate by parts. This moves one derivative from the unknown onto the test function, so the solution only needs first derivatives, and simple piecewise-linear functions on a mesh become allowed. The **Galerkin method** then approximates the solution in a finite set of basis functions and uses the same functions as test functions, which turns the weak form into a linear system **K u = f**. Boundary conditions split into two kinds: **essential** (Dirichlet) conditions must be built into the space of candidate solutions, while **natural** (Neumann) conditions appear by themselves in the boundary integral of the weak form.

**On this page**

- [The weak form](#the-weak-form)
- [Why FEM uses the weak form](#why-fem-uses-the-weak-form)
- [The Galerkin method](#the-galerkin-method)
- [A worked 1D example](#a-worked-1d-example)
- [Essential and natural boundary conditions](#essential-and-natural-boundary-conditions)
- [Energy minimization: the same thing seen differently](#energy-minimization-the-same-thing-seen-differently)
- [Variations: Petrov-Galerkin and stabilization](#variations-petrov-galerkin-and-stabilization)
- [Why this matters for machine learning](#why-this-matters-for-machine-learning)
- [In an interview](#in-an-interview)

## The weak form

Start from the Poisson problem, which describes steady heat conduction, electrostatics and many other things:

```math
-\nabla \cdot (k \nabla u) = f \ \text{ in } \Omega, \qquad u = 0 \ \text{ on } \Gamma_D, \qquad k\,\frac{\partial u}{\partial n} = g \ \text{ on } \Gamma_N .
```

Here Γ_D is the part of the boundary where the temperature is fixed and Γ_N the part where a heat flux g is applied. This is the **strong form**: it asks for a function u with two derivatives that satisfies the equation at every point.

**Step 1: multiply by a test function and integrate.** Take any smooth function v that is zero on Γ_D (a "test function"), multiply the equation by it, and integrate over the domain:

```math
-\int_\Omega \nabla \cdot (k \nabla u)\, v \, dx = \int_\Omega f\, v \, dx .
```

**Step 2: integrate by parts** (Green's identity, the multi-dimensional version of ∫u″v = [u′v] − ∫u′v′):

```math
\int_\Omega k\, \nabla u \cdot \nabla v \, dx \;-\; \int_{\partial\Omega} k\,\frac{\partial u}{\partial n}\, v \, ds \;=\; \int_\Omega f\, v \, dx .
```

**Step 3: use the boundary conditions.** On Γ_D the test function is zero, so that part of the boundary integral vanishes. On Γ_N, k ∂u/∂n is the given flux g. What remains is the **weak form**:

> Find u, with u = 0 on Γ_D, such that for every test function v that is zero on Γ_D:
>
> ∫_Ω k ∇u · ∇v dx = ∫_Ω f v dx + ∫_{Γ_N} g v ds.

The left side is usually written a(u, v) (a *bilinear form*) and the right side ℓ(v) (a *linear form*): find u such that a(u, v) = ℓ(v) for all v.

**Why "weak"?** Because less is demanded of u. The strong form needs second derivatives of u at every point. The weak form needs only first derivatives, and only in an integrated sense (mathematically, u must be in the Sobolev space H¹: square-integrable with a square-integrable gradient). Every strong solution is a weak solution; the reverse also holds when the solution is smooth enough, but weak solutions also exist in cases where the strong form makes no sense, such as a material with a jump in conductivity, or a point load.

## Why FEM uses the weak form

1. **Simple basis functions become allowed.** A piecewise-linear "hat" function has a kink at each node, so its second derivative does not exist there. It cannot satisfy the strong form, but it is perfectly fine in the weak form, which only needs first derivatives. Piecewise polynomials on elements of any shape can now be used to fit any geometry.
2. **Neumann conditions come for free.** The applied flux or traction appears directly in the boundary integral. Nothing special has to be done to impose it. See [below](#essential-and-natural-boundary-conditions).
3. **Discontinuous coefficients and sources are handled naturally.** Material interfaces, concentrated loads and composite parts do not need special treatment at the interface.
4. **There is a strong error theory.** The weak form leads directly to statements like "the FEM solution is the best possible approximation from the chosen functions in the energy norm" (see Galerkin orthogonality below), which in turn give convergence rates and error estimators for adaptive refinement.
5. **It is the energy statement of mechanics.** For elasticity, the weak form is the *principle of virtual work*: internal virtual work equals external virtual work for every admissible virtual displacement. Engineers already think this way.

## The Galerkin method

The weak form is still infinite-dimensional: it must hold for *every* test function. The Galerkin method makes it finite.

1. **Choose a finite set of basis functions** φ₁, …, φₙ. In FEM these are the *shape functions*: each one is 1 at its own node, 0 at all other nodes, and non-zero only on the elements touching its node. See [Elements and shape functions](elements-and-shape-functions.md).
2. **Approximate the solution** as a combination of them:

```math
u_h(x) = \sum_{j=1}^{n} u_j\, \varphi_j(x)
```

3. **Use the same functions as test functions**, v = φᵢ for i = 1, …, n. This is the defining choice of the (Bubnov-)Galerkin method.
4. **Substitute into the weak form.** For each i:

```math
\sum_{j=1}^{n} \underbrace{\left(\int_\Omega k\,\nabla\varphi_i \cdot \nabla\varphi_j\,dx\right)}_{K_{ij}} u_j \;=\; \underbrace{\int_\Omega f\,\varphi_i\,dx + \int_{\Gamma_N} g\,\varphi_i\,ds}_{f_i}
```

This is the linear system **K u = f**: K is the *stiffness matrix*, f the *load vector*. K is symmetric (because a(u, v) = a(v, u)), positive definite once the essential conditions remove rigid-body or constant modes, and sparse (because two shape functions overlap only if their nodes share an element). See [Assembly and sparsity](finite-element-method.md#assembly-and-sparsity).

**Galerkin orthogonality.** Because the exact solution u and the approximation u_h both satisfy the weak form for every test function in the finite space, their difference satisfies a(u − u_h, vₕ) = 0 for all vₕ in that space. The error is "orthogonal" to the approximation space in the energy inner product. For symmetric problems like this one, it follows that u_h is the **best approximation of u in the energy norm** among all functions in the space. (Céa's lemma gives a similar bound, up to a constant, for more general problems.) This is the foundation of FEM error estimates.

## A worked 1D example

A bar of length 1 is fixed at x = 0 and loaded by a uniform distributed load f = 1, with a free end at x = 1. With unit stiffness, the displacement u satisfies

```math
-u'' = 1, \qquad u(0) = 0 \ \text{(fixed end)}, \qquad u'(1) = 0 \ \text{(free end: zero force)} .
```

The exact solution is u(x) = x − x²/2.

**Weak form.** Multiply by v with v(0) = 0, integrate, integrate by parts:

```math
\int_0^1 u'\,v'\,dx - \big[u'\,v\big]_0^1 = \int_0^1 v\,dx .
```

At x = 0 the boundary term vanishes because v(0) = 0. At x = 1 it vanishes because u′(1) = 0. So: find u with u(0) = 0 such that ∫u′v′dx = ∫v dx for every v with v(0) = 0.

Notice what happened to the two boundary conditions. The fixed end u(0) = 0 had to be imposed on the candidate functions. The free end u′(1) = 0 was never imposed: it is satisfied by leaving the boundary term out, and the weak solution satisfies it automatically in the limit of mesh refinement.

**Galerkin with two linear elements.** Nodes at x = 0, 0.5, 1, so h = 0.5. Node 0 is fixed; the unknowns are u₁ (at 0.5) and u₂ (at 1). For one linear element the stiffness matrix is (1/h)[[1, −1], [−1, 1]] and a uniform load contributes h/2 to each of its two nodes. Assembling both elements and removing the fixed node:

```
 (1/0.5) · ┌  2  -1 ┐ ┌u1┐ = ┌ 0.5  ┐
           └ -1   1 ┘ └u2┘   └ 0.25 ┘
```

The solution is u₁ = 0.375 and u₂ = 0.5. The exact values are u(0.5) = 0.375 and u(1) = 0.5, so the nodal values are exact. (For this 1D problem, linear FEM happens to be exact at the nodes; between the nodes the linear interpolation is not.) Note that the last row has "1" instead of "2" on the diagonal: the free end only has one element attached, and no extra equation was needed to impose u′(1) = 0.

## Essential and natural boundary conditions

The example shows the general pattern:

| | Essential (Dirichlet) | Natural (Neumann; Robin) |
| --- | --- | --- |
| Prescribes | The unknown itself: displacement, temperature, velocity | Its flux: traction, heat flux, (Robin: a relation between the two) |
| Order of derivative | The unknown (order 0) | The derivative that appears in the boundary term (order 1 for second-order PDEs) |
| How it is imposed | Built into the space of trial functions; test functions vanish there | Appears by itself in the boundary integral of the weak form |
| In the linear system | Remove (or modify) those rows and columns of K | Adds to the load vector f (Robin also adds to K) |
| If you forget it | The problem may be singular (rigid-body motion) | You have imposed zero flux: a free or insulated boundary |

**Why the names?**

- **Essential** because the weak form cannot hold without them: they must be enforced *essentially*, by choice of the function space. Without enough essential conditions, an elasticity problem has rigid-body motions and K is singular.
- **Natural** because they arise *naturally* from integration by parts. A boundary where you specify nothing gets the "natural" condition of zero flux: a traction-free surface in elasticity, an insulated wall in heat transfer. That is why an un-loaded face in an FEA model behaves as free without any input.

**Rule of thumb for any PDE of order 2m:** conditions on derivatives of order less than m are essential; conditions on derivatives of order m to 2m − 1 are natural. For second-order PDEs (m = 1), conditions on the value are essential and conditions on the first derivative (the flux) are natural. For the fourth-order beam equation (m = 2), displacement and slope are essential; moment and shear force are natural.

**Robin conditions,** such as convective cooling −k ∂T/∂n = h(T − T∞), are also natural. Substituting into the boundary term adds ∫hTv ds to the left side (the stiffness matrix) and ∫hT∞v ds to the right side.

## Energy minimization: the same thing seen differently

For symmetric problems such as linear elasticity and conduction, the weak form is exactly the condition for the minimum of an energy functional:

```math
\Pi(u) = \frac{1}{2}\int_\Omega k\,|\nabla u|^2\,dx \;-\; \int_\Omega f\,u\,dx \;-\; \int_{\Gamma_N} g\,u\,ds .
```

Setting the derivative of Π in every direction v to zero gives back a(u, v) = ℓ(v). In structural mechanics this is the **principle of minimum total potential energy**: of all displacement fields that satisfy the supports, the true one minimizes strain energy minus the work done by the loads. The FEM solution minimizes the same energy over the finite set of shape-function combinations, which is why a displacement-based FE model is always too stiff (it can only find a higher energy minimum), and gets softer, approaching the exact answer, as the mesh is refined.

## Variations: Petrov-Galerkin and stabilization

- **Petrov-Galerkin** methods use *different* test and trial functions. The main application is convection-dominated flow, where standard Galerkin gives oscillations. **SUPG** (streamline-upwind Petrov-Galerkin) adds a test-function perturbation along the flow direction, which acts like upwinding in finite volumes.
- **Mixed formulations** use separate fields (displacement and pressure, velocity and pressure), which avoids volumetric locking in nearly incompressible materials and is needed for incompressible flow. The spaces must be compatible (the inf-sup or LBB condition).
- **Discontinuous Galerkin** methods use basis functions that are discontinuous between elements, joined by numerical fluxes.

## Why this matters for machine learning

- **PINNs use the strong form.** The loss is the squared PDE residual at collocation points, so the network needs second derivatives everywhere, computed by automatic differentiation. Natural boundary conditions must be added as explicit loss terms. See [PINNs](pinns.md).
- **Weak-form and energy-based networks exist.** The Deep Ritz method minimizes the energy functional above with a neural network; variational PINNs use a weak-form residual against test functions. They need fewer derivatives and treat natural conditions automatically, at the cost of numerical integration.
- **Essential conditions can be built into the network output**, just as FEM builds them into the function space: multiply the network by a function that vanishes on the boundary. See [Hard vs soft constraints](hard-vs-soft-constraints.md).
- **Understanding the energy view explains label bias.** FE training data from coarse displacement-based meshes are systematically too stiff. See [Elements and shape functions](elements-and-shape-functions.md#why-linear-tetrahedra-are-too-stiff-in-bending).

## In an interview

**Thirty-second answer to "What is the weak form, and why does FEM use it?"** "Multiply the PDE by a test function, integrate over the domain and integrate by parts. The result only needs first derivatives of the solution, and holds on average rather than pointwise. That lets us use piecewise-linear shape functions on an unstructured mesh, makes Neumann conditions appear automatically in the boundary integral, handles material jumps and point loads, and comes with a best-approximation property that underpins FEM error estimates. For elasticity it is just the principle of virtual work."

**Likely follow-ups**

- *What is the Galerkin method?* Approximate the solution with a finite set of basis functions and use the same functions as test functions; the weak form becomes K u = f.
- *Why are Dirichlet conditions "essential"?* They must be built into the function space; without them the problem is singular. Neumann conditions are "natural" because they appear by themselves from integration by parts, and a boundary with nothing specified is automatically traction-free or insulated.
- *Why is a displacement-based FE model too stiff?* It minimizes the potential energy over a restricted set of functions, so it finds a higher energy minimum than the exact solution.

## References

- [The FEniCSx tutorial: the Poisson problem and its variational formulation (Jørgen S. Dokken)](https://jsdokken.com/dolfinx-tutorial/)
- T. J. R. Hughes, *The Finite Element Method: Linear Static and Dynamic Finite Element Analysis*, Dover, 2000.
- C. Johnson, *Numerical Solution of Partial Differential Equations by the Finite Element Method*, Dover, 2009.
- [The Deep Ritz method (E & Yu)](https://arxiv.org/abs/1710.00211)
