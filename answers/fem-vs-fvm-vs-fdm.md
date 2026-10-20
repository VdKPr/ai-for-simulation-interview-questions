# FEM vs FVM vs FDM

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** All three turn a PDE into a finite algebraic system, but they start from different forms of the equation.

- The **finite difference method (FDM)** works with the PDE at points and replaces each derivative with a difference of neighbouring values on a grid. It is simple and easy to make high-order, but awkward on complex geometry.
- The **finite volume method (FVM)** works with the *integral* (conservation) form: for each cell, the flux in minus the flux out equals what accumulates. It conserves mass, momentum and energy exactly cell by cell, handles unstructured meshes and shocks robustly, and is therefore the standard in CFD.
- The **finite element method (FEM)** works with the *weak* form and approximates the solution with local basis functions on elements of any shape. It handles complex geometry and many kinds of boundary conditions naturally, has a strong error theory, and is the standard in structural mechanics.

**On this page**

- [The same equation, three ways](#the-same-equation-three-ways)
- [Comparison table](#comparison-table)
- [Where each method is used, and why](#where-each-method-is-used-and-why)
- [Related methods](#related-methods)
- [Why this matters for machine learning](#why-this-matters-for-machine-learning)
- [In an interview](#in-an-interview)

## The same equation, three ways

Take steady heat conduction in a rod: −(k u′)′ = f on 0 < x < 1, with the ends held at u = 0. Use a grid of points xᵢ spaced h apart.

**Finite differences: approximate the derivative at a point.** At each grid point, replace u″ with a central difference:

```math
-k\,\frac{u_{i-1} - 2u_i + u_{i+1}}{h^2} = f(x_i)
```

One equation per point, built from Taylor series. The unknowns are point values.

**Finite volumes: balance fluxes over a cell.** Put a small cell (a "control volume") around each point, from x_{i−1/2} to x_{i+1/2}, and integrate the equation over it. The divergence theorem turns the integral of −(k u′)′ into fluxes through the two faces:

```math
\underbrace{-k\,u'(x_{i+1/2})}_{\text{flux out, right face}} \;-\; \underbrace{\left(-k\,u'(x_{i-1/2})\right)}_{\text{flux in, left face}} \;=\; \int_{x_{i-1/2}}^{x_{i+1/2}} f\,dx
```

Approximate each face flux from the two neighbouring cell values, for example u′(x_{i+1/2}) ≈ (u_{i+1} − u_i)/h. The unknowns are cell averages. The key property: the flux leaving one cell through a face is *exactly* the flux entering its neighbour, so nothing is created or lost between cells. That is **local conservation**.

**Finite elements: require the weak form to hold for basis functions.** Write u(x) ≈ Σⱼ uⱼ φⱼ(x), where each φⱼ is a "hat" function that is 1 at node j and 0 at the other nodes. Multiply the equation by each φᵢ, integrate over the domain and integrate by parts:

```math
\sum_j u_j \int_0^1 k\,\varphi_i'\,\varphi_j' \,dx \;=\; \int_0^1 f\,\varphi_i\,dx \qquad \text{for each } i
```

The unknowns are coefficients of the basis functions (here equal to nodal values). See [The weak form and the Galerkin method](weak-form-and-galerkin.md).

**The surprise:** on a uniform 1D grid, all three give essentially the same tridiagonal matrix, (k/h²)·[−1, 2, −1], up to how the source term is sampled. The methods differ where it matters in practice: on irregular meshes, for complex geometry, in how boundary conditions and discontinuous coefficients are handled, in what is conserved, and in what error theory backs them up.

## Comparison table

| | Finite difference | Finite volume | Finite element |
| --- | --- | --- | --- |
| Starts from | PDE at points (strong form) | Integral conservation law over cells | Weak (variational) form |
| Approximates | Derivatives, by Taylor series | Fluxes through cell faces | The solution, by basis functions |
| Unknowns | Values at grid points | Cell averages (usually at cell centres) | Coefficients of basis functions (usually nodal values) |
| Mesh | Structured grids, mainly | Unstructured cells of any polyhedral shape | Unstructured elements of any shape and order |
| Conservation | Not built in | Exact, cell by cell | Global; not cell by cell by default |
| Higher order | Easy on regular grids (wider stencils) | Harder (reconstruction); usually second order | Natural (higher-degree elements) |
| Boundary conditions | Awkward on curved boundaries | Through boundary face fluxes | Natural BCs enter the weak form directly |
| Error theory | Truncation error analysis | Mostly truncation and stability analysis | Strong: a priori and a posteriori estimates |
| Typical home | Research codes, seismic, simple domains, high-order schemes | CFD: OpenFOAM, SU2, Fluent, STAR-CCM+ | Structures and heat: Abaqus, Ansys Mechanical, Nastran, CalculiX, FEniCSx |

## Where each method is used, and why

**FVM in CFD.** Fluid flow is governed by conservation laws, and flows with shocks or sharp gradients punish schemes that do not conserve exactly: a shock can end up in the wrong place. FVM conserves mass, momentum and energy cell by cell, handles arbitrary polyhedral meshes, and has a mature toolbox of upwinding and limiters for convection-dominated flow. That is why nearly every general-purpose CFD code is finite-volume.

**FEM in structural mechanics and heat transfer.** Elasticity and conduction are elliptic problems that come from minimizing an energy, and FEM's weak form *is* that energy statement. Complex geometry, mixed materials, curved boundaries, higher-order elements, shells and beams, contact, and applied tractions all fit naturally. Error estimates also tell you where to refine the mesh. That is why nearly every structural code is finite-element.

**FDM where grids are regular.** Seismic wave propagation, many weather and climate dynamical cores on regular grids, image-like domains, and teaching codes use finite differences because they are simple, fast and easy to make high-order. Complex geometry is the weakness: curved boundaries need tricks such as immersed boundaries or coordinate transformations.

These are tendencies, not laws. FEM is also used for flow (for example in some incompressible and multiphysics codes), and FVM is used for some solid mechanics problems.

## Related methods

- **Spectral methods** use global basis functions (Fourier modes, Chebyshev polynomials). For smooth solutions on simple, often periodic domains they converge extremely fast, and they rely on the FFT. The [Fourier Neural Operator](fourier-neural-operator.md) borrows this idea.
- **Discontinuous Galerkin (DG)** methods combine the element-based flexibility and high order of FEM with the flux-based conservation of FVM, at a higher cost per unknown.
- **Particle and meshless methods** such as smoothed particle hydrodynamics (SPH) for free-surface flows and the discrete element method (DEM) for granular materials. The [Graph Network-based Simulator](graph-network-simulator-gns.md) was trained on particle data of this kind.
- **Lattice Boltzmann methods** simulate the flow through particle distribution functions on a regular lattice and map very well to GPUs.

## Why this matters for machine learning

- **Where the data lives.** FEM results are usually stored at *nodes* (and stresses at integration points); FVM results are usually stored at *cell centres*. Mixing point data and cell data shifts every value by half a cell and can quietly corrupt a training set. Check the data location in the file format (VTK files label arrays as point data or cell data). See [Simulation file formats](simulation-file-formats.md).
- **What the graph looks like.** A GNN on FEM data naturally uses mesh nodes and element edges; on FVM data it may use cell centres connected through shared faces.
- **What can be learned locally.** A message-passing GNN update resembles a stencil or a flux balance: MP-PDE showed that message passing can represent classical finite-difference, finite-volume and WENO schemes. See [Message Passing Neural PDE Solvers](autoregressive-rollouts.md#message-passing-neural-pde-solvers).
- **Which errors the labels carry.** A first-order upwind scheme adds numerical diffusion; linear tetrahedra are too stiff in bending. The surrogate learns those biases along with the physics. See [When the labels have discretization error](verification-validation-and-mesh-convergence.md#when-the-labels-have-discretization-error).

## In an interview

**Thirty-second answer.** "Finite differences approximate derivatives at grid points; they are simple and high-order but need structured grids. Finite volumes integrate the conservation law over each cell and balance face fluxes, so mass, momentum and energy are conserved cell by cell; that is why CFD codes use them. Finite elements use the weak form with local basis functions on any element shape; they handle complex geometry and boundary conditions well and have a strong error theory, which is why structural codes use them."

**Likely follow-ups**

- *Why is conservation so important in CFD?* Without it, shocks and sharp fronts can move at the wrong speed and mass can leak between cells; the discrete solution can converge to the wrong answer.
- *Which one would you use for a new problem with complex geometry?* FEM for elliptic, structural or thermal problems; FVM for convection-dominated flow; and a library rather than writing your own.
- *What is the weak form?* See [The weak form](weak-form-and-galerkin.md#the-weak-form).

## References

- [CFD Python: 12 steps to Navier–Stokes (Lorena Barba)](https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/)
- [The FEniCSx tutorial (Jørgen S. Dokken)](https://jsdokken.com/dolfinx-tutorial/)
- [Message Passing Neural PDE Solvers (Brandstetter, Worrall & Welling)](https://arxiv.org/abs/2202.03376)
- H. K. Versteeg and W. Malalasekera, *An Introduction to Computational Fluid Dynamics: The Finite Volume Method*, 2nd edition, Pearson, 2007.
- T. J. R. Hughes, *The Finite Element Method: Linear Static and Dynamic Finite Element Analysis*, Dover, 2000.
