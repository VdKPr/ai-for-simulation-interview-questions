# The Finite Element Method

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** The finite element method (FEM) splits a domain into small elements, approximates the unknown field inside each element with simple polynomials (shape functions), and requires the *weak* (integral) form of the PDE to hold for those functions. Each element contributes a small matrix that links its own nodes; adding all of them into one global matrix (**assembly**) gives a large linear system **K u = f**. The global stiffness matrix is **sparse** because each node is coupled only to the nodes of the elements it touches, so almost every entry is zero. FEM handles complex geometry and mixed boundary conditions well, has a strong error theory, and is the standard method in structural mechanics and heat transfer.

**On this page**

- [The FEM pipeline](#the-fem-pipeline)
- [Element matrices](#element-matrices)
- [Assembly and sparsity](#assembly-and-sparsity)
- [How sparse, and why it matters](#how-sparse-and-why-it-matters)
- [Applying boundary conditions](#applying-boundary-conditions)
- [Solving and post-processing](#solving-and-post-processing)
- [Code: 1D FEM for the Poisson equation](#code-1d-fem-for-the-poisson-equation)
- [Code: vectorized sparse assembly](#code-vectorized-sparse-assembly)
- [Where FEM meets machine learning](#where-fem-meets-machine-learning)
- [In an interview](#in-an-interview)

## The FEM pipeline

1. **Strong form.** Start from the PDE and its boundary conditions, for example linear elasticity or steady heat conduction.
2. **Weak form.** Multiply by test functions, integrate and integrate by parts. See [The weak form](weak-form-and-galerkin.md#the-weak-form).
3. **Mesh.** Divide the geometry into elements. See [Meshing and mesh quality](meshing-and-mesh-quality.md).
4. **Shape functions.** Choose the element type and order, which fixes how the field varies inside each element. See [Elements and shape functions](elements-and-shape-functions.md).
5. **Element matrices.** For each element, compute its stiffness matrix and load vector by numerical integration.
6. **Assembly.** Add every element's contribution into the global matrix and vector.
7. **Boundary conditions.** Impose fixed values (essential conditions); applied loads and fluxes (natural conditions) are already in the load vector.
8. **Solve** K u = f, with a sparse direct or iterative solver; for nonlinear problems, repeat inside Newton iterations. See [Sparse linear solvers](sparse-linear-solvers.md) and [Newton-Raphson](newton-raphson-and-nonlinear-solvers.md).
9. **Post-process.** Compute derived quantities (strain, stress, heat flux, reaction forces) and check the result. See [Sanity checks](verification-validation-and-mesh-convergence.md#sanity-checks-before-trusting-an-fea-result).

## Element matrices

**The simplest element: a 1D bar.** A bar element of length L, cross-section A and Young's modulus E has two nodes and one axial displacement per node. With linear shape functions, its stiffness matrix is

```math
k_e = \frac{EA}{L}\begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}
```

This is just a spring of stiffness EA/L: pull node 2 by δ while holding node 1, and the force is (EA/L)·δ. The matrix is symmetric, and its rows sum to zero because moving both nodes together (a rigid translation) costs no energy.

**In general,** for an element with shape-function derivatives collected in a strain-displacement matrix B and a material matrix D:

```math
k_e = \int_{\Omega_e} B^{\mathsf T} D\, B \, d\Omega
```

computed with Gauss quadrature on the reference element. A linear tetrahedron in 3D elasticity (4 nodes × 3 displacement components) gives a 12 × 12 matrix; a 10-node quadratic tetrahedron gives 30 × 30; an 8-node hexahedron gives 24 × 24. A 2D example with full code is in [Linear vs quadratic triangles on a cantilever](elements-and-shape-functions.md#code-linear-vs-quadratic-triangles-on-a-cantilever).

## Assembly and sparsity

**Assembly** puts every element matrix into the global matrix. Each element knows which global nodes it uses, through a **connectivity array** (also called the element-to-node table):

```
element   nodes (global numbers)
   0      [0, 1]
   1      [1, 2]
   2      [2, 3]
```

For each element, the entry (a, b) of its local matrix is **added** to the entry (global[a], global[b]) of K. Entries from different elements that share a node accumulate. For three bar elements of unit stiffness:

```
element 0 adds to rows/cols {0,1}     element 1 adds to {1,2}     element 2 adds to {2,3}

            ┌  1 -1  0  0 ┐
   K   =    │ -1  2 -1  0 │      node 1 is shared by elements 0 and 1: 1 + 1 = 2
            │  0 -1  2 -1 │      node 2 is shared by elements 1 and 2: 1 + 1 = 2
            └  0  0 -1  1 ┘
```

**Why K is sparse.** The entry K_ij is an integral of the product of the gradients of shape functions i and j. Each shape function is non-zero only on the elements around its own node. So K_ij is non-zero only if nodes i and j belong to a common element. Every other entry is exactly zero. In a 3D mesh with a million nodes, each node shares elements with only a few dozen other nodes, so each row of K has a few dozen non-zeros out of millions of columns.

**Other properties of K** (for linear elasticity and conduction):

- **Symmetric**, because the weak form is symmetric in u and v.
- **Positive definite** once enough essential boundary conditions remove the rigid-body motions. Without them, K is singular: the classic "the solver says the stiffness matrix is singular" error from an under-constrained model.
- **Banded** if the nodes are numbered so that neighbours have close numbers. Reordering algorithms (reverse Cuthill-McKee, nested dissection) reduce the bandwidth and the fill-in of direct solvers.

## How sparse, and why it matters

**A rough estimate.** Take a 3D elasticity model with 1 million nodes, linear tetrahedra and 3 unknowns per node: 3 million unknowns. In a typical tetrahedral mesh each node is connected to roughly 15 neighbours. Each row of K then has about 3 × (1 + 15) ≈ 48 non-zeros.

| Storage | Size |
| --- | --- |
| Dense: (3 × 10⁶)² entries × 8 bytes | about 72 terabytes |
| Sparse (CSR): 3 × 10⁶ rows × 48 non-zeros × (8 bytes value + 4 bytes column index) | about 1.7 gigabytes |

These numbers are rough (they depend on the mesh), but the conclusion is not: **simulation is only possible because K is sparse.** Every part of the toolchain (storage formats, solvers, preconditioners, parallel partitioning) is built around sparsity. Quadratic elements couple more nodes, so their rows are longer and the matrix is denser, which is part of their extra cost.

The same structure appears in ML for simulation: a mesh *is* a sparse graph, and graph neural networks store edges as sparse index arrays for exactly the same reason. See [GNNs and MeshGraphNets](gnns-and-meshgraphnets.md).

## Applying boundary conditions

- **Natural conditions** (applied tractions, heat fluxes, pressures) are integrated into the load vector f during assembly. Nothing else is needed.
- **Essential conditions** (fixed displacement, fixed temperature) are imposed on the system. Common ways:
  - **Elimination:** remove the rows and columns of the fixed unknowns, moving their known values to the right-hand side. Exact, and what the code below does.
  - **Row replacement:** replace the row with the identity and set the right-hand side to the prescribed value (keeps the matrix size; symmetry is restored by also modifying the columns).
  - **Penalty method:** add a very large stiffness to the diagonal. Simple, but it can harm conditioning.
  - **Lagrange multipliers:** add the constraint as extra equations. Exact, and the multiplier is the reaction force, but the system becomes indefinite.
- **Multi-point constraints** (rigid links, distributing couplings, symmetry) relate several unknowns. See [Bolt loads and remote loads](boundary-conditions.md#bolt-loads-and-remote-loads).

## Solving and post-processing

The solver returns nodal displacements (or temperatures). Everything else is computed from them:

- **Strains** from derivatives of the shape functions, evaluated at the Gauss points, where they are most accurate.
- **Stresses** from strains through the material law, then extrapolated to nodes and often **averaged** across elements for smooth contour plots. A big difference between averaged and unaveraged values at a hot spot signals a mesh that is too coarse there.
- **Reaction forces** at constraints: K u − f evaluated at the fixed unknowns. They should balance the applied loads.
- **Derived quantities:** von Mises stress, safety factors, heat flux, strain energy.

## Code: 1D FEM for the Poisson equation

The equation −u″ = f on (0, 1) with u(0) = u(1) = 0 models a bar under distributed axial load, or steady conduction in a rod. This version uses a dense matrix and an explicit element loop so that every step is visible.

```python
import numpy as np

def fem_1d_poisson(n_el, f):
    """Solve -u'' = f on (0, 1) with u(0) = u(1) = 0 using linear elements."""
    x = np.linspace(0.0, 1.0, n_el + 1)
    n = n_el + 1
    K = np.zeros((n, n))
    F = np.zeros(n)
    for e in range(n_el):
        idx = [e, e + 1]                                          # connectivity of element e
        h = x[e + 1] - x[e]
        ke = (1.0 / h) * np.array([[1.0, -1.0], [-1.0, 1.0]])   # element stiffness
        xm = 0.5 * (x[e] + x[e + 1])
        fe = f(xm) * h / 2.0 * np.ones(2)                         # midpoint-rule load
        K[np.ix_(idx, idx)] += ke                                 # assembly: add, do not overwrite
        F[idx] += fe
    inner = np.arange(1, n - 1)                                   # Dirichlet: eliminate boundary dofs
    u = np.zeros(n)
    u[inner] = np.linalg.solve(K[np.ix_(inner, inner)], F[inner])
    return x, u

x, u = fem_1d_poisson(32, lambda x: np.pi**2 * np.sin(np.pi * x))
print(np.max(np.abs(u - np.sin(np.pi * x))))   # error falls about 4x each time the mesh is halved
```

Running this with 8, 16 and 32 elements gives maximum nodal errors of about 6.5e-3, 1.6e-3 and 4.0e-4: second-order convergence at the nodes.

## Code: vectorized sparse assembly

Real codes never build a dense K. The standard trick is to collect every element's entries as (row, column, value) triplets and let a sparse COO-to-CSR conversion **sum the duplicates**: that sum *is* the assembly.

```python
import numpy as np
import scipy.sparse as sp

def assemble_1d_sparse(n_el):
    """Global stiffness matrix of n_el linear bar elements on (0, 1), in CSR format."""
    x = np.linspace(0.0, 1.0, n_el + 1)
    conn = np.column_stack([np.arange(n_el), np.arange(1, n_el + 1)])   # element -> its 2 nodes
    h = np.diff(x)
    ke = np.array([[1.0, -1.0], [-1.0, 1.0]])
    rows = np.repeat(conn, 2, axis=1).ravel()        # per element: [i, i, j, j]
    cols = np.tile(conn, 2).ravel()                  # per element: [i, j, i, j]
    vals = (ke.ravel()[None, :] / h[:, None]).ravel()
    return sp.coo_matrix((vals, (rows, cols)), shape=(n_el + 1, n_el + 1)).tocsr()  # duplicates summed

K_sparse = assemble_1d_sparse(32)
print(K_sparse.nnz, "non-zeros out of", 33 * 33)     # 97 = 3 * 33 - 2: tridiagonal
```

The same pattern (compute all element matrices in one batched operation, then one sparse conversion) scales to millions of 3D elements, and it is the same scatter-add that a GNN uses to sum messages into nodes.

## Where FEM meets machine learning

- **Meshes become graphs.** Nodes and element edges define the graph a GNN works on; the sparsity of K and the edge list of a mesh graph are the same structure.
- **Message passing resembles assembly.** Edge messages computed from neighbouring nodes and summed into each node look much like element contributions added into rows of K.
- **FEM produces the training labels,** with their discretization bias. See [When the labels have discretization error](verification-validation-and-mesh-convergence.md#when-the-labels-have-discretization-error).
- **Differentiable FEM** (for example JAX-based FEM codes, or FEniCS with dolfin-adjoint) gives gradients of FEM results with respect to inputs, for optimization and hybrid models. See [Differentiable simulators](differentiable-physics-and-hybrid-solvers.md#differentiable-simulators).

## In an interview

**Thirty-second answer to "How is the global stiffness matrix assembled, and why is it sparse?"** "Each element has a small stiffness matrix, for example 12 by 12 for a linear tet in 3D elasticity, computed by integrating Bᵀ D B. The connectivity table maps its local nodes to global node numbers, and each entry is added into the corresponding entry of the global matrix, so contributions from elements sharing a node accumulate. K is sparse because an entry is non-zero only if the two nodes share an element: each row has a few dozen non-zeros out of millions of columns. That is what makes it storable, and what sparse solvers exploit."

**Likely follow-ups**

- *Why is K singular before applying boundary conditions?* Rigid-body motions cost no strain energy, so they are in the null space of K.
- *Why do quadratic elements cost more beyond the extra nodes?* Each node couples to more neighbours, so rows are longer, the matrix is denser and factorization fill-in grows.
- *How do you assemble efficiently in Python?* Compute all element matrices in a batch, build (row, column, value) arrays, and convert COO to CSR so duplicates are summed.

## References

- [The FEniCSx tutorial (Jørgen S. Dokken)](https://jsdokken.com/dolfinx-tutorial/)
- T. J. R. Hughes, *The Finite Element Method: Linear Static and Dynamic Finite Element Analysis*, Dover, 2000.
- [SciPy sparse matrices documentation](https://docs.scipy.org/doc/scipy/reference/sparse.html)
- [Learning Mesh-Based Simulation with Graph Networks (Pfaff et al.)](https://arxiv.org/abs/2010.03409)
