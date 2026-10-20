# Elements and Shape Functions

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** In FEM the solution inside each element is interpolated from its nodal values by **shape functions**: simple polynomials, one per node, that equal 1 at their own node and 0 at the other nodes, and add up to 1 everywhere. **Linear** elements have nodes only at the corners, so the field varies linearly inside each element and its gradient (strain, heat flux) is constant or nearly so. **Quadratic** elements add mid-side nodes, so the field can curve and the strain can vary linearly inside an element. Quadratic elements are much more accurate per degree of freedom for smooth problems and bending, and they follow curved boundaries. Linear tetrahedra in particular are far **too stiff in bending** (a form of *shear locking*), because a constant-strain element cannot represent the linear strain variation through the thickness of a bending member.

**On this page**

- [Shape functions](#shape-functions)
- [Linear vs quadratic elements](#linear-vs-quadratic-elements)
- [Element families](#element-families)
- [Isoparametric mapping and numerical integration](#isoparametric-mapping-and-numerical-integration)
- [How the error falls with mesh size and element order](#how-the-error-falls-with-mesh-size-and-element-order)
- [Why linear tetrahedra are too stiff in bending](#why-linear-tetrahedra-are-too-stiff-in-bending)
- [Code: linear vs quadratic triangles on a cantilever](#code-linear-vs-quadratic-triangles-on-a-cantilever)
- [Volumetric locking and other pathologies](#volumetric-locking-and-other-pathologies)
- [Why this matters for machine learning](#why-this-matters-for-machine-learning)
- [In an interview](#in-an-interview)

## Shape functions

Inside an element, the unknown field (displacement, temperature) is written as a weighted sum of the nodal values:

```math
u(x) \approx \sum_{a=1}^{n_{\text{nodes}}} N_a(x)\, u_a
```

Each **shape function** N_a belongs to one node. Shape functions have three defining properties:

1. **Kronecker delta:** N_a is 1 at node a and 0 at every other node, so the coefficients u_a are exactly the nodal values.
2. **Partition of unity:** the N_a add up to 1 everywhere in the element, so a constant field (a rigid translation) is represented exactly.
3. **Completeness:** they can represent any linear field exactly, so constant strain states (the test every element must pass, the *patch test*) are reproduced.

Assembled over the mesh, the shape functions of all elements sharing a node form that node's global basis function: the "hat" function of the [Galerkin method](weak-form-and-galerkin.md#the-galerkin-method).

**1D linear element** (nodes at ξ = −1 and ξ = +1 in local coordinates):

```math
N_1(\xi) = \tfrac{1}{2}(1 - \xi), \qquad N_2(\xi) = \tfrac{1}{2}(1 + \xi)
```

**1D quadratic element** (nodes at ξ = −1, 0, +1):

```math
N_1 = \tfrac{1}{2}\xi(\xi - 1), \qquad N_2 = 1 - \xi^2, \qquad N_3 = \tfrac{1}{2}\xi(\xi + 1)
```

Check the Kronecker property: at ξ = 0, N₁ = 0, N₂ = 1, N₃ = 0. And N₁ + N₂ + N₃ = 1 for every ξ.

**Triangles** are easiest in *area (barycentric) coordinates* L₁, L₂, L₃, which add up to 1 and are each 1 at one corner. The linear triangle uses N_a = L_a. The quadratic 6-node triangle uses N_a = L_a(2L_a − 1) at the corners and 4L_aL_b at the mid-side nodes.

## Linear vs quadratic elements

| | Linear (first-order) | Quadratic (second-order) |
| --- | --- | --- |
| Nodes | Corners only (tet: 4, hex: 8, triangle: 3) | Corners plus mid-side nodes (tet: 10, hex: 20 or 27, triangle: 6) |
| Field inside the element | Linear (tet, triangle) or bilinear/trilinear (quad, hex) | Quadratic |
| Strain and stress inside | Constant in tets and triangles | Varies linearly |
| Element edges | Straight | Can be curved, so they follow curved geometry |
| Accuracy per degree of freedom | Lower; needs fine meshes, especially in bending | Much higher for smooth problems |
| Cost per element | Lower; fewer integration points | Higher; more nodes, larger element matrices, wider sparsity |
| Typical use | Explicit dynamics (crash), contact-heavy models, very large meshes, CFD | Implicit structural analysis of stress and stiffness, especially with tets |

**A useful rule of thumb in structural FEA:** if the mesh is tetrahedral, use quadratic (10-node) tetrahedra. Most automatic meshers produce tets, and linear tets are too stiff for most stress work unless the mesh is very fine.

**Why stresses look "blocky" with linear elements.** Stress is computed from the derivatives of the displacement. With linear tets, strain is constant inside each element and jumps from element to element. Post-processors average these jumps at the nodes to draw smooth contours, which can hide how coarse the solution really is. Comparing *averaged* and *unaveraged* stresses at a hot spot is a quick check of mesh adequacy.

## Element families

| Element | Shape | Where it is used |
| --- | --- | --- |
| Tetrahedron (4 or 10 nodes) | 3D | Automatic meshing of complex solid geometry |
| Hexahedron, "brick" (8, 20 or 27 nodes) | 3D | Swept or mapped meshes of simple solids; more accurate per element than tets |
| Wedge / prism (6 or 15 nodes) | 3D | Layers near walls (CFD boundary layers), transitions |
| Pyramid (5 or 13 nodes) | 3D | Transition between hexes and tets |
| Shell (3, 4, 6 or 8 nodes) | 2D surface in 3D | Thin-walled structures: car bodies, aircraft skins, sheet metal |
| Beam (2 or 3 nodes) | 1D line in 3D | Frames, stiffeners, bolts |
| Solid-shell, membrane, truss, spring, mass, rigid elements | Special | Idealizations and connections |

Shells and beams are *structural* elements: they build the thin-direction behaviour (bending through the thickness) into the element instead of meshing the thickness with solids. A car body-in-white is mostly shells; meshing it with solid tets would need several elements through each sheet's thickness.

## Isoparametric mapping and numerical integration

Real elements have arbitrary shapes. FEM handles this with **isoparametric mapping**: each element is the image of a fixed reference element (a unit triangle, a cube from −1 to 1) under a map defined by the *same* shape functions as the field:

```math
x(\xi) = \sum_a N_a(\xi)\, x_a, \qquad u(\xi) = \sum_a N_a(\xi)\, u_a
```

The **Jacobian** of this map, J = ∂x/∂ξ, converts derivatives and integrals from reference to physical coordinates. If an element is badly distorted, det J becomes small or negative at some points and the element is invalid; mesh-quality checks report this as the "Jacobian ratio". See [Meshing and mesh quality](meshing-and-mesh-quality.md).

Element integrals are computed with **Gauss quadrature**: a weighted sum of the integrand at a few special points. A 1D Gauss rule with n points integrates polynomials of degree 2n − 1 exactly. Using *fewer* points than needed for exact integration is called **reduced integration**. It is cheaper and, for some elements, cures locking, but it can allow zero-energy deformation modes ("hourglass modes") that need extra control.

## How the error falls with mesh size and element order

For a smooth solution and elements of polynomial degree p, standard FEM theory gives:

- error in the **energy norm** (which measures strains and stresses): proportional to **hᵖ**;
- error in the **L² norm** (which measures displacements or temperatures): proportional to **hᵖ⁺¹**.

So halving the element size reduces the stress error by about 2× with linear elements and about 4× with quadratic elements. Near singularities (sharp re-entrant corners, crack tips) the solution is not smooth and the rate drops for every element order; there, local refinement matters more than element order. See [Stress concentrations and singularities](stress-concentrations-and-singularities.md).

There are two ways to improve accuracy: make elements smaller (**h-refinement**) or raise the polynomial order (**p-refinement**). Combining them (**hp-FEM**) can converge very fast for problems with both smooth regions and singularities.

## Why linear tetrahedra are too stiff in bending

**The mechanism.** In pure bending, strain varies linearly through the thickness: compression on one side, tension on the other, zero at the neutral axis. A linear tetrahedron (or a linear triangle in 2D) has **constant strain** inside the element. It cannot bend smoothly. To approximate the curvature, the element must also deform in *shear*, which costs strain energy that a real bending beam does not have. The model therefore needs more force for the same deflection: it is too stiff, and deflections are under-predicted. Fully integrated linear quadrilaterals and hexahedra show a similar spurious-shear effect. The general name is **shear locking**.

**How large is the error?** The code below solves a 2D plane-stress cantilever (length 10, height 1, tip load) with linear and quadratic triangles and compares the tip deflection with a converged solution:

| Mesh (along × through thickness) | Linear 3-node triangles | Quadratic 6-node triangles |
| --- | --- | --- |
| 10 × 1 | 23.0% of the correct deflection (40 unknowns) | 99.2% (120 unknowns) |
| 20 × 2 | 53.8% (120 unknowns) | 99.8% (400 unknowns) |
| 40 × 4 | 82.2% (400 unknowns) | 100.0% (1,440 unknowns) |

With the *same* 120 unknowns, linear triangles predict about half the true deflection; quadratic triangles are within 1%. With 400 unknowns, linear triangles are still 18% too stiff. Linear tetrahedra in 3D behave the same way.

**Remedies**

- **Quadratic elements** (10-node tets, 20-node hexes): the standard fix.
- **Hexahedra with reduced integration** and hourglass control, or *incompatible-mode* elements designed for bending.
- **More elements through the thickness** if linear elements must be used (explicit crash codes often use linear elements with reduced integration for speed and robustness).
- **Shell or beam elements** for thin structures, so bending is built into the element.

## Code: linear vs quadratic triangles on a cantilever

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def shape_derivs(order, xi, eta):
    """Derivatives of the triangle shape functions w.r.t. (xi, eta); shape (2, n_nodes)."""
    if order == 1:                                   # 3-node linear triangle
        return np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
    L1, L2, L3 = 1 - xi - eta, xi, eta               # 6-node quadratic triangle
    return np.array([
        [-(4*L1 - 1), 4*L2 - 1, 0.0, 4*(L1 - L2), 4*L3, -4*L3],
        [-(4*L1 - 1), 0.0, 4*L3 - 1, -4*L2, 4*L2, 4*(L1 - L3)]])

def gauss_points(order):
    if order == 1:
        return [((1/3, 1/3), 0.5)]                   # constant strain: one point is exact
    return [((1/6, 1/6), 1/6), ((2/3, 1/6), 1/6), ((1/6, 2/3), 1/6)]

def cantilever_mesh(L, H, nx, ny, order):
    m = order
    xs, ys = np.linspace(0, L, m*nx + 1), np.linspace(0, H, m*ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    nid = lambda i, j: i*(m*ny + 1) + j
    tris = []
    for i in range(nx):
        for j in range(ny):
            i0, j0 = m*i, m*j
            if order == 1:
                a, b, c, d = nid(i0, j0), nid(i0+1, j0), nid(i0+1, j0+1), nid(i0, j0+1)
                tris += [[a, b, c], [a, c, d]]
            else:                                    # corners first, then mid-side nodes
                a, b, c, d = nid(i0, j0), nid(i0+2, j0), nid(i0+2, j0+2), nid(i0, j0+2)
                tris.append([a, b, c, nid(i0+1, j0), nid(i0+2, j0+1), nid(i0+1, j0+1)])
                tris.append([a, c, d, nid(i0+1, j0+1), nid(i0+1, j0+2), nid(i0, j0+1)])
    return nodes, np.array(tris)

def tip_deflection(nx, ny, order, L=10.0, H=1.0, E=1000.0, nu=0.3, P=1.0):
    nodes, tris = cantilever_mesh(L, H, nx, ny, order)
    D = E / (1 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])  # plane stress
    rows, cols, vals = [], [], []
    for tri in tris:
        xy = nodes[tri]
        ke = np.zeros((2*len(tri), 2*len(tri)))
        for (xi, eta), w in gauss_points(order):
            dN = shape_derivs(order, xi, eta)
            J = dN @ xy                              # Jacobian of the isoparametric map
            dNdx = np.linalg.solve(J, dN)            # shape-function derivatives in x, y
            B = np.zeros((3, 2*len(tri)))            # strain-displacement matrix
            B[0, 0::2], B[1, 1::2] = dNdx[0], dNdx[1]
            B[2, 0::2], B[2, 1::2] = dNdx[1], dNdx[0]
            ke += B.T @ D @ B * np.linalg.det(J) * w
        dofs = np.ravel(np.column_stack([2*tri, 2*tri + 1]))
        rows.append(np.repeat(dofs, len(dofs)))
        cols.append(np.tile(dofs, len(dofs)))
        vals.append(ke.ravel())
    n = 2 * len(nodes)                               # assembly: duplicates are summed
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    f = np.zeros(n)                                  # uniform shear traction on the tip, total P
    tip = np.where(np.isclose(nodes[:, 0], L))[0]
    tip = tip[np.argsort(nodes[tip, 1])]
    weights = np.zeros(len(tip))
    for k in range(0, len(tip) - 1, order):
        he = nodes[tip[k + order], 1] - nodes[tip[k], 1]
        weights[k:k + order + 1] += he * (np.array([0.5, 0.5]) if order == 1 else np.array([1/6, 4/6, 1/6]))
    f[2*tip + 1] = -P * weights / weights.sum()
    fixed = np.where(np.isclose(nodes[:, 0], 0.0))[0]      # clamped root
    free = np.setdiff1d(np.arange(n), np.concatenate([2*fixed, 2*fixed + 1]))
    u = np.zeros(n)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), f[free])
    return -u[2*tip + 1].mean(), len(free)

ref, _ = tip_deflection(160, 16, order=2)             # converged 2D solution, about 4.024
for nx, ny in [(10, 1), (20, 2), (40, 4)]:
    for order, name in [(1, "linear   "), (2, "quadratic")]:
        d, ndof = tip_deflection(nx, ny, order)
        print(f"{nx:3d} x {ny:2d} {name}: {100 * d / ref:5.1f}% of converged deflection, {ndof} unknowns")
```

The converged 2D deflection (4.024) agrees with Timoshenko beam theory including shear deformation (4.031) to within 0.2%; the small difference comes from the fully clamped root.

## Volumetric locking and other pathologies

- **Volumetric locking.** For nearly incompressible materials (rubber, or metals during plastic flow, where Poisson's ratio approaches 0.5), the element must deform without changing volume. Low-order elements have too few deformation modes that satisfy this, so they lock and become far too stiff. Remedies: mixed displacement-pressure (u-p) formulations, selective or reduced integration, B-bar methods, hybrid elements.
- **Hourglassing.** Reduced-integration linear hexes can deform in zero-energy patterns that look like an hourglass. Explicit codes add hourglass control and report the artificial energy it introduces, which should stay small compared with the internal energy.
- **Distorted elements.** Very skewed or stretched elements lose accuracy and can have negative Jacobians. See [Meshing and mesh quality](meshing-and-mesh-quality.md).

## Why this matters for machine learning

- **Training labels carry element bias.** If a structural dataset was generated with coarse linear tetrahedra, every stress and displacement in it is biased towards being too stiff, and a surrogate trained on it will faithfully reproduce the bias. Ask how the data was meshed before trusting its labels. See [When the labels have discretization error](verification-validation-and-mesh-convergence.md#when-the-labels-have-discretization-error).
- **Mixed element types in one dataset.** Datasets that combine tets, hexes and shells need the model (and the graph construction) to handle different node counts and connectivities. Element type can be a useful input feature.
- **Mid-side nodes.** Quadratic meshes have mid-side nodes that are not corners of any face. Mesh-to-graph conversion must decide whether to keep them as graph nodes. See [Mesh data processing](mesh-data-processing.md#element-types-and-the-bounding-box).
- **Stress at integration points.** Stresses are computed at Gauss points and extrapolated or averaged to nodes. Training on averaged nodal stress smooths the peaks.

## In an interview

**Thirty-second answer to "What are shape functions, and linear vs quadratic elements?"** "Shape functions interpolate the field inside an element from its nodal values: each is one at its own node and zero at the others, and they sum to one. Linear elements have only corner nodes, so strain is constant inside a tet; quadratic elements add mid-side nodes, so strain varies linearly and edges can curve. Quadratic elements are far more accurate per unknown, especially in bending; linear tets are notoriously too stiff, so for tet meshes in stress analysis I would use 10-node tets."

**Likely follow-ups**

- *Why are linear tets too stiff?* Constant strain cannot represent the linear strain variation of bending; the element shears instead and locks. In the cantilever test above, linear triangles give about half the correct deflection with the same number of unknowns as quadratic triangles that are within 1%.
- *What is reduced integration?* Using fewer Gauss points than exact integration needs; cheaper and less locking, but can allow hourglass modes.
- *What convergence rate do you expect?* Energy-norm error proportional to hᵖ and L² error to hᵖ⁺¹ for smooth problems.

## References

- T. J. R. Hughes, *The Finite Element Method: Linear Static and Dynamic Finite Element Analysis*, Dover, 2000.
- R. D. Cook, D. S. Malkus, M. E. Plesha and R. J. Witt, *Concepts and Applications of Finite Element Analysis*, 4th edition, Wiley, 2002.
- [The FEniCSx tutorial (Jørgen S. Dokken)](https://jsdokken.com/dolfinx-tutorial/)
- [SimJEB dataset paper: first-order tetrahedral meshes and connector modelling (Whalen et al.)](https://arxiv.org/abs/2105.03534)
