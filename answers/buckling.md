# Buckling

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Buckling is a sudden loss of stability under compression: above a critical load, a slender or thin-walled structure can no longer stay straight or flat and deflects sideways into a different shape, often at stresses far below yield. A linear buckling analysis is an **eigenvalue problem** because compressive stress *reduces* the effective bending stiffness. The total stiffness is the material stiffness K plus a geometric (stress) stiffness K_G proportional to the load, and the structure becomes unstable at the load factor λ where K + λK_G becomes singular: (K + λK_G)φ = 0. The eigenvalues are the critical load multipliers and the eigenvectors are the buckling mode shapes. For a pinned column, the first eigenvalue is Euler's load P_cr = π²EI/L². Real structures buckle earlier than linear analysis predicts because of imperfections, sometimes much earlier for thin shells.

**On this page**

- [What buckling is](#what-buckling-is)
- [Euler buckling of a column](#euler-buckling-of-a-column)
- [Why linear buckling is an eigenvalue problem](#why-linear-buckling-is-an-eigenvalue-problem)
- [How a linear buckling analysis is run](#how-a-linear-buckling-analysis-is-run)
- [Code: column buckling as an eigenvalue problem](#code-column-buckling-as-an-eigenvalue-problem)
- [Limits of linear buckling: imperfections and nonlinear analysis](#limits-of-linear-buckling-imperfections-and-nonlinear-analysis)
- [Why this matters for machine learning](#why-this-matters-for-machine-learning)
- [In an interview](#in-an-interview)

## What buckling is

Push on the ends of a plastic ruler. At first it stays straight and shortens slightly. At a certain force it suddenly bows sideways, and a little more force makes it bow a lot. Nothing has yielded; the straight shape simply stopped being stable.

**Where buckling governs design:**

- slender columns and struts, frame members, truss members in compression;
- thin plates in compression or shear (aircraft skin panels, ship decks, webs of beams);
- thin-walled shells under axial compression or external pressure (drinks cans, rocket stages, submarine hulls, silos);
- beams bending about their strong axis, which can twist sideways (lateral-torsional buckling);
- thermal buckling of constrained parts heated up (railway tracks in summer);
- wrinkling of thin films and sheets in forming.

The danger is that buckling can be sudden, can happen far below the yield stress, and can lead to collapse with little warning.

## Euler buckling of a column

For an ideal, perfectly straight elastic column with pinned ends, the critical (Euler) load is

```math
P_{\text{cr}} = \frac{\pi^2 E I}{(K L)^2}
```

where E is Young's modulus, I the second moment of area about the weakest axis, L the length and K the **effective length factor** that depends on the end conditions:

| End conditions | K | P_cr relative to pinned-pinned |
| --- | --- | --- |
| Pinned-pinned | 1.0 | 1 |
| Fixed-free (flagpole) | 2.0 | 1/4 |
| Fixed-pinned | about 0.7 | about 2 |
| Fixed-fixed | 0.5 | 4 |

**Two lessons from the formula:**

1. **Stiffness matters, strength does not.** P_cr depends on E and I, not on the yield strength. A high-strength steel column buckles at the same load as a mild steel one of the same shape.
2. **Length matters a lot.** Doubling the length cuts the critical load by four. So does freeing one end (K = 2).

The critical *stress* is σ_cr = π²E/(KL/r)², where r = √(I/A) is the radius of gyration and KL/r the **slenderness ratio**. For stocky columns, σ_cr would exceed the yield stress, so they fail by yielding instead; intermediate columns fail by an interaction of both, which design codes handle with column curves.

## Why linear buckling is an eigenvalue problem

**The key physical idea: compression reduces bending stiffness.** Think of a guitar string: tension makes it stiffer against sideways deflection. Compression does the opposite. When a column carrying a compressive force P deflects sideways, the axial force acting through the deflected shape creates an extra moment that pushes it further. That is a *negative* stiffness proportional to P.

**In finite element form:**

- K is the ordinary (material) stiffness matrix.
- K_G is the **geometric stiffness** (also called stress or initial-stress stiffness) computed from the internal stresses of a reference load. It depends on the stress level linearly: at load λ·P_ref, the geometric stiffness is λ·K_G.
- The tangent stiffness at load factor λ is K + λK_G. For compression, K_G reduces it.

The structure is stable as long as K + λK_G is positive definite. **Buckling happens at the smallest λ for which a non-zero deflection φ can exist with no extra load:**

```math
(K + \lambda K_G)\,\varphi = 0
```

That is a generalized eigenvalue problem:

- each **eigenvalue λᵢ** is a *buckling load factor*: the multiple of the reference load at which the structure buckles in mode i;
- each **eigenvector φᵢ** is the *mode shape*; its amplitude is undetermined (linear buckling says *that* the structure can deflect in that shape, not how much);
- only the lowest eigenvalue normally matters for design: λ₁ = 2.5 means the structure buckles at 2.5 times the applied load.

For the continuous column, the same reasoning gives the differential equation EI w″ + P w = 0. Its non-trivial solutions with w = 0 at both ends are w = sin(nπx/L), which exist only for P = n²π²EI/L²: an eigenvalue problem again, with Euler's load as the first eigenvalue.

This is mathematically the same structure as [modal analysis](modal-analysis.md), where (K − ω²M)φ = 0: the mass matrix is replaced by the geometric stiffness, and the eigenvalue is a load multiplier instead of a squared frequency.

## How a linear buckling analysis is run

1. **Static pre-buckling analysis** under a reference load: compute the stresses with a linear static solve.
2. **Build the geometric stiffness** K_G from those stresses.
3. **Solve the eigenvalue problem** for the lowest few eigenvalues and modes (with a sparse eigensolver such as Lanczos or subspace iteration).
4. **Interpret:** λ₁ is the buckling load factor of the reference load. Check the mode shape: is it a realistic global mode, or a local mode of some small, poorly supported feature?
5. **Compare** with the required factor, usually with a knock-down for imperfections.

Two common gotchas: a *negative* eigenvalue means the structure would buckle if the load were reversed; and if the reference load includes parts that do not scale (such as a bolt preload), the analysis must separate the constant load from the scaled one.

## Code: column buckling as an eigenvalue problem

Euler-Bernoulli beam elements (deflection and rotation at each node), with the element stiffness and the consistent geometric stiffness for a unit compressive load. The eigenvalues are the critical loads.

```python
import numpy as np
import scipy.linalg as sla

def column_buckling_loads(n_el, EI=1.0, L=1.0, supports="pinned-pinned", n_modes=3):
    """Linear buckling of a column with beam elements: (K - P K_G) phi = 0 for compressive P."""
    le = L / n_el
    k = EI / le**3 * np.array([[12, 6*le, -12, 6*le], [6*le, 4*le**2, -6*le, 2*le**2],
                               [-12, -6*le, 12, -6*le], [6*le, 2*le**2, -6*le, 4*le**2]])
    kg = 1.0 / (30 * le) * np.array([[36, 3*le, -36, 3*le], [3*le, 4*le**2, -3*le, -le**2],
                                     [-36, -3*le, 36, -3*le], [3*le, -le**2, -3*le, 4*le**2]])
    ndof = 2 * (n_el + 1)                                # (deflection, rotation) per node
    K, KG = np.zeros((ndof, ndof)), np.zeros((ndof, ndof))
    for e in range(n_el):
        d = slice(2*e, 2*e + 4)
        K[d, d] += k
        KG[d, d] += kg                                   # geometric stiffness per unit axial load
    fixed = [0, ndof - 2] if supports == "pinned-pinned" else [0, 1]   # fixed-free: base clamped
    free = [i for i in range(ndof) if i not in fixed]
    loads = sla.eigh(K[np.ix_(free, free)], KG[np.ix_(free, free)], eigvals_only=True)
    return loads[:n_modes]                               # critical loads, lowest first

euler = np.pi**2                                         # pi^2 EI / L^2 with EI = L = 1
for supports, factor in [("pinned-pinned", 1.0), ("fixed-free", 0.25)]:
    P = column_buckling_loads(20, supports=supports)
    print(f"{supports:14s} FE: {np.round(P, 3)}   Euler first mode: {factor * euler:.3f}")
conv = [round(float(column_buckling_loads(n)[0]), 5) for n in (2, 4, 8)]
print("mesh convergence (pinned-pinned, first mode):", conv, "exact", round(euler, 5))
```

Output:

```
pinned-pinned  FE: [ 9.87  39.479 88.832]   Euler first mode: 9.870
fixed-free     FE: [ 2.467 22.207 61.687]   Euler first mode: 2.467
mesh convergence (pinned-pinned, first mode): [9.94385, 9.87466, 9.86993] exact 9.8696
```

The pinned-pinned loads follow n²π² (9.87, 39.48, 88.83), the fixed-free first load is π²/4, and the first eigenvalue converges to Euler's value from above as the mesh is refined.

## Limits of linear buckling: imperfections and nonlinear analysis

Linear buckling assumes a **perfect** structure and linear behaviour up to the critical load. Reality differs in three ways:

1. **Imperfections.** Real columns are slightly crooked, loads are slightly off-centre, shells have small dents and thickness variations. An imperfect column does not bifurcate suddenly; it bends progressively and its load-deflection curve approaches the critical load. For **thin shells under axial compression**, the effect is dramatic: measured buckling loads can be a fraction of the linear prediction, so design guidelines apply large empirical *knock-down factors*.
2. **Plasticity.** If the material yields before or during buckling, the stiffness drops and the collapse load falls. Linear buckling of a stocky part can wildly over-predict its strength.
3. **Post-buckling behaviour.** Some structures carry *more* load after buckling (stiffened plates redistribute load to their edges and stiffeners); others lose capacity and collapse (axially compressed cylinders). Linear analysis says nothing about which.

**Nonlinear buckling analysis** handles all three: introduce a small imperfection (often shaped like the first linear modes, scaled to a realistic amplitude), include material and geometric nonlinearity, and trace the load-displacement path with an arc-length method through the peak (limit) load. See [Load stepping, line search and arc-length methods](newton-raphson-and-nonlinear-solvers.md#load-stepping-line-search-and-arc-length-methods).

## Why this matters for machine learning

- **The buckling load factor is a good scalar target,** and often the design-driving one for thin-walled parts. Predict it directly with its own uncertainty rather than deriving it from a predicted stress field.
- **Mode switching creates kinks.** As a design parameter changes, the lowest mode can switch from one shape to another (for example from a global mode to a local panel mode). λ₁ as a function of the design is continuous but has kinks where two eigenvalues cross. Smooth surrogates round these corners, and optimizers chase them.
- **Mode shapes are awkward training targets.** An eigenvector's sign is arbitrary (φ and −φ are the same mode), its amplitude is arbitrary, and symmetric structures have repeated eigenvalues whose modes can be any combination of each other. Loss functions on mode shapes must be invariant to sign and normalization, for example by comparing absolute values of a normalized correlation.
- **Imperfection sensitivity is not in linear data.** A surrogate trained on linear buckling results inherits their optimism. For shells, data from nonlinear analyses with imperfections, or a conservative knock-down, is needed.
- **Classifying the failure mode** (buckling first or yielding first) is often more useful to a designer than one number, and is a natural multi-output target.

## In an interview

**Thirty-second answer.** "Buckling is a loss of stability under compression: above a critical load the straight or flat shape is no longer stable and the structure deflects sideways, often far below yield. Linear buckling is an eigenvalue problem because compressive stress adds a negative geometric stiffness proportional to the load: the structure becomes unstable when K plus λ times K_G becomes singular. The eigenvalues are buckling load factors and the eigenvectors the mode shapes. For a pinned column the first one is Euler's π²EI/L². Real structures, especially thin shells, buckle below that because of imperfections, so we apply knock-down factors or run nonlinear analyses with seeded imperfections."

**Likely follow-ups**

- *What does a buckling load factor of 0.8 mean?* The structure buckles at 80% of the applied load: it fails.
- *Why does a high-strength steel column not buckle at a higher load?* Euler's load depends on E and I, not on strength.
- *What is the difference from modal analysis?* Same eigenvalue structure: the geometric stiffness replaces the mass matrix, and the eigenvalue is a load multiplier instead of ω².

## References

- S. P. Timoshenko and J. M. Gere, *Theory of Elastic Stability*, 2nd edition, McGraw-Hill, 1961 (Dover reprint).
- [NASA SP-8007, Buckling of Thin-Walled Circular Cylinders (NASA Technical Reports Server)](https://ntrs.nasa.gov/citations/19690013955)
- R. D. Cook, D. S. Malkus, M. E. Plesha and R. J. Witt, *Concepts and Applications of Finite Element Analysis*, 4th edition, Wiley, 2002 (geometric stiffness).
