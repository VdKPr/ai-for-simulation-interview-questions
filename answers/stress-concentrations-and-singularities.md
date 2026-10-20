# Stress Concentrations and Singularities

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A **stress concentration** is a real, finite peak of stress caused by a change in geometry such as a hole, a fillet or a notch. It is described by the stress concentration factor K_t = σ_max / σ_nominal (K_t = 3 for a small circular hole in a wide plate under tension), and it *converges* when the mesh is refined enough to resolve the feature. A **stress singularity** is a point where the linear elastic model predicts *infinite* stress: a perfectly sharp re-entrant corner, a crack tip, a point load or a point constraint. There the computed peak keeps rising every time the mesh is refined, because the element nearest the point samples a stress that grows like a negative power of the distance. Real parts have small radii and real metals yield locally, so the singular number is not a physical stress. Engineers report concentrations, read stresses away from singular points, and compare models away from where loads are applied (**Saint-Venant's principle**). For machine learning, raw peak stresses at singular points are mesh noise, not labels.

**On this page**

- [Stress concentration vs stress singularity](#stress-concentration-vs-stress-singularity)
- [Stress concentration factors](#stress-concentration-factors)
- [Why the peak stress rises with mesh refinement](#why-the-peak-stress-rises-with-mesh-refinement)
- [Code: the peak keeps growing at a sharp corner](#code-the-peak-keeps-growing-at-a-sharp-corner)
- [What analysts do about singularities](#what-analysts-do-about-singularities)
- [Saint-Venant's principle](#saint-venants-principle)
- [What this means for machine learning](#what-this-means-for-machine-learning)
- [In an interview](#in-an-interview)

## Stress concentration vs stress singularity

| | Stress concentration | Stress singularity |
| --- | --- | --- |
| Cause | Finite-radius geometry change: hole, fillet, groove, shoulder | Zero-radius re-entrant corner, crack tip, point load, point or edge constraint, abrupt material jump |
| Exact elastic solution | A finite maximum | Infinite at the point |
| Under mesh refinement | Peak converges once the feature is resolved | Peak grows without limit |
| Physical meaning | Real; drives yielding and fatigue cracks | A modelling artefact; the real part has a radius, a finite contact area or local yielding |
| What to report | The converged peak (with K_t) | Stresses at a distance, or use fracture-mechanics or fatigue methods |

## Stress concentration factors

The stress concentration factor compares the peak stress with a nominal stress computed by simple formulas (for example force over net cross-section area):

```math
K_t = \frac{\sigma_{\max}}{\sigma_{\text{nominal}}}
```

**Classic results:**

- **Circular hole in a wide plate under uniaxial tension: K_t = 3.** The stress at the edge of the hole, at 90° to the load, is three times the remote stress (Kirsch, 1898). At the top and bottom of the hole, in line with the load, the tangential stress is *compressive*: −1 times the remote stress.
- **Elliptical hole** with semi-axis a perpendicular to the load and b along it: K_t = 1 + 2a/b (Inglis, 1913). A circle (a = b) gives 3. As the ellipse flattens into a crack (b → 0), K_t → ∞: the elliptical hole turns from a concentration into a singularity. This is how Inglis's result led to fracture mechanics.
- **Shoulder fillets, grooves and holes in shafts and plates:** K_t depends on the ratios of fillet radius to diameter and of the two diameters, tabulated in Peterson's charts. Values of 1.5 to 3 are common; small fillet radii push them higher.

**Design rule of thumb:** a larger fillet radius lowers K_t. Many fatigue failures start at a sharp fillet that could have been a larger radius.

**Meshing a concentration:** the peak converges only if the feature is resolved, with several elements across the fillet radius or around the quarter-circumference of a hole, preferably quadratic. A coarse mesh *under*-predicts the peak. Check by refining locally and watching the peak level off.

## Why the peak stress rises with mesh refinement

At a sharp re-entrant corner, the exact linear elastic solution has the form (Williams, 1952)

```math
\sigma(r) \propto r^{\lambda - 1}, \qquad 0 < \lambda < 1,
```

where r is the distance from the corner and λ depends on the corner angle and the boundary conditions. Because λ − 1 is negative, the stress goes to infinity as r goes to zero.

| Corner | λ (leading mode) | Stress behaves like |
| --- | --- | --- |
| Crack (360° of material around the tip) | 0.5 | r^(−0.5) |
| 90° re-entrant corner (L-shape, 270° of material) | about 0.545 | r^(−0.455) |
| Flat edge or convex corner | ≥ 1 | Finite: no singularity |

**What the mesh does.** A finite element model reports stresses at element integration points or nodes, at a distance of order h (the element size) from the corner. So the reported peak behaves like h^(λ−1). Each halving of h multiplies it by 2^(1−λ): about **1.37 for a 90° re-entrant corner** and 1.41 for a crack. The peak never converges; it just reflects how small the elements next to the corner are.

**The same happens at point loads and point supports.** A force on a single node is a force on zero area. In 3D elasticity the exact stress under a point load grows like 1/r² (in 2D, like 1/r). A support that fixes a single node or a sharp edge is a point of reaction force, with the same problem.

**The signature in practice:** refine the mesh twice; if the peak keeps rising at about a constant ratio and the location is a sharp corner, a load point or a constraint, it is a singularity. If it levels off, it is a concentration.

## Code: the peak keeps growing at a sharp corner

The cleanest demonstration uses the scalar analogue: steady heat conduction (−∇²u = 1) in an L-shaped plate with u = 0 on the boundary. At the 270° re-entrant corner the exact solution behaves like r^(2/3), so the gradient (the heat flux, the analogue of stress) behaves like r^(−1/3) and is singular. Elsewhere it is smooth.

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def l_shape_poisson(n):
    """-lap(u) = 1 on [-1, 1]^2 minus the quadrant x > 0, y < 0, with u = 0 on the boundary.
    Returns the largest gradient magnitude and the gradient near (-0.5, 0.5), far from the corner."""
    h = 2.0 / n
    xs = np.linspace(-1, 1, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    interior = ~((X >= -1e-12) & (Y <= 1e-12))          # outside the removed quadrant and its edges
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    idx = -np.ones(X.shape, int)
    idx[interior] = np.arange(interior.sum())
    rows, cols, vals = [], [], []
    for i, j in zip(*np.nonzero(interior)):              # 5-point finite-difference Laplacian
        k = idx[i, j]
        rows.append(k); cols.append(k); vals.append(4.0 / h**2)
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            m = idx[i + di, j + dj]
            if m >= 0:
                rows.append(k); cols.append(m); vals.append(-1.0 / h**2)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(interior.sum(),) * 2)
    U = np.zeros(X.shape)
    U[interior] = spla.spsolve(A.tocsc(), np.ones(interior.sum()))
    gx = (U[1:, 1:] - U[:-1, 1:] + U[1:, :-1] - U[:-1, :-1]) / (2 * h)   # gradient at cell centres
    gy = (U[1:, 1:] - U[1:, :-1] + U[:-1, 1:] - U[:-1, :-1]) / (2 * h)
    cx, cy = (X[1:, 1:] + X[:-1, :-1]) / 2, (Y[1:, 1:] + Y[:-1, :-1]) / 2
    keep = ~((cx > 0) & (cy < 0))
    g = np.hypot(gx, gy)[keep]
    far = np.argmin(np.hypot(cx + 0.5, cy - 0.5)[keep])
    return g.max(), g[far]

peaks = []
for n in [16, 32, 64, 128, 256]:
    gmax, gfar = l_shape_poisson(n)
    peaks.append(gmax)
    growth = f"(x{gmax / peaks[-2]:.2f})" if len(peaks) > 1 else ""
    print(f"h = 2/{n:<4d} peak |grad u| = {gmax:.4f} {growth:8s} |grad u| near (-0.5, 0.5) = {gfar:.4f}")
```

Output:

```
h = 2/16   peak |grad u| = 0.5931          |grad u| near (-0.5, 0.5) = 0.1497
h = 2/32   peak |grad u| = 0.8103 (x1.37)  |grad u| near (-0.5, 0.5) = 0.1439
h = 2/64   peak |grad u| = 1.0571 (x1.30)  |grad u| near (-0.5, 0.5) = 0.1427
h = 2/128  peak |grad u| = 1.3523 (x1.28)  |grad u| near (-0.5, 0.5) = 0.1425
h = 2/256  peak |grad u| = 1.7152 (x1.27)  |grad u| near (-0.5, 0.5) = 0.1425
```

The peak grows by a factor approaching 2^(1/3) ≈ 1.26 per halving and would keep growing forever; the value away from the corner converges to 0.1425. A 2D or 3D elasticity model of an L-shaped bracket with a sharp inner corner behaves the same way, with exponent about −0.455 instead of −1/3.

## What analysts do about singularities

- **Model the real geometry.** Real corners have fillets; model the fillet radius and the singularity becomes a converging concentration.
- **Read the stress at a distance.** Evaluate stresses a small distance away from the singular point, or use linearized stresses through a section (common in pressure-vessel codes), or hot-spot and notch-stress methods (common in weld fatigue), which have defined extrapolation rules.
- **Ignore results at load and constraint points.** Do not report the maximum element stress at a point load, a point constraint or the edge of a rigid coupling. Spread loads and supports over realistic areas.
- **Use fracture mechanics for cracks.** At a real crack, the meaningful quantity is not the stress but the **stress intensity factor** K (the amplitude of the r^(−1/2) field), compared with the fracture toughness.
- **Allow local yielding.** In a ductile metal under static load, a nonlinear (elastic-plastic) analysis will show the peak capped near the yield stress with a small plastic zone. Whether that is acceptable depends on the code and on fatigue.

## Saint-Venant's principle

**Statement (informal):** two different load distributions that are *statically equivalent* (same resultant force and same resultant moment) produce stress fields that differ significantly only close to where the loads are applied. At a distance of roughly the size of the loaded region (the cross-section dimension, for a beam end), the difference has decayed and the stresses are practically the same.

**Example.** A bar pulled by a force applied (a) uniformly over its end face, (b) at a single central point, or (c) through a pin in a hole. Near the end, the stresses are completely different: (b) is singular, (c) has a concentration at the hole. One or two bar widths away, all three show the same uniform stress F/A.

**Why it matters when two models apply the same load differently:**

- Two analysts model a bracket with the bolt load applied through a rigid coupling (RBE2) versus a distributing coupling (RBE3), or with the load on different nodes. Their stresses near the holes disagree, but they agree away from the holes.
- So **compare models, and validate surrogates, away from load-introduction regions,** or model the connection in detail where the local stress matters.
- It also justifies the whole practice of modelling idealized loads and supports: as long as you do not need the stress right at the interface, the idealization does not matter.
- It does not hold for every structure: thin-walled open sections and some anisotropic materials can carry end effects much further.

See [Bolt loads and remote loads](boundary-conditions.md#bolt-loads-and-remote-loads).

## What this means for machine learning

- **Peak-stress labels can be mesh noise.** A dataset where the maximum stress sits at a constraint or a point load encodes mesh size, not physics. Different designs meshed with different element sizes near the singular point get different "peaks". Check where the maxima occur before training on them.
- **Clean the target.** Exclude a zone around constraints, load points and sharp corners; or use a high percentile (for example the 99th) of stress outside that zone; or compute stress at a fixed distance from known singular features.
- **Peaks are under-predicted by smooth models.** Real concentrations are sharp and localized, and neural networks are biased towards smooth functions. See [Spectral bias](spectral-bias-and-fourier-features.md#spectral-bias). That is unconservative, because the safety decision depends on the peak.
- **Predict and evaluate the decision quantity.** Give the peak its own head and uncertainty, and evaluate the error of the peak, not just the average field error. See [Predicting a safety factor with a surrogate](von-mises-stress-and-yield.md#predicting-a-safety-factor-with-a-surrogate).
- **Compare away from interfaces.** By Saint-Venant's principle, field errors near load introductions say more about modelling choices than about the surrogate.

## In an interview

**Thirty-second answer to "Why does the peak stress at a sharp corner keep rising as you refine the mesh?"** "Because the exact linear elastic solution is singular there: at a re-entrant corner the stress grows like r to a negative power, about r^(−0.45) for a 90° corner. The element nearest the corner samples the stress at a distance of order h, so the reported peak scales with h to that power and rises by roughly 37% every time the elements are halved. It never converges. A real part has a fillet, so I would model the fillet, or read the stress away from the corner, and never report the peak at a sharp corner, a point load or a point constraint."

**Likely follow-ups**

- *K_t for a hole in a plate?* 3, at the sides of the hole perpendicular to the load.
- *What is Saint-Venant's principle?* Statically equivalent loads give the same stresses away from where they are applied; differences decay within about one characteristic dimension.
- *How would you deal with singular peaks in a surrogate dataset?* Exclude zones around them, use percentile or distance-based targets, and check the location of maxima before training.

## References

- M. L. Williams, "Stress singularities resulting from various boundary conditions in angular corners of plates in extension", *Journal of Applied Mechanics*, 1952.
- W. D. Pilkey and D. F. Pilkey, *Peterson's Stress Concentration Factors*, 3rd edition, Wiley, 2008.
- S. P. Timoshenko and J. N. Goodier, *Theory of Elasticity*, 3rd edition, McGraw-Hill, 1970 (Kirsch solution; Saint-Venant's principle).
- [SimJEB: a structural dataset in which bolt and load couplings affect stresses near the interfaces (Whalen et al.)](https://arxiv.org/abs/2105.03534)
