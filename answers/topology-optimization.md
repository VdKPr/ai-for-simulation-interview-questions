# Topology Optimization

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Topology optimization decides **where material should go** inside a design space, given loads, supports and a material budget, without assuming the shape's topology (number of holes, members) in advance. The classic problem is to **minimize compliance** (maximize stiffness) for a given volume fraction. The standard **SIMP** method ("solid isotropic material with penalization") gives each finite element a density ρₑ between 0 and 1 and a stiffness Eₑ = E_min + ρₑᵖ(E₀ − E_min) with p ≈ 3: the power makes intermediate densities structurally inefficient, so the optimizer is pushed toward solid-or-void designs. Every iteration needs one finite-element solve; the gradient of compliance comes from the **self-adjoint** adjoint for free; a **filter** prevents checkerboard patterns and sets a minimum member size; and an optimality-criteria or MMA update changes the densities. **Machine learning** is used to predict final layouts from early iterations or directly from loads (fast but needing verification), to **reparameterize** densities with a neural network (Hoyer et al. found better designs 50% more often than the best baseline on 116 tasks), and as **generative models** that propose many candidate topologies (TopoDiff, a diffusion model). Every ML-proposed layout must still be checked, and usually refined, with finite elements.

**On this page**

- [SIMP](#simp)
- [Filters, projections and crisp designs](#filters-projections-and-crisp-designs)
- [Other methods](#other-methods)
- [Making designs manufacturable](#making-designs-manufacturable)
- [Machine learning for topology optimization](#machine-learning-for-topology-optimization)
- [Code: a SIMP topology optimizer in 70 lines](#code-a-simp-topology-optimizer-in-70-lines)
- [In an interview](#in-an-interview)

## SIMP

**Problem.** Discretize the design domain into N elements with densities ρ = (ρ₁, …, ρ_N):

```math
\min_{\rho}\; c(\rho) = F^\top u(\rho) = u^\top K(\rho)\, u
\quad\text{subject to}\quad K(\rho)\,u = F,\qquad \sum_e \rho_e v_e \le V^*,\qquad 0 \le \rho_e \le 1
```

**Material interpolation.** Eₑ(ρₑ) = E_min + ρₑᵖ (E₀ − E_min), with p ≈ 3. Why the power: an element at density 0.5 costs half the material but provides only 0.5³ = 1/8 of the stiffness, so intermediate densities are a bad deal and the optimum tends toward 0 or 1. E_min (around 10⁻⁹ E₀) keeps the stiffness matrix non-singular where material is removed.

**Sensitivities.** Compliance is self-adjoint (the adjoint variable equals the displacement), so after one FE solve every element's derivative is known:

```math
\frac{\partial c}{\partial \rho_e} = -p\,\rho_e^{\,p-1}(E_0 - E_{\min})\; u_e^\top k_0\, u_e
```

where uₑ are the element's displacements and k₀ the element stiffness matrix for unit modulus. All derivatives are negative (more material is always stiffer); the largest magnitudes are where the strain energy is highest. See [Worked example: compliance in topology optimization](adjoint-method.md#worked-example-compliance-in-topology-optimization).

**Update rules.**

- **Optimality criteria (OC):** a heuristic fixed-point update, ρₑ ← ρₑ (−∂c/∂ρₑ / (λ ∂V/∂ρₑ))^{1/2}, with move limits, and the Lagrange multiplier λ found by bisection so that the volume constraint holds. Simple and effective for compliance with one volume constraint.
- **Method of moving asymptotes (MMA, Svanberg 1987):** a general gradient-based method for many constraints (stress, displacement, frequency), the standard in commercial tools.

**Cost.** Dominated by the FE solves: one per iteration, typically 50-300 iterations; large 3D problems (millions of elements) use iterative solvers with multigrid preconditioners and run on HPC or GPUs.

## Filters, projections and crisp designs

- **Checkerboards:** with low-order elements, alternating solid and void elements appear artificially stiff, a numerical artefact. Filters remove them.
- **Density filter:** replace each density by a weighted average of densities within a radius r_min; the optimizer works on the unfiltered variables, the physics sees the filtered ones (with the chain rule applied to the sensitivities). It also enforces a **minimum length scale** and makes results independent of mesh refinement.
- **Sensitivity filter:** average the sensitivities instead (Sigmund's original heuristic).
- **Heaviside projection** with continuation (gradually sharpening a smoothed step) produces crisp 0/1 designs; robust formulations (eroded, intermediate and dilated designs) control minimum sizes of both solid and void.
- **Continuation on p** (start at 1, increase to 3 or more) reduces the risk of poor local optima.

## Other methods

- **Level-set methods:** the boundary is the zero level set of a function evolved by sensitivities; crisp boundaries throughout.
- **Evolutionary structural optimization (ESO/BESO):** remove (and add back) elements based on strain-energy ranking.
- **Homogenization:** the original approach (Bendsøe and Kikuchi, 1988), with microstructures in each element; revived for lattice and multi-scale designs.
- **Moving morphable components:** optimize the positions and sizes of explicit geometric components.

## Making designs manufacturable

- **Minimum member and hole size:** the filter radius and robust projection.
- **Additive manufacturing:** overhang constraints (no unsupported surfaces below a critical angle), minimum feature sizes, and lattice infill.
- **Casting and moulding:** draft angles and no undercuts (a single parting direction); **extrusion** constraints for constant cross-sections; **machining** constraints for tool access.
- **Symmetry and pattern repetition** constraints.
- **Stress constraints** are harder: stress is local, there are as many constraints as elements, and stress-constrained optima can be singular; aggregation (p-norm) and relaxation are needed.
- **From densities to CAD:** threshold or extract an iso-surface, smooth it, rebuild it as CAD (often manually or with automated reconstruction), then **re-verify with FEA** (stress, buckling, fatigue), because the interpreted geometry differs from the optimized density field. See [Keeping designs manufacturable](surrogate-based-optimization.md#keeping-designs-manufacturable).

## Machine learning for topology optimization

1. **Predicting the final layout.** Sosnovik and Oseledets (2017) framed topology optimization as image segmentation: a convolutional encoder-decoder takes densities from an early SIMP iteration (and their changes) and predicts the final layout, accelerating the process. Later work predicts layouts directly from loads and boundary conditions (conditional GANs such as TopologyGAN). Fast, but the prediction can be disconnected, sub-optimal or infeasible, and generalization to new load cases is limited.
2. **Neural reparameterization.** Hoyer, Sohl-Dickstein and Greydanus (2019) optimized the weights of a CNN whose output is the density field, instead of the densities directly, keeping the physics and the objective unchanged. The network's implicit bias changed which local optimum was found: on 116 structural optimization tasks, it produced the best design 50% more often than the best baseline.
3. **Generative models.** TopoDiff (Mazé and Ahmed, 2022), a conditional diffusion model with guidance from surrogate models of compliance and manufacturability, reduced the average error on physical performance eightfold and produced eleven times fewer infeasible samples than a state-of-the-art conditional GAN. Generative models can propose many diverse candidates quickly. See [Building an inverse-design tool](generative-models-for-physics.md#building-an-inverse-design-tool).
4. **Surrogate solvers inside the loop:** neural networks replacing or accelerating the FE solve (for example as preconditioners or multigrid components), keeping the optimizer.
5. **Commercial generative design** tools (Autodesk Fusion generative design, Altair Inspire, Ansys Discovery, nTop) combine topology optimization, manufacturing constraints and many candidate solutions.

**Rule:** treat an ML-proposed topology as a starting point. Evaluate it with an FE solve, and usually run a few SIMP iterations from it, which also repairs disconnected members.

## Code: a SIMP topology optimizer in 70 lines

A Python port of the structure of the widely used "88-line" MATLAB code (Andreassen et al., 2011): the MBB beam (a simply supported beam, modelled as a half with a symmetry condition), 60 × 20 four-node elements, 50% volume, p = 3, a density filter with radius 1.5 elements, and optimality-criteria updates for 60 iterations. The final density field is printed as text (# solid, + intermediate).

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def top88(nelx=60, nely=20, volfrac=0.5, penal=3.0, rmin=1.5, iters=60):
    """Compliance minimization with SIMP, a density filter and optimality-criteria updates
    (a Python port of the structure of Andreassen et al.'s 88-line MATLAB code; MBB half-beam)."""
    E0, Emin, nu = 1.0, 1e-9, 0.3
    k = np.array([1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8, -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8])
    KE = E0 / (1 - nu**2) * np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                                       [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                                       [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                                       [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                                       [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                                       [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                                       [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                                       [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])
    ndof = 2 * (nelx + 1) * (nely + 1)
    nodenrs = np.arange((nelx + 1) * (nely + 1)).reshape(nelx + 1, nely + 1).T
    edofvec = (2 * nodenrs[:-1, :-1] + 2).T.ravel()
    edofmat = edofvec[:, None] + np.array([0, 1, 2 * nely + 2, 2 * nely + 3, 2 * nely, 2 * nely + 1, -2, -1])
    iK = np.kron(edofmat, np.ones((8, 1))).ravel().astype(int)
    jK = np.kron(edofmat, np.ones((1, 8))).ravel().astype(int)
    F = np.zeros(ndof); F[1] = -1.0                                  # downward load at the top-left corner
    fixed = np.union1d(np.arange(0, 2 * (nely + 1), 2), [ndof - 1])  # symmetry line (x fixed) and the roller support
    free = np.setdiff1d(np.arange(ndof), fixed)
    # density filter weights
    rows, cols, vals = [], [], []
    for i in range(nelx):
        for j in range(nely):
            e1 = i * nely + j
            for i2 in range(max(i - int(np.ceil(rmin)) + 1, 0), min(i + int(np.ceil(rmin)), nelx)):
                for j2 in range(max(j - int(np.ceil(rmin)) + 1, 0), min(j + int(np.ceil(rmin)), nely)):
                    w = rmin - np.hypot(i - i2, j - j2)
                    if w > 0:
                        rows.append(e1); cols.append(i2 * nely + j2); vals.append(w)
    H = sp.csr_matrix((vals, (rows, cols)), shape=(nelx * nely, nelx * nely))
    Hs = np.asarray(H.sum(1)).ravel()
    x = np.full(nely * nelx, volfrac); xPhys = x.copy()
    history = []
    for it in range(iters):
        sK = (KE.ravel()[None, :] * (Emin + xPhys[:, None] ** penal * (E0 - Emin))).ravel()
        K = sp.coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        U = np.zeros(ndof)
        U[free] = spla.spsolve(K[free][:, free], F[free])
        ce = np.einsum("ij,jk,ik->i", U[edofmat], KE, U[edofmat])     # element strain energy u_e^T k0 u_e
        c = ((Emin + xPhys ** penal * (E0 - Emin)) * ce).sum()        # compliance
        dc = -penal * (E0 - Emin) * xPhys ** (penal - 1) * ce        # adjoint (self-adjoint) sensitivity
        dc = H @ (dc / Hs)                                          # filter the sensitivities (density filter chain rule)
        dv = H @ (np.ones_like(x) / Hs)
        l1, l2, move = 0.0, 1e9, 0.2                                  # optimality criteria with bisection on the multiplier
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            xnew = np.clip(x * np.sqrt(np.maximum(-dc, 0) / dv / lmid), np.maximum(0, x - move), np.minimum(1, x + move))
            xPhys = (H @ xnew) / Hs
            if xPhys.mean() > volfrac: l1 = lmid
            else: l2 = lmid
        x = xnew
        history.append(c)
    return xPhys.reshape(nelx, nely).T, history

design, hist = top88()
print(f"compliance: start {hist[0]:.1f}, iteration 10 {hist[9]:.1f}, final {hist[-1]:.1f} ({len(hist)} iterations)")
print(f"volume fraction {design.mean():.3f}; share of elements that are nearly solid or void (<0.1 or >0.9): {np.mean((design < 0.1) | (design > 0.9)):.2f}")
for row in design[::2]:                                             # every second row, as text
    print("".join("#" if v > 0.7 else "+" if v > 0.3 else " " for v in row))
```

Output:

```
compliance: start 1007.0, iteration 10 283.2, final 220.4 (60 iterations)
volume fraction 0.500; share of elements that are nearly solid or void (<0.1 or >0.9): 0.66
###########################################+                
##############################################+             
  +#####+              +++    +######+++##++####+           
     #####+            +          +######     #####         
       +#####         +          #######+       +####+      
          ######+   ++        +#####+   ++         #####    
           +#########      +#####        ++          +####  
            +#+++#####+++#####            ++        +###### 
###########################+                +   +##++   +###
##################################################+++++++###
```

Compliance drops by a factor of about 4.6 from the uniform starting design, most of it in the first ten iterations, while the volume stays at exactly 50%. The layout is the familiar MBB truss: a top chord carrying compression from the load at the top-left, a bottom chord in tension, and diagonal members, which emerged from the physics without any member being specified. About a third of the elements are still grey after 60 iterations; continuation on p or a Heaviside projection would make the design crisp before it is interpreted into CAD and re-verified.

## In an interview

**Thirty-second answer to "What is SIMP topology optimization?"** "Each element gets a density between zero and one, and its stiffness is E_min plus density to the power p, around 3, times the solid modulus. You minimize compliance subject to a volume constraint; every iteration is one FE solve, and because compliance is self-adjoint the gradient is just minus p rho to the p minus one times the element strain energy. The penalty makes grey material inefficient, a filter removes checkerboards and sets a minimum member size, and optimality criteria or MMA update the densities. The result is a density field that still has to be turned into CAD and re-checked."

**Likely follow-ups**

- *Why p > 1?* Intermediate densities give less stiffness per unit material, pushing the design to solid or void.
- *Why filter?* Checkerboards are a numerical artefact; filters also give mesh independence and a minimum length scale.
- *How does machine learning help?* Faster layout prediction, neural reparameterization, generative proposals; all followed by FE verification.

## References

- M. P. Bendsøe and O. Sigmund, *Topology Optimization: Theory, Methods, and Applications*, Springer, 2003.
- E. Andreassen et al., "Efficient topology optimization in MATLAB using 88 lines of code", *Structural and Multidisciplinary Optimization*, 2011 ([DTU TopOpt group codes](https://www.topopt.mek.dtu.dk/apps-and-software/efficient-topology-optimization-in-matlab)).
- K. Svanberg, "The method of moving asymptotes", *International Journal for Numerical Methods in Engineering*, 1987.
- [Neural networks for topology optimization (Sosnovik & Oseledets)](https://arxiv.org/abs/1709.09578)
- [Neural reparameterization improves structural optimization (Hoyer, Sohl-Dickstein & Greydanus)](https://arxiv.org/abs/1909.04240)
- [Diffusion Models Beat GANs on Topology Optimization (TopoDiff; Mazé & Ahmed)](https://arxiv.org/abs/2208.09591)
