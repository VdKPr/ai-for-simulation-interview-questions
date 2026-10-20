# Stress, Strain and Linear Elasticity

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** **Stress** is internal force per unit area (in MPa = N/mm²); **strain** is relative deformation (dimensionless). **Young's modulus E** is the stiffness of a material in uniaxial tension: stress divided by strain in the elastic range (about 200 to 210 GPa for steel, 70 GPa for aluminium). **Poisson's ratio ν** is how much a material contracts sideways when stretched (about 0.3 for metals, close to 0.5 for rubber). **Linear elasticity** is the model in which strains and rotations are small, the material returns to its original shape when unloaded, and stress is proportional to strain (Hooke's law). Under these assumptions the response is linear in the loads, so results can be scaled and superposed: double the load, double every stress and displacement. Most structural FEA starts here, and much of the art is knowing when its assumptions stop holding.

**On this page**

- [Stress](#stress)
- [Strain](#strain)
- [Young's modulus and Poisson's ratio](#youngs-modulus-and-poissons-ratio)
- [Hooke's law in three dimensions](#hookes-law-in-three-dimensions)
- [Linear elasticity: the full problem](#linear-elasticity-the-full-problem)
- [A worked example: a steel bar in tension](#a-worked-example-a-steel-bar-in-tension)
- [Plane stress and plane strain](#plane-stress-and-plane-strain)
- [When linear elasticity is not enough](#when-linear-elasticity-is-not-enough)
- [Why this matters for machine learning](#why-this-matters-for-machine-learning)
- [In an interview](#in-an-interview)

## Stress

Cut a loaded part with an imaginary plane. The material on one side pushes and pulls on the other side with a distributed internal force. **Stress** is that force per unit area of the cut.

- The component **normal** to the cut is the **normal stress** σ: positive in tension, negative in compression.
- The components **in** the plane of the cut are **shear stresses** τ.

At a point, the full state of stress needs all three cut orientations, which gives the **stress tensor**, a symmetric 3 × 3 matrix:

```math
\boldsymbol{\sigma} = \begin{bmatrix} \sigma_{xx} & \tau_{xy} & \tau_{xz} \\ \tau_{xy} & \sigma_{yy} & \tau_{yz} \\ \tau_{xz} & \tau_{yz} & \sigma_{zz} \end{bmatrix}
```

Six independent numbers. Its eigenvalues are the **principal stresses** σ₁ ≥ σ₂ ≥ σ₃: the normal stresses on the three planes that carry no shear. A single summary number such as the von Mises stress is computed from them. See [Von Mises stress](von-mises-stress-and-yield.md).

**Units:** 1 MPa = 1 N/mm² = 10⁶ Pa. Structural steels typically yield somewhere between about 250 and several hundred MPa, depending on grade.

## Strain

**Strain** measures deformation relative to the original size. In a bar of length L that stretches by ΔL, the normal strain is ε = ΔL / L. A shear strain γ measures the change of a right angle (in radians).

In 3D, with displacement field u(x), the small-strain tensor is the symmetric part of the displacement gradient:

```math
\boldsymbol{\varepsilon} = \tfrac{1}{2}\left(\nabla \mathbf{u} + \nabla \mathbf{u}^{\mathsf T}\right)
```

The "small" matters: this formula is accurate only when strains and rotations are small (strains of a few percent or less, rotations of a few degrees or less). For large rotations, it predicts spurious strain in a part that has only rotated rigidly, which is one reason *geometric nonlinearity* exists. See [Nonlinear structural analysis](nonlinear-structural-analysis.md).

Typical elastic strains in metals are tiny: steel at yield has a strain of roughly 0.1 to 0.2%.

## Young's modulus and Poisson's ratio

In a uniaxial tensile test, the stress-strain curve starts as a straight line. Its slope is **Young's modulus**:

```math
E = \frac{\sigma}{\varepsilon} \quad \text{(uniaxial, elastic range)}
```

While the bar stretches, it gets thinner. **Poisson's ratio** is the ratio of the lateral contraction to the axial extension:

```math
\nu = -\frac{\varepsilon_{\text{lateral}}}{\varepsilon_{\text{axial}}}
```

| Material | E (approximate) | ν (approximate) |
| --- | --- | --- |
| Steel | 200–210 GPa | 0.3 |
| Aluminium alloys | 70 GPa | 0.33 |
| Titanium alloys | 110 GPa | 0.34 |
| Concrete | 25–40 GPa | 0.2 |
| Engineering polymers | 1–4 GPa | 0.35–0.4 |
| Rubber | a few MPa | close to 0.5 |

For an isotropic material, two constants define everything. The others follow:

```math
G = \frac{E}{2(1+\nu)} \;\;\text{(shear modulus)}, \qquad K = \frac{E}{3(1-2\nu)} \;\;\text{(bulk modulus)}
```

As ν approaches 0.5, the bulk modulus goes to infinity: the material resists any change of volume (it is *incompressible*). That is the source of **volumetric locking** in low-order finite elements. See [Volumetric locking](elements-and-shape-functions.md#volumetric-locking-and-other-pathologies).

## Hooke's law in three dimensions

For an isotropic linear elastic material, stress and strain are related by

```math
\boldsymbol{\sigma} = \lambda\,\mathrm{tr}(\boldsymbol{\varepsilon})\,\mathbf{I} + 2\mu\,\boldsymbol{\varepsilon}, \qquad \mu = G = \frac{E}{2(1+\nu)}, \quad \lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}
```

λ and μ are the **Lamé parameters**. In the inverse direction, for example, a stress σ_xx alone produces ε_xx = σ_xx/E and ε_yy = ε_zz = −ν σ_xx/E.

Non-isotropic materials need more constants: an orthotropic material (wood, unidirectional composite plies) has nine, a fully anisotropic one 21. Composite laminates are modelled ply by ply with orthotropic properties and fibre orientations.

**Thermal strain** adds to the mechanical strain when temperature changes: ε_thermal = α ΔT in every direction for an isotropic material, with α the coefficient of thermal expansion. See [Thermal stress and warpage](thermal-stress-and-warpage.md).

## Linear elasticity: the full problem

Three sets of equations plus boundary conditions:

1. **Equilibrium:** ∇·σ + b = 0 in the body, where b is the body force (gravity, for example). In dynamics, add the inertia term ρü.
2. **Kinematics:** ε = ½(∇u + ∇uᵀ).
3. **Constitutive law:** Hooke's law, above.
4. **Boundary conditions:** prescribed displacements on supports, prescribed tractions σ·n = t on loaded faces, zero traction on free faces. See [Boundary conditions](boundary-conditions.md).

Combined, they form a linear PDE system for the displacement u. Its weak form is the principle of virtual work, and its FEM discretization is K u = f. See [The weak form](weak-form-and-galerkin.md).

**Assumptions of the linear model:**

- small strains and small rotations, so the original geometry can be used for equilibrium;
- linear, elastic, history-independent material;
- supports and contacts that do not change with the load.

**Its most useful consequence is superposition.** Because the problem is linear, the response to load A plus load B equals the response to A plus the response to B, and scaling a load scales the response. Engineers use this constantly: solve a few unit load cases once, then combine them for hundreds of load combinations.

## A worked example: a steel bar in tension

A steel bar 1 m long with a cross-section of 100 mm² carries an axial force of 10 kN. Take E = 210 GPa = 210,000 MPa and ν = 0.3.

- Stress: σ = F/A = 10,000 N / 100 mm² = **100 MPa**.
- Axial strain: ε = σ/E = 100 / 210,000 = **4.76 × 10⁻⁴** (0.0476%).
- Elongation: ΔL = ε·L = 4.76 × 10⁻⁴ × 1,000 mm = **0.476 mm**.
- Lateral strain: −ν ε = **−1.43 × 10⁻⁴**, so a 10 mm wide bar gets narrower by about 1.4 µm.

Every number is linear in the force: at 20 kN, stress, strain and elongation all double, as long as the stress stays below yield.

## Plane stress and plane strain

Two common 2D simplifications:

- **Plane stress:** a thin plate loaded in its own plane. Stresses through the thickness are zero (σ_zz = τ_xz = τ_yz = 0). Used for thin sheets and plates under in-plane loads.
- **Plane strain:** a long body with the same cross-section along its length and constrained along it, such as a dam, a tunnel lining or a long pipe. Strains along the length are zero (ε_zz = 0), so σ_zz = ν(σ_xx + σ_yy) is not zero.

Use the wrong one and stiffness and stress are wrong by a factor related to ν. The cantilever example in [Elements and shape functions](elements-and-shape-functions.md#code-linear-vs-quadratic-triangles-on-a-cantilever) uses plane stress.

## When linear elasticity is not enough

| Situation | Why linear elasticity fails | What is needed |
| --- | --- | --- |
| Stress above yield | Material flows plastically; stress is no longer proportional to strain; unloading leaves permanent strain | Plasticity models ([Nonlinear structural analysis](nonlinear-structural-analysis.md)) |
| Large deflections or rotations | Equilibrium on the undeformed shape is wrong; stiffness changes with deformation (a thin plate stiffens as it deflects) | Geometric nonlinearity |
| Contact | Parts touch or separate; the load path changes with the load | Contact algorithms |
| Slender members in compression | Sudden loss of stiffness at a critical load | Buckling analysis ([Buckling](buckling.md)) |
| Rubber, soft tissue, foams | Large strains and nonlinear elasticity | Hyperelastic material models |
| Polymers over time, hot metals | Time-dependent behaviour | Viscoelastic or creep models |
| Repeated loading | Damage accumulates even below yield | Fatigue assessment ([Fatigue](fatigue.md)) |

A practical warning sign: in a linear analysis, a stress far above the yield strength does not mean the part carries that stress. It means the linear model is invalid in that region and a nonlinear analysis (or a design change) is needed.

## Why this matters for machine learning

- **Linearity can be built in.** If the analysis is linear, the displacement for load magnitude F is exactly F times the displacement for a unit load, and inversely proportional to E. A surrogate should not have to *learn* this: predict the unit-load response and scale it, or non-dimensionalize the outputs by F/E. See [Why non-dimensionalize](nondimensionalization-and-normalization.md#why-non-dimensionalize).
- **Superposition of load cases.** For multiple independent loads, a model can predict each unit-load response and combine them, instead of learning every combination.
- **Tensors transform under rotation.** Rotating the part rotates the stress tensor (σ′ = R σ Rᵀ), while the von Mises stress is invariant. Augmenting or designing a model for rotations must respect this. See [Equivariance and symmetry](equivariance-and-symmetry.md).
- **Labels change meaning outside the linear range.** A dataset of linear analyses with stresses above yield contains numbers that are not physical stresses. Check the regime before training on peak stresses.

## In an interview

**Thirty-second answer.** "Stress is internal force per area, a symmetric tensor with six components; strain is relative deformation, the symmetric part of the displacement gradient. Young's modulus is the slope of stress against strain in uniaxial tension, about 210 GPa for steel; Poisson's ratio is the lateral contraction ratio, about 0.3 for metals. Linear elasticity assumes small strains and rotations and a linear, reversible material, so stress is proportional to strain and the whole response is linear in the loads: we can scale and superpose results."

**Likely follow-ups**

- *What happens as ν approaches 0.5?* The bulk modulus goes to infinity; the material is incompressible, and low-order elements lock.
- *Plane stress or plane strain for a thin bracket plate?* Plane stress. For a long dam cross-section, plane strain.
- *Your linear analysis shows 600 MPa in a steel with a 350 MPa yield. What does it mean?* The linear model is invalid there; the material would yield. Run a nonlinear analysis, or change the design.

## References

- S. P. Timoshenko and J. N. Goodier, *Theory of Elasticity*, 3rd edition, McGraw-Hill, 1970.
- [MIT OpenCourseWare 2.001 Mechanics and Materials I](https://ocw.mit.edu/courses/2-001-mechanics-materials-i-fall-2006/)
- [The FEniCSx tutorial: linear elasticity (Jørgen S. Dokken)](https://jsdokken.com/dolfinx-tutorial/)
