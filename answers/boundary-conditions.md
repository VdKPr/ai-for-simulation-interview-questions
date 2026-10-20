# Boundary Conditions

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Boundary conditions (BCs) tell the solver what happens at the edges of the domain. A **Dirichlet** condition fixes the value of the unknown (a displacement, a temperature, a velocity). A **Neumann** condition fixes its flux or normal derivative (a traction, a heat flux). A **Robin** condition relates the two (a spring support, convective cooling). Together with the geometry and the material, the boundary conditions define the problem: the same part gives completely different results under different supports and loads. Wrong boundary conditions are among the most common reasons a simulation, or a surrogate trained on simulations, gives a confident wrong answer.

**On this page**

- [Dirichlet, Neumann and Robin conditions](#dirichlet-neumann-and-robin-conditions)
- [Physical examples](#physical-examples)
- [Bolt loads and remote loads](#bolt-loads-and-remote-loads)
- [External-aerodynamics boundary conditions](#external-aerodynamics-boundary-conditions)
- [Common mistakes](#common-mistakes)
- [Giving boundary conditions to an ML model](#giving-boundary-conditions-to-an-ml-model)
- [In an interview](#in-an-interview)

## Dirichlet, Neumann and Robin conditions

Take steady heat conduction, −∇·(k∇T) = q, as the example. On each part of the boundary, exactly one condition is given:

**Dirichlet (first kind): fix the value.**

```math
T = T_0 \quad \text{on } \Gamma_D
```

*Example:* a face clamped to a large block kept at 20 °C. In structures: a fixed support (displacement = 0), or a prescribed displacement.

**Neumann (second kind): fix the flux (the normal derivative).**

```math
-k\,\frac{\partial T}{\partial n} = q_n \quad \text{on } \Gamma_N
```

*Example:* a heater delivering 500 W/m² into a face, or an insulated face (q_n = 0). In structures: an applied pressure or traction; a free surface is the special case of zero traction.

**Robin (third kind): relate the value and the flux.**

```math
-k\,\frac{\partial T}{\partial n} = h\,(T - T_\infty) \quad \text{on } \Gamma_R
```

*Example:* a surface cooled by air at T∞ with heat-transfer coefficient h. The hotter the surface, the more heat leaves. In structures: an elastic foundation or a spring support, where the reaction force is proportional to the displacement.

**Three facts interviewers like:**

1. **A pure-Neumann problem is not unique.** If only fluxes are specified everywhere, adding a constant to the temperature (or a rigid-body motion to the displacement) changes nothing. At least one Dirichlet condition, or enough of them in structures to stop all six rigid-body motions in 3D, is needed. Otherwise the stiffness matrix is singular.
2. **The fluxes must also balance** in a pure-Neumann steady problem: the total heat in must equal the total heat out (or the applied loads must be in equilibrium), or no steady solution exists.
3. **In FEM, Dirichlet conditions are "essential" and Neumann and Robin conditions are "natural".** Essential conditions are imposed on the space of candidate solutions; natural ones appear by themselves in the weak form, which is why an unloaded face is automatically free. See [Essential and natural boundary conditions](weak-form-and-galerkin.md#essential-and-natural-boundary-conditions).

## Physical examples

| Domain | Dirichlet | Neumann | Robin |
| --- | --- | --- | --- |
| Structural | Fixed support (u = 0); prescribed displacement; symmetry plane (normal displacement = 0) | Applied pressure or traction; a free surface (zero traction) | Elastic foundation or spring support: traction proportional to displacement |
| Thermal | Fixed surface temperature | Prescribed heat flux; insulated wall or symmetry plane (zero flux) | Convection to a fluid: −k ∂T/∂n = h (T − T∞); linearized radiation |
| Flow | Velocity inlet; no-slip wall (u = 0); fixed temperature wall | Zero-gradient outflow for velocity; prescribed wall heat flux | Some far-field and wall-function treatments |
| Electrostatics | Fixed potential on an electrode | Prescribed surface charge; zero normal field at a symmetry plane | Impedance-type conditions |

**Symmetry planes** combine both kinds: in structures, zero displacement *normal* to the plane (Dirichlet) and zero shear traction *in* the plane (Neumann). Using a symmetry plane halves the model, but only if the geometry *and* the loads *and* the expected response are symmetric. A buckling mode or a vortex-shedding wake may not be.

## Bolt loads and remote loads

A bolt, a bearing or an attachment point usually transfers load into a part through a hole or a face, and the analyst does not want to model the bolt or the neighbouring part in detail. Two families of **multi-point constraints** connect a single reference node (where the load or support is applied) to the nodes of the hole or face:

| | Rigid (kinematic) coupling | Distributing coupling |
| --- | --- | --- |
| Common names | RBE2 (Nastran), kinematic coupling (Abaqus), rigid remote point | RBE3 (Nastran), distributing coupling (Abaqus), deformable remote point |
| What it does | Forces the connected nodes to move as a rigid body with the reference node | Distributes the reference node's force and moment to the connected nodes as a weighted average; adds no stiffness |
| Effect on the part | Stiffens the hole or face: it cannot ovalize or deform | The hole can deform naturally |
| Local stresses | Can be artificially high at the edge of the rigid region | More realistic near the hole, though still idealized |
| Typical use | Bolts clamping a part firmly; supports that really are much stiffer than the part | Applying a remote force (for example an engine load) without adding stiffness |

**Example from a public dataset.** In SimJEB (a dataset of jet-engine bracket designs), the brackets are bolted at their base through rigid spider elements with constraints at the centres, while the loads are applied through distributing spiders over the interface holes. A model trained on SimJEB learns stresses under exactly these idealizations, and different idealizations of the same bracket would give different stresses near the holes.

**More detailed options,** when the connection itself matters: beam elements for the bolt shank with a pretension section to apply preload, contact (with friction) between the bolt head, the washer and the part, or a full 3D bolt model.

**Why two models of the same part can disagree near the bolts but agree elsewhere:** by **Saint-Venant's principle**, statically equivalent loads (same resultant force and moment) produce stress fields that differ only near where they are applied; a few characteristic lengths away the difference fades. So compare models, and validate surrogates, on stresses away from the load-introduction regions. See [Saint-Venant's principle](stress-concentrations-and-singularities.md#saint-venants-principle).

## External-aerodynamics boundary conditions

A typical steady or time-averaged CFD set-up for a road car in a virtual wind tunnel:

```
                       top: slip wall or symmetry (far away)
        ┌───────────────────────────────────────────────────────────────┐
        │                                                               │
 inlet  │  ──▶ U∞                 ┌──────────┐                          │  outlet
 U = U∞ │  ──▶                 ┌──┘   car    └──┐                       │  p = 0 (gauge)
        │  ──▶                 └──o────────o───┘ (no-slip, wheels rotate)
        └───────────────────────────────────────────────────────────────┘
                 ground: moving at U∞ (road car) or fixed (to match a tunnel test)
```

- **Inlet:** uniform free-stream velocity U∞, plus turbulence quantities for the turbulence model (for example a turbulence intensity and a length scale or viscosity ratio). These matter more than people expect, because they affect transition and wake behaviour.
- **Outlet:** fixed static pressure (usually zero gauge), with zero-gradient conditions for the other variables. It must be far enough downstream that the wake leaving the domain does not affect the car.
- **Vehicle surface:** no-slip wall (velocity zero relative to the surface), with a wall treatment that matches the near-wall mesh. See [Boundary layers and wall treatment](boundary-layers-and-wall-treatment.md).
- **Ground:** for an on-road prediction, the ground moves at U∞ relative to the car (a "rolling road"). A stationary ground grows its own boundary layer and changes underbody flow, so it is used only when matching a wind tunnel without a moving belt.
- **Wheels:** rotating, using a rotating wall velocity on the tyre surface and often a moving-reference-frame or sliding-mesh treatment for the rims. Wheels contribute a substantial share of a car's drag, so how they are modelled matters.
- **Top and sides:** slip walls or symmetry conditions, placed far enough away that the walls do not squeeze the flow. Domains commonly extend several car lengths upstream and around ten or more downstream, with a cross-section large enough that the *blockage* (car frontal area divided by domain cross-section) is small, often quoted as below about 1 to 2%.
- **Half-model symmetry:** possible for a symmetric car at zero yaw in steady RANS, but not for crosswind (yaw) cases or for scale-resolving simulations, where the real wake is not symmetric at every instant.

For aircraft, the outer boundary is usually a **far-field** condition based on characteristic (Riemann-invariant) theory, tens of chord lengths away, which lets waves leave the domain without reflecting.

**For ML datasets, these settings are part of the input.** Two datasets of "the same car" with stationary versus moving ground, or different inlet turbulence, are different problems. A surrogate trained on one is out of distribution on the other.

## Common mistakes

- **Too few constraints.** Rigid-body motion makes the stiffness matrix singular: the solver fails, or returns enormous displacements.
- **Over-constraint.** Fully fixing a face that should expand or rotate (for example a thermally loaded part clamped on all sides) creates artificial stress at the support.
- **Point loads and point constraints.** In 2D and 3D elasticity, a force or support at a single node gives a singular stress there that grows without limit as the mesh is refined. Spread loads over a realistic area. See [Stress concentrations and singularities](stress-concentrations-and-singularities.md).
- **Units and signs.** A pressure entered in Pa instead of MPa, or a load applied in the wrong direction, still gives a converged answer.
- **Outflow too close.** In CFD, an outlet placed inside a recirculation zone causes reverse flow at the boundary and inaccurate results.
- **Inconsistent thermal conditions.** Specifying a heat flux on every boundary of a steady problem without balancing it leads to a temperature that drifts or never converges.

## Giving boundary conditions to an ML model

A surrogate only knows the boundary conditions if they are encoded in its inputs:

- **Node-type features.** One-hot labels such as interior, wall, inlet, outlet, fixed or loaded. MeshGraphNets uses node types like these.
- **BC values as features.** Prescribed velocities, loads or temperatures as per-node features on the boundary nodes.
- **Global conditioning.** Scalars such as the inlet speed, angle of attack or load magnitude, fed into the network (for example through FiLM-style feature modulation or adaptive normalization).
- **Geometry-derived features.** Distance to the nearest wall (a signed distance function) and surface normals tell the model where the boundary is.
- **Hard enforcement.** Build the condition into the output so it holds exactly, for example by multiplying the network output by a function that vanishes on the fixed boundary. See [Hard vs soft constraints](hard-vs-soft-constraints.md).
- **Coverage.** A model can only handle BC *types* and value ranges it was trained on. A new load case is a new distribution. See [Out-of-distribution detection](out-of-distribution-and-generalization.md).

## In an interview

**Thirty-second answer.** "Dirichlet fixes the value, like a fixed support or a fixed temperature. Neumann fixes the flux, like an applied traction or a heat flux; a free surface is the zero-traction case. Robin relates them, like convective cooling where the flux is h times the temperature difference, or a spring support. In FEM, Dirichlet conditions are essential and must be imposed on the solution space; Neumann conditions are natural and appear in the weak form automatically."

**Likely follow-ups**

- *RBE2 or RBE3 for a bolt load?* RBE2 if the connection really makes the hole rigid; RBE3 to apply a remote load without adding stiffness. Expect local stresses near the hole to depend on the choice.
- *Moving or fixed ground in car CFD?* Moving, for on-road predictions; fixed only to match a tunnel without a rolling road.
- *Why does a point load give infinite stress?* A force over zero area; the elasticity solution is singular there, so refine-and-watch gives no converged peak.

## References

- [SimJEB: Simulated Jet Engine Bracket Dataset (Whalen et al.)](https://arxiv.org/abs/2105.03534)
- [Learning Mesh-Based Simulation with Graph Networks (MeshGraphNets, Pfaff et al.)](https://arxiv.org/abs/2010.03409)
- [DrivAerML: High-Fidelity CFD Dataset for Road-Car External Aerodynamics (Ashton et al.)](https://arxiv.org/abs/2408.11969)
- [NASA Turbulence Modeling Resource: boundary condition guidance for verification cases](https://turbmodels.larc.nasa.gov/)
