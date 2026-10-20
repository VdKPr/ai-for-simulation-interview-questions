# Multiphysics Coupling

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A multiphysics problem has several physical fields that influence each other: temperature and stress, fluid and structure, current and heat. In **one-way coupling**, one field affects the other but not the reverse: compute the temperature, then apply it as a thermal load in the stress analysis. In **two-way coupling**, each field changes the other (deformation changes the flow, which changes the load, which changes the deformation), so they must be iterated or solved together. Two-way problems can be solved **partitioned**, with separate solvers for each field exchanging interface data and sub-iterating until they agree, or **monolithic**, with all fields assembled into one system. Partitioned coupling reuses specialized solvers and is flexible but can become unstable when the coupling is strong, as with the *added-mass effect* in fluid-structure interaction; monolithic coupling is robust but needs one code with all the physics and large, hard-to-solve systems.

**On this page**

- [What coupling means](#what-coupling-means)
- [One-way vs two-way coupling](#one-way-vs-two-way-coupling)
- [Thermo-mechanical coupling, with numbers](#thermo-mechanical-coupling-with-numbers)
- [Partitioned vs monolithic coupling](#partitioned-vs-monolithic-coupling)
- [The added-mass instability in fluid-structure interaction](#the-added-mass-instability-in-fluid-structure-interaction)
- [Why this matters for machine learning](#why-this-matters-for-machine-learning)
- [In an interview](#in-an-interview)

## What coupling means

| Coupling | What affects what | Examples |
| --- | --- | --- |
| Thermo-mechanical | Temperature causes thermal strain and changes material properties; deformation can change heat paths | Engine parts, brake discs, electronics packages, welding, turbine blades |
| Fluid-structure interaction (FSI) | Fluid pressure and shear deform the structure; deformation changes the flow | Aircraft wing flutter, heart valves and arteries, wind turbine blades, flexible pipes |
| Electro-thermal | Current causes Joule heating; temperature changes electrical resistance | Busbars, power electronics, battery tabs, fuses |
| Electro-chemo-thermal | Electrochemistry generates heat; temperature changes reaction rates | Battery cells and packs |
| Thermo-fluid (conjugate heat transfer) | Flow carries heat; temperature changes density and buoyancy | Electronics cooling, heat exchangers ([Conjugate heat transfer](heat-transfer-basics.md#conjugate-heat-transfer)) |
| Vibro-acoustic | Structural vibration radiates sound; sound pressure loads the structure | Car cabins, loudspeakers, sonar |
| Piezoelectric, electromagnetic-mechanical | Fields cause strain and the reverse | Sensors, actuators, motors |

## One-way vs two-way coupling

**One-way coupling.** Field A influences field B, but B's effect on A is negligible. Solve A, transfer its result to B, solve B. No iteration.

*Example:* a steel bracket next to a hot exhaust. A thermal analysis gives the temperature field; that field is mapped onto the structural mesh and applied as a thermal load; the structural analysis gives thermal stresses and distortion. The bracket's small deformation does not change how heat reaches it.

**Two-way coupling.** Each field significantly changes the other, so they must be solved consistently: iterate between them (each time step, for transient problems) or solve them together.

*Examples:*

- **Thermal contact:** heating causes a joint to open or close, which changes the contact conductance across it, which changes the temperature.
- **Temperature-dependent properties with heating from deformation:** in metal forming and high-rate deformation, plastic work generates heat, which softens the material, which changes the deformation.
- **FSI with significant deformation:** a flexible wing bends, changing its angle of attack and loads; flutter is a dynamic two-way instability.
- **Joule heating with temperature-dependent resistance:** current heats a conductor, raising its resistance, which changes the current distribution and the heating.

**How to decide.** Estimate the feedback. If the second field changes the first by less than the accuracy you need (for example, the deformation changes a gap conductance by a few percent where the uncertainty is larger anyway), one-way coupling is justified. Otherwise couple both ways. When in doubt, run one two-way case and compare with the one-way answer.

## Thermo-mechanical coupling, with numbers

A free bar heated uniformly by ΔT simply gets longer, by the thermal strain ε_th = α ΔT. No stress. **Thermal stress appears only when expansion is prevented**: by supports, by colder neighbouring material, or by a bonded material that expands differently.

**A fully constrained steel bar** heated by ΔT = 100 K, with α = 12 × 10⁻⁶ /K and E = 210 GPa:

```math
\sigma = -E\,\alpha\,\Delta T = -210{,}000\ \text{MPa} \times 12\times10^{-6} \times 100 = -252\ \text{MPa}
```

A 100-degree temperature change produces a compressive stress close to the yield strength of ordinary structural steel. This is why rails, pipelines and bridges have expansion joints, why parts with hot and cold regions crack, and why electronics packages warp. See [Thermal stress and warpage](thermal-stress-and-warpage.md).

**In a simulation workflow:**

1. Thermal analysis (steady or transient), possibly with conjugate heat transfer.
2. Map the temperature field from the thermal mesh to the structural mesh (they are often different meshes).
3. Structural analysis with thermal strain α(T − T_ref) and temperature-dependent material properties.
4. For two-way coupling, feed back the deformation (gaps, contact pressure, geometry) and repeat until converged.

The reference temperature T_ref, at which the part is stress-free, is an easily forgotten input: for a soldered or bonded assembly, it is often the solidification or curing temperature, not room temperature.

## Partitioned vs monolithic coupling

**Partitioned (staggered) coupling.** Each field is solved by its own solver, often existing specialized codes (a CFD code and an FEA code), which exchange data at the interface: the fluid sends pressure and shear forces to the structure, the structure sends displacements back; or one side sends temperatures and the other heat fluxes.

- **Explicit (loose) coupling:** one exchange per time step. Cheap, but the interface conditions are only approximately satisfied, and it can be unstable for strongly coupled problems.
- **Implicit (strong) coupling:** sub-iterate within each time step until the interface quantities converge, using relaxation (for example Aitken's method) or quasi-Newton acceleration (as implemented in the preCICE coupling library). Satisfies the interface conditions, at the cost of several solves per step.
- **Data mapping** between non-matching interface meshes (nearest-neighbour, projection, radial basis functions) must conserve what needs conserving, such as total force or heat.

**Monolithic coupling.** All fields are discretized and assembled into one large system and solved together, typically with Newton's method on the combined residual.

- Interface conditions are satisfied exactly and strong coupling is handled robustly.
- Needs one code that contains all the physics, and the combined system is large, has blocks with very different scales, and needs specialized block preconditioners. See [Sparse linear solvers](sparse-linear-solvers.md#condition-number-and-preconditioning).

| | Partitioned | Monolithic |
| --- | --- | --- |
| Software | Reuse best-in-class solvers for each field | One multiphysics code |
| Flexibility | High: swap solvers, add fields | Lower |
| Strong coupling | Needs sub-iterations, acceleration; can be unstable | Robust |
| Cost per step | Several single-field solves per sub-iteration | One large coupled solve |
| Typical tools | preCICE with OpenFOAM, CalculiX and others; commercial co-simulation environments | Multiphysics FE frameworks, custom coupled solvers |

## The added-mass instability in fluid-structure interaction

When a structure in a fluid accelerates, it must also accelerate the fluid around it. That fluid behaves like extra ("added") mass attached to the structure. In a partitioned scheme with explicit coupling, the fluid force used to move the structure is computed from the *previous* structural motion. If the added mass is large compared with the structure's own mass, the correction overshoots each time, and the error grows from step to step, **whatever the time step**.

It is a problem for light structures in dense fluids: blood flow in arteries and heart valves, thin membranes in water, light panels in liquids. Remedies: strong coupling with sub-iterations and quasi-Newton acceleration, interface preconditioning (for example Robin-type interface conditions), or a monolithic formulation. For an aircraft wing in air (structure much denser than the fluid), loose coupling is usually stable.

## Why this matters for machine learning

- **One-way chains are easy to build from single-physics surrogates.** A thermal surrogate followed by a structural surrogate works, but errors propagate: the structural model sees the thermal model's errors as input shifts, possibly out of its training distribution. Train the second model on inputs that include realistic upstream errors, or train the chain end to end.
- **Two-way coupling of surrogates is harder.** Iterating two learned models to a fixed point may not converge, or may converge to the wrong point, because neither model was trained to be accurate at the other's errors. A single surrogate for the coupled response, trained on coupled simulations, is often more reliable.
- **Interface quantities make good targets.** Interface forces, heat fluxes and temperatures are what the coupling depends on, and they are low-dimensional compared with the full fields.
- **Data is expensive.** Coupled simulations cost more than either single field, which pushes towards multi-fidelity training (many one-way runs, a few two-way runs). See [Multi-fidelity learning](multi-fidelity-learning.md).
- **Surrogates can accelerate coupling itself,** for example as a cheap predictor for the interface state at the start of each sub-iteration, keeping the coupled solver's convergence check in charge.

## In an interview

**Thirty-second answer to "One-way vs two-way coupling?"** "In one-way coupling one field drives another without feedback: solve the thermal problem, map the temperatures and run the stress analysis. In two-way coupling each field changes the other, like a joint whose opening changes contact conductance, or a flexible wing whose deformation changes its loads, so you must iterate or solve them together. I decide by estimating the feedback: if it changes the result less than the accuracy I need, one-way is fine."

**Thirty-second answer to "Partitioned vs monolithic?"** "Partitioned coupling uses separate solvers that exchange interface data, explicitly once per step or implicitly with sub-iterations and acceleration; it is flexible and reuses specialized codes but can be unstable for strong coupling, like the added-mass effect when a light structure moves in a dense fluid. Monolithic coupling solves all fields in one system: robust, but it needs one code with all the physics and good block preconditioners."

**Likely follow-ups**

- *What is the thermal stress in a fully constrained steel bar heated by 100 K?* About −252 MPa: −EαΔT.
- *What is the added-mass instability?* Explicit partitioned FSI with a light structure in a dense fluid overshoots the interface force every step and diverges regardless of time step.
- *How would you couple an OpenFOAM model with a structural solver?* A coupling library such as preCICE, with implicit coupling and quasi-Newton acceleration if the coupling is strong.

## References

- [preCICE: a coupling library for partitioned multi-physics simulations](https://github.com/precice/precice)
- P. Causin, J.-F. Gerbeau and F. Nobile, "Added-mass effect in the design of partitioned algorithms for fluid-structure problems", *Computer Methods in Applied Mechanics and Engineering*, 2005.
- T. L. Bergman, A. S. Lavine, F. P. Incropera and D. P. DeWitt, *Fundamentals of Heat and Mass Transfer*, 8th edition, Wiley, 2017.
