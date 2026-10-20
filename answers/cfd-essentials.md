# CFD Essentials

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Computational fluid dynamics (CFD) solves the Navier–Stokes equations, which express conservation of mass and momentum (and energy, when heat matters), on a mesh around or inside a geometry. The **Reynolds number**, the ratio of inertial to viscous effects, decides the character of the flow: laminar and smooth at low values, turbulent with a huge range of eddy sizes at the values typical of cars, aircraft and pumps. Because resolving all turbulent eddies is unaffordable in industry, most simulations solve time-averaged equations with a turbulence model (RANS); scale-resolving methods (LES, hybrid RANS-LES) are used when unsteady separated flow matters. Below a Mach number of about 0.3 the flow can be treated as incompressible, and then pressure is not given by an equation of state but acts to keep the velocity field divergence-free, which is why solvers need special **pressure-velocity coupling** algorithms. A surrogate trained on CFD inherits every one of these modelling choices.

**On this page**

- [The incompressible Navier-Stokes equations](#the-incompressible-navier-stokes-equations)
- [The Reynolds number](#the-reynolds-number)
- [Incompressible vs compressible flow](#incompressible-vs-compressible-flow)
- [Steady vs transient simulation](#steady-vs-transient-simulation)
- [Pressure-velocity coupling](#pressure-velocity-coupling)
- [The rest of CFD in this guide](#the-rest-of-cfd-in-this-guide)
- [What a surrogate must respect](#what-a-surrogate-must-respect)
- [In an interview](#in-an-interview)

## The incompressible Navier-Stokes equations

For a Newtonian fluid with constant density ρ and viscosity μ:

```math
\nabla \cdot \mathbf{u} = 0
```

```math
\rho\left(\underbrace{\frac{\partial \mathbf{u}}{\partial t}}_{\text{unsteady}} + \underbrace{(\mathbf{u}\cdot\nabla)\mathbf{u}}_{\text{convection}}\right) = \underbrace{-\nabla p}_{\text{pressure}} + \underbrace{\mu \nabla^2 \mathbf{u}}_{\text{viscous diffusion}} + \underbrace{\mathbf{f}}_{\text{body force}}
```

**Term by term:**

- **∇·u = 0 (continuity):** conservation of mass for constant density. Whatever flows into a small volume flows out; the velocity field is *divergence-free*.
- **ρ ∂u/∂t (local acceleration):** the velocity at a fixed point changing in time. Zero in a steady flow.
- **ρ (u·∇)u (convective acceleration):** fluid speeding up or turning as it moves through a non-uniform flow, such as flow accelerating around the top of a wing. It is **nonlinear** (velocity times its own gradient), and it is the source of turbulence, chaos and most of the difficulty of fluid dynamics.
- **−∇p (pressure gradient):** fluid is pushed from high to low pressure.
- **μ∇²u (viscous diffusion):** friction between neighbouring layers of fluid smooths out velocity differences. It dominates near walls, where it creates the boundary layer.
- **f (body forces):** gravity, buoyancy (in a temperature-dependent density approximation), or apparent forces in a rotating frame.

Together, these are four equations (one continuity, three momentum components) for four unknowns: three velocity components and pressure. When heat transfer matters, an **energy equation** is added for the temperature. See [Heat transfer basics](heat-transfer-basics.md).

**Non-dimensional form.** Scaling lengths by L, velocities by U, time by L/U and pressure by ρU² gives

```math
\frac{\partial \mathbf{u}^*}{\partial t^*} + (\mathbf{u}^*\cdot\nabla^*)\mathbf{u}^* = -\nabla^* p^* + \frac{1}{\text{Re}}\nabla^{*2}\mathbf{u}^*, \qquad \nabla^*\cdot\mathbf{u}^* = 0 .
```

The only parameter left is the Reynolds number. Two geometrically similar flows with the same Re have the same non-dimensional solution, which is the principle behind wind-tunnel testing of scale models and a strong reason to non-dimensionalize ML inputs. See [Why non-dimensionalize](nondimensionalization-and-normalization.md#why-non-dimensionalize).

## The Reynolds number

```math
\text{Re} = \frac{\rho U L}{\mu} = \frac{U L}{\nu} = \frac{\text{inertial forces}}{\text{viscous forces}}
```

with U a characteristic velocity, L a characteristic length and ν = μ/ρ the kinematic viscosity (about 1.5 × 10⁻⁵ m²/s for air and 1.0 × 10⁻⁶ m²/s for water at room temperature).

**Typical values:**

| Flow | U | L | Re (order of magnitude) |
| --- | --- | --- | --- |
| Car on a motorway (air) | 30 m/s | 4.5 m (length) | 10⁷ |
| Airliner wing in cruise | 250 m/s | 5 m (chord), thin air at altitude | 10⁷ |
| Water in a household pipe | 1 m/s | 0.02 m (diameter) | 2 × 10⁴ |
| Small drone propeller | 20 m/s | 0.02 m (chord) | 3 × 10⁴ |
| Blood in a capillary | 1 mm/s | 10 µm | 10⁻³ |

For the car: Re = 30 × 4.5 / 1.5 × 10⁻⁵ = 9 × 10⁶.

**What happens physically as Re increases:**

1. **Very low Re (much less than 1):** viscous forces dominate, inertia is negligible. The flow is smooth, reversible and depends linearly on the forcing (Stokes flow: microfluidics, bacteria, lubrication films).
2. **Moderate Re:** inertia matters. Behind a circular cylinder, a pair of steady recirculating vortices forms (Re of roughly 5 to 40).
3. **Onset of unsteadiness:** above about Re ≈ 47 for a cylinder, the wake becomes unsteady and sheds vortices alternately: the Kármán vortex street.
4. **Transition to turbulence:** in pipes, flow is laminar below about Re ≈ 2,300 (based on diameter) and turbulent at higher values; on a flat plate, the boundary layer typically becomes turbulent at a local Reynolds number around 5 × 10⁵, depending on disturbances.
5. **Fully turbulent flow:** chaotic, three-dimensional, with eddies from the size of the body down to tiny dissipative scales. The ratio of the largest to the smallest eddy size grows like Re^(3/4).
6. **Boundary layers get thinner:** a laminar boundary layer's thickness relative to the body length scales like Re^(−1/2), so near-wall meshes must get finer as Re increases. See [Boundary layers and y+](boundary-layers-and-wall-treatment.md).
7. **Separation behaviour changes:** a turbulent boundary layer resists adverse pressure gradients better than a laminar one, which explains the *drag crisis* of spheres and cylinders around Re ≈ 3 × 10⁵, where drag suddenly falls because the boundary layer becomes turbulent and separates later. This is also why golf balls have dimples.

## Incompressible vs compressible flow

All gases are compressible, but at low speeds their density barely changes. The governing number is the **Mach number** M = U/a, the ratio of flow speed to the speed of sound (about 340 m/s in air at sea level). For an isentropic flow, the relative density change is roughly M²/2:

| Mach number | Approximate density change | Treatment |
| --- | --- | --- |
| 0.1 | 0.5% | Incompressible |
| 0.3 | about 4.5% | Incompressible is usually acceptable (the common rule of thumb) |
| 0.5 to 0.7 | 12% to 25% | Compressible |
| 0.8 to 1.2 (transonic) | Large, with shocks on wings | Compressible, shock-capturing schemes |
| Above 1 (supersonic) | Shocks everywhere | Compressible |

**You need a compressible solver when:**

- the Mach number exceeds about 0.3 anywhere important: aircraft in cruise, turbomachinery, high-speed jets, valves and nozzles;
- there are **shocks**, discontinuities across which pressure, density and velocity jump (transonic wings in cruise have them);
- **acoustics** matter: sound is a compressibility effect, even in a low-speed flow (aeroacoustics of side mirrors, fans);
- **large temperature differences** change density strongly even at low speed: combustion, fire, strongly heated flows (often handled with *low-Mach-number* variable-density formulations rather than fully compressible ones).

Compressible solvers add an energy equation and an equation of state (p = ρRT for an ideal gas) and must carry acoustic waves, which limits explicit time steps. See [The CFL condition](time-integration-and-stability.md#the-cfl-condition).

**For ML:** shocks are discontinuities, which smooth neural networks represent poorly and blur. Predictions near shocks need models that can represent sharp features, conservative formulations, or explicit shock-location outputs.

## Steady vs transient simulation

A **steady** simulation looks for a flow that does not change in time (or, with RANS, a steady *mean* flow). A **transient** simulation resolves the evolution in time. Steady RANS is the industry workhorse because it is much cheaper, but it can mislead:

1. **The flow is inherently unsteady.** Bluff bodies shed vortices; separated wakes flap; rotating machinery has rotor-stator interactions. A steady solver may not converge (residuals and forces keep oscillating), or may converge to a solution that represents no real state of the flow.
2. **The time-averaged flow differs from the steady solution.** The average of an unsteady flow is not, in general, the steady solution of the same equations, because the equations are nonlinear. Drag from a steady RANS run of a car with a large wake can differ noticeably from the time-averaged drag of a scale-resolving run.
3. **There can be several solutions.** Near stall, or for asymmetric wakes behind symmetric bodies, a steady solver can land on one of several states depending on the initial guess, with hysteresis.
4. **Fluctuations are the quantity of interest.** Fatigue of structures in the flow, buffeting, vortex-induced vibration, aeroacoustics and peak loads all need the transient signal, not the mean.

**Warning signs** of a misleading steady result: residuals that stall at a high level or oscillate, forces that drift or oscillate with iteration count, results that change with the number of iterations or the initial condition.

**The remedy** is unsteady RANS (URANS), LES or hybrid RANS-LES, with averaging over enough flow-through times or shedding cycles to make the statistics converge. For datasets, check how long the averaging window was: short averages leave statistical noise in the labels. See [Turbulence modelling](turbulence-modelling.md).

## Pressure-velocity coupling

**The problem.** In incompressible flow there is no equation for pressure. Density is constant, so pressure is not linked to density by an equation of state. Instead, pressure takes whatever values are needed to make the velocity satisfy ∇·u = 0 everywhere. Mathematically, it acts like a Lagrange multiplier for the divergence-free constraint. Taking the divergence of the momentum equation gives a **pressure Poisson equation**: pressure at every point depends on the velocity field everywhere. Pressure is *global*: a change near the tail of a car is felt instantly at its nose.

**Segregated solvers** solve for velocity and pressure one after the other and iterate:

1. **SIMPLE** (Semi-Implicit Method for Pressure-Linked Equations), for steady flows:
   1. guess the pressure field;
   2. solve the momentum equations for a provisional velocity;
   3. this velocity does not satisfy continuity; solve a pressure-correction equation (Poisson-like) derived from continuity;
   4. correct pressure and velocity, with under-relaxation for stability;
   5. repeat until converged.
2. **PISO** (Pressure-Implicit with Splitting of Operators), for transient flows: several pressure corrections per time step, which keeps the flow consistent at each step without heavy under-relaxation.
3. **PIMPLE** (OpenFOAM): a combination of the two that allows larger time steps in transient runs by adding outer iterations.

**Coupled solvers** solve the momentum and continuity equations together as one large system. They need more memory, but they converge in far fewer iterations and are more robust for difficult flows, which is why many commercial codes offer them.

**Projection (fractional-step) methods**, common in research and GPU codes, first advance the velocity without the pressure constraint and then project it onto the space of divergence-free fields by solving a Poisson equation for pressure.

**A detail that appears in interviews:** on *collocated* meshes (all variables stored at cell centres, as in most finite-volume codes), naive interpolation lets a "checkerboard" pressure pattern go unnoticed. The Rhie-Chow interpolation of face velocities prevents it; staggered grids are the classic alternative.

**For ML:**

- Because pressure is global, a model predicting pressure needs a **global receptive field**: many message-passing steps, multiscale structure or attention. See [How many message-passing steps](gnns-and-meshgraphnets.md#how-many-message-passing-steps).
- Pressure in incompressible flow is defined only up to a constant unless a reference value is fixed (usually at the outlet). Make sure training data uses a consistent reference, or predict pressure coefficients relative to the free stream.
- A predicted velocity field can be projected onto divergence-free fields exactly, the same operation as in projection methods. See [Enforcing a divergence-free field exactly](hard-vs-soft-constraints.md#enforcing-a-divergence-free-field-exactly).

## The rest of CFD in this guide

- [Turbulence modelling: RANS, LES, DNS and turbulence models](turbulence-modelling.md)
- [Boundary layers, y+, wall functions and separation](boundary-layers-and-wall-treatment.md)
- [Aerodynamic forces and coefficients](aerodynamic-forces-and-coefficients.md)
- [External-aerodynamics boundary conditions](boundary-conditions.md#external-aerodynamics-boundary-conditions)
- [Conjugate heat transfer](heat-transfer-basics.md#conjugate-heat-transfer)
- [Open-source CFD solvers for generating data](generating-simulation-training-data.md#open-source-solvers-for-generating-data)
- [FEM vs FVM vs FDM](fem-vs-fvm-vs-fdm.md)

## What a surrogate must respect

- **Conservation:** zero divergence in incompressible flow; mass flow in equals mass flow out.
- **Boundary conditions:** no-slip at walls, free-stream values far away.
- **Thin, steep boundary layers:** most of the drag-relevant physics sits in a thin layer near the surface, so point sampling and model resolution must concentrate there.
- **Non-local pressure:** pressure at one point depends on the flow everywhere.
- **The regime of the training data:** Reynolds number, Mach number, turbulence model, steady or time-averaged. A surrogate trained on steady RANS reproduces RANS, including its errors in separated flow.

See [Hard vs soft constraints](hard-vs-soft-constraints.md) and [Physics-consistency checks](evaluation-metrics.md#physics-consistency-checks).

## In an interview

**Thirty-second answer to "Explain the incompressible Navier–Stokes equations."** "Continuity, divergence of velocity equals zero, is mass conservation for constant density. Momentum is Newton's second law per unit volume: density times the local acceleration plus the nonlinear convective acceleration equals minus the pressure gradient plus viscous diffusion plus body forces. The convective term makes it nonlinear and causes turbulence; viscosity creates boundary layers; and pressure has no equation of its own, it adjusts to keep the velocity divergence-free. In non-dimensional form the only parameter is the Reynolds number."

**Likely follow-ups**

- *What happens as Re increases?* Thinner boundary layers, transition to turbulence, a wider range of eddy sizes (ratio growing like Re^(3/4)), changes in separation, and drag crisis on curved bodies.
- *When do you need a compressible solver?* Above about Mach 0.3, with shocks, for acoustics, or with strong density changes from heating.
- *How does SIMPLE work?* Guess pressure, solve momentum, solve a pressure-correction equation from continuity, correct with under-relaxation, repeat.

## References

- [NASA Turbulence Modeling Resource](https://turbmodels.larc.nasa.gov/)
- [CFD Python: 12 steps to Navier–Stokes (Lorena Barba)](https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/)
- H. K. Versteeg and W. Malalasekera, *An Introduction to Computational Fluid Dynamics: The Finite Volume Method*, 2nd edition, Pearson, 2007.
- S. B. Pope, *Turbulent Flows*, Cambridge University Press, 2000.
- [DrivAerML: an example of high-fidelity hybrid RANS-LES training data (Ashton et al.)](https://arxiv.org/abs/2408.11969)
