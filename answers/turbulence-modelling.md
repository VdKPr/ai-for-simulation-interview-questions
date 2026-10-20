# Turbulence Modelling

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Turbulence is the chaotic, three-dimensional, unsteady motion that most engineering flows have at high Reynolds number. It is expensive to simulate because it contains eddies of every size, from the size of the body down to tiny scales where viscosity turns kinetic energy into heat, and a direct simulation must resolve all of them in space and time: the cost grows roughly like Re³. **DNS** resolves every scale and is limited to research. **LES** resolves the large, energy-carrying eddies and models only the small ones; it is accurate for separated and unsteady flows but expensive, especially near walls. **RANS** solves for the time-averaged flow and models the effect of *all* turbulent fluctuations with a turbulence model; it is cheap enough for design iterations and is the industry workhorse. Hybrid RANS-LES methods use RANS near walls and LES in separated regions. A RANS turbulence model such as **Spalart-Allmaras** or **k-ω SST** supplies an *eddy viscosity* that represents how turbulence mixes momentum, computed from one or two extra transport equations.

**On this page**

- [Laminar vs turbulent flow](#laminar-vs-turbulent-flow)
- [Why turbulence is expensive to simulate](#why-turbulence-is-expensive-to-simulate)
- [RANS vs LES vs DNS](#rans-vs-les-vs-dns)
- [What a turbulence model does](#what-a-turbulence-model-does)
- [Spalart-Allmaras, k-epsilon, k-omega and SST](#spalart-allmaras-k-epsilon-k-omega-and-sst)
- [Where RANS models fail](#where-rans-models-fail)
- [Why this matters for machine learning](#why-this-matters-for-machine-learning)
- [In an interview](#in-an-interview)

## Laminar vs turbulent flow

- **Laminar flow:** smooth, orderly layers sliding over each other. Predictable, and mixing happens only by molecular diffusion. Typical of low Reynolds numbers: thin films, slow flows, small devices.
- **Turbulent flow:** irregular, swirling eddies of many sizes, constantly forming and breaking up. It mixes momentum, heat and species far more effectively, which raises skin friction and heat transfer, and it is chaotic: tiny differences in initial conditions grow, so only *statistics* (means, fluctuations, spectra) are reproducible.

Transition from laminar to turbulent depends on the Reynolds number, the pressure gradient, surface roughness and disturbances in the incoming flow. See [The Reynolds number](cfd-essentials.md#the-reynolds-number).

**The energy cascade.** Large eddies extract energy from the mean flow, break up into smaller eddies, which break up into smaller ones still, until at the **Kolmogorov scale** η = (ν³/ε)^(1/4) (ε is the dissipation rate) viscosity converts the motion into heat. The ratio of the largest eddy size L to η grows like Re^(3/4).

## Why turbulence is expensive to simulate

**Counting grid points.** To resolve every eddy, the mesh spacing must be of the order of η and the domain must contain the largest eddies. The number of points in each direction scales like L/η ∝ Re^(3/4), so in 3D:

```math
N_{\text{points}} \propto \left(\text{Re}^{3/4}\right)^3 = \text{Re}^{9/4}
```

The time step must also resolve the fastest small-scale motion, adding about another factor of Re^(3/4) in the number of steps. Together, the cost of a direct numerical simulation grows roughly like **Re³**.

**What that means for a car.** At Re ≈ 10⁷, Re^(9/4) is about 10^15.75, of order 10¹⁵ grid points, before even counting the boundary layers. Today's largest DNS runs are many orders of magnitude smaller. Doubling the speed of the car multiplies the DNS cost by about eight.

**Walls make it worse.** Near a wall, the important eddies become very small, so resolving the boundary layer dominates the cost. Choi and Moin (2012) estimated grid-point requirements for a flat-plate boundary layer of roughly Re^(37/14) for DNS, Re^(13/7) for **wall-resolved** LES and only Re (linear) for **wall-modelled** LES, which is why wall models are essential for LES at industrial Reynolds numbers.

**And statistics take time.** Because turbulence is chaotic, a single snapshot is not the answer. Mean forces and fields need averaging over many flow-through times or shedding cycles, which multiplies the run time again.

## RANS vs LES vs DNS

**RANS: Reynolds-averaged Navier–Stokes.** Split each variable into a mean and a fluctuation, u = ū + u′, and average the equations. The averaged equations look like the original ones for ū, plus an extra term, the **Reynolds stresses** −ρ⟨u′u′⟩, that represents how fluctuations transport momentum. They are new unknowns with no equation of their own: this is the **closure problem**, and a turbulence model provides the missing relation.

**LES: large-eddy simulation.** Apply a spatial filter instead of a time average. Eddies larger than the filter width (roughly the mesh size) are resolved in a transient simulation; the effect of smaller, more universal eddies is modelled by a *subgrid-scale model* (Smagorinsky, WALE, dynamic models). Near walls, either resolve the small near-wall eddies (wall-resolved LES, very expensive) or model the inner layer (wall-modelled LES).

**Hybrid RANS-LES** (detached-eddy simulation, DES, and its variants DDES and IDDES): RANS in attached boundary layers where it works well, LES in separated regions and wakes where RANS struggles. This is the practical high-fidelity choice for cars and aircraft; the DrivAerML and AhmedML datasets were produced with hybrid RANS-LES.

**DNS: direct numerical simulation.** No turbulence model: every scale is resolved. Used for fundamental research at modest Reynolds numbers and to generate reference data for model development.

| | What is resolved | What is modelled | Relative cost | Typical use |
| --- | --- | --- | --- | --- |
| **RANS** (steady) | Mean flow only | All turbulence | Lowest: minutes to hours | Design iterations, optimization, most industrial CFD |
| **URANS** | Mean flow plus large, slow unsteadiness | All turbulence | Low to moderate | Vortex shedding, periodic flows |
| **Hybrid RANS-LES** | Large eddies away from walls | Near-wall turbulence, small eddies | High | High-fidelity automotive and aerospace |
| **Wall-modelled LES** | Large eddies, including in the outer boundary layer | Inner wall layer, small eddies | High | Separated flows, aeroacoustics, high-fidelity datasets |
| **Wall-resolved LES** | Large eddies everywhere, including near walls | Small eddies | Very high | Research, low-Re components |
| **DNS** | Everything | Nothing | Prohibitive at industrial Re | Research, reference data |

## What a turbulence model does

In RANS, the job of the turbulence model is to supply the Reynolds stresses. Most industrial models use the **Boussinesq eddy-viscosity hypothesis**: turbulence mixes momentum like an enhanced viscosity, so the Reynolds stresses are proportional to the mean strain rate,

```math
-\rho\,\overline{u_i' u_j'} = \mu_t\left(\frac{\partial \bar u_i}{\partial x_j} + \frac{\partial \bar u_j}{\partial x_i}\right) - \tfrac{2}{3}\rho k\,\delta_{ij},
```

where k is the turbulent kinetic energy and μ_t the **eddy viscosity**. Unlike the molecular viscosity, μ_t is a property of the flow, not the fluid: large in the middle of a turbulent wake, zero in the free stream and at the wall. It can be hundreds or thousands of times the molecular viscosity.

So the model's job reduces to computing μ_t everywhere. One- and two-equation models do it by solving additional transport equations, with production, dissipation and diffusion terms calibrated on canonical flows (channel flow, flat-plate boundary layers, mixing layers, jets).

## Spalart-Allmaras, k-epsilon, k-omega and SST

| Model | Equations | Strengths | Weaknesses | Typical use |
| --- | --- | --- | --- | --- |
| **Spalart-Allmaras (SA)** | One, for a modified eddy viscosity | Robust, cheap, good for attached and mildly separated boundary layers | Less accurate for free shear flows, strong separation, decaying turbulence | External aerodynamics of aircraft and wings |
| **k-ε** | Two: turbulent kinetic energy k and its dissipation ε | Robust for free shear flows (jets, mixing layers) | Poor near walls without wall functions or damping; over-predicts turbulence in stagnation and adverse-pressure-gradient regions, so it predicts separation too late | General industrial flows, internal flows with wall functions |
| **k-ω** (Wilcox) | Two: k and the specific dissipation rate ω | Good near walls, integrates to the wall | Sensitive to the free-stream value of ω | Boundary layers |
| **k-ω SST** (Menter) | Two: k and ω, blended | k-ω near the wall, k-ε behaviour away from it; a limiter on the shear stress improves separation under adverse pressure gradients | Still an eddy-viscosity model with its general limits | A common default for external and internal flows prone to separation |
| **Reynolds stress models** | Transport equations for each Reynolds stress (seven) | Capture anisotropy, curvature and rotation effects better | Costly, less robust | Swirling flows, strong curvature, secondary flows |
| **Transition models** (for example γ-Re_θ) | Extra equations for intermittency | Predict laminar-to-turbulent transition | Need calibration; sensitive to inflow turbulence | Turbomachinery, wind-turbine blades, low-Re aerofoils |

**Spalart-Allmaras** was developed for aerospace flows: it solves one equation for a working variable related to the eddy viscosity, with terms for production, destruction near walls and diffusion. It is the standard in many external-aerodynamics solvers.

**k-ω SST** (shear-stress transport) combines two ideas. Near the wall, the k-ω formulation is accurate and robust; in the free stream, k-ε avoids k-ω's sensitivity to the inlet value of ω. A blending function switches between them. The SST limiter bounds the eddy viscosity so that the turbulent shear stress does not exceed what is observed in adverse pressure gradients, which delays the over-prediction of attached flow that plagues k-ε. The AirfRANS dataset, for example, uses k-ω SST.

**Inlet turbulence quantities matter.** Two-equation models need inlet values of k and ω (or ε), usually from a turbulence intensity and a length scale or viscosity ratio. Different choices change transition-like behaviour and wakes. Datasets should record them.

## Where RANS models fail

Linear eddy-viscosity models assume turbulence behaves like an isotropic viscosity aligned with the mean strain. That breaks down in:

- **massive separation and bluff-body wakes:** drag on a car with a large rear wake can be off by several percent, and separation points can be wrong;
- **flows with strong curvature or rotation:** cyclones, rotating machinery, swirl;
- **secondary flows driven by turbulence anisotropy:** corner flows in square ducts;
- **laminar-to-turbulent transition:** standard models assume the boundary layer is turbulent everywhere;
- **unsteady phenomena:** a steady RANS run cannot show vortex shedding or buffeting. See [Steady vs transient simulation](cfd-essentials.md#steady-vs-transient-simulation).

These failures are why high-fidelity datasets use scale-resolving methods, and why "RANS truth" labels must be treated with care.

## Why this matters for machine learning

- **A surrogate trained on RANS learns RANS,** including its errors in separated flow. Matching the training data closely is not the same as matching reality. See [When the labels have discretization error](verification-validation-and-mesh-convergence.md#when-the-labels-have-discretization-error).
- **Fidelity is a data decision.** Many cheap RANS runs or a few expensive scale-resolving runs? Multi-fidelity learning combines both. See [Multi-fidelity learning](multi-fidelity-learning.md) and [Design: a RANS + LES surrogate](design-rans-les-multi-fidelity-surrogate.md).
- **Scale-resolving labels are noisy.** Time-averaged LES fields carry statistical noise from finite averaging windows; the noise level should be estimated before judging surrogate errors.
- **ML can improve the turbulence model itself.** Data-driven closures learn corrections to the Reynolds stresses or the eddy-viscosity model from high-fidelity data, and learned subgrid or interpolation models inside LES and DNS solvers let coarse grids match finer ones. Kochkov et al. (2021) reported 2D turbulence simulations as accurate as baseline solvers with 8 to 10 times finer resolution in each spatial dimension, for 40 to 80 times speed-ups, by learning parts of the discretization inside the solver. See [Correcting a coarse solver](differentiable-physics-and-hybrid-solvers.md#correcting-a-coarse-solver).
- **Chaos limits what can be predicted.** For turbulent flows, a surrogate can predict statistics and the large-scale evolution for a while, not the exact instantaneous field for long. See [What a surrogate can predict for a chaotic flow](autoregressive-rollouts.md#what-a-surrogate-can-predict-for-a-chaotic-flow).

## In an interview

**Thirty-second answer to "RANS vs LES vs DNS?"** "DNS resolves every eddy down to the Kolmogorov scale, with cost growing like Re cubed, so it is research-only. LES resolves the large, energy-carrying eddies and models the small, more universal ones; it is accurate for separated, unsteady flows but expensive, especially near walls unless wall-modelled. RANS solves only the mean flow and models all turbulence through an eddy viscosity; it is cheap and the workhorse for design, but struggles with massive separation, transition and strong curvature. Hybrid RANS-LES uses RANS in attached boundary layers and LES in wakes, which is what high-fidelity automotive datasets use."

**Thirty-second answer to "What does a turbulence model like Spalart-Allmaras or SST do?"** "RANS averaging creates unknown Reynolds stresses. Eddy-viscosity models assume they act like an extra viscosity proportional to the mean strain rate, and compute that eddy viscosity from transport equations: one equation in Spalart-Allmaras, two in k-ω SST, which blends k-ω near walls with k-ε in the free stream and limits the shear stress to predict separation better."

**Likely follow-ups**

- *Why is DNS of a car impossible?* About Re^(9/4) ≈ 10¹⁵ grid points at Re ≈ 10⁷, plus a similar growth in time steps.
- *Which model would you use for a car?* k-ω SST RANS for design iterations; hybrid RANS-LES for high-fidelity reference data.
- *What is the closure problem?* Averaging the nonlinear equations produces Reynolds stresses with no equation of their own.

## References

- S. B. Pope, *Turbulent Flows*, Cambridge University Press, 2000.
- D. C. Wilcox, *Turbulence Modeling for CFD*, 3rd edition, DCW Industries, 2006.
- [NASA Turbulence Modeling Resource: model descriptions (including SA and SST) and verification cases](https://turbmodels.larc.nasa.gov/)
- H. Choi and P. Moin, "Grid-point requirements for large eddy simulation: Chapman's estimates revisited", *Physics of Fluids*, 2012.
- F. R. Menter, "Two-equation eddy-viscosity turbulence models for engineering applications", *AIAA Journal*, 1994.
- [Machine learning accelerated computational fluid dynamics (Kochkov et al., 2021)](https://arxiv.org/abs/2102.01010)
