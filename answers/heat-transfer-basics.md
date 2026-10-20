# Heat Transfer Basics

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Heat moves in three ways. **Conduction** is diffusion of heat through a material, described by Fourier's law q = −k∇T. **Convection** is heat carried away by a moving fluid, usually summarized by Newton's law of cooling q = h(T_s − T∞) with a heat-transfer coefficient h that depends on the flow, not just the fluid. **Radiation** is emission of electromagnetic energy, q = εσ(T_s⁴ − T_sur⁴), which matters at high temperatures and, surprisingly often, in natural-convection electronics cooling. In a simulation they appear as boundary conditions (fixed temperature, fixed flux, convective and radiative conditions) or are solved together: **conjugate heat transfer** solves conduction in solids and convection in the surrounding fluid as one coupled problem. Dimensionless numbers organize the physics: the **Prandtl** number compares momentum and thermal diffusion in a fluid, the **Nusselt** number is a dimensionless heat-transfer coefficient, and the **Biot** number compares conduction inside a solid with convection at its surface. They are also the natural inputs and outputs for a thermal surrogate.

**On this page**

- [Conduction, convection and radiation](#conduction-convection-and-radiation)
- [Boundary conditions for thermal problems](#boundary-conditions-for-thermal-problems)
- [Conjugate heat transfer](#conjugate-heat-transfer)
- [Dimensionless numbers: Prandtl, Nusselt and Biot](#dimensionless-numbers-prandtl-nusselt-and-biot)
- [Code: heat-transfer coefficients and a Biot check](#code-heat-transfer-coefficients-and-a-biot-check)
- [Why this matters for machine learning](#why-this-matters-for-machine-learning)
- [In an interview](#in-an-interview)

## Conduction, convection and radiation

**Conduction.** Heat flows from hot to cold through a material, in proportion to the temperature gradient:

```math
\mathbf{q} = -k\,\nabla T, \qquad \rho c_p \frac{\partial T}{\partial t} = \nabla\cdot(k\nabla T) + \dot q
```

with k the thermal conductivity, ρc_p the volumetric heat capacity and q̇ a heat source (for example Joule heating in a conductor or a chip). Typical conductivities: copper about 400 W/m·K, aluminium about 200, steels about 15 to 50, glass about 1, plastics about 0.2, still air about 0.026. The ratio α = k/(ρc_p) is the **thermal diffusivity**: how fast temperature changes spread.

**Convection.** A fluid flowing past a surface carries heat away. The detailed physics is the flow and energy equations near the wall (the thermal boundary layer), but engineers summarize it by

```math
q = h\,(T_s - T_\infty)
```

The **heat-transfer coefficient h** is not a property of the fluid: it depends on velocity, geometry, flow regime and position on the surface. Rough ranges:

| Situation | h (W/m²·K) |
| --- | --- |
| Natural convection in air | about 2 to 25 |
| Forced convection in air | about 25 to 250 |
| Forced convection in liquids | about 100 to 20,000 |
| Boiling and condensation | about 2,500 to 100,000 |

*Natural* (free) convection is driven by buoyancy, warm fluid rising; *forced* convection by a fan, a pump or vehicle motion.

**Radiation.** Every surface emits thermal radiation in proportion to the fourth power of its absolute temperature:

```math
q = \varepsilon\,\sigma\left(T_s^4 - T_{\text{sur}}^4\right), \qquad \sigma = 5.67\times10^{-8}\ \text{W/m}^2\text{K}^4
```

with emissivity ε between 0 and 1 (polished metals are low, painted or anodized surfaces high). Between several surfaces, **view factors** describe how much each surface "sees" of the others. Radiation dominates at high temperatures (furnaces, exhausts, spacecraft), but it also matters near room temperature: linearized, it behaves like a heat-transfer coefficient of about 4εσT³, roughly 5.5 W/m²·K at 300 K for ε = 0.9, which is comparable to natural convection in air. Ignoring radiation in a passively cooled electronics enclosure can therefore give badly wrong temperatures.

## Boundary conditions for thermal problems

| Condition | Form | Example |
| --- | --- | --- |
| Fixed temperature (Dirichlet) | T = T₀ | Surface held by a large heat sink or a controlled bath |
| Fixed heat flux (Neumann) | −k ∂T/∂n = q₀ | Heater film, solar load, chip power spread over a face |
| Insulated or symmetry (Neumann, zero flux) | ∂T/∂n = 0 | Well-insulated surfaces, symmetry planes |
| Convection (Robin) | −k ∂T/∂n = h(T − T∞) | Air- or liquid-cooled surfaces, with h from correlations or CFD |
| Radiation (nonlinear) | −k ∂T/∂n = εσ(T⁴ − T_sur⁴) | Hot surfaces, enclosures, space |
| Contact resistance | Temperature jump ΔT = R_c · q across an interface | Bolted or clamped parts, thermal interface materials |
| Volumetric source | q̇ in the heat equation | Joule heating, chips, battery cells, reactions |

The general guide to Dirichlet, Neumann and Robin conditions is in [Boundary conditions](boundary-conditions.md#dirichlet-neumann-and-robin-conditions).

## Conjugate heat transfer

**Definition.** Conjugate heat transfer (CHT) solves heat **conduction in the solid** and **convection in the fluid** together, as one coupled problem. At every solid-fluid interface, two conditions hold:

- the temperature is continuous: T_solid = T_fluid;
- the heat flux is continuous: k_solid ∂T/∂n = k_fluid ∂T/∂n.

So nobody has to guess a heat-transfer coefficient: the fluid simulation computes how heat leaves the surface, and the solid simulation computes how it gets there.

**When you need it:**

- when h is not known in advance or varies strongly over the surface: heat sinks and fins, cooling channels, impinging jets;
- when the solid and fluid strongly influence each other: electronics cooling (chips, heat spreaders, fans, enclosures), battery packs with cooling plates, heat exchangers, turbine blades with internal cooling passages, engine and brake cooling, data-centre airflow;
- when local hot spots matter more than average temperatures.

**When a simpler approach is enough:** if reliable correlations for h exist (for example fully developed pipe flow), a solid-only conduction model with convective boundary conditions is much cheaper.

**Numerical points:**

- **Different time scales.** Fluids respond in fractions of a second, solids take minutes to hours to heat up. Transient CHT is stiff; codes often use steady CHT, or different time steps in the two regions.
- **Near-wall resolution.** Heat-transfer accuracy needs a well-resolved thermal boundary layer (first cells at low y+), and thermal wall functions can be significantly off. See [Wall functions](boundary-layers-and-wall-treatment.md#wall-functions).
- **Coupling strategy.** One solver with a shared mesh (monolithic) or two solvers exchanging interface temperatures and fluxes (partitioned). See [Partitioned vs monolithic coupling](multiphysics-coupling.md#partitioned-vs-monolithic-coupling).

## Dimensionless numbers: Prandtl, Nusselt and Biot

**Prandtl number: what the fluid is like.**

```math
\text{Pr} = \frac{\nu}{\alpha} = \frac{\text{momentum diffusivity}}{\text{thermal diffusivity}}
```

It is a fluid property: about 0.7 for air, about 7 for water at room temperature, hundreds to thousands for oils, about 0.01 for liquid metals. It sets the relative thickness of the velocity and thermal boundary layers: for Pr well above 1 the thermal layer is thinner than the velocity layer (roughly δ_t/δ ∝ Pr^(−1/3) in laminar flow), for liquid metals it is much thicker.

**Nusselt number: how good the convection is.**

```math
\text{Nu} = \frac{h L}{k_{\text{fluid}}}
```

It is the heat-transfer coefficient made dimensionless with the fluid's conductivity: the ratio of convective heat transfer to what pure conduction through the fluid layer would carry. Correlations express it in terms of the other numbers, for example Dittus-Boelter for fully turbulent pipe flow, Nu = 0.023 Re^0.8 Pr^0.4 (fluid being heated). For natural convection, the Rayleigh number Ra = Gr·Pr (buoyancy against viscous and thermal diffusion) replaces the Reynolds number.

**Biot number: whether the solid's interior matters.**

```math
\text{Bi} = \frac{h L_c}{k_{\text{solid}}} = \frac{\text{internal conduction resistance}}{\text{external convection resistance}}
```

with L_c a characteristic length of the solid (volume over surface area). It looks like the Nusselt number but uses the *solid's* conductivity and answers a different question:

- **Bi < 0.1:** conduction inside is fast compared with convection at the surface, so the solid is nearly isothermal. A **lumped-capacitance** model (one temperature for the whole part) is accurate, and the cooling curve is a simple exponential.
- **Bi of order 1 or larger:** significant temperature gradients inside; resolve the conduction.

**Why use them?** They collapse many cases onto one curve. Two pipes of different sizes with different fluids have the same Nusselt number if they have the same Reynolds and Prandtl numbers. That reduces the number of independent inputs, lets results transfer across scales and fluids, and gives surrogates physically meaningful features and outputs. The Fourier number Fo = αt/L² is the dimensionless time of transient conduction, and the Péclet number Pe = Re·Pr compares convection with thermal diffusion.

## Code: heat-transfer coefficients and a Biot check

```python
fluids = {                       # nu (m^2/s), k (W/m K), Pr: approximate values near room temperature
    "air":   (1.5e-5, 0.026, 0.71),
    "water": (1.0e-6, 0.60, 7.0),
}

def pipe_h(fluid, U, D):
    """Heat-transfer coefficient for turbulent flow in a pipe (Dittus-Boelter, fluid being heated)."""
    nu, k, Pr = fluids[fluid]
    Re = U * D / nu
    assert Re >= 1e4, "Dittus-Boelter is for fully turbulent flow, Re >= 10,000"
    Nu = 0.023 * Re**0.8 * Pr**0.4
    return Re, Nu, Nu * k / D

for fluid, U in [("air", 20.0), ("water", 1.0)]:
    Re, Nu, h = pipe_h(fluid, U, D=0.01)
    print(f"{fluid:5s} at {U:4.1f} m/s in a 10 mm pipe: Re = {Re:7.0f}, Nu = {Nu:5.1f}, h = {h:6.0f} W/m^2K")

sigma, eps, T = 5.67e-8, 0.9, 300.0
print(f"linearized radiation coefficient at 300 K, emissivity 0.9: {4 * eps * sigma * T**3:.1f} W/m^2K")

h, k_al, thickness = 50.0, 200.0, 0.002              # air-cooled aluminium fin, 2 mm thick
Bi = h * (thickness / 2) / k_al                      # characteristic length = volume / area ~ half-thickness
print(f"Biot number of the fin: {Bi:.1e} -> {'lumped (uniform) temperature is fine' if Bi < 0.1 else 'resolve conduction'}")
```

Output:

```
air   at 20.0 m/s in a 10 mm pipe: Re =   13333, Nu =  40.0, h =    104 W/m^2K
water at  1.0 m/s in a 10 mm pipe: Re =   10000, Nu =  79.4, h =   4763 W/m^2K
linearized radiation coefficient at 300 K, emissivity 0.9: 5.5 W/m^2K
Biot number of the fin: 2.5e-04 -> lumped (uniform) temperature is fine
```

Water at a twentieth of the speed transfers heat about 45 times better than air, which is why liquid cooling wins for high power densities. Radiation near room temperature is worth about as much as natural convection.

## Why this matters for machine learning

- **Non-dimensional inputs and outputs.** Feeding Re, Pr and geometry ratios, and predicting Nu or a non-dimensional temperature (T − T_ref)/(q L / k), lets one model cover many sizes and fluids, and makes extrapolation errors visible as ranges of dimensionless numbers. See [Non-dimensionalization](nondimensionalization-and-normalization.md#why-non-dimensionalize).
- **Correlations are strong baselines.** A surrogate for a heat exchanger or a cooling channel should beat the textbook correlation before anyone cares.
- **The Biot number tells you whether a field model is needed.** If Bi is small, a single temperature per component (a thermal network) may be enough; if not, the surrogate must predict internal temperature fields.
- **Radiation and contact resistances are easy to forget** in training data generation, and a surrogate trained without them will be confidently wrong on passively cooled designs.
- **Multiple time scales** make transient thermal surrogates closer to stiff problems: slow solid heating driven by fast fluid changes. See [Stiff equations](time-integration-and-stability.md#stiff-equations).

A worked system design is in [Design: a thermal surrogate for a battery pack or an electronics enclosure](design-thermal-surrogate.md).

## In an interview

**Thirty-second answer to "Explain conduction, convection and radiation, and their boundary conditions."** "Conduction is diffusion through a material, q = −k∇T; its boundary conditions are fixed temperature (Dirichlet), fixed flux or insulation (Neumann). Convection is heat carried by a moving fluid, summarized as q = h(T − T∞), which is a Robin condition on the solid, with h depending on the flow. Radiation goes with the fourth power of absolute temperature, εσ(T⁴ − T_sur⁴), a nonlinear boundary condition; near room temperature it is worth about 5 W/m²K, similar to natural convection. When h is not known in advance, conjugate heat transfer solves the fluid and the solid together with continuous temperature and heat flux at the interface."

**Likely follow-ups**

- *What does the Biot number tell you?* Whether the solid can be treated as isothermal (Bi < 0.1) or needs a conduction model.
- *Nusselt vs Biot?* Same form, hL/k, but Nusselt uses the fluid's conductivity and describes convection; Biot uses the solid's and compares internal and external resistance.
- *Why does water cool so much better than air?* Higher conductivity and heat capacity, and a Prandtl number of about 7, giving a far higher h at much lower speed.

## References

- T. L. Bergman, A. S. Lavine, F. P. Incropera and D. P. DeWitt, *Fundamentals of Heat and Mass Transfer*, 8th edition, Wiley, 2017.
- J. H. Lienhard IV and J. H. Lienhard V, [*A Heat Transfer Textbook* (free online edition)](https://ahtt.mit.edu/)
- [preCICE: partitioned coupling, including conjugate heat transfer tutorials](https://github.com/precice/precice)
