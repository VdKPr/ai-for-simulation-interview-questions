# Thermal Stress and Warpage

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** When bonded layers with different coefficients of thermal expansion (CTE) change temperature, each layer tries to expand or shrink by a different amount, but the bond forces them to share the same length at the interface. The layer that wants to grow more is held in compression, the other in tension, and because these forces act at different heights in the stack they create a bending moment: the stack curves, exactly like a bimetallic strip in a thermostat. That curvature is **warpage**. In electronics packages, power modules, printed circuit boards and coated parts it is often built in during manufacturing, because the layers are bonded at a high temperature and then cooled, so the parts are warped and stressed at room temperature even before use. The same mismatch, repeated over thermal cycles, drives solder and interface fatigue. Symmetric stack-ups, CTE matching and compliant layers reduce it; simulation must include the stress-free bonding temperature and, often, plasticity and viscoelasticity of the layers.

**On this page**

- [The mechanism](#the-mechanism)
- [The bimetallic strip: Timoshenko's formula](#the-bimetallic-strip-timoshenkos-formula)
- [Code: curvature and bow of a two-layer strip](#code-curvature-and-bow-of-a-two-layer-strip)
- [Where warpage matters](#where-warpage-matters)
- [How designers reduce warpage](#how-designers-reduce-warpage)
- [Simulating warpage](#simulating-warpage)
- [Why this matters for machine learning](#why-this-matters-for-machine-learning)
- [In an interview](#in-an-interview)

## The mechanism

Think of two strips, copper (CTE about 17 × 10⁻⁶ per kelvin) and a ceramic (about 7 × 10⁻⁶ per kelvin), each 50 mm long and free. Heat both by 100 K: the copper grows by 85 µm, the ceramic by 35 µm. Nothing else happens.

Now bond them together first and then heat them:

1. They must have the same length at the interface. The copper is held back (compressed); the ceramic is pulled along (stretched).
2. The compressive force in the copper and the tensile force in the ceramic are equal and opposite, but they act at different heights in the stack. Together they form a **bending moment**.
3. The stack bends so that the layer that expanded more (the copper) is on the outside (convex) side. On cooling, the reverse: the copper shrinks more and ends up on the concave side.

```
 before heating:   ══════════  copper             after heating:     ╭──────────╮  copper (convex side)
                   ▓▓▓▓▓▓▓▓▓▓  ceramic                               ╰▓▓▓▓▓▓▓▓▓▓╯ ceramic
```

The stresses are largest near the interface and especially at the **free edges** of the layers, where the shear and peeling stresses that try to delaminate the interface concentrate. This is why cracks and delamination often start at the edges and corners of bonded layers.

**The key quantity is the mismatch strain** Δα·ΔT. With a CTE difference of 10 × 10⁻⁶ per kelvin and a temperature change of 200 K, the mismatch is 0.2%, already of the order of the yield strain of many metals.

## The bimetallic strip: Timoshenko's formula

For two bonded elastic layers (thicknesses t₁ and t₂, Young's moduli E₁ and E₂, CTEs α₁ and α₂) after a uniform temperature change ΔT, Timoshenko's 1925 analysis gives the curvature κ = 1/R:

```math
\kappa = \frac{6\,(\alpha_2 - \alpha_1)\,\Delta T\,(1 + m)^2}{h\left[\,3(1 + m)^2 + (1 + m n)\left(m^2 + \dfrac{1}{m n}\right)\right]}, \qquad m = \frac{t_1}{t_2},\; n = \frac{E_1}{E_2},\; h = t_1 + t_2
```

For two layers of equal thickness and stiffness (m = n = 1), it reduces to κ = 1.5 Δα ΔT / h. The lessons:

- curvature grows linearly with the **CTE mismatch** and the **temperature change**;
- it falls as the stack gets **thicker** (κ ∝ 1/h);
- the stiffness and thickness ratios shift it, and it is largest when the layers are comparable.

The out-of-plane deflection ("bow") of a strip of length L bent to a small, uniform curvature is approximately δ = κL²/8, so warpage grows with the **square** of the part size. A large panel warps much more than a small one at the same curvature.

For a **thin film on a thick substrate** (a coating, or a deposited layer on a wafer), **Stoney's equation** relates the film stress σ_f to the substrate curvature: κ = 6 σ_f t_f / (E′_s t_s²), with E′_s the substrate's biaxial modulus. It is used in reverse to measure film stress from wafer curvature.

## Code: curvature and bow of a two-layer strip

```python
def bilayer_curvature(t1, t2, E1, E2, a1, a2, dT):
    """Timoshenko (1925) curvature of a bonded two-layer strip after a uniform temperature change dT.
    Layer 1 and 2: thickness t, Young's modulus E, expansion coefficient a. Positive curvature bends
    the strip towards the layer with the smaller expansion (layer 1 if a1 < a2)."""
    m, n, h = t1 / t2, E1 / E2, t1 + t2
    return 6 * (a2 - a1) * dT * (1 + m) ** 2 / (h * (3 * (1 + m) ** 2 + (1 + m * n) * (m ** 2 + 1 / (m * n))))

def bow(curvature, length):
    """Out-of-plane deflection (sagitta) of a strip of given length bent to a small, uniform curvature."""
    return curvature * length ** 2 / 8

# Check against the closed form for equal thickness and modulus: kappa = 1.5 * da * dT / h
k_check = bilayer_curvature(1e-3, 1e-3, 1e11, 1e11, 10e-6, 20e-6, 100)
print(f"equal layers: {k_check:.4f} 1/m vs closed form {1.5 * 10e-6 * 100 / 2e-3:.4f} 1/m")

# Steel / brass bimetal strip, 0.5 mm each, heated by 100 K, 50 mm long
k = bilayer_curvature(0.5e-3, 0.5e-3, 200e9, 100e9, 12e-6, 19e-6, 100)
print(f"steel/brass bimetal: curvature {k:.3f} 1/m, radius {1 / k:.2f} m, bow over 50 mm = {bow(k, 0.05) * 1e3:.3f} mm")

# Copper (0.3 mm) bonded to a ceramic plate (0.6 mm), cooled by 250 K after bonding, 50 mm long
k = bilayer_curvature(0.6e-3, 0.3e-3, 300e9, 120e9, 7e-6, 17e-6, -250)
print(f"copper on ceramic, cooled 250 K: curvature {k:.3f} 1/m, bow over 50 mm = {bow(k, 0.05) * 1e3:.3f} mm")
```

Output:

```
equal layers: 0.7500 1/m vs closed form 0.7500 1/m
steel/brass bimetal: curvature 1.018 1/m, radius 0.98 m, bow over 50 mm = 0.318 mm
copper on ceramic, cooled 250 K: curvature -2.874 1/m, bow over 50 mm = -0.898 mm
```

A one-sided copper layer cooled from a bonding temperature 250 K above room temperature bows a 50 mm plate by almost a millimetre in this elastic estimate. Real values are smaller because copper yields and the formula ignores edge effects, but the order of magnitude shows why one-sided layers are avoided. (These are illustrative textbook material values, not data for any specific product.)

## Where warpage matters

- **Electronics packages:** a silicon die (CTE about 3 × 10⁻⁶ per kelvin), mould compound and an organic substrate (often 15 × 10⁻⁶ per kelvin or more) cured or soldered at high temperature. Warpage during reflow soldering causes open or bridged solder joints.
- **Solder joint fatigue:** a component and the board expand differently in every power or ambient temperature cycle, shearing the solder joints a little each time. This is one of the main reliability limits of electronics. See [Fatigue](fatigue.md).
- **Power-electronics modules:** metallized ceramic substrates soldered or sintered to baseplates; mismatched layers warp during assembly and drive cracking and delamination in thermal cycling.
- **Printed circuit boards:** uneven copper distribution between layers makes boards warp.
- **Coatings and thin films:** thermal barrier coatings on turbine blades, deposited films on wafers.
- **Composite laminates:** a laminate with an unsymmetric lay-up warps after curing, because the plies have very different CTEs along and across the fibres.
- **Welding and additive manufacturing:** local heating and cooling leave residual stresses and distortion.

## How designers reduce warpage

- **Symmetric stack-ups.** Put matching layers on both sides of the neutral plane, so the bending moments cancel (copper on both sides of a ceramic, symmetric composite lay-ups, balanced copper in PCBs). Symmetry cancels curvature, not the in-plane stresses.
- **CTE matching.** Choose materials whose CTEs are closer, or add intermediate layers with intermediate CTE.
- **Compliant layers.** Soft adhesives, underfills or thin ductile interlayers absorb the mismatch strain.
- **Geometry.** Thicker or stiffer substrates curve less; patterning a metal layer (dimples, slots, separated pads) reduces its effective stiffness and the bending it causes; rounded corners reduce edge stress concentrations.
- **Process control.** Lower bonding temperatures, controlled cooling, pre-bending of parts to compensate.

## Simulating warpage

- **Thermo-mechanical FEA** with thermal strain α(T − T_ref). The **stress-free reference temperature** T_ref is the bonding, curing or solidification temperature, not room temperature. Getting it wrong gives a completely wrong warpage. See [Thermo-mechanical coupling, with numbers](multiphysics-coupling.md#thermo-mechanical-coupling-with-numbers).
- **Temperature-dependent properties:** CTE and modulus change with temperature, sometimes strongly (polymers near their glass transition).
- **Nonlinear materials:** plasticity of metal layers and solder, creep of solder, viscoelasticity of mould compounds and adhesives. Warpage then depends on the thermal history, not only the end temperatures. See [Plasticity and load-history dependence](nonlinear-structural-analysis.md#plasticity-and-load-history-dependence).
- **Geometric nonlinearity** when the deflection is comparable to the thickness of thin plates.
- **Thin layers** need enough elements through each layer's thickness, or layered shell elements.
- **Validation:** warpage is measured with techniques such as shadow moiré or digital image correlation, often over a temperature profile.

## Why this matters for machine learning

- **Warpage surrogates for layout variants.** Predicting warpage or peak interface stress from a layout (for example an image or graph of the metal pattern, plus the stack-up and the thermal profile) is a natural surrogate task: many variants, an expensive nonlinear FE analysis for each. Inputs that carry physics help: metal area fraction on each side, asymmetry, distance to edges.
- **Built-in physics as a baseline.** A Timoshenko-type estimate or a laminate-theory calculation is a cheap baseline and a useful input feature; a model should beat it.
- **History dependence.** With plasticity and creep in the layers, the answer depends on the thermal profile, so the profile must be an input, or the model restricted to one standard process. See [Why path dependence complicates surrogates](nonlinear-structural-analysis.md#why-path-dependence-complicates-surrogates).
- **Peak edge stresses** are local and sensitive to mesh refinement near free edges and corners (they can be singular in the elastic solution), so they need careful label definitions. See [Stress concentrations and singularities](stress-concentrations-and-singularities.md).

## In an interview

**Thirty-second answer to "Why does a CTE mismatch cause warpage in layered parts?"** "Bonded layers with different expansion coefficients try to change length by different amounts when the temperature changes, but the bond forces equal length at the interface. One layer ends up in compression and the other in tension, and because those forces act at different heights they create a bending moment: the stack curves like a bimetallic strip. The curvature scales with the CTE difference times the temperature change, over the thickness, and the bow with the square of the length. In electronics the layers are usually bonded hot and cooled, so parts are warped at room temperature; symmetric stack-ups cancel the curvature."

**Likely follow-ups**

- *What is the stress-free temperature?* The temperature at which the layers were joined and are free of thermal stress; usually the bonding or curing temperature.
- *How do you reduce warpage?* Symmetric layers, closer CTEs, compliant interlayers, thicker substrates, patterned metal, lower process temperatures.
- *Why do cracks start at the edges?* Interface shear and peel stresses concentrate at free edges and corners of bonded layers.

## References

- S. Timoshenko, "Analysis of bi-metal thermostats", *Journal of the Optical Society of America*, 1925.
- G. G. Stoney, "The tension of metallic films deposited by electrolysis", *Proceedings of the Royal Society of London A*, 1909.
- J. H. Lau (editor), *Thermal Stress and Strain in Microelectronics Packaging*, Van Nostrand Reinhold, 1993.
