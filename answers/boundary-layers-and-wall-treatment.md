# Boundary Layers, Wall Treatment and Separation

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A **boundary layer** is the thin region next to a wall where the flow slows from the free-stream velocity to zero at the surface (the no-slip condition). Viscosity matters there even at high Reynolds number, and almost all skin friction and heat transfer happen there. **y+** is the distance from the wall measured in viscous units, y+ = y·u_τ/ν, where u_τ = √(τ_w/ρ) is the friction velocity. It tells you which part of the boundary layer the first mesh cell sits in: the viscous sublayer (y+ below about 5), the buffer layer (about 5 to 30) or the logarithmic layer (above about 30). That decides the near-wall mesh: **wall-resolved** models need the first cell at y+ ≈ 1 with many thin prism layers, while **wall functions** put the first cell in the log layer and use the law of the wall to bridge the gap, which is cheaper but only reliable for attached, near-equilibrium boundary layers. When an adverse pressure gradient slows the near-wall flow until the wall shear stress drops to zero, the boundary layer **separates**, creating recirculation and a wake. Separation is sensitive to small geometry changes, hard for RANS models and steep or even discontinuous as a function of the design, which is why surrogates struggle near it.

**On this page**

- [The boundary layer and y+](#the-boundary-layer-and-y)
- [The law of the wall](#the-law-of-the-wall)
- [How y+ drives near-wall mesh design](#how-y-drives-near-wall-mesh-design)
- [Code: first-cell height for a target y+](#code-first-cell-height-for-a-target-y)
- [Wall functions](#wall-functions)
- [Flow separation](#flow-separation)
- [Why surrogates struggle near separation](#why-surrogates-struggle-near-separation)
- [In an interview](#in-an-interview)

## The boundary layer and y+

At a solid wall the fluid velocity equals the wall velocity (no slip). A short distance away it is close to the free-stream value. The region in between is the boundary layer, and its thickness δ grows along the surface:

- **laminar** (Blasius): δ/x ≈ 5 / √Re_x;
- **turbulent** (a common flat-plate estimate): δ/x ≈ 0.37 Re_x^(−1/5),

where x is the distance from the leading edge and Re_x = Ux/ν. On a car roof 2 m from the front at 30 m/s, Re_x = 4 × 10⁶ and the turbulent boundary layer is only about 35 mm thick, compared with a car length of 4.5 m. Thin, but it controls skin friction, heat transfer and whether the flow separates.

**Wall units.** The natural scales near the wall are set by the wall shear stress τ_w:

```math
u_\tau = \sqrt{\frac{\tau_w}{\rho}}, \qquad y^+ = \frac{y\,u_\tau}{\nu}, \qquad u^+ = \frac{u}{u_\tau}
```

u_τ is the **friction velocity** and ν/u_τ the viscous length scale. y+ is the wall distance in units of that length: y+ = 1 means one viscous length from the wall. Because τ_w varies over a body, the same physical first-cell height gives different y+ values in different places: large where the wall shear is high, close to zero near stagnation and separation points.

## The law of the wall

Experiments and DNS show that, in wall units, the mean velocity profile of an attached turbulent boundary layer has a nearly universal inner structure:

| Region | Range (approximate) | Profile | Physics |
| --- | --- | --- | --- |
| Viscous sublayer | y+ < 5 | u+ = y+ | Viscous stress dominates |
| Buffer layer | 5 < y+ < 30 | Smooth blend | Viscous and turbulent stresses comparable; turbulence production peaks |
| Log layer | 30 < y+ < a few hundred (grows with Re) | u+ = (1/κ) ln y+ + B, with κ ≈ 0.41, B ≈ 5 | Turbulent stress dominates |
| Outer (wake) region | Beyond, up to δ | Depends on pressure gradient and history | Not universal |

```
 u+
 25 ┤                                          ╱‾‾‾ outer region
 20 ┤                                   ╱‾‾‾‾‾      log law: u+ = 2.44 ln y+ + 5
 15 ┤                           ╱‾‾‾‾‾
 10 ┤                  ╭───────╯   buffer
  5 ┤        ╱‾‾‾‾‾‾‾‾╯
    ┤    ╱ viscous sublayer: u+ = y+
  0 ┼────┬────────┬─────────┬──────────┬──── y+ (log scale)
     1    5       30        100        1000
```

This universality is what wall functions exploit.

## How y+ drives near-wall mesh design

There are two strategies, and the turbulence model's wall treatment must match the mesh:

**1. Wall-resolved (low-Re) modelling.** Resolve the whole boundary layer down to the wall. The first cell centre should be at **y+ ≈ 1** (and certainly below about 5), followed by a stack of thin **prism layers** growing gradually (a growth ratio of about 1.2 or less) until they cover the boundary layer. Accurate for skin friction, heat transfer and separation, but the near-wall cells are tiny.

**Worked numbers** for the car roof above: at x = 2 m, U = 30 m/s, a flat-plate correlation gives a skin-friction coefficient of about 0.0030, τ_w ≈ 1.6 Pa and u_τ ≈ 1.16 m/s. For y+ = 1, the first cell centre must be about **0.013 mm** from the wall (code below). To cover the 35 mm boundary layer with a first cell of about 0.026 mm and a growth ratio of 1.2 takes about 30 prism layers. Over a whole car surface, that adds millions of cells.

**2. Wall functions (high-Re modelling).** Place the first cell centre in the log layer, **30 < y+ < about 300**, and use the law of the wall to compute the wall shear stress from the velocity at that cell. For the same car roof, y+ = 30 to 100 means a first cell at about 0.4 to 1.3 mm. Far fewer and larger cells, so much cheaper.

**Avoid the buffer layer** (5 < y+ < 30) with standard wall functions: neither the sublayer nor the log law holds there, and results degrade. Many codes offer *all-y+* or *automatic* wall treatments that blend the two approaches, which makes them less sensitive to where the first cell lands, but the guidance still applies.

**Check it after the run.** Since τ_w comes from the solution, y+ is only known afterwards. Plot the y+ distribution on the surface and refine or coarsen the first layer if needed. See [Sanity checks for CFD](verification-validation-and-mesh-convergence.md#sanity-checks-for-cfd).

## Code: first-cell height for a target y+

```python
import numpy as np

def first_cell_height(U, x, y_plus_target, nu=1.5e-5, rho=1.2):
    """Estimate the wall distance of the first cell centre for a target y+,
    using a flat-plate skin-friction correlation (turbulent boundary layer)."""
    Re_x = U * x / nu
    cf = (2 * np.log10(Re_x) - 0.65) ** (-2.3)        # Schlichting correlation, valid for Re_x < 1e9
    tau_w = 0.5 * rho * U**2 * cf                      # wall shear stress
    u_tau = np.sqrt(tau_w / rho)                       # friction velocity
    return y_plus_target * nu / u_tau, Re_x, cf, u_tau

for y_plus in [1, 30, 100]:
    y, Re_x, cf, u_tau = first_cell_height(U=30.0, x=2.0, y_plus_target=y_plus)
    print(f"y+ = {y_plus:>3}: first cell at {y * 1e3:.4f} mm   (Re_x = {Re_x:.1e}, Cf = {cf:.5f}, u_tau = {u_tau:.3f} m/s)")
delta = 0.37 * 2.0 * (30.0 * 2.0 / 1.5e-5) ** (-0.2)   # turbulent boundary-layer thickness estimate
print(f"boundary-layer thickness at x = 2 m: about {delta * 1e3:.0f} mm")
```

Output:

```
y+ =   1: first cell at 0.0130 mm   (Re_x = 4.0e+06, Cf = 0.00297, u_tau = 1.156 m/s)
y+ =  30: first cell at 0.3892 mm   (Re_x = 4.0e+06, Cf = 0.00297, u_tau = 1.156 m/s)
y+ = 100: first cell at 1.2974 mm   (Re_x = 4.0e+06, Cf = 0.00297, u_tau = 1.156 m/s)
boundary-layer thickness at x = 2 m: about 35 mm
```

This is an estimate for a flat plate; on a real body, pressure gradients change the local wall shear, so the estimate sets the starting mesh and the post-run y+ plot decides.

## Wall functions

**What they do.** Instead of resolving the steep velocity gradient between the wall and the log layer, a wall function assumes the log law holds between the wall and the first cell. Given the velocity at the first cell centre and its distance from the wall, it solves the law of the wall for u_τ, and from that sets the wall shear stress and the near-wall turbulence quantities. Thermal wall functions do the same for heat flux using a temperature law of the wall.

**When they are acceptable:**

- high-Reynolds-number, **attached** flows in near-equilibrium, with mild pressure gradients;
- when the goal is global quantities (pressure drop in a pipe network, approximate forces) rather than detailed separation or heat transfer;
- in early design studies and very large models where wall-resolved meshes are unaffordable;
- in wall-modelled LES, where they make high-Reynolds-number LES feasible.

**When they are not:**

- predicting **separation and reattachment** accurately: the log law does not hold where τ_w goes to zero and the flow reverses;
- strong adverse pressure gradients, strong curvature, impinging jets;
- **low Reynolds numbers** where there is no real log layer;
- **laminar-to-turbulent transition**;
- **accurate heat-transfer coefficients**, where thermal wall functions can be significantly off;
- when the first cell lands in the buffer layer.

## Flow separation

**Mechanism.** Where the pressure rises in the flow direction (an *adverse pressure gradient*, for example on the rear of a car, the aft part of a wing at high angle of attack or a widening diffuser), the slow fluid near the wall is pushed backwards. It decelerates until, at the **separation point**, the velocity gradient at the wall and the wall shear stress reach zero. Downstream, the near-wall flow reverses and the boundary layer lifts off the surface, leaving a **recirculation region** and a wake.

```
  flow ──▶  ════════════════╗
            attached BL      ╲   separation point (wall shear = 0)
                              ╲___
                               ↺ ↺   recirculation (reversed flow near the wall)
                                  ↺ ↺  ── wake ──▶
```

**Consequences:**

- **Pressure drag rises:** the pressure in the wake does not recover, so the low pressure on the rear of the body pulls it backwards. For bluff bodies like cars and trucks, pressure drag dominates skin-friction drag.
- **Lift is lost:** a wing *stalls* when separation spreads over its upper surface.
- **The flow becomes unsteady:** shedding vortices, buffeting, noise.

**Separation is sensitive to geometry.** The classic example is the Ahmed body, a simplified car shape with a slanted rear: as the slant angle increases, drag rises up to a critical angle near 30°, and then *drops abruptly* when the flow switches from staying attached on the slant (with strong side vortices) to fully separating. A few degrees change the flow topology. Real cars show the same kind of sensitivity to rear-end shapes, spoilers and diffusers.

**Turbulent boundary layers resist separation better** than laminar ones because turbulent mixing brings faster fluid towards the wall. That is the reason for the drag crisis of spheres and cylinders and the dimples on golf balls, and why transition location matters. See [The Reynolds number](cfd-essentials.md#the-reynolds-number).

## Why surrogates struggle near separation

1. **Steep and discontinuous responses.** Small geometry changes can move the separation point a long way or switch the flow topology (as on the Ahmed body). Smooth regressors blur these jumps, and averaging losses put little weight on them.
2. **Bistability and hysteresis.** Near the critical angle, two flow states can exist, and which one a simulation finds may depend on the initial condition or solver history. Labels there are effectively noisy.
3. **The labels are least reliable there.** RANS models are weakest in separated flow, and time-averaged scale-resolving labels converge slowly in large unsteady wakes. A surrogate can match its data and still be wrong.
4. **Thin regions decide the outcome.** Separation is triggered inside the boundary layer, so point sampling or subsampling that under-represents the near-wall region loses the information that matters.
5. **Data imbalance.** Most designs in a dataset may be attached or fully separated, with few samples near the transition, which is exactly where accuracy matters.
6. **It is often the decision point.** The interesting designs (lowest drag, highest lift before stall) sit near the edge of separation.

**What helps:** sample more densely near transitions ([active learning](design-of-experiments-and-active-learning.md#active-learning-and-acquisition-functions)); give the model features that carry the physics (surface normals, curvature, wall distance, local pressure gradients); predict a separation indicator or flow regime as an explicit output; report errors separately for attached and separated designs; and use uncertainty estimates to route near-separation designs to the solver. See [Low average error but failures where it matters](evaluation-metrics.md#low-average-error-but-failures-where-it-matters).

## In an interview

**Thirty-second answer to "What is y+ and why does it drive the mesh?"** "y+ is wall distance in viscous units, y times the friction velocity over viscosity. It tells you whether the first cell is in the viscous sublayer, the buffer layer or the log layer. Wall-resolved turbulence models need the first cell at y+ around 1 with about 20 to 30 prism layers growing slowly; wall functions put the first cell in the log layer, y+ of 30 to a few hundred, and use the law of the wall instead. For a car at 30 m/s that is the difference between about 0.01 mm and about 1 mm for the first cell, so it changes the mesh size enormously. You check y+ after the run because it depends on the computed wall shear."

**Likely follow-ups**

- *When are wall functions acceptable?* Attached, high-Re, near-equilibrium flows where global quantities matter; not for separation, transition, low Re or accurate heat transfer.
- *What is flow separation?* An adverse pressure gradient decelerates the near-wall flow until the wall shear stress is zero and the flow reverses; the boundary layer detaches and forms a wake.
- *Why do surrogates struggle near it?* Steep, bistable responses, unreliable labels, thin regions that decide the outcome, and few training samples near the transition.

## References

- H. Schlichting and K. Gersten, *Boundary-Layer Theory*, 9th edition, Springer, 2017.
- S. B. Pope, *Turbulent Flows*, Cambridge University Press, 2000 (the law of the wall).
- [Dimensionless wall distance (y plus), CFD Online wiki](https://www.cfd-online.com/Wiki/Dimensionless_wall_distance_(y_plus))
- S. R. Ahmed, G. Ramm and G. Faltin, "Some salient features of the time-averaged ground vehicle wake", SAE Technical Paper 840300, 1984.
- [AhmedML: High-Fidelity CFD Dataset for Incompressible, Low-Speed Bluff Body Aerodynamics (Ashton et al.)](https://arxiv.org/abs/2407.20801)
