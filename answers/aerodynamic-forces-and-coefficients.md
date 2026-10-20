# Aerodynamic Forces and Coefficients

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** The aerodynamic force on a body is the integral over its surface of two contributions: pressure acting normal to the surface and wall shear stress acting along it, **F = ∮ (−(p − p∞) n + τ_w) dA**, with n the unit normal pointing out of the body into the fluid. **Drag** is the component along the free-stream direction and **lift** the component perpendicular to it. Dividing by the dynamic pressure ½ρU² and a reference area gives the dimensionless **drag and lift coefficients**, C_D = D / (½ρU²A_ref) and C_L = L / (½ρU²A_ref), where the reference area is the frontal area for cars and the wing planform area for aircraft. For bluff bodies such as cars, most of the drag comes from pressure, and it is the small difference between large forces on the front and the rear. That makes drag very sensitive to small pressure errors, which is why a surrogate's drag must be evaluated separately from its field error.

**On this page**

- [Where the force comes from](#where-the-force-comes-from)
- [Drag, lift and the coefficients](#drag-lift-and-the-coefficients)
- [Pressure drag and friction drag](#pressure-drag-and-friction-drag)
- [Computing forces from simulation data](#computing-forces-from-simulation-data)
- [Code: drag on a sphere and d'Alembert's paradox](#code-drag-on-a-sphere-and-dalemberts-paradox)
- [Why drag is hard to predict accurately](#why-drag-is-hard-to-predict-accurately)
- [In an interview](#in-an-interview)

## Where the force comes from

A fluid can only push on a surface in two ways:

- **Pressure** acts perpendicular to each surface element, pushing inwards. Its force on a small area dA is −p n dA, where n points out of the body into the fluid.
- **Wall shear stress** τ_w acts tangentially, dragging the surface along with the flow (viscous friction).

The total force is the sum over the whole wetted surface:

```math
\mathbf{F} = \oint_S \left( -(p - p_\infty)\,\mathbf{n} + \boldsymbol{\tau}_w \right) dA
```

Subtracting the free-stream pressure p∞ does not change the result for a closed surface (a uniform pressure exerts no net force, because ∮ n dA = 0), but it makes numerical integration more accurate and is how the pressure coefficient is defined. **Moments** (pitching, rolling, yawing) come from the same integrand: M = ∮ r × dF about a reference point.

## Drag, lift and the coefficients

- **Drag D:** the force component parallel to the free-stream velocity.
- **Lift L:** the component perpendicular to the free stream, in the plane of symmetry (for a car, negative lift is downforce).
- **Side force:** the remaining component, important in crosswind (yaw).

Dimensionless coefficients make results comparable across speeds and sizes:

```math
C_D = \frac{D}{\tfrac{1}{2}\rho U^2 A_{\text{ref}}}, \qquad C_L = \frac{L}{\tfrac{1}{2}\rho U^2 A_{\text{ref}}}, \qquad C_p = \frac{p - p_\infty}{\tfrac{1}{2}\rho U^2}, \qquad C_f = \frac{|\boldsymbol{\tau}_w|}{\tfrac{1}{2}\rho U^2}
```

- ½ρU² is the **dynamic pressure** q: the pressure rise if the flow were brought to rest without losses. At the front stagnation point, C_p = 1.
- **Reference area:** the frontal (projected) area for cars and trucks, the wing planform area for aircraft. Always state which one is used; comparing coefficients with different reference areas is a classic mistake. Automotive engineers often quote **C_D·A** (drag area) to avoid it.
- **Drag counts:** in aerospace, one count is 0.0001 of C_D; in automotive, engineers often mean 0.001. A modern passenger car has a C_D of roughly 0.25 to 0.35.

**Worked example.** A car with C_D = 0.30 and frontal area 2.2 m² at 30 m/s (108 km/h) in air of density 1.2 kg/m³:

- q = ½ × 1.2 × 30² = 540 Pa;
- D = 540 × 0.30 × 2.2 ≈ **356 N**;
- power to overcome it: D × U ≈ **10.7 kW**.

Drag grows with U² and power with U³: at 130 km/h (36.1 m/s), the same car needs about 516 N and 18.6 kW. This is why a few drag counts matter for the range of an electric car at motorway speed.

## Pressure drag and friction drag

- **Friction (skin-friction, viscous) drag:** the streamwise component of ∮ τ_w dA. Dominant for streamlined bodies with attached flow: an aircraft wing in cruise, a ship hull, a flat plate aligned with the flow.
- **Pressure (form) drag:** the streamwise component of ∮ −(p − p∞) n dA. Dominant for bluff bodies with separated wakes: cars, trucks, cylinders, buildings. The low pressure in the wake behind the body pulls it backwards. See [Flow separation](boundary-layers-and-wall-treatment.md#flow-separation).
- **Induced drag:** on finite wings, the drag associated with producing lift (the tip vortices); it appears in the pressure integral.
- **Wave drag:** from shocks in transonic and supersonic flow.

For a typical car, pressure drag is by far the larger part, and much of it comes from the rear of the car and the wheels. This is why rear-end shape, underbody, wheels and cooling flow are the levers of automotive aerodynamics.

## Computing forces from simulation data

On a triangulated surface, the integral becomes a sum over faces f:

```math
\mathbf{F} \approx \sum_f \left( -(p_f - p_\infty)\,\mathbf{n}_f + \boldsymbol{\tau}_{w,f} \right) A_f
```

Practical checks, all of which come up in coding interviews and real pipelines:

1. **Normals must point out of the body.** For a closed surface, check that Σ n_f A_f ≈ 0 and that normals point away from the body centre for convex parts. Flipped normals flip the pressure force.
2. **Use face values consistently.** If pressure is stored at nodes (point data), average it to faces; if at cells, use the cell values. Mixing them shifts the result.
3. **Use the correct surfaces.** Include the body and the wheels; exclude the ground and the domain walls. Datasets often label surface patches.
4. **Units and reference values.** Pressure in Pa or as C_p? Kinematic pressure (p/ρ, as OpenFOAM's incompressible solvers store it)? Wall shear stress sign convention (some codes store the force on the fluid, not on the wall)?
5. **Compare with the solver's own force report** on a few cases before trusting a post-processing script.

Vectorized code for areas, normals and the force integral is in [Mesh data processing: drag force from pressure and wall shear](mesh-data-processing.md#drag-force-from-pressure-and-wall-shear).

## Code: drag on a sphere and d'Alembert's paradox

A triangulated sphere with two model pressure distributions: the inviscid potential-flow solution, and a crude "separated wake" in which the rear pressure stays low. The second part shows how a small pressure error on the front changes the drag.

```python
import numpy as np

def uv_sphere(n_theta=90, n_phi=180, R=1.0):
    """Triangulated sphere with poles on the x axis; the flow comes from -x."""
    th = np.linspace(0, np.pi, n_theta + 1)
    ph = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing="ij")
    V = R * np.stack([np.cos(T), np.sin(T) * np.cos(P), np.sin(T) * np.sin(P)], -1).reshape(-1, 3)
    idx = np.arange(V.shape[0]).reshape(n_theta + 1, n_phi)
    F = []
    for i in range(n_theta):
        for j in range(n_phi):
            a, b, c, d = idx[i, j], idx[i, (j + 1) % n_phi], idx[i + 1, (j + 1) % n_phi], idx[i + 1, j]
            if i > 0:
                F.append([a, b, c])
            if i < n_theta - 1:
                F.append([a, c, d])
    return V, np.array(F)

def face_force(V, F, p_rel, tau):
    """Force per face: (-(p - p_inf) n + tau) A, with n pointing out of the body."""
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    n = cross / (2 * area[:, None])
    centre = (v0 + v1 + v2) / 3
    n *= np.sign(np.einsum("ij,ij->i", n, centre))[:, None]          # orient normals outwards
    return ((-p_rel[:, None]) * n + tau) * area[:, None], centre

V, F = uv_sphere()
q, R = 1.0, 1.0                                                       # dynamic pressure, radius
_, centre = face_force(V, F, np.zeros(len(F)), np.zeros((len(F), 3)))
theta = np.arccos(np.clip(-centre[:, 0] / R, -1, 1))                 # angle from the front stagnation point
cp_potential = 1 - 2.25 * np.sin(theta) ** 2                          # inviscid potential flow around a sphere
cp_separated = np.where(theta < np.radians(100), cp_potential, -0.3) # crude separated wake

cds = {}
for name, cp in [("potential flow", cp_potential), ("separated wake", cp_separated),
                 ("separated, front Cp +0.02", cp_separated + 0.02 * (centre[:, 0] < 0))]:
    dF, _ = face_force(V, F, cp * q, np.zeros((len(F), 3)))
    cds[name] = dF[:, 0].sum() / (q * np.pi * R**2)                   # reference area = frontal area
    print(f"{name:28s}: Cd = {cds[name]:+.4f}")
```

Output:

```
potential flow              : Cd = +0.0000
separated wake              : Cd = +0.2021
separated, front Cp +0.02   : Cd = +0.2221
```

1. **d'Alembert's paradox.** In inviscid potential flow the pressure on the rear recovers exactly to mirror the front, and the drag is zero. Real drag exists because viscosity creates boundary layers that separate, so the rear pressure does not recover.
2. **Sensitivity.** Raising the pressure on the front half by just 2% of the dynamic pressure raises the drag by 10% in this example. A field model with "2% pressure error" can have a much larger drag error, or, if errors on front and rear partly cancel, a deceptively small one.

## Why drag is hard to predict accurately

- **It is a small difference of large numbers.** Pressure forces on the front and rear of a car are each larger than the net drag; small, systematic errors in either do not cancel.
- **The rear and the wake decide it,** and that is where separation, unsteadiness and turbulence-model errors live. See [Why surrogates struggle near separation](boundary-layers-and-wall-treatment.md#why-surrogates-struggle-near-separation).
- **Friction drag needs the boundary layer**, which sits in a very thin region near the wall that point-sampled models can under-resolve.
- **Field metrics hide it.** A low average pressure error says little about the integral. Always report the error in C_D and C_L directly, along with how well designs are *ranked*. See [Integrated quantities](evaluation-metrics.md#integrated-quantities).
- **Labels have their own uncertainty.** Time-averaged scale-resolving drag carries statistical noise; RANS drag carries model error. The surrogate cannot be more accurate than that floor in any meaningful sense.

Some models predict the integrated coefficients directly with a separate output head, or train with an extra loss on C_D computed from the predicted surface fields, so that the integral is optimized explicitly.

## In an interview

**Thirty-second answer to "How do you compute drag and lift from surface pressure and wall shear?"** "Integrate over the surface: each face contributes minus the pressure relative to free stream times its outward normal times its area, plus the wall shear stress times its area. Drag is the component of the total along the free stream, lift the component perpendicular to it. Divide by the dynamic pressure, half rho U squared, and the reference area to get C_D and C_L. In code: compute face areas and normals with cross products, make sure normals point out of the body, use consistent face values, and check against the solver's force report."

**Likely follow-ups**

- *Why is a car's drag mostly pressure drag?* The flow separates at the rear, so the wake pressure stays low and pulls the car back.
- *What is d'Alembert's paradox?* Inviscid potential flow predicts zero drag on any body; real drag comes from viscous effects and separation.
- *Your model's surface pressure error is 2%. Is the drag good?* Not necessarily: errors integrate unevenly; check C_D directly.

## References

- J. D. Anderson, *Fundamentals of Aerodynamics*, 6th edition, McGraw-Hill, 2016.
- W.-H. Hucho (editor), *Aerodynamics of Road Vehicles*, 4th edition, SAE International, 1998.
- [AirfRANS (metrics on surface forces and drag/lift ranking)](https://arxiv.org/abs/2212.07564)
- [DrivAerML (force coefficients for 500 car variants)](https://arxiv.org/abs/2408.11969)
