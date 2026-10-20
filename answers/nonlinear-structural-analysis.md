# Nonlinear Structural Analysis

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A structural analysis becomes nonlinear when the stiffness depends on the solution. There are three sources. **Material nonlinearity:** the stress-strain law is not a straight line, as in plasticity, hyperelastic rubber or creep. **Geometric nonlinearity:** displacements or rotations are large enough that equilibrium must be written on the deformed shape, as in a thin plate stiffening as it deflects, cable structures, or buckling. **Contact (boundary) nonlinearity:** parts touch, slide or separate, so the supports and load paths change with the load. Nonlinear problems are solved incrementally with Newton iterations. **Plasticity** adds a deeper difficulty: the material remembers its history, so the stress depends on the whole load path, not just the current load. Two parts with the same final load can have completely different stresses. For a surrogate, that means the input must include the history, not just the final state.

**On this page**

- [Three sources of nonlinearity](#three-sources-of-nonlinearity)
- [How nonlinear problems are solved](#how-nonlinear-problems-are-solved)
- [Plasticity and load-history dependence](#plasticity-and-load-history-dependence)
- [Code: same final strain, different stress](#code-same-final-strain-different-stress)
- [Why path dependence complicates surrogates](#why-path-dependence-complicates-surrogates)
- [When to run a nonlinear analysis](#when-to-run-a-nonlinear-analysis)
- [In an interview](#in-an-interview)

## Three sources of nonlinearity

| Source | What is nonlinear | Examples |
| --- | --- | --- |
| **Material** | The stress-strain relation | Yielding of metals (plasticity), rubber seals and tyres (hyperelasticity), polymers and solder over time (viscoelasticity, creep), concrete cracking, damage |
| **Geometric** | The kinematics: equilibrium on the deformed shape; strain measures for large rotations | Thin plates and membranes stiffening as they deflect, cables, slender beams with large deflections, buckling and post-buckling, follower pressure loads that rotate with the surface |
| **Contact / boundary** | The boundary conditions themselves change | Bolted joints opening, gears, press fits, seals, snap fits, impact, friction and sliding |

**Material nonlinearity.** Below yield, steel is linear elastic. Above yield it flows plastically: the stress grows much more slowly with strain, and when unloaded the part keeps a permanent strain. Rubber is nonlinear elastic over large strains (it unloads along the same path but the curve is not straight). Many real parts combine several material behaviours.

**Geometric nonlinearity.** Linear analysis assumes small displacements and rotations, so equilibrium can be written on the original geometry and strains computed from ½(∇u + ∇uᵀ). When a thin plate deflects by more than about its thickness, in-plane membrane forces develop and make it much stiffer than linear theory predicts. A slender column under compression loses stiffness as it bows, until it buckles. Large rotations also break the small-strain formula: a rigid rotation would show spurious strain. Nonlinear analysis uses finite-strain measures and updates the geometry.

**Contact nonlinearity.** Contact is a constraint that switches on or off: a gap closes, a joint opens, a surface sticks or slides. The stiffness of the structure changes abruptly when contact status changes, which is often the hardest part of a nonlinear model to converge. See [When Newton does not converge](newton-raphson-and-nonlinear-solvers.md#when-newton-does-not-converge).

**Real examples usually combine them.** Sheet-metal forming has all three: large plastic strains, large deformation and tool contact. A snap fit has plastic or hyperelastic material, large deflection and contact. A bolted flange has contact and preload, and may have local yielding.

## How nonlinear problems are solved

All three turn the discrete equations into R(u) = F_int(u) − F_ext = 0, where the internal force F_int is no longer K·u with a constant K. The solution procedure:

1. **Apply the load in increments** (load or time steps).
2. In each increment, solve R(u) = 0 with **Newton-Raphson** iterations using the tangent stiffness.
3. **Update the state:** geometry, contact status, and for plasticity the internal variables (plastic strain, hardening) at every integration point.
4. **Cut back** the increment if Newton does not converge; use **arc-length** methods through limit points.

See [Newton-Raphson and nonlinear solvers](newton-raphson-and-nonlinear-solvers.md) for the details. Explicit dynamics (crash, drop tests, forming) avoids Newton iterations altogether by taking many tiny time steps. See [Time integration](time-integration-and-stability.md#which-method-where).

**Superposition no longer holds.** In a nonlinear analysis, the response to two loads together is not the sum of the responses to each, and doubling the load does not double the stress. Every load combination is a new analysis.

## Plasticity and load-history dependence

**The behaviour.** Load a steel bar beyond yield and unload it. On the way up, it follows the elastic line, yields, then hardens. On the way down, it unloads *elastically*, along a line parallel to the original elastic slope, and ends with a permanent plastic strain at zero load. Reload it, and it stays elastic until the stress reaches the *new*, higher yield point that hardening created.

```
stress
  │         ╱‾‾‾‾‾‾‾‾  hardening
  │        ╱  ↙ unload elastically (slope E)
  │       ╱  ╱
  │      ╱  ╱
  │     ╱  ╱
  │    ╱  ╱
  │   ╱  ╱
  │  ╱  ╱
  │ ╱  ╱
  └─────●──────────── strain
        permanent plastic strain at zero load
```

**Why the history matters.** At a given *total* strain, the stress depends on how much of that strain is plastic, and that depends on the path:

- A part strained to 0.4% directly is in tension near the yield stress.
- A part strained to 0.6% and then brought back to 0.4% has more plastic strain and is *compressed*, although its final total strain is the same.

The code below shows exactly this: +255 MPa versus −141 MPa at the same final strain.

**How the material model handles it.** Plasticity models carry **internal state variables** at every integration point: the plastic strain, an accumulated plastic strain that drives isotropic hardening, and often a back stress for kinematic hardening (which models the Bauschinger effect: after yielding in tension, the material yields earlier in compression). Each increment uses the state from the previous increment, through a *return-mapping* algorithm: compute a trial elastic stress, and if it is outside the yield surface, project it back onto the yield surface and update the plastic strain.

**Consequences engineers see:**

- **Residual stresses** remain after unloading (from welding, forming, shot peening, autofrettage of pressure vessels), and they change later strength and fatigue life.
- **Springback** in sheet-metal forming: the elastic part of the strain recovers when the tool is removed, so the final shape depends on the forming history.
- **Cyclic effects:** under repeated loading a structure may *shake down* to elastic behaviour, keep cycling plastically (low-cycle fatigue), or *ratchet* with strain growing every cycle.
- **Order of load cases matters:** applying thermal load then mechanical load can give a different result from the reverse.

## Code: same final strain, different stress

A 1D elastic-plastic material with linear isotropic hardening, driven by a strain history. The state (plastic strain and accumulated plastic strain) is updated at each step with a return mapping.

```python
import numpy as np

E, sigma_y0, H = 200_000.0, 250.0, 2_000.0          # MPa: Young's modulus, initial yield, hardening modulus

def stress_history(strain_path):
    """1D elastic-plastic material with linear isotropic hardening (return mapping).
    The state (plastic strain, accumulated plastic strain) carries the history."""
    eps_p, alpha, out = 0.0, 0.0, []
    for eps in strain_path:
        sigma_trial = E * (eps - eps_p)                  # elastic predictor
        f = abs(sigma_trial) - (sigma_y0 + H * alpha)    # yield function
        if f > 0:                                        # plastic corrector: return to the yield surface
            d_gamma = f / (E + H)
            eps_p += d_gamma * np.sign(sigma_trial)
            alpha += d_gamma
        out.append(E * (eps - eps_p))
    return np.array(out), eps_p

path_a = np.linspace(0, 0.004, 41)                                                    # straight to 0.4%
path_b = np.concatenate([np.linspace(0, 0.006, 61), np.linspace(0.006, 0.004, 21)])  # to 0.6%, back to 0.4%
final = {}
for name, path in [("A: 0 -> 0.4%", path_a), ("B: 0 -> 0.6% -> 0.4%", path_b)]:
    s, eps_p = stress_history(path)
    final[name[0]] = s[-1]
    print(f"path {name:22s} final strain {path[-1]:.4f}: stress {s[-1]:7.1f} MPa, plastic strain {eps_p:.5f}")
```

Output:

```
path A: 0 -> 0.4%           final strain 0.0040: stress   255.4 MPa, plastic strain 0.00272
path B: 0 -> 0.6% -> 0.4%   final strain 0.0040: stress  -140.6 MPa, plastic strain 0.00470
```

Same final strain, opposite signs of stress. A model that only sees "final strain = 0.4%" cannot know which answer is right.

## Why path dependence complicates surrogates

A surrogate normally learns a function: inputs → outputs. With plasticity, the final response is **not a function of the final load** alone. Ways to deal with it:

1. **Restrict the problem** to monotonic, proportional loading (one load case, increasing). Then the final load does determine the response, and an ordinary surrogate works, but only for that kind of loading. Document it as part of the validity range.
2. **Make the history an input.** Feed the load path (a time series of loads or displacements) to a sequence model: a recurrent network, a temporal convolution or a transformer. The data must then cover a variety of *paths*, not only final loads, which multiplies the data needed.
3. **Predict incrementally with a state.** Train a model that maps (current state, load increment) to (new state, response), and roll it out like a time-stepper. It needs internal variables, learned or physical, that play the role of plastic strain. The same error-accumulation issues as any [autoregressive rollout](autoregressive-rollouts.md) apply.
4. **Learn the material, not the structure.** Use ML for the constitutive law at each integration point (a learned, history-aware stress update, often a recurrent network or a model with built-in thermodynamic constraints) inside a standard FE solver, which still handles equilibrium, geometry and contact.

**Other difficulties:**

- **Discontinuities.** Contact changes and the onset of yield create kinks in the response; smooth models blur them.
- **Non-uniqueness and instability.** Buckling and snap-through can have several equilibrium states; the path decides which one occurs.
- **Missing data.** Nonlinear runs fail to converge more often, and failures cluster in exactly the interesting regions (large plastic strain, buckling). Dropping them silently biases the dataset. See [What makes training data physically meaningful](generating-simulation-training-data.md#what-makes-training-data-physically-meaningful-and-sufficient).
- **Crash** is the extreme case: plasticity, contact, large deformation and fracture together, in a transient. See [Crash-simulation surrogates](crash-simulation-surrogates.md).

## When to run a nonlinear analysis

- The linear analysis predicts stresses above yield in regions that matter (beyond tiny singular spots).
- Deflections are comparable to the thickness of thin parts, or rotations are large.
- Parts are joined by contact (bolted flanges, press fits, seals) and the load path depends on whether contact opens.
- The question is about ultimate strength, collapse load, post-buckling behaviour, permanent deformation, forming or springback.
- Rubber or other hyperelastic materials, or creep at high temperature, are involved.

## In an interview

**Thirty-second answer to "What makes an analysis nonlinear?"** "Three things: material, when the stress-strain law is not linear, like plasticity or rubber; geometry, when deformations or rotations are large enough that equilibrium must be written on the deformed shape, like thin plates stiffening or buckling; and contact, when parts touch or separate so the boundary conditions change with the load. All three make stiffness depend on the solution, so we solve incrementally with Newton iterations and superposition no longer holds."

**Thirty-second answer to "Why does plasticity make the answer depend on the load history?"** "Because unloading is elastic but loading past yield adds permanent plastic strain. At the same total strain, the stress depends on how much of it is plastic, which depends on the path: strain a bar to 0.4% directly and it is in tension near yield; take it to 0.6% and back to 0.4% and it is in compression. So the model needs internal state, and a surrogate needs the history as an input, or the problem restricted to monotonic loading."

**Likely follow-ups**

- *What is springback?* The elastic recovery after a formed part is released; its final shape depends on the forming history.
- *Why is contact hard to converge?* Contact status switches between open and closed, so stiffness changes abruptly between iterations.
- *How would you build a surrogate for a plastic forming process?* Include the process path as input or predict incrementally with a learned state, and cover varied paths in the data.

## References

- J. C. Simo and T. J. R. Hughes, *Computational Inelasticity*, Springer, 1998 (return mapping).
- N.-H. Kim, *Introduction to Nonlinear Finite Element Analysis*, Springer, 2015.
- P. Wriggers, *Nonlinear Finite Element Methods*, Springer, 2008.
- [High-Fidelity Industrial Crash Dynamics Prediction via Geometry-Aware Operator Learning (Akhare et al., 2026)](https://arxiv.org/abs/2605.27758)
