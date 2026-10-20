# Design: Thermal Surrogate

[← Questions](../README.md) · [All answers](README.md) · [System design framework](system-design-framework.md)

**Question.** Design a thermal surrogate for a battery pack or an electronics enclosure.

**Short answer.**

- **Decide the use first:** layout exploration (where to put components, fans, cold plates) or operating-envelope checks (maximum cell or junction temperature across drive cycles or workloads).
- **Inputs:** geometry or layout parameters, heat loads per component, coolant flow and inlet temperature, ambient conditions, and for transient cases the load history.
- **Outputs:** the temperature field and, above all, **maximum temperatures and gradients** at critical components, because those decide whether the design passes.
- **Training data:** conjugate heat transfer (CHT) simulations.
- **Model:** pick it by the problem.
  - Parametric layouts with steady loads → a GP or POD with interpolation.
  - Free-form layouts → a mesh or point model with component features.
  - Transient duty cycles → a reduced-order or neural state-space model stepped in time, or a linear-system approach that exploits superposition when properties are temperature-independent.
- **Trust:** intervals on peak temperatures, input-range checks, and CHT verification of any design near the limit.

**On this page**

- [Requirements and clarifying questions](#requirements-and-clarifying-questions)
- [Physics that shapes the design](#physics-that-shapes-the-design)
- [Data](#data)
- [Model options](#model-options)
- [Transient duty cycles](#transient-duty-cycles)
- [Evaluation and trust](#evaluation-and-trust)
- [Architecture](#architecture)
- [Risks and roadmap](#risks-and-roadmap)
- [In an interview](#in-an-interview)

## Requirements and clarifying questions

- **System:** a battery pack (cells, busbars, cold plate, coolant loop) or an electronics enclosure (boards, heat sinks, fans, vents)?
- **Steady or transient:** a steady worst case, or temperature histories over drive cycles, fast charging, workload profiles?
- **Critical outputs:** maximum cell temperature and cell-to-cell spread (battery ageing and safety); junction temperatures against limits (electronics).
- **Variability:** layout, material choices, coolant flow, ambient, heat loads.
- **Latency:** interactive design (seconds) or system simulation coupling (milliseconds per step, many steps).
- **Accuracy:** typically within 1–2 °C of CHT on peak temperatures for design decisions.

## Physics that shapes the design

- **Conduction** in solids (cells, boards, housings), often anisotropic (cells conduct much better in-plane than through the stack); **convection** to coolant or air; **radiation** in sealed enclosures ([Conduction, convection and radiation](heat-transfer-basics.md#conduction-convection-and-radiation)).
- **Conjugate heat transfer** couples solid conduction to fluid flow ([Conjugate heat transfer](heat-transfer-basics.md#conjugate-heat-transfer)).
- **Linearity:** with constant properties and fixed flow, temperature rise is **linear in heat loads**. The response to N heat sources is a superposition of N unit responses. This makes surrogates for varying loads nearly exact and cheap: precompute the influence matrix.
- **Non-linearities** break it: temperature-dependent properties, natural convection (flow depends on temperature), radiation (T⁴), changing fan speeds.
- **Battery specifics:** heat generation depends on current, state of charge and temperature (electro-thermal coupling); thermal runaway is a different, highly nonlinear problem outside a design surrogate's scope.

## Data

- CHT simulations over the design space; for electronics, compact or detailed component models; for batteries, cell models with anisotropic conductivity.
- **Exploit linearity in data generation:** for fixed flow, simulate unit heat loads per component (N runs) instead of sampling load combinations.
- Mesh convergence of peak temperatures; consistent material data ([Simulation data pipeline](design-simulation-data-pipeline.md)).
- Measurements (thermocouples on prototypes) to calibrate contact resistances and correlations.

## Model options

| Situation | Model | Why |
| --- | --- | --- |
| Fixed layout, varying heat loads, fixed flow | Influence-coefficient matrix (superposition) | Exact for linear physics, microseconds |
| Few layout or flow parameters | GP or POD + interpolation | Small data, uncertainty ([Gaussian processes](gaussian-processes.md), [Reduced-order models](reduced-order-models.md)) |
| Free-form layouts | GNN or point transformer with component features (power, material) | Geometry generalization ([GNNs](gnns-and-meshgraphnets.md)) |
| Flow varies with temperature (natural convection) | Neural surrogate trained on full CHT | Nonlinear coupling |

Inputs should include physically meaningful features: component power, thermal resistances, flow rate, ambient temperature. Predict **temperature rise above coolant or ambient** (non-dimensionalized), not absolute temperature ([Non-dimensionalization](nondimensionalization-and-normalization.md)).

## Transient duty cycles

- **Linear systems:** the thermal response is a convolution of the load history with step (or impulse) responses for each source. Fit them from CHT step responses (or as a Foster/Cauer thermal network); evaluating a full drive cycle then takes milliseconds.
- **Reduced-order state-space models:** POD-Galerkin or system identification give a small linear system dx/dt = Ax + Bu that integrates quickly ([Reduced-order models](reduced-order-models.md)).
- **Nonlinear cases:** a neural state-space or autoregressive model stepped in time, trained with rollout-aware methods to avoid drift ([Autoregressive rollouts](autoregressive-rollouts.md)), or a [Neural ODE](neural-odes.md).
- Always check long rollouts against CHT on full drive cycles, not single steps.

## Evaluation and trust

- **Metrics:** peak temperature error (°C) at each critical component; spread (max − min cell temperature); time above limit for transients; field relative L2 ([Evaluation metrics](evaluation-metrics.md)).
- **Splits:** held-out layouts and held-out duty cycles (including a harsher one than seen in training).
- **Trust:** intervals on peaks (ensembles or conformal); input-range checks (power, flow, ambient); designs within a margin of the limit go to CHT.
- **Physics checks:** energy balance (heat in = heat removed at steady state); monotonicity (more power → higher temperature).

## Architecture

```
layout / parameters + heat loads + flow + ambient (+ load history)
        │
        ├─► range checks ─────────────────────────────┐
        ▼                                              ▼
 surrogate (superposition / GP / GNN / state-space) ─► peak T ± interval, spread, field ─► margin check
                                                                                           │
                                         near limit or out of range ─► CHT verification ◄──┘
```

## Risks and roadmap

- **Contact resistances and material data** often dominate the error, more than the model; calibrate with tests.
- **Regime changes:** natural convection onset, fan failure, coolant boiling: separate cases, not extrapolation.
- **v0:** superposition or GP on the parametric case; peak-temperature validation.
- **v1:** transient duty cycles with a state-space or convolution model.
- **v2:** free-form layouts with a mesh model; coupling to an electrical model for batteries.

## In an interview

**Thirty-second answer.** "I'd first establish whether it's steady layout exploration or transient duty cycles, and that the decision hinges on peak temperatures and spreads. Training data comes from conjugate heat transfer simulations. The key physics insight is linearity: with constant properties and fixed flow, temperature rise is a superposition of unit heat-load responses, so for varying loads the surrogate can be nearly exact and very cheap, and transients are a convolution with step responses. For varying layouts I'd use a GP or a mesh model. Every prediction gets an interval on peaks, and anything near the limit is checked with CHT."

**Likely follow-ups**

- *What breaks superposition?* Temperature-dependent properties, natural convection, radiation and changing fan speeds.
- *How would you couple it to a battery electrical model?* Exchange heat generation and temperature each time step; the surrogate must be fast and stable over long rollouts.

## References

- F. P. Incropera et al., *Fundamentals of Heat and Mass Transfer*, Wiley.
- [Heat transfer basics](heat-transfer-basics.md), [Multiphysics coupling](multiphysics-coupling.md), [Thermal stress and warpage](thermal-stress-and-warpage.md)
- [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) (data-center and electronics thermal examples)
