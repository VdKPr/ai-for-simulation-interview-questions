# Crash-Simulation Surrogates

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Crash simulation combines almost everything that makes surrogates hard: large deformation, rate-dependent plasticity, contact and self-contact, failure of welds and materials, all in a transient event of about 100 milliseconds that an explicit solver resolves in roughly 10⁵ tiny time steps on models with millions of elements. The response is path-dependent and can **bifurcate**: a small change in thickness can switch a crash rail from neat progressive folding to global bending, and even repeated runs of nearly identical models show numerical scatter. Data is expensive (each full-vehicle run takes hours on a cluster), outputs are high-dimensional time histories, and designs differ in mesh and parts. A practical crash surrogate therefore starts from a clear, narrow use case, for example ranking design variants in early concept phases, predicts the engineering quantities (intrusions, pulses, energy absorption) with uncertainty, is evaluated against the scatter of the solver itself, and leaves certification to the full simulation.

**On this page**

- [Why crash surrogates are hard](#why-crash-surrogates-are-hard)
- [What a crash surrogate should predict](#what-a-crash-surrogate-should-predict)
- [Modelling options](#modelling-options)
- [Temporal strategies: one-shot, time-conditional, autoregressive](#temporal-strategies-one-shot-time-conditional-autoregressive)
- [System design: a crash surrogate for a vehicle programme](#system-design-a-crash-surrogate-for-a-vehicle-programme)
- [In an interview](#in-an-interview)

## Why crash surrogates are hard

1. **Extreme nonlinearity.** Plasticity with strain-rate effects, large rotations, contact between and within parts (a folding tube contacts itself), friction, spot-weld and adhesive failure, and material fracture with element deletion. See [Nonlinear structural analysis](nonlinear-structural-analysis.md).
2. **Path dependence.** Plastic deformation depends on the history, so the final shape depends on everything that happened before. See [Plasticity and load-history dependence](nonlinear-structural-analysis.md#plasticity-and-load-history-dependence).
3. **Bifurcations and sensitivity.** Thin-walled structures in compression buckle, and which buckling and folding pattern develops can depend on tiny differences: thickness, a weld position, mesh, even numerical round-off on a different number of CPU cores. Crash engineers know this as numerical **scatter**. The "ground truth" is therefore noisy, and the map from design to outcome can jump.
4. **Long transients with tiny steps.** The explicit time step is set by the smallest element (about a microsecond for millimetre-scale steel elements), so a 100 ms event needs on the order of 10⁵ steps. Outputs are typically written every millisecond or so; the surrogate learns on those frames. See [The CFL condition](time-integration-and-stability.md#the-cfl-condition).
5. **Scale.** Full-vehicle models have millions of mostly shell elements and hundreds to thousands of parts; a run takes many hours on a cluster, so a programme produces hundreds to a few thousand runs, not millions.
6. **Changing models.** Design variants change part thicknesses and materials (easy), but also geometry, meshes, part sets and connections (hard). Node correspondence between variants is not guaranteed.
7. **Engineering outputs are specific and sensitive.** The quantities that decide a design (intrusion into the footwell, the deceleration pulse at the occupant, force levels in load paths) are local and transient, and errors in timing matter as much as errors in magnitude.

## What a crash surrogate should predict

| Output | Why engineers care | Notes |
| --- | --- | --- |
| Intrusions at measurement points | Occupant space; consumer and regulatory test ratings | Local displacement time histories |
| Deceleration pulse (acceleration vs time) at occupant locations | Drives restraint design and injury risk | Timing and peak both matter |
| Section forces in load paths | Shows how energy is managed | Force-displacement curves of rails and beams |
| Energy absorbed per part | Crash management design | Integral quantity, often easier to predict |
| Final deformed shape | Folding mode, failure locations | Full-field, very high-dimensional |
| Failure indicators | Weld or material failure | Classification-like |

**Evaluation should use these quantities**, with curve-correlation metrics for time histories (objective rating methods such as CORA or the ISO/TS 18571 metric score phase, magnitude and shape agreement), and should be compared with the **scatter floor**: run the solver a few times with tiny perturbations and measure how much the outputs vary. A surrogate cannot be meaningfully more accurate than the solver's own scatter.

## Modelling options

**1. Scalar or time-history surrogates on design parameters.** If the variants are parametric (gauges, material grades, a few morph parameters), predict key scalars or compressed time histories directly with Gaussian processes, gradient boosting or small networks, often on POD coefficients of the curves. Fast to build, works with tens to hundreds of runs, gives uncertainty. This is the baseline.

**2. Reduced-order field models.** Compress displacement histories with POD (or an autoencoder) and regress the coefficients. Works well while the deformation modes stay similar, breaks when the folding mode changes. See [Reduced-order models](reduced-order-models.md).

**3. Geometry-aware deep models.** Graph neural networks on the mesh (MeshGraphNets-style message passing) or point-cloud and transformer models that take the geometry and part properties (thickness, material) as inputs and predict deformation fields over time. Recent work applies **GeoTransolver**, a geometry-aware physics-attention transformer, to bumper-beam and full-vehicle crash datasets, predicting plastic deformation patterns and acceleration profiles at occupant locations. See [GNNs and MeshGraphNets](gnns-and-meshgraphnets.md) and [Physics transformer architectures](physics-transformer-architectures.md).

**Useful inputs regardless of the model:** per-node part ID, thickness, material parameters (yield, hardening, rate sensitivity), connection features (spot-weld locations), impact speed and barrier type, and geometric features (normals, distances).

## Temporal strategies: one-shot, time-conditional, autoregressive

| Strategy | How it works | Strengths | Weaknesses |
| --- | --- | --- | --- |
| **One-shot** | Predict all output frames at once from the initial state and design | No error accumulation; fast inference | Large output; must learn the whole trajectory at once |
| **Time-conditional** | Predict the state at any requested time t, with t as an input | Query any time; no accumulation | Less explicit temporal coherence |
| **Autoregressive** | Predict frame k+1 from frame k, and roll out | Natural for dynamics; can extend in time | Errors accumulate and can blow up over many steps ([Autoregressive rollouts](autoregressive-rollouts.md)) |

In the GeoTransolver crash study (Akhare et al., 2026), the authors compared one-shot, time-conditional and autoregressive rollout recipes and found that the **one-shot** approach gave state-of-the-art accuracy with significantly lower training overhead and inference latency. They also introduced a low-rank attention modification to reduce memory for the large models.

## System design: a crash surrogate for a vehicle programme

Use the [system design framework](system-design-framework.md). A worked answer:

**1. Clarify the use case.** Who uses it, when, and for which decision?
- *Assume:* crash engineers in the early concept phase, who want to rank 50 to 200 variants of the front-end structure (gauges, materials, rail geometry morphs) for one frontal load case, before committing to full runs.
- Outputs: intrusion at a handful of footwell and dash points, the deceleration pulse at the occupant location, energy absorbed by the front rails, and the final deformed shape for visual checks.
- Accuracy target: correct *ranking* of variants and errors comparable to the solver scatter on key scalars; latency of minutes rather than hours.
- Not in scope: certification, new load cases, new vehicle platforms (each would be a separate validation).

**2. Data.**
- *Sources:* the current programme's design-of-experiments runs (a few hundred) plus archived runs from earlier variants of the same platform.
- *Pipeline:* parse solver output (for example LS-DYNA d3plot files) into a standard format: node coordinates per output frame, part IDs, thickness, material, and the derived engineering outputs computed with the *same* scripts engineers use.
- *Consistency:* resample all time histories to a common time base; filter accelerations with the same channel filters used in testing; record solver version, mesh size, contact settings and termination status.
- *Quality:* flag runs that terminated early or have large energy errors (hourglass energy, added mass from mass scaling); investigate rather than delete, because failures can cluster in interesting designs.
- *Scatter study:* re-run a few designs with tiny perturbations to measure the noise floor.

**3. Baseline.** A GP or gradient-boosted model from the design parameters to each key scalar, and POD + GP for the pulse. Cheap, with uncertainty, and the benchmark any deep model must beat.

**4. Main model.** A geometry-aware field model (GNN or transformer) on the shell mesh, with part thickness and material as node features, predicting nodal displacement over time with a **one-shot** or time-conditional head, plus scalar heads for intrusions and energy. Train with losses weighted towards the critical regions and the key outputs.

**5. Uncertainty and validity.** A deep ensemble (or several seeds) for uncertainty; an out-of-distribution score from geometry embeddings and parameter ranges; physics checks on outputs (energy balance, no inter-penetration in the predicted shape). See [Out-of-distribution detection](out-of-distribution-and-generalization.md).

**6. Evaluation.**
- Held-out designs from the DoE *and* a held-out family of more different morphs.
- Errors on intrusions, pulse correlation scores, energy, and **ranking accuracy** (for example Spearman correlation of predicted versus true rankings).
- Compare every error with the scatter floor.
- Inspect cases where the folding mode changes: does the model predict the mode switch or blur two modes into an average?

**7. Deployment.** Integrate into the pre-processing environment engineers already use, so a variant can be evaluated from the same model set-up; return predictions with uncertainty and an OOD flag; send flagged and top-ranked variants to the full solver.

**8. Monitoring and retraining.** Every full run that engineers do anyway becomes a new labelled sample; track surrogate error on them; retrain when the design space shifts (new parts, new materials).

**9. Risks to state up front.** Bifurcations (the model may average two folding modes), label scatter, changes in model set-up between solver versions, and over-trust. The tool ranks and screens; the full simulation certifies.

## In an interview

**Thirty-second answer to "Why are crash surrogates especially hard?"** "Crash combines rate-dependent plasticity, contact, large deformation and failure in a 100 ms transient that takes about 10⁵ explicit steps on a model with millions of elements, so data is expensive. The response is path-dependent and can bifurcate: a small gauge change can switch the folding mode, and even the solver shows scatter between nearly identical runs. Outputs are local, transient time histories where timing matters. So I would narrow the use case, predict the engineering quantities with uncertainty, evaluate against the solver's scatter, and keep certification with the full simulation."

**Likely follow-ups**

- *One-shot or autoregressive for crash?* One-shot or time-conditional avoids error accumulation over many frames; a recent GeoTransolver crash study found one-shot most accurate and cheapest.
- *What is crash scatter and why does it matter?* Run-to-run variation from tiny perturbations, due to bifurcations; it sets a floor on meaningful surrogate accuracy.
- *How would you evaluate a predicted deceleration pulse?* With an objective curve-correlation metric (phase, magnitude, shape), plus errors on peak and timing.

## References

- [High-Fidelity Industrial Crash Dynamics Prediction via Geometry-Aware Operator Learning with Memory-Efficient Low-Rank Attention (Akhare et al., 2026)](https://arxiv.org/abs/2605.27758)
- [GeoTransolver: Learning Physics on Irregular Domains Using Multi-scale Geometry Aware Physics Attention Transformer (Adams et al., 2025)](https://arxiv.org/abs/2512.20399)
- [Learning Mesh-Based Simulation with Graph Networks (Pfaff et al.)](https://arxiv.org/abs/2010.03409)
- ISO/TS 18571, *Road vehicles: Objective rating metric for non-ambiguous signals*.
