# Design: Stress Prediction Platform

[← Questions](../README.md) · [All answers](README.md) · [System design framework](system-design-framework.md)

**Question.** Design a platform where engineers upload CAD and get stress predictions with a confidence level.

**Short answer.**

- **Scope:** narrow it first. Which part families (brackets, housings), which load cases, which materials, linear elastic or not, and what the stress prediction is for (early design screening, not certification).
- **Pipeline:** CAD (STEP) is meshed automatically and checked. Load cases are described in a structured form. A mesh-based model (GNN or point/mesh transformer) predicts displacement and stress fields; von Mises and peak values are computed from them.
- **Confidence:** comes from three sources: an ensemble or conformal interval, an OOD score for the geometry *and* the load case, and checks such as equilibrium and symmetry.
- **The hard part:** peak stress at fillets and holes. It is local, mesh-sensitive and sometimes singular. So the platform reports peaks with wider intervals, and **routes any design whose margin is small to FEA**.

**On this page**

- [Requirements and clarifying questions](#requirements-and-clarifying-questions)
- [Inputs: CAD, loads and materials](#inputs-cad-loads-and-materials)
- [Data](#data)
- [Model](#model)
- [Confidence level](#confidence-level)
- [Evaluation and acceptance](#evaluation-and-acceptance)
- [Platform architecture](#platform-architecture)
- [Risks and roadmap](#risks-and-roadmap)
- [In an interview](#in-an-interview)

## Requirements and clarifying questions

| Question | Why it matters |
| --- | --- |
| Which part families and how different are they? | A model per family is far more reliable than one model for "any CAD" |
| Linear static only, or contact, plasticity, large deformation? | Nonlinear behaviour needs different data and models ([Nonlinear analysis](nonlinear-structural-analysis.md)) |
| Which outputs: max von Mises, safety factor, displacement field? | Peak values are much harder than averages |
| What decision, and what margin? | If parts are designed with a safety factor of 2, a 10% error may be acceptable for screening |
| Who uploads, how often, how fast must results be? | Interactive (seconds) vs batch |
| Security and IP | CAD often cannot leave the company: on-premises deployment |

## Inputs: CAD, loads and materials

- **Geometry:** STEP is preferred (exact geometry); STL is acceptable for surface-based models. Automated meshing with quality checks ([Meshing and mesh quality](meshing-and-mesh-quality.md)); reject or repair invalid CAD.
- **Load cases:** the hardest input to standardize. Use a structured specification: named faces or regions (bolt holes, mounting faces) with fixed supports, forces, pressures. Free-form load descriptions do not generalize. For a part family, define a small set of standard load cases.
- **Material:** linear elastic properties (E, ν) as inputs, or non-dimensionalize: in linear elasticity, stress scales linearly with load and is independent of E for force-controlled problems with a fixed ν. Use these scaling laws to reduce what the model must learn ([Linear elasticity](stress-strain-and-linear-elasticity.md)).

## Data

- **Generate with FEA:** parametric CAD variants of each part family plus load-case variations, solved with a scripted FE solver (open-source or the customer's), meshed consistently.
- **Mesh convergence:** check peak stresses on a sample with refined meshes; singularities (sharp re-entrant corners, point loads) do not converge, and the training labels there are meaningless ([Stress concentrations and singularities](stress-concentrations-and-singularities.md)).
- **Public analogues:** [SimJEB](https://simjeb.github.io/) and [DeepJEB](https://arxiv.org/abs/2406.09047) (jet-engine brackets).
- **Size:** hundreds to a few thousand cases per family is typical. Fewer if the variation is parametric (a GP or POD then works well).

## Model

- **Baselines:** GP on CAD parameters (where they exist); nearest training design; a coarse FE mesh (a few seconds to minutes of real FEA is a strong baseline).
- **Main model:** a GNN on the FE mesh (MeshGraphNets-style, multiscale for long-range load paths) or a point/mesh transformer. Inputs per node: position, normal, boundary-condition flags (fixed, loaded, free), load vectors; global inputs: material and load magnitude ([GNNs and MeshGraphNets](gnns-and-meshgraphnets.md), [Multiscale GNNs](multiscale-gnns.md)).
- **Outputs:** displacement (3 components) and stress tensor (6 components) per node; von Mises computed from the tensor, not predicted directly, so it is consistent ([Von Mises stress](von-mises-stress-and-yield.md)).
- **Peak stress:** a dedicated head or a weighted loss near high-stress regions, because a field loss averaged over the mesh under-weights the peak.
- **Linearity trick:** for linear problems, predict the response to unit loads and superpose, so load magnitude never has to be extrapolated.

## Confidence level

The "confidence level" must be **calibrated**, otherwise it is decoration:

- **Ensemble or conformal intervals** on peak von Mises, calibrated on held-out designs from the same family ([Conformal prediction](conformal-prediction.md)).
- **OOD checks** on geometry (embedding distance, size and feature ranges) and on load cases (types, magnitudes, locations unseen in training).
- **Physics checks:** global equilibrium (reaction forces balance applied loads); symmetry under symmetric loading; displacements zero at fixed supports.
- **Output to the user:** a traffic light and a number: "max von Mises 182 MPa, 90% interval 160–205 MPa, within training range"; **red** when OOD, when the interval crosses the allowable stress, or when the margin is small. Red means "run FEA", and the platform can do it automatically.

## Evaluation and acceptance

- Grouped splits by part family variant; a held-out family for OOD testing.
- Metrics: field relative L2 on displacement and stress; peak von Mises error and **location** error (distance between predicted and true hot spots); safety-factor classification accuracy (pass/fail against the allowable), with false "pass" as the critical error ([Evaluation metrics](evaluation-metrics.md)).
- Coverage of the stated intervals on held-out data (does the 90% interval contain the truth about 90% of the time?).
- Acceptance agreed with engineers; for example, no false "pass" on the test set at the chosen threshold.

## Platform architecture

```
Upload STEP ─► validate & repair ─► auto-mesh ─► quality check ─┐
Load-case form (named faces, forces, supports, material) ───────┤
                                                                ▼
                                        model ensemble ─► displacement, stress tensor ─► von Mises, peaks
                                                                │
                                OOD (geometry + loads) ─► gate ─┴─► physics checks
                                                                │
                      green: report (VTU + PDF) ◄──────────────┤
                      red: queue FEA job ─► result to user + to training data
```

**Components:** a web front end; a job queue; CPU workers for CAD and meshing; GPU workers for inference; an FE solver pool for fallbacks; a database of designs, predictions, versions and outcomes; monitoring ([Production and deployment](production-and-deployment.md)).

## Risks and roadmap

- **Biggest risk:** a confident wrong "pass" on a peak stress. Mitigation: conservative gating, FEA on anything near the limit, and logging every case for audit.
- **Other risks:** CAD that fails to mesh (a significant fraction of real uploads); inconsistent load-case definitions; singularities producing meaningless peak labels.
- **v0:** one part family, standard load cases, GP and GNN, evaluation report.
- **v1:** platform with gating and FEA fallback; on-premises deployment.
- **v2:** more families, fine-tuning a pretrained model per new family ([Fine-tuning pipeline](design-fine-tuning-pipeline.md)), active learning from fallback runs.

## In an interview

**Thirty-second answer.** "I'd narrow it to specific part families and standard load cases, linear static to start, for screening rather than sign-off. CAD is meshed automatically, load cases are defined on named faces, and FEA generates the training data, with convergence checks so the peak-stress labels mean something. A GNN or mesh transformer predicts displacement and the stress tensor, and von Mises comes from that. The confidence level is a calibrated interval plus OOD checks on both geometry and loads, and equilibrium checks. Anything uncertain or close to the allowable goes to FEA automatically."

**Likely follow-ups**

- *How do you handle a sharp corner with a singular stress?* Report a fillet radius requirement or a stress at a distance from the corner, and exclude singular points from the training loss ([Singularities](stress-concentrations-and-singularities.md)).
- *What if the user's load case is new?* It is OOD: flag it and run FEA; add it to the data if it recurs.

## References

- [SimJEB](https://simjeb.github.io/), [DeepJEB](https://arxiv.org/abs/2406.09047)
- [MeshGraphNets](https://arxiv.org/abs/2010.03409)
- [Stress concentrations and singularities](stress-concentrations-and-singularities.md), [Uncertainty quantification](uncertainty-quantification.md)
