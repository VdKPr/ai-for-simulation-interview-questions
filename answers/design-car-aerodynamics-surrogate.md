# Design: Car Aerodynamics Surrogate

[← Questions](../README.md) · [All answers](README.md) · [System design framework](system-design-framework.md)

**Question.** Design a system that predicts drag and surface pressure for a new car shape in seconds, from its STL.

**Short answer.**

- **Decision:** concept-phase screening and ranking of styling variants. A few percent on drag, and correct rankings, is enough; final designs still go to CFD and the wind tunnel.
- **Data:** a few hundred to a few thousand converged RANS or hybrid RANS–LES runs on morphed variants.
- **Model:** a point-based transformer (Transolver, AB-UPT or DoMINO style) or a surface GNN. It takes points sampled from the STL and predicts C_p and wall shear at any surface point. Drag is **integrated from the predicted fields**, with an optional direct drag head as a check.
- **Trust:** ensemble uncertainty on drag, an OOD score on the geometry embedding, and automatic routing of flagged designs to a CFD queue. Those runs become new training data.
- **Serving:** a GPU service returning VTP files for ParaView, plus a drag number with an interval, in seconds.

**On this page**

- [Requirements and clarifying questions](#requirements-and-clarifying-questions)
- [Data](#data)
- [Geometry processing](#geometry-processing)
- [Model](#model)
- [Training](#training)
- [Evaluation and acceptance](#evaluation-and-acceptance)
- [Trust layer](#trust-layer)
- [Serving and integration](#serving-and-integration)
- [Architecture](#architecture)
- [Risks and roadmap](#risks-and-roadmap)
- [In an interview](#in-an-interview)

## Requirements and clarifying questions

| Question | Typical answer | Why it matters |
| --- | --- | --- |
| Who uses it, for what decision? | Aero engineers and stylists, comparing variants in the concept phase | Sets accuracy: ranking and deltas matter more than absolute drag |
| Which outputs? | C_d (and C_l front/rear), surface C_p and C_f; maybe wake or volume fields later | Fields make results explainable; volume fields cost much more |
| Accuracy? | Drag within about 2–3% (roughly 5–10 drag counts) of CFD; rank correlation > 0.9 | Compare with CFD's own mesh and turbulence-model variation |
| Operating conditions? | Fixed speed, zero yaw at first; ride height and yaw later | Each extra condition multiplies data needs |
| Geometry variation? | Morphs of one or a few platforms, or free-form styling? | Free-form needs a non-parametric geometry model |
| Latency and volume? | Seconds per query; tens to hundreds of queries a day | One GPU is enough at first |
| Deployment? | Customer's cloud or on-premises | CAD is sensitive IP |

## Data

- **Source:** the customer's archived CFD where available, plus a new campaign: morph a few baseline cars (front, roof, rear, diffuser parameters) and sample with a space-filling design ([Design of experiments](design-of-experiments-and-active-learning.md)).
- **Fidelity:** steady RANS is cheap but misses unsteady wake effects; hybrid RANS–LES (as in DrivAerML) is better but costs far more per run. A common compromise is many RANS runs plus a few high-fidelity runs ([Multi-fidelity learning](multi-fidelity-learning.md)).
- **Quality checks:** convergence of the residuals and the drag history, y+ in range, mesh-convergence spot checks, no failed or diverged runs in the training set.
- **Public analogues to start with:** [DrivAerML](https://arxiv.org/abs/2408.11969), [DrivAerNet++](https://arxiv.org/abs/2406.09624) (non-commercial) and [SHIFT-SUV](https://huggingface.co/datasets/luminary-shift/SUV) (non-commercial). Check licences before commercial use.
- **Storage:** surface fields as VTP or Zarr per design, with metadata (solver version, settings, convergence) in a table. See [Simulation data pipeline](design-simulation-data-pipeline.md).

## Geometry processing

1. **Validate** the STL: watertight, consistent outward normals (positive enclosed volume), no non-manifold edges ([Mesh data processing](mesh-data-processing.md)).
2. **Normalize the frame and units:** x along the car, z up, origin at the front axle on the ground; metres. Wrong units are the most common silent error.
3. **Sample** surface points with normals, area-weighted (and denser at sharp features), for the encoder; keep the full surface mesh for query points.
4. **Optional global inputs:** frontal area, wheelbase, ride height; these also help the OOD check.

## Model

| Option | Strengths | Weaknesses |
| --- | --- | --- |
| GP or POD on morph parameters | Strong baseline, uncertainty for free | Needs a shared parametrization; fails on free-form shapes |
| Surface GNN (MeshGraphNets-style) | Works on the mesh directly; local accuracy | Many layers for long-range effects; mesh-dependent |
| Point-based transformer (Transolver, AB-UPT, DoMINO, GeoTransolver) | Mesh-independent; global context; query anywhere; scales to millions of points | Needs more data; heavier to train |

**Recommendation:**

- A **geometry encoder** over sampled surface points produces a latent set or set of tokens.
- A **query decoder** predicts C_p and C_f (three components) at any surface point.
- **Drag is computed by integrating** the predicted fields over the actual surface mesh, which keeps the force consistent with the fields ([Drag from pressure and shear](mesh-data-processing.md#drag-force-from-pressure-and-wall-shear)).
- A small **direct drag head** acts as a cross-check: if the two disagree strongly, flag the prediction.

See [Physics transformer architectures](physics-transformer-architectures.md).

## Training

- **Targets:** non-dimensional C_p = (p − p∞)/q∞ and C_f = τ_w/q∞, normalized per channel with training-set statistics ([Normalization](nondimensionalization-and-normalization.md)).
- **Loss:** area-weighted relative L2 on fields plus a drag loss term; weight the drag term so field accuracy does not trade it away.
- **Scale:** random subsets of query points per step (tens of thousands), full resolution at inference; mixed precision with care on the integration ([Scaling](scaling-to-large-meshes.md)).
- **Augmentation:** left–right mirror symmetry, if the data are symmetric about the centre plane.

## Evaluation and acceptance

- **Splits:** grouped by baseline design or morph family, plus an **OOD split** (a held-out body style) to see what happens outside the family ([Data splits](data-splits-and-leakage.md)).
- **Metrics:** drag absolute error in counts and in percent; Spearman rank correlation of drag across the test set; area-weighted field relative L2; pressure-drag vs friction-drag split; error by region (front, underbody, wake-side rear) ([Evaluation metrics](evaluation-metrics.md)).
- **Baselines:** nearest training design, a GP on morph parameters, a coarse-mesh CFD run of matched cost.
- **End-to-end speed:** STL to result, including geometry processing, compared with the full CFD turnaround.
- **Acceptance**, agreed with the aero team: for example, 90% of in-distribution designs within 2% on drag and a rank correlation above 0.9.

## Trust layer

- **Uncertainty:** a deep ensemble of 3–5 models gives a drag interval; calibrate it with conformal prediction on a held-out set ([Conformal prediction](conformal-prediction.md)).
- **OOD score:** distance from the new design's geometry embedding to its nearest training designs, plus checks on global features (frontal area, length) against training ranges ([OOD detection](out-of-distribution-and-generalization.md#detecting-an-out-of-distribution-geometry)).
- **Fallback:** above a threshold on uncertainty or OOD score, the result is marked "low confidence" and a CFD job is queued. Results of those runs feed the next retraining.
- **Physics checks:** symmetric designs give symmetric predictions; integrated drag matches the direct head.

## Serving and integration

- **API:** upload STL → job ID → result with C_d ± interval, OOD flag, fields as VTP, plus a report comparing with the closest training designs.
- **Hardware:** one GPU handles queries in seconds; batch nightly sweeps.
- **Integration:** a plug-in or script for the customer's CAD or pre-processor; results in ParaView or their post-processor ([Integrating into CAE](production-and-deployment.md#integrating-into-an-engineers-cae-workflow)).
- **Versioning:** model, data snapshot and geometry pipeline versions stored with every prediction.

## Architecture

```
STL ─► validate, units, frame ─► sample points ─► geometry encoder ─► query decoder ─► Cp, Cf
                                         │                                                  │
                                         ├─► OOD score ──┐                     integrate over surface
                                         │               ▼                                  ▼
                                         └─► ensemble ─► uncertainty ─► gate ─► Cd ± interval, VTP report
                                                                          │
                                                       low confidence ───►  CFD queue ─► new training data
```

## Risks and roadmap

- **Physics risks:** rear-end separation and wake sensitivity (small shape changes, large drag changes); rotating wheels and underbody detail; yaw conditions absent from training.
- **Data risks:** CFD set-up differences between old and new runs (solver version, turbulence model) mixed in one dataset.
- **v0 (weeks):** GP on morph parameters plus a surface model on existing data; evaluation report.
- **v1:** the transformer with uncertainty and OOD gating; CAD plug-in.
- **v2:** yaw and ride height as inputs; volume fields; multi-fidelity with LES or tunnel data; active learning ([Active learning loop](design-active-learning-loop.md)).

## In an interview

**Thirty-second answer.** "I'd first confirm it's for concept-phase ranking, where a couple of percent on drag is enough and CFD still signs off. The data is a few hundred to a few thousand converged RANS runs on morphed variants, quality-checked. After validating the STL and fixing units and frame, a point-based transformer predicts surface pressure and shear at any point, and drag comes from integrating those fields. I'd evaluate on held-out design families with drag error and rank correlation against a GP and a nearest-neighbour baseline. Each prediction gets an ensemble interval and an OOD score, and flagged designs go to CFD and become training data."

**Likely follow-ups**

- *Why integrate drag instead of predicting it directly?* Consistency with the fields, interpretability, and more supervision per design; keep a direct head as a cross-check.
- *How would you handle yaw?* Add it as an input, generate data at several yaw angles, and evaluate per angle.
- *What is the memory cost?* See [GPU memory for a GNN](scaling-to-large-meshes.md#gpu-memory-for-a-gnn-on-a-1-million-node-mesh).

## References

- [DrivAerML](https://arxiv.org/abs/2408.11969), [DrivAerNet++](https://arxiv.org/abs/2406.09624), [SHIFT-SUV](https://huggingface.co/datasets/luminary-shift/SUV)
- [AB-UPT](https://arxiv.org/abs/2502.09692), [Transolver](https://arxiv.org/abs/2402.02366), [DoMINO](https://arxiv.org/abs/2501.13350)
- [Aerodynamic forces and coefficients](aerodynamic-forces-and-coefficients.md), [Physics transformer architectures](physics-transformer-architectures.md)
