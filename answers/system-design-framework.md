# System Design Framework for AI-for-Simulation

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Use the same twelve steps for every design question, and spend the first few minutes on requirements: most weak answers jump straight to an architecture. Start from the engineering *decision*, not the model. Interviewers listen for whether you:
- ask what accuracy the decision actually needs;
- name a baseline;
- choose the model from the data and geometry you have;
- design the trust layer (uncertainty, OOD detection, solver fallback);
- close the loop with monitoring and new data.

**On this page**

- [The twelve steps](#the-twelve-steps)
- [Building blocks that recur](#building-blocks-that-recur)
- [Worked designs](#worked-designs)
- [How to run the 45 minutes](#how-to-run-the-45-minutes)
- [How to sound strong](#how-to-sound-strong)
- [In an interview](#in-an-interview)

## The twelve steps

| # | Step | Questions to ask or answer |
| --- | --- | --- |
| 1 | **Decision and user** | Who uses it? Screening 100 concepts, interactive design, or final sign-off? What does a wrong answer cost? |
| 2 | **Outputs and accuracy** | Which QoIs and fields? What accuracy, relative to the decision margin *and* to the solver's own uncertainty? Ranking or absolute values? |
| 3 | **Inputs and variability** | Parametric or free-form geometry? Ranges of boundary conditions and operating points? Is only CAD available at inference? |
| 4 | **Latency and scale** | Seconds or minutes? Number of queries per day? Batch or interactive? Hardware? |
| 5 | **Data** | Existing simulations? Budget for new ones? Fidelity (RANS vs LES)? Experiment plan, quality assurance, formats, licences and IP |
| 6 | **Baselines** | Coarse-mesh solver, nearest neighbour, GP on parameters, existing correlations |
| 7 | **Model** | Justify from the geometry representation and data size: GP or POD for small parametric data; GNN, point-cloud or transformer models for free-form shapes; query decoders for full resolution; multi-fidelity or pretrained models when data is scarce |
| 8 | **Training** | Normalization, losses (field + QoI), physics constraints, subsampling and partitioning for scale |
| 9 | **Evaluation** | Grouped splits; in-distribution vs OOD; field, QoI, ranking and physics checks; end-to-end speed; acceptance criteria agreed with the user |
| 10 | **Trust layer** | Ensembles or conformal intervals; OOD score; route to the solver above a threshold; verify final designs with the solver |
| 11 | **Deployment** | Service, integration into CAE tools, versioning, monitoring, retraining (active learning) |
| 12 | **Risks and roadmap** | Which regimes will fail? What is v0, and what is v1? |

## Building blocks that recur

Almost every design reuses the same components; know each well enough to sketch it in a minute.

| Block | What it does | Details |
| --- | --- | --- |
| Geometry ingestion | Validate CAD or STL, fix units and frame, sample points | [Mesh data processing](mesh-data-processing.md) |
| Data pipeline | DoE, meshing, solving, quality checks, storage, versioned datasets | [Simulation data pipeline](design-simulation-data-pipeline.md) |
| Baselines | Nearest neighbour, GP on parameters, coarse solver | [Classical surrogates](classical-surrogates.md) |
| Model choice | From data size and geometry representation | [Choosing an architecture](choosing-an-architecture.md) |
| Evaluation | Grouped and OOD splits, QoI and field metrics, ranking, slices | [Evaluation metrics](evaluation-metrics.md), [Data splits](data-splits-and-leakage.md) |
| Trust layer | Calibrated uncertainty, OOD score, solver fallback | [Uncertainty](uncertainty-quantification.md), [OOD](out-of-distribution-and-generalization.md) |
| Serving | Queue, GPU inference, caching, versions | [Surrogate serving system](design-surrogate-serving-system.md) |
| Monitoring | Drift detection, audit runs, retraining triggers | [Drift monitoring](design-drift-monitoring.md) |
| Closing the loop | New simulations where the model is weakest | [Active learning loop](design-active-learning-loop.md) |

## Worked designs

Each page applies the framework to one question from the guide:

- [Car aerodynamics surrogate](design-car-aerodynamics-surrogate.md): drag and surface pressure from an STL in seconds
- [Stress prediction platform](design-stress-prediction-platform.md): CAD in, stress with a confidence level out
- [Simulation data pipeline](design-simulation-data-pipeline.md): generating and storing 10,000 CFD runs
- [Active learning loop](design-active-learning-loop.md): choosing the next simulation
- [Bracket lightweighting loop](design-bracket-lightweighting-loop.md): optimization that never ships an unverified design
- [Validation strategy](design-validation-strategy.md): convincing a sceptical customer
- [Heat exchanger digital twin](design-heat-exchanger-digital-twin.md): sensors plus simulation in real time
- [Crash surrogate](crash-simulation-surrogates.md#system-design-a-crash-surrogate-for-a-vehicle-programme): a vehicle programme
- [Thermal surrogate](design-thermal-surrogate.md): battery pack or electronics enclosure
- [RANS–LES multi-fidelity surrogate](design-rans-les-multi-fidelity-surrogate.md): plenty of cheap data, a little expensive data
- [Fine-tuning pipeline](design-fine-tuning-pipeline.md): a pretrained model and a customer's 50 simulations
- [Inverse design tool](design-inverse-design-tool.md): target pressure in, geometry out
- [LLM structural check agent](design-llm-structural-check-agent.md): plain language in, verified FEA out
- [Surrogate serving system](design-surrogate-serving-system.md): 1,000 interactive users
- [Drift monitoring](design-drift-monitoring.md): detecting designs outside the training data

## How to run the 45 minutes

| Minutes | What to do |
| --- | --- |
| 0–8 | Requirements: decision, outputs, accuracy, inputs, latency, data, deployment constraints. Write them down where the interviewer can see them |
| 8–12 | Back-of-envelope numbers: data volume, compute, memory, queries per second |
| 12–30 | Architecture: data → model → evaluation → trust → serving, drawn as boxes; justify each choice against an alternative |
| 30–40 | Deep dive where the interviewer pushes (often the model, the trust layer or scaling) |
| 40–45 | Risks, what you would not trust it for, v0 and v1 roadmap |

## How to sound strong

- Put numbers on things: memory (see [Scaling](scaling-to-large-meshes.md)), latency, data budget.
- Name a baseline before the model.
- Say what you would *not* trust the system for, and how the system knows when it is outside its range.
- Offer a v0 that works in weeks, and a roadmap.
- Keep the solver in the loop: verification of final designs, fallback for flagged cases, new training data.

## In an interview

**Thirty-second answer to "How do you approach an ML system design question for simulation?"** "I start from the engineering decision and its tolerance for error, then pin down inputs, outputs, latency and the data I have or can generate. I do the rough numbers, name a baseline, and choose the model from the geometry representation and data size. Evaluation uses grouped and out-of-distribution splits with the quantities engineers care about. Then the trust layer, calibrated uncertainty, an OOD check and a solver fallback, and finally serving and monitoring that feed new simulations back into training."

**Likely follow-ups**

- *What would you build first?* The data pipeline and a baseline with an honest evaluation; the model comes after.
- *How do you know the system is working in production?* Audit runs with the solver on a random sample, plus drift monitors.

## References

- [Surrogate models](surrogate-models.md), [Evaluation metrics](evaluation-metrics.md), [Uncertainty quantification](uncertainty-quantification.md), [Production and deployment](production-and-deployment.md)
- C. Huyen, *Designing Machine Learning Systems*, O'Reilly, 2022.
- [AB-UPT](https://arxiv.org/abs/2502.09692), [DoMINO](https://arxiv.org/abs/2501.13350), [Transolver](https://arxiv.org/abs/2402.02366)
