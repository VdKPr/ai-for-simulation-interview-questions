# Design: RANS–LES Multi-Fidelity Surrogate

[← Questions](../README.md) · [All answers](README.md) · [System design framework](system-design-framework.md)

**Question.** Design a surrogate that combines plenty of RANS data with a little LES data.

**Short answer.**

- **The idea:** RANS is cheap but biased where turbulence modelling fails (separation, wakes, transition). LES is accurate but costs one to two orders of magnitude more per case. So learn the broad dependence on design parameters from many RANS runs, and learn the **correction** (the RANS-to-LES discrepancy) from a few LES runs placed where they are most informative.
- **Methods, in order of data needs:**
  - co-kriging / autoregressive GPs (y_LES ≈ ρ·y_RANS + δ) for scalar quantities;
  - non-linear multi-fidelity GPs or neural networks that take the RANS prediction as an input;
  - transfer learning: pretrain a field model on RANS, then fine-tune it on LES.
- **Placing the LES runs:** choose them by where the discrepancy is uncertain and where the decision is sensitive.
- **Validation:** held-out LES only. Report the uncertainty of the correction, which is large with few LES runs.

**On this page**

- [Requirements and clarifying questions](#requirements-and-clarifying-questions)
- [Why RANS and LES differ](#why-rans-and-les-differ)
- [Methods](#methods)
- [Allocating the LES budget](#allocating-the-les-budget)
- [Data consistency between fidelities](#data-consistency-between-fidelities)
- [Evaluation](#evaluation)
- [Architecture](#architecture)
- [In an interview](#in-an-interview)

## Requirements and clarifying questions

- **Outputs:** scalars (drag, lift, pressure loss) or fields? Scalars allow GP-based methods; fields need neural models.
- **Budget:** how many RANS and LES runs? A typical ratio is 10:1 to 100:1.
- **Where RANS is wrong:** everywhere a bit, or only in some regimes (separation at high angle of attack)? This decides how the correction is modelled.
- **Time-averaged or unsteady outputs?** LES gives fluctuations RANS cannot.
- **Can the LES runs be chosen**, or are they fixed (from an existing database)?

## Why RANS and LES differ

- RANS models all turbulence; its errors are largest in separated flows, strong curvature, wakes and transition ([Turbulence modelling](turbulence-modelling.md)).
- LES resolves large eddies; it needs fine meshes, small time steps and long averaging, so it costs much more.
- The **discrepancy is structured:** small in attached-flow regions of the design space, large near separation. A good correction model captures this dependence instead of a constant offset.
- **Correlation matters:** multi-fidelity methods help when RANS and LES are correlated across designs (the trends agree even if levels differ). Check this on the first LES runs ([When multi-fidelity fails](multi-fidelity-learning.md#when-multi-fidelity-fails)).

## Methods

| Method | Model | Data needs | Notes |
| --- | --- | --- | --- |
| Correction only | Fit δ(x) = y_LES − y_RANS with a GP; predict y_RANS(x) from a RANS surrogate | Few LES | Simple; assumes ρ = 1 |
| Co-kriging (Kennedy–O'Hagan) | y_LES(x) = ρ y_RANS(x) + δ(x), both GPs | Few LES (5–30), many RANS | Scalar outputs; uncertainty included ([Code: co-kriging](multi-fidelity-learning.md#code-co-kriging-with-4-expensive-and-11-cheap-runs)) |
| Non-linear autoregressive GP (NARGP) | y_LES = g(x, y_RANS(x)) | More LES | Captures non-linear relations between fidelities |
| Multi-fidelity neural networks | Low-fidelity network + correction network with linear and non-linear parts | Moderate LES | Fields and higher dimensions |
| Transfer learning | Pretrain field model on RANS; fine-tune (heads or all layers with small learning rate) on LES | Tens of LES | For field surrogates on complex geometry ([Foundation models](foundation-models-for-physics.md)) |
| Fidelity as an input | One model with a fidelity indicator | Both, mixed | Simple; shares representation |

## Allocating the LES budget

- **Initial LES:** spread across the design space (a subset of the RANS design), including regimes where RANS is suspected wrong.
- **Sequential:** choose further LES runs where the discrepancy's uncertainty is largest **and** the output matters for the decision; cost-aware acquisition can also decide between buying more RANS or another LES ([Allocating the simulation budget](multi-fidelity-learning.md#allocating-the-simulation-budget)).
- **Nested designs** (LES points are a subset of RANS points) make co-kriging simpler and the discrepancy directly observable.

## Data consistency between fidelities

- Same geometry, boundary conditions and reference values for both fidelities, so the discrepancy reflects modelling, not set-up.
- LES averaging time long enough: statistical noise in LES means is part of the "high-fidelity" data; estimate it and include it as noise in the model.
- Field outputs on a common representation (interpolate to the same surface points).
- Record solver settings per run ([Simulation data pipeline](design-simulation-data-pipeline.md)).

## Evaluation

- **Test only against held-out LES** (leave-one-out cross-validation when LES runs are few).
- Compare: RANS-only surrogate, LES-only model on the few LES runs, multi-fidelity model. The multi-fidelity model should beat both.
- Report the uncertainty of the correction; with ten LES runs it is substantial, and the customer must know.
- Check calibration of intervals on the LES test points; check behaviour in regions with no LES data (the model should fall back towards RANS plus mean correction, with wider intervals).

## Architecture

```
design space ─► many RANS runs ─► RANS surrogate ─────────────┐
            └─► few LES runs (chosen) ─► discrepancy model ───┼─► multi-fidelity prediction ± σ
                      ▲                                       │
                      └── next LES: max σ_discrepancy × decision sensitivity
```

## In an interview

**Thirty-second answer.** "RANS gets the trends cheaply but is biased where turbulence modelling fails, so I'd learn the broad dependence from many RANS runs and the correction from a few LES runs. For scalar outputs, co-kriging, LES ≈ ρ·RANS plus a GP discrepancy, works with 5–30 LES runs and gives uncertainty; for fields, pretrain on RANS and fine-tune on LES, or use a multi-fidelity network. I'd first check that RANS and LES are correlated across designs, place the LES runs where the discrepancy is uncertain and matters, keep set-ups identical across fidelities, and validate only on held-out LES."

**Likely follow-ups**

- *What if RANS and LES are not correlated in some regime?* Multi-fidelity cannot help there; spend LES there directly, or restrict the scope.
- *How do you include wind-tunnel data?* As a third fidelity, with its own noise and bias terms.

## References

- M. C. Kennedy and A. O'Hagan, "Predicting the output from a complex computer code when fast approximations are available", *Biometrika*, 2000.
- P. Perdikaris, M. Raissi, A. Damianou, N. D. Lawrence and G. E. Karniadakis, "Nonlinear information fusion algorithms for data-efficient multi-fidelity modelling", *Proc. R. Soc. A*, 2017.
- X. Meng and G. E. Karniadakis, "A composite neural network that learns from multi-fidelity data", *J. Comput. Phys.*, 2020 ([arXiv:1903.00104](https://arxiv.org/abs/1903.00104)).
- [Multi-fidelity learning](multi-fidelity-learning.md), [Turbulence modelling](turbulence-modelling.md)
