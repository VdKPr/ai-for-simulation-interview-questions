# Design: Fine-Tuning Pipeline

[← Questions](../README.md) · [All answers](README.md) · [System design framework](system-design-framework.md)

**Question.** Design a pipeline that fine-tunes a pretrained physics model on a customer's 50 simulations.

**Short answer.**

- **Ingest and harmonize.** Convert the customer's 50 simulations into the pretrained model's input representation: same units, coordinate frame, non-dimensionalization and sampling. Run quality checks.
- **Check fit to the pretraining distribution.** Embedding distance and the zero-shot error on a few cases show how far the customer's problem sits from what the model saw.
- **Choose the adaptation from that distance and the data size.**
  - Zero-shot plus a calibration if the problem is close.
  - Head-only or adapter (LoRA-style) fine-tuning for moderate shift.
  - Full fine-tuning with a small learning rate, early stopping and regularization towards the pretrained weights for larger shift.
- **With 50 cases, the evaluation design is most of the work:** grouped k-fold cross-validation, baselines trained on the same 50 (a GP and a small model from scratch), and conformal calibration of the intervals.
- **Package it** as a reproducible job with versioned data and model, an evaluation report, and the customer's data kept isolated.

**On this page**

- [Requirements and clarifying questions](#requirements-and-clarifying-questions)
- [Pipeline stages](#pipeline-stages)
- [Choosing how much to fine-tune](#choosing-how-much-to-fine-tune)
- [Evaluation with 50 simulations](#evaluation-with-50-simulations)
- [Isolation, IP and reproducibility](#isolation-ip-and-reproducibility)
- [Architecture](#architecture)
- [Risks and roadmap](#risks-and-roadmap)
- [In an interview](#in-an-interview)

## Requirements and clarifying questions

- **What was the model pretrained on?** Physics (incompressible external aerodynamics?), geometry types, Reynolds range, outputs.
- **What are the customer's 50 cases?** Same physics and outputs, or a new quantity (a new output head is needed)? Same regime?
- **What decision, what accuracy?** As always.
- **Can more cases be run?** If yes, active learning can follow ([Active learning loop](design-active-learning-loop.md)).
- **Where must data stay?** Customer cloud, on-premises, or the vendor platform.
- **Repeat use:** a one-off engagement or a product feature run for many customers? The latter needs full automation.

## Pipeline stages

1. **Ingest:** read the customer's formats (VTU, CGNS, EnSight, CSV QoIs); map field names; record solver and settings ([File formats](simulation-file-formats.md)).
2. **Harmonize:** units (mm vs m), coordinate frame and orientation, reference values (p∞, q∞, reference area), non-dimensionalization identical to pretraining ([Normalization](nondimensionalization-and-normalization.md)); sample points as in pretraining.
3. **Quality checks:** convergence, outliers, missing fields, duplicated designs.
4. **Distribution check:** embed the customer's geometries with the pretrained encoder; distance to the pretraining set; zero-shot error on all 50 (no training needed). This is the most informative single diagnostic.
5. **Adapt** (below), using cross-validation folds.
6. **Calibrate** uncertainty (ensembles over folds plus conformal calibration).
7. **Report:** cross-validated accuracy against baselines, calibration, limits.
8. **Deploy** the adapted model with its OOD threshold set from the customer's data.

## Choosing how much to fine-tune

| Situation (from step 4) | Adaptation | Why |
| --- | --- | --- |
| Close to pretraining, zero-shot error already acceptable | Zero-shot + output calibration (scale/offset per QoI) | Fewest parameters, least overfitting |
| Moderate shift | Train the decoder head, or adapters / LoRA on attention layers; freeze the encoder | Tens of thousands of parameters fit 50 cases |
| Larger shift, same physics | Full fine-tuning, learning rate ~10× smaller than pretraining, early stopping on CV, L2-SP regularization towards pretrained weights | Avoids destroying pretrained features |
| Different physics or output | New head on the frozen encoder; compare with training from scratch | Pretraining may not help at all |

The demo in [Foundation models for physics](foundation-models-for-physics.md) shows each case: zero-shot is best inside the pretraining range, fine-tuning is best outside it, and **mismatched pretraining can be worse than training from scratch**. That is why the comparison with from-scratch baselines is part of the pipeline, not optional.

Hyperparameters (learning rate, number of steps, which layers) are chosen on cross-validation folds, never on the final test cases.

## Evaluation with 50 simulations

- **Grouped k-fold cross-validation** (5 folds of 10, grouped by design family so near-duplicates stay together); report mean and spread across folds ([Data splits](data-splits-and-leakage.md)).
- If the customer can spare them, **a separate blind test set** of 5–10 cases evaluated once at the end.
- **Baselines on the same 50:** GP on parameters (if any), a small model from scratch, nearest neighbour, zero-shot pretrained model. The fine-tuned model must beat all of them to justify itself ([Learning from small simulation datasets](learning-from-small-simulation-datasets.md)).
- **Learning curve:** accuracy with 10, 20, 30, 40 cases tells the customer what more data would buy.
- **Uncertainty:** an ensemble of the fold models; conformal calibration on out-of-fold predictions ([Conformal prediction](conformal-prediction.md)).

## Isolation, IP and reproducibility

- Customer data never mixes with other customers' data or goes back into pretraining without a contract allowing it.
- Training runs inside the customer's tenancy or on-premises when required ([When customers will not upload their CAD](production-and-deployment.md#when-customers-will-not-upload-their-cad)).
- Every run is versioned: pretrained checkpoint, customer data manifest, preprocessing version, hyperparameters, seeds ([Experiment tracking](production-and-deployment.md#experiment-tracking-and-model-versions)).
- The pipeline is a container with a config file, so it reruns identically when the customer adds data.

## Architecture

```
customer sims ─► ingest ─► harmonize (units, frame, nondim) ─► QC ─► distribution check (embedding + zero-shot)
                                                                              │
                                    choose adaptation: zero-shot+calib / head / adapters / full FT
                                                                              │
                     grouped k-fold CV ─► baselines (GP, scratch, NN) ─► ensemble + conformal ─► report
                                                                              │
                                              deploy adapted model + OOD threshold ─► monitoring
```

## Risks and roadmap

- **Harmonization errors** (units, reference values) silently ruin fine-tuning; the zero-shot check catches many of them because errors look huge.
- **Overfitting with 50 cases:** few trainable parameters, early stopping and CV.
- **Negative transfer:** always compare with from-scratch training.
- **v0:** manual pipeline for one customer with a report.
- **v1:** automated, containerized pipeline with the decision table built in.
- **v2:** active learning to suggest the customer's next simulations.

## In an interview

**Thirty-second answer.** "The first steps are unglamorous but decisive: convert the 50 cases into exactly the representation the model was pretrained on, with the same units, frame and non-dimensionalization, then measure how far they are from the pretraining data with embedding distances and zero-shot errors. That decides the adaptation: calibration only if it's close, adapters or a new head for moderate shift, full fine-tuning with a small learning rate and early stopping if it's far. With 50 cases I'd use grouped five-fold cross-validation and compare against a GP and a model trained from scratch, because pretraining can hurt if the physics doesn't match. Intervals come from the fold ensemble with conformal calibration, and the customer's data stays isolated."

**Likely follow-ups**

- *How do you know pretraining helped?* Compare with from-scratch and GP baselines on the same folds; show the learning curves.
- *What if the customer needs a new output quantity?* Add a head on the frozen encoder; the data needs grow because that output was never seen in pretraining.

## References

- [Foundation models for physics](foundation-models-for-physics.md), [Learning from small simulation datasets](learning-from-small-simulation-datasets.md)
- E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022 ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685)).
- X. Li, Y. Grandvalet and F. Davoine, "Explicit inductive bias for transfer learning with convolutional networks" (L2-SP), ICML 2018 ([arXiv:1802.01483](https://arxiv.org/abs/1802.01483)).
