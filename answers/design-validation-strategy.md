# Design: Validation Strategy

[← Questions](../README.md) · [All answers](README.md) · [System design framework](system-design-framework.md)

**Question.** Design a validation strategy that would convince a sceptical customer to trust your surrogate.

**Short answer.**

- **Agree the acceptance criteria with the customer before testing.** That means the metrics tied to their decision, the thresholds, and the test designs they choose.
- **Test blind on their held-out designs,** including the hard cases they pick. Use splits that reflect real use (new designs, not interpolated neighbours of training data).
- **Report errors in their units and terms:** drag counts, peak stress in MPa, ranking agreement. Show the error **distribution** and the worst cases, not only the mean.
- **Compare against the solver's own uncertainty** (mesh and model variation) and against simple baselines.
- **Check that the uncertainty is calibrated** and that the OOD detector catches designs outside the training range.
- **Run a shadow phase:** the surrogate runs alongside the existing process, and its predictions are logged and compared, before anyone relies on it.
- **Write it all up** in a validation report with the limits of validity stated.

**On this page**

- [What the sceptic needs to see](#what-the-sceptic-needs-to-see)
- [Acceptance criteria agreed in advance](#acceptance-criteria-agreed-in-advance)
- [Test sets that mean something](#test-sets-that-mean-something)
- [What to measure](#what-to-measure)
- [Uncertainty and OOD validation](#uncertainty-and-ood-validation)
- [Shadow mode and staged adoption](#shadow-mode-and-staged-adoption)
- [The validation report](#the-validation-report)
- [In an interview](#in-an-interview)

## What the sceptic needs to see

A simulation engineer trusts a solver because of verification and validation: known equations, convergence studies, comparison with experiments. For a surrogate, the equivalent evidence is:

1. it matches the solver on **cases like theirs** that it has not seen;
2. it **knows when it does not know** (uncertainty and OOD flags that work);
3. its errors are **small relative to the decision** and to the solver's own uncertainty;
4. failure modes are **documented**, so they know where not to use it.

## Acceptance criteria agreed in advance

Write them down with the customer **before** running the test, so nobody can move the goalposts:

| Item | Example |
| --- | --- |
| Quantity | Drag coefficient; maximum von Mises in region A |
| Metric | Absolute error; rank correlation across variants; pass/fail agreement against an allowable |
| Threshold | 90% of test designs within 2% of CFD drag; no false "pass" on stress |
| Test set | 30 designs chosen by the customer, not seen in training |
| Conditions | Within the stated design space and operating range |
| Uncertainty | 90% intervals cover the truth in 85–95% of test cases |

## Test sets that mean something

- **Customer-chosen blind designs:** they select (and can even generate) the test cases; we do not see the answers before predicting.
- **Grouped splits:** by design family or baseline, so test designs are not near-duplicates of training designs ([Data splits and leakage](data-splits-and-leakage.md)).
- **Slices that matter:** extreme parameter values, designs with flow separation, thin-walled parts, each reported separately ([Evaluation metrics](evaluation-metrics.md)).
- **An explicit OOD set:** designs deliberately outside the training range, to show the detector works and the model's error grows as expected.
- **Temporal split:** if designs evolve over a programme, test on the latest ones.

## What to measure

- **Quantities of interest** in engineering units, with distributions (box plots, worst five cases shown individually).
- **Fields:** relative L2 overall and per region; visual comparisons on the worst cases.
- **Decision-level agreement:** does the surrogate pick the same best design out of N? Does it agree on pass/fail?
- **Physics checks:** conservation, symmetry, force consistency.
- **Baselines:** nearest training design, a GP on parameters, a coarse solver run of equal cost. The surrogate should beat them clearly ([Weak baselines](https://arxiv.org/abs/2407.07218) are a known problem in the field).
- **Reference uncertainty:** run some test cases on a refined mesh or a different turbulence model. If the solver itself varies by 2%, a 1% surrogate error is within noise ([Verification vs validation](verification-validation-and-mesh-convergence.md#verification-vs-validation)).
- **Speed, end to end,** including geometry processing.

## Uncertainty and OOD validation

- **Calibration:** coverage of the stated intervals on the test set; a reliability diagram ([Checking calibration](uncertainty-quantification.md#checking-calibration)).
- **Error vs uncertainty:** high predicted uncertainty should go with high error (rank correlation).
- **Detector performance:** on the OOD set, the fraction flagged (recall) and the fraction of in-distribution designs wrongly flagged (false alarms) ([Validating the detector itself](out-of-distribution-and-generalization.md#validating-the-detector-itself)).
- **Risk–coverage curve:** if all flagged designs go to the solver, what is the error on the remaining ones, and what fraction of queries still need the solver?

## Shadow mode and staged adoption

1. **Shadow mode** (weeks to months): engineers use their normal process; the surrogate predicts every case silently; differences are logged and reviewed.
2. **Advisory use:** surrogate results shown next to solver results; used to prioritize which designs to simulate.
3. **Screening use:** surrogate replaces the solver for early exploration; final designs verified with the solver.
4. **Ongoing monitoring:** the solver-verified cases continue to measure accuracy in production ([Drift monitoring](design-drift-monitoring.md)).

## The validation report

- Scope: design space, operating conditions, quantities predicted.
- Data: number and source of training cases; solver and settings.
- Test protocol: the agreed criteria, how test designs were chosen, who ran them.
- Results: against criteria, with distributions, worst cases and slices.
- Uncertainty and OOD performance.
- **Limits of validity:** where the model must not be used.
- Versioning: model, data and code versions validated; re-validation triggers (new data, new model version).

## In an interview

**Thirty-second answer.** "I'd agree the acceptance criteria with the customer before testing: the quantity, the metric, the threshold and a set of blind test designs they choose themselves. We'd report errors in their units, with distributions and worst cases, compared against simple baselines and against the solver's own mesh and model variation. I'd show that the uncertainty is calibrated and that the OOD detector flags designs outside the training range. Then a shadow phase, where the surrogate runs silently alongside their process, and a written report that states the limits of validity."

**Likely follow-ups**

- *What if they want a guarantee?* Offer a statistical one: conformal intervals with stated coverage on exchangeable data, plus solver verification of final designs ([Conformal prediction](conformal-prediction.md)).
- *What if the test fails the criteria?* Report it, diagnose which slices fail, and propose a narrower scope or more data.

## References

- W. L. Oberkampf and C. J. Roy, *Verification and Validation in Scientific Computing*, Cambridge University Press, 2010.
- ASME V&V 20-2009, *Standard for Verification and Validation in Computational Fluid Dynamics and Heat Transfer*.
- N. McGreivy and A. Hakim, "Weak baselines and reporting biases lead to overoptimism in machine learning for fluid-related PDEs", *Nature Machine Intelligence*, 2024 ([arXiv:2407.07218](https://arxiv.org/abs/2407.07218)).
- [Customer scenarios](customer-scenarios.md#i-dont-trust-black-boxes), [Evaluation metrics](evaluation-metrics.md)
