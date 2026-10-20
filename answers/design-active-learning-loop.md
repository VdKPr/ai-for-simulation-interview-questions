# Design: Active Learning Loop

[← Questions](../README.md) · [All answers](README.md) · [System design framework](system-design-framework.md)

**Question.** Design an active learning loop that decides which simulation to run next.

**Short answer.** Start with a space-filling initial design, then repeat: train the surrogate, score candidate designs with an **acquisition function** that reflects the goal, pick a **batch** of diverse high-scoring candidates, simulate them in parallel, check their quality, add them, and retrain. The acquisition function depends on what the data is for:

- **predictive variance or expected error reduction** for a globally accurate surrogate;
- **expected improvement or similar** when the goal is an optimum (that is Bayesian optimization);
- **misclassification probability near a limit** for finding a failure boundary.

The surrogate must provide **useful uncertainty** (a GP, or an ensemble for deep models). Batches need diversity, or the loop wastes runs on near-duplicates. The loop stops on a budget or when the validation error stops improving. Keep a **fixed random test set** outside the loop: actively chosen data is biased and cannot be used to measure accuracy.

**On this page**

- [Requirements and clarifying questions](#requirements-and-clarifying-questions)
- [The loop](#the-loop)
- [Choosing the acquisition function](#choosing-the-acquisition-function)
- [Batches and parallel simulations](#batches-and-parallel-simulations)
- [Uncertainty for deep surrogates](#uncertainty-for-deep-surrogates)
- [Evaluation without bias](#evaluation-without-bias)
- [Architecture](#architecture)
- [Risks and roadmap](#risks-and-roadmap)
- [In an interview](#in-an-interview)

## Requirements and clarifying questions

- **Goal:** accurate everywhere, find the best design, or find where a constraint fails? This is the most important question.
- **Cost:** how long does one simulation take, and how many can run in parallel? Sets the batch size.
- **Dimensionality:** a few parameters (GPs work well) or free-form geometry (deep models with ensembles, candidates from a generator or a library of shapes).
- **Candidate pool:** a continuous space to optimize over, or a finite pool of feasible designs?
- **Failures:** do some designs fail to mesh or converge? The loop must handle them.
- **Budget and stopping:** total runs, wall-clock time, target accuracy.

## The loop

1. **Initial design:** Latin hypercube or Sobol sampling, about 10 × the number of parameters for low-dimensional problems ([Latin hypercube sampling](design-of-experiments-and-active-learning.md)).
2. **Train** the surrogate on all good data so far.
3. **Generate candidates:** a large random or quasi-random pool in the design space (or a pool of real geometries), filtered for feasibility.
4. **Score** candidates with the acquisition function.
5. **Select a batch** of B diverse candidates (below).
6. **Simulate** in parallel through the data pipeline ([Simulation data pipeline](design-simulation-data-pipeline.md)), with quality checks.
7. **Update:** add good runs; record failed ones as "infeasible" information (a classifier of failure regions can also steer the loop away).
8. **Evaluate** on the fixed test set; log the learning curve; decide whether to stop.

## Choosing the acquisition function

| Goal | Acquisition | Notes |
| --- | --- | --- |
| Global accuracy | Maximum predictive variance; integrated variance reduction (IMSE); expected error from cross-validation | Variance alone favours boundaries of the space; integrated variance accounts for where predictions are used |
| Optimization | Expected improvement, upper confidence bound, knowledge gradient | See [Bayesian optimization](bayesian-optimization.md) |
| Failure boundary or contour | Probability of misclassification, U-function (AK-MCS), expected feasibility | For reliability: find where stress = allowable |
| Multiple outputs | Weighted sum of normalized variances, or the output most relevant to the decision | Avoid one noisy output dominating |
| Multi-fidelity | Cost-weighted variance reduction across fidelities | Pick both the design and the fidelity ([Multi-fidelity learning](multi-fidelity-learning.md)) |

The demo in [Design of experiments and active learning](design-of-experiments-and-active-learning.md) compares variance-based selection with random sampling.

## Batches and parallel simulations

Simulations run in parallel, so the loop picks batches. Picking the top B scores gives near-duplicates. Options:

- **Kriging believer / constant liar:** pick the best point, pretend its result is the current prediction, update the variance, pick again. Cheap and effective with GPs.
- **Local penalization or distance constraints:** forbid candidates within a radius of already selected ones.
- **Clustering:** cluster the top-scoring candidates in input or embedding space and take one per cluster.
- **Determinantal point processes or batch BALD:** principled diversity, more expensive.

Asynchronous loops (start a new run when any finishes) keep the cluster busy when run times vary.

## Uncertainty for deep surrogates

GPs give calibrated variance for small, low-dimensional problems. For GNNs or transformers on free-form geometry:

- **Deep ensembles** (3–5 models) give the most reliable epistemic uncertainty in practice; disagreement is the acquisition signal ([Deep ensembles](uncertainty-quantification.md#deep-ensembles)).
- **Distance in embedding space** to training designs is a cheap complement (an OOD-style score).
- **Retraining cost:** fine-tune from the previous round's weights instead of training from scratch, and retrain fully every few rounds.

## Evaluation without bias

- Actively chosen points cluster in hard regions; their error overstates the typical error, and the loop cannot evaluate itself on them.
- **Keep a held-out test set sampled from the distribution of use** (random or space-filling), never chosen by the loop.
- Compare the learning curve against **random sampling with the same budget**: active learning should reach the target accuracy with fewer runs. If not, the uncertainty is uninformative.

## Architecture

```
initial DoE ─► simulate ─► QC ─► dataset ─► train surrogate (GP / ensemble)
                                    ▲                    │
                                    │                    ▼
             simulate batch ◄── select diverse batch ◄── score candidates (acquisition)
                                                         │
                         fixed test set ─► learning curve ─► stop? (budget, plateau)
```

## Risks and roadmap

- **Uninformative uncertainty:** the loop becomes random sampling at best. Check calibration ([Checking calibration](uncertainty-quantification.md#checking-calibration)).
- **Exploitation of model errors:** the surrogate is confidently wrong in a region, so the loop never samples it. Keep a small fraction of random samples in each batch.
- **Failures cluster:** model the failure region separately.
- **v0:** GP with variance-based batches on a low-dimensional parameter space.
- **v1:** ensembles for a deep surrogate; asynchronous scheduling.
- **v2:** multi-fidelity acquisition; goal-specific acquisition for optimization or reliability.

## In an interview

**Thirty-second answer.** "First I'd ask what the data is for, because that picks the acquisition function: variance reduction for global accuracy, expected improvement for optimization, misclassification probability for a failure boundary. The loop starts from a space-filling design. Each round trains the surrogate with useful uncertainty, either a GP or an ensemble, scores a large candidate pool, and picks a diverse batch so parallel runs aren't duplicates. Runs are quality-checked and failures are recorded. I'd measure progress on a fixed random test set, never on the chosen points, and compare with random sampling at the same budget."

**Likely follow-ups**

- *How big should batches be?* As large as the parallel capacity allows, traded against how much each round's model improves; often 5–50.
- *When do you stop?* Budget reached, or the test error has plateaued for several rounds, or the acquisition values have dropped below a threshold.

## References

- B. Settles, *Active Learning Literature Survey*, University of Wisconsin–Madison, 2009.
- D. R. Jones, M. Schonlau and W. J. Welch, "Efficient Global Optimization of Expensive Black-Box Functions", *Journal of Global Optimization*, 1998.
- B. Echard, N. Gayton and M. Lemaire, "AK-MCS: An active learning reliability method combining Kriging and Monte Carlo simulation", *Structural Safety*, 2011.
- [BoTorch](https://github.com/pytorch/botorch) (batch acquisition functions), [Design of experiments and active learning](design-of-experiments-and-active-learning.md)
