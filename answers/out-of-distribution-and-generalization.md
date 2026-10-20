# Out-of-Distribution Detection and Generalization

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A surrogate is reliable only near its training data: geometries, boundary conditions, physical regimes and mesh types like the ones it has seen. "The model generalizes to new geometries" means something only if it was tested on designs that were genuinely held out, such as whole design families or parameter ranges, not on a random split of similar shapes. In production, every query should get an out-of-distribution (OOD) score and an uncertainty estimate. Queries that score above a validated threshold are sent to the full solver, and their results become new training data.

**On this page**

- [What generalizing to new geometries means](#what-generalizing-to-new-geometries-means)
- [Interpolation vs extrapolation](#interpolation-vs-extrapolation)
- [Trusting one prediction for one new design](#trusting-one-prediction-for-one-new-design)
- [Detecting an out-of-distribution geometry](#detecting-an-out-of-distribution-geometry)
- [Validating the detector itself](#validating-the-detector-itself)
- [When to fall back to the full solver](#when-to-fall-back-to-the-full-solver)
- [Example: trained on Re 1e5 to 1e6, asked for 5e6](#example-trained-on-re-1e5-to-1e6-asked-for-5e6)
- [A new design family arrives in production](#a-new-design-family-arrives-in-production)
- [Code: a nearest-neighbour OOD score](#code-a-nearest-neighbour-ood-score)
- [In an interview](#in-an-interview)

## What generalizing to new geometries means

"Generalizes" is a claim about **which kind of new input** the model handles, and with **what error**. There are several levels, and they are very different in difficulty:

| Level | What changes at test time | Example |
| --- | --- | --- |
| 1. New samples, same distribution | Parameters drawn from the same ranges as training | A new random morph of the same car |
| 2. New parameter ranges | Values outside the training ranges | Longer wheelbase than any training car |
| 3. New design family | A different type of shape | Trained on notchbacks, tested on SUVs |
| 4. New topology | Features that never appeared | A rear spoiler or an extra hole in a bracket |
| 5. New physics regime | The flow or structural behaviour changes | Attached to separated flow; elastic to buckled |
| 6. New discretization | Different mesh, resolution or point sampling of the same shape | A customer's mesher instead of yours |

A random train/test split of a parametric dataset only tests level 1. A paper or vendor saying "generalizes to unseen geometries" may mean only that. To test a real claim:

- **Hold out whole families.** Leave-one-family-out: train on all families but one, test on the one left out, repeat.
- **Split by range.** Train on Reynolds numbers below a threshold and test above it, or train on small brackets and test on large ones.
- **Remesh the same shape.** Run the model on two different meshes or point samplings of the same geometry. The prediction should barely change. See [Why remeshing can change a prediction](geometry-representations.md#why-remeshing-can-change-a-prediction).
- **Report in-distribution and out-of-distribution errors separately**, with the error on the quantities that matter (drag, maximum stress), not just an average field error. The DoMINO paper, for example, reports both in-distribution and out-of-distribution test results.

See [Data splits and leakage](data-splits-and-leakage.md) for the mechanics.

## Interpolation vs extrapolation

- **Interpolation:** the new design lies in a region of design space that the training data covers densely. Errors there are roughly what the test set shows.
- **Extrapolation:** the design is beyond the parameter ranges, or has a new topology, a new regime or a new mesh type. Errors can be much larger, and nothing in the model's output warns you.

Two subtleties make this harder than it sounds:

1. **"Inside the convex hull" is too generous in high dimensions.** With 20 design parameters and 500 samples, almost every new point is inside the convex hull of the training set, yet most of that volume contains no nearby training data at all. What matters is local data **density**: how close the nearest training designs are, measured in a meaningful feature space.
2. **Extrapolation is the normal case in design.** An optimizer looks for the best design, and the best design is usually at the edge of what has been tried. Engineers also deliberately try new ideas. So a surrogate used in a design loop will spend much of its time near or beyond its data. See [Why optimizers exploit surrogates](surrogate-based-optimization.md#why-optimizers-exploit-surrogates).

A one-dimensional picture:

```
 training data:   x  x x  xx x  x xx  x x            (none here)            x
                 |---------- interpolation ----------|-- extrapolation --|
 surrogate error:   small, roughly as on test set     grows, and the model gives no warning
```

## Trusting one prediction for one new design

A test-set average says how good the model is *on average*. An engineer asks a different question: "Can I trust this number, for this design, today?" Answer it with layered checks:

1. **Is the input valid?** Parameters within bounds, sensible units, geometry watertight, orientation and scale as expected, boundary conditions inside the trained range.
2. **Is the design similar to the training data?** Distance from the design's geometry embedding to its nearest training designs. Far away means extrapolation. See [Detecting an out-of-distribution geometry](#detecting-an-out-of-distribution-geometry).
3. **Is the model uncertain?** Disagreement in an ensemble, or a GP's predictive variance. Large spread means the model has not pinned the answer down. See [Uncertainty quantification](uncertainty-quantification.md).
4. **Is that uncertainty calibrated?** On validation data, did 90% intervals contain the truth about 90% of the time? An uncalibrated error bar is decoration. Conformal prediction can give a coverage guarantee on in-distribution data. See [Conformal prediction](conformal-prediction.md).
5. **Is the output physically consistent?** Mass conservation, equilibrium of forces, boundary conditions satisfied, no stress where there is no load. A violation is a strong warning sign. See [Physics-consistency checks](evaluation-metrics.md#physics-consistency-checks).
6. **How did the model do on the nearest neighbours?** The validation error on the most similar training or validation designs is a good local estimate of the error to expect.
7. **How close is the decision to a limit?** If the predicted safety factor is 3.0 and the requirement is 1.5, a 10% error does not change the decision. If it is 1.55, run the solver. See [From uncertainty to a design decision](uncertainty-quantification.md#from-uncertainty-to-a-design-decision).

If the checks pass and the decision has margin, use the prediction. If any check fails, or the decision is close to a limit, run the solver.

## Detecting an out-of-distribution geometry

Useful signals, roughly from cheapest to most expensive:

1. **Input checks.** Parameter bounds; simple geometric descriptors (bounding-box size, volume, surface area, frontal area, number of holes) compared with training ranges; units and orientation.
2. **Distance in an embedding space.** Encode each geometry with the model's encoder (or a separate geometry autoencoder) and measure the distance from a new design to its k nearest training designs. Mahalanobis distance to the training embeddings is another option (Lee et al.). This is often the most useful single signal for geometries.
3. **Ensemble disagreement or predictive variance.** Epistemic uncertainty grows away from the data. It is a good signal, but neural ensembles can agree with each other and still be wrong far from the data, so do not rely on it alone.
4. **Reconstruction error of a geometry autoencoder.** Shapes unlike the training set reconstruct badly.
5. **Physics checks on the output.** Divergence of the predicted velocity, conservation, force balance, boundary-condition violations.
6. **For classifiers,** the maximum softmax probability (Hendrycks and Gimpel) is the classic baseline. For regression surrogates, use predictive variance and distance-based scores instead.

Combine several signals. A design can be close in embedding space but still fall in a new physics regime (the same shape at a much higher speed), which only a regime check on the operating conditions will catch.

## Validating the detector itself

An OOD detector is a model too, and it needs its own test:

- **Build known-OOD test sets:** held-out families, held-out ranges, deliberately unusual shapes.
- **Measure detection quality:** how well the score separates in-distribution from OOD inputs (AUROC is a standard summary).
- **Most important: does the score predict error?** Plot the score against the actual surrogate error on validation data. A good detector's score rises when the error rises.
- **Choose the threshold for a target risk.** For example: "among the queries we serve, fewer than 2% have a drag error above 3%", measured on validation data. Then accept that queries above the threshold go to the solver.

## When to fall back to the full solver

Send a query to the solver when **any** of the following holds:

- an input is outside the validated ranges (parameters, operating conditions, topology);
- the OOD score is above the validated threshold;
- the uncertainty is larger than the tolerance for this decision;
- a physics-consistency check fails;
- the decision is close to a limit, so a small error would change it;
- it is a sign-off or certification case.

The policy looks like this:

```
query ─▶ input checks ─▶ OOD score + uncertainty + physics checks
             │
             ├─ all pass, decision has margin ──▶ serve prediction with error bar
             │
             └─ any fails, or decision is close ─▶ run the solver
                                                    └─▶ add the result to the training data
```

The last arrow matters: every fallback produces a new labelled example in exactly the region where the model was weakest. That is [active learning](design-of-experiments-and-active-learning.md#active-learning-and-acquisition-functions) built into production.

## Example: trained on Re 1e5 to 1e6, asked for 5e6

A customer asks for a prediction at a Reynolds number five times above the training range. A good answer:

1. **Say clearly that this is extrapolation.** The model has never seen this regime, and its error there is unknown.
2. **Explain why the physics may change.** With increasing Reynolds number, boundary layers become thinner, laminar-to-turbulent transition can move, and separation points can shift. On bodies with curved surfaces, such as a cylinder or sphere, there is a well-known sudden drop in drag (the "drag crisis") when the boundary layer becomes turbulent before separating. Whether anything like that happens for this shape between 1e6 and 5e6 is exactly what the training data cannot tell you.
3. **Mention that some flows depend weakly on Reynolds number.** Bluff bodies with sharp edges, where separation is fixed by the geometry, often show little change at high Reynolds number. But that must be **checked**, not assumed.
4. **Also check the mesh assumptions.** The near-wall mesh of the training simulations was designed for a range of y+ values. At 5e6 the same mesh would give larger y+, so even the solver set-up must change. See [Boundary layers and y+](boundary-layers-and-wall-treatment.md#the-boundary-layer-and-y).
5. **Offer a concrete plan.** Run a few CFD cases at 5e6 (and in between), compare with the surrogate's prediction, and if the difference matters, add them to the training data and extend the validated range.

## A new design family arrives in production

Suppose the surrogate was trained on sedans and the customer starts submitting SUVs.

**How you notice:**

- the OOD score distribution of incoming queries shifts upwards;
- more queries fall back to the solver;
- the geometric descriptors (height, frontal area) drift outside training ranges;
- on the cases that were checked with the solver, the error is higher than on the validation set.

This is why production systems log the OOD score, the uncertainty and the input descriptors of every query, and run solver spot-checks on a sample of served predictions. See [Monitoring and retraining](production-and-deployment.md#monitoring-and-retraining) and [Design: drift monitoring](design-drift-monitoring.md).

**What you do:**

1. **Protect users first.** Route the new family to the solver (or flag it clearly) until the model is validated on it.
2. **Measure the gap.** Run the solver on a representative sample of the new family and compute the surrogate's error there.
3. **Collect data.** Use the fallback runs, plus a small [design of experiments](design-of-experiments-and-active-learning.md) over the new family.
4. **Retrain or fine-tune** on the extended dataset. Check that performance on the *old* families has not degraded.
5. **Re-validate and update the documented validity range**, then release a new, versioned model.

## Code: a nearest-neighbour OOD score

A minimal and surprisingly strong baseline: the average distance from a query's embedding to its k nearest training embeddings, with the threshold set from validation data.

```python
import numpy as np
from scipy.spatial import cKDTree

class KnnOODScore:
    """OOD score = mean distance to the k nearest training embeddings."""

    def __init__(self, train_emb, k=5):
        self.k = k
        self.tree = cKDTree(train_emb)

    def score(self, emb):
        d, _ = self.tree.query(emb, k=self.k)
        return d.mean(axis=1)

    def fit_threshold(self, val_emb, quantile=0.95):
        # Threshold such that 95% of in-distribution validation queries pass.
        self.threshold = np.quantile(self.score(val_emb), quantile)
        return self.threshold

    def is_ood(self, emb):
        return self.score(emb) > self.threshold

rng = np.random.default_rng(0)
train = rng.normal(0.0, 1.0, size=(2000, 16))     # embeddings of training designs
val = rng.normal(0.0, 1.0, size=(500, 16))        # in-distribution validation designs
far = rng.normal(3.0, 1.0, size=(500, 16))        # a shifted "new design family"

det = KnnOODScore(train, k=5)
det.fit_threshold(val)
print("flagged in-distribution:", det.is_ood(val).mean())   # about 0.05 by construction
print("flagged shifted family:", det.is_ood(far).mean())    # close to 1.0
```

In practice the embeddings come from the surrogate's own geometry encoder (or a separate autoencoder), and the threshold is chosen so that the flagged queries are the ones with large *actual* errors, not just a fixed quantile.

## In an interview

**Thirty-second answer to "What does generalizes to new geometries mean?"** "It depends on what was held out. A random split of one parametric family only shows interpolation. I would want to see results on held-out design families, on parameter ranges outside training, and on remeshed versions of the same shapes, with errors on the quantities that matter reported separately for in-distribution and out-of-distribution cases. In production, I would pair the model with an OOD score and send anything unfamiliar to the solver."

**Likely follow-ups**

- *How do you detect OOD for geometry?* Distance to the nearest training designs in an embedding space, plus input range checks, ensemble disagreement and physics checks on the output.
- *Can uncertainty alone tell you a prediction is wrong?* No. Ensembles can agree confidently far from the data. Combine uncertainty with distance-based scores and validate that the score correlates with error.
- *What do you do with the flagged cases?* Run the solver, return its result, and add it to the training set.

## References

- [A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks (Hendrycks & Gimpel)](https://arxiv.org/abs/1610.02136)
- [A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks (Lee et al., Mahalanobis distance)](https://arxiv.org/abs/1807.03888)
- [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles (Lakshminarayanan et al.)](https://arxiv.org/abs/1612.01474)
- [DoMINO (reports in-distribution and out-of-distribution results)](https://arxiv.org/abs/2501.13350)
- [AirfRANS (includes Reynolds-number and angle-of-attack extrapolation tasks)](https://arxiv.org/abs/2212.07564)
