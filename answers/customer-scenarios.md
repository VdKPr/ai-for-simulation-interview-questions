# Customer Scenarios

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Physics-AI companies sell to engineers who are trained to distrust numbers they cannot verify, and many roles (forward-deployed, application or customer-facing ML engineers) spend real time with those engineers. Scenario questions test four things: whether you **agree on the decision and the acceptance criteria before building**, whether you are **honest about limits** (and say so early), whether you can **separate error sources** with evidence instead of arguing, and whether you can **explain** ML to non-specialists without overselling. A good answer has a structure: acknowledge the concern, ask the clarifying questions, propose concrete evidence or options, and say what you would agree in writing. Below, each scenario has the reasoning, a model answer and the follow-ups to expect.

**On this page**

- [How to answer a scenario question](#how-to-answer-a-scenario-question)
- [I don't trust black boxes](#i-dont-trust-black-boxes)
- [The surrogate disagrees with wind-tunnel data](#the-surrogate-disagrees-with-wind-tunnel-data)
- [1% accuracy from 30 simulations](#1-accuracy-from-30-simulations)
- [A failure found one week before delivery](#a-failure-found-one-week-before-delivery)
- [Explaining a neural operator to a manager](#explaining-a-neural-operator-to-a-manager)
- [Proof of concept or robust feature](#proof-of-concept-or-robust-feature)
- [Scoping a project in the first meeting](#scoping-a-project-in-the-first-meeting)
- [Disagreeing with a domain expert](#disagreeing-with-a-domain-expert)
- [When ML is the wrong tool](#when-ml-is-the-wrong-tool)
- [In an interview](#in-an-interview)

## How to answer a scenario question

1. **Acknowledge** the concern as legitimate. Engineers are right to be sceptical; arguing with that loses the room.
2. **Clarify** before solving: which decision, which quantity, what accuracy, what it costs to be wrong.
3. **Propose evidence**, not reassurance: a test on their data, a comparison, a learning curve.
4. **Offer options** with trade-offs, and let the customer choose.
5. **Close the loop:** what you agree in writing, what you will monitor, what happens if it fails.

Interviewers mark down three things: promising accuracy you have not measured, hiding bad news, and blaming the data or the customer.

## I don't trust black boxes

**What is behind it.** The engineer's job is to sign off on numbers. A solver has a known basis (equations, discretization, convergence checks); a neural network does not, and it can fail silently on a design unlike its training data. The engineer is right, so the answer is not to argue but to make the model **checkable**.

**What to do.**

- **Agree:** "Trust should be earned with evidence, and the solver stays the reference."
- **Validate on their cases:** hold out designs they choose (ideally ones they consider hard), run the model blind, and compare fields and quantities of interest in their post-processing tool. Show where it fails as well as where it works.
- **Make errors visible:** uncertainty on every prediction (ensembles or conformal intervals) and an OOD score that flags designs unlike the training data. See [Uncertainty quantification](uncertainty-quantification.md) and [Conformal prediction](conformal-prediction.md).
- **Keep physics checks:** conservation, symmetry and integrated forces consistent with the fields. See [Evaluation metrics](evaluation-metrics.md).
- **Start where errors are cheap:** screening and ranking early in design, where a wrong answer costs a solver run, not a product. Final designs are verified with the solver.
- **Audit trail:** each prediction logged with the model version, inputs, uncertainty and OOD score.

**Model answer.** "You're right to be sceptical; I would be too. Let's not argue about it in the abstract. Pick ten designs you know well, including a couple you think will break it, and we'll run them blind against your solver results. Every prediction will come with an uncertainty and a flag when the design is unlike anything it was trained on, and those cases go back to the solver. We'd start using it for screening, where a wrong answer costs a re-run, and you'd still verify the final design with the solver."

**Follow-ups.** *What if the validation shows 8% error where they need 3%?* Say so, show which designs drive it, and offer options (more data in that region, a narrower scope, or the solver for those cases). *Can you explain why the model predicts what it does?* Partly: sensitivity to inputs, nearest training designs, error decomposition; but the real basis for trust is measured performance on relevant cases, not an explanation.

## The surrogate disagrees with wind-tunnel data

**The key insight.** A surrogate trained on CFD can only be as close to the tunnel as the CFD is. There are two separate gaps, and the first job is to measure them separately:

```math
\underbrace{y_\text{surrogate} - y_\text{tunnel}}_\text{what the customer sees}
= \underbrace{(y_\text{surrogate} - y_\text{CFD})}_\text{ML error}
+ \underbrace{(y_\text{CFD} - y_\text{tunnel})}_\text{modelling and set-up error}
```

**What to do.**

1. **Run CFD on the tunnel configurations.** If CFD also disagrees with the tunnel, the surrogate is faithfully reproducing CFD's error.
2. **Check that the comparison is like for like:**
   - geometry detail (the tunnel model may have mirrors, gaps or a rough finish that the CAD lacks);
   - Reynolds number, yaw angle, ride height;
   - ground simulation (moving belt and rotating wheels vs a fixed floor);
   - blockage and other tunnel corrections; where the force is measured and what it includes;
   - turbulence model and transition in the CFD; mesh convergence.
3. **Check the ML error** on these cases: are they inside the training distribution? What is the OOD score?
4. **Fix the right thing:**
   - ML error dominates → more training data near those designs.
   - CFD bias dominates → better CFD set-up, or a calibration or multi-fidelity correction that uses tunnel data as the high-fidelity source. See [Multi-fidelity learning](multi-fidelity-learning.md).
   - **Deltas vs absolutes:** CFD and tunnel often agree better on the *difference* between two designs than on absolute values. If the decision is a ranking, evaluate that.

**Model answer.** "First I'd split the disagreement into two parts: the surrogate against the CFD it was trained on, and that CFD against the tunnel. If I run CFD on the tunnel configurations and CFD also misses, the surrogate is faithfully copying a CFD bias, and the fix is in the CFD set-up or a calibration on tunnel data, not in the network. I'd also check that we compare like with like: ground simulation, Reynolds number, yaw, blockage corrections and geometry detail. And I'd check whether the decision needs absolute values or design-to-design deltas, because deltas often agree much better."

## 1% accuracy from 30 simulations

**What is behind it.** The number is usually a proxy for "accurate enough to make decisions". The answer is to find the real requirement and show, with data, what 30 simulations can deliver.

**What to do.**

- **Clarify the target:** 1% of what? Drag (one number) is very different from 1% everywhere in a pressure field, or 1% on peak stress (a local maximum, much harder). 1% across which design space, and for which decision?
- **Compare with the reference's own uncertainty:** mesh refinement, turbulence models and solver settings often change CFD results by more than 1%. Asking a surrogate to be more accurate than the solver's own variation is not meaningful. See [Mesh convergence](verification-validation-and-mesh-convergence.md#mesh-convergence-and-the-grid-convergence-index).
- **Measure, don't promise:** fit a GP or a small model on 10, 20 and 30 simulations and plot the cross-validated error. A **learning curve** shows what 30 simulations give and roughly how many more would be needed.
- **Offer options:**
  - narrow the design space (fewer parameters, smaller ranges);
  - more simulations, placed by active learning ([Design of experiments](design-of-experiments-and-active-learning.md));
  - cheaper low-fidelity runs plus the 30 high-fidelity ones ([Multi-fidelity learning](multi-fidelity-learning.md));
  - a pretrained model fine-tuned on the 30 ([Foundation models](foundation-models-for-physics.md));
  - accept ranking accuracy instead of absolute accuracy, if the decision is choosing the best designs.
- **Agree acceptance criteria in writing**: metric, test designs, threshold.

**Model answer.** "Let's first pin down what the 1% is for: which quantity, over which range, and for which decision. Then I'd check the solver's own variation, because if a mesh change moves drag by 1.5%, a 1% surrogate target isn't meaningful. With 30 simulations I'd fit a GP and show a learning curve, so we both see what accuracy the data supports. If it's not enough, we can narrow the design space, add simulations where the model is most uncertain, use cheaper low-fidelity runs, or agree a ranking criterion. What I won't do is promise 1% before we've measured anything."

## A failure found one week before delivery

**Principle.** Bad news gets worse with time. Tell the customer now, with the facts and options, not on delivery day.

**What to do.**

1. **Quantify:** which design family, how large the error, how many designs it affects, and whether the OOD score or uncertainty flags it (if it does, the system already protects users partially).
2. **Diagnose quickly:** missing training data for that family? A geometry-processing issue? A physics regime the model has not seen (separation, a new contact)?
3. **Tell the customer with options**, for example:
   - deliver on time with that family routed to the solver automatically (documented limitation);
   - run targeted simulations for that family and deliver an update in N weeks;
   - delay the whole delivery (rarely the best option).
4. **Afterwards:** add that family as a named slice in the evaluation suite and regression tests, and find out why the evaluation missed it (wrong split? family absent from the test set?). See [Data splits and leakage](data-splits-and-leakage.md).

**Model answer.** "I'd tell them straight away, not at delivery. Before the call I'd quantify the problem and check whether our uncertainty or OOD flags catch those designs. Then I'd offer options: deliver on time with that family routed to the solver, or run targeted simulations and ship an update in a few weeks. And I'd add that family to our test suite, because the real failure is that our evaluation didn't catch it earlier."

## Explaining a neural operator to a manager

Aim for two minutes, no jargon, one analogy, and one honest limit.

> "Our solver computes the airflow around a design from the laws of physics. It's accurate but takes hours. A neural operator learns the solver's behaviour from thousands of designs that have already been solved: you give it a new shape and it gives the whole flow field in seconds. Think of an experienced engineer who has seen thousands of results and can sketch the answer at a glance, except that it's quantitative.
>
> It's reliable for designs similar to what it has learned from, and less so for designs that are very different. So we measure its accuracy on test designs, every answer comes with a confidence level, and when a design is unfamiliar it sends the job back to the solver. The value is that your team can explore hundreds of options in the time it takes to simulate one, and use the solver to verify the best ones."

What the manager actually wants to know: what it costs, how fast it is, how much to trust it, and what the risk is. Answer those, not how Fourier layers work. For the technical version, see [Neural operators](neural-operators.md).

## Proof of concept or robust feature

The choice depends on **what is uncertain**:

| Situation | Choose | Why |
| --- | --- | --- |
| Value is unproven: will a surrogate be accurate enough to matter for this customer? | Proof of concept (weeks) | Cheapest way to answer the biggest uncertainty |
| Value is proven and users depend on it | Robust feature | Reliability, monitoring and maintainability now dominate |
| A demo for a deal | Proof of concept, scoped honestly | Do not let a demo become production by accident |

**What is never optional, even in a proof of concept:** honest evaluation (grouped splits, a baseline, held-out designs the customer chose). A proof of concept with leaky evaluation gives the wrong answer to the only question it exists to answer.

**Model answer.** "I ask what we're uncertain about. If we don't know whether the approach can reach useful accuracy, a quick proof of concept answers that cheaply, but with rigorous evaluation, because otherwise it answers the wrong question. Once users depend on it, robustness wins: tests, monitoring, versioning. And I try to make the proof-of-concept code easy to harden rather than throw away."

## Scoping a project in the first meeting

**Questions to ask.**

| Area | Questions |
| --- | --- |
| Decision | What decision will this support? Who makes it, and how often? What happens now, how long does it take and what does it cost? |
| Outputs | Which quantities or fields? What accuracy matters for the decision? Ranking or absolute values? |
| Inputs | How do designs vary: parameters, or free-form CAD? Which operating conditions? |
| Data | How many simulations exist? Which solver, fidelity, formats? Are they converged and consistent? Who owns them? Can more be generated, at what cost? |
| Constraints | Cloud or on-premises? Security and IP rules? Latency? Integration with which tools? |
| Success | What would make this a success in three months? Who signs off? |

**Then:** summarize back what you heard, propose a small first milestone with measurable acceptance criteria (for example, a learning curve on existing data within four weeks), and name the main risk.

**Model answer.** "I'd spend most of the first meeting listening: what decision they're trying to speed up, how they make it today, what data they have and what accuracy would change their process. I'd close by summarizing, proposing a first milestone with a measurable criterion, usually a study on their existing data, and naming the biggest risk, which is usually data coverage."

## Disagreeing with a domain expert

**Principle.** Treat the disagreement as a **hypothesis to test**, not a contest. Domain experts are often right about physics that is not in the data; ML engineers are sometimes right about what the data shows. Evidence decides.

**What to do.**

1. Understand their reasoning fully, and restate it until they agree with your restatement.
2. Find the specific, testable point of disagreement ("the model predicts separation here; you say it shouldn't happen at this Reynolds number").
3. Agree a test **before** running it: a refined simulation, a literature result, a test measurement.
4. Run it and accept the result either way. Write down the decision and the evidence.

**Model answer.** "I'd first make sure I understand their reasoning, because they often know physics the data doesn't show. Then I'd narrow it to something testable and agree with them, in advance, what result would settle it: a refined simulation, a known benchmark, a measurement. If they're right, the model or data gets fixed; if the evidence goes the other way, they've seen it for themselves. Either way it gets written down."

## When ML is the wrong tool

Say no when the speed-up does not pay for the data, or when the risk is in extrapolation:

- **One-off analyses:** a single simulation is cheaper than generating training data.
- **Few simulations with many inputs:** the data cannot cover the design space.
- **Extrapolation:** new physics regimes, new materials, conditions outside all data.
- **Certification and final sign-off:** regulators and engineers need the solver (or the test).
- **A simpler method works:** an analytical correlation, a coarse mesh, a reduced-order model, or a GP on a few parameters.
- **The break-even does not work:** training data costs N solver runs; if the customer will only ever run fewer than N queries, ML loses. See [ML vs numerical solvers](ml-vs-numerical-solvers.md).

**Model answer.** "When the customer will run fewer new cases than it costs to generate the training data, when they need to extrapolate beyond anything simulated, or when the result goes into certification. In those cases I'd say so and suggest the cheaper option, such as a coarser mesh or a reduced-order model. Saying no to a bad use case is what makes the good use cases believable."

## In an interview

**Thirty-second summary of the approach.** "With a customer, I start from the decision they need to make and agree measurable acceptance criteria before building anything. When something disagrees, I separate the error sources with evidence: surrogate vs solver, solver vs reality. I give bad news early with options, and I'm explicit about what the model shouldn't be used for. Trust comes from tests on their own cases, visible uncertainty and a solver fallback, not from reassurance."

**Likely follow-ups**

- *Give an example from your own work.* Prepare one real story per scenario; see [Behavioural questions](behavioral-questions.md).
- *What if the customer insists on a promise?* Offer a staged commitment: a measured result on their data first, then a target based on it.
- *What if your manager wants you to oversell?* Explain the risk to the relationship and propose an honest framing that still shows value.

## References

- [PhysicsX ML Engineer job description](https://job-boards.greenhouse.io/physicsx/jobs/4880947101) (customer-facing project delivery, travel to customer sites)
- [Luminary forward-deployed engineer posting](https://ats.rippling.com/luminarycloud/jobs/c2b9847d-186c-4675-a32f-1a9ef7d478b4) ("rigorous experimentation and honest evaluation")
- W. L. Oberkampf and C. J. Roy, *Verification and Validation in Scientific Computing*, Cambridge University Press, 2010 (separating numerical, modelling and experimental error).
- [Verification, validation and mesh convergence](verification-validation-and-mesh-convergence.md), [Industry notes](../industry.md#which-industries-buy-and-why-customers-hesitate)
