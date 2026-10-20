# ML vs Numerical Solvers

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A numerical solver (FEM, FVM, FDM) computes an answer from the governing equations. If the mesh and time step are fine enough, the answer converges, and the error behaviour is well understood for any input the physical model covers. Each run is expensive. An ML surrogate is cheap per query after training, but it is only as good as its training data, has no built-in error guarantee, and degrades, often silently, outside the training distribution. Use ML when you need many evaluations of *similar* problems and can measure and tolerate its error. Keep the solver for one-off analyses, new regimes and safety-critical sign-off, and use it to check the surrogate.

**On this page**

- [Side by side](#side-by-side)
- [Where ML helps](#where-ml-helps)
- [When not to replace a solver with ML](#when-not-to-replace-a-solver-with-ml)
- [Break-even arithmetic](#break-even-arithmetic)
- [Questions to ask about a 1000x speed-up claim](#questions-to-ask-about-a-1000x-speed-up-claim)
- [Weak baselines and reporting biases](#weak-baselines-and-reporting-biases)
- [In an interview](#in-an-interview)

## Side by side

| | Numerical solver (FEM/FVM) | ML surrogate |
| --- | --- | --- |
| Cost per query | High: minutes to days | Low: milliseconds to seconds |
| Up-front cost | Set-up and meshing for each case | Data generation plus training, once per design family |
| How error is controlled | Refine the mesh or time step; convergence theory says what to expect | Only measured empirically, on test data |
| New geometry or regime | Works if the physical model is valid there | Reliable only near the training data |
| How it fails | Usually loudly: divergence, solver errors, warnings | Often silently: a smooth, plausible-looking field that is wrong |
| Gradients for optimization | Adjoint solvers, where they exist | Automatic differentiation through the model |
| Explaining a result | Equations, mesh, settings: auditable | Harder; needs evaluation evidence |

The two are not competitors so much as tools for different jobs. The solver is the reference. The surrogate is a fast approximation of the reference over a region of design space that you have mapped with data.

## Where ML helps

- **Many-query tasks:** design-space exploration, optimization, robustness and uncertainty studies, where the same kind of simulation runs thousands of times.
- **Real-time use:** interactive design tools and digital twins, where waiting minutes is not acceptable.
- **Inverse problems and data assimilation:** recovering unknown parameters or fields from sparse measurements. See [Inverse problems](inverse-problems-and-regularization.md).
- **Partially unknown physics:** learning a turbulence closure, a constitutive law or a correction from data.
- **Accelerating the solver itself** without changing its answer: warm starts, learned preconditioners, coarse-grid corrections. The solver's convergence check still decides the final answer. See [Newton-Raphson and nonlinear solvers](newton-raphson-and-nonlinear-solvers.md#speeding-up-a-solver-with-ml-without-changing-its-answer) and [Hybrid solvers](differentiable-physics-and-hybrid-solvers.md).

## When not to replace a solver with ML

This is the "must know" question. Good candidates give several concrete cases, not a general "when accuracy matters".

1. **One-off analyses.** If you need the answer for one design, generating hundreds of training simulations costs far more than running that one case.
2. **New regimes or extrapolation.** A model trained on Reynolds numbers 1e5 to 1e6 has seen no examples of the physics at 5e6. The same holds for new load types, new materials and new topologies. The solver handles these as long as its physical model is valid. See [the Reynolds number example](out-of-distribution-and-generalization.md#example-trained-on-re-1e5-to-1e6-asked-for-5e6).
3. **Sign-off and certification.** Regulators and safety engineers need traceable, verifiable results. A surrogate can screen designs, but the final structural check or crash certification is run with the validated solver, unless the surrogate has itself been validated for exactly that use.
4. **When a cheaper classical method is good enough.** A coarser mesh, a reduced-order model, potential flow, beam theory or a lookup table may already be fast enough, with known error.
5. **When the data does not exist and cannot be made.** If the simulations cannot be automated, or each costs a week, there will not be enough training data.
6. **Highly non-smooth responses near the decision boundary.** Buckling onset, contact, flow separation and fracture can switch the answer suddenly. A smooth surrogate blurs exactly the transition you care about.
7. **When nobody can check the output.** A surrogate without uncertainty estimates and out-of-distribution checks, used by people who cannot judge the physics, will eventually produce a confident wrong answer.

The strong answer then adds the **hybrid**: the surrogate screens or warm-starts, the solver verifies the shortlist. This keeps most of the speed and all of the trust.

## Break-even arithmetic

Suppose one solver run costs 1 unit of compute. Generating a training set costs N units, training costs T units and each surrogate query costs almost nothing. Each query then saves about 1 unit, so the surrogate pays for itself after about N + T queries, plus whatever validation runs are needed.

With N = 500 training simulations and T = 50 units of training compute, that is about 550 queries before you are ahead. If the model will be used for 20,000 optimization evaluations over the life of a project, it is clearly worth it. If it will be used for 30 design reviews, it is not.

These numbers are illustrative. In a real conversation, work them out with the customer's actual costs, and include the engineer time for building the data pipeline and the maintenance of the model.

## Questions to ask about a 1000x speed-up claim

A "1000× faster than CFD" headline usually compares a trained network's inference time with the wall time of a full simulation. That can be true and still be misleading. Ask:

1. **Faster at what accuracy?** Was the solver run with a mesh much finer than needed for the accuracy the surrogate achieves? A fair comparison is at *equal accuracy*: what does a coarser simulation that matches the surrogate's error cost?
2. **What solver settings and hardware?** An unoptimized or old solver on a few CPU cores against a network on a modern GPU is not a fair race. Was the solver also run on GPUs, with a good preconditioner and a sensible convergence tolerance?
3. **What is included in the timing?** Geometry cleanup, meshing or SDF computation, point sampling, data transfer, inference and post-processing are all part of the pipeline. Measuring only the network's forward pass leaves out the parts that may dominate.
4. **What about the training cost?** How many simulations were needed to train it, and how much GPU time? For a use case with only a few queries, that cost is the whole story.
5. **Accuracy on which designs?** Interpolation within the training distribution or genuinely new designs? Average error, or the error on the worst designs and on the QoIs that drive decisions?
6. **What baseline?** Was the comparison against a strong classical method, or a weak one? See the next section.
7. **Does the error bar hold up?** Is there any uncertainty estimate, and is it calibrated?

A candidate who asks these questions calmly, without dismissing the result, shows exactly the judgement these companies hire for. The fair-reporting checklist is in [Reporting a speed-up fairly](evaluation-metrics.md#reporting-a-speed-up-fairly).

## Weak baselines and reporting biases

In 2024, Nick McGreivy and Ammar Hakim published a systematic review of machine learning for solving fluid-related PDEs (*Nature Machine Intelligence*). Their main findings:

- **Weak baselines are common.** Of the articles that used ML to solve a fluid-related PDE and claimed to outperform a standard numerical method, **79% (60 of 76)** compared against a weak baseline. Typical weaknesses: comparing with a numerical method at a much higher accuracy than the ML model achieves instead of at equal accuracy, or using a less efficient numerical method than the state of the art for that problem.
- **Reporting biases are widespread,** especially *outcome reporting bias* (reporting the metrics or cases that look good) and *publication bias* (negative results are rarely published).
- **The causes look like those of earlier reproducibility crises:** many free choices for the researcher, and incentives that favour positive results.

**What this means in practice:**

- Always compare at equal accuracy or equal cost, against the best reasonable classical method.
- Report failures and the conditions under which the model is worse.
- Include simple baselines: a coarser mesh, a GP, nearest-neighbour lookup of the most similar training design.
- Be honest in customer conversations. Over-promising is the fastest way to lose an engineering customer's trust.

## In an interview

**Thirty-second answer to "When should you not replace a solver with ML?"** "When I only need a few answers, when the case is outside the training regime, when the result is for sign-off, or when a cheaper classical method is already good enough. ML pays off for many-query work on a well-covered design space. Even then I would keep the solver in the loop to verify the final designs and to handle anything the model flags as uncertain or out of distribution."

**Likely follow-ups**

- *Can a PINN replace FEM for a standard forward problem?* Usually not. For mature forward problems FEM or FVM is typically faster and more accurate. PINNs are more useful for inverse problems and data assimilation. See [When a PINN beats a classical solver](pinns.md#when-a-pinn-beats-a-classical-solver-and-when-it-loses).
- *The customer needs both speed and guarantees. What do you do?* A hybrid: the surrogate screens, the solver verifies the final candidates, and uncertain or out-of-distribution queries are routed to the solver automatically. See [When to fall back to the full solver](out-of-distribution-and-generalization.md#when-to-fall-back-to-the-full-solver).
- *Is ML ever more accurate than the solver?* Not than a converged solver on the same physical model. It can be more accurate than a *cheap* solver (for example a coarse mesh) at the same cost, which is the honest way to claim an advantage. It can also add physics the solver's model lacks, if it is trained on experimental data.

## References

- [Weak baselines and reporting biases lead to overoptimism in machine learning for fluid-related partial differential equations (McGreivy & Hakim, 2024)](https://arxiv.org/abs/2407.07218)
- [Enhancing Computational Fluid Dynamics with Machine Learning (Vinuesa & Brunton)](https://arxiv.org/abs/2110.02085)
- [Machine Learning for Fluid Mechanics (Brunton, Noack & Koumoutsakos)](https://arxiv.org/abs/1905.11075)
- [Machine learning accelerated computational fluid dynamics (Kochkov et al., 2021)](https://arxiv.org/abs/2102.01010)
- [ETH Zürich AISE: Introduction to Hybrid Workflows Part 1 (video)](https://www.youtube.com/watch?v=fJbt6VKYycA)
